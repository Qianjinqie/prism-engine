// path_raster.wgsl — 2D 矢量路径光栅的 WebGPU compute kernel。
//
// 与 Vulkan 侧 `src/vulkan/spirv_path.rs` 发射的 SPIR-V、CPU 侧
// `src/path_raster.rs::segment_area` **逐句同式**（同一求值树、同一
// `COORD_BIAS` 坐标编码、同一 `pos_area` / `tri_factor` 闭式 + 同一饱和短路）。
// 三端都用 f32，只用 + - * / 与比较；无超越函数、无采样、无原子加（一像素一
// invocation，确定性顺序累加）。WGSL 恒为产品 kernel：SPIR-V 那份 `cov·one`
// 输出增益（产品 `one=1.0`，仅负控注入 1.001）在此是恒等式，故直接写 `cov`。
//
// 四个 storage 绑定统一为 array<u32>：片段与输出覆盖率以 f32 **位型**传输，
// 用 bitcast 还原，避免文本往返与第二套舍入口径（与 Vulkan 侧同一份字节）。
//   b0 Params   [w, h, segments_len, rows_len]
//   b1 Segments 每 6 个 u32 一个片段 (h, 0, base, fp, fq, delta)，均为 f32 位型
//   b2 Rows     每行 [start, count]，2*h 个 u32
//   b3 Out      每像素一个 f32 位型覆盖率
//
// 数学恒等式（与 CPU 参考同源）：
//   cov(P) = Σ delta · [pos_area(u0,u1,h,fac) − pos_area(u0−1,u1−1,h,fac)]
//   u = (base − 2^20 − px) + f，f = X(y) − floor(min(xp,xq)) 的小量小数。
//   饱和短路（精确结论，挡住 f32 大数消去）：
//     bias_px+min(fp,fq) > 1 ⇒ delta·h（整列满覆盖）；
//     bias_px+max(fp,fq) < 0 ⇒ 0（整列零覆盖）；否则走一般闭式。

@group(0) @binding(0) var<storage, read> b0: array<u32>;
@group(0) @binding(1) var<storage, read> b1: array<u32>;
@group(0) @binding(2) var<storage, read> b2: array<u32>;
@group(0) @binding(3) var<storage, read_write> b3: array<u32>;

const COORD_BIAS: f32 = 1048576.0;

// ∫(u(y))⁺dy 的解析原函数：两端非负走梯形，否则走正部三角形。
// select(f, t, cond) 与 SPIR-V OpSelect(cond, t, f) 逐式对应。
fn pos_area(u0: f32, u1: f32, h: f32, fac: f32) -> f32 {
  let trap = (u0 + u1) * 0.5 * h;
  let pmax = select(u1, u0, u0 > u1);
  let pp = select(0.0, pmax, pmax > 0.0);
  let tri = fac * pp * pp;
  let pmin = select(u1, u0, u0 < u1);
  return select(tri, trap, pmin >= 0.0);
}

// 三角形模式因子 0.5*h/|u0−u1|；Δ=0（竖直边）时为 0。
// safe_den 让除零支在 select 丢弃前不产生非有限值（与 CPU 参考同结论）。
fn tri_factor(u0: f32, u1: f32, h: f32) -> f32 {
  let den = u0 - u1;
  let neg = -den;
  let abs_den = select(den, neg, den < 0.0);
  let is_zero = den == 0.0;
  let safe_den = select(abs_den, 1.0, is_zero);
  return select(0.5 * h / safe_den, 0.0, is_zero);
}

// 单片段对像素列 px 的覆盖率贡献（与 CPU 参考 / SPIR-V 逐句同式）。
// 字段 0 直接是 y 子区间高度 h（CPU 已用 f64 算好、只舍入一次），字段 1 占位恒 0。
fn segment_area(sb: u32, px: f32) -> f32 {
  let hh = bitcast<f32>(b1[sb]);
  let base = bitcast<f32>(b1[sb + 2u]);
  let fp = bitcast<f32>(b1[sb + 3u]);
  let fq = bitcast<f32>(b1[sb + 4u]);
  let delta = bitcast<f32>(b1[sb + 5u]);
  let bias_px = (base - COORD_BIAS) - px;
  let lo = select(fq, fp, fp < fq);
  let hi = select(fp, fq, fp < fq);
  let u0 = bias_px + fp;
  let u1 = bias_px + fq;
  let fac = tri_factor(u0, u1, hh);
  let a0 = pos_area(u0, u1, hh, fac);
  let a1 = pos_area(u0 - 1.0, u1 - 1.0, hh, fac);
  let normal = delta * (a0 - a1);
  let saturated = delta * hh;
  let inner = select(normal, 0.0, bias_px + hi < 0.0);
  return select(inner, saturated, bias_px + lo > 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn path_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gx = gid.x;
  let gy = gid.y;
  let w = b0[0];
  let h = b0[1];
  if (gx >= w || gy >= h) {
    return;
  }
  let r = gy * 2u;
  let start = b2[r];
  let count = b2[r + 1u];
  let px = f32(gx);
  var cov = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let sb = (start + i) * 6u;
    cov = cov + segment_area(sb, px);
  }
  b3[gy * w + gx] = bitcast<u32>(cov);
}
