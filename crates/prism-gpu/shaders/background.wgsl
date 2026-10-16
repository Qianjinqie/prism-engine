// background.wgsl — GPU 背景渲染 shader（prism-engine 渲染热点兜底）。
//
// 把 demo 的逐像素 CPU 嵌套循环（`paint_background`：垂直渐变 + 二次衰减辉光
// + 极低频噪声 + 暗角 + 呼吸）迁移为 WGSL compute kernel。每个 invocation 负责
// 一个像素，按 (x, y, t, seed) 计算背景色。
//
// 设计约束（与 prism-gpu 既有门禁一致）：
// - D-29：禁用 libm 超越函数（sin/cos/exp/pow/log/atan…）。呼吸项改用
//   手写 Maclaurin 正弦逼近 `sin_approx`，全程仅用 +/-/*// 与 floor/clamp 等内建。
// - 噪声在 shader 内用 hash-based value noise 实现（振幅 ±3%），不再依赖第三方
//   simplex 库；Rust 侧 `render_background_cpu` 复用同一套 hash 公式，保证两端
//   公式一致。GPU 实际执行待 WebGPU 运行时——本沙箱 headless，仅做编译/解析校验。
//
// 参数经 uniform 传入：dim=(width,height)、t、seed。
// 输出写入 storage buffer（每像素一个 u32，RGBA8 little-endian 打包）。

struct Params {
  dim: vec2<u32>,
  t: f32,
  seed: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out_rgba: array<u32>;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

// 确定性 u32 哈希（WGSL u32 运算，与 CPU 参考位级一致）。
fn hash_u32(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

// 整数格点 + seed → [0,1) 伪随机（与 CPU 参考 `hash2i` 同公式）。
fn hash2i(ix: u32, iy: u32, seed: u32) -> f32 {
  var h: u32 = ix * 0x9e3779b1u + iy * 0x85ebca77u + seed * 0xc2b2ae3du;
  h = hash_u32(h);
  return f32(h) * (1.0 / 4294967296.0);
}

// 2D value noise：格点 hash + 双线性插值（smoothstep 权重），返回 [0,1)。
fn value_noise(px: f32, py: f32, seed: u32) -> f32 {
  let ix = u32(floor(px));
  let iy = u32(floor(py));
  let fx = px - floor(px);
  let fy = py - floor(py);
  // smoothstep 权重，手写以避免依赖内建。
  let ux = fx * fx * (3.0 - 2.0 * fx);
  let uy = fy * fy * (3.0 - 2.0 * fy);
  let x0 = ix;
  let y0 = iy;
  let x1 = ix + 1u;
  let y1 = iy + 1u;
  let v00 = hash2i(x0, y0, seed);
  let v10 = hash2i(x1, y0, seed);
  let v01 = hash2i(x0, y1, seed);
  let v11 = hash2i(x1, y1, seed);
  let a = v00 + (v10 - v00) * ux;
  let b = v01 + (v11 - v01) * ux;
  return a + (b - a) * uy;
}

// demo `smooth`：clamp 到 [0,1] 再 t*t*(3-2t)（手写 smoothstep）。
fn smooth01(e: f32) -> f32 {
  let t = clamp(e, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// 手写正弦逼近（Maclaurin 5 阶；仅 +/-//floor，无 libm）。用于呼吸项。
fn sin_approx(x: f32) -> f32 {
  var a = x - TAU * floor(x / TAU);   // [0, TAU)
  if (a > PI) { a = a - TAU; }        // 折到 [-PI, PI]
  if (a < -PI) { a = a + TAU; }
  let xx = a * a;
  let a3 = a * xx;
  let a5 = a3 * xx;
  let a7 = a5 * xx;
  let a9 = a7 * xx;
  return a - a3 / 6.0 + a5 / 120.0 - a7 / 5040.0 + a9 / 362880.0;
}

// 按像素坐标计算背景色（RGB，[0,1]）。与 CPU 参考 `render_background_cpu` 同公式。
fn background_at(px: u32, py: u32, w: u32, h: u32, t: f32, seed: u32) -> vec3<f32> {
  let vx = f32(px) / max(f32(w) - 1.0, 1.0);
  let vy = f32(py) / max(f32(h) - 1.0, 1.0);

  // 1) 垂直渐变（lerp top/bottom，smoothstep）。
  let top = vec3<f32>(0.035, 0.055, 0.130);
  let bottom = vec3<f32>(0.058, 0.090, 0.205);
  let g = smooth01(vy);
  var col = top + (bottom - top) * g;

  // 2) 冷青辉光（二次衰减）。
  let gx = vx - 0.5;
  let gy = vy - 0.40;
  let d2 = gx * gx + gy * gy;
  var glow = 1.0 - d2 * 3.0;
  if (glow < 0.0) { glow = 0.0; }
  glow = glow * glow;
  col = col + vec3<f32>(0.030, 0.070, 0.120) * glow;

  // 3) 极低频噪声（hash value noise，振幅 ±3%，随时间极慢漂移）。
  let nx = vx * 2.2 + t * 0.05;
  let ny = vy * 1.3 - t * 0.035;
  let n = value_noise(nx, ny, seed) * 2.0 - 1.0;   // [0,1) -> [-1,1)
  let modulation = 1.0 + 0.03 * n;

  // 4) 暗角。
  let cx = vx - 0.5;
  let cy = vy - 0.5;
  let rad = sqrt(cx * cx + cy * cy);
  let vig = 1.0 - smooth01((rad - 0.45) / 0.55) * 0.55;

  // 5) 呼吸（正弦逼近，±1.5%）。
  let breath = 1.0 + 0.015 * sin_approx(t * (TAU / 5.0));

  // 合成。
  let k = modulation * breath * vig;
  return col * k;
}

// RGBA8 little-endian 打包：内存字节序 [r, g, b, a]。
fn pack_rgba(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.r, 0.0, 1.0) * 255.0);
  let g = u32(clamp(c.g, 0.0, 1.0) * 255.0);
  let b = u32(clamp(c.b, 0.0, 1.0) * 255.0);
  let a = 255u;
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8, 1)
fn bg_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params.dim.x;
  let h = params.dim.y;
  if (gid.x >= w || gid.y >= h) { return; }
  let idx = gid.y * w + gid.x;
  let col = background_at(gid.x, gid.y, w, h, params.t, params.seed);
  out_rgba[idx] = pack_rgba(col);
}
