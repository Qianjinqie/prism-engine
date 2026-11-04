// prism-rt 着色/采样支撑 kernel —— 采样流 / BSDF / MIS，判对基准 = CpuRef。
//
// # 对应关系（判对基准 = prism-rt/src/{sampler,bsdf,light}.rs）
// - `WfStream` / `wf_stream` / `wf_sample_1d` / `wf_sample_2d`：
//   `SampleStream` 的逐位复刻。基种子 = `pcg::hash4(种子, pack(像素), 帧, 节点, 0)`、
//   Sobol 索引 = `样本 ⊕ hash1(base, IDX_SALT)` 低 32 位、打散掩码 =
//   `hash2_finish(hash2_prefix(base, MASK_SALT), dim)` 低 32 位、坐标 =
//   `Sobol(i, dim) ⊕ mask` 取高 24 位 × 2^-24。64 位整数哈希由 `rt_u64.wgsl`
//   的 (lo,hi) u32 对仿真**逐位**完成（不是近似）。
// - Sobol 方向数表：**真 Joe–Kuo 表**由宿主上传（`WavefrontRequest::
//   sobol_table`；测试/浏览器 harness 用 `prism_rt::sampler::sobol_u32(d,1<<k)`
//   生成 64×32 u32）。CPU 侧同表（sampler.rs 的数据资产）。
// - `power_heuristic`：对齐 light.rs。
// - `bsdf_sample`：Disney 三瓣（GGX VNDF / Burley 漫反 / Snell 透射），
//   对齐 bsdf.rs::sample 的**方向提案**语义。它只服务 `MAT_PBR`（R-1 判据
//   范围是 Lambert 场景类；Pbr 续弹不进逐位判据）。

// ---- 采样流（对齐 sampler.rs；Sobol 表由 CPU 上传）----
//!include rt_common.wgsl
//!include rt_u64.wgsl
//!include rt_scene.wgsl
@group(1) @binding(0) var<storage, read> sobol_dir: array<u32>; // [64][32] 方向数矩阵
@group(1) @binding(1) var<storage, read> scene_buf: array<u32>; // 场景/材质/灯光串联缓冲

// ---- power heuristic（对齐 light.rs）----
fn power_heuristic(a: f32, b: f32) -> f32 {
  let a2 = a * a;
  let den = a2 + b * b;
  if (den == 0.0) { return 0.0; }
  return a2 / den;
}

// Sobol 坐标：⊕ 方向数（dim 行，index 置位位）。与 sampler.rs::sobol_u32
// 逐位相同（GF(2) 异或交换/结合 ⇒ 迭代顺序无关）。表 = 宿主上传的真
// Joe–Kuo 方向数（`sobol_dir`，[64][32] u32 行主序）。
fn sobol_u32(dim: u32, index: u32) -> u32 {
  var v = 0u;
  var bits = index;
  var k = 0u;
  let row = dim * 32u;
  loop {
    if (bits == 0u) { break; }
    if ((bits & 1u) == 1u) { v = v ^ sobol_dir[row + k]; }
    bits = bits >> 1u;
    k = k + 1u;
  }
  return v;
}

// ---------------------------------------------------------------------------
// 采样流（`SampleStream` 的逐位复刻；判对基准 = sampler.rs）
// ---------------------------------------------------------------------------

/// 单像素单样本的采样流句柄：`(base, index, prefix)` 三件缓存与 CPU 的
/// `SampleStream` 字段一一对应（base/prefix 是像素级常量，index 是样本级常量）。
struct WfStream {
  base_lo: u32,
  base_hi: u32,
  index: u32,
  prefix_lo: u32,
  prefix_hi: u32,
}

/// 构造流（`SampleStream::new` 逐式）：
/// * base = `hash4(种子, pack(像素), 帧, 节点, 0)`，pack = (px<<32)|py
///   ⇒ (lo, hi) = (py, px)；
/// * index = `样本 ⊕ hash1(base, IDX_SALT)`（取低 32 位，CPU `as u32` 同截断）；
/// * prefix = `hash2_prefix(base, MASK_SALT)`。
fn wf_stream(seed: u32, frame: u32, node_id: u32, px: u32, py: u32, sample: u32) -> WfStream {
  let base = pcg_hash4(seed, 0u, py, px, frame, 0u, node_id, 0u, 0u, 0u);
  let idx_h = pcg_hash1(base.x, base.y, IDX_SALT_LO, IDX_SALT_HI);
  let prefix = pcg_hash2_prefix(base.x, base.y, MASK_SALT_LO, MASK_SALT_HI);
  var s: WfStream;
  s.base_lo = base.x;
  s.base_hi = base.y;
  s.index = sample ^ idx_h.x;
  s.prefix_lo = prefix.x;
  s.prefix_hi = prefix.y;
  return s;
}

/// 单维坐标值（`dim_value_p` 逐式）：`Sobol(index, dim) ⊕ 打散掩码`，
/// 取高 24 位 × 2^-24。dim ≥ 64 走 PCG 纯函数流兜底（`hash2_f32(base, dim, index)`）。
fn wf_dim_value(st: WfStream, dim: u32) -> f32 {
  if (dim >= 64u) {
    // 超表兜底：`pcg::hash2_f32(base, dim, index)` =
    // unit(hash2_finish(hash2_prefix(base, dim), index))。
    let p = pcg_hash1(st.base_lo, st.base_hi, dim, 0u);
    let h = pcg_hash2_finish(p.x, p.y, st.index, 0u);
    return pcg_unit_f32(h.x, h.y);
  }
  let mask = pcg_hash2_finish(st.prefix_lo, st.prefix_hi, dim, 0u).x;
  let v = sobol_u32(dim, st.index) ^ mask;
  return f32(v >> 8u) * (1.0 / 16777216.0);
}

/// 一维 Sobol 抽取（`SampleStream::sample_1d` 逐位）。
fn wf_sample_1d(st: WfStream, dim: u32) -> f32 {
  return wf_dim_value(st, dim);
}

/// 二维 Sobol 抽取（维度 d 与 d+1 共享同一 index；`sample_2d` 逐位）。
fn wf_sample_2d(st: WfStream, dim: u32) -> vec2<f32> {
  return vec2<f32>(wf_dim_value(st, dim), wf_dim_value(st, dim + 1u));
}

// ---- GGX / Smith / Schlick（对齐 bsdf.rs；Pbr 提案瓣专用）----
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }
fn schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
  let m5 = pow5(max(0.0, 1.0 - cos_theta));
  return f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * m5;
}
fn ggx_d(alpha: f32, cos_theta_m: f32) -> f32 {
  let c2 = cos_theta_m * cos_theta_m;
  let den = c2 * (alpha * alpha - 1.0) + 1.0;
  return alpha * alpha / (PI * den * den);
}
fn smith_g1(alpha: f32, cos_theta: f32) -> f32 {
  let c2 = cos_theta * cos_theta;
  let den = cos_theta + sqrt(alpha * alpha + (1.0 - alpha * alpha) * c2);
  return 2.0 * cos_theta / den;
}
fn refract_dir(w_in: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
  let cos_i = -vdot(w_in, n);
  let sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i);
  if (sin2_t >= 1.0) { return vec3<f32>(0.0, 0.0, 0.0); } // TIR 哨兵
  let cos_t = sqrt(1.0 - sin2_t);
  return w_in * eta + n * (eta * cos_i - cos_t);
}

// ---- BSDF 采样结果 ----
struct BsdfSample {
  dir: vec3<f32>,
  weight: vec3<f32>,
  pdf: f32,       // < 0 表示吸收/无有效瓣
  specular: u32,
  entered: u32,
}

// Disney BSDF 采样（对齐 bsdf.rs::Bsdf::sample 的方向提案；**只服务 MAT_PBR**）。
// 采样维度与 CPU 同槽：u_lobe ← SLOT_BSDF、(u1,u2) ← SLOT_BSDF+1（wf 流）。
fn bsdf_sample(mat: Material, w_in: vec3<f32>, n: vec3<f32>, ev: u32, st: WfStream) -> BsdfSample {
  let wo = -w_in;
  let cos_v = max(0.0, vdot(wo, n));
  let f0 = mat.f0;
  let p_spec = luminance(schlick(f0, cos_v)) + 1e-4;
  let p_diff = luminance(mat.base_color) * (1.0 - mat.metalness) * (1.0 - mat.transmission);
  let p_trans = mat.transmission * (1.0 - mat.metalness);
  let sum = p_spec + p_diff + p_trans;
  var s: BsdfSample;
  s.dir = vec3<f32>(0.0, 0.0, 1.0);
  s.weight = vec3<f32>(0.0, 0.0, 0.0);
  s.pdf = -1.0;
  s.specular = 0u;
  s.entered = 0u;
  if (sum <= 0.0) { return s; }
  let u_lobe = wf_sample_1d(st, bounce_dim(ev, SLOT_BSDF));
  let uv = wf_sample_2d(st, bounce_dim(ev, SLOT_BSDF) + 1u);
  // 透射瓣（玻璃 / SSS 入口）。
  if (u_lobe < p_trans / sum) {
    let eta = 1.0 / mat.ior;
    let r = refract_dir(w_in, n, eta);
    if (all(r == vec3<f32>(0.0, 0.0, 0.0))) {
      s.dir = vnorm(reflect(w_in, n));
      s.weight = schlick(f0, cos_v);
      s.specular = 1u;
      return s;
    }
    s.dir = vnorm(r);
    s.weight = hadamard(mat.base_color, vec3<f32>(1.0, 1.0, 1.0) - schlick(f0, cos_v));
    s.pdf = 0.0;
    s.entered = 1u;
    return s;
  }
  // 漫反瓣（Burley FD90）。
  if (u_lobe < (p_trans + p_diff) / sum) {
    let d_local = cosine_hemisphere(uv.x, uv.y);
    let m = onb(n);
    s.dir = vnorm(m[0] * d_local.x + m[1] * d_local.y + n * d_local.z);
    let cos_i = max(0.0, vdot(s.dir, n));
    s.pdf = cos_i * FRAC_1_PI;
    s.weight = hadamard(mat.base_color, vec3<f32>(1.0, 1.0, 1.0) - schlick(f0, cos_v));
    return s;
  }
  // 镜面瓣（GGX VNDF）。
  let rough = max(mat.roughness, 1e-3);
  let alpha = rough * rough;
  let m = onb(n);
  let wo_local = vec3<f32>(vdot(wo, m[0]), vdot(wo, m[1]), cos_v);
  let h_local = sample_vndf(wo_local, alpha, uv.x, uv.y);
  let h = m[0] * h_local.x + m[1] * h_local.y + n * h_local.z;
  s.dir = vnorm(reflect(-wo, h));
  let cos_l = vdot(s.dir, n);
  if (cos_l <= 0.0) { s.pdf = -1.0; return s; }
  if (rough < 1e-3) {
    s.weight = schlick(f0, cos_v) * (1.0 / max(p_spec / sum, 1e-4));
    s.specular = 1u;
    s.pdf = 0.0;
    return s;
  }
  s.pdf = cos_l * FRAC_1_PI; // 混合 pdf 的漫反部分（GPU 近似；不进 R-1 判据）
  s.weight = schlick(f0, cos_v);
  return s;
}

// GGX VNDF（Heitz 2018；对齐 bsdf.rs::sample_vndf）。
fn sample_vndf(ve_local: vec3<f32>, alpha: f32, u1: f32, u2: f32) -> vec3<f32> {
  let vh = vnorm(vec3<f32>(alpha * ve_local.x, alpha * ve_local.y, ve_local.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3<f32>(1.0, 0.0, 0.0);
  if (lensq > 0.0) { t1 = vec3<f32>(-vh.y, vh.x, 0.0) * (1.0 / sqrt(lensq)); }
  let t2 = cross(vh, t1);
  let r = sqrt(u1);
  let cs = cos_sin(2.0 * PI * u2);
  let t1p = r * cs.x;
  var t2p = r * cs.y;
  let sc = 0.5 * (1.0 + vh.z);
  t2p = (1.0 - sc) * sqrt(max(0.0, 1.0 - t1p * t1p)) + sc * t2p;
  let nz = sqrt(max(0.0, 1.0 - t1p * t1p - t2p * t2p));
  let nh = t1 * t1p + t2 * t2p + vh * nz;
  return vnorm(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}
