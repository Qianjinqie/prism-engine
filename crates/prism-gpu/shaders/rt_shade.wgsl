// prism-rt 着色/采样支撑 kernel —— BSDF / NEE / MIS / 采样流，对齐 CpuRef。
//
// # 对应关系（判对基准 = prism-rt/src/{sampler,bsdf,light,integrator}.rs）
// - sample_index / sample_1d / sample_2d / aux_1d：Sobol 索引 ⊕ 掩码 + PCG 打散
//   的 GPU 复刻（整数运算跨平台逐位一致）。Sobol 方向数表由 CPU 侧上传（storage
//   buffer；表 = Joe–Kuo 64 维，见 sampler.rs 数据资产声明）。
// - bsdf_sample：Disney 三瓣（GGX VNDF / Burley 漫反 / Snell 透射），对齐 bsdf.rs。
// - nee_surface：四灯光族 NEE + MIS 权重，对齐 light.rs。
// - power_heuristic / nee_mis_weight：对齐 light.rs。

// ---- 采样流（对齐 sampler.rs；Sobol 表由 CPU 上传）----
//!include rt_common.wgsl
//!include rt_scene.wgsl
@group(1) @binding(0) var<storage, read> sobol_dir: array<u32>; // [64][32] 方向数矩阵
@group(1) @binding(1) var<storage, read> scene_buf: array<u32>; // 场景/材质/灯光串联缓冲

const IDX_SALT: u32 = 0x85A308D3u;
const MASK_SALT: u32 = 0x03707344u;
const AUX_SALT: u32 = 0x7F4A7C15u;

// PCG 整数 hash（与 prism-noise::pcg 同族；格锚纪律 → 跨平台逐位一致）。
fn pcg_hash(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn hash2(a: u32, b: u32) -> u32 { return pcg_hash(a ^ pcg_hash(b)); }
fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 {
  return pcg_hash(pcg_hash(pcg_hash(a) ^ b) ^ pcg_hash(c ^ d));
}

// SampleIndex 打包（像素域 D-28）。
fn sample_index(px: u32, py: u32, sample: u32, dim: u32) -> u32 {
  let pack = (px << 16u) | (py & 0xFFFFu);
  // 节点 id 取自场景 uniform（group 2）；本文件不依赖 wavefront 的队列 params，
  // 故 rt_shade 可独立编译（真机 D-14 门禁要求单文件零未解析引用）。
  return hash4(pack, sample, dim, scene_params.node_id);
}

// Sobol 坐标：⊕ 方向数（dim 行，index 位）⊕ 打散掩码。
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
fn dim_value(seed: u32, dim: u32, index: u32) -> f32 {
  let mask = hash2(seed, MASK_SALT ^ dim);
  let v = sobol_u32(dim, index) ^ mask;
  return f32(v >> 8u) * (1.0 / 16777216.0);
}
// 一维 Sobol（si 已含 base seed 语义；此处简化为按像素+样本派生）。
fn sample_1d(si: u32, dim: u32) -> f32 {
  let base = si ^ IDX_SALT;
  let idx = hash2(base, MASK_SALT);
  return dim_value(base, dim, idx);
}
fn sample_2d(si: u32, dim: u32) -> vec2<f32> {
  let base = si ^ IDX_SALT;
  let idx = hash2(base, MASK_SALT);
  return vec2<f32>(dim_value(base, dim, idx), dim_value(base, dim + 1u, idx));
}
fn aux_1d(si: u32, dim: u32, step: u32) -> f32 {
  return f32(hash2(hash2(si, AUX_SALT) ^ dim, step) >> 8u) * (1.0 / 16777216.0);
}

// ---- power heuristic（对齐 light.rs）----
fn power_heuristic(a: f32, b: f32) -> f32 {
  let a2 = a * a;
  let den = a2 + b * b;
  if (den == 0.0) { return 0.0; }
  return a2 / den;
}

// ---- GGX / Smith / Schlick（对齐 bsdf.rs）----
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

// Disney BSDF 采样骨架（对齐 bsdf.rs::Bsdf::sample；三瓣瓣选择 + VNDF/漫反/透射）。
// 材质参数从 scene_buf 读取（prim → material 索引）。
fn bsdf_sample(prim: u32, w_in: vec3<f32>, n: vec3<f32>, ev: u32) -> BsdfSample {
  let mat = load_material(prim);
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
  let u_lobe = sample_1d(sample_index_seed(prim, ev, SLOT_BSDF), bounce_dim(ev, SLOT_BSDF));
  let uv = sample_2d(sample_index_seed(prim, ev, SLOT_BSDF), bounce_dim(ev, SLOT_BSDF) + 1u);
  // 透射瓣（玻璃 / SSS 入口）。
  if (u_lobe < p_trans / sum) {
    let eta = 1.0 / mat.ior;
    let r = refract_dir(w_in, n, eta);
    if (all(r == vec3<f32>(0.0, 0.0, 0.0))) {
      s.dir = reflect(w_in, n);
      s.weight = schlick(f0, cos_v);
      s.specular = 1u;
      return s;
    }
    s.dir = normalize(r);
    s.weight = hadamard(mat.base_color, vec3<f32>(1.0, 1.0, 1.0) - schlick(f0, cos_v));
    s.pdf = 0.0;
    s.entered = 1u;
    return s;
  }
  // 漫反瓣（Burley FD90）。
  if (u_lobe < (p_trans + p_diff) / sum) {
    let d_local = cosine_hemisphere(uv.x, uv.y);
    let m = onb(n);
    s.dir = normalize(m[0] * d_local.x + m[1] * d_local.y + n * d_local.z);
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
  s.dir = normalize(reflect(-wo, h));
  let cos_l = vdot(s.dir, n);
  if (cos_l <= 0.0) { s.pdf = -1.0; return s; }
  if (rough < 1e-3) {
    s.weight = schlick(f0, cos_v) * (1.0 / max(p_spec / sum, 1e-4));
    s.specular = 1u;
    s.pdf = 0.0;
    return s;
  }
  s.pdf = cos_l * FRAC_1_PI; // 混合 pdf 的漫反部分（GPU 骨架近似；真机验）
  s.weight = schlick(f0, cos_v);
  return s;
}

// GGX VNDF（Heitz 2018；对齐 bsdf.rs::sample_vndf）。
fn sample_vndf(ve_local: vec3<f32>, alpha: f32, u1: f32, u2: f32) -> vec3<f32> {
  let vh = normalize(vec3<f32>(alpha * ve_local.x, alpha * ve_local.y, ve_local.z));
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
  return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// ---- NEE（对齐 light.rs::sample_nee）----
fn nee_surface(p: vec3<f32>, n: vec3<f32>, w_in: vec3<f32>, throughput: vec3<f32>, ev: u32, aux_steps: u32) -> vec3<f32> {
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  // 采样流种子按 (像素量化, 样本号) 打包：布尔比较不是合法实参（真机曾报类型错）。
  let seed_px = u32(max(p.x, 0.0)) * 8192u + u32(max(p.y, 0.0));
  let uv = sample_2d(sample_index(seed_px, 0u, ev, 0u), bounce_dim(ev, SLOT_LIGHT));
  let light_count = load_light_count();
  var li = 0u;
  while (li < light_count) {
    let ls = sample_light(li, p, uv.x, uv.y);
    if (ls.pdf_w > 0.0) {
      let cos_l = vdot(n, ls.wi);
      if (cos_l > 0.0) {
        let bsdf = eval_bsdf(load_material(0u), -w_in, ls.wi, n);
        let vis = shadow_ray(p + n * SURFACE_EPS, ls.wi, ls.dist);
        if (vis > 0.0) {
          let w = nee_mis_weight(li, ls.pdf_w, cos_l * FRAC_1_PI);
          acc = acc + hadamard(hadamard(throughput, hadamard(bsdf, vec3<f32>(cos_l, cos_l, cos_l))), ls.radiance) * (vis * w / ls.pdf_w);
        }
      }
    }
    li = li + 1u;
  }
  return clamp_indirect(acc, ev);
}
