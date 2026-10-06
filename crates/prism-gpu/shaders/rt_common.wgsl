// prism-rt wavefront — 公共定义（D2；与 CpuRef integrator.rs 语义对齐）
//
// 本文件由各 rt_*.wgsl 通过「拼接包含」方式前置（引擎自研 WGSL 子集汇编器
// 负责 include 展开；本阶段以结构断言校验）。零第三方依赖。
//
// # 与 CpuRef 的对应关系（判对基准 = crates/prism-rt/src/integrator.rs）
// - 维度预算：BASE_BOUNCE=5 / SLOTS_PER_BOUNCE=6（槽 0..5）——同 integrator。
// - 采样：样本流哈希（Sobol 索引 ⊕ 掩码）在 GPU 上以 pcg 整数 hash 复刻
//   （整数运算跨平台逐位一致，见 sampler.rs「格锚纪律」）。
// - 求交：球/盒解析（与 integrator sphere_t/box_t 同式）；网格走 BVH
//   （见 rt_bvh.wgsl，对应 prism-geometry/bvh.rs 遍历）。
// - BSDF/灯光/NEE/MIS：见 rt_shade.wgsl（对齐 bsdf.rs / light.rs）。

// ---- 常量（与 integrator.rs 逐值对齐）----
const PI: f32 = 3.141592653589793;
const FRAC_1_PI: f32 = 0.3183098861837907;
const T_MIN: f32 = 1e-4;          // 自交防护下界
const SURFACE_EPS: f32 = 1e-3;    // 表面偏移
const MISS_T: f32 = 1e30;         // 未命中段长

// 维度预算
const BASE_BOUNCE: u32 = 5u;
const SLOTS_PER_BOUNCE: u32 = 6u;
const SLOT_BSDF: u32 = 0u;
const SLOT_LIGHT: u32 = 2u;
const SLOT_RR: u32 = 4u;
const SLOT_AUX: u32 = 5u;

// 路径状态机状态（wavefront 队列分组键）
const ST_ACTIVE: u32 = 0u;   // 待着色/扩展
const ST_TERMINATED: u32 = 1u;

fn bounce_dim(ev: u32, slot: u32) -> u32 {
  return BASE_BOUNCE + (ev - 1u) * SLOTS_PER_BOUNCE + slot;
}

// ---- Vec3 基础件（与 prism-math Vec3 同运算，仅 ±×÷√）----
fn v3(x: f32, y: f32, z: f32) -> vec3<f32> { return vec3<f32>(x, y, z); }
fn vdot(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x*b.x + a.y*b.y + a.z*b.z; }
fn hadamard(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> { return a * b; }
fn luminance(c: vec3<f32>) -> f32 { return 0.2126*c.x + 0.7152*c.y + 0.0722*c.z; }
fn clampf(x: f32, lo: f32, hi: f32) -> f32 { return min(max(x, lo), hi); }

// ---- 确定性三角件（D-29 禁 libm：Maclaurin 区间逼近 + 固定系数）----
// 与 sampler.rs cos_sin/tan 同构造：mod 2π 归约 → 象限对称 → [0, π/4] 级数。
fn quarter_sin(x: f32) -> f32 {
  let x2 = x * x;
  return x * (1.0 + x2 * (-1.0/6.0 + x2 * (1.0/120.0 + x2 * (-1.0/5040.0 + x2 * (1.0/362880.0)))));
}
fn quarter_cos(x: f32) -> f32 {
  let x2 = x * x;
  return 1.0 + x2 * (-0.5 + x2 * (1.0/24.0 + x2 * (-1.0/720.0 + x2 * (1.0/40320.0))));
}
fn cos_sin(x: f32) -> vec2<f32> {
  const TWO_PI: f32 = 6.283185307179586;
  const FRAC_PI_2: f32 = 1.5707963267948966;
  const FRAC_PI_4: f32 = 0.7853981633974483;
  var r = x - TWO_PI * floor(x / TWO_PI);
  let q = u32(i32(r * (2.0 / PI))) & 3u;
  r = r - f32(q) * FRAC_PI_2;
  var c = 0.0;
  var s = 0.0;
  if (r > FRAC_PI_4) {
    let rr = FRAC_PI_2 - r;
    c = quarter_sin(rr);
    s = quarter_cos(rr);
  } else {
    c = quarter_cos(r);
    s = quarter_sin(r);
  }
  if (q == 0u) { return vec2<f32>(c, s); }
  if (q == 1u) { return vec2<f32>(-s, c); }
  if (q == 2u) { return vec2<f32>(-c, -s); }
  return vec2<f32>(s, -c);
}
fn det_tan(x: f32) -> f32 { let cs = cos_sin(x); return cs.y / cs.x; }

// ---- onb（叉积法，格锚纪律：禁浮点三角）----
fn onb(n: vec3<f32>) -> mat3x3<f32> {
  var up = vec3<f32>(1.0, 0.0, 0.0);
  if (abs(n.x) > 0.9) { up = vec3<f32>(0.0, 1.0, 0.0); }
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// ---- concentric disk（Shirley–Chiu；与 sampler.rs 同式）----
fn concentric_disk(u: f32, v: f32) -> vec2<f32> {
  const FRAC_PI_4: f32 = 0.7853981633974483;
  const FRAC_PI_2: f32 = 1.5707963267948966;
  let ox = 2.0*u - 1.0;
  let oy = 2.0*v - 1.0;
  if (ox == 0.0 && oy == 0.0) { return vec2<f32>(0.0, 0.0); }
  var r = ox;
  var theta = FRAC_PI_4 * (oy / ox);
  if (abs(ox) <= abs(oy)) {
    r = oy;
    theta = FRAC_PI_2 - FRAC_PI_4 * (ox / oy);
  }
  let cs = cos_sin(theta);
  return vec2<f32>(r * cs.x, r * cs.y);
}

// ---- 余弦半球（Malley 法；与 sampler.rs 同式）----
fn cosine_hemisphere(u: f32, v: f32) -> vec3<f32> {
  let d = concentric_disk(u, v);
  let z = sqrt(max(0.0, 1.0 - d.x*d.x - d.y*d.y));
  return vec3<f32>(d.x, d.y, z);
}

// ---- cone_dir（与 sampler.rs 同式）----
fn cone_dir(w: vec3<f32>, cos_theta_max: f32, u: f32, v: f32) -> vec3<f32> {
  let cos_t = 1.0 - u * (1.0 - cos_theta_max);
  let sin_t = sqrt(max(0.0, 1.0 - cos_t*cos_t));
  let cs = cos_sin(2.0 * PI * v);
  let m = onb(w);
  return w * cos_t + m[0] * (sin_t * cs.x) + m[1] * (sin_t * cs.y);
}
