// prism-sim 公共定义 —— 百万粒子 compute 的共享件。
//
// # 对应关系（判对基准 = prism-physics/{sph,cloth}.rs + prism-particles/*）
// - 空间哈希：cell = 核半径 h；哈希键 = floor(p/h)（对齐 sph.rs::SpatialHash）。
// - 排序接入：粒子按 cell 键 radix sort（D0 的 sort.rs），使邻居查询连续。
// - 核函数 2D Poly6 / Spiky（对齐 sph.rs；3D 常数见 GPU-only 迁移位移位）。
// - XPBD 距离约束投影（对齐 xpbd.rs / cloth.rs）。

const PI: f32 = 3.141592653589793;

// ---- 空间哈希：3D 格坐标（百万粒子 3D；2D 由 z=0 退化）----
struct Particle {
  pos: vec3<f32>,
  mass_inv: f32,
  vel: vec3<f32>,
  _pad: f32,
  cell: u32,   // radix sort 键（格坐标线性化）
  _pad2: u32,
}

// cell 键：floor(p/h) 线性化到 u32（对齐 sph.rs::floor_div；偏移保证非负）。
fn cell_floor(v: f32, h: f32) -> i32 { return i32(floor(v / h)); }
fn cell_key(p: vec3<f32>, h: f32) -> u32 {
  let c = vec3<i32>(cell_floor(p.x, h), cell_floor(p.y, h), cell_floor(p.z, h));
  // 每轴偏移 1024（覆盖 [-1024,1024) 格），打包 21 位/轴。
  let bx = u32(c.x + 1024);
  let by = u32(c.y + 1024);
  let bz = u32(c.z + 1024);
  return (bx & 0x7FFu) | ((by & 0x7FFu) << 11u) | ((bz & 0x7FFu) << 22u);
}

// ---- 2D/3D 核（对齐 sph.rs；只用 ±×÷√，禁 libm）----
// 2D Poly6（GPU-only 3D 常数：315/(64πh⁹)——本批实装 2D）。
fn poly6_2d(r: f32, h: f32) -> f32 {
  if (r >= h) { return 0.0; }
  let h2 = h * h;
  let diff = h2 - r * r;
  return 4.0 / (PI * h*h*h*h*h*h*h*h) * diff * diff * diff;
}
// 2D Spiky 梯度标量系数（乘 (x_i − x_j) 得 ∇W）。
fn spiky_grad_scale(r: f32, h: f32) -> f32 {
  if (r < 1e-6 || r > h) { return 0.0; }
  let hmr = h - r;
  return -30.0 / (PI * h*h*h*h*h) * hmr * hmr;
}
// 3D Poly6（迁移位；GPU-only 标注：本批不实装，常数公开数学常量）。
const POLY6_3D_COEF: f32 = 315.0 / (64.0 * 3.141592653589793);
fn poly6_3d(r: f32, h: f32) -> f32 {
  if (r >= h) { return 0.0; }
  let h2 = h * h;
  let diff = h2 - r * r;
  return POLY6_3D_COEF / (h*h*h*h*h*h*h*h*h) * diff * diff * diff;
}

// ---- XPBD 距离约束投影（对齐 xpbd.rs / cloth.rs）----
// Δλ = (−C − α̃λ)/(Σ w|∇C|² + α̃)，Δx_i = w_i ∇_i C Δλ；α̃ = compliance/h²。
fn xpbd_distance_project(pa: ptr<function, vec3<f32>>, pb: ptr<function, vec3<f32>>,
                         wa: f32, wb: f32, rest: f32, compliance: f32, h_sub: f32, lambda: ptr<function, f32>) -> f32 {
  let d = *pb - *pa;
  let len = length(d);
  let c = len - rest;
  let grad = select(vec3<f32>(1.0, 0.0, 0.0), d / max(len, 1e-9), len > 1e-9);
  let at = compliance / (h_sub * h_sub);
  var denom = wa * vdot(grad, grad) + wb * vdot(grad, grad) + at;
  if (denom <= 0.0) { return abs(c); }
  let dl = (-c - at * (*lambda)) / denom;
  *lambda = *lambda + dl;
  *pa = *pa - grad * (wa * dl);
  *pb = *pb + grad * (wb * dl);
  return abs(c);
}

fn vdot(a: vec3<f32>, b: vec3<f32>) -> f32 { return a.x*b.x + a.y*b.y + a.z*b.z; }
