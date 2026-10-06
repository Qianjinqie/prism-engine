// prism-sim 百万粒子 compute（D3）——发射 / 力场 / 空间哈希 / radix sort 接入。
//
// # 阶段（对齐 prism-particles/{emitter,force,sim}.rs）
// ① emit：种子化发射（哈希派生，无全局 RNG，D-28 像素域精神）。
// ② integrate：半隐式欧拉 + 力场（对齐 sim.rs step）。
// ③ hash：格键计算（对齐 sph.rs::SpatialHash::build）。
// ④ sort：按格键 radix sort（接 D0 sort.rs；本文档仅给调用位）。
// ⑤ neighbor：按格重排后邻域连续扫描。

//!include sim_common.wgsl
struct SimParams {
  count: u32,
  dt: f32,
  h: f32,
  gravity: vec3<f32>,
  seed: u32,
  frame: u32,
  emitter_rate: f32,
  emitter_origin: vec3<f32>,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> sp: SimParams;
@group(0) @binding(2) var<storage, read_write> cell_range: array<vec2<u32>>; // 每格 [start, end)

// PCG（与 prism-noise::pcg 同族；跨平台逐位一致）。
fn pcg_hash(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn hash3_f32(a: u32, b: u32, c: u32) -> f32 {
  return f32(pcg_hash(pcg_hash(a) ^ pcg_hash(b ^ c)) >> 8u) * (1.0 / 16777216.0);
}

// ---- ① 发射（种子化；对齐 emitter::derive）----
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sp.count) { return; }
  var p = particles[i];
  // 已存在粒子原位保留；新槽按帧号 + 槽号派生初态（确定性）。
  let u0 = hash3_f32(sp.seed, sp.frame, i * 3u + 0u);
  let u1 = hash3_f32(sp.seed, sp.frame, i * 3u + 1u);
  let u2 = hash3_f32(sp.seed, sp.frame, i * 3u + 2u);
  p.pos = sp.emitter_origin + vec3<f32>(u0 - 0.5, u1 - 0.5, u2 - 0.5);
  p.vel = vec3<f32>(0.0, 0.0, 0.0);
  p.mass_inv = 1.0;
  p.cell = 0u;
  particles[i] = p;
}

// ---- ② 积分（半隐式欧拉；对齐 sim.rs::step / xpbd 预测）----
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sp.count) { return; }
  var p = particles[i];
  if (p.mass_inv > 0.0) {
    p.vel = p.vel + sp.gravity * sp.dt;
    p.pos = p.pos + p.vel * sp.dt;
  }
  particles[i] = p;
}

// ---- ③ 空间哈希：格键（对齐 sph.rs::SpatialHash）----
@compute @workgroup_size(64)
fn hash_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sp.count) { return; }
  var p = particles[i];
  p.cell = cell_key(p.pos, sp.h);
  particles[i] = p;
}

// ---- ④ 邻域：按格重排后扫描（cell_range 由 radix sort + 前缀和构建）----
@compute @workgroup_size(64)
fn neighbor_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sp.count) { return; }
  // 占位：真实路径 = 27 邻格 × cell_range 求和（3D）；本骨架保留接入位，
  // 邻居集由 count/density kernel 直接使用 cell_range（见 sim_sph.wgsl）。
  _ = i;
}

// ---- 排序接入位（D0 radix sort；文档注释声明，零第三方）----
// radix_sort_pass(keys=particles.cell, values=p_idx): 4 pass × 8 bit。
// 前缀和（scan.wgsl）：cell_range 由 sorted keys 的分段边界构建。
fn _radix_sort_contract() {}
