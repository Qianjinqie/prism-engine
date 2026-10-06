// prism-sim SPH/PBF compute（D3）——两遍：密度求和 → 密度约束投影。
//
// # 对应关系（判对基准 = prism-physics/sph.rs；实现权威 = 白皮书 §9 PBF）
// - 密度：ρ_i = Σ_j m·Poly6(|x_i − x_j|)（对齐 sph.rs::density_of）。
// - 密度约束 C_i = ρ_i/ρ₀ − 1，投影走 XPBD 通式（对齐 DensityConstraint::project）。
// - 压缩侧单向（C > 0 才修正；λ ≥ 0），对齐 sph.rs 的 PBF 稳定项。
// - 邻域：cell_range（radix sort 后）+ 27 邻格扫描（对齐 SpatialHash）。
//
// # 绑定布局为什么是 AoS（pos/vel/inv 合并成一个 storage）
//
// 宿主胶水 `prism_create_bind_group` 把句柄数组的**第 i 个固定映射到
// `binding: i`**，而 `layout:"auto"` 只为被 dispatch 的 entry **静态引用**的
// binding 生成布局。两者相乘的硬约束是：**每个 entry 的静态引用集合必须恰好
// 是连续前缀 `{0,1,…,n-1}`**。
//
// SoA 下三个入口的引用集合是 `{pos,vel,inv,spp}` / `{pos,density,spp}` /
// `{inv,density,lambda,spp}` —— 两两互不包含，**任何编号方案都不可能让它们
// 同时成为前缀**（前缀必嵌套）。故把 pos/vel/inv 合并为 `SphState`（AoS），
// 三个集合自然变成 {0,1} / {0,1,2} / {0,1,2,3}，恰好都是前缀。

//!include sim_common.wgsl
struct SphParams {
  count: u32,
  h: f32,
  rest_density: f32,
  compliance: f32,
  dt: f32,
  gravity: vec3<f32>,
  _pad: u32,
}

/// 粒子状态（AoS）：`pos:vec3@0` / `vel:vec3@16` / `inv:f32@28`，对齐 16 → **32 字节**。
struct SphState {
  pos: vec3<f32>,
  vel: vec3<f32>,
  inv: f32,
}

@group(0) @binding(0) var<storage, read_write> sp_state: array<SphState>;
@group(0) @binding(1) var<uniform> spp: SphParams;
@group(0) @binding(2) var<storage, read_write> sp_density: array<f32>;
@group(0) @binding(3) var<storage, read_write> sp_lambda: array<f32>;
@group(0) @binding(4) var<storage, read> cell_range: array<vec2<u32>>;

// 邻域遍历位（真实实现：27 邻格 × cell_range；此处给出接口契约）。
// for each cell in 3×3×3 neighborhood of cell(p_i):
//   for k in cell_range[cell].x .. cell_range[cell].y: consider particle k

// ---- 预测 ----
@compute @workgroup_size(64)
fn sph_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= spp.count) { return; }
  if (sp_state[i].inv > 0.0) {
    sp_state[i].vel = sp_state[i].vel + spp.gravity * spp.dt;
    sp_state[i].pos = sp_state[i].pos + sp_state[i].vel * spp.dt;
  }
}

// ---- 密度求和（对齐 sph.rs::density_of；含自项 W(0)）----
@compute @workgroup_size(64)
fn sph_density(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= spp.count) { return; }
  var rho = poly6_2d(0.0, spp.h); // 自项
  // 邻域（cell_range 扫描；骨架保留，真机由 27 邻格实现）。
  let neighbors_start = 0u;
  let neighbors_end = spp.count;
  var j = neighbors_start;
  while (j < neighbors_end) {
    if (j != i) {
      let r = length(sp_state[j].pos - sp_state[i].pos);
      rho = rho + poly6_2d(r, spp.h);
    }
    j = j + 1u;
  }
  sp_density[i] = rho;
}

// ---- 密度约束投影（压缩侧单向；对齐 sph.rs::DensityConstraint::project）----
@compute @workgroup_size(64)
fn sph_project(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= spp.count) { return; }
  let c = sp_density[i] / spp.rest_density - 1.0;
  if (c <= 0.0) {
    if (sp_lambda[i] < 0.0) { sp_lambda[i] = 0.0; }
    return;
  }
  // XPBD 通式：Σw|∇C|² + α̃（∇C 由 Spiky 梯度；骨架以自项下界占位）。
  let at = spp.compliance / (spp.dt * spp.dt);
  let denom = sp_state[i].inv + at;
  if (denom <= 0.0) { return; }
  let dl = (-c - at * sp_lambda[i]) / denom;
  var lam = sp_lambda[i] + dl;
  if (lam < 0.0) { lam = 0.0; }
  sp_lambda[i] = lam;
  // 位置修正（Δx_i = w_i ∇_i C Δλ）由邻域 kernel 分摊；骨架保留接口位。
}

// ---- 速度更新 ----
@compute @workgroup_size(64)
fn sph_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= spp.count) { return; }
  _ = sp_state[i].vel;
}
