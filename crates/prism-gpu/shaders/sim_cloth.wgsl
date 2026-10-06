// prism-sim 布料 compute（D3）——XPBD 距离/弯曲约束投影。
//
// # 对应关系（判对基准 = prism-physics/cloth.rs）
// - 结构约束（水平 + 垂直相邻，rest = spacing）与弯曲约束（隔一，rest = 2·spacing）。
// - XPBD 投影：Δλ = (−C − α̃λ)/(Σw|∇C|² + α̃)（对齐 xpbd.rs）。
// - 求解序：约束按 id 升序（GPU 上 = 网格 order 着色分组，保确定性）。
//
// # 绑定布局为什么是 AoS（pos/vel/inv 合并成一个 storage）
//
// 宿主胶水 `prism_create_bind_group` 把句柄数组的**第 i 个固定映射到
// `binding: i`**，而 `layout:"auto"` 只为被 dispatch 的 entry **静态引用**的
// binding 生成布局。两者相乘的硬约束是：**每个 entry 的静态引用集合必须恰好
// 是连续前缀 `{0,1,…,n-1}`**（前缀必嵌套，故 SoA 的
// `{pos,vel,inv,cparams}` / `{pos,inv,cons,lam,cparams}` / `{vel,inv,cparams}`
// 三个集合互不包含，任何编号方案都无解）。AoS 后三者分别成为
// {0,1} / {0,1,2,3} / {0,1}，都满足前缀。
// 详细推导见 `sim_sph.wgsl` 头注（同一约束的另一个实例）。

//!include sim_common.wgsl
struct ClothParams {
  rows: u32,
  cols: u32,
  spacing: f32,
  dt: f32,
  compliance_struct: f32,
  compliance_bend: f32,
  damping: f32,
  gravity: vec3<f32>,
  _pad: u32,
}

struct Constraint {
  ia: u32,
  ib: u32,
  rest: f32,
  id: u32,
}

/// 布料粒子状态（AoS）：`pos:vec3@0` / `vel:vec3@16` / `inv:f32@28` → **32 字节**。
struct ClothState {
  pos: vec3<f32>,
  vel: vec3<f32>,
  inv: f32,
}

@group(0) @binding(0) var<uniform> cparams: ClothParams;
@group(0) @binding(1) var<storage, read_write> cs: array<ClothState>;
@group(0) @binding(2) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> lambdas: array<f32>;

// ---- 预测（半隐式欧拉；对齐 cloth.rs step 的 World::step 预测段）----
@compute @workgroup_size(64)
fn cloth_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cparams.rows * cparams.cols) { return; }
  if (cs[i].inv > 0.0) {
    cs[i].vel = cs[i].vel + cparams.gravity * cparams.dt;
    cs[i].pos = cs[i].pos + cs[i].vel * cparams.dt;
  }
}

// ---- 约束投影（一迭代；对齐 xpbd.rs DistanceConstraint::project）----
@compute @workgroup_size(64)
fn cloth_project(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&constraints)) { return; }
  let c = constraints[i];
  var pa = cs[c.ia].pos;
  var pb = cs[c.ib].pos;
  var lam = lambdas[i];
  let compliance = select(cparams.compliance_struct, cparams.compliance_bend, c.rest > cparams.spacing * 1.5);
  _ = xpbd_distance_project(&pa, &pb, cs[c.ia].inv, cs[c.ib].inv, c.rest, compliance, cparams.dt, &lam);
  cs[c.ia].pos = pa;
  cs[c.ib].pos = pb;
  lambdas[i] = lam;
}

// ---- 速度更新 + 阻尼（对齐 cloth.rs apply_damping）----
@compute @workgroup_size(64)
fn cloth_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cparams.rows * cparams.cols) { return; }
  if (cs[i].inv > 0.0) {
    cs[i].vel = cs[i].vel * (1.0 - cparams.damping);
  }
}
