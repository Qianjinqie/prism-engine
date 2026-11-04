//!include rt_common.wgsl
// prism-rt 场景访问层 —— storage buffer 布局 + 材质/灯光/求交/可见性桩。
//
// # 为什么需要本文件
// `rt_shade.wgsl`（BSDF/NEE）与 `rt_wavefront.wgsl`（队列四阶段）都调用
// `load_material` / `sample_light` / `scene_intersect` / `shadow_ray` 等场景访问
// 函数。这些函数在 D2 骨架阶段**从未落地为任何文件**，于是两份 shader 在真机上
// 编译必然报 `unresolved call target`（D-14 真机实测首次暴露）。
// 本文件补齐该层：布局（struct）与算法（求交/灯光/可见性）分离——
//
//   * **布局与绑定**（本文件，真实）：storage buffer 的 struct 定义与 `@group/@binding`
//     声明。这就是真机需要判对的部分。
//   * **算法内核**（`rt_bvh.wgsl`，真实）：BVH 遍历 + Möller–Trumbore 三角形命中。
//   * **间接光遮蔽距离**（`shadow_ray`）：本骨架用 BVH 遍历求最近命中距离
//     （对齐 light.rs::visibility 的 `occluded()` 语义），非解析 sphere/box 近似。
//
// 判对基准 = crates/prism-rt/src/{integrator,bsdf,light}.rs。

// ---- 场景 storage 布局（对齐 scene.rs 的串联缓冲；u32 视图 + 位重解释）----
// 材质：f0/基色各 vec3（打包成 4×u32 定点 u8/u10 视为 u32 数组槽）。
// `kind`（T-B R-1 新增，占原 `_pad0` 槽 @40）：材质种类——CPU 侧
// `MaterialKind` 的判别式，路径模式按它选逐位对拍的分支：
//   0 = Lambert（albedo = base_color；R-1 判据范围）
//   1 = Mirror
//   2 = Pbr（Disney 近似；不进 R-1 判据）
//   3 = Emissive（emissive = emissive）
struct Material {
  base_color: vec3<f32>,
  roughness: f32,
  f0: vec3<f32>,
  metalness: f32,
  transmission: f32,
  ior: f32,
  kind: u32,
  emissive: vec3<f32>,
}

// 材质种类（与 `MaterialKind` 判别式对齐；见上）。
const MAT_LAMBERT: u32 = 0u;
const MAT_MIRROR: u32 = 1u;
const MAT_PBR: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

// 灯光：位置/功率（点光 + 面积光以半径表达，对齐 light.rs 四灯光族）。
struct Light {
  position: vec3<f32>,
  radius: f32,
  color: vec3<f32>,
  _pad: f32,
}

// 求交命中记录（对齐 bvh.rs::Hit）。
struct Hit {
  t: f32,
  prim: u32,
  _pad0: u32,
  _pad1: u32,
  normal: vec3<f32>,
  _pad2: f32,
}

// ---- 场景绑定（group 2 = 场景只读数据；与 group 0 队列 / group 1 采样流分离）----
//
// **绑定次序刻意以"被多少 entry 引用"升序排列**（scene_params 最广→lights 最窄）：
// 宿主把句柄数组第 i 个硬绑到 binding:i，且 layout:"auto" 只为被 dispatch 的 entry
// 静态引用的 binding 生成布局 ⇒ 每个 entry 的引用集必须是连续前缀 {0..n-1}。
// 三个入口的实际引用集：
//   generate → {scene_params}                  = {0}
//   shade    → {materials,lights,prim_mat,scene_params,tris} = {0,1,2,3,4}
//   extend   → {materials,prim_mat,scene_params,tris}        = {0,1,2,3}
// 故 scene_params 必须在 binding 0、lights 必须最后。原次序
// (materials/lights/prim_mat/scene_params/tris) 会让 generate={3}、extend={0,2,3,4}
// 都不是前缀 → createBindGroup 必然验证失败。
@group(2) @binding(0) var<uniform> scene_params: SceneParams;      // 计数 + 环境光 + 节点 id
@group(2) @binding(1) var<storage, read> materials: array<Material>;
@group(2) @binding(2) var<storage, read> prim_mat: array<u32>;    // prim → material 索引
@group(2) @binding(3) var<storage, read> tris: array<Tri>;
@group(2) @binding(4) var<storage, read> lights: array<Light>;
// group(2) @binding(6)：**CpuRef 兼容模式**（W1）的球表。
//
// 为什么需要它：megakernel（`spirv_rt.rs`）的判对基准是 `CpuRefBackend` 的
// 球面单弹射模型，而 `tris`/`materials` 表达不了"球心 + 半径 + 反照率"这套
// 语义。绑定 6 只被 `shade` 的兼容分支静态引用，`shade` 的引用集因此是
// {0,1,2,3,4,5,6,7} —— 仍是连续前缀；`extend` 仍是 {0,1,2,3,4}。
//
// # 为什么是 `array<f32>` 而不是 `struct Sphere`
//
// megakernel 的球表是**平铺 8 f32**（`center.xyz @0..11 / radius @12 /
// albedo.xyz @13..23 / pad @28`）。WGSL 的 `vec3<f32>` 对齐 16，写成 struct
// 会把 albedo 推到 16——与 megakernel 的字节流错位 4 字节。故这里按**字索引**
// 读取（兼容模式专用；路径模式的 `tris`/`materials` 仍是结构体）。
@group(2) @binding(6) var<storage, read> spheres: array<f32>;
// group(2) @binding(7)：兼容模式的**方向光表**（megakernel 的平铺 8 f32：
// `dir.xyz @0..11 / color.xyz @12..23 / intensity @24 / pad @28`）。
// 与 `rt_scene::Light`（`position/radius/color` 的 std430 布局）**不是**
// 同一份字节，故单开一条只读绑定；`shade` 的引用集因此是 {0..7}。
@group(2) @binding(7) var<storage, read> dir_lights: array<f32>;

struct SceneParams {
  material_count: u32,
  light_count: u32,
  prim_count: u32,
  node_id: u32,
  ambient_r: f32,
  ambient_g: f32,
  ambient_b: f32,
  // 原 `_pad`：现承载**球体个数**（CpuRef 兼容模式用；路径模式恒 0）。
  // 位置/大小未动（u32 @ 28），故 32 字节的 uniform 布局与 W2 完全相同。
  sphere_count: u32,
}

// ---- 顶点缓冲（三角形；group 2 binding 3）----
struct Tri {
  a: vec3<f32>,
  _p0: f32,
  b: vec3<f32>,
  _p1: f32,
  c: vec3<f32>,
  _p2: f32,
}

// 间接光钳位开关（wavefront 的 params 在 group 0，此处只读场景级默认）。

// ---- 访问器（对齐 scene.rs / bsdf.rs / light.rs 的取数口径）----
fn load_material(prim: u32) -> Material {
  let mi = prim_mat[prim];
  return materials[mi];
}

fn load_light_count() -> u32 {
  return min(scene_params.light_count, arrayLength(&lights));
}

fn ambient_background() -> vec3<f32> {
  return vec3<f32>(scene_params.ambient_r, scene_params.ambient_g, scene_params.ambient_b);
}

// ---- 几何求交（对齐 prism-geometry/bvh.rs::tri_ray_hit 逐式；T-B R-1 逐位对拍）----
//
// 与 CPU 的逐位契约（每一处都不能"约等"）：
// * `det.abs() < 1e-12` ⇒ 该三角形永不命中（同 CPU）；
// * u / v 的接受域是**闭区间** [0,1]（CPU：`u < 0.0 || u > 1.0` 拒绝）；
// * t 的接受域是**闭区间** [T_MIN, t_max]（CPU：`hit >= t_min && hit <= t_max`）
//   —— 可见性射线的 t_max = dist − T_MIN，端点上的遮挡**算**遮挡；
// * 并列 t 保持先达者（CPU 的 BVH tie-break 按 (t, 原始索引)，线性扫描按
//   先达者；并列 t 是测度零事件，场景设计避开，见 parity 测试注释）；
// * 法线 = 三角形单位几何法线，与 `triangle_unit_normal` 同一棵树：
//   `n·n > 1e-24 ? n·(1/√(n·n)) : 0`。
// ---------------------------------------------------------------------------
// G-8① wavefront 腿的几何实例层级（真实例遍历，不是展平/等价路径）
// ---------------------------------------------------------------------------
//
// # 为什么零新绑定
//
// `layout:"auto"` 的硬约束是「每个 entry 的静态引用集为连续前缀 {0..n-1}」。
// `shade` 引用 group(2) 的 {0..7}、`extend` 引用 {0,1,2,3}；若把实例表挂到
// 一个新的 binding 8/9/…，则 `extend` 的引用集会变成 {0,1,2,3,8,9}（稀疏）
// ⇒ createBindGroup 必然验证失败，且 `web/abi.rs` 的多 bind group harness 与
// `wgsl_lint.mjs` 的前缀门禁都按旧前缀硬编码。故实例数据走**已经同时被
// shade/extend 静态引用**的两条既有缓冲：
//
//   * group(2) b2 `prim_mat: array<u32>` —— 尾部承载实例表 / 原型表 / 描述符；
//   * group(2) b3 `tris: array<Tri>`     —— 尾部承载各 BLAS 的局部三角形。
//
// 绑定剖面一个字没变 ⇒ Vulkan 描述符表与 WebGPU bind group 前缀表都不动，
// 两端消费同一张 `vulkan::wavefront::build_resources` 资源表（跨后端单一
// 真源），实例字节自然同时到达两个后端（§4.4 红线 2）。
//
// # 数据布局（`prim_mat` 尾部 8 字 = 实例描述符；magic=0 ⇒ 无实例，
//    下面的实例段一行都不执行，既有线性三角形路径**逐位不变**）
//
// ```text
//   prim_mat[0 .. total_tri)                 三角形 → 材质索引（普通 + BLAS）
//   prim_mat[instance_base + 28*i + slot]    实例记录（112 B / 28 字；与
//                                           instance.rs::to_bytes 逐字相同）
//        槽 0..12 affine / 12..24 inverse（行主序 3×4）
//        槽 24 geom_id / 25 mat_id / 26 flags / 27 node_base
//   prim_mat[geom_base + 4*g + {0,1,2,3}]    原型表：tri_start / tri_count /
//                                           mat_id / pad
//   prim_mat[W-8 .. W)                       描述符（W = arrayLength）：
//        [0] WF_INST_MAGIC  [1] instance_count  [2] geom_count
//        [3] instance_base  [4] geom_base        [5] plain_tri_count
//        [6] blas_tri_base  [7] total_tri_count
// ```
//
// # 三条与 CPU 权威（`prism_rt::Scene::instance_nearest`）逐式相同的纪律
//
// 1. **`t` 不换算**：局部射线 `o_l = A⁻¹o`、`d_l = A⁻¹d`（方向**不归一化**），
//    于是 `o_l + t·d_l ≡ A⁻¹(o + t·d)` 对同一个 `t` 成立 ⇒ `T_MIN` 与
//    `t_max` 都是世界量，直接当局部查询的界用。identity 实例一次乘法都不做。
// 2. **法线 `det(A)·A⁻ᵀ·n`**：`A⁻ᵀ` 就是 `inverse` 3×3 块的转置（宿主已算好
//    逆，绝不在 shader 里求逆）；`det<0` 由 flags bit1 决定是否反号。归一化
//    与翻到来光侧的口径与 `Scene::normal_at` 的 Instance 臂一致。
// 3. **多原型**：`geom_id` 索引原型表取该实例自己的 `[tri_start, tri_count)`，
//    每条实例只遍历自己那一个 BLAS；最近命中按世界 `t` 严格择优。

const WF_INST_MAGIC: u32 = 0x57384931u;   // "W8I1"（0 ⇒ 无实例）
const WF_INST_TAIL: u32 = 8u;              // 描述符字数
const WF_INST_STRIDE: u32 = 28u;           // == instance::INSTANCE_SLOTS（112 B）
const WF_INST_INVERSE: u32 = 12u;          // 逆矩阵在记录内的槽偏移
const WF_INST_GEOM: u32 = 24u;             // geom_id 槽
const WF_INST_FLAGS: u32 = 26u;            // flags 槽
const WF_INST_FLAG_IDENTITY: u32 = 1u;
const WF_INST_FLAG_MIRRORED: u32 = 2u;
const WF_GEOM_STRIDE: u32 = 4u;            // 原型表 4 字/条

fn wf_inst_active() -> bool {
  let w = arrayLength(&prim_mat);
  return w >= WF_INST_TAIL && prim_mat[w - WF_INST_TAIL] == WF_INST_MAGIC;
}

/// 读尾部描述符的第 `slot` 字（只在 `wf_inst_active()` 为真时调用）。
fn wf_tail(slot: u32) -> u32 {
  return prim_mat[arrayLength(&prim_mat) - WF_INST_TAIL + slot];
}

/// 世界点 → 局部点（`A⁻¹p + t⁻¹`；调用点已排除 identity）。
fn wf_inst_local_point(base: u32, p: vec3<f32>) -> vec3<f32> {
  let b = base + WF_INST_INVERSE;
  return vec3<f32>(
    bitcast<f32>(prim_mat[b]) * p.x + bitcast<f32>(prim_mat[b + 1u]) * p.y
      + bitcast<f32>(prim_mat[b + 2u]) * p.z + bitcast<f32>(prim_mat[b + 3u]),
    bitcast<f32>(prim_mat[b + 4u]) * p.x + bitcast<f32>(prim_mat[b + 5u]) * p.y
      + bitcast<f32>(prim_mat[b + 6u]) * p.z + bitcast<f32>(prim_mat[b + 7u]),
    bitcast<f32>(prim_mat[b + 8u]) * p.x + bitcast<f32>(prim_mat[b + 9u]) * p.y
      + bitcast<f32>(prim_mat[b + 10u]) * p.z + bitcast<f32>(prim_mat[b + 11u]));
}

/// 世界方向 → 局部方向（只乘线性块，**不归一化** ⇒ `t` 保持世界参数）。
fn wf_inst_local_dir(base: u32, d: vec3<f32>) -> vec3<f32> {
  let b = base + WF_INST_INVERSE;
  return vec3<f32>(
    bitcast<f32>(prim_mat[b]) * d.x + bitcast<f32>(prim_mat[b + 1u]) * d.y
      + bitcast<f32>(prim_mat[b + 2u]) * d.z,
    bitcast<f32>(prim_mat[b + 4u]) * d.x + bitcast<f32>(prim_mat[b + 5u]) * d.y
      + bitcast<f32>(prim_mat[b + 6u]) * d.z,
    bitcast<f32>(prim_mat[b + 8u]) * d.x + bitcast<f32>(prim_mat[b + 9u]) * d.y
      + bitcast<f32>(prim_mat[b + 10u]) * d.z);
}

/// 局部绕序法线 → 世界法线（`A⁻ᵀ`，镜像反号），未归一化。
fn wf_inst_world_normal(base: u32, n: vec3<f32>) -> vec3<f32> {
  let b = base + WF_INST_INVERSE;
  let out = vec3<f32>(
    bitcast<f32>(prim_mat[b]) * n.x + bitcast<f32>(prim_mat[b + 4u]) * n.y
      + bitcast<f32>(prim_mat[b + 8u]) * n.z,
    bitcast<f32>(prim_mat[b + 1u]) * n.x + bitcast<f32>(prim_mat[b + 5u]) * n.y
      + bitcast<f32>(prim_mat[b + 9u]) * n.z,
    bitcast<f32>(prim_mat[b + 2u]) * n.x + bitcast<f32>(prim_mat[b + 6u]) * n.y
      + bitcast<f32>(prim_mat[b + 10u]) * n.z);
  if ((prim_mat[base + WF_INST_FLAGS] & WF_INST_FLAG_MIRRORED) != 0u) {
    return -out;
  }
  return out;
}

/// 三角形求交（Möller–Trumbore，逐式照抄原 `scene_intersect` 内联块）。
/// 返回 `(t, hit)`：`hit = 1` 表示 `T_MIN ≤ t ≤ t_max`，否则 `hit = 0`。
fn wf_tri_hit(slot: u32, ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> vec2<f32> {
  let tr = tris[slot];
  let e1 = tr.b - tr.a;
  let e2 = tr.c - tr.a;
  let pvec = cross(rd, e2);
  let det = vdot(e1, pvec);
  if (abs(det) >= 1e-12) {
    let inv_det = 1.0 / det;
    let tvec = ro - tr.a;
    let u = vdot(tvec, pvec) * inv_det;
    if (u >= 0.0 && u <= 1.0) {
      let qvec = cross(tvec, e1);
      let v = vdot(rd, qvec) * inv_det;
      if (v >= 0.0 && u + v <= 1.0) {
        let t = vdot(e2, qvec) * inv_det;
        if (t >= T_MIN && t <= t_max) {
          return vec2<f32>(t, 1.0);
        }
      }
    }
  }
  return vec2<f32>(t_max, 0.0);
}

/// 三角形**未归一化**绕序法线 `cross(b−a, c−a)`（与 `triangle_unit_normal` 同树）。
fn wf_tri_raw_normal(slot: u32) -> vec3<f32> {
  let tr = tris[slot];
  return cross(tr.b - tr.a, tr.c - tr.a);
}

/// 单位化（`n·n > 1e-24` 才归一；否则零向量，与 CPU 同口径）。
fn wf_unit_normal(n: vec3<f32>) -> vec3<f32> {
  let len2 = vdot(n, n);
  if (len2 > 1e-24) {
    return n * (1.0 / sqrt(len2));
  }
  return vec3<f32>(0.0, 0.0, 0.0);
}

fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> Hit {
  var out: Hit;
  out.t = t_max;
  out.prim = 0xFFFFFFFFu;
  out._pad0 = 0u;
  out._pad1 = 0u;
  out.normal = vec3<f32>(0.0, 0.0, 0.0);
  out._pad2 = 0.0;
  // 既有上界表达式**原样保留**（`wgsl_lint` 的 min 三参负控锚在这一行）。
  let n = min(scene_params.prim_count, arrayLength(&tris));
  // 实例模式下 `tris` 尾部是各 BLAS 的局部三角形，只能经实例到达 ⇒ 线性
  // 扫描上界收窄到"普通三角形数"（描述符 [5]），否则局部几何会被当世界
  // 几何画出来。无实例时 `plain == n`，与改动前逐位相同。
  var plain = n;
  if (wf_inst_active()) {
    plain = wf_tail(5u);
  }
  var i = 0u;
  while (i < plain) {
    let h = wf_tri_hit(i, ro, rd, out.t);
    if (h.y > 0.0 && (h.x < out.t || out.prim == 0xFFFFFFFFu)) {
      out.t = h.x;
      out.prim = i;
      out.normal = wf_unit_normal(wf_tri_raw_normal(i));
    }
    i = i + 1u;
  }
  // ---- 实例族（末族；与 `Scene::intersect_impl` 同序：running best 作上界）----
  if (wf_inst_active()) {
    let ni = wf_tail(1u);
    let ib = wf_tail(3u);
    let gb = wf_tail(4u);
    let words = arrayLength(&prim_mat);
    let ntris = arrayLength(&tris);
    var k = 0u;
    while (k < ni) {
      let base = ib + k * WF_INST_STRIDE;
      // 越界守卫：叶子里存的下标必须落在表内，绝不让越界读变成"读到
      // 邻居的逆矩阵"（与 `scene_frame_v3.wgsl::instance_nearest` 同纪律）。
      if (base + WF_INST_STRIDE <= words) {
        let identity = (prim_mat[base + WF_INST_FLAGS] & WF_INST_FLAG_IDENTITY) != 0u;
        var ro_l = ro;
        var rd_l = rd;
        if (!identity) {
          ro_l = wf_inst_local_point(base, ro);
          rd_l = wf_inst_local_dir(base, rd);
        }
        let gi = gb + prim_mat[base + WF_INST_GEOM] * WF_GEOM_STRIDE;
        if (gi + 2u <= words) {
          let t_start = prim_mat[gi];
          let t_count = prim_mat[gi + 1u];
          var j = 0u;
          while (j < t_count) {
            let slot = t_start + j;
            if (slot < ntris) {
              let h = wf_tri_hit(slot, ro_l, rd_l, out.t);
              if (h.y > 0.0 && (h.x < out.t || out.prim == 0xFFFFFFFFu)) {
                out.t = h.x;
                out.prim = slot;
                let nl = wf_tri_raw_normal(slot);
                if (identity) {
                  // identity 与展平版**同一条**归一化路径（零多余运算）。
                  out.normal = wf_unit_normal(nl);
                } else {
                  out.normal = wf_unit_normal(wf_inst_world_normal(base, nl));
                }
              }
            }
            j = j + 1u;
          }
        }
      }
      k = k + 1u;
    }
  }
  return out;
}

// 可见性（对齐 integrator.rs::visibility 的遮挡语义：max_t − T_MIN 为上界，
// 闭区间 ⇒ 端点命中算遮挡；无介质场景 vis ∈ {0,1}）。
fn shadow_ray(origin: vec3<f32>, wi: vec3<f32>, dist: f32) -> f32 {
  let h = scene_intersect(origin, wi, dist - T_MIN);
  if (h.prim == 0xFFFFFFFFu) { return 1.0; }
  return 0.0;
}

// ---- BSDF 求值（Disney 简化：Lambert 漫反 + Schlick 镜面；对齐 bsdf.rs::eval）----
fn eval_bsdf(mat: Material, w_in: vec3<f32>, w_out: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let cos_i = max(0.0, vdot(-w_in, n));
  let cos_o = max(0.0, vdot(w_out, n));
  if (cos_i <= 0.0 || cos_o <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }
  let diffuse = mat.base_color * (1.0 / PI) * (1.0 - mat.metalness);
  let f = schlick_f0(mat, cos_o);
  let spec = f * pow5_local(1.0 - min(cos_i, cos_o));
  return (diffuse + spec) * cos_o;
}

fn schlick_f0(mat: Material, cos_theta: f32) -> vec3<f32> {
  let m5 = pow5_local(max(0.0, 1.0 - cos_theta));
  return mat.f0 + (vec3<f32>(1.0, 1.0, 1.0) - mat.f0) * m5;
}

// pow5 = x^5。本文件必须能独立编译（不依赖 rt_shade），故用独立命名
// `pow5_local`——WGSL 无命名空间，同名 fn 在 include 展开后会 redeclaration。
fn pow5_local(x: f32) -> f32 {
  let x2 = x * x;
  return x2 * x2 * x;
}

// ---- RR / MNEE / 发光读取：已上收 —— ----
// * RR（rr_survive/rr_scale）与间接钳位：路径模式逐位对拍版本在
//   `rt_wavefront.wgsl`（wf_rr_survive / wf_clamp_indirect，消费 group 0 的
//   WFParams 与 wf 采样流）；
// * MNEE 焦散：CPU 侧只在 `scene.refractors()` 非空时触发，路径模式的场景
//   ABI 不携带折射体 ⇒ 该段在 WGSL 中不存在（与 CPU 空循环逐位等价）；
// * 发光命中：直接读 `Material.emissive`（旧 `emissive_radiance` 已删）。
