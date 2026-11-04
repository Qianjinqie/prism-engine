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
fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> Hit {
  var out: Hit;
  out.t = t_max;
  out.prim = 0xFFFFFFFFu;
  out._pad0 = 0u;
  out._pad1 = 0u;
  out.normal = vec3<f32>(0.0, 0.0, 0.0);
  out._pad2 = 0.0;
  let n = min(scene_params.prim_count, arrayLength(&tris));
  var i = 0u;
  while (i < n) {
    let tr = tris[i];
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
          if (t >= T_MIN && t <= out.t) {
            if (t < out.t || out.prim == 0xFFFFFFFFu) {
              out.t = t;
              out.prim = i;
              // 三角形单位几何法线（绕序 a→b→c 右手；与
              // `triangle_unit_normal` 同树：n·n > 1e-24 才归一化）。
              let un = cross(e1, e2);
              let len2 = vdot(un, un);
              if (len2 > 1e-24) {
                out.normal = un * (1.0 / sqrt(len2));
              } else {
                out.normal = vec3<f32>(0.0, 0.0, 0.0);
              }
            }
          }
        }
      }
    }
    i = i + 1u;
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
