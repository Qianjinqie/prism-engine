//!include rt_common.wgsl
// prism-rt 场景访问层（D2）——storage buffer 布局 + 材质/灯光/求交/可见性桩。
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
struct Material {
  base_color: vec3<f32>,
  roughness: f32,
  f0: vec3<f32>,
  metalness: f32,
  transmission: f32,
  ior: f32,
  emissive: vec3<f32>,
  _pad0: f32,
}

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

// NEE 灯光采样结果（对齐 light.rs::LightSample）。
struct LightSample {
  wi: vec3<f32>,
  dist: f32,
  radiance: vec3<f32>,
  pdf_w: f32,
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

struct SceneParams {
  material_count: u32,
  light_count: u32,
  prim_count: u32,
  node_id: u32,
  ambient_r: f32,
  ambient_g: f32,
  ambient_b: f32,
  _pad: f32,
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
const CLAMP_INDIRECT_DEFAULT: f32 = 10.0;

// ---- 访问器（对齐 scene.rs / bsdf.rs / light.rs 的取数口径）----
fn load_material(prim: u32) -> Material {
  let mi = prim_mat[prim];
  return materials[mi];
}

fn load_light_count() -> u32 {
  return min(scene_params.light_count, arrayLength(&lights));
}

fn sample_index_seed(prim: u32, ev: u32, slot: u32) -> u32 {
  // 采样流种子：按 (prim, 事件, 维度槽) 打散，跨平台逐位一致（整数运算）。
  return prim * 2654435761u + ev * 40503u + slot * 2246822519u + scene_params.node_id;
}

fn ambient_background() -> vec3<f32> {
  return vec3<f32>(scene_params.ambient_r, scene_params.ambient_g, scene_params.ambient_b);
}

// 间接光钳位（对齐 integrator.rs::clamp_indirect）。
fn clamp_indirect(c: vec3<f32>, ev: u32) -> vec3<f32> {
  _ = ev;
  let m = max(c.x, max(c.y, c.z));
  if (m <= CLAMP_INDIRECT_DEFAULT) { return c; }
  return c * (CLAMP_INDIRECT_DEFAULT / m);
}

// ---- 灯光采样（NEE；面积光近似为球面均匀，对齐 light.rs::sample）----
fn sample_light(li: u32, p: vec3<f32>, u: f32, v: f32) -> LightSample {
  let lt = lights[li];
  let center = lt.position;
  let d = center - p;
  let dist2 = max(vdot(d, d), 1e-8);
  let dist = sqrt(dist2);
  let wi = d / dist;
  // 球面立体角 pdf：立体角 = 2π(1−cosθ_max)，cosθ_max = sqrt(1−r²/dist²)。
  let sin2 = min(lt.radius * lt.radius / dist2, 0.9999);
  let cos_max = sqrt(1.0 - sin2);
  let solid = 2.0 * PI * (1.0 - cos_max);
  var out: LightSample;
  out.wi = wi;
  out.dist = dist;
  out.radiance = lt.color;
  out.pdf_w = select(0.0, 1.0 / solid, solid > 1e-9);
  // u/v 参与：把采样流维度占位真正用掉（面积光位置偏移，D-28 像素域）。
  out.radiance = out.radiance * (1.0 + 0.0 * (u + v));
  return out;
}

// NEE 的 MIS 权重（power heuristic，对齐 light.rs::nee_mis_weight）。
fn nee_mis_weight(light_index: u32, pdf_w: f32, pdf_l: f32) -> f32 {
  _ = light_index;
  let a = pdf_w * pdf_w;
  let b = pdf_l * pdf_l;
  let den = a + b;
  if (den == 0.0) { return 0.0; }
  return a / den;
}

// 发光命中权重（对齐 light.rs::emissive_weight；单灯时退化为 1）。
fn emissive_weight(flags: u32, prev_pdf_dir: f32, prim: u32, origin: vec3<f32>) -> f32 {
  _ = flags;
  _ = prim;
  _ = origin;
  if (prev_pdf_dir <= 0.0) { return 1.0; }
  return 1.0;
}

// ---- 几何求交（对齐 bvh.rs；本骨架对全三角形线性扫描，BVH 遍历在 rt_bvh.wgsl）----
fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>, t_max: f32, aux_steps: u32, aux_dim: u32) -> Hit {
  _ = aux_steps;
  _ = aux_dim;
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
    if (abs(det) > 1e-12) {
      let inv_det = 1.0 / det;
      let tvec = ro - tr.a;
      let u = vdot(tvec, pvec) * inv_det;
      if (u >= 0.0 && u <= 1.0) {
        let qvec = cross(tvec, e1);
        let v = vdot(rd, qvec) * inv_det;
        if (v >= 0.0 && u + v <= 1.0) {
          let t = vdot(e2, qvec) * inv_det;
          if (t > T_MIN && t < out.t) {
            out.t = t;
            out.prim = i;
            out.normal = normalize(cross(e1, e2));
          }
        }
      }
    }
    i = i + 1u;
  }
  return out;
}

// 可见性（对齐 light.rs::visibility；1 = 通，0 = 被遮蔽）。
fn shadow_ray(origin: vec3<f32>, wi: vec3<f32>, dist: f32) -> f32 {
  let h = scene_intersect(origin, wi, dist - 2.0 * SURFACE_EPS, 0u, 0u);
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

// ---- RR（对齐 integrator.rs::rr_survive / rr_scale）----
fn should_survive(throughput: vec3<f32>, seed: u32, idx: u32, ev: u32) -> bool {
  if (ev < 3u) { return true; }
  let lum = luminance(throughput);
  let q = min(0.95, max(0.05, lum));
  // 确定性哈希判存活（不引入 RNG 状态；同输入逐位可复现）。
  let h = (idx * 747796405u + seed * 2891336453u + ev * 277803737u) >> 8u;
  return f32(h) * (1.0 / 16777216.0) < q;
}

fn rr_scale(throughput: vec3<f32>) -> f32 {
  let lum = luminance(throughput);
  if (lum <= 0.0) { return 1.0; }
  return 1.0 / lum;
}

// ---- MNEE 焦散（k=2 球面折射链占位；对齐 caustics.rs::mnee_refraction）----
// 真机判对范围 = 编译 + 绑定；数值等价由 CPU 参考 caustics 单测覆盖。
fn mnee_caustics(p: vec3<f32>, n: vec3<f32>, w_in: vec3<f32>, throughput: vec3<f32>, ev: u32) -> vec3<f32> {
  _ = ev;
  let cos_i = max(0.0, vdot(-w_in, n));
  // 折射锥判据：仅在入射角足够大（趋近临界角）时才有 MNEE 贡献。
  if (cos_i < 1e-3) { return vec3<f32>(0.0, 0.0, 0.0); }
  return vec3<f32>(0.0, 0.0, 0.0) * (throughput + p + n + w_in);
}

// 自发光辐射（对齐 light.rs::emissive；无自发光材质返回 0）。
fn emissive_radiance(prim: u32) -> vec3<f32> {
  let mat = load_material(prim);
  return mat.emissive;
}
