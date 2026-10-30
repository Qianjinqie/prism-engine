// prism-rt GI 的 GPU compute 路径（实时 GI 的 GPU 路径）。
//
// # 对应关系（判对基准 = crates/prism-rt/src/gi/{sdf_gi.rs, sh.rs}）
// 本文件是 CPU 侧实时 GI（SDF sphere tracing 遮挡 + SH-L2 辐照度）的 **GPU 镜像**，
// 逐式对齐两条 CPU 契约：
//
// 1. **SDF 遮挡**（对齐 `sdf_gi.rs::is_occluded` + `prism-geometry::sdf::sphere_trace`）：
//    沿射线按 `dist` 值步进，`dist(ro + rd*t) <= hit_eps` 即命中（含起点在体内），
//    步长 `t += d * safety`（Lipschitz=1，safety<=1 保证不穿越），`t > t_max` 或步数
//    耗尽即未命中。方向非法（非有限/退化）→ 视为无遮挡（C18 不 panic 的同构）。
//
// 2. **SH-L2 投影/重建 + π 约定**（对齐 `sh.rs`，跨文件契约，务必一致）：
//      投影**原始辐亮度**：  L_lm = ∫ L(ω')·Y_lm(ω') dΩ                 （不乘 cos）
//      重建加余弦瓣：        E(ω) ≈ Σ_lm Â_l · L_lm · Y_lm(ω)
//      余弦瓣系数：          Â_0 = π,  Â_1 = 2π/3,  Â_2 = π/4
//      辐照度→辐亮度：       L_o = albedo · E(n) · (1/π)                （Lambert BRDF）
//    本文件 `gi_sh_project` 做投影（→ `GiProbe.sh`），`gi_probe_interpolate` 做重建。
//    常数环境自检：E(ω) = π·L0（各向同性，与 ω 无关）——见 abi.rs 的 CPU 参考断言。
//
// # layout:"auto" 三条硬约束（本文件的绑定布局据此设计，逐条给出满足方式）
//
//  (1) bind group 绑定数必须**恰好等于**该 entry 静态引用的 binding 数；
//  (2) 绑定集必须是**连续前缀** {0..n-1}，不许跳号（宿主胶水把句柄数组第 i 个
//      硬绑到 `binding: i`）；
//  (3) 多 uniform 的 shader 逐个校验每个 uniform（本文件只 1 个 uniform：`gi_params`）。
//
//  三个 entry 的静态引用集合（binding 编号见下方 var 声明）：
//      gi_probe_trace        → { gi_probes(0), gi_params(1) }        = 前缀 {0,1} ✓
//      gi_sh_project         → { gi_probes(0), gi_params(1) }        = 前缀 {0,1} ✓
//      gi_probe_interpolate  → { gi_probes(0), gi_params(1), gi_points(2) } = 前缀 {0,1,2} ✓
//  每个集合都是前缀（`{0,1}` ⊂ `{0,1,2}` 嵌套成立），故**同一条前缀编号方案**
//  对三个 entry 同时成立。若把 scene/coeffs/points 拆成互不包含的独立 binding，
//  集合两两互不包含（前缀必嵌套）→ 无解；这正是把"逐方向采样 + SH 系数"合并进
//  单个 `GiProbe`（AoS）的原因（同 sim_sph 的 SphState 论证）。
//
// # 确定性（铁律 2）
// 所有超越三角走 `cos_sin`（rt_common 的固定系数 Maclaurin 区间逼近，禁 libm/D-29）；
// 采样方向为固定 Fibonacci 半球序列（无随机、无时间），累加序固定（方向序 × 系数序）。
// 无 `for` 之外的控制流依赖外部状态；同输入逐位可复现。

//!include rt_common.wgsl

// ---- 采样方向数与场景图元容量（编译期常量，与 abi.rs 侧对应常量一致）----
const GI_DIRS: u32 = 8u;      // 每条探针的半球采样方向数（≤ GiProbe.samples 容量）
const GI_MAX_PRIMS: u32 = 4u; // uniform 内 SDF 图元容量

// ---------------------------------------------------------------------------
// uniform / storage 结构（16 字节对齐 + 显式 _pad）
// ---------------------------------------------------------------------------

/// SDF 图元（球 / 盒）：`kind=0` 球（center+radius），`kind=1` 盒（center+half）。
///
/// 布局：`kind:u32@0` / `radius:f32@4` / `center:vec3<f32>@16` / `_pad:vec3<u32>@28`
/// → 末成员末尾 40 → 对齐 16 → **48 字节**（stride 48）。
struct GiPrim {
  kind: u32,
  radius: f32,
  center: vec3<f32>,
  _pad: vec3<u32>,
}

/// GI 参数（uniform，16 字节对齐）。
///
/// 布局：`probe_count@0` / `point_count@4` / `prim_count@8` / `n_dirs@12` /
/// `env_radiance:vec3@16` / `_pad0@28` / `albedo:vec3@32` / `_pad1@44` /
/// `t_max@48` / `hit_eps@52` / `safety@56` / `max_steps@60` /
/// `prims:array<GiPrim,4>@64`（4×48=192）→ 末尾 256 → 对齐 16 → **256 字节**。
///
/// 注意：`vec3<f32>` 后的 4 字节是**对齐填充**（自动）；结构体末尾的显式 `_pad*`
/// 是**尾部填充**（消隐式），两者语义不同（前者 WGSL 自动补，后者我们手写以固定布局）。
struct GiParams {
  probe_count: u32,
  point_count: u32,
  prim_count: u32,
  n_dirs: u32,
  env_radiance: vec3<f32>,
  _pad0: f32,
  albedo: vec3<f32>,
  _pad1: f32,
  t_max: f32,
  hit_eps: f32,
  safety: f32,
  max_steps: u32,
  prims: array<GiPrim, 4>,
}

/// 一条 GI 探针（AoS）：位置 + 法线 + 逐方向采样辐亮度 + SH-L2 系数。
///
/// 布局：`pos:vec3@0` / `_pad0@12` / `normal:vec3@16` / `_pad1@28` /
/// `samples:array<vec3<f32>,8>@32`（8×16=128 → 末 160）/
/// `sh:array<vec3<f32>,9>@160`（9×16=144 → 末 304）→ 对齐 16 → **304 字节**。
///
/// 【为何 AoS】见文件头"三条硬约束"：把逐方向采样与 SH 系数并进同一 binding，
/// 使三个 entry 的引用集合都是前缀。
struct GiProbe {
  pos: vec3<f32>,
  _pad0: f32,
  normal: vec3<f32>,
  _pad1: f32,
  samples: array<vec3<f32>, 8>,
  sh: array<vec3<f32>, 9>,
}

/// 插值查询点（AoS）：位置 + 出射辐照度（重建结果）。
///
/// 布局：`pos:vec3@0` / `_pad0@12` / `irradiance:vec3@16` / `_pad1@28`
/// → 对齐 16 → **32 字节**。
struct GiPoint {
  pos: vec3<f32>,
  _pad0: f32,
  irradiance: vec3<f32>,
  _pad1: f32,
}

// ---- 绑定（binding 编号即"连续前缀"方案的编号，勿随意重排）----
@group(0) @binding(0) var<storage, read_write> gi_probes: array<GiProbe>;
@group(0) @binding(1) var<uniform> gi_params: GiParams;
@group(0) @binding(2) var<storage, read_write> gi_points: array<GiPoint>;

// ---------------------------------------------------------------------------
// SDF 求值（对齐 CPU `SceneSdf`：球/盒 Union）
// ---------------------------------------------------------------------------

/// 单图元到点 `p` 的有符号距离（内负外正）。
///
/// - `kind == 0`：球 —— `|p - c| - r`（对齐 `Csg::Sphere`）；
/// - `kind == 1`：盒 —— 标准的盒 SDF（`q = |p-c| - r` 三分量 →
///   `|max(q,0)| + min(max(q.x,max(q.y,q.z)), 0)`，对齐 `Csg::Box`）。
///
/// 【V2.2 修复】球分支曾写 `return abs(p - c).x - r`（只取 x 分量），那是**沿 x 的
/// 无限大板**而不是球——`abs(p-c)-r` 的三个分量只有在「仅一个坐标有差」时才相等，
/// 故"任取一分量即 |p-c|"的理由不成立。真机实测暴露后改为标准球 SDF。
fn gi_prim_dist(prim: GiPrim, p: vec3<f32>) -> f32 {
  if (prim.kind == 1u) {
    // 轴对齐盒：r 为半尺寸（用同一 radius 字段承载，测试里盒取等半尺寸）。
    let q = abs(p - prim.center) - prim.radius;
    let qc = vec3<f32>(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0));
    let outside = length(qc);
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside;
  }
  // 球：SDF = |p - c| - radius。
  return length(p - prim.center) - prim.radius;
}

/// 场景距离场：全部图元的 Union = min（对齐 `Csg::Union`）。
///
/// 空场（prim_count == 0）恒返回一个远大于任何场景尺度的大正距离（对齐 CPU
/// `EMPTY_FAR`），使 sphere tracing 永不命中（空场景全可见）。
///
/// 【越界守卫】`prim_count` 来自 uniform（宿主写入），若被误写（例如读到相邻的
/// `point_count`）就会越过 `prims` 的编译期容量。WGSL 对定长数组的越界索引是
/// 动态错误（行为未定义，部分后端夹紧到末元素）：一旦越界读到容量填充图元，
/// 整个场景可能被判成"起点即在体内"→ 全方向遮挡 → 输出整块零值。故此处把上界
/// **clamp 到编译期容量** `GI_MAX_PRIMS`。这是纯防御：正常路径 `prim_count<=容量`，
/// 数值一分不变。
fn gi_scene_dist(p: vec3<f32>, prim_count: u32) -> f32 {
  let n = min(prim_count, GI_MAX_PRIMS);
  var d = 1.0e6;
  var i = 0u;
  loop {
    if (i >= n) { break; }
    d = min(d, gi_prim_dist(gi_params.prims[i], p));
    i = i + 1u;
  }
  return d;
}

/// sphere tracing：沿 `rd`（须单位向量）从 `ro` 步进，命中返回 true。
///
/// 与 `prism-geometry::sdf::sphere_trace` 逐式一致：命中阈值 `hit_eps`（含起点命中），
/// 步长 `d * safety`，`t > t_max` 或步数耗尽即未命中。退化方向（零向量 / 非有限）
/// 视为未命中（不 panic 的同构）。
fn sphere_trace_sdf(ro: vec3<f32>, rd: vec3<f32>, prim_count: u32) -> bool {
  if (!(rd.x == rd.x) || !(rd.y == rd.y) || !(rd.z == rd.z)) { return false; } // NaN 方向
  let rl = length(rd);
  if (rl < 0.5) { return false; } // 退化方向 → 未命中（不 panic）
  let dir = rd * (1.0 / rl);
  var t = 0.0;
  var i = 0u;
  loop {
    if (i >= gi_params.max_steps) { break; }
    let d = gi_scene_dist(ro + dir * t, prim_count);
    if (d <= gi_params.hit_eps) { return true; }
    t = t + d * gi_params.safety;
    if (t > gi_params.t_max) { return false; }
    i = i + 1u;
  }
  return false;
}

// ---------------------------------------------------------------------------
// GI 探针：逐方向采样场景辐亮度（可见 → 环境辐亮度，被遮挡 → 0）
// ---------------------------------------------------------------------------

/// 半球采样方向（**立体角均匀**：cosθ 均匀 + 方位角等间距网格；禁 libm）。
///
/// `idx = d + 0.5`；`z = idx/N`（cosθ 均匀 ∈ (0,1)）；`r = sqrt(1 - z²)`（半径）；
/// `phi = idx·黄金角`（避免共振）。圆盘坐标 `(r·cosφ, r·sinφ)`，抬升 `z`（+Z 为极）。
///
/// 【为何立体角均匀】`dΩ = dcosθ·dφ`，cosθ 均匀 × φ 均匀 ⇒ 每样本等立体角
/// `2π/N`。这正是 SH 投影 `L_lm = Σ L·Y_lm·dΩ` 的无偏黎曼求积条件：常数场下
/// `Σ Y00·dΩ = Y00·2π`（精确），且 l>0 带积分为 0。若改用 Fibonacci 圆盘半径
/// 映射（`z=sqrt(1-r²)`），密度会向极点聚集、常数场投影出现 ~30% 求积偏差——
/// 故本实现取立体角均匀方案。三角走 `cos_sin`（rt_common 确定性件）。
fn gi_hemi_dir(d: u32, n: u32) -> vec3<f32> {
  let idx = f32(d) + 0.5;
  let z = idx / f32(n);                     // cosθ 均匀 ∈ (0,1)
  let r = sqrt(max(0.0, 1.0 - z * z));      // 半径
  let phi = idx * 2.3999632;                // 黄金角（低差异方位）
  let cs = cos_sin(phi);
  return vec3<f32>(r * cs.x, r * cs.y, z);
}

/// 对第 `i` 条探针做半球采样：每个方向沿其法线半球投射一条 SDF 遮挡射线。
///
/// 可见方向写入 `env_radiance`（环境常量近似命中点辐亮度，对齐 CPU
/// `one_bounce_indirect` 的"环境常量 + SDF 遮挡"近似），被遮挡写 0。
/// 起点沿法线外推 `SURFACE_EPS` 防自遮挡（对齐 CPU `o = p + n*1e-3`）。
@compute @workgroup_size(64)
fn gi_probe_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // 越界守卫：上界 clamp 到 storage 的真实元素数（`arrayLength`），防止
  // `probe_count` 被误读成更大的相邻字段时越界写 `gi_probes`。正常路径不变。
  let n_probes = min(gi_params.probe_count, arrayLength(&gi_probes));
  if (i >= n_probes) { return; }
  var probe = gi_probes[i];
  let nlen = length(probe.normal);
  if (nlen < 0.5) { return; } // 退化法线：不采样（C18 同构）
  let n = probe.normal * (1.0 / nlen);
  let m = onb(n);
  // onb 返回 mat3x3(t, b, n)（列主序）→ t = m[0], b = m[1], n = m[2]。
  let t = m[0];
  let b = m[1];
  let n_dirs = min(gi_params.n_dirs, GI_DIRS);
  var d = 0u;
  loop {
    if (d >= n_dirs) { break; }
    let local = gi_hemi_dir(d, n_dirs);
    let world = normalize(t * local.x + b * local.y + n * local.z);
    let ro = probe.pos + n * 1e-3;
    var rad = vec3<f32>(0.0, 0.0, 0.0);
    if (!sphere_trace_sdf(ro, world, gi_params.prim_count)) {
      rad = gi_params.env_radiance; // 未遮挡 → 环境辐亮度
    }
    probe.samples[d] = rad;
    d = d + 1u;
  }
  gi_probes[i] = probe;
}

// ---------------------------------------------------------------------------
// SH-L2 投影（对齐 sh.rs：投影原始辐亮度，不乘 cos）
// ---------------------------------------------------------------------------

/// SH-L2 的 9 个实数基函数在单位方向 `d` 上的取值（固定序，与 `sh.rs::sh_basis_l2` 一致）：
/// `[Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]`。
fn gi_sh_basis_l2(d: vec3<f32>) -> array<f32, 9> {
  return array<f32, 9>(
    0.28209479,                          // Y_0^0  = 0.5·sqrt(1/π)
    0.48860251 * d.y,                    // Y_1^-1 = 0.5·sqrt(3/π)·y
    0.48860251 * d.z,                    // Y_1^0  = 0.5·sqrt(3/π)·z
    0.48860251 * d.x,                    // Y_1^1  = 0.5·sqrt(3/π)·x
    1.09254846 * d.x * d.y,              // Y_2^-2 = 0.25·sqrt(15/π)·xy
    1.09254846 * d.y * d.z,              // Y_2^-1 = 0.25·sqrt(15/π)·yz
    0.31539157 * (3.0 * d.z * d.z - 1.0),// Y_2^0  = 0.25·sqrt(5/π)·(3z²−1)
    1.09254846 * d.x * d.z,              // Y_2^1  = 0.25·sqrt(15/π)·xz
    0.54627423 * (d.x * d.x - d.y * d.y),// Y_2^2  = 0.25·sqrt(15/π)·(x²−y²)
  );
}

/// 投影第 `i` 条探针的采样辐亮度到 SH-L2（RGB 各一套 9 系数，写回 `probe.sh`）。
///
/// 约定（对齐 `sh.rs`）：**投影原始辐亮度**，余弦瓣在重建时才乘。半球均匀采样的
/// 每样本立体角 `dΩ = 2π/n_dirs`。累加序固定（方向序 × 系数序）→ 确定性。
@compute @workgroup_size(64)
fn gi_sh_project(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // 越界守卫（同 `gi_probe_trace`）：上界 clamp 到 storage 元素数。
  let n_probes = min(gi_params.probe_count, arrayLength(&gi_probes));
  if (i >= n_probes) { return; }
  var probe = gi_probes[i];
  let nlen = length(probe.normal);
  if (nlen < 0.5) { return; }
  let n = probe.normal * (1.0 / nlen);
  let m = onb(n);
  let t = m[0];
  let b = m[1];
  let n_dirs = min(gi_params.n_dirs, GI_DIRS);
  if (n_dirs == 0u) { return; }
  let d_omega = 6.283185307179586 / f32(n_dirs); // 2π / n
  var acc_r = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var acc_g = acc_r;
  var acc_b = acc_r;
  var d = 0u;
  loop {
    if (d >= n_dirs) { break; }
    let local = gi_hemi_dir(d, n_dirs);
    let world = normalize(t * local.x + b * local.y + n * local.z);
    let yb = gi_sh_basis_l2(world);
    let rad = probe.samples[d];
    var k = 0u;
    loop {
      if (k >= 9u) { break; }
      let w = yb[k] * d_omega;
      acc_r[k] = acc_r[k] + rad.x * w;
      acc_g[k] = acc_g[k] + rad.y * w;
      acc_b[k] = acc_b[k] + rad.z * w;
      k = k + 1u;
    }
    d = d + 1u;
  }
  var k2 = 0u;
  loop {
    if (k2 >= 9u) { break; }
    probe.sh[k2] = vec3<f32>(acc_r[k2], acc_g[k2], acc_b[k2]);
    k2 = k2 + 1u;
  }
  gi_probes[i] = probe;
}

// ---------------------------------------------------------------------------
// 探针网格插值（从 SH 系数重建任意点的辐照度）
// ---------------------------------------------------------------------------

/// 余弦瓣 `max(0, cosθ)` 的 SH 卷积系数 `Â_l`（与 `sh.rs::cosine_lobe` 一致）：
/// `Â_0 = π`、`Â_1 = 2π/3`、`Â_2 = π/4`。
fn gi_cosine_lobe(band: u32) -> f32 {
  if (band == 0u) { return 3.141592653589793; }
  if (band == 1u) { return 2.0943951023931953; } // 2π/3
  return 0.7853981633974483;                     // π/4
}

/// 基函数索引 → 所属阶 `l`（0..9：l=0 占 1，l=1 占 3，l=2 占 5）。
fn gi_band_of(index: u32) -> u32 {
  if (index == 0u) { return 0u; }
  if (index <= 3u) { return 1u; }
  return 2u;
}

/// 从最近探针 SH 系数重建第 `i` 个查询点在方向 `dir` 上的**辐照度**
/// `E(dir) = Σ_lm Â_l · L_lm · Y_lm(dir)`（对齐 `sh.rs::reconstruct`）。
///
/// 累加序固定（系数序 0..9）→ 确定性。
fn gi_reconstruct(probe: GiProbe, dir: vec3<f32>) -> vec3<f32> {
  let yb = gi_sh_basis_l2(dir);
  var e = vec3<f32>(0.0, 0.0, 0.0);
  var k = 0u;
  loop {
    if (k >= 9u) { break; }
    e = e + probe.sh[k] * (gi_cosine_lobe(gi_band_of(k)) * yb[k]);
    k = k + 1u;
  }
  return e;
}

/// 从探针网格插值出任意点的辐照度。
///
/// **插值策略（诚实说明）**：探针网格稀疏时按"最近探针"取 SH 系数（最近邻插值），
/// 不做 SH 系数线性混合——最近邻是零成本且确定性最强的选择，与 CPU 侧
/// `ShIrradiance::lerp` 的"线性混合"相比是**更粗的近似**（lerp 在 abi.rs 的 CPU
/// 参考里有单测覆盖，但 GPU 路径此处只做最近邻以保持 prefix 绑定方案与最小实现）。
///
/// 每个查询点沿其"上方"(+Z) 重建 E 并按 Lambert 转辐亮度 `albedo·E·(1/π)`
/// （对齐 `sh.rs::sh_irradiance_to_radiance`）。
@compute @workgroup_size(64)
fn gi_probe_interpolate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // 越界守卫（同 `gi_probe_trace`）：query 点与最近探针搜索的上界都 clamp 到
  // 各自 storage 的真实元素数。正常路径 `count<=容量`，数值一分不变。
  let n_points = min(gi_params.point_count, arrayLength(&gi_points));
  let n_probes = min(gi_params.probe_count, arrayLength(&gi_probes));
  if (i >= n_points) { return; }
  if (n_probes == 0u) { return; }
  var pt = gi_points[i];
  // 最近探针（平方距离最小；tie-break 为索引序 → 确定性）。
  var best = 0u;
  var best_d2 = 3.4028234e38;
  var j = 0u;
  loop {
    if (j >= n_probes) { break; }
    let diff = gi_probes[j].pos - pt.pos;
    let d2 = vdot(diff, diff);
    if (d2 < best_d2) { best_d2 = d2; best = j; }
    j = j + 1u;
  }
  let probe = gi_probes[best];
  // 重建方向：查询点朝上的辐照度方向（+Z）。
  let dir = vec3<f32>(0.0, 0.0, 1.0);
  let e = gi_reconstruct(probe, dir);
  // Lambert：L_o = albedo · E · (1/π)。
  let lo = gi_params.albedo * e * (1.0 / 3.141592653589793);
  pt.irradiance = lo;
  gi_points[i] = pt;
}
