// prism-rt wavefront 队列 kernel —— 生成/着色/扩展/连接 四阶段。
//
// # 设计（D2：wavefront 队列 kernel——生成/着色/扩展/连接 四阶段）
// 活路径按状态分组批量处理，避免分支发散：
//   ① 生成（rt_generate）：相机射线 → 队列项（对齐 CpuRef 相机段）。
//   ② 着色（rt_shade）：NEE 直接光 + 发光命中 + MNEE 焦散（对齐 integrator 主循环
//      的 NEE 段与发光段）。
//   ③ 连接（rt_connect）：MIS 权重融合（power heuristic，对齐 light.rs）。
//   ④ 扩展（rt_extend）：BSDF/相位抽样续弹或终止（RR 对齐 integrator rr_survive）。
//
// # 队列数据结构
// 每条路径一槽（GPU 版：路径索引 = 全局线程号；CPU 仿真版 prism-rt/wavefront.rs
// 逐条推进以证明估计量等价）。队列项 = PathSlot（射线/吞吐/亮度/状态/深度）。

//!include rt_common.wgsl
//!include rt_shade.wgsl
//!include rt_scene.wgsl
//!include rt_env.wgsl
struct PathSlot {
  origin: vec3<f32>,
  dir: vec3<f32>,
  time: f32,
  throughput: vec3<f32>,
  radiance: vec3<f32>,
  prev_pdf_dir: f32,
  state: u32,           // ST_ACTIVE / ST_TERMINATED
  bounce: u32,          // 已完成事件数
  flags: u32,           // bit0 = prev_specular, bit1 = aux_steps 有效
  aux_steps: u32,
  placeholder: vec3<f32>,
}

struct WFParams {
  width: u32,
  height: u32,
  spp: u32,
  max_bounces: u32,
  clamp_indirect_on: u32,
  clamp_indirect_threshold: f32,
  nee_only: u32,
  global_seed: u32,
  node_id: u32,
  frame: u32,
  // 0 = 路径追踪（默认，W2 语义一行未改）；1 = CpuRef 兼容模式（W1 判据用）。
  mode: u32,
  // 与 megakernel 的 aesthetic 三个开关逐字段同义（兼容模式才读）。
  teal_orange: u32,
  contrast: f32,
  contrast_on: u32,
  filmic: u32,
  _pad1: u32,
}

// group(0) = wavefront 队列局部资源（slots / params / accum）。**刻意把 camera 移出
// 本组**：`generate` 静态引用 {slots, params, camera} 而 `accumulate` 引用
// {slots, params, accum}，二者并集在**同一 group 内**无法同时满足"每个 entry 的
// 绑定集是连续前缀 {0..n-1}"（宿主把数组第 i 个句柄硬绑到 binding:i，见
// `abi.rs` 的 STATE_STRIDE_BYTES 文档与 `web/index.html` 的 prism_create_bind_group）。
// 把 camera 拆到独立的 group(3)（单绑定 @binding(0)）后，本组三个 entry
// （generate/shade/extend）都只引用 {0,1} 前缀，accumulate 引用 {0,1,2}，前缀性质
// 全部成立。
@group(0) @binding(0) var<storage, read_write> slots: array<PathSlot>;
@group(0) @binding(1) var<uniform> params: WFParams;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;

// ---- 生成阶段：相机射线（对齐 camera.rs::cast_ray；fov 由 params 派生）----
// 视点/朝向/焦距由场景 uniform（camera_cb）提供；此处以占位签名声明。
struct Camera {
  eye: vec3<f32>,
  _p0: f32,
  forward: vec3<f32>,
  _p1: f32,
  right: vec3<f32>,
  _p2: f32,
  up: vec3<f32>,
  _p3: f32,
  tan_half_fov: f32,
  aperture_radius: f32,
  focus_distance: f32,
  shutter_open: f32,
  shutter_close: f32,
  // 原结构体尾部填充（offset 84）：CpuRef 兼容模式需要 `cam.aspect` 原值
  // （它可能不等于 width/height）。加在填充位上，uniform 仍是 96 字节。
  aspect: f32,
}
// 独立 group(3)：只被 `generate`（经 cast_ray）引用，单绑定前缀 {0}。
@group(3) @binding(0) var<uniform> camera: Camera;

/// 路径模式：队列四阶段照旧（W2 语义）。
const WF_MODE_PATH: u32 = 0u;
/// CpuRef 兼容模式：**同一估计量**的 wavefront 组织（W1 判据用）。
///
/// 这不是"另一条近似路径"——它是 `CpuRefBackend::render_scene` 的逐式移植：
/// 像素中心射线、球面最近命中、常数环境光 0.08、方向光单弹射、aesthetic
/// 三段、`clamp01 → *255 → floor(+0.5)` 量化。写成 5 个 stage 是为了证明
/// **派发器/绑定/队列语义**本身正确（与 megakernel 逐字节对拍），而不是
/// 为了给产品再添一个渲染器：产品路径仍只走 `WF_MODE_PATH`。
const WF_MODE_CPUREF: u32 = 1u;

/// 与 `Vec3::dot` 同一求值树（`(x*x + y*y) + z*z`）。
fn vdot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// 与 `Vec3::normalize` 同一求值树（逐分量除以 `sqrt(dot)`；退化返回零）。
fn vnorm(v: vec3<f32>) -> vec3<f32> {
  let len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (len < 1e-8) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return vec3<f32>(v.x / len, v.y / len, v.z / len);
}

// ---- 环境 CDF（A2 / W2）：挂在场景组 rt_scene 的尾部 ----
//
// # 为什么不是新开一个 group
//
// WebGPU 的 `maxBindGroups` 默认 4 ⇒ group 索引只能是 0..3，而 shade 已经用到
// {0 slots/params, 1 sobol+scene_buf, 2 场景资源} 三组；再开 group(4) 直接越界。
// 挂在 **group(2) binding 5** 满足宿主 ABI 的唯一硬约束——"每个 entry 的绑定集
// 必须是连续前缀 {0..n-1}"：`shade` 引用 {0,1,2}（原）+ 本绑定 = {0..5} 仍是前缀；
// `extend`/`generate` 引用 {0,1,2} 与 {0,1} ⇒ 各自仍是更短的前缀。
//
// # 语义（与 CPU 权威同一份）
//
// 函数体来自 `rt_env.wgsl`（`env_radiance` / `env_pdf` / 反演 / 双线性 / pdf
// 全部那一条实现），本文件不另写采样。绑定号不同不影响"同源"——同源指的是
// 函数体与 CDF 语义，而不是槽位号。
@group(2) @binding(5) var<storage, read> env_blob: array<f32>;

/// 环境图是否可用：blob 至少装得下头部 **且** 宽高非 0。
///
/// 空环境（宿主给 1 元占位 buffer）⇒ 头部都读不到 ⇒ 一律走 `Light::Ambient`
/// 分支（与 CPU 的 `scene.env() == None` 同判据，不是"读到垃圾再兜"）。
fn env_present() -> bool {
  if (arrayLength(&env_blob) < ENV_HDR) {
    return false;
  }
  return env_w() > 0u && env_h() > 0u;
}

fn cast_ray(pixel: vec2<u32>, sample: u32, si: u32) -> array<vec3<f32>, 2> {
  // 像素抖动（槽 3,4）→ 焦平面 → 光圈圆盘（槽 0,1）→ 快门（槽 2）。
  let jx = sample_1d(si, DIM_PIXEL);
  let jy = sample_1d(si, DIM_PIXEL + 1u);
  let u = ((f32(pixel.x) + jx) / f32(params.width)) * 2.0 - 1.0;
  let v = 1.0 - ((f32(pixel.y) + jy) / f32(params.height)) * 2.0;
  let aspect = f32(params.width) / f32(params.height);
  let d = camera.forward
    + camera.right * (u * camera.tan_half_fov * aspect)
    + camera.up * (v * camera.tan_half_fov);
  let time = camera.shutter_open + sample_1d(si, DIM_SHUTTER) * (camera.shutter_close - camera.shutter_open);
  if (camera.aperture_radius == 0.0) {
    return array<vec3<f32>, 2>(camera.eye, normalize(d));
  }
  let du = normalize(d);
  let t = camera.focus_distance / vdot(du, camera.forward);
  let focal = camera.eye + du * t;
  let lens = sample_2d(si, DIM_LENS);
  let ld = concentric_disk(lens.x, lens.y);
  let origin = camera.eye + camera.right * (ld.x * camera.aperture_radius)
    + camera.up * (ld.y * camera.aperture_radius);
  return array<vec3<f32>, 2>(origin, normalize(focal - origin));
}

// 相机段维度槽（对齐 camera.rs）。
const DIM_LENS: u32 = 0u;
const DIM_SHUTTER: u32 = 2u;
const DIM_PIXEL: u32 = 3u;

// ---- 生成 stage ----
@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  let px = idx % params.width;
  let py = (idx / params.width) % params.height;
  let sample = idx / (params.width * params.height);
  var s: PathSlot;
  if (params.mode == WF_MODE_CPUREF) {
    // CpuRef 兼容：像素中心射线（无抖动 / 无 DoF / 无快门）。
    let nx = (2.0 * (f32(px) + 0.5) / f32(params.width) - 1.0)
      * camera.tan_half_fov * camera.aspect;
    let ny = (1.0 - 2.0 * (f32(py) + 0.5) / f32(params.height)) * camera.tan_half_fov;
    // `nx` 含 aspect、`ny` 不含——与 `CpuRefBackend::render_scene_impl` 逐式相同。
    let d = camera.forward + camera.right * nx + camera.up * ny;
    s.origin = camera.eye;
    s.dir = vnorm(d);
    s.time = 0.0;
    s.throughput = vec3<f32>(1.0, 1.0, 1.0);
    s.radiance = vec3<f32>(0.0, 0.0, 0.0);
    s.prev_pdf_dir = 0.0;
    s.state = ST_ACTIVE;
    s.bounce = 0u;
    s.flags = 1u;
    s.aux_steps = 0u;
    slots[idx] = s;
    return;
  }
  let si = sample_index(px, py, sample, 0u);
  let ray = cast_ray(vec2<u32>(px, py), sample, si);
  s.origin = ray[0];
  s.dir = ray[1];
  s.time = 0.0;
  s.throughput = vec3<f32>(1.0, 1.0, 1.0);
  s.radiance = vec3<f32>(0.0, 0.0, 0.0);
  s.prev_pdf_dir = 0.0;
  s.state = ST_ACTIVE;
  s.bounce = 0u;
  s.flags = 1u; // prev_specular = true（相机段视作镜面链起点）
  s.aux_steps = 0u;
  slots[idx] = s;
}

// ---- 着色 stage：NEE + 发光命中 + MNEE（对齐 integrator 主循环）----

/// 兼容模式的 aesthetic 链（逐式移植 `crate::apply_aesthetic`）。
///
/// 顺序与常量逐字固定：teal-orange → contrast → Reinhard filmic → clamp01。
/// 任何"约等于"都会让 W1 的逐字节对拍变成统计对拍。
fn aesthetic_compat(c0: vec3<f32>) -> vec3<f32> {
  var c = c0;
  if (params.teal_orange != 0u) {
    let luma = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let shadow = vec3<f32>(0.10, 0.35, 0.45) * c * (1.0 - luma) * 0.5;
    let high = vec3<f32>(1.0, 0.7, 0.3) * c * luma * 0.5;
    c = c * 0.6 + shadow + high;
  }
  if (params.contrast_on != 0u) {
    c = (c - vec3<f32>(0.5, 0.5, 0.5)) * params.contrast
      + vec3<f32>(0.5, 0.5, 0.5);
  }
  if (params.filmic != 0u) {
    c = c / (c + vec3<f32>(1.0, 1.0, 1.0));
  }
  return vec3<f32>(
    clampf(c.x, 0.0, 1.0),
    clampf(c.y, 0.0, 1.0),
    clampf(c.z, 0.0, 1.0));
}

/// CpuRef 兼容模式的一段式着色（`CpuRefBackend::render_scene_impl` 的逐式移植）。
///
/// 只消费 `spheres` / `lights` / `camera` 与 aesthetic 四个开关；**不读**
/// `tris`/`materials`/`env_blob`。写出的是**已量化前的线性颜色**，量化在
/// `accumulate` 里做（与 megakernel 的 `quantize` 同一处、同一式）。
fn shade_compat(idx: u32) {
  var s = slots[idx];
  if (s.state != ST_ACTIVE) { return; }
  // ---- 球面最近命中（`CpuRefBackend` 同式：`t = -b - sqrt(disc)`）----
  let nspheres = min(scene_params.sphere_count, arrayLength(&spheres) / 8u);
  var closest: f32 = 1e30;
  var hit_idx: i32 = -1;
  var i = 0u;
  loop {
    if (i >= nspheres) { break; }
    let b0 = i * 8u;
    let center = vec3<f32>(spheres[b0], spheres[b0 + 1u], spheres[b0 + 2u]);
    let radius = spheres[b0 + 3u];
    let oc = s.origin - center;
    let b = vdot3(oc, s.dir);
    let c = vdot3(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc >= 0.0) {
      let t = -b - sqrt(disc);
      if (t > 0.0 && (hit_idx < 0 || t < closest)) {
        closest = t;
        hit_idx = i32(i);
      }
    }
    i = i + 1u;
  }

  var color = vec3<f32>(0.04, 0.05, 0.08); // 背景（与 CpuRef 同值）
  if (hit_idx >= 0) {
    let hb = u32(hit_idx) * 8u;
    let center = vec3<f32>(spheres[hb], spheres[hb + 1u], spheres[hb + 2u]);
    let albedo = vec3<f32>(spheres[hb + 4u], spheres[hb + 5u], spheres[hb + 6u]);
    let p = s.origin + s.dir * closest;
    let n = vnorm(p - center);
    let ambient = vec3<f32>(0.08, 0.08, 0.08);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let nlights = min(scene_params.light_count, arrayLength(&dir_lights) / 8u);
    var li = 0u;
    loop {
      if (li >= nlights) { break; }
      // 兼容模式的灯布局 = megakernel 的平铺 8 f32：
      // dir@0..2 / color@3..5 / intensity@6 / pad@7。
      let lb = li * 8u;
      let ldir = vec3<f32>(dir_lights[lb], dir_lights[lb + 1u], dir_lights[lb + 2u]);
      let lcolor = vec3<f32>(dir_lights[lb + 3u], dir_lights[lb + 4u], dir_lights[lb + 5u]);
      let lint = dir_lights[lb + 6u];
      let diff = max(vdot3(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + albedo * term;
      li = li + 1u;
    }
    color = acc;
  }
  s.radiance = aesthetic_compat(color);
  s.state = ST_TERMINATED;
  slots[idx] = s;
}

/// 逃逸射线看到的环境贡献（**BSDF 策略 + MIS**，逐句移植 `integrator.rs`
/// 逃逸段；`rt_wavefront` 与 CPU 权威的唯一差别是"谁来推进队列"）。
///
/// * `prev_specular`（相机段 / 镜面链）⇒ 权重 1（镜面方向是 δ 分布，
///   CDF 策略对它测度为零）；
/// * 否则 `w = 1 − power_heuristic(pdf_env, pdf_bsdf)`，与 `rt_scene::
///   nee_mis_weight` 的 Environment 臂逐位互补（`w_nee + w_bsdf ≡ 1`）；
/// * `nee_only && ev > 1` ⇒ `w = 0`（能量对照口径，CPU 侧同一条）。
fn env_escape(throughput: vec3<f32>, dir: vec3<f32>, prev_pdf_dir: f32,
              prev_specular: bool, nee_only: bool, ev: u32) -> vec3<f32> {
  if (!env_present()) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let r = env_radiance(dir);
  // 与 CPU 同判据（正值判据，不是"非零"：负辐亮度不进累加）。
  if (!(r.x > 0.0 || r.y > 0.0 || r.z > 0.0)) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  var w = 0.0;
  if (nee_only && ev > 1u) {
    w = 0.0;
  } else if (prev_specular) {
    w = 1.0;
  } else {
    let p_env = env_pdf(dir);
    w = 1.0 - power_heuristic(p_env, prev_pdf_dir);
  }
  return clamp_indirect(hadamard(throughput, r) * w, ev);
}

@compute @workgroup_size(64)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  if (params.mode == WF_MODE_CPUREF) {
    shade_compat(idx);
    return;
  }
  var s = slots[idx];
  if (s.state != ST_ACTIVE) { return; }
  let ev = s.bounce + 1u;
  let aux_dim = bounce_dim(ev, SLOT_AUX);
  let hit = scene_intersect(s.origin, s.dir, MISS_T, s.aux_steps, aux_dim);
  // 介质段 + 体 NEE（对齐 integrator medium 分支）——见 rt_media.wgsl，本骨架挂点位。
  if (hit.t < 1e29) {
    let p = s.origin + s.dir * hit.t;
    let n = hit.normal;
    // 发光命中（MIS 由 connect stage 融合）。
    let em = emissive_radiance(hit.prim);
    if (any(em > vec3<f32>(0.0, 0.0, 0.0))) {
      let w = emissive_weight(s.flags, s.prev_pdf_dir, hit.prim, s.origin);
      s.radiance = s.radiance + clamp_indirect(hadamard(s.throughput, em) * w, ev);
    }
    // NEE 直接光（反射域；材质取命中图元 `hit.prim`）。
    s.radiance = s.radiance + nee_surface(p, n, s.dir, hit.prim, s.throughput, ev, s.aux_steps);
    // MNEE 焦散（k=2 球面折射链；对齐 caustics.rs::mnee_refraction）。
    if (params.nee_only == 0u) {
      s.radiance = s.radiance + mnee_caustics(p, n, s.dir, s.throughput, ev);
    }
  } else {
    // 逃逸：环境背景。
    //
    // A2 / W2 起走**环境图**（CDF 同源，见 `env_escape`）；`Light::Ambient`
    // 与 `Light::Environment` 在 CPU 侧互斥（`validate_layout` 拦），故两条
    // 分支不会双计——此处照抄同一互斥语义：有环境图时 ambient 恒为 0。
    s.radiance = s.radiance + env_escape(
      s.throughput,
      s.dir,
      s.prev_pdf_dir,
      (s.flags & 1u) != 0u,
      params.nee_only != 0u,
      ev,
    );
    s.radiance = s.radiance + clamp_indirect(hadamard(s.throughput, ambient_background()), ev);
    s.state = ST_TERMINATED;
  }
  slots[idx] = s;
}

// ---- 连接 stage：MIS 权重融合（power heuristic，对齐 light.rs::power_heuristic）----
@compute @workgroup_size(64)
fn connect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  var s = slots[idx];
  if (s.state != ST_ACTIVE) { return; }
  // MIS 权重已在 shade 内按备择策略 pdf 计算（与 integrator 一致：NEE 权重
  // nee_mis_weight、发光命中权重 power_heuristic）；本 stage 保留为队列阶段
  // 边界（GPU 实现上可与 shade 合并 dispatch，语义不变）。
  slots[idx] = s;
}

// ---- 扩展 stage：BSDF/相位抽样续弹或终止（对齐 integrator 续弹段）----
@compute @workgroup_size(64)
fn extend(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  var s = slots[idx];
  if (s.state != ST_ACTIVE) { return; }
  let ev = s.bounce + 1u;
  if (ev >= params.max_bounces) { s.state = ST_TERMINATED; slots[idx] = s; return; }
  let hit = scene_intersect(s.origin, s.dir, MISS_T, s.aux_steps, bounce_dim(ev, SLOT_AUX));
  if (hit.t >= 1e29) { s.state = ST_TERMINATED; slots[idx] = s; return; }
  let p = s.origin + s.dir * hit.t;
  let n = hit.normal;
  let bs = bsdf_sample(hit.prim, s.dir, n, ev);
  if (bs.pdf < 0.0) { s.state = ST_TERMINATED; slots[idx] = s; return; }
  s.throughput = hadamard(s.throughput, bs.weight);
  s.prev_pdf_dir = bs.pdf;
  // RR（对齐 integrator rr_survive：ev+1 ≥ 3 时按吞吐存活）。
  if (should_survive(s.throughput, params.global_seed, idx, ev + 1u) == false) {
    s.state = ST_TERMINATED; slots[idx] = s; return;
  }
  s.throughput = s.throughput * rr_scale(s.throughput);
  s.origin = p + n * SURFACE_EPS;
  s.dir = bs.dir;
  s.bounce = s.bounce + 1u;
  slots[idx] = s;
}

// ---- 累积 stage：spp 均值（对齐 accum.rs::mean_samples 固定形状树语义）----

/// `clamp01 → *255 → floor(x+0.5)`（与 megakernel 的 `quantize` 逐式相同）。
fn quantize_compat(x: f32) -> u32 {
  let clamped = clampf(x, 0.0, 1.0);
  let scaled = clamped * 255.0;
  let shifted = scaled + 0.5;
  return u32(floor(shifted));
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let npix = params.width * params.height;
  if (idx >= npix) { return; }
  let base = idx * params.spp;
  if (params.mode == WF_MODE_CPUREF) {
    // 兼容模式：每样本值相同（无采样），直接取首个 slot，量化成 RGBA8 位型。
    // 用 f32 承载 u32 位型是为了不改 `accum` 的 `array<vec4<f32>>` 声明
    // （路径模式仍按 f32 HDR 读回）——宿主读回时按小端 u32 解释第一个分量。
    let c = slots[base].radiance;
    let qr = quantize_compat(c.x);
    let qg = quantize_compat(c.y);
    let qb = quantize_compat(c.z);
    let packed = (255u << 24u) | (qb << 16u) | (qg << 8u) | qr;
    accum[idx] = vec4<f32>(bitcast<f32>(packed), 0.0, 0.0, 1.0);
    return;
  }
  var sum = vec3<f32>(0.0, 0.0, 0.0);
  var k = 0u;
  while (k < params.spp) {
    sum = sum + slots[base + k].radiance;
    k = k + 1u;
  }
  accum[idx] = vec4<f32>(sum / f32(params.spp), 1.0);
}
