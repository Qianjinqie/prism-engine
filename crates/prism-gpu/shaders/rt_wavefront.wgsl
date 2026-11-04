// prism-rt wavefront 队列 kernel —— 生成/着色/扩展/连接 四阶段。
//
// # 设计（D2：wavefront 队列 kernel——生成/着色/扩展/连接 四阶段）
// 活路径按状态分组批量处理，避免分支发散：
//   ① 生成（rt_generate）：相机射线 → 队列项（对齐 CpuRef 相机段）。
//   ② 着色（rt_shade）：NEE 直接光 + 发光命中（对齐 integrator 主循环
//      的 NEE 段与发光段）。
//   ③ 连接（rt_connect）：MIS 权重融合（power heuristic，对齐 light.rs）。
//   ④ 扩展（rt_extend）：BSDF/相位抽样续弹或终止（RR 对齐 integrator rr_survive）。
//
// # 队列数据结构
// 每条路径一槽（GPU 版：路径索引 = 全局线程号；CPU 仿真版 prism-rt/wavefront.rs
// 逐条推进以证明估计量等价）。队列项 = PathSlot（射线/吞吐/亮度/状态/深度）。
//
// # T-B（R-1 逐位对拍）路径模式重写
//
// 路径模式（`WF_MODE_PATH`）的 shade/extend/accumulate 三段是
// `prism_rt::wavefront::advance_one` / `accum::mean_samples` 的**逐式移植**：
// 采样流走 `rt_shade.wgsl` 的 `WfStream`（64 位 PCG 仿真 + 真 Joe–Kuo
// Sobol 表，CPU 同表同流）、求交走 `rt_scene.wgsl::scene_intersect`
// （Möller–Trumbore 与 `tri_ray_hit` 同树）、RR/钳位走本文件的
// `wf_rr` / `wf_clamp_indirect`（与 integrator 同式）。判据范围 =
// **Lambert 材质 + Sphere 灯（不注册为几何）+ 针孔相机 + 无环境图/介质/
// 折射体** 的场景类；`MAT_PBR` 的 Disney 续弹是方向提案近似，**不进
// R-1 逐位判据**（如实标注）。

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

// 相机段维度槽（对齐 camera.rs）。
const DIM_LENS: u32 = 0u;
const DIM_SHUTTER: u32 = 2u;
const DIM_PIXEL: u32 = 3u;

// ---------------------------------------------------------------------------
// 路径模式公共件（T-B R-1；判对基准 = integrator.rs / wavefront.rs）
// ---------------------------------------------------------------------------

/// 间接贡献钳位（integrator.rs::clamp_indirect 逐式：`Some(t) if ev ≥ 2` ⇒
/// 逐分量 min；None/ev<2 ⇒ 原样）。
fn wf_clamp_indirect(c: vec3<f32>, ev: u32) -> vec3<f32> {
  if (params.clamp_indirect_on != 0u && ev >= 2u) {
    let t = params.clamp_indirect_threshold;
    return vec3<f32>(min(c.x, t), min(c.y, t), min(c.z, t));
  }
  return c;
}

/// RR 存活判定（integrator.rs::rr_survive 逐式）。返回 (存活, 吞吐因子)：
/// `next_ev < 3` 恒存活（因子 1）；否则 `u ≥ q` 判死、存活时吞吐 × 1/q，
/// q = clamp(max(吞吐分量), 0.05, 0.95)，u ← 槽 4（SLOT_RR）。
fn wf_rr(st: WfStream, throughput: vec3<f32>, next_ev: u32) -> vec2<f32> {
  if (next_ev < 3u) { return vec2<f32>(1.0, 1.0); }
  let q = clampf(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
  let u = wf_sample_1d(st, bounce_dim(next_ev, SLOT_RR));
  if (u >= q) { return vec2<f32>(0.0, 1.0); }
  return vec2<f32>(1.0, 1.0 / q);
}

/// 槽位 → (px, py, sample)（槽位布局 = `p*spp + s`，与 accumulate 同一把尺）。
fn wf_pixel_of(idx: u32) -> vec3<u32> {
  let p = idx / params.spp;
  let sample = idx - p * params.spp;
  return vec3<u32>(p % params.width, p / params.width, sample);
}

/// 槽位 → 采样流（generate/shade/extend 三段从同一 idx 纯函数重导出，
/// 不占额外存储；同输入逐位同流）。
fn wf_stream_of(idx: u32) -> WfStream {
  let pxs = wf_pixel_of(idx);
  return wf_stream(params.global_seed, params.frame, params.node_id,
                   pxs.x, pxs.y, pxs.z);
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

fn cast_ray(pixel: vec2<u32>, st: WfStream) -> array<vec3<f32>, 2> {
  // 像素抖动（槽 3,4）→ 焦平面 → 光圈圆盘（槽 0,1）→ 快门（槽 2）。
  // 与 camera.rs::cast_ray 同序同式（维度绝对寻址，抽取顺序不改变取值）。
  let j = wf_sample_2d(st, DIM_PIXEL);
  let u = ((f32(pixel.x) + j.x) / f32(params.width)) * 2.0 - 1.0;
  let v = 1.0 - ((f32(pixel.y) + j.y) / f32(params.height)) * 2.0;
  let aspect = f32(params.width) / f32(params.height);
  let d = camera.forward
    + camera.right * (u * camera.tan_half_fov * aspect)
    + camera.up * (v * camera.tan_half_fov);
  let time = camera.shutter_open + wf_sample_1d(st, DIM_SHUTTER) * (camera.shutter_close - camera.shutter_open);
  if (camera.aperture_radius == 0.0) {
    // 针孔快路径（CPU 同：视点 = eye，dir = d.normalize() → vnorm 同树）。
    return array<vec3<f32>, 2>(camera.eye, vnorm(d));
  }
  let du = vnorm(d);
  let t = camera.focus_distance / vdot(du, camera.forward);
  let focal = camera.eye + du * t;
  let lens = wf_sample_2d(st, DIM_LENS);
  let ld = concentric_disk(lens.x, lens.y);
  let origin = camera.eye + camera.right * (ld.x * camera.aperture_radius)
    + camera.up * (ld.y * camera.aperture_radius);
  return array<vec3<f32>, 2>(origin, vnorm(focal - origin));
}

// ---- 生成 stage ----
@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  // 槽位布局（T-B 修正）：slot = p*spp + s，与 accumulate 的 `base = 像素*spp`
  // 同一把尺。旧式（idx = px + py*w + s*w*h）在 spp>1 时两段错位——W3/W2 均用
  // spp=1 故未暴露；这是判据缺陷，不是画质问题。
  let p = idx / params.spp;
  let sample = idx - p * params.spp;
  let px = p % params.width;
  let py = p / params.width;
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
  let st = wf_stream(params.global_seed, params.frame, params.node_id, px, py, sample);
  let ray = cast_ray(vec2<u32>(px, py), st);
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

// ---- 着色 stage：NEE + 发光命中（对齐 integrator 主循环）----

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
/// * 否则 `w = 1 − power_heuristic(pdf_env, pdf_bsdf)`，与 Sphere 灯 NEE 的
///   power heuristic 逐位互补（`w_nee + w_bsdf ≡ 1`）；
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
  return wf_clamp_indirect(hadamard(throughput, r) * w, ev);
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
  let hit = scene_intersect(s.origin, s.dir, MISS_T);
  if (hit.prim == 0xFFFFFFFFu) {
    // ---- 逃逸（advance_one 的 None 臂逐式）----
    // ① 环境贴图（无 env 时 env_escape 返 0，加 0 不改位型）。
    s.radiance = s.radiance + env_escape(
      s.throughput,
      s.dir,
      s.prev_pdf_dir,
      (s.flags & 1u) != 0u,
      params.nee_only != 0u,
      ev,
    );
    // ② 常数环境光（正值判据与 CPU 同：全零不加）。
    let bg = ambient_background();
    if (bg.x > 0.0 || bg.y > 0.0 || bg.z > 0.0) {
      s.radiance = s.radiance + wf_clamp_indirect(hadamard(s.throughput, bg), ev);
    }
    s.state = ST_TERMINATED;
    slots[idx] = s;
    return;
  }
  let p = s.origin + s.dir * hit.t;
  let n = hit.normal;
  // 侧别翻转（与 integrator 同：from_inside ⇒ nv = −n）。
  let from_inside = vdot(s.dir, n) > 0.0;
  let nv = select(n, -n, from_inside);
  let wo = -s.dir;
  let mat = load_material(hit.prim);

  // ---- 发光命中（MIS；ev==1/镜面链 ⇒ 1）----
  // 路径模式的场景 ABI 不携带球几何 ⇒ `light_pdf_sa` 恒无匹配 ⇒ 走 CPU 的
  // None 臂（权重 1）。发光几何 MIS 属 R-1 场景类之外（如实标注）。
  let em = mat.emissive;
  if (em.x > 0.0 || em.y > 0.0 || em.z > 0.0) {
    var w_em = 1.0;
    if (params.nee_only != 0u && ev > 1u) {
      w_em = 0.0;
    }
    // else：ev==1 / prev_specular ⇒ 1；其余 None 臂 ⇒ 1。恒 1，无须分支赋值。
    s.radiance = s.radiance + wf_clamp_indirect(hadamard(s.throughput, em) * w_em, ev);
  }

  // ---- 反射域 NEE（逐灯；Sphere 立体角锥采样，`light.rs::sample_nee` 逐式）----
  let st = wf_stream_of(idx);
  let uv_l = wf_sample_2d(st, bounce_dim(ev, SLOT_LIGHT));
  let light_count = load_light_count();
  var li = 0u;
  while (li < light_count) {
    let lt = lights[li];
    // Light::Sphere 臂：to_c = center − p；d_c = |to_c|（length = sqrt(dot) 同树）；
    // 点在球内不采样（`d_c <= radius` ⇒ continue）。
    let to_c = lt.position - p;
    let d_c = sqrt(vdot(to_c, to_c));
    if (d_c > lt.radius) {
      let cos_max = sqrt(1.0 - lt.radius * lt.radius / (d_c * d_c));
      let wi = cone_dir(to_c * (1.0 / d_c), cos_max, uv_l.x, uv_l.y);
      let pdf_w = 1.0 / (2.0 * PI * (1.0 - cos_max));
      let cos_l = vdot(nv, wi);
      if (cos_l > 0.0) {
        // surface_brdf：Lambert 解析（R-1 判据范围）；其余走 Disney 近似 eval。
        var f = vec3<f32>(0.0, 0.0, 0.0);
        var alt_pdf = 0.0;
        if (mat.kind == MAT_LAMBERT) {
          // wi·nv = cos_l > 0 已判；余下条件 = wo·nv > 0。
          if (vdot(wo, nv) > 0.0) {
            f = mat.base_color * FRAC_1_PI;
          }
          alt_pdf = cos_l * FRAC_1_PI;
        } else {
          f = eval_bsdf(mat, s.dir, wi, nv);
          alt_pdf = cos_l * FRAC_1_PI;
        }
        let f_cos = f * cos_l;
        let ro = p + nv * SURFACE_EPS;
        // 遮挡距离 = d_c − T_MIN（闭区间；visibility 无介质臂同语义）。
        let vis = shadow_ray(ro, wi, d_c);
        if (vis > 0.0) {
          // Sphere 灯 = 几何发光体 ⇒ power heuristic（nee_mis_weight 同臂）。
          let w = power_heuristic(pdf_w, alt_pdf);
          let c = hadamard(hadamard(s.throughput, f_cos), lt.color)
            * (vis * w / pdf_w);
          s.radiance = s.radiance + wf_clamp_indirect(c, ev);
        }
      }
    }
    li = li + 1u;
  }

  // ---- 环境贴图 NEE（`Light::Environment` 臂，integrator::nee_env_surface 逐式）----
  // 路径模式场景 ABI 不携带灯光种类 ⇒ 环境按"灯表尾部项"处理（与
  // `Scene::add_environment` 把 `Light::Environment` 追加到 `lights` 尾部的
  // 语义一致）：Sphere 灯循环**之后**执行，求和顺序与 CPU 同。复用上面已
  // 消耗的 (uv_l)（CPU 的 (u_l1,u_l2) 同源），**不额外消耗样本流**。
  // 全黑环境：`env_sample` 返 pdf=0（CPU `env.sample()==None` 同判），进不了腿。
  if (env_present()) {
    let se = env_sample(uv_l.x, uv_l.y);
    let cos_e = vdot(nv, se.wi);
    // `cos_l <= 0 || !(pdf_w > 0.0)` 双门（nee_env_surface 同序）。
    if (cos_e > 0.0 && se.pdf > 0.0) {
      // surface_brdf / surface_pdf：Lambert 解析（与上面 Sphere 腿同一形态）。
      var f_e = vec3<f32>(0.0, 0.0, 0.0);
      var alt_pdf_e = 0.0;
      if (mat.kind == MAT_LAMBERT) {
        // surface_brdf 的 Lambert 门 = wo·nv > 0（wi·nv > 0 已由 cos_e 保证）。
        if (vdot(wo, nv) > 0.0) {
          f_e = mat.base_color * FRAC_1_PI;
        }
        // surface_pdf = max(0, wi·nv)/π（cos_e > 0 ⇒ 直接 cos_e）。
        alt_pdf_e = cos_e * FRAC_1_PI;
      } else {
        f_e = eval_bsdf(mat, s.dir, se.wi, nv);
        alt_pdf_e = cos_e * FRAC_1_PI;
      }
      let f_cos_e = f_e * cos_e;
      // radiance 全非正 ⇒ 0（nee_env_surface 同门，顺序在 vis 之前）。
      let r_e = env_radiance(se.wi);
      if (r_e.x > 0.0 || r_e.y > 0.0 || r_e.z > 0.0) {
        let ro_e = p + nv * SURFACE_EPS;
        // CPU 侧 max_t = INFINITY（无介质场景 vis ∈ {0,1}）；本场景几何有界，
        // MISS_T − T_MIN 与 INFINITY 的遮挡判定逐位同值。
        let vis_e = shadow_ray(ro_e, se.wi, MISS_T);
        if (vis_e > 0.0) {
          // nee_mis_weight：Environment 与 Sphere 同臂 = power heuristic。
          let w_e = power_heuristic(se.pdf, alt_pdf_e);
          let c_e = hadamard(hadamard(s.throughput, f_cos_e), r_e)
            * (vis_e * w_e / se.pdf);
          s.radiance = s.radiance + wf_clamp_indirect(c_e, ev);
        }
      }
    }
  }

  // 焦散 MNEE：CPU 只在 `scene.refractors()` 非空时触发；路径模式场景 ABI
  // 不携带折射体 ⇒ 空循环，此处无对应段（逐位等价）。
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
  // power_heuristic、发光命中权重按侧别/深度）；本 stage 保留为队列阶段
  // 边界（GPU 实现上可与 shade 合并 dispatch，语义不变）。
  slots[idx] = s;
}

// ---- 扩展 stage：BSDF 抽样续弹或终止（advance_one 续弹段逐式）----
@compute @workgroup_size(64)
fn extend(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
  var s = slots[idx];
  if (s.state != ST_ACTIVE) { return; }
  let ev = s.bounce + 1u;
  // 求交与 shade 段同一纯函数（同输入逐位同结果）。
  let hit = scene_intersect(s.origin, s.dir, MISS_T);
  if (hit.prim == 0xFFFFFFFFu) {
    // 逃逸已在 shade 终止；活动槽走到这里只可能是段间状态错位——按 CPU
    // 语义（None 臂终止）兜一层。
    s.state = ST_TERMINATED;
    slots[idx] = s;
    return;
  }
  let p = s.origin + s.dir * hit.t;
  let n = hit.normal;
  let mat = load_material(hit.prim);
  if (mat.kind == MAT_EMISSIVE) {
    // material_bsdf = None ⇒ 终止（CPU 同）。
    s.state = ST_TERMINATED;
    slots[idx] = s;
    return;
  }
  let st = wf_stream_of(idx);
  if (mat.kind == MAT_LAMBERT) {
    // Lambert 续弹（advance_one Lambert 臂逐式）：(u1,u2) ← SLOT_BSDF+1、
    // cosine_hemisphere → onb(n)（**用 n 不是 nv**，与 CPU 同）→ 归一化。
    let uv = wf_sample_2d(st, bounce_dim(ev, SLOT_BSDF) + 1u);
    let d_local = cosine_hemisphere(uv.x, uv.y);
    let m = onb(n);
    let dir = vnorm(m[0] * d_local.x + m[1] * d_local.y + n * d_local.z);
    s.throughput = hadamard(s.throughput, mat.base_color);
    s.prev_pdf_dir = d_local.z * FRAC_1_PI;
    s.flags = s.flags & 0xFFFFFFFEu; // prev_specular = false
    let rr = wf_rr(st, s.throughput, ev + 1u);
    if (rr.x == 0.0) {
      s.state = ST_TERMINATED;
      slots[idx] = s;
      return;
    }
    s.throughput = s.throughput * rr.y;
    s.origin = p + n * SURFACE_EPS;
    s.dir = dir;
  } else if (mat.kind == MAT_MIRROR) {
    // Mirror 续弹（advance_one Mirror 臂逐式）：δ 方向、pdf 0、镜面链。
    let dir = s.dir - n * (2.0 * vdot(s.dir, n));
    s.prev_pdf_dir = 0.0;
    s.flags = s.flags | 1u; // prev_specular = true
    let rr = wf_rr(st, s.throughput, ev + 1u);
    if (rr.x == 0.0) {
      s.state = ST_TERMINATED;
      slots[idx] = s;
      return;
    }
    s.throughput = s.throughput * rr.y;
    s.origin = p + n * SURFACE_EPS;
    s.dir = dir;
  } else {
    // Pbr：Disney 方向提案（近似；**不进 R-1 逐位判据**，如实标注）。
    let bs = bsdf_sample(mat, s.dir, n, ev, st);
    if (bs.pdf < 0.0) {
      s.state = ST_TERMINATED;
      slots[idx] = s;
      return;
    }
    s.throughput = hadamard(s.throughput, bs.weight);
    s.prev_pdf_dir = bs.pdf;
    // entered（折射/透射）与 specular 都是 δ（镜面）事件（integrator 同口径）。
    s.flags = (s.flags & 0xFFFFFFFEu) | select(0u, 1u, bs.specular != 0u || bs.entered != 0u);
    let rr = wf_rr(st, s.throughput, ev + 1u);
    if (rr.x == 0.0) {
      s.state = ST_TERMINATED;
      slots[idx] = s;
      return;
    }
    s.throughput = s.throughput * rr.y;
    // 出射侧判定（与 integrator 同一规则：dir·n > 0 ⇒ 出射）。
    let eps = select(-SURFACE_EPS, SURFACE_EPS, vdot(bs.dir, n) > 0.0);
    s.origin = p + n * eps;
    s.dir = bs.dir;
  }
  s.bounce = s.bounce + 1u;
  // `bounce ≥ max_bounces` ⇒ 无更多事件（CPU：`st.bounce < max_bounces` 出循环）。
  // 辐亮度已累计，终止只停队列。
  if (s.bounce >= params.max_bounces) {
    s.state = ST_TERMINATED;
  }
  slots[idx] = s;
}

// ---- 累积 stage：spp 均值（accum.rs::mean_samples 固定形状树逐位同树）----

/// `clamp01 → *255 → floor(x+0.5)`（与 megakernel 的 `quantize` 逐式相同）。
fn quantize_compat(x: f32) -> u32 {
  let clamped = clampf(x, 0.0, 1.0);
  let scaled = clamped * 255.0;
  let shifted = scaled + 0.5;
  return u32(floor(shifted));
}

/// 路径模式的 spp 上限：固定形状树用 64 槽 局部数组（prism-core reduce 的
/// 层序配对 + 奇数尾直进）。宿主对 `spp > 64` 的路径模式请求结构化拒绝。
const WF_MAX_SPP: u32 = 64u;

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
  // 路径模式：`reduce` 固定形状树——层序推进、层内左到右、左操作数在前、
  // 奇数尾元素直进下一层；最后逐通道除以 spp。与 `mean_samples` 逐位同树
  // （vec4 分量加法 = 4 条独立 f32 加法，IEEE 无交互）。
  let spp = params.spp;
  var layer: array<vec4<f32>, 64>;
  var k = 0u;
  loop {
    if (k >= spp) { break; }
    let r = slots[base + k].radiance;
    layer[k] = vec4<f32>(r.x, r.y, r.z, 1.0);
    k = k + 1u;
  }
  var n = spp;
  while (n > 1u) {
    var m = 0u;
    var i = 0u;
    loop {
      if (i >= n) { break; }
      if (i + 1u < n) {
        layer[m] = layer[i] + layer[i + 1u];
        m = m + 1u;
        i = i + 2u;
      } else {
        layer[m] = layer[i]; // 奇数尾直进（形状规则）
        m = m + 1u;
        i = i + 1u;
      }
    }
    n = m;
  }
  accum[idx] = layer[0] / vec4<f32>(f32(spp));
}
