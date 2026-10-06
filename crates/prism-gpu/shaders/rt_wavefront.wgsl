// prism-rt wavefront 队列 kernel（D2）——生成/着色/扩展/连接 四阶段。
//
// # 设计（对应路线图 D2 行"wavefront 队列 kernel：生成/着色/扩展/连接 各阶段"）
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
  _pad0: u32,
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
}
// 独立 group(3)：只被 `generate`（经 cast_ray）引用，单绑定前缀 {0}。
@group(3) @binding(0) var<uniform> camera: Camera;

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
  let si = sample_index(px, py, sample, 0u);
  let ray = cast_ray(vec2<u32>(px, py), sample, si);
  var s: PathSlot;
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
@compute @workgroup_size(64)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.width * params.height * params.spp;
  if (idx >= total) { return; }
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
    // NEE 直接光（反射域）。
    s.radiance = s.radiance + nee_surface(p, n, s.dir, s.throughput, ev, s.aux_steps);
    // MNEE 焦散（k=2 球面折射链；对齐 caustics.rs::mnee_refraction）。
    if (params.nee_only == 0u) {
      s.radiance = s.radiance + mnee_caustics(p, n, s.dir, s.throughput, ev);
    }
  } else {
    // 逃逸：环境背景。
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
@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let npix = params.width * params.height;
  if (idx >= npix) { return; }
  var sum = vec3<f32>(0.0, 0.0, 0.0);
  let base = idx * params.spp;
  var k = 0u;
  while (k < params.spp) {
    sum = sum + slots[base + k].radiance;
    k = k + 1u;
  }
  accum[idx] = vec4<f32>(sum / f32(params.spp), 1.0);
}
