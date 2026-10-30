// scene_frame.wgsl — 光追场景整帧 kernel（V2.2-A-1 / T03）。
//
// 第 4 份手写场景源（WGSL 文本态），与 `vulkan/spirv_rt`（SPIR-V 发射器）、
// `metal`（经 MoltenVK 翻译层复用 Vulkan 剖面）共享同一套参数块与绑定剖面
// （真源见 `crate::scene_abi`）。本文件只负责"语义逐句对应"
// `lib.rs` 的 `CpuRefBackend::render_scene`（相机基 → 球循环最近命中 →
// 灯循环漫反射 → teal/contrast/filmic → 量化打包）。
//
// 设计约束（与 prism-gpu 既有门禁一致）：
// - 禁超越函数：不使用任何"逐 GPU 实现精度漂移"的内建（无 sin/cos/exp/pow/log），
//   相机焦距标度 `tan(fov/2)` 由 CPU 预算成 `P_TAN_HALF_FOV` 位型传入（D-29）。
// - 不写 `fma()`（WGSL 无保证的融合乘加，破坏跨端逐位一致性）。
// - 浮点一律位型：参数块与实体缓冲里的 f32 以 `to_bits()` 存入，shader 侧
//   `bitcast<f32>` 还原；不存在第二套十进制/文本往返口径。
// - 绑定连续前缀 `{0,1,2,3}` = params / spheres / lights / out，满足
//   `web/mod.rs` 的"每个 group 的绑定须从 0 起连续"硬约束。

// ---- 参数块 / 槽位（与 `crate::scene_abi` 逐字一致；qa 测试钉偏移） ----
const PARAMS_WORDS_: u32 = 22u;   // 必须等于 scene_abi::PARAMS_WORDS
const SPHERE_STRIDE: u32 = 8u;    // 每球 f32 槽数，必须等于 scene_abi::SPHERE_SLOTS
const LIGHT_STRIDE: u32 = 8u;     // 每灯 f32 槽数，必须等于 scene_abi::LIGHT_SLOTS

// 参数块各字偏移（u32 槽位）。
const P_WIDTH: u32 = 0u;
const P_HEIGHT: u32 = 1u;
const P_EYE: u32 = 2u;
const P_FORWARD: u32 = 5u;
const P_RIGHT: u32 = 8u;
const P_UP: u32 = 11u;
const P_TAN_HALF_FOV: u32 = 14u;
const P_ASPECT: u32 = 15u;
const P_TEAL_ORANGE: u32 = 16u;
const P_CONTRAST: u32 = 17u;
const P_CONTRAST_ON: u32 = 18u;        // contrast != 1.0 的预计算开关（CPU 侧判定）
const P_FILMIC: u32 = 19u;
const P_NUM_SPHERES: u32 = 20u;
const P_NUM_LIGHTS: u32 = 21u;

// ---- 绑定（连续前缀 0..3） ----
@group(0) @binding(0) var<storage, read>       params: array<u32>;
@group(0) @binding(1) var<storage, read>       spheres: array<u32>;
@group(0) @binding(2) var<storage, read>       lights: array<u32>;
@group(0) @binding(3) var<storage, read_write> outbuf: array<u32>;

// 把颜色各分量钳到 [0,1]。
fn clamp01(c: vec3<f32>) -> vec3<f32> {
  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// 分量相乘（albedo * 光照项）。
fn vmul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(a.x * b.x, a.y * b.y, a.z * b.z);
}

// RGBA8 little-endian 打包：内存字节序 [r, g, b, a]（与 `background.wgsl` 同式）。
fn pack_rgba(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x, 0.0, 1.0) * 255.0);
  let g = u32(clamp(c.g, 0.0, 1.0) * 255.0);
  let b = u32(clamp(c.b, 0.0, 1.0) * 255.0);
  let a = 255u;
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8, 1)
fn scene_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params[P_WIDTH];
  let h = params[P_HEIGHT];
  // 尾组守卫：越界 invocation 直接返回（与 CPU 的 `for py in 0..h / for px in 0..w` 同口径）。
  if (gid.x >= w || gid.y >= h) {
    return;
  }
  let idx = gid.y * w + gid.x;

  // ---- 相机基（CPU 已算好，逐字段取出；与 `camera_basis` 同布局） ----
  let eye = vec3<f32>(
    bitcast<f32>(params[P_EYE]),
    bitcast<f32>(params[P_EYE + 1u]),
    bitcast<f32>(params[P_EYE + 2u]));
  let forward = vec3<f32>(
    bitcast<f32>(params[P_FORWARD]),
    bitcast<f32>(params[P_FORWARD + 1u]),
    bitcast<f32>(params[P_FORWARD + 2u]));
  let right = vec3<f32>(
    bitcast<f32>(params[P_RIGHT]),
    bitcast<f32>(params[P_RIGHT + 1u]),
    bitcast<f32>(params[P_RIGHT + 2u]));
  let up = vec3<f32>(
    bitcast<f32>(params[P_UP]),
    bitcast<f32>(params[P_UP + 1u]),
    bitcast<f32>(params[P_UP + 2u]));
  let tan_f = bitcast<f32>(params[P_TAN_HALF_FOV]);
  let aspect = bitcast<f32>(params[P_ASPECT]);
  let teal = params[P_TEAL_ORANGE];
  let contrast = bitcast<f32>(params[P_CONTRAST]);
  let contrast_on = params[P_CONTRAST_ON];
  let filmic = params[P_FILMIC];
  let num_spheres = params[P_NUM_SPHERES];
  let num_lights = params[P_NUM_LIGHTS];

  // ---- 每像素射线（与 CPU 参考逐式对应） ----
  let nx = (2.0 * (f32(gid.x) + 0.5) / f32(w) - 1.0) * tan_f * aspect;
  let ny = (1.0 - 2.0 * (f32(gid.y) + 0.5) / f32(h)) * tan_f;
  let dir = normalize(forward + right * nx + up * ny);
  let origin = eye;

  // ---- 球循环：最近命中（t = -b - sqrt(disc)） ----
  var closest_t: f32 = 0.0;
  var hit_idx: i32 = -1;                     // -1 = 未命中
  var j: u32 = 0u;
  loop {
    if (j >= num_spheres) {
      break;
    }
    let base = j * SPHERE_STRIDE;
    let center = vec3<f32>(
      bitcast<f32>(spheres[base]),
      bitcast<f32>(spheres[base + 1u]),
      bitcast<f32>(spheres[base + 2u]));
    let radius = bitcast<f32>(spheres[base + 3u]);
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc >= 0.0) {
      let t = -b - sqrt(disc);
      if (t > 0.0 && (hit_idx < 0 || t < closest_t)) {
        closest_t = t;
        hit_idx = i32(j);
      }
    }
    j = j + 1u;
  }

  // ---- 着色 ----
  var color = vec3<f32>(0.04, 0.05, 0.08);   // 背景色（与 CPU 参考同值）
  if (hit_idx >= 0) {
    let base = u32(hit_idx) * SPHERE_STRIDE; // 命中球的真实下标
    let center = vec3<f32>(
      bitcast<f32>(spheres[base]),
      bitcast<f32>(spheres[base + 1u]),
      bitcast<f32>(spheres[base + 2u]));
    let p = origin + dir * closest_t;
    let n = normalize(p - center);
    let albedo = vec3<f32>(
      bitcast<f32>(spheres[base + 4u]),
      bitcast<f32>(spheres[base + 5u]),
      bitcast<f32>(spheres[base + 6u]));
    let ambient = vec3<f32>(0.08, 0.08, 0.08);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var l: u32 = 0u;
    loop {
      if (l >= num_lights) {
        break;
      }
      let lbase = l * LIGHT_STRIDE;
      // 方向已在 CPU 侧归一化（`encode_lights`），此处直接取用。
      let ldir = vec3<f32>(
        bitcast<f32>(lights[lbase]),
        bitcast<f32>(lights[lbase + 1u]),
        bitcast<f32>(lights[lbase + 2u]));
      let lcolor = vec3<f32>(
        bitcast<f32>(lights[lbase + 3u]),
        bitcast<f32>(lights[lbase + 4u]),
        bitcast<f32>(lights[lbase + 5u]));
      let lint = bitcast<f32>(lights[lbase + 6u]);
      let diff = max(dot(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + vmul(albedo, term);
      l = l + 1u;
    }
    color = acc;
  }

  // ---- 后处理（与 CPU 参考同式，开关由 CPU 侧预算成 u32 标志） ----
  if (teal != 0u) {
    let luma = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
    let shadow_tint = vec3<f32>(0.10, 0.35, 0.45);
    let high_tint = vec3<f32>(1.0, 0.7, 0.3);
    let shadow_comp = vmul(shadow_tint, color) * (1.0 - luma) * 0.5;
    let high_comp = vmul(high_tint, color) * luma * 0.5;
    color = color * 0.6 + shadow_comp + high_comp;
  }

  if (contrast_on != 0u) {
    color = (color - vec3<f32>(0.5, 0.5, 0.5)) * contrast + vec3<f32>(0.5, 0.5, 0.5);
  }

  if (filmic != 0u) {
    // 简易 Reinhard 色调映射（与 CPU 参考同式）。
    color = color / (color + vec3<f32>(1.0, 1.0, 1.0));
  }

  color = clamp01(color);
  outbuf[idx] = pack_rgba(color);
}
