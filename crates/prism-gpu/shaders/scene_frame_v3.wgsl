// scene_frame_v3.wgsl — 光追场景整帧 kernel，**V3 剖面**（W1-B 纹理）。
//
// 与 `scene_frame.wgsl`（V2，6 绑定）的关系：
//
// | | V2 | 本文件（V3） |
// |---|---|---|
// | 绑定 | 6（params/spheres/lights/triangles/nodes/out） | 9（再追加 textures/texmeta/cdf） |
// | 参数块 | 24 字 | 26 字（再追加 P_NUM_TEXTURES / P_CFD_ROWS） |
// | 材质 | 球自带 albedo | albedo ← `baseColorTexture` 在命中点 uv 处取样 |
// | 输出 | `@binding(5)` | `@binding(8)`（**恒为最后一个**） |
//
// 真源是 `crate::scene_abi`（`V3_BINDINGS` / `PARAMS_WORDS_V3` / 各槽位常量）；
// 本文件里的数字是它的**副本**，由 `shaders::tests::v3_shader_matches_abi`
// 逐格对账（改一处漏一处就红）。
//
// # 纹理口径（与 `prism_rt::texsample` 逐式同构）
//
// 1. 纹素在**线性域**（sRGB 解码已在 CPU 侧一次完成，三后端上传同一份 f32
//    位型；滤波发生在线性域）；
// 2. 纹素中心在 `(i+0.5)/n`：`x = u·w − 0.5`，整数部分即左邻纹素；
// 3. 平铺：索引取 `((i % n) + n) % n`（不是钳制）；
// 4. 三线性：级间 lerp，`lod` 钳到 `[0, levels−1]`，`frac == 0` 不混下一级；
// 5. 各向异性：2×2 SVD → `taps = clamp(ceil(σmax/σmin), 1, 4)`，
//    每 tap 宽 `tap_w = max(σmin, σmax/taps)`、中心跨度 `σmax − tap_w`。
//    并集恰好等于长轴（多一分就糊、少一分就锯齿）。
//
// 禁 `array<vec3<f32>>`：`textures` 的元素是 `vec4<f32>`（16 B，无歧义）。
// 禁超越函数（D-29）：球面 uv 需要的 `atan2`/`acos` 用本文件的 det 级数
// （与 `prism_math::det_trig` 同一套系数与归约），不调 WGSL 内建。

// ---- 参数块 / 槽位（= `crate::scene_abi` 的副本） ----
const PARAMS_WORDS_: u32 = 26u;   // == scene_abi::PARAMS_WORDS_V3
const SPHERE_STRIDE: u32 = 8u;    // == scene_abi::SPHERE_SLOTS
const LIGHT_STRIDE: u32 = 8u;     // == scene_abi::LIGHT_SLOTS
const TEXMETA_STRIDE: u32 = 8u;   // 32 B / 4 = 8 个 u32 槽（两个 vec4<u32>）
const MAX_ANISO_TAPS: u32 = 4u;   // == prism_rt::texsample::MAX_ANISO_TAPS

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
const P_CONTRAST_ON: u32 = 18u;
const P_FILMIC: u32 = 19u;
const P_NUM_SPHERES: u32 = 20u;
const P_NUM_LIGHTS: u32 = 21u;
const P_NUM_TEXTURES: u32 = 24u;  // == scene_abi::P_NUM_TEXTURES
const P_CFD_ROWS: u32 = 25u;      // == scene_abi::P_CFD_ROWS

// 球材质的贴图槽（== `prism_rt::bsdf` 的 MAP_*）。
const MAP_BASE: u32 = 0u;
const MAP_ROUGH: u32 = 1u;
const MAP_NORMAL: u32 = 2u;
const MAP_EMISSIVE: u32 = 3u;

const BG_COLOR: vec3<f32> = vec3<f32>(0.04, 0.05, 0.08);
const AMBIENT: f32 = 0.08;
const NORMALIZE_EPS: f32 = 1e-8;
const FRAC_1_PI: f32 = 0.3183098861837907;
const FRAC_1_TAU: f32 = 0.15915494309189535;  // = 0.5 * FRAC_1_PI
const SURFACE_EPS: f32 = 1e-3;
const NO_TEXTURE: u32 = 0u;

// ---- 绑定（V3 连续前缀 0..8；输出恒为最后一个） ----
@group(0) @binding(0) var<storage, read>       params: array<u32>;
@group(0) @binding(1) var<storage, read>       spheres: array<u32>;
@group(0) @binding(2) var<storage, read>       lights: array<u32>;
@group(0) @binding(3) var<storage, read>       triangles: array<u32>;
@group(0) @binding(4) var<storage, read>       nodes: array<u32>;
// 纹素：`vec4<f32>`（线性域）。**不声明 `array<vec3<f32>>`**。
@group(0) @binding(5) var<storage, read>       textures: array<vec4<f32>>;
// 纹理元数据：`vec4<u32>` × 2 = {w, h, levels, offset} + {fmt, srgb, 0, 0}。
@group(0) @binding(6) var<storage, read>       texmeta: array<vec4<u32>>;
// 环境 CDF（W2 用；本 kernel 只读不采样，占位以固定绑定编号）。
@group(0) @binding(7) var<storage, read>       cdf: array<u32>;
@group(0) @binding(8) var<storage, read_write> outbuf: array<u32>;

// ---------------------------------------------------------------------------
// 确定性反三角（与 `prism_math::det_trig` 同一系数；禁 libm — D-29）
// ---------------------------------------------------------------------------

// `[0, 1]` 上的 atan（Maclaurin，Horner；z^17 截断）。
fn quarter_atan(z: f32) -> f32 {
  let z2 = z * z;
  return z * (1.0 + z2 * (-0.3333333333333333 + z2 * (0.2 + z2 * (-0.14285714285714285
    + z2 * (0.1111111111111111 + z2 * (-0.09090909090909091
    + z2 * (0.07692307692307693 + z2 * (-0.06666666666666667))))))));
}

// `atan(x)`，`x >= 0`，三段归约（只含 + − × ÷）。
fn atan_pos(x: f32) -> f32 {
  const FRAC_PI_2: f32 = 1.5707963267948966;
  const FRAC_PI_4: f32 = 0.7853981633974483;
  const TAN_PI_8: f32 = 0.41421357;
  if (x > 1.0) {
    return FRAC_PI_2 - atan_pos(1.0 / x);
  }
  if (x > TAN_PI_8) {
    return FRAC_PI_4 + quarter_atan((x - 1.0) / (x + 1.0));
  }
  return quarter_atan(x);
}

// `atan2(y, x)`，值域 (−π, π]；特例与 CPU 侧逐条一致。
fn det_atan2(y: f32, x: f32) -> f32 {
  const PI: f32 = 3.141592653589793;
  const FRAC_PI_2: f32 = 1.5707963267948966;
  if (x == 0.0) {
    if (y == 0.0) {
      return 0.0;
    }
    if (y > 0.0) {
      return FRAC_PI_2;
    }
    return -FRAC_PI_2;
  }
  let ax = abs(x);
  let ay = abs(y);
  var r: f32;
  if (ay <= ax) {
    r = atan_pos(ay / ax);
  } else {
    r = FRAC_PI_2 - atan_pos(ax / ay);
  }
  if (x > 0.0) {
    if (y >= 0.0) {
      return r;
    }
    return -r;
  }
  if (y >= 0.0) {
    return PI - r;
  }
  return r - PI;
}

// `acos(x)`（端点钳制；`1−x²` 用 `(1−x)(1+x)` 避相消）。
fn det_acos(x: f32) -> f32 {
  var xc = x;
  if (xc > 1.0) {
    xc = 1.0;
  } else if (xc < -1.0) {
    xc = -1.0;
  }
  let s = sqrt(max((1.0 - xc) * (1.0 + xc), 0.0));
  return det_atan2(s, xc);
}

// ---------------------------------------------------------------------------
// 纹理采样（与 `prism_rt::texsample` 逐式同构）
// ---------------------------------------------------------------------------

fn tex_w(handle: u32) -> u32 { return texmeta[(handle - 1u) * 2u].x; }
fn tex_h(handle: u32) -> u32 { return texmeta[(handle - 1u) * 2u].y; }
fn tex_levels(handle: u32) -> u32 { return max(texmeta[(handle - 1u) * 2u].z, 1u); }
fn tex_offset(handle: u32) -> u32 { return texmeta[(handle - 1u) * 2u].w; }

// 第 level 级的尺寸（每级减半，最小 1）。
fn level_dims(handle: u32, level: u32) -> vec2<u32> {
  let lv = min(level, 31u);
  return vec2<u32>(max(tex_w(handle) >> lv, 1u), max(tex_h(handle) >> lv, 1u));
}

// 第 level 级在扁平纹素流里的起点（前缀各级之和）。
fn level_base(handle: u32, level: u32) -> u32 {
  var base = tex_offset(handle);
  let lv = min(level, tex_levels(handle));
  var l = 0u;
  loop {
    if (l >= lv) { break; }
    let d = level_dims(handle, l);
    base = base + d.x * d.y;
    l = l + 1u;
  }
  return base;
}

// 平铺取模：`((i % n) + n) % n`（u32 版；i 先按 i64 语义折到 [0,n)）。
fn wrap_i(i: i32, n: i32) -> u32 {
  var m = i % n;
  if (m < 0) { m = m + n; }
  return u32(m);
}

// 单级双线性（repeat 平铺）。
fn tex_bilinear(handle: u32, level: u32, uv: vec2<f32>) -> vec4<f32> {
  let d = level_dims(handle, level);
  let base = level_base(handle, level);
  let wf = f32(d.x);
  let hf = f32(d.y);
  let x = uv.x * wf - 0.5;
  let y = uv.y * hf - 0.5;
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let fx = x - floor(x);
  let fy = y - floor(y);
  let i0 = wrap_i(x0, i32(d.x));
  let i1 = wrap_i(x0 + 1, i32(d.x));
  let j0 = wrap_i(y0, i32(d.y));
  let j1 = wrap_i(y0 + 1, i32(d.y));
  let t00 = textures[base + j0 * d.x + i0];
  let t10 = textures[base + j0 * d.x + i1];
  let t01 = textures[base + j1 * d.x + i0];
  let t11 = textures[base + j1 * d.x + i1];
  // 固定结合序：先沿 u 后沿 v（与 CPU 同式，逐位可比）。
  let a = t00 + (t10 - t00) * fx;
  let b = t01 + (t11 - t01) * fx;
  return a + (b - a) * fy;
}

// 三线性：两级双线性 + 级间 lerp。
fn tex_trilinear(handle: u32, uv: vec2<f32>, lod_in: f32) -> vec4<f32> {
  let max_level = f32(tex_levels(handle)) - 1.0;
  let lod = clamp(lod_in, 0.0, max_level);
  let l0 = u32(floor(lod));
  let frac = lod - floor(lod);
  let a = tex_bilinear(handle, l0, uv);
  if (frac <= 0.0) {
    return a;   // 级内精确命中：不混下一级（V1 的逐位前提）
  }
  let b = tex_bilinear(handle, min(l0 + 1u, tex_levels(handle) - 1u), uv);
  return a + (b - a) * frac;
}

// 按足迹采样（双线性 + mip 三线性 + 有界各向异性）。
// `foot_c0/c1` 是**世界足迹两半轴在 uv 空间的像**（与 CPU 侧同名量同义）。
fn tex_sample(handle: u32, uv: vec2<f32>, foot_c0: vec2<f32>, foot_c1: vec2<f32>) -> vec4<f32> {
  if (handle == NO_TEXTURE || handle > params[P_NUM_TEXTURES]) {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);   // 无贴图 = 白（乘法单位元）
  }
  let w = f32(tex_w(handle));
  let h = f32(tex_h(handle));
  // 足迹换到纹素空间再做 2×2 SVD：σ 的单位直接是纹素。
  let c0 = vec2<f32>(foot_c0.x * w, foot_c0.y * h);
  let c1 = vec2<f32>(foot_c1.x * w, foot_c1.y * h);
  let a = dot(c0, c0);
  let b = dot(c0, c1);
  let c = dot(c1, c1);
  let half = (a + c) * 0.5;
  let dd = (a - c) * 0.5;
  let root = sqrt(max(dd * dd + b * b, 0.0));
  let lmax = max(half + root, 0.0);
  let s_max = sqrt(lmax);
  let s_min = sqrt(max(half - root, 0.0));
  if (!(s_max > 0.0)) {
    return tex_bilinear(handle, 0u, uv);    // 零足迹 = 点采样（V1 走这条）
  }
  // 主轴（各向同性时取 +u，保持确定性——与 CPU 侧 `singular_axes` 同约定）。
  var dir = vec2<f32>(1.0, 0.0);
  if (abs(b) > 0.0) {
    dir = normalize(vec2<f32>(b, lmax - a));
  } else if (a < c) {
    dir = vec2<f32>(0.0, 1.0);
  }
  var taps = MAX_ANISO_TAPS;
  if (s_min > 0.0) {
    taps = u32(clamp(ceil(s_max / s_min), 1.0, f32(MAX_ANISO_TAPS)));
  }
  // tap 布局：并集 = 长轴（tap_w 宽、中心跨度 σ_max − tap_w）。
  let tap_w = max(s_min, s_max / f32(taps));
  let lod = log2_det(max(tap_w, 1.0e-8));
  let span = max(s_max - tap_w, 0.0);
  var gap = 0.0;
  if (taps > 1u) {
    gap = span / (f32(taps) - 1.0);
  }
  var acc = vec4<f32>(0.0);
  var k = 0u;
  loop {
    if (k >= taps) { break; }
    let t = f32(k) - (f32(taps) - 1.0) * 0.5;
    let u = vec2<f32>(uv.x + dir.x * gap / w * t, uv.y + dir.y * gap / h * t);
    acc = acc + tex_trilinear(handle, u, lod);
    k = k + 1u;
  }
  return acc / f32(taps);
}

// `log2`（只含 × ÷ 与位操作；与 `texsample::log2_f32` 同法，精度 ~1e-6）。
fn log2_det(x: f32) -> f32 {
  if (!(x > 0.0)) {
    return 0.0;
  }
  let bits = bitcast<u32>(x);
  var e = i32((bits >> 23u) & 0xFFu) - 127;
  let m = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F800000u);   // [1,2)
  var mm = m;
  if (mm >= 1.4142135623730951) {
    mm = mm * 0.5;
    e = e + 1;
  }
  // atanh 级数：log2(m) = 2·atanh(z)/ln2，z = (m−1)/(m+1)，|z| ≤ 0.172
  let z = (mm - 1.0) / (mm + 1.0);
  let z2 = z * z;
  let s = z * (1.0 + z2 * (0.3333333333333333 + z2 * (0.2 + z2 * (0.14285714285714285
    + z2 * (0.1111111111111111 + z2 * 0.09090909090909091)))));
  return f32(e) + s * 2.885390081777927;   // 2/ln2
}

// ---------------------------------------------------------------------------
// 球面 uv（`Scene::uv_at` 的 Sphere 臂逐式移植）
// ---------------------------------------------------------------------------

fn uv_sphere(p: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
  let d = (p - center) / radius;
  let u = det_atan2(d.z, d.x) * FRAC_1_TAU + 0.5;
  let v = det_acos(clamp(d.y, -1.0, 1.0)) * FRAC_1_PI;
  return vec2<f32>(u, v);
}

// 球面 uv 的解析雅可比（世界单位切向 → uv）。球面上
// `∂u/∂(方位切向) = 1/(2πr)`、`∂v/∂(极向切向) = 1/(πr)`。
fn uv_jacobian_sphere(radius: f32, t1: vec3<f32>, t2: vec3<f32>) -> mat2x2<f32> {
  let ju = 1.0 / (6.283185307179586 * radius);
  let jv = 1.0 / (3.141592653589793 * radius);
  return mat2x2<f32>(ju * dot(t1, t1), 0.0, 0.0, jv * dot(t2, t2));
}

// ---------------------------------------------------------------------------
// 取数 / 打包
// ---------------------------------------------------------------------------

fn sphere_center(i: u32) -> vec3<f32> {
  let b = i * SPHERE_STRIDE;
  return vec3<f32>(bitcast<f32>(spheres[b]), bitcast<f32>(spheres[b + 1u]), bitcast<f32>(spheres[b + 2u]));
}
fn sphere_radius(i: u32) -> f32 { return bitcast<f32>(spheres[i * SPHERE_STRIDE + 3u]); }
fn sphere_albedo(i: u32) -> vec3<f32> {
  let b = i * SPHERE_STRIDE + 4u;
  return vec3<f32>(bitcast<f32>(spheres[b]), bitcast<f32>(spheres[b + 1u]), bitcast<f32>(spheres[b + 2u]));
}
fn sphere_map(i: u32, slot: u32) -> u32 { return spheres[i * SPHERE_STRIDE + 7u + slot]; }

fn light_dir(i: u32) -> vec3<f32> {
  let b = i * LIGHT_STRIDE;
  return vec3<f32>(bitcast<f32>(lights[b]), bitcast<f32>(lights[b + 1u]), bitcast<f32>(lights[b + 2u]));
}
fn light_color(i: u32) -> vec3<f32> {
  let b = i * LIGHT_STRIDE + 3u;
  return vec3<f32>(bitcast<f32>(lights[b]), bitcast<f32>(lights[b + 1u]), bitcast<f32>(lights[b + 2u]));
}
fn light_intensity(i: u32) -> f32 { return bitcast<f32>(lights[i * LIGHT_STRIDE + 6u]); }

fn vnormalize(v: vec3<f32>) -> vec3<f32> {
  let l = sqrt(dot(v, v));
  if (l > NORMALIZE_EPS) {
    return v / l;
  }
  return vec3<f32>(0.0, 0.0, 0.0);
}

fn pack_rgba(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x, 0.0, 1.0) * 255.0);
  let g = u32(clamp(c.y, 0.0, 1.0) * 255.0);
  let b = u32(clamp(c.z, 0.0, 1.0) * 255.0);
  return (255u << 24u) | (b << 16u) | (g << 8u) | r;
}

// ---------------------------------------------------------------------------
// 入口（与 `CpuRefBackend::render_scene` 逐句对应；纹理只改 albedo 一项）
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn scene_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = params[P_WIDTH];
  let h = params[P_HEIGHT];
  let gx = gid.x;
  let gy = gid.y;
  if (gx >= w || gy >= h) {
    return;   // 尾组守卫
  }
  let eye = vec3<f32>(bitcast<f32>(params[P_EYE]), bitcast<f32>(params[P_EYE + 1u]), bitcast<f32>(params[P_EYE + 2u]));
  let fwd = vec3<f32>(bitcast<f32>(params[P_FORWARD]), bitcast<f32>(params[P_FORWARD + 1u]), bitcast<f32>(params[P_FORWARD + 2u]));
  let rgt = vec3<f32>(bitcast<f32>(params[P_RIGHT]), bitcast<f32>(params[P_RIGHT + 1u]), bitcast<f32>(params[P_RIGHT + 2u]));
  let upv = vec3<f32>(bitcast<f32>(params[P_UP]), bitcast<f32>(params[P_UP + 1u]), bitcast<f32>(params[P_UP + 2u]));
  let tan_half = bitcast<f32>(params[P_TAN_HALF_FOV]);
  let aspect = bitcast<f32>(params[P_ASPECT]);

  // 像素中心的 NDC → 世界方向。
  let ndx = ((f32(gx) + 0.5) / f32(w) * 2.0 - 1.0) * tan_half * aspect;
  let ndy = (1.0 - (f32(gy) + 0.5) / f32(h) * 2.0) * tan_half;
  let dir = vnormalize(fwd + rgt * ndx + upv * ndy);

  // 最近命中（球族）。
  var closest = 1.0e30;
  var hit: u32 = 0xFFFFFFFFu;
  var t = 0u;
  let n_spheres = params[P_NUM_SPHERES];
  loop {
    if (t >= n_spheres) { break; }
    let oc = eye - sphere_center(t);
    let b = dot(oc, dir);
    let c = dot(oc, oc) - sphere_radius(t) * sphere_radius(t);
    let disc = b * b - c;
    if (disc >= 0.0) {
      let sq = sqrt(disc);
      let t0 = -b - sq;
      let t1 = -b + sq;
      var tt = t1;
      if (t0 > 1.0e-4) { tt = t0; }
      if (tt > 1.0e-4 && tt < closest) {
        closest = tt;
        hit = t;
      }
    }
    t = t + 1u;
  }

  var color = BG_COLOR;
  if (hit < n_spheres) {
    let p = eye + dir * closest;
    let nrm = vnormalize(p - sphere_center(hit));
    // 纹理：base map 在命中点 uv 处取样（无贴图则保持球自带 albedo）。
    var albedo = sphere_albedo(hit);
    let base_map = sphere_map(hit, MAP_BASE);
    if (base_map != NO_TEXTURE) {
      let uv = uv_sphere(p, sphere_center(hit), sphere_radius(hit));
      // 足迹 = 像素锥（全宽 `spread`）× 1/cosθ × uv 雅可比。锥宽与 CPU 侧
      // `trace_one` 的 `spread0 = 2·tan_half/H` 同式。
      let cos_i = clamp(abs(dot(dir, nrm)), 0.02, 1.0);
      let spread = 2.0 * tan_half / f32(h);
      let r = closest * spread;
      // 切平面上的两轴：T1 = view 在切平面上的投影（退化取任意切向）。
      let vt = dir - nrm * dot(dir, nrm);
      var t1 = vnormalize(vt);
      if (dot(t1, t1) <= 0.0) {
        t1 = vnormalize(cross(nrm, vec3<f32>(0.0, 0.0, 1.0)));
      }
      let t2 = cross(nrm, t1);
      let ja = uv_jacobian_sphere(sphere_radius(hit), t1, t2);
      let a = r / cos_i;
      let c0 = vec2<f32>(ja[0][0] * a, ja[0][1] * a);
      let c1 = vec2<f32>(ja[1][0] * r, ja[1][1] * r);
      let s = tex_sample(base_map, uv, c0, c1);
      albedo = vec3<f32>(albedo.x * s.x, albedo.y * s.y, albedo.z * s.z);
    }

    var lit = vec3<f32>(AMBIENT);
    var j = 0u;
    let n_lights = params[P_NUM_LIGHTS];
    loop {
      if (j >= n_lights) { break; }
      let ldir = vnormalize(light_dir(j));
      let diff = max(dot(nrm, ldir), 0.0);
      lit = lit + light_color(j) * (diff * light_intensity(j));
      j = j + 1u;
    }
    color = vec3<f32>(albedo.x * lit.x, albedo.y * lit.y, albedo.z * lit.z);
  }

  // aesthetic：teal-orange → contrast → filmic（与 CPU 参考同序同式）。
  if (params[P_TEAL_ORANGE] != 0u) {
    color = vec3<f32>(color.x, color.y * 0.92, color.z * 0.86);
  }
  if (params[P_CONTRAST_ON] != 0u) {
    let c0 = bitcast<f32>(params[P_CONTRAST]);
    color = vec3<f32>((color.x - 0.5) * c0 + 0.5, (color.y - 0.5) * c0 + 0.5, (color.z - 0.5) * c0 + 0.5);
  }
  if (params[P_FILMIC] != 0u) {
    color = vec3<f32>(color.x / (color.x + 1.0), color.y / (color.y + 1.0), color.z / (color.z + 1.0));
  }
  outbuf[gy * w + gx] = pack_rgba(color);
}
