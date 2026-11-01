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
// 本文件里的数字是它的**副本**，由 `scene_abi::tests::v3_wgsl_bindings_and_types_match_abi`
// 逐格对账（改一处漏一处就红）。
//
// # V3 只是"V2 + 纹理"，不是另一套语义
//
// D-21 首次把本文件交给真编译器（naga）/ 真设备（lavapipe）时，发现它此前
// **从未被执行过一次**，四处偏离了"V2 + 纹理"这条设计前提（均已按 V2 的
// 已验证文本改正，逐条见下）。这类偏离在编辑器里看不出来，只有"编译 + 真跑
// + 与 CPU 参考逐字节对拍"才暴露——这正是本文件现在有 `.spv` 与 `spv_freshness`
// 门禁的理由。
//
// 1. **递归**：`atan_pos` 原先自调用（`return π/2 - atan_pos(1/x)`）。Rust 允许，
//    WGSL 语法**禁止**（naga: "declaration of `atan_pos` is recursive"）。改为
//    `atan_unit`（已归约的内层，非递归）+ 一次归约，算术表达式与 CPU 同序。
// 2. **曲线 BVH 整段缺失**：`triangles` / `nodes` 声明了却一次没读过 ⇒ naga 把
//    两个全局变量当死代码删掉，SPIR-V 里只剩 6 个绑定；而 `params` 里
//    `P_NUM_TRIS` / `P_NUM_NODES` 照写 ⇒ 有曲线的场景会**静默丢掉曲线**。
//    现按 V2 的 `bvh_intersect` 等函数逐句移植（同一份 ABI、同一份
//    `prism_geometry::bvh` 语义）。
// 3. **光照结合序**：原先把 `ambient + Σ(color·diff·intensity)` 求和后再乘 albedo，
//    而 CPU 参考是**逐灯** `acc += albedo·(ambient + color·diff·intensity)`。
//    单灯时两者逐位相同，多灯时舍入不同 ⇒ 恢复 CPU 的结合序。
// 4. **teal-orange 是个占位式**（`color.y*0.92` / `color.z*0.86`），与 CPU 的
//    亮度分流双色补偿毫无关系 ⇒ 照抄 CPU `apply_aesthetic` 的同式实现。
// 5. **量化少了 `+0.5`**：`u32(c*255)` 是**截断**，而 CPU 的 `channel_to_u8`
//    是 `(c*255).round()`（V2 的 SPIR-V 发射器也是 `floor(x+0.5)`）。背景色
//    0.05 的真值 12.75 由此被截成 12（应为 13）⇒ 首次真跑时 97% 的像素差
//    1/255。**这条是"真跑 + 逐像素对拍"才暴露的**：只编译、只 `spirv-val`、
//    只看 PSNR 都发现不了（±1/255 对 PSNR 40 dB 毫无影响）。
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
// **可达性（W1-C 实测）**：W1-B 时 `sphere_map` 读的是球块里的 pad 槽
// （`SPHERE_SLOTS = 8`：center/radius/albedo/pad），而编码器恒把 pad 写 0
// ⇒ `base_map == NO_TEXTURE` 恒成立，上面这整段采样逻辑**一次也走不到**。
// W1-C 把球块步长升到 `SPHERE_SLOTS_V3 = 12`（V2 的 8 槽前缀逐字节不变，
// 尾部槽 8..11 放四个句柄），`sphere_map` 改读 `8u + slot`，`encode_spheres_v3`
// 写句柄 ⇒ MAP_BASE 这条路现在**真的可达**，并由
// `vulkan/v3_frame.rs` 的真机对拍钉住（llvmpipe 上 `differing_pixels == 0`）。
//
// 仍未验的部分（**不要当成已支持**）：
//   * 各向异性 / 三线性（`tex_sample` 的非零足迹分支）：本帧取零足迹点采样；
//   * `MAP_ROUGH` / `MAP_NORMAL` / `MAP_EMISSIVE`：本 kernel 不读（见入口处注释）；
//   * 真机（非软件 ICD）上的逐位一致：只在 llvmpipe 上验过。
//
// 禁 `array<vec3<f32>>`：`textures` 的元素是 `vec4<f32>`（16 B，无歧义）。
// 禁超越函数（D-29）：球面 uv 需要的 `atan2`/`acos` 用本文件的 det 级数
// （与 `prism_math::det_trig` 同一套系数与归约），不调 WGSL 内建。

// ---- 参数块 / 槽位（= `crate::scene_abi` 的副本） ----
const PARAMS_WORDS_: u32 = 26u;   // == scene_abi::PARAMS_WORDS_V3
const SPHERE_STRIDE: u32 = 12u;    // == scene_abi::SPHERE_SLOTS_V3（V2 的 8 槽 + 4 句柄）
const LIGHT_STRIDE: u32 = 8u;     // == scene_abi::LIGHT_SLOTS
const TRI_STRIDE: u32 = 12u;      // == scene_abi::TRIANGLE_SLOTS
const NODE_STRIDE: u32 = 12u;     // == scene_abi::NODE_SLOTS
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
const P_NUM_TRIS: u32 = 22u;      // == scene_abi::P_NUM_TRIS（V2 前缀）
const P_NUM_NODES: u32 = 23u;     // == scene_abi::P_NUM_NODES（V2 前缀）
const P_NUM_TEXTURES: u32 = 24u;  // == scene_abi::P_NUM_TEXTURES
const P_CFD_ROWS: u32 = 25u;      // == scene_abi::P_CFD_ROWS

// 球材质的贴图槽（== `prism_rt::bsdf` 的 MAP_*）。
const MAP_BASE: u32 = 0u;
const MAP_ROUGH: u32 = 1u;
const MAP_NORMAL: u32 = 2u;
const MAP_EMISSIVE: u32 = 3u;

// 曲线命中面的反照率（与 `crate::CURVE_ALBEDO` 同值）。
const CURVE_ALBEDO: vec3<f32> = vec3<f32>(0.72, 0.74, 0.78);
// 曲线求交的 t_min（与 `crate::CURVE_T_MIN` 同值）。
const CURVE_T_MIN: f32 = 1e-4;

const BG_COLOR: vec3<f32> = vec3<f32>(0.04, 0.05, 0.08);
const AMBIENT: f32 = 0.08;
const NORMALIZE_EPS: f32 = 1e-8;
const FRAC_1_PI: f32 = 0.3183098861837907;
const FRAC_1_TAU: f32 = 0.15915494309189535;  // = 0.5 * FRAC_1_PI
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
//
// **注意**：naga 会删掉"从未被读"的全局变量，故 `.spv` 里本绑定**不存在**
// （见 `spv_manifest.txt` 旁的说明）。宿主仍按 `V3_BINDINGS` 建 9 个描述符
// 绑定——Vulkan 允许布局里存在 shader 未消费的绑定，行为与"用不到"一致。
@group(0) @binding(7) var<storage, read>       cdf: array<u32>;
@group(0) @binding(8) var<storage, read_write> outbuf: array<u32>;

// ---------------------------------------------------------------------------
// 基础算子（与 V2 / CPU 参考同式）
// ---------------------------------------------------------------------------

// 把颜色各分量钳到 [0,1]。
fn clamp01(c: vec3<f32>) -> vec3<f32> {
  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// 分量相乘（albedo * 光照项）。
fn vmul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(a.x * b.x, a.y * b.y, a.z * b.z);
}

// 单位化（与 `prism_math::Vec3::normalize` 同口径：长度 ≤ 1e-8 退化到零向量）。
fn vnormalize(v: vec3<f32>) -> vec3<f32> {
  let l = sqrt(dot(v, v));
  if (l > NORMALIZE_EPS) {
    return v / l;
  }
  return vec3<f32>(0.0, 0.0, 0.0);
}

// RGBA8 little-endian 打包：内存字节序 [r, g, b, a]（与 V2 同式）。
//
// `+ 0.5` 后截断 = `floor(x·255 + 0.5)`，逐位等于 CPU 的
// `(v.clamp(0,1) * 255.0).round()`（非负域上 round 与 floor(x+0.5) 同值）。
// **不能省这个 0.5**：背景色 0.05 的真值是 12.75，截断得 12、四舍五入得 13
// ——D-21 首次真跑时 97% 的像素都因此差 1/255。V2 的 WGSL 同样漏了这个 0.5
// （WebGPU 的判据是 PSNR ≥ 40 dB，把末位差吸收掉了，故一直没暴露）；
// 本文件按 ABI 真值（V2 的 SPIR-V 发射器就是 `floor(x+0.5)`）修正。
fn pack_rgba(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x, 0.0, 1.0) * 255.0 + 0.5);
  let g = u32(clamp(c.y, 0.0, 1.0) * 255.0 + 0.5);
  let b = u32(clamp(c.z, 0.0, 1.0) * 255.0 + 0.5);
  let a = 255u;
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

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

// `[0, 1]` 上的 atan —— 三段归约里「已归约」的那一步（**非递归**）。
//
// CPU 侧写成 `if x > 1 { π/2 − atan_pos(1/x) }`（Rust 允许递归），但 **WGSL
// 语法禁止函数递归**。`1/x < 1` ⇒ 递归深度恒为 1，展开成「一次归约 + 纯内层
// 函数」后算术表达式与 CPU 同序（`a - r` 仍是先算 r 再做减法）。
fn atan_unit(v: f32) -> f32 {
  const FRAC_PI_4: f32 = 0.7853981633974483;
  const TAN_PI_8: f32 = 0.41421357;
  if (v > TAN_PI_8) {
    return FRAC_PI_4 + quarter_atan((v - 1.0) / (v + 1.0));
  }
  return quarter_atan(v);
}

// `atan(x)`，`x >= 0`（三段归约；只含 + − × ÷）。
fn atan_pos(x: f32) -> f32 {
  const FRAC_PI_2: f32 = 1.5707963267948966;
  if (x > 1.0) {
    return FRAC_PI_2 - atan_unit(1.0 / x);
  }
  return atan_unit(x);
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
// 扁平缓冲取数（槽位 → 向量）
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

// 球材质的贴图句柄。
//
// V3 的球块步长是 12（`SPHERE_SLOTS_V3`）：槽 0..7 是 V2 的
// `center/radius/albedo/pad`（**逐字节不变**），槽 8..11 是四个句柄。
// 该函数**必须**用 `8u + slot` 而不是 `7u + slot`：7 号槽是 V2 的 pad，
// 读它会得到恒 0（= NO_TEXTURE）——W1-B 之前那版纹理段"编译通过、
// 一次也走不到"就是这个下标错位。
fn sphere_map(i: u32, slot: u32) -> u32 { return spheres[i * SPHERE_STRIDE + 8u + slot]; }

fn light_dir(i: u32) -> vec3<f32> {
  let b = i * LIGHT_STRIDE;
  return vec3<f32>(bitcast<f32>(lights[b]), bitcast<f32>(lights[b + 1u]), bitcast<f32>(lights[b + 2u]));
}
fn light_color(i: u32) -> vec3<f32> {
  let b = i * LIGHT_STRIDE + 3u;
  return vec3<f32>(bitcast<f32>(lights[b]), bitcast<f32>(lights[b + 1u]), bitcast<f32>(lights[b + 2u]));
}
fn light_intensity(i: u32) -> f32 { return bitcast<f32>(lights[i * LIGHT_STRIDE + 6u]); }

fn node_bmin(i: u32) -> vec3<f32> {
  let b = i * NODE_STRIDE;
  return vec3<f32>(bitcast<f32>(nodes[b]), bitcast<f32>(nodes[b + 1u]), bitcast<f32>(nodes[b + 2u]));
}

fn node_bmax(i: u32) -> vec3<f32> {
  let b = i * NODE_STRIDE + 4u;
  return vec3<f32>(bitcast<f32>(nodes[b]), bitcast<f32>(nodes[b + 1u]), bitcast<f32>(nodes[b + 2u]));
}

fn tri_vertex(slot: u32, base: u32) -> vec3<f32> {
  let b = slot * TRI_STRIDE + base;
  return vec3<f32>(
    bitcast<f32>(triangles[b]),
    bitcast<f32>(triangles[b + 1u]),
    bitcast<f32>(triangles[b + 2u]));
}

// slab 盒交（exit 侧 +1e-7 外扩消除边界漏交；与 `bvh.rs::Aabb::ray` 同式）。
// 返回 (enter, exit)；空区间返回哨兵 (1e30, -1e30)。
fn aabb_ray(bmin: vec3<f32>, bmax: vec3<f32>, ro: vec3<f32>, inv: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  let t1 = (bmin - ro) * inv;
  let t2 = (bmax - ro) * inv;
  let lo = min(t1, t2);
  let hi = max(t1, t2);
  let enter = max(max(lo.x, max(lo.y, lo.z)), t_min);
  let exitv = min(min(hi.x, min(hi.y, hi.z)), t_max);
  if (enter <= exitv + 1e-7) {
    return vec2<f32>(enter, exitv);
  }
  return vec2<f32>(1e30, -1e30);
}

// Möller–Trumbore（禁 fma；与 `bvh.rs::tri_ray_hit` 同式）。
// 命中返回 t，未命中返回 -1.0。
fn tri_ray_hit(slot: u32, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
  let va = tri_vertex(slot, 0u);
  let vb = tri_vertex(slot, 4u);
  let vc = tri_vertex(slot, 8u);
  let e1 = vb - va;
  let e2 = vc - va;
  let pvec = cross(rd, e2);
  let det = dot(e1, pvec);
  if (abs(det) < 1e-12) {
    return -1.0;
  }
  let inv_det = 1.0 / det;
  let tvec = ro - va;
  let u = dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return -1.0;
  }
  let qvec = cross(tvec, e1);
  let v = dot(rd, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return -1.0;
  }
  let hit = dot(e2, qvec) * inv_det;
  if (hit >= t_min && hit <= t_max) {
    return hit;
  }
  return -1.0;
}

// 三角形几何法线（朝向来光侧；与 CPU 参考同口径）。
fn tri_normal(slot: u32, rd: vec3<f32>) -> vec3<f32> {
  let va = tri_vertex(slot, 0u);
  let vb = tri_vertex(slot, 4u);
  let vc = tri_vertex(slot, 8u);
  var n = vnormalize(cross(vb - va, vc - va));
  if (dot(n, rd) > 0.0) {
    n = n * -1.0;
  }
  return n;
}

// 栈式 BVH 遍历（最近命中）：返回 (t, tri_slot)；未命中 tri_slot = 0xFFFFFFFF。
// 语义逐句移植 `rt_bvh.wgsl::bvh_intersect`：固定深度显式栈（64）、近子先处理、
// 进入 t 相等先左（确定性 tie-break）。
fn bvh_intersect(node_count: u32, tri_count: u32,
                 ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> vec2<f32> {
  var best_t = t_max;
  var best_tri = 0xFFFFFFFFu;
  if (node_count == 0u || tri_count == 0u) {
    return vec2<f32>(best_t, f32(best_tri));
  }
  let inv = vec3<f32>(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
  var stack: array<u32, 64>;
  var tlo: array<f32, 64>;
  stack[0] = 0u;
  tlo[0] = t_min;
  var sp: i32 = 1;
  loop {
    if (sp <= 0) {
      break;
    }
    sp = sp - 1;
    let id = stack[sp];
    let t_lo = tlo[sp];
    if (t_lo > best_t) {
      continue;
    }
    let nb = id * NODE_STRIDE;
    let count = nodes[nb + 9u];
    if (count > 0u) {
      let start = nodes[nb + 8u];
      var i: u32 = 0u;
      loop {
        if (i >= count) {
          break;
        }
        let slot = start + i;
        let ht = tri_ray_hit(slot, ro, rd, t_min, best_t);
        if (ht >= 0.0 && ht < best_t) {
          best_t = ht;
          best_tri = slot;
        }
        i = i + 1u;
      }
    } else {
      let l = id + 1u;
      let r = nodes[nb + 10u];
      let hl = aabb_ray(node_bmin(l), node_bmax(l), ro, inv, t_min, t_max);
      let hr = aabb_ray(node_bmin(r), node_bmax(r), ro, inv, t_min, t_max);
      let ok_l = hl.x <= hl.y;
      let ok_r = hr.x <= hr.y;
      // 近子后入栈（先处理）；进入 t 相等先左（确定性 tie-break）。
      if (ok_l && ok_r) {
        if (hl.x <= hr.x) {
          stack[sp] = r;
          tlo[sp] = min(hr.x, hr.y);
          sp = sp + 1;
          stack[sp] = l;
          tlo[sp] = min(hl.x, hl.y);
          sp = sp + 1;
        } else {
          stack[sp] = l;
          tlo[sp] = min(hl.x, hl.y);
          sp = sp + 1;
          stack[sp] = r;
          tlo[sp] = min(hr.x, hr.y);
          sp = sp + 1;
        }
      } else if (ok_l) {
        stack[sp] = l;
        tlo[sp] = min(hl.x, hl.y);
        sp = sp + 1;
      } else if (ok_r) {
        stack[sp] = r;
        tlo[sp] = min(hr.x, hr.y);
        sp = sp + 1;
      }
    }
  }
  return vec2<f32>(best_t, f32(best_tri));
}

// ---------------------------------------------------------------------------
// 入口（与 `CpuRefBackend::render_scene` 逐句对应；纹理只改 albedo 一项）
// ---------------------------------------------------------------------------

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

  // ---- 每像素射线（与 CPU 参考逐式对应） ----
  let nx = (2.0 * (f32(gid.x) + 0.5) / f32(w) - 1.0) * tan_f * aspect;
  let ny = (1.0 - 2.0 * (f32(gid.y) + 0.5) / f32(h)) * tan_f;
  let dir = normalize(forward + right * nx + up * ny);
  let origin = eye;

  // ---- 球循环：最近命中（t = -b - sqrt(disc)） ----
  let num_spheres = params[P_NUM_SPHERES];
  var closest_t: f32 = 0.0;
  var hit_idx: i32 = -1;                     // -1 = 未命中
  var j: u32 = 0u;
  loop {
    if (j >= num_spheres) {
      break;
    }
    let center = sphere_center(j);
    let radius = sphere_radius(j);
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

  // ---- A-2 曲线：栈式 BVH 遍历（覆盖球命中若更近；曲线为空 ⇒ 整段跳过） ----
  let num_tris = params[P_NUM_TRIS];
  let num_nodes = params[P_NUM_NODES];
  var curve_slot: i32 = -1;
  if (num_tris > 0u && num_nodes > 0u) {
    let t_max_curve = select(1e30, closest_t, hit_idx >= 0);
    let hitc = bvh_intersect(num_nodes, num_tris, origin, dir, CURVE_T_MIN, t_max_curve);
    if (hitc.y != f32(0xFFFFFFFFu)) {
      closest_t = hitc.x;
      curve_slot = i32(hitc.y);
      hit_idx = -1;
    }
  }

  // ---- 着色 ----
  var color = BG_COLOR;
  let ambient = vec3<f32>(AMBIENT, AMBIENT, AMBIENT);
  if (hit_idx >= 0) {
    let si = u32(hit_idx);
    let center = sphere_center(si);
    let radius = sphere_radius(si);
    let p = origin + dir * closest_t;
    let n = normalize(p - center);           // 球面法线恒朝外（与 CPU 同式）
    var albedo = sphere_albedo(si);

    // ---- 纹理（W1-C）：MAP_BASE 在命中点 uv 处取样，**替换** albedo ----
    //
    // 语义权威在 CPU 侧（`prism_rt::integrator::material_bsdf_at` 的 MAP_BASE
    // 臂：`base_color = 采样值`，是**替换**而非"在因子上再乘"）。本 kernel 只
    // 做同一件事——这是三端对拍能逐字节对齐的唯一口径。
    //
    // # 为什么句柄守卫在调用点、而不是只靠 `tex_sample` 内部那条
    //
    // 越界句柄必须**按 NO_TEXTURE 处理**（退化到球自带 albedo）。若只靠
    // `tex_sample` 内部的 `handle > P_NUM_TEXTURES ⇒ 返回白`，越界球会变成
    // "albedo = 白"——那不是"没有贴图"，负向对照就无法用"与无纹理基线逐字节
    // 相等"来判。故守卫提前到此处，CPU 参考走**同一判据**。
    let base_map = sphere_map(si, MAP_BASE);
    if (base_map != NO_TEXTURE && base_map <= params[P_NUM_TEXTURES]) {
      let uv = uv_sphere(p, center, radius);
      // 本版口径：**零足迹点采样**（lod 0、1 tap），与 `CpuRefBackend` 的纹理
      // 参考实现逐式相同。各向异性/三线性（`tex_sample` 的非零足迹分支）
      // 在此 ABI 上**未验**，故本帧不取——不是"已支持"。
      let s = tex_sample(base_map, uv, vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
      albedo = vec3<f32>(s.x, s.y, s.z);
    }
    // `MAP_ROUGH` / `MAP_NORMAL` / `MAP_EMISSIVE`：**本 kernel 不读**。
    //
    // 理由不是"没接线"，而是当前着色模型里没有落点：投影着色只有
    // `albedo·(ambient + Σ color·diff·intensity)`，既没有 roughness/metalness
    // 项（GGX 在 `prism_rt::bsdf`，尚未移植进本 kernel），也没有切线空间法线
    // 与自发光通道。硬接进来会得到"有代码、无效果"的假支持，故四个句柄里
    // 只有 MAP_BASE 参与着色；其余三个保持 ABI 槽位可读，行为与句柄 0 逐一相同。

    // 逐灯累加：`acc += albedo·(ambient + color·diff·intensity)`——**与 CPU 参考
    // 的循环体逐字同序**（先算 term，再乘 albedo，再累加）。
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var l: u32 = 0u;
    let num_lights = params[P_NUM_LIGHTS];
    loop {
      if (l >= num_lights) {
        break;
      }
      let ldir = light_dir(l);              // 方向已在 CPU 侧归一化（`encode_lights`）
      let lcolor = light_color(l);
      let lint = light_intensity(l);
      let diff = max(dot(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + vmul(albedo, term);
      l = l + 1u;
    }
    color = acc;
  } else if (curve_slot >= 0) {
    // 曲线命中：几何法线 + 固定中性反照率（与 CPU 参考逐式相同）。
    let n = tri_normal(u32(curve_slot), dir);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    var l: u32 = 0u;
    let num_lights = params[P_NUM_LIGHTS];
    loop {
      if (l >= num_lights) {
        break;
      }
      let ldir = light_dir(l);
      let lcolor = light_color(l);
      let lint = light_intensity(l);
      let diff = max(dot(n, ldir), 0.0);
      let term = ambient + lcolor * (diff * lint);
      acc = acc + vmul(CURVE_ALBEDO, term);
      l = l + 1u;
    }
    color = acc;
  }

  // ---- 后处理（与 CPU 参考同式，开关由 CPU 侧预算成 u32 标志） ----
  if (params[P_TEAL_ORANGE] != 0u) {
    let luma = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
    let shadow_tint = vec3<f32>(0.10, 0.35, 0.45);
    let high_tint = vec3<f32>(1.0, 0.7, 0.3);
    let shadow_comp = vmul(shadow_tint, color) * (1.0 - luma) * 0.5;
    let high_comp = vmul(high_tint, color) * luma * 0.5;
    color = color * 0.6 + shadow_comp + high_comp;
  }
  if (params[P_CONTRAST_ON] != 0u) {
    let c0 = bitcast<f32>(params[P_CONTRAST]);
    color = (color - vec3<f32>(0.5, 0.5, 0.5)) * c0 + vec3<f32>(0.5, 0.5, 0.5);
  }
  if (params[P_FILMIC] != 0u) {
    color = color / (color + vec3<f32>(1.0, 1.0, 1.0));
  }
  color = clamp01(color);
  outbuf[idx] = pack_rgba(color);
}
