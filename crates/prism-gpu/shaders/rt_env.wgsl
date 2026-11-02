// rt_env.wgsl —— A2 / W2：等距柱状环境贴图的采样、pdf、双线性取值（WGSL 版）。
//
// # 与 CPU 权威的关系（口径 b）
//
// 本文件是 `crates/prism-rt/src/env.rs` 的**逐式镜像**，不是"另写一版差不多的"：
// * 方向 ↔ uv：`u = atan2(z,x)/(2π)+0.5`（平铺）、`v = acos(y)/π`；
// * texel 内**立体角均匀**：`cosθ` 在行内线性、`φ` 均匀（`sinθ` 由 `1−c²` 反解）；
// * 反演：u32 定点前缀和（`u32::MAX` 归一）+ 二分（`prefix[k+1] > target`）；
// * 两级拆分：亮斑列表（`p_spot` 混合）与残差 CDF（含亮斑的行读压缩表）；
// * pdf：`p_texel / Ω_texel`，CPU 与 WGSL 同源同式。
//
// # 为什么不在这里声明 @group/@binding
//
// blob 由调用方以 **F32 存储缓冲**传入并按 `array<f32>` 声明（u32 用
// `bitcast<u32>` 取回）：宿主侧 `KernelDesc` 只按字节布局建缓冲，元素类型
// 只影响描述符；用 f32 承载 u32 位模式可以走"单输入"的既有 dispatch 形状
// （`VulkanBackend::compute_dispatch`），不必为验证再造一条 ABI。
//
// # blob 布局（**唯一权威在 CPU 侧的打包函数**，见 `qa_w2_env_gpu_parity.rs`）
//
//   [ 0] w                     [ 1] h
//   [ 2] spot_count            [ 3] p_spot（f32 位模式）
//   [ 4] omega_total（f32）    [ 5] has_energy（0/1）
//   [ 6] has_residual（0/1）   [ 7] cond_len      = (w+1)·h
//   [ 8] marg_len = h+1        [ 9] res_cond_len
//   [10] res_row_off_len = h   [11] spot_prefix_len = spot_count+1
//   [12] spot_texels_len       [13] texel_words = 4·w·h
//   [14] cond_off              [15] marg_off
//   [16] res_cond_off          [17] res_row_off_off
//   [18] spot_prefix_off       [19] spot_texels_off
//   [20] texels_off            [21..23] 保留
//
// 所有 off 都是**相对 blob 起点的 f32 元素下标**（= 字下标）。
//
// # D-29（禁 libm）
//
// 方向 ↔ uv 的 `atan2` / `acos`、行立体角的 `cos`、texel 内的 `cos/sin`
// **一律走确定性三角件**：`cos_sin` 来自 `rt_common.wgsl`（与
// `prism_math::det_trig::cos_sin` 同一构造：mod 2π → 象限 → π/4 级数），
// `env_atan2` / `env_acos` 是 `prism_math::det_trig::{atan2, acos}` 的逐式镜像
// （同一组固定系数、同一归约顺序）。理由不是洁癖：V6 对拍实测 lavapipe 的
// 内建 `acos` 在 `v = acos(y)/π` 上偏 4.96e-5，比 `det_trig::acos` 差三个数量级。

//!include rt_common.wgsl

const ENV_HDR: u32 = 24u;
const ENV_PI: f32 = 3.14159265358979323846;
const ENV_TAU: f32 = 6.28318530717958647692;
const ENV_FRAC_1_PI: f32 = 0.31830988618379067154;
const ENV_U32_MAX_F: f32 = 4294967296.0; // u32::MAX as f32（向上舍入到 2^32）
const ENV_INF_TEXEL: u32 = 0xFFFFFFFFu;  // res_row_off 的"无亮斑"哨兵
// `prism_math::det_trig::{atan2, acos}` 的固定系数（逐字拷贝，不许"约等于"）。
const ENV_TAN_PI_8: f32 = 0.41421357;
const ENV_FRAC_PI_2: f32 = 1.5707963267948966;
const ENV_FRAC_PI_4: f32 = 0.7853981633974483;

// # blob 绑定由**引用方**声明（W2 剩余 / wavefront 接线时改）
//
// 本文件**不再**自带 `@group/@binding`：`rt_env_probe.wgsl` 把它挂在
// `@group(0) @binding(0)`，而 `rt_wavefront.wgsl` 的 group(0) 已被
// {slots, params, accum} 占满（宿主把句柄数组第 i 个硬绑到 `binding: i`，
// 且每个 entry 的绑定集必须是**连续前缀**），只能把 env blob 挂到
// `@group(2) @binding(5)`（rt_scene 的场景组尾部）。
//
// 于是"同一份 CDF 语义"靠的是**函数体**共享（本文件），而不是绑定号共享：
// 每个引用方各自声明 `var<storage, read> env_blob: array<f32>;`（名字必须
// 叫 `env_blob`，本文件的取数函数按此符号解析），采样/pdf/双线性一律走
// 下面这一份实现。**禁止**为某个 kernel 另写一套近似。
//
// # 布局（同上，唯一权威在 CPU 打包侧）
//
// 见文件头与 `crates/prism-rt/tests/qa_w2_env_gpu_parity.rs` 的打包函数。

// ---------------------------------------------------------------------------
// 确定性反三角（`det_trig` 的逐式镜像；WGSL 禁递归故把一层归约展开）
// ---------------------------------------------------------------------------

/// `[0, tan(π/8)]` 上的 `atan`（固定系数 Maclaurin，Horner 排布）。
fn env_quarter_atan(z: f32) -> f32 {
  let z2 = z * z;
  return z * (1.0
    + z2 * (-1.0 / 3.0
      + z2 * (1.0 / 5.0
        + z2 * (-1.0 / 7.0
          + z2 * (1.0 / 9.0
            + z2 * (-1.0 / 11.0
              + z2 * (1.0 / 13.0 + z2 * (-1.0 / 15.0))))))));
}

/// `atan(x)`，`x >= 0` → `[0, π/2]`。
///
/// `det_trig::atan_pos` 在 `x > 1` 处只递归一层（`1/x < 1`），这里把那一层
/// **展开写**：WGSL 不允许递归调用。
fn env_atan_pos(x: f32) -> f32 {
  if (x > 1.0) {
    let xi = 1.0 / x; // xi ∈ (0, 1)
    if (xi > ENV_TAN_PI_8) {
      return ENV_FRAC_PI_2 - (ENV_FRAC_PI_4 + env_quarter_atan((xi - 1.0) / (xi + 1.0)));
    }
    return ENV_FRAC_PI_2 - env_quarter_atan(xi);
  }
  if (x > ENV_TAN_PI_8) {
    return ENV_FRAC_PI_4 + env_quarter_atan((x - 1.0) / (x + 1.0));
  }
  return env_quarter_atan(x);
}

/// 确定性 `atan2`：`det_trig::atan2` 的逐式镜像（`(−π, π]`，特例同 IEEE）。
fn env_atan2(y: f32, x: f32) -> f32 {
  // 非有限（NaN / ±∞）→ 0：与 CPU `!y.is_finite() || !x.is_finite()` 同口径。
  if (!(abs(y) < 3.4e38) || !(abs(x) < 3.4e38)) {
    return 0.0;
  }
  if (x == 0.0) {
    if (y == 0.0) {
      return 0.0;
    }
    if (y > 0.0) {
      return ENV_FRAC_PI_2;
    }
    return -ENV_FRAC_PI_2;
  }
  let ax = abs(x);
  let ay = abs(y);
  // r = atan(|y/x|) 的稳定形式（大者作分母），与 CPU 同分支。
  var r = 0.0;
  if (ay <= ax) {
    r = env_atan_pos(ay / ax);
  } else {
    r = ENV_FRAC_PI_2 - env_atan_pos(ax / ay);
  }
  if (x > 0.0) {
    if (y >= 0.0) {
      return r;
    }
    return -r;
  }
  if (y >= 0.0) {
    return ENV_PI - r;
  }
  return r - ENV_PI;
}

/// 确定性 `acos`：`det_trig::acos` 的逐式镜像（`atan2(√((1−x)(1+x)), x)`）。
fn env_acos(x: f32) -> f32 {
  if (!(abs(x) < 3.4e38)) {
    return 0.0; // 非有限 → 0（同 CPU）
  }
  let xc = clamp(x, -1.0, 1.0);
  let s = sqrt(max((1.0 - xc) * (1.0 + xc), 0.0));
  return env_atan2(s, xc);
}

// ---------------------------------------------------------------------------
// blob 取数
// ---------------------------------------------------------------------------

fn env_u(i: u32) -> u32 {
  return bitcast<u32>(env_blob[i]);
}

fn env_f(i: u32) -> f32 {
  return env_blob[i];
}

fn env_w() -> u32 {
  return env_u(0u);
}

fn env_h() -> u32 {
  return env_u(1u);
}

// ---------------------------------------------------------------------------
// 映射（与 env.rs::dir_to_uv / uv_to_dir 同式）
// ---------------------------------------------------------------------------

fn env_dir_to_uv(d: vec3<f32>) -> vec2<f32> {
  var u = env_atan2(d.z, d.x) * (ENV_FRAC_1_PI * 0.5) + 0.5;
  u = u - floor(u);
  if (!(u < 1.0)) {
    u = 0.0; // NaN / 恰为 1.0 的防御（与 CPU 同口径）
  }
  let v = env_acos(clamp(d.y, -1.0, 1.0)) * ENV_FRAC_1_PI;
  return vec2<f32>(u, clamp(v, 0.0, 1.0));
}

fn env_uv_to_dir(u: f32, v: f32) -> vec3<f32> {
  let theta = clamp(v, 0.0, 1.0) * ENV_PI;
  let phi = ENV_TAU * (u - 0.5);
  let ct = cos_sin(theta);
  let cp = cos_sin(phi);
  // `cos_sin` 返回 `(cos, sin)`（与 `det_trig::cos_sin` 同序）。
  return vec3<f32>(ct.y * cp.x, ct.x, ct.y * cp.y);
}

/// 行 `j` 的立体角（精确球冠：`2π(cosθ_j − cosθ_{j+1})`）。
fn env_row_omega(j: u32) -> f32 {
  let h = max(env_h(), 1u);
  let c0 = cos_sin(ENV_PI * f32(j) / f32(h)).x;
  let c1 = cos_sin(ENV_PI * f32(j + 1u) / f32(h)).x;
  return ENV_TAU * (c0 - c1);
}

/// texel `(i, j)` 内的采样方向（`cosθ` 行内均匀、`φ` 均匀 ⇒ 立体角均匀）。
fn env_texel_dir(j: u32, i: u32, fv: f32, fu: f32) -> vec3<f32> {
  let w = f32(max(env_w(), 1u));
  let h = f32(max(env_h(), 1u));
  let c0 = cos_sin(ENV_PI * f32(j) / h).x;
  let c1 = cos_sin(ENV_PI * f32(j + 1u) / h).x;
  let c = clamp(c0 + (c1 - c0) * clamp(fv, 0.0, 1.0), -1.0, 1.0);
  let s = sqrt(max(1.0 - c * c, 0.0));
  let u = (f32(i) + clamp(fu, 0.0, 1.0)) / w;
  let phi = ENV_TAU * (u - 0.5);
  let cp = cos_sin(phi);
  return vec3<f32>(s * cp.x, c, s * cp.y);
}

// ---------------------------------------------------------------------------
// u32 定点 CDF 的二分反演（与 env.rs::pick_u32 同式）
// ---------------------------------------------------------------------------

/// 返回 `(下标, 区间内分数)`；`off` 是前缀和数组在 blob 内的起点，`n` 是区间数。
fn env_pick(off: u32, n: u32, u: f32) -> vec2<f32> {
  if (n == 0u) {
    return vec2<f32>(0.0, 0.0);
  }
  let tgt = u32(clamp(u, 0.0, 1.0) * ENV_U32_MAX_F);
  var lo = 0u;
  var hi = n;
  loop {
    if (lo >= hi) {
      break;
    }
    let mid = (lo + hi) / 2u;
    if (env_u(off + mid + 1u) > tgt) {
      hi = mid;
    } else {
      lo = mid + 1u;
    }
  }
  let k = min(lo, n - 1u);
  let a = env_u(off + k);
  let b = env_u(off + k + 1u);
  var frac = 0.0;
  if (b > a) {
    frac = f32(tgt - a) / f32(b - a);
  }
  return vec2<f32>(f32(k), frac);
}

// ---------------------------------------------------------------------------
// 双线性辐亮度（不做预过滤）
// ---------------------------------------------------------------------------

fn env_texel_rgb(texel: u32) -> vec3<f32> {
  let base = env_u(20u) + 4u * texel;
  return vec3<f32>(env_f(base), env_f(base + 1u), env_f(base + 2u));
}

fn env_radiance(dir: vec3<f32>) -> vec3<f32> {
  let wi = env_w();
  let hi = env_h();
  if (wi == 0u || hi == 0u) {
    return vec3<f32>(0.0);
  }
  let uv = env_dir_to_uv(dir);
  let x = uv.x * f32(wi) - 0.5;
  let y = uv.y * f32(hi) - 0.5;
  let x0 = floor(x);
  let y0 = floor(y);
  let fx = x - x0;
  let fy = y - y0;
  // u 平铺（跨缝取邻居），v 钳制（极点不是环绕）——与 CPU 同口径。
  let i0 = (u32(i32(x0) % i32(wi) + i32(wi)) % wi);
  let i1 = (i0 + 1u) % wi;
  let j0 = u32(clamp(i32(y0), 0, i32(hi) - 1));
  let j1 = u32(clamp(i32(y0) + 1, 0, i32(hi) - 1));
  let t00 = env_texel_rgb(j0 * wi + i0);
  let t10 = env_texel_rgb(j0 * wi + i1);
  let t01 = env_texel_rgb(j1 * wi + i0);
  let t11 = env_texel_rgb(j1 * wi + i1);
  // 固定结合序（先 u 后 v），与 CPU 逐式一致。
  let a = t00 + (t10 - t00) * fx;
  let b = t01 + (t11 - t01) * fx;
  return a + (b - a) * fy;
}

// ---------------------------------------------------------------------------
// pdf（与采样严格同源）
// ---------------------------------------------------------------------------

/// 亮斑列表里的 texel → 亮斑下标（线性扫描；小列表；与 CPU `spot_index_of` 同义）。
fn env_spot_index(texel: u32) -> u32 {
  let n = env_u(2u);
  let off = env_u(19u);
  var k = 0u;
  var found = ENV_INF_TEXEL;
  while (k < n) {
    if (env_u(off + k) == texel) {
      found = k;
      break;
    }
    k = k + 1u;
  }
  return found;
}

fn env_pdf(dir: vec3<f32>) -> f32 {
  if (env_u(5u) == 0u) {
    return 0.0; // 总能量为 0 ⇒ 永不采到（与 CPU 同口径）
  }
  let w = env_w();
  let h = env_h();
  if (w == 0u || h == 0u) {
    return 0.0;
  }
  let uv = env_dir_to_uv(dir);
  let i = min(u32(uv.x * f32(w)), w - 1u);
  let j = min(u32(uv.y * f32(h)), h - 1u);
  let omega = env_row_omega(j) / f32(w); // texel 立体角 = Ω_row / W（与 CPU 同式）
  if (!(omega > 0.0)) {
    return 0.0;
  }
  let texel = j * w + i;
  let k = env_spot_index(texel);
  if (k != ENV_INF_TEXEL) {
    let sp = env_u(18u);
    let den = env_u(sp + env_u(2u));
    if (den == 0u) {
      return 0.0;
    }
    let p = env_u(sp + k + 1u) - env_u(sp + k);
    return env_f(3u) * (f32(p) / f32(den)) / omega;
  }
  if (env_u(6u) == 0u) {
    return 0.0;
  }
  let mg = env_u(15u);
  let lo_m = env_u(mg + j);
  let hi_m = env_u(mg + j + 1u);
  if (hi_m <= lo_m) {
    return 0.0;
  }
  let p_row = f32(hi_m - lo_m) / ENV_U32_MAX_F;
  var row_off = env_u(14u) + j * (w + 1u);
  let roff = env_u(env_u(17u) + j); // 两级：先取表起点，再取第 j 项
  if (roff != ENV_INF_TEXEL) {
    row_off = env_u(16u) + roff;
  }
  let lo_c = env_u(row_off + i);
  let hi_c = env_u(row_off + i + 1u);
  if (hi_c <= lo_c) {
    return 0.0;
  }
  let p_col = f32(hi_c - lo_c) / ENV_U32_MAX_F;
  return (1.0 - env_f(3u)) * p_row * p_col / omega;
}

// ---------------------------------------------------------------------------
// 采样（两级混合）
// ---------------------------------------------------------------------------

struct EnvSampleW {
  wi: vec3<f32>,
  pdf: f32,
  texel: u32,
}

fn env_sample(u1: f32, u2: f32) -> EnvSampleW {
  var out: EnvSampleW;
  out.wi = vec3<f32>(0.0, 1.0, 0.0);
  out.pdf = 0.0;
  out.texel = 0u;
  let w = env_w();
  let h = env_h();
  if (env_u(5u) == 0u || w == 0u || h == 0u) {
    return out; // 全黑环境
  }
  let uu1 = clamp(u1, 0.0, 1.0);
  let uu2 = clamp(u2, 0.0, 1.0);
  let p_spot = env_f(3u);
  let has_spots = env_u(2u) > 0u && p_spot > 0.0;
  let has_res = env_u(6u) != 0u;
  if (has_spots && (!has_res || uu1 < p_spot)) {
    var us = uu1;
    if (p_spot < 1.0) {
      us = uu1 / p_spot;
    }
    let kf = env_pick(env_u(18u), env_u(2u), us);
    let k = u32(kf.x);
    let sp = env_u(18u);
    let den = env_u(sp + env_u(2u));
    let texel = env_u(env_u(19u) + k);
    let i = texel % w;
    let j = texel / w;
    let omega = env_row_omega(j) / f32(w); // texel 立体角 = Ω_row / W（与 CPU 同式）
    let dir = env_texel_dir(j, i, kf.y, uu2);
    out.wi = dir;
    out.texel = texel;
    if (den > 0u && omega > 0.0) {
      let p = env_u(sp + k + 1u) - env_u(sp + k);
      out.pdf = p_spot * (f32(p) / f32(den)) / omega;
    }
    return out;
  }
  var ur = uu1;
  if (p_spot >= 1.0) {
    ur = uu1;
  } else {
    ur = (uu1 - p_spot) / (1.0 - p_spot);
  }
  let mg = env_u(15u);
  let jf = env_pick(mg, h, ur);
  let j = min(u32(jf.x), h - 1u);
  var row_off = env_u(14u) + j * (w + 1u);
  let roff = env_u(env_u(17u) + j); // 两级：先取表起点，再取第 j 项
  if (roff != ENV_INF_TEXEL) {
    row_off = env_u(16u) + roff;
  }
  let cf = env_pick(row_off, w, uu2);
  let i = min(u32(cf.x), w - 1u);
  let omega = env_row_omega(j) / f32(w); // texel 立体角 = Ω_row / W（与 CPU 同式）
  let dir = env_texel_dir(j, i, jf.y, cf.y);
  out.wi = dir;
  out.texel = j * w + i;
  if (omega > 0.0) {
    let lo_m = env_u(mg + j);
    let hi_m = env_u(mg + j + 1u);
    if (hi_m > lo_m) {
      let p_row = f32(hi_m - lo_m) / ENV_U32_MAX_F;
      let lo_c = env_u(row_off + i);
      let hi_c = env_u(row_off + i + 1u);
      if (hi_c > lo_c) {
        let p_col = f32(hi_c - lo_c) / ENV_U32_MAX_F;
        out.pdf = (1.0 - p_spot) * p_row * p_col / omega;
      }
    }
  }
  return out;
}
