// prism-gpu 毛发散射（U3 毛发深化 B）—— Marschner R/TT/TRT 三瓣 WGSL 实装。
//
// # 定位（GPU 侧着色器「库文件」，非 dispatch kernel）
// 本文件是 CPU 侧 `crates/prism-rt/src/hair_scatter.rs`（Marschner R/TT/TRT 三瓣）
// 的 **GPU 镜像**：只提供纯函数（M / N / A 衰减 / 三瓣合成），**不声明任何
// compute/vertex/fragment 入口**——因此登记在 `shaders::LIBRARIES`（库件），
// 与 `rt_scene.wgsl` 同类别，而非 `SHADERS`（kernel，逐份独立编译判对）。
// 它由未来的 hair 材质着色 kernel 通过 `//!include hair_scatter.wgsl` 复用；
// 也可被 rt_shade 的 BSDF 分支引用。
//
// # 【B 瓣未实现 —— 诚实边界，CPU/GPU 两侧文档一致】
// Marschner 完整模型是 A 瓣（R + TT + TRT，本文件已实装）+ **B 瓣**（沿发丝切向
// 散射，来自纤维横截面内的多次散射 / CS 散射）+ **C 瓣**（全局散射，需在毛发
// 网格之上的多次散射追踪）。**本文件与 `hair_scatter.rs` 均只含 A 瓣。**
// - B 瓣的物理形式是「横截面 CS 方位分布 ⊗ 纵向高斯」的**卷积**，
//   Melton & Shirley 2010 / Marschner 2003 §4.4 给出的是分段多项式拟合式，
//   其系数需对**实测 CS 方位分布**做最小二乘拟合——属**数据资产**（账本 U15
//   「论文系数属待自研化数据资产」），不是可以凭空写出的闭式。
// - 门禁 `wgsl_lint.mjs` 规则 **F13** 锁定「不得出现 B 瓣半成品桩」：
//   既不许出现 `a_b` / `b_lobe` 之类半成品函数体（返回常量/恒等式），
//   也不许在 CPU 侧出现 `TODO B 瓣` 之类"看起来在写"的占位。
// - 成本估计：B 瓣实装（系数拟合 + CPU/WGSL 双份 + 门禁 + 对拍）约 2–3 人日，
//   且**必须先有 CS 方位分布实测数据**，故本轮不做。详见汇报「诚实边界」。
//
// # 确定性设计（全链路确定性铁律）
// 1. 禁 libm 超越函数（D-29）：本文件不调用任何内建 sin/cos/exp/pow/log/
//    asin/atan，指数件用本文件的 `det_exp`（区间归约 + 固定系数多项式），
//    三角件复用 rt_common 的 `cos_sin`（Maclaurin + 象限对称，固定系数），
//    反正弦件用本文件的 `det_asin`（→ `det_atan` → `det_atan_small` 级数）。
// 2. 全部近似用**固定系数**的 Horner 求值，**不含任何循环**——循环在不同驱动上
//    展开/重排可能不同，会破坏确定性；Horner 的加乘顺序由源码字面固定。
// 3. 浮点运算顺序显式书写，不做结合律变换（`a*b + c*d` 不改成 `c*d + a*b`）。
// 4. **与 CPU 侧逐条同构**：`det_exp` / `bessel_i0` / `marschner_m` /
//    `marschner_n` 的表达式形状、常量、结合序都与 `hair_scatter.rs` 一致
//    （`sss::det_exp` 为指数件同源实现），使 CPU/GPU 对拍能收敛到 f32 舍入级。
//    这不是"风格偏好"：`2.0*b*b` 与 `2.0*(b*b)`、`x/LN2` 与 `x*(1/LN2)`
//    在 f32 下**不是同一个数**（实测 β=0.7 时 2β² 差 1 ulp），不统一则
//    对拍误差会被实现差异而非算法差异污染。
//
// # 公式出处
// - 单根毛发散射模型：Marschner, Jensen, Cammarano, Worley, Hanrahan,
//   "Light Scattering from Human Hair Fibers", SIGGRAPH 2003。
//   * 纵向散射 M（p.3 eq.4）：高斯，标准差 σ = β_m：
//       M(θ_i, θ_o) = (1 / (2 β_m²)) · exp[−(sinθ_o − sinθ_i)² / (2 β_m²)]
//   * 方位角散射 N（p.4 eq.6）：归一化修正贝塞尔项：
//       N(φ, k) = I₀(k·cos (φ/2))) / (2π · I₀(k/2)²)
//     （φ 为相对入射面的偏角；分母的 **k/2** 来自恒等式
//      `∫₀^{2π} I₀(k·cos(φ/2))dφ = 2π·I₀(k/2)²`，故 ∫N dφ = 1）
//   * 衰减项 A（p.4 §4.3 eq.9–10 的 A_p）：三瓣 R / TT / TRT 各自的 Fresnel 透/反射
//     与介质吸收（Marschner 2003 §3），见下方 a_r / a_tt / a_trt 与 schlick_fresnel。
//   * 三瓣独立方位/纵向偏角（p.4 §4.3 eq.9）：φ_R = φ、φ_TT = φ − π、φ_TRT = φ，
//     以及 α_p 纵向偏角；见下方 hair_phi_p / hair_alpha_p /
//     **hair_scatter_s_precise**（精确三瓣入口）。
// - Schlick 角依赖 Fresnel 近似：C. Schlick, "An Inexpensive BRDF Model for
//   Physically-based Rendering", Computer Graphics Forum 13(3), 1994：
//       F(θ) = f0 + (1 − f0)·(1 − cosθ)^5
//   （Marschner 2003 §4.3 原用精确电介质 Fresnel；本实装以 Schlick 近似替代）。
// - 修正贝塞尔函数 I₀ 近似：Abramowitz & Stegun, *Handbook of Mathematical
//   Functions*, 9.8.1 / 9.8.2（多项式近似，非级数求和）。
//   * 9.8.1（|x| ≤ 3.75）：I₀ 为 t=x/3.75 的 6 次偶多项式，|误差| < 1.6e-7。
//   * 9.8.2（x > 3.75）：I₀ ≈ x^(−1/2)·e^x·(8 项 t=3.75/x 的幂多项式)，
//     |误差| < 1.8e-8（x ≥ 3.75）。本文件写作 `det_exp(x) / sqrt(x) * P(3.75/x)`。
// - 反正弦/反正切：Macin 型级数（与 CPU `hair_scatter.rs::det_asin` 同式）。
//
// # 与 CPU 侧的一致性（判对基准）
// 公式与 `hair_scatter.rs` 的 Marschner 三瓣一致；本文件**不修改** CPU 文件。
// 数值对拍由 `web/host_test.mjs` 的「Marschner CPU/GPU 对拍」节执行：
// CPU 侧数字由 `cargo test -p prism-rt --release` 产出的 golden vector fixture
// （`web/hair_golden.json`，位模式十六进制）提供，GPU 侧数字由
// **SwiftShader/Chromium 上真跑**本文件的 `hair_scatter_s_precise` 取得，
// 逐分量比对 R / TT / TRT。

//!include rt_common.wgsl

const HAIR_TWO_PI: f32 = 6.283185307179586;
const HAIR_INV_TWO_PI: f32 = 0.15915494309189535; // 1 / (2π)

// ---------------------------------------------------------------------------
// 确定性 exp（区间归约 + 固定系数多项式，无循环；与 rt_common 的三角件同纪律）
// ---------------------------------------------------------------------------
//
// 设计：exp x = 2^k · exp r，r ∈ [−ln2/2, ln2/2]。
//   * k = round(x / ln2)。**必须写成除法 `x / LN_2`**，不得写成 `x * (1/LN2)`
//     —— 二者在 f32 下不等价（1/ln2 的舍入误差会被 x 放大），会污染 CPU/GPU
//     对拍。CPU `sss::det_exp` 同为除法形式。
//   * 2^k 用位构造：f32 指数偏置 127，`bitcast<f32>((127 + k) << 23)`。
//     这是**唯一**允许的整数位运算（门禁 F8 白名单只放行这一种 2^k 构造），
//     且 CPU `sss::det_exp` 用 `f32::from_bits` 做同一件事。
//   * exp r 用 7 阶 Taylor（Horner 到 1/7!，|r| ≤ 0.3466 → 误差 < 2e-10），
//     **项数与 CPU 侧一致**（CPU 也是 7 阶，多一项不多一项）。
//   * 下溢守卫：x ≤ −87 时单精度 exp 必下溢，直接返回 0 —— 否则 clamp 会把
//     指数域钉在 −126，破坏 Taylor 归约域使结果翻负。与 CPU 同口径。
//
// 该实现供 bessel_i0 大参数分支、M 瓣高斯与 Beer–Lambert 吸收共用。
fn det_exp(x: f32) -> f32 {
  const LN_2: f32 = 0.6931471805599453;
  // 下溢到零（exp 恒 > 0，绝不产出负值）。
  if (x <= -87.0) {
    return 0.0;
  }
  // k = round(x / ln2)；floor(z + 0.5) 对 z ≥ 0 为四舍五入，负值也正确。
  let kf = floor(x / LN_2 + 0.5);
  // 指数域钳位：保证 (127 + k) 落在 f32 正规数范围（1..254）。
  let k = clamp(kf, -126.0, 127.0);
  let r = x - k * LN_2;
  // exp r，r ∈ [−0.3466, 0.3466]：7 阶 Taylor（Horner，固定系数，无循环）。
  let e = 1.0
    + r * (1.0
    + r * (0.5
    + r * (1.0 / 6.0
    + r * (1.0 / 24.0
    + r * (1.0 / 120.0
    + r * (1.0 / 720.0
    + r * (1.0 / 5040.0)))))));
  // 2^k：f32 位模式 (127 + k) << 23（f32 指数偏置 127）。
  let scale = bitcast<f32>((i32(k) + 127) << 23);
  return e * scale;
}

// ---------------------------------------------------------------------------
// 确定性反三角件（asin / atan）—— 与 CPU hair_scatter.rs::det_asin 同式
// ---------------------------------------------------------------------------
//
// 为什么需要：α_p 纵向偏角修正要 arcsin(sinθ_i / η)（Marschner 2003 eq.9），
// 而 `asin` 是被禁的 libm 超越函数（D-29），故自带确定性实现。
//
// **无递归**：WGSL 禁止函数递归，而 CPU 的 `det_atan` 是自递归的
// （|x|>1 → π/2 − atan(1/x)）。这里拆成 `det_atan`（一次归约）+ `det_atan_unit`
// （|x| ≤ 1 的归约与级数选择）两层，展开后**与 CPU 的递归路径逐层等价**：
// CPU 递归进入第二层时 |1/x| ≤ 1，不可能再次触发 |·|>1 分支，故两层足够。
const HAIR_ATAN_CUT: f32 = 0.41421356; // 2^0.5 − 1
const HAIR_PI_2: f32 = 1.5707963267948966;

// |t| ≤ 0.414 上的 atan Maclaurin 级数（8 项，与 CPU det_atan_small 同式同项数）。
fn det_atan_small(t: f32) -> f32 {
  let t2 = t * t;
  return t * (1.0
    + t2 * (-1.0 / 3.0
    + t2 * (1.0 / 5.0
    + t2 * (-1.0 / 7.0
    + t2 * (1.0 / 9.0
    + t2 * (-1.0 / 11.0
    + t2 * (1.0 / 13.0
    + t2 * (-1.0 / 15.0
    + t2 * (1.0 / 17.0)))))))));
}

// atan 归约层（|x| ≤ 1）：大于切割点时用倍角公式 x/(1+√(1+x²)) 再乘 2。
fn det_atan_unit(x: f32) -> f32 {
  if (abs(x) > HAIR_ATAN_CUT) {
    let r = x / (1.0 + sqrt(1.0 + x * x));
    return 2.0 * det_atan_small(r);
  }
  return det_atan_small(x);
}

// 确定性 arctan：|x| > 1 时 atan(x) = sign·π/2 − atan(1/x)。
fn det_atan(x: f32) -> f32 {
  if (abs(x) > 1.0) {
    let s = select(-1.0, 1.0, x >= 0.0);
    return s * PI * 0.5 - det_atan_unit(1.0 / x);
  }
  return det_atan_unit(x);
}

// 确定性反正弦：asin(x) = atan(x / √(1 − x²))（|x| < 1）；端点取 ±π/2。
// 入参由调用方 clamp 到 [−1,1]（与 CPU det_asin 同口径）。
fn det_asin(s: f32) -> f32 {
  let x = clamp(s, -1.0, 1.0);
  if (abs(x) >= 1.0) {
    return select(-PI * 0.5, PI * 0.5, x >= 0.0);
  }
  let den = sqrt(1.0 - x * x);
  if (den < 1e-8) {
    return select(-PI * 0.5, PI * 0.5, x >= 0.0);
  }
  return det_atan(x / den);
}

// ---------------------------------------------------------------------------
// 修正贝塞尔函数 I₀（Abramowitz & Stegun 9.8.1 / 9.8.2；分段多项式，无循环）
// ---------------------------------------------------------------------------
//
// 相对误差（本实装实测，非仅抄表）：门禁 `wgsl_lint.mjs` 规则 F5 在 [0,20] 上
// 抽查 12 点（0,0.5,1,2,3,3.74,3.76,5,8,12,16,20），以高精度级数 I₀=Σ(x²/4)^m/(m!)²
// 为真值，实测**最大相对误差 1.08e-7 @ x=8**，远优于要求的 1e-4。
//   * A&S 9.8.1（|x| ≤ 3.75）表载 |误差| < 1.6e-7；9.8.2（x > 3.75）表载 < 1.8e-8，
//     但本实装大参数段用**自研 det_exp**（7 阶 Taylor，非 libm exp），故 x>3.75 的
//     实测误差（1.08e-7 量级）略高于原表载值——仍远低于 1e-4 要求，门禁已固化为断言。
//   * 两段在 x = 3.75 处连续（两式端点差 ~1e-7）。
//
// 【与 CPU 侧同构的两处细节】（不是风格，是数值等价性要求）：
//   1. 大参数段的自变量用 `t = 3.75 / ax`（**再乘 t**，不是 `1/(x/3.75)`），
//      且分母用 `sqrt(ax)` 做**除法**而非 `inverseSqrt` 内建——
//      `inverseSqrt` 在部分实现上是快速近似倒数平方根，精度不受 IEEE 约束，
//      会引入实现相关的偏差（违反确定性铁律）。CPU `det_bessel_i0` 同用除法。
//   2. 大参数段的 `sqrt` 与 `det_exp` 的**先后顺序**与 CPU 一致
//      （`det_exp(ax) / sqrt(ax) * poly`），乘法不重结合。
fn bessel_i0(x: f32) -> f32 {
  let ax = abs(x);
  if (ax < 3.75) {
    // 9.8.1：t = x / 3.75，I₀ = 偶多项式（t² 的 0..6 次）。
    let t = x / 3.75;
    let t2 = t * t;
    return 1.0
      + t2 * (3.5156229
      + t2 * (3.0899424
      + t2 * (1.2067492
      + t2 * (0.2659732
      + t2 * (0.0360768
      + t2 * 0.0045813)))));
  }
  // 9.8.2：I₀ = x^(−1/2)·e^x·P，P 为 t=3.75/x 的幂多项式。
  let t = 3.75 / ax;
  let p = 0.39894228
    + t * (0.01328592
    + t * (0.00225319
    + t * (-0.00157565
    + t * (0.00916281
    + t * (-0.02057706
    + t * (0.02635537
    + t * (-0.01647633
    + t * 0.00392377)))))));
  return det_exp(ax) / sqrt(ax) * p;
}

// ---------------------------------------------------------------------------
// 纵向散射 M（Marschner 2003, eq.4）
// ---------------------------------------------------------------------------
//
// M(θ_i, θ_o) = (1 / (2 β_m²)) · exp(−(sinθ_o − sinθ_i)² / (2 β_m²))
//   * β_m 为纵向粗糙度（弧度）；归一化常数 1/(2β_m²) 为论文 eq.4 前置系数。
//   * β_m 下界 BETA_MIN = 1e-4（与 CPU `BETA_MIN` 同值），避免 0 除（C18）。
//   * sin 用确定性 cos_sin（rt_common），避免 libm（D-29）。
//   * `2.0 * bm2` 的写法与 CPU `m_longitudinal_full` 一致（bm2 = bm*bm 先算）。
fn marschner_m(theta_i: f32, theta_o: f32, beta_m: f32) -> f32 {
  let bm = max(beta_m, 1e-4);
  let bm2 = bm * bm;
  let si = cos_sin(theta_i).y;  // sin(θ_i)
  let so = cos_sin(theta_o).y;  // sin(θ_o)
  let d = so - si;
  return (1.0 / (2.0 * bm2)) * det_exp(-(d * d) / (2.0 * bm2));
}

// ---------------------------------------------------------------------------
// 方位角散射 N（Marschner 2003, eq.6）
// ---------------------------------------------------------------------------
//
// N(φ, k) = I₀(k·cos(φ/2)) / (2π · I₀(k/2)²)
//   * φ 为出射方向相对入射面的相对方位角；k 为方位角粗糙度参数。
//   * **分母的 `kk` 是 k/2**（不是 k）：归一性来自修正 Bessel 乘积恒等式
//     `∫₀^{2π} I₀(k·cos(φ/2))dφ = 2π·I₀(k/2)²`，故 ∫₀^{2π} N dφ = 1。
//     CPU `n_azimuth` 同为 `I₀(k*0.5)²`——两侧必须同，否则 N 会差一个
//     常数因子（I₀(k)²/I₀(k/2)² 在 k=8 时约 1430 倍），头发会整体发黑。
fn marschner_n(phi: f32, k: f32) -> f32 {
  let kh = max(k, 1e-4);
  let kk = 0.5 * kh;  // 归一化自变量 = k/2
  let cos_half = cos_sin(0.5 * phi).x;  // cos(φ/2)
  let num = bessel_i0(kh * cos_half);
  let den = HAIR_TWO_PI * bessel_i0(kk) * bessel_i0(kk);
  return num / max(den, 1e-12);
}

// ---------------------------------------------------------------------------
// 三瓣衰减项 A（R / TT / TRT）
// ---------------------------------------------------------------------------
//
// 依 Marschner et al. 2003 §4.3（eq.9–10 的 A_p）与 §3（介质吸收）：
//   * Fresnel 改用 **Schlick 角依赖**（Schlick 1994）：
//       F(θ) = f0 + (1 − f0)·(1 − cosθ)^5
//     （论文原用精确电介质 Fresnel；本实装以 Schlick 近似替代，如实标注）。
//   * 吸收为 Beer–Lambert：T = det_exp(−σ_a · d)，d 为纤维内路径长度
//     （d_tt = 2·r / cosθ_i 的平板近似，d_trt = 2·d_tt）。
//   * 三瓣衰减：
//       a_r   = F(θ_i)
//       a_tt  = (1 − F(θ_i))² · T_tt
//       a_trt = (1 − F(θ_i))² · F(θ_i) · T_trt
//
// # 【签名设计：宿主预算 F 与 T，着色器只做 A 的代数组合】
// `a_r` / `a_tt` / `a_trt` 的入参是**宿主已算好的 F(θ_i) 与 T_tt / T_trt**
// （由 hair_scatter_s_precise 或调用方 kernel 预算）。这样做的原因：
//   1. **可编译性**：早期版本让 a_* 收 (theta_i, f0, sigma_a, radius) 却在
//      hair_scatter_s 里按 (f0) / (f0, absorb_tt) 调用——**实参与形参个数不匹配，
//      该 WGSL 根本无法通过 createShaderModule**。因为本文件登记在
//      `LIBRARIES`（无入口、不参与 kernel 编译），这个缺陷一直"绿灯"。
//      现在签名与调用点严格一致，且门禁 F14 会**逐个函数核对实参个数**。
//   2. F 与 T 只依赖入射角与材质参数（与出射角无关），在片元里预算一次即可，
//      三个瓣共用，避免重复求值（也避免三瓣各自重算导致 f32 结果不一致）。
// 全部实现为**固定系数 Horner / 直写算术，无循环**（确定性铁律）。
fn clamp01(x: f32) -> f32 {
  return clamp(x, 0.0, 1.0);
}

// Schlick 角依赖 Fresnel：F = f0 + (1−f0)·(1−cosθ)^5（Schlick 1994）。
fn schlick_fresnel(f0: f32, cos_theta: f32) -> f32 {
  let f = clamp01(f0);
  let c = clamp01(cos_theta);
  let x = 1.0 - c;
  let x2 = x * x;
  let x5 = x2 * x2 * x;
  return clamp01(f + (1.0 - f) * x5);
}

// R 瓣衰减：a_r = F(θ_i)（单次 Fresnel 反射；R 瓣不穿纤维内壁，无吸收项）。
fn a_r(f0: f32) -> f32 {
  return clamp01(f0);
}
// TT 瓣衰减：a_tt = (1−F(θ_i))²·T_tt（两次折射透射 + Beer 吸收）。
// `f0` 语义为 F(θ_i)（角依赖 Fresnel 项），`absorb_tt` 为 T_tt = exp(−σ_a·d_tt)。
fn a_tt(f0: f32, absorb_tt: f32) -> f32 {
  let f = clamp01(f0);
  let t = 1.0 - f;
  return clamp01(t * t * absorb_tt);
}
// TRT 瓣衰减：a_trt = (1−F(θ_i))²·F(θ_i)·T_trt（透射×2 + 内反射 + 更强吸收）。
fn a_trt(f0: f32, absorb_trt: f32) -> f32 {
  let f = clamp01(f0);
  let t = 1.0 - f;
  return clamp01(t * t * f * absorb_trt);
}

// Beer–Lambert 透射：T = exp(−σ_a·d)。d ≤ 0 或 σ_a ≤ 0 → 1（无吸收，C18）。
fn det_exp_neg(x: f32) -> f32 {
  if (x <= 0.0) {
    return 1.0;
  }
  return det_exp(-x);
}

// ---------------------------------------------------------------------------
// 三瓣偏角（Marschner 2003 §4.3 eq.9）—— CPU hair_phi_p / hair_alpha_p 的镜像
// ---------------------------------------------------------------------------

// 方位偏角 φ_p：R 与 TRT 无偏移（φ）；TT 因透射使入射面翻转 π（φ − π）。
fn hair_phi_p(phi: f32, p: u32) -> f32 {
  if (p == 1u) {
    return phi - PI;
  }
  return phi;
}

// 纵向偏角 α_p：α = ±arcsin(sinθ_i / η)。R 取负号（单次反射横向位移反向），
// TT / TRT 取正号（论文一阶近似，见 CPU hair_alpha_p 的公式编号注释）。
// η ≤ 1（无折射）→ 0（C18 兜底，与 CPU 一致）。
fn hair_alpha_p(theta_i: f32, eta: f32, p: u32) -> f32 {
  if (eta <= 1.0) {
    return 0.0;
  }
  let s = cos_sin(theta_i).y;  // sinθ_i
  let a = det_asin(clamp(s / eta, -1.0, 1.0));
  if (p == 0u) {
    return -a;
  }
  return a;
}

// 垂直入射 Fresnel 反射率 f0 = ((η−1)/(η+1))²（Born & Wolf；CPU hair_f0 镜像）。
fn hair_f0(eta: f32) -> f32 {
  let d = eta + 1.0;
  if (abs(d) < 1e-12) {
    return 0.0;
  }
  let r0 = (eta - 1.0) / d;
  return clamp01(r0 * r0);
}

// ---------------------------------------------------------------------------
// 完整散射 S = M · N · A（返回 R / TT / TRT 三瓣贡献）
// ---------------------------------------------------------------------------
//
// # 两个入口，语义不同，勿混用
//
// ## `hair_scatter_s` —— 共享 M/N 的**简化**三瓣入口
// 三瓣共用同一 M（纵向）与 N（方位角），仅衰减 A 不同。这是对 Marschner
// 论文的**简化**（论文中各瓣的 φ 偏角不同，见下）。保留此入口是因为门禁
// F6 / Rust 测试 `hair_scatter_operation_order_is_m_then_n_then_a` 锁定了它的
// 运算顺序契约 `let mn = m * n;` + `mn * a_*`。
// 入参 `f0` 语义为 **F(θ_i)**（角依赖 Fresnel 项，非垂直入射率）。
fn hair_scatter_s(theta_i: f32, theta_o: f32, phi: f32,
                  beta_m: f32, k: f32,
                  f0: f32, absorb_tt: f32, absorb_trt: f32) -> vec3<f32> {
  let m = marschner_m(theta_i, theta_o, beta_m);
  let n = marschner_n(phi, k);
  let mn = m * n;
  let s_r = mn * a_r(f0);
  let s_tt = mn * a_tt(f0, absorb_tt);
  let s_trt = mn * a_trt(f0, absorb_trt);
  return vec3<f32>(s_r, s_tt, s_trt);
}

// ## `hair_scatter_s_precise` —— **精确三瓣入口（CPU/GPU 对拍基准）**
// 与 CPU `HairScatter::components_precise` **逐条同构**：
//   S_p = M(θ_i, θ_o − α_p) · N(φ_p) · A_p，p ∈ {R(0), TT(1), TRT(2)}
// 其中 φ_p = hair_phi_p(φ_o − φ_i, p)、α_p = hair_alpha_p(θ_i, η, p)、
// F(θ_i) = schlick_fresnel(f0(η), |cosθ_i|)、T_tt/T_trt 由 Beer–Lambert 预算
// （d_tt = 2r/|cosθ_i|，d_trt = 2·d_tt）。
// 与简化入口的**关键差异**：每瓣各有自己的 M（α 偏角修正）与 N（φ 偏角），
// 这正是 Marschner 2003 eq.9 的三瓣分离——CPU 侧早已实装，GPU 侧本轮补齐。
// 运算顺序固定：先 M、再乘 N、最后乘 A，避免不同驱动的重结合。
fn hair_scatter_s_precise(theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32,
                          beta_m: f32, k: f32, eta: f32,
                          sigma_a: f32, radius: f32) -> vec3<f32> {
  // 方位偏差（出射 − 入射）。
  let phi = phi_o - phi_i;
  // 三瓣纵向偏角 α_p（Marschner 2003 eq.9）。
  let a0 = hair_alpha_p(theta_i, eta, 0u);
  let a1 = hair_alpha_p(theta_i, eta, 1u);
  let a2 = hair_alpha_p(theta_i, eta, 2u);
  // 三瓣方位项（φ_R = φ、φ_TT = φ − π、φ_TRT = φ）。
  let n_r = marschner_n(hair_phi_p(phi, 0u), k);
  let n_tt = marschner_n(hair_phi_p(phi, 1u), k);
  let n_trt = marschner_n(hair_phi_p(phi, 2u), k);
  // 三瓣纵向项（用 α 修正后的出射纵向角）。
  let m_r = marschner_m(theta_i, theta_o - a0, beta_m);
  let m_tt = marschner_m(theta_i, theta_o - a1, beta_m);
  let m_trt = marschner_m(theta_i, theta_o - a2, beta_m);
  // 三瓣衰减：Schlick 角依赖 Fresnel + Beer–Lambert 吸收。
  let cos_i = abs(cos_sin(theta_i).x);
  let f = schlick_fresnel(hair_f0(eta), cos_i);
  let csafe = max(cos_i, 1e-3);
  let d_tt = 2.0 * radius / csafe;
  let t_tt = det_exp_neg(sigma_a * d_tt);
  let t_trt = det_exp_neg(sigma_a * (2.0 * d_tt));
  let s_r = m_r * n_r * a_r(f);
  let s_tt = m_tt * n_tt * a_tt(f, t_tt);
  let s_trt = m_trt * n_trt * a_trt(f, t_trt);
  return vec3<f32>(s_r, s_tt, s_trt);
}

// 单瓣求和辅助：三瓣标量之和（供 BSDF 直接取亮度）。
fn hair_scatter_total(s: vec3<f32>) -> f32 {
  return s.x + s.y + s.z;
}
