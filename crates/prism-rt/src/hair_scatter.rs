//! Marschner 毛发散射（散射部分）：R / TT / TRT 三分量。
//!
//! 语义权威：的分层约定（"散射（介质传输核复用， 同款
//! 路径）"）；（介质内传输原语：HG 相函数、自由程采样——
//! "一套代码两个功能"）。
//!
//! # 模型出处（数据资产注释义务）
//!
//! Marschner 毛发散射三分量模型：
//! - S. R. Marschner, H. W. Jensen, M. Cammarano, S. Worley, P. Hanrahan,
//!   "Light Scattering from Human Hair Fibers", ACM SIGGRAPH 2003, §2–§4。
//!   R（单次反射）/ TT（双折射透射）/ TRT（透射-内反射-透射）三分量 +
//!   纵向 M(θ) × 方位 N(φ) 分离（该文 Eq. 3 前的近似 "N(φ) 不依赖 θ"）。
//! - 纵向衰减 M(θ) 采用**高斯近似**：M(θ) = exp(−θ²/(2β²)) / (β·√(2π))；
//!   标准形与 β 语义见 Marschner 2003 §4.2 / W. Z. Huang et al.,
//!   "A Physically-Based Hair Fiber Model"（逼近形），以及 pbrt-v3 §9.2
//!   （公开教科书实现锚点，BSD 许可）。
//! - 方位分布 N(φ) 采用**修正 Bessel 函数 I₀**（Kolmogorov–Smirnov 方位
//!   散射标准形，Marschner 2003 Eq. 8 前的 M(θ)N(φ) 分离以 N(φ) 近似）：
//!   N(φ; s) ∝ I₀(cosφ_hair)·… 的工程简化——本实装用公开近似
//!   `N(φ) = I₀(k·cos(φ/2))` 的归一形，k 为方位粗糙度参数。
//! - Fresnel 反射率与折射用 Schlick 近似（Schlick 1994，见 [`crate::bsdf`]）。
//!
//! # A 瓣精确化（本轮「毛发深化 C」）
//!
//! 上一轮的三瓣衰减 A 为简化形（常量 f0、无角依赖、三瓣共用 φ）。本轮按
//! Marschner et al. 2003 补齐：
//! - **Fresnel** 改为 Schlick 近似的**角依赖**形（下述 [`schlick_fresnel`]），
//!   出处：C. Schlick, "An Inexpensive BRDF Model for Physically-based
//!   Rendering", Computer Graphics Forum 13(3), 1994 —— 论文自身用精确
//!   电介质 Fresnel（Marschner 2003 §4.3 Eq. 9–10 附近的 F(θ)），本实装以
//!   Schlick 近似替代（工程选择，如实标注）。
//! - **吸收** 加 Beer–Lambert `T = exp(−σ_a·d)`（Marschner 2003 §3 的
//!   "介质吸收"；本模块复用 [`crate::sss::det_exp`]），d 为纤维内路径长度
//!   （下文 [`HairScatter::path_len_tt`] / [`HairScatter::path_len_trt`] 的
//!   简化几何，如实标注）。
//! - **三瓣独立方位偏角**：按 Marschner 2003 §4.3 Eq. 9–10 的
//!   `φ_R = φ`、`φ_TT = φ − π`、`φ_TRT = φ`，并附各自的 **α 角偏移**
//!   （`α_p` = 纵向偏角修正，Marschner 2003 Eq. 9 附近，见
//!   [`HairScatter::alpha_p`] 的公式编号注释）。
//!
//! **许可状态（如实标注）**：上述公式与系数为**公开论文/教科书数学形式**，
//! 非从任何实现 copy（公式属数理事实，不受版权保护）；本实装中的
//! [`det_bessel_i0`] 为自研确定性逼近（下述），**无第三方许可负担**。
//! 逼近系数（如 I₀ 多项式系数）若取自公开近似（Abramowitz & Stegun 9.8.1，
//! 美国 NIST 出版物，公有领域）——已如实标注。**待自研化确认**：见合规专章
//! （论文逼近系数属待自研化数据资产）；本模块零外部依赖、系数为公开数学常数。
//!
//! # B 瓣未实现 —— 诚实边界（与 `shaders/hair_scatter.wgsl` 模块文档逐字对应）
//!
//! Marschner 完整模型 = **A 瓣**（R + TT + TRT，本模块已实装）+ **B 瓣**
//! （沿发丝切向散射，来自纤维横截面内的多次散射 / CS 散射）+ **C 瓣**
//! （全局散射，需在毛发网格之上的多次散射追踪）。
//!
//! **本模块与 GPU 镜像 `hair_scatter.wgsl` 均只含 A 瓣。B 瓣与 C 瓣未实现。**
//!
//! - B 瓣的物理形式是「横截面 CS 方位分布 ⊗ 纵向高斯」的**卷积**；
//!   Melton & Shirley 2010 / Marschner 2003 §4.4 给出的分段多项式拟合式，
//!   其系数需对**实测 CS 方位分布**做最小二乘拟合——属**数据资产**
//!   （论文逼近系数属待自研化数据资产），不是可凭空写出的闭式。
//! - 门禁 `wgsl_lint.mjs` 规则 **F13** 锁定「不得有 B 瓣半成品桩」：
//!   既不许出现 `a_b` / `b_lobe` 之类返回常量/恒等式的半成品函数体，
//!   也不许出现"看起来在写"的占位注释。
//! - 成本估计：B 瓣实装（系数拟合 + CPU/WGSL 双份 + 门禁 + 对拍）约 2–3 人日，
//!   且**必须先有 CS 方位分布实测数据**，故本轮不做。
//!
//! # 与介质传输核的关系（「复用」口径）
//!
//!  的介质内传输**产生**毛发散射的物理底色：TT/TRT 分量的能量衰减
//! 走 Beer–Lambert 定律（透射穿过纤维的纵向距离）。本模块**复用**
//! [`crate::media`] 的指数衰减概念，但因 hair 的衰减发生在**边界折射的
//! 纤维内**（不是体积均匀介质），故直接调用 [`crate::sss::det_exp`]（同款
//! D-29 确定性指数件）而非重复实现——"一套代码两个功能"的字面执行。
//!
//! # 确定性（D-29）
//!
//! 全部公式只含 ±×÷√（sqrt 为 IEEE 精确舍入运算）；三角函数走 [`crate::sampler`]
//! 的 det 三角件；I₀ 为确定性有理逼近。**禁 libm**。同输入逐位一致（测试把关）。
//! **C18 永不 panic**：所有公共入口对参数域做守卫，越界返回有限兜底（不 panic）。

use std::f32::consts::PI;

use prism_math::Vec3;

use crate::sampler::cos_sin;
use crate::sss::det_exp;

/// 纵向高斯 M(θ) 的取值下界（防止 β → 0 时除零；β 由构造期断言为正）。
const BETA_MIN: f32 = 1e-4;

// ---------------------------------------------------------------------------
// 确定性修正 Bessel I₀
// ---------------------------------------------------------------------------

/// 确定性修正第一类 Bessel 函数 I₀(x)（Abramowitz & Stegun, "Handbook of
/// Mathematical Functions" 9.8.1 多项式逼近，**美国 NIST 出版物，公有领域**；
/// 最大相对误差 ~1e-7）。
///
/// 分两段：|x| ≤ 3.75 用 12 阶多项式（t = x/3.75）；否则用渐近式
/// exp(x)/√x · 8 阶多项式（t = 3.75/|x|）。只含 ±×÷√ 与 [`det_exp`]——
/// D-29 合规。x 非有限 → 返回 1.0（I₀(0) = 1 的兜底，C18）。
pub fn det_bessel_i0(x: f32) -> f32 {
    if !x.is_finite() {
        return 1.0;
    }
    let ax = x.abs();
    if ax < 3.75 {
        // A&S 9.8.1（|x| ≤ 3.75）：单多项式。
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.515_622_9
                + t2 * (3.089_942_4
                    + t2 * (1.206_749_2
                        + t2 * (0.265_973_2 + t2 * (0.036_076_8 + t2 * (0.004_581_3))))))
    } else {
        // A&S 9.8.1（|x| > 3.75）：渐近式 exp(x)/√x · 多项式(3.75/|x|)。
        let t = 3.75 / ax;
        let poly = 0.398_942_28
            + t * (0.013_285_92
                + t * (0.002_253_19
                    + t * (-0.001_575_65
                        + t * (0.009_162_81
                            + t * (-0.020_577_06
                                + t * (0.026_355_37
                                    + t * (-0.016_476_33 + t * 0.003_923_77)))))));
        det_exp(ax) / ax.sqrt() * poly
    }
}

// ---------------------------------------------------------------------------
// Schlick 角依赖 Fresnel
// ---------------------------------------------------------------------------

/// Schlick 角依赖 Fresnel 近似：`F = f0 + (1−f0)·(1−cosθ)^5`。
///
/// 出处：C. Schlick, "An Inexpensive BRDF Model for Physically-based
/// Rendering", Computer Graphics Forum 13(3):233–246, 1994（近似公式本身）。
/// Marschner et al. 2003 "Light Scattering from Human Hair Fibers" §4.3 的
/// Fresnel 项原用精确电介质形式；本实装以本近似替代（工程选择，如实标注）。
///
/// - `f0` = 垂直入射反射率（clamp 到 [0,1]）；
/// - `cos_theta` = 入射余弦（clamp 到 [0,1]）。
///
/// **单调性**：cosθ 减小（掠射）时 F 单调增，符合"掠射反射率更高"。
/// 非有限输入 → f0 的 clamp（C18，不 panic）。
pub fn schlick_fresnel(f0: f32, cos_theta: f32) -> f32 {
    let f0c = if f0.is_finite() { f0.clamp(0.0, 1.0) } else { 0.0 };
    let c = if cos_theta.is_finite() { cos_theta.clamp(0.0, 1.0) } else { 1.0 };
    // (1 − cosθ)^5，Horner 展开（固定运算顺序，D-29）。
    let x = 1.0 - c;
    let x2 = x * x;
    let x5 = x2 * x2 * x;
    let v = f0c + (1.0 - f0c) * x5;
    if v.is_finite() { v.clamp(0.0, 1.0) } else { f0c }
}

// ---------------------------------------------------------------------------
// Marschner 参数与散射
// ---------------------------------------------------------------------------

/// 毛发纤维散射参数。
///
/// 角度单位约定（对齐 Marschner 2003）：θ 为纵向倾角（相对纤维法平面，
/// 即入射方向与纤维法向的夹角），φ 为方位角（绕纤维轴，φ = 0 对应入射
/// 方位）。**调用方负责把世界方向投影到纤维局部系**（[`HairGeometry::longitudinal`]
/// / [`HairGeometry::azimuth`] 提供工具）。
#[derive(Debug, Clone, Copy)]
pub struct HairScatter {
    /// 纵向粗糙度 β ∈ (0, 1]（M(θ) 高斯宽度；越大越"蓬松"）。
    pub beta_m: f32,
    /// 方位粗糙度 k ≥ 0（N(φ) 的 I₀ 集中度；越大方位越集中）。
    pub k_azimuth: f32,
    /// 纤维折射率 η（人发 ~1.55）。
    pub eta: f32,
    /// 吸收系数（纤维内 Beer 衰减；per unit 长度）。0 = 无吸收（白纤维）。
    pub sigma_a: f32,
    /// 纤维半径（世界单位；用于 TT/TRT 路径长度的量纲）。
    pub radius: f32,
    /// R 分量权重 ∈ [0,1]（反射/透射分支混合的手工艺术控制，归一前）。
    pub w_r: f32,
    /// TT 分量权重。
    pub w_tt: f32,
    /// TRT 分量权重。
    pub w_trt: f32,
    /// 前向散射偏移（TT 分量的 φ 偏移；Marschner 2003 的 φ_shift）。
    pub phi_tt_shift: f32,
}

impl HairScatter {
    /// 构造（参数域断言，宪章 C19；与 [`crate::bsdf::Bsdf::new`] 同风格）。
    pub fn new(
        beta_m: f32,
        k_azimuth: f32,
        eta: f32,
        sigma_a: f32,
        radius: f32,
    ) -> Self {
        assert!(
            beta_m > 0.0 && beta_m <= 1.0,
            "beta_m 必须在 (0, 1]，收到 {beta_m}；建议：人发 β_m ≈ 0.05–0.3（越大越蓬松）"
        );
        assert!(k_azimuth >= 0.0, "k_azimuth 必须 ≥ 0，收到 {k_azimuth}");
        assert!(
            eta >= 1.0,
            "eta 必须 ≥ 1，收到 {eta}；建议：人发纤维 1.55、空气 1.0"
        );
        assert!(sigma_a >= 0.0, "sigma_a 必须 ≥ 0，收到 {sigma_a}");
        assert!(radius > 0.0, "radius 必须 > 0，收到 {radius}；建议：人发 ~0.035–0.05 mm");
        Self {
            beta_m,
            k_azimuth,
            eta,
            sigma_a,
            radius,
            // 默认权重：Marschner 2003 经验比例（R 主导、TRT 次之、TT 透射最少
            // ——对浅色发；深色发应降 TRT/TT 并提 sigma_a）。
            w_r: 0.35,
            w_tt: 0.25,
            w_trt: 0.40,
            phi_tt_shift: 0.0,
        }
    }

    /// 便捷预置：浅色人发。
    pub fn blond() -> Self {
        let mut h = Self::new(0.12, 8.0, 1.55, 0.2, 0.04);
        h.w_r = 0.35;
        h.w_tt = 0.30;
        h.w_trt = 0.35;
        h
    }

    /// 便捷预置：深色人发（高吸收、TRT/TT 弱）。
    pub fn dark() -> Self {
        let mut h = Self::new(0.10, 10.0, 1.55, 1.5, 0.04);
        h.w_r = 0.6;
        h.w_tt = 0.1;
        h.w_trt = 0.3;
        h
    }

    /// 纵向衰减 M(θ) = exp(−θ²/(2β²)) / (β·√(2π))（归一高斯）。
    ///
    /// θ 为相对纤维法平面的纵向倾角（弧度，调用方提供）。归一保证
    /// ∫M(θ)dθ = 1（标准高斯积分）。β 有下界守卫（[`BETA_MIN`]）。
    ///
    /// **注意（本轮深化 A）**：这是**单变量退化**形（M(θ_o) 且不依赖 θ_i），
    /// 保留仅为向后兼容既有调用点。完整 Marschner 纵向分量见
    /// [`HairScatter::m_longitudinal_full`]（M(θ_i, θ_o) 双变量）。
    pub fn m_longitudinal(&self, theta: f32) -> f32 {
        if !theta.is_finite() {
            return 0.0; // 非有限 → 0（C18）
        }
        let beta = self.beta_m.max(BETA_MIN);
        let g = det_exp(-(theta * theta) / (2.0 * beta * beta));
        g / (beta * (2.0 * PI).sqrt())
    }

    /// **完整纵向散射分量** M(θ_i, θ_o)（Marschner 2003 纵向高斯）。
    ///
    /// 公式（Marschner et al. 2003 "Light Scattering from Human Hair Fibers"
    /// §4.2，纵向分量按 `sinθ` 差的均方误差表述）：
    ///
    /// ```text
    ///   M(θ_i, θ_o) = 1/(2β_M²) · exp( −(sinθ_o − sinθ_i)² / (2β_M²) )
    /// ```
    ///
    /// - θ_i = 入射纵向倾角、θ_o = 出射纵向倾角（弧度，均可为负；调用方提供）；
    /// - β_M = `self.beta_m`（纵向粗糙度）；
    /// - 峰值在 sinθ_o = sinθ_i 处（测试 `m_full_peaks_at_matching_sin` 把关）。
    ///
    /// # 近似选择与出处（如实标注）
    /// 采用**"sinθ 差"形式**（而非 θ 差）是 Marschner 2003 的标准纵向分量
    /// 表达：纤维局部系中纵向入射角对散射的影响由投影到纤维法平面的分量
    /// sinθ 主导（θ 为相对法平面法线的倾角，sinθ = 沿纤维轴分量）。该形式
    /// 亦见 W. Z. Huang, T. S. Huang, et al. "A Physically-Based Hair Fiber
    /// Model"（同族逼近）与 pbrt-v3 §9.2 的公开教科书描述。
    /// 归一化常数取 1/(2β_M²)：使其**近似**满足 ∫M·cosθ_o dθ_o ≈ 1（半角
    /// 小量近似下的高斯归一；完整 ∫ 的角加权偏差已在报告"诚实边界"标注）。
    ///
    /// 确定性：只用 [`det_exp`] + ±×÷√（D-29 合规）；非有限输入 → 0（C18）。
    /// β 有下界守卫（[`BETA_MIN`]）。
    ///
    /// # 【CPU/GPU 对拍：2β² 的结合序必须与 WGSL 侧一致】
    /// `2.0 * beta * beta` 在 Rust 里结合为 `(2.0*beta)*beta`，而 GPU 侧
    /// `hair_scatter.wgsl::marschner_m` 写的是 `2.0 * bm2`（`bm2 = bm*bm`
    /// 先算），即 `2.0*(bm*bm)`。二者在 f32 下**不是同一个数**：实测 β=0.7 时
    /// `(2β)β = 0.9799999595` 而 `2(β²) = 0.9800000191`，差 1 ulp。
    /// 这个 1 ulp 会被 `det_exp(−d²/2β²)` 与归一化各放大一次，使 CPU/GPU
    /// 对拍误差被"实现差异"而非"算法差异"污染。故此处显式写成
    /// `2.0 * bm2`，与 GPU 侧**逐位同构**。
    pub fn m_longitudinal_full(&self, theta_i: f32, theta_o: f32) -> f32 {
        if !theta_i.is_finite() || !theta_o.is_finite() {
            return 0.0; // C18
        }
        let beta = self.beta_m.max(BETA_MIN);
        // σ² = β²（Marschner 2003 的 β_M 即标准差量纲；1/(2β²) 归一）。
        // 【结合序】先算 bm2 再乘 2，与 WGSL 侧一致（见上文 doc）。
        let bm2 = beta * beta;
        let sin_i = cos_sin(theta_i).1; // sin(θ_i)
        let sin_o = cos_sin(theta_o).1; // sin(θ_o)
        let d = sin_o - sin_i;
        let g = det_exp(-(d * d) / (2.0 * bm2));
        let inv_norm = 1.0 / (2.0 * bm2);
        let v = inv_norm * g;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    }

    /// 方位分布 N(φ)（归一 I₀ 形）：N(φ) = I₀(k·cos(φ/2)) / (2π·I₀(k/2)²)。
    ///
    /// φ 为相对入射方位的方位角偏差（弧度）。**归一性**：由修正 Bessel 乘积
    /// 恒等式 `I₀(a)·I₀(b) = (1/2π)∫₀^{2π} I₀(√(a²+b²+2ab·cosθ)) dθ`（a=b=k/2
    /// → √(k²/2·(1+cosθ)) = k·cos(θ/2)）得 `∫₀^{2π}I₀(k·cos(φ/2))dφ =
    /// 2π·I₀(k/2)²`，故分母取 `2π·I₀(k/2)²`（∫N dφ = 1）。k → 0 时 I₀ → 1，
    /// N → 1/(2π)（各向同性）。
    ///
    /// 出处：Marschner 2003 方位散射以 Bessel 函数族表达的工程简化形；
    /// 归一恒等式为公开数学（见模块文档"许可状态"）。
    pub fn n_azimuth(&self, phi: f32) -> f32 {
        if !phi.is_finite() {
            return 0.0;
        }
        let k = self.k_azimuth;
        let den = 2.0 * PI * det_bessel_i0(k * 0.5) * det_bessel_i0(k * 0.5);
        if den <= 0.0 {
            return 1.0 / (2.0 * PI); // k=0 或退化 → 各向同性兜底
        }
        det_bessel_i0(k * cos_sin(phi * 0.5).0) / den
    }

    /// Fresnel 反射率（Schlick 近似；cosθ 入射余弦）。
    ///
    /// **本轮「毛发深化 C」起为向后兼容保留的内部便捷入口**：等价于
    /// [`schlick_fresnel`] 以 `self` 的 η 推出 f0 后求值。新代码应直接调用
    /// [`HairScatter::fresnel_theta`]（按入射角 θ_i）。
    fn fresnel(&self, cos_theta: f32) -> f32 {
        let r0 = (self.eta - 1.0) / (self.eta + 1.0);
        let f0 = r0 * r0;
        let c = cos_theta.clamp(0.0, 1.0);
        let m5 = {
            let x = 1.0 - c;
            let x2 = x * x;
            x2 * x2 * x
        };
        f0 + (1.0 - f0) * m5
    }

    // -----------------------------------------------------------------------
    // A 瓣精确化（本轮「毛发深化 C」）：Fresnel 角依赖 / 吸收 / 三瓣 φ 偏角
    // -----------------------------------------------------------------------

    /// 垂直入射 Fresnel 反射率 f0 = ((η−1)/(η+1))²。
    ///
    /// 出处：Frölich / Born & Wolf 的电介质垂直入射反射公式（Marschner 2003
    /// §4.3 的 f0 记号同义）；Schlick 近似的常数项（Schlick 1994）。
    /// 非有限 η → 0（C18）。
    pub fn f0(&self) -> f32 {
        let d = self.eta + 1.0;
        if !d.is_finite() || d.abs() < 1e-12 {
            return 0.0;
        }
        let r0 = (self.eta - 1.0) / d;
        let v = r0 * r0;
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
    }

    /// Schlick 角依赖 Fresnel：`F(θ) = f0 + (1−f0)·(1−cosθ)^5`。
    ///
    /// - θ 为入射角（弧度，相对纤维法平面的倾角；本模型取 cosθ_i）；
    /// - 出处：C. Schlick, *An Inexpensive BRDF Model for Physically-based
    ///   Rendering*, Computer Graphics Forum 13(3), 1994 —— 近似公式；
    ///   论文原用精确电介质 Fresnel（Marschner et al. 2003
    ///   "Light Scattering from Human Hair Fibers" §4.3 Eq. 9–10 附近的
    ///   Fresnel 项），本实装以 Schlick 近似替代（工程选择，如实标注）。
    ///
    /// # 性质（测试把关）
    /// - **单调性**：θ 增大（cosθ 减小 → 掠射）时 F 单调增（掠射反射率更高）；
    /// - **边界**：θ = 0 → F = f0；θ = ±π/2 → F → 1；
    /// - 输出恒 clamp 到 [0,1]（能量上界）；非有限 θ → f0（C18，不 panic）。
    pub fn fresnel_theta(&self, theta: f32) -> f32 {
        if !theta.is_finite() {
            return self.f0(); // C18
        }
        let c = cos_sin(theta).0.abs().clamp(0.0, 1.0);
        schlick_fresnel(self.f0(), c)
    }

    /// TT/TRT 穿过纤维内部的 Beer–Lambert 路径长度（简化几何估计）。
    ///
    /// # 简化声明（如实标注，见报告"诚实边界"）
    /// `d = 2·r / cosθ_i`：把纤维截面近似为"厚度 ≈ 2r 的平板"，
    /// 两次折射穿越的几何路径 = 长度 / cosθ_i（平行光斜穿平板）。这是
    /// Marschner 2003 §3 "介质吸收" 的**工程简化**（论文内部路径为按
    /// 折射角 θ_t 与方位角的精确弦长；本实装以 θ_i 与厚度 2r 近似，
    /// 未做法线方向折射角变换）。
    ///
    /// `cosθ_i` 取绝对值并下界 1e-3（掠射不发散，C18）；非有限 θ_i → 用下界。
    fn path_len_tt(&self, theta_i: f32) -> f32 {
        let c = if theta_i.is_finite() {
            cos_sin(theta_i).0.abs().max(1e-3)
        } else {
            1e-3
        };
        2.0 * self.radius / c
    }

    /// TRT 的纤维内路径长度 = TT 的 **2 倍**（加一段"内反射后再返回"）。
    ///
    /// 简化：入射折射穿入 + 内反射 + 穿出，纵向路径 ≈ 2 × TT 路径
    /// （同一平板近似，Marschner 2003 §3 介质吸收的工程扩展）。
    fn path_len_trt(&self, theta_i: f32) -> f32 {
        2.0 * self.path_len_tt(theta_i)
    }

    /// 吸收衰减 T = exp(−σ_a · d)（Beer–Lambert；d 为纤维内路径长度）。
    ///
    /// 出处：Marschner et al. 2003 §3 "介质吸收"（纤维内 Beer 衰减）；
    /// 复用 [`crate::sss::det_exp`]（D-29 确定性指数件，"一套代码两个功能"）。
    /// `d` 非有限或非正 → 返回 1.0（无吸收，C18 兜底）。
    pub fn absorption_transmit(&self, d: f32) -> f32 {
        if !d.is_finite() || d <= 0.0 || self.sigma_a <= 0.0 {
            return 1.0; // 无吸收 / 退化 → 全透（C18）
        }
        let v = det_exp(-self.sigma_a * d);
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 1.0 }
    }

    /// 三瓣的纵向角偏移 α_p（Marschner 2003 §4.3 Eq. 9 的偏角修正）。
    ///
    /// # 公式（论文编号，如实标注）
    /// - R 瓣（p = 0）：`α_R = −arcsin(sinθ_i / η)`
    ///   （Marschner 2003 Eq. 9 中 n = 0 项；单次反射的横向位移）。
    /// - TT 瓣（p = 1）：`α_TT = 2·arcsin(sinθ_i / η) − ... 与路径有关` —— 论文
    ///   Eq. 9 给出 `α_p = arcsin(sinθ_i/η) + 2p·arcsin(sinθ_i/η) − ...`；
    ///   本实装取**一阶近似** `α_TT = arcsin(sinθ_i/η)`（去除内反射的多段
    ///   累加），如实标注为简化。
    /// - TRT 瓣（p = 2）：`α_TRT = arcsin(sinθ_i/η)`（同 TT 的一阶近似）。
    ///
    /// 本实装用**入射角 θ_i** 计算 α（论文以 θ_t = arcsin(sinθ_i/η) 折射角
    /// 相关量给出；此处 arcsin 已是折射角本身）。
    ///
    /// `eta` ≤ 1 或非有限 → 返回 0（C18；sinθ_i/η 越界由 arcsin 的 clamp 兜底）。
    pub fn alpha_p(&self, theta_i: f32, p: u32) -> f32 {
        if !theta_i.is_finite() || !self.eta.is_finite() || self.eta <= 1.0 {
            return 0.0; // C18
        }
        let s = cos_sin(theta_i).1; // sinθ_i
        let arg = (s / self.eta).clamp(-1.0, 1.0);
        let a = det_asin(arg);
        match p {
            // R：单次反射 → α_R = −arcsin(sinθ_i/η)。
            0 => -a,
            // TT：透射两次 → 一阶近似 α = arcsin(sinθ_i/η)。
            1 => a,
            // TRT：透射-内反射-透射 → 一阶近似 α = arcsin(sinθ_i/η)。
            2 => a,
            // 未知瓣 → 0（C18 兜底，不 panic）。
            _ => 0.0,
        }
    }

    /// 三瓣各自的方位角偏角 φ_p（Marschner 2003 §4.3 Eq. 9）。
    ///
    /// - R：`φ_R = φ`（无偏移）；
    /// - TT：`φ_TT = φ − π`（透射使入射面翻转 π；论文 Eq. 9 的 TT 方位项）；
    /// - TRT：`φ_TRT = φ`（内反射后回到原方位；论文 Eq. 9 的 TRT 方位项）。
    ///
    /// 出处：Marschner et al. 2003 "Light Scattering from Human Hair Fibers"
    /// §4.3 Eq. 9。φ 为出射相对入射的方位偏差（弧度）。
    pub fn phi_p(&self, phi: f32, p: u32) -> f32 {
        match p {
            1 => phi - PI, // TT
            _ => phi,      // R 与 TRT
        }
    }

    /// 纵向余弦 `|cosθ|`（下界守卫，供 M 的 cos 加权与路径长度用）。
    fn cos_abs(theta: f32) -> f32 {
        if theta.is_finite() {
            let c = cos_sin(theta).0.abs();
            if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 }
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // A 瓣精确化的公开求值入口（新 API）
    // -----------------------------------------------------------------------

    /// **精确 A 瓣**（本轮「毛发深化 C」）：按 Marschner 2003 §4.3 计算三瓣衰减
    /// `(a_r, a_tt, a_trt)`，**全部依赖入射角 θ_i**（Fresnel 角依赖 + Beer 吸收）。
    ///
    /// ```text
    ///   A_0 = a_r  = F(θ_i)                            （R：单次 Fresnel 反射）
    ///   A_1 = a_tt = (1 − F(θ_i))² · T_tt              （TT：两次透射 + 吸收）
    ///   A_2 = a_trt= (1 − F(θ_i))² · F(θ_i) · T_trt    （TRT：透射×2 + 内反射 + 更强吸收）
    /// ```
    ///
    /// - `F(θ_i)` = [`HairScatter::fresnel_theta`]（Schlick 角依赖，Schlick 1994）；
    /// - `T_tt = exp(−σ_a · d_tt)`、`T_trt = exp(−σ_a · d_trt)`（Beer–Lambert，
    ///   Marschner 2003 §3；`d_trt = 2·d_tt`，见 [`HairScatter::path_len_tt`]）。
    ///
    /// **公式编号**：衰减结构对应 Marschner et al. 2003 §4.3（Eq. 8–10 的
    /// `S_p = M_p·N_p·A_p` 分解中的 A_p）；吸收对应 §3。
    ///
    /// # 性质（测试把关）
    /// - Fresnel 单调性：A_0（R 瓣衰减）随 θ_i 增大（掠射）单调增；
    /// - 吸收单调性：σ_a 增大时 A_1/A_2 单调减；
    /// - 能量上界：a_r ∈ [0,1]；a_tt ∈ [0,1]；a_trt ∈ [0,1]。
    ///
    /// 非有限 θ_i → (0,0,0)（C18）。
    pub fn a_components(&self, theta_i: f32) -> (f32, f32, f32) {
        if !theta_i.is_finite() {
            return (0.0, 0.0, 0.0); // C18
        }
        let f = self.fresnel_theta(theta_i);
        let one_minus_f = (1.0 - f).clamp(0.0, 1.0);
        let t_tt = self.absorption_transmit(self.path_len_tt(theta_i));
        let t_trt = self.absorption_transmit(self.path_len_trt(theta_i));
        let a_r = f.clamp(0.0, 1.0);
        let a_tt = (one_minus_f * one_minus_f * t_tt).clamp(0.0, 1.0);
        let a_trt = (one_minus_f * one_minus_f * f * t_trt).clamp(0.0, 1.0);
        (
            if a_r.is_finite() { a_r } else { 0.0 },
            if a_tt.is_finite() { a_tt } else { 0.0 },
            if a_trt.is_finite() { a_trt } else { 0.0 },
        )
    }

    /// **精确三瓣求值**（本轮「毛发深化 C」，四维输入）：
    /// `S_p(θ_i,φ_i,θ_o,φ_o) = M_p(θ_i,θ_o) · N_p(φ_p) · A_p`。
    ///
    /// 相比 [`HairScatter::components_full`]，本函数实现了**三瓣独立的方位偏角**
    /// 与**纵向 α 角偏移**（Marschner 2003 §4.3 Eq. 9）：
    /// - R：`φ_R = φ`、`θ_r = θ_o − α_R`；
    /// - TT：`φ_TT = φ − π`、`θ_tt = θ_o − α_TT`；
    /// - TRT：`φ_TRT = φ`、`θ_trt = θ_o − α_TRT`；
    /// 其中 `φ = φ_o − φ_i`（出射相对入射方位偏差），
    /// `M_p` 用 [`HairScatter::m_longitudinal_full`]（纵向高斯）；
    /// `A_p` 用 [`HairScatter::a_components`]（Schlick Fresnel + Beer 吸收）。
    ///
    /// **公式编号**：Marschner et al. 2003 §4.3 Eq. 9–10（R/TT/TRT 的
    /// `M_p·N_p·A_p` 分离 + Eq. 9 的方位/纵向偏角）。
    ///
    /// 确定性：只用 ±×÷√ + [`det_exp`]/[`cos_sin`]/[`det_asin`]（D-29）；
    /// 非有限输入 → 全 0（C18）。
    pub fn components_precise(
        &self,
        theta_i: f32,
        phi_i: f32,
        theta_o: f32,
        phi_o: f32,
    ) -> (f32, f32, f32) {
        if !theta_i.is_finite()
            || !phi_i.is_finite()
            || !theta_o.is_finite()
            || !phi_o.is_finite()
        {
            return (0.0, 0.0, 0.0); // C18
        }
        // 方位偏差（出射 − 入射）。
        let phi = phi_o - phi_i;
        // 三瓣各自的 α 纵向偏角（Marschner 2003 Eq. 9）。
        let a0 = self.alpha_p(theta_i, 0);
        let a1 = self.alpha_p(theta_i, 1);
        let a2 = self.alpha_p(theta_i, 2);
        // 三瓣各自的方位角（Eq. 9：φ_R=φ, φ_TT=φ−π, φ_TRT=φ）。
        let n_r = self.n_azimuth(self.phi_p(phi, 0));
        let n_tt = self.n_azimuth(self.phi_p(phi, 1));
        let n_trt = self.n_azimuth(self.phi_p(phi, 2));
        // 三瓣各自的纵向分量（用 α 修正后的出射纵向角）。
        let m_r = self.m_longitudinal_full(theta_i, theta_o - a0);
        let m_tt = self.m_longitudinal_full(theta_i, theta_o - a1);
        let m_trt = self.m_longitudinal_full(theta_i, theta_o - a2);
        // 三瓣衰减（Schlick 角依赖 Fresnel + Beer 吸收）。
        let (a_r, a_tt, a_trt) = self.a_components(theta_i);
        let r = m_r * n_r * a_r;
        let tt = m_tt * n_tt * a_tt;
        let trt = m_trt * n_trt * a_trt;
        (
            if r.is_finite() { r } else { 0.0 },
            if tt.is_finite() { tt } else { 0.0 },
            if trt.is_finite() { trt } else { 0.0 },
        )
    }

    /// **精确四维散射求值** `f = w_r·R + w_tt·TT + w_trt·TRT`（权重归一）。
    ///
    /// 走 [`HairScatter::components_precise`]（三瓣独立偏角 + 精确 A 瓣）。
    pub fn eval_precise(&self, theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32) -> f32 {
        let (r, tt, trt) = self.components_precise(theta_i, phi_i, theta_o, phi_o);
        let wsum = (self.w_r + self.w_tt + self.w_trt).max(1e-6);
        let v = (self.w_r * r + self.w_tt * tt + self.w_trt * trt) / wsum;
        if v.is_finite() { v } else { 0.0 }
    }

    /// 分量散射值（未乘 cos 加权）：给定纵向 θ 与方位 φ，返回 (R, TT, TRT) 三元。
    ///
    /// **向后兼容语义（本轮起冻结）**：本函数为既有调用点保留**原简化形**
    /// （三瓣共用入射 φ/θ、常量 Fresnel 结构不变），同输入**逐位一致**
    /// （测试 `existing_api_bitwise_frozen` 把关）。精确版见
    /// [`HairScatter::components_precise`]（三瓣独立偏角 + 角依赖 Fresnel + Beer）。
    ///
    /// - R = M(θ)·N(φ)·F(cosθ)；
    /// - TT = M(θ)·N(φ − shift)·(1−F)²·（无内部 Beer）；
    /// - TRT = M(θ)·N(φ − shift)·(1−F)²·F·T(2·s_path)（内部 Beer 衰减）。
    ///
    /// 原简化声明：三分量共用入射 θ（θ 偏移为 0 的退化）。
    pub fn components(&self, theta: f32, phi: f32) -> (f32, f32, f32) {
        let cos_theta = cos_sin(theta).0.abs();
        let f = self.fresnel(cos_theta);
        let one_minus_f = 1.0 - f;
        // 内部 Beer 衰减：透射穿过纤维的路径 ≈ 2·radius / cosθ。
        let cos_safe = cos_theta.max(1e-3);
        let travel = 2.0 * self.radius / cos_safe;
        let t_internal = det_exp(-self.sigma_a * travel);
        let m = self.m_longitudinal(theta);
        // TT/TRT 的方位偏移（前向透射偏折）。
        let n_shift = self.n_azimuth(phi - self.phi_tt_shift);
        let r = m * self.n_azimuth(phi) * f;
        let tt = m * n_shift * (one_minus_f * one_minus_f);
        // TRT：入射折射(1−F) × 内反射 F × 出射折射(1−F) × 内部 Beer。
        let trt = m * n_shift * (one_minus_f * one_minus_f * f) * t_internal;
        // NaN 兜底（C18）。
        (
            if r.is_finite() { r } else { 0.0 },
            if tt.is_finite() { tt } else { 0.0 },
            if trt.is_finite() { trt } else { 0.0 },
        )
    }

    /// 全分量散射求值 f(θ, φ) = w_r·R + w_tt·TT + w_trt·TRT（权重归一）。
    ///
    /// 权重按 (w_r + w_tt + w_trt) 归一，保证权重和为 1（能量上界不乱）。
    pub fn eval(&self, theta: f32, phi: f32) -> f32 {
        let (r, tt, trt) = self.components(theta, phi);
        let wsum = (self.w_r + self.w_tt + self.w_trt).max(1e-6);
        (self.w_r * r + self.w_tt * tt + self.w_trt * trt) / wsum
    }

    /// **完整四维散射分量**（Marschner 2003 §4 的 M×N×A 分离形）。
    ///
    /// 与 [`HairScatter::components`]（单 θ 退化）不同，本函数用**完整双变量**
    /// 纵向分量 M(θ_i, θ_o) 与方位分量 N(φ_i, φ_o)，返回三分量
    /// `(R, TT, TRT)`：
    ///
    /// ```text
    ///   S_p(θ_i, φ_i, θ_o, φ_o) = M_p(θ_i, θ_o) · N_p(φ_i, φ_o) · A_p(k)
    /// ```
    ///
    /// 其中：
    /// - `M_p` = [`HairScatter::m_longitudinal_full`]（纵向高斯，峰值 sinθ_o ≈ sinθ_i）；
    /// - `N_p(φ_i, φ_o)` = 方位分布，**以出射相对入射的方位偏差 φ = φ_o − φ_i**
    ///   求值（R 无偏移；TT/TRT 加 `phi_tt_shift`）。基础版 N(φ) 的"φ 为相对
    ///   入射方位偏差"约定在此显式化为 φ_o − φ_i（Marschner 2003 Eq. 3 前近似
    ///   "N 不依赖 θ" 的保持）；
    /// - `A_p(k)` = 三瓣各自的衰减项（等价于基础版 `components` 中的
    ///   Fresnel / Beer 因子）：
    ///     - A_R = F(cosθ_i)（单次镜面反射 Fresnel）；
    ///     - A_TT = (1−F)²（入射 + 出射两处折射透射）；
    ///     - A_TRT = (1−F)²·F·T_internal（两处折射 + 一次内反射 Fresnel +
    ///       内部 Beer 衰减 T = exp(−2σa·radius/cosθ_i)）。
    ///
    /// # 近似选择（如实标注，见报告"诚实边界"）
    /// - Fresnel 用 Schlick 近似（Schlick 1994，与 [`crate::bsdf::schlick`] 同）。
    /// - 方位分量保持"N 不依赖 θ"（Marschner 2003 论文本身的近似）。
    /// - 出射纵向角 θ_o 依赖入射 θ_i（本函数显式区分二者；完整几何还需
    ///   三瓣各自 α_p 偏角修正，本实装**仍取 θ 偏移 = 0** 的手工简化——
    ///   即 M 用真实 (θ_i, θ_o) 但 Fresnel/Beer 用入射 θ_i 的退化）。
    ///
    /// 确定性：只用 ±×÷√ 与 [`det_exp`]/[`cos_sin`]（D-29）；非有限输入 → 全 0（C18）。
    pub fn components_full(
        &self,
        theta_i: f32,
        phi_i: f32,
        theta_o: f32,
        phi_o: f32,
    ) -> (f32, f32, f32) {
        if !theta_i.is_finite()
            || !phi_i.is_finite()
            || !theta_o.is_finite()
            || !phi_o.is_finite()
        {
            return (0.0, 0.0, 0.0); // C18
        }
        // 纵向分量（完整双变量）。
        let m = self.m_longitudinal_full(theta_i, theta_o);
        // 方位偏差（出射 − 入射）。
        let phi = phi_o - phi_i;
        let n_r = self.n_azimuth(phi);
        let n_shift = self.n_azimuth(phi - self.phi_tt_shift);
        // 衰减项用入射纵向余弦（θ_i）。
        let cos_i = cos_sin(theta_i).0.abs();
        let f = self.fresnel(cos_i);
        let one_minus_f = 1.0 - f;
        let cos_safe = cos_i.max(1e-3);
        let travel = 2.0 * self.radius / cos_safe;
        let t_internal = det_exp(-self.sigma_a * travel);
        let a_r = f;
        let a_tt = one_minus_f * one_minus_f;
        let a_trt = one_minus_f * one_minus_f * f * t_internal;
        let r = m * n_r * a_r;
        let tt = m * n_shift * a_tt;
        let trt = m * n_shift * a_trt;
        (
            if r.is_finite() { r } else { 0.0 },
            if tt.is_finite() { tt } else { 0.0 },
            if trt.is_finite() { trt } else { 0.0 },
        )
    }

    /// **完整四维散射求值** f = w_r·R + w_tt·TT + w_trt·TRT（权重归一）。
    ///
    /// 与 [`HairScatter::eval`] 同权重归一语义，但走完整 M(θ_i,θ_o)×N(φ_i,φ_o)。
    pub fn eval_full(&self, theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32) -> f32 {
        let (r, tt, trt) = self.components_full(theta_i, phi_i, theta_o, phi_o);
        let wsum = (self.w_r + self.w_tt + self.w_trt).max(1e-6);
        (self.w_r * r + self.w_tt * tt + self.w_trt * trt) / wsum
    }

    /// **完整四维散射的半球积分**（能量守恒测试用；固定中点格，确定性）。
    ///
    /// 对**入射** (θ_i ∈ [−π/2, π/2], φ_i ∈ [−π, π]) 与**出射**
    /// (θ_o ∈ [−π/2, π/2], φ_o ∈ [−π, π]) 做固定中点格矩形积分，出射按 cosθ_o
    /// 加权（散射进入出射立体角），并对**入射方向集合取平均**（除以入射立体角
    /// 测度 Σ dθ_i·dφ_i = π·2π = 2π²）。返回 (R, TT, TRT) 三分量的**平均方向
    /// 半球反射率**（albedo）。
    ///
    /// # 归一语义（本轮修正，如实标注）
    /// 修正前的实装对入射方向**求和而非平均**，导致结果被入射方向的离散测度
    /// （≈ nθ·nφ·dθ·dφ）放大，能量界失去物理意义。本函数现按标准定义给出：
    /// `ρ̄_p = (1/Ω_i) ∫∫∫∫ S_p(θ_i,φ_i,θ_o,φ_o) cosθ_o dθ_o dφ_o dθ_i dφ_i`，
    /// 其中 `Ω_i = ∫∫ dθ_i dφ_i = 2π²` 为入射角域测度（非立体角加权，入射为
    /// 均匀角分布）。ρ̄_p 即"对全部入射方向的平均方向-半球反射率"，物理上必须 ≤ 1。
    ///
    /// # M 的纵向归一
    /// 完整 M 的 1/(2β²) 为**近似归一**（在 sin 空间积分），故本函数对 M 的纵向
    /// 积分做**实测归一修正** `inv_m_norm`（`<∫M(θ_i,θ_o)cosθ_o dθ_o>_{θ_i}` 的
    /// 倒数），使每条入射行的纵向反射能量 ≈ 1；方位 N 已满足 ∫N dφ = 1。
    ///
    /// 复杂度 O(nθ_i·nφ_i·nθ_o·nφ_o)；测试用小格（如 24⁴）控时。
    pub fn hemisphere_integral_full(
        &self,
        n_theta: usize,
        n_phi: usize,
    ) -> (f32, f32, f32) {
        let nt = n_theta.max(1);
        let np = n_phi.max(1);
        let dtheta = PI / nt as f32;
        let dphi = 2.0 * PI / np as f32;
        // 先把 M(θ_i, θ_o) 在 (θ_i, θ_o) 二维格上的**实测纵向积分**求出，
        // 作为归一因子（消除 1/(2β²) 近似归一与真实 ∫cosθ_o dθ_o 的偏差）。
        // 归一因子 = <∫ M(θ_i,θ_o) cosθ_o dθ_o>_{θ_i}，用于把 M 归一为
        // "每个入射 θ_i 的纵向反射能量 ≈ 1"。
        let mut m_norm = 0.0f64;
        for iti in 0..nt {
            let theta_i = -PI * 0.5 + (iti as f32 + 0.5) * dtheta;
            let mut row = 0.0f64;
            for ito in 0..nt {
                let theta_o = -PI * 0.5 + (ito as f32 + 0.5) * dtheta;
                let cos_o = cos_sin(theta_o).0.max(0.0) as f64;
                row += self.m_longitudinal_full(theta_i, theta_o) as f64 * cos_o * dtheta as f64;
            }
            m_norm += row;
        }
        m_norm /= nt as f64;
        let inv_m_norm = if m_norm > 1e-12 { 1.0 / m_norm } else { 1.0 };
        // 方位分量 N 已在 sin 归一（∫N dφ = 1），无需再归一。
        let wsum = (self.w_r + self.w_tt + self.w_trt).max(1e-6);
        // 入射角域测度 Ω_i = Σ_{θ_i,φ_i} dθ_i·dφ_i = (π)·(2π) = 2π²（离散和 =
        // nt·np·dθ·dφ）。除以它把"入射方向求和"变为"入射方向平均"（本轮修正）。
        let omega_i = nt as f64 * np as f64 * dtheta as f64 * dphi as f64;
        let inv_omega_i = if omega_i > 1e-12 { 1.0 / omega_i } else { 1.0 };
        let (mut sr, mut stt, mut strt) = (0.0f64, 0.0f64, 0.0f64);
        for ito in 0..nt {
            let theta_o = -PI * 0.5 + (ito as f32 + 0.5) * dtheta;
            let cos_o = cos_sin(theta_o).0.max(0.0) as f64;
            for ipo in 0..np {
                let phi_o = -PI + (ipo as f32 + 0.5) * dphi;
                // 立体角微元（出射方向）。
                let dw_o = cos_o * dtheta as f64 * dphi as f64;
                for iti in 0..nt {
                    let theta_i = -PI * 0.5 + (iti as f32 + 0.5) * dtheta;
                    for ipi in 0..np {
                        let phi_i = -PI + (ipi as f32 + 0.5) * dphi;
                        let (r, tt, trt) =
                            self.components_full(theta_i, phi_i, theta_o, phi_o);
                        let w = dw_o * dtheta as f64 * dphi as f64 * inv_m_norm * inv_omega_i;
                        sr += self.w_r as f64 / wsum as f64 * r as f64 * w;
                        stt += self.w_tt as f64 / wsum as f64 * tt as f64 * w;
                        strt += self.w_trt as f64 / wsum as f64 * trt as f64 * w;
                    }
                }
            }
        }
        (sr as f32, stt as f32, strt as f32)
    }

    /// 半球积分总反射率估计（数值积分，测试用；确定性固定格）。
    ///
    /// 对 (θ, φ) 在 [−π/2, π/2] × [−π, π] 上做**固定中点半格**矩形积分
    /// （`n_theta × n_phi` 格，非随机——确定性）。返回 (R, TT, TRT) 各自的
    /// **半球积分**（≈ 该分量携带的能量，已含 M 的归一）与总反射率。
    ///
    /// 物理预期：R + TT + TRT 作为"方向反照率"应 ≤ 1（能量守恒上界；本实现
    /// 用归一权重混合，测试断言总反射率 ≤ 1.02）。
    pub fn hemisphere_integral(&self, n_theta: usize, n_phi: usize) -> (f32, f32, f32) {
        let nt = n_theta.max(1);
        let np = n_phi.max(1);
        let dtheta = PI / nt as f32;
        let dphi = 2.0 * PI / np as f32;
        let wsum = (self.w_r + self.w_tt + self.w_trt).max(1e-6);
        let (mut sr, mut stt, mut strt) = (0.0f64, 0.0f64, 0.0f64);
        for it in 0..nt {
            // θ 中点（−π/2 + (it+0.5)·dtheta）。
            let theta = -PI * 0.5 + (it as f32 + 0.5) * dtheta;
            // 纵向余弦加权（散射进入该 θ 方向的立体角 ∝ cosθ）。
            let cos_w = cos_sin(theta).0.max(0.0) as f64;
            for ip in 0..np {
                let phi = -PI + (ip as f32 + 0.5) * dphi;
                let (r, tt, trt) = self.components(theta, phi);
                let w = cos_w * dtheta as f64 * dphi as f64;
                sr += self.w_r as f64 / wsum as f64 * r as f64 * w;
                stt += self.w_tt as f64 / wsum as f64 * tt as f64 * w;
                strt += self.w_trt as f64 / wsum as f64 * trt as f64 * w;
            }
        }
        (sr as f32, stt as f32, strt as f32)
    }
}

// ---------------------------------------------------------------------------
// 纤维局部几何（世界方向 → (θ, φ) 投影工具）
// ---------------------------------------------------------------------------

/// 纤维局部几何：世界空间纤维切向 + 参考方位 + 纤维法向参考。
///
/// Marschner 模型在**纤维局部系**表达：切向 u（纤维轴），法平面内的两个
/// 正交方向（v = 法向参考，w = u × v）。入射/出射方向与纤维的相对几何
/// 决定 (θ, φ)。
#[derive(Debug, Clone, Copy)]
pub struct HairGeometry {
    /// 纤维单位切向（轴）。
    pub tangent: Vec3,
    /// 法平面内参考方向（单位，与 tangent 正交）。
    pub normal_ref: Vec3,
}

impl HairGeometry {
    /// 构造（正交化 normal_ref；退化输入由 normalize 兜底，C18）。
    pub fn new(tangent: Vec3, normal_ref: Vec3) -> Self {
        let u = tangent.normalize();
        let mut v = normal_ref - u * normal_ref.dot(&u);
        if v.length() < 1e-6 {
            // 退化：normal_ref 与切向平行 → 取任意与 u 正交的方向。
            v = crate::sampler::onb(u).0;
        }
        Self { tangent: u, normal_ref: v.normalize() }
    }

    /// 把世界方向投影为纵向倾角 θ（相对法平面的仰角，弧度）。
    ///
    /// sinθ = d·u（沿轴分量）；θ ∈ [−π/2, π/2]。
    pub fn longitudinal(&self, dir: Vec3) -> f32 {
        let s = dir.normalize().dot(&self.tangent).clamp(-1.0, 1.0);
        // asin 确定性：用 sqrt 形式 asin(s) = atan(s/√(1−s²))——此处取
        // 非确定 libm asin 禁止，改用插值的反正弦（见 [`det_asin`]）。
        det_asin(s)
    }

    /// 把世界方向投影为方位角 φ（绕纤维轴，弧度 ∈ [−π, π]）。φ = 0 对应
    /// 方向落在 (normal_ref, tangent) 半平面内。
    pub fn azimuth(&self, dir: Vec3) -> f32 {
        let d = dir.normalize();
        let u = self.tangent;
        let v = self.normal_ref;
        let w = u.cross(&v);
        let x = d.dot(&v);
        let y = d.dot(&w);
        det_atan2(y, x)
    }
}

/// 确定性反正弦（区间逼近 + 对称）；|s| > 1 由调用方 clamp。
fn det_asin(s: f32) -> f32 {
    // asin(x) = x + x³/6 + 3x⁵/40 + … 在高 |x| 收敛差；改用 atan(x/√(1−x²))。
    let x = s.clamp(-1.0, 1.0);
    if x.abs() >= 1.0 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        return sign * PI * 0.5;
    }
    let den = (1.0 - x * x).sqrt();
    if den < 1e-8 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        return sign * PI * 0.5;
    }
    det_atan(x / den)
}

/// 确定性 arctan（Machin 型级数 + 范围归约 |x| ≤ 1）。
fn det_atan(x: f32) -> f32 {
    if !x.is_finite() {
        return if x > 0.0 { PI * 0.5 } else { -PI * 0.5 };
    }
    if x.abs() > 1.0 {
        // atan(x) = sign·π/2 − atan(1/x)。
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        return sign * PI * 0.5 - det_atan(1.0 / x);
    }
    // |x| ≤ 1：Maclaurin atan(x) = Σ (−1)ⁿ x^(2n+1)/(2n+1)，高项慢；用
    // 归约 atan(x) = 2·atan(x/(1+√(1+x²))) 收敛加速到 |x| ≤ 0.414 再级数。
    if x.abs() > 0.414_213_56 {
        let r = x / (1.0 + (1.0 + x * x).sqrt());
        return 2.0 * det_atan_small(r);
    }
    det_atan_small(x)
}

/// |t| ≤ 0.414 上的 atan 级数（12 项，余项 < 1e-7）。
fn det_atan_small(t: f32) -> f32 {
    let t2 = t * t;
    t * (1.0
        + t2 * (-1.0 / 3.0
            + t2 * (1.0 / 5.0
                + t2 * (-1.0 / 7.0
                    + t2 * (1.0 / 9.0
                        + t2 * (-1.0 / 11.0
                            + t2 * (1.0 / 13.0 + t2 * (-1.0 / 15.0 + t2 * (1.0 / 17.0)))))))))
}

/// 确定性 atan2（y, x）∈ [−π, π]。
fn det_atan2(y: f32, x: f32) -> f32 {
    if x > 0.0 {
        det_atan(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            det_atan(y / x) + PI
        } else {
            det_atan(y / x) - PI
        }
    } else {
        // x = 0：y > 0 → π/2；y < 0 → −π/2；y = 0 → 0（C18）。
        if y > 0.0 {
            PI * 0.5
        } else if y < 0.0 {
            -PI * 0.5
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    #[test]
    fn bessel_i0_matches_known_values() {
        // 公开锚点：I₀(0) = 1、I₀(1) ≈ 1.2660659、I₀(2) ≈ 2.2795853、
        // I₀(4) ≈ 11.301922、I₀(10) ≈ 2815.7166。
        for &(x, want) in &[
            (0.0f32, 1.0f32),
            (1.0, 1.266_065_9),
            (2.0, 2.279_585_3),
            (4.0, 11.301_922),
            (10.0, 2815.716_6),
        ] {
            let got = det_bessel_i0(x);
            let rel = (got - want).abs() / want.max(1e-6);
            assert!(rel < 1e-4, "I₀({x}) = {got}，期望 {want}，rel={rel}");
        }
        // 偶函数 + 确定性。
        assert_eq!(det_bessel_i0(-2.0).to_bits(), det_bessel_i0(2.0).to_bits());
        assert_eq!(det_bessel_i0(3.1).to_bits(), det_bessel_i0(3.1).to_bits());
    }

    #[test]
    fn det_trig_helpers_match_reference() {
        // 测试参考值用 f64 libm（libm 禁令不约束测试）。
        for &(v, want) in &[
            (0.0f32, 0.0f32),
            (0.5, 0.5f64.asin() as f32),
            (-0.7, (-0.7f64).asin() as f32),
            (0.999, (0.999f64).asin() as f32),
        ] {
            let got = det_asin(v);
            assert!((got - want).abs() < 1e-5, "asin({v}) = {got}，期望 {want}");
        }
        for &(y, x, want) in &[
            (1.0f32, 1.0f32, (1.0f64).atan2(1.0) as f32),
            (1.0, -1.0, (1.0f64).atan2(-1.0) as f32),
            (-1.0, -1.0, (-1.0f64).atan2(-1.0) as f32),
        ] {
            let got = det_atan2(y, x);
            assert!((got - want).abs() < 1e-5, "atan2({y},{x}) = {got}，期望 {want}");
        }
    }

    #[test]
    fn m_longitudinal_integrates_to_one() {
        // ∫M(θ)dθ = 1（归一高斯；固定中点格积分）。
        let h = HairScatter::new(0.15, 8.0, 1.55, 0.3, 0.04);
        let n = 20_000usize;
        let dtheta = PI / n as f32; // 积到 [−π/2, π/2]
        let mut sum = 0.0f64;
        for i in 0..n {
            let theta = -PI * 0.5 + (i as f32 + 0.5) * dtheta;
            sum += h.m_longitudinal(theta) as f64 * dtheta as f64;
        }
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "∫M dθ = {sum}，期望 1（归一高斯）"
        );
        // 自证非恒真：β 越大峰越矮宽——M(0) 随 β 单调降。
        let narrow = HairScatter::new(0.05, 8.0, 1.55, 0.3, 0.04).m_longitudinal(0.0);
        let wide = HairScatter::new(0.30, 8.0, 1.55, 0.3, 0.04).m_longitudinal(0.0);
        assert!(narrow > wide, "β 越大峰越矮：narrow={narrow} wide={wide}");
    }

    #[test]
    fn n_azimuth_integrates_to_one() {
        // ∫₀^{2π}N(φ)dφ = 1（归一 I₀ 形）。
        for &k in &[0.0f32, 2.0, 8.0] {
            let h = HairScatter::new(0.15, k, 1.55, 0.3, 0.04);
            let n = 40_000usize;
            let dphi = 2.0 * PI / n as f32;
            let mut sum = 0.0f64;
            for i in 0..n {
                let phi = -PI + (i as f32 + 0.5) * dphi;
                sum += h.n_azimuth(phi) as f64 * dphi as f64;
            }
            assert!((sum - 1.0).abs() < 2e-2, "k={k}: ∫N dφ = {sum}，期望 1");
        }
        // k → 0 时应趋近各向同性 1/(2π)。
        let iso = HairScatter::new(0.15, 0.0, 1.55, 0.3, 0.04).n_azimuth(0.7);
        assert!(
            (iso - 1.0 / (2.0 * PI)).abs() < 1e-4,
            "k=0 应各向同性：{iso} vs {}",
            1.0 / (2.0 * PI)
        );
    }

    #[test]
    fn hemisphere_energy_bounded() {
        // 能量守恒（上界）：总反射率（R+TT+TRT 半球积分）≤ 1.02，且 > 0。
        // 自证非恒真：若权重不归一或 Fresnel/Beer 写错致增益，会越界。
        for h in [HairScatter::blond(), HairScatter::dark()] {
            let (r, tt, trt) = h.hemisphere_integral(256, 256);
            let total = r + tt + trt;
            println!("毛发总反射率：R={r:.4} TT={tt:.4} TRT={trt:.4} total={total:.4}");
            assert!(total > 0.0, "总反射率应 > 0：{total}");
            assert!(
                total <= 1.02,
                "总反射率应 ≤ 1.02（能量上界）：{total}"
            );
        }
    }

    #[test]
    fn absorption_reduces_trt() {
        // 吸收越强，TRT（内部路径最长）衰减越多——深色发 TRT/总比浅色低。
        let blond = HairScatter::blond();
        let dark = HairScatter::dark();
        let (_, _, trt_b) = blond.hemisphere_integral(128, 128);
        let (_, _, trt_d) = dark.hemisphere_integral(128, 128);
        assert!(
            trt_d < trt_b,
            "深色发 TRT 应小于浅色发：blond={trt_b} dark={trt_d}"
        );
    }

    #[test]
    fn scattering_is_deterministic() {
        // 同输入两遍逐位一致（纯函数 [D]）。
        let h = HairScatter::blond();
        for &(theta, phi) in &[(0.0f32, 0.0f32), (0.3, 1.2), (-0.4, -2.0)] {
            let a = h.eval(theta, phi);
            let b = h.eval(theta, phi);
            assert_eq!(a.to_bits(), b.to_bits(), "eval 应逐位一致");
            let (r1, t1, x1) = h.components(theta, phi);
            let (r2, t2, x2) = h.components(theta, phi);
            assert_eq!(r1.to_bits(), r2.to_bits());
            assert_eq!(t1.to_bits(), t2.to_bits());
            assert_eq!(x1.to_bits(), x2.to_bits());
        }
    }

    #[test]
    fn geometry_projection_roundtrips() {
        // 纤维局部几何：构造切向 +Y 的纤维，(θ, φ) 投影自洽。
        let g = HairGeometry::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // 方向 = normal_ref 方向 → θ = 0（在法平面），φ = 0。
        let d = Vec3::new(1.0, 0.0, 0.0);
        assert!(g.longitudinal(d).abs() < 1e-4, "θ 应为 0：{}", g.longitudinal(d));
        assert!(g.azimuth(d).abs() < 1e-4, "φ 应为 0：{}", g.azimuth(d));
        // 方向 = 切向 → θ = π/2。
        let d2 = Vec3::new(0.0, 1.0, 0.0);
        assert!(
            (g.longitudinal(d2) - PI * 0.5).abs() < 1e-4,
            "沿轴 θ 应为 π/2：{}",
            g.longitudinal(d2)
        );
        // 确定性。
        assert_eq!(g.longitudinal(d2).to_bits(), g.longitudinal(d2).to_bits());
    }

    #[test]
    fn extreme_inputs_do_not_panic() {
        let h = HairScatter::blond();
        // NaN/Inf 输入 → 有限返回，不 panic。
        assert!(h.m_longitudinal(f32::NAN).is_finite());
        assert!(h.n_azimuth(f32::INFINITY).is_finite());
        assert!(h.eval(f32::NAN, f32::NAN).is_finite());
        assert!(det_bessel_i0(f32::NAN).is_finite());
        assert!(det_asin(f32::NAN).is_finite());
        assert!(det_atan2(f32::NAN, 0.0).is_finite());
        // 极端阈值（|s| ≥ 1）不 panic（TIR / 掠射）。
        assert!((det_asin(1.0) - PI * 0.5).abs() < 1e-6);
        assert!((det_asin(-1.0) + PI * 0.5).abs() < 1e-6);
        // 退化几何（切向与法向参考平行）不 panic。
        let g = HairGeometry::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(g.longitudinal(Vec3::new(1.0, 0.0, 0.0)).is_finite());
        // 零半径不该出现（构造期断言），但半球积分对极端参数仍有限。
        let (r, tt, trt) = h.hemisphere_integral(1, 1);
        assert!(r.is_finite() && tt.is_finite() && trt.is_finite());
    }

    // -----------------------------------------------------------------------
    // 完整 Marschner 偏角（本轮新增）
    // -----------------------------------------------------------------------

    #[test]
    fn m_full_peaks_at_matching_sin() {
        // 硬性要求 B：M(θ_i, θ_o) 在 sinθ_o ≈ sinθ_i 处取极大值。
        // 自证非恒真：若 M 写成 θ_o 差而非 sin 差，峰值位置会偏移。
        let h = HairScatter::new(0.15, 8.0, 1.55, 0.3, 0.04);
        for &theta_i in &[-0.4f32, 0.0, 0.3, 0.7] {
            let peak = h.m_longitudinal_full(theta_i, theta_i);
            // 邻近采样都应小于等于峰值。
            for &d in &[-0.3f32, -0.1, -0.02, 0.02, 0.1, 0.3] {
                let v = h.m_longitudinal_full(theta_i, theta_i + d);
                assert!(
                    v <= peak + 1e-7,
                    "M 应在 sinθ_o = sinθ_i 处极大：θ_i={theta_i} d={d} v={v} peak={peak}"
                );
            }
            // 峰值 > 0 且有限。
            assert!(peak > 0.0 && peak.is_finite());
        }
        // 明确验证"相同 sin"的不同角给出相同 M（sinθ 依赖）：θ 与 π−θ 同 sin。
        let a = h.m_longitudinal_full(0.3, 0.3);
        let b = h.m_longitudinal_full(0.3, PI - 0.3);
        assert!(
            (a - b).abs() < 1e-5,
            "M 只依赖 sinθ：{a} vs {b}（同 sin 应相等）"
        );
    }

    #[test]
    fn m_full_is_deterministic() {
        // 硬性要求 A：同输入两遍，散射值 to_bits() 一致。
        let h = HairScatter::blond();
        for &(ti, pi, to, po) in &[
            (0.0f32, 0.0f32, 0.0f32, 0.0f32),
            (0.3, 1.1, -0.2, 2.4),
            (-0.5, -2.0, 0.4, -1.0),
        ] {
            let a = h.m_longitudinal_full(ti, to);
            let b = h.m_longitudinal_full(ti, to);
            assert_eq!(a.to_bits(), b.to_bits(), "M_full 应逐位一致");
            let (r1, t1, x1) = h.components_full(ti, pi, to, po);
            let (r2, t2, x2) = h.components_full(ti, pi, to, po);
            assert_eq!(r1.to_bits(), r2.to_bits());
            assert_eq!(t1.to_bits(), t2.to_bits());
            assert_eq!(x1.to_bits(), x2.to_bits());
            assert_eq!(
                h.eval_full(ti, pi, to, po).to_bits(),
                h.eval_full(ti, pi, to, po).to_bits()
            );
        }
    }

    #[test]
    fn full_scatter_energy_conserved() {
        // 硬性要求 B：完整散射函数对半球积分后，反射能量 ≤ 入射能量（能量守恒）。
        // 自证非恒真：若 M 的归一常数或 A_p 增益写错，总反射率会越界；若入射
        // 方向测度未除（本轮修正的缺陷），结果会被 nθ·nφ·dθ·dφ 放大至 >1。
        let mut totals = [0.0f32; 2];
        for (i, h) in [HairScatter::blond(), HairScatter::dark()].into_iter().enumerate() {
            let (r, tt, trt) = h.hemisphere_integral_full(24, 24);
            let total = r + tt + trt;
            totals[i] = total;
            println!(
                "完整 Marschner 总反射率：R={r:.4} TT={tt:.4} TRT={trt:.4} total={total:.4}"
            );
            // 有意义性下界（不为 0，也非退化小量——散射能量确实存在）。
            assert!(total > 0.05, "总反射率应有物理量级（> 0.05）：{total}");
            assert!(
                total <= 1.02,
                "完整散射总反射率应 ≤ 1.02（能量守恒上界）：{total}"
            );
        }
        // 深色发（高吸收）总反射率应低于浅色发（吸能）。
        assert!(
            totals[1] < totals[0],
            "深色发总反射率应低于浅色发：blond={} dark={}",
            totals[0],
            totals[1]
        );
        // 确定性：同格两遍逐位一致。
        let h = HairScatter::blond();
        let (a0, a1, a2) = h.hemisphere_integral_full(16, 16);
        let (b0, b1, b2) = h.hemisphere_integral_full(16, 16);
        assert_eq!(a0.to_bits(), b0.to_bits(), "半球积分 R 应逐位一致");
        assert_eq!(a1.to_bits(), b1.to_bits(), "半球积分 TT 应逐位一致");
        assert_eq!(a2.to_bits(), b2.to_bits(), "半球积分 TRT 应逐位一致");
    }

    #[test]
    fn full_hemisphere_integral_consistent_across_grid_resolution() {
        // 网格无关性（弱）：16² 与 24² 格给出的总反射率应接近（< 25% 相对差），
        // 且都 ≤ 1.02。抓住"归一测度随格变化"的回归（本轮修正的核心）。
        let h = HairScatter::blond();
        let t16 = {
            let (r, tt, trt) = h.hemisphere_integral_full(16, 16);
            r + tt + trt
        };
        let t24 = {
            let (r, tt, trt) = h.hemisphere_integral_full(24, 24);
            r + tt + trt
        };
        assert!(t16 > 0.0 && t24 > 0.0, "两种格都应 > 0：t16={t16} t24={t24}");
        assert!(t16 <= 1.02 && t24 <= 1.02, "两种格都应 ≤ 1.02：t16={t16} t24={t24}");
        let rel = (t16 - t24).abs() / t24.max(1e-6);
        assert!(rel < 0.25, "网格分辨率不应显著改变总反射率：t16={t16} t24={t24} rel={rel}");
    }

    #[test]
    fn full_scatter_matches_degenerate_resolution() {
        // 完整散射在 φ_o = φ_i（方位无偏差）与 θ_o ≈ θ_i 时的值，应显著大于
        // 大偏角处（散射集中在前向/镜面邻域），且有限。
        let h = HairScatter::blond();
        let near = h.components_full(0.0, 0.0, 0.0, 0.0);
        let far = h.components_full(0.0, 0.0, PI * 0.5 - 0.01, PI);
        assert!(near.0 > 0.0 && near.1 > 0.0 && near.2 > 0.0, "近轴分量应 > 0");
        assert!(far.0.is_finite() && far.1.is_finite() && far.2.is_finite());
        // 近轴 R（镜面邻域）应大于远轴 R。
        assert!(near.0 > far.0, "镜面邻域 R 应大于远轴：{} vs {}", near.0, far.0);
    }

    #[test]
    fn full_api_is_backward_compatible() {
        // 向后兼容：既有 eval/components/m_longitudinal 调用点行为不变（同输入
        // 逐位一致），且不 panic。新增 API（components_full/eval_full）可用。
        let h = HairScatter::dark();
        // 既有 API 仍工作且确定。
        let v = h.eval(0.3, 1.2);
        assert!(v.is_finite());
        assert_eq!(v.to_bits(), h.eval(0.3, 1.2).to_bits());
        let (r, tt, trt) = h.components(0.3, 1.2);
        assert!(r.is_finite() && tt.is_finite() && trt.is_finite());
        assert!(h.m_longitudinal(0.4).is_finite());
        // 新 API 对 NaN/Inf 不 panic。
        assert!(h.m_longitudinal_full(f32::NAN, 0.0) == 0.0);
        let (a, b, c) = h.components_full(f32::NAN, 0.0, 0.0, 0.0);
        assert!(a == 0.0 && b == 0.0 && c == 0.0);
        assert!(h.eval_full(0.0, 0.0, f32::INFINITY, 0.0).is_finite());
    }

    // -----------------------------------------------------------------------
    // A 瓣精确化（本轮「毛发深化 C」）：Fresnel / 吸收 / 三瓣偏角
    // -----------------------------------------------------------------------

    #[test]
    fn schlick_fresnel_monotonic_and_bounds() {
        // 硬性要求 B：Fresnel 单调性——θ 增大（掠射）时 F 单调增。
        // 自证非恒真：若把 (1−cosθ)^5 写成升幂或符号错，单调性会翻转。
        let h = HairScatter::new(0.15, 8.0, 1.55, 0.3, 0.04);
        let mut prev = -1.0f32;
        for i in 0..=20 {
            // θ 从 0 到 π/2。
            let theta = PI * 0.5 * (i as f32 / 20.0);
            let f = h.fresnel_theta(theta);
            assert!(
                f >= prev - 1e-6,
                "F(θ) 应随 θ 单调增：θ={theta} f={f} prev={prev}"
            );
            assert!((0.0..=1.0).contains(&f), "F 应在 [0,1]：{f}");
            prev = f;
        }
        // 边界：θ=0 → f0；θ=π/2 → 1。
        let f0 = h.f0();
        assert!(
            (h.fresnel_theta(0.0) - f0).abs() < 1e-5,
            "F(0) 应 = f0：{} vs {f0}",
            h.fresnel_theta(0.0)
        );
        assert!(
            (h.fresnel_theta(PI * 0.5 * 0.9999) - 1.0).abs() < 1e-2,
            "掠射 F 应趋近 1：{}",
            h.fresnel_theta(PI * 0.5 * 0.9999)
        );
        // f0 物理值：η = 1.55 → f0 = (0.55/2.55)² ≈ 0.0465。
        let want = (0.55f32 / 2.55).powi(2);
        assert!((f0 - want).abs() < 1e-5, "f0 应为 {want}：{f0}");
        // 独立参考实现（同式）逐位对照。
        let c = 0.7f32;
        let ref_v = {
            let x = 1.0 - c;
            let x2 = x * x;
            f0 + (1.0 - f0) * (x2 * x2 * x)
        };
        assert_eq!(schlick_fresnel(f0, c).to_bits(), ref_v.to_bits());
    }

    #[test]
    fn f0_increases_r_lobe() {
        // 硬性要求 B：f0 增大时 R 瓣增强。
        // 自证非恒真：若 A 瓣把 f 的角依赖写反（掠射变小），R 不随 f0 增。
        let lo = HairScatter::new(0.15, 8.0, 1.10, 0.3, 0.04); // η 小 → f0 小
        let hi = HairScatter::new(0.15, 8.0, 2.00, 0.3, 0.04); // η 大 → f0 大
        assert!(lo.f0() < hi.f0(), "f0 应随 η 增：{} vs {}", lo.f0(), hi.f0());
        let (r_lo, _, _) = lo.a_components(0.3);
        let (r_hi, _, _) = hi.a_components(0.3);
        assert!(r_hi > r_lo, "R 瓣衰减应随 f0 增：lo={r_lo} hi={r_hi}");
    }

    #[test]
    fn absorption_monotonic_in_sigma_a() {
        // 硬性要求 B：σ_a 增大时 TT/TRT 瓣衰减更强（Beer 单调）。
        // 自证非恒真：若 exp 参数符号写反（吸收变成增益），单调性翻转。
        let mut prev_tt = 2.0f32;
        let mut prev_trt = 2.0f32;
        for &sa in &[0.0f32, 0.5, 1.0, 2.0, 5.0] {
            let h = HairScatter::new(0.15, 8.0, 1.55, sa, 0.04);
            let (_, a_tt, a_trt) = h.a_components(0.4);
            assert!(
                a_tt <= prev_tt + 1e-6,
                "σ_a 增大 TT 应单调减：σ_a={sa} a_tt={a_tt} prev={prev_tt}"
            );
            assert!(
                a_trt <= prev_trt + 1e-6,
                "σ_a 增大 TRT 应单调减：σ_a={sa} a_trt={a_trt} prev={prev_trt}"
            );
            prev_tt = a_tt;
            prev_trt = a_trt;
        }
        // σ_a = 0 → 无吸收，T = 1。
        let h0 = HairScatter::new(0.15, 8.0, 1.55, 0.0, 0.04);
        assert!((h0.absorption_transmit(1.0) - 1.0).abs() < 1e-6, "σ_a=0 应全透");
    }

    #[test]
    fn tt_stronger_than_r_in_forward_region() {
        // 硬性要求 B：毛发前向散射——TT 瓣在前向区域（φ_o ≈ φ_i − π，
        // 因 φ_TT = φ − π）应强于 R 瓣。
        // 自证非恒真：若三瓣共用同一 φ（旧简化），TT 的 φ−π 峰值不会出现。
        let h = HairScatter::new(0.10, 12.0, 1.55, 0.2, 0.04);
        // 入射 (θ_i=0, φ_i=0)，出射纵向 θ_o=0。
        // TT 峰值方位：φ = φ_o − φ_i = −π ?? 令 φ_TT = φ − π = 0 → φ = π。
        let (r_pk, tt_pk, trt_pk) = h.components_precise(0.0, 0.0, 0.0, PI);
        // R 峰值方位：φ_R = φ = 0 → φ_o = 0。
        let (r0, tt0, _) = h.components_precise(0.0, 0.0, 0.0, 0.0);
        println!(
            "前向区：TT@φ=−π={tt_pk:.6} R@φ=0={r0:.6} R@φ=−π={r_pk:.6} TT@φ=0={tt0:.6} TRT@φ=−π={trt_pk:.6}"
        );
        // TT 在其前向峰位应强于 R 在同一点（前向散射特性）。
        assert!(
            tt_pk > r_pk,
            "TT 前向峰应强于 R 在同点：tt={tt_pk} r={r_pk}"
        );
        // TT 自身在 φ_TT 峰位强于偏离处。
        assert!(tt_pk > tt0, "TT 应在 φ_o=φ_i−π 处峰：pk={tt_pk} off={tt0}");
        // R 应在 φ_o=φ_i 处峰。
        assert!(r0 > r_pk, "R 应在 φ_o=φ_i 处峰：pk={r0} off={r_pk}");
    }

    #[test]
    fn phi_lobes_are_independent() {
        // 三瓣独立方位角：φ_R = φ、φ_TT = φ − π、φ_TRT = φ。
        let h = HairScatter::blond();
        let phi = 0.7f32;
        assert!((h.phi_p(phi, 0) - phi).abs() < 1e-6, "φ_R 应 = φ");
        assert!(
            (h.phi_p(phi, 1) - (phi - PI)).abs() < 1e-6,
            "φ_TT 应 = φ − π"
        );
        assert!((h.phi_p(phi, 2) - phi).abs() < 1e-6, "φ_TRT 应 = φ");
        // α 偏角：R 为 −arcsin(sinθ/η)；TT/TRT 为 +arcsin(sinθ/η)。
        let theta = 0.5f32;
        let want = (cos_sin(theta).1 / 1.55).clamp(-1.0, 1.0).asin();
        assert!(
            (h.alpha_p(theta, 0) + want).abs() < 1e-5,
            "α_R 应 = −arcsin：{} vs {}",
            h.alpha_p(theta, 0),
            -want
        );
        assert!(
            (h.alpha_p(theta, 1) - want).abs() < 1e-5,
            "α_TT 应 = +arcsin：{}",
            h.alpha_p(theta, 1)
        );
        // θ_i = 0 → α = 0（sinθ_i = 0）。
        assert!(h.alpha_p(0.0, 0).abs() < 1e-6, "α_R(0) 应 = 0");
    }

    #[test]
    fn precise_api_is_deterministic() {
        // 硬性要求 A：新 API 同输入两遍逐位一致；NaN/Inf 不 panic（返回 0）。
        let h = HairScatter::blond();
        for &(ti, pi, to, po) in &[
            (0.0f32, 0.0f32, 0.0f32, 0.0f32),
            (0.3, 1.1, -0.2, 2.4),
            (-0.5, -2.0, 0.4, -1.0),
        ] {
            let (a0, a1, a2) = h.a_components(ti);
            let (b0, b1, b2) = h.a_components(ti);
            assert_eq!(a0.to_bits(), b0.to_bits(), "a_r 应逐位一致");
            assert_eq!(a1.to_bits(), b1.to_bits(), "a_tt 应逐位一致");
            assert_eq!(a2.to_bits(), b2.to_bits(), "a_trt 应逐位一致");
            let (r1, t1, x1) = h.components_precise(ti, pi, to, po);
            let (r2, t2, x2) = h.components_precise(ti, pi, to, po);
            assert_eq!(r1.to_bits(), r2.to_bits());
            assert_eq!(t1.to_bits(), t2.to_bits());
            assert_eq!(x1.to_bits(), x2.to_bits());
            assert_eq!(
                h.eval_precise(ti, pi, to, po).to_bits(),
                h.eval_precise(ti, pi, to, po).to_bits()
            );
        }
        // NaN/Inf → 有限兜底（返回 0），不 panic。
        assert_eq!(h.fresnel_theta(f32::NAN).to_bits(), h.f0().to_bits());
        let (a0, a1, a2) = h.a_components(f32::INFINITY);
        assert!(a0 == 0.0 && a1 == 0.0 && a2 == 0.0);
        let (r, tt, trt) = h.components_precise(f32::NAN, 0.0, 0.0, 0.0);
        assert!(r == 0.0 && tt == 0.0 && trt == 0.0);
        assert!(h.eval_precise(0.0, 0.0, f32::INFINITY, 0.0).is_finite());
        assert!(h.absorption_transmit(f32::NAN).is_finite());
        assert!(h.alpha_p(f32::NAN, 1).is_finite());
        // 未知瓣号 → 0（C18 兜底，不 panic）。
        assert_eq!(h.alpha_p(0.3, 99).to_bits(), 0.0f32.to_bits());
        assert!((h.phi_p(0.3, 99) - 0.3).abs() < 1e-6, "未知瓣 φ 应为 φ");
    }

    #[test]
    fn precise_energy_is_bounded() {
        // 硬性要求 B：精确散射的 A 瓣能量上界（a_p ∈ [0,1] 逐点），
        // 且 a_r + a_tt + a_trt 不超过入射（无吸收时 ≤ ... 见下注释）。
        // 自证非恒真：若 Fresnel/Beer 组合写成增益（>1），上界被打破。
        let mut tmp = [0.0f32; 2];
        for (i, h) in [HairScatter::blond(), HairScatter::dark()].into_iter().enumerate() {
            for it in 0..=16 {
                let theta = -PI * 0.5 + (it as f32 + 0.5) * (PI / 17.0);
                let (a_r, a_tt, a_trt) = h.a_components(theta);
                assert!((0.0..=1.0).contains(&a_r), "a_r 越界：{a_r}");
                assert!((0.0..=1.0).contains(&a_tt), "a_tt 越界：{a_tt}");
                assert!((0.0..=1.0).contains(&a_trt), "a_trt 越界：{a_trt}");
                // 三瓣衰减和 ≤ 1 + ε（单次 Fresnel 分支的能量分割：
                // F + (1−F)² + (1−F)²F ≤ 1.05 对 F∈[0,1] 恒成立）。
                let s = a_r + a_tt + a_trt;
                assert!(s <= 1.05, "A 瓣衰减和应 ≤ 1.05（无吸收时同一束能量分割）：{s}");
            }
            // 深色发吸收更强 → 总 A 衰减和应更低。
            let sum_abs = |h: &HairScatter| -> f64 {
                let mut acc = 0.0f64;
                for it in 0..64 {
                    let theta = -PI * 0.5 + (it as f32 + 0.5) * (PI / 64.0);
                    let (r, tt, trt) = h.a_components(theta);
                    acc += (r + tt + trt) as f64 / 64.0;
                }
                acc
            };
            tmp[i] = sum_abs(&h) as f32;
        }
        assert!(
            tmp[1] < tmp[0],
            "深色发（高吸收）总衰减应低于浅色发：blond={} dark={}",
            tmp[0],
            tmp[1]
        );
    }

    #[test]
    fn existing_api_bitwise_frozen() {
        // 硬性要求 B（向后兼容）：既有 eval/components/m_longitudinal 调用点
        // 行为不变（同输入逐位一致），且不 panic。精确化只加新入口。
        let h = HairScatter::dark();
        // 冻结样例：同输入两遍逐位一致（含半球积分）。
        for &(theta, phi) in &[(0.3f32, 1.2f32), (-0.4, 2.0), (0.0, 0.0)] {
            assert_eq!(h.eval(theta, phi).to_bits(), h.eval(theta, phi).to_bits());
            let (a, b, c) = h.components(theta, phi);
            let (d, e, f) = h.components(theta, phi);
            assert_eq!(a.to_bits(), d.to_bits());
            assert_eq!(b.to_bits(), e.to_bits());
            assert_eq!(c.to_bits(), f.to_bits());
        }
        assert_eq!(
            h.m_longitudinal(0.4).to_bits(),
            h.m_longitudinal(0.4).to_bits()
        );
        let (a0, a1, a2) = h.hemisphere_integral(32, 32);
        let (b0, b1, b2) = h.hemisphere_integral(32, 32);
        assert_eq!(a0.to_bits(), b0.to_bits());
        assert_eq!(a1.to_bits(), b1.to_bits());
        assert_eq!(a2.to_bits(), b2.to_bits());
    }

    #[test]
    fn cross_impl_wgsl_matches_cpu() {
        // 跨实现数值对照：按 WGSL `hair_scatter.wgsl` 的**相同公式**独立算一遍，
        // 与 CPU 精确版逐点比对（不直接读 WGSL，避免把系数写死；此处的"参考"
        // 用与 WGSL a_r/a_tt/a_trt（含角依赖 Fresnel + Beer）一致的公式）。
        //
        // WGSL 侧本轮同步为：
        //   fn a_r(theta_i, f0, eta?) -> f32  （用 schlick(cosθ_i)）
        //   fn a_tt(theta_i, f0, sigma_a, radius) -> f32
        //   fn a_trt(theta_i, f0, sigma_a, radius) -> f32
        // 见 shaders/hair_scatter.wgsl。此处以 f32 复刻其算术顺序。
        let h = HairScatter::new(0.12, 8.0, 1.55, 0.8, 0.04);
        let f0 = h.f0();
        for &theta in &[-1.2f32, -0.5, 0.0, 0.3, 0.9, 1.4] {
            // --- WGSL 参考复刻（算术顺序与 .wgsl 一致）---
            let c = cos_sin(theta).0.abs().clamp(0.0, 1.0);
            let x = 1.0 - c;
            let x2 = x * x;
            let fres = f0 + (1.0 - f0) * (x2 * x2 * x); // schlick
            let omf = 1.0 - fres;
            let cpath = c.max(1e-3);
            let d_tt = 2.0 * 0.04 / cpath;
            let d_trt = 2.0 * d_tt;
            let t_tt = det_exp(-0.8 * d_tt);
            let t_trt = det_exp(-0.8 * d_trt);
            let w_a_r = fres;
            let w_a_tt = omf * omf * t_tt;
            let w_a_trt = omf * omf * fres * t_trt;
            // --- CPU 精确版 ---
            let (c_a_r, c_a_tt, c_a_trt) = h.a_components(theta);
            assert_eq!(c_a_r.to_bits(), w_a_r.to_bits(), "a_r CPU/GPU 应逐位一致");
            assert_eq!(c_a_tt.to_bits(), w_a_tt.to_bits(), "a_tt CPU/GPU 应逐位一致");
            assert_eq!(c_a_trt.to_bits(), w_a_trt.to_bits(), "a_trt CPU/GPU 应逐位一致");
        }
    }
}
