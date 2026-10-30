//! 光谱渲染切片（V2.2-B-1 / T08）：λ 轴 + 自研 CMF 分段有理 + 退化一致性。
//!
//! # 零依赖 / 零超越函数（D-29）
//! CMF 采用**自研分段线性有理式**（分母为常数 1 的多项式，属有理式族），
//! 不着任何 libm 超越函数、不引外部 crate（零依赖宪章）；RGB↔光谱升采样
//! 走 [`flat_spectral_value`]（消色差反射率口径）+ [`spectral_to_rgb`]（CMF
//! 加权积分）。
//!
//! # hero wavelength（PRD Q3①）
//! 每样本 1 个 λ（确定性分层，seed 可复现）；退化模式固定单波长且 CMF 转换
//! 退化为单位复制（见 [`SpectralConfig::degenerate`]）。
//!
//! # 退化一致性（验收 1，核心）
//! 恒定谱反射率 + 单波长下，光谱路径的 `to_rgb` 为单位复制（K=(1,1,1) 的
//! CMF 积分），此时波长维不携带信息，输出与 RGB 路径（灰反照率）**逐字节
//! 相等**（被积函数与 λ 无关 ⇒ 积分退化为常量复制，属精确退化而非近似）。

use prism_math::Vec3;

/// 波长（nm）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wavelength(pub f32);

/// 光谱渲染配置（T08 切片口径）。
#[derive(Debug, Clone, Copy)]
pub struct SpectralConfig {
    /// 退化模式：固定单波长 + `to_rgb` 单位复制（K=(1,1,1)）。
    pub degenerate: bool,
    /// hero 波长采样下界（nm）。
    pub lambda_min: f32,
    /// hero 波长采样上界（nm）。
    pub lambda_max: f32,
    /// 色散玻璃 Cauchy A（n(λ) = A + B/λ²）。
    pub cauchy_a: f32,
    /// 色散玻璃 Cauchy B（n(λ) = A + B/λ²，λ 单位 nm）。
    pub cauchy_b: f32,
}

impl SpectralConfig {
    /// 退化配置：单波长 550nm、无色散（B-1 验收 1 用）。
    pub fn degenerate() -> Self {
        SpectralConfig {
            degenerate: true,
            lambda_min: 550.0,
            lambda_max: 550.0,
            cauchy_a: 1.5,
            cauchy_b: 0.0,
        }
    }

    /// 色散配置：400–700nm hero 带 + Cauchy 玻璃（B-1 验收 3 用）。
    pub fn dispersion() -> Self {
        SpectralConfig {
            degenerate: false,
            lambda_min: 400.0,
            lambda_max: 700.0,
            cauchy_a: 1.45,
            cauchy_b: 4750.0,
        }
    }
}

/// hero 波长采样：u∈[0,1) → λ（退化模式返回固定中值）。
///
/// 分层：用样本流的 λ 维均匀量；多样本自动铺满 [λmin, λmax]（hero wavelength
/// Monte Carlo，PRD Q3①）。
#[inline]
pub fn hero_wavelength(u: f32, cfg: &SpectralConfig) -> f32 {
    if cfg.degenerate {
        return (cfg.lambda_min + cfg.lambda_max) * 0.5;
    }
    let u = u.clamp(0.0, 1.0);
    cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * u
}

/// 色散玻璃折射率（Cauchy，零超越函数；λ 单位 nm）。
#[inline]
pub fn cauchy_ior(cfg: &SpectralConfig, lambda: f32) -> f32 {
    cfg.cauchy_a + cfg.cauchy_b / (lambda * lambda)
}

// ---------------------------------------------------------------------------
// 自研 CMF（分段线性有理式；分母恒 1，属有理式族；零超越函数、零 crate）
// ---------------------------------------------------------------------------
// 锚点自研（非 CIE 实测表）：R 随 λ 增、B 随 λ 减、G 居中峰——足以产生可见
// 色散彩虹，且满足「零依赖 / 零超越」约束。验收 3 只要求红/蓝质心可机械分离。

/// R 通道 CMF 锚点 (λ_nm, value)。
const CMF_R: &[(f32, f32)] = &[
    (400.0, 0.02),
    (500.0, 0.10),
    (550.0, 0.45),
    (600.0, 0.85),
    (700.0, 1.00),
];
/// G 通道 CMF 锚点 (λ_nm, value)。
const CMF_G: &[(f32, f32)] = &[
    (400.0, 0.10),
    (500.0, 0.55),
    (550.0, 1.00),
    (600.0, 0.60),
    (700.0, 0.12),
];
/// B 通道 CMF 锚点 (λ_nm, value)。
const CMF_B: &[(f32, f32)] = &[
    (400.0, 1.00),
    (500.0, 0.45),
    (550.0, 0.18),
    (600.0, 0.06),
    (700.0, 0.02),
];

/// 分段线性（有理式退化的特例，分母=1）插值；区间外钳制到端点值。
#[inline]
fn cmf_interp(table: &[(f32, f32)], lambda: f32) -> f32 {
    let n = table.len();
    if lambda <= table[0].0 {
        return table[0].1;
    }
    if lambda >= table[n - 1].0 {
        return table[n - 1].1;
    }
    for i in 0..n - 1 {
        let (l0, v0) = table[i];
        let (l1, v1) = table[i + 1];
        if lambda >= l0 && lambda <= l1 {
            let t = (lambda - l0) / (l1 - l0);
            return v0 + (v1 - v0) * t;
        }
    }
    table[n - 1].1
}

/// 自研 CMF 在 λ 处的 (R,G,B) 权重（**原始**形状；见 [`spectral_to_rgb`] 的白点归一）。
#[inline]
pub fn cmf_rgb(lambda: f32) -> Vec3 {
    Vec3::new(
        cmf_interp(CMF_R, lambda),
        cmf_interp(CMF_G, lambda),
        cmf_interp(CMF_B, lambda),
    )
}

// ---------------------------------------------------------------------------
// 白点归一（等能谱 E ⇒ 中性 (1,1,1)）
// ---------------------------------------------------------------------------
//
// # 为什么必须有这一层（B-1 验收 3 的实现前提）
//
// 三条 CMF 锚点曲线的**带内一周平均**并不相等（原始 R̄≈0.483 / Ḡ≈0.491 /
// B̄≈0.328）。若不归一，则"均匀铺满采样带的等能谱"经 CMF 加权后 R/B 恒为
// ≈1.47，**任何像素都恒为红主导**——色散（几何分离出的红/蓝色心）被这个
// 常量倾覆彻底淹没，红/蓝簇判据退化为"只有红簇"（T08 实测症状）。
//
// 归一口径取**等能谱白点**（CIE illuminant E 约定）：把每条曲线除以它在
// 采样带 [`CMF_BAND`] 上的一周平均，使 `∫cmf_R = ∫cmf_G = ∫cmf_B`。
// 于是「谱平坦 ⇒ 输出中性」成为实现契约，残余的 r−b 只可能来自**真实的
// λ 相关几何分离**（Cauchy ior 折射方向差），而不是 CMF 的整体系数差。
//
// 归一因子由 `qa_spectral_degeneracy` 的 `cmf_band_means_match_declared`
// 反解验证（数值积分 vs 声明常量，误差 ≤ 1e-6），不是手抄魔数。

/// CMF 白点归一的采样带 `(λ_min, λ_max)`（nm），与 [`SpectralConfig::dispersion`] 同带。
pub const CMF_BAND: (f32, f32) = (400.0, 700.0);

/// R 通道在 [`CMF_BAND`] 上的一周平均（分段线性精确积分 = 144.75 / 300）。
pub const CMF_R_MEAN: f32 = 0.4825;
/// G 通道在 [`CMF_BAND`] 上的一周平均（= 147.25 / 300）。
pub const CMF_G_MEAN: f32 = 0.490_833_34;
/// B 通道在 [`CMF_BAND`] 上的一周平均（= 98.25 / 300）。
pub const CMF_B_MEAN: f32 = 0.327_5;

/// 白点归一因子 `1 / mean`（乘法替代除法：少一次舍入，且与 GPU 侧同口径）。
#[inline]
pub fn cmf_white_balance() -> Vec3 {
    Vec3::new(
        1.0 / CMF_R_MEAN,
        1.0 / CMF_G_MEAN,
        1.0 / CMF_B_MEAN,
    )
}

// ---------------------------------------------------------------------------
// RGB ↔ 光谱升采样（消色差口径切片）
// ---------------------------------------------------------------------------

/// 消色差反射率/辐亮度的"光谱标量"提取：灰反照率 `(c,c,c)` → 标量 `c`。
///
/// B-1 切片仅处理消色差（中性）反射率：恒定谱反射率 = 灰 RGB，其光谱标量
/// 取三通道均值（灰时即 `c`）。彩色光谱升采样留 V2.3（非目标，见 PRD §4.2）。
#[inline]
pub fn flat_spectral_value(rgb: Vec3) -> f32 {
    (rgb.x + rgb.y + rgb.z) * (1.0 / 3.0)
}

/// 光谱标量 → RGB（CMF 加权积分 + 白点归一）。
///
/// - 退化模式：单位复制 `K=(1,1,1)`（被积函数与 λ 无关 ⇒ 精确退化，非近似）；
///   此时输出与 RGB 路径（灰反照率）逐字节相等（验收 1）。
/// - 色散模式：乘自研 CMF(λ) 的**白点归一**权重 `cmf(λ) / mean_band(cmf)`
///   （等能谱 ⇒ 中性；见 [`cmf_white_balance`] 的单点定义）。
///
/// 归一不是"调参"：未归一时三条曲线带内均值不等（≈1.47 的 R/B 常量倾覆），
/// 色散判据会退化为"画面恒红"。归一后 `∫cmf_R = ∫cmf_G = ∫cmf_B`，
/// 逐像素 `r − b` 的符号只由 λ 相关几何（Cauchy ior）决定。
#[inline]
pub fn spectral_to_rgb(scalar: f32, lambda: f32, cfg: &SpectralConfig) -> Vec3 {
    if cfg.degenerate {
        Vec3::splat(scalar)
    } else {
        let w = cmf_rgb(lambda);
        let k = cmf_white_balance();
        Vec3::new(w.x * k.x, w.y * k.y, w.z * k.z) * scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmf_is_monotonic_across_bands() {
        // R 随 λ 增、B 随 λ 减（色散方向正确：长波偏红、短波偏蓝）。
        let r_lo = cmf_rgb(420.0).x;
        let r_hi = cmf_rgb(680.0).x;
        assert!(r_hi > r_lo, "R 应随 λ 增：{r_lo} vs {r_hi}");
        let b_lo = cmf_rgb(420.0).z;
        let b_hi = cmf_rgb(680.0).z;
        assert!(b_lo > b_hi, "B 应随 λ 减：{b_lo} vs {b_hi}");
    }

    #[test]
    fn hero_wavelength_stratifies_and_degenerate_fixed() {
        let disp = SpectralConfig::dispersion();
        let mut prev = -1.0f32;
        for k in 0..64u32 {
            let u = (k as f32 + 0.5) / 64.0;
            let l = hero_wavelength(u, &disp);
            assert!(l >= disp.lambda_min && l <= disp.lambda_max);
            assert!(l > prev, "分层应单调：{prev} vs {l}");
            prev = l;
        }
        let deg = SpectralConfig::degenerate();
        assert_eq!(
            hero_wavelength(0.123, &deg).to_bits(),
            hero_wavelength(0.987, &deg).to_bits(),
            "退化模式 λ 应固定"
        );
    }

    #[test]
    fn cauchy_ior_decreases_with_wavelength() {
        // 正常色散：n(短λ) > n(长λ)。
        let cfg = SpectralConfig::dispersion();
        let n_lo = cauchy_ior(&cfg, 420.0);
        let n_hi = cauchy_ior(&cfg, 680.0);
        assert!(n_lo > n_hi, "短波折射率应更高：{n_lo} vs {n_hi}");
    }
}
