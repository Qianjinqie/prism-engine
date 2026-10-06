//! 多段 EQ（B12）：级联 biquad（RBJ Audio EQ Cookbook 系数），低架 / 峰 / 高架。
//!
//! 每段是一个 Direct-Form-I 双二次滤波器，系数由中心频率/增益/Q 在 `process`
//! 前一次性算出（仅 ±×÷ 与 [`crate::det`] 三角件，D-29 确定性）。多段按**固定
//! 顺序**级联（低 → 中 → 高），同输入逐位一致。
//!
//! 系数公式（RBJ）：
//! ```text
//! ω0 = 2π·f0/SR,  c = cosω0,  s = sinω0,  α = s/(2Q)
//! 峰(peaking):  A = 10^(dBgain/40)
//!   b0 = 1 + αA, b1 = −2c, b2 = 1 − αA
//!   a0 = 1 + α/A, a1 = −2c, a2 = 1 − α/A
//! 低架(lowShelf):  A, α = s/2·√((A+1/A)(1/S−1)+2), β = 2√A·α
//!   b = A[(A+1) − (A−1)c + β],  …（见下实现）
//! ```

use super::MonoFx;
use crate::det;

/// 滤波段类型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shelf {
    /// 峰值均衡（bell）。
    Peaking,
    /// 低架。
    LowShelf,
    /// 高架。
    HighShelf,
}

/// 单个 EQ 段：类型 + 中心/转折频率 + 增益（dB）+ Q。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqBand {
    /// 段类型。
    pub kind: Shelf,
    /// 中心/转折频率（Hz）。
    pub freq: f32,
    /// 增益（dB；正提升、负衰减）。
    pub gain_db: f32,
    /// 品质因数 Q。
    pub q: f32,
}

impl EqBand {
    /// 计算该段的 biquad 系数 `(b0,b1,b2,a1,a2)`（a0 已归一，a1/a2 符号按
    /// `y = b0x + b1x₁ + b2x₂ − a1y₁ − a2y₂` 口径）。
    ///
    /// **退化采样率**（C18 永不panic）：`sr = 0` 时 Nyquist 上界 `sr·0.49 = 0`，
    /// 直接 `.clamp(1.0, 0.0)` 会因 `min > max` panic（实测在 debug 下崩）。
    /// 故对退化/过小采样率直接返回**零系数旁通**——`b0 = 0` 使该段输出恒 0，
    /// 语义为「该采样率下该 EQ 段不参与运算」。**诚实边界**：这不是单位增益
    /// 响应（`magnitude_at` 在退化 SR 下会得 0 而非 1.0）；退化采样率本不是
    /// 有效音频输入（Nyquist 为 0，无可表示频率），故取「不参与」而非「透传」。
    pub fn coeffs(&self, sample_rate: u32) -> [f32; 5] {
        const BYPASS: [f32; 5] = [0.0; 5];
        let sr = sample_rate as f64;
        // 上界不得低于下界 1.0，否则 clamp panic；且 sr 过小会使 w0 溢出为非有限。
        // （写成 `sr < 8.0 ||` 而非 `!(sr >= 8.0)`：后者对 NaN 成立但触发
        // clippy::neg_cmp_op_on_partial_ord；本处 sr 来自 u32 转换不可能 NaN。）
        if sr < 8.0 || !self.freq.is_finite() {
            return BYPASS;
        }
        let f0 = (self.freq as f64).clamp(1.0, sr * 0.49);
        let w0 = 2.0 * std::f64::consts::PI * f0 / sr;
        if !w0.is_finite() {
            return BYPASS;
        }
        let c = det::cos_f64(w0);
        let s = det::sin_f64(w0);
        let q = (self.q as f64).max(1e-3);
        let a = pow10(self.gain_db as f64 / 40.0);
        match self.kind {
            Shelf::Peaking => {
                let alpha = s / (2.0 * q);
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * c;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * c;
                let a2 = 1.0 - alpha / a;
                [(b0 / a0) as f32, (b1 / a0) as f32, (b2 / a0) as f32, (a1 / a0) as f32, (a2 / a0) as f32]
            }
            Shelf::LowShelf => {
                let alpha = s / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
                let beta = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * c + beta);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * c);
                let b2 = a * ((a + 1.0) - (a - 1.0) * c - beta);
                let a0 = (a + 1.0) + (a - 1.0) * c + beta;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * c);
                let a2 = (a + 1.0) + (a - 1.0) * c - beta;
                [(b0 / a0) as f32, (b1 / a0) as f32, (b2 / a0) as f32, (a1 / a0) as f32, (a2 / a0) as f32]
            }
            Shelf::HighShelf => {
                let alpha = s / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
                let beta = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * c + beta);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
                let b2 = a * ((a + 1.0) + (a - 1.0) * c - beta);
                let a0 = (a + 1.0) - (a - 1.0) * c + beta;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
                let a2 = (a + 1.0) - (a - 1.0) * c - beta;
                [(b0 / a0) as f32, (b1 / a0) as f32, (b2 / a0) as f32, (a1 / a0) as f32, (a2 / a0) as f32]
            }
        }
    }
}

/// 10^x（确定性）：`10^x = e^(x·ln10)`，走 [`det::exp_f64`]（D-29 禁 libm pow）。
#[inline]
fn pow10(x: f64) -> f64 {
    det::exp_f64(x * std::f64::consts::LN_10)
}

/// 多段参数均衡。
#[derive(Debug, Clone, PartialEq)]
pub struct MultiEq {
    /// 段列表（按此顺序级联）。
    pub bands: Vec<EqBand>,
}

impl MultiEq {
    /// 由段列表构造。
    pub fn new(bands: Vec<EqBand>) -> Self {
        Self { bands }
    }

    /// 该滤波器在频率 `f` 的幅度响应（Hz）——供频响断言与可视化。
    pub fn magnitude_at(&self, f: f32, sample_rate: u32) -> f32 {
        let mut mag = 1.0_f64;
        for band in &self.bands {
            let [b0, b1, b2, a1, a2] = band.coeffs(sample_rate);
            // H(e^{jω}) 用 z = e^{jω} 直接求值（ω = 2πf/SR）。
            let w = 2.0 * std::f64::consts::PI * f as f64 / sample_rate as f64;
            let (cw, sw) = (det::cos_f64(w), det::sin_f64(w));
            // z^-1 = cos − j·sin 的共轭两点，逐项复数乘加。
            let num = cplx_poly(b0 as f64, b1 as f64, b2 as f64, cw, sw);
            let den = cplx_poly(1.0, a1 as f64, a2 as f64, cw, sw);
            mag *= num / den;
        }
        mag as f32
    }
}

/// `|b0 + b1·z⁻¹ + b2·z⁻²|`，`z⁻¹ = cosω − j·sinω`（返回模）。
fn cplx_poly(b0: f64, b1: f64, b2: f64, cw: f64, sw: f64) -> f64 {
    // z⁻¹ = cw − j·sw；z⁻² = (cw²−sw²) − j·2cw·sw。
    let z1r = cw;
    let z1i = -sw;
    let z2r = cw * cw - sw * sw;
    let z2i = -2.0 * cw * sw;
    let re = b0 + b1 * z1r + b2 * z2r;
    let im = b1 * z1i + b2 * z2i;
    (re * re + im * im).sqrt()
}

impl MonoFx for MultiEq {
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut signal = input.to_vec();
        // 每段独立 DF-I 滤波，逐段覆盖（固定顺序）。
        for band in &self.bands {
            let [b0, b1, b2, a1, a2] = band.coeffs(sample_rate);
            let (mut x1, mut x2, mut y1, mut y2) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for x in signal.iter_mut() {
                let xin = *x;
                let y = b0 * xin + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = xin;
                y2 = y1;
                y1 = y;
                *x = y;
            }
        }
        signal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 峰段：中心频率增益 ≈ +6 dB（幅度 ×2）；远离中心 ≈ 0 dB。
    #[test]
    fn peaking_band_frequency_response() {
        let eq = MultiEq::new(vec![EqBand {
            kind: Shelf::Peaking,
            freq: 1_000.0,
            gain_db: 6.0,
            q: 1.0,
        }]);
        let at_center = eq.magnitude_at(1_000.0, 48_000);
        let want = db_to_lin_local(6.0);
        assert!((at_center - want).abs() < want * 0.05, "中心增益 {at_center} 应 ≈ {want}");
        let far_low = eq.magnitude_at(50.0, 48_000);
        assert!((far_low - 1.0).abs() < 0.05, "远离中心应 ≈ 0 dB，实际 {far_low}");
    }

    /// 低架：低频提升，高频不变。
    #[test]
    fn low_shelf_boosts_low_only() {
        let eq = MultiEq::new(vec![EqBand {
            kind: Shelf::LowShelf,
            freq: 200.0,
            gain_db: 6.0,
            q: 0.707,
        }]);
        assert!(eq.magnitude_at(20.0, 48_000) > 1.5, "低频应提升");
        assert!((eq.magnitude_at(10_000.0, 48_000) - 1.0).abs() < 0.1, "高频应不变");
    }

    /// 高架：高频提升，低频不变。
    #[test]
    fn high_shelf_boosts_high_only() {
        let eq = MultiEq::new(vec![EqBand {
            kind: Shelf::HighShelf,
            freq: 4_000.0,
            gain_db: -6.0,
            q: 0.707,
        }]);
        assert!(eq.magnitude_at(15_000.0, 48_000) < 0.8, "高频应衰减");
        assert!((eq.magnitude_at(100.0, 48_000) - 1.0).abs() < 0.1, "低频应不变");
    }

    /// 多段级联频响 = 各段乘积；处理确定性。
    #[test]
    fn cascade_and_determinism() {
        let eq = MultiEq::new(vec![
            EqBand { kind: Shelf::LowShelf, freq: 150.0, gain_db: 4.0, q: 0.707 },
            EqBand { kind: Shelf::HighShelf, freq: 6_000.0, gain_db: -4.0, q: 0.707 },
        ]);
        let x: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.05).sin()).collect();
        let first = eq.process(&x, 48_000);
        for _ in 0..50 {
            assert_eq!(eq.process(&x, 48_000), first);
        }
        assert!(first.iter().all(|v| v.is_finite()), "滤波不应发散");
    }

    /// 局部 db→lin（测试内，避免与 compressor 循环依赖）。
    fn db_to_lin_local(db: f32) -> f32 {
        super::pow10(db as f64 / 20.0) as f32
    }
}
