//! 状态变量滤波器 SVF（合成器升级）：多模态 LP/HP/BP/Notch（TPT 拓扑）。
//!
//! 采用 **Topology-Preserving Transform**（Zavalishin）零延迟反馈 SVF——数值
//! 稳定、通带到 Nyquist、系数只依赖 `tan(πf/SR)`。三状态 `(ic1, ic2)` 递推：
//!
//! ```text
//! g = tan(π·fc/SR)              （确定性三角件，D-29）
//! k = 1/Q
//! a1 = 1/(1 + g(g+k)); a2 = g·a1; a3 = g·a2
//! v3 = x − ic2;  v1 = a1·ic1 + a2·v3;  v2 = ic2 + a2·ic1 + a3·v3
//! ic1 = 2v1 − ic1;  ic2 = 2v2 − ic2
//! LP = v2;  BP = v1;  HP = x − k·v1 − v2;  Notch = HP + LP = x − k·v1
//! ```
//!
//! 因含反馈状态，滤波器是 x 序列的因果递推——`process` 内单遍索引递增（固定序），
//! 无跨调用状态、无 RNG，同输入逐位一致。

use crate::det;

/// 滤波模态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvfMode {
    /// 低通。
    LowPass,
    /// 高通。
    HighPass,
    /// 带通。
    BandPass,
    /// 陷波。
    Notch,
}

/// SVF 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Svf {
    /// 截止/中心频率（Hz）。
    pub cutoff_hz: f32,
    /// 谐振 Q（0.5 = 临界阻尼 … 越大共振越尖）。
    pub q: f32,
    /// 模态。
    pub mode: SvfMode,
}

impl Default for Svf {
    fn default() -> Self {
        Self { cutoff_hz: 1_000.0, q: 0.707, mode: SvfMode::LowPass }
    }
}

impl Svf {
    /// 由截止频率与采样率算 TPT 系数 `(g, k, a1, a2, a3)`。
    fn coeffs(&self, sample_rate: u32) -> (f32, f32, f32, f32, f32) {
        let fc = self.cutoff_hz.clamp(1.0, sample_rate as f32 * 0.49);
        let w = std::f64::consts::PI * fc as f64 / sample_rate as f64;
        let g = tan_det(w) as f32;
        let k = 1.0 / self.q.max(1e-3);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        (g, k, a1, a2, a3)
    }

    /// 逐样本滤波（含状态递推，单遍固定序）。
    pub fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let (g, k, a1, a2, a3) = self.coeffs(sample_rate);
        let _ = g;
        let mut ic1 = 0.0_f32;
        let mut ic2 = 0.0_f32;
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            let v3 = x - ic2;
            let v1 = a1 * ic1 + a2 * v3;
            let v2 = ic2 + a2 * ic1 + a3 * v3;
            ic1 = 2.0 * v1 - ic1;
            ic2 = 2.0 * v2 - ic2;
            let y = match self.mode {
                SvfMode::LowPass => v2,
                SvfMode::BandPass => v1,
                SvfMode::HighPass => x - k * v1 - v2,
                SvfMode::Notch => x - k * v1,
            };
            out.push(y);
        }
        out
    }
}

/// `tan(x)`，`x ∈ (−π/2, π/2)`：`tan = sin/cos`（确定性三角件，D-29）。
#[inline]
fn tan_det(x: f64) -> f64 {
    let c = det::cos_f64(x);
    det::sin_f64(x) / if c.abs() < 1e-12 { 1e-12 } else { c }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 频响：低通在直流增益 ≈ 1，高频被抑制。
    #[test]
    fn lowpass_frequency_response() {
        let svf = Svf { cutoff_hz: 1_000.0, q: 0.707, mode: SvfMode::LowPass };
        let dc = gain_at(&svf, 50.0, 48_000);
        let high = gain_at(&svf, 15_000.0, 48_000);
        assert!(dc > 0.9, "直流增益 {dc} 应 ≈ 1");
        assert!(high < 0.2, "高频增益 {high} 应被压低");
    }

    /// 高通：与低通互补（直流抑制、高频通过）。
    #[test]
    fn highpass_complements_lowpass() {
        let hp = Svf { cutoff_hz: 1_000.0, q: 0.707, mode: SvfMode::HighPass };
        assert!(gain_at(&hp, 50.0, 48_000) < 0.2, "高通应抑制直流");
        assert!(gain_at(&hp, 15_000.0, 48_000) > 0.8, "高通应通过高频");
    }

    /// 带通：中心频率增益 ≈ 1，两侧下降。
    #[test]
    fn bandpass_peaks_at_center() {
        let bp = Svf { cutoff_hz: 2_000.0, q: 2.0, mode: SvfMode::BandPass };
        let center = gain_at(&bp, 2_000.0, 48_000);
        let below = gain_at(&bp, 200.0, 48_000);
        let above = gain_at(&bp, 15_000.0, 48_000);
        assert!(center > below && center > above, "带通应在中心峰");
    }

    /// 稳定性：处理不发散；确定性复算逐位一致。
    #[test]
    fn svf_stable_and_deterministic() {
        let svf = Svf { cutoff_hz: 3_000.0, q: 1.5, mode: SvfMode::LowPass };
        let x: Vec<f32> = (0..2_000).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = svf.process(&x, 48_000);
        assert!(a.iter().all(|v| v.is_finite()), "不应发散");
        for _ in 0..50 {
            assert_eq!(svf.process(&x, 48_000), a);
        }
    }

    /// 正弦扫频测增益（测试工具）：稳态幅度比。
    #[test]
    fn notch_rejects_center() {
        let notch = Svf { cutoff_hz: 1_000.0, q: 1.0, mode: SvfMode::Notch };
        let center = gain_at(&notch, 1_000.0, 48_000);
        let away = gain_at(&notch, 100.0, 48_000);
        assert!(center < 0.3, "陷波应抑制中心频率：{center}");
        assert!(away > 0.7, "远离中心应通过：{away}");
    }

    /// 稳态增益估计：跳过瞬态后取后段 RMS / 输入 RMS。
    fn gain_at(svf: &Svf, freq: f32, sr: u32) -> f32 {
        let n = 16_384_usize;
        let x: Vec<f32> = (0..n)
            .map(|i| det::sin_f64(std::f64::consts::TAU * freq as f64 * i as f64 / sr as f64) as f32)
            .collect();
        let y = svf.process(&x, sr);
        // 稳态段（后半）。
        let tail = &y[n / 2..];
        let in_tail = &x[n / 2..];
        let rms = |s: &[f32]| (s.iter().map(|&v| (v * v) as f64).sum::<f64>() / s.len() as f64).sqrt();
        (rms(tail) / rms(in_tail)) as f32
    }
}
