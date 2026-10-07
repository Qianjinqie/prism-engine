//! 失真/饱和：磁带软饱和 / 硬削 / 三次波形整形。
//!
//! 全部为**无记忆非线性**（逐样本映射，纯函数）：`y = f(x)`。三种特性：
//! - [`DistortionKind::Soft`] — `tanh` 风格软饱和（自研 exp 实现，D-29）；
//! - [`DistortionKind::Hard`] — 硬削 `clamp(x·drive, −1, 1)`；
//! - [`DistortionKind::Cubic`] — 三次整形 `x − x³/3`（小信号近线性）。
//!
//! 输出经 `output` 增益与 `mix` 干湿比混合。

use super::MonoFx;
use crate::det;

/// 失真特性。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionKind {
    /// 软饱和（tanh）。
    Soft,
    /// 硬削。
    Hard,
    /// 三次波形整形。
    Cubic,
}

/// 失真参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distortion {
    /// 特性。
    pub kind: DistortionKind,
    /// 驱动增益（输入前乘）。
    pub drive: f32,
    /// 输出增益。
    pub output: f32,
    /// 干湿比 ∈ [0, 1]（0 = 全干，1 = 全湿）。
    pub mix: f32,
}

impl Default for Distortion {
    fn default() -> Self {
        Self { kind: DistortionKind::Soft, drive: 2.0, output: 1.0, mix: 1.0 }
    }
}

/// 软饱和 `tanh(x)`：`(e^{2x}−1)/(e^{2x}+1)`，走确定性 exp（D-29）。
#[inline]
fn tanh_det(x: f64) -> f64 {
    let e = det::exp_f64(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

impl MonoFx for Distortion {
    fn process(&self, input: &[f32], _sample_rate: u32) -> Vec<f32> {
        let mix = self.mix.clamp(0.0, 1.0);
        input
            .iter()
            .map(|&xin| {
                let d = (xin * self.drive) as f64;
                let shaped = match self.kind {
                    DistortionKind::Soft => tanh_det(d),           // ∈ (−1, 1)
                    DistortionKind::Hard => d.clamp(-1.0, 1.0),
                    DistortionKind::Cubic => d - d * d * d / 3.0,   // 小信号 ≈ d
                } as f32;
                let wet = shaped * self.output;
                xin * (1.0 - mix) + wet * mix
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 软饱和：奇函数、单调、有界（|y| < 1），小信号近似线性。
    #[test]
    fn soft_saturation_is_odd_bounded_monotonic() {
        let fx = Distortion { kind: DistortionKind::Soft, drive: 1.0, output: 1.0, mix: 1.0 };
        let out = fx.process(&[-3.0, -0.001, 0.0, 0.001, 3.0], 8_000);
        assert!((out[2]).abs() < 1e-9, "奇函数过零");
        assert!((out[0] + out[4]).abs() < 1e-6, "奇对称");
        assert!(out.iter().all(|&y| y.abs() < 1.0), "有界");
        assert!((out[3] / 0.001 - 1.0).abs() < 0.05, "小信号近似线性");
    }

    /// 硬削：超过阈值钳到 ±1。
    #[test]
    fn hard_clip_limits_to_unit() {
        let fx = Distortion { kind: DistortionKind::Hard, drive: 2.0, output: 1.0, mix: 1.0 };
        let out = fx.process(&[-0.9, -0.1, 0.1, 0.9], 8_000);
        assert_eq!(out[0], -1.0);
        assert_eq!(out[3], 1.0);
        assert!((out[1] + 0.2).abs() < 1e-6 && (out[2] - 0.2).abs() < 1e-6);
    }

    /// 干湿比：mix=0 原样直通；mix=1 全湿。
    #[test]
    fn mix_blends_dry_wet() {
        let dry = Distortion { kind: DistortionKind::Hard, drive: 4.0, output: 1.0, mix: 0.0 };
        assert_eq!(dry.process(&[0.5, -0.5], 8_000), vec![0.5, -0.5]);
        let half = Distortion { kind: DistortionKind::Hard, drive: 4.0, output: 1.0, mix: 0.5 };
        let out = half.process(&[0.5], 8_000);
        // 0.5·0.5 + 1.0·0.5 = 0.75。
        assert!((out[0] - 0.75).abs() < 1e-6);
    }

    /// 确定性复算逐位一致。
    #[test]
    fn distortion_is_bitwise_reproducible() {
        let fx = Distortion::default();
        let x: Vec<f32> = (0..300).map(|i| (i as f32 * 0.07).sin()).collect();
        let first = fx.process(&x, 44_100);
        for _ in 0..100 {
            assert_eq!(fx.process(&x, 44_100), first);
        }
    }
}
