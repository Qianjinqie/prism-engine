//! LFO（低频振荡器，B12 合成器升级）：调制音高/音量/滤波。
//!
//! LFO 输出 ∈ [−1, 1] 的周期信号（与主振荡器同款纯函数相位——索引的纯函数，
//! 无状态机、seek 等价）。支持多种波形与 `depth`/`offset` 映射到目标参数。

use crate::det;
use crate::synth::Waveform;

/// LFO：波形 + 速率（Hz）+ 深度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lfo {
    /// 波形（支持 Sine / Square / Saw / Triangle）。
    pub waveform: Waveform,
    /// 速率（Hz）。
    pub rate_hz: f32,
    /// 相位偏移 ∈ [0, 1)。
    pub phase_offset: f32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self { waveform: Waveform::Sine, rate_hz: 5.0, phase_offset: 0.0 }
    }
}

impl Lfo {
    /// 在第 `index` 样本处（采样率 `sr`）的 LFO 值 ∈ [−1, 1]（纯函数）。
    pub fn value(&self, index: usize, sample_rate: u32) -> f32 {
        let p = (self.rate_hz as f64 * index as f64 / sample_rate as f64
            + self.phase_offset as f64)
            .rem_euclid(1.0);
        match self.waveform {
            Waveform::Sine => det::sin_f64(std::f64::consts::TAU * p) as f32,
            Waveform::Square => if p < 0.5 { 1.0 } else { -1.0 },
            Waveform::Saw => (2.0 * p - 1.0) as f32,
            Waveform::Triangle => (4.0 * (p - 0.5).abs() - 1.0) as f32,
        }
    }

    /// 调制后目标值：`base + depth·lfo`（`lfo` 已完成 offset 映射）。
    pub fn modulate(&self, base: f32, depth: f32, index: usize, sample_rate: u32) -> f32 {
        base + depth * self.value(index, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 正弦 LFO：1Hz @ 1000 SR，¼ 周期处峰值。
    #[test]
    fn sine_lfo_waveform_shape() {
        let lfo = Lfo { waveform: Waveform::Sine, rate_hz: 1.0, phase_offset: 0.0 };
        assert!(lfo.value(0, 1_000).abs() < 1e-6, "起点 0");
        assert!((lfo.value(250, 1_000) - 1.0).abs() < 1e-5, "¼ 周期峰值");
        assert!((lfo.value(500, 1_000)).abs() < 1e-5);
        assert!((lfo.value(750, 1_000) + 1.0).abs() < 1e-5);
    }

    /// 方波 LFO 只取 ±1；相位偏移平移波形。
    #[test]
    fn square_and_phase_offset() {
        let sq = Lfo { waveform: Waveform::Square, rate_hz: 4.0, phase_offset: 0.0 };
        assert_eq!(sq.value(0, 1_000), 1.0);
        assert_eq!(sq.value(200, 1_000), -1.0);
        let shifted = Lfo { waveform: Waveform::Square, rate_hz: 4.0, phase_offset: 0.5 };
        assert_eq!(shifted.value(0, 1_000), -1.0, "半圈偏移翻转");
    }

    /// modulate：深度 0 时返回基准值；深度非零按 LFO 调制。
    #[test]
    fn modulate_applies_depth() {
        let lfo = Lfo { waveform: Waveform::Sine, rate_hz: 1.0, phase_offset: 0.0 };
        assert_eq!(lfo.modulate(440.0, 0.0, 250, 1_000), 440.0);
        let m = lfo.modulate(440.0, 10.0, 250, 1_000);
        assert!((m - 450.0).abs() < 1e-4, "¼ 周期应 +depth");
    }

    /// 确定性复算。
    #[test]
    fn lfo_deterministic() {
        let lfo = Lfo::default();
        let a: Vec<f32> = (0..500).map(|i| lfo.value(i, 44_100)).collect();
        let b: Vec<f32> = (0..500).map(|i| lfo.value(i, 44_100)).collect();
        assert_eq!(a, b);
    }
}
