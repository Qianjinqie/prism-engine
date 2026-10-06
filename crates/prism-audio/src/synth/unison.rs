//! 多振荡 unison（B12 合成器升级）：多个轻微失调的振荡器叠加，增强厚度。
//!
//! `voices` 个振荡器，围绕中心频率对称失调（detune）：第 k 个振荡器频率比
//! `2^(offsetₖ·detune_cents/1200)`，`offsetₖ` 在 [−1, 1] 均匀分布（含端点）。
//! 各振荡器相位错开（spread），避免同相叠加。输出做 `1/√voices` 归一（保持感知
//! 响度，能量非相干叠加近似一致）。
//!
//! 相位仍是**采样索引的纯函数**（沿用 [`crate::synth::sample_osc`] 的范式）——
//! 无状态机、seek 等价。失调比走 [`crate::sfz::semitone_ratio`]（禁 pow，D-29）。

use crate::sfz::semitone_ratio;
use crate::synth::{sample_osc, Waveform};

/// 多振荡 unison 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unison {
    /// 波形。
    pub waveform: Waveform,
    /// 振荡器数量（≥1）。
    pub voices: usize,
    /// 最大失调（cents；±此值对称分布）。
    pub detune_cents: f32,
    /// 相位错开（0 = 全部同相；1 = 均匀铺满一圈）。
    pub phase_spread: f32,
}

impl Default for Unison {
    fn default() -> Self {
        Self { waveform: Waveform::Saw, voices: 3, detune_cents: 12.0, phase_spread: 0.5 }
    }
}

impl Unison {
    /// 第 `k` 个振荡器的频率比（`k ∈ [0, voices)`）。
    fn voice_ratio(&self, k: usize) -> f32 {
        if self.voices <= 1 {
            return 1.0;
        }
        // offset ∈ [−1, 1]：voices=1 → 0；voices=n → k∈[0,n-1] 映射到 [−1,1]。
        let offset = 2.0 * k as f32 / (self.voices - 1) as f32 - 1.0;
        semitone_ratio(offset * self.detune_cents / 100.0)
    }

    /// 第 `k` 个振荡器的相位偏移（∈ [0, 1)）。
    fn voice_phase(&self, k: usize) -> f32 {
        if self.voices <= 1 {
            return 0.0;
        }
        let base = k as f32 / self.voices as f32;
        base * self.phase_spread.clamp(0.0, 1.0)
    }

    /// 第 `index` 个样本的 unison 叠加值（纯函数）。
    pub fn sample(&self, freq: f32, index: usize, sample_rate: u32) -> f32 {
        let n = self.voices.max(1);
        let mut acc = 0.0_f32;
        for k in 0..n {
            let fk = freq * self.voice_ratio(k);
            // 相位偏移 = 起始相位 offset（按索引折算为样本延迟）。
            let phase = self.voice_phase(k);
            let delayed_index = index as f64 + phase as f64 * sample_rate as f64 / fk.max(1e-3) as f64;
            acc += sample_osc(self.waveform, fk, delayed_index as usize, sample_rate);
        }
        acc / (n as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 单振荡器 unison ≡ 裸振荡器。
    #[test]
    fn single_voice_equals_plain_osc() {
        let u = Unison { waveform: Waveform::Sine, voices: 1, detune_cents: 0.0, phase_spread: 0.0 };
        for i in [0_usize, 1, 50, 999] {
            let a = u.sample(440.0, i, 44_100);
            let b = sample_osc(Waveform::Sine, 440.0, i, 44_100);
            assert!((a - b).abs() < 1e-5, "单声部应等于裸振荡器");
        }
    }

    /// 多声部能量上界：|输出| ≤ voices/√voices = √voices（相干最坏情况）。
    #[test]
    fn unison_bounded() {
        let u = Unison { waveform: Waveform::Saw, voices: 5, detune_cents: 30.0, phase_spread: 0.9 };
        for i in 0..2000 {
            let v = u.sample(220.0, i, 44_100);
            assert!(v.is_finite() && v.abs() <= 5.0_f32.sqrt() + 1e-3, "越界 {v}");
        }
    }

    /// 失调：voices 增多时频谱展宽（拍频更强 → 包络起伏更大）。
    #[test]
    fn detune_increases_amplitude_variation() {
        let env_variation = |voices: usize| -> f32 {
            let u = Unison { waveform: Waveform::Sine, voices, detune_cents: 40.0, phase_spread: 0.0 };
            let s: Vec<f32> = (0..8_000).map(|i| u.sample(440.0, i, 44_100)).collect();
            let max = s.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
            let min = s.iter().fold(f32::MAX, |m, &x| m.min(x.abs()));
            max - min
        };
        assert!(env_variation(5) > env_variation(1), "多声部失调应产生更强拍频起伏");
    }

    /// 确定性：同参数逐位一致。
    #[test]
    fn unison_deterministic() {
        let u = Unison::default();
        let a: Vec<f32> = (0..1000).map(|i| u.sample(330.0, i, 44_100)).collect();
        let b: Vec<f32> = (0..1000).map(|i| u.sample(330.0, i, 44_100)).collect();
        assert_eq!(a, b);
    }
}
