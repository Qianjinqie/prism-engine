//! 延时线效果器（B12）：带反馈与干湿比的整数延迟。
//!
//! 延迟长度 `delay_secs` 量化为整数样本 `d = round(delay_secs·SR)`（格点对齐
//! ——白皮书 §23 定点口径，避免浮点索引漂移）。反馈拓扑：
//!
//! ```text
//! y[i] = x[i]·dry + tap[i]
//! tap[i] = w·y[i−d]                     （w = feedback ∈ [0, ~0.99)）
//! ```
//!
//! 缓冲在 `process` 内一次性构造并按索引递增单遍填充（固定序、无跨调用状态），
//! 故 seek 到第 N 样本等价于顺序推进（重算等价）。

use super::MonoFx;

/// 延时参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delay {
    /// 延迟时长（秒；≥0）。
    pub delay_secs: f32,
    /// 反馈系数 ∈ [0, 1)（0 = 单次回声）。
    pub feedback: f32,
    /// 干信号混合 ∈ [0, 1]。
    pub dry: f32,
    /// 湿信号混合 ∈ [0, 1]。
    pub wet: f32,
}

impl Default for Delay {
    fn default() -> Self {
        Self { delay_secs: 0.25, feedback: 0.4, dry: 1.0, wet: 0.5 }
    }
}

impl MonoFx for Delay {
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let d = (self.delay_secs.max(0.0) * sample_rate as f32).round() as usize;
        if d == 0 {
            // 零延迟：退化为干湿直接混合。
            return input.iter().map(|&x| x * (self.dry + self.wet)).collect();
        }
        let fb = self.feedback.clamp(0.0, 0.999);
        // 单遍填充：read[i] 为延迟线内容（净回声），先合成湿信号再写回延迟线。
        // 第一回声（i<d）= 0，其余 = 前一输出湿分量的反馈。
        let mut out = vec![0.0_f32; input.len()];
        let mut line = vec![0.0_f32; d];
        for i in 0..input.len() {
            let delayed = line[i % d]; // 延迟 d 个样本的「湿」内容
            let wet = delayed; // wet 分量为纯延迟线内容
            line[i % d] = input[i] + delayed * fb; // 延迟线 = 输入 + 反馈
            out[i] = input[i] * self.dry + wet * self.wet;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 冲激响应：第 0 样本 = dry，第 d 样本 = wet·fb⁰，第 2d 样本 = wet·fb¹…（等比回声）。
    #[test]
    fn impulse_response_is_geometric_echoes() {
        let sr = 1_000_u32;
        let d = 10_usize;
        let fx = Delay { delay_secs: 10.0 / sr as f32, feedback: 0.5, dry: 1.0, wet: 1.0 };
        let mut input = vec![0.0_f32; 100];
        input[0] = 1.0;
        let out = fx.process(&input, sr);
        assert!((out[0] - 1.0).abs() < 1e-6, "干直通");
        assert!((out[d] - 1.0).abs() < 1e-6, "第一回声");
        assert!((out[2 * d] - 0.5).abs() < 1e-6, "第二回声 ×fb");
        assert!((out[3 * d] - 0.25).abs() < 1e-6, "第三回声 ×fb²");
        assert!(out[1].abs() < 1e-6 && out[d - 1].abs() < 1e-6, "回声之间应静默");
    }

    /// 延时位置 = 舍入后的整数样本（格点对齐）；feedback=0 仅一次回声。
    #[test]
    fn delay_position_is_quantized_and_feedback_zero_single_tap() {
        let sr = 100_u32;
        let fx = Delay { delay_secs: 0.074, feedback: 0.0, dry: 0.0, wet: 1.0 };
        // 0.074·100 = 7.4 → round = 7。
        let mut input = vec![0.0_f32; 30];
        input[0] = 1.0;
        let out = fx.process(&input, sr);
        assert!((out[7] - 1.0).abs() < 1e-6, "7 号样本应为回声（7.4 舍入）");
        assert!(out[8].abs() < 1e-6, "无后续回声（feedback=0）");
    }

    /// 确定性：同输入重复调用逐位一致。
    #[test]
    fn delay_is_bitwise_reproducible() {
        let fx = Delay::default();
        let input: Vec<f32> = (0..500).map(|i| (i as f32 * 0.03).sin()).collect();
        let first = fx.process(&input, 44_100);
        for _ in 0..100 {
            assert_eq!(fx.process(&input, 44_100), first);
        }
    }
}
