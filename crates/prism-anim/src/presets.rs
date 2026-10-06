//! 预置特解参数包（工作流 A2；D-23 L1 参数级：一个预设名 = 一组特解参数）。
//!
//! 命名对照行业惯例（宪章 C4/C5：行业已有概念照抄行业词）——`smooth/gentle/
//! snappy/wobbly/bouncy` 对齐 React Spring / framer-motion 的预设词汇，参数
//! 以 (ω, ζ) 直接给出，物理含义显式可调。
//!
//! 全部预设是常量数据（纯声明，零运行时成本，架构 §2.1 别名层同款哲学）。

use crate::spring::Spring;

/// 弹簧预设：预置特解参数包（ω：自然角频率 rad/s；ζ：阻尼比）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringPreset {
    /// 自然角频率 ω（rad/s）。
    pub omega: f32,
    /// 阻尼比 ζ。
    pub zeta: f32,
}

impl SpringPreset {
    /// 平滑临界弹簧（ζ = 1，无过冲，最快无振荡到位）。
    pub const SMOOTH: Self = Self { omega: 6.0, zeta: 1.0 };
    /// 柔和弹簧（轻微过冲，缓起缓收）。
    pub const GENTLE: Self = Self { omega: 8.0, zeta: 0.8 };
    /// 利落弹簧（高频、微过冲，快速到位）。
    pub const SNAPPY: Self = Self { omega: 14.0, zeta: 0.9 };
    /// 摆动弹簧（明显回弹）。
    pub const WOBBLY: Self = Self { omega: 10.0, zeta: 0.4 };
    /// 弹跳弹簧（强回弹，多周期振荡）。
    pub const BOUNCY: Self = Self { omega: 12.0, zeta: 0.25 };

    /// 以预设参数实例化弹簧：从 `x0` 到 `target`，初速度 0。
    pub fn spring(&self, x0: f32, target: f32) -> Spring {
        Spring::new(self.omega, self.zeta, x0, 0.0, target)
    }
}

// ---------------------------------------------------------------------------
// 测试（预设的物理行为健全性 + 收敛性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const PRESETS: [(&str, SpringPreset); 5] = [
        ("SMOOTH", SpringPreset::SMOOTH),
        ("GENTLE", SpringPreset::GENTLE),
        ("SNAPPY", SpringPreset::SNAPPY),
        ("WOBBLY", SpringPreset::WOBBLY),
        ("BOUNCY", SpringPreset::BOUNCY),
    ];

    /// 全部预设收敛到 target（ζ ≥ 0.25 的欠阻尼在 40s 内必然衰减到位）。
    #[test]
    fn all_presets_converge() {
        for (name, preset) in PRESETS {
            let spring = preset.spring(0.0, 1.0);
            let settled = spring.sample(40.0);
            assert!(
                (settled - 1.0).abs() < 1e-6,
                "{name} 未收敛：{settled}"
            );
        }
    }

    /// 过冲行为分级：SMOOTH 不过冲；SNAPPY 微过冲；WOBBLY/BOUNCY 明显回弹。
    #[test]
    fn overshoot_grading_matches_preset_intent() {
        let peak_of = |preset: SpringPreset| {
            (1..600)
                .map(|i| preset.spring(0.0, 1.0).sample(i as f32 * 0.05))
                .fold(0.0_f32, f32::max)
        };
        assert!(peak_of(SpringPreset::SMOOTH) <= 1.0, "SMOOTH 不应过冲");
        let snappy = peak_of(SpringPreset::SNAPPY);
        assert!(snappy > 1.0 && snappy < 1.1, "SNAPPY 微过冲，峰值 {snappy}");
        assert!(peak_of(SpringPreset::WOBBLY) > 1.15, "WOBBLY 应明显回弹");
        assert!(peak_of(SpringPreset::BOUNCY) > 1.3, "BOUNCY 应强回弹");
    }

    /// 预设 → 弹簧的初值语义：从 x0 出发，t=0 精确取 x0。
    #[test]
    fn preset_spring_starts_at_x0() {
        for (_, preset) in PRESETS {
            let spring = preset.spring(5.0, -5.0);
            assert_eq!(spring.sample(0.0).to_bits(), 5.0_f32.to_bits());
        }
    }
}
