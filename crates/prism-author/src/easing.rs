//! 缓动函数（本 crate **自研**，不依赖 `prism-anim`，避免与其他工程师对
//! 该 crate 的改动耦合）。
//!
//! 全部函数为进度 `u ∈ [0, 1]` 上的**纯函数**，命名对齐 CSS / AE /
//! Remotion 行业惯例。除 `Elastic*` 族使用标准库 `f32::sin` / `f32::exp2`
//! 外，其余仅用 `+ - × ÷` 与整数幂连乘——标准库超越函数是输入的纯函数，
//! 同进程内逐次求值位模式一致（"黄金"确定性测试据此成立）。
//!
//! # 确定性
//! 无内部可变状态、无随机源；给定 `u` 必得同一结果。

/// 缓动种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EasingKind {
    /// 线性（默认）。
    #[default]
    Linear,
    /// 二次缓入。
    QuadIn,
    /// 二次缓出。
    QuadOut,
    /// 三次缓入。
    CubicIn,
    /// 三次缓出。
    CubicOut,
    /// 三次缓入缓出。
    CubicInOut,
    /// 弹性缓入。
    ElasticIn,
    /// 弹性缓出。
    ElasticOut,
    /// 弹性缓入缓出。
    ElasticInOut,
}

impl EasingKind {
    /// 求值：先把进度 `u` 钳制到 `[0, 1]`，再按种类分派到对应纯函数。
    #[inline]
    pub fn apply(self, u: f32) -> f32 {
        let u = if u < 0.0 {
            0.0
        } else if u > 1.0 {
            1.0
        } else {
            u
        };
        match self {
            Self::Linear => u,
            Self::QuadIn => u * u,
            Self::QuadOut => {
                let x = 1.0 - u;
                1.0 - x * x
            }
            Self::CubicIn => u * u * u,
            Self::CubicOut => {
                let x = 1.0 - u;
                1.0 - x * x * x
            }
            Self::CubicInOut => {
                if u < 0.5 {
                    4.0 * u * u * u
                } else {
                    let x = -2.0 * u + 2.0;
                    1.0 - x * x * x / 2.0
                }
            }
            Self::ElasticIn => elastic_in(u),
            Self::ElasticOut => elastic_out(u),
            Self::ElasticInOut => elastic_in_out(u),
        }
    }
}

/// 2π 常量（避免重复引用 `std::f32::consts`）。
const TAU: f32 = std::f32::consts::TAU;

/// 弹性缓出（CSS easeOutElastic，c4 = 2π/3）。
#[inline]
fn elastic_out(u: f32) -> f32 {
    const C4: f32 = TAU / 3.0;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else {
        f32::exp2(-10.0 * u) * ((u * 10.0 - 0.75) * C4).sin() + 1.0
    }
}

/// 弹性缓入（CSS easeInElastic，c4 = 2π/3）。
#[inline]
fn elastic_in(u: f32) -> f32 {
    const C4: f32 = TAU / 3.0;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else {
        -(f32::exp2(10.0 * u - 10.0) * ((u * 10.0 - 10.75) * C4).sin())
    }
}

/// 弹性缓入缓出（CSS easeInOutElastic，c5 = 2π/4.5）。
#[inline]
fn elastic_in_out(u: f32) -> f32 {
    const C5: f32 = TAU / 4.5;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else if u < 0.5 {
        (f32::exp2(20.0 * u - 10.0) * ((20.0 * u - 11.125) * C5).sin()) * 0.5
    } else {
        (f32::exp2(-20.0 * u + 10.0) * ((20.0 * u - 11.125) * C5).sin()) * 0.5 + 1.0
    }
}

// ---------------------------------------------------------------------------
// 测试：端点恒等 / 手写期望值 / 单调性 / 越界钳制
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kinds_hit_endpoints_exactly() {
        const ALL: [EasingKind; 9] = [
            EasingKind::Linear,
            EasingKind::QuadIn,
            EasingKind::QuadOut,
            EasingKind::CubicIn,
            EasingKind::CubicOut,
            EasingKind::CubicInOut,
            EasingKind::ElasticIn,
            EasingKind::ElasticOut,
            EasingKind::ElasticInOut,
        ];
        for kind in ALL {
            assert_eq!(kind.apply(0.0), 0.0, "{kind:?}(0) 应为 0");
            assert_eq!(kind.apply(1.0), 1.0, "{kind:?}(1) 应为 1");
        }
    }

    #[test]
    fn known_values_match_css_formulas() {
        assert_eq!(EasingKind::Linear.apply(0.3), 0.3);
        assert_eq!(EasingKind::QuadIn.apply(0.5), 0.25);
        assert_eq!(EasingKind::QuadOut.apply(0.5), 0.75);
        assert_eq!(EasingKind::CubicOut.apply(0.5), 0.875);
        assert_eq!(EasingKind::CubicInOut.apply(0.5), 0.5);
        // ElasticOut(0.5) 手写期望值（理想解）。
        let expected = f32::exp2(-5.0) * ((5.0 - 0.75) * (TAU / 3.0)).sin() + 1.0;
        assert!((EasingKind::ElasticOut.apply(0.5) - expected).abs() < 1e-5);
    }

    #[test]
    fn monotone_families_are_non_decreasing() {
        // 仅校验"无过冲"的 In/Out/InOut 族：Elastic 族按定义会过冲/回弹，
        // 不满足单调不减（这是 CSS 弹性缓动的预期行为），故排除。
        const MONOTONE: [EasingKind; 6] = [
            EasingKind::Linear,
            EasingKind::QuadIn,
            EasingKind::QuadOut,
            EasingKind::CubicIn,
            EasingKind::CubicOut,
            EasingKind::CubicInOut,
        ];
        for kind in MONOTONE {
            let mut prev = kind.apply(0.0);
            for i in 1..=100 {
                let v = kind.apply(i as f32 / 100.0);
                assert!(v >= prev - 1e-6, "{kind:?} 在 u={} 单调性破坏: {v} < {prev}", i as f32 / 100.0);
                prev = v;
            }
        }
    }

    #[test]
    fn out_of_range_progress_is_clamped() {
        for kind in [EasingKind::CubicOut, EasingKind::ElasticOut, EasingKind::ElasticInOut] {
            assert_eq!(kind.apply(-2.0), kind.apply(0.0), "{kind:?} 负进度应钳 0");
            assert_eq!(kind.apply(7.5), kind.apply(1.0), "{kind:?} 超进度应钳 1");
        }
    }

    #[test]
    fn elastic_is_finite_everywhere() {
        for i in 0..=100 {
            let u = i as f32 / 100.0;
            assert!(EasingKind::ElasticIn.apply(u).is_finite());
            assert!(EasingKind::ElasticOut.apply(u).is_finite());
            assert!(EasingKind::ElasticInOut.apply(u).is_finite());
        }
    }
}
