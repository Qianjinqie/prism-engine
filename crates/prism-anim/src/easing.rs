//! 缓动函数表（工作流 A2；S9 分段多项式的插值核；白皮书 §25.1 再生核化身）。
//!
//! 全部函数是 `u ∈ [0, 1]` 上的**纯函数**（进度 → 缓动值），命名与 CSS /
//! AE / Remotion 行业惯例逐字一致（宪章 C5：通用概念用通用名；对照表 #14
//! "Easing / 缓动（CSS、AE、Remotion）→ 同名函数"）。
//!
//! # 确定性
//! 只用 `+ - × ÷`、整数幂连乘（禁 `pow`，架构 §4.3 格锚表）、IEEE 精确的
//! `sqrt`，以及本 crate [`det`](crate::det) 固定系数多项式的 sin/cos/exp2
//! ——无 libm 超越函数，跨平台逐位一致（D-29）。

/// 缓动函数种类（CSS easing-functions 全集 + Linear）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingKind {
    /// 线性。
    Linear,
    /// 二次缓入。
    QuadIn,
    /// 二次缓出。
    QuadOut,
    /// 二次缓入缓出。
    QuadInOut,
    /// 三次缓入。
    CubicIn,
    /// 三次缓出。
    CubicOut,
    /// 三次缓入缓出。
    CubicInOut,
    /// 四次缓入。
    QuartIn,
    /// 四次缓出。
    QuartOut,
    /// 四次缓入缓出。
    QuartInOut,
    /// 五次缓入。
    QuintIn,
    /// 五次缓出。
    QuintOut,
    /// 五次缓入缓出。
    QuintInOut,
    /// 正弦缓入。
    SineIn,
    /// 正弦缓出。
    SineOut,
    /// 正弦缓入缓出。
    SineInOut,
    /// 指数缓入。
    ExpoIn,
    /// 指数缓出。
    ExpoOut,
    /// 指数缓入缓出。
    ExpoInOut,
    /// 圆弧缓入。
    CircIn,
    /// 圆弧缓出。
    CircOut,
    /// 圆弧缓入缓出。
    CircInOut,
    /// 回退缓入（过冲后返回）。
    BackIn,
    /// 回退缓出（过冲）。
    BackOut,
    /// 回退缓入缓出。
    BackInOut,
    /// 弹性缓入（振荡趋近）。
    ElasticIn,
    /// 弹性缓出（振荡趋近）。
    ElasticOut,
    /// 弹性缓入缓出。
    ElasticInOut,
    /// 弹跳缓入。
    BounceIn,
    /// 弹跳缓出。
    BounceOut,
    /// 弹跳缓入缓出。
    BounceInOut,
}

impl EasingKind {
    /// 求值：先把进度 u 钳到 `[0, 1]`，再按种类分派到对应纯函数。
    pub fn apply(self, u: f32) -> f32 {
        let u = if u < 0.0 {
            0.0
        } else if u > 1.0 {
            1.0
        } else {
            u
        };
        match self {
            Self::Linear => linear(u),
            Self::QuadIn => quad_in(u),
            Self::QuadOut => quad_out(u),
            Self::QuadInOut => quad_in_out(u),
            Self::CubicIn => cubic_in(u),
            Self::CubicOut => cubic_out(u),
            Self::CubicInOut => cubic_in_out(u),
            Self::QuartIn => quart_in(u),
            Self::QuartOut => quart_out(u),
            Self::QuartInOut => quart_in_out(u),
            Self::QuintIn => quint_in(u),
            Self::QuintOut => quint_out(u),
            Self::QuintInOut => quint_in_out(u),
            Self::SineIn => sine_in(u),
            Self::SineOut => sine_out(u),
            Self::SineInOut => sine_in_out(u),
            Self::ExpoIn => expo_in(u),
            Self::ExpoOut => expo_out(u),
            Self::ExpoInOut => expo_in_out(u),
            Self::CircIn => circ_in(u),
            Self::CircOut => circ_out(u),
            Self::CircInOut => circ_in_out(u),
            Self::BackIn => back_in(u),
            Self::BackOut => back_out(u),
            Self::BackInOut => back_in_out(u),
            Self::ElasticIn => elastic_in(u),
            Self::ElasticOut => elastic_out(u),
            Self::ElasticInOut => elastic_in_out(u),
            Self::BounceIn => bounce_in(u),
            Self::BounceOut => bounce_out(u),
            Self::BounceInOut => bounce_in_out(u),
        }
    }
}

// ---------------------------------------------------------------------------
// 纯函数本体（u ∈ [0, 1]；整数幂一律连乘）
// ---------------------------------------------------------------------------

/// 线性：f(u) = u。
#[inline]
pub fn linear(u: f32) -> f32 {
    u
}

/// 二次缓入：f(u) = u²。
#[inline]
pub fn quad_in(u: f32) -> f32 {
    u * u
}

/// 二次缓出：f(u) = 1 − (1−u)²。
#[inline]
pub fn quad_out(u: f32) -> f32 {
    let x = 1.0 - u;
    1.0 - x * x
}

/// 二次缓入缓出。
#[inline]
pub fn quad_in_out(u: f32) -> f32 {
    if u < 0.5 {
        2.0 * u * u
    } else {
        let x = -2.0 * u + 2.0;
        1.0 - x * x / 2.0
    }
}

/// 三次缓入：f(u) = u³。
#[inline]
pub fn cubic_in(u: f32) -> f32 {
    u * u * u
}

/// 三次缓出：f(u) = 1 − (1−u)³。
#[inline]
pub fn cubic_out(u: f32) -> f32 {
    let x = 1.0 - u;
    1.0 - x * x * x
}

/// 三次缓入缓出。
#[inline]
pub fn cubic_in_out(u: f32) -> f32 {
    if u < 0.5 {
        4.0 * u * u * u
    } else {
        let x = -2.0 * u + 2.0;
        1.0 - x * x * x / 2.0
    }
}

/// 四次缓入：f(u) = u⁴。
#[inline]
pub fn quart_in(u: f32) -> f32 {
    u * u * u * u
}

/// 四次缓出：f(u) = 1 − (1−u)⁴。
#[inline]
pub fn quart_out(u: f32) -> f32 {
    let x = 1.0 - u;
    1.0 - x * x * x * x
}

/// 四次缓入缓出。
#[inline]
pub fn quart_in_out(u: f32) -> f32 {
    if u < 0.5 {
        8.0 * u * u * u * u
    } else {
        let x = -2.0 * u + 2.0;
        1.0 - x * x * x * x / 2.0
    }
}

/// 五次缓入：f(u) = u⁵。
#[inline]
pub fn quint_in(u: f32) -> f32 {
    u * u * u * u * u
}

/// 五次缓出：f(u) = 1 − (1−u)⁵。
#[inline]
pub fn quint_out(u: f32) -> f32 {
    let x = 1.0 - u;
    1.0 - x * x * x * x * x
}

/// 五次缓入缓出。
#[inline]
pub fn quint_in_out(u: f32) -> f32 {
    if u < 0.5 {
        16.0 * u * u * u * u * u
    } else {
        let x = -2.0 * u + 2.0;
        1.0 - x * x * x * x * x / 2.0
    }
}

/// 正弦缓入：f(u) = 1 − cos(u·π/2)。
#[inline]
pub fn sine_in(u: f32) -> f32 {
    1.0 - crate::det::cos_f32(u * std::f32::consts::FRAC_PI_2)
}

/// 正弦缓出：f(u) = sin(u·π/2)。
#[inline]
pub fn sine_out(u: f32) -> f32 {
    crate::det::sin_f32(u * std::f32::consts::FRAC_PI_2)
}

/// 正弦缓入缓出：f(u) = −(cos(π·u) − 1) / 2。
#[inline]
pub fn sine_in_out(u: f32) -> f32 {
    -(crate::det::cos_f32(u * std::f32::consts::PI) - 1.0) / 2.0
}

/// 指数缓入：f(u) = 2^(10u − 10)；u = 0 处精确取 0。
#[inline]
pub fn expo_in(u: f32) -> f32 {
    if u == 0.0 {
        0.0
    } else {
        crate::det::exp2_f32(10.0 * u - 10.0)
    }
}

/// 指数缓出：f(u) = 1 − 2^(−10u)；u = 1 处精确取 1。
#[inline]
pub fn expo_out(u: f32) -> f32 {
    if u == 1.0 {
        1.0
    } else {
        1.0 - crate::det::exp2_f32(-10.0 * u)
    }
}

/// 指数缓入缓出。
#[inline]
pub fn expo_in_out(u: f32) -> f32 {
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else if u < 0.5 {
        crate::det::exp2_f32(20.0 * u - 10.0) / 2.0
    } else {
        (2.0 - crate::det::exp2_f32(-20.0 * u + 10.0)) / 2.0
    }
}

/// 圆弧缓入：f(u) = 1 − √(1 − u²)。
#[inline]
pub fn circ_in(u: f32) -> f32 {
    1.0 - (1.0 - u * u).sqrt()
}

/// 圆弧缓出：f(u) = √(1 − (u−1)²)。
#[inline]
pub fn circ_out(u: f32) -> f32 {
    let x = u - 1.0;
    (1.0 - x * x).sqrt()
}

/// 圆弧缓入缓出（CSS easeInOutCirc）。
#[inline]
pub fn circ_in_out(u: f32) -> f32 {
    if u < 0.5 {
        let x = 2.0 * u;
        (1.0 - (1.0 - x * x).sqrt()) / 2.0
    } else {
        let x = -2.0 * u + 2.0;
        ((1.0 - x * x).sqrt() + 1.0) / 2.0
    }
}

/// 回退缓入（CSS easeInBack，c1 = 1.70158）。
#[inline]
pub fn back_in(u: f32) -> f32 {
    const C1: f32 = 1.701_58;
    let c3 = C1 + 1.0;
    c3 * u * u * u - C1 * u * u
}

/// 回退缓出（CSS easeOutBack）。
#[inline]
pub fn back_out(u: f32) -> f32 {
    const C1: f32 = 1.701_58;
    let c3 = C1 + 1.0;
    let x = u - 1.0;
    1.0 + c3 * x * x * x + C1 * x * x
}

/// 回退缓入缓出（CSS easeInOutBack，c2 = c1·1.525）。
#[inline]
pub fn back_in_out(u: f32) -> f32 {
    const C1: f32 = 1.701_58;
    let c2 = C1 * 1.525;
    if u < 0.5 {
        let x = 2.0 * u;
        (x * x * ((c2 + 1.0) * x - c2)) / 2.0
    } else {
        let x = 2.0 * u - 2.0;
        (x * x * ((c2 + 1.0) * x + c2) + 2.0) / 2.0
    }
}

/// 弹性缓入（CSS easeInElastic，c4 = 2π/3）。
#[inline]
pub fn elastic_in(u: f32) -> f32 {
    const C4: f32 = std::f32::consts::TAU / 3.0;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else {
        -crate::det::exp2_f32(10.0 * u - 10.0) * crate::det::sin_f32((u * 10.0 - 10.75) * C4)
    }
}

/// 弹性缓出（CSS easeOutElastic，c4 = 2π/3）。
#[inline]
pub fn elastic_out(u: f32) -> f32 {
    const C4: f32 = std::f32::consts::TAU / 3.0;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else {
        crate::det::exp2_f32(-10.0 * u) * crate::det::sin_f32((u * 10.0 - 0.75) * C4) + 1.0
    }
}

/// 弹性缓入缓出（CSS easeInOutElastic，c5 = 2π/4.5）。
#[inline]
pub fn elastic_in_out(u: f32) -> f32 {
    const C5: f32 = std::f32::consts::TAU / 4.5;
    if u == 0.0 {
        0.0
    } else if u == 1.0 {
        1.0
    } else if u < 0.5 {
        crate::det::exp2_f32(20.0 * u - 10.0) * crate::det::sin_f32((20.0 * u - 11.125) * C5) / 2.0
    } else {
        crate::det::exp2_f32(-20.0 * u + 10.0) * crate::det::sin_f32((20.0 * u - 11.125) * C5) / 2.0
            + 1.0
    }
}

/// 弹跳缓出（CSS easeOutBounce，分段二次，n1 = 7.5625、d1 = 2.75）。
#[inline]
pub fn bounce_out(u: f32) -> f32 {
    const N1: f32 = 7.562_5;
    const D1: f32 = 2.75;
    if u < 1.0 / D1 {
        N1 * u * u
    } else if u < 2.0 / D1 {
        let x = u - 1.5 / D1;
        N1 * x * x + 0.75
    } else if u < 2.5 / D1 {
        let x = u - 2.25 / D1;
        N1 * x * x + 0.937_5
    } else {
        let x = u - 2.625 / D1;
        N1 * x * x + 0.984_375
    }
}

/// 弹跳缓入：f(u) = 1 − bounce_out(1 − u)。
#[inline]
pub fn bounce_in(u: f32) -> f32 {
    1.0 - bounce_out(1.0 - u)
}

/// 弹跳缓入缓出。
#[inline]
pub fn bounce_in_out(u: f32) -> f32 {
    if u < 0.5 {
        (1.0 - bounce_out(1.0 - 2.0 * u)) / 2.0
    } else {
        (1.0 + bounce_out(2.0 * u - 1.0)) / 2.0
    }
}

// ---------------------------------------------------------------------------
// 测试（端点恒等 / 手写期望值 / 单调性 / 越界钳制）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 全部 31 种缓动的端点恒等：f(0) = 0、f(1) = 1（精确）。
    #[test]
    fn all_kinds_hit_endpoints_exactly() {
        const ALL: [EasingKind; 31] = [
            EasingKind::Linear,
            EasingKind::QuadIn,
            EasingKind::QuadOut,
            EasingKind::QuadInOut,
            EasingKind::CubicIn,
            EasingKind::CubicOut,
            EasingKind::CubicInOut,
            EasingKind::QuartIn,
            EasingKind::QuartOut,
            EasingKind::QuartInOut,
            EasingKind::QuintIn,
            EasingKind::QuintOut,
            EasingKind::QuintInOut,
            EasingKind::SineIn,
            EasingKind::SineOut,
            EasingKind::SineInOut,
            EasingKind::ExpoIn,
            EasingKind::ExpoOut,
            EasingKind::ExpoInOut,
            EasingKind::CircIn,
            EasingKind::CircOut,
            EasingKind::CircInOut,
            EasingKind::BackIn,
            EasingKind::BackOut,
            EasingKind::BackInOut,
            EasingKind::ElasticIn,
            EasingKind::ElasticOut,
            EasingKind::ElasticInOut,
            EasingKind::BounceIn,
            EasingKind::BounceOut,
            EasingKind::BounceInOut,
        ];
        for kind in ALL {
            assert_eq!(kind.apply(0.0), 0.0, "{kind:?}(0) 应为 0");
            assert_eq!(kind.apply(1.0), 1.0, "{kind:?}(1) 应为 1");
        }
    }

    /// 手写期望值（与 CSS easing-functions 规范公式对拍）。
    #[test]
    fn known_values_match_css_formulas() {
        assert_eq!(EasingKind::Linear.apply(0.3), 0.3);
        assert_eq!(EasingKind::QuadIn.apply(0.5), 0.25);
        assert_eq!(EasingKind::CubicOut.apply(0.5), 0.875);
        assert_eq!(EasingKind::QuartIn.apply(0.5), 0.0625);
        assert_eq!(EasingKind::QuintInOut.apply(0.25), 0.015625);
        // SineOut(0.5) = sin(π/4) = √2/2。
        assert!((EasingKind::SineOut.apply(0.5) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        // ExpoOut(0.5) = 1 − 2^(−5) = 0.96875。
        assert!((EasingKind::ExpoOut.apply(0.5) - 0.96875).abs() < 1e-6);
        // CircOut(0.5) = √(1 − 0.25) = √3/2。
        assert!((EasingKind::CircOut.apply(0.5) - 0.866_025_4).abs() < 1e-6);
        // BackOut(0.5) = 1 + c3·(−0.5)³ + c1·(−0.5)² = 1.0876975。
        assert!((EasingKind::BackOut.apply(0.5) - 1.087_697_5).abs() < 1e-6);
        // BounceOut(0.5) = 7.5625·(1/22)² + 0.75 = 0.765625（精确）。
        assert_eq!(EasingKind::BounceOut.apply(0.5), 0.765_625);
        // ElasticOut(0.5) = 2^(−5)·sin((5−0.75)·2π/3) + 1 = 1.015625（理想值）。
        assert!((EasingKind::ElasticOut.apply(0.5) - 1.015_625).abs() < 1e-5);
    }

    /// 单调性：缓入/缓出族（无过冲的 In/Out 全部）在 [0,1] 上不减。
    #[test]
    fn monotone_families_are_non_decreasing() {
        const MONOTONE: [EasingKind; 13] = [
            EasingKind::Linear,
            EasingKind::QuadIn,
            EasingKind::QuadOut,
            EasingKind::CubicIn,
            EasingKind::CubicOut,
            EasingKind::QuartIn,
            EasingKind::QuartOut,
            EasingKind::QuintIn,
            EasingKind::QuintOut,
            EasingKind::SineIn,
            EasingKind::SineOut,
            EasingKind::ExpoIn,
            EasingKind::ExpoOut,
        ];
        for kind in MONOTONE {
            let mut prev = kind.apply(0.0);
            for i in 1..=100 {
                let v = kind.apply(i as f32 / 100.0);
                assert!(v >= prev, "{kind:?} 在 u={} 处单调性破坏: {v} < {prev}", i as f32 / 100.0);
                prev = v;
            }
        }
    }

    /// 越界钳制：u < 0 取 0 支、u > 1 取 1 支（端点恒等的推论）。
    #[test]
    fn out_of_range_progress_is_clamped() {
        for kind in [EasingKind::CubicOut, EasingKind::ElasticOut, EasingKind::BounceInOut] {
            assert_eq!(kind.apply(-2.0), kind.apply(0.0), "{kind:?} 负进度应钳 0");
            assert_eq!(kind.apply(7.5), kind.apply(1.0), "{kind:?} 超进度应钳 1");
        }
    }
}
