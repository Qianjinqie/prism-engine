//! D-29 确定性三角件（自研区间逼近多项式 + 固定系数，**禁 libm**）。
//!
//! `prism-math` 全程零外部依赖、仅用标准库；`.cos()/.sin()/.tan()` 会落到
//! 平台 libm（跨平台逐位不一致，违反 C17/C18 确定性铁律），故此处提供自研
//! 替代件。算法与 `prism-rt::sampler` 的 det 三角件一致：
//!
//! 1. 范围归约 `x mod 2π`（`floor` 为精确舍入运算，非超越函数）；
//! 2. 象限对称归约到 `[0, π/4]`；
//! 3. `[0, π/4]` 上用固定系数 Maclaurin 级数（Horner 排布）。
//!
//! 只含 `± × ÷` 与精确 `floor`，同输入跨平台逐位一致。`[0, π/4]` 处截断
//! 误差 `sin ~2.5e-8`、`cos ~2.5e-8`，覆盖引擎全部相机/相位角使用场景。

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// `[0, π/4]` 上的 `sin`（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0
            + x2 * (1.0 / 120.0
                + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
}

/// `[0, π/4]` 上的 `cos`（固定系数 Maclaurin，Horner 排布）。
#[inline]
fn quarter_cos(x: f32) -> f32 {
    let x2 = x * x;
    1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
}

/// 任意角 `(cos x, sin x)`：`mod 2π` 归约（精确 floor）→ 象限对称 →
/// `[0, π/4]` 级数。跨平台逐位一致（只含 `± × ÷` 与精确 `floor`）。
pub fn cos_sin(x: f32) -> (f32, f32) {
    const TWO_PI: f32 = 2.0 * PI;
    // 归约到 [0, 2π)；引擎输入量级小（相机角/相位角），floor 精度充分。
    let mut r = x - TWO_PI * (x / TWO_PI).floor();
    // 象限号：q = floor(r / (π/2)) ∈ {0,1,2,3}（r ≥ 0 保证非负）。
    let q = (r * (2.0 / PI)) as i32 & 3;
    r -= q as f32 * FRAC_PI_2; // r ∈ [0, π/2]
    // 半区归约到 [0, π/4]。
    let (c, s) = if r > FRAC_PI_4 {
        let rr = FRAC_PI_2 - r;
        (quarter_sin(rr), quarter_cos(rr))
    } else {
        (quarter_cos(r), quarter_sin(r))
    };
    match q {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// 确定性 `sin`（`cos_sin` 同归约路径）。
#[inline]
pub fn sin(x: f32) -> f32 {
    cos_sin(x).1
}

/// 确定性 `cos`（`cos_sin` 同归约路径）。
#[inline]
pub fn cos(x: f32) -> f32 {
    cos_sin(x).0
}

/// 确定性 `tan`（`sin/cos` 同一归约路径）。
///
/// 唯一调用点 = 相机焦距标度 `tan(fov/2)`，`fov ∈ (0°,180°)` 开区间，此时
/// `cos(fov/2) > 0` 无奇点。精度 ~1e-6 量级，足够相机几何。
#[inline]
pub fn tan(x: f32) -> f32 {
    let (c, s) = cos_sin(x);
    s / c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_cos_match_reference_points() {
        // 关键角参考值（与平台 libm 比对仅用于测试断言，产品代码不含 libm）。
        let cases = [
            (0.0f32, 0.0f32, 1.0f32),
            (FRAC_PI_2, 1.0, 0.0),
            (PI, 0.0, -1.0),
            (-FRAC_PI_2, -1.0, 0.0),
            (FRAC_PI_4, 0.707_106_77, 0.707_106_77),
        ];
        for (x, want_s, want_c) in cases {
            let (c, s) = cos_sin(x);
            assert!((s - want_s).abs() < 1e-6, "sin({x})={s} 期望 {want_s}");
            assert!((c - want_c).abs() < 1e-6, "cos({x})={c} 期望 {want_c}");
        }
    }

    #[test]
    fn sin_cos_are_deterministic_and_odd_even() {
        for x in [-3.0f32, -1.2, 0.5, 2.7, 5.9] {
            assert_eq!(cos_sin(x), cos_sin(x), "同输入必同输出");
            let (c, s) = cos_sin(x);
            let (cn, sn) = cos_sin(-x);
            assert!((c - cn).abs() < 1e-6, "cos 偶对称");
            assert!((s + sn).abs() < 1e-6, "sin 奇对称");
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        for x in [-2.5f32, -0.3, 0.9, 1.7, 4.1] {
            let (c, s) = cos_sin(x);
            assert!((c * c + s * s - 1.0).abs() < 1e-5, "sin²+cos²=1");
        }
    }

    #[test]
    fn tan_matches_ratio_and_camera_scale() {
        // fov = 90° → tan(45°) = 1。
        assert!((tan(FRAC_PI_2 / 2.0) - 1.0).abs() < 1e-5);
        let (c, s) = cos_sin(0.7);
        assert!((tan(0.7) - s / c).abs() < 1e-6);
    }
}
