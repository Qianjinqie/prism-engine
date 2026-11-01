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

// ---------------------------------------------------------------------------
// 反三角（W1-B：uv 参数化需要 atan2；D-29 同款纪律，禁 libm）
// ---------------------------------------------------------------------------

/// `[0, 1]` 上的 `atan`（固定系数 Maclaurin，Horner 排布）。
///
/// 调用方先把自变量归约到 `|z| ≤ tan(π/8) = 0.41421356`（见 [`atan_pos`]）。
/// `z^17/17` 截断项在 `z = 0.41421` 处为 **1.8e-8**（绝对），相对 π 为 6e-9，
/// 远低于 uv 参数化所需精度（uv 的分辨率是纹素，最细纹理也只需 ~1e-5）。
#[inline]
fn quarter_atan(z: f32) -> f32 {
    let z2 = z * z;
    z * (1.0
        + z2 * (-1.0 / 3.0
            + z2 * (1.0 / 5.0
                + z2 * (-1.0 / 7.0
                    + z2 * (1.0 / 9.0
                        + z2 * (-1.0 / 11.0
                            + z2 * (1.0 / 13.0 + z2 * (-1.0 / 15.0))))))))
}

/// `atan(x)`，`x >= 0`，返回 `[0, π/2]`。
///
/// 三段归约（每段只含 `± × ÷`，故跨平台逐位一致）：
/// 1. `x > 1`：`atan(x) = π/2 − atan(1/x)`；
/// 2. `x ∈ (tan(π/8), 1]`：`atan(x) = π/4 + atan((x−1)/(x+1))`（`|z| ≤ tan(π/8)`）；
/// 3. `[0, tan(π/8)]`：直接级数。
#[inline]
fn atan_pos(x: f32) -> f32 {
    const TAN_PI_8: f32 = 0.414_213_57;
    if x > 1.0 {
        return FRAC_PI_2 - atan_pos(1.0 / x);
    }
    if x > TAN_PI_8 {
        return FRAC_PI_4 + quarter_atan((x - 1.0) / (x + 1.0));
    }
    quarter_atan(x)
}

/// 确定性 `atan2(y, x)`（弧度，值域 `(−π, π]`）。
///
/// # 为什么需要它
///
/// uv 参数化（球 / 柱 / 锥 / 环 / 旋转面）**必须**知道绕轴方位角，而
/// `y.atan2(x)` 落平台 libm（C17 禁）。本实现只含 `± × ÷` 与比较，
/// 同输入跨平台逐位一致。
///
/// # 特例（与 IEEE `atan2` 口径一致，且都不产生 NaN）
///
/// - `x == 0 && y == 0` → `0`（方向未定义；uv 参数化在极点处本就测度零）；
/// - `x == 0` → `±π/2`；`y == 0` → `0` 或 `±π`；
/// - 非有限输入 → `0`（防御：不让 NaN 渗进 uv，再由采样器钳制）。
#[inline]
pub fn atan2(y: f32, x: f32) -> f32 {
    if !y.is_finite() || !x.is_finite() {
        return 0.0;
    }
    if x == 0.0 {
        if y == 0.0 {
            return 0.0;
        }
        return if y > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }
    let ax = x.abs();
    let ay = y.abs();
    // r = atan(|y/x|) 的稳定形式（避免比值溢出）：大者作分母。
    let r = if ay <= ax {
        atan_pos(ay / ax)
    } else {
        FRAC_PI_2 - atan_pos(ax / ay)
    };
    if x > 0.0 {
        if y >= 0.0 {
            r
        } else {
            -r
        }
    } else if y >= 0.0 {
        PI - r
    } else {
        r - PI
    }
}

/// 确定性 `acos(x)`（值域 `[0, π]`；`|x| > 1` 时按端点钳制）。
///
/// `acos(x) = atan2(√(1−x²), x)`：`x` 靠近 ±1 时 `1−x²` 用 `(1−x)(1+x)`
/// 展开求值（避免相消），仍是精确舍入的乘减 ⇒ 逐位可复现。
#[inline]
pub fn acos(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let xc = if x > 1.0 {
        1.0
    } else if x < -1.0 {
        -1.0
    } else {
        x
    };
    let s = ((1.0 - xc) * (1.0 + xc)).max(0.0).sqrt();
    atan2(s, xc)
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

    // ---- W1-B：atan2 / acos（uv 参数化的依赖件）----

    #[test]
    fn atan2_matches_reference_quadrants() {
        // 参考值来自解析角（与平台 libm 比对仅用于测试断言）。
        let cases: [(f32, f32, f32); 8] = [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, FRAC_PI_2),
            (0.0, -1.0, PI),
            (-1.0, 0.0, -FRAC_PI_2),
            (1.0, 1.0, FRAC_PI_4),
            (-1.0, 1.0, -FRAC_PI_4),
            (1.0, -1.0, 3.0 * FRAC_PI_4),
            (-1.0, -1.0, -3.0 * FRAC_PI_4),
        ];
        for (y, x, want) in cases {
            let got = atan2(y, x);
            assert!((got - want).abs() < 1e-5, "atan2({y},{x}) = {got}，期望 {want}");
        }
    }

    #[test]
    fn atan2_error_is_below_one_microradian_over_full_circle() {
        // 全圆周扫掠：与解析角比对，最大误差须 < 1e-6（uv 精度需求 ~1e-5）。
        let mut max_err = 0.0f32;
        for k in 0..2048 {
            let a = -PI + 2.0 * PI * (k as f32) / 2048.0;
            let (c, s) = cos_sin(a);
            let got = atan2(s, c);
            // 角度差取环绕最短（±π 处符号跳变不算误差）。
            let mut d = got - a;
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            max_err = max_err.max(d.abs());
        }
        assert!(max_err < 1e-6, "atan2 全圆周最大误差 {max_err} ≥ 1e-6");
    }

    #[test]
    fn atan2_is_deterministic_and_handles_degenerate_inputs() {
        assert_eq!(atan2(0.3, -0.7).to_bits(), atan2(0.3, -0.7).to_bits());
        // 退化：原点 / 零轴 / 非有限 —— 一律不产生 NaN。
        assert_eq!(atan2(0.0, 0.0), 0.0);
        assert_eq!(atan2(0.0, -1.0), PI);
        assert_eq!(atan2(f32::NAN, 1.0), 0.0);
        assert_eq!(atan2(1.0, f32::INFINITY), 0.0);
        // 奇对称（同一 x）：atan2(−y,x) == −atan2(y,x)，除 ±π 边界外。
        for (y, x) in [(0.4f32, 0.9f32), (0.9, 0.4), (-0.2, -0.8)] {
            assert_eq!(atan2(-y, x), -atan2(y, x));
        }
    }

    #[test]
    fn acos_matches_reference_points() {
        assert!((acos(1.0) - 0.0).abs() < 1e-6);
        assert!((acos(0.0) - FRAC_PI_2).abs() < 1e-6);
        assert!((acos(-1.0) - PI).abs() < 1e-5);
        assert!((acos(0.5) - 1.047_197_6).abs() < 1e-5);
        // 域外钳制（不产生 NaN）。
        assert_eq!(acos(1.5), 0.0);
        assert!((acos(-1.5) - PI).abs() < 1e-5);
        assert_eq!(acos(f32::NAN), 0.0);
    }

    #[test]
    fn pythagorean_identity_survives_acos_roundtrip() {
        // v = acos(y)/π 是球面 v 参数化的核心：反解 cos(v·π) 须回到原 y。
        for k in 0..=64 {
            let y = -1.0 + 2.0 * (k as f32) / 64.0;
            let v = acos(y) / PI;
            let (c, _s) = cos_sin(v * PI);
            assert!((c - y).abs() < 2e-6, "acos/cos 往返 y={y} 得到 {c}");
        }
    }
}

