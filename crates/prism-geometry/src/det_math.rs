//! 确定性数学件（D-29 简化件）：libm 超越函数的确定性替代。
//!
//! # 与 rt 先例的关系
//! 同 `prism-rt` 已落地的 det 件（`sampler::cos_sin`/`tan`、`media::det_ln`）
//! 同思路同式：纯 IEEE 运算（`+ − × ÷`、位运算、固定常量多项式/级数），
//! 无 libm、无环境依赖，同输入逐位一致。落点说明：geometry 的 winding
//! 生产链不得反向依赖 render 层，故独立成件；待 prism-math 的 D-29 确定性
//! 数学件上收时与 rt 侧合并（D-16 避免双实现的最终归宿）。
//!
//! # 件清单与误差（对照 f64 libm 参考，仅测试内用 libm）
//! - [`det_cos_sin`] / [`det_cos`] / [`det_sin`]：mod 2π 归约（精确 floor）→
//!   象限对称 → [0, π/4] Maclaurin；截断 ~2.5e-8。
//! - [`det_atan2`]：斜率归约（长边分母，构造上不溢出）→ 两次 atan 恒等式
//!   归约到 [0, tan(π/12)] → 奇 Maclaurin（含 x¹³ 项，截断 < 2e-9）；
//!   全域对照 < 1e-6 rad。
//! - [`det_asin`]：恒等式 `asin z = atan2(z, √((1−z)(1+z)))`，复用
//!   [`det_atan2`] 零新增多项式；全域对照 < 1e-6 rad。
//!
//! sqrt 为 IEEE 精确舍入运算，不在 D-29 禁列（附注同 rt 先例）。

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// 2π（mod 归约模数）。
const TWO_PI: f32 = 2.0 * PI;
/// π/12（atan 恒等式归约每次加的角）。
const FRAC_PI_12: f32 = 0.261_799_39;
/// tan(π/12) = 2 − √3（归约常量）。恒等式 `atan x = atan c + atan((x−c)/(1+x·c))`
/// 对任意 c > 0 精确成立（atan 加法公式），常量本身无需精确等于 tan(π/12)——
/// 只需与加角 π/12 单调一致（误差 ~1e-9，远低于断言阈）。
const TAN_PI_12: f32 = 0.267_949_19;

/// `[0, π/4]` 上的 sin（固定系数 Maclaurin，Horner 排布；与 rt 同式）。
#[inline]
fn quarter_sin(x: f32) -> f32 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362_880.0)))))
}

/// `[0, π/4]` 上的 cos（固定系数 Maclaurin，Horner 排布；与 rt 同式）。
#[inline]
fn quarter_cos(x: f32) -> f32 {
    let x2 = x * x;
    1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40_320.0))))
}

/// 任意角 `(cos x, sin x)`：mod 2π 归约（精确 floor）→ 象限对称 →
/// `[0, π/4]` 级数。跨平台逐位一致（只含 ±×÷ 与精确 floor；输入量级按
/// 引擎场景设计——winding 链锥角 ≤ π/2，同 rt 相机角/相位角）。
pub fn det_cos_sin(x: f32) -> (f32, f32) {
    // 归约到 [0, 2π)。
    let mut r = x - TWO_PI * (x / TWO_PI).floor();
    // 象限号：q = floor(r / (π/2)) ∈ {0,1,2,3}。
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

/// 确定性 cos（[`det_cos_sin`] 的标量取件）。
#[inline]
pub fn det_cos(x: f32) -> f32 {
    det_cos_sin(x).0
}

/// 确定性 sin（[`det_cos_sin`] 的标量取件）。
#[inline]
pub fn det_sin(x: f32) -> f32 {
    det_cos_sin(x).1
}

/// `atan(a)`，a ∈ [0, 1]（内部件；域由 [`det_atan2`] 的斜率归约构造性保证）。
///
/// 至多两次恒等式归约 `atan(x) = π/12 + atan((x−c)/(1+x·c))`（c = tan(π/12)，
/// 第一次把 (c, 1] 压到 [0, tan(π/6)]，第二次压到 [0, tan(π/12)]），再上奇
/// Maclaurin（含 x¹³ 项；截断余项 x¹⁵/15 < 2e-9）。只含 ±×÷ 与固定常量。
#[inline]
fn det_atan01(a: f32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&a), "det_atan01 定义域 [0,1]，收到 {a}");
    let mut x = a;
    let mut acc = 0.0_f32;
    if x > TAN_PI_12 {
        x = (x - TAN_PI_12) / (1.0 + x * TAN_PI_12);
        acc += FRAC_PI_12;
    }
    if x > TAN_PI_12 {
        x = (x - TAN_PI_12) / (1.0 + x * TAN_PI_12);
        acc += FRAC_PI_12;
    }
    let x2 = x * x;
    acc
        + x * (1.0
            + x2 * (-1.0 / 3.0
                + x2 * (1.0 / 5.0
                    + x2 * (-1.0 / 7.0
                        + x2 * (1.0 / 9.0 + x2 * (-1.0 / 11.0 + x2 * (1.0 / 13.0)))))))
}

/// 全象限确定性 `atan2(y, x)`（弧度 ∈ [−π, π]；D-29：纯 IEEE 运算）。
///
/// 斜率归约：长边做分母把斜率压入 [0,1]——num ≤ den 为实数构造，商的最近
/// 舍入不超过 1，无溢出通道；短长互换时以 π/2 翻转。象限由 (x 符号, y 符号)
/// 的 IEEE 比较显式装配（比较精确，无分支歧义）。约定：`atan2(0, 0) = 0`；
/// y = ±0 且 x < 0 统一返回 π（引擎内不区分 ±0）；NaN 输入按 IEEE 传播。
pub fn det_atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 {
        return if y > 0.0 {
            FRAC_PI_2
        } else if y < 0.0 {
            -FRAC_PI_2
        } else {
            0.0
        };
    }
    let (ax, ay) = (x.abs(), y.abs());
    let mut ang = if ay <= ax {
        det_atan01(ay / ax)
    } else {
        FRAC_PI_2 - det_atan01(ax / ay)
    };
    if x < 0.0 {
        // 左半平面：上侧 (0, π]，下侧 (−π, 0)。
        ang = if y >= 0.0 { PI - ang } else { ang - PI };
    } else if y < 0.0 {
        // 右半平面下侧。
        ang = -ang;
    }
    ang
}

/// 确定性 `asin(z)`，z ∈ [0, 1]（winding 视线锥半角用；域外 debug_assert）。
///
/// 恒等式 `asin z = atan2(z, √((1−z)(1+z)))`——零新增多项式，复用
/// [`det_atan2`]；`(1−z)(1+z)` 逐因子相乘避免 1−z² 的尾位消去；z = 1 经
/// atan2 的 x = 0 分支精确到 π/2。
#[inline]
pub fn det_asin(z: f32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&z), "det_asin 定义域 [0,1]，收到 {z}");
    det_atan2(z, ((1.0 - z) * (1.0 + z)).sqrt())
}

// ---------------------------------------------------------------------------
// 测试（精度锚点：f64 libm 参考——D-29 禁令只约束引擎确定性路径，不约束测试）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_noise::pcg;

    /// 代表样本格（象限 × 量级 × 轴上特例）上 det_atan2 vs libm < 1e-6 rad。
    #[test]
    fn det_atan2_matches_libm_on_representative_grid() {
        let mags = [
            0.0_f32, 1e-6, 1e-3, 0.05, 0.2, 0.5, 0.707_106_8, 1.0, 1.5, 3.0, 100.0, 1e6,
            -1e-6, -0.05, -0.5, -1.0, -3.0, -100.0, -1e6,
        ];
        let mut max_err = 0.0_f64;
        for &x in &mags {
            for &y in &mags {
                let got = det_atan2(y, x) as f64;
                let want = (y as f64).atan2(x as f64);
                let err = (got - want).abs();
                max_err = max_err.max(err);
                assert!(err < 1e-6, "det_atan2({y}, {x}) = {got}，libm {want}，err {err:e}");
            }
        }
        println!("det_atan2 代表格最大误差 {max_err:.3e} rad");
    }

    /// 随机扫 2 万点（[−10,10]²）全域精度。
    #[test]
    fn det_atan2_random_sweep_matches_libm() {
        let mut max_err = 0.0_f64;
        for i in 0..20_000_u64 {
            let x = pcg::hash2_f32(0xA7_02, i, 1) * 20.0 - 10.0;
            let y = pcg::hash2_f32(0xA7_02, i, 2) * 20.0 - 10.0;
            let got = det_atan2(y, x) as f64;
            let want = (y as f64).atan2(x as f64);
            let err = (got - want).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-6, "det_atan2({y}, {x}) = {got}，libm {want}，err {err:e}");
        }
        println!("det_atan2 随机扫最大误差 {max_err:.3e} rad");
    }

    /// 象限装配锚点（轴上特例与对角线，公开数学常量）。
    #[test]
    fn det_atan2_quadrant_anchors() {
        let close = |got: f32, want: f64| assert!((got as f64 - want).abs() < 1e-6);
        close(det_atan2(0.0, 1.0), 0.0);
        close(det_atan2(1.0, 1.0), std::f64::consts::FRAC_PI_4);
        close(det_atan2(1.0, 0.0), std::f64::consts::FRAC_PI_2);
        close(det_atan2(0.0, -1.0), std::f64::consts::PI);
        close(det_atan2(1.0, -1.0), 3.0 * std::f64::consts::FRAC_PI_4);
        close(det_atan2(-1.0, -1.0), -3.0 * std::f64::consts::FRAC_PI_4);
        close(det_atan2(-1.0, 1.0), -std::f64::consts::FRAC_PI_4);
        close(det_atan2(-1.0, 0.0), -std::f64::consts::FRAC_PI_2);
        assert_eq!(det_atan2(0.0, 0.0).to_bits(), 0.0f32.to_bits(), "atan2(0,0) 约定为 0");
    }

    /// det_asin vs libm < 1e-6 rad（端点 + 代表点 + 随机扫）。
    #[test]
    fn det_asin_matches_libm() {
        let zs = [0.0_f32, 1e-6, 0.05, 0.25, 0.5, 0.707_106_8, 0.9, 0.99, 0.999_9, 1.0];
        let mut max_err = 0.0_f64;
        for &z in &zs {
            let got = det_asin(z) as f64;
            let want = (z as f64).asin();
            let err = (got - want).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-6, "det_asin({z}) = {got}，libm {want}，err {err:e}");
        }
        for i in 0..10_000_u64 {
            let z = pcg::hash2_f32(0xA5_1A, i, 3);
            let got = det_asin(z) as f64;
            let want = (z as f64).asin();
            let err = (got - want).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-6, "det_asin({z}) = {got}，libm {want}，err {err:e}");
        }
        println!("det_asin 最大误差 {max_err:.3e} rad");
    }

    /// det 三角件数学常量锚点（非表数据）。
    #[test]
    fn det_trig_matches_known_constants() {
        let cases: [(f32, (f64, f64)); 5] = [
            (1.0, (0.540_302_3, 0.841_471_0)),
            (2.0, (-0.416_146_8, 0.909_297_4)),
            (std::f32::consts::FRAC_PI_6, (0.866_025_4, 0.5)),
            (std::f32::consts::FRAC_PI_3, (0.5, 0.866_025_4)),
            (5.0, (0.283_662_2, -0.958_924_3)),
        ];
        for (x, (ec, es)) in cases {
            let (c, s) = det_cos_sin(x);
            assert!((c as f64 - ec).abs() < 2.0e-6, "cos({x}) = {c} vs {ec}");
            assert!((s as f64 - es).abs() < 2.0e-6, "sin({x}) = {s} vs {es}");
        }
    }

    /// det 三角件随机扫 vs f64 libm（|x| ≤ 10，覆盖全部象限与归约路径）。
    #[test]
    fn det_trig_random_sweep_matches_libm() {
        let mut max_err = 0.0_f64;
        for i in 0..20_000_u64 {
            let x = pcg::hash2_f32(0xC0_5E, i, 1) * 20.0 - 10.0;
            let (c, s) = det_cos_sin(x);
            let (cw, sw) = ((x as f64).cos(), (x as f64).sin());
            let err = (c as f64 - cw).abs().max((s as f64 - sw).abs());
            max_err = max_err.max(err);
            assert!(err < 1e-5, "det_cos_sin({x}) = ({c}, {s})，err {err:e}");
        }
        println!("det_cos_sin 随机扫最大误差 {max_err:.3e}");
    }

    /// 确定性：同输入两遍逐位一致（纯函数性质的显式锚）。
    #[test]
    fn det_pieces_are_bit_deterministic() {
        for x in [0.3_f32, 1.7, -2.9, 44.25] {
            let a = det_cos_sin(x);
            let b = det_cos_sin(x);
            assert_eq!(a.0.to_bits(), b.0.to_bits());
            assert_eq!(a.1.to_bits(), b.1.to_bits());
            assert_eq!(det_cos(x).to_bits(), a.0.to_bits());
            assert_eq!(det_sin(x).to_bits(), a.1.to_bits());
        }
        for (y, x) in [(0.5_f32, 1.5), (-2.0, -0.25), (3.0, -3.0), (0.0, -7.0)] {
            assert_eq!(det_atan2(y, x).to_bits(), det_atan2(y, x).to_bits());
        }
        assert_eq!(det_asin(0.42).to_bits(), det_asin(0.42).to_bits());
    }
}
