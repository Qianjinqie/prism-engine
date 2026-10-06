//! W9 Round-2 独立复核探针（QA 严过关 Edward）——P2 det_trig 自研三角件。
//! 精度对照参考值 + 极大角归约 + NaN/Inf 防挂死。
//! 只读探针，不改源码。所有断言带超时意图（若挂死即暴露）。

use prism_math::det_trig::{cos, cos_sin, sin, tan};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// P2-a：对照参考值（断言容差 1e-6，独立探针不信任被测自带断言）。
#[test]
fn p2_precision_vs_reference() {
    let cases: &[(f32, f32, f32)] = &[
        (0.0, 0.0, 1.0),
        (FRAC_PI_4, 0.707_106_78, 0.707_106_78),
        (FRAC_PI_2, 1.0, 0.0),
        (PI, 0.0, -1.0),
        (1.0, 0.841_470_98, 0.540_302_3),
        (-1.0, -0.841_470_98, 0.540_302_3),
        (2.0, 0.909_297_4, -0.416_146_8),
    ];
    for &(x, rs, rc) in cases {
        let (c, s) = cos_sin(x);
        assert!((s - rs).abs() < 1e-6, "sin({x})={s} 期望 {rs}");
        assert!((c - rc).abs() < 1e-6, "cos({x})={c} 期望 {rc}");
        assert!((sin(x) - s).abs() < 1e-7 && (cos(x) - c).abs() < 1e-7, "sin/cos 与 cos_sin 一致");
    }
}

/// P2-b：极大角度归约——防上轮 exp_f64 式挂死/灾难性精度损失。
/// 对多个大角验证 sin²+cos²≈1，且结果有界（不做无界循环）。
#[test]
fn p2_huge_angle_reduction_bounded() {
    for x in [1.0e4_f32, 1.0e5, 1.0e6, 1.0e7, -1.0e6, 123456.789_f32] {
        let (c, s) = cos_sin(x);
        assert!(c.is_finite() && s.is_finite(), "大角 x={x} 结果须有限，实测 c={c} s={s}");
        // 归约后仍在单位圆（精度随量级退化，放宽到 1e-2）。
        let norm = c * c + s * s;
        assert!((norm - 1.0).abs() < 1e-2, "x={x} 归约后应贴近单位圆，norm={norm}");
    }
}

/// P2-c：NaN/Inf 输入不得挂死、不得 panic；结果允许 NaN（数学上未定义），
/// 但必须**快速返回**。此处若实现存在无界循环会超时暴露。
#[test]
fn p2_nan_inf_no_hang() {
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (c, s) = cos_sin(x);
        let t = tan(x);
        // 不 panic 即可；标记非有限是诚实行为。
        let _ = (c, s, t);
    }
    // 若走到这里即未挂死。
    assert!(true, "NaN/Inf 输入未导致挂死");
}

/// P2-d：确定性——同输入逐位一致（跨平台铁律的可测代理）。
#[test]
fn p2_bitwise_deterministic() {
    for x in [0.3_f32, -2.7, 4.1, 1.0e5] {
        assert_eq!(cos_sin(x), cos_sin(x), "同输入必逐位一致");
    }
}

/// P2-e：tan 在相机半角区间内无奇点（fov∈(0,π) → fov/2∈(0,π/2)）。
#[test]
fn p2_tan_camera_scale_finite() {
    for fov in [0.1_f32, 0.5, 1.0, FRAC_PI_2, 3.0] {
        let t = tan(fov / 2.0);
        assert!(t.is_finite(), "fov={fov} tan(fov/2)={t} 应有限");
    }
}
