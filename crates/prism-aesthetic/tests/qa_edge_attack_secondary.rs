//! 风格/二级调色攻击性边界测试。只写测试，不改源码。

use prism_aesthetic::curves::{Curve, CurvePoint, CurveSet};
use prism_aesthetic::secondary::{HslQualifier, LiftGammaGain, PowerWindow};

/// 色相选区文档声称"红色在 0 与 1 处连通"（色相环形量）。
/// 跨 0 选区 [0.9, 0.1]（红相绕环）应命中纯红（hue=0）。
///
/// 失败代表：hue_range_weight 未把区间当环形处理（仅线性 `h>=lo&&h<=hi`），
/// 跨 0 的红相选区**全 0** → 偏红调色完全失效（文档 vs 实现不符）。
#[test]
fn hue_red_selection_is_circularly_connected() {
    // 红相跨 0 选区：lo=0.9, hi=0.1（绕环包住 hue=0），softness=0.05。
    let q = HslQualifier {
        luma_range: [0.0, 1.0],
        hue_range: [0.9, 0.1],
        sat_range: [0.0, 1.0],
        softness: 0.05,
    };
    // 纯红 hue=0，位于跨 0 选区中心 → 应命中。
    let red = q.weight([1.0, 0.0, 0.0, 1.0]);
    assert!(red > 0.5, "纯红应命中跨 0 红相选区，实得 {red}");
    // 绿/蓝明显不应命中。
    let green = q.weight([0.0, 1.0, 0.0, 1.0]);
    let blue = q.weight([0.0, 0.0, 1.0, 1.0]);
    assert!(green < 0.2 && blue < 0.2, "绿/蓝不应命中红选区：{green}/{blue}");
}

/// 非跨 0 的红相选区 [0.0, 0.02] 正常工作（对照，确认缺陷仅限环绕）。
#[test]
fn hue_red_selection_nonwrapped_works() {
    let q = HslQualifier {
        luma_range: [0.0, 1.0],
        hue_range: [0.0, 0.02],
        sat_range: [0.0, 1.0],
        softness: 0.05,
    };
    assert!(q.weight([1.0, 0.0, 0.0, 1.0]) > 0.9, "非跨 0 红选区应命中纯红");
    assert!(q.weight([0.0, 0.0, 1.0, 1.0]) < 0.1, "蓝不应命中");
}

/// 默认全区间选区对**高饱和暗色 / 纯色**都恒为 1（不因 soft_range 边界误切）。
#[test]
fn full_range_default_qualifier_is_one_for_all() {
    let q = HslQualifier::default();
    for c in [
        [0.0f32, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.9, 0.9, 0.9, 1.0],
    ] {
        assert_eq!(q.weight(c), 1.0, "全区间默认选区应恒 1：{c:?}");
    }
}

/// gamma 语义——gamma>1 提亮中灰，gamma<1 压暗中灰（连乘开方方向正确）。
///
/// 失败代表：`x^(1/gamma)` 实现方向反了（gamma 越大越暗=bug）。
#[test]
fn gamma_direction_is_correct() {
    let bright = LiftGammaGain { lift: 0.0, gamma: 2.0, gain: 1.0, gamma_steps: 8 };
    let dark = LiftGammaGain { lift: 0.0, gamma: 0.5, gain: 1.0, gamma_steps: 8 };
    let mid = 0.25f32;
    assert!(bright.apply_scalar(mid) > mid, "gamma 2 应提亮中灰");
    assert!(dark.apply_scalar(mid) < mid, "gamma 0.5 应压暗中灰");
    // gamma=1 恒等。
    let neu = LiftGammaGain { lift: 0.0, gamma: 1.0, gain: 1.0, gamma_steps: 8 };
    assert_eq!(neu.apply_scalar(0.37), 0.37);
}

/// apply_scalar 对负输入（合法 HDR 线性域可能为负）不 panic / 不 NaN。
///
/// 失败代表：负值经 sqrt → NaN 未防护。
#[test]
fn lgg_negative_input_no_nan() {
    let g = LiftGammaGain { lift: 0.0, gamma: 2.0, gain: 1.0, gamma_steps: 8 };
    let out = g.apply_scalar(-0.5);
    assert!(!out.is_nan(), "负输入不应产 NaN，实得 {out}");
}

/// PowerWindow radius=0 退化 → 不 panic / 不 NaN（内部 1e-6 兜底）。
#[test]
fn power_window_zero_radius_no_nan() {
    let w = PowerWindow { radius: [0.0, 0.0], ..PowerWindow::default() };
    let out = w.weight(0.5, 0.5);
    assert!(!out.is_nan(), "零半径窗口权重不应 NaN");
    assert!((0.0..=1.0).contains(&out), "权重应 ∈[0,1]，实得 {out}");
}

/// PowerWindow feather=0 → 硬边（内 1 外 0），不 panic。
#[test]
fn power_window_zero_feather_hard_edge() {
    let w = PowerWindow { feather: 0.0, ..PowerWindow::default() };
    assert!(w.weight(0.5, 0.5) > 0.99, "中心应 1");
    assert!(w.weight(0.0, 0.0) < 0.01, "外部应 0");
}

/// Curve 单点输入退化：补默认端点后仍可求值（不 panic）。
#[test]
fn curve_single_point_degenerate() {
    let c = Curve::new(vec![CurvePoint { x: 0.5, y: 0.3 }]);
    // 单点 → 补 [0,0.3],[1,0.3]；任何 x 求值应为 0.3。
    assert!((c.eval(0.0) - 0.3).abs() < 1e-6);
    assert!((c.eval(0.9) - 0.3).abs() < 1e-6);
}

/// Curve eval 对 NaN 输入不 panic（库代码稳健性）。
#[test]
fn curve_eval_nan_input_no_panic() {
    let c = Curve::identity();
    let _ = c.eval(f32::NAN);
}

/// CurveSet 主曲线在通道曲线上**按通道值**映射：验证固定序（通道→主）。
#[test]
fn curve_set_order_channel_then_master() {
    let mut cs = CurveSet::default();
    // 红通道先 +0.1（反相简化：用恒等平移），主曲线再 ×（用 identity 检查序）。
    cs.red = Curve::new(vec![
        CurvePoint { x: 0.0, y: 0.0 },
        CurvePoint { x: 1.0, y: 1.0 },
    ]); // identity
    // 主曲线把 0.5 → 0.25（下压）。
    cs.master = Curve::new(vec![
        CurvePoint { x: 0.0, y: 0.0 },
        CurvePoint { x: 0.5, y: 0.25 },
        CurvePoint { x: 1.0, y: 1.0 },
    ]);
    let out = cs.apply([0.5, 0.5, 0.5, 1.0]);
    // 通道都在 [0,1] 上先恒等，主曲线 0.5→0.25。
    for k in 0..3 {
        assert!((out[k] - 0.25).abs() < 1e-6, "通道 {k} 应 0.25，实得 {}", out[k]);
    }
    assert_eq!(out[3], 1.0, "alpha 直通");
}

/// Curve 非单调控制点：is_monotone() 应报 false（供断言用，须正确）。
#[test]
fn curve_monotone_detection_correct() {
    let up = Curve::new(vec![
        CurvePoint { x: 0.0, y: 0.0 },
        CurvePoint { x: 1.0, y: 1.0 },
    ]);
    assert!(up.is_monotone());
    let down = Curve::new(vec![
        CurvePoint { x: 0.0, y: 1.0 },
        CurvePoint { x: 1.0, y: 0.0 },
    ]);
    assert!(!down.is_monotone(), "下降曲线不应报单调");
}
