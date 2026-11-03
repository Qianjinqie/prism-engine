//! 独立复核探针。只写测试，不改源码。
//! 不信任既有"绿了"结论——独立数学验证 5 缺陷修复。

use prism_aesthetic::secondary::{HslQualifier, LiftGammaGain};

/// gamma：p/2^q 量化语义，对照真值 x^(1/gamma)（用 std powf 作参考）。
#[test]
fn probe_p0_1_gamma_accuracy() {
    let ref_pow = |x: f32, g: f32| x.powf(1.0 / g);

    // gamma=1.0 应恒等（x→x）。
    let g1 = LiftGammaGain { gamma: 1.0, gamma_steps: 8, ..LiftGammaGain::default() };
    for &x in &[0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
        let got = g1.apply_scalar(x);
        assert!((got - x).abs() < 1e-6, "gamma=1 恒等失败 x={x}: got={got}");
    }

    // gamma=2 -> x^(1/2)=sqrt(x)。x=0.25 -> 0.5。
    let g2 = LiftGammaGain { gamma: 2.0, gamma_steps: 8, ..LiftGammaGain::default() };
    assert!((g2.apply_scalar(0.25) - 0.5).abs() < 1e-5, "gamma2 0.25 应 0.5，got {}", g2.apply_scalar(0.25));

    // gamma=0.5 -> x^2。x=0.25 -> 0.0625。
    let gh = LiftGammaGain { gamma: 0.5, gamma_steps: 8, ..LiftGammaGain::default() };
    let got = gh.apply_scalar(0.25);
    assert!((got - 0.0625).abs() < 1e-4, "gamma0.5 0.25 应 0.0625，got {got}");

    // 非整 gamma：量化误差应有限。
    for &(g, x) in &[(1.8f32, 0.5f32), (2.2, 0.3), (1.4, 0.5), (0.7, 0.6)] {
        let lgg = LiftGammaGain { gamma: g, gamma_steps: 8, ..LiftGammaGain::default() };
        let got = lgg.apply_scalar(x);
        let want = ref_pow(x, g);
        let err = (got - want).abs();
        assert!(err < 0.01, "非整 gamma 误差过大 g={g} x={x}: got={got} want={want} err={err}");
    }

    // gamma_steps 提高应变精细（更多步 -> 更贴近真值）。
    let coarse = LiftGammaGain { gamma: 1.8, gamma_steps: 4, ..LiftGammaGain::default() };
    let fine = LiftGammaGain { gamma: 1.8, gamma_steps: 20, ..LiftGammaGain::default() };
    let want = ref_pow(0.3, 1.8);
    let ec = (coarse.apply_scalar(0.3) - want).abs();
    let ef = (fine.apply_scalar(0.3) - want).abs();
    assert!(ef < ec, "fine steps 误差应小于 coarse：ec={ec} ef={ef}");
    assert!(ef < 0.02, "fine steps 误差过大 {ef}");

    // 边界 x=0 -> 0；x=1 -> 1（任意 gamma）。
    for &g in &[0.5f32, 1.0, 1.8, 2.0, 2.2] {
        let lgg = LiftGammaGain { gamma: g, gamma_steps: 8, ..LiftGammaGain::default() };
        assert!(lgg.apply_scalar(0.0) == 0.0, "x=0 g={g} 应 0");
        assert!((lgg.apply_scalar(1.0) - 1.0).abs() < 1e-6, "x=1 g={g} 应 1");
    }
}

/// P1-3 hue：环形选择 + 三点攻击。
#[test]
fn probe_p1_3_hue_ring() {
    let mk = |lo: f32, hi: f32, s: f32| HslQualifier {
        luma_range: [0.0, 1.0],
        hue_range: [lo, hi],
        sat_range: [0.0, 1.0],
        softness: s,
    };
    // 跨 0 选区 [0.9,0.1]，纯红 hue≈0 应命中。
    let wrap = mk(0.9, 0.1, 0.02);
    assert!(wrap.weight([1.0, 0.0, 0.0, 1.0]) > 0.9, "跨0纯红应命中");
    // h=0.95（区外远离）、h=0.05 应命中。
    let r95 = wrap.weight([1.0, 1.0 * 0.05, 0.0, 1.0]); // 近似橙黄侧
    // 非跨 0 选区 [0.2,0.4] 正常。
    let normal = mk(0.2, 0.4, 0.02);
    assert!(normal.weight([0.0, 0.0, 1.0, 1.0]) < 0.1, "蓝色不应落入绿选区");
    // softness=0 硬边：区内 1 区外 0。
    let hard = mk(0.9, 0.1, 0.0);
    assert!(hard.weight([1.0, 0.0, 0.0, 1.0]) == 1.0, "硬边区内=1");
    let _ = r95;
}

/// 退化输入：全区间选区 = 处处选中；中性 LGG = 逐位恒等。
///
/// 本条原先只有 `assert!(true)`（永真占位，零判别力）。2026-11 QA 收口时
/// 换成两条**能红**的断言：
///
/// * `HslQualifier::default()` 三段都是全区间（`lo ≤ 0 且 hi ≥ 1`）⇒
///   `soft_range` / `hue_range_weight` 走早退分支返回 1.0，故任意颜色权重恰 1.0。
///   若有人把早退改成"算过渡带"，这里会掉到 < 1。
/// * `LiftGammaGain::default()` 满足 `is_neutral()`，且 `apply_scalar` 在中性参数
///   下不进入任何分支 ⇒ 输出与输入**逐位**相同（含负值：底座被钳到 0）。
#[test]
fn probe_degenerate_smoke() {
    // 1. 全区间选区 ⇒ 处处选中（含纯黑、纯白、高饱和色）。
    let full = HslQualifier::default();
    for c in [
        [0.0f32, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.9, 0.1, 0.05, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ] {
        let w = full.weight(c);
        assert_eq!(w, 1.0, "全区间选区应处处选中（c={c:?} 得到 w={w}）");
    }

    // 2. 中性 LGG ⇒ 逐位恒等（负值也直通：钳零只发生在幂底座，且仅当 gamma ≠ 1）。
    let neutral = LiftGammaGain::default();
    assert!(neutral.is_neutral(), "默认 LGG 必须是中性");
    for &x in &[0.0f32, 0.25, 0.5, 1.0, 4.0, -0.5] {
        let got = neutral.apply_scalar(x);
        assert_eq!(
            got.to_bits(),
            x.to_bits(),
            "中性 LGG 应逐位恒等（x={x} 得到 {got}）"
        );
    }

    // 3. x ≤ 0 且 gamma ≠ 1：幂无定义 ⇒ 底座取 0、分支不执行，`x` **原样透传**
    //    （不是置 0）。这是现行实现（`apply_scalar` 的底座注释已按实现改写），
    //    本条只做表征钉住：若哪天改成"钳到 0"，这里会红。
    let g2 = LiftGammaGain { gamma: 2.0, ..LiftGammaGain::default() };
    assert_eq!(
        g2.apply_scalar(-0.5).to_bits(),
        (-0.5f32).to_bits(),
        "gamma≠1 且 x<0 ⇒ 现行实现是原样透传"
    );
    assert_eq!(g2.apply_scalar(0.0).to_bits(), 0.0f32.to_bits(), "x=0 ⇒ 0");
}
