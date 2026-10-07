//! 第 2 轮独立复核探针。
//! 不信任「绿了」，用独立断言直接打 P0/P2 修复边界。
//! 只读源码 + 独立探针，不改源码。

use prism_audio::det::exp_f64;
use prism_audio::sfz::parse_sfz;

/// P0 复核：非有限短路 + k 边界。必须有界、不挂死。
#[test]
fn p0_exp_f64_non_finite_and_huge_finite_are_bounded() {
    assert!(exp_f64(f64::NAN).is_nan());
    assert_eq!(exp_f64(f64::INFINITY), f64::INFINITY);
    assert_eq!(exp_f64(f64::NEG_INFINITY), 0.0);
    // 极大有限值（会令 k 巨量）必须立即返回、不进入 scale_pow2 长循环。
    assert_eq!(exp_f64(-1e9), 0.0);
    assert_eq!(exp_f64(1e9), f64::INFINITY);
    assert_eq!(exp_f64(-1e300), 0.0);
    assert_eq!(exp_f64(1e300), f64::INFINITY);
}

/// P0 边界：f64 最大正常 exp 边界 x≈709.78（k≈1023）不得被判成 +inf 前置短路。
#[test]
fn p0_exp_f64_upper_normal_region() {
    let v = exp_f64(709.0);
    assert!(v.is_finite() && v > 1e307, "exp(709) 应接近 f64 上限，got {v}");
    // 精确上限 ln(f64::MAX) ≈ 709.782712893384：此处数学结果仍为有限（≈MAX）。
    // 若返回 +inf 则为「上界短路过激」缺陷（k≥1024 提前触发）。
    let vmax = exp_f64(709.782712893384);
    assert!(vmax.is_finite(), "exp(ln MAX) 应有限（≈1.8e308），got {vmax}");
    assert!((vmax - f64::MAX).abs() < 1e292, "exp(ln MAX)≈MAX got {vmax}");
    // 超过上限应 → +inf
    assert_eq!(exp_f64(710.0), f64::INFINITY);
}

/// P0 边界：下溢边界 x≈-745（k≈-1075 处）。
#[test]
fn p0_exp_f64_lower_underflow_region() {
    // f64 最小次正规 ≈ 5e-324，对应 ln ≈ -744.44
    let v = exp_f64(-745.0);
    assert!(v >= 0.0 && v.is_finite(), "exp(-745) 应有限非负，got {v}");
    assert!(v < 1e-322, "exp(-745) 应为极次正规数，got {v}");
    // 更负：应迅速落 0
    assert_eq!(exp_f64(-800.0), 0.0);
    assert_eq!(exp_f64(-1075.0), 0.0);
    // -1023 附近仍可非零（k≈-1476? 需精确核对）；-1000 应已为 0
    assert_eq!(exp_f64(-1000.0), 0.0);
}

/// P0 单调性/中间区间可控：扫 -700..700 不挂死且单调不减。
#[test]
fn p0_exp_f64_monotone_midrange() {
    let mut prev = 0.0_f64;
    let mut x = -700.0_f64;
    while x <= 700.0 {
        let v = exp_f64(x);
        assert!(v.is_finite(), "exp({x}) 应有限");
        assert!(v >= prev, "exp 单调性破坏: exp({x})={v} < prev={prev}");
        prev = v;
        x += 13.0;
    }
}

/// P0 与标准库对照（测试内允许用 exp 参考）。
#[test]
fn p0_exp_f64_accuracy_vs_std() {
    let mut x = -20.0_f64;
    while x <= 20.0 {
        let got = exp_f64(x);
        let want = x.exp();
        let rel = if want != 0.0 { (got - want).abs() / want.abs() } else { got.abs() };
        assert!(rel < 1e-12, "exp({x}) 相对误差 {rel}: got {got} want {want}");
        x += 0.37;
    }
}

/// P2 复核：正常 region 不被误伤。
#[test]
fn p2_sfz_normal_region_kept_empty_skipped() {
    let ok = "<region> sample=kick.wav lokey=36 hikey=40\n";
    let regions = parse_sfz(ok).unwrap();
    assert_eq!(regions.len(), 1, "有 sample 的 region 不能丢: {regions:?}");
    assert_eq!(regions[0].sample, "kick.wav");

    let empty = "<region> lokey=36 hikey=40\n";
    let r2 = parse_sfz(empty).unwrap();
    assert_eq!(r2.len(), 0, "无 sample 的 region 必须跳过: {r2:?}");

    // 混合：一个空、一个有 sample
    let mix = "<region> lokey=0 hikey=10\n<region> sample=a.wav\n<region> lokey=20 hikey=30\n";
    let r3 = parse_sfz(mix).unwrap();
    assert_eq!(r3.len(), 1, "混合应只留 1 个有效 region: {r3:?}");
    assert_eq!(r3[0].sample, "a.wav");
}
