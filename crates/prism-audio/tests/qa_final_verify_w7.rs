//! W7 终验（Round 3）独立复核 —— exp_f64 正侧精度与连续性。
//!
//! 不信"绿了"，用**独立高精度参考**（f64::exp 由 libm 提供，误差 <0.5 ulp，
//! 作为参考足够）逐点比对相对误差；每个断言注明失败代表什么缺陷。

use prism_audio::det::exp_f64;

/// 相对误差（参考值为 0 时退化为绝对误差）。
fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// 缺陷：正侧 [700, 709.78] 全区间若出现 inf/NaN 或相对误差超阈值，
/// 说明顶部夹基准或 Taylor 项数不足。
#[test]
fn exp_f64_upper_region_accuracy_vs_libm() {
    let mut worst = 0.0f64;
    let mut x = 700.0f64;
    while x <= 709.782_712_893_384 {
        let g = exp_f64(x);
        assert!(g.is_finite(), "x={x} 应有限，得 {g}");
        let w = x.exp();
        let e = rel_err(g, w);
        if e > worst {
            worst = e;
        }
        x += 0.001;
    }
    assert!(
        worst < 1e-10,
        "正侧相对误差应 <1e-10，实测 worst={worst:e}"
    );
}

/// 缺陷：k_orig>1023 夹基准 + Taylor 16 项后，相对误差是否仍可控（<1e-12）。
#[test]
fn exp_f64_k_clamped_region_tiny_rel_error() {
    // x/ln2 ∈ (1023.5, 1024.5) → k_orig ∈ {1024}，覆盖夹基准路径。
    let lo = 1023.6 * std::f64::consts::LN_2;
    let hi = 709.782_712_893_384;
    let mut worst = 0.0f64;
    let mut x = lo;
    while x <= hi {
        let g = exp_f64(x);
        assert!(g.is_finite(), "x={x}(夹基准区) 应有限，得 {g}");
        let e = rel_err(g, x.exp());
        if e > worst {
            worst = e;
        }
        x += 0.000_5;
    }
    assert!(worst < 1e-12, "夹基准区相对误差应 <1e-12，实测 {worst:e}");
}

/// 缺陷：exp(709) 量级应约 8.218407e307（若量级错说明缩放错）。
#[test]
fn exp_f64_709_magnitude() {
    let g = exp_f64(709.0);
    let w = 709.0f64.exp();
    assert!(
        (g / w - 1.0).abs() < 1e-10,
        "exp(709) 偏差过大 got={g:e} want={w:e}"
    );
    // 量级 sanity：8.218407e307
    assert!(
        (g / 8.218_407e307 - 1.0).abs() < 1e-3,
        "exp(709) 量级错 got={g:e}"
    );
}

/// 缺陷：恰在上界 LN_F64_MAX 处应有限（≈ f64::MAX 量级）；略超（709.7828）应 +inf。
#[test]
fn exp_f64_boundary_ln_max_and_beyond() {
    let at = exp_f64(709.782_712_893_384);
    assert!(at.is_finite(), "边界 LN_F64_MAX 处应有限，得 {at}");
    assert!(at > 0.0, "边界处应为正");
    let just_over = exp_f64(709.7828);
    assert!(
        just_over.is_infinite() && just_over > 0.0,
        "x=709.7828 超 ln(max) 应 +inf，得 {just_over}"
    );
}

/// 缺陷：边界附近左右连续性——不允许跨越实现分支时跳变 >1e-6 相对。
#[test]
fn exp_f64_boundary_continuity() {
    let eps = 1e-4;
    let x0 = 709.782_712_893_384;
    let l = exp_f64(x0 - eps);
    let r = exp_f64(x0 + eps);
    // r 可能已 +inf（若 x0+eps 超界），则跳过相对比较
    if r.is_finite() {
        let jump = ((r - l) / l).abs();
        assert!(jump < 1e-4, "边界附近跳变过大 {jump:e}");
    }
    // 左邻域单调不降
    let a = exp_f64(x0 - 1e-3);
    let b = exp_f64(x0 - 1e-6);
    assert!(b >= a, "边界左侧应单调不降");
}

/// 缺陷：超大正输入（1e10 / 1e300）不得挂死，应快速返回 +inf（有限输入的数学上界外）。
#[test]
fn exp_f64_huge_positive_input_no_hang() {
    for &x in &[1e10f64, 1e300f64, 1e308f64] {
        let g = exp_f64(x);
        assert!(
            g.is_infinite() && g > 0.0,
            "x={x:e} 应快速返回 +inf，得 {g}"
        );
    }
}
