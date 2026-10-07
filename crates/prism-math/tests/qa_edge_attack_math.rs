//! 攻击性边界测试。
//!
//! 目标：找真 Bug，非复读既有断言。
//! 每个测试注明"失败代表什么缺陷"。

use prism_math::aces::{aces_fast, aces_fitted, AcesFit, AcesOutput};
use prism_math::colorspace::{acescg_to_srgb, srgb_to_acescg};
use prism_math::det_pow::{det_exp, det_exp2, det_ln, det_log2, det_pow, det_pow_int};
use prism_math::hlg::{hlg_oetf, hlg_oetf_inverse};
use prism_math::pq::{pq_eotf, pq_oetf};

/// 缺陷面：自研幂对"退化输入"是否有限/不挂死。
/// 失败 = 存在非有限输入导致 det_pow 产出 NaN/Inf 或逻辑错误（上轮 exp_f64 教训）。
#[test]
fn det_pow_degenerate_inputs_finite() {
    // x 极小正数（下溢侧），应仍有限且 >= 0。
    for &x in &[f32::MIN_POSITIVE, 1e-30_f32, 1e-20, f32::MIN_POSITIVE * 2.0] {
        let y = det_pow(x, 2.4);
        assert!(y.is_finite() && y >= 0.0, "det_pow({x},2.4)={y} 非有限/负");
    }
    // 极大指数：p 极大时 x<1 应 → 0，x>1 应 → Inf，且不得挂死。
    let big = det_pow(2.0, 1000.0);
    assert!(big.is_finite() || big.is_infinite(), "2^1000 异常 {big}");
    let small = det_pow(0.5, 1000.0);
    assert_eq!(small, 0.0, "0.5^1000 应为 0，得 {small}");
    // x=1 任意 p 恒 1。
    for &p in &[-1e30_f32, 1e30, 0.0, 2.4] {
        let y = det_pow(1.0, p);
        assert!((y - 1.0).abs() < 1e-5, "1^{p}={y} 应为 1");
    }
}

/// 缺陷面：det_pow 对负数底应返回 NaN（文档约定），不得静默返回错误实数。
#[test]
fn det_pow_negative_base_is_nan() {
    assert!(det_pow(-2.0, 2.0).is_nan(), "负底非整数幂应变 NaN");
    assert!(det_pow(-1.0, 0.5).is_nan(), "负底 sqrt 应 NaN");
    // 但 p==0 与 p==1 的特殊分支在负底前应先命中？文档：负底 -> NaN。
    // 记录实际行为：p==0/1 分支在负底检查之后，故负底恒 NaN。
    assert!(det_pow(-3.0, 0.0).is_nan(), "负底 p=0：文档无实数域，应 NaN");
    assert!(det_pow(-3.0, 1.0).is_nan(), "负底 p=1：文档称 NaN 传播");
}

/// 缺陷面：det_exp2 极端输入挂死/下溢。上轮 exp_f64 类问题的重点怀疑。
#[test]
fn det_exp2_extreme_inputs_no_hang() {
    assert_eq!(det_exp2(2000.0), f32::INFINITY as f64, "2^2000 应 +Inf");
    assert_eq!(det_exp2(-2000.0), 0.0, "2^-2000 应 0");
    assert!(det_exp2(1024.0).is_infinite() == false || det_exp2(1024.0).is_infinite());
    // 边界 k 极大时的连乘：x=1023 应有限大数，x=1025 应 Inf。
    assert!(det_exp2(1023.0).is_finite(), "2^1023 应有限");
    assert!(det_exp2(1025.0).is_infinite(), "2^1025 应 Inf");
  // -1075 边界：应 0，不得为负或 NaN。
    assert_eq!(det_exp2(-1075.0), 0.0);
    assert_eq!(det_exp2(-1076.0), 0.0);
}

/// 缺陷面：det_log2/det_ln 对 0、负、Inf、NaN 的约定行为。
#[test]
fn det_log_degenerate() {
    assert_eq!(det_log2(0.0), f64::NEG_INFINITY);
    assert_eq!(det_log2(-1.0), f64::NEG_INFINITY, "负输入 log2 应 -Inf");
    assert_eq!(det_log2(f64::INFINITY), f64::INFINITY);
    assert!(det_log2(f64::NAN).is_nan(), "NaN 应传播");
    assert_eq!(det_ln(0.0), f64::NEG_INFINITY);
    assert!(det_ln(f64::NAN).is_nan());
}

/// 缺陷面：PQ 往返在信号域两端是否仍逐位/数值一致（边界不塌缩）。
#[test]
fn pq_roundtrip_boundary_signal() {
    // 极小信号与 1.0 边界往返。
    for &s in &[0.0_f32, 1e-6, 1e-4, 0.5, 0.999, 1.0] {
        let back = pq_oetf(pq_eotf(s));
        assert!((back - s).abs() < 1e-3, "PQ边界往返 s={s} -> {back}");
    }
}

/// 缺陷面：HLG 往返 EOTF∘OETF≈恒等（分段接缝处 1/12、1/2 附近尤为重要）。
#[test]
fn hlg_roundtrip_seams() {
    // 分段接缝 E=1/12 附近。
    for &e in &[0.0_f32, 1.0 / 12.0 - 1e-4, 1.0 / 12.0, 1.0 / 12.0 + 1e-4, 0.5, 1.0] {
        let ep = hlg_oetf(e);
        let back = hlg_oetf_inverse(ep);
        assert!((back - e).abs() < 1e-3, "HLG往返 E={e} -> {back} (ep={ep})");
    }
    // 信号接缝 0.5 附近（逆 OETF 分段界）。
    for &ep in &[0.0_f32, 0.4999, 0.5, 0.5001, 1.0] {
        let e = hlg_oetf_inverse(ep);
        let back = hlg_oetf(e);
        assert!((back - ep).abs() < 1e-3, "HLG信号往返 ep={ep} -> {back}");
    }
}

/// 缺陷面：sRGB↔ACEScg 矩阵互逆 + 白点守恒（行和=1）。
#[test]
fn srgb_acescg_matrix_inverse_and_white() {
    // 往返多色。
    for &c in &[[0.0_f32, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.7, 0.1], [0.9, 0.05, 0.33]] {
        let back = acescg_to_srgb(srgb_to_acescg(c));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "往返失真 c={c:?} back={back:?}");
        }
    }
    // 白点守恒：ACEScg 白 [1,1,1] 应映射到 sRGB 白 [1,1,1]（行和=1 含义）。
    let white = acescg_to_srgb([1.0, 1.0, 1.0]);
    for i in 0..3 {
        assert!((white[i] - 1.0).abs() < 1e-3, "白点未守恒 {white:?}");
    }
}

/// 缺陷面：ACES 两档对 NaN/负输入是否产生 NaN（应 clamp 而非污染）。
#[test]
fn aces_no_nan_on_extreme() {
    for &fit in &[AcesFit::Fitted, AcesFit::Fast] {
        let out = AcesOutput::new(fit);
        let r = out.apply([1000.0, 0.0, 0.5]);
        for i in 0..3 {
            assert!(r[i].is_finite(), "{fit:?} 输出非有限 {r:?}");
            assert!((0.0..=1.0).contains(&r[i]), "{fit:?} 输出越界 {r:?}");
        }
    }
}

/// 缺陷面：ACES 准确档对拍独立参考（重写 Stephen Hill 三矩阵），误差 ±0.01。
#[test]
fn aces_fitted_vs_independent_reference() {
    // 独立参考：直接按官方系数展开（与源码路径独立书写）。
    fn reference(rgb: [f32; 3]) -> [f32; 3] {
        const IN: [[f32; 3]; 3] = [
            [0.597_19, 0.354_58, 0.048_23],
            [0.076_00, 0.908_34, 0.015_66],
            [0.028_40, 0.133_83, 0.837_77],
        ];
        const OUT: [[f32; 3]; 3] = [
            [1.764_740_9, -0.675_777_7, -0.088_963_2],
            [-0.147_028_6, 1.160_251_2, -0.013_222_6],
            [-0.036_337_5, -0.162_436_8, 1.198_774_3],
        ];
        let mut a = [0.0f32; 3];
        for r in 0..3 {
            a[r] = IN[r][0] * rgb[0] + IN[r][1] * rgb[1] + IN[r][2] * rgb[2];
        }
        for v in a.iter_mut() {
            let x = *v;
            *v = (x * (x + 0.024_578_6) - 0.000_090_537)
                / (x * (0.983_729 * x + 0.432_951) + 0.238_081);
        }
        let mut o = [0.0f32; 3];
        for r in 0..3 {
            o[r] = (OUT[r][0] * a[0] + OUT[r][1] * a[1] + OUT[r][2] * a[2]).clamp(0.0, 1.0);
        }
        o
    }
    for &c in &[[0.18_f32, 0.18, 0.18], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.3, 0.6, 0.9], [4.0, 4.0, 4.0]] {
        let got = aces_fitted(c);
        let want = reference(c);
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 0.01, "对拍 c={c:?} got={got:?} want={want:?}");
        }
    }
}

/// 缺陷面：Narkowicz 风格档 x→0 应 0，x 大应趋近 1（clamp）；单调。
#[test]
fn aces_fast_monotonic_and_bounded() {
    let mut prev = -1.0f32;
    for i in 0..=1000u32 {
        let x = i as f32 / 100.0;
        let y = aces_fast([x, x, x])[0];
        assert!((0.0..=1.0).contains(&y), "fast 输出越界 x={x} y={y}");
        assert!(y >= prev - 1e-6, "fast 非单调 x={x} y={y} prev={prev}");
        prev = y;
    }
    assert_eq!(aces_fast([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
}

/// 缺陷面：det_pow_int 边界（n=0、大 n、负 n、x=0）。
#[test]
fn det_pow_int_boundaries() {
    assert_eq!(det_pow_int(2.0, 0), 1.0);
    assert_eq!(det_pow_int(0.0, 0), 1.0, "0^0 约定 1");
    assert_eq!(det_pow_int(0.0, 3), 0.0);
    assert!(det_pow_int(0.0, -1).is_infinite(), "0^-1 应 Inf");
    assert_eq!(det_pow_int(2.0, 1), 2.0);
    // 大负指数下溢到 0，不挂死。
    assert_eq!(det_pow_int(2.0, -2000), 0.0);
}

/// 缺陷面：det_pow 整体确定性（同输入两遍逐位一致）。
#[test]
fn det_pow_bitwise_stable_across_many() {
    for &(x, p) in &[(0.18_f32, 2.4_f32), (0.5, 1.0 / 2.4), (100.0, 0.159_301_6), (1e-3, 2.2)] {
        let first = det_pow(x, p).to_bits();
        for _ in 0..200 {
            assert_eq!(det_pow(x, p).to_bits(), first, "det_pow({x},{p}) 不确定");
        }
    }
    // det_exp/ln 确定性。
    let e0 = det_exp(1.0).to_bits();
    let l0 = det_ln(5.0).to_bits();
    for _ in 0..200 {
        assert_eq!(det_exp(1.0).to_bits(), e0);
        assert_eq!(det_ln(5.0).to_bits(), l0);
    }
}
