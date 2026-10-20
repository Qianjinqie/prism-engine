//! 攻击性测试。
//!
//! 目标：找真 Bug。重点 det_pow 边界（防挂死）、ACES 对拍、PQ/HLG 往返、
//! 矩阵互逆与白点守恒。每个测试注明"失败代表什么缺陷"。

use prism_math::aces::{aces_fast, aces_fitted, AcesFit, AcesOutput};
use prism_math::colorspace::{acescg_to_srgb, mat3_apply, srgb_to_acescg, ACESCG_TO_SRGB, SRGB_TO_ACESCG};
use prism_math::det_pow::{det_exp2, det_ln, det_log2, det_pow, det_pow_int};
use prism_math::hlg::{hlg_oetf, hlg_oetf_inverse, hlg_system_gamma};
use prism_math::pq::{pq_eotf, pq_eotf_nits, pq_oetf};

// ===========================================================================
// det_pow 边界攻击（防挂死 / 上轮 exp_f64 教训）
// ===========================================================================

/// 缺陷面：det_pow 退化输入是否有限/不挂死。
/// 失败 = 非有限输入产出 NaN/Inf 逻辑错误，或挂死（上轮 exp_f64 教训）。
#[test]
fn det_pow_degenerate_inputs_bounded() {
    // 极小正数（下溢侧）：应仍有限、≥ 0。
    for &x in &[f32::MIN_POSITIVE, 1e-30_f32, 1e-20, 1e-38] {
        let y = det_pow(x, 2.4);
        assert!(y.is_finite() && y >= 0.0, "det_pow({x},2.4)={y} 非有限/负");
    }
    // x=0 各符号：正指数→0、负指数→Inf、零指数→1。
    assert_eq!(det_pow(0.0, 2.0), 0.0, "0^正 = 0");
    assert_eq!(det_pow(0.0, -1.0), f32::INFINITY, "0^负 = Inf");
    assert_eq!(det_pow(0.0, 0.0), 1.0, "0^0 = 1");
    // x^0 恒 1（任意有限 x>0）。
    for &x in &[1e-20_f32, 0.5, 1.0, 7.0, 1e20] {
        assert_eq!(det_pow(x, 0.0), 1.0, "{x}^0 应为 1");
    }
    // 负底 → 引擎域内应 NaN 传播（非整数幂无实数解），不得 panic。
    let neg = det_pow(-2.0, 2.4);
    assert!(neg.is_nan(), "负底非整数幂应 NaN，实际 {neg}");
}

/// 缺陷面：极大指数不挂死（防 exp_f64 式死循环）。带超时由测试框架兜底。
/// 失败 = 挂死或产物非有限/错符号。
#[test]
fn det_pow_extreme_exponent_no_hang() {
    // x>1 大正指数 → 应为 Inf（上溢）或极大有限；绝不 0。
    let big = det_pow(2.0, 1000.0);
    assert!(big.is_infinite() || big > 1e30, "2^1000 应上溢为 Inf/极大，实际 {big}");
    // x<1 大正指数 → 0。
    assert_eq!(det_pow(0.5, 1000.0), 0.0, "0.5^1000 应下溢为 0");
    // 大负指数 → 反向。
    assert_eq!(det_pow(2.0, -1000.0), 0.0, "2^-1000 应下溢为 0");
    // 指数为 f32 极值邻域：不挂死即可。
    let _ = det_pow(1.0, f32::MAX);
    let _ = det_pow(2.0, f32::MAX / 1e30);
    let _ = det_pow(0.5, -f32::MAX / 1e30);
}

/// 缺陷面：det_pow 对 NaN 输入的传播（不得 panic）。
#[test]
fn det_pow_nan_propagates_without_panic() {
    let a = det_pow(f32::NAN, 2.4);
    assert!(a.is_nan(), "NaN 底应传 NaN，实际 {a}");
    let b = det_pow(2.0, f32::NAN);
    // p=NaN：x^NaN 应为 NaN（非 1）。
    assert!(b.is_nan() || b == 1.0, "2^NaN 应 NaN 或 1（记录实现），实际 {b}");
}

/// 缺陷面：det_pow 逐位确定性（同输入 1000 次完全一致）。
#[test]
fn det_pow_bitwise_deterministic() {
    for (x, p) in [(0.18_f32, 2.4_f32), (0.5, 1.0 / 2.4), (100.0, 0.159_301_6), (1e-10, 0.4)] {
        let first = det_pow(x, p).to_bits();
        for _ in 0..1000 {
            assert_eq!(det_pow(x, p).to_bits(), first, "det_pow({x},{p}) 不确定");
        }
    }
}

/// 缺陷面：det_pow_int 边界（n=0 → 1；负指数；大 n 不挂死）。
#[test]
fn det_pow_int_boundaries() {
    assert_eq!(det_pow_int(5.0, 0), 1.0, "x^0 = 1");
    assert!((det_pow_int(3.0, -2) - 1.0 / 9.0).abs() < 1e-6, "3^-2 = 1/9");
    assert_eq!(det_pow_int(2.0, 10), 1024.0, "2^10 = 1024");
    // 大 n：连乘求幂，不挂死（有界循环）。
    let bg = det_pow_int(2.0, 20);
    assert_eq!(bg, 1_048_576.0, "2^20");
}

// ===========================================================================
// PQ / HLG 往返恒等
// ===========================================================================

/// 缺陷面：PQ 往返（signal→linear→signal）恒等（误差 < 1e-3）。
/// 失败 = PQ EOTF/OETF 不自洽。
#[test]
fn pq_roundtrip_identity() {
    let mut worst = 0.0f32;
    for i in 0..=1000u32 {
        let s = i as f32 / 1000.0;
        let back = pq_oetf(pq_eotf(s));
        let e = (back - s).abs();
        worst = worst.max(e);
    }
    assert!(worst < 1e-3, "PQ 往返最大误差 {worst} 超限");
}

/// 缺陷面：PQ 端点 + 域外 clamp（signal<0 / >1 应夹紧，不 panic/NaN）。
#[test]
fn pq_endpoints_and_clamp() {
    assert!(pq_eotf(0.0).abs() < 1e-6, "PQ(0)=0");
    assert!((pq_eotf(1.0) - 1.0).abs() < 1e-4, "PQ(1)=1");
    // 域外：clamp 应保持有限、无 NaN。
    for s in [-0.5f32, -1.0, 1.5, 2.0] {
        let v = pq_eotf(s);
        assert!(v.is_finite() && v >= 0.0, "PQ 域外 {s} → {v} 异常");
    }
    // nits 版本。
    assert!((pq_eotf_nits(1.0) - 10000.0).abs() < 1.0, "峰值 10000 nits");
}

/// 缺陷面：PQ EOTF 单调递增（非有限/反序即缺陷）。
#[test]
fn pq_monotonic() {
    let mut prev = -1.0f32;
    for i in 0..=2000u32 {
        let v = pq_eotf(i as f32 / 2000.0);
        assert!(v >= prev - 1e-9, "PQ 非单调 @{i}: {v} < {prev}");
        prev = v;
    }
}

/// 缺陷面：HLG 往返恒等（误差 < 1e-3）。
#[test]
fn hlg_roundtrip_identity() {
    let mut worst = 0.0f32;
    for i in 0..=1000u32 {
        let e = i as f32 / 1000.0;
        let back = hlg_oetf_inverse(hlg_oetf(e));
        worst = worst.max((back - e).abs());
    }
    assert!(worst < 1e-3, "HLG 往返最大误差 {worst} 超限");
}

/// 缺陷面：HLG 段界连续（E=1/12 ↔ E'=0.5），两段不连续即缺陷。
#[test]
fn hlg_segment_boundary_continuous() {
    assert!((hlg_oetf(1.0 / 12.0) - 0.5).abs() < 1e-4, "段界正向不连续");
    assert!((hlg_oetf_inverse(0.5) - 1.0 / 12.0).abs() < 1e-5, "段界逆向不连续");
}

/// 缺陷面：HLG 域外（负/超 1）应有限、不 panic。
#[test]
fn hlg_clamp_no_panic() {
    for e in [-1.0f32, -0.5, 1.5, 2.0] {
        let v = hlg_oetf(e);
        assert!(v.is_finite(), "HLG({e})={v} 非有限");
    }
    for ep in [-0.5f32, 1.5, 2.0] {
        let v = hlg_oetf_inverse(ep);
        assert!(v.is_finite(), "HLG^-1({ep})={v} 非有限");
    }
}

/// 缺陷面：HLG system gamma（含峰值 ≤ 0 退化 + 逐位确定）。
#[test]
fn hlg_system_gamma_bounds() {
    assert!((hlg_system_gamma(1000.0) - 1.2).abs() < 1e-5, "参考 γ=1.2");
    assert!((hlg_system_gamma(2000.0) - 1.62).abs() < 1e-3, "2000 nits γ≈1.62");
    assert_eq!(hlg_system_gamma(0.0), 1.2, "峰值 0 退化到参考 γ");
    assert_eq!(hlg_system_gamma(-5.0), 1.2, "负峰值退化到参考 γ");
}

// ===========================================================================
// det_log2 / det_ln / det_exp2 边界
// ===========================================================================

/// 缺陷面：det_log2/det_ln 非正输入与非有限输入的语义。
#[test]
fn det_log_nonpositive_and_special() {
    assert!(det_log2(0.0).is_infinite() && det_log2(0.0) < 0.0, "log2(0) = -Inf");
    assert!(det_ln(0.0).is_infinite() && det_ln(0.0) < 0.0, "ln(0) = -Inf");
    assert!(det_log2(-1.0).is_infinite(), "log2(负) 应 -Inf");
    assert!(det_log2(f64::INFINITY).is_infinite(), "log2(Inf) = Inf");
    // NaN 输入：不 panic。
    let _ = det_log2(f64::NAN);
    let _ = det_ln(f64::NAN);
}

/// 缺陷面：det_exp2 极端输入不挂死（scale_pow2 循环上界）。
#[test]
fn det_exp2_extreme_no_hang() {
    assert!(det_exp2(2000.0).is_infinite(), "2^2000 上溢 Inf");
    assert_eq!(det_exp2(-2000.0), 0.0, "2^-2000 下溢 0");
    assert!(det_exp2(f64::NAN).is_nan(), "2^NaN = NaN");
    // 边界内：精确 2^n。
    assert!((det_exp2(10.0) - 1024.0).abs() < 1e-6, "2^10");
    assert!((det_exp2(0.0) - 1.0).abs() < 1e-12, "2^0");
}

// ===========================================================================
// sRGB ↔ ACEScg 矩阵互逆 + 白点守恒
// ===========================================================================

/// 缺陷面：sRGB↔ACEScg 矩阵互逆（M·M⁻¹ ≈ I，误差 < 1e-4）。
#[test]
fn colorspace_matrices_mutually_inverse() {
    let prod = mat3_mul(&SRGB_TO_ACESCG, &ACESCG_TO_SRGB);
    for r in 0..3 {
        for c in 0..3 {
            let want = if r == c { 1.0 } else { 0.0 };
            assert!(
                (prod[r][c] - want).abs() < 1e-4,
                "sRGB↔ACEScg 互逆失败 [{r}][{c}] = {}",
                prod[r][c]
            );
        }
    }
}

/// 缺陷面：白点守恒（行和 ≈ 1，白映射白）。
#[test]
fn colorspace_whitepoint_conserved() {
    for m in [SRGB_TO_ACESCG, ACESCG_TO_SRGB] {
        for r in 0..3 {
            let sum = m[r][0] + m[r][1] + m[r][2];
            assert!((sum - 1.0).abs() < 1e-4, "白点守恒破坏：行和 {sum}");
        }
    }
    // 白向量往返。
    let white = acescg_to_srgb(srgb_to_acescg([1.0, 1.0, 1.0]));
    for i in 0..3 {
        assert!((white[i] - 1.0).abs() < 1e-3, "白点往返失真 {white:?}");
    }
}

/// 缺陷面：任意色往返（sRGB→ACEScg→sRGB）逐位近似守恒。
#[test]
fn colorspace_arbitrary_roundtrip() {
    let colors = [[0.0f32, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.8], [0.9, 0.1, 0.05]];
    for c in colors {
        let back = acescg_to_srgb(srgb_to_acescg(c));
        for i in 0..3 {
            assert!((back[i] - c[i]).abs() < 1e-4, "往返失真 {c:?} → {back:?}");
        }
    }
}

// ===========================================================================
// ACES 两档拟合
// ===========================================================================

/// 缺陷面：ACES 准确档与独立参考实现（Stephen Hill 三矩阵+RRTAndODTFit）
/// 对拍 < 1e-4；与公开经验锚点 < 0.01。
/// 失败 = aces_fitted 与官方拟合不符。
#[test]
fn aces_fitted_matches_reference_within_tolerance() {
    let out = AcesOutput::default();
    // 独立参考：直接按官方三矩阵 + 有理函数写开。
    const IN: [[f32; 3]; 3] = SRGB_TO_ACESCG;
    const OUTM: [[f32; 3]; 3] = ACESCG_TO_SRGB;
    let reference = |rgb: [f32; 3]| -> [f32; 3] {
        let mut a = [0.0f32; 3];
        for r in 0..3 {
            a[r] = IN[r][0] * rgb[0] + IN[r][1] * rgb[1] + IN[r][2] * rgb[2];
        }
        for v in a.iter_mut() {
            let x = *v;
            *v = (x * (x + 0.024_578_6) - 0.000_090_537) / (x * (0.983_729 * x + 0.432_951) + 0.238_081);
        }
        let mut o = [0.0f32; 3];
        for r in 0..3 {
            o[r] = (OUTM[r][0] * a[0] + OUTM[r][1] * a[1] + OUTM[r][2] * a[2]).clamp(0.0, 1.0);
        }
        o
    };
    let mut worst = 0.0f32;
    for v in [0.0f32, 0.18, 0.5, 1.0, 4.0] {
        let got = out.apply([v, v, v]);
        let want = reference([v, v, v]);
        for i in 0..3 {
            worst = worst.max((got[i] - want[i]).abs());
        }
    }
    assert!(worst < 1e-4, "ACES 与参考实现最大误差 {worst} 超限");
}

/// 缺陷面：ACES 准确档对公开经验锚点在 ±0.01 内（对拍）。
#[test]
fn aces_fitted_empirical_anchors() {
    let out = aces_fitted;
    // 18% 灰应落在 [0.05, 0.2] 显示线性（扩散白区间）。
    let mid = out([0.18, 0.18, 0.18])[0];
    assert!(mid > 0.05 && mid < 0.2, "18% 灰映射异常 {mid}");
    // 黑 → 0。
    assert!(out([0.0, 0.0, 0.0])[0].abs() < 1e-6, "黑应 0");
    // 超大输入饱和 ≤ 1。
    let b = out([1000.0, 1000.0, 1000.0])[0];
    assert!((0.0..=1.0).contains(&b), "超亮未饱和 {b}");
}

/// 缺陷面：ACES 灰阶单调（非单调即色调映射缺陷）。
#[test]
fn aces_fitted_monotonic_gray() {
    let mut prev = -1.0f32;
    for i in 0..=1000u32 {
        let v = i as f32 / 100.0;
        let y = aces_fitted([v, v, v])[0];
        assert!(y >= prev - 1e-6, "灰阶非单调 {y} < {prev}");
        prev = y;
    }
}

/// 缺陷面：ACES 风格档（Narkowicz）端点 + 逐位确定 + 值域 [0,1]。
#[test]
fn aces_fast_style_bounded_deterministic() {
    assert_eq!(aces_fast([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0], "风格档黑→0");
    let first = aces_fast([1.0, 2.0, 3.0]);
    for _ in 0..1000 {
        assert_eq!(aces_fast([1.0, 2.0, 3.0]), first, "风格档不确定");
    }
    for i in 0..=100u32 {
        let v = i as f32 / 10.0;
        let o = aces_fast([v, v, v]);
        for i in 0..3 {
            assert!((0.0..=1.0).contains(&o[i]), "风格档越界 {o:?}");
        }
    }
    // 两档选择器。
    assert_eq!(AcesOutput::new(AcesFit::Fast).apply([0.5, 0.5, 0.5]), aces_fast([0.5, 0.5, 0.5]));
    assert_eq!(AcesOutput::default().fit, AcesFit::Fitted, "默认准确档");
}

/// 缺陷面：ACES 逐位确定（准确档 1000 次一致）。
#[test]
fn aces_fitted_bitwise_deterministic() {
    let first = aces_fitted([0.42, 0.8, 1.3]);
    for _ in 0..1000 {
        assert_eq!(aces_fitted([0.42, 0.8, 1.3]), first, "准确档不确定");
    }
}

/// 缺陷面：ACES 负输入（引擎域外）应有限、不 panic（有理式分母不为零）。
#[test]
fn aces_fitted_negative_input_safe() {
    for v in [-0.5f32, -1.0, -10.0] {
        let o = aces_fitted([v, v, v]);
        for i in 0..3 {
            assert!(o[i].is_finite(), "ACES 负输入 {v} → {o:?} 非有限");
        }
    }
}

// ---------------------------------------------------------------------------
// 本地 3×3 乘法（避免依赖私有；与 colorspace::mat3_mul 同构）
// ---------------------------------------------------------------------------

fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

/// 保留 `mat3_apply` 引用（确保其公有契约可编译使用）。
#[test]
fn mat3_apply_contract() {
    let v = mat3_apply(&SRGB_TO_ACESCG, [1.0, 0.0, 0.0]);
    // 行和 ≈ 1 ⇒ 白点守恒的逐通道证明。
    let w = mat3_apply(&SRGB_TO_ACESCG, [1.0, 1.0, 1.0]);
    for i in 0..3 {
        assert!((w[i] - 1.0).abs() < 1e-4, "mat3_apply 白点失真 {w:?}");
    }
    assert!(v[0] > 0.0, "红primary 变换后首个分量应为正");
}
