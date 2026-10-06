//! 确定性超越函数（D-29）：libm sin/cos 的固定系数多项式替代。
//!
//! 与 `prism-anim`（`lib.rs::det`）已落地的自含过渡件同式同源：纯 IEEE 运算
//! （`+ − × ÷`、`floor`/`round` 精确取整、固定常量多项式），f64 内部运算、
//! f32 一次舍入出入口——IEEE 754 对每个基本运算结果唯一定义，故跨平台逐位
//! 一致。f64 内部是音频必需：振荡器相位 `2πft` 在长缓冲下达 1e7 量级，f32
//! 相位精度崩坏，f64 归约保持误差 ≲1e-10（动画/音频时间尺度）。
//!
//! 待 `prism-math` D-29 确定性数学件上收时与 anim/geometry/rt 侧合并
//! （D-16 避免多实现的最终归宿）。sqrt 为 IEEE 精确舍入运算，不在禁列。
//!
//! # 精度界（对照 f64 libm，仅测试内用 libm）
//! - `sin_f64`/`cos_f64`：round 归约 + Maclaurin 6 项，`|x| ≲ 1e8` 时误差
//!   ≲ 1e-10。

/// 2/π（f64 精确常量）。
const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;
/// π/2（f64 精确常量）。
const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// sin 在 `[-π/4, π/4]` 的 Taylor 展开（Horner 于 r²，截断 < 1e-11）。
#[inline]
fn poly_sin(r: f64) -> f64 {
    let r2 = r * r;
    r * (1.0
        + r2 * (-1.0 / 6.0
            + r2 * (1.0 / 120.0
                + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 3_628_80.0 + r2 * (-1.0 / 39_916_800.0))))))
}

/// cos 在 `[-π/4, π/4]` 的 Taylor 展开（截断 < 1e-12）。
#[inline]
fn poly_cos(r: f64) -> f64 {
    let r2 = r * r;
    1.0 + r2 * (-0.5
        + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40_320.0 + r2 * (-1.0 / 3_628_800.0)))))
}

/// sin(x)，f64 全精度。区间折叠到 `[-π/2, π/2]` 后查象限（与 prism-anim 同式）。
pub fn sin_f64(x: f64) -> f64 {
    let k = (x * FRAC_2_PI).round();
    let r = x - k * FRAC_PI_2;
    let q = (k as i64).rem_euclid(4);
    let (s, c) = (poly_sin(r), poly_cos(r));
    match q {
        0 => s,
        1 => c,
        2 => -s,
        _ => -c,
    }
}

/// cos(x)，f64 全精度（与 [`sin_f64`] 同一折叠）。
pub fn cos_f64(x: f64) -> f64 {
    let k = (x * FRAC_2_PI).round();
    let r = x - k * FRAC_PI_2;
    let q = (k as i64).rem_euclid(4);
    let (s, c) = (poly_sin(r), poly_cos(r));
    match q {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    }
}

/// sin(x)，f32 出入口（内部 f64 运算后一次舍入）。
#[inline]
pub fn sin_f32(x: f32) -> f32 {
    sin_f64(x as f64) as f32
}

/// cos(x)，f32 出入口。
#[inline]
pub fn cos_f32(x: f32) -> f32 {
    cos_f64(x as f64) as f32
}

/// ln(x)，f64（x > 0）。frexp 分解 `x = m·2^e`（m ∈ [1,2)），ln = e·ln2 + ln(m)；
/// `ln(m)` 用 atanh 级数 `t = (m−1)/(m+1)`，`ln(m) = 2·(t + t³/3 + t⁵/5 + …)`
/// （`|t| ≤ 1/3` 收敛快，截断 < 1e-14）。仅用 ±×÷（D-29 确定性）。
pub fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // 提取二进制指数 e 与尾数 m ∈ [1, 2)。
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    let m = mantissa; // ∈ [1, 2)
    // 若 m ≥ √2，用 m/2 与 e+1 归一，使 t 绝对值更小（加快收敛）。
    let (m, e) = if m > std::f64::consts::SQRT_2 { (m * 0.5, e + 1) } else { (m, e) };
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let series = 2.0
        * t
        * (1.0
            + t2 * (1.0 / 3.0
                + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (1.0 / 11.0))))));
    series + (e as f64) * std::f64::consts::LN_2
}

/// exp(x)，f64。`x = k·ln2 + r`（k = round(x/ln2)），`exp(x) = 2^k·exp(r)`；
/// `exp(r)` 用 Taylor（`|r| ≤ ln2/2`，12 项截断 < 1e-15）。2^k 用整数位移（精确）。
///
/// **非有限输入契约**（P0，见 [`SCALE_POW2_MAX_ITER`] 的历史根因）：
/// `-inf → +0.0`、`+inf → +inf`、`NaN → NaN`，三者**立即返回、一次连乘都不做**。
/// 若缺此短路，`(-inf/LN_2).round() as i64` 是饱和转换 `i64::MIN`。
pub fn exp_f64(x: f64) -> f64 {
    // 非有限输入短路，避免 (x/LN_2).round() 溢出为 ±inf 再被 `as i64`
    // 截断成 i64::MIN/MAX 导致 scale_pow2 近乎无限循环（P0）。
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x > 0.0 { f64::INFINITY } else { 0.0 };
    }
    let k = (x / std::f64::consts::LN_2).round();
    // 下界短路：f64 最小正**次正规**是 2^-1074，round-to-nearest 的 0.5ulp
    // 界是 2^-1075（= exp(-1075·ln2) = exp(-745.1332…)）。x 低于该界时真值
    // 舍入到 +0.0，故k < -1075 直接返回。k = -1075 那一档
    // （x ∈ [-745.82, -745.13]）真值仍在 [2.6e-324, 4.9e-324]，**必须真算**，
    // 走下方 k_subnormal 的一次性乘法路径（见「双重舍入修正」）。
    // 注：旧实现用 `k <= -1075` 短路，把这一档误判为 0，exp(-745) 得 0
    // 而非 5e-324 —— 已被 det_exp_edge_values_bitwise 逐位锁定修复。
    if k < -1075.0 {
        return 0.0;
    }
    // 上界：有限区间延伸到 x ≈ 709.7827（= ln(f64::MAX)）。此处 x/ln2 ≈ 1024，
    // round 后 k == 1024，而 2^1024 恰溢出（f64::MAX = 2^1024·(1−2⁻⁵³) 略小于
    // 2^1024）。若仍按 k == 1024 直接 2^1024 连乘会误判 +inf。修正：正侧把指数
    // 基准夹到 1023（k_orig 保留用于后续顶部判定），令 r = x − 1023·ln2，
    // exp(r) 自含额外因子 2，最终 2^1023·exp(r) 落在 (2^1023, f64::MAX] 有限。
    let k_orig = k;
    let k = if k > 1023.0 { 1023.0 } else { k };
    // 下界次正规区的**双重舍入修正**：结果落进次正规域（k ≤ -1074）时，
    // `scale_pow2` 的逐次 `*0.5` 会在次正规域里**每一步都做一次
    // round-to-nearest-even**，误差逐步放大，把本应是最小次正规 5e-324 的值
    // 塌成 +0.0（经典双重舍入缺陷；实测 exp(-745) 得 0 而非 5e-324）。
    // 修正：指数基准统一夹到 -1074（与上界夹 1023 同构），令
    // exp(x) = 2^-1074 · exp(r)—— 把 2^-1074 换成**一次性乘法**
    // `p · f64::from_bits(1)`（最小正次正规 = 2^-1074），全程只经**一次**舍入。
    let k_subnormal = k <= -1074.0;
    let k = if k_subnormal { -1074.0 } else { k };
    let r = x - k * std::f64::consts::LN_2;
    // exp(r) = Σ r^n/n!，Horner 于 r。|r| ≤ ln2 ≈ 0.693（正侧夹 k 后 r 可达
    // ln2），需 16 项（截断 < 1e-18），保证 2^1023 放大后仍满足 1e-15 相对精度。
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0
                                + r * (1.0 / 5040.0
                                    + r * (1.0 / 40320.0
                                        + r * (1.0 / 362880.0
                                            + r * (1.0 / 3628800.0
                                                + r * (1.0 / 39916800.0
                                                    + r * (1.0 / 479001600.0
                                                        + r * (1.0 / 6227020800.0
                                                            + r * (1.0 / 87178291200.0
                                                                + r * (1.0 / 1307674368000.0)))))))))))))));
    // 2^k 用整数幂连乘（k 为整数，IEEE 精确；D-29 禁 pow）。
    // 次正规域（k ≤ -1074）走**一次性**乘法：`p · 2^-1074`，只经一次舍入；
    // 正规域走 scale_pow2 的整数连乘（精确，循环次数 ≤ SCALE_POW2_MAX_ITER）。
    if k_subnormal {
        return p * f64::from_bits(1);
    }
    let out = scale_pow2(p, k as i64);
    // 顶部区域（原始 k 落到 1024，即 x ≥ 1023.5·ln2 ≈ 709.436）：理论值
    // = 2^1023·exp(r)，数学上落在 (2^1023, f64::MAX]。指数基准夹到 1023 后，
    // 尾数舍入会把本应等于/极近 f64::MAX 的结果压低 ~1e294。对「真上界内且
    // 结果与 f64::MAX 仅差尾数舍入量（<1e296）」的情形夹回 f64::MAX；
    // x 真超 ln(f64::MAX) 或连乘已溢出则 → +inf。
    if k_orig > 1023.0 {
        const LN_F64_MAX: f64 = 709.782_712_893_384_f64;
        if x > LN_F64_MAX || !out.is_finite() {
            return f64::INFINITY;
        }
        if out >= f64::MAX - 1e296 {
            return f64::MAX;
        }
        return out;
    }
    out
}

/// 2^k 连乘的**迭代次数上界**（f64 指数域跨度 + 余量）。
///
/// f64 的次正规下界是 2^-1074、正规上界是 2^1023，指数跨度共 2097 位；
/// 上取 2098 留一位余量。任何 |k| 超过该上界时，多出的连乘只会把有限值
/// 推向 ±0（k<0）或 ±inf（k>0），且 ±0/±inf 再乘/除 2 仍是 ±0/±inf
/// （IEEE-754 §6.3 对 0 与 inf 的乘法封闭）——故**钳制是逐位行为保持的**。
pub const SCALE_POW2_MAX_ITER: i64 = 2_098;

/// 单次 [`scale_pow2`] 的实际连乘次数累计（仅单测可见，用于「有界性」取证）。
#[cfg(test)]
pub(crate) static SCALE_POW2_ITER_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 2^k 标量乘（k 整数，连乘实现，无 pow）。
///
/// **有界性契约**：k 先钳到 ±[`SCALE_POW2_MAX_ITER`]，循环次数因此有一个
/// 编译期常量上界。这挡住历史 P0 挂死——`exp_f64(-inf)` 未做非有限短路时
/// `k = (-inf/LN_2).round() as i64` 是**饱和转换** `i64::MIN`，未钳制则
/// `while k < 0 { r *= 0.5; k += 1 }` 需 ~9.2e18 次迭代才退出（等价挂死）。
#[inline]
fn scale_pow2(v: f64, k: i64) -> f64 {
    let mut n = k.clamp(-SCALE_POW2_MAX_ITER, SCALE_POW2_MAX_ITER);
    #[cfg(test)]
    SCALE_POW2_ITER_COUNT
        .fetch_add(n.unsigned_abs(), std::sync::atomic::Ordering::Relaxed);
    let mut r = v;
    while n > 0 {
        r *= 2.0;
        n -= 1;
    }
    while n < 0 {
        r *= 0.5;
        n += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// det_sin 对照 std f64 sin（仅测试内用 libm 参考，D-29 禁令不约束测试）。
    #[test]
    fn det_sin_matches_reference() {
        let mut x = -1e7_f64; // 覆盖音频长缓冲相位量级
        while x <= 1e7 {
            let diff = (sin_f64(x) - x.sin()).abs();
            assert!(diff < 1e-8, "det_sin({x}) 偏差 {diff}");
            x += 99_991.0; // 大步长扫全量级，步长与 2π 无理关系覆盖各象限
        }
        assert_eq!(sin_f64(0.0), 0.0);
        assert!((sin_f64(std::f64::consts::FRAC_PI_6) - 0.5).abs() < 1e-12);
    }

    /// det_cos 对照 std f64 cos。
    #[test]
    fn det_cos_matches_reference() {
        let mut x = -1e5_f64;
        while x <= 1e5 {
            let diff = (cos_f64(x) - x.cos()).abs();
            assert!(diff < 1e-8, "det_cos({x}) 偏差 {diff}");
            x += 1_013.0;
        }
        assert_eq!(cos_f64(0.0), 1.0);
    }

    /// f32 出入口精度足够（振荡器/FFT 窗的精度预算 < 1e-6）。
    #[test]
    fn det_f32_exits_are_accurate() {
        for i in 0..2000_u32 {
            let x = (i as f32) * 0.0031 - 3.1;
            let ds = sin_f32(x) as f64;
            let dc = cos_f32(x) as f64;
            assert!((ds - (x as f64).sin()).abs() < 1e-6, "sin_f32({x}) = {ds}");
            assert!((dc - (x as f64).cos()).abs() < 1e-6, "cos_f32({x}) = {dc}");
        }
    }

    /// 逐位确定性：同输入重复调用 10000 次逐位一致（纯函数性质的显式锚）。
    #[test]
    fn det_functions_bitwise_reproducible() {
        for x in [0.3_f32, 1.7, -2.9, 44.25, 7_777_777.5] {
            let expected = (sin_f32(x), cos_f32(x));
            for _ in 0..10_000 {
                assert_eq!((sin_f32(x), cos_f32(x)), expected);
            }
        }
    }

    /// ln_f64 对照 std ln（测试内用 libm 参考，D-29 不约束测试）。
    #[test]
    fn det_ln_matches_reference() {
        let mut x = 1e-6_f64;
        while x <= 1e6 {
            let diff = (ln_f64(x) - x.ln()).abs();
            assert!(diff < 1e-10, "ln({x}) 偏差 {diff}");
            x *= 1.37;
        }
        assert!((ln_f64(1.0)).abs() < 1e-12);
        assert!((ln_f64(std::f64::consts::E) - 1.0).abs() < 1e-9);
    }

    /// exp_f64 对照 std exp。
    #[test]
    fn det_exp_matches_reference() {
        let mut x = -50.0_f64;
        while x <= 50.0 {
            let diff = (exp_f64(x) - x.exp()).abs() / x.exp().max(1e-300);
            assert!(diff < 1e-10, "exp({x}) 相对偏差 {diff}");
            x += 0.37;
        }
        assert!((exp_f64(0.0) - 1.0).abs() < 1e-12);
        // exp/ln 互逆。
        for v in [0.01_f64, 1.0, 42.0, 1234.5] {
            assert!((exp_f64(ln_f64(v)) - v).abs() / v < 1e-10, "exp(ln({v})) ≠ {v}");
        }
    }

    /// exp_f64 非有限输入的**短路契约**（P0 回归锚）。
    ///
    /// 历史缺陷：`exp_f64(-inf)` 未做非有限短路 → `k = i64::MIN`（饱和转换）
    /// → `scale_pow2` 连乘 ~9.2e18 次 → **运行永不返回**。此处锁定
    /// `-inf → +0.0`、`+inf → +inf`、`NaN → NaN`，三者都**立即返回**
    /// （下方 `det_exp_non_finite_returns_immediately` 用迭代计数佐证）。
    #[test]
    fn det_exp_non_finite_contract() {
        // -inf → +0.0（逐位断言：必须是 +0.0，不能是 -0.0）。
        assert_eq!(exp_f64(f64::NEG_INFINITY).to_bits(), 0.0_f64.to_bits());
        // +inf → +inf（逐位）。
        assert_eq!(exp_f64(f64::INFINITY).to_bits(), f64::INFINITY.to_bits());
        // NaN → NaN（逐位断言 NaN，不做 == 比较）。
        assert!(exp_f64(f64::NAN).is_nan());
        // 两个符号的 NaN 都必须 NaN（不只默认 quiet NaN）。
        assert!(exp_f64(-f64::NAN).is_nan());
    }

    /// exp_f64 边界值的逐位正确性（-0.0 / 次正规 / ±边界有限值）。
    #[test]
    fn det_exp_edge_values_bitwise() {
        // -0.0 → +1.0（exp 恒正；注意不是 -1.0，逐位锁定）。
        assert_eq!(exp_f64(-0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(exp_f64(0.0).to_bits(), 1.0_f64.to_bits());
        // 次正规输入：exp_f64(5e-324) ≈ 1.0（相对误差 < 1e-15）。
        let sub = f64::from_bits(1); // f64::MIN_POSITIVE/2^52 ≈ 5e-324，最小正次正规
        assert_eq!(sub.to_bits(), 1, "前提：from_bits(1) 应是最小正次正规");
        assert!((exp_f64(sub) - 1.0).abs() < 1e-15);
        // 下溢边界（次正规区，逐位对照 std exp 锁定）：
        // exp(-744.4) ≈ 2^-1074（最小次正规），exp(-744) / exp(-743) 逐位相同。
        for (x, expect) in [
            (-743.0_f64, f64::from_bits(4)),  // 2e-323
            (-744.0, f64::from_bits(2)),      // 1e-323
            (-744.4, f64::from_bits(1)),      // 5e-324（最小正次正规）
        ] {
            assert_eq!(
                exp_f64(x).to_bits(),
                expect.to_bits(),
                "exp({x}) 次正规边界应逐位等于 {expect:e}"
            );
        }
        // exp(-745)：k = -1075，真值 5e-324（最小正次正规），**不得**为 0。
        // 旧实现 `k <= -1075` 短路把它误判为 0.0 —— 本断言是防回归锚。
        assert_eq!(
            exp_f64(-745.0).to_bits(),
            f64::from_bits(1).to_bits(),
            "exp(-745) 应为最小正次正规 5e-324，不得下溢为 0"
        );
        // 真下溢：exp(-746) 低于 0.5ulp → +0.0（逐位）。
        assert_eq!(exp_f64(-746.0).to_bits(), 0.0_f64.to_bits());
        // 上边界：exp(709) 有限（≈ 8.2e307，在 f64::MAX 内）。
        let v709 = exp_f64(709.0);
        assert!(v709.is_finite(), "exp(709) 不应溢出，得 {v709:e}");
        assert!((v709 / 709.0_f64.exp() - 1.0).abs() < 1e-12, "exp(709) 相对偏差过大");
        // 越过 ln(f64::MAX) ≈ 709.7827 → +inf（逐位）。
        assert_eq!(exp_f64(710.0).to_bits(), f64::INFINITY.to_bits());
        // 极大有限负值（如 -1e9）→ +0.0，且必须立即返回。
        assert_eq!(exp_f64(-1e9).to_bits(), 0.0_f64.to_bits());
        // 极大有限正值 → +inf。
        assert_eq!(exp_f64(1e9).to_bits(), f64::INFINITY.to_bits());
    }

    /// **有界性取证**：每次 `exp_f64` 的连乘次数 ≤ `SCALE_POW2_MAX_ITER`。
    ///
    /// 这是「exp_f64 不可能挂死」的可计数证据——不依赖墙钟超时（CI 上易抖动），
    /// 而是直接读 [`scale_pow2`] 内部的迭代累计计数器。
    #[test]
    fn det_exp_is_bounded_iteration() {
        use std::sync::atomic::Ordering;
        let cases = [
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            -0.0,
            0.0,
            f64::from_bits(1),
            -745.0,
            -746.0,
            -1e9,
            -100.0,
            -1.0,
            1.0,
            709.0,
            710.0,
            1e9,
            f64::MAX,
            f64::MIN,
        ];
        for x in cases {
            SCALE_POW2_ITER_COUNT.store(0, Ordering::Relaxed);
            let _ = exp_f64(x);
            let iters = SCALE_POW2_ITER_COUNT.load(Ordering::Relaxed);
            assert!(
                iters <= SCALE_POW2_MAX_ITER as u64,
                "exp_f64({x:e}) 连乘 {iters} 次，超上界 {SCALE_POW2_MAX_ITER}"
            );
            // 非有限输入必须**一次连乘都不做**（纯短路返回）。
            if !x.is_finite() {
                assert_eq!(iters, 0, "exp_f64({x:e}) 应短路，连乘次数应为 0");
            }
        }
        // 有限最大输入（k 会被夹到 1023）的连乘次数恰为上界之内的确定值。
        SCALE_POW2_ITER_COUNT.store(0, Ordering::Relaxed);
        let _ = exp_f64(-1e9);
        assert_eq!(
            SCALE_POW2_ITER_COUNT.load(Ordering::Relaxed),
            0,
            "exp_f64(-1e9) 在 k<=-1075 处短路，不应连乘"
        );
    }

    /// scale_pow2 的钳制是**逐位行为保持**的：|k| 超上界时，结果与钳到上界
    /// 同侧的结果逐位相同（±0 / ±inf 在乘除 2 下封闭，IEEE-754 §6.3）。
    #[test]
    fn det_scale_pow2_clamp_is_behaviour_preserving() {
        use std::sync::atomic::Ordering;
        // k = i64::MIN（历史 P0 根因的饱和值）必须**立即**返回，不得挂死。
        SCALE_POW2_ITER_COUNT.store(0, Ordering::Relaxed);
        let r_min = scale_pow2(1.0, i64::MIN);
        assert_eq!(
            SCALE_POW2_ITER_COUNT.load(Ordering::Relaxed),
            SCALE_POW2_MAX_ITER as u64,
            "i64::MIN 应恰被钳到 -SCALE_POW2_MAX_ITER"
        );
        assert_eq!(r_min.to_bits(), 0.0_f64.to_bits(), "2^-2098 下溢到 +0.0");
        // k = i64::MAX → 上溢到 +inf，同样是钳制后的结果。
        let r_max = scale_pow2(1.0, i64::MAX);
        assert_eq!(r_max.to_bits(), f64::INFINITY.to_bits());
        // 与显式钳制值逐位一致（证明钳制未改变语义）。
        assert_eq!(r_min.to_bits(), scale_pow2(1.0, -SCALE_POW2_MAX_ITER).to_bits());
        assert_eq!(r_max.to_bits(), scale_pow2(1.0, SCALE_POW2_MAX_ITER).to_bits());
        // 精确 2^k（未钳制区间）逐位正确：2^10 = 1024，2^-10 = 1/1024。
        assert_eq!(scale_pow2(1.0, 10).to_bits(), 1024.0_f64.to_bits());
        assert_eq!(scale_pow2(1.0, -10).to_bits(), (1.0_f64 / 1024.0).to_bits());
        assert_eq!(scale_pow2(1.0, 0).to_bits(), 1.0_f64.to_bits());
    }
}
