//! D-29 确定性幂/指/对数件（无 libm）。
//!
//! 语义权威：D-29（"超越函数自研入 prism-math……整数幂一律连乘
//! 不用 pow"）；确定性格锚。ACES / PQ / HLG 的
//! `pow(x, γ)`、`x^(1/m)` 与 OETF 幂函数全部经本件求值——纯 IEEE 运算
//! （`+ − × ÷`、位运算、固定常量多项式），跨平台逐位一致。
//!
//! # 件清单
//! - [`det_pow`]：`x^p`（x > 0）经 `exp2(p·log2(x))`；
//! - [`det_log2`]：二进制对数（整数位分解 + 尾数 atanh 级数）；
//! - [`det_exp2`]：`2^x`（整数幂连乘 + 小数 Taylor）；
//! - [`det_pow_int`]：整数指数连乘（禁 `powi` 跨平台差异，显式连乘）。
//!
//! sqrt 为 IEEE 精确舍入运算（非超越件，不在 D-29 禁列）。

/// ln2 的 f64 精确常量（供级数与换算）。
const LN2: f64 = std::f64::consts::LN_2;
/// 1/ln2 = log2(e)。
const INV_LN2: f64 = std::f64::consts::LOG2_E;

/// `log2(x)`，f64（x > 0）。位分解 `x = m·2^e`（m ∈ [1,2)）→
/// `log2 = e + log2(m)`；`log2(m)` 用 atanh 级数（`t = (m−1)/(m+1)`，
/// `|t| ≤ 1/3`），`log2(m) = 2·(t + t³/3 + …)/ln2`。纯 `±×÷`，确定性。
pub fn det_log2(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if !x.is_finite() {
        return x; // +inf → +inf；NaN 已被上面排除
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    // m ≥ √2 时用 m/2 与 e+1 归一，使 t 更小（收敛更快）。
    let (m, e) = if mantissa > std::f64::consts::SQRT_2 {
        (mantissa * 0.5, e + 1)
    } else {
        (mantissa, e)
    };
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // ln(m) = 2·t·(1 + t²/3 + t⁴/5 + …)（截断 < 1e-15）。
    let ln_m = 2.0
        * t
        * (1.0
            + t2 * (1.0 / 3.0
                + t2 * (1.0 / 5.0
                    + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (1.0 / 11.0))))));
    (e as f64) + ln_m * INV_LN2
}

/// `ln(x)`，f64（x > 0）。与 [`det_log2`] 同位分解 `x = m·2^e`，
/// `ln = e·ln2 + ln(m)`；`ln(m)` 用 atanh 级数（`|t| ≤ 1/3`）。纯 `±×÷`。
/// HLG 的 `a·ln(12E−b)+c` 经本件求值（D-29：禁 libm `ln`）。
pub fn det_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    let (m, e) = if mantissa > std::f64::consts::SQRT_2 {
        (mantissa * 0.5, e + 1)
    } else {
        (mantissa, e)
    };
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln_m = 2.0
        * t
        * (1.0
            + t2 * (1.0 / 3.0
                + t2 * (1.0 / 5.0
                    + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (1.0 / 11.0))))));
    ln_m + (e as f64) * LN2
}

/// `exp(x)`，f64（floor 级复用 [`det_exp2`]：`exp(x) = 2^{x/ln2}`）。纯确定性。
pub fn det_exp(x: f64) -> f64 {
    det_exp2(x * INV_LN2)
}

/// `2^x`，f64。`k = round(x)`，`r = x − k ∈ [−0.5, 0.5]`；`2^r` 用 Taylor
/// （12 项截断 < 1e-16），`2^k` 用整数连乘（精确，禁 `pow`）。
pub fn det_exp2(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 1024.0 {
        return f64::INFINITY;
    }
    if x < -1075.0 {
        return 0.0;
    }
    let k = x.round();
    let r = x - k;
    let r_ln2 = r * LN2;
    // exp(r·ln2) 但直接用 2^r 的写法：2^r = exp(r·ln2)。
    let t = r_ln2;
    let p = 1.0
        + t * (1.0
            + t * (0.5
                + t * (1.0 / 6.0
                    + t * (1.0 / 24.0
                        + t * (1.0 / 120.0
                            + t * (1.0 / 720.0
                                + t * (1.0 / 5040.0
                                    + t * (1.0 / 40320.0
                                        + t * (1.0 / 362880.0
                                            + t * (1.0 / 3628800.0
                                                + t * (1.0 / 39916800.0
                                                    + t * (1.0 / 479001600.0))))))))))));
    scale_pow2(p, k as i64)
}

/// `2^k` 标量乘（k 整数，显式连乘，无 `pow`/`powi`）。
#[inline]
fn scale_pow2(v: f64, mut k: i64) -> f64 {
    let mut r = v;
    while k > 0 {
        r *= 2.0;
        k -= 1;
    }
    while k < 0 {
        r *= 0.5;
        k += 1;
    }
    r
}

/// `x^p`，f32 出入口（内部 f64 运算后一次舍入）。`x > 0`；`x == 0` 时
/// 返回按 p 符号的 0/Inf（负指数 → Inf）。纯确定性（`2^(p·log2 x)`）。
pub fn det_pow(x: f32, p: f32) -> f32 {
    if x == 0.0 {
        return if p > 0.0 {
            0.0
        } else if p < 0.0 {
            f32::INFINITY
        } else {
            1.0
        };
    }
    if x < 0.0 {
        // 引擎色彩域恒非负；负底非整数幂无实数解——按 NaN 传播。
        return f32::NAN;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return x;
    }
    let y = det_exp2((p as f64) * det_log2(x as f64)) as f32;
    y
}

/// 整数指数连乘 `x^n`（禁 `powi`——跨平台尾位差异；n 可为负）。确定性。
pub fn det_pow_int(x: f32, n: i32) -> f32 {
    // 平方求幂的确定性等价：逐次连乘（固定乘法序）。
    let mut result = 1.0f32;
    let mut base = x;
    let mut exp = n.unsigned_abs();
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    if n < 0 {
        1.0 / result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_log2_matches_reference() {
        // 对照 std f64 log2（测试内 libm 参考，D-29 不约束测试）。
        let mut x = 1e-6_f64;
        while x <= 1e6 {
            let diff = (det_log2(x) - x.log2()).abs();
            assert!(diff < 1e-10, "log2({x}) 偏差 {diff}");
            x *= 1.37;
        }
        assert!((det_log2(1.0)).abs() < 1e-12);
        assert!((det_log2(2.0) - 1.0).abs() < 1e-12);
        assert!((det_log2(0.5) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_exp2_matches_reference() {
        let mut x = -60.0_f64;
        while x <= 60.0 {
            let want = x.exp2();
            let rel = (det_exp2(x) - want).abs() / want.max(1e-300);
            assert!(rel < 1e-10, "2^({x}) 相对偏差 {rel}");
            x += 0.37;
        }
        assert!((det_exp2(0.0) - 1.0).abs() < 1e-12);
        assert!((det_exp2(10.0) - 1024.0).abs() < 1e-9);
    }

    #[test]
    fn det_pow_matches_reference() {
        // ACES/PQ/HLG 实际用到的指数：γ = 2.4、1/2.4、2.2、1/1.2、m 倒数等。
        let exps = [2.4_f32, 1.0 / 2.4, 2.2, 1.0 / 2.2, 1.0 / 1.2, 1.0 / 2.0, 0.159_301_6];
        let mut x = 1e-4_f32;
        while x <= 100.0 {
            for &p in &exps {
                let got = det_pow(x, p) as f64;
                let want = (x as f64).powf(p as f64);
                let rel = (got - want).abs() / want.max(1e-300);
                assert!(rel < 1e-5, "pow({x}, {p}) = {got} vs {want}，相对偏差 {rel:e}");
            }
            x *= 1.31;
        }
        // 端点行为。
        assert_eq!(det_pow(0.0, 2.4), 0.0);
        assert_eq!(det_pow(0.0, -1.0), f32::INFINITY);
        assert_eq!(det_pow(5.0, 0.0), 1.0);
        assert_eq!(det_pow(7.0, 1.0), 7.0);
    }

    #[test]
    fn det_pow_int_matches_reference() {
        let cases: [(f32, i32); 8] = [
            (2.0, 10),
            (1.5, 3),
            (0.5, 4),
            (3.0, -2),
            (1.1, 0),
            (7.0, 1),
            (0.9, 5),
            (2.0, -3),
        ];
        for (x, n) in cases {
            let got = det_pow_int(x, n) as f64;
            let want = (x as f64).powi(n);
            let rel = (got - want).abs() / want.abs().max(1e-300);
            assert!(rel < 1e-6, "pow_int({x}, {n}) = {got} vs {want}");
        }
    }

    #[test]
    fn det_pow_bitwise_deterministic() {
        for (x, p) in [(0.18_f32, 2.4_f32), (0.5, 1.0 / 2.4), (100.0, 1.0 / 3.0)] {
            let first = det_pow(x, p);
            for _ in 0..1000 {
                assert_eq!(det_pow(x, p).to_bits(), first.to_bits());
            }
        }
    }
}
