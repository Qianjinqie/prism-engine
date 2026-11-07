//! 确定性标量函数族（V3.0 W1 `det_*` 覆盖补齐）。
//!
//! `docs/v3.0-语言规格.md` §6 规定：超出既有 `det_pow` / `det_trig` /
//! `colorspace_registry::det_atan2` 覆盖面的函数，**先在 `prism-math` 补零 libm
//! 实现并加单测，再开放语言语法**。本模块就是那批补齐实现。
//!
//! 硬约束：
//! - **零 libm**：不调用 `f64::sqrt` / `f64::sin` / `f64::floor` / `f64::powi`
//!   等平台方法；只用 IEEE-754 位操作、整数运算、`+ - * /` 与比较。
//! - **迭代次数与输入无关**：`det_sqrt` 固定 6 步牛顿，不设收敛提前退出，
//!   保证求值轨迹与分支行为只由调用序列决定。
//! - **与既有族同源**：`det_log10` 复用 `det_pow::det_ln`，`det_asin` /
//!   `det_atan` 复用 `colorspace_registry::det_atan2`，不另造第二份数学实现。

use crate::{colorspace_registry, det_pow, det_trig};

/// 自然对数 10（常量，避免运行时求值）。
const LN_10: f64 = 2.302_585_092_994_045_684_017_991_454_684_4;

/// 圆周率 / 2（`det_asin` 端点值）。
const FRAC_PI_2: f64 = 1.570_796_326_794_896_619_231_321_691_639_8;

/// 符号位掩码取反：清掉最高位即取绝对值（对 NaN / Inf 也按位型处理）。
pub fn det_abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

/// 符号函数：正数 `1`、负数 `-1`、零保留其 `±0`、NaN 原样返回。
pub fn det_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// 确定性最小值（不用 `f64::min`，避免 NaN 传播规则依赖平台 intrinsic）。
pub fn det_min(a: f64, b: f64) -> f64 {
    if a <= b {
        a
    } else {
        b
    }
}

/// 确定性最大值（见 [`det_min`]）。
pub fn det_max(a: f64, b: f64) -> f64 {
    if a >= b {
        a
    } else {
        b
    }
}

/// 钳制到 `[lo, hi]`；`lo > hi` 时按 `det_min` 优先返回 `hi`（口径固定、不取默认）。
pub fn det_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    det_max(lo, det_min(x, hi))
}

/// 朝零截断（位操作，不调 `f64::trunc`）。
pub fn det_trunc(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let bits: u64 = x.to_bits();
    let exp: i32 = (((bits >> 52) & 0x7ff) as i32) - 1023;
    if exp >= 52 {
        return x;
    }
    if exp < 0 {
        return if x < 0.0 { -0.0 } else { 0.0 };
    }
    let frac_bits: u32 = (52 - exp) as u32;
    let mask: u64 = !((1u64 << frac_bits) - 1);
    f64::from_bits(bits & mask)
}

/// 向下取整（负半轴按 `trunc - 1` 处理，不用 `f64::floor`）。
pub fn det_floor(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let trunc: f64 = det_trunc(x);
    if x < 0.0 && trunc > x {
        trunc - 1.0
    } else {
        trunc
    }
}

/// 向上取整（恒等式 `ceil(x) = -floor(-x)`）。
pub fn det_ceil(x: f64) -> f64 {
    -det_floor(-x)
}

/// 四舍五入（半远离零，口径固定，不依赖平台舍入模式）。
pub fn det_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if x > 0.0 {
        det_floor(x + 0.5)
    } else {
        det_ceil(x - 0.5)
    }
}

/// 确定性平方根：位型初值 + **固定 6 步**牛顿迭代。
///
/// 不用 `f64::sqrt`（libm / 平台 intrinsic），不设「收敛即退出」分支，
/// 迭代次数与 `x` 无关。负输入返回 NaN，Inf / ±0 原样返回。
pub fn det_sqrt(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x;
    }
    // IEEE-754 binary64 的指数折半初值；相对误差 < 12%，6 步牛顿后达 ~1 ulp。
    let seed_bits: u64 = (x.to_bits() >> 1).wrapping_add(0x1ff8_0000_0000_0000);
    let mut y: f64 = f64::from_bits(seed_bits);
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// 常用对数（复用 `det_pow::det_ln`，常量分母）。
pub fn det_log10(x: f64) -> f64 {
    det_pow::det_ln(x) / LN_10
}

/// 反正切（复用 `colorspace_registry::det_atan2`）。
pub fn det_atan(x: f64) -> f64 {
    colorspace_registry::det_atan2(x, 1.0)
}

/// 反正弦：`asin(x) = atan2(x, sqrt(1 - x²))`，端点取 `±π/2`。
///
/// 复用既有 `det_atan2` 与本模块 `det_sqrt`，不另造反三角实现。
pub fn det_asin(x: f64) -> f64 {
    if x >= 1.0 {
        return FRAC_PI_2;
    }
    if x <= -1.0 {
        return -FRAC_PI_2;
    }
    colorspace_registry::det_atan2(x, det_sqrt(1.0 - x * x))
}

/// 线性插值（纯算术；`t` 不钳制，超出 `[0,1]` 按线性外推，口径固定）。
pub fn det_lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// 确定性取余：`a - trunc(a / b) * b`，不调 `f64::rem_euclid` / `%` 的 frem intrinsic。
pub fn det_fmod(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || b == 0.0 {
        return f64::NAN;
    }
    a - det_trunc(a / b) * b
}

/// 浮点转 `f32` 的确定性三角族入口（`det_trig` 的公开面是 `f32`）。
///
/// 语言执行器只经此函数进三角族，保证转换点唯一、不散落 `as f32`。
pub fn det_sin_f64(x: f64) -> f64 {
    f64::from(det_trig::sin(x as f32))
}

/// 见 [`det_sin_f64`]。
pub fn det_cos_f64(x: f64) -> f64 {
    f64::from(det_trig::cos(x as f32))
}

/// 见 [`det_sin_f64`]。
pub fn det_tan_f64(x: f64) -> f64 {
    f64::from(det_trig::tan(x as f32))
}

/// 见 [`det_sin_f64`]。
pub fn det_acos_f64(x: f64) -> f64 {
    f64::from(det_trig::acos(x as f32))
}

/// 确定性幂（复用 `det_pow::det_pow`，其公开面是 `f32`）。
pub fn det_pow_f64(x: f64, p: f64) -> f64 {
    f64::from(det_pow::det_pow(x as f32, p as f32))
}

/// 确定性指数（`det_pow` 的 `f64` 面）。
pub fn det_exp_f64(x: f64) -> f64 {
    det_pow::det_exp(x)
}

/// 确定性自然对数（`det_pow` 的 `f64` 面）。
pub fn det_ln_f64(x: f64) -> f64 {
    det_pow::det_ln(x)
}

/// 确定性以 2 为底对数（`det_pow` 的 `f64` 面）。
pub fn det_log2_f64(x: f64) -> f64 {
    det_pow::det_log2(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_sign_are_bit_exact() {
        assert_eq!(det_abs(-3.5), 3.5);
        assert_eq!(det_abs(3.5), 3.5);
        assert_eq!(det_abs(-0.0), 0.0);
        assert!(det_abs(-0.0).to_bits() == 0.0f64.to_bits());
        assert_eq!(det_sign(-7.0), -1.0);
        assert_eq!(det_sign(7.0), 1.0);
        assert_eq!(det_sign(0.0), 0.0);
        assert_eq!(det_sign(-0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn min_max_clamp_are_deterministic() {
        assert_eq!(det_min(2.0, 3.0), 2.0);
        assert_eq!(det_max(2.0, 3.0), 3.0);
        assert_eq!(det_min(-1.0, -2.0), -2.0);
        assert_eq!(det_clamp(5.0, 0.0, 1.0), 1.0);
        assert_eq!(det_clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(det_clamp(0.25, 0.0, 1.0), 0.25);
    }

    #[test]
    fn trunc_floor_ceil_round_match_analytic_values() {
        let xs: [f64; 8] = [2.0, 2.25, 2.5, 2.75, -2.25, -2.5, -2.75, -0.25];
        for &x in &xs {
            assert_eq!(det_trunc(x), x.trunc(), "trunc {x}");
            assert_eq!(det_floor(x), x.floor(), "floor {x}");
            assert_eq!(det_ceil(x), x.ceil(), "ceil {x}");
        }
        assert_eq!(det_round(2.5), 3.0);
        assert_eq!(det_round(2.4), 2.0);
        assert_eq!(det_round(-2.5), -3.0);
        assert_eq!(det_round(-2.4), -2.0);
        assert_eq!(det_floor(1e300), 1e300);
        assert_eq!(det_trunc(-0.25).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn sqrt_matches_analytic_within_1e_12() {
        let xs: [f64; 10] = [
            1.0, 2.0, 3.0, 4.0, 0.25, 0.0001, 1e-8, 1e8, 12345.6789, 0.999999,
        ];
        for &x in &xs {
            let got: f64 = det_sqrt(x);
            let want: f64 = x.sqrt(); // 仅测试参考值允许用标准库，实现里禁用
            let rel: f64 = ((got - want) / want).abs();
            assert!(rel < 1e-12, "sqrt({x}) = {got}, want {want}, rel {rel}");
        }
        assert_eq!(det_sqrt(0.0), 0.0);
        assert!(det_sqrt(-1.0).is_nan());
    }

    #[test]
    fn sqrt_is_bit_reproducible_across_two_runs() {
        let a: Vec<u64> = (1..=64)
            .map(|i| det_sqrt(i as f64 * 0.137).to_bits())
            .collect();
        let b: Vec<u64> = (1..=64)
            .map(|i| det_sqrt(i as f64 * 0.137).to_bits())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn log10_asin_atan_reuse_existing_families() {
        assert!((det_log10(1000.0) - 3.0).abs() < 1e-9);
        assert!((det_log10(1.0)).abs() < 1e-12);
        assert!((det_atan(1.0) - FRAC_PI_2 / 2.0).abs() < 1e-6);
        assert!((det_asin(1.0) - FRAC_PI_2).abs() < 1e-12);
        assert!((det_asin(0.5) - 0.523_598_775_598_298_8).abs() < 1e-6);
        assert!((det_asin(-0.5) + 0.523_598_775_598_298_8).abs() < 1e-6);
    }

    #[test]
    fn lerp_fmod_and_wrappers_are_pure_arithmetic() {
        assert_eq!(det_lerp(2.0, 6.0, 0.25), 3.0);
        assert_eq!(det_lerp(2.0, 6.0, 0.0), 2.0);
        assert_eq!(det_fmod(7.0, 3.0), 1.0);
        assert_eq!(det_fmod(-7.0, 3.0), -1.0);
        assert!(det_fmod(1.0, 0.0).is_nan());
        assert!((det_sin_f64(0.0)).abs() < 1e-12);
        assert!((det_cos_f64(0.0) - 1.0).abs() < 1e-12);
        assert!((det_pow_f64(2.0, 10.0) - 1024.0).abs() < 1e-2);
    }

    #[test]
    fn no_libm_calls_in_source() {
        // 静态门禁：本模块实现里不得出现平台数学方法调用。
        let src: &str = include_str!("det_scalar.rs");
        let before: &str = src.split("#[cfg(test)]").next().unwrap_or(src);
        for banned in [
            ".sqrt()", ".sin()", ".cos()", ".tan()", ".floor()", ".ceil()", ".trunc()",
            ".round()", ".powi(", ".powf(", ".ln()", ".log2(", ".log10(", ".exp()",
            ".min(", ".max(", ".abs()",
        ] {
            assert!(!before.contains(banned), "实现里禁用的 libm 调用: {banned}");
        }
    }
}
