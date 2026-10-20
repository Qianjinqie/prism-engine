//! 四次方程精确求根（Ferrari 解法；圆环图元求交内核）。
//!
//! # 为什么必须精确（钉死口径）
//! 圆环（torus）的射线求交导出四次方程。规格钉死：**四次方程精确求根；
//! 禁止牛顿迭代**，除非给出确定性收敛证明并写进代码注释。本实现即
//! Ferrari 闭式解：降次 → 预解三次方程 → 分解两个二次因子，全程闭式求值，
//! **零迭代**，同输入逐位一致（D-29）。
//!
//! # 数值稳健口径
//! - 预解三次方程按判别式符号分流：`Δ > 0`（单实根）走 Cardano
//!   （`cbrt` 为闭式函数求值，非迭代）；`Δ <= 0`（三实根）走三角法，
//!   `acos` 经 [`det_atan2`] 求值、`cos` 经 [`det_cos`] 求值（禁平台 libm）。
//! - 选预解根 m 的确定性规则：取**最大实根**。可证 `g(p/2) = −q² <= 0`
//!   且 `g(+∞) = +∞`，故最大实根恒满足 `2m − p >= 0`（分解因子实系数
//!   的前提）；`q == 0`（双二次）走专用捷径，不触该边界。
//! - 失败/退化一律「少返回根」而非填假根：调用方按候选根逐个验证
//!   （`t >= T_MIN && t < t_max`），NaN 根被自然排除。

use crate::det_math::{det_atan2, det_cos};
use std::f32::consts::PI;

/// 解四次方程 `c4·t⁴ + c3·t³ + c2·t² + c1·t + c0 = 0`（c4 != 0）。
///
/// 返回实根数（0..=4）；实根写入 `out[..n]`（**不排序**——调用方取最小
/// 有效值即可，排序只引入无谓的结合序差异）。
///
/// `c4 == 0`（退化为三次）时按三次方程求解（共用预解三次件）。
pub fn solve_quartic(c4: f32, c3: f32, c2: f32, c1: f32, c0: f32, out: &mut [f32; 4]) -> usize {
    if !(c4.is_finite() && c3.is_finite() && c2.is_finite() && c1.is_finite() && c0.is_finite()) {
        return 0; // 非有限系数 ⇒ 无根（调用方图元本就有入口有限性检查）
    }
    if c4 == 0.0 {
        // 三次至多 3 个实根：写入 out[..3]（out 容量 4，类型收窄用子切片）。
        return solve_cubic(c3, c2, c1, c0, &mut out[..3]);
    }
    // 归一化：t⁴ + a·t³ + b·t² + c·t + d = 0。
    let a = c3 / c4;
    let b = c2 / c4;
    let c = c1 / c4;
    let d = c0 / c4;

    // 降次（t = x − a/4）：x⁴ + p·x² + q·x + r = 0。
    let a2 = a * a;
    let p = b - 3.0 * a2 / 8.0;
    let q = a * a2 / 8.0 - a * b / 2.0 + c;
    let r = -3.0 * a2 * a2 / 256.0 + a2 * b / 16.0 - a * c / 4.0 + d;

    // 双二次捷径（q == 0）：x⁴ + p·x² + r = 0 ⇒ u = x² 的二次方程。
    // q == 0 时预解根 m = p/2 使 α = √(2m−p) = 0，Ferrari 分解的
    // β = −q/(2α) 除零——故显式分流（同时更快）。
    if q == 0.0 {
        return solve_biquadratic(p, r, a / 4.0, out);
    }

    // 预解三次方程（Ferrari）：8m³ − 4p·m² − 8r·m + (4pr − q²) = 0。
    // 除以 8 降首项：m³ − (p/2)m² − r·m + (pr/2 − q²/8) = 0。
    let mut cub_roots = [0.0f32; 3];
    let n_cub = solve_cubic(1.0, -p / 2.0, -r, p * r / 2.0 - q * q / 8.0, &mut cub_roots);
    if n_cub == 0 {
        return 0; // 理论上不可达（实系数三次恒有实根）；防御性少返回根
    }
    // 确定性规则：取最大实根（保证 2m − p >= 0，见模块文档证明）。
    let m = cub_roots[..n_cub]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let alpha2 = 2.0 * m - p;
    if !(alpha2 > 0.0) {
        // 数值边角：最大预解根恰落 p/2（仅 q==0 可达，已分流）或非有限。
        // 保守退双二次——不填假根。
        return solve_biquadratic(p, r, a / 4.0, out);
    }
    let alpha = alpha2.sqrt();
    let beta = -q / (2.0 * alpha);
    // x⁴ + p·x² + q·x + r = (x² + αx + (m+β))(x² − αx + (m−β))。
    // 注意：`solve_quadratic_shifted` 返回的是**累计**总根数（不是新增
    // 数），故用赋值而非累加——累加会重复计数（2+4=6 越界，
    // 圆环求交对拍修复：两因子均出根时 `roots[..n]` 直接 panic）。
    let mut n = 0usize;
    n = solve_quadratic_shifted(1.0, alpha, m + beta, a / 4.0, out, n);
    n = solve_quadratic_shifted(1.0, -alpha, m - beta, a / 4.0, out, n);
    n
}

/// 双二次方程 `x⁴ + p·x² + r = 0`（q == 0 捷径；`shift = a/4`，
/// 根平移 `t = x − shift`）。
fn solve_biquadratic(p: f32, r: f32, shift: f32, out: &mut [f32; 4]) -> usize {
    // u = x²：u² + p·u + r = 0。
    let disc = p * p - 4.0 * r;
    if disc < 0.0 {
        return 0;
    }
    let sq = disc.sqrt();
    let mut n = 0usize;
    for &u in &[(-p - sq) / 2.0, (-p + sq) / 2.0] {
        if u < 0.0 {
            continue; // x 无实解
        }
        if u == 0.0 {
            push_root(out, &mut n, -shift); // x = 0 单根
        } else {
            let s = u.sqrt();
            push_root(out, &mut n, s - shift);
            push_root(out, &mut n, -s - shift);
        }
    }
    n
}

/// 二次方程 `A·x² + B·x + C = 0`，根平移 `t = x − shift`（Ferrari 因子用，
/// A = 1）。返回写入 `out` 的总根数（`n` 为已有根数）。
fn solve_quadratic_shifted(
    a2: f32,
    b2: f32,
    c2: f32,
    shift: f32,
    out: &mut [f32; 4],
    mut n: usize,
) -> usize {
    let disc = b2 * b2 - 4.0 * a2 * c2;
    if disc < 0.0 {
        return n;
    }
    let sq = disc.sqrt();
    push_root(out, &mut n, (-b2 - sq) / (2.0 * a2) - shift);
    push_root(out, &mut n, (-b2 + sq) / (2.0 * a2) - shift);
    n
}

/// 三次方程 `A·t³ + B·t² + C·t + D = 0` 的实根（Cardano / 三角法闭式）。
///
/// 返回实根数（1..=3）；实根写入 `out[..n]`（不排序）。`out` 容量须 ≥ 3。
pub fn solve_cubic(a3: f32, b3: f32, c3: f32, d3: f32, out: &mut [f32]) -> usize {
    if !(a3.is_finite() && b3.is_finite() && c3.is_finite() && d3.is_finite()) {
        return 0;
    }
    if a3 == 0.0 {
        return solve_quadratic_public(b3, c3, d3, out);
    }
    // 归一化 + 降次：z³ + P·z + Q = 0（t = z − B/(3A)）。
    let a = b3 / a3;
    let b = c3 / a3;
    let c = d3 / a3;
    let a2 = a * a;
    let p = b - a2 / 3.0;
    let q = 2.0 * a * a2 / 27.0 - a * b / 3.0 + c;
    let shift = a / 3.0;

    if p == 0.0 {
        // z³ = −q：单实根（cbrt 闭式）。
        return write_roots(out, &[(-q).cbrt() - shift]);
    }
    let disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if disc > 0.0 {
        // 单实根：Cardano（cbrt 闭式函数求值，非迭代）。
        let sq = disc.sqrt();
        let z = (-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt();
        write_roots(out, &[z - shift])
    } else {
        // 三实根：三角法。acos 经 det_atan2、cos 经 det_cos（禁平台 libm）。
        let arg = ((3.0 * q / (2.0 * p)) * (-3.0 / p).sqrt()).clamp(-1.0, 1.0);
        let theta = det_atan2((1.0 - arg * arg).max(0.0).sqrt(), arg) / 3.0;
        let scale = 2.0 * (-p / 3.0).sqrt();
        write_roots(
            out,
            &[
                scale * det_cos(theta) - shift,
                scale * det_cos(theta - 2.0 * PI / 3.0) - shift,
                scale * det_cos(theta + 2.0 * PI / 3.0) - shift,
            ],
        )
    }
}

/// 二次方程公开入口（三次退化 / 测试用）：`A·t² + B·t + C = 0`。
/// `out` 容量须 ≥ 2。
pub fn solve_quadratic_public(a2: f32, b2: f32, c2: f32, out: &mut [f32]) -> usize {
    if !(a2.is_finite() && b2.is_finite() && c2.is_finite()) {
        return 0;
    }
    if a2 == 0.0 {
        if b2 == 0.0 {
            return 0;
        }
        return write_roots(out, &[-c2 / b2]);
    }
    let disc = b2 * b2 - 4.0 * a2 * c2;
    if disc < 0.0 {
        return 0;
    }
    let sq = disc.sqrt();
    write_roots(out, &[(-b2 - sq) / (2.0 * a2), (-b2 + sq) / (2.0 * a2)])
}

/// 写入根（跳过 NaN/非有限——调用方不接收假根）。
fn write_roots(out: &mut [f32], roots: &[f32]) -> usize {
    let mut n = 0usize;
    for &r in roots {
        if r.is_finite() {
            out[n] = r;
            n += 1;
        }
    }
    n
}

/// 写入单根（内部便捷件；容量由调用方保证）。
#[inline]
fn push_root(out: &mut [f32; 4], n: &mut usize, r: f32) {
    if *n < out.len() && r.is_finite() {
        out[*n] = r;
        *n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    /// 已知值对拍（手算参考）：(t−1)(t−2)(t−3)(t−4) 展开为
    /// t⁴−10t³+35t²−50t+24 ⇒ 根 {1,2,3,4}。
    ///
    /// 第二组 (t−1)(t−2)(t−4)(t−8) = t⁴−15t³+70t²−120t+64 走
    /// **Ferrari 主路径**（q=−16.875≠0，且两因子均出实根）——专杀
    /// 「累计数被当新增数累加」的重复计数回归（n=2+4=6 越界）。
    #[test]
    fn quartic_known_roots() {
        let mut out = [0.0f32; 4];
        let n = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0, &mut out);
        assert_eq!(n, 4);
        let mut got = out[..n].to_vec();
        got.sort_by(f32::total_cmp);
        for (g, want) in got.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((g - want).abs() < 1e-5, "根 {g} 偏离 {want}");
        }

        let n2 = solve_quartic(1.0, -15.0, 70.0, -120.0, 64.0, &mut out);
        assert_eq!(n2, 4, "Ferrari 主路径两因子均出根时应报 4（防重复计数回归）");
        let mut got2 = out[..n2].to_vec();
        got2.sort_by(f32::total_cmp);
        for (g, want) in got2.iter().zip([1.0, 2.0, 4.0, 8.0]) {
            assert!((g - want).abs() < 1e-4, "根 {g} 偏离 {want}");
        }
    }

    /// 重根 + 复根混合：(t−1)²(t²+1) 展开为 t⁴−2t³+2t²−2t+1 ⇒ 实根 {1,1}。
    #[test]
    fn quartic_double_root_and_complex_pair() {
        let mut out = [0.0f32; 4];
        let n = solve_quartic(1.0, -2.0, 2.0, -2.0, 1.0, &mut out);
        assert_eq!(n, 2, "只应报实根");
        for g in &out[..n] {
            assert!((g - 1.0).abs() < 1e-5, "重根应为 1，得到 {g}");
        }
    }

    /// 全复根：t⁴+1 = 0 ⇒ 0 实根（不填假根）。
    #[test]
    fn quartic_all_complex_returns_none() {
        let mut out = [0.0f32; 4];
        assert_eq!(solve_quartic(1.0, 0.0, 0.0, 0.0, 1.0, &mut out), 0);
    }

    /// 双二次捷径：t⁴ − 5t² + 4 = 0 ⇒ {±1, ±2}。
    #[test]
    fn biquadratic_shortcut() {
        let mut out = [0.0f32; 4];
        let n = solve_quartic(1.0, 0.0, -5.0, 0.0, 4.0, &mut out);
        assert_eq!(n, 4);
        let mut got = out[..n].to_vec();
        got.sort_by(f32::total_cmp);
        for (g, want) in got.iter().zip([-2.0, -1.0, 1.0, 2.0]) {
            assert!((g - want).abs() < 1e-5);
        }
    }

    /// 圆环四次方程实战形态（torus_t 同构系数构造，手算参考）：
    /// 环心原点、R=2、r=0.5，射线 o=(0,0,5) d=(0,0,−1) 沿 −z 穿过环平面
    /// （x=0 ⇒ ρ=|z|，y=0）。管截面在 (ρ,y) 平面为环带 ρ∈[R−r, R+r]=[1.5,2.5]：
    /// z=5−t ⇒ t=2.5（ρ=2.5 前外表）、t=3.5（ρ=1.5 前内表）、穿过后继续
    /// t=6.5（ρ=1.5 后内表）、t=7.5（ρ=2.5 后外表）——闭式解须给全 4 个
    /// 实根（闭合实体前后两半都记录，最近 t>0 由求交方挑选）。
    #[test]
    fn torus_quartic_known_hits() {
        let (cx, cy, cz) = (0.0f32, 0.0f32, 0.0f32);
        let (major, minor) = (2.0f32, 0.5f32);
        let ro = Vec3::new(cx, cy, cz + 5.0);
        let rd = Vec3::new(0.0, 0.0, -1.0);
        let q = ro - Vec3::new(cx, cy, cz);
        let u = major * major - minor * minor;
        let a2 = rd.dot(&rd);
        let a1 = 2.0 * q.dot(&rd);
        let a0 = q.dot(&q);
        let b2 = rd.x * rd.x + rd.z * rd.z;
        let b1 = 2.0 * (q.x * rd.x + q.z * rd.z);
        let b0 = q.x * q.x + q.z * q.z;
        let c4 = a2 * a2;
        let c3 = 2.0 * a2 * a1;
        let c2 = a1 * a1 + 2.0 * a2 * (a0 + u) - 4.0 * major * major * b2;
        let c1 = 2.0 * a1 * (a0 + u) - 4.0 * major * major * b1;
        let c0 = (a0 + u) * (a0 + u) - 4.0 * major * major * b0;
        let mut out = [0.0f32; 4];
        let n = solve_quartic(c4, c3, c2, c1, c0, &mut out);
        let mut got: Vec<f32> = out[..n].iter().copied().filter(|t| *t > 0.0).collect();
        got.sort_by(f32::total_cmp);
        assert_eq!(got.len(), 4, "正根应恰四个：{got:?}");
        for (g, want) in got.iter().zip([2.5, 3.5, 6.5, 7.5]) {
            assert!((g - want).abs() < 1e-4, "根 {g} 偏离手算参考 {want}");
        }
    }
}
