//! 阻尼弹簧闭式解析解（工作流 A2；S7 闭式线性响应）。
//!
//! 求解初值问题：`x'' + 2ζω·x' + ω²·x = ω²·T`，`x(0) = x0`，`x'(0) = v0`，
//! `T` 为目标值。对偏差量 `y = x − T` 解齐次方程，按阻尼比 ζ 三分支闭式：
//!
//! - **欠阻尼**（ζ < 1）：`y(t) = e^(−ζωt)·(y0·cos(ω_d t) + b·sin(ω_d t))`，
//!   `ω_d = ω·√(1−ζ²)`，`b = (v0 + ζω·y0)/ω_d`；
//! - **临界**（|ζ − 1| ≤ 1e-3）：`y(t) = e^(−ωt)·(y0 + (v0 + ω·y0)·t)`；
//! - **过阻尼**（ζ > 1）：`y(t) = A·e^(r1 t) + B·e^(r2 t)`，
//!   `r1,2 = ω(−ζ ± √(ζ²−1))`，`A = (v0 − r2·y0)/(r1 − r2)`，`B = y0 − A`。
//!
//! # 确定性（的核心主张）
//!
//! - **无迭代**：任意 t 一次代入 O(1) 求值——对照的 RK4 数值积分仅在本模
//!   块测试内出现（判据线：误差 < 1e-3）；
//! - **seek 即求值**：跳到任意 t 无需从头重放，数值积分的 O(t/dt) 与累积
//!   浮点漂移不存在；
//! - **同 t 重算逐位一致**（测试断言 `to_bits` 相等）；
//! - 超越函数用本 crate [`det`](crate::det) 固定系数多项式（禁 libm，D-29）。
//!
//! 文档化退化行为：`t ≤ 0` 精确取 `x0`（初值）；`ω ≤ 0` 取 `target`（静态，
//! DSL 层防呆拦截非法参数）；深度衰减下溢由 `det::exp` 钳为 0——即收敛态
//! `target`（附注）。

/// 临界分支判定半径：|ζ − 1| 在此半径内按临界处理（极限形式，数值稳定——
/// ζ 极接近 1 时欠阻尼式的 ω_d 舍入噪声会放大）。
const CRITICAL_EPS: f32 = 1e-3;

/// 阻尼弹簧特解参数（阶跃响应：目标 T 为常量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    /// 自然角频率 ω（rad/s，> 0）。
    pub omega: f32,
    /// 阻尼比 ζ（< 1 欠阻尼 / ≈ 1 临界 / > 1 过阻尼）。
    pub zeta: f32,
    /// 初位置 x(0)。
    pub x0: f32,
    /// 初速度 x'(0)。
    pub v0: f32,
    /// 目标值（终态）。
    pub target: f32,
}

impl Spring {
    /// 以 ω / ζ / 初值 / 初速 / 目标构造弹簧。
    pub fn new(omega: f32, zeta: f32, x0: f32, v0: f32, target: f32) -> Self {
        Self { omega, zeta, x0, v0, target }
    }

    /// 闭式解采样：t 的纯函数，O(1) 无迭代，同 t 逐位一致。
    pub fn sample(&self, t: f32) -> f32 {
        if t <= 0.0 {
            return self.x0;
        }
        if self.omega <= 0.0 {
            return self.target;
        }
        let y0 = self.x0 - self.target;
        let (omega, zeta) = (self.omega, self.zeta);
        let deviation = if zeta < 1.0 - CRITICAL_EPS {
            // 欠阻尼：共轭复根，衰减振荡。
            let wd = omega * (1.0 - zeta * zeta).sqrt();
            let decay = crate::det::exp_f32(-zeta * omega * t);
            let a = y0;
            let b = (self.v0 + zeta * omega * y0) / wd;
            decay * (a * crate::det::cos_f32(wd * t) + b * crate::det::sin_f32(wd * t))
        } else if zeta > 1.0 + CRITICAL_EPS {
            // 过阻尼：两个负实根。
            let s = (zeta * zeta - 1.0).sqrt();
            let r1 = -zeta * omega + omega * s;
            let r2 = -zeta * omega - omega * s;
            let a = (self.v0 - r2 * y0) / (r1 - r2);
            let b = y0 - a;
            a * crate::det::exp_f32(r1 * t) + b * crate::det::exp_f32(r2 * t)
        } else {
            // 临界：二重负实根的极限形式。
            let decay = crate::det::exp_f32(-omega * t);
            decay * (y0 + (self.v0 + omega * y0) * t)
        };
        deviation + self.target
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 1：三分支解析解 vs RK4 数值积分 < 1e-3；同 t 重算逐位一致）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// RK4 数值积分参考（**仅测试内**使用；f64 步进， 的对照基线）。
    ///
    /// 状态方程：x' = v，v' = ω²·(T − x) − 2ζω·v。
    fn rk4_reference(spring: &Spring, t_end: f32, dt: f64) -> f64 {
        let (omega, zeta) = (spring.omega as f64, spring.zeta as f64);
        let target = spring.target as f64;
        let deriv = |x: f64, v: f64| (v, omega * omega * (target - x) - 2.0 * zeta * omega * v);
        let mut x = spring.x0 as f64;
        let mut v = spring.v0 as f64;
        let steps = (t_end as f64 / dt).round() as usize;
        let h = t_end as f64 / steps as f64;
        for _ in 0..steps {
            let (k1x, k1v) = deriv(x, v);
            let (k2x, k2v) = deriv(x + h / 2.0 * k1x, v + h / 2.0 * k1v);
            let (k3x, k3v) = deriv(x + h / 2.0 * k2x, v + h / 2.0 * k2v);
            let (k4x, k4v) = deriv(x + h * k3x, v + h * k3v);
            x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }
        x
    }

    /// 关卡 1：三分支各一场景，与 RK4 的最大误差 < 1e-3。
    #[test]
    fn closed_form_matches_rk4_all_three_branches() {
        let cases = [
            ("欠阻尼", Spring::new(6.0, 0.35, 0.0, 0.0, 1.0)),
            ("临界", Spring::new(6.0, 1.0, 0.0, 0.0, 1.0)),
            ("过阻尼", Spring::new(6.0, 1.8, 0.0, 0.0, 1.0)),
        ];
        for (name, spring) in cases {
            let mut max_err = 0.0_f64;
            // 在 [0.05, 1.5] 的采样网格上对比（t=0 处两者同为初值，跳过）。
            let mut i = 1;
            while i <= 30 {
                let t = i as f32 * 0.05;
                let closed = spring.sample(t) as f64;
                let reference = rk4_reference(&spring, t, 1.0 / 1000.0);
                let err = (closed - reference).abs();
                max_err = max_err.max(err);
                i += 1;
            }
            assert!(
                max_err < 1e-3,
                "{name}：解析解 vs RK4 最大误差 {max_err}（判据线 1e-3）"
            );
            println!("关卡1 [{name}] 解析解 vs RK4 最大误差 = {max_err:.3e}（判据线 1e-3）");
        }
    }

    /// 关卡 1：解析解同 t 重算 10000 次逐位一致（无迭代、无漂移）。
    #[test]
    fn closed_form_bitwise_reproducible() {
        let spring = Spring::new(6.0, 0.35, 0.0, 0.0, 1.0);
        for t in [0.0_f32, 0.05, 0.5, 1.0, 7.3, 100.0] {
            let expected = spring.sample(t).to_bits();
            for _ in 0..10_000 {
                assert_eq!(spring.sample(t).to_bits(), expected, "t={t} 重算逐位漂移");
            }
        }
    }

    /// 初值条件：t=0 精确取 x0；带初速度时亦然；负 t 钳初值。
    #[test]
    fn initial_conditions_exact() {
        let spring = Spring::new(6.0, 0.35, 2.0, 3.0, -1.0);
        assert_eq!(spring.sample(0.0).to_bits(), 2.0_f32.to_bits());
        assert_eq!(spring.sample(-1.0).to_bits(), 2.0_f32.to_bits(), "负 t 钳初值");
    }

    /// 收敛性：三分支长时间后都到达 target（下溢钳为收敛态， 附注）。
    #[test]
    fn converges_to_target_for_all_branches() {
        for zeta in [0.35_f32, 1.0, 1.8] {
            let spring = Spring::new(6.0, zeta, 0.0, 0.0, 1.0);
            let settled = spring.sample(40.0);
            assert!(
                (settled - 1.0).abs() < 1e-6,
                "ζ={zeta} 未收敛：{settled}"
            );
        }
    }

    /// 欠阻尼过冲、过阻尼不过冲（物理行为的健全性检查）。
    #[test]
    fn overshoot_behavior_by_damping() {
        let under = Spring::new(6.0, 0.35, 0.0, 0.0, 1.0);
        let over = Spring::new(6.0, 1.8, 0.0, 0.0, 1.0);
        let under_max = (1..300).map(|i| under.sample(i as f32 * 0.01)).fold(0.0_f32, f32::max);
        let over_max = (1..300).map(|i| over.sample(i as f32 * 0.01)).fold(0.0_f32, f32::max);
        assert!(under_max > 1.05, "欠阻尼应过冲（峰值 {under_max}）");
        assert!(over_max <= 1.0, "过阻尼不应过冲（峰值 {over_max}）");
    }

    /// 退化参数：ω ≤ 0 取 target（文档化行为）。
    #[test]
    fn degenerate_omega_returns_target() {
        let spring = Spring::new(0.0, 0.35, 0.0, 0.0, 1.0);
        assert_eq!(spring.sample(0.5), 1.0);
    }
}
