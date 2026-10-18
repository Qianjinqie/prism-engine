//! 帧级率控（water-filling 位分配 / 等失真证书 / B 帧闭式解 / 依赖权重）。
//!
//! V1.5-E：把"逐帧 P 帧残差质量"从固定启发式（`q−17` 夹到 `[50,80]`）升级为
//! **基于实测后验的位分配**：每个 GOP 编码前，用上一 GOP 观测到的每帧残差失真
//! 敏感度 `θ_i = mse_i · 2^{2·R_i}` 做水注（water-filling），在总位预算
//! `bits_total` 下最小化最大失真，得到每帧的目标码率 `R_i`，再由一阶线性控制器
//! 转成 JPEG 质量（`Δq = ΔR · QUALITY_PER_BIT`）。
//!
//! # 确定性契约（D-29）
//!
//! 本模块只用 `± × ÷` 与整数运算：`log2` 走 [`prism_math::det_pow::det_log2`]，
//! `2^x` 走 [`prism_math::det_pow::det_exp2`]，**禁** `f64::log2/exp2/cos` 等 libm
//! 入口。水位二分固定 200 步、无早停、无迭代容差，故同输入两跑逐位一致。
//!
//! # 目标函数（为什么是 max 失真）
//!
//! 每帧失真模型 `D_i = θ_i · 2^{−2·R_i}`（高码率近似，指数率 2）。水注在
//! `Σ R_i = B`、`R_i ≥ 0` 下最小化 `max_i D_i`：拉格朗日条件给出"活跃帧失真全等"
//! ——这正是 [`equal_distortion_ratio`] 输出的证书值（应≈0）。

use prism_math::det_pow::{det_exp2, det_log2};

/// 质量/(bit/符号) 常数：`Δq = ΔR · QUALITY_PER_BIT`（Δq 为 JPEG 质量步长，
/// ΔR 为 bit/符号 步长；两者相除即 质量/(bit/符号)，与 bit/符号 口径的 ΔR 相乘
/// 才能回到质量步长）。
///
/// 标定（主理人实测，合成残差样平面 q68→76）：slope 口径为 bit/符号/q
/// （编码字节数差 ÷ 符号数 ÷ 8）；64×64 slope=0.005524→181.0；
/// 96×54 slope=0.005908→169.3；160×90 slope=0.005885→169.9。大尺寸收敛 ~0.0059，
/// 取 170。一阶控制器近似：`q_new = q_base + (R_target − R_base)·170`。
pub const QUALITY_PER_BIT: f64 = 170.0;

/// 水注位分配：在 `Σ R_i = budget`（`R_i ≥ 0`）下把每条"信息通道"的目标码率解出。
///
/// 输入 `theta[i]` 为第 `i` 条通道的失真敏感度（`D_i = θ_i·2^{−2R_i}`，越大越
/// 需要位）。返回 `R_i`（bit/符号）。
///
/// # 边界
/// - 空输入 → `[]`；
/// - `budget ≤ 0` 或 NaN → 全 0（不分配）；
/// - `theta[i] ≤ 0`、非有限（含 NaN）→ 该项不活跃（`L_i = −∞`，`R_i = 0`）。
///
/// # 算法（对数域水位二分）
/// `L_i = log2 θ_i`；水位 `mid` 下 `alloc(mid) = Σ_active max(0, ½(L_i − mid))`
/// 关于 `mid` 单调不增。取 `lo = lmin − 2·budget − 1`（保证 `alloc(lo) > budget`）、
/// `hi = lmax`（`alloc = 0`），二分 200 步定水位，再令 `R_i = max(0, ½(L_i − hi))`。
pub fn water_filling(theta: &[f64], budget: f64) -> Vec<f64> {
    let g = theta.len();
    if g == 0 {
        return Vec::new();
    }
    // 预算无效：全 0（不静默分配负位）。
    if !(budget > 0.0) || budget.is_nan() {
        return vec![0.0; g];
    }
    let mut l = vec![f64::NEG_INFINITY; g];
    let mut lmin = f64::INFINITY;
    let mut lmax = f64::NEG_INFINITY;
    for i in 0..g {
        let t = theta[i];
        if t.is_finite() && t > 0.0 {
            let li = det_log2(t);
            l[i] = li;
            if li < lmin {
                lmin = li;
            }
            if li > lmax {
                lmax = li;
            }
        }
    }
    // 无活跃通道：全部 0（预算无处可去）。
    if !lmin.is_finite() || !lmax.is_finite() {
        return vec![0.0; g];
    }
    let mut lo = lmin - 2.0 * budget - 1.0;
    let mut hi = lmax;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let mut alloc = 0.0f64;
        for i in 0..g {
            if l[i].is_finite() {
                let v = 0.5 * (l[i] - mid);
                if v > 0.0 {
                    alloc += v;
                }
            }
        }
        if alloc > budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut r = vec![0.0f64; g];
    for i in 0..g {
        if l[i].is_finite() {
            let v = 0.5 * (l[i] - hi);
            r[i] = if v > 0.0 { v } else { 0.0 };
        }
    }
    r
}

/// 等失真证书：活跃通道失真的 `max/min − 1`。水注最优解应为 0（容差内）。
///
/// 活跃定义：`θ_i > 0` 且 `R_i > 0`（被分到位的通道；`R_i = 0` 的通道受
/// `R ≥ 0` 约束而"顶到地板"，其失真不必等于水位）。无活跃通道 → 0。
pub fn equal_distortion_ratio(theta: &[f64], r: &[f64]) -> f64 {
    let mut dmin = f64::INFINITY;
    let mut dmax = f64::NEG_INFINITY;
    let g = theta.len().min(r.len());
    for i in 0..g {
        let t = theta[i];
        if t.is_finite() && t > 0.0 && r[i] > 0.0 && r[i].is_finite() {
            let d = t * det_exp2(-2.0 * r[i]);
            if d < dmin {
                dmin = d;
            }
            if d > dmax {
                dmax = d;
            }
        }
    }
    if !dmin.is_finite() || !dmax.is_finite() || dmin <= 0.0 {
        return 0.0;
    }
    dmax / dmin - 1.0
}

/// B 帧闭式解（**只落函数与单测，不入容器**）：`R_B = ½·log2(θ_B/θ_P)`。
///
/// 在 `D_B = D_P` 的等失真条件下解出 B 帧相对 P 帧应得的码率差。`θ ≤ 0` 或非
/// 有限 → 0（未定义，按"无收益"处理）。
pub fn b_frame_closed_form(theta_p: f64, theta_b: f64) -> f64 {
    if !(theta_p.is_finite() && theta_p > 0.0) || !(theta_b.is_finite() && theta_b > 0.0) {
        return 0.0;
    }
    0.5 * det_log2(theta_b / theta_p)
}

/// GOP 依赖权重：第 `i` 个 P 帧（0 基）影响其后 `gop − i` 帧，故权重 `gop − i`。
pub fn dependency_weights(gop: usize) -> Vec<f64> {
    (0..gop).map(|i| (gop - i) as f64).collect()
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 线性同余发生器（与全仓测试同一族参数；确定性、无外部 crate）。
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Lcg(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
        /// [0,1) 均匀。
        fn next_f64(&mut self) -> f64 {
            (self.next_u32() >> 8) as f64 / ((1u32 << 24) as f64)
        }
    }

    fn sum(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |a, b| a + b)
    }

    #[test]
    fn canonical_case_matches_finalized_numbers() {
        // 定稿算例：θ=[1.0,0.5,0.3,0.2,0.1]、B=6.0。
        let theta = [1.0, 0.5, 0.3, 0.2, 0.1];
        let r = water_filling(&theta, 6.0);
        assert_eq!(r.len(), 5);
        // 全帧活跃 ⇒ 等失真 ⇒ 水位 = 2^hi，ΣD = G·水位。
        let ratio = equal_distortion_ratio(&theta, &r);
        assert!(ratio.abs() <= 1e-9, "等失真证书 {ratio:e} 超 1e-9");
        let d_total: f64 = theta
            .iter()
            .zip(r.iter())
            .map(|(&t, &ri)| t * det_exp2(-2.0 * ri))
            .sum();
        assert!(
            (d_total - 0.296430).abs() <= 1e-6,
            "ΣD={d_total:.9} 应 ≈ 0.296430"
        );
        // 水位（= 单帧失真）= ΣD/G ≈ 0.059286。
        let level = d_total / 5.0;
        assert!(
            (level - 0.059286).abs() <= 1e-6,
            "水位={level:.9} 应 ≈ 0.059286"
        );
        // 预算精确用尽。
        assert!((sum(&r) - 6.0).abs() <= 1e-9, "ΣR={:.12} 应 = 6", sum(&r));
    }

    #[test]
    fn large_gop_across_six_decades() {
        // GOP=1000，θ 跨 6 个数量级：预算用尽 + 证书成立。
        let mut lcg = Lcg::new(2024);
        let theta: Vec<f64> = (0..1000)
            .map(|_| det_exp2(-10.0 + 20.0 * lcg.next_f64())) // 2^-10 … 2^10
            .collect();
        let budget = 20.0;
        let r = water_filling(&theta, budget);
        assert_eq!(r.len(), 1000);
        assert!(
            (sum(&r) - budget).abs() <= 1e-9,
            "GOP=1000 ΣR={:.15} 应 = {budget}",
            sum(&r)
        );
        let ratio = equal_distortion_ratio(&theta, &r);
        assert!(ratio.abs() <= 1e-9, "GOP=1000 证书 {ratio:e} 超 1e-9");
        for (i, &ri) in r.iter().enumerate() {
            assert!(ri >= 0.0 && ri.is_finite(), "第 {i} 项 R={ri} 非法");
        }
    }

    #[test]
    fn edge_cases() {
        // 空输入。
        assert!(water_filling(&[], 1.0).is_empty());
        // G=1：全部预算给唯一通道。
        let one = water_filling(&[0.37], 3.25);
        assert_eq!(one.len(), 1);
        assert!((one[0] - 3.25).abs() <= 1e-9, "G=1 R={} 应 = 3.25", one[0]);
        // θ 全等 → 均分且逐位一致。
        let eq = water_filling(&[1.0; 4], 2.0);
        for i in 1..4 {
            assert_eq!(
                eq[i].to_bits(),
                eq[0].to_bits(),
                "θ 全等时第 {i} 项未逐位均分"
            );
        }
        assert!((eq[0] - 0.5).abs() <= 1e-13, "均分 R={} 应 = 0.5", eq[0]);
        // B=0 → 全 0。
        assert_eq!(water_filling(&[1.0, 2.0], 0.0), vec![0.0, 0.0]);
        // θ 含 0 → 该项 0，其余仍用尽预算。
        let with_zero = water_filling(&[1.0, 0.0, 4.0], 2.0);
        assert_eq!(with_zero[1], 0.0, "θ=0 项必须为 0");
        assert!((sum(&with_zero) - 2.0).abs() <= 1e-9);
        // 负 / NaN / inf 防御：不 panic，且都当不活跃。
        let bad = water_filling(&[-1.0, f64::NAN, f64::INFINITY, 1.0], 1.0);
        assert_eq!(bad[0], 0.0);
        assert_eq!(bad[1], 0.0);
        assert_eq!(bad[2], 0.0);
        assert!((bad[3] - 1.0).abs() <= 1e-9);
        // 全不活跃 + 非法预算：仍返回同长全 0。
        assert_eq!(water_filling(&[0.0, -2.0], 5.0), vec![0.0, 0.0]);
        assert_eq!(water_filling(&[1.0, 1.0], -3.0), vec![0.0, 0.0]);
        assert_eq!(water_filling(&[1.0, 1.0], f64::NAN), vec![0.0, 0.0]);
    }

    #[test]
    fn golden_water_levels() {
        // 三对黄金水位（容差 1e-13）：R = ½(L − hi)。
        // (1) ([2.0,1.0,0.5], B=1.5)：L=[1,0,-1] → hi=−1 → R=[1.0, 0.5, 0.0]。
        let a = water_filling(&[2.0, 1.0, 0.5], 1.5);
        assert!((a[0] - 1.0).abs() <= 1e-13, "a0={:.17}", a[0]);
        assert!((a[1] - 0.5).abs() <= 1e-13, "a1={:.17}", a[1]);
        assert!(a[2].abs() <= 1e-13, "a2={:.17}", a[2]);
        // (2) ([1.0;4], B=2.0) → hi=−1 → 各 0.5。
        let b = water_filling(&[1.0; 4], 2.0);
        for (i, &v) in b.iter().enumerate() {
            assert!((v - 0.5).abs() <= 1e-13, "b{i}={v:.17}");
        }
        // (3) ([4.0,1.0], B=1.0)：L=[2,0] → hi=0 → R=[1.0, 0.0]。
        let c = water_filling(&[4.0, 1.0], 1.0);
        assert!((c[0] - 1.0).abs() <= 1e-13, "c0={:.17}", c[0]);
        assert!(c[1].abs() <= 1e-13, "c1={:.17}", c[1]);
    }

    /// 确定性伪随机搜索：在 `ΣR ≤ budget` 的可行域里采样，返回找到的最小 `ΣD`。
    ///
    /// 作为水注解的**上界对照**（水注是解析最优，搜索只能更差）。用固定 LCG 种子，
    /// 故每次运行数值完全一致。
    fn search_best_sum_d(theta: &[f64], budget: f64, samples: usize, seed: u32) -> f64 {
        let g = theta.len();
        let mut lcg = Lcg::new(seed);
        let mut best = f64::INFINITY;
        for _ in 0..samples {
            // 采样非负权重，归一化到预算 → 可行解。
            let mut w: Vec<f64> = (0..g).map(|_| lcg.next_f64() + 1e-6).collect();
            let tot = sum(&w);
            let mut used = 0.0f64;
            for v in w.iter_mut() {
                *v = *v / tot * budget;
                used += *v;
            }
            // 数值兜底：确保 ΣR ≤ budget（严格可行）。
            if used > budget && used > 0.0 {
                let k = budget / used;
                for v in w.iter_mut() {
                    *v *= k;
                }
            }
            let mut d = 0.0f64;
            for i in 0..g {
                let di = theta[i] * det_exp2(-2.0 * w[i]);
                if di.is_finite() {
                    d += di;
                }
            }
            if d < best {
                best = d;
            }
        }
        best
    }

    #[test]
    fn water_filling_is_optimal_against_search() {
        // 100 组随机 θ，各 2000 次可行采样（合计 20 万次搜索）作为上界对照。
        let mut lcg = Lcg::new(0xC0FF_EE11);
        for case in 0..100 {
            let g = 2 + (lcg.next_u32() as usize % 6); // 2..=7
            let theta: Vec<f64> = (0..g).map(|_| det_exp2(-6.0 + 12.0 * lcg.next_f64())).collect();
            let budget = 0.25 + 6.0 * lcg.next_f64();
            let r = water_filling(&theta, budget);
            let d_wf: f64 = theta
                .iter()
                .zip(r.iter())
                .map(|(&t, &ri)| t * det_exp2(-2.0 * ri))
                .sum();
            // 预算必须用尽（活跃集非空时）。
            assert!(
                (sum(&r) - budget).abs() <= 1e-9,
                "case {case} ΣR={:.15} ≠ {budget}",
                sum(&r)
            );
            let d_search =
                search_best_sum_d(&theta, budget, 2000, 0x5EED_0000 ^ case as u32);
            assert!(
                d_wf <= 1.001 * d_search,
                "case {case}: 水注 ΣD={d_wf:.12} 劣于搜索上界 {d_search:.12} 的 1.001 倍"
            );
            // 证书：等失真。
            let ratio = equal_distortion_ratio(&theta, &r);
            assert!(ratio.abs() <= 1e-9, "case {case} 证书 {ratio:e}");
        }
    }

    #[test]
    fn b_frame_closed_form_known_value() {
        // θ_P=1, θ_B=0.25 ⇒ ½·log2(0.25) = −1.0。
        let v = b_frame_closed_form(1.0, 0.25);
        assert!((v + 1.0).abs() <= 1.1e-16, "b_frame={v:.20} 应 = −1.0");
        // 非正 / 非有限 → 0（未定义按无收益）。
        assert_eq!(b_frame_closed_form(0.0, 1.0), 0.0);
        assert_eq!(b_frame_closed_form(1.0, -1.0), 0.0);
        assert_eq!(b_frame_closed_form(f64::NAN, 1.0), 0.0);
        // 对称性：θ_B=θ_P ⇒ 0。
        assert!(b_frame_closed_form(3.0, 3.0).abs() <= 1e-16);
    }

    #[test]
    fn dependency_weights_shape() {
        assert_eq!(dependency_weights(0), Vec::<f64>::new());
        assert_eq!(dependency_weights(1), vec![1.0]);
        assert_eq!(dependency_weights(4), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn quality_per_bit_calibration_reproduced() {
        // 合成残差样平面（160×90），q68 / q76 各编码一次，测 slope = Δ(bits/符号)/Δq。
        // 标定实测 slope≈0.005885 ⇒ QUALITY_PER_BIT≈169.9，与常量 170 一致。
        let w = 160u32;
        let h = 90u32;
        let mut lcg = Lcg::new(0x1234_5678);
        let mut plane = vec![0u8; (w * h) as usize];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let s = lcg.next_u32();
                let v = 128 + ((x * 7 + y * 3) % 41) - 20 + ((s >> 24) % 17) as i32 - 8;
                plane[(y as u32 * w + x as u32) as usize] = v.clamp(0, 255) as u8;
            }
        }
        // 标定口径（与主理人实测同源）：q68 → q76 各编码一次，取
        // `slope = Δ(每符号字节数) / Δq`。实测 160×90 得 0.005885 ⇒ 1/slope = 169.9，
        // 与常量 `QUALITY_PER_BIT = 170` 吻合。
        let bytes68 = crate::jpeg::encode_jpeg_gray(
            &plane,
            w,
            h,
            crate::JpegConfig { quality: 68, grayscale: true },
        )
        .unwrap()
        .len() as f64;
        let bytes76 = crate::jpeg::encode_jpeg_gray(
            &plane,
            w,
            h,
            crate::JpegConfig { quality: 76, grayscale: true },
        )
        .unwrap()
        .len() as f64;
        let sym = (w * h) as f64;
        let slope = ((bytes76 - bytes68) / sym) / 8.0;
        assert!(
            (0.0055..=0.0062).contains(&slope),
            "标定 slope={slope:.6} 落在 [0.0055,0.0062] 之外（QUALITY_PER_BIT 需重新标定）"
        );
        // 换算出的 QUALITY_PER_BIT 估计与常量同量级（±10%）。
        let est = 1.0 / slope;
        assert!(
            (est - QUALITY_PER_BIT).abs() / QUALITY_PER_BIT <= 0.10,
            "标定估计 {est:.1} 偏离常量 {QUALITY_PER_BIT} 超 10%"
        );
    }
}
