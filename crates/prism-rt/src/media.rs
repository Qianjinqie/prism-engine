//! 均匀 HG 体积介质：相函数抽样 + Delta tracking（白皮书 §3；异质网格 W6+）。
//!
//! # 无偏估计量（推导锚点）
//! - 自由程 t = −ln(u)/σt：碰撞点 pdf = σt·e^(−σt·t)；散射分支概率 = 反照率
//!   σs/σt → 散射事件联合 pdf = σs·e^(−σt·t) = 被积因子 T(t)·σs 本身 →
//!   **散射处 throughput 不变**（T(t) 与 pdf 相消）。
//! - 到面存活 = 隐式二值事件：未碰撞走到面的概率 E = e^(−σt·d) = T(d)，
//!   面贡献自动带权——运行期零 exp。
//! - 阴影透射率 = 单抽二值估计：自由程 ≥ 段长 → 1，否则 0；
//!   E[T̂] = e^(−σt·d) 精确等于 Beer 定律（测试断言之）。
//!
//! # 确定性数学（D-29 简化件，规范缺口上报项）
//! [`det_ln`] 为运行期必需（自由程）；位分解 + atanh 级数，纯 ±×÷ 与整数位
//! 运算，禁 libm，跨平台逐位一致。后续随 prism-math 的 D-29 确定性数学件上收。
//! （透射率测试参考值用 f64 exp——libm 禁令只约束引擎确定性路径，不约束测试。）

use std::f32::consts::PI;

use prism_math::{clamp, Vec3};

use crate::sampler::{cos_sin, onb};

/// u 的夹持下界（= 2^-24，Sobol 坐标量化格）：u = 0 时 −ln(u) = ∞ 会毒化
/// 路径；量化格上 u=0 的质量可忽略，注释声明（与 sampler 高 24 位换算同格）。
const U_EPS: f32 = 1.0 / 16_777_216.0;
/// ln 2（公开数学常量；D-30：与 `std::f32::consts::LN_2` `to_bits()` 逐位相同）。
const LN_2: f32 = std::f32::consts::LN_2;

/// 确定性自然对数（x > 0）。
///
/// 位分解 x = 2^e·m（m ∈ [1,2)）→ ln m = 2·atanh(s)，s = (m−1)/(m+1) ∈
/// [0, 1/3) → atanh 级数 7 项（s¹⁵ 余项 < 1e-8）。次正规数先乘 2^24（精确）
/// 挪入正规域再减 24·ln2。只含 ±×÷ 与整数位运算——D-29。
pub fn det_ln(x: f32) -> f32 {
    assert!(x > 0.0 && x.is_finite(), "det_ln 定义域为正有限数，收到 {x}");
    let bits = x.to_bits();
    if (bits >> 23) & 0xFF == 0 {
        // 次正规数：精确幂缩放进正规域（递归深度 1）。
        return det_ln(x * 16_777_216.0) - 24.0 * LN_2;
    }
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // m ∈ [1,2)
    let f = m - 1.0;
    let s = f / (2.0 + f); // ∈ [0, 1/3)
    let s2 = s * s;
    // atanh(s) = s·(1 + s²/3 + s⁴/5 + … + s¹²/13)，Horner 排布。
    let atanh = s
        * (1.0
            + s2 * (1.0 / 3.0
                + s2 * (1.0 / 5.0
                    + s2 * (1.0 / 7.0
                        + s2 * (1.0 / 9.0
                            + s2 * (1.0 / 11.0 + s2 * (1.0 / 13.0)))))));
    e as f32 * LN_2 + 2.0 * atanh
}

/// 均匀 HG 介质（σt 总消光、反照率 σs/σt、各向异性 g）。
#[derive(Debug, Clone, Copy)]
pub struct HomogeneousMedium {
    /// 总消光系数 σt = σa + σs（1/length）。
    pub sigma_t: f32,
    /// 单次散射反照率 σs/σt ∈ [0,1]。
    pub albedo: f32,
    /// HG 各向异性参数 g ∈ (−1,1)：g > 0 前向散射。
    pub g: f32,
}

impl HomogeneousMedium {
    /// 构造（断言系数域，宪章 C19）。
    pub fn new(sigma_t: f32, albedo: f32, g: f32) -> Self {
        assert!(
            sigma_t > 0.0,
            "σt 必须为正，收到 {sigma_t}；建议：σt = σa + σs 为总消光系数（单位 1/长度），检查介质密度与单位换算"
        );
        assert!(
            (0.0..=1.0).contains(&albedo),
            "反照率必须在 [0,1]，收到 {albedo}；建议：albedo = σs/σt，0 为纯吸收、1 为纯散射，常见材质 0.1–0.9"
        );
        assert!(
            g > -1.0 && g < 1.0,
            "g 必须在 (-1,1) 开区间，收到 {g}；建议：前向散射取正值（雾/皮肤常用 0.3–0.8），后向取负，各向同性填 0"
        );
        HomogeneousMedium { sigma_t, albedo, g }
    }

    /// 散射系数 σs。
    pub fn sigma_s(&self) -> f32 {
        self.sigma_t * self.albedo
    }

    /// 吸收系数 σa。
    pub fn sigma_a(&self) -> f32 {
        self.sigma_t * (1.0 - self.albedo)
    }

    /// 自由程抽样：t = −ln(u)/σt。
    pub fn free_path(&self, u: f32) -> f32 {
        -det_ln(clamp(u, U_EPS, 1.0 - U_EPS)) / self.sigma_t
    }

    /// 阴影透射率（delta tracking 单抽二值）：自由程 ≥ dist → 1，否则 0。
    /// E[T̂] = e^(−σt·dist)（Beer 定律），无偏。
    pub fn shadow_transmittance(&self, u: f32, dist: f32) -> f32 {
        if self.free_path(u) >= dist {
            1.0
        } else {
            0.0
        }
    }

    /// HG 相函数 pdf（按 cosθ）：(1−g²) / (4π·(1+g²−2g·cosθ)^{3/2})。
    pub fn phase(&self, cos_theta: f32) -> f32 {
        let g = self.g;
        let den = 1.0 + g * g - 2.0 * g * cos_theta;
        (1.0 - g * g) / (4.0 * PI * den * den.sqrt())
    }

    /// HG 方向抽样：绕入射（传播）方向 w 散射出的新方向。
    /// cosθ = [1+g² − ((1−g²)/(1−g+2g·u₁))²] / (2g)；g → 0 退化为均匀球。
    pub fn sample_phase(&self, w: Vec3, u1: f32, u2: f32) -> Vec3 {
        let g = self.g;
        let cos_theta = if g.abs() < 1e-6 {
            1.0 - 2.0 * u1 // 各向同性极限
        } else {
            let a = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            (1.0 + g * g - a * a) / (2.0 * g)
        };
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let (t, b) = onb(w);
        let (c, s) = cos_sin(2.0 * PI * u2);
        w * cos_theta + t * (sin_theta * c) + b * (sin_theta * s)
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 4：HG 体积 vs Beer 定律解析值）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler;
    use prism_core::{EvalCtx, SampleIndex};

    #[test]
    fn det_ln_matches_known_values() {
        // 公开数学常量锚点。
        assert_eq!(det_ln(1.0).to_bits(), 0.0f32.to_bits());
        for (x, expect) in [
            (2.0, 0.693_147_2),
            (0.5, -0.693_147_2),
            (10.0, 2.302_585),
            (0.25, -1.386_294_4),
            (std::f32::consts::E, 1.0),
            (1.0e-6, -13.815_511),
            (16_777_216.0, 16.635_532),
        ] {
            let got = det_ln(x);
            assert!(
                (got - expect).abs() < 2e-5,
                "ln({x}) = {got}，期望 {expect}"
            );
        }
        // 确定性：两遍逐位一致。
        assert_eq!(det_ln(0.37).to_bits(), det_ln(0.37).to_bits());
    }

    #[test]
    fn free_path_mean_is_inverse_sigma_t() {
        let m = HomogeneousMedium::new(0.8, 0.9, 0.2);
        let ctx = EvalCtx::new(31, 7, 3);
        let n = 50_000u32;
        let mut sum = 0.0f64;
        for k in 0..n {
            let si = SampleIndex {
                frame: 3,
                pixel: (5, 6),
                sample: k,
                dimension: 40,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            sum += m.free_path(u) as f64;
        }
        let mean = sum / n as f64;
        let expected = 1.0 / m.sigma_t as f64;
        assert!(
            (mean - expected).abs() / expected < 0.02,
            "E[自由程] = {mean}，期望 {expected}"
        );
    }

    #[test]
    fn shadow_transmittance_matches_beers_law() {
        let m = HomogeneousMedium::new(0.7, 1.0, 0.0);
        let dist = 1.3f32;
        let ctx = EvalCtx::new(97, 7, 3);
        let n = 100_000u32;
        let mut ones = 0u32;
        for k in 0..n {
            let si = SampleIndex {
                frame: 3,
                pixel: (1, 1),
                sample: k,
                dimension: 41,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            let t = m.shadow_transmittance(u, dist);
            assert!(t == 0.0 || t == 1.0, "二值估计出现非二值 {t}");
            ones += t as u32;
        }
        let est = ones as f64 / n as f64;
        let beer = (-(m.sigma_t as f64) * dist as f64).exp(); // 测试参考值可用 f64
        assert!(
            (est - beer).abs() / beer < 0.02,
            "E[T̂] = {est}，Beer = {beer}"
        );
    }

    #[test]
    fn phase_integrates_to_one_and_moments_match() {
        for g in [0.3f32, -0.4] {
            let m = HomogeneousMedium::new(1.0, 1.0, g);
            let ctx = EvalCtx::new(11, 7, 3);
            let n = 40_000u32;
            // 归一化 ∫p dΩ = 1 必须用**均匀球面方向**做 MC（E[4π·p] = 1）；
            // 用 HG 抽样方向配 p 会变成 ∫p²——那是矩检验（下半段）的事。
            let (mut sum_p, mut sum_cos, mut sum_cos2) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..n {
                let si = SampleIndex {
                    frame: 3,
                    pixel: (2, 2),
                    sample: k,
                    dimension: 42,
                    node_id: 7,
                };
                let (u1, u2) = sampler::sample_2d(&ctx, &si);
                // 均匀球面方向：z = 1−2u ∈ [−1,1]，φ = 2πv。
                let z = 1.0 - 2.0 * u1;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let (c, s) = crate::sampler::cos_sin(2.0 * std::f32::consts::PI * u2);
                let d = Vec3::new(r * c, r * s, z); // 轴 = w = (0,0,1)
                let cos = d.z as f64;
                sum_p += 4.0 * std::f64::consts::PI * m.phase(cos as f32) as f64;
                // 矩检验用 HG 抽样方向。
                let d_hg = m.sample_phase(Vec3::new(0.0, 0.0, 1.0), u1, u2);
                let cos_hg = d_hg.dot(&Vec3::new(0.0, 0.0, 1.0)) as f64;
                assert!((d_hg.length() - 1.0).abs() < 1e-4, "g={g}: 方向未单位化");
                sum_cos += cos_hg;
                sum_cos2 += cos_hg * cos_hg;
            }
            let norm = sum_p / n as f64;
            assert!((norm - 1.0).abs() < 0.01, "g={g}: ∫p dΩ = {norm}");
            let mean_cos = sum_cos / n as f64;
            assert!(
                (mean_cos - g as f64).abs() < 0.01,
                "g={g}: E[cosθ] = {mean_cos}"
            );
            // HG 恒等式：E[cos²θ] = (1+2g²)/3。
            let mean_cos2 = sum_cos2 / n as f64;
            let expected = (1.0 + 2.0 * (g as f64) * (g as f64)) / 3.0;
            assert!(
                (mean_cos2 - expected).abs() < 0.02,
                "g={g}: E[cos²θ] = {mean_cos2}，期望 {expected}"
            );
        }
    }
}
