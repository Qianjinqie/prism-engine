//! 球谐（spherical harmonics）SH-L2 辐照度表示（账三 U4；架构 v2.4 S2 卷积/调和分析核）。
//!
//! # 为什么是 SH（D-21）
//! 辐照度是**方向**的函数，探针须用少量系数紧凑编码该方向性。SH 是球面上
//! 的正交基（S2 的傅里叶基），前 9 项（L0+L1+L2）即可表示"有多少光来自
//! 上方/下方/各轴"——正是漫反射辐照度所需（Ramamoorthi & Hanrahan 2001
//! 证明漫反射辐照度 99% 能量落在 L≤2）。用 SH-L2 比"半球均匀"档更强、
//! 比"逐方向直存"档更省，正是账本"质量-预算曲线"的中间档。
//!
//! # ★ 方向与 π 约定（最容易错的地方，务必看懂再改）★
//!
//! 本模块采用 **Ramamoorthi & Hanrahan / PBRT 标准的"辐射度投影 + 余弦瓣卷积"约定**：
//!
//! ```text
//! 基函数 Y_l^m（实数形式，单位方向 ω = (x, y, z)）：
//!   Y_0^0  = 0.282095                       ( = 0.5·sqrt(1/π) )
//!   Y_1^-1 = 0.488603 · y
//!   Y_1^0  = 0.488603 · z
//!   Y_1^1  = 0.488603 · x
//!   Y_2^-2 = 1.092548 · x·y
//!   Y_2^-1 = 1.092548 · y·z
//!   Y_2^0  = 0.315392 · (3z² − 1)
//!   Y_2^1  = 1.092548 · x·z
//!   Y_2^2  = 0.546274 · (x² − y²)
//! ```
//!
//! **关键：投影的是"入射辐亮度 L(ω')"，不是辐照度。余弦项在重建时才加，
//! 且通过"余弦瓣的 SH 卷积系数 Â_l"实现——这是本实现与"投影时就乘 cos"
//! 的朴素做法（只对单一法线有效）的本质区别。**
//!
//! ```text
//! 投影（forward，把 L 映到 SH）：   L_lm = ∫_S L(ω') · Y_lm(ω') dω'          （原始辐射度）
//! 重建（inverse，余弦瓣卷积）：     E(ω) ≈ Σ_lm Â_l · L_lm · Y_lm(ω)
//! 漫反射出射辐亮度：                L_o  = albedo · E(n) · (1/π)   （Lambert BRDF = albedo/π）
//! ```
//!
//! 其中余弦瓣 `max(0, cosθ)` 的 SH 卷积系数（Zonal 谐波）为：
//!
//! ```text
//! Â_0 = π          （l = 0）
//! Â_1 = 2π/3       （l = 1）
//! Â_2 = π/4        （l = 2）
//! ```
//!
//! **π 因子的去向（两处，各司其职）**：
//! 1. 余弦瓣系数 `Â_l` 含 π（来自 `∫_hemisphere cosθ·Y_lm dω`）；
//! 2. 最后转辐亮度时再乘 `1/π`（Lambert BRDF = albedo/π）。
//!
//! **常数环境自检（能量守恒）**：常数环境 L(ω') = L0 下，唯一非零系数是
//! `L_00 = L0 · Y00 · 4π = L0/(Y00)⁻¹...` 精确地 `L_00 = L0 · 4π · Y00`（因为
//! `∫ Y_i dω = 4π·Y00·δ_{i0}`），重建得 `E(ω) = Â_0 · L_00 · Y00 = π·L0`
//! （**与 ω 无关**，各向同性）。这是测试 `constant_env_reconstructs_to_pi_l0`
//! 的解析锚点。
//!
//! # 本实现的选择说明（诚实）
//! 上式是"投影原始辐亮度 + 重建卷积"——它对**任意重建方向 ω** 都给出正确的
//! E(ω)（各向同性场处处 π·L0）。若改为"投影时乘 cos"（只在单一法线处有效），
//! 常数场只在那一个方向对，换方向就错（本模块早期版本即此坑，测试暴露后修正）。
//!
//! # 确定性
//! 系数按**固定次序列**（0..9）逐样本累加，无并行、无顺序依赖；基函数只用
//! `+ - × ÷` 与常量，无 libm。同输入逐位可复现。
//!
//! # 预算档位
//! [`ShOrder`] 控制使用前几阶：`L0`（1 系数，最省）、`L1`（4 系数，方向性
//! 有突变）、`L2`（9 系数，平滑方向性，默认）。重建时未启用阶的系数留 0。

use std::f32::consts::PI;

use prism_math::Vec3;

/// 分量乘积（`prism_math::Vec3` 未提供 component-wise mul）。
#[inline]
pub(crate) fn hadamard(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// 余弦瓣 `max(0, cosθ)` 的 SH 卷积系数 `Â_l`（l = 0, 1, 2）。
///
/// 由 `Â_l = ∫_hemisphere cosθ·Y_l0(θ) dω / Y_l0(0)` 解析给出：
/// `Â_0 = π`、`Â_1 = 2π/3`、`Â_2 = π/4`。
#[inline]
fn cosine_lobe(order: usize) -> f32 {
    match order {
        0 => PI,
        1 => 2.0 * PI / 3.0,
        2 => PI / 4.0,
        _ => 0.0,
    }
}

/// 基函数索引 → 所属阶 l（0..9：l=0 占 1，l=1 占 3，l=2 占 5）。
#[inline]
fn band_of(index: usize) -> usize {
    match index {
        0 => 0,
        1..=3 => 1,
        _ => 2,
    }
}

/// 实数 SH 基函数的系数常量（与 PBRT 一致）。
const SH_C0: f32 = 0.282_094_79; // 0.5·sqrt(1/π)
const SH_C1: f32 = 0.488_602_51; // 0.5·sqrt(3/π)
const SH_C2_0: f32 = 1.092_548_46; // 0.25·sqrt(15/π)  —— 与 C2_2 相同值
const SH_C2_1: f32 = 0.315_391_57; // 0.25·sqrt(5/π)
const SH_C2_2: f32 = 0.546_274_23; // 0.25·sqrt(15/π)

/// SH 阶数（预算档位）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShOrder {
    /// 仅常数项（1 系数）：各向同性环境，无方向性（最省）。
    L0,
    /// L0 + L1（4 系数）：单调方向性。
    L1,
    /// L0 + L1 + L2（9 系数）：平滑方向性，默认档。
    L2,
}

impl ShOrder {
    /// 该阶使用的系数个数（L0→1，L1→4，L2→9）。
    #[inline]
    pub fn count(self) -> usize {
        match self {
            ShOrder::L0 => 1,
            ShOrder::L1 => 4,
            ShOrder::L2 => 9,
        }
    }
}

/// SH-L2 的 9 个基函数在单位方向 `d` 上的取值。
///
/// `d` 无需归一化（内部不归一，约定调用方传单位向量）。返回固定序：
/// `[Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]`。
#[inline]
pub fn sh_basis_l2(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        SH_C0,
        SH_C1 * y,
        SH_C1 * z,
        SH_C1 * x,
        SH_C2_0 * x * y,
        SH_C2_0 * y * z,
        SH_C2_1 * (3.0 * z * z - 1.0),
        SH_C2_0 * x * z,
        SH_C2_2 * (x * x - y * y),
    ]
}

/// 辐亮度 SH 表示（9 系数，`L2` 满档存储；重建时按 [`ShOrder`] 截断）。
///
/// 系数语义见模块文档：**辐射度投影**（`L_lm = ∫ L·Y_lm dΩ`），余弦瓣卷积
/// 系数 `Â_l` 在 [`reconstruct`](Self::reconstruct) 时才乘。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShIrradiance {
    /// 9 个辐亮度 SH 系数（固定序，见 [`sh_basis_l2`]）。
    pub coeffs: [f32; 9],
}

impl Default for ShIrradiance {
    fn default() -> Self {
        Self { coeffs: [0.0; 9] }
    }
}

impl ShIrradiance {
    /// 全零（无光）系数。
    #[inline]
    pub fn zero() -> Self {
        Self { coeffs: [0.0; 9] }
    }

    /// 从常数辐照度构造（仅 DC 项）。`irradiance` = E，重建后恒为该值。
    ///
    /// 重建式 `E(ω) = Â_0·c0·Y00`（Â_0 = π，Y00 = 0.282095）。欲恒等于 `e`
    /// 需 `c0 = e / (π·Y00)`。这是"该辐照度场对应的辐亮度 DC 系数"。
    #[inline]
    pub fn constant(irradiance: f32) -> Self {
        let mut c = [0.0f32; 9];
        c[0] = irradiance / (PI * SH_C0);
        Self { coeffs: c }
    }

    /// 投影：把离散入射样本 `(direction, radiance)` 积到 SH 系数（**辐射度**）。
    ///
    /// 每个样本按 `L(ω') · Y_i(ω') · dΩ` 累加（**不乘 cos**——余弦在重建时经
    /// `Â_l` 加入）。`d_omega` 为每样本立体角（均匀半球抽样取 `2π/n`，均匀
    /// 全球取 `4π/n`）。累加序固定（样本序 × 系数序）→ 确定性。
    ///
    /// # 参数
    /// - `samples`：`(入射方向单位向量, 该方向辐亮度)` 序列；
    /// - `normal`：**本参数保留用于光照半球裁剪**（仅累积正半球样本更省）；
    ///   传零向量表示"不做半球裁剪，累积全部样本"。为兼容旧调用，此处仍接受
    ///   该参数但**不改变投影语义**（余弦不进投影）；
    /// - `d_omega`：每样本立体角。
    pub fn project(samples: &[(Vec3, Vec3)], normal: Vec3, d_omega: f32) -> Self {
        let cull = normal.length() > 0.5;
        let mut coeffs = [0.0f32; 9];
        for (dir, radiance) in samples {
            if cull && dir.dot(&normal) <= 0.0 {
                continue; // 背向样本不贡献（半球外）
            }
            let y = sh_basis_l2(*dir);
            // 三分量以均值投影（灰度通道；RGB 请用 project_rgb）。
            let lum = (radiance.x + radiance.y + radiance.z) / 3.0;
            for i in 0..9 {
                coeffs[i] += lum * y[i] * d_omega;
            }
        }
        Self { coeffs }
    }

    /// 投影（RGB 分量独立）——每个颜色通道各自一套 9 系数。
    ///
    /// 返回 `[R 系数; 9], [G 系数; 9], [B 系数; 9]`。累加序固定 → 确定性。
    pub fn project_rgb(
        samples: &[(Vec3, Vec3)],
        normal: Vec3,
        d_omega: f32,
    ) -> [[f32; 9]; 3] {
        let cull = normal.length() > 0.5;
        let mut out = [[0.0f32; 9]; 3];
        for (dir, radiance) in samples {
            if cull && dir.dot(&normal) <= 0.0 {
                continue;
            }
            let y = sh_basis_l2(*dir);
            let rgb = [radiance.x, radiance.y, radiance.z];
            for ch in 0..3 {
                for i in 0..9 {
                    out[ch][i] += rgb[ch] * y[i] * d_omega;
                }
            }
        }
        out
    }

    /// 重建：从系数还原指定方向 `d` 的**辐照度**（按 `order` 截断）。
    ///
    /// `E(d) ≈ Σ_lm Â_l · L_lm · Y_lm(d)`（仅启用阶；余弦瓣系数见模块文档）。
    /// 常数环境 L0 下对任意 `d` 均得 `π·L0`。
    #[inline]
    pub fn reconstruct(&self, d: Vec3, order: ShOrder) -> f32 {
        let y = sh_basis_l2(d);
        let n = order.count();
        let mut e = 0.0f32;
        for i in 0..n {
            e += cosine_lobe(band_of(i)) * self.coeffs[i] * y[i];
        }
        e
    }

    /// 探针间平滑（S2 卷积 / 调和分析核）：系数线性插值。
    ///
    /// `t = 0` 取 `self`，`t = 1` 取 `other`；`t` 越界钳到 `[0,1]`。
    /// 逐系数固定序线性混合 → 平滑、无漏光（不引入新方向能量）。
    #[inline]
    pub fn lerp(&self, other: &ShIrradiance, t: f32) -> ShIrradiance {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let mut c = [0.0f32; 9];
        for i in 0..9 {
            c[i] = self.coeffs[i] + (other.coeffs[i] - self.coeffs[i]) * t;
        }
        ShIrradiance { coeffs: c }
    }

    /// 系数加性合并（多探针加权和用）。逐系数固定序 → 确定性。
    #[inline]
    pub fn add_scaled(&mut self, other: &ShIrradiance, s: f32) {
        for i in 0..9 {
            self.coeffs[i] += other.coeffs[i] * s;
        }
    }

    /// 是否含 NaN / Inf（C18 防御：极端输入下作废系数）。
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_finite())
    }
}

/// 辐照度 → 漫反射出射辐亮度：`L_o = albedo · E(n) · (1/π)`。
///
/// 这是 **π 因子的唯一去处**：Lambert BRDF = albedo/π，shading 时 `∫BRDF·L·cos`
/// 的 cos 已含在系数里，故此处只补 1/π。
#[inline]
pub fn sh_irradiance_to_radiance(albedo: Vec3, irradiance: Vec3) -> Vec3 {
    hadamard(albedo, irradiance) * (1.0 / PI)
}

/// RGB 辐照度重建（自由函数形式）：对三套系数各求 `Σ Â_l·L_lm·Y_lm(d)`。
#[inline]
pub fn sh_reconstruct_rgb(rgb: &[[f32; 9]; 3], d: Vec3, order: ShOrder) -> Vec3 {
    let y = sh_basis_l2(d);
    let n = order.count();
    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        let mut e = 0.0f32;
        for i in 0..n {
            e += cosine_lobe(band_of(i)) * rgb[ch][i] * y[i];
        }
        out[ch] = e;
    }
    Vec3::new(out[0], out[1], out[2])
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ctx_seed() -> u64 {
        0x5A17_D00D // 任意常量（防抖动回退）
    }

    /// 确定性：同输入两遍 bit 逐位一致。
    #[test]
    fn project_is_bitwise_deterministic() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let mut samples = Vec::new();
        for i in 0..64u32 {
            let t = (i as f32) * 0.1;
            let d = Vec3::new(t.cos(), t.sin(), 0.3).normalize();
            samples.push((d, Vec3::new(0.5, 0.4, 0.3)));
        }
        let a = ShIrradiance::project(&samples, normal, 0.1);
        let b = ShIrradiance::project(&samples, normal, 0.1);
        for i in 0..9 {
            assert_eq!(a.coeffs[i].to_bits(), b.coeffs[i].to_bits(), "系数 {i} 不确定");
        }
        // seed 常量参与（防未用告警）——保持接口一致。
        let _ = test_ctx_seed();
    }

    /// 正反向一致性：单方向光投影后再沿该方向重建，应显著大于背向重建。
    #[test]
    fn single_direction_sharpens_along_its_direction() {
        // 一个沿 +Y 的入射光（法线 +Y，cos=1）。
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let samples = vec![(dir, Vec3::splat(1.0))];
        // 均匀半球测度近似：单样本立体角取半球面积 2π（粗，但方向对比度足够）。
        let sh = ShIrradiance::project(&samples, normal, 2.0 * PI);
        let e_up = sh.reconstruct(Vec3::new(0.0, 1.0, 0.0), ShOrder::L2);
        let e_down = sh.reconstruct(Vec3::new(0.0, -1.0, 0.0), ShOrder::L2);
        assert!(e_up > e_down, "朝光方向辐照度应更大：{e_up} vs {e_down}");
        assert!(e_up > 0.0, "朝光方向辐照度应为正：{e_up}");
        // 量化断言：L2 下单方向光朝光/背光对比度至少 2×（方向性被正确编码）。
        assert!(e_up > 2.0 * e_down.abs().max(1e-6), "对比度不足：{e_up} / {e_down}");
    }

    /// 能量守恒：常数环境光下重建辐照度 ≈ π·L0（各向同性解析解）。
    #[test]
    fn constant_env_reconstructs_to_pi_l0() {
        // 常数环境 L0 充满**全空间**：投影原始辐亮度（全 4π 均匀求积），
        // 重建用余弦瓣 Â_l。解析锚点：E(ω) = ∫_hemisphere L0 cosθ dω = π·L0
        // （与 ω 无关）。全空间均匀求积：z 均匀 ∈ [−1,1]，φ 网格，dΩ = 4π/n。
        let l0 = 0.7f32;
        let n_z = 64usize;
        let n_phi = 64usize;
        let mut samples = Vec::new();
        for jz in 0..n_z {
            let cos_theta = (jz as f32 + 0.5) / n_z as f32 * 2.0 - 1.0; // ∈ (−1,1)
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            for jp in 0..n_phi {
                let phi = (jp as f32 + 0.5) / n_phi as f32 * 2.0 * PI;
                let (sp, cp) = (phi.sin(), phi.cos());
                let d = Vec3::new(sin_theta * cp, sin_theta * sp, cos_theta);
                samples.push((d, Vec3::splat(l0)));
            }
        }
        // 全空间均匀求积：dΩ = 4π/n。零法线 → 不做半球裁剪（累积全部）。
        let d_omega = 4.0 * PI / samples.len() as f32;
        let sh = ShIrradiance::project(&samples, Vec3::splat(0.0), d_omega);
        let expected = PI * l0;
        // 沿一族方向重建都应 ≈ π·L0（各向同性）。
        for d in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            let e = sh.reconstruct(d, ShOrder::L2);
            let rel = (e - expected).abs() / expected;
            assert!(rel < 0.05, "常数环境重建 {e} vs 解析 π·L0={expected}，相对误差 {rel}");
        }
    }

    /// 档位单调：L2 的方向对比度 ≥ L1 ≥ L0（L0 恒为常数）。
    #[test]
    fn order_monotone_contrast() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let mut samples = Vec::new();
        for i in 0..128 {
            let t = (i as f32 + 0.5) / 128.0;
            let cos_theta = t;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = i as f32 * 2.399_963_2;
            let (sp, cp) = (phi.sin(), phi.cos());
            // 亮度随 cosθ 变化（上亮下暗）。
            let d = Vec3::new(sin_theta * cp, cos_theta, sin_theta * sp);
            let l = 0.2 + 0.8 * cos_theta;
            samples.push((d, Vec3::splat(l)));
        }
        let d_omega = 2.0 * PI / 128.0;
        let sh = ShIrradiance::project(&samples, normal, d_omega);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let down = Vec3::new(0.0, -1.0, 0.0);
        let c0 = (sh.reconstruct(up, ShOrder::L0) - sh.reconstruct(down, ShOrder::L0)).abs();
        let c1 = (sh.reconstruct(up, ShOrder::L1) - sh.reconstruct(down, ShOrder::L1)).abs();
        let c2 = (sh.reconstruct(up, ShOrder::L2) - sh.reconstruct(down, ShOrder::L2)).abs();
        assert!(c0 < 1e-5, "L0 应无方向对比度，实际 {c0}");
        assert!(c1 > c0, "L1 对比度应 > L0：{c1} vs {c0}");
        assert!(c2 >= c1 - 1e-4, "L2 对比度不应低于 L1：{c2} vs {c1}");
    }

    /// SH 正交性自检：不同基函数在球面上的内积 ≈ 0（对角非零）。
    #[test]
    fn basis_is_orthogonal_on_sphere() {
        let n = 8192usize;
        let mut basis = vec![[0.0f32; 9]; n];
        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let cos_theta = 1.0 - 2.0 * t; // 均匀球面
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = i as f32 * 2.399_963_2;
            let (sp, cp) = (phi.sin(), phi.cos());
            let d = Vec3::new(sin_theta * cp, sin_theta * sp, cos_theta);
            basis[i] = sh_basis_l2(d);
        }
        let d_omega = 4.0 * PI / n as f32;
        // Y00 与任一 l>0 基的正交（数值积分）。
        let mut off = 0.0f32;
        for i in 0..n {
            off += basis[i][0] * basis[i][3] * d_omega; // <Y00, Y1x>
        }
        assert!(off.abs() < 0.02, "Y00 与 Y11 内积应≈0，实际 {off}");
        // Y00 自内积 ≈ 1/π 的倒数关系（归一化）：∫Y00² dω = 0.2821²·4π ≈ 1.0。
        let mut self_ip = 0.0f32;
        for i in 0..n {
            self_ip += basis[i][0] * basis[i][0] * d_omega;
        }
        assert!((self_ip - 1.0).abs() < 0.02, "Y00 自内积应≈1，实际 {self_ip}");
    }

    /// 极端输入：NaN 方向、空样本 → 不 panic 且结果有限（C18）。
    #[test]
    fn extreme_inputs_no_panic() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let bad = vec![(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::splat(1.0))];
        let sh = ShIrradiance::project(&bad, normal, 0.1);
        // NaN 方向的 Y_i 含 NaN → 系数可能含 NaN；is_finite() 如实报告，
        // 关键是不 panic（此处仅断言调用成功且 is_finite 可判定）。
        let _ = sh.is_finite();
        // 空样本 → 全零。
        let empty = ShIrradiance::project(&[], normal, 0.1);
        assert_eq!(empty, ShIrradiance::zero());
        // 退化法线（零向量）→ 不做半球裁剪（cull=false），仍可投影不 panic。
        let zer = ShIrradiance::project(
            &[(Vec3::new(0.0, 1.0, 0.0), Vec3::splat(1.0))],
            Vec3::splat(0.0),
            0.1,
        );
        assert!(zer.is_finite(), "合法样本 + 退化法线应得到有限系数");
        // lerp 的 NaN t → 钳到 0（取 self）。
        let a = ShIrradiance::constant(1.0);
        let b = ShIrradiance::constant(2.0);
        let c = a.lerp(&b, f32::NAN);
        assert_eq!(c.coeffs[0].to_bits(), a.coeffs[0].to_bits());
    }

    /// 常数构造 + 重建自洽（DC 项语义正确）。
    #[test]
    fn constant_round_trip() {
        let e = 3.3f32;
        let sh = ShIrradiance::constant(e);
        for d in [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            let r = sh.reconstruct(d, ShOrder::L0);
            assert!((r - e).abs() < 1e-5, "常数重建失配：{r} vs {e}");
        }
    }

    /// 测试自证非恒真：手算重建（含余弦瓣 Â_l）应逐位等于接口重建——
    /// 证明测试确实在检验乘积和，而非恒真。
    #[test]
    fn test_not_vacuous_wrong_basis_differs() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let samples = vec![(Vec3::new(0.0, 1.0, 0.0), Vec3::splat(1.0))];
        let sh = ShIrradiance::project(&samples, normal, 1.0);
        let y = sh_basis_l2(Vec3::new(0.0, 1.0, 0.0));
        let mut manual = 0.0f32;
        for i in 0..9 {
            manual += cosine_lobe(band_of(i)) * sh.coeffs[i] * y[i];
        }
        assert_eq!(
            manual.to_bits(),
            sh.reconstruct(Vec3::new(0.0, 1.0, 0.0), ShOrder::L2).to_bits()
        );
        // 反例检查：若把余弦瓣去掉（错误实现），重建值应显著不同——
        // 证明 Â_l 因子确实参与运算（非可有可无）。
        let mut wrong = 0.0f32;
        for i in 0..9 {
            wrong += sh.coeffs[i] * y[i];
        }
        assert!(
            (wrong - manual).abs() > 1e-6,
            "去掉余弦瓣后结果不应等同（Â_l 必须在起作用）"
        );
    }
}
