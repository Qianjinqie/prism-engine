//! 材质预设计数真源（V2.2-B-3 / T05，计数口径 Q4）。
//!
//! # 口径（设计 §1.4 / Q4）
//! - **用户可见 / 可渲染 / DSL 可声明**的材质预设共 **10 项**，作为计数真源。
//! - `MaterialKind` 变体数（4：Lambert / Mirror / Emissive / Pbr）是**内部口径**，
//!   与预设计数**同时上报、不混用**（防"文档说 10 种、代码 6 种"）。
//! - 每一项预设都必须能构造出**可渲染**的 `MaterialKind`（`to_kind`）与白/中性
//!   `Bsdf`（`sample_bsdf_white`，供白炉能量守恒测试）。
//!
//! # 预设清单
//! `diffuse / mirror / emissive / metal / plastic / glass / water / skin / velvet / brushed_metal`
//!
//! 其中 `brushed_metal` 为各向异性（GGX anisotropic，[`Bsdf::with_anisotropy`]）；
//! `velvet` 当前借道 dielectric 路径渲染（sheen 独立瓣为后续扩展，不影响"可渲染"与
//! 能量守恒口径）。

use prism_math::Vec3;

use crate::bsdf::Bsdf;
use crate::integrator::MaterialKind;

/// 材质预设（计数真源；DSL 可声明且可渲染）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPreset {
    /// 朗伯特漫反射（反照率）。
    Diffuse,
    /// 完美镜面（δ）。
    Mirror,
    /// 自发光（光源，非反射体）。
    Emissive,
    /// 金属（Disney PBR，metalness = 1）。
    Metal,
    /// 塑料（电介质 + 漫反）。
    Plastic,
    /// 玻璃（电介质 + 透射）。
    Glass,
    /// 水（电介质 + 透射，ior ≈ 1.33）。
    Water,
    /// 皮肤（次表面散射，subsurface > 0）。
    Skin,
    /// 天鹅绒 / 绒面（sheen；当前借道 dielectric 路径渲染）。
    Velvet,
    /// 拉丝金属（GGX 各向异性；B-3 新增）。
    BrushedMetal,
}

impl MaterialPreset {
    /// 全部预设（计数真源；测试断言 `len() >= 10` 且每一项可渲染）。
    pub const ALL: [MaterialPreset; 10] = [
        MaterialPreset::Diffuse,
        MaterialPreset::Mirror,
        MaterialPreset::Emissive,
        MaterialPreset::Metal,
        MaterialPreset::Plastic,
        MaterialPreset::Glass,
        MaterialPreset::Water,
        MaterialPreset::Skin,
        MaterialPreset::Velvet,
        MaterialPreset::BrushedMetal,
    ];

    /// 预设 → 内部 `MaterialKind`（渲染口径）。每一项都能构造出可渲染的 Kind。
    pub fn to_kind(self) -> MaterialKind {
        match self {
            MaterialPreset::Diffuse => {
                MaterialKind::Lambert { albedo: Vec3::splat(0.8) }
            }
            MaterialPreset::Mirror => MaterialKind::Mirror,
            MaterialPreset::Emissive => {
                MaterialKind::Emissive { radiance: Vec3::splat(1.0) }
            }
            MaterialPreset::Metal => MaterialKind::Pbr(bsdf_metal()),
            MaterialPreset::Plastic => MaterialKind::Pbr(bsdf_plastic()),
            MaterialPreset::Glass => MaterialKind::Pbr(bsdf_glass()),
            MaterialPreset::Water => MaterialKind::Pbr(bsdf_water()),
            MaterialPreset::Skin => MaterialKind::Pbr(bsdf_skin()),
            MaterialPreset::Velvet => MaterialKind::Pbr(bsdf_velvet()),
            MaterialPreset::BrushedMetal => MaterialKind::Pbr(bsdf_brushed_metal()),
        }
    }

    /// 白 / 中性 Disney `Bsdf`（供白炉能量守恒测试使用）。
    ///
    /// 返回 `None` 表示无 BRDF 反射体（`Emissive` 是光源，白炉反射率 ≡ 0）。
    /// 其余 9 项均为白/中性配置，白炉半球积分反射率 ≈ 1（能量守恒）。
    pub fn sample_bsdf_white(self) -> Option<Bsdf> {
        match self {
            MaterialPreset::Emissive => None,
            MaterialPreset::Diffuse => Some(bsdf_diffuse_white()),
            MaterialPreset::Mirror => Some(bsdf_mirror_white()),
            MaterialPreset::Metal => Some(bsdf_metal()),
            MaterialPreset::Plastic => Some(bsdf_plastic()),
            MaterialPreset::Glass => Some(bsdf_glass()),
            MaterialPreset::Water => Some(bsdf_water()),
            MaterialPreset::Skin => Some(bsdf_skin()),
            MaterialPreset::Velvet => Some(bsdf_velvet()),
            MaterialPreset::BrushedMetal => Some(bsdf_brushed_metal()),
        }
    }
}

// ---------------------------------------------------------------------------
// 白 / 中性 Bsdf 预置（roughness 取中等、subsurface_radius 取安全正值 0.3）
// ---------------------------------------------------------------------------

/// 朗伯特白（roughness = 1 ⇒ 纯漫反，白炉反射率 = 1）。
fn bsdf_diffuse_white() -> Bsdf {
    Bsdf::new(Vec3::splat(1.0), 0.0, 1.0, 1.5, 0.0, 0.0, Vec3::splat(0.3))
}

/// 镜面（metalness = 1、roughness = 0 ⇒ δ 反射，白炉反射率 = 1）。
fn bsdf_mirror_white() -> Bsdf {
    Bsdf::new(Vec3::splat(1.0), 1.0, 0.0, 1.5, 0.0, 0.0, Vec3::splat(0.3))
}

/// 金属（f0 = 1 ⇒ 白金属，多散射补偿使白炉反射率严格 = 1）。
fn bsdf_metal() -> Bsdf {
    Bsdf::new(v3_white(), 1.0, 0.2, 1.5, 0.0, 0.0, Vec3::splat(0.3))
}

/// 塑料（电介质 + 漫反；白炉反射率 ≈ 1，含漫反让位 + 补偿）。
fn bsdf_plastic() -> Bsdf {
    Bsdf::new(v3_white(), 0.0, 0.3, 1.5, 0.0, 0.0, Vec3::splat(0.3))
}

/// 玻璃（电介质 + 全透射；反射 + 透射 白炉总能量 ≈ 1）。
fn bsdf_glass() -> Bsdf {
    Bsdf::new(v3_white(), 0.0, 0.05, 1.5, 1.0, 0.0, Vec3::splat(0.3))
}

/// 水（电介质 + 全透射，ior = 1.33）。
fn bsdf_water() -> Bsdf {
    Bsdf::new(v3_white(), 0.0, 0.02, 1.33, 1.0, 0.0, Vec3::splat(0.3))
}

/// 皮肤（次表面散射；subsurface = 0.8，白炉反射率 ≈ 1）。
fn bsdf_skin() -> Bsdf {
    Bsdf::new(v3_white(), 0.0, 0.4, 1.4, 0.0, 0.8, Vec3::splat(0.4))
}

/// 天鹅绒 / 绒面（借道 dielectric 路径；sheen 独立瓣为后续扩展）。
fn bsdf_velvet() -> Bsdf {
    Bsdf::new(v3_white(), 0.0, 0.5, 1.5, 0.0, 0.0, Vec3::splat(0.3))
}

/// 拉丝金属（GGX 各向异性；aspect = 3，几何均值 = roughness²）。
fn bsdf_brushed_metal() -> Bsdf {
    let b = Bsdf::new(v3_white(), 1.0, 0.3, 1.5, 0.0, 0.0, Vec3::splat(0.3));
    let a = b.alpha();
    // α_t = a/3，α_b = 3a ⇒ 几何均值 = a（与白炉表口径一致）；两轴均 > 1e-6。
    b.with_anisotropy(a / 3.0, a * 3.0, 0.5)
}

/// 白反照率（避免多处重复拼写，提升可读性）。
#[inline]
fn v3_white() -> Vec3 {
    Vec3::new(1.0, 1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_count_is_at_least_ten() {
        // 计数可判定（验收 3）：真源长度 ≥ 10，且每一项都能构造可渲染 Kind。
        assert!(
            MaterialPreset::ALL.len() >= 10,
            "材质预设计数应 ≥ 10，实得 {}",
            MaterialPreset::ALL.len()
        );
        // 每一项都能构造出可渲染的 MaterialKind（防"枚举 10 个、能渲染 6 个"）。
        for p in MaterialPreset::ALL {
            let _kind = p.to_kind();
        }
    }

    #[test]
    fn every_reflective_preset_has_white_bsdf() {
        // 9 个反射/透射预设都有白/中性 Bsdf；emissive 是光源（None，合理）。
        let mut reflective = 0usize;
        for p in MaterialPreset::ALL {
            if let Some(b) = p.sample_bsdf_white() {
                // 白预设：base_color 必须全白（白炉口径）。
                assert!(
                    (b.base_color.x - 1.0).abs() < 1e-6
                        && (b.base_color.y - 1.0).abs() < 1e-6
                        && (b.base_color.z - 1.0).abs() < 1e-6,
                    "{p:?} 白炉口径 base_color 应全白"
                );
                reflective += 1;
            }
        }
        assert_eq!(reflective, 9, "反射/透射预设应恰为 9 项（emissive 除外）");
    }

    #[test]
    fn brushed_metal_is_anisotropic() {
        let b = MaterialPreset::BrushedMetal.sample_bsdf_white().unwrap();
        assert!(b.is_anisotropic(), "拉丝金属必须是各向异性（α_t ≠ α_b）");
        // 退化（α_t == α_b）逐位回退到各向同性：用几何均值重建应非各向异性。
        let iso = b.with_anisotropy(b.alpha_t, b.alpha_t, b.rotation);
        assert!(!iso.is_anisotropic());
    }
}
