//! 场景附加能力的**请求侧 DTO**（T0 产品路径接线）。
//!
//! # 为什么这些类型在 `prism-gpu`
//!
//! 本 crate 拥有一切"请求侧"DTO（[`crate::CameraReq`] / [`crate::LightReq`] /
//! [`crate::SphereReq`] / [`crate::CurveReq`]）—— 它们是 `prism-core` 的高层
//! 描述与三后端之间的**唯一交接面**。附加能力（介质 / 网格 / 实例 / 纹理 /
//! 毛发 / 位移槽）沿用同一位置，理由同 [`crate::SphereReq`]：`prism-core` 不能
//! 依赖 `prism-gpu`（历史上正是为此把 `build_render_request` 下沉到
//! `prism-render`），`prism-gpu` 也不能反引 `prism-core`（会成环），故请求侧
//! 必须有一份自己的同构影子类型。
//!
//! 语义权威在 `prism-core::attach`（用户/DSL 写的）与 `prism-rt::Scene`（算的）。
//! 本模块**只描述**，任何一处语义分歧都以那两端为准。
//!
//! # 空 ⇒ 逐字节同帧
//!
//! [`SceneGeoReq::default()`] 全空。全部列表为空时 `translate_scene` 不产生任何
//! 新的 rt 侧条目，渲染路径与加字段前逐字节相同（判据 P1）。

use prism_math::Vec3;

/// 介质盒（请求侧）。语义见 `prism_core::attach::MediumBox`（同一组字段）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediumBoxReq {
    /// 盒的最小角。
    pub bounds_min: Vec3,
    /// 盒的最大角。
    pub bounds_max: Vec3,
    /// 总消光系数 σt（1/长度，正）。
    pub sigma_t: f32,
    /// 单次散射反照率 ∈ [0, 1]。
    pub albedo: f32,
    /// HG 各向异性 ∈ (−1, 1)。
    pub g: f32,
    /// 重叠时的优先级（大者胜）。
    pub priority: u32,
}

/// 位移材质槽（请求侧）。语义见 `prism_core::attach::DisplacementSlot`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplaceSlotReq {
    /// 高度图句柄（1 起；0 = 无贴图）。
    pub texture: u32,
    /// 幅值（世界单位；恰 0 ⇒ 整条通路短路）。
    pub amplitude: f32,
    /// uv 缩放。
    pub scale: f32,
    /// 细分级别（有上界，翻译层截断）。
    pub levels: u32,
}

impl Default for DisplaceSlotReq {
    fn default() -> Self {
        Self { texture: 0, amplitude: 0.0, scale: 1.0, levels: 0 }
    }
}

impl DisplaceSlotReq {
    /// 是否恒等（幅值恰为 0 / ±0）。
    pub fn is_identity(&self) -> bool {
        self.amplitude == 0.0
    }
}

/// 索引三角网格（请求侧）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshReq {
    /// 顶点位置。
    pub positions: Vec<Vec3>,
    /// 三角形索引流。
    pub indices: Vec<u32>,
    /// 顶点域 uv（空 = 无）。
    pub uv: Vec<[f32; 2]>,
    /// 位移材质槽（`None` = 无）。
    pub displacement: Option<DisplaceSlotReq>,
    /// 折痕边 `(a, b, w)`（G-9；`w ∈ [0,1]`）。
    pub crease: Vec<(u32, u32, f32)>,
    /// 角点顶点（G-9）。
    pub corners: Vec<u32>,
}

/// 几何实例（请求侧）：3×4 仿射行主序 + 共享几何下标 + 材质句柄。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceReq {
    /// 共享几何原型下标。
    pub geom_id: u32,
    /// 3×4 仿射（行主序）。
    pub affine: [f32; 12],
    /// 材质覆盖句柄；`u32::MAX` = 取自几何自身。
    pub mat_id: u32,
}

/// 位图纹理（请求侧，RGBA8）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextureReq {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 是否 sRGB 编码域。
    pub srgb: bool,
    /// RGBA8 像素流（`w*h*4`）。
    pub rgba8: Vec<u8>,
}

/// 毛发散射参数（请求侧）。字段与 `prism_rt::hair_scatter::HairScatter` 对齐。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairReq {
    /// 纵向粗糙度 β_m。
    pub beta_m: f32,
    /// 方位粗糙度 k。
    pub k_azimuth: f32,
    /// 纤维折射率 η。
    pub eta: f32,
    /// 纤维内吸收系数。
    pub sigma_a: f32,
    /// 纤维半径。
    pub radius: f32,
    /// R 分量权重。
    pub w_r: f32,
    /// TT 分量权重。
    pub w_tt: f32,
    /// TRT 分量权重。
    pub w_trt: f32,
    /// TT 方位偏移。
    pub phi_tt_shift: f32,
}

impl Default for HairReq {
    fn default() -> Self {
        Self {
            beta_m: 0.22,
            k_azimuth: 3.0,
            eta: 1.55,
            sigma_a: 0.8,
            radius: 0.02,
            w_r: 0.35,
            w_tt: 0.55,
            w_trt: 0.10,
            phi_tt_shift: 0.0,
        }
    }
}

/// 单股毛发（请求侧）：折线 + 逐点半径 + 散射参数。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StrandReq {
    /// 折线控制点（≥ 2）。
    pub points: Vec<Vec3>,
    /// 逐点半径（空 = 统一取翻译层缺省）。
    pub radii: Vec<f32>,
    /// 散射参数。
    pub params: HairReq,
}

/// 场景附加能力的请求侧聚合（[`crate::SceneRenderRequest::geo`] 的类型）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneGeoReq {
    /// 介质区域（A6）。
    pub media: Vec<MediumBoxReq>,
    /// 直接进场景的三角网格。
    pub meshes: Vec<MeshReq>,
    /// 实例的共享几何原型（**不**直接进场景）。
    pub instance_geoms: Vec<MeshReq>,
    /// 实例列表。
    pub instances: Vec<InstanceReq>,
    /// 位图纹理（句柄 1 起，按顺序编号）。
    pub textures: Vec<TextureReq>,
    /// 毛发股。
    pub strands: Vec<StrandReq>,
}

impl SceneGeoReq {
    /// 是否全空（翻译层快路径判据）。
    pub fn is_empty(&self) -> bool {
        self.media.is_empty()
            && self.meshes.is_empty()
            && self.instance_geoms.is_empty()
            && self.instances.is_empty()
            && self.textures.is_empty()
            && self.strands.is_empty()
    }

    /// 条目合计（诊断用）。
    pub fn entry_count(&self) -> usize {
        self.media.len()
            + self.meshes.len()
            + self.instance_geoms.len()
            + self.instances.len()
            + self.textures.len()
            + self.strands.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 缺省必须全空（P1 的结构性前提）。
    #[test]
    fn default_geo_is_empty() {
        let g = SceneGeoReq::default();
        assert!(g.is_empty());
        assert_eq!(g.entry_count(), 0);
    }

    /// 位移槽恒等判据（±0 都算 0）。
    #[test]
    fn displace_slot_identity() {
        assert!(DisplaceSlotReq::default().is_identity());
        assert!(DisplaceSlotReq { amplitude: -0.0, ..Default::default() }.is_identity());
        assert!(!DisplaceSlotReq { amplitude: 0.5, ..Default::default() }.is_identity());
    }
}
