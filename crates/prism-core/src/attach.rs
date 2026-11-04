//! 场景附加能力（T0 产品路径接线）：介质 / 网格 / 实例 / 纹理 / 毛发 / 位移槽。
//!
//! # 为什么需要这个模块（缺口原文）
//!
//! [`crate::Scene`] 原本只有 `radius / orbit_center / orbit_radius / material /
//! lights / camera / aesthetic` 七个字段 —— 恰好够描述"一个环绕的红球"。W1–W7
//! 补进来的纹理（A1）、介质（A6）、实例（A3）、毛发（A7）、细分（G-9）、位移
//! （G-10）、自适应（A5）**一条都出不了产品片**，因为产品出片的唯一通路是
//! `prism_core::Scene` → `SceneRenderRequest` → `translate_scene` →
//! `prism_rt::Scene`，而这条链上只有球和方向光。
//!
//! # 设计纪律：只做"描述"，不自创语义
//!
//! 本模块的每个类型都必须**逐字段对得上** `prism_rt::Scene` 已存在的能力
//! （`crates/prism-rt/src/integrator.rs:875` 起）。core 侧不能依赖 `prism-rt`
//! （依赖方向是 `prism-rt → prism-core`），所以这里是"同构的影子描述"，由
//! `prism-render::rt_backend::translate_scene` 做一次搬运。**任何 rt 侧没有的
//! 能力不许在这里先长出来** —— 那会让翻译层出现"永远返回 `Err` 的字段"。
//!
//! | core 类型 | 对齐的 rt 能力 | 搬运落点 |
//! |---|---|---|
//! | [`MediumBox`] | [`prism_rt::integrator::MediumRegion`]（A6 互斥区域 + priority） | `Scene::media` |
//! | [`MeshDesc`] | `Scene::add_mesh` / `Mesh::{positions,indices,uv}` | `Scene::meshes` |
//! | [`InstanceDesc`] | `prism_rt::instance::Instance`（112 B 槽，3×4 仿射 + 逆） | `Scene::instances` + `instance_geoms` |
//! | [`TextureDesc`] | `Scene::add_texture_rgba8` | `Scene::textures` + `texels` |
//! | [`StrandDesc`] | `CurveReq::Strand`（折线 + 逐点半径）+ `MaterialKind::Hair` | 每股一个 `Mesh` |
//! | [`DisplacementSlot`] | `MaterialKind::PbrDisplaced`（G-10 运行期槽） | 网格在装配期细分 + 位移 |
//!
//! # 空 ⇒ 逐字节同帧（这是硬约束，不是期望）
//!
//! [`SceneAttach`] 的 `Default` 是**全空**。当六个列表都为空、且没有任何网格
//! 挂位移槽时，翻译层**不产生任何新的 rt 侧条目**，于是渲染路径与加字段前
//! 走的是同一串浮点运算、同一串采样索引 —— 输出逐字节相同。
//! 判据：`crates/prism-render/tests/t0_product_path.rs` 的 P1（md5 对账）。
//!
//! # 类型纪律
//!
//! 只用 `f32` / `u32` / `u8` / `[f32; N]` / `Vec`，与 `prism-math` 的 `Vec3`。
//! 不引入任何第三方 crate（workspace 红线）。

use prism_math::Vec3;

/// 均匀介质盒（A6 / R-7 的"互斥区域"；与 `prism_rt::MediumRegion` 同构）。
///
/// # 语义（与 rt 侧**逐字一致**，不在 core 侧重定义）
///
/// 区域是轴对齐盒 `[bounds_min, bounds_max]`，装一个**均匀**介质
/// （`sigma_t` / `albedo` / `g`）。多个区域**不得重叠**（路线图 §7 定稿：
/// 重叠是规格违规，不是能力缺口）；重叠输入只保证 `select_region` 全序确定。
///
/// `priority` 大者胜 —— 但它**只在重叠时才说话**（见 rt 侧字段文档）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediumBox {
    /// 盒的最小角。
    pub bounds_min: Vec3,
    /// 盒的最大角。
    pub bounds_max: Vec3,
    /// 总消光系数 σt = σa + σs（1/长度，必须为正）。
    pub sigma_t: f32,
    /// 单次散射反照率 σs/σt ∈ [0, 1]。
    pub albedo: f32,
    /// HG 各向异性 g ∈ (−1, 1)：`g > 0` 前向散射。
    pub g: f32,
    /// 重叠时的优先级（大者胜；缺省 0）。
    pub priority: u32,
}

impl MediumBox {
    /// 以盒 + 介质参数构造（`priority = 0`）。
    pub fn new(
        bounds_min: Vec3,
        bounds_max: Vec3,
        sigma_t: f32,
        albedo: f32,
        g: f32,
    ) -> Self {
        Self { bounds_min, bounds_max, sigma_t, albedo, g, priority: 0 }
    }

    /// 指定优先级（链式）。
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// 参数是否可翻译（有限、`min <= max`、`sigma_t > 0`、`albedo ∈ [0,1]`、
    /// `|g| < 1`）。
    ///
    /// 翻译层用它做**显式跳过**：非法输入不 panic、也不喂给 rt 侧断言。
    pub fn is_translatable(&self) -> bool {
        let f = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        f(self.bounds_min)
            && f(self.bounds_max)
            && self.bounds_min.x <= self.bounds_max.x
            && self.bounds_min.y <= self.bounds_max.y
            && self.bounds_min.z <= self.bounds_max.z
            && self.sigma_t.is_finite()
            && self.sigma_t > 0.0
            && self.albedo.is_finite()
            && (0.0..=1.0).contains(&self.albedo)
            && self.g.is_finite()
            && self.g > -1.0
            && self.g < 1.0
    }
}

/// 位移材质槽（G-10 / T2：**运行期**通路）。
///
/// # 为什么挂在网格条目上而不是 `prism_core::Material` 上
///
/// 位移需要 `uv` 与拓扑（顶点域 uv ⇒ 共享边无裂缝，这是 V4 判据的结构性前提）。
/// 一个只有 `albedo/roughness/metalness` 的 `prism_core::Material` 不带这两样，
/// 把槽挂在那里等于让"有没有 uv"变成运行期要检查的第二件事。挂在
/// [`MeshDesc`] 上，槽的存在本身就蕴含"这个网格有 uv 与拓扑"。
///
/// rt 侧的对应物是 `MaterialKind::PbrDisplaced { bsdf, slot }`（同样只能挂在
/// 网格材质上）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementSlot {
    /// 高度图句柄（**1 起**；`0` = 无贴图 ⇒ 幅值恒 0 ⇒ 构造性短路）。
    ///
    /// 口径与 `prism_rt::texture::NO_TEXTURE` 一致，不在 core 侧重定义。
    pub texture: u32,
    /// 幅值（世界单位）。**恰为 `0.0` ⇒ 整条运行期细分 + 位移通路短路**
    /// （判据 V1：与"没有位移槽"逐位相同）。
    pub amplitude: f32,
    /// uv 缩放（位移采样前的 uv 乘子；`1.0` = 原样）。
    pub scale: f32,
    /// 细分级别（运行期 tessellation 的级数；`0` = 只位移不改拓扑）。
    ///
    /// 有上界（`prism_geometry::subdiv::MAX_LEVELS`），翻译层会截断。
    pub levels: u32,
}

/// 位移槽的缺省：无贴图、零幅值（= 恒等）。
impl Default for DisplacementSlot {
    fn default() -> Self {
        Self { texture: 0, amplitude: 0.0, scale: 1.0, levels: 0 }
    }
}

impl DisplacementSlot {
    /// 构造。
    pub fn new(texture: u32, amplitude: f32, scale: f32, levels: u32) -> Self {
        Self { texture, amplitude, scale, levels }
    }

    /// 是否恒等（幅值恰为 0 / 非有限 ⇒ 恒等）。
    ///
    /// **`0.0` 与 `-0.0` 都算恒等**：`-0.0` 在 IEEE 下与 `0.0` 比较相等，
    /// 且"向内位移 0"物理上就是不动。判据 V1 靠这条短路。
    pub fn is_identity(&self) -> bool {
        self.amplitude == 0.0
    }
}

/// 索引三角网格描述（与 `prism_rt::Scene::add_mesh` 的入参一一对应）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshDesc {
    /// 顶点位置流。
    pub positions: Vec<Vec3>,
    /// 三角形索引流（每 3 个一三角形）。
    pub indices: Vec<u32>,
    /// 顶点域 uv（`[u, v]`，与 `positions` 同长同序；空 = 无 uv）。
    pub uv: Vec<[f32; 2]>,
    /// 位移材质槽（`None` = 无位移）。
    pub displacement: Option<DisplacementSlot>,
    /// 折痕边 `(a, b, w)`（G-9；`w ∈ [0,1]`，`1` = 永不衰减的硬边）。
    ///
    /// # 为什么放在网格上而不是位移槽上
    ///
    /// 折痕是**几何属性**（"这条边是硬边"），不是材质属性。它只在细分发生时
    /// 生效，而运行期细分目前**只**由位移槽触发（`tessellate_displace`）——
    /// 故没有位移槽时本字段不产生任何几何影响，这是口径而不是缺陷：
    /// 细分本身没发生，谈"细分曲面的折痕"没有意义。
    pub crease: Vec<(u32, u32, f32)>,
    /// 角点顶点（G-9；极限位置等于自身）。
    pub corners: Vec<u32>,
}

impl MeshDesc {
    /// 无 uv、无位移槽的网格。
    pub fn new(positions: Vec<Vec3>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            indices,
            uv: Vec::new(),
            displacement: None,
            crease: Vec::new(),
            corners: Vec::new(),
        }
    }

    /// 带顶点域 uv 的网格。
    pub fn with_uv(positions: Vec<Vec3>, indices: Vec<u32>, uv: Vec<[f32; 2]>) -> Self {
        Self { positions, indices, uv, displacement: None, crease: Vec::new(), corners: Vec::new() }
    }

    /// 挂位移材质槽（链式）。
    pub fn with_displacement(mut self, slot: DisplacementSlot) -> Self {
        self.displacement = Some(slot);
        self
    }

    /// 标一条折痕边（链式；`w` 钳到 `[0,1]`，非有限 ⇒ `0`）。
    pub fn with_crease(mut self, a: u32, b: u32, w: f32) -> Self {
        if a != b {
            let w: f32 = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
            self.crease.push((a, b, w));
        }
        self
    }

    /// 标一个角点（链式；重复幂等）。
    pub fn with_corner(mut self, v: u32) -> Self {
        if !self.corners.contains(&v) {
            self.corners.push(v);
        }
        self
    }

    /// 是否有任何折痕/角点标记。
    pub fn has_marks(&self) -> bool {
        !self.corners.is_empty() || self.crease.iter().any(|c| c.2 != 0.0)
    }

    /// 三角形数（`indices.len() / 3`）。
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// 是否可翻译：至少一个非退化三角形，且顶点全有限。
    pub fn is_translatable(&self) -> bool {
        if self.tri_count() == 0 {
            return false;
        }
        self.positions
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
    }
}

/// 几何实例描述（与 `prism_rt::instance::Instance` 的语义一一对应）。
///
/// `affine` 是 **3×4 行主序**（`[m00,m01,m02,tx, m10,…]`）。逆矩阵**不在 core
/// 侧给**：rt 侧 `Instance::from_affine` 是唯一构造入口，它在 CPU 算一次逆，
/// 三后端读同一份字节（"禁止各后端各自求逆"，见 rt 侧模块文档）。core 只给
/// 仿射，正是为了不在这一层制造第二个事实源。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceDesc {
    /// 共享几何原型下标（索引进 [`SceneAttach::instances`] 所引用网格表）。
    pub geom_id: u32,
    /// 3×4 仿射（行主序，12 个 f32）。
    pub affine: [f32; 12],
    /// 材质覆盖句柄；`u32::MAX` = 材质取自共享几何自身。
    ///
    /// 与 `prism_rt::instance::INSTANCE_MAT_FROM_GEOM` 同值（u32::MAX），
    /// 不在 core 侧另立哨兵。
    pub mat_id: u32,
}

impl InstanceDesc {
    /// 恒等实例（`M = I`）。
    pub const fn identity(geom_id: u32) -> Self {
        Self {
            geom_id,
            affine: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            mat_id: u32::MAX,
        }
    }

    /// 平移实例。
    pub const fn translated(geom_id: u32, t: Vec3) -> Self {
        let mut a: [f32; 12] = InstanceDesc::identity(geom_id).affine;
        a[3] = t.x;
        a[7] = t.y;
        a[11] = t.z;
        Self { geom_id, affine: a, mat_id: u32::MAX }
    }

    /// 仿射的 12 个分量是否全部有限。
    pub fn is_translatable(&self) -> bool {
        self.affine.iter().all(|v| v.is_finite())
    }
}

/// 位图纹理描述（RGBA8；`srgb` 决定 rt 侧是否走 sRGB 解码 LUT）。
///
/// 与 `prism_rt::Scene::add_texture_rgba8(w, h, srgb, pixels)` 同参。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextureDesc {
    /// 宽（1 起；0 ⇒ 不可翻译）。
    pub width: u32,
    /// 高（1 起；0 ⇒ 不可翻译）。
    pub height: u32,
    /// 数据是否为 sRGB 编码域（`true` ⇒ rt 侧一次性解码到线性域）。
    pub srgb: bool,
    /// RGBA8 像素流（`width × height × 4` 字节）。
    pub rgba8: Vec<u8>,
}

impl TextureDesc {
    /// 构造。
    pub fn new(width: u32, height: u32, srgb: bool, rgba8: Vec<u8>) -> Self {
        Self { width, height, srgb, rgba8 }
    }

    /// 尺寸与像素流是否自洽（非零、长度恰为 `w*h*4`）。
    pub fn is_translatable(&self) -> bool {
        let need = (self.width as u64) * (self.height as u64) * 4;
        self.width > 0 && self.height > 0 && need == self.rgba8.len() as u64
    }
}

/// 毛发散射参数（与 `prism_rt::hair_scatter::HairScatter` **同字段同序**）。
///
/// 只搬运**参数**；三瓣数学在 `hair_scatter.rs`，本模块一行不碰。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairParams {
    /// 纵向粗糙度 β_m ∈ (0, 1]（M(θ) 高斯宽度）。
    pub beta_m: f32,
    /// 方位粗糙度 k ≥ 0（N(φ) 的 I₀ 集中度）。
    pub k_azimuth: f32,
    /// 纤维折射率 η（人发 ~1.55）。
    pub eta: f32,
    /// 纤维内吸收系数（Beer；0 = 白纤维）。
    pub sigma_a: f32,
    /// 纤维半径（世界单位）。
    pub radius: f32,
    /// R 分量权重。
    pub w_r: f32,
    /// TT 分量权重。
    pub w_tt: f32,
    /// TRT 分量权重。
    pub w_trt: f32,
    /// TT 分量的方位偏移（Marschner 的 φ_shift）。
    pub phi_tt_shift: f32,
}

impl Default for HairParams {
    /// 人发量级的常用起点（与 `HairScatter::new` 的断言域相容）。
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

/// 单股毛发（折线 + 逐点半径）——与 `CurveReq::Strand { points, radii }` 同构。
///
/// # 为什么"一股一条"而不是"一束一条"（T3 的选型理由）
///
/// `MaterialKind::Hair` 携带的是**一个**纤维标架（切向 + 法平面参考）。同向股束
/// 共用一个标架是精确的；**异向**股束共用一个标架则只有一股精确。
///
/// 两条修法：
///
/// 1. **按股切 Mesh**（本模块采用）：翻译层每股建一个 `Mesh`，标架由该股自己的
///    切向算。命中路径不变（仍是一个 Mesh 一个材质），`hair_scatter.rs` 与
///    `surface_brdf` **一行不改**；代价是每股一次 BVH 构建（股数 × log 段数）。
/// 2. 按股存标架、命中时查表：要把标架塞进命中结果（`ShapeRef` → 每股帧），
///    触及 `intersect` 的返回类型、`surface_brdf` 的签名与三后端 ABI。为了
///    一个"每股一个 3×3"的量改动命中数据流，收益与风险不成比例。
///
/// 选 1：**精确**（不是近似）、**零数学改动**、**零 ABI 改动**。这条在报告
/// §28 里记为 T3 的实现选择与理由。
#[derive(Debug, Clone, PartialEq)]
pub struct StrandDesc {
    /// 折线控制点（≥ 2 个；段数 = `len - 1`）。
    pub points: Vec<Vec3>,
    /// 逐点半径（与 `points` 同长；空 = 全部取 [`StrandDesc::DEFAULT_RADIUS`]）。
    pub radii: Vec<f32>,
    /// 散射参数。
    pub params: HairParams,
}

impl StrandDesc {
    /// 半径缺省值（世界单位；与 `curve_tess` 的毛股缺省同量级）。
    pub const DEFAULT_RADIUS: f32 = 0.02;

    /// 以折线 + 统一半径构造。
    pub fn new(points: Vec<Vec3>, radius: f32) -> Self {
        let radii = vec![radius; points.len()];
        Self { points, radii, params: HairParams::default() }
    }

    /// 以折线 + 逐点半径构造。
    pub fn with_radii(points: Vec<Vec3>, radii: Vec<f32>) -> Self {
        Self { points, radii, params: HairParams::default() }
    }

    /// 指定散射参数（链式）。
    pub fn with_params(mut self, params: HairParams) -> Self {
        self.params = params;
        self
    }

    /// 该股的**单位切向**（首尾方向；退化 ⇒ `+Y`）。
    ///
    /// 这就是 T3 "按股标架" 的标架来源：每股用它自己的切向构造
    /// `HairGeometry::new(tangent, normal_ref)`，于是异向股束里每一股都精确。
    pub fn tangent(&self) -> Vec3 {
        let n = self.points.len();
        if n < 2 {
            return Vec3::new(0.0, 1.0, 0.0);
        }
        let d: Vec3 = self.points[n - 1] - self.points[0];
        let len: f32 = d.length();
        if !(len.is_finite() && len > 1e-9) {
            return Vec3::new(0.0, 1.0, 0.0);
        }
        d * (1.0 / len)
    }

    /// 是否可翻译（≥ 2 个有限点，半径为正有限或为空）。
    pub fn is_translatable(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }
        if !self
            .points
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            return false;
        }
        self.radii.iter().all(|r| r.is_finite() && *r > 0.0)
    }
}

/// 场景附加能力集合（[`crate::Scene::attach`] 的类型）。
///
/// **`Default` 全空 ⇒ 与加字段前的 `Scene` 逐字节同帧**（判据 P1）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneAttach {
    /// 介质区域列表（A6；**互斥**，见 [`MediumBox`]）。
    pub media: Vec<MediumBox>,
    /// 三角网格列表（G-2 的输入；每项一条 `Scene::meshes`）。
    pub meshes: Vec<MeshDesc>,
    /// 共享几何原型（实例的 BLAS 源；**不**直接出现在场景里）。
    pub instance_geoms: Vec<MeshDesc>,
    /// 实例列表（引用 [`SceneAttach::instance_geoms`]）。
    pub instances: Vec<InstanceDesc>,
    /// 位图纹理表（句柄 1 起，按本列表顺序编号）。
    pub textures: Vec<TextureDesc>,
    /// 毛发股（每股一个 Mesh + 该股自己的纤维标架）。
    pub strands: Vec<StrandDesc>,
    /// V2.5-B1：`config.ocio` 路径（色彩管理，非 rt 几何）。
    ///
    /// `None` ⇒ 沿用既有 ACES 路径，**逐字节不变**。该字段由 `prism-app`
    /// 的渲染出口消费（`cmd_render --ocio` / v1 CLI），不参与
    /// [`Self::is_empty`]（那是"有无 rt 侧附加几何"的判据）——刻意不混，
    /// 否则一个纯色彩配置会让翻译层误以为场景带了几何附加能力。
    /// 静态门禁 `v25_b1_ocio::v6_static_gate_*` 防止"字段加了但没接"。
    pub ocio_config: Option<String>,
}

impl SceneAttach {
    /// 是否全空（翻译层据此走"与加字段前相同"的快路径）。
    pub fn is_empty(&self) -> bool {
        self.media.is_empty()
            && self.meshes.is_empty()
            && self.instance_geoms.is_empty()
            && self.instances.is_empty()
            && self.textures.is_empty()
            && self.strands.is_empty()
    }

    /// 附加能力条数合计（诊断 / 日志用）。
    pub fn entry_count(&self) -> usize {
        self.media.len()
            + self.meshes.len()
            + self.instance_geoms.len()
            + self.instances.len()
            + self.textures.len()
            + self.strands.len()
    }
}

// ---------------------------------------------------------------------------
// 测试：类型自身的形状契约（跨 crate 语义在 prism-render 的 P1/P2/P3）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `SceneAttach::default()` 必须全空 —— 这是 P1（逐字节同帧）的结构性前提。
    #[test]
    fn default_attach_is_empty() {
        let a: SceneAttach = SceneAttach::default();
        assert!(a.is_empty(), "Default 必须全空（否则空场景也会改渲染路径）");
        assert_eq!(a.entry_count(), 0);
    }

    /// 位移槽：幅值 0 / −0 都算恒等（V1 的短路判据）。
    #[test]
    fn displacement_zero_is_identity() {
        assert!(DisplacementSlot::default().is_identity());
        assert!(DisplacementSlot::new(1, 0.0, 1.0, 3).is_identity());
        assert!(DisplacementSlot::new(1, -0.0, 1.0, 3).is_identity());
        assert!(!DisplacementSlot::new(1, 1e-7, 1.0, 3).is_identity());
    }

    /// 介质盒的合法性判据逐条覆盖（翻译层靠它做显式跳过）。
    #[test]
    fn medium_box_translatable_rules() {
        let ok = MediumBox::new(Vec3::splat(-1.0), Vec3::splat(1.0), 0.5, 0.9, 0.2);
        assert!(ok.is_translatable());
        // min > max
        let bad = MediumBox::new(Vec3::splat(1.0), Vec3::splat(-1.0), 0.5, 0.9, 0.2);
        assert!(!bad.is_translatable());
        // σt ≤ 0
        let bad = MediumBox::new(Vec3::splat(-1.0), Vec3::splat(1.0), 0.0, 0.9, 0.2);
        assert!(!bad.is_translatable());
        // albedo 越界
        let bad = MediumBox::new(Vec3::splat(-1.0), Vec3::splat(1.0), 0.5, 1.5, 0.2);
        assert!(!bad.is_translatable());
        // |g| ≥ 1
        let bad = MediumBox::new(Vec3::splat(-1.0), Vec3::splat(1.0), 0.5, 0.9, 1.0);
        assert!(!bad.is_translatable());
        // 非有限
        let bad = MediumBox::new(
            Vec3::new(f32::NAN, -1.0, -1.0),
            Vec3::splat(1.0),
            0.5,
            0.9,
            0.2,
        );
        assert!(!bad.is_translatable());
    }

    /// 纹理尺寸自洽（长度不等于 `w*h*4` ⇒ 不可翻译，不 panic）。
    #[test]
    fn texture_length_must_match() {
        assert!(TextureDesc::new(2, 2, false, vec![0; 16]).is_translatable());
        assert!(!TextureDesc::new(2, 2, false, vec![0; 15]).is_translatable());
        assert!(!TextureDesc::new(0, 2, false, Vec::new()).is_translatable());
    }

    /// 毛股切向：每股**自己的**方向（T3 的标架来源）。
    #[test]
    fn strand_tangent_is_per_strand() {
        let x = StrandDesc::new(vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)], 0.01);
        let y = StrandDesc::new(vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0)], 0.01);
        let z = StrandDesc::new(vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -4.0)], 0.01);
        assert!((x.tangent() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((y.tangent() - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert!((z.tangent() - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
        // 退化（单点）回落 +Y，绝不产 NaN。
        let deg = StrandDesc::new(vec![Vec3::splat(0.0)], 0.01);
        assert_eq!(deg.tangent().y, 1.0);
    }

    /// 实例：恒等 / 平移两种构造的槽位正确、逆由 rt 侧算（core 不给）。
    #[test]
    fn instance_slots() {
        let id = InstanceDesc::identity(3);
        assert_eq!(id.geom_id, 3);
        assert_eq!(id.mat_id, u32::MAX);
        assert_eq!(id.affine[0], 1.0);
        assert_eq!(id.affine[5], 1.0);
        assert_eq!(id.affine[10], 1.0);
        assert_eq!(id.affine[3], 0.0);
        let tr = InstanceDesc::translated(0, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!((tr.affine[3], tr.affine[7], tr.affine[11]), (1.0, 2.0, 3.0));
        assert!(tr.is_translatable());
    }
}
