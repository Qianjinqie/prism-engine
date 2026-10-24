//! Prism 引擎核心：高层场景描述 + 时间模型。
//!
//! 本 crate 定义用户/DSL 代码构造的、以时间 `t` 为参数的高层 [`Scene`]，
//! 并提供把时间映射到相机机位的时间模型。真正面向 GPU 的“场景 → 底层渲染
//! 请求”编译（原 `build_render_request`）已下沉到 `prism-render`，见下方“依赖
//! 解耦”说明。
//!
//! # 依赖解耦（本次变更）
//!
//! 历史上本 crate 直接依赖 `prism-gpu`，仅为构造其 5 个纯数据 DTO
//! （`CameraReq` / `LightReq` / `SphereReq` / `AestheticReq` / `SceneRenderRequest`）。
//! 这使 `prism-core` 硬依赖 `prism-gpu`，导致**任何 GPU 侧编辑都会打断
//! `prism-physics` / `prism-rt` 等下游 crate 的编译**（它们依赖 core 而不依赖
//! gpu，却被牵连重编）。
//!
//! 解耦方案（最小改动路径，D-16 Mini Code）：把唯一真正“碰 GPU 类型”的函数
//! `build_render_request` 及其 5 个测试整体下沉到它的**唯一外部消费者**
//! `prism-render`（本就同时依赖 `prism-core` 与 `prism-gpu`），本 crate 删除
//! `prism-gpu` 依赖。未引入任何新 trait / 转发层 / 新 crate / feature 门控。
//!
//! # 约束
//! - 零第三方依赖，仅使用 `std`（本文件未直接使用）与 workspace 内的
//!   `prism_math`、`prism_anim` 两个 crate。
//!
//! # 错误码段位表（workspace 文档单点）
//!
//! 错误码首段数字 = 段位，段位 × crate 从属如下（本表为 docs 口径单点，
//! 不含运行时逻辑）：
//! - **P1xxx**：DSL / physics 各自语境。注意 P1001-P1004 在 prism-dsl 与
//!   prism-physics 中语义不同，按 crate 命名空间隔离解析；
//! - **P2xxx**：DSL 表达式（表达式语法 / 求值校验）；
//! - **P3xxx**：core 图 / 签名 / 注册表（本 crate：graph / signature / eval / registry）；
//! - **P4xxx**：packs（pack 清单与兼容性校验）；
//! - **PF1xxx**：prism-postfx 效果注册与 LUT 解析（PF1001–PF1007，2026-10-02 B4 登记新增段——独立前缀避免与 P4xxx packs 视觉混淆）。
//!
//! 码段 × crate 从属声明：**跨 crate 复用编号属允许**——错误均在各自
//! crate 命名空间内解析，同号跨 crate 不构成语义冲突。

// ---------------------------------------------------------------------------
// v2 节点模型：节点三元组 + 特解注册表（第一批）与图 + 求值调度（第二批）
// ---------------------------------------------------------------------------

pub mod node;
pub mod registry;
pub mod graph;
pub mod eval;
pub mod property;
pub mod signature;
pub mod depth;
pub mod compose;
pub mod nodes;
// v1.5-C（ADR-18）：转场类型下沉 + 场景序列（u8 整数混合 / 凸权重半群 /
// Morph 几何核心）。零浮点、零第三方、同输入两跑 bitwise 相同。
pub mod sceneseq;
// W3：环推进（不动点迭代）——推进-停机四元组的环实例（M3 定数迭代 + 残差
// 记录 + 发散检测 + 帧间 seek 驱动）。语义权威见 `docs/设计/研究/环推进-不动点实例.md`。
pub mod fixpoint;

pub use node::{EvalCtx, Node, NodeId, Params, Payload, SampleIndex, StructureId};
pub use registry::{NodeRegistry, RegistryError};
pub use depth::{DepthComposite, DepthLayer, DepthMap};
pub use compose::{apply_fog, blend_pixel, ComposeNode, LayerBuf};
// v1.5-C：转场 / 场景序列 / Morph 几何（ADR-18 下沉件）。
pub use sceneseq::{
    dissolve_hash, fold_closed, mix_u8, push_c, wipe_edge_q8, wipe_weight, FrameBuf, MorphSpec,
    PixelCtx, PushDir, SceneSeq, Transition, TransitionChain, TransitionKind, WIPE_BAND, W_ONE,
};
pub use nodes::{collect as collect_c1_nodes, default_registry as c1_node_registry};
pub use graph::{
    ceil_log2, error_bound_coeff, reduce, reduce_sorted_by_key, reduce_unordered, reduction_shape,
    Graph, GraphError, ReductionShape,
};
pub use eval::{
    eval, payload_from_bytes, payload_to_bytes, snapshot_read, snapshot_write, EvalError,
    EvalOutput, SamplePlane,
};
pub use property::{sample_binding, PropertyChannel};
pub use fixpoint::{
    loop_frame_signature, run_fixed_point, solve_mode_def, DivergenceDetector, FixpointReport,
    LoopCodec, LoopDriver, SolveMode, SolveModeMeta, DIVERGENCE_WINDOW_DEFAULT,
    RESIDUAL_EPS_DEFAULT, SOLVE_MODE_CHANNEL_TAG,
};
pub use signature::{
    cache_key, channel_definition_hash, graph_canonical_bytes, graph_signature, graph_signatures,
    node_canonical_bytes, node_signature, snapshot_signature, CacheDims, ChannelDef, ChannelDefs,
    SignatureError, SIGNATURE_SCHEMA_VERSION,
};

use prism_math::Vec3;

/// 一个完整圆周对应的弧度（2π），用于把归一化时间比例映射到环绕角度。
const TAU: f32 = std::f32::consts::PI * 2.0;

// ---------------------------------------------------------------------------
// 高层场景描述（用户 / DSL 构造）
// ---------------------------------------------------------------------------

/// 环绕球体的表面材质。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub albedo: Vec3,
    pub roughness: f32,
    pub metalness: f32,
}

impl Material {
    /// 以给定反照率、粗糙度、金属度构造材质。
    pub fn new(albedo: Vec3, roughness: f32, metalness: f32) -> Self {
        Self { albedo, roughness, metalness }
    }
}

/// 方向光光源。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Light {
    /// 以给定方向、颜色、强度构造光源。
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self { direction, color, intensity }
    }
}

/// 环绕相机描述。`eye` 由时间推导（见 [`eval_camera_orbit`]）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov: f32,
}

impl Camera {
    /// 以给定初始机位、注视点、视场角（**弧度**，调用方负责 `to_radians()`，参见 `prism-app/main.rs` 用法）构造相机。
    pub fn new(eye: Vec3, target: Vec3, fov: f32) -> Self {
        Self { eye, target, fov }
    }
}

/// 后期 / 调色风格（look-development）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aesthetic {
    pub teal_orange: bool,
    pub contrast: f32,
    pub filmic: bool,
}

impl Aesthetic {
    /// 以给定青橙映射、对比度、胶片曲线开关构造风格。
    pub fn new(teal_orange: bool, contrast: f32, filmic: bool) -> Self {
        Self { teal_orange, contrast, filmic }
    }
}

/// 一个完整的、以时间参数化的 Prism 场景。
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    pub radius: f32,
    pub orbit_center: Vec3,
    pub orbit_radius: f32,
    pub material: Material,
    pub lights: Vec<Light>,
    pub camera: Camera,
    pub aesthetic: Aesthetic,
}

// ---------------------------------------------------------------------------
// 时间模型
// ---------------------------------------------------------------------------

/// 计算给定时刻 `t`（秒）相机环绕后的 `eye` 位置。
///
/// 相机在半径为 `orbit_radius` 的水平圆上绕 `orbit_center` 环绕。`t / duration`
/// 这一归一化进度被线性映射到整圆（`2π`）。`t` 可略大于 `duration`，此时相机
/// 继续旋转（允许多圈）。
///
/// - `t == 0` 时，相机位于中心 `-Z` 侧（offset = (0, 0, -orbit_radius)）；
/// - `t == duration / 2` 时，相机已绕到中心 `+Z` 侧。
pub fn eval_camera_orbit(scene: &Scene, t: f32, duration: f32) -> Vec3 {
    let angle: f32 = (t / duration) * TAU;
    let offset: Vec3 = Vec3::new(
        angle.sin() * scene.orbit_radius,
        0.0_f32,
        -angle.cos() * scene.orbit_radius,
    );
    scene.orbit_center + offset
}


// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
//
// 说明：原 `build_render_request` 相关的 5 个测试已随该函数一起下沉到
// `prism-render`（见本文件头部“依赖解耦”说明）。本 crate 只保留不涉 GPU
// 的时间模型（[`eval_camera_orbit`]）测试。

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    /// 构造一个用于验证的样例场景：轨道半径 4，持续 10 秒。
    fn sample_scene() -> Scene {
        Scene {
            radius: 1.0,
            orbit_center: Vec3::new(0.0, 0.0, 0.0),
            orbit_radius: 4.0,
            material: Material::new(Vec3::new(0.8, 0.2, 0.2), 0.4, 0.1),
            lights: vec![Light::new(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                1.0,
            )],
            camera: Camera::new(Vec3::new(0.0, 0.0, -4.0), Vec3::new(0.0, 0.0, 0.0), 60.0_f32.to_radians()),
            aesthetic: Aesthetic::new(true, 1.2, true),
        }
    }

    #[test]
    fn orbit_eye_differs_between_start_and_half() {
        let scene: Scene = sample_scene();
        let duration: f32 = 10.0;
        let eye_t0: Vec3 = eval_camera_orbit(&scene, 0.0, duration);
        let eye_thalf: Vec3 = eval_camera_orbit(&scene, duration / 2.0, duration);
        // 用向量减法 + 长度判断，避免依赖 Vec3 的 PartialEq。
        let delta: f32 = (eye_t0 - eye_thalf).length();
        assert!(delta > 1e-5, "t=0 与 t=duration/2 的相机位置应不同");
    }

    #[test]
    fn orbit_eye_positions_symmetric_about_center() {
        let scene: Scene = sample_scene();
        let duration: f32 = 10.0;
        let e0: Vec3 = eval_camera_orbit(&scene, 0.0, duration);
        let eh: Vec3 = eval_camera_orbit(&scene, duration / 2.0, duration);

        // 对称性：e0 + eh == 2 * orbit_center（关于中心点对称）。
        let center2: Vec3 = scene.orbit_center * 2.0;
        assert!(
            ((e0 + eh) - center2).length() < 1e-4,
            "t=0 与 t=duration/2 的 eye 应关于 orbit_center 对称"
        );

        // 两个位置到中心的距离都等于轨道半径。
        assert!(
            ((e0 - scene.orbit_center).length() - scene.orbit_radius).abs() < 1e-4,
            "t=0 的 eye 到中心距离应等于 orbit_radius"
        );
        assert!(
            ((eh - scene.orbit_center).length() - scene.orbit_radius).abs() < 1e-4,
            "t=duration/2 的 eye 到中心距离应等于 orbit_radius"
        );
    }
}
