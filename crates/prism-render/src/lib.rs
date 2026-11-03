//! `prism-render` — Prism 引擎的渲染调度层。
//!
//! 本 crate 不依赖任何第三方库，仅使用 Rust 标准库（`std`），并对以下
//! 兄弟 crate 的稳定契约负责：
//!
//! - `prism_gpu::{GpuDevice, Framebuffer}`：GPU 设备抽象与帧缓冲结构。
//! - `prism_core::{Scene, eval_camera_orbit}`：场景模型与时间模型；本 crate 的
//!   [`build_render_request`] 负责把它们编译为下述 GPU 请求。
//! - `prism_math::Vec3`：数学基元（本层不直接使用，仅保留类型契约）。
//!
//! 渲染调度的中枢是 [`FrameGraph`]，它将"采样参数"（分辨率、帧率、时长）
//! 与"执行动作"（交给 [`GpuDevice`] 渲染）解耦，从而让上层只需描述"要渲
//! 什么、渲多长"，而不必关心逐帧循环的细节。
//!
//! A5 起新增 tile 渲染骨架：[`tile`]（TileGrid + halo 协议 + 最小 halo 测试
//! 效果）与 [`orchestrator`]（tile 化求值编排），存储域为 `prism_io::fbuf`
//! 的 RGBA16F 半精度帧缓冲。

use prism_gpu::{AestheticReq, CameraReq, Framebuffer, GpuDevice, LightReq, SceneRenderRequest, SphereReq};
use prism_core::Scene;

/// 渲染单帧：把场景在时刻 `t` 编译成设备请求，再交给 GPU 设备渲染。
///
/// # 参数
/// - `device`：可变借用的 GPU 设备（trait 对象），用于实际执行渲染。
/// - `scene`：待渲染的场景模型（不可变借用）。
/// - `t`：当前时刻（秒），取值范围应在 `[0.0, duration]`。
/// - `duration`：整段视频时长（秒），用于场景内部的时间归一化。
/// - `w` / `h`：输出帧宽 / 高（像素）。
///
/// # 返回
/// 设备渲染出的 [`Framebuffer`]。
pub fn render_frame(
    device: &mut dyn GpuDevice,
    scene: &Scene,
    t: f32,
    duration: f32,
    w: u32,
    h: u32,
) -> Framebuffer {
    // 将场景在时刻 t 编译为底层渲染请求；字段与 prism_gpu 的 DTO 契约严格一致。
    let req = build_render_request(scene, t, duration, w, h);
    // 交给 GPU 设备执行单帧渲染。
    device.render_scene(&req)
}

/// 把“某时刻的场景”编译为底层 [`prism_gpu::SceneRenderRequest`]。
///
/// # 迁移说明（依赖解耦，本次变更）
///
/// 本函数原位于 `prism-core`。为解除 `prism-core → prism-gpu` 的硬依赖
/// （使 GPU 侧编辑不再打断 `prism-physics` / `prism-rt` 等 core 下游编译），
/// 将其整体下沉至此——`prism-render` 本就是本函数的**唯一外部消费者**，且
/// 同时依赖 `prism-core` 与 `prism-gpu`，故移动零新增依赖、零新增抽象层，
/// 满足 D-16 Mini Code 的最小改动原则。
///
/// 这是“场景图 → 渲染请求”的核心映射，供本 crate 与 GPU 设备使用：
/// - 相机经 [`prism_core::eval_camera_orbit`] 放到环绕轨道上，注视
///   `orbit_center`，纵横比由 `width / height` 推导；
/// - 每盏 [`prism_core::Light`] 一对一映射为 [`LightReq`]；
/// - 在 `orbit_center` 处发射一个球体，使用场景的 `radius` 与 `material`；
/// - [`prism_core::Aesthetic`] 原样转发为 [`AestheticReq`]。
pub fn build_render_request(
    scene: &Scene,
    t: f32,
    duration: f32,
    width: u32,
    height: u32,
) -> SceneRenderRequest {
    let eye: prism_math::Vec3 = prism_core::eval_camera_orbit(scene, t, duration);
    let aspect: f32 = if height == 0 { 1.0_f32 } else { width as f32 / height as f32 };

    let camera: CameraReq = CameraReq {
        eye,
        target: scene.orbit_center,
        fov: scene.camera.fov,
        aspect,
    };

    let lights: Vec<LightReq> = scene
        .lights
        .iter()
        .map(|light| LightReq {
            direction: light.direction,
            color: light.color,
            intensity: light.intensity,
        })
        .collect();

    let spheres: Vec<SphereReq> = vec![SphereReq {
        center: scene.orbit_center,
        radius: scene.radius,
        albedo: scene.material.albedo,
        roughness: scene.material.roughness,
        metalness: scene.material.metalness,
        maps: prism_gpu::NO_MAPS,
    }];

    let aesthetic: AestheticReq = AestheticReq {
        teal_orange: scene.aesthetic.teal_orange,
        contrast: scene.aesthetic.contrast,
        filmic: scene.aesthetic.filmic,
    };

    SceneRenderRequest {
        width,
        height,
        camera,
        lights,
        spheres,
        aesthetic,
    curves: Vec::new(),
    }
}

/// 渲染整段视频：按 `fps` 在 `[0, duration]` 区间均匀采样若干帧。
///
/// 采样帧数取 `ceil(fps * duration)`，保证末帧时刻不超出 `duration`。
/// 每一帧都委托给 [`render_frame`] 完成，结果按时间先后顺序收集返回。
///
/// # 参数
/// - `device`：可变借用的 GPU 设备。
/// - `scene`：待渲染的场景模型。
/// - `w` / `h`：输出帧宽 / 高。
/// - `fps`：采样帧率（帧/秒）。
/// - `duration`：视频时长（秒）。
///
/// # 返回
/// 按时间顺序排列的 [`Framebuffer`] 序列。
pub fn render_video(
    device: &mut dyn GpuDevice,
    scene: &Scene,
    w: u32,
    h: u32,
    fps: u32,
    duration: f32,
) -> Vec<Framebuffer> {
    let frame_count: usize = (fps as f32 * duration).ceil() as usize;
    (0..frame_count)
        .map(|i| {
            // 第 i 帧对应的时间：等间隔采样，i/fps 落在 [0, duration] 之内。
            let t: f32 = i as f32 / fps as f32;
            render_frame(device, scene, t, duration, w, h)
        })
        .collect()
}

/// 帧图（Frame Graph）：渲染调度的中枢。
///
/// `FrameGraph` 只描述"要渲染成什么规格"，不持有任何 GPU 资源；真正的
/// 执行发生在 [`FrameGraph::execute`]。这种"描述与执行分离"的结构使得：
///
/// - 上层（编辑器 / 时间线）只需构造一个 `FrameGraph` 即可表达一段渲染任务；
/// - 下层（本 crate 或未来的真实 GPU 后端）可以在 `execute` 内部替换调度策略，
///   而不影响调用方。
///
/// 在真实 GPU 后端中，`execute` 内部会承担：
/// 1. **自动资源屏障（Resource Barrier）**：推断 Pass 间的读写依赖，插入同步点；
/// 2. **Pass 排序（Pass Sorting）**：按依赖关系对渲染 Pass 做拓扑排序与合并；
/// 3. **异步队列（Async Queue）**：将独立 Pass 分发到不同硬件队列并行执行。
///
/// 当前为 PoC（概念验证）实现，采用**单 Pass 直渲**：`execute` 直接调用
/// [`render_video`] 逐帧提交，不做任何多 Pass / 屏障逻辑（无 GPU 平台下完整可运行）。
/// 但代码结构已预留出"帧图即渲染调度中枢"的清晰边界，便于后续替换为完整后端。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameGraph {
    /// 输出帧宽度（像素）。
    pub width: u32,
    /// 输出帧高度（像素）。
    pub height: u32,
    /// 采样帧率（帧/秒）。
    pub fps: u32,
    /// 视频时长（秒）。
    pub duration: f32,
}

impl FrameGraph {
    /// 构造一个帧图。
    ///
    /// # 参数
    /// - `width` / `height`：输出分辨率。
    /// - `fps`：采样帧率。
    /// - `duration`：视频时长（秒）。
    pub fn new(width: u32, height: u32, fps: u32, duration: f32) -> Self {
        FrameGraph {
            width,
            height,
            fps,
            duration,
        }
    }

    /// 执行帧图：将内部描述的渲染任务交付给 GPU 设备，得到完整帧序列。
    ///
    /// 当前实现为单 Pass 直渲，直接转发给 [`render_video`]；
    /// 未来的真实后端可在此处插入资源屏障、Pass 排序与异步队列调度。
    ///
    /// # 参数
    /// - `device`：可变借用的 GPU 设备。
    /// - `scene`：待渲染的场景模型。
    ///
    /// # 返回
    /// 按时间顺序排列的渲染帧序列。
    pub fn execute(&self, device: &mut dyn GpuDevice, scene: &Scene) -> Vec<Framebuffer> {
        render_video(
            device,
            scene,
            self.width,
            self.height,
            self.fps,
            self.duration,
        )
    }
}

// ---------------------------------------------------------------------------
// D-19 双轨：prism-rt 路径追踪的 GpuDevice 适配器（W0-B 产品接线）
//
// 默认档（本后端）走完整路径追踪；`--fast` 仍走 `prism_gpu::CpuRefBackend`。
// 适配器落在本 crate 而非 `prism-rt`，理由见 `rt_backend` 模块文档。
// ---------------------------------------------------------------------------

pub mod rt_backend;
/// 单帧 tile 化并行求值（W0-C L3）：固定 32×32 tile + 固定序合并，
/// 线程数不进结果（D-20）。
pub mod rt_tiles;

pub use rt_backend::{RtBackend, RT_DEFAULT_BOUNCES, RT_DEFAULT_SPP};
pub use rt_tiles::{auto_threads, RT_TILE_H, RT_TILE_W};

// ---------------------------------------------------------------------------
// tile 渲染骨架——TileGrid / halo 协议 / 编排器
// ---------------------------------------------------------------------------

pub mod tile;
pub mod orchestrator;

pub use tile::{BoxBlur3, Tile, TileEffect, TileGrid};
pub use orchestrator::{PixelSource, TileOrchestrator};

// ---------------------------------------------------------------------------
// DSL v2 成片直达——CompileV2Output → 帧序列 Deliverable
// ---------------------------------------------------------------------------

pub mod movie;

pub use movie::{render_movie, Deliverable, MovieResult, RenderError};

// ---------------------------------------------------------------------------
// C3：增量渲染——脏区间检测与复用（C3 行）
// ---------------------------------------------------------------------------

pub mod incremental;

pub use incremental::{
    bytes_to_fbuf, decide, decide_with_mode, fbuf_to_bytes, reuse_stats_from_dirty,
    ChannelIntervals, ChannelKind, DirtyDetector, DirtySet, FrameRange, IncrementalError,
    IncrementalSession, ParamChange, ReuseCache, ReuseDecision, ReuseStats, ReuseVerdict,
    TileArtifact,
};

// ---------------------------------------------------------------------------
// C3：节点注册（D-27 静态汇入制）——脏区间闭包特解
// ---------------------------------------------------------------------------

pub mod nodes;

pub use nodes::DirtyRange;

// ---------------------------------------------------------------------------
// 分布式渲染农场调度器
//  （固定形状归约代数 + 帧级并行 + 确定性切分）
// ---------------------------------------------------------------------------

pub mod farm;

pub use farm::{
    frame_count, frame_payload, manifest_digest, merge_shards, merge_shards_full,
    parallel_manifest_digest, plan_dispatch, reduce_layered_parallel, serial_manifest_digest,
    slot_counts, slot_ranges, split_frames, split_frames_shape_aligned, ConfigHash, DispatchPlan,
    FarmError, FarmNode, FarmQueue, FrameShard, FrameTask, FrameValue, LocalScheduler, MergeAudit,
    QueueState, ResourceNeed, RetryPolicy, ShardOutcome, SubmitOutcome, TaskId, TaskOrderKey,
    ThroughputLedger, md5_hex,
};

// ---------------------------------------------------------------------------
// EXR 序列交付路径（逐帧 `.exr`；复用 prism-io 的 exr_write，不重写编解码）
//
// 交付矩阵（`docs/交付矩阵.md`）的 `EXR-seq` 一列由本模块承载：
// 无损、逐帧独立、浮点通道，面向合成/调色/后期。
// ---------------------------------------------------------------------------

pub mod exr_seq;

pub use exr_seq::{
    frame_file_name, frame_to_planes, plane_value_to_u8, read_exr_frame, write_exr_sequence,
    ExrSeqError, ExrSeqOptions,
};

// ---------------------------------------------------------------------------
// D-2：性能剖析——分配/拷贝计数、帧池、零拷贝身份判据（T09 / V2.2-D-2）
//   设计 §6.2 / Line 516：#[global_allocator] 只在专用测试二进制安装，生产零改动。
// ---------------------------------------------------------------------------

pub mod measure;
pub mod frame_pool;
pub mod zero_copy;

pub use frame_pool::{render_frame_pooled, FramePool, PoolStats, pool_stats, reset_pool_stats};
pub use measure::{
    copy_bytes, peak_rss_kib, rss_report, reset_counters, snapshot, Counters, CountingAllocator,
};
pub use zero_copy::{buffer_reborrow, is_zero_copy_path, verify_zero_copy, ZERO_COPY_WHITELIST};

#[cfg(test)]
mod tests {
    use super::*;

    // 仅做结构层面的断言，确保公共 API 签名与契约一致；
    // 不在此处构造真实 GpuDevice（无 GPU 后端）。
    #[test]
    fn frame_graph_carries_parameters() {
        let fg = FrameGraph::new(1920, 1080, 30, 2.0);
        assert_eq!(fg.width, 1920);
        assert_eq!(fg.height, 1080);
        assert_eq!(fg.fps, 30);
        assert_eq!(fg.duration, 2.0);
    }

    // -----------------------------------------------------------------------
    // build_render_request 测试（自 prism-core 随函数迁移而来）
    // -----------------------------------------------------------------------

    /// 构造一个用于验证 `build_render_request` 的样例场景：轨道半径 4。
    fn sample_scene() -> Scene {
        use prism_core::{Aesthetic, Camera, Light, Material};
        use prism_math::Vec3;
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
            camera: Camera::new(
                Vec3::new(0.0, 0.0, -4.0),
                Vec3::new(0.0, 0.0, 0.0),
                60.0_f32.to_radians(),
            ),
            aesthetic: Aesthetic::new(true, 1.2, true),
        }
    }

    #[test]
    fn render_request_uses_orbit_eye_and_maps_fields() {
        use prism_math::Vec3;

        let scene: Scene = sample_scene();
        let duration: f32 = 10.0;
        let req: SceneRenderRequest = build_render_request(&scene, 0.0, duration, 800, 600);

        // 相机 eye 应与时间模型一致。
        let expected_eye: Vec3 = prism_core::eval_camera_orbit(&scene, 0.0, duration);
        assert!((req.camera.eye - expected_eye).length() < 1e-5);

        // 注视点为轨道中心，纵横比取自分辨率。
        assert!((req.camera.target - scene.orbit_center).length() < 1e-5);
        assert!((req.camera.aspect - 800.0 / 600.0).abs() < 1e-5);

        // 一盏灯 → 一个光源，材质映射到一个球体。
        assert_eq!(req.lights.len(), 1);
        assert_eq!(req.spheres.len(), 1);
        assert_eq!(req.spheres[0].radius, scene.radius);
        assert!((req.spheres[0].albedo - scene.material.albedo).length() < 1e-5);

        // 风格原样转发。
        assert_eq!(req.aesthetic.teal_orange, true);
        assert!((req.aesthetic.contrast - 1.2).abs() < 1e-5);
        assert_eq!(req.aesthetic.filmic, true);
    }

    #[test]
    fn build_render_request_maps_multiple_lights_and_material() {
        use prism_core::Light;
        use prism_math::Vec3;

        let mut scene: Scene = sample_scene();
        // 两盏不同的灯，验证一对一映射与字段透传。
        scene.lights = vec![
            Light::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0),
            Light::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.5, 0.6, 0.7), 2.5),
        ];
        // fov 以弧度透传（运行时契约，与 prism-app/main.rs 的 to_radians 用法一致）；
        // 透传断言同时验证本函数不对 fov 做任何改写。
        let fov_rad: f32 = 55.0_f32.to_radians();
        scene.camera.fov = fov_rad;

        let req: SceneRenderRequest = build_render_request(&scene, 1.0, 10.0, 640, 360);

        // 分辨率与纵横比。
        assert_eq!(req.width, 640);
        assert_eq!(req.height, 360);
        assert!((req.camera.aspect - 640.0 / 360.0).abs() < 1e-5, "aspect 应等于 width/height");
        assert!((req.camera.fov - fov_rad).abs() < 1e-6, "fov 应原样透传（弧度）");

        // 灯光一对一映射，方向/颜色/强度逐字段透传。
        assert_eq!(req.lights.len(), 2);
        for (want, got) in scene.lights.iter().zip(req.lights.iter()) {
            assert!((want.direction - got.direction).length() < 1e-6);
            assert!((want.color - got.color).length() < 1e-6);
            assert!((want.intensity - got.intensity).abs() < 1e-6);
        }

        // 球体：半径 + 材质（albedo/roughness/metalness）完整透传。
        assert_eq!(req.spheres.len(), 1);
        let s = &req.spheres[0];
        assert!((s.center - scene.orbit_center).length() < 1e-6);
        assert!((s.radius - scene.radius).abs() < 1e-6, "球体半径应透传");
        assert!((s.albedo - scene.material.albedo).length() < 1e-6);
        assert!((s.roughness - scene.material.roughness).abs() < 1e-6, "roughness 应透传");
        assert!((s.metalness - scene.material.metalness).abs() < 1e-6, "metalness 应透传");
    }

    #[test]
    fn build_render_request_aspect_guards_zero_height() {
        let scene: Scene = sample_scene();
        // height == 0 时 aspect 应回退为 1.0，避免除零。
        let req: SceneRenderRequest = build_render_request(&scene, 0.0, 10.0, 800, 0);
        assert!(
            (req.camera.aspect - 1.0).abs() < 1e-6,
            "height=0 时 aspect 应回退为 1.0，实际 {}",
            req.camera.aspect
        );
    }

    /// 构造一个环绕红球的测试场景：固定顶光，相机绕轨道中心旋转，
    /// 关闭全部后期特效以保证输出确定、帧间差异可归因于相机运动。
    fn orbit_scene() -> Scene {
        use prism_core::{Aesthetic, Camera, Light, Material};
        use prism_math::Vec3;
        Scene {
            radius: 1.0,
            orbit_center: Vec3::new(0.0, 0.0, 0.0),
            orbit_radius: 3.0,
            material: Material::new(Vec3::new(0.8, 0.2, 0.2), 0.4, 0.1),
            lights: vec![Light::new(
                // 斜向主光。注意：不能取纯竖直 (0,-1,0)——那会与相机轨道的
                // 旋转轴（Y 轴）共线，使“球+光+相机”整体绕 Y 轴旋转对称，
                // 相机环绕时画面逐像素不变，帧差异断言将失去意义。
                Vec3::new(0.3, -1.0, 0.6),
                Vec3::new(1.0, 1.0, 1.0),
                1.0,
            )],
            // 运行时契约：fov 以弧度计（60° → π/3），与 prism-app/main.rs 一致。
            camera: Camera::new(Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 0.0), 60.0_f32.to_radians()),
            aesthetic: Aesthetic::new(false, 1.0, false),
        }
    }

    #[test]
    fn render_video_returns_expected_frame_count_and_motion() {
        use prism_gpu::CpuRefBackend;

        let scene = orbit_scene();
        let mut device = CpuRefBackend;
        let frames = render_video(&mut device, &scene, 32, 18, 4, 2.0);

        // fps=4、duration=2.0 → ceil(4 * 2.0) = 8 帧。
        assert_eq!(frames.len(), 8, "fps=4、duration=2.0 应渲染 8 帧");
        for (i, fb) in frames.iter().enumerate() {
            assert_eq!(
                fb.pixels.len(),
                32 * 18 * 4,
                "第 {} 帧缓冲长度应为 32*18*4",
                i
            );
        }

        // 相机环绕：首帧（t=0）与末帧（t=1.75s）画面必须不同（有运动）。
        assert_ne!(
            frames[0].pixels,
            frames[frames.len() - 1].pixels,
            "首帧与末帧像素完全一致说明相机没有产生运动"
        );
    }

    #[test]
    fn render_frame_produces_requested_dimensions() {
        use prism_gpu::CpuRefBackend;

        let scene = orbit_scene();
        let mut device = CpuRefBackend;
        let fb = render_frame(&mut device, &scene, 0.5, 2.0, 16, 9);
        assert_eq!(fb.width, 16);
        assert_eq!(fb.height, 9);
        assert_eq!(fb.pixels.len(), 16 * 9 * 4);
    }
}
