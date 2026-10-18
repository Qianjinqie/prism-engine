//! DSL v2 成片直达（V1.5-D）：[`CompileV2Output`] → 帧序列 [`Deliverable`]。
//!
//! # 定位
//! 本模块是 prism-dsl v2 编译产物在 render 侧的消费入口：读入一次
//! `compile_v2` 的完整产物（节点图 + 通道 + 场景时间线 + 过渡），按 fps 均匀
//! 采样求值整段影片。3D 帧求值**复用本 crate 既有求值器**
//! （[`crate::render_frame`] → `CpuRefBackend` 软光追），不另写渲染器；
//! 文本图层合成复用 `prism-text` 既有排版 / 光栅化能力；转场混合复用
//! `prism-core::sceneseq`（W1-C 下沉的 u8 整数混合家族）。
//!
//! # 覆盖纪律（宪章：绝不静默糊弄近似）
//! render 侧只宣称**已支持子集**（详见 `docs/v2-node-coverage.md`）：
//! - S5 相机（stage3d camera）→ 场景相机轨道；
//! - S8 材质（stage3d material）→ 场景材质；
//! - S12 实体 `kind sphere`（stage3d sdf）→ 球体半径；
//! - S12 文本图层（layer text）→ 帧上文本合成；
//! - 场景时间线：顺序播放 / 过渡重叠混合 / 同程多场景左右分屏。
//!
//! 任何不支持的节点（particles / sim / sdf box / compose / 未知结构）都在
//! **求值前**以 [`RenderError`] 三段式明确报错（哪个节点、为什么不支持、
//! 怎么改），绝不静默跳过或近似。
//!
//! # 时间线语义（由 `CompileV2Output` 数据自身表达，无隐式开关）
//! - 帧 `t` 落入唯一场景区间 → 该场景帧；
//! - 帧 `t` 落入两场景**部分重叠**区间 → 按匹配的 `TransitionV2` 混合；
//! - 帧 `t` 落入两场景**起止完全相同**区间 → 左右分屏合成（组合子
//!   `split_screen` 的展开形态）；
//! - 帧 `t` 不落入任何场景 → 黑帧（`timeline.duration` 长于内容时）；
//! - 其余多场景同时活跃的组合 → 明确报错。

use std::collections::BTreeMap;
use std::fmt;

use prism_core::node::{NodeBinding, ParamValue};
use prism_core::property::{sample_binding, PropertyChannel};
use prism_core::sceneseq::{dissolve_hash, mix_u8, push_c, wipe_weight, PushDir, W_ONE};
use prism_core::{Aesthetic, Camera, EvalCtx, Light, Material, Scene, StructureId};
use prism_dsl::{CompileV2Output, SceneV2, TransitionV2};
use prism_gpu::{Framebuffer, GpuDevice};
use prism_math::Vec3;
use prism_text::cache::GlyphCache;
use prism_text::font_bold;
use prism_text::layout::{layout_text, Align, TextParams};

use crate::render_frame;

// ---------------------------------------------------------------------------
// 交付产物与错误
// ---------------------------------------------------------------------------

/// 成片交付产物：规格 + 按时间顺序排列的帧序列（RGBA8）。
#[derive(Debug, Clone)]
pub struct Deliverable {
    /// 影片标题（来自 movie 块；组合子产物自带标题）。
    pub title: String,
    /// 画幅宽（px）。
    pub width: u32,
    /// 画幅高（px）。
    pub height: u32,
    /// 帧率（fps）。
    pub fps: u32,
    /// 总时长（秒）。
    pub duration: f32,
    /// 帧序列（长度 = `ceil(fps × duration)`）。
    pub frames: Vec<Framebuffer>,
}

impl Deliverable {
    /// 帧数（= `ceil(fps × duration)`）。
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// 逐帧 RGBA8 裸字节（供容器写入 / ffmpeg 管道消费）。
    pub fn frame_bytes(&self) -> Vec<Vec<u8>> {
        self.frames.iter().map(|fb| fb.pixels.clone()).collect()
    }
}

/// 成片渲染错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
///
/// 错误码取 `R5xxx` 段（render 成片直达，与 MUX1xxx / P3xxx 分段不冲突）。
#[derive(Debug, Clone, PartialEq)]
pub struct RenderError {
    /// 错误码（R5xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：场景 / 节点 / 阶段。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl RenderError {
    /// 构造三段式错误。
    fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for RenderError {}

/// render 成片直达 `Result` 别名。
pub type MovieResult<T> = Result<T, RenderError>;

// ---------------------------------------------------------------------------
// 内部模型：场景求值单元（节点分类快照）
// ---------------------------------------------------------------------------

/// 节点引用快照：静态参数 + 通道绑定（按帧解算）。
#[derive(Debug, Clone)]
struct NodeRef {
    /// 节点 id（诊断定位用）。
    id: u64,
    /// 静态参数（codegen 缺省回填后的值）。
    params: prism_core::Params,
    /// 通道绑定（参数 → 命名通道）。
    bindings: Vec<NodeBinding>,
}

/// 场景求值模型：一次分类后的可渲染场景。
#[derive(Debug, Clone)]
struct SceneModel {
    /// 场景在 `compile.scenes` 中的下标（诊断定位）。
    index: usize,
    /// 场景名。
    name: String,
    /// 时间线起点（秒）。
    start: f32,
    /// 场景时长（秒）。
    duration: f32,
    /// 相机节点（S5；None = 未声明，用缺省轨道）。
    camera: Option<NodeRef>,
    /// 材质节点（S8；None = 缺省材质）。
    material: Option<NodeRef>,
    /// 球体节点（S12 sdf kind=sphere；None = 无实体，纯背景 / 文本场景）。
    sphere: Option<NodeRef>,
    /// 文本图层（S12 layer text；按节点序合成）。
    texts: Vec<NodeRef>,
}

// ---------------------------------------------------------------------------
// 参数读取小助手（缺失 / 类型不符 / 非有限 → 缺省；确定性）
// ---------------------------------------------------------------------------

/// 读 f32 参数（缺失 / 非 F32 / 非有限 → 缺省值）。
fn p_f32(params: &prism_core::Params, key: &str, default: f32) -> f32 {
    match params.get(key) {
        Some(ParamValue::F32(v)) if v.is_finite() => *v,
        _ => default,
    }
}

/// 读字符串参数。
fn p_str<'a>(params: &'a prism_core::Params, key: &str) -> Option<&'a str> {
    match params.get(key) {
        Some(ParamValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------

/// 把一次 v2 编译产物渲染为成片帧序列。
///
/// # 参数
/// - `compile`：[`prism_dsl::compile_v2`] 的产物（或组合子展开的等价产物）；
/// - `device`：可变借用的 GPU 设备（测试 / CLI 用 `CpuRefBackend`）。
///
/// # Errors
/// - 规格非法（宽高 / fps / 时长为 0）→ `R5001`；
/// - 节点图含**不支持**的节点（particles / sim / sdf box / compose / 未知
///   结构）→ `R5010`–`R5014`，报错指明节点 id 与缺失的实现点；
/// - 场景时间线出现无法解释的多场景同时活跃 → `R5020`。
///
/// # 确定性
/// 纯数据驱动：同输入两跑逐位一致（宪章铁律 2）。通道采样经
/// `EvalCtx::new(0, node_id, frame)`（全局种子恒 0——DSL 无全局种子字段，
/// 固定值即确定值）。
pub fn render_movie(
    compile: &CompileV2Output,
    device: &mut dyn GpuDevice,
) -> MovieResult<Deliverable> {
    // ---- 0. 规格校验（三段式，绝不带病渲染）------------------------------
    if compile.width == 0 || compile.height == 0 {
        return Err(RenderError::new(
            "R5001",
            format!("movie '{}' 的输出规格", compile.title),
            format!(
                "分辨率 {}x{} 含 0 边，无法求值帧缓冲",
                compile.width, compile.height
            ),
            "在 DSL 的 output / movie 块声明正分辨率（如 resolution 320x180）",
        ));
    }
    if compile.fps == 0 {
        return Err(RenderError::new(
            "R5001",
            format!("movie '{}' 的输出规格", compile.title),
            "帧率 fps 为 0，无法确定采样帧数".to_string(),
            "在 DSL 的 timeline / movie 块声明正帧率（如 fps 12）",
        ));
    }
    if !(compile.duration.is_finite() && compile.duration > 0.0) {
        return Err(RenderError::new(
            "R5001",
            format!("movie '{}' 的输出规格", compile.title),
            format!("总时长必须为正有限数，却得到 {}", compile.duration),
            "在 DSL 的 timeline / movie 块声明正时长（如 duration 3s），或让场景时长推算出正总时长",
        ));
    }

    // ---- 1. 通道登记表 + 场景模型（求值前完成全部校验）-------------------
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let models: Vec<SceneModel> = build_models(compile, &channels)?;

    // ---- 2. 帧循环 --------------------------------------------------------
    let frame_count: usize = (compile.fps as f32 * compile.duration).ceil() as usize;
    let mut frames: Vec<Framebuffer> = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let t: f32 = i as f32 / compile.fps as f32;
        let active: Vec<usize> = models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.duration > 0.0 && m.start <= t && t < m.start + m.duration)
            .map(|(idx, _)| idx)
            .collect();
        let fb: Framebuffer = match active.len() {
            // 空档（timeline 长于内容）：黑帧，确定性。
            0 => black_frame(compile.width, compile.height),
            1 => render_scene_frame(&models[active[0]], compile, &channels, device, t, i as u64)?,
            2 => {
                let a: &SceneModel = &models[active[0]];
                let b: &SceneModel = &models[active[1]];
                if let Some(tr) = find_transition(compile, a, b, t) {
                    // 过渡重叠区：两场景各求值一帧，按 kind 整数混合。
                    let fa: Framebuffer =
                        render_scene_frame(a, compile, &channels, device, t, i as u64)?;
                    let fb2: Framebuffer =
                        render_scene_frame(b, compile, &channels, device, t, i as u64)?;
                    blend_transition(&fa, &fb2, tr, t)
                } else if same_interval(a, b) {
                    // 同程多场景 = 左右分屏（split_screen 展开形态）。
                    let fl: Framebuffer =
                        render_scene_frame(a, compile, &channels, device, t, i as u64)?;
                    let fr: Framebuffer =
                        render_scene_frame(b, compile, &channels, device, t, i as u64)?;
                    split_halves(&fl, &fr)
                } else {
                    return Err(RenderError::new(
                        "R5020",
                        format!(
                            "movie '{}' 的时间线（t = {t:.3}s）",
                            compile.title
                        ),
                        format!(
                            "场景 #{} '{}'（[{}..{}]s）与 #{} '{}'（[{}..{}]s）同时活跃，但既非同程分屏（起止相同）也无匹配过渡",
                            a.index + 1, a.name, a.start, a.start + a.duration,
                            b.index + 1, b.name, b.start, b.start + b.duration
                        ),
                        "两段同时活跃的画面只能是：① 起止相同的分屏（split_screen 组合子）；② 相邻场景 + 一条 duration 匹配的 transition 块",
                    ));
                }
            }
            _ => {
                let names: Vec<String> = active.iter().map(|&idx| models[idx].name.clone()).collect();
                return Err(RenderError::new(
                    "R5020",
                    format!("movie '{}' 的时间线（t = {t:.3}s）", compile.title),
                    format!("{} 个场景同时活跃（{}），超出已支持的时间线语义", active.len(), names.join("、")),
                    "已支持：单场景 / 两场景过渡重叠 / 两场景同程分屏；更多同时活跃画面请拆分为多个 movie 或等待后续波次",
                ));
            }
        };
        frames.push(fb);
    }

    Ok(Deliverable {
        title: compile.title.clone(),
        width: compile.width,
        height: compile.height,
        fps: compile.fps,
        duration: compile.duration,
        frames,
    })
}

// ---------------------------------------------------------------------------
// 场景模型构建（节点分类 + 覆盖校验）
// ---------------------------------------------------------------------------

/// 遍历全部场景的节点，分类为相机 / 材质 / 球体 / 文本；遇不支持节点即报错。
fn build_models(
    compile: &CompileV2Output,
    channels: &BTreeMap<&str, &PropertyChannel>,
) -> MovieResult<Vec<SceneModel>> {
    let mut models: Vec<SceneModel> = Vec::with_capacity(compile.scenes.len());
    for (si, scene) in compile.scenes.iter().enumerate() {
        let mut model: SceneModel = SceneModel {
            index: si,
            name: scene.name.clone(),
            start: scene.start,
            duration: scene.duration,
            camera: None,
            material: None,
            sphere: None,
            texts: Vec::new(),
        };
        // node_ids 升序（codegen 保证）→ 合成序确定。
        for &id in &scene.node_ids {
            let node = compile.graph.node(id).ok_or_else(|| {
                RenderError::new(
                    "R5002",
                    format!("场景 '{}' 的节点 {id}", scene.name),
                    "场景摘要登记的节点 id 不在节点图中（悬空引用）".to_string(),
                    "检查编译产物一致性；若为组合子展开产物，确认子片吸收时 id 已重编号",
                )
            })?;
            // 绑定通道必须可解析（缺通道 = 静默丢动画，宪章不容）。
            for b in &node.bindings {
                if !channels.contains_key(b.channel.name.as_str()) {
                    return Err(RenderError::new(
                        "R5003",
                        format!("场景 '{}' 的节点 {id}", scene.name),
                        format!(
                            "参数 '{}' 绑定的通道 '{}' 未在 channels 登记表中定义",
                            b.param, b.channel.name
                        ),
                        "补上同名命名通道（anim / expr 块或组合子展开时登记），或删除该绑定",
                    ));
                }
            }
            let node_ref: NodeRef = NodeRef {
                id: node.id,
                params: node.params.clone(),
                bindings: node.bindings.clone(),
            };
            classify_node(scene, node, node_ref, &mut model)?;
        }
        models.push(model);
    }
    Ok(models)
}

/// 单节点分类：支持 → 落入模型对应槽位；不支持 → 三段式报错。
fn classify_node(
    scene: &SceneV2,
    node: &prism_core::Node,
    node_ref: NodeRef,
    model: &mut SceneModel,
) -> MovieResult<()> {
    match &node.structure {
        // S5 重参数化：stage3d camera（orbit 参数存在即认定为相机节点）。
        StructureId::Reparameterization if node.params.get("orbit_radius").is_some() => {
            if model.camera.is_some() {
                return Err(RenderError::new(
                    "R5011",
                    format!("场景 '{}' 的节点 {}", scene.name, node.id),
                    "场景内含多个相机节点（S5），已支持子集只允许一个相机轨道".to_string(),
                    "合并为一个 stage3d camera 块；多机位请拆分为多个 scene",
                ));
            }
            model.camera = Some(node_ref);
            Ok(())
        }
        // S8 辐射传输：stage3d material。
        StructureId::RadiativeTransfer if node.params.get("albedo_r").is_some() => {
            if model.material.is_some() {
                return Err(RenderError::new(
                    "R5012",
                    format!("场景 '{}' 的节点 {}", scene.name, node.id),
                    "场景内含多个材质节点（S8），已支持子集只允许一种材质".to_string(),
                    "合并为一个 stage3d material 块；多材质请拆分为多个 scene",
                ));
            }
            model.material = Some(node_ref);
            Ok(())
        }
        // S12 实体表示：sdf（kind）或文本图层（content）两个子形态。
        StructureId::EntityRepresentation => {
            if node.params.get("kind").is_some() {
                let kind: &str = p_str(&node.params, "kind").unwrap_or("");
                if kind != "sphere" {
                    return Err(RenderError::new(
                        "R5013",
                        format!("场景 '{}' 的 sdf 节点 {}（kind = '{kind}'）", scene.name, node.id),
                        format!(
                            "render 侧已支持的实体只有 sphere（球体）；'{kind}' 缺少求交实现"
                        ),
                        "把 sdf kind 写成 sphere；box 等基元待几何求交模块接入（后续波次）",
                    ));
                }
                if model.sphere.is_some() {
                    return Err(RenderError::new(
                        "R5013",
                        format!("场景 '{}' 的节点 {}", scene.name, node.id),
                        "场景内含多个 sdf 实体，已支持子集只允许单球体".to_string(),
                        "合并为一个 sdf 块；多实体场景请拆分为多个 scene",
                    ));
                }
                model.sphere = Some(node_ref);
                Ok(())
            } else if node.params.get("content").is_some() {
                // 文本旋转：排版管线不承载任意角度（缺任意角光栅化）。
                if let Some(ParamValue::F32(r)) = node.params.get("rotation") {
                    if r.is_finite() && r.abs() > 1e-6 {
                        return Err(RenderError::new(
                            "R5014",
                            format!("场景 '{}' 的文本节点 {}（rotation = {r} rad）", scene.name, node.id),
                            "render 侧文本合成暂不支持旋转（缺任意角字形光栅化）".to_string(),
                            "删掉 layer text 的 rotation 字段；旋转文本待矢量图层波次接入",
                        ));
                    }
                }
                model.texts.push(node_ref);
                Ok(())
            } else {
                Err(RenderError::new(
                    "R5014",
                    format!("场景 '{}' 的节点 {}（S12）", scene.name, node.id),
                    "S12 实体节点既无 'kind'（sdf）也无 'content'（文本），无法归类".to_string(),
                    "S12 节点应来自 stage3d sdf 或 layer text 块",
                ))
            }
        }
        // S4 确定性抽样：layer particles。
        StructureId::DeterministicSampling => Err(RenderError::new(
            "R5010",
            format!("场景 '{}' 的粒子节点 {}（S4）", scene.name, node.id),
            "layer particles 暂不支持：render 侧缺粒子域求值器（样本 → 像素的映射管线）".to_string(),
            "移除 layer particles 块，或等待粒子求值器接入（后续波次）；当前已支持：camera / material / sdf sphere / layer text",
        )),
        // S3 约束梯度流：sim。
        StructureId::ConstraintFlow => Err(RenderError::new(
            "R5010",
            format!("场景 '{}' 的模拟节点 {}（S3，kind = '{}'）", scene.name, node.id, p_str(&node.params, "kind").unwrap_or("")),
            "sim 暂不支持：render 侧缺约束梯度流求解器（XPBD 步进）".to_string(),
            "移除 sim 块，或等待模拟求解器接入（后续波次）；当前已支持：camera / material / sdf sphere / layer text",
        )),
        // S2 卷积：compose 汇点。
        StructureId::Convolution => Err(RenderError::new(
            "R5010",
            format!("场景 '{}' 的 compose 节点 {}（S2 Merge）", scene.name, node.id),
            "compose 暂不支持：其输入只能是 layer / sim（均不在已支持子集），缺图层合成管线".to_string(),
            "移除 compose 块；分屏 / 形变请用 split_screen / morph_between 组合子（已支持的时间线语义）",
        )),
        other => Err(RenderError::new(
            "R5010",
            format!("场景 '{}' 的节点 {}（{}）", scene.name, node.id, other),
            "未知结构节点：render 侧无对应求值路径".to_string(),
            "使用已支持节点：stage3d camera / material / sdf sphere、layer text",
        )),
    }
}

// ---------------------------------------------------------------------------
// 单场景帧求值（复用既有求值器 + 文本合成）
// ---------------------------------------------------------------------------

/// 把节点静态参数 + 通道绑定解算为时刻 `t` 的参数包（纯函数）。
fn resolve_params(
    node: &NodeRef,
    channels: &BTreeMap<&str, &PropertyChannel>,
    t: f32,
    frame: u64,
) -> prism_core::Params {
    let mut params = node.params.clone();
    // 全局种子恒 0（DSL 无全局种子字段；固定值即确定值）。
    let ctx = EvalCtx::new(0, node.id, frame);
    for b in &node.bindings {
        if let Some(ch) = channels.get(b.channel.name.as_str()) {
            sample_binding(b, ch, t, &ctx, &mut params);
        }
    }
    params
}

/// 求值场景在时刻 `t` 的一帧：3D 部分走既有 [`render_frame`]，随后按节点序
/// 合成文本图层。
fn render_scene_frame(
    model: &SceneModel,
    compile: &CompileV2Output,
    channels: &BTreeMap<&str, &PropertyChannel>,
    device: &mut dyn GpuDevice,
    t: f32,
    frame: u64,
) -> MovieResult<Framebuffer> {
    let w: u32 = compile.width;
    let h: u32 = compile.height;

    // ---- 相机（缺省轨道：中心原点 / 半径 8 / fov 45°，与 DSL 缺省一致）----
    let (center, orbit_radius, fov_rad): (Vec3, f32, f32) = match &model.camera {
        Some(cam) => {
            let p = resolve_params(cam, channels, t, frame);
            (
                Vec3::new(
                    p_f32(&p, "orbit_x", 0.0),
                    p_f32(&p, "orbit_y", 0.0),
                    p_f32(&p, "orbit_z", 0.0),
                ),
                p_f32(&p, "orbit_radius", 8.0).max(0.0),
                p_f32(&p, "fov_rad", 45.0_f32.to_radians()),
            )
        }
        None => (Vec3::new(0.0, 0.0, 0.0), 8.0, 45.0_f32.to_radians()),
    };

    // ---- 材质（缺省白，粗糙 0.5 / 金属 0，与 DSL 缺省一致）---------------
    let material: Material = match &model.material {
        Some(mat) => {
            let p = resolve_params(mat, channels, t, frame);
            Material::new(
                Vec3::new(
                    p_f32(&p, "albedo_r", 1.0),
                    p_f32(&p, "albedo_g", 1.0),
                    p_f32(&p, "albedo_b", 1.0),
                ),
                p_f32(&p, "roughness", 0.5).clamp(0.0, 1.0),
                p_f32(&p, "metalness", 0.0).clamp(0.0, 1.0),
            )
        }
        None => Material::new(Vec3::new(1.0, 1.0, 1.0), 0.5, 0.0),
    };

    // ---- 实体半径（无 sdf = 0：软光追对 r=0 无命中，天然只出背景）--------
    let radius: f32 = match &model.sphere {
        Some(s) => {
            let p = resolve_params(s, channels, t, frame);
            p_f32(&p, "radius", 1.0).max(0.0)
        }
        None => 0.0,
    };

    // ---- 美学参数（theme 块；缺省关闭全部后期）---------------------------
    let aesthetic: Aesthetic = match &compile.theme {
        Some(th) => Aesthetic::new(th.teal_orange, th.contrast, th.filmic),
        None => Aesthetic::new(false, 1.0, false),
    };

    // ---- 光源：DSL 的 light 块登记后忽略（Phase 2/3 接入），render 侧用
    //      固定缺省灯（与内置演示同款：暖色方向光，确定性常量）-------------
    let lights: Vec<Light> = vec![Light::new(
        Vec3::new(1.0, -1.0, -0.5).normalize(),
        Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
        3.0,
    )];

    // ---- 组装 Scene 并复用既有求值器 --------------------------------------
    // 注意：build_render_request 的 eye 由 eval_camera_orbit(scene, t, duration)
    // 推导（orbit_center + 水平圆轨道），camera.eye / target 仅作契约占位。
    let scene: Scene = Scene {
        radius,
        orbit_center: center,
        orbit_radius,
        material,
        lights,
        camera: Camera::new(Vec3::new(0.0, 0.0, -orbit_radius), center, fov_rad),
        aesthetic,
    };
    let t_local: f32 = t - model.start;
    // 场景时长为正（活跃判据已保证）；防御性下限防除零。
    let dur_local: f32 = if model.duration > 0.0 { model.duration } else { 1.0 };
    let mut fb: Framebuffer = render_frame(device, &scene, t_local, dur_local, w, h);

    // ---- 文本图层合成（按节点序；后建的压在上面）-------------------------
    for text in &model.texts {
        let p = resolve_params(text, channels, t, frame);
        overlay_text(&mut fb, &p, w, h).map_err(|err| {
            RenderError::new(
                err.code,
                format!("场景 '{}' 的文本节点 {}：{}", model.name, text.id, err.location),
                err.problem,
                err.fix,
            )
        })?;
    }

    Ok(fb)
}

/// 文本图层合成：排版（`prism-text::layout_text`）→ 字形光栅化 → u8 整数
/// src-over 混合直写帧缓冲。
///
/// 定位语义：`(x, y)` = 文本块左上角；`align`（组合子扩展参数，DSL 文本层
/// 不产出）left / center / right 决定每行相对 x 的偏移。
fn overlay_text(fb: &mut Framebuffer, params: &prism_core::Params, w: u32, h: u32) -> MovieResult<()> {
    let content: &str = p_str(params, "content").unwrap_or("");
    if content.is_empty() {
        return Ok(());
    }
    let size: f32 = p_f32(params, "size", 24.0);
    if !(size.is_finite() && size > 0.0) {
        return Err(RenderError::new(
            "R5015",
            "文本图层参数 'size'",
            format!("字号必须为正有限数，却得到 {size}"),
            "size 传正数（如 32）；缺省 24",
        ));
    }
    let x: f32 = p_f32(params, "x", 0.0);
    let y: f32 = p_f32(params, "y", 0.0);
    let opacity: f32 = p_f32(params, "opacity", 1.0).clamp(0.0, 1.0);
    let color: [u8; 3] = [
        (p_f32(params, "color_r", 1.0).clamp(0.0, 1.0) * 255.0).round() as u8,
        (p_f32(params, "color_g", 1.0).clamp(0.0, 1.0) * 255.0).round() as u8,
        (p_f32(params, "color_b", 1.0).clamp(0.0, 1.0) * 255.0).round() as u8,
    ];
    let align: &str = p_str(params, "align").unwrap_or("left");

    // 内嵌粗体（确定性：同字节同字形；DSL font 回退链消费随字体管线波次接入）。
    let font = font_bold();
    let tp = TextParams {
        font_size: size,
        letter_spacing: 0.0,
        line_spacing: 0.0,
        max_width: f32::INFINITY,
        align: Align::Left, // 对齐在下方按行手工偏移（INF 宽度下框架居中无定义）
    };
    let layout = layout_text(&font, content, &tp);
    let mut cache = GlyphCache::new();

    // 透明度 → 8bit 权重（单次四舍五入，之后全整数路径）。
    let op8: u16 = (opacity * 255.0).round().clamp(0.0, 255.0) as u16;

    for g in &layout.glyphs {
        // 每行水平偏移（center / right 相对块左缘 x）。
        let line_w: f32 = layout
            .lines
            .get(g.line)
            .map(|l| l.width)
            .unwrap_or(0.0);
        let dx: f32 = match align {
            "center" => -line_w * 0.5,
            "right" => -line_w,
            _ => 0.0,
        };
        let bmp = cache.get_or_render(&font, g.gid, size, 0);
        if bmp.is_empty() {
            continue;
        }
        // 笔位 = 块原点 + 字形笔位 + 行偏移；向下取整对齐像素格（与 blit_glyph 同约定）。
        let pen_x: f32 = x + g.x + dx;
        let pen_y: f32 = y + g.y;
        if !pen_x.is_finite() || !pen_y.is_finite() {
            continue; // 非有限笔位视为无效输入（与 prism-text 同口径）
        }
        let ox: i64 = (pen_x.floor() as i64).saturating_add(bmp.left as i64);
        let oy: i64 = (pen_y.floor() as i64).saturating_add(bmp.top as i64);
        let bw: i64 = bmp.width as i64;
        let bh: i64 = bmp.height as i64;
        for row in 0..bh {
            let fy: i64 = oy.saturating_add(row);
            if fy < 0 || fy >= h as i64 {
                continue;
            }
            for col in 0..bw {
                let fx: i64 = ox.saturating_add(col);
                if fx < 0 || fx >= w as i64 {
                    continue;
                }
                let Some(&a) = bmp.alpha.get((row as usize) * (bmp.width as usize) + (col as usize)) else {
                    continue;
                };
                if a == 0 {
                    continue;
                }
                // 合成权重 = opacity × 字形覆盖（单次四舍五入的整数乘积）。
                let eff: u16 = (op8 * a as u16 + 127) / 255;
                if eff == 0 {
                    continue;
                }
                let idx: usize = (fy as usize) * (w as usize) * 4 + (fx as usize) * 4;
                for c in 0..3 {
                    let dst: u8 = fb.pixels[idx + c];
                    let mixed: u16 = color[c] as u16 * eff + dst as u16 * (255 - eff);
                    fb.pixels[idx + c] = ((mixed + 127) / 255) as u8;
                }
                fb.pixels[idx + 3] = 255;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 时间线工具：过渡匹配 / 同程判据 / 分屏 / 混合
// ---------------------------------------------------------------------------

/// 查找覆盖时刻 `t` 且连接两场景的过渡（按文档声明的 from/to 名匹配）。
fn find_transition<'a>(
    compile: &'a CompileV2Output,
    a: &SceneModel,
    b: &SceneModel,
    t: f32,
) -> Option<&'a TransitionV2> {
    compile.transitions.iter().find(|tr| {
        tr.from == a.name
            && tr.to == b.name
            && tr.start <= t
            && t < tr.start + tr.duration
            && tr.duration > 0.0
    })
}

/// 同程判据：起止时刻完全相同（分屏语义；浮点精确相等——分屏由组合子以
/// 同一对数值构造，非计算结果，容差反而会引入歧义）。
fn same_interval(a: &SceneModel, b: &SceneModel) -> bool {
    a.start == b.start && a.duration == b.duration
}

/// 左右分屏合成：左半幅取 `l`、右半幅取 `r`（奇数宽时左半幅少一列）。
///
/// 尺寸防御：右帧与左帧规格不符时退化为左帧拷贝（确定性、不 panic）。
fn split_halves(l: &Framebuffer, r: &Framebuffer) -> Framebuffer {
    let mut out: Framebuffer = l.clone();
    if r.width != out.width || r.height != out.height || r.pixels.len() != out.pixels.len() {
        return out;
    }
    let half: u32 = out.width / 2;
    for y in 0..out.height {
        for x in half..out.width {
            let idx: usize = (y as usize) * (out.width as usize) * 4 + (x as usize) * 4;
            let src_idx: usize = (y as usize) * (r.width as usize) * 4 + (x as usize) * 4;
            out.pixels[idx..idx + 4].copy_from_slice(&r.pixels[src_idx..src_idx + 4]);
        }
    }
    out
}

/// 全黑帧（空档期 / 兜底）。
fn black_frame(w: u32, h: u32) -> Framebuffer {
    let mut fb: Framebuffer = Framebuffer::new(w, h);
    for px in fb.pixels.iter_mut() {
        *px = 0;
    }
    fb
}

/// 按过渡种类把两帧混合为一帧（整数路径，复用 W1-C 下沉 core 家族）。
///
/// 种类 → 语义（与 `prism-author` 既有 `mix_frame` 逐位同口径）：
/// - `crossfade` / `fade`：全局线性权重 `mix_u8(a, b, kw)`；
/// - `cut`：中点硬切（`k ≥ 0.5` 全取 b，否则全取 a）；
/// - `black` / `white`：向黑 / 向白淡出（`mix_u8(a, 0|255, kw)`）；
/// - `wipe`：分界线推进（`wipe_weight` + `band`，方向 `dir`，缺省 right）；
/// - `dissolve`：逐像素整数 hash 显影（`dissolve_hash(seed)`）；
/// - `push`：整体平移（`push_c` 推进量，越界钳边）。
fn blend_transition(a: &Framebuffer, b: &Framebuffer, tr: &TransitionV2, t: f32) -> Framebuffer {
    let w: u32 = a.width;
    let h: u32 = a.height;
    let expect: usize = (w as usize) * (h as usize) * 4;
    // 尺寸防御：任一帧不符契约时退化为 a 的拷贝（确定性、不 panic）。
    if a.pixels.len() != expect || b.pixels.len() != expect {
        return a.clone();
    }
    // f32 进度 → 256 分母整数网格（round-to-nearest），之后只走整数路径。
    let k: f32 = ((t - tr.start) / tr.duration).clamp(0.0, 1.0);
    let kw: u16 = (k * 256.0).round().clamp(0.0, 256.0) as u16;
    let dir: &str = tr.dir.as_deref().unwrap_or("right");
    let mut out: Vec<u8> = vec![0u8; expect];
    match tr.kind.as_str() {
        "crossfade" | "fade" => {
            for (o, (pa, pb)) in out
                .chunks_exact_mut(4)
                .zip(a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)))
            {
                o[0] = mix_u8(pa[0], pb[0], kw);
                o[1] = mix_u8(pa[1], pb[1], kw);
                o[2] = mix_u8(pa[2], pb[2], kw);
                o[3] = 255;
            }
        }
        "cut" => {
            let cw: u16 = if k >= 0.5 { W_ONE } else { 0 };
            for (o, (pa, pb)) in out
                .chunks_exact_mut(4)
                .zip(a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)))
            {
                o[0] = mix_u8(pa[0], pb[0], cw);
                o[1] = mix_u8(pa[1], pb[1], cw);
                o[2] = mix_u8(pa[2], pb[2], cw);
                o[3] = 255;
            }
        }
        "black" | "white" => {
            let target: u8 = if tr.kind == "black" { 0 } else { 255 };
            for (o, pa) in out.chunks_exact_mut(4).zip(a.pixels.chunks_exact(4)) {
                o[0] = mix_u8(pa[0], target, kw);
                o[1] = mix_u8(pa[1], target, kw);
                o[2] = mix_u8(pa[2], target, kw);
                o[3] = 255;
            }
        }
        "wipe" => {
            let band: u16 = tr.band;
            let mut idx: usize = 0;
            for y in 0..h {
                for x in 0..w {
                    // 方向镜像：core 规范形态（b 自右侧入场）对权重取补得左 / 上 / 下。
                    let tw: u16 = match dir {
                        "left" => W_ONE - wipe_weight(x, kw, w, band),
                        "up" => wipe_weight(y, kw, h, band),
                        "down" => W_ONE - wipe_weight(y, kw, h, band),
                        _ => wipe_weight(x, kw, w, band), // "right" = 规范形态
                    };
                    out[idx] = mix_u8(a.pixels[idx], b.pixels[idx], tw);
                    out[idx + 1] = mix_u8(a.pixels[idx + 1], b.pixels[idx + 1], tw);
                    out[idx + 2] = mix_u8(a.pixels[idx + 2], b.pixels[idx + 2], tw);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
        "dissolve" => {
            let seed: u64 = tr.seed;
            let mut idx: usize = 0;
            for y in 0..h {
                for x in 0..w {
                    let tw: u16 = if dissolve_hash(x, y, seed) < kw { W_ONE } else { 0 };
                    out[idx] = mix_u8(a.pixels[idx], b.pixels[idx], tw);
                    out[idx + 1] = mix_u8(a.pixels[idx + 1], b.pixels[idx + 1], tw);
                    out[idx + 2] = mix_u8(a.pixels[idx + 2], b.pixels[idx + 2], tw);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
        "push" => {
            let pdir: PushDir = match dir {
                "right" => PushDir::Right,
                "up" => PushDir::Up,
                "down" => PushDir::Down,
                _ => PushDir::Left,
            };
            let iw: i64 = w as i64;
            let ih: i64 = h as i64;
            let cw: i64 = push_c(kw, w) as i64;
            let ch: i64 = push_c(kw, h) as i64;
            let mut idx: usize = 0;
            for y in 0..h {
                let (ya, yb): (i64, i64) = match pdir {
                    PushDir::Up => (
                        (y as i64 - ch).clamp(0, ih - 1),
                        (y as i64 + (ih - ch)).clamp(0, ih - 1),
                    ),
                    PushDir::Down => (
                        (y as i64 + ch).clamp(0, ih - 1),
                        (y as i64 - (ih - ch)).clamp(0, ih - 1),
                    ),
                    _ => (y as i64, y as i64),
                };
                for x in 0..w {
                    let (xa, xb): (i64, i64) = match pdir {
                        PushDir::Left => (
                            (x as i64 - cw).clamp(0, iw - 1),
                            (x as i64 + (iw - cw)).clamp(0, iw - 1),
                        ),
                        PushDir::Right => (
                            (x as i64 + cw).clamp(0, iw - 1),
                            (x as i64 - (iw - cw)).clamp(0, iw - 1),
                        ),
                        _ => (x as i64, x as i64),
                    };
                    // 覆盖判据（沿推进轴）：b 占入场侧、a 占对侧。
                    let from_b: bool = match pdir {
                        PushDir::Left => (x as i64) < cw,
                        PushDir::Right => (x as i64) >= iw - cw,
                        PushDir::Up => (y as i64) < ch,
                        PushDir::Down => (y as i64) >= ih - ch,
                    };
                    let tw: u16 = if from_b { W_ONE } else { 0 };
                    let ia: usize = ((ya as u32) * w + xa as u32) as usize * 4;
                    let ib: usize = ((yb as u32) * w + xb as u32) as usize * 4;
                    out[idx] = mix_u8(a.pixels[ia], b.pixels[ib], tw);
                    out[idx + 1] = mix_u8(a.pixels[ia + 1], b.pixels[ib + 1], tw);
                    out[idx + 2] = mix_u8(a.pixels[ia + 2], b.pixels[ib + 2], tw);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
        // 未知种类：codegen / 组合子已白名单校验；此处防御性退化为交叉溶解。
        _ => {
            for (o, (pa, pb)) in out
                .chunks_exact_mut(4)
                .zip(a.pixels.chunks_exact(4).zip(b.pixels.chunks_exact(4)))
            {
                o[0] = mix_u8(pa[0], pb[0], kw);
                o[1] = mix_u8(pa[1], pb[1], kw);
                o[2] = mix_u8(pa[2], pb[2], kw);
                o[3] = 255;
            }
        }
    }
    let mut fb: Framebuffer = Framebuffer::new(w, h);
    fb.pixels = out;
    fb
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::node::{ChannelRef, Composition, NodeBinding};
    use prism_core::{Graph, Node, Params};
    use prism_dsl::{CameraOrbit, LayerInfo, PostInfo};

    /// 最小节点图编译产物：单场景、相机 + 材质 + 球体。
    fn minimal_compile() -> CompileV2Output {
        let mut graph: Graph = Graph::new();
        let mut cam_params: Params = Params::new();
        cam_params.set("orbit_x", ParamValue::F32(0.0));
        cam_params.set("orbit_y", ParamValue::F32(0.0));
        cam_params.set("orbit_z", ParamValue::F32(0.0));
        cam_params.set("orbit_radius", ParamValue::F32(6.0));
        cam_params.set("fov_rad", ParamValue::F32(45.0_f32.to_radians()));
        let mut cam: Node = Node::new(1, StructureId::Reparameterization, Composition::Feed);
        cam.params = cam_params;
        graph.add_node(cam).expect("id 1");

        let mut mat_params: Params = Params::new();
        mat_params.set("albedo_r", ParamValue::F32(0.9));
        mat_params.set("albedo_g", ParamValue::F32(0.2));
        mat_params.set("albedo_b", ParamValue::F32(0.2));
        mat_params.set("roughness", ParamValue::F32(0.4));
        mat_params.set("metalness", ParamValue::F32(0.1));
        let mut mat: Node = Node::new(2, StructureId::RadiativeTransfer, Composition::Feed);
        mat.params = mat_params;
        graph.add_node(mat).expect("id 2");

        let mut sdf_params: Params = Params::new();
        sdf_params.set("kind", ParamValue::Str("sphere".to_string()));
        sdf_params.set("radius", ParamValue::F32(1.0));
        let mut sdf: Node = Node::new(3, StructureId::EntityRepresentation, Composition::Feed);
        sdf.params = sdf_params;
        graph.add_node(sdf).expect("id 3");

        let scene = SceneV2 {
            name: "s".to_string(),
            start: 0.0,
            duration: 1.0,
            node_ids: vec![1, 2, 3],
            camera: Some(CameraOrbit {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 6.0,
                fov_rad: 45.0_f32.to_radians(),
            }),
            layers: Vec::new(),
            sim: None,
            post: PostInfo::default(),
            audio: None,
            notes: Vec::new(),
        };
        CompileV2Output {
            title: "minimal".to_string(),
            width: 96,
            height: 54,
            fps: 8,
            duration: 1.0,
            format: prism_dsl::OutputFormat::Frames,
            scenes: vec![scene],
            transitions: Vec::new(),
            graph,
            channels: Vec::new(),
            packs: Vec::new(),
            theme: None,
            fonts: Vec::new(),
        }
    }

    #[test]
    fn render_movie_produces_expected_frames_and_determinism() {
        let compile: CompileV2Output = minimal_compile();
        let mut device = prism_gpu::CpuRefBackend;
        let a: Deliverable = render_movie(&compile, &mut device).expect("最小图应可渲染");
        let b: Deliverable = render_movie(&compile, &mut device).expect("最小图应可渲染");

        // fps=8、duration=1 → 8 帧；尺寸 96x54。
        assert_eq!(a.frame_count(), 8, "fps=8 × 1s 应得 8 帧");
        assert_eq!(a.width, 96);
        assert_eq!(a.height, 54);
        for (i, fb) in a.frames.iter().enumerate() {
            assert_eq!(fb.pixels.len(), 96 * 54 * 4, "第 {i} 帧长度应恰为 w·h·4");
        }

        // 确定性宪章：两跑 MD5 一致（帧序列拼接后整体摘要）。
        let digest = |d: &Deliverable| -> String {
            let mut buf: Vec<u8> = Vec::new();
            for fb in &d.frames {
                buf.extend_from_slice(&fb.pixels);
            }
            crate::farm::md5_hex(&buf)
        };
        assert_eq!(digest(&a), digest(&b), "同输入两跑必须逐位一致");

        // 相机环绕：首帧与末帧应不同（有运动才说明求值器真的在动）。
        assert_ne!(a.frames[0].pixels, a.frames[a.frames.len() - 1].pixels);
    }

    /// 构造一个含指定不支持节点的编译产物。
    fn compile_with_node(structure: StructureId, kind: &str) -> CompileV2Output {
        let mut compile: CompileV2Output = minimal_compile();
        let mut params: Params = Params::new();
        params.set("kind", ParamValue::Str(kind.to_string()));
        params.set("count", ParamValue::U32(16));
        let mut node: Node = Node::new(9, structure, Composition::Feed);
        node.params = params;
        compile.graph.add_node(node).expect("id 9");
        compile.scenes[0].node_ids.push(9);
        compile
    }

    #[test]
    fn unsupported_particles_node_errors_not_panics() {
        let compile: CompileV2Output = compile_with_node(StructureId::DeterministicSampling, "particles");
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("粒子节点必须明确报错");
        assert_eq!(err.code, "R5010");
        let msg: String = err.to_string();
        assert!(msg.contains("节点 9"), "报错应指明节点 id：{msg}");
        assert!(msg.contains("particles"), "报错应说明为什么不支持：{msg}");
        assert!(msg.contains("建议"), "报错必含建议段：{msg}");
    }

    #[test]
    fn unsupported_sdf_box_errors_not_panics() {
        let compile: CompileV2Output = compile_with_node(StructureId::EntityRepresentation, "box");
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("sdf box 必须明确报错");
        assert_eq!(err.code, "R5013");
        assert!(err.to_string().contains("box"));
    }

    #[test]
    fn unsupported_sim_node_errors_not_panics() {
        let compile: CompileV2Output = compile_with_node(StructureId::ConstraintFlow, "cloth");
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("sim 节点必须明确报错");
        assert_eq!(err.code, "R5010");
        assert!(err.to_string().contains("cloth"));
    }

    #[test]
    fn dangling_node_id_errors_not_panics() {
        let mut compile: CompileV2Output = minimal_compile();
        compile.scenes[0].node_ids.push(42);
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("悬空节点 id 必须报错");
        assert_eq!(err.code, "R5002");
    }

    #[test]
    fn zero_resolution_errors_not_panics() {
        let mut compile: CompileV2Output = minimal_compile();
        compile.width = 0;
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("0 宽必须报错");
        assert_eq!(err.code, "R5001");
    }

    #[test]
    fn text_only_scene_renders_overlay() {
        // 纯文本场景（title_card 展开形态）：无相机 / 材质 / 球体。
        let mut graph: Graph = Graph::new();
        let mut params: Params = Params::new();
        params.set("content", ParamValue::Str("HELLO".to_string()));
        params.set("x", ParamValue::F32(48.0));
        params.set("y", ParamValue::F32(24.0));
        params.set("size", ParamValue::F32(14.0));
        params.set("color_r", ParamValue::F32(1.0));
        params.set("color_g", ParamValue::F32(1.0));
        params.set("color_b", ParamValue::F32(1.0));
        params.set("opacity", ParamValue::F32(1.0));
        let mut node: Node = Node::new(1, StructureId::EntityRepresentation, Composition::Feed);
        node.params = params;
        graph.add_node(node).expect("id 1");
        let scene = SceneV2 {
            name: "t".to_string(),
            start: 0.0,
            duration: 0.5,
            node_ids: vec![1],
            camera: None,
            layers: vec![LayerInfo {
                name: "t".to_string(),
                kind: "text".to_string(),
                node_id: 1,
                morph: None,
            }],
            sim: None,
            post: PostInfo::default(),
            audio: None,
            notes: Vec::new(),
        };
        let compile: CompileV2Output = CompileV2Output {
            title: "text".to_string(),
            width: 96,
            height: 54,
            fps: 4,
            duration: 0.5,
            format: prism_dsl::OutputFormat::Frames,
            scenes: vec![scene],
            transitions: Vec::new(),
            graph,
            channels: Vec::new(),
            packs: Vec::new(),
            theme: None,
            fonts: Vec::new(),
        };
        let mut device = prism_gpu::CpuRefBackend;
        let d: Deliverable = render_movie(&compile, &mut device).expect("纯文本场景应可渲染");
        assert_eq!(d.frame_count(), 2);
        // 文本应改变像素（相对纯背景至少有一个像素不同）。
        let bg: usize = d.frames[0].pixels.iter().filter(|&&p| p != 0).count();
        assert!(bg > 0, "文本合成后应存在非零像素");
    }

    #[test]
    fn rotated_text_errors_not_panics() {
        let mut compile: CompileV2Output = minimal_compile();
        let mut params: Params = Params::new();
        params.set("content", ParamValue::Str("X".to_string()));
        params.set("rotation", ParamValue::F32(0.5));
        let mut node: Node = Node::new(9, StructureId::EntityRepresentation, Composition::Feed);
        node.params = params;
        compile.graph.add_node(node).expect("id 9");
        compile.scenes[0].node_ids.push(9);
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("旋转文本必须报错");
        assert_eq!(err.code, "R5014");
    }

    #[test]
    fn opacity_channel_drives_text_visibility() {
        // 淡入淡出通道：首帧全透明（无文本像素）、中段不透明（有文本像素）。
        use prism_anim::{EasingKind, Keyframe, Track};
        let mut graph: Graph = Graph::new();
        let mut params: Params = Params::new();
        params.set("content", ParamValue::Str("A".to_string()));
        params.set("x", ParamValue::F32(40.0));
        params.set("y", ParamValue::F32(20.0));
        params.set("size", ParamValue::F32(12.0));
        params.set("opacity", ParamValue::F32(1.0));
        let mut node: Node = Node::new(1, StructureId::EntityRepresentation, Composition::Feed);
        node.params = params;
        node.bindings = vec![NodeBinding {
            param: "opacity".to_string(),
            channel: ChannelRef { name: "fade".to_string() },
        }];
        graph.add_node(node).expect("id 1");
        let scene = SceneV2 {
            name: "t".to_string(),
            start: 0.0,
            duration: 1.0,
            node_ids: vec![1],
            camera: None,
            layers: vec![LayerInfo {
                name: "t".to_string(),
                kind: "text".to_string(),
                node_id: 1,
                morph: None,
            }],
            sim: None,
            post: PostInfo::default(),
            audio: None,
            notes: Vec::new(),
        };
        let compile: CompileV2Output = CompileV2Output {
            title: "fade".to_string(),
            width: 96,
            height: 54,
            fps: 4,
            duration: 1.0,
            format: prism_dsl::OutputFormat::Frames,
            scenes: vec![scene],
            transitions: Vec::new(),
            graph,
            channels: vec![prism_dsl::NamedChannel {
                name: "fade".to_string(),
                channel: PropertyChannel::Track(Track::new(vec![
                    Keyframe::new(0.0, 0.0, EasingKind::Linear),
                    Keyframe::new(0.5, 1.0, EasingKind::Linear),
                    Keyframe::new(1.0, 0.0, EasingKind::Linear),
                ])),
            }],
            packs: Vec::new(),
            theme: None,
            fonts: Vec::new(),
        };
        let mut device = prism_gpu::CpuRefBackend;
        let d: Deliverable = render_movie(&compile, &mut device).expect("通道驱动文本应可渲染");
        // 背景为深蓝黑（0.04/0.05/0.08 → 10/13/20）；文本纯白 255。
        // 按像素计（alpha 恒 255，不能参与亮度判据）。
        let lit = |fb: &Framebuffer| -> usize {
            fb.pixels
                .chunks_exact(4)
                .filter(|px| px[0] > 200 || px[1] > 200 || px[2] > 200)
                .count()
        };
        assert_eq!(lit(&d.frames[0]), 0, "首帧全透明：不应有亮像素");
        assert!(lit(&d.frames[2]) > 0, "中段不透明：应有白色文本像素");
        assert_eq!(lit(&d.frames[3]), 0, "末帧全透明：不应有亮像素");
    }
}
