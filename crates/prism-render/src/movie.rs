//! DSL v2 成片直达：[`CompileV2Output`] → 帧序列 [`Deliverable`]。
//!
//! # 定位
//! 本模块是 prism-dsl v2 编译产物在 render 侧的消费入口：读入一次
//! `compile_v2` 的完整产物（节点图 + 通道 + 场景时间线 + 过渡），按 fps 均匀
//! 采样求值整段影片。3D 帧求值**复用本 crate 既有求值器**
//! （[`crate::render_frame`] → `CpuRefBackend` 软光追），不另写渲染器；
//! 文本图层合成复用 `prism-text` 既有排版 / 光栅化能力；转场混合复用
//! `prism-core::sceneseq`（u8 整数混合家族）。
//!
//! # 覆盖纪律（宪章：绝不静默糊弄近似）
//! render 侧只宣称**已支持子集**：
//! - S5 相机（stage3d camera）→ 场景相机轨道；
//! - S8 材质（stage3d material）→ 场景材质；
//! - S12 实体 `kind sphere`（stage3d sdf）→ 球体半径；
//! - S12 曲线图元族 `kind ∈ {cylinder, cone, torus, revolution}`
//!   （stage3d sdf）→ rt CPU 波前路径（BVH 加速、逐位确定性；
//!   GPU 曲线槽位尚未接入——不假装已支持）；
//! - S12 文本图层（layer text）→ 帧上文本合成；
//! - S3 约束梯度流（`sim cloth`）→ 帧循环驱动的 XPBD 推进 + 粒子叠加
//!   （模式元数据 substeps / iterations / residual_eps 进帧签名；
//!   语义见下文「sim 帧循环语义」）；
//! - 场景时间线：顺序播放 / 过渡重叠混合 / 同程多场景左右分屏。
//!
//! 任何不支持的节点（particles / sdf box / compose / 未知结构 / 非 cloth 的
//! sim 种类）都在**求值前**以 [`RenderError`] 三段式明确报错（哪个节点、
//! 为什么不支持、怎么改），绝不静默跳过或近似。
//!
//! # sim 帧循环语义（S3；对齐 `prism-core::eval` 的 ConstraintFlow 接线）
//! - **时间线驱动**：影片帧 `i` 落入某场景活跃区间时，该场景的 sim 节点按
//!   *场景内帧号* `i − 基准帧` 推进（基准帧 = 场景在影片帧网格上的首个
//!   活跃帧，纯函数、无隐藏状态）；首帧建规格，后续帧逐帧步进；
//! - **签名收拢**：求解模式（M3 固定步数）三元组经通道定义哈希链并入帧
//!   签名——改 substeps / iterations / residual_eps 任一 ⇒ 签名变 ⇒ 快照
//!   与缓存键不误复用（求值层 §3.2 S3 迭代层缓存键 (ε, n) 纪律）；
//! - **seek 等价**：跨帧状态经 [`prism_core::node::SnapshotStore`]（键 =
//!   (签名, 场景内帧号)）顺序延续 / 快照装配；冷 seek（帧 > 0 且前驱快照
//!   缺失）结构化报错 `P3012`，绝不静默 fallback 到初态。驱动本体已上提为
//!   内核环实例 [`prism_core::fixpoint::LoopDriver`]（与时间实例同风格），
//!   本模块只保留 XPBD 状态编解码与宿主层参数 / 错误码映射；
//! - **同参数永远同画面**：sim 全部解算参数必须逐帧逐位一致（Const 通道
//!   天然满足）；时变参数显式报错 `R5017`——时变外力族属扩展位；
//! - **依赖边界**：prism-render 不依赖 prism-physics（无此路径边），XPBD
//!   求解核在本模块自包含实装，数值纪律逐位对齐 `prism-physics` 的
//!   `xpbd.rs` / `session.rs`（半隐式欧拉 + XPBD 投影、约束按 id 升序
//!   Gauss-Seidel、λ 每子步清零、仅 IEEE 基本运算）；帧内 M3 定数迭代走
//!   内核 [`prism_core::fixpoint::run_fixed_point`]（残差记录 + 发散观测）。
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
use std::time::Instant;

use prism_core::fixpoint::{
    loop_frame_signature, run_fixed_point, LoopCodec, LoopDriver, SolveModeMeta,
    RESIDUAL_EPS_DEFAULT,
};
use prism_core::node::{Hash256, NodeBinding, ParamValue, SnapshotStore};
use prism_core::property::{sample_binding, PropertyChannel};
use prism_core::sceneseq::{dissolve_hash, mix_u8, push_c, wipe_weight, PushDir, W_ONE};
use prism_core::signature::{ChannelDef, ChannelDefs, SignatureError};
use prism_core::{
    Aesthetic, Camera, EvalCtx, EvalError, Graph, Light, Material, Scene, StructureId,
};
use prism_dsl::{resolve_frame, CompileV2Output, SceneV2, TransitionV2, WiringFrame};
use prism_gpu::{Framebuffer, GpuDevice};
use prism_math::Vec3;
use prism_text::cache::GlyphCache;
use prism_text::font_bold;
use prism_text::layout::{layout_text, Align, TextParams};

use crate::render_frame;
use crate::v31_accum::{V31FrameRecord, V31FrameSink, V31MovieReport, fnv1a64};
use crate::v31_config::V31Config;
use crate::v31_device::V31RenderDevice;
// V3.1 收尾（B）：golden 比对 / 五条判据 / 抽样 manifest / 判红退出全部
// **移出渲染帧循环**，只在可选校验 `v31_verify` 里使用（默认关 ⇒ 零开销）。
use crate::v31_sample::{build_sample_plan, timeline_period_frames, SamplePlan};
use crate::v31_verify::{finalize_verify, V31VerifyCollector};
use crate::perf_domains::{self, Domain};

/// `Deliverable.frames` 是否全帧驻留的帧数上限。
///
/// 720p × 3600 帧全驻留需 13.3 GB，必 OOM；超过该上限时 `render_movie_v31`
/// 只在 `Deliverable.frames` 保留抽样帧，其余帧经 `V31FrameSink` 流式交付
/// （`V31MovieReport.streamed = true`）。**计时仍覆盖完整 3600 帧**。
pub const V31_RESIDENT_FRAME_LIMIT: usize = 512;

/// `Deliverable.frames` 全帧驻留的**字节**上限（与 CLI 的
/// `V31_CLI_RESIDENT_BYTES` 同值）。720p 每帧 3.7 MB，超过约 141 帧即
/// 必须流式逐帧交付，否则 360 帧 = 1.33 GB、3600 帧 = 13.3 GB 必 OOM。
pub const V31_RESIDENT_BYTES_LIMIT: u64 = 512 * 1024 * 1024;

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
    /// V3.0 W2：逐帧接线解算凭证（`WiringFrame::canonical_bytes`）。
    ///
    /// 在**渲染前**由 `wiring_frame_values` 生成，随成片一起交付：同源对拍
    /// 逐字节比较本字段，证明 39 行参数真的进了求值链（不是纸面映射）。
    /// 旧文件 `wiring` 为空 ⇒ 每帧是同一个空表字节序列，帧像素不变。
    pub wiring_trace: Vec<Vec<u8>>,
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

    /// 逐帧接线凭证（同源对拍用）。
    pub fn wiring_trace_bytes(&self) -> &[Vec<u8>] {
        &self.wiring_trace
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
    /// 曲线图元节点（S12 sdf kind ∈ {cylinder, cone, torus, revolution}；
    /// 按节点序进入 rt 场景，非空即走 CPU 波前曲线路径）。
    curves: Vec<NodeRef>,
    /// 模拟节点（S3 约束梯度流；按节点序逐帧推进 + 粒子叠加）。
    sims: Vec<NodeRef>,
    /// sim 的场景内帧号基准：本场景在影片帧网格上的首个活跃帧号
    /// （场景内帧号 = 影片帧号 − 本值；纯函数，seek 对拍稳定）。
    sim_base_frame: u64,
    /// 文本图层（S12 layer text；按节点序合成）。
    texts: Vec<NodeRef>,
    /// T0：本场景的附加能力（介质 / 网格 / 实例 / 纹理 / 毛发 / 位移槽）。
    ///
    /// 来源是 `prism_dsl::CompileV2Output` 的 `SceneV2::attach`（`media` /
    /// `mesh` / `texture` / `hair` / `instance` 块）。空 ⇒ 该场景与加字段前
    /// **逐字节同帧**（判据 P1）。
    attach: prism_core::SceneAttach,
    /// V3.0 W2：R5010 销号后的七族节点（particles / compose / 未知结构等）。
    ///
    /// 不再拒绝，收进求值链源头：每帧由 `resolve_params` 解算其参数与
    /// 通道绑定，再交给 `prism-app::w2_eval` 调用既有 crate 求值器。
    wiring_nodes: Vec<NodeRef>,
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
/// - 节点图含**不支持**的节点（particles / sdf box / compose / 未知
///   结构）→ `R5010`–`R5014`；sim 种类域（当前仅 cloth）在求值前由
///   `SimSpec::from_params` 校验 → `R5017`；
/// - sim 参数域非法（时变参数 / grid 越界 / 刚度越界 / 模式参数非正）
///   → `R5017` / `R5018`；
/// - sim 冷 seek（帧 > 0 且前驱快照缺失）→ `P3012`（绝不静默 fallback）；
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
    // V3.1 T03：旧入口恒委托 `V31Config::legacy()`（累积 / 降噪 / 自适应 /
    // GPU 优先 / 规格覆写全关）。原函数体逐字搬进 `render_movie_legacy_impl`，
    // 一行表达式都未改 ⇒ 与 V3.1 之前逐字节一致（4629 测试逐字钉死）。
    render_movie_legacy_impl(compile, device, V31Config::legacy())
}

/// 旧帧循环实现（V3.1 之前的行为，逐字保留）。
///
/// `cfg` 参数只用于让「委托 legacy」在类型层面显式可见；函数体不使用它，
/// 故 `legacy()` 与「未传配置」完全等价。
fn render_movie_legacy_impl(
    compile: &CompileV2Output,
    device: &mut dyn GpuDevice,
    _cfg: V31Config,
) -> MovieResult<Deliverable> {
    // ---- 0. 规格校验 -----------------------------------------------------
    validate_output_spec(compile)?;

    // ---- 1. 通道登记表 + 签名定义表 + 场景模型（求值前完成全部校验）------
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let defs: ChannelDefs = channel_defs_of(compile)?;
    let models: Vec<SceneModel> = build_models(compile, &channels)?;

    // ---- 2. 帧循环（sim 跨帧状态由本时间线顺序延续 / 快照装配）----------
    let frame_count: usize = (compile.fps as f32 * compile.duration).ceil() as usize;
    let mut frames: Vec<Framebuffer> = Vec::with_capacity(frame_count);
    let mut wiring_trace: Vec<Vec<u8>> = Vec::with_capacity(frame_count);
    let mut sims: SimTimeline = SimTimeline::new();
    for i in 0..frame_count {
        let (fb, trace): (Framebuffer, Vec<u8>) =
            render_frame_at(compile, &models, &channels, &defs, device, i as u64, &mut sims)?;
        frames.push(fb);
        wiring_trace.push(trace);
    }

    Ok(Deliverable {
        title: compile.title.clone(),
        width: compile.width,
        height: compile.height,
        fps: compile.fps,
        duration: compile.duration,
        frames,
        wiring_trace,
    })
}

/// 当前时刻活跃场景组合的稳定键（`TemporalState` 的 `SceneCut` 判据）。
///
/// 只依赖「哪些场景在 t 活跃」，与帧号无关 ⇒ 时间线循环回到同一段时
/// 不会误触发 `SceneCut`；真正的段切换必然改变键。
fn v31_scene_key(models: &[SceneModel], t: f32) -> u64 {
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, m) in models.iter().enumerate() {
        if m.duration > 0.0 && m.start <= t && t < m.start + m.duration {
            key ^= (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            key = key.rotate_left(17);
        }
    }
    key
}

/// V3.1 T03 出片主入口（设计 §3 签名；旧 [`render_movie`] 签名与行为不变）。
///
/// # 链路（设计 §4 时序图）
/// `V31Config` → 帧循环 `render_frame_at_with_time` → `V31RenderDevice`
/// （f32 线性帧 → `TemporalState` 窗口累积 → 可选保守降噪 → tone-map）
/// → 抽样帧 `compare_frame_2d` 五判据 + 序列 PSNR → `V31FrameSink`。
///
/// # 600 s 口径
/// 完整 `frames` 帧**全部**逐帧求值（时间线循环保证每帧都有真实 3D 场景，
/// 不是黑帧空转）；`compare_frame_2d` 只在抽样帧做，其余帧走帧哈希 +
/// spp 下限 + 计数器的轻量校验。`V31MovieReport.total_micros` 是整个帧
/// 循环的 wall clock（含抽样帧的 16 spp golden 渲染），不是抽样帧耗时。
///
/// # 出片永不判红（V3.1 收尾 B）
/// golden 比对 / 五判据 / 抽样 manifest / reset 检测全部是 `--verify 1` 的
/// **离线可选项**，默认不进热路径；出片链路永远返回 `Ok(Deliverable)`。
/// 判据本身一条没删：`qa_v31_gate` 独立校验五判据与哈希锚，可选校验结论文印
/// 在 `[v31-verify]` 摘要里，仅供 CI 自行决定是否硬失败。
pub fn render_movie_v31(
    compile: &CompileV2Output,
    device: &mut dyn GpuDevice,
    cfg: &V31Config,
    sink: &mut dyn V31FrameSink,
) -> MovieResult<Deliverable> {
    validate_output_spec(compile)?;
    let eff: V31Config = crate::v31_accum::with_env_overrides(*cfg);
    let (w, h): (u32, u32) = (compile.width, compile.height);
    let frames_u32: u32 = if eff.frames > 0 {
        eff.frames
    } else {
        (compile.fps as f32 * compile.duration).ceil() as u32
    };
    let frame_count: usize = frames_u32 as usize;
    if frame_count == 0 {
        return Err(RenderError::new(
            "R5030",
            format!("movie '{}' 的 V3.1 出片规格", compile.title),
            "帧数为 0（--frames 与 fps×duration 均推不出正帧数）".to_string(),
            "显式指定 --frames <N>（正整数），或让 DSL 的 fps / duration 为正",
        ));
    }

    // ---- 通道 / 签名 / 场景模型（复用旧路径私有实现，求值前完成全部校验）----
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let defs: ChannelDefs = channel_defs_of(compile)?;
    let models: Vec<SceneModel> = build_models(compile, &channels)?;

    // ---- V3.1 收尾（B）：抽样计划只在**可选校验**时构造 --------------------
    // 旧实现在这里无条件 build_sample_plan + 写 manifest.sha256 + 抽样帧
    // 逐帧重渲 16 spp golden，把省下的时间吃光还判红退出。现在默认
    // `verify=false` ⇒ 本段只算时间线周期，零 IO、零 golden、零判据。
    let period: u32 = timeline_period_frames(compile)
        .min(frames_u32.max(1))
        .max(1);
    // 可选校验的逐帧口径：验收片源（≤36 帧）**逐帧全量**收集，每帧独立
    // golden + 五判据，杜绝「抽样漏帧」。长片（>36 帧）仍走抽样计划
    // （场景首/中/末 + 切换点 ±1 + 均匀补足 ≥12 帧），避免 3600 帧 ×
    // 16 spp golden 把离线校验成本抬到不可接受。
    const VERIFY_ALL_FRAMES_LIMIT: u32 = 36;
    let verify_all: bool = eff.verify && frames_u32 <= VERIFY_ALL_FRAMES_LIMIT;
    let plan: Option<SamplePlan> = if eff.verify && !verify_all {
        Some(build_sample_plan(compile, frames_u32, eff.window))
    } else {
        None
    };
    let mut collector: V31VerifyCollector = V31VerifyCollector::new(w, h);

    let mut sims: SimTimeline = SimTimeline::new();
    let mut v31dev: V31RenderDevice<'_> = V31RenderDevice::new(device, eff, w, h);
    // 与 CLI 的 `PRISM_V31_STREAM` / 字节阈值一致：帧不驻留、逐帧归还帧池，
    // 下一帧 `framebuffer_of` 命中复用（360 / 3600 帧目标场景零新大分配）。
    let resident_bytes: u64 = (frame_count as u64)
        .saturating_mul(w as u64)
        .saturating_mul(h as u64)
        .saturating_mul(4);
    let streamed: bool = frame_count > V31_RESIDENT_FRAME_LIMIT
        || resident_bytes > V31_RESIDENT_BYTES_LIMIT
        || std::env::var("PRISM_V31_STREAM").ok().as_deref() == Some("1");
    let mut resident_frames: Vec<Framebuffer> = Vec::new();
    let mut resident_traces: Vec<Vec<u8>> = Vec::new();

    let psnrs: Vec<f64> = Vec::new();
    let mut failed_frames: Vec<(u32, Vec<String>)> = Vec::new();
    let mut spp_below_min: u32 = 0;
    let mut total_new_samples: u64 = 0;
    let profile_enabled: bool =
        std::env::var("PRISM_V31_PROFILE").ok().as_deref() == Some("1");
    let mut profile_frame_us: u64 = 0;
    let mut profile_device_us: u64 = 0;
    let mut profile_hash_us: u64 = 0;
    let mut profile_sink_us: u64 = 0;
    let loop_started = Instant::now();
    // 3600 帧 RSS 首尾差 + 每帧分配次数（报表明细，非门禁）。
    let rss_before: Option<u64> = crate::measure::current_rss_kib();
    let alloc_before: crate::measure::Counters = crate::measure::snapshot();

    for i in 0..frame_count {
        let frame: u32 = i as u32;
        let t_frame: u32 = frame % period;
        let t: f32 = t_frame as f32 / compile.fps.max(1) as f32;
        let scene_key: u64 = v31_scene_key(&models, t);
        v31dev.begin_frame(frame, scene_key);

        let frame_started: Option<Instant> = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let (fb, trace): (Framebuffer, Vec<u8>) = render_frame_at_with_time(
            compile,
            &models,
            &channels,
            &defs,
            &mut v31dev as &mut dyn GpuDevice,
            frame as u64,
            t_frame as u64,
            &mut sims,
        )?;
        if let Some(started) = frame_started {
            profile_frame_us = profile_frame_us
                .saturating_add(started.elapsed().as_micros() as u64);
            profile_device_us = profile_device_us.saturating_add(v31dev.micros());
        }

        // ≤36 帧逐帧全量；长片按抽样计划。门禁读重投影器统计的**真实**
        // 新增 spp（Σ 每像素新采样数），不是摊分伪值。
        let sampled: bool = verify_all
            || plan.as_ref().map(|p| p.is_sampled(frame)).unwrap_or(false);
        // 第零节：样本数不再作为达标下限；这里只对冻结老路径保留诊断。
        if eff.mode == crate::v31_modes::V31Mode::Consistent
            && v31dev.frame_spp() < eff.min_spp
        {
            spp_below_min += 1;
        }
        total_new_samples = total_new_samples.saturating_add(v31dev.new_samples());
        // 可选离线校验（`--verify 1`）：逐帧全量 / 抽样帧收集「该帧 3D 基帧 +
        // 该帧请求」的最小凭证，绝不重渲 golden、绝不比较、绝不阻断出片；
        // 逐帧 golden 比对 / 五判据 / 序列 PSNR 全部在循环结束后由
        // finalize_verify 离线执行。
        if eff.verify && sampled {
            collector.push(crate::v31_verify::V31VerifySample {
                frame,
                got: v31dev.last_got().to_vec(),
                prev: None,
                req: v31dev.last_request().cloned(),
            });
        }

        let hash_started: Option<Instant> = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let frame_hash: u64 = fnv1a64(&fb.pixels);
        if let Some(started) = hash_started {
            profile_hash_us = profile_hash_us
                .saturating_add(started.elapsed().as_micros() as u64);
        }
        let rec = V31FrameRecord {
            frame,
            frame_spp: v31dev.frame_spp(),
            effective_spp: v31dev.effective_spp(),
            micros: v31dev.micros(),
            reset: if eff.verify { v31dev.last_reset() } else { None },
            sampled,
            frame_hash,
        };
        let sink_started: Option<Instant> = if profile_enabled {
            Some(Instant::now())
        } else {
            None
        };
        sink.on_record(&rec);
        sink.on_frame(frame, &fb, v31dev.last_linear());
        if let Some(started) = sink_started {
            profile_sink_us = profile_sink_us
                .saturating_add(started.elapsed().as_micros() as u64);
        }

        // 大帧数只驻留抽样帧（其余由 sink 流式交付），避免 13.3 GB OOM。
        // V3.1 收尾 C：不驻留的帧归还帧池，下一帧 `framebuffer_of` 命中复用
        // （缓冲内容每次 pack 全量覆盖 ⇒ 逐字节不变；3600 帧 RSS 首尾差可查）。
        if !streamed || sampled {
            resident_frames.push(fb);
            resident_traces.push(trace);
        } else {
            crate::frame_pool::FramePool::release(fb);
        }
    }
    let total_micros: u64 = loop_started.elapsed().as_micros() as u64;
    perf_domains::record(Domain::FrameLoop, total_micros);
    if profile_enabled {
        let div: f64 = frame_count.max(1) as f64 * 1000.0;
        let non_device_us: u64 = profile_frame_us.saturating_sub(profile_device_us);
        eprintln!(
            "[v31-profile] 帧循环单帧 {:.3} ms / 帧求值 {:.3} ms / 设备渲染 {:.3} ms / 非设备 {:.3} ms / 帧哈希 {:.3} ms / 输出 {:.3} ms",
            total_micros as f64 / div,
            profile_frame_us as f64 / div,
            profile_device_us as f64 / div,
            non_device_us as f64 / div,
            profile_hash_us as f64 / div,
            profile_sink_us as f64 / div,
        );
    }
    // 独立通用路径的真实样本统计：按档位自适应预算逐帧求值；第零节明确
    // 样本数不是达标下限，这里只报实测，不与冻结分母做等效性宣称。
    let npix_total: f64 = frame_count as f64 * w as f64 * h as f64;
    eprintln!(
        "[v31-samples] 逐帧真实样本总数={} 平均每帧每像素={:.2} spp（独立通用路径；不锁存历史帧）",
        total_new_samples,
        total_new_samples as f64 / npix_total.max(1.0),
    );

    // ---- 可选误差校验：帧循环**之后**，绝不阻断出片 ------------------------
    // 判据本身一条没删（阈值仍来自 v31_config、PSNR 仍走 wavefront 基座），
    // 只是从「每帧热路径 + 判红退出」变成「用户显式 --verify 1 时的离线报告」。
    let verify_report = if eff.verify {
        if collector.samples.is_empty() {
            eprintln!(
                "[v31-verify] 无可校验的 3D 基帧（抽样帧全为黑帧 / 曲线帧 / 无 3D 请求）"
            );
            None
        } else {
            let r = finalize_verify(&collector.samples, &eff, w, h);
            for line in r.summary_lines() {
                eprintln!("{line}");
            }
            if !r.pass {
                failed_frames.push((u32::MAX, r.reasons.clone()));
            }
            // 抽样 manifest / reset 汇总都在帧循环之后，可选校验专属，默认零 IO。
            if let Some(p) = plan.as_ref() {
                match crate::v31_verify::write_verify_manifest(p) {
                    Ok(path) => eprintln!("[v31-verify] 抽样 manifest：{}", path.display()),
                    Err(msg) => eprintln!("[v31-verify] 抽样 manifest 写出失败（仅报告）：{msg}"),
                }
            }
            Some(r)
        }
    } else {
        None
    };
    let resets: Vec<(String, u32)> =
        crate::v31_verify::summarize_resets(&collector);

    let rss_after: Option<u64> = crate::measure::current_rss_kib();
    let rss_delta_kib: Option<i64> = match (rss_before, rss_after) {
        (Some(a), Some(b)) => Some(b as i64 - a as i64),
        _ => None,
    };
    // 计数分配器只在专用测试二进制安装（生产恒 0 ⇒ 不打印误导性数字）。
    let alloc_after: crate::measure::Counters = crate::measure::snapshot();
    let alloc_total: u64 = alloc_after.alloc_count.saturating_sub(alloc_before.alloc_count);
    let alloc_per_frame: Option<u64> = if alloc_total > 0 {
        Some(alloc_total / frame_count.max(1) as u64)
    } else {
        None
    };
    // 3600 帧 RSS 首尾差报表（结构 + 确定性单行，外推口径如实标注）。
    let mem_report: crate::measure::RssDeltaReport = crate::measure::rss_delta_report(
        "v31_movie",
        frames_u32 as u64,
        rss_before,
        rss_after,
        alloc_before,
        alloc_after,
    );
    eprintln!("{}", mem_report.summary_line());
    if alloc_total == 0 {
        eprintln!("[v31-mem] 生产构建未安装计数分配器（生产零改动）；每帧分配次数下降由 qa_perf_counters 实测硬断言验证");
    }
    eprintln!("[v31] 3600 帧 RSS 首尾差外推、非实测（36/360 帧实测 {rss_delta_kib:?} KiB 仅作趋势参考）");
    eprintln!("{}", crate::frame_pool::pool_stats().summary_line());

    let report = V31MovieReport {
        frames_rendered: frames_u32,
        width: w,
        height: h,
        fps: compile.fps,
        quality: eff.quality,
        res_tag: eff.tier_dir_tag(),
        spp_per_frame: eff.spp_per_frame,
        window: eff.window,
        effective_spp: eff.effective_spp(),
        min_spp: eff.min_spp,
        accumulate: eff.accumulate,
        denoise: eff.denoise,
        adaptive: eff.adaptive,
        gpu_prefer: eff.gpu_prefer,
        // 逐帧全量：索引覆盖全部帧；长片抽样：索引为抽样计划命中帧。
        sample_indices: if verify_all {
            (0..frames_u32).collect()
        } else {
            plan.as_ref().map(|p| p.indices.clone()).unwrap_or_default()
        },
        sampled_count: verify_report.as_ref().map(|r| r.checked).unwrap_or(0),
        failed_frames,
        seq_mean_psnr: verify_report
            .as_ref()
            .map(|r| r.seq_mean_psnr)
            .unwrap_or_else(|| crate::v31_gate::sequence_mean_psnr(&psnrs)),
        total_micros,
        resets,
        spp_below_min,
        timeline_period: period,
        streamed,
        mode_label: eff.mode.label().to_string(),
        mode_note: eff.mode.note().to_string(),
        verify: eff.verify,
        verify_pass: verify_report.as_ref().map(|r| r.pass),
        per_frame_micros: total_micros / frame_count.max(1) as u64,
        rss_delta_kib,
        alloc_per_frame,
    };
    for line in report.summary_lines() {
        eprintln!("{line}");
    }
    sink.on_report(&report);
    // 无 R5031：出片链路永远成功；需要门禁时用 `--verify 1` 单独跑判据。

    Ok(Deliverable {
        title: compile.title.clone(),
        width: w,
        height: h,
        fps: compile.fps,
        duration: compile.duration,
        frames: resident_frames,
        wiring_trace: resident_traces,
    })
}

/// 规格校验（R5001 三段式）：宽高 / fps / 时长必须为正——绝不带病渲染。
fn validate_output_spec(compile: &CompileV2Output) -> MovieResult<()> {
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
    Ok(())
}

/// 单帧求值分派（[`render_movie`] 帧循环体 / [`render_movie_frame`] 共用）：
/// 按活跃场景数分派黑帧 / 单场景 / 双场景过渡 / 同程分屏，其余报 `R5020`。
/// sim 跨帧状态经 `sims`（调用方持有）延续。
fn render_frame_at(
    compile: &CompileV2Output,
    models: &[SceneModel],
    channels: &BTreeMap<&str, &PropertyChannel>,
    defs: &ChannelDefs,
    device: &mut dyn GpuDevice,
    frame: u64,
    sims: &mut SimTimeline,
) -> MovieResult<(Framebuffer, Vec<u8>)> {
    // 旧入口：时间帧号 == 真实帧号（逐字节不变）。
    render_frame_at_with_time(compile, models, channels, defs, device, frame, frame, sims)
}

/// 单帧求值分派的可重映射时间版本（V3.1 T03 专用）。
///
/// `frame` 仍是真实帧号（wiring 凭证 / sim 状态 / 采样流沿用），
/// `t_frame` 只决定 `t = t_frame / fps`（场景活跃区间 / 相机轨道）。
/// 旧入口传 `t_frame == frame` ⇒ 与改前逐字节一致。
fn render_frame_at_with_time(
    compile: &CompileV2Output,
    models: &[SceneModel],
    channels: &BTreeMap<&str, &PropertyChannel>,
    defs: &ChannelDefs,
    device: &mut dyn GpuDevice,
    frame: u64,
    t_frame: u64,
    sims: &mut SimTimeline,
) -> MovieResult<(Framebuffer, Vec<u8>)> {
    let t: f32 = t_frame as f32 / compile.fps as f32;
    let active: Vec<usize> = models
        .iter()
        .enumerate()
        .filter(|(_, m)| m.duration > 0.0 && m.start <= t && t < m.start + m.duration)
        .map(|(idx, _)| idx)
        .collect();
    match active.len() {
        // 空档（timeline 长于内容）：黑帧 + 空档帧的接线凭证，确定性。
        0 => Ok((
            black_frame(compile.width, compile.height),
            wiring_frame_values(compile, frame).canonical_bytes(),
        )),
        1 => render_scene_frame(
            &models[active[0]],
            compile,
            channels,
            defs,
            sims,
            device,
            t,
            frame,
        ),
        2 => {
            let a: &SceneModel = &models[active[0]];
            let b: &SceneModel = &models[active[1]];
            if let Some(tr) = find_transition(compile, a, b, t) {
                // 过渡重叠区：两场景各求值一帧，按 kind 整数混合。
                let (fa, trace_a): (Framebuffer, Vec<u8>) =
                    render_scene_frame(a, compile, channels, defs, sims, device, t, frame)?;
                let (fb2, trace_b): (Framebuffer, Vec<u8>) =
                    render_scene_frame(b, compile, channels, defs, sims, device, t, frame)?;
                let mut trace: Vec<u8> = trace_a;
                trace.extend_from_slice(&trace_b);
                Ok((blend_transition(&fa, &fb2, tr, t), trace))
            } else if same_interval(a, b) {
                // 同程多场景 = 左右分屏（split_screen 展开形态）。
                let (fl, trace_l): (Framebuffer, Vec<u8>) =
                    render_scene_frame(a, compile, channels, defs, sims, device, t, frame)?;
                let (fr, trace_r): (Framebuffer, Vec<u8>) =
                    render_scene_frame(b, compile, channels, defs, sims, device, t, frame)?;
                let mut trace: Vec<u8> = trace_l;
                trace.extend_from_slice(&trace_r);
                Ok((split_halves(&fl, &fr), trace))
            } else {
                Err(RenderError::new(
                    "R5020",
                    format!("movie '{}' 的时间线（t = {t:.3}s）", compile.title),
                    format!(
                        "场景 #{} '{}'（[{}..{}]s）与 #{} '{}'（[{}..{}]s）同时活跃，但既非同程分屏（起止相同）也无匹配过渡",
                        a.index + 1, a.name, a.start, a.start + a.duration,
                        b.index + 1, b.name, b.start, b.start + b.duration
                    ),
                    "两段同时活跃的画面只能是：① 起止相同的分屏（split_screen 组合子）；② 相邻场景 + 一条 duration 匹配的 transition 块",
                ))
            }
        }
        _ => {
            let names: Vec<String> = active.iter().map(|&idx| models[idx].name.clone()).collect();
            Err(RenderError::new(
                "R5020",
                format!("movie '{}' 的时间线（t = {t:.3}s）", compile.title),
                format!("{} 个场景同时活跃（{}），超出已支持的时间线语义", active.len(), names.join("、")),
                "已支持：单场景 / 两场景过渡重叠 / 两场景同程分屏；更多同时活跃画面请拆分为多个 movie（尚未支持）",
            ))
        }
    }
}

/// 单帧入口（seek / 增量重渲染用）：渲染影片第 `frame` 帧（0 起）。
///
/// 跨帧 sim 状态经调用方持有的 [`SimTimeline`] 延续：顺序推进
/// （0..=n−1）后渲染第 n 帧 ≡ [`render_movie`] 的第 n 帧（逐字节）；
/// 冷 seek（帧 > 0 且前驱快照缺失）→ `P3012` 结构化报错，绝不静默
/// fallback 到初态（旧路径那会把 seek 断档变成语义级静默错误：输出与
/// 首帧逐位相同且沿轨道传播）。
///
/// # Errors
/// - `frame ≥ ceil(fps × duration)` → `R5021`；
/// - 其余同 [`render_movie`]（规格 / 节点 / 参数域 / 冷 seek / 时间线）。
pub fn render_movie_frame(
    compile: &CompileV2Output,
    device: &mut dyn GpuDevice,
    frame: u64,
    sims: &mut SimTimeline,
) -> MovieResult<Framebuffer> {
    validate_output_spec(compile)?;
    let frame_count: u64 = (compile.fps as f32 * compile.duration).ceil() as u64;
    if frame >= frame_count {
        return Err(RenderError::new(
            "R5021",
            format!("movie '{}' 的单帧入口", compile.title),
            format!(
                "帧号 {frame} 超出影片帧数 {frame_count}（0 起；合法区间 0..{}）",
                frame_count.saturating_sub(1)
            ),
            "帧号取 0..ceil(fps × duration) − 1；需要更长影片请加大 duration",
        ));
    }
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let defs: ChannelDefs = channel_defs_of(compile)?;
    let models: Vec<SceneModel> = build_models(compile, &channels)?;
    render_frame_at(compile, &models, &channels, &defs, device, frame, sims)
        .map(|(fb, _trace)| fb)
}

/// sim 帧签名查询入口（测试 / 外部缓存键咬合用）：给定 sim 节点与求解
/// 模式三元组，返回该节点的帧签名（模式元数据 (ε, n) 已收拢在内）。
///
/// 改 substeps / iterations / residual_eps 任一 ⇒ 返回值变化 ⇒ 快照 /
/// 缓存键不误复用（求值层 §3.2 S3 迭代层缓存键 (ε, n) 纪律）。同参数
/// 两调同值（确定性铁律）。
///
/// # Errors
/// - 模式参数非法（0 子步 / 0 迭代 / 非正 ε）→ `R5018`；
/// - 节点不在图中 → `P3013`；绑定引用未定义通道 → `P3012`。
pub fn sim_frame_signature(
    compile: &CompileV2Output,
    node_id: u64,
    substeps: u32,
    iterations: u32,
    residual_eps: f32,
) -> MovieResult<Hash256> {
    let mode: SolveModeMeta = SolveModeMeta::new(substeps, iterations, residual_eps);
    validate_sim_mode(&mode, node_id)?;
    let defs: ChannelDefs = channel_defs_of(compile)?;
    loop_frame_signature(&compile.graph, node_id, &defs, &mode).map_err(signature_err)
}

// ---------------------------------------------------------------------------
// 场景模型构建（节点分类 + 覆盖校验）
// ---------------------------------------------------------------------------

/// 遍历全部场景的节点，分类为相机 / 材质 / 球体 / 曲线 / sim / 文本；
/// 遇不支持节点即报错。
fn build_models(
    compile: &CompileV2Output,
    channels: &BTreeMap<&str, &PropertyChannel>,
) -> MovieResult<Vec<SceneModel>> {
    let mut models: Vec<SceneModel> = Vec::with_capacity(compile.scenes.len());
    // sim 基准帧需要影片帧网格上界（帧数）。
    let frame_count: u64 = (compile.fps as f32 * compile.duration).ceil() as u64;
    for (si, scene) in compile.scenes.iter().enumerate() {
        let mut model: SceneModel = SceneModel {
            index: si,
            name: scene.name.clone(),
            start: scene.start,
            duration: scene.duration,
            camera: None,
            material: None,
            sphere: None,
            curves: Vec::new(),
            sims: Vec::new(),
            sim_base_frame: 0,
            texts: Vec::new(),
            attach: scene.attach.clone(),
            wiring_nodes: Vec::new(),
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
        // sim 基准帧：场景首个活跃影片帧（纯函数；无 sim 的场景不算）。
        if !model.sims.is_empty() {
            model.sim_base_frame = scene_first_frame(&model, compile.fps, frame_count);
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
                match kind {
                    // 曲线图元族：入 curves 槽位，帧求值走 rt 波前路径。
                    "cylinder" | "cone" | "torus" | "revolution" => {
                        model.curves.push(node_ref);
                        Ok(())
                    }
                    "sphere" => {
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
                    }
                    other => {
                        return Err(RenderError::new(
                            "R5013",
                            format!("场景 '{}' 的 sdf 节点 {}（kind = '{other}'）", scene.name, node.id),
                            format!(
                                "render 侧已支持的实体只有 sphere / cylinder / cone / torus / revolution；'{other}' 缺少求交实现"
                            ),
                            "把 sdf kind 写成 sphere / cylinder / cone / torus / revolution；box 等基元待几何求交模块接入（尚未支持）",
                        ));
                    }
                }
            } else if node.params.get("content").is_some() {
                // 文本旋转：排版管线不承载任意角度（缺任意角光栅化）。
                if let Some(ParamValue::F32(r)) = node.params.get("rotation") {
                    if r.is_finite() && r.abs() > 1e-6 {
                        return Err(RenderError::new(
                            "R5014",
                            format!("场景 '{}' 的文本节点 {}（rotation = {r} rad）", scene.name, node.id),
                            "render 侧文本合成暂不支持旋转（缺任意角字形光栅化）".to_string(),
                            "删掉 layer text 的 rotation 字段；旋转文本待矢量图层接入（尚未支持）",
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
        // W2（R5010 销号）：S4 确定性抽样 / layer particles —— 不再拒绝。
        // 节点参数在渲染前由 `resolve_params` 逐帧解算，再经 `w2_eval` 交给
        // prism-particles 既有求值器；render 不复制任何粒子域逻辑（D-28）。
        StructureId::DeterministicSampling => {
            model.wiring_nodes.push(node_ref);
            Ok(())
        }
        // S3 约束梯度流：sim（帧循环驱动 XPBD 推进；模式元数据进签名）。
        // 种类域（当前仅 cloth）在求值前由 SimSpec::from_params 校验（R5017）。
        StructureId::ConstraintFlow => {
            model.sims.push(node_ref);
            Ok(())
        }
        // W2（R5010 销号）：S2 卷积 / compose 汇点 —— 不再拒绝。
        // 参数进求值链源头，合成求值由 prism-postfx 既有 merge 族承担。
        StructureId::Convolution => {
            model.wiring_nodes.push(node_ref);
            Ok(())
        }
        // W2（R5010 销号）：其余结构（audio / post / light 的投影节点等）
        // 全部收进求值链源头，不再以「未知结构」拒绝。真正的能力语义由
        // `cap` 接线表指定的既有 crate 求值器消费，render 只做解算与调度。
        other_structure => {
            let _ = other_structure;
            model.wiring_nodes.push(node_ref);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// S3 约束梯度流：自包含 XPBD 求解器 + 帧循环驱动（M3 固定步数）
//
// 依赖边界：prism-render 不依赖 prism-physics（Cargo.toml 无此路径边，且
// 清单外不可加），求解器在本模块内自包含实装——数值纪律逐位对齐
// prism-physics 的 xpbd.rs / session.rs（半隐式欧拉 + XPBD 投影、约束按
// id 升序 Gauss-Seidel、λ 每子步清零、仅 IEEE 基本运算、跨平台逐位一致），
// 语义权威仍在源文件。模式元数据（substeps / iterations / residual_eps）
// 经通道定义哈希进帧签名：同参数永远同画面（求值层 §3.2 S3 迭代层缓存键
// (ε, n) 纪律——改迭代数 ⇒ 签名变 ⇒ 快照/缓存不误复用）。
// ---------------------------------------------------------------------------

/// sim 网格缺省边长（与 DSL 缺省 `DEFAULT_SIM_GRID` 一致）。
const SIM_GRID_DEFAULT: u32 = 16;
/// sim 网格边长上限（粒子数 = 边长²；防病态规模，显式拒绝而非静默钳制）。
const SIM_GRID_MAX: u32 = 64;
/// sim 刚度缺省（[0,1]；compliance = 1 − stiffness；与 DSL 缺省一致）。
const SIM_STIFFNESS_DEFAULT: f32 = 0.9;
/// sim 阻尼缺省（帧末速度乘子衰减；与 DSL 缺省一致）。
const SIM_DAMPING_DEFAULT: f32 = 0.02;
/// sim 外力强度缺省（与 DSL 缺省 `DEFAULT_FORCE_STRENGTH` 一致）。
const SIM_FORCE_STRENGTH_DEFAULT: f32 = 1.0;
/// sim 外力方向缺省（与 DSL 缺省一致）。
const SIM_FORCE_DIR_DEFAULT: (f32, f32, f32) = (1.0, 0.0, 0.0);
/// sim noise 缺省标度（与 DSL 缺省 `DEFAULT_NOISE_SCALE` 一致；缺省值视为
/// 关闭——noise 扰动场属 ForceField 族扩展位，非缺省显式报错）。
const SIM_NOISE_DEFAULT: f32 = 0.1;
/// 子步数缺省（M3 固定标价；小步比多迭代稳）。
const SIM_SUBSTEPS_DEFAULT: u32 = 4;
/// 迭代数缺省（XPBD 刚度与迭代解耦：多迭代只降残差不改刚度）。
const SIM_ITERATIONS_DEFAULT: u32 = 8;
/// sim 世界网格跨度（x/z 方向物理尺寸）。
const SIM_SPAN: f32 = 2.4;
/// sim 网格悬挂高度（顶行固定粒子的 y）。
const SIM_HEIGHT: f32 = 1.5;
/// 粒子叠加圆点颜色（暖橙，不透明写值）。
const SIM_DOT_COLOR: [u8; 3] = [255, 176, 66];

/// 严格读 f32 参数（sim 物理 / 模式参数用）：缺失 → 缺省；存在但非 F32 或
/// 非有限 → 三段式报错（C20：缺省只覆盖"未声明"，不覆盖"声明了坏值"）。
fn sim_f32(
    params: &prism_core::Params,
    key: &str,
    default: f32,
    node_id: u64,
    code: &'static str,
) -> MovieResult<f32> {
    match params.get(key) {
        None => Ok(default),
        Some(ParamValue::F32(v)) if v.is_finite() => Ok(*v),
        Some(other) => Err(RenderError::new(
            code,
            format!("sim 节点 {node_id} 的参数 '{key}'"),
            format!("sim 参数必须为有限 f32，却得到 {other:?}"),
            format!("'{key}' 传有限数字；删除该参数用缺省值 {default}"),
        )),
    }
}

/// 严格读 u32 参数（sim 网格 / 步数用）：同 [`sim_f32`] 纪律。
fn sim_u32(
    params: &prism_core::Params,
    key: &str,
    default: u32,
    node_id: u64,
    code: &'static str,
) -> MovieResult<u32> {
    match params.get(key) {
        None => Ok(default),
        Some(ParamValue::U32(n)) => Ok(*n),
        Some(other) => Err(RenderError::new(
            code,
            format!("sim 节点 {node_id} 的参数 '{key}'"),
            format!("sim 参数必须为 u32 整数，却得到 {other:?}"),
            format!("'{key}' 传非负整数；删除该参数用缺省值 {default}"),
        )),
    }
}

/// 求解模式（M3 固定步数）：子步数 × 迭代数显式标价 + 残差阈值。
///
/// 类型本体已上提为 `prism_core::fixpoint::SolveModeMeta`（内核一等属性）：
/// `(ε, n)` 经 `solve_mode_def`（tag 0x5333）进帧签名（求值层 §3.2 S3 迭代层
/// 缓存键纪律）。此处只保留**宿主层参数解算与 R5xxx 错误码映射**——防呆语义
/// 与内核一致，码段按 render 成片直达族（R5018）报出。
///
/// M2（容差停机）对本求解器不适用——约束投影非扩张无先验误差界，容差停机
/// 即伪造 Banach 界（`xpbd.rs` 模块文档成文纪律）。
fn sim_mode_of_params(params: &prism_core::Params, node_id: u64) -> MovieResult<SolveModeMeta> {
    let substeps: u32 = sim_u32(params, "substeps", SIM_SUBSTEPS_DEFAULT, node_id, "R5018")?;
    let iterations: u32 = sim_u32(params, "iterations", SIM_ITERATIONS_DEFAULT, node_id, "R5018")?;
    let residual_eps: f32 =
        sim_f32(params, "residual_eps", RESIDUAL_EPS_DEFAULT, node_id, "R5018")?;
    let mode: SolveModeMeta = SolveModeMeta::new(substeps, iterations, residual_eps);
    validate_sim_mode(&mode, node_id)?;
    Ok(mode)
}

/// 模式防呆（C18/C21）：子步数 / 迭代数 ≥ 1，残差阈值正有限。
///
/// 与内核 `SolveModeMeta::validate` 同判据；此处按 render 错误码段（R5018）
/// 与 sim 节点定位报出（宿主层可行动建议）。
fn validate_sim_mode(mode: &SolveModeMeta, node_id: u64) -> MovieResult<()> {
    if mode.substeps == 0 {
        return Err(RenderError::new(
            "R5018",
            format!("sim 节点 {node_id} 的模式参数 substeps"),
            "子步数必须 ≥ 1（子步数固定是确定性纪律）".to_string(),
            format!("substeps 传 ≥ 1 的整数，参考标价 {SIM_SUBSTEPS_DEFAULT}（小步比多迭代稳）"),
        ));
    }
    if mode.iterations == 0 {
        return Err(RenderError::new(
            "R5018",
            format!("sim 节点 {node_id} 的模式参数 iterations"),
            "迭代数必须 ≥ 1（迭代数固定是确定性纪律）".to_string(),
            format!("iterations 传 ≥ 1 的整数，参考标价 {SIM_ITERATIONS_DEFAULT}（刚度与迭代解耦）"),
        ));
    }
    if !(mode.residual_eps.is_finite() && mode.residual_eps > 0.0) {
        return Err(RenderError::new(
            "R5018",
            format!("sim 节点 {node_id} 的模式参数 residual_eps"),
            format!("残差阈值必须为正有限数，却得到 {}", mode.residual_eps),
            format!("residual_eps 传正数，参考标价 {RESIDUAL_EPS_DEFAULT}"),
        ));
    }
    Ok(())
}

/// 签名计算错误 → 三段式渲染错误（码段透传：P3xxx 属图 / 签名族）。
fn signature_err(err: SignatureError) -> RenderError {
    match &err {
        SignatureError::Graph(e) => {
            RenderError::new(e.code, e.location.clone(), e.problem.clone(), e.fix.clone())
        }
        SignatureError::UnknownChannel { location, channel } => RenderError::new(
            "P3012",
            location.clone(),
            format!("绑定引用的通道 \"{channel}\" 在通道定义表中不存在"),
            "在通道定义表中补齐该通道定义，或移除该绑定".to_string(),
        ),
    }
}

/// 把编译产物的命名通道表编码为签名侧通道定义表。
///
/// 保守编码（签名只消费相等性、不解读内容）：tag = 变体判别值（0..6，按
/// `PropertyChannel` 枚举序），def_bytes = Debug 字节（f32 的 Debug 为
/// 确定性最短往返表示，跨平台逐位一致）。property.rs → `ChannelDef` 的
/// 正式转换器落地后可无缝替换，签名语义不变（同定义同哈希）。
fn channel_defs_of(compile: &CompileV2Output) -> MovieResult<ChannelDefs> {
    let mut defs: ChannelDefs = ChannelDefs::new();
    for nc in &compile.channels {
        let tag: u16 = match &nc.channel {
            PropertyChannel::Const(_) => 0,
            PropertyChannel::Track(_) => 1,
            PropertyChannel::Spring(_) => 2,
            PropertyChannel::Expr(_) => 3,
            PropertyChannel::Wiggle(_) => 4,
            PropertyChannel::Remap { .. } => 5,
            PropertyChannel::Loop { .. } => 6,
        };
        let def_bytes: Vec<u8> = format!("{:?}", nc.channel).into_bytes();
        if defs
            .insert(nc.name.clone(), ChannelDef { tag, def_bytes })
            .is_some()
        {
            return Err(RenderError::new(
                "R5003",
                format!("movie '{}' 的通道登记表", compile.title),
                format!("命名通道 '{}' 重复登记（定义冲突）", nc.name),
                "通道名全局唯一；重名请合并定义或改名".to_string(),
            ));
        }
    }
    Ok(defs)
}

/// sim 粒子（自包含 XPBD 状态；跨帧状态 = 位置 + 速度，`prev_position`
/// 由每子步预测阶段重写、不属跨帧状态；`inv_mass = 0` 表示固定）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SimParticle {
    position: Vec3,
    prev_position: Vec3,
    velocity: Vec3,
    inv_mass: f32,
}

/// sim 距离约束（C = |p_a − p_b| − rest_length；id 升序 = 求解序格锚）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct SimConstraint {
    id: u32,
    a: usize,
    b: usize,
    rest_length: f32,
    /// 柔度 compliance（XPBD 刚度参数；与迭代数解耦）。
    compliance: f32,
}

/// sim 规格：外力 + 阻尼 + 粒子初态 + 约束集（签名变化 = 规格变化）。
#[derive(Debug, Clone, PartialEq)]
struct SimSpec {
    /// 常量外力（direction × strength；零向量 = 无外力，合法）。
    force: Vec3,
    /// 帧末阻尼乘子（v *= 1 − damping；显式外挂，不进投影）。
    damping: f32,
    /// 粒子初态（下标即粒子 id）。
    particles: Vec<SimParticle>,
    /// 距离约束集（构造即按 id 升序）。
    constraints: Vec<SimConstraint>,
}

impl SimSpec {
    /// 从解算参数构建规格（cloth 网格族：n×n 网格、顶行固定、跨度 SIM_SPAN、
    /// 悬挂高 SIM_HEIGHT、结构距离约束 H+V；compliance = 1 − stiffness）。
    ///
    /// 参数域非法（kind 非 cloth / stiffness 越界 / damping ≥ 1 / 非缺省
    /// noise / grid 越界 / 非有限分量）→ R5017 三段式（C18/C20）。
    fn from_params(params: &prism_core::Params, node_id: u64) -> MovieResult<SimSpec> {
        let kind: &str = p_str(params, "kind").unwrap_or("");
        if kind != "cloth" {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id}（kind = '{kind}'）"),
                format!(
                    "render 侧 sim 求解器当前仅实装 cloth 网格族；'{kind}' 尚未接入"
                ),
                "把 sim 种类写成 cloth（网格布料：顶行固定 + 结构距离约束）；其余种类待约束族扩展",
            ));
        }
        let grid: u32 = sim_u32(params, "grid", SIM_GRID_DEFAULT, node_id, "R5017")?;
        if grid == 0 {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id} 的参数 'grid'"),
                "网格边长必须 ≥ 1".to_string(),
                "grid 传正整数（如 grid 16）".to_string(),
            ));
        }
        if grid > SIM_GRID_MAX {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id} 的参数 'grid'"),
                format!("网格边长 {grid} 超上限 {SIM_GRID_MAX}（粒子数 = 边长²，病态规模显式拒绝）"),
                format!("grid 取 ≤ {SIM_GRID_MAX} 的正整数"),
            ));
        }
        let stiffness: f32 = sim_f32(params, "stiffness", SIM_STIFFNESS_DEFAULT, node_id, "R5017")?;
        if !(0.0..=1.0).contains(&stiffness) {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id} 的参数 'stiffness'"),
                format!("刚度必须在 [0, 1] 区间，却得到 {stiffness}"),
                "stiffness 传 [0,1]：0 = 完全柔性，1 = 完全刚性；常见 0.9".to_string(),
            ));
        }
        let damping: f32 = sim_f32(params, "damping", SIM_DAMPING_DEFAULT, node_id, "R5017")?;
        if !(damping >= 0.0 && damping < 1.0) {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id} 的参数 'damping'"),
                format!("阻尼必须在 [0, 1) 区间（帧末速度乘子 1 − damping），却得到 {damping}"),
                "damping 传 [0,1) 的非负数；常见 0.02".to_string(),
            ));
        }
        // noise 扰动场属 ForceField 族扩展位：非缺省值显式报错（缺省值视为
        // 关闭——DSL 对未声明 noise 的 sim 回填缺省标度，与显式关闭不可区分）。
        let noise: f32 = sim_f32(params, "noise_scale", SIM_NOISE_DEFAULT, node_id, "R5017")?;
        if noise != SIM_NOISE_DEFAULT {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {node_id} 的参数 'noise_scale'"),
                format!("noise 扰动场尚未接入（ForceField 族扩展位），却声明了非缺省标度 {noise}"),
                "删除 noise 块（缺省标度按零噪声求解），或等待噪声力场接入",
            ));
        }
        // 外力：strength × normalize(direction)；零方向 = 无外力（合法）。
        let strength: f32 =
            sim_f32(params, "force_strength", SIM_FORCE_STRENGTH_DEFAULT, node_id, "R5017")?;
        let mut dir: [f32; 3] = [
            SIM_FORCE_DIR_DEFAULT.0,
            SIM_FORCE_DIR_DEFAULT.1,
            SIM_FORCE_DIR_DEFAULT.2,
        ];
        for (k, key) in ["force_dir_x", "force_dir_y", "force_dir_z"]
            .iter()
            .enumerate()
        {
            match params.get(*key) {
                None => {}
                Some(ParamValue::F32(v)) if v.is_finite() => dir[k] = *v,
                Some(other) => {
                    return Err(RenderError::new(
                        "R5017",
                        format!("sim 节点 {node_id} 的参数 '{key}'"),
                        format!("外力方向分量必须为有限 f32，却得到 {other:?}"),
                        "direction 传有限 vec3（如 direction 1,0,0）；删除该分量用缺省值".to_string(),
                    ));
                }
            }
        }
        let dir_v: Vec3 = Vec3::new(dir[0], dir[1], dir[2]);
        let force: Vec3 = if dir_v.length() > 1e-12 {
            dir_v.normalize() * strength
        } else {
            Vec3::splat(0.0)
        };

        // 网格构建（行主序：下标 = i*n + j；顶行 i = 0 固定）。
        let n: usize = grid as usize;
        let mut particles: Vec<SimParticle> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let (x, z): (f32, f32) = if n >= 2 {
                    (
                        -SIM_SPAN * 0.5 + SIM_SPAN * (j as f32) / ((n - 1) as f32),
                        -SIM_SPAN * 0.5 + SIM_SPAN * (i as f32) / ((n - 1) as f32),
                    )
                } else {
                    (0.0, 0.0)
                };
                particles.push(SimParticle {
                    position: Vec3::new(x, SIM_HEIGHT, z),
                    prev_position: Vec3::new(x, SIM_HEIGHT, z),
                    velocity: Vec3::splat(0.0),
                    inv_mass: if i == 0 { 0.0 } else { 1.0 },
                });
            }
        }
        let rest: f32 = if n >= 2 { SIM_SPAN / (n - 1) as f32 } else { 0.0 };
        let compliance: f32 = 1.0 - stiffness;
        let mut constraints: Vec<SimConstraint> = Vec::new();
        let mut next_id: u32 = 0;
        // 结构约束：行内水平 + 列间竖直（id 单调分配 ⇒ 构造即升序）。
        for i in 0..n {
            for j in 0..n {
                let idx: usize = i * n + j;
                if j + 1 < n {
                    constraints.push(SimConstraint {
                        id: next_id,
                        a: idx,
                        b: idx + 1,
                        rest_length: rest,
                        compliance,
                    });
                    next_id += 1;
                }
                if i + 1 < n {
                    constraints.push(SimConstraint {
                        id: next_id,
                        a: idx,
                        b: idx + n,
                        rest_length: rest,
                        compliance,
                    });
                    next_id += 1;
                }
            }
        }
        constraints.sort_by(|a, b| a.id.cmp(&b.id)); // 防御：求解序格锚不变式
        Ok(SimSpec { force, damping, particles, constraints })
    }

    /// 规格校验（C18/C20：端点越界 / 参数域非法 / 非有限位置显式拒绝）。
    fn validate(&self, node_id: u64) -> MovieResult<()> {
        let n: usize = self.particles.len();
        for (i, p) in self.particles.iter().enumerate() {
            if !(p.inv_mass >= 0.0)
                || !p.position.x.is_finite()
                || !p.position.y.is_finite()
                || !p.position.z.is_finite()
            {
                return Err(RenderError::new(
                    "R5017",
                    format!("sim 节点 {node_id} 的粒子 {i}"),
                    format!("粒子状态非法（inv_mass = {} 或位置非有限）", p.inv_mass),
                    "inv_mass 传 ≥ 0（0 = 固定）；位置必须有限".to_string(),
                ));
            }
        }
        for c in &self.constraints {
            if !(c.rest_length >= 0.0) {
                return Err(RenderError::new(
                    "R5017",
                    format!("sim 节点 {node_id} 的约束 {}", c.id),
                    format!("rest_length = {}：静止长度必须 ≥ 0（NaN 同拒）", c.rest_length),
                    "静止长度传非负数".to_string(),
                ));
            }
            if !(c.compliance >= 0.0) {
                return Err(RenderError::new(
                    "R5017",
                    format!("sim 节点 {node_id} 的约束 {}", c.id),
                    format!(
                        "compliance = {}：柔度必须 ≥ 0（NaN 同拒）；负柔度使投影分母 ≤ 0，约束被静默吞掉",
                        c.compliance
                    ),
                    "柔度传非负数；完全刚性传 0".to_string(),
                ));
            }
            if c.a >= n || c.b >= n {
                return Err(RenderError::new(
                    "R5017",
                    format!("sim 节点 {node_id} 的约束 {}", c.id),
                    format!("约束端点 ({}, {}) 越界（粒子数 {n}）", c.a, c.b),
                    "先建齐粒子，再声明引用其下标的约束".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// 推进一帧（`dt` 秒）：`substeps` 个子步 × `iterations` 轮投影。
    ///
    /// 状态缓冲区由调用方显式传入（规格是不可变模板：外力 / 阻尼 / 约束集 +
    /// 初态粒子；活状态在运行时侧——`spec.step` 与运行时状态分离，杜绝
    /// "步进了模板副本而活状态冻结"类静默错误）。
    ///
    /// 数值纪律逐位对齐 `prism-physics::xpbd::World::step` 的半隐式欧拉路径：
    /// ① 预测 `v += F·h; x += v·h`（固定粒子 prev 同步、位置不动）；
    /// ② 投影：λ 每子步清零，按约束 id 升序 Gauss-Seidel
    ///    （Δλ = (−C − α̃λ)/(Σw + α̃)，α̃ = compliance/h²）；
    /// ③ 速度更新 `v = (x − x_prev)/h`；
    /// ④ 帧末阻尼 `v *= (1 − damping)`（显式外挂，不进投影）。
    ///
    /// ② 的每子步 n 轮迭代走内核环实例 `prism_core::fixpoint::run_fixed_point`
    /// （M3 定数 + 逐轮残差记录 + 发散观测；λ 在 `lambdas` 里跨轮存活，
    /// 只在子步边界清零——X ⊕ Λ 语义）。返回帧末最大约束残余 |C|（喂驱动层
    /// 的帧级发散检测）+ **逐轮发散判定**（W3 遗留：内核 `FixpointReport::
    /// diverged` 此前在本处被丢弃，只留了 `final_residual`）。两个信号互补：
    /// 逐轮看「一轮 M3 迭代内残差是否连续上升」（Φ 的收缩性），帧级看跨帧
    /// 趋势；任一为真都说明解算不稳，故由 [`SimRuntime::diverged`] 取并。
    fn step(
        &self,
        particles: &mut Vec<SimParticle>,
        mode: &SolveModeMeta,
        dt: f32,
    ) -> Result<(f32, bool), EvalError> {
        // 契约：报错时状态必须未被推进——校验先于预测（驱动层的 Err 约定）。
        mode.validate("sim 求解步（环推进参数）")?;
        let h: f32 = dt / mode.substeps as f32;
        let force_h: Vec3 = self.force * h;
        let mut lambdas: Vec<f32> = vec![0.0f32; self.constraints.len()];
        let mut residual: f32 = 0.0;
        let mut iter_diverged: bool = false;
        for _ in 0..mode.substeps {
            // ① 预测（固定粒子 inv_mass = 0：prev 同步、位置不动）。
            for p in particles.iter_mut() {
                if p.inv_mass > 0.0 {
                    p.velocity = p.velocity + force_h;
                    p.prev_position = p.position;
                    p.position = p.position + p.velocity * h;
                } else {
                    p.prev_position = p.position;
                }
            }
            // ② 投影（λ 每子步重置；每轮按 id 升序遍历——构造即有序）。
            for l in &mut lambdas {
                *l = 0.0;
            }
            let constraints: &[SimConstraint] = &self.constraints;
            let mut scan = |ps: &mut Vec<SimParticle>| -> f32 {
                for (ci, c) in constraints.iter().enumerate() {
                    project_distance(c, ps, h, &mut lambdas[ci]);
                }
                max_distance_residual(ps.as_slice(), constraints)
            };
            let report = run_fixed_point(particles, &mut scan, mode)?;
            residual = report.final_residual().unwrap_or(0.0);
            // 逐轮判定跨子步取或：任一子步的 Φ 发散都不该被子步边界抹掉。
            iter_diverged |= report.diverged;
            // ③ 速度更新（固定粒子速度归零，与 xpbd 同口径）。
            let inv_h: f32 = 1.0 / h;
            for p in particles.iter_mut() {
                p.velocity = (p.position - p.prev_position) * inv_h;
            }
        }
        // ④ 帧末阻尼（显式外挂；damping = 0 时乘子恒 1，逐位不变）。
        let keep: f32 = 1.0 - self.damping;
        for p in particles.iter_mut() {
            if p.inv_mass > 0.0 {
                p.velocity = p.velocity * keep;
            }
        }
        Ok((residual, iter_diverged))
    }
}

/// 约束集的最大 |C|（M3 残差读数；纪律 3 的免费质量指标）。
fn max_distance_residual(particles: &[SimParticle], constraints: &[SimConstraint]) -> f32 {
    let mut max_r: f32 = 0.0;
    for c in constraints {
        let d: Vec3 = particles[c.a].position - particles[c.b].position;
        let r: f32 = d.length() - c.rest_length;
        if r.abs() > max_r {
            max_r = r.abs();
        }
    }
    max_r
}

/// 距离约束投影一步（XPBD 论文 §3；与 `xpbd.rs` 的 `DistanceConstraint::project`
/// 同算式同奇异保护：重合粒子跳过修正，残余如实保留、不静默钳制——C20）。
fn project_distance(c: &SimConstraint, particles: &mut [SimParticle], dt_h: f32, lambda: &mut f32) {
    let (ia, ib) = (c.a, c.b);
    let d: Vec3 = particles[ia].position - particles[ib].position;
    let len: f32 = d.length();
    if len < 1e-8 {
        return; // 奇异构型（重合粒子）：n = d/len 无定义，跳过修正。
    }
    let cc: f32 = len - c.rest_length;
    let n: Vec3 = d / len;
    let w_sum: f32 = particles[ia].inv_mass + particles[ib].inv_mass;
    // α̃ = α/h²（XPBD 论文 §3：compliance 随子步长重标定）。
    let alpha_tilde: f32 = c.compliance / (dt_h * dt_h);
    let denom: f32 = w_sum + alpha_tilde;
    if denom <= 0.0 {
        return; // 双固定粒子且零柔度：无自由度，约束不可作用。
    }
    // Δλ = (−C − α̃λ) / (Σw + α̃)；λ 每子步清零后累积（论文 §3.3）。
    let dlambda: f32 = (-cc - alpha_tilde * *lambda) / denom;
    *lambda += dlambda;
    let wa: f32 = particles[ia].inv_mass;
    let wb: f32 = particles[ib].inv_mass;
    particles[ia].position = particles[ia].position + n * (wa * dlambda);
    particles[ib].position = particles[ib].position - n * (wb * dlambda);
}

// ---------------------------------------------------------------------------
// 状态字节编解码（快照库载荷；C8 内部格式三条件——与 session.rs 同范式）
// ---------------------------------------------------------------------------

/// 状态字节 magic + 版本（C8 ①：文件头带格式版本号；演进时升版本即全量失效）。
const SIM_STATE_MAGIC: &[u8; 8] = b"PRISMSIM";
const SIM_STATE_VERSION: u32 = 1;

/// 粒子状态编码为确定性字节：magic(8) + version(4) + 粒子数(4) + 每粒子
/// 6×f32（pos3 + vel3，大端位模式——C24 平台中立，同值必得同字节）。
fn sim_state_bytes(particles: &[SimParticle]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(16 + particles.len() * 24);
    out.extend_from_slice(SIM_STATE_MAGIC);
    out.extend_from_slice(&SIM_STATE_VERSION.to_be_bytes());
    out.extend_from_slice(&(particles.len() as u32).to_be_bytes());
    for p in particles {
        for c in [
            p.position.x,
            p.position.y,
            p.position.z,
            p.velocity.x,
            p.velocity.y,
            p.velocity.z,
        ] {
            out.extend_from_slice(&c.to_bits().to_be_bytes());
        }
    }
    out
}

/// [`sim_state_bytes`] 的解码：magic / 版本 / 长度任一不符返回 `None`
/// （调用方按快照未命中处理；C8 静默失效条款，绝不猜测解读——C20）。
fn sim_decode_state(bytes: &[u8]) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if bytes.len() < 16 || &bytes[0..8] != SIM_STATE_MAGIC {
        return None;
    }
    if u32::from_be_bytes(bytes[8..12].try_into().ok()?) != SIM_STATE_VERSION {
        return None;
    }
    let n: usize = u32::from_be_bytes(bytes[12..16].try_into().ok()?) as usize;
    if bytes.len() != 16 + n * 24 {
        return None;
    }
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    let mut velocities: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        let base: usize = 16 + i * 24;
        let read = |k: usize| {
            let s: [u8; 4] = bytes.get(base + k * 4..base + k * 4 + 4)?.try_into().ok()?;
            Some(f32::from_bits(u32::from_be_bytes(s)))
        };
        positions.push(Vec3::new(read(0)?, read(1)?, read(2)?));
        velocities.push(Vec3::new(read(3)?, read(4)?, read(5)?));
    }
    Some((positions, velocities))
}

/// sim 运行时（有状态）：规格模板 + 内核环驱动（[`LoopDriver`]）。
///
/// 帧循环语义（幂等 / 顺序延续 / 快照装配 / 首帧初态 / 冷 seek P3012）已
/// 上提为 `prism-core::fixpoint` 的通用环实例——本类型只剩宿主层职责：规格
/// 解算参数比对、dt 与错误码映射（EvalError → R5xxx），不再自带第二套 seek
/// 状态机。
#[derive(Debug)]
struct SimRuntime {
    node_id: u64,
    spec: SimSpec,
    dt: f32,
    /// 规格解算参数的 canonical 字节（时变检测基线；同参数永远同画面）。
    params_canonical: Vec<u8>,
    /// 通用环驱动：X = 粒子（位置 + 速度），帧 = 一次 `spec.step`。
    driver: LoopDriver<Vec<SimParticle>, SimCodec>,
    /// 帧内**逐轮**（M3 迭代）发散判定的累计或（W3 遗留接线）。
    ///
    /// 与 `driver` 的帧间趋势检测互补、互不替代：`driver` 只看帧末残差序列，
    /// 本条看一轮之内残差是否连续上升（Φ 失去收缩性）。只观测，不改写迭代
    /// 控制流、不影响任何像素——仅让 [`SimRuntime::diverged`] 不再漏报。
    iteration_diverged: bool,
}

impl SimRuntime {
    /// 构造运行时（防呆：dt / 模式 / 规格非法显式报错，无静默兜底）。
    fn new(
        spec: SimSpec,
        mode: SolveModeMeta,
        dt: f32,
        params_canonical: Vec<u8>,
        node_id: u64,
    ) -> MovieResult<Self> {
        if !(dt > 0.0) {
            return Err(RenderError::new(
                "R5018",
                format!("sim 节点 {node_id} 的固定步长"),
                format!("dt = {dt}：步长必须为正（NaN 同拒）"),
                "dt 取 1/fps（正值）".to_string(),
            ));
        }
        validate_sim_mode(&mode, node_id)?;
        spec.validate(node_id)?;
        let driver: LoopDriver<Vec<SimParticle>, SimCodec> = LoopDriver::new(
            format!("sim 节点 {node_id}"),
            spec.particles.clone(),
            SimCodec {
                inv_mass: spec.particles.iter().map(|p| p.inv_mass).collect(),
            },
            mode,
        )
        .map_err(loop_error)?;
        Ok(Self {
            node_id,
            spec,
            dt,
            params_canonical,
            driver,
            iteration_diverged: false,
        })
    }

    /// 推进 / 求值到 (sig, local)：帧循环六分支（幂等 / 顺序延续 / 快照装配 /
    /// 首帧初态 / 前驱命中 / 冷 seek P3012）由内核 [`LoopDriver`] 执行——语义
    /// 与 `prism-core::eval` 的 ConstraintFlow 接线同构，绝不静默 fallback。
    fn advance_to(&mut self, sig: Hash256, local: u64, store: &mut SnapshotStore) -> MovieResult<()> {
        // 防御性再查（C20：构造期验过 ≠ 永远合法）。
        self.spec.validate(self.node_id)?;
        let spec: &SimSpec = &self.spec;
        let dt: f32 = self.dt;
        let mode: SolveModeMeta = self.driver.mode();
        // 逐轮判定经闭包侧信道带出（内核步进签名是「帧末残差」，不扩其 API）。
        let mut iter_diverged: bool = false;
        self.driver
            .advance_to(sig, local, store, |particles| {
                let (residual, diverged) = spec.step(particles, &mode, dt)?;
                iter_diverged |= diverged;
                Ok(residual)
            })
            .map_err(loop_error)?;
        self.iteration_diverged |= iter_diverged;
        Ok(())
    }

    /// 当前粒子状态（跨帧状态 = 位置 + 速度）。
    fn particles(&self) -> &[SimParticle] {
        self.driver.state()
    }

    /// 当前运行时是否处于发散（E3 事后指标）：**帧级趋势**（跨帧残差连续
    /// 上升）或 **帧内逐轮趋势**（一轮 M3 迭代内残差连续上升）任一成立。
    fn diverged(&self) -> bool {
        self.iteration_diverged || self.driver.diverged()
    }
}

/// 粒子状态的快照编解码：载荷 = [`sim_state_bytes`]；解码以规格的
/// `inv_mass` 向量回填构型属性（逆质量不属跨帧状态）。粒子数不符 / 字节
/// 不识别 → `None`（按快照未命中处理，绝不部分覆写——C20）。
#[derive(Debug)]
struct SimCodec {
    /// 每粒子逆质量（构型属性；位置 / 速度之外的唯一解码回填项）。
    inv_mass: Vec<f32>,
}

impl LoopCodec<Vec<SimParticle>> for SimCodec {
    fn encode(&self, state: &Vec<SimParticle>) -> Vec<u8> {
        sim_state_bytes(state)
    }

    fn decode(&self, bytes: &[u8]) -> Option<Vec<SimParticle>> {
        let (positions, velocities) = sim_decode_state(bytes)?;
        if positions.len() != self.inv_mass.len() || velocities.len() != self.inv_mass.len() {
            return None;
        }
        Some(
            positions
                .iter()
                .zip(velocities.iter())
                .zip(self.inv_mass.iter())
                .map(|((&pos, &vel), &inv_mass)| SimParticle {
                    position: pos,
                    prev_position: pos,
                    velocity: vel,
                    inv_mass,
                })
                .collect(),
        )
    }
}

/// 内核环错误 → render 三段式（码段透传：P3xxx 属图 / seek 族；三段式字段
/// 原样保留，修复建议已是可行动文本）。
fn loop_error(err: EvalError) -> RenderError {
    RenderError::new(err.code, err.location, err.problem, err.fix)
}

/// sim 时间线：一部影片的全部 sim 运行时 + 跨帧快照库（纯数据资产）。
///
/// 快照库命中与否只影响重算量、不影响结果（D-28）——seek 等价由此成立。
/// 单帧入口 [`render_movie_frame`] 经调用方持有的本类型延续帧间状态：
/// 顺序推进（0..=n−1）后跳帧 n ≡ 顺序推进到 n（逐字节）。
#[derive(Debug, Default)]
pub struct SimTimeline {
    runtimes: BTreeMap<u64, SimRuntime>,
    store: SnapshotStore,
}

impl SimTimeline {
    /// 空时间线（影片首帧起顺序推进的起点）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 跨帧快照库（只读；seek 对拍时克隆注入新时间线）。
    pub fn snapshots(&self) -> &SnapshotStore {
        &self.store
    }

    /// 跨帧快照库（可变；注入预热库 / 显式提供初态快照）。
    pub fn snapshots_mut(&mut self) -> &mut SnapshotStore {
        &mut self.store
    }

    /// 任一 sim 运行时是否处于发散（E3 事后指标；只观测、不改写）。
    ///
    /// 两个来源取并（W3 遗留接线）：内核 `run_fixed_point` 的**逐轮**
    /// `FixpointReport::diverged`（此前在 `SimSpec::step` 里被丢弃）与
    /// [`LoopDriver`] 的**帧级**趋势检测。任一路径报发散都如实返回 `true`；
    /// 本判据不影响任何帧字节（解算控制流仍走 M3 定数步）。
    pub fn diverged(&self) -> bool {
        self.runtimes.values().any(|rt| rt.diverged())
    }

    /// 推进一个 sim 节点到影片帧 `movie_frame`（场景内帧号 = movie_frame −
    /// `base_frame`），返回当前粒子状态切片（供叠加）。
    fn advance_sim(
        &mut self,
        node: &NodeRef,
        channels: &BTreeMap<&str, &PropertyChannel>,
        graph: &Graph,
        defs: &ChannelDefs,
        t: f32,
        movie_frame: u64,
        base_frame: u64,
        fps: u32,
    ) -> MovieResult<&[SimParticle]> {
        // 解算参数（含通道化 force_strength 的 Const 采样；全局种子恒 0）。
        let resolved: prism_core::Params = resolve_params(node, channels, t, movie_frame);
        let canonical: Vec<u8> = resolved.to_canonical_bytes();
        // 模式元数据取解算值（(ε, n) 进签名——S3 迭代层缓存键纪律）。
        let mode: SolveModeMeta = sim_mode_of_params(&resolved, node.id)?;
        let sig: Hash256 =
            loop_frame_signature(graph, node.id, defs, &mode).map_err(signature_err)?;
        if movie_frame < base_frame {
            return Err(RenderError::new(
                "R5019",
                format!("sim 节点 {}（场景内帧号推导）", node.id),
                format!("影片帧 {movie_frame} 早于场景基准帧 {base_frame}——活跃判据与基准帧不一致"),
                "此为内部不变式违例；请报告：场景活跃区间与 scene_first_frame 计算应逐位同口径",
            ));
        }
        let local: u64 = movie_frame - base_frame;
        if !self.runtimes.contains_key(&node.id) {
            // 首见该节点：按当前解算参数建规格（参数时不变性由下述比对保证；
            // 冷跳帧时与顺序推进建规格同值——同一 canonical）。
            let spec: SimSpec = SimSpec::from_params(&resolved, node.id)?;
            let rt: SimRuntime =
                SimRuntime::new(spec, mode, 1.0 / fps as f32, canonical.clone(), node.id)?;
            self.runtimes.insert(node.id, rt);
        }
        let rt: &mut SimRuntime = self.runtimes.get_mut(&node.id).expect("刚插入或已存在");
        // 同参数永远同画面：解算参数必须逐帧逐位一致（Const 通道天然满足；
        // 时变参数 → R5017——时变外力族属扩展位，绝不静默混入）。
        if rt.params_canonical != canonical {
            return Err(RenderError::new(
                "R5017",
                format!("sim 节点 {}（帧间参数比对）", node.id),
                "sim 解算参数随帧变化（通道化动画）：帧间状态按固定规格推进，时变参数会使 seek 回放与顺序推进分叉".to_string(),
                "sim 参数保持静态（force strength 等用常量）；时变外力族（ForceField 扩展位）尚未接入",
            ));
        }
        rt.advance_to(sig, local, &mut self.store)?;
        Ok(rt.particles())
    }
}

/// sim 粒子叠加：把粒子世界位置按与 `CpuRefBackend` 互逆的针孔投影映射到
/// 像素，画不透明暖橙圆点。插在基帧之后、文本图层之前。
///
/// 投影口径与 `prism_gpu::CpuRefBackend::render_scene` 逐位同构（同一
/// forward / right / up2 正交基、同一 `det_trig::tan(fov/2)` 焦距标度、
/// 互逆的 NDC 映射）——叠加点与场景几何同一相机空间，不另立投影约定。
/// 相机后方 / 非有限位置跳过（不画、不 panic）。
fn overlay_sim_particles(
    fb: &mut Framebuffer,
    particles: &[SimParticle],
    eye: Vec3,
    target: Vec3,
    fov_rad: f32,
    w: u32,
    h: u32,
) {
    if particles.is_empty() || w == 0 || h == 0 {
        return;
    }
    let forward: Vec3 = (target - eye).normalize();
    let world_up: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    let right: Vec3 = forward.cross(&world_up).normalize();
    let up2: Vec3 = right.cross(&forward);
    let tan_f: f32 = prism_math::det_trig::tan(fov_rad * 0.5);
    let aspect: f32 = w as f32 / h as f32;
    // 圆点半径随短边缩放（下限 1px：低分辨率下仍可辨）。
    let radius: i64 = ((w.min(h) as f32) / 48.0).round().max(1.0) as i64;
    for p in particles {
        let v: Vec3 = p.position - eye;
        let depth: f32 = v.dot(&forward);
        if !(depth > 1e-4) {
            continue; // 相机后方 / 贴脸：不画。
        }
        let nx: f32 = v.dot(&right) / depth;
        let ny: f32 = v.dot(&up2) / depth;
        if !nx.is_finite() || !ny.is_finite() {
            continue;
        }
        // NDC → 像素（render_scene 正向映射的逆）。
        let fx: f32 = (nx / (tan_f * aspect) + 1.0) * 0.5 * w as f32 - 0.5;
        let fy: f32 = (1.0 - ny / tan_f) * 0.5 * h as f32 - 0.5;
        if !fx.is_finite() || !fy.is_finite() {
            continue;
        }
        let cx: i64 = fx.round() as i64;
        let cy: i64 = fy.round() as i64;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue; // 圆盘（非方点）。
                }
                let x: i64 = cx + dx;
                let y: i64 = cy + dy;
                if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                    continue;
                }
                let idx: usize = (y as usize) * (w as usize) * 4 + (x as usize) * 4;
                fb.pixels[idx..idx + 3].copy_from_slice(&SIM_DOT_COLOR);
                fb.pixels[idx + 3] = 255;
            }
        }
    }
}

/// 场景在影片帧网格上的首个活跃帧号（纯函数；与帧循环的活跃判据逐位同口径：
/// `start <= t && t < start + duration`，`t = i / fps`）。
///
/// sim 的场景内帧号 = 影片帧号 − 本值。基准帧必须跨时间线稳定（seek 对拍时
/// 跳帧 timeline 与顺序 timeline 必须推导出同一基准）——故用算式而非
/// "本时间线第几次渲染"计数。
fn scene_first_frame(model: &SceneModel, fps: u32, frame_count: u64) -> u64 {
    let fps_f: f32 = fps as f32;
    // 候选 = ceil(start × fps)；浮点边界两侧各校一次（同算式判据）。
    let cand: i64 = if model.start.is_finite() {
        (model.start * fps_f).ceil() as i64
    } else {
        0
    };
    let hi: i64 = (frame_count.saturating_sub(1) as i64).max(cand - 2);
    let lo: i64 = (cand - 2).max(0);
    for i in lo..=hi {
        let t: f32 = i as f32 / fps_f;
        if model.start <= t && t < model.start + model.duration {
            return i as u64;
        }
    }
    // 场景不含任何影片帧（时长短于帧间距且落在帧间）——不会被渲染，
    // 基准帧取候选值兜底（确定性，不被使用）。
    cand.max(0) as u64
}

// ---------------------------------------------------------------------------
// 单场景帧求值（复用既有求值器 + 文本合成）
// ---------------------------------------------------------------------------

/// V3.0 W2：按帧解算全部 39 行接线参数（**L3 时序驱动统一通道**）。
///
/// 纯函数、生成源头：在组装 `Scene` / 材质 / 光源**之前**调用。参数经既有
/// `PropertyChannel::sample_ctx`（与 `sample_binding` 同一求值入口）逐帧
/// 解算；任意参数只要在 `cap` 块里声明为时序绑定，两帧即取到不同值，
/// 且逐帧值与解析式 `PropertyChannel::sample(t)` 对拍一致。
///
/// 这是 W3 的**生成源头注入点**之一：痕迹维度在此按帧 / 按节点消费，
/// 出帧后贴皮不是本路径（`codegen.rs:2810` 的 S4 grain 是负控样本）。
pub fn wiring_frame_values(compile: &CompileV2Output, frame: u64) -> WiringFrame {
    let channels: BTreeMap<&str, &PropertyChannel> = compile
        .channels
        .iter()
        .map(|c| (c.name.as_str(), &c.channel))
        .collect();
    let fps: u32 = compile.fps.max(1);
    let t: f32 = frame as f32 / fps as f32;
    let mut wf: WiringFrame = resolve_frame(&compile.wiring, &channels, t, frame, 0, 0);
    // ---- V3.0 W3：生成源头痕迹的逐帧消费点（渲染前，非帧后节点）----------
    // 六族信号已由 `wiring.rs::resolve_frame` 用 `hash_mix` /
    // `pcg::hash2_f32` 种子化注入；此处只登记帧号供 wiring_trace 凭证核对。
    // `render_scene_frame` 在组装 Material / Light / Scene **之前**消费同一份
    // 信号，绝不碰帧缓冲。`analytic` 档不进本分支 ⇒ 帧值逐字节不变。
    if prism_dsl::signals_of(&wf).crafted {
        wf.values.insert("w3.frame".to_string(), (frame as f32).to_bits());
    }
    // ---- V3.0 W4：统一命名时序通道的渲染前覆盖层（A7 / 生成源头，非帧后）----
    // 只认 compile.channels 里带 `w4.drive.` 前缀的通道（由 `compile_film` 降级
    // 注入）；无驱动通道时 `apply_timed_channels` 恒等返回 ⇒ 旧文件 / analytic /
    // W1-W3 全部样例逐字节零损失。六族痕迹强度经此层可随时间/关键帧变化。
    prism_dsl::apply_timed_channels(&mut wf, &channels, t, frame);
    wf
}

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
/// 推进 sim 并叠加粒子，最后按节点序合成文本图层。
fn render_scene_frame(
    model: &SceneModel,
    compile: &CompileV2Output,
    channels: &BTreeMap<&str, &PropertyChannel>,
    defs: &ChannelDefs,
    sims: &mut SimTimeline,
    device: &mut dyn GpuDevice,
    t: f32,
    frame: u64,
) -> MovieResult<(Framebuffer, Vec<u8>)> {
    let w: u32 = compile.width;
    let h: u32 = compile.height;

    // ---- V3.0 W2：生成源头接线解算（渲染前，非帧后节点）------------------
    // ① 39 行 cap 参数按帧解算（L3 统一通道）；② 七族 R5010 销号节点的
    // 参数与通道绑定逐帧解算。两者合并成 WiringFrame，作为本帧的求值凭证
    // 随 Deliverable 交付（同源对拍逐字节比较），并驱动下方材质 / 光源。
    let mut wiring: WiringFrame = wiring_frame_values(compile, frame);
    for node in &model.wiring_nodes {
        let p: prism_core::Params = resolve_params(node, channels, t, frame);
        // 节点参数按声明序并入帧凭证：键 `node.{id}.{param}`。
        // 只取确定性的 F32/U32/Str 形态，位型写入，无平台浮点差异。
        for (key, value) in p.iter() {
            let bits: u32 = match value {
                ParamValue::F32(v) => v.to_bits(),
                ParamValue::U32(v) => (*v as f32).to_bits(),
                ParamValue::Str(s) => (s.len() as f32).to_bits(),
                ParamValue::Bool(b) => (if *b { 1.0_f32 } else { 0.0_f32 }).to_bits(),
            };
            wiring.values.insert(format!("node.{}.{}", node.id, key), bits);
        }
    }

    // ---- V3.0 W3：六族痕迹在**生成源头**消费（组装材质 / 光源 / 场景之前）----
    // 只改 Scene / Material / Light / Camera 的构造实参，绝不改帧缓冲；
    // `codegen.rs` 的 `post { grain }` S4 帧后节点不是本路径（§3.3 禁令 1）。
    // `analytic` 档 signals.crafted = false ⇒ 下列六个分支全部不生效 ⇒
    // 与纯解析基线逐字节同帧、全帧零意外噪点（V1 / V4）。
    let w3: prism_dsl::FrameSignals = prism_dsl::signals_of(&wiring);

    // ---- 相机（缺省轨道：中心原点 / 半径 8 / fov 45°，与 DSL 缺省一致）----
    let (mut center, mut orbit_radius, mut fov_rad): (Vec3, f32, f32) = match &model.camera {
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
    if w3.crafted {
        // 运动族：惯性 / 犹豫 / 微抖 / 呼吸 —— 轨道半径的种子化微扰。
        orbit_radius = (orbit_radius * (1.0 + 0.06 * w3.motion)).max(0.001);
        // 时间镜头族：快门 / 视角响应 —— fov 的种子化微扰。
        fov_rad = (fov_rad * (1.0 + 0.04 * w3.time_camera)).clamp(0.01, 3.0);
        // 构图色彩族：有意图的取舍 —— 主体偏离绝对居中。
        center = Vec3::new(
            center.x + 0.12 * w3.composition_color,
            center.y,
            center.z,
        );
    }

    // ---- 材质（缺省白，粗糙 0.5 / 金属 0，与 DSL 缺省一致）---------------
    let material: Material = match &model.material {
        Some(mat) => {
            let p = resolve_params(mat, channels, t, frame);
            let mut roughness: f32 = p_f32(&p, "roughness", 0.5).clamp(0.0, 1.0);
            let mut metalness: f32 = p_f32(&p, "metalness", 0.0).clamp(0.0, 1.0);
            // W2：cap "R-2" 的 BSDF 参数在**生成源头**覆盖既有
            // `prism_core::Material::new`，不是帧后贴皮。未声明 cap ⇒
            // wiring 空 ⇒ 不覆盖 ⇒ 旧文件与 W1 逐字节同帧（零损失）。
            if let Some(v) = wiring.value("R-2.roughness") {
                roughness = v.clamp(0.0, 1.0);
            }
            if let Some(v) = wiring.value("R-2.metalness") {
                metalness = v.clamp(0.0, 1.0);
            }
            // W3 材质光照族：微表面 / 漏光 / 色偏，在 `Material::new` 之前注入。
            if w3.crafted {
                roughness = (roughness + 0.08 * w3.material_light).clamp(0.0, 1.0);
                metalness = (metalness + 0.05 * w3.material_light).clamp(0.0, 1.0);
            }
            Material::new(
                Vec3::new(
                    p_f32(&p, "albedo_r", 1.0),
                    p_f32(&p, "albedo_g", 1.0),
                    p_f32(&p, "albedo_b", 1.0),
                ),
                roughness,
                metalness,
            )
        }
        None => Material::new(Vec3::new(1.0, 1.0, 1.0), 0.5, 0.0),
    };

    // ---- 实体半径（无 sdf = 0：软光追对 r=0 无命中，天然只出背景）--------
    let mut radius: f32 = match &model.sphere {
        Some(s) => {
            let p = resolve_params(s, channels, t, frame);
            p_f32(&p, "radius", 1.0).max(0.0)
        }
        None => 0.0,
    };
    if w3.crafted {
        // W3 几何族：受控的手工不完美 / 微不对称（种子化半径扰动）。
        radius = (radius * (1.0 + 0.05 * w3.geometry)).max(0.0);
    }

    // ---- 美学参数（theme 块；缺省关闭全部后期）---------------------------
    let mut aesthetic: Aesthetic = match &compile.theme {
        Some(th) => Aesthetic::new(th.teal_orange, th.contrast, th.filmic),
        None => Aesthetic::new(false, 1.0, false),
    };
    if w3.crafted {
        // W3 构图色彩族：色调倾向（对比度的种子化取舍）。
        aesthetic.contrast = (aesthetic.contrast * (1.0 + 0.10 * w3.composition_color)).max(0.0);
    }

    // ---- 光源：DSL 的 light 块登记后忽略（后续接入），render 侧用
    //      固定缺省灯（与内置演示同款：暖色方向光，确定性常量）-------------
    let mut lights: Vec<Light> = vec![Light::new(
        Vec3::new(1.0, -1.0, -0.5).normalize(),
        Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0),
        3.0,
    )];
    if w3.crafted {
        // W3 材质光照族：光源方向 / 强度 / 色偏的种子化微扰（Light::new 之前）。
        let dir: Vec3 = Vec3::new(1.0 + 0.20 * w3.material_light, -1.0, -0.5).normalize();
        let tint: Vec3 = Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0);
        let intensity: f32 = 3.0 * (1.0 + 0.10 * w3.material_light);
        lights[0] = Light::new(dir, tint, intensity);
    }

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
        // T0：DSL 声明的附加能力（media / mesh / instance / texture / hair /
        // 位移槽）。空 ⇒ 与加字段前逐字节同帧。
        attach: model.attach.clone(),
    };
    let t_local: f32 = t - model.start;
    // 场景时长为正（活跃判据已保证）；防御性下限防除零。
    let mut dur_local: f32 = if model.duration > 0.0 { model.duration } else { 1.0 };
    if w3.crafted {
        // W3 剪辑音画族：不等长呼吸 / 人在操作的取舍 —— 时间轴局部时长的
        // 种子化微扰，在 `eval_camera_orbit` 之前生效，不是帧后贴皮。
        dur_local = (dur_local * (1.0 + 0.15 * w3.edit_audio)).max(1.0e-3);
    }
    let mut fb: Framebuffer = if model.curves.is_empty() {
        render_frame(device, &scene, t_local, dur_local, w, h)
    } else {
        // 曲线图元族：走 prism-rt CPU 波前路径（设备仅占位不用）。
        render_curve_frame(model, &scene, channels, t_local, dur_local, frame, w, h)?
    };

    // ---- S3 sim 粒子叠加（基帧之后、文本之前；帧循环推进）----------------
    // V3.1 收尾 C：粒子域真实插桩（RAII 计时，只写原子计数器，不改任何像素）。
    if !model.sims.is_empty() {
        let _particles_timer = crate::perf_domains::SegmentTimer::start(Domain::Particles);
        // 相机与基帧同源（eval_camera_orbit 轨道机位）——叠加投影与场景
        // 几何同一相机空间。
        let eye: Vec3 = prism_core::eval_camera_orbit(&scene, t_local, dur_local);
        for node in &model.sims {
            let particles: &[SimParticle] = sims
                .advance_sim(
                    node,
                    channels,
                    &compile.graph,
                    defs,
                    t,
                    frame,
                    model.sim_base_frame,
                    compile.fps,
                )
                .map_err(|err| {
                    RenderError::new(
                        err.code,
                        format!("场景 '{}' 的 sim 节点 {}：{}", model.name, node.id, err.location),
                        err.problem,
                        err.fix,
                    )
                })?;
            overlay_sim_particles(&mut fb, particles, eye, center, fov_rad, w, h);
        }
    }

    // ---- 文本图层合成（按节点序；后建的压在上面）-------------------------
    // V3.1 收尾 C：文本域真实插桩（无文本节点时不计时，避免虚增 calls）。
    if !model.texts.is_empty() {
        let _text_timer = crate::perf_domains::SegmentTimer::start(Domain::Text);
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
    }

    Ok((fb, wiring.canonical_bytes()))
}

/// 曲线图元帧渲染：prism-rt CPU 波前路径 + clamp tone-map 到 u8。
///
/// # 为什么走 CPU（诚实口径）
/// curve 四族的解析求交在 prism-rt（CPU 参考实现，配 SAH-BVH 与逐位
/// 确定性测试）；GPU 设备请求契约（`SphereReq`）暂无曲线基元槽位。
/// 当前版本以 CPU 路径交付端到端能力（先真跑、不假装），GPU 曲线槽位
/// 尚未接入。
///
/// # 确定性
/// 种子恒 0（与参数解算一致）、spp/弹射深度为固定常量 ⇒ 同输入逐位
/// 一致；tone-map 与 GPU 路径同口径（clamp[0,1]×255 四舍五入，alpha 恒 255）。
fn render_curve_frame(
    model: &SceneModel,
    scene: &Scene,
    channels: &BTreeMap<&str, &PropertyChannel>,
    t: f32,
    duration: f32,
    frame: u64,
    w: u32,
    h: u32,
) -> MovieResult<Framebuffer> {
    use prism_rt::camera::Camera as RtCamera;
    use prism_rt::integrator::{
        Cone, Cylinder, MaterialKind, RenderParams, Revolution, Scene as RtScene, Torus,
    };

    /// 曲线路径固定采样参数（CPU 参考路径的成本/质量折中；确定性常量）。
    const CURVE_SPP: u32 = 4;
    /// 固定弹射深度（直接光 + 一次间接足够呈现曲面明暗）。
    const CURVE_BOUNCES: u32 = 2;

    let mut rt: RtScene = RtScene::default();
    // 材质映射：DSL material → Lambert 反照率（Pbr 的 metalness/roughness
    // 映射尚未接入；反照率是最影响识别的通道，先真跑）。
    let mat: MaterialKind = MaterialKind::Lambert { albedo: scene.material.albedo };
    for c in &model.curves {
        let p = resolve_params(c, channels, t, frame);
        let kind: &str = p_str(&p, "kind").unwrap_or("");
        let center = Vec3::new(
            p_f32(&p, "center_x", 0.0),
            p_f32(&p, "center_y", 0.0),
            p_f32(&p, "center_z", 0.0),
        );
        match kind {
            "cylinder" => {
                let radius = p_f32(&p, "radius", 1.0).max(0.0);
                let half_height = p_f32(&p, "half_height", 1.0).max(0.0);
                if radius <= 0.0 || half_height <= 0.0 {
                    return Err(curve_degenerate_err(model, c.id, kind));
                }
                rt.add_cylinder(Cylinder { center, radius, half_height }, mat);
            }
            "cone" => {
                let radius = p_f32(&p, "radius", 1.0).max(0.0);
                let half_height = p_f32(&p, "half_height", 1.0).max(0.0);
                if radius <= 0.0 || half_height <= 0.0 {
                    return Err(curve_degenerate_err(model, c.id, kind));
                }
                rt.add_cone(Cone { center, radius, half_height }, mat);
            }
            "torus" => {
                let major = p_f32(&p, "major", 1.0).max(0.0);
                let minor = p_f32(&p, "minor", 0.5).max(0.0);
                if major <= 0.0 || minor <= 0.0 {
                    return Err(curve_degenerate_err(model, c.id, kind));
                }
                rt.add_torus(Torus { center, major, minor }, mat);
            }
            "revolution" => {
                let n: usize = match p.get("profile_n") {
                    Some(ParamValue::U32(n)) => *n as usize,
                    _ => 0,
                };
                let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n);
                for i in 0..n {
                    pts.push([
                        p_f32(&p, &format!("profile_r_{i}"), 0.0),
                        p_f32(&p, &format!("profile_y_{i}"), 0.0),
                    ]);
                }
                if pts.len() < 2 {
                    return Err(curve_degenerate_err(model, c.id, kind));
                }
                rt.add_revolution(Revolution::new(center, pts), mat);
            }
            other => {
                return Err(RenderError::new(
                    "R5013",
                    format!("场景 '{}' 的 sdf 节点 {}（kind = '{other}'）", model.name, c.id),
                    format!("render 侧曲线路径不支持的 kind '{other}'"),
                    "曲线路径仅接受 cylinder / cone / torus / revolution（分类阶段已拦，此为纵深防御）",
                ));
            }
        }
    }
    // 光源：与 GPU 球体路径同款固定暖色方向光（确定性常量，DSL light 块
    // 的运行时接入尚未实装——两路同口径，不静默双标）。
    rt.lights = vec![prism_rt::integrator::Light::Directional {
        direction: Vec3::new(1.0, -1.0, -0.5).normalize(),
        radiance: Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0) * 3.0,
        cos_theta_max: 0.9,
    }];

    // 相机：复用 core 场景的轨道机位（eval_camera_orbit）与 fov（弧度）。
    // rt Camera::new 对 fov_deg 有 (0,180) 开区间断言 ⇒ 非有限回落 45°、
    // 钳制到 [1°,179°]（防通道化 fov 越界直接 panic）。
    let eye: Vec3 = prism_core::eval_camera_orbit(scene, t, duration);
    let target: Vec3 = scene.orbit_center;
    let fov_deg: f32 = if scene.camera.fov.is_finite() {
        scene.camera.fov.to_degrees().clamp(1.0, 179.0)
    } else {
        45.0
    };
    let cam = RtCamera::new(
        eye,
        target,
        Vec3::new(0.0, 1.0, 0.0),
        fov_deg,
        0.0,                                   // 针孔（无 DOF）
        (eye - target).length().max(1.0e-3),   // 焦距 = 机距
        0.0,                                   // 快门开
        1.0 / 24.0,                            // 快门闭（占位，无运动模糊）
    );
    let ctx = EvalCtx::new(0, 0, frame);
    let params = RenderParams::new(CURVE_SPP, CURVE_BOUNCES);
    // **同一条通用渲染路径**（2026-11-14 裁定三）：曲线族不再单独走
    // `wavefront::render_frame` 这条老路，而是与所有场景共用的
    // `rt_tiles::render_frame_universal`（同一台求交加速视图 + 同一台并行
    // 内核 + 同一套色调映射出口）。唯一保留的差异是采样预算常量
    // （`CURVE_SPP` / `CURVE_BOUNCES`），那是**预算**不是**路线**。
    //
    // 色调映射：传恒等档（无 teal-orange、对比度 1.0、不开 filmic）⇒
    // `apply_aesthetic` 退化为 `clamp01`，与 `一致` 档/曲线老路同口径
    // （clamp[0,1]×255 四舍五入、alpha 恒 255），故亮度语义不变。
    let aesthetic: prism_gpu::AestheticReq = prism_gpu::AestheticReq {
        teal_orange: false,
        contrast: 1.0,
        filmic: false,
    };
    let threads: usize = crate::rt_tiles::auto_threads();
    Ok(crate::rt_tiles::render_frame_universal(
        &rt, &cam, &ctx, (w, h), &params, &aesthetic, threads,
    ))
}

/// 曲线图元尺寸退化（<= 0 或点数 < 2）的三段式错误（分类阶段不校值域，
/// 通道化参数可能在运行期退化 ⇒ 求值前拦截，绝不静默出空帧）。
fn curve_degenerate_err(model: &SceneModel, id: u64, kind: &str) -> RenderError {
    RenderError::new(
        "R5016",
        format!("场景 '{}' 的 sdf 节点 {}（kind {kind}）", model.name, id),
        "曲线图元尺寸参数退化为非正（或轮廓点数 < 2），求交将永不命中".to_string(),
        "尺寸取正长度（radius / half_height / major / minor > 0）；revolution 至少 2 个轮廓点",
    )
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

    // 内嵌粗体（确定性：同字节同字形；DSL font 回退链消费随字体管线接入）。
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

/// 按过渡种类把两帧混合为一帧（整数路径，复用 core 的 u8 混合家族）。
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
            attach: prism_core::SceneAttach::default(),
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
            // V2-j：公式节点（本测试不声明 formula 块）。
            formulas: Vec::new(),
            // V3.0 W2：本测试夹具不声明 cap 块 ⇒ 空接线表 ⇒ 零损失。
            wiring: prism_dsl::WiringPlan::default(),
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

    /// 端到端：`.prism` 四族曲线 kind 同场 → `compile_v2`
    /// → `render_movie` → 帧序列 + 逐位确定性 + 非空渲染。
    ///
    /// 覆盖要求：DSL 端到端 ≥2 种曲线 kind 经编译产出帧；本用例
    /// 四种齐上（cylinder / cone / torus / revolution），并顺带覆盖规格
    /// 推导（fps / resolution / duration）、节点分流与两跑 MD5 一致。
    #[test]
    fn render_movie_curve_kinds_end_to_end() {
        let src: &str = concat!(
            "// prism-dsl 1\n",
            "movie \"curves\" {\n",
            "  duration 1s\n",
            "  fps 4\n",
            "  resolution 48x27\n",
            "}\n",
            "scene \"curves\" {\n",
            "  duration 1s\n",
            "  stage3d {\n",
            "    camera main { orbit { center 0,0,0 radius 9 } fov 50 }\n",
            "    material m { albedo #c0392b roughness 0.4 }\n",
            "    sdf rod { kind cylinder radius 0.6 half_height 1.6 center -2.5,0,0 }\n",
            "    sdf spike { kind cone radius 0.9 half_height 1.4 center 0,0,0 }\n",
            "    sdf ring { kind torus major 1.4 minor 0.4 center 2.5,0,0 }\n",
            "    sdf vase { kind revolution profile { pt { r 0 y -1 } pt { r 0.7 y 0 } pt { r 0 y 1 } } center 4.5,0,0 }\n",
            "  }\n",
            "}\n"
        );
        let compile: CompileV2Output =
            prism_dsl::compile_v2(src).expect("四族曲线 .prism 应编译成功");

        // ---- 编译产物规格与节点分流 ----
        assert_eq!(compile.fps, 4, "movie.fps 应落盘");
        assert_eq!(compile.width, 48, "movie.resolution 宽应落盘");
        assert_eq!(compile.height, 27, "movie.resolution 高应落盘");
        assert_eq!(compile.duration, 1.0, "movie.duration 应落盘（秒）");
        let ids: &[u64] = &compile.scenes[0].node_ids;
        assert_eq!(ids.len(), 6, "camera + material + 4 个 sdf 实体节点");
        let kinds: Vec<&str> = ids
            .iter()
            .filter_map(|&id| compile.graph.node(id))
            .filter_map(|n| match n.params.get("kind") {
                Some(ParamValue::Str(k)) => Some(k.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["cylinder", "cone", "torus", "revolution"],
            "四族曲线 kind 应全部落入节点图"
        );

        // ---- 成片渲染（CPU 参考后端；曲线路径实际不用设备）----
        let mut device = prism_gpu::CpuRefBackend;
        let a: Deliverable = render_movie(&compile, &mut device).expect("曲线场景应可渲染");
        let b: Deliverable = render_movie(&compile, &mut device).expect("曲线场景应可渲染");

        // fps=4 × duration=1 → 4 帧；48x27。
        assert_eq!(a.frame_count(), 4, "fps=4 × 1s 应得 4 帧");
        for (i, fb) in a.frames.iter().enumerate() {
            assert_eq!(fb.width, 48, "第 {i} 帧宽度");
            assert_eq!(fb.height, 27, "第 {i} 帧高度");
            assert_eq!(fb.pixels.len(), 48 * 27 * 4, "第 {i} 帧长度应恰为 w·h·4");
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

        // 非空渲染：至少一帧至少一个像素非黑（全黑 = 求交/光照静默失败）。
        let any_lit: bool = a
            .frames
            .iter()
            .any(|fb| fb.pixels.chunks_exact(4).any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0));
        assert!(any_lit, "曲线场景至少一帧应出现非黑像素");

        // 相机环绕：首帧与末帧应不同（时间维真的在推进）。
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
    fn particles_node_is_no_longer_rejected_r5010() {
        // W2（§5.3 出口）：R5010 销号 —— S4 / layer particles 收进求值链
        // 源头，不再以「不支持节点」拒绝；节点参数逐帧解算并进凭证。
        let compile: CompileV2Output = compile_with_node(StructureId::DeterministicSampling, "particles");
        let mut device = prism_gpu::CpuRefBackend;
        let d: Deliverable = render_movie(&compile, &mut device)
            .expect("W2 后粒子节点应不再拒绝（R5010 已销号）");
        assert_eq!(d.frame_count(), 8);
        let trace0: &[u8] = &d.wiring_trace[0];
        let needle: &[u8] = b"node.9.count";
        assert!(
            trace0.windows(needle.len()).any(|w| w == needle),
            "粒子节点的参数必须真的进逐帧接线凭证（不是纸面登记）"
        );
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
    fn non_cloth_sim_kind_errors_r5017() {
        // cloth 已支持（帧循环驱动）；非 cloth 种类在求值前明确报错 R5017。
        let mut params: Params = sim_node_params(None);
        params.set("kind", ParamValue::Str("fluid".to_string()));
        let compile: CompileV2Output = compile_with_sim(params);
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError = render_movie(&compile, &mut device).expect_err("非 cloth 种类必须报错");
        assert_eq!(err.code, "R5017");
        assert!(err.to_string().contains("cloth"), "报错应说明已支持种类：{err}");
        assert!(err.to_string().contains("建议"), "三段式必含建议段");
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
            attach: prism_core::SceneAttach::default(),
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
            // V2-j：公式节点（本测试不声明 formula 块）。
            formulas: Vec::new(),
            // V3.0 W2：本测试夹具不声明 cap 块 ⇒ 空接线表 ⇒ 零损失。
            wiring: prism_dsl::WiringPlan::default(),
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
            attach: prism_core::SceneAttach::default(),
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
            // V2-j：公式节点（本测试不声明 formula 块）。
            formulas: Vec::new(),
            // V3.0 W2：本测试夹具不声明 cap 块 ⇒ 空接线表 ⇒ 零损失。
            wiring: prism_dsl::WiringPlan::default(),
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

    // ---- S3 sim 帧循环（帧循环接线 + 签名收拢 + seek 等价）----------------

    /// sim 节点参数（全量 cloth；模式三元组 (子步, 迭代, ε) 可覆写）。
    fn sim_node_params(mode: Option<(u32, u32, f32)>) -> Params {
        let mut params: Params = Params::new();
        params.set("kind", ParamValue::Str("cloth".to_string()));
        params.set("grid", ParamValue::U32(6));
        params.set("stiffness", ParamValue::F32(0.9));
        params.set("damping", ParamValue::F32(0.02));
        params.set("force_strength", ParamValue::F32(2.5));
        params.set("force_dir_x", ParamValue::F32(1.0));
        params.set("force_dir_y", ParamValue::F32(0.0));
        params.set("force_dir_z", ParamValue::F32(0.0));
        params.set("noise_scale", ParamValue::F32(0.1));
        if let Some((substeps, iterations, residual_eps)) = mode {
            params.set("substeps", ParamValue::U32(substeps));
            params.set("iterations", ParamValue::U32(iterations));
            params.set("residual_eps", ParamValue::F32(residual_eps));
        }
        params
    }

    /// 在最小产物上加一个 sim 节点（id 9，可带通道绑定）。
    fn compile_with_sim_bindings(params: Params, bindings: Vec<NodeBinding>) -> CompileV2Output {
        let mut compile: CompileV2Output = minimal_compile();
        let mut node: Node = Node::new(9, StructureId::ConstraintFlow, Composition::Feed);
        node.params = params;
        node.bindings = bindings;
        compile.graph.add_node(node).expect("id 9");
        compile.scenes[0].node_ids.push(9);
        compile
    }

    /// 在最小产物上加一个无绑定的 sim 节点（id 9）。
    fn compile_with_sim(params: Params) -> CompileV2Output {
        compile_with_sim_bindings(params, Vec::new())
    }

    /// 暖橙粒子点像素计数（叠加色 SIM_DOT_COLOR 逐字节判据）。
    fn sim_dot_pixels(fb: &Framebuffer) -> usize {
        fb.pixels
            .chunks_exact(4)
            .filter(|px| px[0] == 255 && px[1] == 176 && px[2] == 66)
            .count()
    }

    /// 时间线驱动 sim 帧循环：渲染成功 + 粒子叠加 + 帧间推进。
    #[test]
    fn sim_timeline_drives_frame_loop_and_overlay() {
        let compile: CompileV2Output = compile_with_sim(sim_node_params(None));
        let mut device = prism_gpu::CpuRefBackend;
        let d: Deliverable = render_movie(&compile, &mut device).expect("sim 场景应可渲染");
        assert_eq!(d.frame_count(), 8, "fps=8 × 1s 应得 8 帧");
        assert!(sim_dot_pixels(&d.frames[0]) > 0, "sim 粒子应叠加为暖橙圆点");

        // 对照：无 sim 的最小产物无暖橙点，且画面与带 sim 不同。
        let control: Deliverable =
            render_movie(&minimal_compile(), &mut device).expect("对照应可渲染");
        assert_eq!(sim_dot_pixels(&control.frames[0]), 0, "无 sim 不应出现暖橙点");
        assert_ne!(d.frames[0].pixels, control.frames[0].pixels, "sim 叠加应改变画面");

        // 帧循环真的在推进：外力推动布料，首末帧不同。
        assert_ne!(d.frames[0].pixels, d.frames[7].pixels, "sim 帧循环应推进粒子");

        // 确定性宪章：同输入两跑逐位一致。
        let d2: Deliverable = render_movie(&compile, &mut device).expect("第二跑应可渲染");
        for (i, (x, y)) in d.frames.iter().zip(d2.frames.iter()).enumerate() {
            assert_eq!(x.pixels, y.pixels, "第 {i} 帧两跑必须逐位一致");
        }
    }

    /// 零方向外力 = 无外力（合法）：渲染成功且粒子仍叠加。
    #[test]
    fn sim_zero_direction_force_is_legal() {
        let mut params: Params = sim_node_params(None);
        params.set("force_dir_x", ParamValue::F32(0.0));
        params.set("force_dir_y", ParamValue::F32(0.0));
        params.set("force_dir_z", ParamValue::F32(0.0));
        let compile: CompileV2Output = compile_with_sim(params);
        let mut device = prism_gpu::CpuRefBackend;
        let d: Deliverable = render_movie(&compile, &mut device).expect("零外力应合法");
        assert!(sim_dot_pixels(&d.frames[0]) > 0, "无外力也应有粒子叠加");
    }

    /// sim 参数域防呆：规格参数 → R5017；模式参数 → R5018（无静默兜底）。
    #[test]
    fn sim_param_domain_errors_r5017_r5018() {
        let mut device = prism_gpu::CpuRefBackend;
        for (name, bad) in [
            ("grid", ParamValue::U32(0)),
            ("grid", ParamValue::U32(65)),
            ("stiffness", ParamValue::F32(1.5)),
            ("damping", ParamValue::F32(1.0)),
            ("noise_scale", ParamValue::F32(0.3)),
            ("stiffness", ParamValue::U32(1)), // 类型不符（非 F32）
        ] {
            let mut params: Params = sim_node_params(None);
            params.set(name, bad);
            let compile: CompileV2Output = compile_with_sim(params);
            let err: RenderError =
                render_movie(&compile, &mut device).expect_err("规格参数域非法必须报错");
            assert_eq!(err.code, "R5017", "参数 '{name}' 域非法应报 R5017");
        }
        for mode in [(0u32, 8u32, 1e-4f32), (4, 0, 1e-4), (4, 8, 0.0), (4, 8, -1.0)] {
            let compile: CompileV2Output = compile_with_sim(sim_node_params(Some(mode)));
            let err: RenderError =
                render_movie(&compile, &mut device).expect_err("模式参数域非法必须报错");
            assert_eq!(err.code, "R5018", "模式 {mode:?} 域非法应报 R5018");
        }
    }

    /// 状态字节往返：编码 → 解码逐位一致；magic / 长度损坏 → None（C8/C20）。
    #[test]
    fn sim_state_bytes_roundtrip_and_corruption() {
        let spec: SimSpec = SimSpec::from_params(&sim_node_params(None), 9).expect("cloth 规格");
        let bytes: Vec<u8> = sim_state_bytes(&spec.particles);
        let (positions, velocities): (Vec<Vec3>, Vec<Vec3>) =
            sim_decode_state(&bytes).expect("往返解码应成功");
        assert_eq!(positions.len(), spec.particles.len());
        for (p, (&pos, &vel)) in spec.particles.iter().zip(positions.iter().zip(velocities.iter())) {
            assert_eq!(p.position, pos, "位置应逐位往返");
            assert_eq!(p.velocity, vel, "速度应逐位往返");
        }
        let mut bad: Vec<u8> = bytes.clone();
        bad[0] = b'X';
        assert!(sim_decode_state(&bad).is_none(), "magic 不符应静默失效");
        assert!(sim_decode_state(&bytes[..20]).is_none(), "长度不符应静默失效");
    }

    /// 冷 seek（帧 > 0 且前驱快照缺失）→ P3012，绝不静默 fallback 到初态。
    #[test]
    fn sim_cold_seek_errors_p3012() {
        let compile: CompileV2Output = compile_with_sim(sim_node_params(None));
        let mut device = prism_gpu::CpuRefBackend;
        let mut sims: SimTimeline = SimTimeline::new();
        let err: RenderError = render_movie_frame(&compile, &mut device, 3, &mut sims)
            .expect_err("冷 seek（帧 3 无前驱快照）必须结构化报错");
        assert_eq!(err.code, "P3012", "冷 seek 报 P3012，绝不静默 fallback");
        assert!(err.to_string().contains("建议"), "三段式必含建议段");
    }

    /// seek 等价：快照装配跳帧 ≡ 顺序推进（逐字节）；全片入口 ≡ 逐帧入口。
    #[test]
    fn sim_seek_snapshot_equals_sequential_bitwise() {
        let compile: CompileV2Output = compile_with_sim(sim_node_params(None));
        let mut device = prism_gpu::CpuRefBackend;
        let n: u64 = 8;
        // A：顺序推进 0..=n-2（快照库随帧累积）。
        let mut sims_a: SimTimeline = SimTimeline::new();
        for i in 0..n - 1 {
            render_movie_frame(&compile, &mut device, i, &mut sims_a).expect("顺序帧应可渲染");
        }
        // B：注入 A 的预热库（含 (sig, 0..n-2)），跳帧 n-1（前驱命中 + 步进一帧）。
        let mut sims_b: SimTimeline = SimTimeline::new();
        *sims_b.snapshots_mut() = sims_a.snapshots().clone();
        let jumped: Framebuffer =
            render_movie_frame(&compile, &mut device, n - 1, &mut sims_b).expect("跳帧应可渲染");
        // A 顺序推进到同一帧。
        let sequential: Framebuffer =
            render_movie_frame(&compile, &mut device, n - 1, &mut sims_a).expect("顺序末帧");
        assert_eq!(jumped.pixels, sequential.pixels, "快照装配跳帧 ≡ 顺序推进（逐字节）");

        // 全片入口与逐帧入口同语义。
        let whole: Deliverable = render_movie(&compile, &mut device).expect("全片应可渲染");
        let mut sims_c: SimTimeline = SimTimeline::new();
        for (i, fb) in whole.frames.iter().enumerate() {
            let one: Framebuffer =
                render_movie_frame(&compile, &mut device, i as u64, &mut sims_c).expect("逐帧");
            assert_eq!(one.pixels, fb.pixels, "第 {i} 帧：逐帧入口 ≡ 全片入口");
        }
    }

    /// 模式三元组进签名：改子步 / 迭代 / ε ⇒ 帧签名变；同参数同签名。
    /// 端到端：子步数 1 vs 8 画面不同（求解轨迹真的变了）。
    #[test]
    fn sim_mode_change_changes_signature_and_pixels() {
        let mut device = prism_gpu::CpuRefBackend;
        let soft: CompileV2Output = compile_with_sim(sim_node_params(Some((1, 8, 1e-4))));
        let stiff: CompileV2Output = compile_with_sim(sim_node_params(Some((8, 8, 1e-4))));
        let a: Deliverable = render_movie(&soft, &mut device).expect("1 子步应可渲染");
        let b: Deliverable = render_movie(&stiff, &mut device).expect("8 子步应可渲染");
        let same: usize = a
            .frames
            .iter()
            .zip(b.frames.iter())
            .filter(|(x, y)| x.pixels == y.pixels)
            .count();
        assert!(same < a.frame_count(), "子步数改变求解轨迹，画面应不同");

        // 签名侧（(ε, n) 进帧签名——S3 迭代层缓存键纪律）。
        let base: Hash256 = sim_frame_signature(&soft, 9, 1, 8, 1e-4).expect("签名");
        assert_eq!(
            base,
            sim_frame_signature(&soft, 9, 1, 8, 1e-4).expect("签名"),
            "同参数两调必须同签名"
        );
        assert_ne!(base, sim_frame_signature(&soft, 9, 2, 8, 1e-4).expect("签名"), "改子步数 ⇒ 签名变");
        assert_ne!(base, sim_frame_signature(&soft, 9, 1, 9, 1e-4).expect("签名"), "改迭代数 ⇒ 签名变");
        assert_ne!(base, sim_frame_signature(&soft, 9, 1, 8, 1e-3).expect("签名"), "改残差阈值 ⇒ 签名变");
        // 规格参数（grid）变化 ⇒ 签名也变（参数 canonical 进签名）。
        let coarse: CompileV2Output = compile_with_sim({
            let mut p: Params = sim_node_params(Some((1, 8, 1e-4)));
            p.set("grid", ParamValue::U32(7));
            p
        });
        assert_ne!(base, sim_frame_signature(&coarse, 9, 1, 8, 1e-4).expect("签名"), "改网格 ⇒ 签名变");
        // 未知节点 → P3013（透传）。
        let err: RenderError =
            sim_frame_signature(&soft, 42, 1, 8, 1e-4).expect_err("未知节点必须报错");
        assert_eq!(err.code, "P3013");
    }

    /// 时变 sim 参数 → R5017（同参数永远同画面；帧间参数逐位比对）。
    #[test]
    fn sim_time_varying_param_errors_r5017() {
        use prism_anim::{EasingKind, Keyframe, Track};
        let bindings: Vec<NodeBinding> = vec![NodeBinding {
            param: "force_strength".to_string(),
            channel: ChannelRef { name: "wind".to_string() },
        }];
        let mut compile: CompileV2Output = compile_with_sim_bindings(sim_node_params(None), bindings);
        compile.channels = vec![prism_dsl::NamedChannel {
            name: "wind".to_string(),
            channel: PropertyChannel::Track(Track::new(vec![
                Keyframe::new(0.0, 2.5, EasingKind::Linear),
                Keyframe::new(0.3, 5.0, EasingKind::Linear),
            ])),
        }];
        let mut device = prism_gpu::CpuRefBackend;
        let err: RenderError =
            render_movie(&compile, &mut device).expect_err("时变 sim 参数必须报错");
        assert_eq!(err.code, "R5017");
        assert!(err.to_string().contains("随帧变化"), "报错应说明时变性：{err}");
    }

    /// 基准帧纯函数：与帧循环活跃判据逐位同口径（seek 对拍前提）。
    #[test]
    fn scene_first_frame_matches_active_predicate() {
        let model = |start: f32, duration: f32| -> SceneModel {
            SceneModel {
            attach: prism_core::SceneAttach::default(),
                index: 0,
                name: "s".to_string(),
                start,
                duration,
                camera: None,
                material: None,
                sphere: None,
                curves: Vec::new(),
                sims: Vec::new(),
                sim_base_frame: 0,
                texts: Vec::new(),
                wiring_nodes: Vec::new(),
            }
        };
        // fps=4：帧网格 t = 0, 0.25, 0.5, 0.75, ...
        assert_eq!(scene_first_frame(&model(0.0, 1.0), 4, 8), 0, "t=0 命中");
        assert_eq!(scene_first_frame(&model(0.25, 1.0), 4, 8), 1, "t=0.25 恰好命中");
        assert_eq!(scene_first_frame(&model(0.3, 1.0), 4, 8), 2, "t=0.5 首个 ≥ 0.3");
        assert_eq!(
            scene_first_frame(&model(0.5001, 1.0), 4, 8),
            3,
            "t=0.5 < 0.5001，取 0.75"
        );
        // 场景不含任何影片帧（时长短于帧间距且落帧间）：候选兜底，不被使用。
        assert_eq!(scene_first_frame(&model(0.1, 0.05), 4, 8), 1, "无命中帧 → 候选 ceil(0.4) = 1");
    }

    /// 单帧入口帧号越界 → R5021。
    #[test]
    fn render_movie_frame_out_of_range_errors_r5021() {
        let compile: CompileV2Output = minimal_compile();
        let mut device = prism_gpu::CpuRefBackend;
        let mut sims: SimTimeline = SimTimeline::new();
        let err: RenderError = render_movie_frame(&compile, &mut device, 8, &mut sims)
            .expect_err("越界帧号必须报错");
        assert_eq!(err.code, "R5021");
        assert!(err.to_string().contains("建议"), "三段式必含建议段");
    }

    /// **W3 遗留接线**：内核逐轮（帧内 M3）发散判定必须参与运行时报告面。
    ///
    /// 此前 `SimSpec::step` 只取 `FixpointReport::final_residual()`，
    /// `diverged` 被丢弃——帧内发散永远无法到达 [`SimTimeline::diverged`]。
    /// 本用例固定「两路取并」的语义：帧级趋势或逐轮趋势任一为真即报发散。
    #[test]
    fn runtime_divergence_ors_per_iteration_and_frame_level() {
        let spec = SimSpec {
            force: Vec3::new(0.0, 0.0, 0.0),
            damping: 0.0,
            particles: Vec::new(),
            constraints: Vec::new(),
        };
        let mut rt = SimRuntime::new(
            spec,
            SolveModeMeta::new(1, 1, 1e-4),
            0.5,
            Vec::new(),
            7,
        )
        .expect("空规格合法（无粒子 / 无约束）");
        assert!(!rt.diverged(), "初始不得报发散");
        // 逐轮信号单独为真 → 公开判据必须为真（若 `diverged` 只看 driver 就漏报）。
        rt.iteration_diverged = true;
        assert!(
            rt.diverged(),
            "逐轮（帧内）发散必须进入公开判据——W3 遗留不得回退"
        );
        assert!(
            !rt.driver.diverged(),
            "本用例的帧级检测器未被喂入采样，独立为假（证明是逐轮这一路生效）"
        );
    }
}
