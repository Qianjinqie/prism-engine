//! 组合子库（V1.5-D）：高层语义 → 等价 v2 节点图展开。
//!
//! # 定位
//! 每个组合子是一件"视频修辞"（标题卡 /  lower third / 分屏 /  Ken Burns /
//! 形变转场），语义上等价于一段手写 v2 节点图。本模块提供：
//!
//! - **展开函数**（`expand_title_card` 等）：`CombinatorArgs` → [`NodeGraph`]，
//!   纯数据、纯函数、无 IO——展开产物与 `codegen_v2` 的 [`CompileV2Output`]
//!   同构（同一节点图 + 通道 + 场景时间线模型）；
//! - **注册面**：展开函数登记进 `codegen.rs` 的 [`crate::codegen::COMBINATOR_TABLE`]
//!   （name → 展开 fn），供后续波次的 DSL 糖化指令与 Rust 侧统一查表。
//!
//! # 纪律
//! - **DSL v2 语法本轮不加新指令**（parser 刚稳定不碰）：本模块只被 Rust 侧
//!   调用；糖化语法留后续波次；
//! - 展开只使用既有 v2 节点类型（S5 相机 / S8 材质 / S12 实体），**不发明
//!   新结构号**——组合子是语法糖，不是新引擎能力；
//! - 转场混合数学全部复用 `prism-core::sceneseq`（W1-C 下沉件），本模块不
//!   实现任何混合/插值内核，只产出声明数据；
//! - 所有错误走 [`CombinatorError`] 三段式（在哪 / 什么事 / 怎么改），
//!   绝不 panic、绝不静默兜底。

use std::collections::BTreeMap;

use prism_anim::{EasingKind, Keyframe, Track};
use prism_core::node::{ChannelRef, Composition, NodeBinding, ParamValue};
use prism_core::{Graph, Node, NodeId, Params, PropertyChannel, StructureId};
use prism_math::Vec3;

use crate::codegen::{CameraOrbit, CompileV2Output, LayerInfo, NamedChannel, SceneV2, TransitionV2};

// ---------------------------------------------------------------------------
// 入参 / 产物 / 错误
// ---------------------------------------------------------------------------

/// 组合子展开产物：等价 v2 节点图（可独立成片，也可作为子片被
/// `split_screen` / `morph_between` / `ken_burns` 再组合）。
///
/// 字段与 [`CompileV2Output`] 的节点图部分一一对应；`into_compile_output`
/// 补齐装包清单 / theme / 字体链的空缺，得到可直接交付 `render_movie` 的
/// 完整编译产物。
#[derive(Debug, Clone)]
pub struct NodeGraph {
    /// 影片标题。
    pub title: String,
    /// 画幅宽（px）。
    pub width: u32,
    /// 画幅高（px）。
    pub height: u32,
    /// 帧率（fps）。
    pub fps: u32,
    /// 总时长（秒）。
    pub duration: f32,
    /// 场景列表（时间线顺序）。
    pub scenes: Vec<SceneV2>,
    /// 过渡列表。
    pub transitions: Vec<TransitionV2>,
    /// 全文档节点图（prism-core）。
    pub graph: Graph,
    /// 命名通道登记表。
    pub channels: Vec<NamedChannel>,
}

impl NodeGraph {
    /// 组装为完整编译产物（组合子展开 → 可独立交付的影片）。
    ///
    /// 装包清单 / theme / 字体链留空：组合子是纯节点图糖，不携带装包与
    /// 美学声明（消费方按需在成品外层另行套用）。
    pub fn into_compile_output(self) -> CompileV2Output {
        CompileV2Output {
            title: self.title,
            width: self.width,
            height: self.height,
            fps: self.fps,
            duration: self.duration,
            format: crate::OutputFormat::Frames,
            scenes: self.scenes,
            transitions: self.transitions,
            graph: self.graph,
            channels: self.channels,
            packs: Vec::new(),
            theme: None,
            fonts: Vec::new(),
            // 组合子不声明公式（formula 块是文档级声明，展开结果不带公式）。
            formulas: Vec::new(),
        }
    }

    /// 借用形态的 [`Self::into_compile_output`]（不消耗自身）。
    pub fn to_compile_output(&self) -> CompileV2Output {
        self.clone().into_compile_output()
    }

    /// 图中已占用的最大节点 id（空图返回 0）。
    fn max_node_id(&self) -> NodeId {
        self.graph.nodes().map(|n| n.id).max().unwrap_or(0)
    }
}

/// 组合子入参：公共时间/画幅字段 + 名称键值参数 + 子片列表。
///
/// 设计为名称键值（而非每组合子一个强类型 struct）是为了让
/// [`crate::codegen::COMBINATOR_TABLE`] 的 `name → fn(&CombinatorArgs)` 签名
/// 统一；类型化构造入口在 `prism-author::combinators`（薄封装层）。
#[derive(Debug, Clone, Default)]
pub struct CombinatorArgs {
    /// 目标时长（秒；组合子可解释为成片时长或子片时长）。
    pub duration: f32,
    /// 帧率（fps）。
    pub fps: u32,
    /// 画幅宽（px）。
    pub width: u32,
    /// 画幅高（px）。
    pub height: u32,
    /// 组合子私有参数（`title` / `size` / `in_sec` / `kind` ……）。
    pub params: BTreeMap<String, ParamValue>,
    /// 子片列表（`split_screen` 取 `[左, 右]`；`morph_between` 取
    /// `[前, 后]`；`ken_burns` 取 `[底片]`）。
    pub sources: Vec<NodeGraph>,
}

impl CombinatorArgs {
    /// 读取 f32 参数（缺失 / 类型不符 → 缺省值；NaN/Inf → 缺省值）。
    pub fn f32_or(&self, name: &str, default: f32) -> f32 {
        match self.params.get(name) {
            Some(ParamValue::F32(v)) if v.is_finite() => *v,
            _ => default,
        }
    }

    /// 读取 u32 参数（缺失 / 类型不符 → 缺省值）。
    pub fn u32_or(&self, name: &str, default: u32) -> u32 {
        match self.params.get(name) {
            Some(ParamValue::U32(v)) => *v,
            _ => default,
        }
    }

    /// 读取字符串参数（缺失 / 类型不符 → 缺省值）。
    pub fn str_or(&self, name: &str, default: &str) -> String {
        match self.params.get(name) {
            Some(ParamValue::Str(s)) => s.clone(),
            _ => default.to_string(),
        }
    }

    /// 公共时长（缺失或非正 → 错误由调用方判定，这里只归一化）。
    pub fn duration_or(&self, default: f32) -> f32 {
        if self.duration.is_finite() && self.duration > 0.0 {
            self.duration
        } else {
            default
        }
    }

    /// 公共画幅（非正 → 缺省 480x270，与 DSL 缺省一致）。
    pub fn size_or_default(&self) -> (u32, u32) {
        let w: u32 = if self.width > 0 { self.width } else { 480 };
        let h: u32 = if self.height > 0 { self.height } else { 270 };
        (w, h)
    }

    /// 公共帧率（0 → 缺省 12，与 DSL 缺省一致）。
    pub fn fps_or_default(&self) -> u32 {
        if self.fps > 0 {
            self.fps
        } else {
            12
        }
    }
}

/// 组合子展开错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
#[derive(Debug, Clone, PartialEq)]
pub struct CombinatorError {
    /// 错误码（C5xxx 段：组合子）。
    pub code: &'static str,
    /// 三段式一【在哪】：组合子名 / 参数名。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl std::fmt::Display for CombinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for CombinatorError {}

/// 构造组合子错误的便捷函数。
fn c_err(code: &'static str, location: impl Into<String>, problem: impl Into<String>, fix: impl Into<String>) -> CombinatorError {
    CombinatorError {
        code,
        location: location.into(),
        problem: problem.into(),
        fix: fix.into(),
    }
}

/// 取子片（越界 / 空 → 三段式错误）。
fn source_at<'a>(
    args: &'a CombinatorArgs,
    idx: usize,
    who: &str,
    what: &str,
) -> Result<&'a NodeGraph, CombinatorError> {
    args.sources.get(idx).ok_or_else(|| {
        c_err(
            "C5001",
            format!("combinator '{who}' 的 sources[{idx}]"),
            format!("缺少{what}子片（sources 列表长度不足）"),
            format!("调用方应先构造{what}子片（NodeGraph）并放入 sources[{idx}]"),
        )
    })
}

// ---------------------------------------------------------------------------
// 构建小助手（id 分配 / 节点构造 / 通道构造 / 子片吸收）
// ---------------------------------------------------------------------------

/// 组合子图构建器：图 + id 分配 + 通道登记。
struct GraphBuilder {
    graph: Graph,
    next_id: NodeId,
    channels: Vec<NamedChannel>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            next_id: 1,
            channels: Vec::new(),
        }
    }

    /// 分配节点 id（递增，确定性）。
    fn alloc(&mut self) -> NodeId {
        let id: NodeId = self.next_id;
        self.next_id += 1;
        id
    }

    /// 加入节点（递增 id 不应重复；防御性错误转 panic-free 的 expect 语义）。
    fn add(&mut self, node: Node) {
        self.graph
            .add_node(node)
            .expect("组合子构建：递增 id 不应重复（内部不变量）");
    }

    /// 登记命名通道。
    fn channel(&mut self, name: impl Into<String>, channel: PropertyChannel) {
        self.channels.push(NamedChannel {
            name: name.into(),
            channel,
        });
    }

    /// 从最大子片 id 之后继续分配（吸收子片前调用，防 id 碰撞）。
    fn reserve_after(&mut self, min_next: NodeId) {
        if min_next > self.next_id {
            self.next_id = min_next;
        }
    }
}

/// 淡入淡出轨道（0 → 1 → 0）：`in_sec` 升 / `out_sec` 降，中点稳态 1。
///
/// 时长防呆：`in_sec` 钳到 `[0, duration/2]`、`out_sec` 钳到
/// `[0, duration − in_sec]`；零时长淡入淡出 = 硬起硬止（关键帧值直接取 1）。
fn fade_track(in_sec: f32, out_sec: f32, duration: f32) -> PropertyChannel {
    let d: f32 = if duration.is_finite() && duration > 0.0 { duration } else { 1.0 };
    let tin: f32 = if in_sec.is_finite() { in_sec.clamp(0.0, d * 0.5) } else { 0.0 };
    let tout: f32 = if out_sec.is_finite() { out_sec.clamp(0.0, d - tin) } else { 0.0 };
    let mut keys: Vec<Keyframe> = Vec::new();
    // 单调递增守卫：时间相同的关键帧只保留第一枚（确定性去重）。
    let push_key = |keys: &mut Vec<Keyframe>, t: f32, v: f32, e: EasingKind| {
        if keys.last().map_or(true, |k| t > k.time) {
            keys.push(Keyframe::new(t, v, e));
        }
    };
    push_key(&mut keys, 0.0, if tin <= 0.0 { 1.0 } else { 0.0 }, EasingKind::QuadOut);
    if tin > 0.0 {
        push_key(&mut keys, tin, 1.0, EasingKind::Linear);
    }
    if tout > 0.0 {
        push_key(&mut keys, d - tout, 1.0, EasingKind::Linear);
    }
    push_key(&mut keys, d, if tout <= 0.0 { 1.0 } else { 0.0 }, EasingKind::QuadIn);
    PropertyChannel::Track(Track::new(keys))
}

/// 文本图层节点（S12 实体表示）：content / 位置 / 字号 / 颜色 / 透明度 /
/// 可选对齐（`align` 为组合子扩展参数，DSL 文本层不产出，render 侧缺省 left）。
fn text_node(
    b: &mut GraphBuilder,
    name: &str,
    content: &str,
    x: f32,
    y: f32,
    size: f32,
    color: (f32, f32, f32),
    opacity: f32,
    align: &str,
    opacity_channel: Option<&str>,
) -> NodeId {
    let id: NodeId = b.alloc();
    let mut params: Params = Params::new();
    params.set("content", ParamValue::Str(content.to_string()));
    params.set("x", ParamValue::F32(x));
    params.set("y", ParamValue::F32(y));
    params.set("size", ParamValue::F32(size));
    params.set("color_r", ParamValue::F32(color.0));
    params.set("color_g", ParamValue::F32(color.1));
    params.set("color_b", ParamValue::F32(color.2));
    params.set("opacity", ParamValue::F32(opacity));
    params.set("align", ParamValue::Str(align.to_string()));
    let mut node: Node = Node::new(id, StructureId::EntityRepresentation, Composition::Feed);
    node.params = params;
    if let Some(ch) = opacity_channel {
        node.bindings.push(NodeBinding {
            param: "opacity".to_string(),
            channel: ChannelRef { name: ch.to_string() },
        });
    }
    b.add(node);
    let _ = name; // 图层名由场景摘要承载（节点本身不具名）
    id
}

/// SDF 球体节点（S12 实体表示）。
fn sphere_node(b: &mut GraphBuilder, radius: f32) -> NodeId {
    let id: NodeId = b.alloc();
    let mut params: Params = Params::new();
    params.set("kind", ParamValue::Str("sphere".to_string()));
    params.set("radius", ParamValue::F32(radius));
    let mut node: Node = Node::new(id, StructureId::EntityRepresentation, Composition::Feed);
    node.params = params;
    b.add(node);
    id
}

/// 材质节点（S8 辐射传输）。
fn material_node(b: &mut GraphBuilder, albedo: (f32, f32, f32), roughness: f32, metalness: f32) -> NodeId {
    let id: NodeId = b.alloc();
    let mut params: Params = Params::new();
    params.set("albedo_r", ParamValue::F32(albedo.0));
    params.set("albedo_g", ParamValue::F32(albedo.1));
    params.set("albedo_b", ParamValue::F32(albedo.2));
    params.set("roughness", ParamValue::F32(roughness));
    params.set("metalness", ParamValue::F32(metalness));
    let mut node: Node = Node::new(id, StructureId::RadiativeTransfer, Composition::Feed);
    node.params = params;
    b.add(node);
    id
}

/// 相机节点（S5 重参数化）：轨道中心 / 半径 / fov；`radius_channel` 有值时
/// 把 `orbit_radius` 绑到命名通道（Ken Burns 的缩放动画入口）。
fn camera_node(
    b: &mut GraphBuilder,
    center: Vec3,
    orbit_radius: f32,
    fov_rad: f32,
    radius_channel: Option<&str>,
) -> NodeId {
    let id: NodeId = b.alloc();
    let mut params: Params = Params::new();
    params.set("orbit_x", ParamValue::F32(center.x));
    params.set("orbit_y", ParamValue::F32(center.y));
    params.set("orbit_z", ParamValue::F32(center.z));
    params.set("orbit_radius", ParamValue::F32(orbit_radius));
    params.set("fov_rad", ParamValue::F32(fov_rad));
    let mut node: Node = Node::new(id, StructureId::Reparameterization, Composition::Feed);
    node.params = params;
    if let Some(ch) = radius_channel {
        node.bindings.push(NodeBinding {
            param: "orbit_radius".to_string(),
            channel: ChannelRef { name: ch.to_string() },
        });
    }
    b.add(node);
    id
}

/// 空场景摘要（单场景组合子的通用骨架）。
fn plain_scene(name: &str, duration: f32, node_ids: Vec<NodeId>, camera: Option<CameraOrbit>, layers: Vec<LayerInfo>) -> SceneV2 {
    SceneV2 {
        name: name.to_string(),
        start: 0.0,
        duration,
        node_ids,
        camera,
        layers,
        sim: None,
        post: crate::codegen::PostInfo::default(),
        audio: None,
        notes: Vec::new(),
    }
}

/// 把子片 `src` 吸收进 `dst`：节点 id 平移重编号，通道名与绑定引用加
/// `prefix` 前缀（防跨子片命名碰撞），场景名加前缀。
///
/// 返回旧 id → 新 id 映射（调用方极少需要；保留供未来组合子使用）。
fn absorb(dst: &mut GraphBuilder, src: &NodeGraph, prefix: &str) -> BTreeMap<NodeId, NodeId> {
    let offset: NodeId = dst.next_id.max(src.max_node_id() + 1);
    let mut map: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    for node in src.graph.nodes() {
        let mut cloned: Node = node.clone();
        let new_id: NodeId = node.id + offset;
        cloned.id = new_id;
        // 绑定引用的通道名同步加前缀（否则解析到不到前缀后的通道）。
        for binding in &mut cloned.bindings {
            binding.channel.name = format!("{prefix}{}", binding.channel.name);
        }
        dst.graph
            .add_node(cloned)
            .expect("组合子吸收：平移后 id 不应重复（内部不变量）");
        map.insert(node.id, new_id);
    }
    dst.reserve_after(src.max_node_id() + offset + 1);
    for ch in &src.channels {
        dst.channel(format!("{prefix}{}", ch.name), ch.channel.clone());
    }
    map
}

/// 场景的 id 重映射 + 名前缀（配合 [`absorb`] 使用）。
fn remap_scene(scene: &SceneV2, map: &BTreeMap<NodeId, NodeId>, prefix: &str) -> SceneV2 {
    let mut out: SceneV2 = scene.clone();
    out.name = format!("{prefix}{}", scene.name);
    out.node_ids = scene.node_ids.iter().map(|id| map.get(id).copied().unwrap_or(*id)).collect();
    for layer in &mut out.layers {
        layer.node_id = map.get(&layer.node_id).copied().unwrap_or(layer.node_id);
    }
    if let Some(sim) = &mut out.sim {
        sim.node_id = map.get(&sim.node_id).copied().unwrap_or(sim.node_id);
    }
    out
}

// ---------------------------------------------------------------------------
// 基片子片构造助手（组合子的常见输入）
// ---------------------------------------------------------------------------

/// 构造一个单球体 3D 场景子片（相机 + 材质 + sdf），作为 `split_screen` /
/// `morph_between` / `ken_burns` 的常见输入，也可直接成片。
///
/// # 参数
/// - `albedo`：球体反照率（0..1 线性 RGB）；
/// - `radius` / `orbit_radius`：球半径 / 相机轨道半径；
/// - `fov_deg`：视场角（度）；
/// - `duration` / `width` / `height` / `fps`：子片规格。
pub fn sphere_fragment(
    albedo: (f32, f32, f32),
    radius: f32,
    orbit_radius: f32,
    fov_deg: f32,
    duration: f32,
    width: u32,
    height: u32,
    fps: u32,
) -> NodeGraph {
    let mut b: GraphBuilder = GraphBuilder::new();
    let fov_rad: f32 = fov_deg.to_radians();
    let sphere: NodeId = sphere_node(&mut b, radius);
    let material: NodeId = material_node(&mut b, albedo, 0.4, 0.1);
    let camera: NodeId = camera_node(&mut b, Vec3::new(0.0, 0.0, 0.0), orbit_radius, fov_rad, None);
    let scene: SceneV2 = plain_scene(
        "sphere",
        duration,
        vec![sphere, material, camera],
        Some(CameraOrbit {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: orbit_radius,
            fov_rad,
        }),
        Vec::new(),
    );
    NodeGraph {
        title: "sphere".to_string(),
        width,
        height,
        fps,
        duration,
        scenes: vec![scene],
        transitions: Vec::new(),
        graph: b.graph,
        channels: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// 组合子一：title_card（标题卡）
// ---------------------------------------------------------------------------

/// 展开 `title_card`：居中大标题 + 淡入淡出（不依赖任何子片）。
///
/// 参数（`CombinatorArgs::params`）：
/// - `content`（Str，缺省 "untitled"）：标题文本；
/// - `size`（F32，缺省 `height × 0.14`）：字号（px）；
/// - `color_r/g/b`（F32 0..1，缺省 1/1/1）：文本色；
/// - `in_sec` / `out_sec`（F32，缺省 0.4 / 0.4）：淡入 / 淡出时长；
/// - `title`（Str，缺省 "title card"）：影片标题（产物元数据）。
pub fn expand_title_card(args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let (w, h): (u32, u32) = args.size_or_default();
    let fps: u32 = args.fps_or_default();
    let duration: f32 = args.duration_or(2.0);
    let wf: f32 = w as f32;
    let hf: f32 = h as f32;

    let content: String = args.str_or("content", "untitled");
    let size: f32 = args.f32_or("size", hf * 0.14);
    if !(size.is_finite() && size > 0.0) {
        return Err(c_err(
            "C5002",
            "combinator 'title_card' 的参数 'size'",
            format!("字号必须为正有限数，却得到 {size}"),
            "size 传正数（如 48）；缺省为画幅高的 14%",
        ));
    }
    let color: (f32, f32, f32) = (
        args.f32_or("color_r", 1.0),
        args.f32_or("color_g", 1.0),
        args.f32_or("color_b", 1.0),
    );
    let in_sec: f32 = args.f32_or("in_sec", 0.4);
    let out_sec: f32 = args.f32_or("out_sec", 0.4);
    let title: String = args.str_or("title", "title card");

    let mut b: GraphBuilder = GraphBuilder::new();
    // 淡入淡出通道：opacity 0→1→0（整段成片时长）。
    b.channel("title_card/opacity", fade_track(in_sec, out_sec, duration));
    let text_id: NodeId = text_node(
        &mut b,
        "title",
        &content,
        wf * 0.5,
        hf * 0.42,
        size,
        color,
        1.0,
        "center",
        Some("title_card/opacity"),
    );

    let scene: SceneV2 = plain_scene(
        "title",
        duration,
        vec![text_id],
        None,
        vec![LayerInfo {
            name: "title".to_string(),
            kind: "text".to_string(),
            node_id: text_id,
            morph: None,
        }],
    );

    Ok(NodeGraph {
        title,
        width: w,
        height: h,
        fps,
        duration,
        scenes: vec![scene],
        transitions: Vec::new(),
        graph: b.graph,
        channels: b.channels,
    })
}

// ---------------------------------------------------------------------------
// 组合子二：lower_third（左下三分之一条：主标题 + 副标题）
// ---------------------------------------------------------------------------

/// 展开 `lower_third`：左下方主标题 + 副标题两行文本，共用淡入淡出。
///
/// 参数：
/// - `title`（Str，缺省 "untitled"）：主标题；
/// - `subtitle`（Str，缺省 ""）：副标题（空串 = 不建副标题节点）；
/// - `size`（F32，缺省 `height × 0.085`）：主标题字号；副标题字号取
///   `sub_size`（缺省 `size × 0.5`）；
/// - `x` / `y`（F32，缺省 `width × 0.08` / `height × 0.70`）：文本块左上角；
/// - `color_r/g/b`（F32 0..1，缺省 1/0.96/0.90）：主标题色；
/// - `in_sec` / `out_sec`（F32，缺省 0.35 / 0.35）：淡入淡出。
pub fn expand_lower_third(args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let (w, h): (u32, u32) = args.size_or_default();
    let fps: u32 = args.fps_or_default();
    let duration: f32 = args.duration_or(2.5);
    let wf: f32 = w as f32;
    let hf: f32 = h as f32;

    let title: String = args.str_or("title", "untitled");
    let subtitle: String = args.str_or("subtitle", "");
    let size: f32 = args.f32_or("size", hf * 0.085);
    if !(size.is_finite() && size > 0.0) {
        return Err(c_err(
            "C5002",
            "combinator 'lower_third' 的参数 'size'",
            format!("主标题字号必须为正有限数，却得到 {size}"),
            "size 传正数（如 32）；缺省为画幅高的 8.5%",
        ));
    }
    let sub_size: f32 = args.f32_or("sub_size", size * 0.5);
    let x: f32 = args.f32_or("x", wf * 0.08);
    let y: f32 = args.f32_or("y", hf * 0.70);
    let color: (f32, f32, f32) = (
        args.f32_or("color_r", 1.0),
        args.f32_or("color_g", 0.96),
        args.f32_or("color_b", 0.90),
    );
    let in_sec: f32 = args.f32_or("in_sec", 0.35);
    let out_sec: f32 = args.f32_or("out_sec", 0.35);

    let mut b: GraphBuilder = GraphBuilder::new();
    b.channel("lower_third/opacity", fade_track(in_sec, out_sec, duration));

    let mut node_ids: Vec<NodeId> = Vec::new();
    let mut layers: Vec<LayerInfo> = Vec::new();
    let title_id: NodeId = text_node(
        &mut b,
        "headline",
        &title,
        x,
        y,
        size,
        color,
        1.0,
        "left",
        Some("lower_third/opacity"),
    );
    node_ids.push(title_id);
    layers.push(LayerInfo {
        name: "headline".to_string(),
        kind: "text".to_string(),
        node_id: title_id,
        morph: None,
    });
    if !subtitle.trim().is_empty() {
        let sub_id: NodeId = text_node(
            &mut b,
            "subhead",
            &subtitle,
            x,
            y + size * 1.5,
            sub_size,
            (color.0, color.1, color.2),
            1.0,
            "left",
            Some("lower_third/opacity"),
        );
        node_ids.push(sub_id);
        layers.push(LayerInfo {
            name: "subhead".to_string(),
            kind: "text".to_string(),
            node_id: sub_id,
            morph: None,
        });
    }

    let scene: SceneV2 = plain_scene("lower_third", duration, node_ids, None, layers);
    Ok(NodeGraph {
        title: format!("lower third: {title}"),
        width: w,
        height: h,
        fps,
        duration,
        scenes: vec![scene],
        transitions: Vec::new(),
        graph: b.graph,
        channels: b.channels,
    })
}

// ---------------------------------------------------------------------------
// 组合子三：split_screen（左右分屏）
// ---------------------------------------------------------------------------

/// 展开 `split_screen`：`sources[0]`（左）与 `sources[1]`（右）两个子片
/// **同程并行**（起止相同），render 侧按左右半幅合成。
///
/// 参数：
/// - `duration`（公共字段，缺省取左子片时长）：成片时长；
/// - `fps` / `width` / `height`（公共字段，缺省继承左子片）。
///
/// 语义注记：多场景子片按同程并行展示（分屏组合子的输入约定为单场景子片；
/// 多场景输入不报错，但时间语义为"全部场景同时播放"，由数据本身表达）。
pub fn expand_split_screen(args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let left: &NodeGraph = source_at(args, 0, "split_screen", "左侧")?;
    let right: &NodeGraph = source_at(args, 1, "split_screen", "右侧")?;

    let duration: f32 = args.duration_or(left.duration);
    if !(duration.is_finite() && duration > 0.0) {
        return Err(c_err(
            "C5003",
            "combinator 'split_screen' 的时长",
            format!("分屏时长必须为正有限数，却得到 {duration}"),
            "duration 传正数（如 2.0），或让左子片携带正时长",
        ));
    }
    let fps: u32 = if args.fps > 0 { args.fps } else { left.fps };
    let (w, h): (u32, u32) = if args.width > 0 && args.height > 0 {
        (args.width, args.height)
    } else {
        (left.width, left.height)
    };

    let mut b: GraphBuilder = GraphBuilder::new();
    // 左子片原样吸收（id 平移 + 通道名前缀 l_）。
    let map_l: BTreeMap<NodeId, NodeId> = absorb(&mut b, left, "l_");
    // 右子片吸收（前缀 r_）。
    let map_r: BTreeMap<NodeId, NodeId> = absorb(&mut b, right, "r_");

    let mut scenes: Vec<SceneV2> = Vec::new();
    for s in &left.scenes {
        let mut sc: SceneV2 = remap_scene(s, &map_l, "split_l/");
        sc.start = 0.0;
        sc.duration = duration;
        scenes.push(sc);
    }
    for s in &right.scenes {
        let mut sc: SceneV2 = remap_scene(s, &map_r, "split_r/");
        sc.start = 0.0;
        sc.duration = duration;
        scenes.push(sc);
    }
    if scenes.is_empty() {
        return Err(c_err(
            "C5004",
            "combinator 'split_screen' 的子片场景",
            "左右子片均无场景，分屏产物为空",
            "子片至少各含一个场景（如一个 stage3d 球体场景）",
        ));
    }

    Ok(NodeGraph {
        title: format!("split: {} | {}", left.title, right.title),
        width: w,
        height: h,
        fps,
        duration,
        scenes,
        transitions: Vec::new(),
        graph: b.graph,
        channels: b.channels,
    })
}

// ---------------------------------------------------------------------------
// 组合子四：ken_burns（Ken Burns：缓慢推拉 + 漂移）
// ---------------------------------------------------------------------------

/// 展开 `ken_burns`：对 `sources[0]`（底片）套用相机轨道动画——半径从
/// `from_radius` 缓动到 `to_radius`（推拉），中心三轴各从 `center_from_*`
/// 缓动到 `center_to_*`（摇移）。底片自带相机时被替换（组合子语义 = 接管相机）。
///
/// 参数：
/// - `from_radius` / `to_radius`（F32，缺省 8 / 4.5）：轨道半径起止；
/// - `center_from_x/y/z` / `center_to_x/y/z`（F32，缺省 0）：轨道中心起止；
/// - `fov_deg`（F32，缺省 45）：视场角（度；底片自带相机时沿用其 fov）；
/// - `easing`（Str，缺省 "quadInOut"）：缓动名（31 种，同 DSL easing 表）。
pub fn expand_ken_burns(args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let base: &NodeGraph = source_at(args, 0, "ken_burns", "底片")?;
    let duration: f32 = args.duration_or(base.duration);
    if !(duration.is_finite() && duration > 0.0) {
        return Err(c_err(
            "C5003",
            "combinator 'ken_burns' 的时长",
            format!("推拉时长必须为正有限数，却得到 {duration}"),
            "duration 传正数（如 3.0），或让底片携带正时长",
        ));
    }
    let fps: u32 = if args.fps > 0 { args.fps } else { base.fps };
    let (w, h): (u32, u32) = if args.width > 0 && args.height > 0 {
        (args.width, args.height)
    } else {
        (base.width, base.height)
    };

    let from_radius: f32 = args.f32_or("from_radius", 8.0);
    let to_radius: f32 = args.f32_or("to_radius", 4.5);
    if !(from_radius.is_finite() && from_radius > 0.0) {
        return Err(c_err(
            "C5002",
            "combinator 'ken_burns' 的参数 'from_radius'",
            format!("起始轨道半径必须为正有限数，却得到 {from_radius}"),
            "from_radius 传正数（如 8）",
        ));
    }
    if !(to_radius.is_finite() && to_radius > 0.0) {
        return Err(c_err(
            "C5002",
            "combinator 'ken_burns' 的参数 'to_radius'",
            format!("终止轨道半径必须为正有限数，却得到 {to_radius}"),
            "to_radius 传正数（如 4.5）",
        ));
    }
    let center_from: Vec3 = Vec3::new(
        args.f32_or("center_from_x", 0.0),
        args.f32_or("center_from_y", 0.0),
        args.f32_or("center_from_z", 0.0),
    );
    let center_to: Vec3 = Vec3::new(
        args.f32_or("center_to_x", 0.0),
        args.f32_or("center_to_y", 0.0),
        args.f32_or("center_to_z", 0.0),
    );
    // 底片 fov 沿用（有相机节点时）；否则 45°。
    let base_fov: f32 = base
        .scenes
        .iter()
        .filter_map(|s| s.camera.as_ref())
        .map(|c| c.fov_rad.to_degrees())
        .next()
        .unwrap_or(45.0);
    let fov_rad: f32 = args.f32_or("fov_deg", base_fov).to_radians();
    let easing_name: String = args.str_or("easing", "quadInOut");
    let easing: EasingKind = easing_kind(&easing_name).ok_or_else(|| {
        c_err(
            "C5005",
            format!("combinator 'ken_burns' 的参数 'easing'（值 '{easing_name}'）"),
            "未知的缓动名".to_string(),
            "linear / quadInOut / cubicOut / bounceInOut 等 31 种（同 DSL easing 表）",
        )
    })?;

    // ---- 重建图：剔除底片相机节点（S5），其余原样保留 --------------------
    let mut b: GraphBuilder = GraphBuilder::new();
    let mut kept_ids: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    for node in base.graph.nodes() {
        if matches!(node.structure, StructureId::Reparameterization) && node.params.get("orbit_radius").is_some() {
            continue; // 相机被 ken_burns 接管
        }
        b.add(node.clone());
        kept_ids.insert(node.id, node.id);
    }
    b.reserve_after(base.max_node_id() + 1);
    for ch in &base.channels {
        b.channel(ch.name.clone(), ch.channel.clone());
    }

    // ---- 新相机：半径与中心三轴各走一条起止轨道（推拉 + 摇移）-----------
    // 每条通道直接承载"参数现值"（绑定即写入），避免在 render 侧发明
    // 进度→取值的二次解释——通道语义 = 该参数的时间函数。
    let axis_track = |from: f32, to: f32| -> PropertyChannel {
        PropertyChannel::Track(Track::new(vec![
            Keyframe::new(0.0, from, easing),
            Keyframe::new(duration, to, easing),
        ]))
    };
    b.channel("ken_burns/radius", axis_track(from_radius, to_radius));
    b.channel("ken_burns/center_x", axis_track(center_from.x, center_to.x));
    b.channel("ken_burns/center_y", axis_track(center_from.y, center_to.y));
    b.channel("ken_burns/center_z", axis_track(center_from.z, center_to.z));

    let cam_id: NodeId = b.alloc();
    let mut params: Params = Params::new();
    params.set("orbit_x", ParamValue::F32(center_from.x));
    params.set("orbit_y", ParamValue::F32(center_from.y));
    params.set("orbit_z", ParamValue::F32(center_from.z));
    params.set("orbit_radius", ParamValue::F32(from_radius));
    params.set("fov_rad", ParamValue::F32(fov_rad));
    let mut cam_node: Node = Node::new(cam_id, StructureId::Reparameterization, Composition::Feed);
    cam_node.params = params;
    cam_node.bindings = vec![
        NodeBinding {
            param: "orbit_radius".to_string(),
            channel: ChannelRef { name: "ken_burns/radius".to_string() },
        },
        NodeBinding {
            param: "orbit_x".to_string(),
            channel: ChannelRef { name: "ken_burns/center_x".to_string() },
        },
        NodeBinding {
            param: "orbit_y".to_string(),
            channel: ChannelRef { name: "ken_burns/center_y".to_string() },
        },
        NodeBinding {
            param: "orbit_z".to_string(),
            channel: ChannelRef { name: "ken_burns/center_z".to_string() },
        },
    ];
    b.add(cam_node);

    // ---- 场景：剔除旧相机 id，接入新相机，时间线归一到 [0, duration] ------
    let mut scenes: Vec<SceneV2> = Vec::new();
    for s in &base.scenes {
        let mut sc: SceneV2 = s.clone();
        sc.start = 0.0;
        sc.duration = duration;
        sc.node_ids = sc
            .node_ids
            .iter()
            .filter(|id| kept_ids.contains_key(id))
            .copied()
            .collect();
        sc.node_ids.push(cam_id);
        sc.camera = Some(CameraOrbit {
            center: center_from,
            radius: from_radius,
            fov_rad,
        });
        scenes.push(sc);
    }
    if scenes.is_empty() {
        return Err(c_err(
            "C5004",
            "combinator 'ken_burns' 的底片场景",
            "底片无场景，推拉产物为空",
            "底片至少含一个场景（如一个 stage3d 球体场景）",
        ));
    }

    Ok(NodeGraph {
        title: format!("ken burns: {}", base.title),
        width: w,
        height: h,
        fps,
        duration,
        scenes,
        transitions: Vec::new(),
        graph: b.graph,
        channels: b.channels,
    })
}

/// 缓动名 → `EasingKind`（与 DSL `easing_from_name` 同表；缺失返回 None）。
fn easing_kind(name: &str) -> Option<EasingKind> {
    Some(match name {
        "linear" | "Linear" => EasingKind::Linear,
        "quadIn" | "easeInQuad" => EasingKind::QuadIn,
        "quadOut" | "easeOutQuad" => EasingKind::QuadOut,
        "quadInOut" | "easeInOutQuad" => EasingKind::QuadInOut,
        "cubicIn" | "easeInCubic" => EasingKind::CubicIn,
        "cubicOut" | "easeOutCubic" => EasingKind::CubicOut,
        "cubicInOut" | "easeInOutCubic" => EasingKind::CubicInOut,
        "quartIn" | "easeInQuart" => EasingKind::QuartIn,
        "quartOut" | "easeOutQuart" => EasingKind::QuartOut,
        "quartInOut" | "easeInOutQuart" => EasingKind::QuartInOut,
        "quintIn" | "easeInQuint" => EasingKind::QuintIn,
        "quintOut" | "easeOutQuint" => EasingKind::QuintOut,
        "quintInOut" | "easeInOutQuint" => EasingKind::QuintInOut,
        "sineIn" | "easeInSine" => EasingKind::SineIn,
        "sineOut" | "easeOutSine" => EasingKind::SineOut,
        "sineInOut" | "easeInOutSine" => EasingKind::SineInOut,
        "expoIn" | "easeInExpo" => EasingKind::ExpoIn,
        "expoOut" | "easeOutExpo" => EasingKind::ExpoOut,
        "expoInOut" | "easeInOutExpo" => EasingKind::ExpoInOut,
        "circIn" | "easeInCirc" => EasingKind::CircIn,
        "circOut" | "easeOutCirc" => EasingKind::CircOut,
        "circInOut" | "easeInOutCirc" => EasingKind::CircInOut,
        "backIn" | "easeInBack" => EasingKind::BackIn,
        "backOut" | "easeOutBack" => EasingKind::BackOut,
        "backInOut" | "easeInOutBack" => EasingKind::BackInOut,
        "elasticIn" | "easeInElastic" => EasingKind::ElasticIn,
        "elasticOut" | "easeOutElastic" => EasingKind::ElasticOut,
        "elasticInOut" | "easeInOutElastic" => EasingKind::ElasticInOut,
        "bounceIn" | "easeInBounce" => EasingKind::BounceIn,
        "bounceOut" | "easeOutBounce" => EasingKind::BounceOut,
        "bounceInOut" | "easeInOutBounce" => EasingKind::BounceInOut,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// 组合子五：morph_between（场景间形变转场）
// ---------------------------------------------------------------------------

/// 展开 `morph_between`：`sources[0]`（前片）与 `sources[1]`（后片）顺序
/// 拼接，重叠区按 `kind` 形变混合。混合数学（整数 wipe / dissolve / push /
/// mix）全部复用 `prism-core::sceneseq`（W1-C 下沉件），本模块只产出声明。
///
/// 参数：
/// - `kind`（Str，缺省 "crossfade"）：crossfade / fade / cut / black /
///   white / wipe / dissolve / push；
/// - `overlap`（F32，缺省 `min(前片时长, 后片时长) × 0.5`）：重叠时长；
/// - `band`（U32，缺省 2）：wipe 过渡带半宽（px）；
/// - `seed`（U32，缺省 0）：dissolve 显影种子；
/// - `dir`（Str，缺省 "left"）：wipe / push 方向（left / right / up / down）。
pub fn expand_morph_between(args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let from: &NodeGraph = source_at(args, 0, "morph_between", "前片")?;
    let to: &NodeGraph = source_at(args, 1, "morph_between", "后片")?;

    let fd: f32 = from.duration;
    let td: f32 = to.duration;
    if !(fd.is_finite() && fd > 0.0) {
        return Err(c_err(
            "C5003",
            "combinator 'morph_between' 的前片时长",
            format!("前片时长必须为正有限数，却得到 {fd}"),
            "前片子片携带正时长（duration > 0）",
        ));
    }
    if !(td.is_finite() && td > 0.0) {
        return Err(c_err(
            "C5003",
            "combinator 'morph_between' 的后片时长",
            format!("后片时长必须为正有限数，却得到 {td}"),
            "后片子片携带正时长（duration > 0）",
        ));
    }

    let kind: String = args.str_or("kind", "crossfade");
    if !matches!(
        kind.as_str(),
        "crossfade" | "fade" | "cut" | "black" | "white" | "wipe" | "dissolve" | "push"
    ) {
        return Err(c_err(
            "C5006",
            format!("combinator 'morph_between' 的参数 'kind'（值 '{kind}'）"),
            "未知的形变种类".to_string(),
            "crossfade / fade / cut / black / white / wipe / dissolve / push 八选一",
        ));
    }

    // 重叠时长：缺省 min(fd, td)/2；钳到 (0, min(fd, td)) 开区间内。
    let overlap_default: f32 = fd.min(td) * 0.5;
    let overlap: f32 = args.f32_or("overlap", overlap_default);
    let overlap: f32 = if overlap.is_finite() { overlap } else { overlap_default };
    let overlap: f32 = overlap.clamp(1e-4, fd.min(td) * 0.999);
    if !(overlap > 0.0) {
        return Err(c_err(
            "C5003",
            "combinator 'morph_between' 的重叠时长",
            format!("重叠时长必须为正，却得到 {overlap}"),
            "overlap 传正数且小于两侧片段时长（如 0.5）",
        ));
    }

    let dir: String = args.str_or("dir", "left");
    if !matches!(dir.as_str(), "left" | "right" | "up" | "down") {
        return Err(c_err(
            "C5006",
            format!("combinator 'morph_between' 的参数 'dir'（值 '{dir}'）"),
            "未知的方向".to_string(),
            "left / right / up / down 四选一（wipe / push 生效）",
        ));
    }

    let fps: u32 = if args.fps > 0 { args.fps } else { from.fps.max(to.fps) };
    let (w, h): (u32, u32) = if args.width > 0 && args.height > 0 {
        (args.width, args.height)
    } else {
        (from.width.max(to.width), from.height.max(to.height))
    };

    // ---- 吸收两片子片（前缀 mf_ / mt_），后片整体平移 (fd − overlap) ----
    let mut b: GraphBuilder = GraphBuilder::new();
    let map_f: BTreeMap<NodeId, NodeId> = absorb(&mut b, from, "mf_");
    let map_t: BTreeMap<NodeId, NodeId> = absorb(&mut b, to, "mt_");

    let mut scenes: Vec<SceneV2> = Vec::new();
    for s in &from.scenes {
        scenes.push(remap_scene(s, &map_f, "morph_from/"));
    }
    let shift: f32 = fd - overlap;
    for s in &to.scenes {
        let mut sc: SceneV2 = remap_scene(s, &map_t, "morph_to/");
        sc.start += shift;
        scenes.push(sc);
    }
    if scenes.is_empty() {
        return Err(c_err(
            "C5004",
            "combinator 'morph_between' 的子片场景",
            "前后子片均无场景，形变产物为空",
            "子片至少各含一个场景",
        ));
    }

    // 过渡挂接：前片末场景 → 后片首场景。
    let from_name: String = scenes
        .iter()
        .filter(|s| s.name.starts_with("morph_from/"))
        .last()
        .map(|s| s.name.clone())
        .ok_or_else(|| {
            c_err(
                "C5004",
                "combinator 'morph_between' 的前片场景",
                "前片无场景可挂接过渡",
                "前片子片至少含一个场景",
            )
        })?;
    let to_name: String = scenes
        .iter()
        .find(|s| s.name.starts_with("morph_to/"))
        .map(|s| s.name.clone())
        .ok_or_else(|| {
            c_err(
                "C5004",
                "combinator 'morph_between' 的后片场景",
                "后片无场景可挂接过渡",
                "后片子片至少含一个场景",
            )
        })?;

    let transition: TransitionV2 = TransitionV2 {
        name: "morph".to_string(),
        from: from_name,
        to: to_name,
        start: shift,
        duration: overlap,
        kind,
        band: args.u32_or("band", 2) as u16,
        seed: args.u32_or("seed", 0) as u64,
        dir: Some(dir),
    };

    let total: f32 = fd + td - overlap;
    Ok(NodeGraph {
        title: format!("morph: {} -> {}", from.title, to.title),
        width: w,
        height: h,
        fps,
        duration: total,
        scenes,
        transitions: vec![transition],
        graph: b.graph,
        channels: b.channels,
    })
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用最小球体场景子片（薄封装公共 `sphere_fragment`）。
    fn sphere_fragment(albedo: (f32, f32, f32), duration: f32) -> NodeGraph {
        super::sphere_fragment(albedo, 1.0, 8.0, 45.0, duration, 320, 180, 12)
    }

    fn args_with(pairs: &[(&str, ParamValue)]) -> CombinatorArgs {
        let mut params: BTreeMap<String, ParamValue> = BTreeMap::new();
        for (k, v) in pairs {
            params.insert((*k).to_string(), v.clone());
        }
        CombinatorArgs {
            duration: 0.0,
            fps: 0,
            width: 0,
            height: 0,
            params,
            sources: Vec::new(),
        }
    }

    #[test]
    fn title_card_expands_to_single_text_scene() {
        let mut args: CombinatorArgs = args_with(&[
            ("content", ParamValue::Str("你好，Prism".to_string())),
            ("in_sec", ParamValue::F32(0.3)),
            ("out_sec", ParamValue::F32(0.3)),
        ]);
        args.duration = 2.0;
        args.fps = 12;
        args.width = 320;
        args.height = 180;
        let g: NodeGraph = expand_title_card(&args).expect("标题卡应展开成功");
        assert_eq!(g.scenes.len(), 1);
        assert_eq!(g.scenes[0].layers.len(), 1);
        assert_eq!(g.scenes[0].layers[0].kind, "text");
        assert_eq!(g.channels.len(), 1);
        assert_eq!(g.channels[0].name, "title_card/opacity");
        // 淡入淡出通道在 t=0 / t=duration 处为 0、中段为 1。
        let ch: &PropertyChannel = &g.channels[0].channel;
        assert!(ch.sample(0.0).abs() < 1e-6, "首帧应全透明");
        assert!(ch.sample(1.0).abs() - 1.0 < 1e-6, "中段应不透明");
        assert!(ch.sample(g.duration).abs() < 1e-6, "末帧应全透明");
        // 组装为编译产物后字段齐全。
        let out: CompileV2Output = g.into_compile_output();
        assert_eq!(out.width, 320);
        assert_eq!(out.fps, 12);
        assert!((out.duration - 2.0).abs() < 1e-6);
    }

    #[test]
    fn title_card_rejects_non_positive_size() {
        let mut args: CombinatorArgs = args_with(&[("size", ParamValue::F32(0.0))]);
        args.duration = 1.0;
        let err: CombinatorError = expand_title_card(&args).expect_err("字号 0 应被拒绝");
        assert_eq!(err.code, "C5002");
        assert!(err.to_string().contains("建议"), "错误必含建议段");
    }

    #[test]
    fn lower_third_builds_two_text_layers() {
        let mut args: CombinatorArgs = args_with(&[
            ("title", ParamValue::Str("主讲人".to_string())),
            ("subtitle", ParamValue::Str("工程师".to_string())),
        ]);
        args.duration = 2.5;
        args.width = 320;
        args.height = 180;
        let g: NodeGraph = expand_lower_third(&args).expect("lower third 应展开成功");
        assert_eq!(g.scenes[0].layers.len(), 2, "主副标题应各一个文本节点");
        assert_eq!(g.scenes[0].layers[0].name, "headline");
        assert_eq!(g.scenes[0].layers[1].name, "subhead");
    }

    #[test]
    fn lower_third_skips_empty_subtitle() {
        let mut args: CombinatorArgs = args_with(&[
            ("title", ParamValue::Str("只有主标题".to_string())),
            ("subtitle", ParamValue::Str("   ".to_string())),
        ]);
        args.duration = 2.0;
        let g: NodeGraph = expand_lower_third(&args).expect("空副标题应被跳过而非报错");
        assert_eq!(g.scenes[0].layers.len(), 1);
    }

    #[test]
    fn split_screen_merges_two_sources_same_interval() {
        let left: NodeGraph = sphere_fragment((0.9, 0.2, 0.2), 2.0);
        let right: NodeGraph = sphere_fragment((0.2, 0.4, 0.9), 2.0);
        let mut args: CombinatorArgs = CombinatorArgs {
            duration: 2.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![left, right],
        };
        args.duration = 2.0;
        let g: NodeGraph = expand_split_screen(&args).expect("分屏应展开成功");
        assert_eq!(g.scenes.len(), 2, "左右子片各贡献场景");
        for s in &g.scenes {
            assert!((s.start - 0.0).abs() < 1e-6, "分屏场景应同程起点 0");
            assert!((s.duration - 2.0).abs() < 1e-6, "分屏场景应同程时长");
        }
        // 节点 id 不碰撞（左 3 + 右 3 = 6）。
        assert_eq!(g.graph.len(), 6);
        // 通道名加前缀（本例无通道，仅验证不破坏空表）。
        assert!(g.channels.is_empty());
    }

    #[test]
    fn split_screen_requires_two_sources() {
        let args: CombinatorArgs = CombinatorArgs {
            duration: 1.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![sphere_fragment((1.0, 0.0, 0.0), 1.0)],
        };
        let err: CombinatorError = expand_split_screen(&args).expect_err("缺右侧子片应报错");
        assert_eq!(err.code, "C5001");
    }

    #[test]
    fn ken_burns_replaces_camera_and_binds_radius_channel() {
        let base: NodeGraph = sphere_fragment((0.8, 0.3, 0.3), 3.0);
        let mut args: CombinatorArgs = CombinatorArgs {
            duration: 3.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![base],
        };
        args.params.insert("from_radius".to_string(), ParamValue::F32(9.0));
        args.params.insert("to_radius".to_string(), ParamValue::F32(4.0));
        let g: NodeGraph = expand_ken_burns(&args).expect("Ken Burns 应展开成功");
        assert_eq!(g.scenes.len(), 1);
        // 旧相机被剔除：场景应只含 1 个相机（新）。
        let cam_count: usize = g
            .graph
            .nodes()
            .filter(|n| matches!(n.structure, StructureId::Reparameterization))
            .count();
        assert_eq!(cam_count, 1, "底片相机应被替换为唯一新相机");
        // 新相机半径绑定到 ken_burns/radius 通道。
        let cam = g
            .graph
            .nodes()
            .find(|n| matches!(n.structure, StructureId::Reparameterization))
            .expect("应存在相机节点");
        assert!(
            cam.bindings.iter().any(|b| b.channel.name == "ken_burns/radius"),
            "半径应绑定推拉通道"
        );
        // 通道在首末帧的值 = 起止半径。
        let ch: &PropertyChannel = g
            .channels
            .iter()
            .find(|c| c.name == "ken_burns/radius")
            .map(|c| &c.channel)
            .expect("应登记半径通道");
        assert!((ch.sample(0.0) - 9.0).abs() < 1e-6);
        assert!((ch.sample(3.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn ken_burns_rejects_unknown_easing() {
        let mut args: CombinatorArgs = CombinatorArgs {
            duration: 2.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![sphere_fragment((1.0, 1.0, 1.0), 2.0)],
        };
        args.params.insert("easing".to_string(), ParamValue::Str("warp".to_string()));
        let err: CombinatorError = expand_ken_burns(&args).expect_err("未知缓动应报错");
        assert_eq!(err.code, "C5005");
    }

    #[test]
    fn morph_between_builds_overlap_transition() {
        let from: NodeGraph = sphere_fragment((0.9, 0.2, 0.2), 2.0);
        let to: NodeGraph = sphere_fragment((0.2, 0.4, 0.9), 2.0);
        let mut args: CombinatorArgs = CombinatorArgs {
            duration: 0.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![from, to],
        };
        args.params.insert("kind".to_string(), ParamValue::Str("wipe".to_string()));
        args.params.insert("overlap".to_string(), ParamValue::F32(0.5));
        args.params.insert("dir".to_string(), ParamValue::Str("right".to_string()));
        let g: NodeGraph = expand_morph_between(&args).expect("形变应展开成功");
        assert_eq!(g.transitions.len(), 1);
        let t: &TransitionV2 = &g.transitions[0];
        assert_eq!(t.kind, "wipe");
        assert_eq!(t.dir.as_deref(), Some("right"));
        assert!((t.duration - 0.5).abs() < 1e-6, "重叠时长应透传");
        assert!((t.start - 1.5).abs() < 1e-6, "过渡起点 = 前片时长 − 重叠");
        assert!((g.duration - 3.5).abs() < 1e-6, "总时长 = 2 + 2 − 0.5");
        // 后片场景应平移到重叠起点。
        let to_scene = g.scenes.iter().find(|s| s.name.starts_with("morph_to/")).expect("后片场景应在");
        assert!((to_scene.start - 1.5).abs() < 1e-6);
    }

    #[test]
    fn morph_between_rejects_unknown_kind() {
        let mut args: CombinatorArgs = CombinatorArgs {
            duration: 0.0,
            fps: 12,
            width: 320,
            height: 180,
            params: BTreeMap::new(),
            sources: vec![
                sphere_fragment((1.0, 0.0, 0.0), 1.0),
                sphere_fragment((0.0, 1.0, 0.0), 1.0),
            ],
        };
        args.params.insert("kind".to_string(), ParamValue::Str("spin".to_string()));
        let err: CombinatorError = expand_morph_between(&args).expect_err("未知种类应报错");
        assert_eq!(err.code, "C5006");
    }

    #[test]
    fn combinator_table_registers_all_five() {
        let names: Vec<&str> = crate::codegen::COMBINATOR_TABLE.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec!["title_card", "lower_third", "split_screen", "ken_burns", "morph_between"]
        );
    }
}
