//! codegen：AST → 引擎产物。
//!
//! v1（M0/M1）路径逐字保留：`codegen(&DocumentAst) -> CompileOutput`；
//! v2 路径：`codegen_v2(&DocumentV2Ast) -> CompileV2Output`，落点为
//! prism-core 节点图（`Graph` + `Node.params` + `Node.bindings`）与属性通道
//! （`PropertyChannel`），深语义（渲染 / 模拟步进 / 音频）留 Phase 2/3。
//!
//! # v2 语义决策（文档化）
//! - **节点映射**（友好名 → 结构表）：layer text → S12 实体表示；
//!   layer particles → S4 确定性抽样；sim → S3 约束梯度流；stage3d sdf →
//!   S12；stage3d material → S8 辐射传输；stage3d camera → S5 重参数化；
//!   compose → S2 卷积（Merge(op) 汇点）。
//! - **compose op 映射**：add→Merge(Plus)、multiply→Merge(Multiply)、
//!   over→Over、screen→Screen、mask→Mask（Over/Screen/Mask 在标量域 eval
//!   时由 prism-core 如实报 P3012，DSL 层不静默禁用）。
//! - **单位归一化**（C17）：时间 ms/1000、帧/fps；角度 deg→rad；百分比 /100；
//!   px / 裸数字原值透传（归一化标量豁免）。
//! - **时间线**：scene[i+1].start = scene[i].start + scene[i].duration −
//!   过渡重叠；总时长 = timeline.duration → movie.duration → 末场景结束。
//! - **fps 次序**：timeline.fps → movie.fps → 缺省 12（与 v1 缺省一致）。
//! - **环检测**：compose 引用图 DFS，三色标记，环路径带每个块的**定义行号**
//!   （验收点名项）；错误码与 prism-core 图错误码对齐（P3005 环 / P3006 悬空）。

use std::collections::BTreeMap;

use prism_anim::{EasingKind, Expr, Keyframe, LoopMode, Track};
use prism_core::{Graph, Node, NodeId, Params, PropertyChannel, StructureId};
use prism_core::node::{ChannelRef, Composition, MergeOp, NodeBinding, ParamValue};
use prism_math::Vec3;

use crate::ast::{
    AngleUnit, AudioAst, CameraV2Ast, ComposeAst, ComposeRef, DocumentAst, DocumentV2Ast,
    FieldValue, FormulaAst, LayerAst, LightV2Ast, MaterialV2Ast, NamedField, NoteAst,
    OutputFormatAst, PostAst, SceneItem, SdfAst, SimAst, Stage3dAst, TimeVal, TimeUnit,
    TransitionAst,
    AngleVal,
};
use crate::{DslError, err_at};
use crate::packs::PackManifest;
use crate::OutputFormat;

// ---------------------------------------------------------------------------
// 缺省值（v1 原样保留 + v2 新增）
// ---------------------------------------------------------------------------

// 缺省值（刻意与黄金示例的"电影感"参数保持一致的风格）。
const DEFAULT_RADIUS: f32 = 1.0;
const DEFAULT_ORBIT_RADIUS: f32 = 8.0;
const DEFAULT_FOV_DEG: f32 = 45.0;
const DEFAULT_ROUGHNESS: f32 = 0.5;
const DEFAULT_METALNESS: f32 = 0.0;
const DEFAULT_LIGHT_INTENSITY: f32 = 1.0;
const DEFAULT_CONTRAST: f32 = 1.0;
const DEFAULT_DURATION: f32 = 3.0;
const DEFAULT_FPS: u32 = 12;
const DEFAULT_WIDTH: u32 = 480;
const DEFAULT_HEIGHT: u32 = 270;
// v2 新增缺省。
/// 单场景无时长信息时的缺省时长。
const DEFAULT_SCENE_DURATION: f32 = 2.0;
/// transition.band 缺省值（px；与 `prism_core::sceneseq::WIPE_BAND` 同值）。
///
/// 不从 core 引入常量以保持本 crate 的「产物层零额外耦合」，两处注释互指。
pub const DEFAULT_WIPE_BAND: u16 = 2;

/// transition 无 duration 时的缺省时长。
const DEFAULT_TRANSITION_DURATION: f32 = 0.5;
/// layer text 缺省字号（px）。
const DEFAULT_TEXT_SIZE: f32 = 24.0;
/// layer particles 缺省粒子数。
const DEFAULT_PARTICLE_COUNT: u32 = 100;
/// layer particles 缺省速度。
const DEFAULT_PARTICLE_SPEED: f32 = 0.5;
/// sim 缺省网格边长。
const DEFAULT_SIM_GRID: u32 = 16;
/// sim 缺省刚度。
const DEFAULT_SIM_STIFFNESS: f32 = 0.9;
/// sim 缺省阻尼。
const DEFAULT_SIM_DAMPING: f32 = 0.02;
/// sim force 缺省强度。
const DEFAULT_FORCE_STRENGTH: f32 = 1.0;
/// sim noise 缺省尺度。
const DEFAULT_NOISE_SCALE: f32 = 0.1;
/// audio music 缺省 bpm / gain / duck（duck 单位 dB）。
const DEFAULT_BPM: f32 = 120.0;
const DEFAULT_GAIN: f32 = 0.8;
const DEFAULT_DUCK_DB: f32 = 12.0;

/// `#RRGGBB` → 各分量 0..1 的 Vec3。
fn hex_to_vec3(c: (u8, u8, u8)) -> Vec3 {
    Vec3::new(c.0 as f32 / 255.0, c.1 as f32 / 255.0, c.2 as f32 / 255.0)
}

// ---------------------------------------------------------------------------
// v1 codegen（逐字保留；format 经 OutputFormatAst 转换）
// ---------------------------------------------------------------------------

/// 把 v1 AST 映射为引擎产物。至此语法 / 语义已全部校验通过，本函数不会失败。
pub(crate) fn codegen(doc: &DocumentAst) -> crate::CompileOutput {
    // ---- sphere / material（缺省：半径 1，白色中性材质） ----
    let sphere: Option<&crate::ast::SphereAst> = doc.scene.sphere.as_ref();
    let radius: f32 = sphere.map_or(DEFAULT_RADIUS, |s| s.radius);
    let material: prism_core::Material = match sphere.and_then(|s| s.material.as_ref()) {
        Some(m) => prism_core::Material::new(
            m.albedo.map_or(Vec3::new(1.0, 1.0, 1.0), hex_to_vec3),
            m.roughness.unwrap_or(DEFAULT_ROUGHNESS),
            m.metalness.unwrap_or(DEFAULT_METALNESS),
        ),
        // 缺省材质：纯白 albedo / 中等粗糙 / 非金属。
        None => prism_core::Material::new(Vec3::new(1.0, 1.0, 1.0), DEFAULT_ROUGHNESS, DEFAULT_METALNESS),
    };

    // ---- camera / orbit（缺省：中心原点、轨道半径 8、fov 45°） ----
    let cam: Option<&crate::ast::CameraAst> = doc.scene.camera.as_ref();
    let orbit_center: Vec3 =
        cam.and_then(|c| c.orbit_center).unwrap_or_else(|| Vec3::new(0.0, 0.0, 0.0));
    let orbit_radius: f32 = cam.and_then(|c| c.orbit_radius).unwrap_or(DEFAULT_ORBIT_RADIUS);
    let fov_deg: f32 = cam.map_or(DEFAULT_FOV_DEG, |c| c.fov_deg);
    // eye 为占位值（t=0 时刻的轨道位置）；运行时由 prism_core::eval_camera_orbit 按时间推导。
    let eye: Vec3 = Vec3::new(0.0, 0.0, -orbit_radius);
    let camera: prism_core::Camera = prism_core::Camera::new(eye, orbit_center, fov_deg.to_radians());

    // ---- lights（逐盏透传 direction，不取模、不归一化） ----
    let lights: Vec<prism_core::Light> = doc
        .scene
        .lights
        .iter()
        .map(|l| {
            prism_core::Light::new(
                l.direction,
                l.color.map_or(Vec3::new(1.0, 1.0, 1.0), hex_to_vec3),
                l.intensity.unwrap_or(DEFAULT_LIGHT_INTENSITY),
            )
        })
        .collect();

    // ---- aesthetic ----
    let aesthetic: prism_core::Aesthetic = match doc.aesthetic.as_ref() {
        Some(a) => {
            prism_core::Aesthetic::new(a.teal_orange, a.contrast.unwrap_or(DEFAULT_CONTRAST), a.filmic)
        }
        None => prism_core::Aesthetic::new(false, DEFAULT_CONTRAST, false),
    };

    // ---- timeline / output ----
    let timeline: Option<&crate::ast::TimelineAst> = doc.timeline.as_ref();
    let duration: f32 = timeline.and_then(|t| t.duration).unwrap_or(DEFAULT_DURATION);
    let fps: u32 = timeline.and_then(|t| t.fps).unwrap_or(DEFAULT_FPS);
    let output: Option<&crate::ast::OutputAst> = doc.output.as_ref();
    let (width, height): (u32, u32) =
        output.and_then(|o| o.resolution).unwrap_or((DEFAULT_WIDTH, DEFAULT_HEIGHT));
    let format: OutputFormat = output
        .and_then(|o| o.format)
        .map_or(OutputFormat::Frames, |f| match f {
            OutputFormatAst::Frames => OutputFormat::Frames,
            OutputFormatAst::Png => OutputFormat::Png,
        });

    let scene: prism_core::Scene = prism_core::Scene {
        radius,
        orbit_center,
        orbit_radius,
        material,
        lights,
        camera,
        aesthetic,
    };

    crate::CompileOutput { scene, width, height, fps, duration, format }
}

// ---------------------------------------------------------------------------
// v2 产物类型
// ---------------------------------------------------------------------------

/// 场景相机轨道摘要（v2）。
#[derive(Debug, Clone, PartialEq)]
pub struct CameraOrbit {
    /// 轨道中心。
    pub center: Vec3,
    /// 轨道半径。
    pub radius: f32,
    /// 视场角（弧度；由 fov 字段按单位归一化）。
    pub fov_rad: f32,
}

/// 图层摘要（v2）：节点图之外的命名索引。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerInfo {
    /// 图层名（缺省同种类名）。
    pub name: String,
    /// 图层种类（text / particles / …）。
    pub kind: String,
    /// 对应节点 id。
    pub node_id: NodeId,
    /// v1.5-C：`morph from "路径A" to "路径B"` 声明（None = 无变形）。
    pub morph: Option<(String, String)>,
}

/// 模拟摘要（v2）。
#[derive(Debug, Clone, PartialEq)]
pub struct SimInfo {
    /// 模拟种类（cloth / fluid / …）。
    pub kind: String,
    /// 模拟名。
    pub name: String,
    /// 对应节点 id。
    pub node_id: NodeId,
}

/// post 处理参数摘要（v2）：字段名 → 值（BTreeMap 固定序）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostInfo {
    pub params: BTreeMap<String, f32>,
}

/// audio 参数摘要（v2）：music 子块的 bpm / gain / duck 平铺。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioInfo {
    pub music: BTreeMap<String, f32>,
}

/// note 块摘要（作曲系统）：块名 + 音阶 / 根音 / 速度 / 小节 / 种子 / 琶音开关。
///
/// 字段值经 codegen 白名单校验并回填缺省（缺省走情绪无关的通用作曲缺省）。
#[derive(Debug, Clone, PartialEq)]
pub struct NoteInfo {
    /// 块名（如 "主题"）。
    pub name: String,
    /// 音阶类型标识（major / minor / pentatonic / dorian / phrygian / lydian /
    /// mixolydian / minor-pentatonic）。
    pub scale: String,
    /// 根音 MIDI 音高。
    pub root: i32,
    /// 速度（BPM）。
    pub bpm: f32,
    /// 小节数。
    pub bars: u32,
    /// 派生种子。
    pub seed: u64,
    /// 是否启用琶音织体。
    pub arpeggio: bool,
}

/// note 块缺省：C 大调、A4=440 附近的根音、120 BPM、4 小节。
const NOTE_DEFAULT_SCALE: &str = "major";
const NOTE_DEFAULT_ROOT: i32 = 60;
const NOTE_DEFAULT_BPM: f32 = 120.0;
const NOTE_DEFAULT_BARS: u32 = 4;
const NOTE_DEFAULT_SEED: u64 = 0;

/// note 块允许的音阶标识白名单。
const NOTE_SCALES: [&str; 8] = [
    "major",
    "minor",
    "pentatonic",
    "minor-pentatonic",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
];

/// theme 块美学参数摘要（v2；字段集与 v1 aesthetic 相同，见）。
#[derive(Debug, Clone, PartialEq)]
pub struct ThemeInfo {
    /// teal-orange 调色开关。
    pub teal_orange: bool,
    /// 对比度（缺省 1.0，与 v1 aesthetic 一致）。
    pub contrast: f32,
    /// filmic 色调映射开关。
    pub filmic: bool,
}

/// v2 场景产物：时间线位置 + 节点 id 区间 + 各条目摘要。
#[derive(Debug, Clone, PartialEq)]
pub struct SceneV2 {
    /// 场景名。
    pub name: String,
    /// 时间线起点（秒；含过渡重叠折算）。
    pub start: f32,
    /// 场景时长（秒）。
    pub duration: f32,
    /// 本场景产出的全部节点 id（升序）。
    pub node_ids: Vec<NodeId>,
    /// 相机轨道（stage3d.camera；None = 未声明）。
    pub camera: Option<CameraOrbit>,
    /// 图层列表。
    pub layers: Vec<LayerInfo>,
    /// 模拟（至多一个；None = 未声明）。
    pub sim: Option<SimInfo>,
    /// post 参数。
    pub post: PostInfo,
    /// audio 参数。
    pub audio: Option<AudioInfo>,
    /// note 块列表（作曲系统；可多个）。
    pub notes: Vec<NoteInfo>,
}

/// v2 过渡产物：重叠区间（[start, start+duration) 与后一场景重叠）。
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionV2 {
    /// 过渡名。
    pub name: String,
    /// 起始场景名。
    pub from: String,
    /// 结束场景名。
    pub to: String,
    /// 重叠区间起点（= 后一场景的 start）。
    pub start: f32,
    /// 重叠时长（秒）。
    pub duration: f32,
    /// 过渡种类（crossfade / cut / black / white，v1.5-C 追加引擎级
    /// fade / wipe / dissolve / push）。
    pub kind: String,
    /// v1.5-C：擦除过渡带半宽（px；`kind wipe` 生效，缺省 2）。
    pub band: u16,
    /// v1.5-C：溶解显影种子（`kind dissolve` 生效，缺省 0）。
    pub seed: u64,
    /// v1.5-C：方向（`kind wipe` / `push` 生效；缺省 None = 渲染侧缺省方向）。
    pub dir: Option<String>,
}

/// 命名通道（anim / expr 块与字段级关键帧的登记产物）。
#[derive(Debug, Clone, PartialEq)]
pub struct NamedChannel {
    /// 通道名（字段级：`<scene>/<layer>.<field>`；顶层：定义名）。
    pub name: String,
    /// 通道本体。
    pub channel: PropertyChannel,
}

/// v2 一次成功编译的完整产物。
#[derive(Debug, Clone)]
pub struct CompileV2Output {
    /// 影片标题（movie 块；缺省 "untitled"）。
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// 总时长（timeline.duration → movie.duration → 末场景结束）。
    pub duration: f32,
    pub format: OutputFormat,
    /// 场景列表（时间线顺序）。
    pub scenes: Vec<SceneV2>,
    /// 过渡列表（按相邻场景对顺序）。
    pub transitions: Vec<TransitionV2>,
    /// 全文档节点图（prism-core）。
    pub graph: Graph,
    /// 命名通道登记表（anim / expr / 字段级关键帧）。
    pub channels: Vec<NamedChannel>,
    /// 装包清单（engine 范围已校验）。
    pub packs: Vec<PackManifest>,
    /// theme 块美学参数（None = 文档未声明 theme）。
    pub theme: Option<ThemeInfo>,
    /// 字体回退链构造序列（V1.5-B）：push 顺序 = DSL 声明顺序 = 回退顺序。
    /// 空 = 文档未声明 font 指令（消费方按需自行补内嵌字体兜底）。
    pub fonts: Vec<FontSlotSpec>,
    /// 公式节点列表（V2-j `formula` 块；按声明顺序）。
    pub formulas: Vec<FormulaNode>,
}

/// `FontSet` 构造序列的一项（V1.5-B）。
///
/// codegen 不直接持有 `prism-text::FontSet`（DSL 层不依赖文本管线，保持
/// 编译期产物纯数据）：只产出**有序构造序列**，消费方按 `index` 升序逐项
/// `FontSet::push_file(spec.path)` 即得与 DSL 声明完全一致的回退链。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontSlotSpec {
    /// 槽位标签（DSL 声明的路径原文；对接 `prism_text::FontSlot::label`）。
    pub label: String,
    /// 装载入口实参（`FontSet::push_file` 的路径字符串）。
    pub path: String,
    /// 回退链槽位号（0 = 首选；n>0 即第 n 个回退项，等于 `push_file` 返回值）。
    pub index: usize,
    /// 该槽位的声明行号（装载失败诊断定位用）。
    pub line: usize,
}

/// 公式对齐方式（`formula` 块的 `align` 字段；渲染侧据此把整条公式居中 / 贴边）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FormulaAlign {
    /// 左对齐。
    Left,
    /// 居中（缺省：公式通常以锚点为中心摆放）。
    #[default]
    Center,
    /// 右对齐。
    Right,
}

impl FormulaAlign {
    /// `align` 字段字面 → 对齐方式（表外 → `None`）。
    pub fn parse(text: &str) -> Option<FormulaAlign> {
        match text {
            "left" => Some(FormulaAlign::Left),
            "center" => Some(FormulaAlign::Center),
            "right" => Some(FormulaAlign::Right),
            _ => None,
        }
    }

    /// 对齐方式的 DSL 字面（诊断 / 回写用）。
    pub fn as_str(self) -> &'static str {
        match self {
            FormulaAlign::Left => "left",
            FormulaAlign::Center => "center",
            FormulaAlign::Right => "right",
        }
    }
}

/// `formula` 块（V2-j 公式排版引擎化）的编译产物：**引擎节点 + 排版参数**。
///
/// DSL 层不依赖文本管线（编译期产物纯数据）：本节点只承载
/// 「记法原文 + 归一化排版参数」，消费侧把它交给
/// `prism-text::math_ast::MathExpr::parse`（记法 → AST）与
/// `prism-text::math_layout::layout_math`（AST → 字形布局）即可渲染出字。
#[derive(Debug, Clone, PartialEq)]
pub struct FormulaNode {
    /// 公式名（文档内唯一）。
    pub name: String,
    /// 记法原文（LaTeX 风格；`\frac{dy}{dx}` / `x^2` / `\sqrt[3]{x}` / …）。
    pub notation: String,
    /// 基座字号（px，归一化后的正数；缺省 32）。
    pub size: f32,
    /// 锚点 x（px；按 `align` 解释为左缘 / 中心 / 右缘）。
    pub x: f32,
    /// 锚点 y（px；公式**基线**的设备 y）。
    pub y: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 对齐方式。
    pub align: FormulaAlign,
    /// formula 关键字所在行（诊断定位）。
    pub line: usize,
}

impl FormulaNode {
    /// 公式锚点按对齐方式折算为**左缘** x（渲染侧定位公式盒用）。
    pub fn left_x(&self, width: f32) -> f32 {
        match self.align {
            FormulaAlign::Left => self.x,
            FormulaAlign::Center => self.x - width * 0.5,
            FormulaAlign::Right => self.x - width,
        }
    }
}

// ---------------------------------------------------------------------------
// v2 单位归一化（C17：显式单位 → 确定性规范单位）
// ---------------------------------------------------------------------------

/// 时间字面量 → 秒（frames 按最终 fps 折算；ms/1000）。
fn time_to_secs(tv: TimeVal, fps: u32) -> f32 {
    match tv.unit {
        TimeUnit::Seconds => tv.value,
        TimeUnit::Millis => tv.value / 1000.0,
        TimeUnit::Frames => tv.value / fps as f32,
    }
}

/// 角度字面量 → 弧度（deg → to_radians；rad 原值）。
fn angle_to_radians(av: AngleVal) -> f32 {
    match av.unit {
        AngleUnit::Deg => av.value.to_radians(),
        AngleUnit::Rad => av.value,
    }
}

// ---------------------------------------------------------------------------
// v2 小助手：字段值类型提取
// ---------------------------------------------------------------------------

/// 字段值 → f32（仅 Num）。
fn num_of(fv: Option<&FieldValue>) -> Option<f32> {
    match fv? {
        FieldValue::Num(v) => Some(*v),
        _ => None,
    }
}

/// 字段值 → 字符串。
fn str_of<'a>(fv: Option<&'a FieldValue>) -> Option<&'a str> {
    match fv? {
        FieldValue::Str(s) => Some(s.as_str()),
        _ => None,
    }
}

/// 字段值 → 颜色（0..255 原值）。
fn color_of(fv: Option<&FieldValue>) -> Option<(u8, u8, u8)> {
    match fv? {
        FieldValue::Color(r, g, b) => Some((*r, *g, *b)),
        _ => None,
    }
}

/// 字段值 → Vec3。
fn vec3_of(fv: Option<&FieldValue>) -> Option<Vec3> {
    match fv? {
        FieldValue::Vec3(v) => Some(*v),
        _ => None,
    }
}

/// 字段值 → 标识符。
fn ident_of<'a>(fv: Option<&'a FieldValue>) -> Option<&'a str> {
    match fv? {
        FieldValue::Ident(s) => Some(s.as_str()),
        _ => None,
    }
}

/// 字段值 → 子块（返回 (子块名或空串, 字段列表)；NamedBlock / Block 两态兼容）。
fn block_of(fv: Option<&FieldValue>) -> Option<(&str, &[NamedField])> {
    match fv? {
        FieldValue::Block(fields) => Some(("", fields.as_slice())),
        FieldValue::NamedBlock { name, fields } => Some((name.as_str(), fields.as_slice())),
        _ => None,
    }
}

/// 字段值 → 带类型不匹配建议的错误（P2xxx 类型段统一入口）。
fn type_err(line: usize, what: &str, want: &str, got: Option<&FieldValue>) -> DslError {
    let got_desc: String = match got {
        Some(FieldValue::Str(s)) => format!("字符串 \"{}\"", s),
        Some(FieldValue::Num(v)) => format!("数字 {}", v),
        Some(FieldValue::Color(r, g, b)) => format!("颜色 #{:02x}{:02x}{:02x}", r, g, b),
        Some(FieldValue::Ident(s)) => format!("'{}'", s),
        Some(FieldValue::Time(_)) => "时间字面量".to_string(),
        Some(FieldValue::Angle(_)) => "角度字面量".to_string(),
        Some(FieldValue::Percent(v)) => format!("百分比 {}%", v),
        Some(FieldValue::Px(v)) => format!("长度 {}px", v),
        Some(FieldValue::Vec3(v)) => format!("vec3({}, {}, {})", v.x, v.y, v.z),
        Some(FieldValue::Keys(_)) => "关键帧序列".to_string(),
        Some(FieldValue::NamedBlock { .. }) => "带名子块".to_string(),
        Some(FieldValue::Block(_)) => "子块".to_string(),
        None => "（缺失）".to_string(),
    };
    err_at(
        line,
        format!(
            "[P2100] {} 的值类型不符：期望{}，却得到 {}\n  建议：参考字段表给出正确类型的值",
            what, want, got_desc
        ),
    )
}

// ---------------------------------------------------------------------------
// v2 easing / track / channel
// ---------------------------------------------------------------------------

/// easing 名 → `prism_anim::EasingKind`（支持直接名与 ease 前缀别名）。
fn easing_from_name(name: &str, line: usize) -> Result<EasingKind, DslError> {
    let kind: EasingKind = match name {
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
        other => {
            return Err(err_at(
                line,
                format!(
                    "[P2103] 未知的 easing 名 '{}'\n  建议：linear / quadIn / easeOutCubic / bounceInOut 等（10 个基础族 × In/Out/InOut 共 31 种）",
                    other
                ),
            ));
        }
    };
    Ok(kind)
}

/// 关键帧序列 → `prism_anim::Track`（时间归一化到秒 + 乱序防呆 P1020 +
/// easing 名映射 P2103）。返回 (track, 末帧时间)。
fn build_track(keys: &[crate::ast::KeyAst], fps: u32, desc: &str) -> Result<(Track, f32), DslError> {
    let mut prev_t: Option<f32> = None;
    let mut kfs: Vec<Keyframe> = Vec::with_capacity(keys.len());
    for k in keys {
        let t: f32 = time_to_secs(k.time, fps);
        if let Some(p) = prev_t {
            if t <= p {
                return Err(err_at(
                    k.line,
                    format!(
                        "[P1020] {} 的关键帧时间必须严格递增（{}s 出现在 {}s 之后）\n  建议：按时间升序书写 key 条目；混合单位（36f / 0.5s）会先按 fps 折算再比较",
                        desc, t, p
                    ),
                ));
            }
        }
        let easing: EasingKind = easing_from_name(&k.easing, k.line)?;
        kfs.push(Keyframe::new(t, k.value, easing));
        prev_t = Some(t);
    }
    let end: f32 = prev_t.unwrap_or(0.0);
    Ok((Track::new(kfs), end))
}

/// 关键帧序列（+ 可选 loop 模式）→ 属性通道。loop 需要 ≥2 个不同时间的 key。
fn build_channel(
    keys: &[crate::ast::KeyAst],
    loop_mode: Option<&str>,
    fps: u32,
    desc: &str,
    line: usize,
) -> Result<PropertyChannel, DslError> {
    let (track, end): (Track, f32) = build_track(keys, fps, desc)?;
    let inner: PropertyChannel = PropertyChannel::Track(track);
    match loop_mode {
        None | Some("direct") => Ok(inner),
        Some("loop") | Some("pingpong") => {
            if end <= 0.0 {
                return Err(err_at(
                    line,
                    format!(
                        "[P1020] {} 的 loop/pingpong 需要至少两个不同时间的 key（当前轨道时长 0s）\n  建议：补一个时间更晚的 key 条目",
                        desc
                    ),
                ));
            }
            let mode: LoopMode =
                if loop_mode == Some("loop") { LoopMode::Loop } else { LoopMode::PingPong };
            Ok(PropertyChannel::Loop {
                inner: Box::new(inner),
                mode,
                duration: end,
            })
        }
        Some(other) => Err(err_at(
            line,
            format!(
                "[P2104] 未知的 loop 模式 '{}'\n  建议：direct（直接截断）/ loop（循环）/ pingpong（往复）三选一",
                other
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// v2 构建器
// ---------------------------------------------------------------------------

/// v2 codegen 状态机：图 + id 分配 + 通道登记 + 场景内块名 → 节点索引。
struct V2Builder {
    fps: u32,
    /// 当前场景名（通道命名前缀）。
    scene_name: String,
    graph: Graph,
    next_id: NodeId,
    channels: Vec<NamedChannel>,
    /// (场景序号, 块名) → 节点 id（compose 解引用；点查不迭代 → 确定性）。
    block_nodes: BTreeMap<(usize, &'static str, String), NodeId>,
}

impl V2Builder {
    fn new(fps: u32) -> Self {
        Self {
            fps,
            scene_name: String::new(),
            graph: Graph::new(),
            next_id: 1,
            channels: Vec::new(),
            block_nodes: BTreeMap::new(),
        }
    }

    /// 通道命名的场景前缀。
    fn scene_hint(&self) -> &str {
        &self.scene_name
    }

    fn alloc_id(&mut self) -> NodeId {
        let id: NodeId = self.next_id;
        self.next_id += 1;
        id
    }

    /// 标量字段值：Num 原值 / Percent÷100 / Px 原值 / Keys→通道（t=0 采样为
    /// 静态值并登记）。返回 (静态值, 可选通道)。
    fn scalar_or_channel(
        &mut self,
        v: Option<&FieldValue>,
        line: usize,
        what: &str,
        ch_name: &str,
    ) -> Result<(f32, Option<PropertyChannel>), DslError> {
        let fv: &FieldValue = match v {
            Some(fv) => fv,
            None => {
                return Err(err_at(
                    line,
                    format!("[P2100] {} 缺少值\n  建议：给出数字 / 百分比 / 像素 / 关键帧块之一", what),
                ));
            }
        };
        match fv {
            FieldValue::Num(n) => Ok((*n, None)),
            FieldValue::Percent(p) => Ok((p / 100.0, None)),
            FieldValue::Px(p) => Ok((*p, None)),
            FieldValue::Keys(keys) => {
                let ch: PropertyChannel =
                    build_channel(keys, None, self.fps, what, line)?;
                let v0: f32 = ch.sample(0.0);
                self.channels.push(NamedChannel { name: ch_name.to_string(), channel: ch.clone() });
                Ok((v0, Some(ch)))
            }
            other => Err(type_err(line, what, "数字 / 百分比 / 像素 / 关键帧块", Some(other))),
        }
    }
}

// ---------------------------------------------------------------------------
// v2 环检测 / 引用校验（建图前的纯图检查，错误带行号）
// ---------------------------------------------------------------------------

/// 全文档 scene 名查重（P3008 家族防呆）：transition 的 from / to 按场景名
/// 匹配，重名即匹配歧义，编译期显式拒绝；错误指出两处定义行号。
fn check_scene_name_dup(doc: &DocumentV2Ast) -> Result<(), DslError> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new(); // 场景名 → 首次定义行
    for scene in &doc.scenes {
        if let Some(&first_line) = seen.get(scene.name.as_str()) {
            return Err(err_at(
                scene.line,
                format!(
                    "[P3008] scene 名 '{}' 重复：第 {first_line} 行与第 {} 行各定义一次\n  建议：改名其一；transition from/to 按场景名匹配，重名会产生歧义",
                    scene.name, scene.line
                ),
            ));
        }
        seen.insert(scene.name.as_str(), scene.line);
    }
    Ok(())
}

/// 场景内具名块的引用检查：重名（P3008）→ 悬空（P3006）→ 环（P3005，带行号）。
fn check_scene_refs(doc: &DocumentV2Ast) -> Result<(), DslError> {
    for (si, scene) in doc.scenes.iter().enumerate() {
        // 1) 收集具名块（layer / sim / compose）。
        let mut names: BTreeMap<String, usize> = BTreeMap::new(); // 块名 → 定义行
        let mut composes: Vec<&ComposeAst> = Vec::new();
        for item in &scene.items {
            match item {
                SceneItem::Layer(l) => check_dup(&mut names, &l.name, l.line, "layer", si)?,
                SceneItem::Sim(s) => check_dup(&mut names, &s.name, s.line, "sim", si)?,
                SceneItem::Compose(c) => {
                    check_dup(&mut names, &c.name, c.line, "compose", si)?;
                    composes.push(c);
                }
                _ => {}
            }
        }
        // 2) 悬空引用检查 + compose → compose 边（环只可能经过 compose）。
        let mut edges: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
        for c in &composes {
            for op in &c.ops {
                for r in &op.refs {
                    if !matches!(r.kind.as_str(), "layer" | "sim" | "compose") {
                        return Err(err_at(
                            r.line,
                            format!(
                                "[P3006] compose 操作 '{}' 的引用 '{}:{}' 种类未知\n  建议：compose 只能引用 layer / sim / compose 三类块，如 add layer:text",
                                op.op, r.kind, r.name
                            ),
                        ));
                    }
                    if !names.contains_key(&r.name) {
                        return Err(err_at(
                            r.line,
                            format!(
                                "[P3006] compose 操作 '{}' 引用的块 '{}' 未在本场景定义\n  建议：引用只在本 scene 内解析；现有块：{}",
                                op.op,
                                r.name,
                                names.keys().cloned().collect::<Vec<_>>().join("、")
                            ),
                        ));
                    }
                    if r.kind == "compose" {
                        edges.entry(c.name.clone()).or_default().push((r.name.clone(), r.line));
                    }
                }
            }
        }
        // 3) DFS 环检测（三色标记；灰点回边 → 环）。
        let mut color: BTreeMap<&str, u8> = names.keys().map(|k| (k.as_str(), 0u8)).collect();
        let mut stack: Vec<(String, usize)> = Vec::new(); // (块名, 定义行)
        for name in names.keys() {
            if color.get(name.as_str()).copied().unwrap_or(2) == 0 {
                dfs_cycle(
                    name,
                    &edges,
                    &names,
                    &mut color,
                    &mut stack,
                )?;
            }
        }
    }
    Ok(())
}

/// 重名检测（同场景内 layer / sim / compose 共享命名空间）。
fn check_dup(
    names: &mut BTreeMap<String, usize>,
    name: &str,
    line: usize,
    kind: &str,
    scene_idx: usize,
) -> Result<(), DslError> {
    if let Some(first_line) = names.get(name).copied() {
        return Err(err_at(
            line,
            format!(
                "[P3008] 第 {} 个 scene 中块名 '{}' 重复（先前作为 {} 定义于第 {} 行）\n  建议：改名其一；compose 引用按名字解析，重名会产生歧义",
                scene_idx + 1,
                name,
                kind,
                first_line
            ),
        ));
    }
    names.insert(name.to_string(), line);
    Ok(())
}

/// DFS 找环：发现灰点回边时报错，路径链带每个 compose 的定义行号。
fn dfs_cycle<'a>(
    u: &'a str,
    edges: &'a BTreeMap<String, Vec<(String, usize)>>,
    names: &'a BTreeMap<String, usize>,
    color: &mut BTreeMap<&'a str, u8>,
    stack: &mut Vec<(String, usize)>,
) -> Result<(), DslError> {
    color.insert(u, 1);
    stack.push((u.to_string(), names.get(u).copied().unwrap_or(0)));
    if let Some(nexts) = edges.get(u) {
        for (v, ref_line) in nexts {
            match color.get(v.as_str()).copied().unwrap_or(2) {
                1 => {
                    // 环：从栈中 v 的位置截取到栈顶。
                    let start: usize =
                        stack.iter().position(|(n, _)| n == v).unwrap_or(0);
                    let mut chain: Vec<String> = Vec::new();
                    for (n, ln) in &stack[start..] {
                        chain.push(format!("'{}'（第 {} 行）", n, ln));
                    }
                    chain.push(format!("'{}'（引用于第 {} 行）", v, ref_line));
                    return Err(err_at(
                        *ref_line,
                        format!(
                            "[P3005] compose 引用成环：{}\n  建议：断开环中一条引用（如删掉 add compose:... 的一端），让合成图保持有向无环",
                            chain.join(" → ")
                        ),
                    ));
                }
                0 => {
                    dfs_cycle(v, edges, names, color, stack)?;
                }
                _ => {}
            }
        }
    }
    color.insert(u, 2);
    stack.pop();
    Ok(())
}

// ---------------------------------------------------------------------------
// v2 codegen 主流程
// ---------------------------------------------------------------------------

/// v2 AST → 产物。语法已校验；本层做跨块语义（引用 / 环 / 时间线 / 单位归一化）。
pub(crate) fn codegen_v2(doc: &DocumentV2Ast) -> Result<CompileV2Output, DslError> {
    // ---- 1. 全局参数（fps / 输出 / 标题；次序 timeline → movie → 缺省） ----
    let fps: u32 = doc
        .timeline
        .as_ref()
        .and_then(|t| t.fps)
        .or_else(|| doc.movie.as_ref().and_then(|m| m.fps))
        .unwrap_or(DEFAULT_FPS);
    let (width, height): (u32, u32) = doc
        .output
        .as_ref()
        .and_then(|o| o.resolution)
        .or_else(|| doc.movie.as_ref().and_then(|m| m.resolution))
        .unwrap_or((DEFAULT_WIDTH, DEFAULT_HEIGHT));
    let fmt_ast: Option<OutputFormatAst> = doc
        .output
        .as_ref()
        .and_then(|o| o.format)
        .or_else(|| doc.movie.as_ref().and_then(|m| m.format));
    let format: OutputFormat = match fmt_ast {
        Some(OutputFormatAst::Frames) | None => OutputFormat::Frames,
        Some(OutputFormatAst::Png) => OutputFormat::Png,
    };
    let title: String =
        doc.movie.as_ref().map_or("untitled".to_string(), |m| m.title.clone());

    // ---- 2. packs（同名查重 P4003 + engine 范围校验，C12/V9） ----
    let mut packs: Vec<PackManifest> = Vec::new();
    // 包名 → 首次声明的行号（同名重复登记 = 加载歧义，显式报错优于幂等去重）。
    let mut pack_first_line: BTreeMap<String, usize> = BTreeMap::new();
    for imp in &doc.imports {
        if let Some(&first_line) = pack_first_line.get(&imp.name) {
            return Err(err_at(
                imp.line,
                format!(
                    "[P4003] import packs '{}' 重复登记（首次声明于第 {first_line} 行）\n  建议：同名包只声明一次：删除重复的 import packs 块（显式优于隐式，不做幂等去重）",
                    imp.name
                ),
            ));
        }
        pack_first_line.insert(imp.name.clone(), imp.line);
        let m: PackManifest = PackManifest::from_fields(&imp.name, &imp.fields, imp.line)?;
        m.check_engine()?;
        packs.push(m);
    }

    // ---- 2.5 scene 名查重（P3008：transition from/to 按场景名匹配，重名即歧义） ----
    check_scene_name_dup(doc)?;

    // ---- 3. 引用 / 环检测（建图前） ----
    check_scene_refs(doc)?;

    // ---- 3.5 transition 场景引用校验（P3007：from / to 必须指向已定义场景） ----
    let scene_names: Vec<String> = doc.scenes.iter().map(|s| s.name.clone()).collect();
    for t in &doc.transitions {
        for (label, end) in [("from", t.from.as_ref()), ("to", t.to.as_ref())] {
            if let Some((name, line)) = end {
                if !scene_names.iter().any(|s| s == name) {
                    return Err(err_at(
                        *line,
                        format!(
                            "[P3007] transition '{}' 的 {} 场景 '{}' 未在文档中定义\n  建议：{} 的场景名必须与某个 scene 块的名字一致；现有场景：{}",
                            t.name, label, name, label,
                            scene_names.join("、")
                        ),
                    ));
                }
            }
        }
    }

    // ---- 3.6 theme（与 v1 aesthetic 同构；对比度缺省 1.0） ----
    let theme: Option<ThemeInfo> = doc.theme.as_ref().map(|t| ThemeInfo {
        teal_orange: t.teal_orange,
        contrast: t.contrast.unwrap_or(DEFAULT_CONTRAST),
        filmic: t.filmic,
    });

    // ---- 4. 场景时长与过渡重叠计算（时间线计算表） ----
    let scene_count: usize = doc.scenes.len();
    let durations: Vec<f32> = doc
        .scenes
        .iter()
        .map(|s| match s.duration {
            Some(tv) => time_to_secs(tv, fps),
            None => doc
                .movie
                .as_ref()
                .and_then(|m| m.duration)
                .map(|md| time_to_secs(md, fps) / scene_count as f32)
                .unwrap_or(DEFAULT_SCENE_DURATION),
        })
        .collect();
    let mut starts: Vec<f32> = vec![0.0];
    let mut transitions_v2: Vec<TransitionV2> = Vec::new();
    for i in 0..scene_count.saturating_sub(1) {
        // 匹配"相邻场景对"的过渡（from = scenes[i]，to = scenes[i+1]）。
        let matched: Vec<&TransitionAst> = doc
            .transitions
            .iter()
            .filter(|t| {
                t.from.as_ref().map_or(false, |(f, _)| *f == scene_names[i])
                    && t.to.as_ref().map_or(false, |(t2, _)| *t2 == scene_names[i + 1])
            })
            .collect();
        let overlap: f32 = match matched.len() {
            0 => 0.0,
            1 => {
                let t: &TransitionAst = matched[0];
                let d: f32 = t
                    .duration
                    .map_or(DEFAULT_TRANSITION_DURATION, |tv| time_to_secs(tv, fps));
                if d >= durations[i] || d >= durations[i + 1] {
                    return Err(err_at(
                        t.line,
                        format!(
                            "[P1011] transition '{}' 的重叠时长 {}s 必须短于两侧场景时长（{}s / {}s）\n  建议：缩短 transition.duration，或加长两侧 scene 的 duration",
                            t.name, d, durations[i], durations[i + 1]
                        ),
                    ));
                }
                transitions_v2.push(TransitionV2 {
                    name: t.name.clone(),
                    from: t.from.as_ref().map_or(String::new(), |(f, _)| f.clone()),
                    to: t.to.as_ref().map_or(String::new(), |(t2, _)| t2.clone()),
                    start: starts[i] + durations[i] - d,
                    duration: d,
                    kind: t.kind.clone(),
                    band: t.band.unwrap_or(DEFAULT_WIPE_BAND),
                    seed: t.seed.unwrap_or(0),
                    dir: t.dir.clone(),
                });
                d
            }
            _ => {
                return Err(err_at(
                    matched[1].line,
                    format!(
                        "[P1001] 场景 '{}' 与 '{}' 之间出现多条 transition\n  建议：一对相邻场景至多一条过渡；合并或删除多余 transition 块",
                        scene_names[i], scene_names[i + 1]
                    ),
                ));
            }
        };
        starts.push(starts[i] + durations[i] - overlap);
    }

    // ---- 5. 节点图构建（逐场景；compose 递归建点 + 连边） ----
    let mut b: V2Builder = V2Builder::new(fps);
    let mut scenes_v2: Vec<SceneV2> = Vec::new();
    for (si, scene) in doc.scenes.iter().enumerate() {
        b.scene_name = scene.name.clone();
        let id_begin: NodeId = b.next_id;
        let mut camera: Option<CameraOrbit> = None;
        let mut layers: Vec<LayerInfo> = Vec::new();
        let mut sim: Option<SimInfo> = None;
        let mut post: PostInfo = PostInfo::default();
        let mut audio: Option<AudioInfo> = None;
        let mut notes: Vec<NoteInfo> = Vec::new();
        // 本场景的 compose 定义列表（build_compose 递归解析引用用）。
        let scene_composes: Vec<&ComposeAst> = scene
            .items
            .iter()
            .filter_map(|it| match it {
                SceneItem::Compose(c) => Some(c),
                _ => None,
            })
            .collect();
        for item in &scene.items {
            match item {
                SceneItem::Stage3d(st) => {
                    let cam: CameraOrbit = build_stage3d(&mut b, st, si)?;
                    camera = Some(cam);
                }
                SceneItem::Layer(l) => {
                    let node_id: NodeId = build_layer(&mut b, l, si)?;
                    let morph: Option<(String, String)> = l
                        .morph
                        .as_ref()
                        .map(|m| (m.from.clone(), m.to.clone()));
                    layers.push(LayerInfo {
                        name: l.name.clone(),
                        kind: l.kind.clone(),
                        node_id,
                        morph,
                    });
                }
                SceneItem::Sim(s) => {
                    let node_id: NodeId = build_sim(&mut b, s, si)?;
                    sim = Some(SimInfo { kind: s.kind.clone(), name: s.name.clone(), node_id });
                }
                SceneItem::Compose(c) => {
                    build_compose(&mut b, c, si, &scene_composes)?;
                }
                SceneItem::Audio(a) => {
                    audio = Some(build_audio(a)?);
                }
                SceneItem::Note(n) => {
                    notes.push(build_note(n)?);
                }
                SceneItem::Post(p) => {
                    post = build_post(p)?;
                }
            }
        }
        let node_ids: Vec<NodeId> = (id_begin..b.next_id).collect();
        scenes_v2.push(SceneV2 {
            name: scene.name.clone(),
            start: starts[si],
            duration: durations[si],
            node_ids,
            camera,
            layers,
            sim,
            post,
            audio,
            notes,
        });
    }

    // ---- 6. 顶层 anim / expr 通道登记 ----
    for anim in &doc.anims {
        let ch: PropertyChannel = build_channel(
            &anim.keys,
            anim.loop_mode.as_deref(),
            fps,
            &format!("anim '{}'", anim.target),
            anim.line,
        )?;
        b.channels.push(NamedChannel { name: anim.target.clone(), channel: ch });
    }
    for e in &doc.exprs {
        let expr: Expr = Expr::parse(&e.body).map_err(|parse_err| {
            err_at(
                e.line,
                format!(
                    "[P2105] expr '{}' 解析失败：{:?}\n  建议：受限表达式只支持变量 t/p/w/h、数字与 + - * / ( ) 运算（函数调用等完整语法随后续版本）",
                    e.name, parse_err
                ),
            )
        })?;
        b.channels.push(NamedChannel { name: e.name.clone(), channel: PropertyChannel::Expr(expr) });
    }

    // ---- 7. 总时长（timeline → movie → 末场景结束） ----
    let content_end: f32 = starts
        .last()
        .copied()
        .map_or(0.0, |last_start| last_start + durations[durations.len() - 1]);
    let duration: f32 = doc
        .timeline
        .as_ref()
        .and_then(|t| t.duration)
        .or_else(|| doc.movie.as_ref().and_then(|m| m.duration).map(|md| time_to_secs(md, fps)))
        .unwrap_or(content_end);

    // ---- 8. 字体回退链构造序列（V1.5-B：声明顺序 = push 顺序） ----
    let fonts: Vec<FontSlotSpec> = codegen_fonts(doc);

    // ---- 9. formula 块 → 引擎公式节点（V2-j：记法 + 归一化排版参数） ----
    let mut formulas: Vec<FormulaNode> = Vec::with_capacity(doc.formulas.len());
    for f in &doc.formulas {
        if formulas.iter().any(|n| n.name == f.name) {
            return Err(err_at(
                f.line,
                format!(
                    "[P1001] formula '{}' 重名（同一文档内公式名必须唯一）\n  建议：给其中一个改名，如 formula \"{}2\"",
                    f.name, f.name
                ),
            ));
        }
        formulas.push(build_formula(f)?);
    }

    Ok(CompileV2Output {
        title,
        width,
        height,
        fps,
        duration,
        format,
        scenes: scenes_v2,
        transitions: transitions_v2,
        graph: b.graph,
        channels: b.channels,
        packs,
        theme,
        fonts,
        formulas,
    })
}

/// font 指令序列 → `FontSet` 构造序列（V1.5-B）。
///
/// 顺序纪律：多条 font 指令按出现顺序、每条内首选后 fallback，展平为
/// **单条**回退链；`index` = 展平后下标 = `FontSet::push_file` 返回的槽位号。
/// 纯函数、无隐式兜底：文档未声明 font 就产出空序列，绝不暗中塞内嵌字体。
fn codegen_fonts(doc: &DocumentV2Ast) -> Vec<FontSlotSpec> {
    let mut out: Vec<FontSlotSpec> = Vec::new();
    for decl in &doc.fonts {
        out.push(FontSlotSpec {
            label: decl.path.clone(),
            path: decl.path.clone(),
            index: out.len(),
            line: decl.line,
        });
        for fb in &decl.fallbacks {
            out.push(FontSlotSpec {
                label: fb.clone(),
                path: fb.clone(),
                index: out.len(),
                line: decl.line,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// v2 各块构建
// ---------------------------------------------------------------------------

/// stage3d 块 → 节点（camera→S5、sdf→S12、material→S8、light 校验后登记忽略），
/// 返回场景相机摘要。
fn build_stage3d(b: &mut V2Builder, st: &Stage3dAst, si: usize) -> Result<CameraOrbit, DslError> {
    // camera：至多一个（parser 已保证），无则缺省轨道。
    let mut orbit: CameraOrbit =
        CameraOrbit { center: Vec3::new(0.0, 0.0, 0.0), radius: DEFAULT_ORBIT_RADIUS, fov_rad: DEFAULT_FOV_DEG.to_radians() };
    if let Some(cam) = st.camera.as_ref() {
        orbit = build_camera(b, cam, si)?;
    }
    for sdf in &st.sdfs {
        build_sdf(b, sdf, si)?;
    }
    for mat in &st.materials {
        build_material(b, mat, si)?;
    }
    // light：参数白名单校验（direction / color / intensity），产物登记忽略（Phase 2/3 接入光源求解）。
    for light in &st.lights {
        validate_light(light)?;
    }
    Ok(orbit)
}

/// camera 条目 → S5 重参数化节点 + 相机摘要。fov 裸数字按"度"（C17 豁免），
/// 也接受 `45deg` / `0.5rad`；超界 (0°, 180°) → P1014。
fn build_camera(b: &mut V2Builder, cam: &CameraV2Ast, si: usize) -> Result<CameraOrbit, DslError> {
    let mut center: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    let mut radius: f32 = DEFAULT_ORBIT_RADIUS;
    let mut fov_rad: f32 = DEFAULT_FOV_DEG.to_radians();
    for f in &cam.fields {
        match f.name.as_str() {
            "orbit" => {
                let (_, sub): (&str, &[NamedField]) =
                    block_of(f.value.as_ref()).ok_or_else(|| {
                        type_err(f.line, "camera 块中字段 'orbit'", "子块（center / radius）", f.value.as_ref())
                    })?;
                for sf in sub {
                    match sf.name.as_str() {
                        "center" => {
                            center = vec3_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "camera.orbit 中字段 'center'", "vec3（数字,数字,数字）", sf.value.as_ref())
                            })?;
                        }
                        "radius" => {
                            let r: f32 = num_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "camera.orbit 中字段 'radius'", "数字", sf.value.as_ref())
                            })?;
                            if r < 0.0 {
                                return Err(err_at(
                                    sf.line,
                                    format!(
                                        "[P1015] camera.orbit.radius 不能为负（得到 {}）\n  建议：轨道半径是非负长度，如 radius 8",
                                        r
                                    ),
                                ));
                            }
                            radius = r;
                        }
                        other => {
                            return Err(err_at(
                                sf.line,
                                format!(
                                    "[P1002] camera.orbit 子块中未知字段 '{}'（允许：center / radius）\n  建议：center 写 vec3（如 0,0,0），radius 写数字",
                                    other
                                ),
                            ));
                        }
                    }
                }
            }
            "fov" => {
                fov_rad = match f.value.as_ref() {
                    Some(FieldValue::Num(deg)) => {
                        if !(*deg > 0.0 && *deg < 180.0) {
                            return Err(err_at(
                                f.line,
                                format!(
                                    "[P1014] camera.fov 应在 (0, 180) 度开区间内，却得到 {} 度\n  建议：常见取值 30 / 45 / 60 / 90（fov 裸数字按度解释，属单位豁免项）",
                                    deg
                                ),
                            ));
                        }
                        deg.to_radians()
                    }
                    Some(FieldValue::Angle(av)) => {
                        let rad: f32 = angle_to_radians(*av);
                        let deg: f32 = rad.to_degrees();
                        if !(deg > 0.0 && deg < 180.0) {
                            return Err(err_at(
                                f.line,
                                format!(
                                    "[P1014] camera.fov 应在 (0°, 180°) 开区间内，却得到约 {} 度\n  建议：常见取值 30deg / 45deg / 60deg / 90deg",
                                    deg
                                ),
                            ));
                        }
                        rad
                    }
                    other => {
                        return Err(type_err(f.line, "camera 块中字段 'fov'", "数字（度，豁免单位）或角度字面量", other));
                    }
                };
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] camera 块中未知字段 '{}'（允许：orbit / fov）\n  建议：orbit {{ center .. radius .. }} 与 fov 45",
                        other
                    ),
                ));
            }
        }
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::Reparameterization, Composition::Feed);
    node.params.set("orbit_x", ParamValue::F32(center.x));
    node.params.set("orbit_y", ParamValue::F32(center.y));
    node.params.set("orbit_z", ParamValue::F32(center.z));
    node.params.set("orbit_radius", ParamValue::F32(radius));
    node.params.set("fov_rad", ParamValue::F32(fov_rad));
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "camera", cam.name.clone()), id);
    Ok(CameraOrbit { center, radius, fov_rad })
}

/// sdf 条目 → S12 实体表示节点。kind 必填（sphere / box）；radius 缺省 1，负值 P1015。
fn build_sdf(b: &mut V2Builder, sdf: &SdfAst, si: usize) -> Result<NodeId, DslError> {
    let mut kind: &str = "";
    let mut radius: f32 = DEFAULT_RADIUS;
    let mut radius_channel: Option<PropertyChannel> = None;
    for f in &sdf.fields {
        match f.name.as_str() {
            "kind" => {
                kind = ident_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "sdf 块中字段 'kind'", "标识符（sphere / box）", f.value.as_ref())
                })?;
            }
            "radius" => {
                // 负值防呆（P1015）在静态数字路径；通道化值域交由运行期语义。
                if let Some(FieldValue::Num(n)) = f.value.as_ref() {
                    if *n < 0.0 {
                        return Err(err_at(
                            f.line,
                            format!("[P1015] sdf.radius 不能为负（得到 {}）\n  建议：半径是非负长度，如 radius 1", n),
                        ));
                    }
                }
                let (r, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                    f.value.as_ref(),
                    f.line,
                    "sdf 块中字段 'radius'",
                    &format!("{}/{}.radius", b.scene_hint(), sdf.name),
                )?;
                radius = r;
                radius_channel = ch;
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] sdf 块中未知字段 '{}'（允许：kind / radius）\n  建议：kind sphere / kind box，radius 1",
                        other
                    ),
                ));
            }
        }
    }
    if kind.is_empty() {
        return Err(err_at(
            sdf.line,
            format!(
                "[P1003] sdf '{}' 缺少必填字段 'kind'\n  建议：kind sphere 或 kind box（最小支持集）",
                sdf.name
            ),
        ));
    }
    if !matches!(kind, "sphere" | "box") {
        return Err(err_at(
            sdf.line,
            format!(
                "[P1002] 未知的 sdf kind '{}'\n  建议：当前支持 sphere / box 两种基元（更多基元随几何模块接入）",
                kind
            ),
        ));
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::EntityRepresentation, Composition::Feed);
    node.params.set("kind", ParamValue::Str(kind.to_string()));
    node.params.set("radius", ParamValue::F32(radius));
    if radius_channel.is_some() {
        node.bindings.push(NodeBinding {
            param: "radius".to_string(),
            channel: ChannelRef { name: format!("{}/{}.radius", b.scene_hint(), sdf.name) },
        });
    }
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "sdf", sdf.name.clone()), id);
    Ok(id)
}

/// material 条目 → S8 辐射传输节点。albedo 缺省白；roughness / metalness ∈ [0,1]。
fn build_material(b: &mut V2Builder, mat: &MaterialV2Ast, si: usize) -> Result<NodeId, DslError> {
    let mut albedo: (u8, u8, u8) = (255, 255, 255);
    let mut roughness: f32 = DEFAULT_ROUGHNESS;
    let mut metalness: f32 = DEFAULT_METALNESS;
    for f in &mat.fields {
        match f.name.as_str() {
            "albedo" => {
                albedo = color_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "material 块中字段 'albedo'", "#RRGGBB 颜色", f.value.as_ref())
                })?;
            }
            "roughness" => {
                let r: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "material 块中字段 'roughness'", "数字", f.value.as_ref())
                })?;
                if !(0.0..=1.0).contains(&r) {
                    return Err(err_at(
                        f.line,
                        format!("[P1017] material.roughness 应在 [0, 1] 区间，却得到 {}\n  建议：0 = 全镜面，1 = 全粗糙；常见 0.3 / 0.5", r),
                    ));
                }
                roughness = r;
            }
            "metalness" => {
                let m: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "material 块中字段 'metalness'", "数字", f.value.as_ref())
                })?;
                if !(0.0..=1.0).contains(&m) {
                    return Err(err_at(
                        f.line,
                        format!("[P1016] material.metalness 应在 [0, 1] 区间，却得到 {}\n  建议：0 = 电介质，1 = 金属", m),
                    ));
                }
                metalness = m;
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] material 块中未知字段 '{}'（允许：albedo / roughness / metalness）\n  建议：albedo #c0392b roughness 0.3 metalness 0.05",
                        other
                    ),
                ));
            }
        }
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::RadiativeTransfer, Composition::Feed);
    node.params.set("albedo_r", ParamValue::F32(albedo.0 as f32 / 255.0));
    node.params.set("albedo_g", ParamValue::F32(albedo.1 as f32 / 255.0));
    node.params.set("albedo_b", ParamValue::F32(albedo.2 as f32 / 255.0));
    node.params.set("roughness", ParamValue::F32(roughness));
    node.params.set("metalness", ParamValue::F32(metalness));
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "material", mat.name.clone()), id);
    Ok(id)
}

/// light 条目参数白名单校验（direction / color / intensity；产物登记忽略）。
fn validate_light(light: &LightV2Ast) -> Result<(), DslError> {
    for f in &light.fields {
        match f.name.as_str() {
            "direction" => {
                if vec3_of(f.value.as_ref()).is_none() {
                    return Err(type_err(
                        f.line,
                        "light 块中字段 'direction'",
                        "vec3（如 (1,-1,-0.5) 或 1,-1,-0.5）",
                        f.value.as_ref(),
                    ));
                }
            }
            "color" => {
                if color_of(f.value.as_ref()).is_none() {
                    return Err(type_err(
                        f.line,
                        "light 块中字段 'color'",
                        "#RRGGBB 颜色",
                        f.value.as_ref(),
                    ));
                }
            }
            "intensity" => {
                if num_of(f.value.as_ref()).is_none() {
                    return Err(type_err(
                        f.line,
                        "light 块中字段 'intensity'",
                        "数字",
                        f.value.as_ref(),
                    ));
                }
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] light 块中未知字段 '{}'（允许：direction / color / intensity）\n  建议：direction(1,-1,-0.5) color #fff4e6 intensity 3",
                        other
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// layer 块 → 节点（text→S12 / particles→S4），填充 params 与通道绑定。
fn build_layer(b: &mut V2Builder, layer: &LayerAst, si: usize) -> Result<NodeId, DslError> {
    match layer.kind.as_str() {
        "text" => build_layer_text(b, layer, si),
        "particles" => build_layer_particles(b, layer, si),
        other => Err(err_at(
            layer.line,
            format!(
                "[P1002] 未知的 layer 种类 '{}'\n  建议：当前支持 text / particles 两种图层（更多图层随渲染模块接入）",
                other
            ),
        )),
    }
}

/// text 图层：content / x / y / size / color / rotation / opacity。
fn build_layer_text(b: &mut V2Builder, layer: &LayerAst, si: usize) -> Result<NodeId, DslError> {
    let mut params: Params = Params::new();
    let mut bindings: Vec<NodeBinding> = Vec::new();
    let mut rotation_rad: Option<f32> = None;
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for f in &layer.fields {
        if let Some(first_line) = seen.get(&f.name).copied() {
            return Err(err_at(
                f.line,
                format!(
                    "[P3008] layer '{}' 中字段 '{}' 重复出现（先前在第 {} 行）\n  建议：删除多余字段；动画请用关键帧块而非重复字段",
                    layer.name, f.name, first_line
                ),
            ));
        }
        seen.insert(f.name.clone(), f.line);
        let ch_name: String = format!("{}/{}.{}", b.scene_hint(), layer.name, f.name);
        match f.name.as_str() {
            "content" => {
                let s: &str = str_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "layer text 块中字段 'content'", "字符串", f.value.as_ref())
                })?;
                params.set("content", ParamValue::Str(s.to_string()));
            }
            "x" | "y" => {
                let (v, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                    f.value.as_ref(),
                    f.line,
                    &format!("layer text 块中字段 '{}'", f.name),
                    &ch_name,
                )?;
                params.set(f.name.as_str(), ParamValue::F32(v));
                if ch.is_some() {
                    bindings.push(NodeBinding {
                        param: f.name.clone(),
                        channel: ChannelRef { name: ch_name },
                    });
                }
            }
            "size" => {
                let (v, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                    f.value.as_ref(),
                    f.line,
                    "layer text 块中字段 'size'",
                    &ch_name,
                )?;
                params.set("size", ParamValue::F32(v));
                if ch.is_some() {
                    bindings.push(NodeBinding {
                        param: "size".to_string(),
                        channel: ChannelRef { name: ch_name },
                    });
                }
            }
            "color" => {
                let c: (u8, u8, u8) = color_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "layer text 块中字段 'color'", "#RRGGBB 颜色", f.value.as_ref())
                })?;
                params.set("color_r", ParamValue::F32(c.0 as f32 / 255.0));
                params.set("color_g", ParamValue::F32(c.1 as f32 / 255.0));
                params.set("color_b", ParamValue::F32(c.2 as f32 / 255.0));
            }
            "rotation" => match f.value.as_ref() {
                // C17：角度必须显式单位；fov 之外不接受裸角度。
                Some(FieldValue::Angle(av)) => rotation_rad = Some(angle_to_radians(*av)),
                Some(FieldValue::Num(n)) => {
                    return Err(err_at(
                        f.line,
                        format!(
                            "[P2101] layer text 块中字段 'rotation' 应带角度单位，却得到裸数字 {}\n  建议：写成 15deg（度）或 0.5rad（弧度）",
                            n
                        ),
                    ));
                }
                other => {
                    return Err(type_err(
                        f.line,
                        "layer text 块中字段 'rotation'",
                        "角度字面量（15deg / 0.5rad）",
                        Some(other.unwrap_or(&FieldValue::Num(0.0))),
                    ));
                }
            },
            "opacity" => {
                let (v, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                    f.value.as_ref(),
                    f.line,
                    "layer text 块中字段 'opacity'",
                    &ch_name,
                )?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(err_at(
                        f.line,
                        format!("[P1022] layer text 块中字段 'opacity' 应在 [0, 1] 区间，却得到 {}\n  建议：0 = 全透明，1 = 不透明", v),
                    ));
                }
                params.set("opacity", ParamValue::F32(v));
                if ch.is_some() {
                    bindings.push(NodeBinding {
                        param: "opacity".to_string(),
                        channel: ChannelRef { name: ch_name },
                    });
                }
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] layer text 块中未知字段 '{}'（允许：content / x / y / size / color / rotation / opacity）\n  建议：检查字段名拼写",
                        other
                    ),
                ));
            }
        }
    }
    // 缺省值回填（未声明字段的确定性缺省）。
    if params.get("content").is_none() {
        params.set("content", ParamValue::Str(String::new()));
    }
    if params.get("x").is_none() {
        params.set("x", ParamValue::F32(0.0));
    }
    if params.get("y").is_none() {
        params.set("y", ParamValue::F32(0.0));
    }
    if params.get("size").is_none() {
        params.set("size", ParamValue::F32(DEFAULT_TEXT_SIZE));
    }
    if params.get("opacity").is_none() {
        params.set("opacity", ParamValue::F32(1.0));
    }
    if params.get("color_r").is_none() {
        params.set("color_r", ParamValue::F32(1.0));
        params.set("color_g", ParamValue::F32(1.0));
        params.set("color_b", ParamValue::F32(1.0));
    }
    if let Some(r) = rotation_rad {
        params.set("rotation", ParamValue::F32(r));
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::EntityRepresentation, Composition::Feed);
    node.params = params;
    node.bindings = bindings;
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "layer", layer.name.clone()), id);
    Ok(id)
}

/// particles 图层：count / speed。
fn build_layer_particles(
    b: &mut V2Builder,
    layer: &LayerAst,
    si: usize,
) -> Result<NodeId, DslError> {
    let mut params: Params = Params::new();
    let mut bindings: Vec<NodeBinding> = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for f in &layer.fields {
        if let Some(first_line) = seen.get(&f.name).copied() {
            return Err(err_at(
                f.line,
                format!(
                    "[P3008] layer '{}' 中字段 '{}' 重复出现（先前在第 {} 行）\n  建议：删除多余字段",
                    layer.name, f.name, first_line
                ),
            ));
        }
        seen.insert(f.name.clone(), f.line);
        let ch_name: String = format!("{}/{}.{}", b.scene_hint(), layer.name, f.name);
        match f.name.as_str() {
            "count" => {
                let n: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "layer particles 块中字段 'count'", "正整数", f.value.as_ref())
                })?;
                if n.fract() != 0.0 || n < 1.0 {
                    return Err(err_at(
                        f.line,
                        format!(
                            "[P1012] layer particles 块中字段 'count' 应为正整数（至少 1），却得到 {}\n  建议：count 是粒子数量，如 count 200",
                            n
                        ),
                    ));
                }
                params.set("count", ParamValue::U32(n as u32));
            }
            "speed" => {
                let (v, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                    f.value.as_ref(),
                    f.line,
                    "layer particles 块中字段 'speed'",
                    &ch_name,
                )?;
                params.set("speed", ParamValue::F32(v));
                if ch.is_some() {
                    bindings.push(NodeBinding {
                        param: "speed".to_string(),
                        channel: ChannelRef { name: ch_name },
                    });
                }
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] layer particles 块中未知字段 '{}'（允许：count / speed）\n  建议：count 200 speed 0.5",
                        other
                    ),
                ));
            }
        }
    }
    if params.get("count").is_none() {
        params.set("count", ParamValue::U32(DEFAULT_PARTICLE_COUNT));
    }
    if params.get("speed").is_none() {
        params.set("speed", ParamValue::F32(DEFAULT_PARTICLE_SPEED));
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::DeterministicSampling, Composition::Feed);
    node.params = params;
    node.bindings = bindings;
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "layer", layer.name.clone()), id);
    Ok(id)
}

/// sim 块 → S3 约束梯度流节点（grid / stiffness / damping / force / noise）。
fn build_sim(b: &mut V2Builder, sim: &SimAst, si: usize) -> Result<NodeId, DslError> {
    let mut params: Params = Params::new();
    params.set("kind", ParamValue::Str(sim.kind.clone()));
    let mut pending_bindings: Vec<NodeBinding> = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for f in &sim.fields {
        if let Some(first_line) = seen.get(&f.name).copied() {
            return Err(err_at(
                f.line,
                format!(
                    "[P3008] sim '{}' 中字段 '{}' 重复出现（先前在第 {} 行）\n  建议：删除多余字段",
                    sim.name, f.name, first_line
                ),
            ));
        }
        seen.insert(f.name.clone(), f.line);
        match f.name.as_str() {
            "grid" => {
                let n: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "sim 块中字段 'grid'", "正整数", f.value.as_ref())
                })?;
                if n.fract() != 0.0 || n < 1.0 {
                    return Err(err_at(
                        f.line,
                        format!(
                            "[P1012] sim 块中字段 'grid' 应为正整数（至少 1），却得到 {}\n  建议：grid 是网格边长，如 grid 16",
                            n
                        ),
                    ));
                }
                params.set("grid", ParamValue::U32(n as u32));
            }
            "stiffness" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "sim 块中字段 'stiffness'", "数字", f.value.as_ref())
                })?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(err_at(
                        f.line,
                        format!("[P1017] sim.stiffness 应在 [0, 1] 区间，却得到 {}\n  建议：0 = 完全柔性，1 = 完全刚性；常见 0.9", v),
                    ));
                }
                params.set("stiffness", ParamValue::F32(v));
            }
            "damping" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "sim 块中字段 'damping'", "数字", f.value.as_ref())
                })?;
                if v < 0.0 {
                    return Err(err_at(
                        f.line,
                        format!("[P1015] sim.damping 不能为负（得到 {}）\n  建议：阻尼是非负数；常见 0.02", v),
                    ));
                }
                params.set("damping", ParamValue::F32(v));
            }
            "force" => {
                let (fname, sub): (&str, &[NamedField]) =
                    block_of(f.value.as_ref()).ok_or_else(|| {
                        type_err(f.line, "sim 块中字段 'force'", "子块（strength / direction）", f.value.as_ref())
                    })?;
                let force_name: &str = if fname.is_empty() { "wind" } else { fname };
                let mut strength: f32 = DEFAULT_FORCE_STRENGTH;
                let mut direction: Vec3 = Vec3::new(1.0, 0.0, 0.0);
                let mut has_strength_channel: Option<(PropertyChannel, String)> = None;
                for sf in sub {
                    match sf.name.as_str() {
                        "strength" => {
                            let ch_base: String =
                                format!("{}/{}/force/{}/strength", b.scene_hint(), sim.name, force_name);
                            let (v, ch): (f32, Option<PropertyChannel>) = b.scalar_or_channel(
                                sf.value.as_ref(),
                                sf.line,
                                "sim force 子块中字段 'strength'",
                                &ch_base,
                            )?;
                            strength = v;
                            if let Some(ch) = ch {
                                has_strength_channel = Some((ch, ch_base));
                            }
                        }
                        "direction" => {
                            direction = vec3_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "sim force 子块中字段 'direction'", "vec3", sf.value.as_ref())
                            })?;
                        }
                        other => {
                            return Err(err_at(
                                sf.line,
                                format!(
                                    "[P1002] sim force 子块中未知字段 '{}'（允许：strength / direction）\n  建议：force wind {{ strength 3 direction(1,0,0) }}",
                                    other
                                ),
                            ));
                        }
                    }
                }
                // strength 通道：关键帧用解析产物；静态值也登记 Const 通道，
                // 运行期统一经命名通道系统取值（确定性缺省同路径）。
                let (strength_ch, strength_name): (PropertyChannel, String) =
                    match has_strength_channel {
                        Some((ch, name)) => (ch, name),
                        None => (
                            PropertyChannel::Const(strength),
                            format!("{}/{}/force/{}/strength", b.scene_hint(), sim.name, force_name),
                        ),
                    };
                params.set("force_strength", ParamValue::F32(strength_ch.sample(0.0)));
                params.set("force_dir_x", ParamValue::F32(direction.x));
                params.set("force_dir_y", ParamValue::F32(direction.y));
                params.set("force_dir_z", ParamValue::F32(direction.z));
                b.channels.push(NamedChannel { name: strength_name.clone(), channel: strength_ch });
                pending_bindings.push(NodeBinding {
                    param: "force_strength".to_string(),
                    channel: ChannelRef { name: strength_name },
                });
            }
            "noise" => {
                let (_, sub): (&str, &[NamedField]) =
                    block_of(f.value.as_ref()).ok_or_else(|| {
                        type_err(f.line, "sim 块中字段 'noise'", "子块（scale）", f.value.as_ref())
                    })?;
                for sf in sub {
                    match sf.name.as_str() {
                        "scale" => {
                            let v: f32 = num_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "sim noise 子块中字段 'scale'", "数字", sf.value.as_ref())
                            })?;
                            params.set("noise_scale", ParamValue::F32(v));
                        }
                        other => {
                            return Err(err_at(
                                sf.line,
                                format!(
                                    "[P1002] sim noise 子块中未知字段 '{}'（允许：scale）\n  建议：noise {{ scale 0.1 }}",
                                    other
                                ),
                            ));
                        }
                    }
                }
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] sim 块中未知字段 '{}'（允许：grid / stiffness / damping / force / noise）\n  建议：检查字段名拼写",
                        other
                    ),
                ));
            }
        }
    }
    // 缺省回填。
    if params.get("grid").is_none() {
        params.set("grid", ParamValue::U32(DEFAULT_SIM_GRID));
    }
    if params.get("stiffness").is_none() {
        params.set("stiffness", ParamValue::F32(DEFAULT_SIM_STIFFNESS));
    }
    if params.get("damping").is_none() {
        params.set("damping", ParamValue::F32(DEFAULT_SIM_DAMPING));
    }
    if params.get("force_strength").is_none() {
        params.set("force_strength", ParamValue::F32(DEFAULT_FORCE_STRENGTH));
    }
    if params.get("noise_scale").is_none() {
        params.set("noise_scale", ParamValue::F32(DEFAULT_NOISE_SCALE));
    }
    let id: NodeId = b.alloc_id();
    let mut node: Node = Node::new(id, StructureId::ConstraintFlow, Composition::Feed);
    node.params = params;
    node.bindings = std::mem::take(&mut pending_bindings);
    b.graph.add_node(node).expect("递增 id 不应重复");
    b.block_nodes.insert((si, "sim", sim.name.clone()), id);
    Ok(id)
}

/// compose 块 → S2 卷积汇点节点（Merge(op)）+ 边连接（含 compose→compose 递归）。
/// `scene_composes` 为本场景的全部 compose 定义（递归解析引用用）。
fn build_compose(
    b: &mut V2Builder,
    compose: &ComposeAst,
    si: usize,
    scene_composes: &[&ComposeAst],
) -> Result<NodeId, DslError> {
    // memo：已建的 compose 直接复用（防重复建点；环已在 check_scene_refs 拦截）。
    if let Some(&id) = b.block_nodes.get(&(si, "compose", compose.name.clone())) {
        return Ok(id);
    }
    let op_kind: &str = compose.ops[0].op.as_str();
    let merge_op: MergeOp = match op_kind {
        "add" => MergeOp::Plus,
        "multiply" => MergeOp::Multiply,
        "over" => MergeOp::Over,
        "screen" => MergeOp::Screen,
        "mask" => MergeOp::Mask,
        other => {
            return Err(err_at(
                compose.ops[0].line,
                format!(
                    "[P1002] 未知的 compose 操作 '{}'\n  建议：add（加算）/ multiply（正片叠底）/ over（覆盖）/ screen（滤色）/ mask（蒙版）五选一",
                    other
                ),
            ));
        }
    };
    if compose.ops.len() > 1 {
        return Err(err_at(
            compose.ops[1].line,
            format!(
                "[P1001] compose '{}' 中出现第二个操作 '{}'（当前一个 compose 块只支持一条操作）\n  建议：拆成多个 compose 块并用引用串联，或合并操作数",
                compose.name, compose.ops[1].op
            ),
        ));
    }
    // 先递归建被引用的 compose（拓扑序分配 id），再建自身。
    let mut src_ids: Vec<NodeId> = Vec::new();
    for r in &compose.ops[0].refs {
        let id: NodeId = match r.kind.as_str() {
            "compose" => {
                let target: &ComposeAst = scene_composes
                    .iter()
                    .copied()
                    .find(|c| c.name == r.name)
                    .ok_or_else(|| dangling(r))?;
                build_compose(b, target, si, scene_composes)?
            }
            "layer" | "sim" => {
                *b.block_nodes
                    .get(&(si, r.kind.as_str(), r.name.clone()))
                    .ok_or_else(|| dangling(r))?
            }
            other => {
                return Err(err_at(
                    r.line,
                    format!(
                        "[P3006] compose 引用种类 '{}' 未知\n  建议：layer / sim / compose 三选一",
                        other
                    ),
                ));
            }
        };
        src_ids.push(id);
    }
    let id: NodeId = b.alloc_id();
    let node: Node = Node::new(id, StructureId::Convolution, Composition::Merge(merge_op));
    b.graph.add_node(node).expect("递增 id 不应重复");
    for (slot, src) in src_ids.into_iter().enumerate() {
        b.graph
            .connect(src, id, slot as u8)
            .expect("新节点槽位不应冲突");
    }
    b.block_nodes.insert((si, "compose", compose.name.clone()), id);
    Ok(id)
}

/// 悬空引用错误（建图阶段的兜底；check_scene_refs 已前置拦截，此为防御）。
fn dangling(r: &ComposeRef) -> DslError {
    err_at(
        r.line,
        format!(
            "[P3006] compose 引用的块 '{}' 未在本场景定义\n  建议：引用只在本 scene 内解析；先定义同名 layer / sim / compose",
            r.name
        ),
    )
}

/// audio 块 → AudioInfo 摘要（music 子块的 bpm / gain / duck 平铺 + 防呆）。
fn build_audio(audio: &AudioAst) -> Result<AudioInfo, DslError> {
    let mut info: AudioInfo = AudioInfo::default();
    for f in &audio.fields {
        match f.name.as_str() {
            "music" => {
                let (_, sub): (&str, &[NamedField]) =
                    block_of(f.value.as_ref()).ok_or_else(|| {
                        type_err(f.line, "audio 块中字段 'music'", "子块（bpm / gain / duck）", f.value.as_ref())
                    })?;
                let mut music: BTreeMap<String, f32> = BTreeMap::new();
                for sf in sub {
                    match sf.name.as_str() {
                        "bpm" => {
                            let v: f32 = num_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "audio.music 子块中字段 'bpm'", "数字", sf.value.as_ref())
                            })?;
                            if v <= 0.0 {
                                return Err(err_at(
                                    sf.line,
                                    format!("[P1019] audio.music.bpm 应为正数，却得到 {}\n  建议：bpm 是每分钟拍数，如 bpm 120", v),
                                ));
                            }
                            music.insert("bpm".to_string(), v);
                        }
                        "gain" => {
                            let v: f32 = num_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "audio.music 子块中字段 'gain'", "数字", sf.value.as_ref())
                            })?;
                            if !(0.0..=1.0).contains(&v) {
                                return Err(err_at(
                                    sf.line,
                                    format!("[P1022] audio.music.gain 应在 [0, 1] 区间，却得到 {}\n  建议：0 = 静音，1 = 满增益", v),
                                ));
                            }
                            music.insert("gain".to_string(), v);
                        }
                        "duck" => {
                            let v: f32 = num_of(sf.value.as_ref()).ok_or_else(|| {
                                type_err(sf.line, "audio.music 子块中字段 'duck'", "数字（dB）", sf.value.as_ref())
                            })?;
                            if v <= 0.0 {
                                return Err(err_at(
                                    sf.line,
                                    format!("[P1018] audio.music.duck 应为正 dB 值，却得到 {}\n  建议：duck 是闪避深度（dB），如 duck 12", v),
                                ));
                            }
                            music.insert("duck".to_string(), v);
                        }
                        other => {
                            return Err(err_at(
                                sf.line,
                                format!(
                                    "[P1002] audio.music 子块中未知字段 '{}'（允许：bpm / gain / duck）\n  建议：music {{ bpm 120 gain 0.8 duck 12 }}",
                                    other
                                ),
                            ));
                        }
                    }
                }
                // 缺省值回填（未声明字段的确定性缺省）。
                music.entry("bpm".to_string()).or_insert(DEFAULT_BPM);
                music.entry("gain".to_string()).or_insert(DEFAULT_GAIN);
                music.entry("duck".to_string()).or_insert(DEFAULT_DUCK_DB);
                info.music = music;
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] audio 块中未知字段 '{}'（允许：music）\n  建议：audio {{ music {{ bpm 120 gain 0.8 duck 12 }} }}",
                        other
                    ),
                ));
            }
        }
    }
    Ok(info)
}

/// note 块 → NoteInfo 摘要：scale/root/bpm/bars/seed/arpeggio 白名单 + 防呆。
fn build_note(note: &NoteAst) -> Result<NoteInfo, DslError> {
    let mut scale: String = NOTE_DEFAULT_SCALE.to_string();
    let mut root: i32 = NOTE_DEFAULT_ROOT;
    let mut bpm: f32 = NOTE_DEFAULT_BPM;
    let mut bars: u32 = NOTE_DEFAULT_BARS;
    let mut seed: u64 = NOTE_DEFAULT_SEED;
    let mut arpeggio: bool = false;
    for f in &note.fields {
        match f.name.as_str() {
            "scale" => {
                // 音阶：裸标识符（如 scale major）。
                let s: &str = ident_of(f.value.as_ref()).or_else(|| str_of(f.value.as_ref()))
                    .ok_or_else(|| {
                        type_err(f.line, "note 块中字段 'scale'", "音阶标识符", f.value.as_ref())
                    })?;
                if !NOTE_SCALES.contains(&s) {
                    return Err(err_at(
                        f.line,
                        format!(
                            "[P1023] note 块中未知音阶 '{}'（允许：{}）\n  建议：如 scale major / scale phrygian",
                            s,
                            NOTE_SCALES.join(" / ")
                        ),
                    ));
                }
                scale = s.to_string();
            }
            "root" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "note 块中字段 'root'", "数字（MIDI 音高）", f.value.as_ref())
                })?;
                if !(0.0..=127.0).contains(&v) {
                    return Err(err_at(
                        f.line,
                        format!("[P1024] note 块中字段 'root' 应在 [0, 127]，却得到 {v}\n  建议：root 为 MIDI 音高，如 root 60（C4）"),
                    ));
                }
                root = v.round() as i32;
            }
            "bpm" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "note 块中字段 'bpm'", "数字", f.value.as_ref())
                })?;
                if v <= 0.0 {
                    return Err(err_at(
                        f.line,
                        format!("[P1019] note 块中字段 'bpm' 应为正数，却得到 {v}\n  建议：bpm 是每分钟拍数，如 bpm 120"),
                    ));
                }
                bpm = v;
            }
            "bars" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "note 块中字段 'bars'", "数字（小节数）", f.value.as_ref())
                })?;
                if v < 1.0 {
                    return Err(err_at(
                        f.line,
                        format!("[P1025] note 块中字段 'bars' 应为 ≥ 1 的整数，却得到 {v}\n  建议：bars 为小节数，如 bars 4"),
                    ));
                }
                bars = v.round() as u32;
            }
            "seed" => {
                let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
                    type_err(f.line, "note 块中字段 'seed'", "数字", f.value.as_ref())
                })?;
                if v < 0.0 {
                    return Err(err_at(
                        f.line,
                        format!("[P1026] note 块中字段 'seed' 应为非负数，却得到 {v}\n  建议：seed 为非负整数，如 seed 7"),
                    ));
                }
                seed = v.round() as u64;
            }
            "arpeggio" => {
                // 标志字段（无值 = true）或布尔数字（1/0）。
                arpeggio = match f.value.as_ref() {
                    None => true,
                    Some(FieldValue::Ident(s)) => s != "off" && s != "false",
                    Some(FieldValue::Num(n)) => *n != 0.0,
                    other => {
                        return Err(type_err(
                            f.line,
                            "note 块中字段 'arpeggio'",
                            "标志或布尔（0/1）",
                            other,
                        ));
                    }
                };
            }
            other => {
                return Err(err_at(
                    f.line,
                    format!(
                        "[P1002] note 块中未知字段 '{}'（允许：scale / root / bpm / bars / seed / arpeggio）\n  建议：note \"主题\" {{ scale major root 60 bpm 120 bars 4 seed 7 }}",
                        other
                    ),
                ));
            }
        }
    }
    Ok(NoteInfo { name: note.name.clone(), scale, root, bpm, bars, seed, arpeggio })
}

/// 字段值 → 长度（px）：接受裸数字与 `120px`；其余 → `None`。
fn length_of(fv: Option<&FieldValue>) -> Option<f32> {
    match fv? {
        FieldValue::Num(v) => Some(*v),
        FieldValue::Px(v) => Some(*v),
        _ => None,
    }
}

/// `formula` 块 → [`FormulaNode`]（V2-j：记法必填 + 参数归一化 + 白名单校验）。
///
/// 校验清单：
/// - `notation` 必填且非空（语法校验；结构校验由 `prism-text::math_ast` 负责）；
/// - `size` 为正有限数（缺省 32）；`x` / `y` 有限（缺省 0）；
/// - `color` 为 `#RRGGBB`（缺省白）；`align` ∈ {left, center, right}（缺省 center）；
/// - 未知字段 → P1002（说人话 + 建议）。
fn build_formula(f: &FormulaAst) -> Result<FormulaNode, DslError> {
    let notation: String = match &f.notation {
        Some(s) if !s.trim().is_empty() => s.clone(),
        Some(_) => {
            return Err(err_at(
                f.line,
                format!(
                    "[P1022] formula '{}' 的 notation 为空\n  建议：填入公式记法，如 notation \"\\\\frac{{dy}}{{dx}} = 2x\"",
                    f.name
                ),
            ));
        }
        None => {
            return Err(err_at(
                f.line,
                format!(
                    "[P2107] formula '{}' 缺少 notation 字段（公式记法）\n  建议：写成 formula \"{}\" {{ notation \"\\\\frac{{dy}}{{dx}} = 2x\" size 34 }}",
                    f.name, f.name
                ),
            ));
        }
    };
    let mut size: f32 = 32.0;
    let mut x: f32 = 0.0;
    let mut y: f32 = 0.0;
    let mut color: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let mut align: FormulaAlign = FormulaAlign::Center;
    for field in &f.fields {
        match field.name.as_str() {
            "size" => {
                let v: f32 = length_of(field.value.as_ref()).ok_or_else(|| {
                    type_err(field.line, "formula 块中字段 'size'", "数字或 px 长度", field.value.as_ref())
                })?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(err_at(
                        field.line,
                        format!(
                            "[P1021] formula '{}' 的 size 必须为正有限数，却得到 {}\n  建议：size 是基座字号（px），如 size 34",
                            f.name, v
                        ),
                    ));
                }
                size = v;
            }
            "x" | "y" => {
                let v: f32 = length_of(field.value.as_ref()).ok_or_else(|| {
                    type_err(
                        field.line,
                        &format!("formula 块中字段 '{}'", field.name),
                        "数字或 px 长度",
                        field.value.as_ref(),
                    )
                })?;
                if !v.is_finite() {
                    return Err(err_at(
                        field.line,
                        format!("[P1021] formula '{}' 的 {} 必须为有限数", f.name, field.name),
                    ));
                }
                if field.name == "x" {
                    x = v;
                } else {
                    y = v;
                }
            }
            "color" => match field.value.as_ref() {
                Some(FieldValue::Color(r, g, b)) => {
                    color = [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0];
                }
                other => {
                    return Err(type_err(field.line, "formula 块中字段 'color'", "颜色 #RRGGBB", other));
                }
            },
            "align" => {
                let text: &str = match field.value.as_ref() {
                    Some(FieldValue::Ident(s)) => s.as_str(),
                    Some(FieldValue::Str(s)) => s.as_str(),
                    other => {
                        return Err(type_err(
                            field.line,
                            "formula 块中字段 'align'",
                            "标识符 left / center / right",
                            other,
                        ));
                    }
                };
                align = FormulaAlign::parse(text).ok_or_else(|| {
                    err_at(
                        field.line,
                        format!(
                            "[P1021] formula '{}' 的 align 只能是 left / center / right，却得到 '{}'\n  建议：align center",
                            f.name, text
                        ),
                    )
                })?;
            }
            other => {
                return Err(err_at(
                    field.line,
                    format!(
                        "[P1002] formula 块中未知字段 '{}'（允许：notation / size / x / y / color / align）\n  建议：formula \"名\" {{ notation \"x^2\" size 34 x 200 y 300 }}",
                        other
                    ),
                ));
            }
        }
    }
    Ok(FormulaNode { name: f.name.clone(), notation, size, x, y, color, align, line: f.line })
}

/// post 块 → PostInfo 摘要（bloom / grain / vignette / chromatic ∈ [0,1]）。
fn build_post(post: &PostAst) -> Result<PostInfo, DslError> {
    let mut info: PostInfo = PostInfo::default();
    for f in &post.fields {
        let v: f32 = num_of(f.value.as_ref()).ok_or_else(|| {
            type_err(f.line, &format!("post 块中字段 '{}'", f.name), "数字", f.value.as_ref())
        })?;
        if !(0.0..=1.0).contains(&v) {
            return Err(err_at(
                f.line,
                format!(
                    "[P1021] post 块中字段 '{}' 应在 [0, 1] 区间，却得到 {}\n  建议：post 强度是归一化比例：0 = 关闭，1 = 最强",
                    f.name, v
                ),
            ));
        }
        if !matches!(f.name.as_str(), "bloom" | "grain" | "vignette" | "chromatic") {
            return Err(err_at(
                f.line,
                format!(
                    "[P1002] post 块中未知字段 '{}'（允许：bloom / grain / vignette / chromatic）\n  建议：post {{ bloom 0.3 grain 0.1 }}",
                    f.name
                ),
            ));
        }
        info.params.insert(f.name.clone(), v);
    }
    Ok(info)
}

// ---------------------------------------------------------------------------
// 组合子注册面（V1.5-D）：name → 展开 fn
// ---------------------------------------------------------------------------

pub use crate::combinators::{
    expand_ken_burns, expand_lower_third, expand_morph_between, expand_split_screen,
    expand_title_card, CombinatorArgs, CombinatorError, NodeGraph,
};

/// 组合子表：组合子名 → 等价 v2 节点图展开函数（V1.5-D）。
///
/// # 用途与边界
/// - **本轮**：DSL v2 语法不加新指令（parser 刚稳定不碰），本表供 Rust 侧
///   （`prism-author::combinators`）与文档按名查表展开；
/// - **后续波次**：DSL 糖化指令（如 `use title_card { ... }`）经本表把语法糖
///   展开为节点图，codegen 主流程零分支改动；
/// - 展开产物 [`NodeGraph`] 与 [`CompileV2Output`] 的节点图部分同构
///   （`NodeGraph::into_compile_output` 补齐空缺字段）。
pub static COMBINATOR_TABLE: &[(&str, fn(&CombinatorArgs) -> Result<NodeGraph, CombinatorError>)] = &[
    ("title_card", expand_title_card),
    ("lower_third", expand_lower_third),
    ("split_screen", expand_split_screen),
    ("ken_burns", expand_ken_burns),
    ("morph_between", expand_morph_between),
];
