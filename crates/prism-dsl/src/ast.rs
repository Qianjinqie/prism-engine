//! AST：v1（M0/M1）文档模型逐字保留；v2 新增全块语法模型。
//!
//! # v2 建模策略
//! - **单位字面量**：[`TimeVal`] / [`AngleVal`] 保留"原值 + 单位"，解析期不做
//!   归一化（帧 → 秒依赖 fps，而 fps 由 timeline/movie 决定，归一化在 codegen
//!   统一完成，保证确定性）；
//! - **开放参数集的块**（layer / sim / audio / post / force / noise 等）用
//!   [`NamedField`] 通用字段列表承载，由 codegen 按块类型做白名单校验；
//! - **结构固定的条目**（stage3d 内的 camera / sdf / material / light、
//!   transition、anim 关键帧）用专用强类型，保证核心语义可靠。

// 纯数据模型：个别 v1 字段 / 变体暂未被 codegen 消费（为后续扩展保留语义位置），
// 逐个标注会污染定义，此处模块级豁免。
#![allow(dead_code)]

use prism_math::Vec3;

// ---------------------------------------------------------------------------
// v1 文档模型（原样保留，字段升为 pub(crate) 供 parser / codegen 跨模块访问）
// ---------------------------------------------------------------------------

/// 顶层文档：scene 必需，其余可选。
pub(crate) struct DocumentAst {
    pub(crate) scene: SceneAst,
    pub(crate) timeline: Option<TimelineAst>,
    pub(crate) aesthetic: Option<AestheticAst>,
    pub(crate) output: Option<OutputAst>,
}

/// scene 块。语义约束：至多一个 sphere、一个 camera；light 可多个。
pub(crate) struct SceneAst {
    /// 场景名（暂不参与编译产物，保留给未来诊断 / 多场景导出）。
    pub(crate) name: String,
    pub(crate) sphere: Option<SphereAst>,
    pub(crate) camera: Option<CameraAst>,
    pub(crate) lights: Vec<LightAst>,
}

/// sphere 条目。
pub(crate) struct SphereAst {
    /// 条目名（暂不参与编译产物，保留给未来多物体扩展）。
    pub(crate) name: String,
    pub(crate) radius: f32,
    pub(crate) material: Option<MaterialAst>,
}

/// material 块。
pub(crate) struct MaterialAst {
    pub(crate) albedo: Option<(u8, u8, u8)>,
    pub(crate) roughness: Option<f32>,
    pub(crate) metalness: Option<f32>,
}

/// camera 条目。
pub(crate) struct CameraAst {
    /// 条目名（暂不参与编译产物，保留给未来多相机扩展）。
    pub(crate) name: String,
    pub(crate) orbit_center: Option<Vec3>,
    pub(crate) orbit_radius: Option<f32>,
    pub(crate) fov_deg: f32,
}

/// light 条目。
pub(crate) struct LightAst {
    /// 条目名（暂不参与编译产物，保留给未来诊断）。
    pub(crate) name: String,
    pub(crate) direction: Vec3,
    pub(crate) color: Option<(u8, u8, u8)>,
    pub(crate) intensity: Option<f32>,
}

/// timeline 块。
pub(crate) struct TimelineAst {
    pub(crate) duration: Option<f32>,
    pub(crate) fps: Option<u32>,
}

/// aesthetic 块（v2 的 theme 块复用本类型：字段集相同，见）。
pub(crate) struct AestheticAst {
    pub(crate) teal_orange: bool,
    pub(crate) contrast: Option<f32>,
    pub(crate) filmic: bool,
}

/// output 块。
pub(crate) struct OutputAst {
    pub(crate) resolution: Option<(u32, u32)>,
    pub(crate) format: Option<OutputFormatAst>,
}

/// 输出格式（AST 层；与 lib.rs 公开的 [`crate::OutputFormat`] 在 codegen 转换）。
///
/// 为避免 v1 parser / codegen 之间引入对公开枚举的依赖，AST 层独立持有
/// 同构枚举，取值与 [`crate::OutputFormat`] 一一对应。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormatAst {
    Frames,
    Png,
}

// ---------------------------------------------------------------------------
// v2：单位字面量
// ---------------------------------------------------------------------------

/// 时间单位（`3s` / `150ms` / `36f`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeUnit {
    /// 秒。
    Seconds,
    /// 毫秒（→ 秒 = 值 / 1000）。
    Millis,
    /// 帧（→ 秒 = 值 / fps；fps 由 timeline → movie → 缺省次序确定后在
    /// codegen 归一化，保证 lattice 纪律下的确定性）。
    Frames,
}

/// 时间字面量：原值 + 单位（解析期逐字保真，不做归一化）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimeVal {
    pub(crate) value: f32,
    pub(crate) unit: TimeUnit,
    pub(crate) line: usize,
}

/// 角度单位（`15deg` / `0.5rad`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AngleUnit {
    /// 度（→ 弧度 = 值.to_radians()）。
    Deg,
    /// 弧度（原值透传）。
    Rad,
}

/// 角度字面量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AngleVal {
    pub(crate) value: f32,
    pub(crate) unit: AngleUnit,
    pub(crate) line: usize,
}

// ---------------------------------------------------------------------------
// v2：通用字段模型
// ---------------------------------------------------------------------------

/// v2 通用字段值（覆盖所有块的字段值形态）。
#[derive(Debug, Clone)]
pub(crate) enum FieldValue {
    /// 普通数字（含负数 / 小数）。
    Num(f32),
    /// 双引号字符串。
    Str(String),
    /// `#RRGGBB` 颜色（0..255 原值）。
    Color(u8, u8, u8),
    /// 时间字面量（`3s` / `150ms` / `36f`）。
    Time(TimeVal),
    /// 角度字面量（`15deg` / `0.5rad`）。
    Angle(AngleVal),
    /// 百分比 `50%`。
    Percent(f32),
    /// 像素长度 `120px`。
    Px(f32),
    /// 裸标识符（如 `kind sphere` 的 `sphere`、`loop pingpong`）。
    Ident(String),
    /// vec3：`数字,数字,数字`（如 force 的 direction）。
    Vec3(Vec3),
    /// 关键帧序列（块体形式：`{ key 0s 0 linear ... }`）。
    Keys(Vec<KeyAst>),
    /// 带名子块（`force wind { ... }` → name = "wind"；无名子块用 Block）。
    NamedBlock { name: String, fields: Vec<NamedField> },
    /// 任意子块（orbit / music / noise 等，codegen 白名单校验）。
    Block(Vec<NamedField>),
}

/// v2 通用字段：`名 值`（值缺省 = 无值标志字段，如 `filmic`）。
#[derive(Debug, Clone)]
pub(crate) struct NamedField {
    pub(crate) name: String,
    pub(crate) value: Option<FieldValue>,
    pub(crate) line: usize,
}

// ---------------------------------------------------------------------------
// v2：全块语法 AST
// ---------------------------------------------------------------------------

/// movie 块：文档级时长 / 帧率 / 分辨率 / 格式（可选，至多一个）。
pub(crate) struct MovieAst {
    /// 影片标题（字符串；暂不参与编译产物，保留给诊断 / 元数据）。
    pub(crate) title: String,
    pub(crate) duration: Option<TimeVal>,
    pub(crate) fps: Option<u32>,
    pub(crate) resolution: Option<(u32, u32)>,
    pub(crate) format: Option<OutputFormatAst>,
    /// movie 关键字所在行（保留给未来诊断）。
    pub(crate) line: usize,
}

/// scene v2 块体内的条目。
pub(crate) enum SceneItem {
    Stage3d(Stage3dAst),
    Layer(LayerAst),
    Sim(SimAst),
    Compose(ComposeAst),
    Audio(AudioAst),
    Note(NoteAst),
    Post(PostAst),
}

/// scene v2 块：`scene "名" { duration? 条目* }`（可多个场景，构成时间线）。
pub(crate) struct SceneV2Ast {
    pub(crate) name: String,
    /// 场景时长（可选；缺省按 movie.duration 平分或 DEFAULT_SCENE_DURATION）。
    pub(crate) duration: Option<TimeVal>,
    pub(crate) items: Vec<SceneItem>,
    /// scene 关键字所在行（诊断定位用：scene 名重复 P3008 等）。
    pub(crate) line: usize,
}

/// stage3d 内的 camera 条目：`camera 名 { orbit { ... } fov 45 }`。
///
/// fov 沿用 v1 豁免：裸数字按"度"解释（C17 豁免清单明确 fov 为度数）。
pub(crate) struct CameraV2Ast {
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// stage3d 内的 sdf 条目：`sdf 名 { kind sphere radius 1 }`。
pub(crate) struct SdfAst {
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// stage3d 内的 material 条目：`material 名 { albedo #.. roughness .. metalness .. }`。
pub(crate) struct MaterialV2Ast {
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// stage3d 内的 light 条目：`light 名 { direction(1,-1,-0.5) color #.. intensity .. }`。
pub(crate) struct LightV2Ast {
    /// 条目名（校验用；产物登记忽略）。
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// stage3d 块：`stage3d [名] { camera? sdf* material* light* }`。
pub(crate) struct Stage3dAst {
    /// 块名（暂不参与产物，保留给多 stage3d 扩展）。
    pub(crate) name: String,
    pub(crate) camera: Option<CameraV2Ast>,
    pub(crate) sdfs: Vec<SdfAst>,
    pub(crate) materials: Vec<MaterialV2Ast>,
    pub(crate) lights: Vec<LightV2Ast>,
    /// 块关键字所在行（保留给未来诊断）。
    pub(crate) line: usize,
}

/// layer 块：`layer <种类> [名] { 字段* }`（种类：text / particles / …，名缺省同种类）。
pub(crate) struct LayerAst {
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// sim 块：`sim <种类> [名] { 字段* force 块? noise 块? }`
/// （种类：cloth / fluid / rigid 等，语义最小可用，深语义留 Phase 2/3）。
pub(crate) struct SimAst {
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// compose 引用：`layer:text` / `sim:cloth` / `compose:main`。
#[derive(Debug, Clone)]
pub(crate) struct ComposeRef {
    /// 被引用块的种类前缀（layer / sim / compose）。
    pub(crate) kind: String,
    /// 被引用块的名字。
    pub(crate) name: String,
    /// 引用所在行（环检测 / 悬空引用报错定位）。
    pub(crate) line: usize,
}

/// compose 操作：`add layer:text layer:particles`（op = 加权合成等操作名）。
pub(crate) struct ComposeOpAst {
    /// 操作名（add / over / multiply …；codegen 映射到 MergeOp）。
    pub(crate) op: String,
    /// 操作数引用列表。
    pub(crate) refs: Vec<ComposeRef>,
    pub(crate) line: usize,
}

/// compose 块：`compose [名] { 操作* }`（名用于被其他 compose 引用）。
pub(crate) struct ComposeAst {
    pub(crate) name: String,
    pub(crate) ops: Vec<ComposeOpAst>,
    /// compose 关键字所在行（保留给未来诊断）。
    pub(crate) line: usize,
}

/// audio 块：`audio { music { bpm 120 gain 0.8 duck 12 } }`（开放字段，codegen 白名单）。
pub(crate) struct AudioAst {
    pub(crate) fields: Vec<NamedField>,
    /// audio 关键字所在行（保留给未来诊断）。
    pub(crate) line: usize,
}

/// note 块：`note "主题" { scale major root 60 bpm 120 bars 4 seed 7 }`。
///
/// 作曲系统 DSL 入口——音阶/根音/速度/小节数/种子驱动 [`prism_audio`] 的
/// 作曲引擎；字段沿用 [`NamedField`] 通用模型，codegen 白名单校验。
pub(crate) struct NoteAst {
    /// 块名（如 "主题"）。
    pub(crate) name: String,
    pub(crate) fields: Vec<NamedField>,
    /// note 关键字所在行。
    pub(crate) line: usize,
}

/// post 块：`post { bloom 0.3 grain 0.1 }`（开放字段，codegen 白名单）。
pub(crate) struct PostAst {
    pub(crate) fields: Vec<NamedField>,
    /// post 关键字所在行（保留给未来诊断）。
    pub(crate) line: usize,
}

/// transition 块：`transition "名" { from "场景A" to "场景B" duration 0.5s kind crossfade }`。
pub(crate) struct TransitionAst {
    pub(crate) name: String,
    /// 起始场景名 + 定义行号。
    pub(crate) from: Option<(String, usize)>,
    /// 结束场景名 + 定义行号。
    pub(crate) to: Option<(String, usize)>,
    pub(crate) duration: Option<TimeVal>,
    /// 过渡种类（crossfade / cut / …；缺省 crossfade）。
    pub(crate) kind: String,
    pub(crate) line: usize,
}

/// 单个关键帧：`key 0s 0 linear`（easing 可省略，缺省 linear）。
#[derive(Debug, Clone)]
pub(crate) struct KeyAst {
    pub(crate) time: TimeVal,
    pub(crate) value: f32,
    /// easing 名（linear / quadIn / easeOutCubic …；codegen 映射到 EasingKind）。
    pub(crate) easing: String,
    pub(crate) line: usize,
}

/// anim 块：`anim "目标路径" { key ... loop direct }`（命名动画，登记到产物）。
pub(crate) struct AnimAst {
    /// 动画目标（如 `ball.x`；Phase 2/3 接入属性绑定）。
    pub(crate) target: String,
    pub(crate) keys: Vec<KeyAst>,
    /// 循环模式（direct / loop / pingpong；缺省 direct）。
    pub(crate) loop_mode: Option<String>,
    pub(crate) line: usize,
}

/// expr 块：`expr "名" "t*10 + p*0.5"`（受限算术表达式，语法校验后登记到产物）。
pub(crate) struct ExprDefAst {
    pub(crate) name: String,
    pub(crate) body: String,
    pub(crate) line: usize,
}

/// import packs 声明：`import packs "名" { engine ">=0.4, <0.5" ... }`（块可省略）。
pub(crate) struct ImportPacksAst {
    pub(crate) name: String,
    /// 清单字段（engine 等；codegen 交由 packs.rs 校验）。
    pub(crate) fields: Vec<NamedField>,
    pub(crate) line: usize,
}

/// font 指令：`font "首选.ttf" fallback "次级.ttf", "末级.ttf"`。
///
/// 顶层**资产声明**（与 import packs 同层）：一条指令声明一个字体的回退链，
/// 多条指令按**出现顺序**串成完整回退链（V1.5-B）。`fallbacks` 可空
/// （= 一条只有一个槽位的链）；链末端恒由 `prism-text` 的程序化符号表兜底。
pub(crate) struct FontDecl {
    /// 首选字体文件路径（DSL 中声明的原文，相对路径按渲染进程工作目录解析）。
    pub(crate) path: String,
    /// 回退字体路径列表（按声明顺序逐级尝试）。
    pub(crate) fallbacks: Vec<String>,
    /// font 关键字所在行（装载失败诊断定位用）。
    pub(crate) line: usize,
}

/// v2 顶层文档：全块语法。
#[derive(Default)]
pub(crate) struct DocumentV2Ast {
    pub(crate) movie: Option<MovieAst>,
    /// 字体回退链声明（按出现顺序 = 链路顺序）。
    pub(crate) fonts: Vec<FontDecl>,
    /// v1 timeline 块复用（v2 中 timeline.fps 优先于 movie.fps）。
    pub(crate) timeline: Option<TimelineAst>,
    /// v2 theme 块 = v1 aesthetic 字段集（teal-orange / contrast / filmic）。
    pub(crate) theme: Option<AestheticAst>,
    pub(crate) imports: Vec<ImportPacksAst>,
    pub(crate) scenes: Vec<SceneV2Ast>,
    pub(crate) transitions: Vec<TransitionAst>,
    pub(crate) anims: Vec<AnimAst>,
    pub(crate) exprs: Vec<ExprDefAst>,
    pub(crate) output: Option<OutputAst>,
}
