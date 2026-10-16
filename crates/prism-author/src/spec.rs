//! 声明式规格：作者对外唯一需要构造的数据结构。
//!
//! 作者只写一份 [`ClipSpec`]（画幅 / 背景 / 若干轨道），再调用一次
//! [`crate::render_short`] 即可产出 MP4。所有"脏活"都收进渲染实现内部。

use crate::easing::EasingKind;
use prism_text::presets::CaptionStyle;
use std::sync::Arc;

/// 字体选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontKind {
    /// 粗体。
    Bold,
    /// 常规。
    Regular,
}

/// 背景预设。
#[derive(Debug, Clone)]
pub enum Background {
    /// 自定义垂直渐变（top → bottom，RGB）。
    Gradient {
        /// 顶部颜色（线性 RGB，0..1）。
        top: [f32; 3],
        /// 底部颜色（线性 RGB，0..1）。
        bottom: [f32; 3],
    },
    /// 项目默认：深藏青 → 靛蓝 + 冷青辉光（即原 demo 视觉）。
    DarkGlow,
    /// 极光预设：深色渐变上叠加青绿 / 青蓝水平流动色带（`Fbm` 噪声驱动）。
    Aurora,
    /// 扫描线预设：暗渐变 + 等距水平细扫描线（CRT 质感）。
    Scanline,
}

/// 逐字 kinetic 入场轨道。
#[derive(Debug, Clone)]
pub struct KineticText {
    /// 文本内容。
    pub content: String,
    /// 字体。
    pub font: FontKind,
    /// 字号（px）。
    pub size: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 入场起始秒。
    pub at: f32,
    /// 相邻字符入场间隔（秒）。
    pub stagger: f32,
    /// 单字入场时长（秒）。
    pub per_char: f32,
    /// 入场起始缩放（1.0 = 无缩放动画）。
    pub scale_from: f32,
    /// 入场纵向升起幅度（px）。
    pub rise_px: f32,
    /// 入场缓动。
    pub easing: EasingKind,
    /// 基线 y（像素）。
    pub y: f32,
    /// 全局淡入时长（秒）；0 = 无额外淡入（沿用逐字缓动）。
    pub in_sec: f32,
    /// 全局淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 水平锚点：文本布局盒的水平中心（像素）；`None` = 画幅水平中心
    /// `width * 0.5`（与 v1 行为逐位一致）。
    pub center_x: Option<f32>,
}

/// 静态（带字距、可淡入淡出）文字轨道。
#[derive(Debug, Clone)]
pub struct StaticText {
    /// 文本内容。
    pub content: String,
    /// 字体。
    pub font: FontKind,
    /// 字号（px）。
    pub size: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 字间追踪（px）。
    pub letter_spacing: f32,
    /// 基线 y（像素）。
    pub y: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 水平锚点：文本布局盒的水平中心（像素）；`None` = 画幅水平中心
    /// `width * 0.5`（与 v1 行为逐位一致）。
    pub center_x: Option<f32>,
}

/// 打字机文字轨道：逐字显示 + 可选闪烁光标。
///
/// 渲染逻辑（见 `render::draw_typewriter`）：可见字符数
/// `n = floor((t - at) * cps).clamp(0, glyphs.len())`，只画前 `n` 个字形；
/// 光标在最后一个可见字形之后按 `fract(t * cursor_blink_hz)` 闪烁。
#[derive(Debug, Clone)]
pub struct TypewriterText {
    /// 文本内容。
    pub content: String,
    /// 字体。
    pub font: FontKind,
    /// 字号（px）。
    pub size: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 出现起始秒。
    pub at: f32,
    /// 基线 y（像素）。
    pub y: f32,
    /// 打字速度（字符/秒；建议 8..20，量级约 12；`<= 0` 视为立即全文显示）。
    pub cps: f32,
    /// 是否显示闪烁光标。
    pub cursor: bool,
    /// 光标闪烁频率（Hz；`<= 0` 时常亮）。
    pub cursor_blink_hz: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 字间追踪（px）。
    pub letter_spacing: f32,
    /// 水平锚点：文本布局盒的水平中心（像素）；`None` = 画幅水平中心
    /// `width * 0.5`（与 v1 行为逐位一致）。
    pub center_x: Option<f32>,
}

/// 装饰线（矩形色条，可淡入淡出）。
#[derive(Debug, Clone)]
pub struct AccentLine {
    /// 中心 x（像素）。
    pub center_x: f32,
    /// 顶部 y（像素）。
    pub y: f32,
    /// 宽（像素）。
    pub w: f32,
    /// 高（像素）。
    pub h: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
}

/// 收尾字幕条轨道（复用 `prism-text::presets::render_caption`）。
#[derive(Debug, Clone)]
pub struct Caption {
    /// 文本内容。
    pub content: String,
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 字幕条样式。
    pub style: CaptionStyle,
}

/// 叠加层图源：纯色面板 / 垂直渐变面板 / 预渲染 RGBA 缓冲。
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// 纯色面板（直通 alpha 线性 RGBA，0..1）。
    Solid([f32; 4]),
    /// 垂直渐变面板（top → bottom，直通 alpha 线性 RGBA，0..1）。
    Gradient {
        /// 顶部颜色。
        top: [f32; 4],
        /// 底部颜色。
        bottom: [f32; 4],
    },
    /// 预渲染 RGBA8 缓冲（行主序，`data.len() == width·height·4`）。
    Buffer {
        /// 像素数据（RGBA8，行主序）。
        data: Vec<u8>,
        /// 宽（像素）。
        width: u32,
        /// 高（像素）。
        height: u32,
    },
}

/// 图像叠加层轨道：用一张 RGBA 缓冲（或纯色 / 渐变面板）做叠加，支持
/// 淡入淡出与位移（入场由 `move_from` 缓动归零，出场退回 `move_from`）。
#[derive(Debug, Clone)]
pub struct ImageTrack {
    /// 图源。
    pub source: ImageSource,
    /// 左上角 x（像素，未含位移偏移）。
    pub x: f32,
    /// 左上角 y（像素，未含位移偏移）。
    pub y: f32,
    /// 宽（像素）。
    pub w: f32,
    /// 高（像素）。
    pub h: f32,
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 入场起始位移（设备像素）；淡入时由该位移缓动归零，淡出时退回。
    pub move_from: [f32; 2],
}

/// 图元种类（用 `prism-raster` 的 `fill_path` 光栅化）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeKind {
    /// 直角矩形。
    Rect,
    /// 圆角矩形。
    RoundRect {
        /// 圆角半径（px）。
        radius: f32,
    },
    /// 椭圆。
    Ellipse,
}

/// 矢量图元叠加轨道（圆角矩形 / 椭圆等）。
#[derive(Debug, Clone)]
pub struct ShapeTrack {
    /// 图元种类。
    pub kind: ShapeKind,
    /// 包围盒左上角 x（像素）。
    pub x: f32,
    /// 包围盒左上角 y（像素）。
    pub y: f32,
    /// 包围盒宽（像素）。
    pub w: f32,
    /// 包围盒高（像素）。
    pub h: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
}

/// 折线路径轨道：自由曲线 / 折线的描边或半透明面积填充。
///
/// 与 [`ShapeTrack`] 同为声明式矢量叠加层，但不受图元种类限制——曲线由
/// 调用方采样成折线后交给本轨道（如 `y = sin(x)/x` 采样 ~48 点）。渲染
/// 逻辑（见 `render::draw_path_track`）：
/// - `fill = false`：按 `width` 逐段描边（`prism-raster::draw_line`）；
/// - `fill = true`：自动闭合整条路径并填充（`prism-raster::fill_path`）；
/// - `draw_sec > 0` 时描边带绘制进度动画（只画到折线总长的
///   `p = clamp((t-at)/draw_sec, 0, 1)` 比例，最后一段可截断）；
/// - `at` / `in_sec` / `out_sec` 淡入淡出语义与其它轨道一致。
#[derive(Debug, Clone)]
pub struct PathTrack {
    /// 折线顶点（设备像素，至少 2 个点；少于 2 个点渲染时跳过该轨道）。
    pub points: Vec<[f32; 2]>,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 描边宽度（px；`fill = false` 时生效）。
    pub width: f32,
    /// `true` = 闭合路径并填充（半透明面积）；`false` = 描边。
    pub fill: bool,
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 绘制进度动画时长（秒；0 = 立即全显，仅描边模式生效）。
    pub draw_sec: f32,
    /// 是否闭合（描边时首尾相连画闭合段；填充时自动闭合）。
    pub closed: bool,
}

/// 时间参数化折线变形轨道（Morph）：从折线 `from` 连续变形到折线 `to`，
/// 用于数学动画（割线收敛成切线、公式符号变形等）。
///
/// 帧内求值（见 `render::draw_morph`）：按场景时间 `t` 计算
/// `u = clamp((t - at) / dur, 0, 1)`，`e = ease(u)`，逐点
/// `lerp(from[i], to[i], e)` 得到当前折线；随后进入与 [`PathTrack`]
/// **完全相同的光栅核心**（`render::paint_polyline`）——描边 / 填充 /
/// `closed` / `draw_sec` 绘制进度 / `at` / `in_sec` / `out_sec` /
/// `out_at` 淡入淡出语义全部与 [`PathTrack`] 一致。
///
/// 点数不等时按索引配对，只渲染 `min(from.len(), to.len())` 个点（多出的
/// 点忽略）。
///
/// 字段说明：本仓库无 `PathStyle` / `FillStyle` / `FadeSpec` 类型，按设计
/// 指示照抄 [`PathTrack`] 的现有字段——`color` + `width` 即描边样式，
/// `fill: bool` 即填充开关（填充色同 `color`），`at` / `in_sec` /
/// `out_sec` / `out_at` 即淡入淡出；`dur <= 0` 时 `t >= at` 直接取 `to`。
/// 折线家族（[`PathTrack`] / [`DynPathTrack`]）均无 `center_x`（几何为
/// 绝对设备像素，文本变体的水平锚点语义不适用），故本结构亦不设该字段。
#[derive(Debug, Clone)]
pub struct MorphSpec {
    /// 起点折线（设备像素）。
    pub from: Vec<[f32; 2]>,
    /// 终点折线（设备像素；与 `from` 同点数为正常情形）。
    pub to: Vec<[f32; 2]>,
    /// 颜色（直通 alpha 线性 RGBA，0..1；描边与填充共用）。
    pub color: [f32; 4],
    /// 描边宽度（px；`fill = false` 时生效）。
    pub width: f32,
    /// `true` = 闭合路径并填充（半透明面积）；`false` = 描边。
    pub fill: bool,
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到
    /// `at`。语义与 [`PathTrack::out_at`] 一致。
    pub out_at: Option<f32>,
    /// 绘制进度动画时长（秒；0 = 立即全显，仅描边模式生效；语义与
    /// [`PathTrack::draw_sec`] 一致）。
    pub draw_sec: f32,
    /// 是否闭合（描边时首尾相连画闭合段；填充时自动闭合）。
    pub closed: bool,
    /// 变形缓动（作用于进度 `u`）。
    pub ease: EasingKind,
    /// 变形时长（秒；`<= 0` 时 `t >= at` 直接显示 `to`，否则显示 `from`）。
    pub dur: f32,
}

/// 动态几何闭包类型：场景绝对时间 `t`（秒）→ 设备像素折线。
///
/// - 必须是纯函数（禁止内部可变状态、时钟、随机源）；
/// - `Fn + Send + Sync` 保证可并行渲染时跨线程安全；
/// - 用 [`Arc`] 装载：`Track` 的 `derive(Clone)` 依赖 `Arc<dyn Fn>` 的
///   `Clone`（`Arc<T: ?Sized>` 即 `Clone`）；`Debug` 由 [`DynPathTrack`]
///   手工实现（`geom` 打印为 `<dyn geom>`），从而 `Track` / `ClipSpec` 的
///   `#[derive(Debug, Clone)]` 不破。
pub type GeomFn = Arc<dyn Fn(f32) -> Vec<[f32; 2]> + Send + Sync>;

/// 计数器读数的时间函数（v3-E「f∘ease 复合」）：场景绝对时间 `t`（秒）
/// → 读数标量。
///
/// 与 [`GeomFn`] 同纪律：纯函数（禁内部可变状态 / 时钟 / 随机源）、
/// `Fn + Send + Sync`（并行渲染跨线程安全）、[`Arc`] 装载（`Clone` 依赖）。
///
/// 存在意义：单缓动 Counter 只能表达 `from + (to−from)·easing(u)`——**一重**
/// 函数；而动点沿曲线运动需要 `y = f(x(t))` 的**复合**追踪（x 走缓动、
/// y 由 x 经 f 得出）。把该复合显式化为时间函数后，HUD 读数与
/// [`DynPathTrack`] 几何可派生自**同一个纯函数**（同参同时钟），「读数
/// 恒等」铁律（架构 §7.3）从「近似恒等」升级为「逐位恒等」。
pub type ValueFn = Arc<dyn Fn(f32) -> f32 + Send + Sync>;

/// 时间参数化折线轨道（E1）：每帧求值 `geom(t)` 后进入与 [`PathTrack`]
/// **完全相同的光栅核心**（`render::paint_polyline`）。
///
/// 不变量（渲染层强制，违约跳过轨道不 panic，与 [`PathTrack`] 既有纪律
/// 一致）：
/// 1. `geom(t)` 返回点数 < 2 → 跳过；
/// 2. 任一点坐标非有限（NaN/Inf）→ 跳过；
/// 3. `env <= 0`（at 之前 / 淡出完毕）→ 跳过；
/// 4. `fill = true` 时忽略 `width` / `draw_sec` / `closed`（填充自动闭合）。
#[derive(Clone)]
pub struct DynPathTrack {
    /// 几何发生器；参数为场景绝对时间 `t`（秒，非 `t - at`）。
    pub geom: GeomFn,
    /// 颜色（直通 alpha 线性 RGBA，0..1；渲染层乘 env 信封）。
    pub color: [f32; 4],
    /// 描边宽度（px；`fill = false` 且 `width <= 0` 时跳过描边）。
    pub width: f32,
    /// `true` = 闭合路径并填充（半透明面积）；`false` = 描边。
    pub fill: bool,
    /// 出现起始秒。
    pub at: f32,
    /// 淡入时长（秒）；0 = 无额外淡入。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出（锚 duration 末，同 [`PathTrack`]）。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 绘制进度动画时长（秒；0 = 立即全显，仅描边模式生效）。
    pub draw_sec: f32,
    /// 是否闭合（描边时首尾相连画闭合段；填充时自动闭合）。
    pub closed: bool,
}

impl std::fmt::Debug for DynPathTrack {
    /// 手工 `Debug`：打印除 `geom` 外全部字段（`dyn Fn` 无 `Debug`），
    /// `geom` 显示为 `<dyn geom>`。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynPathTrack")
            .field("geom", &"<dyn geom>")
            .field("color", &self.color)
            .field("width", &self.width)
            .field("fill", &self.fill)
            .field("at", &self.at)
            .field("in_sec", &self.in_sec)
            .field("out_sec", &self.out_sec)
            .field("draw_sec", &self.draw_sec)
            .field("closed", &self.closed)
            .finish()
    }
}

/// 实时数值读数轨道（E2，Manif 式教学 HUD）：逐帧把
/// `from + (to - from) * easing(u)` 格式化为 ASCII 定点文本，复用
/// [`render::draw_static`] 既有文本管线（零新光栅代码、零新字体风险）。
///
/// 读数只含拉丁字母 / 数字 / `.` / `-` / 空格（PrismSansSC 字体内齐全），
/// **不使用科学计数法**（`{:.N}` 定点格式）；`decimals > 6` 钳 6。
///
/// 「读数恒等」工程约定（架构 §7.3）：`(from, to, dur_sec, easing)` 必须与
/// 对应 [`DynPathTrack`] 的几何参数派生自**同一个纯函数**，禁止两处各写
/// 常数——这是「HUD 读数与画面恒等」的工程保证。
///
/// # v3-E：`value_at` 输出映射（f∘ease）
///
/// [`CounterText::value_at`] 为 `Some(f)` 时，读数改写为 `f(t)`（场景绝对
/// 秒，与 [`DynPathTrack::geom`] 同一时钟）——动点沿曲线 `f(x(t))` 运动时，
/// HUD 与几何可共享同一纯函数，逐位恒等；此时 `from` / `to` / `dur_sec` /
/// `easing` 退化为**设计包络备案**（窗口与量程文档），不参与求值。
#[derive(Clone)]
pub struct CounterText {
    /// 文本前缀，如 `"m = "` / `"S = "` / `"n = "`。
    pub prefix: String,
    /// 滚动起始值。
    pub from: f32,
    /// 滚动终止值。
    pub to: f32,
    /// 起始秒。
    pub at: f32,
    /// 滚动时长（秒；`<= 0` 时 `t >= at` 直接显示 `to`）。
    pub dur_sec: f32,
    /// 缓动（与对应几何动画同参，保证「读数恒等」）。
    pub easing: EasingKind,
    /// 小数位（0..=6；>6 钳 6）。
    pub decimals: u32,
    /// 字体。
    pub font: FontKind,
    /// 字号（px）。
    pub size: f32,
    /// 颜色（直通 alpha 线性 RGBA，0..1）。
    pub color: [f32; 4],
    /// 水平锚点（E3 同语义）；`None` = 画幅中心。
    pub center_x: Option<f32>,
    /// 基线 y（像素）。
    pub y: f32,
    /// 字间追踪（px）。
    pub letter_spacing: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
    /// 读数时间函数（v3-E f∘ease；`None` = 沿用 `from`/`to`/`easing` 单缓动）。
    ///
    /// `Some(f)` ⇒ 读数 = `f(t)`（场景绝对秒，与 [`DynPathTrack::geom`] 同
    /// 时钟；`f` 须为纯函数且返回有限值，`-0.0` 由渲染层归一）。此时
    /// `from` / `to` / `dur_sec` / `easing` 不参与求值（仅设计包络备案）。
    pub value_at: Option<ValueFn>,
}

impl std::fmt::Debug for CounterText {
    /// 手工 `Debug`：打印除 `value_at` 外全部字段（`dyn Fn` 无 `Debug`），
    /// `value_at` 显示为 `<dyn value_at>`（与 [`DynPathTrack`] 的 `geom`
    /// 同款处理，保住 `Track` / `ClipSpec` 的 `derive(Debug)`）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CounterText")
            .field("prefix", &self.prefix)
            .field("from", &self.from)
            .field("to", &self.to)
            .field("at", &self.at)
            .field("dur_sec", &self.dur_sec)
            .field("easing", &self.easing)
            .field("decimals", &self.decimals)
            .field("font", &self.font)
            .field("size", &self.size)
            .field("color", &self.color)
            .field("center_x", &self.center_x)
            .field("y", &self.y)
            .field("letter_spacing", &self.letter_spacing)
            .field("in_sec", &self.in_sec)
            .field("out_sec", &self.out_sec)
            .field("out_at", &self.out_at)
            .field("value_at", &self.value_at.as_ref().map(|_| "<dyn value_at>"))
            .finish()
    }
}

/// 帧序列轨道（E4）：RGBA8 帧序列 + 源帧率，按场景时间取帧后经
/// [`render::blit_rgba8`] 合成（与 [`ImageTrack`] 的 Buffer 分支共用
/// blit 核心，零拷贝借入）。
///
/// 取帧公式：`idx = clamp(floor((t - at) * src_fps), 0, frames.len()-1)`
/// （渲染层以帧索引整数算术实现，避免浮点边界误差，见
/// [`render::draw_sequence`]）。src_fps=15、画布 fps=30 ⇒ 每源帧连续显示
/// 2 个画布帧。
///
/// 不变量（违约跳过轨道不 panic）：`frames` 空 / 帧字节数 ≠
/// `width·height·4` / `width|height ≠ 画幅` / `src_fps ≤ 0` ⇒ 跳过。
#[derive(Debug, Clone)]
pub struct SequenceTrack {
    /// RGBA8 行主序帧序列，每帧 `width·height·4` 字节。
    pub frames: Vec<Vec<u8>>,
    /// 源帧宽（像素；必须等于画幅宽，不符则该轨跳过）。
    pub width: u32,
    /// 源帧高（像素；必须等于画幅高，不符则该轨跳过）。
    pub height: u32,
    /// 源帧率（帧/秒；>0 生效，<=0 跳过）。
    pub src_fps: u32,
    /// 场景内起始秒。
    pub at: f32,
    /// 淡入时长（秒）。
    pub in_sec: f32,
    /// 淡出时长（秒）；0 = 不淡出。
    pub out_sec: f32,
    /// 淡出绝对锚点（场景秒）；`Some(a)` 时淡出自 `a` 起（`out_sec` 秒内
    /// 平滑归 0，`out_sec = 0` 时 `a` 处硬切），支撑分拍阶跃 staging
    /// （同一位置多条顺序轨轮流显现）；`None` = 缺省锚定片尾
    /// （`duration - out_sec`）。非有限值按 `None` 处理；早于 `at` 时钳到 `at`。
    pub out_at: Option<f32>,
}

/// 轨道枚举：一支短片由若干轨道构成。
#[derive(Debug, Clone)]
pub enum Track {
    /// 逐字 kinetic 入场文字。
    Kinetic(KineticText),
    /// 静态文字。
    Static(StaticText),
    /// 打字机文字（逐字显示 + 可选闪烁光标）。
    Typewriter(TypewriterText),
    /// 装饰线。
    Accent(AccentLine),
    /// 字幕条。
    Caption(Caption),
    /// 图像 / 面板叠加层。
    Image(ImageTrack),
    /// 矢量图元叠加层。
    Shape(ShapeTrack),
    /// 折线路径叠加层（自由曲线描边 / 面积填充，带绘制进度动画）。
    Path(PathTrack),
    /// 时间参数化折线变形叠加层（Morph：从 `from` 连续变形到 `to`，
    /// 与 [`PathTrack`] 共用光栅核心）。
    Morph(MorphSpec),
    /// 时间参数化折线叠加层（每帧求值 `geom(t)`，与 [`PathTrack`] 共用
    /// 光栅核心）。
    DynPath(DynPathTrack),
    /// 实时数值读数（ASCII 定点文本 HUD）。
    Counter(CounterText),
    /// 帧序列叠加层（按场景时间从源帧序列取帧）。
    Sequence(SequenceTrack),
}

/// 短片声明式规格——作者对外唯一需要构造的结构。
#[derive(Debug, Clone)]
pub struct ClipSpec {
    /// 画幅宽（像素）。
    pub width: u32,
    /// 画幅高（像素）。
    pub height: u32,
    /// 帧率（帧/秒）。
    pub fps: u32,
    /// 时长（秒）。
    pub duration: f32,
    /// 背景预设。
    pub background: Background,
    /// JPEG 质量（1..=100）。
    pub jpeg_quality: u8,
    /// 轨道列表（任意类型，可空）。
    pub tracks: Vec<Track>,
}

impl ClipSpec {
    /// 帧数（向上取整，保证末秒不丢帧；至少 1 帧）。
    #[inline]
    pub fn frame_count(&self) -> usize {
        (self.fps as f32 * self.duration).ceil().max(1.0) as usize
    }
}

// ---- 多场景拼接：转场 ----

/// 擦除方向（[`Transition::Wipe`] 用）。
///
/// 语义统一描述为"新片段（b）的入场侧与分界线推进方向"：
/// - [`WipeDir::Left`]：b 自左侧入场，分界线向右推进；
/// - [`WipeDir::Right`]：b 自右侧入场，分界线向左推进；
/// - [`WipeDir::Up`]：b 自底部入场，分界线向上推进；
/// - [`WipeDir::Down`]：b 自顶部入场，分界线向下推进。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDir {
    /// b 自左侧入场（分界线向右推进）。
    Left,
    /// b 自右侧入场（分界线向左推进）。
    Right,
    /// b 自底部入场（分界线向上推进）。
    Up,
    /// b 自顶部入场（分界线向下推进）。
    Down,
}

/// 场景间转场类型（**纯后处理**：只在帧序列层做混合，不触碰渲染核心）。
///
/// 由 [`crate::render_short_concat`] / [`crate::render_short_concat_avi`]
/// 消费：相邻两段子片的重叠区内逐帧按本枚举混合。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transition {
    /// 交叉溶解：重叠帧 RGB 逐通道线性插值（`k = (i+1)/(t+1)`），
    /// alpha 恒 255，结果 clamp 0..255。
    Fade,
    /// 线性擦除：按方向用位置阈值逐像素选源（带 2px 平滑过渡带，
    /// 避免硬边），见 [`concat_with_transition`](crate::render) 的文档。
    Wipe {
        /// 擦除方向。
        dir: WipeDir,
    },
    /// 随机像素溶解（v3-E；语义对齐 `prism-postfx` 的 `Dissolve`——
    /// 宪章 C4 对照 #4「剪映 / iMovie Dissolve」）：逐像素以
    /// `hash2(seed, x, y) < k` 决定显影新片段 `b`，否则保留旧片段 `a`。
    ///
    /// 与 [`Transition::Fade`] 的差别：Fade 是全帧线性插值（每像素同时
    /// 含 A/B，中间帧是「稀释的叠影」）；Dissolve 是像素级阈值显影
    /// （每像素非 A 即 B，中间帧是「A/B 像素并存的颗粒过渡」）——
    /// 后者是 NLE 语境「溶解转场」的标准形态，颗粒感还能自然掩盖
    /// 两场景画风/噪点差异（如 S6→S7 的 2D→3D 画质断层）。
    ///
    /// 端点精确：`hash2 ∈ [0,1)`，`k=0` 时无像素显影（逐位等于 `a`）、
    /// `k=1` 时全部显影（逐位等于 `b`）。
    Dissolve {
        /// 显影种子（决定哪些像素先显影的确定性图案；同 seed 同图案）。
        seed: u64,
    },
    /// 推移（v3-E）：两帧作为**同一画面**整体平移——`a` 退出一侧、
    /// `b` 从对侧入场（硬接缝，无平滑带，保端点逐位精确）。
    ///
    /// 语义与 [`WipeDir`] 对齐（描述 `b` 的入场侧）：`b` 自 `dir` 侧
    /// 入场，`a` 从对侧退出。推移带明确的方向性（段落推进感），适合
    /// 视觉语言连续的相邻章节（S4→S5 / S5→S6 的计算场景链）。
    Push {
        /// `b` 的入场侧（`a` 从对侧退出）。
        dir: WipeDir,
    },
}
