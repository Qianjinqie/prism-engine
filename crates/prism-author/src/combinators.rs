//! 组合子库：`prism-author` 侧的 Rust API 构造入口。
//!
//! # 分层
//! - **展开函数**（`to_node_graph` 的本体）落在 `prism-dsl::combinators`：
//!   组合子注册面（`prism_dsl::COMBINATOR_TABLE`：name → 展开 fn）与
//!   codegen 同 crate，fn 指针可直接登记；且展开只碰 prism-core 数据
//!   （节点图 / 通道），不依赖任何渲染或 IO 能力；
//! - **本模块**是薄封装：把每个组合子的语义参数收拢为类型化建造者
//!   （`TitleCard::new(...).fade(...)`），内部转成 `CombinatorArgs` 后
//!   经注册面展开。DSL v2 语法当前版本不加新指令（糖化语法留待后续支持），故本层
//!   是组合子当前唯一的用户入口。
//!
//! # 依赖方向
//! `prism-author → prism-dsl` 合法：prism-dsl 只依赖 {prism-math,
//! prism-core, prism-anim}，无一环回 prism-author（Cargo.toml 已注明）。
//!
//! # 用法
//! ```
//! use prism_author::TitleCard;
//! use prism_gpu::CpuRefBackend;
//!
//! // 1s @ 8fps = 8 帧：组合子展开 → v2 编译产物 → 成片直达路径出帧。
//! let card = TitleCard::new("你好，Prism")
//!     .duration(1.0)
//!     .frame(96, 54)
//!     .fps(8)
//!     .to_node_graph()
//!     .expect("标题卡应展开");
//! let movie = card.into_compile_output();
//! let deliverable =
//!     prism_render::render_movie(&movie, &mut CpuRefBackend).expect("成片应可渲染");
//! assert_eq!(deliverable.frame_count(), 8);
//! ```

use prism_dsl::{
    expand_ken_burns, expand_lower_third, expand_morph_between, expand_split_screen,
    expand_title_card, CombinatorArgs, CombinatorError, NodeGraph,
};

/// 经组合子注册面按名展开（演示 `COMBINATOR_TABLE` 的查表入口；DSL 糖化
/// 指令糖化语法落地后直接复用它）。
///
/// # Errors
/// 未登记的组合子名 → `C5000`（三段式，列出全部已登记名）。
pub fn expand_by_name(name: &str, args: &CombinatorArgs) -> Result<NodeGraph, CombinatorError> {
    let entry = prism_dsl::COMBINATOR_TABLE
        .iter()
        .find(|(n, _)| *n == name)
        .ok_or_else(|| CombinatorError {
            code: "C5000",
            location: format!("combinator '{name}'"),
            problem: "组合子未登记".to_string(),
            fix: format!(
                "已登记组合子：{}",
                prism_dsl::COMBINATOR_TABLE
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join("、")
            ),
        })?;
    (entry.1)(args)
}

/// 基片子片：单球体 3D 场景（转导 `prism_dsl::sphere_fragment`，作为
/// `SplitScreen` / `MorphBetween` / `KenBurns` 的常见输入）。
pub use prism_dsl::sphere_fragment as sphere_scene;

// ---------------------------------------------------------------------------
// 组合子一：TitleCard（标题卡）
// ---------------------------------------------------------------------------

/// 标题卡建造者：居中大标题 + 淡入淡出。
#[derive(Debug, Clone)]
pub struct TitleCard {
    content: String,
    title: String,
    size: Option<f32>,
    color: (f32, f32, f32),
    in_sec: f32,
    out_sec: f32,
    duration: f32,
    width: u32,
    height: u32,
    fps: u32,
}

impl TitleCard {
    /// 以标题文本构造（其余取组合子语义缺省）。
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            title: "title card".to_string(),
            size: None,
            color: (1.0, 1.0, 1.0),
            in_sec: 0.4,
            out_sec: 0.4,
            duration: 2.0,
            width: 480,
            height: 270,
            fps: 12,
        }
    }

    /// 字号（px；缺省画幅高的 14%）。
    pub fn size(mut self, px: f32) -> Self {
        self.size = Some(px);
        self
    }

    /// 文本色（0..1 线性 RGB；缺省白）。
    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = (r, g, b);
        self
    }

    /// 淡入 / 淡出时长（秒）。
    pub fn fade(mut self, in_sec: f32, out_sec: f32) -> Self {
        self.in_sec = in_sec;
        self.out_sec = out_sec;
        self
    }

    /// 成片时长（秒）。
    pub fn duration(mut self, sec: f32) -> Self {
        self.duration = sec;
        self
    }

    /// 画幅（px）。
    pub fn frame(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 帧率（fps）。
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// 影片标题（产物元数据）。
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// 展开为等价 v2 节点图。
    pub fn to_node_graph(&self) -> Result<NodeGraph, CombinatorError> {
        let mut params = std::collections::BTreeMap::new();
        params.insert("content".to_string(), prism_core::node::ParamValue::Str(self.content.clone()));
        params.insert("title".to_string(), prism_core::node::ParamValue::Str(self.title.clone()));
        params.insert("color_r".to_string(), prism_core::node::ParamValue::F32(self.color.0));
        params.insert("color_g".to_string(), prism_core::node::ParamValue::F32(self.color.1));
        params.insert("color_b".to_string(), prism_core::node::ParamValue::F32(self.color.2));
        params.insert("in_sec".to_string(), prism_core::node::ParamValue::F32(self.in_sec));
        params.insert("out_sec".to_string(), prism_core::node::ParamValue::F32(self.out_sec));
        if let Some(s) = self.size {
            params.insert("size".to_string(), prism_core::node::ParamValue::F32(s));
        }
        let args = CombinatorArgs {
            duration: self.duration,
            fps: self.fps,
            width: self.width,
            height: self.height,
            params,
            sources: Vec::new(),
        };
        expand_title_card(&args)
    }
}

// ---------------------------------------------------------------------------
// 组合子二：LowerThird（左下三分之一条）
// ---------------------------------------------------------------------------

/// lower third 建造者：左下方主标题 + 副标题，共用淡入淡出。
#[derive(Debug, Clone)]
pub struct LowerThird {
    title: String,
    subtitle: String,
    size: Option<f32>,
    sub_size: Option<f32>,
    x: Option<f32>,
    y: Option<f32>,
    color: (f32, f32, f32),
    in_sec: f32,
    out_sec: f32,
    duration: f32,
    width: u32,
    height: u32,
    fps: u32,
}

impl LowerThird {
    /// 以主标题构造（副标题缺省空）。
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            subtitle: String::new(),
            size: None,
            sub_size: None,
            x: None,
            y: None,
            color: (1.0, 0.96, 0.90),
            in_sec: 0.35,
            out_sec: 0.35,
            duration: 2.5,
            width: 480,
            height: 270,
            fps: 12,
        }
    }

    /// 副标题（空串 = 不建副标题节点）。
    pub fn subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = subtitle.into();
        self
    }

    /// 主标题字号（px；缺省画幅高的 8.5%）。
    pub fn size(mut self, px: f32) -> Self {
        self.size = Some(px);
        self
    }

    /// 副标题字号（px；缺省主标题的 50%）。
    pub fn sub_size(mut self, px: f32) -> Self {
        self.sub_size = Some(px);
        self
    }

    /// 文本块左上角（px；缺省 `width×0.08`, `height×0.70`）。
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.x = Some(x);
        self.y = Some(y);
        self
    }

    /// 主标题色（0..1 线性 RGB）。
    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.color = (r, g, b);
        self
    }

    /// 淡入 / 淡出时长（秒）。
    pub fn fade(mut self, in_sec: f32, out_sec: f32) -> Self {
        self.in_sec = in_sec;
        self.out_sec = out_sec;
        self
    }

    /// 成片时长（秒）。
    pub fn duration(mut self, sec: f32) -> Self {
        self.duration = sec;
        self
    }

    /// 画幅（px）。
    pub fn frame(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 帧率（fps）。
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// 展开为等价 v2 节点图。
    pub fn to_node_graph(&self) -> Result<NodeGraph, CombinatorError> {
        let mut params = std::collections::BTreeMap::new();
        params.insert("title".to_string(), prism_core::node::ParamValue::Str(self.title.clone()));
        params.insert("subtitle".to_string(), prism_core::node::ParamValue::Str(self.subtitle.clone()));
        params.insert("color_r".to_string(), prism_core::node::ParamValue::F32(self.color.0));
        params.insert("color_g".to_string(), prism_core::node::ParamValue::F32(self.color.1));
        params.insert("color_b".to_string(), prism_core::node::ParamValue::F32(self.color.2));
        params.insert("in_sec".to_string(), prism_core::node::ParamValue::F32(self.in_sec));
        params.insert("out_sec".to_string(), prism_core::node::ParamValue::F32(self.out_sec));
        if let Some(s) = self.size {
            params.insert("size".to_string(), prism_core::node::ParamValue::F32(s));
        }
        if let Some(s) = self.sub_size {
            params.insert("sub_size".to_string(), prism_core::node::ParamValue::F32(s));
        }
        if let Some(x) = self.x {
            params.insert("x".to_string(), prism_core::node::ParamValue::F32(x));
        }
        if let Some(y) = self.y {
            params.insert("y".to_string(), prism_core::node::ParamValue::F32(y));
        }
        let args = CombinatorArgs {
            duration: self.duration,
            fps: self.fps,
            width: self.width,
            height: self.height,
            params,
            sources: Vec::new(),
        };
        expand_lower_third(&args)
    }
}

// ---------------------------------------------------------------------------
// 组合子三：SplitScreen（左右分屏）
// ---------------------------------------------------------------------------

/// 分屏建造者：`left` / `right` 两个子片同程并行，render 侧按半幅合成。
#[derive(Debug, Clone)]
pub struct SplitScreen {
    left: NodeGraph,
    right: NodeGraph,
    duration: f32,
    fps: u32,
    width: u32,
    height: u32,
}

impl SplitScreen {
    /// 以左右子片构造（画幅 / 帧率 / 时长缺省继承左子片）。
    pub fn new(left: NodeGraph, right: NodeGraph) -> Self {
        Self {
            left,
            right,
            duration: 0.0,
            fps: 0,
            width: 0,
            height: 0,
        }
    }

    /// 成片时长（秒；缺省继承左子片）。
    pub fn duration(mut self, sec: f32) -> Self {
        self.duration = sec;
        self
    }

    /// 帧率（缺省继承左子片）。
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// 画幅（缺省继承左子片）。
    pub fn frame(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 展开为等价 v2 节点图（两场景同程 + 左右半幅合成语义）。
    pub fn to_node_graph(&self) -> Result<NodeGraph, CombinatorError> {
        let args = CombinatorArgs {
            duration: self.duration,
            fps: self.fps,
            width: self.width,
            height: self.height,
            params: std::collections::BTreeMap::new(),
            sources: vec![self.left.clone(), self.right.clone()],
        };
        expand_split_screen(&args)
    }
}

// ---------------------------------------------------------------------------
// 组合子四：KenBurns（缓慢推拉 + 摇移）
// ---------------------------------------------------------------------------

/// Ken Burns 建造者：对底片套用相机轨道动画（半径推拉 + 中心摇移）。
#[derive(Debug, Clone)]
pub struct KenBurns {
    base: NodeGraph,
    from_radius: f32,
    to_radius: f32,
    center_from: (f32, f32, f32),
    center_to: (f32, f32, f32),
    fov_deg: Option<f32>,
    easing: String,
    duration: f32,
    fps: u32,
    width: u32,
    height: u32,
}

impl KenBurns {
    /// 以底片构造（缺省 8 → 4.5 推拉、无摇移、quadInOut）。
    pub fn new(base: NodeGraph) -> Self {
        Self {
            base,
            from_radius: 8.0,
            to_radius: 4.5,
            center_from: (0.0, 0.0, 0.0),
            center_to: (0.0, 0.0, 0.0),
            fov_deg: None,
            easing: "quadInOut".to_string(),
            duration: 0.0,
            fps: 0,
            width: 0,
            height: 0,
        }
    }

    /// 轨道半径起止（推拉幅度）。
    pub fn zoom(mut self, from_radius: f32, to_radius: f32) -> Self {
        self.from_radius = from_radius;
        self.to_radius = to_radius;
        self
    }

    /// 轨道中心起止（摇移幅度）。
    pub fn pan(mut self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Self {
        self.center_from = from;
        self.center_to = to;
        self
    }

    /// 视场角（度；缺省沿用底片相机的 fov，无相机时 45°）。
    pub fn fov(mut self, deg: f32) -> Self {
        self.fov_deg = Some(deg);
        self
    }

    /// 缓动名（31 种，同 DSL easing 表）。
    pub fn easing(mut self, name: impl Into<String>) -> Self {
        self.easing = name.into();
        self
    }

    /// 成片时长（秒；缺省继承底片）。
    pub fn duration(mut self, sec: f32) -> Self {
        self.duration = sec;
        self
    }

    /// 帧率（缺省继承底片）。
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// 画幅（缺省继承底片）。
    pub fn frame(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 展开为等价 v2 节点图（底片相机被接管，半径 / 中心绑动画通道）。
    pub fn to_node_graph(&self) -> Result<NodeGraph, CombinatorError> {
        let mut params = std::collections::BTreeMap::new();
        params.insert("from_radius".to_string(), prism_core::node::ParamValue::F32(self.from_radius));
        params.insert("to_radius".to_string(), prism_core::node::ParamValue::F32(self.to_radius));
        params.insert("center_from_x".to_string(), prism_core::node::ParamValue::F32(self.center_from.0));
        params.insert("center_from_y".to_string(), prism_core::node::ParamValue::F32(self.center_from.1));
        params.insert("center_from_z".to_string(), prism_core::node::ParamValue::F32(self.center_from.2));
        params.insert("center_to_x".to_string(), prism_core::node::ParamValue::F32(self.center_to.0));
        params.insert("center_to_y".to_string(), prism_core::node::ParamValue::F32(self.center_to.1));
        params.insert("center_to_z".to_string(), prism_core::node::ParamValue::F32(self.center_to.2));
        params.insert("easing".to_string(), prism_core::node::ParamValue::Str(self.easing.clone()));
        if let Some(d) = self.fov_deg {
            params.insert("fov_deg".to_string(), prism_core::node::ParamValue::F32(d));
        }
        let args = CombinatorArgs {
            duration: self.duration,
            fps: self.fps,
            width: self.width,
            height: self.height,
            params,
            sources: vec![self.base.clone()],
        };
        expand_ken_burns(&args)
    }
}

// ---------------------------------------------------------------------------
// 组合子五：MorphBetween（场景间形变转场）
// ---------------------------------------------------------------------------

/// 形变转场种类（映射到 DSL v2 transition 的 kind 词表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphKind {
    /// 交叉溶解（全局线性权重）。
    Crossfade,
    /// 淡入淡出（同 Crossfade 的整数权重路径）。
    Fade,
    /// 中点硬切。
    Cut,
    /// 向黑淡出。
    Black,
    /// 向白淡出。
    White,
    /// 线性擦除（带平滑带）。
    Wipe,
    /// 随机像素溶解。
    Dissolve,
    /// 整体推移。
    Push,
}

impl MorphKind {
    /// DSL transition kind 词表名。
    pub fn as_str(self) -> &'static str {
        match self {
            MorphKind::Crossfade => "crossfade",
            MorphKind::Fade => "fade",
            MorphKind::Cut => "cut",
            MorphKind::Black => "black",
            MorphKind::White => "white",
            MorphKind::Wipe => "wipe",
            MorphKind::Dissolve => "dissolve",
            MorphKind::Push => "push",
        }
    }
}

/// 形变转场建造者：`from` / `to` 两子片顺序拼接，重叠区按种类混合。
#[derive(Debug, Clone)]
pub struct MorphBetween {
    from: NodeGraph,
    to: NodeGraph,
    kind: MorphKind,
    overlap: Option<f32>,
    band: u32,
    seed: u32,
    dir: String,
    fps: u32,
    width: u32,
    height: u32,
}

impl MorphBetween {
    /// 以前后子片构造（缺省 crossfade、重叠 = min(前,后)/2、dir = left）。
    pub fn new(from: NodeGraph, to: NodeGraph) -> Self {
        Self {
            from,
            to,
            kind: MorphKind::Crossfade,
            overlap: None,
            band: 2,
            seed: 0,
            dir: "left".to_string(),
            fps: 0,
            width: 0,
            height: 0,
        }
    }

    /// 形变种类。
    pub fn kind(mut self, kind: MorphKind) -> Self {
        self.kind = kind;
        self
    }

    /// 重叠时长（秒；缺省 `min(前片时长, 后片时长) × 0.5`）。
    pub fn overlap(mut self, sec: f32) -> Self {
        self.overlap = Some(sec);
        self
    }

    /// wipe 过渡带半宽（px；缺省 2）。
    pub fn band(mut self, px: u32) -> Self {
        self.band = px;
        self
    }

    /// dissolve 显影种子（缺省 0）。
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// wipe / push 方向（left / right / up / down；缺省 left）。
    pub fn dir(mut self, dir: impl Into<String>) -> Self {
        self.dir = dir.into();
        self
    }

    /// 帧率（缺省取两子片较大者）。
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// 画幅（缺省取两子片较大者）。
    pub fn frame(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// 展开为等价 v2 节点图（顺序时间线 + 一条 overlap 过渡）。
    pub fn to_node_graph(&self) -> Result<NodeGraph, CombinatorError> {
        let mut params = std::collections::BTreeMap::new();
        params.insert("kind".to_string(), prism_core::node::ParamValue::Str(self.kind.as_str().to_string()));
        params.insert("band".to_string(), prism_core::node::ParamValue::U32(self.band));
        params.insert("seed".to_string(), prism_core::node::ParamValue::U32(self.seed));
        params.insert("dir".to_string(), prism_core::node::ParamValue::Str(self.dir.clone()));
        if let Some(o) = self.overlap {
            params.insert("overlap".to_string(), prism_core::node::ParamValue::F32(o));
        }
        let args = CombinatorArgs {
            duration: 0.0,
            fps: self.fps,
            width: self.width,
            height: self.height,
            params,
            sources: vec![self.from.clone(), self.to.clone()],
        };
        expand_morph_between(&args)
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gpu::CpuRefBackend;

    /// 把节点图组装为编译产物并渲染出帧（小画幅，快）。
    fn render_graph(g: &NodeGraph) -> prism_render::Deliverable {
        let compile = g.to_compile_output();
        let mut device = CpuRefBackend;
        prism_render::render_movie(&compile, &mut device).expect("组合子展开产物应可渲染")
    }

    /// 小画幅球体子片。
    fn ball(albedo: (f32, f32, f32), duration: f32) -> NodeGraph {
        sphere_scene(albedo, 1.0, 6.0, 45.0, duration, 96, 54, 8)
    }

    #[test]
    fn title_card_renders_frames() {
        let g = TitleCard::new("你好，Prism")
            .duration(1.0)
            .frame(96, 54)
            .fps(8)
            .to_node_graph()
            .expect("标题卡应展开");
        let d = render_graph(&g);
        assert_eq!(d.frame_count(), 8, "1s @ 8fps = 8 帧");
        assert_eq!(d.frames[0].pixels.len(), 96 * 54 * 4);
        // 中段（淡入完成）应出现白色文本像素。
        let lit = |fb: &prism_gpu::Framebuffer| -> usize {
            fb.pixels
                .chunks_exact(4)
                .filter(|px| px[0] > 200 || px[1] > 200 || px[2] > 200)
                .count()
        };
        assert!(lit(&d.frames[4]) > 0, "中段应渲染出文本");
    }

    #[test]
    fn lower_third_renders_frames() {
        let g = LowerThird::new("主讲人")
            .subtitle("工程师")
            .duration(1.0)
            .frame(96, 54)
            .fps(8)
            .to_node_graph()
            .expect("lower third 应展开");
        let d = render_graph(&g);
        assert_eq!(d.frame_count(), 8);
    }

    #[test]
    fn split_screen_left_right_halves_come_from_sources() {
        let left = ball((0.9, 0.2, 0.2), 1.0);
        let right = ball((0.2, 0.4, 0.9), 1.0);
        let g = SplitScreen::new(left.clone(), right.clone())
            .duration(1.0)
            .fps(8)
            .to_node_graph()
            .expect("分屏应展开");
        let d = render_graph(&g);
        assert_eq!(d.frame_count(), 8);

        // 左右子片单独渲染的首帧作为基准。
        let dl = render_graph(&left);
        let dr = render_graph(&right);
        let split = &d.frames[0];
        let lf = &dl.frames[0];
        let rf = &dr.frames[0];
        let w = 96u32;
        let half = w / 2;
        let h = 54u32;
        let at = |fb: &prism_gpu::Framebuffer, x: u32, y: u32| {
            let idx = (y * fb.width + x) as usize * 4;
            [fb.pixels[idx], fb.pixels[idx + 1], fb.pixels[idx + 2]]
        };
        // 抽验：左半幅若干点与左子片一致、右半幅与右子片一致（含角点背景
        // 与分屏线内侧的球体点——球体在各自底片居中，分屏取半幅后其可见
        // 部分贴在分屏线两侧）。
        for &(x, y) in &[
            (2u32, 2u32),
            (half - 3, 30),
            (half, 5),
            (w - 2, 50),
            (half - 2, 27),
            (half + 2, 27),
        ] {
            let got = at(split, x, y);
            let want = if x < half { at(lf, x, y) } else { at(rf, x, y) };
            assert_eq!(got, want, "分屏帧 ({x},{y}) 应取自正确半幅来源");
        }
        // 结构性：左右半幅整体不同（两个子片颜色不同）——背景两半相同，
        // 但球体区域一红一蓝，逐像素比对两半幅必有差异。
        let diff = (0..h)
            .flat_map(|y| (0..half).map(move |x| (x, y)))
            .filter(|&(x, y)| at(split, x, y) != at(split, x + half, y))
            .count();
        assert!(diff > 0, "左右半幅应整体不同（红球 vs 蓝球）");
    }

    #[test]
    fn ken_burns_has_interframe_motion() {
        let g = KenBurns::new(ball((0.8, 0.3, 0.3), 1.0))
            .zoom(8.0, 4.0)
            .duration(1.0)
            .fps(8)
            .to_node_graph()
            .expect("Ken Burns 应展开");
        let d = render_graph(&g);
        assert_eq!(d.frame_count(), 8);
        assert_ne!(
            d.frames[0].pixels, d.frames[4].pixels,
            "轨道半径动画下帧间必须有位移"
        );
        // 首帧与推拉前（静态半径 8 = from_radius）一致：通道在 t=0 取 from_radius。
        let base = sphere_scene((0.8, 0.3, 0.3), 1.0, 8.0, 45.0, 1.0, 96, 54, 8);
        let db = render_graph(&base);
        assert_eq!(d.frames[0].pixels, db.frames[0].pixels, "t=0 应与底片静态渲染一致");
    }

    #[test]
    fn morph_between_renders_transition_frames() {
        let g = MorphBetween::new(ball((0.9, 0.2, 0.2), 1.0), ball((0.2, 0.4, 0.9), 1.0))
            .kind(MorphKind::Crossfade)
            .overlap(0.5)
            .fps(8)
            .to_node_graph()
            .expect("形变应展开");
        let d = render_graph(&g);
        // 总时长 1 + 1 − 0.5 = 1.5s @ 8fps = 12 帧。
        assert_eq!(d.frame_count(), 12);
        assert_eq!(g.transitions.len(), 1);
    }

    #[test]
    fn expand_by_name_dispatches_registry() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("content".to_string(), prism_core::node::ParamValue::Str("X".to_string()));
        let args = CombinatorArgs {
            duration: 1.0,
            fps: 8,
            width: 96,
            height: 54,
            params,
            sources: Vec::new(),
        };
        let g = expand_by_name("title_card", &args).expect("按名展开应成功");
        assert_eq!(g.scenes.len(), 1);
        let err = expand_by_name("nope", &args).expect_err("未登记名应报错");
        assert_eq!(err.code, "C5000");
    }
}
