//! 矢量符号库与公式混排（T04 symbols.rs，F1）：
//!
//! - [`SymbolKind`] + [`symbol_path`]：lim / dy/dx / ∫ / ∑ / ε / δ / π /
//!   ∀ / ∃ / ⇒ / ∞ / √ / ′ / − / × 与上下标小数字的 **PathBuilder 手绘**
//!   （单位坐标设计，`size` 烘焙为像素；基线 y=0，数学 y↑）；
//! - [`assemble_formula`]：文本 run（PrismSansSC，宽经 `layout_text`
//!   **实测**）与符号按基线 x 累加混排，整体以 `center_x` 居中；
//!   [`FormulaPart::Limits`] 限位组只做 ∫ / ∑ 等大型算子的**纵向堆叠**
//!   （advance 计入 0，不横向双计——v3-F 任务 5·缺陷 3）；
//! - [`math_gap`] + [`GapKind`]：数学间距表（LaTeX 原子间距的工程
//!   近似）——二元算子两侧呼吸、变量与括号 / 变量与变量贴近、上下标与
//!   ∫ 与被积式贴合（零 / 负间距）；由 [`assemble_formula`] 在相邻
//!   部件间接入（限位组旁恒 0；文本 run 边界自带显式空格时抑制，
//!   不与 run 内空格双计）；
//! - [`flatten_strokes`]：开放安全平铺——引擎 `flatten` 为填充语义会給
//!   未闭合子路径补闭合点，本 helper 精确裁掉该补点，使 ∫ / ∑ / √ 等
//!   开放笔画描边时不被错误闭合（架构 §7.1 平铺纪律）。
//!
//! 设计纪律（架构 §3.3 / §7.1）：符号局部坐标 `min_x >= 0`（无左悬挂）
//! 且 advance = 包围盒宽 + `SYM_GAP·size`，相邻符号永不叠 ink；全部
//! 坐标有限，非法输入（非有限 `size` / 锚点）不 panic。

use prism_author::{FontKind, PathTrack, StaticText, Track};
use prism_raster::geom::{Point, Transform};
use prism_raster::path::{PathBuilder, PathElement, SubPath};
use prism_text::layout::{layout_text, Align, TextParams};
use prism_text::ttf::Font;
use prism_text::{font_bold, font_regular};

/// 符号与相邻 run 之间的间隙（单位坐标 × `size`）。
pub const SYM_GAP: f32 = 0.12;
/// 上标缩放（相对 `size`）。
pub const SUP_SCALE: f32 = 0.42;
/// 上标抬升（基线以上，单位坐标）。
pub const SUP_RISE: f32 = 0.46;
/// 下标缩放（相对 `size`）。
pub const SUB_SCALE: f32 = 0.42;
/// 下标下沉（基线以下，单位坐标；数字顶 therefore 低于基线）。
pub const SUB_DROP: f32 = 0.52;
/// 单位坐标下单个数字的步进宽（上下标多数字排版用）。
pub const DIGIT_ADVANCE: f32 = 0.58;
/// 限位组（`FormulaPart::Limits`）与算子墨迹的纵向间隙（× 限位字号）。
pub const LIM_GAP_K: f32 = 0.12;
/// 限位数字帽高比（≈ 0.72 em；下限基线 = 算子墨迹底 + gap + 帽高）。
///
/// 与 [`SUP_RISE`] / [`SUB_DROP`] 同族的工程近似比值——仅决定限位
/// 与算子墨迹的间距观感，不参与任何「测量即渲染」路径（限位文本走
/// [`assemble_formula`] 的 `StaticText` 管线，位置由本常量换算基线）。
pub const LIM_CAP_K: f32 = 0.72;
/// 符号描边宽系数（`size·SYM_STROKE_K`，下限 [`MIN_SYM_STROKE_W`]）。
///
/// v3 排版升级：0.07 → 0.05——分数线 / 符号描边粗细 ≈ 字高的
/// 0.04–0.06 倍（LaTeX 分数线比例），整体向 LaTeX 细杠气质靠拢。
pub const SYM_STROKE_K: f32 = 0.05;
/// 符号描边宽下限（px，小字号可读性）。
pub const MIN_SYM_STROKE_W: f32 = 1.5;
/// 二元算子两侧呼吸间距（em；LaTeX `\medmuskip` 4/18em ≈ 0.22 气质，
/// 大于普通符号间隙 [`SYM_GAP`]）。
pub const MATH_GAP_BREATH: f32 = 0.22;
/// 变量与变量之间的小间距（em；贴近但可辨）。
pub const MATH_GAP_THIN: f32 = 0.05;
/// ∫ / ∑ / ∏ 与被积式之间的负间距（em；贴合——LaTeX 大算子与被积式
/// 之间无气隙）。
pub const MATH_GAP_TIGHT: f32 = -0.04;
/// 标点（逗号 / 分号 / 冒号）之后的间距（em；LaTeX `\thinmuskip` 气质）。
pub const MATH_GAP_PUNCT: f32 = 0.17;
/// 圆 / 椭圆四次贝塞尔逼近常数（标准 0.5523）。
const K_CIRCLE: f32 = 0.5523;

/// 符号种类（架构 §4.2 签名逐项对齐；v3-A 扩充数学运算符 14 项）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// lim（手绘衬线感字母组）。
    Lim,
    /// dy/dx（竖式分数）。
    DyDx,
    /// 积分号 ∫。
    Integral,
    /// 求和号 ∑。
    Sum,
    /// ε（月牙 epsilon：中杠 + 左展双臂 + 右汇弧）。
    Epsilon,
    /// δ（圆身 + 右上挑笔）。
    Delta,
    /// π。
    Pi,
    /// θ（圆身 + 中杠）。
    Theta,
    /// Δ（实心三角 Stroke 化）。
    DeltaBig,
    /// ∂（反 6 形偏微分）。
    Partial,
    /// ∏（大 pi 乘积号）。
    Product,
    /// ∀。
    Forall,
    /// ∃。
    Exists,
    /// ⇒。
    Implies,
    /// ∞。
    Inf,
    /// 根号 √。
    Sqrt,
    /// 撇号 ′。
    Prime,
    /// 减号 −。
    Minus,
    /// 乘号 ×。
    Times,
    /// ≈（双波浪约等）。
    Approx,
    /// ≤（小于等于）。
    Le,
    /// ≥（大于等于）。
    Ge,
    /// ≠（不等）。
    Ne,
    /// →（右箭头）。
    Arrow,
    /// ∈（属于号：中杠探左 + 右 bulge 双臂）。
    Element,
    /// ∪（并集）。
    Union,
    /// ∩（交集）。
    Intersect,
    /// ±（正负号）。
    PlusMinus,
    /// ÷（除号）。
    Divide,
    /// =（等号：两条等长横线，双线可辨）。
    Equals,
    /// 上标数字（如 `Sup(2)` = x² 的 2）。
    Sup(usize),
    /// 下标数字（如 `Sub(0)` = x₀ 的 0）。
    Sub(usize),
}

/// 公式组成单元：文本 run（Latin / 数字走字体）、矢量符号或限位组。
#[derive(Debug, Clone, PartialEq)]
pub enum FormulaPart {
    /// 文本 run：内容 / 字体 / 字号 / 颜色（直通 alpha 线性 RGBA）。
    Text(String, FontKind, f32, [f32; 4]),
    /// 矢量符号：种类 / 字号 / 颜色。
    Sym(SymbolKind, f32, [f32; 4]),
    /// 限位组（v3-F 任务 5·缺陷 3）：大型算子（∫ / ∑ 等）的纵向堆叠
    /// 上下限——`hi`（上限）堆于算子墨迹**上方**、`lo`（下限）堆于
    /// **下方**，水平 **advance 计入 0**（算子自身 advance 已覆盖其
    /// slot，后续 run 不再被双计推开）。
    ///
    /// 修复的缺陷：旧式把 `Sym(Sub(0))` / `Sym(Sup(2))` 紧排在 ∫ 之后，
    /// 预走把两者当横向 advance（双计 ~19px）⇒ ∫ 的 ₀ / ² 漂在 ∫ 右侧
    /// 8–21px（应纵向上/下堆叠）。本变体把限位语义显式化：只参与纵向
    /// 堆叠、不横向推进。`lo` / `hi` 为空串表示无限位（不发射）。
    Limits {
        /// 下限文本（如 `"0"`；堆于算子墨迹下方）。
        lo: String,
        /// 上限文本（如 `"2"` / `"x"`；堆于算子墨迹上方）。
        hi: String,
        /// 限位字号（px）。
        size: f32,
        /// 限位颜色（直通 alpha 线性 RGBA）。
        color: [f32; 4],
    },
}

/// 全部符号种类（含参数化上下标代表值；测试遍历与新增登记用）。
pub fn all_symbol_kinds() -> Vec<SymbolKind> {
    vec![
        SymbolKind::Lim,
        SymbolKind::DyDx,
        SymbolKind::Integral,
        SymbolKind::Sum,
        SymbolKind::Epsilon,
        SymbolKind::Delta,
        SymbolKind::Pi,
        SymbolKind::Theta,
        SymbolKind::DeltaBig,
        SymbolKind::Partial,
        SymbolKind::Product,
        SymbolKind::Forall,
        SymbolKind::Exists,
        SymbolKind::Implies,
        SymbolKind::Inf,
        SymbolKind::Sqrt,
        SymbolKind::Prime,
        SymbolKind::Minus,
        SymbolKind::Times,
        SymbolKind::Approx,
        SymbolKind::Le,
        SymbolKind::Ge,
        SymbolKind::Ne,
        SymbolKind::Arrow,
        SymbolKind::Element,
        SymbolKind::Union,
        SymbolKind::Intersect,
        SymbolKind::PlusMinus,
        SymbolKind::Divide,
        SymbolKind::Equals,
        SymbolKind::Sup(2),
        SymbolKind::Sub(3),
    ]
}

/// 生成符号的局部坐标路径（基线 y=0，数学 y↑；`size` 烘焙为像素）。
///
/// 非法 `size`（非有限 / <= 0）钳为 0：路径退化为原点重合点集——非空
/// 且全有限（不 panic，渲染层自然跳过零长笔画）。
pub fn symbol_path(kind: SymbolKind, size: f32) -> PathBuilder {
    let s = if size.is_finite() && size > 0.0 { size } else { 0.0 };
    let mut unit = PathBuilder::new();
    match kind {
        SymbolKind::Lim => unit_lim(&mut unit),
        SymbolKind::DyDx => unit_dydx(&mut unit),
        SymbolKind::Integral => unit_integral(&mut unit),
        SymbolKind::Sum => unit_sum(&mut unit),
        SymbolKind::Epsilon => unit_epsilon(&mut unit),
        SymbolKind::Delta => unit_delta(&mut unit),
        SymbolKind::Pi => unit_pi(&mut unit),
        SymbolKind::Forall => unit_forall(&mut unit),
        SymbolKind::Exists => unit_exists(&mut unit),
        SymbolKind::Implies => unit_implies(&mut unit),
        SymbolKind::Inf => unit_inf(&mut unit),
        SymbolKind::Sqrt => unit_sqrt(&mut unit),
        SymbolKind::Prime => unit_prime(&mut unit),
        SymbolKind::Minus => unit_minus(&mut unit),
        SymbolKind::Times => unit_times(&mut unit),
        SymbolKind::Theta => unit_theta(&mut unit),
        SymbolKind::DeltaBig => unit_delta_big(&mut unit),
        SymbolKind::Partial => unit_partial(&mut unit),
        SymbolKind::Product => unit_product(&mut unit),
        SymbolKind::Approx => unit_approx(&mut unit),
        SymbolKind::Le => unit_le(&mut unit),
        SymbolKind::Ge => unit_ge(&mut unit),
        SymbolKind::Ne => unit_ne(&mut unit),
        SymbolKind::Arrow => unit_arrow(&mut unit),
        SymbolKind::Element => unit_element(&mut unit),
        SymbolKind::Union => unit_union(&mut unit),
        SymbolKind::Intersect => unit_intersect(&mut unit),
        SymbolKind::PlusMinus => unit_plus_minus(&mut unit),
        SymbolKind::Divide => unit_divide(&mut unit),
        SymbolKind::Equals => unit_equals(&mut unit),
        SymbolKind::Sup(n) => {
            for (i, ch) in n.to_string().chars().enumerate() {
                let d = ch.to_digit(10).unwrap_or(0);
                digit_path(&mut unit, d, i as f32 * DIGIT_ADVANCE, SUP_RISE, SUP_SCALE);
            }
        }
        SymbolKind::Sub(n) => {
            for (i, ch) in n.to_string().chars().enumerate() {
                let d = ch.to_digit(10).unwrap_or(0);
                digit_path(&mut unit, d, i as f32 * DIGIT_ADVANCE, -SUB_DROP, SUB_SCALE);
            }
        }
    }
    scale_builder(&unit, s)
}

// ---- 数学间距表（v3 排版升级：向 LaTeX 原子间距看齐） ----

/// 间距原子类别（LaTeX 原子类型的工程近似；供 [`math_gap`] 查表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapKind {
    /// 二元算子 / 关系符（= + − × ÷ ≤ ≥ ≈ ≠ ⇒ → ∪ ∩ ∈ ± …）：两侧呼吸。
    Binary,
    /// 变量 / 数字 / 希腊变量字母（x y f t a b n ε δ π θ Δ ∂ …）。
    Var,
    /// 左括号 / 根号（与内容贴近）。
    Open,
    /// 右括号 / 竖线。
    Close,
    /// 上下标（与基座零 / 负间距）。
    Script,
    /// 大型算子 ∫ ∑ ∏（与被积式贴合）。
    BigOp,
    /// 函数名（lim / sin / cos …；与左括号、变量贴近）。
    Func,
    /// 标点（, ; : .）。
    Punct,
    /// 撇号标记 ′（`f′` 贴基座）。
    Mark,
    /// 量词 ∀ ∃（与被约束变量贴近）。
    Quant,
    /// 边界已带显式空白 ⇒ 间距抑制（不与 run 内空格双计）。
    Spaced,
    /// 其余（∞ / dy/dx 等）。
    Ord,
}

/// 数学间距表（em 单位）：`prev` → `next` 的横向间距。
///
/// 规则（向 LaTeX 原子间距看齐）：
/// - 二元算子**两侧**呼吸（[`MATH_GAP_BREATH`]，大于普通间隙
///   [`SYM_GAP`]；两侧皆算子时只记一份，不叠加）；
/// - 变量与括号 / 变量与变量贴近（0 / [`MATH_GAP_THIN`]）；
/// - 上下标与基座、∫ 与被积式：零或负间距（[`MATH_GAP_TIGHT`]，贴合）；
/// - 函数名 / 撇号 / 量词边界零间距（`sin(x)` / `f′` / `∀e`）；
/// - 边界任一侧为 [`GapKind::Spaced`]（文本 run 自带显式空格）⇒ 0，
///   避免与 run 内空格双计。
pub fn math_gap(prev: GapKind, next: GapKind) -> f32 {
    use GapKind::{
        Binary, BigOp, Close, Func, Mark, Open, Punct, Quant, Script, Spaced,
        Var,
    };
    // 显式空白边界：间距抑制。
    if matches!(prev, Spaced) || matches!(next, Spaced) {
        return 0.0;
    }
    match (prev, next) {
        // 上下标贴基座 / 贴后续 run。
        (Script, _) | (_, Script) => 0.0,
        // ∫ / ∑ / ∏：与被积式负间距贴合；被二元算子引领时保留算子侧呼吸。
        (BigOp, Var) => MATH_GAP_TIGHT,
        (Binary, BigOp) => MATH_GAP_BREATH,
        (BigOp, _) | (_, BigOp) => 0.0,
        // 二元算子两侧呼吸。
        (Binary, _) | (_, Binary) => MATH_GAP_BREATH,
        // 撇号 / 量词：零间距。
        (Mark, _) | (_, Mark) => 0.0,
        (Quant, _) | (_, Quant) => 0.0,
        // 括号与内容：零间距。
        (Open, _) | (_, Close) => 0.0,
        // 变量与变量：小间距。
        (Var, Var) => MATH_GAP_THIN,
        // 函数名边界：零间距。
        (Func, _) | (_, Func) => 0.0,
        // 标点之后：thin。
        (Punct, _) => MATH_GAP_PUNCT,
        _ => 0.0,
    }
}

/// 公式文本中常见的函数名（首词 / 尾词识别用）。
const FUNC_NAMES: [&str; 16] = [
    "arcsin", "arccos", "arctan", "arccot", "lim", "sin", "cos", "tan", "cot", "sec", "csc", "ln",
    "log", "exp", "max", "min",
];

/// 符号的间距原子类别。
fn sym_gap_kind(kind: SymbolKind) -> GapKind {
    match kind {
        SymbolKind::Minus
        | SymbolKind::Times
        | SymbolKind::Approx
        | SymbolKind::Le
        | SymbolKind::Ge
        | SymbolKind::Ne
        | SymbolKind::PlusMinus
        | SymbolKind::Divide
        | SymbolKind::Equals
        | SymbolKind::Implies
        | SymbolKind::Arrow
        | SymbolKind::Union
        | SymbolKind::Intersect
        | SymbolKind::Element => GapKind::Binary,
        SymbolKind::Integral | SymbolKind::Sum | SymbolKind::Product => GapKind::BigOp,
        SymbolKind::Sup(_) | SymbolKind::Sub(_) => GapKind::Script,
        SymbolKind::Prime => GapKind::Mark,
        SymbolKind::Sqrt => GapKind::Open,
        SymbolKind::Forall | SymbolKind::Exists => GapKind::Quant,
        SymbolKind::Epsilon
        | SymbolKind::Delta
        | SymbolKind::Pi
        | SymbolKind::Theta
        | SymbolKind::DeltaBig
        | SymbolKind::Partial => GapKind::Var,
        SymbolKind::Lim => GapKind::Func,
        SymbolKind::DyDx | SymbolKind::Inf => GapKind::Ord,
    }
}

/// 文本 run 左缘的间距原子（空 / 前导空白 ⇒ [`GapKind::Spaced`]）。
fn text_left_kind(text: &str) -> GapKind {
    match text.chars().next() {
        None => GapKind::Spaced,
        Some(c) if c.is_whitespace() => GapKind::Spaced,
        Some(c) => char_gap_kind(c, text, true),
    }
}

/// 文本 run 右缘的间距原子（空 / 尾随空白 ⇒ [`GapKind::Spaced`]）。
fn text_right_kind(text: &str) -> GapKind {
    match text.chars().last() {
        None => GapKind::Spaced,
        Some(c) if c.is_whitespace() => GapKind::Spaced,
        Some(c) => char_gap_kind(c, text, false),
    }
}

/// 单字符间距原子（`at_left` 决定函数名按首词 / 尾词匹配）。
fn char_gap_kind(c: char, text: &str, at_left: bool) -> GapKind {
    match c {
        '0'..='9' => GapKind::Var,
        'a'..='z' | 'A'..='Z' => {
            if func_name_at_edge(text, at_left) {
                GapKind::Func
            } else {
                GapKind::Var
            }
        }
        '(' | '[' | '{' => GapKind::Open,
        ')' | ']' | '}' | '|' => GapKind::Close,
        '+' | '-' | '*' | '/' | '=' | '<' | '>' => GapKind::Binary,
        '−' | '×' | '÷' | '±' | '≤' | '≥' | '≈' | '≠' | '→' | '⇒' | '∈' | '∪' | '∩' => {
            GapKind::Binary
        }
        ',' | ';' | ':' | '.' => GapKind::Punct,
        _ => GapKind::Ord,
    }
}

/// 文本首 / 尾是否为函数名（词边界安全：相邻字符须为非字母或串端）。
fn func_name_at_edge(text: &str, at_left: bool) -> bool {
    FUNC_NAMES.iter().any(|f| {
        if at_left {
            text.starts_with(f)
                && text[f.len()..]
                    .chars()
                    .next()
                    .map_or(true, |c| !c.is_alphabetic())
        } else {
            text.ends_with(f)
                && text[..text.len() - f.len()]
                    .chars()
                    .next_back()
                    .map_or(true, |c| !c.is_alphabetic())
        }
    })
}

/// 部件左缘的间距原子（限位组视作 Spaced：对横向间距透明）。
fn part_left_kind(part: &FormulaPart) -> GapKind {
    match part {
        FormulaPart::Text(t, ..) => text_left_kind(t),
        FormulaPart::Sym(k, ..) => sym_gap_kind(*k),
        FormulaPart::Limits { .. } => GapKind::Spaced,
    }
}

/// 部件右缘的间距原子（限位组视作 Spaced：对横向间距透明）。
fn part_right_kind(part: &FormulaPart) -> GapKind {
    match part {
        FormulaPart::Text(t, ..) => text_right_kind(t),
        FormulaPart::Sym(k, ..) => sym_gap_kind(*k),
        FormulaPart::Limits { .. } => GapKind::Spaced,
    }
}

/// 部件名义字号（间距折算用）。
fn part_size(part: &FormulaPart) -> f32 {
    match part {
        FormulaPart::Text(_, _, s, _) | FormulaPart::Sym(_, s, _) => *s,
        FormulaPart::Limits { size, .. } => *size,
    }
}

/// 相邻部件横向间距（px）：[`math_gap`] 查表 × 两侧较大字号。
///
/// 限位组（纵向堆叠注解）两侧间距恒 0（透明）；非有限 / 非正字号 ⇒ 0。
fn boundary_gap(prev: &FormulaPart, next: &FormulaPart) -> f32 {
    if matches!(prev, FormulaPart::Limits { .. }) || matches!(next, FormulaPart::Limits { .. }) {
        return 0.0;
    }
    let g = math_gap(part_right_kind(prev), part_left_kind(next));
    if g == 0.0 {
        return 0.0;
    }
    let s = part_size(prev).max(part_size(next));
    if s.is_finite() && s > 0.0 {
        g * s
    } else {
        0.0
    }
}

/// 文本 run 与符号按基线混排为轨道列表，整体以 `center_x` 水平居中。
///
/// 算法（架构 §4.2 / R5）：
/// 1. 字体只解析一次（`font_bold` / `font_regular`）；
/// 2. 预走累加 advance：文本宽 = `layout_text` 实测（参数与
///    `render::draw_static` 逐位一致：`letter_spacing = 0`、
///    `max_width = INFINITY`、`Align::Center`）；符号 advance =
///    包围盒右缘（含曲线控制点，保守）+ `SYM_GAP·size`；
///    [`FormulaPart::Limits`] advance = **0**（只纵向堆叠，见其文档）；
///    相邻部件间再插入 [`math_gap`] 间距（× 两侧较大字号；限位组旁为
///    0，文本 run 边界自带显式空格时抑制——不双计）；
/// 3. 左缘 = `center_x − total/2`，逐 run 发射：文本 →
///    `StaticText`（`center_x = pen + w/2`，`y = baseline_y`）；
///    符号 → y 翻转变换后每轮廓一条描边 `PathTrack`；
///    限位组 → [`emit_limits`]（附着到**最近的前置算子** slot）。
///
/// 非法锚点（非有限 `center_x` / `baseline_y`）或空 parts → 空列表。
pub fn assemble_formula(parts: &[FormulaPart], center_x: f32, baseline_y: f32) -> Vec<Track> {
    let mut tracks: Vec<Track> = Vec::new();
    if !center_x.is_finite() || !baseline_y.is_finite() {
        return tracks;
    }
    let bold = font_bold();
    let regular = font_regular();

    // 1) 预走：advance 表（纯函数，发射期复用同值）。
    //    限位组 advance = 0（任务 5·缺陷 3：只纵向堆叠，不横向推进）。
    let advances: Vec<f32> = parts
        .iter()
        .map(|p| match p {
            FormulaPart::Text(text, kind, size, _) => {
                text_advance(font_for(kind, &bold, &regular), text, *size)
            }
            FormulaPart::Sym(kind, size, _) => symbol_advance(&symbol_path(*kind, *size), *size),
            FormulaPart::Limits { .. } => 0.0,
        })
        .collect();
    // 1b) 相邻部件数学间距（`math_gap` 表；限位组透明；末部件后无间距）。
    let gaps: Vec<f32> = (0..parts.len().saturating_sub(1))
        .map(|i| boundary_gap(&parts[i], &parts[i + 1]))
        .collect();
    let total: f32 = advances.iter().sum::<f32>() + gaps.iter().sum::<f32>();
    if !total.is_finite() {
        return tracks;
    }

    // 2) 预走：pen 落位 + 限位锚点。每个 Limits 记录**最近的前置算子**
    //    （大型 Sym）的 slot 起点与墨迹包围盒（slot 局部像素，数学 y↑）；
    //    无前置算子时锚点为 None ⇒ 发射期退化为「贴当前 pen 的标称堆叠」
    //    （契约外输入的健全退化，不 panic）。算子路径在本预走重建一次
    //    （与发射期 `symbol_path` 同参同值——纯函数，逐位一致）。
    let mut pens: Vec<f32> = Vec::with_capacity(parts.len());
    let mut limit_anchors: Vec<Option<LimitAnchor>> = Vec::new();
    let mut pen = center_x - total * 0.5;
    let mut prev_op: Option<LimitAnchor> = None;
    for (i, (part, adv)) in parts.iter().zip(advances.iter()).enumerate() {
        pens.push(pen);
        match part {
            FormulaPart::Text(..) => pen += adv,
            FormulaPart::Sym(kind, size, _) => {
                prev_op = Some(LimitAnchor {
                    slot_start: pen,
                    ink: ink_bbox(&symbol_path(*kind, *size)),
                });
                pen += adv;
            }
            FormulaPart::Limits { .. } => limit_anchors.push(prev_op),
        }
        // 部件后接数学间距（末部件 / 限位组旁恒 0）。
        pen += gaps.get(i).copied().unwrap_or(0.0);
    }

    // 3) 发射（pen 全部取自预走表，与 advance 表同源）。
    //    limit_anchors 只按 Limits 部件收集（与部件全局索引 i 不同口径），
    //    须用独立计数 li 取锚点——文本/符号部件插在中间即索引越界。
    let mut li = 0usize;
    for (i, part) in parts.iter().enumerate() {
        let pen = pens[i];
        match part {
            FormulaPart::Text(text, kind, size, color) => {
                let font = font_for(kind, &bold, &regular);
                let w = text_advance(font, text, *size);
                if !text.is_empty() && w > 0.0 {
                    tracks.push(Track::Static(StaticText {
                        content: text.clone(),
                        font: *kind,
                        size: *size,
                        color: *color,
                        at: 0.0,
                        in_sec: 0.0,
                        letter_spacing: 0.0,
                        y: baseline_y,
                        out_sec: 0.0,
                        out_at: None,
                        center_x: Some(pen + w * 0.5),
                    }));
                }
            }
            FormulaPart::Sym(kind, size, color) => {
                let path = symbol_path(*kind, *size);
                // 局部坐标（y↑，基线 0）→ 设备像素（y↓）：(x, y) → (x+pen, −y+baseline)。
                let tf = Transform {
                    a: 1.0,
                    b: 0.0,
                    c: 0.0,
                    d: -1.0,
                    tx: pen,
                    ty: baseline_y,
                };
                let stroke_w = (*size * SYM_STROKE_K).max(MIN_SYM_STROKE_W);
                for contour in flatten_strokes(&path, &tf) {
                    if contour.len() < 2 {
                        continue; // 渲染层跳过 <2 点轨；此处直接不发射
                    }
                    tracks.push(Track::Path(PathTrack {
                        points: contour,
                        color: *color,
                        width: stroke_w,
                        fill: false,
                        at: 0.0,
                        in_sec: 0.0,
                        out_sec: 0.0,
                        out_at: None,
                        draw_sec: 0.0,
                        closed: false,
                    }));
                }
            }
            FormulaPart::Limits { lo, hi, size, color } => {
                // 锚点按 Limits 部件序取（li），防御式 get：契约外输入不 panic。
                let anchor = limit_anchors.get(li).copied().flatten();
                li += 1;
                emit_limits(
                    &mut tracks,
                    lo,
                    hi,
                    *size,
                    *color,
                    pen,
                    anchor,
                    baseline_y,
                );
            }
        }
    }
    tracks
}

/// 限位组锚点（内部）：前置算子 slot 起点（设备 x）+ 算子墨迹包围盒
/// （slot 局部像素坐标 `[min_x, max_x, min_y, max_y]`，数学 y↑）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct LimitAnchor {
    /// 算子 slot 起点（设备像素 x；公式左缘起算的 pen）。
    slot_start: f32,
    /// 算子墨迹包围盒（含曲线控制点的保守上界，与 [`symbol_advance`]
    /// 同口径；空路径退化为原点零包络）。
    ink: [f32; 4],
}

/// 发射限位组（内部）：`hi` / `lo` 各一条居中 `StaticText`，分别堆于
/// 算子墨迹上方 / 下方；水平 advance 已在预走计入 0（pen 不动）。
///
/// 纵向位置（设备 px，y↓）：
/// - 上限基线 = 算子墨迹顶 − [`LIM_GAP_K`]·size（数字坐在基线上，墨迹
///   自然整体高出算子顶一个间隙）；
/// - 下限基线 = 算子墨迹底 + [`LIM_GAP_K`]·size + [`LIM_CAP_K`]·size
///   （数字顶贴在算子底下方一个间隙）。
///
/// 无前置算子（锚点 `None`）时退化为以当前 pen 为水平中心、以基线为
/// 算子中心的标称堆叠（契约外输入不 panic；限位组按约定应紧邻其算子
/// 之后）。非法 `size`（非有限 / <= 0）整组跳过（与文本 run 同纪律）。
fn emit_limits(
    tracks: &mut Vec<Track>,
    lo: &str,
    hi: &str,
    size: f32,
    color: [f32; 4],
    pen: f32,
    anchor: Option<LimitAnchor>,
    baseline_y: f32,
) {
    if !size.is_finite() || size <= 0.0 {
        return;
    }
    let gap = LIM_GAP_K * size;
    let cap = LIM_CAP_K * size;
    let (op_center_x, op_top_y, op_bot_y) = match anchor {
        Some(a) => (
            a.slot_start + (a.ink[0] + a.ink[1]) * 0.5, // 墨迹水平中心
            baseline_y - a.ink[3],                      // 墨迹顶（设备最小 y）
            baseline_y - a.ink[2],                      // 墨迹底（设备最大 y）
        ),
        None => (pen, baseline_y - size * 0.5, baseline_y + size * 0.5),
    };
    // 上限：基线落在算子顶上方一个间隙处。
    if !hi.is_empty() {
        tracks.push(limit_text(hi, size, color, op_center_x, op_top_y - gap));
    }
    // 下限：数字顶贴在算子底下方一个间隙处 ⇒ 基线 = 底 + 间隙 + 帽高。
    if !lo.is_empty() {
        tracks.push(limit_text(lo, size, color, op_center_x, op_bot_y + gap + cap));
    }
}

/// 限位文本轨（内部）：与公式文本 run 同管线（测量即渲染），`at` /
/// `in_sec` / `out_sec` / `out_at` 全零——发射期不控时间，由调用方
/// 统一 patch（与 [`assemble_formula`] 的既有约定一致）。
fn limit_text(content: &str, size: f32, color: [f32; 4], center_x: f32, baseline_y: f32) -> Track {
    Track::Static(StaticText {
        content: String::from(content),
        font: FontKind::Regular,
        size,
        color,
        at: 0.0,
        in_sec: 0.0,
        letter_spacing: 0.0,
        y: baseline_y,
        out_sec: 0.0,
        out_at: None,
        center_x: Some(center_x),
    })
}

/// 开放安全平铺：逐子路径经 `tf` 平铺为折线，**不补闭合点**。
///
/// 引擎 `PathBuilder::flatten` 为填充语义会给未闭合子路径追加「终点 →
/// 起点」的闭合边；对描边符号（∫ / ∑ / √ / ∀ / ′ 等开放笔画）该闭合边
/// 是视觉错误。本 helper 复用引擎平铺（自适应细分同源），并精确识别
/// 该补点：未闭合子路径的补点恒等于变换后起点（`flatten` 内部以
/// `contour.first()` 追加），故当「原子路径末点 ≠ 起点」时平铺结果末点
/// 必为补点，裁之；子路径本身首尾重合（未 `close` 的几何闭合轮廓，如
/// 圆）时不动。
pub fn flatten_strokes(path: &PathBuilder, tf: &Transform) -> Vec<Vec<[f32; 2]>> {
    let mut out: Vec<Vec<[f32; 2]>> = Vec::with_capacity(path.subpaths.len());
    for sub in &path.subpaths {
        // 单子路径临时构建 → 复用引擎 flatten（曲线自适应平铺唯一实现）。
        let mut single = PathBuilder::new();
        single.subpaths.push(sub.clone());
        let mut contour: Vec<[f32; 2]> = single
            .flatten(tf)
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|p| [p.x, p.y])
            .collect();
        if !sub.closed && contour.len() >= 2 {
            let first = contour[0];
            let last = contour[contour.len() - 1];
            // 原子路径末点（变换后）；None = 空子路径（仅 move_to）。
            let end = sub_endpoint(sub).map(|p| {
                let q = tf.apply(p);
                [q.x, q.y]
            });
            let appended_close = match end {
                Some(e) => e[0] != first[0] || e[1] != first[1],
                None => false,
            };
            if appended_close && last[0] == first[0] && last[1] == first[1] {
                contour.pop(); // 裁掉填充语义的自动闭合点
            }
        }
        out.push(contour);
    }
    out
}

// ---- 内部辅助 ----

/// 字体选择（`FontKind` → 已解析字体引用）。
fn font_for<'a>(kind: &FontKind, bold: &'a Font<'a>, regular: &'a Font<'a>) -> &'a Font<'a> {
    match kind {
        FontKind::Bold => bold,
        FontKind::Regular => regular,
    }
}

/// 文本 run 宽（`layout_text` 实测；参数与 `render::draw_static` 逐位
/// 一致，保证测量即渲染）。非法字号 → 0。
fn text_advance(font: &Font, text: &str, size: f32) -> f32 {
    if text.is_empty() || !size.is_finite() || size <= 0.0 {
        return 0.0;
    }
    let params = TextParams {
        font_size: size,
        letter_spacing: 0.0,
        line_spacing: 0.0,
        max_width: f32::INFINITY,
        align: Align::Center,
    };
    let w = layout_text(font, text, &params).width;
    if w.is_finite() {
        w
    } else {
        0.0
    }
}

/// 符号 advance：包围盒**右缘**（含曲线控制点，保守上界）+
/// `SYM_GAP·size`——以笔位为 slot 起点计量。
///
/// 全部符号 `min_x >= 0`（无左悬挂），故 ink 必落于
/// `[pen + min_x, pen + max_x] ⊆ [pen, pen + advance − gap]`：相邻符号
/// 永不叠 ink（左悬挂空白计入 advance，宁松不叠）。
fn symbol_advance(path: &PathBuilder, size: f32) -> f32 {
    let gap = if size.is_finite() { SYM_GAP * size } else { 0.0 };
    let mut max_x = f32::NEG_INFINITY;
    for sub in &path.subpaths {
        visit_points(sub, &mut |p| {
            max_x = max_x.max(p.x);
        });
    }
    if !max_x.is_finite() {
        return gap; // 空路径（无任何点）
    }
    (max_x + gap).max(0.0)
}

/// 符号墨迹包围盒（slot 局部像素坐标 `[min_x, max_x, min_y, max_y]`；
/// 含曲线控制点，与 [`symbol_advance`] 同口径的保守上界）。
///
/// 空路径（无任何点，如 `size <= 0` 的退化符号）退化为原点零包络
/// `[0, 0, 0, 0]`（不 panic；调用方按零高算子处理）。
fn ink_bbox(path: &PathBuilder) -> [f32; 4] {
    let (mut x0, mut x1) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
    for sub in &path.subpaths {
        visit_points(sub, &mut |p| {
            x0 = x0.min(p.x);
            x1 = x1.max(p.x);
            y0 = y0.min(p.y);
            y1 = y1.max(p.y);
        });
    }
    if !x0.is_finite() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    [x0, x1, y0, y1]
}

/// 文本 run 宽度实测（px；与 [`assemble_formula`] 预走及引擎
/// `render::draw_static` **逐位同参**——`layout_text` + 调用方字体 /
/// 字号 / 字距 ⇒ 「测量即渲染」）。
///
/// 用途：HUD 读数定位探针——如 S4 斜率 HUD 负 / 非负双段的位宽补偿、
/// S5 拍间 `n = ` 读数的等左缘锚定（任务 5·缺陷 1 / 2）：避免在场景
/// 代码里硬编码字体度量（minus 宽、数字位宽等），串宽随字体实测走。
/// 空文本 / 非法字号 → 0（不 panic）。
// v3-E 遗留：左锚改造（缺陷 1/2）消费本探针前，二进制构建暂无调用方；
// 测试模块已在用（limits_group_* 对拍）。保留 API，勿删。
#[allow(dead_code)]
pub fn text_run_width(text: &str, kind: FontKind, size: f32, letter_spacing: f32) -> f32 {
    if text.is_empty() || !size.is_finite() || size <= 0.0 {
        return 0.0;
    }
    // font_for 返回值借用 bold/regular（统一生命周期），临时字体须先绑定再取引用，
    // 否则 E0716（temporary value dropped while borrowed）。
    let font_bold = font_bold();
    let font_regular = font_regular();
    let font = font_for(&kind, &font_bold, &font_regular);
    let params = TextParams {
        font_size: size,
        letter_spacing,
        line_spacing: 0.0,
        max_width: f32::INFINITY,
        align: Align::Center,
    };
    let w = layout_text(font, text, &params).width;
    if w.is_finite() {
        w
    } else {
        0.0
    }
}

/// 遍历子路径全部几何点（起点 / 线段终点 / 曲线控制点与终点）。
fn visit_points(sub: &SubPath, f: &mut impl FnMut(Point)) {
    f(sub.start);
    for el in &sub.elements {
        match *el {
            PathElement::Line(p) => f(p),
            PathElement::Quad { ctrl, to } => {
                f(ctrl);
                f(to);
            }
        }
    }
}

/// 子路径末元素终点（平铺结果的实际末点来源）。
fn sub_endpoint(sub: &SubPath) -> Option<Point> {
    sub.elements.last().map(|el| match *el {
        PathElement::Line(p) => p,
        PathElement::Quad { to, .. } => to,
    })
}

/// 单位坐标路径整体缩放为像素坐标（保留闭合标记）。
fn scale_builder(src: &PathBuilder, s: f32) -> PathBuilder {
    let mut out = PathBuilder::new();
    for sub in &src.subpaths {
        out.move_to(Point::new(sub.start.x * s, sub.start.y * s));
        for el in &sub.elements {
            match *el {
                PathElement::Line(p) => {
                    out.line_to(Point::new(p.x * s, p.y * s));
                }
                PathElement::Quad { ctrl, to } => {
                    out.quad_to(
                        Point::new(ctrl.x * s, ctrl.y * s),
                        Point::new(to.x * s, to.y * s),
                    );
                }
            }
        }
        if sub.closed {
            out.close();
        }
    }
    out
}

/// 闭合圆（四次贝塞尔逼近）。
fn push_circle(b: &mut PathBuilder, cx: f32, cy: f32, r: f32) {
    push_ellipse(b, cx, cy, r, r);
}

/// 闭合椭圆（四次贝塞尔逼近，起于正上方）。
fn push_ellipse(b: &mut PathBuilder, cx: f32, cy: f32, rx: f32, ry: f32) {
    b.move_to(Point::new(cx, cy + ry));
    b.quad_to(Point::new(cx + K_CIRCLE * rx, cy + ry), Point::new(cx + rx, cy));
    b.quad_to(Point::new(cx + rx, cy - K_CIRCLE * ry), Point::new(cx, cy - ry));
    b.quad_to(Point::new(cx - K_CIRCLE * rx, cy - ry), Point::new(cx - rx, cy));
    b.quad_to(Point::new(cx - K_CIRCLE * rx, cy + ry), Point::new(cx, cy + ry));
    b.close();
}

// ---- 单位坐标符号几何（基线 y=0，y↑；宽高见各注释） ----

/// lim：l 主干 / i 主干+点 / m 双拱（宽 ~1.16，高 0.72）。
fn unit_lim(b: &mut PathBuilder) {
    b.move_to(Point::new(0.10, 0.0));
    b.line_to(Point::new(0.10, 0.72));
    b.move_to(Point::new(0.40, 0.0));
    b.line_to(Point::new(0.40, 0.52));
    push_circle(b, 0.40, 0.68, 0.05); // i 点
    b.move_to(Point::new(0.66, 0.0));
    b.line_to(Point::new(0.66, 0.52));
    b.move_to(Point::new(0.66, 0.52));
    b.quad_to(Point::new(0.79, 0.74), Point::new(0.86, 0.52));
    b.move_to(Point::new(0.86, 0.52));
    b.quad_to(Point::new(0.96, 0.74), Point::new(1.06, 0.52));
    b.move_to(Point::new(1.06, 0.0));
    b.line_to(Point::new(1.06, 0.52));
}

/// d 字母（主干 + 右圆碗），`(x, y)` 左下角，`k` 缩放。
fn push_d(b: &mut PathBuilder, x: f32, y: f32, k: f32) {
    b.move_to(Point::new(x, y));
    b.line_to(Point::new(x, y + 0.62 * k));
    push_circle(b, x + 0.31 * k, y + 0.31 * k, 0.31 * k);
}

/// y 字母（双斜笔交汇 + 左下尾，尾探到基线下）。
fn push_y(b: &mut PathBuilder, x: f32, y: f32, k: f32) {
    b.move_to(Point::new(x, y + 0.50 * k));
    b.line_to(Point::new(x + 0.26 * k, y + 0.06 * k));
    b.move_to(Point::new(x + 0.52 * k, y + 0.50 * k));
    b.line_to(Point::new(x + 0.26 * k, y + 0.06 * k));
    b.move_to(Point::new(x + 0.26 * k, y + 0.06 * k));
    b.line_to(Point::new(x + 0.08 * k, y - 0.24 * k));
}

/// x 字母（双交叉斜笔）。
fn push_x(b: &mut PathBuilder, x: f32, y: f32, k: f32) {
    b.move_to(Point::new(x, y + 0.50 * k));
    b.line_to(Point::new(x + 0.46 * k, y));
    b.move_to(Point::new(x, y));
    b.line_to(Point::new(x + 0.46 * k, y + 0.50 * k));
}

/// dy/dx：竖式分数（分子 dy / 分数杠 / 分母 dx；宽 ~1.10，高 ~0.93）。
fn unit_dydx(b: &mut PathBuilder) {
    const K: f32 = 0.5;
    push_d(b, 0.05, 0.50, K); // 分子
    push_y(b, 0.72, 0.50, K);
    b.move_to(Point::new(0.0, 0.38)); // 分数杠
    b.line_to(Point::new(1.10, 0.38));
    push_d(b, 0.05, 0.0, K); // 分母
    push_x(b, 0.72, 0.0, K);
}

/// ∫：高挑 S 形（宽 ~0.62，高 ~1.22 ≈ 行高 0.72 的 1.7 倍，底探到基线下）。
///
/// v3 排版升级：旧几何高 0.98（≈1.36 倍行高）偏小；拉高到 1.22 后与
/// LaTeX 显示式积分号（1.6–1.8 倍行高）比例看齐。宽度不变（advance 稳
/// 定），单开放子路径结构不变（`flatten_strokes` 回归锁保持）。
fn unit_integral(b: &mut PathBuilder) {
    b.move_to(Point::new(0.44, 0.98));
    b.quad_to(Point::new(0.06, 0.92), Point::new(0.12, 0.60));
    b.quad_to(Point::new(0.18, 0.28), Point::new(0.38, 0.18));
    b.quad_to(Point::new(0.62, 0.12), Point::new(0.56, -0.24));
}

/// ∑：上横 + 左尖中点 + 下横（单开放笔画，宽 0.82，高 0.92）。
///
/// v3 排版升级：旧几何 0.70×0.60 与普通字母同档，无「大算子」气质；
/// 拉高到 0.92（≈1.28 倍帽高）、加宽到 0.82，向 LaTeX 显示式 ∑ 看齐。
/// 仍为 5 点单开放折线（`flatten_strokes` 五段回归锁保持）。
fn unit_sum(b: &mut PathBuilder) {
    b.move_to(Point::new(0.0, 0.92));
    b.line_to(Point::new(0.82, 0.92));
    b.line_to(Point::new(0.35, 0.46));
    b.line_to(Point::new(0.82, 0.0));
    b.line_to(Point::new(0.0, 0.0));
}

/// ε（lunate epsilon Ɛ 形）：中横杠左展 + 上下双臂自右脊左挑 + 右汇弧
/// （宽 ~0.52，高 ~0.56）。
///
/// v3-A 重画：旧几何为「右 bulge 尖灭月牙」（两段右鼓弧），小字号下
/// 读作 ">"/σ（审计截图「σ0:41」缺陷）。新几何钉死「反 3 加杠」的
/// 教科书结构——中杠左端伸到双臂开口之外，右脊弧线汇合，与 ∃（右挑
/// 臂）镜像可辨。
fn unit_epsilon(b: &mut PathBuilder) {
    // 中横杠：左端探出双臂开口（ε 的判别特征）。
    b.move_to(Point::new(0.02, 0.30));
    b.line_to(Point::new(0.42, 0.30));
    // 上臂：自右脊向左上挑（轻微上拱）。
    b.move_to(Point::new(0.10, 0.48));
    b.quad_to(Point::new(0.28, 0.56), Point::new(0.42, 0.42));
    // 下臂：自右脊向左下挑。
    b.move_to(Point::new(0.10, 0.12));
    b.quad_to(Point::new(0.28, 0.04), Point::new(0.42, 0.18));
    // 右脊汇弧（ bulge 向右）。
    b.move_to(Point::new(0.42, 0.42));
    b.quad_to(Point::new(0.52, 0.30), Point::new(0.42, 0.18));
}

/// δ：圆身 + 右上挑笔（ascender 起笔没入圆身，探出圆顶右缘；宽 ~0.58，
/// 高 ~0.66）。
///
/// v3-A 重画：挑笔加高（顶 0.66）并令起笔点落在圆身内部（旧几何起笔
/// 在圆缘外，与圆环之间有可见断口），保证 δ 与 σ/ε 的形体差异。
fn unit_delta(b: &mut PathBuilder) {
    push_ellipse(b, 0.27, 0.27, 0.25, 0.27);
    b.move_to(Point::new(0.36, 0.44));
    b.quad_to(Point::new(0.50, 0.62), Point::new(0.58, 0.66));
}

/// θ：圆身 + 中横杠（宽 ~0.52，高 ~0.56）。
fn unit_theta(b: &mut PathBuilder) {
    push_ellipse(b, 0.26, 0.28, 0.24, 0.28);
    b.move_to(Point::new(0.02, 0.28));
    b.line_to(Point::new(0.50, 0.28));
}

/// Δ：实心三角 Stroke 化（开放三笔，宽 ~0.54，高 ~0.62）。
fn unit_delta_big(b: &mut PathBuilder) {
    b.move_to(Point::new(0.0, 0.0));
    b.line_to(Point::new(0.27, 0.62));
    b.line_to(Point::new(0.54, 0.0));
}

/// ∂：反 6 形偏微分（右竖弧 + 左下碗，单开放笔画；宽 ~0.44，高 ~0.66）。
fn unit_partial(b: &mut PathBuilder) {
    b.move_to(Point::new(0.40, 0.66));
    b.quad_to(Point::new(0.40, 0.40), Point::new(0.36, 0.20));
    b.quad_to(Point::new(0.20, 0.04), Point::new(0.14, 0.20));
    b.quad_to(Point::new(0.08, 0.40), Point::new(0.22, 0.48));
    b.quad_to(Point::new(0.32, 0.52), Point::new(0.40, 0.44));
}

/// ∏：大 pi 乘积号（双腿 + 顶横；宽 0.58，高 0.88）。
///
/// v3 排版升级：旧几何高 0.58 与 ∑ 的旧值同档偏小；拉高到 0.88 与
/// 新 ∑ 同档（大算子家族一致），向 LaTeX 显示式 ∏ 看齐。
fn unit_product(b: &mut PathBuilder) {
    b.move_to(Point::new(0.10, 0.0));
    b.line_to(Point::new(0.10, 0.88));
    b.move_to(Point::new(0.48, 0.0));
    b.line_to(Point::new(0.48, 0.88));
    b.move_to(Point::new(0.0, 0.88));
    b.line_to(Point::new(0.58, 0.88));
}

/// ≈：双波浪约等（上下两行 S 波；宽 ~0.48，高 ~0.48）。
fn unit_approx(b: &mut PathBuilder) {
    b.move_to(Point::new(0.0, 0.40));
    b.quad_to(Point::new(0.12, 0.48), Point::new(0.24, 0.40));
    b.quad_to(Point::new(0.36, 0.32), Point::new(0.48, 0.40));
    b.move_to(Point::new(0.0, 0.24));
    b.quad_to(Point::new(0.12, 0.32), Point::new(0.24, 0.24));
    b.quad_to(Point::new(0.36, 0.16), Point::new(0.48, 0.24));
}

/// ≤：左尖 + 底横（宽 ~0.48，高 ~0.52）。
fn unit_le(b: &mut PathBuilder) {
    b.move_to(Point::new(0.44, 0.52));
    b.line_to(Point::new(0.14, 0.30));
    b.line_to(Point::new(0.44, 0.08));
    b.move_to(Point::new(0.06, 0.20));
    b.line_to(Point::new(0.46, 0.20));
}

/// ≥：右尖 + 底横（宽 ~0.46，高 ~0.52）。
fn unit_ge(b: &mut PathBuilder) {
    b.move_to(Point::new(0.04, 0.52));
    b.line_to(Point::new(0.34, 0.30));
    b.line_to(Point::new(0.04, 0.08));
    b.move_to(Point::new(0.06, 0.20));
    b.line_to(Point::new(0.46, 0.20));
}

/// ≠：双横 + 反斜杠（宽 ~0.42，高 ~0.50）。
fn unit_ne(b: &mut PathBuilder) {
    b.move_to(Point::new(0.02, 0.38));
    b.line_to(Point::new(0.42, 0.38));
    b.move_to(Point::new(0.02, 0.22));
    b.line_to(Point::new(0.42, 0.22));
    b.move_to(Point::new(0.06, 0.50));
    b.line_to(Point::new(0.40, 0.10));
}

/// →：箭杆 + 箭头（宽 ~0.42，居中于 0.26）。
fn unit_arrow(b: &mut PathBuilder) {
    b.move_to(Point::new(0.02, 0.26));
    b.line_to(Point::new(0.36, 0.26));
    b.move_to(Point::new(0.24, 0.38));
    b.line_to(Point::new(0.42, 0.26));
    b.line_to(Point::new(0.24, 0.14));
}

/// ∈：属于号（中杠探左 + 右 bulge 双臂；宽 ~0.40，高 ~0.46）。
fn unit_element(b: &mut PathBuilder) {
    b.move_to(Point::new(0.0, 0.26));
    b.line_to(Point::new(0.26, 0.26));
    b.move_to(Point::new(0.26, 0.42));
    b.quad_to(Point::new(0.40, 0.46), Point::new(0.40, 0.28));
    b.move_to(Point::new(0.26, 0.10));
    b.quad_to(Point::new(0.40, 0.06), Point::new(0.40, 0.24));
}

/// ∪：并集（双腿 + 底弧；宽 ~0.32，高 ~0.52）。
fn unit_union(b: &mut PathBuilder) {
    b.move_to(Point::new(0.08, 0.52));
    b.line_to(Point::new(0.08, 0.22));
    b.move_to(Point::new(0.32, 0.52));
    b.line_to(Point::new(0.32, 0.22));
    b.move_to(Point::new(0.08, 0.22));
    b.quad_to(Point::new(0.20, 0.04), Point::new(0.32, 0.22));
}

/// ∩：交集（双腿 + 顶弧；宽 ~0.32，高 ~0.48）。
fn unit_intersect(b: &mut PathBuilder) {
    b.move_to(Point::new(0.08, 0.0));
    b.line_to(Point::new(0.08, 0.30));
    b.move_to(Point::new(0.32, 0.0));
    b.line_to(Point::new(0.32, 0.30));
    b.move_to(Point::new(0.08, 0.30));
    b.quad_to(Point::new(0.20, 0.48), Point::new(0.32, 0.30));
}

/// ±：正负号（上十字 + 下横；宽 ~0.36，高 ~0.64）。
fn unit_plus_minus(b: &mut PathBuilder) {
    b.move_to(Point::new(0.04, 0.26));
    b.line_to(Point::new(0.36, 0.26));
    b.move_to(Point::new(0.20, 0.46));
    b.line_to(Point::new(0.20, 0.64));
    b.move_to(Point::new(0.10, 0.55));
    b.line_to(Point::new(0.30, 0.55));
}

/// ÷：除号（上横 + 上下点；宽 ~0.36，高 ~0.56）。
fn unit_divide(b: &mut PathBuilder) {
    b.move_to(Point::new(0.04, 0.40));
    b.line_to(Point::new(0.36, 0.40));
    push_circle(b, 0.20, 0.52, 0.035);
    push_circle(b, 0.20, 0.10, 0.035);
}

/// π：双竖 + 顶横（宽 0.72，高 0.56）。
fn unit_pi(b: &mut PathBuilder) {
    b.move_to(Point::new(0.10, 0.0));
    b.line_to(Point::new(0.10, 0.56));
    b.move_to(Point::new(0.62, 0.0));
    b.line_to(Point::new(0.62, 0.56));
    b.move_to(Point::new(0.02, 0.56));
    b.line_to(Point::new(0.70, 0.56));
}

/// ∀：顶弧 + 外撇双腿（宽 0.90，高 0.62）。
fn unit_forall(b: &mut PathBuilder) {
    b.move_to(Point::new(0.05, 0.50));
    b.quad_to(Point::new(0.25, 0.62), Point::new(0.45, 0.62));
    b.quad_to(Point::new(0.65, 0.62), Point::new(0.85, 0.50));
    b.move_to(Point::new(0.05, 0.50));
    b.line_to(Point::new(0.28, 0.0));
    b.move_to(Point::new(0.85, 0.50));
    b.line_to(Point::new(0.62, 0.0));
}

/// ∃：竖主干 + 上下两右挑横笔（主干贴 x=0，宽 0.30，高 0.56）。
fn unit_exists(b: &mut PathBuilder) {
    b.move_to(Point::new(0.0, 0.0));
    b.line_to(Point::new(0.0, 0.56));
    b.move_to(Point::new(0.0, 0.48));
    b.quad_to(Point::new(0.18, 0.46), Point::new(0.30, 0.40));
    b.quad_to(Point::new(0.18, 0.34), Point::new(0.0, 0.30));
    b.move_to(Point::new(0.0, 0.26));
    b.quad_to(Point::new(0.18, 0.24), Point::new(0.30, 0.18));
    b.quad_to(Point::new(0.18, 0.12), Point::new(0.0, 0.08));
}

/// ⇒：长箭杆 + 箭头（宽 0.90，居中于 0.28）。
fn unit_implies(b: &mut PathBuilder) {
    b.move_to(Point::new(0.05, 0.28));
    b.line_to(Point::new(0.75, 0.28));
    b.move_to(Point::new(0.55, 0.46));
    b.line_to(Point::new(0.90, 0.28));
    b.line_to(Point::new(0.55, 0.10));
}

/// ∞：双圆横八字（宽 1.00，高 0.54）。
fn unit_inf(b: &mut PathBuilder) {
    push_ellipse(b, 0.26, 0.28, 0.24, 0.26);
    push_ellipse(b, 0.74, 0.28, 0.24, 0.26);
}

/// √：左下斜笔 → 顶横 → 右短竖（单开放笔画，宽 0.80，高 0.78）。
///
/// v3 排版升级：旧几何高 0.56 仅齐字母帽高，根号罩不住被开方式；
/// 顶升高到 0.78（超出帽高 0.72），向 LaTeX 根号（延伸至被开方项
/// 之上）看齐。仍为 4 点单开放折线。
fn unit_sqrt(b: &mut PathBuilder) {
    b.move_to(Point::new(0.02, 0.08));
    b.line_to(Point::new(0.28, 0.78));
    b.line_to(Point::new(0.80, 0.78));
    b.line_to(Point::new(0.80, 0.56));
}

/// ′：帽高短斜笔（宽 0.24，顶 0.74）。
fn unit_prime(b: &mut PathBuilder) {
    b.move_to(Point::new(0.14, 0.50));
    b.quad_to(Point::new(0.17, 0.62), Point::new(0.24, 0.74));
}

/// −：减号（居中于 0.26）。
fn unit_minus(b: &mut PathBuilder) {
    b.move_to(Point::new(0.05, 0.26));
    b.line_to(Point::new(0.65, 0.26));
}

/// =：等号双横线（宽 0.60em，居中于数学轴 0.27em）。
///
/// 双线净间距 0.20em（26px 字号下 ≈ 5.2px，远大于笔宽
/// [`SYM_STROKE_K`]·size ⇒ 双线可辨、不会糊成一条）。旧实现把「=」当
/// 文本 run 过 PrismSansSC，该字形极细近乎缺失；此处改矢量笔画，笔宽
/// 由 [`assemble_formula`] 统一按字号折算（与数字字高同体系）。
fn unit_equals(b: &mut PathBuilder) {
    b.move_to(Point::new(0.04, 0.17));
    b.line_to(Point::new(0.64, 0.17));
    b.move_to(Point::new(0.04, 0.37));
    b.line_to(Point::new(0.64, 0.37));
}

/// ×：乘号双交叉斜笔（居中于 0.25）。
fn unit_times(b: &mut PathBuilder) {
    b.move_to(Point::new(0.08, 0.44));
    b.line_to(Point::new(0.52, 0.06));
    b.move_to(Point::new(0.08, 0.06));
    b.line_to(Point::new(0.52, 0.44));
}

/// 数字 0-9 单位笔画（高 0.5·k，宽 ~0.5·k；`(x0, y0)` 左下角）。
fn digit_path(b: &mut PathBuilder, d: u32, x0: f32, y0: f32, k: f32) {
    match d % 10 {
        0 => push_ellipse(b, x0 + 0.26 * k, y0 + 0.25 * k, 0.23 * k, 0.26 * k),
        1 => {
            b.move_to(Point::new(x0 + 0.02 * k, y0 + 0.40 * k));
            b.line_to(Point::new(x0 + 0.14 * k, y0 + 0.50 * k));
            b.move_to(Point::new(x0 + 0.14 * k, y0));
            b.line_to(Point::new(x0 + 0.14 * k, y0 + 0.50 * k));
            b.move_to(Point::new(x0, y0));
            b.line_to(Point::new(x0 + 0.30 * k, y0));
        }
        2 => {
            b.move_to(Point::new(x0 + 0.03 * k, y0 + 0.36 * k));
            b.quad_to(Point::new(x0 + 0.05 * k, y0 + 0.50 * k), Point::new(x0 + 0.22 * k, y0 + 0.50 * k));
            b.quad_to(Point::new(x0 + 0.40 * k, y0 + 0.50 * k), Point::new(x0 + 0.43 * k, y0 + 0.34 * k));
            b.quad_to(Point::new(x0 + 0.45 * k, y0 + 0.18 * k), Point::new(x0 + 0.30 * k, y0 + 0.05 * k));
            b.line_to(Point::new(x0 + 0.02 * k, y0));
            b.line_to(Point::new(x0 + 0.47 * k, y0));
        }
        3 => {
            b.move_to(Point::new(x0 + 0.06 * k, y0 + 0.42 * k));
            b.quad_to(Point::new(x0 + 0.40 * k, y0 + 0.54 * k), Point::new(x0 + 0.42 * k, y0 + 0.32 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 + 0.12 * k), Point::new(x0 + 0.10 * k, y0 + 0.12 * k));
            b.move_to(Point::new(x0 + 0.10 * k, y0 + 0.12 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 + 0.12 * k), Point::new(x0 + 0.42 * k, y0 - 0.02 * k));
            b.quad_to(Point::new(x0 + 0.40 * k, y0 - 0.10 * k), Point::new(x0 + 0.12 * k, y0 - 0.02 * k));
        }
        4 => {
            b.move_to(Point::new(x0 + 0.30 * k, y0 + 0.50 * k));
            b.line_to(Point::new(x0 + 0.04 * k, y0 + 0.14 * k));
            b.move_to(Point::new(x0 + 0.02 * k, y0 + 0.14 * k));
            b.line_to(Point::new(x0 + 0.44 * k, y0 + 0.14 * k));
            b.move_to(Point::new(x0 + 0.30 * k, y0 + 0.50 * k));
            b.line_to(Point::new(x0 + 0.30 * k, y0));
        }
        5 => {
            b.move_to(Point::new(x0 + 0.06 * k, y0 + 0.48 * k));
            b.line_to(Point::new(x0 + 0.40 * k, y0 + 0.48 * k));
            b.move_to(Point::new(x0 + 0.06 * k, y0 + 0.48 * k));
            b.line_to(Point::new(x0 + 0.07 * k, y0 + 0.26 * k));
            b.move_to(Point::new(x0 + 0.07 * k, y0 + 0.26 * k));
            b.quad_to(Point::new(x0 + 0.42 * k, y0 + 0.28 * k), Point::new(x0 + 0.42 * k, y0 + 0.12 * k));
            b.quad_to(Point::new(x0 + 0.42 * k, y0 - 0.04 * k), Point::new(x0 + 0.24 * k, y0 - 0.04 * k));
            b.quad_to(Point::new(x0 + 0.06 * k, y0 - 0.04 * k), Point::new(x0 + 0.06 * k, y0 + 0.12 * k));
        }
        6 => {
            b.move_to(Point::new(x0 + 0.42 * k, y0 + 0.40 * k));
            b.quad_to(Point::new(x0 + 0.08 * k, y0 + 0.38 * k), Point::new(x0 + 0.05 * k, y0 + 0.14 * k));
            b.quad_to(Point::new(x0 + 0.02 * k, y0 - 0.06 * k), Point::new(x0 + 0.24 * k, y0 - 0.06 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 - 0.06 * k), Point::new(x0 + 0.44 * k, y0 + 0.16 * k));
            b.move_to(Point::new(x0 + 0.44 * k, y0 + 0.16 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 + 0.34 * k), Point::new(x0 + 0.24 * k, y0 + 0.34 * k));
            b.quad_to(Point::new(x0 + 0.06 * k, y0 + 0.34 * k), Point::new(x0 + 0.06 * k, y0 + 0.14 * k));
        }
        7 => {
            b.move_to(Point::new(x0 + 0.04 * k, y0 + 0.48 * k));
            b.line_to(Point::new(x0 + 0.44 * k, y0 + 0.48 * k));
            b.move_to(Point::new(x0 + 0.44 * k, y0 + 0.48 * k));
            b.line_to(Point::new(x0 + 0.18 * k, y0));
        }
        8 => {
            push_ellipse(b, x0 + 0.24 * k, y0 + 0.36 * k, 0.19 * k, 0.16 * k);
            push_ellipse(b, x0 + 0.24 * k, y0 + 0.13 * k, 0.22 * k, 0.17 * k);
        }
        _ => {
            // 9：上环 + 左下尾。
            b.move_to(Point::new(x0 + 0.08 * k, y0 + 0.12 * k));
            b.quad_to(Point::new(x0 + 0.08 * k, y0 + 0.38 * k), Point::new(x0 + 0.26 * k, y0 + 0.38 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 + 0.38 * k), Point::new(x0 + 0.44 * k, y0 + 0.16 * k));
            b.quad_to(Point::new(x0 + 0.44 * k, y0 + 0.02 * k), Point::new(x0 + 0.28 * k, y0 + 0.02 * k));
            b.move_to(Point::new(x0 + 0.28 * k, y0 + 0.02 * k));
            b.line_to(Point::new(x0 + 0.30 * k, y0 - 0.10 * k));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{CURVE_RGBA, GOLD_RGBA, TEXT_RGBA};
    use prism_raster::path::PathBuilder as PB;

    /// 收集（展平后的）全部点，供断言。
    fn all_points(path: &PathBuilder) -> Vec<Point> {
        let mut pts = Vec::new();
        for sub in &path.subpaths {
            visit_points(sub, &mut |p| pts.push(p));
        }
        pts
    }

    /// 轨道里全部 Path 点。
    fn track_points(tracks: &[Track]) -> Vec<[f32; 2]> {
        let mut pts = Vec::new();
        for t in tracks {
            if let Track::Path(p) = t {
                pts.extend_from_slice(&p.points);
            }
        }
        pts
    }

    #[test]
    fn all_symbol_kinds_paths_nonempty_finite_on_baseline_band() {
        let size = 40.0f32;
        for kind in all_symbol_kinds() {
            let path = symbol_path(kind, size);
            assert!(!path.subpaths.is_empty(), "{kind:?} 必须有笔画");
            let pts = all_points(&path);
            assert!(!pts.is_empty(), "{kind:?} 必须有点");
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for p in &pts {
                assert!(p.x.is_finite() && p.y.is_finite(), "{kind:?} 坐标必须有限：{p:?}");
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
            // 无左悬挂（相邻符号不叠 ink 的几何前提）。
            assert!(min_x >= -1e-6, "{kind:?} min_x 必须 >= 0，实际 {min_x}");
            // 基线纪律带：上下标允许 [-0.6, 1.1]·size，其余 [-0.25, 1.0]·size。
            let (lo, hi) = match kind {
                SymbolKind::Sup(_) | SymbolKind::Sub(_) => (-0.6 * size, 1.1 * size),
                _ => (-0.25 * size, 1.0 * size),
            };
            assert!(
                min_y >= lo && max_y <= hi,
                "{kind:?} 纵向越出基线纪律带 [{lo}, {hi}]：[{min_y}, {max_y}]"
            );
        }
    }

    #[test]
    fn assemble_formula_advances_match_hand_computation() {
        // f′(x) = ：文本 "f" + Prime + 文本 "(x) = "，中心 480、基线 300。
        let parts = [
            FormulaPart::Text(String::from("f"), FontKind::Bold, 40.0, CURVE_RGBA),
            FormulaPart::Sym(SymbolKind::Prime, 40.0, GOLD_RGBA),
            FormulaPart::Text(String::from("(x) = "), FontKind::Bold, 40.0, TEXT_RGBA),
        ];
        // 手算：文本宽 = layout_text 实测；符号 advance = 包围盒 + SYM_GAP·size。
        let bold = font_bold();
        let params = |sz: f32| TextParams {
            font_size: sz,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Center,
        };
        let w1 = layout_text(&bold, "f", &params(40.0)).width;
        let w3 = layout_text(&bold, "(x) = ", &params(40.0)).width;
        let sym_path = symbol_path(SymbolKind::Prime, 40.0);
        let pts = all_points(&sym_path);
        let mx = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let adv_sym = mx + SYM_GAP * 40.0; // 右缘 + 间隙（symbol_advance 同式）
        let total = w1 + adv_sym + w3;
        let left = 480.0 - total * 0.5;

        let tracks = assemble_formula(&parts, 480.0, 300.0);
        // 两段文本 + 至少一段 Prime 笔画。
        let statics: Vec<&StaticText> = tracks
            .iter()
            .filter_map(|t| match t {
                Track::Static(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(statics.len(), 2, "两段文本 run");
        // 第一段中心 = left + w1/2（逐位一致）。
        assert_eq!(statics[0].center_x, Some(left + w1 * 0.5));
        assert_eq!(statics[0].content, "f");
        assert_eq!(statics[0].y, 300.0);
        assert_eq!(statics[0].letter_spacing, 0.0, "测量与渲染同参");
        // Prime 笔画从左缘 + w1 处起（局部 min_x >= 0 ⇒ 设备 min_x >= pen）。
        let sym_pts = track_points(&tracks);
        let sym_min_x = sym_pts.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        assert!(sym_min_x >= left + w1 - 1e-3, "符号不得侵入前序 run");
        assert!(sym_min_x <= left + w1 + 0.2 * 40.0, "符号应紧贴前序 run 起笔");
        // 第二段中心 = left + w1 + adv_sym + w3/2。
        assert_eq!(statics[1].center_x, Some(left + w1 + adv_sym + w3 * 0.5));
        assert_eq!(statics[1].content, "(x) = ");
    }

    #[test]
    fn text_run_widths_come_from_layout_text_not_estimates() {
        // "lim" 与 "iii" 同字符数——若按字数估算宽度二者相等；实测必不等。
        let bold = font_bold();
        let params = TextParams {
            font_size: 40.0,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Center,
        };
        let w_lim = layout_text(&bold, "lim", &params).width;
        let w_iii = layout_text(&bold, "iii", &params).width;
        assert!(w_lim.is_finite() && w_iii.is_finite() && w_lim > 0.0 && w_iii > 0.0);
        assert!(
            (w_lim - w_iii).abs() > 1e-3,
            "同字数不同字形，实测宽必须可辨（{w_lim} vs {w_iii}）"
        );
        // 混排落位必须用实测宽：第二段中心 = left + w_lim + w_iii/2。
        let parts = [
            FormulaPart::Text(String::from("lim"), FontKind::Bold, 40.0, TEXT_RGBA),
            FormulaPart::Text(String::from("iii"), FontKind::Bold, 40.0, TEXT_RGBA),
        ];
        let tracks = assemble_formula(&parts, 480.0, 300.0);
        let centers: Vec<f32> = tracks
            .iter()
            .filter_map(|t| match t {
                Track::Static(s) => s.center_x,
                _ => None,
            })
            .collect();
        assert_eq!(centers.len(), 2);
        let total = w_lim + w_iii;
        let left = 480.0 - total * 0.5;
        assert_eq!(centers[0], left + w_lim * 0.5);
        assert_eq!(centers[1], left + w_lim + w_iii * 0.5);
    }

    #[test]
    fn sup_sub_offsets_and_ordering() {
        // 上标必须在基线之上、下标在基线之下，且顺序即 parts 序。
        let baseline = 300.0f32;
        let sup = assemble_formula(
            &[FormulaPart::Sym(SymbolKind::Sup(2), 40.0, GOLD_RGBA)],
            480.0,
            baseline,
        );
        let sup_pts = track_points(&sup);
        assert!(!sup_pts.is_empty(), "Sup(2) 必须有笔画");
        for p in &sup_pts {
            assert!(p[1] < baseline, "上标必须在基线上方（设备 y 更小）：{p:?}");
        }
        let sub = assemble_formula(
            &[FormulaPart::Sym(SymbolKind::Sub(2), 40.0, GOLD_RGBA)],
            480.0,
            baseline,
        );
        let sub_pts = track_points(&sub);
        assert!(!sub_pts.is_empty(), "Sub(2) 必须有笔画");
        for p in &sub_pts {
            assert!(p[1] > baseline, "下标必须在基线下方：{p:?}");
        }
        // 顺序：Sup(2) 在文本 "x" 左侧。
        let mixed = assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Sup(2), 40.0, GOLD_RGBA),
                FormulaPart::Text(String::from("x"), FontKind::Bold, 40.0, TEXT_RGBA),
            ],
            480.0,
            baseline,
        );
        let sup_max_x = track_points(&mixed)
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let text_cx = mixed
            .iter()
            .find_map(|t| match t {
                Track::Static(s) => s.center_x,
                _ => None,
            })
            .expect("必须有文本 run");
        let bold = font_bold();
        let w_x = layout_text(
            &bold,
            "x",
            &TextParams {
                font_size: 40.0,
                letter_spacing: 0.0,
                line_spacing: 0.0,
                max_width: f32::INFINITY,
                align: Align::Center,
            },
        )
        .width;
        assert!(sup_max_x < text_cx - w_x * 0.5, "上标必须整体在文本左缘之左");
        // 多数字上标：Sup(10) 比 Sup(1) 更长（步进累加）。
        let a1 = assemble_formula(&[FormulaPart::Sym(SymbolKind::Sup(1), 40.0, GOLD_RGBA)], 480.0, baseline);
        let a10 = assemble_formula(&[FormulaPart::Sym(SymbolKind::Sup(10), 40.0, GOLD_RGBA)], 480.0, baseline);
        let w1 = track_points(&a1).iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max)
            - track_points(&a1).iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let w10 = track_points(&a10).iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max)
            - track_points(&a10).iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        assert!(w10 > w1, "多数字上标必须更宽（{w10} vs {w1}）");
    }

    #[test]
    fn adjacent_symbols_do_not_overlap_ink() {
        // 全部种类 + Lim 相邻两符号（异色区分）：前者 ink 右缘必须严格在后者 ink 左缘之左。
        for kind in all_symbol_kinds() {
            let tracks = assemble_formula(
                &[
                    FormulaPart::Sym(kind, 40.0, CURVE_RGBA),
                    FormulaPart::Sym(SymbolKind::Lim, 40.0, GOLD_RGBA),
                ],
                480.0,
                300.0,
            );
            let mut a_max = f32::NEG_INFINITY;
            let mut b_min = f32::INFINITY;
            let mut n_path = 0usize;
            for t in &tracks {
                if let Track::Path(p) = t {
                    n_path += 1;
                    for q in &p.points {
                        if p.color == CURVE_RGBA {
                            a_max = a_max.max(q[0]);
                        } else if p.color == GOLD_RGBA {
                            b_min = b_min.min(q[0]);
                        }
                    }
                }
            }
            assert!(n_path >= 2, "{kind:?} 两符号都应有笔画");
            assert!(
                a_max < b_min,
                "{kind:?} 与 Lim 相邻叠 ink（a_max={a_max}, b_min={b_min}）"
            );
        }
    }

    #[test]
    fn flatten_strokes_keeps_open_strokes_open() {
        let tf = Transform { a: 1.0, b: 0.0, c: 0.0, d: -1.0, tx: 100.0, ty: 200.0 };
        // ∫：单开放子路径——平铺结果不得含自动闭合点（末点 ≠ 首点）。
        let integral = symbol_path(SymbolKind::Integral, 40.0);
        let contours = flatten_strokes(&integral, &tf);
        assert_eq!(contours.len(), 1);
        let c = &contours[0];
        assert!(c.len() >= 3, "∫ 平铺应有多段");
        let first = c[0];
        let last = c[c.len() - 1];
        assert!(
            first[0] != last[0] || first[1] != last[1],
            "开放笔画不得被补闭合点：首 {first:?} 尾 {last:?}"
        );
        // ∑：五段开放折线——同样保持开放。
        let sum = symbol_path(SymbolKind::Sum, 40.0);
        let sc = &flatten_strokes(&sum, &tf)[0];
        assert_eq!(sc.len(), 5, "∑ 五段折线，不得多出闭合点");
        assert!(sc[0][0] != sc[4][0] || sc[0][1] != sc[4][1]);
        // ∞：两个显式闭合椭圆——保持闭合（末点 == 首点）。
        let inf = symbol_path(SymbolKind::Inf, 40.0);
        let ic = flatten_strokes(&inf, &tf);
        assert_eq!(ic.len(), 2);
        for c in &ic {
            let f = c[0];
            let l = c[c.len() - 1];
            assert_eq!((f[0].to_bits(), f[1].to_bits()), (l[0].to_bits(), l[1].to_bits()), "闭合轮廓必须首尾重合");
        }
        // 空路径 / 单点路径不 panic。
        let empty = PB::new();
        assert!(flatten_strokes(&empty, &tf).is_empty());
        let mut single = PB::new();
        single.move_to(Point::new(1.0, 2.0));
        let sc = flatten_strokes(&single, &tf);
        assert_eq!(sc.len(), 1);
        assert_eq!(sc[0].len(), 1);
    }

    #[test]
    fn assemble_formula_invalid_inputs_safe() {        // 空 parts / 非有限锚点 → 空。
        assert!(assemble_formula(&[], 480.0, 300.0).is_empty());
        assert!(assemble_formula(&[], f32::NAN, 300.0).is_empty());
        assert!(assemble_formula(&[], 480.0, f32::INFINITY).is_empty());
        // 非法 size / 空文本 / NaN 颜色：不 panic，产出有限或忽略。
        let tracks = assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Integral, 0.0, GOLD_RGBA),
                FormulaPart::Sym(SymbolKind::Sqrt, -10.0, GOLD_RGBA),
                FormulaPart::Sym(SymbolKind::Sum, f32::NAN, GOLD_RGBA),
                FormulaPart::Text(String::new(), FontKind::Bold, 40.0, TEXT_RGBA),
                FormulaPart::Text(String::from("ok"), FontKind::Regular, f32::NAN, TEXT_RGBA),
            ],
            480.0,
            300.0,
        );
        for t in &tracks {
            match t {
                Track::Static(s) => {
                    assert!(s.y.is_finite());
                    if let Some(cx) = s.center_x {
                        assert!(cx.is_finite(), "StaticText center_x 必须有限");
                    }
                }
                Track::Path(p) => {
                    for q in &p.points {
                        assert!(q[0].is_finite() && q[1].is_finite(), "PathTrack 点必须有限");
                    }
                }
                _ => panic!("assemble_formula 只应产出 Static/Path"),
            }
        }
        // 符号 size=0 时 advance=0、文本 NaN size 不发射——公式仍以 "ok" 之外无可见 run。
        let has_static = tracks.iter().any(|t| matches!(t, Track::Static(_)));
        assert!(!has_static, "空文本 / NaN 字号不得发射文本 run");
    }

    // -----------------------------------------------------------------------
    // v3-A 新增符号测试（规格验收：每符号非空、与参考符号互异、
    // ε/δ 重画结构锁定、超宽文本边界）
    // -----------------------------------------------------------------------

    /// ε 的 lunate 结构锁定：中杠左端必须探出双臂左端之外（判别特征），
    /// 右脊 bulge 向右（与 ∃ 的右挑臂镜像可辨；审计缺陷「ε 读作 σ/＞」的
    /// 回归锁）。
    #[test]
    fn epsilon_structure_is_lunate() {
        let eps = symbol_path(SymbolKind::Epsilon, 40.0);
        let pts = all_points(&eps);
        // 三段开放笔画：中杠 / 上臂 / 下臂 / 右脊（4 条子路径）。
        assert_eq!(eps.subpaths.len(), 4, "ε = 杠 + 上臂 + 下臂 + 右脊");
        // 中杠（水平段）：左端 x 必须小于上下臂左端 x（探出开口之外）。
        let bar = &eps.subpaths[0];
        let bar_x0 = bar.start.x.min(bar_endpoint_x(bar));
        let mut arms_x0 = f32::INFINITY;
        for s in &eps.subpaths[1..3] {
            arms_x0 = arms_x0.min(contour_x_range(s).0);
        }
        assert!(
            bar_x0 < arms_x0,
            "ε 中杠左端（{bar_x0}）必须探出双臂左端（{arms_x0}）之外"
        );
        // 右脊 bulge：子路径 3 的最大 x 大于中杠右端。
        let spine_x1 = contour_x_range(&eps.subpaths[3]).1;
        assert!(spine_x1 > contour_x_range(bar).1, "右脊须向右 bulge");
        // 全部点有限且在基线纪律带内。
        for p in &pts {
            assert!(p.x.is_finite() && p.y.is_finite());
            assert!(p.y >= -0.25 * 40.0 && p.y <= 1.0 * 40.0, "ε 纵向越带：{p:?}");
        }
    }

    /// δ 挑笔必须没入圆身再探出（起笔点在圆内部 ⇒ 无断口），且顶高于圆顶。
    #[test]
    fn delta_ascender_is_embedded_then_protrudes() {
        let d = symbol_path(SymbolKind::Delta, 40.0);
        assert_eq!(d.subpaths.len(), 2, "δ = 圆身 + 挑笔");
        // 圆身子路径（闭合椭圆）与挑笔（开放）。
        let (body, asc) = if d.subpaths[0].closed {
            (&d.subpaths[0], &d.subpaths[1])
        } else {
            (&d.subpaths[1], &d.subpaths[0])
        };
        // 圆身包围盒（近似：以展平点集计）。
        let (bx0, bx1, by0, by1) = contour_bbox(body);
        // 挑笔起笔点须落在圆身包围盒内（没入 ink）。
        let a0 = asc.start;
        assert!(
            a0.x >= bx0 && a0.x <= bx1 && a0.y >= by0 && a0.y <= by1,
            "挑笔起笔点必须没入圆身：{a0:?} ∉ [{bx0},{bx1}]×[{by0},{by1}]"
        );
        // 挑笔终点高于圆顶（ascender 探出）。
        let a1 = sub_endpoint(asc).expect("挑笔有末元素");
        assert!(a1.y > by1, "挑笔终点（{}）须高于圆顶（{}）", a1.y, by1);
    }

    /// 14 个新符号几何两两互异（点集签名不同），且与旧符号互异。
    #[test]
    fn new_symbol_kinds_are_pairwise_distinct() {
        let new_kinds = [
            SymbolKind::Theta,
            SymbolKind::DeltaBig,
            SymbolKind::Partial,
            SymbolKind::Product,
            SymbolKind::Approx,
            SymbolKind::Le,
            SymbolKind::Ge,
            SymbolKind::Ne,
            SymbolKind::Arrow,
            SymbolKind::Element,
            SymbolKind::Union,
            SymbolKind::Intersect,
            SymbolKind::PlusMinus,
            SymbolKind::Divide,
        ];
        // 签名 = 量化点集排序后的序列（±1e-3 容差内视同相等）。
        let sig = |kind: SymbolKind| -> Vec<(i32, i32)> {
            let mut v: Vec<(i32, i32)> = all_points(&symbol_path(kind, 40.0))
                .iter()
                .map(|p| ((p.x * 100.0).round() as i32, (p.y * 100.0).round() as i32))
                .collect();
            v.sort_unstable();
            v
        };
        let mut all: Vec<(SymbolKind, Vec<(i32, i32)>)> =
            all_symbol_kinds().into_iter().map(|k| (k, sig(k))).collect();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(
                    all[i].1, all[j].1,
                    "{:?} 与 {:?} 几何签名相同（必须可辨）",
                    all[i].0, all[j].0
                );
            }
        }
        // 新符号逐一非空 + advance 为正（相邻不叠 ink 测试已覆盖全集）。
        for k in new_kinds {
            let path = symbol_path(k, 40.0);
            assert!(!path.subpaths.is_empty(), "{k:?} 必须有笔画");
            assert!(symbol_advance(&path, 40.0) > 0.0, "{k:?} advance 必须为正");
        }
        // 引用 all 防止借用生命周期告警的同时确认全集规模。
        assert_eq!(all.len(), 32, "全集 32 项（30 单位符号 + Sup/Sub 代表值 2；含缺陷修复新增 =）");
        all.clear();
    }

    /// [缺陷修复 2] 等号必须是**可见双横线**：矢量两笔画，横向够长、
    /// 双线净间距远大于笔宽（不与 σ/− 单横线混淆）。
    #[test]
    fn equals_is_visible_double_bar() {
        // S3 ε/δ 标签字号（与之一致地验证实际用例尺度）。
        let size = 26.0f32;
        let tracks = assemble_formula(&[FormulaPart::Sym(SymbolKind::Equals, size, TEXT_RGBA)], 480.0, 140.0);
        let paths: Vec<&PathTrack> = tracks
            .iter()
            .filter_map(|t| if let Track::Path(p) = t { Some(p) } else { None })
            .collect();
        assert_eq!(paths.len(), 2, "等号必须是 2 条独立笔画，实际 {}", paths.len());

        let mut ys: Vec<f32> = Vec::with_capacity(2);
        for p in &paths {
            assert_eq!(p.points.len(), 2, "每条横线必须是端点直连的 2 点");
            let (a, b) = (p.points[0], p.points[1]);
            assert!((a[1] - b[1]).abs() < 1e-6, "横线必须水平（y 同值）：{a:?} {b:?}");
            let len = (b[0] - a[0]).abs();
            assert!(len >= 0.5 * size, "横线长度须 ≥ 0.5·size（实测 {len}）");
            // 笔宽 = size·SYM_STROKE_K（MIN_SYM_STROKE_W 兜底）：同 ε/δ 体系。
            assert!(p.width >= MIN_SYM_STROKE_W, "笔宽须 ≥ {MIN_SYM_STROKE_W}，实际 {}", p.width);
            ys.push(a[1]);
        }
        ys.sort_by(f32::total_cmp);
        let gap = ys[1] - ys[0];
        assert!(gap >= 2.0 * MIN_SYM_STROKE_W, "双线净间距须 > 2×笔宽才不糊成一条：{gap}");
        // 整体垂直于数学轴 0.27em（+ device 翻转）：基线上方、不探出基线带。
        let mid = (ys[0] + ys[1]) * 0.5;
        let expect = 140.0 - 0.27 * size;
        assert!((mid - expect).abs() < 1e-3, "双线中心应居中于数学轴 {expect}，实测 {mid}");
    }

    /// 超宽文本边界（截断用例）：`assemble_formula` 对超长 run 不
    /// panic、产出有限轨、测量宽与渲染宽一致（不产出"半个字形"）。
    #[test]
    fn superwide_text_run_is_safe_and_whole() {
        let long = "ε".repeat(200) + &"Δ".repeat(200) + &"=0.41 ".repeat(50);
        let tracks = assemble_formula(
            &[FormulaPart::Text(long.clone(), FontKind::Regular, 28.0, TEXT_RGBA)],
            480.0,
            300.0,
        );
        // 单个 StaticText run：内容完整（不截半）。
        let statics: Vec<&StaticText> = tracks
            .iter()
            .filter_map(|t| match t {
                Track::Static(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(statics.len(), 1, "超宽文本仍是单一 run（截断由排版层负责）");
        assert_eq!(statics[0].content, long, "内容不得被截半");
        assert!(statics[0].center_x.map(|c| c.is_finite()).unwrap_or(false));
        // 实测宽 = layout_text 宽（测量即渲染）。
        let regular = font_regular();
        let params = TextParams {
            font_size: 28.0,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Center,
        };
        let w = layout_text(&regular, &long, &params).width;
        assert!((statics[0].center_x.unwrap() - (480.0 - w * 0.5 + w * 0.5)).abs() < 1e-3);
    }

    /// 子路径 x 范围（起点 + 全部几何点）。
    fn contour_x_range(sub: &prism_raster::path::SubPath) -> (f32, f32) {
        let (lo, hi, _, _) = contour_bbox(sub);
        (lo, hi)
    }

    /// 子路径包围盒（原始几何点，含曲线控制点——保守上界）。
    fn contour_bbox(sub: &prism_raster::path::SubPath) -> (f32, f32, f32, f32) {
        let mut pts = Vec::new();
        visit_points(sub, &mut |p| pts.push(p));
        let x0 = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let x1 = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let y0 = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let y1 = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        (x0, x1, y0, y1)
    }

    /// 子路径首点 x（中杠左端判定用）。
    fn bar_endpoint_x(sub: &prism_raster::path::SubPath) -> f32 {
        sub_endpoint(sub).map(|p| p.x).unwrap_or(sub.start.x)
    }

    // ---- [任务 5·缺陷 3] 限位组（FormulaPart::Limits） ----

    /// S5 积分式 parts（`∫₀² x² dx = 8/3`；限位组承载 ₀ / ²）。
    fn s5_integral_parts() -> Vec<FormulaPart> {
        vec![
            FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
            FormulaPart::Limits {
                lo: String::from("0"),
                hi: String::from("2"),
                size: 26.0,
                color: TEXT_RGBA,
            },
            FormulaPart::Text(String::from(" x"), FontKind::Regular, 30.0, TEXT_RGBA),
            FormulaPart::Sym(SymbolKind::Sup(2), 30.0, TEXT_RGBA),
            FormulaPart::Text(String::from(" dx = 8/3"), FontKind::Regular, 30.0, TEXT_RGBA),
        ]
    }

    /// 限位组纵向堆叠：hi/lo 落于 ∫ 墨迹上下方，且 advance = 0
    /// （后续 run 不被双计推开——修复前 ∫ 的 ₀ / ² 漂在 ∫ 右侧 8–21px）。
    #[test]
    fn limits_group_stacks_above_below_with_zero_advance() {
        let baseline = 140.0f32;
        let with = assemble_formula(&s5_integral_parts(), 480.0, baseline);
        // 对照：去掉限位组的同公式（后续 run 位置必须逐位一致）。
        let without = assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" x"), FontKind::Regular, 30.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Sup(2), 30.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" dx = 8/3"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            480.0,
            baseline,
        );
        let x_cx = |tracks: &[Track]| {
            tracks
                .iter()
                .find_map(|t| match t {
                    Track::Static(s) if s.content == " x" => s.center_x,
                    _ => None,
                })
                .expect("须有 x run")
        };
        assert_eq!(x_cx(&with), x_cx(&without), "限位组 advance 必须为 0");

        // ∫ 墨迹包围盒（slot 局部 px；数学 y↑ ⇒ 设备 y↓ = baseline − y）。
        let int_path = symbol_path(SymbolKind::Integral, 44.0);
        let bb = ink_bbox(&int_path);
        let int_top_dev = baseline - bb[3]; // 墨迹顶（设备最小 y）
        let int_bot_dev = baseline - bb[2]; // 墨迹底（设备最大 y）
        // 限位水平中心 = ∫ slot 起点 + 墨迹水平中心（不漂右）。
        // slot 起点 = 公式左缘 = center_x − total/2；total 含**全部** run 的
        // advance（∫ + Limits(0) + " x" + Sup(2) + " dx = 8/3"）——只计入
        // int_adv 会低估左缘偏移，期望值随之错位。
        let int_adv = symbol_advance(&int_path, 44.0);
        let total = int_adv
            + text_run_width(" x", FontKind::Regular, 30.0, 0.0)
            + symbol_advance(&symbol_path(SymbolKind::Sup(2), 30.0), 30.0)
            + text_run_width(" dx = 8/3", FontKind::Regular, 30.0, 0.0);
        let slot_start = 480.0 - total * 0.5; // ∫ 为首 part ⇒ slot 起点 = 公式左缘
        let expect_cx = slot_start + (bb[0] + bb[1]) * 0.5;

        let (hi, lo) = {
            let mut hi = None;
            let mut lo = None;
            for t in &with {
                if let Track::Static(s) = t {
                    match s.content.as_str() {
                        "2" => hi = Some((s.center_x, s.y)),
                        "0" => lo = Some((s.center_x, s.y)),
                        _ => {}
                    }
                }
            }
            (hi.expect("上限 2"), lo.expect("下限 0"))
        };
        // 水平：限位居中于 ∫ 墨迹（而非其右侧）。
        assert!((hi.0.unwrap() - expect_cx).abs() < 1e-3, "上限水平居中于 ∫");
        assert!((lo.0.unwrap() - expect_cx).abs() < 1e-3, "下限水平居中于 ∫");
        // 纵向：上限基线 = ∫ 顶 − gap；下限基线 = ∫ 底 + gap + 帽高。
        assert!(
            (hi.1 - (int_top_dev - LIM_GAP_K * 26.0)).abs() < 1e-3,
            "上限贴 ∫ 顶（{} vs {}）",
            hi.1,
            int_top_dev - LIM_GAP_K * 26.0
        );
        assert!(
            (lo.1 - (int_bot_dev + LIM_GAP_K * 26.0 + LIM_CAP_K * 26.0)).abs() < 1e-3,
            "下限贴 ∫ 底（{} vs {}）",
            lo.1,
            int_bot_dev + LIM_GAP_K * 26.0 + LIM_CAP_K * 26.0
        );
        assert!(hi.1 < int_top_dev, "上限墨迹须在 ∫ 顶上方");
        assert!(lo.1 > int_bot_dev, "下限墨迹须在 ∫ 底下方");
    }

    /// 限位组边角：空限位不发射；无前置算子时健全退化（不 panic、
    /// 位置有限）；非法字号整组跳过；S6 的字母上限（"x"）同构堆叠。
    #[test]
    fn limits_group_edges_and_letter_hi() {
        // 空限位：不发射任何轨；后续 run 位置不受影响。
        let empty = assemble_formula(
            &[
                FormulaPart::Limits {
                    lo: String::new(),
                    hi: String::new(),
                    size: 26.0,
                    color: TEXT_RGBA,
                },
                FormulaPart::Text(String::from("ok"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            480.0,
            140.0,
        );
        assert_eq!(empty.len(), 1, "空限位不发射（仅 ok run）");

        // 无前置算子：退化为标称堆叠（2 条限位 + 位置有限）。
        let orphan = assemble_formula(
            &[FormulaPart::Limits {
                lo: String::from("0"),
                hi: String::from("x"),
                size: 22.0,
                color: TEXT_RGBA,
            }],
            480.0,
            140.0,
        );
        assert_eq!(orphan.len(), 2, "无限位的 Limits 仍发射上下限两条");
        for t in &orphan {
            if let Track::Static(s) = t {
                assert!(s.center_x.map_or(false, |c| c.is_finite()) && s.y.is_finite());
            }
        }

        // 非法字号：整组跳过（与文本 run 同纪律，不 panic）。
        let bad = assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
                FormulaPart::Limits {
                    lo: String::from("0"),
                    hi: String::from("2"),
                    size: f32::NAN,
                    color: TEXT_RGBA,
                },
            ],
            480.0,
            140.0,
        );
        assert!(bad.iter().all(|t| matches!(t, Track::Path(_))), "非法字号时限位不发射");

        // S6 FTC 式：字母上限 "x" 与数字下限 "0" 同构堆叠于 ∫ 上下方。
        let ftc = assemble_formula(
            &[
                FormulaPart::Text(String::from("d/dx "), FontKind::Regular, 30.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
                FormulaPart::Limits {
                    lo: String::from("0"),
                    hi: String::from("x"),
                    size: 22.0,
                    color: TEXT_RGBA,
                },
                FormulaPart::Text(String::from(" f(t) dt = f(x)"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            480.0,
            250.0,
        );
        let (hi, lo) = {
            let mut hi = None;
            let mut lo = None;
            for t in &ftc {
                if let Track::Static(s) = t {
                    match s.content.as_str() {
                        "x" => hi = Some(s.y),
                        "0" => lo = Some(s.y),
                        _ => {}
                    }
                }
            }
            (hi.expect("上限 x"), lo.expect("下限 0"))
        };
        assert!(hi < 250.0 && lo > 250.0, "字母上限亦纵向上/下堆叠（{hi} / {lo}）");
    }

    /// [任务 5] 位宽探针：`text_run_width` 与 `layout_text` 逐位同参
    /// （测量即渲染）；负串较非负串多一枚 minus 宽（S4 过零补偿基准）；
    /// 非法输入 → 0（不 panic）。
    #[test]
    fn text_run_width_matches_layout_and_minus_delta() {
        let w_neg = text_run_width("m = -2.0000", FontKind::Bold, 30.0, 1.0);
        let w_pos = text_run_width("m = 2.0000", FontKind::Bold, 30.0, 1.0);
        assert!(
            w_neg > w_pos && (w_neg - w_pos) > 4.0,
            "minus 宽差须可测（{w_neg} vs {w_pos}）"
        );
        // 与 layout_text 直接实测一致（同参：Bold 30 / 字距 1.0）。
        let bold = font_bold();
        let params = TextParams {
            font_size: 30.0,
            letter_spacing: 1.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Center,
        };
        let direct = layout_text(&bold, "m = -2.0000", &params).width;
        assert!((w_neg - direct).abs() < 1e-3, "探针须与 layout_text 逐位一致");
        // 表宽数字：同位宽 ⇒ 段内读数串宽恒定（补偿不变式前提）。
        let w_a = text_run_width("n = 2", FontKind::Bold, 28.0, 1.0);
        let w_b = text_run_width("n = 32", FontKind::Bold, 28.0, 1.0);
        assert!((w_b - w_a) > 4.0, "1 位与 2 位 n 读数宽差可测（{w_a} vs {w_b}）");
        let w_8 = text_run_width("n = 8", FontKind::Bold, 28.0, 1.0);
        assert!((w_8 - w_a).abs() < 1e-3, "同位宽数字串宽恒定（表宽数字）");
        // 非法输入 → 0。
        assert_eq!(text_run_width("", FontKind::Bold, 30.0, 1.0), 0.0);
        assert_eq!(text_run_width("x", FontKind::Bold, f32::NAN, 1.0), 0.0);
        assert_eq!(text_run_width("x", FontKind::Bold, -1.0, 1.0), 0.0);
    }

    // -----------------------------------------------------------------------
    // v3 排版升级：数学间距表（math_gap）接入与符号比例重调
    // -----------------------------------------------------------------------

    /// 二元算子两侧呼吸间距 > 变量-括号贴近间距，且 > 普通符号间隙
    /// `SYM_GAP`；变量与变量为可辨小间距。
    #[test]
    fn math_gap_binary_breathing_beats_var_paren_tightness() {
        let breath_r = math_gap(GapKind::Binary, GapKind::Var);
        let breath_l = math_gap(GapKind::Var, GapKind::Binary);
        let var_paren = math_gap(GapKind::Var, GapKind::Open);
        let paren_var = math_gap(GapKind::Open, GapKind::Var);
        assert!(
            breath_r > var_paren && breath_l > paren_var,
            "二元算子两侧（{breath_r} / {breath_l}）须大于变量-括号（{var_paren} / {paren_var}）"
        );
        assert!(
            breath_r > SYM_GAP && breath_l > SYM_GAP,
            "呼吸间距须大于普通符号间隙 {SYM_GAP}"
        );
        let vv = math_gap(GapKind::Var, GapKind::Var);
        assert!(vv > 0.0 && vv < SYM_GAP, "变量间小间距贴近：{vv}");
    }

    /// 上下标与基座、∫ 与被积式：零 / 负间距（贴合），明显小于普通间隙；
    /// 函数名 / 撇号 / 量词 / 显式空白边界同样零间距（不双计）。
    #[test]
    fn math_gap_scripts_and_bigops_stay_tight() {
        assert!(math_gap(GapKind::Var, GapKind::Script) <= 0.0, "上标贴基座");
        assert!(math_gap(GapKind::Var, GapKind::Script) <= 0.0, "下标贴基座");
        assert!(math_gap(GapKind::BigOp, GapKind::Var) <= 0.0, "∫ 贴被积式");
        assert!(math_gap(GapKind::Var, GapKind::Script) < SYM_GAP);
        assert!(math_gap(GapKind::BigOp, GapKind::Var) < SYM_GAP);
        assert_eq!(math_gap(GapKind::Var, GapKind::Mark), 0.0, "f′ 贴基座");
        assert_eq!(math_gap(GapKind::Func, GapKind::Open), 0.0, "sin( 贴");
        assert_eq!(math_gap(GapKind::Quant, GapKind::Var), 0.0, "∀e 贴");
        assert_eq!(math_gap(GapKind::Spaced, GapKind::Binary), 0.0, "显式空白抑制");
        assert_eq!(math_gap(GapKind::Binary, GapKind::Spaced), 0.0, "显式空白抑制");
    }

    /// 接入验证：紧凑 run「x − y」减号两侧呼吸间距落地（跨 Sym 的
    /// 盒缘跨度 = 呼吸 + 符号 advance + 呼吸，逐位可算）；「x(y)」
    /// 变量-括号零间距——前者明显宽于后者。
    #[test]
    fn assemble_formula_wires_math_gap_between_tight_runs() {
        let minus = assemble_formula(
            &[
                FormulaPart::Text(String::from("x"), FontKind::Bold, 40.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Minus, 40.0, TEXT_RGBA),
                FormulaPart::Text(String::from("y"), FontKind::Bold, 40.0, TEXT_RGBA),
            ],
            480.0,
            300.0,
        );
        let paren = assemble_formula(
            &[
                FormulaPart::Text(String::from("x"), FontKind::Bold, 40.0, TEXT_RGBA),
                FormulaPart::Text(String::from("(y)"), FontKind::Bold, 40.0, TEXT_RGBA),
            ],
            480.0,
            300.0,
        );
        // 第一段文本右盒缘 → 第二段文本左盒缘的跨度（盒缘 = 中心 ∓ 半宽）。
        let span_after_first = |tracks: &[Track]| -> f32 {
            let statics: Vec<&StaticText> = tracks
                .iter()
                .filter_map(|t| match t {
                    Track::Static(s) => Some(s),
                    _ => None,
                })
                .collect();
            assert!(statics.len() >= 2, "至少两段文本 run");
            let w1 = text_run_width(&statics[0].content, FontKind::Bold, 40.0, 0.0);
            let w2 = text_run_width(&statics[1].content, FontKind::Bold, 40.0, 0.0);
            (statics[1].center_x.unwrap() - w2 * 0.5) - (statics[0].center_x.unwrap() + w1 * 0.5)
        };
        let g_paren = span_after_first(&paren);
        assert!(g_paren.abs() < 1e-3, "变量-括号零间距：{g_paren}");
        let span = span_after_first(&minus);
        let expect = MATH_GAP_BREATH * 40.0
            + symbol_advance(&symbol_path(SymbolKind::Minus, 40.0), 40.0)
            + MATH_GAP_BREATH * 40.0;
        assert!(
            (span - expect).abs() < 1e-3,
            "减号两侧呼吸间距落地：{span} vs {expect}"
        );
        assert!(span > g_paren + 1.0, "二元算子间距明显大于变量-括号贴近");
    }

    /// 比例重调锁定：∫ 高 ≈ 行高 0.72 的 1.6–1.8 倍；符号描边（分数线）
    /// 0.04–0.06·size；∑ / ∏ 升至大算子尺寸；√ 顶超出帽高。
    #[test]
    fn symbol_proportions_lean_latex() {
        let size = 40.0f32;
        let int_bb = ink_bbox(&symbol_path(SymbolKind::Integral, size));
        let ratio = (int_bb[3] - int_bb[2]) / (0.72 * size);
        assert!((1.6..=1.8).contains(&ratio), "∫ 高 / 行高 = {ratio}");
        let stroke = size * SYM_STROKE_K;
        assert!(
            (0.04 * size..=0.06 * size).contains(&stroke),
            "分数线粗细 = {stroke}"
        );
        for kind in [SymbolKind::Sum, SymbolKind::Product] {
            let bb = ink_bbox(&symbol_path(kind, size));
            assert!(bb[3] > 0.85 * size, "{kind:?} 大算子高度不足：{}", bb[3]);
        }
        let sqrt_bb = ink_bbox(&symbol_path(SymbolKind::Sqrt, size));
        assert!(sqrt_bb[3] >= 0.75 * size, "√ 顶升高：{}", sqrt_bb[3]);
    }
}
