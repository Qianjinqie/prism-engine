//! 公式排版引擎（V2-j 第二步）：[`MathExpr`] → 字形布局。
//!
//! # 输出模型
//! [`MathLayout`] = 字形序列（[`MathGlyph`]：字符 / gid / 坐标 / 字号 / 步进）
//! + 规则线（[`MathRule`]：分式线、根号横线）+ 盒尺寸（`width` / `ascent` /
//! `descent`）。坐标原点在**公式基线左缘**，`x` 向右、`y` 向**上**（数学坐标；
//! 与 `layout.rs` 的设备坐标 y 向下的互操作见 [`MathLayout::place_on_baseline`]
//! 与 [`crate::layout::place_formula_on_baseline`]）。
//!
//! # 排版规则（工程近似，全部为确定性的 em 比值）
//! - **上下标**：脚本字号 = `script_scale`（0.7em）；上标基线上抬 `sup_rise`，
//!   下标基线下沉 `sub_drop`；脚本紧随基座右缘（薄间隙）；
//! - **分式**：分数线居中于数学轴 `axis_height`，粗细 `rule_thickness`；
//!   分子基线上移（线上 + `frac_gap` + 分子下降部），分母基线下移对称；
//!   分子 / 分母各自在分数线宽度内**居中**；
//! - **根号**：√ 字形按被开方体高度**纵向伸展**（`rad_size` 使墨迹覆盖横线），
//!   横线从 √ 右缘起**覆盖整个被开方体**并带右端余量；根指数缩到
//!   `sqrt_index_scale` 贴左上角；
//! - **大型算子**：算子放大到 `big_op_scale`，上下限在算子**正上/正下居中**
//!   （限位堆叠，不做横向推进——与 `symbols.rs` 的 `FormulaPart::Limits`
//!   语义一致）；
//! - **矩阵**：列宽 = 列内最大单元宽，列内每个单元**在该列居中**（同列
//!   x 中心一致）；行基线按行高累加 + `matrix_row_gap`；
//! - **原子间距**：二元算子两侧 `med_space`、关系符两侧 `thick_space`、
//!   标点后 `thin_space`（LaTeX 原子间距的工程近似）。
//!
//! # 字体与兜底
//! 字形 gid 解析：字体 cmap 优先，缺字回退 [`crate::math_symbols`] 合成符号表
//! （与 `layout.rs` 同一策略，绝不静默豆腐块）。非有限字号 / 非正字号 →
//! 空布局（不 panic）。
//!
//! # 确定性契约
//! 布局是 (字体字节, AST, 参数) 的纯函数：无隐藏状态、无随机源，浮点只用
//! ±×÷；同输入必得同 [`MathLayout`]（单测为几何断言）。

use crate::layout::{baseline_metrics, BaselineMetrics};
use crate::math_ast::{BigOpKind, MathExpr};
use crate::math_symbols::{advance_for_gid, fallback_gid, MATH_GID_BASE, MATH_UNITS_PER_EM};
use crate::ttf::Font;

/// 排版深度上限（防御恶意深嵌套，超限退化为空盒而非栈溢出）。
pub const MAX_LAYOUT_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// 排版参数
// ---------------------------------------------------------------------------

/// 公式排版参数（全部为相对基座字号的 em 比值；缩放时整体等比）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MathParams {
    /// 基座字号（px，em 框高）。
    pub font_size: f32,
    /// 上下标缩放比。
    pub script_scale: f32,
    /// 上标基线上抬（em）。
    pub sup_rise: f32,
    /// 下标基线下沉（em）。
    pub sub_drop: f32,
    /// 数学轴高度（em；分数线中心所在高度）。
    pub axis_height: f32,
    /// 规则线（分数线 / 根号横线）粗细（em）。
    pub rule_thickness: f32,
    /// 分式左右内边距（em）。
    pub frac_pad: f32,
    /// 分子 / 分母与分数线的间隙（em）。
    pub frac_gap: f32,
    /// 根号横线与被开方体之间的间隙（em）。
    pub sqrt_pad: f32,
    /// 根指数缩放比。
    pub sqrt_index_scale: f32,
    /// 大型算子放大比。
    pub big_op_scale: f32,
    /// 限位与算子墨迹的间隙（em，相对算子字号）。
    pub limit_gap: f32,
    /// 矩阵列间距（em）。
    pub matrix_col_gap: f32,
    /// 矩阵行间距（em）。
    pub matrix_row_gap: f32,
    /// 薄间距（em；标点后 / 脚本贴基座）。
    pub thin_space: f32,
    /// 中等间距（em；二元算子两侧）。
    pub med_space: f32,
    /// 厚间距（em；关系符两侧）。
    pub thick_space: f32,
    /// 字形墨迹上伸比（em，相对该字形字号）。
    pub ink_ascent: f32,
    /// 字形墨迹下伸比（em）。
    pub ink_descent: f32,
}

impl Default for MathParams {
    fn default() -> Self {
        MathParams {
            font_size: 32.0,
            script_scale: 0.70,
            sup_rise: 0.42,
            sub_drop: 0.20,
            axis_height: 0.25,
            rule_thickness: 0.05,
            frac_pad: 0.10,
            frac_gap: 0.10,
            sqrt_pad: 0.10,
            sqrt_index_scale: 0.50,
            big_op_scale: 1.45,
            limit_gap: 0.12,
            matrix_col_gap: 0.60,
            matrix_row_gap: 0.35,
            thin_space: 0.166,
            med_space: 0.222,
            thick_space: 0.277,
            ink_ascent: 0.70,
            ink_descent: 0.20,
        }
    }
}

// ---------------------------------------------------------------------------
// 输出模型
// ---------------------------------------------------------------------------

/// 字形在公式中的结构角色（诊断 / 测试分类用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphRole {
    /// 基座字形（普通变量 / 数字 / 符号）。
    Base,
    /// 上下标。
    Script,
    /// 分式分子。
    Numerator,
    /// 分式分母。
    Denominator,
    /// 被开方体。
    Radicand,
    /// 根指数。
    Index,
    /// 大型算子本体。
    BigOp,
    /// 大型算子上限。
    UpperLimit,
    /// 大型算子下限。
    LowerLimit,
    /// 矩阵单元。
    Matrix,
    /// 根号符号本体。
    Radical,
}

/// 规则线种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleKind {
    /// 分数线。
    FracBar,
    /// 根号横线。
    SqrtOverline,
}

/// 已定位字形（公式坐标：原点 = 基线左缘，x 右、y 上）。
#[derive(Debug, Clone, PartialEq)]
pub struct MathGlyph {
    /// 字符。
    pub ch: char,
    /// 字形 id（cmap 缺字 → `math_symbols` 合成 gid；双缺失 → 0 = .notdef）。
    pub gid: u16,
    /// 笔位 x（公式原点起，px）。
    pub x: f32,
    /// 基线上抬量（px，y 向上；正 = 基线之上）。
    pub y: f32,
    /// 该字形字号（px；上下标 / 根指数等按缩放比缩小）。
    pub size: f32,
    /// 步进宽（px）。
    pub advance: f32,
    /// 结构角色。
    pub role: GlyphRole,
}

/// 规则线（分式线 / 根号横线）：水平线段 + 粗细。
#[derive(Debug, Clone, PartialEq)]
pub struct MathRule {
    /// 种类。
    pub kind: RuleKind,
    /// 左端 x（px）。
    pub x0: f32,
    /// 右端 x（px）。
    pub x1: f32,
    /// 中心线高度（px，y 向上）。
    pub y: f32,
    /// 粗细（px）。
    pub thickness: f32,
}

impl MathRule {
    /// 线宽（右端 − 左端，恒 ≥ 0）。
    pub fn span(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    /// 线的上缘（y 向上）。
    pub fn top(&self) -> f32 {
        self.y + self.thickness * 0.5
    }

    /// 线的下缘（y 向上）。
    pub fn bottom(&self) -> f32 {
        self.y - self.thickness * 0.5
    }
}

/// 排版结果（公式坐标，原点 = 基线左缘）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MathLayout {
    /// 字形（按视觉阅读序）。
    pub glyphs: Vec<MathGlyph>,
    /// 规则线（分数线 / 根号横线）。
    pub rules: Vec<MathRule>,
    /// 内容宽（px）。
    pub width: f32,
    /// 基线以上高度（px）。
    pub ascent: f32,
    /// 基线以下深度（px）。
    pub descent: f32,
}

impl MathLayout {
    /// 取首个指定字符字形（测试 / 诊断）。
    pub fn glyph(&self, ch: char) -> Option<&MathGlyph> {
        self.glyphs.iter().find(|g| g.ch == ch)
    }

    /// 取首个指定角色的字形。
    pub fn glyph_with_role(&self, role: GlyphRole) -> Option<&MathGlyph> {
        self.glyphs.iter().find(|g| g.role == role)
    }

    /// 指定角色的全部字形。
    pub fn glyphs_with_role(&self, role: GlyphRole) -> impl Iterator<Item = &MathGlyph> {
        self.glyphs.iter().filter(move |g| g.role == role)
    }

    /// 内容盒高（ascent + descent，px）。
    pub fn height(&self) -> f32 {
        self.ascent + self.descent
    }

    /// 是否为空格（无字形且无线）。
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty() && self.rules.is_empty()
    }

    /// 字形序列（诊断 / 断言用：渲染出字的直接证据）。
    pub fn text(&self) -> String {
        self.glyphs.iter().map(|g| g.ch).collect()
    }

    /// 与文本基线互操作：把公式字形换算为**设备坐标**（y 向下的 [`DeviceGlyph`]）。
    ///
    /// 语义：`baseline_y` 是公式基线的设备 y（与 `layout.rs` 的
    /// `GlyphPos::y` 同一口径），`dx` 是公式左缘在设备 x 上的位置。
    /// 输出字形 y = `baseline_y − glyph.y`（数学 y 向上 → 设备 y 向下翻转），
    /// 因此公式基座字形（`y == 0`）**与同行文本基线逐位重合**。
    pub fn place_on_baseline(&self, dx: f32, baseline_y: f32) -> Vec<DeviceGlyph> {
        self.glyphs
            .iter()
            .map(|g| DeviceGlyph {
                ch: g.ch,
                gid: g.gid,
                x: dx + g.x,
                y: baseline_y - g.y,
                advance: g.advance,
            })
            .collect()
    }

    /// 基线互操作所需的行高（px）：取内容盒与字体行度量的较大者。
    ///
    /// 复用 [`crate::layout::baseline_metrics`]（唯一基线度量入口），保证公式
    /// 与普通文本同行混排时行盒高度口径一致。
    pub fn line_extent(&self, font: &Font, font_size: f32) -> (f32, f32) {
        let m: BaselineMetrics = baseline_metrics(font, font_size);
        (self.ascent.max(m.ascent), self.descent.max(m.descent))
    }
}

/// 设备坐标字形（y 向下，与 `layout.rs::GlyphPos` 同口径的精简形式）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceGlyph {
    /// 字符。
    pub ch: char,
    /// 字形 id。
    pub gid: u16,
    /// 笔位 x（px）。
    pub x: f32,
    /// 基线 y（px，设备坐标 y 向下）。
    pub y: f32,
    /// 步进宽（px）。
    pub advance: f32,
}

// ---------------------------------------------------------------------------
// 排版入口
// ---------------------------------------------------------------------------

/// 排版公式（字体 + AST + 参数 → 字形布局）。
///
/// 非法参数（非有限 / 非正字号）→ 空布局（不 panic）。
pub fn layout_math(font: &Font, expr: &MathExpr, params: &MathParams) -> MathLayout {
    if !params.font_size.is_finite() || params.font_size <= 0.0 {
        return MathLayout::default();
    }
    let b: MBox = layout_expr(font, expr, params.font_size, params, 0);
    MathLayout {
        glyphs: b.glyphs,
        rules: b.rules,
        width: b.width,
        ascent: b.ascent,
        descent: b.descent,
    }
}

/// 排版公式（缺省参数，字号 `font_size`）。
pub fn layout_math_sized(font: &Font, expr: &MathExpr, font_size: f32) -> MathLayout {
    layout_math(font, expr, &MathParams { font_size, ..MathParams::default() })
}

// ---------------------------------------------------------------------------
// 内部盒模型
// ---------------------------------------------------------------------------

/// 中间盒（相对自身基线左缘；组合时平移）。
#[derive(Debug, Clone, Default)]
struct MBox {
    glyphs: Vec<MathGlyph>,
    rules: Vec<MathRule>,
    width: f32,
    ascent: f32,
    descent: f32,
}

impl MBox {
    /// 整体平移（+x 右、+y 上）。
    fn shift(&mut self, dx: f32, dy: f32) {
        for g in &mut self.glyphs {
            g.x += dx;
            g.y += dy;
        }
        for r in &mut self.rules {
            r.x0 += dx;
            r.x1 += dx;
            r.y += dy;
        }
    }

    /// 把 `other` 追加到右缘（间隔 `gap`），并归并盒尺寸。
    fn append(&mut self, other: &MBox, gap: f32) {
        if other.glyphs.is_empty() && other.rules.is_empty() {
            // 空盒不占位（空矩阵单元 / 空分组）。
            return;
        }
        let mut moved: MBox = other.clone();
        moved.shift(self.width + gap, 0.0);
        self.glyphs.extend(moved.glyphs);
        self.rules.extend(moved.rules);
        self.width = (self.width + gap + other.width).max(0.0);
        self.ascent = self.ascent.max(other.ascent);
        self.descent = self.descent.max(other.descent);
    }

    /// 重设字形角色（结构归属：分子 / 分母 / 上限 …）。
    fn set_role(&mut self, role: GlyphRole) {
        for g in &mut self.glyphs {
            g.role = role;
        }
    }
}

// ---------------------------------------------------------------------------
// 字形度量
// ---------------------------------------------------------------------------

/// 字符 → gid（字体 cmap 优先，缺字回退合成符号表，双缺失 → 0 = .notdef）。
fn resolve_gid(font: &Font, ch: char) -> u16 {
    font.glyph_index(ch).or_else(|| fallback_gid(ch)).unwrap_or(0)
}

/// gid 的步进宽（px）：合成 gid 走符号表（字体单位 1000），真实字形走 hmtx。
fn advance_px(font: &Font, gid: u16, size: f32) -> f32 {
    let (units, upem): (u16, f32) = if gid >= MATH_GID_BASE {
        (advance_for_gid(gid).unwrap_or(0), MATH_UNITS_PER_EM as f32)
    } else {
        (font.advance(gid), font.units_per_em() as f32)
    };
    units as f32 * (size / upem)
}

/// 单字盒（一个字形 + 墨迹上下伸）。
fn leaf(font: &Font, ch: char, size: f32, role: GlyphRole, params: &MathParams) -> MBox {
    let gid: u16 = resolve_gid(font, ch);
    let adv: f32 = advance_px(font, gid, size);
    MBox {
        glyphs: vec![MathGlyph { ch, gid, x: 0.0, y: 0.0, size, advance: adv, role }],
        rules: Vec::new(),
        width: adv,
        ascent: params.ink_ascent * size,
        descent: params.ink_descent * size,
    }
}

/// 字符串逐字符横排为盒（标识符 / 数字串）。
fn text_box(
    font: &Font,
    text: &str,
    size: f32,
    role: GlyphRole,
    params: &MathParams,
) -> MBox {
    let mut out: MBox = MBox::default();
    for ch in text.chars() {
        let b: MBox = leaf(font, ch, size, role, params);
        out.append(&b, 0.0);
    }
    out
}

// ---------------------------------------------------------------------------
// 原子间距
// ---------------------------------------------------------------------------

/// 间距原子类别（LaTeX 原子类型的工程近似）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// 普通原子（变量 / 数字 / 大结构）。
    Ord,
    /// 二元算子（+ − × ÷ ±）。
    Bin,
    /// 关系符（= < > ≤ ≥ ≈ ≠ → ⇒）。
    Rel,
    /// 开括号。
    Open,
    /// 闭括号。
    Close,
    /// 标点（, ; :）。
    Punct,
}

/// 表达式**首**原子的类别。
fn first_class(e: &MathExpr) -> Class {
    match e {
        MathExpr::Row(items) => items.first().map_or(Class::Ord, first_class),
        other => atom_class(other),
    }
}

/// 表达式**末**原子的类别。
fn last_class(e: &MathExpr) -> Class {
    match e {
        MathExpr::Row(items) => items.last().map_or(Class::Ord, last_class),
        other => atom_class(other),
    }
}

/// 单节点原子类别。
fn atom_class(e: &MathExpr) -> Class {
    let ch: Option<char> = match e {
        MathExpr::Sym(c) | MathExpr::Text(c) => Some(*c),
        _ => None,
    };
    match ch {
        Some('+') | Some('−') | Some('-') | Some('±') | Some('∓') | Some('×') | Some('÷')
        | Some('⋅') | Some('*') => Class::Bin,
        Some('=') | Some('<') | Some('>') | Some('≤') | Some('≥') | Some('≈') | Some('≅')
        | Some('≡') | Some('≠') | Some('∼') | Some('→') | Some('←') | Some('⇒') | Some('⇔')
        | Some('∈') | Some('≪') => Class::Rel,
        Some('(') | Some('[') | Some('{') => Class::Open,
        Some(')') | Some(']') | Some('}') | Some('|') => Class::Close,
        Some(',') | Some(';') | Some(':') => Class::Punct,
        _ => Class::Ord,
    }
}

/// 相邻原子间距（px）：二元算子 / 关系符两侧呼吸，标点后薄间距，括号内侧贴合。
fn space_between(prev: Class, next: Class, size: f32, params: &MathParams) -> f32 {
    use Class::{Bin, Close, Open, Punct, Rel};
    let k: f32 = match (prev, next) {
        // 括号与内容贴合（LaTeX open/close 内侧零间距）。
        (Open, _) | (_, Close) => 0.0,
        // 单目正负号（行首 / 相邻算子）不呼吸。
        (Bin, Bin) => 0.0,
        (Bin, _) | (_, Bin) => params.med_space,
        (Rel, _) | (_, Rel) => params.thick_space,
        (Punct, _) => params.thin_space,
        _ => 0.0,
    };
    k * size
}

// ---------------------------------------------------------------------------
// 递归排版
// ---------------------------------------------------------------------------

/// 递归排版一个表达式（返回相对自身基线左缘的盒）。
fn layout_expr(font: &Font, expr: &MathExpr, size: f32, params: &MathParams, depth: usize) -> MBox {
    if depth > MAX_LAYOUT_DEPTH {
        return MBox::default();
    }
    let base_role: GlyphRole = GlyphRole::Base;
    match expr {
        MathExpr::Row(items) => {
            let mut out: MBox = MBox::default();
            let mut prev: Option<Class> = None;
            for item in items {
                let b: MBox = layout_expr(font, item, size, params, depth + 1);
                if b.glyphs.is_empty() && b.rules.is_empty() {
                    continue; // 空槽（空单元）不参与间距
                }
                let gap: f32 = match prev {
                    Some(p) => space_between(p, first_class(item), size, params),
                    None => 0.0,
                };
                out.append(&b, gap);
                prev = Some(last_class(item));
            }
            out
        }
        MathExpr::Ident(s) => text_box(font, s, size, base_role, params),
        MathExpr::Num(s) => text_box(font, s, size, base_role, params),
        MathExpr::Sym(c) | MathExpr::Text(c) => leaf(font, *c, size, base_role, params),
        MathExpr::Sup(base, sup) => layout_scripts(font, base, None, Some(sup), size, params, depth),
        MathExpr::Sub(base, sub) => layout_scripts(font, base, Some(sub), None, size, params, depth),
        MathExpr::SubSup(base, sub, sup) => {
            layout_scripts(font, base, Some(sub), Some(sup), size, params, depth)
        }
        MathExpr::Frac(num, den) => layout_frac(font, num, den, size, params, depth),
        MathExpr::Sqrt { index, body } => {
            layout_sqrt(font, index.as_deref(), body, size, params, depth)
        }
        MathExpr::BigOp { op, sub, sup } => layout_big_op(
            font,
            *op,
            sub.as_deref(),
            sup.as_deref(),
            size,
            params,
            depth,
        ),
        MathExpr::Matrix(m) => layout_matrix(font, m, size, params, depth),
    }
}

/// 上下标排版：基座 + 右上 / 右下脚本盒。
fn layout_scripts(
    font: &Font,
    base: &MathExpr,
    sub: Option<&MathExpr>,
    sup: Option<&MathExpr>,
    size: f32,
    params: &MathParams,
    depth: usize,
) -> MBox {
    let mut b: MBox = layout_expr(font, base, size, params, depth + 1);
    let script_size: f32 = size * params.script_scale;
    let kern: f32 = params.thin_space * size; // 基座与脚本之间的薄间隙
    let script_x: f32 = b.width + kern;
    let base_ascent: f32 = b.ascent;
    let base_descent: f32 = b.descent;

    let mut sup_box: Option<MBox> = sup.map(|e| {
        let mut sb: MBox = layout_expr(font, e, script_size, params, depth + 1);
        sb.set_role(GlyphRole::Script);
        sb
    });
    let mut sub_box: Option<MBox> = sub.map(|e| {
        let mut sb: MBox = layout_expr(font, e, script_size, params, depth + 1);
        sb.set_role(GlyphRole::Script);
        sb
    });

    let mut asc: f32 = base_ascent;
    let mut desc: f32 = base_descent;
    let mut right: f32 = b.width;
    if let Some(sb) = sup_box.as_mut() {
        let dy: f32 = params.sup_rise * size;
        sb.shift(script_x, dy);
        asc = asc.max(dy + sb.ascent);
        right = right.max(script_x + sb.width);
        b.glyphs.extend(sb.glyphs.iter().cloned());
        b.rules.extend(sb.rules.iter().cloned());
    }
    if let Some(sb) = sub_box.as_mut() {
        let dy: f32 = -params.sub_drop * size;
        sb.shift(script_x, dy);
        desc = desc.max(-dy + sb.descent);
        right = right.max(script_x + sb.width);
        b.glyphs.extend(sb.glyphs.iter().cloned());
        b.rules.extend(sb.rules.iter().cloned());
    }
    b.width = right;
    b.ascent = asc;
    b.descent = desc;
    b
}

/// 分式排版：线上分子、线下分母，各自居中于分数线。
fn layout_frac(
    font: &Font,
    num: &MathExpr,
    den: &MathExpr,
    size: f32,
    params: &MathParams,
    depth: usize,
) -> MBox {
    let inner_size: f32 = size * 0.95; // 分子 / 分母相对基座略缩（观感接近 LaTeX）
    let mut num_box: MBox = layout_expr(font, num, inner_size, params, depth + 1);
    let mut den_box: MBox = layout_expr(font, den, inner_size, params, depth + 1);
    num_box.set_role(GlyphRole::Numerator);
    den_box.set_role(GlyphRole::Denominator);

    let thickness: f32 = params.rule_thickness * size;
    let axis: f32 = params.axis_height * size;
    let gap: f32 = params.frac_gap * size;
    let pad: f32 = params.frac_pad * size;

    let width: f32 = num_box.width.max(den_box.width) + pad * 2.0;
    // 分子基线上移：分数线上缘 + 间隙 + 分子下降部。
    let bar_top: f32 = axis + thickness * 0.5;
    let bar_bottom: f32 = axis - thickness * 0.5;
    let num_dy: f32 = bar_top + gap + num_box.descent;
    let den_dy: f32 = bar_bottom - gap - den_box.ascent;

    let num_dx: f32 = (width - num_box.width) * 0.5;
    let den_dx: f32 = (width - den_box.width) * 0.5;
    num_box.shift(num_dx, num_dy);
    den_box.shift(den_dx, den_dy);
    let num_ascent: f32 = num_box.ascent;
    let den_descent: f32 = den_box.descent;

    let mut out: MBox = num_box;
    out.glyphs.extend(den_box.glyphs);
    out.rules.extend(den_box.rules);
    out.rules.push(MathRule {
        kind: RuleKind::FracBar,
        x0: 0.0,
        x1: width,
        y: axis,
        thickness,
    });
    out.width = width;
    // 盒尺寸：上缘 = 分子基线 + 分子上伸；下缘 = 分母基线 − 分母下伸。
    out.ascent = num_dy + num_ascent;
    out.descent = -den_dy + den_descent;
    out
}

/// 根号排版：伸展的 √ + 覆盖被开方体的横线（+ 可选根指数）。
fn layout_sqrt(
    font: &Font,
    index: Option<&MathExpr>,
    body: &MathExpr,
    size: f32,
    params: &MathParams,
    depth: usize,
) -> MBox {
    let mut body_box: MBox = layout_expr(font, body, size, params, depth + 1);
    body_box.set_role(GlyphRole::Radicand);

    let pad: f32 = params.sqrt_pad * size;
    let thickness: f32 = params.rule_thickness * size;
    // 横线高度 = 被开方体上缘 + 间隙。
    let overline_y: f32 = body_box.ascent + pad;
    // √ 纵向伸展：使其墨迹上伸覆盖横线（rad_size · ink_ascent ≥ overline_y）。
    let rad_size: f32 = (overline_y / params.ink_ascent).max(size);
    let rad_box: MBox = leaf(font, '√', rad_size, GlyphRole::Radical, params);
    let rad_w: f32 = rad_box.width;

    let right_pad: f32 = pad * 0.5;
    let width: f32 = rad_w + body_box.width + right_pad;

    let mut out: MBox = rad_box;
    out.glyphs.extend(body_box.glyphs.iter().cloned());
    out.rules.extend(body_box.rules.iter().cloned());
    // 被开方体右移 √ 的步进宽（基线对齐）。
    for g in out.glyphs.iter_mut().filter(|g| g.role == GlyphRole::Radicand) {
        g.x += rad_w;
    }
    for r in out.rules.iter_mut().filter(|r| r.kind != RuleKind::SqrtOverline) {
        r.x0 += rad_w;
        r.x1 += rad_w;
    }
    // 横线：从 √ 右缘略内收（与根号笔画接合）到被开方体右缘之外。
    out.rules.push(MathRule {
        kind: RuleKind::SqrtOverline,
        x0: rad_w * 0.55,
        x1: width,
        y: overline_y,
        thickness,
    });

    let mut asc: f32 = overline_y + thickness * 0.5;
    asc = asc.max(rad_size * params.ink_ascent);

    // 根指数：缩到 sqrt_index_scale，贴横线左上方。
    if let Some(idx) = index {
        let idx_size: f32 = size * params.sqrt_index_scale;
        let mut idx_box: MBox = layout_expr(font, idx, idx_size, params, depth + 1);
        idx_box.set_role(GlyphRole::Index);
        let idx_dy: f32 = overline_y + thickness * 0.5 + params.thin_space * size * 0.25;
        idx_box.shift(0.0, idx_dy);
        asc = asc.max(idx_dy + idx_box.ascent);
        out.glyphs.extend(idx_box.glyphs);
        out.rules.extend(idx_box.rules);
    }

    out.width = width;
    out.ascent = asc;
    out.descent = body_box.descent;
    out
}

/// 大型算子排版：放大算子 + 上下限正上 / 正下居中堆叠。
fn layout_big_op(
    font: &Font,
    op: BigOpKind,
    sub: Option<&MathExpr>,
    sup: Option<&MathExpr>,
    size: f32,
    params: &MathParams,
    depth: usize,
) -> MBox {
    let op_size: f32 = size * params.big_op_scale;
    let mut op_box: MBox = leaf(font, op.symbol(), op_size, GlyphRole::BigOp, params);
    let limit_size: f32 = size * params.script_scale;
    let gap: f32 = params.limit_gap * op_size;

    let mut sup_box: Option<MBox> =
        sup.map(|e| layout_expr(font, e, limit_size, params, depth + 1));
    let mut sub_box: Option<MBox> =
        sub.map(|e| layout_expr(font, e, limit_size, params, depth + 1));
    if let Some(b) = sup_box.as_mut() {
        b.set_role(GlyphRole::UpperLimit);
    }
    if let Some(b) = sub_box.as_mut() {
        b.set_role(GlyphRole::LowerLimit);
    }

    // 内容宽 = 算子 / 上限 / 下限三者最宽（算子自身在槽内居中）。
    let mut inner: f32 = op_box.width;
    if let Some(b) = &sup_box {
        inner = inner.max(b.width);
    }
    if let Some(b) = &sub_box {
        inner = inner.max(b.width);
    }

    let op_dx: f32 = (inner - op_box.width) * 0.5;
    op_box.shift(op_dx, 0.0);
    let op_ascent: f32 = op_box.ascent;
    let op_descent: f32 = op_box.descent;

    let mut out: MBox = MBox::default();
    let mut asc: f32 = op_ascent;
    let mut desc: f32 = op_descent;
    if let Some(b) = sup_box.as_mut() {
        let dy: f32 = op_ascent + gap;
        let dx: f32 = (inner - b.width) * 0.5;
        b.shift(dx, dy);
        asc = asc.max(dy + b.ascent);
        out.glyphs.extend(b.glyphs.iter().cloned());
        out.rules.extend(b.rules.iter().cloned());
    }
    out.glyphs.extend(op_box.glyphs);
    out.rules.extend(op_box.rules);
    if let Some(b) = sub_box.as_mut() {
        let dy: f32 = -(op_descent + gap + b.ascent);
        let dx: f32 = (inner - b.width) * 0.5;
        b.shift(dx, dy);
        desc = desc.max(-dy + b.descent);
        out.glyphs.extend(b.glyphs.iter().cloned());
        out.rules.extend(b.rules.iter().cloned());
    }
    out.width = inner;
    out.ascent = asc;
    out.descent = desc;
    out
}

/// 矩阵排版：列宽取列内最大、同列单元居中、行基线累加。
fn layout_matrix(
    font: &Font,
    m: &crate::math_ast::MatrixExpr,
    size: f32,
    params: &MathParams,
    depth: usize,
) -> MBox {
    let cell_size: f32 = size * 0.95;
    let empty: MathExpr = MathExpr::Row(Vec::new());
    let mut cells: Vec<Vec<MBox>> = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let mut row: Vec<MBox> = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            let e: &MathExpr = m.cell(r, c).unwrap_or(&empty);
            let mut b: MBox = layout_expr(font, e, cell_size, params, depth + 1);
            b.set_role(GlyphRole::Matrix);
            row.push(b);
        }
        cells.push(row);
    }

    let col_gap: f32 = params.matrix_col_gap * size;
    let row_gap: f32 = params.matrix_row_gap * size;

    // 列宽 / 行上下伸。
    let mut col_w: Vec<f32> = vec![0.0; m.cols];
    let mut row_asc: Vec<f32> = vec![0.0; m.rows];
    let mut row_desc: Vec<f32> = vec![0.0; m.rows];
    for r in 0..m.rows {
        for c in 0..m.cols {
            let b: &MBox = &cells[r][c];
            col_w[c] = col_w[c].max(b.width);
            row_asc[r] = row_asc[r].max(b.ascent);
            row_desc[r] = row_desc[r].max(b.descent);
        }
    }
    // 列起点 x。
    let mut col_x: Vec<f32> = vec![0.0; m.cols];
    let mut acc: f32 = 0.0;
    for c in 0..m.cols {
        col_x[c] = acc;
        acc += col_w[c] + col_gap;
    }
    let width: f32 = (acc - col_gap).max(0.0);
    // 行基线 y（第 0 行最高）。
    let mut row_y: Vec<f32> = vec![0.0; m.rows];
    row_y[0] = row_asc[0];
    for r in 1..m.rows {
        row_y[r] = row_y[r - 1] - row_desc[r - 1] - row_gap - row_asc[r];
    }

    let mut out: MBox = MBox::default();
    for r in 0..m.rows {
        for c in 0..m.cols {
            let mut b: MBox = cells[r][c].clone();
            let dx: f32 = col_x[c] + (col_w[c] - b.width) * 0.5; // 列内居中
            b.shift(dx, row_y[r]);
            out.glyphs.extend(b.glyphs);
            out.rules.extend(b.rules);
        }
    }
    out.width = width;
    out.ascent = row_asc[0];
    out.descent = if m.rows == 0 { 0.0 } else { -row_y[m.rows - 1] + row_desc[m.rows - 1] };
    out
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;
    use crate::math_ast::MathExpr;

    /// 记号 → 布局（缺省字号 32）。
    fn lay(src: &str) -> MathLayout {
        let e: MathExpr = MathExpr::parse(src).unwrap_or_else(|err| panic!("记法 '{src}' 解析失败：{err}"));
        let f = font_regular();
        layout_math_sized(&f, &e, 32.0)
    }

    /// 结构 1：上标在基线之上且缩到脚本字号；下标在基线之下。
    #[test]
    fn sup_sub_geometry() {
        let l: MathLayout = lay("x^2_1");
        let base: &MathGlyph = l.glyph('x').expect("基座 x");
        let sup: &MathGlyph = l.glyph('2').expect("上标 2");
        let sub: &MathGlyph = l.glyph('1').expect("下标 1");
        assert!(sup.y > base.y + 1.0, "上标须抬到基线之上：sup.y={}", sup.y);
        assert!(sub.y < base.y - 1.0, "下标须压到基线之下：sub.y={}", sub.y);
        assert!(sup.size < base.size * 0.8, "上标必须缩小：{} vs {}", sup.size, base.size);
        assert!(sup.x >= base.x + base.advance - 0.01, "上标须在基座右缘之后");
        assert!(sub.x >= base.x + base.advance - 0.01, "下标须在基座右缘之后");
        // 盒尺寸须覆盖上下标墨迹。
        assert!(l.ascent >= sup.y + sup.size * 0.6, "ascent 须覆盖上标");
        assert!(l.descent > 0.0, "下标撑出下降部");
    }

    /// 结构 2：分子在分数线上方、分母在线下方；分数线居中且覆盖分式宽。
    #[test]
    fn frac_geometry() {
        let l: MathLayout = lay("\\frac{1}{2}");
        let bar: &MathRule = l
            .rules
            .iter()
            .find(|r| r.kind == RuleKind::FracBar)
            .expect("分式必须有分数线");
        let num: &MathGlyph = l.glyph('1').expect("分子 1");
        let den: &MathGlyph = l.glyph('2').expect("分母 2");
        assert_eq!(num.role, GlyphRole::Numerator);
        assert_eq!(den.role, GlyphRole::Denominator);
        // 分子基线 > 分数线上缘 > 分数线下缘 > 分母基线。
        assert!(num.y > bar.top(), "分子须在横线上方：num.y={} bar.top={}", num.y, bar.top());
        assert!(den.y + den.size * 0.7 < bar.bottom(), "分母须在横线下方");
        assert!((bar.x0 - 0.0).abs() < 1e-4, "分数线自左缘起");
        assert!(bar.span() > 0.0 && (bar.x1 - l.width).abs() < 1e-4, "分数线覆盖整个分式宽");
        // 分子 / 分母在分数线宽度内居中：中心差 ≤ 半个字宽。
        let num_c: f32 = num.x + num.advance * 0.5;
        let den_c: f32 = den.x + den.advance * 0.5;
        assert!((num_c - l.width * 0.5).abs() < num.advance, "分子须居中：{num_c} / {}", l.width * 0.5);
        assert!((den_c - l.width * 0.5).abs() < den.advance, "分母须居中");
        // 盒尺寸：ascent 覆盖分子，descent 覆盖分母。
        assert!(l.ascent > num.y, "ascent 覆盖分子");
        assert!(l.descent > den.size * 0.7, "descent 覆盖分母");
    }

    /// 结构 3：根号横线覆盖被开方体且在被开方体上方；√ 在左缘。
    #[test]
    fn sqrt_geometry() {
        let l: MathLayout = lay("\\sqrt{x+1}");
        let rule: &MathRule = l
            .rules
            .iter()
            .find(|r| r.kind == RuleKind::SqrtOverline)
            .expect("根号必须有横线");
        let rad: &MathGlyph = l.glyph('√').expect("根号字形");
        assert_eq!(rad.role, GlyphRole::Radical);
        assert!(rad.x <= 1e-4, "√ 在最左缘");
        let body: Vec<&MathGlyph> = l.glyphs_with_role(GlyphRole::Radicand).collect();
        assert!(!body.is_empty(), "必须有被开方体字形");
        for g in &body {
            assert!(g.x >= rule.x0 - 1e-4, "被开方体须在横线覆盖范围内（左端）");
            assert!(g.x + g.advance <= rule.x1 + 0.5, "被开方体须在横线覆盖范围内（右端）");
            assert!(rule.top() > g.y, "横线须在被开方体之上：rule.top={} g.y={}", rule.top(), g.y);
            assert!((g.y - 0.0).abs() < 1e-4, "被开方体应在基线上");
        }
        assert!(l.ascent >= rule.top() - 1e-4, "ascent 覆盖横线：{} vs {}", l.ascent, rule.top());
    }

    /// 结构 3b：带根指数 —— 指数缩到左下、位于横线之上。
    #[test]
    fn sqrt_index_geometry() {
        let l: MathLayout = lay("\\sqrt[3]{x}");
        let idx: &MathGlyph = l.glyph_with_role(GlyphRole::Index).expect("根指数字形");
        let rule: &MathRule = l
            .rules
            .iter()
            .find(|r| r.kind == RuleKind::SqrtOverline)
            .expect("横线");
        assert!(idx.y >= rule.top() - 1e-4, "根指数须在横线之上");
        assert!(idx.size < 32.0 * 0.6, "根指数须缩小：{}", idx.size);
        assert!(idx.x < 8.0, "根指数贴左缘");
    }

    /// 结构 4：大型算子上下限在算子正上 / 正下居中。
    #[test]
    fn big_op_limits_geometry() {
        let l: MathLayout = lay("\\sum_{i=1}^{n}");
        let op: &MathGlyph = l.glyph_with_role(GlyphRole::BigOp).expect("∑ 字形");
        let up: &MathGlyph = l.glyph_with_role(GlyphRole::UpperLimit).expect("上限 n");
        let lo: &MathGlyph = l.glyph_with_role(GlyphRole::LowerLimit).expect("下限 i");
        assert!(op.size > 32.0, "算子放大：{}", op.size);
        // 上限在算子上方、下限在算子下方（算子墨迹之外）。
        assert!(up.y > op.y, "上限须在算子上方");
        assert!(lo.y < op.y, "下限须在算子下方");
        assert!(up.y - up.size * 0.2 > op.y + op.size * 0.6, "上限须在算子墨迹上方");
        assert!(lo.y + lo.size * 0.7 < op.y - op.size * 0.2, "下限须在算子墨迹下方");
        // 居中：上限 / 下限**盒**的水平中心 ≈ 算子的水平中心。
        let op_c: f32 = op.x + op.advance * 0.5;
        let box_c = |role: GlyphRole| -> f32 {
            let mut lo: f32 = f32::INFINITY;
            let mut hi: f32 = f32::NEG_INFINITY;
            for g in l.glyphs_with_role(role) {
                lo = lo.min(g.x);
                hi = hi.max(g.x + g.advance);
            }
            (lo + hi) * 0.5
        };
        let up_c: f32 = box_c(GlyphRole::UpperLimit);
        let lo_c: f32 = box_c(GlyphRole::LowerLimit);
        assert!((up_c - op_c).abs() < 1e-3, "上限盒须水平居中于算子：{up_c} vs {op_c}");
        assert!((lo_c - op_c).abs() < 1e-3, "下限盒须水平居中于算子：{lo_c} vs {op_c}");
        assert!(l.ascent > up.y && l.descent > 0.0);
    }

    /// 结构 5：矩阵 —— 同列单元 x 中心对齐，行自上而下排列。
    #[test]
    fn matrix_geometry() {
        let l: MathLayout = lay("\\matrix{a & b \\\\ c & d}");
        let g: Vec<&MathGlyph> = l.glyphs_with_role(GlyphRole::Matrix).collect();
        assert_eq!(g.len(), 4, "2×2 矩阵应有 4 个字形");
        let find = |ch: char| -> &MathGlyph { l.glyph(ch).unwrap_or_else(|| panic!("缺少 {ch}")) };
        let (a, b, c, d) = (find('a'), find('b'), find('c'), find('d'));
        // 列对齐：同列（a/c、b/d）的 x 中心一致。
        assert!(
            ((a.x + a.advance * 0.5) - (c.x + c.advance * 0.5)).abs() < 1e-3,
            "第 0 列单元须同 x 中心：a={} c={}",
            a.x,
            c.x
        );
        assert!(
            ((b.x + b.advance * 0.5) - (d.x + d.advance * 0.5)).abs() < 1e-3,
            "第 1 列单元须同 x 中心"
        );
        // 行对齐：同一行 y 相同；第 0 行在第 1 行之上。
        assert!((a.y - b.y).abs() < 1e-3, "同行 y 必须一致");
        assert!((c.y - d.y).abs() < 1e-3, "同行 y 必须一致");
        assert!(a.y > c.y, "第 0 行须在第 1 行上方：{} vs {}", a.y, c.y);
        // 第 1 列在第 0 列右侧。
        assert!(b.x > a.x, "第 1 列须在第 0 列右侧");
        // 4×4 矩阵列严格递增（列对齐的更强证据）。
        let wide: MathLayout = lay("\\matrix{1 & 2 & 3 \\\\ 4 & 5 & 6 \\\\ 7 & 8 & 9}");
        let mut cols: Vec<f32> = wide
            .glyphs_with_role(GlyphRole::Matrix)
            .take(3)
            .map(|g| g.x + g.advance * 0.5)
            .collect();
        let n = cols.len();
        assert_eq!(n, 3);
        let sorted: bool = cols.windows(2).all(|w| w[1] > w[0]);
        assert!(sorted, "列中心须单调右移：{cols:?}");
        cols.clear();
    }

    /// 五类结构全覆盖 → 字形非空（渲染出字的直接证据）。
    #[test]
    fn all_structures_produce_glyphs() {
        let cases: [&str; 5] =
            ["x^2_1", "\\frac{dy}{dx}", "\\sqrt[3]{x}", "\\int_{0}^{1} f", "\\matrix{a & b \\\\ c & d}"];
        for src in cases {
            let l: MathLayout = lay(src);
            assert!(!l.glyphs.is_empty(), "'{src}' 必须排版出字形");
            assert!(l.width > 0.0, "'{src}' 宽度须为正");
            assert!(l.ascent > 0.0 && l.descent >= 0.0, "'{src}' 盒尺寸须非负");
        }
    }

    /// 非法参数 → 空布局（不 panic）。
    #[test]
    fn invalid_params_degrade() {
        let f = font_regular();
        let e: MathExpr = MathExpr::parse("x").expect("x");
        for size in [0.0f32, -4.0, f32::NAN, f32::INFINITY] {
            let l: MathLayout = layout_math(&f, &e, &MathParams { font_size: size, ..MathParams::default() });
            assert!(l.is_empty(), "非法字号 {size} 须退化为空布局");
        }
    }

    /// 深嵌套不 panic（防御性上限）。
    #[test]
    fn deep_nesting_does_not_panic() {
        let mut e: MathExpr = MathExpr::num("1");
        for _ in 0..40 {
            e = MathExpr::frac(e, MathExpr::num("2"));
        }
        let f = font_regular();
        let l: MathLayout = layout_math_sized(&f, &e, 20.0);
        // 只要不 panic 且输出有限即通过。
        assert!(l.width.is_finite() && l.ascent.is_finite() && l.descent.is_finite());
    }
}
