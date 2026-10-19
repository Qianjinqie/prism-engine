//! 水平文本布局：度量、贪心断行、中文行首行末禁则、对齐。
//!
//! 排版算法（文字管线中段）：
//! - **度量**：`advance = hmtx / upem × font_size + letter_spacing`（f32 确定性）；
//! - **断行**：贪心行填充。断点确定后依次做三步修正——ASCII 词回退
//!   （词内不断）、行末禁则回退（开括号/开引号不入行末）、行首禁则悬挂
//!   （句读点并入上一行末；悬挂允许行宽超界——禁则优先于宽度）；
//! - **对齐**：Left / Center / Right（参考宽 = max_width；非有限时退化为
//!   内容最大行宽）。
//!
//! # 基线互操作（V2-j）
//! 公式（[`crate::math_layout`]）与普通文本在**同一行混排**时共用一条基线：
//! [`baseline_metrics`] 是唯一的基线度量入口（字号 → 上伸 / 下伸 px），
//! [`place_formula_on_baseline`] 把公式字形（数学 y 向上）翻转到设备坐标
//! （y 向下）并贴到给定基线，[`layout_mixed_line`] 则是「文本 + 公式」一行
//! 混排的成品入口（文本先按 [`layout_text`] 定位，公式紧随其后并共用第 0
//! 行基线）。
//!
//! # 确定性契约
//! 布局是 (字体字节, 文本, 参数) 的纯函数：无隐藏状态、无随机源、
//! 浮点仅 ±×÷。同输入必得同 `Layout`（golden-file 测试锁定）。
//!
//! 注：GPOS kern 表不解析（子集未保留；字距用 letter_spacing 追踪），
//! 为已知实现边界，见 crate 根 README 的遗留问题清单。

use crate::math_layout::MathLayout;
use crate::math_symbols::{advance_for_gid, fallback_gid, MATH_GID_BASE};
use crate::ttf::Font;

/// 水平对齐方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    /// 左对齐（默认）。
    #[default]
    Left,
    /// 居中。
    Center,
    /// 右对齐。
    Right,
}

/// 文本排版参数。
#[derive(Debug, Clone, PartialEq)]
pub struct TextParams {
    /// 字号（px，em 框高）。
    pub font_size: f32,
    /// 字间追踪（px，逐字符追加）。
    pub letter_spacing: f32,
    /// 额外行距（px，叠加在字体行高之上）。
    pub line_spacing: f32,
    /// 换行宽度（px；非有限 = 不换行）。
    pub max_width: f32,
    /// 对齐方式。
    pub align: Align,
}

impl Default for TextParams {
    fn default() -> Self {
        TextParams {
            font_size: 24.0,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Left,
        }
    }
}

/// 已定位字形（笔位坐标，像素；设备坐标 y 向下）。
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphPos {
    /// 字符。
    pub ch: char,
    /// 字形 id（cmap 查不到 → 0 = .notdef）。
    pub gid: u16,
    /// 笔位 x（该字符 em 框左缘）。
    pub x: f32,
    /// 所在行基线 y。
    pub y: f32,
    /// 步进宽（px，含 letter_spacing）。
    pub advance: f32,
    /// 行号（0 起）。
    pub line: usize,
    /// 原文本字符索引（`chars().enumerate()` 序）。
    pub index: usize,
}

/// 行盒。
#[derive(Debug, Clone, PartialEq)]
pub struct LineBox {
    /// 起始字符索引（含）。
    pub start: usize,
    /// 终止字符索引（不含）。
    pub end: usize,
    /// 行宽（px；含行尾悬挂的禁则字符——禁则优先于宽度）。
    pub width: f32,
    /// 基线 y（px）。
    pub baseline: f32,
}

/// 排版结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layout {
    /// 全部已定位字形（按阅读序）。
    pub glyphs: Vec<GlyphPos>,
    /// 行盒列表。
    pub lines: Vec<LineBox>,
    /// 最大行宽（px）。
    pub width: f32,
    /// 总高 = 行数 × 行高（px）。
    pub height: f32,
}

/// 行高（px）= (ascender − descender + line_gap)/upem × font_size + line_spacing。
pub fn line_height(font: &Font, params: &TextParams) -> f32 {
    let scale = params.font_size / font.units_per_em() as f32;
    let lh_units = (font.ascender() - font.descender() + font.line_gap()) as f32;
    lh_units * scale + params.line_spacing
}

/// 基线度量（px；`ascent` = 基线以上、`descent` = 基线以下，均非负）。
///
/// 文字与公式共用的**唯一**基线度量入口（V2-j）：公式盒的高度口径与此
/// 逐位一致，因此同一行混排时基线不会错位。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineMetrics {
    /// 基线以上高度（px，正数）。
    pub ascent: f32,
    /// 基线以下深度（px，正数）。
    pub descent: f32,
}

/// 字号 → 基线度量（`ascender/upem × size`、`−descender/upem × size`）。
///
/// 非有限 / 非正字号 → 全零度量（调用方退化为无行盒，不 panic）。
pub fn baseline_metrics(font: &Font, font_size: f32) -> BaselineMetrics {
    if !font_size.is_finite() || font_size <= 0.0 {
        return BaselineMetrics { ascent: 0.0, descent: 0.0 };
    }
    let scale = font_size / font.units_per_em() as f32;
    BaselineMetrics {
        ascent: font.ascender() as f32 * scale,
        descent: -(font.descender() as f32) * scale,
    }
}

/// 把公式字形贴到文本基线（设备坐标，y 向下）：公式基座字形与文本基线重合。
///
/// - `formula`：公式布局（数学坐标，原点 = 基线左缘，y 向上）；
/// - `dx`：公式左缘的设备 x；
/// - `baseline_y`：公式基线的设备 y（与 [`GlyphPos::y`] 同口径）；
/// - `index_base`：输出字形的原文本索引起点（接在文本字符之后）。
///
/// 输出 `layout = 0` 的字形序列，供 [`Layout`] 直接拼接（同一行混排）。
pub fn place_formula_on_baseline(
    formula: &MathLayout,
    dx: f32,
    baseline_y: f32,
    index_base: usize,
    line: usize,
) -> Vec<GlyphPos> {
    formula
        .place_on_baseline(dx, baseline_y)
        .into_iter()
        .enumerate()
        .map(|(i, g)| GlyphPos {
            ch: g.ch,
            gid: g.gid,
            x: g.x,
            y: g.y,
            advance: g.advance,
            line,
            index: index_base + i,
        })
        .collect()
}

/// 文本 + 公式**同行混排**（基线对齐）。
///
/// 文本按 [`layout_text`] 排成单列行（`max_width` 建议有限），公式整体排在
/// **第 0 行基线**、文本内容右缘之后 `gap` 处：
/// 文本字形 y = 行基线，公式基座字形 y = 同一行基线（数学 y 向上翻转），
/// 因此两者基线逐位相同（`layout.text_baseline()` 可查）。
///
/// 行盒高度取「文本行高」与「公式盒高」的较大者（[`MathLayout::line_extent`]
/// 复用 [`baseline_metrics`]），行宽 = 文本行宽 + gap + 公式宽。
/// 空文本 / 空公式退化为另一侧的纯结果。
pub fn layout_mixed_line(
    font: &Font,
    text: &str,
    params: &TextParams,
    formula: &MathLayout,
    gap: f32,
) -> Layout {
    let mut text_layout: Layout = layout_text(font, text, params);
    let text_width: f32 = text_layout.width;
    let metrics: BaselineMetrics = baseline_metrics(font, params.font_size);
    // 基线：有文本行 → 第 0 行基线；纯公式 → 字体上伸（与文本单行基线同口径）。
    let baseline_y: f32 = text_layout.lines.first().map_or(metrics.ascent, |l| l.baseline);
    let safe_gap: f32 = if gap.is_finite() && gap > 0.0 { gap } else { 0.0 };
    let dx: f32 = text_width + safe_gap;
    let index_base: usize = text.chars().count();
    let has_formula: bool = !formula.is_empty();
    if has_formula {
        let glyphs: Vec<GlyphPos> = place_formula_on_baseline(formula, dx, baseline_y, index_base, 0);
        text_layout.glyphs.extend(glyphs);
    }
    let (ext_ascent, ext_descent): (f32, f32) = formula.line_extent(font, params.font_size);
    let ascent: f32 = metrics.ascent.max(ext_ascent);
    let descent: f32 = metrics.descent.max(ext_descent);
    let line_height_px: f32 = ascent + descent;
    let total_width: f32 = text_width + if has_formula { safe_gap + formula.width } else { 0.0 };
    if let Some(line) = text_layout.lines.first_mut() {
        line.width = total_width;
    } else if has_formula {
        // 纯公式：补一条行盒（基线口径与文本单行一致，行高按混排盒高）。
        text_layout.lines.push(LineBox {
            start: 0,
            end: index_base,
            width: total_width,
            baseline: baseline_y,
        });
    }
    text_layout.width = total_width;
    text_layout.height = if text_layout.lines.is_empty() { 0.0 } else { line_height_px };
    text_layout
}

/// 行首禁则：句读点与闭合符号不得居于行首（GB/T 15834 行首规则）；
/// 布局实现为**悬挂**——该字符并入上一行末尾。
pub fn is_no_line_start(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。' | '、' | '；' | '：' | '？' | '！' | '」' | '』' | '）' | '】' | '〕'
            | '》' | '〉' | '…' | '—' | '·' | '～' | '”' | '’' | ',' | '.' | '!' | '?'
            | ';' | ':' | ')' | ']' | '}' | '>' | '%'
    )
}

/// 行末禁则：开括号与开引号不得居于行末（GB/T 15834 行末规则）；
/// 布局实现为**回退**——该字符连同其后内容移入下一行。
pub fn is_no_line_end(ch: char) -> bool {
    matches!(
        ch,
        '「' | '『' | '（' | '【' | '〔' | '《' | '〈' | '“' | '‘' | '(' | '[' | '{' | '<'
    )
}

/// ASCII 词字符（词内不断行）。
fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// 排版文本（水平）。
///
/// 两遍式：先贪心断行得行区间（含禁则修正），再统一对齐定位。
/// 空文本产出空 `Layout`（零行零字形）。
pub fn layout_text(font: &Font, text: &str, params: &TextParams) -> Layout {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let scale = params.font_size / font.units_per_em() as f32;
    // gid 解析：字体 cmap 优先；查不到（希腊/数学符号区段缺失）时回退到
    // `math_symbols` 程序化符号表（v3-A：修复「ε/δ 渲染成 .notdef 豆腐块」）。
    // 已覆盖字符行为逐位不变（真实字形永不走合成 gid）。
    let gids: Vec<u16> = chars
        .iter()
        .map(|&c| font.glyph_index(c).or_else(|| fallback_gid(c)).unwrap_or(0))
        .collect();
    // advance：合成 gid 走回退表步进宽（hmtx 对越界 id 只会钳到末项，
    // 给出错值）；真实字形走 hmtx（逐位不变）。
    let advances: Vec<f32> = gids
        .iter()
        .map(|&g| {
            let adv_units = if g >= MATH_GID_BASE {
                advance_for_gid(g).unwrap_or(0)
            } else {
                font.advance(g)
            };
            adv_units as f32 * scale + params.letter_spacing
        })
        .collect();
    // 前缀和：prefix[i] = Σ advances[0..i]（行宽与笔位均由差值给出，确定性）。
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f32);
    for &a in &advances {
        let last = *prefix.last().unwrap_or(&0.0);
        prefix.push(last + a);
    }

    // ---- 第一遍：贪心断行 → 行区间（含禁则修正）----
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut line_start = 0usize;
    let mut x = 0.0f32; // 当前行 [line_start, i) 的累计宽
    let mut i = 0usize;
    while i < n {
        let cw = advances[i];
        let overflow = params.max_width.is_finite() && x + cw > params.max_width;
        if overflow && i > line_start {
            // 断行：字符 i 起归新行（brk 为新行首字符索引，先设为 i）。
            let mut brk = i;
            // ASCII 词内不断：整词回退（词首下移）。
            if is_word_char(chars[brk]) {
                while brk > line_start + 1 && is_word_char(chars[brk - 1]) {
                    brk -= 1;
                }
            }
            // 行末禁则：开括号/开引号不入行末，连同回退。
            while brk > line_start + 1 && is_no_line_end(chars[brk - 1]) {
                brk -= 1;
            }
            // 行首禁则：句读点悬挂并入上一行（brk 前推）。
            while brk < n && is_no_line_start(chars[brk]) {
                brk += 1;
            }
            // 新行首 ASCII 空格剥除。
            while brk < n && chars[brk] == ' ' {
                brk += 1;
            }
            if brk <= line_start {
                brk = line_start + 1; // 单字符超界硬断，保证进展（防死循环）。
            }
            ranges.push((line_start, brk));
            line_start = brk;
            if brk > i {
                // 字符 i 本身被悬挂吞入上一行：跳过其入新行。
                x = 0.0;
                i += 1;
                continue;
            }
            // 字符 i 归新行：重算其在行内累计宽。
            x = prefix[i] - prefix[brk];
            continue;
        }
        x += cw;
        i += 1;
    }
    if line_start < n {
        ranges.push((line_start, n));
    }

    // ---- 第二遍：对齐与定位 ----
    let lh = line_height(font, params);
    let asc = font.ascender() as f32 * scale;
    let content_w =
        ranges.iter().map(|&(s, e)| prefix[e] - prefix[s]).fold(0.0f32, f32::max);
    let ref_w = if params.max_width.is_finite() { params.max_width } else { content_w };
    let mut out = Layout::default();
    for (li, &(s, e)) in ranges.iter().enumerate() {
        if s >= e {
            continue;
        }
        let lw = prefix[e] - prefix[s];
        // 右/居中对齐取非负平移；悬挂超宽行在 Right 下贴右缘（左端溢出，
        // 语义诚实：禁则优先于对齐）。
        let dx = match params.align {
            Align::Left => 0.0,
            Align::Center => ((ref_w - lw) * 0.5).max(0.0),
            Align::Right => (ref_w - lw).max(0.0),
        };
        let baseline = li as f32 * lh + asc;
        for k in s..e {
            out.glyphs.push(GlyphPos {
                ch: chars[k],
                gid: gids[k],
                x: dx + (prefix[k] - prefix[s]),
                y: baseline,
                advance: advances[k],
                line: li,
                index: k,
            });
        }
        out.lines.push(LineBox { start: s, end: e, width: lw, baseline });
    }
    out.width = content_w;
    out.height = out.lines.len() as f32 * lh;
    out
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    /// Noto Sans SC 全角字符 advance = 1000 upem → font_size 24 时每字 24px。
    fn params(max_width: f32) -> TextParams {
        TextParams { font_size: 24.0, max_width, ..TextParams::default() }
    }

    #[test]
    fn single_line_metrics_advance() {
        let f = font_regular();
        let l = layout_text(&f, "中文", &params(f32::INFINITY));
        assert_eq!(l.lines.len(), 1);
        assert_eq!(l.glyphs.len(), 2);
        // 全角 advance = 1000/1000 × 24 = 24。
        assert!((l.glyphs[0].advance - 24.0).abs() < 1e-4);
        assert!((l.glyphs[1].x - 24.0).abs() < 1e-4);
        // 基线 = ascender/upem × 24（Noto ascender = 1160 → 27.84）。
        let asc = f.ascender() as f32 * (24.0 / f.units_per_em() as f32);
        assert!((l.glyphs[0].y - asc).abs() < 1e-4);
    }

    #[test]
    fn greedy_wrap_by_width() {
        let f = font_regular();
        // 每字 24px；max 96 = 4 字/行。
        let l = layout_text(&f, "一二三四五六七", &params(96.0));
        assert_eq!(l.lines.len(), 2, "4+3 断行");
        assert_eq!(l.lines[0].end, 4);
        assert_eq!(l.lines[1].start, 4);
    }

    #[test]
    fn ascii_word_never_split() {
        let f = font_regular();
        // 词 "Hello"（窄）+ 全角字：断点应回退到词首，词不被劈开。
        let l = layout_text(&f, "中文Hello世界", &params(120.0));
        for (li, _line) in l.lines.iter().enumerate() {
            // 行首字形 = 该行第一个（glyphs 按阅读序）。
            let first = l.glyphs.iter().find(|g| g.line == li).unwrap();
            if first.ch == 'H' {
                // 词首在行首 → 整词未被拆（本行含全部 5 个词字符）。
                let word: Vec<char> = l
                    .glyphs
                    .iter()
                    .filter(|g| g.line == li)
                    .map(|g| g.ch)
                    .collect();
                let s: String = word.into_iter().collect();
                assert!(s.contains("Hello"), "词必须完整成行，实得「{s}」");
            }
        }
    }

    #[test]
    fn period_never_at_line_start_hangs() {
        let f = font_regular();
        // 5 字满行（120px），句号本应下行首 → 悬挂回上一行末。
        let l = layout_text(&f, "一二三四五。六", &params(120.0));
        assert_eq!(l.lines.len(), 2);
        // 句号必须归属第 0 行（悬挂），第 1 行首为 "六"。
        let period = l.glyphs.iter().find(|g| g.ch == '。').expect("句号在文本中");
        assert_eq!(period.line, 0, "句号悬挂于上一行，不得居行首");
        let line1_first = l.glyphs.iter().find(|g| g.line == 1).unwrap();
        assert_eq!(line1_first.ch, '六');
        // 悬挂行宽超界（5×24 + 24 = 144 > 120）——禁则优先于宽度。
        assert!((l.lines[0].width - 144.0).abs() < 1e-3);
    }

    #[test]
    fn open_bracket_never_at_line_end() {
        let f = font_regular();
        // 「 在断点前一位 → 行末禁则回退，「连同后面字符下移。
        let l = layout_text(&f, "一二三「四", &params(96.0));
        let line0_last = l
            .glyphs
            .iter()
            .filter(|g| g.line == 0)
            .last()
            .expect("第 0 行非空");
        assert_ne!(line0_last.ch, '「', "开括号不得居行末");
        assert_eq!(l.lines[0].end, 3, "「 移入下一行");
    }

    #[test]
    fn align_center_and_right() {
        let f = font_regular();
        let mut p = params(120.0);
        p.align = Align::Center;
        let l = layout_text(&f, "中文", &p);
        // 行宽 48，版心 120 → 居中平移 36。
        assert!((l.glyphs[0].x - 36.0).abs() < 1e-4);
        p.align = Align::Right;
        let l = layout_text(&f, "中文", &p);
        assert!((l.glyphs[0].x - (120.0 - 48.0)).abs() < 1e-4);
    }

    #[test]
    fn empty_text_is_empty_layout() {
        let f = font_regular();
        let l = layout_text(&f, "", &params(120.0));
        assert!(l.glyphs.is_empty() && l.lines.is_empty());
        assert_eq!(l.width, 0.0);
        assert_eq!(l.height, 0.0);
    }

    /// V2-j：公式与普通文本混排时**同一基线**（公式基座字形 y == 文本基线）。
    #[test]
    fn formula_shares_text_baseline() {
        use crate::math_ast::MathExpr;
        use crate::math_layout::layout_math_sized;

        let f = font_regular();
        let expr: MathExpr = MathExpr::parse("\\frac{dy}{dx} = 2x").expect("记法可解析");
        let formula = layout_math_sized(&f, &expr, 24.0);
        let p: TextParams = params(f32::INFINITY);
        let mixed: Layout = layout_mixed_line(&f, "导数 ", &p, &formula, 6.0);

        // 文本基线 = 混排行盒基线（文本字形 y 与之相同）。
        let text_baseline: f32 = mixed.lines[0].baseline;
        let text_glyph: &GlyphPos = mixed.glyphs.iter().find(|g| g.ch == '数').expect("文本字形");
        assert!((text_glyph.y - text_baseline).abs() < 1e-4, "文本字形须在行基线上");

        // 公式左缘 = 文本宽 + gap；公式字形不得越过该线。
        let expected_dx: f32 = layout_text(&f, "导数 ", &p).width + 6.0;
        let formula_glyphs: Vec<&GlyphPos> =
            mixed.glyphs.iter().filter(|g| g.index >= 3).collect();
        assert!(!formula_glyphs.is_empty(), "必须有公式字形（索引接在文本字符之后）");
        for g in &formula_glyphs {
            assert!(g.x >= expected_dx - 1e-3, "公式字形不得越过公式左缘：{} < {expected_dx}", g.x);
        }

        // 基线重合：存在公式基座字形（y == 文本基线）。
        let on_baseline: usize = formula_glyphs
            .iter()
            .filter(|g| (g.y - text_baseline).abs() < 1e-4)
            .count();
        assert!(on_baseline >= 2, "公式基座字形须与文本基线重合（'=' / '2' / 'x'）");

        // 分式结构：分子在基线上方、分母在基线下方（设备 y 向下 → 上=更小）。
        let above: usize = formula_glyphs.iter().filter(|g| g.y < text_baseline - 1.0).count();
        let below: usize = formula_glyphs.iter().filter(|g| g.y > text_baseline + 1.0).count();
        assert!(above >= 2, "分子（dy）须在基线之上，实得 {above} 字形");
        assert!(below >= 2, "分母（dx）须在基线之下，实得 {below} 字形");

        // 行盒：宽含公式，高覆盖公式上下伸。
        assert!(mixed.width > expected_dx, "行宽须包含公式");
        assert!(mixed.height > line_height(&f, &p) - 1e-3, "行高须覆盖公式盒");
    }

    /// V2-j：`baseline_metrics` 与 `GlyphPos::y`（单行基线）同口径。
    #[test]
    fn baseline_metrics_matches_single_line_baseline() {
        let f = font_regular();
        let p: TextParams = params(f32::INFINITY);
        let l: Layout = layout_text(&f, "A", &p);
        let m: BaselineMetrics = baseline_metrics(&f, p.font_size);
        assert!((l.lines[0].baseline - m.ascent).abs() < 1e-4);
        assert!(m.descent > 0.0, "descender 为正深度");
        // 非正字号退化为零度量。
        assert_eq!(baseline_metrics(&f, 0.0), BaselineMetrics { ascent: 0.0, descent: 0.0 });
    }
}
