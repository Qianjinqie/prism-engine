//! 竖排 CJK 排版（任务书 B5：右起列、标点转向、列首禁则）。
//!
//! 规则：
//! - **列序**：第 0 列在最右，向左生长（传统中文/日文竖排方向）；
//!   `VChar.x` 给出第 c 列 em 框左缘 = `−c·col_advance`（负向左延伸，
//!   调用方以任意锚点平移渲染）；
//! - **字向**：CJK 表意字符/假名/注音**直立**；句读点与 ASCII 可见字符
//!   视觉**顺时针 90°** 转向（`rotate_cw`）；
//! - **断列**：贪心列填充；列首禁则——句读点不入列首，悬挂并入上一
//!   列底（与水平行首禁则同一谓词 [`crate::layout::is_no_line_start`]）。
//!
//! # 转向近似说明（诚实工程边界）
//! 标准竖排对句读点（。，、）应做**竖排替换字形**（Unicode FE10/FE12 等
//! vert 变体）；本引擎的 OFL 子集未含该等变形码位，故采用整字顺时针
//! 90° 旋转近似——对括号/引号/破折号类与标准一致，对句读点圈位与
//! 替换字形存在已知的角位差异。该取舍记录于 crate README 遗留问题。
//!
//! # 确定性契约
//! 同输入同输出；浮点仅 ±×÷；无隐藏状态。

use crate::layout::is_no_line_start;
use crate::ttf::Font;

/// 竖排参数。
#[derive(Debug, Clone, PartialEq)]
pub struct VerticalParams {
    /// 字号（px，em 框高）。
    pub font_size: f32,
    /// 字间纵向追踪（px）。
    pub letter_spacing: f32,
    /// 列间距（px，列与列 em 框之间的空隙）。
    pub column_gap: f32,
    /// 列最大高（px；非有限 = 不断列）。
    pub max_height: f32,
}

impl Default for VerticalParams {
    fn default() -> Self {
        VerticalParams {
            font_size: 24.0,
            letter_spacing: 0.0,
            column_gap: 8.0,
            max_height: f32::INFINITY,
        }
    }
}

/// 竖排单字（em 框左上角坐标，px）。
#[derive(Debug, Clone, PartialEq)]
pub struct VChar {
    /// 字符。
    pub ch: char,
    /// 字形 id（查不到 → 0）。
    pub gid: u16,
    /// em 框左缘 x（第 0 列 = 0，向左负延伸）。
    pub x: f32,
    /// em 框顶 y（列顶起向下累计）。
    pub y: f32,
    /// 是否顺时针 90° 转向渲染。
    pub rotated: bool,
    /// 原文本字符索引。
    pub index: usize,
}

/// 列盒。
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnBox {
    /// 起始字符索引（含）。
    pub start: usize,
    /// 终止字符索引（不含）。
    pub end: usize,
    /// 该列 em 框左缘 x。
    pub x: f32,
    /// 该列内容高（px）。
    pub height: f32,
}

/// 竖排结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VerticalLayout {
    /// 全部已定位字符（按阅读序）。
    pub chars: Vec<VChar>,
    /// 列盒列表（阅读序 = 右→左）。
    pub columns: Vec<ColumnBox>,
    /// 总宽（px）= 列数 em 宽 + 列距总和。
    pub width: f32,
    /// 最高列内容高（px）。
    pub height: f32,
}

/// 转向表：`true` = 视觉顺时针 90° 渲染。
///
/// 直立集：CJK 统一表意（含扩展 A）、假名、注音、部首、兼容表意、
/// 全角空格与叠字/圈码（々〆〇）。其余（全角句读点、括号、ASCII 可见
/// 字符等）转向。
pub fn is_rotated_cw(ch: char) -> bool {
    let c = ch as u32;
    let upright = matches!(c,
        0x3000              // 全角空格
        | 0x3005..=0x3007   // 々〆〇
        | 0x3041..=0x30FF   // 平假名/片假名
        | 0x3105..=0x312F   // 注音符号
        | 0x31F0..=0x31FF   // 小型片假名
        | 0x2E80..=0x2EFF   // CJK 部首
        | 0x3400..=0x4DBF   // CJK 扩展 A
        | 0x4E00..=0x9FFF   // CJK 统一表意
        | 0xF900..=0xFAFF   // CJK 兼容表意
    );
    !upright
}

/// 竖排排版。
///
/// 两遍式：先按列高贪心切列（含列首禁则悬挂），再统一定位。
/// 空文本产出空 `VerticalLayout`。
pub fn layout_vertical(font: &Font, text: &str, params: &VerticalParams) -> VerticalLayout {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let scale = params.font_size / font.units_per_em() as f32;
    let _ = scale; // 定位不依赖字体度量（全角 em 框步进），保留供扩展。
    let step = params.font_size + params.letter_spacing;
    let col_adv = params.font_size + params.column_gap;
    // 每列字数上限（至少 1；max_height 不足一字高时仍放一字）。
    let per_col: usize = if params.max_height.is_finite() && params.max_height >= 0.0 {
        ((params.max_height / step) as usize).max(1)
    } else {
        usize::MAX
    };

    // ---- 第一遍：切列（含列首禁则悬挂）----
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut col_start = 0usize;
    let mut count = 0usize;
    let mut k = 0usize;
    while k < n {
        if count == per_col {
            // 断列于 k；列首禁则字符悬挂并入上一列底。
            let mut brk = k;
            while brk < n && is_no_line_start(chars[brk]) {
                brk += 1;
            }
            if brk <= col_start {
                brk = col_start + 1; // 进展保证。
            }
            ranges.push((col_start, brk));
            col_start = brk;
            count = 0;
            if brk > k {
                k = brk; // 悬挂字符跳过。
                continue;
            }
            continue;
        }
        count += 1;
        k += 1;
    }
    if col_start < n {
        ranges.push((col_start, n));
    }

    // ---- 第二遍：定位 ----
    let mut out = VerticalLayout::default();
    for (ci, &(s, e)) in ranges.iter().enumerate() {
        if s >= e {
            continue;
        }
        let x = -(ci as f32) * col_adv;
        let mut y = 0.0f32;
        for k in s..e {
            let ch = chars[k];
            out.chars.push(VChar {
                ch,
                gid: font.glyph_index(ch).unwrap_or(0),
                x,
                y,
                rotated: is_rotated_cw(ch),
                index: k,
            });
            y += step;
        }
        let col_h = y - params.letter_spacing;
        out.columns.push(ColumnBox { start: s, end: e, x, height: col_h });
        out.height = out.height.max(col_h);
    }
    let ncols = out.columns.len();
    if ncols > 0 {
        out.width = (ncols as f32 - 1.0) * col_adv + params.font_size;
    }
    out
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    fn params(max_height: f32) -> VerticalParams {
        VerticalParams { font_size: 24.0, max_height, ..VerticalParams::default() }
    }

    #[test]
    fn upright_han_rotated_punct() {
        let f = font_regular();
        let l = layout_vertical(&f, "中。国", &params(f32::INFINITY));
        assert_eq!(l.chars.len(), 3);
        assert!(!l.chars[0].rotated, "汉字直立");
        assert!(l.chars[1].rotated, "句号转向");
        assert!(!l.chars[2].rotated, "汉字直立");
        // 全角 em 框步进 24。
        assert!((l.chars[1].y - 24.0).abs() < 1e-5);
        assert!((l.chars[2].y - 48.0).abs() < 1e-5);
    }

    #[test]
    fn columns_grow_right_to_left() {
        let f = font_regular();
        // 列高 2 字 → "中中中中" 成 2 列。
        let l = layout_vertical(&f, "中中中中", &params(48.0));
        assert_eq!(l.columns.len(), 2);
        // 第 0 列 x=0，第 1 列 x=−(24+8)=−32（向左）。
        assert!((l.columns[0].x).abs() < 1e-6);
        assert!((l.columns[1].x + 32.0).abs() < 1e-6);
        // 列内 y 顺序向下。
        assert!(l.chars[0].y < l.chars[1].y);
    }

    #[test]
    fn period_never_at_column_head() {
        let f = font_regular();
        // 列高 2 字："中中。中" → 第 2 列本应始于 。 → 悬挂入列 1 底。
        let l = layout_vertical(&f, "中中。中", &params(48.0));
        let cols = &l.columns;
        assert_eq!(cols.len(), 2);
        // 列 1（右起第二列）含句号（悬挂），列 2 首字符为中。
        let c1: Vec<char> = l
            .chars
            .iter()
            .filter(|c| c.x == cols[0].x)
            .map(|c| c.ch)
            .collect();
        let c2: Vec<char> = l
            .chars
            .iter()
            .filter(|c| c.x == cols[1].x)
            .map(|c| c.ch)
            .collect();
        assert!(c1.contains(&'。'), "句号悬挂于上一列底，实得 {c1:?}");
        assert_eq!(c2.first(), Some(&'中'), "新列首不可能是句号");
    }

    #[test]
    fn ascii_is_rotated() {
        assert!(is_rotated_cw('A'));
        assert!(is_rotated_cw('9'));
        assert!(!is_rotated_cw('漢'));
        assert!(!is_rotated_cw('ひ'));
        assert!(!is_rotated_cw('\u{3000}'), "全角空格直立（无形态差异）");
    }

    #[test]
    fn empty_text_is_empty_layout() {
        let f = font_regular();
        let l = layout_vertical(&f, "", &params(48.0));
        assert!(l.chars.is_empty() && l.columns.is_empty());
        assert_eq!(l.width, 0.0);
    }
}
