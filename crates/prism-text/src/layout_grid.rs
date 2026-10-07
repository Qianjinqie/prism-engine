//! 版式网格（版式系统）。
//!
//! # 定位
//!
//! 网格把画布切分为**规则栅格**，并给出栏（column）/ 行（row）/ 边距（margin /
//! gutter）的对齐定位。与既有 [`crate::layout`]（流式排版）正交：网格负责
//! "区域从哪里开始、有多宽"，`layout_text` 负责区域内文本流式落位。
//!
//! # 模型（D-21：解析定位）
//!
//! 设画布 `canvas_w × canvas_h`，外边距 `margin_{t,r,b,l}`，栏间/行间间隙
//! `gutter_x/gutter_y`，栏数 `cols`、行数 `rows`。**版心**为
//! `content_w = canvas_w − margin_l − margin_r`；**单栏宽**由剩余空间均分：
//!
//! ```text
//!   cell_w = (content_w − gutter_x·(cols − 1)) / cols
//!   x(i)   = margin_l + i·(cell_w + gutter_x)         （i 为 0 起栏号）
//! ```
//!
//! 行向同理。全部为解析式，无迭代；坐标单位 px，原点左上、y 向下（与
//! `prism-text` 设备坐标一致）。
//!
//! # 确定性契约
//! 纯函数；浮点只用 `+ − × ÷`。

use crate::layout::{layout_text, Align, TextParams};
use crate::ttf::Font;

/// 边距（px；各向独立）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Margin {
    /// 上边距。
    pub top: f32,
    /// 右边距。
    pub right: f32,
    /// 下边距。
    pub bottom: f32,
    /// 左边距。
    pub left: f32,
}

impl Margin {
    /// 四向等值边距。
    pub fn all(v: f32) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }

    /// 水平合计（左 + 右）。
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }

    /// 垂直合计（上 + 下）。
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// 单元格矩形（px；原点左上，y 向下）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    /// 左缘 x。
    pub x: f32,
    /// 上缘 y。
    pub y: f32,
    /// 宽。
    pub w: f32,
    /// 高。
    pub h: f32,
}

impl Cell {
    /// 水平中心 x（对齐用）。
    pub fn center_x(&self) -> f32 {
        self.x + self.w * 0.5
    }

    /// 垂直中心 y。
    pub fn center_y(&self) -> f32 {
        self.y + self.h * 0.5
    }

    /// 内缩边距后的子单元格（用于嵌套网格/内边距）。
    pub fn inset(&self, m: Margin) -> Cell {
        Cell {
            x: self.x + m.left,
            y: self.y + m.top,
            w: (self.w - m.horizontal()).max(0.0),
            h: (self.h - m.vertical()).max(0.0),
        }
    }
}

/// 版式网格：栅格切分 + 对齐定位。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// 画布宽（px）。
    pub canvas_w: f32,
    /// 画布高（px）。
    pub canvas_h: f32,
    /// 栏数（≥1）。
    pub cols: u32,
    /// 行数（≥1）。
    pub rows: u32,
    /// 外边距。
    pub margin: Margin,
    /// 栏间隙（px）。
    pub gutter_x: f32,
    /// 行间隙（px）。
    pub gutter_y: f32,
}

impl Grid {
    /// 构造网格（栏/行数下限 1——0 视为 1，防除零，确定性钳制）。
    pub fn new(canvas_w: f32, canvas_h: f32, cols: u32, rows: u32, margin: Margin) -> Self {
        Self {
            canvas_w,
            canvas_h,
            cols: cols.max(1),
            rows: rows.max(1),
            margin,
            gutter_x: 0.0,
            gutter_y: 0.0,
        }
    }

    /// 设置栏/行间隙（builder 风格）。
    pub fn with_gutter(mut self, gx: f32, gy: f32) -> Self {
        self.gutter_x = gx;
        self.gutter_y = gy;
        self
    }

    /// 版心宽 = 画布宽 − 左右边距。
    pub fn content_w(&self) -> f32 {
        self.canvas_w - self.margin.horizontal()
    }

    /// 版心高 = 画布高 − 上下边距。
    pub fn content_h(&self) -> f32 {
        self.canvas_h - self.margin.vertical()
    }

    /// 单栏宽 `(content_w − gutter_x·(cols−1)) / cols`（下限 0）。
    pub fn cell_w(&self) -> f32 {
        let w = self.content_w() - self.gutter_x * (self.cols as f32 - 1.0);
        if w <= 0.0 {
            0.0
        } else {
            w / self.cols as f32
        }
    }

    /// 单行高 `(content_h − gutter_y·(rows−1)) / rows`（下限 0）。
    pub fn cell_h(&self) -> f32 {
        let h = self.content_h() - self.gutter_y * (self.rows as f32 - 1.0);
        if h <= 0.0 {
            0.0
        } else {
            h / self.rows as f32
        }
    }

    /// 第 `i` 栏左缘 x（越界钳到合法区间——确定性）。
    pub fn col_x(&self, i: u32) -> f32 {
        let i = i.min(self.cols.saturating_sub(1)) as f32;
        self.margin.left + i * (self.cell_w() + self.gutter_x)
    }

    /// 第 `j` 行上缘 y。
    pub fn row_y(&self, j: u32) -> f32 {
        let j = j.min(self.rows.saturating_sub(1)) as f32;
        self.margin.top + j * (self.cell_h() + self.gutter_y)
    }

    /// 位于 (col, row) 的单元格。越界索引按 [`Self::col_x`]/[`Self::row_y`] 钳制。
    pub fn cell(&self, col: u32, row: u32) -> Cell {
        Cell { x: self.col_x(col), y: self.row_y(row), w: self.cell_w(), h: self.cell_h() }
    }

    /// 跨 `span` 栏的单元格（用于跨栏标题；span 限 1..=cols−col）。
    pub fn cell_span(&self, col: u32, row: u32, span: u32) -> Cell {
        let span = span.max(1).min(self.cols - col.min(self.cols.saturating_sub(1)));
        let w = self.cell_w() * span as f32 + self.gutter_x * (span as f32 - 1.0);
        Cell { x: self.col_x(col), y: self.row_y(row), w: w.max(0.0), h: self.cell_h() }
    }

    /// 对齐锚点：把内容包围盒 `(content_w, content_h)` 放进 `cell` 按 `align`
    /// （水平）与 `valign`（垂直）对位，返回内容左上角 (x, y)。
    ///
    /// `valign` 语义：0=顶、1=中、2=底（整数枚举，见 [`VAlign`]）。
    pub fn align_in(&self, cell: Cell, content_w: f32, content_h: f32, align: Align, valign: VAlign) -> (f32, f32) {
        let x = match align {
            Align::Left => cell.x,
            Align::Center => cell.x + (cell.w - content_w) * 0.5,
            Align::Right => cell.x + (cell.w - content_w),
        };
        let y = match valign {
            VAlign::Top => cell.y,
            VAlign::Middle => cell.y + (cell.h - content_h) * 0.5,
            VAlign::Bottom => cell.y + (cell.h - content_h),
        };
        (x, y)
    }

    /// 便利：在单元格内对文本做流式排版（复用 `crate::layout`），返回每行左缘 x
    /// 与首行基线 y 的平移量，使文本在单元格内按 `align`/`valign` 对齐。
    ///
    /// 返回 `(dx, dy)`：调用方把 `layout_text` 结果整体平移 (dx, dy) 即完成
    /// "网格内对齐"。`dy` 使文本块垂直居中/贴顶/贴底（基于文本总高）。
    pub fn place_text(
        &self,
        font: &Font,
        text: &str,
        params: &TextParams,
        cell: Cell,
        align: Align,
        valign: VAlign,
    ) -> (f32, f32) {
        // 单元格内先按栏宽限制换行。
        let mut p = params.clone();
        p.max_width = cell.w;
        p.align = Align::Left; // 先左排得内容盒，再整体对齐平移。
        let l = layout_text(font, text, &p);
        self.align_in(cell, l.width, l.height, align, valign)
    }
}

/// 垂直对齐（网格内内容盒定位）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VAlign {
    /// 顶对齐（默认）。
    #[default]
    Top,
    /// 垂直居中。
    Middle,
    /// 底对齐。
    Bottom,
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    #[test]
    fn cell_geometry_divides_content_box() {
        // 1000 宽，左右各 100 边距 → 版心 800；3 栏无间隙 → 每栏 266.67。
        let g = Grid::new(1000.0, 600.0, 3, 2, Margin { top: 50.0, right: 100.0, bottom: 50.0, left: 100.0 });
        assert!((g.content_w() - 800.0).abs() < 1e-4);
        let cw = g.cell_w();
        assert!((cw - 800.0 / 3.0).abs() < 1e-4);
        // 栏左缘：100, 100+266.67, 100+2×266.67。
        assert!((g.col_x(0) - 100.0).abs() < 1e-4);
        assert!((g.col_x(1) - (100.0 + cw)).abs() < 1e-4);
        assert!((g.col_x(2) - (100.0 + 2.0 * cw)).abs() < 1e-4);
        // 行高：版心 500 / 2 = 250。
        assert!((g.cell_h() - 250.0).abs() < 1e-4);
        assert!((g.row_y(1) - (50.0 + 250.0)).abs() < 1e-4);
    }

    #[test]
    fn gutter_reduces_cell_size() {
        let g = Grid::new(820.0, 600.0, 3, 1, Margin::all(10.0)).with_gutter(10.0, 0.0);
        // 版心 800；2 道间隙 20 → 780/3 = 260。
        assert!((g.cell_w() - 260.0).abs() < 1e-4);
        assert!((g.col_x(1) - (10.0 + 260.0 + 10.0)).abs() < 1e-4);
    }

    #[test]
    fn align_in_places_content_box() {
        let g = Grid::new(100.0, 100.0, 1, 1, Margin::all(0.0));
        let c = g.cell(0, 0);
        let (x, y) = g.align_in(c, 40.0, 20.0, Align::Center, VAlign::Middle);
        assert!((x - 30.0).abs() < 1e-4);
        assert!((y - 40.0).abs() < 1e-4);
        let (x, y) = g.align_in(c, 40.0, 20.0, Align::Right, VAlign::Bottom);
        assert!((x - 60.0).abs() < 1e-4);
        assert!((y - 80.0).abs() < 1e-4);
    }

    #[test]
    fn span_cell_covers_multiple_columns() {
        let g = Grid::new(900.0, 300.0, 3, 1, Margin::all(0.0)).with_gutter(0.0, 0.0);
        // 每栏 300；跨 2 栏 = 600。
        let s = g.cell_span(0, 0, 2);
        assert!((s.w - 600.0).abs() < 1e-4);
        assert!((s.x).abs() < 1e-4);
    }

    #[test]
    fn degenerate_counts_clamped_to_one() {
        let g = Grid::new(100.0, 100.0, 0, 0, Margin::all(0.0));
        assert_eq!(g.cols, 1);
        assert_eq!(g.rows, 1);
        assert!((g.cell_w() - 100.0).abs() < 1e-4);
    }

    #[test]
    fn place_text_centers_block_in_cell() {
        let f = font_regular();
        let g = Grid::new(400.0, 200.0, 1, 1, Margin::all(0.0));
        let c = g.cell(0, 0);
        let p = TextParams { font_size: 20.0, ..TextParams::default() };
        let (dx, dy) = g.place_text(&f, "中文", &p, c, Align::Center, VAlign::Middle);
        // 两字 40px 宽 → 居中 dx = (400−40)/2 = 180。
        assert!((dx - 180.0).abs() < 1e-3, "dx = {dx}");
        // 垂直居中：dy = (200 − 单行高)/2 > 0。
        assert!(dy > 0.0 && dy < 200.0, "dy = {dy}");
    }

    #[test]
    fn inset_shrinks_cell() {
        let g = Grid::new(100.0, 100.0, 1, 1, Margin::all(0.0));
        let inner = g.cell(0, 0).inset(Margin { top: 5.0, right: 10.0, bottom: 5.0, left: 10.0 });
        assert!((inner.x - 10.0).abs() < 1e-4);
        assert!((inner.w - 80.0).abs() < 1e-4);
        assert!((inner.h - 90.0).abs() < 1e-4);
    }
}
