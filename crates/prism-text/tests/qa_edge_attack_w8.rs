//! W8-B15 文字版式攻击性边界测试（QA 严过关）。只写测试，不改源码。

use prism_raster::geom::Point;
use prism_text::layout_grid::{Grid, Margin, VAlign};
use prism_text::path_text::{place_on_path, Seg};
use prism_text::reveal::{RevealPlan, StrokeReveal};
use prism_text::layout::Align;

// --------------------------- B15 网格版式 ---------------------------

/// B15-G1: cols=0 → 钳到 1（防除零）；cell_w 不除零/不 NaN。
#[test]
fn grid_zero_cols_clamped() {
    let g = Grid::new(100.0, 80.0, 0, 0, Margin::all(10.0));
    assert_eq!(g.cols, 1);
    assert_eq!(g.rows, 1);
    assert!(!g.cell_w().is_nan() && !g.cell_h().is_nan());
}

/// B15-G2: 版心列宽几何自洽：cols 栏 + gutter 恰好铺满 content_w。
#[test]
fn grid_cells_fill_content_width() {
    let g = Grid::new(200.0, 120.0, 4, 3, Margin::all(10.0)).with_gutter(8.0, 6.0);
    // cell_w·cols + gutter·(cols-1) == content_w。
    let sum = g.cell_w() * 4.0 + 8.0 * 3.0;
    assert!((sum - g.content_w()).abs() < 1e-3, "列未铺满版心：{sum} vs {}", g.content_w());
    let sum_h = g.cell_h() * 3.0 + 6.0 * 2.0;
    assert!((sum_h - g.content_h()).abs() < 1e-3, "行未铺满版心：{sum_h} vs {}", g.content_h());
}

/// B15-G3: col_x 越界钳到合法区间（确定性，不越出右缘）。
#[test]
fn grid_out_of_range_index_clamped() {
    let g = Grid::new(100.0, 100.0, 3, 3, Margin::all(0.0));
    let last = g.col_x(2);
    let over = g.col_x(999);
    assert_eq!(last.to_bits(), over.to_bits(), "越界栏应钳到末栏");
    assert!(over + g.cell_w() <= 100.0 + 1e-3, "末栏不应越出画布");
}

/// B15-G4: 负版心（边距 > 画布）→ cell_w/cell_h 下限 0，不产负数 / NaN。
#[test]
fn grid_negative_content_degenerate() {
    let g = Grid::new(50.0, 50.0, 2, 2, Margin::all(40.0)); // content = -30
    assert_eq!(g.cell_w(), 0.0, "负版心应钳 0，实得 {}", g.cell_w());
    assert_eq!(g.cell_h(), 0.0);
    assert!(!g.col_x(1).is_nan());
}

/// B15-G5: cell_span 全宽标题跨越所有栏 = content_w；span 越界被钳。
#[test]
fn grid_cell_span_full_width() {
    let g = Grid::new(200.0, 100.0, 4, 4, Margin::all(10.0)).with_gutter(5.0, 5.0);
    let full = g.cell_span(0, 0, 99); // span 越界 → 钳到全部剩余栏
    assert!((full.w - g.content_w()).abs() < 1e-3, "跨全栏标题宽应=版心，实得 {}", full.w);
}

/// B15-G6: align_in 水平三档：左=cell.x、右=cell.x+(w-content)、中=中点。
#[test]
fn grid_align_in_horizontal() {
    let g = Grid::new(100.0, 100.0, 1, 1, Margin::all(0.0));
    let cell = g.cell(0, 0); // x=0,w=100
    let (xl, _) = g.align_in(cell, 40.0, 10.0, Align::Left, VAlign::Top);
    let (xc, _) = g.align_in(cell, 40.0, 10.0, Align::Center, VAlign::Top);
    let (xr, _) = g.align_in(cell, 40.0, 10.0, Align::Right, VAlign::Top);
    assert!((xl - 0.0).abs() < 1e-4);
    assert!((xc - 30.0).abs() < 1e-4, "居中应 30，实得 {xc}");
    assert!((xr - 60.0).abs() < 1e-4, "右对齐应 60，实得 {xr}");
}

/// B15-G7: Grid 纯函数确定性（同参两遍逐位一致）。
#[test]
fn grid_deterministic() {
    let mk = || Grid::new(333.0, 222.0, 5, 3, Margin::all(13.0)).with_gutter(7.0, 9.0);
    let a = mk();
    let b = mk();
    assert_eq!(a, b);
    assert_eq!(a.cell(2, 1).x.to_bits(), b.cell(2, 1).x.to_bits());
}

/// B15-T1: 空路径（无段）不 panic；total_len=0；字形被钳到 0 处。
#[test]
fn path_text_empty_segments_no_panic() {
    let pt = place_on_path(&[], &[10.0, 10.0], 0.0, 16);
    // 无段：定位必得 NaN 段索引 → 会 panic 吗？契约未定义，此处仅断言不崩。
    // total_len 应为 0（空表 acc=0）。
    assert!(pt.total_len.abs() < 1e-6, "空路径总长应为 0，实得 {}", pt.total_len);
}

/// B15-T2: 零长度路径（p0==p1）弧长表退化 → 不 panic / 不 NaN。
#[test]
fn path_text_zero_length_line_no_nan() {
    let segs = [Seg::Line { p0: Point::new(3.0, 3.0), p1: Point::new(3.0, 3.0) }];
    let pt = place_on_path(&segs, &[5.0, 5.0], 0.0, 8);
    assert_eq!(pt.total_len, 0.0, "零长度路径总长应为 0");
    for g in &pt.glyphs {
        assert!(!g.pos.x.is_nan() && !g.pos.y.is_nan(), "零长度路径产生 NaN 位置");
        assert!(!g.angle.is_nan(), "零长度路径产生 NaN 角");
    }
}

/// B15-T3: 单点路径（Line 起终同点）+ 空 advances → 空结果不 panic。
#[test]
fn path_text_single_point_empty_advances() {
    let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(0.0, 0.0) }];
    let pt = place_on_path(&segs, &[], 0.0, 4);
    assert!(pt.glyphs.is_empty());
}

/// B15-T4: 直线路径切线角精确为 0（自研 atan2 正确性，含 y 向下）。
#[test]
fn path_text_line_angle_zero() {
    let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(100.0, 0.0) }];
    let pt = place_on_path(&segs, &[10.0; 4], 0.0, 32);
    for g in &pt.glyphs {
        assert!(g.angle.abs() < 1e-3, "水平直线切线角应 0，实得 {}", g.angle);
    }
}

/// B15-T5: 竖直向下直线（y 增大）切线角应 ≈ +π/2（y 向下坐标系）。
///
/// 失败代表：自研 atan2 符号/象限错误（影响沿曲线排字的旋转）。
#[test]
fn path_text_vertical_down_angle() {
    let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(0.0, 100.0) }];
    let pt = place_on_path(&segs, &[10.0; 3], 0.0, 32);
    for g in &pt.glyphs {
        assert!(
            (g.angle - std::f32::consts::FRAC_PI_2).abs() < 5e-2,
            "向下直线切线角应 ≈π/2，实得 {}",
            g.angle
        );
    }
}

/// B15-T6: 负数 subdiv → 内部 max(1) 兜底，不 panic / 不除零。
#[test]
fn path_text_zero_subdiv_no_panic() {
    let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(50.0, 0.0) }];
    let pt = place_on_path(&segs, &[10.0; 3], 0.0, 0);
    assert_eq!(pt.glyphs.len(), 3);
    assert!(!pt.total_len.is_nan());
}

/// B15-T7: reveal 进度单调不变量——非均匀 advance 下仍单调（核心不变量）。
#[test]
fn reveal_monotone_nonuniform_advances() {
    let plan = RevealPlan::from_advances(&[3.0, 1.0, 7.0, 2.0, 5.0]);
    let mut prev = plan.ink_at(0.0);
    for k in 1..=200 {
        let p = k as f32 / 200.0;
        let cur = plan.ink_at(p);
        for (a, b) in prev.iter().zip(cur.iter()) {
            assert!(b.ink >= a.ink - 1e-6, "非均匀 advance 墨量非单调 @p={p}");
        }
        prev = cur;
    }
}

/// B15-T8: reveal 起点/终点边界：p=0 全 0；p=1 全 1（非均匀步进，检查累计归一化）。
#[test]
fn reveal_endpoints_exact() {
    let plan = RevealPlan::from_advances(&[3.0, 1.0, 7.0, 2.0, 5.0]);
    assert!(plan.ink_at(0.0).iter().all(|g| g.ink == 0.0), "p=0 应全未见");
    assert!(plan.ink_at(1.0).iter().all(|g| g.is_complete()), "p=1 应全完整");
    // 末尾 span 应精确合到 1。
    assert!((plan.spans.last().unwrap().1 - 1.0).abs() < 1e-6);
}

/// B15-T9: reveal 负 advance（非法输入）不应 panic / 产生 NaN ink。
#[test]
fn reveal_negative_advance_no_nan() {
    let plan = RevealPlan::from_advances(&[-5.0, 10.0, 5.0]);
    let out = plan.ink_at(0.5);
    assert_eq!(out.len(), 3);
    for g in &out {
        assert!(!g.ink.is_nan(), "负 advance 产生 NaN ink");
    }
}

/// B15-T10: StrokeReveal 越界 t 钳到 [0,1]，同 t 逐位一致（确定性）。
#[test]
fn reveal_track_determinism_and_clamp() {
    let sr = StrokeReveal::linear(&[10.0; 5], 1.5);
    assert_eq!(sr.progress_at(-3.0), 0.0);
    assert_eq!(sr.progress_at(999.0), 1.0);
    let a = sr.glyphs_at(0.75);
    let b = sr.glyphs_at(0.75);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.ink.to_bits(), y.ink.to_bits());
    }
}
