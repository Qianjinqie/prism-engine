//! 独立复核探针 —— prism-text P1-4 空/退化路径。
use prism_raster::geom::Point;
use prism_text::path_text::{place_on_path, Seg};

/// 空 segs -> 空布局（glyphs 空, total_len 0）。
#[test]
fn probe_p1_4_empty_segs() {
    let pt = place_on_path(&[], &[10.0; 5], 0.0, 16);
    assert!(pt.glyphs.is_empty(), "空 segs 应无字形");
    assert_eq!(pt.total_len, 0.0, "空 segs total_len 应 0");
}

/// 新攻击：单点退化路径（p0==p1，长度 0）——非空但退化，不应 panic/NaN。
#[test]
fn probe_p1_4_single_point_degenerate() {
    let segs = [Seg::Line { p0: Point::new(5.0, 5.0), p1: Point::new(5.0, 5.0) }];
    let pt = place_on_path(&segs, &[4.0; 3], 0.0, 8);
    // 退化路径仍应产出字形（在端点），但所有坐标有限。
    for g in &pt.glyphs {
        assert!(g.pos.x.is_finite() && g.pos.y.is_finite(), "退化路径坐标 NaN/Inf");
        assert!(g.angle.is_finite(), "退化路径角 NaN/Inf");
    }
    assert!(pt.total_len.is_finite() && pt.total_len >= 0.0, "total_len 非法");
}
