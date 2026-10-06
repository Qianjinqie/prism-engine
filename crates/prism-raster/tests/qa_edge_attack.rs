//! prism-raster 对抗性边界测试（QA 通道）。
//!
//! 病态几何不 panic、不越界、输出确定；覆盖率数学性质（∈[0,1]、
//! 面积守恒、半开边界语义）在攻击面下仍成立。

use prism_raster::geom::{Point, Transform};
use prism_raster::path::PathBuilder;
use prism_raster::{rasterize, FillRule};

fn ok_coverage(cov: &prism_raster::Coverage) {
    for v in &cov.data {
        assert!(*v >= 0.0 && *v <= 1.0 + 1e-6, "覆盖率越界 {v}");
        assert!(!v.is_nan(), "覆盖率 NaN");
    }
}

#[test]
fn degenerate_contours_produce_empty() {
    // 空 / 单点 / 两点 / 全共线：无面积 → 全零覆盖率。
    let empty: Vec<Vec<Point>> = vec![];
    let cov = rasterize(&empty, FillRule::Winding, 4, 4);
    assert!(cov.data.iter().all(|v| *v == 0.0));

    let degenerate: Vec<Vec<Point>> = vec![
        vec![Point::new(1.0, 1.0)],
        vec![Point::new(0.0, 0.0), Point::new(3.0, 3.0)],
        vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ],
    ];
    let cov = rasterize(&degenerate, FillRule::EvenOdd, 4, 4);
    assert!(cov.data.iter().all(|v| *v == 0.0), "退化轮廓不得产出覆盖");
}

#[test]
fn zero_size_buffer_is_safe() {
    let tri = vec![Point::new(0.0, 0.0), Point::new(4.0, 0.0), Point::new(0.0, 4.0)];
    let a = rasterize(&[tri.clone()], FillRule::Winding, 0, 4);
    assert_eq!(a.data.len(), 0);
    let b = rasterize(&[tri], FillRule::Winding, 4, 0);
    assert_eq!(b.data.len(), 0);
}

#[test]
fn self_intersecting_bowtie_area() {
    // 经典 8 字形 (0,0)(2,2)(2,0)(0,2)：两瓣 winding 相反（+1/−1）。
    // 非零规则两瓣皆内（≠0）→ 覆盖 2；奇偶规则两瓣皆奇 → 同为 2。
    let bowtie = vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(2.0, 0.0),
        Point::new(0.0, 2.0),
    ];
    let w = rasterize(&[bowtie.clone()], FillRule::Winding, 4, 4);
    ok_coverage(&w);
    let e = rasterize(&[bowtie], FillRule::EvenOdd, 4, 4);
    let sum_w: f32 = w.data.iter().sum();
    let sum_e: f32 = e.data.iter().sum();
    assert!((sum_w - 2.0).abs() < 1e-4, "非零规则 8 字形覆盖 2，实际 {sum_w}");
    assert!((sum_e - 2.0).abs() < 1e-4, "奇偶规则 8 字形覆盖 2，实际 {sum_e}");
}

#[test]
fn vertex_on_band_boundary_no_double_count() {
    // 顶点恰在整数行上：上下两带共享边不得重复积分。
    // 方块 [0,2]×[0,2]（顶点在 0/2 上）：总面积恰 4。
    let sq = vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(0.0, 2.0),
    ];
    let cov = rasterize(&[sq], FillRule::Winding, 4, 4);
    ok_coverage(&cov);
    let sum: f32 = cov.data.iter().sum();
    assert!((sum - 4.0).abs() < 1e-5, "顶点对齐方块面积 4，实际 {sum}");
}

#[test]
fn thin_sliver_stays_bounded() {
    // 极细长条（宽 0.01px）：覆盖率 ∈ [0,1]，总量守恒。
    let sliver = vec![
        Point::new(0.0, 1.0),
        Point::new(10.0, 1.02),
        Point::new(10.0, 1.03),
        Point::new(0.0, 1.01),
    ];
    let cov = rasterize(&[sliver], FillRule::Winding, 12, 3);
    ok_coverage(&cov);
    let sum: f32 = cov.data.iter().sum();
    assert!((sum - 0.1).abs() < 0.05, "细条面积 ≈ 10×0.01 = 0.1，实际 {sum}");
}

#[test]
fn transformed_path_rasterizes_equivalently() {
    // 同一形状：直接以设备坐标光栅 vs 整数平移后光栅 → 覆盖率平移一致
    // （可精确表示坐标 + 整数平移，位模式不变）。
    let mut pb = PathBuilder::new();
    pb.move_to(Point::new(0.5, 0.5));
    pb.line_to(Point::new(2.5, 0.5));
    pb.line_to(Point::new(2.5, 2.5));
    pb.line_to(Point::new(0.5, 2.5));
    pb.close();
    let a = rasterize(&pb.flatten(&Transform::IDENTITY), FillRule::Winding, 6, 6);
    let shift = Transform::translation(2.0, 1.0);
    let b = rasterize(&pb.flatten(&shift), FillRule::Winding, 6, 6);
    for y in 0..6u32 {
        for x in 0..6u32 {
            let vb = if x + 2 < 6 && y + 1 < 6 { b.get(x + 2, y + 1) } else { 0.0 };
            assert_eq!(a.get(x, y).to_bits(), vb.to_bits(), "平移等价性破坏 ({x},{y})");
        }
    }
}

#[test]
fn huge_coordinates_dont_panic() {
    // 巨大坐标（1e6 量级）：不 panic、不越界、可见区域语义正确。
    let big = vec![
        Point::new(-1.0e6, -1.0e6),
        Point::new(2.0e6, -1.0e6),
        Point::new(2.0e6, 2.0e6),
        Point::new(-1.0e6, 2.0e6),
    ];
    let cov = rasterize(&[big], FillRule::Winding, 8, 8);
    ok_coverage(&cov);
    assert!((cov.get(4, 4) - 1.0).abs() < 1e-6, "巨大方块内部全覆盖");
}
