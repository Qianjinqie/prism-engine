//! 生成艺术/矢量攻击性边界测试。只写测试，不改源码。

use prism_genart::draw_on::progress_contours;
use prism_genart::generators::{
    CellularAutomata, DifferentialGrowth, FlowField, FractalMandelbrot, NoiseModulation, VoronoiArt,
};
use prism_raster::geom::{Point, Transform};
use prism_vector::humanize::{humanize_with, Humanize};
use prism_vector::morph::{morph, resample};
use prism_vector::path::{Path, SubPath, Vertex};

// --------------------------- morph ---------------------------

fn square(s: f32) -> Path {
    Path::from_polygon(&[
        Point::new(0.0, 0.0),
        Point::new(s, 0.0),
        Point::new(s, s),
        Point::new(0.0, s),
    ])
}

/// morph 端点吻合：u=0 → 源、u=1 → 目标（逐顶点位）。
#[test]
fn morph_endpoints_exact_vertices() {
    let a = square(10.0);
    let b = square(20.0);
    let m0 = morph(&a, &b, 0.0, 16);
    let m1 = morph(&a, &b, 1.0, 16);
    // u=0 的每个顶点与 resample(a) 逐位一致（插入 u*(b-a)=0）。
    let ra = resample(
        &[Point::new(0.0, 0.0), Point::new(10.0, 0.0), Point::new(10.0, 10.0), Point::new(0.0, 10.0)],
        true,
        16,
    );
    for (v, r) in m0.subpaths[0].vertices.iter().zip(ra.iter()) {
        assert!((v.p.x - r.x).abs() < 1e-4 && (v.p.y - r.y).abs() < 1e-4, "u=0 应回到源");
    }
    // u=1 的包围盒应达 20。
    let maxx = m1.subpaths[0].vertices.iter().map(|v| v.p.x).fold(f32::MIN, f32::max);
    assert!((maxx - 20.0).abs() < 1e-3, "u=1 应达目标尺度 20，实得 {maxx}");
}

/// morph segments=0 退化 → 不 panic（resample 返回单点）。
#[test]
fn morph_zero_segments_no_panic() {
    let a = square(10.0);
    let b = square(20.0);
    let m = morph(&a, &b, 0.5, 0);
    assert_eq!(m.subpaths.len(), 1);
}

/// morph 空路径 / 子路径数不一致 → 不 panic（取较小者）。
#[test]
fn morph_mismatched_subpaths_no_panic() {
    let mut a = square(10.0);
    a.push(SubPath { vertices: vec![Vertex::anchor(0.0, 0.0), Vertex::anchor(1.0, 1.0)], closed: false });
    let b = square(20.0); // 只有 1 条
    let m = morph(&a, &b, 0.5, 8);
    assert_eq!(m.subpaths.len(), 1, "应取较小子路径数");
}

/// morph u 越界（<0 / >1）→ 线性外推（确定性；文档未钳，仅核查不 panic/不 NaN）。
#[test]
fn morph_out_of_range_u_no_nan() {
    let a = square(10.0);
    let b = square(20.0);
    for u in [-0.5f32, 1.5] {
        let m = morph(&a, &b, u, 8);
        for v in &m.subpaths[0].vertices {
            assert!(!v.p.x.is_nan() && !v.p.y.is_nan(), "u={u} 产 NaN");
        }
    }
}

/// resample 空点集不 panic。
#[test]
fn resample_empty_no_panic() {
    let r = resample(&[], false, 8);
    assert!(r.is_empty() || r.len() == 9);
    let r0 = resample(&[], false, 0);
    assert!(r0.is_empty());
}

/// resample 零长度退化（全部点重合）→ 全返回同点，不 NaN。
#[test]
fn resample_degenerate_no_nan() {
    let pts = [Point::new(5.0, 5.0), Point::new(5.0, 5.0), Point::new(5.0, 5.0)];
    let r = resample(&pts, false, 4);
    for p in &r {
        assert!((p.x - 5.0).abs() < 1e-6 && (p.y - 5.0).abs() < 1e-6, "退化应全同点");
    }
}

// --------------------------- humanize ---------------------------

/// humanize 同种子逐位一致；不同种子去相关（核心铁律）。
#[test]
fn humanize_seed_determinism() {
    let p = square(12.0);
    let a = humanize_with(&p, &Humanize::new(0x1234, 2.0));
    let b = humanize_with(&p, &Humanize::new(0x1234, 2.0));
    assert_eq!(a, b, "同种子逐位一致");
    let c = humanize_with(&p, &Humanize::new(0x5678, 2.0));
    assert_ne!(a, c, "异种子应去相关");
}

/// humanize jitter=0 恒等（逐位）。
#[test]
fn humanize_zero_jitter_identity() {
    let p = square(12.0);
    let a = humanize_with(&p, &Humanize::new(9, 0.0));
    assert_eq!(a, p, "jitter=0 应逐位恒等");
}

/// humanize 扰动幅度有界（锚点 ≤ jitter，控制点 ≤ jitter·scale）。
#[test]
fn humanize_amplitude_bounded() {
    let jitter = 1.5f32;
    let scale = 2.0f32;
    let mut p = square(10.0);
    p.subpaths[0]
        .vertices
        .insert(1, Vertex::control(5.0, -5.0));
    let h = humanize_with(&p, &Humanize { seed: 3, jitter, control_scale: scale });
    let base = &p.subpaths[0].vertices;
    let out = &h.subpaths[0].vertices;
    for (v0, v1) in base.iter().zip(out.iter()) {
        let bound = if v0.control { jitter * scale } else { jitter };
        let dx = (v1.p.x - v0.p.x).abs();
        let dy = (v1.p.y - v0.p.y).abs();
        assert!(dx <= bound + 1e-4 && dy <= bound + 1e-4, "扰动越界 dx={dx} dy={dy} bound={bound}");
    }
}

// --------------------------- draw-on ---------------------------

/// draw-on 进度单调（面积不随 s 回退）；s=0 空、s=1 全。
#[test]
fn draw_on_progress_monotone() {
    let p = square(40.0);
    assert!(progress_contours(&p, 0.0, &Transform::IDENTITY).is_empty());
    let mut last_len = 0usize;
    for k in 0..=10 {
        let s = k as f32 / 10.0;
        let c = progress_contours(&p, s, &Transform::IDENTITY);
        let pts: usize = c.iter().map(|x| x.len()).sum();
        assert!(pts >= last_len, "s={s} 轮廓点退回：{pts} < {last_len}");
        last_len = pts;
    }
    assert!(last_len > 0, "s=1 应有轮廓");
}

/// draw-on 空路径不 panic。
#[test]
fn draw_on_empty_path_no_panic() {
    let p = Path::new();
    let c = progress_contours(&p, 0.5, &Transform::IDENTITY);
    assert!(c.is_empty());
}

/// draw-on 同 s 逐位一致（确定性）。
#[test]
fn draw_on_deterministic_bitwise() {
    let p = square(30.0);
    let a = progress_contours(&p, 0.37, &Transform::IDENTITY);
    let b = progress_contours(&p, 0.37, &Transform::IDENTITY);
    assert_eq!(a, b);
}

// --------------------------- generators ---------------------------

/// 六个生成器同种子同尺寸逐位一致（核心铁律）。
#[test]
fn generators_deterministic_all_six() {
    macro_rules! chk {
        ($g:expr) => {{
            let g = $g;
            let a = g.generate(48, 48);
            let b = g.generate(48, 48);
            assert_eq!(a.coverage.data, b.coverage.data, "生成器非确定性");
        }};
    }
    chk!(FlowField::default());
    chk!(DifferentialGrowth::default());
    chk!(CellularAutomata::default());
    chk!(VoronoiArt::default());
    chk!(FractalMandelbrot::default());
    chk!(NoiseModulation::default());
}

/// 零尺寸画布（0×0 / 1×0）→ 不 panic。
#[test]
fn generators_zero_size_no_panic() {
    let _ = FlowField::default().generate(0, 0);
    let _ = VoronoiArt::default().generate(1, 0);
    let _ = CellularAutomata::default().generate(0, 1);
    let _ = FlowField::default().generate(1, 1);
}
