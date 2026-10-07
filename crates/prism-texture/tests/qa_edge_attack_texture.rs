//! 程序化纹理攻击性边界测试。只写测试，不改源码。
//! 与既有 seed_determinism.rs 不重叠：聚焦**退化参数 / 空输入 / 极端频率**。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;
use prism_texture::gradient::{Gradient, GradientKind, Stop};
use prism_texture::paper::Paper;
use prism_texture::voronoi::{Voronoi, VoronoiMode};

fn fixture() -> Fbuf16F {
    let mut fb = Fbuf16F::new(16, 16);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let v = (x + y) as f32 / 30.0;
            fb.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
        }
    }
    fb
}

/// B14-TX1: Gradient 空色标（stops 为空，非法输入）——库代码不应 panic。
///
/// 失败代表：eval_stops 无条件索引 stops[0]，空输入 index out of bounds。
#[test]
fn gradient_empty_stops_no_panic() {
    let g = Gradient {
        kind: GradientKind::Linear,
        angle: 0.0,
        center: [0.5, 0.5],
        stops: vec![],
    };
    let win = fixture();
    // 不应 panic；若 panic 即缺陷。
    let _ = g.eval_window(&win);
}

/// B14-TX2: Gradient 单色标（1 个）——至少 2 个是契约，1 个不应 panic。
#[test]
fn gradient_single_stop_no_panic() {
    let g = Gradient {
        kind: GradientKind::Radial,
        angle: 0.0,
        center: [0.5, 0.5],
        stops: vec![Stop { at: 0.0, color: [0.2, 0.4, 0.6] }],
    };
    let win = fixture();
    let _ = g.eval_window(&win);
}

/// B14-TX3: 极端角度（0/360/负/超大）不产 NaN。
#[test]
fn gradient_extreme_angle_no_nan() {
    let mut g = Gradient::two_color([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    g.kind = GradientKind::Linear;
    for a in [0.0f32, 360.0, -45.0, 720.5] {
        g.angle = a;
        let out = g.eval_window(&fixture());
        for y in 0..out.height {
            for x in 0..out.width {
                let px = out.get(x, y);
                for k in 0..3 {
                    assert!(!px[k].is_nan(), "angle={a} 产 NaN");
                }
            }
        }
    }
}

/// B14-TX4: Gradient 确定性（同参两遍逐位一致）。
#[test]
fn gradient_deterministic() {
    let g = Gradient {
        kind: GradientKind::Angular,
        angle: 30.0,
        center: [0.5, 0.5],
        stops: vec![
            Stop { at: 0.0, color: [0.0, 0.0, 0.0] },
            Stop { at: 0.5, color: [1.0, 0.0, 0.0] },
            Stop { at: 1.0, color: [1.0, 1.0, 1.0] },
        ],
    };
    let w = fixture();
    assert_eq!(g.eval_window(&w), g.eval_window(&w));
}

/// B14-TX5: Paper 极端频率（0 / 极大）不产 NaN、不 panic。
#[test]
fn paper_extreme_freq_no_nan() {
    for ff in [0.0f32, 1e6] {
        let p = Paper { fiber_freq: ff, blotch_freq: ff, ..Paper::default() };
        let out = p.eval_window(&fixture());
        for y in 0..out.height {
            for x in 0..out.width {
                let px = out.get(x, y);
                assert!(!px[0].is_nan() && !px[1].is_nan() && !px[2].is_nan(), "freq={ff} 产 NaN");
            }
        }
    }
}

/// B14-TX6: Paper multiply 模式与 replace 模式均确定性。
#[test]
fn paper_multiply_deterministic() {
    let w = fixture();
    let p = Paper { multiply: true, seed: 7, ..Paper::default() };
    assert_eq!(p.eval_window(&w), p.eval_window(&w));
    let q = Paper { multiply: false, seed: 7, ..Paper::default() };
    assert_eq!(q.eval_window(&w), q.eval_window(&w));
}

/// B14-TX7: Voronoi 各模式 1×1 退化窗口不 panic。
#[test]
fn voronoi_tiny_window_no_panic() {
    let w = Fbuf16F::new(1, 1);
    for mode in [VoronoiMode::Cells, VoronoiMode::Edges, VoronoiMode::F1] {
        let v = Voronoi { mode, ..Voronoi::default() };
        let _ = v.eval_window(&w);
    }
}
