//! 独立复核探针 —— prism-texture P1-5 空/退化 stops。
use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;
use prism_texture::gradient::{Gradient, GradientKind, Stop};

fn fixture() -> Fbuf16F {
    let mut fb = Fbuf16F::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            fb.set(x, y, [0.5, 0.5, 0.5, 1.0]);
        }
    }
    fb
}

/// 空 stops 应安全（且语义上）输出黑。
#[test]
fn probe_p1_5_empty_stops() {
    let g = Gradient {
        kind: GradientKind::Linear,
        angle: 0.0,
        center: [0.5, 0.5],
        stops: vec![],
    };
    let out = g.eval_window(&fixture());
    // 所有像素应为 (0,0,0)（透明黑约定），至少 alpha 之外rgb=0。
    for y in 0..8 {
        for x in 0..8 {
            let px = out.get(x, y);
            assert_eq!(&px[0..3], &[0.0, 0.0, 0.0], "空 stops 非黑输出 @ ({x},{y})");
        }
    }
}

/// 新攻击：单 stop（非空但退化）——不应 panic，输出应为该色（或合理）。
#[test]
fn probe_p1_5_single_stop() {
    let g = Gradient {
        kind: GradientKind::Linear,
        angle: 0.0,
        center: [0.5, 0.5],
        stops: vec![Stop { at: 0.5, color: [0.2, 0.4, 0.6] }],
    };
    let out = g.eval_window(&fixture());
    for y in 0..8 {
        for x in 0..8 {
            let px = out.get(x, y);
            assert!(px[0].is_finite() && px[1].is_finite() && px[2].is_finite(), "单 stop NaN");
        }
    }
}
