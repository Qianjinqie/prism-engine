//! 胶片/NPR 攻击性边界测试。只写测试，不改源码。

use prism_io::fbuf::Fbuf16F;
use prism_postfx::effects::film::{
    halation_energy_budget, CurveKind, FilmCurve, FilmGrain, Halation, LayeredBloom,
};
use prism_postfx::effects::npr::{Halftone, Kuwahara, Pixelate, Sketch, Toon, Watercolor};
use prism_render::tile::TileEffect;

fn noise_window(w: u32, h: u32, seed: u64) -> Fbuf16F {
    let mut win = Fbuf16F::new(w, h);
    let mut s = seed;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((s >> 33) & 0xFFFF) as f32 / 65535.0;
            win.set(x, y, [v, v * 0.7, 1.0 - v, 1.0]);
        }
    }
    win
}

/// B14-F1: FilmCurve 越界输入钳到 [0,1]（不 panic / 输出 ∈[0,1]）。
#[test]
fn film_curve_out_of_domain_clamped() {
    for kind in [CurveKind::Srgb, CurveKind::Vision3, CurveKind::Eterna] {
        let c = FilmCurve { kind, shoulder: 0.72, toe: 0.18 };
        for &x in &[-5.0f32, -0.0, 1.5, 100.0] {
            let y = c.eval(x);
            assert!((0.0..=1.0).contains(&y), "{kind:?} 越界 x={x} 输出 {y} 不在 [0,1]");
        }
    }
}

/// B14-F2: 胶片段单调（Srgb/Vision3/Eterna）——单调性破坏会致色调翻转。
#[test]
fn film_curves_monotone_dense() {
    for kind in [CurveKind::Srgb, CurveKind::Vision3, CurveKind::Eterna] {
        let c = FilmCurve { kind, shoulder: 0.72, toe: 0.18 };
        let mut prev = c.eval(0.0);
        for i in 1..=400 {
            let x = i as f32 / 400.0;
            let y = c.eval(x);
            assert!(y >= prev - 1e-6, "{kind:?} 非单调 @ x={x}: {prev} → {y}");
            prev = y;
        }
        assert_eq!(c.eval(1.0), 1.0, "{kind:?} 端点应 1");
    }
}

/// B14-F3: halation 能量守恒（threshold=0 全幅高光）——delta ≥ 0 且 ≤ 上界。
#[test]
fn halation_energy_bounded_threshold_zero() {
    let win = noise_window(24, 24, 0xABCD);
    let h = Halation { threshold: 0.0, ..Halation::default() };
    let (delta, bound) = halation_energy_budget(&win, &h);
    assert!(delta >= 0.0, "halation 不得减能：delta={delta}");
    assert!(delta <= bound + 1e-3, "halation 注入越界：{delta} > {bound}");
}

/// B14-F4: halation 负 tint（非法输入）不 panic；仅核查稳健性。
#[test]
fn halation_negative_tint_no_panic() {
    let win = noise_window(8, 8, 1);
    let h = Halation { tint: [-0.5, 0.0, 2.0], ..Halation::default() };
    let _ = h.eval_window(&win);
}

/// B14-F5: halation 零 levels（退化）不 panic / 不 vacuous。
#[test]
fn halation_zero_levels_no_panic() {
    let win = noise_window(8, 8, 2);
    let h = Halation { levels: 0, base_radius: 0, ..Halation::default() };
    let out = h.eval_window(&win);
    assert_eq!(out.width, 8);
}

/// B14-F6: LayeredBloom 低于阈值逐位恒等（阈值高，全幅不过阈）。
#[test]
fn layered_bloom_below_threshold_identity() {
    let win = noise_window(16, 12, 5); // 值 ∈[0,1]
    let b = LayeredBloom { threshold: 2.0, ..LayeredBloom::default() };
    assert_eq!(b.eval_window(&win), win, "全幅不越阈 → 恒等");
}

/// B14-F7: FilmGrain seed 决定论 + frame 去相关（核心铁律）。
#[test]
fn film_grain_seed_and_frame_determinism() {
    let win = noise_window(20, 16, 7);
    let g = FilmGrain { amount: 0.08, seed: 42, frame: 0, ..FilmGrain::default() };
    let a = g.eval_window(&win);
    let b = g.eval_window(&win);
    assert_eq!(a, b, "同 seed/frame 逐位一致");
    let f1 = FilmGrain { frame: 1, ..g }.eval_window(&win);
    assert_ne!(a.pixels, f1.pixels, "不同 frame 必须去相关");
}

/// B14-N1: Toon levels=0（退化）不 panic。
#[test]
fn toon_zero_levels_no_panic() {
    let win = noise_window(8, 8, 3);
    let t = Toon { levels: 0, softness: 0.0, edge_threshold: 0.0, edge_dark: 1.0 };
    let _ = t.eval_window(&win);
}

/// B14-N2: Toon 对纯黑像素（luma≈0）不产 NaN（scale 分支需保护）。
#[test]
fn toon_black_pixel_no_nan() {
    let mut win = Fbuf16F::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            win.set(x, y, [0.0, 0.0, 0.0, 1.0]);
        }
    }
    let out = Toon::default().eval_window(&win);
    for y in 0..4 {
        for x in 0..4 {
            let c = out.get(x, y);
            for k in 0..3 {
                assert!(!c[k].is_nan(), "纯黑像素 toon 产 NaN");
            }
        }
    }
}

/// B14-N3: NPR 族全体对 1×1 窗口不 panic（退化 tile）。
#[test]
fn npr_tiny_window_no_panic() {
    let win = noise_window(1, 1, 9);
    let _ = Toon::default().eval_window(&win);
    let _ = Kuwahara::default().eval_window(&win);
    let _ = Watercolor::default().eval_window(&win);
    let _ = Sketch::default().eval_window(&win);
    let _ = Halftone::default().eval_window(&win);
    let _ = Pixelate::default().eval_window(&win);
}

/// B14-N4: NPR 族确定性——同参两遍逐位一致。
#[test]
fn npr_family_deterministic() {
    let win = noise_window(13, 11, 0xBEEF);
    assert_eq!(Toon::default().eval_window(&win), Toon::default().eval_window(&win));
    assert_eq!(Kuwahara::default().eval_window(&win), Kuwahara::default().eval_window(&win));
    assert_eq!(Sketch::default().eval_window(&win), Sketch::default().eval_window(&win));
    assert_eq!(Watercolor::default().eval_window(&win), Watercolor::default().eval_window(&win));
}
