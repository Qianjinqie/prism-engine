//! B14 集成测试（prism-aesthetic）：二级调色选区数学断言 + golden 指纹。
//!
//! - Power Window 权重数学：中心 1 / 边界 0 / 羽化带单调；
//! - HSL 选区：全区间恒 1、色相环连通（红在 0/1 处同选）、亮度/饱和度区间；
//! - lift-gamma-gain：中性逐位恒等、lift/gain/gamma 方向正确、幂零 libm；
//! - golden：固定输入 + 参数 → 位模式指纹冻结。

use prism_aesthetic::secondary::{
    HslQualifier, LiftGammaGain, PowerWindow, Secondary, WindowShape,
};
use prism_aesthetic::curves::{Curve, CurvePoint, CurveSet};
use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

fn buffer_md5(fb: &Fbuf16F) -> u64 {
    // FNV-1a 64 位位模式指纹（集成测试用；golden 值冻结后稳定）。
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for px in &fb.pixels {
        for byte in px.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn fixture() -> Fbuf16F {
    let mut fb = Fbuf16F::new(24, 18);
    for y in 0..18u32 {
        for x in 0..24u32 {
            let r = x as f32 / 23.0;
            let g = y as f32 / 17.0;
            let b = if (x + y) % 2 == 0 { 0.9 } else { 0.1 };
            fb.set(x, y, [r, g, b, 1.0]);
        }
    }
    fb
}

#[test]
fn power_window_weight_math() {
    let w = PowerWindow { center: [0.5, 0.5], radius: [0.4, 0.4], feather: 0.2, shape: WindowShape::Ellipse, invert: false };
    assert_eq!(w.weight(0.5, 0.5), 1.0, "中心权重恒 1");
    assert_eq!(w.weight(0.95, 0.5), 0.0, "半径外权重 0");
    // 羽化带内单调不增。
    let mut prev = 1.0f32;
    for i in 0..=20 {
        let u = 0.5 + 0.4 * (i as f32 / 20.0);
        let cur = w.weight(u, 0.5);
        assert!(cur <= prev + 1e-6, "羽化带必须单调不增 @ {u}");
        prev = cur;
    }
}

#[test]
fn hsl_qualifier_hue_wrap_and_ranges() {
    // 色相环连通：红相区间 [0.9, 1.0] ∪ [0.0, 0.1] 应同时选中 hue≈0 与 ≈1。
    let q = HslQualifier {
        luma_range: [0.0, 1.0],
        hue_range: [0.95, 1.0],
        sat_range: [0.0, 1.0],
        softness: 0.05,
    };
    let pure_red = [1.0f32, 0.0, 0.0, 1.0];
    assert!(q.weight(pure_red) > 0.5, "红相区间应含 hue=0");
    // 亮度区间断言。
    let shadow = HslQualifier {
        luma_range: [0.0, 0.3],
        hue_range: [0.0, 1.0],
        sat_range: [0.0, 1.0],
        softness: 0.05,
    };
    assert!(shadow.weight([0.1, 0.1, 0.1, 1.0]) > 0.99);
    assert!(shadow.weight([0.9, 0.9, 0.9, 1.0]) < 0.01);
}

#[test]
fn lift_gamma_gain_neutral_and_directions() {
    let n = LiftGammaGain::default();
    assert!(n.is_neutral());
    assert_eq!(n.apply([0.3, 0.6, 0.9, 0.5]), [0.3, 0.6, 0.9, 0.5], "中性逐位恒等");

    // lift 抬阴影。
    let lift = LiftGammaGain { lift: 0.2, ..LiftGammaGain::default() };
    assert!((lift.apply_scalar(0.0) - 0.2).abs() < 1e-6);
    // gain 倍增高光。
    let gain = LiftGammaGain { gain: 1.5, ..LiftGammaGain::default() };
    assert!((gain.apply_scalar(0.5) - 0.75).abs() < 1e-6);
    // gamma > 1 提亮中灰（幂为 sqrt 混合，零 libm）。
    let g2 = LiftGammaGain { gamma: 2.0, gamma_steps: 8, ..LiftGammaGain::default() };
    assert!(g2.apply_scalar(0.25) > 0.25, "gamma 2 提亮中灰");
}

#[test]
fn curves_neutral_identity_and_monotone() {
    let cs = CurveSet::default();
    assert!(cs.is_identity());
    let c = [0.2f32, 0.5, 0.8, 1.0];
    assert_eq!(cs.apply(c), c, "中性曲线集逐位恒等");

    let curve = Curve::new(vec![
        CurvePoint { x: 0.0, y: 0.0 },
        CurvePoint { x: 0.5, y: 0.3 },
        CurvePoint { x: 1.0, y: 1.0 },
    ]);
    assert!(curve.is_monotone());
    let mut prev = -1.0;
    for i in 0..=100 {
        let y = curve.eval(i as f32 / 100.0);
        assert!(y >= prev - 1e-6, "曲线单调");
        prev = y;
    }
}

#[test]
fn secondary_golden_fingerprint() {
    let win = fixture();
    let sec = Secondary {
        qualifier: HslQualifier {
            luma_range: [0.2, 0.8],
            hue_range: [0.0, 0.5],
            sat_range: [0.1, 1.0],
            softness: 0.1,
        },
        lgg: LiftGammaGain { lift: 0.05, gamma: 1.4, gain: 1.1, gamma_steps: 8 },
        window: PowerWindow::default(),
    };
    let h = buffer_md5(&sec.eval_window(&win));
    assert_eq!(h, SECONDARY_GOLDEN, "secondary golden 指纹漂移：{h}");
}

/// secondary golden 指纹（FNV-1a 64）。
///
/// 本指纹由 **P0-1 修复后（先开方后连乘，规避 `x^p` 下溢）的正确实现**
/// 重新生成——此前 `0xfe21_2aef_9cbf_36ee` 是冻结在错误下溢输出上的
/// 错误 golden（fixture 用 `gamma: 1.4` 恰为缺陷值），已废弃。
const SECONDARY_GOLDEN: u64 = 0x4d30_1ec1_f803_8ed6;

#[test]
fn secondary_is_deterministic() {
    let win = fixture();
    let sec = Secondary {
        lgg: LiftGammaGain { lift: 0.05, gamma: 1.4, gain: 1.1, gamma_steps: 8 },
        ..Secondary::default()
    };
    assert_eq!(sec.eval_window(&win), sec.eval_window(&win), "纯函数契约");
    assert_eq!(sec.halo_radius(), 0);
}
