//! B14 集成测试（prism-texture）：程序化纹理族 **seed 决定论** 全集扫描。
//!
//! 对五种纹理（paper / noise / gradient / pattern / voronoi）+ 各模式的
//! 全部参数组合：同 seed 逐位一致（纯函数）、异 seed 去相关（随机源经
//! PCG 派生）；golden 指纹冻结。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;
use prism_texture::gradient::{Gradient, GradientKind};
use prism_texture::noise::{NoiseTexture, NoiseTextureKind};
use prism_texture::paper::Paper;
use prism_texture::pattern::{Pattern, PatternKind};
use prism_texture::voronoi::{Voronoi, VoronoiMode};
use prism_texture::default_registry;

fn fixture() -> Fbuf16F {
    let mut fb = Fbuf16F::new(32, 32);
    for y in 0..32u32 {
        for x in 0..32u32 {
            let v = ((x * 7 + y * 13) % 17) as f32 / 16.0;
            fb.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
        }
    }
    fb
}

fn buffer_md5(fb: &Fbuf16F) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for px in &fb.pixels {
        for byte in px.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

#[test]
fn noise_texture_full_seed_determinism() {
    let win = fixture();
    for kind in [
        NoiseTextureKind::Value,
        NoiseTextureKind::Simplex,
        NoiseTextureKind::FbmValue,
        NoiseTextureKind::FbmSimplex,
    ] {
        let t = NoiseTexture { kind, seed: 7, ..NoiseTexture::default() };
        assert_eq!(t.eval_window(&win), t.eval_window(&win), "{kind:?} 逐位一致");
        let s = NoiseTexture { kind, seed: 8, ..NoiseTexture::default() };
        assert_ne!(
            t.eval_window(&win).pixels,
            s.eval_window(&win).pixels,
            "{kind:?} seed 必须去相关"
        );
    }
}

#[test]
fn paper_texture_seed_determinism() {
    let win = fixture();
    let p = Paper { seed: 3, ..Paper::default() };
    assert_eq!(p.eval_window(&win), p.eval_window(&win));
    let q = Paper { seed: 4, ..Paper::default() };
    assert_ne!(p.eval_window(&win).pixels, q.eval_window(&win).pixels);
}

#[test]
fn gradient_all_kinds_deterministic() {
    let win = fixture();
    for kind in [GradientKind::Linear, GradientKind::Radial, GradientKind::Angular] {
        let mut g = Gradient::two_color([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        g.kind = kind;
        assert_eq!(g.eval_window(&win), g.eval_window(&win), "{kind:?} 逐位一致");
    }
}

#[test]
fn pattern_all_kinds_deterministic() {
    let win = fixture();
    for kind in [
        PatternKind::Checker,
        PatternKind::Stripe,
        PatternKind::Grid,
        PatternKind::Dot,
    ] {
        let p = Pattern { kind, ..Pattern::default() };
        assert_eq!(p.eval_window(&win), p.eval_window(&win), "{kind:?} 逐位一致");
    }
}

#[test]
fn voronoi_all_modes_seed_determinism() {
    let win = fixture();
    for mode in [VoronoiMode::F1, VoronoiMode::Edges, VoronoiMode::Cells] {
        let v = Voronoi { mode, seed: 11, ..Voronoi::default() };
        assert_eq!(v.eval_window(&win), v.eval_window(&win), "{mode:?} 逐位一致");
        let w = Voronoi { mode, seed: 12, ..Voronoi::default() };
        assert_ne!(
            v.eval_window(&win).pixels,
            w.eval_window(&win).pixels,
            "{mode:?} seed 必须去相关"
        );
    }
}

#[test]
fn texture_family_golden_fingerprints() {
    let win = fixture();
    let goldens: [(&str, u64); 5] = [
        ("paper", buffer_md5(&Paper::default().eval_window(&win))),
        ("noise", buffer_md5(&NoiseTexture::default().eval_window(&win))),
        ("gradient", buffer_md5(&Gradient::two_color([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).eval_window(&win))),
        ("pattern", buffer_md5(&Pattern::default().eval_window(&win))),
        ("voronoi", buffer_md5(&Voronoi::default().eval_window(&win))),
    ];
    for (name, got) in &goldens {
        let frozen = GOLDEN_TEXTURE.iter().find(|(n, _)| n == name).map(|(_, h)| *h);
        assert_eq!(Some(*got), frozen, "texture {name} golden 指纹漂移：{got}");
    }
}

/// 纹理族 golden 指纹（FNV-1a 64；首运行冻结）。
const GOLDEN_TEXTURE: [(&str, u64); 5] = [
    ("paper", 0xd07d_21a0_cc1a_f6bc),
    ("noise", 0xe53d_c327_8279_dd6a),
    ("gradient", 0xb78c_4d0c_a6dd_d7e5),
    ("pattern", 0x5ce1_6c54_6c29_6f25),
    ("voronoi", 0xf302_807b_b748_39a5),
];

#[test]
fn registry_builds_all_textures() {
    let reg = default_registry();
    for name in reg.names() {
        let t = reg.create(name).expect("已注册纹理必须可构造");
        let _ = t.halo_radius();
    }
}
