//! B14 集成测试：NPR 族与胶片仿真包的 **golden-file** 与 **seed 决定论**。
//!
//! # 判定方式
//!
//! - **golden-file**：固定输入窗口 + 固定参数 → 输出位模式（RGBA16F 小端
//!  序列化）的 MD5 指纹与冻结常量比对（同输入逐位复现 ——
//!   格锚）。指纹用 RFC 1321 官方向量锚定的整数 MD5（零依赖、禁 libm）。
//! - **seed 决定论**：同 seed 逐位一致、异 seed 去相关。
//!
//! # 验证集
//!
//! toon / kuwahara / halftone / watercolor / pixelate / sketch（NPR 六风格）
//! + halation / layered_bloom / halation_bloom / film_grain（胶片包四算子）。

use prism_io::fbuf::Fbuf16F;
use prism_postfx::effects::npr::{Halftone, Kuwahara, Pixelate, Sketch, Toon, Watercolor};
use prism_postfx::effects::{
    FilmGrain, FilmFormat, Halation, HalationAndBloom, LayeredBloom,
};
use prism_render::tile::TileEffect;

// ---------------------------------------------------------------------------
// MD5（RFC 1321）：整数实现（零依赖），官方向量另由 tile_consistency 锚定。
// ---------------------------------------------------------------------------

mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
        5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee, 0xf57c_0faf, 0x4787_c62a, 0xa830_4613,
        0xfd46_9501, 0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be, 0x6b90_1122, 0xfd98_7193,
        0xa679_438e, 0x49b4_0821, 0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa, 0xd62f_105d,
        0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8, 0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
        0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a, 0xfffa_3942, 0x8771_f681, 0x6d9d_6122,
        0xfde5_380c, 0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70, 0x289b_7ec6, 0xeaa1_27fa,
        0xd4ef_3085, 0x0488_1d05, 0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665, 0xf429_2244,
        0x432a_ff97, 0xab94_23a7, 0xfc93_a039, 0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
        0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1, 0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb,
        0xeb86_d391,
    ];

    fn round_f(i: usize, x: u32, y: u32, z: u32) -> u32 {
        match i / 16 {
            0 => (x & y) | (!x & z),
            1 => (x & z) | (y & !z),
            2 => x ^ y ^ z,
            _ => y ^ (x | !z),
        }
    }

    pub fn hex(data: &[u8]) -> String {
        let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
        msg.extend_from_slice(data);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
        let (mut a0, mut b0, mut c0, mut d0) =
            (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, word) in m.iter_mut().enumerate() {
                *word = u32::from_le_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64 {
                let g: usize = match i / 16 {
                    0 => i,
                    1 => (5 * i + 1) % 16,
                    2 => (3 * i + 5) % 16,
                    _ => (7 * i) % 16,
                };
                let tmp: u32 = round_f(i, b, c, d)
                    .wrapping_add(a)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(tmp.rotate_left(S[i]));
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        [a0, b0, c0, d0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn buffer_md5(fb: &Fbuf16F) -> String {
    let bytes: Vec<u8> = fb.pixels.iter().flat_map(|h| h.to_le_bytes()).collect();
    md5::hex(&bytes)
}

/// 固定测试图：32×24，确定性图案（中央亮块 + 斜向渐变 + 硬边）。
fn fixture() -> Fbuf16F {
    let mut fb = Fbuf16F::new(32, 24);
    for y in 0..24u32 {
        for x in 0..32u32 {
            let base = (x + y) as f32 / 55.0;
            let hard = if x < 16 { 0.15 } else { 0.85 };
            let r = (base + hard) * 0.5;
            let g = (1.0 - base) * 0.7;
            let b = if (8..24).contains(&x) && (6..18).contains(&y) { 0.9 } else { 0.2 };
            fb.set(x, y, [r, g, b, 1.0]);
        }
    }
    fb
}

// ---------------------------------------------------------------------------
// golden-file：各效果指纹冻结
// ---------------------------------------------------------------------------

#[test]
fn npr_family_golden_fingerprints() {
    let win = fixture();
    let goldens: [(&str, String); 6] = [
        ("toon", buffer_md5(&Toon::default().eval_window(&win))),
        ("kuwahara", buffer_md5(&Kuwahara::default().eval_window(&win))),
        ("halftone", buffer_md5(&Halftone::default().eval_window(&win))),
        ("watercolor", buffer_md5(&Watercolor::default().eval_window(&win))),
        ("pixelate", buffer_md5(&Pixelate::default().eval_window(&win))),
        ("sketch", buffer_md5(&Sketch::default().eval_window(&win))),
    ];
    // 冻结指纹（本机生成；跨平台逐位一致由 D-29 全整数/精确运算保证）。
    // 断言复算与冻结值一致（golden-file）；值随首次运行固定后写死。
    for (name, got) in &goldens {
        // 与冻结常量表比对（见 GOLDEN_NPR）。
        let frozen = GOLDEN_NPR.iter().find(|(n, _)| n == name).map(|(_, h)| *h);
        assert_eq!(
            Some(got.as_str()),
            frozen,
            "NPR {name} golden 指纹漂移：{got}"
        );
    }
}

/// NPR golden 指纹冻结表（32×24 fixture + default 参数）。
///
/// halation/bloom 类因 fixture 亮度低于阈值而逐位恒等（正确语义），故该
/// 测试只覆盖会改变像素的 NPR 风格算子。
const GOLDEN_NPR: [(&str, &str); 6] = [
    ("toon", "24274c0e0d4e51c74f4461db32aae126"),
    ("kuwahara", "05ab34754520f0f32adb998d0fdd40c4"),
    ("halftone", "14de269443bc8ff2027119cdbbe5fa98"),
    ("watercolor", "171f00e7dd50307b2570c97695b5f2f5"),
    ("pixelate", "ce71c33a63dd8f0cd9bc4f3251774c26"),
    ("sketch", "3659fdfb1f0ed1ae4b85fa99a42ec50a"),
];

#[test]
fn film_pack_golden_fingerprints() {
    let win = fixture();
    let goldens: [(&str, String); 4] = [
        ("halation", buffer_md5(&Halation::default().eval_window(&win))),
        ("layered_bloom", buffer_md5(&LayeredBloom::default().eval_window(&win))),
        ("halation_bloom", buffer_md5(&HalationAndBloom::default().eval_window(&win))),
        ("film_grain", buffer_md5(&FilmGrain::default().eval_window(&win))),
    ];
    for (name, got) in &goldens {
        let frozen = GOLDEN_FILM.iter().find(|(n, _)| n == name).map(|(_, h)| *h);
        assert_eq!(Some(got.as_str()), frozen, "film {name} golden 指纹漂移：{got}");
    }
}

/// 胶片包 golden 指纹冻结表。
///
/// 注：fixture 亮度低于 default 阈值 1.0 ⇒ halation / layered_bloom /
/// halation_bloom 输出逐位恒等（输入指纹），这是"低于阈值恒等"语义的
/// golden 锚定；film_grain 为逐像素颗粒（有独立指纹）。
const GOLDEN_FILM: [(&str, &str); 4] = [
    ("halation", "a04bbc4341c417d0b37730d8dffbbde4"),
    ("layered_bloom", "a04bbc4341c417d0b37730d8dffbbde4"),
    ("halation_bloom", "a04bbc4341c417d0b37730d8dffbbde4"),
    ("film_grain", "b30bc80082ee7e0eae30dc036b806dea"),
];

// ---------------------------------------------------------------------------
// seed 决定论
// ---------------------------------------------------------------------------

#[test]
fn npr_seeded_effects_are_deterministic_and_seed_sensitive() {
    let win = fixture();
    for seed in [0u64, 1, 12345] {
        let wc_a = Watercolor { seed, ..Watercolor::default() }.eval_window(&win);
        let wc_b = Watercolor { seed, ..Watercolor::default() }.eval_window(&win);
        assert_eq!(wc_a, wc_b, "watercolor seed={seed} 逐位一致");
    }
    let wc0 = Watercolor { seed: 0, ..Watercolor::default() }.eval_window(&win);
    let wc1 = Watercolor { seed: 1, ..Watercolor::default() }.eval_window(&win);
    assert_ne!(wc0.pixels, wc1.pixels, "watercolor seed 必须去相关");

    let fg0 = FilmGrain::default().eval_window(&win);
    let fg1 = FilmGrain { seed: 1, ..FilmGrain::default() }.eval_window(&win);
    assert_ne!(fg0.pixels, fg1.pixels, "film_grain seed 必须去相关");
    // 同 seed 同参逐位一致。
    assert_eq!(fg0, FilmGrain::default().eval_window(&win));
}

#[test]
fn film_grain_format_presets_change_texture() {
    let win = fixture();
    let g35 = FilmGrain { format: FilmFormat::Mm35, ..FilmGrain::default() }.eval_window(&win);
    let g16 = FilmGrain { format: FilmFormat::Mm16, ..FilmGrain::default() }.eval_window(&win);
    let g8 = FilmGrain { format: FilmFormat::Super8, ..FilmGrain::default() }.eval_window(&win);
    assert_ne!(g35.pixels, g16.pixels, "35mm vs 16mm 颗粒必须不同");
    assert_ne!(g16.pixels, g8.pixels, "16mm vs Super8 颗粒必须不同");
}

#[test]
fn npr_effects_are_pure() {
    let win = fixture();
    assert_eq!(
        Toon::default().eval_window(&win),
        Toon::default().eval_window(&win)
    );
    assert_eq!(
        Kuwahara::default().eval_window(&win),
        Kuwahara::default().eval_window(&win)
    );
    assert_eq!(
        Halftone::default().eval_window(&win),
        Halftone::default().eval_window(&win)
    );
    assert_eq!(
        Sketch::default().eval_window(&win),
        Sketch::default().eval_window(&win)
    );
    assert_eq!(
        Pixelate::default().eval_window(&win),
        Pixelate::default().eval_window(&win)
    );
    assert_eq!(
        Halation::default().eval_window(&win),
        Halation::default().eval_window(&win)
    );
}
