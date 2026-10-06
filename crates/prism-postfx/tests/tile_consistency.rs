//! 验收关卡 1（B4）：tile 化渲染与整幅渲染**逐字节一致**（复用 A5
//! `TileOrchestrator`，MD5 位模式指纹判定，RFC 1321 官方向量锚定）。
//!
//! # 验证集（可平移不变 = halo 型协议效果）
//! blur / bloom / ca / sharpen / denoise(à-trous) / warp / grade / lut3d
//! ——输出像素只依赖输入窗口内 `p ± halo` 的邻域，halo 足额申报 ⇒ tile
//! 化（16×9 网格 = 144 块）与整幅逐位一致。
//!
//! # 排除集（位置键控效果——协议缺口，非缺陷）
//! grain / glitch / vignette / dissolve / plasma / mandelbrot_bg 的图案
//! 锚定在**窗口局部坐标**；`TileEffect::eval_window` 不携带全局原点（A5
//! trait 只读对接，B4 不得私改协议），其整幅求值为规范语义（post 链 CPU
//! 参考路径）。tile 一致性承载需协议增补全局原点上下文——上报完工汇报。

use prism_core::SampleIndex;
use prism_io::fbuf::Fbuf16F;
use prism_postfx::denoise::AtrousSpatial;
use prism_postfx::effects::{
    ApplyLut, Blur, Bloom, ChromaticAberration, Grade, Lut3D, Sharpen,
};
use prism_postfx::warp::Warp;
use prism_render::orchestrator::{PixelSource, TileOrchestrator};
use prism_render::tile::{TileEffect, TileGrid};

// ---------------------------------------------------------------------------
// MD5（RFC 1321）指纹助手：整数运算实现（零依赖、确定性），
// 正确性以 RFC 官方向量锚定（见下方首个测试）。
// ---------------------------------------------------------------------------

mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
        5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    // T[i] = floor(2³² × |sin(i+1)|)（RFC 1321 表格常量，硬编码——禁 libm）。
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

    /// 摘要的十六进制串。
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

/// 缓冲指纹：RGBA16F 位模式序列化（小端字节）→ MD5。
fn buffer_md5(fb: &Fbuf16F) -> String {
    let bytes: Vec<u8> = fb.pixels.iter().flat_map(|h| h.to_le_bytes()).collect();
    md5::hex(&bytes)
}

// ---------------------------------------------------------------------------
// 确定性测试源：si 哈希派生 + 全局坐标平滑项混频（同 A5 关卡 1 的 HashSource）
// ---------------------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z: u64 = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct HashSource {
    seed: u64,
}

impl PixelSource for HashSource {
    fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4] {
        let mut z: u64 = self.seed
            ^ si.frame.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (((si.pixel.0 as u64) << 32) | si.pixel.1 as u64)
                .wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ si.node_id.rotate_left(11);
        let h: u64 = splitmix64(&mut z);
        let a: f32 = (h >> 40) as f32 / 16_777_216.0;
        let b: f32 = ((h >> 16) & 0xFF_FFFF) as f32 / 16_777_216.0;
        let cx: f32 = x as f32 * 0.03125; // 坐标平滑项（邻域效果的输入成分）
        let cy: f32 = y as f32 * 0.03125;
        [
            (cx + a).fract(),
            (cy + b).fract(),
            ((cx + cy) * 0.5 + a * b).fract(),
            1.0 - a * 0.5,
        ]
    }
}

// ---------------------------------------------------------------------------
// 指纹器自锚 + 关卡 1 判定
// ---------------------------------------------------------------------------

#[test]
fn md5_rfc1321_official_vectors() {
    assert_eq!(md5::hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5::hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        md5::hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
}

#[test]
fn checkpoint1_postfx_tiled_16x9_equals_fullframe_bit_identical() {
    // 关卡 1 主判定：192×108 ÷ 16 列 × 9 行 = 144 块 12×12 tile。
    // B4 halo 型效果逐个：tile 化与整幅的 RGBA16F 位模式 MD5 必须一致。
    let grid = TileGrid::counts(192, 108, 16, 9);
    let orch = TileOrchestrator::new(grid);
    let source = HashSource { seed: 0xB4 };

    let lut = ApplyLut::new(Lut3D::identity(17));
    let bloom = Bloom { threshold: 0.8, radius: 2, intensity: 0.3 };
    let cases: [(&str, &dyn TileEffect); 8] = [
        ("blur", &Blur { radius: 2 }),
        ("bloom", &bloom),
        ("ca", &ChromaticAberration { shift: (1.0, 0.0) }),
        ("sharpen", &Sharpen { amount: 0.5, radius: 1 }),
        ("denoise", &AtrousSpatial { iterations: 3, sigma_color: 0.1 }),
        ("warp", &Warp { scale: 1.5 }),
        ("grade", &Grade {
            exposure: 1.25,
            contrast: 1.15,
            saturation: 0.85,
            slope: [1.1, 1.0, 0.9],
            offset: [0.02, 0.0, 0.0],
        }),
        ("lut3d", &lut),
    ];

    for (name, effect) in cases {
        let full = orch.render_frame_full(&source, &[effect]);
        let tiled = orch.render_frame_tiled(&source, &[effect]);
        let (mf, mt) = (buffer_md5(&full), buffer_md5(&tiled));
        println!(
            "checkpoint1 {name:9} halo={:2}  full={mf}  tiled={mt}",
            effect.halo_radius()
        );
        assert_eq!(mf, mt, "{name}：tile 化必须与整幅逐字节一致");
    }
}

#[test]
fn checkpoint1_postfx_effect_chain_tiled_equals_fullframe() {
    // 效果链（路线图 §2.2：post { chain [...] } 任意组合）：总 halo =
    // 各级申报之和，逐级归纳保证核心区邻域读取与整幅一致。
    let grid = TileGrid::counts(192, 108, 16, 9);
    let orch = TileOrchestrator::new(grid);
    let source = HashSource { seed: 0xB4 };

    let blur = Blur { radius: 2 };
    let grade = Grade {
        exposure: 1.1,
        contrast: 1.05,
        saturation: 0.9,
        slope: [1.0, 1.0, 1.0],
        offset: [0.0, 0.0, 0.0],
    };
    let ca = ChromaticAberration { shift: (0.5, 0.0) };
    let bloom = Bloom { threshold: 0.8, radius: 2, intensity: 0.3 };
    let chain: [&dyn TileEffect; 4] = [&blur, &grade, &ca, &bloom];
    let sum_halo: u32 = chain.iter().map(|e| e.halo_radius()).sum();

    let full = orch.render_frame_full(&source, &chain);
    let tiled = orch.render_frame_tiled(&source, &chain);
    let (mf, mt) = (buffer_md5(&full), buffer_md5(&tiled));
    println!(
        "checkpoint1 chain(blur→grade→ca→bloom) sum_halo={sum_halo}  full={mf}  tiled={mt}"
    );
    assert_eq!(mf, mt, "效果链 tile 化必须与整幅逐字节一致");
}

#[test]
fn checkpoint1_holds_on_indivisible_frames_and_mixed_halos() {
    // 加固：非整除帧（边缘 tile 收窄 + 帧边界钳制）、素数 tile、单 tile、
    // halo 0 与 halo > 0 混合链。
    let source = HashSource { seed: 42 };
    let blur = Blur { radius: 1 };
    let warp = Warp { scale: 1.25 };
    let grade = Grade {
        exposure: 1.0,
        contrast: 1.0,
        saturation: 1.0,
        slope: [1.0, 1.0, 1.0],
        offset: [0.0, 0.0, 0.0],
    };
    let chain: [&dyn TileEffect; 3] = [&grade, &blur, &warp]; // halo 0 + 1 + 2
    for grid in [
        TileGrid::new(64, 48, 16, 9),    // 非整除：4×6 网格，末行末列收窄
        TileGrid::new(64, 48, 7, 5),     // 素数尺寸 tile
        TileGrid::new(64, 48, 200, 90),  // 单 tile 覆盖全帧
    ] {
        let orch = TileOrchestrator::new(grid);
        let full = orch.render_frame_full(&source, &chain);
        let tiled = orch.render_frame_tiled(&source, &chain);
        assert_eq!(
            buffer_md5(&full),
            buffer_md5(&tiled),
            "网格 {}×{}（tile {}×{}）混合 halo 链必须逐字节一致",
            grid.cols(),
            grid.rows(),
            grid.tile_w,
            grid.tile_h
        );
    }
}
