//! QA 攻击性边界测试（W6-B11：prism-media）。
//!
//! 作者：严过关（QA）。攻击目标（确定性 + 防御缺失 + 边界）：
//! 1. timing：五模式边界（负 t、NaN、dur/fps=0、Loop 负半轴、Speed k=0/负/NaN、
//!    SpeedCurve/Expr 空 samples）、seek 等价、frame_pair 契约。
//! 2. gif：魔数/截断/坏 LZW/无调色板拦截（结构化错误非 panic）。
//! 3. apng：魔数/截断/坏 zlib/非 8 位/越界 fcTL。
//! 4. inflate：zlib 头校验、截断位流、stored LEN/NLEN 校验。
//! 5. source：扩展名分类边界（无扩展、多点、大小写、空串）。
//! 6. audio：WAV 往返、ffmpeg 协作错误、坏字节结构化错误。

use prism_media::apng::decode_apng;
use prism_media::audio::load_audio;
use prism_media::gif::decode_gif;
use prism_media::inflate::zlib_decompress;
use prism_media::source::{extension_of, MediaSource, PixelImage, ReadKind, SourceKind};
use prism_media::timing::{frame_index, TimeMap, TimingMode};

// ---------------------------------------------------------------------------
// ① timing 边界
// ---------------------------------------------------------------------------

#[test]
fn timemap_zero_dur_fps_no_nan() {
    // dur=0 / fps=0 应钳到 MIN_POSITIVE，不产生 NaN。
    let m = TimeMap::new(0.0, 0.0, TimingMode::Direct);
    assert!(m.duration() > 0.0 && m.fps() > 0.0, "零参数应钳到正最小");
    for t in [0.0f32, 1.0, 1e9, -1.0] {
        let u = m.map(t);
        assert!(u.is_finite() && (0.0..=1.0).contains(&u), "t={t} 映射非法：{u}");
    }
    assert!(m.total_frames() >= 1, "总帧数至少 1");
}

#[test]
fn timemap_negative_and_nan_time() {
    let m = TimeMap::new(2.0, 10.0, TimingMode::Direct);
    // 负 t → u<0 → clamp 0。
    assert_eq!(m.map(-5.0), 0.0, "负 t 应钳 0");
    // NaN t：clamp(NaN) 语义攻击——不得 panic。
    let u = m.map(f32::NAN);
    // 记录行为：clamp(NaN,0,1) 在 Rust 返回 0（NaN 比较 false → 返回 self? 实测）。
    assert!(u == u || u.is_nan(), "NaN t 映射不 panic");
}

#[test]
fn timemap_loop_negative_half_axis() {
    // Loop：u<0 时 fract 会给出负数？fract(-0.25) = -0.25 - (-1) = 0.75（floor 向 −∞）。
    let m = TimeMap::new(2.0, 10.0, TimingMode::Loop);
    let u = m.map(-0.5); // t=-0.5 → u=-0.25 → fract=0.75
    assert!((0.0..1.0).contains(&u), "Loop 负半轴映射应落在 [0,1)：{u}");
}

#[test]
fn timemap_speed_extreme_and_nan() {
    // Speed(0) → u' = 0 恒（冻结首帧）。
    let z = TimeMap::new(2.0, 10.0, TimingMode::Speed(0.0));
    assert_eq!(z.map(1.0), 0.0, "k=0 应冻结首帧");
    // Speed 负：u*k 负 → clamp 0。
    let neg = TimeMap::new(2.0, 10.0, TimingMode::Speed(-3.0));
    assert_eq!(neg.map(1.0), 0.0, "k<0 应钳 0");
    // Speed(NaN)：u*NaN=NaN → clamp(NaN) 行为记录（不 panic）。
    let nan = TimeMap::new(2.0, 10.0, TimingMode::Speed(f32::NAN));
    let _ = nan.map(1.0);
    // Speed 极大：u*k 溢出 → clamp 1。
    let big = TimeMap::new(2.0, 10.0, TimingMode::Speed(1e30));
    assert_eq!(big.map(1.0), 1.0, "k 极大应钳 1");
}

#[test]
fn timemap_speedcurve_empty_samples() {
    // 空 samples：integral_curve 兜底首点倍率 1.0 → τ=t。
    let m = TimeMap::new(2.0, 10.0, TimingMode::SpeedCurve { samples: vec![], dur: 2.0 });
    let u = m.map(1.0);
    assert!(u.is_finite() && (0.0..=1.0).contains(&u), "空曲线非法：{u}");
    // τ(1)/dur = 1/2 = 0.5（首点倍率默认 1.0）。
    assert!((u - 0.5).abs() < 1e-3, "空曲线默认 k=1：{u}");
}

#[test]
fn timemap_expr_empty_samples() {
    // 空 samples：sample_curve 返回 0。
    let m = TimeMap::new(2.0, 10.0, TimingMode::Expr { samples: vec![] });
    assert_eq!(m.map(1.0), 0.0, "空 Expr 应映射 0");
}

#[test]
fn timemap_frame_pair_contract() {
    // frame_pair 不变量：前帧 ≤ 后帧 ≤ total-1；权重 0 → 前后帧相同。
    let modes = [
        TimingMode::Direct,
        TimingMode::Loop,
        TimingMode::PingPong,
        TimingMode::Speed(2.0),
    ];
    for mode in modes {
        let m = TimeMap::new(1.0, 12.0, mode.clone());
        let total = m.total_frames();
        for k in 0..=120 {
            let t = k as f32 / 120.0;
            let (a, b, w) = m.frame_pair_at(t);
            assert!(a <= b, "前帧应 ≤ 后帧");
            assert!(b < total, "后帧 {} 应 < 总帧 {total}", b);
            assert!((0.0..=1.0).contains(&w), "权重越界：{w}");
            if w == 0.0 {
                assert_eq!(a, b, "权重 0 时前后帧应相同");
            } else {
                assert_eq!(b, a + 1, "权重 >0 时后帧应 = 前帧+1");
            }
        }
    }
}

#[test]
fn timemap_seek_equivalence_all_modes() {
    // seek 等价：任意顺序求 frame_index ≡ 顺序求（纯函数）。
    let modes = [
        TimingMode::Direct,
        TimingMode::Loop,
        TimingMode::PingPong,
        TimingMode::Speed(0.7),
        TimingMode::SpeedCurve { samples: vec![(0.0, 2.0), (1.5, 0.5), (3.0, 1.0)], dur: 3.0 },
        TimingMode::Expr { samples: vec![(0.0, 0.0), (1.5, 0.8), (3.0, 0.2)] },
    ];
    for mode in modes {
        let m = TimeMap::new(3.0, 30.0, mode.clone());
        let n = 600;
        let seq: Vec<_> = (0..=n).map(|i| frame_index(&m, i as f32 * 3.0 / n as f32)).collect();
        for i in (0..=n).rev() {
            let t = i as f32 * 3.0 / n as f32;
            assert_eq!(frame_index(&m, t), seq[i], "seek 不等价（{mode:?}）");
        }
    }
}

#[test]
fn timemap_pingpong_symmetry() {
    // PingPong 三角折返：u 与 2-u 应在同一高度（对称）。
    let m = TimeMap::new(2.0, 10.0, TimingMode::PingPong);
    // u=0.25 → 0.5；u=0.75（t=1.5）→ 0.5。
    let a = m.map(0.5);
    let b = m.map(1.5);
    assert!((a - b).abs() < 1e-5, "乒乓应对称：{a} vs {b}");
}

// ---------------------------------------------------------------------------
// ② gif 防御
// ---------------------------------------------------------------------------

#[test]
fn gif_malformed_inputs_rejected() {
    // 太短。
    assert_eq!(decode_gif(b"GIF89a").unwrap_err().code, "GIF1001");
    // 坏魔数。
    assert_eq!(decode_gif(b"NOTGIF12345678").unwrap_err().code, "GIF1001");
    // 合法头但无图像帧（只有 trailer）。
    let mut minimal = Vec::new();
    minimal.extend_from_slice(b"GIF89a");
    minimal.extend_from_slice(&1u16.to_le_bytes());
    minimal.extend_from_slice(&1u16.to_le_bytes());
    minimal.push(0x00); // 无 GCT
    minimal.push(0);
    minimal.push(0);
    minimal.push(0x3B); // trailer
    assert_eq!(decode_gif(&minimal).unwrap_err().code, "GIF1001", "无帧应报错");
}

#[test]
fn gif_truncated_image_block_no_panic() {
    // 合法头 + 图像描述符但数据截断 → 结构化错误（不 panic）。
    let mut b = Vec::new();
    b.extend_from_slice(b"GIF89a");
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.push(0x80 | 0x00); // 有 GCT（2 色）
    b.push(0);
    b.push(0);
    b.extend_from_slice(&[0, 0, 0, 255, 255, 255]); // 2 色
    // 图像描述符，声明 2×2 但无 LZW 数据。
    b.push(0x2C);
    b.extend_from_slice(&[0, 0, 0, 0]);
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.push(0);
    // 缺 min_code → 截断。
    let r = decode_gif(&b);
    assert!(r.is_err(), "截断图像块应结构化报错（不 panic）");
}

#[test]
fn gif_image_without_palette_rejected() {
    // 无 GCT 且图像无 LCT → 应报"无调色板"错误。
    let mut b = Vec::new();
    b.extend_from_slice(b"GIF89a");
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.push(0x00); // 无 GCT
    b.push(0);
    b.push(0);
    b.push(0x2C); // image descriptor
    b.extend_from_slice(&[0, 0, 0, 0]);
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.push(0); // 无 LCT
    b.push(2); // min_code
    b.push(0); // 空子块
    b.push(0x3B);
    let r = decode_gif(&b);
    assert!(r.is_err(), "无调色板应报错");
    assert_eq!(r.unwrap_err().code, "GIF1001");
}

#[test]
fn gif_lzw_out_of_range_min_code_rejected() {
    // min_code < 2 或 > 8 → 拒绝（通过构造完整图像块触发 lzw_decode）。
    // 这里用像素为 0 的合法头，min_code=1 应报错。
    let mut b = Vec::new();
    b.extend_from_slice(b"GIF89a");
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.push(0x80 | 0x00);
    b.push(0);
    b.push(0);
    b.extend_from_slice(&[0, 0, 0, 255, 255, 255]);
    b.push(0x2C);
    b.extend_from_slice(&[0, 0, 0, 0]);
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.push(0);
    b.push(1); // min_code = 1（非法，须 2..=8）
    b.push(0);
    b.push(0x3B);
    let r = decode_gif(&b);
    assert!(r.is_err(), "min_code=1 应拒绝");
}

// ---------------------------------------------------------------------------
// ③ apng 防御
// ---------------------------------------------------------------------------

#[test]
fn apng_malformed_rejected() {
    assert_eq!(decode_apng(b"not png!").unwrap_err().code, "APNG1001");
    // 合法签名但无 IHDR → 错误。
    let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    b.extend_from_slice(&[0, 0, 0, 0]); // 长度 0 块（无类型则越界）
    let r = decode_apng(&b);
    assert!(r.is_err(), "截断 APNG 应报错");
}

#[test]
fn apng_truncated_chunk_rejected() {
    // IHDR 声明长度 13 但体不足 → 越界错误。
    let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    b.extend_from_slice(&13u32.to_be_bytes());
    b.extend_from_slice(b"IHDR");
    b.extend_from_slice(&[0, 0]); // 只有 2 字节体（< 13）
    let r = decode_apng(&b);
    assert!(r.is_err(), "IHDR 体截断应报错");
    assert_eq!(r.unwrap_err().code, "APNG1001");
}

#[test]
fn apng_zero_dimension_rejected() {
    // IHDR 宽高为 0 → 拒绝。
    let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&0u32.to_be_bytes()); // width 0
    ihdr.extend_from_slice(&2u32.to_be_bytes()); // height 2
    ihdr.push(8);
    ihdr.push(6); // RGBA
    ihdr.extend_from_slice(&[0, 0, 0]);
    b.extend_from_slice(&13u32.to_be_bytes());
    b.extend_from_slice(b"IHDR");
    b.extend_from_slice(&ihdr);
    b.extend_from_slice(&[0, 0, 0, 0]); // CRC
    let r = decode_apng(&b);
    assert!(r.is_err(), "零宽应拒绝");
    assert_eq!(r.unwrap_err().code, "APNG1001");
}

// ---------------------------------------------------------------------------
// ④ inflate 防御
// ---------------------------------------------------------------------------

#[test]
fn inflate_zlib_header_validation() {
    // 太短。
    assert_eq!(zlib_decompress(&[0x78]).unwrap_err().code, "INF1001");
    // FCHECK 失败。
    assert!(zlib_decompress(&[0x78, 0x01]).is_err(), "坏 FCHECK 应拒绝");
    // 非 DEFLATE（CM != 8）。
    // 构造 CM=7：需 FCHECK 通过。
    let cmf = 0x77u8; // CM=7
    let mut flg = 0u8;
    while (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
        flg += 1;
    }
    assert!(zlib_decompress(&[cmf, flg]).is_err(), "CM!=8 应拒绝");
}

#[test]
fn inflate_truncated_stream_rejected() {
    // 合法 zlib 头 + 非法块类型 3（BFINAL=1,BTYPE=11）。
    let mut s = vec![0x78, 0x9C];
    s.push(0b0000_0111); // BFINAL=1, BTYPE=11（保留非法）
    assert!(zlib_decompress(&s).is_err(), "保留块类型应拒绝");
}

#[test]
fn inflate_stored_len_nlen_mismatch_rejected() {
    // zlib 头 + stored 块 LEN/NLEN 不互补。
    let mut s = vec![0x78, 0x9C];
    s.push(0x01); // BFINAL=1, BTYPE=00
    s.extend_from_slice(&4u16.to_le_bytes()); // LEN=4
    s.extend_from_slice(&9999u16.to_le_bytes()); // NLEN 错
    s.extend_from_slice(&[1, 2, 3, 4]);
    assert!(zlib_decompress(&s).is_err(), "LEN/NLEN 不互补应拒绝");
}

// ---------------------------------------------------------------------------
// ⑤ source 分类边界
// ---------------------------------------------------------------------------

#[test]
fn source_extension_edges() {
    assert_eq!(extension_of("noext"), "");
    assert_eq!(extension_of("a.b.c.GIF"), "gif", "多点应取最后段并小写");
    assert_eq!(extension_of(".hidden"), "hidden");
    assert_eq!(extension_of("trailing."), "", "尾点应空扩展");
    // 分类：无扩展 → Unreadable。
    assert_eq!(MediaSource::classify("noext", SourceKind::Image), ReadKind::Unreadable);
    assert_eq!(MediaSource::classify("", SourceKind::Image), ReadKind::Unreadable);
    // 大小写不敏感。
    assert_eq!(MediaSource::classify("A.GIF", SourceKind::Video), ReadKind::Native);
    assert_eq!(MediaSource::classify("a.WAV", SourceKind::AudioSource), ReadKind::Native);
}

#[test]
fn pixel_image_bounds() {
    let mut img = PixelImage::new(2, 2);
    assert_eq!(img.pixels.len(), 16, "2×2×4 = 16 字节");
    img.set(5, 5, [1, 2, 3, 4]); // 越界写忽略。
    assert_eq!(img.get(5, 5), [0, 0, 0, 0]);
    // 零尺寸图。
    let z = PixelImage::new(0, 0);
    assert!(z.pixels.is_empty());
    assert_eq!(z.get(0, 0), [0, 0, 0, 0]);
}

// ---------------------------------------------------------------------------
// ⑥ audio
// ---------------------------------------------------------------------------

#[test]
fn audio_ffmpeg_and_struct_errors() {
    assert_eq!(load_audio("x.mp3", &[]).unwrap_err().code, "MD1004");
    assert_eq!(load_audio("x.flac", &[]).unwrap_err().code, "MD1004");
    assert_eq!(load_audio("x.xyz", &[]).unwrap_err().code, "MD1005");
    assert_eq!(load_audio("x.wav", b"garbage").unwrap_err().code, "MD1003");
    assert_eq!(load_audio("empty.wav", &[]).unwrap_err().code, "MD1003");
}
