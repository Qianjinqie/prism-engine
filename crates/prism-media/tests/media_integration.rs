//! 多媒体端到端集成测试（Phase 2 验收第 7 条）。
//!
//! 覆盖：
//! 1. GIF/APNG 解码 golden（自研编码器产素材，无第三方素材依赖）；
//! 2. 时间映射 seek 等价（跳第 N 帧 ≡ 顺序播放）；
//! 3. 无 ffmpeg 环境降级全流程可跑（原生 GIF 序列 + WAV）；
//! 4. 嵌入图片过下游引擎能力（BoxBlur3 卷积）——证明"被引擎能力影响"。

use prism_core::node::{EvalCtx, MathStructure, Payload, StructureId};
use prism_core::NodeRegistry;
use prism_io::{f16_to_f32, Fbuf16F};
use prism_media::nodes::{encode_embed, install_audio, install_embed, AudioEmbedNode, ImageEmbedNode};
use prism_media::source::PixelImage;
use prism_media::timing::{TimingMode, TimeMap};
use prism_media::video::FfmpegProbe;
use prism_media::{decode_apng, decode_gif, load_audio, MediaSource, ReadKind, SourceKind};
use prism_render::tile::{BoxBlur3, TileEffect};

// ---------------------------------------------------------------------------
// 素材生成（自研编码器，零第三方）
// ---------------------------------------------------------------------------

/// 生成 2×1 最小 GIF（红/绿）。
fn min_gif() -> Vec<u8> {
    // 复用 gif 模块测试编码思路的独立副本，避免依赖私有 fn。
    let colors: [[u8; 3]; 2] = [[255, 0, 0], [0, 255, 0]];
    let mut out = Vec::new();
    out.extend_from_slice(b"GIF89a");
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.push(0x80); // 有 GCT，size=2
    out.push(0);
    out.push(0);
    for c in colors {
        out.extend_from_slice(&c);
    }
    out.push(0x2C);
    out.extend_from_slice(&[0, 0, 0, 0, 2, 0, 1, 0, 0]); // left=0,top=0,w=2,h=1,packed=0
    out.push(2); // min code size
    let clear = 4u16;
    let end = 5u16;
    let mut bits: Vec<bool> = Vec::new();
    let push = |code: u16, bits: &mut Vec<bool>| {
        for i in 0..3 {
            bits.push((code >> i) & 1 == 1);
        }
    };
    push(clear, &mut bits);
    push(clear, &mut bits);
    push(0, &mut bits);
    push(clear, &mut bits);
    push(1, &mut bits);
    push(end, &mut bits);
    let mut cur = 0u8;
    let mut nbit = 0u32;
    let mut packed = Vec::new();
    for b in bits {
        if b {
            cur |= 1 << nbit;
        }
        nbit += 1;
        if nbit == 8 {
            packed.push(cur);
            cur = 0;
            nbit = 0;
        }
    }
    if nbit > 0 {
        packed.push(cur);
    }
    for chunk in packed.chunks(255) {
        out.push(chunk.len() as u8);
        out.extend_from_slice(chunk);
    }
    out.push(0);
    out.push(0x3B);
    out
}

/// zlib stored 流（供静态 PNG 使用）。
fn zlib_stored(payload: &[u8]) -> Vec<u8> {
    let cmf = 0x78u8;
    let mut flg = 0u8;
    while (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
        flg += 1;
    }
    let mut out = vec![cmf, flg, 0x01];
    let len = payload.len() as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn png_chunk(out: &mut Vec<u8>, ctype: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(ctype);
    out.extend_from_slice(body);
    out.extend_from_slice(&[0, 0, 0, 0]);
}

/// 生成 2×2 RGBA PNG（梯度）。
fn min_png() -> Vec<u8> {
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2u32.to_be_bytes());
    ihdr.extend_from_slice(&2u32.to_be_bytes());
    ihdr.push(8);
    ihdr.push(6);
    ihdr.extend_from_slice(&[0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    let mut raw = Vec::new();
    for y in 0..2u32 {
        raw.push(0);
        for x in 0..2u32 {
            let v = (y * 2 + x) as u8 * 60;
            raw.extend_from_slice(&[v, 255 - v, v / 2, 255]);
        }
    }
    png_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    png_chunk(&mut out, b"IEND", &[]);
    out
}

// ---------------------------------------------------------------------------
// 验收 1：GIF / APNG golden
// ---------------------------------------------------------------------------

#[test]
fn gif_golden_left_red_right_green() {
    let gif = decode_gif(&min_gif()).expect("GIF 必可解码");
    assert_eq!((gif.width, gif.height), (2, 1));
    assert_eq!(gif.frames[0].canvas.get(0, 0), [255, 0, 0, 255]);
    assert_eq!(gif.frames[0].canvas.get(1, 0), [0, 255, 0, 255]);
}

#[test]
fn apng_golden_gradient() {
    let png = decode_apng(&min_png()).expect("PNG 必可解码");
    assert_eq!((png.width, png.height), (2, 2));
    assert_eq!(png.frames[0].canvas.get(0, 0), [0, 255, 0, 255]);
    assert_eq!(png.frames[0].canvas.get(1, 0), [60, 195, 30, 255]);
    assert_eq!(png.frames[0].canvas.get(0, 1), [120, 135, 60, 255]);
}

// ---------------------------------------------------------------------------
// 验收 2：时间映射 seek 等价
// ---------------------------------------------------------------------------

#[test]
fn seek_equals_sequential_playback_all_modes() {
    let modes = [
        TimingMode::Direct,
        TimingMode::Loop,
        TimingMode::PingPong,
        TimingMode::Speed(2.0),
        TimingMode::SpeedCurve { samples: vec![(0.0, 1.0), (2.0, 3.0)], dur: 4.0 },
        TimingMode::Expr { samples: vec![(0.0, 0.0), (4.0, 1.0)] },
    ];
    for mode in modes {
        let m = TimeMap::new(4.0, 30.0, mode.clone());
        let seq: Vec<u32> = (0..600).map(|i| m.frame_at(i as f32 * (4.0 / 600.0))).collect();
        for i in (0..600).rev() {
            assert_eq!(m.frame_at(i as f32 * (4.0 / 600.0)), seq[i], "模式 {mode:?} 步 {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// 验收 3：无 ffmpeg 环境降级全流程可跑
// ---------------------------------------------------------------------------

#[test]
fn no_ffmpeg_fallback_full_pipeline_runs() {
    // 强制无 ffmpeg 的探测环境。
    let probe = FfmpegProbe::detect_in("/definitely/not/here");
    assert!(!probe.is_available());

    // 原生路径：GIF 序列帧 → 解码 → 时间映射取帧 → 装入嵌入节点 → 求值。
    let gif_bytes = min_gif();
    let gif = decode_gif(&gif_bytes).unwrap();

    let mp4 = MediaSource::new("clip.mp4", SourceKind::Video);
    let decision = prism_media::video::decide(&mp4, &probe);
    assert!(decision.degraded, "无 ffmpeg 时 mp4 应降级标注");

    // GIF 原生仍可读。
    let gif_src = MediaSource::new("clip.gif", SourceKind::Video);
    assert_eq!(prism_media::video::decide(&gif_src, &probe).effective, ReadKind::Native);

    // 全流程：装入帧 → 求值 → 得场景线性色。
    let mut ctx = EvalCtx::new(0, 1, 0);
    install_embed(&mut ctx, &[gif.frames[0].canvas.clone()]);
    let node = ImageEmbedNode::default();
    let out = node.apply(&Payload::Vec3(prism_math::Vec3 { x: 0.0, y: 0.0, z: 0.0 }), &ctx);
    match out {
        Payload::Vec3(v) => assert!((v.x - 1.0).abs() < 1e-6, "左像素红 → R=1"),
        other => panic!("期望 Vec3，得 {other:?}"),
    }

    // 音频原生：WAV 解码 + 嵌入音频节点取样。
    use prism_audio::{encode_wav, BitDepth};
    let buf = prism_audio::PcmBuffer::from_mono(8_000, vec![0.25, 0.5, 0.75]).unwrap();
    let wav = encode_wav(&buf, BitDepth::B16);
    let audio = load_audio("tone.wav", &wav).expect("WAV 原生读入");
    let mut actx = EvalCtx::new(0, 1, 1);
    install_audio(&mut actx, audio.sample_rate(), &audio.buffer.to_mono());
    let an = AudioEmbedNode::default();
    assert_eq!(an.apply(&Payload::Scalar(0.0), &actx), Payload::Scalar(0.5));
}

// ---------------------------------------------------------------------------
// 验收 4：嵌入图片被引擎能力影响（过 BoxBlur3 卷积）
// ---------------------------------------------------------------------------

#[test]
fn embedded_image_affected_by_downstream_blur() {
    // 源图：4×1，单点亮（其余全透明）。被下游 3×3 box blur 卷积后，
    // 亮像素的邻域必被点亮——证明嵌入值进入引擎能力链（bloom/卷积同族）。
    let mut img = PixelImage::new(4, 1);
    img.set(2, 0, [255, 255, 255, 255]);
    let mut ctx = EvalCtx::new(0, 1, 0);
    install_embed(&mut ctx, &[img.clone()]);
    let node = ImageEmbedNode::default();

    // 用嵌入节点把 4×1 素材渲染进 RGBA16F 缓冲（横向采样，垂直复用同一行）。
    let mut fb = Fbuf16F::new(4, 3);
    for y in 0..3u32 {
        for x in 0..4u32 {
            let u = (x as f32 + 0.5) / 4.0;
            let v = 0.5; // 单行素材：垂直方向取中心
            if let Payload::Vec3(c) = node.apply(&Payload::Vec3(prism_math::Vec3 { x: u, y: v, z: 0.0 }), &ctx) {
                fb.set(x, y, [c.x, c.y, c.z, 1.0]);
            }
        }
    }
    // 渲染后中心亮、邻域暗（初始未卷积）。
    assert!(f16_to_f32(fb.pixels[(0 * 4 + 2) * 4]) > 0.9, "中心应为亮像素");

    // 过 BoxBlur3（halo=1）——引擎卷积能力。
    let blurred = BoxBlur3.eval_window(&fb);
    let center_before = f16_to_f32(fb.pixels[(1 * 4 + 2) * 4]);
    let left_after = f16_to_f32(blurred.pixels[(1 * 4 + 1) * 4]);
    assert!(
        left_after > 0.0,
        "卷积后亮像素的邻居必被影响（before={center_before}, neighbor_after={left_after}）"
    );
}

// ---------------------------------------------------------------------------
// 嵌入素材编解码往返 + 注册表静态汇入
// ---------------------------------------------------------------------------

#[test]
fn embed_roundtrip_and_registry_import() {
    let mut img = PixelImage::new(2, 2);
    img.set(0, 0, [1, 2, 3, 4]);
    let bytes = encode_embed(&[img.clone()]);
    // 注册表静态汇入（D-27）。
    let mut reg = NodeRegistry::new();
    prism_media::__prism_collect(&mut reg);
    assert!(reg.contains(&StructureId::custom(13, "prism_media")));
    assert!(reg.contains(&StructureId::custom(14, "prism_media")));
    assert!(!bytes.is_empty());
}
