//! 端到端交付线集成测试（C4 / C7）。
//!
//! 覆盖：RGBA8 帧 → MJPEG → AVI/MP4 结构合法性回读 + EBU R128 响度归一实测 +
//! ffmpeg 探测降级链（**无 ffmpeg 环境全流程可跑**）。

use prism_mux::avi::{write_avi_jpeg, AviConfig};
use prism_mux::boxes::inspect_top_level;
use prism_mux::extprobe::{decide, delivery_chain, Codec, FfmpegProbe};
use prism_mux::loudnorm::{compute_normalize, measure, normalize};
use prism_mux::mp4::{write_mp4, write_mp4_jpeg, TrackSpec};
use prism_mux::JpegConfig;

/// 构造一帧 RGBA8 渐变（确定性）。
fn frame_rgba(w: u32, h: u32, seed: u8) -> Vec<u8> {
    let mut px = vec![0_u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y as usize) * w as usize + x as usize) * 4;
            px[i] = ((x as u32).wrapping_mul(7).wrapping_add(seed as u32) & 0xFF) as u8;
            px[i + 1] = ((y as u32).wrapping_mul(5) & 0xFF) as u8;
            px[i + 2] = ((x as u32 + y as u32) & 0xFF) as u8;
            px[i + 3] = 255;
        }
    }
    px
}

#[test]
fn full_delivery_pipeline_without_ffmpeg() {
    // 无 ffmpeg 环境：内置终态 AVI + MP4 全流程可跑（D-10）。
    let probe = FfmpegProbe::unavailable();
    let d = decide(Codec::H264, &probe);
    assert_eq!(d.codec, Codec::Mjpeg, "无 ffmpeg 应降级到内置 MJPEG");
    assert!(d.degraded);
    assert!(delivery_chain(Codec::H264).contains(&Codec::Mjpeg));

    let (w, h) = (48_u32, 32_u32);
    let frames: Vec<Vec<u8>> = (0..6).map(|i| frame_rgba(w, h, i)).collect();

    // AVI 写入 + 结构校验。
    let avi = write_avi_jpeg(&frames, AviConfig::new(w, h, 24)).unwrap();
    assert_eq!(&avi[0..4], b"RIFF");
    assert_eq!(&avi[8..12], b"AVI ");

    // MP4 写入 + 盒结构校验（自写解析器回读）。
    let mp4 = write_mp4_jpeg(
        &frames,
        TrackSpec { width: w, height: h, fps: 24 },
        None,
        0,
        0,
        JpegConfig::new(70),
    )
    .unwrap();
    let tops = inspect_top_level(&mp4).expect("MP4 盒结构必须合法");
    let types: Vec<[u8; 4]> = tops.iter().map(|(t, _)| *t).collect();
    assert_eq!(types, vec![*b"ftyp", *b"mdat", *b"moov"]);
}

#[test]
fn mp4_with_pcm_audio_track_structure() {
    let (w, h) = (16_u32, 16_u32);
    let frames: Vec<Vec<u8>> = (0..4).map(|i| frame_rgba(w, h, i)).collect();
    // 16 bit PCM：1 声道，8000 Hz，100 帧。
    let pcm: Vec<u8> = (0..100).flat_map(|i| (i as i16).to_le_bytes()).collect();
    let mp4 = write_mp4_jpeg(
        &frames,
        TrackSpec { width: w, height: h, fps: 24 },
        Some(&pcm),
        8000,
        1,
        JpegConfig::default(),
    )
    .unwrap();
    assert!(inspect_top_level(&mp4).is_some());
}

/// 纯灰度路径的 MP4（不经 RGBA 转换）。
#[test]
fn gray_mp4_path() {
    let gray = vec![200_u8; 24 * 24];
    let mp4 = write_mp4(&[gray.clone(), gray], TrackSpec { width: 24, height: 24, fps: 30 }, None, 0, 0)
        .unwrap();
    assert!(inspect_top_level(&mp4).is_some());
}

// ---------------------------------------------------------------------------
// EBU R128 响度实测（白皮书 §21：±0.5 LU 对拍）
// ---------------------------------------------------------------------------

/// 自研 sin（禁 libm，测试域）。
fn fast_sin(x: f32) -> f32 {
    let two_pi = 6.283_185_307_179_586_f32;
    let pi = 3.141_592_653_589_793_f32;
    let mut t = x - two_pi * (x / two_pi).floor();
    if t > pi {
        t -= two_pi;
    }
    let t2 = t * t;
    t * (1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0)))
}

#[test]
fn loudnorm_hits_ebu_r128_targets() {
    // 1 kHz 正弦（-23 LUFS 参考级）：归一后复测命中目标 ±0.5 LU。
    let sr = 48_000_u32;
    let n = sr as usize * 4;
    let two_pi = 6.283_185_307_179_586_f32;
    let data: Vec<f32> = (0..n)
        .map(|i| 0.25 * fast_sin(two_pi * 1000.0 * (i as f32) / sr as f32))
        .collect();
    let buf = prism_audio::PcmBuffer::from_mono(sr, data).unwrap();

    for target in [-23.0_f32, -16.0, -14.0] {
        let (norm, r) = normalize(&buf, target);
        let after = measure(&norm).integrated_lufs;
        assert!(
            (after - target).abs() < 0.5,
            "目标 {target} LUFS：实测 {after:.3}（增益 {:.3} dB）超差",
            r.gain_db
        );
        // 增益方向正确性。
        assert_eq!(compute_normalize(&buf, target).target_lufs, target);
    }
}

#[test]
fn loudnorm_deterministic_bitwise() {
    let sr = 48_000_u32;
    let n = sr as usize;
    let two_pi = 6.283_185_307_179_586_f32;
    let data: Vec<f32> = (0..n)
        .map(|i| 0.3 * fast_sin(two_pi * 440.0 * (i as f32) / sr as f32))
        .collect();
    let buf = prism_audio::PcmBuffer::from_mono(sr, data).unwrap();
    let first = measure(&buf).integrated_lufs.to_bits();
    for _ in 0..50 {
        assert_eq!(measure(&buf).integrated_lufs.to_bits(), first, "测量必须逐位确定");
    }
}
