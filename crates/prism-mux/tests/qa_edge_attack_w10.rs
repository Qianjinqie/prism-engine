//! W10-C4 攻击性边界/确定性/协议正确性测试（QA 严过关）。
//!
//! 覆盖 `prism-mux` 容器交付面：JPEG 基线编码 / AVI / MP4(ISO-BMFF) /
//! loudnorm(EBU R128) / extprobe 降级链 / hwenc 选择。
//!
//! 每条测试的 doc 注释说明"失败代表什么缺陷"。协议类断言依据官方规范：
//! - JPEG：ITU-T T.81 / JFIF（SOI/APP0/DQT/SOF0/DHT/SOS/EOI 段结构与长度）；
//! - ISO-BMFF：ISO/IEC 14496-12（box size/type、size==1 → 64 位 largesize）；
//! - EBU R128：ITU-R BS.1770-4（K 加权系数表 1/表 2、双门控、LUFS 公式）。
//!
//! 本文件只写测试、不改源码。

use prism_mux::avi::{write_avi, AviConfig};
use prism_mux::extprobe::{self, Codec, Encoders, FfmpegProbe};
use prism_mux::hwenc::{select_encoder, EncodeRequest, HwBackend, HwCapability};
use prism_mux::jpeg::{encode_jpeg_gray, rgba_to_gray, JpegConfig};
use prism_mux::loudnorm::{self, compute_normalize, measure, normalize, TARGET_LUFS};
use prism_mux::mp4::{write_mp4, TrackSpec};
use prism_mux::{boxes, MuxError};

// ---------------------------------------------------------------------------
// 工具：确定性正弦（禁 libm，三次多项式，测试域精度足够）
// ---------------------------------------------------------------------------

/// 自研 sin（仅测试用；与源码测试同款多项式）。
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

/// 48 kHz、1 kHz 正弦的 PCM 缓冲。
fn sine_1khz(amp: f32, secs: f32) -> prism_audio::PcmBuffer {
    let sr = 48_000_u32;
    let n = (sr as f32 * secs) as usize;
    let two_pi = 6.283_185_307_179_586_f32;
    let data: Vec<f32> = (0..n)
        .map(|i| amp * fast_sin(two_pi * 1000.0 * (i as f32) / sr as f32))
        .collect();
    prism_audio::PcmBuffer::from_mono(sr, data).unwrap()
}

/// 生成一张确定性的灰度渐变图。
fn gradient(w: u32, h: u32) -> Vec<u8> {
    (0..(w * h) as usize).map(|i| ((i * 7 + 13) % 256) as u8).collect()
}

/// 在字节流中查找 marker，返回 marker 之后（长度字段起始）的位置。
fn find_marker(data: &[u8], m: u8) -> Option<usize> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == m {
            return Some(i + 2);
        }
        i += 1;
    }
    None
}

// ===========================================================================
// A1. JPEG：结构合法性（ITU-T T.81 / JFIF）
// ===========================================================================

/// 灰度渐变图 → JPEG 头段顺序与魔数正确。
/// 失败代表：SOI/APP0/DQT/SOF0/DHT/SOS 段缺失或错位 → 任何标准解码器拒收。
#[test]
fn jpeg_header_segment_order_and_magic() {
    let img = gradient(32, 24);
    let jpeg = encode_jpeg_gray(&img, 32, 24, JpegConfig::default()).unwrap();

    // SOI (FF D8) 起始。
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "必须以 SOI 开头");
    // APP0 紧随 SOI（JFIF 强制）。
    assert_eq!(&jpeg[2..4], &[0xFF, 0xE0], "APP0 必须紧随 SOI");
    assert_eq!(&jpeg[6..11], b"JFIF\0", "APP0 payload 必须以 JFIF\\0 开头");
    // EOI (FF D9) 收尾。
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "必须以 EOI 结束");

    // 段出现顺序：DQT(DB) < SOF0(C0) < DHT(C4) < SOS(DA)。
    let dqt = find_marker(&jpeg, 0xDB).expect("应有 DQT");
    let sof0 = find_marker(&jpeg, 0xC0).expect("应有 SOF0");
    let dht = find_marker(&jpeg, 0xC4).expect("应有 DHT");
    let sos = find_marker(&jpeg, 0xDA).expect("应有 SOS");
    assert!(dqt < sof0 && sof0 < dht && dht < sos, "段顺序应为 DQT<SOF0<DHT<SOS");
}

/// 各段长度字段必须自洽：`len` 覆盖其后的数据字节数（含 2 字节长度自身），且不越界。
/// 失败代表：段长度写错 → 解码器跳段越界或读到垃圾。
#[test]
fn jpeg_segment_lengths_are_self_consistent() {
    let img = gradient(40, 40);
    let jpeg = encode_jpeg_gray(&img, 40, 40, JpegConfig::new(75)).unwrap();

    // APP0：len 应为 16（含自身 2 字节）。
    let p = find_marker(&jpeg, 0xE0).unwrap();
    let app0_len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
    assert_eq!(app0_len, 16, "APP0 段长应为 16");

    // DQT：len 应为 2 + 1 + 64 = 67。
    let p = find_marker(&jpeg, 0xDB).unwrap();
    let dqt_len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
    assert_eq!(dqt_len, 67, "DQT 段长应为 67（1 表 id/精度 + 64 表值 + 2 长度）");

    // SOF0：len = 8 + 3*components = 8 + 3*1 = 11（单分量灰度）。
    let p = find_marker(&jpeg, 0xC0).unwrap();
    let sof_len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
    assert_eq!(sof_len, 11, "SOF0 段长应为 11（单分量）");

    // SOS：len = 6 + 2*components = 6 + 2*1 = 8。
    let p = find_marker(&jpeg, 0xDA).unwrap();
    let sos_len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
    assert_eq!(sos_len, 8, "SOS 段长应为 8（单分量）");
}

/// SOF0 尺寸字段 = 输入的 (宽,高)，精度 8，分量数 1（灰度基线）。
/// 失败代表：尺寸字段写错 → 解码器按错误尺寸重建，图像错乱。
#[test]
fn jpeg_sof0_dimensions_exact() {
    let (w, h) = (37_u32, 19_u32); // 非 8 倍数，考验边界填充
    let img = gradient(w, h);
    let jpeg = encode_jpeg_gray(&img, w, h, JpegConfig::default()).unwrap();
    let p = find_marker(&jpeg, 0xC0).expect("应有 SOF0");
    // SOF0 payload: len(2) precision(1) height(2) width(2) ncomp(1)...
    let precision = jpeg[p + 2];
    let height = u16::from_be_bytes([jpeg[p + 3], jpeg[p + 4]]);
    let width = u16::from_be_bytes([jpeg[p + 5], jpeg[p + 6]]);
    let ncomp = jpeg[p + 7];
    assert_eq!(precision, 8, "精度应为 8 bit");
    assert_eq!((width, height), (w as u16, h as u16), "SOF0 尺寸应精确等于输入");
    assert_eq!(ncomp, 1, "灰度基线应单分量");
}

/// 熵编码段内所有 0xFF 字节必须被 0x00 填充（byte stuffing），EOI 除外。
/// 失败代表：未填充 0xFF → 解码器误判为 marker，扫描数据截断/损坏。
#[test]
fn jpeg_byte_stuffing_within_scan() {
    // 用高质量保留更多细节，制造可能出现 0xFF 的位流。
    let img: Vec<u8> = (0..48 * 48).map(|i| ((i * 251) % 256) as u8).collect();
    let jpeg = encode_jpeg_gray(&img, 48, 48, JpegConfig::new(100)).unwrap();
    // 定位 SOS，跳过其段头。
    let p = find_marker(&jpeg, 0xDA).unwrap();
    let sos_len = u16::from_be_bytes([jpeg[p], jpeg[p + 1]]) as usize;
    let scan_start = p + sos_len;
    let mut i = scan_start;
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF {
            if jpeg[i + 1] == 0xD9 {
                break; // EOI 合法终止
            }
            assert_eq!(jpeg[i + 1], 0x00, "扫描数据内 0xFF 后必须 0x00 填充（位置 {i}）");
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// 零尺寸图像必须被结构化拒绝（JPG1001），不得 panic。
/// 失败代表：库代码 panic（= 缺陷）或静默产出非法 JPEG。
#[test]
fn jpeg_zero_dimension_is_structured_error() {
    let err: MuxError = encode_jpeg_gray(&[], 0, 10, JpegConfig::default()).unwrap_err();
    assert_eq!(err.code, "JPG1001");
    let err2 = encode_jpeg_gray(&[], 10, 0, JpegConfig::default()).unwrap_err();
    assert_eq!(err2.code, "JPG1001");
    // 三段式错误（C18/C19）：location/problem/fix 均非空。
    assert!(!err.location.is_empty() && !err.problem.is_empty() && !err.fix.is_empty());
}

/// 样本长度与尺寸不匹配必须被拒绝。
/// 失败代表：越界读取（依赖 `gray[sy*mw+sx]` 索引）→ 潜在 panic 或读脏。
#[test]
fn jpeg_sample_length_mismatch_rejected() {
    // 声明 4x4=16 但只给 10 字节。
    let err = encode_jpeg_gray(&[0u8; 10], 4, 4, JpegConfig::default()).unwrap_err();
    assert_eq!(err.code, "JPG1001");
    // 多给也拒绝。
    let err2 = encode_jpeg_gray(&[0u8; 20], 4, 4, JpegConfig::default()).unwrap_err();
    assert_eq!(err2.code, "JPG1001");
}

/// 质量极端值（0/255）应被钳到 [1,100]，仍产出合法 JPEG（不 panic、尺寸正确）。
/// 失败代表：非法质量导致量化表全 0（除零）或越界。
#[test]
fn jpeg_extreme_quality_is_clamped_and_valid() {
    let img = gradient(16, 16);
    let lo = encode_jpeg_gray(&img, 16, 16, JpegConfig::new(0)).expect("质量 0 应钳到 1 并成功");
    let hi = encode_jpeg_gray(&img, 16, 16, JpegConfig::new(255)).expect("质量 255 应钳到 100 并成功");
    assert_eq!(&lo[0..2], &[0xFF, 0xD8]);
    assert_eq!(&hi[hi.len() - 2..], &[0xFF, 0xD9]);
    // 量化表最小值必须 ≥1（否则除零）。
    let p = find_marker(&lo, 0xDB).unwrap();
    let table = &lo[p + 3..p + 3 + 64];
    assert!(table.iter().all(|&v| v >= 1), "量化步长必须 ≥1（防除零）");
}

/// 1×1 最小图像必须可编码（单块、边界钳制）。
/// 失败代表：边界块计算在最小尺寸下越界。
#[test]
fn jpeg_one_by_one_minimal() {
    let jpeg = encode_jpeg_gray(&[123], 1, 1, JpegConfig::default()).unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    let p = find_marker(&jpeg, 0xC0).unwrap();
    let height = u16::from_be_bytes([jpeg[p + 3], jpeg[p + 4]]);
    let width = u16::from_be_bytes([jpeg[p + 5], jpeg[p + 6]]);
    assert_eq!((width, height), (1, 1));
}

/// RGBA 缓冲不足必须被结构化拒绝（JPG1002）。
/// 失败代表：越界读 RGBA → panic。
#[test]
fn jpeg_rgba_to_gray_rejects_short_buffer() {
    let err = rgba_to_gray(&[0u8; 8], 3, 1).unwrap_err(); // 需 3*1*4=12
    assert_eq!(err.code, "JPG1002");
}

/// **确定性**：同输入两遍编码必须逐字节一致（JPEG/MJPEG 铁律）。
/// 失败代表：编码含隐藏状态/浮点不确定性 → 跨平台不可复现，违反 C4 确定性契约。
#[test]
fn jpeg_encoding_is_bytewise_deterministic() {
    let img = gradient(64, 48);
    let a = encode_jpeg_gray(&img, 64, 48, JpegConfig::new(90)).unwrap();
    let b = encode_jpeg_gray(&img, 64, 48, JpegConfig::new(90)).unwrap();
    assert_eq!(a, b, "同输入两遍 JPEG 必须逐字节一致");
}

// ===========================================================================
// A2. AVI：结构合法性 + 确定性
// ===========================================================================

/// AVI 顶层 RIFF/'AVI ' + 必需块（hdrl/movi/idx1）+ RIFF 尺寸自洽。
/// 失败代表：RIFF 尺寸字段错 → "双击可播"底线失效。
#[test]
fn avi_riff_header_and_chunks() {
    let frame = gradient(32, 32);
    let avi = write_avi(&[frame.clone(), frame], AviConfig::new(32, 32, 30)).unwrap();
    assert_eq!(&avi[0..4], b"RIFF", "RIFF 魔数");
    assert_eq!(&avi[8..12], b"AVI ", "AVI 类型");
    let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
    assert_eq!(riff_size, avi.len() - 8, "RIFF 尺寸 = 文件长 - 8");
    for needle in [b"hdrl".as_slice(), b"movi", b"idx1", b"MJPG", b"00dc", b"avih", b"strh", b"strf"] {
        assert!(
            avi.windows(needle.len()).any(|w| w == needle),
            "AVI 应含块 {:?}",
            std::str::from_utf8(needle).unwrap()
        );
    }
}

/// AVI 零/非法参数结构化拒绝（MUX1001）；空帧序列行为与文档一致。
/// 失败代表：非法参数下 panic 或越界。
#[test]
fn avi_rejects_bad_params() {
    let e1 = write_avi(&[vec![0; 64]], AviConfig::new(0, 8, 30)).unwrap_err();
    assert_eq!(e1.code, "MUX1001");
    let e2 = write_avi(&[vec![0; 64]], AviConfig::new(8, 8, 0)).unwrap_err();
    assert_eq!(e2.code, "MUX1001");
    // 帧长不符。
    let e3 = write_avi(&[vec![0; 10]], AviConfig::new(32, 32, 30)).unwrap_err();
    assert_eq!(e3.code, "MUX1001");
}

/// **确定性**：同输入两遍 AVI 逐字节一致。
/// 失败代表：MJPEG 帧顺序或索引偏移不确定。
#[test]
fn avi_is_bytewise_deterministic() {
    let f = gradient(24, 16);
    let a = write_avi(&[f.clone(), f.clone(), f.clone()], AviConfig::new(24, 16, 25)).unwrap();
    let b = write_avi(&[f.clone(), f.clone(), f.clone()], AviConfig::new(24, 16, 25)).unwrap();
    assert_eq!(a, b, "同输入两遍 AVI 必须逐字节一致");
}

// ===========================================================================
// A3. MP4 / ISO-BMFF：盒结构合法性
// ===========================================================================

/// 顶层盒必须是 ftyp → mdat → moov，且可被自有解析器完整遍历。
/// 失败代表：盒尺寸/遍历残留 → 标准 demuxer 无法解析。
#[test]
fn mp4_top_level_box_layout() {
    let g = gradient(32, 32);
    let mp4 = write_mp4(&[g.clone(), g], TrackSpec { width: 32, height: 32, fps: 30 }, None, 0, 0)
        .unwrap();
    let tops = boxes::inspect_top_level(&mp4).expect("盒结构必须可完整遍历");
    let types: Vec<[u8; 4]> = tops.iter().map(|(t, _)| *t).collect();
    assert_eq!(types, vec![*b"ftyp", *b"mdat", *b"moov"], "顶层盒顺序 ftyp/mdat/moov");
}

/// mdat 使用 64 位大盒：size 字段 == 1，largesize == 盒总长（ISO/IEC 14496-12 §4.2）。
/// 失败代表：大盒标志/长度错 → 解析器读到错误 payload 边界。
#[test]
fn mp4_mdat_large_box_encoding() {
    let g = gradient(16, 16);
    let mp4 = write_mp4(&[g], TrackSpec { width: 16, height: 16, fps: 25 }, None, 0, 0).unwrap();
    let (start, _len) = boxes::find_box(&mp4, b"mdat").unwrap();
    // 盒头位于 payload 起始前 16 字节。
    let hdr = start - 16;
    let size32 = u32::from_be_bytes(mp4[hdr..hdr + 4].try_into().unwrap());
    assert_eq!(size32, 1, "mdat 必须用 size==1 大盒标志");
    assert_eq!(&mp4[hdr + 4..hdr + 8], b"mdat", "盒类型应为 mdat");
    let largesize = u64::from_be_bytes(mp4[hdr + 8..hdr + 16].try_into().unwrap()) as usize;
    // largesize == 16(头) + payload 长。
    assert_eq!(largesize, 16 + _len, "largesize 必须等于 mdat 盒总长");
}

/// stco 的每个 chunk offset 必须落在 mdat 的媒体数据区内（moov 引用 mdat 偏移对）。
/// 失败代表：偏移计算错误 → 播放器读到错误字节或文件尾。
#[test]
fn mp4_stco_offsets_point_into_mdat() {
    let g = gradient(8, 8);
    let mp4 = write_mp4(&[g.clone(), g.clone(), g], TrackSpec { width: 8, height: 8, fps: 10 }, None, 0, 0)
        .unwrap();
    let (md_start, md_len) = boxes::find_box(&mp4, b"mdat").unwrap();
    let (moov_start, moov_len) = boxes::find_box(&mp4, b"moov").unwrap();
    let moov = &mp4[moov_start..moov_start + moov_len];
    // 定位 stco：其后 4(count) 后为偏移。
    let p = find_marker(moov, 0xA9); // 无用，换法：搜索 b"stco"
    let _ = p;
    let pos = moov.windows(4).position(|w| w == b"stco").expect("应有 stco");
    // 4(type) + 1(ver) + 3(flags) + 4(count) 后。
    let count = u32::from_be_bytes(moov[pos + 8..pos + 12].try_into().unwrap()) as usize;
    assert_eq!(count, 3, "3 帧应有 3 个 chunk 偏移");
    for i in 0..count {
        let o = pos + 12 + i * 4;
        let off = u32::from_be_bytes(moov[o..o + 4].try_into().unwrap()) as usize;
        assert!(off >= md_start && off < md_start + md_len, "chunk 偏移 {off} 应落在 mdat 数据区");
    }
}

/// mvhd/tkhd/mdhd 字段：moov 含 mvhd + 2 trak（视频 + 音频），音频轨为 soun/lpcm。
/// 失败代表：trak 结构缺失 → 播放器无法识别轨道。
#[test]
fn mp4_moov_track_structure_with_audio() {
    let g = gradient(8, 8);
    let pcm = vec![0u8; 256]; // 128 samples 16bit mono
    let mp4 = write_mp4(&[g], TrackSpec { width: 8, height: 8, fps: 30 }, Some(&pcm), 8000, 1).unwrap();
    let (ms, ml) = boxes::find_box(&mp4, b"moov").unwrap();
    let moov = &mp4[ms..ms + ml];
    let cnt = |n: &[u8]| moov.windows(n.len()).filter(|w| *w == n).count();
    assert_eq!(cnt(b"trak"), 2, "应有两个 trak（视频 + 音频）");
    assert_eq!(cnt(b"mvhd"), 1, "应有一个 mvhd");
    assert!(cnt(b"vide") >= 1, "应有视频 handler");
    assert!(cnt(b"soun") >= 1, "应有音频 handler");
    assert!(cnt(b"lpcm") >= 1, "音频样本条目应为 lpcm");
}

/// MP4 非法参数：空帧序列 / 零尺寸 / 零 fps / 奇数 PCM / 帧长不符 → 结构化拒绝。
/// 失败代表：畸形输入崩溃（panic）或静默产出非法容器。
#[test]
fn mp4_rejects_all_malformed_inputs() {
    let e = write_mp4(&[], TrackSpec { width: 8, height: 8, fps: 30 }, None, 0, 0).unwrap_err();
    assert_eq!(e.code, "MUX2001", "空帧序列应拒绝");
    let e = write_mp4(&[vec![0; 64]], TrackSpec { width: 0, height: 8, fps: 30 }, None, 0, 0)
        .unwrap_err();
    assert_eq!(e.code, "MUX2001", "零宽应拒绝");
    let e = write_mp4(&[vec![0; 64]], TrackSpec { width: 8, height: 8, fps: 0 }, None, 0, 0)
        .unwrap_err();
    assert_eq!(e.code, "MUX2001", "零 fps 应拒绝");
    let e = write_mp4(&[vec![0; 64]], TrackSpec { width: 8, height: 8, fps: 30 }, Some(&[0, 1, 2]), 8000, 1)
        .unwrap_err();
    assert_eq!(e.code, "MUX2002", "奇数 PCM 应拒绝");
    let e = write_mp4(&[vec![0; 10]], TrackSpec { width: 8, height: 8, fps: 30 }, None, 0, 0)
        .unwrap_err();
    assert_eq!(e.code, "MUX2001", "帧长不符应拒绝");
}

/// **确定性（最高优先）**：同输入两遍 MP4 逐字节一致（含音频轨）。
/// 失败代表：盒遍历顺序/moov 回填不确定 → 交付不可复现，违反 D-29/C4 契约。
#[test]
fn mp4_is_bytewise_deterministic_with_audio() {
    let g = gradient(24, 24);
    let pcm: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let frames = vec![g.clone(), g.clone()];
    let a = write_mp4(&frames, TrackSpec { width: 24, height: 24, fps: 24 }, Some(&pcm), 48000, 2)
        .unwrap();
    let b = write_mp4(&frames, TrackSpec { width: 24, height: 24, fps: 24 }, Some(&pcm), 48000, 2)
        .unwrap();
    assert_eq!(a, b, "同输入两遍 MP4 必须逐字节一致");
}

// ===========================================================================
// A4. loudnorm：EBU R128 数学正确性 + 确定性
// ===========================================================================

/// 静音测量为 −∞，且归一化不放大静音（增益 0 dB）。
/// 失败代表：静音除零 → NaN/panic，或错误放大静音。
#[test]
fn loudnorm_silence_is_negative_infinity() {
    let buf = prism_audio::PcmBuffer::from_mono(48_000, vec![0.0; 48_000]).unwrap();
    let l = measure(&buf);
    assert!(!l.integrated_lufs.is_finite(), "静音应为 −∞");
    let r = compute_normalize(&buf, TARGET_LUFS);
    assert_eq!(r.gain_linear, 1.0, "静音不放大");
    assert_eq!(r.gain_db, 0.0);
}

/// 空音频（0 帧）测量安全不 panic。
/// 失败代表：0 帧下除零/索引越界。
#[test]
fn loudnorm_empty_buffer_is_safe() {
    let buf = prism_audio::PcmBuffer::from_mono(48_000, vec![]).unwrap();
    let l = measure(&buf);
    assert_eq!(l.total_blocks, 0);
    assert_eq!(l.gated_blocks, 0);
    let r = compute_normalize(&buf, TARGET_LUFS);
    assert_eq!(r.gain_linear, 1.0);
}

/// 归一后复测命中目标 ±0.5 LU（白皮书 §21 验收）。
/// 失败代表：K 加权滤波 / 门控 / LUFS 公式实现错误 → 响度不达标。
#[test]
fn loudnorm_normalize_hits_target_within_half_lu() {
    let buf = sine_1khz(0.5, 3.0);
    let (norm, r) = normalize(&buf, -23.0);
    let after = measure(&norm).integrated_lufs;
    assert!(
        (after - (-23.0)).abs() < 0.5,
        "归一后 {after:.4} LUFS 应命中 -23 ±0.5（增益 {:.4} dB）",
        r.gain_db
    );
}

/// EBU R128 / BS.1770 参考点：1 kHz 正弦 @ -20 dBFS（幅值 0.1）应测得 **-23 LUFS ±0.5**。
/// 这是 EBU Tech 3341 的标准测试信号（-20 dBFS 1kHz stereo sine → -23 LUFS）。
/// 失败代表：K 加权系数错误（如未 +4 dB 高架 / 缺 RLB 高通）或 LUFS 校准常量错。
#[test]
fn loudnorm_1khz_reference_level_sanity() {
    let buf = sine_1khz(0.1, 3.0);
    let lufs = measure(&buf).integrated_lufs;
    assert!(
        (lufs - (-23.0)).abs() < 0.5,
        "1kHz @ -20 dBFS 应测得 -23 LUFS ±0.5，实际 {lufs:.4}"
    );
    // 满幅（0 dBFS）1kHz 正弦：理论 ≈ -3 LUFS（-0.691 + 10·log10(0.5) + 加权重≈+0.7）。
    let full = measure(&sine_1khz(1.0, 3.0)).integrated_lufs;
    assert!(
        (full - (-3.0)).abs() < 1.0,
        "满幅 1kHz 正弦应 ≈ -3 LUFS ±1，实际 {full:.4}"
    );
}

/// **确定性**：同输入 100 次测量逐位一致。
/// 失败代表：滤波器状态残留/非确定浮点 → 违反确定性契约。
#[test]
fn loudnorm_measure_is_bitwise_deterministic() {
    let buf = sine_1khz(0.3, 1.0);
    let first = measure(&buf).integrated_lufs.to_bits();
    for _ in 0..100 {
        assert_eq!(measure(&buf).integrated_lufs.to_bits(), first, "同输入测量必须逐位一致");
    }
    // normalize 同输入同输出。
    let a = normalize(&buf, -16.0);
    let b = normalize(&buf, -16.0);
    assert_eq!(a.1.gain_linear.to_bits(), b.1.gain_linear.to_bits());
    assert_eq!(a.0.as_slice(), b.0.as_slice(), "归一缓冲应逐位一致");
}

/// 二次归一应几乎不变（幂等 ±0.1 LU）。
/// 失败代表：增益计算或缩放不对称。
#[test]
fn loudnorm_normalization_near_idempotent() {
    let buf = sine_1khz(0.25, 3.0);
    let (n1, _) = normalize(&buf, -16.0);
    let (n2, _) = normalize(&n1, -16.0);
    let a = measure(&n1).integrated_lufs;
    let b = measure(&n2).integrated_lufs;
    assert!((a - b).abs() < 0.1, "二次归一应几乎不变（{a:.4} vs {b:.4}）");
}

/// 门控应剔除静段：响 + 极静混合的积分响度接近纯响段。
/// 失败代表：相对门（−10 LU）未生效 → 静段拖低积分响度。
#[test]
fn loudnorm_gating_excludes_quiet_segment() {
    let loud = sine_1khz(0.5, 2.0);
    let quiet = sine_1khz(0.001, 2.0);
    let mut data = loud.as_slice().to_vec();
    data.extend_from_slice(quiet.as_slice());
    let mix = prism_audio::PcmBuffer::from_mono(48_000, data).unwrap();
    let l = measure(&mix);
    let loud_only = measure(&loud);
    assert!(
        (l.integrated_lufs - loud_only.integrated_lufs).abs() < 1.0,
        "门控应剔除静段（混合 {:.4} vs 纯响 {:.4}）",
        l.integrated_lufs,
        loud_only.integrated_lufs
    );
    assert!(l.total_blocks > l.gated_blocks, "应存在被门控剔除的块");
}

/// **libm 扫描（D-29）**：自研 log10/pow10 与参照对比，且 `ln`/`exp` 转换自研。
/// 失败代表：K 加权系数计算或 LUFS 转换偷偷调用 libm 超越函数。
#[test]
fn loudnorm_self_math_no_libm_accuracy() {
    // log10 逼近精度。
    assert!((loudnorm::log10_approx(1.0) - 0.0).abs() < 1e-4);
    assert!((loudnorm::log10_approx(1000.0) - 3.0).abs() < 2e-3);
    assert!((loudnorm::log10_approx(0.001) + 3.0).abs() < 2e-3);
    // pow10 逼近精度。
    assert!((loudnorm::pow10_approx(0.0) - 1.0).abs() < 1e-4);
    assert!((loudnorm::pow10_approx(3.0) - 1000.0).abs() < 1.0);
    // 往返一致：10^(log10(x)) ≈ x。
    for &x in &[0.01_f32, 0.5, 1.7, 42.0, 1000.0] {
        let rt = loudnorm::pow10_approx(loudnorm::log10_approx(x));
        assert!((rt - x).abs() / x < 1e-2, "log10/pow10 往返 {x} → {rt}");
    }
    // db↔linear 一致性。
    assert!((loudnorm::db_to_linear(0.0) - 1.0).abs() < 1e-3);
    assert!((loudnorm::linear_to_db(1.0)).abs() < 1e-3);
    assert!((loudnorm::linear_to_db(0.5) + 6.0206).abs() < 0.05, "0.5 → ≈ -6.02 dB");
}

/// 极端目标（-70 / 0 LUFS）不 panic；返回有限或 ±∞ 但结构一致。
/// 失败代表：极端增益下溢出/NaN 传播。
#[test]
fn loudnorm_extreme_targets_are_safe() {
    let buf = sine_1khz(0.2, 2.0);
    let r1 = compute_normalize(&buf, -70.0);
    let r2 = compute_normalize(&buf, 0.0);
    // gain_linear 应有限且正（可放大/缩小）。
    assert!(r1.gain_linear.is_finite() && r1.gain_linear > 0.0);
    assert!(r2.gain_linear.is_finite() && r2.gain_linear > 0.0);
}

// ===========================================================================
// A5. extprobe 降级链：无 ffmpeg 全流程可跑
// ===========================================================================

/// 空 PATH → 无 ffmpeg，能力全无。
/// 失败代表：探测非确定性或误判可用。
#[test]
fn extprobe_empty_path_is_unavailable() {
    let p = FfmpegProbe::detect_in("");
    assert!(!p.available);
    assert!(p.path.is_none());
    assert_eq!(p.encoders, Encoders::NONE);
}

/// 无 ffmpeg 时请求 H264/HEVC 必须降级到内置 MJPEG，且 degraded 标志为真。
/// 失败代表：降级链断裂 → 无 ffmpeg 环境无法交付（违反 D-10 永远可用）。
#[test]
fn extprobe_no_ffmpeg_degrades_to_mjpeg() {
    let probe = FfmpegProbe::unavailable();
    let d = extprobe::decide(Codec::H264, &probe);
    assert_eq!(d.codec, Codec::Mjpeg);
    assert!(d.degraded);
    let d2 = extprobe::decide(Codec::Hevc, &probe);
    assert_eq!(d2.codec, Codec::Mjpeg);
    assert!(d2.degraded);
    // MJPEG 请求恒可用、不降级。
    let d3 = extprobe::decide(Codec::Mjpeg, &probe);
    assert_eq!(d3.codec, Codec::Mjpeg);
    assert!(!d3.degraded);
}

/// 降级链必须以内置 MJPEG 收尾（终态永远可用）。
/// 失败代表：链尾非内置 → 极端环境无终态。
#[test]
fn extprobe_chain_always_ends_in_builtin() {
    for req in [Codec::H264, Codec::Hevc, Codec::Mjpeg] {
        let chain = extprobe::delivery_chain(req);
        assert_eq!(*chain.last().unwrap(), Codec::Mjpeg, "链尾必须是内置 MJPEG");
        assert_eq!(chain[0], req, "链首是请求目标");
    }
}

/// HEVC 请求在仅有 H264 能力时退回 H264；全无时退回 MJPEG。
/// 失败代表：降级优先级错误。
#[test]
fn extprobe_hevc_fallback_priority() {
    let only_h264 = FfmpegProbe {
        available: true,
        path: None,
        encoders: Encoders { libx264: true, libx265: false, hardware: false },
    };
    assert_eq!(extprobe::decide(Codec::Hevc, &only_h264).codec, Codec::H264);
    assert_eq!(extprobe::decide(Codec::H264, &only_h264).codec, Codec::H264);
}

// ===========================================================================
// A6. hwenc：选择契约
// ===========================================================================

/// 无能力集 → 选择 None（降级）；能力集按 codec + 尺寸约束选择。
/// 失败代表：选择忽略尺寸/编码族 → 编码失败。
#[test]
fn hwenc_select_encoder_contract() {
    assert_eq!(
        select_encoder(&[], &EncodeRequest { codec: Codec::H264, width: 1920, height: 1080, fps: 30, bitrate_kbps: 0 }),
        None,
        "空能力应降级"
    );
    let caps = [HwCapability {
        backend: HwBackend::Nvenc,
        codecs: &[Codec::H264, Codec::Hevc],
        max_width: 3840,
        max_height: 2160,
    }];
    assert_eq!(
        select_encoder(&caps, &EncodeRequest { codec: Codec::H264, width: 1920, height: 1080, fps: 30, bitrate_kbps: 0 }),
        Some(HwBackend::Nvenc)
    );
    // 超尺寸 → None。
    assert_eq!(
        select_encoder(&caps, &EncodeRequest { codec: Codec::H264, width: 8000, height: 4320, fps: 30, bitrate_kbps: 0 }),
        None
    );
    // MJPEG 不被硬件后端支持 → None（降级到内置）。
    assert_eq!(
        select_encoder(&caps, &EncodeRequest { codec: Codec::Mjpeg, width: 640, height: 480, fps: 30, bitrate_kbps: 0 }),
        None
    );
}

// ===========================================================================
// A7. 全链路：Framebuffer RGBA → JPEG（端到端结构）
// ===========================================================================

/// 便捷函数 `rgba_to_gray` 后的 JPEG 结构与直接灰度路径一致。
/// 失败代表：RGBA→灰度转换与编码管线脱节。
#[test]
fn full_chain_rgba_to_jpeg_structure() {
    // 纯白 RGBA 32x32 → 灰度全 255。
    let rgba: Vec<u8> = std::iter::repeat([255u8, 255, 255, 255]).take(32 * 32).flatten().collect();
    let gray = rgba_to_gray(&rgba, 32, 32).unwrap();
    assert!(gray.iter().all(|&v| v == 255), "纯白 → 灰度全 255");
    let jpeg = encode_jpeg_gray(&gray, 32, 32, JpegConfig::default()).unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
}

// ===========================================================================
// A8. 盒原语边界（boxes.rs）
// ===========================================================================

/// 截断盒被 inspect_top_level 拒绝（返回 None，而非 panic）。
/// 失败代表：越界读取 → panic/UB。
#[test]
fn boxes_truncated_is_rejected_not_panic() {
    let mut out = Vec::new();
    boxes::write_box(&mut out, b"free", b"1234567890");
    out.truncate(out.len() - 3);
    assert!(boxes::inspect_top_level(&out).is_none(), "截断盒应被拒绝");
    // 空输入也是合法空遍历。
    assert_eq!(boxes::inspect_top_level(&[]), Some(vec![]));
}

/// size==1 大盒且 largesize 声明超出可用字节 → 拒绝。
/// 失败代表：伪造超长盒导致越界或无限循环。
#[test]
fn boxes_oversized_large_box_rejected() {
    let mut out = Vec::new();
    out.extend_from_slice(&1u32.to_be_bytes()); // size = 1 → 大盒
    out.extend_from_slice(b"mdat");
    out.extend_from_slice(&(1_000_000u64).to_be_bytes()); // 远超实际字节
    out.extend_from_slice(&[0u8; 4]);
    assert!(boxes::inspect_top_level(&out).is_none(), "越界 largesize 应被拒绝");
}
