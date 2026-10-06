//! P2 边界测试：MP4 `stsd` / JPEG `SOF` 宽高的 u16 截断（静默错数据）。
//!
//! 契约：格式字段为 u16（ISO/IEC 14496-12 §8.3.3.1 `VisualSampleEntry`、
//! ISO/IEC 10918-1 §B.2.2 `SOF0`），上限 65535。超限必须返回结构化错误，
//! **不得**静默写出错误分辨率（65536 → 0、100000 → 34464）。
//!
//! 零第三方依赖，仅 std + #[test]。

use prism_mux::boxes;
use prism_mux::jpeg::{encode_jpeg_gray, JpegConfig};
use prism_mux::mp4::{write_mp4, TrackSpec};

/// 从 `stsd` 字节流中解出 `mjpg` 条目里的 width/height（u16 BE）。
///
/// 布局：`stsd` 全盒头(8) + version/flags(4) + entry_count(4) + `mjpg` 头(8)
/// + reserved(6) + data_ref_index(2) + pre_defined(2) + reserved(2)
/// + pre_defined[3](12) → 偏移 48 与 50 处即 width/height。
fn stsd_dims(bytes: &[u8]) -> (u16, u16) {
    let w = u16::from_be_bytes([bytes[48], bytes[49]]);
    let h = u16::from_be_bytes([bytes[50], bytes[51]]);
    (w, h)
}

// ---------------------------------------------------------------------------
// 边界值：合法区间两端与越界值
// ---------------------------------------------------------------------------

#[test]
fn p2_stsd_accepts_1_and_65535_writes_exact_values() {
    for (w, h) in [(1_u32, 1_u32), (65535, 65535), (65535, 1), (1, 65535), (1920, 1080)] {
        let out = boxes::stsd_mjpeg(w, h).expect("合法分辨率应成功");
        let (dw, dh) = stsd_dims(&out);
        assert_eq!((dw as u32, dh as u32), (w, h), "{}×{} 写入的分辨率须逐值一致", w, h);
    }
}

#[test]
fn p2_stsd_rejects_over_u16_without_silent_truncation() {
    for (w, h) in [
        (65536_u32, 16_u32),
        (16, 65536),
        (100_000, 100_000),
        (u32::MAX, 1),
        (1, u32::MAX),
        (u32::MAX, u32::MAX),
    ] {
        let r = boxes::stsd_mjpeg(w, h);
        assert!(r.is_err(), "{}×{} 超 u16 上限，必须报错而非静默截断", w, h);
        let e = r.expect_err("应返回错误");
        assert_eq!(e.code, "MUX1003", "错误码须为 MUX1003，实际 {}", e.code);
        assert!(!e.problem.is_empty() && !e.fix.is_empty(), "错误须含事实与建议段");
    }
}

#[test]
fn p2_stsd_rejects_zero_dimension() {
    for (w, h) in [(0_u32, 16_u32), (16, 0), (0, 0)] {
        let e = boxes::stsd_mjpeg(w, h).expect_err("0 尺寸必须报错");
        assert_eq!(e.code, "MUX1003", "0 尺寸错误码须为 MUX1003");
    }
}

// ---------------------------------------------------------------------------
// 端到端：write_mp4 入口必须拒绝超限分辨率
// ---------------------------------------------------------------------------

#[test]
fn p2_write_mp4_rejects_over_u16_video_spec() {
    let g: Vec<u8> = vec![128_u8; 16 * 16];
    for (w, h) in [(65536_u32, 16_u32), (16, 65536), (100_000, 100_000)] {
        let e = write_mp4(&[g.clone()], TrackSpec { width: w, height: h, fps: 30 }, None, 0, 0)
            .expect_err("超限分辨率必须报错");
        assert_eq!(e.code, "MUX1003", "{}×{} 错误码须为 MUX1003，实际 {}", w, h, e.code);
    }
}

#[test]
fn p2_write_mp4_accepts_boundary_and_still_writes_correct_dims() {
    // 1×1：最小合法画幅，产出文件的 stsd 里宽高须为 1。
    let g1: Vec<u8> = vec![200_u8; 1];
    let mp4 = write_mp4(&[g1], TrackSpec { width: 1, height: 1, fps: 30 }, None, 0, 0)
        .expect("1×1 应成功");
    // 定位 `stsd` 盒起点：宽高字段距 `stsd` 盒首 48 字节（见 stsd_dims 的布局注释）。
    let idx: usize = mp4
        .windows(4)
        .position(|w| w == b"stsd")
        .expect("应含 stsd 盒");
    let (dw, dh) = stsd_dims(&mp4[idx - 4..]);
    assert_eq!((dw, dh), (1, 1), "1×1 画幅须原样写入 stsd");
}

// ---------------------------------------------------------------------------
// JPEG SOF0 同类点
// ---------------------------------------------------------------------------

#[test]
fn p2_jpeg_rejects_over_u16_dimension() {
    // 65535 上限本身合法，但样本量过大不便构造；此处只验超限被拒。
    for (w, h) in [(65536_u32, 1_u32), (1, 65536), (100_000, 2), (2, u32::MAX)] {
        let small: Vec<u8> = vec![0_u8; 4]; // 故意与尺寸不符：超限检查须先于长度检查
        let e = encode_jpeg_gray(&small, w, h, JpegConfig::default())
            .expect_err("超限尺寸必须报错");
        assert_eq!(e.code, "MUX1003", "{}×{} 错误码须为 MUX1003，实际 {}", w, h, e.code);
    }
}

#[test]
fn p2_jpeg_sof_writes_exact_dims_at_boundary() {
    for (w, h) in [(1_u32, 1_u32), (255, 255), (65535, 1)] {
        let g: Vec<u8> = vec![120_u8; (w * h) as usize];
        let out = encode_jpeg_gray(&g, w, h, JpegConfig::default()).expect("合法尺寸应成功");
        // SOF0 段：marker(2) + len(2) + precision(1) → 偏移 5/7 为 height/width。
        let sof: usize = out
            .windows(2)
            .position(|w| w == [0xFF, 0xC0])
            .expect("应含 SOF0 段");
        let sh = u16::from_be_bytes([out[sof + 5], out[sof + 6]]);
        let sw = u16::from_be_bytes([out[sof + 7], out[sof + 8]]);
        assert_eq!((sw as u32, sh as u32), (w, h), "{}×{} 的 SOF0 宽高须逐值一致", w, h);
    }
}

// ---------------------------------------------------------------------------
// 确定性：同输入两遍逐字节一致（含错误路径的一致性）
// ---------------------------------------------------------------------------

#[test]
fn p2_errors_are_deterministic_across_runs() {
    for _ in 0..3 {
        let a = boxes::stsd_mjpeg(100_000, 100_000).expect_err("应报错");
        let b = boxes::stsd_mjpeg(100_000, 100_000).expect_err("应报错");
        assert_eq!(a, b, "错误对象须逐字段一致（确定性铁律）");
    }
    let a = boxes::stsd_mjpeg(1920, 1080).expect("应成功");
    let b = boxes::stsd_mjpeg(1920, 1080).expect("应成功");
    assert_eq!(a, b, "同输入两遍 stsd 应逐字节一致");
}
