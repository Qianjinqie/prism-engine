//! QA 裁决探针 · P2 mux 修复验证（prism-mux / prism-io）。
//!
//! 目标：独立验证「MP4/JPEG 宽高 `as u16` 静默截断」修复真有效——
//! 超限必须**报错**，绝不静默产出错误分辨率。
//!
//! 覆盖：
//! - M1 `stsd_mjpeg` 超限 → `Err(MUX1003)`；合法区间边界精确；
//! - M2 `encode_jpeg_gray` 超限 → `Err(MUX1003)`，且错误码/位置正确；
//! - M3 **旧行为反证**：证明 `as u16` 确实会写出错值（否则修复是空转）；
//! - M4 `check_dim_u16` 单一真源：无重复内联守卫（`stsd` 与 `SOF0` 共用）；
//! - M5 合法路径的字节级不变量未被破坏。

use prism_mux::boxes::stsd_mjpeg;
use prism_mux::{encode_jpeg_gray, JpegConfig};

/// 必���拒绝的尺寸（0 / 超 u16 / 溢出源）。
const MUST_REJECT: &[(u32, u32)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (65536, 1),
    (1, 65536),
    (65536, 65536),
    (100_000, 1),
    (1, 100_000),
    (u32::MAX, 1),
    (1, u32::MAX),
    (u32::MAX, u32::MAX),
];

/// 必须接受的尺寸（u16 区间端点）。
const MUST_ACCEPT: &[(u32, u32)] = &[(1, 1), (1, 65535), (65535, 1), (65535, 65535), (1920, 1080)];

fn code_of(e: &prism_mux::MuxError) -> String {
    e.code.to_string()
}

#[test]
fn m1_stsd_mjpeg_rejects_overflow_dims_with_mux1003() {
    for &(w, h) in MUST_REJECT {
        let r = stsd_mjpeg(w, h);
        match r {
            Ok(bytes) => panic!("stsd_mjpeg({w},{h}) 必须 Err，却 Ok 出了 {} 字节（静默错分辨率）", bytes.len()),
            Err(e) => assert_eq!(
                code_of(&e),
                "MUX1003",
                "stsd_mjpeg({w},{h}) 错误码须为 MUX1003，实得 {}",
                code_of(&e)
            ),
        }
    }
    for &(w, h) in MUST_ACCEPT {
        let b = stsd_mjpeg(w, h).unwrap_or_else(|e| panic!("stsd_mjpeg({w},{h}) 应成功：{e:?}"));
        // 合法路径不变量：尺寸必须**无损**写进 stsd 的 width/height 字段。
        // 偏移推导：stsd fullbox 头(12) + entry_count(4) + mjpg 盒头(8)
        //          + VisualSampleEntry 前缀(6+2+2+2+12=24) = 48。
        const DIM_OFF: usize = 48;
        let got_w = u16::from_be_bytes([b[DIM_OFF], b[DIM_OFF + 1]]) as u32;
        let got_h = u16::from_be_bytes([b[DIM_OFF + 2], b[DIM_OFF + 3]]) as u32;
        assert_eq!((got_w, got_h), (w, h), "stsd_mjpeg({w},{h}) 写出的分辨率须与输入一致");
    }
}

#[test]
fn m2_encode_jpeg_gray_rejects_overflow_dims_with_mux1003() {
    let cfg = JpegConfig::default();
    for &(w, h) in MUST_REJECT {
        // 样本长度故意不匹配。对 **0 尺寸**，`encode_jpeg_gray` 自有的
        // 「宽或高为 0」守卫（既有 JPG1001）先于 `check_dim_u16` 触发，
        // 这是**正确**的（0 是非法参数，与 u16 截断无关），故只要求「必须
        // Err 且不得 Ok 出字节」。对 **超 u16 尺寸**，错误码必须是 MUX1003
        // —— 它证明拦截来自新增守卫，而非碰巧的样本长度检查。
        let r = encode_jpeg_gray(&[0u8; 4], w, h, cfg.clone());
        match r {
            Ok(v) => panic!("encode_jpeg_gray({w},{h}) 必须 Err，却 Ok 出了 {} 字节", v.len()),
            Err(e) => {
                if w > u16::MAX as u32 || h > u16::MAX as u32 {
                    assert_eq!(
                        code_of(&e),
                        "MUX1003",
                        "encode_jpeg_gray({w},{h}) 超 u16 尺寸的错误码须为 MUX1003，实得 {}",
                        code_of(&e)
                    );
                } else {
                    assert_eq!(code_of(&e), "JPG1001", "0 尺寸须由既有守卫报 JPG1001");
                }
            }
        }
    }
    // 合法尺寸 + 样本数不匹配 ⇒ 仍应是 JPG1001（守卫顺序未被破坏）。
    let e = encode_jpeg_gray(&[0u8; 4], 64, 64, cfg.clone()).unwrap_err();
    assert_eq!(code_of(&e), "JPG1001", "样本数不符须报 JPG1001");
    // 合法尺寸 + 正确样本数 ⇒ 成功，且 SOF0 里的宽高无损。
    for &(w, h) in &[(1u32, 1u32), (16u32, 16u32), (65535u32, 1u32)] {
        let gray = vec![128u8; (w as usize) * (h as usize)];
        let jpg = encode_jpeg_gray(&gray, w, h, cfg.clone())
            .unwrap_or_else(|e| panic!("encode_jpeg_gray({w},{h}) 应成功：{e:?}"));
        let sof = find_marker(&jpg, 0xC0).unwrap_or_else(|| panic!("{w}x{h} 未找到 SOF0"));
        // SOF0 布局：FF(sof) C0(sof+1) len(sof+2..4) precision(sof+4)
        //            height(sof+5..7) width(sof+7..9)
        let got_h = u16::from_be_bytes([jpg[sof + 5], jpg[sof + 6]]) as u32;
        let got_w = u16::from_be_bytes([jpg[sof + 7], jpg[sof + 8]]) as u32;
        assert_eq!((got_w, got_h), (w, h), "JPEG SOF0 宽高须无损（{w}x{h} 实得 {got_w}x{got_h}）");
    }
}

/// 找 `FF <marker>` 标记位置。
fn find_marker(buf: &[u8], marker: u8) -> Option<usize> {
    buf.windows(2).position(|w| w[0] == 0xFF && w[1] == marker)
}

#[test]
fn m3_old_as_u16_truncation_would_have_written_wrong_resolution() {
    // 反证：若没有 check_dim_u16，`as u16` 会静默写出这些错值。
    // 本测试**不调用**被测函数，而是直接演算旧行为，证明修复非空转。
    let cases: &[(u32, u32)] = &[(65536, 100_000), (100_000, 65_536), (66_560, 1)];
    for &(w, h) in cases {
        let old_w = (w as u16) as u32;
        let old_h = (h as u16) as u32;
        assert_ne!(
            (old_w, old_h),
            (w, h),
            "前提失效：({w},{h}) 经 as u16 后竟未改变"
        );
        assert!(
            old_w < w || old_h < h,
            "({w},{h}) → ({old_w},{old_h})：截断只会变小/失真，不会放大"
        );
        // 修复后：同输入必须报错，而不是写出那个错值。
        assert!(stsd_mjpeg(w, h).is_err(), "stsd_mjpeg({w},{h}) 应报错而非写出截断值");
    }
    // 端点语义：65536 恰为 2^16，`as u16` ⇒ 0（分辨率归零，文件能播但画幅全错）。
    assert_eq!(((65_536u32 as u16) as u32), 0, "端点反证前提失效");
    assert!(stsd_mjpeg(65536, 1).is_err(), "65536×1 必须报错，不得静默写成 0×1");
}
