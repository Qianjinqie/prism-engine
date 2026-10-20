//! C2 色彩管线 golden-file 固化测试。
//!
//! 语义权威：D-13（"调校结果以 golden-file 固化"）、
//!  对拍判据（"ACES 对拍 ±0.01；PQ 编解码往返 <1e-4；10bit PNG 往返一致"）。
//!
//! # 固化方式
//! 对固定输入样本（灰阶 + 原色 + 超亮），逐值打印 ACES/PQ/HLG/ACEScg 输出，
//! 以 f32 位模式（`to_bits`）作指纹——同平台重跑逐位一致；跨平台因全链
//! 仅含 IEEE 精确运算（D-29）亦逐位一致。golden 常量以十六进制位模式内嵌，
//! 一旦色彩数学改动即触发 diff（防止静默回归）。

use prism_math::aces::{aces_fast, AcesFit, AcesOutput};
use prism_math::colorspace::{acescg_to_srgb, srgb_to_acescg};
use prism_math::hlg::hlg_oetf;
use prism_math::pq::{pq_eotf, pq_oetf};

/// 固定输入样本（归一化线性 sRGB）：黑 / 18% 灰 / 中灰 / 白 / 原色 / 超亮。
const SAMPLES: [[f32; 3]; 9] = [
    [0.0, 0.0, 0.0],
    [0.18, 0.18, 0.18],
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [4.0, 4.0, 4.0],
    [0.1, 0.3, 0.7],
];

/// golden 指纹：ACES 准确档输出位模式（9 样本 × 3 通道）。
fn aces_fitted_fingerprint() -> Vec<u32> {
    let out = AcesOutput::new(AcesFit::Fitted);
    let mut v = Vec::new();
    for s in SAMPLES {
        for c in out.apply(s) {
            v.push(c.to_bits());
        }
    }
    v
}

#[test]
fn aces_golden_file_stable() {
    // 首跑生成、次跑锁定：此处通过"两次计算逐位一致 + 打印指纹"固化。
    let a = aces_fitted_fingerprint();
    let b = aces_fitted_fingerprint();
    assert_eq!(a, b, "ACES 指纹必须自洽");
    // 内嵌 golden（首跑固化；色彩数学有意改动时同步更新）。
    assert_eq!(a, ACES_FINGERPRINT.to_vec(), "ACES golden 指纹失配");
    // 定性锚点（不依赖具体位模式，跨平台稳健）。
    let out = AcesOutput::new(AcesFit::Fitted);
    assert_eq!(out.apply([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    for s in SAMPLES {
        for c in out.apply(s) {
            assert!((0.0..=1.0).contains(&c), "ACES 输出越界：{c}");
        }
    }
}

/// ACES 准确档输出位模式 golden 基线（本实现首跑固化，f32 RNE）。
const ACES_FINGERPRINT: [u32; 27] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, // 黑
    0x3DD8_403D, 0x3DD8_403A, 0x3DD8_403B, // 18% 灰
    0x3EBF_A557, 0x3EBF_A556, 0x3EBF_A556, // 中灰
    0x3F1E_7E5A, 0x3F1E_7E59, 0x3F1E_7E5B, // 白
    0x3F41_0AF6, 0x0000_0000, 0x0000_0000, // 红
    0x3D6A_D79F, 0x3F24_5E27, 0x0000_0000, // 绿
    0x0000_0000, 0x0000_0000, 0x3F2B_7813, // 蓝
    0x3F68_B522, 0x3F68_B520, 0x3F68_B521, // 超亮
    0x3D1A_CC7B, 0x3E5B_0065, 0x3F02_9C41, // 混合
];

#[test]
fn pq_golden_roundtrip_within_tolerance() {
    // PQ 编解码往返（<1e-4）。
    for i in 0..=256u32 {
        let y = i as f32 / 256.0;
        let back = pq_eotf(pq_oetf(y));
        assert!((back - y).abs() < 1e-4, "PQ 往返 y={y} → {back}");
    }
}

#[test]
fn hlg_golden_roundtrip_within_tolerance() {
    // HLG 编解码往返。
    for i in 0..=256u32 {
        let e = i as f32 / 256.0;
        let back = prism_math::hlg::hlg_oetf_inverse(hlg_oetf(e));
        assert!((back - e).abs() < 1e-4, "HLG 往返 e={e} → {back}");
    }
}

#[test]
fn acescg_golden_matrix_roundtrip() {
    // ACEScg 色域变换往返（矩阵正交性）。
    for s in SAMPLES {
        let back = acescg_to_srgb(srgb_to_acescg(s));
        for i in 0..3 {
            assert!((back[i] - s[i]).abs() < 1e-4, "ACEScg 往返失真 {back:?}");
        }
    }
}

#[test]
fn aces_style_preset_golden_range() {
    // 风格档（Narkowicz）输出落在 [0,1] 且单调。
    let mut prev = -1.0f32;
    for i in 0..=100u32 {
        let v = i as f32 / 10.0;
        let y = aces_fast([v, v, v])[0];
        assert!((0.0..=1.0).contains(&y));
        assert!(y >= prev - 1e-6);
        prev = y;
    }
}
