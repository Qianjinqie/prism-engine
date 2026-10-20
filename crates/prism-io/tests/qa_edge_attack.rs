//! QA 边界反例攻击（prism-io 侧）。
//!
//! 期望值全部由**独立来源**生成（Python hashlib / numpy float16 / 手动 RNE
//! 移位模拟），不引用被测实现的任何输出——保证对拍有效性。
//!
//! 本文件同时承担"双 SHA-256 实现同向量一致"的 io 侧锚定：
//! 与 `prism-core` 测试里的 `channel_definition_hash` 六向量（同一数据模式、
//! 同一 hashlib 基线族）互为镜像证据。

use prism_io::{f16_to_f32, f32_to_f16, sha256};

// ---------------------------------------------------------------------------
// C-④ f16 非正规数边界（2⁻²⁵ / 2⁻²⁴ 邻域）
// ---------------------------------------------------------------------------

/// C-④：非正规下溢边界。2⁻²⁵ 恰为最小半正规 2⁻²⁴ 的**一半**——RNE
/// ties-to-even 舍到 0x0000（偶数）；仅 +1 f32 ulp 越过中点即跳到 0x0001。
/// 期望值由 numpy float16 独立对拍生成。
#[test]
fn qa_f16_subnormal_boundaries() {
    let two_pow_m25 = f32::from_bits(0x3300_0000); // 2^-25
    let next_up = f32::from_bits(0x3300_0001); // 2^-25 的上邻 f32
    let prev_down = f32::from_bits(0x32FF_FFFF); // 2^-25 的下邻 f32

    assert_eq!(
        f32_to_f16(two_pow_m25),
        0x0000,
        "2⁻²⁵ = 半正规中点 → ties-even 舍到 0"
    );
    assert_eq!(
        f32_to_f16(next_up),
        0x0001,
        "2⁻²⁵+1ulp > 中点 → 舍到最小半正规"
    );
    assert_eq!(f32_to_f16(prev_down), 0x0000, "2⁻²⁵−1ulp < 中点 → 舍到 0");

    assert_eq!(f32_to_f16(2.0_f32.powi(-24)), 0x0001, "2⁻²⁴ = 最大半正规");
    assert_eq!(
        f32_to_f16(2.0_f32.powi(-15)),
        0x0200,
        "2⁻¹⁵ = 半正规区上邻（尾数 0x200）"
    );
    assert_eq!(f32_to_f16(2.0_f32.powi(-14)), 0x0400, "2⁻¹⁴ = 最小正规");

    // 反向展开精确（半正规 f16_to_f32 无舍入）。
    assert_eq!(f16_to_f32(0x0001), 2.0_f32.powi(-24));
    assert_eq!(f16_to_f32(0x0200), 2.0_f32.powi(-15));
    assert_eq!(f16_to_f32(0x0400), 2.0_f32.powi(-14));
}

/// C-④：NaN 载荷保持——IEEE 要求 NaN 收缩不得丢失全部载荷、不得变 ∞。
/// 期望位模式按实现的文档化规则（尾数 >>13 + RNE，m==0 强制 quiet）由独立
/// Python 移位模拟生成：0x7FC12345 → 0x7E09。
#[test]
fn qa_f16_nan_payload_preserved() {
    assert_eq!(
        f32_to_f16(f32::from_bits(0x7FC0_0000)),
        0x7E00,
        "标准 quiet NaN"
    );
    assert_eq!(
        f32_to_f16(f32::from_bits(0x7FC1_2345)),
        0x7E09,
        "带载荷 NaN：尾数 0x412345 >>13 = 0x209，舍入位 0 → 0x7C00|0x209"
    );
    assert_eq!(
        f32_to_f16(f32::from_bits(0x7F80_0001)),
        0x7C01,
        "信号 NaN（载荷 1）：m==0 强制 quiet bit → 0x7C01"
    );
    assert_eq!(f32_to_f16(f32::from_bits(0xFFC0_0001)), 0xFE00, "负号保留");

    // 收缩结果回展仍是 NaN（不是 inf、不是有限值）。
    for bits in [0x7FC0_0000_u32, 0x7FC1_2345, 0x7F80_0001, 0xFFC0_0001] {
        let back = f16_to_f32(f32_to_f16(f32::from_bits(bits)));
        assert!(
            back.is_nan(),
            "0x{bits:08X} 收缩回展应仍为 NaN，实际 {back}"
        );
        assert!(!back.is_infinite());
    }
}

/// C-④：正规上界锚——half 最大正规 65504 保持；溢出中点 65520 ties-even
/// 到 +∞（fbuf.rs 单元测试已覆盖穷举恒等，此处固化两个最易记错的点）。
#[test]
fn qa_f16_max_normal_and_overflow_midpoint() {
    assert_eq!(f32_to_f16(65504.0), 0x7BFF, "half 最大正规");
    assert_eq!(
        f32_to_f16(65520.0),
        0x7C00,
        "溢出中点 65520 → ties-even 到 +inf"
    );
    assert_eq!(f32_to_f16(65519.0), 0x7BFF, "中点之下仍最大正规");
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
}

// ---------------------------------------------------------------------------
// C-⑥ SHA-256 块边界（55/56/57/63/64/65 字节）
// ---------------------------------------------------------------------------

/// 测试数据生成器：data[i] = (i*7+3) mod 256（与 Python 基线脚本同一模式）。
fn attack_data(n: usize) -> Vec<u8> {
    (0..n).map(|i: usize| ((i * 7 + 3) % 256) as u8).collect()
}

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// C-⑥：块边界六长度——55（padding 后恰 1 块）、56（padding 后跨 2 块）、
/// 57、63、64（长度字段本身触发第二块）、65。期望值由 Python hashlib
/// 独立生成（hex 字面量硬编码，与实现零耦合）。
#[test]
fn qa_sha256_block_boundary_lengths() {
    let cases: [(usize, &str); 6] = [
        (
            55,
            "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b",
        ),
        (
            56,
            "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27",
        ),
        (
            57,
            "35df609437dcfea3279283ab79fd554e2bf78f8f7ae2de532d8ee300b09e8f73",
        ),
        (
            63,
            "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055",
        ),
        (
            64,
            "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241",
        ),
        (
            65,
            "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e",
        ),
    ];
    for (n, want) in cases {
        let got = hex(&sha256(&attack_data(n)));
        assert_eq!(got, want, "SHA-256 {n} 字节块边界向量不符");
    }
}

/// D 项（io 侧）：FIPS 180-4 官方锚（与 hashlib 基线共同构成双锚）。
#[test]
fn qa_sha256_fips_official_anchors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
