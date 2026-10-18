//! JPEG 基线解码器测试（V1.5-W1-A / P0-2）。
//!
//! 三层对拍：
//! 1. **IDCT 金参考**：`jpeg_idct_input.bin`（2000 自然块）→ `jpeg_idct_golden.bin`
//!    （python `math.cos` 全精度 f64 参考，禁引擎浮点路径作参考）；判据 = 误差界
//!    `(2^-16+2^-19)·Σ|D| + 0.5 LSB` 的实际落点（≤2 LSB）+ PSNR 粗门（≥55 dB）。
//! 2. **阈值分流 / 双路径一致**：A3 两侧块（8191·2^16 ≤ 阈值 < 8192·2^16）与
//!    1000 块随机抽样上 i64 / i128 两路径必须逐位相同。
//! 3. **互操**：Pillow+libjpeg 生成的真实 JPEG（4:4:4 / 4:2:2 / 4:2:0 / 灰度）
//!    与本解码器逐像素 PSNR ≥ 40 dB。

use std::path::{Path, PathBuf};

use prism_io::jpeg_baseline::{
    decode_baseline, idct_8x8_fixed, idct_8x8_fixed_wide, IDCT_STAGE2_I64_THRESHOLD,
    IDCT_W_TABLE,
};

/// 固件目录。
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("jpeg")
}

/// 读固件字节；缺失时给出可行动的 panic 文案。
fn read_fixture(name: &str) -> Vec<u8> {
    let p: PathBuf = fixtures_dir().join(name);
    std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "读不到固件 {}（{}）：请先运行 \
             `python3 crates/prism-io/tests/jpeg_golden_gen.py`",
            p.display(),
            e
        )
    })
}

/// 解析块输入流：每块 64×i16 LE（自然序系数）+ 64×u8（量化表）。
fn parse_blocks(bytes: &[u8]) -> Vec<([i32; 64], [i32; 64])> {
    const BLOCK: usize = 64 * 2 + 64;
    let mut out: Vec<([i32; 64], [i32; 64])> = Vec::with_capacity(bytes.len() / BLOCK);
    let mut off: usize = 0;
    while off + BLOCK <= bytes.len() {
        let mut coef = [0i32; 64];
        let mut q = [0i32; 64];
        for i in 0..64 {
            let hi: u8 = bytes[off + i * 2];
            let lo: u8 = bytes[off + i * 2 + 1];
            coef[i] = i16::from_le_bytes([hi, lo]) as i32;
        }
        for i in 0..64 {
            q[i] = bytes[off + 128 + i] as i32;
        }
        out.push((coef, q));
        off += BLOCK;
    }
    out
}

/// PSNR（u8 值域，max = 255）。
fn psnr_u8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR 输入长度必须一致");
    let mut se: f64 = 0.0;
    for i in 0..a.len() {
        let d: f64 = a[i] as f64 - b[i] as f64;
        se += d * d;
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    let mse: f64 = se / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// IDCT 金参考对拍：2000 自然块 vs python 精确 f64 参考。
#[test]
fn idct_matches_python_golden() {
    let input: Vec<u8> = read_fixture("jpeg_idct_input.bin");
    let golden: Vec<u8> = read_fixture("jpeg_idct_golden.bin");
    let blocks: Vec<([i32; 64], [i32; 64])> = parse_blocks(&input);
    assert!(!blocks.is_empty(), "固件块数为 0（固件损坏）");
    assert_eq!(blocks.len() * 64, golden.len(), "金色输出长度应与块数 × 64 一致");

    let mut got: Vec<u8> = Vec::with_capacity(golden.len());
    let mut max_abs: i32 = 0;
    let mut fast_blocks: usize = 0;
    for (coef, q) in blocks.iter() {
        let mut out = [0i32; 64];
        idct_8x8_fixed(&mut out, coef, q);
        // 记录分流覆盖（用于断言 i64 快路径确实被走到）。
        let mut max_dw: i64 = 0;
        for i in 0..64 {
            let dw: i64 = ((coef[i] as i64) * (q[i] as i64) * (IDCT_W_TABLE[i] as i64)).abs();
            if dw > max_dw {
                max_dw = dw;
            }
        }
        if max_dw <= IDCT_STAGE2_I64_THRESHOLD {
            fast_blocks += 1;
        }
        for v in out.iter() {
            got.push(*v as u8);
        }
    }
    for i in 0..golden.len() {
        let d: i32 = (got[i] as i32 - golden[i] as i32).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let p: f64 = psnr_u8(&golden, &got);
    // 判据：误差界 (2^-16+2^-19)·Σ|D| + 0.5 LSB 的实际落点（稀疏大 Σ|D| 档到 2 LSB）。
    assert!(max_abs <= 2, "定点 IDCT 与 f64 参考最大差 {} LSB > 2", max_abs);
    assert!(p >= 55.0, "定点 IDCT vs f64 参考 PSNR {:.3} dB < 55 dB", p);
    // 双路覆盖：自然块中必须有 i64 快路径样本（否则阈值分流未被验证）。
    assert!(fast_blocks > 0, "自然块固件未覆盖 i64 快路径（阈值分流未验证）");
    println!(
        "IDCT 金参考：{} 块，max {} LSB，PSNR {:.3} dB，i64 快路径 {} 块",
        blocks.len(),
        max_abs,
        p,
        fast_blocks
    );
}

/// 阈值分流：固件 A3 两块 + 本地边界块，i64 / i128 两路径逐位一致。
///
/// **固件口径校正**（W1-A）：`jpeg_idct_a3.bin` 的两块只把 `coef[0]` 设为
/// 8191 / 8192 且 `q=1`，而 `W[0][0] = 32768 = 2^15`（不是 2^16），故实际
/// `max|D·W| = 8191·2^15 = 268402688` / `8192·2^15 = 268435456`，**两块都在
/// 阈值内**——固件注释里的「8191·2^16」把 W 当成了 65536。真正的阈值两侧点由
/// 下面的本地块给出（`coef[0] = 2·8191` 与 `2·8192`）。
#[test]
fn idct_threshold_two_sides() {
    let a3: Vec<u8> = read_fixture("jpeg_idct_a3.bin");
    let blocks: Vec<([i32; 64], [i32; 64])> = parse_blocks(&a3);
    assert_eq!(blocks.len(), 2, "A3 固件应含两块，实际 {}", blocks.len());

    let below: [i32; 64] = blocks[0].0;
    let qb: [i32; 64] = blocks[0].1;
    let above: [i32; 64] = blocks[1].0;
    let qa: [i32; 64] = blocks[1].1;
    assert_eq!((below[0] as i64) * (qb[0] as i64) * 32768, 8191 * (1 << 15));
    assert_eq!((above[0] as i64) * (qa[0] as i64) * 32768, 8192 * (1 << 15));

    // 固件两块：双路径输出必须逐位一致（同值域下的等价性）。
    for (coef, q) in blocks.iter() {
        let mut fast = [0i32; 64];
        let mut wide = [0i32; 64];
        idct_8x8_fixed(&mut fast, coef, q);
        idct_8x8_fixed_wide(&mut wide, coef, q);
        assert_eq!(fast, wide, "A3 块 i64/i128 路径输出必须逐位一致");
    }

    // 真正的阈值两侧（W[0]=2^15 ⇒ D = 2·8191 / 2·8192）。
    let mut lo = [0i32; 64];
    let mut hi = [0i32; 64];
    lo[0] = 2 * 8191;
    hi[0] = 2 * 8192;
    let q = [1i32; 64];
    let max_lo: i64 = (lo[0] as i64) * 32768;
    let max_hi: i64 = (hi[0] as i64) * 32768;
    assert_eq!(max_lo, 8191 * (1 << 16), "阈内块 max|D·W| 应为 8191·2^16");
    assert_eq!(max_hi, 8192 * (1 << 16), "超界块 max|D·W| 应为 8192·2^16");
    assert!(max_lo <= IDCT_STAGE2_I64_THRESHOLD, "8191·2^16 应落 i64 快路径");
    assert!(
        max_hi > IDCT_STAGE2_I64_THRESHOLD,
        "8192·2^16 必须超阈值一个单位（走 i128）"
    );
    for coef in [lo, hi] {
        let mut fast = [0i32; 64];
        let mut wide = [0i32; 64];
        idct_8x8_fixed(&mut fast, &coef, &q);
        idct_8x8_fixed_wide(&mut wide, &coef, &q);
        assert_eq!(fast, wide, "阈值两侧块 i64/i128 路径输出必须逐位一致");
    }
}

/// 双路径一致性抽样 1000 块（含极端块流，全走 i128；自然块流混合双路）。
#[test]
fn idct_dual_path_agrees_bitwise_1000_blocks() {
    let mut checked: usize = 0;
    for name in ["jpeg_idct_input.bin", "jpeg_idct_extreme.bin"] {
        let bytes: Vec<u8> = read_fixture(name);
        for (coef, q) in parse_blocks(&bytes).iter().take(500) {
            let mut fast = [0i32; 64];
            let mut wide = [0i32; 64];
            idct_8x8_fixed(&mut fast, coef, q);
            idct_8x8_fixed_wide(&mut wide, coef, q);
            assert_eq!(fast, wide, "i64/i128 路径输出不一致（文件 {}）", name);
            checked += 1;
        }
    }
    assert!(checked >= 1000, "双路径抽样仅 {} 块，应 ≥1000", checked);
}

/// 极端块：不 panic、不回绕（输出必须落在 [0,255]）。
#[test]
fn idct_extreme_blocks_no_panic_and_in_range() {
    let bytes: Vec<u8> = read_fixture("jpeg_idct_extreme.bin");
    let blocks = parse_blocks(&bytes);
    assert!(!blocks.is_empty(), "极端块固件为空");
    for (coef, q) in blocks.iter() {
        let mut out = [0i32; 64];
        idct_8x8_fixed(&mut out, coef, q);
        for v in out.iter() {
            assert!((0..=255).contains(v), "极端块输出越界：{}", v);
        }
    }
}

/// 互操：Pillow/libjpeg 真实 JPEG vs 自研解码器，PSNR ≥ 40 dB。
#[test]
fn interop_real_jpeg_psnr_over_40db() {
    for name in ["interop_444", "interop_422", "interop_420", "interop_gray"] {
        let jpg: Vec<u8> = read_fixture(&format!("{}.jpg", name));
        let expect: Vec<u8> = read_fixture(&format!("{}.rgb.bin", name));
        let img = decode_baseline(&jpg).unwrap_or_else(|e| panic!("{} 解码失败：{}", name, e));
        assert_eq!(
            img.rgb.len(),
            expect.len(),
            "{} 像素数不符（解码 {} / 期望 {}）",
            name,
            img.rgb.len(),
            expect.len()
        );
        let p: f64 = psnr_u8(&expect, &img.rgb);
        assert!(p >= 40.0, "{} 互操 PSNR {:.2} dB < 40 dB", name, p);
        println!("{}: {}x{} PSNR {:.2} dB", name, img.width, img.height, p);
    }
}

/// 同输入两跑逐位相同（确定性：无浮点、无 libm）。
#[test]
fn decode_is_bitwise_deterministic() {
    let jpg: Vec<u8> = read_fixture("interop_420.jpg");
    let a = decode_baseline(&jpg).expect("解码应成功");
    let b = decode_baseline(&jpg).expect("解码应成功");
    assert_eq!(a, b, "同输入两跑必须逐位相同");
    assert_eq!(a.width, 96);
    assert_eq!(a.height, 72);
    assert_eq!(a.rgb.len(), 96 * 72 * 3);
}

/// 非 JPEG 输入必须报错（诚实边界）。
#[test]
fn rejects_non_jpeg() {
    let err: String = decode_baseline(b"not a jpeg").unwrap_err();
    assert!(err.contains("SOI"), "错误文案应指明缺少 SOI，实际：{}", err);
}
