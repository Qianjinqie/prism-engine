//! DWAA / DWAB 压缩率实测（交付要求：渐变 / 常量块 / 半随机 三档）。
//!
//! 跑法：
//! ```bash
//! cargo test -p prism-io --test dwa_ratio -- --nocapture
//! ```
//!
//! 输出 `原始字节 →压缩后字节 → 比值` 三列表格。所有数据由**本引擎实测**得出，
//! 不引用任何外部数值。

use prism_io::exr::PixelType;
use prism_io::fbuf::f32_to_f16;

/// 确定性伪随机（LCG），保证全链路可复现。
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// 生成 `width × lines` 的 half 平面原始字节（布局：扫描线 × 通道 × 像素）。
fn raw_plane(width: usize, lines: usize, gen: &dyn Fn(usize, usize) -> f32) -> Vec<u8> {
    let mut raw = Vec::with_capacity(width * lines * 2);
    for y in 0..lines {
        for x in 0..width {
            raw.extend_from_slice(&f32_to_f16(gen(x, y)).to_le_bytes());
        }
    }
    raw
}

#[test]
fn dwa_compression_ratio_table() {
    const W: usize = 256;
    const H: usize = 256;
    let raw_bytes = W * H * 2;

    // 三档图案。
    let gradient = |x: usize, y: usize| ((x + y) as f32) / (W + H) as f32;
    let constant = |_x: usize, _y: usize| 0.5f32;
    let semi_random = {
        let mut s: u32 = 0x5eed_1234;
        // 预生成一张表，保证三档之间除图案外完全同构。
        let mut table = vec![0.0f32; W * H];
        for v in table.iter_mut() {
            *v = lcg(&mut s) * 4.0 - 2.0;
        }
        move |x: usize, y: usize| table[y * W + x]
    };

    let cases: [(&str, &dyn Fn(usize, usize) -> f32); 3] = [
        ("渐变", &gradient),
        ("常量块", &constant),
        ("半随机", &semi_random),
    ];

    println!();
    println!("=== DWAA / DWAB 压缩率实测（{W}×{H} 单通道 half）===");
    println!("{:<10} {:>8} {:>14} {:>12} {:>10}", "图案", "方案", "原始字节", "压缩后字节", "比值");
    println!("{}", "-".repeat(60));

    for (name, gen) in cases {
        let raw = raw_plane(W, H, gen);
        assert_eq!(raw.len(), raw_bytes);
        let channels = vec![("Y".to_string(), PixelType::Half, false)];
        for (label, lines) in [("DWAA", 32usize), ("DWAB", 256usize)] {
            // 每个块单独压缩后累加，等价于整图按块切分。
            let mut total = 0usize;
            let step = lines;
            let mut y = 0usize;
            while y < H {
                let end = (y + step).min(H);
                let block = compress_rows(&raw, W, y, end);
                total += prism_io::exr_dwa::dwa_compress(&block, &channels, W, end - y)
                    .unwrap_or_else(|e| panic!("{label} 压缩失败: {e}"))
                    .len();
                y = end;
            }
            let ratio = raw_bytes as f64 / total as f64;
            println!(
                "{name:<10} {label:>8} {raw_bytes:>14} {total:>14} {ratio:>9.3}x"
            );
        }
    }
    println!();
    println!("注：比值 > 1 表示压缩有效；常量块因 AC 全零应接近 RLE/zlib 上限。");
}

/// 抽取 `raw` 的第 `y0..y1` 行（每行 `W × 2` 字节）。
fn compress_rows(raw: &[u8], w: usize, y0: usize, y1: usize) -> Vec<u8> {
    let stride = w * 2;
    raw[y0 * stride..y1 * stride].to_vec()
}