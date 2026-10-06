//! `exr_compress` 测试：四种压缩的**逐位**往返证据 + 畸形输入结构化拒绝。
//!
//! # 证据分级
//!
//! 1. **最强**：`同一图像 → None 路径的原始块字节 == Xxx 编码后再解出的块字节`
//!    （逐位相等）。这同时验证了「块内 EXR 布局契约」与「编解码对逆」。
//! 2. **端到端**：构造真实 EXR 文件字节流 → [`crate::exr::read_image`] 读回，
//!    与 None 路径的读回结果逐位对拍。
//! 3. **参考常量对拍**：`lines_per_block`、`PXR24 float→24bit` 语义、
//!    `pLinear=true` 的结构化拒绝路径。
//! 4. **确定性**：同输入连解两遍字节级一致。
//!
//! B44/B44A 的 **HALF 通道在格式上就是有损的**（固定 32/14 压缩率），
//! 故 HALF 用「误差有界 + 平场精确」断言；**UINT/FLOAT 通道逐位无损**
//! （B44 对它们原样复制），这条走断言 1。PXR24 的 **FLOAT 通道同理有损**
//! （32bit→24bit），HALF/UINT 逐位无损。

use super::*;
// 编码器全部 `#[cfg(test)]` 门控在 `exr_compress.rs` 内（不进生产构建），
// 测试模块需显式引入；`f32_to_f16` 供 `quantize` 把 float 像素写成 half 位模式。
use crate::fbuf::f32_to_f16;
use crate::exr::{Compression, DataWindow, ExrChannel};

// ===========================================================================
//  脚手架
// ===========================================================================

/// 造一条 [`ExrChannel`]。
fn ch(name: &str, pt: PixelType) -> ExrChannel {
    ExrChannel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }
}

/// 造一条 `p_linear = true` 的 HALF 通道（B44 的 exp/log 路径专用）。
fn ch_linear(name: &str) -> ExrChannel {
    ExrChannel {
        p_linear: true,
        ..ch(name, PixelType::Half)
    }
}

/// 把 f32 量化成通道原始位模式字节（与 `exr_write` 写入侧一致）。
fn quantize(v: f32, pt: PixelType) -> [u8; 4] {
    let mut out = [0u8; 4];
    match pt {
        PixelType::Half => out[..2].copy_from_slice(&f32_to_f16(v).to_le_bytes()),
        PixelType::Float => out.copy_from_slice(&v.to_le_bytes()),
        PixelType::Uint => {
            let c = if v.is_nan() {
                0.0
            } else {
                v.max(0.0).min(u32::MAX as f32)
            };
            out.copy_from_slice(&(c as u32).to_le_bytes());
        }
    }
    out
}

/// 按「通道 → 块内行 → 像素」把 f32 平面编成块的**原始 EXR 布局字节**（小端）。
///
/// `planes[ci][y][x]` = 通道 `ci`、块内第 `y` 行、第 `x` 列的 f32 值。
fn planes_to_raw(
    planes: &[Vec<Vec<f32>>],
    width: usize,
    channels: &[ExrChannel],
    lines: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(width * lines * 8);
    for y in 0..lines {
        for (ci, c) in channels.iter().enumerate() {
            for x in 0..width {
                let v = planes
                    .get(ci)
                    .and_then(|p| p.get(y))
                    .and_then(|r| r.get(x))
                    .copied()
                    .unwrap_or(0.0);
                out.extend_from_slice(&quantize(v, c.pixel_type)[..c.pixel_type.byte_width()]);
            }
        }
    }
    out
}

/// 用确定性图案（跨零、含负、含大值，half 可精确表示）造 f32 平面。
///
/// 图案由 `seed` 与通道下标决定 —— **无 `HashMap`、无随机数**，可复现。
fn synth(
    nch: usize,
    width: usize,
    lines: usize,
    pt: PixelType,
    seed: u32,
) -> Vec<Vec<Vec<f32>>> {
    (0..nch)
        .map(|ci| {
            (0..lines)
                .map(|y| {
                    (0..width)
                        .map(|x| {
                            let i = (y * width + x) as u32;
                            let a = (i.wrapping_mul(37)
                                .wrapping_add(seed.wrapping_mul(101).wrapping_add(ci as u32 * 7919))
                                % 251) as f32;
                            let b = i as f32 * 0.125 - (seed as f32) * 3.5 + (ci as f32) * 11.0;
                            match pt {
                                PixelType::Uint => a + b,
                                _ => a * 0.03125 + b,
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// 整块取绝对值（UINT 通道需要非负）。
fn abs_all(planes: &mut [Vec<Vec<f32>>]) {
    for p in planes.iter_mut() {
        for r in p.iter_mut() {
            for v in r.iter_mut() {
                *v = v.abs();
            }
        }
    }
}

/// 造块上下文。
fn ctx_of<'a>(channels: &'a [ExrChannel], width: usize, lines: usize) -> BlockCtx<'a> {
    BlockCtx {
        channels,
        width,
        height: lines,
    }
}

/// 抽取原始块里某通道某行的字节区间（用于「逐通道逐行」断言）。
///
/// 块内布局是「扫描线 × 通道 × 像素」，故第 `ci` 个通道在第 `y` 行的起点
/// = `y × 通道字节和` + 前面通道的字节和（此处调用方都传「前 `ci` 个通道
/// 各自的字节宽度之和」作 `x_off`）。
fn row_slice(raw: &[u8], y: usize, x_off: usize, w: usize, bw: usize) -> &[u8] {
    let stride = x_off + w * bw;
    let off = y * stride + x_off;
    raw.get(off..off + w * bw).unwrap_or(&[])
}

// ===========================================================================
//  1) PIZ
// ===========================================================================

/// PIZ：half 通道 —— `原始块 → PIZ 编码 → 解码` **逐位相同**。
#[test]
fn piz_half_block_round_trips_bit_exact() {
    let (w, lines) = (16usize, 32usize);
    let channels = vec![ch("R", PixelType::Half), ch("G", PixelType::Half)];
    let planes = synth(channels.len(), w, lines, PixelType::Half, 1);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let back = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.len(), raw.len());
    assert_eq!(back, raw, "PIZ half 往返失真");
}

/// PIZ：float + uint 混合通道，逐位相同。
#[test]
fn piz_float_and_uint_round_trip_bit_exact() {
    let (w, lines) = (12usize, 32usize);
    let channels = vec![ch("Y", PixelType::Float), ch("id", PixelType::Uint)];
    let mut planes = synth(channels.len(), w, lines, PixelType::Float, 2);
    abs_all(&mut planes);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let back = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, raw, "PIZ float/uint 往返失真");
}

/// PIZ：非 2 的幂 / 非 4 的倍数的尺寸，覆盖小波的奇列 / 奇行尾部分支。
#[test]
fn piz_odd_dimensions_round_trip_bit_exact() {
    for (w, lines) in [(1usize, 1usize), (3, 5), (5, 7), (7, 13), (17, 3)] {
        let channels = vec![ch("Z", PixelType::Half)];
        let planes = synth(1, w, lines, PixelType::Half, 3);
        let raw = planes_to_raw(&planes, w, &channels, lines);
        let ctx = ctx_of(&channels, w, lines);
        let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e} · {w}×{lines}"));
        let back = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e} · {w}×{lines}"));
        assert_eq!(back, raw, "PIZ {w}×{lines} 往返失真");
    }
}

/// PIZ：全零块（bitmap 全空 ⇒ `minNonZero > maxNonZero`，不写 bitmap 段）。
#[test]
fn piz_all_zero_block_round_trips() {
    let (w, lines) = (8usize, 4usize);
    let channels = vec![ch("R", PixelType::Half)];
    let raw = vec![0u8; w * lines * 2];
    let ctx = ctx_of(&channels, w, lines);
    let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    // 头 4 字节：minNonZero = 8191 > maxNonZero = 0 ⇒ 无 bitmap 段。
    assert_eq!(u16::from_le_bytes([comp[0], comp[1]]), 8191);
    assert_eq!(u16::from_le_bytes([comp[2], comp[3]]), 0);
    let back = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, raw, "PIZ 全零块往返失真");
}

/// PIZ：值域大 ⇒ 小波走 **16bit 模运算** 分支（`mx ≥ 2^14`）。
#[test]
fn piz_wide_value_range_uses_w16_branch() {
    let (w, lines) = (8usize, 8usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let mut raw = Vec::with_capacity(w * lines * 2);
    for i in 0..(w * lines) {
        raw.extend_from_slice(&if i % 2 == 0 { 0xFFFFu16 } else { 0x0000 }.to_le_bytes());
    }
    let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let back = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, raw, "PIZ w16 分支往返失真");
}

/// 小波原语：正/逆变换互逆（覆盖 14bit 与 16bit 两条 lifting 路径）。
///
/// **值域契约**（OpenEXR 靠这个前提保证 w14 可逆）：`mx < 2^14` 时数据必须
/// 落在 14bit 内（PIZ 在小波前先做了 LUT，值域被压到 `[0, maxValue]`）；
/// `mx ≥ 2^14` 才走 16bit 模运算。测试按此契约给数据。
#[test]
fn wav2_encode_decode_are_mutual_inverses() {
    for mx in [0u16, 100, 8191, 16383, 16384, 65535] {
        let w14 = mx < (1 << 14);
        for (nx, ny, ox) in [(8usize, 8usize, 1usize), (5, 9, 2), (16, 4, 1)] {
            let n = nx * ny * ox;
            // LCG 伪随机（确定性，不引第三方）。
            let mut seed = 0x1234_5678u32;
            let mut d: Vec<u16> = Vec::with_capacity(n);
            for _ in 0..n {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let raw = (seed >> 13) as u16;
                // w14 分支：压到 14bit（含负半区，模拟 lifting 后的中间值）。
                d.push(if w14 { raw & 0x3FFF } else { raw });
            }
            let orig = d.clone();
            wav2_encode(&mut d, nx, ox, ny, nx * ox, mx).unwrap_or_else(|e| panic!("{e}"));
            wav2_decode(&mut d, nx, ox, ny, nx * ox, mx).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d, orig, "小波往返失真：mx={mx} {nx}×{ny}×{ox}");
        }
    }
}

/// PIZ：畸形输入结构化拒绝（6 条），不 panic。
#[test]
fn piz_malformed_inputs_are_structured_errors() {
    let (w, lines) = (8usize, 4usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let planes = synth(1, w, lines, PixelType::Half, 9);
    let good = piz_compress(&planes_to_raw(&planes, w, &channels, lines), &ctx)
        .unwrap_or_else(|e| panic!("{e}"));

    // ① 头部不足 4 字节。
    let e = piz_decompress(&good[..2], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ② maxNonZero 超过 bitmap 尺寸（8192）。
    let mut bad = good.clone();
    bad[2] = 0x00;
    bad[3] = 0x40; // 16384 ≥ 8192
    let e = piz_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
    assert!(format!("{e}").contains("bitmap"), "{e}");

    // ③ / ④ Huf 段长度字段损坏（负 / 超界）。
    let mnz = u16::from_le_bytes([good[0], good[1]]) as usize;
    let mxz = u16::from_le_bytes([good[2], good[3]]) as usize;
    let len_at = if mnz <= mxz { 4 + (mxz - mnz + 1) } else { 4 };
    let mut bad = good.clone();
    bad[len_at..len_at + 4].copy_from_slice(&(-1i32).to_le_bytes());
    let e = piz_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    let mut bad = good.clone();
    bad[len_at..len_at + 4].copy_from_slice(&(1_000_000i32).to_le_bytes());
    let e = piz_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ⑤ Huf 头被破坏（im 越界）。
    let mut bad = good.clone();
    let h = len_at + 4;
    bad[h..h + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let e = piz_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ⑥ 空负载。
    let e = piz_decompress(&[], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
}

/// PIZ：解码确定性（同输入解两遍字节级一致）。
#[test]
fn piz_decoding_is_deterministic() {
    let (w, lines) = (13usize, 9usize);
    let channels = vec![ch("R", PixelType::Half), ch("G", PixelType::Float)];
    let planes = synth(2, w, lines, PixelType::Float, 5);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = piz_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let a = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let b = piz_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a, b);
    assert_eq!(a, raw);
}

/// Huf：自研编码 → `exr_dwa::huf_uncompress` 逐位解回（**跨模块复用**证据）。
#[test]
fn huf_compress_is_decoded_bit_exact_by_dwa_decoder() {
    for n in [1usize, 2, 17, 300, 5000] {
        let mut seed = 0xC0FF_EEu32;
        let raw: Vec<u16> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(2_203_261_655).wrapping_add(12_345);
                if i % 7 == 0 {
                    (seed >> 20) as u16 % 5
                } else {
                    (seed >> 13) as u16
                }
            })
            .collect();
        let comp = huf_compress(&raw).unwrap_or_else(|e| panic!("{e} · n={n}"));
        let back = crate::exr_dwa::huf_uncompress(&comp, n).unwrap_or_else(|e| panic!("{e} · n={n}"));
        assert_eq!(back, raw, "Huf 往返失真 n={n}");
    }
}

// ===========================================================================
//  2) PXR24
// ===========================================================================

/// PXR24：half + uint 通道**无损**，逐位相同。
#[test]
fn pxr24_half_and_uint_round_trip_bit_exact() {
    let (w, lines) = (16usize, 16usize);
    let channels = vec![ch("R", PixelType::Half), ch("id", PixelType::Uint)];
    let mut planes = synth(2, w, lines, PixelType::Half, 7);
    abs_all(&mut planes);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = pxr24_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let back = pxr24_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, raw, "PXR24 half/uint 往返失真");
}

/// PXR24：float 通道格式固有**有损**（32bit→24bit），误差须有界。
#[test]
fn pxr24_float_is_lossy_but_bounded() {
    let (w, lines) = (8usize, 16usize);
    let channels = vec![ch("Y", PixelType::Float)];
    // 用**满尾数**的 float（1 + k/2^23 的随机 k）—— 只有这种值才会被 24bit 量化舍入。
    let mut seed = 0x9E37_79B9u32;
    let planes = vec![vec![vec![0.0f32; w]; lines]];
    let mut p0 = planes.get(0).cloned().unwrap_or_default();
    for row in p0.iter_mut() {
        for v in row.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // 符号 1 位 + 指数（避开 0/NaN/inf）+ 23 位随机尾数
            let bits = 0x3F00_0000u32 | (seed & 0x007F_FFFF);
            *v = f32::from_bits(bits);
        }
    }
    let raw = planes_to_raw(&[p0], w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = pxr24_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let back = pxr24_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.len(), raw.len());
    let mut worst = 0.0f64;
    for i in 0..(w * lines) {
        let a = f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        let b = f32::from_le_bytes([back[i * 4], back[i * 4 + 1], back[i * 4 + 2], back[i * 4 + 3]]);
        // 24bit 尾数 ⇒ 相对误差 ≤ 2^-14。
        let rel = ((a - b) as f64).abs() / (a as f64).abs().max(1e-30);
        worst = worst.max(rel);
        assert!(rel < 6.1e-5, "像素 {i}: {a} → {b}，相对误差 {rel} 超界");
    }
    assert!(worst > 0.0, "float 通道应确有舍入（否则本测试无意义）");
}

/// PXR24：`floatToFloat24` 对拍 OpenEXR 语义。
#[test]
fn pxr24_float_to_float24_matches_reference_semantics() {
    // 24bit 值 = (sign>>8) | (指数与尾数右移 8 位)。
    for f in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let bits = f.to_bits();
        let expect = ((bits >> 8) & 0x007f_ffff) | ((bits >> 8) & 0x0080_0000);
        assert_eq!(float_to_float24(f), expect, "floatToFloat24({f})");
    }
    // NaN：保符号 + 尾数高 15 位，且至少 1 位（不能退化成 inf）。
    let nan24 = float_to_float24(f32::NAN);
    assert_ne!(nan24 & 0x7f80_0000, 0x7f80_0000, "NaN 不得退化为 inf");
    assert_ne!(nan24 & 0x007f_ffff, 0, "NaN 尾数不得全零");
    // FLT_MAX 附近不得因四舍五入导致指数上溢成 inf。
    let v24 = float_to_float24(f32::from_bits(0x7f7f_ffff));
    assert!(v24 < 0x7f8_000, "接近 FLT_MAX 时须截断尾数：{v24:#x}");
}

/// PXR24：各种块高（含末块不满）逐位往返。
#[test]
fn pxr24_short_block_round_trips_bit_exact() {
    for lines in [1usize, 5, 15, 16] {
        let w = 9usize;
        let channels = vec![ch("R", PixelType::Half)];
        let planes = synth(1, w, lines, PixelType::Half, 13);
        let raw = planes_to_raw(&planes, w, &channels, lines);
        let ctx = ctx_of(&channels, w, lines);
        let comp = pxr24_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let back = pxr24_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, raw, "PXR24 {lines} 行往返失真");
    }
}

/// PXR24：畸形输入结构化拒绝（5 条），不 panic。
#[test]
fn pxr24_malformed_inputs_are_structured_errors() {
    let (w, lines) = (8usize, 4usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let planes = synth(1, w, lines, PixelType::Half, 17);
    let good = pxr24_compress(&planes_to_raw(&planes, w, &channels, lines), &ctx)
        .unwrap_or_else(|e| panic!("{e}"));

    // ① 非 zlib 流（CM != 8）。
    let mut bad = good.clone();
    bad[0] = 0x00;
    let e = pxr24_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ② 头部截断。
    let e = pxr24_decompress(&good[..1], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ③ 负载截断。
    let e = pxr24_decompress(&good[..good.len() / 2], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ④ 解出的字节数与块尺寸不符。
    let fake = crate::exr_write::deflate_zlib(&[0u8; 7]);
    let e = pxr24_decompress(&fake, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
    assert!(format!("{e}").contains("字节"), "{e}");

    // ⑤ 空负载。
    let e = pxr24_decompress(&[], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
}

/// PXR24：解码确定性。
#[test]
fn pxr24_decoding_is_deterministic() {
    let (w, lines) = (11usize, 6usize);
    let channels = vec![ch("R", PixelType::Half), ch("G", PixelType::Half)];
    let planes = synth(2, w, lines, PixelType::Half, 19);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = pxr24_compress(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let a = pxr24_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let b = pxr24_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a, b);
    assert_eq!(a, raw);
}

// ===========================================================================
//  3) B44 / B44A
// ===========================================================================

/// B44：UINT / FLOAT 通道**原样复制**，逐位无损（走最强证据路径）。
#[test]
fn b44_uint_and_float_are_verbatim_bit_exact() {
    let (w, lines) = (8usize, 32usize);
    let channels = vec![ch("Y", PixelType::Float), ch("id", PixelType::Uint)];
    let mut planes = synth(2, w, lines, PixelType::Float, 23);
    abs_all(&mut planes);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = b44_compress(&raw, &ctx, false).unwrap_or_else(|e| panic!("{e}"));
    let back = b44_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, raw, "B44 uint/float 往返失真");
}

/// B44/B44A：HALF 平场块**必须精确**（块内差分全 0）。
#[test]
fn b44_flat_half_block_is_exact() {
    let (w, lines) = (16usize, 32usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let planes = vec![vec![vec![1.5f32; w]; lines]];
    let raw = planes_to_raw(&planes, w, &channels, lines);

    // B44A：允许 3 字节平场路径。
    let a = b44_compress(&raw, &ctx, true).unwrap_or_else(|e| panic!("{e}"));
    let back_a = b44_decompress(&a, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back_a, raw, "B44A 平场块必须精确还原");

    // B44：不许 3 字节 ⇒ 走 14 字节路径，负载更长，但仍精确。
    let b = b44_compress(&raw, &ctx, false).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        b.len() > a.len(),
        "B44 不允许平场 3 字节路径，负载应更长：{} vs {}",
        b.len(),
        a.len()
    );
    let back_b = b44_decompress(&b, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back_b, raw, "B44 14 字节平场块应精确");
}

/// B44：HALF 渐变块格式固有有损，误差须有界。
///
/// 图案起点远离 0（B44 是**相对**/差分编码，0 附近无法用差分表示有效位），
/// 且跨度足够大（步长 100）：4×4 块内的差分必须右移 `shift > 0` 位，舍入不可避免。
/// **断言**：逐像素相对误差 < 5%（B44 固定 32B→14B 的量化上界），且确有非零舍入
/// （否则该图案退化为无损、本测试无意义）。
#[test]
fn b44_half_gradient_is_lossy_within_bound() {
    let (w, lines) = (16usize, 32usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let planes = vec![vec![(0..w)
        .map(|x| 8.0 + x as f32 * 100.0)
        .collect::<Vec<f32>>(); lines]];
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let comp = b44_compress(&raw, &ctx, false).unwrap_or_else(|e| panic!("{e}"));
    let back = b44_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let mut worst = 0.0f64;
    for i in 0..(w * lines) {
        let a = crate::fbuf::f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
        let b = crate::fbuf::f16_to_f32(u16::from_le_bytes([back[i * 2], back[i * 2 + 1]]));
        let rel = ((a - b) as f64).abs() / (a as f64).abs().max(1e-9);
        worst = worst.max(rel);
        assert!(rel < 5.0e-2, "像素 {i}: {a} → {b}，相对误差 {rel} 超界");
    }
    assert!(worst > 0.0, "大跨度渐变块应确有舍入（否则本测试无意义）");
}

/// B44：非 4 倍数的宽/高（边缘块补齐路径）；UINT 通道逐位无损。
#[test]
fn b44_non_multiple_of_four_dimensions_round_trip() {
    for (w, lines) in [(1usize, 1usize), (3, 3), (5, 6), (7, 9), (13, 11)] {
        let channels = vec![ch("R", PixelType::Half), ch("id", PixelType::Uint)];
        let mut planes = synth(2, w, lines, PixelType::Half, 29);
        abs_all(&mut planes);
        let raw = planes_to_raw(&planes, w, &channels, lines);
        let ctx = ctx_of(&channels, w, lines);
        let comp = b44_compress(&raw, &ctx, true).unwrap_or_else(|e| panic!("{e} · {w}×{lines}"));
        let back = b44_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e} · {w}×{lines}"));
        assert_eq!(back.len(), raw.len(), "{w}×{lines} 长度不符");
        // HALF 在前（2 字节/像素），UINT 在后（4 字节/像素）—— 逐行对拍 UINT。
        let ub = channels[1].pixel_type.byte_width();
        let x_off = w * channels[0].pixel_type.byte_width();
        for y in 0..lines {
            let a = row_slice(&raw, y, x_off, w, ub);
            let b = row_slice(&back, y, x_off, w, ub);
            assert_eq!(a, b, "B44 uint 通道 {w}×{lines} 第 {y} 行失真");
        }
    }
}

/// B44 与 B44A 的**解码路径完全一致**（同一负载解出相同字节）。
#[test]
fn b44a_and_b44_share_the_same_decode_path() {
    let (w, lines) = (8usize, 8usize);
    let channels = vec![ch("R", PixelType::Half)];
    let planes = synth(1, w, lines, PixelType::Half, 31);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = b44_compress(&raw, &ctx, true).unwrap_or_else(|e| panic!("{e}"));
    let via_b44 = decompress_block(Compression::B44, &comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let via_b44a =
        decompress_block(Compression::B44A, &comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(via_b44, via_b44a, "B44 与 B44A 的解码结果必须逐位相同");
    assert_eq!(via_b44a.len(), raw.len());
}

/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。
/// B44：畸形输入结构化拒绝（5 条），不 panic。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。
/// B44/B44A：`pLinear = true` 的 **HALF 通道被结构化拒绝**（不静默走错路径）。
///
/// 该路径依赖 OpenEXR 的 `expTable`/`logTable`——两张各128 KB 的**离线生成
/// 常量表**。本模块既不内嵌也不按公式重算：公式重算在饱和区无法与参考表
/// 逐位对齐（索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
/// 参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。
#[test]
fn b44_p_linear_half_is_structurally_rejected() {
    for comp in [Compression::B44, Compression::B44A] {
        let (w, lines) = (8usize, 8usize);
        let channels = vec![ch_linear("R")];
        let ctx = ctx_of(&channels, w, lines);

        // ① 解码侧：不论负载内容一律给三段式 EXR2xxx 错误。
        let payload = vec![0u8; 16];
        let e = decompress_block(comp, &payload, &ctx).unwrap_err();
        assert_eq!(e.code, "EXR2001", "{comp:?} 拒绝码应为 EXR2001，实得 {e}");
        assert!(
            e.problem.contains("pLinear"),
            "{comp:?} 的 problem 段应点明 pLinear，实得：{}",
            e.problem
        );
        assert!(
            e.fix.contains("pLinear"),
            "{comp:?} 的 fix 段应给出可操作指引，实得：{}",
            e.fix
        );
        assert!(!e.location.is_empty(), "{comp:?} 的 location 段不得为空");

        // ② 混入普通 HALF 通道也不改变判定（拒绝发生在通道级）。
        let mixed = vec![ch("R", PixelType::Half), ch_linear("G")];
        let ctx2 = ctx_of(&mixed, w, lines);
        let e = decompress_block(comp, &payload, &ctx2).unwrap_err();
        assert_eq!(e.code, "EXR2001", "{comp:?} 混合通道表也应被拒：{e}");

        // ③ UINT / FLOAT 通道带 pLinear **不受影响**（B44 只压 HALF，
        //    参考实现对非 HALF 通道根本不看 pLinear）。
        let uint_lin = vec![ExrChannel {
            p_linear: true,
            ..ch("R", PixelType::Uint)
        }];
        let ctx3 = ctx_of(&uint_lin, w, lines);
        let raw_u = vec![0u8; w * lines * 4];
        let ok = decompress_block(comp, &raw_u, &ctx3)
            .unwrap_or_else(|e| panic!("{comp:?} UINT+pLinear 应原样通过，却报错：{e}"));
        assert_eq!(ok, raw_u, "{comp:?} UINT+pLinear 应逐位原样返回");
    }
}

/// B44/B44A：测试侧编码器同样对 `pLinear = true` 的 HALF 通道报结构化错误，
/// 保证"能编出来的测试负载"与"能被解码器接受的负载"边界一致。
#[test]
fn b44_p_linear_encoder_also_refuses() {
    let (w, lines) = (8usize, 8usize);
    let channels = vec![ch_linear("R")];
    let ctx = ctx_of(&channels, w, lines);
    let planes = vec![vec![vec![2.0f32; w]; lines]];
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let e = b44_compress(&raw, &ctx, true).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
    assert!(e.problem.contains("pLinear"), "{e}");
}




#[test]
fn b44_malformed_inputs_are_structured_errors() {
    let (w, lines) = (8usize, 4usize);
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, w, lines);
    let planes = synth(1, w, lines, PixelType::Half, 37);
    let good = b44_compress(&planes_to_raw(&planes, w, &channels, lines), &ctx, true)
        .unwrap_or_else(|e| panic!("{e}"));

    // ① 空负载。
    let e = b44_decompress(&[], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ② 负载被截断（半个 14 字节块）。
    let e = b44_decompress(&good[..good.len() / 2], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ③ 尾部多余字节。
    let mut bad = good.clone();
    bad.extend_from_slice(&[0u8; 5]);
    let e = b44_decompress(&bad, &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ④ 3 字节平场块被截断（只剩 1 字节）。
    let e = b44_decompress(&[0x00, 0x3c], &ctx).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");

    // ⑤ 空通道表 / 零尺寸上下文。
    let empty = ctx_of(&[], w, lines);
    let e = b44_decompress(&good, &empty).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
    let zero = ctx_of(&channels, 0, lines);
    let e = b44_decompress(&good, &zero).unwrap_err();
    assert_eq!(e.code, "EXR2001", "{e}");
}

/// B44：解码确定性（同负载解两遍字节级一致）。
///
/// **不**断言 `a == raw`：本用例含 HALF 通道，B44 对 HALF **格式固有有损**；
/// 只断言（①）两次解码逐位一致，（②）**FLOAT 通道逐位无损**（B44 原样复制）。
#[test]
fn b44_decoding_is_deterministic() {
    let (w, lines) = (12usize, 8usize);
    let channels = vec![ch("R", PixelType::Half), ch("Y", PixelType::Float)];
    let planes = synth(2, w, lines, PixelType::Float, 41);
    let raw = planes_to_raw(&planes, w, &channels, lines);
    let ctx = ctx_of(&channels, w, lines);
    let comp = b44_compress(&raw, &ctx, true).unwrap_or_else(|e| panic!("{e}"));
    let a = b44_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    let b = b44_decompress(&comp, &ctx).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a, b, "B44 解码不确定");
    // FLOAT 通道（第 2 个通道，每像素 4 字节）在「扫描线 × 通道 × 像素」布局下逐行逐位无损。
    let fw = channels[1].pixel_type.byte_width();
    let x_off = w * channels[0].pixel_type.byte_width();
    for y in 0..lines {
        let ra = row_slice(&raw, y, x_off, w, fw);
        let rb = row_slice(&a, y, x_off, w, fw);
        assert_eq!(ra, rb, "B44 FLOAT 通道第 {y} 行失真（应原样复制）");
    }
}

// ===========================================================================
//  4) 跨压缩：分派 / 元数据 / 端到端 EXR 文件
// ===========================================================================

/// 四种压缩的 `is_supported` / `lines_per_block` / `code` 与 OpenEXR 一致。
#[test]
fn four_compressions_metadata_matches_openexr() {
    for (c, code, lpb) in [
        (Compression::Piz, 4u8, 32u32),
        (Compression::Pxr24, 5, 16),
        (Compression::B44, 6, 32),
        (Compression::B44A, 7, 32),
    ] {
        assert!(c.is_supported(), "{} 应标记为已实现", c.code());
        assert!(c.is_exr_compress(), "{} 应走 exr_compress", c.code());
        assert!(!c.is_dwa(), "{} 不是 DWA 族", c.code());
        assert_eq!(c.code(), code);
        assert_eq!(c.lines_per_block(), lpb, "{} 块高不符", c.code());
        assert_eq!(Compression::from_code(code), Some(c));
    }
}

/// 分派表拒载不属于本模块的压缩类型（结构化错误，不 panic）。
#[test]
fn exr_compress_rejects_foreign_compressions() {
    let channels = vec![ch("R", PixelType::Half)];
    let ctx = ctx_of(&channels, 4, 4);
    for c in [
        Compression::None,
        Compression::Rle,
        Compression::Zip,
        Compression::Zips,
        Compression::Dwaa,
        Compression::Dwab,
    ] {
        let e = decompress_block(c, &[0u8; 4], &ctx).unwrap_err();
        assert_eq!(e.code, "EXR2001", "{}: {e}", c.code());
    }
}

/// 端到端：同一图像 → None 编一份、四种压缩各编一份 → 读回**逐位对拍**。
///
/// 这是任务书要求的「最强证据」：同一图像用 `None` 编、`Piz` 编码后再解，
/// 结果应逐位相同。
#[test]
fn end_to_end_four_compressions_match_none_path_bitwise() {
    let (w, h) = (24u32, 40u32); // PIZ/B44 = 32 行/块 ⇒ 2 块；PXR24 = 16 ⇒ 3 块
    let channels = vec![
        ch("R", PixelType::Half),
        ch("G", PixelType::Half),
        ch("B", PixelType::Half),
        ch("Z", PixelType::Float),
        ch("id", PixelType::Uint),
    ];
    let ref_bytes = build_exr_blockwise(w, h, &channels, Compression::None);
    let ref_img = crate::exr::read_image(&ref_bytes).unwrap_or_else(|e| panic!("None: {e}"));
    assert_eq!(ref_img.planes.len(), channels.len());

    for comp in [
        Compression::Piz,
        Compression::Pxr24,
        Compression::B44,
        Compression::B44A,
    ] {
        let bytes = build_exr_blockwise(w, h, &channels, comp);
        let img = crate::exr::read_image(&bytes).unwrap_or_else(|e| panic!("{}: {e}", comp.code()));
        assert_eq!(img.header.compression, comp);
        assert_eq!(img.planes.len(), ref_img.planes.len());

        for (rp, ip) in ref_img.planes.iter().zip(img.planes.iter()) {
            assert_eq!(rp.name, ip.name);
            // B44 的 HALF 通道误差是**块内绝对量化**（每 4×4 块一个 `shift`，6 位残差），
            // 故正确的界是「绝对误差 ≤ 通道峰值的 5%」，而**不是**逐像素相对误差 ——
            // 接近 0 的像素相对误差天然会爆（0.34 的绝对误差 / 值 2 ⇒ 17%）。
            let peak = if matches!(comp, Compression::B44 | Compression::B44A)
                && rp.pixel_type == PixelType::Half
            {
                rp.values.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()))
            } else {
                0.0
            };
            let abs_tol = peak * 0.05;
            for (i, (&a, &b)) in rp.values.iter().zip(ip.values.iter()).enumerate() {
                // PXR24 的 FLOAT 通道格式固有有损（32bit→24bit）。
                if rp.pixel_type == PixelType::Float && comp == Compression::Pxr24 {
                    let rel = ((a - b) as f64).abs() / (a as f64).abs().max(1e-30);
                    assert!(rel < 6.1e-5, "{} {} 像素 {i}: {a} → {b}", comp.code(), rp.name);
                    continue;
                }
                // B44 / B44A 的 HALF 通道格式固有有损（固定 32B→14B 块量化）。
                if rp.pixel_type == PixelType::Half
                    && matches!(comp, Compression::B44 | Compression::B44A)
                {
                    let abs = (a - b) as f64;
                    assert!(
                        abs.abs() <= abs_tol.max(1e-9),
                        "{} {} 像素 {i}: {a} → {b}，绝对误差 {} 超过峰值 {peak} 的 5%",
                        comp.code(),
                        rp.name,
                        abs.abs()
                    );
                    continue;
                }
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{} 通道 {} 像素 {i} 失真：{a} vs {b}",
                    comp.code(),
                    rp.name
                );
            }
        }
    }
}

/// 端到端确定性：同输入写两遍字节级一致；解码两遍也一致。
#[test]
fn end_to_end_encoding_and_decoding_are_deterministic() {
    let (w, h) = (12u32, 20u32);
    let channels = vec![ch("Y", PixelType::Half), ch("id", PixelType::Uint)];
    for comp in [
        Compression::Piz,
        Compression::Pxr24,
        Compression::B44,
        Compression::B44A,
    ] {
        let a = build_exr_blockwise(w, h, &channels, comp);
        let b = build_exr_blockwise(w, h, &channels, comp);
        assert_eq!(a, b, "{} 两次写出须字节相同", comp.code());
        let ia = crate::exr::read_image(&a).unwrap_or_else(|e| panic!("{e}"));
        let ib = crate::exr::read_image(&b).unwrap_or_else(|e| panic!("{e}"));
        for (pa, pb) in ia.planes.iter().zip(ib.planes.iter()) {
            for (va, vb) in pa.values.iter().zip(pb.values.iter()) {
                assert_eq!(va.to_bits(), vb.to_bits(), "{} 解码不确定", comp.code());
            }
        }
    }
}

/// 端到端：块数与 `lines_per_block()` 相符，且末块不满整块也能正确读回。
#[test]
fn end_to_end_block_counts_follow_lines_per_block() {
    let h = 70u32;
    let channels = vec![ch("Y", PixelType::Half)];
    for (comp, expect_blocks) in [
        (Compression::Piz, 3u32),
        (Compression::Pxr24, 5),
        (Compression::B44, 3),
        (Compression::B44A, 3),
    ] {
        let bytes = build_exr_blockwise(8, h, &channels, comp);
        let hdr = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(hdr.block_count(), expect_blocks as usize, "{}", comp.code());
        let img = crate::exr::read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let p = img.planes.first().unwrap_or_else(|| panic!("无平面"));
        assert_eq!(p.values.len(), (8 * h) as usize, "{}", comp.code());
    }
}

/// 端到端：PXR24 的 HALF 通道在真实文件路径上也**逐位无损**。
#[test]
fn end_to_end_pxr24_half_is_bit_exact_through_real_file() {
    let (w, h) = (20u32, 35u32);
    let channels = vec![ch("Y", PixelType::Half), ch("id", PixelType::Uint)];
    let a = build_exr_blockwise(w, h, &channels, Compression::None);
    let b = build_exr_blockwise(w, h, &channels, Compression::Pxr24);
    let ia = crate::exr::read_image(&a).unwrap_or_else(|e| panic!("{e}"));
    let ib = crate::exr::read_image(&b).unwrap_or_else(|e| panic!("{e}"));
    for (pa, pb) in ia.planes.iter().zip(ib.planes.iter()) {
        for (i, (&x, &y)) in pa.values.iter().zip(pb.values.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "PXR24 {} 像素 {i}: {x} vs {y}", pa.name);
        }
    }
}

/// 端到端：PIZ 的全部通道类型在真实文件路径上**逐位无损**。
#[test]
fn end_to_end_piz_all_channel_types_bit_exact() {
    let (w, h) = (18u32, 33u32);
    let channels = vec![
        ch("R", PixelType::Half),
        ch("Z", PixelType::Float),
        ch("id", PixelType::Uint),
    ];
    let a = build_exr_blockwise(w, h, &channels, Compression::None);
    let b = build_exr_blockwise(w, h, &channels, Compression::Piz);
    let ia = crate::exr::read_image(&a).unwrap_or_else(|e| panic!("{e}"));
    let ib = crate::exr::read_image(&b).unwrap_or_else(|e| panic!("{e}"));
    for (pa, pb) in ia.planes.iter().zip(ib.planes.iter()) {
        for (i, (&x, &y)) in pa.values.iter().zip(pb.values.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "PIZ {} 像素 {i}: {x} vs {y}", pa.name);
        }
    }
}

// ===========================================================================
//  端到端 EXR 构造器
// ===========================================================================

fn put_attr(out: &mut Vec<u8>, name: &str, ty: &str, value: &[u8]) {
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(ty.as_bytes());
    out.push(0);
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
}

fn channels_attr(chs: &[ExrChannel]) -> Vec<u8> {
    let mut v = Vec::new();
    for c in chs {
        v.extend_from_slice(c.name.as_bytes());
        v.push(0);
        v.extend_from_slice(&c.pixel_type.code().to_le_bytes());
        v.push(u8::from(c.p_linear));
        v.extend_from_slice(&[0, 0, 0]);
        v.extend_from_slice(&1i32.to_le_bytes());
        v.extend_from_slice(&1i32.to_le_bytes());
    }
    v.push(0);
    v
}

/// 测试用确定性像素图案：`f(i, ci)`（跨零、含负、half 可精确表示）。
fn pixel_value(i: usize, ci: usize, pt: PixelType) -> f32 {
    let base = i as f32 * 0.0625 - 8.0 + (ci as f32) * 5.0;
    match pt {
        PixelType::Uint => ((i + ci * 7) % 97) as f32,
        _ => base,
    }
}

/// 按压缩类型把一个块编成负载。
fn encode_block(comp: Compression, raw: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    match comp {
        Compression::None => Ok(raw.to_vec()),
        Compression::Piz => piz_compress(raw, ctx),
        Compression::Pxr24 => pxr24_compress(raw, ctx),
        Compression::B44 => b44_compress(raw, ctx, false),
        Compression::B44A => b44_compress(raw, ctx, true),
        other => Err(comp_err(
            LOC,
            format!("测试构造器不支持压缩类型 {}", other.code()),
            "只应传 None/PIZ/PXR24/B44/B44A",
        )),
    }
}

/// 构造一个扫描线 EXR 文件（`compression` 决定每块如何编码）。
fn build_exr_blockwise(
    width: u32,
    height: u32,
    channels: &[ExrChannel],
    comp: Compression,
) -> Vec<u8> {
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(&crate::exr::EXR_MAGIC.to_le_bytes());
    head.extend_from_slice(&2u32.to_le_bytes());
    put_attr(&mut head, "channels", "chlist", &channels_attr(channels));
    put_attr(&mut head, "compression", "compression", &[comp.code()]);
    let win = DataWindow::new(width, height);
    let mut dw = Vec::new();
    for v in [win.x_min, win.y_min, win.x_max, win.y_max] {
        dw.extend_from_slice(&v.to_le_bytes());
    }
    put_attr(&mut head, "dataWindow", "box2i", &dw);
    put_attr(&mut head, "displayWindow", "box2i", &dw);
    put_attr(&mut head, "lineOrder", "lineOrder", &[0]);
    head.push(0);

    let lpb = comp.lines_per_block() as usize;
    let w = width as usize;
    let blocks = (height as usize).div_ceil(lpb);
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let lines = (height as usize - b * lpb).min(lpb);
        // 本块的原始字节：逐扫描线 × 逐通道 × 逐像素。
        let mut raw = Vec::with_capacity(w * lines * 8);
        for line in 0..lines {
            for (ci, c) in channels.iter().enumerate() {
                for x in 0..w {
                    let v = pixel_value((b * lpb + line) * w + x, ci, c.pixel_type);
                    raw.extend_from_slice(&quantize(v, c.pixel_type)[..c.pixel_type.byte_width()]);
                }
            }
        }
        let ctx = ctx_of(channels, w, lines);
        let payload = encode_block(comp, &raw, &ctx)
            .unwrap_or_else(|e| panic!("{} 块 {b}: {e}", comp.code()));
        let mut chunk = Vec::with_capacity(payload.len() + 8);
        // 块首行号用**dataWindow 绝对坐标**（读取侧据此校验偏移表一致性）。
        chunk.extend_from_slice(&((b * lpb) as i32).to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        chunk.extend_from_slice(&payload);
        chunks.push(chunk);
    }

    let table_bytes = chunks.len() * 8;
    let mut at = head.len() + table_bytes;
    let mut table = Vec::with_capacity(table_bytes);
    let mut body = Vec::new();
    for c in &chunks {
        table.extend_from_slice(&(at as u64).to_le_bytes());
        body.extend_from_slice(c);
        at += c.len();
    }
    let mut out = head;
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}


/// 小波：奇数 / 小尺寸下的奇列、奇行尾部分支必须互逆。
///
/// **值域契约**（OpenEXR 靠此保证 `w14` 可逆）：`mx < 2^14` 时输入必须落在
/// 14bit 内（PIZ 在小波前先做 LUT，把值域压到 `[0, maxValue]`）；
/// 否则走 `w16` 模运算分支，输入可为全 16bit。
#[test]
fn wav2_odd_dimensions_round_trip_both_branches() {
    for (w, lines) in [(1usize, 1usize), (3, 5), (5, 7), (7, 13), (17, 3)] {
        let n = w * lines;
        // 确定性 LCG 图案。
        let mut seed = 0x9E37_79B9u32;
        let mut gen = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        // ① 14bit 分支：mx = 8191，输入压到 [0, 8191]。
        let mut d14: Vec<u16> = (0..n).map(|_| ((gen() >> 18) & 0x1FFF) as u16).collect();
        let o14 = d14.clone();
        wav2_encode(&mut d14, w, 1, lines, w, 8191).unwrap_or_else(|e| panic!("{w}×{lines} w14 enc: {e}"));
        wav2_decode(&mut d14, w, 1, lines, w, 8191).unwrap_or_else(|e| panic!("{w}×{lines} w14 dec: {e}"));
        assert_eq!(d14, o14, "{w}×{lines} w14 小波非互逆");
        // ② 16bit 分支：mx ≥ 2^14，输入取全 16bit。
        let mut d16: Vec<u16> = (0..n).map(|_| (gen() >> 11) as u16).collect();
        let o16 = d16.clone();
        wav2_encode(&mut d16, w, 1, lines, w, 65535).unwrap_or_else(|e| panic!("{w}×{lines} w16 enc: {e}"));
        wav2_decode(&mut d16, w, 1, lines, w, 65535).unwrap_or_else(|e| panic!("{w}×{lines} w16 dec: {e}"));
        assert_eq!(d16, o16, "{w}×{lines} w16 小波非互逆");
    }
}
