//! APNG 原生解码（B11；D-17 原生降级格式）。
//!
//! 规范：PNG（RFC 2083）容器 + APNG 动画扩展（acTL/fcTL/fdAT）。解析块流，
//! 用 [`crate::inflate::zlib_decompress`] 解压 `IDAT`/`fdAT`，按 PNG 滤波
//! （None/Sub/Up/Average/Paeth）重建像素，逐帧合成 RGBA8。
//!
//! # 帧合成
//!
//! 帧序列：`IDAT` 前的 `fcTL` 描述默认图（帧 0），后续 `fcTL`+`fdAT` 描述后续帧；
//! 每帧按 `x_offset/y_offset` 覆盖到画布（前一帧作合成基底，语义同 GIF）。
//! 支持 8 位灰度/RGB/调色板/灰度+alpha/RGBA 五种颜色类型。
//!
//! # 确定性
//!
//! 固定块序 + 滤波固定逆序（Sub/Up/Average/Paeth 逆运算）+ inflate 无状态——
//! 同字节流逐位一致。

use crate::inflate::zlib_decompress;
use crate::source::{PixelImage, SourceError};
use crate::{MediaError, MediaResult};

/// APNG 单帧。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApngFrame {
    /// 合成后 RGBA8 画面。
    pub canvas: PixelImage,
    /// 帧延时分数（分子/分母，秒；fcTL 提供）。
    pub delay_num: u16,
    /// 帧延时分母。
    pub delay_den: u16,
}

/// APNG 解码结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApngImage {
    /// 画布宽。
    pub width: u32,
    /// 画布高。
    pub height: u32,
    /// 是否含动画（`acTL` 存在且帧数 > 1）。
    pub animated: bool,
    /// 帧序列。
    pub frames: Vec<ApngFrame>,
}

/// PNG 头（IHDR）信息。
#[derive(Debug, Clone, Copy)]
struct Ihdr {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    channels: usize,
}

/// 一个待解码帧的数据（原始压缩字节，可能为多段）。
#[derive(Debug, Clone, Default)]
struct RawFrame {
    /// 帧尺寸与偏移（fcTL）。
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    delay_num: u16,
    delay_den: u16,
    /// 拼接后的压缩图像数据。
    data: Vec<u8>,
}

/// 解码 APNG（或静态 PNG）。
pub fn decode_apng(bytes: &[u8]) -> MediaResult<ApngImage> {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() < 8 || bytes[0..8] != SIG {
        return Err(src_err("apng::decode_apng", "缺少 PNG 魔数", "输入必须是 PNG/APNG"));
    }
    let mut pos = 8usize;
    let mut ihdr: Option<Ihdr> = None;
    let mut plte: Option<Vec<[u8; 4]>> = None;
    let mut trns: Option<Vec<u8>> = None; // 调色板 alpha 或灰度/真彩透明色
    let mut frames: Vec<RawFrame> = Vec::new();
    let mut cur: Option<RawFrame> = None;
    let mut animated = false;

    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let ctype = &bytes[pos + 4..pos + 8];
        let body_start = pos + 8;
        let body_end = body_start + len;
        if body_end + 4 > bytes.len() {
            return Err(src_err("apng::decode_apng", "块体越界（文件截断）", "文件损坏"));
        }
        let body = &bytes[body_start..body_end];
        pos = body_end + 4; // 跳过 CRC

        match ctype {
            b"IHDR" => {
                if len < 13 {
                    return Err(src_err("apng::decode_apng", "IHDR 不足 13 字节", "文件损坏"));
                }
                let width = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                let height = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                let bit_depth = body[8];
                let color_type = body[9];
                let channels = match color_type {
                    0 => 1, // 灰度
                    2 => 3, // RGB
                    3 => 1, // 调色板
                    4 => 2, // 灰度 + alpha
                    6 => 4, // RGBA
                    other => {
                        return Err(src_err(
                            "apng::decode_apng",
                            format!("颜色类型 {other} 不支持"),
                            "支持 0/2/3/4/6（灰度/RGB/调色板/灰度A/RGBA）",
                        ))
                    }
                };
                ihdr = Some(Ihdr { width, height, bit_depth, color_type, channels });
                if width == 0 || height == 0 {
                    return Err(src_err("apng::decode_apng", "宽或高为 0", "IHDR 非法"));
                }
            }
            b"acTL" => {
                if len >= 4 {
                    let nframes = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    animated = nframes > 1;
                }
            }
            b"PLTE" => {
                let mut t = Vec::with_capacity(len / 3);
                for c in body.chunks_exact(3) {
                    t.push([c[0], c[1], c[2], 255]);
                }
                plte = Some(t);
            }
            b"tRNS" => {
                trns = Some(body.to_vec());
            }
            b"fcTL" => {
                // 结束上一帧：若已在帧内则提交。
                if let Some(f) = cur.take() {
                    frames.push(f);
                }
                if len < 26 {
                    return Err(src_err("apng::decode_apng", "fcTL 不足 26 字节", "文件损坏"));
                }
                let w = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                let h = u32::from_be_bytes([body[8], body[9], body[10], body[11]]);
                let x = u32::from_be_bytes([body[12], body[13], body[14], body[15]]);
                let y = u32::from_be_bytes([body[16], body[17], body[18], body[19]]);
                let delay_num = u16::from_be_bytes([body[20], body[21]]);
                let delay_den = u16::from_be_bytes([body[22], body[23]]);
                cur = Some(RawFrame {
                    x,
                    y,
                    w,
                    h,
                    delay_num,
                    delay_den,
                    data: Vec::new(),

                });
            }
            b"IDAT" => {
                // 默认图数据；无 fcTL 时为单独静态帧。
                if cur.is_none() {
                    let ih = ihdr.ok_or_else(|| {
                        src_err("apng::decode_apng", "IDAT 出现在 IHDR 之前", "文件损坏")
                    })?;
                    cur = Some(RawFrame {
                        x: 0,
                        y: 0,
                        w: ih.width,
                        h: ih.height,
                        delay_num: 0,
                        delay_den: 0,
                        data: Vec::new(),

                    });
                }
                if let Some(f) = cur.as_mut() {
                    f.data.extend_from_slice(body);
                }
            }
            b"fdAT" => {
                if len < 4 {
                    return Err(src_err("apng::decode_apng", "fdAT 不足 4 字节", "文件损坏"));
                }
                if let Some(f) = cur.as_mut() {
                    f.data.extend_from_slice(&body[4..]); // 跳过序列号
                }
            }
            b"IEND" => {
                if let Some(f) = cur.take() {
                    frames.push(f);
                }
                break;
            }
            _ => {} // 未知/辅助块跳过
        }
    }
    if let Some(f) = cur.take() {
        frames.push(f);
    }

    let ih = ihdr.ok_or_else(|| src_err("apng::decode_apng", "缺少 IHDR", "文件损坏"))?;
    if frames.is_empty() {
        return Err(src_err("apng::decode_apng", "无图像帧", "PNG 至少须含 IDAT"));
    }

    // 逐帧解码并合成。
    let mut base: Vec<u8> = vec![0u8; (ih.width as usize) * (ih.height as usize) * 4];
    let mut out_frames: Vec<ApngFrame> = Vec::with_capacity(frames.len());
    for (i, rf) in frames.iter().enumerate() {
        let raw = zlib_decompress(&rf.data)?;
        let rgba = reconstruct(&ih, &raw, &plte, &trns, rf.w, rf.h)?;
        let mut canvas = base.clone();
        blit_rgba(&mut canvas, ih.width, ih.height, rf.x, rf.y, rf.w, rf.h, &rgba);
        out_frames.push(ApngFrame {
            canvas: PixelImage { width: ih.width, height: ih.height, pixels: canvas.clone() },
            delay_num: rf.delay_num,
            delay_den: rf.delay_den,
        });
        base = canvas;
        let _ = i;
    }

    Ok(ApngImage { width: ih.width, height: ih.height, animated, frames: out_frames })
}

/// 反滤波 + 反交织为 RGBA8（`w × h` 帧），再按调色板/tRNS 转为 RGBA。
fn reconstruct(
    ih: &Ihdr,
    raw: &[u8],
    plte: &Option<Vec<[u8; 4]>>,
    trns: &Option<Vec<u8>>,
    w: u32,
    h: u32,
) -> MediaResult<Vec<u8>> {
    let bpp = ih.channels * (ih.bit_depth as usize).div_ceil(8);
    let row_bytes = (w as usize) * bpp;
    let expected = (row_bytes + 1) * (h as usize);
    if raw.len() < expected {
        return Err(src_err(
            "apng::reconstruct",
            format!("图像数据 {} 字节，期望至少 {expected}", raw.len()),
            "文件损坏或帧尺寸不符",
        ));
    }

    // 反滤波：逐行按 filter 类型逆运算（前一行是已还原数据）。
    let mut img: Vec<u8> = vec![0u8; row_bytes * (h as usize)];
    let mut prev_row = vec![0u8; row_bytes];
    for y in 0..h as usize {
        let filter = raw[y * (row_bytes + 1)];
        let src = &raw[y * (row_bytes + 1) + 1..y * (row_bytes + 1) + 1 + row_bytes];
        let mut row = src.to_vec();
        unfilter(filter, bpp, &mut row, &prev_row)?;
        img[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(&row);
        prev_row = row;
    }

    // 反交织为 RGBA8。
    let px = (w as usize) * (h as usize);
    let mut out = vec![0u8; px * 4];
    for i in 0..px {
        let s = i * bpp;
        let d = i * 4;
        out[d..d + 4].copy_from_slice(&expand_pixel(ih, plte, trns, &img[s..s + bpp])?);
    }
    Ok(out)
}

/// 单像素按颜色类型展开为 RGBA8。
fn expand_pixel(
    ih: &Ihdr,
    plte: &Option<Vec<[u8; 4]>>,
    trns: &Option<Vec<u8>>,
    s: &[u8],
) -> MediaResult<[u8; 4]> {
    if ih.bit_depth != 8 {
        return Err(src_err(
            "apng::expand_pixel",
            format!("位深 {} 不支持（仅 8 位）", ih.bit_depth),
            "用 8 位 PNG/APNG 重新导出",
        ));
    }
    match ih.color_type {
        0 => {
            // 灰度 + 可选 tRNS 透明色。
            let g = s[0];
            let a = match trns {
                Some(t) if t.len() >= 2 && t[0] == g && t[1] == 0 => 0,
                _ => 255,
            };
            Ok([g, g, g, a])
        }
        2 => Ok([s[0], s[1], s[2], 255]),
        3 => {
            let idx = s[0] as usize;
            let base = plte
                .as_ref()
                .and_then(|t| t.get(idx).copied())
                .unwrap_or([0, 0, 0, 255]);
            let a = trns
                .as_ref()
                .and_then(|t| t.get(idx).copied())
                .unwrap_or(255);
            Ok([base[0], base[1], base[2], a])
        }
        4 => Ok([s[0], s[0], s[0], s[1]]),
        _ => Ok([s[0], s[1], s[2], s[3]]),
    }
}

/// PNG 反滤波（RFC 2083 §6）：None/Sub/Up/Average/Paeth，逐字节模 256。
fn unfilter(filter: u8, bpp: usize, row: &mut [u8], prev: &[u8]) -> MediaResult<()> {
    let n = row.len();
    match filter {
        0 => {} // None
        1 => {
            // Sub
            for i in bpp..n {
                row[i] = row[i].wrapping_add(row[i - bpp]);
            }
        }
        2 => {
            // Up
            for i in 0..n {
                row[i] = row[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            // Average
            for i in 0..n {
                let a = if i >= bpp { row[i - bpp] as u32 } else { 0 };
                let b = prev[i] as u32;
                row[i] = row[i].wrapping_add(((a + b) / 2) as u8);
            }
        }
        4 => {
            // Paeth
            for i in 0..n {
                let a = if i >= bpp { row[i - bpp] as i32 } else { 0 };
                let b = prev[i] as i32;
                let c = if i >= bpp { prev[i - bpp] as i32 } else { 0 };
                row[i] = row[i].wrapping_add(paeth(a, b, c) as u8);
            }
        }
        other => {
            return Err(src_err(
                "apng::unfilter",
                format!("滤波类型 {other} 非法"),
                "PNG 滤波类型须为 0..=4",
            ))
        }
    }
    Ok(())
}

/// Paeth 预测子（RFC 2083 §6.6）。
fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// 把 `rw × rh` 的 RGBA8 帧按 (x,y) 覆盖到画布（越界裁剪，alpha 直通覆盖）。
#[allow(clippy::too_many_arguments)]
fn blit_rgba(
    canvas: &mut [u8],
    cw: u32,
    ch: u32,
    x: u32,
    y: u32,
    rw: u32,
    rh: u32,
    src: &[u8],
) {
    for row in 0..rh {
        let dy = y + row;
        if dy >= ch {
            break;
        }
        for col in 0..rw {
            let dx = x + col;
            if dx >= cw {
                break;
            }
            let si = ((row * rw + col) as usize) * 4;
            if si + 4 > src.len() {
                continue;
            }
            let rgba = &src[si..si + 4];
            if rgba[3] == 0 {
                continue; // 全透明保留基底
            }
            let di = ((dy * cw + dx) as usize) * 4;
            canvas[di..di + 4].copy_from_slice(rgba);
        }
    }
}

fn src_err(loc: &str, problem: impl Into<String>, fix: impl Into<String>) -> MediaError {
    let e: SourceError = SourceError::new("APNG1001", loc, problem, fix);
    e.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflate::adler32;

    /// 直接构造 zlib stored 流（CMF/FLG + FCHECK 补齐），供无压缩 PNG 数据使用。
    ///
    /// **必须补上尾部 4 字节 Adler-32**（RFC 1950 §2.2）：真实 PNG 的 `IDAT`
    /// 一定带校验和，缺失即 spec 违规。此前本 fixture 漏掉了尾部，靠
    /// `prism-media` 旧 inflate「不校验尾部存在性」的宽松行为蒙混过关——
    /// `prism-bytes` 按 RFC 校尾部长度后，这里立刻暴露。
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let cmf = 0x78u8;
        // 选 FLG 使 (cmf<<8|flg) % 31 == 0 且无 FDICT。
        let mut flg = 0u8;
        while (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
            flg += 1;
        }
        out.push(cmf);
        out.push(flg);
        out.push(0x01); // BFINAL=1, BTYPE=00（低 3 位；后续位对齐到下一字节）
        let len = payload.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&adler32(payload).to_be_bytes()); // 尾部 Adler-32
        out
    }

    fn chunk(out: &mut Vec<u8>, ctype: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(ctype);
        out.extend_from_slice(body);
        out.extend_from_slice(&[0, 0, 0, 0]); // CRC（本解码器不校验 CRC）
    }

    /// 生成 2×2 RGBA8 静态 PNG（filter=None，stored zlib）。
    fn encode_png() -> Vec<u8> {
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.push(8); // bit depth
        ihdr.push(6); // RGBA
        ihdr.extend_from_slice(&[0, 0, 0]); // 压缩/滤波/交织
        chunk(&mut out, b"IHDR", &ihdr);
        // 每行：filter byte 0 + w*bpp bytes。2 行 × (1 + 8) = 18 字节。
        let mut raw = Vec::new();
        for y in 0..2u32 {
            raw.push(0); // filter None
            for x in 0..2u32 {
                let v = (y * 2 + x) as u8 * 10;
                raw.extend_from_slice(&[v, v + 1, v + 2, 255]);
            }
        }
        chunk(&mut out, b"IDAT", &zlib_stored(&raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    fn decode_static_png_golden() {
        let bytes = encode_png();
        let img = decode_apng(&bytes).expect("自产 PNG 必可解码");
        assert_eq!((img.width, img.height), (2, 2));
        assert!(!img.animated);
        assert_eq!(img.frame_count_test(), 1);
        let px = &img.frames[0].canvas;
        assert_eq!(px.get(0, 0), [0, 1, 2, 255]);
        assert_eq!(px.get(1, 1), [30, 31, 32, 255]);
    }

    #[test]
    fn paeth_predictor_matches_reference() {
        // RFC 2083 §6.6 参考点。
        assert_eq!(paeth(1, 1, 1), 1);
        assert_eq!(paeth(0, 0, 0), 0);
        assert_eq!(paeth(10, 20, 15), 15, "p=15 时三点距离相同取 a 优先级");
    }

    #[test]
    fn bad_magic_rejected() {
        assert_eq!(decode_apng(b"not png data here").unwrap_err().code, "APNG1001");
    }

    impl ApngImage {
        /// 测试辅助：帧数。
        fn frame_count_test(&self) -> usize {
            self.frames.len()
        }
    }
}
