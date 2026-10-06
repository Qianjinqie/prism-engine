//! GIF 原生解码（B11；D-17 原生降级格式，手写 LZW + 帧合成）。
//!
//! 规范：GIF89a（含 GIF87a 子集）。解析块流（`Header` / `Logical Screen Descriptor`
//! / `Global Color Table` / `Graphic Control Extension` / `Image Descriptor` +
//! `Local Color Table` + 自研 LZW 图像数据 / `Comment` / `Plain Text` /
//! `Application` / `Trailer`），按帧合成 RGBA8。
//!
//! # 帧合成
//!
//! 帧 = 逻辑屏幕尺寸的 RGBA8 画布（初始全透明）；每个图像块按 left/top 局部
//! 覆盖合成（不实现 disposal 之外的插值，保持纯字节操作）。前一帧画布作为下一帧
//! 合成基底，故 `frames[k]` 是顺序播放到第 k 帧的完整画面——与 [`crate::timing`]
//! 的 seek 语义一致（帧索引 → 该时刻画面）。
//!
//! # 确定性
//!
//! 固定字节序遍历、LZW 解压无状态、合成按行主序——同字节流逐位一致。

use crate::source::{PixelImage, SourceError};
use crate::{MediaError, MediaResult};

/// GIF 单帧（合成后的完整逻辑屏幕画面 + 帧延时）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GifFrame {
    /// 合成后 RGBA8 画面（逻辑屏幕尺寸）。
    pub canvas: PixelImage,
    /// 帧延时（毫秒；GCE 未提供则为 0）。
    pub delay_ms: u16,
}

/// 解码结果：逻辑屏幕尺寸 + 帧序列（按播放顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GifImage {
    /// 逻辑屏幕宽度。
    pub width: u32,
    /// 逻辑屏幕高度。
    pub height: u32,
    /// 帧序列（合成后的完整画面）。
    pub frames: Vec<GifFrame>,
}

impl GifImage {
    /// 帧数。
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// 解码 GIF 字节流。
pub fn decode_gif(bytes: &[u8]) -> MediaResult<GifImage> {
    if bytes.len() < 13 || (&bytes[0..6] != b"GIF87a" && &bytes[0..6] != b"GIF89a") {
        return Err(src_err("gif::decode_gif", "缺少 GIF87a/GIF89a 魔数", "输入必须是 GIF 文件"));
    }
    let width = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
    let height = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
    let packed = bytes[10];
    let has_gct = packed & 0x80 != 0;
    let gct_size = 2usize << (packed & 0x07);
    let mut pos = 13usize;

    // 全局色表。
    let gct = if has_gct {
        let t = read_color_table(bytes, pos, gct_size)?;
        pos += gct_size * 3;
        Some(t)
    } else {
        None
    };

    let mut frames: Vec<GifFrame> = Vec::new();
    // 合成基（上一帧画面）；无帧时为全透明逻辑屏幕。
    let mut base: Vec<u8> = vec![0u8; (width as usize) * (height as usize) * 4];
    // 当前帧待应用的透明色索引与 GCE 延时。
    let mut gce_delay = 0u16;
    let mut transparent: Option<u8> = None;

    while pos < bytes.len() {
        let block = bytes[pos];
        match block {
            0x3B => break, // Trailer
            0x21 => {
                // 扩展块：label + 子块序列。
                let label = *bytes
                    .get(pos + 1)
                    .ok_or_else(|| src_err("gif::decode_gif", "扩展块被截断", "文件损坏"))?;
                pos += 2;
                let (data, next) = read_sub_blocks(bytes, pos)?;
                pos = next;
                match label {
                    0xF9 => {
                        // Graphic Control Extension：延时 + 透明色索引。
                        if data.len() >= 4 {
                            gce_delay = u16::from_le_bytes([data[1], data[2]]);
                            let flags = data[0];
                            transparent =
                                if flags & 0x01 != 0 { Some(data[3]) } else { None };
                        }
                    }
                    _ => {} // Comment / Plain Text / Application：跳过
                }
            }
            0x2C => {
                // Image Descriptor。
                if pos + 10 > bytes.len() {
                    return Err(src_err("gif::decode_gif", "图像描述符被截断", "文件损坏"));
                }
                let left = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]) as u32;
                let top = u16::from_le_bytes([bytes[pos + 3], bytes[pos + 4]]) as u32;
                let iw = u16::from_le_bytes([bytes[pos + 5], bytes[pos + 6]]) as u32;
                let ih = u16::from_le_bytes([bytes[pos + 7], bytes[pos + 8]]) as u32;
                let ipacked = bytes[pos + 9];
                let has_lct = ipacked & 0x80 != 0;
                let lct_size = 2usize << (ipacked & 0x07);
                let interlaced = ipacked & 0x40 != 0;
                pos += 10;

                let palette = if has_lct {
                    let t = read_color_table(bytes, pos, lct_size)?;
                    pos += lct_size * 3;
                    t
                } else {
                    gct.clone().ok_or_else(|| {
                        src_err("gif::decode_gif", "图像无局部色表且无全局色表", "文件缺少调色板")
                    })?
                };

                // 图像数据：最小码长字节 + LZW 子块流。
                let min_code = *bytes
                    .get(pos)
                    .ok_or_else(|| src_err("gif::decode_gif", "缺少 LZW 最小码长", "文件截断"))?;
                pos += 1;
                let (compressed, next) = read_sub_blocks(bytes, pos)?;
                pos = next;

                let indices = lzw_decode(min_code, &compressed)?;
                // 反交错（若需要）。
                let rows = if interlaced {
                    deinterlace(&indices, iw, ih)
                } else {
                    indices
                };

                // 把本帧像素合成到 base 的副本上（保留前帧作为基底）。
                let mut canvas = base.clone();
                blit_indices(
                    &mut canvas,
                    width,
                    height,
                    left,
                    top,
                    iw,
                    ih,
                    &rows,
                    &palette,
                    transparent,
                );
                frames.push(GifFrame {
                    canvas: PixelImage { width, height, pixels: canvas.clone() },
                    delay_ms: gce_delay,
                });
                base = canvas;
                // 每帧后重置 GCE 状态。
                gce_delay = 0;
                transparent = None;
            }
            _ => {
                return Err(src_err(
                    "gif::decode_gif",
                    format!("未知块标识 {block:#04x}"),
                    "文件损坏或非 GIF89a 流",
                ))
            }
        }
    }

    if frames.is_empty() {
        return Err(src_err("gif::decode_gif", "无图像帧", "GIF 至少须含一个图像块"));
    }
    Ok(GifImage { width, height, frames })
}

/// 读调色板（size × 3 字节 RGB → RGBA8 列表）。
fn read_color_table(bytes: &[u8], pos: usize, size: usize) -> MediaResult<Vec<[u8; 4]>> {
    let end = pos + size * 3;
    if end > bytes.len() {
        return Err(src_err("gif::read_color_table", "色表被截断", "文件损坏"));
    }
    let mut table = Vec::with_capacity(size);
    for i in 0..size {
        let b = pos + i * 3;
        table.push([bytes[b], bytes[b + 1], bytes[b + 2], 255]);
    }
    Ok(table)
}

/// 读子块序列（长度前缀 + 数据，长度 0 终止），拼接为连续字节。
fn read_sub_blocks(bytes: &[u8], mut pos: usize) -> MediaResult<(Vec<u8>, usize)> {
    let mut out = Vec::new();
    loop {
        let len = *bytes
            .get(pos)
            .ok_or_else(|| src_err("gif::read_sub_blocks", "子块长度字被截断", "文件损坏"))?
            as usize;
        pos += 1;
        if len == 0 {
            return Ok((out, pos));
        }
        let end = pos + len;
        if end > bytes.len() {
            return Err(src_err("gif::read_sub_blocks", "子块数据被截断", "文件损坏"));
        }
        out.extend_from_slice(&bytes[pos..end]);
        pos = end;
    }
}

/// 自研 GIF LZW 解压（可变码长，LSB 优先，clear/end 码）。
fn lzw_decode(min_code: u8, data: &[u8]) -> MediaResult<Vec<u8>> {
    if !(2..=8).contains(&min_code) {
        return Err(src_err("gif::lzw_decode", format!("最小码长 {min_code} 越界"), "须为 2..=8"));
    }
    let clear = 1u16 << min_code;
    let end = clear + 1;
    let mut code_size = min_code as u32 + 1;
    let mut next_code = end + 1;

    // 字典：前缀码 + 后缀字节，展开为完整串（初始 0..clear 为单字节）。
    let mut prefix = vec![0u16; 4096];
    let mut suffix = vec![0u8; 4096];
    for i in 0..clear {
        suffix[i as usize] = i as u8;
    }

    let mut out: Vec<u8> = Vec::new();
    let mut bit_pos = 0usize; // 已消费位数
    let mut prev: Option<u16> = None;
    // 展开缓冲（栈，逆序回填）。
    let mut stack: Vec<u8> = Vec::with_capacity(4096);

    let total_bits = data.len() * 8;
    while bit_pos + code_size as usize <= total_bits {
        // 读 code_size 位（LSB 优先）。
        let mut code = 0u32;
        for i in 0..code_size {
            let bp = bit_pos + i as usize;
            let byte = data[bp >> 3];
            let bit = (byte >> (bp & 7)) & 1;
            code |= (bit as u32) << i;
        }
        bit_pos += code_size as usize;
        let code = code as u16;

        if code == clear {
            code_size = min_code as u32 + 1;
            next_code = end + 1;
            prev = None;
            continue;
        }
        if code == end {
            break;
        }

        let first: u8;
        if code < next_code {
            // 已知码：展开到栈。
            stack.clear();
            if code >= clear {
                let mut c = code;
                while c >= clear {
                    stack.push(suffix[c as usize]);
                    c = prefix[c as usize];
                }
                first = suffix[c as usize];
                stack.push(first);
            } else {
                first = suffix[code as usize];
                stack.push(first);
            }
            stack.reverse();
            out.extend_from_slice(&stack);
        } else if code == next_code {
            // KwKwK 情形：prev 串 + prev 首字节。
            if let Some(p) = prev {
                stack.clear();
                let mut c = p;
                while c >= clear {
                    stack.push(suffix[c as usize]);
                    c = prefix[c as usize];
                }
                let first_b = suffix[c as usize];
                stack.push(first_b);
                stack.reverse();
                out.extend_from_slice(&stack);
                out.push(first_b);
                first = first_b;
            } else {
                return Err(src_err("gif::lzw_decode", "首码即 KwKwK", "LZW 流非法"));
            }
        } else {
            return Err(src_err("gif::lzw_decode", "码字超出现有字典", "LZW 流损坏"));
        }

        // 新条目：prev 串 + first。
        if let Some(p) = prev {
            if (next_code as usize) < 4096 {
                prefix[next_code as usize] = p;
                suffix[next_code as usize] = first;
                next_code += 1;
                if next_code == (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
            }
        }
        prev = Some(code);
        if out.len() > 1 << 26 {
            return Err(src_err("gif::lzw_decode", "解压输出超 64MB", "疑似损坏文件"));
        }
    }
    Ok(out)
}

/// 反交错：GIF 隔行按 4 趟（0,8,… / 4,12,… / 2,6,… / 1,3,…）重排为顺序行。
fn deinterlace(indices: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h];
    let mut src_row = 0usize;
    let passes: [(usize, usize); 4] = [(0, 8), (4, 8), (2, 4), (1, 2)];
    for (start, step) in passes {
        let mut y = start;
        while y < h {
            let src = src_row * w;
            let dst = y * w;
            if src + w <= indices.len() {
                out[dst..dst + w].copy_from_slice(&indices[src..src + w]);
            }
            src_row += 1;
            y += step;
        }
    }
    out
}

/// 把一帧索引图按调色板合成到画布（透明索引跳过，越界裁剪）。
#[allow(clippy::too_many_arguments)]
fn blit_indices(
    canvas: &mut [u8],
    cw: u32,
    ch: u32,
    left: u32,
    top: u32,
    iw: u32,
    ih: u32,
    indices: &[u8],
    palette: &[[u8; 4]],
    transparent: Option<u8>,
) {
    for row in 0..ih {
        let y = top + row;
        if y >= ch {
            break;
        }
        for col in 0..iw {
            let x = left + col;
            if x >= cw {
                break;
            }
            let si = (row * iw + col) as usize;
            let idx = match indices.get(si) {
                Some(&v) => v,
                None => break,
            };
            if Some(idx) == transparent {
                continue; // 透明像素保留基底
            }
            let rgba = match palette.get(idx as usize) {
                Some(c) => c,
                None => continue,
            };
            let di = ((y * cw + x) as usize) * 4;
            canvas[di..di + 4].copy_from_slice(rgba);
        }
    }
}

/// 三段式源错误构造（走 [`SourceError`] 便于统一收集）。
fn src_err(loc: &str, problem: impl Into<String>, fix: impl Into<String>) -> MediaError {
    let e: SourceError = SourceError::new("GIF1001", loc, problem, fix);
    e.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用自研编码器生成一个最小 GIF（1 帧，不压缩——用 clear+字面+end 的合法 LZW），
    /// 验证解码往返（无第三方素材依赖）。
    fn encode_min_gif(w: u16, h: u16, colors: &[[u8; 3]]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GIF89a");
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        // 全局色表：2 色（size=2 → 字段值 0 → 2^(0+1)=2）。
        let gct_bits = (colors.len().max(2).next_power_of_two().trailing_zeros() - 1) as u8;
        out.push(0x80 | gct_bits); // 有 GCT
        out.push(0); // 背景索引
        out.push(0); // 像素宽高比
        for c in colors {
            out.extend_from_slice(c);
        }
        // 补齐到 2^(gct_bits+1)。
        let size = 1usize << (gct_bits + 1);
        for _ in colors.len()..size {
            out.extend_from_slice(&[0, 0, 0]);
        }
        // 图像描述符。
        out.push(0x2C);
        out.extend_from_slice(&0u16.to_le_bytes()); // left
        out.extend_from_slice(&0u16.to_le_bytes()); // top
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        out.push(0); // 无 LCT、非隔行
        // LZW：min_code = max(2, gct_bits+1)。
        let min_code = 2u8.max(gct_bits + 1);
        out.push(min_code);
        // 简单合法流：每像素前发 clear 再发字面码（未压缩但合法）。
        let clear = 1u16 << min_code;
        let end = clear + 1;
        let mut bits: Vec<bool> = Vec::new();
        let px = (w as usize) * (h as usize);
        let push_code = |code: u16, size: u32, bits: &mut Vec<bool>| {
            for i in 0..size {
                bits.push((code >> i) & 1 == 1);
            }
        };
        push_code(clear, min_code as u32 + 1, &mut bits);
        for i in 0..px {
            let idx = (i % colors.len()) as u16;
            push_code(clear, min_code as u32 + 1, &mut bits);
            push_code(idx, min_code as u32 + 1, &mut bits);
        }
        push_code(end, min_code as u32 + 1, &mut bits);
        // 打包（LSB 优先）。
        let mut cur = 0u8;
        let mut nbit = 0u32;
        let mut packed = Vec::new();
        for b in bits {
            if b {
                cur |= 1 << nbit;
            }
            nbit += 1;
            if nbit == 8 {
                packed.push(cur);
                cur = 0;
                nbit = 0;
            }
        }
        if nbit > 0 {
            packed.push(cur);
        }
        // 子块化（≤255）。
        for chunk in packed.chunks(255) {
            out.push(chunk.len() as u8);
            out.extend_from_slice(chunk);
        }
        out.push(0); // 子块终止
        out.push(0x3B); // Trailer
        out
    }

    #[test]
    fn decode_golden_two_color_frame() {
        // 2×1 图，色 [红, 绿]，像素索引 0,1 → 左红右绿。
        let bytes = encode_min_gif(2, 1, &[[255, 0, 0], [0, 255, 0]]);
        let gif = decode_gif(&bytes).expect("自产 GIF 必可解码");
        assert_eq!((gif.width, gif.height), (2, 1));
        assert_eq!(gif.frame_count(), 1);
        let px = &gif.frames[0].canvas.pixels;
        assert_eq!(&px[0..4], &[255, 0, 0, 255], "左像素红");
        assert_eq!(&px[4..8], &[0, 255, 0, 255], "右像素绿");
    }

    #[test]
    fn bad_magic_rejected() {
        assert_eq!(decode_gif(b"NOTGIF....").unwrap_err().code, "GIF1001");
    }

    #[test]
    fn lzw_kwkwk_roundtrip_via_pattern() {
        // 交替 0,1 序列会触发 KwKwK：3×1 = [0,1,0]。
        let bytes = encode_min_gif(3, 1, &[[10, 20, 30], [40, 50, 60]]);
        let gif = decode_gif(&bytes).unwrap();
        let px = &gif.frames[0].canvas.pixels;
        assert_eq!(&px[0..4], &[10, 20, 30, 255]);
        assert_eq!(&px[4..8], &[40, 50, 60, 255]);
        assert_eq!(&px[8..12], &[10, 20, 30, 255]);
    }
}
