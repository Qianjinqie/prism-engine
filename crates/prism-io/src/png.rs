//! PNG 解码器（非交错，8/16-bit，色型 0/2/3/4/6）—— V1.5-W1-A / P0-1。
//!
//! # 能力矩阵
//!
//! | 项 | 支持 |
//! |----|------|
//! | 签名 | PNG 8 字节签名强校验（错一字节即报错） |
//! | 关键块 | `IHDR` / `PLTE` / `tRNS` / `IDAT`（多块拼接）/ `IEND` |
//! | 辅助块 | 其余块按长度跳过（仍校验 CRC，避免「静默读半张图」） |
//! | 位深 | 8、16 |
//! | 色型 | 0 灰度、2 真彩、3 调色板、4 灰度+alpha、6 真彩+alpha |
//! | 滤波 | 0 None / 1 Sub / 2 Up / 3 Average / 4 Paeth 全覆盖 |
//! | 压缩 | zlib（[`crate::inflate`]：手写 DEFLATE，零第三方 crate） |
//! | 交错 | **不支持**（Adam7 报 L2 明确错误，绝不静默当成非交错解） |
//!
//! # 输出口径（与黄金固件一致）
//!
//! [`PngImage::samples`] 是**反滤波之后的原始样本流**：行主序、逐像素逐通道、
//! 16-bit 为大端、调色板图为索引值——不做任何色彩解释。上层要 RGBA 时调
//! [`PngImage::to_rgba8`] / [`PngImage::to_rgba16`]。

use crate::inflate::zlib_decompress;
use crate::crc32;

/// PNG 8 字节签名。
pub const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// 解码后的 PNG 图像（保留原始样本流，色彩解释延迟到 `to_rgba*`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngImage {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// 位深（8 或 16）。
    pub bit_depth: u8,
    /// 色型（0/2/3/4/6）。
    pub color_type: u8,
    /// 调色板（色型 3 时有效；每项 RGB）。
    pub palette: Vec<[u8; 3]>,
    /// `tRNS` 原始数据（灰度/真彩为 2/6 字节样本值，调色板为 alpha 表）。
    pub trns: Vec<u8>,
    /// 反滤波后的原始样本流（见模块文档的「输出口径」）。
    pub samples: Vec<u8>,
    /// 每行使用的滤波类型（诊断用，长度 = height）。
    pub filters: Vec<u8>,
}

impl PngImage {
    /// 每像素通道数（调色板图为 1，因为样本是索引）。
    pub fn channels(&self) -> usize {
        match self.color_type {
            0 => 1,
            2 => 3,
            3 => 1,
            4 => 2,
            6 => 4,
            _ => 0,
        }
    }

    /// 每像素字节数（16-bit 为通道数 × 2）。
    pub fn bytes_per_pixel(&self) -> usize {
        let per_sample: usize = if self.bit_depth == 16 { 2 } else { 1 };
        self.channels() * per_sample
    }

    /// 单条扫描行的字节数（不含滤波字节）。
    pub fn row_bytes(&self) -> usize {
        (self.width as usize) * self.bytes_per_pixel()
    }

    /// 是否 16-bit。
    pub fn is_16bit(&self) -> bool {
        self.bit_depth == 16
    }

    /// 取第 `i` 个样本（8-bit 图返回 0..=255；16-bit 图返回完整 16-bit 值）。
    #[inline]
    fn sample_at(&self, i: usize) -> u16 {
        if self.is_16bit() {
            let p: usize = i * 2;
            u16::from_be_bytes([self.samples[p], self.samples[p + 1]])
        } else {
            self.samples[i] as u16
        }
    }

    /// 8-bit 输出时单通道取值：16-bit 取高字节（截断镜像，与固件口径一致）。
    #[inline]
    fn sample8(&self, v: u16) -> u8 {
        if self.is_16bit() {
            (v >> 8) as u8
        } else {
            v as u8
        }
    }

    /// 转 RGBA8。
    ///
    /// 16-bit 图按「每通道取高字节」截断（与黄金固件 `.rgba.bin` 的口径一致）。
    /// 灰度 / 灰度+alpha 自动扩展为 RGB 三通道复本。
    pub fn to_rgba8(&self) -> Vec<u8> {
        let n: usize = (self.width as usize) * (self.height as usize);
        let mut out: Vec<u8> = Vec::with_capacity(n * 4);
        match self.color_type {
            // 灰度
            0 => {
                let trns_gray: Option<u16> = if self.trns.len() >= 2 {
                    Some(u16::from_be_bytes([self.trns[0], self.trns[1]]))
                } else {
                    None
                };
                for i in 0..n {
                    let raw: u16 = self.sample_at(i);
                    let v: u8 = self.sample8(raw);
                    let a: u8 = if Some(raw) == trns_gray { 0 } else { 255 };
                    out.extend_from_slice(&[v, v, v, a]);
                }
            }
            // 真彩
            2 => {
                let trns_rgb: Option<[u16; 3]> = if self.trns.len() >= 6 {
                    Some([
                        u16::from_be_bytes([self.trns[0], self.trns[1]]),
                        u16::from_be_bytes([self.trns[2], self.trns[3]]),
                        u16::from_be_bytes([self.trns[4], self.trns[5]]),
                    ])
                } else {
                    None
                };
                for i in 0..n {
                    let r: u16 = self.sample_at(i * 3);
                    let g: u16 = self.sample_at(i * 3 + 1);
                    let b: u16 = self.sample_at(i * 3 + 2);
                    let a: u8 = match trns_rgb {
                        Some(t) if t[0] == r && t[1] == g && t[2] == b => 0,
                        _ => 255,
                    };
                    out.extend_from_slice(&[
                        self.sample8(r),
                        self.sample8(g),
                        self.sample8(b),
                        a,
                    ]);
                }
            }
            // 调色板（samples 存索引）
            3 => {
                for i in 0..n {
                    let idx: usize = self.sample_at(i) as usize;
                    let rgb: [u8; 3] = if idx < self.palette.len() {
                        self.palette[idx]
                    } else {
                        // 索引越界：如实报错不如给可辨识值，但仍保持不 panic。
                        [0u8, 0, 0]
                    };
                    let a: u8 = self.trns.get(idx).copied().unwrap_or(255);
                    out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
                }
            }
            // 灰度 + alpha
            4 => {
                for i in 0..n {
                    let g: u16 = self.sample_at(i * 2);
                    let a: u16 = self.sample_at(i * 2 + 1);
                    let gv: u8 = self.sample8(g);
                    out.extend_from_slice(&[gv, gv, gv, self.sample8(a)]);
                }
            }
            // 真彩 + alpha
            6 => {
                for i in 0..n {
                    let r: u16 = self.sample_at(i * 4);
                    let g: u16 = self.sample_at(i * 4 + 1);
                    let b: u16 = self.sample_at(i * 4 + 2);
                    let a: u16 = self.sample_at(i * 4 + 3);
                    out.extend_from_slice(&[
                        self.sample8(r),
                        self.sample8(g),
                        self.sample8(b),
                        self.sample8(a),
                    ]);
                }
            }
            _ => {}
        }
        out
    }

    /// 转 RGBA16（8-bit 图按 `v*257` 扩展到全量程，16-bit 图直通）。
    pub fn to_rgba16(&self) -> Vec<u16> {
        let n: usize = (self.width as usize) * (self.height as usize);
        let scale = |v: u16| -> u16 { if self.is_16bit() { v } else { v * 257 } };
        let mut out: Vec<u16> = Vec::with_capacity(n * 4);
        match self.color_type {
            0 => {
                for i in 0..n {
                    let v: u16 = scale(self.sample_at(i));
                    out.extend_from_slice(&[v, v, v, 65535]);
                }
            }
            2 => {
                for i in 0..n {
                    out.extend_from_slice(&[
                        scale(self.sample_at(i * 3)),
                        scale(self.sample_at(i * 3 + 1)),
                        scale(self.sample_at(i * 3 + 2)),
                        65535,
                    ]);
                }
            }
            3 => {
                for i in 0..n {
                    let idx: usize = self.sample_at(i) as usize;
                    let rgb: [u8; 3] =
                        if idx < self.palette.len() { self.palette[idx] } else { [0, 0, 0] };
                    let a: u8 = self.trns.get(idx).copied().unwrap_or(255);
                    out.extend_from_slice(&[
                        rgb[0] as u16 * 257,
                        rgb[1] as u16 * 257,
                        rgb[2] as u16 * 257,
                        a as u16 * 257,
                    ]);
                }
            }
            4 => {
                for i in 0..n {
                    let g: u16 = scale(self.sample_at(i * 2));
                    let a: u16 = scale(self.sample_at(i * 2 + 1));
                    out.extend_from_slice(&[g, g, g, a]);
                }
            }
            6 => {
                for i in 0..n {
                    out.extend_from_slice(&[
                        scale(self.sample_at(i * 4)),
                        scale(self.sample_at(i * 4 + 1)),
                        scale(self.sample_at(i * 4 + 2)),
                        scale(self.sample_at(i * 4 + 3)),
                    ]);
                }
            }
            _ => {}
        }
        out
    }

    /// 单通道灰度样本（8-bit 图；16-bit 取高字节）。仅对色型 0/4 有意义。
    pub fn gray_u8(&self) -> Vec<u8> {
        let n: usize = (self.width as usize) * (self.height as usize);
        let step: usize = if self.color_type == 4 { 2 } else { 1 };
        (0..n).map(|i| self.sample8(self.sample_at(i * step))).collect()
    }

    /// 单通道灰度样本（完整 16-bit）。
    pub fn gray_u16(&self) -> Vec<u16> {
        let n: usize = (self.width as usize) * (self.height as usize);
        let step: usize = if self.color_type == 4 { 2 } else { 1 };
        (0..n).map(|i| self.sample_at(i * step)).collect()
    }
}

/// 解码一段 PNG 字节流。
///
/// # 错误（诚实边界，全部带「在哪 / 什么事」）
/// * 签名不符 → 「不是 PNG」；
/// * `IHDR` 非法（尺寸 0、位深非 8/16、色型未知、压缩/滤波方法非 0）→ 明确说明；
/// * 交错（Adam7）→ 「交错 PNG 不支持（L2）」；
/// * CRC 不符 / 块越界 / 缺 IDAT / zlib 失败 → 透传具体位置。
pub fn decode_png(bytes: &[u8]) -> Result<PngImage, String> {
    if bytes.len() < 8 || bytes[0..8] != PNG_SIGNATURE {
        return Err("不是 PNG 字节流（前 8 字节与 PNG 签名不符）".to_string());
    }

    let mut pos: usize = 8usize;
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let mut bit_depth: u8 = 0;
    let mut color_type: u8 = 0;
    let mut saw_ihdr: bool = false;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();

    while pos + 8 <= bytes.len() {
        let len: usize = u32::from_be_bytes([
            bytes[pos],
            bytes[pos + 1],
            bytes[pos + 2],
            bytes[pos + 3],
        ]) as usize;
        let ctype: [u8; 4] = [bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]];
        let data_start: usize = pos + 8;
        let data_end: usize = data_start + len;
        if data_end + 4 > bytes.len() {
            return Err(format!(
                "PNG 块 {} 越界（声明长度 {}，剩余字节 {}）",
                String::from_utf8_lossy(&ctype),
                len,
                bytes.len().saturating_sub(data_start)
            ));
        }
        let data: &[u8] = &bytes[data_start..data_end];

        // CRC32 作用域 = 类型 ++ 数据。
        let mut crc_input: Vec<u8> = Vec::with_capacity(4 + len);
        crc_input.extend_from_slice(&ctype);
        crc_input.extend_from_slice(data);
        let expect_crc: u32 = u32::from_be_bytes([
            bytes[data_end],
            bytes[data_end + 1],
            bytes[data_end + 2],
            bytes[data_end + 3],
        ]);
        let got_crc: u32 = crc32(&crc_input);
        if expect_crc != got_crc {
            return Err(format!(
                "PNG 块 {} CRC 不符（期望 {:08X}，实得 {:08X}）：文件损坏或被截断",
                String::from_utf8_lossy(&ctype),
                expect_crc,
                got_crc
            ));
        }

        match &ctype {
            b"IHDR" => {
                if len != 13 {
                    return Err(format!("IHDR 长度应为 13，实际 {}", len));
                }
                width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                bit_depth = data[8];
                color_type = data[9];
                let compression: u8 = data[10];
                let filter_method: u8 = data[11];
                let interlace: u8 = data[12];
                if width == 0 || height == 0 {
                    return Err(format!("IHDR 尺寸非法（{}x{}）", width, height));
                }
                if compression != 0 {
                    return Err(format!("PNG 压缩方法 {} 不支持（仅 0 = deflate）", compression));
                }
                if filter_method != 0 {
                    return Err(format!("PNG 滤波方法 {} 不支持（仅 0）", filter_method));
                }
                if bit_depth != 8 && bit_depth != 16 {
                    return Err(format!(
                        "PNG 位深 {} 不在 V1.5-A 支持范围（仅 8/16；1/2/4-bit 属 L2）",
                        bit_depth
                    ));
                }
                match color_type {
                    0 | 4 => {
                        if !(bit_depth == 8 || bit_depth == 16) {
                            return Err(format!("灰度色型位深 {} 非法", bit_depth));
                        }
                    }
                    2 | 6 => {}
                    3 => {
                        if bit_depth != 8 {
                            return Err(format!(
                                "调色板色型位深 {} 不在支持范围（仅 8）",
                                bit_depth
                            ));
                        }
                    }
                    other => {
                        return Err(format!("PNG 色型 {} 非法（合法值 0/2/3/4/6）", other));
                    }
                }
                if interlace == 1 {
                    return Err(
                        "交错 PNG（Adam7）不支持（L2）：请先用非交错方式重存后再试".to_string()
                    );
                }
                if interlace != 0 {
                    return Err(format!("PNG 交错方法 {} 非法（合法值 0/1）", interlace));
                }
                saw_ihdr = true;
            }
            b"PLTE" => {
                if len % 3 != 0 {
                    return Err(format!("PLTE 长度 {} 不是 3 的倍数：调色板块损坏", len));
                }
                let mut p: usize = 0;
                while p + 3 <= len {
                    palette.push([data[p], data[p + 1], data[p + 2]]);
                    p += 3;
                }
                if palette.len() > 256 {
                    return Err(format!("PLTE 项数 {} 超过 256", palette.len()));
                }
            }
            b"tRNS" => {
                trns = data.to_vec();
            }
            b"IDAT" => {
                if !saw_ihdr {
                    return Err("IDAT 出现在 IHDR 之前（PNG 结构损坏）".to_string());
                }
                idat.extend_from_slice(data);
            }
            b"IEND" => {
                break;
            }
            _ => {
                // 未知 / 辅助块：按长度跳过（CRC 已校验）。
            }
        }
        pos = data_end + 4;
    }

    if !saw_ihdr {
        return Err("PNG 缺少 IHDR 块".to_string());
    }
    if idat.is_empty() {
        return Err("PNG 缺少 IDAT 块（无图像数据）".to_string());
    }
    if color_type == 3 && palette.is_empty() {
        return Err("调色板 PNG 缺少 PLTE 块".to_string());
    }

    // ---- zlib 解压 ----
    let raw: Vec<u8> = zlib_decompress(&idat)
        .map_err(|e| format!("PNG IDAT zlib 解压失败：{}", e))?;

    let channels: usize = match color_type {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return Err(format!("PNG 色型 {} 非法", color_type)),
    };
    let per_sample: usize = if bit_depth == 16 { 2 } else { 1 };
    let bpp: usize = channels * per_sample;
    let row_bytes: usize = (width as usize) * bpp;
    let need: usize = (row_bytes + 1) * (height as usize);
    if raw.len() < need {
        return Err(format!(
            "PNG 像素数据不足（需 {} 字节，解压得 {}）：文件被截断",
            need,
            raw.len()
        ));
    }

    // ---- 逐行反滤波 ----
    let mut samples: Vec<u8> = vec![0u8; row_bytes * (height as usize)];
    let mut filters: Vec<u8> = Vec::with_capacity(height as usize);
    for y in 0..height as usize {
        let src_off: usize = y * (row_bytes + 1);
        let ftype: u8 = raw[src_off];
        filters.push(ftype);
        let row_in: &[u8] = &raw[src_off + 1..src_off + 1 + row_bytes];
        // 上一行（已反滤波）切片；首行视为全零。
        let (prev_rows, cur_rows) = samples.split_at_mut(y * row_bytes);
        let prev: Option<&[u8]> = if y == 0 { None } else { Some(&prev_rows[(y - 1) * row_bytes..]) };
        let cur: &mut [u8] = &mut cur_rows[..row_bytes];
        unfilter_row(ftype, row_in, prev, cur, bpp).map_err(|e| {
            format!("PNG 第 {} 行反滤波失败：{}", y, e)
        })?;
    }

    Ok(PngImage {
        width,
        height,
        bit_depth,
        color_type,
        palette,
        trns,
        samples,
        filters,
    })
}

/// 单行反滤波（PNG 规范 §9：0 None / 1 Sub / 2 Up / 3 Average / 4 Paeth）。
///
/// # 参数
/// * `ftype` — 滤波类型字节（0..=4）。
/// * `row_in` — 滤波后的行数据（长度 = 行字节数）。
/// * `prev` — 上一行**已反滤波**数据（首行为 `None`，等价全零）。
/// * `cur` — 输出缓冲（长度 = 行字节数）。
/// * `bpp` — 每像素字节数（Sub/Average/Paeth 的左侧参考步长）。
fn unfilter_row(
    ftype: u8,
    row_in: &[u8],
    prev: Option<&[u8]>,
    cur: &mut [u8],
    bpp: usize,
) -> Result<(), String> {
    let n: usize = cur.len();
    if row_in.len() != n {
        return Err(format!("行数据长度 {} 与行字节数 {} 不符", row_in.len(), n));
    }
    // 首行等价于上一行全零。
    let zero_row: [u8; 1] = [0u8];
    let up = |i: usize| -> u8 {
        match prev {
            Some(p) => p[i],
            None => zero_row[0],
        }
    };
    match ftype {
        0 => {
            cur.copy_from_slice(row_in);
        }
        1 => {
            // Sub：X + left
            for i in 0..n {
                let left: u8 = if i >= bpp { cur[i - bpp] } else { 0 };
                cur[i] = row_in[i].wrapping_add(left);
            }
        }
        2 => {
            // Up：X + up
            for i in 0..n {
                cur[i] = row_in[i].wrapping_add(up(i));
            }
        }
        3 => {
            // Average：X + floor((left + up) / 2)
            for i in 0..n {
                let left: u32 = if i >= bpp { cur[i - bpp] as u32 } else { 0 };
                let u: u32 = up(i) as u32;
                cur[i] = row_in[i].wrapping_add(((left + u) / 2) as u8);
            }
        }
        4 => {
            // Paeth：X + PaethPredictor(left, up, upleft)
            for i in 0..n {
                let left: u8 = if i >= bpp { cur[i - bpp] } else { 0 };
                let u: u8 = up(i);
                let ul: u8 = if i >= bpp { up(i - bpp) } else { 0 };
                cur[i] = row_in[i].wrapping_add(paeth(left, u, ul));
            }
        }
        other => {
            return Err(format!("滤波类型 {} 非法（合法值 0..=4）", other));
        }
    }
    Ok(())
}

/// PNG Paeth 预测器（规范 §9.4）：返回距离 `a+b-c` 最近的 `a`/`b`/`c`。
#[inline]
pub fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p: i32 = (a as i32) + (b as i32) - (c as i32);
    let pa: i32 = (p - a as i32).abs();
    let pb: i32 = (p - b as i32).abs();
    let pc: i32 = (p - c as i32).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

// ===========================================================================
//  单元测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 签名不符必须报错（不静默当成空图）。
    #[test]
    fn rejects_bad_signature() {
        let err: String = decode_png(b"not a png at all").unwrap_err();
        assert!(err.contains("不是 PNG"), "错误文案应指明非 PNG，实际：{}", err);
    }

    /// Paeth 预测器规范向量（手算）：a=b=c 时返回 a。
    #[test]
    fn paeth_basic() {
        assert_eq!(paeth(10, 10, 10), 10);
        // p = 5+3-4 = 4；pa=1, pb=1, pc=0 → c 最近 → 4。
        assert_eq!(paeth(5, 3, 4), 4);
    }

    /// 反滤波 Sub：已知输入的恢复结果。
    #[test]
    fn unfilter_sub_roundtrip() {
        let raw = [2u8, 4, 6]; // 滤波后（bpp=1）
        let mut cur = [0u8; 3];
        unfilter_row(1, &raw, None, &mut cur, 1).unwrap();
        assert_eq!(cur, [2, 6, 12]);
    }
}
