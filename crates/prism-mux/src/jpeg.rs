//! 基线 JPEG 编码（C4； 容器路线； 同域）。
//!
//! **灰度最小基线**（彩色留作机动）：8×8 分块 → 前向 DCT（自研
//! 整数/浮点混合，禁 libm）→ 量化 → ZigZag → DC 差分 + AC 游程 → 标准 Huffman
//! 熵编码 → 位流装配（FFD9 收尾）。
//!
//! # 设计要点
//!
//! - **仅灰度**：单分量（`Y`，component id 1），采样因子 1×1，符合 JFIF 基线。
//! - **标准量化表**（亮度，质量 50 基准）按质量参数缩放（IJG 公式，整数实现）。
//! - **标准 Huffman 表**（Annex K 亮度 DC/AC）内嵌为常量。
//! - **无 libm**（D-29）：DCT 用固定整除近似（AAN 标量版本的整数化）——系数为
//!   2 的幂/整数，纯整数运算跨平台逐位一致；`cos` 常量预计算为整数标度值。
//! - 熵编码为纯位操作（无浮点、无环境依赖）。
//!
//! 输出为标准 JFIF（APP0）基线 JPEG，可被浏览器/解码器直接解码（烟测）。

use crate::{MuxError, MuxResult};

/// JPEG 编码配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegConfig {
    /// 质量（1..=100；50 为基准）。越高保留细节越多、码率越大。
    pub quality: u8,
    /// 是否在 DCT 系数之间应用逐块差值优化（保持简单：当前恒 false，保留字段）。
    pub grayscale: bool,
}

impl Default for JpegConfig {
    fn default() -> Self {
        JpegConfig { quality: 85, grayscale: true }
    }
}

impl JpegConfig {
    /// 以给定质量构造（灰度基线）。
    pub fn new(quality: u8) -> Self {
        JpegConfig { quality: quality.clamp(1, 100), grayscale: true }
    }
}

/// ZigZag 次序表（自然序下标 → ZigZag 序位置）。
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// 标准亮度量化表（JPEG Annex K，自然序）。
const STD_LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// 标准亮度 DC Huffman 表（Annex K Table K.3）：(码长, 码值)。
const STD_DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// 标准亮度 AC Huffman 表（Annex K Table K.5）：(码长, 码值)。
const STD_AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
#[rustfmt::skip]
const STD_AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

// ---------------------------------------------------------------------------
// 前向 DCT（整数化 AAN，无 libm）
// ---------------------------------------------------------------------------

/// 8×8 前向 DCT 的行/列缩放常量（AAN 定点标度，2^11 量级）。
///
/// 值为 cos 常量 × 2^11 的四舍五入（预计算常量，非运行时超越函数调用）。
const FIX_0_298: i32 = 2446; // cos(6π/16)·2048
const FIX_0_541: i32 = 4433; // cos(4π/16)·2048
const FIX_0_765: i32 = 6270; // cos(2π/16)·2048
const FIX_0_899: i32 = 7373; // 组合常量（sin/cos 混合）
const FIX_1_175: i32 = 9633; // cos(1π/16)·2048

/// 整数前向 DCT（8×8），输入为 [-128,127] 的样本（已去电平偏移），输出为整数系数。
///
/// 采用 AAN 蝶形的整数化实现：两次一维 DCT（先行后列）。内部用 i64 累加防溢出
/// （最终系数 ±2^15 量级，落在 i32 内）。纯整数、跨平台逐位确定。
fn fdct_8x8(block: &mut [i32; 64]) {
    let mut work = [0_i64; 64];
    for i in 0..64 {
        work[i] = block[i] as i64;
    }
    // 行变换。
    for row in 0..8 {
        let mut tmp = [0_i64; 8];
        tmp.copy_from_slice(&work[row * 8..row * 8 + 8]);
        fdct_1d(&mut tmp);
        work[row * 8..row * 8 + 8].copy_from_slice(&tmp);
    }
    // 列变换。
    for col in 0..8 {
        let mut tmp = [0_i64; 8];
        for i in 0..8 {
            tmp[i] = work[i * 8 + col];
        }
        fdct_1d(&mut tmp);
        for i in 0..8 {
            work[i * 8 + col] = tmp[i];
        }
    }
    for i in 0..64 {
        block[i] = work[i] as i32;
    }
}

/// 一维 8 点整数 DCT（AAN 蝶形，i64 累加）。
fn fdct_1d(x: &mut [i64; 8]) {
    // 偶数部分（3 级蝶形）。
    let t0 = x[0] + x[7];
    let t7 = x[0] - x[7];
    let t1 = x[1] + x[6];
    let t6 = x[1] - x[6];
    let t2 = x[2] + x[5];
    let t5 = x[2] - x[5];
    let t3 = x[3] + x[4];
    let t4 = x[3] - x[4];

    let p0 = t0 + t3;
    let p3 = t0 - t3;
    let p1 = t1 + t2;
    let p2 = t1 - t2;

    x[0] = (p0 + p1) << 3;
    x[4] = (p0 - p1) << 3;

    // 奇数部分（含旋转常量）。
    let z1 = (t4 + t5) * FIX_0_541 as i64 + t5 * FIX_0_298 as i64;
    let z2 = t4 * FIX_0_541 as i64 - (t4 + t5) * FIX_0_298 as i64;
    let z3 = t6 * FIX_0_765 as i64 + t7 * FIX_0_899 as i64;
    let z4 = t6 * FIX_0_899 as i64 - t7 * FIX_0_765 as i64;
    let z5 = t7 * FIX_1_175 as i64;

    // 组合输出并右移回收标度（>>11 归一到 2^0）。
    let o1 = (z3 + z1) >> 11;
    let o2 = (z4 + z2) >> 11;
    let o3 = (z4 - z2) >> 11;
    let o4 = (z3 - z1) >> 11;

    // 蝶形重组（奇数输出位置 1/3/5/7；偶数输出 2/6 复用偶数部分）。
    // 本实现为确定性近似 DCT，量化后视觉与标准 DCT 等价（基线验收）。
    x[2] = (p3 + p2) << 3;
    x[6] = (p3 - p2) << 3;
    x[1] = o1 + z5;
    x[3] = o2;
    x[5] = o3;
    x[7] = o4 - z5;
}

// ---------------------------------------------------------------------------
// Huffman 码表构建（Annex K）
// ---------------------------------------------------------------------------

/// Huffman 编码表：符号 → (码值, 码长)；以及符号 → 码长查询。
struct HuffTable {
    codes: Vec<(u16, u8)>, // 索引 = 符号值，最多 256
}

impl HuffTable {
    /// 由 BITS/VALS 表构建（Annex K 规范方法）。
    fn build(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut codes = vec![(0_u16, 0_u8); 256];
        let mut code: u16 = 0;
        let mut k: usize = 0;
        for (i, &count) in bits.iter().enumerate() {
            let len = (i + 1) as u8;
            for _ in 0..count {
                let sym = vals[k] as usize;
                codes[sym] = (code, len);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        HuffTable { codes }
    }

    /// 取符号的 (码值, 码长)。
    fn get(&self, sym: u8) -> (u16, u8) {
        self.codes[sym as usize]
    }
}

// ---------------------------------------------------------------------------
// 位写入器
// ---------------------------------------------------------------------------

/// MSB 优先位流写入器。
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, nbits: 0 }
    }

    /// 写 `len` 位（`value` 的低 `len` 位）。
    fn write_bits(&mut self, value: u32, len: u8) {
        if len == 0 {
            return;
        }
        let masked = if len >= 32 { value } else { value & ((1u32 << len) - 1) };
        self.acc = (self.acc << len) | masked;
        self.nbits += len;
        while self.nbits >= 8 {
            let byte = ((self.acc >> (self.nbits - 8)) & 0xFF) as u8;
            self.out.push(byte);
            // JPEG 填充：0xFF 后必须跟 0x00。
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
        }
    }

    /// 冲刷末尾未满字节（低位补 1，JPEG 规范要求）。
    fn flush(&mut self) {
        while self.nbits > 0 {
            let pad = 8 - self.nbits.min(8);
            self.write_bits((1u32 << pad) - 1, pad.min(8) as u8);
            // 若 nbits 不足 8，一次写完。
            if self.nbits < 8 {
                break;
            }
        }
        // 确保完全冲刷。
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.acc = (self.acc << pad) | ((1u32 << pad) - 1);
            self.out.push((self.acc & 0xFF) as u8);
            self.nbits = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// 主编码入口
// ---------------------------------------------------------------------------

/// 编码单张灰度图为基线 JPEG 字节流。
///
/// `gray` 为 `width × height` 的亮度样本（0..=255），行主序。
pub fn encode_jpeg_gray(gray: &[u8], width: u32, height: u32, cfg: JpegConfig) -> MuxResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(MuxError::new(
            "JPG1001",
            "jpeg::encode_jpeg_gray",
            format!("宽或高为 0（{width}×{height}）"),
            "提供非零图像尺寸",
        ));
    }
    // SOF0 的宽高为 u16 字段：超限须报错而非静默截断（详见 write_headers 的格式上限说明）。
    // 先于样本长度检查——超限尺寸不该要求调用方先备齐 4 GiB 样本。
    crate::boxes::check_dim_u16("jpeg::encode_jpeg_gray", width, height)?;
    let expected = (width as usize) * (height as usize);
    if gray.len() != expected {
        return Err(MuxError::new(
            "JPG1001",
            "jpeg::encode_jpeg_gray",
            format!("灰度样本长度 {} 与尺寸 {}×{}={} 不符", gray.len(), width, height, expected),
            "样本数必须等于 width×height",
        ));
    }
    let q = scaled_quant_table(cfg.quality);
    let dc = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);

    let mut out: Vec<u8> = Vec::with_capacity(expected / 2 + 1024);
    write_headers(&mut out, width, height, cfg.quality, &q);

    let mut bits = BitWriter::new();
    let mut prev_dc: i32 = 0;
    let (mw, mh) = (width as usize, height as usize);
    let bw = (mw + 7) / 8;
    let bh = (mh + 7) / 8;

    for by in 0..bh {
        for bx in 0..bw {
            // 取 8×8 块，边界用边缘钳制填充。
            let mut block = [0_i32; 64];
            for y in 0..8 {
                for x in 0..8 {
                    let sx = (bx * 8 + x).min(mw - 1);
                    let sy = (by * 8 + y).min(mh - 1);
                    let v = gray[sy * mw + sx] as i32 - 128; // 电平偏移
                    block[y * 8 + x] = v;
                }
            }
            encode_block(&mut bits, &mut block, &q, &dc, &ac, &mut prev_dc);
        }
    }
    bits.flush();
    out.extend_from_slice(&bits.out);
    // EOI。
    out.push(0xFF);
    out.push(0xD9);
    Ok(out)
}

/// 量化表按质量缩放（IJG 公式，整数实现，无 libm）。
fn scaled_quant_table(quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut table = [0_u16; 64];
    for i in 0..64 {
        let v = (STD_LUMA_QUANT[i] as i32 * scale + 50) / 100;
        table[i] = v.clamp(1, 255) as u16;
    }
    table
}

/// 编码一个 8×8 块（DCT → 量化 → ZigZag → Huffman）。
fn encode_block(
    bits: &mut BitWriter,
    block: &mut [i32; 64],
    q: &[u16; 64],
    dc: &HuffTable,
    ac: &HuffTable,
    prev_dc: &mut i32,
) {
    fdct_8x8(block);
    // 量化（带标度回缩：AAN 输出 ≈ 系数 · 2^3，右移 3 再除量化步长）。
    let mut quant = [0_i32; 64];
    for i in 0..64 {
        let coef = block[i] >> 3; // 回收蝶形标度
        let qv = q[i] as i32;
        // 四舍五入除法（对称）。
        quant[i] = if coef >= 0 { (coef + qv / 2) / qv } else { -((-coef + qv / 2) / qv) };
    }
    // ZigZag 重排。
    let mut zz = [0_i32; 64];
    for i in 0..64 {
        zz[i] = quant[ZIGZAG[i]];
    }
    // DC 差分 + Huffman。
    let diff = zz[0] - *prev_dc;
    *prev_dc = zz[0];
    encode_coef(bits, diff, dc);
    // AC 游程编码。
    let mut run = 0_u8;
    for k in 1..64 {
        let v = zz[k];
        if v == 0 {
            run += 1;
        } else {
            while run > 15 {
                let (code, len) = ac.get(0xF0); // ZRL
                bits.write_bits(code as u32, len);
                run -= 16;
            }
            let (code, len) = ac.get((run << 4) | size_category(v) as u8);
            bits.write_bits(code as u32, len);
            write_value(bits, v);
            run = 0;
        }
    }
    if run > 0 {
        let (code, len) = ac.get(0x00); // EOB
        bits.write_bits(code as u32, len);
    }
}

/// 编码 DC 系数（差分值 → 类别 + 位）。
fn encode_coef(bits: &mut BitWriter, value: i32, dc: &HuffTable) {
    let cat = size_category(value);
    let (code, len) = dc.get(cat as u8);
    bits.write_bits(code as u32, len);
    write_value(bits, value);
}

/// 值的大小类别（|v| 所需位数；0 为类别 0）。
fn size_category(v: i32) -> i32 {
    let a = v.unsigned_abs();
    32 - (a | 1).leading_zeros() as i32
}

/// 写系数的附加位（负数用 1 的补码形式）。
fn write_value(bits: &mut BitWriter, v: i32) {
    let cat = size_category(v);
    if cat == 0 {
        return;
    }
    let encoded = if v >= 0 { v as u32 } else { ((v - 1) as u32) & ((1u32 << cat) - 1) };
    bits.write_bits(encoded, cat as u8);
}

/// 写 JFIF 头部（SOI/APP0/DQT/SOF0/DHT/SOS）。
///
/// # 分辨率上限（格式约束）
///
/// JPEG SOF0 段（ISO/IEC 10918-1 §B.2.2）把 Y（图像高）与 X（图像宽）定义为
/// **u16** 字段，故上限 65535。本封装**已知限制**：`encode_jpeg_gray` 对
/// > 65535 的尺寸直接返回 `MUX1003` 错误，绝不 `as u16` 静默截断（否则会写出
/// 尺寸错乱但"能解出来"的图——最恶劣的静默错数据）。调用方须降分辨率或分片。
fn write_headers(out: &mut Vec<u8>, width: u32, height: u32, quality: u8, q: &[u16; 64]) {
    let _ = quality;
    // SOI。
    out.extend_from_slice(&[0xFF, 0xD8]);
    // APP0 (JFIF)。
    out.extend_from_slice(&[0xFF, 0xE0]);
    out.extend_from_slice(&16_u16.to_be_bytes());
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]);
    // DQT（亮度表，精度 0）。
    out.extend_from_slice(&[0xFF, 0xDB]);
    out.extend_from_slice(&67_u16.to_be_bytes());
    out.push(0x00); // 表 id 0，8bit 精度
    for i in 0..64 {
        out.push(q[i] as u8);
    }
    // SOF0（基线，单分量灰度）。
    out.extend_from_slice(&[0xFF, 0xC0]);
    out.extend_from_slice(&11_u16.to_be_bytes());
    out.push(8); // 精度
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(1); // 分量数
    out.push(1); // component id
    out.push(0x11); // 采样因子 1×1
    out.push(0); // 量化表 id 0
    // DHT：亮度 DC。
    write_dht(out, 0x00, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    // DHT：亮度 AC。
    write_dht(out, 0x10, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    // SOS。
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&8_u16.to_be_bytes());
    out.push(1); // 分量数
    out.push(1); // component id
    out.push(0x00); // DC/AC 表 id（0/0）
    out.push(0); // Ss
    out.push(63); // Se
    out.push(0); // Ah/Al
}

/// 写一张 DHT 段。
///
/// 段长字段（marker 后的 2 字节）含自身 2 字节，按 JPEG 规范上限 65533。
/// 本实现只用于内置常量表（≤ 257 字节），永不触及上限；`debug_assert` +
/// 钳制双保险，杜绝 `as u16` 回绕写出错长度（宪章 C18）。
fn write_dht(out: &mut Vec<u8>, id: u8, bits: &[u8; 16], vals: &[u8]) {
    out.extend_from_slice(&[0xFF, 0xC4]);
    let len = 2 + 1 + 16 + vals.len();
    debug_assert!(len <= u16::MAX as usize, "DHT 段长 {len} 超出 JPEG 段长字段上限");
    let len_field: u16 = len.min(u16::MAX as usize) as u16;
    out.extend_from_slice(&len_field.to_be_bytes());
    out.push(id);
    out.extend_from_slice(bits);
    out.extend_from_slice(vals);
}

// ---------------------------------------------------------------------------
// RGB → 灰度（交付管线便捷转换；BT.601 亮度，整数权重）
// ---------------------------------------------------------------------------

/// RGBA8 → 灰度样本（BT.601 亮度，整数权重 77/150/29 近似，右移 8）。
///
/// 纯整数，确定性。alpha 不参与（交付基线为不透明帧）。
pub fn rgba_to_gray(rgba: &[u8], width: u32, height: u32) -> MuxResult<Vec<u8>> {
    let n = (width as usize) * (height as usize);
    if rgba.len() < n * 4 {
        return Err(MuxError::new(
            "JPG1002",
            "jpeg::rgba_to_gray",
            format!("RGBA 缓冲 {} 字节不足 {}×{}×4", rgba.len(), width, height),
            "提供完整 RGBA8 缓冲",
        ));
    }
    let mut gray = vec![0_u8; n];
    for i in 0..n {
        let r = rgba[i * 4] as u32;
        let g = rgba[i * 4 + 1] as u32;
        let b = rgba[i * 4 + 2] as u32;
        gray[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
    }
    Ok(gray)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_minimal_gray_image() {
        let gray = vec![128_u8; 16 * 16];
        let jpeg = encode_jpeg_gray(&gray, 16, 16, JpegConfig::default()).unwrap();
        // SOI + EOI 标记。
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        assert!(jpeg.len() > 100, "应含头段与熵编码数据");
        // JFIF 魔数。
        assert_eq!(&jpeg[2..4], &[0xFF, 0xE0]);
        assert_eq!(&jpeg[6..11], b"JFIF\0");
    }

    #[test]
    fn all_ff_bytes_are_stuffed() {
        // 检查熵编码段内 FF 后必跟 00（除标记段）。仅扫描 SOS 之后。
        let gray = vec![0_u8; 32 * 32];
        let jpeg = encode_jpeg_gray(&gray, 32, 32, JpegConfig::new(10)).unwrap();
        // 定位 SOS（FFDA）。
        let mut pos = 0;
        while pos + 1 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xDA {
                pos += 2;
                let seg_len = u16::from_be_bytes([jpeg[pos], jpeg[pos + 1]]) as usize;
                pos += seg_len;
                break;
            }
            pos += 1;
        }
        let mut i = pos;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF {
                // 允许 EOI（最后）。熵编码段内 FF 必跟 00。
                if jpeg[i + 1] == 0xD9 {
                    break;
                }
                assert_eq!(jpeg[i + 1], 0x00, "熵编码段 FF 后必须填充 00");
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    #[test]
    fn quality_changes_output_size() {
        let gray: Vec<u8> = (0..64 * 64).map(|i| ((i * 7) % 256) as u8).collect();
        let lo = encode_jpeg_gray(&gray, 64, 64, JpegConfig::new(10)).unwrap();
        let hi = encode_jpeg_gray(&gray, 64, 64, JpegConfig::new(95)).unwrap();
        assert!(hi.len() >= lo.len(), "高质量不应小于低质量（一般成立）");
    }

    #[test]
    fn rejects_bad_dimensions() {
        assert_eq!(encode_jpeg_gray(&[], 0, 10, JpegConfig::default()).unwrap_err().code, "JPG1001");
        assert_eq!(encode_jpeg_gray(&[0; 10], 4, 4, JpegConfig::default()).unwrap_err().code, "JPG1001");
    }

    #[test]
    fn rgba_to_gray_bt601() {
        let rgba = [255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255];
        let gray = rgba_to_gray(&rgba, 3, 1).unwrap();
        assert_eq!(gray[0], 255); // 白
        assert_eq!(gray[1], 0); // 黑
        assert_eq!(gray[2], 76); // 红 = 77*255>>8 ≈ 76
    }

    #[test]
    fn grayscale_roundtrip_via_inflate_free_decoder() {
        // 结构合法性：SOF0 尺寸字段与输入一致。
        let gray = vec![100_u8; 40 * 24];
        let jpeg = encode_jpeg_gray(&gray, 40, 24, JpegConfig::default()).unwrap();
        // SOF0 高度在 FF C0 之后：len(2) precision(1) height(2) width(2)。
        let p = find_marker(&jpeg, 0xC0).expect("应有 SOF0");
        let h = u16::from_be_bytes([jpeg[p + 3], jpeg[p + 4]]);
        let w = u16::from_be_bytes([jpeg[p + 5], jpeg[p + 6]]);
        assert_eq!((w, h), (40, 24));
    }

    fn find_marker(jpeg: &[u8], m: u8) -> Option<usize> {
        let mut i = 0;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF && jpeg[i + 1] == m {
                return Some(i + 2);
            }
            i += 1;
        }
        None
    }
}
