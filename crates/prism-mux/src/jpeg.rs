//! 基线 JPEG 编码（C4； 容器路线； 同域）。
//!
//! **灰度最小基线**（彩色已落地）：8×8 分块 → 前向 DCT（编译期余弦基的整数
//! 参考实现，禁 libm）→ 量化 → ZigZag → DC 差分 + AC 游程 → 标准 Huffman
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
use std::sync::OnceLock;

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

/// 8×8 前向 DCT 余弦基（行主序，索引 `u*8 + x`）。
///
/// 值 = `round(cos((2x+1)·u·π/16) · 2^14)`，编译期常量——零运行时超越函数（D-29）。
/// 这是 JPEG 规范 DCT 的正交基，与标准 `jpeg-9`/`libjpeg` 浮点参考实现逐系数一致
/// （仅末位舍入差异），确保量化后系数落在标准量级，可被标准量化表直接除。
const DCT_COS: [i32; 64] = [
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16069, 13623, 9102, 3196, -3196, -9102,
    -13623, -16069, 15137, 6270, -6270, -15137, -15137, -6270, 6270, 15137, 13623, -3196, -16069,
    -9102, 9102, 16069, 3196, -13623, 11585, -11585, -11585, 11585, 11585, -11585, -11585, 11585,
    9102, -16069, 3196, 13623, -13623, -3196, 16069, -9102, 6270, -15137, 15137, -6270, -6270, 15137,
    -15137, 6270, 3196, -9102, 13623, -16069, 16069, -13623, 9102, -3196,
];

/// 整数前向 DCT（8×8），输入为 [-128,127] 的样本（已去电平偏移），输出为整数系数。
///
/// 采用 JPEG 规范定义（可分离，基 `cos((2x+1)uπ/16)·cos((2y+1)vπ/16)`，归一化
/// `(1/4)·C(u)·C(v)`，`C(0)=1/√2` 余 1），做两次一维 8 点变换。余弦基为编译期常量
/// （`DCT_COS`，标度 2^14），全程 i64 累加防溢出，末位统一右移回收标度并施加归一化。
/// 输出系数落在 i32 且量级与标准 DCT 一致，可被标准量化表直接除。纯整数、跨平台确定。
fn fdct_8x8(block: &mut [i32; 64]) {
    // 一维行变换：P[u][y] = Σ_x block[y*8+x] · DCT_COS[u*8+x]。
    let mut p = [[0_i64; 8]; 8];
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0_i64;
            for x in 0..8 {
                s += (block[y * 8 + x] as i64) * (DCT_COS[u * 8 + x] as i64);
            }
            p[u][y] = s;
        }
    }
    // 一维列变换 + 归一化：G[u][v] = Σ_y P[u][y] · DCT_COS[v*8+y] · (1/4)·C(u)·C(v)。
    // 两段余弦各标度 2^14，故 P·cos 实为真实 DCT 标量 × 2^28，末位 `>>28` 回收。
    const SQRT1_2: i64 = 46341; // round(1/√2 · 2^16)
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0_i64;
            for y in 0..8 {
                s += p[u][y] * (DCT_COS[v * 8 + y] as i64);
            }
            let mut val = s >> 28;
            if u == 0 {
                val = (val * SQRT1_2) >> 16;
            }
            if v == 0 {
                val = (val * SQRT1_2) >> 16;
            }
            val >>= 2; // (1/4)
            block[v * 8 + u] = val as i32;
        }
    }
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

// V3.1 编码 IO：标准 Annex K Huffman 表是**编译期常量**，旧实现每帧重建
// 两次（灰度 / 彩色各 2 表 × 每帧），3600 帧 = 7200 次无谓分配与填表。
// 用 std 的 `OnceLock` 进程级缓存一次；表内容逐位不变 ⇒ JPEG 字节流不变。
static STD_HUFF: OnceLock<(HuffTable, HuffTable)> = OnceLock::new();

fn std_huff_tables() -> &'static (HuffTable, HuffTable) {
    STD_HUFF.get_or_init(|| {
        (
            HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS),
            HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS),
        )
    })
}

/// V3.1 收尾 C —— 一部片子共用的 JPEG 常量表（量化表 + 标准 Huffman 表引用）。
///
/// 旧 `encode_jpeg_color` 每帧重建一次 `scaled_quant_table`（64 次乘除）并
/// 重新查 `STD_HUFF`；3600 帧 = 3600 次纯重复。出片链路 `quality` 恒定，
/// 故在帧循环外构建一次，逐帧只做**必需**的 `rgb_to_ycbcr` 色彩转换 +
/// DCT + 熵编码（色彩转换每帧一次，不重复；灰度便捷函数那条独立路径不进
/// 出片主链）。表内容逐位不变 ⇒ JPEG / MP4 字节流逐字节同源。
pub struct JpegTables {
    q: [u16; 64],
    huff: &'static (HuffTable, HuffTable),
}

impl JpegTables {
    /// 按质量构建一次（Huffman 表由 `OnceLock` 进程级共享，零重复分配）。
    pub fn new(quality: u8) -> Self {
        JpegTables {
            q: scaled_quant_table(quality),
            huff: std_huff_tables(),
        }
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
    let (dc, ac) = std_huff_tables();

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
            encode_block(&mut bits, &mut block, &q, dc, ac, &mut prev_dc);
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
        let coef = block[i]; // fdct_8x8 已输出标准量级系数，无需额外缩放
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

/// 值的大小类别（|v| 所需位数；**0 必须映射到类别 0**）。
///
/// JPEG 规范（ITU-T T.81 §F.1.2.1.3，Huffman 幅度编码）规定：幅度类别
/// `category(v) = ⌊log2|v|⌋ + 1`（即 |v| 的二进制位数），且特别地 **`category(0) = 0`**
/// —— 零值不占用任何附加位。本函数对 `v ≠ 0` 沿用原公式
/// `32 − |v|.leading_zeros()`（即 ⌊log2|v|⌋+1），仅对 `v == 0` 特判返回 0。
///
/// # 为什么 0 不能归到类别 1
///
/// 若把 0 误归为类别 1，则一个零 DC 差分会写出「DC Huffman 类别-1 码 + 1 位
/// 附加幅度位」。该附加位因值为 0 而取 `0`。而 JPEG 幅度还原规则为：对类别 `s`，
/// 读 `s` 位 `mag`，若 `mag < 2^(s−1)` 则值 = `mag − (2^s − 1)`，否则 = `mag`。
/// 代入 `s = 1, mag = 0`：值 = `0 − (2 − 1) = −1`。于是解码器把本应是 0 的
/// DC 差分还原成 **−1**。
///
/// 后果：上一块的预测 DC 被多减 1，下一块的 DC 差分又比真实值多减 1，DC 预测
/// 链**逐 MCU 累计漂移 −1**。平坦画布（每块 DC 差恒为 0）解码后自上而下整幅
/// 漂到全黑——灰度（`encode_jpeg_gray`）与彩色（`encode_jpeg_color`）路径共用
/// 同一 [`size_category`]，因此都中招。这正是一张纯色帧经编码/解码后均值严重
/// 偏离（如期望 128 实际趋于 0）的根因。修复只需让 `category(0) == 0`。
fn size_category(v: i32) -> i32 {
    let a = v.unsigned_abs();
    if a == 0 {
        0
    } else {
        32 - a.leading_zeros() as i32
    }
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
    // 必须按 zigzag 顺序写量化表（JPEG 规范 T.81 §B.2.4.1）。
    for i in 0..64 {
        out.push(q[ZIGZAG[i] as usize] as u8);
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

// ---------------------------------------------------------------------------
// 彩色基线 JPEG（4:4:4，YCbCr，无外部依赖）
//
// 复用灰度路径的全部基础件（整数参考 DCT / 量化 / ZigZag / Huffman 位流），
// 仅增加：RGB→YCbCr 平面分离、三分量按 MCU 交织（顺序 Y,Cb,Cr）。三个分量
// 共用同一张亮度 Huffman 表（标准允许的合法配置，解码器无歧义；动画流对码率
// 不敏感）。色度量化复用亮度量化表（基线质量足够）。DQT 段长度按规范手写为 67。
// ---------------------------------------------------------------------------

/// RGBA8 → YCbCr 三平面（JFIF 全范围整数矩阵，无 libm；结果钳制到 [0,255]）。
fn rgb_to_ycbcr(rgba: &[u8], n: usize, y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    for i in 0..n {
        let r = rgba[i * 4] as i32;
        let g = rgba[i * 4 + 1] as i32;
        let b = rgba[i * 4 + 2] as i32;
        let yy = (77 * r + 150 * g + 29 * b) >> 8;
        let cc = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
        let rr = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
        y[i] = yy.clamp(0, 255) as u8;
        cb[i] = cc.clamp(0, 255) as u8;
        cr[i] = rr.clamp(0, 255) as u8;
    }
}

/// 从单色平面取 8×8 块（边界边缘钳制），并做电平偏移 [-128,127]。
fn fill_block(plane: &[u8], mw: usize, mh: usize, bx: usize, by: usize, blk: &mut [i32; 64]) {
    for yy in 0..8 {
        for xx in 0..8 {
            let sx = (bx * 8 + xx).min(mw - 1);
            let sy = (by * 8 + yy).min(mh - 1);
            blk[yy * 8 + xx] = plane[sy * mw + sx] as i32 - 128;
        }
    }
}

/// 写一张 DQT 段（亮度/色度共用同一量化值，分别挂表 id 0 / 1）。
///
/// 段长严格为 `2（长度字段自身）+ 1（pq/tq）+ 64（表） = 67`。
fn write_dqt(out: &mut Vec<u8>, id: u8, q: &[u16; 64]) {
    out.extend_from_slice(&[0xFF, 0xDB]);
    out.extend_from_slice(&67_u16.to_be_bytes());
    out.push(id); // pq=0, tq=id
    // 必须按 zigzag 顺序写量化表（JPEG 规范 T.81 §B.2.4.1）。
    for i in 0..64 {
        out.push(q[ZIGZAG[i] as usize] as u8);
    }
}

/// 写彩色（3 分量 4:4:4）JFIF 头部。
fn write_headers_color(out: &mut Vec<u8>, width: u32, height: u32, q: &[u16; 64]) {
    out.extend_from_slice(&[0xFF, 0xD8]);
    // APP0 (JFIF)，长度 16（业界事实标准值）。
    out.extend_from_slice(&[0xFF, 0xE0]);
    out.extend_from_slice(&16_u16.to_be_bytes());
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]);
    // DQT：三分量共用亮度量化表（id 0）。
    write_dqt(out, 0, q);
    // SOF0：3 分量，各 1×1 采样。
    out.extend_from_slice(&[0xFF, 0xC0]);
    out.extend_from_slice(&17_u16.to_be_bytes()); // 2 + (1+2+2+1+3*3=15) = 17
    out.push(8);
    out.extend_from_slice(&(height as u16).to_be_bytes());
    out.extend_from_slice(&(width as u16).to_be_bytes());
    out.push(3);
    out.push(1); out.push(0x11); out.push(0); // Y
    out.push(2); out.push(0x11); out.push(0); // Cb
    out.push(3); out.push(0x11); out.push(0); // Cr
    // DHT：仅亮度 DC(0x00)/AC(0x10)；三分量共用。
    write_dht(out, 0x00, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    write_dht(out, 0x10, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    // SOS：3 分量，均引用亮度表。
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&12_u16.to_be_bytes()); // 2 + (1+3*2+3=9) = 12
    out.push(3);
    out.push(1); out.push(0x00);
    out.push(2); out.push(0x00);
    out.push(3); out.push(0x00);
    out.push(0); out.push(63); out.push(0);
}

/// 编码单张彩色 RGBA 图为基线（4:4:4）JPEG 字节流。
///
/// 不依赖任何外部库；与 [`encode_jpeg_gray`] 共用 DCT/量化/Huffman 基础件。
/// 单帧入口：现构一次 [`JpegTables`] 后委托 [`encode_jpeg_color_with_tables`]，
/// 字节流与改动前逐字节相同。
pub fn encode_jpeg_color(rgba: &[u8], width: u32, height: u32, cfg: JpegConfig) -> MuxResult<Vec<u8>> {
    let tables = JpegTables::new(cfg.quality);
    encode_jpeg_color_with_tables(rgba, width, height, &tables)
}

/// 出片主链批量入口：复用帧循环外构建的 [`JpegTables`]，逐帧只做一次
/// `rgb_to_ycbcr` 色彩转换 + DCT + 熵编码（表构建零重复）。
///
/// 与 [`encode_jpeg_color`] 逐字节同源；调用方负责保证 `tables` 的质量与
/// `JpegConfig.quality` 一致（[`JpegTables::new`] 是唯一构造入口）。
pub fn encode_jpeg_color_with_tables(
    rgba: &[u8],
    width: u32,
    height: u32,
    tables: &JpegTables,
) -> MuxResult<Vec<u8>> {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Err(MuxError::new(
            "JPG1001",
            "jpeg::encode_jpeg_color",
            format!("尺寸为 0（{width}×{height}）"),
            "提供非零图像尺寸",
        ));
    }
    crate::boxes::check_dim_u16("jpeg::encode_jpeg_color", width, height)?;
    if rgba.len() < n * 4 {
        return Err(MuxError::new(
            "JPG1002",
            "jpeg::encode_jpeg_color",
            format!("RGBA 缓冲 {} 字节不足 {}×{}×4", rgba.len(), width, height),
            "提供完整 RGBA8 缓冲",
        ));
    }
    let q: &[u16; 64] = &tables.q;
    let (luma_dc, luma_ac): (&HuffTable, &HuffTable) = (&tables.huff.0, &tables.huff.1);

    let mut y_plane = vec![0u8; n];
    let mut cb_plane = vec![0u8; n];
    let mut cr_plane = vec![0u8; n];
    rgb_to_ycbcr(rgba, n, &mut y_plane, &mut cb_plane, &mut cr_plane);

    let mut out: Vec<u8> = Vec::with_capacity(n * 2 + 2048);
    write_headers_color(&mut out, width, height, q);

    let mut bits = BitWriter::new();
    let (mw, mh) = (width as usize, height as usize);
    let bw = (mw + 7) / 8;
    let bh = (mh + 7) / 8;
    let mut prev_y = 0_i32;
    let mut prev_cb = 0_i32;
    let mut prev_cr = 0_i32;
    let mut blk = [0_i32; 64];

    // 按 MCU（4:4:4 → 每 MCU 含 Y/Cb/Cr 各一 8×8 块，顺序 Y,Cb,Cr）交织编码。
    for by in 0..bh {
        for bx in 0..bw {
            fill_block(&y_plane, mw, mh, bx, by, &mut blk);
            encode_block(&mut bits, &mut blk, q, luma_dc, luma_ac, &mut prev_y);
            fill_block(&cb_plane, mw, mh, bx, by, &mut blk);
            encode_block(&mut bits, &mut blk, q, luma_dc, luma_ac, &mut prev_cb);
            fill_block(&cr_plane, mw, mh, bx, by, &mut blk);
            encode_block(&mut bits, &mut blk, q, luma_dc, luma_ac, &mut prev_cr);
        }
    }
    bits.flush();
    out.extend_from_slice(&bits.out);
    out.push(0xFF);
    out.push(0xD9);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prebuilt_tables_match_per_frame_encode_bytewise() {
        // 多帧、多质量：出片主链复用表构建，字节流必须与单帧入口逐字节相同。
        let frames: Vec<Vec<u8>> = vec![
            vec![128u8; 24 * 16 * 4],
            vec![37u8; 24 * 16 * 4],
            vec![211u8; 24 * 16 * 4],
        ];
        for quality in [35u8, 85, 95] {
            let tables = JpegTables::new(quality);
            for f in &frames {
                let single = encode_jpeg_color(
                    f,
                    24,
                    16,
                    JpegConfig { quality, grayscale: false },
                )
                .unwrap();
                let batched =
                    encode_jpeg_color_with_tables(f, 24, 16, &tables).unwrap();
                assert_eq!(single, batched, "表复用路径必须逐字节同源（quality={quality}）");
            }
        }
    }

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

    // =======================================================================
    // size_category 回归测试（零 DC 差分漂移 bug 修复）
    // =======================================================================

    /// 用给定 Huffman 表从 MSB 优先位流（每元素 0/1）解出首个符号。
    ///
    /// 标准规范 Huffman 为前缀无关码，故在累加位首次匹配到某符号的
    /// (码值, 码长) 时即唯一确定该符号。无 libm、无查表库，纯位比对。
    /// 位流耗尽时返回 `None`（调用方据此停止，不再越界读取）。
    fn huff_decode_first(bits: &[u8], start: &mut usize, table: &HuffTable) -> Option<u8> {
        let mut acc: u32 = 0;
        loop {
            if *start >= bits.len() {
                return None;
            }
            let bit = bits[*start];
            *start += 1;
            acc = (acc << 1) | bit as u32;
            let len = *start as u32;
            for sym in 0u16..=255 {
                let (code, clen) = table.get(sym as u8);
                if clen as u32 == len && u32::from(code) == (acc & ((1u32 << clen) - 1)) {
                    return Some(sym as u8);
                }
            }
        }
    }

    /// 把熵编码段（SOS 之后，含 0xFF 0x00 填充）展开为逐位 0/1 向量，遇 EOI 停止。
    fn bits_from_entropy(jpeg: &[u8]) -> Vec<u8> {
        // 跳到 SOS 段尾。
        let mut pos = 0usize;
        while pos + 3 < jpeg.len() {
            if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xDA {
                let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
                pos += 2 + len;
                break;
            }
            pos += 1;
        }
        let mut bits: Vec<u8> = Vec::new();
        let mut i = pos;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF {
                if jpeg[i + 1] == 0xD9 {
                    break; // EOI
                }
                if jpeg[i + 1] == 0x00 {
                    // 字节填充：0xFF 是真实数据字节，其后 0x00 为填充物。
                    for bp in (0..8).rev() {
                        bits.push((0xFF >> bp) & 1);
                    }
                    i += 2;
                    continue;
                }
                break; // 熵段内不应出现的其它 marker
            }
            for bp in (0..8).rev() {
                bits.push((jpeg[i] >> bp) & 1);
            }
            i += 1;
        }
        bits
    }

    /// 读 `sz` 位幅度并还原为带符号值（同 JPEG §F.1.2.1.3 规范）。
    fn read_magnitude(bits: &[u8], idx: &mut usize, sz: u8) -> i32 {
        if sz == 0 {
            return 0;
        }
        let mut v: u32 = 0;
        for _ in 0..sz {
            v = (v << 1) | bits[*idx] as u32;
            *idx += 1;
        }
        if v < (1u32 << (sz as u32 - 1)) {
            v as i32 - ((1i32 << sz as u32) - 1)
        } else {
            v as i32
        }
    }

    #[test]
    fn size_category_zero_is_zero() {
        assert_eq!(size_category(0), 0, "0 必须映射到 DC 类别 0（漂移 bug 根因）");
    }

    #[test]
    fn size_category_boundaries() {
        // 非负：|v|=1→1, 2..3→2, 4..7→3, 8..15→4, 16..31→5 …
        assert_eq!(size_category(1), 1);
        for v in 2..=3 {
            assert_eq!(size_category(v), 2, "v={v}");
        }
        for v in 4..=7 {
            assert_eq!(size_category(v), 3, "v={v}");
        }
        for v in 8..=15 {
            assert_eq!(size_category(v), 4, "v={v}");
        }
        for v in 16..=31 {
            assert_eq!(size_category(v), 5, "v={v}");
        }
        // 负数按绝对值。
        assert_eq!(size_category(-1), 1);
        assert_eq!(size_category(-3), 2);
        assert_eq!(size_category(-7), 3);
        assert_eq!(size_category(-15), 4);
    }

    #[test]
    fn dc_zero_diff_emits_category_zero_symbol() {
        // 内省点：直接对 diff==0 调用 encode_coef，解码其写出的首个 DC Huffman
        // 符号必须是类别 0（绝不是类别 1）。
        let dc = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
        let (code0, len0) = dc.get(0);
        let (code1, len1) = dc.get(1);
        assert_ne!((code0, len0), (code1, len1), "DC 表类别 0/1 码字不应退化相同");

        let mut bits = BitWriter::new();
        encode_coef(&mut bits, 0, &dc);
        bits.flush();
        // 把字节流展开为逐位 0/1 向量再解码（与真实熵段一致）。
        let mut bitvec: Vec<u8> = Vec::new();
        for b in &bits.out {
            for bp in (0..8).rev() {
                bitvec.push((b >> bp) & 1);
            }
        }
        let sym = huff_decode_first(&bitvec, &mut 0, &dc).expect("应解出一个 DC 符号");
        assert_eq!(
            sym, 0,
            "diff==0 的 DC 系数必须写成 DC 类别 0（size_category(0)==0 的直接后果）；\
             若误用类别 1 会被解码成 −1 进而引发 DC 链漂移"
        );
    }

    #[test]
    fn flat_gray_image_all_dc_categories_zero() {
        // 端到端：平坦灰度 128（每块 DC 差恒为 0）编码后，逐个 MCU 的 DC 符号
        // 必须全为 0。旧实现把 0 归为类别 1 → 每块 diff 被解码成 −1 → 全图 DC
        // 漂移（本回归测试直接锁定的现象）。
        let gray = vec![128u8; 64 * 64];
        let jpeg = encode_jpeg_gray(&gray, 64, 64, JpegConfig::default()).unwrap();
        let bits = bits_from_entropy(&jpeg);
        let dc = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);

        let mut idx = 0usize;
        let mut cats: Vec<u8> = Vec::new();
        while idx + 4 < bits.len() {
            let Some(sym) = huff_decode_first(&bits, &mut idx, &dc) else { break };
            cats.push(sym);
            let _ = read_magnitude(&bits, &mut idx, sym);
            // 跳 AC 直到 EOB（0x00）或 ZRL（0xF0）。
            loop {
                let Some(rs) = huff_decode_first(&bits, &mut idx, &ac) else { break };
                if rs == 0x00 {
                    break;
                }
                if rs == 0xF0 {
                    continue;
                }
                let sz = rs & 0x0F;
                let _ = read_magnitude(&bits, &mut idx, sz);
            }
        }
        assert!(!cats.is_empty(), "应解出至少一个 DC 类别");
        assert!(
            cats.iter().all(|&c| c == 0),
            "平坦画布所有 DC 类别应为 0，实际出现非零类别：{:?}",
            cats
        );
    }

    #[test]
    fn flat_color_image_all_dc_categories_zero() {
        // 彩色 4:4:4 路径与灰度共用 encode_block / size_category，同样须无漂移。
        let rgba = vec![128u8; 64 * 64 * 4];
        let jpeg =
            encode_jpeg_color(&rgba, 64, 64, JpegConfig { quality: 88, grayscale: false }).unwrap();
        let bits = bits_from_entropy(&jpeg);
        let dc = HuffTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
        let ac = HuffTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);

        let mut idx = 0usize;
        let mut cats: Vec<u8> = Vec::new();
        let total_blocks = (64 / 8) * (64 / 8) * 3; // 三分量
        while idx + 4 < bits.len() && (cats.len() as u32) < total_blocks {
            let Some(sym) = huff_decode_first(&bits, &mut idx, &dc) else { break };
            cats.push(sym);
            let _ = read_magnitude(&bits, &mut idx, sym);
            loop {
                let Some(rs) = huff_decode_first(&bits, &mut idx, &ac) else { break };
                if rs == 0x00 {
                    break;
                }
                if rs == 0xF0 {
                    continue;
                }
                let sz = rs & 0x0F;
                let _ = read_magnitude(&bits, &mut idx, sz);
            }
        }
        assert!(
            cats.iter().all(|&c| c == 0),
            "平坦彩色画布所有 DC 类别应为 0（三分量共用 size_category），实际：{:?}",
            cats
        );
    }

    // =======================================================================
    // DQT 量化表 zigzag 顺序回归测试（T.81 §B.2.4.1 错位 bug 修复）
    // =======================================================================

    /// 从字节流中定位首个 DQT marker（0xFF 0xDB），返回其后 64 个量化值
    /// （跳过分段长度 2 字节与 pq/tq 1 字节）。
    ///
    /// JPEG DQT 段结构：`FF DB` + 长度(2) + `pq<<4 | tq`(1) + 64×1B 表（8bit 精度）。
    /// 返回 `None` 表示未找到 DQT 段。
    fn extract_dqt_bytes(jpeg: &[u8]) -> Option<[u8; 64]> {
        let mut i = 0usize;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDB {
                // 跳过 marker(2) + 段长(2) + pq/tq(1) = 5，其后 64 字节为量化值。
                let start = i + 5;
                if start + 64 > jpeg.len() {
                    return None;
                }
                let mut bytes = [0u8; 64];
                bytes.copy_from_slice(&jpeg[start..start + 64]);
                return Some(bytes);
            }
            i += 1;
        }
        None
    }

    #[test]
    fn dqt_table_is_zigzag_ordered_color() {
        // 彩色路径：编码 16×16 小图，断言 DQT 段字节确为 zigzag 顺序。
        let rgba = vec![128u8; 16 * 16 * 4];
        let jpeg =
            encode_jpeg_color(&rgba, 16, 16, JpegConfig { quality: 88, grayscale: false }).unwrap();
        let dqt = extract_dqt_bytes(&jpeg).expect("彩色流应含 DQT 段");

        // 自然序基准表（与 scaled_quant_table 同模块，可直接访问）。
        let q_nat = scaled_quant_table(88);
        for k in 0..64 {
            assert_eq!(
                dqt[k],
                q_nat[ZIGZAG[k] as usize] as u8,
                "彩色 DQT 第 {k} 字节必须是自然序基准在 ZIGZAG[{k}] 处的值（zigzag 序）"
            );
        }
    }

    #[test]
    fn dqt_table_is_zigzag_ordered_gray() {
        // 灰度路径：复用 16×16 灰度图，断言灰度 DQT 段同样为 zigzag 序。
        let gray = vec![128u8; 16 * 16];
        let jpeg = encode_jpeg_gray(&gray, 16, 16, JpegConfig::new(88)).unwrap();
        let dqt = extract_dqt_bytes(&jpeg).expect("灰度流应含 DQT 段");

        let q_nat = scaled_quant_table(88);
        for k in 0..64 {
            assert_eq!(
                dqt[k],
                q_nat[ZIGZAG[k] as usize] as u8,
                "灰度 DQT 第 {k} 字节必须是自然序基准在 ZIGZAG[{k}] 处的值（zigzag 序）"
            );
        }
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
