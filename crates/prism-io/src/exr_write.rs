//! EXR 写入器 —— 与 [`crate::exr`] 的读取器对称（U7 收口 · 路L 补完）。
//!
//! # 覆盖范围
//!
//! - **flat 扫描线压缩**：`None` / `Rle` / `Zip` / `Zips` / `Piz` / `Pxr24` /
//!   `B44` / `B44A`（复用 [`crate::exr_compress`] 的编码器）+
//!   `Dwaa` / `Dwab`（W11 路B 补完，见 [`crate::exr_dwa::dwa_compress`]）；
//! - **flat 分块（tiled）**：[`write_tiled_image`]（W14 路2 补完）——同样的十种压缩，
//!   `ONE_LEVEL` 单层 tile 网格，**边缘 tile 按裁剪尺寸写**；
//! - **deep 扫描线**：[`write_deep_image`]，压缩支持 `None` / `Rle` / `Zips` /
//!   `Zip`（逐扫描线一块）；
//! - **deep tiled**、multipart → **结构化拒绝**（见 [`check_writable`] 的理由）。
//!
//! # 分块（tiled）布局规格（W14 路2 · 全部经官方 OpenEXR 3.1.5 实测反解）
//!
//! | 环节 | 本写入器的口径 | 官方实测依据 |
//! |---|---|---|
//! | `tiles` 属性 | `xSize(u32) + ySize(u32) + mode(u8)`，共 **9 字节**，`mode = 0`（ONE_LEVEL） | `ImfTileDescription` 默认 `xSize=32,ySize=32,ONE_LEVEL`；实测属性 `size == 9` |
//! | 版本字段 | `2 | TILED(0x200)` | 实测 `0x202` |
//! | tile 块头 | `tileX(i32) tileY(i32) levelX(i32) levelY(i32) dataSize(i32)` | 官方 3.1.5 写出的 4 个 chunk 逐个反解，`levelX = levelY = 0` |
//! | `tileX`/`tileY` | **tile 索引**（0 基），**不是**像素坐标 | dataWindow 原点平移到 `(10,20)` 后实测四块仍是 `(0,0)(1,0)(0,1)(1,1)` |
//! | 块序 | 行主序 `block = ty * tiles_x + tx` | 实测 chunk 顺序 `(0,0)(1,0)(0,1)(1,1)` |
//! | **边缘 tile** | 负载**只含裁剪后**的 `tw × th` | 64×48 图 + 32×32 tile：上排 `len=4096`（32×32×4B），下排 `len=2048`（32×**16**×4B） |
//!
//! **tile 尺寸无「32 的倍数」硬约束**（任务书假设已被实测推翻）：官方 3.1.5 对
//! `1×1` / `3×5` / `17×32` / `32×17` / `33×32` 一律**正常写出并读回**，无异常。
//! 故本写入器只校验「非 0」，不强制对齐；需要跨实现最高兼容时由调用方选 32 的倍数。
//!
//! # deflate 实现层级（重要声明）
//!
//! [`deflate_zlib`] 是一条 **真压缩** zlib 流：DEFLATE 层同时具备
//!
//! - **fixed Huffman（BTYPE=01）** —— 默认路径，LZ77 回溯 + RFC 1951 §3.2.6
//!   的固定码表；
//! - **stored（BTYPE=00）** —— 回退路径（输入过短或不可压缩时，装进 stored 更小）。
//!
//! 二者取**体积较小**者输出（相等时优先 stored，保证短输入的字节稳定性）。
//! 合法性判据：能被 [`crate::exr`]（底层 [`prism_bytes::inflate`]，全引擎唯一
//! DEFLATE 解码真源）逐位解回。
//!
//! # 确定性
//!
//! - 属性顺序**固定**（见 [`write_scanline_image`]），不依赖 `HashMap`；
//! - 多通道写入顺序 = [`ExrHeader::channels`] 的顺序；
//! - deflate 的 LZ77 匹配用**定序线性扫描 + 定长回溯窗口**（无哈希随机化，
//!   无 `HashMap`），同输入必同字节；
//! - 同输入连写两遍，字节流 `==`（有测试断言）。

use crate::exr::{
    chan_err, comp_err, rle_compress, struct_err, ChannelPlane, Compression, DataWindow,
    DeepPixel, ExrChannel, ExrHeader, ExrResult, PixelType, TileDesc,
};
use crate::exr_compress::{b44_compress, piz_compress, pxr24_compress, BlockCtx};
use crate::fbuf::f32_to_f16;
// ===========================================================================
//  一、DEFLATE 编码（fixed Huffman + stored 回退）
// ===========================================================================

/// LZ77 回溯窗口上限（2^15，DEFLATE 的最大距离之一；取 32K 全窗）。
const DEFLATE_WINDOW: usize = 32768;
/// 单次匹配的最大长度（DEFLATE 长度码上限）。
const DEFLATE_MAX_MATCH: usize = 258;
/// 触发匹配的最小长度（< 3 的回溯不划算）。
const DEFLATE_MIN_MATCH: usize = 3;

/// 逐字节写出检查的位写入器：普通字段 **LSB 优先**，Huffman 码字 **MSB 优先**。
#[derive(Default)]
struct DeflateBits {
    out: Vec<u8>,
    cur: u32,
    nbit: u32,
}

impl DeflateBits {
    fn with_capacity(n: usize) -> Self {
        Self {
            out: Vec::with_capacity(n),
            cur: 0,
            nbit: 0,
        }
    }

    /// 写 `n` 位整数，**低位在前**（RFC 1951 §3.1.1）。
    fn push(&mut self, mut value: u32, n: u32) {
        for _ in 0..n {
            if value & 1 == 1 {
                self.cur |= 1 << self.nbit;
            }
            value >>= 1;
            self.step();
        }
    }

    /// 写一个 Huffman 码字，**高位在前**（RFC 1951 §3.1.1）。
    fn push_msb(&mut self, code: u32, n: u32) {
        for i in (0..n).rev() {
            if (code >> i) & 1 == 1 {
                self.cur |= 1 << self.nbit;
            }
            self.step();
        }
    }

    #[inline]
    fn step(&mut self) {
        self.nbit += 1;
        if self.nbit == 8 {
            self.out.push(self.cur as u8);
            self.cur = 0;
            self.nbit = 0;
        }
    }

    /// 收尾：把不足一字节的填充位补齐（填 0）并回传字节流。
    fn finish(mut self) -> Vec<u8> {
        if self.nbit > 0 {
            self.out.push(self.cur as u8);
        }
        self.out
    }
}

/// fixed Huffman 字面/长度码（RFC 1951 §3.2.6）：返回 `(码字, 位数)`。
fn fixed_lit_code(sym: u16) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym as u32, 8),
        144..=255 => (0x190 + (sym - 144) as u32, 9),
        256..=279 => ((sym - 256) as u32, 7),
        _ => (0xC0 + (sym - 280) as u32, 8),
    }
}

/// 长度码基值（符号 257..=285，RFC 1951 §3.2.5 表）。
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227, 258,
];
/// 长度码附加值位数。
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// 距离码基值（符号 0..=29）。
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// 距离码附加值位数。
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13,
];

/// 长度（3..=258）→ `(码符号, 附加值, 附加位数)`。
fn length_code(len: usize) -> (u16, u32, u32) {
    // 从后往前找第一个 `base <= len` 的码（表升序，small-n 线性扫为 Mini Code 取向）。
    let mut idx = 0usize;
    for (i, &base) in LENGTH_BASE.iter().enumerate() {
        if base as usize <= len {
            idx = i;
        } else {
            break;
        }
    }
    let base = LENGTH_BASE[idx] as usize;
    (
        257 + idx as u16,
        (len - base) as u32,
        LENGTH_EXTRA[idx] as u32,
    )
}

/// 距离（1..=32768）→ `(码符号, 附加值, 附加位数)`。
fn distance_code(dist: usize) -> (u16, u32, u32) {
    let mut idx = 0usize;
    for (i, &base) in DIST_BASE.iter().enumerate() {
        if base as usize <= dist {
            idx = i;
        } else {
            break;
        }
    }
    let base = DIST_BASE[idx] as usize;
    (idx as u16, (dist - base) as u32, DIST_EXTRA[idx] as u32)
}

/// 3 字节哈希表大小（2^15；`DEFLATE_WINDOW` 同量级，链长受控）。
const HASH_BITS: u32 = 15;
/// 每条哈希链最多回溯的候选数（截断 → 上界时间；确定性不受影响）。
const HASH_CHAIN_LIMIT: usize = 64;

/// 由 3 字节算出定长哈希（**纯函数、无随机源** → 确定性）。
#[inline]
fn hash3(data: &[u8], pos: usize) -> usize {
    let a = data.get(pos).copied().unwrap_or(0) as u32;
    let b = data.get(pos + 1).copied().unwrap_or(0) as u32;
    let c = data.get(pos + 2).copied().unwrap_or(0) as u32;
    // 经典组合（与 zlib 无关的自选常量），取低 `HASH_BITS` 位。
    (((a << 10) ^ (b << 5) ^ c) as usize) & ((1 << HASH_BITS) - 1)
}

/// LZ77 匹配查找器：定长 3 字节哈希链（`head` / `prev`）。
///
/// **确定性**：哈希是纯函数、链按位置定序插入，遍历顺序恒为「就近回溯」；
/// 不依赖 `HashMap` 的随机哈希种子，也不依赖平台指针序 → 同输入必同结果。
struct MatchFinder<'a> {
    data: &'a [u8],
    head: Vec<i32>,
    prev: Vec<i32>,
}

impl<'a> MatchFinder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            head: vec![-1; 1 << HASH_BITS],
            // `prev[pos]` = 同哈希的上一个位置（-1 表示无）。
            prev: vec![-1; data.len()],
        }
    }

    /// 把位置 `pos` 插入哈希链（供后续匹配使用）。
    fn insert(&mut self, pos: usize) {
        if pos + 2 >= self.data.len() {
            return;
        }
        let h = hash3(self.data, pos);
        if let Some(slot) = self.prev.get_mut(pos) {
            *slot = self.head[h];
        }
        if let Some(slot) = self.head.get_mut(h) {
            *slot = pos as i32;
        }
    }

    /// 在 `[window_start, pos)` 内为 `data[pos..]` 找最长回溯匹配。
    ///
    /// 遍历顺序：链上**由近到远**；长度严格更大才更新（等长保留更近的）。
    fn find(&self, pos: usize) -> Option<(usize, usize)> {
        let n = self.data.len();
        if pos + DEFLATE_MIN_MATCH > n {
            return None;
        }
        let max_len = (n - pos).min(DEFLATE_MAX_MATCH);
        let window_start = pos.saturating_sub(DEFLATE_WINDOW);
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut cand = self.head[hash3(self.data, pos)];
        let mut steps = 0usize;
        while cand >= 0 && steps < HASH_CHAIN_LIMIT {
            let c = cand as usize;
            if c < window_start {
                break;
            }
            let dist = pos - c;
            // 先比末位（若候选足够长）以快速否决，再逐字节。
            let mut l = 0usize;
            while l < max_len && c + l < n && self.data[c + l] == self.data[pos + l] {
                l += 1;
            }
            if l > best_len {
                best_len = l;
                best_dist = dist;
                if l == max_len {
                    break;
                }
            }
            cand = self.prev.get(c).copied().unwrap_or(-1);
            steps += 1;
        }
        if best_len >= DEFLATE_MIN_MATCH {
            Some((best_len, best_dist))
        } else {
            None
        }
    }
}

/// 用 **fixed Huffman（BTYPE=01）** 编码一条 DEFLATE 块（含 BFINAL=1）。
///
/// 输出不含 zlib 头/尾（[`deflate_zlib`] 负责包装）。
fn deflate_fixed(payload: &[u8]) -> Vec<u8> {
    let mut w = DeflateBits::with_capacity(payload.len() + payload.len() / 8 + 16);
    // 块头：BFINAL=1（1 位）+ BTYPE=01（2 位，值 1，**低位在前**）。
    w.push(1, 1);
    w.push(1, 2);

    let mut mf = MatchFinder::new(payload);
    let mut pos = 0usize;
    while pos < payload.len() {
        if let Some((len, dist)) = mf.find(pos) {
            let (lsym, lextra, lbits) = length_code(len);
            let (lc, ln) = fixed_lit_code(lsym);
            w.push_msb(lc, ln);
            if lbits > 0 {
                w.push(lextra, lbits);
            }
            let (dsym, dextra, dbits) = distance_code(dist);
            // fixed 距离码恒 5 位、码值 = 符号本身。
            w.push_msb(dsym as u32, 5);
            if dbits > 0 {
                w.push(dextra, dbits);
            }
            // 匹配覆盖的每个位置都要插入哈希链（供后续查找）。
            let end = pos + len;
            while pos < end {
                mf.insert(pos);
                pos += 1;
            }
        } else {
            let (lc, ln) = fixed_lit_code(payload[pos] as u16);
            w.push_msb(lc, ln);
            mf.insert(pos);
            pos += 1;
        }
    }
    // 块结束符号 256。
    let (ec, en) = fixed_lit_code(256);
    w.push_msb(ec, en);
    w.finish()
}

/// DEFLATE stored 块流（裸 DEFLATE，无 zlib 头；EXR 的 ZIP 块内层）。
///
/// 合法但**无压缩比**；作为 [`deflate_raw_best`] 的回退路径。
pub fn deflate_stored(payload: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(payload.len() + payload.len() / 65535 * 5 + 5);
    if payload.is_empty() {
        // 空负载：单个 stored 块，LEN=0。
        out.push(0x01);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
        return out;
    }
    let mut off = 0usize;
    loop {
        let n = (payload.len() - off).min(65535);
        let last = off + n >= payload.len();
        // BFINAL(1 bit) + BTYPE=00(2 bits)，共 3 位 → 最低 3 位。
        out.push(if last { 0x01 } else { 0x00 });
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&payload[off..off + n]);
        off += n;
        if last {
            break;
        }
    }
    out
}

/// 选**体积较小**的 DEFLATE 表示（fixed vs stored；相等时优先 stored）。
///
/// 返回裸 DEFLATE 码流（无 zlib 头/尾）。
pub fn deflate_raw_best(payload: &[u8]) -> Vec<u8> {
    let fixed = deflate_fixed(payload);
    let stored = deflate_stored(payload);
    if fixed.len() < stored.len() {
        fixed
    } else {
        stored
    }
}

/// Adler-32（RFC 1950）。
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// 生成 zlib 流（RFC 1950）：`0x78 0x01` 头 + 最优 DEFLATE + Adler-32。
///
/// `0x78 0x01` = CM=8(deflate) / CINFO=7(32K 窗口) / FLEVEL=0 / FCHECK 使
/// `(0x78<<8|0x01) % 31 == 0`（`0x7801 = 30721`，`30721 % 31 == 0` ✓）。
///
/// **真压缩**：内部走 [`deflate_raw_best`]（fixed Huffman，短输入回退 stored）。
pub fn deflate_zlib(payload: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(payload.len() + 16);
    out.push(0x78);
    out.push(0x01);
    out.extend_from_slice(&deflate_raw_best(payload));
    out.extend_from_slice(&adler32(payload).to_be_bytes());
    out
}

// ===========================================================================
//  二、块编码分派（接入 PIZ / PXR24 / B44 / B44A）
// ===========================================================================

/// 写入器不支持的压缩类型的说明。
fn writer_rejection(c: Compression) -> &'static str {
    match c {
        Compression::Dwaa => "DWAA（DCT + 量化 + Huf，本写入器只读不写）",
        Compression::Dwab => "DWAB（DCT + 量化 + Huf，本写入器只读不写）",
        _ => "该压缩类型",
    }
}

/// 写入器**是否实现**该压缩类型的编码。
///
/// 与读取侧的 [`Compression::is_supported`] **不同**：读取侧十种全支持，写入侧
/// 做 `None/Rle/Zip/Zips/Piz/Pxr24/B44/B44A` 八种 + `Dwaa/Dwab`（W11 路B 补完）。
pub fn writer_supports(c: Compression) -> bool {
    matches!(
        c,
        Compression::None
            | Compression::Rle
            | Compression::Zip
            | Compression::Zips
            | Compression::Piz
            | Compression::Pxr24
            | Compression::B44
            | Compression::B44A
            | Compression::Dwaa
            | Compression::Dwab
    )
}

/// 按压缩类型编码一个 flat 块的负载。
///
/// `raw` 是块内「扫描线 × 通道 × 像素」的小端原始字节；`ctx` 供 PIZ/PXR24/B44
/// 按通道重排（宽高与通道表来自调用方），DWA 另需「通道名 + 类型 + pLinear」
/// 三元组（通道分类依赖名称后缀）。
fn encode_block(
    compression: Compression,
    raw: &[u8],
    ctx: &BlockCtx<'_>,
) -> ExrResult<Vec<u8>> {
    match compression {
        Compression::None => Ok(raw.to_vec()),
        Compression::Rle => Ok(rle_compress(raw)),
        Compression::Zip | Compression::Zips => Ok(deflate_zlib(raw)),
        Compression::Piz => piz_compress(raw, ctx),
        Compression::Pxr24 => pxr24_compress(raw, ctx),
        Compression::B44 => b44_compress(raw, ctx, false),
        Compression::B44A => b44_compress(raw, ctx, true),
        // DWAA / DWAB（W11 路B）：算法一致，块行数由 `lines_per_block()` 决定。
        Compression::Dwaa | Compression::Dwab => {
            let chans: Vec<(String, PixelType, bool)> = ctx
                .channels
                .iter()
                .map(|c| (c.name.clone(), c.pixel_type, c.p_linear))
                .collect();
            crate::exr_dwa::dwa_compress(raw, &chans, ctx.width, ctx.height)
        }
    }
}

// ===========================================================================
//  三、头部工具
// ===========================================================================

/// 把 f32 量化成通道原始位模式字节。
fn quantize(v: f32, pt: PixelType) -> Vec<u8> {
    match pt {
        PixelType::Half => f32_to_f16(v).to_le_bytes().to_vec(),
        PixelType::Float => v.to_le_bytes().to_vec(),
        PixelType::Uint => {
            // uint 通道：负值钳到 0，NaN 归 0；上界钳到 u32::MAX。
            let clamped = if v.is_nan() {
                0.0f32
            } else {
                v.max(0.0).min(u32::MAX as f32)
            };
            (clamped as u32).to_le_bytes().to_vec()
        }
    }
}

/// 写一个属性（`name\0type\0` + `size(u32)` + `value`）。
fn put_attr(out: &mut Vec<u8>, name: &str, ty: &str, value: &[u8]) {
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(ty.as_bytes());
    out.push(0);
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
}

/// 组装 `channels` 属性负载（顺序 = 传入顺序，确定性）。
fn channels_attr(chs: &[ExrChannel]) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    for c in chs {
        v.extend_from_slice(c.name.as_bytes());
        v.push(0);
        v.extend_from_slice(&c.pixel_type.code().to_le_bytes());
        v.push(u8::from(c.p_linear));
        v.extend_from_slice(&[0, 0, 0]); // reserved
        v.extend_from_slice(&c.x_sampling.to_le_bytes());
        v.extend_from_slice(&c.y_sampling.to_le_bytes());
    }
    v.push(0); // 空名终止
    v
}

/// 组装 `box2i` 属性负载。
fn box2i_attr(win: DataWindow) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(16);
    for x in [win.x_min, win.y_min, win.x_max, win.y_max] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// `tiles` 属性负载（`xSize(u32) + ySize(u32) + mode(u8)`，共 9 字节）。
///
/// `mode` 取 `0` = ONE_LEVEL（`ImfTileDescription` 的 `LevelMode::ONE_LEVEL`）。
/// **不写 `roundingMode`**——官方写出的属性 `size` 实测恰为 9，多一个字节会让
/// 部分严格实现判「属性长度非法」。
fn tiles_attr(td: &TileDesc) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(9);
    v.extend_from_slice(&td.x_size.to_le_bytes());
    v.extend_from_slice(&td.y_size.to_le_bytes());
    v.push(td.level_mode);
    v
}

/// 拼装公共属性表（`channels`…`screenWindowWidth`），返回头部字节（不含偏移表）。
///
/// 属性顺序**固定**（确定性输出的前提）。`tiled` 为真时在 `displayWindow` 之后
/// 追加 `tiles` 属性（EXR 属性表无顺序要求，但固定顺序是确定性的前提）。
fn write_header(
    header: &ExrHeader,
    version_flags: u32,
) -> ExrResult<Vec<u8>> {
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(&crate::exr::EXR_MAGIC.to_le_bytes());
    head.extend_from_slice(&(2u32 | version_flags).to_le_bytes());
    put_attr(&mut head, "channels", "chlist", &channels_attr(&header.channels));
    put_attr(
        &mut head,
        "compression",
        "compression",
        &[header.compression.code()],
    );
    put_attr(&mut head, "dataWindow", "box2i", &box2i_attr(header.data_window));
    put_attr(&mut head, "displayWindow", "box2i", &box2i_attr(header.display_window));
    // tiled：`tiles` 属性紧随 `displayWindow`（官方亦如此排列）。
    if header.tiled {
        let td = header.tile.ok_or_else(|| {
            struct_err(
                "exr_write::write_header",
                "分块头部缺 `tiles` 描述",
                "header.tiled == true 时 tile 字段必须给出 x_size / y_size",
            )
        })?;
        put_attr(&mut head, "tiles", "tiledesc", &tiles_attr(&td));
    }
    put_attr(&mut head, "lineOrder", "lineOrder", &[header.line_order]);
    put_attr(&mut head, "pixelAspectRatio", "float", &1.0f32.to_le_bytes());
    put_attr(
        &mut head,
        "screenWindowCenter",
        "v2f",
        &[0.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat(),
    );
    put_attr(&mut head, "screenWindowWidth", "float", &1.0f32.to_le_bytes());
    head.push(0); // 属性表结束
    Ok(head)
}

/// 由头部与块负载 + 每块额外前导字节组装最终文件（偏移表 + 体）。
///
/// `prefix_of_block(b)` 回传第 `b` 块的**前导字节**（在负载之前写入 chunk 的部分，
/// 例如扫描线块的 `y + size`），`payloads[b]` 是块负载。
/// `sentinel` 为真时偏移表末尾补一个哨兵条目（deep 需要 `块数 + 1`）。
fn assemble(head: Vec<u8>, payloads: &[Vec<u8>], sentinel: bool) -> Vec<u8> {
    let entries = payloads.len() + usize::from(sentinel);
    let table_off = head.len();
    let body_start = table_off + entries * 8;
    let mut table: Vec<u8> = Vec::with_capacity(entries * 8);
    let mut body: Vec<u8> = Vec::new();
    let mut at = body_start;
    for p in payloads {
        table.extend_from_slice(&(at as u64).to_le_bytes());
        body.extend_from_slice(p);
        at += p.len();
    }
    if sentinel {
        table.extend_from_slice(&(at as u64).to_le_bytes());
    }
    let mut out = head;
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

// ===========================================================================
//  四、flat 写入（扫描线 + 分块）
// ===========================================================================

/// 校验 flat 头部可写性（扫描线入口用：tiled 由 [`write_tiled_image`] 处理）。
///
/// `check_writable` 是**扫描线**入口的前置；tiled 不再被结构性拒绝，而是由调用方
/// 显式选择 [`write_tiled_image`]（布局是头部的属性，不该被静默改写）。
fn check_writable(header: &ExrHeader) -> ExrResult<()> {
    if header.tiled {
        return Err(struct_err(
            "exr_write::check_writable",
            "这是分块（tiled）图像，请用 `write_tiled_image`",
            "`write_scanline_image` 只产出扫描线布局；分块请走 tiled 入口",
        ));
    }
    check_flat_common(header)
}

/// flat 图像（扫描线 / 分块）共用的校验：非 deep、单部件、压缩受支持。
fn check_flat_common(header: &ExrHeader) -> ExrResult<()> {
    if header.deep {
        return Err(struct_err(
            "exr_write::check_flat_common",
            "deep EXR 请用 `write_deep_image`",
            "flat 写入不处理逐像素样本表；deep 用专用入口",
        ));
    }
    if header.multipart {
        return Err(struct_err(
            "exr_write::check_flat_common",
            "多部件 EXR 写入未实装",
            "本写入器只产出单部件文件；多部件需逐部件属性表与 chunk 索引，\
             且本引擎读取器同样结构化拒绝 MULTI_PART（写了也读不回）",
        ));
    }
    if !writer_supports(header.compression) {
        return Err(comp_err(
            "exr_write::check_flat_common",
            format!(
                "压缩类型 {}（{}）本写入器不实现",
                header.compression.code(),
                writer_rejection(header.compression)
            ),
            "请改用 None / Rle / Zip / Zips / Piz / Pxr24 / B44 / B44A / Dwaa / Dwab",
        ));
    }
    check_channels_and_window(header)
}

/// 校验分块（tiled）头部可写性。
///
/// # 校验项
///
/// - `header.tiled == true`（否则应走扫描线入口）；
/// - `header.tile` 存在且 `x_size`/`y_size` **非 0**（0 会让 `tiles_x/y` 除零）；
/// - `level_mode == 0`（ONE_LEVEL）—— MIPMAP/RIPMAP 需多层 tile 偏移，
///   本写入器**未实装**，明确拒绝而非静默按单层写。
///
/// **不校验 tile 尺寸是否 32 的倍数**：官方 OpenEXR 3.1.5 实测对 `1×1` / `3×5` /
/// `17×32` / `32×17` / `33×32` 一律正常写出并读回，无任何异常（见模块级规格表）。
fn check_tiled_writable(header: &ExrHeader) -> ExrResult<()> {
    const LOC: &str = "exr_write::check_tiled_writable";
    if !header.tiled {
        return Err(struct_err(
            LOC,
            "`write_tiled_image` 需要 `header.tiled == true`",
            "扫描线图像请用 `write_scanline_image`（或 `header_for_tiled` 构造头部）",
        ));
    }
    let td = header.tile.ok_or_else(|| {
        struct_err(
            LOC,
            "分块头部缺 `tiles` 描述",
            "header.tiled == true 时必须给出 tile 尺寸（见 `header_for_tiled`）",
        )
    })?;
    if td.x_size == 0 || td.y_size == 0 {
        return Err(struct_err(
            LOC,
            format!("tile 尺寸为 {}×{}（须均 ≥ 1）", td.x_size, td.y_size),
            "0 会让 tile 网格列数/行数除零；请给出合法的 tile 尺寸",
        ));
    }
    if td.level_mode != 0 {
        return Err(struct_err(
            LOC,
            format!(
                "level_mode = {}（{}），本写入器只实装 ONE_LEVEL(0)",
                td.level_mode,
                match td.level_mode {
                    1 => "MIPMAP_LEVELS",
                    2 => "RIPMAP_LEVELS",
                    _ => "未知",
                }
            ),
            "多层 tile 需每层各自的偏移表与 ROUND_UP/DOWN 尺寸推导，本写入器未实装；\
             请展平为单层（ONE_LEVEL）",
        ));
    }
    check_flat_common(header)
}

/// 通道表 / 数据窗口 / 采样率的公共校验（flat 与 deep 共用）。
fn check_channels_and_window(header: &ExrHeader) -> ExrResult<()> {
    if header.channels.is_empty() {
        return Err(chan_err(
            "exr_write::check_writable",
            "通道表为空",
            "至少需要一个通道才能写出 EXR",
        ));
    }
    for c in &header.channels {
        if c.x_sampling != 1 || c.y_sampling != 1 {
            return Err(chan_err(
                "exr_write::check_writable",
                format!("通道 `{}` 的采样为 {}×{}（只支持 1×1）", c.name, c.x_sampling, c.y_sampling),
                "请用全分辨率通道；子采样需重定义值域",
            ));
        }
    }
    let win = header.data_window;
    if win.width() == 0 || win.height() == 0 {
        return Err(struct_err(
            "exr_write::check_writable",
            format!("dataWindow 宽高为 0（{}×{}）", win.width(), win.height()),
            "空图像无法写出；确认属性正确",
        ));
    }
    Ok(())
}

/// 把平面按通道名对齐到头部通道顺序，并校验尺寸。
fn align_planes<'a>(
    header: &ExrHeader,
    planes: &'a [ChannelPlane],
) -> ExrResult<Vec<&'a ChannelPlane>> {
    const LOC: &str = "exr_write::align_planes";
    let w = header.data_window.width() as usize;
    let h = header.data_window.height() as usize;
    let mut out: Vec<&ChannelPlane> = Vec::with_capacity(header.channels.len());
    for c in &header.channels {
        let p = planes.iter().find(|p| p.name == c.name).ok_or_else(|| {
            chan_err(
                LOC,
                format!("缺少通道 `{}` 的平面数据", c.name),
                "平面列表必须覆盖头部声明的每个通道",
            )
        })?;
        if p.width as usize != w || p.height as usize != h {
            return Err(chan_err(
                LOC,
                format!(
                    "通道 `{}` 平面尺寸 {}×{}，期望 {w}×{h}",
                    c.name, p.width, p.height
                ),
                "平面尺寸必须与 dataWindow 一致",
            ));
        }
        if p.values.len() != w * h {
            return Err(chan_err(
                LOC,
                format!(
                    "通道 `{}` 平面数据 {} 项，期望 {} 项",
                    c.name,
                    p.values.len(),
                    w * h
                ),
                "平面数据长度必须等于 width × height",
            ));
        }
        out.push(p);
    }
    Ok(out)
}

/// 把一个块（`lines` 行、自 `first_line` 起）打成「扫描线 × 通道 × 像素」小端字节。
fn build_block_raw(
    header: &ExrHeader,
    ordered: &[&ChannelPlane],
    first_line: usize,
    lines: usize,
) -> Vec<u8> {
    let width = header.data_window.width() as usize;
    let per_row: usize = header
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width())
        .sum();
    let mut raw: Vec<u8> = Vec::with_capacity(per_row * width * lines);
    for line in 0..lines {
        let y = first_line + line;
        for (ci, c) in header.channels.iter().enumerate() {
            let plane: &[f32] = ordered.get(ci).map(|p| p.values.as_slice()).unwrap_or(&[]);
            for x in 0..width {
                let v = plane.get(y * width + x).copied().unwrap_or(0.0);
                raw.extend_from_slice(&quantize(v, c.pixel_type));
            }
        }
    }
    raw
}

/// 写出一个 flat 扫描线 EXR（`header.compression` 决定压缩）。
///
/// 属性顺序**固定**（见 [`write_header`]）：`channels` → `compression` →
/// `dataWindow` → `displayWindow` → `lineOrder` → `pixelAspectRatio` →
/// `screenWindowCenter` → `screenWindowWidth` → 属性表结束。
///
/// 块序 = 偏移表序 = 升序 y；每块头 = `y(i32)` + `dataSize(i32)` + 负载。
pub fn write_scanline_image(header: &ExrHeader, planes: &[ChannelPlane]) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr_write::write_scanline_image";
    check_writable(header)?;
    let ordered = align_planes(header, planes)?;

    let head = write_header(header, 0)?;

    let lpb = header.compression.lines_per_block() as usize;
    if lpb == 0 {
        return Err(comp_err(
            LOC,
            format!("压缩类型 {} 的块高为 0", header.compression.code()),
            "分派表与 lines_per_block 不一致",
        ));
    }
    let dwin = header.data_window;
    let height = dwin.height() as usize;
    let width = dwin.width() as usize;
    let blocks = height.div_ceil(lpb);

    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let first_line = b * lpb;
        let lines = (height - first_line).min(lpb);
        let raw = build_block_raw(header, &ordered, first_line, lines);
        let ctx = BlockCtx {
            channels: &header.channels,
            width,
            height: lines,
        };
        let payload = encode_block(header.compression, &raw, &ctx)?;
        let mut chunk: Vec<u8> = Vec::with_capacity(payload.len() + 8);
        // 块首行号用 **dataWindow 绝对坐标**（读取侧据此校验一致性）。
        chunk.extend_from_slice(&(dwin.y_min + first_line as i32).to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        chunk.extend_from_slice(&payload);
        chunks.push(chunk);
    }
    Ok(assemble(head, &chunks, false))
}

/// 写出一个 flat 扫描线 EXR（自动选择压缩；`compression` 为 `None` 时走无压缩）。
///
/// 等价于把 `header.compression` 设为 `compression` 后调 [`write_scanline_image`]。
pub fn write_image(
    header: &ExrHeader,
    planes: &[ChannelPlane],
    compression: Compression,
) -> ExrResult<Vec<u8>> {
    let mut h = header.clone();
    h.compression = compression;
    // 块高随压缩变化，`chunk_offsets` 无意义 → 清空以免误导。
    h.chunk_offsets.clear();
    write_scanline_image(&h, planes)
}

// ===========================================================================
//  四之二、flat 分块（tiled）写入 —— W14 路2
// ===========================================================================

/// tile 网格中第 `(tx, ty)` 个 tile 的**裁剪后**像素尺寸。
///
/// ⚠️ **边缘 tile 必须按裁剪尺寸写**（W13 坑#3，`ImfDwaCompressor.cpp:689`）：
/// 官方按 `ceil(_height / 8.0)` 计算 DCT 块行数，其中 `_height` 是**裁剪后**的行数。
/// 若按完整 tile 高度写，末行/末列 tile 会多写数据，读端索要块数与文件不自洽。
///
/// 返回 `(tw, th)`（像素数，可能小于 `x_size × y_size`）。
fn clipped_tile_size(win: DataWindow, td: &TileDesc, tx: u32, ty: u32) -> (usize, usize) {
    let x0 = win.x_min + (tx as i32) * (td.x_size as i32);
    let y0 = win.y_min + (ty as i32) * (td.y_size as i32);
    // 网格由 `tiles_x/y` 推出 ⇒ tile 必与窗口相交，此处 `max(0)` 只是防御。
    let tw = (win.x_max - x0 + 1).min(td.x_size as i32).max(0) as usize;
    let th = (win.y_max - y0 + 1).min(td.y_size as i32).max(0) as usize;
    (tw, th)
}

/// 切出第 `(tx, ty)` 个 tile 的原始字节（`tw × th` × 通道，**裁剪后**尺寸）。
///
/// 行主序取自整幅平面：`plane[(y0 + dy) * img_w + (x0 + dx)]`。
fn build_tile_raw(
    header: &ExrHeader,
    ordered: &[&ChannelPlane],
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<u8> {
    let img_w = header.data_window.width() as usize;
    let per_row: usize = header
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width())
        .sum();
    let mut raw: Vec<u8> = Vec::with_capacity(per_row * tw * th);
    for dy in 0..th {
        for (ci, c) in header.channels.iter().enumerate() {
            let plane: &[f32] = ordered.get(ci).map(|p| p.values.as_slice()).unwrap_or(&[]);
            for dx in 0..tw {
                let idx = (y0 + dy) * img_w + (x0 + dx);
                let v = plane.get(idx).copied().unwrap_or(0.0);
                raw.extend_from_slice(&quantize(v, c.pixel_type));
            }
        }
    }
    raw
}

/// 写出一个 **flat 分块（tiled）** EXR（`header.compression` 决定压缩）。
///
/// # 布局规格（全部经官方 OpenEXR 3.1.5 实测反解，见模块级文档）
///
/// - 版本字段置 `TILED(0x200)`，并写 `tiles` 属性（`xSize/ySize/mode`，共 9 字节）；
/// - 块序 = 行主序 `block = ty * tiles_x + tx`（与 [`crate::exr::read_tile`] 一致）；
/// - 每块头 = `tileX(i32) tileY(i32) levelX(i32) levelY(i32) dataSize(i32)`，
///   其中 `tileX/tileY` 是 **0 基 tile 索引**（**不是**像素坐标——官方在
///   dataWindow 原点平移后仍写索引）、`levelX = levelY = 0`；
/// - **边缘 tile 的负载只含裁剪后的 `tw × th`**（W13 坑#3）。
///
/// # 支持范围
///
/// 十种压缩全部支持（与扫描线共用 [`encode_block`]）。`ONE_LEVEL` 单层。
/// MIPMAP / RIPMAP 多层、以及 deep tiled，见 [`check_tiled_writable`] 的明确拒绝。
pub fn write_tiled_image(
    header: &ExrHeader,
    planes: &[ChannelPlane],
) -> ExrResult<Vec<u8>> {
    check_tiled_writable(header)?;
    let ordered = align_planes(header, planes)?;
    let td = header.tile.expect("check_tiled_writable 已保证 tile 为 Some");

    let head = write_header(header, crate::exr::version_flags::TILED)?;

    let win = header.data_window;
    let cols = td.tiles_x(&win);
    let rows = td.tiles_y(&win);
    let tiles = td
        .tile_count(&win)
        .ok_or_else(|| struct_err("exr_write::write_tiled_image", "tile 数量溢出", "使用更小的图像或更小的 tile"))?;

    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(tiles);
    // 行主序：ty 外层、tx 内层（官方 chunk 顺序实测为 (0,0)(1,0)(0,1)(1,1)）。
    for ty in 0..rows {
        for tx in 0..cols {
            let (tw, th) = clipped_tile_size(win, &td, tx, ty);
            let (px0, py0) = (
                tx as usize * td.x_size as usize,
                ty as usize * td.y_size as usize,
            );
            let raw = build_tile_raw(header, &ordered, px0, py0, tw, th);
            // 压缩上下文用**裁剪后**的 tile 宽高——PIZ/PXR24/B44/DWA 都按
            // `ctx.width/height` 重排，用完整 tile 尺寸会让边缘 tile 排布错位。
            let ctx = BlockCtx {
                channels: &header.channels,
                width: tw,
                height: th,
            };
            let payload = encode_block(header.compression, &raw, &ctx)?;
            let mut chunk: Vec<u8> = Vec::with_capacity(payload.len() + 20);
            chunk.extend_from_slice(&(tx as i32).to_le_bytes());
            chunk.extend_from_slice(&(ty as i32).to_le_bytes());
            chunk.extend_from_slice(&0i32.to_le_bytes()); // levelX
            chunk.extend_from_slice(&0i32.to_le_bytes()); // levelY
            chunk.extend_from_slice(&(payload.len() as i32).to_le_bytes());
            chunk.extend_from_slice(&payload);
            chunks.push(chunk);
        }
    }
    Ok(assemble(head, &chunks, false))
}

/// 写出一个 flat **分块** EXR（自动选择压缩）。
///
/// 等价于把 `header.compression` 设为 `compression` 后调 [`write_tiled_image`]。
pub fn write_tiled_image_with(
    header: &ExrHeader,
    planes: &[ChannelPlane],
    compression: Compression,
) -> ExrResult<Vec<u8>> {
    let mut h = header.clone();
    h.compression = compression;
    h.chunk_offsets.clear();
    write_tiled_image(&h, planes)
}

// ===========================================================================
//  五、deep 扫描线写入
// ===========================================================================

/// deep 写入器支持的压缩类型（逐扫描线一块）。
///
/// deep 块内布局是「像素序 → 样本序」，与 PIZ/PXR24/B44 的「扫描线 × 通道 × 像素」
/// 重排不兼容，故 deep 只做**字节级**压缩：`None` / `Rle` / `Zips` / `Zip`。
fn deep_supports(c: Compression) -> bool {
    matches!(
        c,
        Compression::None | Compression::Rle | Compression::Zips | Compression::Zip
    )
}

/// 校验 deep 头部可写性。
fn check_deep_writable(header: &ExrHeader) -> ExrResult<()> {
    if header.tiled {
        // ⚠️ 理由必须写准：**格式本身完全支持** deep tiled（官方 OpenEXR 3.1.5
        // 可写可读），本模块**未实装**的是 deep tiled 的**样本表布局**——
        // `write_deep_image` 的块内布局是「扫描线 × 像素序 → 样本序」，
        // 而 tiled 还需按 tile 边界切分样本计数表并逐 tile 建索引。
        // 参照：flat **非** deep 的 tiled 已由 `write_tiled_image` 实装。
        return Err(struct_err(
            "exr_write::check_deep_writable",
            "deep tiled EXR 写入未实装（本模块只实装 flat tiled）",
            "deep tiled 的逐 tile 样本表切分与索引未实装；\
             flat 分块请用 `write_tiled_image`，deep 扫描线请用 `write_deep_image`",
        ));
    }
    if !header.deep {
        return Err(struct_err(
            "exr_write::check_deep_writable",
            "`write_deep_image` 需要 `header.deep == true`",
            "flat 图像请用 `write_scanline_image`",
        ));
    }
    if header.multipart {
        return Err(struct_err(
            "exr_write::check_deep_writable",
            "多部件 deep EXR 写入未实现",
            "本写入器只产出单部件文件",
        ));
    }
    if !deep_supports(header.compression) {
        return Err(comp_err(
            "exr_write::check_deep_writable",
            format!(
                "deep 不支持压缩类型 {}（{}）",
                header.compression.code(),
                writer_rejection(header.compression)
            ),
            "deep 只支持 None / Rle / Zips / Zip（逐扫描线字节级压缩）",
        ));
    }
    check_channels_and_window(header)
}

/// 写出一个 **deep 扫描线** EXR，可被 [`crate::exr::read_deep_image`] 逐位读回。
///
/// # 参数
///
/// - `header`：`deep` 必须为 `true`；`compression` ∈ {None, Rle, Zips, Zip}；
/// - `pixels`：长度须等于 `width × height`，行主序、行内 x 升序；每像素的样本按
///   **最近优先**（`index` 从 0 递增）。样本的 `values` 顺序 = `header.channels`。
///
/// # 块布局（本模块 writer/reader 对称口径）
///
/// 每块 = 1 条扫描线：
/// ```text
/// y(i32)
/// packedOffsetOfSampleCountTable(u64)   // 恒 20，相对块首（保留字段）
/// packedSizeOfSampleTable(u64)          // 样本数表压缩后长度
/// [样本数表：None → 原样 width×4；否则整表压缩]
/// [通道偏移表：C × i64，相对「数据区起点」]
/// [通道数据：None → 原样；否则整段压缩]
/// ```
///
/// 偏移表末尾补一个哨兵条目（deep 需 `块数 + 1` 项）。
pub fn write_deep_image(header: &ExrHeader, pixels: &[DeepPixel]) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr_write::write_deep_image";
    check_deep_writable(header)?;

    let dwin = header.data_window;
    let width = dwin.width() as usize;
    let height = dwin.height() as usize;
    let expect = width
        .checked_mul(height)
        .ok_or_else(|| struct_err(LOC, "deep 像素数溢出", "使用更小的图像"))?;
    if pixels.len() != expect {
        return Err(chan_err(
            LOC,
            format!("deep 像素数 {} 与 dataWindow {}×{} 不符", pixels.len(), width, height),
            "像素列表须覆盖 dataWindow 的每个像素（行主序）",
        ));
    }
    let count = header.channels.len();
    if count == 0 {
        return Err(chan_err(LOC, "deep 图像无通道", "至少需要一个通道"));
    }

    // 头部：deep 标志位置位。
    let head = write_header(header, crate::exr::version_flags::DEEP_DATA)?;

    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(height);
    for y in 0..height {
        // ① 本行各像素的样本数。
        let mut counts: Vec<u32> = Vec::with_capacity(width);
        for x in 0..width {
            let idx = y * width + x;
            let n = pixels.get(idx).map(|p| p.samples.len()).unwrap_or(0);
            counts.push(n.min(u32::MAX as usize) as u32);
        }
        // ② 样本数表原始字节（width × u32）。
        let mut sc_raw: Vec<u8> = Vec::with_capacity(width * 4);
        for &n in &counts {
            sc_raw.extend_from_slice(&n.to_le_bytes());
        }
        // ③ 通道数据原始字节：逐通道连续，通道内「像素序 → 样本序」展平。
        let mut chan_raw: Vec<u8> = Vec::new();
        for (ci, ch) in header.channels.iter().enumerate() {
            for x in 0..width {
                let idx = y * width + x;
                let px = pixels.get(idx);
                let ns = counts.get(x).copied().unwrap_or(0) as usize;
                for s in 0..ns {
                    let v = px
                        .and_then(|p| p.samples.get(s))
                        .and_then(|samp| samp.values.get(ci))
                        .copied()
                        .unwrap_or(0.0);
                    chan_raw.extend_from_slice(&quantize(v, ch.pixel_type));
                }
            }
        }
        // ④ 通道偏移表。偏移基准与**读取侧口径**严格对齐：
        //    - 无压缩：相对 `data_pos`（偏移表起点） → 首通道偏移 = 偏移表长度；
        //    - 压缩  ：相对**解压后通道数据区起点** → 首通道偏移 = 0。
        let table_bytes = count * 8;
        let total_samples: usize = counts.iter().map(|&c| c as usize).sum();
        let mut offs: Vec<u8> = Vec::with_capacity(table_bytes);
        let mut rel = if header.compression == Compression::None {
            table_bytes as u64
        } else {
            0u64
        };
        for ci in 0..count {
            offs.extend_from_slice(&(rel as i64).to_le_bytes());
            // 该通道未压缩长度 = 本行该通道样本总数 × 每样本字节数。
            let ch_bytes = total_samples
                .saturating_mul(header.channels.get(ci).map(|c| c.pixel_type.byte_width()).unwrap_or(0));
            rel += ch_bytes as u64;
        }
        // ⑤ 逐段压缩（None → 原样；否则整段字节级压缩）。
        let sc_packed: Vec<u8> = if header.compression == Compression::None {
            sc_raw.clone()
        } else {
            encode_deep_segment(header, &sc_raw)?
        };
        let data_packed: Vec<u8> = if header.compression == Compression::None {
            chan_raw.clone()
        } else {
            encode_deep_segment(header, &chan_raw)?
        };

        // ⑥ 组装块。
        let mut chunk: Vec<u8> = Vec::with_capacity(20 + sc_packed.len() + offs.len() + data_packed.len());
        chunk.extend_from_slice(&(dwin.y_min + y as i32).to_le_bytes());
        chunk.extend_from_slice(&20u64.to_le_bytes()); // 样本数表偏移（相对块首）
        chunk.extend_from_slice(&(sc_packed.len() as u64).to_le_bytes());
        chunk.extend_from_slice(&sc_packed);
        chunk.extend_from_slice(&offs);
        chunk.extend_from_slice(&data_packed);
        chunks.push(chunk);
    }
    Ok(assemble(head, &chunks, true))
}

/// deep 段（样本数表 / 通道数据）的字节级压缩。
///
/// `None` 由调用方直接返回原样；此处处理 Rle / Zips / Zip，统一走 [`encode_block`]
/// 的字节压缩分支（Rle→[`rle_compress`]，Zip/Zips→[`deflate_zlib`]）。PIZ/PXR24/B44
/// 已被 [`deep_supports`] 挡在上游。
fn encode_deep_segment(header: &ExrHeader, raw: &[u8]) -> ExrResult<Vec<u8>> {
    // 段压缩是**纯字节压缩**（与通道布局无关），故不走 `encode_block`（后者需
    // 面向 flat 重排的 `ctx`）。PIZ/PXR24/B44 已由 `deep_supports` 挡在上游。
    match header.compression {
        Compression::None => Ok(raw.to_vec()),
        Compression::Rle => Ok(rle_compress(raw)),
        Compression::Zip | Compression::Zips => Ok(deflate_zlib(raw)),
        other => Err(comp_err(
            "exr_write::encode_deep_segment",
            format!("deep 段不支持压缩类型 {}", other.code()),
            "deep 只支持 None / Rle / Zips / Zip",
        )),
    }
}

// ===========================================================================
//  六、便捷构造
// ===========================================================================

/// 便捷构造：给定尺寸与通道表，生成可写的 **flat** [`ExrHeader`] 骨架。
///
/// `channels` 顺序即写出顺序（不做任何排序）。
pub fn header_for(
    width: u32,
    height: u32,
    channels: &[(&str, PixelType)],
    compression: Compression,
) -> ExrHeader {
    ExrHeader {
        version: 2,
        tiled: false,
        deep: false,
        multipart: false,
        compression,
        data_window: DataWindow::new(width, height),
        display_window: DataWindow::new(width, height),
        // 【W13 修正】块序 = y 升序（`write_scanline_image` 的 `b` 递增），
        // 故 `lineOrder` 必须标 `INCREASING_Y`，否则官方按声明的降序
        // 反向遍历偏移表会读出「Early end of file」。详见
        // [`crate::exr::LINE_ORDER_INCREASING`] 的完整根因。
        line_order: crate::exr::LINE_ORDER_INCREASING,
        channels: channels
            .iter()
            .map(|(n, pt)| ExrChannel {
                name: (*n).to_string(),
                pixel_type: *pt,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect(),
        tile: None,
        chunk_offsets: Vec::new(),
        little_endian: true,
    }
}

/// 便捷构造：给定尺寸与通道表，生成可写的 **deep** [`ExrHeader`] 骨架。
pub fn header_for_deep(
    width: u32,
    height: u32,
    channels: &[(&str, PixelType)],
    compression: Compression,
) -> ExrHeader {
    let mut h = header_for(width, height, channels, compression);
    h.deep = true;
    h
}

/// 便捷构造：给定尺寸、通道表与 **tile 尺寸**，生成可写的 **flat tiled** 头部骨架。
///
/// `tile_w` / `tile_h` 是单个 tile 的**标称**尺寸（边缘 tile 实际按裁剪尺寸写）。
/// 官方对 tile 尺寸**无「32 的倍数」硬约束**（实测 `1×1`/`3×5`/`17×32` 均可），
/// 本函数只要求 ≥ 1（由 [`check_tiled_writable`] 校验）。
pub fn header_for_tiled(
    width: u32,
    height: u32,
    channels: &[(&str, PixelType)],
    compression: Compression,
    tile_w: u32,
    tile_h: u32,
) -> ExrHeader {
    let mut h = header_for(width, height, channels, compression);
    h.tiled = true;
    h.tile = Some(TileDesc {
        x_size: tile_w,
        y_size: tile_h,
        // ONE_LEVEL：本写入器只实装单层（MIPMAP/RIPMAP 被显式拒绝）。
        level_mode: 0,
    });
    h
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exr::{read_image, DeepSample};
    use crate::f16_to_f32;

    /// 造一个 `w × h` 的平面（`f(i) = (i as f32) * 0.5 - 3.0`，跨零便于抓符号错误）。
    fn plane(name: &str, pt: PixelType, w: u32, h: u32) -> ChannelPlane {
        ChannelPlane {
            name: name.to_string(),
            pixel_type: pt,
            width: w,
            height: h,
            values: (0..(w * h) as usize).map(|i| i as f32 * 0.5 - 3.0).collect(),
        }
    }

    /// 断言 `read_image(bytes)` 的每个通道与 `want` 逐位一致。
    fn assert_round_trip(_h: &ExrHeader, bytes: &[u8], want: &[ChannelPlane]) {
        let img = read_image(bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.planes.len(), want.len(), "通道数须一致");
        for wp in want {
            let got = img
                .plane(&wp.name)
                .unwrap_or_else(|| panic!("缺少通道 {}", wp.name));
            assert_eq!(got.width, wp.width);
            assert_eq!(got.height, wp.height);
            for (i, (&g, &w)) in got.values.iter().zip(wp.values.iter()).enumerate() {
                match wp.pixel_type {
                    PixelType::Float => assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{} 像素 {i}: 读回 {g} 期望 {w}",
                        wp.name
                    ),
                    PixelType::Half => {
                        let q = f16_to_f32(f32_to_f16(w));
                        assert_eq!(
                            f16_to_f32(f32_to_f16(g)).to_bits(),
                            q.to_bits(),
                            "{} 像素 {i}: 读回 {g} 期望(half) {q}",
                            wp.name
                        );
                    }
                    PixelType::Uint => assert_eq!(
                        g as u64,
                        (w.max(0.0) as u32) as u64,
                        "{} 像素 {i}",
                        wp.name
                    ),
                }
            }
        }
    }

    /// 允许误差的往返断言（B44 有损 / PXR24-float 有损）。
    fn assert_round_trip_bounded(bytes: &[u8], want: &[ChannelPlane], tol: f32) {
        let img = read_image(bytes).unwrap_or_else(|e| panic!("{e}"));
        for wp in want {
            let got = img
                .plane(&wp.name)
                .unwrap_or_else(|| panic!("缺少通道 {}", wp.name));
            for (i, (&g, &w)) in got.values.iter().zip(wp.values.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= tol,
                    "{} 像素 {i}: 读回 {g} 期望 {w}（误差 {} > {tol}）",
                    wp.name,
                    (g - w).abs()
                );
            }
        }
    }

    /// DWA（DWAA/DWAB）往返的**幅值相关**误差上界。
    ///
    /// 【为什么不能用固定绝对容差】DWA 的 AC 量化发生在**非线性（对数）域**
    /// （与官方 OpenEXR 一致：`from_linear` 先把线性值压到 `1+ln|y|/2.2` 域，
    /// DCT + 量化都在该域进行，解码端 `to_linear` 再还原）。故量化步长
    /// `QUANT_STEP = 1/64` 在**线性域被逐像素放大**，放大因子为该段导数：
    ///
    /// ```text
    /// |y| ≤ 1 : to_linear = sign·|x|^2.2      → 增益 2.2·|y|^1.2 ≤ 2.2
    /// |y| > 1 : to_linear = sign·e^(2.2(x-1))  → 增益 2.2·|y|（线性增长）
    /// ```
    ///
    /// 实测（`dwa_lossy_error_is_bounded_relative_not_absolute`钉死）：
    /// 误差与**幅值成正比**，相对误差稳定在 ~2.5%–3.9%，且**无系统性偏置**
    /// （各图案 signed bias 均值在噪声量级）。故用固定绝对阈值 0.05 判定
    /// 大幅值像素本身就是**错误的误差模型**——实测 `Y=200` 处绝对误差达5.1，
    /// 相对误差仅 2.6%。本函数按幅值给上界：
    ///
    /// ```text
    /// tol(y) = DWA_TOL_REL · |y| + DWA_TOL_ABS
    /// ```
    ///
    /// - `DWA_TOL_REL = 0.05`：实测最坏相对误差 3.87%（伪随机大动态范围图案）
    ///   上取整到 5% 留~29% 余量。
    /// - `DWA_TOL_ABS = 0.05`：`|y| ≤ 1` 段增益封顶在 2.2，加half 舍入后
    ///   实测最坏 0.0137，取 0.05 覆盖 from/toLinear 的 half 中转误差。
    const DWA_TOL_REL: f32 = 0.05;
    /// 见 [`DWA_TOL_REL`]：`|y| ≤ 1` 段与 half 舍入的绝对误差地板。
    const DWA_TOL_ABS: f32 = 0.05;

    /// 按DWA 的幅值相关误差模型断言往返。
    fn assert_dwa_round_trip(bytes: &[u8], want: &[ChannelPlane]) {
        let img = read_image(bytes).unwrap_or_else(|e| panic!("{e}"));
        for wp in want {
            let got = img
                .plane(&wp.name)
                .unwrap_or_else(|| panic!("缺少通道 {}", wp.name));
            for (i, (&g, &w)) in got.values.iter().zip(wp.values.iter()).enumerate() {
                let tol = DWA_TOL_REL * w.abs() + DWA_TOL_ABS;
                assert!(
                    (g - w).abs() <= tol,
                    "{} 像素 {i}: 读回 {g} 期望 {w}（误差 {} > {:.5} = 0.05·|{w}|+0.05）",
                    wp.name,
                    (g - w).abs(),
                    tol
                );
            }
        }
    }

    // -- 1. 四种原有压缩 round-trip ---------------------------------------

    #[test]
    fn round_trip_none_rle_zip_and_zips() {
        let (w, h) = (5u32, 7u32);
        for comp in [
            Compression::None,
            Compression::Rle,
            Compression::Zip,
            Compression::Zips,
        ] {
            let want = vec![
                plane("Y", PixelType::Float, w, h),
                plane("A", PixelType::Half, w, h),
            ];
            let hdr = header_for(
                w,
                h,
                &[("Y", PixelType::Float), ("A", PixelType::Half)],
                comp,
            );
            let bytes =
                write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e} · {}", comp.code()));
            assert_round_trip(&hdr, &bytes, &want);
        }
    }

    #[test]
    fn zip_blocks_use_sixteen_lines_and_multi_block_round_trips() {
        let (w, h) = (3u32, 35u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::Zip);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.compression.lines_per_block(), 16);
        assert_eq!(rh.block_count(), 3);
        assert_round_trip(&hdr, &bytes, &want);
    }

    // -- 2. 新增：PIZ / PXR24 / B44 / B44A round-trip ----------------------

    #[test]
    fn piz_write_round_trips_bit_exact_half_float_uint() {
        // PIZ 无损：half/float/uint 混合通道逐位一致。
        let (w, h) = (6u32, 5u32);
        let chans = [
            ("R", PixelType::Half),
            ("G", PixelType::Float),
            ("id", PixelType::Uint),
        ];
        let want: Vec<ChannelPlane> = chans
            .iter()
            .map(|(n, pt)| {
                let mut p = plane(n, *pt, w, h);
                if *pt == PixelType::Uint {
                    p.values = (0..(w * h) as usize).map(|i| i as f32).collect();
                }
                p
            })
            .collect();
        let hdr = header_for(w, h, &chans, Compression::Piz);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn pxr24_write_multi_block_partial_last_round_trips() {
        // PXR24 = 16 行/块：h=35 → 3 块（16+16+3，末块不满）。
        let (w, h) = (4u32, 35u32);
        let want = vec![plane("A", PixelType::Half, w, h)];
        let hdr = header_for(w, h, &[("A", PixelType::Half)], Compression::Pxr24);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.block_count(), 3, "PXR24 16 行/块 → 3 块");
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn b44_write_multi_block_partial_last_round_trips() {
        // B44 = 32 行/块：h=70 → 3 块（32+32+6，末块不满）。
        let (w, h) = (4u32, 70u32);
        let mut p = plane("A", PixelType::Half, w, h);
        p.values = (0..(w * h) as usize)
            .map(|i| ((i % 4) as f32) * 0.1 + 0.2)
            .collect();
        let want = vec![p];
        let hdr = header_for(w, h, &[("A", PixelType::Half)], Compression::B44);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.block_count(), 3, "B44 32 行/块 → 3 块");
        assert_round_trip_bounded(&bytes, &want, 0.05);
    }

    #[test]
    fn piz_write_multi_block_round_trips() {
        // PIZ = 32 行/块：h=70 → 3 块（32+32+6）。
        let (w, h) = (4u32, 70u32);
        let want = vec![plane("Y", PixelType::Half, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Half)], Compression::Piz);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.block_count(), 3);
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn pxr24_write_half_and_uint_round_trip_bit_exact() {
        // PXR24 对 half/uint 无损。
        let (w, h) = (5u32, 4u32);
        let chans = [("A", PixelType::Half), ("id", PixelType::Uint)];
        let want: Vec<ChannelPlane> = chans
            .iter()
            .map(|(n, pt)| {
                let mut p = plane(n, *pt, w, h);
                if *pt == PixelType::Uint {
                    p.values = (0..(w * h) as usize).map(|i| (i * 3) as f32).collect();
                }
                p
            })
            .collect();
        let hdr = header_for(w, h, &chans, Compression::Pxr24);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn pxr24_write_float_is_lossy_but_bounded() {
        // PXR24 对 float 有损（32→24bit），误差上界取相对 2^-16 量级的安全界。
        let (w, h) = (8u32, 3u32);
        let want = vec![plane("Z", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Z", PixelType::Float)], Compression::Pxr24);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip_bounded(&bytes, &want, 0.01);
    }

    #[test]
    fn b44_write_half_round_trips_within_error_bound() {
        // B44 对 half 有损（4×4 块 packing）。误差上界由 B44 的**块内量程**决定：
        // 每个 4×4 块取块内最大值并量化到 10 位尾数，块量程 ≤ 1 时绝对误差
        // ≲ 2^-10 · 量程 + 半值量化步长。这里取块量程 ≤ 1、tol = 0.05 的安全界
        // （实测最大相对误差远小于此）。
        let (w, h) = (8u32, 8u32);
        let mut p = plane("A", PixelType::Half, w, h);
        // 块内量程控制在 [0,1) 的平滑正值。
        p.values = (0..(w * h) as usize)
            .map(|i| ((i % w as usize) as f32 / w as f32) * 0.5 + 0.25)
            .collect();
        let want = vec![p];
        let hdr = header_for(w, h, &[("A", PixelType::Half)], Compression::B44);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip_bounded(&bytes, &want, 0.05);
    }

    #[test]
    fn b44_write_copies_float_channel_verbatim() {
        // B44 对 FLOAT/UINT 通道**原样复制**（无损）。
        let (w, h) = (4u32, 4u32);
        let want = vec![plane("Z", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Z", PixelType::Float)], Compression::B44);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn b44a_write_round_trips_within_error_bound() {
        // B44A 与 B44 编码路径相同（额外允许 3 字节平场），误差界同上。
        let (w, h) = (8u32, 4u32);
        let mut p = plane("A", PixelType::Half, w, h);
        p.values = (0..(w * h) as usize)
            .map(|i| ((i % w as usize) as f32 / w as f32) * 0.5 + 0.25)
            .collect();
        let want = vec![p];
        let hdr = header_for(w, h, &[("A", PixelType::Half)], Compression::B44A);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_round_trip_bounded(&bytes, &want, 0.05);
    }

    #[test]
    fn pimg_write_two_times_is_byte_identical_all_supported() {
        // 每种受支持压缩：同输入连写两遍字节 `==`。
        let (w, h) = (7u32, 9u32);
        for comp in [
            Compression::None,
            Compression::Rle,
            Compression::Zip,
            Compression::Zips,
            Compression::Piz,
            Compression::Pxr24,
            Compression::B44,
            Compression::B44A,
        ] {
            let want = vec![
                plane("R", PixelType::Half, w, h),
                plane("Z", PixelType::Float, w, h),
            ];
            let hdr = header_for(
                w,
                h,
                &[("R", PixelType::Half), ("Z", PixelType::Float)],
                comp,
            );
            let a = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            let b = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{} 两次写出须字节相同", comp.code());
        }
    }

    // -- 3. deflate 升级证据 ---------------------------------------------

    #[test]
    fn deflate_fixed_round_trips_through_prism_bytes_inflate() {
        // deflate fixed：必须能被 `prism_bytes::inflate` 解回，且逐位一致。
        for n in [0usize, 1, 2, 3, 17, 255, 256, 4096, 70000] {
            let payload: Vec<u8> = (0..n).map(|i| (i % 7) as u8).collect();
            let raw = deflate_raw_best(&payload);
            let back = prism_bytes::inflate::inflate_raw(&raw)
                .unwrap_or_else(|e| panic!("{e} · n={n}"));
            assert_eq!(back, payload, "deflate 往返 n={n}");
        }
    }

    #[test]
    fn deflate_zlib_fixed_round_trips_and_is_legal() {
        // zlib 包装：RFC 1950 头自洽 + 解回一致（含真实压缩过的输入）。
        for n in [0usize, 1, 100, 4096, 70000] {
            let payload: Vec<u8> = (0..n).map(|i| (i % 13) as u8).collect();
            let z = deflate_zlib(&payload);
            assert_eq!(z[0] & 0x0f, 8, "CM 须为 8，n={n}");
            assert_eq!(((z[0] as u16) << 8 | z[1] as u16) % 31, 0, "FCHECK 自洽");
            assert_eq!(z[1] & 0x20, 0, "不得声明 FDICT");
            let back = crate::exr::zlib_decompress(&z).unwrap_or_else(|e| panic!("{e} · n={n}"));
            assert_eq!(back, payload, "zlib 往返 n={n}");
        }
    }

    #[test]
    fn deflate_fixed_is_deterministic() {
        // 确定性：同输入写两遍字节 `==`（无哈希随机化）。
        let payload: Vec<u8> = (0..5000u32).map(|i| ((i / 3) % 11) as u8).collect();
        let a = deflate_fixed(&payload);
        let b = deflate_fixed(&payload);
        assert_eq!(a, b, "fixed Huffman 输出须确定");
        assert_eq!(deflate_zlib(&payload), deflate_zlib(&payload));
    }

    #[test]
    fn deflate_fixed_beats_stored_on_compressible_input() {
        // 压缩率证据的**断言版**：可压缩输入上 fixed 必须显著小于 stored。
        let payload: Vec<u8> = (0..4096u32).map(|i| (i / 64) as u8).collect();
        let fixed = deflate_fixed(&payload);
        let stored = deflate_stored(&payload);
        assert!(
            fixed.len() < stored.len(),
            "fixed({}) 应小于 stored({})",
            fixed.len(),
            stored.len()
        );
    }

    #[test]
    fn deflate_compression_ratio_table_for_gradient_input() {
        // 典型输入（渐变图）的 `原始 / stored / fixed` 体积对比表。
        // 256×256×2 通道 half → 原始 = 256*256*2*2 = 262144 字节。
        let (w, h) = (256usize, 256usize);
        let mut raw: Vec<u8> = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = ((x as f32 / w as f32) + (y as f32 / h as f32)) * 0.5;
                let half = f32_to_f16(v).to_le_bytes();
                raw.extend_from_slice(&half); // R
                raw.extend_from_slice(&half); // G
            }
        }
        let stored = deflate_stored(&raw);
        let fixed = deflate_fixed(&raw);
        let best = deflate_raw_best(&raw);
        assert_eq!(raw.len(), 262_144);
        assert!(
            fixed.len() < stored.len(),
            "渐变图 fixed 必须优于 stored：fixed={} stored={}",
            fixed.len(),
            stored.len()
        );
        assert_eq!(best.len(), fixed.len().min(stored.len()));
        // 打印对比表（`cargo test -- --nocapture` 可见）。
        println!(
            "deflate 压缩率对比（渐变图 {w}×{h}×2ch half）：原始={} 字节 | stored={} 字节 | fixed={} 字节 | 最优={} 字节",
            raw.len(),
            stored.len(),
            fixed.len(),
            best.len()
        );
        // 附带三种代表性输入的对比（供报告引用；也断言最优 = 二者较小）。
        let cases: [(&str, Vec<u8>); 3] = [
            ("常量块（1MiB 同值）", vec![0xA5u8; 1 << 20]),
            ("半随机（伪噪声 256KiB）", {
                let mut v = Vec::with_capacity(1 << 18);
                let mut s: u32 = 0x1234_5678;
                for _ in 0..(1 << 18) {
                    // xorshift（定长、确定）——只用作出数据，不参与压缩路径。
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    v.push((s & 0xFF) as u8);
                }
                v
            }),
            ("短输入（8 字节）", vec![b'a', b'b', b'c', b'd', b'e', b'f', b'g', 0]),
        ];
        for (label, data) in &cases {
            let st = deflate_stored(data);
            let fx = deflate_fixed(data);
            let bst = deflate_raw_best(data);
            assert_eq!(bst.len(), fx.len().min(st.len()), "{label}");
            // 都能被 inflate 解回。
            assert_eq!(
                prism_bytes::inflate::inflate_raw(&bst).unwrap_or_else(|e| panic!("{e}")),
                *data,
                "{label}"
            );
            println!(
                "  · {label}: 原始={} | stored={} | fixed={} | 最优={}",
                data.len(),
                st.len(),
                fx.len(),
                bst.len()
            );
        }
        // 全部合法可解。
        assert_eq!(prism_bytes::inflate::inflate_raw(&best).unwrap(), raw);
    }

    #[test]
    fn deflate_stored_still_handles_empty_and_multi_block() {
        // stored 回退路径的边界：空负载与 > 65535 多块。
        let empty = deflate_stored(&[]);
        assert_eq!(
            prism_bytes::inflate::inflate_raw(&empty).unwrap_or_else(|e| panic!("{e}")),
            Vec::<u8>::new()
        );
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 256) as u8).collect();
        let back = prism_bytes::inflate::inflate_raw(&deflate_stored(&big))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, big);
    }

    #[test]
    fn adler32_matches_rfc1950_reference_values() {
        assert_eq!(adler32(b""), 0x0000_0001);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"abc"), 0x024d_0127);
        assert_eq!(adler32(b"message digest"), 0x2975_0586);
        assert_eq!(adler32(b"abcdefghijklmnopqrstuvwxyz"), 0x9086_0b20);
    }

    // -- 4. deep 写入 ----------------------------------------------------

    /// 造 deep 像素序列：第 `i` 像素有 `n_spp(i)` 个样本，值确定。
    fn deep_pixels(header: &ExrHeader) -> Vec<DeepPixel> {
        let win = header.data_window;
        let w = win.width() as usize;
        let h = win.height() as usize;
        let mut out: Vec<DeepPixel> = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // 样本数在 [0,3) 变化（含 0 样本像素，考验边界）。
                let n = (x + y) % 3;
                let mut samples: Vec<DeepSample> = Vec::with_capacity(n);
                for s in 0..n {
                    let z = 1.0 + (x * 10 + y) as f32 + s as f32 * 0.25;
                    let id = (x * 100 + y * 10 + s) as f32;
                    samples.push(DeepSample {
                        index: s as u32,
                        z,
                        values: vec![z, id],
                    });
                }
                out.push(DeepPixel {
                    x: win.x_min + x as i32,
                    y: win.y_min + y as i32,
                    samples,
                });
            }
        }
        out
    }

    /// 断言 deep 往返：sample count / 值 / Z 逐位一致。
    fn assert_deep_round_trip(header: &ExrHeader, pixels: &[DeepPixel]) {
        let bytes = write_deep_image(header, pixels).unwrap_or_else(|e| panic!("{e}"));
        let img = crate::exr::read_deep_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.pixels.len(), pixels.len(), "像素数须一致");
        for (want, got) in pixels.iter().zip(img.pixels.iter()) {
            assert_eq!(got.x, want.x);
            assert_eq!(got.y, want.y);
            assert_eq!(
                got.samples.len(),
                want.samples.len(),
                "像素 ({},{}) 样本数",
                want.x,
                want.y
            );
            for (ws, gs) in want.samples.iter().zip(got.samples.iter()) {
                assert_eq!(gs.index, ws.index, "样本序号");
                // Z 来自 depth 通道（这里 Zfloat 优先），逐位一致。
                assert_eq!(
                    gs.z.to_bits(),
                    ws.z.to_bits(),
                    "样本 Z 须逐位一致（像素 ({},{})）",
                    want.x,
                    want.y
                );
                for (wv, gv) in ws.values.iter().zip(gs.values.iter()) {
                    assert_eq!(wv.to_bits(), gv.to_bits(), "样本值须逐位一致");
                }
            }
        }
    }

    #[test]
    fn deep_write_none_round_trips_bit_exact() {
        let (w, h) = (4u32, 3u32);
        let hdr = header_for_deep(
            w,
            h,
            &[("Zfloat", PixelType::Float), ("id", PixelType::Uint)],
            Compression::None,
        );
        let pixels = deep_pixels(&hdr);
        // 前置：样例里确有 0 样本与多样本像素。
        assert!(pixels.iter().any(|p| p.samples.is_empty()));
        assert!(pixels.iter().any(|p| p.samples.len() >= 2));
        assert_deep_round_trip(&hdr, &pixels);
    }

    #[test]
    fn deep_write_zip_round_trips_bit_exact() {
        let (w, h) = (5u32, 4u32);
        let hdr = header_for_deep(
            w,
            h,
            &[("Zfloat", PixelType::Float), ("id", PixelType::Uint)],
            Compression::Zip,
        );
        let pixels = deep_pixels(&hdr);
        assert_deep_round_trip(&hdr, &pixels);
    }

    #[test]
    fn deep_write_zips_round_trips_bit_exact() {
        let (w, h) = (3u32, 3u32);
        let hdr = header_for_deep(
            w,
            h,
            &[("Zfloat", PixelType::Float)],
            Compression::Zips,
        );
        let pixels = deep_pixels(&hdr);
        assert_deep_round_trip(&hdr, &pixels);
    }

    #[test]
    fn deep_write_rle_round_trips_bit_exact() {
        let (w, h) = (3u32, 2u32);
        let hdr = header_for_deep(w, h, &[("Zfloat", PixelType::Float)], Compression::Rle);
        let pixels = deep_pixels(&hdr);
        assert_deep_round_trip(&hdr, &pixels);
    }

    #[test]
    fn deep_version_flag_is_set_and_header_reads_back() {
        // deep 标志位正确置位，且 read_header 能识别。
        let hdr = header_for_deep(2, 2, &[("Zfloat", PixelType::Float)], Compression::None);
        let pixels = deep_pixels(&hdr);
        let bytes = write_deep_image(&hdr, &pixels).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert!(rh.deep, "版本字段须置 DEEP_DATA 标志");
        assert!(!rh.tiled && !rh.multipart);
        assert_eq!(rh.block_count(), 2, "deep 恒为每行一块");
    }

    #[test]
    fn deep_write_two_times_is_byte_identical() {
        let (w, h) = (4u32, 4u32);
        for comp in [Compression::None, Compression::Zip, Compression::Zips, Compression::Rle] {
            let hdr = header_for_deep(w, h, &[("Zfloat", PixelType::Float)], comp);
            let pixels = deep_pixels(&hdr);
            let a = write_deep_image(&hdr, &pixels).unwrap_or_else(|e| panic!("{e}"));
            let b = write_deep_image(&hdr, &pixels).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "deep {} 两次写出须字节相同", comp.code());
        }
    }

    // -- 5. 结构化拒绝（≥ 6 条）------------------------------------------

    #[test]
    fn writer_supports_covers_ten_types() {
        // 写入侧能力表：十种全部实现（DWA 于 W11 路B 补完）。
        for c in [
            Compression::None,
            Compression::Rle,
            Compression::Zip,
            Compression::Zips,
            Compression::Piz,
            Compression::Pxr24,
            Compression::B44,
            Compression::B44A,
            Compression::Dwaa,
            Compression::Dwab,
        ] {
            assert!(writer_supports(c), "{} 应受支持", c.code());
        }
    }

    // -- 7. DWAA / DWAB 写入（W11 路B）------------------------------------

    /// DWA 有损往返：读回值与原始值在 `tol` 内（有损阈值见 `exr_dwa` 模块注释）。
    #[test]
    fn dwaa_write_round_trips_within_lossy_bound_half() {
        let (w, h) = (9u32, 5u32);
        // R/G/B 三通道 → 触发 CSC 组路径；值域控制在 [0,1] 平滑渐变。
        let names = ["R", "G", "B"];
        let want: Vec<ChannelPlane> = names
            .iter()
            .enumerate()
            .map(|(k, n)| {
                let mut p = plane(n, PixelType::Half, w, h);
                p.values = (0..(w * h) as usize)
                    .map(|i| (((i + k * 7) % 16) as f32) / 16.0)
                    .collect();
                p
            })
            .collect();
        let hdr = header_for(w, h, &[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)], Compression::Dwaa);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.compression, Compression::Dwaa, "header.Compression 须为 8");
        assert_dwa_round_trip(&bytes, &want);
    }

    #[test]
    fn dwab_write_round_trips_then_reads_header_compression_nine() {
        let (w, h) = (17u32, 300u32); // > 256 行 → 2 块（256 + 44）
        let mut p = plane("Y", PixelType::Half, w, h);
        p.values = (0..(w * h) as usize)
            .map(|i| (((i % 24) as f32) / 24.0) - 0.5)
            .collect();
        let want = vec![p];
        let hdr = header_for(w, h, &[("Y", PixelType::Half)], Compression::Dwab);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.compression, Compression::Dwab);
        assert_eq!(rh.compression.code(), 9, "DWAB 属性字节 = 9");
        assert_eq!(rh.block_count(), 2, "DWAB 256 行/块 → 300 行 = 2 块");
        assert_dwa_round_trip(&bytes, &want);
    }

    #[test]
    fn dwaa_dwab_two_writes_are_byte_identical() {
        let (w, h) = (13u32, 40u32);
        let want = vec![
            plane("R", PixelType::Half, w, h),
            plane("G", PixelType::Half, w, h),
            plane("B", PixelType::Half, w, h),
        ];
        for comp in [Compression::Dwaa, Compression::Dwab] {
            let hdr = header_for(
                w,
                h,
                &[("R", PixelType::Half), ("G", PixelType::Half), ("B", PixelType::Half)],
                comp,
            );
            let a = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            let b = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{} 两次写出须字节相同", comp.code());
        }
    }

    #[test]
    fn dwaa_block_boundaries_exact_and_partial_multiples() {
        // DWAA = 32 行/块：32（整倍）与 97（3 块：32+32+33... 实为 32/32/32/1）。
        for h in [32u32, 97u32] {
            let (w, _) = (5u32, h);
            let want = vec![plane("Y", PixelType::Half, w, h)];
            let hdr = header_for(w, h, &[("Y", PixelType::Half)], Compression::Dwaa);
            let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(rh.block_count(), (h as usize).div_ceil(32), "h={h}");
            assert_dwa_round_trip(&bytes, &want);
        }
    }

    #[test]
    fn dwab_block_boundary_exact_multiple_of_256() {
        let (w, h) = (4u32, 256u32);
        let want = vec![plane("Y", PixelType::Half, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Half)], Compression::Dwab);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(rh.block_count(), 1, "256 行恰为 1 块");
        assert_dwa_round_trip(&bytes, &want);
    }

    #[test]
    fn dwaa_single_channel_and_multi_channel_both_round_trip() {
        // 单通道（Y，非 CSC）与混合通道（R/G/B + A 的 RLE + Z 的 Unknown）。
        let (w, h) = (8u32, 8u32);
        let single = vec![plane("Y", PixelType::Half, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Half)], Compression::Dwaa);
        let bytes = write_scanline_image(&hdr, &single).unwrap_or_else(|e| panic!("{e}"));
        assert_dwa_round_trip(&bytes, &single);

        // 混合：A（half）命中 legacy/default RLE 规则；Z（float）不命中 → Unknown。
        let chans = [
            ("R", PixelType::Half),
            ("G", PixelType::Half),
            ("B", PixelType::Half),
            ("A", PixelType::Half),
            ("Z", PixelType::Float),
        ];
        let mut want: Vec<ChannelPlane> = chans
            .iter()
            .map(|(n, pt)| {
                let mut p = plane(n, *pt, w, h);
                p.values = (0..(w * h) as usize).map(|i| ((i % 8) as f32) / 8.0).collect();
                p
            })
            .collect();
        // Z 通道（Unknown）须逐位无损 —— 用 half 可精确表示的值，隔离 DCT 有损。
        if let Some(z) = want.last_mut() {
            z.values = (0..(w * h) as usize).map(|i| ((i % 8) as f32) / 8.0).collect();
        }
        let hdr = header_for(w, h, &chans, Compression::Dwaa);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_dwa_round_trip(&bytes, &want);
    }

    #[test]
    fn dwaa_float_channel_is_lossy_but_bounded_by_half_precision() {
        // float 通道经 DWA 后先落到 half 精度（解码器语义），故上界取 ~1e-3 量级。
        let (w, h) = (8u32, 8u32);
        let mut p = plane("Y", PixelType::Float, w, h);
        p.values = (0..(w * h) as usize)
            .map(|i| (((i % 8) as f32) / 8.0) * 0.5 + 0.25)
            .collect();
        let want = vec![p];
        let hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::Dwaa);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        assert_dwa_round_trip(&bytes, &want);
    }

    #[test]
    fn flat_write_rejects_deep_flagged_header() {
        let (w, h) = (2u32, 2u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let mut hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::None);
        hdr.deep = true;
        let err = write_scanline_image(&hdr, &want).unwrap_err();
        assert_eq!(err.code, "EXR1001", "{err}");
        assert!(format!("{err}").contains("write_deep_image"), "须指向 deep 入口：{err}");
    }

    #[test]
    fn tiled_and_multipart_flat_headers_are_rejected() {
        let (w, h) = (2u32, 2u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let base = header_for(w, h, &[("Y", PixelType::Float)], Compression::None);
        for (label, keyword) in [("tiled", "分块"), ("multipart", "多部件")] {
            let mut hdr = base.clone();
            if label == "tiled" {
                hdr.tiled = true;
            } else {
                hdr.multipart = true;
            }
            let err = write_scanline_image(&hdr, &want).unwrap_err();
            assert_eq!(err.code, "EXR1001", "{label}: {err}");
            assert!(format!("{err}").contains(keyword), "{label} 须点名：{err}");
        }
    }

    #[test]
    fn deep_write_rejects_piz_pxr24_b44_and_dwa() {
        // deep 不支持面向 flat 重排的压缩与 DWA → 结构化拒绝。
        let (w, h) = (2u32, 2u32);
        for comp in [
            Compression::Piz,
            Compression::Pxr24,
            Compression::B44,
            Compression::B44A,
            Compression::Dwaa,
            Compression::Dwab,
        ] {
            let hdr = header_for_deep(w, h, &[("Zfloat", PixelType::Float)], comp);
            let pixels = deep_pixels(&hdr);
            let err = write_deep_image(&hdr, &pixels).unwrap_err();
            assert!(
                err.code == "EXR2001" || err.code == "EXR1001",
                "{}: {err}",
                comp.code()
            );
            assert!(format!("{err}").contains("deep"), "{} 须点名 deep：{err}", comp.code());
        }
    }

    #[test]
    fn deep_write_rejects_wrong_pixel_count() {
        let hdr = header_for_deep(2, 2, &[("Zfloat", PixelType::Float)], Compression::None);
        let mut pixels = deep_pixels(&hdr);
        pixels.pop(); // 少一个像素
        let err = write_deep_image(&hdr, &pixels).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");
    }

    #[test]
    fn deep_write_requires_deep_flag() {
        let (w, h) = (2u32, 2u32);
        let hdr = header_for(w, h, &[("Zfloat", PixelType::Float)], Compression::None);
        let pixels = deep_pixels(&hdr);
        let err = write_deep_image(&hdr, &pixels).unwrap_err();
        assert_eq!(err.code, "EXR1001", "{err}");
        assert!(format!("{err}").contains("deep"), "{err}");
    }

    #[test]
    fn deep_write_rejects_multipart() {
        let (w, h) = (2u32, 2u32);
        let mut hdr = header_for_deep(w, h, &[("Zfloat", PixelType::Float)], Compression::None);
        hdr.multipart = true;
        let pixels = deep_pixels(&hdr);
        let err = write_deep_image(&hdr, &pixels).unwrap_err();
        assert_eq!(err.code, "EXR1001", "{err}");
        assert!(format!("{err}").contains("多部件"), "{err}");
    }

    // -- 6. 既有行为回归 --------------------------------------------------

    #[test]
    fn multi_channel_half_float_and_uint_round_trip() {
        let (w, h) = (4u32, 4u32);
        let chans = [
            ("R", PixelType::Half),
            ("G", PixelType::Float),
            ("B", PixelType::Half),
            ("id", PixelType::Uint),
        ];
        let want: Vec<ChannelPlane> = chans
            .iter()
            .map(|(n, pt)| {
                let mut p = plane(n, *pt, w, h);
                if *pt == PixelType::Uint {
                    p.values = (0..(w * h) as usize).map(|i| i as f32).collect();
                }
                p
            })
            .collect();
        let hdr = header_for(w, h, &chans, Compression::None);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let got_names: Vec<&str> = img.planes.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(got_names, vec!["R", "G", "B", "id"], "通道顺序须与 header 一致");
        assert_round_trip(&hdr, &bytes, &want);
    }

    #[test]
    fn planes_may_be_supplied_in_any_order() {
        let (w, h) = (3u32, 3u32);
        let a = plane("A", PixelType::Float, w, h);
        let b = plane("B", PixelType::Float, w, h);
        let hdr = header_for(w, h, &[("A", PixelType::Float), ("B", PixelType::Float)], Compression::None);
        let forward = write_scanline_image(&hdr, &[a.clone(), b.clone()]).unwrap_or_else(|e| panic!("{e}"));
        let reversed = write_scanline_image(&hdr, &[b, a]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(forward, reversed, "平面传入顺序不得影响输出字节");
    }

    /// W13互通对拍：`lineOrder` 必须与**实际块序**一致（本仓按y 升序写块）。
    ///
    /// 回归测试：原先 `header_for` 把 `lineOrder` 标成 `DECREASING_Y(1)`，
    /// 而 `write_scanline_image` 实际按 **y 升序**写块。官方读取器按声明的
    /// 降序反向遍历偏移表 → 任何压缩（含 `None`）都报
    /// 「Early end of file: read 0 out of 4 requested bytes」。
    ///
    /// 断言「`lineOrder == INCREASING_Y`」+「块首行号严格升序」两条同时成立，
    /// 这样即便日后有人改块序而忘了改属性，本测试立刻红。
    #[test]
    fn line_order_matches_the_block_order_actually_written() {
        let (w, h) = (4u32, 96u32); // 96 行 → DWAA(32 行/块) 共 3 块
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::Dwaa);
        assert_eq!(
            hdr.line_order,
            crate::exr::LINE_ORDER_INCREASING,
            "块按 y 升序写入，lineOrder 必须标 INCREASING_Y"
        );
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));

        // 复核块首行号确实升序（避免「属性对了但块序反了」的另一半风险）。
        // 用本仓**自己的头解析器**定位偏移表，避免在测试里重写一遍属性 walker。
        let parsed = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("解析头失败: {e:?}"));
        assert_eq!(
            parsed.line_order,
            crate::exr::LINE_ORDER_INCREASING,
            "写出的字节里 lineOrder 仍不是 INCREASING_Y"
        );
        let blocks = (h as usize).div_ceil(32);
        let mut chunk_first_lines = Vec::with_capacity(blocks);
        for k in 0..blocks {
            let off = parsed.chunk_offsets[k] as usize;
            let y = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            chunk_first_lines.push(y);
        }
        assert!(
            chunk_first_lines.windows(2).all(|w| w[0] < w[1]),
            "块首行号必须严格升序，实测 {chunk_first_lines:?}"
        );
    }

    #[test]
    fn header_attribute_order_is_fixed() {
        let (w, h) = (2u32, 2u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::None);
        let bytes = write_scanline_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let order = [
            "channels",
            "compression",
            "dataWindow",
            "displayWindow",
            "lineOrder",
            "pixelAspectRatio",
            "screenWindowCenter",
            "screenWindowWidth",
        ];
        let mut at = 8usize;
        for name in order {
            let pat = {
                let mut v = name.as_bytes().to_vec();
                v.push(0);
                v
            };
            let found = bytes[at..]
                .windows(pat.len())
                .position(|w| w == pat.as_slice())
                .map(|p| p + at)
                .unwrap_or_else(|| panic!("未找到属性 {name}"));
            assert!(found >= at, "属性 {name} 的位置倒退，顺序不是固定的");
            at = found + 1;
        }
    }

    #[test]
    fn write_image_selects_compression_without_mutating_caller_header() {
        let (w, h) = (3u32, 4u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for(w, h, &[("Y", PixelType::Float)], Compression::None);
        for comp in [Compression::Rle, Compression::Zip, Compression::Piz, Compression::None] {
            let bytes = write_image(&hdr, &want, comp).unwrap_or_else(|e| panic!("{e}"));
            let rh = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(rh.compression, comp, "写出的文件须带所选压缩");
        }
        assert_eq!(hdr.compression, Compression::None, "write_image 不得改动调用方 header");
    }

    #[test]
    fn missing_misaligned_and_wrong_sized_planes_are_rejected() {
        let (w, h) = (3u32, 3u32);
        let hdr = header_for(
            w,
            h,
            &[("A", PixelType::Float), ("B", PixelType::Float)],
            Compression::None,
        );
        let a = plane("A", PixelType::Float, w, h);
        let b = plane("B", PixelType::Float, w, h);

        let err = write_scanline_image(&hdr, &[a.clone()]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");

        let mut wrong = b.clone();
        wrong.width = w + 1;
        wrong.values.push(0.0);
        let err = write_scanline_image(&hdr, &[a.clone(), wrong]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");

        let mut short = b.clone();
        short.values.pop();
        let err = write_scanline_image(&hdr, &[a, short]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");
    }

    #[test]
    fn empty_channel_table_and_zero_sized_window_are_rejected() {
        let hdr = header_for(2, 2, &[], Compression::None);
        let err = write_scanline_image(&hdr, &[]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");

        let p = plane("Y", PixelType::Float, 1, 1);
        let mut hdr = header_for(1, 1, &[("Y", PixelType::Float)], Compression::None);
        hdr.data_window = DataWindow::new(0, 0);
        let err = write_scanline_image(&hdr, &[p]).unwrap_err();
        assert_eq!(err.code, "EXR1001", "{err}");
    }

    #[test]
    fn subsampled_channels_are_rejected() {
        let mut hdr = header_for(2, 2, &[("Y", PixelType::Float)], Compression::None);
        hdr.channels[0].x_sampling = 2;
        let err = write_scanline_image(&hdr, &[plane("Y", PixelType::Float, 2, 2)]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");
        assert!(format!("{err}").contains("1×1"), "须说明只支持 1×1：{err}");
    }

    // =======================================================================
    //  8. flat 分块（tiled）写入 —— W14 路2
    // =======================================================================

    /// tiled 往返断言：经 [`crate::exr::read_all_tiles`] 读回，逐位/容差内一致。
    fn assert_tiled_round_trip(bytes: &[u8], want: &[ChannelPlane], tol: Option<f32>) {
        let header = crate::exr::read_header(bytes).unwrap_or_else(|e| panic!("{e}"));
        assert!(header.tiled, "版本字段须置 TILED 标志");
        assert!(header.tile.is_some(), "须带 `tiles` 属性");
        let img = crate::exr::read_all_tiles(bytes, &header).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.planes.len(), want.len(), "通道数须一致");
        for wp in want {
            let got = img
                .plane(&wp.name)
                .unwrap_or_else(|| panic!("缺少通道 {}", wp.name));
            assert_eq!((got.width, got.height), (wp.width, wp.height));
            for (i, (&g, &v)) in got.values.iter().zip(wp.values.iter()).enumerate() {
                match tol {
                    // 无损路径：half 先量化再比位模式（避免 f16 中转误差混入）。
                    None => {
                        let expect = if wp.pixel_type == PixelType::Half {
                            f16_to_f32(f32_to_f16(v))
                        } else {
                            v
                        };
                        assert_eq!(
                            g.to_bits(),
                            expect.to_bits(),
                            "{} 像素 {i}: 读回 {g} 期望 {expect}",
                            wp.name
                        );
                    }
                    // 有损路径（B44 / DWA）：按幅值相对容差。
                    Some(t) => {
                        let bound = t * v.abs() + t;
                        assert!(
                            (g - v).abs() <= bound,
                            "{} 像素 {i}: 读回 {g} 期望 {v}（误差 {} > {bound}）",
                            wp.name,
                            (g - v).abs()
                        );
                    }
                }
            }
        }
    }

    /// 逐 tile 读回（`read_tile`）并与整幅读回比对 —— 抓「tile 内数据排布错」。
    fn assert_tiled_tile_by_tile(bytes: &[u8]) {
        let header = crate::exr::read_header(bytes).unwrap_or_else(|e| panic!("{e}"));
        let td = header.tile.expect("已断言 tiled");
        let win = header.data_window;
        let all = crate::exr::read_all_tiles(bytes, &header).unwrap_or_else(|e| panic!("{e}"));
        let img_w = win.width() as usize;
        for ty in 0..td.tiles_y(&win) {
            for tx in 0..td.tiles_x(&win) {
                let t = crate::exr::read_tile(bytes, &header, tx, ty).unwrap_or_else(|e| panic!("{e}"));
                // 块头里的 tileX/tileY 是 **索引**；`read_tile` 回填的 data_window
                // 才是像素原点 —— 二者须一致，否则说明写侧把索引当坐标写了。
                let (tw, th) = clipped_tile_size(win, &td, tx, ty);
                assert_eq!(t.planes[0].width, tw as u32, "tile({tx},{ty}) 宽须裁剪");
                assert_eq!(t.planes[0].height, th as u32, "tile({tx},{ty}) 高须裁剪");
                let ox = (t.header.data_window.x_min - win.x_min) as usize;
                let oy = (t.header.data_window.y_min - win.y_min) as usize;
                for (ci, tp) in t.planes.iter().enumerate() {
                    let ap = &all.planes[ci];
                    for y in 0..th {
                        for x in 0..tw {
                            let got = tp.values[y * tw + x];
                            let whole = ap.values[(oy + y) * img_w + (ox + x)];
                            assert_eq!(
                                got.to_bits(),
                                whole.to_bits(),
                                "tile({tx},{ty}) 通道 {} ({x},{y}): 逐 tile 读 {got} ≠ 整幅读 {whole}",
                                tp.name
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_write_round_trips_all_eight_lossless_compressions() {
        // 无损压缩 × 四种图幅/tile 组合。后两组刻意用**非方形 tile**（8×16 / 16×8）——
        // 方形 tile（16×16）下把 `tiles` 属性的 `x_size`/`y_size` 写反**不可见**，
        // 必须有非方形 tile 才能抓住那类缺陷（变异自证 M2 已实测）。
        let comps = [
            Compression::None,
            Compression::Rle,
            Compression::Zip,
            Compression::Zips,
            Compression::Piz,
            Compression::Pxr24,
        ];
        for comp in comps {
            for (w, h, tw, th) in [
                (32u32, 32u32, 16u32, 16u32), // 整 tile 对齐
                (30u32, 21u32, 16u32, 16u32), // 两轴都非对齐边缘
                (30u32, 21u32, 8u32, 16u32),  // 非方形 tile：宽 < 高
                (30u32, 21u32, 16u32, 8u32),  // 非方形 tile：宽 > 高
            ] {
                let want = vec![
                    plane("R", PixelType::Half, w, h),
                    plane("Y", PixelType::Float, w, h),
                ];
                let hdr =
                    header_for_tiled(w, h, &[("R", Half), ("Y", Float)], comp, tw, th);
                let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
                // 非方形 tile 时 `tiles` 属性必须逐字段复现调用方的意图。
                let parsed = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
                let td = parsed.tile.expect("须有 tiles");
                assert_eq!(
                    (td.x_size, td.y_size),
                    (tw, th),
                    "压缩 {} 的 tile {tw}×{th}: `tiles` 属性须原样写出标称尺寸",
                    comp.code()
                );
                assert_tiled_round_trip(&bytes, &want, None);
                assert_tiled_tile_by_tile(&bytes);
            }
        }
    }

    /// 便捷别名（让上面的表读起来像 `Half`/`Float`）。
    use PixelType::{Float as Float, Half as Half};

    #[test]
    fn tiled_write_round_trips_b44_family_within_error_bound() {
        // B44 / B44A 对 half 有损（4×4 块 packing），对 float/uint 无损。
        for comp in [Compression::B44, Compression::B44A] {
            let (w, h) = (30u32, 21u32); // 边缘非对齐
            let mut p = plane("A", PixelType::Half, w, h);
            p.values = (0..(w * h) as usize)
                .map(|i| ((i as f32) / (w * h) as f32) * 0.5 + 0.25)
                .collect();
            let want = vec![p];
            let hdr = header_for_tiled(w, h, &[("A", Half)], comp, 16, 16);
            let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            assert_tiled_round_trip(&bytes, &want, Some(0.05));

            // float 通道：B44 原样复制 ⇒ 逐位无损。
            let fv = vec![plane("Z", PixelType::Float, w, h)];
            let hdr = header_for_tiled(w, h, &[("Z", Float)], comp, 16, 16);
            let bytes = write_tiled_image(&hdr, &fv).unwrap_or_else(|e| panic!("{e}"));
            assert_tiled_round_trip(&bytes, &fv, None);
        }
    }

    #[test]
    fn tiled_write_dwaa_dwab_round_trips_within_lossy_bound() {
        // DWAA/DWAB + tiled：DCT 块行数按**裁剪后**的 tile 高算（W13 坑#3）。
        // 32×32 tile + 48 行图 ⇒ 末行 tile 只有 16 行 ⇒ 2×4 个 8×8 块而非 4×4。
        for comp in [Compression::Dwaa, Compression::Dwab] {
            for (w, h) in [(32u32, 48u32), (64u32, 48u32)] {
                let mut p = plane("Y", PixelType::Half, w, h);
                p.values = (0..(w * h) as usize)
                    .map(|i| ((i % 24) as f32) / 24.0)
                    .collect();
                let want = vec![p];
                let hdr = header_for_tiled(w, h, &[("Y", Half)], comp, 32, 32);
                let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
                assert_tiled_round_trip(&bytes, &want, Some(0.05));
            }
        }
    }

    #[test]
    fn tiled_edge_tiles_are_written_at_clipped_size_not_full_tile_size() {
        // ⚠️ W13 坑#3 的直接钉死：64×48 图 + 32×32 tile。
        // 上排 tile 32 行 ⇒ 32×32×4B = 4096；下排 tile 只 16 行 ⇒ 2048。
        // 若按完整 tile 尺寸写，下排会是 4096 → 本测试判红。
        let (w, h) = (64u32, 48u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 32, 32);
        let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let parsed = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let td = parsed.tile.expect("须有 tiles");
        assert_eq!((td.x_size, td.y_size, td.level_mode), (32, 32, 0));
        assert_eq!(parsed.chunk_offsets.len(), 4, "2×2 tile ⇒ 偏移表 4 项");
        let mut lens: Vec<usize> = (0..4)
            .map(|b| {
                let o = parsed.chunk_offsets[b] as usize;
                i32::from_le_bytes(bytes[o + 16..o + 20].try_into().unwrap()) as usize
            })
            .collect();
        // 行主序：(0,0)(1,0) 满行 32 → 4096；(0,1)(1,1) 半行 16 → 2048。
        lens.sort_unstable();
        assert_eq!(lens, vec![2048, 2048, 4096, 4096], "边缘 tile 必须按裁剪尺寸写");
        assert_tiled_round_trip(&bytes, &want, None);
    }

    #[test]
    fn tiled_chunk_header_carries_tile_indices_and_level_zero() {
        // 块头 = tileX tileY levelX levelY size。tileX/tileY 必须是 **tile 索引**，
        // 且按行主序；levelX/levelY 必须为 0（ONE_LEVEL）。
        let (w, h) = (64u32, 48u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 32, 32);
        let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let parsed = crate::exr::read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let seen: Vec<(i32, i32, i32, i32)> = (0..4)
            .map(|b| {
                let o = parsed.chunk_offsets[b] as usize;
                let g = |k: usize| {
                    i32::from_le_bytes(bytes[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                };
                (g(0), g(1), g(2), g(3))
            })
            .collect();
        assert_eq!(
            seen,
            vec![(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)],
            "块头须是行主序的 tile 索引 + level 0"
        );
    }

    #[test]
    fn tiled_header_declares_tiles_attribute_and_tiled_version_flag() {
        // 版本字段须含 TILED(0x200)，且 `tiles` 属性恰 9 字节（不写 roundingMode）。
        let hdr = header_for_tiled(8, 8, &[("Y", Float)], Compression::None, 4, 4);
        let want = vec![plane("Y", PixelType::Float, 8, 8)];
        let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(
            version & crate::exr::version_flags::TILED,
            crate::exr::version_flags::TILED,
            "版本字段须置 TILED(0x200)，实测 {version:#x}"
        );
        // `tiles\0tiledesc\0` + size(9) —— 精确匹配属性头。
        let pat = b"tiles\0tiledesc\0";
        let at = bytes
            .windows(pat.len())
            .position(|w| w == pat)
            .unwrap_or_else(|| panic!("未找到 tiles 属性"));
        assert_eq!(
            u32::from_le_bytes(bytes[at + pat.len()..at + pat.len() + 4].try_into().unwrap()),
            9,
            "`tiles` 属性须恰 9 字节（官方实测值；多写 roundingMode 会判非法）"
        );
    }

    #[test]
    fn tiled_write_is_byte_identical_across_two_runs() {
        // 确定性：同输入连写两遍字节 `==`（含 DWA）。
        for comp in [
            Compression::None,
            Compression::Zip,
            Compression::Piz,
            Compression::Dwaa,
            Compression::Dwab,
        ] {
            let (w, h) = (37u32, 29u32); // 两轴都非对齐
            let want = vec![plane("Y", PixelType::Half, w, h)];
            let hdr = header_for_tiled(w, h, &[("Y", Half)], comp, 16, 16);
            let a = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            let b = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{} tiled 两次写出须字节相同", comp.code());
        }
    }

    #[test]
    fn tiled_write_covers_single_pixel_and_single_row_images() {
        // 退化图幅：1×1 图（唯一 tile 即边缘 tile）与 1×N / N×1。
        for (w, h, tw, th) in [(1u32, 1u32, 32u32, 32u32), (1u32, 40u32, 32u32, 32u32), (40u32, 1u32, 8u32, 8u32)] {
            let want = vec![plane("Y", PixelType::Float, w, h)];
            let hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::Zip, tw, th);
            let bytes = write_tiled_image(&hdr, &want).unwrap_or_else(|e| panic!("{e}"));
            assert_tiled_round_trip(&bytes, &want, None);
        }
    }

    #[test]
    fn tiled_write_rejects_missing_tile_desc_zero_size_and_multilevel() {
        let (w, h) = (8u32, 8u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];

        // ① 未标 tiled → 须指向 tiled 入口的互斥错误。
        let mut hdr = header_for(w, h, &[("Y", Float)], Compression::None);
        hdr.tile = Some(TileDesc { x_size: 4, y_size: 4, level_mode: 0 });
        let err = write_tiled_image(&hdr, &want).unwrap_err();
        assert!(format!("{err}").contains("header.tiled"), "{err}");

        // ② 标了 tiled 但缺 tiles 描述。
        let mut hdr2 = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
        hdr2.tile = None;
        let err = write_tiled_image(&hdr2, &want).unwrap_err();
        assert!(format!("{err}").contains("tiles"), "{err}");

        // ③ tile 尺寸为 0（会让网格除零）。
        let mut hdr3 = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 0, 4);
        hdr3.tile = Some(TileDesc { x_size: 0, y_size: 4, level_mode: 0 });
        let err = write_tiled_image(&hdr3, &want).unwrap_err();
        assert!(format!("{err}").contains("≥ 1"), "{err}");

        // ④ MIPMAP / RIPMAP：未实装，须**明确**说未实装而非静默按单层写。
        for mode in [1u8, 2u8] {
            let mut hdr4 = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
            hdr4.tile = Some(TileDesc { x_size: 4, y_size: 4, level_mode: mode });
            let err = write_tiled_image(&hdr4, &want).unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("未实装"), "mode={mode} 须写明未实装：{msg}");
            assert!(msg.contains("ONE_LEVEL"), "mode={mode} 须点名 ONE_LEVEL：{msg}");
        }
    }

    #[test]
    fn tiled_write_rejects_deep_and_multipart_headers() {
        let (w, h) = (8u32, 8u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let mut hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
        hdr.deep = true;
        let err = write_tiled_image(&hdr, &want).unwrap_err();
        assert!(format!("{err}").contains("deep"), "{err}");

        let mut hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
        hdr.multipart = true;
        let err = write_tiled_image(&hdr, &want).unwrap_err();
        assert!(format!("{err}").contains("多部件"), "{err}");
    }

    #[test]
    fn scanline_entry_still_rejects_tiled_and_points_at_tiled_entry() {
        // 布局是头部属性：扫描线入口不得**静默**把 tiled 当扫描线写。
        let (w, h) = (8u32, 8u32);
        let want = vec![plane("Y", PixelType::Float, w, h)];
        let hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
        let err = write_scanline_image(&hdr, &want).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("write_tiled_image"), "须指向 tiled 入口：{msg}");
        assert!(msg.contains("分块"), "须点名分块：{msg}");
    }

    #[test]
    fn deep_tiled_rejection_states_not_implemented_not_unsupported_format() {
        // 纪律：deep tiled 的拒绝理由必须是「未实装」，且指向已实装的 flat tiled。
        let mut hdr = header_for_deep(4, 4, &[("Zfloat", Float)], Compression::None);
        hdr.tiled = true;
        hdr.tile = Some(TileDesc { x_size: 2, y_size: 2, level_mode: 0 });
        let pixels = deep_pixels(&hdr);
        let err = write_deep_image(&hdr, &pixels).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("未实装"), "须写明「未实装」：{msg}");
        assert!(
            msg.contains("write_tiled_image"),
            "须指向已实装的 flat tiled 入口：{msg}"
        );
    }

    #[test]
    fn tiled_write_rejects_wrong_sized_planes_like_scanline() {
        // 平面对齐校验与扫描线共用：尺寸不符必须报错而非静默写出坏图。
        let (w, h) = (8u32, 8u32);
        let hdr = header_for_tiled(w, h, &[("Y", Float)], Compression::None, 4, 4);
        let mut wrong = plane("Y", PixelType::Float, w, h);
        wrong.values.pop();
        let err = write_tiled_image(&hdr, &[wrong]).unwrap_err();
        assert_eq!(err.code, "EXR3001", "{err}");
    }
}
