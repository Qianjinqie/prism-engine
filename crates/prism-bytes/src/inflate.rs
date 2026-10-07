//! RFC 1951 DEFLATE 解压（**裸码流**，不含 zlib 容器头/尾）。
//!
//! 本模块是全引擎唯一的 DEFLATE 解码实现（详见 crate 文档「为什么需要它」）。
//! zlib 容器在 [`crate::zlib`]。
//!
//! # 范围
//!
//! - 三类块：`stored`（BTYPE=0）/ `fixed Huffman`（BTYPE=1）/ `dynamic Huffman`（BTYPE=2）；
//! - BTYPE=3 为保留值 → 结构化拒绝；
//! - 码长表的码长码（HCLEN / HLIT / HDIST）按 RFC 1951 §3.2.7 的 `CLCL_ORDER` 排列。
//!
//! # 位序（DEFLATE 最易错处，RFC 1951 §3.1.1）
//!
//! - 普通整数字段（HLIT / HCLEN / 长度与距离附加值）**LSB 优先**；
//! - Huffman 码字本身**MSB 优先**（与其它字段相反）。
//!
//! # 确定性
//!
//! 固定遍历顺序（位流低位→高位、Huffman 码长 1→15、符号升序），无内部状态、
//! 无 HashMap：同输入逐字节一致，跨平台一致。

use crate::{BytesError, BytesResult};

/// 构造三段式错误（`location` 已是 `&'static str`）。
fn err(location: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> BytesError {
    BytesError::new(location, problem, fix)
}

// ===========================================================================
//  一、位读取器
// ===========================================================================

/// DEFLATE 位读取器（位序：**自低位到高位**）。
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, bit: 0 }
    }

    /// 读 `n`（≤ 24）位，低位在前。
    fn read_bits(&mut self, n: u32) -> BytesResult<u32> {
        let mut out = 0u32;
        let mut got = 0u32;
        while got < n {
            let byte = match self.data.get(self.pos) {
                Some(&b) => b as u32,
                None => {
                    return Err(err(
                        "inflate::read_bits",
                        "DEFLATE 位流提前耗尽",
                        "压缩块被截断；确认文件完整",
                    ))
                }
            };
            let avail = 8 - self.bit;
            let take = (n - got).min(avail);
            out |= ((byte >> self.bit) & ((1u32 << take) - 1)) << got;
            got += take;
            self.bit += take;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(out)
    }

    /// 对齐到字节边界。
    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }

    /// 读 1 字节（须先对齐）。
    fn read_u8(&mut self) -> BytesResult<u8> {
        self.align();
        let b = *self.data.get(self.pos).ok_or_else(|| {
            err("inflate::read_u8", "压缩块提前结束", "压缩块被截断；确认文件完整")
        })?;
        self.pos += 1;
        Ok(b)
    }

    /// 读 2 字节小端（须先对齐）。
    fn read_u16(&mut self) -> BytesResult<u16> {
        let lo = self.read_u8()? as u16;
        let hi = self.read_u8()? as u16;
        Ok(lo | (hi << 8))
    }
}

// ===========================================================================
//  二、Huffman 表
// ===========================================================================

/// canonical Huffman 表：按码长（1..=15）分桶的 `(码, 符号)` 线性查表。
///
/// 小表（DEFLATE 字母表 ≤ 288）线性扫描为 Mini Code 取向；确定性由固定遍历
/// 顺序保证（同表同输入必同符号）。
#[derive(Debug)]
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    /// 由码长数组构建 canonical 表（RFC 1951 §3.2.2）。
    fn from_lengths(lengths: &[u8]) -> BytesResult<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            let li = l as usize;
            if li > 15 {
                return Err(err(
                    "inflate::huffman",
                    format!("码长 {li} 超过 15 位上限"),
                    "Huffman 码表损坏；该压缩类型可能不是 ZIP",
                ));
            }
            counts[li] += 1;
        }
        counts[0] = 0;
        // 数组偏移 = 所有更短码长的累计码字数（canonical 排列顺序）。
        let mut offsets = [0usize; 16];
        let mut acc = 0usize;
        for len in 1..16 {
            offsets[len] = acc;
            acc += counts[len] as usize;
        }
        let mut symbols = vec![0u16; acc];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                if let Some(slot) = symbols.get_mut(offsets[l as usize]) {
                    *slot = sym as u16;
                    offsets[l as usize] += 1;
                }
            }
        }
        Ok(Self { counts, symbols })
    }

    /// 解一个符号（逐位累加码字，码长自 1 递增）。
    ///
    /// Huffman 码在位流中按 **MSB 优先** 排列（与 DEFLATE 其它字段相反）。
    fn decode(&self, br: &mut BitReader<'_>) -> BytesResult<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0usize;
        for len in 1..16usize {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                let idx = index + (code - first) as usize;
                return self.symbols.get(idx).copied().ok_or_else(|| {
                    err(
                        "inflate::decode",
                        "Huffman 符号索引越界",
                        "码表损坏；确认该块确为 DEFLATE 压缩",
                    )
                });
            }
            index += count as usize;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(err(
            "inflate::decode",
            "码长超过 15 位仍未匹配",
            "Huffman 码表非法；该块可能不是 DEFLATE 流",
        ))
    }
}

// ===========================================================================
//  三、RFC 1951 §3.2.5 常量表
// ===========================================================================

/// 长度码（257..=285）→ 基础长度。
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// 长度码附加位数。
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// 距离码（0..=29）→ 基础距离。
const DIST_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// 距离码附加位数。
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// 码长字母表的码长重排顺序（RFC 1951 §3.2.7）。
///
/// **同时导出**（`pub`）：`prism-io/src/exr.rs` 的**测试侧自建 dynamic 编码器**
/// 需要这张表来排码长码。解码器本身不依赖它对外可见——只是同一份 RFC 常量，
/// 导出一份比让两个 crate 各抄一份更可靠（改错一处就会静默分叉）。
pub const CLCL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ===========================================================================
//  四、块解码
// ===========================================================================

/// 块类型 0：stored（对齐后 LEN / NLEN + 原始字节）。
fn inflate_stored(br: &mut BitReader<'_>, out: &mut Vec<u8>) -> BytesResult<()> {
    br.align();
    let len = br.read_u16()?;
    let nlen = br.read_u16()?;
    if len != !nlen {
        return Err(err(
            "inflate::stored",
            format!("LEN = {len} 与 NLEN = {nlen} 不互补"),
            "stored 块头损坏；确认压缩数据完整",
        ));
    }
    for _ in 0..len {
        out.push(br.read_u8()?);
    }
    Ok(())
}

/// 块类型 1：fixed Huffman 表（RFC 1951 §3.2.6）。
///
/// **返回 `(字面表, 距离表)` 两个独立表**——这是 RFC 的硬要求，也是本 crate
/// 修掉的第二个历史缺陷：旧 `prism-io` 侧只建了一张 288 项的字面表并把它
/// 同时当字面表与距离表用，导致 fixed 块里**任何回溯都会解出错误符号**。
/// fixed 距离码是 5 位定长、30 项（表）。
fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit_lengths = vec![0u8; 288];
    for (i, l) in lit_lengths.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    // fixed 距离码：全部 5 位、30 项（RFC 1951 §3.2.6）。
    let dist_lengths = vec![5u8; 30];
    // 两张表都由常量构造，码长必然 ∈ 1..=9、符号数 ≤ 288 < 65536，故
    // `from_lengths` 不会失败；不返回 Result（Mini Code：常量表的正确性由
    // 测试 `fixed_huffman_backreference_decodes` 守护）。
    let empty = || Huffman { counts: [0; 16], symbols: Vec::new() };
    (
        Huffman::from_lengths(&lit_lengths).unwrap_or_else(|_| empty()),
        Huffman::from_lengths(&dist_lengths).unwrap_or_else(|_| empty()),
    )
}

/// 块类型 2：dynamic Huffman（读 HLIT / HDIST / HCLEN + 码长表 + 码长解码）。
fn dynamic_tables(br: &mut BitReader<'_>) -> BytesResult<(Huffman, Huffman)> {
    let hlit = br.read_bits(5)? as usize + 257;
    let hdist = br.read_bits(5)? as usize + 1;
    let hclen = br.read_bits(4)? as usize + 4;

    let mut cl = [0u8; 19];
    for i in 0..hclen {
        cl[CLCL_ORDER[i]] = br.read_bits(3)? as u8;
    }
    let cl_table = Huffman::from_lengths(&cl)?;

    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0usize;
    while i < total {
        let sym = cl_table.decode(br)?;
        match sym {
            0..=15 => {
                if let Some(slot) = lengths.get_mut(i) {
                    *slot = sym as u8;
                }
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(err(
                        "inflate::dynamic",
                        "码长重复码 16 出现在首位",
                        "码长表非法",
                    ));
                }
                let rep = br.read_bits(2)? as usize + 3;
                let prev = lengths[i - 1];
                for _ in 0..rep {
                    if i >= total {
                        return Err(err(
                            "inflate::dynamic",
                            "码长重复越过表尾",
                            "码长表非法",
                        ));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let rep = br.read_bits(3)? as usize + 3;
                i = (i + rep).min(total);
            }
            18 => {
                let rep = br.read_bits(7)? as usize + 11;
                i = (i + rep).min(total);
            }
            _ => {
                return Err(err(
                    "inflate::dynamic",
                    format!("码长符号 {sym} > 18"),
                    "码长表非法",
                ))
            }
        }
    }
    let lit = Huffman::from_lengths(lengths.get(..hlit).ok_or_else(|| {
        err("inflate::dynamic", "HLIT 越过码长表尾", "码长表非法")
    })?)?;
    let dist = Huffman::from_lengths(lengths.get(hlit..).ok_or_else(|| {
        err("inflate::dynamic", "HDIST 越过码长表尾", "码长表非法")
    })?)?;
    Ok((lit, dist))
}

/// 解一个 Huffman 块（字面 / 长度-距离对）。
fn inflate_block(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
) -> BytesResult<()> {
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let idx = (sym - 257) as usize;
                let extra = LENGTH_EXTRA
                    .get(idx)
                    .copied()
                    .ok_or_else(|| err("inflate::block", "长度码越界", "字面表非法"))?;
                let base = LENGTH_BASE.get(idx).copied().unwrap_or(0);
                let length = base as usize + br.read_bits(extra as u32)? as usize;
                let dsym = dist.decode(br)? as usize;
                let dbase = DIST_BASE
                    .get(dsym)
                    .copied()
                    .ok_or_else(|| err("inflate::block", "距离码越界", "距离表非法"))?;
                let dextra = DIST_EXTRA.get(dsym).copied().unwrap_or(0);
                let distance = dbase as usize + br.read_bits(dextra as u32)? as usize;
                if distance == 0 || distance > out.len() {
                    return Err(err(
                        "inflate::block",
                        format!("回溯距离 {distance} 越界（已产出 {} 字节）", out.len()),
                        "压缩流损坏；确认文件完整",
                    ));
                }
                let start = out.len() - distance;
                // 逐字节前推：允许 length > distance 的重叠拷贝（RFC 1951 §3.2.3），
                // 这是 LZ77 的本质（"ABABABA" 可由 "AB" + 长度 5/距离 2 得到）。
                for k in 0..length {
                    let byte = out.get(start + k).copied().unwrap_or(0);
                    out.push(byte);
                }
            }
            _ => {
                return Err(err(
                    "inflate::block",
                    format!("字面符号 {sym} 越界（≤ 285）"),
                    "字面表非法",
                ))
            }
        }
    }
}

/// DEFLATE **裸码流**解压（不含 zlib 头/尾）。
///
/// 逐块解码直到 `BFINAL = 1`；BTYPE=3（保留）为结构化错误。
pub fn inflate_raw(data: &[u8]) -> BytesResult<Vec<u8>> {
    inflate_raw_prefix(data).map(|(out, _)| out)
}

/// DEFLATE 裸码流解压，**额外回传已消费的输入字节数**。
///
/// 消费量 = 解码结束时位读取器所在字节下标（已向上对齐到字节边界），
/// 即 DEFLATE 码流在 `data` 中的实际长度。zlib 容器靠它定位尾部 4 字节
/// Adler-32（[`crate::zlib`]）。
///
/// 语义与 [`inflate_raw`] 完全一致，只是多带一个元信息；两者共用同一实现。
pub fn inflate_raw_prefix(data: &[u8]) -> BytesResult<(Vec<u8>, usize)> {
    let mut br = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            0 => inflate_stored(&mut br, &mut out)?,
            1 => {
                let (lit, dist) = fixed_tables();
                inflate_block(&mut br, &mut out, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut br)?;
                inflate_block(&mut br, &mut out, &lit, &dist)?;
            }
            _ => {
                return Err(err(
                    "inflate::inflate_raw",
                    "块类型 3 为保留值，非法",
                    "该压缩数据不是 DEFLATE 流；确认 ZIP 压缩未被改写",
                ))
            }
        }
        if last == 1 {
            break;
        }
    }
    // 最后一个块的填充位补齐到字节边界后，才是码流的真实长度。
    br.align();
    Ok((out, br.pos))
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- 测试用 DEFLATE 构造器（自研压缩，不依赖任何第三方） ------------

    /// 位写入器：整数字段 LSB 优先，Huffman 码字 MSB 优先（RFC 1951 §3.1.1）。
    #[derive(Default)]
    struct BitWriter {
        out: Vec<u8>,
        cur: u32,
        nbit: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self::default()
        }

        /// 写 `n` 位整数，LSB 优先。
        fn push(&mut self, value: u32, n: u32) {
            let mut v = value;
            for _ in 0..n {
                if v & 1 == 1 {
                    self.cur |= 1 << self.nbit;
                }
                v >>= 1;
                self.step();
            }
        }

        /// 写一个 Huffman 码字，MSB 优先。
        fn push_msb(&mut self, code: u32, n: u32) {
            for i in (0..n).rev() {
                if (code >> i) & 1 == 1 {
                    self.cur |= 1 << self.nbit;
                }
                self.step();
            }
        }

        fn step(&mut self) {
            self.nbit += 1;
            if self.nbit == 8 {
                self.out.push(self.cur as u8);
                self.cur = 0;
                self.nbit = 0;
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.nbit > 0 {
                self.out.push(self.cur as u8);
                self.nbit = 0;
            }
            self.out
        }
    }

    /// 由码长数组求 canonical Huffman 码字（RFC 1951 §3.2.2）。
    fn canonical(lengths: &[u8]) -> Vec<(u32, u32)> {
        let mut counts = [0u32; 16];
        for &l in lengths {
            if l != 0 && l < 16 {
                counts[l as usize] += 1;
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for len in 1..16usize {
            next[len] = code;
            code = (code + counts[len]) << 1;
        }
        let mut out = vec![(0u32, 0u32); lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 && l < 16 {
                let li = l as usize;
                out[sym] = (next[li], l as u32);
                next[li] += 1;
            }
        }
        out
    }

    /// 把码长数组压成 `(码长符号, 附加值, 附加位数)` 序列（零游程用 17/18）。
    fn encode_cl_lengths(all: &[u8]) -> Vec<(u8, u32, u32)> {
        let mut seq: Vec<(u8, u32, u32)> = Vec::new();
        let mut i = 0usize;
        while i < all.len() {
            if all[i] != 0 {
                seq.push((all[i], 0, 0));
                i += 1;
                continue;
            }
            let mut run = 1usize;
            while i + run < all.len() && all[i + run] == 0 {
                run += 1;
            }
            let mut left = run;
            while left > 138 {
                seq.push((18, 138 - 11, 7));
                left -= 138;
            }
            if left >= 11 {
                seq.push((18, left as u32 - 11, 7));
            } else if left >= 3 {
                seq.push((17, left as u32 - 3, 3));
            } else {
                for _ in 0..left {
                    seq.push((0, 0, 0));
                }
            }
            i += run;
        }
        seq
    }

    /// 单个 stored 块。
    fn stored_block(payload: &[u8], last: bool) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(if last { 0x01 } else { 0x00 });
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// stored 块流（跨 65535 自动分块）。
    fn deflate_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut off = 0usize;
        loop {
            let n = (payload.len() - off).min(65535);
            let last = off + n >= payload.len();
            out.extend_from_slice(&stored_block(&payload[off..off + n], last));
            off += n;
            if last {
                break;
            }
        }
        out
    }

    /// fixed 块的一条指令。
    enum FixedOp {
        Lit(u16),
        /// `(长度码 257..=285, 长度附加值, 距离码 0..=29, 距离附加值)`
        Match(u16, u32, u16, u32),
        End,
    }

    /// fixed 字面/长度码字（RFC 1951 §3.2.6）。
    fn fixed_lit_code(sym: u16) -> (u32, u32) {
        match sym {
            0..=143 => (0x30 + sym as u32, 8),
            144..=255 => (0x190 + (sym as u32 - 144), 9),
            256..=279 => (sym as u32 - 256, 7),
            _ => (0xc0 + (sym as u32 - 280), 8),
        }
    }

    /// 构造 fixed Huffman 块流（BFINAL=1, BTYPE=01）。
    fn deflate_fixed(ops: &[FixedOp]) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.push(1, 1);
        w.push(1, 2);
        for op in ops {
            match *op {
                FixedOp::Lit(sym) => {
                    let (c, n) = fixed_lit_code(sym);
                    w.push_msb(c, n);
                }
                FixedOp::Match(lsym, lextra, dsym, dextra) => {
                    let (c, n) = fixed_lit_code(lsym);
                    w.push_msb(c, n);
                    w.push(lextra, LENGTH_EXTRA[(lsym - 257) as usize] as u32);
                    w.push_msb(dsym as u32, 5); // fixed 距离码恒 5 位
                    w.push(dextra, DIST_EXTRA[dsym as usize] as u32);
                }
                FixedOp::End => {
                    let (c, n) = fixed_lit_code(256);
                    w.push_msb(c, n);
                }
            }
        }
        w.finish()
    }

    /// dynamic Huffman 块流（自研压缩，仅编码 'A'/'B' 字面，不发回溯）。
    fn deflate_dynamic(payload: &[u8]) -> Vec<u8> {
        assert!(
            payload.iter().all(|&b| b == b'A' || b == b'B'),
            "deflate_dynamic 只编码 'A'/'B' 字面"
        );
        // 码长符号集合恰为 {0, 1, 2, 17, 18}：3 个 2 位 + 2 个 3 位，Kraft 和 = 1。
        let mut cl_len = [0u8; 19];
        for &sym in &[0u8, 2, 17] {
            cl_len[sym as usize] = 2;
        }
        for &sym in &[1u8, 18] {
            cl_len[sym as usize] = 3;
        }
        let hclen = CLCL_ORDER
            .iter()
            .rposition(|&s| cl_len[s] != 0)
            .map(|i| i + 1)
            .unwrap_or(4)
            .clamp(4, 19);
        let cl_codes = canonical(&cl_len);

        let mut lit_len = [0u8; 257];
        lit_len[b'A' as usize] = 2;
        lit_len[b'B' as usize] = 2;
        lit_len[256] = 2;
        let dist_len = [1u8; 1];
        let lit_codes = canonical(&lit_len);

        let mut w = BitWriter::new();
        w.push(1, 1);
        w.push(2, 2);
        w.push(0, 5);
        w.push(0, 5);
        w.push((hclen - 4) as u32, 4);
        for &s in CLCL_ORDER.iter().take(hclen) {
            w.push(cl_len[s] as u32, 3);
        }
        let all: Vec<u8> = lit_len.iter().copied().chain(dist_len.iter().copied()).collect();
        for (sym, extra, bits) in encode_cl_lengths(&all) {
            let c = cl_codes[sym as usize];
            w.push_msb(c.0, c.1);
            if bits > 0 {
                w.push(extra, bits);
            }
        }
        for &b in payload {
            let c = lit_codes[b as usize];
            w.push_msb(c.0, c.1);
        }
        let end = lit_codes[256];
        w.push_msb(end.0, end.1);
        w.finish()
    }

    // -- 1. stored --------------------------------------------------------

    #[test]
    fn stored_block_roundtrip() {
        let payload: Vec<u8> = (0..100u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(inflate_raw(&deflate_stored(&payload)).expect("stored 必可解码"), payload);
    }

    #[test]
    fn stored_empty_payload_is_empty_output() {
        assert_eq!(inflate_raw(&deflate_stored(b"")).expect("空 stored 必可解码"), Vec::<u8>::new());
    }

    #[test]
    fn stored_multi_block_across_65535_boundary() {
        // 70000 > 65535，强制走多块路径。
        let payload: Vec<u8> = (0..70000u32).map(|i| (i * 31 % 251) as u8).collect();
        assert_eq!(inflate_raw(&deflate_stored(&payload)).expect("多块 stored 必可解码"), payload);
    }

    #[test]
    fn stored_len_nlen_not_complementary_rejected() {
        let mut s = vec![0x01];
        s.extend_from_slice(&5u16.to_le_bytes());
        s.extend_from_slice(&9999u16.to_le_bytes()); // NLEN 错
        s.extend_from_slice(&[1, 2, 3, 4, 5]);
        assert!(inflate_raw(&s).is_err(), "LEN/NLEN 不互补必须拒绝");
    }

    // -- 2. fixed Huffman -------------------------------------------------

    #[test]
    fn fixed_huffman_known_stream() {
        // 字面 'A'(65) 'B'(66) 'C'(67) + 块结束(256)。
        // fixed 表 0..=143 → 8 位码（码字 = sym + 0x30，MSB 优先）。
        let stream = deflate_fixed(&[
            FixedOp::Lit(65),
            FixedOp::Lit(66),
            FixedOp::Lit(67),
            FixedOp::End,
        ]);
        assert_eq!(inflate_raw(&stream).expect("fixed 流必可解码"), b"ABC");
    }

    #[test]
    fn fixed_huffman_backreference_decodes() {
        // 回归守卫：旧 `prism-io` 侧把 288 项字面表**同时**当距离表用，
        // fixed 块里的任何回溯都解错。本测试覆盖的 5 位距离码。
        let stream = deflate_fixed(&[
            FixedOp::Lit(b'A' as u16),
            FixedOp::Match(257, 0, 0, 0), // 长度 3、距离 1 → "AAAA"
            FixedOp::End,
        ]);
        assert_eq!(inflate_raw(&stream).expect("fixed 回溯必可解码"), b"AAAA");
    }

    #[test]
    fn fixed_huffman_overlapping_copy_distance_less_than_length() {
        // LZ77 重叠拷贝："AB" + 长度 3/距离 2 → "ABABA"（RFC 1951 §3.2.3）。
        let stream = deflate_fixed(&[
            FixedOp::Lit(b'A' as u16),
            FixedOp::Lit(b'B' as u16),
            FixedOp::Match(257, 0, 1, 0),
            FixedOp::End,
        ]);
        assert_eq!(inflate_raw(&stream).expect("重叠拷贝必可解码"), b"ABABA");
    }

    #[test]
    fn fixed_huffman_rle_run_of_258() {
        // 长度码 285 → 基础长度 258（附加位 0），距离 1。
        let stream = deflate_fixed(&[
            FixedOp::Lit(b'A' as u16),
            FixedOp::Match(285, 0, 0, 0),
            FixedOp::End,
        ]);
        let out = inflate_raw(&stream).expect("258 连跑必可解码");
        assert_eq!(out.len(), 259);
        assert!(out.iter().all(|&b| b == b'A'));
    }

    #[test]
    fn distance_code_first_entry_dist_base_head() {
        // DIST_BASE 首项：距离码 0 → 距离 1。
        assert_eq!(DIST_BASE[0], 1);
        let stream = deflate_fixed(&[
            FixedOp::Lit(b'Z' as u16),
            FixedOp::Match(257, 0, 0, 0),
            FixedOp::End,
        ]);
        assert_eq!(inflate_raw(&stream).expect("DIST_BASE 首项必可解码"), b"ZZZZ");
    }

    #[test]
    fn distance_code_last_entry_dist_base_tail_with_zero_extra() {
        // DIST_BASE 末项：距离码 29 → 基础距离 24577、附加位 13 位取 0。
        assert_eq!(DIST_BASE[29], 24577);
        let base: Vec<u8> = (0..24577u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut stream = stored_block(&base, false);
        stream.extend_from_slice(&deflate_fixed(&[FixedOp::Match(285, 0, 29, 0), FixedOp::End]));
        let out = inflate_raw(&stream).expect("DIST_BASE 末项必可解码");
        let mut want = base.clone();
        want.extend_from_slice(&base[..258]);
        assert_eq!(out, want);
    }

    #[test]
    fn distance_code_last_entry_extra_bits_saturated() {
        // 距离码 29 + 附加位打满（8191）→ 距离 24577 + 8191 = 32768。
        let base: Vec<u8> = (0..32768u32).map(|i| (i * 13 % 251) as u8).collect();
        let mut stream = stored_block(&base, false);
        stream.extend_from_slice(&deflate_fixed(&[FixedOp::Match(257, 0, 29, 8191), FixedOp::End]));
        let out = inflate_raw(&stream).expect("距离码附加位打满必可解码");
        let mut want = base.clone();
        want.extend_from_slice(&base[..3]);
        assert_eq!(out, want);
    }

    #[test]
    fn fixed_huffman_length_code_overflow_symbol_rejected() {
        // 字面符号 286/287 在 fixed 表里无定义（长度码止于 285）。
        let mut w = BitWriter::new();
        w.push(1, 1);
        w.push(1, 2);
        w.push_msb(0xc0 + 6, 8); // 280..287 段的第 7 个 = 286
        let stream = w.finish();
        assert!(inflate_raw(&stream).is_err(), "字面符号 286 必须结构化拒绝");
    }

    // -- 3. dynamic Huffman -----------------------------------------------

    #[test]
    fn dynamic_huffman_round_trips_through_cl_alphabet() {
        // 覆盖 18（≥11 零）、17（3~10 零）、0（1~2 零）三条分支，
        // 以及零游程跨越 lit/dist 表边界的情形。
        for (a, b) in [
            (1usize, 0usize),
            (3, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (64, 0),
            (140, 1),
            (3, 4),
        ] {
            let mut payload = vec![b'A'; a];
            payload.extend(std::iter::repeat_n(b'B', b));
            let got = inflate_raw(&deflate_dynamic(&payload))
                .unwrap_or_else(|e| panic!("dynamic {a}+{b}: {e}"));
            assert_eq!(got, payload, "dynamic Huffman 往返失真（{a}+{b}）");
        }
    }

    #[test]
    fn dynamic_huffman_block_type_bits_are_two() {
        // 防止测试退化到别的编码路径。
        let s = deflate_dynamic(b"AAB");
        assert_eq!(s[0] & 0x06, 0x04, "BTYPE 应为 2（dynamic），实际 {:#04x}", s[0]);
    }

    #[test]
    fn dynamic_huffman_truncation_never_panics() {
        let full = deflate_dynamic(&[b'A'; 40]);
        for cut in 1..full.len() {
            let r = inflate_raw(&full[..cut]);
            assert!(r.is_err() || r.is_ok(), "截断到 {cut} 字节不得 panic");
        }
        assert!(inflate_raw(&full[..3]).is_err(), "dynamic 头被截断应结构化报错");
    }

    // -- 4. 畸形流 --------------------------------------------------------

    #[test]
    fn reserved_block_type_3_rejected() {
        // BFINAL=1, BTYPE=11。
        let bad = [0x07u8, 0, 0, 0, 0];
        assert_eq!(
            inflate_raw(&bad).expect_err("块类型 3 必须拒绝").location,
            "inflate::inflate_raw"
        );
    }

    #[test]
    fn empty_stream_is_structured_error() {
        assert!(inflate_raw(&[]).is_err(), "空流必须结构化报错而非 panic");
    }

    #[test]
    fn truncated_stored_header_is_structured_error() {
        assert!(inflate_raw(&[0x01]).is_err(), "截断流必须结构化报错");
        assert!(inflate_raw(&[0x01, 0x05]).is_err(), "stored 头被截断必须报错");
    }

    #[test]
    fn backreference_distance_beyond_output_rejected() {
        // 空输出上直接发一个距离 1 的回溯 → 越界。
        let stream = deflate_fixed(&[FixedOp::Match(257, 0, 0, 0), FixedOp::End]);
        assert_eq!(
            inflate_raw(&stream).expect_err("回溯越界必须拒绝").location,
            "inflate::block"
        );
    }

    #[test]
    fn huffman_code_length_over_15_rejected() {
        // 码长 16 超过 RFC 1951 的 15 位上限。
        let lengths = [16u8, 1, 1];
        assert_eq!(
            Huffman::from_lengths(&lengths).expect_err("码长 > 15 必须拒绝").location,
            "inflate::huffman"
        );
    }

    #[test]
    fn huffman_decode_unmatched_code_is_structured_error() {
        // 全零码长表 → 表里没有任何码字 → 匹配必然失败。
        // 直接调 `decode`（而非走 `inflate_raw`）：后者遇 BTYPE=1 会走
        // fixed 表，压根碰不到这张表。位流给足 3 字节，保证 decode 逐位
        // 累加 16 次都不会先在 read_bits 上耗尽。
        let t = Huffman::from_lengths(&[0u8; 4]).expect("零码长表可构建");
        assert_eq!(t.counts, [0u16; 16], "零码长表不应有任何码字");
        let bytes = [0b1111_1111u8, 0xff, 0xff];
        let mut br = BitReader::new(&bytes);
        let e = t.decode(&mut br).expect_err("无匹配码字必须结构化报错");
        assert_eq!(e.location, "inflate::decode");
        assert!(e.problem.contains("15 位"), "问题段须点明 15 位上限：{}", e.problem);
    }

    #[test]
    fn dynamic_code_length_repeat_at_head_rejected() {
        // 码长重复码 16 出现在首位是非法用法（无前驱可复制）。
        let mut cl = [0u8; 19];
        cl[16] = 1;
        let cl_table = Huffman::from_lengths(&cl).expect("cl 表可构建");
        assert!(cl_table.counts[1] > 0, "cl 表应含 1 位码");
        let mut w = BitWriter::new();
        w.push(1, 1);
        w.push(2, 2);
        w.push(0, 5); // HLIT
        w.push(0, 5); // HDIST
        w.push(0, 4); // HCLEN = 4
        // HCLEN=4 → CLCL_ORDER 前 4 项 = [16, 17, 18, 0]，各写 3 位。
        w.push(1, 3); // 符号 16 → 1 位
        w.push(0, 3);
        w.push(0, 3);
        w.push(0, 3);
        w.push_msb(0, 1); // 解出符号 16
        w.push(0, 2); // 附加值
        let stream = w.finish();
        let e = inflate_raw(&stream).expect_err("首位重复码必须拒绝");
        assert_eq!(e.location, "inflate::dynamic");
    }

    // -- 5. 迁移不变性 ---------------------------------------------------

    /// **迁移不变性**：固定输入 → 固定期望字节数组。
    ///
    /// 这批期望值是**迁移前**从 `prism-media` 侧旧实现跑出来的实际输出
    /// （旧 `prism-io` 侧在 fixed 回溯上是错的，故不作为期望源）。
    /// 迁移只改代码位置，不改解码结果，故本测试必须恒过。
    #[test]
    fn migration_invariance_fixed_expected_bytes() {
        // ① stored：0..100 的 (i*7%251) 字节。
        let p100: Vec<u8> = (0..100u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(inflate_raw(&deflate_stored(&p100)).expect("stored"), p100);

        // ② fixed 纯字面。
        assert_eq!(
            inflate_raw(&deflate_fixed(&[
                FixedOp::Lit(65),
                FixedOp::Lit(66),
                FixedOp::Lit(67),
                FixedOp::End
            ]))
            .expect("fixed"),
            b"ABC"
        );

        // ③ fixed 回溯（迁移前旧 media 侧输出 259 个 'A'）。
        assert_eq!(
            inflate_raw(&deflate_fixed(&[
                FixedOp::Lit(b'A' as u16),
                FixedOp::Match(285, 0, 0, 0),
                FixedOp::End
            ]))
            .expect("fixed rle"),
            vec![b'A'; 259]
        );

        // ④ fixed 重叠拷贝（迁移前旧 media 侧输出 "ABABA"）。
        assert_eq!(
            inflate_raw(&deflate_fixed(&[
                FixedOp::Lit(b'A' as u16),
                FixedOp::Lit(b'B' as u16),
                FixedOp::Match(257, 0, 1, 0),
                FixedOp::End
            ]))
            .expect("fixed overlap"),
            b"ABABA"
        );

        // ⑤ dynamic 往返（迁移前旧 media 侧一致）。
        let mut payload = vec![b'A'; 140];
        payload.push(b'B');
        assert_eq!(inflate_raw(&deflate_dynamic(&payload)).expect("dynamic"), payload);
    }
}
