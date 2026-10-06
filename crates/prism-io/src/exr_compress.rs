//! EXR 压缩补全：**PIZ / PXR24 / B44 / B44A** 的真解码（+ 自研编码器，路E）。
//!
//! # 为什么要这个模块
//!
//! `exr.rs` 原本只支持 `None/Rle/Zips/Zip`（+ `exr_dwa.rs` 的 DWAA/DWAB）。
//! 真实产线里 **PIZ**（Maya/Nuke 动画管线常见输出）、**B44 / B44A**（老式常用）
//! 与 **PXR24** 都是合法 `compression` 字节，本模块把它们补齐。
//!
//! # 铁律
//!
//! - **零第三方 crate**：只用工作区内部 crate + `std`；
//! - **确定性**：所有遍历顺序 = 文件中 `channels` 的出现顺序 × 固定升序下标；
//!   Huffman 码表构建用定序最小堆（同频按符号号定序），**不依赖 HashMap 顺序**；
//! - **C18 永不 panic**：库代码无 `unwrap/expect/panic!/越界索引/整数溢出`
//!   （全部走 `get()/get_mut()` 与 `checked_*`），畸形输入返回三段式 [`ExrError`]。
//!
//! # 复用证据（不重写已有基础设施）
//!
//! | 本模块用到 | 来自 | 说明 |
//! |---|---|---|
//! | `exr_dwa::huf_uncompress` | [`crate::exr_dwa`] | **PIZ 的 Huf 解码层直接复用**（`piz_decompress` 步骤 ③）。OpenEXR 的 `ImfHuf.cpp` 里 `hufUncompress` 是 DWA 与 PIZ **共用的同一个函数**（`PizCompressor::uncompress` 直接调它），本仓库亦然，故**不重写第二份 Huf 解码器**。 |
//! | `exr::zlib_decompress` | [`crate::exr`] | PXR24 的负载是 **zlib 流**（OpenEXR 调 `zlib::compress2`）。 |
//! | `exr::rle_compress` / `exr_write::deflate_zlib` | [`crate::exr`] / [`crate::exr_write`] | 编码器（测试对拍用）。 |
//!
//! **未复用而新写的部分**：`huf` 的**编码侧**（`exr_dwa` 只有解码器；OpenEXR 用
//! `std::make_heap` 无法照搬，本模块用 `BinaryHeap<Reverse<(freq, symbol)>>` 给出
//! 等价且**定序**的码长），以及 `wav2Encode/wav2Decode`、B44 `pack/unpack`、
//! PXR24 定点拆分 —— 这些在 `prism-io` 内**不存在**可比实现。
//!
//! # 与任务书描述的差异（以 OpenEXR 3.1.5 源码为准）
//!
//! 任务书写的是"概念示意"，与 OpenEXR 真实算法有出入。本模块**按真实源码实现**，
//! 否则打不开真实文件：
//!
//! - **PIZ**：真实布局是 `u16 minNonZero` / `u16 maxNonZero` / **bitmap 变长段**
//!   （`minNonZero..=maxNonZero`，bitmap 共 8192 字节）/ `i32 长度` / **Huf 流**。
//!   熵编码层是 Huf（`hufCompress`），**不是**"luminance 段码长表"；
//!   小波是 `wav2Decode`（14/16 bit 两套 lifting 的 2D 级联），
//!   **不是** `_wavelet_decode_offset/size/border` 那套索引表（那是 OpenEXR 1.x 的旧 API）。
//! - **PXR24**：真实算法是「逐通道差分 → 字节平面转置 → **zlib**」，
//!   **既不是** `(A,B,T,P)` 变换，**也不走 RLE**。FLOAT 通道**有损**（32bit→24bit）。
//! - **B44 / B44A**：真实算法是「HALF 通道 4×4 块 → 14 字节（或平场 3 字节）」，
//!   UINT/FLOAT 通道**原样复制**，**没有 RLE、没有 8×8 通道级小波**。
//!   B44 与 B44A 的**解码路径完全相同**，唯一差别是 B44A 允许 3 字节平场表示。
//!
//! # 诚实边界
//!
//! - `pLinear == true` 的 **B44/B44A HALF 通道被结构化拒绝**（见
//!   [`p_linear_unsupported`]）。该路径需要 OpenEXR 的
//!   `expTable`/`logTable`——两张各 128 KB 的**离线生成常量表**。本模块
//!   **不内嵌**它们，**也不按公式重算**：公式重算在饱和区无法与参考表逐位
//!   对齐（例：索引 `0x6000` 处 `exp(f16(0x6000)/8)` 远超 `HALF_MAX`，
//!   参考表给 `0x7c00`=+inf，公式给 `0x7bff`），属"看似支持、实则位级不可保"。
//!   `pLinear == false`（OpenEXR 的默认，绝大多数文件都是）完整实现。
//! - PIZ 编码器在「码长 > 58」时结构化报错（OpenEXR 在此处是 `assert`，即
//!   未定义行为）；这只影响**写出**，解码侧对真实文件不受影响。
//! - tiled + PIZ/PXR24/B44/B44A 走同一 `decompress_block` 上下文路径，已支持。

/// Huf **编码侧**（PIZ 熵层）需要的容器；解码侧复用 `exr_dwa`，不引入。
///
/// 写入器（`exr_write`）现在会调用 [`piz_compress`]，故这两个导入不再 `#[cfg(test)]`。
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::exr::{comp_err, ExrChannel, ExrResult, PixelType};
// half 位模式 ↔ f32：B44 的 exp/log 表重算与测试脚手架都用。

/// 统一错误定位前缀（本模块所有结构化错误的 `location`）。
const LOC: &str = "exr_compress";

// ===========================================================================
//  一、块解码上下文
// ===========================================================================

/// 一个压缩块的解码上下文。
///
/// PIZ / PXR24 / B44 三族都**不是**"整块字节流"压缩：它们要按
/// 「通道 → 扫描线 → 像素」重排像素，并且 B44 还要看通道的 `pLinear` 标志。
/// 这与 DWA 需要通道上下文同理（见 [`crate::exr_dwa::DwaContext`]）。
#[derive(Debug, Clone, Copy)]
pub struct BlockCtx<'a> {
    /// 通道表（顺序 = 文件出现顺序，确定性来源）。
    pub channels: &'a [ExrChannel],
    /// 块宽（像素）。
    pub width: usize,
    /// 块高（扫描线数；末块可能不足整块）。
    pub height: usize,
}

impl<'a> BlockCtx<'a> {
    /// 块解压后的**原始字节数**（EXR 块内布局：扫描线 × 通道 × 像素）。
    pub fn raw_bytes(&self) -> usize {
        let per_px: usize = self
            .channels
            .iter()
            .map(|c| c.pixel_type.byte_width())
            .sum();
        per_px.saturating_mul(self.width).saturating_mul(self.height)
    }

    /// 校验上下文自洽（宽高非零、通道非空、子采样已在上游被拒）。
    fn validate(&self, who: &'static str) -> ExrResult<()> {
        if self.channels.is_empty() {
            return Err(comp_err(
                who,
                "通道表为空，无法确定块内布局",
                "确认文件完整（channels 属性未被截断）",
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(comp_err(
                who,
                format!("块尺寸非法（{}×{}）", self.width, self.height),
                "确认 dataWindow 与块偏移表一致",
            ));
        }
        Ok(())
    }
}

/// 边界安全的只读字节游标（本模块私有；`exr::Cursor` 是私有的，不重复暴露）。
struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
    who: &'static str,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8], who: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            who,
        }
    }

    fn take(&mut self, n: usize) -> ExrResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            comp_err(
                self.who,
                format!("偏移 {} + 长度 {n} 溢出", self.pos),
                "压缩块被截断或长度字段损坏",
            )
        })?;
        let s = self.data.get(self.pos..end).ok_or_else(|| {
            comp_err(
                self.who,
                format!(
                    "需要 [{}..{}) 共 {n} 字节，块仅 {} 字节",
                    self.pos,
                    end,
                    self.data.len()
                ),
                "压缩块被截断；确认文件完整",
            )
        })?;
        self.pos = end;
        Ok(s)
    }

    fn u16(&mut self) -> ExrResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i32(&mut self) -> ExrResult<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

/// 按压缩类型解码一个块，产出 **EXR 块内布局的原始字节**（小端）。
///
/// 调用方（[`crate::exr`]）随后交给既有 `unpack_block`，与 `None/Zip/Rle` 同一契约。
pub fn decompress_block(
    compression: crate::exr::Compression,
    payload: &[u8],
    ctx: &BlockCtx<'_>,
) -> ExrResult<Vec<u8>> {
    use crate::exr::Compression as C;
    match compression {
        C::Piz => piz_decompress(payload, ctx),
        C::Pxr24 => pxr24_decompress(payload, ctx),
        // B44 与 B44A 的**解码路径完全一致**（3 字节平场表示两者都接受，
        // 差异只在编码器是否允许产出它）。
        C::B44 | C::B44A => b44_decompress(payload, ctx),
        other => Err(comp_err(
            LOC,
            format!("压缩类型 {} 不在本模块的解码分派表", other.code()),
            "PIZ/PXR24/B44/B44A 之外的压缩请走 exr.rs 的 None/Rle/Zip 路径",
        )),
    }
}

// ===========================================================================
//  二、PIZ —— 无损：LUT 范围压缩 + 2D 小波 + Huf 熵编码
// ===========================================================================

/// Huf 符号空间大小（`(1 << 16) + 1`：0..=65536，65536 是 RLC 伪符号）。
const PIZ_HUF_ENCSIZE: usize = (1 << 16) + 1;
/// PIZ bitmap 字节数（`USHORT_RANGE >> 3` = 8192）。
const PIZ_BITMAP_SIZE: usize = 1 << 13;
/// PIZ LUT / wavelet 的值域上界（u16 全域）。
const PIZ_USHORT_RANGE: usize = 1 << 16;

/// PIZ 解码。
///
/// 布局（严格照 OpenEXR `PizCompressor::uncompress`）：
/// `u16 minNonZero` `u16 maxNonZero` `bitmap[min..=max]` `i32 length` `Huf[length]`
fn piz_decompress(payload: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::piz";
    ctx.validate(W)?;
    let mut cur = ByteCursor::new(payload, W);

    // ---- ① 范围压缩头 ----
    let min_non_zero = cur.u16()?;
    let max_non_zero = cur.u16()?;
    if max_non_zero as usize >= PIZ_BITMAP_SIZE {
        return Err(comp_err(
            W,
            format!("maxNonZero = {max_non_zero} 超过 bitmap 大小 {PIZ_BITMAP_SIZE}"),
            "PIZ 块头损坏；确认 compression 标注为 4 且文件未被改写",
        ));
    }
    let mut bitmap = [0u8; PIZ_BITMAP_SIZE];
    if min_non_zero <= max_non_zero {
        let n = (max_non_zero - min_non_zero) as usize + 1;
        let seg = cur.take(n)?;
        bitmap[min_non_zero as usize..=max_non_zero as usize].copy_from_slice(seg);
    }

    // ---- ② 反向 LUT：序号 → 原始 u16 值 ----
    // 规则与 OpenEXR `reverseLutFromBitmap` 一致：i==0 恒映射（即使 bitmap[0] 的
    // bit0 被清零），其余按 bitmap 置位顺序编号；返回值 = LUT 最大下标 = maxValue。
    let mut lut = vec![0u16; PIZ_USHORT_RANGE];
    let mut k = 0usize;
    for i in 0..PIZ_USHORT_RANGE {
        if i == 0 || bitmap[i >> 3] & (1u8 << (i & 7)) != 0 {
            if let Some(slot) = lut.get_mut(k) {
                *slot = i as u16;
            }
            k += 1;
        }
    }
    // i==0 恒置位 ⇒ k ≥ 1。
    let max_value = (k - 1) as u16;

    // ---- ③ Huf 熵解码（**复用** `exr_dwa::huf_uncompress`）----
    // OpenEXR 的 `ImfHuf.cpp` 里 `hufUncompress` 是 DWA 与 PIZ **共用的同一个
    // 函数**（`PizCompressor` 直接调它），故此处不重写第二份解码器。
    let length = cur.i32()?;
    if length < 0 {
        return Err(comp_err(
            W,
            format!("Huf 段长度 {length} 为负"),
            "PIZ 块头损坏；确认文件完整（未被截断）",
        ));
    }
    let body = cur.take(length as usize)?;

    // ---- ④ 通道几何（临时缓冲按通道连续，每通道占 nx*ny*size 个 u16）----
    let nx = ctx.width;
    let ny = ctx.height;
    let geo = channel_geometry(ctx)?;
    let total_words = geo.iter().map(|g| g.words).sum();
    let mut words = crate::exr_dwa::huf_uncompress(body, total_words)?;

    // ---- ⑤ 逆小波（逐通道 × 逐 16bit 子平面）----
    for g in &geo {
        for j in 0..g.size {
            // 子平面 j 的切片取 `[start + j, start + words)`，与编码侧严格对逆。
            let base = g.start + j;
            let end = g
                .start
                .checked_add(g.words)
                .ok_or_else(|| comp_err(W, "小波平面下标溢出", "块尺寸异常"))?;
            // 先取 `words.len()`，避免在 `get_mut` 的错误闭包里二次不可变借用。
            let buf_len = words.len();
            let plane = words.get_mut(base..end).ok_or_else(|| {
                comp_err(
                    W,
                    format!("小波平面 [{base}..{end}) 越界（临时缓冲 {buf_len} 项）"),
                    "PIZ 块头与通道表不一致",
                )
            })?;
            wav2_decode(plane, nx, g.size, ny, nx * g.size, max_value)?;
        }
    }

    // ---- ⑥ 还原值域 ----
    for w in words.iter_mut() {
        *w = lut.get(*w as usize).copied().unwrap_or(0);
    }

    // ---- ⑦ 重排回「扫描线 × 通道 × 像素」并写出小端 ----
    // 临时缓冲是**通道连续**的：通道内按 `[y][x][j]` 排布，故行距 = `nx * size`。
    let mut out = Vec::with_capacity(ctx.raw_bytes());
    for y in 0..ny {
        for g in geo.iter() {
            let row = g.start + y * (nx * g.size);
            for i in 0..(nx * g.size) {
                let v = words.get(row + i).copied().unwrap_or(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// PIZ 编码（与 [`piz_decompress`] 严格对逆）。
///
/// 既用于测试对拍，也被写入器 [`crate::exr_write`] 调用（PIZ 写出路径）。
pub fn piz_compress(raw: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::piz_encode";
    ctx.validate(W)?;
    let geo = channel_geometry(ctx)?;
    let total_words: usize = geo.iter().map(|g| g.words).sum();

    // ---- ① 整块原始字节 → 通道连续的 u16 缓冲 ----
    // 目标布局：通道内按 `[y][x][j]` 排布（行距 = `nx * size`），
    // 与 `piz_decompress` 步骤 ⑦ 严格对逆。
    let mut words = vec![0u16; total_words];
    let nx = ctx.width;
    let ny = ctx.height;
    let mut at = 0usize;
    for y in 0..ny {
        for g in &geo {
            let row = g.start + y * (nx * g.size);
            for i in 0..(nx * g.size) {
                let b = raw
                    .get(at..at + 2)
                    .ok_or_else(|| comp_err(W, "原始块字节不足", "块尺寸与通道表不一致"))?;
                if let Some(slot) = words.get_mut(row + i) {
                    *slot = u16::from_le_bytes([b[0], b[1]]);
                }
                at += 2;
            }
        }
    }

    // ---- ② bitmapFromData ----
    let mut bitmap = [0u8; PIZ_BITMAP_SIZE];
    for &v in words.iter() {
        let idx = (v >> 3) as usize;
        if let Some(slot) = bitmap.get_mut(idx) {
            *slot |= 1u8 << (v & 7);
        }
    }
    if let Some(slot) = bitmap.get_mut(0) {
        *slot &= !1u8; // 0 不入 bitmap（约定：数据里总有 0）
    }
    let mut min_non_zero = PIZ_BITMAP_SIZE - 1;
    let mut max_non_zero = 0usize;
    for (i, &b) in bitmap.iter().enumerate() {
        if b != 0 {
            min_non_zero = min_non_zero.min(i);
            max_non_zero = max_non_zero.max(i);
        }
    }

    // ---- ③ 正向 LUT + 应用 ----
    let mut lut = vec![0u16; PIZ_USHORT_RANGE];
    let mut k = 0usize;
    for i in 0..PIZ_USHORT_RANGE {
        if i == 0 || bitmap[i >> 3] & (1u8 << (i & 7)) != 0 {
            if let Some(slot) = lut.get_mut(i) {
                *slot = (k.min(u16::MAX as usize)) as u16;
            }
            k += 1;
        }
    }
    let max_value = (k - 1) as u16;
    for w in words.iter_mut() {
        *w = lut.get(*w as usize).copied().unwrap_or(0);
    }

    // ---- ④ 写头 + bitmap 变长段 ----
    let mut out: Vec<u8> = Vec::with_capacity(raw.len() / 2 + 64);
    let mnz = (min_non_zero).min(u16::MAX as usize) as u16;
    let mxz = (max_non_zero).min(u16::MAX as usize) as u16;
    out.extend_from_slice(&mnz.to_le_bytes());
    out.extend_from_slice(&mxz.to_le_bytes());
    if mnz <= mxz {
        let seg = bitmap
            .get(mnz as usize..=mxz as usize)
            .ok_or_else(|| comp_err(W, "bitmap 段越界", "内部错误"))?;
        out.extend_from_slice(seg);
    }

    // ---- ⑤ 正小波 ----
    for g in &geo {
        for j in 0..g.size {
            // 子平面 j 的元素下标 = `start + y*(nx*size) + x*size + j`，
            // 故切片取 `[start + j, start + words)`（末元素 `start + words - 1`）。
            let base = g.start + j;
            let end = g
                .start
                .checked_add(g.words)
                .ok_or_else(|| comp_err(W, "小波平面下标溢出", "块尺寸异常"))?;
            let plane = words.get_mut(base..end).ok_or_else(|| {
                comp_err(W, "小波平面越界", "块尺寸与通道表不一致")
            })?;
            wav2_encode(plane, nx, g.size, ny, nx * g.size, max_value)?;
        }
    }

    // ---- ⑥ Huf 熵编码 ----
    let huf = huf_compress(&words)?;
    out.extend_from_slice(&(huf.len() as i32).to_le_bytes());
    out.extend_from_slice(&huf);
    Ok(out)
}

/// PIZ 临时缓冲里一个通道的几何。
#[derive(Debug, Clone, Copy)]
struct ChanGeo {
    /// 在 u16 缓冲中的起始下标。
    start: usize,
    /// 该通道占用的 u16 数（= `nx * ny * size`）。
    words: usize,
    /// 每像素的 16bit 子平面数（HALF = 1，FLOAT/UINT = 2）。
    size: usize,
}

/// 计算 PIZ 临时缓冲的逐通道几何（采样恒 1×1，故 `nx = width`、`ny = height`）。
fn channel_geometry(ctx: &BlockCtx<'_>) -> ExrResult<Vec<ChanGeo>> {
    let mut out = Vec::with_capacity(ctx.channels.len());
    let mut start = 0usize;
    for c in ctx.channels {
        if c.x_sampling != 1 || c.y_sampling != 1 {
            return Err(comp_err(
                LOC,
                format!("通道 `{}` 的采样为 {}×{}", c.name, c.x_sampling, c.y_sampling),
                "子采样通道的块内布局需重定义；请用 OIIO 展平为 1×1",
            ));
        }
        let size = c.pixel_type.byte_width() / 2;
        if size == 0 {
            return Err(comp_err(
                LOC,
                format!("通道 `{}` 的字节宽度非法", c.name),
                "像素类型损坏；确认 channels 属性",
            ));
        }
        let words = ctx
            .width
            .checked_mul(ctx.height)
            .and_then(|n| n.checked_mul(size))
            .ok_or_else(|| comp_err(LOC, "块像素数溢出", "使用更小的图像"))?;
        out.push(ChanGeo { start, words, size });
        start = start.checked_add(words).ok_or_else(|| {
            comp_err(LOC, "临时缓冲尺寸溢出", "使用更小的图像")
        })?;
    }
    Ok(out)
}

// ===========================================================================
//  三、2D Haar 小波（14bit / 16bit 两套 lifting，OpenEXR `ImfWav.cpp`）
// ===========================================================================

const WAVE_LOC: &str = "exr_compress::wavelet";

/// 小波平面越界读。
fn wave_rd(d: &[u16], i: usize) -> ExrResult<u16> {
    d.get(i).copied().ok_or_else(|| {
        comp_err(
            WAVE_LOC,
            format!("小波下标 {i} 越界（平面 {} 项）", d.len()),
            "块尺寸与通道表不一致；确认 dataWindow 与 channels 匹配",
        )
    })
}

/// 小波平面越界写。
fn wave_wr(d: &mut [u16], i: usize, v: u16) -> ExrResult<()> {
    if let Some(slot) = d.get_mut(i) {
        *slot = v;
        Ok(())
    } else {
        Err(comp_err(
            WAVE_LOC,
            format!("小波下标 {i} 越界（平面 {} 项）", d.len()),
            "块尺寸与通道表不一致；确认 dataWindow 与 channels 匹配",
        ))
    }
}

/// 14bit lifting 正变换（`wenc14`）：无模算术，压缩率更好。
///
/// **环绕语义**：C++ 参考用 `short` 做加减（溢出在实践中回绕），这里用
/// `wrapping_*` 显式固化，避免 Rust 的 debug 溢出 panic（C18）。
#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let x = a as i16 as i32;
    let y = b as i16 as i32;
    let ms = ((x + y) >> 1) as i16;
    let ds = x.wrapping_sub(y) as i16;
    (ms as u16, ds as u16)
}

/// 14bit lifting 逆变换（`wdec14`），`wenc14` 的精确逆。
#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let ls = l as i16 as i32;
    let hi = h as i16 as i32;
    let ai = ls.wrapping_add(hi & 1).wrapping_add(hi >> 1);
    (ai as i16 as u16, ai.wrapping_sub(hi) as i16 as u16)
}

/// 16bit 模运算 lifting 正变换（`wenc16`）：全 16bit 值域可用。
#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    const A_OFFSET: i32 = 1 << 15;
    const M_OFFSET: i32 = 1 << 15;
    const MOD_MASK: i32 = (1 << 16) - 1;
    let ao = (a as i32 + A_OFFSET) & MOD_MASK;
    let mut m = (ao + b as i32) >> 1;
    let d = ao - b as i32;
    if d < 0 {
        m = (m + M_OFFSET) & MOD_MASK;
    }
    (m as u16, (d & MOD_MASK) as u16)
}

/// 16bit 模运算 lifting 逆变换（`wdec16`），`wenc16` 的精确逆。
#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    const A_OFFSET: i32 = 1 << 15;
    const MOD_MASK: i32 = (1 << 16) - 1;
    let m = l as i32;
    let d = h as i32;
    let bb = (m - (d >> 1)) & MOD_MASK;
    let aa = (d + bb - A_OFFSET) & MOD_MASK;
    (aa as u16, bb as u16)
}

/// 2D 小波**逆**变换（`wav2Decode`）：就地变换 `in[y*oy + x*ox]`。
///
/// 层级按 `min(nx, ny)` 递减；`mx < 2^14` 走 14bit lifting，否则走 16bit 模运算。
/// 循环形状与 OpenEXR 逐行对应（`py`/`px` 为 do-while 退出位置，尾部
/// 「奇列 / 奇行」分支依赖它），因此写成 `loop { ... if 越界 break }` 而非 `for`。
pub fn wav2_decode(
    d: &mut [u16],
    nx: usize,
    ox: usize,
    ny: usize,
    oy: usize,
    mx: u16,
) -> ExrResult<()> {
    let w14 = mx < (1 << 14);
    if nx == 0 || ny == 0 {
        return Ok(());
    }
    let n = nx.min(ny);
    let mut p: usize = 1;
    while p <= n {
        p <<= 1;
    }
    p >>= 1;
    let mut p2 = p;
    p >>= 1;

    while p >= 1 {
        let ey = oy
            .checked_mul(ny - p2)
            .ok_or_else(|| comp_err(WAVE_LOC, "小波行界限溢出", "块尺寸异常"))?;
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        let mut py: usize = 0;
        loop {
            let ex = py
                .checked_add(ox * (nx - p2))
                .ok_or_else(|| comp_err(WAVE_LOC, "小波列界限溢出", "块尺寸异常"))?;
            let mut px = py;
            loop {
                let p01 = px + ox1;
                let p10 = px + oy1;
                let p11 = p10 + ox1;
                let (v00, v01, v10, v11) = (
                    wave_rd(d, px)?,
                    wave_rd(d, p01)?,
                    wave_rd(d, p10)?,
                    wave_rd(d, p11)?,
                );
                // 二维逆 lifting：先沿 x 拆 (i00,i10) / (i01,i11)，再沿 y 合并。
                let (out_px, out_p01, out_p10, out_p11) = if w14 {
                    let (i00, i10) = wdec14(v00, v10);
                    let (i01, i11) = wdec14(v01, v11);
                    let (a, b) = wdec14(i00, i01);
                    let (c, d2) = wdec14(i10, i11);
                    (a, b, c, d2)
                } else {
                    let (i00, i10) = wdec16(v00, v10);
                    let (i01, i11) = wdec16(v01, v11);
                    let (a, b) = wdec16(i00, i01);
                    let (c, d2) = wdec16(i10, i11);
                    (a, b, c, d2)
                };
                wave_wr(d, px, out_px)?;
                wave_wr(d, p01, out_p01)?;
                wave_wr(d, p10, out_p10)?;
                wave_wr(d, p11, out_p11)?;
                px += ox2;
                if px > ex {
                    break;
                }
            }
            // 奇列（nx 的 bit `log2(p)` 置位）
            if nx & p != 0 {
                let p10 = px + oy1;
                let (v, w) = (wave_rd(d, px)?, wave_rd(d, p10)?);
                let (a, b) = if w14 { wdec14(v, w) } else { wdec16(v, w) };
                wave_wr(d, p10, b)?;
                wave_wr(d, px, a)?;
            }
            py += oy2;
            if py > ey {
                break;
            }
        }
        // 奇行
        if ny & p != 0 {
            let ex = py
                .checked_add(ox * (nx - p2))
                .ok_or_else(|| comp_err(WAVE_LOC, "小波奇行界限溢出", "块尺寸异常"))?;
            let mut px = py;
            loop {
                let p01 = px + ox1;
                let (v, w) = (wave_rd(d, px)?, wave_rd(d, p01)?);
                let (a, b) = if w14 { wdec14(v, w) } else { wdec16(v, w) };
                wave_wr(d, p01, b)?;
                wave_wr(d, px, a)?;
                px += ox2;
                if px > ex {
                    break;
                }
            }
        }
        p2 = p;
        p >>= 1;
    }
    Ok(())
}

/// 2D 小波**正**变换（`wav2Encode`）：[`wav2_decode`] 的精确逆。
pub fn wav2_encode(
    d: &mut [u16],
    nx: usize,
    ox: usize,
    ny: usize,
    oy: usize,
    mx: u16,
) -> ExrResult<()> {
    let w14 = mx < (1 << 14);
    if nx == 0 || ny == 0 {
        return Ok(());
    }
    let n = nx.min(ny);
    let mut p: usize = 1;
    let mut p2: usize = 2;

    while p2 <= n {
        let ey = oy
            .checked_mul(ny - p2)
            .ok_or_else(|| comp_err(WAVE_LOC, "小波行界限溢出", "块尺寸异常"))?;
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        let mut py: usize = 0;
        loop {
            let ex = py
                .checked_add(ox * (nx - p2))
                .ok_or_else(|| comp_err(WAVE_LOC, "小波列界限溢出", "块尺寸异常"))?;
            let mut px = py;
            loop {
                let p01 = px + ox1;
                let p10 = px + oy1;
                let p11 = p10 + ox1;
                let (v00, v01, v10, v11) = (
                    wave_rd(d, px)?,
                    wave_rd(d, p01)?,
                    wave_rd(d, p10)?,
                    wave_rd(d, p11)?,
                );
                // 二维正 lifting：先沿 x 拆 (i00,i01) / (i10,i11)，再沿 y 合并。
                let (out_px, out_p01, out_p10, out_p11) = if w14 {
                    let (i00, i01) = wenc14(v00, v01);
                    let (i10, i11) = wenc14(v10, v11);
                    let (a, c) = wenc14(i00, i10);
                    let (b, d2) = wenc14(i01, i11);
                    (a, b, c, d2)
                } else {
                    let (i00, i01) = wenc16(v00, v01);
                    let (i10, i11) = wenc16(v10, v11);
                    let (a, c) = wenc16(i00, i10);
                    let (b, d2) = wenc16(i01, i11);
                    (a, b, c, d2)
                };
                wave_wr(d, px, out_px)?;
                wave_wr(d, p01, out_p01)?;
                wave_wr(d, p10, out_p10)?;
                wave_wr(d, p11, out_p11)?;
                px += ox2;
                if px > ex {
                    break;
                }
            }
            if nx & p != 0 {
                let p10 = px + oy1;
                let (v, w) = (wave_rd(d, px)?, wave_rd(d, p10)?);
                let (a, b) = if w14 { wenc14(v, w) } else { wenc16(v, w) };
                wave_wr(d, p10, b)?;
                wave_wr(d, px, a)?;
            }
            py += oy2;
            if py > ey {
                break;
            }
        }
        if ny & p != 0 {
            let ex = py
                .checked_add(ox * (nx - p2))
                .ok_or_else(|| comp_err(WAVE_LOC, "小波奇行界限溢出", "块尺寸异常"))?;
            let mut px = py;
            loop {
                let p01 = px + ox1;
                let (v, w) = (wave_rd(d, px)?, wave_rd(d, p01)?);
                let (a, b) = if w14 { wenc14(v, w) } else { wenc16(v, w) };
                wave_wr(d, p01, b)?;
                wave_wr(d, px, a)?;
                px += ox2;
                if px > ex {
                    break;
                }
            }
        }
        p = p2;
        p2 <<= 1;
    }
    Ok(())
}

// ===========================================================================
//  四、Huf 熵编码（PIZ 熵层；格式 = OpenEXR `ImfHuf.cpp`）
// ===========================================================================

/// 统一错误定位前缀（Huf **编码**子模块；解码侧在 `exr_dwa`）。
const HUF_LOC: &str = "exr_compress::huf_encode";
/// 码长表的零游程码（`SHORT_ZEROCODE_RUN`）。
const HUF_SHORT_ZEROCODE_RUN: u32 = 59;
/// 码长表的零游程码（`LONG_ZEROCODE_RUN`）。
const HUF_LONG_ZEROCODE_RUN: u32 = 63;
/// 长零游程的最短长度（`SHORTEST_LONG_RUN` = 2 + 63 - 59）。
const HUF_SHORTEST_LONG_RUN: i32 = 2 + 63 - 59;
/// 长零游程的最长长度（`LONGEST_LONG_RUN`）。
const HUF_LONGEST_LONG_RUN: i32 = 255 + HUF_SHORTEST_LONG_RUN;
/// Huf 码长上限（`hcode` 低 6 位只能装 0..=58）。
const HUF_MAX_CODE_LEN: u32 = 58;
// 解码侧的 `HUF_DECBITS` / `HUF_DECSIZE` 归 [`crate::exr_dwa::huf_uncompress`] 所有，
// 本模块只负责编码侧，故不再重复定义。

/// Huf 位写入器：**MSB 优先**（与 DEFLATE 的 LSB 相反，OpenEXR `outputBits`）。
///
/// 累积器只保留 `lc < 8` 位；单次写入的码长可高达 58 位，故按 ≤ 24 位分批左移，
/// 位流与一次性写入完全等价，但不会在 `u64` 上溢出。
struct HufWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl HufWriter {
    fn with_capacity(n: usize) -> Self {
        Self {
            out: Vec::with_capacity(n),
            acc: 0,
            nbits: 0,
        }
    }

    /// 写 `n`（1..=58）位整数。
    fn output_bits(&mut self, n: u32, bits: u64) {
        let mut left = n;
        let rest = bits;
        while left > 0 {
            let take = left.min(24);
            let hi = if take == 64 {
                rest
            } else {
                (rest >> (left - take)) & ((1u64 << take) - 1)
            };
            self.acc = (self.acc << take) | hi;
            self.nbits += take;
            while self.nbits >= 8 {
                self.nbits -= 8;
                let byte = ((self.acc >> self.nbits) & 0xFF) as u8;
                self.out.push(byte);
            }
            left -= take;
        }
    }

    /// 写一个 Huf 码字（`hcode` 打包格式：`[63:6] = 码值 | [5:0] = 码长`）。
    fn output_code(&mut self, code: u64) {
        self.output_bits((code & 63) as u32, code >> 6);
    }

    /// 收尾：返回 `(字节流, 有效比特数)`。末字节高位补 0。
    fn finish(mut self) -> (Vec<u8>, usize) {
        let n_bits = self.out.len() * 8 + self.nbits as usize;
        if self.nbits > 0 {
            let byte = ((self.acc << (8 - self.nbits)) & 0xFF) as u8;
            self.out.push(byte);
        }
        (self.out, n_bits)
    }
}

/// canonical Huf 码表（`hufCanonicalCodeTable`）：由**码长**推出码值。
///
/// 规则：短码（右侧补 0 后）数值大于长码；同码长下码值随符号号递增。
/// 入参 `lengths[i]` = 符号 i 的码长（0 = 不使用）；出参为 `len | (code << 6)`。
fn huf_canonical_code_table(lengths: &mut [u64]) {
    let mut n = [0u64; 59];
    for &l in lengths.iter() {
        let li = l as usize;
        if li < 59 {
            n[li] += 1;
        }
    }
    let mut c = 0u64;
    for i in (1..59usize).rev() {
        let nc = (c + n[i]) >> 1;
        n[i] = c;
        c = nc;
    }
    for slot in lengths.iter_mut() {
        let l = *slot;
        if l > 0 && (l as usize) < 59 {
            *slot = l | (n[l as usize] << 6);
            n[l as usize] += 1;
        }
    }
}

/// 由频次定序地构造 Huf 码长（等价 OpenEXR `hufBuildEncTable` 的树构建部分）。
///
/// **确定性**：最小堆键为 `(频次, 节点下标)`，同频按下标定序 —— 不用 `HashMap`、
/// 不用指针序（OpenEXR 用 `make_heap` + 指针比较），给出等价但平台无关的定序。
///
/// 节点编号：叶子 `0..leaves.len()`（**按 `leaves` 下标**，不是符号号），
/// 内部节点 `leaves.len()..`；符号号另存于 `leaves`，避免两者混淆。
fn huf_code_lengths(freq: &[u64]) -> ExrResult<Vec<u64>> {
    let leaves: Vec<usize> = freq
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .map(|(i, _)| i)
        .collect();
    if leaves.len() < 2 {
        return Err(comp_err(
            HUF_LOC,
            format!("Huf 有效符号仅 {} 个（至少需 2：数据符号 + RLC 伪符号）", leaves.len()),
            "编码输入退化；确认块内像素数据非空且已 LUT 化",
        ));
    }
    let nl = leaves.len();
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(nl);
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = leaves
        .iter()
        .enumerate()
        .map(|(k, &s)| Reverse((freq[s], k)))
        .collect();
    while heap.len() > 1 {
        let (fa, a) = pop_min(&mut heap)?;
        let (fb, b) = pop_min(&mut heap)?;
        let id = nl + children.len();
        children.push((a, b));
        heap.push(Reverse((fa.saturating_add(fb), id)));
    }
    let root = match heap.pop() {
        Some(Reverse((_, id))) => id,
        None => return Err(comp_err(HUF_LOC, "Huf 堆意外为空", "内部错误")),
    };

    // 迭代 DFS 求深度（树深可达数千，显式栈避免栈溢出）。
    let total = nl + children.len();
    let mut depth = vec![0u32; total];
    let mut stack: Vec<(usize, u32)> = vec![(root, 0)];
    while let Some((id, d)) = stack.pop() {
        if let Some(slot) = depth.get_mut(id) {
            *slot = d;
        }
        if id >= nl {
            if let Some(&(l, r)) = children.get(id - nl) {
                stack.push((l, d.saturating_add(1)));
                stack.push((r, d.saturating_add(1)));
            }
        }
    }
    let mut lengths = vec![0u64; freq.len()];
    let mut max_len = 0u64;
    for (k, &sym) in leaves.iter().enumerate() {
        let d = u64::from(depth.get(k).copied().unwrap_or(0));
        max_len = max_len.max(d);
        if let Some(slot) = lengths.get_mut(sym) {
            *slot = d;
        }
    }
    if max_len == 0 || max_len > u64::from(HUF_MAX_CODE_LEN) {
        return Err(comp_err(
            HUF_LOC,
            format!("Huf 码长 {max_len} 非法（须 1..={HUF_MAX_CODE_LEN}）"),
            "该块符号分布过于极端；请改用 ZIP 压缩写出",
        ));
    }
    Ok(lengths)
}

/// 从最小堆弹出 `(频次, 节点)`（`Reverse` ⇒ 堆顶最小）。
fn pop_min(heap: &mut BinaryHeap<Reverse<(u64, usize)>>) -> ExrResult<(u64, usize)> {
    match heap.pop() {
        Some(Reverse(v)) => Ok(v),
        None => Err(comp_err(HUF_LOC, "Huf 堆意外为空", "内部错误")),
    }
}

/// 打包 Huf 码长表（`hufPackEncTable`）：6 位码长，零游程用 59..=63。
fn huf_pack_enc_table(hcode: &[u64], im: usize, i_m: usize, w: &mut HufWriter) {
    let mut i = im;
    while i <= i_m {
        let l = hcode.get(i).copied().unwrap_or(0) & 63;
        if l == 0 {
            let mut zerun = 1usize;
            while i < i_m && zerun < HUF_LONGEST_LONG_RUN as usize {
                let nxt = hcode.get(i + 1).copied().unwrap_or(0) & 63;
                if nxt > 0 {
                    break;
                }
                i += 1;
                zerun += 1;
            }
            if zerun >= 2 {
                if zerun >= HUF_SHORTEST_LONG_RUN as usize {
                    w.output_bits(6, HUF_LONG_ZEROCODE_RUN as u64);
                    w.output_bits(8, (zerun - HUF_SHORTEST_LONG_RUN as usize) as u64);
                } else {
                    w.output_bits(
                        6,
                        (HUF_SHORT_ZEROCODE_RUN as usize + zerun - 2) as u64,
                    );
                }
                i += 1;
                continue;
            }
        }
        w.output_bits(6, l);
        i += 1;
    }
}

/// 发一个符号（`sendCode`）：游程比逐个发更短时用 RLC 伪符号 + 8 位计数。
fn huf_send_code(w: &mut HufWriter, s_code: u64, run_count: i32, run_code: u64) {
    let ls = (s_code & 63) as i64;
    let lr = (run_code & 63) as i64;
    if ls + lr + 8 < ls * run_count as i64 {
        w.output_code(s_code);
        w.output_code(run_code);
        w.output_bits(8, run_count as u64);
    } else {
        for _ in 0..=run_count {
            w.output_code(s_code);
        }
    }
}

/// Huf 熵编码数据段（`hufEncode`）：返回 `(字节流, 比特数)`。
fn huf_encode(hcode: &[u64], raw: &[u16], rlc: usize) -> (Vec<u8>, usize) {
    let mut w = HufWriter::with_capacity(raw.len() + raw.len() / 8 + 16);
    let first = raw.first().copied().unwrap_or(0);
    let mut s = first;
    let mut cs: i32 = 0;
    for &v in raw.iter().skip(1) {
        if s == v && cs < 255 {
            cs += 1;
        } else {
            huf_send_code(
                &mut w,
                hcode.get(s as usize).copied().unwrap_or(0),
                cs,
                hcode.get(rlc).copied().unwrap_or(0),
            );
            cs = 0;
        }
        s = v;
    }
    huf_send_code(
        &mut w,
        hcode.get(s as usize).copied().unwrap_or(0),
        cs,
        hcode.get(rlc).copied().unwrap_or(0),
    );
    w.finish()
}

/// Huf 压缩（`hufCompress`）：20 字节头 + 码长表 + 数据段。
pub fn huf_compress(raw: &[u16]) -> ExrResult<Vec<u8>> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut freq = vec![0u64; PIZ_HUF_ENCSIZE];
    for &v in raw {
        if let Some(slot) = freq.get_mut(v as usize) {
            *slot += 1;
        }
    }
    // im = 最小非零频次下标；iM = 最大非零下标 + 1（该位是 RLC 伪符号，频次 1）。
    let mut im = 0usize;
    while im < PIZ_HUF_ENCSIZE && freq[im] == 0 {
        im += 1;
    }
    if im >= PIZ_HUF_ENCSIZE {
        return Err(comp_err(
            HUF_LOC,
            "Huf 频次表全零",
            "编码输入为空；确认块内像素数据非空",
        ));
    }
    let mut i_m = im;
    for (i, &f) in freq.iter().enumerate().skip(im) {
        if f != 0 {
            i_m = i;
        }
    }
    i_m += 1;
    if i_m >= PIZ_HUF_ENCSIZE {
        return Err(comp_err(
            HUF_LOC,
            format!("Huf RLC 伪符号下标 {i_m} 超出 {PIZ_HUF_ENCSIZE}"),
            "数据值域异常；确认块内像素已 LUT 化到 0..=65535",
        ));
    }
    if let Some(slot) = freq.get_mut(i_m) {
        *slot = 1;
    }

    let mut hcode = huf_code_lengths(&freq)?;
    huf_canonical_code_table(&mut hcode);

    let mut tw = HufWriter::with_capacity(64);
    huf_pack_enc_table(&hcode, im, i_m, &mut tw);
    let (table, _) = tw.finish();
    let (data, n_bits) = huf_encode(&hcode, raw, i_m);

    let mut out = Vec::with_capacity(20 + table.len() + data.len());
    out.extend_from_slice(&(im as u32).to_le_bytes());
    out.extend_from_slice(&(i_m as u32).to_le_bytes());
    out.extend_from_slice(&(table.len() as u32).to_le_bytes());
    out.extend_from_slice(&(n_bits as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // 预留扩展字段
    out.extend_from_slice(&table);
    out.extend_from_slice(&data);
    Ok(out)
}

// ===========================================================================
//  五、PXR24 —— 24bit 定点 + 字节平面转置 + zlib
// ===========================================================================

/// PXR24 每像素的**中间平面数**：UINT = 4，HALF = 2，FLOAT = 3（24bit）。
fn pxr24_planes(pt: PixelType) -> usize {
    match pt {
        PixelType::Uint => 4,
        PixelType::Half => 2,
        PixelType::Float => 3,
    }
}

/// PXR24 解码：zlib → 逐通道差分还原 → 「扫描线 × 通道 × 像素」小端字节。
fn pxr24_decompress(payload: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::pxr24";
    ctx.validate(W)?;
    let tmp_bytes: usize = ctx
        .channels
        .iter()
        .map(|c| pxr24_planes(c.pixel_type) * ctx.width)
        .sum::<usize>()
        .saturating_mul(ctx.height);
    let tmp = crate::exr::zlib_decompress(payload)?;
    if tmp.len() != tmp_bytes {
        return Err(comp_err(
            W,
            format!(
                "PXR24 zlib 解出 {} 字节，本块应有 {tmp_bytes} 字节（宽 {} × {} 行 × 通道平面数）",
                tmp.len(),
                ctx.width,
                ctx.height
            ),
            "PXR24 负载与通道表/块尺寸不一致；确认文件完整",
        ));
    }

    let mut out = Vec::with_capacity(ctx.raw_bytes());
    let mut base = 0usize;
    for _y in 0..ctx.height {
        for c in ctx.channels {
            let n = ctx.width;
            let planes = pxr24_planes(c.pixel_type);
            let mut pixel: u32 = 0;
            for x in 0..n {
                // 平面是**转置**存放的：第 `p` 个平面的第 `x` 个字节在
                // `base + p * n + x`（与 `pxr24_compress` 的写入严格对逆）。
                let plane_byte = |p: usize| -> u32 {
                    u32::from(tmp.get(base + p * n + x).copied().unwrap_or(0))
                };
                // 平面合成：UINT 4 字节、HALF 2 字节、FLOAT 3 字节左移到高 24/16 位。
                let diff: u32 = match c.pixel_type {
                    PixelType::Uint => {
                        (plane_byte(0) << 24) | (plane_byte(1) << 16) | (plane_byte(2) << 8) | plane_byte(3)
                    }
                    PixelType::Half => (plane_byte(0) << 8) | plane_byte(1),
                    PixelType::Float => (plane_byte(0) << 24) | (plane_byte(1) << 16) | (plane_byte(2) << 8),
                };
                pixel = pixel.wrapping_add(diff);
                for k in 0..c.pixel_type.byte_width() {
                    out.push(((pixel >> (8 * k)) & 0xFF) as u8);
                }
            }
            base += planes * n;
        }
    }
    Ok(out)
}

/// PXR24 编码（与 [`pxr24_decompress`] 严格对逆）。
///
/// 既用于测试对拍，也被写入器 [`crate::exr_write`] 调用（PXR24 写出路径）。
pub fn pxr24_compress(raw: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::pxr24_encode";
    ctx.validate(W)?;
    let mut tmp: Vec<u8> = Vec::with_capacity(raw.len());
    let mut at = 0usize;
    for _y in 0..ctx.height {
        for c in ctx.channels {
            let bw = c.pixel_type.byte_width();
            let planes = pxr24_planes(c.pixel_type);
            let mut prev: u32 = 0;
            // 平面暂存：本扫描线 × 通道的 planes 组。
            let base = tmp.len();
            tmp.resize(base + planes * ctx.width, 0);
            for x in 0..ctx.width {
                let mut v: u32 = 0;
                for k in 0..bw {
                    v |= (raw.get(at + k).copied().unwrap_or(0) as u32) << (8 * k);
                }
                at += bw;
                let q = match c.pixel_type {
                    PixelType::Float => float_to_float24(f32::from_bits(v)),
                    _ => v,
                };
                let diff = q.wrapping_sub(prev);
                prev = q;
                for p in 0..planes {
                    let shift = 8 * (planes - 1 - p);
                    if let Some(slot) = tmp.get_mut(base + p * ctx.width + x) {
                        *slot = ((diff >> shift) & 0xFF) as u8;
                    }
                }
            }
        }
    }
    Ok(crate::exr_write::deflate_zlib(&tmp))
}

/// f32 → 24bit 定点（`floatToFloat24`，Pixar 贡献）。
///
/// 保留符号与指数，尾数四舍五入到 15 位；NaN 保符号 + 尾数高 15 位（至少 1 位），
/// ±inf 只留符号 + 指数；接近 `FLT_MAX` 时截断尾数以避免指数上溢。
fn float_to_float24(f: f32) -> u32 {
    let i = f.to_bits();
    let s = i & 0x8000_0000;
    let e = i & 0x7f80_0000;
    let m = i & 0x007f_ffff;
    if e == 0x7f80_0000 {
        if m != 0 {
            let m24 = m >> 8;
            (s >> 8) | (e >> 8) | m24 | u32::from(m24 == 0)
        } else {
            (s >> 8) | (e >> 8)
        }
    } else {
        let mut v = ((e | m) + (m & 0x0000_0080)) >> 8;
        if v >= 0x7f8_000 {
            v = (e | m) >> 8; // 舍入导致指数上溢 → 改为截断
        }
        (s >> 8) | v
    }
}

// ===========================================================================
//  六、B44 / B44A —— 4×4 块 packing（HALF 有损，UINT/FLOAT 原样）
// ===========================================================================

/// B44/B44A 的 HALF 块打包（14 字节 / 平场 3 字节）。
fn b44_decompress(payload: &[u8], ctx: &BlockCtx<'_>) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::b44";
    ctx.validate(W)?;
    let nx = ctx.width;
    let ny = ctx.height;
    let mut cur = ByteCursor::new(payload, W);
    let mut words: Vec<u16> = Vec::with_capacity(ctx.raw_bytes() / 2);

    for c in ctx.channels {
        if c.x_sampling != 1 || c.y_sampling != 1 {
            return Err(comp_err(
                W,
                format!("通道 `{}` 的采样为 {}×{}", c.name, c.x_sampling, c.y_sampling),
                "子采样通道的 B44 块布局未定义；请用 OIIO 展平为 1×1",
            ));
        }
        let size = c.pixel_type.byte_width() / 2;
        let plane_words = nx
            .checked_mul(ny)
            .and_then(|n| n.checked_mul(size))
            .ok_or_else(|| comp_err(W, "B44 平面尺寸溢出", "使用更小的图像"))?;
        let base = words.len();
        words.resize(base + plane_words, 0);

        if c.pixel_type != PixelType::Half {
            // UINT / FLOAT：**原样复制**（B44 只压 HALF）。
            let bytes = cur.take(plane_words * 2)?;
            for (i, pair) in bytes.chunks_exact(2).enumerate() {
                if let Some(slot) = words.get_mut(base + i) {
                    *slot = u16::from_le_bytes([pair[0], pair[1]]);
                }
            }
            continue;
        }

        if c.p_linear {
            return Err(p_linear_unsupported(W, &c.name));
        }
        // HALF：4×4 块。
        for by in (0..ny).step_by(4) {
            for bx in (0..nx).step_by(4) {
                let mut s = [0u16; 16];
                let peek = *cur
                    .data
                    .get(cur.pos + 2)
                    .ok_or_else(|| {
                        comp_err(
                            W,
                            format!("B44 块在偏移 {} 处不足 3 字节", cur.pos),
                            "B44 块被截断；确认文件完整",
                        )
                    })?;
                // shift 字段（b[2] >> 2）≥ 13 即为「不可能的移位值」⇒ 3 字节平场。
                if peek >= (13 << 2) {
                    let b = cur.take(3)?;
                    b44_unpack3(b, &mut s);
                } else {
                    let b = cur.take(14)?;
                    b44_unpack14(b, &mut s);
                }
                // `pLinear` 通道：解出的仍是**非线性**值，须过 `logTable` 还原线性。
                // 写回：只写块内真实存在的行/列（边缘块不写越界部分）。
                let cols = (nx - bx).min(4);
                let rows = (ny - by).min(4);
                for j in 0..rows {
                    for i in 0..cols {
                        let idx = (by + j) * nx + (bx + i);
                        if let Some(slot) = words.get_mut(base + idx) {
                            *slot = s[j * 4 + i];
                        }
                    }
                }
            }
        }
    }
    if cur.remaining() > 0 {
        return Err(comp_err(
            W,
            format!("B44 块解完仍有 {} 字节未消费", cur.remaining()),
            "B44 负载与通道表/块尺寸不一致；确认文件完整",
        ));
    }

    // 重排回「扫描线 × 通道 × 像素」。
    // `words` 的布局是「通道连续、通道内扫描线连续」：第 `ci` 个通道的
    // (y, x) 元素在 `chan_base[ci] + (y * nx + x) * size`。
    let mut chan_base: Vec<usize> = Vec::with_capacity(ctx.channels.len());
    {
        let mut acc = 0usize;
        for c in ctx.channels {
            chan_base.push(acc);
            acc += nx * ny * (c.pixel_type.byte_width() / 2);
        }
    }
    let mut out = Vec::with_capacity(ctx.raw_bytes());
    for y in 0..ny {
        for (ci, c) in ctx.channels.iter().enumerate() {
            let size = c.pixel_type.byte_width() / 2;
            let base = chan_base.get(ci).copied().unwrap_or(0);
            for x in 0..nx {
                for k in 0..size {
                    let v = words.get(base + (y * nx + x) * size + k).copied().unwrap_or(0);
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    Ok(out)
}

/// B44/B44A 编码（与 [`b44_decompress`] 对逆；`opt_flat` = B44A）。
///
/// 既用于测试对拍，也被写入器 [`crate::exr_write`] 调用（B44/B44A 写出路径）。
pub fn b44_compress(raw: &[u8], ctx: &BlockCtx<'_>, opt_flat: bool) -> ExrResult<Vec<u8>> {
    const W: &str = "exr_compress::b44_encode";
    ctx.validate(W)?;
    let nx = ctx.width;
    let ny = ctx.height;
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    // 先把原始块（扫描线 × 通道 × 像素）拆成「通道连续、通道内扫描线连续」的
    // u16 平面（与 OpenEXR `_tmpBuffer` 的布局一致）。
    let per_line: usize = ctx
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width() * nx)
        .sum();
    let mut planes: Vec<(Vec<u16>, usize)> = Vec::with_capacity(ctx.channels.len());
    for (ci, c) in ctx.channels.iter().enumerate() {
        let size = c.pixel_type.byte_width() / 2;
        let n = nx * ny * size;
        let mut p = vec![0u16; n];
        let c_off = ctx
            .channels
            .iter()
            .take(ci)
            .map(|cc| cc.pixel_type.byte_width() * nx)
            .sum::<usize>();
        for y in 0..ny {
            for x in 0..nx {
                let byte_off = y * per_line + c_off + x * c.pixel_type.byte_width();
                for k in 0..size {
                    let at = byte_off + k * 2;
                    let b = raw
                        .get(at..at + 2)
                        .ok_or_else(|| comp_err(W, "原始块字节不足", "块尺寸与通道表不一致"))?;
                    p[y * nx * size + x * size + k] = u16::from_le_bytes([b[0], b[1]]);
                }
            }
        }
        planes.push((p, size));
    }

    for (ci, c) in ctx.channels.iter().enumerate() {
        let (plane, size) = planes.get(ci).cloned().unwrap_or_default();
        if c.pixel_type != PixelType::Half {
            // UINT / FLOAT：原样复制。
            for v in plane.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            continue;
        }
        if c.p_linear {
            return Err(p_linear_unsupported(W, &c.name));
        }
        for by in (0..ny).step_by(4) {
            for bx in (0..nx).step_by(4) {
                let mut s = [0u16; 16];
                // 边缘块：重复最右列 / 最下行补齐到 4×4（OpenEXR `compress` 同策略）。
                for j in 0..4 {
                    let sy = (by + j).min(ny - 1);
                    for i in 0..4 {
                        let sx = (bx + i).min(nx - 1);
                        s[j * 4 + i] = plane.get(sy * nx + sx).copied().unwrap_or(0);
                    }
                }
                // `exactMax`：令块内最大值被精确表示（`pLinear` 已在上游被拒）。
                let packed = b44_pack(&s, opt_flat, true);
                out.extend_from_slice(&packed);
            }
        }
        let _ = size;
    }
    Ok(out)
}

/// B44 块解包（14 字节形式，`unpack14`）。
fn b44_unpack14(b: &[u8], s: &mut [u16; 16]) {
    // `sh` 来自任意负载字节，理论上可达 63；直接 `<<` 会触发 debug 溢出 panic。
    // OpenEXR 参考实现写的是 `int sh = in[2] >> 2; int bias = 0x20 << sh;`，
    // 在 x86-64 上 `shl` 会把计数按 2^5 取模，故这里显式 `& 31` 以复刻实际行为
    // 并保证不 panic。正常负载由调用点的 `>= (13 << 2)` 判据把 `sh` 限死在 0..=12。
    let sh = (b.get(2).copied().unwrap_or(0) >> 2) as u32 & 0x1F;
    let bias = (0x20u32 << sh) as u16;
    let at = |i: usize| -> u32 { b.get(i).copied().unwrap_or(0) as u32 };
    // 参考实现全程用 `int`：先取 6 bit 差分再左移 sh，故这里也保持 u16 运算。
    let d6 = |v: u32| -> u16 { ((v & 0x3F) << sh) as u16 };
    s[0] = ((at(0) << 8) | at(1)) as u16;
    // 参考实现是 16 条顺序赋值语句。闭包捕获 `s` 会与后续 `s[i]` 读取冲突，
    // 故先把 15 个派生值全算出来，再一次性写回。
    let v0 = s[0];
    let v4 = v0.wrapping_add(d6((at(2) << 4) | (at(3) >> 4))).wrapping_sub(bias);
    let v8 = v4.wrapping_add(d6((at(3) << 2) | (at(4) >> 6))).wrapping_sub(bias);
    let v12 = v8.wrapping_add(d6(at(4))).wrapping_sub(bias);
    let v1 = v0.wrapping_add(d6(at(5) >> 2)).wrapping_sub(bias);
    let v5 = v4.wrapping_add(d6((at(5) << 4) | (at(6) >> 4))).wrapping_sub(bias);
    let v9 = v8.wrapping_add(d6((at(6) << 2) | (at(7) >> 6))).wrapping_sub(bias);
    let v13 = v12.wrapping_add(d6(at(7))).wrapping_sub(bias);
    let v2 = v1.wrapping_add(d6(at(8) >> 2)).wrapping_sub(bias);
    let v6 = v5.wrapping_add(d6((at(8) << 4) | (at(9) >> 4))).wrapping_sub(bias);
    let v10 = v9.wrapping_add(d6((at(9) << 2) | (at(10) >> 6))).wrapping_sub(bias);
    let v14 = v13.wrapping_add(d6(at(10))).wrapping_sub(bias);
    let v3 = v2.wrapping_add(d6(at(11) >> 2)).wrapping_sub(bias);
    let v7 = v6.wrapping_add(d6((at(11) << 4) | (at(12) >> 4))).wrapping_sub(bias);
    let v11 = v10.wrapping_add(d6((at(12) << 2) | (at(13) >> 6))).wrapping_sub(bias);
    let v15 = v14.wrapping_add(d6(at(13))).wrapping_sub(bias);
    s[1] = v1;
    s[2] = v2;
    s[3] = v3;
    s[4] = v4;
    s[5] = v5;
    s[6] = v6;
    s[7] = v7;
    s[8] = v8;
    s[9] = v9;
    s[10] = v10;
    s[11] = v11;
    s[12] = v12;
    s[13] = v13;
    s[14] = v14;
    s[15] = v15;
    // 反查表：从 t 域的保序映射还原 half 位模式。
    for v in s.iter_mut() {
        *v = if *v & 0x8000 != 0 { *v & 0x7FFF } else { !*v };
    }
}

/// B44 块解包（3 字节平场形式，`unpack3`）。
fn b44_unpack3(b: &[u8], s: &mut [u16; 16]) {
    let hi = b.first().copied().unwrap_or(0) as u16;
    let lo = b.get(1).copied().unwrap_or(0) as u16;
    let v = (hi << 8) | lo;
    let v = if v & 0x8000 != 0 { v & 0x7FFF } else { !v };
    for slot in s.iter_mut() {
        *slot = v;
    }
}

/// `pLinear == true` 的 HALF 通道所需的 exp/log 表缺失时的**结构化拒绝**。
///
/// OpenEXR 的 `b44ExpLogTable.h` 是两张各 128 KB 的**离线生成常量表**，本仓库
/// 不内嵌它们，也不按公式重算：重算结果在饱和区与非有限输入区无法与参考表
/// 逐位对齐（实测 `expTable` 失配 8819/65536 项、`logTable` 失配 33791/65536
/// 项，如索引 `0x618c`：参考表给 `0x7c00`=+inf，公式重算给 `0x7bff`），属
/// "看似支持、实则位级不可保"。按裁定只对 `pLinear == false` 提供完整实现。
fn p_linear_unsupported(w: &'static str, channel: &str) -> crate::exr::ExrError {
    comp_err(
        w,
        format!("B44/B44A 的 HALF 通道 `{channel}` 带有 pLinear=true 标志"),
        "该路径需要 OpenEXR 的 expTable/logTable（两张 128 KB 离线常量表），本解码器不内嵌也不按公式重算（公式在饱和区与参考表有 8819/33791 项失配）；请在写入侧去掉 pLinear 标志（OpenEXR 默认不写）后再读",
    )
}

/// 4×4 HALF 块打包（`pack`）：返回 14 字节或 3 字节。
fn b44_pack(s: &[u16; 16], opt_flat: bool, exact_max: bool) -> Vec<u8> {
    // ① 保序映射到 t 域；NaN/inf → 0x8000（浮点零）。
    let mut t = [0u16; 16];
    for (i, slot) in t.iter_mut().enumerate() {
        let v = s.get(i).copied().unwrap_or(0);
        *slot = if v & 0x7C00 == 0x7C00 {
            0x8000
        } else if v & 0x8000 != 0 {
            !v
        } else {
            v | 0x8000
        };
    }
    let mut t_max = 0u16;
    for v in t.iter() {
        if t_max < *v {
            t_max = *v;
        }
    }
    // ② 找一个移位量，使全部差分落在 [-32, 31]。
    const BIAS: i32 = 0x20;
    let mut shift: i32 = -1;
    let mut d = [0i32; 16];
    let mut r = [0i32; 15];
    let mut r_min = 0i32;
    let mut r_max = 0i32;
    loop {
        shift += 1;
        if shift > 24 {
            break; // 防御：正常 16bit 输入在 shift ≤ 11 即收敛
        }
        for i in 0..16 {
            d[i] = b44_shift_and_round(t_max as i32 - t[i] as i32, shift as u32);
        }
        r[0] = d[0] - d[4] + BIAS;
        r[1] = d[4] - d[8] + BIAS;
        r[2] = d[8] - d[12] + BIAS;
        r[3] = d[0] - d[1] + BIAS;
        r[4] = d[4] - d[5] + BIAS;
        r[5] = d[8] - d[9] + BIAS;
        r[6] = d[12] - d[13] + BIAS;
        r[7] = d[1] - d[2] + BIAS;
        r[8] = d[5] - d[6] + BIAS;
        r[9] = d[9] - d[10] + BIAS;
        r[10] = d[13] - d[14] + BIAS;
        r[11] = d[2] - d[3] + BIAS;
        r[12] = d[6] - d[7] + BIAS;
        r[13] = d[10] - d[11] + BIAS;
        r[14] = d[14] - d[15] + BIAS;
        r_min = r[0];
        r_max = r[0];
        for v in r.iter().skip(1) {
            if r_min > *v {
                r_min = *v;
            }
            if r_max < *v {
                r_max = *v;
            }
        }
        if r_min >= 0 && r_max <= 0x3F {
            break;
        }
    }
    // ③ 平场特例（B44A 允许；B44 不允许）。
    if r_min == BIAS && r_max == BIAS && opt_flat {
        return vec![(t[0] >> 8) as u8, t[0] as u8, 0xFC];
    }
    // ④ 让等于 tMax 的那个像素被精确表示。
    if exact_max {
        let d0 = d[0];
        t[0] = (t_max as i32 - (d0 << shift.max(0))) as u16;
    }
    // ⑤ 打包 14 字节。
    let sh = shift as u32;
    let mut b = [0u8; 14];
    b[0] = (t[0] >> 8) as u8;
    b[1] = t[0] as u8;
    b[2] = (((sh << 2) | ((r[0] >> 4) as u32 & 0x3)) & 0xFF) as u8;
    b[3] = (((r[0] << 4) | (r[1] >> 2)) & 0xFF) as u8;
    b[4] = (((r[1] << 6) | r[2]) & 0xFF) as u8;
    b[5] = (((r[3] << 2) | (r[4] >> 4)) & 0xFF) as u8;
    b[6] = (((r[4] << 4) | (r[5] >> 2)) & 0xFF) as u8;
    b[7] = (((r[5] << 6) | r[6]) & 0xFF) as u8;
    b[8] = (((r[7] << 2) | (r[8] >> 4)) & 0xFF) as u8;
    b[9] = (((r[8] << 4) | (r[9] >> 2)) & 0xFF) as u8;
    b[10] = (((r[9] << 6) | r[10]) & 0xFF) as u8;
    b[11] = (((r[11] << 2) | (r[12] >> 4)) & 0xFF) as u8;
    b[12] = (((r[12] << 4) | (r[13] >> 2)) & 0xFF) as u8;
    b[13] = (((r[13] << 6) | r[14]) & 0xFF) as u8;
    b.to_vec()
}

/// `y = round_half_even(x * 2^-shift)`（`shiftAndRound`）。
fn b44_shift_and_round(x: i32, shift: u32) -> i32 {
    let x = x << 1;
    let a = (1i32 << shift) - 1;
    let sh = shift + 1;
    let b = (x >> sh) & 1;
    (x + a + b) >> sh
}

#[cfg(test)]
mod tests;
