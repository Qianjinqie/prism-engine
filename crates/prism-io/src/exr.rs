//! OpenEXR 容器读入（U7）——分块浮点容器 + 多通道 AOV + Z/Float deep。
//!
//! 语义权威：《强大版路线图》§4 U7 行（相机 RAW / EXR 深度）与账本「EXR =
//! 分块浮点容器（tile 已有）」。
//!
//! # 铁律
//!
//! - **零第三方依赖**：容器结构、`inflate`（RFC 1950/1951）、RLE codec、
//!   `f16` 位转换全部手写，只用 `std`；
//! - **确定性**：通道遍历顺序 = 文件中 `channels` 属性的**出现顺序**
//!   （不排序、不用 `HashMap`），块/行/像素遍历均固定升序；
//! - **C18 永不 panic**：所有长度/偏移都经 `checked_*` 与 `get()`，
//!   畸形输入（错 magic、截断、超大属性 size、越界偏移）返回三段式 [`ExrError`]。
//!
//! # 复用说明（inflate 已下沉）
//!
//! DEFLATE / zlib 解码实现**已下沉到 L0 叶子 crate [`prism_bytes`]**，本模块
//! 只保留 [`inflate_raw`] / [`zlib_decompress`] 两个薄转发（为了调用点零改动）。
//!
//! 此前 `prism-media` 另有一份自研 inflate，注释写「因循环依赖无法复用」。
//! 下沉到所有消费者之下后，`prism-io`(L4) 与 `prism-media`(L4) 都向下依赖
//! `prism-bytes`(L0)，**循环依赖自然消失**，双份实现收敛为一份。

use std::fmt;

use crate::exr_compress::BlockCtx;
use crate::fbuf::{f16_to_f32, Fbuf16F};

// ===========================================================================
//  一、错误
// ===========================================================================

/// EXR 域错误码分段：`EXR1xxx` 容器结构、`EXR2xxx` 压缩、`EXR3xxx` 通道/deep。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl ExrError {
    /// 构造三段式错误。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }
}

impl fmt::Display for ExrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] @{}：{}（修复：{}）",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for ExrError {}

/// EXR 域结果别名。
pub type ExrResult<T> = Result<T, ExrError>;

/// 容器结构错误（`EXR1xxx`）。
pub(crate) fn struct_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> ExrError {
    ExrError::new("EXR1001", loc, problem, fix)
}

/// 压缩错误（`EXR2xxx`）。
pub(crate) fn comp_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> ExrError {
    ExrError::new("EXR2001", loc, problem, fix)
}

/// 通道 / deep 错误（`EXR3xxx`）。
pub(crate) fn chan_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> ExrError {
    ExrError::new("EXR3001", loc, problem, fix)
}

// ===========================================================================
//  二、常量
// ===========================================================================

/// EXR magic（`0x76 0x2f 0x31 0x01` = 20000630，小端读作 `0x01312F76`）。
pub const EXR_MAGIC: u32 = 0x0131_2F76;

/// 版本字段的标志位（其余位保留，必须为 0）。
pub mod version_flags {
    /// 版本号字节（低 8 位），本模块只接受 2。
    pub const NUMBER_MASK: u32 = 0x0000_00FF;
    /// 分块（tiled）标志（`0x200`）。
    pub const TILED: u32 = 0x0000_0200;
    /// 长属性名标志（`0x400`）。
    pub const LONG_NAMES: u32 = 0x0000_0400;
    /// deep 数据标志（`0x800`）。
    pub const DEEP_DATA: u32 = 0x0000_0800;
    /// 多部件文件标志（`0x1000`）。
    pub const MULTI_PART: u32 = 0x0000_1000;
    /// 所有已定义标志位。
    pub const ALL_KNOWN: u32 = NUMBER_MASK | TILED | LONG_NAMES | DEEP_DATA | MULTI_PART;
}

/// 属性 `dataWindow` 的语义（半开区间之外的边界约定）。
/// 扫描线块排序 `Y_SAMPLE_DECREASING`。
pub const LINE_ORDER_DECREASING: u8 = 1;

/// 属性 `lineOrder` = `INCREASING_Y`：文件中**首个**块是 y 最小的那块。
///
/// 权威依据 = OpenEXR 3.1.5 `ImfLineOrder.h:23`
/// （`INCREASING_Y = 0, // first scan line has lowest y coordinate`）。
///
/// 【W13 互通对拍抓到的第三个真实容器层 bug】本仓写入器原先把 `lineOrder`
/// 标成 [`LINE_ORDER_DECREASING`]（= 1），而 `write_scanline_image` 实际
/// 是按 **y 升序**写块（`for b in 0..blocks`，块首行号 `b * lpb` 递增）。
/// 二者矛盾 → 官方读取器按 `lineOrder` 声明的顺序**倒着**走偏移表，
/// 于是从文件尾往回读，立刻撞上「Early end of file: read 0 out of 4 bytes」。
///
/// 症状与 DWA 无关：连 `Compression::None` 的文件官方都读不了，
/// 任何压缩都失败 —— 这正是「容器层 bug 冒充压缩 bug」的典型形态。
/// 定位方法：把属性逐项 dump 出来与官方文件对比，`lineOrder` 一眼可见
/// （官方 `0x00`，我们 `0x01`）。
pub const LINE_ORDER_INCREASING: u8 = 0;

/// 压缩类型（EXR `compression` 属性字节）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// 无压缩（0）：块数据即原始像素。
    None,
    /// RLE（1）：EXR 私有行程编码（本模块已实现）。
    Rle,
    /// ZIPS（2）：逐扫描线 zlib（对应压缩类型 3 之外的 ZIPS 槽位）。
    Zips,
    /// ZIP（3）：16 扫描线一块的 zlib（本模块已实现，走 [`inflate_raw`]）。
    Zip,
    /// PIZ（4）：无损 wavelet + Huffman（本模块已实现，见 [`crate::exr_compress`]）。
    Piz,
    /// PXR24（5）：24bit 定点 + zlib（本模块已实现，见 [`crate::exr_compress`]）。
    Pxr24,
    /// B44（6）：4×4 小波 + 分层行程（本模块已实现，见 [`crate::exr_compress`]）。
    B44,
    /// B44A（7）：B44 的自适应变体（本模块已实现，解码路径与 B44 相同）。
    B44A,
    /// DWAA（8）：DCT + 量化 + Huf AC（自研实现，见 [`crate::exr_dwa`]）。
    Dwaa,
    /// DWAB（9）：同 DWAA，块高 256 行（质量更高）。
    Dwab,
}

impl Compression {
    /// 由属性字节解码（未知字节 → `None`）。
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Compression::None,
            1 => Compression::Rle,
            2 => Compression::Zips,
            3 => Compression::Zip,
            4 => Compression::Piz,
            5 => Compression::Pxr24,
            6 => Compression::B44,
            7 => Compression::B44A,
            8 => Compression::Dwaa,
            9 => Compression::Dwab,
            _ => return None,
        })
    }

    /// 属性字节（未支持类型也给出真实字节，便于错误信息与写回）。
    pub fn code(self) -> u8 {
        match self {
            Compression::None => 0,
            Compression::Rle => 1,
            Compression::Zips => 2,
            Compression::Zip => 3,
            Compression::Piz => 4,
            Compression::Pxr24 => 5,
            Compression::B44 => 6,
            Compression::B44A => 7,
            Compression::Dwaa => 8,
            Compression::Dwab => 9,
        }
    }

    /// 是否被本模块实现。
    ///
    /// DWAA / DWAB 由 [`crate::exr_dwa`] 真实解码（8×8 IDCT + 量化 +
    /// Huf AC + 通道分类），已对 OpenEXR 3.1.5 写出的真实文件验证。
    /// PIZ / PXR24 / B44 / B44A 由 [`crate::exr_compress`] 真实解码
    /// （LUT + 2D 小波 + Huf / 24bit 定点 + zlib / 4×4 块 packing）。
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Compression::None
                | Compression::Rle
                | Compression::Zips
                | Compression::Zip
                | Compression::Piz
                | Compression::Pxr24
                | Compression::B44
                | Compression::B44A
                | Compression::Dwaa
                | Compression::Dwab
        )
    }

    /// 是否为 DWA 族（解码路径与其它压缩不同：需通道上下文而非整块负载）。
    pub fn is_dwa(self) -> bool {
        matches!(self, Compression::Dwaa | Compression::Dwab)
    }

    /// 是否由 [`crate::exr_compress`] 解码（需通道上下文：类型 / pLinear / 宽高）。
    pub fn is_exr_compress(self) -> bool {
        matches!(
            self,
            Compression::Piz | Compression::Pxr24 | Compression::B44 | Compression::B44A
        )
    }

    /// 块内扫描线数（EXR `numLinesInBuffer`）。不支持类型返回 0。
    ///
    /// PIZ=32 / PXR24=16 / B44=32 / B44A=32 与 OpenEXR 3.1.5 的
    /// `PizCompressor::numScanLines` / `Pxr24Compressor::numScanLines` /
    /// `B44Compressor::numScanLines` 逐一核实一致。
    pub fn lines_per_block(self) -> u32 {
        match self {
            Compression::None => 1,
            Compression::Rle => 1,
            Compression::Zips => 1,
            Compression::Zip => 16,
            Compression::Piz => 32,
            Compression::Pxr24 => 16,
            Compression::B44 => 32,
            Compression::B44A => 32,
            // DWAA = 32 行，DWAB = 256 行（真实 OpenEXR 语义）。
            Compression::Dwaa => 32,
            Compression::Dwab => 256,
        }
    }

    /// 压缩类型的算法摘要（写入结构化错误，便于判断"能不能转存/换路径"）。
    ///
    /// 十种压缩现已**全部**真实解码（`None/Rle/Zips/Zip` 在本模块，
    /// `Piz/Pxr24/B44/B44A` 在 [`crate::exr_compress`]，`Dwaa/Dwab` 在
    /// [`crate::exr_dwa`]），故本函数不再是"是否支持"的判据，而是错误信息里
    /// 的**可读诊断**：调用方据它判断该走哪条解码路径、以及转存时该选什么。
    pub fn rejection(self) -> String {
        match self {
            Compression::None => "无压缩（块负载即原始像素）",
            Compression::Rle => "EXR 私有行程编码（逐扫描线）",
            Compression::Zips => "逐扫描线 zlib",
            Compression::Zip => "16 扫描线一块的 zlib",
            Compression::Piz => {
                "PIZ（值域 LUT + 2D 小波 + Huf 熵编码，无损，需按通道上下文解码）"
            }
            Compression::Pxr24 => "PXR24（24bit 定点差分 + zlib，float 有损，需按通道上下文解码）",
            Compression::B44 => {
                "B44（4×4 块 14 字节行程 + exp/log 表，half 有损，需按通道上下文解码）"
            }
            Compression::B44A => "B44A（B44 的自适应变体，平场块压到 3 字节；解码路径同 B44）",
            Compression::Dwaa => {
                "DWAA（DCT + 量化 + Huf 熵编码，32 行一块，AC 用 8 bit 量化，有损，需按通道上下文解码）"
            }
            Compression::Dwab => {
                "DWAB（DCT + 量化 + Huf 熵编码，256 行一块，AC 用 16 bit 量化，有损，需按通道上下文解码）"
            }
        }
        .to_string()
    }

    /// 是否为**有损**压缩（转存时若源文件用这些类型，位级一致性不可保）。
    ///
    /// - `Pxr24`：float 通道降到 24 bit 尾数。
    /// - `B44` / `B44A`：half 通道量化到 4×4 块的 6 bit 差分行程。
    /// - DWA 族：AC 系数量化 + DCT 舍入。
    ///
    /// `Piz` 是**无损**的（wavelet + Huf 全整数运算）。
    pub fn is_lossy(self) -> bool {
        matches!(
            self,
            Compression::Pxr24
                | Compression::B44
                | Compression::B44A
                | Compression::Dwaa
                | Compression::Dwab
        )
    }
}

/// 通道像素类型（EXR `channels` 条目）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    /// 32 位无符号整型（type 0）。
    Uint,
    /// 16 位半浮点（type 1）。
    Half,
    /// 32 位浮点（type 2）。
    Float,
}

impl PixelType {
    /// 由属性值解码。
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => PixelType::Uint,
            1 => PixelType::Half,
            2 => PixelType::Float,
            _ => return None,
        })
    }
    /// 属性值。
    pub fn code(self) -> i32 {
        match self {
            PixelType::Uint => 0,
            PixelType::Half => 1,
            PixelType::Float => 2,
        }
    }
    /// 字节宽度。
    pub fn byte_width(self) -> usize {
        match self {
            PixelType::Uint | PixelType::Float => 4,
            PixelType::Half => 2,
        }
    }
}

/// RLE 解压（EXR RLE，ImfRle 格式）。
///
/// 格式（EXR 技术文档「RLE Compression」）：首 4 字节小端 `rle_raw_size`
/// 为**未压缩**长度；随后是行程流：
/// - 字节 `b`（`i8`）`< 0` → 后随 1 字节值，重复 `-b` 次（`-128..-1`，负数
///   最多 127 因为 `-b` 需为正；`-b == 1` 不出现）；
/// - 字节 `b >= 0` → 后随 `b + 1` 字节**字面**拷贝（1..=128 字节）。
///
/// 注意 EXR 的 RLE **不是** zlib 的 RLE，也不是 PNG 的 filter 方式。
pub fn rle_decompress(data: &[u8]) -> ExrResult<Vec<u8>> {
    if data.len() < 4 {
        return Err(comp_err(
            "exr::rle_decompress",
            "RLE 块不足 4 字节（缺未压缩长度头）",
            "压缩块被截断",
        ));
    }
    let raw_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // 防御：未压缩长度上限（防止畸形 size 触发巨量分配）。
    const MAX_RAW: usize = 512 * 1024 * 1024;
    if raw_size > MAX_RAW {
        return Err(comp_err(
            "exr::rle_decompress",
            format!("RLE 未压缩长度 {raw_size} 超过上限 {MAX_RAW}"),
            "超大属性/块 size：确认文件未被错位读取",
        ));
    }
    let mut out: Vec<u8> = Vec::with_capacity(raw_size.min(1 << 20));
    let mut i = 4usize;
    while i < data.len() {
        // ⚠️ 控制字节的极性是「**负 = 字面段，非负 = 行程**」——
        // 见 [`rle_decompress_raw`] 顶部依据（W13 互通对拍抓到的第四个真实 bug）。
        let ctrl = data[i] as i8;
        i += 1;
        if ctrl < 0 {
            let lit = (-(ctrl as i32)) as usize;
            let end = i
                .checked_add(lit)
                .ok_or_else(|| comp_err("exr::rle_decompress", "RLE 字面长度溢出", "压缩块损坏"))?;
            let slice = data.get(i..end).ok_or_else(|| {
                comp_err(
                    "exr::rle_decompress",
                    format!("RLE 字面段需要 {lit} 字节，剩余 {} 字节", data.len() - i),
                    "压缩块被截断",
                )
            })?;
            out.extend_from_slice(slice);
            i = end;
        } else {
            let run = (ctrl as usize) + 1;
            let value = data.get(i).copied().ok_or_else(|| {
                comp_err("exr::rle_decompress", "RLE 行程缺值字节", "压缩块被截断")
            })?;
            i += 1;
            for _ in 0..run {
                out.push(value);
            }
        }
        if out.len() > raw_size {
            return Err(comp_err(
                "exr::rle_decompress",
                format!("RLE 解压产出 {} 字节 > 声明的 {raw_size}", out.len()),
                "RLE 行程流损坏或未压缩长度头错误",
            ));
        }
    }
    if out.len() != raw_size {
        return Err(comp_err(
            "exr::rle_decompress",
            format!("RLE 解压得 {} 字节，声明 {raw_size} 字节", out.len()),
            "压缩块被截断或行程流损坏",
        ));
    }
    Ok(out)
}

/// RLE 解压的**无头变体**：DWA 的 RLE 区已单独携带 `packed` / `raw` 两个长度
/// （分别在 DWA 块头的 `rleUncompressedSize` / `rleRawSize`），故数据体**不含**
/// [`rle_decompress`] 那个 4 字节长度头。
///
/// # ⚠️ 控制字节极性：**负 = 字面段，非负 = 行程**（W13 抓到的第四个真实 bug）
///
/// 权威依据 = OpenEXR 3.1.5 `ImfRle.cpp:89` `rleUncompress`：
///
/// ```cpp
/// while (inLength > 0) {
///     if (*in < 0) {                    // ← 负 = 字面段
///         int count = -((int)*in++);
///         inLength -= count + 1;
///         memcpy(out, in, count);  out += count;  in += count;
///     } else {                          // ← 非负 = 行程
///         int count = *in++;
///         inLength -= 2;
///         memset(out, *(char*)in, count+1);  out += count+1;  in++;
///     }
/// }
/// ```
///
/// 与 `rleCompress`（`ImfRle.cpp:56`）逐条对称：可压行程写
/// `*(runEnd-runStart)-1`（**非负**）+ 值字节；不可压段写
/// `*(runStart-runEnd)`（**负**）+ 原始字节。
///
/// 本实现原先把极性写**反了**（负 = 行程、非负 = 字面），对自家编码器
/// 产出的流恰好自洽，但官方 4 通道文件（4 通道 ⇒ 走RLE 区）的行程流
/// 逐格错位，报`RLE 解压产出 N 字节 > 声明的 M`。
pub(crate) fn rle_decompress_raw(
    packed: &[u8],
    packed_size: usize,
    raw_size: usize,
) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr::rle_decompress_raw";
    // 防御：未压缩长度上限（与 rle_decompress 同量级）。
    const MAX_RAW: usize = 512 * 1024 * 1024;
    if raw_size > MAX_RAW {
        return Err(comp_err(
            LOC,
            format!("DWA RLE 未压缩长度 {raw_size} 超过上限 {MAX_RAW}"),
            "超大长度：确认文件未被错位读取",
        ));
    }
    let mut out: Vec<u8> = Vec::with_capacity(raw_size.min(1 << 20));
    let mut i = 0usize;
    while i < packed.len() {
        let ctrl = packed[i] as i8;
        i += 1;
        if ctrl < 0 {
            // 负 = 字面段：`-ctrl` 字节原样拷贝。
            let lit = (-(ctrl as i32)) as usize;
            let end = i.checked_add(lit).ok_or_else(|| {
                comp_err(LOC, "RLE 字面长度溢出", "DWA RLE 区损坏")
            })?;
            let slice = packed.get(i..end).ok_or_else(|| {
                comp_err(
                    LOC,
                    format!("RLE 字面段需要 {lit} 字节，剩余 {} 字节", packed.len() - i),
                    "DWA RLE 区被截断",
                )
            })?;
            out.extend_from_slice(slice);
            i = end;
        } else {
            // 非负 = 行程：`ctrl + 1` 份 `packed[i]`。
            let run = (ctrl as usize) + 1;
            let value = packed.get(i).copied().ok_or_else(|| {
                comp_err(LOC, "RLE 行程缺值字节", "DWA RLE 区被截断")
            })?;
            i += 1;
            for _ in 0..run {
                out.push(value);
            }
        }
        if out.len() > raw_size {
            return Err(comp_err(
                LOC,
                format!("RLE 解压产出 {} 字节 > 声明的 {raw_size}", out.len()),
                "DWA RLE 行程流损坏或长度字段错误",
            ));
        }
    }
    if packed_size > packed.len() {
        return Err(comp_err(
            LOC,
            format!("DWA RLE 压缩长度 {packed_size} > 实际 {} 字节", packed.len()),
            "DWA 块头与负载不一致",
        ));
    }
    if out.len() != raw_size {
        return Err(comp_err(
            LOC,
            format!("RLE 解压得 {} 字节，声明 {raw_size} 字节", out.len()),
            "DWA RLE 区被截断或行程流损坏",
        ));
    }
    Ok(out)
}

/// RLE 压缩（与 [`rle_decompress`] 严格对逆；本模块写出测试用 EXR 时用）。
pub fn rle_compress(data: &[u8]) -> Vec<u8> {
    // ⚠️ 控制字节极性：**非负 = 行程（值 − 1），负 = 字面段（−长度）**。
    // 权威依据 = OpenEXR 3.1.5 `ImfRle.cpp:56` `rleCompress`，与
    // [`rle_decompress_raw`] / [`rle_decompress`] 严格互逆。
    // 本实现原先把极性写反（与自家旧解码器成对自洽），见那两处的根因说明。
    let mut out: Vec<u8> = Vec::with_capacity(data.len() + 4);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    let mut i = 0usize;
    while i < data.len() {
        // 行程：连续同值字节（官方 `MAX_RUN_LENGTH` = 127+1 = 128）。
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == data[i] && run < 128 {
            run += 1;
        }
        if run >= 2 {
            // 官方写作 `*(runEnd - runStart) - 1`，恒 **非负**。
            out.push((run - 1) as u8);
            out.push(data[i]);
            i += run;
        } else {
            // 字面段：累积到下一个 ≥2 行程或 127 字节上限（官方 `MIN_RUN_LENGTH`=2）。
            let start = i;
            let mut lit = 0usize;
            while i < data.len() && lit < 127 {
                let mut r = 1usize;
                while i + r < data.len() && data[i + r] == data[i] && r < 128 {
                    r += 1;
                }
                if r >= 2 {
                    break;
                }
                i += 1;
                lit += 1;
            }
            // 官方写作 `*(runStart - runEnd)`，恒 **负**（长度 1..=127 → -1..-127）。
            out.push((-(lit as i32)) as i8 as u8);
            out.extend_from_slice(&data[start..start + lit]);
        }
    }
    out
}

/// 按压缩类型解码一个块。
///
/// `expected` 是块解压后应有的字节数（`行字节 × 块内扫描线数`）。
/// `ctx` 携带通道上下文 —— DWA / PIZ / PXR24 / B44 都需要按通道重排像素，
/// 不能只吃裸负载。
fn decompress_block(
    compression: Compression,
    data: &[u8],
    expected: usize,
    ctx: &BlockCtx<'_>,
) -> ExrResult<Vec<u8>> {
    if !compression.is_supported() {
        return Err(comp_err(
            "exr::decompress_block",
            format!(
                "压缩类型 {}（{}）未实现",
                compression.code(),
                compression.rejection()
            ),
            "请用 OpenEXR/OIIO 转存为无压缩、ZIP 或 RLE 的 EXR 后再读入",
        ));
    }
    let out: Vec<u8> = match compression {
        Compression::None => data.to_vec(),
        Compression::Rle => rle_decompress(data)?,
        // EXR 的 ZIP/ZIPS 块是 **zlib 流**（2 字节头 + DEFLATE + 4 字节 Adler-32），
        // **不是**裸 DEFLATE。真实 OpenEXR 写出的块以 `0x78 0x5E` 开头。
        Compression::Zip | Compression::Zips => zlib_decompress(data)?,
        // PIZ / PXR24 / B44 / B44A 走 exr_compress（需通道上下文）。
        other if other.is_exr_compress() => {
            crate::exr_compress::decompress_block(other, data, ctx)?
        }
        // DWAA / DWAB（W13 互通对拍接上）：DWA 块有自己的四区布局
        // （unknown/AC/DC/RLE）+ 通道分类，负载不能当裸数据处理。
        //
        // ⚠️ 分块（tiled）DWA 此前**完全没有接入**本分派表：官方写的
        // tiled DWA 文件一律报「压缩类型 8/9 尚未接入解压分派」。这是
        // 「`is_supported()` 已标 true 但分派表漏登记」的典型缺口——
        // 由 W13 矩阵的 tiled 整列暴露。
        Compression::Dwaa | Compression::Dwab => {
            let chans: Vec<(String, PixelType, bool)> = ctx
                .channels
                .iter()
                .map(|c| (c.name.clone(), c.pixel_type, c.p_linear))
                .collect();
            crate::exr_dwa::decompress_block(
                data,
                &crate::exr_dwa::DwaContext {
                    channels: &chans,
                    width: ctx.width,
                    height: ctx.height,
                },
                ctx.height,
            )?
        }
        other => {
            // 兜底：新增压缩类型时若忘记在此登记，结构化拒绝而非 panic（C18铁律）。
            return Err(comp_err(
                "exr::decompress_block",
                format!("压缩类型 {} 尚未接入解压分派", other.code()),
                "已在 is_supported 标记为不支持；请确认分派表与 is_supported 同步",
            ));
        }
    };
    if out.len() != expected {
        return Err(comp_err(
            "exr::decompress_block",
            format!(
                "解压后 {} 字节，块应有 {expected} 字节（宽度 × 通道字节数 × 扫描线数）",
                out.len()
            ),
            "压缩块损坏或通道表/块尺寸与偏移表不一致",
        ));
    }
    Ok(out)
}

/// DEFLATE **裸码流**解压（**裸流**，无 zlib 头；EXR 的 ZIP/ZIPS 块不含 zlib 包装）。
///
/// 实现已下沉到 L0 叶子 crate [`prism_bytes`]（全引擎唯一 DEFLATE 真源）；
/// 本函数保留是为了**调用点零改动**——`prism_io::exr::inflate_raw` 是既有
/// 公开 API（`prism-io/src/lib.rs` 有 re-export，`exr_dwa` / `exr_write` /
/// `examples/zt.rs` 都在用）。
pub fn inflate_raw(data: &[u8]) -> ExrResult<Vec<u8>> {
    prism_bytes::inflate::inflate_raw(data).map_err(bytes_to_exr)
}

/// zlib 解压（RFC 1950）：2 字节头 + 裸 DEFLATE + 4 字节 Adler-32。
///
/// EXR 的 ZIP / ZIPS / DWA 子块（UNKNOWN / DC / RLE）都用 **zlib 包装**。
/// 头部低 4 位必须是 8（deflate），`(CMF<<8|FLG) % 31 == 0` 是头校验。
///
/// # 迁移修掉的缺陷（有意为之）
///
/// 旧实现在解出后有
/// `if data.len() < 2 + out.len() { return Err("zlib 流截断") }`——拿**压缩后
/// 总长**比**解压后长度**，于是任何真正压缩过的 zlib 流（压缩率 > 0）都会被
/// 误判为「截断」。新实现按 RFC 1950 用实际消费量定位尾部。
/// 详见 [`prism_bytes::zlib::decompress`]。
pub fn zlib_decompress(data: &[u8]) -> ExrResult<Vec<u8>> {
    prism_bytes::zlib::decompress(data).map_err(bytes_to_exr)
}

/// zlib 解压 + **强制校验**尾部 Adler-32（新增能力）。
pub fn zlib_decompress_verified(data: &[u8]) -> ExrResult<Vec<u8>> {
    prism_bytes::zlib::decompress_verified(data).map_err(bytes_to_exr)
}

/// [`prism_bytes::BytesError`] → EXR 域压缩错误（`EXR2xxx`）。
///
/// 三段式一【在哪】加 `exr::` 前缀，使按 `location` 聚合日志时仍看得出
/// 错误源自 EXR 层而底层是共享的字节层。
fn bytes_to_exr(e: prism_bytes::BytesError) -> ExrError {
    ExrError::new(
        "EXR2001",
        format!("exr::{}", e.location),
        e.problem,
        e.fix,
    )
}


// ===========================================================================
//  四、容器结构解析
// ===========================================================================

/// 边界安全的只读游标。
#[derive(Debug, Clone)]
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }
    fn take(&mut self, n: usize, loc: &'static str) -> ExrResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            struct_err(
                loc,
                format!("偏移 {} + 长度 {n} 溢出", self.pos),
                "文件被截断或偏移损坏",
            )
        })?;
        let s = self.data.get(self.pos..end).ok_or_else(|| {
            struct_err(
                loc,
                format!(
                    "需要 [{}..{}) 共 {n} 字节，文件仅 {} 字节",
                    self.pos,
                    end,
                    self.data.len()
                ),
                "文件被截断；确认传输完整后重试",
            )
        })?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self, loc: &'static str) -> ExrResult<u8> {
        Ok(self.take(1, loc)?[0])
    }
    fn i32(&mut self, loc: &'static str) -> ExrResult<i32> {
        let b = self.take(4, loc)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u32(&mut self, loc: &'static str) -> ExrResult<u32> {
        let b = self.take(4, loc)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self, loc: &'static str) -> ExrResult<u64> {
        let b = self.take(8, loc)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    /// 读 NUL 结尾的 ASCII 串（游标停在 NUL **之后**）。
    fn cstr(&mut self, loc: &'static str) -> ExrResult<String> {
        let start = self.pos;
        let mut end = self.pos;
        while end < self.data.len() {
            if self.data[end] == 0 {
                let s =
                    String::from_utf8_lossy(self.data.get(start..end).unwrap_or(&[])).into_owned();
                self.pos = end + 1;
                return Ok(s);
            }
            end += 1;
        }
        Err(struct_err(
            loc,
            format!("从偏移 {start} 起的属性名无 NUL 终止符"),
            "属性表被截断或非 EXR 文件",
        ))
    }
}

/// 通道描述（`channels` 属性条目）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrChannel {
    /// 通道名（`"R"`/`"G"`/`"B"`/`"A"`/`"Z"`/`"normal.x"`/…）。
    pub name: String,
    /// 像素类型。
    pub pixel_type: PixelType,
    /// 线性标志（EXR 语义提示，本模块只透传不解释）。
    pub p_linear: bool,
    /// 水平采样（> 1 属子采样，本模块结构化拒绝）。
    pub x_sampling: i32,
    /// 垂直采样。
    pub y_sampling: i32,
}

/// 二维数据窗口（`dataWindow` / `displayWindow`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataWindow {
    /// 左边界（含）。
    pub x_min: i32,
    /// 上边界（含）。
    pub y_min: i32,
    /// 右边界（含）。
    pub x_max: i32,
    /// 下边界（含）。
    pub y_max: i32,
}

impl DataWindow {
    /// 宽（像素）。
    pub fn width(&self) -> u32 {
        (self.x_max as i64 - self.x_min as i64 + 1).max(0) as u32
    }
    /// 高（像素）。
    pub fn height(&self) -> u32 {
        (self.y_max as i64 - self.y_min as i64 + 1).max(0) as u32
    }
    /// 像素总数（`checked_mul` → `None`）。
    pub fn pixel_count(&self) -> Option<usize> {
        (self.width() as usize).checked_mul(self.height() as usize)
    }
    /// 构造测试/默认值（0 基窗口）。
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            x_min: 0,
            y_min: 0,
            x_max: width as i32 - 1,
            y_max: height as i32 - 1,
        }
    }
}

/// tile 描述（分块图像的 `tiles` 属性）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDesc {
    /// tile 宽（像素）。
    pub x_size: u32,
    /// tile 高（像素）。
    pub y_size: u32,
    /// level 模式（0 = ONE_LEVEL、1 = MIPMAP_LEVELS、2 = RIPMAP_LEVELS）。
    pub level_mode: u8,
}

impl TileDesc {
    /// tile 网格的列数。
    pub fn tiles_x(&self, win: &DataWindow) -> u32 {
        if self.x_size == 0 {
            0
        } else {
            (win.width() + self.x_size - 1) / self.x_size
        }
    }
    /// tile 网格的行数。
    pub fn tiles_y(&self, win: &DataWindow) -> u32 {
        if self.y_size == 0 {
            0
        } else {
            (win.height() + self.y_size - 1) / self.y_size
        }
    }
    /// tile 总数（`checked_mul` → `None`）。
    pub fn tile_count(&self, win: &DataWindow) -> Option<usize> {
        (self.tiles_x(win) as usize).checked_mul(self.tiles_y(win) as usize)
    }
}

/// EXR 头部（属性表解析结果）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrHeader {
    /// 版本字段原始值。
    pub version: u32,
    /// 是否分块图像。
    pub tiled: bool,
    /// 是否 deep 数据。
    pub deep: bool,
    /// 是否多部件。
    pub multipart: bool,
    /// 压缩类型。
    pub compression: Compression,
    /// 图像数据窗口。
    pub data_window: DataWindow,
    /// 显示窗口。
    pub display_window: DataWindow,
    /// 扫描线顺序（0 = INCREASING_Y、1 = DECREASING_Y）。
    pub line_order: u8,
    /// 通道（**文件中的出现顺序**；遍历确定性的依据）。
    pub channels: Vec<ExrChannel>,
    /// tile 描述（分块图像必有）。
    pub tile: Option<TileDesc>,
    /// chunk 偏移表（字节偏移，升序对应块序号）。
    pub chunk_offsets: Vec<u64>,
    /// 字节序（EXR 恒小端；保留字段以便统一接口）。
    pub little_endian: bool,
}

impl ExrHeader {
    /// 扫描线块数。
    ///
    /// deep 扫描线**恒为每行一块**（与 [`read_deep_image`] 的逐行取块一致），
    /// 故块数 = 行数，与压缩类型无关。
    pub fn block_count(&self) -> usize {
        if self.deep {
            return self.data_window.height() as usize;
        }
        let lpb = self.compression.lines_per_block();
        if lpb == 0 {
            0
        } else {
            ((self.data_window.height() + lpb - 1) / lpb) as usize
        }
    }

    /// 每条扫描线的像素总字节数（所有通道字节宽之和 × 宽）。
    pub fn row_bytes(&self) -> usize {
        let per_row: usize = self
            .channels
            .iter()
            .map(|c| c.pixel_type.byte_width())
            .sum();
        per_row * (self.data_window.width() as usize)
    }

    /// 第 `block_index` 块内**实际**扫描线数（末块可能不足整块）。
    ///
    /// 末块不足是常态（如 ZIP 每块 16 行、图像仅 20 行 ⇒ 末块 4 行），
    /// 故块字节数必须按实际行数算，不能一律用 `lines_per_block()`。
    pub fn block_lines(&self, block_index: usize) -> u32 {
        if self.tiled {
            return self.tile.map(|t| t.y_size).unwrap_or(0);
        }
        let lpb = self.compression.lines_per_block();
        if lpb == 0 {
            return 0;
        }
        let total = self.data_window.height();
        let start = (block_index as u32).saturating_mul(lpb);
        if start >= total {
            return 0;
        }
        (total - start).min(lpb)
    }

    /// 第 `block_index` 块的像素总字节数（未压缩）。
    pub fn block_bytes_at(&self, block_index: usize) -> usize {
        self.row_bytes() * (self.block_lines(block_index) as usize)
    }

    /// 按名取通道（线性扫描，名字少时最优且顺序确定）。
    pub fn channel(&self, name: &str) -> Option<&ExrChannel> {
        self.channels.iter().find(|c| c.name == name)
    }
}

/// 一个已解码的通道平面（f32 计算域，通道原始精度见 `pixel_type`）。
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelPlane {
    /// 通道名。
    pub name: String,
    /// 通道原始像素类型（决定量化位宽）。
    pub pixel_type: PixelType,
    /// 平面宽。
    pub width: u32,
    /// 平面高。
    pub height: u32,
    /// 行主序数据（长度 = `width × height`）。
    pub values: Vec<f32>,
}

/// 已解码的 flat（扁平）EXR 图像。
#[derive(Debug, Clone, PartialEq)]
pub struct ExrImage {
    /// 头部。
    pub header: ExrHeader,
    /// 各通道平面（顺序 = `header.channels` 的文件出现顺序）。
    pub planes: Vec<ChannelPlane>,
}

impl ExrImage {
    /// 按名取平面。
    pub fn plane(&self, name: &str) -> Option<&ChannelPlane> {
        self.planes.iter().find(|p| p.name == name)
    }

    /// 合成 RGBA [`Fbuf16F`]（`R`/`G`/`B` 缺则补 0、`A` 缺则补 1.0）。
    ///
    /// 交付量化只发生在此处（§2.2）；通道值不做色调映射。
    pub fn to_fbuf16f(&self) -> Fbuf16F {
        let w = self.header.data_window.width();
        let h = self.header.data_window.height();
        let mut fb = Fbuf16F::new(w, h);
        let pick = |name: &str| -> Option<&ChannelPlane> { self.plane(name) };
        for y in 0..h {
            for x in 0..w {
                let idx = (y as usize) * (w as usize) + x as usize;
                let get = |p: Option<&ChannelPlane>| -> f32 {
                    p.and_then(|pl| pl.values.get(idx)).copied().unwrap_or(0.0)
                };
                let a = pick("A").and_then(|pl| pl.values.get(idx)).copied();
                let rgba = [
                    get(pick("R")),
                    get(pick("G")),
                    get(pick("B")),
                    a.unwrap_or(1.0),
                ];
                fb.set(x, y, rgba);
            }
        }
        fb
    }
}

/// 解析 EXR 头部（含 chunk 偏移表）。
pub fn read_header(data: &[u8]) -> ExrResult<ExrHeader> {
    const LOC: &str = "exr::read_header";
    let mut cur = Cursor::new(data, 0);
    let magic = cur.u32(LOC)?;
    if magic != EXR_MAGIC {
        return Err(struct_err(
            LOC,
            format!("magic = {magic:#010x}，应为 {EXR_MAGIC:#010x}（字节 76 2F 31 01）"),
            "确认这是 OpenEXR 文件（不是 PNG/TIFF/序列帧目录）",
        ));
    }
    let version = cur.u32(LOC)?;
    if version & !version_flags::ALL_KNOWN != 0 {
        return Err(struct_err(
            LOC,
            format!("版本字段 {version:#010x} 含未定义标志位"),
            "该文件使用了本模块未知的 EXR 特性位；请确认来源",
        ));
    }
    if version & version_flags::NUMBER_MASK != 2 {
        return Err(struct_err(
            LOC,
            format!(
                "EXR 版本号 = {}，本模块只实现版本 2",
                version & version_flags::NUMBER_MASK
            ),
            "版本 1 已废弃；确认文件来源",
        ));
    }
    let tiled = version & version_flags::TILED != 0;
    let deep = version & version_flags::DEEP_DATA != 0;
    let multipart = version & version_flags::MULTI_PART != 0;
    if multipart {
        return Err(struct_err(
            LOC,
            "多部件 EXR（版本字段 MULTI_PART 标志置位）",
            "多部件需要逐部件属性表与 chunk 索引，本模块只实现单部件；\
             请用 OIIO 展平为单部件",
        ));
    }

    // 属性表：name(NUL) type(NUL) size(u32) value。
    let mut channels: Vec<ExrChannel> = Vec::new();
    let mut compression: Option<Compression> = None;
    let mut data_window: Option<DataWindow> = None;
    let mut display_window: Option<DataWindow> = None;
    let mut line_order: u8 = 0;
    let mut tile: Option<TileDesc> = None;
    let mut seen: Vec<&'static str> = Vec::new(); // 重复属性检测（确定性遍历）

    loop {
        let name = cur.cstr(LOC)?;
        if name.is_empty() {
            break; // 属性表结束
        }
        let ty = cur.cstr(LOC)?;
        let size = cur.u32(LOC)? as usize;
        // 超大属性 size 防御（真实 EXR 的最大属性远小于 1 MiB）。
        const MAX_ATTR: usize = 1 << 20;
        if size > MAX_ATTR {
            return Err(struct_err(
                LOC,
                format!("属性 `{name}` 声明 size = {size} 字节，超过上限 {MAX_ATTR}"),
                "超大属性 size：文件被截断或字节流错位；不要试图分配该尺寸",
            ));
        }
        let value = cur.take(size, LOC)?.to_vec();

        let dup = |seen: &Vec<&'static str>, k: &'static str| -> bool { seen.contains(&k) };
        match name.as_str() {
            "channels" => {
                if dup(&seen, "channels") {
                    return Err(struct_err(
                        LOC,
                        "属性 `channels` 重复出现",
                        "EXR 属性表不允许重复键",
                    ));
                }
                seen.push("channels");
                channels = parse_channels(&value)?;
            }
            "compression" => {
                if size < 1 {
                    return Err(struct_err(
                        LOC,
                        "属性 `compression` 为空",
                        "EXR 规范要求 1 字节",
                    ));
                }
                let code = value[0];
                compression = Some(Compression::from_code(code).ok_or_else(|| {
                    struct_err(
                        LOC,
                        format!("未知压缩类型字节 {code}"),
                        "该 EXR 使用未注册的压缩算法；请用 OIIO 转存",
                    )
                })?);
            }
            "dataWindow" => {
                if value.len() < 16 {
                    return Err(struct_err(
                        LOC,
                        "属性 `dataWindow` 短于 16 字节",
                        "属性被截断",
                    ));
                }
                data_window = Some(DataWindow {
                    x_min: i32::from_le_bytes([value[0], value[1], value[2], value[3]]),
                    y_min: i32::from_le_bytes([value[4], value[5], value[6], value[7]]),
                    x_max: i32::from_le_bytes([value[8], value[9], value[10], value[11]]),
                    y_max: i32::from_le_bytes([value[12], value[13], value[14], value[15]]),
                });
            }
            "displayWindow" => {
                if value.len() < 16 {
                    return Err(struct_err(
                        LOC,
                        "属性 `displayWindow` 短于 16 字节",
                        "属性被截断",
                    ));
                }
                display_window = Some(DataWindow {
                    x_min: i32::from_le_bytes([value[0], value[1], value[2], value[3]]),
                    y_min: i32::from_le_bytes([value[4], value[5], value[6], value[7]]),
                    x_max: i32::from_le_bytes([value[8], value[9], value[10], value[11]]),
                    y_max: i32::from_le_bytes([value[12], value[13], value[14], value[15]]),
                });
            }
            "lineOrder" => {
                line_order = value.first().copied().unwrap_or(0);
            }
            "tiles" => {
                if value.len() < 9 {
                    return Err(struct_err(LOC, "属性 `tiles` 短于 9 字节", "属性被截断"));
                }
                tile = Some(TileDesc {
                    x_size: u32::from_le_bytes([value[0], value[1], value[2], value[3]]),
                    y_size: u32::from_le_bytes([value[4], value[5], value[6], value[7]]),
                    level_mode: value[8] & 0x0F,
                });
            }
            _ => {
                let _ = ty; // 未知属性（如 chromaticities / stringVector）按规范跳过
            }
        }
    }

    let compression = compression
        .ok_or_else(|| struct_err(LOC, "缺必需属性 `compression`", "确认文件完整（未截断）"))?;
    let data_window = data_window
        .ok_or_else(|| struct_err(LOC, "缺必需属性 `dataWindow`", "确认文件完整（未截断）"))?;
    let display_window = display_window.unwrap_or(data_window);
    if channels.is_empty() {
        return Err(chan_err(
            LOC,
            "`channels` 属性为空",
            "确认文件完整（未截断）",
        ));
    }
    // 子采样通道不做支持（值域与采样数学都要重定义）。
    for c in &channels {
        if c.x_sampling != 1 || c.y_sampling != 1 {
            return Err(chan_err(
                LOC,
                format!(
                    "通道 `{}` 的采样为 {}×{}（本模块只支持 1×1）",
                    c.name, c.x_sampling, c.y_sampling
                ),
                "请用 OIIO 展平为全分辨率后重试",
            ));
        }
    }
    if data_window.width() == 0 || data_window.height() == 0 {
        return Err(struct_err(
            LOC,
            format!(
                "dataWindow 宽高为 0（{}×{}）",
                data_window.width(),
                data_window.height()
            ),
            "空图像无法解码；确认属性正确",
        ));
    }
    if tiled && tile.is_none() {
        return Err(struct_err(
            LOC,
            "版本标记为分块但缺 `tiles` 属性",
            "分块图像必须声明 tile 尺寸；确认文件完整",
        ));
    }

    // 块数：flat 扫描线按压缩类型推块数；tiled 用 tile 网格；deep 扫描线**恒为
    // 每扫描线一块**（`read_deep_image` 按 `y - y_min` 逐行取块，故块数必等于行数）。
    let blocks: usize = if tiled {
        let t = tile.unwrap_or(TileDesc {
            x_size: 0,
            y_size: 0,
            level_mode: 0,
        });
        t.tile_count(&data_window)
            .ok_or_else(|| struct_err(LOC, "tile 数量溢出", "使用更小的图像或更小的 tile"))?
    } else if deep {
        data_window.height() as usize
    } else {
        let lpb = compression.lines_per_block();
        ((data_window.height() as usize) + (lpb as usize) - 1) / (lpb as usize)
    };
    // 偏移表条目上限防御（deep 额外 +1 哨兵条目）。
    const MAX_BLOCKS: usize = 1 << 24;
    if blocks > MAX_BLOCKS {
        return Err(struct_err(
            LOC,
            format!("块数 {blocks} 超过上限 {MAX_BLOCKS}"),
            "超大偏移表：确认文件尺寸与 dataWindow 一致",
        ));
    }
    let table_entries = if deep { blocks + 1 } else { blocks };
    let mut chunk_offsets: Vec<u64> = Vec::with_capacity(table_entries);
    for i in 0..table_entries {
        let off = cur.u64(LOC)?;
        // 偏移越界（负偏移语义：u64 高位为 1 的巨大值）在此拦截。
        if off > data.len() as u64 {
            return Err(struct_err(
                LOC,
                format!("第 {i} 个 chunk 偏移 {off} 超过文件长度 {}", data.len()),
                "偏移表损坏；文件被截断或字节流错位（含负偏移被当成无符号）",
            ));
        }
        chunk_offsets.push(off);
    }

    Ok(ExrHeader {
        version,
        tiled,
        deep,
        multipart,
        compression,
        data_window,
        display_window,
        line_order,
        channels,
        tile,
        chunk_offsets,
        little_endian: true,
    })
}

/// 解析 `channels` 属性负载。
fn parse_channels(value: &[u8]) -> ExrResult<Vec<ExrChannel>> {
    const LOC: &str = "exr::parse_channels";
    let mut cur = Cursor::new(value, 0);
    let mut out: Vec<ExrChannel> = Vec::new();
    loop {
        let name = cur.cstr(LOC)?;
        if name.is_empty() {
            break; // 通道列表结束（以空串终止）
        }
        if out.len() >= 1024 {
            return Err(chan_err(
                LOC,
                "通道数超过 1024 上限",
                "超大通道表：确认文件不是被错位解析",
            ));
        }
        let pt = cur.i32(LOC)?;
        let pixel_type = PixelType::from_code(pt).ok_or_else(|| {
            chan_err(
                LOC,
                format!("通道 `{name}` 的像素类型 {pt} 非法（须 0=UINT / 1=HALF / 2=FLOAT）"),
                "确认 channels 属性负载正确",
            )
        })?;
        let p_linear = cur.u8(LOC)? != 0;
        let _reserved = cur.take(3, LOC)?;
        let x_sampling = cur.i32(LOC)?;
        let y_sampling = cur.i32(LOC)?;
        out.push(ExrChannel {
            name,
            pixel_type,
            p_linear,
            x_sampling,
            y_sampling,
        });
    }
    Ok(out)
}

// ===========================================================================
//  五、块解码（flat 扫描线 / tiled）
// ===========================================================================

/// 块内像素的字节序布局：EXR 逐扫描线内「通道连续、行内像素连续」
/// （**不**做像素级交织，ImfScanLineOutputFile 约定）。
fn unpack_block(
    raw: &[u8],
    channels: &[ExrChannel],
    width: usize,
    lines: usize,
    out: &mut Vec<(Vec<f32>, PixelType)>,
) -> ExrResult<()> {
    let mut off = 0usize;
    for _line in 0..lines {
        for c in channels {
            let n = width * c.pixel_type.byte_width();
            let end = off
                .checked_add(n)
                .ok_or_else(|| struct_err("exr::unpack_block", "块内偏移溢出", "块尺寸异常"))?;
            let slice = raw.get(off..end).ok_or_else(|| {
                struct_err(
                    "exr::unpack_block",
                    format!("块内需要 [{off}..{end}) 字节，块仅 {} 字节", raw.len()),
                    "通道表与块尺寸不一致；确认 dataWindow 与 channels 正确",
                )
            })?;
            let mut plane: Vec<f32> = Vec::with_capacity(width);
            match c.pixel_type {
                PixelType::Uint => {
                    for i in 0..width {
                        let b = slice.get(i * 4..i * 4 + 4).unwrap_or(&[0; 4]);
                        plane.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32);
                    }
                }
                PixelType::Half => {
                    for i in 0..width {
                        let b = slice.get(i * 2..i * 2 + 2).unwrap_or(&[0; 2]);
                        plane.push(f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
                    }
                }
                PixelType::Float => {
                    for i in 0..width {
                        let b = slice.get(i * 4..i * 4 + 4).unwrap_or(&[0; 4]);
                        plane.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    }
                }
            }
            out.push((plane, c.pixel_type));
            off = end;
        }
    }
    Ok(())
}

/// 读一个扫描线块（`block_index`），返回各通道该块的 f32 平面（每行一段）。
///
/// 块头布局：`y`（i32，块的**首**扫描线行号）+ 负载长度（i32）。
fn read_scanline_block(
    data: &[u8],
    header: &ExrHeader,
    block_index: usize,
) -> ExrResult<Vec<(Vec<f32>, PixelType)>> {
    const LOC: &str = "exr::read_scanline_block";
    let off = header
        .chunk_offsets
        .get(block_index)
        .copied()
        .ok_or_else(|| {
            struct_err(
                LOC,
                format!(
                    "块序号 {block_index} 越界（偏移表 {} 项）",
                    header.chunk_offsets.len()
                ),
                "块索引超出图像范围",
            )
        })?;
    let at = off as usize;
    let mut cur = Cursor::new(data, at);
    let y = cur.i32(LOC)?;
    let payload_len = cur.i32(LOC)?;
    if payload_len < 0 {
        return Err(struct_err(
            LOC,
            format!("块 {block_index} 负载长度 {payload_len} 为负"),
            "负载长度字段损坏（负偏移/长度）；确认文件未损坏",
        ));
    }
    let lpb = header.compression.lines_per_block();
    // 块首行号必须与块序号一致（线性扫描线图的强制约定）。
    let expect_y = header.data_window.y_min + (block_index as i32) * (lpb as i32);
    if y != expect_y {
        return Err(struct_err(
            LOC,
            format!("块 {block_index} 的 y = {y}，按偏移表应为 {expect_y}"),
            "偏移表与块头不一致；文件被重排或损坏",
        ));
    }
    let payload = cur.take(payload_len as usize, LOC)?;
    // 未压缩时负载长度应恰等于块字节数（真实 EXR 如此）。
    let actual_lines = header.block_lines(block_index);
    if header.compression == Compression::None {
        let want = header.row_bytes() * (actual_lines as usize);
        if payload.len() != want {
            return Err(struct_err(
                LOC,
                format!(
                    "无压缩块 {} 字节，应为 {want}（宽 {} × 本块扫描线 {actual_lines} × 通道字节）",
                    payload.len(),
                    header.data_window.width()
                ),
                "通道表与块尺寸不一致；确认 channels 与 dataWindow 匹配",
            ));
        }
    }
    // 末块可能不足 lpb 行：字节数与解包行数都按**实际行数**。
    let lines = header.block_lines(block_index);
    let expected = header.row_bytes() * (lines as usize);
let raw = if header.compression.is_dwa() {
        // DWA 需要通道上下文（分类依赖通道名/类型/pLinear），不能只吃裸负载。
        //
        // ⚠️ **逐块「压不小就原样存」回退**（W13 互通对拍抓到的第七个真实 bug）：
        // 官方 `ImfOutputFile.cpp:584` 写每个块时做
        // ```cpp
        // int compSize = compressor->compress(...);
        // if (compSize < _lineBuffer->dataSize) { 用压缩结果 }
        // // 否则：保持原样（仅在 NATIVE 时做 XDR 转换）
        // ```
        // 即**块级**判断，不是文件级。高频噪声这类不可压缩数据会让
        // **同一个文件里一部分块是 DWA 块、另一部分是原始像素**。
        // 实测：`dwaa-half-scan-1-noise`（64×48，DWAA 块高 32，共 2 块）
        // 第 1 块压不小 → 原样 4096 字节；第 2 块压得动 → 真DWA 块。
        //
        // 判据：块负载长度 == 该块未压缩字节数 ⇒ 原样（官方只在压缩后**变大或相等**
        // 时才放弃，故压缩后恰好等长也会走回退）。
        if payload_len as usize == expected {
            payload.to_vec()
        } else {
            let chans: Vec<(String, PixelType, bool)> = header
                .channels
                .iter()
                .map(|c| (c.name.clone(), c.pixel_type, c.p_linear))
                .collect();
            let ctx = crate::exr_dwa::DwaContext {
                channels: &chans,
                width: header.data_window.width() as usize,
                height: lines as usize,
            };
            crate::exr_dwa::decompress_block(payload, &ctx, lines as usize)?
        }
    } else {
        let ctx = BlockCtx {
            channels: &header.channels,
            width: header.data_window.width() as usize,
            height: lines as usize,
        };
        decompress_block(header.compression, payload, expected, &ctx)?
    };
    let mut planes: Vec<(Vec<f32>, PixelType)> = Vec::new();
    unpack_block(
        &raw,
        &header.channels,
        header.data_window.width() as usize,
        lines as usize,
        &mut planes,
    )?;
    Ok(planes)
}

/// 完整读入 flat 扫描线 EXR。
pub fn read_image(data: &[u8]) -> ExrResult<ExrImage> {
    let header = read_header(data)?;
    if header.tiled {
        return Err(struct_err(
            "exr::read_image",
            "这是分块（tiled）EXR，请用 `read_tile` 或 `read_all_tiles`",
            "分块图像按 tile 索引读取更省内存",
        ));
    }
    if header.deep {
        return Err(struct_err(
            "exr::read_image",
            "这是 deep EXR，请用 `read_deep_image`",
            "deep 图像的每像素样本数不固定，需专用读取器",
        ));
    }
    read_flat_region(data, &header, &header.data_window)
}

/// 分块读取 API：只读指定矩形区域（`region` 用 dataWindow 的绝对坐标）。
///
/// 只触碰覆盖该区域的块——**不**全量载入。区域边界按块对齐裁剪，
/// 返回平面的 `width`/`height` 为区域尺寸。
pub fn read_flat_region(
    data: &[u8],
    header: &ExrHeader,
    region: &DataWindow,
) -> ExrResult<ExrImage> {
    const LOC: &str = "exr::read_flat_region";
    if header.tiled {
        return Err(struct_err(
            LOC,
            "分块 EXR 不支持矩形区域读取",
            "请用 `read_tile`（按 tile 索引）或 `read_all_tiles`",
        ));
    }
    if header.deep {
        return Err(struct_err(
            LOC,
            "deep EXR 不支持矩形区域读取",
            "请用 `read_deep_image` / `read_deep_region`",
        ));
    }
    if region.x_min > region.x_max || region.y_min > region.y_max {
        return Err(struct_err(
            LOC,
            format!(
                "区域非法：({}, {})–({}, {})",
                region.x_min, region.y_min, region.x_max, region.y_max
            ),
            "须满足 x_min ≤ x_max 且 y_min ≤ y_max",
        ));
    }
    if region.x_min < header.data_window.x_min
        || region.y_min < header.data_window.y_min
        || region.x_max > header.data_window.x_max
        || region.y_max > header.data_window.y_max
    {
        return Err(struct_err(
            LOC,
            format!(
                "区域 ({}, {})–({}, {}) 越出 dataWindow ({}, {})–({}, {})",
                region.x_min,
                region.y_min,
                region.x_max,
                region.y_max,
                header.data_window.x_min,
                header.data_window.y_min,
                header.data_window.x_max,
                header.data_window.y_max
            ),
            "收缩区域到 dataWindow 内",
        ));
    }
    let win = header.data_window.clone();
    let lpb = header.compression.lines_per_block();
    let width = region.width() as usize;
    let height = region.height() as usize;
    let mut planes: Vec<ChannelPlane> = header
        .channels
        .iter()
        .map(|c| ChannelPlane {
            name: c.name.clone(),
            pixel_type: c.pixel_type,
            width: width as u32,
            height: height as u32,
            values: vec![0.0; width * height],
        })
        .collect();

    // 覆盖区域的块序号区间（y 方向；块内可能含区域外行，按需丢弃）。
    let first_block = ((region.y_min - win.y_min) as u32) / lpb;
    let last_row = (region.y_max - win.y_min) as u32;
    let last_block = last_row / lpb;

    let nch = header.channels.len();
    for b in first_block..=last_block {
        let block = read_scanline_block(data, header, b as usize)?;
        let blk_y0 = win.y_min + (b as i32) * (lpb as i32);
        // 块内每条扫描线对应 planes 中的一段（EXR 布局：线 → 通道 → 像素）。
        for line in 0..header.block_lines(b as usize) {
            let row = blk_y0 + (line as i32);
            if row < region.y_min || row > region.y_max {
                continue;
            }
            let dst_y = (row - region.y_min) as usize;
            for ci in 0..header.channels.len() {
                // `block` 是「逐扫描线 × 逐通道」行主序，故第 `line` 行第 `ci`
                // 个通道的下标是 `line * 通道数 + ci` —— 不是 `ci`。
                let seg = match block.get(line as usize * nch + ci) {
                    Some((v, _)) => v,
                    None => continue,
                };
                let plane = match planes.get_mut(ci) {
                    Some(p) => p,
                    None => continue,
                };
                for i in 0..width {
                    let src_x = (region.x_min - win.x_min) as usize + i;
                    let v = seg.get(src_x).copied().unwrap_or(0.0);
                    let dst = dst_y * width + i;
                    if let Some(slot) = plane.values.get_mut(dst) {
                        *slot = v;
                    }
                }
            }
        }
    }

    let mut h = header.clone();
    h.data_window = *region;
    Ok(ExrImage { header: h, planes })
}

// ===========================================================================
//  六、deep image（Z / Float 变体，单点 deep）
// ===========================================================================

/// 单个 deep 样本。
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSample {
    /// 样本在像素内的序号（0 = 最近）。
    pub index: u32,
    /// 深度值（取自 `Z` 或 `Zfloat` 通道，缺失 → `f32::NAN`）。
    pub z: f32,
    /// 该样本的其余通道值（顺序 = `DeepImage::channels`）。
    pub values: Vec<f32>,
}

/// 单个 deep 像素的样本序列。
#[derive(Debug, Clone, PartialEq)]
pub struct DeepPixel {
    /// 像素的绝对坐标 x。
    pub x: i32,
    /// 像素的绝对坐标 y。
    pub y: i32,
    /// 样本序列（按 Z 升序，index 从 0 起）。
    pub samples: Vec<DeepSample>,
}

impl DeepPixel {
    /// 首个（最近）样本的深度。
    pub fn front_z(&self) -> Option<f32> {
        self.samples.first().map(|s| s.z)
    }
}

/// 已解码的 deep 图像。
#[derive(Debug, Clone, PartialEq)]
pub struct DeepImage {
    /// 头部。
    pub header: ExrHeader,
    /// 通道名（顺序 = 文件出现顺序）。
    pub channels: Vec<String>,
    /// 各像素的样本（行主序，行内 x 升序）。
    pub pixels: Vec<DeepPixel>,
}

impl DeepImage {
    /// 取 `(x, y)` 像素（`x`/`y` 为 dataWindow 绝对坐标）。
    pub fn pixel(&self, x: i32, y: i32) -> Option<&DeepPixel> {
        let win = self.header.data_window;
        if x < win.x_min || x > win.x_max || y < win.y_min || y > win.y_max {
            return None;
        }
        let idx = ((y - win.y_min) as usize) * (win.width() as usize) + (x - win.x_min) as usize;
        self.pixels.get(idx)
    }
    /// 深度通道名（`Zfloat` 优先，其次 `Z`）。
    pub fn depth_channel(&self) -> Option<&str> {
        if self.channels.iter().any(|c| c == "Zfloat") {
            Some("Zfloat")
        } else if self.channels.iter().any(|c| c == "Z") {
            Some("Z")
        } else {
            None
        }
    }
}

/// 读 deep 图像。
///
/// # 支持范围（重要）
///
/// **只实现「每个像素至多 1 个样本」的单点 deep**（sample count ≤ 1），
/// 这是 deep 最常见形态（Z/Float AOV 的深度图）与任务书指定的目标形态。
/// 多点 deep（每像素 > 1 样本，如体积雾/半透明）的**数据布局与解包已完整
/// 实现**（本函数照常解出全部样本并写入 [`DeepPixel::samples`]），
/// 但**不做**多重样本特有的语义工作：不做合并/排序重排、不做 ID 通道
/// 解析、不做 AOV 与样本的关联表（`part` 通道语义）。若文件含多点样本，
/// 本函数**如实**返回全部样本（不做任何裁剪），由消费方决定如何处理。
pub fn read_deep_image(data: &[u8]) -> ExrResult<DeepImage> {
    const LOC: &str = "exr::read_deep_image";
    let header = read_header(data)?;
    if !header.deep {
        return Err(struct_err(
            LOC,
            "该 EXR 的版本字段未置 DEEP_DATA 标志",
            "普通 flat EXR 请用 `read_image`",
        ));
    }
    if header.tiled {
        return Err(struct_err(
            LOC,
            "deep tiled EXR 未实现",
            "本模块只实现 deep 扫描线；请用 OIIO 转为 deep scanline",
        ));
    }
    // deep 块压缩：只有「逐扫描线整块压缩」的 None / Rle / Zips / Zip 与本模块的
    // deep 布局兼容（样本数表与通道数据各自整段压缩）。PIZ/PXR24/B44/B44A/DWA
    // 的块内重排面向 flat 的「扫描线 × 通道 × 像素」，与 deep 的「像素序 → 样本序」
    // 布局不兼容，结构化拒绝。
    if !matches!(
        header.compression,
        Compression::None | Compression::Rle | Compression::Zips | Compression::Zip
    ) {
        return Err(comp_err(
            LOC,
            format!(
                "deep 不支持压缩类型 {}（{}）",
                header.compression.code(),
                header.compression.rejection()
            ),
            "deep 只支持 None / Rle / Zips / Zip；请用 OpenEXR 转存",
        ));
    }
    let win = header.data_window;
    let width = win.width() as usize;
    let height = win.height() as usize;
    // deep 扫描线：每块 1 条扫描线（故块数 = 高，无需另算）。
    let count: u32 = header.channels.len() as u32;
    if count == 0 {
        return Err(chan_err(LOC, "deep 图像无通道", "确认文件完整"));
    }
    let depth_idx: Option<usize> = header
        .channels
        .iter()
        .position(|c| c.name == "Zfloat")
        .or_else(|| header.channels.iter().position(|c| c.name == "Z"));

    let mut pixels: Vec<DeepPixel> = Vec::with_capacity(width * height);
    // deep 扫描线每块 = 1 条扫描线，故**每行解码一次**，再按 x 切成像素。
    for y in win.y_min..=win.y_max {
        let block_index = (y - win.y_min) as usize;
        let off = header
            .chunk_offsets
            .get(block_index)
            .copied()
            .ok_or_else(|| {
                struct_err(
                    LOC,
                    format!(
                        "deep 块 {block_index} 的偏移缺失（偏移表 {} 项）",
                        header.chunk_offsets.len()
                    ),
                    "偏移表被截断",
                )
            })?;
        let mut cur = Cursor::new(data, off as usize);
        let blk_y = cur.i32(LOC)?;
        if blk_y != y {
            return Err(struct_err(
                LOC,
                format!("deep 块 {block_index} 的 y = {blk_y}，应为 {y}"),
                "偏移表与块头不一致",
            ));
        }
        // deep 块头：`packedOffsetOfSampleCountTable` + `packedSizeOfDataTable`。
        // 规范语义：前者是样本数表相对**块首**的偏移；后者是样本数表的
        // **打包长度**（无压缩时 = 宽度 × 4；压缩时 = 压缩后长度）。
        // 通道数据区起点 = 两者之和。
        let sample_count_offset = cur.u64(LOC)?;
        let packed_sample_table_size = cur.u64(LOC)?;
        let sc_pos = (off as u64)
            .checked_add(sample_count_offset)
            .ok_or_else(|| struct_err(LOC, "样本数表偏移溢出", "deep 块头损坏"))?;
        let sc_end = sc_pos
            .checked_add(packed_sample_table_size)
            .ok_or_else(|| struct_err(LOC, "样本数表末尾溢出", "deep 块头损坏"))?;
        if sc_end > data.len() as u64 {
            return Err(struct_err(
                LOC,
                format!("deep 样本数表越过文件尾（偏移 {sc_pos}，宽 {width}）"),
                "文件被截断",
            ));
        }
        // 样本数表：无压缩时是 `width` 个 u32 原样；压缩时（Rle/Zips/Zip）是
        // 整表压缩后的字节流，先解回 `width × 4` 字节再逐项读（本模块 deep 写入
        // 与读取**共用同一压缩口径**：逐扫描线一块 + 整表压缩）。
        let want_table = (width as u64).saturating_mul(4);
        let sc_raw: Vec<u8> = if header.compression == Compression::None {
            if packed_sample_table_size != want_table {
                return Err(struct_err(
                    LOC,
                    format!(
                        "deep 样本数表打包长度 {packed_sample_table_size}，宽度 {width} 应为 {want_table}"
                    ),
                    "无压缩 deep 的样本数表长度须为 宽度 × 4",
                ));
            }
            let start = sc_pos as usize;
            let end = sc_end as usize;
            data.get(start..end)
                .ok_or_else(|| struct_err(LOC, "deep 样本数表越界", "文件被截断"))?
                .to_vec()
        } else {
            let start = sc_pos as usize;
            let end = sc_end as usize;
            let packed = data
                .get(start..end)
                .ok_or_else(|| struct_err(LOC, "deep 样本数表越界", "文件被截断"))?;
            decompress_block(header.compression, packed, want_table as usize, &BlockCtx {
                channels: &header.channels,
                width: width.max(1),
                height: 1,
            })?
        };
        // 读本行每像素样本数。
        let mut counts: Vec<u32> = Vec::with_capacity(width);
        for i in 0..width {
            let at = i * 4;
            let b = sc_raw
                .get(at..at + 4)
                .ok_or_else(|| struct_err(LOC, "deep 样本数越界", "样本数表长度与宽度不符"))?;
            counts.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        // 通道数据区起点 = 样本数表末尾（规范：偏移表紧跟样本数表）。
        let data_pos = sc_end as usize;
        if data_pos >= data.len() && !counts.iter().all(|&c| c == 0) {
            return Err(struct_err(
                LOC,
                format!("deep 数据区偏移 {data_pos} 越过文件尾"),
                "文件被截断",
            ));
        }
        // 通道偏移表：count 个 i64。
        //
        // - **无压缩**：偏移相对 `data_pos`（偏移表起点），游标直接索引进文件
        //   `data`（沿用本模块既有 deep 布局，测试构造器同此口径）。
        // - **压缩**：偏移相对**解压后通道数据区起点**，游标索引进解压缓冲 `blk`。
        let mut chan_offsets: Vec<usize> = Vec::with_capacity(count as usize);
        for ci in 0..count as usize {
            let at = data_pos + ci * 8;
            let b = data.get(at..at + 8).ok_or_else(|| {
                struct_err(
                    LOC,
                    format!("deep 通道偏移表越界（通道 {ci}，偏移 {at}）"),
                    "文件被截断",
                )
            })?;
            let rel = i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            if rel < 0 {
                return Err(struct_err(
                    LOC,
                    format!("deep 通道 {ci} 的相对偏移 {rel} 为负"),
                    "负偏移：文件损坏或版本不匹配",
                ));
            }
            chan_offsets.push(rel as usize);
        }
        // 通道数据区：
        //   无压缩 → 文件内 `data_pos` 起原样（含偏移表，游标相对 `data_pos`）；
        //   压缩   → 偏移表之后整段解压为 `blk`（游标相对 `blk` 起点）。
        let table_end = data_pos + count as usize * 8;
        let (chan_base, blk, compressed): (usize, Vec<u8>, bool) =
            if header.compression == Compression::None {
                (data_pos, Vec::new(), false)
            } else {
                // 压缩后数据区的**未压缩**长度由样本数决定：
                // sum(counts) × Σ(每样本通道字节数)。写入器把整段通道数据作为
                // **一个整体**压缩（与样本数表同口径）。
                let bytes_per_sample: usize = header
                    .channels
                    .iter()
                    .map(|c| c.pixel_type.byte_width())
                    .sum();
                let total_samples: usize = counts.iter().map(|&c| c as usize).sum();
                let want_data = total_samples.saturating_mul(bytes_per_sample);
                // 压缩数据区的**上界** = 下一个块的偏移（deep 偏移表有 `块数 + 1`
                // 项，末块的下一个是哨兵）。RLE 等「无显式长度」的压缩需要这个
                // 精确上界，否则会把后续块的字节一起喂进解码器。
                let next_off = header
                    .chunk_offsets
                    .get(block_index + 1)
                    .copied()
                    .unwrap_or(data.len() as u64) as usize;
                let end = next_off.min(data.len()).max(table_end);
                let packed = data
                    .get(table_end..end)
                    .ok_or_else(|| struct_err(LOC, "deep 压缩数据区越界", "文件被截断"))?;
                let back = decompress_block(header.compression, packed, want_data, &BlockCtx {
                    channels: &header.channels,
                    width: width.max(1),
                    height: 1,
                })?;
                (0, back, true)
            };
        // 逐像素解样本。通道平面在块内是「像素序 → 样本序」一维展平，
        // 故每通道一个游标，按读出顺序单调前进（确定性：顺序即文件顺序）。
        let mut cursors: Vec<usize> = chan_offsets;
        // 无压缩时游标索引进文件 `data`（偏移相对 `data_pos`）；压缩时进 `blk`。
        let buf: &[u8] = if compressed { &blk } else { data };
        for px in 0..width {
            let n = counts.get(px).copied().unwrap_or(0);
            let mut samples: Vec<DeepSample> = Vec::with_capacity(n as usize);
            for s in 0..n {
                let mut values: Vec<f32> = Vec::with_capacity(count as usize);
                for (ci, c) in header.channels.iter().enumerate() {
                    let rel = cursors.get(ci).copied().unwrap_or(0);
                    let v = read_channel_value(buf, chan_base + rel, c.pixel_type);
                    if let Some(cur_pos) = cursors.get_mut(ci) {
                        *cur_pos = rel.saturating_add(c.pixel_type.byte_width());
                    }
                    values.push(v);
                }
                let z = depth_idx
                    .and_then(|i| values.get(i).copied())
                    .unwrap_or(f32::NAN);
                samples.push(DeepSample {
                    index: s,
                    z,
                    values,
                });
            }
            pixels.push(DeepPixel {
                x: win.x_min + (px as i32),
                y,
                samples,
            });
        }
    }
    // 通道名列表（文件顺序）。
    let channels: Vec<String> = header.channels.iter().map(|c| c.name.clone()).collect();
    Ok(DeepImage {
        header,
        channels,
        pixels,
    })
}

/// 从字节流读一个通道值（小端）。
fn read_channel_value(data: &[u8], at: usize, pt: PixelType) -> f32 {
    match pt {
        PixelType::Half => {
            let b = data.get(at..at + 2).unwrap_or(&[0, 0]);
            f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
        }
        PixelType::Float => {
            let b = data.get(at..at + 4).unwrap_or(&[0; 4]);
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }
        PixelType::Uint => {
            let b = data.get(at..at + 4).unwrap_or(&[0; 4]);
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32
        }
    }
}

// ===========================================================================
//  七、tiled 图像
// ===========================================================================

/// 读一个 tile（`tx`/`ty` 为 0 基 tile 索引，列/行序固定升序）。
pub fn read_tile(data: &[u8], header: &ExrHeader, tx: u32, ty: u32) -> ExrResult<ExrImage> {
    const LOC: &str = "exr::read_tile";
    let tile = header.tile.ok_or_else(|| {
        struct_err(
            LOC,
            "该 EXR 不是分块图像（无 `tiles` 属性）",
            "普通扫描线 EXR 请用 `read_image`",
        )
    })?;
    let win = header.data_window;
    let cols = tile.tiles_x(&win);
    let rows = tile.tiles_y(&win);
    if tx >= cols || ty >= rows {
        return Err(struct_err(
            LOC,
            format!("tile ({tx}, {ty}) 越界（网格 {cols}×{rows}）"),
            "tile 索引须在 [0, cols) × [0, rows) 内",
        ));
    }
    // 块号 = ty * cols + tx（EXR 规定 tile 索引为行主序）。
    let block = (ty as usize)
        .checked_mul(cols as usize)
        .and_then(|v| v.checked_add(tx as usize))
        .ok_or_else(|| struct_err(LOC, "tile 块号溢出", "使用更小的图像"))?;
    let off = header.chunk_offsets.get(block).copied().ok_or_else(|| {
        struct_err(
            LOC,
            format!(
                "tile 块 {block} 偏移缺失（偏移表 {} 项）",
                header.chunk_offsets.len()
            ),
            "偏移表被截断",
        )
    })?;
    let mut cur = Cursor::new(data, off as usize);
    // tile 块头：tileX(i32) tileY(i32) levelX(i32) levelY(i32) 长度(i32)。
    let _tile_x = cur.i32(LOC)?;
    let _tile_y = cur.i32(LOC)?;
    let level_x = cur.i32(LOC)?;
    let level_y = cur.i32(LOC)?;
    if level_x != 0 || level_y != 0 {
        return Err(struct_err(
            LOC,
            format!("tile ({tx}, {ty}) 的 level = ({level_x}, {level_y})，本模块只读 level 0"),
            "mipmap / ripmap 层级需递归读取；本模块只实现 ONE_LEVEL",
        ));
    }
    let len = cur.i32(LOC)?;
    if len < 0 {
        return Err(struct_err(
            LOC,
            format!("tile ({tx}, {ty}) 负载长度 {len} 为负"),
            "长度字段损坏（负偏移）；确认文件未损坏",
        ));
    }
    let payload = cur.take(len as usize, LOC)?;
    // tile 尺寸：**裁剪到实际覆盖窗口的范围**。
    //
    // ⚠️ 边缘 tile 的 DWA 负载**只含裁剪后**的块，**不是**补齐到完整 tile 尺寸。
    // 实测（官方 OpenEXR 3.1.5 写、64×48 图像、32×32 tile、共 2×2 tile）：
    // ```text
    // chunk0 tile(0,0) dcCnt=16   ← 完整 32 行 ⇒ 4×4 = 16 个 DCT 块
    // chunk1 tile(1,0) dcCnt=16
    // chunk2 tile(0,1) dcCnt=8    ← 裁剪到 16 行 ⇒ 4×2 = 8 个 DCT 块
    // chunk3 tile(1,1) dcCnt=8
    // ```
    // 官方按 `ceil(_height / 8.0)` 计算块行数，其中 `_height` 是**裁剪后**的行数
    // （`ImfDwaCompressor.cpp:689`）。若按完整 tile 高度（32）算，末行 tile 会
    // 索要 16 个 DC 而文件只有 8 个 ⇒ 报「DC 缓冲耗尽」。
    let tx0 = win.x_min + (tx as i32) * (tile.x_size as i32);
    let ty0 = win.y_min + (ty as i32) * (tile.y_size as i32);
    let tw = (win.x_max - tx0 + 1).min(tile.x_size as i32).max(0) as usize;
    let th = (win.y_max - ty0 + 1).min(tile.y_size as i32).max(0) as usize;
    let expect = header
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width())
        .sum::<usize>()
        * tw
        * th;
    let ctx = BlockCtx {
        channels: &header.channels,
        width: tw,
        height: th,
    };
    // 与扫描线路径同理：官方按**块**判断「压不小就原样存」
    // （`ImfOutputFile.cpp:584`），故分块文件同样可能出现未压缩 tile。
    let raw = if header.compression.is_dwa() && len as usize == expect {
        payload.to_vec()
    } else {
        decompress_block(header.compression, payload, expect, &ctx)?
    };
    let mut flat: Vec<(Vec<f32>, PixelType)> = Vec::new();
    unpack_block(&raw, &header.channels, tw, th, &mut flat)?;
    // `flat` 的布局是「逐扫描线 × 逐通道」（EXR 约定），故第 `line` 行第 `ci`
    // 个通道的下标是 `line * 通道数 + ci` —— 不是 `ci`。
    let nch = header.channels.len();

    // 组装成 tile 平面。`tw`/`th` 已是裁剪后尺寸（见上），直接复用。
    let x0 = tx0;
    let y0 = ty0;
    let w = tw as u32;
    let h = th as u32;
    let planes: Vec<ChannelPlane> = header
        .channels
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let mut values = vec![0.0f32; (w as usize) * (h as usize)];
            for y in 0..h as usize {
                let (src, _) = flat
                    .get(y * nch + ci)
                    .cloned()
                    .unwrap_or((Vec::new(), c.pixel_type));
                for x in 0..w as usize {
                    // `src` 已是「第 y 行、第 ci 通道」的一整行（宽 tw），
                    // 故行内下标就是 `x`，不能再乘 tw。
                    let s = x;
                    let d = y * (w as usize) + x;
                    if let (Some(sv), Some(dv)) = (src.get(s), values.get_mut(d)) {
                        *dv = *sv;
                    }
                }
            }
            ChannelPlane {
                name: c.name.clone(),
                pixel_type: c.pixel_type,
                width: w,
                height: h,
                values,
            }
        })
        .collect();
    let mut h2 = header.clone();
    h2.data_window = DataWindow {
        x_min: x0,
        y_min: y0,
        x_max: x0 + w as i32 - 1,
        y_max: y0 + h as i32 - 1,
    };
    Ok(ExrImage { header: h2, planes })
}

/// 读全部分块（按 tile 行主序遍历，结果与逐 tile 读取一致）。
pub fn read_all_tiles(data: &[u8], header: &ExrHeader) -> ExrResult<ExrImage> {
    let tile = header.tile.ok_or_else(|| {
        struct_err(
            "exr::read_all_tiles",
            "该 EXR 不是分块图像",
            "扫描线 EXR 请用 `read_image`",
        )
    })?;
    let win = header.data_window;
    let cols = tile.tiles_x(&win);
    let rows = tile.tiles_y(&win);
    let w = win.width() as usize;
    let h = win.height() as usize;
    let mut planes: Vec<ChannelPlane> = header
        .channels
        .iter()
        .map(|c| ChannelPlane {
            name: c.name.clone(),
            pixel_type: c.pixel_type,
            width: win.width(),
            height: win.height(),
            values: vec![0.0; w * h],
        })
        .collect();
    for ty in 0..rows {
        for tx in 0..cols {
            let t = read_tile(data, header, tx, ty)?;
            let x0 = (t.header.data_window.x_min - win.x_min) as usize;
            let y0 = (t.header.data_window.y_min - win.y_min) as usize;
            for (ci, p) in t.planes.iter().enumerate() {
                if let Some(dst) = planes.get_mut(ci) {
                    for y in 0..p.height as usize {
                        for x in 0..p.width as usize {
                            let s = y * (p.width as usize) + x;
                            let d = (y0 + y) * w + (x0 + x);
                            if let (Some(sv), Some(dv)) = (p.values.get(s), dst.values.get_mut(d)) {
                                *dv = *sv;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut h2 = header.clone();
    h2.data_window = win;
    Ok(ExrImage { header: h2, planes })
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbuf::f32_to_f16;
    // 测试侧自建 DEFLATE dynamic 编码器需要 RFC 1951 §3.2.7 的 `CLCL_ORDER`。
    // 解码实现已下沉到 `prism-bytes`，但**编码器刻意留在本文件独立实现**
    // （不共用写入侧 `exr_write::deflate_zlib`），否则读/写两侧同错会互相掩盖。
    use prism_bytes::inflate::CLCL_ORDER;

    // -----------------------------------------------------------------------
    // 测试脚手架：构造最小合法 EXR 字节流
    // -----------------------------------------------------------------------

    /// 写入一个属性（name/type/size/value）。
    fn put_attr(out: &mut Vec<u8>, name: &str, ty: &str, value: &[u8]) {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(ty.as_bytes());
        out.push(0);
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value);
    }

    /// 组装 `channels` 属性负载。
    fn channels_attr(chs: &[(&str, PixelType)]) -> Vec<u8> {
        let mut v = Vec::new();
        for (name, pt) in chs {
            v.extend_from_slice(name.as_bytes());
            v.push(0);
            v.extend_from_slice(&pt.code().to_le_bytes());
            v.push(0); // pLinear
            v.extend_from_slice(&[0, 0, 0]); // reserved
            v.extend_from_slice(&1i32.to_le_bytes()); // xSampling
            v.extend_from_slice(&1i32.to_le_bytes()); // ySampling
        }
        v.push(0); // 空名终止
        v
    }

    /// 未压缩扫描线 EXR 构造描述。
    struct FlatSpec {
        width: u32,
        height: u32,
        compression: Compression,
        channels: Vec<(String, PixelType)>,
        /// 每通道行主序数据（f32 语义，按 pixel_type 量化写入）。
        data: Vec<Vec<f32>>,
    }

    /// 把 f32 量化成通道原始位模式字节。
    fn quantize(v: f32, pt: PixelType) -> Vec<u8> {
        match pt {
            PixelType::Half => f32_to_f16(v).to_le_bytes().to_vec(),
            PixelType::Float => v.to_le_bytes().to_vec(),
            PixelType::Uint => (v.max(0.0) as u32).to_le_bytes().to_vec(),
        }
    }

    /// 构造一个（未压缩 / RLE / ZIP-with-stored-blocks）的 flat EXR。
    fn build_flat_exr(spec: &FlatSpec) -> Vec<u8> {
        // ① 头部（占位，稍后回填偏移表）。
        let mut header: Vec<u8> = Vec::new();
        header.extend_from_slice(&EXR_MAGIC.to_le_bytes());
        let mut version = 2u32;
        if spec.compression == Compression::Zip || spec.compression == Compression::Zips {
            // 版本号字节恒 2；标志位只在 tiled 时置位。ZIP 不改版本。
        }
        version |= 0;
        header.extend_from_slice(&version.to_le_bytes());
        let chs: Vec<(&str, PixelType)> = spec
            .channels
            .iter()
            .map(|(n, t)| (n.as_str(), *t))
            .collect();
        put_attr(&mut header, "channels", "chlist", &channels_attr(&chs));
        put_attr(
            &mut header,
            "compression",
            "compression",
            &[spec.compression.code()],
        );
        let win = DataWindow::new(spec.width, spec.height);
        let mut dw = Vec::new();
        for v in [win.x_min, win.y_min, win.x_max, win.y_max] {
            dw.extend_from_slice(&v.to_le_bytes());
        }
        put_attr(&mut header, "dataWindow", "box2i", &dw);
        put_attr(&mut header, "displayWindow", "box2i", &dw);
        put_attr(&mut header, "lineOrder", "lineOrder", &[0]);
        header.push(0); // 属性表结束

        // ② 块数据：每块 = y(i32) + len(i32) + 负载。
        let lpb = spec.compression.lines_per_block() as usize;
        let width = spec.width as usize;
        let blocks = (spec.height as usize).div_ceil(lpb);
        // PIZ / PXR24 / B44 需要通道上下文（类型 / pLinear / 宽高）来编码。
        let chans: Vec<ExrChannel> = spec
            .channels
            .iter()
            .map(|(n, pt)| ExrChannel {
                name: n.clone(),
                pixel_type: *pt,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect();
        let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let lines = (spec.height as usize - b * lpb).min(lpb);
            // EXR 布局：逐扫描线、逐通道、逐像素。
            let mut raw: Vec<u8> = Vec::new();
            for line in 0..lines {
                let y = (b * lpb + line) as usize;
                for (ci, (_, pt)) in spec.channels.iter().enumerate() {
                    let plane = spec.data.get(ci).cloned().unwrap_or_default();
                    for x in 0..width {
                        let v = plane.get(y * width + x).copied().unwrap_or(0.0);
                        raw.extend_from_slice(&quantize(v, *pt));
                    }
                }
            }
            let bctx = BlockCtx {
                channels: &chans,
                width,
                height: lines,
            };
            let payload: Vec<u8> = match spec.compression {
                Compression::None => raw,
                Compression::Rle => rle_compress(&raw),
                // 真实 EXR 的 ZIP/ZIPS 块是 **zlib 流**，不是裸 DEFLATE。
                Compression::Zip | Compression::Zips => zlib_stream(&raw),
                // PIZ / PXR24 / B44 / B44A：用 `exr_compress` 的真实编码器产出，
                // 使测试走的是与 OpenEXR 同构的字节流（而非绕过压缩的自造负载）。
                Compression::Piz => crate::exr_compress::piz_compress(&raw, &bctx)
                    .unwrap_or_else(|e| panic!("PIZ 编码失败: {e}")),
                Compression::Pxr24 => crate::exr_compress::pxr24_compress(&raw, &bctx)
                    .unwrap_or_else(|e| panic!("PXR24 编码失败: {e}")),
                Compression::B44 => crate::exr_compress::b44_compress(&raw, &bctx, false)
                    .unwrap_or_else(|e| panic!("B44 编码失败: {e}")),
                Compression::B44A => crate::exr_compress::b44_compress(&raw, &bctx, true)
                    .unwrap_or_else(|e| panic!("B44A 编码失败: {e}")),
                Compression::Dwaa | Compression::Dwab => {
                    // DWA 由 `exr_dwa` 的独立脚手架覆盖（负载格式完全不同）。
                    raw
                }
            };
            let mut chunk: Vec<u8> = Vec::new();
            chunk.extend_from_slice(&((b * lpb) as i32).to_le_bytes());
            chunk.extend_from_slice(&(payload.len() as i32).to_le_bytes());
            chunk.extend_from_slice(&payload);
            chunks.push(chunk);
        }

        // ③ 偏移表 + 块数据。
        let table_off = header.len();
        let table_bytes = blocks * 8;
        let mut body_off = table_off + table_bytes;
        let mut table: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        for c in &chunks {
            table.extend_from_slice(&(body_off as u64).to_le_bytes());
            body.extend_from_slice(c);
            body_off += c.len();
        }
        let mut out = header;
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    /// 生成 DEFLATE stored 块流（自研压缩，仅用于构造测试用 ZIP 负载）。
    fn deflate_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut off = 0usize;
        loop {
            let n = (payload.len() - off).min(65535);
            let last = off + n >= payload.len();
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

    /// 生成 **zlib 流**（RFC 1950）：`0x78 0x01` 头 + stored DEFLATE + Adler-32。
    ///
    /// **为什么测试构造器必须发 zlib 而不是裸 DEFLATE**：真实 OpenEXR 的
    /// `ImfZipCompressor::compress` 调用的是 zlib 的 `compress2()`，产物带
    /// 2 字节 zlib 头（典型 `0x78 0x5E`）与 4 字节 Adler-32 尾。读取侧的
    /// [`zlib_decompress`] 按真实格式校验 `CM & 0xF == 8`，故测试负载也必须是
    /// zlib 流，否则测的是"自造裸 DEFLATE"这条不存在的路径。
    ///
    /// 刻意**不**复用 `exr_write::deflate_zlib`：那会让读取侧测试依赖写入侧实现，
    /// 两边同错时互相掩盖。此处独立实现，使读/写两侧各自被独立验证。
    fn zlib_stream(payload: &[u8]) -> Vec<u8> {
        const MOD: u32 = 65521;
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in payload {
            a = (a + byte as u32) % MOD;
            b = (b + a) % MOD;
        }
        let mut out = Vec::with_capacity(payload.len() + 16);
        out.push(0x78);
        out.push(0x01);
        out.extend_from_slice(&deflate_stored(payload));
        out.extend_from_slice(&((b << 16) | a).to_be_bytes());
        out
    }

    /// 生成 DEFLATE dynamic-Huffman 块流（自研压缩，仅用于构造测试用 ZIP 负载）。
    ///
    /// **为什么需要它**：真实 OpenEXR 写出的 ZIP 块几乎总是 dynamic Huffman（BTYPE=2），
    /// fixed Huffman（BTYPE=1）只在极小负载时出现。若只测 fixed，则 dynamic 代码路径
    /// （HLIT/HDIST/HCLEN 头 + 码长字母表 + 17/18 重复码）将完全未被执行。
    ///
    /// 本函数手写一个**可完整解码**的 dynamic 块（只发纯字面，不发回溯）：
    /// - 码长字母表：给符号 `0`(字面 0)、`2`(字面码长 2)、`1`(字面码长 1)、
    ///   `17`(零游程 3~10)、`18`(零游程 11~138) 分配码长 —— 取 3 个 2 位 +
    ///   2 个 3 位，Kraft 和 = 3·2⁻² + 2·2⁻³ = 3/4 + 1/4 = 1，**恰好完备**。
    /// - 字面表：`'A'`(65)、`'B'`(66)、结束符 256 各 2 位；其余字面码长 0。
    /// - 距离表：符号 `0` 给 1 位码长（本函数不发回溯，故永不解到距离码）。
    ///
    /// 注意码长字母表必须**恰好覆盖本函数用到的每个码长符号**：漏掉任何一个都会让它
    /// 拿到 0 位码字，写出零位宽的码，从而毁掉整条码长序列（实测会解出空输出）。
    ///
    /// **限制**：只支持由 `'A'` / `'B'` 构成的负载（其他字节无码字）。
    fn deflate_dynamic(payload: &[u8]) -> Vec<u8> {
        assert!(
            payload.iter().all(|&b| b == b'A' || b == b'B'),
            "deflate_dynamic 只编码 'A'/'B' 字面（测试用构造器）"
        );
        // ---- 码长字母表（cl alphabet）----
        // 用到的码长符号集合恰为 {0, 1, 2, 17, 18}：字面值 0/2/1 与零游程 17/18。
        // 码长取 3 个 2 位 + 2 个 3 位 → Kraft 和 = 3/4 + 2/8 = 1（完备，前缀码）。
        let mut cl_len = [0u8; 19];
        for &sym in &[0u8, 2, 17] {
            cl_len[sym as usize] = 2;
        }
        for &sym in &[1u8, 18] {
            cl_len[sym as usize] = 3;
        }
        // HCLEN 必须覆盖 CLCL_ORDER 中最后一个被赋值的符号。
        let hclen = CLCL_ORDER
            .iter()
            .rposition(|&s| cl_len[s] != 0)
            .map(|i| i + 1)
            .unwrap_or(4)
            .clamp(4, 19);
        let cl_codes = canonical(&cl_len);

        // ---- 字面表 / 距离表 ----
        let mut lit_len = [0u8; 257];
        lit_len[b'A' as usize] = 2;
        lit_len[b'B' as usize] = 2;
        lit_len[256] = 2; // 块结束符
        let dist_len = [1u8; 1];
        let lit_codes = canonical(&lit_len);

        // ---- 写块头 ----
        let mut w = BitWriter::default();
        w.push(1, 1); // BFINAL = 1
        w.push(2, 2); // BTYPE  = 2（dynamic）
        w.push(0, 5); // HLIT  - 257 = 0 → 257 个字面码（符号 0..=256）
        w.push(0, 5); // HDIST - 1   = 0 → 1 个距离码
        w.push((hclen - 4) as u32, 4);
        // 码长「码长」表：按 CLCL_ORDER 顺序，各写 3 位。
        for &s in CLCL_ORDER.iter().take(hclen) {
            w.push(cl_len[s] as u32, 3);
        }

        // ---- 写字面表 + 距离表的码长序列（258 项）----
        // 序列语义（RFC 1951 §3.2.7）：0=字面 0，1..15=字面码长，
        // 17=零游程 3~10，18=零游程 11~138。
        let all: Vec<u8> = lit_len
            .iter()
            .copied()
            .chain(dist_len.iter().copied())
            .collect();
        for (sym, extra, bits) in encode_cl_lengths(&all) {
            let c = cl_codes[sym as usize];
            w.push_msb(c.0, c.1);
            if bits > 0 {
                w.push(extra, bits);
            }
        }

        // ---- 写数据（纯字面）----
        for &b in payload {
            let c = lit_codes[b as usize];
            w.push_msb(c.0, c.1);
        }
        let end = lit_codes[256];
        w.push_msb(end.0, end.1);
        w.finish()
    }

    /// 把「码长数组」压成 `(码长符号, 附加值, 附加位数)` 序列（零游程用 17/18）。
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

    /// 由码长数组求 canonical Huffman 码字（RFC 1951 §3.2.2）。
    ///
    /// 返回 `Vec<(code, bit_len)>`，下标即符号号；码长 0 的符号填 `(0, 0)`。
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

    /// DEFLATE 位写入器。
    ///
    /// **两种位序都要区分**（这是 DEFLATE 最易错之处，RFC 1951 §3.1.1）：
    /// - 普通整数字段（HLIT/HDIST/HCLEN、码长附加值、拷贝长度/距离附加值）
    ///   按 **LSB 优先**打包 → [`BitWriter::push`]；
    /// - **Huffman 码字本身按高位在前**写出 → [`BitWriter::push_msb`]。
    #[derive(Default)]
    struct BitWriter {
        out: Vec<u8>,
        cur: u32,
        nbit: u32,
    }

    impl BitWriter {
        /// 写 `n` 位整数（`n ≤ 32`），**LSB 优先**（字节内低位先填）。
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

        /// 写一个 Huffman 码字，**高位在前**（码字的最高位先入流）。
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

        /// 收尾：末字节高位补 0。
        fn finish(mut self) -> Vec<u8> {
            if self.nbit > 0 {
                self.out.push(self.cur as u8);
            }
            self.out
        }
    }

    /// 构造一个分块（tiled, ONE_LEVEL, 无压缩）EXR。
    fn build_tiled_exr(
        width: u32,
        height: u32,
        tile_w: u32,
        tile_h: u32,
        chs: &[(&str, PixelType)],
        plane: &[f32],
    ) -> Vec<u8> {
        let cols = (width + tile_w - 1) / tile_w;
        let rows = (height + tile_h - 1) / tile_h;
        let mut header: Vec<u8> = Vec::new();
        header.extend_from_slice(&EXR_MAGIC.to_le_bytes());
        header.extend_from_slice(&(2u32 | version_flags::TILED).to_le_bytes());
        put_attr(&mut header, "channels", "chlist", &channels_attr(chs));
        put_attr(
            &mut header,
            "compression",
            "compression",
            &[Compression::None.code()],
        );
        let win = DataWindow::new(width, height);
        let mut dw = Vec::new();
        for v in [win.x_min, win.y_min, win.x_max, win.y_max] {
            dw.extend_from_slice(&v.to_le_bytes());
        }
        put_attr(&mut header, "dataWindow", "box2i", &dw);
        put_attr(&mut header, "displayWindow", "box2i", &dw);
        put_attr(&mut header, "lineOrder", "lineOrder", &[0]);
        let mut tiles = Vec::new();
        tiles.extend_from_slice(&tile_w.to_le_bytes());
        tiles.extend_from_slice(&tile_h.to_le_bytes());
        tiles.push(0); // level mode = ONE_LEVEL
        put_attr(&mut header, "tiles", "tiledesc", &tiles);
        header.push(0);

        let mut chunks: Vec<Vec<u8>> = Vec::new();
        for ty in 0..rows {
            for tx in 0..cols {
                // 【W13 修正】边缘 tile 的负载**按裁剪后的尺寸**写，**不补齐**
                // 到完整 tile 尺寸。权威依据：官方写出的 64×48 图像 /
                // 32×32 tile，tile(0,1) 的 DWA 头 `dcUncompressedCount = 8`
                // = `ceil(32/8) × ceil(16/8)`（16 = 48 − 32，裁剪后的行数）。
                // 读取侧 `read_tile` 也按裁剪尺寸解，两边必须一致。
                let cw = (width - tx * tile_w).min(tile_w);
                let ch = (height - ty * tile_h).min(tile_h);
                let mut raw: Vec<u8> = Vec::new();
                for line in 0..ch {
                    for (_, pt) in chs.iter() {
                        for col in 0..cw {
                            let gx = tx * tile_w + col;
                            let gy = ty * tile_h + line;
                            // 裁剪后必然都在窗口内。
                            let v = plane
                                .get((gy * width + gx) as usize)
                                .copied()
                                .unwrap_or(0.0);
                            raw.extend_from_slice(&quantize(v, *pt));
                        }
                    }
                }
                let mut chunk: Vec<u8> = Vec::new();
                chunk.extend_from_slice(&(tx as i32).to_le_bytes());
                chunk.extend_from_slice(&(ty as i32).to_le_bytes());
                chunk.extend_from_slice(&0i32.to_le_bytes()); // levelX
                chunk.extend_from_slice(&0i32.to_le_bytes()); // levelY
                chunk.extend_from_slice(&(raw.len() as i32).to_le_bytes());
                chunk.extend_from_slice(&raw);
                chunks.push(chunk);
            }
        }
        let table_bytes = chunks.len() * 8;
        let mut body_off = header.len() + table_bytes;
        let mut table: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        for c in &chunks {
            table.extend_from_slice(&(body_off as u64).to_le_bytes());
            body.extend_from_slice(c);
            body_off += c.len();
        }
        let mut out = header;
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    /// 构造一个 deep 扫描线 EXR（每像素至多 `n` 个样本）。
    fn build_deep_exr(width: u32, height: u32, z: &[f32], samples_per_pixel: u32) -> Vec<u8> {
        let mut header: Vec<u8> = Vec::new();
        header.extend_from_slice(&EXR_MAGIC.to_le_bytes());
        header.extend_from_slice(&(2u32 | version_flags::DEEP_DATA).to_le_bytes());
        put_attr(
            &mut header,
            "channels",
            "chlist",
            &channels_attr(&[("Zfloat", PixelType::Float), ("id", PixelType::Uint)]),
        );
        put_attr(
            &mut header,
            "compression",
            "compression",
            &[Compression::None.code()],
        );
        let win = DataWindow::new(width, height);
        let mut dw = Vec::new();
        for v in [win.x_min, win.y_min, win.x_max, win.y_max] {
            dw.extend_from_slice(&v.to_le_bytes());
        }
        put_attr(&mut header, "dataWindow", "box2i", &dw);
        put_attr(&mut header, "displayWindow", "box2i", &dw);
        put_attr(&mut header, "lineOrder", "lineOrder", &[0]);
        header.push(0);

        let mut chunks: Vec<Vec<u8>> = Vec::new();
        // 无压缩时块内布局（严格对齐 OpenEXR deep 扫描线规范）：
        //   [0, 20)                块头：y(i32) + 样本数表偏移(u64) + 样本数表长度(u64)
        //   [20, 20+4W)            样本数表：u32 × W
        //   [20+4W, 20+4W+8C)      通道偏移表：i64 × C（相对**样本数表末尾**）
        //   [20+4W+8C, …)          通道数据：逐通道连续（每通道内：像素序 → 样本序）
        let count_bytes = (width as usize) * 4;
        let nch = 2usize; // Zfloat + id
        let data_start = 20u64 + count_bytes as u64;
        let table_bytes = (width as usize * samples_per_pixel as usize * 4) as u64;
        // 通道偏移相对**偏移表末尾**（即数据区起点）—— 故通道 0 偏移 = 偏移表长度。
        let chan_z_rel = (nch * 8) as u64;
        let chan_id_rel = chan_z_rel + table_bytes;
        for y in 0..height as usize {
            let mut chunk: Vec<u8> = Vec::new();
            chunk.extend_from_slice(&(y as i32).to_le_bytes());
            chunk.extend_from_slice(&20u64.to_le_bytes()); // 样本数表偏移（相对块首）
            chunk.extend_from_slice(&(count_bytes as u64).to_le_bytes()); // 样本数表长度
            for _ in 0..width {
                chunk.extend_from_slice(&samples_per_pixel.to_le_bytes());
            }
            chunk.extend_from_slice(&chan_z_rel.to_le_bytes());
            chunk.extend_from_slice(&chan_id_rel.to_le_bytes());
            for i in 0..width as usize {
                for s in 0..samples_per_pixel as usize {
                    let idx = y * width as usize + i;
                    let v = z
                        .get(idx * samples_per_pixel as usize + s)
                        .copied()
                        .unwrap_or(0.0);
                    chunk.extend_from_slice(&v.to_le_bytes());
                }
            }
            for i in 0..(width as usize * samples_per_pixel as usize) {
                chunk.extend_from_slice(&(1000u32 + i as u32).to_le_bytes());
            }
            debug_assert_eq!(chunk.len() as u64, data_start + 8 * 2 + table_bytes * 2);
            chunks.push(chunk);
        }
        // deep 偏移表比块数多 1 个哨兵条目。
        let table_bytes = (chunks.len() + 1) * 8;
        let mut body_off = header.len() + table_bytes;
        let mut table: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        for c in &chunks {
            table.extend_from_slice(&(body_off as u64).to_le_bytes());
            body.extend_from_slice(c);
            body_off += c.len();
        }
        table.extend_from_slice(&(body_off as u64).to_le_bytes()); // 哨兵
        let mut out = header;
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    // -----------------------------------------------------------------------
    // 1) 头解析 + 基础扫描线
    // -----------------------------------------------------------------------

    #[test]
    fn minimal_scanline_exr_parses_dimensions_and_channels() {
        let spec = FlatSpec {
            width: 3,
            height: 2,
            compression: Compression::None,
            channels: vec![
                ("R".into(), PixelType::Half),
                ("G".into(), PixelType::Half),
                ("B".into(), PixelType::Half),
            ],
            data: vec![
                (0..6).map(|i| i as f32 * 0.5).collect(),
                (0..6).map(|i| i as f32 * 0.25).collect(),
                (0..6).map(|i| i as f32 * 0.125).collect(),
            ],
        };
        let bytes = build_flat_exr(&spec);
        let h = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(h.version & 0xFF, 2);
        assert!(!h.tiled && !h.deep && !h.multipart);
        assert_eq!(h.compression, Compression::None);
        assert_eq!((h.data_window.width(), h.data_window.height()), (3, 2));
        assert_eq!(h.channels.len(), 3);
        assert_eq!(h.channels[0].name, "R");
        assert_eq!(h.channels[2].pixel_type, PixelType::Half);
        assert_eq!(h.block_count(), 2); // 无压缩 → 每块 1 行

        let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.planes.len(), 3);
        // half 精度：0.5 精确；0.125 精确。
        assert_eq!(
            img.plane("R").unwrap().values,
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        );
        assert_eq!(img.plane("G").unwrap().values[4], 1.0);
        assert_eq!(img.plane("B").unwrap().values[4], 0.5);
        // 通道遍历顺序 = 文件出现顺序（确定性）。
        let names: Vec<&str> = img.planes.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["R", "G", "B"]);
    }

    #[test]
    fn multi_channel_aov_half_and_float_precision() {
        // 6 通道 AOV：R/G/B(A=half) + Z(float) + normal.x(float) + id(uint)
        let spec = FlatSpec {
            width: 2,
            height: 1,
            compression: Compression::None,
            channels: vec![
                ("R".into(), PixelType::Half),
                ("G".into(), PixelType::Half),
                ("B".into(), PixelType::Half),
                ("Z".into(), PixelType::Float),
                ("normal.x".into(), PixelType::Float),
                ("id".into(), PixelType::Uint),
            ],
            data: vec![
                vec![1.0, 0.25],
                vec![0.5, 0.125],
                vec![0.75, 0.0],
                vec![100.5, 200.25], // float：须逐位精确
                vec![-1.5, 3.25],    // float：含负值
                vec![7.0, 65535.0],  // uint
            ],
        };
        let bytes = build_flat_exr(&spec);
        let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.planes.len(), 6);
        let names: Vec<&str> = img.planes.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["R", "G", "B", "Z", "normal.x", "id"]);
        // half 精确值。
        assert_eq!(img.plane("R").unwrap().values, vec![1.0, 0.25]);
        // float 逐位精确（含非 half 可表示值 100.5 / 200.25 —— 这正是
        // 必须区分 half/float 两档精度的原因）。
        assert_eq!(
            img.plane("Z").unwrap().values[0].to_bits(),
            100.5f32.to_bits()
        );
        assert_eq!(
            img.plane("Z").unwrap().values[1].to_bits(),
            200.25f32.to_bits()
        );
        assert_eq!(
            img.plane("normal.x").unwrap().values[0].to_bits(),
            (-1.5f32).to_bits()
        );
        assert_eq!(img.plane("id").unwrap().values, vec![7.0, 65535.0]);
        // 两种精度的位宽差异。
        assert_eq!(img.plane("R").unwrap().pixel_type, PixelType::Half);
        assert_eq!(img.plane("Z").unwrap().pixel_type, PixelType::Float);
    }

    // -----------------------------------------------------------------------
    // 2) 压缩
    // -----------------------------------------------------------------------

    #[test]
    fn no_compression_and_rle_and_zip_all_round_trip() {
        let w = 4u32;
        let h = 6u32; // > ZIP 的 16 行块？不，6 < 16；另测 h=20 覆盖多块
        for comp in [
            Compression::None,
            Compression::Rle,
            Compression::Zip,
            Compression::Zips,
        ] {
            let data: Vec<Vec<f32>> = (0..2)
                .map(|c| {
                    (0..(w * h) as usize)
                        .map(|i| (i as f32) * 0.5 - (c as f32) * 3.0)
                        .collect()
                })
                .collect();
            let spec = FlatSpec {
                width: w,
                height: h,
                compression: comp,
                channels: vec![
                    ("R".into(), PixelType::Float),
                    ("G".into(), PixelType::Float),
                ],
                data,
            };
            let bytes = build_flat_exr(&spec);
            let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e} · comp={}", comp.code()));
            assert_eq!(img.planes.len(), 2, "comp={}", comp.code());
            for ci in 0..2 {
                for i in 0..(w * h) as usize {
                    let want = (i as f32) * 0.5 - (ci as f32) * 3.0;
                    assert_eq!(
                        img.planes[ci].values[i].to_bits(),
                        want.to_bits(),
                        "comp={} 通道 {ci} 像素 {i}",
                        comp.code()
                    );
                }
            }
        }
    }

    #[test]
    fn zip_multi_block_16_lines_matches_full_read() {
        // 20 行 → ZIP 每块 16 行 ⇒ 2 块（16 + 4），验证多块拼接。
        let w = 3u32;
        let h = 20u32;
        let plane: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.25).collect();
        let spec = FlatSpec {
            width: w,
            height: h,
            compression: Compression::Zip,
            channels: vec![("Y".into(), PixelType::Float)],
            data: vec![plane.clone()],
        };
        let bytes = build_flat_exr(&spec);
        let hh = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(hh.compression.lines_per_block(), 16);
        assert_eq!(hh.block_count(), 2);
        let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(img.plane("Y").unwrap().values, plane);
    }

    #[test]
    fn rle_codec_round_trips_and_is_self_consistent() {
        // 压缩 → 解压往返（自研 codec 的对逆性）。
        for n in [0usize, 1, 2, 3, 127, 128, 129, 1000] {
            let payload: Vec<u8> = (0..n).map(|i| ((i / 3) % 7) as u8).collect();
            let comp = rle_compress(&payload);
            let back = rle_decompress(&comp).unwrap_or_else(|e| panic!("{e} · n={n}"));
            assert_eq!(back, payload, "RLE 往返失败 n={n}");
        }
        // 纯常量块应压成「行程 + 值」序列（每段最多 127 字节）。
        // 200 字节常量 = 127 + 73 两段行程 → 4 字节长度头 + 2×2 字节。
        let flat = vec![7u8; 200];
        let comp = rle_compress(&flat);
        assert_eq!(
            comp.len(),
            4 + 2 * 2,
            "200 字节常量 → 2 段行程（127 + 73），共 4+4 字节"
        );
        assert_eq!(
            rle_decompress(&comp).unwrap_or_else(|e| panic!("{e}")),
            flat
        );
        // 127 字节常量恰好一段行程。
        let flat1 = vec![7u8; 127];
        let comp1 = rle_compress(&flat1);
        assert_eq!(comp1.len(), 4 + 2, "127 字节常量 → 1 段行程");
        assert_eq!(
            rle_decompress(&comp1).unwrap_or_else(|e| panic!("{e}")),
            flat1
        );
        // 字面段上限 128 字节：129 个互异字节 = 128 字面 + 1 字面。
        let diff: Vec<u8> = (0..129u16).map(|i| i as u8).collect();
        let cd = rle_compress(&diff);
        assert_eq!(rle_decompress(&cd).unwrap_or_else(|e| panic!("{e}")), diff);
    }

    #[test]
    fn inflate_stored_and_fixed_huffman_known_streams() {
        // ① stored 往返。
        let payload: Vec<u8> = (0..200u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(
            inflate_raw(&deflate_stored(&payload)).unwrap_or_else(|e| panic!("{e}")),
            payload
        );
        // ② fixed Huffman 手工构造：BFINAL=1, BTYPE=01，固定表下 'A''B''C' + 结束码。
        // 固定表 0..=143 → 8 位码（码字 = sym + 0x30，MSB 优先入流）。
        let mut bytes: Vec<u8> = Vec::new();
        let mut cur = 0x03u32; // BFINAL=1, BTYPE=01
        let mut nbit = 3u32;
        let put = |code: u16, n: u32, cur: &mut u32, nbit: &mut u32, out: &mut Vec<u8>| {
            for i in (0..n).rev() {
                if (code >> i) & 1 == 1 {
                    *cur |= 1 << *nbit;
                }
                *nbit += 1;
                if *nbit == 8 {
                    out.push(*cur as u8);
                    *cur = 0;
                    *nbit = 0;
                }
            }
        };
        for sym in [65u16, 66, 67] {
            put(sym + 0x30, 8, &mut cur, &mut nbit, &mut bytes);
        }
        put(0, 7, &mut cur, &mut nbit, &mut bytes); // 256 = 块结束
        if nbit > 0 {
            bytes.push(cur as u8);
        }
        assert_eq!(
            inflate_raw(&bytes).unwrap_or_else(|e| panic!("{e}")),
            b"ABC"
        );
        // ⑤ 非法块类型 3 → 结构化错误。
        let bad = [0x07u8, 0, 0, 0, 0];
        assert_eq!(inflate_raw(&bad).unwrap_err().code, "EXR2001");
        // ⑥ 截断流 → 结构化错误（不 panic）。
        assert_eq!(inflate_raw(&[0x01]).unwrap_err().code, "EXR2001");
        assert_eq!(inflate_raw(&[]).unwrap_err().code, "EXR2001");
    }

    /// **迁移回归守卫**：fixed Huffman 的**回溯**（长度-距离对）必须能解。
    ///
    /// 旧实现（`exr.rs` 内自研的那份）用 `fixed_tables()` 只建了**一张**
    /// 288 项字面表，并把它同时当字面表与距离表用——RFC 1951 §3.2.6 规定
    /// fixed 距离码是**独立的 5 位定长表**（30 项）。于是 fixed 块里**任何**
    /// 回溯都会解出错误符号并报 `EXR2001`。本测试在迁移后必须通过。
    #[test]
    fn inflate_fixed_huffman_backreference_decodes() {
        // BFINAL=1, BTYPE=01；字面 'A'；长度码 257（长度 3）；距离码 0（距离 1）；结束码 256。
        let mut bytes: Vec<u8> = Vec::new();
        let mut cur = 0x03u32; // BFINAL=1, BTYPE=01
        let mut nbit = 3u32;
        let put = |code: u16, n: u32, cur: &mut u32, nbit: &mut u32, out: &mut Vec<u8>| {
            for i in (0..n).rev() {
                if (code >> i) & 1 == 1 {
                    *cur |= 1 << *nbit;
                }
                *nbit += 1;
                if *nbit == 8 {
                    out.push(*cur as u8);
                    *cur = 0;
                    *nbit = 0;
                }
            }
        };
        put((b'A' as u16) + 0x30, 8, &mut cur, &mut nbit, &mut bytes); // 字面 'A'
        put(1, 7, &mut cur, &mut nbit, &mut bytes); // 长度码 257（256..279 → 7 位）
        put(0, 5, &mut cur, &mut nbit, &mut bytes); // 距离码 0（fixed 距离恒 5 位）
        put(0, 7, &mut cur, &mut nbit, &mut bytes); // 256 块结束
        if nbit > 0 {
            bytes.push(cur as u8);
        }
        // 'A' + 长度 3/距离 1 → "AAAA"
        assert_eq!(
            inflate_raw(&bytes).unwrap_or_else(|e| panic!("fixed 回溯必须可解: {e}")),
            b"AAAA"
        );
    }

    /// dynamic Huffman（BTYPE=2）路径：码长字母表 + 17/18 零游程 + canonical 解码。
    #[test]
    fn inflate_dynamic_huffman_round_trips_through_cl_alphabet() {
        // 遍历多种长度组合，覆盖 18（≥11 零）、17（3~10 零）、0（1~2 零）三条分支，
        // 以及「零游程跨越 lit/dist 表边界」的情形（dist 表只有 1 项）。
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
            let stream = deflate_dynamic(&payload);
            let got = inflate_raw(&stream).unwrap_or_else(|e| panic!("dynamic {a}+{b}: {e}"));
            assert_eq!(got, payload, "dynamic Huffman 往返失真（{a}+{b}）");
        }
        // 该路径确实产出了 dynamic 块（首块 BTYPE=2）—— 防止测试退化到别的编码。
        let s = deflate_dynamic(b"AAB");
        assert_eq!(
            s[0] & 0x06,
            0x04,
            "BTYPE 应为 2（dynamic），实际首字节 {:#04x}",
            s[0]
        );
        // 动态 Huffman 块被截断 → 结构化错误，不 panic。
        let full = deflate_dynamic(&[b'A'; 40]);
        for cut in 1..full.len() {
            let r = inflate_raw(&full[..cut]);
            assert!(r.is_err() || r.is_ok(), "截断到 {cut} 字节不得 panic");
        }
        assert_eq!(
            inflate_raw(&full[..3]).unwrap_err().code,
            "EXR2001",
            "dynamic 头被截断应结构化报错"
        );
    }

    #[test]
    fn every_known_compression_byte_is_now_supported() {
        // 十种压缩（字节 0..=9）全部真实解码。历史上 Piz/Pxr24/B44/B44A 曾被拒，
        // 现由 `exr_compress` 承担（见 `exr_compress` 模块的 round-trip 对拍测试）。
        for code in 0u8..=9 {
            let comp = Compression::from_code(code).unwrap_or_else(|| panic!("字节 {code}"));
            assert!(
                comp.is_supported(),
                "压缩字节 {code}（{}）应已实现",
                comp.rejection()
            );
            // `rejection()` 现在是可读诊断，不是"是否支持"的判据：
            // 每种都必须给出非空且互不相同的算法摘要。
            let text = comp.rejection();
            assert!(!text.is_empty(), "压缩字节 {code} 缺算法摘要");
            assert!(
                !text.contains("未实现"),
                "压缩字节 {code} 已实现，摘要不应再称未实现：{text}"
            );
        }
        // 有损 / 无损分类必须与实现一致（转存时决定能否保位级一致）。
        assert!(!Compression::None.is_lossy());
        assert!(!Compression::Rle.is_lossy());
        assert!(!Compression::Zip.is_lossy());
        assert!(!Compression::Piz.is_lossy(), "PIZ 全整数运算，应无损");
        assert!(Compression::Pxr24.is_lossy(), "PXR24 的 float 降到 24bit");
        assert!(Compression::B44.is_lossy());
        assert!(Compression::B44A.is_lossy());
        // 摘要互不相同（否则诊断信息无区分度）。
        let all: Vec<String> = (0u8..=9)
            .filter_map(Compression::from_code)
            .map(|c| c.rejection())
            .collect();
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "十种压缩的算法摘要应互不相同");
    }

    #[test]
    fn unknown_compression_byte_is_structurally_rejected() {
        // 未知压缩字节（10）→ 头解析即结构化拒绝（EXR 规范只定义 0..=9）。
        let mut bad = build_flat_exr(&FlatSpec {
            width: 1,
            height: 1,
            compression: Compression::None,
            channels: vec![("R".into(), PixelType::Float)],
            data: vec![vec![0.0]],
        });
        // 找到 compression 属性字节并改成 10。
        if let Some(p) = bad.windows(11).position(|w| w == b"compression") {
            let vpos = p + 12; // name(10+1) + type("compression"+1)... 用定长定位：
                               // name = "compression\0" = 12 字节，type = "compression\0" = 12 字节，
                               // size = 4 字节 → 值在 12+12+4 = 28。
            if let Some(s) = bad.get_mut(vpos + 16) {
                *s = 10;
            }
        }
        let err = read_header(&bad).unwrap_err();
        assert_eq!(err.code, "EXR1001", "{err}");
        // 结构化错误三段式齐全：code / location / problem / fix。
        assert!(
            err.location.starts_with("exr::"),
            "location 须指向 exr 模块（各函数私有 LOC 前缀），实得 {}",
            err.location
        );
        assert!(!err.problem.is_empty(), "problem 不得为空");
        assert!(!err.fix.is_empty(), "fix 不得为空（须给可行动建议）");
    }

    #[test]
    fn truncation_across_every_compression_is_structurally_rejected() {
        // 每种压缩的块被逐字节截断时，都必须返回结构化错误而非 panic（C18 铁律）。
        // 负载用真实编码器产出，故覆盖的是"真实字节流"的截断路径。
        let cases: [(Compression, PixelType); 4] = [
            (Compression::Piz, PixelType::Half),
            (Compression::Pxr24, PixelType::Float),
            (Compression::B44, PixelType::Half),
            (Compression::B44A, PixelType::Half),
        ];
        for (comp, pt) in cases {
            let spec = FlatSpec {
                width: 8,
                height: comp.lines_per_block() as u32,
                compression: comp,
                channels: vec![("R".into(), pt)],
                data: vec![(0..(8 * comp.lines_per_block()) as usize)
                    .map(|i| (i % 17) as f32 * 0.25)
                    .collect()],
            };
            let bytes = build_flat_exr(&spec);
            let full = read_image(&bytes).unwrap_or_else(|e| panic!("{}: {e}", comp.rejection()));
            // 头必须能解析（压缩字节合法）。
            assert_eq!(
                read_header(&bytes).unwrap_or_else(|e| panic!("{e}")).compression,
                comp
            );
            let _ = full;
            // 逐位置截断：从头中部一路切到尾部，每个前缀都必须结构化拒绝。
            for cut in (1..bytes.len()).step_by(7) {
                let r = read_image(&bytes[..cut]);
                assert!(
                    r.is_err(),
                    "{}: 截断到 {cut} 字节竟成功读出？",
                    comp.rejection()
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3) 分块读取
    // -----------------------------------------------------------------------

    #[test]
    fn region_read_matches_full_read() {
        let w = 6u32;
        let h = 5u32;
        let plane: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.5).collect();
        let spec = FlatSpec {
            width: w,
            height: h,
            compression: Compression::None,
            channels: vec![("Y".into(), PixelType::Float)],
            data: vec![plane.clone()],
        };
        let bytes = build_flat_exr(&spec);
        let h0 = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let full = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));

        // 取 4 个不同区域（含对齐块边界与不对齐块边界）。
        let regions = [
            DataWindow {
                x_min: 0,
                y_min: 0,
                x_max: 1,
                y_max: 1,
            },
            DataWindow {
                x_min: 2,
                y_min: 1,
                x_max: 4,
                y_max: 3,
            },
            DataWindow {
                x_min: 1,
                y_min: 4,
                x_max: 5,
                y_max: 4,
            },
            DataWindow {
                x_min: 5,
                y_min: 0,
                x_max: 5,
                y_max: 4,
            },
        ];
        for r in regions {
            let sub = read_flat_region(&bytes, &h0, &r).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(sub.planes.len(), 1);
            let p = sub.plane("Y").unwrap();
            assert_eq!((p.width, p.height), (r.width(), r.height()));
            for y in r.y_min..=r.y_max {
                for x in r.x_min..=r.x_max {
                    let want =
                        full.plane("Y").unwrap().values[((y as u32) * w + x as u32) as usize];
                    let got = p.values[((y - r.y_min) * r.width() as i32 + (x - r.x_min)) as usize];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "({x},{y}) 区域读值须与全量一致"
                    );
                }
            }
        }
    }

    #[test]
    fn zip_region_read_equals_full_read() {
        // ZIP（16 行块）下区域读：验证块索引 + 块内行丢弃逻辑。
        let w = 4u32;
        let h = 20u32;
        let plane: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.25).collect();
        let spec = FlatSpec {
            width: w,
            height: h,
            compression: Compression::Zip,
            channels: vec![("Y".into(), PixelType::Float)],
            data: vec![plane.clone()],
        };
        let bytes = build_flat_exr(&spec);
        let h0 = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let full = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        // 区域跨第 0 块（0..15）与第 1 块（16..19）。
        let r = DataWindow {
            x_min: 1,
            y_min: 14,
            x_max: 3,
            y_max: 18,
        };
        let sub = read_flat_region(&bytes, &h0, &r).unwrap_or_else(|e| panic!("{e}"));
        let p = sub.plane("Y").unwrap();
        for y in r.y_min..=r.y_max {
            for x in r.x_min..=r.x_max {
                let want = full.plane("Y").unwrap().values[((y as u32) * w + x as u32) as usize];
                let got = p.values[((y - r.y_min) * r.width() as i32 + (x - r.x_min)) as usize];
                assert_eq!(got.to_bits(), want.to_bits(), "({x},{y})");
            }
        }
        // 越界区域 → 结构化错误。
        let oob = DataWindow {
            x_min: 0,
            y_min: 0,
            x_max: 99,
            y_max: 0,
        };
        assert_eq!(
            read_flat_region(&bytes, &h0, &oob).unwrap_err().code,
            "EXR1001"
        );
        // 反向区域 → 结构化错误。
        let inv = DataWindow {
            x_min: 3,
            y_min: 3,
            x_max: 1,
            y_max: 1,
        };
        assert_eq!(
            read_flat_region(&bytes, &h0, &inv).unwrap_err().code,
            "EXR1001"
        );
    }

    #[test]
    fn tiled_image_reads_per_tile_and_whole() {
        let w = 5u32;
        let h = 3u32;
        let tw = 2u32;
        let th = 2u32;
        let plane: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.5).collect();
        let bytes = build_tiled_exr(w, h, tw, th, &[("Y", PixelType::Float)], &plane);
        let h0 = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert!(h0.tiled);
        let tile = h0.tile.unwrap();
        assert_eq!((tile.x_size, tile.y_size), (2, 2));
        assert_eq!(tile.level_mode, 0);
        // 网格 = 3 列 × 2 行；边缘 tile 被裁剪。
        assert_eq!(tile.tiles_x(&h0.data_window), 3);
        assert_eq!(tile.tiles_y(&h0.data_window), 2);
        // 逐 tile 读，与全量一致。
        let whole = read_all_tiles(&bytes, &h0).unwrap_or_else(|e| panic!("{e}"));
        let wp = whole.plane("Y").unwrap();
        assert_eq!((wp.width, wp.height), (w, h));
        for ty in 0..2u32 {
            for tx in 0..3u32 {
                let t = read_tile(&bytes, &h0, tx, ty).unwrap_or_else(|e| panic!("{e}"));
                let p = t.plane("Y").unwrap();
                let x0 = tx * tw;
                let y0 = ty * th;
                let cw = (w - x0).min(tw);
                let ch = (h - y0).min(th);
                assert_eq!((p.width, p.height), (cw, ch), "tile ({tx},{ty})");
                for y in 0..ch as usize {
                    for x in 0..cw as usize {
                        assert_eq!(
                            p.values[y * cw as usize + x].to_bits(),
                            plane[((y0 + y as u32) * w + x0 + x as u32) as usize].to_bits(),
                            "tile ({tx},{ty}) 局部 ({x},{y})"
                        );
                    }
                }
            }
        }
        assert_eq!(wp.values, plane, "全量 tile 拼接须与源数据一致");
        // tile 越界 → 结构化错误。
        assert_eq!(read_tile(&bytes, &h0, 9, 0).unwrap_err().code, "EXR1001");
        // 对分块文件调 read_image → 结构化拒绝（提示用 read_tile）。
        assert_eq!(read_image(&bytes).unwrap_err().code, "EXR1001");
    }

    // -----------------------------------------------------------------------
    // 4) deep image
    // -----------------------------------------------------------------------

    #[test]
    fn deep_single_sample_z_float_reads_all_pixels() {
        let w = 3u32;
        let h = 2u32;
        let z: Vec<f32> = (0..(w * h) as usize).map(|i| 10.0 + i as f32).collect();
        let bytes = build_deep_exr(w, h, &z, 1);
        let img = read_deep_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert!(img.header.deep);
        assert_eq!(img.channels, vec!["Zfloat".to_string(), "id".to_string()]);
        assert_eq!(img.depth_channel(), Some("Zfloat"));
        assert_eq!(img.pixels.len(), (w * h) as usize);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let p = img.pixel(x, y).unwrap_or_else(|| panic!("缺 ({x},{y})"));
                assert_eq!(p.samples.len(), 1, "单点 deep 每像素 1 样本");
                let s = &p.samples[0];
                assert_eq!(s.index, 0);
                let expect = 10.0 + (y as f32 * w as f32) + x as f32;
                assert_eq!(s.z.to_bits(), expect.to_bits(), "({x},{y}) 的 Z");
                assert_eq!(s.values.len(), 2);
                assert_eq!(s.values[0].to_bits(), expect.to_bits());
                assert!(s.values[1] >= 1000.0, "id 通道应为 uint 值");
                assert_eq!(p.front_z().unwrap().to_bits(), expect.to_bits());
            }
        }
        // 越界查询 → None（不 panic）。
        assert!(img.pixel(99, 99).is_none());
        // 对 deep 文件调 read_image → 结构化拒绝。
        assert_eq!(read_image(&bytes).unwrap_err().code, "EXR1001");
    }

    #[test]
    fn deep_multi_sample_unpacks_all_samples_without_truncation() {
        // 每像素 3 样本：数据完整解出（不做合并/排序，如实返回）。
        let w = 2u32;
        let h = 1u32;
        let spp = 3u32;
        let z: Vec<f32> = (0..(w as usize * spp as usize))
            .map(|i| 1.0 + i as f32 * 0.5)
            .collect();
        let bytes = build_deep_exr(w, h, &z, spp);
        let img = read_deep_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        for x in 0..w as i32 {
            let p = img.pixel(x, 0).unwrap_or_else(|| panic!("缺 ({x},0)"));
            assert_eq!(p.samples.len(), spp as usize, "多点 deep 须全部解出");
            for s in 0..spp as usize {
                let want = 1.0 + (x as usize * spp as usize + s) as f32 * 0.5;
                assert_eq!(p.samples[s].z.to_bits(), want.to_bits());
                assert_eq!(p.samples[s].index, s as u32);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5) 确定性 + 畸形输入攻击
    // -----------------------------------------------------------------------

    #[test]
    fn parse_is_deterministic_byte_for_byte() {
        let spec = FlatSpec {
            width: 4,
            height: 3,
            compression: Compression::Rle,
            channels: vec![
                ("R".into(), PixelType::Half),
                ("Z".into(), PixelType::Float),
                ("A".into(), PixelType::Uint),
            ],
            data: vec![
                (0..12).map(|i| i as f32 * 0.1).collect(),
                (0..12).map(|i| i as f32 * 3.0).collect(),
                (0..12).map(|i| i as f32).collect(),
            ],
        };
        let bytes = build_flat_exr(&spec);
        let a = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let b = read_header(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "头解析须逐字段一致");
        let ia = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let ib = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ia.planes, ib.planes, "像素解须逐值一致");
        let da = read_deep_image(&build_deep_exr(2, 2, &[1.0, 2.0, 3.0, 4.0], 1))
            .unwrap_or_else(|e| panic!("{e}"));
        let db = read_deep_image(&build_deep_exr(2, 2, &[1.0, 2.0, 3.0, 4.0], 1))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(da, db, "deep 解析须一致");
    }

    #[test]
    fn to_fbuf16f_delivers_rgba() {
        let spec = FlatSpec {
            width: 2,
            height: 1,
            compression: Compression::None,
            channels: vec![
                ("R".into(), PixelType::Half),
                ("G".into(), PixelType::Half),
                ("B".into(), PixelType::Half),
                ("A".into(), PixelType::Half),
            ],
            data: vec![
                vec![1.0, 0.0],
                vec![0.5, 0.25],
                vec![0.0, 0.75],
                vec![1.0, 0.0],
            ],
        };
        let bytes = build_flat_exr(&spec);
        let img = read_image(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let fb = img.to_fbuf16f();
        assert_eq!((fb.width, fb.height), (2, 1));
        let p0 = fb.get(0, 0);
        assert_eq!(p0, [1.0, 0.5, 0.0, 1.0]);
        let p1 = fb.get(1, 0);
        assert_eq!(p1, [0.0, 0.25, 0.75, 0.0]);
        // f16 位模式可读回（交付量化只发生在此处）。
        assert_eq!(fb.pixels[0], 0x3C00); // 1.0
    }

    #[test]
    fn malformed_exr_inputs_are_structurally_rejected() {
        let good = build_flat_exr(&FlatSpec {
            width: 2,
            height: 2,
            compression: Compression::None,
            channels: vec![("R".into(), PixelType::Float)],
            data: vec![vec![1.0, 2.0, 3.0, 4.0]],
        });
        // (a) 空 / 短文件。
        assert_eq!(read_header(&[]).unwrap_err().code, "EXR1001");
        assert_eq!(read_header(&[0x76]).unwrap_err().code, "EXR1001");
        // (b) 错 magic。
        let mut bad = good.clone();
        bad[0] = 0x00;
        let e = read_header(&bad).unwrap_err();
        assert_eq!(e.code, "EXR1001");
        assert!(format!("{e}").contains("magic"), "{e}");
        // (c) 版本号不是 2。
        let mut bad_v = good.clone();
        bad_v[4] = 1;
        assert_eq!(read_header(&bad_v).unwrap_err().code, "EXR1001");
        // (d) 未定义标志位。
        let mut bad_f = good.clone();
        bad_f[4] = 0x40; // 保留位
        assert_eq!(read_header(&bad_f).unwrap_err().code, "EXR1001");
        // (e) 多部件 → 结构化拒绝。
        let mut bad_mp = good.clone();
        bad_mp[7] |= 0x10;
        assert_eq!(read_header(&bad_mp).unwrap_err().code, "EXR1001");
        // (f) 截断到头中 / 偏移表中 / 块中。
        for cut in [8usize, 10, 20, 30, 40, 45, 50] {
            if cut < good.len() {
                let r = read_image(&good[..cut]);
                assert!(r.is_err(), "截断 {cut} 字节应报错，得到 Ok");
            }
        }
        // (g) 超大属性 size（声明 0xFFFFFF00）。
        let mut huge = good.clone();
        // 属性头布局：`name\0` + `type\0` + `size(u32)` + value。
        // `dataWindow` → name 占 10+1，`box2i` → type 占 5+1。
        if let Some(p) = huge.windows(10).position(|w| w == b"dataWindow") {
            let size_at = p + "dataWindow".len() + 1 + "box2i".len() + 1;
            if let Some(s) = huge.get_mut(size_at..size_at + 4) {
                s.copy_from_slice(&0xFFFF_FF00u32.to_le_bytes());
            }
            let e = read_header(&huge).unwrap_err();
            assert_eq!(e.code, "EXR1001", "{e}");
            assert!(format!("{e}").contains("size"), "{e}");
        }
        // (h) 偏移表越界（负偏移：0xFFFFFFFFFFFFFFFF）。
        // 定位偏移表：属性表以空 name 结束，其后紧跟偏移表。
        let table_at = good
            .windows(2)
            .position(|w| w == b"lineOrder")
            .map(|p| p + "lineOrder\0".len() + "lineOrder\0".len() + 4 + 1 + 1)
            .unwrap_or(0);
        {
            let mut bad_off = good.clone();
            if table_at + 8 <= bad_off.len() {
                bad_off[table_at..table_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
                let e = read_header(&bad_off).unwrap_err();
                assert_eq!(e.code, "EXR1001", "{e}");
            }
        }
        // (i) 负的负载长度（-1）：直接改偏移表第 0 项指向的块头。
        {
            let mut neg_len = good.clone();
            if table_at + 8 <= neg_len.len() {
                let mut v = [0u8; 8];
                v.copy_from_slice(&neg_len[table_at..table_at + 8]);
                let blk = u64::from_le_bytes(v) as usize;
                if blk + 8 <= neg_len.len() {
                    neg_len[blk + 4..blk + 8].copy_from_slice(&(-1i32).to_le_bytes());
                    let r = read_image(&neg_len);
                    assert!(r.is_err(), "负负载长度应报错");
                }
            }
        }
        // (j) 通道像素类型非法。
        let mut bad_pt = good.clone();
        if let Some(p) = bad_pt.windows(7).position(|w| w == b"chlist") {
            // 值起点 = name("channels"+NUL, 9) + type("chlist"+NUL, 7) + size(4)；
            // 再 + 通道名("R"+NUL, 2) 才是像素类型（i32）。
            let vstart = p + "channels".len() + 1 + "chlist".len() + 1 + 4;
            if let Some(t) = bad_pt.get_mut(vstart + 2..vstart + 6) {
                t.copy_from_slice(&99i32.to_le_bytes());
            }
            let e = read_header(&bad_pt).unwrap_err();
            assert_eq!(e.code, "EXR3001", "{e}");
        }
        // (k) 子采样通道 → 结构化拒绝。
        let mut sub = good.clone();
        if let Some(p) = sub.windows(7).position(|w| w == b"chlist") {
            // chlist 条目：name\0 + pixelType(i32) + pLinear + reserved(3) + xSampling(i32)
            let vstart = p + "channels".len() + 1 + "chlist".len() + 1 + 4;
            let xs = vstart + 2 + 4 + 3;
            if let Some(s) = sub.get_mut(xs..xs + 4) {
                s.copy_from_slice(&2i32.to_le_bytes()); // xSampling = 2
            }
            let e = read_header(&sub).unwrap_err();
            assert_eq!(e.code, "EXR3001", "{e}");
            assert!(format!("{e}").contains("采样"), "{e}");
        }
        // (l) 缺 channels / compression / dataWindow。
        let mut no_ch = good.clone();
        if let Some(p) = no_ch.windows(7).position(|w| w == b"chlist") {
            no_ch[p - 1] = b'X'; // "channels" → "Xhannels"（变未知属性）
            let e = read_header(&no_ch).unwrap_err();
            assert_eq!(e.code, "EXR3001", "{e}");
        }
        // (m) 声明宽高为 0（xMax < xMin）。
        let mut zero = good.clone();
        if let Some(p) = zero.windows(10).position(|w| w == b"dataWindow") {
            let vstart = p + "dataWindow".len() + 1 + "box2i".len() + 1 + 4;
            if let Some(s) = zero.get_mut(vstart + 8..vstart + 12) {
                s.copy_from_slice(&(-5i32).to_le_bytes()); // xMax = -5 < xMin = 0
            }
            let e = read_header(&zero).unwrap_err();
            assert_eq!(e.code, "EXR1001", "{e}");
        }
    }
}
