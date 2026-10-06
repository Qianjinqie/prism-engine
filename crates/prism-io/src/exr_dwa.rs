//! OpenEXR DWA（DWAA / DWAB）解码 —— 自研实现，零第三方依赖。
//!
//! # 来源与正确性依据（本文件的实现依据，非任务书描述）
//!
//! 本模块按 **OpenEXR 3.1.5 真实格式** 实现，权威依据为：
//! - `ImfDwaCompressor.cpp`（`DwaCompressor::uncompress` /
//!   `LossyDctDecoderBase::execute` / `classifyChannels` / `Classifier`）
//! - `ImfDwaCompressorSimd.h`（`dctInverse8x8_scalar` / `fromHalfZigZag_scalar` /
//!   `csc709Inverse`）
//! - `ImfHuf.cpp`（`hufUncompress` / `hufUnpackEncTable` / `hufBuildDecTable` /
//!   `hufCanonicalCodeTable`）
//! - `dwaLookups.cpp::generateToLinear`（非线性 ⇄ 线性 LUT 的**公式**）
//!
//! ⚠️ **任务书中的 DWA 描述与真实格式不符，本实现以真实格式为准。**
//! 任务书称块头为「version / unknownUncompressedSize / unknownCompressedSize
//! 三 × u32」；真实块头是 **11 × i64**（`NUM_SIZES_SINGLE`），且 version ≥ 2 时
//! 块首还有**通道分类规则表**。若照任务书实现将完全无法解码真实文件。
//!
//! # 真实 DWA 块结构
//!
//! ```text
//! [11 × i64 大小/计数表]              (88 字节，XDR 大端 → 本模块按小端读)
//! [version ≥ 2 ? 规则表 : 无]
//! [unknown 区]  zlib 压缩的「原样拷贝」通道
//! [AC 区]       STATIC_HUFFMAN(本模块) 或 DEFLATE
//! [DC 区]       官方 Imf::Zip（**非**裸 zlib：交织 + delta 预测器 + deflate）
//! [RLE 区]      zlib 压缩的 EXR RLE（控制字节极性：负 = 字面段，非负 = 行程）
//! ```
//!
//! # 互通状态：已与官方 OpenEXR 3.1.5 逐格对拍通过
//!
//! **对拍矩阵**：2 压缩（DWAA/DWAB）× 2 像素类型（half/float）× 2 布局
//! （scanline/tiled）× 3 通道数（1/3/4）× 3 图案（gradient/noise/const）= **72 格**，
//! 三方向逐格实跑（2025-10-06，64×48，seed 20251006）：
//!
//! | 方向 | 含义 | 结果 |
//! |---|---|---|
//! | A | 官方写 → 我们读 | **72/72 通过**。gradient 与 const 共 50 格**逐位全等**（`BITDIFF=0`）；noise 格 204/589824 样本差 2~4 个 half-ULP，最差 `MAX_ABS=7.8125e-03`、`MAX_REL=2.442997e-03` |
//! | B | 我们写 → 官方读 | **36/36 scanline 格通过**（官方无异常读完）。24 个 tiled 格**显式 SKIP**——本仓 `exr_write` 未实装 tiled 写出（结构性拒绝），非静默 pass |
//! | C | 两份文件均由官方解码器解，各自与同一源数据比 | **36/36 通过**，劣化比 1.01~55.97 |
//!
//! 门禁：`crates/prism-io/tests/interop/interop_gate.mjs`（`--self-test` 注入式
//! 反向自证 15/15）。官方 OpenEXR 是**验证工具而非项目依赖**：C++ harness 源码
//! 只以 `.cpp` 形式躺在 `tests/interop/`，由脚本在 `/tmp` 现场编译，
//! `crates/` 下无 `build.rs`、无 C++ 链接、无 `cc` crate。
//!
//! **noise 格为何不是 bit-exact**：官方 IDCT 走 SSE2/SIMD
//! （`ImfDwaCompressorSimd.h`），本模块走标量 f32，**浮点加法次序不同**，
//! 末位舍入有差。反证：gradient / const 两类共 50 格逐位全等——若是比特流
//! 解析错误，那两类也会错位。
//!
//! # 诚实边界
//!
//! - **DC-only 快路径 / CSC 色彩空间转换**按 OpenEXR 语义实现；
//! - `toLinear` 非线性 LUT：公式现场计算（`dwaLookups.cpp::generateToLinear`
//!   的 `pow(x, 2.2)` 语义）。已用官方 `dwaLookups.h` 的**全部 65536 项**
//!   静态表逐位核对过（离线核对，非仓内穷举测试）；仓内以 512 项
//!   `OFFICIAL_LUT_PAIRS` 抽样回归。
//! - **编码侧**（`compress_block` / `zip_pack`）已能产出官方可读文件（方向 B 全绿），
//!   但**量化表比官方粗**：1 通道高频噪声下 max-abs 约为官方自己的 56 倍
//!   （见 `interop_gate.mjs` 的 `c_max_ratio` 阈值说明）。这是编码器质量问题，
//!   不是互通缺陷，未做优化。
//! - tiled **写入**未实装；tiled **读取**已实装且方向 A 全绿。

use crate::exr::ExrResult;
use crate::fbuf::{f16_to_f32, f32_to_f16};

/// DWA 大小表项数（`NUM_SIZES_SINGLE`）。
const NUM_SIZES_SINGLE: usize = 11;

/// 单个 AC/DC 系数 short 的字节数。
const USHORT_BYTES: usize = 2;

/// DCT 块边长（DWA 用 8×8，不是任务书所说的 32×32）。
const DCT_N: usize = 8;

/// 8×8 = 64 个系数。
const DCT_COEFFS: usize = DCT_N * DCT_N;

/// 通道分类方案（`CompressorScheme`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwaScheme {
    /// 未匹配任何规则 → 原样存储（zlib）。
    Unknown,
    /// 有损 DCT + 量化。
    LossyDct,
    /// EXR RLE + zlib。
    Rle,
}

impl DwaScheme {
    /// 由规则字节的 scheme 位解码（2 位字段，仅 0..=2 合法）。
    fn from_bits(bits: u8) -> Option<Self> {
        Some(match bits {
            0 => DwaScheme::Unknown,
            1 => DwaScheme::LossyDct,
            2 => DwaScheme::Rle,
            _ => return None,
        })
    }
}

/// 通道分类规则（`DwaCompressor::Classifier`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwaRule {
    /// 通道名后缀（`R`/`G`/`Z`/…；不含前缀部分）。
    pub suffix: String,
    /// 压缩方案。
    pub scheme: DwaScheme,
    /// 像素类型。
    pub pixel_type: crate::exr::PixelType,
    /// CSC 组内序号（0=R,1=G,2=B；-1 = 不参与 CSC）。
    pub csc_idx: i32,
    /// 后缀匹配是否忽略大小写。
    pub case_insensitive: bool,
}

impl DwaRule {
    /// 该规则是否匹配给定后缀与类型。
    fn matches(&self, suffix: &str, pt: crate::exr::PixelType) -> bool {
        if self.pixel_type != pt {
            return false;
        }
        if self.case_insensitive {
            suffix.eq_ignore_ascii_case(&self.suffix)
        } else {
            suffix == self.suffix
        }
    }
}

/// 序列化一条规则（供写出/测试构造）。
pub fn encode_rule(rule: &DwaRule) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(rule.suffix.len() + 3);
    v.extend_from_slice(rule.suffix.as_bytes());
    v.push(0);
    let csc_bits = ((rule.csc_idx + 1) as u8) & 15;
    let scheme_bits = match rule.scheme {
        DwaScheme::Unknown => 0u8,
        DwaScheme::LossyDct => 1,
        DwaScheme::Rle => 2,
    };
    v.push(
        (csc_bits << 4) | (scheme_bits << 2) | (u8::from(rule.case_insensitive) & 1),
    );
    v.push(rule.pixel_type.code() as u8);
    v
}

/// version ≥ 2 时文件内自带规则表；缺失/为空的默认表（`initializeDefaultChannelRules`）。
pub fn default_rules() -> Vec<DwaRule> {
    use crate::exr::PixelType::{Float, Half, Uint};
    let mut v = Vec::new();
    let mut push = |suffix: &str, scheme: DwaScheme, pt: crate::exr::PixelType, csc: i32| {
        v.push(DwaRule {
            suffix: suffix.to_string(),
            scheme,
            pixel_type: pt,
            csc_idx: csc,
            case_insensitive: false,
        });
    };
    for (name, idx) in [("R", 0), ("G", 1), ("B", 2)] {
        push(name, DwaScheme::LossyDct, Half, idx);
        push(name, DwaScheme::LossyDct, Float, idx);
    }
    for name in ["Y", "BY", "RY"] {
        push(name, DwaScheme::LossyDct, Half, -1);
        push(name, DwaScheme::LossyDct, Float, -1);
    }
    push("A", DwaScheme::Rle, Uint, -1);
    push("A", DwaScheme::Rle, Half, -1);
    push("A", DwaScheme::Rle, Float, -1);
    v
}

/// version < 2 时的内置规则表（`initializeLegacyChannelRules`）。
///
/// 真实 OpenEXR 的 legacy 表含大量小写别名（`r`/`red`/`g`/`grn`/`green`/…）
/// 且**全部大小写不敏感**；此处按真实表实现。
pub fn legacy_rules() -> Vec<DwaRule> {
    use crate::exr::PixelType::{Float, Half};
    let mut v = Vec::new();
    let mut push = |suffix: &str, scheme: DwaScheme, pt: crate::exr::PixelType, csc: i32| {
        v.push(DwaRule {
            suffix: suffix.to_string(),
            scheme,
            pixel_type: pt,
            csc_idx: csc,
            case_insensitive: true,
        });
    };
    for (name, idx) in [("r", 0), ("red", 0), ("g", 1), ("grn", 1), ("green", 1), ("b", 2), ("blu", 2), ("blue", 2)] {
        push(name, DwaScheme::LossyDct, Half, idx);
        push(name, DwaScheme::LossyDct, Float, idx);
    }
    for name in ["y", "by", "ry"] {
        push(name, DwaScheme::LossyDct, Half, -1);
        push(name, DwaScheme::LossyDct, Float, -1);
    }
    push("a", DwaScheme::Rle, crate::exr::PixelType::Uint, -1);
    push("a", DwaScheme::Rle, Half, -1);
    push("a", DwaScheme::Rle, Float, -1);
    v
}

/// 通道分类结果（单个通道的解码方案）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwaChannelPlan {
    /// 压缩方案。
    pub scheme: DwaScheme,
    /// 像素类型。
    pub pixel_type: crate::exr::PixelType,
    /// 是否 `pLinear`（决定是否走非线性 LUT 还原）。
    pub p_linear: bool,
    /// 在块内通道序列中的下标。
    pub index: usize,
}

/// CSC 三通道组（`cscSets`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwaCscSet {
    /// R / G / B 在块内通道序列中的下标。
    pub idx: [usize; 3],
}

/// 分类结果：逐通道方案 + CSC 组。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwaPlan {
    /// 逐通道方案（顺序 = 输入通道顺序，确定性）。
    pub channels: Vec<DwaChannelPlan>,
    /// CSC 组（按前缀首次出现顺序，确定性）。
    pub csc_sets: Vec<DwaCscSet>,
}

/// 按规则表分类通道（`classifyChannels`）。
///
/// 确定性：前缀分组用 `Vec` + 线性查找（**不用 HashMap**），遍历顺序 = 输入顺序。
pub fn classify_channels(
    names: &[(String, crate::exr::PixelType, bool)],
    rules: &[DwaRule],
) -> DwaPlan {
    let mut schemes: Vec<DwaScheme> = vec![DwaScheme::Unknown; names.len()];
    // 前缀 → 三元组槽位（-1 = 缺）。
    let mut prefixes: Vec<String> = Vec::new();
    let mut slots: Vec<[i32; 3]> = Vec::new();

    for (i, (name, pt, _)) in names.iter().enumerate() {
        let (prefix, suffix) = match name.rfind('.') {
            Some(dot) => (&name[..dot], &name[dot + 1..]),
            None => ("", name.as_str()),
        };
        let pi = match prefixes.iter().position(|p| p == prefix) {
            Some(k) => k,
            None => {
                prefixes.push(prefix.to_string());
                slots.push([-1, -1, -1]);
                prefixes.len() - 1
            }
        };
        // 规则按声明顺序遍历，**后者覆盖前者**（与 OpenEXR 一致）。
        for r in rules {
            if r.matches(suffix, *pt) {
                schemes[i] = r.scheme;
                let c = r.csc_idx;
                if c >= 0 {
                    if let Some(slot) = slots.get_mut(pi) {
                        if let Some(entry) = slot.get_mut(c as usize) {
                            *entry = i as i32;
                        }
                    }
                }
            }
        }
    }

    // CSC 组枚举：OpenEXR 用 `std::map<std::string, CscChannelSet>`，迭代序是
    // **前缀字典序**（不是插入序，也不是通道序）。此处对前缀做稳定字典序排序
    // 以复刻该顺序 —— 顺序决定 AC/DC 消费次序，必须逐位对齐。
    let mut order: Vec<usize> = (0..prefixes.len()).collect();
    order.sort_by(|&a, &b| prefixes[a].cmp(&prefixes[b]));

    let mut csc_sets: Vec<DwaCscSet> = Vec::new();
    for pi in order {
        let s = slots[pi];
        if s[0] >= 0 && s[1] >= 0 && s[2] >= 0 {
            csc_sets.push(DwaCscSet {
                idx: [s[0] as usize, s[1] as usize, s[2] as usize],
            });
        }
    }

    let channels = names
        .iter()
        .enumerate()
        .map(|(i, (_, pt, pl))| DwaChannelPlan {
            scheme: schemes[i],
            pixel_type: *pt,
            p_linear: *pl,
            index: i,
        })
        .collect();
    DwaPlan {
        channels,
        csc_sets,
    }
}

// ===========================================================================
//  Huf（AC 区的 STATIC_HUFFMAN 熵解码）
// ===========================================================================

/// Huf 编码表大小（`HUF_ENCSIZE`）。
const HUF_ENCSIZE: usize = (1 << 16) + 1;
/// Huf 解码表位数（`HUF_DECBITS`）。
const HUF_DECBITS: u32 = 14;
/// Huf 解码表大小。
const HUF_DECSIZE: usize = 1 << HUF_DECBITS;
/// 短零码游程标记（`SHORT_ZEROCODE_RUN`）。
const SHORT_ZEROCODE_RUN: u32 = 59;
/// 长零码游程标记（`LONG_ZEROCODE_RUN`）。
const LONG_ZEROCODE_RUN: u32 = 63;
/// 最短长游程（`SHORTEST_LONG_RUN`）。
const SHORTEST_LONG_RUN: i32 = 2 + 63 - 59;

/// MSB 优先位读取器（Huf 流语义）。
struct HufBits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> HufBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// 取一个字节（越界返回 0，与 OpenEXR 的越界读行为一致但不 panic）。
    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos = self.pos.saturating_add(1);
        b
    }

    /// 已消费字节数。
    fn consumed(&self) -> usize {
        self.pos
    }
}

/// 由码长数组构建 canonical Huffman 码（`hufCanonicalCodeTable`）。
///
/// 返回 `(code, length)`：码字在**低 6 位以上的位**、长度在低 6 位，
/// 与 OpenEXR 的 `hcode` 打包方式一致。
fn huf_canonical_code_table(lengths: &mut [u32]) {
    // n[i] = 长度为 i 的码字数；随后原地转成「该长度的起始码」。
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
            *slot = l | ((n[l as usize] << 6) as u32);
            n[l as usize] += 1;
        }
    }
}

/// 解包 Huf 编码表（`hufUnpackEncTable`）。
fn huf_unpack_enc_table(bits: &mut HufBits<'_>, ni: usize, im: i32, i_m: i32) -> ExrResult<Vec<u32>> {
    let mut hcode: Vec<u32> = vec![0; HUF_ENCSIZE];
    let start = im.max(0) as usize;
    let end = (i_m.max(-1) as usize).min(HUF_ENCSIZE - 1);
    if i_m < im {
        return Ok(hcode);
    }
    let mut c: u64 = 0;
    let mut lc: i32 = 0;

    let mut i = start;
    while i <= end {
        if bits.consumed() > ni {
            return Err(crate::exr::comp_err(
                "exr::dwa::huf_table",
                "Huf 编码表越界读取",
                "AC 区损坏；确认文件完整",
            ));
        }
        // 读 6 位码长（与 OpenEXR `getBits` 的 lc>=6 保证一致）。
        while lc < 6 {
            c = (c << 8) | bits.byte() as u64;
            lc += 8;
        }
        lc -= 6;
        let l = ((c >> lc) & 63) as u32;
        hcode[i] = l;
        if l == LONG_ZEROCODE_RUN {
            if bits.consumed() > ni {
                return Err(crate::exr::comp_err(
                    "exr::dwa::huf_table",
                    "Huf 长零游程越界读取",
                    "AC 区损坏；确认文件完整",
                ));
            }
            while lc < 8 {
                c = (c << 8) | bits.byte() as u64;
                lc += 8;
            }
            lc -= 8;
            let zerun = (((c >> lc) & 0xFF) as i32) + SHORTEST_LONG_RUN;
            if i as i32 + zerun > i_m + 1 {
                return Err(crate::exr::comp_err(
                    "exr::dwa::huf_table",
                    format!("Huf 零游程越过表尾（{zerun} 项）"),
                    "AC 区编码表损坏",
                ));
            }
            for _ in 0..zerun {
                if i <= end {
                    hcode[i] = 0;
                }
                i += 1;
            }
            i -= 1;
        } else if l >= SHORT_ZEROCODE_RUN {
            let zerun = (l as i32) - (SHORT_ZEROCODE_RUN as i32) + 2;
            if i as i32 + zerun > i_m + 1 {
                return Err(crate::exr::comp_err(
                    "exr::dwa::huf_table",
                    format!("Huf 短零游程越过表尾（{zerun} 项）"),
                    "AC 区编码表损坏",
                ));
            }
            for _ in 0..zerun {
                if i <= end {
                    hcode[i] = 0;
                }
                i += 1;
            }
            i -= 1;
        }
        i += 1;
    }
    huf_canonical_code_table(&mut hcode);
    Ok(hcode)
}

/// 构建 Huf 解码表（`hufBuildDecTable`）。
///
/// 为 Mini Code 与 C18（不分配 16384 个 `Vec`），采用**线性探测式**结构：
/// `shorts: Vec<(u8 /*len*/, u32 /*lit*/)>` 按 `HUF_DECSIZE` 索引，
/// `longs: Vec<Vec<u32>>` 仅在存在长码时分配。
///
/// **符号用 `u32` 而非 `u16`**：HUF 符号空间是 `0..=65536`（`HUF_ENCSIZE = 65537`），
/// 其中 **65536 是 RLC 伪符号**（`hufBuildEncTable` 在最大数据符号之后追加）。
/// 数据里出现符号 65535 时 RLC 就落在 65536，`as u16` 会截断成 0，与真实符号 0
/// 混淆 ⇒ 解码出错误数据。DWA 的 AC 系数取不到 65535，故该缺陷此前不可见；
/// PIZ 的小波输出（`a - b` 的有符号差）会取到，故这里必须用 `u32`。
fn huf_build_dec_table(hcode: &[u32], im: i32, i_m: i32) -> ExrResult<(Vec<(u8, u32)>, Vec<Vec<u32>>)> {
    let mut shorts: Vec<(u8, u32)> = vec![(0u8, 0u32); HUF_DECSIZE];
    let mut longs: Vec<Vec<u32>> = vec![Vec::new(); HUF_DECSIZE];
    let start = im.max(0) as usize;
    let end = (i_m.max(-1) as usize).min(HUF_ENCSIZE - 1);
    for i in start..=end {
        let entry = match hcode.get(i) {
            Some(&e) => e,
            None => continue,
        };
        let l = (entry & 63) as u32;
        let code = (entry >> 6) as u64;
        if l == 0 {
            continue;
        }
        if l > HUF_DECBITS {
            if (code >> l) != 0 {
                return Err(crate::exr::comp_err(
                    "exr::dwa::huf_dec_table",
                    format!("Huf 码字 {code} 超出其 {l} 位长度"),
                    "AC 区编码表损坏",
                ));
            }
            let slot = (code >> (l - HUF_DECBITS)) as usize;
            if slot >= HUF_DECSIZE {
                return Err(crate::exr::comp_err(
                    "exr::dwa::huf_dec_table",
                    "Huf 长码槽位越界",
                    "AC 区编码表损坏",
                ));
            }
            if let Some(s) = shorts.get(slot) {
                if s.0 != 0 {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_dec_table",
                        "Huf 表项冲突：短码与长码重叠",
                        "AC 区编码表损坏",
                    ));
                }
            }
            if let Some(v) = longs.get_mut(slot) {
                v.push(i as u32);
            }
        } else {
            let base = (code << (HUF_DECBITS - l)) as usize;
            let span = 1usize << (HUF_DECBITS - l);
            for k in 0..span {
                let idx = base + k;
                if let Some(s) = shorts.get_mut(idx) {
                    if s.0 != 0 {
                        return Err(crate::exr::comp_err(
                            "exr::dwa::huf_dec_table",
                            "Huf 表项冲突：重复短码",
                            "AC 区编码表损坏",
                        ));
                    }
                    *s = (l as u8, i as u32);
                }
            }
        }
    }
    Ok((shorts, longs))
}

/// Huf AC 解码（`hufUncompress` + `hufDecode`）。
///
/// `raw` 必须恰好解出 `n_raw` 个 `u16`，否则结构化报错。
pub fn huf_uncompress(compressed: &[u8], n_raw: usize) -> ExrResult<Vec<u16>> {
    let empty = Vec::new();
    if compressed.len() < 20 {
        if n_raw == 0 {
            return Ok(empty);
        }
        return Err(crate::exr::comp_err(
            "exr::dwa::huf_uncompress",
            format!("Huf 负载仅 {} 字节（需 ≥ 20 字节头）", compressed.len()),
            "AC 区被截断；确认文件完整",
        ));
    }
    let rd = |at: usize| -> u32 {
        u32::from_le_bytes([
            compressed.get(at).copied().unwrap_or(0),
            compressed.get(at + 1).copied().unwrap_or(0),
            compressed.get(at + 2).copied().unwrap_or(0),
            compressed.get(at + 3).copied().unwrap_or(0),
        ])
    };
    let im = rd(0) as i32;
    let i_m = rd(4) as i32;
    let n_bits = rd(12) as usize;
    if im < 0 || im as usize >= HUF_ENCSIZE || i_m < 0 || i_m as usize >= HUF_ENCSIZE {
        return Err(crate::exr::comp_err(
            "exr::dwa::huf_uncompress",
            format!("Huf 表范围非法（im={im}, iM={i_m}）"),
            "AC 区编码表损坏",
        ));
    }
    let body = &compressed[20..];
    let n_bytes = n_bits.div_ceil(8);
    if n_bytes > body.len() {
        return Err(crate::exr::comp_err(
            "exr::dwa::huf_uncompress",
            format!("Huf 数据需 {n_bytes} 字节，实际 {} 字节", body.len()),
            "AC 区被截断；确认文件完整",
        ));
    }

    let mut bits = HufBits::new(body);
    let hcode = huf_unpack_enc_table(&mut bits, body.len(), im, i_m)?;
    let remaining = body.len() - bits.consumed();
    if n_bits > 8 * remaining {
        return Err(crate::exr::comp_err(
            "exr::dwa::huf_uncompress",
            format!("Huf 声明 {n_bits} 比特但仅剩 {} 字节", remaining),
            "AC 区被截断；确认文件完整",
        ));
    }
    let (shorts, longs) = huf_build_dec_table(&hcode, im, i_m)?;

    // 符号取 run-length 码（= iM）时按游程展开（`getCode` 宏）。
    let mut out: Vec<u16> = Vec::with_capacity(n_raw);
    let mut c: u64 = 0;
    let mut lc: i32 = 0;
    let mut reader = HufBits::new(&body[bits.consumed()..]);
    let ie = reader.data.len();
    // RLC 伪符号用 `u32` 比较（其合法取值含 65536，见 `huf_build_dec_table` 文档）。
    let rlc = i_m as u32;

    // 取一个符号并处理游程；返回 false 表示输入耗尽。
    macro_rules! emit {
        ($sym:expr) => {{
            let sym: u32 = $sym;
            if sym == rlc {
                if lc < 8 {
                    if reader.pos < ie {
                        c = (c << 8) | reader.byte() as u64;
                        lc += 8;
                    }
                }
                lc -= 8;
                let cs = if lc >= 0 {
                    ((c >> lc) & 0xFF) as usize
                } else {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        "Huf 游程长度位不足",
                        "AC 区损坏",
                    ));
                };
                if out.len() + cs > n_raw {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        format!("Huf 游程展开后 {} 项超过声明的 {n_raw}", out.len() + cs),
                        "AC 区损坏或 nRaw 字段错误",
                    ));
                }
                if out.is_empty() {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        "Huf 游程出现在首项（无前值可重复）",
                        "AC 区损坏",
                    ));
                }
                let prev = *out.last().unwrap_or(&0);
                for _ in 0..cs {
                    out.push(prev);
                }
            } else {
                if out.len() >= n_raw {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        format!("Huf 解出项数超过声明的 {n_raw}"),
                        "AC 区损坏或 nRaw 字段错误",
                    ));
                }
                out.push(sym as u16);
            }
        }};
    }

    while reader.pos < ie {
        if reader.pos < ie {
            c = (c << 8) | reader.byte() as u64;
            lc += 8;
        }
        while lc >= HUF_DECBITS as i32 {
            let idx = ((c >> (lc - HUF_DECBITS as i32)) as usize) & (HUF_DECSIZE - 1);
            let (slen, slit) = shorts.get(idx).copied().unwrap_or((0, 0));
            if slen != 0 {
                lc -= slen as i32;
                if lc < 0 {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        "Huf 码长超出可用比特",
                        "AC 区损坏",
                    ));
                }
                emit!(slit);
            } else {
                let cand = longs.get(idx).cloned().unwrap_or_default();
                if cand.is_empty() {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        "Huf 遇到未定义码字",
                        "AC 区损坏或编码表不匹配",
                    ));
                }
                let mut found = false;
                for &sym in &cand {
                    let e = hcode.get(sym as usize).copied().unwrap_or(0);
                    let l = (e & 63) as i32;
                    while lc < l && reader.pos < ie {
                        c = (c << 8) | reader.byte() as u64;
                        lc += 8;
                    }
                    if lc >= l {
                        let code = (e >> 6) as u64;
                        if code == ((c >> (lc - l)) & ((1u64 << l) - 1)) {
                            lc -= l;
                            emit!(sym);
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    return Err(crate::exr::comp_err(
                        "exr::dwa::huf_decode",
                        "Huf 长码未匹配",
                        "AC 区损坏或编码表不匹配",
                    ));
                }
            }
        }
    }

    // 末尾不足一字节的残余比特。
    let tail = ((8 - (n_bits & 7)) & 7) as i32;
    if tail > 0 && lc >= tail {
        c >>= tail;
        lc -= tail;
    }
    while lc > 0 {
        let idx = ((c as usize) << (HUF_DECBITS as i32 - lc)) & (HUF_DECSIZE - 1);
        let (slen, slit) = shorts.get(idx).copied().unwrap_or((0, 0));
        if slen == 0 {
            return Err(crate::exr::comp_err(
                "exr::dwa::huf_decode",
                "Huf 末尾残余比特不是合法短码",
                "AC 区损坏",
            ));
        }
        lc -= slen as i32;
        if lc < 0 {
            return Err(crate::exr::comp_err(
                "exr::dwa::huf_decode",
                "Huf 末尾码长越界",
                "AC 区损坏",
            ));
        }
        emit!(slit);
    }

    if out.len() != n_raw {
        return Err(crate::exr::comp_err(
            "exr::dwa::huf_uncompress",
            format!("Huf 解出 {} 项，声明 {n_raw} 项", out.len()),
            "AC 区损坏或 nRaw 字段错误",
        ));
    }
    Ok(out)
}

// ===========================================================================
//  8×8 逆 DCT / 反 zigzag / CSC
// ===========================================================================

/// 反 zigzag：`fromHalfZigZag_scalar`，把 zigzag 序的 64 个 half 位模式
/// 按 OpenEXR 的固定置换展开成自然序的 64 个 f32 系数。
///
/// 置换表为 OpenEXR 的**逐项常量映射**（`ImfDwaCompressorSimd.h`），
/// 以 `DST[i] = SRC[ZIGZAG_TO_RASTER[i]]` 形式固化，顺序固定 → 确定性。
fn from_half_zig_zag(src: &[u16; DCT_COEFFS], dst: &mut [f32; DCT_COEFFS]) {
    // OpenEXR `fromHalfZigZag_scalar` 的 dst[i] = (float)srcHalf[MAP[i]]。
    const MAP: [usize; DCT_COEFFS] = [
        0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
        18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21,
        34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
    ];
    for (i, &m) in MAP.iter().enumerate() {
        dst[i] = f16_to_f32(src[m]);
    }
}

/// 8×8 逆 DCT（`dctInverse8x8_scalar<0>`，标量路径）。
///
/// **顺序固定**：先逐行、再逐列（与 OpenEXR 标量实现一致，SIMD 路径亦同）。
fn dct_inverse_8x8(data: &mut [f32; DCT_COEFFS]) {
    const A: f32 = 0.353_553_4; // 0.5f * cos(3.14159/4)
    const B: f32 = 0.490_392_6; // 0.5f * cos(3.14159/16)
    const C: f32 = 0.461_939_8; // 0.5f * cos(3.14159/8)
    const D: f32 = 0.415_734_9; // 0.5f * cos(3*3.14159/16)
    const E: f32 = 0.277_785_5; // 0.5f * cos(5*3.14159/16)
    const F: f32 = 0.191_342_2; // 0.5f * cos(3*3.14159/8)
    const G: f32 = 0.097_545_73; // 0.5f * cos(7*3.14159/16)

    // 第一遍：逐行。
    for row in 0..8usize {
        let r = row * 8;
        let p = &data[r..r + 8];
        let (p0, p1, p2, p3, p4, p5, p6, p7) = (
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        );
        let alpha0 = C * p2;
        let alpha1 = F * p2;
        let alpha2 = C * p6;
        let alpha3 = F * p6;
        let beta0 = B * p1 + D * p3 + E * p5 + G * p7;
        let beta1 = D * p1 - G * p3 - B * p5 - E * p7;
        let beta2 = E * p1 - B * p3 + G * p5 + D * p7;
        let beta3 = G * p1 - E * p3 + D * p5 - B * p7;
        let theta0 = A * (p0 + p4);
        let theta3 = A * (p0 - p4);
        let theta1 = alpha0 + alpha3;
        let theta2 = alpha1 - alpha2;
        let g0 = theta0 + theta1;
        let g1 = theta3 + theta2;
        let g2 = theta3 - theta2;
        let g3 = theta0 - theta1;
        data[r] = g0 + beta0;
        data[r + 1] = g1 + beta1;
        data[r + 2] = g2 + beta2;
        data[r + 3] = g3 + beta3;
        data[r + 4] = g3 - beta3;
        data[r + 5] = g2 - beta2;
        data[r + 6] = g1 - beta1;
        data[r + 7] = g0 - beta0;
    }

    // 第二遍：逐列。
    for col in 0..8usize {
        let d = |k: usize| -> f32 { *data.get(k * 8 + col).unwrap_or(&0.0) };
        let (d0, d1, d2, d3, d4, d5, d6, d7) = (
            d(0),
            d(1),
            d(2),
            d(3),
            d(4),
            d(5),
            d(6),
            d(7),
        );
        let alpha0 = C * d2;
        let alpha1 = F * d2;
        let alpha2 = C * d6;
        let alpha3 = F * d6;
        let beta0 = B * d1 + D * d3 + E * d5 + G * d7;
        let beta1 = D * d1 - G * d3 - B * d5 - E * d7;
        let beta2 = E * d1 - B * d3 + G * d5 + D * d7;
        let beta3 = G * d1 - E * d3 + D * d5 - B * d7;
        let theta0 = A * (d0 + d4);
        let theta3 = A * (d0 - d4);
        let theta1 = alpha0 + alpha3;
        let theta2 = alpha1 - alpha2;
        let g0 = theta0 + theta1;
        let g1 = theta3 + theta2;
        let g2 = theta3 - theta2;
        let g3 = theta0 - theta1;
        if let Some(v) = data.get_mut(col) {
            *v = g0 + beta0;
        }
        if let Some(v) = data.get_mut(8 + col) {
            *v = g1 + beta1;
        }
        if let Some(v) = data.get_mut(16 + col) {
            *v = g2 + beta2;
        }
        if let Some(v) = data.get_mut(24 + col) {
            *v = g3 + beta3;
        }
        if let Some(v) = data.get_mut(32 + col) {
            *v = g3 - beta3;
        }
        if let Some(v) = data.get_mut(40 + col) {
            *v = g2 - beta2;
        }
        if let Some(v) = data.get_mut(48 + col) {
            *v = g1 - beta1;
        }
        if let Some(v) = data.get_mut(56 + col) {
            *v = g0 - beta0;
        }
    }
}

/// DC 逆变换的增益：`0.3535536² = 1/8`（官方 `dctInverse8x8DcOnly` 的字面量）。
///
/// 官方两处都写死 `data[0] * 3.535536e-01f * 3.535536e-01f`
/// （`ImfDwaCompressorSimd.h:812`标量版 / `:824` SSE 版），即 1D 正交 DCT-III
/// 基向量首项 `A = 0.5·cos(π/4)` 的平方。与 [`dct_inverse_8x8`] 的 DC 基
/// 幅度一致（那边也是 `A² = 1/8`，见 `dct_inverse_8x8_of_pure_dc_is_a_constant_one_eighth`）。
const DC_ONLY_GAIN: f32 = 3.535_536e-01;

/// 仅逆 DC（`dctInverse8x8DcOnly`）：所有 AC 为 0 时把 DC 铺满整块。
///
/// # ⚠️ 必须乘 `0.3535536² = 1/8`（W13 互通对拍抓到的第三个真实 bug）
///
/// 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressorSimd.h:808`：
/// ```cpp
/// void dctInverse8x8DcOnly (float *data)
/// {
///     float val = data[0] * 3.535536e-01f * 3.535536e-01f;
///     for (int i = 0; i < 64; ++i) data[i] = val;
/// }
/// ```
///
/// 本实现原先写成 `*v = dc`（**漏了 1/8 增益**）。对自家往返测试完全自洽
/// ——因为自家编码器也没乘，两边错误抵消；但官方常量块立刻露馅：
/// 官方 DC = `0x45d7`（half 5.83984375），官方解出 **0.50048828125**，
/// 我们解出 **42080.0**（正好是 `toLinear(5.83984375)`，即少乘了 1/8）。
/// 倍数恰为 8：`0.50048828125 × 8 = 4.00390625`，再经 toLinear 曲线放大到 42080。
fn dct_inverse_dc_only(data: &mut [f32; DCT_COEFFS]) {
    let val = data[0] * DC_ONLY_GAIN * DC_ONLY_GAIN;
    for v in data.iter_mut() {
        *v = val;
    }
}

/// 709 逆 CSC，单分量（`csc709Inverse`）。
fn csc709_inverse(c0: &mut f32, c1: &mut f32, c2: &mut f32) {
    let (s0, s1, s2) = (*c0, *c1, *c2);
    *c0 = s0 + 1.5747 * s2;
    *c1 = s0 - 0.1873 * s1 - 0.4682 * s2;
    *c2 = s0 + 1.8556 * s1;
}

/// 709 逆 CSC，整块（`csc709Inverse64`）：逐元素应用 [`csc709_inverse`]。
fn csc709_inverse64(c0: &mut [f32; DCT_COEFFS], c1: &mut [f32; DCT_COEFFS], c2: &mut [f32; DCT_COEFFS]) {
    for i in 0..DCT_COEFFS {
        csc709_inverse(&mut c0[i], &mut c1[i], &mut c2[i]);
    }
}

/// 非线性 → 线性 half LUT（`dwaCompressorToLinear`）的**公式版**。
///
/// 与 OpenEXR `dwaLookups.cpp::generateToLinear` 逐位一致（已用 65536 项
/// 全表比对验证）。`pLinear` 通道不调用本函数（恒等）。
///
/// **W13 对拍修正**：本函数此前有两处与官方 `dwaLookups.h` 静态表不符，
/// 合计 65536 项中 **27628 项**有差异：
/// 1. 溢出被 clamp 到 ±65504（`0x7BFF`），官方一律 `±inf`（`0x7C00`）；
/// 2. `r` 为非有限时直接返回 0，吞掉了本应输出 `±inf` 的溢出情形。
///
/// 根因是 `f32::powf` 在 half 输入域内**通常不溢出**（f32 指数范围远大于
/// half），例如 `0x460B` → `65810.26` 仍是有限 f32；要让它溢出须 `≥ 0x5BFF`。
/// 因此正确做法是**不做任何预 clamp，也不拦非有限**，一律交给
/// `fbuf::f32_to_f16` 按 IEEE RNE 落盘：`exp16 >= 0x1F → sign | 0x7C00`，
/// 与官方 `Xdr::write<half>` 结果一致。修正后仅余 2 项 1-ULP 尾数差
/// （`0x24F8`/`0xA4F8`，源于 f32 vs f64 `powf` 末位舍入），见
/// `dwa_to_linear_matches_official_edges_across_all_exponents`。
fn dwa_to_linear(bits: u16) -> u16 {
    // NaN / Inf → 0（与 OpenEXR 一致）。
    if bits & 0x7c00 == 0x7c00 {
        return 0;
    }
    let v = f16_to_f32(bits);
    let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
    let a = v.abs();
    // logBase = pow(2.7182818, 2.2)（f32 语义）。
    // D-30：字面量 2.7182818 与 `std::f32::consts::E` 的 `to_bits()` 逐位相同
    // （0x402DF854），替换不改变 powf 结果，仅消除 deny 级 lint。
    let log_base = std::f32::consts::E.powf(2.2f32);
    let r = if a <= 1.0 {
        sign * a.powf(2.2f32)
    } else {
        sign * log_base.powf(a - 1.0f32)
    };
    // 不预 clamp、不拦非有限：`f32_to_f16` 负责把超出 half 范围的值
    // 映射为 `±inf`（`exp16 >= 0x1F` 分支），与官方静态表逐位一致。
    f32_to_f16(r)
}

// ===========================================================================
//  块解码主流程
// ===========================================================================

/// DWA 块头解析结果（11 × i64）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DwaSizes {
    /// 块格式版本（0/1/2 合法）。
    pub version: u64,
    /// unknown 区解压后长度。
    pub unknown_uncompressed_size: u64,
    /// unknown 区压缩后长度。
    pub unknown_compressed_size: u64,
    /// AC 区压缩后长度。
    pub ac_compressed_size: u64,
    /// DC 区压缩后长度。
    pub dc_compressed_size: u64,
    /// RLE 区压缩后长度。
    pub rle_compressed_size: u64,
    /// RLE 区解压后长度。
    pub rle_uncompressed_size: u64,
    /// RLE 反编码后（原始）长度。
    pub rle_raw_size: u64,
    /// AC 解压后的 u16 项数。
    pub ac_uncompressed_count: u64,
    /// DC 解压后的 u16 项数。
    pub dc_uncompressed_count: u64,
    /// AC 压缩策略（0 = STATIC_HUFFMAN，1 = DEFLATE）。
    pub ac_compression: u64,
}

/// 解码所需的几何/通道上下文（由调用方从 `ExrHeader` 提供）。
#[derive(Debug, Clone)]
pub struct DwaContext<'a> {
    /// 通道名 + 像素类型 + `pLinear`（顺序 = 文件出现顺序）。
    pub channels: &'a [(String, crate::exr::PixelType, bool)],
    /// 块宽（像素）。
    pub width: usize,
    /// 块高（扫描线数）。
    pub height: usize,
}

/// zlib 解压（EXR 的 UNKNOWN / DC / RLE 区用 **zlib 包装**，
/// 与裸 DEFLATE 不同 —— 头部 2 字节 + 尾部 4 字节 Adler-32）。
fn zlib_decompress(data: &[u8], expect: usize) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr::dwa::zlib";
    if data.len() < 2 {
        return Err(crate::exr::comp_err(
            LOC,
            "zlib 流短于 2 字节头",
            "DWA 子块被截断；确认文件完整",
        ));
    }
    let cmf = data[0];
    let flg = data[1];
    if cmf & 0x0f != 8 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("zlib CMF = {cmf:#04x}（低 4 位须为 8 = deflate）"),
            "该子块不是 zlib 流；确认压缩类型与 DWA 头一致",
        ));
    }
    if ((cmf as u16) << 8 | flg as u16) % 31 != 0 {
        return Err(crate::exr::comp_err(
            LOC,
            "zlib 头 FCHECK 校验失败",
            "zlib 头损坏；确认文件完整",
        ));
    }
    if flg & 0x20 != 0 {
        return Err(crate::exr::comp_err(
            LOC,
            "zlib 流声明了 FDICT（预置字典）",
            "OpenEXR 不使用预置字典；该流不是本模块可解的 EXR 子块",
        ));
    }
    let out = crate::exr::inflate_raw(&data[2..])?;
    if out.len() != expect {
        return Err(crate::exr::comp_err(
            LOC,
            format!("zlib 解压得 {} 字节，声明 {expect} 字节", out.len()),
            "DWA 子块损坏或头字段错误",
        ));
    }
    Ok(out)
}

/// AC 区解码：按 `ac_compression` 分派。
fn decode_ac(compression: u64, buf: &[u8], count: usize) -> ExrResult<Vec<u16>> {
    match compression {
        0 => huf_uncompress(buf, count),
        1 => {
            let bytes = count.checked_mul(USHORT_BYTES).ok_or_else(|| {
                crate::exr::comp_err(
                    "exr::dwa::ac",
                    "AC 项数溢出",
                    "DWA 头字段损坏",
                )
            })?;
            let raw = zlib_decompress(buf, bytes)?;
            let mut out = Vec::with_capacity(count);
            for c in raw.chunks_exact(USHORT_BYTES) {
                let lo = c.first().copied().unwrap_or(0);
                let hi = c.get(1).copied().unwrap_or(0);
                out.push(u16::from_le_bytes([lo, hi]));
            }
            Ok(out)
        }
        other => Err(crate::exr::comp_err(
            "exr::dwa::ac",
            format!("未知 AC 压缩策略 {other}"),
            "仅支持 0（Huffman）/ 1（deflate）；确认文件未损坏",
        )),
    }
}

/// ZIP 预测器还原 + 字节交织的逆变换（官方 `ImfZip.cpp` 的 `reconstruct_scalar`
/// + `interleave_scalar`）。
///
/// # 为什么 DC 区需要这一步（W13 互通对拍抓到的第二个真实 bug）
///
/// DWA 的四个区里**只有 DC 区**走官方的 `Imf::Zip` 压缩器
/// （`ImfDwaCompressor.cpp:2241compress` / `:2592 uncompress`），
/// 而 `Imf::Zip` 不是「裸 zlib」——它的 `compress` 在 deflate **之前**先把
/// 字节按奇偶位重排（`_tmpBuffer` 前半 = 原字节 0,2,4…；后半 = 1,3,5…），
/// `uncompress` 在 inflate **之后**依次做
/// ① **delta 预测器还原** ② **奇偶交织还原**。
///
/// 本实现原先对 DC 区只做裸 inflate，于是每个 u16 的两个字节被错位摆放，
/// 解出的是**完全无关的数值**。实测症状：官方常量块 `0.5`（1024 像素，
/// 官方读出checksum 512.5）被我们读成 `-0.0`——整幅图全零。
///
/// 算术自证（`c_dwaa-half-scan-1-const.exr`，DC 区 14 字节 → 32 字节）：
/// ```text
/// 裸 inflate       : d7 80 80 80 80 … 80 ee 80 80…   ← 噪声，DC = -1.28e-5
/// + 预测器还原     : d7 d7 d7 d7 … d7 45 45 45 …      ← 呈现"分组"结构
/// + 交织还原       : d7 45 d7 45 d7 45 …            ← 全部 = 0x45d7
/// ```
/// `0x45d7` 作为 half = 5.83984375；经 `dctInverse8x8DcOnly`
/// （×0.3535536² = ×0.125）得 0.72998133 → half `0x39d7`
/// → `_toLinear` 查表得 `0x3801` = **0.50048828125**，与官方一致。
///
/// AC 区（`STATIC_HUFFMAN` / `DEFLATE`）、UNKNOWN 区、RLE 区都**不用**
/// `Imf::Zip`，是裸 zlib，故 [`zlib_decompress`] 对它们仍然正确。
pub fn zip_decompress(data: &[u8], expect: usize) -> ExrResult<Vec<u8>> {
    // （错误统一由 `zlib_decompress` 以同一LOC 结构化上报。）
    // ① zlib 解压（`::uncompress`）。
    let mut buf = zlib_decompress(data, expect)?;
    if buf.is_empty() {
        return Ok(buf);
    }
    // ② delta 预测器还原：`t[i] = t[i-1] + t[i] - 128`（逐字节，符号扩展 int）。
    //    对应官方 `reconstruct_scalar`。
    for i in 1..buf.len() {
        let prev = i32::from(buf[i - 1]);
        let cur = i32::from(buf[i]);
        buf[i] = (prev + cur - 128).rem_euclid(256) as u8;
    }
    // ③ 奇偶交织还原：前半（升）取偶下标，后半取奇下标，交叉写出。
    //    对应官方 `interleave_scalar`。
    let half = buf.len().div_ceil(2);
    let mut out: Vec<u8> = Vec::with_capacity(buf.len());
    let (mut lo, mut hi) = (0usize, half);
    while out.len() < buf.len() {
        out.push(buf[lo]);
        lo += 1;
        if out.len() < buf.len() {
            out.push(buf[hi]);
            hi += 1;
        }
    }
    debug_assert_eq!(out.len(), buf.len());
    Ok(out)
}

/// 解析 DWA 块头（11 × i64，小端）。
pub fn parse_sizes(data: &[u8]) -> ExrResult<DwaSizes> {
    const LOC: &str = "exr::dwa::parse_sizes";
    let need = NUM_SIZES_SINGLE * 8;
    if data.len() < need {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 块头仅 {} 字节（需 {need} 字节）", data.len()),
            "DWA 块被截断；确认文件完整",
        ));
    }
    let mut v = [0u64; NUM_SIZES_SINGLE];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        for (k, byte) in b.iter_mut().enumerate() {
            *byte = data.get(i * 8 + k).copied().unwrap_or(0);
        }
        *slot = u64::from_le_bytes(b);
    }
    let s = DwaSizes {
        version: v[0],
        unknown_uncompressed_size: v[1],
        unknown_compressed_size: v[2],
        ac_compressed_size: v[3],
        dc_compressed_size: v[4],
        rle_compressed_size: v[5],
        rle_uncompressed_size: v[6],
        rle_raw_size: v[7],
        ac_uncompressed_count: v[8],
        dc_uncompressed_count: v[9],
        ac_compression: v[10],
    };
    // 负值检测（OpenEXR 按 int64 解释并拒绝负数）。
    for (name, val) in [
        ("unknownUncompressedSize", s.unknown_uncompressed_size),
        ("unknownCompressedSize", s.unknown_compressed_size),
        ("acCompressedSize", s.ac_compressed_size),
        ("dcCompressedSize", s.dc_compressed_size),
        ("rleCompressedSize", s.rle_compressed_size),
        ("rleUncompressedSize", s.rle_uncompressed_size),
        ("rleRawSize", s.rle_raw_size),
        ("acUncompressedCount", s.ac_uncompressed_count),
        ("dcUncompressedCount", s.dc_uncompressed_count),
    ] {
        if (val as i64) < 0 {
            return Err(crate::exr::comp_err(
                LOC,
                format!("DWA 头字段 {name} = {val} 为负（按 int64 解释）"),
                "DWA 块头损坏；确认文件完整",
            ));
        }
    }
    Ok(s)
}

/// DWA 块内四个数据区的**字节定位**（相对块负载起点）。
///
/// # 用途
///
/// 定位表（[`parse_sizes`]）只给**长度**，不给**位置**——四个区在负载里是
/// 按 `unknown → AC → DC → RLE` 的固定顺序紧密排列的，要取某一区的原始字节
/// 必须自己把前面几区的长度加起来。互通对拍（`tests/interop/`）需要把 AC 区
/// 单独抽出来交给官方 `Imf::hufUncompress` 逐项比对，就属这类需求。
///
/// 属**诊断/验证**接口：正常解码走 [`decompress_block`]，无需本结构。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwaBlockRegions {
    /// version ≥ 2 时块首规则表占用的字节数（version 0/1 为 0）。
    pub rules_len: usize,
    /// UNKNOWN 区（zlib）字节区间。
    pub unknown: (usize, usize),
    /// AC 区字节区间。
    pub ac: (usize, usize),
    /// DC 区（zlib）字节区间。
    pub dc: (usize, usize),
    /// RLE 区（zlib）字节区间。
    pub rle: (usize, usize),
}

/// 定位一个 DWA 块负载内四个数据区的字节区间。
///
/// 与 [`decompress_block`] 用**同一套**偏移算法（同一份 [`parse_sizes`] +
/// 同一份规则表长度计算），故两者不可能漂移。
pub fn block_regions(payload: &[u8]) -> ExrResult<(DwaSizes, DwaBlockRegions)> {
    const LOC: &str = "exr::dwa::block_regions";
    let sizes = parse_sizes(payload)?;
    let head = NUM_SIZES_SINGLE * 8;
    // 规则表长度：version ≥ 2 时块首带表，需实际解析才知道占多少字节。
    let rules_len = if sizes.version >= 2 {
        parse_rules(payload.get(head..).unwrap_or(&[]))?.1
    } else {
        0
    };
    let mut off = head + rules_len;
    let mut take = |len: u64| -> (usize, usize) {
        let start = off;
        off += len as usize;
        (start, len as usize)
    };
    let regions = DwaBlockRegions {
        rules_len,
        unknown: take(sizes.unknown_compressed_size),
        ac: take(sizes.ac_compressed_size),
        dc: take(sizes.dc_compressed_size),
        rle: take(sizes.rle_compressed_size),
    };
    // 越界检查：定位表必须落在负载内，否则说明块头与实际长度不符。
    if off > payload.len() {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 子块需 {off} 字节，实际 {} 字节", payload.len()),
            "DWA 块被截断；确认文件完整",
        ));
    }
    Ok((sizes, regions))
}

/// 解析 version ≥ 2 的通道规则表。
fn parse_rules(data: &[u8]) -> ExrResult<(Vec<DwaRule>, usize)> {
    const LOC: &str = "exr::dwa::parse_rules";
    if data.len() < 2 {
        return Err(crate::exr::comp_err(
            LOC,
            "DWA 规则表短于 2 字节长度头",
            "DWA 块被截断；确认文件完整",
        ));
    }
    let rule_size = u16::from_le_bytes([data[0], data[1]]) as usize;
    if rule_size < 2 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 规则表长度 {rule_size} 非法（须 ≥ 2）"),
            "DWA 块头损坏",
        ));
    }
    if data.len() < rule_size {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 规则表需 {rule_size} 字节，实际 {} 字节", data.len()),
            "DWA 块被截断；确认文件完整",
        ));
    }
    let mut rules: Vec<DwaRule> = Vec::new();
    let mut p = 2usize;
    let end = rule_size;
    while p < end {
        // 后缀：NUL 结尾的 ASCII，限长 `Name::SIZE`。
        let mut suffix = String::new();
        while p < end {
            let c = data.get(p).copied().unwrap_or(0);
            p += 1;
            if c == 0 {
                break;
            }
            if suffix.len() < 255 {
                suffix.push(c as char);
            }
        }
        let v1 = data.get(p).copied().unwrap_or(0);
        p += 1;
        let csc_idx = ((v1 >> 4) as i32) - 1;
        if !(-1..3).contains(&csc_idx) {
            return Err(crate::exr::comp_err(
                LOC,
                format!("DWA 规则 cscIdx = {csc_idx} 越界（须 -1..2）"),
                "DWA 规则表损坏",
            ));
        }
        let scheme = DwaScheme::from_bits((v1 >> 2) & 3).ok_or_else(|| {
            crate::exr::comp_err(
                LOC,
                format!("DWA 规则 scheme = {} 非法（须 0..2）", (v1 >> 2) & 3),
                "DWA 规则表损坏",
            )
        })?;
        let case_insensitive = v1 & 1 != 0;
        let v2 = data.get(p).copied().unwrap_or(0);
        p += 1;
        let pt = crate::exr::PixelType::from_code(v2 as i32).ok_or_else(|| {
            crate::exr::comp_err(
                LOC,
                format!("DWA 规则像素类型 {v2} 非法"),
                "DWA 规则表损坏",
            )
        })?;
        rules.push(DwaRule {
            suffix,
            scheme,
            pixel_type: pt,
            csc_idx,
            case_insensitive,
        });
    }
    Ok((rules, rule_size))
}

/// 从 `tail` 的 `*off` 处取 `n` 字节子块并前移游标（越界则结构化报错）。
fn take_sub<'a>(tail: &'a [u8], off: &mut usize, n: u64) -> ExrResult<&'a [u8]> {
    const LOC: &str = "exr::dwa::decompress_block";
    let n_usize = n as usize;
    let end = off.checked_add(n_usize).ok_or_else(|| {
        crate::exr::comp_err(LOC, "DWA 子块偏移溢出", "DWA 块头损坏")
    })?;
    let s = tail.get(*off..end).ok_or_else(|| {
        crate::exr::comp_err(
            LOC,
            format!("DWA 子块需要 [{}..{end}) 字节，剩余 {} 字节", *off, tail.len()),
            "DWA 块被截断",
        )
    })?;
    *off = end;
    Ok(s)
}

/// 解一个 DWA 块，输出**块内 EXR 布局**的原始字节
/// （逐扫描线 × 逐通道 × 逐像素，与 `unpack_block` 的期望一致）。
///
/// - `payload`：DWA 块负载（不含外层 `y` / `dataSize`）。
/// - `ctx`：几何与通道上下文。
/// - `out_lines`：块内**实际**扫描线数（末块可能不足整块）。
pub fn decompress_block(
    payload: &[u8],
    ctx: &DwaContext<'_>,
    out_lines: usize,
) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr::dwa::decompress_block";
    let sizes = parse_sizes(payload)?;
    if sizes.version > 2 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 块版本 {} > 2（当前只支持 0/1/2）", sizes.version),
            "该文件由更新版 OpenEXR 写出；请用 OpenEXR 3.1 转存为 DWA v2",
        ));
    }

    // 规则表（version ≥ 2 时位于块首）。
    let mut cur = NUM_SIZES_SINGLE * 8;
    let rules: Vec<DwaRule> = if sizes.version < 2 {
        legacy_rules()
    } else {
        let (r, used) = parse_rules(payload.get(cur..).unwrap_or(&[]))?;
        cur += used;
        r
    };

    // 四个数据区按 fixed 顺序排列。
    let total = sizes
        .unknown_compressed_size
        .checked_add(sizes.ac_compressed_size)
        .and_then(|v| v.checked_add(sizes.dc_compressed_size))
        .and_then(|v| v.checked_add(sizes.rle_compressed_size))
        .ok_or_else(|| crate::exr::comp_err(LOC, "DWA 子块长度求和溢出", "DWA 块头损坏"))?;
    let need = (cur as u64).checked_add(total).ok_or_else(|| {
        crate::exr::comp_err(LOC, "DWA 子块总长求和溢出", "DWA 块头损坏")
    })?;
    if (payload.len() as u64) < need {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 子块需 {need} 字节，实际 {} 字节", payload.len()),
            "DWA 块被截断；确认文件完整",
        ));
    }
    let tail = payload.get(cur..).unwrap_or(&[]);
    let mut off = 0usize;
    let unknown_buf = take_sub(tail, &mut off, sizes.unknown_compressed_size)?;
    let ac_buf = take_sub(tail, &mut off, sizes.ac_compressed_size)?;
    let dc_buf = take_sub(tail, &mut off, sizes.dc_compressed_size)?;
    let rle_buf = take_sub(tail, &mut off, sizes.rle_compressed_size)?;

    // 解码四个区。
    let unknown_raw = if sizes.unknown_compressed_size > 0 {
        zlib_decompress(unknown_buf, sizes.unknown_uncompressed_size as usize)?
    } else {
        Vec::new()
    };
    let ac = if sizes.ac_compressed_size > 0 {
        decode_ac(
            sizes.ac_compression,
            ac_buf,
            sizes.ac_uncompressed_count as usize,
        )?
    } else {
        Vec::new()
    };
    let dc = if sizes.dc_compressed_size > 0 {
        let bytes = (sizes.dc_uncompressed_count as usize)
            .checked_mul(USHORT_BYTES)
            .ok_or_else(|| {
                crate::exr::comp_err(LOC, "DC 项数溢出", "DWA 块头损坏")
            })?;
        // ⚠️ DC 区走官方 `Imf::Zip`（预测器 + 奇偶交织），**不是**裸 zlib。
        // AC / UNKNOWN / RLE 才是裸 zlib。见 [`zip_decompress`] 的完整根因。
        let raw = zip_decompress(dc_buf, bytes)?;
        raw.chunks_exact(USHORT_BYTES)
            .map(|c| {
                u16::from_le_bytes([c.first().copied().unwrap_or(0), c.get(1).copied().unwrap_or(0)])
            })
            .collect()
    } else {
        if sizes.dc_uncompressed_count != 0 {
            return Err(crate::exr::comp_err(
                LOC,
                "DC 压缩长度为 0 但声明了非零解压长度",
                "DWA 块头损坏",
            ));
        }
        Vec::new()
    };
    let rle_raw: Vec<u8> = if sizes.rle_raw_size > 0 {
        let packed = zlib_decompress(rle_buf, sizes.rle_uncompressed_size as usize)?;
        crate::exr::rle_decompress_raw(
            &packed,
            sizes.rle_uncompressed_size as usize,
            sizes.rle_raw_size as usize,
        )?
    } else {
        Vec::new()
    };

    assemble(
        &sizes,
        ctx,
        out_lines,
        &rules,
        &unknown_raw,
        &ac,
        &dc,
        &rle_raw,
    )
}

/// 把四个区组装成块内 EXR 布局字节。
#[allow(clippy::too_many_arguments)]
fn assemble(
    _sizes: &DwaSizes,
    ctx: &DwaContext<'_>,
    out_lines: usize,
    rules: &[DwaRule],
    unknown_raw: &[u8],
    ac: &[u16],
    dc: &[u16],
    rle_raw: &[u8],
) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr::dwa::assemble";
    let plan = classify_channels(ctx.channels, rules);
    let width = ctx.width;
    let nch = plan.channels.len();

    // 每通道的像素字节数。
    let widths: Vec<usize> = plan
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width())
        .collect();
    let row_bytes: usize = widths.iter().sum::<usize>().saturating_mul(width);
    let total_bytes = row_bytes.saturating_mul(out_lines);

    // ---- 逐通道平面（每通道 out_lines × width 的原始像素字节）----
    let mut planes: Vec<Vec<u8>> = vec![Vec::new(); nch];

    // 游标：unknown / rle 区的消费位置（按方案顺序推进，顺序确定）。
    let mut unknown_cur = 0usize;
    let mut rle_cur = 0usize;

    // DC / AC 游标：与 OpenEXR 一致，**每个解码器（decoder）顺序消费**：
    // 先全部 CSC 组（按 csc_sets 序），再按通道序处理剩余通道。
    let mut dc_cur = 0usize;
    let mut ac_cur = 0usize;

    let blocks_x = width.div_ceil(DCT_N);
    let blocks_y = out_lines.div_ceil(DCT_N);

    // ---- LOSSY_DCT 通道解码（单通道或 CSC 三通道）----
    let mut decoded = vec![false; nch];

    // CSC 组：OpenEXR 保证组内三通道都是 LOSSY_DCT（否则它自己抛错）。
    for set in &plan.csc_sets {
        for k in 0..3 {
            let ci = set.idx[k];
            if plan.channels.get(ci).map(|c| c.scheme) != Some(DwaScheme::LossyDct) {
                return Err(crate::exr::comp_err(
                    LOC,
                    "CSC 组内通道不是 LOSSY_DCT",
                    "DWA 通道分类与数据不一致；确认文件未损坏",
                ));
            }
        }
        // CSC 三通道共享**同一个** DC/AC 解码器：AC 按 blocky→blockx→comp
        // 交错消费；DC 是 comp-major 的独立平面（偏移 comp * bx * by）。
        // OpenEXR 对 CSC 组**无条件**应用 dwaCompressorToLinear（不看 pLinear）。
        let group = [
            (set.idx[0], plan.channels[set.idx[0]].pixel_type),
            (set.idx[1], plan.channels[set.idx[1]].pixel_type),
            (set.idx[2], plan.channels[set.idx[2]].pixel_type),
        ];
        let out = decode_lossy_dct_group(
            width,
            out_lines,
            blocks_x,
            blocks_y,
            &group,
            true, // apply_to_linear：CSC 恒为 true
            dc,
            &mut dc_cur,
            ac,
            &mut ac_cur,
        )?;
        for (k, bytes) in out.into_iter().enumerate() {
            let ci = set.idx[k];
            if let Some(p) = planes.get_mut(ci) {
                *p = bytes;
            }
            if let Some(d) = decoded.get_mut(ci) {
                *d = true;
            }
        }
    }

    for ci in 0..nch {
        if decoded[ci] {
            continue;
        }
        let c = plan.channels[ci];
        let pw = widths[ci];
        let scan = out_lines.saturating_mul(width).saturating_mul(pw);
        match c.scheme {
            DwaScheme::LossyDct => {
                // 单通道解码器；`pLinear` 通道不做非线性 LUT 还原。
                let group = [(ci, c.pixel_type)];
                let out = decode_lossy_dct_group(
                    width,
                    out_lines,
                    blocks_x,
                    blocks_y,
                    &group,
                    !c.p_linear,
                    dc,
                    &mut dc_cur,
                    ac,
                    &mut ac_cur,
                )?;
                if let Some(bytes) = out.into_iter().next() {
                    if let Some(p) = planes.get_mut(ci) {
                        *p = bytes;
                    }
                }
            }
            DwaScheme::Rle => {
                // RLE 区按字节平面拆分（`interleaveByte2` 的 4 字节推广）：
                // 布局 = [字节0 平面][字节1 平面]…，每个平面 `stride` 字节，
                // 其中 `stride` = **像素数** = width × out_lines。
                //
                // ⚠️ 此处 `stride` 必须是像素数而非 `width × pw × out_lines`：
                // 索引式 `rle_cur + b * stride + pix` 里 `pix ∈ [0, stride)`，
                // 若把 `pw` 乘进去，第 1 个及之后的字节平面会整段越界，
                // 4 字节通道（uint / float）将只解出第 0 字节平面。
                // 2 字节分支 [`interleave_half_rle`] 用的正是 `width * out_lines`，
                // 两处必须一致。
                let mut plane: Vec<u8> = vec![0u8; scan];
                let stride = width.saturating_mul(out_lines);
                if pw == 2 {
                    interleave_half_rle(
                        &mut plane,
                        rle_raw,
                        &mut rle_cur,
                        width,
                        out_lines,
                    )?;
                } else {
                    for y in 0..out_lines {
                        for x in 0..width {
                            let pix = y * width + x;
                            for b in 0..pw {
                                let si = rle_cur.saturating_add(b.saturating_mul(stride)).saturating_add(pix);
                                let di = (pix * pw) + b;
                                if let (Some(sv), Some(dv)) = (rle_raw.get(si), plane.get_mut(di)) {
                                    *dv = *sv;
                                }
                            }
                        }
                    }
                    rle_cur = rle_cur.saturating_add(pw.saturating_mul(stride));
                }
                if let Some(p) = planes.get_mut(ci) {
                    *p = plane;
                }
            }
            DwaScheme::Unknown => {
                let mut plane: Vec<u8> = vec![0u8; scan];
                let end = unknown_cur.saturating_add(scan);
                if unknown_raw.len() < end {
                    return Err(crate::exr::comp_err(
                        LOC,
                        format!(
                            "UNKNOWN 区在通道 {ci} 处耗尽（需 {end} 字节，共 {} 字节）",
                            unknown_raw.len()
                        ),
                        "UNKNOWN 区长度与通道表不一致",
                    ));
                }
                plane.copy_from_slice(&unknown_raw[unknown_cur..end]);
                unknown_cur = end;
                if let Some(p) = planes.get_mut(ci) {
                    *p = plane;
                }
            }
        }
        if let Some(d) = decoded.get_mut(ci) {
            *d = true;
        }
    }

    // ---- 交织成 EXR 块布局：逐扫描线 × 逐通道 × 逐像素 ----
    let mut out: Vec<u8> = Vec::with_capacity(total_bytes);
    let line_bytes: Vec<usize> = widths.iter().map(|w| w * width).collect();
    for y in 0..out_lines {
        for ci in 0..nch {
            let lb = line_bytes[ci];
            let start = y.saturating_mul(lb);
            let plane = planes.get(ci).and_then(|p| p.get(start..start + lb));
            match plane {
                Some(s) => out.extend_from_slice(s),
                None => {
                    // 理论不可达：平面长度已按 out_lines × width 精确构造。
                    return Err(crate::exr::comp_err(
                        LOC,
                        format!("通道 {ci} 平面长度不足（缺 {} 字节）", lb),
                        "DWA 内部状态错误",
                    ));
                }
            }
        }
    }
    Ok(out)
}

/// 交织 2 字节 RLE 通道（`interleaveByte2`）。
fn interleave_half_rle(
    plane: &mut [u8],
    rle_raw: &[u8],
    rle_cur: &mut usize,
    width: usize,
    out_lines: usize,
) -> ExrResult<()> {
    const LOC: &str = "exr::dwa::interleave_rle";
    let stride = width.saturating_mul(out_lines);
    for y in 0..out_lines {
        for x in 0..width {
            let pix = y.saturating_mul(width).saturating_add(x);
            for b in 0..2usize {
                let si = rle_cur.saturating_add(b * stride).saturating_add(pix);
                let di = pix.saturating_mul(2).saturating_add(b);
                let sv = rle_raw.get(si).copied().ok_or_else(|| {
                    crate::exr::comp_err(
                        LOC,
                        format!("RLE 区在 ({x},{y}) 越界（需 {si} 字节，共 {} 字节）", rle_raw.len()),
                        "RLE 区长度与通道表不一致",
                    )
                })?;
                if let Some(dv) = plane.get_mut(di) {
                    *dv = sv;
                }
            }
        }
    }
    *rle_cur = rle_cur.saturating_add(2 * stride);
    Ok(())
}

/// 一个 LOSSY_DCT 解码器覆盖的通道（1 个 = 单通道解码器，3 个 = CSC 解码器）。
///
/// 与 OpenEXR `LossyDctDecoderBase::_rowPtrs` 一一对应：`_rowPtrs.size()`
/// 只允许 1 或 3。
type DctGroup<'a> = [(usize, crate::exr::PixelType)];

/// LOSSY_DCT 解码器（`LossyDctDecoderBase::execute`）。
///
/// **严格复刻 OpenEXR 的消费顺序**，这是数值正确性的关键：
///
/// - 外层 `blocky` → `blockx` → `comp`（AC 交错）；
/// - **DC 是 comp-major 的独立平面**：`currDcComp[comp]` 起点相隔
///   `blocks_x * blocks_y`，组内 comp 各自顺序消费自己那一面；
/// - **AC 是组内各 comp 交错的单一顺序流**（`currAcComp` 不分面）；
/// - `blockIsConstant`（全部 comp 只有 DC）时走 DC-only 快路径 + 单点 CSC；
/// - CSC 逆变换在 **IDCT 之后、half 转换之前**逐元素做；
/// - `toLinear` LUT 在 CSC 路径**无条件**应用，在单通道路径仅 `!pLinear` 时应用。
///
/// 返回值与输入 `comps` 顺序一一对应的像素字节（half → 2 字节小端；
/// float → 4 字节小端，由 half 值零扩展/转换而来，与 OpenEXR 的
/// "float 通道也先解成 half 再展开" 一致）。
#[allow(clippy::too_many_arguments)]
fn decode_lossy_dct_group(
    width: usize,
    out_lines: usize,
    blocks_x: usize,
    blocks_y: usize,
    comps: &DctGroup<'_>,
    apply_to_linear: bool,
    dc: &[u16],
    dc_cur: &mut usize,
    ac: &[u16],
    ac_cur: &mut usize,
) -> ExrResult<Vec<Vec<u8>>> {
    const LOC: &str = "exr::dwa::lossy_dct";
    let num_comp = comps.len();
    if num_comp != 1 && num_comp != 3 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("LOSSY_DCT 解码器只支持 1 或 3 通道，实际 {num_comp}"),
            "DWA 通道分类异常",
        ));
    }
    let plane_len = width.saturating_mul(out_lines);
    // 每 comp 一个 half 位模式缓冲。
    let mut half_out: Vec<Vec<u16>> = (0..num_comp).map(|_| vec![0u16; plane_len]).collect();
    // 每 comp 一个 f32 块缓冲（IDCT 输出，natural order）。
    let mut dct_buf: Vec<[f32; DCT_COEFFS]> = vec![[0.0f32; DCT_COEFFS]; num_comp];
    // 每 comp 的 zigzag 半精度块。
    let mut zig_buf: Vec<[u16; DCT_COEFFS]> = vec![[0u16; DCT_COEFFS]; num_comp];

    // DC 平面基址：comp c 从 `dc_base + c * blocks_x * blocks_y` 起。
    let plane_size = blocks_x.saturating_mul(blocks_y);
    let dc_base = *dc_cur;
    let mut dc_comp = [0usize; 3];
    for (c, slot) in dc_comp.iter_mut().enumerate().take(num_comp) {
        *slot = dc_base.saturating_add(c.saturating_mul(plane_size));
    }
    let dc_end = dc_base.saturating_add(num_comp.saturating_mul(plane_size));

for blocky in 0..blocks_y {
            for blockx in 0..blocks_x {
                let block_index = blocky.saturating_mul(blocks_x).saturating_add(blockx);

            for comp in 0..num_comp {
                // ---- DC：comp-major 平面内的第 block_index 项 ----
                let di = dc_comp[comp].saturating_add(block_index);
                let dc_val = dc.get(di).copied().ok_or_else(|| {
                    crate::exr::comp_err(
                        LOC,
                        format!(
                            "DC 缓冲耗尽（需第 {di} 项 / 共 {} 项，声明上限 {dc_end}）",
                            dc.len()
                        ),
                        "DC 区长度与块数不一致",
                    )
                })?;

                // ---- AC：组内 comp 交错的单一顺序流 ----
                let zig = &mut zig_buf[comp];
                *zig = [0u16; DCT_COEFFS];
                zig[0] = dc_val;
                let last_non_zero = un_rle_ac(ac, ac_cur, zig)?;

                let coefs = &mut dct_buf[comp];
                if last_non_zero == 0 {
                    // DC-only：`_dctData[comp]._buffer[0] = (float)h` 后铺满。
                    *coefs = [0.0f32; DCT_COEFFS];
                    coefs[0] = f16_to_f32(zig[0]);
                    dct_inverse_dc_only(coefs);
                } else {
                    from_half_zig_zag(zig, coefs);
                    dct_inverse_8x8(coefs);
                }
            }

            // ---- CSC 逆变换（IDCT 之后、half 转换之前）----
            if num_comp == 3 {
                // 三通道取 `dct_buf[0] / [1] / [2]` 三个**互不重叠**的可变引用。
                // ⚠️ 此前写作 `let (ab, c) = split_at_mut(2); let (b2, c2) =
                // c.split_at_mut(1);` —— `c` 只含 comp2，再切一刀得到的 `c2` 长度
                // 为 0，`c2[0]` 必然越界 panic（C18 违例）。该缺陷在只有单通道
                // 解码测试时不可见；R/G/B 三通道（CSC 组）一来就触发，已修。
                let mut it = dct_buf.iter_mut();
                let a0 = it.next();
                let a1 = it.next();
                let a2 = it.next();
                if let (Some(a0), Some(a1), Some(a2)) = (a0, a1, a2) {
                    // 【修正】**无条件**整块逆 CSC。
                    //
                    // 此前用块级 `block_is_constant` 分派：全块 DC-only 时只对
                    // `[0]` 逆 CSC。但 DC-only 路径已由 `dct_inverse_dc_only`
                    // 把 DC **铺满全部 64 个元素**，元素 1..=63 仍是前向 CSC
                    // （Y/Cb/Cr）域的值 —— 逆变换漏掉 63/64 个像素 ⇒ 彩色块整片偏色。
                    //
                    // 改为无条件 `csc709_inverse64`：常量块各元素本就相同，逆变换
                    // 后仍是同一常量，**结果与只逆 `[0]` 一致**，但不再依赖块级/通道级
                    // 判定是否一致（3 通道 AC/DC 分叉时该判定本就不可靠）。
                    csc709_inverse64(a0, a1, a2);
                }
            }

            // ---- float → half，随后（可选）toLinear LUT，写回块内位置 ----
            for comp in 0..num_comp {
                for ly in 0..DCT_N {
                    let y = blocky * DCT_N + ly;
                    if y >= out_lines {
                        break;
                    }
                    for lx in 0..DCT_N {
                        let x = blockx * DCT_N + lx;
                        if x >= width {
                            break;
                        }
                        let h = f32_to_f16(dct_buf[comp][ly * DCT_N + lx]);
                        let bits = if apply_to_linear { dwa_to_linear(h) } else { h };
                        if let Some(slot) = half_out[comp].get_mut(y * width + x) {
                            *slot = bits;
                        }
                    }
                }
            }
        }
    }
    *dc_cur = dc_end;

    // half 位模式 → 目标像素类型的字节。
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(num_comp);
    for (comp, (_, ptype)) in comps.iter().enumerate() {
        let mut bytes: Vec<u8> = Vec::with_capacity(half_out[comp].len() * ptype.byte_width());
        for h in &half_out[comp] {
            let v = f16_to_f32(*h);
            match ptype {
                crate::exr::PixelType::Half => bytes.extend_from_slice(&h.to_le_bytes()),
                crate::exr::PixelType::Float => bytes.extend_from_slice(&v.to_le_bytes()),
                other => {
                    return Err(crate::exr::comp_err(
                        LOC,
                        format!("LOSSY_DCT 不支持像素类型 {other:?}"),
                        "DWA 只对 HALF / FLOAT 通道做有损 DCT",
                    ));
                }
            }
        }
        out.push(bytes);
    }
    Ok(out)
}

/// AC 游程展开（`unRleAc`）：返回 zigzag 序中最后一个非零项的下标（0 = DC-only）。
///
/// # 循环条件必须是「`dct_comp < 64`」，不是「读到 `0xff00` 为止」
///
/// 这是 W13 互通对拍抓到的**第一个真实互操作 bug**，且根因就在这个条件上。
/// 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressor.cpp:1205`
/// （`LossyDctDecoderBase::unRleAc`）：
///
/// ```cpp
/// int dctComp = 1;
/// while (dctComp < 64) {
///     if (*currAcComp == 0xff00)   dctComp = 64;            // EOB（可选！）
///     else if ((*currAcComp)>>8 == 0xff) dctComp += (*currAcComp) & 0xff;
///     else { halfZigBlock[dctComp] = *currAcComp; dctComp++; }
///     _packedAcCount++; currAcComp++;
/// }
/// ```
///
/// 即：**AC 流是跨 comp 扁平拼接的，每个 comp 的结束由「写满 64 个系数槽」
/// 决定，`0xff00` 只是编码器在「末尾零游程恰好填满」时顺带写下的 EOB，
/// **并非每comp 必有**。看��侧 `rleAc`（同文件:1706）的写法更清楚：
/// 只有 `runLen + dctComp == 64` 时才写 `0xff00`，否则写 `0xff00|runLen`。
///
/// 本实现原先以「读到 `0xff00`」为终止条件。对**自家编码器**产出的流碰巧成立
/// （自家 `rle_ac_encode` 无条件补一枚 `0xff00`），于是内部往返 100% 全绿——
/// 正是账本 v2.8 记的那个风险：「自洽 ≠ 符合规范」。但官方写的文件里
/// 高频噪声这种形态**一个 `0xff00` 都没有**（实测 6047 项里 `0xff00` 出现
/// **0** 次，只有 1 枚游程标记 `0xff02`），本实现随即在读完 6047 项后
/// 越界报错 `AC 缓冲耗尽`。
///
/// 算术自证（DWAA 半 3 通道 64×48，块高 32 → 8×4×3 = 96 comp）：
/// `96 comp × 63 AC 槽 = 6048`，而流里是 6046 个字面量 + 1 枚 `0xff02`
/// （覆盖 2 个零）= **6048**，一位不多一位不少 —— 证实「扁平流 + 无 EOB」。
fn un_rle_ac(ac: &[u16], ac_cur: &mut usize, zig: &mut [u16; DCT_COEFFS]) -> ExrResult<usize> {
    const LOC: &str = "exr::dwa::unrle_ac";
    let mut last_non_zero = 0usize;
    // zigzag 下标 0 是 DC（已由 DC 区给出），AC 从 1 开始 —— 与官方同。
    let mut dct_comp = 1usize;
    // 官方块是零初始化的（注释原文："Start with a zero'ed block, so we don't
    // have to write when we hit a run symbol"），故游程覆盖的槽天然为 0。
    // 调用方传入的 `zig` 也必须是零块，本函数只写非零项。
    while dct_comp < DCT_COEFFS {
        let v = ac.get(*ac_cur).copied().ok_or_else(|| {
            crate::exr::comp_err(
                LOC,
                format!("AC 缓冲耗尽（已取 {} 项，共 {} 项）", *ac_cur, ac.len()),
                "AC 区长度与块数不一致；确认文件完整",
            )
        })?;
        *ac_cur += 1;
        if v == 0xff00 {
            // EOB：官方是 `dctComp = 64`（填满即止），不 break 出外层语义差别，
            // 但本循环条件恰是 `dct_comp < 64`，故直接落到同一条终止路径。
            dct_comp = DCT_COEFFS;
        } else if (v >> 8) == 0xff {
            // 游程：高字节 0xff，低字节为后续零项数（这些槽保持 0）。
            dct_comp = dct_comp.saturating_add((v & 0xff) as usize);
        } else {
            last_non_zero = dct_comp;
            if let Some(slot) = zig.get_mut(dct_comp) {
                *slot = v;
            }
            dct_comp += 1;
        }
    }
    Ok(last_non_zero)
}

// ===========================================================================
//  编码器（DWAA / DWAB）—— 与上面的解码器逐字段对逆
// ===========================================================================
//
// # 设计口径
//
// 编码 = 解码的**严格对逆**：量化 → 正向 DCT 8×8 → 之字形 → AC 游程（
// `un_rle_ac` 的对逆）→ AC 熵编码（Huffman 或 deflate）+ DC/UNKNOWN/RLE 三区
// 各自 zlib → 11 × i64 块头。逐字段与解码器的消费顺序对齐（CSC 组优先、
// comp-major 的 DC 平面、组内交错的单一 AC 流）。
//
// ## 有损是规范行为（诚实边界）
//
// DWA 本身是**有损**压缩（AC 系数量化 + DCT 舍入 + half 存储），往返断言使用
// **有损阈值**而非逐位相等。阈值依据见模块级 `dwa_round_trip_error_bound`
// 测试：量化把 AC 系数按 `QUANT_STEP` 步长取整，单系数绝对误差 ≤ `QUANT_STEP/2`；
// 经 IDCT（增益 ≤ 1）后单像素绝对误差上界 ≈ `QUANT_STEP/2 + half 舍入`。
//
// ## 与官方 OpenEXR 比特流的关系（不互通，如实声明）
//
// 本编码器产出的是**本引擎自洽**的 DWA 块：头部/子块布局/AC 游程/Huffman 信封
// 严格遵循 OpenEXR 3.1.5 的真实格式（可被官方解码器解析结构），但**量化表、
// 通道规则表的默认内容、Huffman 码表构造**是本引擎自研版本，**未**与官方
// `ImfDwaCompressor` 的比特流做过逐位比对。故 **不保证与官方 OpenEXR 比特流
// 互换**；保证的是「本引擎编码 → 本引擎解码」往返自洽（有测试）。

/// DWA 基础量化误差 `quantBaseError`（官方 `ImfDwaCompressor.cpp:2081`）。
///
/// 官方算法：`quantBaseError = _dwaCompressionLevel / 100000.f`，其中
/// `_dwaCompressionLevel` 来自 `hdr.dwaCompressionLevel()`，默认值
/// `s_DefaultDwaCompressionLevel = 45.f`（`ImfHeader.cpp:69`）。
/// 故 `= 45 / 100000 = 4.5e-4`。
const QUANT_BASE_ERROR: f32 = 45.0 / 100000.0;

/// 官方 JPEG 示例量化表（`ImfDwaCompressor.cpp:1297`，逐项抄录，**raster 序**）。
///
/// 官方注释：「take the generic JPEG quantization tables and normalize them by
/// the smallest component in each. This gives us a relationship amongst the DCT
/// components, in terms of how sensitive each component is to error. A higher
/// normalized value means we can quantize more.」
const JPEG_QUANT_TABLE_Y: [u16; DCT_COEFFS] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60,
    55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87,
    80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
    104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98,
    112, 100, 103, 99,
];

/// Y 表的最小值，归一化分母（`jpegQuantTableYMin`，官方硬编码 10）。
const JPEG_QUANT_TABLE_Y_MIN: u16 = 10;

/// 官方 JPEG 示例色度量化表（`ImfDwaCompressor.cpp:1315`，逐项抄录，**raster 序**）。
const JPEG_QUANT_TABLE_CBCR: [u16; DCT_COEFFS] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99,
    99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99,
];

/// CbCr 表的最小值，归一化分母（`jpegQuantTableCbCrMin`，官方硬编码 17）。
const JPEG_QUANT_TABLE_CBCR_MIN: u16 = 17;

/// 某系数序号 `raster_idx` 的**容差**：官方用「表值 × `quantBaseError`」。
///
/// 对应 `ImfDwaCompressor.cpp:1536`（comp 0 = Y）与 `:1545`（comp 1/2 = CbCr）：
/// ```text
/// quantize ((half)buf[i], _quantBaseError * _quantTableY[i]);     // chan == 0
/// quantize ((half)buf[i], _quantBaseError * _quantTableCbCr[i]);  // chan  >  0
/// ```
///
/// 【W14 路1 修正】本实现原先用**统一**步长 `QUANT_STEP = 1/64`
/// （单系数误差界 `7.8125e-3`）。逐项对比官方容差：
///
/// | 项 | 官方容差 | 我们误差界 | 粗倍数 |
/// |---|---|---|---|
/// | Y 表最小项（idx 2，`10/10`） | `4.500e-4` | `7.8125e-3` | **17.4×** |
/// | Y 表 idx 0（`16/10`） | `7.200e-4` | `7.8125e-3` | **10.9×** |
/// | Y 表 idx 1（`11/10`） | `4.950e-4` | `7.8125e-3` | **15.8×** |
/// | CbCr 表最小项（idx 0，`17/17`） | `4.500e-4` | `7.8125e-3` | **17.4×** |
/// | Y 表最大项（idx 53，`121/10`） | `5.445e-3` | `7.8125e-3` | 1.43× |
///
/// 即：**低频项被我们粗了 10~17 倍** —— 而低频 DCT 项的 IDCT 增益最高，
/// 误差直接放大到像素域。这正是单通道 half + 高频噪声格
/// （`dwaa-half-scan-1-noise`）误差大 56 倍的根因。
#[inline]
fn quant_tolerance(comp: usize, raster_idx: usize) -> f32 {
    let i = raster_idx % DCT_COEFFS;
    if comp == 0 {
        QUANT_BASE_ERROR * (JPEG_QUANT_TABLE_Y[i] as f32 / JPEG_QUANT_TABLE_Y_MIN as f32)
    } else {
        QUANT_BASE_ERROR * (JPEG_QUANT_TABLE_CBCR[i] as f32 / JPEG_QUANT_TABLE_CBCR_MIN as f32)
    }
}

/// 之字形扫描顺序（`ZIGZAG_TO_RASTER` 的逆序：raster → zigzag）。
///
/// 与 [`from_half_zig_zag`] 的 `MAP` **严格互为逆置换**：`ZIGZAG[MAP[i]] == i`。
/// 编码端用它把自然序的 64 个系数排成 zigzag 序。
const RASTER_TO_ZIGZAG: [usize; DCT_COEFFS] = {
    // MAP = ZIGZAG_TO_RASTER（见 `from_half_zig_zag`）；此处硬编码其逆。
    // 通过对 MAP 逐项求逆得到，编译期常量（无运行时计算 → 确定性）。
    let map: [usize; DCT_COEFFS] = [
        0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9,
        11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
    ];
    let mut inv = [0usize; DCT_COEFFS];
    let mut i = 0usize;
    while i < DCT_COEFFS {
        inv[map[i]] = i;
        i += 1;
    }
    inv
};

/// DCT 系数矩阵 `M` 的**符号-常量索引表**（8×8，仅测试用于验证 `MF = I`）。
///
/// `M[j][i]` = `SIGN_CONST[table[j][i]]`；取值编码：0 表示 0；1..=7 表示
/// `+C0..+C6`（A..G）；129..=135 表示 `-C0..-C6`。
///
/// 与 [`dct_inverse_8x8`] 的 1D 运算**逐项对应**（同一套 A..G 常量、同一套
/// 加减组合）。生产路径不需要它（正向变换直接用正交 DCT-II），此处仅作
/// 「解码端 1D 是正交 DCT-III」这一事实的**可测试契约**。
///
///【W12 路E 修正】此表**曾是错的**（落盘时抄坏），现已按 [`dct_inverse_8x8`]
/// 的实测值逐格改正，并由 `m_table_matches_implementation` 钉死：
///
/// - 旧第 0 行是 `[1,0,3,0,1,0,6,0]`——**只抄了偶数下标**（漏掉 B/D/E/G项），
///   正确应为 `[1,2,3,4,1,5,6,7]`；
/// - 旧第 1 行末位 `132` 应为 `133`（`-E` 而非 `-D`）；
/// - 旧第 3 行第 3 列 `132` 应为 `133`（`-E` 而非 `-D`）。
///
/// 改正的依据不是「看起来更整齐」，而是
/// `m_table_matches_implementation` 里那条**从实现反推** `M` 的测量路径。
#[cfg(test)]
const M_TABLE: [[u8; DCT_N]; DCT_N] = [
    [1, 2, 3, 4, 1, 5, 6, 7],
    [1, 4, 6, 135, 129, 130, 131, 133],
    [1, 5, 134, 130, 129, 7, 3, 4],
    [1, 7, 131, 133, 1, 4, 134, 130],
    [1, 135, 131, 5, 1, 132, 134, 2],
    [1, 133, 134, 2, 129, 135, 3, 132],
    [1, 132, 6, 7, 129, 2, 131, 5],
    [1, 130, 3, 132, 1, 133, 6, 135],
];

/// 取 `M_TABLE` 编码对应的 f32 值（`A..G` 顺序 = `C0..C6`）。
#[cfg(test)]
fn m_value(code: u8) -> f32 {
    const C: [f32; 7] = [
        0.353_553_4,  // A = 0.5*cos(pi/4)
        0.490_392_6,  // B = 0.5*cos(pi/16)
        0.461_939_8,  // C = 0.5*cos(pi/8)
        0.415_734_9,  // D = 0.5*cos(3*pi/16)
        0.277_785_5,  // E = 0.5*cos(5*pi/16)
        0.191_342_2,  // F = 0.5*cos(3*pi/8)
        0.097_545_73, // G = 0.5*cos(7*pi/16)
    ];
    if code == 0 {
        return 0.0;
    }
    let neg = code >= 129;
    let idx = (code - if neg { 129 } else { 1 }) as usize;
    let v = C.get(idx).copied().unwrap_or(0.0);
    if neg {
        -v
    } else {
        v
    }
}

/// 1D 正向变换（**正交 DCT-II**，`dct_inverse_8x8` 1D 部分的严格逆）。
///
/// `out[k] = c_k · Σ_n x[n] · cos(π·(n+0.5)·k / 8)`，其中 `c_0 = √(1/8)`、
/// `c_k = √(2/8)`（`k ≥ 1`）。
///
/// # 为什么是这个式子（不是 `Mᵀ` 归一化）
///
/// 解码侧的 `dct_inverse_8x8` 1D 部分（`M`）**不是**「正交矩阵 × 对角缩放」：
/// 实测 `MᵀM` 有明显非对角项（如 `(0,1) = -0.173`），故 `Mᵀ 归一化` 不构成逆。
/// 把 `M` 与正交 DCT-II 矩阵 `F` 相乘得**单位矩阵**（已数值验证 `MF = I`，
/// 逐元素最大偏差 < 1e-7），即 `M` 恰是**正交 DCT-III**、`F = M⁻¹`。
/// 故正向变换取正交 DCT-II。（`dct_forward_inverts_inverse` 测试即此断言。）
fn dct1d_forward(x: &[f32; DCT_N], out: &mut [f32; DCT_N]) {
    // √(1/8) 与 √(2/8)（f32 常量；`cos` 为 libm，跨平台舍入由测试界兜住）。
    const C0: f32 = 0.353_553_4; // sqrt(1/8)
    const CK: f32 = 0.5; // sqrt(2/8) = 0.5
    for k in 0..DCT_N {
        let scale = if k == 0 { C0 } else { CK };
        let mut acc = 0.0f32;
        for (n, &v) in x.iter().enumerate() {
            // cos(π·(n+0.5)·k / 8)：n 升序 → 定序求和。
            let angle = std::f32::consts::PI * (n as f32 + 0.5) * (k as f32) / 8.0;
            acc += v * angle.cos();
        }
        out[k] = scale * acc;
    }
}

/// 8×8 正向 DCT（正交 DCT-II）：**先列后行**（与解码端「先行后列」对逆）。
///
/// 正交变换下先后次序不影响数学结果，仅影响 f32 舍入；此处固定为列→行以保证
/// **确定性**（同输入必同字节）。
fn dct_forward_8x8(data: &mut [f32; DCT_COEFFS]) {
    let mut tmp = [0.0f32; DCT_N];
    // ① 逐列。
    for col in 0..DCT_N {
        let mut src = [0.0f32; DCT_N];
        for k in 0..DCT_N {
            src[k] = *data.get(k * DCT_N + col).unwrap_or(&0.0);
        }
        dct1d_forward(&src, &mut tmp);
        for k in 0..DCT_N {
            if let Some(v) = data.get_mut(k * DCT_N + col) {
                *v = tmp[k];
            }
        }
    }
    // ② 逐行。
    for row in 0..DCT_N {
        let r = row * DCT_N;
        let mut src = [0.0f32; DCT_N];
        for k in 0..DCT_N {
            src[k] = *data.get(r + k).unwrap_or(&0.0);
        }
        dct1d_forward(&src, &mut tmp);
        for k in 0..DCT_N {
            if let Some(v) = data.get_mut(r + k) {
                *v = tmp[k];
            }
        }
    }
}

/// 线性 half → DWA 非线性 half（[`dwa_to_linear`] 的**公式逆**）。
///
/// 解码端 `toLinear`：`|x| ≤ 1 → sign·|x|^2.2`；`|x| > 1 → sign·logBase^(|x|-1)`，
/// `logBase = e^2.2`。本函数是其逆：
/// - `|y| ≤ 1 → sign·|y|^(1/2.2)`；
/// - `|y| > 1 → sign·(1 + ln|y| / 2.2)`（因 `logBase^(a-1) = e^(2.2(a-1))`）。
///
/// NaN / Inf → 0（与解码端一致）。`pLinear` 通道不调用本函数。
fn dwa_from_linear(bits: u16) -> u16 {
    if bits & 0x7c00 == 0x7c00 {
        return 0;
    }
    let v = f16_to_f32(bits);
    let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
    let a = v.abs();
    let r = if a <= 1.0 {
        sign * a.powf(1.0f32 / 2.2f32)
    } else {
        sign * (1.0f32 + a.ln() / 2.2f32)
    };
    if !r.is_finite() {
        return 0;
    }
    let clamped = r.max(-65504.0f32).min(65504.0f32);
    f32_to_f16(clamped)
}

/// 709 正向 CSC（`csc709Forward`）—— **逐字复刻官方矩阵**。
///
/// 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressorSimd.h:223`
/// （`csc709Forward64`）：
///
/// ```text
/// comp0 =  0.2126·src0 + 0.7152·src1 + 0.0722·src2
/// comp1 = -0.1146·src0 - 0.3854·src1 + 0.5000·src2
/// comp2 =  0.5000·src0 - 0.4542·src1 - 0.0458·src2
/// ```
///
/// # ⚠️ 不可用「解方程」代替矩阵（W13 互通对拍抓到的第六个真实 bug）
///
/// 本实现原先**没有**用这张矩阵，而是由 [`csc709_inverse`] 的三个式子
/// **现场除法反解**出`csc709_forward`。二者不是同一个变换：官方正向矩阵
/// 是 709 亮度基（行和 ≈ 1.0），而反解出来的变换行和 ≈ 0.83 —— 意味着
/// 「自研正向 ∘ 自研逆向」虽能互逆（故`csc709_forward_inverts_inverse`
/// 全绿），但**正向那一步就不是官方的了**：我们写的文件会被官方解码成
/// 偏色的像素。
///
/// 现象：3 通道 CSC 常量块自写→自读时 DC 系数被放大到 `0xe598`
/// （half 58776），远超 `COEFF_CLAMP = 1024`，一路overflow 成 `-inf`。
fn csc709_forward(c0: &mut f32, c1: &mut f32, c2: &mut f32) {
    let (s0, s1, s2) = (*c0, *c1, *c2);
    *c0 = 0.2126f32 * s0 + 0.7152f32 * s1 + 0.0722f32 * s2;
    *c1 = -0.1146f32 * s0 - 0.3854f32 * s1 + 0.5000f32 * s2;
    *c2 = 0.5000f32 * s0 - 0.4542f32 * s1 - 0.0458f32 * s2;
}

/// 量化单个 DCT 系数：**逐字复刻官方 `LossyDctEncoderBase::quantize`**。
///
/// # 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressor.cpp:1660`
///
/// ```text
/// half quantize (half src, float errorTolerance) {
///     float srcFloat   = (float)src;
///     int  numSetBits  = countSetBits(src.bits());
///     const unsigned short *closest = closestData + closestDataOffset[src.bits()];
///     for (int targetNumSetBits = numSetBits - 1; targetNumSetBits >= 0; --targetNumSetBits) {
///         tmp.setBits (*closest);
///         if (fabs ((float)tmp - srcFloat) < errorTolerance) return tmp;
///         closest++;
///     }
///     return src;
/// }
/// ```
///
/// 官方用一张 65536 项的 `closestData` 加速表（`dwaLookups.h`，2MB）预先算好
/// 「置位数比 `src` 少 1、再少 2、…、少到 0」时**各自离 `src` 最近**的那个 half，
/// 再按置位数升序排列。本实现**不引入该表**（Mini Code + 零依赖），
/// 改为在**值序**上从 `src` 双向游走、取窗口内置位数最少的候选 —— 语义等价，
/// 已用官方 `closestData` 交叉验证 **768000 / 768000 逐位一致**
/// （覆盖全部 64 项 × 两张表 × 随机 half，含次正规/负值/极值）。
///
/// # 三个必须与官方一致的细节（任一错则退化为“自洽但不互通”）
///
/// 1. **候选必须置位数严格少于 `src`**（官方循环上界是 `numSetBits - 1`）。
///    允许等于会额外压低精度，误差比官方大。
/// 2. **在值序上走，不是位模式序**。`-0.0`(0x8000) 与 `+0.0`(0x0000) 之间隔着
///    整个正数区，位模式序相邻但值序相邻的是 `-0 → +0`。按位模式走会漏掉
///    官方置位数-0 的候选（`0x0000`），导致极小系数被量化成 `-0.0`。
/// 3. **无候选时原样返回 `src`**，不做任何额外舍入。
#[inline]
fn quantize_half(src: u16, error_tolerance: f32) -> u16 {
    let src_f = f16_to_f32(src);
    if !src_f.is_finite() {
        return src;
    }
    let src_pop = count_set_bits(src);
    if src_pop == 0 {
        return src;
    }
    let mut best: Option<(u8, f32, u16)> = None;
    let base = half_value_ord(src);
    for dir in [-1i32, 1i32] {
        let mut o = base as i32 + dir;
        while o >= 0 && (o as u32) <= HALF_ORD_INF {
            let cand = half_pattern_at_ord(o as u32);
            if !is_finite_half(cand) {
                o += dir;
                continue;
            }
            let dist = (f16_to_f32(cand) - src_f).abs();
            // 值序单调 ⇒ 越走越远，可提前退出。
            if dist >= error_tolerance {
                break;
            }
            let pop = count_set_bits(cand);
            if pop < src_pop {
                let better = match best {
                    None => true,
                    Some((bp, bd, bb)) => {
                        pop < bp || (pop == bp && (dist < bd || (dist == bd && cand < bb)))
                    }
                };
                if better {
                    best = Some((pop, dist, cand));
                }
            }
            o += dir;
        }
    }
    best.map_or(src, |(_, _, b)| b)
}

/// 16 位模式里 1 的个数（官方 `countSetBits`，`ImfDwaCompressor.cpp:1622`）。
#[inline]
fn count_set_bits(src: u16) -> u8 {
    (u32::from(src).count_ones() & 0xff) as u8
}

/// half 是否有限（排除 Inf/NaN —— 它们不是合法 DCT 系数）。
#[inline]
fn is_finite_half(bits: u16) -> bool {
    (bits & 0x7c00) != 0x7c00
}

/// half 在**升序值序**中的最大下标（`+inf`）。
const HALF_ORD_INF: u32 = 63489;

/// half 位模式 → 升序值序下标（Inf 亦在表内，用于界定搜索范围）。
///
/// 值序为 `-inf | 负数降序 | -0 | +0 | 正数升序 | +inf`。half 位模式只在
/// **同一符号内**单调，`-0`(0x8000) 与 `+0`(0x0000) 在位模式上相隔 32768，
/// 故必须显式重排，否则游走会跨不过零点（详见 [`quantize_half`] 细节 2）。
#[inline]
fn half_value_ord(bits: u16) -> u32 {
    match bits {
        0xfc00 => 0,                        // -inf
        0x8000 => 0x7c00,                   // -0
        0x0000 => 0x7c01,                   // +0
        0x7c00 => HALF_ORD_INF,             // +inf
        b if b > 0x8000 => 0xfc00 - u32::from(b), // 0x8001..=0xfbff → 1..=31743
        b => 0x7c01 + u32::from(b),         // 0x0001..=0x7bff → 31746..=63488
    }
}

/// [`half_value_ord`] 的严格逆（`half_pattern_at_ord(half_value_ord(b)) == b`）。
#[inline]
fn half_pattern_at_ord(ord: u32) -> u16 {
    match ord {
        0 => 0xfc00,
        1..=0x7bff => (0xfc00 - ord) as u16,
        0x7c00 => 0x8000,
        0x7c01 => 0x0000,
        _ => (ord - 0x7c01) as u16,
    }
}

/// 把系数的 half 位模式钳到 **AC 游程安全**的取值域。
///
/// 游程解码把高字节 `0xff` 视为转义（`0xff00` = 结束、`0xff|k` = 跳过 k 个零），
/// 故需保证 `bits >> 8 != 0xff`（等价于 half 非 Inf/NaN）。
///
/// # 【W14 路1】原 `COEFF_CLAMP` 已移除
///
/// 修正前本模块用「量化后统一 clamp 到 ±1024」来防止产出 `0xff00` 转义位模式。
/// 改用官方 `quantize()`（[`quantize_half`]）后该clamp **在语义上不再存在**：
///
/// - 官方 `quantize()` 只在**容差窗口内**挑选置位数更少的 half，**窗口以
///   `src` 为中心**，故结果的幅值**不会超过 `src`**（最多向 0 收缩）——
///   即它天然只会让系数变小，不会像「步长取整 + clamp」那样把系数顶到边界。
/// - 系数幅值的上界因此由**输入本身**决定，无需人为常量；而 Inf/NaN 在
///   `quantize_half` 入口即被原样返回（`!src_f.is_finite()`），由本函数的
///   位模式判定一并拦掉。
///
/// 保留一层**防御性**位模式判定（而非幅值 clamp）即可覆盖转义风险，
/// 且对任何输入（含极端 half）都成立。
#[inline]
fn safe_coeff_bits(v: f32) -> u16 {
    let bits = f32_to_f16(v);
    if (bits >> 8) == 0xff || (bits & 0x7fff) >= 0x7c00 {
        0
    } else {
        bits
    }
}

/// AC 游程编码（[`un_rle_ac`] 的对逆，严格复刻官方 `LossyDctEncoderBase::rleAc`）。
///
/// 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressor.cpp:1706`。逐条对齐其三个分支：
///
/// | 条件（官方原文） | 写出 | 本实现 |
/// |---|---|---|
/// | `block[dctComp] != 0` | 该系数位模式 | 同 |
/// | `runLen == 1`（单个零） | **原样写 `0x0000`** | 同 |
/// | `runLen + dctComp == 64`（零游程恰好填满） | **`0xff00`（EOB）** | 同 |
/// | 其余 | `0xff00 \| runLen` | 同 |
///
/// # 两个曾经踩过的坑（W13 互通对拍）
///
/// 1. **EOB 不是必写的。** 本实现原先在末尾**无条件**补一枚 `0xff00`，
///    解码端也以「读到 `0xff00`」为终止条件——**自家往返 100% 全绿**，
///    但对官方文件（扁平 AC 流、无 EOB）直接越界报错。见 [`un_rle_ac`]
///    的完整根因分析。
/// 2. **单个零必须原样写 `0x0000`，不能并进游程标记。** 官方对 `runLen == 1`
///    特意走 verbatim 分支：写成 `0xff01` 虽能被自己的解码器读回，但与官方
///    字节流不同，`_numAcComp` 也会多算，跨实现的 AC 项数对不上。
///
/// 因为官方 `runLen` 上界是「`dctComp + runLen < 64`」⇒ `runLen ≤ 62`，
/// 单字节装得下，**不需要**本实现原先那套「超过 255 就拆多个标记」的逻辑
/// （那是基于错误的 EOB 假设引入的，官方游程标记永远 ≤ 62）。
fn rle_ac_encode(zig: &[u16; DCT_COEFFS], out: &mut Vec<u16>) {
    let mut dct_comp = 1usize;
    while dct_comp < DCT_COEFFS {
        let v = *zig.get(dct_comp).unwrap_or(&0);
        if v != 0 {
            // 非零：原样写出。
            out.push(v);
            dct_comp += 1;
            continue;
        }
        // 坐在零上：量出游程长度。
        let mut run_len = 1usize;
        while dct_comp + run_len < DCT_COEFFS && *zig.get(dct_comp + run_len).unwrap_or(&0) == 0 {
            run_len += 1;
        }
        if run_len == 1 {
            // 单个零：官方原样写 0x0000（不并入游程标记）。
            out.push(0);
        } else if run_len + dct_comp == DCT_COEFFS {
            // 零游程恰好填满本comp 的剩余槽 → 写 EOB。
            out.push(0xff00);
        } else {
            // 普通游程标记：`run_len ≤ 62`，单字节装得下。
            out.push(0xff00 | run_len as u16);
        }
        dct_comp += run_len;
    }
    // 官方此处**不**补 EOB：循环条件 `dctComp < 64` 本身就是终止条件。
}

// ---------------------------------------------------------------------------
//  Huf 编码（AC 区的 STATIC_HUFFMAN）
// ---------------------------------------------------------------------------

/// MSB 优先位写入器（[`huf_uncompress`] 的 `HufBits` 的对逆）。
///
/// 语义与 OpenEXR `outputBits` 一致：`c` 左移 `nBits` 后或入 `bits`，满 8 位即
/// 冲出；不足 8 位的尾巴由 [`finish`](HufBitWriter::finish) 左移补 **0**。
#[derive(Default)]
struct HufBitWriter {
    out: Vec<u8>,
    c: u64,
    lc: u32,
}

impl HufBitWriter {
    fn with_capacity(n: usize) -> Self {
        Self {
            out: Vec::with_capacity(n),
            c: 0,
            lc: 0,
        }
    }

    /// 写 `n` 位（`1 ≤ n ≤ 32`），**高位在前**（`bits` 的低 `n` 位有效）。
    fn put(&mut self, n: u32, bits: u64) {
        debug_assert!((1..=32).contains(&n), "n 越界：{n}");
        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        self.c = (self.c << n) | (bits & mask);
        self.lc += n;
        while self.lc >= 8 {
            self.lc -= 8;
            self.out.push(((self.c >> self.lc) & 0xff) as u8);
        }
    }

    /// 已写出的**比特数**。
    fn nbits(&self) -> usize {
        self.out.len() * 8 + self.lc as usize
    }

    /// 收尾：尾巴左移补 0 成整字节。
    fn finish(mut self) -> Vec<u8> {
        if self.lc > 0 {
            self.out.push(((self.c << (8 - self.lc)) & 0xff) as u8);
        }
        self.out
    }
}

/// Huffman 霍夫曼树节点（用于建树；`Vec` 索引即句柄，确定性）。
#[derive(Clone, Copy)]
struct HufNode {
    /// 左子（-1 = 叶）。
    left: i32,
    /// 右子（-1 = 叶）。
    right: i32,
    /// 符号（叶有效）。
    symbol: u32,
    /// 频次。
    freq: u64,
}

/// 由频次表建**码长表**（`hufBuildEncTable` 的自研版：标准霍夫曼 + 定序选择）。
///
/// 返回 `(lengths, im, iM)`：`lengths[i]` 为符号 `i` 的码长（0 = 无码），
/// `im` = 最小有码符号、`iM` = 最大**数据**符号 + 1（其位置放 RLC 伪符号）。
///
/// **确定性**：待合并队列用 `Vec` 线性找最小（同频取**下标小者**），不依赖
/// 任何哈希/堆的隐式顺序。
/// **不变量**：`im`/`iM` 一定包含 RLC 伪符号位（否则解码端会把最大数据符号
/// 当成游程标记）。
fn huf_build_enc_table(frq: &[u64]) -> Option<(Vec<u32>, usize, usize)> {
    // 找最小 / 最大非零频次符号。
    let mut im = usize::MAX;
    let mut i_m = 0usize;
    for (i, &f) in frq.iter().enumerate().take(HUF_ENCSIZE - 1) {
        if f > 0 {
            if im == usize::MAX {
                im = i;
            }
            i_m = i;
        }
    }
    if im == usize::MAX {
        return None; // 全零频次：调用方退回 deflate
    }
    // 追加 RLC 伪符号（频次 1），其下标 = 最大数据符号 + 1。
    let rlc = i_m + 1;
    if rlc >= HUF_ENCSIZE {
        return None; // 符号空间耗尽（理论不可达：AC 值被钳到远小于 65535）
    }
    let mut nodes: Vec<HufNode> = Vec::new();
    let mut live: Vec<i32> = Vec::new();
    for i in im..=rlc {
        let f = if i == rlc { 1 } else { frq[i] };
        if f == 0 {
            continue;
        }
        nodes.push(HufNode {
            left: -1,
            right: -1,
            symbol: i as u32,
            freq: f,
        });
        live.push((nodes.len() - 1) as i32);
    }
    if live.len() == 1 {
        // 单符号：给它 1 位码（解码端需要至少一个可解的码）。
        let mut lengths = vec![0u32; HUF_ENCSIZE];
        lengths[live[0] as usize] = 0; // 占位：下面按 symbol 回填
        let sym = nodes[live[0] as usize].symbol as usize;
        let mut l = vec![0u32; HUF_ENCSIZE];
        l[sym] = 1;
        l[rlc] = 1;
        return Some((l, im, rlc));
    }
    // 反复取两个最小频次合并（同频取下标小者 → 确定）。
    while live.len() > 1 {
        let (a_pos, b_pos) = {
            let mut a = 0usize;
            for k in 1..live.len() {
                if nodes[live[k] as usize].freq < nodes[live[a] as usize].freq {
                    a = k;
                }
            }
            let mut b = if a == 0 { 1 } else { 0 };
            for k in 0..live.len() {
                if k != a && nodes[live[k] as usize].freq < nodes[live[b] as usize].freq {
                    b = k;
                }
            }
            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        };
        let (na, nb) = (live[a_pos], live[b_pos]);
        let f = nodes[na as usize].freq.saturating_add(nodes[nb as usize].freq);
        nodes.push(HufNode {
            left: na,
            right: nb,
            symbol: 0,
            freq: f,
        });
        let new_idx = (nodes.len() - 1) as i32;
        // 删除较大的下标先（避免移位影响）。
        live.remove(b_pos);
        live.remove(a_pos);
        live.push(new_idx);
    }
    // 由树回填码长（DFS，定序：先左后右）。
    let mut lengths = vec![0u32; HUF_ENCSIZE];
    let root = *live.first()?;
    let mut stack: Vec<(i32, u32)> = vec![(root, 0)];
    while let Some((idx, depth)) = stack.pop() {
        let node = nodes.get(idx as usize)?;
        if node.left < 0 && node.right < 0 {
            let sym = node.symbol as usize;
            if sym < HUF_ENCSIZE {
                lengths[sym] = depth.max(1);
            }
        } else {
            stack.push((node.right, depth + 1));
            stack.push((node.left, depth + 1));
        }
    }
    Some((lengths, im, rlc))
}

/// 打包 Huffman 码长表（`hufPackEncTable`，MSB 优先位流）。
///
/// 零码长用游程压缩：`zerun ∈ 2..=5` 走短码（`l = 59 + zerun - 2`），
/// `zerun ≥ 6` 走长码（`l = 63` + 8 位 `zerun - 6`）。
///
/// ⚠️ **长码的 8 位字段只装得下 `zerun - 6 ≤ 255`**（即 `zerun ≤ 261`）。
/// 超过 261 的零游程必须**拆成多段**，否则 `w.put(8, …)` 会被截断，解码端
/// 读回一个偏小的 `zerun` ⇒ 表位置错位 ⇒ 最终越界报错（C18：不 panic 但静默
/// 数据错 ⇒ 更糟）。此处按 261 上限切段。
fn huf_pack_enc_table(lengths: &[u32], w: &mut HufBitWriter, im: usize, i_m: usize) {
    let mut i = im;
    while i <= i_m {
        let l = lengths.get(i).copied().unwrap_or(0);
        if l == 0 {
            // 零码长：统计连续零游程（上限 261 项/段，见上）。
            let mut zerun = 1usize;
            while i + zerun <= i_m && lengths.get(i + zerun).copied().unwrap_or(0) == 0 {
                zerun += 1;
            }
            if zerun >= 2 {
                let mut left = zerun;
                while left > 0 {
                    if left >= SHORTEST_LONG_RUN as usize {
                        let chunk = left.min(SHORTEST_LONG_RUN as usize + 255);
                        w.put(6, LONG_ZEROCODE_RUN as u64);
                        w.put(8, (chunk - SHORTEST_LONG_RUN as usize) as u64);
                        left -= chunk;
                    } else {
                        // 2..=5 项走短码（1 项走下面的普通 0 码长）。
                        if left == 1 {
                            w.put(6, 0);
                        } else {
                            w.put(6, (SHORT_ZEROCODE_RUN as usize + left - 2) as u64);
                        }
                        left = 0;
                    }
                }
                i += zerun;
                continue;
            }
        }
        w.put(6, l as u64);
        i += 1;
    }
}

/// Huf AC 编码（`hufCompress`）：`raw` → 完整 Huf 负载（20 字节信封 + 表 + 数据）。
///
/// 信封 = `im(u32) | iM(u32) | tableLength(u32) | nBits(u32) | reserved(u32)`，
/// 随后是 code-length 表与 MSB 优先的数据位流。
///
/// 不自校验——是否可用由调用方用 [`huf_uncompress`] 验后决定（见 [`ac_encode`]）。
fn huf_compress(raw: &[u16]) -> Option<Vec<u8>> {
    if raw.is_empty() {
        return None;
    }
    let mut frq = vec![0u64; HUF_ENCSIZE];
    for &v in raw {
        if let Some(s) = frq.get_mut(v as usize) {
            *s += 1;
        } else {
            return None;
        }
    }
    let (lengths, im, i_m) = huf_build_enc_table(&frq)?;
    // canonical 码（与解码端同一套：`hufCanonicalCodeTable` 的逆用）。
    let mut codes = lengths.clone();
    huf_canonical_code_table(&mut codes);

    // 表区。
    let mut table_w = HufBitWriter::with_capacity((i_m - im + 8) * 2);
    huf_pack_enc_table(&lengths, &mut table_w, im, i_m);
    let n_table_bits = table_w.nbits();
    let table = table_w.finish();
    if table.len() > u32::MAX as usize {
        return None;
    }

    // 数据区：逐符号查 canonical 码，MSB 优先写出。
    let mut data_w = HufBitWriter::with_capacity(raw.len() * 2);
    for &v in raw {
        let e = codes.get(v as usize).copied().unwrap_or(0);
        let l = e & 63;
        let code = (e >> 6) as u64;
        if l == 0 {
            return None; // 符号无码（理论不可达）
        }
        data_w.put(l, code);
    }
    let n_bits = data_w.nbits();
    let data = data_w.finish();

    // 信封：im / iM / tableLength(字节) / nBits / reserved。
    let mut out: Vec<u8> = Vec::with_capacity(20 + table.len() + data.len());
    out.extend_from_slice(&(im as u32).to_le_bytes());
    out.extend_from_slice(&(i_m as u32).to_le_bytes());
    out.extend_from_slice(&(n_table_bits.div_ceil(8) as u32).to_le_bytes());
    out.extend_from_slice(&(n_bits as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&table);
    out.extend_from_slice(&data);
    Some(out)
}

/// AC 区编码：在 **STATIC_HUFFMAN** 与 **DEFLATE** 之间选更小者，且必须**可被
/// 本模块解码器解回**（自校验）。
///
/// 返回 `(payload, ac_compression)`：0 = Huffman、1 = deflate。
///
/// 自校验理由（诚实工程）：Huffman 路径要求所有码长对解码器的**短码表**
/// （14 位）可覆盖，且尾部位流的解码必须恰好产出 `raw.len()` 项。为杜绝
/// "编码产出自己解不回的块"这类静默缺陷，此处**强制**用 [`huf_uncompress`]
/// 验回；不通过则退回 DEFLATE（其正确性由 zlib 往返保证）。
fn ac_encode(raw: &[u16]) -> Option<(Vec<u8>, u64)> {
    if raw.is_empty() {
        return Some((Vec::new(), 0));
    }
    let huf = huf_compress(raw);
    if let Some(buf) = huf {
        if let Ok(back) = huf_uncompress(&buf, raw.len()) {
            if back == raw {
                let deflate = zlib_pack(raw);
                if deflate.len() < buf.len() {
                    return Some((deflate, 1));
                }
                return Some((buf, 0));
            }
        }
    }
    // Huffman 不可用 → DEFLATE（解码器 `decode_ac` 的 1 号分支）。
    Some((zlib_pack(raw), 1))
}

/// u16 序列 → 裸 zlib（小端），供 **AC(DEFLATE)** 区使用。
fn zlib_pack(vals: &[u16]) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    crate::exr_write::deflate_zlib(&bytes)
}

/// u16 序列 → **官方 `Imf::Zip`** 格式（小端），供 **DC 区**使用。
///
/// 是 [`zip_decompress`] 的严格对逆：先做 delta 预测器**正向**、再做奇偶交织，
/// 最后 deflate。顺序与官方 `ImfZip.cpp::compress` 逐条对齐：
/// ```text
/// compress   : 重排(偶/奇拆分) → 预测器 → deflate
/// uncompress : inflate      → 预测器还原 → 交织还原
/// ```
/// 故本函数是「交织 → 预测器 → deflate」（压缩侧与解压侧互为逆运算，
/// 但官方两个函数的中间步骤次序相反——**不可照抄解压顺序**）。
///
/// ⚠️ 若这里继续用裸 `zlib_pack`，我们写出的 DC 区**官方解不出来**（方向 B 会红）。
fn zip_pack(vals: &[u16]) -> Vec<u8> {
    // ① 交织：偶下标字节进前半，奇下标字节进后半（官方 `compress` 第一步）。
    let n = vals.len() * 2;
    let mut reordered: Vec<u8> = vec![0u8; n];
    let (mut lo, mut hi) = (0usize, n.div_ceil(2));
    let mut out = 0usize;
    while out < n {
        reordered[lo] = byte_at(vals, out);
        lo += 1;
        out += 1;
        if out < n {
            reordered[hi] = byte_at(vals, out);
            hi += 1;
            out += 1;
        }
    }
    // ② 预测器正向：`s[i] = r[i] - r[i-1] + 128`（i ≥ 1），r 的首字节原样。
    //
    // ⚠️【W13 互通对拍抓到的第五个真实 bug】`p` 必须是 **原始** 的前一个字节，
    // 而**不是**已被本循环覆盖过的 `reordered[i-1]`。官方 `ImfZip.cpp:88`：
    // ```c
    // int p = t[-1];
    // while (t < stop) {
    //     int d = int (t[0]) - p + (128 + 256);
    //     p = t[0];   // ← 在写回之前读**原值**
    //     t[0] = d;
    //     ++t;
    // }
    // ```
    // 官方刻意**先存 `p` 再写 `t[0]`**，正是为了让 `p` 始终是原值。
    // 本实现原先原地读写、用 `reordered[i-1]` 当 `p`，那已是**变换后**的值，
    // 于是 `zip_pack` 与 [`zip_decompress`] **不互逆**：单元素（n=1）恰好成立，
    // 一旦 DC 区有 ≥2 个 u16 就整体错位。实测症状：3 通道常量块
    // `zip_pack([0x4404,0x3d00,0x2f80])` 解回 `[0xc404,0xc100,0x7304]`，
    // 再经 `dctInverse8x8DcOnly` + `toLinear` 直接溢出成 `-inf`。
    //
    // 另一个官方细节：**首字节不做预测**（`t` 从 `_tmpBuffer + 1` 起），
    // 对应 SIMD 版注释「The first element doesn't have its high bit flipped」。
    let mut p = i32::from(reordered[0]);
    for i in 1..n {
        let cur = i32::from(reordered[i]);
        reordered[i] = (cur - p + 128).rem_euclid(256) as u8;
        p = cur; // 原值，不是变换后的值
    }
    // ③ deflate（zlib 包装）。
    crate::exr_write::deflate_zlib(&reordered)
}

/// 取 `vals` 的第 `byte_index` 个字节（小端展开后的第 n 字节）。
#[inline]
fn byte_at(vals: &[u16], byte_index: usize) -> u8 {
    let v = vals.get(byte_index / 2).copied().unwrap_or(0);
    // 小端：偶数下标 = 低字节，奇数下标 = 高字节。
    if byte_index % 2 == 0 {
        (v & 0xff) as u8
    } else {
        (v >> 8) as u8
    }
}

/// 逐通道平面：按方案分别编码 UNKNOWN / RLE / LOSSY_DCT 三条支路。
#[allow(clippy::too_many_arguments)]
fn encode_lossy_dct_group(
    width: usize,
    out_lines: usize,
    blocks_x: usize,
    blocks_y: usize,
    comps: &[(usize, crate::exr::PixelType)],
    apply_from_linear: bool,
    planes: &[Vec<u16>],
    dc: &mut Vec<u16>,
    ac: &mut Vec<u16>,
) -> ExrResult<()> {
    const LOC: &str = "exr::dwa::encode_lossy_dct";
    let num_comp = comps.len();
    if num_comp != 1 && num_comp != 3 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("LOSSY_DCT 编码器只支持 1 或 3 通道，实际 {num_comp}"),
            "DWA 通道分类异常",
        ));
    }
    let plane_size = blocks_x.saturating_mul(blocks_y);
    // DC 是 comp-major 平面：先为组内每个 comp 预留一个 plane_size 长度的槽位。
    let dc_base = dc.len();
    dc.resize(dc_base + num_comp * plane_size, 0u16);
    // 组内各 comp 的 DC 写入游标。
    let mut dc_cur = [0usize; 3];
    for (c, slot) in dc_cur.iter_mut().enumerate().take(num_comp) {
        *slot = dc_base + c * plane_size;
    }
    // 每 comp 的临时缓冲：zigzag 序 half 系数（长度 64）。
    let mut zig: [u16; DCT_COEFFS] = [0u16; DCT_COEFFS];
    // 三通道 CSC 时一次取三块；单通道用 `csc_buf[0]`。
    let mut csc_buf: [[f32; DCT_COEFFS]; 3] = [[0.0f32; DCT_COEFFS]; 3];

    for blocky in 0..blocks_y {
        for blockx in 0..blocks_x {
            let block_index = blocky * blocks_x + blockx;
            for comp in 0..num_comp {
                let ci = comps.get(comp).map(|c| c.0).unwrap_or(0);
                let src = planes.get(ci);
                for ly in 0..DCT_N {
                    let y = blocky * DCT_N + ly;
                    for lx in 0..DCT_N {
                        let x = blockx * DCT_N + lx;
                        let v = if y < out_lines && x < width {
                            src.and_then(|p| p.get(y * width + x)).copied().unwrap_or(0)
                        } else {
                            // 越界像素补 0（与 OpenEXR 的块填充一致；解码端会裁掉）。
                            0
                        };
                        let f = f16_to_f32(v);
                        csc_buf[comp][ly * DCT_N + lx] = f;
                    }
                }
            }
            // ① CSC 正向（仅 3 通道组）：对 / 前先做 fromLinear（若需要）。
            if num_comp == 3 {
                for i in 0..DCT_COEFFS {
                    if apply_from_linear {
                        for k in 0..3 {
                            // 逐元素 fromLinear：half → 线性 half → f16_to_f32。
                            let h = f32_to_f16(csc_buf[k][i]);
                            csc_buf[k][i] = f16_to_f32(dwa_from_linear(h));
                        }
                    }
                    let (mut a, mut b, mut c) = (
                        csc_buf[0][i],
                        csc_buf[1][i],
                        csc_buf[2][i],
                    );
                    csc709_forward(&mut a, &mut b, &mut c);
                    csc_buf[0][i] = a;
                    csc_buf[1][i] = b;
                    csc_buf[2][i] = c;
                }
            } else if apply_from_linear {
                // 单通道：逐元素 fromLinear。
                for i in 0..DCT_COEFFS {
                    let h = f32_to_f16(csc_buf[0][i]);
                    csc_buf[0][i] = f16_to_f32(dwa_from_linear(h));
                }
            }

            for comp in 0..num_comp {
                // ② 正向 DCT + 量化。
                let mut block: [f32; DCT_COEFFS] = csc_buf[comp];
                dct_forward_8x8(&mut block);
                // ③ 自然序 → zigzag 序；DC（index 0）量化并写入 DC 平面。
                //
                // 【W14 路1】改用官方 `quantize()`：逐项容差 = `quantBaseError ×
                // 归一化 JPEG 表[raster_i]`（comp 0 用 Y 表，comp 1/2 用 CbCr 表），
                // 取代原先的统一 `QUANT_STEP = 1/64`。官方对 **DC 也量化**
                // （`i = 0` 同样进 `quantize`，`tol = baseError × Y[0] = 7.2e-4`），
                // 故此处 DC 与 AC 走同一条路径，不再另设分支。
                for (i, v) in block.iter_mut().enumerate() {
                    let bits = f32_to_f16(*v);
                    let q = quantize_half(bits, quant_tolerance(comp, i));
                    *v = f16_to_f32(q);
                }
                zig[0] = safe_coeff_bits(block[0]);
                for j in 1..DCT_COEFFS {
                    let raster = RASTER_TO_ZIGZAG[j];
                    zig[j] = safe_coeff_bits(*block.get(raster).unwrap_or(&0.0));
                }
                // ④【W13 互通对拍修正】DC 写入 comp-major 平面。
                //
                // 解码端有**两条**互不相同的逆路径（见 `uncompress` 的
                // `last_non_zero == 0` 分支）：
                //   - AC 全零 → `dct_inverse_dc_only`：`val = dc × 0.3535536²`
                //     （= **×1/8**）后铺满全块；
                //   - AC 非零 → `dct_inverse_8x8`：标准 2D IDCT，纯 DC=1 产出
                //     常量 **1/8**。
                //
                // 本编码端的 `dct_forward_8x8` 是**正交 DCT-II**：常量 c 的
                // 2D DC = `√(1/8)·√(1/8)·64·c` = **8c**（已由
                // `dct_forward_inverts_inverse` 的正逆互逆断言钉死）。
                //
                // 因为两条解码路径的 DC 基幅度**现已都是 1/8**（W13 修正了
                // `dct_inverse_dc_only` 漏乘 1/8 的 bug），编码端**两条路径
                // 都直接写 `zig[0]`（= 8c）**，不再需要任何 ÷8 补偿。
                //
                // 【历史】修正前 `dct_inverse_dc_only` 漏乘 1/8（直接铺满 dc），
                // 与正交 IDCT 的 1/8 相差恰好 8 倍，于是这里曾用
                // `block[0] / 8.0` 去对齐那条错误路径——两侧错误抵消、内部
                // 往返 100% 全绿，跨实现才露馅（官方常量块 0.5 被读成 42080.0）。
                // 现两侧都按官方的 1/8 口径，补偿必须**移除**，否则又变成
                // 新的"自洽但不互通"。
                let _ = &mut dc_cur[comp];
                if let Some(dc_slot) = dc.get_mut(dc_cur[comp] + block_index) {
                    *dc_slot = zig[0];
                }
                // ⑤ AC 游程写入组内交错的单一流。
                rle_ac_encode(&zig, ac);
            }
        }
    }
    Ok(())
}

/// RLE 通道（`DwaScheme::Rle`）的**字节平面**重排（`interleaveByte2` 的 4 字节推广）。
///
/// 输出 = `[字节0 平面][字节1 平面]…`，每平面 `width × out_lines` 字节，
/// 像素按行主序 —— 与解码端 [`interleave_half_rle`]（2 字节）及 `assemble`
/// 的 4 字节分支严格一致。
fn pack_rle_planes(plane: &[u8], pw: usize, width: usize, out_lines: usize, out: &mut Vec<u8>) {
    let stride = width.saturating_mul(out_lines);
    for b in 0..pw {
        for pix in 0..stride {
            let di = pix * pw + b;
            out.push(plane.get(di).copied().unwrap_or(0));
        }
    }
}

/// 对一个块做 DWAA/DWAB 编码（两者**算法一致**，仅块行数不同，由调用方切块）。
///
/// `raw` = 块内「扫描线 × 通道 × 像素」小端字节；`ctx` 给出通道表与几何。
/// 返回**可直接交给 [`decompress_block`]** 的 DWA 块负载。
///
/// 结构（与解码器的消费顺序逐字段对齐）：
/// ```text
/// [11 × i64 大小表]   version = 2
/// [u16 规则表长度 + 规则表]  （default_rules，确定性序列化）
/// [unknown 区]  zlib（通道序）
/// [AC 区]       STATIC_HUFFMAN 或 DEFLATE
/// [DC 区]       zlib（comp-major，组序）
/// [RLE 区]      zlib(EXR-RLE 无头)
/// ```
pub fn compress_block(raw: &[u8], ctx: &DwaContext<'_>) -> ExrResult<Vec<u8>> {
    const LOC: &str = "exr::dwa::compress_block";
    let width = ctx.width;
    let out_lines = ctx.height;
    if width == 0 || out_lines == 0 {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 编码块尺寸非法（{width}×{out_lines}）"),
            "块宽高必须非零",
        ));
    }
    if ctx.channels.is_empty() {
        return Err(crate::exr::comp_err(
            LOC,
            "DWA 编码无通道",
            "至少需要一个通道",
        ));
    }

    let rules = default_rules();
    let plan = classify_channels(ctx.channels, &rules);
    let nch = plan.channels.len();
    let widths: Vec<usize> = plan
        .channels
        .iter()
        .map(|c| c.pixel_type.byte_width())
        .collect();
    let per_row: usize = widths.iter().sum();
    let expect = per_row
        .saturating_mul(width)
        .saturating_mul(out_lines);
    if raw.len() != expect {
        return Err(crate::exr::comp_err(
            LOC,
            format!("DWA 编码输入 {} 字节，期望 {expect} 字节", raw.len()),
            "块字节数须等于「行字节 × 块内扫描线数」",
        ));
    }

    // ---- 拆成逐通道平面（原始字节，行主序）----
    let line_bytes: Vec<usize> = widths.iter().map(|w| w * width).collect();
    let mut planes_bytes: Vec<Vec<u8>> = vec![Vec::new(); nch];
    for (ci, lb) in line_bytes.iter().enumerate() {
        let mut p: Vec<u8> = Vec::with_capacity(lb * out_lines);
        for y in 0..out_lines {
            // 行内偏移 = Σ_{j<ci} line_bytes[j]
            let row_start: usize = per_row * width * y;
            let off: usize = line_bytes.iter().take(ci).sum();
            let start = row_start + off;
            let end = start + lb;
            match raw.get(start..end) {
                Some(s) => p.extend_from_slice(s),
                None => p.resize(p.len() + lb, 0),
            }
        }
        planes_bytes[ci] = p;
    }

    // ---- 逐通道 half 位模式平面（LOSSY_DCT 用）----
    let to_half_plane = |ci: usize| -> Vec<u16> {
        let pt = plan.channels[ci].pixel_type;
        let src = &planes_bytes[ci];
        let n = width.saturating_mul(out_lines);
        let mut out: Vec<u16> = Vec::with_capacity(n);
        for i in 0..n {
            let b = pt.byte_width();
            let s = i * b;
            let v = match pt {
                crate::exr::PixelType::Half => {
                    u16::from_le_bytes([
                        src.get(s).copied().unwrap_or(0),
                        src.get(s + 1).copied().unwrap_or(0),
                    ])
                }
                crate::exr::PixelType::Float => f32_to_f16(f32::from_le_bytes([
                    src.get(s).copied().unwrap_or(0),
                    src.get(s + 1).copied().unwrap_or(0),
                    src.get(s + 2).copied().unwrap_or(0),
                    src.get(s + 3).copied().unwrap_or(0),
                ])),
                crate::exr::PixelType::Uint => 0,
            };
            out.push(v);
        }
        out
    };

    let blocks_x = width.div_ceil(DCT_N);
    let blocks_y = out_lines.div_ceil(DCT_N);

    let mut unknown_raw: Vec<u8> = Vec::new();
    let mut rle_raw: Vec<u8> = Vec::new();
    let mut dc: Vec<u16> = Vec::new();
    let mut ac: Vec<u16> = Vec::new();
    let mut decoded = vec![false; nch];

    // ---- CSC 组优先（与解码端一致）----
    for set in &plan.csc_sets {
        for k in 0..3 {
            let ci = set.idx[k];
            let ok = plan
                .channels
                .get(ci)
                .map(|c| c.scheme == DwaScheme::LossyDct)
                .unwrap_or(false);
            if !ok {
                return Err(crate::exr::comp_err(
                    LOC,
                    "CSC 组内通道不是 LOSSY_DCT（编码侧）",
                    "DWA 通道分类异常",
                ));
            }
        }
        let comps: [(usize, crate::exr::PixelType); 3] = [
            (set.idx[0], plan.channels[set.idx[0]].pixel_type),
            (set.idx[1], plan.channels[set.idx[1]].pixel_type),
            (set.idx[2], plan.channels[set.idx[2]].pixel_type),
        ];
        let halfs: Vec<Vec<u16>> = (0..3).map(|k| to_half_plane(set.idx[k])).collect();
        encode_lossy_dct_group(
            width,
            out_lines,
            blocks_x,
            blocks_y,
            &comps,
            true, // CSC 组恒做 fromLinear（与解码端无条件 toLinear 对逆）
            &halfs,
            &mut dc,
            &mut ac,
        )?;
        for k in 0..3 {
            if let Some(d) = decoded.get_mut(set.idx[k]) {
                *d = true;
            }
        }
    }

    // ---- 其余通道（按索引序）----
    for ci in 0..nch {
        if decoded[ci] {
            continue;
        }
        let c = plan.channels[ci];
        let pw = widths[ci];
        match c.scheme {
            DwaScheme::LossyDct => {
                let comps = [(ci, c.pixel_type)];
                let halfs = vec![to_half_plane(ci)];
                encode_lossy_dct_group(
                    width,
                    out_lines,
                    blocks_x,
                    blocks_y,
                    &comps,
                    !c.p_linear,
                    &halfs,
                    &mut dc,
                    &mut ac,
                )?;
            }
            DwaScheme::Rle => {
                pack_rle_planes(&planes_bytes[ci], pw, width, out_lines, &mut rle_raw);
            }
            DwaScheme::Unknown => {
                unknown_raw.extend_from_slice(&planes_bytes[ci]);
            }
        }
        if let Some(d) = decoded.get_mut(ci) {
            *d = true;
        }
    }

    // ---- 四区压缩 ----
    let unknown = if unknown_raw.is_empty() {
        Vec::new()
    } else {
        crate::exr_write::deflate_zlib(&unknown_raw)
    };
    let (ac_buf, ac_mode) = ac_encode(&ac).ok_or_else(|| {
        crate::exr::comp_err(LOC, "DWA AC 区编码失败", "内部状态错误")
    })?;
    let dc_buf = if dc.is_empty() {
        Vec::new()
    } else {
        // ⚠️ DC 区必须是官方 `Imf::Zip` 格式（含预测器 + 交织），
        // 用裸 zlib 的话官方解码器读我们的文件会得到错值。
        zip_pack(&dc)
    };
    // RLE 区：EXR-RLE（**无**长度头）+ zlib；两个长度都要记。
    let (rle_buf, rle_packed_len) = if rle_raw.is_empty() {
        (Vec::new(), 0u64)
    } else {
        // `rle_compress` 会带 4 字节长度头 —— DWA 区不要它，故剥掉。
        let full = crate::exr::rle_compress(&rle_raw);
        let body = full.get(4..).unwrap_or(&[]).to_vec();
        let packed_len = body.len() as u64;
        (crate::exr_write::deflate_zlib(&body), packed_len)
    };

    // ---- 规则表序列化（version 2，确定性）----
    //
    // ⚠️ 只写**与本块通道相关**的规则，且每条规则至多写一次（首次命中即break）。
    // 权威依据 = OpenEXR 3.1.5 `ImfDwaCompressor.cpp:3223`
    // （`DwaCompressor::relevantChannelRules`）：
    // ```cpp
    // for (i : _channelRules)          // 遍历整张默认表
    //     for (cd : _channelData)      // 遍历本块通道
    //         if (_channelRules[i].match(suffixes[cd], _channelData[cd].type)) {
    //             rules.push_back(_channelRules[i]);
    //             break;               // ← 命中即停，一条规则只写一次
    //         }
    // ```
    // 即：**默认表里那些没被任何通道命中的规则不写进文件**。
    // 实测差异（3 通道 float RGB，DWAA）：
    // ```text
    // 官方 RULES_LEN = 14 字节（Y/float 一条 + R/G/B 各一条…按命中情况）
    // 本实现原先     = 66 字节（把整张 15 条默认表全写了）
    // ```
    // 写全表虽能被自己的解码器读回，但会让官方解码器按文件里的规则重新分类
    // 通道——多出来的规则仍会命中同名后缀，故官方**读我们的文件时分类结果可能
    // 与编码端不一致**，表现为「Early end of file」这类尺寸对不上的错误。
    let suffixes: Vec<String> = ctx
        .channels
        .iter()
        .map(|(name, _, _): &(String, crate::exr::PixelType, bool)| {
            //官方取最后一个 '.' 之后的段作为后缀。
            name.rsplit_once('.')
                .map_or_else(|| name.clone(), |(_, s)| s.to_string())
        })
        .collect();
    let relevant: Vec<&DwaRule> = rules
        .iter()
        .filter(|r| {
            ctx.channels
                .iter()
                .enumerate()
                .any(|(i, (_, pt, _))| r.matches(&suffixes[i], *pt))
        })
        .collect();
    let mut rule_body: Vec<u8> = Vec::new();
    for r in relevant {
        rule_body.extend_from_slice(&encode_rule(r));
    }
    let rule_size = 2 + rule_body.len();
    if rule_size > u16::MAX as usize {
        return Err(crate::exr::comp_err(
            LOC,
            "DWA 规则表过大",
            "内部状态错误",
        ));
    }

    // ---- 11 × i64 头部 ----
    let sizes: [u64; NUM_SIZES_SINGLE] = [
        2,                                 // version
        unknown_raw.len() as u64,          // unknownUncompressedSize
        unknown.len() as u64,              // unknownCompressedSize
        ac_buf.len() as u64,               // acCompressedSize
        dc_buf.len() as u64,               // dcCompressedSize
        rle_buf.len() as u64,              // rleCompressedSize
        rle_packed_len,                    // rleUncompressedSize（zlib 前）
        rle_raw.len() as u64,              // rleRawSize
        ac.len() as u64,                   // acUncompressedCount
        dc.len() as u64,                   // dcUncompressedCount
        ac_mode,                           // acCompression（0=Huf,1=deflate）
    ];

    let mut out: Vec<u8> = Vec::with_capacity(
        NUM_SIZES_SINGLE * 8 + rule_size + unknown.len() + ac_buf.len() + dc_buf.len() + rle_buf.len(),
    );
    for s in sizes {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&(rule_size as u16).to_le_bytes());
    out.extend_from_slice(&rule_body);
    out.extend_from_slice(&unknown);
    out.extend_from_slice(&ac_buf);
    out.extend_from_slice(&dc_buf);
    out.extend_from_slice(&rle_buf);
    Ok(out)
}

/// 计算一个块的 DWA 编码（供 [`crate::exr_write`] 的压缩分派调用）。
///
/// `lines` 为块内实际扫描线数（DWAA ≤ 32、DWAB ≤ 256，末块可不满）。
///
/// # 【W14 路1】逐块「压不小就原样存」回退（官方语义，本实现此前缺失）
///
/// 权威依据 = OpenEXR 3.1.5 `ImfOutputFile.cpp:576-604`：
/// ```cpp
/// int compSize = compressor->compress (_lineBuffer->dataPtr,
///                                      _lineBuffer->dataSize, minY, compPtr);
/// if (compSize < _lineBuffer->dataSize)   // ← 严格小于才采用压缩结果
/// {
///     _lineBuffer->dataSize = compSize;
///     _lineBuffer->dataPtr  = compPtr;
/// }
/// else if (_ofd->format == Compressor::NATIVE)
/// {
///     convertToXdr (...);   // 原样存，且按 XDR 字节序
/// }
/// ```
/// 即**块级**（不是文件级）判断，且压缩后**等于或大于**原样都要放弃压缩。
/// 读端据此反推（`ImfScanLineInputFile.cpp:522-542`）：
/// ```cpp
/// if (_lineBuffer->compressor && (size_t)_lineBuffer->dataSize < uncompressedSize)
///     … decompress …          // dataSize < 未压缩字节数 ⇒ 走解压
/// else
///     _lineBuffer->format = Compressor::XDR;   // 否则原样像素，且是 XDR 序
/// ```
/// 本仓读端已实装该反推（见 `exr.rs` 的 `payload_len == expected` 分支），
/// 但**写端此前只有 DWA 编码一条路**，从不回退 —— 这正是
/// `KI-DWA-QUANT-1CH-NOISE` 的真正根因（详见下）。
///
/// ## 为什么这才是 `KI-DWA-QUANT-1CH-NOISE` 的根因（登记簿原记载有误）
///
/// 登记簿写的是「我们的 DWA 量化步长比官方粗，官方对非 CSC 通道使用更细的
/// 量化表」。**实测证据不支持这个说法**：对 `dwaa-half-scan-1-noise`
/// （64×48、half、1 通道），官方写出的文件**根本没有 DWA 块**——
///
/// ```text
/// 官方 a_*.exr：chunk0 dataSize=4096 = rawChunk(64×2×32)  ← 恰好等于
///               chunk1 dataSize=2048 = rawChunk(64×2×16)  ← 恰好等于
///               块首 88 字节按 i64 解释是垃圾（version=3726514378048814387）
///               负载按 LE half 解释 = 源数据半精度值（1.299805, 0.748047, …）
/// ```
///
/// 官方走的就是上面那个 `else` 分支：**DWA 压不动这块高频噪声，于是原样存
/// 像素**。所以官方那 `1.939e-3` 的「劣化」根本不是量化误差，而是
/// **float→half 的表示误差**（源是 FLOAT 通道，文件是 half 通道）。
///
/// 我们则always 产出 DWA 块（`dataSize=2368 < 4096`），于是把 DWA 的
/// **有损量化误差** `1.085e-1` 暴露出来，与官方的 half 表示误差不可比。
/// 修量化表只能把 `1.085e-1` 往下压，压不到 `1.939e-3` 的量级——因为
/// 官方那一格**压根没做 DWA**。
///
/// 补上这个回退后，该格与官方走同一条路（存 raw），误差回到 half 表示误差
/// 量级，劣化比远低于阈值 10；且**这不是降低质量**，而是与官方行为对齐。
pub fn dwa_compress(
    raw: &[u8],
    channels: &[(String, crate::exr::PixelType, bool)],
    width: usize,
    lines: usize,
) -> ExrResult<Vec<u8>> {
    let ctx = DwaContext {
        channels,
        width,
        height: lines,
    };
    let enc = compress_block(raw, &ctx)?;
    // 官方判据是 `compSize < dataSize`（严格小于）。相等也回退。
    //
    // ⚠️ `raw` 已是「扫描线 × 通道 × 像素」的小端字节，正是官方
    // `_lineBuffer->dataSize` 的同一量，故直接比长度即可。
    // 回退时原样返回 `raw`：本仓读写两侧统一用小端（= OpenEXR 的
    // `GLOBAL_SYSTEM_LITTLE_ENDIAN` 为真时 `DwaCompressor::format()` 返回
    // `NATIVE` 的等价口径），无需再做 `convertToXdr`。
    if enc.len() < raw.len() {
        Ok(enc)
    } else {
        Ok(raw.to_vec())
    }
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exr::PixelType;
    use crate::fbuf::f16_to_f32;
    use crate::fbuf::f32_to_f16;

    /// 单通道 half 平面 → 块内原始字节（帮助编码测试构造输入）。
    fn raw_half_plane(vals: &[f32], width: usize, lines: usize) -> Vec<u8> {
        let mut raw: Vec<u8> = Vec::with_capacity(width * lines * 2);
        for i in 0..(width * lines) {
            raw.extend_from_slice(&f32_to_f16(vals.get(i).copied().unwrap_or(0.0)).to_le_bytes());
        }
        raw
    }

    // -----------------------------------------------------------------------
    // 11) 编码器：DCT 正逆复合 / CSC 正逆复合
    // -----------------------------------------------------------------------

    /// 从 [`dct_inverse_8x8`] 的**实现**反推 1D 矩阵 `M`，与 [`M_TABLE`] 逐格比对。
    ///
    /// # 这条测试的来历（W12 路E 收口）
    ///
    /// `M_TABLE` / `m_value` 是路B/DWA 收尾时落盘的「1D 是正交 DCT-III」契约表，
    /// 但**从未被任何测试调用**（`dead_code` 警告），且表里的值**抄错了**。
    /// 二者都不是「删掉就完事」：
    ///
    /// - 它**本该服务于**这条契约——把「`M` 的每个元素」钉成可读的符号式，
    ///   比只断言 `MF = I`（`dct_forward_inverts_inverse`）更容易定位错误；
    /// - 但**表本身错了**，直接接上会红。错在第 0 行只抄了偶数下标，
    ///   以及两处 `-E` 被写成 `-D`。故先按实现改正，再接上。
    ///
    /// # 测量方法（不重写IDCT，只测它）
    ///
    /// 令输入的第 0 行为单位向量 `e_i`、其余行为 0，则
    /// 「先行」阶段把第 0 行变成 `M` 的第 `i` **列**，其余行仍为 0；
    /// 「后列」阶段作用在第 `j` 列的向量 `(M[j][i], 0, …, 0)` 上。第 0 行
    /// 非零的列只有第 `j` 个元素非零，故该列经 1D 变换后每项都等于
    /// `A · M[j][i]`（`A = 0.5·cos(π/4)`，即 `M[j][0]`，恒等）。于是
    /// **`M[j][i] = out[j] / A`** —— 一行公式即可反推整张表。
    #[test]
    fn m_table_matches_implementation() {
        const A: f32 = 0.353_553_4; // M[j][0]，恒为 A（每行首项）
        for i in 0..DCT_N {
            // 第 0 行 = e_i，其余行 = 0。
            let mut blk = [0.0f32; DCT_COEFFS];
            blk[i] = 1.0;
            dct_inverse_8x8(&mut blk);
            for j in 0..DCT_N {
                let measured = blk[j] / A; // = M[j][i]
                let declared = m_value(M_TABLE[j][i]);
                assert!(
                    (measured - declared).abs() <= 1e-5,
                    "M[{j}][{i}] 表值与实现不符：表={declared}、实测={measured}\
                     （M_TABLE 第 {j} 行第 {i} 列编码 = {}）",
                    M_TABLE[j][i]
                );
            }
        }
    }

    #[test]
    fn dct_forward_inverts_inverse() {
        // 任意块：`IDCT(forward(x)) == x`（f32 舍入内）。这是 1D 逆矩阵
        // `diag(1/c)·Mᵀ` 必须成立的硬证据（列正交性）。
        let mut x = [0.0f32; DCT_COEFFS];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.037 - 1.1;
        }
        let mut y = x;
        dct_forward_8x8(&mut y);
        dct_inverse_8x8(&mut y);
        for i in 0..DCT_COEFFS {
            let err = (y[i] - x[i]).abs();
            assert!(
                err <= 1e-3 * x[i].abs().max(1.0),
                "第 {i} 项正逆复合误差过大：{} vs {}（err={err}）",
                y[i],
                x[i]
            );
        }
        // 纯 DC 的 forward：常量块应几乎全部落到 DC（index 0），AC 近零。
        //
        // 【D-32 修正】期望值原写 0.25，是**漏了 2D 正交 DCT-II 的 ×8 因子**。
        // 正交 DCT-II：`c_k = √(1/8)`(k=0) / `√(2/8)`(k≥1)，故常量块 c 的 2D DC
        // = `c₀·c₀·ΣΣ` = `√(1/8)·√(1/8)·64·c` = **8c**。此处 c = 0.25 ⇒ DC = 2.0。
        // 该实现给出 2.0000002 是**正确**的（原断言 0.25 才是错的）。
        // 与解码侧互逆：`dct_inverse_8x8` 对纯 DC=1 产出常量 1/8（见
        // `dct_inverse_8x8_of_pure_dc_is_a_constant_one_eighth`），二者恰好相消。
        let mut c = [0.25f32; DCT_COEFFS];
        dct_forward_8x8(&mut c);
        assert!(
            (c[0] - 2.0).abs() < 1e-4,
            "常量块 forward 的 DC 应 ≈ 8×0.25 = 2.0，实得 {}",
            c[0]
        );

        for i in 1..DCT_COEFFS {
            assert!(
                c[i].abs() < 1e-4,
                "常量块 forward 的第 {i} 个 AC 应 ≈ 0，实得 {}",
                c[i]
            );
        }
    }

    #[test]
    fn csc709_forward_inverts_inverse() {
        // W13 修正：正向 CSC 已改为**逐字复刻官方 `csc709Forward64` 的矩阵**
        // （见 `csc709_forward` 的根因说明）。官方这两张矩阵**并非精确互逆**——
        // 它们的系数各自独立取到 4 位小数，`F · I ≈ I` 只到 ~1e-4
        // （实测最大偏差 1.7e-5 / 8.5e-5 / 1.6e-5，见下断言）。
        // 这份残差是**官方自带的有损**，不是我们的 bug。
        //
        // 【实测数据（W13 实跑，`numpy` 按官方两张矩阵精确复算）】
        // ```text
        // max|F·I − I|          = 8.518e-05
        // 逐样本最大偏离 (1,-1,0.5) = 1.0352e-04   ← 三个样本里最差的一个
        // ```
        // 阈值取 **3e-4 ≈ 实测 2.9 倍**。按门禁纪律反向自证：这远低于
        // "放水线"（实测 ×10 = 1.04e-3），而要判红只需把偏移做到 3e-4 以上——
        // 即**任何真实回归**（哪怕只错 1e-4）都在阈值外。
        const CSC_FWD_INV_TOL: f32 = 3e-4;
        for (a, b, c) in [
            (0.5f32, 0.25f32, 0.125f32),
            (-0.3, 0.9, 0.0),
            (1.0, -1.0, 0.5),
        ] {
            let (mut x, mut y, mut z) = (a, b, c);
            csc709_forward(&mut x, &mut y, &mut z);
            csc709_inverse(&mut x, &mut y, &mut z);
            assert!(
                (x - a).abs() < CSC_FWD_INV_TOL,
                "R: {x} vs {a}（偏离需 < {CSC_FWD_INV_TOL}，实测最大 1.035e-4）"
            );
            assert!(
                (y - b).abs() < CSC_FWD_INV_TOL,
                "G: {y} vs {b}（偏离需 < {CSC_FWD_INV_TOL}，实测最大 1.035e-4）"
            );
            assert!(
                (z - c).abs() < CSC_FWD_INV_TOL,
                "B: {z} vs {c}（偏离需 < {CSC_FWD_INV_TOL}，实测最大 1.035e-4）"
            );
        }
    }

    /// 反向自证：正向 CSC 必须**逐位**等于官方矩阵，不得是"解方程"版本。
    ///
    /// 钉死W13 修掉的第六个 bug——若有人再把 `csc709_forward` 改回由
    /// `csc709_inverse` 反解的实现，本条立刻红（反解版的行和 ≈0.83，
    /// 与官方的 1.0 / 0.0 / 0.0 差得��远）。
    #[test]
    fn csc709_forward_matches_official_matrix_bitwise() {
        // 官方 `ImfDwaCompressorSimd.h:233-235` 的三个式子，逐项钉死。
        for (r, g, b) in [(0.5f32, 0.25f32, 0.125f32), (1.0, 0.0, -1.0)] {
            let (mut c0, mut c1, mut c2) = (r, g, b);
            csc709_forward(&mut c0, &mut c1, &mut c2);
            let want0 = 0.2126f32 * r + 0.7152f32 * g + 0.0722f32 * b;
            let want1 = -0.1146f32 * r - 0.3854f32 * g + 0.5000f32 * b;
            let want2 = 0.5000f32 * r - 0.4542f32 * g - 0.0458f32 * b;
            assert_eq!(c0.to_bits(), want0.to_bits(), "comp0 必须逐位等于官方 0.2126/0.7152/0.0722");
            assert_eq!(c1.to_bits(), want1.to_bits(), "comp1 必须逐位等于官方 -0.1146/-0.3854/0.5000");
            assert_eq!(c2.to_bits(), want2.to_bits(), "comp2 必须逐位等于官方 0.5000/-0.4542/-0.0458");
        }
        // 行和是区分「官方矩阵」与「解方程反解」最快的判据：
        // 官方行和 = (1.0, 0.0, 0.0)（亮度基）；反解版 ≈ (0.83, …, …)。
        let (mut c0, mut c1, mut c2) = (1.0f32, 1.0f32, 1.0f32);
        csc709_forward(&mut c0, &mut c1, &mut c2);
        assert!((c0 - 1.0).abs() < 1e-6, "常量灰的前向 CSC 亮度分量须保持，行和={c0}");
        assert!(c1.abs() < 1e-6, "常量灰的色度分量须为 0，实得 {c1}");
        assert!(c2.abs() < 1e-6, "常量灰的色度分量须为 0，实得 {c2}");
    }

    #[test]
    fn dwa_from_linear_inverts_to_linear() {
        // toLinear 的公式逆：复合误差在 half 精度内。
        for v in [0.0f32, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, -0.5, -2.0] {
            let h = f32_to_f16(v);
            let back = dwa_to_linear(dwa_from_linear(h));
            let got = f16_to_f32(back);
            assert!(
                (got - v).abs() <= 2e-3 * v.abs().max(1.0),
                "fromLinear∘toLinear 不闭合：v={v} got={got}"
            );
        }
    }

    #[test]
    fn compress_block_constant_half_plane_is_dc_only() {
        // 常量块 → forward DCT 全部落 DC ⇒ AC 区只有结束符一项。
        let (width, lines) = (8usize, 8usize);
        let raw = raw_half_plane(&[0.25f32; 64], width, lines);
        let chans = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let s = parse_sizes(&enc).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.dc_uncompressed_count, 1, "1 个 DCT 块 → 1 个 DC");
        assert_eq!(s.ac_uncompressed_count, 1, "常量块 AC 仅 1 个结束符");
        // 往返：常量 0.25 应基本还原（Y 的 p_linear=true → 无 LUT 往返）。
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&out, &chans, width, lines);
        for (i, &v) in planes[0].iter().enumerate() {
            assert!(
                (v - 0.25).abs() < 1e-3,
                "像素 {i} 应 ≈0.25，实得 {v}"
            );
        }
    }

    #[test]
    fn compress_block_round_trips_single_channel_varies() {
        // 平滑渐变（块内变化幅度小 → 量化误差可控）。
        let (width, lines) = (8usize, 8usize);
        let vals: Vec<f32> = (0..64).map(|i| ((i % 8) as f32) / 16.0).collect();
        let raw = raw_half_plane(&vals, width, lines);
        let chans = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&out, &chans, width, lines);
        for (i, &v) in planes[0].iter().enumerate() {
            assert!(
                (v - vals[i]).abs() <= 0.05,
                "像素 {i}: {:?} vs {:?}",
                v,
                vals[i]
            );
        }
    }

    #[test]
    fn compress_block_round_trips_three_channel_csc() {
        // R/G/B（CSC 组）+ 非 pLinear（触发无条件 fromLinear/toLinear）。
        let (width, lines) = (8usize, 8usize);
        let mk = |k: usize| -> Vec<f32> {
            (0..64)
                .map(|i| (((i + k * 5) % 16) as f32) / 16.0)
                .collect()
        };
        let r = mk(0);
        let g = mk(1);
        let b = mk(2);
        // 块内布局：扫描线 × 通道 × 像素。
        let mut raw: Vec<u8> = Vec::new();
        for y in 0..lines {
            for (k, src) in [&r, &g, &b].iter().enumerate() {
                let _ = k;
                for x in 0..width {
                    let i = y * width + x;
                    raw.extend_from_slice(&f32_to_f16(src[i]).to_le_bytes());
                }
            }
        }
        let chans = chans(&[
            ("R", PixelType::Half, false),
            ("G", PixelType::Half, false),
            ("B", PixelType::Half, false),
        ]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&out, &chans, width, lines);
        for (ci, src) in [&r, &g, &b].iter().enumerate() {
            for (i, &v) in planes[ci].iter().enumerate() {
                assert!(
                    (v - src[i]).abs() <= 0.06,
                    "通道 {ci} 像素 {i}: {v} vs {}",
                    src[i]
                );
            }
        }
    }

    #[test]
    fn compress_block_unknown_and_rle_channels_round_trip_verbatim() {
        // A（half）→ RLE；Z（float）→ Unknown。两者都应**无损**还原。
        let (width, lines) = (4usize, 4usize);
        let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.03).collect();
        let z: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25).collect();
        let mut raw: Vec<u8> = Vec::new();
        for y in 0..lines {
            for x in 0..width {
                let i = y * width + x;
                raw.extend_from_slice(&f32_to_f16(a[i]).to_le_bytes());
            }
            for x in 0..width {
                let i = y * width + x;
                raw.extend_from_slice(&z[i].to_le_bytes());
            }
        }
        let chans = chans(&[("A", PixelType::Half, false), ("Z", PixelType::Float, false)]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.len(), raw.len());
        let planes = split_planes(&out, &chans, width, lines);
        // A：half 精确还原。
        for (i, &v) in planes[0].iter().enumerate() {
            assert_eq!(
                hbits(f32_to_f16(v)),
                hbits(f32_to_f16(a[i])),
                "A 像素 {i} 须无损"
            );
        }
        // Z：Unknown 区逐位还原（float 也按 half 中转，故用 half 值域）。
        for (i, &v) in planes[1].iter().enumerate() {
            assert_eq!(
                hbits(f32_to_f16(v)),
                hbits(f32_to_f16(z[i])),
                "Z 像素 {i} 须无损"
            );
        }
    }

    #[test]
    fn compress_block_is_deterministic_bitwise() {
        let (width, lines) = (9usize, 5usize);
        let vals: Vec<f32> = (0..45).map(|i| ((i % 7) as f32) / 7.0).collect();
        let raw = raw_half_plane(&vals, width, lines);
        let chans = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&chans, width, lines);
        let a = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let b = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "同输入两次编码须逐字节相同");
    }

    #[test]
    fn huf_compress_round_trips_through_huf_uncompress() {
        // 构造一个 AC 样式序列：大量零游程 + 少量非零 half 位模式。
        let mut raw: Vec<u16> = Vec::new();
        for i in 0..200usize {
            if i % 3 == 0 {
                raw.push(f32_to_f16((i as f32) / 256.0));
            } else {
                raw.push(0xff01 + (i as u16 % 5));
            }
        }
        let buf = huf_compress(&raw).unwrap_or_else(|| panic!("huf_compress 返回 None"));
        let im = u32::from_le_bytes([buf[0],buf[1],buf[2],buf[3]]);
        let imx = u32::from_le_bytes([buf[4],buf[5],buf[6],buf[7]]);
        let tl = u32::from_le_bytes([buf[8],buf[9],buf[10],buf[11]]);
        let nb = u32::from_le_bytes([buf[12],buf[13],buf[14],buf[15]]);
        eprintln!("DBG im={im} iM={imx} tableLen={tl} nBits={nb} total={}", buf.len());
        let back = huf_uncompress(&buf, raw.len()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, raw, "Huf 编码 → 本模块解码须逐项一致");
    }

    #[test]
    fn ac_encode_prefers_usable_path_and_is_self_verified() {
        let raw: Vec<u16> = vec![0xff20, 0x3c00, 0xff00];
        let (buf, mode) = ac_encode(&raw).unwrap_or_else(|| panic!("ac_encode 返回 None"));
        assert!(mode == 0 || mode == 1);
        let back = decode_ac(mode, &buf, raw.len()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, raw, "AC 编码须能被 decode_ac 解回（mode={mode}）");
    }

    #[test]
    fn compress_block_rejects_wrong_raw_length_and_empty_channels() {
        let chans = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&chans, 8, 8);
        let err = compress_block(&[0u8; 10], &ctx).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("期望"), "须回显期望字节数：{err}");

        let empty: Vec<(String, PixelType, bool)> = Vec::new();
        let ctx2 = ctx_of(&empty, 8, 8);
        let err = compress_block(&[0u8; 128], &ctx2).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    /// `u16` 没有 `to_bits()`；统一走这个helper，断言里只出现一处转换。
    #[inline]
    fn hbits(v: u16) -> u16 {
        v
    }

    // -----------------------------------------------------------------------
    // 测试脚手架：合成一个最小但**合法**的 DWA 块
    // -----------------------------------------------------------------------

    /// zlib 流（RFC 1950）= `0x78 0x01` + stored DEFLATE + Adler-32。
    ///
    /// 刻意独立于 `exr_write::deflate_zlib` 实现：若共用一份代码，DWA 测试
    /// 与写入器测试会一起错、互相掩盖。这里独立写一遍（RFC 1950 只有 ~20 行）。
    fn zlib(payload: &[u8]) -> Vec<u8> {
        const MOD: u32 = 65521;
        let (mut a, mut b) = (1u32, 0u32);
        for &x in payload {
            a = (a + x as u32) % MOD;
            b = (b + a) % MOD;
        }
        let mut out = Vec::with_capacity(payload.len() + 16);
        out.push(0x78);
        out.push(0x01);
        // stored DEFLATE 块
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
        out.extend_from_slice(&((b << 16) | a).to_be_bytes());
        out
    }

    /// 合成的 DWA 块描述（各区的原始内容，压缩交给 [`build_block`]）。
    #[derive(Default, Clone)]
    struct BlockSpec {
        version: u64,
        rules: Option<Vec<u8>>,
        unknown: Vec<u8>,
        dc: Vec<u16>,
        ac: Vec<u16>,
        ac_huf: bool,
        rle_raw: Vec<u8>,
    }

    impl BlockSpec {
        /// 一个 `blocks_x × blocks_y` 的单通道 LOSSY_DCT 块规格，
        /// 每个 DCT 块给定 DC（AC 恒空 → 走 DC-only 快路径）。
        ///
        /// # 参数 `dc_pixel`：每个块的**像素常量**（人能读懂的意图），不是 DC 系数
        ///
        /// 【W13 修正】DC 区存的是正交 DCT-II 的 DC 系数（= 常量 c 的 **8c**），
        /// 解码端 `dct_inverse_dc_only` 按官方 `×0.3535536²`（=×1/8）还原。
        /// 修正前解码端漏乘 1/8，夹具相应直接写像素值，两侧错误抵消——
        /// 现在解码端是官方语义，**缩放必须由夹具侧补上**。
        ///
        /// 把这一步收进本构造函数，而不是让每个测试各写一遍 `* 8.0`：
        /// 后者 inevitably 会有漏改的（本次就有 3 个测试漏改），
        /// 前者只此一处、且函数名已声明参数是「像素常量」。
        fn dc_only(version: u64, blocks_x: usize, blocks_y: usize, dc_pixel: Vec<f32>) -> Self {
            let mut spec = Self {
                version,
                ..Self::default()
            };
            // 正交 DCT-II：常量 c ⇒2D DC = √(1/8)·√(1/8)·64·c = 8c。
            spec.dc = dc_pixel
                .iter()
                .map(|&c| f32_to_f16(c * 8.0))
                .collect();
            // AC：官方 `rleAc` 对「整块全零且零游程恰好填满」写一枚 `0xff00` EOB，
            // 其余情况不写。这里逐块一枚，与官方可产出的一致形态。
            let n = blocks_x * blocks_y;
            spec.ac = vec![0xff00; n];
            spec.ac_huf = false; // DC-only 用 deflate AC 最省事
            spec
        }
    }

    /// 按真实布局拼出一个 DWA 块负载。
    fn build_block(spec: &BlockSpec) -> Vec<u8> {
        let unknown_z = if spec.unknown.is_empty() {
            Vec::new()
        } else {
            zlib(&spec.unknown)
        };
        // DC 区走官方 `Imf::Zip`（**预测器 + 奇偶交织** + deflate），
        // 不是裸 zlib —— 与 `zip_pack` / `zip_decompress` 同一条路径。
        // 夹具必须复用生产编码器，否则会重演"夹具与实现各写一份、语义漂移"。
        let dc_z = if spec.dc.is_empty() {
            Vec::new()
        } else {
            zip_pack(&spec.dc)
        };
        // AC：本测试只用 deflate 策略（ac_compression = 1）。
        let mut ac_bytes: Vec<u8> = Vec::with_capacity(spec.ac.len() * 2);
        for v in &spec.ac {
            ac_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let ac_z = if ac_bytes.is_empty() {
            Vec::new()
        } else {
            zlib(&ac_bytes)
        };
        // RLE 区：EXR RLE（**无**长度头，由块头携带两个长度）。
        // 先算 zlib **之前**的行程流长度（= `rleUncompressedSize`），再 zlib。
        //
        // ⚠️ 直接复用 [`crate::exr::rle_compress`]（剥掉 4 字节长度头），**不再**
        // 在测试夹具里手写一份行程编码——W13 互通对拍的教训：手写副本与生产
        // 编码器各自演化出**相反的控制字节极性**，夹具全绿而生产数据全错。
        // 「夹具与实现共用一份代码」是这里唯一能防住漂移的做法。
        let rle_packed: Vec<u8> = if spec.rle_raw.is_empty() {
            Vec::new()
        } else {
            let mut full = crate::exr::rle_compress(&spec.rle_raw);
            full.drain(0..4); // 去掉 4 字节长度头
            full
        };
        // `rleUncompressedSize` = 行程流（zlib 前）长度；`rleRawSize` = 原始字节长度。
        let rle_packed_pre_zlib = rle_packed.len() as u64;
        let rle_z = if rle_packed.is_empty() {
            Vec::new()
        } else {
            zlib(&rle_packed)
        };

        let mut out: Vec<u8> = Vec::new();
        let sizes: [u64; NUM_SIZES_SINGLE] = [
            spec.version,
            spec.unknown.len() as u64,
            unknown_z.len() as u64,
            ac_z.len() as u64,
            dc_z.len() as u64,
            rle_z.len() as u64,
            rle_packed_pre_zlib,
            spec.rle_raw.len() as u64,
            spec.ac.len() as u64,
            spec.dc.len() as u64,
            // 0 = STATIC_HUFFMAN，1 = deflate。`ac_huf` 为真 → 用 Huffman。
            u64::from(!spec.ac_huf),
        ];
        for s in sizes {
            out.extend_from_slice(&s.to_le_bytes());
        }
        if let Some(r) = &spec.rules {
            out.extend_from_slice(r);
        }
        out.extend_from_slice(&unknown_z);
        out.extend_from_slice(&ac_z);
        out.extend_from_slice(&dc_z);
        out.extend_from_slice(&rle_z);
        out
    }

    /// 通道上下文简写。
    /// 构造测试用通道表（`Vec`，供 `DwaContext` 借用）。
    fn chans(names: &[(&str, PixelType, bool)]) -> Vec<(String, PixelType, bool)> {
        names
            .iter()
            .map(|(n, t, p)| ((*n).to_string(), *t, *p))
            .collect()
    }

    /// 通道上下文简写（借用调用方的通道表）。
    fn ctx_of(
        chans: &[(String, PixelType, bool)],
        width: usize,
        height: usize,
    ) -> DwaContext<'_> {
        DwaContext {
            channels: chans,
            width,
            height,
        }
    }

    /// 把块内布局字节拆成逐通道平面（f32 值）。
    ///
    /// 块内布局是 **逐扫描线 × 逐通道 × 逐像素**（`scanline × channel × pixel`），
    /// 故第 `y` 条扫描线的起点是**所有通道行字节之和**（`Σ wᵢ·width`），而通道
    /// `ci` 在该扫描线内的偏移是 `Σ_{j<ci} wⱼ·width`——不是 `y · w_ci · width`。
    /// 后者只在**单通道**时恰好成立；多通道下会把后续扫描线读成别的通道的数据。
    fn split_planes(
        raw: &[u8],
        names: &[(String, PixelType, bool)],
        width: usize,
        height: usize,
    ) -> Vec<Vec<f32>> {
        let widths: Vec<usize> = names.iter().map(|(_, pt, _)| pt.byte_width()).collect();
        let row_bytes: usize = widths.iter().copied().sum::<usize>().saturating_mul(width);
        let mut out: Vec<Vec<f32>> = names.iter().map(|_| Vec::new()).collect();
        for y in 0..height {
            let mut ch_off = 0usize;
            for (ci, (_, pt, _)) in names.iter().enumerate() {
                let bw = pt.byte_width();
                for x in 0..width {
                    let start = y * row_bytes + ch_off + x * bw;
                    let v = match pt {
                        PixelType::Half => {
                            let b = [
                                raw.get(start).copied().unwrap_or(0),
                                raw.get(start + 1).copied().unwrap_or(0),
                            ];
                            f16_to_f32(u16::from_le_bytes(b))
                        }
                        PixelType::Float => {
                            let b = [
                                raw.get(start).copied().unwrap_or(0),
                                raw.get(start + 1).copied().unwrap_or(0),
                                raw.get(start + 2).copied().unwrap_or(0),
                                raw.get(start + 3).copied().unwrap_or(0),
                            ];
                            f32::from_le_bytes(b)
                        }
                        PixelType::Uint => {
                            let b = [
                                raw.get(start).copied().unwrap_or(0),
                                raw.get(start + 1).copied().unwrap_or(0),
                                raw.get(start + 2).copied().unwrap_or(0),
                                raw.get(start + 3).copied().unwrap_or(0),
                            ];
                            u32::from_le_bytes(b) as f32
                        }
                    };
                    out[ci].push(v);
                }
                ch_off += widths[ci].saturating_mul(width);
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // 1) parse_sizes
    // -----------------------------------------------------------------------

    #[test]
    fn parse_sizes_reads_little_endian_header_in_fixed_order() {
        let spec = BlockSpec::dc_only(2, 1, 1, vec![1.0]);
        let bytes = build_block(&spec);
        let s = parse_sizes(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.version, 2);
        assert_eq!(s.dc_uncompressed_count, 1);
        assert_eq!(s.ac_uncompressed_count, 1);
        assert_eq!(s.ac_compression, 1);
        assert_eq!(s.unknown_compressed_size, 0);
        // 子块偏移：11×8 = 88 字节头，随后规则表（本例无）→ unknown 区在 88。
        assert_eq!(s.unknown_compressed_size, 0);
        assert!(s.dc_compressed_size > 0, "DC 区应非空");
    }

    #[test]
    fn parse_sizes_rejects_truncated_header() {
        let err = parse_sizes(&[0u8; 40]).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("88"), "须说明所需字节数：{err}");
    }

    #[test]
    fn parse_sizes_rejects_negative_fields_read_as_signed() {
        let spec = BlockSpec::dc_only(1, 1, 1, vec![1.0]);
        let mut bytes = build_block(&spec);
        // 把 dcUncompressedCount（第 9 项，偏移 72）写成 -1（i64 全 1）。
        bytes[72..80].copy_from_slice(&(-1i64).to_le_bytes());
        let err = parse_sizes(&bytes).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("负"), "须指出字段为负：{err}");
    }

    // -----------------------------------------------------------------------
    // 2) classify_channels
    // -----------------------------------------------------------------------

    #[test]
    fn classify_channels_assigns_scheme_and_type_per_channel() {
        let rules = default_rules();
        let names = vec![
            ("R".to_string(), PixelType::Half, false),
            ("G".to_string(), PixelType::Half, false),
            ("B".to_string(), PixelType::Half, false),
            ("A".to_string(), PixelType::Uint, false),
            ("Z".to_string(), PixelType::Float, false),
        ];
        let plan = classify_channels(&names, &rules);
        assert_eq!(plan.channels[0].scheme, DwaScheme::LossyDct);
        assert_eq!(plan.channels[3].scheme, DwaScheme::Rle);
        // Z 不匹配任何规则 → Unknown（原样存储）。
        assert_eq!(plan.channels[4].scheme, DwaScheme::Unknown);
        // R/G/B 组成一个 CSC 组。
        assert_eq!(plan.csc_sets.len(), 1);
        assert_eq!(plan.csc_sets[0].idx, [0, 1, 2]);
    }

    #[test]
    fn classify_channels_is_deterministic_and_prefix_grouped() {
        let rules = default_rules();
        // 同一前缀 "beauty." 下的 R/G/B 应归为一组。
        let names = vec![
            ("beauty.R".to_string(), PixelType::Half, false),
            ("beauty.G".to_string(), PixelType::Half, false),
            ("beauty.B".to_string(), PixelType::Half, false),
        ];
        let a = classify_channels(&names, &rules);
        let b = classify_channels(&names, &rules);
        assert_eq!(a, b, "同输入两次分类须完全一致");
        assert_eq!(a.csc_sets.len(), 1);
        assert_eq!(a.csc_sets[0].idx, [0, 1, 2]);
    }

    #[test]
    fn legacy_rules_are_case_insensitive_and_default_rules_are_not() {
        let legacy = legacy_rules();
        let default = default_rules();
        // legacy 表全部大小写不敏感：大写 "R" 也能匹配小写规则 "r"。
        let upper = vec![("R".to_string(), PixelType::Half, false)];
        assert_eq!(
            classify_channels(&upper, &legacy).channels[0].scheme,
            DwaScheme::LossyDct
        );
        // 默认表大小写敏感：小写 "r" 不匹配 "R"。
        let lower = vec![("r".to_string(), PixelType::Half, false)];
        assert_eq!(
            classify_channels(&lower, &default).channels[0].scheme,
            DwaScheme::Unknown
        );
    }

    // -----------------------------------------------------------------------
    // 3) DWA 块解码：DC-only 快路径（half / float）
    // -----------------------------------------------------------------------

    #[test]
    fn dwab_dc_only_half_block_decodes_to_a_constant_plane() {
        // 1×1 个 DCT 块，像素常量 0.5。DC 区存的是正交 DCT-II 的 DC 系数
        // （=8c），缩放由 `BlockSpec::dc_only` 统一处理，见那里的说明。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.5]);
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let raw = decompress_block(&bytes, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), 8 * 8 * 2, "8×8 half = 128 字节");
        let planes = split_planes(&raw, &names, 8, 8);
        for (i, &v) in planes[0].iter().enumerate() {
            // p_linear=true → 不做非线性 LUT 还原，故值精确等于 half(0.5)。
            assert_eq!(
                hbits(f32_to_f16(v)),
                hbits(f32_to_f16(0.5)),
                "像素 {i} 应为 0.5，实为 {v}"
            );
        }
    }

    #[test]
    fn dwab_dc_only_float_block_decodes_to_a_constant_plane() {
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.25]);
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Float, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let raw = decompress_block(&bytes, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), 8 * 8 * 4);
        let planes = split_planes(&raw, &names, 8, 8);
        for (i, &v) in planes[0].iter().enumerate() {
            // float 通道也先解成 half 再展开，故期望是 half(0.25) 的 f32 值。
            assert_eq!(v.to_bits(), 0.25f32.to_bits(), "像素 {i} 应为 0.25，实为 {v}");
        }
    }

    #[test]
    fn dc_only_multi_block_plane_places_each_block_independently() {
        // 2×2 个 DCT 块（16×16 图像），四个块给不同 DC。
        let dcs = [0.5f32, 0.25, 1.0, 2.0];
        let spec = BlockSpec::dc_only(1, 2, 2, dcs.to_vec());
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 16, 16);
        let raw = decompress_block(&bytes, &ctx, 16).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&raw, &names, 16, 16);
        for by in 0..2usize {
            for bx in 0..2usize {
                let want = dcs[by * 2 + bx];
                for ly in 0..8usize {
                    for lx in 0..8usize {
                        let idx = (by * 8 + ly) * 16 + (bx * 8 + lx);
                        assert_eq!(
                            hbits(f32_to_f16(planes[0][idx])),
                            hbits(f32_to_f16(want)),
                            "块 ({bx},{by}) 像素 ({lx},{ly})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn partial_edge_block_is_clipped_to_image_bounds() {
        // 宽 5 高 3（不足一个 8×8 块）：越界像素必须被裁掉，不得越界写。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![1.0]);
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 5, 3);
        let raw = decompress_block(&bytes, &ctx, 3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), 5 * 3 * 2, "只输出 5×3 像素");
        let planes = split_planes(&raw, &names, 5, 3);
        assert_eq!(planes[0].len(), 15);
        for &v in &planes[0] {
            assert_eq!(hbits(f32_to_f16(v)), hbits(f32_to_f16(1.0)));
        }
    }

    // -----------------------------------------------------------------------
    // 4) DWA 块解码：AC / IDCT 路径
    // -----------------------------------------------------------------------

    #[test]
    fn ac_path_with_only_dc_matches_dc_only_fast_path() {
        // AC 存在但整块仍无非零 AC（立即 0xff00）→ 与 DC-only 同解。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.75]);
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let raw = decompress_block(&bytes, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&raw, &names, 8, 8);
        for &v in &planes[0] {
            assert_eq!(hbits(f32_to_f16(v)), hbits(f32_to_f16(0.75)));
        }
    }

    #[test]
    fn ac_path_with_nonzero_coefficients_goes_through_idct() {
        // 给一个非零 AC 系数：结果必须不等于 DC-only 的常量块，
        // 且落在 IDCT 的有界范围内（|值| ≤ 系数幅值 × 1.5 + DC）。
        let dc = f32_to_f16(0.0);
        let ac_first = f32_to_f16(0.5); // zigzag 位置 1
        let spec = BlockSpec {
            version: 1,
            dc: vec![dc],
            ac: vec![ac_first, 0xff00],
            ..BlockSpec::default()
        };
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let raw = decompress_block(&bytes, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&raw, &names, 8, 8);
        // 非恒定：至少有两个不同的值。
        let first = hbits(f32_to_f16(planes[0][0]));
        assert!(
            planes[0].iter().any(|&v| hbits(f32_to_f16(v)) != first),
            "非零 AC 必须让块内产生变化，实际整块恒为 {:?}",
            first
        );
        // 所有值都是有限 half，且幅值有界（单个 AC=0.5 的行/列基波最大 0.5）。
        for &v in &planes[0] {
            assert!(v.is_finite(), "IDCT 输出须有限，实为 {v}");
            assert!(v.abs() <= 1.0, "IDCT 输出幅值须有界，实为 {v}");
        }
    }

    #[test]
    fn ac_zero_runs_are_expanded() {
        // 0xff 前缀 = 游程：0xff00 结束，0xff|k 表示后续 k 个零。
        // 这里给 0xff02（跳 2 个零）后跟一个非零值，验证游程展开生效。
        let spec = BlockSpec {
            version: 1,
            dc: vec![f32_to_f16(0.0)],
            ac: vec![0xff02, f32_to_f16(0.25), 0xff00],
            ..BlockSpec::default()
        };
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let raw = decompress_block(&bytes, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        let planes = split_planes(&raw, &names, 8, 8);
        for &v in &planes[0] {
            assert!(v.is_finite());
        }
        // 与「无游程、系数放在位置 1」的解码结果须不同（游程确实改变了位置）。
        let spec2 = BlockSpec {
            version: 1,
            dc: vec![f32_to_f16(0.0)],
            ac: vec![f32_to_f16(0.25), 0xff00],
            ..BlockSpec::default()
        };
        let bytes2 = build_block(&spec2);
        let raw2 = decompress_block(&bytes2, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_ne!(raw, raw2, "游程展开须改变系数落位");
    }

    // -----------------------------------------------------------------------
    // 5) DWA 块解码：Unknown（非 AC）回退路径
    // -----------------------------------------------------------------------

    #[test]
    fn unknown_scheme_channel_round_trips_verbatim() {
        // 通道名 Z 不匹配任何规则 → Unknown → 原样存进 unknown 区。
        // 4×4 uint = 64 字节，须按「整平面」大小提供。
        let vals: Vec<u8> = (0..64u32).map(|i| (i * 7 % 251) as u8).collect();
        let spec = BlockSpec {
            version: 1,
            unknown: vals.clone(),
            ..BlockSpec::default()
        };
        let bytes = build_block(&spec);
        let names = chans(&[("Z", PixelType::Uint, false)]);
        let ctx = ctx_of(&names, 4, 4);
        let raw = decompress_block(&bytes, &ctx, 4).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), 4 * 4 * 4, "4×4 uint = 64 字节");
        assert_eq!(&raw[..], &vals[..], "Unknown 区须逐字节原样还原");
    }

    #[test]
    fn unknown_scheme_multi_channel_consumes_in_channel_order() {
        // 两个 Unknown 通道（2×4 uint，各 32 字节）。
        // ⚠️ 输出是 **EXR 块布局**：逐扫描线 × 逐通道 × 逐像素，
        // 所以同一扫描线内是 [ZA 该行 8B][ZB 该行 8B]，不是「ZA 全部在前」。
        let a: Vec<u8> = (0..32u8).collect();
        let b: Vec<u8> = (100..132u8).collect();
        let mut unknown = a.clone();
        unknown.extend_from_slice(&b);
        let spec = BlockSpec {
            version: 1,
            unknown,
            ..BlockSpec::default()
        };
        let bytes = build_block(&spec);
        let names = chans(&[
            ("ZA", PixelType::Uint, false),
            ("ZB", PixelType::Uint, false),
        ]);
        let ctx = ctx_of(&names, 2, 4);
        let raw = decompress_block(&bytes, &ctx, 4).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), 4 * 2 * 2 * 4, "4 行 × 2 通道 × 2 像素 × 4 字节");
        // 逐扫描线核对：ZA 与 ZB 的像素数据须按通道顺序交替出现。
        let row = 2 * 4; // 一行每通道 8 字节
        for y in 0..4usize {
            let got_a = &raw[y * row * 2..y * row * 2 + row];
            let got_b = &raw[y * row * 2 + row..(y + 1) * row * 2];
            assert_eq!(
                got_a,
                &a[y * row..(y + 1) * row],
                "第 {y} 行 ZA 平面"
            );
            assert_eq!(
                got_b,
                &b[y * row..(y + 1) * row],
                "第 {y} 行 ZB 平面"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6) DWA 块解码：RLE 通道路径
    // -----------------------------------------------------------------------

    #[test]
    fn rle_scheme_uint_channel_round_trips() {
        // 通道 A（Uint）→ RLE 方案。RLE 区按字节平面组织：
        // uint 是 4 字节/像素 → 4 个字节平面依次排列。
        let px: Vec<u32> = (0..4u32).map(|i| 1000 + i * 11).collect();
        let width = 2usize;
        let height = 2usize;
        let stride = width * height;
        // 字节平面：b0 平面、B1 平面、b2 平面、b3 平面。
        let mut rle_raw: Vec<u8> = Vec::with_capacity(stride * 4);
        for b in 0..4usize {
            for &p in &px {
                rle_raw.push(((p >> (8 * b)) & 0xff) as u8);
            }
        }
        let spec = BlockSpec {
            version: 1,
            rle_raw,
            ..BlockSpec::default()
        };
        let bytes = build_block(&spec);
        let names = chans(&[("A", PixelType::Uint, false)]);
        let ctx = ctx_of(&names, width, height);
        let raw = decompress_block(&bytes, &ctx, height).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw.len(), width * height * 4);
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            let got = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(got, px[i], "像素 {i}: 期望 {} 实得 {got}", px[i]);
        }
    }

    // -----------------------------------------------------------------------
    // 7) 确定性
    // -----------------------------------------------------------------------

    #[test]
    fn decoding_the_same_block_twice_is_byte_identical() {
        // 两个 LOSSY_DCT 通道（R/G 各自一个解码器），验证多解码器下仍确定性。
        let dcs = [0.1f32, 0.2, 0.3, 0.4];
        let mut spec = BlockSpec::dc_only(1, 2, 2, dcs.to_vec());
        // 第二通道的 DC（comp-major 平面，紧随第一通道）。
        let dcs2 = [0.5f32, 0.6, 0.7, 0.8];
        spec.dc.extend(dcs2.iter().map(|&v| f32_to_f16(v)));
        spec.ac.extend(std::iter::repeat(0xff00).take(4)); // 第二通道的 AC 结束符
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true), ("BY", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 16, 16);
        let a = decompress_block(&bytes, &ctx, 16).unwrap_or_else(|e| panic!("{e}"));
        let b = decompress_block(&bytes, &ctx, 16).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "同输入解码两遍须字节相同");
    }

    // -----------------------------------------------------------------------
    // 8) 畸形输入：结构化错误，绝不 panic
    // -----------------------------------------------------------------------

    #[test]
    fn malformed_dwa_blocks_are_structured_errors_not_panics() {
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let good = build_block(&BlockSpec::dc_only(1, 1, 1, vec![0.5]));

        // 逐字节截断：任何前缀都不得 panic。
        for cut in 0..good.len() {
            let r = decompress_block(&good[..cut], &ctx, 8);
            assert!(r.is_err() || r.is_ok(), "截断到 {cut} 字节不得 panic");
        }
        // 尾部多出的垃圾字节不得改变结果（多余字节被忽略）。
        let mut padded = good.clone();
        padded.extend_from_slice(&[0xab; 16]);
        let a = decompress_block(&good, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        let b = decompress_block(&padded, &ctx, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "尾部多余字节须被忽略");
    }

    #[test]
    fn dwa_version_above_two_is_structurally_rejected() {
        let spec = BlockSpec::dc_only(3, 1, 1, vec![0.5]);
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains('3'), "须回显实际版本：{err}");
    }

    #[test]
    fn dwa_oversized_subblock_lengths_are_structurally_rejected() {
        // 把 acCompressedSize 改成极大值 → 越界必须结构化报错。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.5]);
        let mut bytes = build_block(&spec);
        bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes()); // acCompressedSize
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn dwa_subblock_length_overflow_is_structurally_rejected() {
        // 各子块长度取 i64 上界（2^63-1）附近：单个都过负值检查、求和**不**溢出
        // u64，但必然远超实际负载长度 → 必须在越界检查处被拦下。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.5]);
        let mut bytes = build_block(&spec);
        let big = (1u64 << 63) - 1;
        bytes[16..24].copy_from_slice(&big.to_le_bytes()); // unknownCompressedSize
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(
            format!("{err}").contains("截断") || format!("{err}").contains("需"),
            "须明确指出负载长度不足：{err}"
        );
    }

    #[test]
    fn dwa_header_prefix_plus_subblocks_cannot_overflow_the_cursor() {
        // `cur`(规则表已消费字节) + `total`(四区长度之和) 的加法是 checked_*，
        // 且随后还要过一次「实际负载够不够」的比较。这里用最大合法长度驱动，
        // 断言得到的是**结构化错误**而不是 panic 或错误的长度回绕。
        let spec = BlockSpec::dc_only(2, 1, 1, vec![0.5]);
        let mut bytes = build_block(&spec);
        let big = (1u64 << 62) + (1u64 << 61);
        // unknown / ac / dc / rle 四个长度都设成大值（各自 i64 视角为正）。
        for off in [16usize, 24, 32, 40, 48] {
            bytes[off..off + 8].copy_from_slice(&big.to_le_bytes());
        }
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        // 不得 panic（无论是 Err 还是 Ok）。
        let r = decompress_block(&bytes, &ctx, 8);
        assert!(r.is_err(), "超长子块长度必须报错，实得 {:?}", r.map(|v| v.len()));
    }

    #[test]
    fn dwa_subblock_length_above_i64_max_is_rejected_as_negative() {
        // 2^63 与 2^64-1 在 OpenEXR 的 int64 语义下都是**负数**，
        // 必须被「负值」检查拦下（而不是当成巨大的合法长度）。
        for v in [1u64 << 63, u64::MAX] {
            let spec = BlockSpec::dc_only(1, 1, 1, vec![0.5]);
            let mut bytes = build_block(&spec);
            bytes[16..24].copy_from_slice(&v.to_le_bytes());
            let names = chans(&[("Y", PixelType::Half, true)]);
            let ctx = ctx_of(&names, 8, 8);
            let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
            assert_eq!(err.code, "EXR2001", "{err}");
            assert!(
                format!("{err}").contains("负"),
                "v={v} 须被识别为负值长度：{err}"
            );
        }
    }

    #[test]
    fn dwa_zlib_subblock_with_corrupt_header_is_rejected() {
        // DC 区是 zlib 流；把它的 2 字节头破坏掉 → CMF 校验必须失败。
        let spec = BlockSpec::dc_only(1, 1, 1, vec![0.5]);
        let good = build_block(&spec);
        // 定位 DC 区起点：头 88 字节（version 1 → 无规则表）+ unknown(0) + AC 区。
        // AC 区长度可从 sizes[3] 读出。
        let ac_len = u64::from_le_bytes(good[24..32].try_into().unwrap_or([0; 8])) as usize;
        let dc_off = 88 + ac_len;
        let mut bytes = good.clone();
        // DC 区首字节改成 0x00（低 4 位 = 0 ≠ 8）。
        if let Some(b) = bytes.get_mut(dc_off) {
            *b = 0x00;
        } else {
            panic!("DC 区起点 {dc_off} 越界（块长 {}）", good.len());
        }
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("zlib"), "须指出是 zlib 头问题：{err}");
    }

    /// 组装一条 version-2 规则表：`u16 总长` + 规则体。
    ///
    /// 总长**必须**等于 2 + 规则体长度，否则 `parse_rules` 会提前停止、
    /// 让后续子块整体错位（测试就会因为无关的 zlib 错误而假失败）。
    fn rules_table(body: &[u8]) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::with_capacity(2 + body.len());
        v.extend_from_slice(&((2 + body.len()) as u16).to_le_bytes());
        v.extend_from_slice(body);
        v
    }

    /// 一条规则体：NUL 结尾后缀 + 标志字节 + 像素类型字节。
    fn rule_body(suffix: &str, flags: u8, pt: PixelType) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        v.extend_from_slice(suffix.as_bytes());
        v.push(0);
        v.push(flags);
        v.push(pt.code() as u8);
        v
    }

    #[test]
    fn dwa_rule_table_with_bad_scheme_bits_is_rejected() {
        // version 2 + 规则表 scheme 位 = 3（非法，须 0..2）。
        let body = rule_body("Y", (0 << 4) | (3 << 2), PixelType::Half);
        let mut spec = BlockSpec::dc_only(2, 1, 1, vec![0.5]);
        spec.rules = Some(rules_table(&body));
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("scheme"), "须指出 scheme 非法：{err}");
    }

    #[test]
    fn dwa_rule_table_with_bad_csc_index_is_rejected() {
        // cscIdx 编码为 15 → 解码后 = 14，越界（须 -1..2）。
        let body = rule_body("Y", (15 << 4) | (1 << 2), PixelType::Half);
        let mut spec = BlockSpec::dc_only(2, 1, 1, vec![0.5]);
        spec.rules = Some(rules_table(&body));
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        assert!(format!("{err}").contains("cscIdx"), "{err}");
    }

    #[test]
    fn dwa_rule_table_with_bad_pixel_type_is_rejected() {
        // 像素类型字节 = 9（非法，须 0..2）。
        let mut body = rule_body("Y", 1 << 2, PixelType::Half);
        let last = body.len() - 1;
        body[last] = 9;
        let mut spec = BlockSpec::dc_only(2, 1, 1, vec![0.5]);
        spec.rules = Some(rules_table(&body));
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn dwa_rule_table_shorter_than_length_prefix_is_rejected() {
        // 声明长度 2 但规则体被截断 → 结构化错误。
        let mut spec = BlockSpec::dc_only(2, 1, 1, vec![0.5]);
        spec.rules = Some(vec![8u8, 0]); // 声明 8 字节，实际只有 2
        let bytes = build_block(&spec);
        let names = chans(&[("Y", PixelType::Half, true)]);
        let ctx = ctx_of(&names, 8, 8);
        let err = decompress_block(&bytes, &ctx, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn dwa_rule_table_round_trips_through_encode_rule() {
        // encode_rule → parse_rules 往返（version 2 的自造规则表）。
        let rule = DwaRule {
            suffix: "Y".to_string(),
            scheme: DwaScheme::LossyDct,
            pixel_type: PixelType::Half,
            csc_idx: 1,
            case_insensitive: true,
        };
        let encoded = encode_rule(&rule);
        let mut table: Vec<u8> = Vec::new();
        let size = 2 + encoded.len();
        table.extend_from_slice(&(size as u16).to_le_bytes());
        table.extend_from_slice(&encoded);
        let (rules, used) = parse_rules(&table).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(used, size);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0], rule, "规则往返须逐字段一致");
    }

    // -----------------------------------------------------------------------
    // 9) Huf（AC 的 Huffman 路径）
    // -----------------------------------------------------------------------

    #[test]
    fn huf_uncompress_rejects_short_and_out_of_range_tables() {
        // 太短 → 结构化错误。
        let err = huf_uncompress(&[0u8; 10], 4).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
        // im / iM 越界（HUF_ENCSIZE = 65537）。
        let mut hdr = vec![0u8; 20];
        hdr[0..4].copy_from_slice(&(-1i32).to_le_bytes());
        let err = huf_uncompress(&hdr, 4).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn huf_uncompress_with_zero_raw_is_empty_for_short_input() {
        // n_raw = 0 且负载短 → 返回空 Vec（不是错误）。
        assert!(huf_uncompress(&[0u8; 4], 0).unwrap_or_else(|e| panic!("{e}")).is_empty());
    }

    #[test]
    fn huf_uncompress_rejects_truncated_data_section() {
        // 声明 nBits 超过实际剩余字节 → 结构化错误。
        let mut buf = vec![0u8; 20];
        buf[0..4].copy_from_slice(&0i32.to_le_bytes()); // im
        buf[4..8].copy_from_slice(&0i32.to_le_bytes()); // iM
        buf[12..16].copy_from_slice(&(9999u32).to_le_bytes()); // nBits 过大
        buf[16..20].copy_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 2]);
        let err = huf_uncompress(&buf, 8).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn huf_uncompress_rejects_undefined_codeword() {
        // 合法表头 + 垃圾码字 → 不得 panic，须结构化错误。
        let mut buf: Vec<u8> = vec![0u8; 20];
        buf[0..4].copy_from_slice(&0i32.to_le_bytes());
        buf[4..8].copy_from_slice(&1i32.to_le_bytes());
        buf[12..16].copy_from_slice(&64u32.to_le_bytes()); // nBits = 8 字节
        buf[16..20].copy_from_slice(&0u32.to_le_bytes());
        // 6 位码长全 0（符号 0 无长度），随后数据位 → 必然命中未定义码字。
        buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff]);
        let err = huf_uncompress(&buf, 4).unwrap_err();
        assert_eq!(err.code, "EXR2001", "{err}");
    }

    #[test]
    fn huf_canonical_code_table_assigns_distinct_codes_per_length() {
        // 三个符号长度 2 → canonical 码 0,1,2（最高位不同）。
        let mut lengths = vec![2u32, 2, 2];
        huf_canonical_code_table(&mut lengths);
        let code = |v: u32| v >> 6;
        let len = |v: u32| v & 63;
        assert_eq!(len(lengths[0]), 2);
        assert_eq!(code(lengths[0]), 0);
        assert_eq!(code(lengths[1]), 1);
        assert_eq!(code(lengths[2]), 2);
    }

    // -----------------------------------------------------------------------
    // 10) 非线性 LUT / 反 zigzag / IDCT 的数值性质
    // -----------------------------------------------------------------------

    #[test]
    fn dwa_to_linear_is_identity_on_zero_and_sign_preserving() {
        assert_eq!(dwa_to_linear(0x0000), 0x0000, "half 0 → 0");
        // NaN / Inf → 0（与 OpenEXR 一致）。
        assert_eq!(dwa_to_linear(0x7c00), 0, "Inf → 0");
        assert_eq!(dwa_to_linear(0x7e00), 0, "NaN → 0");
        // 负数保持符号。
        let neg = f32_to_f16(-0.5);
        assert_eq!(dwa_to_linear(neg) & 0x8000, 0x8000, "负值须保持符号位");
    }

    /// W13 互操作对拍：从官方 `dwaLookups.h::dwaCompressorToLinear` 静态表
    /// 提取的**每指数域边界**真值（`exp:首索引:首值:末索引:末值`）。
    ///
    /// 这张表固化了本次修复的关键行为：half 指数域 `0x11`（十进制 17）起，
    /// `logBase^(x-1)` 超出 half 表示范围，官方一律输出 `±inf`。
    /// 修复前我方带"clamp 到 65504"逻辑，在这些条目上全部输出 `0x7BFF`/`0xFBFF`，
    /// 与官方相差 27626 项——而当时的单元测试因只测 0/1/0.5 等小值而全绿（假绿）。
    const LUT_EDGES: &[(u16, u16, u16, u16, u16)] = &[
        (0x00, 0x0000, 0x0000, 0x83FF, 0x8000),
        (0x01, 0x0400, 0x0000, 0x87FF, 0x8000),
        (0x02, 0x0800, 0x0000, 0x8BFF, 0x8000),
        (0x03, 0x0C00, 0x0000, 0x8FFF, 0x8001),
        (0x04, 0x1000, 0x0001, 0x93FF, 0x8004),
        (0x05, 0x1400, 0x0004, 0x97FF, 0x8012),
        (0x06, 0x1800, 0x0012, 0x9BFF, 0x8054),
        (0x07, 0x1C00, 0x0054, 0x9FFF, 0x8184),
        (0x08, 0x2000, 0x0184, 0xA3FF, 0x86F5),
        (0x09, 0x2400, 0x06F7, 0xA7FF, 0x8FFE),
        (0x0A, 0x2800, 0x1000, 0xABFF, 0x9897),
        (0x0B, 0x2C00, 0x1898, 0xAFFF, 0xA146),
        (0x0C, 0x3000, 0x2147, 0xB3FF, 0xAA0E),
        (0x0D, 0x3400, 0x2A10, 0xB7FF, 0xB2F5),
        (0x0E, 0x3800, 0x32F7, 0xBBFF, 0xBBFE),
        (0x0F, 0x3C00, 0x3C00, 0xBFFF, 0xC881),
        (0x10, 0x4000, 0x4883, 0xC3FF, 0xE1B8),
        (0x11, 0x4400, 0x61BE, 0xC7FF, 0xFC00),
        (0x12, 0x4800, 0x7C00, 0xCBFF, 0xFC00),
        (0x13, 0x4C00, 0x7C00, 0xCFFF, 0xFC00),
        (0x14, 0x5000, 0x7C00, 0xD3FF, 0xFC00),
        (0x15, 0x5400, 0x7C00, 0xD7FF, 0xFC00),
        (0x16, 0x5800, 0x7C00, 0xDBFF, 0xFC00),
        (0x17, 0x5C00, 0x7C00, 0xDFFF, 0xFC00),
        (0x18, 0x6000, 0x7C00, 0xE3FF, 0xFC00),
        (0x19, 0x6400, 0x7C00, 0xE7FF, 0xFC00),
        (0x1A, 0x6800, 0x7C00, 0xEBFF, 0xFC00),
        (0x1B, 0x6C00, 0x7C00, 0xEFFF, 0xFC00),
        (0x1C, 0x7000, 0x7C00, 0xF3FF, 0xFC00),
        (0x1D, 0x7400, 0x7C00, 0xF7FF, 0xFC00),
        (0x1E, 0x7800, 0x7C00, 0xFBFF, 0xFC00),
        (0x1F, 0x7C00, 0x0000, 0xFFFF, 0x0000),
    ];

    #[test]
    fn dwa_to_linear_matches_official_edges_across_all_exponents() {
        for &(exp, first, first_val, last, last_val) in LUT_EDGES {
            assert_eq!(
                dwa_to_linear(first),
                first_val,
                "exp 0x{exp:02X} 首索引 0x{first:04X} 应为 0x{first_val:04X}"
            );
            assert_eq!(
                dwa_to_linear(last),
                last_val,
                "exp 0x{exp:02X} 末索引 0x{last:04X} 应为 0x{last_val:04X}"
            );
        }
    }

    /// W13 互操作对拍：**512 项官方真值对**，覆盖全部 32 个half 指数域×
    /// 16 个尾数分区（含每域首尾与次正规/正规边界）。
    ///
    /// 真值取自 OpenEXR 3.1.5 `dwaLookups.h::dwaCompressorToLinear`
    /// （65536 项预生成静态表）。这组断言的价值在于它能捕获
    /// "小样本全绿、大范围语义错"的假绿——修复前的 clamp bug
    /// 在 `0x0E38` 这类 subnormal 与 `0x460B` 这类溢出点上都会露馅。
    const OFFICIAL_LUT_PAIRS: &[(u16, u16)] = &[
        (0x0000, 0x0000), (0x0001, 0x0000), (0x0002, 0x0000), (0x0003, 0x0000),
        (0x0004, 0x0000), (0x0007, 0x0000), (0x000F, 0x0000), (0x001F, 0x0000),
        (0x003F, 0x0000), (0x007F, 0x0000), (0x00FF, 0x0000), (0x01FF, 0x0000),
        (0x03FC, 0x0000), (0x03FD, 0x0000), (0x03FE, 0x0000), (0x03FF, 0x0000),
        (0x0400, 0x0000), (0x0401, 0x0000), (0x0402, 0x0000), (0x0403, 0x0000),
        (0x0404, 0x0000), (0x0407, 0x0000), (0x040F, 0x0000), (0x041F, 0x0000),
        (0x043F, 0x0000), (0x047F, 0x0000), (0x04FF, 0x0000), (0x05FF, 0x0000),
        (0x07FC, 0x0000), (0x07FD, 0x0000), (0x07FE, 0x0000), (0x07FF, 0x0000),
        (0x0800, 0x0000), (0x0801, 0x0000), (0x0802, 0x0000), (0x0803, 0x0000),
        (0x0804, 0x0000), (0x0807, 0x0000), (0x080F, 0x0000), (0x081F, 0x0000),
        (0x083F, 0x0000), (0x087F, 0x0000), (0x08FF, 0x0000), (0x09FF, 0x0000),
        (0x0BFC, 0x0000), (0x0BFD, 0x0000), (0x0BFE, 0x0000), (0x0BFF, 0x0000),
        (0x0C00, 0x0000), (0x0C01, 0x0000), (0x0C02, 0x0000), (0x0C03, 0x0000),
        (0x0C04, 0x0000), (0x0C07, 0x0000), (0x0C0F, 0x0000), (0x0C1F, 0x0000),
        (0x0C3F, 0x0000), (0x0C7F, 0x0000), (0x0CFF, 0x0000), (0x0DFF, 0x0000),
        (0x0FFC, 0x0001), (0x0FFD, 0x0001), (0x0FFE, 0x0001), (0x0FFF, 0x0001),
        (0x1000, 0x0001), (0x1001, 0x0001), (0x1002, 0x0001), (0x1003, 0x0001),
        (0x1004, 0x0001), (0x1007, 0x0001), (0x100F, 0x0001), (0x101F, 0x0001),
        (0x103F, 0x0001), (0x107F, 0x0001), (0x10FF, 0x0001), (0x11FF, 0x0002),
        (0x13FC, 0x0004), (0x13FD, 0x0004), (0x13FE, 0x0004), (0x13FF, 0x0004),
        (0x1400, 0x0004), (0x1401, 0x0004), (0x1402, 0x0004), (0x1403, 0x0004),
        (0x1404, 0x0004), (0x1407, 0x0004), (0x140F, 0x0004), (0x141F, 0x0004),
        (0x143F, 0x0005), (0x147F, 0x0005), (0x14FF, 0x0007), (0x15FF, 0x000A),
        (0x17FC, 0x0012), (0x17FD, 0x0012), (0x17FE, 0x0012), (0x17FF, 0x0012),
        (0x1800, 0x0012), (0x1801, 0x0012), (0x1802, 0x0012), (0x1803, 0x0012),
        (0x1804, 0x0013), (0x1807, 0x0013), (0x180F, 0x0013), (0x181F, 0x0014),
        (0x183F, 0x0015), (0x187F, 0x0018), (0x18FF, 0x001E), (0x19FF, 0x002D),
        (0x1BFC, 0x0054), (0x1BFD, 0x0054), (0x1BFE, 0x0054), (0x1BFF, 0x0054),
        (0x1C00, 0x0054), (0x1C01, 0x0055), (0x1C02, 0x0055), (0x1C03, 0x0055),
        (0x1C04, 0x0055), (0x1C07, 0x0056), (0x1C0F, 0x0057), (0x1C1F, 0x005A),
        (0x1C3F, 0x0060), (0x1C7F, 0x006D), (0x1CFF, 0x008A), (0x1DFF, 0x00CE),
        (0x1FFC, 0x0182), (0x1FFD, 0x0183), (0x1FFE, 0x0183), (0x1FFF, 0x0184),
        (0x2000, 0x0184), (0x2001, 0x0185), (0x2002, 0x0186), (0x2003, 0x0187),
        (0x2004, 0x0187), (0x2007, 0x018A), (0x200F, 0x0191), (0x201F, 0x019E),
        (0x203F, 0x01BA), (0x207F, 0x01F6), (0x20FF, 0x0279), (0x21FF, 0x03B1),
        (0x23FC, 0x06EF), (0x23FD, 0x06F1), (0x23FE, 0x06F3), (0x23FF, 0x06F5),
        (0x2400, 0x06F7), (0x2401, 0x06FB), (0x2402, 0x06FF), (0x2403, 0x0702),
        (0x2404, 0x0706), (0x2407, 0x0712), (0x240F, 0x0731), (0x241F, 0x0770),
        (0x243F, 0x07F1), (0x247F, 0x0881), (0x24FF, 0x09AE), (0x25FF, 0x0C3E),
        (0x27FC, 0x0FF7), (0x27FD, 0x0FF9), (0x27FE, 0x0FFC), (0x27FF, 0x0FFE),
        (0x2800, 0x1000), (0x2801, 0x1002), (0x2802, 0x1004), (0x2803, 0x1007),
        (0x2804, 0x1009), (0x2807, 0x100F), (0x280F, 0x1021), (0x281F, 0x1045),
        (0x283F, 0x1090), (0x287F, 0x112C), (0x28FF, 0x1286), (0x29FF, 0x14E0),
        (0x2BFC, 0x1893), (0x2BFD, 0x1894), (0x2BFE, 0x1896), (0x2BFF, 0x1897),
        (0x2C00, 0x1898), (0x2C01, 0x189B), (0x2C02, 0x189D), (0x2C03, 0x18A0),
        (0x2C04, 0x18A2), (0x2C07, 0x18AA), (0x2C0F, 0x18BF), (0x2C1F, 0x18E8),
        (0x2C3F, 0x193D), (0x2C7F, 0x19F1), (0x2CFF, 0x1B7E), (0x2DFF, 0x1D99),
        (0x2FFC, 0x2141), (0x2FFD, 0x2143), (0x2FFE, 0x2144), (0x2FFF, 0x2146),
        (0x3000, 0x2147), (0x3001, 0x214A), (0x3002, 0x214D), (0x3003, 0x2150),
        (0x3004, 0x2153), (0x3007, 0x215C), (0x300F, 0x2173), (0x301F, 0x21A3),
        (0x303F, 0x2205), (0x307F, 0x22D4), (0x30FF, 0x244E), (0x31FF, 0x266E),
        (0x33FC, 0x2A09), (0x33FD, 0x2A0B), (0x33FE, 0x2A0D), (0x33FF, 0x2A0E),
        (0x3400, 0x2A10), (0x3401, 0x2A13), (0x3402, 0x2A17), (0x3403, 0x2A1A),
        (0x3404, 0x2A1D), (0x3407, 0x2A28), (0x340F, 0x2A43), (0x341F, 0x2A79),
        (0x343F, 0x2AEA), (0x347F, 0x2BD7), (0x34FF, 0x2CF2), (0x35FF, 0x2F63),
        (0x37FC, 0x32EF), (0x37FD, 0x32F1), (0x37FE, 0x32F3), (0x37FF, 0x32F5),
        (0x3800, 0x32F7), (0x3801, 0x32FB), (0x3802, 0x32FF), (0x3803, 0x3302),
        (0x3804, 0x3306), (0x3807, 0x3312), (0x380F, 0x3331), (0x381F, 0x3370),
        (0x383F, 0x33F1), (0x387F, 0x3481), (0x38FF, 0x35AE), (0x39FF, 0x383E),
        (0x3BFC, 0x3BF7), (0x3BFD, 0x3BF9), (0x3BFE, 0x3BFC), (0x3BFF, 0x3BFE),
        (0x3C00, 0x3C00), (0x3C01, 0x3C02), (0x3C02, 0x3C04), (0x3C03, 0x3C07),
        (0x3C04, 0x3C09), (0x3C07, 0x3C10), (0x3C0F, 0x3C22), (0x3C1F, 0x3C47),
        (0x3C3F, 0x3C94), (0x3C7F, 0x3D41), (0x3CFF, 0x3EEB), (0x3DFF, 0x41FF),
        (0x3FFC, 0x4879), (0x3FFD, 0x487C), (0x3FFE, 0x487E), (0x3FFF, 0x4881),
        (0x4000, 0x4883), (0x4001, 0x4888), (0x4002, 0x488D), (0x4003, 0x4892),
        (0x4004, 0x4897), (0x4007, 0x48A6), (0x400F, 0x48D0), (0x401F, 0x4928),
        (0x403F, 0x49EA), (0x407F, 0x4BCA), (0x40FF, 0x4EC0), (0x41FF, 0x5512),
        (0x43FC, 0x61A5), (0x43FD, 0x61AB), (0x43FE, 0x61B2), (0x43FF, 0x61B8),
        (0x4400, 0x61BE), (0x4401, 0x61CB), (0x4402, 0x61D8), (0x4403, 0x61E5),
        (0x4404, 0x61F2), (0x4407, 0x6219), (0x440F, 0x6288), (0x441F, 0x637F),
        (0x443F, 0x64EF), (0x447F, 0x6847), (0x44FF, 0x6E6C), (0x45FF, 0x7B3F),
        (0x47FC, 0x7C00), (0x47FD, 0x7C00), (0x47FE, 0x7C00), (0x47FF, 0x7C00),
        (0x4800, 0x7C00), (0x4801, 0x7C00), (0x4802, 0x7C00), (0x4803, 0x7C00),
        (0x4804, 0x7C00), (0x4807, 0x7C00), (0x480F, 0x7C00), (0x481F, 0x7C00),
        (0x483F, 0x7C00), (0x487F, 0x7C00), (0x48FF, 0x7C00), (0x49FF, 0x7C00),
        (0x4BFC, 0x7C00), (0x4BFD, 0x7C00), (0x4BFE, 0x7C00), (0x4BFF, 0x7C00),
        (0x4C00, 0x7C00), (0x4C01, 0x7C00), (0x4C02, 0x7C00), (0x4C03, 0x7C00),
        (0x4C04, 0x7C00), (0x4C07, 0x7C00), (0x4C0F, 0x7C00), (0x4C1F, 0x7C00),
        (0x4C3F, 0x7C00), (0x4C7F, 0x7C00), (0x4CFF, 0x7C00), (0x4DFF, 0x7C00),
        (0x4FFC, 0x7C00), (0x4FFD, 0x7C00), (0x4FFE, 0x7C00), (0x4FFF, 0x7C00),
        (0x5000, 0x7C00), (0x5001, 0x7C00), (0x5002, 0x7C00), (0x5003, 0x7C00),
        (0x5004, 0x7C00), (0x5007, 0x7C00), (0x500F, 0x7C00), (0x501F, 0x7C00),
        (0x503F, 0x7C00), (0x507F, 0x7C00), (0x50FF, 0x7C00), (0x51FF, 0x7C00),
        (0x53FC, 0x7C00), (0x53FD, 0x7C00), (0x53FE, 0x7C00), (0x53FF, 0x7C00),
        (0x5400, 0x7C00), (0x5401, 0x7C00), (0x5402, 0x7C00), (0x5403, 0x7C00),
        (0x5404, 0x7C00), (0x5407, 0x7C00), (0x540F, 0x7C00), (0x541F, 0x7C00),
        (0x543F, 0x7C00), (0x547F, 0x7C00), (0x54FF, 0x7C00), (0x55FF, 0x7C00),
        (0x57FC, 0x7C00), (0x57FD, 0x7C00), (0x57FE, 0x7C00), (0x57FF, 0x7C00),
        (0x5800, 0x7C00), (0x5801, 0x7C00), (0x5802, 0x7C00), (0x5803, 0x7C00),
        (0x5804, 0x7C00), (0x5807, 0x7C00), (0x580F, 0x7C00), (0x581F, 0x7C00),
        (0x583F, 0x7C00), (0x587F, 0x7C00), (0x58FF, 0x7C00), (0x59FF, 0x7C00),
        (0x5BFC, 0x7C00), (0x5BFD, 0x7C00), (0x5BFE, 0x7C00), (0x5BFF, 0x7C00),
        (0x5C00, 0x7C00), (0x5C01, 0x7C00), (0x5C02, 0x7C00), (0x5C03, 0x7C00),
        (0x5C04, 0x7C00), (0x5C07, 0x7C00), (0x5C0F, 0x7C00), (0x5C1F, 0x7C00),
        (0x5C3F, 0x7C00), (0x5C7F, 0x7C00), (0x5CFF, 0x7C00), (0x5DFF, 0x7C00),
        (0x5FFC, 0x7C00), (0x5FFD, 0x7C00), (0x5FFE, 0x7C00), (0x5FFF, 0x7C00),
        (0x6000, 0x7C00), (0x6001, 0x7C00), (0x6002, 0x7C00), (0x6003, 0x7C00),
        (0x6004, 0x7C00), (0x6007, 0x7C00), (0x600F, 0x7C00), (0x601F, 0x7C00),
        (0x603F, 0x7C00), (0x607F, 0x7C00), (0x60FF, 0x7C00), (0x61FF, 0x7C00),
        (0x63FC, 0x7C00), (0x63FD, 0x7C00), (0x63FE, 0x7C00), (0x63FF, 0x7C00),
        (0x6400, 0x7C00), (0x6401, 0x7C00), (0x6402, 0x7C00), (0x6403, 0x7C00),
        (0x6404, 0x7C00), (0x6407, 0x7C00), (0x640F, 0x7C00), (0x641F, 0x7C00),
        (0x643F, 0x7C00), (0x647F, 0x7C00), (0x64FF, 0x7C00), (0x65FF, 0x7C00),
        (0x67FC, 0x7C00), (0x67FD, 0x7C00), (0x67FE, 0x7C00), (0x67FF, 0x7C00),
        (0x6800, 0x7C00), (0x6801, 0x7C00), (0x6802, 0x7C00), (0x6803, 0x7C00),
        (0x6804, 0x7C00), (0x6807, 0x7C00), (0x680F, 0x7C00), (0x681F, 0x7C00),
        (0x683F, 0x7C00), (0x687F, 0x7C00), (0x68FF, 0x7C00), (0x69FF, 0x7C00),
        (0x6BFC, 0x7C00), (0x6BFD, 0x7C00), (0x6BFE, 0x7C00), (0x6BFF, 0x7C00),
        (0x6C00, 0x7C00), (0x6C01, 0x7C00), (0x6C02, 0x7C00), (0x6C03, 0x7C00),
        (0x6C04, 0x7C00), (0x6C07, 0x7C00), (0x6C0F, 0x7C00), (0x6C1F, 0x7C00),
        (0x6C3F, 0x7C00), (0x6C7F, 0x7C00), (0x6CFF, 0x7C00), (0x6DFF, 0x7C00),
        (0x6FFC, 0x7C00), (0x6FFD, 0x7C00), (0x6FFE, 0x7C00), (0x6FFF, 0x7C00),
        (0x7000, 0x7C00), (0x7001, 0x7C00), (0x7002, 0x7C00), (0x7003, 0x7C00),
        (0x7004, 0x7C00), (0x7007, 0x7C00), (0x700F, 0x7C00), (0x701F, 0x7C00),
        (0x703F, 0x7C00), (0x707F, 0x7C00), (0x70FF, 0x7C00), (0x71FF, 0x7C00),
        (0x73FC, 0x7C00), (0x73FD, 0x7C00), (0x73FE, 0x7C00), (0x73FF, 0x7C00),
        (0x7400, 0x7C00), (0x7401, 0x7C00), (0x7402, 0x7C00), (0x7403, 0x7C00),
        (0x7404, 0x7C00), (0x7407, 0x7C00), (0x740F, 0x7C00), (0x741F, 0x7C00),
        (0x743F, 0x7C00), (0x747F, 0x7C00), (0x74FF, 0x7C00), (0x75FF, 0x7C00),
        (0x77FC, 0x7C00), (0x77FD, 0x7C00), (0x77FE, 0x7C00), (0x77FF, 0x7C00),
        (0x7800, 0x7C00), (0x7801, 0x7C00), (0x7802, 0x7C00), (0x7803, 0x7C00),
        (0x7804, 0x7C00), (0x7807, 0x7C00), (0x780F, 0x7C00), (0x781F, 0x7C00),
        (0x783F, 0x7C00), (0x787F, 0x7C00), (0x78FF, 0x7C00), (0x79FF, 0x7C00),
        (0x7BFC, 0x7C00), (0x7BFD, 0x7C00), (0x7BFE, 0x7C00), (0x7BFF, 0x7C00),
        (0x7C00, 0x0000), (0x7C01, 0x0000), (0x7C02, 0x0000), (0x7C03, 0x0000),
        (0x7C04, 0x0000), (0x7C07, 0x0000), (0x7C0F, 0x0000), (0x7C1F, 0x0000),
        (0x7C3F, 0x0000), (0x7C7F, 0x0000), (0x7CFF, 0x0000), (0x7DFF, 0x0000),
        (0x7FFC, 0x0000), (0x7FFD, 0x0000), (0x7FFE, 0x0000), (0x7FFF, 0x0000),
    ];

    #[test]
    fn dwa_to_linear_matches_official_512_point_probe() {
        let mut bad: Vec<String> = Vec::new();
        for &(idx, expect) in OFFICIAL_LUT_PAIRS {
            let got = dwa_to_linear(idx);
            if got != expect {
                bad.push(format!("0x{idx:04X}: got 0x{got:04X}, want 0x{expect:04X}"));
            }
        }
        assert!(
            bad.is_empty(),
            "与官方 dwaCompressorToLinear 不一致 {} 项:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    #[test]
    fn dwa_to_linear_overflow_yields_inf_not_clamped_finite() {
        // 回归防线：修复前此处返回 0x7BFF(65504)，官方为 0x7C00(+inf)。
        // 阈值样本：half 6.04296875（0x460B）起 logBase^(x-1) 溢出 half。
        for bits in [0x460Bu16, 0x4BC8, 0x5000, 0x5BFF] {
            assert_eq!(
                dwa_to_linear(bits),
                0x7C00,
                "0x{bits:04X} 溢出须为 +inf 而非 clamp 后的 65504"
            );
        }
        for bits in [0xC60Bu16, 0xCCC8, 0xD000] {
            assert_eq!(
                dwa_to_linear(bits),
                0xFC00,
                "0x{bits:04X} 负向溢出须为 -inf"
            );
        }
    }

    #[test]
    fn dwa_to_linear_is_idempotent_on_unit_range() {
        // |x| ≤ 1 时 toLinear(x) = sign · |x|^2.2，单调且与恒等函数在 0/1 交点。
        assert_eq!(dwa_to_linear(f32_to_f16(1.0)), f32_to_f16(1.0), "1.0 是不动点");
        // 0.5^2.2 ≈ 0.2176。
        let got = f16_to_f32(dwa_to_linear(f32_to_f16(0.5)));
        assert!(
            (got - 0.5f32.powf(2.2)).abs() < 1e-3,
            "0.5 的 toLinear 应约 0.2176，实得 {got}"
        );
    }

    #[test]
    fn from_half_zig_zag_is_a_permutation() {
        // 用**可被 half 精确表示**的整数值（0..64），避免 denormal 碰撞。
        let mut src = [0u16; DCT_COEFFS];
        for (i, s) in src.iter_mut().enumerate() {
            *s = f32_to_f16(i as f32);
        }
        let mut dst = [0.0f32; DCT_COEFFS];
        from_half_zig_zag(&src, &mut dst);
        // 每个源值恰好出现一次（置换 ⇒ 双射）。
        let mut sorted: Vec<f32> = dst.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let want: Vec<f32> = (0..DCT_COEFFS).map(|v| v as f32).collect();
        assert_eq!(sorted, want, "反 zigzag 必须是置换");
        // MAP[0] = 0 ⇒ dst[0] = src[0] = 0。
        assert_eq!(dst[0], 0.0);
        // MAP 是常量表，验证其本身为 0..63 的一个排列（防呆断言）。
        const MAP: [usize; DCT_COEFFS] = [
            0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
            18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21,
            34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
        ];
        let mut seen = [false; DCT_COEFFS];
        for &m in MAP.iter() {
            assert!(m < DCT_COEFFS, "MAP 越界: {m}");
            assert!(!seen[m], "MAP 重复: {m}");
            seen[m] = true;
        }
    }

    /// W13 互操作对拍：**DC-only 解码链路端到端真值表**（10 项）。
    ///
    /// # 这是外部真值，不是我们自己算的
    ///
    /// **来源**：OpenEXR 3.1.5 官方 `ImfDwaCompressor` 写出的 `.exr`，
    /// DC 区内容由**本仓** `exr_interop dcdump` 取出，像素值由**官方**
    /// `InputFile` 读回逐位打印；两者再经**官方参考实现 `refdc.cpp`**
    /// （`tests/interop/refdc.cpp`，复刻 `dctInverse8x8DcOnly` + `toLinear`）
    /// 独立计算，三方逐位一致后才落盘为本表。
    ///
    /// **提取方式**（可复现，不依赖任何临时目录）：
    /// ```text
    /// oexr_interop gen q.exr 8 8 dwaa-half-scan-1-c<value> 1   # 官方写
    /// exr_interop  dcdump q.exr dc.txt                          # 取 DC 区 u16
    /// oexr_interop dump q.exr q.txt                # 官方读回，f32 位模式
    /// refdc <dc_hex>                                    # 官方参考实现交叉验证
    /// ```
    ///
    /// 每项为 `(DC 区 u16 的 half 位模式, 官方解出的像素 f32 位模式)`。
    ///
    /// # 为什么以 DC 为输入维度，而不是源像素常量
    ///
    /// 初版曾误以为链路是 `c → DCT → DC=8c`，实测**不成立**：官方在 DCT
    /// **之前**先施加 `toLinear` 的逆（非线性化），故 DC 存的是 `8·c^(1/2.2)`。
    /// 例如 `c=0.5` 实际 DC = `0x45D7`（half 5.839844），而非 `4.0`。
    /// 解码侧公式为 **`pixel = toLinear((half)(DC × 1/8))`**，本表即按此
    /// 口径固化 —— 钉的正是解码器要实现的那一步。
    ///
    /// # 这组真值钉住的是什么
    ///
    /// 解码链路 `DC → ×1/8 增益 → half 舍入 → toLinear 查表` 上任何一环
    /// 写错（漏 1/8 增益、增益常量取"更准"的 1/√8、`toLinear` 公式漂移、
    /// half 舍入模式错）都会让本表大面积变红。
    /// 已有的 `dwa_to_linear_matches_official_512_point_probe` 只钉住链路末端
    /// （65536 项 LUT 抽样），本表钉住的是**含 DC 逆变换与 half 往返的整链**。
    ///
    /// # 实测结论（据此定阈值，非猜测）
    ///
    /// 10 格中`DC=0x4800`（half 8.0）解回 `1.0` 精确，其余 9 格因
    /// `toLinear` 后的 half 量化而有 1~2 ULP 偏差。这些偏差是**官方自带**的
    /// 量化噪声（官方自己重读自己写的文件也是这个值），不是我们的缺陷。
    const OFFICIAL_DC_DECODE_PAIRS: &[(u16, u32)] = &[
        // DC=0.1210938 → 9.91225243e-05   refdc: pixBits=0x23C0 lut=0x067F
        (0x2FC0, 0x38CF_E000),
        // DC=0.3461914 → 0.000999450684   refdc: pixBits=0x298A lut=0x1418
        (0x358A, 0x3A83_0000),
        // DC=0.9863281 → 0.0100021362     refdc: pixBits=0x2FE4 lut=0x211F
        (0x3BE4, 0x3C23_E000),
        // DC=2.808594  → 0.0999755859     refdc: pixBits=0x359E lut=0x2E66
        (0x419E, 0x3DCC_C000),
        // DC=5.839844  → 0.500488281      refdc: pixBits=0x39D7 lut=0x3801
        // （W13 抓到的第三个真实 bug 就是漏了这个 ×1/8：漏了会解成 42080）
        (0x45D7, 0x3F00_2000),
        // DC=8.0       → 1.0（精确，1/8 增益与toLinear 在此双双精确）
        (0x4800, 0x3F80_0000),
        // DC=10.523438 → 2.00195312       refdc: pixBits=0x3D43 lut=0x4001
        (0x4943, 0x4000_2000),
        // DC=13.039063 → 3.99804688       refdc: pixBits=0x3E85 lut=0x43FF
        (0x4A85, 0x407F_E000),
        // DC=18.078125 → 15.984375        refdc: pixBits=0x4085 lut=0x4BFE
        (0x4C85, 0x417F_C000),
        // DC=33.21875  → 1028.0           refdc: pixBits=0x4427 lut=0x6404
        (0x5027, 0x4480_8000),
    ];

    #[test]
    fn dc_only_decode_chain_matches_official_10_point_truth() {
        // 逐项把DC **真的走一遍本仓解码链路**（×1/8 逆变换 + half + toLinear），
        // 而不是在测试里重算期望值 —— 否则实现与测试同源错误会互相掩护。
        let mut bad: Vec<String> = Vec::new();
        for &(dc_bits, official_bits) in OFFICIAL_DC_DECODE_PAIRS {
            let dc = f16_to_f32(dc_bits);
            let mut block = [0.0f32; DCT_COEFFS];
            block[0] = dc;
            dct_inverse_dc_only(&mut block);
            let ours = f16_to_f32(dwa_to_linear(f32_to_f16(block[0])));
            let official = f32::from_bits(official_bits);
            if ours.to_bits() != official.to_bits() {
                bad.push(format!(
                    "DC=0x{dc_bits:04X} ({dc}): got {ours} (0x{:08X}), \
                     official {official} (0x{official_bits:08X})",
                    ours.to_bits()
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "DC-only 解码链与官方不一致 {} 项:\n{}",
            bad.len(),
            bad.join("\n")
        );

        // 反向自证：把某个真值破坏掉（DC=0x45D7 的官方值 0.500488281 篡改成
        // 4.00390625，即恰好是"漏乘 1/8 增益"那个真 bug 的错误值），
        // 用同一段判据跑一遍必须被抓出来 —— 确认判据**会红**。
        let tampered = 4.00390625f32; // = 0.500488281 × 8
        let dc = f16_to_f32(0x45D7);
        let mut block = [0.0f32; DCT_COEFFS];
        block[0] = dc;
        dct_inverse_dc_only(&mut block);
        let ours = f16_to_f32(dwa_to_linear(f32_to_f16(block[0])));
        assert_eq!(ours, 0.500488281, "基准值自身不符，真值表已失效");
        assert!(
            ours.to_bits() != tampered.to_bits(),
            "反向自证失效：篡改值竟与本仓输出一致（判据抓不住 bug）"
        );
        // 漏乘 1/8 会被 toLinear 放大到 42080 —— 与第三个真实 bug 的症状一致。
        let mut no_gain = [0.0f32; DCT_COEFFS];
        no_gain[0] = dc;
        let wrong = f16_to_f32(dwa_to_linear(f32_to_f16(no_gain[0])));
        assert!(
            (wrong - 42080.0).abs() / 42080.0 < 1e-3,
            "漏乘 1/8 应爆到 ~42080，实得 {wrong} —— 回归症状已变，需重估"
        );
    }

    #[test]
    fn dct_inverse_dc_only_fills_every_coefficient() {
        // W13 互通对拍修正：官方 `dctInverse8x8DcOnly` 是
        // `val = data[0] * 3.535536e-01f * 3.535536e-01f`（= ×1/8）后铺满，
        // 本实现原先漏了增益，两侧错误抵消故内部往返全绿、跨实现才露馅。
        // 现断言的是**带增益**的官方语义。
        let mut d = [0.0f32; DCT_COEFFS];
        d[0] = 3.25;
        dct_inverse_dc_only(&mut d);
        let expect: f32 = 3.25 * DC_ONLY_GAIN * DC_ONLY_GAIN;
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                expect.to_bits(),
                "DC-only 须按 data[0]×0.3535536² 铺满整块（第 {i} 项）"
            );
        }
        // 【实测】两条路径的 DC 基幅度相对差 = 1.027e-06（官方两处字面量
        // 不同：`3.535536e-01f` vs `A = 0.5·cos(π/4)`，末位差 1 ULP 级）。
        // 阈值取 **1e-5 ≈ 实测 9.7 倍**，且要判红只需偏离达 1e-5——
        // 比"能过就放宽"严一个量级。
        const DC_GAIN_REL_TOL: f32 = 1e-5;
        let mut full = [0.0f32; DCT_COEFFS];
        full[0] = 3.25;
        dct_inverse_8x8(&mut full);
        let rel = (full[0] - expect).abs() / expect.abs().max(1e-30);
        assert!(
            rel < DC_GAIN_REL_TOL,
            "DC-only 快路径({expect})与完整 IDCT({})的 DC 基幅度应数值相近，\
             实际相对差 {rel}（实测 1.027e-06）",
            full[0]
        );
    }

    #[test]
    fn dct_inverse_8x8_of_pure_dc_is_a_constant_one_eighth() {
        // IDCT 的 DC 基向量：全块同值。2D IDCT = 1D 变换做两遍，纯 DC=1 经第一遍
        // 得 A=0.5cos(π/4)，第二遍再乘一次 A ⇒ 常量 A² = 1/8。
        // （这与 `dct_inverse_dc_only` 不同：后者按 OpenEXR 语义把 DC **直接**铺满，
        //  不做 1/8 缩放；两条路径的差异是真实格式的既定行为。）
        let mut d = [0.0f32; DCT_COEFFS];
        d[0] = 1.0;
        dct_inverse_8x8(&mut d);
        let first = d[0];
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(v.to_bits(), first.to_bits(), "纯 DC 的 IDCT 须为常量块（第 {i} 项）");
        }
        assert!(
            (first - 0.125).abs() < 1e-6,
            "2D 纯 DC 常量应为 A² = 1/8，实得 {first}"
        );
    }

    #[test]
    fn dct_inverse_8x8_is_deterministic_and_linear() {
        // ① 确定性：同输入两遍结果逐位相同。
        let mut a = [0.0f32; DCT_COEFFS];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32) * 0.125 - 3.0;
        }
        let mut a1 = a;
        let mut a2 = a;
        dct_inverse_8x8(&mut a1);
        dct_inverse_8x8(&mut a2);
        for i in 0..DCT_COEFFS {
            assert_eq!(a1[i].to_bits(), a2[i].to_bits(), "IDCT 须确定性（第 {i} 项）");
        }
        // ② 线性：IDCT(2x) == 2·IDCT(x)。
        let mut b1 = a;
        let mut b2 = a;
        for v in b1.iter_mut() {
            *v *= 2.0;
        }
        for v in b2.iter_mut() {
            *v *= 2.0;
        }
        dct_inverse_8x8(&mut b1);
        dct_inverse_8x8(&mut b2);
        for i in 0..DCT_COEFFS {
            let lhs = b1[i];
            let rhs = 2.0 * a1[i];
            assert!(
                (lhs - rhs).abs() <= 1e-4 * rhs.abs().max(1.0),
                "IDCT 线性在第 {i} 项被破坏：{lhs} vs {rhs}"
            );
        }
    }

    // =======================================================================
    //  W12-B新增回归测试（8 条交付要求逐条钉死）
    // =======================================================================

    /// 三通道 CSC 组往返：R/G/B 全部非常量（含 AC）时逐通道有界。
    ///
    /// 覆盖「3 通道 CSC 往返」交付项。
    #[test]
    fn three_channel_csc_round_trip_stays_within_relative_bound() {
        let (width, lines) = (8usize, 8usize);
        let mk = |k: usize| -> Vec<f32> {
            (0..(width * lines))
                .map(|i| (((i + k * 5) % 16) as f32) / 16.0)
                .collect()
        };
        let srcs = [mk(0), mk(1), mk(2)];
        let mut raw: Vec<u8> = Vec::new();
        for y in 0..lines {
            for plane in srcs.iter() {
                for x in 0..width {
                    raw.extend_from_slice(&f32_to_f16(plane[y * width + x]).to_le_bytes());
                }
            }
        }
        let chans = chans(&[
            ("R", PixelType::Half, false),
            ("G", PixelType::Half, false),
            ("B", PixelType::Half, false),
        ]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let got = split_planes(&out, &chans, width, lines);
        for (ci, src) in srcs.iter().enumerate() {
            for (i, &w) in src.iter().enumerate() {
                // 幅值相关容差：与 exr_write 的 `assert_dwa_round_trip` 同一模型。
                let tol = 0.05 * w.abs() + 0.05;
                assert!(
                    (got[ci][i] - w).abs() <= tol,
                    "通道 {ci} 像素 {i}: {} vs {w}（err {:.5} > {tol:.5}）",
                    got[ci][i],
                    (got[ci][i] - w).abs()
                );
            }
        }
    }

    /// **反向测试**：构造「块级常量判定」与「通道级 AC/DC 判定」分叉的块。
    ///
    /// 背景：解码端曾用**块级** `block_is_constant` 决定是否整块逆 CSC，
    /// 而 AC/DC 判定是**逐通道**的。三通道里若仅 comp1 有 AC、comp0/comp2 为
    /// DC-only，块级判定会误判为「非常量」——反之若三通道皆 DC-only，块级
    /// 判定为「常量」而只逆 CSC 的 `[0]` 元素，**漏掉 63/64 个像素**。
    ///
    /// 本测试用「三通道全 DC-only 的彩色常量块」钉死修复后行为：
    /// 每个元素都必须被逆 CSC，块内64 个像素须**彼此相等**且接近原值。
    #[test]
    fn zip_pack_is_the_exact_inverse_of_zip_decompress() {
        //回归测试：官方 `ImfZip.cpp:88` 的预测器循环里 `p = t[0]` 在写回**之前**
        // 执行，故 `p` 恒为**原始**前字节。本实现曾用已被覆盖的
        // `reordered[i-1]` 当 `p`，导致 `zip_pack` 与 `zip_decompress` **不互逆**：
        // 长度为 1 时恰好成立（只有一个字节，无预测步），≥2 个 u16 起整体错位。
        //
        // 覆盖 n = 1（回归临界点）、n = 3（3 通道 CSC 单块，最小真实场景）、
        // n = 17（奇数长度 + 非单调值，逼出交织的奇偶边界处理）。
        let cases: Vec<Vec<u16>> = vec![
            vec![0x4404],
            vec![0x4404, 0x3d00, 0x2f80],
            vec![0xe598, 0x9220, 0x1fd8],
            (0..17u16).map(|i| 1000u16.wrapping_add(i * 7)).collect(),
        ];
        for c in cases {
            let packed = zip_pack(&c);
            let back = zip_decompress(&packed, c.len() * 2)
                .unwrap_or_else(|e| panic!("zip_decompress 失败（n={}）：{e}", c.len()));
            let got: Vec<u16> = back
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            assert_eq!(got, c, "zip 往返不闭合（n={}）", c.len());
        }
    }

    #[test]
    fn reverse_test_all_dc_only_csc_block_inverts_every_element() {
        let (width, lines) = (8usize, 8usize);
        // 三个**不同**的常量 → 前向 CSC 后 Y/Cb/Cr 各不相同，
        // 若只逆 `[0]` 则元素 1..63 会残留前向 CSC 的 Y/Cb/Cr 比例。
        let consts = [0.25f32, 0.5, 0.75];
        let mut raw: Vec<u8> = Vec::new();
        for _y in 0..lines {
            for v in consts.iter() {
                for _x in 0..width {
                    raw.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
                }
            }
        }
        let chans = chans(&[
            ("R", PixelType::Half, false),
            ("G", PixelType::Half, false),
            ("B", PixelType::Half, false),
        ]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let got = split_planes(&out, &chans, width, lines);
        for (ci, want) in consts.iter().enumerate() {
            let plane = &got[ci];
            // 块内 64 个像素必须**全部**接近原常量 —— 钉死「整块逆 CSC」。
            for (i, &g) in plane.iter().enumerate() {
                assert!(
                    (g - want).abs() <= 0.05,
                    "通道 {ci} 像素 {i}: {g} 期望 {want}（DC-only 块须整块逆 CSC）"
                );
            }
            // 并逐位确认块内一致性（不只是首元素）。
            for (i, &g) in plane.iter().enumerate().skip(1) {
                assert_eq!(
                    g.to_bits(),
                    plane[0].to_bits(),
                    "通道 {ci} 像素 {i} 与像素 0 不一致 ⇒ 逆 CSC 未覆盖整块"
                );
            }
        }
    }

    /// **反向测试（分叉的另一侧）**：块内**逐通道** AC/DC 判定分叉。
    ///
    /// 注意 CSC 是**逐像素混合**三通道的：即便输入的 comp0/comp2 是常量，
    /// 只要 comp1 有梯度，前向 CSC 后的 Y 也会随 comp1 变化 ⇒ comp0 同样带 AC。
    /// 故要构造真正的分叉，必须让**前向 CSC 之后**的某一通道 AC 全零 ——
    /// 这里取「comp1 = comp0 = comp2」（灰度 ⇒ Cb/Cr 恒 0，只有 Y 带 AC），
    /// 此时 Y 有 AC 而 Cb/Cr 是 DC-only，块级与通道级判定必然分叉。
    ///
    /// 修复后无条件整块逆 CSC，两条路径必须**收敛到同一结果**。
    #[test]
    fn reverse_test_mixed_dc_and_ac_across_csc_channels_agree() {
        let (width, lines) = (8usize, 8usize);
        // 灰度渐变：R=G=B=g(x)。前向 CSC 后 Cb=Cr=0（DC-only），Y=g(x)（带 AC）。
        let mut raw: Vec<u8> = Vec::new();
        for _y in 0..lines {
            for _plane in 0..3 {
                for x in 0..width {
                    let v = (x as f32) / 16.0;
                    raw.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
        }
        let chans = chans(&[
            ("R", PixelType::Half, false),
            ("G", PixelType::Half, false),
            ("B", PixelType::Half, false),
        ]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let got = split_planes(&out, &chans, width, lines);
        // 三通道须**各自**还原为同一灰度渐变（逆 CSC 正确性 + 块内一致性）。
        for ci in 0..3 {
            for (i, &w) in got[ci].iter().enumerate() {
                let want = (i % width) as f32 / 16.0;
                let tol = 0.05 * want.abs() + 0.05;
                assert!(
                    (w - want).abs() <= tol,
                    "通道 {ci} 像素 {i}: {w} 期望 {want}（err {:.5}）",
                    (w - want).abs()
                );
            }
        }
        // 块内逐列一致性：同一 x 的三通道必须相等（灰色 ⇒ 无色度残留）。
        for y in 0..lines {
            for x in 0..width {
                let i = y * width + x;
                assert_eq!(
                    got[0][i].to_bits(),
                    got[1][i].to_bits(),
                    "像素 ({x},{y}) 的 R/G 不等 ⇒ 逆 CSC 未整块生效"
                );
                assert_eq!(
                    got[1][i].to_bits(),
                    got[2][i].to_bits(),
                    "像素 ({x},{y}) 的 G/B 不等 ⇒ 逆 CSC 未整块生效"
                );
            }
        }
        // 渐变须被真实还原（不是被误当常量抹平）。
        assert!(
            got[0][7] - got[0][0] > 0.05,
            "AC 渐变被抹平：{} → {}",
            got[0][0],
            got[0][7]
        );
    }

    /// Unknown 通道（float，不命中任何规则）逐位无损。
    #[test]
    fn unknown_channel_float_is_bit_exact_through_dwa() {
        let (width, lines) = (8usize, 8usize);
        let z: Vec<f32> = (0..(width * lines)).map(|i| (i as f32) * 0.125).collect();
        let mut raw: Vec<u8> = Vec::new();
        for i in 0..(width * lines) {
            raw.extend_from_slice(&z[i].to_le_bytes());
        }
        let chans = chans(&[("Z", PixelType::Float, false)]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(out.len(), raw.len(), "Unknown 区字节长度须一致");
        let got = split_planes(&out, &chans, width, lines);
        for (i, &w) in z.iter().enumerate() {
            assert_eq!(
                got[0][i].to_bits(),
                w.to_bits(),
                "Z 像素 {i} 须逐位无损：{} vs {w}",
                got[0][i]
            );
        }
    }

    /// 窄图（`width < DCT_N = 8`）的 RLE 通道打包逐位无损。
    ///
    /// 覆盖「窄图 RLE 打包」交付项：`width=4 < 8` 时 `pack_rle_planes` 的
    /// `stride` 与块几何极易写错，这里用逐位比对钉死。
    #[test]
    fn narrow_image_rle_channel_packs_and_unpacks_bit_exact() {
        for width in [1usize, 3, 4, 7] {
            let lines = 8usize;
            let a: Vec<f32> = (0..(width * lines)).map(|i| (i as f32) * 0.0625).collect();
            let mut raw: Vec<u8> = Vec::new();
            for i in 0..(width * lines) {
                raw.extend_from_slice(&f32_to_f16(a[i]).to_le_bytes());
            }
            let chans = chans(&[("A", PixelType::Half, false)]);
            let ctx = ctx_of(&chans, width, lines);
            let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
            let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
            let got = split_planes(&out, &chans, width, lines);
            assert_eq!(got[0].len(), a.len(), "width={width} 像素数须一致");
            for (i, &w) in a.iter().enumerate() {
                assert_eq!(
                    hbits(f32_to_f16(got[0][i])),
                    hbits(f32_to_f16(w)),
                    "width={width} A 像素 {i} 须逐位无损：{} vs {w}",
                    got[0][i]
                );
            }
        }
    }

    /// 块边界非整数倍：宽高均非 8/32/256 的整数倍时仍须正确裁剪与还原。
    #[test]
    fn partial_edge_blocks_are_clipped_and_restored() {
        for (width, lines) in [(5usize, 3usize), (9, 17), (13, 33), (7, 65)] {
            let vals: Vec<f32> = (0..(width * lines))
                .map(|i| ((i % 8) as f32) / 8.0)
                .collect();
            let raw = raw_half_plane(&vals, width, lines);
            let chans = chans(&[("Y", PixelType::Half, false)]);
            let ctx = ctx_of(&chans, width, lines);
            let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
            let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
            let got = split_planes(&out, &chans, width, lines);
            assert_eq!(got[0].len(), vals.len(), "{width}x{lines} 像素数须一致");
            for (i, &w) in vals.iter().enumerate() {
                let tol = 0.05 * w.abs() + 0.05;
                assert!(
                    (got[0][i] - w).abs() <= tol,
                    "{width}x{lines} 像素 {i}: {} vs {w}",
                    got[0][i]
                );
            }
        }
    }

    /// DWA 有损阈值：误差是**幅值相关**的，且相对误差有稳定上界。
    ///
    /// 钉死 [`exr_write::tests::assert_dwa_round_trip`] 所用容差模型
    /// （`0.05·|v| + 0.05`）的**依据**：DWA 量化在非线性域进行，
    /// `to_linear` 的增益为 `2.2·|y|`（`|y|>1` 段线性增长），
    /// 故误差必然与幅值成正比 —— 固定绝对阈值是错误的误差模型。
    #[test]
    fn dwa_lossy_error_is_bounded_relative_not_absolute() {
        // 大幅值处：绝对误差必然超过 0.05，但相对误差须 < 5%。
        let vals: Vec<f32> = (0..256).map(|i| i as f32 * 4.0 - 3.0).collect();
        let (width, lines) = (16usize, 16usize);
        let raw = raw_half_plane(&vals, width, lines);
        let chans = chans(&[("Y", PixelType::Half, false)]);
        let ctx = ctx_of(&chans, width, lines);
        let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let got = split_planes(&out, &chans, width, lines);
        let mut saw_abs_over_005 = false;
        for (i, &w) in vals.iter().enumerate() {
            let e = (got[0][i] - w).abs();
            if e > 0.05 {
                saw_abs_over_005 = true;
                // 超过固定绝对阈值时，相对误差仍须在 5% 内。
                assert!(
                    e <= 0.05 * w.abs() + 0.05,
                    "像素 {i}: {:?} vs {w}，绝对误差 {e} 超固定阈值但相对误差亦失控",
                    got[0][i].to_bits()
                );
            }
        }
        assert!(
            saw_abs_over_005,
            "本测试须观察到绝对误差 > 0.05 的像素，才能证明绝对阈值是错误模型"
        );
        // 小幅值处须**更准**（`|y| ≤ 1` 段 to_linear 增益封顶在 2.2）。
        // 注意：须真正取 `|y| ≤ 1` 的样本（上表的 vals 是 ±4 斜坡，尾部才进入该段）。
        let small: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0 - 0.5).collect();
        let raw_s = raw_half_plane(&small, width, lines);
        let enc_s = compress_block(&raw_s, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let out_s = decompress_block(&enc_s, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let got_s = split_planes(&out_s, &chans, width, lines);
        for (i, &w) in small.iter().enumerate() {
            assert!(
                (got_s[0][i] - w).abs() <= 0.05,
                "小幅值像素 {i}: {} vs {w} 须落在绝对地板 0.05 内",
                got_s[0][i]
            );
        }
    }

    /// DWAA 与 DWAB 各自独立往返，且两者的块切分数不同。
    #[test]
    fn dwaa_and_dwab_each_round_trip_with_their_own_block_split() {
        let (width, lines) = (8usize, 70usize); // DWAA=32 行/块 → 3 块；DWAB=256 → 1 块
        let vals: Vec<f32> = (0..(width * lines))
            .map(|i| ((i % 8) as f32) / 8.0)
            .collect();
        let raw = raw_half_plane(&vals, width, lines);
        let chans = chans(&[("Y", PixelType::Half, false)]);
        let ctx = ctx_of(&chans, width, lines);
        for name in ["DWAA", "DWAB"] {
            let enc = compress_block(&raw, &ctx).unwrap_or_else(|e| panic!("{e}"));
            let out = decompress_block(&enc, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
            let got = split_planes(&out, &chans, width, lines);
            for (i, &w) in vals.iter().enumerate() {
                let tol = 0.05 * w.abs() + 0.05;
                assert!(
                    (got[0][i] - w).abs() <= tol,
                    "{name} 像素 {i}: {} vs {w}",
                    got[0][i]
                );
            }
        }
    }

    /// 确定性：同输入两次编码逐位相同；且解码亦确定。
    #[test]
    fn encode_then_decode_is_reproducible_bitwise() {
        let (width, lines) = (11usize, 19usize);
        let chans = chans(&[
            ("R", PixelType::Half, false),
            ("G", PixelType::Half, false),
            ("B", PixelType::Half, false),
        ]);
        let ctx = ctx_of(&chans, width, lines);
        let mut raw3: Vec<u8> = Vec::new();
        for y in 0..lines {
            for plane in 0..3 {
                for x in 0..width {
                    let v = ((y * width + x) as f32 + plane as f32 * 3.0) / 32.0;
                    raw3.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
        }
        let e1 = compress_block(&raw3, &ctx).unwrap_or_else(|e| panic!("{e}"));
        let e2 = compress_block(&raw3, &ctx).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(e1, e2, "同输入两次编码须字节相同");
        let d1 = decompress_block(&e1, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        let d2 = decompress_block(&e1, &ctx, lines).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d1, d2, "同块两次解码须字节相同");
    }

    #[test]
    fn csc709_inverse_matches_its_own_definition() {
        let mut a = 0.5f32;
        let mut b = 0.25f32;
        let mut c = 0.125f32;
        csc709_inverse(&mut a, &mut b, &mut c);
        // 按模块内公式重算（s0/s1/s2 为入参快照）。
        let (s0, s1, s2) = (0.5f32, 0.25f32, 0.125f32);
        assert!((a - (s0 + 1.5747 * s2)).abs() < 1e-6);
        assert!((b - (s0 - 0.1873 * s1 - 0.4682 * s2)).abs() < 1e-6);
        assert!((c - (s0 + 1.8556 * s1)).abs() < 1e-6);
    }
}

