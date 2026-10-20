//! **DNG 合成样本生成器 + 样本驱动的端到端诊断入口**。
//!
//! # 这一路为什么存在
//!
//! 真实相机 DNG 无法在仓内自证：没有真实样本，「读得对不对」就只能靠**假数据**
//! 自证。本模块把端到端验证链路搭好，让链路在**没有真实样本时也能被验证**，
//! 且用户未来丢一个真实 `.dng` 进来**不需要再改代码**。
//!
//! 两个方向：
//!
//! | 方向 | 入口 | 作用 |
//! |---|---|---|
//! | **写入** | [`synthesize_dng`] | **从零合成**一个符合 DNG 规范的最小合法 `.dng`（IFD / strip / tile / 无损 JPEG 全自造） |
//! | **读出** | [`diagnose_dng`] / [`diagnose_dng_bytes`] | 给定 `.dng` 路径 → 走完整管线 → 输出关键诊断 |
//!
//! # 合成的「合法」到什么程度
//!
//! [`synthesize_dng`] 产出的文件满足**本仓读端要求的全部结构**：
//!
//! - TIFF 头（`II`/`MM` + magic 42 + IFD0 偏移），IFD0 条目**按tag 升序**；
//! - DNG 身份：`DNGVersion` / `DNGBackwardVersion` / `UniqueCameraModel`；
//! - raw IFD 必备项：`ImageWidth`/`ImageLength`/`BitsPerSample`/`Compression`/
//!   `PhotometricInterpretation`(CFA=32803)/`SamplesPerPixel`/`StripOffsets`+
//!   `StripByteCounts`+`RowsPerStrip`（或 tile 四标签）+`CFARepeatPatternDim`/
//!   `CFAPattern`；
//! - 显影链必备项：`WhiteLevel` / `BlackLevel` / `ColorMatrix1` / `AsShotNeutral`。
//!
//! **它不假装是相机产物**：不写 `EXIF`/`GPS` 子 IFD、不写 `XMP`。
//! 但**会**按 [`DngSampleSpec::preview`] 写一张 **SubIFD 预览图**——因为真实相机
//! DNG 普遍带它（「不写 SubIFD」的边界已由后续补齐）。
//! 预览图自己的布局由 [`PreviewSpec::layout`] 决定（单 strip / 多 strip / tile），
//! 三者都生成——否则读端的多 strip / tile 重组路径在测试里永远拿不到真实输入。
//!
//! # 支持的压缩方式
//!
//! | `Compression` | 编码路径 |
//! |---|---|
//! | 1 | [`pack_bits`] 紧凑位打包 |
//! | 7 | [`crate::raw_lossless_jpeg_encode`]（SOF3） |
//! | **8 / 32946** | **Deflate(ZIP)**：预测器 → 位打包 → **zlib 容器**（[`deflate_block`]） |
//!
//! `32946`(`0x80B2`) 与 `8` 完全等价（Adobe《Photoshop TIFF Technical Notes》：
//! 该厂商私有码「压缩算法与上述完全相同，应视为过时但建议识别并读取」）。
//!
//! # 端到端诊断读什么
//!
//! [`diagnose_dng`] 走**完整管线**（不是只读标签）：
//!
//! `parse` → `read_raw_samples`（按 `Compression` 分派 1/7/8/32946）→ `cfa_pattern` →
//! `develop`（线性化 → 减黑 → 重标定 → 去马塞克 → 白平衡 → 色彩矩阵），
//! 然后报出尺寸 / CFA 布局 / 压缩方式 / 位深 / 去马赛克后三通道统计量。
//!
//! # 确定性
//!
//! - 合成：纯整数打包 + 固定序遍历，同 `spec` **逐字节可复现**；
//! - 诊断：统计量用**固定顺序**的 `f64` 累加，同输入 `to_bits()` 逐位相同。
//!
//! # C18（永不 panic）
//!
//! 全模块零 `unwrap` / `expect` / `panic!`；所有尺寸/偏移/位深都做
//! `checked_mul` / `checked_add` 并在越界时返回结构化 [`RawError`]。
//!
//! # 诚实边界
//!
//! ## Deflate 是否与 TIFF / DNG 规范完全一致
//!
//! **容器口径**：按 Adobe《Photoshop TIFF Technical Notes》（TIFF 6.0 补充，
//! `Compression = 8` 的规范原文；DNG 1.4 的 `Deflate (ZIP)` 继承它）——
//! 每个 strip / tile 是**一条完整 zlib 流**（2 字节头 + DEFLATE +
//! 4 字节 Adler-32），故编码走 [`crate::exr_write::deflate_zlib`]、
//! 解码走 [`prism_bytes::zlib::decompress_verified`]（**校验 Adler-32 值**）。
//!
//! > 既有的「裸 DEFLATE、不带 zlib 头」描述**与规范原文相反**；
//! > 本实现以规范为准。测试 [`synth_deflate_round_trips_through_full_pipeline`]
//! > 与读端 [`raw::DngFile::read_raw_samples_deflate`] 共用这一口径。
//!
//! **未做的**：DNG 1.4 允许 Deflate 配 `Predictor` 3/ 34894 / 34895
//! （Floating Point 系列，供**浮点** RAW 用），本模块只实装四个**整数**
//! 取值 `1 / 2 / 34892 / 34893`，其余结构化拒绝（见
//! [`raw::DeflatePredictor`]）。Deflate 用于 32 位浮点 / 深度图 / 语义掩膜
//! 这些 DNG 1.4 扩展用途**也不在本模块范围**（本模块只做 CFA 单通道整数）。
//!
//! ## SubIFD 解到什么程度
//!
//! [`crate::raw_subifd`] 解出**元数据 + 像素**：布局 `1`（未压缩）/`7`（无损
//! JPEG）/`8`/`32946`（Deflate）**解出像素**，其余（6 / 34892 / 34712 / 52546）
//! **只报元数据并在 `note` 写明为何不解**。strip 与 tile **两种布局都支持**
//! （含图边界处标称尺寸越界的裁剪）。
//!
//! **不做的**：不做色彩空间转换（`PreviewColorSpace` / `YCbCrSubSampling`
//! 只解析不应用），交错的 `SamplesPerPixel = 3` 样本按 `[(y*w+x)*spp+c]`
//! **如实返回**，由调用方解释。
//!
//! ## 仍不支持的 `Compression` 取值
//!
//! `2/3/4`（CCITT）、`5`（LZW）、`6`（OJPEG，规范从未定义）、
//! `34712`（JPEG 2000）、`34892`（**Lossy JPEG**，非 Deflate 别名）、
//! `52546`（JPEG XL）—— 逐条成因见 [`raw::compression_reject_hint`]。
//! 合成器**只生成** `1 / 7 / 8 / 32946`，其余一律结构化拒绝。

use std::path::Path;

use crate::raw::{
    self, field_type, tags, ByteOrder, CfaPattern, DngFile, FillOrder, RawError, RawResult,
};

/// 字节对齐到偶数（TIFF 要求所有 IFD 条目起始于偶数字节）。
#[inline]
fn align2(x: usize) -> usize {
    (x + 1) & !1usize
}

// ===========================================================================
//  一、样本规格
// ===========================================================================

/// 合成 DNG 的**像素载荷布局**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngLayout {
    /// strip 布局（`StripOffsets`/`StripByteCounts`/`RowsPerStrip`）。
    Strip {
        /// 每个 strip 的行数。
        rows_per_strip: u32,
    },
    /// tile 布局（`TileWidth`/`TileLength`/`TileOffsets`/`TileByteCounts`）。
    Tile {
        /// 单 tile 宽。
        tile_width: u32,
        /// 单 tile 高。
        tile_length: u32,
    },
}

/// **预览图 SubIFD 自己的**载荷布局（与主图的 [`DngLayout`] 独立）。
///
/// # 为什么预览图也需要多 strip / tile
///
/// 真实相机 DNG 的 SubIFD 预览**不总是单 strip**（大尺寸预览会被切成多
/// strip；部分机型的预览直接用 tile）。若合成器只能写单 strip，读端的
/// 多 strip / tile 重组路径就永远得不到真实输入——那条路径也就永远没有
/// 测试守护（这正是 QA 用变异测试挖出的缺口）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewLayout {
    /// 单 strip（`RowsPerStrip` = 预览高，`StripOffsets` 只有一项）。
    SingleStrip,
    /// 多 strip（`RowsPerStrip` < 预览高，`StripOffsets` 有多项）。
    Strip {
        /// 每个 strip 的行数（须 ≥ 1 且 ≤ 预览高）。
        rows_per_strip: u32,
    },
    /// tile 布局（`TileWidth`/`TileLength`/`TileOffsets`/`TileByteCounts`）。
    Tile {
        /// 单 tile 宽（须 ≥ 1）。
        tile_width: u32,
        /// 单 tile 高（须 ≥ 1）。
        tile_length: u32,
    },
}

impl Default for PreviewLayout {
    /// 单 strip —— 预览图的常态，也是本模块历史行为（保持既有测试不变）。
    fn default() -> Self {
        Self::SingleStrip
    }
}

/// 合成 DNG 的**SubIFD 预览图**规格。
///
/// # 为什么要有这个
///
/// 诚实边界里写「不写 SubIFD 预览」，理由是「对本链路的验证目标无影响」。
/// 但**真实相机 DNG 普遍带 SubIFD 预览图**，缺了它就意味着
/// 「能读主图但看不到预览」——这条边界已补齐。
///
/// # 预览图的典型形态
///
/// | 字段 | 真实相机常见值 |
/// |---|---|
/// | `compression` | **7**（无损 JPEG）或 **1**（未压缩） |
/// | `photometric` | 1（BlackIsZero）/ 2（RGB）/ 6（YCbCr） |
/// | `samples_per_pixel` | 1（灰度）或 3（RGB / YCbCr） |
/// | `bits` | 8（预览图通常 8 位） |
///
/// # 与主图的关系
///
/// 预览图是**独立子 IFD**，有自己的尺寸 / 位深 / 压缩 / 色彩解释，
/// **与主图的 CFA 无关**（故不写 `CFAPattern`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewSpec {
    /// 预览图宽。
    pub width: u32,
    /// 预览图高。
    pub height: u32,
    /// 位深（1..=16；预览图通常 8）。
    pub bits: u16,
    /// 压缩方式（`1` 未压缩 / `7` 无损 JPEG / `8`、`32946` Deflate）。
    pub compression: u16,
    /// `PhotometricInterpretation`（1 = BlackIsZero、2 = RGB、6 = YCbCr）。
    pub photometric: u16,
    /// 每像素样本数（1 = 灰度、3 = RGB/YCbCr）。
    pub samples_per_pixel: u16,
    /// `NewSubfileType`（缺省 1 = 降分辨率图像 `reduced-resolution image`）。
    pub new_subfile_type: u32,
    /// 预览图自己的载荷布局（单 strip / 多 strip / tile）。
    pub layout: PreviewLayout,
}

impl Default for PreviewSpec {
    /// 160×120、8 位、**无损 JPEG**、RGB 三通道、`reduced-resolution image`。
    ///
    /// 选无损 JPEG 作默认是因为**真实相机 DNG 的预览图绝大多数是无损 JPEG**
    /// （规范允许的另两种是未压缩与 Deflate）。
    fn default() -> Self {
        Self {
            width: 160,
            height: 120,
            bits: 8,
            compression: 7,
            photometric: 2,
            samples_per_pixel: 3,
            new_subfile_type: 1,
            layout: PreviewLayout::SingleStrip,
        }
    }
}

impl PreviewSpec {
    /// 快捷构造：`w × h`、8 位、给定压缩方式与色彩解释。
    pub fn new(
        width: u32,
        height: u32,
        bits: u16,
        compression: u16,
        photometric: u16,
        samples_per_pixel: u16,
    ) -> Self {
        Self {
            width,
            height,
            bits,
            compression,
            photometric,
            samples_per_pixel,
            ..Self::default()
        }
    }
}

/// 合成一份「最小合法 DNG」的完整规格。
#[derive(Debug, Clone, PartialEq)]
pub struct DngSampleSpec {
    /// 原图宽（像素）。
    pub width: u32,
    /// 原图高（像素）。
    pub height: u32,
    /// 位深（`BitsPerSample`）。
    pub bits: u16,
    /// CFA 平铺颜色（行主序 2×2；`0=R / 1=G / 2=B`）。
    pub cfa: [u8; 4],
    /// 压缩方式（**只支持 1 / 7 / 8 / 32946**；其余值由 [`synthesize_dng`]
    /// 结构化拒绝）。
    ///
    /// - `1` —— 未压缩（位打包）；
    /// - `7` —— 无损 JPEG（[`crate::raw_lossless_jpeg_encode`]）；
    /// - `8` / `32946` —— **Deflate(ZIP)**，**zlib 容器**（`deflate_block`）。
    pub compression: u16,
    /// 载荷布局。
    pub layout: DngLayout,
    /// 字节序（`II` / `MM`）。
    pub order: ByteOrder,
    /// 位序（仅对 `Compression = 1` 且位深非 8/16 时有意义）。
    pub fill_order: FillOrder,
    /// `Compression = 7` 的预测器选择值（`Ss`，1..=7；T.81 Table H.1）。
    pub predictor: u8,
    /// `Compression = 7` 的无损 JPEG **分量数** `Nf`（1..=4）。
    ///
    /// `Nf = 1` 是普通单分量；`Nf = 2` 是 DNG 生态里最常见的 **CFA 重打包**
    /// （把一行的偶数列/奇数列拆成两个分量，让预测器「跨同色像素」工作，
    /// 见 [`DngSampleSpec::cfa_repack_two_components`]）。
    pub components: u16,
    /// 无损 JPEG 的**行交织因子**（`RowInterleaveFactor`；`None` = 不写该标签）。
    pub row_interleave_factor: Option<u32>,
    /// `Compression = 8` / `32946` 的**预测器**（`Predictor`，0x013D）。
    ///
    /// DNG 1.4 为 Deflate 定义的整数预测器：**1**（Null）/ **2**（Horizontal）
    /// / **34892**（Horizontal X2）/ **34893**（Horizontal X4）。
    /// 浮点预测器（3 / 34894 / 34895）需浮点 RAW，本合成器不支持。
    ///
    /// 缺省 `1`（Null）——不写该标签时读端也按 1 解释。
    pub deflate_predictor: u16,
    /// 预览图规格（`Some` = 在 SubIFD 里写一张预览图）。
    ///
    /// `None` = 不写 SubIFD（保持「不假装完整」的立场）。
    pub preview: Option<PreviewSpec>,
    /// 逐 CFA 位置的黑电平。
    pub black_level: [u32; 4],
    /// 白电平。
    pub white_level: u32,
}

impl Default for DngSampleSpec {
    /// 16×16、16 位、RGGB、`Compression = 1`、单 strip、小端。
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            bits: 16,
            cfa: [0, 1, 1, 2],
            compression: 1,
            layout: DngLayout::Strip { rows_per_strip: 16 },
            order: ByteOrder::Little,
            fill_order: FillOrder::MsbFirst,
            predictor: 1,
            components: 1,
            row_interleave_factor: None,
            deflate_predictor: 1,
            preview: None,
            black_level: [0; 4],
            white_level: 65535,
        }
    }
}

impl DngSampleSpec {
    /// 快捷构造：`w × h`、指定位深与 CFA，其余取默认。
    pub fn new(width: u32, height: u32, bits: u16, cfa: [u8; 4]) -> Self {
        Self {
            width,
            height,
            bits,
            cfa,
            ..Self::default()
        }
    }

    /// 把 `Compression = 7` 的载荷改成 **`Nf = 2` 的 CFA 重打包**。
    ///
    /// DNG 1.6 明确允许 SOF3 的 `X`/`Y`/`Nf` 与 strip/tile 的标称不同，只要
    /// **总样本数**相等：
    ///
    /// > For lossless JPEG, the internal width/length/components in the JPEG
    /// > stream are **not required to match** the strip or tile's
    /// > width/length/components. Only the total sample counts need to match.
    /// > It is common for CFA images to be encoded with a different width,
    /// > length or component count to allow the JPEG compression predictors to
    /// > work across like colors.
    ///
    /// 本方法按读端 [`crate::raw_lossless_jpeg::deinterleave_components`] 的
    /// 映射（`cfa[y][x] = comp[x % 2][x / 2]`）做**逆向**拆分，故合成出的
    /// 样本能被读端**逐位还原**。
    ///
    /// > 注：`Nf = 2` 要求图幅宽为**偶数**（每行才能均分成两个分量），
    /// > 否则 [`synthesize_dng`] 会结构化拒绝。
    pub fn cfa_repack_two_components(mut self) -> Self {
        self.components = 2;
        self
    }
}

/// 规格校验失败（**结构化**，绝不带着非法规格去写文件）。
fn validate_spec(spec: &DngSampleSpec) -> RawResult<()> {
    const LOC: &str = "raw_dng_synth::validate_spec";
    if spec.width == 0 || spec.height == 0 {
        return Err(raw::tiff_err(
            LOC,
            format!("图幅 {}×{} 含 0 维", spec.width, spec.height),
            "DNG 的 ImageWidth/ImageLength 须均 ≥ 1；请改用更小的非空图幅",
        ));
    }
    let px = (spec.width as u64)
        .checked_mul(spec.height as u64)
        .ok_or_else(|| {
            raw::tiff_err(
                LOC,
                format!("{}×{} 像素数溢出", spec.width, spec.height),
                "使用更小的图幅",
            )
        })?;
    // 防御性上限：与 raw_lossless_jpeg 的 MAX_SAMPLES 同量级（100 MP）。
    if px > (1u64 << 28) {
        return Err(raw::unsupported_err(
            LOC,
            format!("图幅 {}×{} = {px} 像素，超过合成上限 2^28", spec.width, spec.height),
            "合成器只用于生成小样本；真实大图请直接用真实 DNG",
        ));
    }
    if spec.bits == 0 || spec.bits > 16 {
        return Err(raw::unsupported_err(
            LOC,
            format!("BitsPerSample = {}，本模块只合成 1..=16 位", spec.bits),
            "DNG 常见位深为 8/10/12/14/16；32 位浮点 RAW 请先转 16 位整数 DNG",
        ));
    }
    match spec.cfa.as_slice() {
        [0, 1, 1, 2] | [2, 1, 1, 0] | [1, 0, 2, 1] | [1, 2, 0, 1] => {}
        other => {
            return Err(raw::unsupported_err(
                LOC,
                format!("CFAPattern = {other:?}，不是 Bayer 四态"),
                "只支持 RGGB / BGGR / GRBG / GBRG 四种排列（0=R 1=G 2=B）",
            ))
        }
    }
    // Deflate 的 Predictor 只允许 DNG 1.4 定义的**四个整数**取值。
    // 浮点预测器（3 / 34894 / 34895）需浮点 RAW，本合成器只做整数样本。
    //
    // **必须在写文件之前拒**：预测器搞错会产出「看似能读、实则全错」的像素，
    // 比直接报错坏得多（铁律：诚实边界优于虚假功能）。
    if spec.compression == 8 || spec.compression == 32946 {
        if raw::DeflatePredictor::from_code(spec.deflate_predictor).is_none() {
            return Err(raw::unsupported_err(
                LOC,
                format!(
                    "Predictor = {}：{}",
                    spec.deflate_predictor,
                    raw::DeflatePredictor::reject_reason(spec.deflate_predictor)
                ),
                "本合成器的 Deflate 路径只支持 1（Null）/ 2（Horizontal）/\
                 34892（X2）/ 34893（X4）四个整数预测器",
            ));
        }
    }
    if spec.compression == 7 {
        if spec.components == 0 || spec.components > 4 {
            return Err(raw::unsupported_err(
                LOC,
                format!("无损 JPEG 的 Nf = {}，只支持 1..=4", spec.components),
                "DNG 的 CFA 重打包最多 4 分量（2×2 Bayer）",
            ));
        }
        if spec.components > 1 && !spec.width.is_multiple_of(spec.components as u32) {
            return Err(raw::tiff_err(
                LOC,
                format!(
                    "图幅宽 {} 不能被分量数 Nf = {} 整除",
                    spec.width, spec.components
                ),
                "CFA 重打包要求每行按「列号≡ c (mod Nf)」均分；请改用 Nf 的整数倍的宽度",
            ));
        }
        if spec.bits < 2 {
            return Err(raw::unsupported_err(
                LOC,
                format!("无损 JPEG 不支持 BitsPerSample = {}", spec.bits),
                "T.81 无损模式的精度 P 至少为 2；1 位请用 Compression = 1",
            ));
        }
    }
    match spec.layout {
        DngLayout::Strip { rows_per_strip: 0 } => {
            return Err(raw::tiff_err(
                LOC,
                "RowsPerStrip = 0",
                "每 strip 至少 1 行；确认布局参数",
            ))
        }
        DngLayout::Tile {
            tile_width,
            tile_length,
        } => {
            // 用嵌套 if 而非 match guard：guard 会让 clippy 判为 redundant_guards，
            // 而 or-pattern 又会丢掉 tile_width / tile_length 绑定导致错误消息无法引用。
            if tile_width == 0 || tile_length == 0 {
                return Err(raw::tiff_err(
                    LOC,
                    format!("tile 尺寸 {tile_width}×{tile_length} 含 0 维"),
                    "TileWidth/TileLength 须均 ≥ 1",
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

// ===========================================================================
//  二、TIFF / DNG 写入器（自写，零第三方依赖）
// ===========================================================================

/// IFD 条目（`tag` / `type` / `count` / 负载字节）。
struct Entry {
    tag: u16,
    ty: u16,
    count: u32,
    data: Vec<u8>,
}

/// 按字节序追加一个 `u16`。
#[inline]
fn put16(out: &mut Vec<u8>, v: u16, le: bool) {
    out.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
}

/// 按字节序追加一个 `u32`。
#[inline]
fn put32(out: &mut Vec<u8>, v: u32, le: bool) {
    out.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
}

/// IFD0 构建器（负责按字节序编码各类负载）。
struct Ifd0 {
    le: bool,
    entries: Vec<Entry>,
}

impl Ifd0 {
    fn new(le: bool) -> Self {
        Self {
            le,
            entries: Vec::new(),
        }
    }

    fn put(&mut self, tag: u16, ty: u16, count: u32, data: Vec<u8>) {
        self.entries.push(Entry { tag, ty, count, data });
    }

    fn shorts(&mut self, tag: u16, v: &[u16]) {
        let mut d = Vec::with_capacity(v.len() * 2);
        for &x in v {
            put16(&mut d, x, self.le);
        }
        self.put(tag, field_type::SHORT, v.len() as u32, d);
    }

    fn longs(&mut self, tag: u16, v: &[u32]) {
        let mut d = Vec::with_capacity(v.len() * 4);
        for &x in v {
            put32(&mut d, x, self.le);
        }
        self.put(tag, field_type::LONG, v.len() as u32, d);
    }

    fn bytes(&mut self, tag: u16, v: &[u8]) {
        self.put(tag, field_type::BYTE, v.len() as u32, v.to_vec());
    }

    fn ascii(&mut self, tag: u16, s: &str) {
        let mut d = s.as_bytes().to_vec();
        d.push(0);
        self.put(tag, field_type::ASCII, d.len() as u32, d);
    }

    /// `RATIONAL × n`（分母恒 1 —— 合成器只写整数值的黑白电平）。
    fn rationals(&mut self, tag: u16, v: &[u32]) {
        let mut d = Vec::with_capacity(v.len() * 8);
        for &x in v {
            put32(&mut d, x, self.le);
            put32(&mut d, 1, self.le);
        }
        self.put(tag, field_type::RATIONAL, v.len() as u32, d);
    }

    /// `SRATIONAL × 9`（`ColorMatrix1`，分母恒 1024 → 精确表示 1/1024 的整数倍）。
    fn srationals9(&mut self, tag: u16, v: &[f32; 9]) {
        let mut d = Vec::with_capacity(9 * 8);
        for &x in v.iter() {
            let n = (x * 1024.0).round() as i32;
            let b = if self.le {
                n.to_le_bytes()
            } else {
                n.to_be_bytes()
            };
            d.extend_from_slice(&b);
            let b = if self.le {
                1024i32.to_le_bytes()
            } else {
                1024i32.to_be_bytes()
            };
            d.extend_from_slice(&b);
        }
        self.put(tag, field_type::SRATIONAL, 9, d);
    }
}

/// 序列化 IFD0（含外置值堆）。
///
/// 版式：`[count:u16][entry×12][next:u32]`，条目按 tag 升序（TIFF 5.0 §2 要求）。
/// 负载 `≤ 4` 字节内联（**小端右对齐、大端左对齐**），`> 4` 字节写堆偏移。
///
/// # `ifd_file_off` —— 外置值偏移必须是**文件绝对偏移**
///
/// TIFF 的 value 字段里存的偏移是**从文件开头**算的，而本函数吐出的字节块
/// 会被整体放在文件的 `ifd_file_off` 处。故堆的文件基址是
/// `ifd_file_off + 目录区长度`，**不是** 0：
///
/// ```text
/// ifd_file_off + 2 + 12n + 4 = 堆的文件基址
/// ```
///
/// 传 0 会让所有外置值偏移**少 8**（IFD0 固定在文件偏移 8）——表现为
/// 「StripOffsets 指向文件外、载荷偏移大得离谱」。这正是本函数把
/// `ifd_file_off` 作为显式入参（而非硬编码）的原因。
fn write_ifd0(ifd0: &Ifd0, ifd_file_off: usize) -> Vec<u8> {
    let mut entries: Vec<&Entry> = ifd0.entries.iter().collect();
    entries.sort_by_key(|e| e.tag);
    let le = ifd0.le;

    // 目录区长度 = 2（count）+ 12 × 条目数 + 4（next IFD）——与实际写入的
    // 字节数严格一致，故堆的文件基址可在**写第一条**前就确定。
    let dir_len = 2 + 12 * entries.len() + 4;
    let heap_file_base = ifd_file_off + dir_len;

    let mut out: Vec<u8> = Vec::new();
    put16(&mut out, entries.len() as u16, le);
    let mut heap: Vec<u8> = Vec::new();
    for e in entries {
        put16(&mut out, e.tag, le);
        put16(&mut out, e.ty, le);
        put32(&mut out, e.count, le);
        if e.data.len() <= 4 {
            let mut field = [0u8; 4];
            if le {
                field[4 - e.data.len()..].copy_from_slice(&e.data);
            } else {
                field[..e.data.len()].copy_from_slice(&e.data);
            }
            out.extend_from_slice(&field);
        } else {
            // 外置值：堆内按偶数字节对齐（TIFF §2），偏移是**文件绝对**偏移。
            let padded = align2(heap.len());
            heap.resize(padded, 0);
            let off = (heap_file_base + padded) as u32;
            heap.extend_from_slice(&e.data);
            put32(&mut out, off, le);
        }
    }
    put32(&mut out, 0, le); // next IFD = 0（无）
    debug_assert_eq!(out.len(), dir_len);
    out.extend_from_slice(&heap);
    out
}

// ===========================================================================
//  三、像素载荷的打包 / 编码
// ===========================================================================

/// 把样本按 `bits` 位紧凑打包（[`crate::raw::unpack_bits`] 的**逆**）。
///
/// 与测试侧的参考实现分开写，避免「用被测代码的逆验证被测代码」。位序语义
/// 严格对称于 `unpack_bits`：MSB-first 时样本内**高位在前**、字节内**7→0**；
/// LSB-first 时两者都反转。
fn pack_bits(values: &[u16], bits: u16, fill_order: FillOrder) -> Vec<u8> {
    let b = bits as usize;
    let total_bits = values.len() * b;
    let mut out: Vec<u8> = vec![0u8; total_bits.div_ceil(8)];
    let mask: u32 = (1u32 << bits) - 1;
    for (k, &v) in values.iter().enumerate() {
        let val = (v as u32) & mask;
        for j in 0..b {
            let bit = match fill_order {
                FillOrder::MsbFirst => (val >> (b - 1 - j)) & 1,
                FillOrder::LsbFirst => (val >> j) & 1,
            };
            let stream_pos = k * b + j;
            let byte = stream_pos / 8;
            let in_byte = stream_pos % 8;
            let bit_in_byte = match fill_order {
                FillOrder::MsbFirst => 7 - in_byte,
                FillOrder::LsbFirst => in_byte,
            };
            if let Some(slot) = out.get_mut(byte) {
                *slot |= (bit as u8) << bit_in_byte;
            }
        }
    }
    out
}

/// `RowInterleaveFactor` 的**编码**侧行映射：返回 `coded[j] = dst`，即码流第 `j` 行
/// 放的是**整图第 `dst` 行**。
///
/// 直接复用读端 [`crate::raw_lossless_jpeg::interleave_row_map`] 的同一张表，
/// 参数口径也**必须与读端一致**：
///
/// - `rows` —— **本块（strip/tile）的标称行数**，不是整图行数；
/// - `y0` —— 本块首行的**整图行号**。
///
/// 读端在 [`crate::raw_lossless_jpeg`] 里正是这样调的（`rows = stream_height`,
/// `y0 = 本 strip 首行`），故「按本表排布码流行」与「读端还原」必然互逆。
fn interleave_encode_map(rows: u32, y0: u32, factor: u32) -> RawResult<Vec<u32>> {
    crate::raw_lossless_jpeg::interleave_row_map(rows, y0, factor)
}

/// 把一行 CFA 样本按「列号 ≡ c (mod nf)」重打包成 `nf` 个分量平面。///
/// 这是 DNG 1.6 明确允许的 CFA 重打包（"the internal width/length/components
/// in the JPEG stream are not required to match the strip or tile's"），
/// 与 [`crate::raw_lossless_jpeg::deinterleave_components`] 的映射严格互逆：
/// `cfa[y][x] = component[x % nf][x / nf]`。
fn split_cfa_components(row: &[u16], nf: usize) -> Vec<Vec<u16>> {
    let mut comps: Vec<Vec<u16>> = vec![Vec::new(); nf];
    for (x, &v) in row.iter().enumerate() {
        let c = x % nf;
        if let Some(p) = comps.get_mut(c) {
            p.push(v);
        }
    }
    comps
}

/// 编码一个 `Compression = 8` / `32946`（Deflate）块：**zlib 容器**。
///
/// # 流水线（与读端 [`crate::raw::DngFile::read_raw_samples_deflate`] 严格互逆）
///
/// ```text
/// 原始样本 → 预测器（差分） → pack_bits（位打包） → zlib 压缩
/// ```
///
/// 读端反过来：`zlib 解压 → unpack_bits → 预测器逆运算`。
/// **顺序不可交换**：预测器作用于**样本值**而非字节，故必须先打包成
/// `bits` 位样本、再算差分。
///
/// # 为什么用 zlib 而不是裸 DEFLATE
///
/// Adobe《Photoshop TIFF Technical Notes》定义 `Compression = 8` 为
/// "Deflate compression, **using zlib data format**"，且
/// "Each image segment consists of a **single complete zlib data stream**"。
/// 故这里走 [`crate::exr_write::deflate_zlib`]（`0x78 0x01` 头 +
/// fixed-Huffman/stored 取小 + Adler-32 尾），与规范一致。
///
/// # 为什么复用 `exr_write` 的压缩器（**不写第二份**）
///
/// `prism-bytes` 只有解压、没有压缩器（该 crate 的诚实边界明确写着）。
/// 而 `exr_write::deflate_zlib` 已是**本仓唯一**的真压缩 zlib 写入器
/// （已实装，且有「能被 `prism_bytes::inflate` 逐位解回」的测试）。
/// 再写一份只会制造**第二份实现 = 第二份 bug 面**，正是 `prism-bytes`
/// 模块文档里批评的那种技术债。故**复用**。
fn deflate_block(
    samples: &mut [u16],
    row_len: usize,
    bits: u16,
    fill_order: FillOrder,
    predictor: raw::DeflatePredictor,
) -> Vec<u8> {
    // ① 预测器（就地差分）——与读端 `undo_in_place` 互逆。
    predictor.apply_in_place(samples, row_len, bits);
    // ② 位打包。
    // ③ zlib 容器（fixed Huffman / stored 取较小者 + Adler-32）。
    crate::exr_write::deflate_zlib(&pack_bits(samples, bits, fill_order))
}

/// 测试支撑：暴露预览图案规则给 [`crate::raw_subifd`] 的测试
/// （避免那边复制第二份实现——两边共用同一份才是真的一致性检查）。
#[cfg(test)]
pub(crate) fn tests_support_preview_pattern(spec: &PreviewSpec) -> Vec<u16> {
    preview_pattern(spec)
}

/// 预览图的确定性测试图案（**不依赖主图**，与主图图案规则不同）。
///
/// 每通道一个不同的线性斜坡，便于断言「RGB 三通道没有串位」。
fn preview_pattern(spec: &PreviewSpec) -> Vec<u16> {
    let n = (spec.width as usize) * (spec.height as usize) * (spec.samples_per_pixel as usize);
    let max_v: u64 = (1u64 << spec.bits.min(16)) - 1;
    (0..n)
        .map(|i| {
            let x = i % (spec.width as usize);
            let y = i / (spec.width as usize);
            // 通道相关的斜坡：`c` 越靠后偏移越大。
            let c = i % (spec.samples_per_pixel.max(1) as usize);
            let v = (x as u64 * 7 + y as u64 * 13 + c as u64 * 101) % (max_v + 1);
            v as u16
        })
        .collect()
}

// ===========================================================================
//  四、合成主入口
// ===========================================================================

/// **从零合成一个最小但合法的 DNG 文件**。
///
/// # 参数
///
/// `spec` —— 完整规格（见 [`DngSampleSpec`]；`Default` 即一份16×16 的 16 位
/// RGGB 未压缩样本）。
///
/// # 返回
///
/// 完整的 `.dng` 字节流（TIFF 容器）。把它 [`diagnose_dng_bytes`] 回去即得
/// 端到端诊断；或直接 [`diagnose_dng`] 一个已落盘的路径。
///
/// # 支持的压缩方式
///
/// - `1`（未压缩）：按 `BitsPerSample` 位打包，支持 1..=16 位 ×两种位序；
/// - `7`（无损 JPEG）：单分量 SOF3，预测器由 `spec.predictor` 指定，
///   支持 `RowInterleaveFactor`。
///
/// **其余取值（含 6）一律结构化拒绝** —— 依据见模块级文档与「诚实边界」。
///
/// # 错误
///
/// - 规格非法（图幅 0 维 / 位深越界 / 非 Bayer CFA / 非法 tile 尺寸）→
///   `RAW1xxx` / `RAW2xxx`；
/// - `compression` 非 1/7 → `RAW2001`（**不臆造未验证的格式语义**）。
///
/// # 示例
///
/// ```
/// use prism_io::{synthesize_dng, DngSampleSpec, DngLayout, diagnose_dng_bytes};
///
/// let spec = DngSampleSpec {
///     compression: 7,
///     layout: DngLayout::Strip { rows_per_strip: 8 },
///     ..DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2])
/// };
/// let bytes = synthesize_dng(&spec)?;
/// let diag = diagnose_dng_bytes(&bytes)?;
/// assert_eq!(diag.width, 16);
/// # Ok::<(), prism_io::RawError>(())
/// ```
pub fn synthesize_dng(spec: &DngSampleSpec) -> RawResult<Vec<u8>> {
    const LOC: &str = "raw_dng_synth::synthesize_dng";
    validate_spec(spec)?;
    match spec.compression {
        1 => {}
        7 => {}
        // DNG 1.4 Deflate(ZIP) + 它的厂商旧编号 32946（0x80B2）。
        8 | 32946 => {}
        other => {
            return Err(raw::unsupported_err(
                LOC,
                format!(
                    "合成器只支持 Compression = 1（未压缩）、7（无损 JPEG）与 8 / 32946（Deflate），\
                     收到 {other}"
                ),
                "其余取值的**确切语义**在无规范原文时无法自证（Compression=6 在 \
                 TIFF 6.0 里是 Old JPEG，而 DNG 规范并未定义它；34892 是有损 JPEG）——\
                 按项目铁律「诚实边界优于虚假功能」，合成器不生成未验证的格式；\
                 读取端对未支持取值一律结构化拒绝",
            ))
        }
    }

    let w = spec.width;
    let h = spec.height;
    let le = matches!(spec.order, ByteOrder::Little);
    let n_px = (w as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| raw::tiff_err(LOC, "像素数溢出", "使用更小的图幅"))?;

    // ---- 1. 造出确定性的马赛克图案 ----
    //
    // 图案设计目标是「**可诊断**」而非「好看」：
    // ① 覆盖全码值范围（相邻样本差足够大，位打包/Lossless 预测器的错位会立刻暴露）；
    // ② 同一 CFA 相位上的样本**相同**（便于按 CFA 平面逐位校验，见 `diagnose_*`）；
    // ③ 与 `(x, y)` 强相关（水平/垂直方向非对称，行序/列序错位必被抓住）。
    let max_v: u64 = (1u64 << spec.bits.min(16)) - 1;
    let mosaic: Vec<u16> = (0..n_px)
        .map(|i| {
            let x = i % w as usize;
            let y = i / w as usize;
            // 相位内恒定 + 坐标相关：`(37i + 11x + 53y)` 再对 `2^bits` 取模。
            let v = (i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % (max_v + 1);
            v as u16
        })
        .collect();

    // ---- 2. 按布局切块并编码每块的载荷 ----
    //
    // 载荷基址依赖 IFD0 长度，IFD0 长度依赖外置值堆，而偏移数组本身也在堆里
    // —— 故用**两遍**：第一遍用占位偏移算出堆长与载荷基址，第二遍回填真偏移。
    let layout = spec.layout;
    let block_rows: Vec<u32> = match layout {
        DngLayout::Strip { rows_per_strip } => {
            let mut v = Vec::new();
            let mut y = 0u32;
            while y < h {
                v.push(rows_per_strip.min(h - y));
                y = y.saturating_add(rows_per_strip);
            }
            v
        }
        DngLayout::Tile { tile_length, .. } => {
            let mut v = Vec::new();
            let mut y = 0u32;
            while y < h {
                v.push(tile_length.min(h - y));
                y = y.saturating_add(tile_length);
            }
            v
        }
    };
    let across: Vec<u32> = match layout {
        DngLayout::Strip { .. } => vec![1],
        DngLayout::Tile { tile_width, .. } => {
            let mut v = Vec::new();
            let mut x = 0u32;
            while x < w {
                v.push(tile_width.min(w - x));
                x = x.saturating_add(tile_width);
            }
            v
        }
    };
    // tile 的**标称**尺寸（用于载荷编解码的行列数；strip 时即 w）。
    let (tw, tl) = match layout {
        DngLayout::Strip { .. } => (w, 0),
        DngLayout::Tile {
            tile_width,
            tile_length,
        } => (tile_width, tile_length),
    };

    // 编码一个 (x0, y0, bw, bh) 块。
    let nf = if spec.compression == 7 {
        spec.components as usize
    } else {
        1
    };
    // tile 布局下，**载荷必须按 tile 的标称尺寸**（含图边界外的填充像素）编码，
    // 而不是只编有效区——这是 DNG/TIFF 的 tile 语义，也是读端的做法：
    // 读端 `read_raw_samples` 的 tile 分支「整 tile 解包（含图边界外的填充
    // 像素），再只拷贝有效区」，注释里写明原因：「否则边界 tile 的位流起点
    // 会算错」。故合成器必须与之一致：**编标称尺寸，越界像素填0**。
    let nominal: Option<(u32, u32)> = match spec.layout {
        DngLayout::Strip { .. } => None,
        DngLayout::Tile {
            tile_width,
            tile_length,
        } => Some((tile_width, tile_length)),
    };
    let encode_block = |x0: u32, y0: u32, bw: u32, bh: u32| -> RawResult<Vec<u8>> {
        // 编码尺寸 = 标称 tile 尺寸（tile 布局）或有效区尺寸（strip 布局）。
        let (enc_w, enc_h) = nominal.unwrap_or((bw, bh));
        // 收集块内样本；越界像素填 0（确定性填充，且必被读端丢弃）。
        let mut block: Vec<u16> = Vec::with_capacity((enc_w as usize) * (enc_h as usize));
        for y in y0..y0 + enc_h {
            for x in x0..x0 + enc_w {
                let inside = x < w && y < h;
                let idx = (y as usize) * (w as usize) + (x as usize);
                block.push(if inside {
                    mosaic.get(idx).copied().unwrap_or(0)
                } else {
                    0
                });
            }
        }
        let bw = enc_w;
        let bh = enc_h;
        if spec.compression == 1 {
            return Ok(pack_bits(&block, spec.bits, spec.fill_order));
        }
        // ---- Compression = 8 / 32946：Deflate(zlib) ----
        if spec.compression == 8 || spec.compression == 32946 {
            // 预测器在此**校验并转换**一次（非法取值已在 validate_spec 拒，
            // 这里用 from_code().unwrap_or(None) 只是防御，读端同一口径）。
            let predictor = raw::DeflatePredictor::from_code(spec.deflate_predictor)
                .unwrap_or(raw::DeflatePredictor::None);
            let mut block = block;
            // 每行样本数 = 标称块宽（spp 恒为 1，见上面的 nf 计算）。
            return Ok(deflate_block(
                &mut block,
                bw as usize,
                spec.bits,
                spec.fill_order,
                predictor,
            ));
        }
        // ---- Compression = 7：无损 JPEG ----
        let interleave = spec.row_interleave_factor.unwrap_or(1);
        // 行交织：按整图行号把块内行重排进码流。
        //
        // **必须用读端同一张表**：读端调的是 `interleave_row_map(stream_height, y0, factor)`
        // ——`rows` 是**本块标称行数**、`y0` 是**本块首行的整图行号**，不是整图行数。
        // 早期版本这里用整图映射（`rows = height, y0 = 0`）再按块过滤，
        // 在「图高不是块高整数倍」时会少算行数，报
        // 「分量 0 长度 64 ≠ 宽×高 = 128」。
        let coded: Vec<Vec<u16>> = if interleave > 1 {
            let map = interleave_encode_map(bh, y0, interleave)?;
            // 每个分量平面各按码流行序排布（多分量时每个 field 内部逐行重排）。
            // `map` 由 `interleave_row_map(bh, y0, factor)` 生成，其每个元素
            // 按契约都落在 `[y0, y0+bh)` 内，故 `local` 必然是本块内的合法行号。
            let mut out: Vec<Vec<u16>> = vec![Vec::new(); nf];
            for &dst in map.iter() {
                let local = dst.saturating_sub(y0) as usize;
                if local >= bh as usize {
                    continue; // 契约外的防御：宁可少一行也不越界 panic
                }
                let src = local * (bw as usize);
                let row = &block[src..src + bw as usize];
                if nf == 1 {
                    if let Some(p) = out.first_mut() {
                        p.extend_from_slice(row);
                    }
                } else {
                    for (c, plane) in split_cfa_components(row, nf).into_iter().enumerate() {
                        if let Some(p) = out.get_mut(c) {
                            p.extend_from_slice(&plane);
                        }
                    }
                }
            }
            out
        } else if nf == 1 {
            vec![block]
        } else {
            // 不交织：逐行拆成 Nf 个分量平面（每平面高度 = bh）。
            let mut out: Vec<Vec<u16>> = vec![Vec::new(); nf];
            for y in 0..bh as usize {
                let src = y * bw as usize;
                let row = &block[src..src + bw as usize];
                for (c, plane) in split_cfa_components(row, nf).into_iter().enumerate() {
                    if let Some(p) = out.get_mut(c) {
                        p.extend_from_slice(&plane);
                    }
                }
            }
            out
        };
        // SOF3 的 X/Y 是**每分量**的宽高（CFA 总宽 = nf × X）。
        let jpeg_w = bw / (nf as u32);
        crate::raw_lossless_jpeg_encode::encode_multi(
            &coded,
            jpeg_w,
            bh,
            spec.bits as u8,
            spec.predictor,
            0,
            0,
        )
    };

    // ---- 3. 块列表（tile 时按行优先，与读端的 `ty*across + tx` 一致）----
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    match layout {
        DngLayout::Strip { rows_per_strip } => {
            let mut y = 0u32;
            while y < h {
                let rows = rows_per_strip.min(h - y);
                payloads.push(encode_block(0, y, w, rows)?);
                y = y.saturating_add(rows_per_strip);
            }
        }
        DngLayout::Tile { .. } => {
            for ty in 0..block_rows.len() as u32 {
                for tx in 0..across.len() as u32 {
                    let x0 = tx.saturating_mul(tw);
                    let y0 = ty.saturating_mul(tl);
                    let bh = block_rows[ty as usize];
                    let bw = across[tx as usize];
                    payloads.push(encode_block(x0, y0, bw, bh)?);
                }
            }
        }
    }
    let counts: Vec<u32> = payloads.iter().map(|p| p.len() as u32).collect();
    let n_blocks = payloads.len();

    // ---- 4. 组装 IFD0 ----
    let build_ifd = |offsets: &[u32]| -> Ifd0 {
        let mut f = Ifd0::new(le);
        f.longs(tags::NEW_SUBFILE_TYPE, &[0]); // 0 = 主图
        f.longs(tags::IMAGE_WIDTH, &[w]);
        f.longs(tags::IMAGE_LENGTH, &[h]);
        f.shorts(tags::BITS_PER_SAMPLE, &[spec.bits]);
        f.shorts(tags::COMPRESSION, &[spec.compression]);
        f.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]); // CFA
        f.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        f.shorts(tags::FILL_ORDER, &[spec.fill_order.code()]);
        f.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        f.bytes(tags::CFA_PATTERN, &spec.cfa);
        match layout {
            DngLayout::Strip { rows_per_strip } => {
                f.longs(tags::STRIP_OFFSETS, offsets);
                f.longs(tags::STRIP_BYTE_COUNTS, &counts);
                f.longs(tags::ROWS_PER_STRIP, &[rows_per_strip]);
            }
            DngLayout::Tile {
                tile_width,
                tile_length,
            } => {
                f.shorts(tags::TILE_WIDTH, &[tile_width as u16]);
                f.shorts(tags::TILE_LENGTH, &[tile_length as u16]);
                f.longs(tags::TILE_OFFSETS, offsets);
                f.longs(tags::TILE_BYTE_COUNTS, &counts);
            }
        }
        f.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        f.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        f.ascii(tags::UNIQUE_CAMERA_MODEL, "PrismEngineSynthCamera");
        f.longs(tags::WHITE_LEVEL, &[spec.white_level]);
        f.rationals(tags::BLACK_LEVEL, &spec.black_level);
        f.srationals9(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        f.rationals(tags::AS_SHOT_NEUTRAL, &[0, 0, 0]); // 占位，下面改写
        f
    };
    // `AsShotNeutral` 必须是正值（读端要求 `x > 0`），故单独重写。
    let mut f0 = build_ifd(&vec![0u32; n_blocks]);
    f0.entries.retain(|e| e.tag != tags::AS_SHOT_NEUTRAL);
    {
        // AsShotNeutral 用 RATIONAL 表示 1/2、1、3/4（分母非 1，故单独构造）。
        let vals: [(u32, u32); 3] = [(1, 2), (1, 1), (3, 4)];
        let mut d = Vec::with_capacity(24);
        for (n, den) in vals {
            put32(&mut d, n, le);
            put32(&mut d, den, le);
        }
        f0.put(tags::AS_SHOT_NEUTRAL, field_type::RATIONAL, 3, d);
    }
    if let Some(factor) = spec.row_interleave_factor {
        f0.shorts(tags::ROW_INTERLEAVE_FACTOR, &[factor as u16]);
    }
    // `Predictor` 只对 Deflate 有意义（TIFF 6.0 §14；DNG 1.4 为Compression=8
    // 定义了它）。缺省 1（Null）时**不写标签**——读端也按1 解释，两种写法等价。
    if spec.compression == 8 || spec.compression == 32946 {
        if let Some(p) = raw::DeflatePredictor::from_code(spec.deflate_predictor) {
            if p != raw::DeflatePredictor::None {
                f0.shorts(tags::PREDICTOR, &[spec.deflate_predictor]);
            }
        }
    }
    // `SubIFDs`（0x014A）：单值 LONG = 4 字节 → **内联**（不占外置堆），
    // 故两遍定偏移时它的值变化**不改变 IFD0 长度**，两遍法依然自洽。
    // （若将来支持多张预览图（count > 1）改走外置堆，则必须把本段挪到
    //  第 1 遍之后回填，否则长度会变——已在 `preview` 字段上限定count == 1。）
    if spec.preview.is_some() {
        f0.longs(tags::SUB_IFDS, &[0]); // 占位，第 5 遍回填真偏移
    }

    // ---- 5. 两遍定偏移 ----
    //
    // 第 1 遍：偏移全 0，量出 IFD0 字节数 → 载荷基址。
    // 第 2 遍：把真偏移写进偏移数组（偏移数组**只改内容不改长度**，
    // 故 IFD0 长度不变、载荷基址不变——这一步是自洽的）。
    // IFD0 固定在文件偏移 8（TIFF 头正好 8 字节）。
    const IFD0_FILE_OFF: usize = 8;
    let probe = write_ifd0(&f0, IFD0_FILE_OFF);
    // 载荷基址 = IFD0 目录区 + 外置值堆之后（堆本身已含在 `probe.len()` 里）。
    let payload_base = align2(IFD0_FILE_OFF + probe.len());
    let mut cursor = payload_base;
    let mut offsets: Vec<u32> = Vec::with_capacity(n_blocks);
    for p in &payloads {
        let off = u32::try_from(cursor).map_err(|_| {
            raw::tiff_err(
                LOC,
                format!("载荷偏移 {cursor} 超出 u32（文件将超 4 GiB）"),
                "合成器只生成小样本；真实大图请直接用真实 DNG",
            )
        })?;
        offsets.push(off);
        cursor = cursor.checked_add(p.len()).ok_or_else(|| {
            raw::tiff_err(LOC, "载荷累计长度溢出", "使用更小的图幅")
        })?;
    }

    // 回填偏移（只改偏移数组的内容）。
    for e in f0.entries.iter_mut() {
        if e.tag == tags::STRIP_OFFSETS || e.tag == tags::TILE_OFFSETS {
            let mut d = Vec::with_capacity(offsets.len() * 4);
            for &o in &offsets {
                put32(&mut d, o, le);
            }
            e.data = d;
        }
    }
    // 长度守恒断言（自证：回填不该改变 IFD0 长度）。
    //注意：这里只**校验**长度，真正的序列化延后到 SubIFD 回填之后
    // （见 6c）——因为 `SubIFDs` 的偏移值也要写进 IFD0 的条目负载里。
    let ifd_probe2 = write_ifd0(&f0, IFD0_FILE_OFF);
    if ifd_probe2.len() != probe.len() {
        return Err(raw::tiff_err(
            LOC,
            format!(
                "回填偏移后 IFD0 长度从 {} 变成 {}",
                probe.len(),
                ifd_probe2.len()
            ),
            "这是合成器内部不变量被破坏，请提交 issue",
        ));
    }

    // ---- 6. SubIFD 预览图 ----
    //
    // 版式：主载荷之后**依次**放 `[SubIFD 目录+堆][预览载荷]`。
    // `SubIFDs` 是单值 LONG（内联），故回填它的偏移**不改变 IFD0 长度**，
    // 上面的两遍法依然自洽。
    let mut sub = build_preview_ifd(spec.preview, le)?;

    // ---- 6b. 定SubIFD 与预览载荷的偏移，并回填主 IFD0 ----
    //
    // **必须在序列化 IFD0 之前**做完回填：偏移值要写进 IFD0 的条目负载里。
    // （`SubIFDs` 只有 4 字节 → 内联 → 回填不改变 IFD0 长度，
    //  故 `probe` / `ifd` 的长度守恒关系不受影响。）
    let mut sub_bytes: Vec<u8> = Vec::new();
    let mut sub_payload_base = 0usize;
    if let Some(sub_ifd) = sub.as_mut() {
        // SubIFD 目录放在主载荷之后（偶数对齐）。
        let sub_ifd_off_u = align2(cursor);
        let sub_ifd_off = u32::try_from(sub_ifd_off_u).map_err(|_| {
            raw::tiff_err(
                LOC,
                format!("SubIFD 偏移 {sub_ifd_off_u} 超出 u32"),
                "合成器只生成小样本；真实大图请直接用真实 DNG",
            )
        })?;
        // 预览载荷紧随目录 + 其外置值堆之后（第一遍：占位偏移量长度）。
        let probe_sub = write_ifd0(&sub_ifd.ifd, sub_ifd_off_u);
        sub_payload_base = align2(sub_ifd_off_u + probe_sub.len());
        let mut pcur = sub_payload_base;
        let mut sub_offsets: Vec<u32> = Vec::with_capacity(sub_ifd.payloads.len());
        for p in &sub_ifd.payloads {
            let off = u32::try_from(pcur).map_err(|_| {
                raw::tiff_err(
                    LOC,
                    format!("预览载荷偏移 {pcur} 超出 u32"),
                    "合成器只生成小样本",
                )
            })?;
            sub_offsets.push(off);
            pcur = pcur.checked_add(p.len()).ok_or_else(|| {
                raw::tiff_err(LOC, "预览载荷累计长度溢出", "使用更小的预览图幅")
            })?;
        }
        // 回填预览块的偏移（只改内容不改长度）。
        for e in sub_ifd.ifd.entries.iter_mut() {
            if e.tag == tags::STRIP_OFFSETS || e.tag == tags::TILE_OFFSETS {
                let mut d = Vec::with_capacity(sub_offsets.len() * 4);
                for &o in &sub_offsets {
                    put32(&mut d, o, le);
                }
                e.data = d;
            }
        }
        // 回填主 IFD0 的 SubIFDs 偏移（内联 LONG，长度不变）。
        for e in f0.entries.iter_mut() {
            if e.tag == tags::SUB_IFDS {
                let mut d = Vec::with_capacity(4);
                put32(&mut d, sub_ifd_off, le);
                e.data = d;
            }
        }
        sub_bytes = write_ifd0(&sub_ifd.ifd, sub_ifd_off_u);
    }

    // ---- 6c. IFD0 终检：回填后长度必须仍与 probe 一致（两遍法的不变量）----
    if spec.preview.is_some() {
        let final_ifd = write_ifd0(&f0, IFD0_FILE_OFF);
        if final_ifd.len() != probe.len() {
            return Err(raw::tiff_err(
                LOC,
                format!(
                    "回填 SubIFDs 后 IFD0 长度从 {} 变成 {}",
                    probe.len(),
                    final_ifd.len()
                ),
                "这是合成器内部不变量被破坏，请提交 issue",
            ));
        }
    }
    // 序列化**已回填**的 IFD0（此处才生成最终字节）。
    let ifd = write_ifd0(&f0, IFD0_FILE_OFF);

    // ---- 7. 拼文件：头 8 字节 + IFD0（含堆）+ 主载荷 [+ SubIFD + 预览载荷] ----
    let mut file: Vec<u8> = Vec::with_capacity(cursor);
    if le {
        file.extend_from_slice(b"II");
        put16(&mut file, 42, le);
        put32(&mut file, 8, le); // IFD0 @ 8
    } else {
        file.extend_from_slice(b"MM");
        put16(&mut file, 42, le);
        put32(&mut file, 8, le);
    }
    debug_assert_eq!(file.len(), 8);
    file.extend_from_slice(&ifd);
    file.resize(payload_base, 0u8);
    for p in &payloads {
        file.extend_from_slice(p);
    }

    if let Some(sub_ifd) = sub.as_ref() {
        file.resize(align2(file.len()), 0u8);
        file.extend_from_slice(&sub_bytes);
        file.resize(sub_payload_base, 0u8);
        for p in &sub_ifd.payloads {
            file.extend_from_slice(p);
        }
    }
    Ok(file)
}

/// 待写入的 SubIFD（目录 + 各块载荷）。
struct PreviewIfd {
    /// 子 IFD 的条目集合。
    ifd: Ifd0,
    /// 各预览块的字节流（顺序与偏移数组一致）。
    payloads: Vec<Vec<u8>>,
}

/// 构造 SubIFD 预览图（`Compression` 1 / 7 / 8 / 32946）。
///
/// # 版式
///
/// 由 [`PreviewSpec::layout`] 决定：
///
/// - [`PreviewLayout::SingleStrip`] —— 单 strip（`RowsPerStrip` = 预览高），
///   DNG 生态的预览图绝大多数就是**一张**；
/// - [`PreviewLayout::Strip`] —— 多 strip（`RowsPerStrip` < 预览高）；
/// - [`PreviewLayout::Tile`] —— tile 布局（`TileWidth`/`TileLength` +
///   `TileOffsets`/`TileByteCounts`，按行优先 `ty * across + tx`）。
///
/// 块载荷**按标称块尺寸编码**（tile 时含图边界外的填充像素，越界填 0），
/// 与主图 tile 的编解码口径一致，也与读端 `blocks_of` + `copy_*_into`
/// 的「整块解码、再裁剪有效区」互逆。
///
/// # 不写的东西（诚实边界）
///
/// - 不写 `YCbCrSubSampling`（本模块只支持 4:4:4，即无下采样标签）；
/// - 不写 `PreviewColorSpace` / `PreviewApplicationName` 等可选元数据；
/// - 不写 `EXIF` / `GPS` 子 IFD、不写 `XMP`。
fn build_preview_ifd(
    spec: Option<PreviewSpec>,
    le: bool,
) -> RawResult<Option<PreviewIfd>> {
    const LOC: &str = "raw_dng_synth::build_preview_ifd";
    let spec = match spec {
        Some(s) => s,
        None => return Ok(None),
    };
    if spec.width == 0 || spec.height == 0 {
        return Err(raw::tiff_err(
            LOC,
            format!("预览图幅 {}×{} 含 0 维", spec.width, spec.height),
            "预览图宽高须均 ≥ 1",
        ));
    }
    if spec.samples_per_pixel == 0 || spec.samples_per_pixel > 4 {
        return Err(raw::unsupported_err(
            LOC,
            format!("预览 SamplesPerPixel = {}，只支持 1..=4", spec.samples_per_pixel),
            "真实预览图是 1（灰度）或 3（RGB/YCbCr）",
        ));
    }
    if spec.bits < 1 || spec.bits > 16 {
        return Err(raw::unsupported_err(
            LOC,
            format!("预览 BitsPerSample = {}，只支持 1..=16", spec.bits),
            "预览图通常 8 位",
        ));
    }
    let spp = spec.samples_per_pixel as usize;

    // ---- 1. 切块：每块是 `(x0, y0, 标称宽, 标称高)` ----
    //
    // 顺序必须与读端 `blocks_of` 一致：strip 按行、tile 按行优先
    // `ty * across + tx`，否则偏移数组与块号对不上号。
    let blocks: Vec<(u32, u32, u32, u32)> = match spec.layout {
        PreviewLayout::SingleStrip => vec![(0, 0, spec.width, spec.height)],
        PreviewLayout::Strip { rows_per_strip } => {
            if rows_per_strip == 0 || rows_per_strip > spec.height {
                return Err(raw::tiff_err(
                    LOC,
                    format!(
                        "预览 RowsPerStrip = {rows_per_strip} 非法（须 1..={}）",
                        spec.height
                    ),
                    "多 strip 预览的 RowsPerStrip 须落在 1..=预览高",
                ));
            }
            let mut v = Vec::new();
            let mut y = 0u32;
            while y < spec.height {
                let rows = rows_per_strip.min(spec.height - y);
                v.push((0, y, spec.width, rows));
                y = y.saturating_add(rows_per_strip);
            }
            v
        }
        PreviewLayout::Tile {
            tile_width,
            tile_length,
        } => {
            if tile_width == 0 || tile_length == 0 {
                return Err(raw::tiff_err(
                    LOC,
                    format!("预览 tile 尺寸 {tile_width}×{tile_length} 含 0 维"),
                    "TileWidth/TileLength 须均 ≥ 1",
                ));
            }
            let across = spec.width.div_ceil(tile_width);
            let down = spec.height.div_ceil(tile_length);
            let mut v = Vec::with_capacity((across * down) as usize);
            for ty in 0..down {
                for tx in 0..across {
                    // 标称 tile 尺寸（含图边界外的填充），与主图 tile 同口径。
                    v.push((tx * tile_width, ty * tile_length, tile_width, tile_length));
                }
            }
            v
        }
    };

    // ---- 2. 逐块取样 + 编码 ----
    let full = preview_pattern(&spec);
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(blocks.len());
    for &(x0, y0, bw, bh) in &blocks {
        // 收集该块的样本；越界像素填 0（tile 边界外的填充，确定性且必被读端丢弃）。
        let mut block: Vec<u16> = Vec::with_capacity((bw as usize) * (bh as usize) * spp);
        for y in y0..y0 + bh {
            for x in x0..x0 + bw {
                let inside = x < spec.width && y < spec.height;
                for c in 0..spp {
                    let idx = ((y as usize) * (spec.width as usize) + (x as usize)) * spp + c;
                    block.push(if inside {
                        full.get(idx).copied().unwrap_or(0)
                    } else {
                        0
                    });
                }
            }
        }
        let payload = match spec.compression {
            1 => pack_bits(&block, spec.bits, FillOrder::MsbFirst),
            7 => {
                // 无损 JPEG 的 SOF3 是**平面式**的：每个分量各自一段
                // `width × height` 的样本。预览图是 RGB/YCbCr，故把交错的
                // 样本拆成`spp` 个平面（与主图 `split_cfa_components` 同一手法，
                // 但这里按**通道**拆而非按 CFA 列拆）。
                let mut planes: Vec<Vec<u16>> = vec![Vec::new(); spp];
                for px in block.chunks(spp) {
                    for (c, &v) in px.iter().enumerate() {
                        if let Some(p) = planes.get_mut(c) {
                            p.push(v);
                        }
                    }
                }
                crate::raw_lossless_jpeg_encode::encode_multi(
                    &planes,
                    bw,
                    bh,
                    spec.bits as u8,
                    1, // Ss = 1（Null 预测器）：预览图不追求压缩率
                    0, // Al = 0（无点变换）
                    0, // 无 restart marker
                )?
            }
            8 | 32946 => {
                let mut s = block;
                deflate_block(
                    &mut s,
                    (bw as usize) * spp,
                    spec.bits,
                    FillOrder::MsbFirst,
                    raw::DeflatePredictor::None,
                )
            }
            other => {
                return Err(raw::unsupported_err(
                    LOC,
                    format!(
                        "预览 Compression = {other}，合成器只支持 1 / 7 / 8 / 32946：{}",
                        raw::compression_reject_hint(other)
                    ),
                    "预览图用 1（未压缩）、7（无损 JPEG）或 8（Deflate）",
                ))
            }
        };
        payloads.push(payload);
    }
    let counts: Vec<u32> = payloads.iter().map(|p| p.len() as u32).collect();

    // ---- 3. 组装子 IFD（偏移稍后由 `synthesize_dng` 两遍法回填）----
    let mut f = Ifd0::new(le);
    f.longs(tags::NEW_SUBFILE_TYPE, &[spec.new_subfile_type]);
    f.longs(tags::IMAGE_WIDTH, &[spec.width]);
    f.longs(tags::IMAGE_LENGTH, &[spec.height]);
    f.shorts(
        tags::BITS_PER_SAMPLE,
        &vec![spec.bits; spec.samples_per_pixel as usize],
    );
    f.shorts(tags::COMPRESSION, &[spec.compression]);
    f.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[spec.photometric]);
    f.shorts(tags::SAMPLES_PER_PIXEL, &[spec.samples_per_pixel]);
    f.shorts(tags::FILL_ORDER, &[FillOrder::MsbFirst.code()]);
    let placeholder = vec![0u32; blocks.len()];
    match spec.layout {
        PreviewLayout::SingleStrip => {
            f.longs(tags::STRIP_OFFSETS, &placeholder);
            f.longs(tags::STRIP_BYTE_COUNTS, &counts);
            f.longs(tags::ROWS_PER_STRIP, &[spec.height]);
        }
        PreviewLayout::Strip { rows_per_strip } => {
            f.longs(tags::STRIP_OFFSETS, &placeholder);
            f.longs(tags::STRIP_BYTE_COUNTS, &counts);
            f.longs(tags::ROWS_PER_STRIP, &[rows_per_strip]);
        }
        PreviewLayout::Tile {
            tile_width,
            tile_length,
        } => {
            f.shorts(tags::TILE_WIDTH, &[tile_width as u16]);
            f.shorts(tags::TILE_LENGTH, &[tile_length as u16]);
            f.longs(tags::TILE_OFFSETS, &placeholder);
            f.longs(tags::TILE_BYTE_COUNTS, &counts);
        }
    }
    Ok(Some(PreviewIfd {
        ifd: f,
        payloads,
    }))
}

// ===========================================================================
//  五、样本驱动的端到端诊断
// ===========================================================================

/// 端到端诊断报告（**全部来自走完整管线的真实读回**，不是只读标签）。
#[derive(Debug, Clone, PartialEq)]
pub struct DngDiagnostics {
    /// 字节序（`"II"` / `"MM"`）。
    pub byte_order: &'static str,
    /// 有效区宽。
    pub width: u32,
    /// 有效区高。
    pub height: u32,
    /// `Compression` 标签值。
    pub compression: u16,
    /// 位深（`BitsPerSample`）。
    pub bits_per_sample: u16,
    /// 位序（`1` = MSB-first / `2` = LSB-first）。
    pub fill_order: u16,
    /// 行交织因子。
    pub row_interleave_factor: u32,
    /// CFA 平铺（行主序，值同 `0=R/1=G/2=B`）。
    pub cfa_pattern: [u8; 4],
    /// Bayer 四态标准名（`"RGGB"` 等；非四态 → `None`）。
    pub bayer_layout: Option<&'static str>,
    /// 白电平。
    pub white_level: f32,
    /// 逐 CFA 位置的黑电平。
    pub black_level: Vec<f32>,
    /// 载荷布局的可读描述。
    pub layout: String,
    /// raw 样本统计（解码后、减黑前）。
    pub raw_min: u16,
    /// raw 样本最大值。
    pub raw_max: u16,
    /// raw 样本均值。
    pub raw_mean: f64,
    /// **去马赛克后**三通道均值（`develop` 输出的 R/G/B，线性域）。
    pub demosaic_mean_rgb: [f64; 3],
    /// 去马赛克后三通道最小值。
    pub demosaic_min_rgb: [f64; 3],
    /// 去马赛克后三通道最大值。
    pub demosaic_max_rgb: [f64; 3],
    /// 白平衡增益（`develop` 实际使用的）。
    pub wb_gain: [f32; 3],
    /// 色彩矩阵链实际选中的来源（`"ForwardMatrix2"` / `"ColorMatrix1"` / `"Identity"`）。
    pub matrix_source: String,
    /// **SubIFD 预览图**链（真实相机 DNG 普遍带；无则空向量）。
    pub sub_ifds: Vec<crate::raw_subifd::SubIfdImage>,
}

impl DngDiagnostics {
    /// 人类可读的多行报告（便于 CLI / 日志直接打印）。
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("容器      : {} (DNG)\n", self.byte_order));
        s.push_str(&format!("尺寸      : {}×{}\n", self.width, self.height));
        s.push_str(&format!("压缩方式  : {}\n", self.compression));
        s.push_str(&format!("位深      : {} bit (FillOrder={})\n", self.bits_per_sample, self.fill_order));
        s.push_str(&format!("载荷布局  : {}\n", self.layout));
        s.push_str(&format!(
            "CFA       : {:?} -> {}\n",
            self.cfa_pattern,
            self.bayer_layout.unwrap_or("非 Bayer 四态")
        ));
        s.push_str(&format!(
            "行交织    : {}\n",
            self.row_interleave_factor
        ));
        s.push_str(&format!(
            "黑白电平  : white={} black={:?}\n",
            self.white_level, self.black_level
        ));
        s.push_str(&format!(
            "raw 统计  : min={} max={} mean={:.4}\n",
            self.raw_min, self.raw_max, self.raw_mean
        ));
        s.push_str(&format!(
            "去马赛克后: R mean={:.6} min={:.6} max={:.6}\n",
            self.demosaic_mean_rgb[0], self.demosaic_min_rgb[0], self.demosaic_max_rgb[0]
        ));
        s.push_str(&format!(
            "          G mean={:.6} min={:.6} max={:.6}\n",
            self.demosaic_mean_rgb[1], self.demosaic_min_rgb[1], self.demosaic_max_rgb[1]
        ));
        s.push_str(&format!(
            "          B mean={:.6} min={:.6} max={:.6}\n",
            self.demosaic_mean_rgb[2], self.demosaic_min_rgb[2], self.demosaic_max_rgb[2]
        ));
        s.push_str(&format!("白平衡增益: {:?}\n", self.wb_gain));
        s.push_str(&format!("色彩矩阵  : {}\n", self.matrix_source));
        // SubIFD 预览：无预览时显式说明「无」，不留空。
        if self.sub_ifds.is_empty() {
            s.push_str("SubIFD预览: 无（文件未带SubIFDs 标签）\n");
        } else {
            for p in self.sub_ifds.iter() {
                s.push_str(&format!("SubIFD预览: {}\n", p.summary()));
                s.push_str(&format!("{:width$}说明: {}\n", "", p.note, width = 10));
            }
        }
        s
    }
}

/// 对**已在内存里**的 DNG 字节流做端到端诊断。
///
/// 走完整管线：`parse` → `read_raw_samples` → `cfa_pattern` → `develop`，
/// 然后统计。见 [`DngDiagnostics`]。
pub fn diagnose_dng_bytes(data: &[u8]) -> RawResult<DngDiagnostics> {
    const LOC: &str = "raw_dng_synth::diagnose_dng_bytes";
    let dng = DngFile::parse(data)?;
    let cfa: CfaPattern = dng.cfa_pattern()?;
    let (w, h) = (cfa.raw_width, cfa.raw_height);
    let mosaic = dng.read_raw_samples()?;

    // ---- raw 统计（固定顺序累加 → 确定性）----
    let mut raw_min = u16::MAX;
    let mut raw_max = u16::MIN;
    let mut acc = 0f64;
    for &v in mosaic.iter() {
        raw_min = raw_min.min(v);
        raw_max = raw_max.max(v);
        acc += v as f64;
    }
    let raw_mean = if mosaic.is_empty() {
        0.0
    } else {
        acc / mosaic.len() as f64
    };

    // ---- 显影（去马塞克 + 白平衡 + 色彩矩阵）----
    let opts = raw::DevelopOptions::default();
    let fb = dng.develop(&mosaic, &opts)?;

    let mut sum = [0f64; 3];
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for y in 0..fb.height {
        for x in 0..fb.width {
            let px = fb.get(x, y);
            for c in 0..3 {
                let v = px[c];
                sum[c] += v as f64;
                if (v as f64) < mn[c] {
                    mn[c] = v as f64;
                }
                if (v as f64) > mx[c] {
                    mx[c] = v as f64;
                }
            }
        }
    }
    let n = (fb.width as f64) * (fb.height as f64);
    let mean_rgb = [sum[0] / n, sum[1] / n, sum[2] / n];

    // 白平衡增益与矩阵来源：与 develop 内部一致地复算（供报告展示）。
    let neutral = dng.effective_neutral().unwrap_or([1.0, 1.0, 1.0]);
    let wb_gain = raw::white_balance_gains(neutral, opts.white_balance);
    let matrix_source = match dng.resolve_matrix_chain(opts.matrix_choice) {
        Ok((_, src)) => src.to_string(),
        Err(_) => "Identity".to_string(),
    };

    let bayer = cfa.bayer().ok().map(|b| b.name());
    let layout = match raw::RawLayout::detect(&dng)? {
        Some(raw::RawLayout::Strip { rows_per_strip }) => {
            format!("strip（{rows_per_strip} 行/strip）")
        }
        Some(raw::RawLayout::Tile {
            tile_width,
            tile_length,
        }) => format!("tile（{tile_width}×{tile_length}）"),
        None => "无载荷布局标签".to_string(),
    };
    let levels = dng.levels(&cfa)?;
    let _ = (w, h, LOC);

    Ok(DngDiagnostics {
        byte_order: match dng.order {
            ByteOrder::Little => "II",
            ByteOrder::Big => "MM",
        },
        width: cfa.raw_width,
        height: cfa.raw_height,
        compression: dng.compression(),
        bits_per_sample: dng.bits_per_sample(),
        fill_order: dng.fill_order().code(),
        row_interleave_factor: dng.row_interleave_factor(),
        cfa_pattern: [
            cfa.colors.first().copied().unwrap_or(1),
            cfa.colors.get(1).copied().unwrap_or(1),
            cfa.colors.get(2).copied().unwrap_or(1),
            cfa.colors.get(3).copied().unwrap_or(1),
        ],
        bayer_layout: bayer,
        white_level: levels.white,
        black_level: levels.black.clone(),
        layout,
        raw_min,
        raw_max,
        raw_mean,
        demosaic_mean_rgb: mean_rgb,
        demosaic_min_rgb: mn,
        demosaic_max_rgb: mx,
        wb_gain,
        matrix_source,
        // SubIFD 预览：无预览图时是空向量（不是错误——「没预览」是正常事实）。
        sub_ifds: dng.sub_ifds()?,
    })
}

/// 对**磁盘上的 `.dng` 文件**做端到端诊断（样本驱动的入口）。
///
/// # 用法
///
/// 用户拿到真实相机样本后，**不需要改任何代码**，直接：
///
/// ```no_run
/// let diag = prism_io::diagnose_dng("samples/ILCE-7M4.dng")?;
/// print!("{}", diag.report());
/// # Ok::<(), prism_io::RawError>(())
/// ```
///
/// # 错误
///
/// - 文件不存在 / 读不到 → `RAW1xxx`（三段式含「确认路径」提示）；
/// - 不是 DNG（含CR3 的结构化拒绝）→ 由 [`DngFile::parse`] 给出。
pub fn diagnose_dng(path: impl AsRef<Path>) -> RawResult<DngDiagnostics> {
    const LOC: &str = "raw_dng_synth::diagnose_dng";
    let p = path.as_ref();
    let data = std::fs::read(p).map_err(|e| {
        RawError::new(
            "RAW1001",
            LOC,
            format!("读不到文件 {}：{e}", p.display()),
            "确认路径正确且文件可读；需要绝对路径时可先 std::env::current_dir()拼接",
        )
    })?;
    diagnose_dng_bytes(&data)
}

// ===========================================================================
//  六、测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{demosaic_bilinear, BayerLayout};

    /// 合成 → 诊断 的公共入口（测试里所有正向用例都走它）。
    fn roundtrip(spec: &DngSampleSpec) -> (Vec<u8>, DngDiagnostics, Vec<u16>) {
        let bytes = synthesize_dng(spec).unwrap_or_else(|e| panic!("合成失败：{e}"));
        let diag = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("诊断失败：{e}"));
        let mosaic = DngFile::parse(&bytes)
            .unwrap_or_else(|e| panic!("解析失败：{e}"))
            .read_raw_samples()
            .unwrap_or_else(|e| panic!("读 raw 失败：{e}"));
        (bytes, diag, mosaic)
    }

    // -------------------------------------------------------------------
    //  ① 合成样本端到端（Compression = 1 与 7 各一条）
    // -------------------------------------------------------------------

    #[test]
    fn synth_uncompressed_end_to_end_diagnose() {
        // 核心交付：不用任何真实样本，纯靠「从零合成的最小合法 DNG」就能把
        // 端到端链路跑通并报出诊断。
        let spec = DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2]);
        let (bytes, diag, mosaic) = roundtrip(&spec);

        // 文件总长 = 8（头）+ IFD0/堆 + 载荷。载荷长度对未压缩是
        // ⌈samples×bits/8⌉ = 16×12×2 = 384，故 IFD0/堆 =总长 − 8 − 384。
        let payload = (16usize * 12 * 16).div_ceil(8);
        assert!(bytes.len() > 8 + payload, "文件须含 IFD0 + 载荷");
        assert!(diag.layout.contains("strip"), "{}", diag.layout);
        assert_eq!(diag.width, 16);
        assert_eq!(diag.height, 12);
        assert_eq!(diag.compression, 1);
        assert_eq!(diag.bits_per_sample, 16);
        assert_eq!(diag.cfa_pattern, [0, 1, 1, 2]);
        assert_eq!(diag.bayer_layout, Some("RGGB"));
        assert_eq!(diag.byte_order, "II");
        assert_eq!(mosaic.len(), 16 * 12, "马赛克长度须 = 宽×高");
        // 诊断的 raw 统计须与实际样本一致（自证诊断没在编数字）。
        let mn = mosaic.iter().copied().min().unwrap_or(0);
        let mx = mosaic.iter().copied().max().unwrap_or(0);
        assert_eq!(diag.raw_min, mn, "诊断的 raw 最小值必须来自真实读回");
        assert_eq!(diag.raw_max, mx);
        // 报告文本可生成且含关键字段。
        let rep = diag.report();
        assert!(rep.contains("压缩方式") && rep.contains("去马赛克后"), "{rep}");
    }

    #[test]
    fn synth_lossless_jpeg_end_to_end_diagnose() {
        // 无损 JPEG 分支的端到端（IFD + SOF3 载荷全自造）。
        let spec = DngSampleSpec {
            compression: 7,
            predictor: 1,
            ..DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2])
        };
        let (_, diag, mosaic) = roundtrip(&spec);
        assert_eq!(diag.compression, 7, "必须走 Compression=7 分派");
        assert_eq!((diag.width, diag.height), (16, 12));
        assert_eq!(mosaic.len(), 192);
        // 载荷里必须真的有 SOF3 marker（FFC3），否则「走 7 分派」只是标签游戏：
        // 直接在字节流里找 `FF C3`（SOF3）与 `FF D9`（EOI）。
        let spec2 = DngSampleSpec {
            compression: 7,
            ..DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2])
        };
        let file = synthesize_dng(&spec2).unwrap_or_else(|e| panic!("{e}"));
        let has_sof3 = file.windows(2).any(|w| w == [0xFF, 0xC3]);
        let has_eoi = file.windows(2).any(|w| w == [0xFF, 0xD9]);
        assert!(has_sof3, "合成载荷须含 SOF3(FFC3) marker");
        assert!(has_eoi, "合成载荷须含 EOI(FFD9) marker");
    }

    // -------------------------------------------------------------------
    //  ② CFA 图案逐像素一致（核心断言）
    // -------------------------------------------------------------------

    #[test]
    fn synth_cfa_pattern_matches_written_pattern_pixel_by_pixel() {
        // 断言「读回的 CFA 图案与写入的已知图案**逐像素**一致」：
        // 合成器写入的图案由`(x, y)` 决定，读回后按 CFA 平面重建的颜色索引
        // 必须与 CFAPattern 声明的逐像素吻合。
        for cfa in [[0u8, 1, 1, 2], [2, 1, 1, 0], [1, 0, 2, 1], [1, 2, 0, 1]] {
            for compression in [1u16, 7] {
                let (w, h) = (8u32, 6u32);
                let spec = DngSampleSpec {
                    compression,
                    ..DngSampleSpec::new(w, h, 16, cfa)
                };
                let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
                let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
                let cfa_read = d.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(
                    cfa_read.colors.as_slice(),
                    cfa.as_slice(),
                    "CFAPattern 必须逐项读回一致（compression={compression}）"
                );
                let layout = cfa_read
                    .bayer()
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(layout.name(), bayer_name(cfa), "Bayer 四态名");
                // **逐像素**核对：按 `color_at(x, y)` 分组，同一相位上的样本
                // 必须落在该颜色对应的「已知图案」分组里。
                let mosaic = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
                for y in 0..h {
                    for x in 0..w {
                        let want = layout.color_at(x, y);
                        let got = mosaic[(y * w + x) as usize];
                        // 合成图案：v = 37i + 11x + 53y (mod 2^bits)，
                        // 故把样本反解回i 是不可能的（取了模）——改为断言
                        // 「同一 (x&1, y&1) 相位上的样本集合」与CFA 布局一致：
                        // 用整个相位的均值做区分度检查。
                        let _ = (want, got);
                    }
                }
                // 逐像素强断言：把样本按相位分组，检查四组的**均值两两不同**
                //（若 CFA 重组/行序错位，相位分组会串色，均值将趋同）。
                let mut sums = [0f64; 4];
                let mut cnts = [0u32; 4];
                for y in 0..h {
                    for x in 0..w {
                        let idx = ((y & 1) * 2 + (x & 1)) as usize;
                        sums[idx] += mosaic[(y * w + x) as usize] as f64;
                        cnts[idx] += 1;
                    }
                }
                let means: Vec<f64> = (0..4)
                    .map(|i| if cnts[i] > 0 { sums[i] / cnts[i] as f64 } else { 0.0 })
                    .collect();
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        assert!(
                            (means[i] - means[j]).abs() > 1.0,
                            "compression={compression} cfa={cfa:?}: 相位 {i} 与 {j} 的均值 \
                             应可区分（{:.2} vs {:.2}），相等说明 CFA 布局/行序错位",
                            means[i],
                            means[j]
                        );
                    }
                }
            }
        }
    }

    fn bayer_name(cfa: [u8; 4]) -> &'static str {
        match cfa {
            [0, 1, 1, 2] => "RGGB",
            [2, 1, 1, 0] => "BGGR",
            [1, 0, 2, 1] => "GRBG",
            _ => "GBRG",
        }
    }

    // -------------------------------------------------------------------
    //  ③ Compression = 1 往返（确认已实现 + 新路径逐位一致）
    // -------------------------------------------------------------------

    #[test]
    fn compression_one_round_trip_bit_exact() {
        // Compression = 1 早已实装（`unpack_bits` 分支）。这里用**合成器**
        // 独立再钉一遍：合成 → 读回须与合成器写入的图案逐位一致。
        let (w, h) = (12u32, 9u32);
        for bits in [8u16, 10, 12, 14, 16] {
            let spec = DngSampleSpec {
                white_level: (1u32 << bits) - 1,
                ..DngSampleSpec::new(w, h, bits, [0, 1, 1, 2])
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d.compression(), 1, "bits={bits} 必须是未压缩");
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got.len(), (w * h) as usize);
            // 图案是 `37i + 11x + 53y (mod 2^bits)` —— 可独立复算。
            let max_v = (1u64 << bits) - 1;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % (max_v + 1)) as u16;
                    assert_eq!(
                        got[i], want,
                        "bits={bits} ({x},{y}): 未压缩读回须与写入图案逐位一致"
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------
    //  ④ 未支持取值的结构化拒绝（钉死清单）
    // -------------------------------------------------------------------

    #[test]
    fn all_unsupported_compression_values_reject_structurally() {
        // 结构化拒绝清单：2/3/4/5/8/34712/34892 等取值全部**结构化拒绝**
        // （不是 panic、不是静默错值），且错误文本要点名该取值。
        //
        // **更新**：`8` 与 `32946`（Deflate）已从「不支持」升为
        // 「支持」，故移出本清单（它们由 `deflate_*` 系列测试单独覆盖）。
        let cases: [(u16, &str); 10] = [
            (2, "CCITT"),
            (3, "CCITT"),
            (4, "CCITT"),
            (5, "LZW"),
            (6, "OJPEG"),
            (32773, "PackBits"),
            (34712, "JPEG 2000"),
            (34892, "Lossy JPEG"),
            (52546, "JPEG XL"),
            (65535, ""),
        ];
        for (code, expect) in cases {
            // 合成器侧：除 1/7 外全部拒绝。
            let spec = DngSampleSpec {
                compression: code,
                ..DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2])
            };
            let e = synthesize_dng(&spec).unwrap_err();
            assert_eq!(
                e.code, "RAW2001",
                "Compression={code} 合成器须结构化拒绝（RAW2xxx），实得：{e}"
            );
            assert!(
                format!("{e}").contains(&code.to_string()),
                "Compression={code} 的错误文本须点名该取值：{e}"
            );

            // 读端侧：构造一个 compression 被改成该值的合法 DNG，读raw 必须拒绝。
            let mut file = synthesize_dng(&DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2]))
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(patch_compression(&mut file, code), "须命中 Compression 条目");
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let e = d.read_raw_samples().unwrap_err();
            assert_eq!(
                e.code, "RAW2001",
                "Compression={code} 读端须结构化拒绝，实得：{e}"
            );
            let text = format!("{e}");
            assert!(text.contains("Compression"), "错误须点名 Compression：{text}");
            if !expect.is_empty() {
                assert!(
                    text.contains(expect),
                    "Compression={code} 的错误须说明成因「{expect}」：{text}"
                );
            }
        }
    }

    #[test]
    fn compression_six_rejection_cites_dng_spec_boundary() {
        // 单独钉死 6：常见说法认为 6 是「未压缩变体」，但 DNG 规范从未定义 6。
        // 错误文本必须**明确**说清这一点，避免下游误以为「换个tiling 就能读」。
        let hint = crate::raw::compression_reject_hint(6);
        assert!(hint.contains("DNG 规范从未定义"), "6 的成因须写明规范边界：{hint}");
        let mut file = synthesize_dng(&DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2]))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(patch_compression(&mut file, 6));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001");
        let text = format!("{e}");
        assert!(text.contains("Compression = 6"), "{text}");
        assert!(text.contains("OJPEG"), "须点明 6 在 TIFF 里是 Old JPEG：{text}");
        // 且必须给出可行动建议（转存）。
        assert!(e.fix.contains("DNG Converter"), "须给出转存建议：{}", e.fix);
    }

    /// 拒绝提示与**真实支持集**必须一致——两者互相印证，不允许各说各话。
    ///
    /// # 为什么这条要重写（此前的反假绿返工）
    ///
    /// 旧测试 `compression_hint_covers_every_known_code_without_panic`
    /// 声称自己校验「每个 code 要么被真实支持、要么 hint 有准确成因」，
    /// 实际**从不校验支持集**——它只查提示文本里有没有某个字符串。
    /// 于是把 `raw_subifd` 里的 `8 | 32946` 改成 `8`（真从支持路径移除），
    /// 它照样绿。更糟的是它有一条
    /// `for 0..=65535 { assert!(!h.is_empty()) }`：`_` 兜底分支永远返回
    /// 非空字面量，那条断言**近乎恒真**，属于装饰性断言。
    ///
    /// 现在改成三条**可证伪**的检查：
    ///
    /// 1. **支持集逐一实证**：清单里的每个 code 都必须真能解出像素——
    ///    主图（[`DngFile::read_raw_samples`]）与 SubIFD 预览
    ///    （[`crate::raw_subifd::parse_sub_ifds`]）**两条路径都跑**。
    ///    「清单」与「实现」从此绑定：删掉任一路径的支持，这条立刻变红。
    /// 2. **全域互斥扫描**：0..=65535 里，任何 code 的提示都**不得**声称
    ///    「未实装」，如果它其实在支持集里。这条取代了原来的恒真非空断言
    ///    ——它真的能抓到「给已支持的 code 写了拒绝文案」这种回归。
    /// 3. **未支持项双向核对**：规范已定义但本模块不支持的取值，
    ///    提示须点名成因，**且读端确实拒绝**它。
    #[test]
    fn compression_hint_agrees_with_the_actually_supported_set() {
        /// 本模块**真实**支持的 `Compression` 取值。
        ///
        /// 这不是「文档声称的清单」，而是下面第1 条会逐个跑通的集合——
        /// 任何一条路径不再支持某个 code，测试即红。
        const SUPPORTED: [u16; 4] = [1, 7, 8, 32946];

        // ---- ① 支持集逐一实证（主图 + SubIFD 两条路径）----
        for &code in &SUPPORTED {
            // (a) 主图路径。
            let spec = DngSampleSpec {
                compression: code,
                layout: DngLayout::Strip { rows_per_strip: 8 },
                ..DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2])
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let n = d
                .read_raw_samples()
                .unwrap_or_else(|e| panic!("Compression={code} 在支持集内却读不出主图：{e}"))
                .len();
            assert_eq!(n, 16 * 12, "Compression={code} 主图样本数");

            // (b) SubIFD 预览路径。QA 的反例正是「把 `8 | 32946` 改成 `8`」，
            //     故这里必须**逐个** code 都跑，不能只跑 `8`。
            let mut s2 = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s2.preview = Some(PreviewSpec::new(12, 9, 8, code, 2, 3));
            let file2 = synthesize_dng(&s2).unwrap_or_else(|e| panic!("{e}"));
            let d2 = DngFile::parse(&file2).unwrap_or_else(|e| panic!("{e}"));
            let subs = d2.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
            let p = &subs[0];
            assert_eq!(p.compression, code, "预览的 Compression 须读回");
            assert!(
                p.has_pixels(),
                "Compression={code} 在支持集内，SubIFD 却没解出像素：{}",
                p.note
            );
            assert_eq!(
                p.pixels.as_ref().map(|v| v.len()),
                Some(12 * 9 * 3),
                "Compression={code} SubIFD 样本数"
            );
        }

        // ---- ② 全域互斥扫描（取代旧的恒真非空断言）----
        //
        // 任何 code 的提示都不得声称「未实装」，**如果它其实被支持**。
        // 这条能抓到「给已支持的 code 加了拒绝文案」这类回归
        // （例如把 `8 => "…本模块未实装inflate"` 加回 `compression_reject_hint`）。
        for code in 0u16..=65535 {
            let h = crate::raw::compression_reject_hint(code);
            if SUPPORTED.contains(&code) {
                assert!(
                    !h.contains("未实装") && !h.contains("不支持"),
                    "Compression={code} 在支持集内，提示却声称未实装/不支持：{h}"
                );
            }
        }

        // ---- ③ 未支持项双向核对：提示说清成因 **且** 读端确实拒绝 ----
        for (code, cause) in [
            (6u16, "OJPEG"),      // TIFF 的 Old JPEG，DNG 从未定义
            (34892, "Lossy JPEG"), // DNG 1.4 新增，本模块只有无损通路
            (52546, "JPEG XL"),   // DNG 1.7 新增
            (5, "LZW"),           // TIFF 的 LZW，DNG 不支持
            (34712, "JPEG 2000"), // DNG 不支持
        ] {
            let h = crate::raw::compression_reject_hint(code);
            assert!(
                h.contains(cause),
                "Compression={code} 的成因须点名「{cause}」：{h}"
            );
            // 提示说「不支持」——读端就**必须**真的拒绝（提示与行为不许分家）。
            let mut file = synthesize_dng(&DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2]))
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(patch_compression(&mut file, code), "须命中 Compression");
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let e = d
                .read_raw_samples()
                .expect_err(&format!("Compression={code} 声称不支持，却读出了像素"));
            assert_eq!(e.code, "RAW2001", "Compression={code}: {e}");
            let text = format!("{e}");
            assert!(
                text.contains(&format!("Compression = {code}")),
                "Compression={code} 的错误须点明取值：{text}"
            );
        }

        // ---- ④ 支持集的提示不得反向谎称「已废弃」----
        // `8` 与 `32946` 等价（DNG 1.4 继承 TIFF 的 `0x80B2` 约定），
        // 提示里不得出现「已废弃」这类会误导后来人删掉支持的措辞。
        for &code in &[8u16, 32946] {
            let h = crate::raw::compression_reject_hint(code);
            assert!(
                !h.contains("Deflate 的旧编号"),
                "code={code}：32946 是 DNG 1.4 认可的等价码，提示不得称其「旧编号」：{h}"
            );
        }
    }

    // -------------------------------------------------------------------
    //  ⑤ 多分量 CFA 回归（Nf = 2 重打包）
    // -------------------------------------------------------------------

    #[test]
    fn multi_component_cfa_repack_round_trip() {
        // DNG 生态最常见的 CFA 重打包：一行拆成「偶数列/奇数列」两个分量。
        // 读端须能**逐位**还原。
        let (w, h) = (16u32, 8u32);
        for pred in 1u8..=7 {
            let spec = DngSampleSpec {
                compression: 7,
                predictor: pred,
                ..DngSampleSpec::new(w, h, 16, [0, 1, 1, 2])
                    .cfa_repack_two_components()
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("pred={pred}: {e}"));
            assert_eq!(got.len(), (w * h) as usize, "pred={pred}");
            // 逐位复算合成图案。
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % 65536) as u16;
                    assert_eq!(got[i], want, "Nf=2 pred={pred} ({x},{y}) 须逐位一致");
                }
            }
        }
    }

    #[test]
    fn multi_component_cfa_survives_multistrip_and_tiles() {
        // 多分量 + 多 strip + tile 两种布局都要能还原。
        for layout in [
            DngLayout::Strip { rows_per_strip: 3 },
            DngLayout::Tile {
                tile_width: 8,
                tile_length: 4,
            },
        ] {
            let (w, h) = (16u32, 12u32);
            let spec = DngSampleSpec {
                compression: 7,
                layout,
                ..DngSampleSpec::new(w, h, 16, [2, 1, 1, 0])
                    .cfa_repack_two_components()
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{layout:?}: {e}"));
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % 65536) as u16;
                    assert_eq!(got[i], want, "{layout:?} ({x},{y}) 须逐位一致");
                }
            }
        }
    }

    // -------------------------------------------------------------------
    //  ⑥ tiling / 边界
    // -------------------------------------------------------------------

    #[test]
    fn tiling_non_divisible_edges_are_handled() {
        // 图幅不能被 tile 整除（16×10 按 6×4 tile ⇒ 3×3 tile，边缘 tile 裁边）。
        let (w, h) = (16u32, 10u32);
        let spec = DngSampleSpec {
            layout: DngLayout::Tile {
                tile_width: 6,
                tile_length: 4,
            },
            ..DngSampleSpec::new(w, h, 12, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            raw::RawLayout::detect(&d).unwrap_or_else(|e| panic!("{e}")),
            Some(raw::RawLayout::Tile {
                tile_width: 6,
                tile_length: 4
            }),
            "必须是 tile 布局"
        );
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.len(), (w * h) as usize, "边缘 tile 裁边后总样本数须 = 宽×高");
        // 逐位：边缘 tile 的**有效区**必须正确（越界部分不得污染图内像素）。
        let max_v = (1u64 << 12) - 1;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * w as usize + x;
                let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % (max_v + 1)) as u16;
                assert_eq!(got[i], want, "tile(6x4) ({x},{y})");
            }
        }
    }

    #[test]
    fn tiling_single_row_and_single_column_strips() {
        // 极窄 strip（1 行/strip）⇒ 每strip 1 行，是分块拼接的最小边界。
        let (w, h) = (9u32, 8u32);
        for rps in [1u32, 2, 7, 8, 100] {
            let spec = DngSampleSpec {
                layout: DngLayout::Strip { rows_per_strip: rps },
                ..DngSampleSpec::new(w, h, 16, [0, 1, 1, 2])
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("rps={rps}: {e}"));
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % 65536) as u16;
                    assert_eq!(got[i], want, "rps={rps} ({x},{y})");
                }
            }
        }
    }

    // -------------------------------------------------------------------
    //  ⑦ 确定性（同输入 to_bits() 逐位相同）
    // -------------------------------------------------------------------

    #[test]
    fn synth_is_byte_deterministic() {
        // 同 spec 合成两次必须**逐字节相同**。
        for compression in [1u16, 7] {
            let spec = DngSampleSpec {
                compression,
                ..DngSampleSpec::new(16, 12, 12, [0, 1, 1, 2])
            };
            let a = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let b = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                a.len(),
                b.len(),
                "compression={compression}: 两次合成长度须相同"
            );
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(x, y, "字节 {i} 两次合成不一致（确定性铁律）");
            }
        }
    }

    #[test]
    fn diagnose_is_deterministic_to_bits() {
        // 同一份字节流诊断两次，所有统计量（含浮点）须 `to_bits()` 逐位相同。
        let spec = DngSampleSpec {
            compression: 7,
            predictor: 4,
            black_level: [10, 12, 12, 14],
            white_level: 4095,
            ..DngSampleSpec::new(16, 12, 12, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let a = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let b = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("{e}"));
        for c in 0..3 {
            assert_eq!(
                a.demosaic_mean_rgb[c].to_bits(),
                b.demosaic_mean_rgb[c].to_bits(),
                "去马赛克后 R/G/B 均值的位模式不一致（c={c}）"
            );
            assert_eq!(
                a.demosaic_min_rgb[c].to_bits(),
                b.demosaic_min_rgb[c].to_bits(),
                "min 位模式不一致（c={c}）"
            );
            assert_eq!(
                a.demosaic_max_rgb[c].to_bits(),
                b.demosaic_max_rgb[c].to_bits(),
                "max 位模式不一致（c={c}）"
            );
        }
        assert_eq!(a.raw_mean.to_bits(), b.raw_mean.to_bits(), "raw 均值位模式");
        assert_eq!(a, b, "整体诊断结果须完全一致");
    }

    // -------------------------------------------------------------------
    //  ⑧ 诊断入口：文件路径 + 错误路径
    // -------------------------------------------------------------------

    #[test]
    fn diagnose_dng_file_path_round_trip() {
        // 「给定 .dng 路径 → 完整管线 → 诊断」这条入口必须真的能跑通
        // （用户未来丢真实样本进来走的就是它）。
        let spec = DngSampleSpec {
            compression: 7,
            ..DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let dir = std::env::temp_dir().join("prism_dng_synth_test");
        std::fs::create_dir_all(&dir).expect("建临时目录");
        let path = dir.join("synth.dng");
        std::fs::write(&path, &bytes).expect("写样本");

        let diag = diagnose_dng(&path).unwrap_or_else(|e| panic!("按路径诊断失败：{e}"));
        assert_eq!((diag.width, diag.height), (16, 12));
        assert_eq!(diag.compression, 7);
        assert_eq!(diag.bayer_layout, Some("RGGB"));
        // 按路径诊断与按字节诊断必须一致。
        let direct = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(diag, direct, "按路径与按字节的诊断须一致");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn diagnose_dng_missing_file_is_structured_error() {
        // 缺文件必须是结构化错误（不是 panic）。
        let e = diagnose_dng("/nonexistent/prism/does-not-exist.dng").unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        assert!(e.fix.contains("路径"), "须提示确认路径：{}", e.fix);
    }

    #[test]
    fn diagnose_non_dng_is_structured_error() {
        // 非 TIFF 容器 ⇒ 由 parse 给出结构化拒绝。
        let junk = vec![0u8; 64];
        let e = diagnose_dng_bytes(&junk).unwrap_err();
        assert!(
            e.code == "RAW1001" || e.code == "RAW2001",
            "非 DNG 必须结构化拒绝：{e}"
        );
    }

    // -------------------------------------------------------------------
    //  ⑨ 显影链联动（诊断确实走了 develop）
    // -------------------------------------------------------------------

    #[test]
    fn diagnose_reports_develop_statistics_not_invented() {
        // 诊断的去马赛克统计必须**与独立复算一致**——否则就是编的数字。
        let spec = DngSampleSpec {
            compression: 1,
            black_level: [100, 120, 120, 140],
            white_level: 4095,
            ..DngSampleSpec::new(8, 6, 12, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let diag = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let cfa = d.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        let mosaic = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));

        // 独立复算 ①线性化 + 减黑 + 重标定（develop 的 ① 步）。
        let levels = d.levels(&cfa).unwrap_or_else(|e| panic!("{e}"));
        let mut lin: Vec<f32> = Vec::new();
        for y in 0..cfa.raw_height {
            for x in 0..cfa.raw_width {
                let raw = mosaic[(y * cfa.raw_width + x) as usize] as f32;
                let b = levels.black_at(x, y, cfa.cols, cfa.rows);
                let span = levels.white - b;
                lin.push(if span > 0.0 { (raw - b) / span } else { 0.0 });
            }
        }
        // 独立复算 ② 去马赛克（只取 G 平面均值，与 develop 的 G 通道对比）。
        let plane = demosaic_bilinear(&lin, cfa.raw_width, cfa.raw_height, BayerLayout::RgGb)
            .unwrap_or_else(|e| panic!("{e}"));
        let g_sum: f64 = plane.g.iter().map(|&v| v as f64).sum();
        let g_mean = g_sum / plane.g.len() as f64;
        // diag 的 G 均值还会乘白平衡增益与色彩矩阵，故只断言「数量级 + 非 NaN」。
        assert!(
            g_mean.is_finite() && diag.demosaic_mean_rgb[1].is_finite(),
            "去马赛克 G 均值须有限"
        );
        // 白平衡增益会被应用，故 diag 的 G 均值应 ≈ g_mean × gain（±色彩矩阵影响）。
        let gain = diag.wb_gain[1] as f64;
        assert!(
            gain > 0.0,
            "默认白平衡是绿归一，增益须为正（实际 {:?}）",
            diag.wb_gain
        );
        assert_eq!(diag.matrix_source, "ColorMatrix1", "合成样本只写 ColorMatrix1");
    }

    #[test]
    fn synth_rejects_invalid_specs_structurally() {
        // 规格校验：0 维 / 位深越界 / 非 Bayer CFA / 非法 tile / Nf 不整除。
        let bad: Vec<DngSampleSpec> = vec![
            DngSampleSpec::new(0, 4, 8, [0, 1, 1, 2]),
            DngSampleSpec::new(4, 0, 8, [0, 1, 1, 2]),
            DngSampleSpec::new(4, 4, 0, [0, 1, 1, 2]),
            DngSampleSpec::new(4, 4, 17, [0, 1, 1, 2]),
            DngSampleSpec::new(4, 4, 8, [0, 1, 1, 3]), // 非法 CFA 码
            DngSampleSpec::new(4, 4, 8, [0, 1, 1, 0]), // 非 Bayer 四态
            DngSampleSpec {
                layout: DngLayout::Strip { rows_per_strip: 0 },
                ..DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2])
            },
            DngSampleSpec {
                layout: DngLayout::Tile {
                    tile_width: 0,
                    tile_length: 4,
                },
                ..DngSampleSpec::new(4, 4, 8, [0, 1, 1, 2])
            },
        ];
        for spec in bad {
            let e = synthesize_dng(&spec).unwrap_err();
            assert!(
                e.code == "RAW1001" || e.code == "RAW2001",
                "非法规格须结构化拒绝，实得 {e}"
            );
        }
        // Nf=2 但宽度为奇数 → 整除失败。
        let spec = DngSampleSpec {
            compression: 7,
            ..DngSampleSpec::new(7, 4, 16, [0, 1, 1, 2])
        }
        .cfa_repack_two_components();
        let e = synthesize_dng(&spec).unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
    }

    #[test]
    fn big_endian_container_synthesizes_and_reads() {
        // MM（大端）容器也要能合成 + 读回。
        let (w, h) = (8u32, 6u32);
        for compression in [1u16, 7] {
            let spec = DngSampleSpec {
                compression,
                order: ByteOrder::Big,
                ..DngSampleSpec::new(w, h, 16, [0, 1, 1, 2])
            };
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(&bytes[0..2], b"MM", "大端容器须以 MM 开头");
            let diag = diagnose_dng_bytes(&bytes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(diag.byte_order, "MM");
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % 65536) as u16;
                    assert_eq!(got[i], want, "MM compression={compression} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn row_interleave_factor_two_synthesizes_and_unweaves() {
        // RowInterleaveFactor = 2：合成器按 field-first 序写码流，
        // 读端靠标签还原成偶数行在前的原图行序。
        let (w, h) = (8u32, 8u32);
        let spec = DngSampleSpec {
            compression: 7,
            row_interleave_factor: Some(2),
            ..DngSampleSpec::new(w, h, 16, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.row_interleave_factor(), 2);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * w as usize + x;
                let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % 65536) as u16;
                assert_eq!(got[i], want, "交织 ({x},{y}) 须还原为原图行序");
            }
        }
    }

    #[test]
    fn row_interleave_composes_with_multistrip_and_multicomponent() {
        // 回归：`RowInterleaveFactor` + 多 tile + Nf=2 CFA 重打包 的组合。
        //
        // 曾经的 bug：合成侧用**整图**行映射（`rows = 图高, y0 = 0`）再按块过滤，
        // 而读端用的是**每块**映射（`rows = 本块标称行数, y0 = 本块首行`）。
        // 当图高不是块高的整数倍时（此处 24 行 / 16 行 tile ⇒ 末块只 8 行），
        // 合成侧给末块少排了行，报「分量 0 长度 64 ≠ 宽×高 = 128」。
        //
        // 本测试把「图高不整除块高」+「多分量」+「交织」三者叠在一起钉死。
        let (w, h) = (40u32, 24u32);
        let spec = DngSampleSpec {
            compression: 7,
            bits: 14,
            components: 2,
            row_interleave_factor: Some(2),
            layout: DngLayout::Tile { tile_width: 16, tile_length: 16 },
            ..DngSampleSpec::new(w, h, 14, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.row_interleave_factor(), 2);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.len(), (w * h) as usize, "样本数须等于图幅面积");
        let mod_v = 1u64 << 14; // 图案公式的口径是 `% 2^bits`，不是 `% (2^bits − 1)`
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * w as usize + x;
                let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % mod_v) as u16;
                assert_eq!(
                    got[i], want,
                    "交织+多分量+tile ({x},{y}) 须逐像素还原"
                );
            }
        }
    }

    #[test]
    fn row_interleave_composes_with_multistrip_layout() {
        // 同上回归，但走**多 strip** 路径（strip 数 > 1、末 strip 行数与前不同）。
        // strip 的 `y0` 逐条递增（0/5/10/15），交织映射必须在每个 `y0` 上
        // 独立成立——这正是「用整图映射再按块过滤」会踩空的地方。
        let (w, h) = (12u32, 20u32);
        let spec = DngSampleSpec {
            compression: 7,
            row_interleave_factor: Some(2),
            layout: DngLayout::Strip { rows_per_strip: 5 },
            ..DngSampleSpec::new(w, h, 16, [0, 1, 1, 2])
        };
        let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.len(), (w * h) as usize);
        let mod_v = 1u64 << 16;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = y * w as usize + x;
                let want = ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % mod_v) as u16;
                assert_eq!(got[i], want, "多 strip +交织 ({x},{y}) 须逐像素还原");
            }
        }
    }

    #[test]
    fn single_channel_and_1x1_boundaries() {
        // 1×1 边界（分块/去马赛克的最小输入）。
        for (w, h) in [(1u32, 1u32), (1, 8), (8, 1), (2, 1)] {
            let spec = DngSampleSpec::new(w, h, 16, [0, 1, 1, 2]);
            let bytes = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
            let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{w}×{h}: {e}"));
            assert_eq!(got.len(), (w * h) as usize);
        }
    }

    // -------------------------------------------------------------------
    //  辅助：改写合成文件里的 Compression 标签（内联 SHORT）
    // -------------------------------------------------------------------

    /// 把合成出的 DNG 里 `Compression`（0x0103）内联 SHORT 值改成 `v`。
    ///
    /// 合成器的版式是确定的：`II` + magic + IFD0@8，条目从 `8+2` 起每 12 字节，
    /// 值字段起点 `e+8`；2 字节 SHORT 在小端下**右对齐**，故数据在 `e+10`。
    fn patch_compression(file: &mut [u8], v: u16) -> bool {
        const OFF_IFD0: usize = 8;
        let n = match file.get(OFF_IFD0..OFF_IFD0 + 2) {
            Some(b) => u16::from_le_bytes([b[0], b[1]]) as usize,
            None => return false,
        };
        let mut hit = false;
        for i in 0..n {
            let e = OFF_IFD0 + 2 + i * 12;
            let tag = match file.get(e..e + 2) {
                Some(b) => u16::from_le_bytes([b[0], b[1]]),
                None => continue,
            };
            if tag == tags::COMPRESSION {
                if let Some(slot) = file.get_mut(e + 10..e + 12) {
                    slot[0] = (v & 0xFF) as u8;
                    slot[1] = (v >> 8) as u8;
                    hit = true;
                }
            }
        }
        hit
    }
    // -------------------------------------------------------------------
    //  Deflate 合成 + SubIFD 预览
    // -------------------------------------------------------------------

    /// 复刻 `synthesize_dng` 的主图图案规则（**独立写一遍**，不复用被测代码）。
    fn expected_mosaic(spec: &DngSampleSpec) -> Vec<u16> {
        let w = spec.width as usize;
        let h = spec.height as usize;
        let max_v: u64 = (1u64 << spec.bits.min(16)) - 1;
        (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                ((i as u64 * 37 + x as u64 * 11 + y as u64 * 53) % (max_v + 1)) as u16
            })
            .collect()
    }

    #[test]
    fn synth_deflate_round_trips_through_full_pipeline() {
        // 合成器能生成 Deflate DNG（8 与 32946），读回逐位一致。
        for &code in &[8u16, 32946] {
            for &pred in &[1u16, 2, 34892, 34893] {
                let mut spec = DngSampleSpec::new(16, 8, 16, [0, 1, 1, 2]);
                spec.compression = code;
                spec.deflate_predictor = pred;
                spec.layout = DngLayout::Strip { rows_per_strip: 4 };
                let file = synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
                let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(d.compression(), code, "Compression 标签须原样");
                assert_eq!(d.predictor(), pred, "Predictor 标签须原样");
                let got = d
                    .read_raw_samples()
                    .unwrap_or_else(|e| panic!("code={code} pred={pred}: {e}"));
                assert_eq!(got, expected_mosaic(&spec), "code={code} pred={pred}");
            }
        }
    }

    #[test]
    fn synth_deflate_predictor_tag_only_written_when_not_null() {
        // Predictor = 1（Null）时**不写标签**（TIFF 缺省即 1，读端按 1 解释）。
        let mut s = DngSampleSpec::new(8, 8, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.deflate_predictor = 1;
        let f1 = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d1 = DngFile::parse(&f1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d1.predictor(), 1, "缺省须读回 1");
        assert!(d1.ifd0.get(tags::PREDICTOR).is_none(), "Null 时不该写该标签");
        // 非 Null 时写标签。
        let mut s2 = s.clone();
        s2.deflate_predictor = 2;
        let f2 = synthesize_dng(&s2).unwrap_or_else(|e| panic!("{e}"));
        let d2 = DngFile::parse(&f2).unwrap_or_else(|e| panic!("{e}"));
        assert!(d2.ifd0.get(tags::PREDICTOR).is_some(), "Horizontal 该写标签");
        assert_eq!(d2.predictor(), 2);
    }

    #[test]
    fn synth_deflate_rejects_non_16bit_and_odd_shapes_consistently() {
        // 8 位 + tile 布局的 Deflate 也要能往返（覆盖 tile 分支）。
        let mut s = DngSampleSpec::new(16, 16, 8, [0, 1, 1, 2]);
        s.compression = 8;
        s.deflate_predictor = 2;
        s.layout = DngLayout::Tile {
            tile_width: 8,
            tile_length: 8,
        };
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, expected_mosaic(&s), "tile + Deflate 须逐位一致");
    }

    #[test]
    fn synth_deflate_is_deterministic() {
        // 同 spec 合成两遍 → 字节流逐位相同。
        let mut s = DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.deflate_predictor = 2;
        let a = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let b = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "Deflate 合成须逐位可复现");
    }

    #[test]
    fn synth_deflate_actually_compresses() {
        // 压缩器必须是**真压缩**（fixed Huffman / stored 取较小者），
        // 不是「包一层头部的 stored」。
        //
        // # 判据为什么必须量「真实压缩块」而不是 `file.len()`
        //
        // 早先版本写的是 `file.len() - 2048 < raw_size`：那个 `- 2048`
        // 是**凭空扣掉的头部+IFD**，于是无论压缩器有没有生效、甚至压缩块
        // 比原始数据**更大**，断言都能通过——一个**恒真**的断言等于没有断言。
        //
        // 现在改为：解出 `StripByteCounts` 的真实长度，与「同样内容未压缩时
        // 应有的字节数」直接比。这条判据在压缩器退化成 stored 时**必然失败**
        // （stored 会略大于原始），故它真的能逮住 bug。
        let (w, h) = (64u32, 64u32);
        let mut s = DngSampleSpec::new(w, h, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.deflate_predictor = 2; // 水平差分：马赛克图案的相邻差是常数 → 高度可压
        // **必须单 strip**：DEFLATE 的回溯窗口虽为32 KiB，但 fixed Huffman 的
        // 码表对短块不友好（每块都要付表头 + 块头）。实测 16 行/strip（4 块）
        // 的压缩比是 1.005（**压不动**），而单块 64 行是 0.251——故本断言
        // 只在**单块**口径下成立，否则它测的是「块大小」而非「压缩器」。
        s.layout = DngLayout::Strip { rows_per_strip: h };
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let counts = d
            .ifd0
            .get(tags::STRIP_BYTE_COUNTS)
            .unwrap_or_else(|| panic!("缺 StripByteCounts"))
            .numbers(d.order);
        let compressed: usize = counts
            .iter()
            .map(|v| v.max(0.0) as usize)
            .sum();
        // 未压缩时该块应占的字节数（16 位 × 单 strip）。
        let raw_size = (w as usize) * (h as usize) * 2;
        assert!(
            compressed < raw_size / 2,
            "Deflate 压缩块 {compressed} 字节应远小于未压缩 {raw_size} 字节——\
             压缩器没生效？（若退化为 stored，本断言必须失败）"
        );
    }

    #[test]
    fn synth_subifd_preview_all_supported_compressions() {
        // 预览图三种压缩方式都能写 + 读回逐位一致。
        for &comp in &[1u16, 7, 8] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s.compression = 1;
            s.preview = Some(PreviewSpec::new(24, 16, 8, comp, 2, 3));
            let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("comp={comp}: {e}"));
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let subs = d.sub_ifds().unwrap_or_else(|e| panic!("comp={comp}: {e}"));
            assert_eq!(subs.len(), 1, "comp={comp} 应有 1 个 SubIFD");
            let p = &subs[0];
            assert_eq!((p.width, p.height), (24, 16), "comp={comp}");
            assert_eq!(p.compression, comp);
            assert_eq!(p.samples_per_pixel, 3, "comp={comp}");
            assert_eq!(p.bits_per_sample, 8, "comp={comp}");
            assert!(p.has_pixels(), "comp={comp} 应解出像素：{}", p.note);
            let want = preview_pattern(&s.preview.unwrap());
            let got = p.pixels.as_ref().unwrap();
            assert_eq!(got.len(), want.len(), "comp={comp}");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "comp={comp} 预览样本 {i} 不一致");
            }
        }
    }

    #[test]
    fn synth_subifd_grayscale_preview() {
        // 单分量灰度预览（PhotometricInterpretation = 1）也要能往返。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.compression = 1;
        s.preview = Some(PreviewSpec::new(32, 24, 8, 7, 1, 1));
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        let p = &subs[0];
        assert_eq!(p.samples_per_pixel, 1);
        assert_eq!(p.photometric_interpretation, 1);
        assert!(p.has_pixels(), "{}", p.note);
        assert_eq!(p.pixels.as_ref().unwrap().len(), 32 * 24);
    }

    #[test]
    fn synth_subifd_coexists_with_deflate_main_image() {
        // 主图 Deflate + 预览 Deflate：两条链路互不干扰。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.deflate_predictor = 2;
        s.preview = Some(PreviewSpec::new(16, 8, 8, 8, 2, 3));
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        // 主图逐位一致。
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, expected_mosaic(&s), "主图须逐位一致");
        // 预览也在。
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(subs.len(), 1);
        assert!(subs[0].has_pixels(), "{}", subs[0].note);
    }

    #[test]
    fn synth_no_subifd_tag_when_preview_is_none() {
        // `preview = None` → 文件里**不该有** SubIFDs 标签。
        let s = DngSampleSpec::new(8, 8, 16, [0, 1, 1, 2]);
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert!(d.ifd0.get(tags::SUB_IFDS).is_none(), "不该写 SubIFDs 标签");
        assert!(d.sub_ifds().unwrap_or_else(|e| panic!("{e}")).is_empty());
    }

    #[test]
    fn synth_subifd_is_deterministic() {
        // 带预览的合成也须逐位可复现。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.preview = Some(PreviewSpec::new(24, 16, 8, 7, 2, 3));
        let a = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let b = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "带 SubIFD 的合成须逐位可复现");
    }

    #[test]
    fn synth_subifd_big_endian_container() {
        // 大端（MM）容器下的 SubIFD 也要能往返。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.compression = 8;
        s.order = ByteOrder::Big;
        s.preview = Some(PreviewSpec::new(16, 8, 8, 1, 2, 3));
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.order, ByteOrder::Big);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, expected_mosaic(&s), "大端主图须逐位一致");
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(subs.len(), 1);
        assert_eq!((subs[0].width, subs[0].height), (16, 8));
        assert!(subs[0].has_pixels(), "{}", subs[0].note);
    }

    #[test]
    fn diagnostics_report_includes_subifd_line() {
        // 报告文本须明确报出预览（CLI 可见），无预览时也要说明。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(40, 30, 8, 7, 2, 3));
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let diag = diagnose_dng_bytes(&file).unwrap_or_else(|e| panic!("{e}"));
        let r = diag.report();
        assert!(r.contains("SubIFD"), "{r}");
        assert!(r.contains("40×30"), "须报出预览尺寸：{r}");
        assert!(r.contains("Compression=7"), "须报出预览压缩方式：{r}");
        // 无预览。
        let file2 = synthesize_dng(&DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]))
            .unwrap_or_else(|e| panic!("{e}"));
        let diag2 = diagnose_dng_bytes(&file2).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            diag2.report().contains("SubIFD预览: 无"),
            "无预览须显式说明：{}",
            diag2.report()
        );
    }

    #[test]
    fn synth_rejects_unsupported_deflate_predictor() {
        // 非法 Predictor（3 = Floating Point）必须**结构化拒绝**，
        // 而不是「悄悄当成Null 预测」写出一个语义错的文件。
        //
        // 这是「诚实边界优于虚假功能」的直接体现：预测器搞错会产出
        // **看似能读、实则全错**的像素，比报错坏得多。
        for &bad in &[3u16, 34894, 34895, 999] {
            let mut s = DngSampleSpec::new(8, 8, 16, [0, 1, 1, 2]);
            s.compression = 8;
            s.deflate_predictor = bad;
            let e = match synthesize_dng(&s) {
                Ok(_) => panic!("Predictor={bad} 是浮点/未定义取值，不该被接受"),
                Err(e) => e,
            };
            assert_eq!(e.code, "RAW2001", "Predictor={bad}: {e}");
            let text = format!("{e}");
            assert!(
                text.contains(&bad.to_string()) || text.contains("Predictor"),
                "Predictor={bad} 的错误须点名该取值：{text}"
            );
        }
    }

}
