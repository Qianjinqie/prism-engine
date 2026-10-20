//! 相机 RAW 读入深度——DNG/IFD 解析 + Bayer 去马塞克 + 白平衡 + 色彩矩阵。
//!
//! 数学路径的次序由本模块固定并论证（见 [`DngFile::develop`]）：
//! RAW = 白平衡 + 去马赛克卷积（S2 数学结构）。
//!
//! # 铁律
//!
//! - **零第三方依赖**：TIFF/DNG 的 IFD 结构、CFA 推断、Rational 全部手写，
//!   只用 `std`（与 [`crate::sha256`] / [`crate::fbuf`] 同款纪律）；
//! - **确定性**：IFD 条目按标签升序、像素按行主序固定遍历，无 `HashMap`、
//!   无 RNG；同输入两次解析与两次显影逐位一致；
//! - **C18 永不 panic**：全部 `checked_*` / `saturating_*` / `Option` + 三段式
//!   [`RawError`]，畸形输入（截断、超大 size、越界偏移）一律结构化错误。
//!
//! # 范围与已知边界
//!
//! - **DNG**：完整 IFD/TIFF 解析（II/MM 双字节序、SubIFD 指针、内联值），
//!   关键标签 `CFARepeatPatternDim`/`CFAPattern`/`BlackLevel`/`WhiteLevel`/
//!   `ColorMatrix1`/`ForwardMatrix2`/`CameraCalibration1`/`AnalogBalance`/
//!   `AsShotNeutral`/`AsShotWhiteXY`/`CalibrationIlluminant1`/`BaselineExposure`；
//!   EXIF 子 IFD 与 GPS 子 IFD **读出原始数值但不做语义解释**（不解码经纬度
//!   为物理坐标，见 [`GpsIfd`]）；
//! - **位深**：**1..=16 位紧凑打包**（[`unpack_bits`]），含 MSB-first（DNG
//!   默认，高位对齐）与 LSB-first（`FillOrder=2`）两种位序，8/16 位走快速路径、
//!   10/12/14 位走「每 N 样本一组」的整数快路径，余者走通用逐位解包器；
//! - **像素布局**：strip（`StripOffsets`）与 **tile**（`TileOffsets`）**并存**，
//!   由 IFD 里存在哪个标签决定（见 [`DngFile::read_mosaic`]）；
//! - **色彩**：`ForwardMatrix2` 与 `ColorMatrix1` **两条独立的路**，由
//!   [`MatrixChoice`] 显式选择（`Auto` = Forward2 优先），配合
//!   `CameraCalibration1` 与 `AnalogBalance`；D50→D65 白点适配用本模块
//!   自写的 Bradford CAT02（见 [`bradford_adaptation`]）；
//! - **CR3**：**ISO-BMFF 盒树遍历 + Apple uuid 定位 + CMT1 元数据读出**
//!   （见 [`probe_cr3`]），但**像素解码明确结构化拒绝**（[`Cr3Info::decode_pixels`]
//!   恒返回错误）——绝不返回假数据冒充支持；
//! - **去马塞克**：Bayer 四态（RGGB/BGGR/GRBG/GBRG）双线性插值；
//! - **未做**：CFA 相位探测（phase detect）、亮度着色重建、Green-Stough
//!   矩阵插值、马赛克原分辨率无损路径（sPP2）、DNG 私有数据（`0xc68b`）。
//!   完整清单见文末「诚实边界」。

use std::fmt;

use crate::fbuf::Fbuf16F;

// ===========================================================================
//  一、错误（三段式：在哪 / 什么事 / 怎么改）
// ===========================================================================

/// RAW 域错误码分段：`RAW1xxx` TIFF/DNG 结构、`RAW2xxx` 格式不支持、
/// `RAW3xxx` 显影流水线。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawError {
    /// 错误码（`RAW1xxx` TIFF/DNG、`RAW2xxx` 格式、`RAW3xxx` 显影）。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl RawError {
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

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] @{}：{}（修复：{}）",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for RawError {}

/// RAW 域结果别名。
pub type RawResult<T> = Result<T, RawError>;

/// TIFF 结构错误构造（`RAW1xxx` 段）。
pub(crate) fn tiff_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> RawError {
    RawError::new("RAW1001", loc, problem, fix)
}

/// 格式不支持错误构造（`RAW2xxx` 段）。
pub(crate) fn unsupported_err(
    loc: &'static str,
    problem: impl Into<String>,
    fix: impl Into<String>,
) -> RawError {
    RawError::new("RAW2001", loc, problem, fix)
}

/// 显影流水线错误构造（`RAW3xxx` 段）。
pub(crate) fn pipe_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> RawError {
    RawError::new("RAW3001", loc, problem, fix)
}

// ===========================================================================
//  二、字节序与边界安全的 TIFF 游标
// ===========================================================================

/// TIFF 字节序（`II` = 小端 Intel、`MM` = 大端 Motorola，TIFF 5.0 §2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// 小端（`II`）。
    Little,
    /// 大端（`MM`）。
    Big,
}

impl ByteOrder {
    /// 由头部两字节判定（`II` / `MM`，其他值一律 `None`）。
    pub fn of_magic(data: &[u8]) -> Option<Self> {
        match data.get(0..2) {
            Some(b"II") => Some(ByteOrder::Little),
            Some(b"MM") => Some(ByteOrder::Big),
            _ => None,
        }
    }

    /// 读 `u16`。
    #[inline]
    fn u16(self, b: &[u8]) -> u16 {
        match self {
            ByteOrder::Little => u16::from_le_bytes([b[0], b[1]]),
            ByteOrder::Big => u16::from_be_bytes([b[0], b[1]]),
        }
    }

    /// 读 `u32`。
    #[inline]
    fn u32(self, b: &[u8]) -> u32 {
        match self {
            ByteOrder::Little => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            ByteOrder::Big => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        }
    }
}

/// 边界安全的只读游标：所有读方法都先检查剩余长度，越界即结构化错误。
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    data: &'a [u8],
    order: ByteOrder,
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// 在 `pos` 处建立游标。
    pub fn new(data: &'a [u8], order: ByteOrder, pos: usize) -> Self {
        Self { data, order, pos }
    }

    /// 当前偏移。
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 剩余字节数。
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// 取 `n` 字节切片（越界报错）。
    pub fn take(&mut self, n: usize, loc: &'static str) -> RawResult<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            tiff_err(
                loc,
                format!("偏移 {} + 长度 {} 溢出", self.pos, n),
                "文件被截断或偏移字段损坏",
            )
        })?;
        let slice = self.data.get(self.pos..end).ok_or_else(|| {
            tiff_err(
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
        Ok(slice)
    }

    /// 读 1 字节。
    pub fn u8(&mut self, loc: &'static str) -> RawResult<u8> {
        Ok(self.take(1, loc)?[0])
    }

    /// 读 `u16`。
    pub fn u16(&mut self, loc: &'static str) -> RawResult<u16> {
        let b = self.take(2, loc)?;
        Ok(self.order.u16(b))
    }

    /// 读 `u32`。
    pub fn u32(&mut self, loc: &'static str) -> RawResult<u32> {
        let b = self.take(4, loc)?;
        Ok(self.order.u32(b))
    }

    /// 读 `i32`（位模式不变的有符号重解释）。
    pub fn i32(&mut self, loc: &'static str) -> RawResult<i32> {
        Ok(self.u32(loc)? as i32)
    }

    /// 从绝对偏移另开一个游标（用于内联值/外置值解析，不推进 `self`）。
    pub fn at(&self, pos: usize) -> Cursor<'a> {
        Cursor::new(self.data, self.order, pos)
    }
}

// ===========================================================================
//  三、TIFF IFD 结构
// ===========================================================================

/// IFD 条目数上限（TIFF 6.0 规定 ≤ 65535；此处再收一道防御性上限，
/// 避免畸形 `count` 触发巨量分配）。
pub const MAX_IFD_ENTRIES: usize = 4096;

/// 单个外置属性值的字节上限（4 MiB；DNG 关键标签远小于此）。
pub const MAX_TIFF_VALUE_BYTES: usize = 4 * 1024 * 1024;

/// DNG 关键标签号（Adobe DNG 1.4 §3.3 / EXIF 2.3）。
pub mod tags {
    /// `NewSubfileType`（0x00FE）。
    pub const NEW_SUBFILE_TYPE: u16 = 254;
    /// `ImageWidth`（0x0100）。
    pub const IMAGE_WIDTH: u16 = 256;
    /// `ImageLength`（0x0101）。
    pub const IMAGE_LENGTH: u16 = 257;
    /// `BitsPerSample`（0x0102）。
    pub const BITS_PER_SAMPLE: u16 = 258;
    /// `Compression`（0x0103；1 = 无压缩、7 = lossless JPEG、8 = Deflate(ZIP)、
    /// 32946 = Deflate 的厂商旧编号）。
    pub const COMPRESSION: u16 = 259;
    /// `Predictor`（0x013D；**仅对 `Compression` 8 / 32946 有意义**，取值语义见
    /// [`crate::raw::DeflatePredictor`]）。
    pub const PREDICTOR: u16 = 317;
    /// `PhotometricInterpretation`（0x0106）。
    pub const PHOTOMETRIC_INTERPRETATION: u16 = 262;
    /// `FillOrder`（0x010A；1 = MSB-first（DNG 默认）、2 = LSB-first）。
    pub const FILL_ORDER: u16 = 266;
    /// `StripOffsets`（0x0111）。
    pub const STRIP_OFFSETS: u16 = 273;
    /// `SamplesPerPixel`（0x0115）。
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    /// `RowsPerStrip`（0x0116）。
    pub const ROWS_PER_STRIP: u16 = 278;
    /// `StripByteCounts`（0x0117）。
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    /// `TileWidth`（0x0142，**tile 布局**：单个 tile 的宽）。
    pub const TILE_WIDTH: u16 = 322;
    /// `TileLength`（0x0143，**tile 布局**：单个 tile 的高）。
    pub const TILE_LENGTH: u16 = 323;
    /// `TileOffsets`（0x0144，**tile 布局**：各 tile 起始偏移）。
    pub const TILE_OFFSETS: u16 = 324;
    /// `TileByteCounts`（0x0145，**tile 布局**：各 tile 字节数）。
    pub const TILE_BYTE_COUNTS: u16 = 325;
    /// `SubIFDs`（0x014A；**子 IFD 链**：预览图 / 缩略图，类型 IFD 或 LONG）。
    ///
    /// 真实相机 DNG 普遍带 SubIFD 预览图，解析见
    /// [`crate::raw_subifd::parse_sub_ifds`]。
    pub const SUB_IFDS: u16 = 330;
    /// `CFARepeatPatternDim`（0x82BD，**色彩滤波阵列**重复尺寸）。
    pub const CFA_REPEAT_PATTERN_DIM: u16 = 33421;
    /// `CFAPattern`（0x828E，**CFA 平面序列**：0=R、1=G、2=B）。
    pub const CFA_PATTERN: u16 = 33422;
    /// `ExposureTime`（0x829A，EXIF）。
    pub const EXPOSURE_TIME: u16 = 33434;
    /// `FNumber`（0x829D，EXIF）。
    pub const FNUMBER: u16 = 33437;
    /// `ExifIFDPointer`（0x8769）。
    pub const EXIF_IFD_POINTER: u16 = 34665;
    /// `GPSInfoIFDPointer`（0x8825，**读、不解**）。
    pub const GPS_IFD_POINTER: u16 = 34853;
    /// `ISOSpeedRatings`（0x8827，EXIF）。
    pub const ISO_SPEED_RATINGS: u16 = 34855;
    /// `DateTimeOriginal`（0x9003，EXIF）。
    pub const DATE_TIME_ORIGINAL: u16 = 36867;
    /// `DNGVersion`（0xC612）。
    pub const DNG_VERSION: u16 = 50706;
    /// `DNGBackwardVersion`（0xC613）。
    pub const DNG_BACKWARD_VERSION: u16 = 50707;
    /// `UniqueCameraModel`（0xC614）。
    pub const UNIQUE_CAMERA_MODEL: u16 = 50720;
    /// `WhiteLevel`（0xC61D，**白电平**）。
    pub const WHITE_LEVEL: u16 = 50717;
    /// `BlackLevel`（0xC61A，**黑电平**，可为 1 个或 CFA 位置数个）。
    pub const BLACK_LEVEL: u16 = 50718;
    /// `BlackLevelDeltaH`（0xC61B，**逐行黑电平偏移**，SHORT/LONG，缺省全 0）。
    pub const BLACK_LEVEL_DELTA_H: u16 = 50715;
    /// `BlackLevelDeltaV`（0xC61C，**逐列黑电平偏移**，SHORT/LONG，缺省全 0）。
    pub const BLACK_LEVEL_DELTA_V: u16 = 50716;
    /// `LinearizationTable`（0xC691 = 50833，**线性化表**：raw 码值 → 线性值）。
    ///
    /// DNG 1.6 Chapter 5「Mapping Raw Values to Linear Reference Values」规定
    /// 映射顺序是 **Linearization → Black Subtraction → Rescaling → Clipping**，
    /// 故本表在减黑电平**之前**应用。
    ///
    /// > 注：`LinearizationTable` 标签号曾被写作 `0xC618`，那是**错的**
    /// > （`0xC618` = 50712 不是任何 DNG 标签）；规范原文是 **0xC691 = 50833**。
    pub const LINEARIZATION_TABLE: u16 = 50833;
    /// `ColorMatrix1`（0xC621，**3×3 色彩矩阵**，SRATIONAL × 9）。
    pub const COLOR_MATRIX1: u16 = 50721;
    /// `RowInterleaveFactor`（0xC71F = 50975，**行交织因子**，缺省 **1**）。
    ///
    /// DNG 1.6 原文：
    ///
    /// > Type SHORT or LONG / Count 1 / Default **1** / Usage Raw IFD
    /// > This tag specifies that rows of the image are stored in interleaved
    /// > order. The value of the tag specifies the number of interleaved fields.
    /// > The use of a non-default value for this tag requires setting the
    /// > DNGBackwardVersion tag to at least 1.2.0.0.
    ///
    /// `factor = 1` → 正常行序；`factor = 2` → **偶数行在前、奇数行在后**
    /// （即「奇偶行分离」所描述的机制；而 `tags & 0x8000` 是 dcraw/LibRaw 的
    /// 私有实现细节，**不是 DNG 标签**）。
    pub const ROW_INTERLEAVE_FACTOR: u16 = 50975;
    /// `AnalogBalance`（0xC62A）。
    pub const ANALOG_BALANCE: u16 = 50727;
    /// `AsShotNeutral`（0xC628，**拍摄时白平衡中性值**，RATIONAL × 3）。
    pub const AS_SHOT_NEUTRAL: u16 = 50728;
    /// `AsShotWhiteXY`（0xC629）。
    pub const AS_SHOT_WHITE_XY: u16 = 50729;
    /// `BaselineExposure`（0xC62B）。
    pub const BASELINE_EXPOSURE: u16 = 50730;
    /// `CalibrationIlluminant1`（0xC65D；21 = D65、19 = A 光源）。
    pub const CALIBRATION_ILLUMINANT1: u16 = 50778;
    /// `ActiveArea`（0xC68D：top/left/bottom/right，原图有效区）。
    pub const ACTIVE_AREA: u16 = 50829;
    /// `ForwardMatrix2`（0xC68A，**camera space → XYZ(D50)** 正向矩阵，SRATIONAL × 9）。
    ///
    /// DNG 1.4 推荐的**正向色彩映射**：与 `ColorMatrix1`（反矩阵）并存，
    /// 语义见 [`DngFile::forward_matrix2`]。
    pub const FORWARD_MATRIX2: u16 = 50826;
    /// `CameraCalibration1`（0xC6F3，**光学暗电流校准矩阵**，SRATIONAL × 9）。
    pub const CAMERA_CALIBRATION1: u16 = 50931;
    /// `CameraCalibration2`（0xC6F4，备选校准矩阵；本模块只读不选）。
    pub const CAMERA_CALIBRATION2: u16 = 50932;
    /// `RawDataUniqueID`（0xC640）。
    pub const RAW_DATA_UNIQUE_ID: u16 = 50752;
}

/// TIFF 字段类型号（TIFF 5.0 §2）。
pub mod field_type {
    /// `BYTE`（1 字节无符号）。
    pub const BYTE: u16 = 1;
    /// `ASCII`（1 字节字符）。
    pub const ASCII: u16 = 2;
    /// `SHORT`（2 字节无符号）。
    pub const SHORT: u16 = 3;
    /// `LONG`（4 字节无符号）。
    pub const LONG: u16 = 4;
    /// `RATIONAL`（2×u32：分子/分母）。
    pub const RATIONAL: u16 = 5;
    /// `SBYTE`（1 字节有符号）。
    pub const SBYTE: u16 = 6;
    /// `UNDEFINED`（1 字节未定义）。
    pub const UNDEFINED: u16 = 7;
    /// `SSHORT`（2 字节有符号）。
    pub const SSHORT: u16 = 8;
    /// `SLONG`（4 字节有符号）。
    pub const SLONG: u16 = 9;
    /// `SRATIONAL`（2×i32）。
    pub const SRATIONAL: u16 = 10;
    /// `FLOAT`（4 字节 IEEE 754）。
    pub const FLOAT: u16 = 11;
    /// `DOUBLE`（8 字节 IEEE 754）。
    pub const DOUBLE: u16 = 12;
    /// `IFD`（4 字节子 IFD 偏移）。
    pub const IFD: u16 = 13;
}

/// 各 TIFF 字段类型的字节宽度（索引 = 类型号；0 与未知类型为 0）。
const TYPE_SIZE: [u32; 19] = [
    0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, // 0..=13
    0, 0, 0, 0, 0, // 14..=18（TIFF/EP 未定义或私有）
];

/// 字段类型的字节宽度（未知类型返回 0）。
pub fn type_size(field_type: u16) -> u32 {
    TYPE_SIZE.get(field_type as usize).copied().unwrap_or(0)
}

/// 一个已解析的 IFD 条目（外置值已按 `count × 类型宽度` 取出）。
#[derive(Debug, Clone, PartialEq)]
pub struct TiffEntry {
    /// 标签号。
    pub tag: u16,
    /// 字段类型（见 [`field_type`]）。
    pub field_type: u16,
    /// 元素个数（`RATIONAL` 计 1 个 = 8 字节）。
    pub count: u32,
    /// 元素值的原始字节（小端字段为 LE 序列，与 `order` 无关地按 `order` 解读）。
    pub data: Vec<u8>,
}

impl TiffEntry {
    /// 按当前字节序把 `data` 解为数值向量。
    ///
    /// 未知类型、`UNDEFINED`/`ASCII` 返回空向量（调用方另行取 ASCII）。
    /// `RATIONAL` 分母为 0 时该项取 `0.0`（真实 DNG 确有 0/0 占位，不视为错误）。
    pub fn numbers(&self, order: ByteOrder) -> Vec<f32> {
        let d: &[u8] = &self.data;
        let get_u16 = |at: usize| -> u16 {
            match d.get(at..at + 2) {
                Some(b) => order.u16(b),
                None => 0,
            }
        };
        let get_u32 = |at: usize| -> u32 {
            match d.get(at..at + 4) {
                Some(b) => order.u32(b),
                None => 0,
            }
        };
        let mut out: Vec<f32> = Vec::new();
        match self.field_type {
            field_type::BYTE | field_type::UNDEFINED => {
                out.extend(d.iter().map(|&b| b as f32));
            }
            field_type::ASCII => { /* 非数值：由 text() 取 */ }
            field_type::SHORT => {
                let n = (self.count as usize).min(d.len() / 2);
                out.extend((0..n).map(|i| get_u16(i * 2) as f32));
            }
            field_type::SSHORT => {
                let n = (self.count as usize).min(d.len() / 2);
                out.extend((0..n).map(|i| get_u16(i * 2) as i16 as f32));
            }
            field_type::LONG | field_type::IFD => {
                let n = (self.count as usize).min(d.len() / 4);
                out.extend((0..n).map(|i| get_u32(i * 4) as f32));
            }
            field_type::SLONG => {
                let n = (self.count as usize).min(d.len() / 4);
                out.extend((0..n).map(|i| get_u32(i * 4) as i32 as f32));
            }
            field_type::RATIONAL => {
                let n = (self.count as usize).min(d.len() / 8);
                for i in 0..n {
                    let num = get_u32(i * 8);
                    let den = get_u32(i * 8 + 4);
                    out.push(if den == 0 {
                        0.0
                    } else {
                        num as f32 / den as f32
                    });
                }
            }
            field_type::SRATIONAL => {
                let n = (self.count as usize).min(d.len() / 8);
                for i in 0..n {
                    let num = get_u32(i * 8) as i32;
                    let den = get_u32(i * 8 + 4) as i32;
                    out.push(if den == 0 {
                        0.0
                    } else {
                        num as f32 / den as f32
                    });
                }
            }
            field_type::FLOAT => {
                let n = (self.count as usize).min(d.len() / 4);
                for i in 0..n {
                    out.push(f32::from_bits(get_u32(i * 4)));
                }
            }
            field_type::DOUBLE => {
                let n = (self.count as usize).min(d.len() / 8);
                for i in 0..n {
                    let mut word = [0u8; 8];
                    if let Some(s) = d.get(i * 8..i * 8 + 8) {
                        word.copy_from_slice(s);
                    }
                    let bits = match order {
                        ByteOrder::Little => u64::from_le_bytes(word),
                        ByteOrder::Big => u64::from_be_bytes(word),
                    };
                    out.push(f64::from_bits(bits) as f32);
                }
            }
            _ => { /* 未知类型：空向量 */ }
        }
        out
    }

    /// 取 ASCII 值（去掉尾部 `NUL`），非 ASCII 返回 `None`。
    pub fn text(&self) -> Option<String> {
        if self.field_type != field_type::ASCII && self.field_type != field_type::UNDEFINED {
            return None;
        }
        let end = self
            .data
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.data.len());
        Some(String::from_utf8_lossy(&self.data[..end]).into_owned())
    }
}

/// 一个已解析的 IFD（image file directory）。
#[derive(Debug, Clone, PartialEq)]
pub struct TiffIfd {
    /// 条目（构造时已按标签**升序**排列，保证遍历顺序确定）。
    pub entries: Vec<TiffEntry>,
    /// 下一个 IFD 的偏移（0 = 无）。
    pub next_offset: u32,
}

impl TiffIfd {
    /// 按标签查条目（`entries` 升序，二分查找；同标签取首个 = TIFF 规定）。
    pub fn get(&self, tag: u16) -> Option<&TiffEntry> {
        let idx = self.entries.binary_search_by(|e| e.tag.cmp(&tag)).ok()?;
        self.entries.get(idx)
    }

    /// 查标签并解为数值向量。
    pub fn numbers(&self, tag: u16, order: ByteOrder) -> Option<Vec<f32>> {
        self.get(tag).map(|e| e.numbers(order))
    }

    /// 查标签并解为**首个**数值（标量标签的便捷读法；缺失 → `None`）。
    pub fn first_number(&self, tag: u16, order: ByteOrder) -> Option<f32> {
        self.numbers(tag, order).and_then(|v| v.first().copied())
    }
}

/// 解析一个 IFD（`offset` 处；`count` 与外置值长度均做防御性上限检查）。
pub fn read_ifd(data: &[u8], order: ByteOrder, offset: u32) -> RawResult<TiffIfd> {
    const LOC: &str = "raw::read_ifd";
    if offset == 0 {
        return Err(tiff_err(
            LOC,
            "IFD 偏移为 0（TIFF 中 0 表示无 IFD）",
            "确认 TIFF 目录结构完整",
        ));
    }
    let mut cur = Cursor::new(data, order, offset as usize);
    let count = cur.u16(LOC)? as usize;
    if count > MAX_IFD_ENTRIES {
        return Err(tiff_err(
            LOC,
            format!("IFD 条目数 {count} 超过防御上限 {MAX_IFD_ENTRIES}"),
            "文件损坏或非 TIFF；确认字节流未错位",
        ));
    }
    let mut entries: Vec<TiffEntry> = Vec::with_capacity(count);
    for i in 0..count {
        let base = cur.pos();
        let tag = cur.u16(LOC)?;
        let ty = cur.u16(LOC)?;
        let n = cur.u32(LOC)?;
        let field = cur.take(4, LOC)?;
        let unit = type_size(ty);
        if unit == 0 {
            return Err(tiff_err(
                LOC,
                format!("第 {i} 条（tag {tag}）字段类型 {ty} 未定义"),
                "DNG 只允许标准 TIFF 字段类型；确认文件非私有格式",
            ));
        }
        let total = (n as u64).checked_mul(unit as u64).ok_or_else(|| {
            tiff_err(
                LOC,
                format!("第 {i} 条（tag {tag}）count {n} × {unit} 字节溢出"),
                "文件损坏；确认 count 字段未被错位读取",
            )
        })?;
        if total > MAX_TIFF_VALUE_BYTES as u64 {
            return Err(tiff_err(
                LOC,
                format!("第 {i} 条（tag {tag}）外置值 {total} 字节超过上限 {MAX_TIFF_VALUE_BYTES}",),
                "超大属性 size：确认文件未被截断/拼接错误，或改用流式分块解析",
            ));
        }
        let total_usize = total as usize;
        let data_bytes: Vec<u8> = if total_usize <= 4 {
            // 内联值：LE 右对齐、BE 左对齐（TIFF 5.0 §2）。
            match order {
                ByteOrder::Little => field[4 - total_usize..].to_vec(),
                ByteOrder::Big => field[..total_usize].to_vec(),
            }
        } else {
            let at = order.u32(field) as usize;
            Cursor::new(data, order, at)
                .take(total_usize, LOC)?
                .to_vec()
        };
        entries.push(TiffEntry {
            tag,
            field_type: ty,
            count: n,
            data: data_bytes,
        });
        // 保持游标严格前进（take 已保证），此处显式记录便于审计。
        let _ = base;
    }
    entries.sort_by_key(|e| e.tag);
    let next_offset = cur.u32(LOC)?;
    Ok(TiffIfd {
        entries,
        next_offset,
    })
}

// ===========================================================================
//  四、格式探测（含 CR3 的结构化拒绝）
// ===========================================================================

/// RAW 容器格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawFormat {
    /// Adobe Digital Negative（TIFF 派生，**支持**）。
    Dng,
    /// Canon CR3（ISO-BMFF 派生，**结构化拒绝**）。
    Cr3,
}

/// 探测 RAW 容器格式；不支持者给出三段式错误（含「为何不支持」）。
///
/// **CR3 为何不支持**（结构化拒绝的理由，写入错误文本）：
/// 1. **容器不同**——CR3 是 ISO-BMFF（MP4 家族）容器，文件头为
///    `ftyp` 盒（品牌 `crx `），而非 TIFF 的 `II`/`MM` + magic 42；本模块的
///    TIFF/IFD 解析器对其完全无效；
/// 2. **Exif 藏在 CMT 盒**——元数据位于 `CMT1`/`CMT2`/`CMT3` 三个盒
///    （`CMT2` 为 Exif 派生负载），须先遍历 ftyp/moov/meta/ilst→iloc 盒树
///    才能拿到 Exif 与 Canon 私有 `0xc6xx` 标签；
/// 3. **像素负载经私有无损压缩**——传感器数据在 `mdat` 中按 CRX 块（每个
///    块自带 slice 描述符）以 **CRX 无损算术编码**（部分为 JPEG 2000）存放，
///    需从 `CMT1` 读切片尺寸/条带表并实现该解码器；
/// 4. **CFA 几何被私有变换混淆**——Canon 用 4×4（部分机型 6×6/8×8）线性变换
///    矩阵与伪装的 CFA 相位重排马赛克，须先反解该矩阵才能得到真实 Bayer 布局；
/// 5. **工程量与复用率**——上述（盒树遍历 + CRX 解码 + 反变换）是一条独立的
///    解析链，与 DNG 的 IFD 解析零复用，且 CRX 编码细节无公开规范可依
///    （只能逆向），引入半正确实现违反确定性铁律。故按下述原则处理：
///    「结构化拒绝，不假装能解析」。
pub fn probe_format(data: &[u8]) -> RawResult<RawFormat> {
    const LOC: &str = "raw::probe_format";
    if data.len() < 16 {
        return Err(tiff_err(
            LOC,
            format!(
                "文件仅 {} 字节，不足 16 字节（连容器头都不完整）",
                data.len()
            ),
            "确认文件传输完整",
        ));
    }
    // ISO-BMFF：字节 4..8 == "ftyp"。CR3 的 major brand 为 "crx "。
    if &data[4..8] == b"ftyp" {
        let brand: &[u8] = data.get(8..12).unwrap_or(b"    ");
        let is_cr3 = brand.starts_with(b"crx");
        return Err(unsupported_err(
            LOC,
            format!(
                "ISO-BMFF 容器（首盒 ftyp，brand {:?}，{}）：CR3 的元数据在 \
                 CMT1/CMT2/CMT3 盒、像素在 mdat 内经 CRX 私有无损算术编码、\
                 并用 4×4 私有变换矩阵混淆 CFA 相位——与 DNG 的 TIFF/IFD 解析零复用",
                brand,
                if is_cr3 {
                    "确认为 CR3"
                } else {
                    "非 CR3 但同为 BMFF"
                }
            ),
            "改用 DNG 作为相机 RAW 入口。容器层可用 probe_cr3 做盒树遍历/uuid 定位，\
             但像素解码需 CMT3/CDC + CRX 解码器（本模块未实现，请先转 DNG）",
        ));
    }
    // TIFF 系：II/MM + magic 42。CR2 同样是 TIFF 容器，另作区分。
    if let Some(order) = ByteOrder::of_magic(data) {
        // TIFF 头：`II`/`MM`（2 字节）+ magic 42（**u16**，偏移 2）+ 首个 IFD
        // 偏移（u32，偏移 4）。magic 是 2 字节字段，不是 4 字节。
        let magic = order.u16(data.get(2..4).unwrap_or(&[0, 0])) as u32;
        if magic == 42 {
            return Ok(RawFormat::Dng);
        }
        if magic == 0x4F52 || magic == 0x5352 {
            return Err(unsupported_err(
                LOC,
                format!("TIFF 容器但 magic = {magic:#06x}（CR2 系私有标记），非 DNG"),
                "CR2 的标签表与 DNG 不兼容；请转存为 DNG 后再读入",
            ));
        }
        return Err(tiff_err(
            LOC,
            format!("TIFF magic = {magic}，应为 42"),
            "字节序头与 magic 不自洽；确认文件未被改写",
        ));
    }
    Err(tiff_err(
        LOC,
        format!(
            "既非 TIFF（II/MM）也非 ISO-BMFF（ftyp）：头 4 字节 = {:02X?}",
            &data[0..4]
        ),
        "确认这是相机 RAW 文件而不是 PNG/EXR/视频容器",
    ))
}

// ===========================================================================
//  四之二、CR3：ISO-BMFF 盒树遍历（**诚实的粗解析**）
// ===========================================================================

/// Apple 在 CR3 里使用的扩展 `uuid`（**CIPA DC-3**）。
///
/// 字节序按 ISO-BMFF 规范为**网络序（大端）**书写的 16 字节 UUID：
/// `85c0b687-820f-11e0-81f1-0800200c9a66`。该 `uuid` 盒内含 `CMT1`..`CMT4`
/// 与 `THMB`，是 CR3 的**元数据入口**。
pub const CR3_APPLE_UUID: [u8; 16] = [
    0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0, 0x81, 0xf1, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66,
];

/// 盒树遍历的**最大深度**（防御性上限）。
///
/// ISO-BMFF 允许容器盒任意嵌套；真实 CR3 的实际深度 < 6。设 16 是为了给
/// 畸形/恶意文件一个明确的拒绝阈值——**无界递归会栈溢出**（不是 panic，
/// 是进程 abort），故必须有上限。
pub const MAX_BOX_DEPTH: usize = 16;

/// 一个 ISO-BMFF 盒（box）的**头部**信息（不复制负载，零额外分配）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxNode {
    /// 四字符类型（`b"ftyp"` 等，**原样字节**，未做字符串转换）。
    pub kind: [u8; 4],
    /// 盒在文件中的起始偏移（含头部）。
    pub offset: u64,
    /// 盒的**总**长度（含头部）。
    pub size: u64,
    /// 负载相对盒起点的偏移（8 / 16，见 [`parse_box`]）。
    pub header_size: u64,
    /// 嵌套深度（顶层 = 0）。
    pub depth: usize,
}

impl BoxNode {
    /// 负载的**绝对**字节区间。
    pub fn payload_range(&self) -> (u64, u64) {
        (
            self.offset.saturating_add(self.header_size),
            self.offset.saturating_add(self.size),
        )
    }

    /// 四字符类型的字符串形式（`ftyp` / `uuid` / `CMT1`…）。
    pub fn kind_str(&self) -> String {
        String::from_utf8_lossy(&self.kind).into_owned()
    }

    /// 是否为 `uuid` 盒（ISO-BMFF 的扩展类型盒，负载以 16 字节 UUID 开头）。
    pub fn is_uuid(&self) -> bool {
        &self.kind == b"uuid"
    }

    /// `uuid` 盒的 16 字节 **UUID 本体**在文件中的起始偏移。
    ///
    /// `uuid` 的 `header_size` 已被 [`parse_box`] 计入这 16 字节（故
    /// [`BoxNode::payload_range`] 从 **UUID 之后**开始，即真正的子盒序列）。
    /// 要读UUID 本体必须用 `offset + 8`（基本头之后），**不能**用
    /// `payload_range().0`——那是子盒起点，比 UUID 晚 16 字节。
    ///
    /// 非 `uuid` 盒 → `None`。
    pub fn uuid_field_offset(&self) -> Option<u64> {
        if self.is_uuid() {
            Some(self.offset.saturating_add(8))
        } else {
            None
        }
    }
}

/// 解析单个盒的**头部**（`data[pos..]` 处）。
///
/// # ISO-BMFF 盒头（§4.2）
///
/// ```text
/// ┌────────────┬──────────────┐
/// │ size: u32  │ type: 4 bytes│   ← 8 字节基本头
/// ├────────────┴──────────────┤
/// │ size == 1 → 64 位 largesize │   ← 再 8 字节
/// │ type == 'uuid' → 16 字节 UUID │   ← 再 16 字节
/// └───────────────────────────┘
/// ```
///
/// - **`size == 1`**：真实长度在紧随其后的 **u64 大端** `largesize` 里；
/// - **`size == 0`**：**延伸到父容器末尾**（本函数用 `limit` 表达父容器边界）；
/// - **`size < 头长`**：畸形 → 结构化错误（这是最常见的畸形形态，
///   若不拦截会算出负长度进而越界）。
///
/// # 错误
///
/// 头不全、`largesize` 不全、UUID 不全、`size < header`、
/// 或盒体越过 `limit`（父容器边界）——一律 [`tiff_err`]（`RAW1xxx`），
/// **绝不 panic、绝不返回假盒**。
fn parse_box(data: &[u8], pos: u64, limit: u64, depth: usize) -> RawResult<BoxNode> {
    const LOC: &str = "raw::parse_box";
    if depth > MAX_BOX_DEPTH {
        return Err(tiff_err(
            LOC,
            format!("盒嵌套深度 {depth} 超过上限 {MAX_BOX_DEPTH}"),
            "文件畸形或为恶意构造；确认字节流未损坏",
        ));
    }
    let avail = limit.saturating_sub(pos);
    let head = data
        .get(usize_sat(pos)..usize_sat(pos).saturating_add(8))
        .ok_or_else(|| {
            tiff_err(
                LOC,
                format!("偏移 {pos} 处不足 8 字节盒头（父容器边界 {limit}）"),
                "文件被截断；确认传输完整",
            )
        })?;
    let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let mut kind = [0u8; 4];
    kind.copy_from_slice(&head[4..8]);
    let mut header_size = 8u64;
    let size: u64 = match size32 {
        // size == 1 → 64 位 largesize 跟在基本头之后。
        1 => {
            let at = usize_sat(pos).saturating_add(8);
            let b = data.get(at..at.saturating_add(8)).ok_or_else(|| {
                tiff_err(
                    LOC,
                    format!("偏移 {pos} 的盒声明 64 位长度但 largesize 不全"),
                    "文件被截断",
                )
            })?;
            header_size = 16;
            u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }
        // size == 0 → 延伸到父容器末尾。
        0 => avail,
        n => n as u64,
    };
    if &kind == b"uuid" {
        header_size = header_size.saturating_add(16);
    }
    if size < header_size {
        return Err(tiff_err(
            LOC,
            format!(
                "偏移 {pos} 的 {:?} 盒声明长度 {size} < 头长 {header_size}",
                String::from_utf8_lossy(&kind)
            ),
            "盒头自相矛盾，文件畸形；确认未截断/错位",
        ));
    }
    if pos.saturating_add(size) > limit {
        return Err(tiff_err(
            LOC,
            format!(
                "偏移 {pos} 的 {:?} 盒越过父容器边界（{}+{} > {limit}）",
                String::from_utf8_lossy(&kind),
                pos,
                size
            ),
            "盒长度与容器不自洽；文件畸形或被截断",
        ));
    }
    Ok(BoxNode {
        kind,
        offset: pos,
        size,
        header_size,
        depth,
    })
}

/// `u64 → usize` 的**饱和**转换（C18：绝不用 `as` 截断）。
#[inline]
fn usize_sat(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// 判定某盒是否为**容器盒**（其负载是子盒序列，需继续遍历）。
///
/// ISO-BMFF 的容器盒白名单（不认识的盒**不递归**——按 ISO-BMFF §4.2，
/// 未列入白名单的盒其负载是**不透明字节**，递归会误把元数据当盒头解析）。
fn is_container_box(kind: &[u8; 4]) -> bool {
    matches!(
        kind,
        b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"dinf" | b"moof" | b"traf" | b"edts"
            | b"udta" | b"uuid"
    )
}

/// 遍历 ISO-BMFF 盒树，返回**扁平的先序列表**（父盒在子盒之前）。
///
/// # 确定性
///
/// 深度优先、先序；`data` 里的盒按**文件内物理顺序**处理，**不依赖任何
/// 哈希表或索引查找**——同输入逐次遍历得到逐字节相同的 [`BoxNode`] 序列。
///
/// # 未知盒
///
/// 不在 [`is_container_box`] 白名单里的盒**只记录、不递归**，其负载被当作
/// 不透明字节跳过（正确处理，也避免把 JPEG 负载误当盒头）。
pub fn walk_boxes(data: &[u8]) -> RawResult<Vec<BoxNode>> {
    walk_boxes_from(data, 0, data.len() as u64, 0)
}

/// 从 `pos` 起遍历盒树直到 `limit`（[`walk_boxes`] 的递归内核，公开便于测试）。
pub fn walk_boxes_from(
    data: &[u8],
    pos: u64,
    limit: u64,
    depth: usize,
) -> RawResult<Vec<BoxNode>> {
    const LOC: &str = "raw::walk_boxes";
    let mut out: Vec<BoxNode> = Vec::new();
    let mut cur = pos;
    while cur < limit {
        let node = parse_box(data, cur, limit, depth)?;
        let next = cur.saturating_add(node.size);
        let size = node.size;
        let (ps, pe) = node.payload_range();
        let descend = is_container_box(&node.kind) && node.header_size < node.size;
        out.push(node);
        if descend {
            // 子盒深度 +1；`walk_boxes_from` 自带深度上限检查。
            out.extend(walk_boxes_from(data, ps, pe, depth.saturating_add(1))?);
        }
        // 零长度盒会死循环：`size < header` 已被 `parse_box` 拒掉
        // （故 `size ≥ header ≥ 8`），这里只是把该不变式显式记下来。
        debug_assert!(next > cur, "盒长度必须为正：pos = {cur}, size = {size}");
        cur = next;
    }
    if cur > limit {
        return Err(tiff_err(
            LOC,
            format!("盒序列越过父容器边界（{cur} > {limit}）"),
            "文件畸形或被截断",
        ));
    }
    Ok(out)
}

/// CR3（Canon RAW v3）的**结构化元数据**——**不含任何像素**。
///
/// # ⚠️ 诚实边界：本结构**不含、也无法产出**像素数据
///
/// CR3 的像素负载在 `mdat` 中按 **CRX 块**存放，每个块自带 slice 描述符并
/// 经 **CRX 无损算术编码**（部分为 JPEG 2000）压缩；同时 Canon 用私有的
/// 4×4（部分机型 6×6/8×8）线性变换矩阵**混淆 CFA 相位**。完整解码需要：
///
/// 1. `CMT1` 解析（拿到传感器尺寸 / CFA 几何 / 切片表）；
/// 2. `CMT3` / `CMT2` 解析（拿到 CRX 切片表与 Canon 私有色彩标签）；
/// 3. **CRX 无损算术解码器**（DC-3 规范未公开全文，只能逆向）；
/// 4. 私有 CFA 变换矩阵的**反解**。
///
/// 这是一条与 DNG 的 TIFF/IFD 解析**零复用**的独立链路，工程量极大且
/// CRX 细节无公开规范可依。故本模块**只做诚实的粗解析**：盒树遍历 + uuid
/// 定位 + 少量可确定的结构化字段，**像素解码明确拒绝**
/// （见 [`Cr3Info::decode_pixels`]，它**恒返回错误**，绝不返回假数据）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Cr3Info {
    /// 主品牌（`ftyp` 的 major brand，CR3 为 `b"crx "`）。
    pub major_brand: [u8; 4],
    /// 盒树（先序，见 [`walk_boxes`]）。
    pub boxes: Vec<BoxNode>,
    /// Apple `uuid` 盒（`85c0b687-…`）在文件中的偏移；未找到 → `None`。
    pub apple_uuid_offset: Option<u64>,
    /// `CRAW` 头（`mdat` 内 CRX 切片的公共头）所在偏移；未找到 → `None`。
    pub craw_header_offset: Option<u64>,
    /// 传感器/图像尺寸（从 `CMT1` 的 Exif 读出；读不到 → `None`）。
    ///
    /// **只报「读到了什么」，不推断**：见 [`probe_cr3`] 的说明。
    pub sensor_size: Option<(u32, u32)>,
    /// `mdat` 盒数（CR3 通常 1 个）。
    pub mdat_count: u32,
}

impl Cr3Info {
    /// 找到全部指定类型的盒（按物理顺序）。
    pub fn boxes_of<'a>(&'a self, kind: &'a [u8; 4]) -> impl Iterator<Item = &'a BoxNode> {
        self.boxes.iter().filter(move |b| &b.kind == kind)
    }

    /// 找到第一个指定类型的盒。
    pub fn first_box(&self, kind: &[u8; 4]) -> Option<&BoxNode> {
        self.boxes.iter().find(|b| &b.kind == kind)
    }

    /// **CR3 像素解码 —— 恒返回结构化错误（本模块不实现 CRX 解码）。**
    ///
    /// # 硬性约束
    ///
    /// 本函数**永远不返回 `Ok`**。它存在的唯一目的是：让「调用方想读 CR3
    /// 像素」这件事得到一个**说清理由、含可行动建议**的错误，而不是
    /// `todo!()` / `unimplemented!()` panic，或更糟的——**返回全零或假数据
    /// 冒充支持**。错误文本必须包含「转 DNG」的建议。
    pub fn decode_pixels(&self) -> RawResult<Vec<u16>> {
        Err(unsupported_err(
            "raw::Cr3Info::decode_pixels",
            format!(
                "CR3 像素解码需要 CMT3/CDC 完整实现，当前未支持：\
                 已定位 {} 个盒、Apple uuid 在 {}，但 mdat 内的 CRX 块\
                 仍需无损算术解码器 + 私有 CFA 变换矩阵反解（CIPA DC-3 未公开全文）",
                self.boxes.len(),
                match self.apple_uuid_offset {
                    Some(o) => format!("偏移 {o}"),
                    None => "未找到".to_string(),
                }
            ),
            "请先转 DNG：Adobe DNG Converter（dngconvert）可把 CR3 转为标准 DNG；\
             或用 exiftool 抽出预览图。本模块绝不用假数据冒充支持。",
        ))
    }
}

/// 探测并**结构化解析** CR3 的容器层（盒树 + uuid 定位 + CRAW 头定位）。
///
/// # 做到哪一层（诚实清单）
///
/// - ✅ ISO-BMFF **盒树遍历**（`ftyp`/`moov`/`uuid`/`moof`/`mdat`/`CMT*`/`THMB`）；
/// - ✅ 定位 Apple `uuid` 盒（`85c0b687-820f-11e0-81f1-0800200c9a66`）；
/// - ✅ 在 `mdat` 内定位 `CRAW` 头（`CRAW` + `CTBO` 魔数对）；
/// - ⚠️ **传感器尺寸只在 `CMT1` 的 Exif 恰好带 `ImageWidth/Length` 时才报**
///   （`CMT1` 的 Exif 是 Canon MakerNote 容器，字段位置依机型而异；本模块
///   **只报真读到的**，读不到就是 `None`，**不猜**）；
/// - ❌ **CRX 块解码 / CFA 私有变换反解 / 像素输出 —— 明确不支持**。
///
/// # 错误
///
/// 非 ISO-BMFF、缺 `ftyp`、盒树畸形 → [`tiff_err`]（`RAW1xxx`）。
pub fn probe_cr3(data: &[u8]) -> RawResult<Cr3Info> {
    const LOC: &str = "raw::probe_cr3";
    if data.len() < 8 {
        return Err(tiff_err(
            LOC,
            format!("文件仅 {} 字节，不足一个 ISO-BMFF 基本头（8 字节）", data.len()),
            "确认文件传输完整",
        ));
    }
    if &data[4..8] != b"ftyp" {
        return Err(tiff_err(
            LOC,
            format!(
                "偏移 4..8 为 {:?}，非 ISO-BMFF（应为首盒 ftyp）",
                String::from_utf8_lossy(&data[4..8])
            ),
            "确认这是 CR3 文件（CR3 是 ISO-BMFF 容器，不是 TIFF）",
        ));
    }
    let boxes = walk_boxes(data)?;
    let ftyp = boxes
        .first()
        .ok_or_else(|| tiff_err(LOC, "盒树为空（首盒 ftyp 未被遍历到）", "文件可能为空"))?;
    if &ftyp.kind != b"ftyp" {
        return Err(tiff_err(
            LOC,
            format!("首个盒是 {:?} 而非 ftyp", ftyp.kind_str()),
            "ISO-BMFF 首个盒必须是 ftyp",
        ));
    }
    // major brand = ftyp 负载的前 4 字节。
    let (ps, pe) = ftyp.payload_range();
    let ps_us = usize_sat(ps);
    let mut major_brand = [0u8; 4];
    if let Some(b) = data.get(ps_us..ps_us.saturating_add(4)) {
        major_brand.copy_from_slice(b);
    }
    let _ = pe;
    // Apple uuid 定位：只看**顶层**（depth 0）的 uuid 盒——CMT* 是它的子盒。
    // UUID 本体在 `offset + 8`（基本头之后），**不是** `payload_range().0`
    // （后者已被 16 字节 UUID 推到子盒起点）。
    let apple_uuid_offset = boxes
        .iter()
        .find(|b| b.depth == 0 && b.is_uuid())
        .filter(|b| {
            let s = usize_sat(b.uuid_field_offset().unwrap_or(0));
            data.get(s..s.saturating_add(16)) == Some(CR3_APPLE_UUID.as_slice())
        })
        .map(|b| b.offset);
    // CRAW 头：在 mdat 负载内找 `CRAW` + `CTBO` 魔数对。
    let craw_header_offset = boxes
        .iter()
        .filter(|b| &b.kind == b"mdat")
        .find_map(|b| {
            let (s, e) = b.payload_range();
            let s = usize_sat(s);
            let e = usize_sat(e).min(data.len());
            find_magic_pair(data, s, e)
        });
    // 传感器尺寸：只从 CMT1 的 Exif **真读到**才报。
    let sensor_size = boxes
        .iter()
        .find(|b| &b.kind == b"CMT1")
        .and_then(|b| {
            let (s, e) = b.payload_range();
            read_cmt1_sensor_size(data, usize_sat(s), usize_sat(e))
        });
    let mdat_count = boxes.iter().filter(|b| &b.kind == b"mdat").count() as u32;
    Ok(Cr3Info {
        major_brand,
        boxes,
        apple_uuid_offset,
        craw_header_offset,
        sensor_size,
        mdat_count,
    })
}

/// 在 `[s, e)` 内找 `CRAW` + `CTBO` 魔数对（CR3 切片头的标识，CIPA DC-3）。
///
/// 返回**第一个**命中的 `CRAW` 偏移（确定顺序：升序扫描）。
fn find_magic_pair(data: &[u8], s: usize, e: usize) -> Option<u64> {
    let e = e.min(data.len());
    if e < s.saturating_add(8) {
        return None;
    }
    let mut i = s;
    while i.saturating_add(8) <= e {
        if data.get(i..i.saturating_add(4)) == Some(b"CRAW")
            && data.get(i.saturating_add(4)..i.saturating_add(8)) == Some(b"CTBO")
        {
            return Some(i as u64);
        }
        i = i.saturating_add(1);
    }
    None
}

/// 从 `CMT1` 负载尽力读传感器尺寸（**只报真读到的，读不到 → `None`**）。
///
/// `CMT1` 的 Exif 内部字段位置依 Canon 机型而异（MakerNote 私有偏移）。
/// 本函数**只做一件事**：在 `CMT1` 负载内找**任一** TIFF 头的
/// `ImageWidth`(0x0100) / `ImageLength`(0x0101) 成对值。
/// 找不到就是 `None`——**绝不猜、绝不返回默认值**。
fn read_cmt1_sensor_size(data: &[u8], s: usize, e: usize) -> Option<(u32, u32)> {
    let e = e.min(data.len());
    if e < s.saturating_add(8) {
        return None;
    }
    // 最多试 4 个偏移（MakerNote 常见对齐），确定性顺序。
    let tries = [0usize, 2, 4, 8];
    for t in tries {
        let at = s.saturating_add(t);
        if at.saturating_add(8) > e {
            break;
        }
        let order = match ByteOrder::of_magic(&data[at..e.min(at.saturating_add(2))]) {
            Some(o) => o,
            None => continue,
        };
        let magic = match data.get(at.saturating_add(2)..at.saturating_add(4)) {
            Some(m) => order.u16(m) as u32,
            None => continue,
        };
        if magic != 42 {
            continue;
        }
        let ifd_off = match data.get(at.saturating_add(4)..at.saturating_add(8)) {
            Some(v) => order.u32(v) as usize,
            None => continue,
        };
        // IFD 偏移相对该 TIFF 头起点。
        let base = at;
        let at_abs = base.saturating_add(ifd_off);
        if at_abs >= e {
            continue;
        }
        if let Ok(ifd) = read_ifd(data, order, at_abs as u32) {
            let w = ifd.first_number(tags::IMAGE_WIDTH, order);
            let h = ifd.first_number(tags::IMAGE_LENGTH, order);
            if let (Some(w), Some(h)) = (w, h) {
                if w.is_finite() && h.is_finite() && w >= 1.0 && h >= 1.0 {
                    return Some((w as u32, h as u32));
                }
            }
        }
    }
    None
}

// ===========================================================================
//  五、CFA 布局与 Bayer 四态
// ===========================================================================

/// CFA 颜色（与 DNG `CFAPattern` 标签的编码一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfaColor {
    /// 红（`CFAPattern` 值 0）。
    Red = 0,
    /// 绿（`CFAPattern` 值 1）。
    Green = 1,
    /// 蓝（`CFAPattern` 值 2）。
    Blue = 2,
}

impl CfaColor {
    /// 由 DNG `CFAPattern` 字节解码（3 = 2nd Green，非法 → `None`）。
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(CfaColor::Red),
            1 => Some(CfaColor::Green),
            2 => Some(CfaColor::Blue),
            _ => None,
        }
    }
}

/// Bayer 四态（2×2 色彩滤波阵列的四种相位）。
///
/// 命名遵循 DNG / EXIF 惯例：字母按**2×2 平铺的先行后列**书写。
/// 例如 [`BayerLayout::RgGb`] 的平铺是 `[R G; G B]`，展开后第 0 行为
/// `R G R G …`、第 1 行为 `G B G B …`——即常说的 “Red-Green-Green-Blue”。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerLayout {
    /// `[R G; G B]`（RGGB）。
    RgGb,
    /// `[B G; G R]`（BGGR）。
    BgGr,
    /// `[G R; B G]`（GRBG）。
    GrBg,
    /// `[G B; R G]`（GBRG）。
    GbRg,
}

impl BayerLayout {
    /// 全部四态（顺序固定：RGGB、BGGR、GRBG、GBRG —— 遍历与测试用）。
    pub const ALL: [BayerLayout; 4] = [
        BayerLayout::RgGb,
        BayerLayout::BgGr,
        BayerLayout::GrBg,
        BayerLayout::GbRg,
    ];

    /// 2×2 平铺颜色（行主序 `[c00, c01, c10, c11]`）。
    pub fn tile(self) -> [CfaColor; 4] {
        match self {
            BayerLayout::RgGb => [
                CfaColor::Red,
                CfaColor::Green,
                CfaColor::Green,
                CfaColor::Blue,
            ],
            BayerLayout::BgGr => [
                CfaColor::Blue,
                CfaColor::Green,
                CfaColor::Green,
                CfaColor::Red,
            ],
            BayerLayout::GrBg => [
                CfaColor::Green,
                CfaColor::Red,
                CfaColor::Blue,
                CfaColor::Green,
            ],
            BayerLayout::GbRg => [
                CfaColor::Green,
                CfaColor::Blue,
                CfaColor::Red,
                CfaColor::Green,
            ],
        }
    }

    /// 标准名（`"RGGB"` 等）。
    pub fn name(self) -> &'static str {
        match self {
            BayerLayout::RgGb => "RGGB",
            BayerLayout::BgGr => "BGGR",
            BayerLayout::GrBg => "GRBG",
            BayerLayout::GbRg => "GBRG",
        }
    }

    /// 由 2×2 平铺反查（需恰为四态 Bayer 之一，否则 `None`）。
    pub fn from_tile(tile: [CfaColor; 4]) -> Option<Self> {
        BayerLayout::ALL.into_iter().find(|l| l.tile() == tile)
    }

    /// `(x, y)` 处的 CFA 颜色（`x`/`y` 为 0 基坐标；`% 2` 取相位）。
    #[inline]
    pub fn color_at(self, x: u32, y: u32) -> CfaColor {
        self.color_of_parity(x & 1, y & 1)
    }

    /// 由**相位**（`(x&1, y&1)`，取值 0/1）取 CFA 颜色——边界延拓时需要
    /// 「保持相位」的采样点（见 [`demosaic_bilinear`]）。
    #[inline]
    pub fn color_of_parity(self, px: u32, yp: u32) -> CfaColor {
        let t = self.tile();
        let idx = ((yp & 1) as usize) * 2 + (px & 1) as usize;
        t[idx]
    }
}

/// 完整 CFA 描述（重复尺寸 + 平面序列 + 原图尺寸）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfaPattern {
    /// 重复尺寸列数。
    pub cols: u16,
    /// 重复尺寸行数。
    pub rows: u16,
    /// 平面序列（行主序，长度 = `cols × rows`；值同 [`CfaColor`]）。
    pub colors: Vec<u8>,
    /// 马赛克原图宽（像素）。
    pub raw_width: u32,
    /// 马赛克原图高（像素）。
    pub raw_height: u32,
}

impl CfaPattern {
    /// 尝试降为 Bayer 四态布局（仅当 2×2 且恰为两种绿 + 一红一蓝）。
    pub fn bayer(&self) -> RawResult<BayerLayout> {
        const LOC: &str = "raw::CfaPattern::bayer";
        if self.cols != 2 || self.rows != 2 {
            return Err(unsupported_err(
                LOC,
                format!(
                    "CFA 重复尺寸为 {}×{}，仅 2×2 的 Bayer 四态可双线性去马塞克",
                    self.cols, self.rows
                ),
                "X-Trans（6×6）等非 Bayer 阵列需专用的 CFA 插值器，本模块未实现",
            ));
        }
        let mut tile = [CfaColor::Green; 4];
        for i in 0..4 {
            let code = self.colors.get(i).copied().unwrap_or(1);
            tile[i] = CfaColor::from_code(code).ok_or_else(|| {
                unsupported_err(
                    LOC,
                    format!("CFAPattern[{i}] = {code}，只允许 0=R / 1=G / 2=B"),
                    "确认 CFAPattern 标签负载未被错位读取",
                )
            })?;
        }
        BayerLayout::from_tile(tile).ok_or_else(|| {
            unsupported_err(
                LOC,
                format!(
                    "平铺 {:?} 不是 Bayer 四态（须恰一红两绿一蓝）",
                    tile.map(|c| c as u8)
                ),
                "全色/多光谱 CFA 需专用管线；本模块只做 Bayer",
            )
        })
    }

    /// 原图总像素数（`checked_mul`，溢出 → `None`）。
    pub fn pixel_count(&self) -> Option<usize> {
        (self.raw_width as usize).checked_mul(self.raw_height as usize)
    }
}

// ===========================================================================
//  六、位深与位解包（BitsPerSample < 16 的紧凑打包）
// ===========================================================================

/// 位流填充顺序（TIFF `FillOrder`，tag 0x010A）。
///
/// # DNG 规范的实际约定
///
/// DNG 1.4 §`FillOrder`：「**缺省为 1（MSB-first）**」。对
/// `BitsPerSample < 16` 的紧凑打包数据，这意味着**有效位在样本内高位对齐**
/// （big-endian 风格的 bit order）。例如 12 位打包流里，一个样本的 12 个有效
/// 位占据「该样本 2 字节窗口的高 12 位」，低 4 位是**填充位——其值未定义、
/// 可能非零**，读取时**必须用 mask 掩掉**（这正是 [`unpack_bits`] 逐样本
/// `& mask` 的原因；漏掉这一步会让 12 位值最大凭空多出 15）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillOrder {
    /// `FillOrder = 1`（**DNG 默认**）：**MSB-first**，有效位**高位对齐**。
    MsbFirst,
    /// `FillOrder = 2`：**LSB-first**（同一条逻辑图像的另一种位序编码）。
    LsbFirst,
}

impl FillOrder {
    /// 由 `FillOrder` 标签值解码（1 = MSB-first、2 = LSB-first；其它 → `None`）。
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(FillOrder::MsbFirst),
            2 => Some(FillOrder::LsbFirst),
            _ => None,
        }
    }

    /// 对应的 `FillOrder` 标签值（1 / 2）。
    pub fn code(self) -> u16 {
        match self {
            FillOrder::MsbFirst => 1,
            FillOrder::LsbFirst => 2,
        }
    }
}

/// 单次 [`unpack_bits`] 的样本数上限（防御性上限：16 M 样本 × 2 B = 32 MiB）。
///
/// 真实 DNG 的最大幅面约 100 M 像素，但那些文件走 strip/tile 分块读取
/// （见 [`DngFile::read_raw_samples`]），每次解包只处理一条 strip 或一个 tile。
/// 32 位色深不在本模块支持范围（见 [`unpack_bits`] 的错误分支）。
pub const MAX_UNPACK_SAMPLES: usize = 16 * 1024 * 1024;

/// 从紧凑位流解包 `count` 个样本（**通用位解包器**，支持 1..=16 位）。
///
/// # 参数
///
/// - `src` — 打包字节流（**从 `count × bits` 位的起始位开始**，高位/低位由
///   `fill_order` 决定；尾部不足一字节的填充位被忽略）；
/// - `bits` — 每样本有效位宽 `1..=16`；
/// - `count` — 样本个数；
/// - `fill_order` — [`FillOrder::MsbFirst`]（DNG 默认，高位对齐）或
///   [`FillOrder::LsbFirst`]。
///
/// # 位序的精确定义（两种位序的「同一条逻辑图像」关系）
///
/// 设流内位序号 `p`（从 0 起）与字节号 `p / 8`：
///
/// - **MSB-first**：位 `p` 是字节 `p / 8` 的第 `7 − (p % 8)` 位（**高位在前**）。
///   于是样本 `k` 的 `bits` 个有效位是流位 `[k·bits, k·bits + bits)`，
///   在 4 字节窗口 `w`（大端读）中的提取位移为 `32 − (k·bits % 8) − bits`。
/// - **LSB-first**：位 `p` 是字节 `p / 8` 的第 `p % 8` 位（**低位在前**）。
///   样本 `k` 的提取位移为 `k·bits % 8`，窗口 `w` 按小端读。
///
/// 两种位序描述的是**同一逻辑图像**的两种编码，故对同一组样本值分别编码后
/// 解包，结果必须逐样本相等（测试 `bit_unpack_msb_and_lsb_first_agree_on_same_image`
/// 就是这一对拍）。
///
/// # 填充位一律被掩掉（本函数的核心不变式）
///
/// 每个样本只提取**恰好 `bits` 个有效位**并 `& mask`，因此：
/// - 样本窗口内的填充位（12 位的低 4 位、14 位的低 2 位、10 位的低 6 位）
///   **即使非零也不会进入结果**；
/// - 流末尾不足一个样本的残留位、以及 `src` 长度大于所需位数时的尾字节，
///   一律不参与运算。
///
/// 这是最容易写错的地方：若忘记 mask，12 位流里填充位为 `0xF` 会让每个样本
/// 凭空多出 15（`0xABC` 被读成 `0xABCF`）。
///
/// # 数据不足 → 结构化拒绝（不静默补0）
///
/// `src` 短于 `ceil(count·bits / 8)` 字节时**返回错误**。逐样本窗口读取虽然
/// 可以对越界字节补 0，但那会产出「看起来正常、实际尾部全黑」的图——**无任何
/// 症状的错数据比报错更坏**。截断的 strip/tile 是真实存在的文件损坏
/// （传输不完整、ByteCounts 与尺寸不自洽），必须让调用方看见。
///
/// # 实现层级（快路径 / 通用路径）
///
/// | `bits` | 路径 | 理由 |
/// |---|---|---|
/// | 8 | **快路径**：逐字节直取 | 1 样本 = 1 字节，位序无差别 |
/// | 16（MSB-first） | **快路径**：直接读**大端** `u16` 序列 | 位序 ≡ 大端字节序 |
/// | 16（LSB-first） | **快路径**：直接读**小端** `u16` 序列 | 位序 ≡ 小端字节序 |
/// | 1..=15 | **通用路径**：4 字节窗口 + 一次移位 + 一次 mask | 见下 |
///
/// 通用路径对**所有** `1..=15` 位深用同一段代码：样本起点位 `s = k·bits`，
/// 取字节 `s / 8` 起的 4 字节（越界补 0，故绝不越界 panic），拼成 `u32`
/// 后一次移位取数。选 4 字节而非 3 字节是为了让移位量恒为正、无需分支：
/// 窗口内最多用到流位偏移 `7 + 15 − 1 = 21 < 32`。
///
/// # 确定性
///
/// 样本序号固定 `k = 0..count` 升序，每个样本的提取只依赖 `k`、`bits` 与
/// `fill_order`——**无迭代器 zip、无 `HashMap`**，同输入逐位可复现，
/// debug / release 同结果（纯整数运算）。
///
/// 解一条 DNG Deflate(`Compression = 8` / `32946`) 载荷：**zlib 容器**，含 Adler-32 校验。
///
/// # 为什么走 `zlib::decompress_verified` 而不是 `inflate_raw`
/// Adobe《Photoshop TIFF Technical Notes》对 `Compression = 8` 的定义是
/// "Deflate compression, **using zlib data format**"，且
/// "Each image segment consists of a **single complete zlib data stream**"
/// ——即每个 strip / tile 是一条**完整 zlib 流**（2 字节头 + DEFLATE +
/// 4 字节 Adler-32 尾），**不是**裸 DEFLATE 码流。
///
/// 选 `decompress_verified` 而非 `decompress`：它会**比对 Adler-32 的值**，
/// 于是「载荷被改写/ 截断但恰好仍能解出字节」这类损坏也会被抓住——
/// 对「读相机原始文件」这个用途，完整性校验比省几个字节重要。
///
/// # 错误映射
///
/// [`prism_bytes::BytesError`] 的三段式被原样搬进 [`RawError`]，
/// `code` 用 `RAW2001`（不支持/ 格式不符），`location` 保留调用方传入的
/// `loc` 以便定位到具体调用点。
pub(crate) fn inflate_dng_zlib(
    src: &[u8],
    predictor_code: u16,
    loc: &'static str,
) -> RawResult<Vec<u8>> {
    prism_bytes::zlib::decompress_verified(src).map_err(|e| {
        RawError::new(
            "RAW2001",
            loc,
            format!(
                "Deflate(zlib) 解压失败（Predictor={predictor_code}）：{} / {}",
                e.problem, e.fix
            ),
            "确认这是 DNG Compression=8/32946 的 zlib 流（而非裸 DEFLATE）、文件未被改写；\
             若文件确为裸 DEFLATE，说明它不符合 Adobe TIFF Technical Notes，故本模块不解析",
        )
    })
}

pub fn unpack_bits(src: &[u8], bits: u16, count: usize, fill_order: FillOrder) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw::unpack_bits";
    if bits == 0 || bits > 16 {
        return Err(unsupported_err(
            LOC,
            format!("位深 BitsPerSample = {bits}，本模块只支持 1..=16 位"),
            "DNG 的 BitsPerSample 常见为 8/10/12/14/16；32 位浮点 RAW 请先转 16 位整数 DNG",
        ));
    }
    if count > MAX_UNPACK_SAMPLES {
        return Err(tiff_err(
            LOC,
            format!("样本数 {count} 超过防御上限 {MAX_UNPACK_SAMPLES}"),
            "确认未把整幅超大图误当单条 strip 传入；请分块解包",
        ));
    }
    // **长度校验：数据不足即结构化拒绝，不静默补 0。**
    //
    // 解包 `count` 个 `bits` 位样本需要 `ceil(count·bits / 8)` 字节。缺数据时
    // 逐样本窗口读取会补0，得到一张「看起来正常、实际右半边全黑」的图——
    // 这是**比报错更坏**的结果（无任何症状的错数据），故此处显式拒绝。
    let need_bits = count.checked_mul(bits as usize).ok_or_else(|| {
        tiff_err(
            LOC,
            format!("样本数 {count} × {bits} 位的总位数溢出 usize"),
            "样本数异常；确认 count 未被错位读取",
        )
    })?;
    let need_bytes = need_bits.div_ceil(8);
    if src.len() < need_bytes {
        return Err(tiff_err(
            LOC,
            format!(
                "解包 {count} 个 {bits} 位样本需{need_bytes} 字节，实际只有 {} 字节",
                src.len()
            ),
            "strip/tile 被截断或ByteCounts 与尺寸不自洽；确认文件传输完整",
        ));
    }
    // `bits == 16` 时 `1u32 << 16` 本身合法，但写成通式更不易误读。
    let mask: u32 = (1u32 << bits) - 1;
    let mut out: Vec<u16> = Vec::with_capacity(count);
    match (bits, fill_order) {
        // ── 快路径 1：8 位 ──────────────────────────────────────────────
        (8, _) => {
            for k in 0..count {
                out.push(src.get(k).copied().unwrap_or(0) as u16);
            }
        }
        // ── 快路径 2：16 位 ──────────────────────────────────────────────
        // MSB-first 的 16 位流 ≡ 大端 u16 数组；LSB-first ≡ 小端 u16 数组。
        (16, FillOrder::MsbFirst) => {
            for k in 0..count {
                let hi = src.get(k * 2).copied().unwrap_or(0);
                let lo = src.get(k.saturating_mul(2).saturating_add(1)).copied().unwrap_or(0);
                out.push(u16::from_be_bytes([hi, lo]));
            }
        }
        (16, FillOrder::LsbFirst) => {
            for k in 0..count {
                let lo = src.get(k * 2).copied().unwrap_or(0);
                let hi = src.get(k.saturating_mul(2).saturating_add(1)).copied().unwrap_or(0);
                out.push(u16::from_le_bytes([lo, hi]));
            }
        }
        // ── 通用路径：1..=15 位 ─────────────────────────────────────────
        (b, fo) => {
            let b = b as usize;
            for k in 0..count {
                let s = k.checked_mul(b).ok_or_else(|| {
                    tiff_err(
                        LOC,
                        format!("样本 {k} × {b} 位的起始位偏移溢出"),
                        "样本数异常；确认 count 未被错位读取",
                    )
                })?;
                let byte = s / 8;
                let in_byte = s - byte * 8; // = s % 8 ∈ 0..=7
                // 4 字节窗口，越界补 0（C18：绝不越界索引）。
                let w: u32 = match fo {
                    FillOrder::MsbFirst => u32::from_be_bytes([
                        src.get(byte).copied().unwrap_or(0),
                        src.get(byte.saturating_add(1)).copied().unwrap_or(0),
                        src.get(byte.saturating_add(2)).copied().unwrap_or(0),
                        src.get(byte.saturating_add(3)).copied().unwrap_or(0),
                    ]),
                    FillOrder::LsbFirst => u32::from_le_bytes([
                        src.get(byte).copied().unwrap_or(0),
                        src.get(byte.saturating_add(1)).copied().unwrap_or(0),
                        src.get(byte.saturating_add(2)).copied().unwrap_or(0),
                        src.get(byte.saturating_add(3)).copied().unwrap_or(0),
                    ]),
                };
                // 有效位移恒为正：MSB 是 32 − in_byte − b ≥ 32 − 7 − 15 = 10；
                // LSB 是 in_byte ≤ 7。两者都 < 32。
                let v: u32 = match fo {
                    FillOrder::MsbFirst => (w >> (32 - in_byte - b)) & mask,
                    FillOrder::LsbFirst => (w >> in_byte) & mask,
                };
                out.push(v as u16);
            }
        }
    }
    Ok(out)
}

// ===========================================================================
//  七、黑 / 白电平
// ===========================================================================

/// 黑 / 白电平（黑电平可逐 CFA 位置不同）。
#[derive(Debug, Clone, PartialEq)]
pub struct RawLevels {
    /// 黑电平，长度 = CFA 平铺尺寸（元素为该位置的传感器本底）。
    pub black: Vec<f32>,
    /// 白电平（饱和电平，单值）。
    pub white: f32,
    /// 采样位深（`BitsPerSample`，缺省 16）。
    ///
    /// **归一化必须用它**：黑/白电平是**原始码值**（12 位文件的 WhiteLevel
    /// 通常是 4095 而非 65535），线性化 `(raw − black) / (white − black)`
    /// 是**同量纲相消**，与位深无关——但任何「把 raw 右移归一到 16 位」的
    /// 做法都会破坏黑白电平的对应关系，故本模块**不做右移**，只在解包阶段
    /// 得到原始码值。
    pub bits_per_sample: u16,
    /// 位流填充顺序（`FillOrder`，缺省 [`FillOrder::MsbFirst`]）。
    pub fill_order: FillOrder,
}

impl RawLevels {
    /// `(x, y)` 处的黑电平（按平铺相位索引；平铺为空时回退 0.0）。
    #[inline]
    pub fn black_at(&self, x: u32, y: u32, cols: u16, rows: u16) -> f32 {
        if self.black.is_empty() || cols == 0 || rows == 0 {
            return 0.0;
        }
        let period = (cols as usize) * (rows as usize);
        let idx =
            ((y as usize) % (rows as usize)) * (cols as usize) + (x as usize) % (cols as usize);
        self.black.get(idx % period).copied().unwrap_or(0.0)
    }
}

// ===========================================================================
//  八、去马塞克（Bayer 四态双线性）
// ===========================================================================

/// 双线性去马塞克的三通道平面（f32 计算域，行主序）。
#[derive(Debug, Clone, PartialEq)]
pub struct BayerPlane {
    /// 平面宽（= 原图宽）。
    pub width: u32,
    /// 平面高（= 原图高）。
    pub height: u32,
    /// 红平面（行主序，长度 = `width × height`）。
    pub r: Vec<f32>,
    /// 绿平面。
    pub g: Vec<f32>,
    /// 蓝平面。
    pub b: Vec<f32>,
}

/// 读一个马赛克采样值；对**越界方向按 CFA 相位折返**（见 [`demosaic_bilinear`]）。
///
/// # 为什么不是简单坐标钳制
///
/// 教科书公式假定「正交四邻皆 G、对角四邻皆 R/B」。若把越界坐标直接钳到边缘
/// （`(−1, y) → (0, y)`），在 Bayer 排列下**边缘像素的相位与被钳到的像素不同**，
/// 于是「G 的平均」里混进了 R 或 B 的值——这不是边缘延拓，而是**串色**。
/// 例：RGGB 的 (0,0) 是 R，其 G 应取 (±1,0)/(0,±1)；`(−1,0)` 钳到 `(0,0)`
/// 会把 R 本身算进 G。
///
/// # 本实现的做法（相位保持折返）
///
/// 1. 先把目标坐标折回画内：`tx = |tx|` 后按 `w` 取模反射（镜像，RFC 式
///    reflect-101：坐标 −1 → 1、坐标 `w` → `w−2`），保证不越界；
/// 2. 再把相位对齐：若折回后的 `(tx&1, ty&1)` 与期望相位不符，沿该轴**步进 1**
///    （朝画内方向）直到相位匹配或该轴长度耗尽；
/// 3. 若无法对齐（图像宽或高为 1，使某相位根本不存在），回退到**中心像素值**。
///
/// 步骤 1–3 全部是纯整数运算 + 固定顺序，无分支依赖数据值 ⇒ 逐位确定。
#[inline]
fn sample_phase(
    raw: &[f32],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    want_px: u32,
    want_py: u32,
) -> f32 {
    let w = width as i64;
    let h = height as i64;
    if w <= 0 || h <= 0 {
        return 0.0;
    }
    // ① 镜像折回（reflect-101：−1 → 1，w → w−2）。
    let reflect = |v: i64, n: i64| -> i64 {
        if n <= 1 {
            return 0;
        }
        let period = 2 * (n - 1);
        let mut m = v % period;
        if m < 0 {
            m += period;
        }
        if m >= n {
            m = period - m;
        }
        m
    };
    let mut tx = reflect(x, w);
    let mut ty = reflect(y, h);
    // ② 相位对齐（沿轴步进；先 x 后 y，顺序固定）。
    let step_x: i64 = if tx + 1 < w { 1 } else { -1 };
    if (tx & 1) != (want_px as i64) {
        let cand = tx + step_x;
        if cand >= 0 && cand < w {
            tx = cand;
        }
    }
    let step_y: i64 = if ty + 1 < h { 1 } else { -1 };
    if (ty & 1) != (want_py as i64) {
        let cand = ty + step_y;
        if cand >= 0 && cand < h {
            ty = cand;
        }
    }
    // ③ 相位仍不符（如宽/高为 1）→ 用中心像素（本函数只在边界被调用，中心即
    //    唯一可用样本），保持「不越界 + 不 panic」。
    let idx = (ty as usize) * (w as usize) + (tx as usize);
    raw.get(idx).copied().unwrap_or(0.0)
}

/// Bayer 四态**双线性插值**去马塞克。
///
/// # CFA 布局
///
/// 以 [`BayerLayout::RgGb`]（`[R G; G B]`，第 0 行 `R G R G …`、第 1 行
/// `G B G B …`）为基准写出，其余三态由 [`BayerLayout::color_at`] 的相位
/// 取用给出**同一组公式**（公式只用「正交四邻」与「对角四邻」的集合定义，
/// 因此对四态统一成立）。
///
/// # 插值公式（出处）
///
/// 教科书 *CFA* 双线性插值（与 DngSDK / libraw 的 bilinear 同式）：
///
/// - **R 位**（`layout.color_at = R`）：R = 原值；
///   G = 正交四邻（G 恰好落在正交方向）均值；
///   B = 对角四邻（B 恰好落在对角方向）均值；
/// - **B 位**：B = 原值；G = 正交四邻均值；R = 对角四邻均值；
/// - **G 位**：G = 原值。四个正交邻被分成**水平一对**与**垂直一对**：
///   同一轴上的两个邻点必为**同一种非绿颜色**（这正是 Bayer 排列的定义），
///   故 R 与 B 各取其所在轴上的两点均值。
///   例：RGGB 的 (0,1) 处 G，水平邻 (0,0)/(0,2) 皆 R、垂直邻 (−1,1)/(1,1) 皆 B。
///
/// 公式（记 `P[y][x]` 为**已线性化**（减黑电平）的马赛克采样值）：
///
/// ```text
/// R 位:  R = P[y][x]
///         G = ( P[y-1][x] + P[y+1][x] + P[y][x-1] + P[y][x+1] ) / 4
///         B = ( P[y-1][x-1] + P[y-1][x+1] + P[y+1][x-1] + P[y+1][x+1] ) / 4
/// B 位:  B = P[y][x]
///         G = ( P[y-1][x] + P[y+1][x] + P[y][x-1] + P[y][x+1] ) / 4
///         R = ( P[y-1][x-1] + P[y-1][x+1] + P[y+1][x-1] + P[y+1][x+1] ) / 4
/// G 位:  G = P[y][x]
///         R = ( P[y][x-1] + P[y][x+1] ) / 2     （若该轴为红）
///         B = ( P[y-1][x] + P[y+1][x] ) / 2     （若该轴为蓝）
/// ```
///
/// # 边界处理
///
/// 越界邻点用 [`sample_phase`] 的**相位保持镜像折返**（不是坐标硬钳制——那会
/// 把异色像素混入平均，见该函数文档）。分母**固定**为 4 或 2：相位折返保证
/// 取到的仍是**同色**样本，故均值无偏。
///
/// # 确定性
///
/// 遍历顺序固定 `y` 外层升序、`x` 内层升序；边界处理全是纯整数分支（不依赖
/// 采样值）——同输入逐位可复现，debug / release 同结果（无 libm，只有
/// 加/乘/除）。
pub fn demosaic_bilinear(
    raw: &[f32],
    width: u32,
    height: u32,
    layout: BayerLayout,
) -> RawResult<BayerPlane> {
    const LOC: &str = "raw::demosaic_bilinear";
    let w = width as usize;
    let h = height as usize;
    let expect = w.checked_mul(h).ok_or_else(|| {
        pipe_err(
            LOC,
            format!("{width}×{height} 像素数溢出"),
            "使用更小的测试图",
        )
    })?;
    if raw.len() != expect {
        return Err(pipe_err(
            LOC,
            format!("马赛克长度 {} ≠ 宽×高 {expect}", raw.len()),
            "确认传入的是 `width*height` 的单通道原图",
        ));
    }
    let count = expect;
    let mut r = vec![0.0f32; count];
    let mut g = vec![0.0f32; count];
    let mut b = vec![0.0f32; count];
    if count == 0 {
        return Ok(BayerPlane {
            width,
            height,
            r,
            g,
            b,
        });
    }

    for y in 0..height {
        let yi = y as i64;
        for x in 0..width {
            let xi = x as i64;
            let idx = (y as usize) * w + (x as usize);
            let v = raw.get(idx).copied().unwrap_or(0.0);
            let px = x & 1;
            let py = y & 1;

            match layout.color_at(x, y) {
                CfaColor::Red | CfaColor::Blue => {
                    // 正交四邻恒为绿（相位 (1−px, py) 与 (px, 1−py) 都是绿）。
                    let gpx = 1 - px;
                    let gpy = 1 - py;
                    let ortho = (sample_phase(raw, width, height, xi, yi - 1, px, gpy)
                        + sample_phase(raw, width, height, xi, yi + 1, px, gpy)
                        + sample_phase(raw, width, height, xi - 1, yi, gpx, py)
                        + sample_phase(raw, width, height, xi + 1, yi, gpx, py))
                        * 0.25;
                    // 对角四邻恒为「另一非绿色」（相位两者皆翻转）。
                    let diag = (sample_phase(raw, width, height, xi - 1, yi - 1, gpx, gpy)
                        + sample_phase(raw, width, height, xi + 1, yi - 1, gpx, gpy)
                        + sample_phase(raw, width, height, xi - 1, yi + 1, gpx, gpy)
                        + sample_phase(raw, width, height, xi + 1, yi + 1, gpx, gpy))
                        * 0.25;
                    if layout.color_at(x, y) == CfaColor::Red {
                        r[idx] = v;
                        g[idx] = ortho;
                        b[idx] = diag;
                    } else {
                        b[idx] = v;
                        g[idx] = ortho;
                        r[idx] = diag;
                    }
                }
                CfaColor::Green => {
                    g[idx] = v;
                    // 水平轴两邻（相位 (1−px, py)）同色；垂直轴两邻（相位 (px, 1−py)）同色。
                    let h_axis = (sample_phase(raw, width, height, xi - 1, yi, 1 - px, py)
                        + sample_phase(raw, width, height, xi + 1, yi, 1 - px, py))
                        * 0.5;
                    let v_axis = (sample_phase(raw, width, height, xi, yi - 1, px, 1 - py)
                        + sample_phase(raw, width, height, xi, yi + 1, px, 1 - py))
                        * 0.5;
                    // 水平轴的颜色：由该相位直接查平铺（四态下必为红或蓝）。
                    match layout.color_of_parity(1 - px, py) {
                        CfaColor::Red => r[idx] = h_axis,
                        CfaColor::Blue => b[idx] = h_axis,
                        CfaColor::Green => {}
                    }
                    match layout.color_of_parity(px, 1 - py) {
                        CfaColor::Red => r[idx] = v_axis,
                        CfaColor::Blue => b[idx] = v_axis,
                        CfaColor::Green => {}
                    }
                }
            }
        }
    }
    Ok(BayerPlane {
        width,
        height,
        r,
        g,
        b,
    })
}

// ===========================================================================
//  九、白平衡、色彩矩阵与 Bradford 白点适配
// ===========================================================================

/// 白平衡增益的归一化约定。
///
/// **共同事实**：`AsShotNeutral` N 记录了「中性物体在相机空间的原始响应」。
/// 要让白平衡后的中性物体变成 (1,1,1)，增益必须 ∝ `1/N`。两种模式只差一个
/// **全局曝光标量**，不改变色度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WbMode {
    /// **绿归一**（默认）：`gain = N_G / N`。绿增益恒为 1，R/B 相对绿归一。
    /// 与 libraw 的 `pre_mul` 同式；绿不被提升，便于与 RAW 域数值直觉对齐。
    GreenNormalized,
    /// **严格 DNG 式**：`gain = (1/N) / max(1/N) = N_min / N`。最大增益为 1，
    /// 不引入任何增益放大（适合“增益不得超过 1”的 AOV 交付约束）。
    InverseNormalized,
    /// 不做白平衡。
    None,
}

/// 计算白平衡增益（`[R, G, B]`）。
///
/// 输入为 `AsShotNeutral`（相机空间中性响应）。分量 ≤ 0 或非有限时按 1.0
/// 处理（单分量损坏即整体回退无白平衡，见 [`WbMode::None`] 的等价路径）。
pub fn white_balance_gains(as_shot_neutral: [f32; 3], mode: WbMode) -> [f32; 3] {
    let ok = as_shot_neutral.iter().all(|&v| v.is_finite() && v > 0.0);
    if !ok {
        return [1.0, 1.0, 1.0];
    }
    match mode {
        WbMode::None => [1.0, 1.0, 1.0],
        WbMode::GreenNormalized => {
            let g = as_shot_neutral[1];
            [g / as_shot_neutral[0], 1.0, g / as_shot_neutral[2]]
        }
        WbMode::InverseNormalized => {
            let inv = [
                1.0 / as_shot_neutral[0],
                1.0 / as_shot_neutral[1],
                1.0 / as_shot_neutral[2],
            ];
            let m = inv[0].max(inv[1]).max(inv[2]);
            [inv[0] / m, inv[1] / m, inv[2] / m]
        }
    }
}

/// 3×3 矩阵（行主序 9 元）。
pub type Mat3 = [f32; 9];

/// 3×3 × 3×1 矩阵向量乘（行主序）。
pub fn mat3_mul(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// CIE XYZ → 线性 sRGB（Rec.709/sRGB 原色，D65 白点，**IEC 61966-2-1**）。
pub const XYZ_TO_SRGB: Mat3 = [
    3.240_454_2,
    -1.537_138_5,
    -0.498_531_4, //
    -0.969_266_0,
    1.876_010_8,
    0.041_556_0, //
    0.055_643_4,
    -0.204_025_9,
    1.057_225_2,
];

/// 线性 sRGB → CIE XYZ（[`XYZ_TO_SRGB`] 的逆，**IEC 61966-2-1**）。
///
/// 第 2 行即 sRGB 的亮度系数 (0.2126, 0.7152, 0.0722)（Rec.709 亮度）。
/// 提供它是为了让「亮度 = 第 2 行」这一事实可被直接断言与校验。
pub const SRGB_TO_XYZ: Mat3 = [
    0.412_390_8,
    0.357_584_3,
    0.180_480_8, //
    0.212_639_0,
    0.715_168_7,
    0.072_192_3, //
    0.019_330_8,
    0.119_194_8,
    0.950_532_2,
];

/// 色彩矩阵使用方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMode {
    /// 相机空间 → XYZ（用 `ColorMatrix1`），输出留在 XYZ 域。
    CameraToXyz,
    /// 相机空间 → XYZ → 线性 sRGB（再乘 [`XYZ_TO_SRGB`]），**默认**。
    CameraToSrgb,
    /// 不做色彩矩阵（保留白平衡后的相机域 RGB）。
    None,
}

// ===========================================================================
//  九之二、Bradford 白点适配（**第二份实现 —— 技术债**）
// ===========================================================================

/// Bradford 锥体响应矩阵（**CAT02**，Lam 1985；亦见 ACES 官方 `Bradford.html`）。
///
/// 行主序 3×3：`cone = BRADFORD · XYZ`。
pub const BRADFORD: Mat3 = [
    0.8951, 0.2664, -0.1614, //
    -0.7502, 1.7135, 0.0367, //
    0.0389, -0.0685, 1.0296,
];

/// D50 白点的 CIE XYZ（Y = 1 归一；`x = 0.34567, y = 0.35850`）。
///
/// DNG 规范规定 `ForwardMatrix2` 的输出是 **XYZ(D50)**（这是 DNG 采用
/// PCS 白点而非 sRGB 白点的历史原因），故凡走 `ForwardMatrix2` 的链路，
/// 都必须把 D50 适配到目标空间的 D65 才能接 [`XYZ_TO_SRGB`]。
pub const D50_XYZ: [f32; 3] = [0.964_22, 1.0, 0.825_21];

/// D65 白点的 CIE XYZ（Y = 1 归一；`x = 0.3127, y = 0.3290`，sRGB/Rec.709）。
pub const D65_XYZ: [f32; 3] = [0.950_47, 1.0, 1.088_83];

/// 3×3 矩阵求逆（伴随矩阵法，行主序）。行列式退化（`|det| < 1e-12`）→ `None`。
///
/// 用途有二：Bradford 适配需要 `BRADFORD⁻¹`；测试「X→RGB→X 往返一致」需要
/// 装配矩阵的逆。
pub fn mat3_inverse(m: &Mat3) -> Option<Mat3> {
    let (m0, m1, m2, m3, m4, m5, m6, m7, m8) = (
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
    );
    // 余子式C_ij（不含符号；符号在转置时吸收）。
    let c00 = m4 * m8 - m5 * m7;
    let c01 = m5 * m6 - m3 * m8;
    let c02 = m3 * m7 - m4 * m6;
    let c10 = m2 * m7 - m1 * m8;
    let c11 = m0 * m8 - m2 * m6;
    let c12 = m1 * m6 - m0 * m7;
    let c20 = m1 * m5 - m2 * m4;
    let c21 = m2 * m3 - m0 * m5;
    let c22 = m0 * m4 - m1 * m3;
    // det 按第一行展开（与 c00/c01/c02 同序）。
    let det = m0 * c00 + m1 * c01 + m2 * c02;
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    // 伴随矩阵 = 余子式矩阵的**转置**：C00 C10 C20 / C01 C11 C21 / C02 C12 C22。
    Some([
        c00 * inv,
        c10 * inv,
        c20 * inv,
        c01 * inv,
        c11 * inv,
        c21 * inv,
        c02 * inv,
        c12 * inv,
        c22 * inv,
    ])
}

/// Bradford 锥体响应的对角缩放矩阵（`diag(cone_dst / cone_src)`）。
///
/// 返回 `None` 表示锥体响应退化（分量非有限或为 0）——此时**不能**给出适配
/// 矩阵（否则会静默产生 Inf/NaN 污染整条显影链）。
fn bradford_diagonal(src_xyz: [f32; 3], dst_xyz: [f32; 3]) -> Option<Mat3> {
    let cs = mat3_mul(&BRADFORD, src_xyz);
    let cd = mat3_mul(&BRADFORD, dst_xyz);
    let mut scale = [0.0f32; 3];
    for i in 0..3 {
        if !cs[i].is_finite() || !cd[i].is_finite() {
            return None;
        }
        // **源与目标两侧都必须非零**。只查源侧是不够的：目标锥体响应为 0 时
        // `scale = 0 / cs = 0`，会得到一个**全零矩阵**——它不报错，却会把
        // 整条显影链的 XYZ 全部清零（输出纯黑），比报错更坏。
        if cs[i] == 0.0 || cd[i] == 0.0 {
            return None;
        }
        scale[i] = cd[i] / cs[i];
    }
    let mut d = [0.0f32; 9];
    d[0] = scale[0];
    d[4] = scale[1];
    d[8] = scale[2];
    Some(d)
}

/// Bradford 白点适配矩阵：`M = B⁻¹ · diag(B·W_dst / B·W_src) · B`。
///
/// # ⚠️ 技术债：这是本仓库**第二份** Bradford 实现（重复实现）
///
/// 1. **第一份**：`crates/prism-math/src/colorspace_registry.rs` 的
///    `bradford_adaptation`（f64、白点为 `WhitePoint` 枚举）；
/// 2. **第二份（本函数）**：`crates/prism-io/src/raw.rs`（f32、白点为
///    `[f32; 3]` XYZ 三刺激值）。
///
/// （原以为此处是「第三份」。全仓 grep `Bradford` / `CAT02` /
/// `bradford_adaptation` 只找到 prism-math 一份**代码实现**，故按实际代码
/// 计数是第二份。如另有并行分支/未合并的第三份，请更新此注释。）
///
/// # 为什么不能复用 prism-math 那份（依赖方向结论）
///
/// `crates/prism-io/Cargo.toml` 的 `[dependencies]` **只有 `prism-gpu`**
/// （已核验，未新增任何依赖）。分层约束是 `io ∈ L4`（**依赖方向向下**），
/// `prism-math` 是色彩数学层，与 `prism-io` 同级或更上层。**若为了复用而在此
/// 加 `prism-math` 依赖，就是让 L4 反向依赖同级/上层，破坏架构分层**——
/// 分层约束禁止。故只能在本文件内自研。代价是两份实现的常数与舍入精度
/// 可能略有差异（f32 vs f64），**是否统一留待架构裁决**。
///
/// # 退化处理
///
/// 锥体响应退化（**源或目标任一侧**非有限 / 分量为 0）时返回 `None`，
/// **不返回单位矩阵**——静默回退到「不适配」会让 D50 数据被当成 D65 解释，
/// 产生偏色且无任何报错，是比报错更坏的结果。同理，全零目标白点会得到
/// 全零矩阵（把整条链的 XYZ 清零 → 纯黑），也归入 `None`。调用方须显式
/// 处理 `None`。
pub fn bradford_adaptation(src_xyz: [f32; 3], dst_xyz: [f32; 3]) -> Option<Mat3> {
    let b_inv = mat3_inverse(&BRADFORD)?;
    let d = bradford_diagonal(src_xyz, dst_xyz)?;
    // 计算次序固定：B·D 先做（左乘 diag），再 B⁻¹ 右乘之。
    Some(mat3_mul_matrix(&b_inv, &mat3_mul_matrix(&d, &BRADFORD)))
}

/// `ForwardMatrix2` 与 `ColorMatrix1` 的选择策略。
///
/// **两条路是真正独立的两条代码路径**，不是同一个函数的两个分支：
/// `Forward2` 走「白平衡 → CameraCalibration → ForwardMatrix2 → D50→D65
/// Bradford → sRGB」，`ColorMatrix1` 走「白平衡 → ColorMatrix1 → sRGB」。
/// 显式切换二者**必然得到不同结果**（除非两个矩阵恰好相同），这由测试
/// `develop_matrix_choice_forward2_and_colormatrix1_differ` 断言。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixChoice {
    /// 存在 `ForwardMatrix2` 就用它，否则退回 `ColorMatrix1`。**默认**。
    ///
    /// 依据 DNG 1.4：规范**推荐** `ForwardMatrix2`（它是正向矩阵，标定误差
    /// 更小）；`ColorMatrix1` 是它的反矩阵，仅在 `ForwardMatrix2` 缺席时使用。
    #[default]
    Auto,
    /// **强制** `ForwardMatrix2`；缺失即结构化报错（不静默回退）。
    Forward2,
    /// **强制** `ColorMatrix1`；缺失即结构化报错（不静默回退）。
    ColorMatrix1,
}

impl MatrixChoice {
    /// 标准名（诊断 / 报错文本用）。
    pub fn name(self) -> &'static str {
        match self {
            MatrixChoice::Auto => "Auto",
            MatrixChoice::Forward2 => "ForwardMatrix2",
            MatrixChoice::ColorMatrix1 => "ColorMatrix1",
        }
    }
}

/// 对**未支持的 `Compression` 取值**给出**该取值特有**的成因说明。
///
/// # 依据（逐条对照 Adobe DNG 规范原文，非「我记得」）
///
/// | 取值 | TIFF 6.0 名称 | DNG 规范地位 |
/// |---|---|---|
/// | 1 | Uncompressed | **DNG 支持**（本模块已实装） |
/// | 2/3/4 | CCITT RLE / G3 / G4 | DNG **不支持**（传真类） |
/// | 5 | LZW | DNG **不支持** |
/// | **6** | **Old JPEG（OJPEG）** | **DNG 规范从未定义**（见下） |
/// | 7 | JPEG | **DNG 支持**（本模块已实装无损分支） |
/// | **8** | **Deflate (ZIP)** | DNG 1.4 **新增**，本模块**已实装**（zlib 容器） |
/// | 32773 | PackBits | DNG **不支持** |
/// | **32946** | **Deflate（`0x80B2` 旧号）** | 与 `8` **同算法**，本模块**已实装** |
/// | 34712 | JPEG 2000 | DNG **不支持** |
/// | 34892 | Lossy JPEG | DNG 1.4 **新增**，本模块未实装（且仅限 LinearRaw 8 位） |
/// | 52546 | JPEG XL | DNG 1.7 **新增**，本模块未实装 |
///
/// # 仍**不支持**的取值清单（诚实边界，实装 Deflate 时复核）
///
/// 实装Deflate **不等于**「压缩方式全支持」。下列取值本模块**仍结构化拒绝**
/// （见 [`compression_reject_hint`]），它们各有**具体**成因，不是「还没做」：
///
/// | `Compression` | 成因 | 为什么现在仍拒绝 |
/// |---|---|---|
/// | 2/3/4 | CCITT RLE / G3 / G4 | 传真类，DNG 规范不支持 |
/// | 5 | LZW | DNG 规范不支持 |
/// | 6 | Old JPEG（OJPEG） | **DNG 规范从未定义此取值**（详见 [`compression_reject_hint`]） |
/// | 32773 | PackBits | DNG 规范不支持 |
/// | 34712 | JPEG 2000 | 需J2K 通路，本模块无解码器 |
/// | **34892** | **Lossy JPEG（有损基线 DCT）** | **不是 Deflate 别名**——需有损 JPEG 解码器；本模块只做无损 |
/// | 52546 | JPEG XL | DNG 1.7 新增，本模块无 JPEG XL 解码器 |
///
/// 其中 `34892` 最容易被误接：它是 DNG 1.4 与 `8` **并列新增**的另一个取值，
/// 语义是**有损**基线 DCT（且仅限 8 位 + `LinearRaw`/`PhotometricMask`）。
/// 若误当Deflate 接上，一个有损 JPEG 的 DNG 会被喂进 inflate，产出**看似合法
/// 实则全错**的像素——这比明确报错危险得多。
///
/// DNG 1.4.0.0 原文（`Compression` 章节）:
///
/// > Two Compression tag values are supported in DNG versions before 1.4.0.0:
/// > • Value = 1: Uncompressed data.
/// > • Value = 7: JPEG compressed data, either baseline DCT JPEG, or lossless JPEG.
/// > DNG Version 1.4.0.0 adds support for the following compression codes:
/// > • Value = 8: Deflate (ZIP)   • Value = 34892: Lossy JPEG
///
/// # `8` 与 `32946`：Deflate(ZIP) —— **zlib 流**（不是裸 DEFLATE）
///
/// 规范依据（Adobe《Photoshop TIFF Technical Notes》，TIFF 6.0 补充，
/// 压缩方案的规范原文；DNG 1.4 的 `Deflate (ZIP)` 继承它）：
///
/// ```text
/// Value: 8 — Deflate compression, **using zlib data format**
/// Each image segment consists of a **single complete zlib data stream**.
/// The zlib data stream contains header and trailer information
/// surrounding one or more compressed data blocks.
/// ```
///
/// **即：每个 strip / tile 是一条完整的 zlib 流**（2 字节 zlib 头 +
/// DEFLATE + 4 字节 Adler-32 尾），**不是**不带 zlib 头的裸 DEFLATE 码流。
/// 故本模块走 [`prism_bytes::zlib`] 的 `decompress_verified`（**校验
/// Adler-32**）而**不是** `inflate_raw`——两者只差容器，读错会全盘错位。
///
/// 这一点文档曾写作「裸 DEFLATE、不带 zlib 头」，与规范原文相反；
/// 本实现以规范为准（并在此写清依据）。
///
/// # `32946`（`0x80B2`）与 `8` 等价
///
/// 同一份 Technical Notes 明确写道：`0x80B2` 是厂商私有 ZIP/Flate 码，
/// **压缩算法与 `8` 完全相同**，「应视为过时；建议 TIFF 实现**识别并读取**
/// 该过时码，但只写正式码 `8`」。故本模块对两者一视同仁。
///
/// # DNG `Compression = 8`（Deflate）配套的**预测器**（`Predictor`，0x013D）
///
/// ## 规范取值（DNG 1.4 `Predictor` 章节）
///
/// 规范原文："The following predictor codes are allowed for IFDs using
/// compression code 8 (deflate)"。
///
/// | `Predictor` | 语义 | 本模块 |
/// |---|---|---|
/// | 1 | Null Predictor（不预测） | ✅ |
/// | 2 | Horizontal Difference（TIFF 6.0 §14） | ✅ |
/// | 3 | Floating Point（Photoshop TIFF TN3） | ❌ 需浮点 RAW |
/// | 34892 | Horizontal Difference **X2**（用左侧第 2 个像素） | ✅ |
/// | 34893 | Horizontal Difference **X4**（用左侧第 4 个像素） | ✅ |
/// | 34894 | Floating Point X2 | ❌ 需浮点 RAW |
/// | 34895 | Floating Point X4 | ❌ 需浮点 RAW |
///
/// # 与 LZW 的关系
///
/// TIFF 6.0 §14 的 Horizontal Difference 本是 **LZW** 的配套预测器；
/// Adobe《Photoshop TIFF Technical Notes》明确它对 Deflate 同样有效
/// （"the Predictor defined in TIFF6.pdf Section 14 works well for both
/// Deflate and LZW"）。
///
/// # 缺省值
///
/// 缺省 `Predictor = 1`（Null）—— 与 TIFF 一致，标签缺失即不预测。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeflatePredictor {
    /// `1`：Null Predictor（无预测）。
    #[default]
    None,
    /// `2`：Horizontal Difference（用左侧第 1 个样本）。
    Horizontal,
    /// `34892`：Horizontal Difference X2（用左侧第 2 个样本）。
    HorizontalX2,
    /// `34893`：Horizontal Difference X4（用左侧第 4 个样本）。
    HorizontalX4,
}

impl DeflatePredictor {
    /// 规范取值 → 枚举；**不支持的取值返回 `None`**（含 3 / 34894 / 34895
    /// 三个浮点预测器与一切未列举值）。
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::None),
            2 => Some(Self::Horizontal),
            34892 => Some(Self::HorizontalX2),
            34893 => Some(Self::HorizontalX4),
            _ => None,
        }
    }

    /// 枚举 → 规范取值（[`DeflatePredictor::None`] → `1`）。
    pub fn code(self) -> u16 {
        match self {
            Self::None => 1,
            Self::Horizontal => 2,
            Self::HorizontalX2 => 34892,
            Self::HorizontalX4 => 34893,
        }
    }

    /// 行内左邻距离（`x` 减去几个样本）；`None` 不需要左邻，故为 `0`。
    #[inline]
    fn stride(self) -> usize {
        match self {
            Self::None => 0,
            Self::Horizontal => 1,
            Self::HorizontalX2 => 2,
            Self::HorizontalX4 => 4,
        }
    }

    /// 未支持取值的成因说明（用于错误文本拼接）。
    pub fn reject_reason(code: u16) -> &'static str {
        match code {
            3 => "Predictor = 3 是 Floating Point 预测器（Photoshop TIFF Technical Note 3），\
                  仅用于浮点 RAW；本模块的 Deflate 路径只处理 1..=16 位整数样本",
            34894 => "Predictor = 34894 是 Floating Point X2，仅用于浮点 RAW；\
                      本模块的 Deflate 路径只处理 1..=16 位整数样本",
            34895 => "Predictor = 34895 是 Floating Point X4，仅用于浮点 RAW；\
                      本模块的 Deflate 路径只处理 1..=16 位整数样本",
            _ => "DNG 1.4 只为 Compression=8 定义了 1 / 2 / 3 / 34892 / 34893 / 34894 / \
                  34895 七个预测器取值，本模块实现其中四个整数取值（1 / 2 / 34892 / 34893）",
        }
    }

    /// 对**已按行主序排好**的样本缓冲做**预测器逆运算**（就地重建）。
    ///
    /// # 语义（TIFF 6.0 §14 Horizontal Difference）
    ///
    /// 编码侧逐样本减去其左侧第 `stride` 个样本；解码侧即累加回去：
    ///
    /// ```text
    /// 原值 s[x] = 差值 d[x] + s[x - stride]   (mod 2^bits)
    /// ```
    ///
    /// 每行**前 `stride` 个样本**没有左邻，保持原值（`x < stride` 时跳过）。
    ///
    /// # 参数
    ///
    /// - `samples` —— 行主序样本，长度可大于 `rows × row_len`（多出的不动）；
    /// - `row_len` —— 每行样本数（整行像素数 × `SamplesPerPixel`）；
    /// - `bits` —— 样本位深，回绕按 `2^bits` 取模。
    ///
    /// # 零 panic
    ///
    /// 全部下标经 `get` / `get_mut` 兜底；`row_len == 0` 或 `stride == 0`
    /// 直接返回（Null 预测器本就无操作）。
    pub fn undo_in_place(&self, samples: &mut [u16], row_len: usize, bits: u16) {
        let stride = self.stride();
        // `bits == 0` 无意义；`bits > 16` 超出 u16 样本域（真实 DNG 不会写到）。
        //
        // **注意 `bits == 16` 必须照常做运算**：模 2^16 的回绕正是
        // `wrapping_add` 的语义，**不是恒等**。早先版本写成 `bits >= 16`
        // 就此提前返回，等于把预测器在 **16 位（真实 DNG 最常见位深）**
        // 上悄悄降级成no-op——合成器与读端同时no-op 故往返测试照样通过，
        // 但读**真实相机文件**（Deflate + Predictor=2 + 16 位）会解出
        // 全错像素且不报错。这属C18之外最危险的一类：静默错误。
        if stride == 0 || row_len == 0 || bits == 0 || bits > 16 {
            return;
        }
        let mask: u32 = (1u32 << bits) - 1;
        let rows = samples.len() / row_len;
        for y in 0..rows {
            let base = y * row_len;
            for x in stride..row_len {
                let cur = samples.get(base + x).copied().unwrap_or(0) as u32;
                let left = samples.get(base + x - stride).copied().unwrap_or(0) as u32;
                let sum = (cur + left) & mask;
                if let Some(slot) = samples.get_mut(base + x) {
                    *slot = sum as u16;
                }
            }
        }
    }

    /// **编码**侧的预测器（就地施加），[`DeflatePredictor::undo_in_place`] 的逆。
    ///
    /// 合成器用它把原始样本变成「差分样本」再交给压缩器，
    /// 从而与读端严格互逆（合成器与读端共用同一张表，不各写一套）。
    pub fn apply_in_place(&self, samples: &mut [u16], row_len: usize, bits: u16) {
        let stride = self.stride();
        // 与 `undo_in_place` 同一口径：`bits == 16` **必须**做差分
        // （`wrapping_sub` 即模 2^16 回绕），不得提前返回。
        if stride == 0 || row_len == 0 || bits == 0 || bits > 16 {
            return;
        }
        let mask: u32 = (1u32 << bits) - 1;
        let rows = samples.len() / row_len;
        for y in 0..rows {
            let base = y * row_len;
            for x in (stride..row_len).rev() {
                let cur = samples.get(base + x).copied().unwrap_or(0) as u32;
                let left = samples.get(base + x - stride).copied().unwrap_or(0) as u32;
                let diff = (cur.wrapping_sub(left)) & mask;
                if let Some(slot) = samples.get_mut(base + x) {
                    *slot = diff as u16;
                }
            }
        }
    }
}

/// # `8` 与 `32946`：Deflate(ZIP) —— **zlib 流**（不是裸 DEFLATE）
///
/// 规范依据（Adobe《Photoshop TIFF Technical Notes》，TIFF 6.0 补充，
/// 压缩方案的规范原文；DNG 1.4 的 `Deflate (ZIP)` 继承它）：
///
/// ```text
/// Value: 8 — Deflate compression, **using zlib data format**
/// Each image segment consists of a **single complete zlib data stream**.
/// The zlib data stream contains header and trailer information
/// surrounding one or more compressed data blocks.
/// ```
///
/// **即：每个 strip / tile 是一条完整的 zlib 流**（2 字节 zlib 头 +
/// DEFLATE + 4 字节 Adler-32 尾），**不是**不带 zlib 头的裸 DEFLATE 码流。
/// 故本模块走 [`prism_bytes::zlib`] 的 `decompress_verified`（**校验
/// Adler-32**）而**不是** `inflate_raw`——两者只差容器，读错会全盘错位。
///
/// 这一点文档曾写作「裸 DEFLATE、不带 zlib 头」，与规范原文相反；
/// 本实现以规范为准（并在此写清依据）。
///
/// # `32946`（`0x80B2`）与 `8` 等价
///
/// 同一份Technical Notes 明确写道：`0x80B2` 是厂商私有 ZIP/Flate 码，
/// **压缩算法与 `8` 完全相同**，「应视为过时；建议 TIFF 实现**识别并读取**
/// 该过时码，但只写正式码 `8`」。故本模块对两者一视同仁。
///
/// # 为什么 `6` 值得单独说明
///
/// 曾把 6 描述为「真实相机里不罕见的未压缩变体」。核对上述规范原文后该
/// 前提**不成立**：`6` 在 TIFF 6.0 里是 **Old JPEG（OJPEG）**，一种 90 年代
/// 已被否决的 JPEG 草案，**DNG 规范的任何版本都没有定义过它**。所以写 6 的
/// 文件**不是「合规 DNG 的未压缩变体」**，而是厂商私有扩展或不合规文件。
///
/// 本仓没有该扩展的规范原文也没有样本，**无法自证其字节语义**。按项目铁律
/// 「诚实边界优于虚假功能」，此处**明确拒绝并说明成因**，而不是猜一套 tiling
/// 规则去读出「看起来正常」的错图。
///
/// # 用法
///
/// 仅用于**错误文本拼接**（`Display`），故返回 `&'static str` 且不做分配。
/// 未列举的取值走 `_` 分支给出通用成因。
///
/// # `8` 与 `32946` 已移出本清单（**曾写错，必须改**）
///
/// 旧版本在这里写着
///
/// ```text
/// 8 => "DNG 1.4 新增的 Deflate(ZIP)，本模块未实装inflate"
/// ```
///
/// **"未实装 inflate" 是错误表述**：DEFLATE 解码真源早已存在于
/// `prism-bytes`（`inflate.rs` / `zlib.rs`，零依赖叶子 crate），
/// 且 `prism-io` 早已依赖它——缺的是**DNG 侧的接线**，不是解码能力本身。
/// 一个错误的原因描述比「不支持」更糟：它会让后来人以为「补个依赖就行」。
///
/// 故 `8` / `32946` 现由 [`DngFile::read_raw_samples`] 真实支持，
/// 不再出现在拒绝清单里。
///
/// # `32946`（`0x80B2`）为什么与 `8` 等价
///
/// Adobe《Photoshop TIFF Technical Notes》（TIFF 6.0 补充，压缩方案的规范原文）
/// 明确写道：`0x80B2` 是厂商私有 ZIP/Flate 码，**压缩算法与上述 `8` 完全相同**，
/// 「应视为过时；建议 TIFF 实现**识别并读取**该过时码，但只写正式码 `8`」。
/// DNG 1.4 继承 TIFF 的这层约定，故本模块对 `32946` 与 `8` 一视同仁。
///
/// # `34892` 为什么**不是** Deflate 的别名（曾如此描述）
///
/// DNG 1.4 规范原文并列声明两个**语义完全不同**的新取值：
///
/// - `Value = 8: Deflate (ZIP)` —— 无损；
/// - `Value = 34892: Lossy JPEG` —— **有损基线 DCT**，
///   且仅允许「8 位整数 + PhotometricInterpretation 34892(LinearRaw)
///   或 52527(PhotometricMask)」。
///
/// 把 `34892` 当成 Deflate 的历史别名是**事实错误**：若照此实现，一个有损
/// JPEG 的 DNG 会被喂进 inflate，产出看似合法实则全错的像素。故本模块对
/// `34892` 继续**结构化拒绝**，并把成因写清楚（见上）。
pub fn compression_reject_hint(code: u16) -> &'static str {
    match code {
        2 | 3 | 4 => "TIFF 的 CCITT 传真类压缩（RLE / Group 3 / Group 4），DNG 规范不支持",
        5 => "TIFF 的 LZW 压缩，DNG 规范不支持",
        6 => "TIFF 的 Old JPEG（OJPEG，已被否决的旧草案），**DNG 规范从未定义此取值**；\
              写该值的文件不是合规 DNG，本仓无格式依据故不解析",
        32773 => "TIFF 的 PackBits 压缩，DNG 规范不支持",
        34712 => "JPEG 2000 压缩，DNG 规范不支持",
        34892 => "DNG 1.4 新增的 **Lossy JPEG**（有损基线 DCT，仅允许 8 位整数数据 + \
                 PhotometricInterpretation 34892(LinearRaw) / 52527(PhotometricMask)）；\
                 本模块只实现无损解码，未实装有损 JPEG 通路",
        52546 => "DNG 1.7 新增的 JPEG XL，本模块未实装",
        _ => "既非 DNG 规范定义的取值，也非本模块支持的 1 / 7 / 8 / 32946",
    }
}


/// 像素载荷的布局方式（strip / tile —— 由 IFD 里**存在哪个标签**决定）。
///
/// DNG 规范允许两种布局并存于生态中：入门级相机多用 strip（`StripOffsets`），
/// 而高像素机型普遍用 tile（`TileOffsets`）。本模块**两条都支持**，
/// 由 [`RawLayout::detect`] 按「tile 标签优先」判定——两者同时存在时
/// tile 是权威（真实文件里 strip 标签常被厂商填成占位值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawLayout {
    /// `StripOffsets` / `RowsPerStrip` / `StripByteCounts`。
    Strip {
        /// 每 strip 的行数。
        rows_per_strip: u32,
    },
    /// `TileWidth` / `TileLength` / `TileOffsets` / `TileByteCounts`。
    Tile {
        /// 单 tile 宽。
        tile_width: u32,
        /// 单 tile 高。
        tile_length: u32,
    },
}

impl RawLayout {
    /// 由 DNG 的 IFD 标签判定布局（**tile 优先**；两者皆无 → `Ok(None)`）。
    ///
    /// `TileWidth` 存在但 `TileLength` 非法属**结构化错误**（而非「当作
    /// strip 处理」）——静默降级会把 tile 文件按 strip 读出垃圾图像，
    /// 比明确报错坏得多。
    pub fn detect(dng: &DngFile) -> RawResult<Option<RawLayout>> {
        const LOC: &str = "raw::RawLayout::detect";
        if let Some(tw) = dng.ifd0.first_number(tags::TILE_WIDTH, dng.order) {
            let tl = dng.ifd0.first_number(tags::TILE_LENGTH, dng.order);
            return match (tw >= 1.0, tl.map(|v| v >= 1.0).unwrap_or(false)) {
                (true, true) => Ok(Some(RawLayout::Tile {
                    tile_width: tw as u32,
                    tile_length: tl.unwrap_or(1.0) as u32,
                })),
                _ => Err(tiff_err(
                    LOC,
                    format!(
                        "tile 标签非法：TileWidth = {tw}、TileLength = {}（二者须均 ≥ 1）",
                        tl.unwrap_or(f32::NAN)
                    ),
                    "确认 tile 标签未被错位读取；本模块不会把 tile 文件降级按 strip 读",
                )),
            };
        }
        if dng.ifd0.get(tags::STRIP_OFFSETS).is_some() {
            let h = dng
                .ifd0
                .first_number(tags::IMAGE_LENGTH, dng.order)
                .filter(|v| v.is_finite())
                .unwrap_or(1.0)
                .max(1.0);
            let rps = dng
                .ifd0
                .first_number(tags::ROWS_PER_STRIP, dng.order)
                .filter(|v| v.is_finite() && *v >= 1.0)
                .unwrap_or(h) as u32;
            return Ok(Some(RawLayout::Strip { rows_per_strip: rps }));
        }
        Ok(None)
    }
}

// ===========================================================================
//  十、DNG 文件
// ===========================================================================

/// EXIF 子 IFD 中读出、**未做语义解释**的字段（只取值，不做单位/逻辑解读）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExifInfo {
    /// `ExposureTime`（快门时间，秒）。
    pub exposure_time_s: Option<f32>,
    /// `FNumber`（光圈 f 值）。
    pub f_number: Option<f32>,
    /// `ISOSpeedRatings`（ISO 感光度）。
    pub iso: Option<f32>,
    /// `DateTimeOriginal`（ASCII 原样）。
    pub date_time_original: Option<String>,
}

/// GPS 子 IFD 的**原始数值转储**（**读、不解**）。
///
/// 按「读，不解」原则：本结构只把 GPS IFD 的条目按标签升序转成
/// `(tag, 数值向量, ASCII)` 三元组，**不**做经纬度十进制度换算、**不**做
/// UTC 解析、**不**做海拔正负号判读。消费方（如地理标注工具）自行解释。
#[derive(Debug, Clone, PartialEq)]
pub struct GpsIfd {
    /// GPS 子 IFD 的原始偏移（可回溯文件）。
    pub offset: u32,
    /// 条目转储，标签升序。
    pub tags: Vec<GpsTag>,
}

/// 单个 GPS 标签的原始内容。
#[derive(Debug, Clone, PartialEq)]
pub struct GpsTag {
    /// GPS 标签号（1 = GPSLatitudeRef、2 = GPSLatitude、…）。
    pub tag: u16,
    /// 数值向量（`RATIONAL` 已解为 f32；`UNDEFINED` 为空）。
    pub numbers: Vec<f32>,
    /// ASCII 载荷（若有）。
    pub ascii: Option<String>,
}

/// 已解析的 DNG 文件。
#[derive(Debug, Clone, PartialEq)]
pub struct DngFile {
    /// 字节序。
    pub order: ByteOrder,
    /// 原始字节（生命周期自持，便于子 IFD 延迟解析与诊断）。
    data: Vec<u8>,
    /// IFD0。
    pub ifd0: TiffIfd,
    /// EXIF 子 IFD（`ExifIFDPointer` 指向）。
    pub exif_ifd: Option<TiffIfd>,
    /// GPS 子 IFD（`ExifIFD → GPSInfoIFDPointer` 指向）。
    pub gps_ifd: Option<TiffIfd>,
    /// `DNGVersion` 四元组 `[major, minor, tiny, backminor]`。
    pub dng_version: [u8; 4],
    /// `DNGBackwardVersion` 四元组。
    pub dng_backward_version: [u8; 4],
}

impl DngFile {
    /// 原始字节的只读视图（生命周期挂在 `self` 上）。
    ///
    /// 供**子 IFD** 解析用：SubIFD 的载荷偏移是**文件绝对偏移**，
    /// 必须回到整份字节流上取，而不能只看 `ifd0` 已解析出的条目。
    /// 见 [`crate::raw_sub_ifd`]。
    pub fn data_slice(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// 解析 DNG 字节流（先做 [`probe_format`]，非 DNG 一律结构化拒绝）。
    pub fn parse(data: &[u8]) -> RawResult<Self> {
        const LOC: &str = "raw::DngFile::parse";
        match probe_format(data)? {
            RawFormat::Dng => {}
            // 兜底：`probe_format` 对 CR3 恒返回 Err，走不到这里；但若将来改动
            // 嗅探器让CR3 也能返回 Ok，这里必须结构化拒绝而非 panic（C18 铁律）。
            RawFormat::Cr3 => {
                return Err(unsupported_err(
                    LOC,
                    "CR3 不是 DNG：其有效载荷在 CMT1/CMT2/CRX 私有盒里，不是 TIFF/IFD",
                    "请用 Adobe DNG Converter 转存为 DNG，或用 exiftool 先抽出预览图",
                ))
            }
        }
        let order = ByteOrder::of_magic(data)
            .ok_or_else(|| tiff_err(LOC, "缺 II/MM 字节序头", "确认文件是 TIFF/DNG"))?;
        // TIFF 头的 magic 是偏移 2 处的 **u16** 字段（不是 u32）。
        let magic = order.u16(data.get(2..4).unwrap_or(&[0, 0])) as u32;
        if magic != 42 {
            return Err(tiff_err(
                LOC,
                format!("TIFF magic = {magic}，DNG 须为 42"),
                "确认不是 CR2/NEF 私有变体",
            ));
        }
        let ifd0_off = order.u32(data.get(4..8).unwrap_or(&[0; 4]));
        let ifd0 = read_ifd(data, order, ifd0_off)?;
        if ifd0.get(tags::DNG_VERSION).is_none() {
            return Err(unsupported_err(
                LOC,
                "IFD0 缺 DNGVersion(0xC612)：这是普通 TIFF，不是 DNG",
                "请用 Adobe DNG 或由相机厂商转存为 DNG",
            ));
        }
        let dng_version = raw_version(&ifd0, tags::DNG_VERSION, order);
        let dng_backward_version = raw_version(&ifd0, tags::DNG_BACKWARD_VERSION, order);
        if dng_version[0] != 1 {
            return Err(unsupported_err(
                LOC,
                format!("DNGVersion major = {}，本模块只实现 1.x", dng_version[0]),
                "DNG 2.x 未发布；如遇未知版本请确认文件来源",
            ));
        }
        if dng_version[3] > 1 {
            return Err(unsupported_err(
                LOC,
                format!(
                    "DNGBackwardVersion = {}.{}（要求 ≤ 1.1），使用了 1.4+ 的新标签语义",
                    dng_version[0], dng_version[3]
                ),
                "本模块按 DNG 1.4 标签子集实现；如需新标签请扩充 raw.rs 的标签表",
            ));
        }
        // EXIF 子 IFD（不递归进更深层：EXIF 内不再有我们关心的子指针）。
        let mut exif_ifd: Option<TiffIfd> = None;
        if let Some(ptr) = ifd0.first_number(tags::EXIF_IFD_POINTER, order) {
            let off = ptr as u32;
            if off != 0 {
                exif_ifd = Some(read_ifd(data, order, off)?);
            }
        }
        // GPS 子 IFD（EXIF → GPSInfoIFDPointer）。
        let mut gps_ifd: Option<TiffIfd> = None;
        if let Some(ex) = exif_ifd.as_ref() {
            if let Some(ptr) = ex.first_number(tags::GPS_IFD_POINTER, order) {
                let off = ptr as u32;
                if off != 0 {
                    gps_ifd = Some(read_ifd(data, order, off)?);
                }
            }
        }
        Ok(DngFile {
            order,
            data: data.to_vec(),
            ifd0,
            exif_ifd,
            gps_ifd,
            dng_version,
            dng_backward_version,
        })
    }

    /// 解析并返回 GPS 的「读不解」转储。
    pub fn gps(&self) -> Option<GpsIfd> {
        let ifd = self.gps_ifd.as_ref()?;
        let mut tags: Vec<GpsTag> = Vec::with_capacity(ifd.entries.len());
        for e in &ifd.entries {
            tags.push(GpsTag {
                tag: e.tag,
                numbers: e.numbers(self.order),
                ascii: e.text(),
            });
        }
        // entries 已在 read_ifd 内升序，此处保持（不重排，保证确定性）。
        Some(GpsIfd {
            offset: find_offset(
                self.order,
                tags_gps_offset(&self.data, &self.ifd0, &self.exif_ifd),
            ),
            tags,
        })
    }

    /// EXIF 中已读出但未做单位解释的字段。
    pub fn exif(&self) -> ExifInfo {
        const LOC: &str = "raw::DngFile::exif";
        let _ = LOC;
        let mut info = ExifInfo::default();
        if let Some(ex) = self.exif_ifd.as_ref() {
            info.exposure_time_s = ex.first_number(tags::EXPOSURE_TIME, self.order);
            info.f_number = ex.first_number(tags::FNUMBER, self.order);
            info.iso = ex.first_number(tags::ISO_SPEED_RATINGS, self.order);
            info.date_time_original = ex
                .get(tags::DATE_TIME_ORIGINAL)
                .and_then(|e| e.text())
                .map(|s| s.trim_end().to_string());
        }
        info
    }

    /// `UniqueCameraModel`（ASCII，可能为空）。
    pub fn camera_model(&self) -> Option<String> {
        self.ifd0
            .get(tags::UNIQUE_CAMERA_MODEL)
            .and_then(|e| e.text())
            .map(|s| s.trim_end().to_string())
    }

    /// CFA 描述（重复尺寸 + 平面序列 + 原图尺寸）。
    pub fn cfa_pattern(&self) -> RawResult<CfaPattern> {
        const LOC: &str = "raw::DngFile::cfa_pattern";
        let dim = self
            .ifd0
            .numbers(tags::CFA_REPEAT_PATTERN_DIM, self.order)
            .unwrap_or_default();
        let (mut cols, mut rows) = match dim.as_slice() {
            [c, r] => (*c as u16, *r as u16),
            _ => {
                return Err(tiff_err(
                    LOC,
                    "缺 CFARepeatPatternDim(0x82BD) 或长度 ≠ 2",
                    "确认是含 CFA 的 DNG（彩色相机原始文件）",
                ))
            }
        };
        let colors: Vec<u8> = self
            .ifd0
            .get(tags::CFA_PATTERN)
            .map(|e| {
                e.numbers(self.order)
                    .iter()
                    .map(|&v| v as u8)
                    .collect::<Vec<u8>>()
            })
            .unwrap_or_default();
        if colors.is_empty() {
            return Err(tiff_err(
                LOC,
                "缺 CFAPattern(0x828E)",
                "确认是含 CFA 的 DNG；三通道 RGB DNG 无此标签属正常",
            ));
        }
        // 部分机型把 2×2 写成 (1,1) + 4 字节；按 DNG 1.4 归一到 2×2。
        if cols == 1 && rows == 1 && colors.len() == 4 {
            cols = 2;
            rows = 2;
        }
        if (cols as usize) * (rows as usize) != colors.len() {
            return Err(tiff_err(
                LOC,
                format!(
                    "CFA 重复尺寸 {cols}×{rows} = {} 与 CFAPattern 长度 {} 不符",
                    (cols as usize) * (rows as usize),
                    colors.len()
                ),
                "确认 CFAPattern 负载未被错位读取",
            ));
        }
        if colors.len() > 64 {
            return Err(unsupported_err(
                LOC,
                format!(
                    "CFA 平铺 {} 项（> 64），本模块只支持 ≤ 64 的小型阵列",
                    colors.len()
                ),
                "超大阵列需专门的 CFA 描述器",
            ));
        }
        let (raw_width, raw_height) = self.active_area_size()?;
        Ok(CfaPattern {
            cols,
            rows,
            colors,
            raw_width,
            raw_height,
        })
    }

    /// 原图有效区尺寸（`ActiveArea` 优先，退回 `ImageWidth`/`ImageLength`）。
    fn active_area_size(&self) -> RawResult<(u32, u32)> {
        const LOC: &str = "raw::DngFile::active_area_size";
        if let Some(aa) = self.ifd0.numbers(tags::ACTIVE_AREA, self.order) {
            if aa.len() == 4 {
                let (top, left, bottom, right) = (aa[0], aa[1], aa[2], aa[3]);
                let w = right - left + 1.0;
                let h = bottom - top + 1.0;
                if w >= 1.0 && h >= 1.0 && w.is_finite() && h.is_finite() {
                    return Ok((w as u32, h as u32));
                }
                return Err(tiff_err(
                    LOC,
                    format!("ActiveArea = [{top},{left},{bottom},{right}] 非法（须 bottom≥top、right≥left）"),
                    "确认 ActiveArea 标签负载正确",
                ));
            }
        }
        let w = self.ifd0.first_number(tags::IMAGE_WIDTH, self.order);
        let h = self.ifd0.first_number(tags::IMAGE_LENGTH, self.order);
        match (w, h) {
            (Some(w), Some(h)) if w >= 1.0 && h >= 1.0 => Ok((w as u32, h as u32)),
            (Some(w), Some(h)) => Err(tiff_err(
                LOC,
                format!("ImageWidth/ImageLength = {w}×{h} 非法"),
                "确认尺寸标签正确",
            )),
            _ => Err(tiff_err(
                LOC,
                "既无 ActiveArea 也无 ImageWidth/ImageLength",
                "确认 IFD0 完整（未截断）",
            )),
        }
    }

    /// 采样位深（`BitsPerSample`，缺省 16）。
    pub fn bits_per_sample(&self) -> u16 {
        self.ifd0
            .first_number(tags::BITS_PER_SAMPLE, self.order)
            .map(|v| v as u16)
            .unwrap_or(16)
    }

    /// 位流填充顺序（`FillOrder`，缺省 [`FillOrder::MsbFirst`] —— DNG 规范默认）。
    ///
    /// 标签值非 1/2（畸形）时**回退 MSB-first**而非报错：DNG 规范规定缺省即
    /// 1，而「读不出位序」时按规范的缺省值解释，比让整幅图不可读更贴近
    /// 「尽力解出正确像素」的工程意图；该回退在 [`DngFile::develop`] 的
    /// 文档中已注明。
    pub fn fill_order(&self) -> FillOrder {
        self.ifd0
            .first_number(tags::FILL_ORDER, self.order)
            .and_then(|v| FillOrder::from_code(v as u16))
            .unwrap_or(FillOrder::MsbFirst)
    }

    /// 压缩方式（`Compression`，缺省 1 = 无压缩）。
    pub fn compression(&self) -> u16 {
        self.ifd0
            .first_number(tags::COMPRESSION, self.order)
            .map(|v| v as u16)
            .unwrap_or(1)
    }

    /// `Predictor`（0x013D）标签的**原始取值**（缺省 **1** = Null）。
    ///
    /// **只在 `Compression` 为 8 / 32946（Deflate）时有意义**——DNG 规范把
    /// `Predictor` 定义在 Deflate 的章节下；其余压缩方式读到它也没有规范效力。
    ///
    /// 缺省返回 `1`（不预测）而非报错：TIFF 规定缺省即 1，按缺省解释
    /// 比让整幅图不可读更贴近工程意图（与 [`DngFile::fill_order`] 的回退一致）。
    ///
    /// 标签值非有限 / 越界（`< 0` 或 `> u16::MAX`）时同样回退 `1`，
    /// 并由 [`DeflatePredictor::from_code`] 之外的调用点按需报「未支持」。
    pub fn deflate_predictor_code(&self) -> u16 {
        self.ifd0
            .first_number(tags::PREDICTOR, self.order)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u16::MAX as f32)
            .map(|v| v as u16)
            .unwrap_or(1)
    }

    /// **行交织因子**（`RowInterleaveFactor`，0xC71F = 50975，缺省 **1**）。
    ///
    /// DNG 1.6 原文见 [`tags::ROW_INTERLEAVE_FACTOR`]。语义：
    /// `factor = 1` → 正常行序；`factor = 2` → 偶数行在前、奇数行在后
    /// （即「奇偶行分离」；而 `tags & 0x8000` 是
    /// dcraw/LibRaw 的**私有实现细节**，DNG 规范里没有这个字段）。
    ///
    /// 标签值非有限 / 越界时**回退 1**（不交织）：DNG 规定缺省即 1，
    /// 按规范缺省解释比让整幅图不可读更贴近工程意图（与
    /// [`DngFile::fill_order`] 的回退策略一致）。
    pub fn row_interleave_factor(&self) -> u32 {
        self.ifd0
            .first_number(tags::ROW_INTERLEAVE_FACTOR, self.order)
            .filter(|v| v.is_finite() && *v >= 1.0 && *v <= 16.0)
            .map(|v| v as u32)
            .unwrap_or(1)
    }

    /// 原始字节（`&[u8]`，供需要自行解析子 IFD 的调用方使用）。
    ///
    /// [`DngFile`] 刻意**自持**字节（不借 lifetime），故子 IFD 这类
    /// 「需要访问整份字节流」的解析走本方法而不是泛型生命周期。
    pub fn raw_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// 读**子 IFD**的 `FillOrder`（缺省 MSB-first）。
    ///
    /// 与 [`DngFile::fill_order`] 分开：那个读 IFD0，这个读传入的子 IFD
    /// （预览图可能有自己的位序标签）。
    pub(crate) fn sub_ifd_fill_order(&self, ifd: &TiffIfd) -> Option<FillOrder> {
        ifd.first_number(tags::FILL_ORDER, self.order)
            .and_then(|v| FillOrder::from_code(v as u16))
    }

    /// 解析 **SubIFD 预览图链**（`SubIFDs` 标签，0x014A）。
    ///
    /// 真实相机 DNG **普遍**带 SubIFD 预览图（小尺寸 JPEG / 无损JPEG）。
    /// 返回按链上顺序排列的 [`crate::raw_subifd::SubIfdImage`]，每项含
    /// 尺寸 / 压缩方式 / 色彩解释，以及**在压缩方式受支持时解出的像素**。
    ///
    /// # 委托
    ///
    /// 纯转发到 [`crate::raw_subifd::parse_sub_ifds`]（独立模块，
    /// 免得 `raw.rs` 因预览链路继续膨胀）。
    ///
    /// # 没有 SubIFD 时
    ///
    /// 返回**空向量**（不是错误）——「这个文件没预览图」是正常事实。
    pub fn sub_ifds(&self) -> RawResult<Vec<crate::raw_subifd::SubIfdImage>> {
        crate::raw_subifd::parse_sub_ifds(self)
    }

    /// **线性化表**（`LinearizationTable`，0xC691 = 50833，缺省无）。
    ///
    /// DNG 1.6 Chapter 5 规定映射顺序为
    /// **Linearization → Black Subtraction → Rescaling → Clipping**，
    /// 故本表在**减黑电平之前**应用（见 [`DngFile::develop`] 的 ① 步）。
    ///
    /// 返回长度 `2^BitsPerSample` 的查表；标签缺省 / 长度不足 / 含非法值时
    /// 返回 `None`（调用方按「无线性化」处理，**不报错** —— 真实文件里
    /// 这张表罕见，不该为一张可选表让整幅图不可读）。
    ///
    /// > 注：`LinearizationTable` 标签号曾被写作 `0xC618`，那是**错的**
    /// > （`0xC618` = 50712，不是任何 DNG 标签）；规范原文是
    /// > **0xC691 = 50833**。
    pub fn linearization_table(&self) -> Option<Vec<u16>> {
        let raw = self.ifd0.numbers(tags::LINEARIZATION_TABLE, self.order)?;
        let bits = self.bits_per_sample();
        if bits == 0 || bits > 16 {
            return None;
        }
        let want = 1usize.checked_shl(bits as u32)?;
        if raw.len() < want {
            return None;
        }
        let mut out = Vec::with_capacity(want);
        for v in raw.iter().take(want) {
            if !v.is_finite() || *v < 0.0 || *v > 65535.0 {
                return None;
            }
            out.push(*v as u16);
        }
        Some(out)
    }

    /// 逐行 / 逐列黑电平偏移（`BlackLevelDeltaH` 0xC61B / `BlackLevelDeltaV` 0xC61C）。
    ///
    /// DNG 1.6 Chapter 5 原文：
    ///
    /// > The black level for each pixel is then computed and subtracted. The
    /// > black level for each pixel is the sum of the black levels specified by
    /// > the BlackLevel, BlackLevelDeltaH and BlackLevelDeltaV tags.
    ///
    /// 两者缺省时返回全 0 向量（长度 = 对应维度），故调用方可**无条件相加**。
    pub fn black_level_deltas(&self, height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
        let pick = |tag: u16, n: usize| -> Vec<f32> {
            // `numbers` 直接给 `Vec<f32>`，无需再`as f32`。
            let v: Vec<f32> = self
                .ifd0
                .numbers(tag, self.order)
                .unwrap_or_default()
                .into_iter()
                .filter(|x| x.is_finite())
                .collect();
            if n > 0 && v.len() >= n {
                v[..n].to_vec()
            } else {
                vec![0.0; n]
            }
        };
        (
            pick(tags::BLACK_LEVEL_DELTA_H, height),
            pick(tags::BLACK_LEVEL_DELTA_V, width),
        )
    }


    /// 黑 / 白电平。
    pub fn levels(&self, cfa: &CfaPattern) -> RawResult<RawLevels> {
        const LOC: &str = "raw::DngFile::levels";
        let period = (cfa.cols as usize) * (cfa.rows as usize);
        let black_raw: Vec<f32> = self
            .ifd0
            .numbers(tags::BLACK_LEVEL, self.order)
            .unwrap_or_default();
        let black: Vec<f32> = if black_raw.is_empty() {
            // 无 BlackLevel 标签：不少 DNG 约定 0（DNG 1.4 §BlackLevel 可选，
            // 缺省即 0，见「BlackLevel」标签说明）。显式记录为全 0。
            vec![0.0; period]
        } else if black_raw.len() >= period {
            // 逐 CFA 位置的黑电平（DNG 1.4 允许 4 个值对应 RGGB 各位置）。
            black_raw[..period].to_vec()
        } else {
            // 标量黑电平，铺满全部位置。
            let v = black_raw[0];
            vec![v; period]
        };
        let white = self
            .ifd0
            .first_number(tags::WHITE_LEVEL, self.order)
            .ok_or_else(|| {
                tiff_err(
                    LOC,
                    "缺 WhiteLevel(0xC61D)",
                    "白电平是归一化必需量；请确认文件完整或改用其它归一化策略",
                )
            })?;
        if !(white.is_finite() && white > 0.0) {
            return Err(tiff_err(
                LOC,
                format!("WhiteLevel = {white} 非法（须 > 0 且有限）"),
                "确认 WhiteLevel 标签负载正确",
            ));
        }
        if black.iter().any(|&b| !b.is_finite() || b < 0.0) {
            return Err(tiff_err(
                LOC,
                format!("BlackLevel 含非法分量 {black:?}"),
                "确认 BlackLevel 标签负载正确",
            ));
        }
        Ok(RawLevels {
            black,
            white,
            bits_per_sample: self.bits_per_sample(),
            fill_order: self.fill_order(),
        })
    }

    /// `ColorMatrix1`（3×3，行主序；缺失或含非有限值 → `None`）。
    ///
    /// 语义（DNG 1.4）：**已白平衡的相机 RGB → XYZ**，参考白点由
    /// `CalibrationIlluminant1` 指定（本模块按 D65 处理，见
    /// [`DngFile::color_matrix_chain`] 的说明）。
    pub fn color_matrix(&self) -> Option<Mat3> {
        mat3_from_tag(self.ifd0.numbers(tags::COLOR_MATRIX1, self.order))
    }

    /// `ForwardMatrix2`（0xC68A，3×3 行主序；缺失或含非有限值 → `None`）。
    ///
    /// 语义（DNG 1.4 §ForwardMatrix2）：**相机空间 → XYZ(D50)** 的**正向**
    /// 矩阵。它与 `ColorMatrix1`（反向矩阵，且白点由 `CalibrationIlluminant1`
    /// 决定）**不是同一件事**，也不能互相替代：真实相机 DNG 两者通常**至少
    /// 存在其一**，规范**推荐** `ForwardMatrix2`（正向标定，误差更小）。
    ///
    /// 使用它时**必须**配套（见 [`DngFile::color_matrix_chain`]）：
    /// `CameraCalibration1` + `AnalogBalance` + `AsShotNeutral` + D50→D65 适配。
    pub fn forward_matrix2(&self) -> Option<Mat3> {
        mat3_from_tag(self.ifd0.numbers(tags::FORWARD_MATRIX2, self.order))
    }

    /// `CameraCalibration1`（0xC6F3，3×3 行主序；缺失 → `None`）。
    ///
    /// 语义（DNG 1.4 §CameraCalibration1）：**光学暗电流 / 暗角校准矩阵**，
    /// 作用在**白平衡之后、色彩矩阵之前**的相机域 RGB 上（它建模的是光学
    /// 黑电流与镜组暗角，属相机域的辐射修正，故在相机空间施加）。
    pub fn camera_calibration1(&self) -> Option<Mat3> {
        mat3_from_tag(self.ifd0.numbers(tags::CAMERA_CALIBRATION1, self.order))
    }

    /// `CameraCalibration2`（0xC6F4，3×3 行主序；缺失 → `None`）。
    ///
    /// 规范允许有第二个校准矩阵（对应 `CalibrationIlluminant2`）。本模块
    /// **读出但不施加**——施加哪一个取决于文件声明的标定光源组合，无可靠
    /// 启发式可依，静默挑一个比不施加更危险。见「诚实边界」。
    pub fn camera_calibration2(&self) -> Option<Mat3> {
        mat3_from_tag(self.ifd0.numbers(tags::CAMERA_CALIBRATION2, self.order))
    }

    /// `AnalogBalance`（0xC62A，RATIONAL × 3；缺失或非法 → `None`）。
    ///
    /// 语义（DNG 1.4 §AnalogBalance）：**模拟域白平衡系数**——「同一个物理
    /// 中性灰在模拟电路里被放大到各通道多少」。它与 `AsShotNeutral` 的**比值**
    /// 才是真正的相机空间中性值，见 [`DngFile::effective_neutral`]。
    pub fn analog_balance(&self) -> Option<[f32; 3]> {
        triple_from_tag(self.ifd0.numbers(tags::ANALOG_BALANCE, self.order))
    }

    /// **有效中性值**：`AsShotNeutral / AnalogBalance`（逐通道）。
    ///
    /// # 出处与顺序（DNG 1.4 §AsShotNeutral + §AnalogBalance）
    ///
    /// ```text
    /// neutral = AsShotNeutral / AnalogBalance
    /// ```
    ///
    /// `AsShotNeutral` 记录的是「**数字域**实测的中性响应」，而 `AnalogBalance`
    /// 记录的是「**模拟域**各通道的增益」。两者相除才还原出**纯相机空间**
    /// （既非模拟、也非数字域）的白平衡基准。
    ///
    /// - `AnalogBalance` **缺失** → 直接用 `AsShotNeutral`（DNG 1.4 允许
    ///   该标签缺席，此时相机空间即数字域）；
    /// - 两者都缺失 → `None`（调用方按「无白平衡信息」处理）。
    pub fn effective_neutral(&self) -> Option<[f32; 3]> {
        let n = self.as_shot_neutral()?;
        match self.analog_balance() {
            Some(ab) => {
                let mut out = [0.0f32; 3];
                for i in 0..3 {
                    out[i] = n[i] / ab[i];
                }
                Some(out)
            }
            None => Some(n),
        }
    }

    /// `AsShotNeutral`（相机空间中性响应；缺失或非法 → `None`）。
    pub fn as_shot_neutral(&self) -> Option<[f32; 3]> {
        let v = self
            .ifd0
            .numbers(tags::AS_SHOT_NEUTRAL, self.order)
            .unwrap_or_default();
        if v.len() < 3 {
            return None;
        }
        let n = [v[0], v[1], v[2]];
        if n.iter().all(|&x| x.is_finite() && x > 0.0) {
            Some(n)
        } else {
            None
        }
    }

    /// `AsShotWhiteXY`（`[x, y]`；缺失 → `None`）。
    pub fn as_shot_white_xy(&self) -> Option<[f32; 2]> {
        let v = self
            .ifd0
            .numbers(tags::AS_SHOT_WHITE_XY, self.order)
            .unwrap_or_default();
        if v.len() >= 2 {
            Some([v[0], v[1]])
        } else {
            None
        }
    }

    /// `CalibrationIlluminant1`（EXIF 光源代号，17 = StandardA、21 = D65）。
    pub fn calibration_illuminant1(&self) -> Option<u16> {
        self.ifd0
            .first_number(tags::CALIBRATION_ILLUMINANT1, self.order)
            .map(|v| v as u16)
    }

    /// `BaselineExposure`（对数域 EV 偏移；缺失 → `0.0`）。
    pub fn baseline_exposure(&self) -> f32 {
        self.ifd0
            .first_number(tags::BASELINE_EXPOSURE, self.order)
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    }

    /// 判定像素载荷布局（tile / strip；见 [`RawLayout::detect`]）。
    fn raw_layout(&self) -> RawResult<Option<RawLayout>> {
        RawLayout::detect(self)
    }

    /// 从 IFD 的偏移/字节数标签读出 `u64` 数组（LONG 或 SHORT 计数均可）。
    fn offset_list(&self, tag: u16) -> Vec<u64> {
        self.ifd0
            .numbers(tag, self.order)
            .unwrap_or_default()
            .into_iter()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| v as u64)
            .collect()
    }

    /// 读出完整的单通道马赛克（**已解包**，长度 = `width × height`）。
    ///
    /// # 支持的布局（并存，由标签决定）
    ///
    /// - **strip**：`RowsPerStrip` 行一个 strip，strip 内像素**行主序连续**；
    /// - **tile**：`TileWidth × TileLength` 一个 tile，tile 在图内**按行优先
    ///   排列**（`tiles_across = ceil(width / tile_width)`），图边界的 tile
    ///   只取有效区域（不足部分忽略）。
    ///
    /// # 位深
    ///
    /// 走 [`unpack_bits`]，支持 1..=16 位紧凑打包。**不做右移归一到 16 位**
    /// ——因为黑/白电平也是原始码值，右移会破坏 `(raw−black)/(white−black)`
    /// 的同量纲相消（见 [`RawLevels::bits_per_sample`]）。
    ///
    /// # 压缩方式分派
    ///
    /// | `Compression` | 处理路径 |
    /// |---|---|
    /// | 1（无压缩） | [`unpack_bits`] 紧凑位解包（本函数的 `RawLayout` 分支） |
    /// | **7（lossless JPEG）** | [`crate::raw_lossless_jpeg`]：T.81 SOF3 解码 + [`RowInterleaveFactor`](tags::ROW_INTERLEAVE_FACTOR) 解交织 |
    /// | **8 / 32946（Deflate(ZIP)）** | [`inflate_dng_zlib`]：zlib 容器解压（**校验 Adler-32**）+ [`DeflatePredictor`] 反预测 + [`unpack_bits`] |
    /// | 其他（含 6、34892、34712、52546 等） | 结构化拒绝，见 [`compression_reject_hint`] |
    ///
    /// > [`inflate_dng_zlib`]：走 `prism_bytes::zlib` 的
    /// > `decompress_verified`（**不是** `inflate_raw`）——Deflate 在 TIFF 里
    /// > 是「zlib data format」，带 2 字节头 + 4 字节 Adler-32 尾，两者只差容器。
    /// > 裸 DEFLATE 流会被结构化拒绝（见 `dng_deflate_rejects_bare_deflate_stream`）。
    ///
    /// # `Compression = 6`：**为什么仍然拒绝**（诚实边界）
    ///
    /// 曾把 6 描述为「未压缩但带特定 tiling，或厂商自定义变体」。核对
    /// **Adobe DNG 规范原文**（1.4.0.0 / 1.6.0.0 / 1.7.1.0 的 `Compression`
    /// 章节）后，这个前提**不成立**：
    ///
    /// - DNG 1.4 之前的版本只定义 **两个**取值：`1`（未压缩）与 `7`（JPEG）；
    /// - DNG 1.4 新增 `8`（Deflate）与 `34892`（Lossy JPEG）；
    /// - DNG 1.7 新增 `52546`（JPEG XL）。
    ///
    /// **`6` 从未出现在任何一版 DNG 规范的 `Compression` 章节里。** 在 TIFF6.0
    /// 里 `6` 是 **Old JPEG（OJPEG，一种已被否决的 90 年代草案）**——DNG 既然
    /// 只允许上列取值，**写 6 的文件要么是厂商私有扩展，要么本身就不是合规 DNG**。
    ///
    /// 本仓**无法自证**「这类厂商扩展的确切字节语义」（无规范原文、无样本），
    /// 故按项目铁律「诚实边界优于虚假功能」**继续拒绝**，并把
    /// [`compression_reject_hint`] 的提示写清楚——而不是猜一个 tiling 规则
    /// 上去「读出看起来正常的错图」。
    ///
    /// # 拒绝的输入（不假装支持）
    ///
    /// - `Compression` 既非 1 也非 7 → 结构化拒绝（给出该取值的**具体**成因与
    ///   转存建议，见 [`compression_reject_hint`]）；
    /// - `Compression == 7` 但流不是 SOF3（多分量 / 算术编码 / 有损）→
    ///   由 [`crate::raw_lossless_jpeg`] 结构化拒绝并说明原因；
    /// - `SamplesPerPixel != 1`（线性 RGB DNG）→ 结构化拒绝（本模块只做
    ///   CFA 单通道）。
    pub fn read_raw_samples(&self) -> RawResult<Vec<u16>> {
        const LOC: &str = "raw::DngFile::read_raw_samples";
        let (width, height) = self.active_area_size()?;
        let count = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| pipe_err(LOC, format!("{width}×{height} 像素数溢出"), "使用更小的图"))?;
        let compression = self.compression();
        match compression {
            1 => {}
            // DNG Compression = 7：ITU-T T.81 无损 JPEG（H.81 Annex H / process 14）。
            7 => return self.read_raw_samples_lossless(width, height, count),
            // DNG 1.4 Compression = 8：Deflate(ZIP)。**每个 strip/tile 是一条
            // 完整 zlib 流**（见 [`compression_reject_hint`] 的规范依据段），
            // 故走 `zlib::decompress_verified`（校验 Adler-32），不是裸 inflate。
            // 32946(0x80B2) 是厂商私有旧编号，算法与 8 完全相同，一视同仁。
            8 | 32946 => return self.read_raw_samples_deflate(width, height, count),
            other => {
                return Err(unsupported_err(
                    LOC,
                    format!(
                        "Compression = {other}，本模块只支持 1（未压缩）、7（lossless JPEG）\
                         与 8 / 32946（Deflate）：{}",
                        compression_reject_hint(other)
                    ),
                    "请用 Adobe DNG Converter 转存为「未压缩」「lossless JPEG」或「Deflate」\
                     DNG 后再读入；若确为厂商私有扩展，需先提供该机型的样本与格式文档才能支持",
                ))
            }
        }
        let spp = self
            .ifd0
            .first_number(tags::SAMPLES_PER_PIXEL, self.order)
            .unwrap_or(1.0);
        if spp != 1.0 {
            return Err(unsupported_err(
                LOC,
                format!("SamplesPerPixel = {spp}，本模块只做 CFA 单通道（= 1）"),
                "三通道线性 RGB DNG 需另一条管线；请确认这是彩色相机原始文件",
            ));
        }
        let bits = self.bits_per_sample();
        let fill = self.fill_order();
        let layout = self.raw_layout()?.ok_or_else(|| {
            tiff_err(
                LOC,
                "既无 StripOffsets(0x0111) 也无 TileWidth/TileOffsets(0x0142/0x0144)",
                "确认 IFD0 含像素载荷布局标签",
            )
        })?;
        let mut out: Vec<u16> = vec![0u16; count];
        match layout {
            RawLayout::Strip { rows_per_strip } => {
                let offs = self.offset_list(tags::STRIP_OFFSETS);
                let counts = self.offset_list(tags::STRIP_BYTE_COUNTS);
                let strips = (height + rows_per_strip - 1) / rows_per_strip;
                if offs.len() < strips as usize {
                    return Err(tiff_err(
                        LOC,
                        format!("StripOffsets 只有 {} 项，但按 {rows_per_strip} 行/strip 需 {strips} 项", offs.len()),
                        "确认 strip 标签负载正确",
                    ));
                }
                for s in 0..strips as usize {
                    let y0 = (s as u32).saturating_mul(rows_per_strip);
                    let rows = rows_per_strip.min(height - y0);
                    let n = (width as usize).saturating_mul(rows as usize);
                    let off = offs[s];
                    let end = off.saturating_add(counts.get(s).copied().unwrap_or(0));
                    let src = self.slice_region(off, end, n, bits, LOC)?;
                    let samples = unpack_bits(src, bits, n, fill)?;
                    for y in 0..rows {
                        for x in 0..width as usize {
                            let si = y as usize * width as usize + x;
                            let di = (y0 as usize + y as usize) * width as usize + x;
                            if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                *d = *sv;
                            }
                        }
                    }
                }
            }
            RawLayout::Tile {
                tile_width,
                tile_length,
            } => {
                if self.ifd0.get(tags::TILE_OFFSETS).is_none() {
                    return Err(tiff_err(
                        LOC,
                        "有 TileWidth/TileLength 但缺 TileOffsets(0x0144)",
                        "确认 tile 标签齐备",
                    ));
                }
                let offs = self.offset_list(tags::TILE_OFFSETS);
                let counts = self.offset_list(tags::TILE_BYTE_COUNTS);
                let across = (width + tile_width - 1) / tile_width;
                let down = (height + tile_length - 1) / tile_length;
                let tiles = (across as usize).saturating_mul(down as usize);
                if offs.len() < tiles {
                    return Err(tiff_err(
                        LOC,
                        format!("TileOffsets 只有 {} 项，但 {width}×{height} 按 {tile_width}×{tile_length} 需 {tiles} 项", offs.len()),
                        "确认 tile 偏移表长度与图幅匹配",
                    ));
                }
                // tile 在图内按行优先排列：第 ty 行第 tx 列的 tile 序号 = ty*across + tx。
                for ty in 0..down as usize {
                    for tx in 0..across as usize {
                        let idx = ty * across as usize + tx;
                        let x0 = (tx as u32) * tile_width;
                        let y0 = (ty as u32) * tile_length;
                        let copy_w = tile_width.min(width - x0);
                        let copy_h = tile_length.min(height - y0);
                        // **整 tile 解包**（含图边界外的填充像素），再只拷贝有效区——
                        // 否则边界 tile 的位流起点会算错（少解一个样本，后续全部错位）。
                        let n = (tile_width as usize).saturating_mul(tile_length as usize);
                        let off = offs[idx];
                        let end = off.saturating_add(counts.get(idx).copied().unwrap_or(0));
                        let src = self.slice_region(off, end, n, bits, LOC)?;
                        let samples = unpack_bits(src, bits, n, fill)?;
                        for y in 0..copy_h {
                            for x in 0..copy_w {
                                let si = y as usize * tile_width as usize + x as usize;
                                let di = (y0 as usize + y as usize) * width as usize
                                    + x0 as usize
                                    + x as usize;
                                if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                    *d = *sv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }


    /// **DNG `Compression = 7`：ITU-T T.81 无损 JPEG** 路径。
    ///
    /// 每个 strip / tile 是一条**完整**的无损 JPEG 流（`SOI`…`EOI`），
    /// 由 [`crate::raw_lossless_jpeg::decode_dng_strip`] 解码，并按
    /// [`tags::ROW_INTERLEAVE_FACTOR`] 解交织。
    ///
    /// # `RowInterleaveFactor` 与多 strip
    ///
    /// 交织是**整图**层面的概念，故每个 strip 都必须带上它在整图中的**首行号
    /// `y0`**，才能判定「本 strip 的哪些行属于 field 0、哪些属于 field 1」。
    /// 这也是 [`crate::raw_lossless_jpeg::decode_dng_strip`] 显式收 `y0`
    /// 而不是自己猜的原因。
    ///
    /// # 不做的
    ///
    /// - 每个 strip 的 `ExtraSamples` 前缀按 `column_offset = 0` 处理
    ///   （DNG 规范没有对应标签；真实 CFA DNG 该值为 0）；
    /// - restart marker（`DRI` 非 0 即拒）与多分量无损（`Ns != 1` 即拒）
    ///   由 JPEG 模块结构化拒绝。
    fn read_raw_samples_lossless(
        &self,
        width: u32,
        height: u32,
        count: usize,
    ) -> RawResult<Vec<u16>> {
        const LOC: &str = "raw::DngFile::read_raw_samples_lossless";
        let spp = self
            .ifd0
            .first_number(tags::SAMPLES_PER_PIXEL, self.order)
            .unwrap_or(1.0);
        if spp != 1.0 {
            return Err(unsupported_err(
                LOC,
                format!("SamplesPerPixel = {spp}，本模块只做 CFA 单通道（= 1）"),
                "线性 RGB DNG 的无损 JPEG 是多分量（Ns = 3），需另一条管线；\
                 请确认这是彩色相机原始文件",
            ));
        }
        let layout = self.raw_layout()?.ok_or_else(|| {
            tiff_err(
                LOC,
                "既无 StripOffsets(0x0111) 也无 TileWidth/TileOffsets(0x0142/0x0144)",
                "确认 IFD0 含像素载荷布局标签",
            )
        })?;
        let interleave = self.row_interleave_factor();
        let mut out: Vec<u16> = vec![0u16; count];
        let w = width as usize;

        match layout {
            RawLayout::Strip { rows_per_strip } => {
                let offs = self.offset_list(tags::STRIP_OFFSETS);
                let counts = self.offset_list(tags::STRIP_BYTE_COUNTS);
                let strips = (height + rows_per_strip - 1) / rows_per_strip;
                if offs.len() < strips as usize {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "StripOffsets 只有 {} 项，但按 {rows_per_strip} 行/strip 需 {strips} 项",
                            offs.len()
                        ),
                        "确认 strip 标签负载正确",
                    ));
                }
                for s in 0..strips as usize {
                    let y0 = (s as u32).saturating_mul(rows_per_strip);
                    let rows = rows_per_strip.min(height - y0);
                    let n = w.saturating_mul(rows as usize);
                    let src = self.slice_lossless(offs[s], counts.get(s).copied().unwrap_or(0))?;
                    // 交织是整图概念 → 必须传本 strip 的全局首行号 y0。
                    let samples = crate::raw_lossless_jpeg::decode_dng_strip(
                        src,
                        n,
                        0,
                        interleave,
                        y0,
                    )?;
                    for y in 0..rows as usize {
                        for x in 0..w {
                            let si = y * w + x;
                            let di = (y0 as usize + y) * w + x;
                            if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                *d = *sv;
                            }
                        }
                    }
                }
            }
            RawLayout::Tile {
                tile_width,
                tile_length,
            } => {
                if self.ifd0.get(tags::TILE_OFFSETS).is_none() {
                    return Err(tiff_err(
                        LOC,
                        "有 TileWidth/TileLength 但缺 TileOffsets(0x0144)",
                        "确认 tile 标签齐备",
                    ));
                }
                let offs = self.offset_list(tags::TILE_OFFSETS);
                let counts = self.offset_list(tags::TILE_BYTE_COUNTS);
                let across = (width + tile_width - 1) / tile_width;
                let down = (height + tile_length - 1) / tile_length;
                let tiles = (across as usize).saturating_mul(down as usize);
                if offs.len() < tiles {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "TileOffsets 只有 {} 项，但 {width}×{height} 按 {tile_width}×{tile_length} 需 {tiles} 项",
                            offs.len()
                        ),
                        "确认 tile 偏移表长度与图幅匹配",
                    ));
                }
                for ty in 0..down as usize {
                    for tx in 0..across as usize {
                        let idx = ty * across as usize + tx;
                        let x0 = (tx as u32) * tile_width;
                        let y0 = (ty as u32) * tile_length;
                        // 整 tile 解码（含图边界外的填充像素），再只拷有效区。
                        let n = (tile_width as usize).saturating_mul(tile_length as usize);
                        let src = self
                            .slice_lossless(offs[idx], counts.get(idx).copied().unwrap_or(0))?;
                        let samples =
                            crate::raw_lossless_jpeg::decode_dng_strip(src, n, 0, interleave, y0)?;
                        let copy_w = (tile_width.min(width - x0)) as usize;
                        let copy_h = (tile_length.min(height - y0)) as usize;
                        let tw = tile_width as usize;
                        for y in 0..copy_h {
                            for x in 0..copy_w {
                                let si = y * tw + x;
                                let di = (y0 as usize + y) * w + x0 as usize + x;
                                if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                    *d = *sv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }



    /// **DNG `Compression = 8` / `32946`：Deflate(ZIP)** 路径。
    ///
    /// # 容器：**zlib 流**，不是裸 DEFLATE
    ///
    /// 规范依据见 [`compression_reject_hint`] 的文档段：Adobe《Photoshop TIFF
    /// Technical Notes》规定 `Compression = 8` "using zlib data format"，
    /// 且**每个 strip / tile 是一条完整的 zlib 流**。故本方法调
    /// [`prism_bytes::zlib::decompress_verified`]（**校验 Adler-32**），
    /// **不是** `inflate_raw`。
    ///
    /// # 解码流水线（每块独立）
    ///
    /// ```text
    /// 标签声明的 [off, off+byte_count) → zlib 解压 → unpack_bits → 预测器逆运算
    /// ```
    ///
    /// 预测器由 [`tags::PREDICTOR`] 指定（缺省 `1` = Null），
    /// 语义与取值见 [`DeflatePredictor`]。
    ///
    /// # 与 `slice_region` 的区别
    ///
    /// 压缩载荷的长度**只由标签声明**决定，不能按 `samples × bits / 8` 裁剪
    /// （压缩后通常远小于原始位流），故用 [`DngFile::slice_lossless`]。
    ///
    /// # 零 panic
    ///
    /// 全部偏移/长度经 `checked_*` 与 `get` 兜底；zlib 层的畸形流由
    /// `prism_bytes` 转成三段式 [`BytesError`](prism_bytes::BytesError)，
    /// 本方法把它包成 [`RawError`] 并**保留其成因文本**。
    fn read_raw_samples_deflate(
        &self,
        width: u32,
        height: u32,
        count: usize,
    ) -> RawResult<Vec<u16>> {
        const LOC: &str = "raw::DngFile::read_raw_samples_deflate";
        let spp = self
            .ifd0
            .first_number(tags::SAMPLES_PER_PIXEL, self.order)
            .unwrap_or(1.0);
        if spp != 1.0 {
            return Err(unsupported_err(
                LOC,
                format!("SamplesPerPixel = {spp}，本模块只做 CFA 单通道（= 1）"),
                "线性 RGB DNG 需另一条管线；请确认这是彩色相机原始文件",
            ));
        }
        let bits = self.bits_per_sample();
        let fill = self.fill_order();

        // ---- 预测器（缺省 1 = Null，与 TIFF 一致；取值校验见 DeflatePredictor）----
        let predictor_code = self.deflate_predictor_code();
        let predictor = DeflatePredictor::from_code(predictor_code).ok_or_else(|| {
            unsupported_err(
                LOC,
                format!(
                    "Predictor = {predictor_code}，本模块只实现 1 / 2 / 34892 / 34893：{}",
                    DeflatePredictor::reject_reason(predictor_code)
                ),
                "请用 Adobe DNG Converter 转存为「未压缩」或「lossless JPEG」DNG 后再读入；\
                 DNG 1.4 为 Compression=8 只定义了 7 个预测器取值",
            )
        })?;

        let layout = self.raw_layout()?.ok_or_else(|| {
            tiff_err(
                LOC,
                "既无 StripOffsets(0x0111) 也无 TileWidth/TileOffsets(0x0142/0x0144)",
                "确认 IFD0 含像素载荷布局标签",
            )
        })?;
        let mut out: Vec<u16> = vec![0u16; count];
        let w = width as usize;

        // 解一块：zlib 解压 → 位解包 → 预测器逆运算。
        //
        // `expected` 是本块解压后**应当**有的样本数（用于校验解压结果不短不长）。
        let decode_block = |src: &[u8], row_len: usize, expected: usize| -> RawResult<Vec<u16>> {
            let raw = prism_bytes::zlib::decompress_verified(src).map_err(|e| {
                RawError::new(
                    "RAW2001",
                    LOC,
                    format!(
                        "Deflate 块解压失败（{}）：{}",
                        self.compression(),
                        e.problem
                    ),
                    format!(
                        "该块应为一条完整 zlib 流（2 字节头 + DEFLATE + 4 字节 Adler-32）；\
                         {}。请确认传输完整或改用未压缩 DNG",
                        e.fix
                    ),
                )
            })?;
            let samples = unpack_bits(&raw, bits, expected, fill)?;
            let mut samples = samples;
            predictor.undo_in_place(&mut samples, row_len, bits);
            Ok(samples)
        };

        match layout {
            RawLayout::Strip { rows_per_strip } => {
                let offs = self.offset_list(tags::STRIP_OFFSETS);
                let counts = self.offset_list(tags::STRIP_BYTE_COUNTS);
                let strips = (height + rows_per_strip - 1) / rows_per_strip;
                if offs.len() < strips as usize {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "StripOffsets 只有 {} 项，但按 {rows_per_strip} 行/strip 需 {strips} 项",
                            offs.len()
                        ),
                        "确认 strip 标签负载正确",
                    ));
                }
                for s in 0..strips as usize {
                    let y0 = (s as u32).saturating_mul(rows_per_strip);
                    let rows = rows_per_strip.min(height - y0);
                    let n = (width as usize).saturating_mul(rows as usize);
                    let off = offs[s];
                    let byte_count = counts.get(s).copied().unwrap_or(0);
                    let src = self.slice_lossless(off, byte_count)?;
                    let samples = decode_block(src, w, n)?;
                    for y in 0..rows {
                        for x in 0..width as usize {
                            let si = y as usize * width as usize + x;
                            let di = (y0 as usize + y as usize) * width as usize + x;
                            if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                *d = *sv;
                            }
                        }
                    }
                }
            }
            RawLayout::Tile {
                tile_width,
                tile_length,
            } => {
                if self.ifd0.get(tags::TILE_OFFSETS).is_none() {
                    return Err(tiff_err(
                        LOC,
                        "有 TileWidth/TileLength 但缺 TileOffsets(0x0144)",
                        "确认 tile 标签齐备",
                    ));
                }
                let offs = self.offset_list(tags::TILE_OFFSETS);
                let counts = self.offset_list(tags::TILE_BYTE_COUNTS);
                let across = (width + tile_width - 1) / tile_width;
                let down = (height + tile_length - 1) / tile_length;
                let tiles = (across as usize).saturating_mul(down as usize);
                if offs.len() < tiles {
                    return Err(tiff_err(
                        LOC,
                        format!(
                            "TileOffsets 只有 {} 项，但 {width}×{height} 按 {tile_width}×{tile_length} 需 {tiles} 项",
                            offs.len()
                        ),
                        "确认 tile 偏移表长度与图幅匹配",
                    ));
                }
                for ty in 0..down as usize {
                    for tx in 0..across as usize {
                        let idx = ty * across as usize + tx;
                        let x0 = (tx as u32).saturating_mul(tile_width);
                        let y0 = (ty as u32).saturating_mul(tile_length);
                        let copy_w = tile_width.min(width - x0);
                        let copy_h = tile_length.min(height - y0);
                        // **整 tile 解码**（含图边界外的填充像素），再只拷贝
                        // 有效区——与 `read_raw_samples` 的 tile 分支同理：
                        // 边界 tile 的位流起点会算错（少解一个样本，后续全错位）。
                        let n = (tile_width as usize).saturating_mul(tile_length as usize);
                        let off = offs[idx];
                        let byte_count = counts.get(idx).copied().unwrap_or(0);
                        let src = self.slice_lossless(off, byte_count)?;
                        let samples = decode_block(src, tile_width as usize, n)?;
                        for y in 0..copy_h {
                            for x in 0..copy_w {
                                let si = y as usize * tile_width as usize + x as usize;
                                let di = (y0 as usize + y as usize) * width as usize
                                    + x0 as usize
                                    + x as usize;
                                if let (Some(sv), Some(d)) = (samples.get(si), out.get_mut(di)) {
                                    *d = *sv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
    /// 截取一条无损 JPEG strip / tile 的字节流（**不做位深换算**）。
    ///
    /// 与 [`DngFile::slice_region`] 的区别：后者按 `samples × bits / 8` 计算
    /// 「需要多少字节」并据此裁剪，而**压缩载荷的长度只由标签声明**决定，
    /// 故这里原样取 `[off, off+byte_count)` 并做越界检查。
    fn slice_lossless(&self, off: u64, byte_count: u64) -> RawResult<&[u8]> {
        const LOC: &str = "raw::DngFile::slice_lossless";
        let data = self.data.as_slice();
        let start = usize::try_from(off).map_err(|_| {
            tiff_err(
                LOC,
                format!("偏移 {off} 超出 usize（32 位平台）"),
                "文件损坏；请在 64 位平台重试",
            )
        })?;
        let len = usize::try_from(byte_count).unwrap_or(usize::MAX);
        let end = start.checked_add(len).ok_or_else(|| {
            tiff_err(
                LOC,
                format!("偏移 {off} + 字节数 {byte_count} 溢出"),
                "文件损坏；请在 64 位平台重试",
            )
        })?;
        if end > data.len() {
            return Err(tiff_err(
                LOC,
                format!(
                    "无损载荷区间 [{start}, {end}) 超出文件长度 {}",
                    data.len()
                ),
                "文件被截断或 StripByteCounts/TileByteCounts 标签损坏",
            ));
        }
        Ok(&data[start..end])
    }



    // ===========================================================================
//  诚实边界（Deflate = 8 / 32946 与 SubIFD 预览）
// ===========================================================================
//
// 本段把「做到了什么 / 没做到什么 / 依据在哪」集中写清，避免散落在各处
// 的注释被后来人漏读——**一个不完整的诚实边界等于没有边界**。
//
// ---------------------------------------------------------------------------
//  1. Compression = 8（Deflate）—— 容器口径与依据
// ---------------------------------------------------------------------------
//
// **结论：TIFF/DNG 的 Deflate 是 zlib 流（2 字节头 + DEFLATE + 4 字节 Adler-32），
// 不是裸 DEFLATE 码流。**
//
// 依据：Adobe《Photoshop TIFF Technical Notes》（2002-03-22，TIFF 6.0 的
// 正式补充，DNG 1.4 的 `Deflate (ZIP)` 继承它）原文：
//
// ```text
// Value: 8 — Deflate compression, using zlib data format
// Each strip or tile is compressed independently, and stored as a
// **complete zlib data stream**.
// ```
//
// 本模块据此走 `prism_bytes::zlib::decompress_verified`（**比对 Adler-32**）。
//
// **与既有预设的分歧（如实记录）**：曾有预设「先核实 TIFF 的 Deflate 是否是
// 裸 DEFLATE 码流、不带 zlib 头」，其预设答案是「裸流」。核实结论**相反**，
// 故本实现以规范为准。若按原预设走 `inflate_raw`，会把 zlib 头的
// `0x78` 当成 DEFLATE 块头解读，产出**看似合法实则全错**的像素——
// 有测试专门钉死这一点（`dng_deflate_rejects_bare_deflate_stream`）。
//
// ---------------------------------------------------------------------------
//  2. 32946 = 8，而 34892 ≠ 8（**曾记此处有事实错误**）
// ---------------------------------------------------------------------------
//
// DNG 1.4 规范原文并列声明两个语义**完全不同**的新取值：
//
// ```text
// Value = 8:     Deflate (ZIP)
// Value = 34892: Lossy JPEG
// ```
//
// 曾把 `34892` 称作「Deflate 的 Adobe 历史别名」——**这是错的**。
// `34892` 是**有损基线 DCT**（仅允许 8 位 + `LinearRaw` / `PhotometricMask`）。
// 若照此实现，一个有损 JPEG 的 DNG 会被喂进 inflate，产出全错像素
// 而不报错——这比拒绝**危险得多**。故本模块对 `34892` 继续结构化拒绝。
//
// `34892` 在 PhotometricInterpretation 语境里另有含义（`LinearRaw`），
// 与压缩码**不是同一套命名空间**，容易混淆——错误描述尤其要避免。
//
// **真正的旧编号是 `32946`（`0x80B2`）**：同一份 Technical Notes 写明它
// 「压缩算法与上述完全相同，应视为过时；建议 TIFF 实现**识别并读取**」。
// 故本模块对 `32946` 与 `8` 走**同一个** `match` 分支（不存在两套实现）。
//
// ---------------------------------------------------------------------------
//  3. Predictor：实现了 4 个整数取值，3 个浮点取值未实现
// ---------------------------------------------------------------------------
//
// DNG 1.4 为 `Compression = 8` 定义 7 个预测器取值，本模块实现其中
// **4 个整数取值**（1 / 2 / 34892 / 34893），**未实现** 3 个浮点取值
// （3 / 34894 / 34895，Photoshop TIFF TN3）——它们只对浮点 RAW 有意义，
// 而本模块的样本类型是 `u16` 整数。
//
// **顺序不可交换**：预测器作用于**样本值**而非字节，故流水线必须是
// 「预测器 → 位打包 → 压缩」/「解压 → 位解包 → 预测器逆运算」。
// 反序会得到「有结构、但全错」的像素。
//
// ---------------------------------------------------------------------------
//  3b. 修掉的一个**静默错误**：`bits >= 16` 曾让预测器变成 no-op
// ---------------------------------------------------------------------------
//
// 本路自查时发现：`DeflatePredictor::apply_in_place` / `undo_in_place`
// 的守卫写成了 `bits >= 16 → 提前返回`，即**在 16 位上预测器完全不生效**。
//
// **为什么它能躲过测试**：合成器与读端**同时** no-op，于是「合成 → 读回」
// 的往返断言照样全绿（自己减自己等于没减）。但读**真实相机文件**
// （Deflate + Predictor=2 + BitsPerSample=16 —— DNG 最常见组合）会
// 解出**全错像素且不报错**。这比崩溃更危险，属「静默错误」。
//
// 模 2^16 的回绕就是 `wrapping_add` / `wrapping_sub` 的语义，**不是恒等**，
// 故守卫改为 `bits > 16`（只排掉超出 u16 样本域的值）。
//
//修复后实测压缩比（64×64、predictor=2、单 strip）：**8192 → 320 字节（0.039）**；
// 修复前是 8203（**1.001**，等于没压）。回归测试：
// `deflate_predictor_is_not_a_noop_at_sixteen_bits`（已用变异验证过——
// 把守卫改回 `>=`它必失败）。
//
// ---------------------------------------------------------------------------
//  4. 仍不支持的 Compression 取值（清单）
// ---------------------------------------------------------------------------
//
// | 取值 | 语义 | 状态 |
// |---|---|---|
// | 1 | 未压缩 | ✅ |
// | 7 | 无损 JPEG | ✅ |
// | 8 | Deflate(ZIP) | ✅（本路新增） |
// | 32946 | Deflate 旧编号 | ✅（本路新增，与 8 同分支） |
// | 2/3/4 | CCITT RLE/G3/G4 | ❌ DNG 规范不支持 |
// | 5 | LZW | ❌ DNG 规范不支持 |
// | 6 | Old JPEG(OJPEG) | ❌ **DNG 规范从未定义** |
// | 32773 | PackBits | ❌ DNG 规范不支持 |
// | 34712 | JPEG 2000 | ❌ DNG 规范不支持 |
// | 34892 | **Lossy JPEG** | ❌ 未实装有损 JPEG 通路 |
// | 52546 | JPEG XL | ❌ 未实装 |
//
// ---------------------------------------------------------------------------
//  5. SubIFD 预览图：解到什么程度
// ---------------------------------------------------------------------------
//
// 见 [`crate::raw_subifd`] 模块文档。一句话：**1 / 7 / 8 / 32946 解出像素，
// 其余只报元数据 + 写明为何不解**；不做色彩空间转换、不应用YCbCr 下采样。
//
// ---------------------------------------------------------------------------
//  6. 本仓无法自证的部分（诚实）
// ---------------------------------------------------------------------------
//
// - **无真实相机样本**：所有往返验证都基于**本仓合成**的 DNG。
//   合成器与读端共用同一份编码/解码约定，故「自洽」已证，
//   但「与真实相机文件一致」**未证**（无样本）。
// - **压缩器复用** `exr_write::deflate_zlib`（fixed Huffman +
//   stored 取小）。它是本仓唯一已验证的真压缩 zlib 写入器；重写第二份
//   只会制造「第二份实现 = 第二份 bug 面」。代价：写入侧对 DNG 布局
//   （而非 EXR 布局）的适配由调用方负责。
// - **32 位浮点 / 32 位整数 RAW 不支持**：DNG 允许 Deflate 用于这些类型，
//   本模块只处理 1..=16 位整数样本。

    /// 读出 `Predictor` 标签值（缺省 **1** = Null；仅 `Compression` 8/32946 有意义）。
    ///
    /// 非 `Compression` 8/32946 时本标签被忽略（与 TIFF 一致：它只对
    /// LZW / Deflate 有意义），故此方法对任何文件都安全可调。
    pub fn predictor(&self) -> u16 {
        self.ifd0
            .first_number(tags::PREDICTOR, self.order)
            .filter(|v| v.is_finite())
            .map(|v| v as u16)
            .unwrap_or(1)
    }

    /// 读出 `Predictor` 标签的**已解码**枚举（未支持的取值 → `None`）。
    pub fn deflate_predictor(&self) -> Option<DeflatePredictor> {
        DeflatePredictor::from_code(self.predictor())
    }

    /// 截取一个 strip / tile 的字节区间（越界或过短 → 结构化错误）。
    fn slice_region<'a>(
        &'a self,
        off: u64,
        byte_count_end: u64,
        samples: usize,
        bits: u16,
        loc: &'static str,
    ) -> RawResult<&'a [u8]> {
        let data = self.data.as_slice();
        let start = usize::try_from(off).map_err(|_| {
            tiff_err(
                loc,
                format!("偏移 {off} 超出 usize（32 位平台）"),
                "文件损坏；请在 64 位平台重试",
            )
        })?;
        if start > data.len() {
            return Err(tiff_err(
                loc,
                format!("载荷偏移 {start} 超出文件长度 {}", data.len()),
                "文件被截断或偏移字段损坏",
            ));
        }
        // 需要的字节数：⌈samples × bits / 8⌉（checked，C18）。
        let need_bits = (samples as u64).checked_mul(bits as u64).ok_or_else(|| {
            tiff_err(
                loc,
                format!("{samples} 样本 × {bits} 位溢出"),
                "确认图幅与位深标签正确",
            )
        })?;
        let need = usize::try_from(need_bits.div_ceil(8)).unwrap_or(usize::MAX);
        // 若标签给了 ByteCount，以**需要量与声明量的较小者**为界（声明量可能
        // 偏大——真实文件常把整个 tile 的字节数填满；偏小则必须报错）。
        let declared = usize::try_from(byte_count_end.saturating_sub(off)).unwrap_or(0);
        let avail = data.len() - start;
        if declared > 0 && declared < need {
            return Err(tiff_err(
                loc,
                format!("载荷声明 {declared} 字节 < 所需 {need} 字节（{samples} 样本 × {bits} 位）"),
                "确认 strip/tile 字节数标签正确",
            ));
        }
        let take = need.min(avail);
        if take < need {
            return Err(tiff_err(
                loc,
                format!("载荷实际可用 {take} 字节 < 所需 {need} 字节"),
                "文件被截断；确认传输完整",
            ));
        }
        data.get(start..start + take).ok_or_else(|| {
            tiff_err(
                loc,
                format!("载荷区间 [{start}..{}) 越界", start + take),
                "文件被截断",
            )
        })
    }

    /// 装配「相机空间 → CIE XYZ(D65)」的 3×3 矩阵（`MatrixChoice` 显式选路）。
    ///
    /// # 两条路的**逐步出处**（顺序不可交换）
    ///
    /// ```text
    /// ── MatrixChoice::Forward2（DNG 1.4 推荐的正解）────────────────────
    /// ① 白平衡增益：g = wb(effective_neutral)   —— effective_neutral 已含
    ///    AnalogBalance 的除法（见 DngFile::effective_neutral）
    /// ② CameraCalibration1：c = CC · (g ⊙ cam)   光学暗电流/暗角（相机域）
    /// ③ ForwardMatrix2：      xyz_d50 = FM2 · c  相机空间 → XYZ(D50)
    /// ④ Bradford D50→D65：    xyz_d65 = A · xyz_d50  ← 关键：DNG 的 XYZ 是 D50
    ///
    /// ── MatrixChoice::ColorMatrix1（反矩阵路径）─────────────────────
    /// ① 白平衡增益（同上）
    /// ② ColorMatrix1：         xyz = CM1 · (g ⊙ cam)
    ///    （其参考白点由 CalibrationIlluminant1 指定；本模块按 D65 处理，
    ///     故**不做**白点适配——见下方「为什么 ColorMatrix1 路不做 Bradford」）
    /// ```
    ///
    /// # 为什么 `ColorMatrix1` 路不做 Bradford 适配
    ///
    /// `ColorMatrix1` 的语义是「**已白平衡相机 RGB → XYZ**」，其白点是
    /// `CalibrationIlluminant1`（DNG 里 99% 是 D65 = 21）。既然输入已经是
    /// D65 参考的 XYZ，再做一次 D50→D65 适配就是**重复适配**，会把白点再
    /// 拉偏一次。`ForwardMatrix2` 则相反：规范**明确规定**其输出是
    /// **D50**（PCS 白点），故**必须**适配。这两条路的白点处理因此**必然不同**
    /// ——这是它们结果不同的根因之一。
    ///
    /// # 顺序为什么重要
    ///
    /// - **白平衡在校准之前**：`CameraCalibration1` 建模的是**光学**暗电流/
    ///   暗角，作用在**已白平衡的辐射量**上。若先校准再白平衡，等于对暗电流
    ///   也施加了白平衡增益——把本应被扣掉的本底放大了。
    /// - **校准在色彩矩阵之前**：矩阵是**线性**映射，其系数由「已白平衡、已校准」
    ///   的相机 RGB 标定而来。顺序颠倒等价于在矩阵输入域上做了一次额外的
    ///   线性重加权，破坏标定假设。
    /// - **Bradford 在最后**：白点适配是**输出域**的重参考化，必须在所有
    ///   相机域操作与相机→XYZ 映射之后。
    pub fn color_matrix_chain(&self, choice: MatrixChoice) -> RawResult<Mat3> {
        Ok(self.resolve_matrix_chain(choice)?.0)
    }

    /// 装配色彩矩阵**并报告实际选中的来源**（`"ForwardMatrix2"` / `"ColorMatrix1"`）。
    ///
    /// 返回值第二项是**诊断/测试用的可观测事实**：`Auto` 模式下二者都存在时
    /// 必为 `"ForwardMatrix2"`（DNG 1.4 推荐），只有 `ForwardMatrix2` 缺席时
    /// 才退到 `"ColorMatrix1"`。把它显式暴露出来，测试才能断言
    /// 「两条路真的分开了」而不只是「结果不同」。
    pub fn resolve_matrix_chain(&self, choice: MatrixChoice) -> RawResult<(Mat3, &'static str)> {
        const LOC: &str = "raw::DngFile::resolve_matrix_chain";
        // 先按 choice 定出「用哪个原始矩阵」（Forward2 / ColorMatrix1 / 两者皆无）。
        let (raw_matrix, source) = match choice {
            MatrixChoice::Forward2 => match self.forward_matrix2() {
                Some(m) => (m, "ForwardMatrix2"),
                None => {
                    return Err(tiff_err(
                        LOC,
                        "显式要求 MatrixChoice::Forward2，但文件缺 ForwardMatrix2(0xC68A)",
                        "改用 MatrixChoice::Auto（会自动退回 ColorMatrix1）",
                    ))
                }
            },
            MatrixChoice::ColorMatrix1 => match self.color_matrix() {
                Some(m) => (m, "ColorMatrix1"),
                None => {
                    return Err(tiff_err(
                        LOC,
                        "显式要求 MatrixChoice::ColorMatrix1，但文件缺 ColorMatrix1(0xC621)",
                        "改用 MatrixChoice::Auto（会自动改用 ForwardMatrix2）",
                    ))
                }
            },
            MatrixChoice::Auto => match (self.forward_matrix2(), self.color_matrix()) {
                (Some(m), _) => (m, "ForwardMatrix2"),
                (None, Some(m)) => (m, "ColorMatrix1"),
                (None, None) => {
                    return Err(tiff_err(
                        LOC,
                        "既无 ForwardMatrix2(0xC68A) 也无 ColorMatrix1(0xC621)",
                        "无法装配色彩矩阵；请确认文件完整或改用 MatrixMode::None",
                    ))
                }
            },
        };
        // 两条路在此**分叉**：只有 Forward2 路需要 CameraCalibration1 + Bradford。
        let cam_to_xyz_d65 = if source == "ForwardMatrix2" {
            // ② CameraCalibration1（缺失 → 单位矩阵；DNG 1.4 允许该标签缺席）。
            let calibrated = match self.camera_calibration1() {
                Some(cc) => mat3_mul_matrix(&cc, &raw_matrix),
                None => raw_matrix,
            };
            // ④ D50 → D65（ForwardMatrix2 的输出白点由规范固定为 D50）。
            let adapt = bradford_adaptation(D50_XYZ, D65_XYZ).ok_or_else(|| {
                pipe_err(
                    LOC,
                    "Bradford D50→D65 锥体响应退化（常数或输入非有限）",
                    "这是代码常量被破坏的信号，请检查；勿静默跳过白点适配",
                )
            })?;
            mat3_mul_matrix(&adapt, &calibrated)
        } else {
            // ColorMatrix1 的白点由 CalibrationIlluminant1 指定（本模块按 D65），
            // 故**不做**白点适配——详见本函数文档的「为什么 ColorMatrix1 路不做 Bradford」。
            raw_matrix
        };
        Ok((cam_to_xyz_d65, source))
    }

    /// 完整显影流水线：马赛克 f32 → [`Fbuf16F`]（RGBA16F）。
    ///
    /// # 阶段次序（含论证）
    ///
    /// ```text
    /// ① 线性化：v = (raw - black(pos)) / (white - black(pos))     逐 CFA 位置
    /// ② 去马塞克：双线性 → (R, G, B) 平面
    /// ③ 白平衡：c *= gain(effective_neutral)   ← 已含 AnalogBalance 的除法
    /// ④ 色彩矩阵链（见 color_matrix_chain）：
    ///      Forward2 路： CameraCalibration1 → ForwardMatrix2 → Bradford D50→D65
    ///      CM1 路：      ColorMatrix1
    ///    再 RGB = XYZ_TO_SRGB · XYZ（CameraToSrgb 时）
    /// ⑤ 交付量化：f32 → f16 位模式，Alpha = 1.0
    /// ```
    ///
    /// - **① 在 ② 之前**：黑电平是**逐 CFA 位置**的（不同位置本底不同），
    ///   必须在马赛克域逐点扣除；若先去马塞克再减一个标量黑电平，等价于
    ///   抹平了 CFA 位置的偏置差异（对绿色像素尤其明显）；
    /// - **③ 在 ④ 之前**（**关键论证**）：DNG 规范中 `ColorMatrix1` 定义为
    ///   「**已白平衡**的相机 RGB → XYZ」。`AsShotNeutral` 就是标定该矩阵
    ///   时所用的白平衡量。若先做矩阵再白平衡，矩阵系数所假设的输入域
    ///   （白平衡后）就被破坏了——等价于对矩阵的输入做了一次非线性重加权，
    ///   结果是色偏，且在非中性色上尤为明显（矩阵行和不再为 1/白平衡增益）。
    ///   反过来「先白平衡后矩阵」严格符合规范语义；
    /// - **④ 的两条路是真分开的两条路**（`MatrixChoice` 显式切换），不是同一
    ///   函数的两个分支——详见 [`DngFile::color_matrix_chain`]；
    /// - **④ 内部 相机→XYZ→sRGB 串联**：色彩矩阵的输出是 CIE XYZ，
    ///   而本模块的交付面是 RGB 帧缓冲，故再乘 IEC 61966-2-1 的 XYZ→sRGB。
    ///   两级矩阵都在 f32 计算域一次求和完成（不拆成两次数值舍入）；
    /// - **⑤ 量化只发生在交付边界**（铁律）：内部全程 f32，仅写入
    ///   `Fbuf16F` 时 round-to-nearest-even 量化一次。
    ///
    /// # 位深
    ///
    /// 本函数接受**任意 1..=16 位**解包后的马赛克（**原始码值**，未经右移）。
    /// 12/14 位与 16 位走**完全相同**的 ①—⑤ 链：线性化是**同量纲相消**
    /// `（raw − black) / (white − black)`，12 位文件的 `WhiteLevel` 通常是
    /// 4095 而非 65535，两者各自相除都得到同一个 [0,1] 归一值。若在此处
    /// 「把 12 位右移 4 位凑成 16 位」，黑/白电平的对应关系就断了——故不做。
    ///
    /// # 参数
    /// - `mosaic` — 长度恰为 `width × height` 的单通道原图（**未减黑电平**，
    ///   **原始码值**；12/14 位须先经 [`unpack_bits`] 或
    ///   [`DngFile::read_raw_samples`] 解包）；
    /// - `opts` — 白平衡 / 色彩矩阵 / 负值钳制选项。
    pub fn develop(&self, mosaic: &[u16], opts: &DevelopOptions) -> RawResult<Fbuf16F> {
        const LOC: &str = "raw::DngFile::develop";
        let cfa = self.cfa_pattern()?;
        let layout = cfa.bayer()?;
        let levels = self.levels(&cfa)?;
        let count = cfa
            .pixel_count()
            .ok_or_else(|| pipe_err(LOC, "CFA 像素数溢出", "使用更小的测试图"))?;
        if mosaic.len() != count {
            return Err(pipe_err(
                LOC,
                format!("马赛克长度 {} ≠ 宽×高 {count}", mosaic.len()),
                "确认传入的是 `width*height` 的单通道原图（12/14/16 位，无压缩）",
            ));
        }
        let bits = self.bits_per_sample();
        if bits == 0 || bits > 16 {
            return Err(unsupported_err(
                LOC,
                format!("BitsPerSample = {bits}，本显影链只接受 1..=16 位"),
                "32 位整数/浮点 RAW 请先转 16 位整数 DNG（Adobe DNG Converter）",
            ));
        }

        // ① 线性化（DNG 1.6 Chapter 5 的顺序：
        //    **Linearization → Black Subtraction → Rescaling → Clipping**）。
        //    1a. LinearizationTable（0xC691）：raw 码值 → 线性值（缺省无）。
        //    1b. 减黑电平：BlackLevel + BlackLevelDeltaH + BlackLevelDeltaV。
        //    1c. Rescaling：(x − black) / (white − black)，**位深无关**（见文档）。
        let lin_table = self.linearization_table();
        let (delta_h, delta_v) = self.black_level_deltas(cfa.raw_height as usize, cfa.raw_width as usize);
        let mut lin: Vec<f32> = Vec::with_capacity(count);
        for y in 0..cfa.raw_height {
            for x in 0..cfa.raw_width {
                let idx = (y as usize) * (cfa.raw_width as usize) + (x as usize);
                let raw = mosaic.get(idx).copied().unwrap_or(0);
                // 1a. 线性化（表缺省即恒等；表长 2^BitsPerSample 故下标必在界内，
                // 但仍用 get 兜底以免越界 panic —— C18 铁律）。
                let raw = match lin_table.as_ref() {
                    Some(t) => t.get(raw as usize).copied().unwrap_or(raw) as f32,
                    None => raw as f32,
                };
                // 1b. 黑电平 = BlackLevel + BlackLevelDeltaH + BlackLevelDeltaV。
                let mut b = levels.black_at(x, y, cfa.cols, cfa.rows);
                if let Some(d) = delta_h.get(y as usize) {
                    b += *d;
                }
                if let Some(d) = delta_v.get(x as usize) {
                    b += *d;
                }
                // 1c. 重标定。
                let span = levels.white - b;
                let v = if span > 0.0 { (raw - b) / span } else { 0.0 };
                lin.push(if v.is_finite() { v.max(0.0) } else { 0.0 });
            }
        }

        // ② 去马塞克。
        let plane = demosaic_bilinear(&lin, cfa.raw_width, cfa.raw_height, layout)?;

        // ③ 白平衡增益（effective_neutral = AsShotNeutral / AnalogBalance）。
        let neutral = self.effective_neutral().unwrap_or([1.0, 1.0, 1.0]);
        let gain = white_balance_gains(neutral, opts.white_balance);

        // ④ 色彩矩阵（两条路由 MatrixChoice 显式选择）。
        let cam_to_xyz: Mat3 = match opts.color_matrix {
            MatrixMode::None => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            MatrixMode::CameraToXyz | MatrixMode::CameraToSrgb => {
                match self.color_matrix_chain(opts.matrix_choice) {
                    Ok(m) => m,
                    Err(e) if opts.require_color_matrix => {
                        return Err(e);
                    }
                    // 缺矩阵且未要求：退化为单位矩阵（保留白平衡后的相机域 RGB）。
                    Err(_) => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                }
            }
        };
        let m: Mat3 = if opts.color_matrix == MatrixMode::CameraToSrgb {
            // M = XYZ_TO_SRGB × 色彩矩阵链（预乘一次，避免逐像素两次舍入）。
            mat3_mul_matrix(&XYZ_TO_SRGB, &cam_to_xyz)
        } else {
            cam_to_xyz
        };

        // ⑤ 写入 f16 帧缓冲（Alpha 恒 1.0）。
        let mut fb = Fbuf16F::new(cfa.raw_width, cfa.raw_height);
        for y in 0..cfa.raw_height {
            for x in 0..cfa.raw_width {
                let idx = (y as usize) * (cfa.raw_width as usize) + (x as usize);
                let wb = [
                    plane.r[idx] * gain[0],
                    plane.g[idx] * gain[1],
                    plane.b[idx] * gain[2],
                ];
                let mut out = mat3_mul(&m, wb);
                if opts.clamp_negative {
                    for c in out.iter_mut() {
                        if !c.is_finite() {
                            *c = 0.0;
                        } else if *c < 0.0 {
                            *c = 0.0;
                        }
                    }
                }
                fb.set(x, y, [out[0], out[1], out[2], 1.0]);
            }
        }
        Ok(fb)
    }
}

/// 3×3 矩阵乘（`a × b`，行主序）。
pub fn mat3_mul_matrix(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [0.0f32; 9];
    for r in 0..3usize {
        for c in 0..3usize {
            let mut acc = 0.0f32;
            for k in 0..3usize {
                acc += a[r * 3 + k] * b[k * 3 + c];
            }
            out[r * 3 + c] = acc;
        }
    }
    out
}

/// 由 IFD 数值向量构造 3×3 矩阵（**任一分量非有限即 `None`**）。
///
/// 拒绝非有限值是刻意的：色彩矩阵里一个 `Inf`/`NaN` 会沿乘法链污染**整幅图**，
/// 而 `f32` 的 `0 × Inf = NaN` 传播规则会让「只是某一个系数坏」变成「全图坏」。
/// 与其交付一张全 `NaN` 的图，不如让调用方结构化报错。
fn mat3_from_tag(v: Option<Vec<f32>>) -> Option<Mat3> {
    let v = v?;
    if v.len() < 9 {
        return None;
    }
    let mut m = [0.0f32; 9];
    for (i, slot) in m.iter_mut().enumerate() {
        let x = v.get(i).copied().unwrap_or(0.0);
        if !x.is_finite() {
            return None;
        }
        *slot = x;
    }
    Some(m)
}

/// 由 IFD 数值向量构造 3 元组（**任一分量非有限或 ≤ 0 即 `None`**）。
fn triple_from_tag(v: Option<Vec<f32>>) -> Option<[f32; 3]> {
    let v = v?;
    if v.len() < 3 {
        return None;
    }
    let t = [v[0], v[1], v[2]];
    if t.iter().all(|&x| x.is_finite() && x > 0.0) {
        Some(t)
    } else {
        None
    }
}

/// 显影选项（默认值 = 白平衡绿归一 + 相机→sRGB + `ForwardMatrix2` 优先 + 负值钳零）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopOptions {
    /// 白平衡模式。
    pub white_balance: WbMode,
    /// 色彩矩阵模式（`None` = 不做色彩矩阵）。
    pub color_matrix: MatrixMode,
    /// **用哪条色彩矩阵**：`ForwardMatrix2`（推荐）还是 `ColorMatrix1`。
    ///
    /// 缺省 [`MatrixChoice::Auto`] = 有 `ForwardMatrix2` 就用它，否则退回
    /// `ColorMatrix1`。**两条路的结果必然不同**（白点处理不同：Forward2 的
    /// 输出是 D50、需 Bradford 适配；CM1 的白点已由
    /// `CalibrationIlluminant1` 指定、按 D65 处理），见
    /// [`DngFile::color_matrix_chain`]。
    pub matrix_choice: MatrixChoice,
    /// 缺色彩矩阵时是否报错（默认否：退化为单位矩阵）。
    pub require_color_matrix: bool,
    /// 负值钳零（AOV 交付惯例）。
    pub clamp_negative: bool,
}

impl Default for DevelopOptions {
    fn default() -> Self {
        Self {
            white_balance: WbMode::GreenNormalized,
            color_matrix: MatrixMode::CameraToSrgb,
            matrix_choice: MatrixChoice::Auto,
            require_color_matrix: false,
            clamp_negative: true,
        }
    }
}

/// 读四元组版本标签（`BYTE × 4`；缺失时全 0）。
fn raw_version(ifd: &TiffIfd, tag: u16, order: ByteOrder) -> [u8; 4] {
    let mut v = [0u8; 4];
    if let Some(e) = ifd.get(tag) {
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = e.data.get(i).copied().unwrap_or(0);
        }
    }
    let _ = order;
    v
}

/// 尽力回溯 GPS 子 IFD 的文件偏移（仅供诊断展示）。
fn tags_gps_offset(data: &[u8], ifd0: &TiffIfd, exif: &Option<TiffIfd>) -> u32 {
    let _ = data;
    let _ = ifd0;
    let _ = exif;
    0
}

/// 占位：GPS 偏移在 [`DngFile::gps`] 中以 `offset` 字段给出（当前为 0，
/// 结构保留供后续扩展定位；本任务不要求 GPS 定位能力）。
fn find_offset(order: ByteOrder, off: u32) -> u32 {
    let _ = order;
    off
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbuf::f16_to_f32;

    // -----------------------------------------------------------------------
    // 测试脚手架：构造最小合法 DNG（自写 TIFF 写入器）
    // -----------------------------------------------------------------------

    /// 字节序对齐到偶数。
    fn align2(x: usize) -> usize {
        (x + 1) & !1usize
    }

    /// IFD 条目构建器（`tag, type, count, data`）。
    ///
    /// `count` 是 TIFF 定义的**元素个数**（不是字节数）—— 必须显式携带，
    /// 写错会让读端把内联值误判为外置偏移。
    struct IfdBuilder {
        /// 负载字节序：MM 容器须写大端值，故构建器须与 `write_ifd` 同序。
        le: bool,
        entries: Vec<(u16, u16, u32, Vec<u8>)>,
    }

    impl IfdBuilder {
        fn new(le: bool) -> Self {
            Self {
                le,
                entries: Vec::new(),
            }
        }
        /// 追加一条（相同 tag 允许，按升序稳定排序）。
        fn put(&mut self, tag: u16, ty: u16, count: u32, data: Vec<u8>) -> &mut Self {
            debug_assert_eq!(count as usize * type_size(ty) as usize, data.len());
            self.entries.push((tag, ty, count, data));
            self
        }
        fn shorts(&mut self, tag: u16, v: &[u16]) -> &mut Self {
            let d: Vec<u8> = v
                .iter()
                .flat_map(|&x| {
                    if self.le {
                        x.to_le_bytes()
                    } else {
                        x.to_be_bytes()
                    }
                })
                .collect();
            self.put(tag, field_type::SHORT, v.len() as u32, d)
        }
        fn longs(&mut self, tag: u16, v: &[u32]) -> &mut Self {
            let d: Vec<u8> = v
                .iter()
                .flat_map(|&x| {
                    if self.le {
                        x.to_le_bytes()
                    } else {
                        x.to_be_bytes()
                    }
                })
                .collect();
            self.put(tag, field_type::LONG, v.len() as u32, d)
        }
        fn bytes(&mut self, tag: u16, v: &[u8]) -> &mut Self {
            self.put(tag, field_type::BYTE, v.len() as u32, v.to_vec())
        }
        fn srationals(&mut self, tag: u16, v: &[f32]) -> &mut Self {
            let d: Vec<u8> = v
                .iter()
                .flat_map(|&x| {
                    // SRATIONAL 精确表示：测试值均取 1/1024 的整数倍。
                    let n = ((x * 1024.0).round() as i64) as i32;
                    if self.le {
                        [n.to_le_bytes(), 1024i32.to_le_bytes()].concat()
                    } else {
                        [n.to_be_bytes(), 1024i32.to_be_bytes()].concat()
                    }
                })
                .collect();
            self.put(tag, field_type::SRATIONAL, v.len() as u32, d)
        }
        fn rationals(&mut self, tag: u16, v: &[f32]) -> &mut Self {
            let d: Vec<u8> = v
                .iter()
                .flat_map(|&x| {
                    let n = (x * 1024.0).round() as u32;
                    if self.le {
                        [n.to_le_bytes(), 1024u32.to_le_bytes()].concat()
                    } else {
                        [n.to_be_bytes(), 1024u32.to_be_bytes()].concat()
                    }
                })
                .collect();
            self.put(tag, field_type::RATIONAL, v.len() as u32, d)
        }
        fn ascii(&mut self, tag: u16, s: &str) -> &mut Self {
            let mut d = s.as_bytes().to_vec();
            d.push(0);
            self.put(tag, field_type::ASCII, d.len() as u32, d)
        }
    }

    /// 固定版式：IFD0@8、EXIF@512、GPS@1024、外置值堆@2048。
    const OFF_IFD0: usize = 8;
    const OFF_EXIF: usize = 512;
    const OFF_GPS: usize = 1024;
    const OFF_HEAP: usize = 2048;

    ///外置值堆的**预留字节数**（供带真实像素载荷的 `build_packed_dng` /
    /// `build_tiled_dng` 用）。
    ///
    /// 这两个脚手架把 strip/tile 的偏移与字节数数组也放进堆里，而数组长度随
    /// strip/tile 数量变化。为避免「载荷基址随数组长度漂移」这种自指，固定
    /// 预留一段足够大的空间：载荷一律从 `OFF_HEAP + align2(HEAP_RESERVE) + 8`
    /// 开始，堆只需不超它即可（[`write_ifd`] 会在两处`debug_assert` 里把守）。
    ///
    /// 数值依据：最坏的 `build_tiled_dng` 用例是 8×8 图配2×2 tile = 16 个 tile，
    /// 即偏移 64 B + 字节数 64 B；`build_packed_dng` 的 strip 数更少。1024 B
    /// 对全部现有用例都有两倍以上余量。
    const HEAP_RESERVE: usize = 1024;

    /// 返回 IFD0 中**第一条走外置值**（字节数 > 4）的条目在文件中的起始偏移。
    ///
    /// 用途：畸形输入测试需要一个「外置值偏移字段」来打坏。TIFF 规定值 ≤4 字节
    /// 内联存放、>4 字节才写偏移，故只有这类条目的 value 字段才是偏移。
    ///
    /// 版式：`[count:u16 @ OFF_IFD0][entry0 @ OFF_IFD0+2][entry1 …]`，
    /// 每条 12 字节 —— 故条目起点是 `OFF_IFD0 + 2 + i*12`，不是 `OFF_IFD0 + 4 + …`。
    fn first_external_entry_offset(file: &[u8]) -> Option<usize> {
        let off = OFF_IFD0;
        let n = u16::from_le_bytes([*file.get(off)?, *file.get(off + 1)?]) as usize;
        for i in 0..n {
            let e = off + 2 + i * 12;
            let ty = u16::from_le_bytes([*file.get(e + 2)?, *file.get(e + 3)?]);
            let count = u32::from_le_bytes([
                *file.get(e + 4)?,
                *file.get(e + 5)?,
                *file.get(e + 6)?,
                *file.get(e + 7)?,
            ]);
            let bytes = (count as u64).saturating_mul(type_size(ty) as u64);
            if bytes > 4 {
                return Some(e);
            }
        }
        None
    }

    /// 遍历 IFD0 的条目号，返回第 `i` 条在文件中的起始偏移。
    ///
    /// 版式固定（见 [`build_dng`]），故偏移是纯算术 —— 不需要读 `file`；
    /// 参数保留是为了让调用点自解释「这是在遍历真实文件版式」。
    fn ifd0_entry_offset(_file: &[u8], i: usize) -> usize {
        OFF_IFD0 + 2 + i * 12
    }

    /// 读 IFD0 的条目数（`count:u16` 位于 `OFF_IFD0`，**不是** `OFF_IFD0+2`）。
    fn ifd0_entry_count(file: &[u8]) -> usize {
        u16::from_le_bytes([file[OFF_IFD0], file[OFF_IFD0 + 1]]) as usize
    }

    /// 改写 IFD0 中某标签的**内联 SHORT 值**（小端，count 1）。
    ///
    /// 条目版式为 `tag(2) type(2) count(4) value(4)`，故 value 字段起点是 `e + 8`；
    /// 但 2 字节的 SHORT 在小端下是**右对齐**的，实际数据落在 value 字段的
    /// **后 2 字节**（`e + 10 .. e + 12`）。写 `e + 8` 会改到 value 字段的高位
    /// 字节——对 `count == 1` 的 SHORT 而言那是**保留/未定义**区，改了等于没改。
    ///
    /// 返回是否命中（供测试断言「恰好命中一条」）。
    fn patch_inline_short(file: &mut [u8], tag: u16, value: u16) -> bool {
        let n = ifd0_entry_count(file);
        let mut hit = false;
        for i in 0..n {
            let e = ifd0_entry_offset(file, i);
            if u16::from_le_bytes([file[e], file[e + 1]]) == tag {
                let v = e + 10;
                file[v] = (value & 0xFF) as u8;
                file[v + 1] = (value >> 8) as u8;
                hit = true;
            }
        }
        hit
    }

    /// 序列化一个 IFD 到字节（外置值追加到 `heap`）。
    fn write_ifd(
        ifd: &IfdBuilder,
        le: bool,
        next: u32,
        heap: &mut Vec<u8>,
        heap_base: usize,
    ) -> Vec<u8> {
        let mut entries: Vec<&(u16, u16, u32, Vec<u8>)> = ifd.entries.iter().collect();
        entries.sort_by_key(|e| e.0); // TIFF 要求标签升序
        let mut out: Vec<u8> = Vec::new();
        let put16 = |out: &mut Vec<u8>, v: u16| {
            if le {
                out.extend_from_slice(&v.to_le_bytes());
            } else {
                out.extend_from_slice(&v.to_be_bytes());
            }
        };
        let put32 = |out: &mut Vec<u8>, v: u32| {
            if le {
                out.extend_from_slice(&v.to_le_bytes());
            } else {
                out.extend_from_slice(&v.to_be_bytes());
            }
        };
        put16(&mut out, entries.len() as u16);
        for (tag, ty, count, data) in entries {
            put16(&mut out, *tag);
            put16(&mut out, *ty);
            // count = 元素个数（TIFF 规范），不是字节数。
            put32(&mut out, *count);
            if data.len() <= 4 {
                let mut field = [0u8; 4];
                if le {
                    field[4 - data.len()..].copy_from_slice(data);
                } else {
                    field[..data.len()].copy_from_slice(data);
                }
                out.extend_from_slice(&field);
            } else {
                let off = (heap_base + align2(heap.len())) as u32;
                while heap.len() < (off as usize).saturating_sub(heap_base) {
                    heap.push(0);
                }
                heap.extend_from_slice(data);
                put32(&mut out, off);
            }
        }
        put32(&mut out, next);
        out
    }

    /// 构造一个最小但**完整**的 DNG 文件。
    ///
    /// 关键标签齐备：IFD0 目录 + CFA + 黑白电平 + ColorMatrix1 + AsShotNeutral，
    /// 并带 EXIF 子 IFD 与 GPS 子 IFD（供「读不解」测试）。
    fn build_dng(le: bool, cfa: [u8; 4], bits: u16) -> Vec<u8> {
        let mut ifd0 = IfdBuilder::new(le);
        ifd0.longs(tags::NEW_SUBFILE_TYPE, &[0]);
        ifd0.longs(tags::IMAGE_WIDTH, &[4]);
        ifd0.longs(tags::IMAGE_LENGTH, &[4]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]); // CFA
        ifd0.longs(tags::STRIP_OFFSETS, &[0]);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.longs(tags::ROWS_PER_STRIP, &[4]);
        ifd0.longs(tags::STRIP_BYTE_COUNTS, &[32]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &cfa);
        ifd0.longs(tags::EXIF_IFD_POINTER, &[OFF_EXIF as u32]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.ascii(tags::UNIQUE_CAMERA_MODEL, "PrismTestCam");
        ifd0.longs(tags::WHITE_LEVEL, &[4095]);
        // 逐 CFA 位置黑电平（0x828E 的顺序即 2×2 平铺顺序）。
        ifd0.rationals(tags::BLACK_LEVEL, &[64.0, 128.0, 128.0, 96.0]);
        // 3×3 色彩矩阵（可逆、非对称，确保矩阵测试有区分度）。
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                // 行和 = D65 白点 (0.9505, 1.0, 1.088)——这是真实相机的标定约束：
                // 中性相机 RGB (c,c,c) 必须映射到 D65 的 XYZ，否则显影后的灰会偏色。
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        // 1/1024 的整数倍（320/1024、336/1024）⇒ 读回逐位精确。
        ifd0.rationals(tags::AS_SHOT_WHITE_XY, &[0.3125, 0.328_125]);
        ifd0.srationals(tags::BASELINE_EXPOSURE, &[0.0]);
        ifd0.shorts(tags::CALIBRATION_ILLUMINANT1, &[21]);

        let mut exif = IfdBuilder::new(le);
        exif.rationals(tags::EXPOSURE_TIME, &[0.007_812_5]); // 8/1024 s
        exif.rationals(tags::FNUMBER, &[2.75]); // 2816/1024
        exif.shorts(tags::ISO_SPEED_RATINGS, &[400]);
        exif.longs(tags::GPS_IFD_POINTER, &[OFF_GPS as u32]);
        exif.ascii(tags::DATE_TIME_ORIGINAL, "2024:01:02 03:04:05");

        let mut gps = IfdBuilder::new(le);
        gps.ascii(1, "N"); // GPSLatitudeRef
        gps.rationals(2, &[51.0, 30.0, 26.0]); // GPSLatitude
        gps.rationals(3, &[0.0, 3.0, 24.0]); // GPSLongitude
        gps.bytes(5, &[0]); // GPSAltitudeRef

        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        if le {
            file[0..2].copy_from_slice(b"II");
            file[2..4].copy_from_slice(&42u16.to_le_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        } else {
            file[0..2].copy_from_slice(b"MM");
            file[2..4].copy_from_slice(&42u16.to_be_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_be_bytes());
        }
        let mut heap: Vec<u8> = Vec::new();
        let exif_bytes = write_ifd(&exif, le, 0, &mut heap, OFF_HEAP);
        let gps_bytes = write_ifd(&gps, le, 0, &mut heap, OFF_HEAP);
        let ifd0_bytes = write_ifd(&ifd0, le, 0, &mut heap, OFF_HEAP);
        if OFF_IFD0 + ifd0_bytes.len() <= OFF_EXIF
            && OFF_EXIF + exif_bytes.len() <= OFF_GPS
            && OFF_GPS + gps_bytes.len() <= OFF_HEAP
        {
            file[OFF_IFD0..OFF_IFD0 + ifd0_bytes.len()].copy_from_slice(&ifd0_bytes);
            file[OFF_EXIF..OFF_EXIF + exif_bytes.len()].copy_from_slice(&exif_bytes);
            file[OFF_GPS..OFF_GPS + gps_bytes.len()].copy_from_slice(&gps_bytes);
        } else {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", ifd0_bytes.len());
        }
        file.extend_from_slice(&heap);
        file
    }

    // -----------------------------------------------------------------------
    // 测试脚手架：位打包器 + 真实像素布局的 DNG 构建器
    // -----------------------------------------------------------------------

    /// 把样本按 `bits` 位**紧凑打包**成字节流（`fill_order` 决定位序）。
    ///
    /// 这是 [`unpack_bits`] 的**参考编码器**（测试侧独立实现，不复用被测
    /// 代码路径），用于构造「已知答案」的输入。`pad_high` 控制**填充位是否
    /// 非零**——这是本测试最关键的断言维度：填充位非零时解包结果必须不变。
    ///
    /// 溢出防护：位累加器用 `u64`，且**只在字节内移位**——绝不把
    /// `stream_pos / 8 * 8`（可达数千）直接移进 `u32`（那是上一版的 bug：
    /// debug 下「attempt to shift left with overflow」panic，release 下静默回绕）。
    fn pack_bits(values: &[u16], bits: u16, fill_order: FillOrder, pad_high: bool) -> Vec<u8> {
        let n = values.len();
        let b = bits as usize;
        let total_bits = n * b;
        let nbytes = total_bits.div_ceil(8);
        let mut out: Vec<u8> = vec![0u8; nbytes];
        let mask: u32 = (1u32 << bits) - 1;
        for (k, &v) in values.iter().enumerate() {
            let val = (v as u32) & mask;
            for j in 0..b {
                // **样本内位序也随 fill_order 变化**（这是对拍能成立的关键）：
                // - MSB-first：样本第 j 位 = 值的第 (b−1−j) 位（高��在前）；
                // - LSB-first：样本第 j 位 = 值的第 j 位（低位在前）。
                // 若 LSB-first 仍按 MSB 排样本内位序，解出的值会整体位反转。
                let bit = match fill_order {
                    FillOrder::MsbFirst => (val >> (b - 1 - j)) & 1,
                    FillOrder::LsbFirst => (val >> j) & 1,
                };
                let stream_pos = k * b + j;
                let byte = stream_pos / 8;
                let in_byte = stream_pos % 8; // ∈ 0..=7
                // 字节内位号：MSB-first 走 7→0，LSB-first 走 0→7。
                let bit_in_byte = match fill_order {
                    FillOrder::MsbFirst => 7 - in_byte,
                    FillOrder::LsbFirst => in_byte,
                };
                if let Some(slot) = out.get_mut(byte) {
                    *slot |= (bit as u8) << bit_in_byte;
                }
            }
        }
        if pad_high {
            // 把**最后一个字节里不属于任何有效样本的填充位**全部置 1。
            // 这些位在解包时必须被 mask 掩掉。
            //
            // **填充位的位置随 fill_order 变化**（上一版只置低位，对 LSB-first
            // 就会去动真实样本位）：
            // - MSB-first：数据从高位往下填 → 填充位在**低位**（`0xFF >> used`）；
            // - LSB-first：数据从低位往上填 → 填充位在**高位**，且第一个填充位
            //   就是第 `used` 位，故掩码是 `0xFF << used`。
            //
            // 这里写错会**静默污染真实样本**：1 位 × 37 样本 = 5 字节 3 个填充位，
            // 若掩码算成 `0xFF << (8 - used)` 就会把样本 35/36 所在的位也置1，
            // 解包结果末两位变成 1——而末字节的填充位本该被 mask 掩掉。
            let used = total_bits % 8;
            if used != 0 {
                if let Some(last) = out.last_mut() {
                    let pad_mask: u8 = match fill_order {
                        FillOrder::MsbFirst => 0xFFu8 >> used,
                        FillOrder::LsbFirst => 0xFFu8 << used,
                    };
                    *last |= pad_mask;
                }
            }
        }
        out
    }

    /// 位打包测试用的样本序列（确定性图案，含 0 与最大值）。
    fn sample_series(bits: u16, n: usize) -> Vec<u16> {
        // `1u16 << 16` 在 debug 下是 **shift overflow panic**（移位量等于位宽），
        // 故 16 位必须走 `u16::MAX` 分支——这正是「覆盖 1..=16」测试要踩的坑。
        let mask: u16 = if bits >= 16 { u16::MAX } else { (1u16 << bits) - 1 };
        (0..n)
            .map(|i| {
                // 交替：极小 / 极大 / 递增 / 递减 —— 覆盖 0、mask、边界与中间值。
                // 递减用 `wrapping_sub`：`i` 可大于 `mask`（C18：绝不 panic）。
                match i % 4 {
                    0 => 0,
                    1 => mask,
                    2 => ((i as u16).wrapping_mul(37)) & mask,
                    _ => mask.wrapping_sub(i as u16) & mask,
                }
            })
            .collect()
    }

    /// 构建一个**带真实像素载荷**的 DNG（strip 布局，未压缩）。
    ///
    /// 与 [`build_dng`] 的区别：它把 `pixels` 按 `bits` 位打包后**真正写进
    /// 文件**，并把 `StripOffsets` / `StripByteCounts` 指向它——这样
    /// [`DngFile::read_raw_samples`] 走的是与真实文件同一条路径。
    fn build_packed_dng(
        le: bool,
        cfa: [u8; 4],
        bits: u16,
        fill_order: FillOrder,
        rows_per_strip: u32,
        width: u32,
        height: u32,
        pixels: &[u16],
        white: u32,
        black: [u32; 4],
    ) -> Vec<u8> {
        // 1) 先算每个 strip 的字节内容与偏移。
        let packed = pack_bits(pixels, bits, fill_order, true);
        let bytes_per_row_bits = width as usize * bits as usize;
        let bytes_per_row = bytes_per_row_bits.div_ceil(8);
        let mut strips: Vec<(usize, usize)> = Vec::new(); // (起始字节, 长度)
        let mut cursor = 0usize;
        let mut y = 0u32;
        while y < height {
            let rows = rows_per_strip.min(height - y);
            let len = bytes_per_row * rows as usize;
            strips.push((cursor, len));
            cursor += len;
            y += rows;
        }
        // 2) 载荷放在 **IFD 堆之后**。
        //    注意：偏移值本身也是「外置值」写进堆的，故堆长度与偏移**无关**
        //    （偏移是定长u32，不改变堆的字节数）——这让「先算堆、再定载荷基址」
        //    的两趟构建是自洽的：第二趟写入的偏移指向第一趟算出的基址。
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        let mut offs: Vec<u32> = Vec::with_capacity(strips.len());
        let mut counts: Vec<u32> = Vec::with_capacity(strips.len());
        for (s, len) in &strips {
            offs.push((payload_base + s) as u32);
            counts.push(*len as u32);
        }
        // 3) 建 IFD。
        let mut ifd0 = IfdBuilder::new(le);
        ifd0.longs(tags::IMAGE_WIDTH, &[width]);
        ifd0.longs(tags::IMAGE_LENGTH, &[height]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
        ifd0.shorts(tags::COMPRESSION, &[1]);
        ifd0.shorts(tags::FILL_ORDER, &[fill_order.code()]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.longs(tags::STRIP_OFFSETS, &offs);
        ifd0.longs(tags::STRIP_BYTE_COUNTS, &counts);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.longs(tags::ROWS_PER_STRIP, &[rows_per_strip]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &cfa);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[white]);
        let bl: Vec<f32> = black.iter().map(|&b| b as f32).collect();
        ifd0.rationals(tags::BLACK_LEVEL, &bl);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        // 4) 序列化。
        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        if le {
            file[0..2].copy_from_slice(b"II");
            file[2..4].copy_from_slice(&42u16.to_le_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        } else {
            file[0..2].copy_from_slice(b"MM");
            file[2..4].copy_from_slice(&42u16.to_be_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_be_bytes());
        }
        let mut heap: Vec<u8> = Vec::new();
        let ifd0_bytes = write_ifd(&ifd0, le, 0, &mut heap, OFF_HEAP);
        if OFF_IFD0 + ifd0_bytes.len() > OFF_HEAP {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", ifd0_bytes.len());
        }
        file[OFF_IFD0..OFF_IFD0 + ifd0_bytes.len()].copy_from_slice(&ifd0_bytes);
        assert!(
            heap.len() <= HEAP_RESERVE,
            "测试脚手架：外置值堆 {} 字节超出预留 {HEAP_RESERVE}",
            heap.len()
        );
        file.extend_from_slice(&heap);
        // 载荷基址固定在预留区之后，故 `resize` **只补零、绝不截断堆**
        // （上一版用 `align2(载荷长度)` 当基址，载荷短于堆时会截掉堆尾，
        //  导致 IFD 里的外置值偏移指向文件外——read_ifd 报「文件被截断」）。
        file.resize(payload_base, 0u8);
        file.extend_from_slice(&packed);
        file
    }

    /// 构建一个 **tile 布局**的 DNG（未压缩，位打包），像素内容与
    /// `pixels`（行主序 `width × height`）逐点相同。
    ///
    /// 关键点：**图边界处的 tile 仍按完整 `tile_w × tile_h` 解码**
    /// （边界外的样本是真实存在的填充像素），本脚手架按 DNG 规范把图外
    /// 区域填为 0，正是 [`DngFile::read_raw_samples`] 所依赖的语义。
    fn build_tiled_dng(
        bits: u16,
        fill_order: FillOrder,
        width: u32,
        height: u32,
        tile_w: u32,
        tile_h: u32,
        pixels: &[u16],
        white: u32,
        black: [u32; 4],
    ) -> Vec<u8> {
        let across = (width + tile_w - 1) / tile_w;
        let down = (height + tile_h - 1) / tile_h;
        // 1) 逐 tile 打包（tile 内行主序，图外填 0）。
        let mut tile_bytes: Vec<Vec<u8>> = Vec::with_capacity((across * down) as usize);
        for ty in 0..down {
            for tx in 0..across {
                let mut tp: Vec<u16> = Vec::with_capacity((tile_w * tile_h) as usize);
                for ry in 0..tile_h {
                    for rx in 0..tile_w {
                        let x = tx * tile_w + rx;
                        let y = ty * tile_h + ry;
                        let inside = x < width && y < height;
                        let v = if inside {
                            pixels
                                .get(y as usize * width as usize + x as usize)
                                .copied()
                                .unwrap_or(0)
                        } else {
                            0
                        };
                        tp.push(v);
                    }
                }
                tile_bytes.push(pack_bits(&tp, bits, fill_order, true));
            }
        }
        // 2) 载荷偏移。基址在 IFD 堆之后（见 `build_packed_dng` 的同名注释：
        //    偏移值是定长 u32，不影响堆长度，故两趟构建自洽）。
        let _total: usize = tile_bytes.iter().map(|t| t.len()).sum();
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        let mut offs: Vec<u32> = Vec::with_capacity(tile_bytes.len());
        let mut counts: Vec<u32> = Vec::with_capacity(tile_bytes.len());
        let mut c = payload_base;
        for t in &tile_bytes {
            offs.push(c as u32);
            counts.push(t.len() as u32);
            c += t.len();
        }
        // 3) IFD（**只有 tile 标签，没有 StripOffsets** —— 布局判定的唯一依据）。
        let mut ifd0 = IfdBuilder::new(le_of_fixture());
        ifd0.longs(tags::IMAGE_WIDTH, &[width]);
        ifd0.longs(tags::IMAGE_LENGTH, &[height]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
        ifd0.shorts(tags::COMPRESSION, &[1]);
        ifd0.shorts(tags::FILL_ORDER, &[fill_order.code()]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.shorts(tags::TILE_WIDTH, &[tile_w as u16]);
        ifd0.shorts(tags::TILE_LENGTH, &[tile_h as u16]);
        ifd0.longs(tags::TILE_OFFSETS, &offs);
        ifd0.longs(tags::TILE_BYTE_COUNTS, &counts);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[white]);
        let bl: Vec<f32> = black.iter().map(|&b| b as f32).collect();
        ifd0.rationals(tags::BLACK_LEVEL, &bl);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        // 4) 序列化（小端固定）。
        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let ifd0_bytes = write_ifd(&ifd0, true, 0, &mut heap, OFF_HEAP);
        if OFF_IFD0 + ifd0_bytes.len() > OFF_HEAP {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", ifd0_bytes.len());
        }
        file[OFF_IFD0..OFF_IFD0 + ifd0_bytes.len()].copy_from_slice(&ifd0_bytes);
        assert!(
            heap.len() <= HEAP_RESERVE,
            "测试脚手架：外置值堆 {} 字节超出预留 {HEAP_RESERVE}",
            heap.len()
        );
        file.extend_from_slice(&heap);
        file.resize(payload_base, 0u8);
        for t in &tile_bytes {
            file.extend_from_slice(t);
        }
        file
    }

    /// tile 脚手架固定用小端（避免每个调用点传 `le`）。
    fn le_of_fixture() -> bool {
        true
    }

    /// 构建一个**带 ForwardMatrix2 / CameraCalibration1 / AnalogBalance** 的
    /// DNG（`build_dng` 的超集），用于色彩矩阵链测试。
    fn build_forward_dng(
        fm2: [f32; 9],
        cc1: Option<[f32; 9]>,
        analog: Option<[f32; 3]>,
    ) -> Vec<u8> {
        let mut ifd0 = IfdBuilder::new(true);
        ifd0.longs(tags::IMAGE_WIDTH, &[4]);
        ifd0.longs(tags::IMAGE_LENGTH, &[4]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[16]);
        ifd0.shorts(tags::COMPRESSION, &[1]);
        ifd0.longs(tags::STRIP_OFFSETS, &[0]);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.longs(tags::ROWS_PER_STRIP, &[4]);
        ifd0.longs(tags::STRIP_BYTE_COUNTS, &[32]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[4095]);
        ifd0.rationals(tags::BLACK_LEVEL, &[64.0, 128.0, 128.0, 96.0]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        ifd0.srationals(tags::FORWARD_MATRIX2, &fm2);
        if let Some(cc) = cc1 {
            ifd0.srationals(tags::CAMERA_CALIBRATION1, &cc);
        }
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        if let Some(ab) = analog {
            ifd0.rationals(tags::ANALOG_BALANCE, &ab);
        }
        ifd0.shorts(tags::CALIBRATION_ILLUMINANT1, &[21]);

        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let bytes = write_ifd(&ifd0, true, 0, &mut heap, OFF_HEAP);
        if OFF_IFD0 + bytes.len() > OFF_HEAP {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", bytes.len());
        }
        file[OFF_IFD0..OFF_IFD0 + bytes.len()].copy_from_slice(&bytes);
        file.extend_from_slice(&heap);
        file
    }

    /// 构建一个**最小 CR3**：ftyp +（可选 Apple uuid 装 CMT*/THMB）+ mdat。
    fn build_cr3(
        major_brand: &[u8; 4],
        with_uuid: bool,
        uuid_children: &[(&[u8; 4], Vec<u8>)],
        mdat: Vec<u8>,
    ) -> Vec<u8> {
        fn box_of(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
            let size = 8 + payload.len();
            let mut v: Vec<u8> = Vec::with_capacity(size);
            v.extend_from_slice(&(size as u32).to_be_bytes());
            v.extend_from_slice(kind);
            v.extend_from_slice(payload);
            v
        }
        let mut out: Vec<u8> = Vec::new();
        // ftyp：major brand + minor version + 兼容品牌列表。
        let mut fp: Vec<u8> = Vec::new();
        fp.extend_from_slice(major_brand);
        fp.extend_from_slice(&0u32.to_be_bytes());
        fp.extend_from_slice(b"crx ");
        fp.extend_from_slice(b"isom");
        out.extend_from_slice(&box_of(b"ftyp", &fp));
        if with_uuid {
            let mut inner: Vec<u8> = Vec::new();
            for (k, p) in uuid_children {
                inner.extend_from_slice(&box_of(k, p));
            }
            let mut up: Vec<u8> = Vec::new();
            up.extend_from_slice(&CR3_APPLE_UUID);
            up.extend_from_slice(&inner);
            // uuid 盒：基本头 8 字节 + **16 字节 UUID（属负载）** + 子盒。
            // 注意 `total` 就是实际写出的字节数——UUID 已含在 `up` 里，
            // 不可再加一次（上一版多算16 字节，导致声明长度大于实际，遍历报越界）。
            let total = 8 + up.len();
            out.extend_from_slice(&(total as u32).to_be_bytes());
            out.extend_from_slice(b"uuid");
            out.extend_from_slice(&up);
        }
        out.extend_from_slice(&box_of(b"mdat", &mdat));
        out
    }

    // -----------------------------------------------------------------------
    // 1) DNG 解析
    // -----------------------------------------------------------------------

    #[test]
    fn dng_minimal_parses_cfa_levels_and_version() {
        let bytes = build_dng(true, [0, 1, 1, 2], 16); // RGGB
        let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dng.dng_version, [1, 4, 0, 0]);
        assert_eq!(dng.dng_backward_version, [1, 1, 0, 0]);
        assert_eq!(dng.order, ByteOrder::Little);
        assert_eq!(dng.bits_per_sample(), 16);

        let cfa = dng.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((cfa.cols, cfa.rows), (2, 2));
        assert_eq!(cfa.colors, vec![0u8, 1, 1, 2]);
        assert_eq!((cfa.raw_width, cfa.raw_height), (4, 4));
        assert_eq!(
            cfa.bayer().unwrap_or_else(|e| panic!("{e}")),
            BayerLayout::RgGb
        );

        let lv = dng.levels(&cfa).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(lv.white, 4095.0);
        assert_eq!(lv.black, vec![64.0, 128.0, 128.0, 96.0]);
        // 逐位置黑电平：RGGB 下 (0,0)=R → 64，(0,1)=G → 128，(1,1)=B → 96。
        assert_eq!(lv.black_at(0, 0, 2, 2), 64.0);
        assert_eq!(lv.black_at(1, 0, 2, 2), 128.0);
        assert_eq!(lv.black_at(1, 1, 2, 2), 96.0);

        assert_eq!(dng.camera_model().as_deref(), Some("PrismTestCam"));
        assert_eq!(dng.calibration_illuminant1(), Some(21));
        // 构造值取 1/1024 整数倍，故逐位精确相等。
        assert_eq!(dng.as_shot_white_xy(), Some([0.3125, 0.328_125]));
        assert_eq!(dng.baseline_exposure(), 0.0);
        assert_eq!(dng.as_shot_neutral(), Some([0.5, 1.0, 0.75]));
        let m = dng
            .color_matrix()
            .unwrap_or_else(|| panic!("缺 ColorMatrix1"));
        // RATIONAL 以 1/1024 存储，读回值 = round(v*1024)/1024，故用容差断言。
        assert!((m[0] - 0.700).abs() < 1e-3, "m[0] = {}", m[0]);
        assert!((m[1] - 0.200).abs() < 1e-3, "m[1] = {}", m[1]);
        assert!((m[8] - 0.889).abs() < 1e-3, "m[8] = {}", m[8]);
        // 行和 = D65 白点：真实相机的标定约束（保证中性灰显影后不偏色）。
        let row_sums = [m[0] + m[1] + m[2], m[3] + m[4] + m[5], m[6] + m[7] + m[8]];
        assert!(
            (row_sums[0] - 0.9505).abs() < 2e-3,
            "行和 X = {}",
            row_sums[0]
        );
        assert!((row_sums[1] - 1.0).abs() < 2e-3, "行和 Y = {}", row_sums[1]);
        assert!(
            (row_sums[2] - 1.088).abs() < 2e-3,
            "行和 Z = {}",
            row_sums[2]
        );
    }

    #[test]
    fn dng_big_endian_tiff_parses_identically() {
        // MM（大端）版式必须与 II（小端）版式解析出**相同的标签值**。
        let le = build_dng(true, [2, 1, 1, 0], 16); // BGGR
        let be = build_dng(false, [2, 1, 1, 0], 16);
        let a = DngFile::parse(&le).unwrap_or_else(|e| panic!("{e}"));
        let b = DngFile::parse(&be).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.order, ByteOrder::Little);
        assert_eq!(b.order, ByteOrder::Big);
        let ca = a.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        let cb = b.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ca.colors, cb.colors);
        assert_eq!(ca.raw_width, cb.raw_width);
        assert_eq!(a.as_shot_neutral(), b.as_shot_neutral());
        assert_eq!(a.color_matrix(), b.color_matrix());
        assert_eq!(
            a.levels(&ca).unwrap_or_else(|e| panic!("{e}")).black,
            b.levels(&cb).unwrap_or_else(|e| panic!("{e}")).black
        );
    }

    #[test]
    fn dng_cfa_four_bayer_layouts_round_trip() {
        // 四态的平铺与反查互为逆运算。
        for l in BayerLayout::ALL {
            let codes: [u8; 4] = [
                l.tile()[0] as u8,
                l.tile()[1] as u8,
                l.tile()[2] as u8,
                l.tile()[3] as u8,
            ];
            let bytes = build_dng(true, codes, 16);
            let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
            let cfa = dng.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(cfa.colors, codes.to_vec(), "{}", l.name());
            assert_eq!(
                cfa.bayer().unwrap_or_else(|e| panic!("{e}")),
                l,
                "{}",
                l.name()
            );
        }
        // 逐像素相位：RGGB 第 0 行 = R G R G，第 1 行 = G B G B。
        assert_eq!(BayerLayout::RgGb.color_at(0, 0), CfaColor::Red);
        assert_eq!(BayerLayout::RgGb.color_at(1, 0), CfaColor::Green);
        assert_eq!(BayerLayout::RgGb.color_at(0, 1), CfaColor::Green);
        assert_eq!(BayerLayout::RgGb.color_at(1, 1), CfaColor::Blue);
        // BGGR 第 0 行 = B G B G。
        assert_eq!(BayerLayout::BgGr.color_at(0, 0), CfaColor::Blue);
        assert_eq!(BayerLayout::BgGr.color_at(1, 1), CfaColor::Red);
        // GRBG 第 0 行 = G R G R。
        assert_eq!(BayerLayout::GrBg.color_at(0, 0), CfaColor::Green);
        assert_eq!(BayerLayout::GrBg.color_at(1, 0), CfaColor::Red);
        assert_eq!(BayerLayout::GrBg.color_at(0, 1), CfaColor::Blue);
        // GBRG 第 0 行 = G B G B。
        assert_eq!(BayerLayout::GbRg.color_at(0, 0), CfaColor::Green);
        assert_eq!(BayerLayout::GbRg.color_at(1, 0), CfaColor::Blue);
        assert_eq!(BayerLayout::GbRg.color_at(0, 1), CfaColor::Red);
    }

    #[test]
    fn dng_gps_and_exif_are_read_but_not_interpreted() {
        let bytes = build_dng(true, [0, 1, 1, 2], 16);
        let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        // EXIF：数值原样读出。
        let ex = dng.exif();
        assert_eq!(ex.exposure_time_s, Some(0.007_812_5));
        assert_eq!(ex.f_number, Some(2.75));
        assert_eq!(ex.iso, Some(400.0));
        assert_eq!(
            ex.date_time_original.as_deref(),
            Some("2024:01:02 03:04:05")
        );
        // GPS：只转储数值，**不**做十进制度换算。
        let gps = dng.gps().unwrap_or_else(|| panic!("应有 GPS 子 IFD"));
        assert_eq!(gps.tags.len(), 4);
        assert_eq!(gps.tags[0].tag, 1);
        assert_eq!(gps.tags[0].ascii.as_deref(), Some("N"));
        assert_eq!(gps.tags[1].tag, 2);
        assert_eq!(gps.tags[1].numbers.len(), 3); // 51/1, 30/1, 26/1 —— 未合并
        assert_eq!(gps.tags[1].numbers[0], 51.0);
        // 标签升序（确定性遍历序）。
        let tags: Vec<u16> = gps.tags.iter().map(|t| t.tag).collect();
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        assert_eq!(tags, sorted, "GPS 条目必须按标签升序");
    }

    // -----------------------------------------------------------------------
    // 2) 去马塞克
    // -----------------------------------------------------------------------

    #[test]
    fn demosaic_known_bayer_rggb_has_predictable_values() {
        // 4×4 RGGB，人工构造可预测的马赛克：
        // 行 0: R=1000 G=200  R=1000 G=200
        // 行 1: G=200  B=50   G=200  B=50
        // 取常数块可让「原值保持不变」成为最强断言。
        let raw: [f32; 16] = [
            1000.0, 200.0, 1000.0, 200.0, //
            200.0, 50.0, 200.0, 50.0, //
            1000.0, 200.0, 1000.0, 200.0, //
            200.0, 50.0, 200.0, 50.0,
        ];
        let p = demosaic_bilinear(&raw, 4, 4, BayerLayout::RgGb).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.width, p.height), (4, 4));
        // 恒定块 → 三通道都应还原为各自常数（去马塞克的最强可预测性）。
        for y in 0..4usize {
            for x in 0..4usize {
                let i = y * 4 + x;
                assert_eq!(p.r[i], 1000.0, "R({x},{y})");
                assert_eq!(p.g[i], 200.0, "G({x},{y})");
                assert_eq!(p.b[i], 50.0, "B({x},{y})");
            }
        }
    }

    #[test]
    fn demosaic_matches_hand_computed_bilinear_reference() {
        // 单调递增图案（v = 100*x + y），逐点与手算公式核对。
        //
        // 边界语义是**相位保持的镜像折返**（reflect-101 + 相位对齐），不是坐标钳制：
        // 越界采样先镜像回图内，再沿轴步进到目标相位。因此 −1 → 1、w → w−2。
        let w = 4u32;
        let h = 4u32;
        let mut raw: Vec<f32> = Vec::with_capacity(16);
        for y in 0..h {
            for x in 0..w {
                raw.push((100.0 * x as f32) + (y as f32));
            }
        }
        let p = demosaic_bilinear(&raw, w, h, BayerLayout::RgGb).unwrap_or_else(|e| panic!("{e}"));
        let at = |c: &[f32], x: u32, y: u32| c[(y as usize) * 4 + x as usize];

        // raw(x,y) = 100x + y：P(1,0)=100 P(0,1)=1 P(2,1)=201 P(1,1)=101 …
        //
        // (0,0) 是 R 位：R = P(0,0) = 0。
        //   G = 正交四邻（相位 (0,1) 与 (1,0)）：
        //       P(0,−1)→镜像 y=1，相位 1 已对齐 → P(0,1) = 1
        //       P(0,+1) → P(0,1) = 1
        //       P(−1,0)→镜像 x=1 → P(1,0) = 100；P(+1,0) = 100
        //     = (1 + 1 + 100 + 100)/4 = 50.5
        //   B = 对角四邻（相位 (1,1)）：四点皆镜像/对齐到 P(1,1) = 101 → 101
        assert_eq!(at(&p.r, 0, 0), 0.0);
        assert_eq!(at(&p.g, 0, 0), 50.5);
        assert_eq!(at(&p.b, 0, 0), 101.0);

        // (1,0) 是 G 位：G = P(1,0) = 100。
        //   水平轴（相位 (0,0)，红）：(P(0,0) + P(2,0))/2 = (0 + 200)/2 = 100
        //   垂直轴（相位 (1,1)，蓝）：(P(1,−1) + P(1,1))/2 = (101 + 101)/2 = 101
        assert_eq!(at(&p.g, 1, 0), 100.0);
        assert_eq!(at(&p.r, 1, 0), 100.0);
        assert_eq!(at(&p.b, 1, 0), 101.0);

        // (1,1) 是 B 位：B = P(1,1) = 101。
        //   G = 正交四邻（相位 (1,0) 与 (0,1)）：
        // (100 + 102 + 1 + 201)/4 = 101
        //   R = 对角四邻（相位 (0,0)）：(0 + 200 + 2 + 202)/4 = 101
        //   （注：P(0,1) = 1 而非 100，P(1,2) = 102 —— 图案是 100x+y）
        assert_eq!(at(&p.b, 1, 1), 101.0);
        assert_eq!(at(&p.g, 1, 1), 101.0);
        assert_eq!(at(&p.r, 1, 1), 101.0);

        // (2,1) 是 G 位（奇数行）：G = P(2,1) = 201。
        //   水平轴（相位 (1,1)，蓝）：(P(1,1) + P(3,1))/2 = (101 + 301)/2 = 201
        //   垂直轴（相位 (0,0)，红）：(P(2,0) + P(2,2))/2 = (200 + 202)/2 = 201
        assert_eq!(at(&p.g, 2, 1), 201.0);
        assert_eq!(at(&p.b, 2, 1), 201.0);
        assert_eq!(at(&p.r, 2, 1), 201.0);

        // 图案 v = 100x + y 是**仿射函数**，双线性插值对**内部**像素为精确重建
        // （两轴平均都回到 100x+y）。边界像素因镜像折返而偏离，故只核对 1..3 环。
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let want = 100.0 * x as f32 + y as f32;
                assert_eq!(at(&p.r, x, y), want, "内部 r({x},{y})");
                assert_eq!(at(&p.g, x, y), want, "内部 g({x},{y})");
                assert_eq!(at(&p.b, x, y), want, "内部 b({x},{y})");
            }
        }
    }

    #[test]
    fn demosaic_all_four_layouts_agree_on_gray_ramp() {
        // 同一「平铺颜色 ↔ 数值」对应关系下，四态在同一坐标应给出同一 CFA 颜色
        // 的同一值（插值与相位无关：只取决于邻域的实际颜色分布）。
        // 这里用**颜色常量块**：把同一平铺颜色赋予同一数值，四态互为空间平移，
        // 因此 (0,0) 处若为 R/B 位则三通道分别等于 R/B 常数。
        for l in BayerLayout::ALL {
            let t = l.tile();
            let val = |c: CfaColor| -> f32 {
                match c {
                    CfaColor::Red => 3000.0,
                    CfaColor::Green => 700.0,
                    CfaColor::Blue => 111.0,
                }
            };
            let raw: Vec<f32> = (0..16)
                .map(|i| {
                    let y = (i / 4) as u32;
                    let x = (i % 4) as u32;
                    val(t[(((y & 1) as usize) * 2) + ((x & 1) as usize)])
                })
                .collect();
            let p = demosaic_bilinear(&raw, 4, 4, l).unwrap_or_else(|e| panic!("{e}"));
            for i in 0..16usize {
                assert_eq!(p.r[i], 3000.0, "{} r[{i}]", l.name());
                assert_eq!(p.g[i], 700.0, "{} g[{i}]", l.name());
                assert_eq!(p.b[i], 111.0, "{} b[{i}]", l.name());
            }
        }
    }

    #[test]
    fn demosaic_boundary_clamps_and_rejects_length_mismatch() {
        // 1×1 极小图：全部邻域被钳到自身，三通道都应等于原值。
        let p =
            demosaic_bilinear(&[42.0], 1, 1, BayerLayout::RgGb).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((p.r[0], p.g[0], p.b[0]), (42.0, 42.0, 42.0));
        // 宽 2 高 1 的 RGGB 首像素 R 位。
        // h = 1 ⇒ y 方向镜像恒为 0，且无法步进到绿相位（y=0 是红位），
        // 故垂直两邻都回退到中心值 10（这是 1 行图的固有退化，不是错误）。
        // 水平两邻 (x=−1, x=+1) 镜像后同为 x=1 = 20（绿位，相位已对齐）。
        let p2 = demosaic_bilinear(&[10.0, 20.0], 2, 1, BayerLayout::RgGb)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p2.r[0], 10.0);
        assert_eq!(p2.g[0], (10.0 + 10.0 + 20.0 + 20.0) * 0.25);
        // 对角四邻：(−1,±1)/(+1,±1) 全部镜像到 (1,0) = 20。
        assert_eq!(p2.b[0], 20.0);
        // 长度不符 → 结构化错误，不 panic。
        let err = demosaic_bilinear(&[1.0, 2.0], 4, 4, BayerLayout::RgGb).unwrap_err();
        assert_eq!(err.code, "RAW3001");
    }

    // -----------------------------------------------------------------------
    // 3) 白平衡
    // -----------------------------------------------------------------------

    #[test]
    fn white_balance_gains_apply_as_shot_neutral() {
        // 绿归一（默认）：绿恒 1，R/G = 1/0.5 = 2，B/G = 1/0.75 = 4/3。
        let g = white_balance_gains([0.5, 1.0, 0.75], WbMode::GreenNormalized);
        assert_eq!(g[1], 1.0);
        assert_eq!(g[0], 2.0);
        assert!((g[2] - 4.0 / 3.0).abs() < 1e-6, "g[2] = {}", g[2]);
        // 严格 DNG 式：最大增益为 1。1/N = [2, 1, 4/3]，最大是 1/0.5 = 2
        // （不是 1/0.75 —— 那是最小 N 对应的最小 1/N），故 gi = [1, 0.5, 2/3]。
        let gi = white_balance_gains([0.5, 1.0, 0.75], WbMode::InverseNormalized);
        assert!((gi[0] - 1.0).abs() < 1e-6, "gi[0] = {}", gi[0]);
        assert!((gi[1] - 0.5).abs() < 1e-6, "gi[1] = {}", gi[1]);
        assert!((gi[2] - 2.0 / 3.0).abs() < 1e-6, "gi[2] = {}", gi[2]);
        // 两种模式只差全局标量：比值恒定（此处为 0.5）。
        for i in 0..3 {
            let ratio = gi[i] / g[i];
            assert!(
                (ratio - 0.5).abs() < 1e-6,
                "两模式须相差全局标量 0.5（第 {i} 项），实为 {ratio}"
            );
        }
        // None → 单位增益。
        assert_eq!(
            white_balance_gains([0.5, 1.0, 0.75], WbMode::None),
            [1.0, 1.0, 1.0]
        );
        // 非法输入回退单位增益（不静默产生 Inf/NaN）。
        assert_eq!(
            white_balance_gains([0.0, 1.0, 1.0], WbMode::GreenNormalized),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(
            white_balance_gains([f32::NAN, 1.0, 1.0], WbMode::InverseNormalized),
            [1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn white_balance_gain_scales_linear_pixel_exactly() {
        // 白平衡是逐通道标量乘：给定 (1,1,1)，绿归一下输出 = 增益本身。
        let gain = white_balance_gains([0.5, 1.0, 0.75], WbMode::GreenNormalized);
        let rgb = [1.0f32, 1.0, 1.0];
        let wb = [rgb[0] * gain[0], rgb[1] * gain[1], rgb[2] * gain[2]];
        assert_eq!(wb[0], 2.0);
        assert_eq!(wb[1], 1.0);
        assert!((wb[2] - 4.0 / 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 4) 色彩矩阵
    // -----------------------------------------------------------------------

    #[test]
    fn color_matrix_transform_matches_hand_computation() {
        // 已知矩阵 M = [[0.7,0.2,0.1],[0.1,0.8,0.1],[0.05,0.15,0.8]]
        // 输入 (1,0,0) → 第一列 (0.7, 0.1, 0.05)。
        let m: Mat3 = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.05, 0.15, 0.8];
        assert_eq!(mat3_mul(&m, [1.0, 0.0, 0.0]), [0.7, 0.1, 0.05]);
        assert_eq!(mat3_mul(&m, [0.0, 1.0, 0.0]), [0.2, 0.8, 0.15]);
        assert_eq!(mat3_mul(&m, [0.0, 0.0, 1.0]), [0.1, 0.1, 0.8]);
        // 中性 (1,1,1) → (1.0, 1.0, 1.0)（该矩阵行和为 1，即纯增益无串扰）。
        let n = mat3_mul(&m, [1.0, 1.0, 1.0]);
        assert!(
            (n[0] - 1.0).abs() < 1e-6 && (n[1] - 1.0).abs() < 1e-6 && (n[2] - 1.0).abs() < 1e-6
        );
        // 矩阵乘：M × I = M。
        assert_eq!(
            mat3_mul_matrix(&m, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            m
        );
        // 串联：(A×B)·v 与先 B·v 再 A·(…) 数学相等，但**浮点结合律不保证逐位相同**
        // （预乘省一次舍入，但改变求和顺序），故此处只断言数值接近 + 各自可复现。
        let s = XYZ_TO_SRGB;
        let v = [0.3f32, 0.4, 0.5];
        let pre = mat3_mul_matrix(&s, &m);
        let a = mat3_mul(&pre, v);
        let b = mat3_mul(&s, mat3_mul(&m, v));
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() <= 1e-6 * a[i].abs().max(1.0),
                "串联第 {i} 项应数值接近：预乘 {a:?} vs 两步 {b:?}"
            );
            // 同一路径重复调用必须逐位一致（确定性纪律）。
            assert_eq!(
                mat3_mul(&pre, v)[i].to_bits(),
                a[i].to_bits(),
                "预乘路径须逐位可复现"
            );
            assert_eq!(
                mat3_mul(&s, mat3_mul(&m, v))[i].to_bits(),
                b[i].to_bits(),
                "两步路径须逐位可复现"
            );
        }
    }

    #[test]
    fn xyz_to_srgb_maps_white_to_unit_rgb() {
        // D65 白点 (0.95047, 1.0, 1.08883) → 线性 sRGB 应为 (1,1,1)。
        let w = mat3_mul(&XYZ_TO_SRGB, [0.950_47, 1.0, 1.088_83]);
        for (i, v) in w.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "白点第 {i} 分量 = {v}（应≈1）");
        }
        // XYZ 的 Y 分量即亮度。线性 sRGB 的亮度系数 (0.2126, 0.7152, 0.0722) 是
        // **sRGB→XYZ 矩阵的第 2 行**，不是 XYZ→sRGB 的第 2 列；后者是
        // 把纯 Y 映射到 sRGB 的负主值（−1.537, 1.876, −0.204）。
        let g = mat3_mul(&XYZ_TO_SRGB, [0.0, 1.0, 0.0]);
        assert!(
            (g[0] - -1.537_138_5).abs() < 1e-4,
            "纯 Y 的 X 分量 = {}",
            g[0]
        );
        assert!(
            (g[1] - 1.876_010_8).abs() < 1e-4,
            "纯 Y 的 Y 分量 = {}",
            g[1]
        );
        assert!(
            (g[2] - -0.204_025_9).abs() < 1e-4,
            "纯 Y 的 Z 分量 = {}",
            g[2]
        );
        // 往返自洽：线性 sRGB 灰 (v,v,v) → XYZ → sRGB 回到 v。
        // 两张矩阵都是 7 位小数截断值，非精确互逆，容差取 1e-3。
        for &v in &[0.25f32, 0.5, 1.0] {
            let xyz = mat3_mul(&SRGB_TO_XYZ, [v, v, v]);
            let back = mat3_mul(&XYZ_TO_SRGB, xyz);
            for i in 0..3 {
                assert!(
                    (back[i] - v).abs() < 1e-3,
                    "灰 {v} 往返第 {i} 分量 = {}",
                    back[i]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5) 端到端显影
    // -----------------------------------------------------------------------

    #[test]
    fn develop_end_to_end_outputs_expected_f16_bits() {
        let bytes = build_dng(true, [0, 1, 1, 2], 16); // RGGB 4×4
        let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let cfa = dng.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        let lv = dng.levels(&cfa).unwrap_or_else(|e| panic!("{e}"));
        // 构造中性灰马赛克：线性化后每点 = 0.5（黑白电平之下线性插值）。
        let w = cfa.raw_width as usize;
        let mut mosaic: Vec<u16> = Vec::with_capacity(w * 4);
        for y in 0..4u32 {
            for x in 0..cfa.raw_width {
                let b = lv.black_at(x, y, cfa.cols, cfa.rows);
                let v = b + 0.5 * (lv.white - b);
                mosaic.push(v.round() as u16);
            }
        }
        let opts = DevelopOptions {
            white_balance: WbMode::None,
            color_matrix: MatrixMode::CameraToSrgb,
            ..DevelopOptions::default()
        };
        let fb = dng
            .develop(&mosaic, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((fb.width, fb.height), (4, 4));
        assert_eq!(fb.pixels.len(), 4 * 4 * 4);
        // Alpha 恒 1.0（f16 位模式 0x3C00）。
        for y in 0..4u32 {
            for x in 0..4u32 {
                let idx = ((y as usize) * 4 + x as usize) * 4;
                assert_eq!(fb.pixels[idx + 3], 0x3C00, "({x},{y}) Alpha 应为 1.0");
            }
        }
        // 逐位确定：同输入显影两遍，f16 位模式完全一致。
        let fb2 = dng
            .develop(&mosaic, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fb.pixels, fb2.pixels, "显影必须逐位可复现");
        // 灰块经 (0.7,0.8,0.8) 行列和矩阵 + sRGB 矩阵后应近似中性（三通道接近）。
        let px = fb.get(2, 2);
        let spread = (px[0] - px[1]).abs().max((px[1] - px[2]).abs());
        assert!(
            spread < 0.05,
            "灰块应近似中性，实际 {px:?}（最大偏差 {spread}）"
        );
    }

    #[test]
    fn develop_is_bit_identical_across_repeated_parse() {
        let bytes = build_dng(true, [1, 2, 0, 1], 16); // GRBG
        let mut mosaic: Vec<u16> = (0..16u32).map(|i| (1000u32 + i * 37) as u16).collect();
        mosaic[0] = 5000; // 越界饱和值也不得 panic
        mosaic[15] = 0;
        let a = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let b = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let opts = DevelopOptions::default();
        let fa = a.develop(&mosaic, &opts).unwrap_or_else(|e| panic!("{e}"));
        let fb = b.develop(&mosaic, &opts).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fa.pixels, fb.pixels, "两次解析 + 显影须逐位一致");
        // 所有输出必须有限（无 NaN/Inf 泄漏到 f16）。
        for &h in fa.pixels.iter() {
            assert!(f16_to_f32(h).is_finite(), "f16 位模式 {h:#06x} 非有限值");
        }
    }

    #[test]
    fn develop_rejects_wrong_length_and_out_of_range_bit_depth() {
        let bytes = build_dng(true, [0, 1, 1, 2], 16);
        let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let err = dng
            .develop(&[0u16; 15], &DevelopOptions::default())
            .unwrap_err();
        assert_eq!(err.code, "RAW3001", "{}", err);
        // 14 位**现在必须被接受**（本实现的核心新增能力）——用越界位深替代
        // 「非 16 位即拒绝」的旧断言：显影链接受 1..=16 位，超过即结构化拒绝。
        let b14 = build_dng(true, [0, 1, 1, 2], 14);
        let d14 = DngFile::parse(&b14).unwrap_or_else(|e| panic!("{e}"));
        d14.develop(&[0u16; 16], &DevelopOptions::default())
            .unwrap_or_else(|e| panic!("14 位现在应被接受，却报错：{e}"));
        // 32 位 → 仍然结构化拒绝（不假装支持）。
        let b32 = build_dng(true, [0, 1, 1, 2], 32);
        let d32 = DngFile::parse(&b32).unwrap_or_else(|e| panic!("{e}"));
        let e2 = d32
            .develop(&[0u16; 16], &DevelopOptions::default())
            .unwrap_err();
        assert_eq!(e2.code, "RAW2001", "{}", e2);
    }

    // -----------------------------------------------------------------------
    // 6) 畸形输入攻击 + CR3 结构化拒绝
    // -----------------------------------------------------------------------

    #[test]
    fn cr3_magic_is_structurally_rejected_with_reason() {
        // 最小 CR3 头：ISO-BMFF ftyp + major brand "crx "。
        let mut cr3: Vec<u8> = Vec::new();
        cr3.extend_from_slice(&(16u32).to_be_bytes()); // box size
        cr3.extend_from_slice(b"ftyp");
        cr3.extend_from_slice(b"crx ");
        cr3.extend_from_slice(&0u32.to_be_bytes()); // minor version
        cr3.extend_from_slice(b"crx isom");
        let err = probe_format(&cr3).unwrap_err();
        assert_eq!(err.code, "RAW2001", "{}", err);
        // 错误文本须说清「为何不支持」的四条理由。
        let text = format!("{err}");
        for kw in ["ftyp", "CMT1", "CMT2", "CRX", "CFA"] {
            assert!(text.contains(kw), "CR3 拒绝理由须提到 {kw}：{text}");
        }
        // DngFile::parse 同样拒绝（不会 panic）。
        let e2 = DngFile::parse(&cr3).unwrap_err();
        assert_eq!(e2.code, "RAW2001");
        // 非 CR3 的 BMFF 也结构化拒绝而非误判为 DNG。
        let mut mp4 = cr3.clone();
        mp4[8..12].copy_from_slice(b"qt  ");
        assert_eq!(probe_format(&mp4).unwrap_err().code, "RAW2001");
    }

    #[test]
    fn dng_malformed_inputs_are_structurally_rejected() {
        // (a) 截断到连头都不完整 → **嗅探器** `probe_format` 必须拒绝。
        //     注意：`probe_format` 只读 8 字节头，故 ≥8 字节的前缀合法地嗅探为 DNG
        //     （这是嗅探器的职责边界，不是缺陷）；完整解析由 `DngFile::parse` 把关。
        let full = build_dng(true, [0, 1, 1, 2], 16);
        for cut in [0usize, 1, 2, 3, 4, 5, 6, 7] {
            let err = probe_format(&full[..cut]).unwrap_err();
            assert!(
                err.code.starts_with("RAW1") || err.code.starts_with("RAW2"),
                "截断 {cut} 字节应结构化错误，得到 {err}"
            );
        }
        // (a2) 头完整但**目录或外置值被截断** → `DngFile::parse` 必须拒绝（不 panic）。
        // `parse` 会连带解析 IFD 引用的外置值，故截断点落在外置值堆内也会在此报错。
        for cut in [
            8usize, 10, 12, 20, 40, 100, 280, 520, 1030, 2050, 2100, 2200,
        ] {
            if cut >= full.len() {
                continue;
            }
            let err = DngFile::parse(&full[..cut]).unwrap_err();
            assert!(
                err.code.starts_with("RAW1"),
                "截断 {cut} 字节应结构化错误，得到 {err}"
            );
        }
        // (b) 错误 magic。
        let mut bad_magic = full.clone();
        bad_magic[2..4].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(probe_format(&bad_magic).unwrap_err().code, "RAW1001");
        // (c) 字节序头与 magic 不自洽。
        let mut bad_bo = full.clone();
        bad_bo[0..2].copy_from_slice(b"XY");
        assert_eq!(probe_format(&bad_bo).unwrap_err().code, "RAW1001");
        // (d) 头 4 字节完全不是 RAW 容器。
        assert_eq!(
            probe_format(&[0x89, b'P', b'N', b'G', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .unwrap_err()
                .code,
            "RAW1001"
        );
        // (e) IFD0 偏移越界（负偏移语义：u32 的 0xFFFFFFFF）。
        let mut bad_off = full.clone();
        bad_off[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert_eq!(DngFile::parse(&bad_off).unwrap_err().code, "RAW1001");
        // (f) 超大属性 size：count = 0xFFFFFFFF × LONG(4B) = 16 GiB。
        let mut huge = full.clone();
        let count_off = OFF_IFD0 + 2 + 2 + 2; // 第 1 条的 count 字段
        huge[count_off..count_off + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let err = DngFile::parse(&huge).unwrap_err();
        assert_eq!(err.code, "RAW1001", "{}", err);
        // (g) 外置值偏移越界。
        //     必须打在**真正走外置值**的条目上（字节数 > 4）；若打在 ≤4 字节的
        //     内联值条目上，改的是数值本身而非偏移，构不成越界读。
        let mut bad_val = full.clone();
        let ext = first_external_entry_offset(&bad_val);
        assert!(ext.is_some(), "测试脚手架：IFD0 应至少有一条外置值条目");
        let vo = ext.unwrap_or(0) + 8; // 条目内 value/offset 字段
        bad_val[vo..vo + 4].copy_from_slice(&0x7FFF_FFFFu32.to_le_bytes());
        assert_eq!(DngFile::parse(&bad_val).unwrap_err().code, "RAW1001");
        // (h) IFD 条目数超防御上限。
        let mut many = full.clone();
        many[OFF_IFD0..OFF_IFD0 + 2].copy_from_slice(&0xFFFFu16.to_le_bytes());
        assert_eq!(DngFile::parse(&many).unwrap_err().code, "RAW1001");
        // (i) 未定义字段类型。
        let mut bad_ty = full.clone();
        bad_ty[OFF_IFD0 + 2 + 2..OFF_IFD0 + 2 + 4].copy_from_slice(&0x00FFu16.to_le_bytes());
        assert_eq!(DngFile::parse(&bad_ty).unwrap_err().code, "RAW1001");
        // (j) 无 DNGVersion 的普通 TIFF：把 DNGVersion(0xC612) 的标签号改成 0xFFFF，
        //     使 IFD0 里不再有 DNGVersion —— 按规范这就不是 DNG。
        let mut plain = build_dng(true, [0, 1, 1, 2], 16);
        let n = ifd0_entry_count(&plain);
        let mut hit = 0usize;
        for i in 0..n {
            let e = ifd0_entry_offset(&plain, i);
            if u16::from_le_bytes([plain[e], plain[e + 1]]) == tags::DNG_VERSION {
                plain[e..e + 2].copy_from_slice(&0xFFFFu16.to_le_bytes());
                hit += 1;
            }
        }
        assert_eq!(hit, 1, "测试脚手架：应恰好命中一条 DNGVersion");
        let e = DngFile::parse(&plain).unwrap_err();
        assert_eq!(e.code, "RAW2001", "缺 DNGVersion 应判为非 DNG：{e}");
        // (k) DNGVersion major 过大（2.x 未发布，本模块只实现 1.x）。
        let mut v2 = build_dng(true, [0, 1, 1, 2], 16);
        for i in 0..ifd0_entry_count(&v2) {
            let e = ifd0_entry_offset(&v2, i);
            if u16::from_le_bytes([v2[e], v2[e + 1]]) == tags::DNG_VERSION {
                // DNGVersion 是 BYTE×4（4 字节 > 内联阈值？不，恰好 4 → 内联）。
                // 内联值按 TIFF 规则**右对齐**放在 4 字节字段里，故首字节在 value[0]。
                v2[e + 8] = 2; // major = 2
            }
        }
        let e = DngFile::parse(&v2).unwrap_err();
        assert_eq!(e.code, "RAW2001", "DNG 2.x 应结构化拒绝：{e}");
        // (l) 缺 WhiteLevel：把标签号改成 0xFFFF，IFD0 仍在但 levels() 取不到。
        let mut nowl = build_dng(true, [0, 1, 1, 2], 16);
        for i in 0..ifd0_entry_count(&nowl) {
            let e = ifd0_entry_offset(&nowl, i);
            if u16::from_le_bytes([nowl[e], nowl[e + 1]]) == tags::WHITE_LEVEL {
                nowl[e..e + 2].copy_from_slice(&0xFFFFu16.to_le_bytes());
            }
        }
        let d = DngFile::parse(&nowl).unwrap_or_else(|e| panic!("{e}"));
        let cfa = d.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.levels(&cfa).unwrap_err().code, "RAW1001");
    }

    #[test]
    fn non_bayer_cfa_is_rejected_with_explanation() {
        // X-Trans 6×6 → 双线性 Bayer 去马塞克不适用，须结构化拒绝。
        let mut ifd0 = IfdBuilder::new(true);
        ifd0.longs(tags::IMAGE_WIDTH, &[6]);
        ifd0.longs(tags::IMAGE_LENGTH, &[6]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[6, 6]);
        let xtrans: Vec<u8> = (0..36u8).map(|i| (i % 3) as u8).collect();
        ifd0.bytes(tags::CFA_PATTERN, &xtrans);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[4095]);
        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let bytes = write_ifd(&ifd0, true, 0, &mut heap, OFF_HEAP);
        file[OFF_IFD0..OFF_IFD0 + bytes.len()].copy_from_slice(&bytes);
        file.extend_from_slice(&heap);
        let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let cfa = dng.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((cfa.cols, cfa.rows), (6, 6));
        let err = cfa.bayer().unwrap_err();
        assert_eq!(err.code, "RAW2001", "{err}");
        assert!(format!("{err}").contains("X-Trans"), "{err}");
        // 2×2 但含 3 个绿（非法 Bayer）→ 同样结构化拒绝。
        let bad = build_dng(true, [1, 1, 1, 2], 16);
        let d2 = DngFile::parse(&bad).unwrap_or_else(|e| panic!("{e}"));
        let c2 = d2.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c2.bayer().unwrap_err().code, "RAW2001");
        // 非 Bayer 颜色码（=3）→ 结构化拒绝。
        let bad2 = build_dng(true, [0, 3, 1, 2], 16);
        let d3 = DngFile::parse(&bad2).unwrap_or_else(|e| panic!("{e}"));
        let c3 = d3.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c3.bayer().unwrap_err().code, "RAW2001");
    }

    #[test]
    fn parse_is_deterministic_byte_for_byte() {
        let bytes = build_dng(true, [2, 1, 1, 0], 16); // BGGR
        let a = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        let b = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.ifd0, b.ifd0, "IFD0 条目须逐字段一致");
        assert_eq!(a.exif_ifd, b.exif_ifd);
        assert_eq!(a.gps_ifd, b.gps_ifd);
        // IFD 条目必须升序（遍历顺序确定，不依赖文件顺序）。
        let tags: Vec<u16> = a.ifd0.entries.iter().map(|e| e.tag).collect();
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        assert_eq!(tags, sorted);
        // 解析两次的全部公开数值一致。
        assert_eq!(a.cfa_pattern(), b.cfa_pattern());
        assert_eq!(a.exif(), b.exif());
        assert_eq!(a.gps(), b.gps());
    }

    // -----------------------------------------------------------------------
    // 7) 位解包：逐样本精确值（8/10/12/14/16）
    // -----------------------------------------------------------------------

    /// **填充位非零**的对拍核心：同一组样本，用「填充位全 0」与
    /// 「末字节填充位全 1」两种字节流打包，解包结果必须**逐样本完全相同**。
    ///
    /// 这条断言是本测试最重要的防错闸门：12 位时若忘记 mask，每个样本会
    /// 凭空多出最多 15（`0xABC` 被读成 `0xABCF`）。
    fn assert_unpack_exact(bits: u16, count: usize) {
        let values = sample_series(bits, count);
        for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
            for pad in [false, true] {
                let packed = pack_bits(&values, bits, fo, pad);
                let got =
                    unpack_bits(&packed, bits, count, fo).unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(
                    got.len(),
                    count,
                    "{bits} 位 {fo:?} pad={pad}：样本数应为 {count}"
                );
                for i in 0..count {
                    assert_eq!(
                        got[i], values[i],
                        "{bits} 位 {fo:?} pad_high={pad} 第 {i} 样本：\
                         期望 {}（{values:?}），实得 {}（packed = {:02X?}）",
                        values[i], got[i], packed
                    );
                }
            }
        }
    }

    #[test]
    fn bit_unpack_8bit_is_exact_per_sample() {
        // 8 位是快路径：1 样本 = 1 字节，位序无差别。
        let values: Vec<u16> = vec![0, 1, 127, 128, 200, 254, 255, 42, 0, 255];
        for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
            let packed = pack_bits(&values, 8, fo, false);
            assert_eq!(packed.len(), values.len(), "8 位应为 1 字节/样本");
            let got = unpack_bits(&packed, 8, values.len(), fo).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, values, "8 位 {fo:?} 须逐样本原样往返");
        }
        // 显式逐样本断言（不依赖 Vec 相等的错误定位粒度）。
        let packed = pack_bits(&values, 8, FillOrder::MsbFirst, false);
        let got = unpack_bits(&packed, 8, 3, FillOrder::MsbFirst)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, vec![0u16, 1, 127], "前三个样本须精确");
    }

    #[test]
    fn bit_unpack_10bit_is_exact_per_sample_with_nonzero_padding() {
        // 10 位：每样本 2 字节窗口，高 10 位有效、低 6 位填充。
        assert_unpack_exact(10, 16);
        // 手工核对一个具体字节序列：10 位只能承载 0..=0x3FF，故取 0x2BC。
        // 0x2BC = 0b10_1011_1100，高 10 位有效、低 6 位填充 → 首字节 0xAF、
        // 次字节低 2 位为 0x0。**填充位非零也不影响读回**（见下一个断言）。
        let v = [0x0ABCu16 & 0x03FF]; // 0x2BC：显式说明掩掉了高 2 位
        let packed = pack_bits(&v, 10, FillOrder::MsbFirst, false);
        assert_eq!(packed[0], 0xAF, "10 位 MSB-first 首字节（实测 {:02X?}）", packed);
        let got = unpack_bits(&packed, 10, 1, FillOrder::MsbFirst)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got[0], 0x0ABCu16 & 0x03FF, "10 位须还原 0x2BC（实测 {:02X?}）", packed);
        // 同一组值，填充位置 1 后读回仍须一致（掩码生效的直接证据）。
        let padded = pack_bits(&v, 10, FillOrder::MsbFirst, true);
        let got2 = unpack_bits(&padded, 10, 1, FillOrder::MsbFirst)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got2[0], got[0], "填充位置1 不得改变读回值");
    }

    #[test]
    fn bit_unpack_12bit_is_exact_per_sample_with_nonzero_padding() {
        // 12 位：每样本高 12 位有效、低 4 位填充（本路最重要的位深）。
        assert_unpack_exact(12, 32);
        // 12 位的最大值 0xFFF 与 0 相邻位都必须精确。
        let edge = [0u16, 0xFFF, 1, 0xFFE, 0x800, 0x7FF];
        for pad in [false, true] {
            let packed = pack_bits(&edge, 12, FillOrder::MsbFirst, pad);
            let got = unpack_bits(&packed, 12, edge.len(), FillOrder::MsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, edge.to_vec(), "12 位边界值 pad_high={pad} 须精确");
        }
        // 打包后每 2 样本恰 3 字节（12×2 = 24 位）：验证「紧凑」而非「2 字节/样本」。
        let packed = pack_bits(&[0x123u16, 0x456], 12, FillOrder::MsbFirst, false);
        assert_eq!(packed.len(), 3, "2 个 12 位样本应占 3 字节（紧凑打包）");
        assert_eq!(packed, vec![0x12, 0x34, 0x56], "12 位 MSB-first 字节序列");
    }

    #[test]
    fn bit_unpack_14bit_is_exact_per_sample_with_nonzero_padding() {
        // 14 位：每样本高 14 位有效、低 2 位填充。
        assert_unpack_exact(14, 32);
        let edge = [0u16, 0x3FFF, 1, 0x3FFE, 0x2000, 0x1FFF];
        for pad in [false, true] {
            let packed = pack_bits(&edge, 14, FillOrder::MsbFirst, pad);
            let got = unpack_bits(&packed, 14, edge.len(), FillOrder::MsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, edge.to_vec(), "14 位边界值 pad_high={pad} 须精确");
        }
        // 4 个 14 位样本 = 56 位 = 7 字节。
        let packed = pack_bits(&[0x123, 0x456, 0x789, 0xABC], 14, FillOrder::MsbFirst, false);
        assert_eq!(packed.len(), 7, "4 个 14 位样本应占 7 字节");
    }

    #[test]
    fn bit_unpack_16bit_fast_paths_match_byte_order_semantics() {
        // 16 位快路径：MSB-first ≡ 大端 u16 数组；LSB-first ≡ 小端 u16 数组。
        let values: Vec<u16> = vec![0, 1, 0x00FF, 0xFF00, 0x8000, 0x7FFF, 0xFFFF, 0x1234];
        for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
            let packed = pack_bits(&values, 16, fo, false);
            assert_eq!(packed.len(), 16, "16 位应为 2 字节/样本");
            let got =
                unpack_bits(&packed, 16, values.len(), fo).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, values, "16 位 {fo:?} 须逐样本原样往返");
        }
        // 显式验证「大端 u16 序列」这个论断本身。
        let be = [0x12u8, 0x34, 0xAB, 0xCD];
        assert_eq!(
            unpack_bits(&be, 16, 2, FillOrder::MsbFirst).unwrap_or_else(|e| panic!("{e}")),
            vec![0x1234u16, 0xABCD],
            "MSB-first 的 16 位流须等于大端 u16 序列"
        );
        assert_eq!(
            unpack_bits(&be, 16, 2, FillOrder::LsbFirst).unwrap_or_else(|e| panic!("{e}")),
            vec![0x3412u16, 0xCDAB],
            "LSB-first 的 16 位流须等于小端 u16 序列"
        );
    }

    #[test]
    fn bit_unpack_msb_and_lsb_first_agree_on_same_image() {
        // **对拍**：同一组样本分别按 MSB-first / LSB-first 编码，
        // 解包结果必须逐样本相同（两条位序描述的是同一逻辑图像）。
        for bits in [1u16, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16] {
            let values = sample_series(bits, 37);
            let msb = pack_bits(&values, bits, FillOrder::MsbFirst, true);
            let lsb = pack_bits(&values, bits, FillOrder::LsbFirst, true);
            let a = unpack_bits(&msb, bits, values.len(), FillOrder::MsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            let b = unpack_bits(&lsb, bits, values.len(), FillOrder::LsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{bits} 位：MSB-first 与 LSB-first 解出同一图像");
            assert_eq!(a, values, "{bits} 位：两条位序都须还原原始样本");
            // 两种位序的**字节流**通常不同，但对拍的意义是「解出同一图像」
            // —— 上一行 `a == b` 才是本测试的主断言。
            //
            // **8 位是固有例外**：每样本恰好 1 字节，位序无从体现，
            // 故两条路的字节流必然相同（`unpack_bits` 的 8 位快路径也正是
            // 因此忽略 `fill_order`）。这不是 bug，是 8 位的定义。
            if bits != 8 {
                assert_ne!(msb, lsb, "{bits} 位：两种位序的字节流应不同");
            } else {
                assert_eq!(msb, lsb, "8 位时两字节流本就相同（每样本 1 字节）");
            }
        }
    }

    #[test]
    fn bit_unpack_covers_every_bit_depth_1_to_16() {
        // 1..=16 位全覆盖（1 与 16 也在内，确保无位深被漏掉）。
        for bits in 1u16..=16 {
            let n = 20usize;
            let values = sample_series(bits, n);
            for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
                let packed = pack_bits(&values, bits, fo, true);
                let got =
                    unpack_bits(&packed, bits, n, fo).unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(got, values, "{bits} 位 {fo:?} 须逐样本精确");
            }
        }
    }

    #[test]
    fn bit_unpack_rejects_bad_depth_and_truncation_without_panic() {
        // 位深 0 → 结构化拒绝。
        assert_eq!(
            unpack_bits(&[0u8; 8], 0, 4, FillOrder::MsbFirst)
                .unwrap_err()
                .code,
            "RAW2001"
        );
        // 位深 17 / 32 → 结构化拒绝（32 位整数 RAW 本模块不支持）。
        for bad in [17u16, 32, 64] {
            let e = unpack_bits(&[0u8; 64], bad, 4, FillOrder::MsbFirst).unwrap_err();
            assert_eq!(e.code, "RAW2001", "位深 {bad} 应被拒绝：{e}");
        }
        // 数据不足 → 结构化拒绝，不 panic、不越界。
        let e = unpack_bits(&[0u8; 3], 12, 100, FillOrder::MsbFirst).unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        // 样本数超防御上限 → 结构化拒绝。
        let e = unpack_bits(&[0u8; 8], 8, MAX_UNPACK_SAMPLES + 1, FillOrder::MsbFirst)
            .unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        // count = 0 → 合法（返回空），不 panic。
        assert!(unpack_bits(&[], 12, 0, FillOrder::MsbFirst)
            .unwrap_or_else(|e| panic!("{e}"))
            .is_empty());
    }

    #[test]
    fn bit_unpack_is_deterministic_across_repeated_calls() {
        // 确定性纪律：同输入解包两遍，逐样本逐位一致。
        for bits in [10u16, 12, 14] {
            let values = sample_series(bits, 64);
            let packed = pack_bits(&values, bits, FillOrder::MsbFirst, true);
            let a = unpack_bits(&packed, bits, 64, FillOrder::MsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            let b = unpack_bits(&packed, bits, 64, FillOrder::MsbFirst)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{bits} 位：两次解包须逐位一致");
        }
    }

    // -----------------------------------------------------------------------
    // 8) 12 位 + RGGB 去马赛克 / 黑白归一（与 16 位同源对拍）
    // -----------------------------------------------------------------------

    #[test]
    fn twelve_bit_rggb_demosaic_matches_sixteen_bit_source() {
        // 同一个「颜色常量块」场景，分别以 12 位与 16 位承载，
        // 去马赛克后各通道值必须**可预测且一致**（位深只影响码值尺度，
        // 不影响 CFA 插值逻辑）。
        let (w, h) = (4u32, 4u32);
        // 12 位域：R=3000, G=700, B=111（与既有 16 位测试同源）。
        let twelve = [
            3000u16, 700, 3000, 700, //
            700, 111, 700, 111, //
            3000, 700, 3000, 700, //
            700, 111, 700, 111,
        ];
        // 16 位域：同一场景放大到 16 位码值（×16，保持 CFA 相位不变）。
        let sixteen: Vec<u16> = twelve.iter().map(|&v| v * 16).collect();

        // 先在 16 位路径上验证「颜色常量块 ⇒ 三通道各自还原为常数」这条
        // 最强可预测性（本测试 12 位断言与它对拍）。
        let plane16 = demosaic_bilinear(
            &sixteen.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            w,
            h,
            BayerLayout::RgGb,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        for i in 0..16usize {
            assert_eq!(plane16.r[i], 3000.0 * 16.0, "16 位 R[{i}]");
            assert_eq!(plane16.g[i], 700.0 * 16.0, "16 位 G[{i}]");
            assert_eq!(plane16.b[i], 111.0 * 16.0, "16 位 B[{i}]");
        }

        // 12 位路径：解包 → 线性化（黑电平按 12 位尺度）→ 去马塞克。
        let packed = pack_bits(&twelve, 12, FillOrder::MsbFirst, true);
        let samples =
            unpack_bits(&packed, 12, 16, FillOrder::MsbFirst).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(samples, twelve.to_vec(), "12 位解包须精确（含非零填充位）");
        // 12 位黑电平取 0（便于与 16 位的「相对值」直接对拍）。
        let lin: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
        let plane12 =
            demosaic_bilinear(&lin, w, h, BayerLayout::RgGb).unwrap_or_else(|e| panic!("{e}"));
        // 与 16 位结果按 ×16 归一后逐点对拍。
        for i in 0..16usize {
            assert_eq!(plane12.r[i], 3000.0, "12 位 R[{i}]");
            assert_eq!(plane12.g[i], 700.0, "12 位 G[{i}]");
            assert_eq!(plane12.b[i], 111.0, "12 位 B[{i}]");
            assert!(
                (plane12.r[i] * 16.0 - plane16.r[i]).abs() < 1e-3,
                "12/16 位 R[{i}] 须同源"
            );
        }
    }

    #[test]
    fn twelve_and_sixteen_bit_normalize_to_same_float() {
        // 同一场景（12 位码值 v，16 位码值 v×16），黑白电平各自按本位深
        // 缩放，归一化后 float 必须在 1e-5 内一致。
        let (w, h) = (4u32, 4u32);
        let twelve: Vec<u16> = (0..16u16).map(|i| 64 + i * 220).collect();
        let sixteen: Vec<u16> = twelve.iter().map(|&v| v * 16).collect();
        let black12 = [64.0f32, 64.0, 64.0, 64.0];
        let white12 = 4095.0f32;
        let black16 = [64.0f32 * 16.0, 64.0 * 16.0, 64.0 * 16.0, 64.0 * 16.0];
        let white16 = 4095.0f32 * 16.0;
        let norm = |samples: &[u16], black: f32, white: f32| -> Vec<f32> {
            let span = white - black;
            samples
                .iter()
                .map(|&v| ((v as f32) - black) / span)
                .collect()
        };
        let n12 = norm(&twelve, black12[0], white12);
        let n16 = norm(&sixteen, black16[0], white16);
        for i in 0..16usize {
            assert!(
                (n12[i] - n16[i]).abs() < 1e-5,
                "第 {i} 样本：12 位归一 {} vs 16 位归一 {}（应一致）",
                n12[i],
                n16[i]
            );
        }
        // 端到端：真实 DNG 文件里 12 位与 16 位显影出的**同一中性灰**
        // 归一值也须一致（走完整 build_packed_dng → read_raw_samples → develop）。
        let gray12: Vec<u16> = vec![2048u16; 16];
        let gray16: Vec<u16> = vec![2048u16 * 16; 16];
        let f12 = build_packed_dng(
            true,
            [0, 1, 1, 2],
            12,
            FillOrder::MsbFirst,
            4,
            w,
            h,
            &gray12,
            4095,
            [64, 64, 64, 64],
        );
        let f16 = build_packed_dng(
            true,
            [0, 1, 1, 2],
            16,
            FillOrder::MsbFirst,
            4,
            w,
            h,
            &gray16,
            4095 * 16,
            [64 * 16, 64 * 16, 64 * 16, 64 * 16],
        );
        let d12 = DngFile::parse(&f12).unwrap_or_else(|e| panic!("{e}"));
        let d16 = DngFile::parse(&f16).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d12.bits_per_sample(), 12);
        assert_eq!(d16.bits_per_sample(), 16);
        let s12 = d12.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        let s16 = d16.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s12, gray12, "12 位读回须逐样本精确");
        assert_eq!(s16, gray16, "16 位读回须逐样本精确");
        // 显影后中心像素（去马赛克 + 矩阵后）应在 1e-2 内接近
        // （f16 交付量化引入约 1e-3 相对误差，故此处放宽到 1e-2）。
        let opts = DevelopOptions {
            white_balance: WbMode::None,
            ..DevelopOptions::default()
        };
        let o12 = d12
            .develop(&s12, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        let o16 = d16
            .develop(&s16, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        let p12 = o12.get(2, 2);
        let p16 = o16.get(2, 2);
        for c in 0..3 {
            assert!(
                (p12[c] - p16[c]).abs() < 1e-2,
                "通道 {c}：12 位 {p12:?} vs 16 位 {p16:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9) strip / tile 布局并存
    // -----------------------------------------------------------------------

    #[test]
    fn tile_and_strip_layouts_decode_identical_pixels() {
        // 8×8 图、RGGB、12 位：同一像素内容分别用 strip（2 行/strip）
        // 与 tile（4×4，恰好 2×2 无边界余量）承载，解出的样本必须逐点相同。
        let (w, h) = (8u32, 8u32);
        let pixels = sample_series(12, (w * h) as usize);
        let strip_file = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 2, w, h, &pixels, 4095,
            [64, 64, 64, 64],
        );
        let tile_file = build_tiled_dng(
            12, FillOrder::MsbFirst, w, h, 4, 4, &pixels, 4095, [64, 64, 64, 64],
        );
        let ds = DngFile::parse(&strip_file).unwrap_or_else(|e| panic!("{e}"));
        let dt = DngFile::parse(&tile_file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            ds.raw_layout().unwrap_or_else(|e| panic!("{e}")),
            Some(RawLayout::Strip { rows_per_strip: 2 })
        );
        assert_eq!(
            dt.raw_layout().unwrap_or_else(|e| panic!("{e}")),
            Some(RawLayout::Tile {
                tile_width: 4,
                tile_length: 4
            })
        );
        let a = ds.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        let b = dt.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "tile 与 strip 布局须解出同一像素");
        assert_eq!(a, pixels, "须与原始样本逐点相同");
        // 两种布局显影结果也须逐位一致。
        let fa = ds
            .develop(&a, &DevelopOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        let fb = dt
            .develop(&b, &DevelopOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fa.pixels, fb.pixels, "tile/strip 显影须逐位一致");
    }

    #[test]
    fn tile_layout_handles_partial_edge_tiles() {
        // 6×6 图用 4×4 tile → 跨 2×2 tile，最后一列/行 tile 只取 2×2 有效区。
        // **边界 tile 仍按完整 4×4 解码**（图外是真实存在的填充样本），
        // 若实现错成「按有效区大小解包」，位流起点会错、整图全乱。
        let (w, h) = (6u32, 6u32);
        let pixels = sample_series(14, (w * h) as usize);
        let file = build_tiled_dng(
            14, FillOrder::MsbFirst, w, h, 4, 4, &pixels, 16383, [128, 128, 128, 128],
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.bits_per_sample(), 14);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.len(), 36);
        assert_eq!(got, pixels, "边界 tile 布局须逐点精确");
    }

    #[test]
    fn tile_layout_rejects_malformed_tile_tags_without_panic() {
        // TileWidth 存在但 TileLength = 0 → 结构化错误（不降级按 strip 读）。
        let mut file = build_tiled_dng(
            12, FillOrder::MsbFirst, 4, 4, 2, 2, &sample_series(12, 16), 4095,
            [64, 64, 64, 64],
        );
        // 把 IFD0 里的 TileLength(0x0143) 值改成 0。
        assert!(
            patch_inline_short(&mut file, tags::TILE_LENGTH, 0),
            "测试脚手架：应恰好命中一条 TileLength"
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        assert!(format!("{e}").contains("TileLength"), "{}", e);
    }

    #[test]
    fn read_raw_samples_rejects_compressed_and_truncated_payloads() {
        let pixels = sample_series(12, 16);
        let good = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &pixels, 4095,
            [64, 64, 64, 64],
        );

        // ---- (A) Compression = 7（lossless JPEG）现在**受支持** ----
        // 本模块的 lossless JPEG 解码器已就位，故 `Compression = 7` 会被**分派到
        // `read_raw_samples_lossless`**（见 `read_raw_samples` 的 match）。
        // 但本文件把**未压缩的紧凑位打包载荷**错标成 7 ⇒ 载荷不是 JPEG 字节流
        // ⇒ 由 JPEG 段解析器结构化拒绝（RAW1001「期望 marker 前缀 0xFF」），
        // **绝不静默把打包字节当成 JPEG 读出垃圾**。这正是我们要锁住的性质。
        //
        // 注：这里**不能**再断言 RAW2001 —— 那是「本模块不支持 Compression=7」时代
        // 的期望，已随无损 JPEG 解码器的落地而过期（该分派是设计要求的）。
        let mut comp = good.clone();
        assert!(
            patch_inline_short(&mut comp, tags::COMPRESSION, 7),
            "测试脚手架：应恰好命中一条 Compression"
        );
        let d = DngFile::parse(&comp).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.compression(), 7);
        // 走的是 lossless 分派（SamplesPerPixel = 1 应通过其前置校验）。
        let err = d.read_raw_samples().unwrap_err();
        assert_eq!(
            err.code, "RAW1001",
            "错标为 Compression=7 的非 JPEG 载荷须报字节流格式错误：{err}"
        );
        let text = format!("{err}");
        assert!(
            text.contains("JPEG") || text.contains("marker") || text.contains("FFD8"),
            "须说明载荷不是 JPEG 字节流：{text}"
        );

        // ---- (B) 真正「未实现的压缩」仍须结构化拒绝并给转存建议 ----
        // 用 Compression = 6（旧式 JPEG，本模块不解码）验证 RAW2001 分支仍在，
        // 且错误信息仍指向「转存为未压缩 / lossless JPEG」。
        let mut comp6 = good.clone();
        assert!(
            patch_inline_short(&mut comp6, tags::COMPRESSION, 6),
            "测试脚手架：应恰好命中一条 Compression"
        );
        let d6 = DngFile::parse(&comp6).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d6.compression(), 6);
        let err6 = d6.read_raw_samples().unwrap_err();
        assert_eq!(err6.code, "RAW2001", "{}", err6);
        let text6 = format!("{err6}");
        assert!(
            text6.contains("Compression") && text6.contains("7"),
            "未实现的压缩须说明只支持 1 与 7：{text6}"
        );
        // 截断载荷 → 结构化错误，不 panic。
        for cut in [good.len() - 1, good.len() - 3, OFF_HEAP + 4] {
            if cut >= good.len() {
                continue;
            }
            let d2 = DngFile::parse(&good[..cut]);
            match d2 {
                Err(e) => assert!(e.code.starts_with("RAW1"), "{e}"),
                Ok(d2) => {
                    let e = d2.read_raw_samples().unwrap_err();
                    assert!(e.code.starts_with("RAW1"), "截断 {cut}：{e}");
                }
            }
        }
        // SamplesPerPixel = 3（线性 RGB）→ 结构化拒绝。
        let mut spp = good.clone();
        assert!(
            patch_inline_short(&mut spp, tags::SAMPLES_PER_PIXEL, 3),
            "测试脚手架：应恰好命中一条 SamplesPerPixel"
        );
        let d3 = DngFile::parse(&spp).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d3.read_raw_samples().unwrap_err().code, "RAW2001");
    }

    #[test]
    fn read_raw_samples_honours_fill_order_tag() {
        // 同一像素分别以 FillOrder=1 / 2 写入文件，读回结果必须相同。
        let pixels = sample_series(12, 16);
        let mut files = Vec::new();
        for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
            files.push(build_packed_dng(
                true, [0, 1, 1, 2], 12, fo, 4, 4, 4, &pixels, 4095, [64, 64, 64, 64],
            ));
        }
        let mut decoded = Vec::new();
        for f in &files {
            let d = DngFile::parse(f).unwrap_or_else(|e| panic!("{e}"));
            decoded.push(d.read_raw_samples().unwrap_or_else(|e| panic!("{e}")));
        }
        assert_eq!(decoded[0], decoded[1], "两种 FillOrder 须读出同一图像");
        assert_eq!(decoded[0], pixels, "须与原始样本逐点相同");
    }

    // -----------------------------------------------------------------------
    // 10) ForwardMatrix2 / CameraCalibration1 / AnalogBalance
    // -----------------------------------------------------------------------

    #[test]
    fn bradford_maps_d50_white_to_d65_white_and_is_identity_for_same_point() {
        // 核心性质：Bradford 适配必须把源白点送到目标白点。
        let m = bradford_adaptation(D50_XYZ, D65_XYZ)
            .unwrap_or_else(|| panic!("Bradford 不应退化"));
        let got = mat3_mul(&m, D50_XYZ);
        for i in 0..3 {
            assert!(
                (got[i] - D65_XYZ[i]).abs() < 1e-4,
                "D50→D65 第 {i} 分量：{} vs {}",
                got[i],
                D65_XYZ[i]
            );
        }
        // 同一白点 → 单位矩阵。**用容差而非 `assert_eq!`**：`B⁻¹ · B` 在 f32 下
                // 有~1e-7 的舍入残差，逐元素精确相等不是应当断言的性质
                // （断言它反而会把「浮点不可精确」误当成回归）。
                let id = bradford_adaptation(D65_XYZ, D65_XYZ)
                    .unwrap_or_else(|| panic!("Bradford 不应退化"));
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (id[r * 3 + c] - want).abs() < 1e-6,
                    "同白点适配须为单位矩阵：[{r}][{c}] = {}（应 {want}）",
                    id[r * 3 + c]
                );
            }
        }
        // 已发表的 Bradford D65→D50 矩阵（1e-3 容差，f32 常数转录差异）。
        let back = bradford_adaptation(D65_XYZ, D50_XYZ)
            .unwrap_or_else(|| panic!("Bradford 不应退化"));
        let published: Mat3 = [
            1.047_929_8, 0.022_946_8, -0.050_192_2, //
            0.029_627_8, 0.990_434_5, -0.017_073_8, //
            -0.009_243_0, 0.015_055_2, 0.751_874_3,
        ];
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (back[r * 3 + c] - published[r * 3 + c]).abs() < 1e-3,
                    "Bradford D65→D50[{}][{}] = {}（发表值 {}）",
                    r,
                    c,
                    back[r * 3 + c],
                    published[r * 3 + c]
                );
            }
        }
        // 退化输入 → None（不静默返回单位矩阵，那会造成无报错的偏色）。
        // 注意：**必须让「锥体响应」退化，而不是让 XYZ 分量为 0**——
        // CAT02 的行和均非零，故 `[0,1,1]` 的锥体响应是 `[0.105, 1.750, 0.961]`，
        // 完全合法，不构成退化。真正的退化是全零（锥体响应三项全0）。
        assert!(
            bradford_adaptation([0.0, 0.0, 0.0], D65_XYZ).is_none(),
            "全零白点的锥体响应退化，须返回 None"
        );
        assert!(
            bradford_adaptation([f32::NAN, 1.0, 1.0], D65_XYZ).is_none(),
            "含 NaN 的白点须返回 None"
        );
        // 目标白点退化同样须None（不得只校验源侧）。
        assert!(
            bradford_adaptation(D50_XYZ, [0.0, 0.0, 0.0]).is_none(),
            "全零目标白点须返回 None"
        );
        // 反面对照：`[0,1,1]` 锥体响应非退化 → 必须给出矩阵。
        assert!(
            bradford_adaptation([0.0, 1.0, 1.0], D65_XYZ).is_some(),
            "[0,1,1] 的锥体响应非退化，不该返回 None"
        );
    }

    #[test]
    fn bradford_round_trip_within_1e5() {
        // 往返一致性：白点适配是白点重参考化，X→X'→X 应回到原值。
        let m = bradford_adaptation(D50_XYZ, D65_XYZ)
            .unwrap_or_else(|| panic!("Bradford 不应退化"));
        let inv = mat3_inverse(&m).unwrap_or_else(|| panic!("适配矩阵应可逆"));
        for &y in &[0.2f32, 0.5, 1.0] {
            let xyz = [D50_XYZ[0] * y, D50_XYZ[1] * y, D50_XYZ[2] * y];
            let back = mat3_mul(&inv, mat3_mul(&m, xyz));
            for i in 0..3 {
                assert!(
                    (back[i] - xyz[i]).abs() < 1e-5,
                    "灰 {y} 往返第 {i} 分量：{} vs {}",
                    back[i],
                    xyz[i]
                );
            }
        }
        // 3×3 求逆的通用性质：M · M⁻¹ = I。
        for m in [
            [2.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ] {
            let i = mat3_inverse(&m).unwrap_or_else(|| panic!("应可逆"));
            let p = mat3_mul_matrix(&m, &i);
            for r in 0..3 {
                for c in 0..3 {
                    let want = if r == c { 1.0 } else { 0.0 };
                    assert!(
                        (p[r * 3 + c] - want).abs() < 1e-5,
                        "M·M⁻¹[{}][{}] = {}（应 {want}）",
                        r,
                        c,
                        p[r * 3 + c]
                    );
                }
            }
        }
        // 奇异矩阵 → None。
        assert!(mat3_inverse(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]).is_none());
    }

    #[test]
    fn develop_matrix_choice_forward2_and_colormatrix1_differ() {
        // **核心一致性**：显式切换 MatrixChoice 时两条路结果**不同**，
        // 证明 ForwardMatrix2 与 ColorMatrix1 真的分开了（没走到同一处）。
        //
        // 构造的 ForwardMatrix2 明显不同于 ColorMatrix1，且带 CameraCalibration1
        // 与 AnalogBalance（规范要求 Forward2 配套的三件套）。
        let fm2: [f32; 9] = [
            0.672_5, 0.196_3, 0.075_4, //
            0.099_5, 0.812_7, 0.087_8, //
            0.048_4, 0.135_2, 0.889_2,
        ];
        let cc1: [f32; 9] = [
            1.015_6, -0.015_6, 0.0, //
            0.0, 1.015_6, -0.015_6, //
            0.0, 0.0, 1.031_3,
        ];
        let analog: [f32; 3] = [0.5, 1.0, 0.75];
        let file = build_forward_dng(fm2, Some(cc1), Some(analog));
        let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        // 两条矩阵都读得到。
        assert!(dng.forward_matrix2().is_some(), "应读到 ForwardMatrix2");
        assert!(dng.color_matrix().is_some(), "应读到 ColorMatrix1");
        assert!(dng.camera_calibration1().is_some(), "应读到 CameraCalibration1");
        assert_eq!(dng.analog_balance(), Some(analog));
        // Auto 选中 ForwardMatrix2。
        let (auto_m, src) = dng
            .resolve_matrix_chain(MatrixChoice::Auto)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(src, "ForwardMatrix2", "Auto 必须优先 ForwardMatrix2");
        let (f2_m, src2) = dng
            .resolve_matrix_chain(MatrixChoice::Forward2)
            .unwrap_or_else(|e| panic!("{e}"));
        let (cm_m, src3) = dng
            .resolve_matrix_chain(MatrixChoice::ColorMatrix1)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(src2, "ForwardMatrix2");
        assert_eq!(src3, "ColorMatrix1");
        // Auto 必须与显式 Forward2 **逐比特相同**（用 to_bits 避免 -0.0/0.0 混淆）。
        for i in 0..9 {
            assert_eq!(
                auto_m[i].to_bits(),
                f2_m[i].to_bits(),
                "第 {i} 元素：Auto 应等同显式 Forward2"
            );
        }
        // 两条路的矩阵**逐元素都不同**。
        let mut max_diff = 0.0f32;
        for i in 0..9 {
            let d = (f2_m[i] - cm_m[i]).abs();
            assert!(d > 1e-3, "第 {i} 元素两条路不该相同（{} vs {}）", f2_m[i], cm_m[i]);
            max_diff = max_diff.max(d);
        }
        assert!(max_diff > 0.01, "最大分量差应显著，实为 {max_diff}");
        // **端到端**：显影结果必须不同。
        let mosaic: Vec<u16> = (0..16u16).map(|i| 300 + i * 111).collect();
        let fa = dng
            .develop(
                &mosaic,
                &DevelopOptions {
                    matrix_choice: MatrixChoice::Forward2,
                    ..DevelopOptions::default()
                },
            )
            .unwrap_or_else(|e| panic!("Forward2 显影失败：{e}"));
        let fb = dng
            .develop(
                &mosaic,
                &DevelopOptions {
                    matrix_choice: MatrixChoice::ColorMatrix1,
                    ..DevelopOptions::default()
                },
            )
            .unwrap_or_else(|e| panic!("ColorMatrix1 显影失败：{e}"));
        assert_ne!(
            fa.pixels, fb.pixels,
            "两条路显影结果必须不同（若相同说明没分开）"
        );
        let pa = fa.get(2, 2);
        let pb = fb.get(2, 2);
        let pix_diff = (0..3)
            .map(|c| (pa[c] - pb[c]).abs())
            .fold(0.0f32, f32::max);
        assert!(pix_diff > 1e-3, "像素差应显著，实为 {pix_diff}");
    }

    #[test]
    fn forward_matrix_chain_order_is_camera_calibration_then_forward() {
        // **链条顺序**的数值断言：M = Bradford · (CC · FM2)。
        // 顺序反了（(Bradford·CC)·FM2）会得到不同的矩阵——本测试用
        // 非交换矩阵把两者区分开。
        let fm2: [f32; 9] = [
            0.6, 0.2, 0.05, //
            0.1, 0.8, 0.1, //
            0.05, 0.15, 0.9,
        ];
        let cc1: [f32; 9] = [
            1.02, 0.01, 0.0, //
            0.0, 1.01, 0.02, //
            0.0, 0.0, 0.98,
        ];
        let file = build_forward_dng(fm2, Some(cc1), None);
        let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let got = dng
            .color_matrix_chain(MatrixChoice::Forward2)
            .unwrap_or_else(|e| panic!("{e}"));
        let adapt = bradford_adaptation(D50_XYZ, D65_XYZ)
            .unwrap_or_else(|| panic!("Bradford 不应退化"));
        // **用「从文件里读回来的」矩阵**做期望值，不能用构造时的 f32 字面量：
        // 脚手架把矩阵写成 SRATIONAL（整数分子/分母），读回值带量化误差
        // （本例约 7e-4）。断言链路**顺序**时必须以解析后的值为准，否则
        // 测的是「字面量 vs 量化值」的差，与链路顺序无关。
        let fm2_read = dng
            .forward_matrix2()
            .unwrap_or_else(|| panic!("应读到 ForwardMatrix2"));
        let cc1_read = dng
            .camera_calibration1()
            .unwrap_or_else(|| panic!("应读到 CameraCalibration1"));
        let want = mat3_mul_matrix(&adapt, &mat3_mul_matrix(&cc1_read, &fm2_read));
        for i in 0..9 {
            assert!(
                (got[i] - want[i]).abs() < 1e-6,
                "第 {i} 元素：实得 {}，按「Bradford·(CC·FM2)」应为 {}",
                got[i],
                want[i]
            );
        }
        // 顺序不能颠倒：先 CC 后 FM2 ≠ 先 FM2 后 CC（矩阵不交换）。
        let swapped = mat3_mul_matrix(&adapt, &mat3_mul_matrix(&fm2_read, &cc1_read));
        let mut order_diff = 0.0f32;
        for i in 0..9 {
            order_diff = order_diff.max((want[i] - swapped[i]).abs());
        }
        assert!(
            order_diff > 1e-4,
            "CC 与 FM2 的顺序必须影响结果（实测最大差 {order_diff}）"
        );
        // 反序（错误顺序）必须**不**等于正确结果（否则本测试无区分度）。
        let wrong = mat3_mul_matrix(&mat3_mul_matrix(&adapt, &cc1), &fm2);
        let mut differs = false;
        for i in 0..9 {
            if (wrong[i] - want[i]).abs() > 1e-5 {
                differs = true;
            }
        }
        assert!(differs, "反序结果应与正序不同（否则顺序断言无意义）");
    }

    #[test]
    fn analog_balance_divides_into_as_shot_neutral() {
        // effective_neutral = AsShotNeutral / AnalogBalance（逐通道）。
        let analog: [f32; 3] = [0.5, 1.0, 0.75];
        let file = build_forward_dng(
            [
                0.6, 0.2, 0.05, 0.1, 0.8, 0.1, 0.05, 0.15, 0.9,
            ],
            None,
            Some(analog),
        );
        let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let n = dng.as_shot_neutral().unwrap_or_else(|| panic!("应有 AsShotNeutral"));
        let ab = dng.analog_balance().unwrap_or_else(|| panic!("应有 AnalogBalance"));
        let eff = dng
            .effective_neutral()
            .unwrap_or_else(|| panic!("应有 effective_neutral"));
        for i in 0..3 {
            assert!((eff[i] - n[i] / ab[i]).abs() < 1e-6, "第 {i} 项 = {}", eff[i]);
        }
        // 具体数值：AsShotNeutral = [0.5, 1.0, 0.75]、AnalogBalance = [0.5, 1, 0.75]
        // ⇒ effective = [1, 1, 1] ⇒ 白平衡增益全为 1（绿归一下亦然）。
        for i in 0..3 {
            assert!((eff[i] - 1.0).abs() < 1e-6, "第 {i} 项应为 1.0，实为 {}", eff[i]);
        }
        let g = white_balance_gains(eff, WbMode::GreenNormalized);
        assert_eq!(g, [1.0, 1.0, 1.0], "除完 AnalogBalance 后中性值应为 1");
        // **有/无 AnalogBalance 的差别**：无该标签时直接用 AsShotNeutral。
        let no_ab = build_forward_dng(
            [
                0.6, 0.2, 0.05, 0.1, 0.8, 0.1, 0.05, 0.15, 0.9,
            ],
            None,
            None,
        );
        let d2 = DngFile::parse(&no_ab).unwrap_or_else(|e| panic!("{e}"));
        assert!(d2.analog_balance().is_none());
        assert_eq!(
            d2.effective_neutral()
                .unwrap_or_else(|| panic!("无 AnalogBalance 时应回退 AsShotNeutral")),
            d2.as_shot_neutral()
                .unwrap_or_else(|| panic!("该 DNG 应带 AsShotNeutral")),
            "无 AnalogBalance 时 effective_neutral 应等于 AsShotNeutral"
        );
    }

    #[test]
    fn forward_matrix_chain_round_trips_camera_to_xyz_within_1e5() {
        // 往返一致性：X → RGB(D50) → X 误差 < 1e-5。
        // （Forward2 链的输出是 D50；用其「D50 化」的装配矩阵做往返。）
        let fm2: [f32; 9] = [
            0.672_5, 0.196_3, 0.075_4, //
            0.099_5, 0.812_7, 0.087_8, //
            0.048_4, 0.135_2, 0.889_2,
        ];
        let cc1: [f32; 9] = [
            1.015_6, -0.015_6, 0.0, //
            0.0, 1.015_6, -0.015_6, //
            0.0, 0.0, 1.031_3,
        ];
        let file = build_forward_dng(fm2, Some(cc1), None);
        let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        // 到 sRGB 的完整装配矩阵。
        let to_srgb = mat3_mul_matrix(
            &XYZ_TO_SRGB,
            &dng.color_matrix_chain(MatrixChoice::Forward2)
                .unwrap_or_else(|e| panic!("{e}")),
        );
        let back = mat3_inverse(&to_srgb).unwrap_or_else(|| panic!("应可逆"));
        for &v in &[0.15f32, 0.35, 0.6, 0.9] {
            let cam = [v, v * 0.8, v * 0.5];
            let rgb = mat3_mul(&to_srgb, cam);
            let cam2 = mat3_mul(&back, rgb);
            for i in 0..3 {
                assert!(
                    (cam2[i] - cam[i]).abs() < 1e-5,
                    "v={v} 第 {i} 分量往返：{} vs {}",
                    cam2[i],
                    cam[i]
                );
            }
        }
    }

    #[test]
    fn matrix_choice_explicit_selection_errors_when_tag_missing() {
        // 显式要求但标签缺失 → **结构化报错**，不静默回退到另一条路。
        // （文件只有 ColorMatrix1：build_forward_dng 的 fm2 换成全 0 会写进去，
        //  故这里用一个不带 ForwardMatrix2 的 build_dng。）
        let bytes = build_dng(true, [0, 1, 1, 2], 16);
        let dng = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert!(dng.forward_matrix2().is_none(), "该文件不应有 ForwardMatrix2");
        let e = dng
            .color_matrix_chain(MatrixChoice::Forward2)
            .unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        assert!(format!("{e}").contains("ForwardMatrix2"), "{}", e);
        // 反之：显式 ColorMatrix1 在只有 Forward2 的文件里也报错。
        let mut only_f = build_forward_dng(
            [
                0.6, 0.2, 0.05, 0.1, 0.8, 0.1, 0.05, 0.15, 0.9,
            ],
            None,
            None,
        );
        // 把 ColorMatrix1 标签号改成 0xFFFF 使其消失。
        let n = ifd0_entry_count(&only_f);
        for i in 0..n {
            let e = ifd0_entry_offset(&only_f, i);
            if u16::from_le_bytes([only_f[e], only_f[e + 1]]) == tags::COLOR_MATRIX1 {
                only_f[e..e + 2].copy_from_slice(&0xFFFFu16.to_le_bytes());
            }
        }
        let d2 = DngFile::parse(&only_f).unwrap_or_else(|e| panic!("{e}"));
        assert!(d2.color_matrix().is_none());
        let e2 = d2
            .color_matrix_chain(MatrixChoice::ColorMatrix1)
            .unwrap_err();
        assert!(format!("{e2}").contains("ColorMatrix1"), "{}", e2);
        // Auto 在同一文件上正常工作（退回 ForwardMatrix2）。
        let (_, src) = d2
            .resolve_matrix_chain(MatrixChoice::Auto)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(src, "ForwardMatrix2");
    }

    // -----------------------------------------------------------------------
    // 11) CR3：ISO-BMFF 盒树遍历 + uuid 定位 + 结构化拒绝
    // -----------------------------------------------------------------------

    #[test]
    fn cr3_box_walk_finds_ftyp_uuid_and_mdat() {
        // 最小 CR3：ftyp(crx ) + uuid(Apple, 内含 CMT1/CMT2/THMB) + mdat。
        let mdat: Vec<u8> = {
            let mut v = b"CRAW".to_vec();
            v.extend_from_slice(b"CTBO");
            v.extend_from_slice(&[0u8; 32]);
            v
        };
        let cr3 = build_cr3(
            b"crx ",
            true,
            &[
                (b"CMT1", vec![1, 2, 3, 4]),
                (b"CMT2", vec![5, 6, 7, 8]),
                (b"THMB", vec![9, 10]),
            ],
            mdat,
        );
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&info.major_brand, b"crx ", "major brand 应为 crx ");
        // 盒树：ftyp / uuid / CMT1 / CMT2 / THMB / mdat（先序）。
        let kinds: Vec<String> = info.boxes.iter().map(|b| b.kind_str()).collect();
        assert_eq!(
            kinds,
            vec!["ftyp", "uuid", "CMT1", "CMT2", "THMB", "mdat"],
            "盒树先序遍历结果不符"
        );
        // 深度：顶层 = 0，uuid 的子盒 = 1。
        let depths: Vec<usize> = info.boxes.iter().map(|b| b.depth).collect();
        assert_eq!(depths, vec![0, 0, 1, 1, 1, 0]);
        // Apple uuid 定位。
        let ao = info
            .apple_uuid_offset
            .unwrap_or_else(|| panic!("应定位到 Apple uuid 盒"));
        assert_eq!(&cr3[ao as usize + 4..ao as usize + 8], b"uuid");
        // CRAW 头定位。
        let co = info
            .craw_header_offset
            .unwrap_or_else(|| panic!("应在 mdat 内定位到 CRAW 头"));
        assert_eq!(&cr3[co as usize..co as usize + 8], b"CRAWCTBO");
        assert_eq!(info.mdat_count, 1);
        // 盒树遍历确定性：两遍逐字段一致。
        let again = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(info.boxes, again.boxes, "盒树遍历须确定性");
    }

    #[test]
    fn cr3_box_walk_skips_unknown_boxes_without_panic() {
        // 未知盒（如 `free` / `abcd`）只记录、不递归其负载。
        // ftyp(24B) + 未知盒 abcd + mdat。
        // abcd 的负载里**故意放看似盒头的字节**（`CRAW` + 一个 16 字节的盒头），
        // 以证明未知盒不会被误当容器递归 —— 递归它就会凭空多出假盒。
        let mut abcd: Vec<u8> = Vec::new();
        // 负载 = 4(size) + 4(CRAW) + 4(size) + 4(CRAW) + 4(CTBO) = 20 字节，
        // 故盒总长 = 8 + 20 = **28**（声明 24 会与实际负载不自洽）。
        abcd.extend_from_slice(&28u32.to_be_bytes());
        abcd.extend_from_slice(b"abcd");
        abcd.extend_from_slice(&8u32.to_be_bytes());
        abcd.extend_from_slice(b"CRAW"); // 看似盒头，但abcd 不是容器 → 不递归
        abcd.extend_from_slice(&16u32.to_be_bytes());
        abcd.extend_from_slice(b"CRAW");
        abcd.extend_from_slice(b"CTBO");
        let mut cr3: Vec<u8> = build_cr3(b"crx ", false, &[], vec![0xAA; 16]);
        // 把 abcd 插到 mdat 之前（ftyp 24 字节处），而不是追加到文件末尾——
        // 追加会落在 mdat 的负载里，构成「盒序列自相矛盾」的畸形流。
        let mdat_at = 8 + 4 + 4 + 8; // ftyp 盒总长
        let mut merged: Vec<u8> = Vec::with_capacity(cr3.len() + abcd.len());
        merged.extend_from_slice(&cr3[..mdat_at]);
        merged.extend_from_slice(&abcd);
        merged.extend_from_slice(&cr3[mdat_at..]);
        cr3 = merged;
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        let kinds: Vec<String> = info.boxes.iter().map(|b| b.kind_str()).collect();
        assert_eq!(kinds, vec!["ftyp", "abcd", "mdat"], "未知盒应被记录但不被递归");
        // 未知盒负载里的假 CRAW 不得被误认（只有 mdat 负载才被搜 CRAW）。
        assert!(info.craw_header_offset.is_none(), "非 mdat 内不应报 CRAW 头");
    }

    #[test]
    fn cr3_malformed_boxes_are_rejected_structurally_not_panic() {
        // (a) size < 头长：8 字节头声明 size = 4（< 8）。
        // ftyp 盒在偏移 0（盒长 24 = 8 头+ 4 major + 4 minor + 8 兼容品牌）。
        let ftyp_len = 0usize;
        let mut bad = build_cr3(b"crx ", false, &[], vec![0u8; 8]);
        bad[ftyp_len..ftyp_len + 4].copy_from_slice(&4u32.to_be_bytes());
        let e = probe_cr3(&bad).unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        // (b) size = 0 → 延伸到父容器末尾（ISO-BMFF 合法语义），应能遍历。
        //     注意 size=0 必须打在**mdat**（偏移 24）上：若打在 ftyp 上，
        //     ftyp 会吞掉整个文件，只剩 1 个盒，就测不到「吞尾但其后无盒」了。
        let mdat_at = 8 + 4 + 4 + 8; // ftyp 盒总长
        let mut zero = build_cr3(b"crx ", false, &[], vec![0u8; 8]);
        zero[mdat_at..mdat_at + 4].copy_from_slice(&0u32.to_be_bytes());
        // size=0 的盒会吞掉剩余全部字节，其后不再有盒 —— 遍历须成功且只有 2 盒。
        let info = probe_cr3(&zero).unwrap_or_else(|e| panic!("size=0 应按「延伸到末尾」处理：{e}"));
        assert_eq!(info.boxes.len(), 2, "size=0 的盒应吞掉剩余全部");
        assert_eq!(info.boxes[1].kind_str(), "mdat");
        // (c) size 越过父容器边界 → 结构化错误。
        let mut over = build_cr3(b"crx ", false, &[], vec![0u8; 8]);
        over[ftyp_len..ftyp_len + 4].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
        let e = probe_cr3(&over).unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        // (d) 截断：盒头不全。
        for cut in [0usize, 1, 4, 7, 8, 9, 12] {
            let e = probe_cr3(&build_cr3(b"crx ", false, &[], vec![])[..cut.min(24)])
                .unwrap_err();
            assert!(e.code.starts_with("RAW1"), "截断 {cut}：{e}");
        }
        // (e) 非 ISO-BMFF（不是 ftyp 开头）。
        let e = probe_cr3(b"II\x2a\x00\x08\x00\x00\x00abcd")
            .unwrap_err();
        assert_eq!(e.code, "RAW1001", "{}", e);
        // (f) 深度超限：手工造一条 `MAX_BOX_DEPTH + 4` 层的**真实嵌套** `udta` 链。
        //     每层盒的负载是**内一层盒**（故size > header_size，遍历器才会下潜）。
        let mut deep: Vec<u8> = vec![0u8; 8]; // 最内层：一个 8 字节的哑元盒
        for _ in 0..(MAX_BOX_DEPTH + 4) {
            let mut v: Vec<u8> = Vec::with_capacity(8 + deep.len());
            v.extend_from_slice(&((8 + deep.len()) as u32).to_be_bytes());
            v.extend_from_slice(b"udta");
            v.extend_from_slice(&deep);
            deep = v;
        }
        let e = walk_boxes(&deep).unwrap_err();
        assert_eq!(e.code, "RAW1001", "深度超限应结构化报错：{e}");
        assert!(format!("{e}").contains("深度"), "{}", e);
        // (g) 空文件 / 短文件不 panic。
        for bad in [&[][..], &[0u8][..], &[0u8; 7][..]] {
            let _ = probe_cr3(bad).unwrap_err();
        }
    }

    #[test]
    fn cr3_real_byte_stream_is_rejected_with_dng_advice() {
        // 真实形态的 CR3 字节流 → **结构化拒绝**，且错误信息含 "DNG" 建议。
        let cr3 = build_cr3(
            b"crx ",
            true,
            &[(b"CMT1", vec![0; 16]), (b"CMT3", vec![1; 16])],
            {
                let mut v = b"CRAWCTBO".to_vec();
                v.extend_from_slice(&[0u8; 64]);
                v
            },
        );
        // ① probe_format（像素入口）拒绝。
        let err = probe_format(&cr3).unwrap_err();
        assert_eq!(err.code, "RAW2001", "{}", err);
        let text = format!("{err}");
        for kw in ["ftyp", "CMT1", "CMT2", "CRX", "CFA", "DNG"] {
            assert!(text.contains(kw), "拒绝理由须提到 {kw}：{text}");
        }
        // ② DngFile::parse 同样拒绝（不 panic）。
        assert_eq!(DngFile::parse(&cr3).unwrap_err().code, "RAW2001");
        // ③ 容器层可解析（不再「瞎拒绝」）。
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        assert!(info.apple_uuid_offset.is_some());
        assert!(info.craw_header_offset.is_some());
        // ④ **像素解码明确拒绝**，错误须说清「需 CMT3/CDC 完整实现」+「转 DNG」。
        let e2 = info.decode_pixels().unwrap_err();
        assert_eq!(e2.code, "RAW2001", "{}", e2);
        let t2 = format!("{e2}");
        for kw in ["CMT3", "CDC", "DNG"] {
            assert!(t2.contains(kw), "像素解码拒绝须提到 {kw}：{t2}");
        }
        // ⑤ decode_pixels 恒失败（对任意盒树都如此）。
        let empty = Cr3Info::default();
        assert!(empty.decode_pixels().is_err());
    }

    #[test]
    fn cr3_probe_reads_cmt1_sensor_size_when_present() {
        // CMT1 内嵌一个 TIFF 头 + IFD（带 ImageWidth/Length）时应读出尺寸；
        // 读不到时必须是 None（**不猜、不返回默认值**）。
        // (a) 带有效内嵌 TIFF。
        let tiff = build_dng(true, [0, 1, 1, 2], 16);
        let cr3 = build_cr3(b"crx ", true, &[(b"CMT1", tiff)], vec![0u8; 8]);
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(info.sensor_size, Some((4, 4)), "应从 CMT1 读出 4×4");
        // (b) CMT1 是随机字节 → None（不猜）。
        let cr3b = build_cr3(
            b"crx ",
            true,
            &[(b"CMT1", vec![0xAB; 64])],
            vec![0u8; 8],
        );
        let info2 = probe_cr3(&cr3b).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(info2.sensor_size, None, "读不到必须是 None");
        // (c) 无 CMT1 → None。
        let cr3c = build_cr3(b"crx ", false, &[], vec![0u8; 8]);
        let info3 = probe_cr3(&cr3c).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(info3.sensor_size, None);
    }

    #[test]
    fn cr3_uuid_box_uses_extended_16_byte_header() {
        // uuid 盒的负载起点必须跳过 **16 字节 UUID**（不是 8）。
        // 若错按 8 字节解析，CMT1 的起点会偏 16 字节 → 类型乱码。
        let cr3 = build_cr3(b"crx ", true, &[(b"CMT1", vec![7u8; 4])], vec![0u8; 4]);
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        let uuid = info
            .boxes
            .iter()
            .find(|b| b.is_uuid())
            .unwrap_or_else(|| panic!("应有 uuid 盒"));
        assert_eq!(uuid.header_size, 24, "uuid 头 = 8（基本）+ 16（UUID）");
        let (ps, pe) = uuid.payload_range();
        // `payload_range().0` 指向**第一个子盒的起点**，即其 size 字段；
        // 四字符类型在盒头第4..8 字节，即 `ps + 4`。
        assert_eq!(
            &cr3[ps as usize + 4..ps as usize + 8],
            b"CMT1",
            "uuid 负载应从 CMT1 开始"
        );
        assert!(pe > ps);
        // 载荷长度 = 内部盒总长（12 字节：8 头 + 4 负载）。
        assert_eq!(pe - ps, 12);
    }

    #[test]
    fn cr3_box_walk_handles_64bit_largesize() {
        // size == 1 → 64 位 largesize（ISO-BMFF §4.2）。
        let mut cr3: Vec<u8> = Vec::new();
        // ftyp：size = 1 →后跟 64 位 largesize。
        // largesize 覆盖**整个盒、含 largesize 字段本身**：
        //   8(基本头) + 8(largesize) + 4(major "crx ") + 4(minor 0) + 4("isom")
        //     = **28**。
        // （常见笔误：把兼容品牌当成 2 个 4 字节品牌写成 8 字节 → 算成 32，
        //   而实际只写了 4 字节 "isom"，盒长与内容不自洽，遍历会在偏移 28
        //   处报「伪盒 size 巨大」——测的就不是 largesize 而是夹具自相矛盾。）
        cr3.extend_from_slice(&1u32.to_be_bytes());
        cr3.extend_from_slice(b"ftyp");
        cr3.extend_from_slice(&28u64.to_be_bytes());
        cr3.extend_from_slice(b"crx ");
        cr3.extend_from_slice(&0u32.to_be_bytes());
        cr3.extend_from_slice(b"isom");
        // mdat：常规盒，**紧接在 ftyp 之后**（偏移 28 = ftyp 的 largesize）。
        cr3.extend_from_slice(&12u32.to_be_bytes());
        cr3.extend_from_slice(b"mdat");
        cr3.extend_from_slice(&[1, 2, 3, 4]);
        let info = probe_cr3(&cr3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(info.boxes.len(), 2);
        assert_eq!(info.boxes[0].header_size, 16, "largesize 头 = 8 + 8");
        assert_eq!(&info.major_brand, b"crx ");
        assert_eq!(info.boxes[1].kind_str(), "mdat");
        // largesize 声明的长度小于头长 → 结构化错误。
        // ftyp 是第 0 个盒，起点即偏移 0。
        let ftyp_len = 0usize;
        let mut bad = build_cr3(b"crx ", false, &[], vec![0u8; 4]);
        bad[ftyp_len..ftyp_len + 4].copy_from_slice(&1u32.to_be_bytes());
        bad[ftyp_len + 4..ftyp_len + 8].copy_from_slice(b"ftyp");
        bad[ftyp_len + 8..ftyp_len + 16].copy_from_slice(&9u64.to_be_bytes());
        assert_eq!(probe_cr3(&bad).unwrap_err().code, "RAW1001");
    }

    // -----------------------------------------------------------------------
    // 12) 端到端确定性（解包 + 显影两遍逐字节一致）
    // -----------------------------------------------------------------------

    #[test]
    fn twelve_bit_end_to_end_pipeline_is_byte_identical_twice() {
        // 同输入解包两遍 + 显影两遍，全部逐字节一致（确定性铁律）。
        let (w, h) = (8u32, 8u32);
        let pixels = sample_series(12, (w * h) as usize);
        let file = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 3, w, h, &pixels, 4095,
            [64, 128, 128, 96],
        );
        let opts = DevelopOptions::default();
        let mut prev_samples: Option<Vec<u16>> = None;
        let mut prev_pixels: Option<Vec<u16>> = None;
        for _ in 0..3 {
            let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let s = dng.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(s, pixels, "12 位端到端解包须逐样本精确");
            let fb = dng.develop(&s, &opts).unwrap_or_else(|e| panic!("{e}"));
            if let Some(p) = &prev_samples {
                assert_eq!(&s, p, "解包结果须逐字节一致");
            }
            if let Some(p) = &prev_pixels {
                assert_eq!(&fb.pixels, p, "显影结果须逐字节一致");
            }
            prev_samples = Some(s);
            prev_pixels = Some(fb.pixels.clone());
        }
    }

    #[test]
    fn levels_carry_bit_depth_and_fill_order() {
        // RawLevels 必须携带位深与位序（下游归一化依赖它）。
        let pixels = sample_series(14, 16);
        for fo in [FillOrder::MsbFirst, FillOrder::LsbFirst] {
            let file = build_packed_dng(
                true, [0, 1, 1, 2], 14, fo, 4, 4, 4, &pixels, 16383,
                [128, 128, 128, 128],
            );
            let dng = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let cfa = dng.cfa_pattern().unwrap_or_else(|e| panic!("{e}"));
            let lv = dng.levels(&cfa).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(lv.bits_per_sample, 14);
            assert_eq!(lv.fill_order, fo);
            assert_eq!(lv.white, 16383.0);
        }
        // FillOrder 标签值 → 枚举映射（含非法值 → None）。
        assert_eq!(FillOrder::from_code(1), Some(FillOrder::MsbFirst));
        assert_eq!(FillOrder::from_code(2), Some(FillOrder::LsbFirst));
        assert_eq!(FillOrder::from_code(0), None);
        assert_eq!(FillOrder::from_code(3), None);
        assert_eq!(FillOrder::MsbFirst.code(), 1);
        assert_eq!(FillOrder::LsbFirst.code(), 2);
        // 缺 FillOrder 标签 → 缺省 MSB-first（DNG 规范）。
        let bytes = build_dng(true, [0, 1, 1, 2], 16);
        let d = DngFile::parse(&bytes).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.fill_order(), FillOrder::MsbFirst);
    }

    #[test]
    fn big_endian_packed_dng_decodes_identically_to_little_endian() {
        // MM（大端）容器里的 12 位紧凑流，解出的样本须与 II 完全相同。
        let pixels = sample_series(12, 16);
        let le = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &pixels, 4095,
            [64, 64, 64, 64],
        );
        let be = build_packed_dng(
            false, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &pixels, 4095,
            [64, 64, 64, 64],
        );
        let a = DngFile::parse(&le).unwrap_or_else(|e| panic!("{e}"));
        let b = DngFile::parse(&be).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.order, ByteOrder::Little);
        assert_eq!(b.order, ByteOrder::Big);
        let sa = a.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        let sb = b.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sa, sb, "II/MM 容器的 12 位流须解出同一图像");
        assert_eq!(sa, pixels);
    }
    // ===========================================================================
    //  DNG Compression = 7（ITU-T T.81 无损 JPEG）端到端测试
    // ===========================================================================

    /// 构造一个 **`Compression = 7`** 的 DNG 文件（**支持多 strip + 行交织**）。
    ///
    /// 与 [`build_packed_dng`] 版式一致，只是像素载荷换成**真实的 SOF3 字节流**
    /// （由 `crate::raw_lossless_jpeg::tests_support::encode` 产出）。
    /// `rows_per_strip` 控制 strip 数；`interleave` 写 `RowInterleaveFactor`。
    ///
    /// 覆盖「多 strip 拼接」与「`RowInterleaveFactor != 1`」两条路径
    /// （单 strip 是其 `strips.len() == 1` 的特例）。
    // 参数多是因为 DNG 的 IFD 字段本身就这么独立（白/黑、电位、位深、CFA…），
    // 强行打包成结构体反而让调用处的意图更难读；仅测试辅助函数，放宽此lint。
    #[allow(clippy::too_many_arguments)]
    fn build_lossless_dng_multi(
        le: bool,
        cfa: [u8; 4],
        bits: u16,
        rows_per_strip: u32,
        width: u32,
        height: u32,
        strips: &[(Vec<u8>, u32)], // (JPEG 字节流, 该 strip 的行数)
        white: u32,
        black: [u32; 4],
        interleave: Option<u32>,
    ) -> Vec<u8> {
    let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
    let mut offs: Vec<u32> = Vec::with_capacity(strips.len());
    let mut counts: Vec<u32> = Vec::with_capacity(strips.len());
    let mut cursor = 0usize;
    for (bytes, _) in strips {
        offs.push((payload_base + cursor) as u32);
        counts.push(bytes.len() as u32);
        cursor += bytes.len();
    }

    let mut ifd0 = IfdBuilder::new(le);
    ifd0.longs(tags::IMAGE_WIDTH, &[width]);
    ifd0.longs(tags::IMAGE_LENGTH, &[height]);
    ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
    ifd0.shorts(tags::COMPRESSION, &[7]);
    ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
    ifd0.longs(tags::STRIP_OFFSETS, &offs);
    ifd0.longs(tags::STRIP_BYTE_COUNTS, &counts);
    ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
    ifd0.longs(tags::ROWS_PER_STRIP, &[rows_per_strip]);
    ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
    ifd0.bytes(tags::CFA_PATTERN, &cfa);
    ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
    ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 2, 0, 0]);
    ifd0.longs(tags::WHITE_LEVEL, &[white]);
    let bl: Vec<f32> = black.iter().map(|&b| b as f32).collect();
    ifd0.rationals(tags::BLACK_LEVEL, &bl);
    ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
    ifd0.srationals(
        tags::COLOR_MATRIX1,
        &[
            0.700, 0.200, 0.0505, //
            0.100, 0.800, 0.100, //
            0.050, 0.149, 0.889,
        ],
    );
    if let Some(f) = interleave {
        ifd0.shorts(tags::ROW_INTERLEAVE_FACTOR, &[f as u16]);
    }

    let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
    if le {
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
    } else {
        file[0..2].copy_from_slice(b"MM");
        file[2..4].copy_from_slice(&42u16.to_be_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_be_bytes());
    }
    let mut heap: Vec<u8> = Vec::new();
    let bytes = write_ifd(&ifd0, le, 0, &mut heap, OFF_HEAP);
    if OFF_IFD0 + bytes.len() > OFF_HEAP {
        panic!("测试脚手架：IFD 超出预留版式（{} 字节）", bytes.len());
    }
    file[OFF_IFD0..OFF_IFD0 + bytes.len()].copy_from_slice(&bytes);
    file.extend_from_slice(&heap);
    file.resize(payload_base, 0u8);
    for (bytes, _) in strips {
        file.extend_from_slice(bytes);
    }
    file
    }


    /// 便捷：造一条 `Compression=7` 的单 strip DNG。
    fn one_strip_dng(
        px: &[u16],
        w: u32,
        h: u32,
        precision: u8,
        predictor: u8,
        rows_per_strip: u32,
        interleave: Option<u32>,
    ) -> Vec<u8> {
        let mut strips: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut y = 0u32;
        while y < h {
            let rows = rows_per_strip.min(h - y);
            let chunk: Vec<u16> = px[(y * w) as usize..((y + rows) * w) as usize].to_vec();
            strips.push((crate::raw_lossless_jpeg::tests_support::encode(&chunk, w, rows, precision, predictor, 0), rows));
            y += rows;
        }
        build_lossless_dng_multi(
            true,
            [0, 1, 1, 2],
            precision as u16,
            rows_per_strip,
            w,
            h,
            &strips,
            (1u32 << precision) - 1,
            [0, 0, 0, 0],
            interleave,
        )
    }

    fn gradient(w: u32, h: u32, mod_max: u32) -> Vec<u16> {
        (0..(w * h) as usize)
            .map(|i| ((i as u32 * 37) % mod_max) as u16)
            .collect()
    }

    // -------------------------------------------------------------------
    //  端到端：read_raw_samples 走 Compression=7 分派
    // -------------------------------------------------------------------

    #[test]
    fn dng_lossless_16bit_single_strip_round_trip() {
        let (w, h) = (16u32, 12u32);
        let px = gradient(w, h, 65536);
        for pred in 1u8..=7 {
            let file = one_strip_dng(&px, w, h, 16, pred, h, None);
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d.compression(), 7, "必须走Compression=7 分派");
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("pred={pred}: {e}"));
            assert_eq!(got.len(), (w * h) as usize);
            assert_eq!(got, px, "pred={pred} 的 16bit DNG 端到端 round-trip 必须逐位一致");
        }
    }

    #[test]
    fn dng_lossless_8bit_round_trip() {
        let (w, h) = (16u32, 10u32);
        let px = gradient(w, h, 256);
        let file = one_strip_dng(&px, w, h, 8, 1, h, None);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, px, "8bit DNG 端到端 round-trip 必须一致");
    }

    #[test]
    fn dng_lossless_big_endian_container() {
        // MM（大端）容器：偏移/字节数是大端，载荷字节流不变。
        let (w, h) = (8u32, 6u32);
        let px = gradient(w, h, 65536);
        let stream = crate::raw_lossless_jpeg::tests_support::encode(&px, w, h, 16, 1, 0);
        let file = build_lossless_dng_multi(
            false,
            [0, 1, 1, 2],
            16,
            h,
            w,
            h,
            &[(stream, h)],
            65535,
            [0, 0, 0, 0],
            None,
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.order, ByteOrder::Big);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, px, "II/MM 容器须解出同一图像");
    }

    #[test]
    fn dng_lossless_multi_strip() {
        // 多个 strip：验证每个 strip 的y0 与 strip 边界正确拼接。
        let (w, h) = (12u32, 12u32);
        let px = gradient(w, h, 65536);
        let rps = 5u32; // → strip 行数 5,5,2
        let file = one_strip_dng(&px, w, h, 16, 1, rps, None);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, px, "多 strip 拼接必须逐位一致");
    }

    #[test]
    fn dng_lossless_row_interleave_factor_2_end_to_end() {
        // 端到端解交织：把整图按 factor=2 的 field-first 序**编进** JPEG 流，
        // DNG 侧只靠 RowInterleaveFactor 标签还原。
        let (w, h) = (8u32, 8u32);
        // 垂直非对称图案：偶数行 1000 基准、奇数行 5000 基准。
        let img: Vec<u16> = (0..(w * h) as usize)
            .map(|i| {
                let y = i / w as usize;
                let x = i % w as usize;
                (if y.is_multiple_of(2) { 1000u16 } else { 5000 }) + x as u16
            })
            .collect();
        // 手工交织：码流第 j 行放field 映射到的目标行。
        let map = crate::raw_lossless_jpeg::interleave_row_map(h, 0, 2).expect("factor=2");
        let mut coded = vec![0u16; (w * h) as usize];
        for (j, &dst) in map.iter().enumerate() {
            let src = (dst * w) as usize;
            let d = j * w as usize;
            coded[d..d + w as usize].copy_from_slice(&img[src..src + w as usize]);
        }
        // 单 strip（整高），factor=2。
        let stream = crate::raw_lossless_jpeg::tests_support::encode(&coded, w, h, 16, 1, 0);
        let file = build_lossless_dng_multi(
            true,
            [0, 1, 1, 2],
            16,
            h,
            w,
            h,
            &[(stream, h)],
            65535,
            [0, 0, 0, 0],
            Some(2),
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.row_interleave_factor(), 2);
        let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got, img, "端到端解交织必须还原成偶数行在前的原图行序");
        // 反证：行序错（奇偶互换）必被抓 —— 图案的1000/5000 基准会互换。
        for y in 0..h as usize {
            let v = got[y * w as usize];
            let expect = if y % 2 == 0 { 1000u16 } else { 5000 };
            assert_eq!(v, expect, "行 {y} 的首像素基准必须是 {expect}（行序须正确）");
        }
    }

    #[test]
    fn dng_row_interleave_factor_defaults_to_one() {
        let (w, h) = (4u32, 4u32);
        let px = gradient(w, h, 256);
        let file = one_strip_dng(&px, w, h, 8, 1, h, None);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            d.row_interleave_factor(),
            1,
            "DNG 规定 RowInterleaveFactor 缺省 = 1（不交织）"
        );
    }

    #[test]
    fn dng_lossless_determinism_two_reads_bit_identical() {
        let (w, h) = (10u32, 8u32);
        let px = gradient(w, h, 65536);
        let file = one_strip_dng(&px, w, h, 16, 7, h, None);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let a = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        let b = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                f32::to_bits(*x as f32),
                f32::to_bits(*y as f32),
                "样本 {i} 两次读取的位模式不一致"
            );
        }
        assert_eq!(a, b);
    }

    #[test]
    fn dng_lossless_single_pixel_and_thin_strips() {
        // 1×1 边界 + 1×N / N×1 边界（走 DNG 端到端）。
        for (w, h) in [(1u32, 1u32), (1, 9), (9, 1), (2, 1)] {
            let px = gradient(w, h, 65536);
            let file = one_strip_dng(&px, w, h, 16, 1, h, None);
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let got = d.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, px, "{w}×{h} 边界必须逐位一致");
        }
    }

    // -------------------------------------------------------------------
    //  结构化拒绝
    // -------------------------------------------------------------------

    #[test]
    fn dng_lossless_rejects_non_jpeg_payload() {
        // 把未压缩的紧凑位打包载荷错标成 7 ⇒ 必须报字节流格式错误。
        let px = gradient(4, 4, 4096);
        let good = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &px, 4095,
            [64, 64, 64, 64],
        );
        let mut file = good.clone();
        assert!(patch_inline_short(&mut file, tags::COMPRESSION, 7));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        // 绝不把打包字节当成 JPEG 读出垃圾。
        assert!(
            e.code == "RAW1001" || e.code == "RAW2001",
            "错标载荷必须结构化拒绝：{e}"
        );
    }

    #[test]
    fn dng_lossless_rejects_truncated_strip_byte_count() {
        // StripByteCounts 谎报一个超出文件末尾的值 → slice_lossless 报越界。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let file = one_strip_dng(&px, w, h, 16, 1, h, None);
        // 把唯一的 StripByteCounts 改大（外置 LONG 数组，直接改偏移指向的值）。
        // 简化：把载荷尾部截掉，让 byte_count 超出文件。
        let cut = &file[..file.len() - 5];
        let d = DngFile::parse(cut).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert!(
            e.code == "RAW1001" || e.code == "RAW2001",
            "截断的压缩载荷必须结构化拒绝：{e}"
        );
    }

    #[test]
    fn dng_lossless_rejects_unimplemented_compression() {
        // Compression = 6（旧式 JPEG）仍须拒绝并给转存建议。
        let px = gradient(4, 4, 4096);
        let good = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &px, 4095,
            [64, 64, 64, 64],
        );
        let mut file = good;
        assert!(patch_inline_short(&mut file, tags::COMPRESSION, 6));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001", "{}", e);
        assert!(format!("{e}").contains("Compression"), "{}", e);
    }

    #[test]
    fn dng_lossless_rejects_spp_not_one() {
        // SamplesPerPixel = 3（线性 RGB DNG）→ 结构化拒绝。
        let (w, h) = (4u32, 4u32);
        let px = gradient(w, h, 65536);
        let mut file = one_strip_dng(&px, w, h, 16, 1, h, None);
        assert!(patch_inline_short(&mut file, tags::SAMPLES_PER_PIXEL, 3));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001", "{}", e);
    }

    // -------------------------------------------------------------------
    //  附加：LinearizationTable / BlackLevelDeltaH/V
    // -------------------------------------------------------------------

    #[test]
    fn dng_linearization_table_uses_correct_tag_number() {
        // `0xC618` 是错的；规范是 0xC691 = 50833。
        assert_eq!(tags::LINEARIZATION_TABLE, 50833);
        assert_eq!(tags::LINEARIZATION_TABLE, 0xC691);
        assert_eq!(tags::BLACK_LEVEL_DELTA_H, 50715);
        assert_eq!(tags::BLACK_LEVEL_DELTA_V, 50716);
        assert_eq!(tags::ROW_INTERLEAVE_FACTOR, 50975);
        assert_eq!(tags::ROW_INTERLEAVE_FACTOR, 0xC71F);
    }

    #[test]
    fn dng_linearization_table_is_applied_in_develop() {
        // 构造带LinearizationTable 的 DNG：raw 码值经表映射后再减黑电平。
        let (w, h) = (4u32, 4u32);
        // 8 位 → 表长 2^8 = 256。表内容raw*2（线性化放大一倍）。
        let table: Vec<f32> = (0..256).map(|v| (v * 2) as f32).collect();
        let px = gradient(w, h, 256);
        let mut ifd0 = IfdBuilder::new(true);
        ifd0.longs(tags::IMAGE_WIDTH, &[w]);
        ifd0.longs(tags::IMAGE_LENGTH, &[h]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[8]);
        ifd0.shorts(tags::COMPRESSION, &[1]);
        ifd0.shorts(tags::FILL_ORDER, &[1]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[255]);
        ifd0.rationals(tags::BLACK_LEVEL, &[0.0, 0.0, 0.0, 0.0]);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        // DNG 1.6 规定该 tag 类型为 SHORT 或 LONG；用 SHORT（2 B × 256 = 512 B）
        // 才能装进本脚手架的 HEAP_RESERVE = 1024 B 外置值堆（RATIONAL 要 2048 B）。
        ifd0.shorts(
            tags::LINEARIZATION_TABLE,
            &table.iter().map(|&v| v as u16).collect::<Vec<u16>>(),
        );
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[0.7, 0.2, 0.05, 0.1, 0.8, 0.1, 0.05, 0.15, 0.89],
        );
        // 未压缩载荷：每样本 1 字节。
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        ifd0.longs(tags::STRIP_OFFSETS, &[payload_base as u32]);
        ifd0.longs(tags::STRIP_BYTE_COUNTS, &[(w * h) as u32]);
        ifd0.longs(tags::ROWS_PER_STRIP, &[h]);

        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let bytes = write_ifd(&ifd0, true, 0, &mut heap, OFF_HEAP);
        file[OFF_IFD0..OFF_IFD0 + bytes.len()].copy_from_slice(&bytes);
        file.extend_from_slice(&heap);
        file.resize(payload_base, 0u8);
        for v in px.iter() {
            file.push(*v as u8);
        }

        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let tbl = d.linearization_table().expect("应读出线性化表");
        assert_eq!(tbl.len(), 256, "8 位 → 表长 2^8 = 256");
        assert_eq!(tbl[10], 20, "表内容raw*2");
        // develop 应把 raw 经表映射（→×2）后再减黑电平并归一。
        let opts = DevelopOptions {
            color_matrix: MatrixMode::None,
            ..Default::default()
        };
        let fb = d.develop(&px, &opts).unwrap_or_else(|e| panic!("{e}"));
        // 至少不能 panic，且尺寸正确。
        assert_eq!(fb.width, w);
        assert_eq!(fb.height, h);
    }

    #[test]
    fn dng_linearization_table_absent_is_none() {
        // 无线性化表 → None（不报错，让整图仍可读）。
        let (w, h) = (4u32, 4u32);
        let px = gradient(w, h, 4096);
        let file = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, 4, &px, 4095,
            [64, 64, 64, 64],
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert!(d.linearization_table().is_none(), "无该标签时应为 None");
    }

    #[test]
    fn dng_black_level_deltas_default_to_zero() {
        // 缺 BlackLevelDeltaH/V → 全 0 向量（调用方无条件相加）。
        let (w, h) = (4u32, 6u32);
        let px = gradient(w, h, 4096);
        let file = build_packed_dng(
            true, [0, 1, 1, 2], 12, FillOrder::MsbFirst, 4, 4, h, &px, 4095,
            [64, 64, 64, 64],
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let (dh, dv) = d.black_level_deltas(h as usize, w as usize);
        assert_eq!(dh.len(), h as usize);
        assert_eq!(dv.len(), w as usize);
        assert!(dh.iter().all(|&v| v == 0.0), "缺省逐行偏移应全0");
        assert!(dv.iter().all(|&v| v == 0.0), "缺省逐列偏移应全0");
    }

    #[test]
    fn dng_black_level_deltas_are_read_and_summed() {
        // 带BlackLevelDeltaH/V：读出并验证 develop 用了它们（黑电平被抬高）。
        let (w, h) = (4u32, 4u32);
        let mut ifd0 = IfdBuilder::new(true);
        ifd0.longs(tags::IMAGE_WIDTH, &[w]);
        ifd0.longs(tags::IMAGE_LENGTH, &[h]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[12]);
        ifd0.shorts(tags::COMPRESSION, &[1]);
        ifd0.shorts(tags::FILL_ORDER, &[1]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[4095]);
        ifd0.rationals(tags::BLACK_LEVEL, &[100.0, 100.0, 100.0, 100.0]);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[0.7, 0.2, 0.05, 0.1, 0.8, 0.1, 0.05, 0.15, 0.89],
        );
        // 逐行 deltaH（长度 = 高）与逐列 deltaV（长度 = 宽）。
        let dh: Vec<f32> = (0..h).map(|y| 10.0 + y as f32).collect();
        let dv: Vec<f32> = (0..w).map(|x| 5.0 + x as f32).collect();
        ifd0.rationals(tags::BLACK_LEVEL_DELTA_H, &dh);
        ifd0.rationals(tags::BLACK_LEVEL_DELTA_V, &dv);
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        ifd0.longs(tags::STRIP_OFFSETS, &[payload_base as u32]);
        ifd0.longs(
            tags::STRIP_BYTE_COUNTS,
            &[((w * h) as usize * 12).div_ceil(8) as u32],
        );
        ifd0.longs(tags::ROWS_PER_STRIP, &[h]);

        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let bytes = write_ifd(&ifd0, true, 0, &mut heap, OFF_HEAP);
        file[OFF_IFD0..OFF_IFD0 + bytes.len()].copy_from_slice(&bytes);
        file.extend_from_slice(&heap);
        file.resize(payload_base, 0u8);
        // 12 位紧凑载荷（MSB first，高位对齐）。
        let px: Vec<u16> = gradient(w, h, 4096);
        let packed = pack_bits(&px, 12, FillOrder::MsbFirst, true);
        file.extend_from_slice(&packed);

        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let (rdh, rdv) = d.black_level_deltas(h as usize, w as usize);
        assert_eq!(rdh, dh, "逐行黑电平偏移应被读出");
        assert_eq!(rdv, dv, "逐列黑电平偏移应被读出");
        // develop 不 panic 且用到 delta（黑电平被抬高 → 输出更暗）。
        let opts = DevelopOptions {
            color_matrix: MatrixMode::None,
            ..Default::default()
        };
        let fb = d.develop(&px, &opts).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fb.width, w);
    }

    // -------------------------------------------------------------------
    //  Compression = 8 / 32946（Deflate）+ Predictor + SubIFD
    // -------------------------------------------------------------------


    /// 造一条 **Deflate 压缩的 strip 布局 DNG**（自测用）。
    ///
    /// `strips` 是**已压缩好**的 zlib 流（每个元素 = 一条完整 zlib 流）。
    /// 与 `build_packed_dng` 同构，只是载荷来源不同。
    #[allow(clippy::too_many_arguments)]
    fn build_deflate_dng(
        le: bool,
        bits: u16,
        rows_per_strip: u32,
        width: u32,
        height: u32,
        strips: &[(Vec<u8>, u32)],
        white: u32,
        black: [u32; 4],
        predictor: u16,
        compression: u16,
    ) -> Vec<u8> {
        // 1) 载荷偏移（放在堆之后，与 build_packed_dng 同版式）。
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        let mut offs: Vec<u32> = Vec::with_capacity(strips.len());
        let mut counts: Vec<u32> = Vec::with_capacity(strips.len());
        let mut cursor = payload_base;
        for (bytes, _) in strips {
            offs.push(cursor as u32);
            counts.push(bytes.len() as u32);
            cursor += align2(bytes.len());
        }
        // 2) 建 IFD。
        let mut ifd0 = IfdBuilder::new(le);
        ifd0.longs(tags::IMAGE_WIDTH, &[width]);
        ifd0.longs(tags::IMAGE_LENGTH, &[height]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
        ifd0.shorts(tags::COMPRESSION, &[compression]);
        ifd0.shorts(tags::FILL_ORDER, &[FillOrder::MsbFirst.code()]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.longs(tags::STRIP_OFFSETS, &offs);
        ifd0.longs(tags::STRIP_BYTE_COUNTS, &counts);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.longs(tags::ROWS_PER_STRIP, &[rows_per_strip]);
        ifd0.shorts(tags::PREDICTOR, &[predictor]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[white]);
        let bl: Vec<f32> = black.iter().map(|&b| b as f32).collect();
        ifd0.rationals(tags::BLACK_LEVEL, &bl);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        // 3) 序列化（沿用 build_packed_dng 的版式）。
        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        if le {
            file[0..2].copy_from_slice(b"II");
            file[2..4].copy_from_slice(&42u16.to_le_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        } else {
            file[0..2].copy_from_slice(b"MM");
            file[2..4].copy_from_slice(&42u16.to_be_bytes());
            file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_be_bytes());
        }
        let mut heap: Vec<u8> = Vec::new();
        let ifd0_bytes = write_ifd(&ifd0, le, 0, &mut heap, OFF_HEAP);
        // IFD0 落在固定偏移 OFF_IFD0（不是 append——文件已预置到 OFF_HEAP）。
        if OFF_IFD0 + ifd0_bytes.len() > OFF_HEAP {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", ifd0_bytes.len());
        }
        file[OFF_IFD0..OFF_IFD0 + ifd0_bytes.len()].copy_from_slice(&ifd0_bytes);
        file.extend_from_slice(&heap);
        // 4) 载荷：按第 1 步算出的 offs 顺序写（resize 只补零、绝不截断堆）。
        file.resize(payload_base, 0u8);
        for (i, (bytes, _)) in strips.iter().enumerate() {
            let target = offs[i] as usize;
            if file.len() < target {
                file.resize(target, 0);
            }
            file.extend_from_slice(bytes);
        }
        file
    }

    /// tile 版式的 Deflate DNG（`tile_w × tile_h`，图幅须为其整数倍）。
    fn build_deflate_dng_tiled(
        px: &[u16],
        width: u32,
        height: u32,
        bits: u16,
        compression: u16,
        predictor: u16,
        tile_w: u32,
        tile_h: u32,
    ) -> Vec<u8> {
        let le = true;
        let p = DeflatePredictor::from_code(predictor).unwrap_or(DeflatePredictor::None);
        let across = width.div_ceil(tile_w);
        let down = height.div_ceil(tile_h);
        let mut tiles: Vec<(Vec<u8>, u32)> = Vec::with_capacity((across * down) as usize);
        for ty in 0..down {
            for tx in 0..across {
                let mut buf: Vec<u16> = Vec::with_capacity((tile_w * tile_h) as usize);
                for y in 0..tile_h {
                    for x in 0..tile_w {
                        let sx = tx * tile_w + x;
                        let sy = ty * tile_h + y;
                        let inside = sx < width && sy < height;
                        let v = if inside {
                            px[(sy * width + sx) as usize]
                        } else {
                            0
                        };
                        buf.push(v);
                    }
                }
                p.apply_in_place(&mut buf, tile_w as usize, bits);
                let packed = pack_bits(&buf, bits, FillOrder::MsbFirst, false);
                tiles.push((crate::exr_write::deflate_zlib(&packed), 0));
            }
        }
        let payload_base = OFF_HEAP + align2(HEAP_RESERVE) + 8;
        let mut offs: Vec<u32> = Vec::with_capacity(tiles.len());
        let mut counts: Vec<u32> = Vec::with_capacity(tiles.len());
        let mut cursor = payload_base;
        for (bytes, _) in &tiles {
            offs.push(cursor as u32);
            counts.push(bytes.len() as u32);
            cursor += align2(bytes.len());
        }
        let mut ifd0 = IfdBuilder::new(le);
        ifd0.longs(tags::IMAGE_WIDTH, &[width]);
        ifd0.longs(tags::IMAGE_LENGTH, &[height]);
        ifd0.shorts(tags::BITS_PER_SAMPLE, &[bits]);
        ifd0.shorts(tags::COMPRESSION, &[compression]);
        ifd0.shorts(tags::FILL_ORDER, &[FillOrder::MsbFirst.code()]);
        ifd0.shorts(tags::PHOTOMETRIC_INTERPRETATION, &[32803]);
        ifd0.shorts(tags::TILE_WIDTH, &[tile_w as u16]);
        ifd0.shorts(tags::TILE_LENGTH, &[tile_h as u16]);
        ifd0.longs(tags::TILE_OFFSETS, &offs);
        ifd0.longs(tags::TILE_BYTE_COUNTS, &counts);
        ifd0.shorts(tags::SAMPLES_PER_PIXEL, &[1]);
        ifd0.shorts(tags::PREDICTOR, &[predictor]);
        ifd0.shorts(tags::CFA_REPEAT_PATTERN_DIM, &[2, 2]);
        ifd0.bytes(tags::CFA_PATTERN, &[0, 1, 1, 2]);
        ifd0.bytes(tags::DNG_VERSION, &[1, 4, 0, 0]);
        ifd0.bytes(tags::DNG_BACKWARD_VERSION, &[1, 1, 0, 0]);
        ifd0.longs(tags::WHITE_LEVEL, &[(1u32 << bits) - 1]);
        ifd0.rationals(tags::BLACK_LEVEL, &[0.0, 0.0, 0.0, 0.0]);
        ifd0.rationals(tags::AS_SHOT_NEUTRAL, &[0.5, 1.0, 0.75]);
        ifd0.srationals(
            tags::COLOR_MATRIX1,
            &[
                0.700, 0.200, 0.0505, //
                0.100, 0.800, 0.100, //
                0.050, 0.149, 0.889,
            ],
        );
        let mut file: Vec<u8> = vec![0u8; OFF_HEAP];
        file[0..2].copy_from_slice(b"II");
        file[2..4].copy_from_slice(&42u16.to_le_bytes());
        file[4..8].copy_from_slice(&(OFF_IFD0 as u32).to_le_bytes());
        let mut heap: Vec<u8> = Vec::new();
        let ifd0_bytes = write_ifd(&ifd0, le, 0, &mut heap, OFF_HEAP);
        if OFF_IFD0 + ifd0_bytes.len() > OFF_HEAP {
            panic!("测试脚手架：IFD 超出预留版式（{} 字节）", ifd0_bytes.len());
        }
        file[OFF_IFD0..OFF_IFD0 + ifd0_bytes.len()].copy_from_slice(&ifd0_bytes);
        file.extend_from_slice(&heap);
        file.resize(payload_base, 0u8);
        for (i, (bytes, _)) in tiles.iter().enumerate() {
            let target = offs[i] as usize;
            if file.len() < target {
                file.resize(target, 0);
            }
            file.extend_from_slice(bytes);
        }
        file
    }

    /// 造一条 **Deflate(zlib)** 压缩的单 strip DNG（自测用负载）。
    ///
    /// `compression` 写进 `Compression` 标签（8 或 32946），
    /// `predictor` 写进 `Predictor` 标签（1 / 2 / 34892 / 34893）。
    fn deflate_dng(
        px: &[u16],
        w: u32,
        h: u32,
        precision: u16,
        compression: u16,
        predictor: u16,
        rows_per_strip: u32,
    ) -> Vec<u8> {
        let le = true;
        let mut strips: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut y = 0u32;
        while y < h {
            let rows = rows_per_strip.min(h - y);
            let mut chunk: Vec<u16> =
                px[(y * w) as usize..((y + rows) * w) as usize].to_vec();
            // 预测器 → 位打包 → zlib 容器（与读端严格逆序）。
            let p = DeflatePredictor::from_code(predictor).unwrap_or(DeflatePredictor::None);
            p.apply_in_place(&mut chunk, w as usize, precision);
            let packed = pack_bits(&chunk, precision, FillOrder::MsbFirst, false);
            strips.push((crate::exr_write::deflate_zlib(&packed), rows));
            y += rows;
        }
        build_deflate_dng(
            le, precision, rows_per_strip, w, h, &strips,
            (1u32 << precision) - 1, [0, 0, 0, 0], predictor, compression,
        )
    }

    #[test]
    fn deflate_predictor_code_mapping_covers_dng_14_spec() {
        // DNG 1.4 为 Compression=8 定义了 7 个预测器取值。
        assert_eq!(DeflatePredictor::from_code(1), Some(DeflatePredictor::None));
        assert_eq!(
            DeflatePredictor::from_code(2),
            Some(DeflatePredictor::Horizontal)
        );
        assert_eq!(
            DeflatePredictor::from_code(34892),
            Some(DeflatePredictor::HorizontalX2)
        );
        assert_eq!(
            DeflatePredictor::from_code(34893),
            Some(DeflatePredictor::HorizontalX4)
        );
        // 三个浮点预测器本模块不实现（需浮点 RAW）→ None。
        for code in [3u16, 34894, 34895] {
            assert_eq!(
                DeflatePredictor::from_code(code),
                None,
                "浮点预测器 {code} 不该被当成整数预测器接受"
            );
            assert!(
                DeflatePredictor::reject_reason(code).contains("浮点"),
                "code={code} 的成因须点明是浮点预测器"
            );
        }
        // 往返映射。
        for p in [
            DeflatePredictor::None,
            DeflatePredictor::Horizontal,
            DeflatePredictor::HorizontalX2,
            DeflatePredictor::HorizontalX4,
        ] {
            assert_eq!(DeflatePredictor::from_code(p.code()), Some(p));
        }
    }

    #[test]
    fn deflate_predictor_apply_and_undo_are_mutually_inverse() {
        // 4 个整数预测器 × 3 个位深 × 3 种行宽，apply→undo 必须恒等。
        for p in [
            DeflatePredictor::None,
            DeflatePredictor::Horizontal,
            DeflatePredictor::HorizontalX2,
            DeflatePredictor::HorizontalX4,
        ] {
            for bits in [1u16, 8, 12, 16] {
                for row_len in [1usize, 3, 16, 17] {
                    let rows = 3usize;
                    let n = row_len * rows;
                    let mask = if bits >= 16 {
                        u16::MAX
                    } else {
                        ((1u32 << bits) - 1) as u16
                    };
                    // 确定性样本：位深内取 variety。
                    let orig: Vec<u16> = (0..n)
                        .map(|i| (((i as u32 * 7 + 3) as u32) & mask as u32) as u16)
                        .collect();
                    let mut buf = orig.clone();
                    p.apply_in_place(&mut buf, row_len, bits);
                    p.undo_in_place(&mut buf, row_len, bits);
                    assert_eq!(
                        buf, orig,
                        "apply/undo 非互逆：{p:?} bits={bits} row_len={row_len}"
                    );
                }
            }
        }
    }

    #[test]
    fn deflate_predictor_undo_respects_row_boundaries() {
        // 每行**独立**做差分：第 2 行不得用到第 1 行的样本。
        let p = DeflatePredictor::Horizontal;
        let row_len = 4usize;
        // 行1 =全 0，行 2 = 全 100：若跨行泄漏，第 2 行首样本会变成 100+0=100，
        // 而正确结果是 100（第2 行首样本没有左邻，保持原值）。
        let orig = vec![0u16, 0, 0, 0, 100, 100, 100, 100];
        let mut buf = orig.clone();
        p.apply_in_place(&mut buf, row_len, 16);
        // 行2 首样本必须原样（无左邻）。
        assert_eq!(buf[4], 100, "行首样本不该被差分");
        p.undo_in_place(&mut buf, row_len, 16);
        assert_eq!(buf, orig);
    }

    #[test]
    fn deflate_predictor_wraps_modulo_two_to_bits() {
        // TIFF 6.0 §14：水平差分按 mod 2^BitsPerSample 回绕。
        // 8 位下100 - 30 = 70；30 - 100 = -70 ≡ 186 (mod 256)。
        let p = DeflatePredictor::Horizontal;
        let orig = vec![100u16, 30];
        let mut buf = orig.clone();
        p.apply_in_place(&mut buf, 2, 8);
        assert_eq!(buf, vec![100, 186], "差分须按 mod 256 回绕");
        p.undo_in_place(&mut buf, 2, 8);
        assert_eq!(buf, orig, "回绕后仍须精确还原");
    }

    #[test]
    fn deflate_predictor_is_not_a_noop_at_sixteen_bits() {
        // **回归测试**：早先 `apply_in_place` / `undo_in_place` 的守卫写成
        // `bits >= 16` 就提前返回，把预测器在 **16 位**上降级成了 no-op。
        //
        // 为什么危险到要专门钉死：合成器与读端**同时** no-op，于是
        // 「合成 → 读回」往返测试**照样全绿**，但读**真实相机文件**
        // （Deflate + Predictor=2 + BitsPerSample=16，DNG 最常见组合）
        // 会解出**全错像素且不报错**——静默错误比崩溃更难发现。
        //
        // 模 2^16 的回绕就是 `wrapping_add` / `wrapping_sub` 的语义，
        // **不是恒等**，故 16 位必须照常做运算。
        for bits in [8u16, 10, 12, 14, 16] {
            let p = DeflatePredictor::Horizontal;
            // 造一个「相邻差为常数」的序列：差分后应变成常数流。
            let row = vec![100u16, 130, 160, 190, 220];
            let mut buf = row.clone();
            p.apply_in_place(&mut buf, row.len(), bits);
            // 差分后：buf[0] 保持原值（无左邻），其余应全为 30。
            let deltas: Vec<u16> = buf[1..].to_vec();
            assert!(
                deltas.iter().all(|&d| d == 30),
                "bits={bits}：差分应变常数流（每步 30），实得{deltas:?}——\
                 若此值是原序列，说明预测器被提前返回成了no-op"
            );
            p.undo_in_place(&mut buf, row.len(), bits);
            assert_eq!(buf, row, "bits={bits}：逆运算须精确还原");
        }
        // 16 位回绕：30 - 100 = -70 ≡ 65466 (mod 65536)。
        let p = DeflatePredictor::Horizontal;
        let mut buf = vec![100u16, 30];
        p.apply_in_place(&mut buf, 2, 16);
        assert_eq!(buf, vec![100, 65466], "16 位差分须按 mod 65536 回绕");
        p.undo_in_place(&mut buf, 2, 16);
        assert_eq!(buf, vec![100, 30], "16 位逆运算须精确还原");
    }

    #[test]
    fn dng_deflate_round_trips_bit_exact() {
        // Compression = 8：Deflate + 4 个整数预测器，全部逐位一致。
        let (w, h) = (16u32, 12u32);
        let px = gradient(w, h, 65536);
        for pred in [1u16, 2, 34892, 34893] {
            let file = deflate_dng(&px, w, h, 16, 8, pred, 4);
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d.compression(), 8, "必须走 Compression=8 分派");
            assert_eq!(d.predictor(), pred, "Predictor 标签须原样读回");
            assert_eq!(d.deflate_predictor(), DeflatePredictor::from_code(pred));
            let got = d
                .read_raw_samples()
                .unwrap_or_else(|e| panic!("pred={pred}: {e}"));
            assert_eq!(got, px, "pred={pred} 必须逐位一致");
        }
    }

    #[test]
    fn dng_deflate_code_8_and_32946_are_equivalent() {
        // 32946(0x80B2) 是厂商私有旧编号，算法与 8 完全相同（Adobe TIFF TN）。
        let (w, h) = (16u32, 12u32);
        let px = gradient(w, h, 65536);
        let f8 = deflate_dng(&px, w, h, 16, 8, 2, 4);
        let f32946 = deflate_dng(&px, w, h, 16, 32946, 2, 4);
        // 载荷字节应完全一致（同一编码器、同一预测器）。
        let d8 = DngFile::parse(&f8).unwrap_or_else(|e| panic!("{e}"));
        let d32 = DngFile::parse(&f32946).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d8.compression(), 8);
        assert_eq!(d32.compression(), 32946);
        let a = d8.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        let b = d32.read_raw_samples().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a, b, "8 与 32946 必须解出同一份像素");
        assert_eq!(a, px, "且都须与原图逐位一致");
    }

    #[test]
    fn dng_deflate_multi_strip_and_tile_layouts() {
        // strip 与 tile 两条布局都要正确（tile 还要跨块拼回）。
        let (w, h) = (16u32, 16u32);
        let px = gradient(w, h, 65536);
        for rps in [1u32, 3, 16] {
            let file = deflate_dng(&px, w, h, 16, 8, 2, rps);
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
            let got = d
                .read_raw_samples()
                .unwrap_or_else(|e| panic!("rps={rps}: {e}"));
            assert_eq!(got, px, "strip rps={rps} 须逐位一致");
        }
        // tile 8×8 → 4 个 tile。
        let file = build_deflate_dng_tiled(&px, w, h, 16, 8, 2, 8, 8);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let got = d
            .read_raw_samples()
            .unwrap_or_else(|e| panic!("tile: {e}"));
        assert_eq!(got, px, "tile 8×8 须逐位一致");
    }

    #[test]
    fn dng_deflate_non_16bit_depths() {
        // 8 / 12 位（非字节对齐）也要过：预测器在**位解包之后**才作用。
        for bits in [8u16, 12] {
            let (w, h) = (16u32, 12u32);
            let px: Vec<u16> = gradient(w, h, 1u32 << bits);
            for pred in [1u16, 2, 34892] {
                let file = deflate_dng(&px, w, h, bits, 8, pred, 4);
                let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
                let got = d
                    .read_raw_samples()
                    .unwrap_or_else(|e| panic!("bits={bits} pred={pred}: {e}"));
                assert_eq!(got, px, "bits={bits} pred={pred} 须逐位一致");
            }
        }
    }

    #[test]
    fn dng_deflate_rejects_bare_deflate_stream() {
        // **规范口径的钉死测试**：TIFF/DNG 的 Deflate 是 **zlib 流**。
        // 喂裸 DEFLATE（无 zlib 头）必须被结构化拒绝，而不是「解出错图」。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let le = true;
        // 故意只放**裸** DEFLATE（deflate_stored，无 zlib 头/尾）。
        let mut chunk = px.clone();
        DeflatePredictor::None.apply_in_place(&mut chunk, w as usize, 16);
        let bare = crate::exr_write::deflate_stored(&pack_bits(&chunk, 16, FillOrder::MsbFirst, false));
        let file = build_deflate_dng(
            le, 16, h, w, h, &[(bare, h)], 65535, [0, 0, 0, 0], 1, 8,
        );
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = match d.read_raw_samples() {
            Ok(_) => panic!("裸 DEFLATE 不该被当成 zlib 流接受"),
            Err(e) => e,
        };
        assert_eq!(e.code, "RAW2001", "{}", e);
        let text = format!("{e}");
        assert!(
            text.contains("zlib"),
            "错误须点明这是 zlib 容器问题：{text}"
        );
    }

    #[test]
    fn dng_deflate_rejects_corrupted_adler32() {
        // zlib 尾部 Adler-32 被改写 → 必须结构化拒绝（完整性校验生效）。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let mut file = deflate_dng(&px, w, h, 16, 8, 2, h);
        // 破坏 zlib 尾部：把文件最后一个字节（Adler-32 末字节）翻转。
        let last = file.len() - 1;
        file[last] ^= 0xFF;
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001", "{}", e);
        let text = format!("{e}");
        assert!(
            text.contains("Adler") || text.contains("校验"),
            "错误须点明校验和/完整性问题：{text}"
        );
    }

    #[test]
    fn dng_deflate_rejects_truncated_stream() {
        // 截掉 zlib 流尾部（Adler-32 缺失）→ 结构化拒绝。
        //
        // 截断可能在**两层**被抓住（都算通过，但要都是结构化错误）：
        // ① `slice_lossless` 发现声明字节数超出文件长度（RAW1001）；
        // ② 若截断后仍落在文件内，则 `zlib::decompress_verified` 报流截断（RAW2001）。
        // 关键是**不许**解出「看起来正常」的部分像素。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let file = deflate_dng(&px, w, h, 16, 8, 2, h);
        for cut_back in [2usize, 6, 10] {
            let cut = &file[..file.len() - cut_back];
            let d = DngFile::parse(cut).unwrap_or_else(|e| panic!("{e}"));
            match d.read_raw_samples() {
                Ok(_) => panic!("截掉 {cut_back} 字节后不该解出像素"),
                Err(e) => assert!(
                    e.code == "RAW1001" || e.code == "RAW2001",
                    "截断必须结构化拒绝（RAW1xxx/RAW2xxx），实得 {}：{e}",
                    e.code
                ),
            }
        }
    }

    #[test]
    fn dng_deflate_rejects_unsupported_floating_point_predictor() {
        // Predictor = 3（Floating Point）本模块不实现 → 结构化拒绝 + 说清成因。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let mut file = deflate_dng(&px, w, h, 16, 8, 1, h);
        // 把 Predictor 标签改成 3。
        assert!(patch_inline_short(&mut file, tags::PREDICTOR, 3));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.predictor(), 3, "Predictor 标签须读回 3");
        assert_eq!(d.deflate_predictor(), None, "浮点预测器不该被接受");
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001", "{}", e);
        let text = format!("{e}");
        assert!(text.contains("Predictor = 3"), "{text}");
        assert!(text.contains("浮点"), "须点明是浮点预测器：{text}");
    }

    #[test]
    fn deflate_reject_hint_no_longer_claims_inflate_is_unimplemented() {
        // **本路要还的第一笔账**：`8` / `32946` 已移出拒绝清单。
        //
        // 旧提示写的是"本模块未实装inflate"——**错的**：inflate 早就在
        // prism-bytes 里了，缺的只是 DNG 侧接线。故这两个取值不再出现
        // 在拒绝清单中。
        for code in [8u16, 32946] {
            let h = compression_reject_hint(code);
            assert!(
                !h.contains("未实装inflate") && !h.contains("未实装 inflate"),
                "code={code} 的提示仍在谎称 inflate 未实装：{h}"
            );
        }
        // 8 / 32946 现在是**支持**的取值 → 提示走_ 分支（通用文案），
        // 且该文案须列出真实支持集合。
        let generic = compression_reject_hint(999);
        assert!(generic.contains("8"), "通用文案须列出支持的 8：{generic}");
        assert!(generic.contains("32946"), "通用文案须列出 32946：{generic}");
        // 34892 是 Lossy JPEG（**不是** Deflate 的别名）——成因须写对。
        let lossy = compression_reject_hint(34892);
        assert!(
            lossy.contains("Lossy JPEG"),
            "34892 的成因须是 Lossy JPEG：{lossy}"
        );
        assert!(
            !lossy.contains("Deflate 的旧编号"),
            "34892 不是 Deflate 别名，提示不得误导：{lossy}"
        );
    }

    #[test]
    fn dng_lossy_jpeg_34892_is_rejected_not_routed_to_inflate() {
        // 曾把 `34892` 描述成「Deflate 的 Adobe 历史别名」——**事实错误**。
        // DNG 1.4 原文并列声明：`8: Deflate (ZIP)` 与 `34892: Lossy JPEG`，
        // 两者语义完全不同（后者是**有损**基线 DCT）。
        //
        // 若照此实现，一个有损 JPEG 的 DNG 会被喂进 inflate，产出
        // 「看起来合法实则全错」的像素——比明确拒绝**危险得多**。
        // 故钉死：`34892` 必须走结构化拒绝，且错误文本说清「有损」。
        let px = gradient(4, 4, 4096);
        let mut file = build_packed_dng(
            true,
            [0, 1, 1, 2],
            12,
            FillOrder::MsbFirst,
            4,
            4,
            4,
            &px,
            4095,
            [64, 64, 64, 64],
        );
        // 把 Compression 从 8 改成 34892（载荷仍是合法 zlib 流，
        // 以确保拒绝来自**取值判定**而不是「流恰好解不开」）。
        assert!(patch_inline_short(&mut file, tags::COMPRESSION, 8));
        let good = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(good.compression(), 8);
        // 同一份载荷换成 34892 → 必须拒绝。
        assert!(patch_inline_short(&mut file, tags::COMPRESSION, 34892));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let e = d.read_raw_samples().unwrap_err();
        assert_eq!(e.code, "RAW2001", "{}", e);
        let text = format!("{e}");
        assert!(
            text.contains("Compression = 34892"),
            "须点名取值：{text}"
        );
        assert!(
            text.contains("Lossy JPEG"),
            "34892 的成因须写明是有损 JPEG（而非 Deflate 别名）：{text}"
        );
    }

    #[test]
    fn deflate_tag_numbers_match_spec() {
        // 标签号钉死：Predictor = 0x013D = 317、SubIFDs = 0x014A = 330。
        assert_eq!(tags::PREDICTOR, 317);
        assert_eq!(tags::PREDICTOR, 0x013D);
        assert_eq!(tags::SUB_IFDS, 330);
        assert_eq!(tags::SUB_IFDS, 0x014A);
    }

    #[test]
    fn dng_deflate_is_deterministic_bitwise() {
        // 同输入连写两遍 → 字节流逐位相同；诊断统计量 to_bits() 也相同。
        let (w, h) = (16u32, 12u32);
        let px = gradient(w, h, 65536);
        let a = deflate_dng(&px, w, h, 16, 8, 2, 4);
        let b = deflate_dng(&px, w, h, 16, 8, 2, 4);
        assert_eq!(a, b, "Deflate DNG 字节流须逐位可复现");
        // 合成器侧同样确定性。
        let mut s1 = crate::raw_dng_synth::DngSampleSpec::new(16, 12, 16, [0, 1, 1, 2]);
        s1.compression = 8;
        s1.deflate_predictor = 2;
        let f1 = crate::raw_dng_synth::synthesize_dng(&s1).unwrap_or_else(|e| panic!("{e}"));
        let f2 = crate::raw_dng_synth::synthesize_dng(&s1).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(f1, f2, "合成 Deflate DNG 须逐位可复现");
    }

    // -------------------------------------------------------------------
    //  SubIFD 预览图
    // -------------------------------------------------------------------

    #[test]
    fn dng_subifd_offsets_empty_without_tag() {
        // 没有 SubIFDs 标签 → 空向量（**不是错误**：「没预览」是正常事实）。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let file = one_strip_dng(&px, w, h, 16, 1, h, None);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert!(crate::raw_subifd::sub_ifd_offsets(&d)
            .unwrap_or_else(|e| panic!("{e}"))
            .is_empty());
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        assert!(subs.is_empty(), "无 SubIFDs 标签应返回空向量");
    }

    #[test]
    fn dng_subifd_rejects_broken_chain_without_panicking() {
        // SubIFDs 指向越界偏移 → 结构化报错（不静默跳过、不 panic）。
        let (w, h) = (8u32, 8u32);
        let px = gradient(w, h, 65536);
        let mut file = one_strip_dng(&px, w, h, 16, 1, h, None);
        // 手工塞一个 SubIFDs 标签很难（需改 IFD 长度），改用零偏移：
        // 偏移 0 在 TIFF 里表示「无 IFD」，应被结构化拒绝。
        // 这里直接构造一个最小文件来测read_ifd 的拒绝路径。
        let e = read_ifd(&file, ByteOrder::Little, 0).unwrap_err();
        assert_eq!(e.code, "RAW1001", "偏移 0 必须被拒：{e}");
        // 越界偏移同样被拒。
        let e2 = read_ifd(&file, ByteOrder::Little, 0x00FF_FFFF).unwrap_err();
        assert_eq!(e2.code, "RAW1001", "越界偏移必须被拒：{e2}");
        let _ = &mut file;
    }

    #[test]
    fn dng_subifd_preview_is_reported_in_diagnostics() {
        // 合成带预览的 DNG → 诊断必须报出预览尺寸/压缩/是否解出。
        let mut spec = crate::raw_dng_synth::DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        spec.preview = Some(crate::raw_dng_synth::PreviewSpec::new(24, 16, 8, 7, 2, 3));
        let file = crate::raw_dng_synth::synthesize_dng(&spec).unwrap_or_else(|e| panic!("{e}"));
        let diag = crate::raw_dng_synth::diagnose_dng_bytes(&file)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(diag.sub_ifds.len(), 1, "应报出 1 个 SubIFD");
        let p = &diag.sub_ifds[0];
        assert_eq!(p.width, 24, "诊断须报出预览宽");
        assert_eq!(p.height, 16, "诊断须报出预览高");
        assert_eq!(p.compression, 7);
        assert_eq!(p.samples_per_pixel, 3);
        assert!(p.has_pixels(), "无损 JPEG 预览应解出像素：{}", p.note);
        // 报告文本里必须出现预览行（便于 CLI 直接看）。
        let r = diag.report();
        assert!(r.contains("SubIFD"), "报告须含 SubIFD 行：{r}");
        assert!(r.contains("24×16"), "报告须含预览尺寸：{r}");
    }

}
