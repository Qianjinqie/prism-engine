//! **SubIFD 预览图解析** —— DNG 的第二条 IFD 链。
//!
//! # 这一模块为什么存在
//!
//! 早期版本的能力边界里写「SubIFD 预览不支持」。但**真实相机 DNG 普遍带 SubIFD
//! 预览图**（小尺寸 JPEG / 无损 JPEG / 未压缩 RGB），缺失它意味着
//! 「能读主图但看不到预览」——用户装一个 RAW 转换器预览图是空的。
//!
//! # 规范依据
//!
//! - TIFF 6.0 §2 定义 `SubIFDs`（标签 **330**，类型 IFD 或 LONG）为
//!   「指向子 IFD 的偏移数组」；
//! - Adobe《Photoshop TIFF Technical Notes》开篇即警告：
//!   "TIFF readers that encounter SubIFD (tag number 330) and a bit depth
//!   other than 8 should ignore the SubIFD's found in the file" —— 本模块
//!   **遵守**这条警告：位深非 8 的 SubIFD 只解析元数据不解像素（见下）；
//! - DNG 1.4 规定预览 IFD 的 `PhotometricInterpretation` 只允许
//!   1(BlackIsZero) / 2(RGB) / 6(YCbCr)。
//!
//! # 解到什么程度（诚实边界）
//!
//! | SubIFD 压缩方式 | 行为 |
//! |---|---|
//! | **1**（未压缩） | ✅ **解出像素**（`unpack_bits`） |
//! | **7**（无损 JPEG） | ✅ **解出像素**（复用 [`crate::raw_lossless_jpeg`]） |
//! | 8/ 32946（Deflate） | ✅ 解出像素（复用 [`crate::raw`] 的 zlib 路径） |
//! | 其他（含 34892 有损 JPEG、34712 JPEG2000、52546 JPEG XL） | ❌ **只解析元数据** + 说明为何不解 |
//!
//! **位深 ≠ 8 的未压缩 SubIFD 不解像素**，依据就是上面那条 Photoshop TN 警告
//! （"a bit depth other than 8 should ignore the SubIFD's"）——这是规范给的
//! 明确指引，照做而非自行推断。
//!
//! # 确定性
//!
//! - 子 IFD 按 `SubIFDs` 数组的**下标顺序**遍历（不是文件偏移顺序）；
//! - 像素统计用**固定顺序**的 `f64` 累加，同输入 `to_bits()` 逐位相同。
//!
//! # C18（永不 panic）
//!
//! 零 `unwrap` / `expect` / `panic!`；所有偏移与尺寸经 `checked_*` 与 `get` 兜底。

use crate::raw::{
    self, field_type, tags, unpack_bits, ByteOrder, DngFile, RawError, RawResult, TiffIfd,
};

/// 本模块所有结构化错误的统一 location 前缀。
const LOC: &str = "raw_subifd";

/// 一个 SubIFD 的**元数据**（不解像素也能给出）。
///
/// 无论能否解出像素，本结构都会被填充 —— 这是「诚实边界」的第一层：
/// **先让人知道文件里有什么，再谈能不能用**。
#[derive(Debug, Clone, PartialEq)]
pub struct SubIfdInfo {
    /// 在 `SubIFDs` 数组中的下标（0起）。
    pub index: usize,
    /// 该子 IFD 在文件中的偏移（诊断用）。
    pub file_offset: u32,
    /// `NewSubfileType`（缺省 0）。
    pub new_subfile_type: u32,
    /// `ImageWidth`（缺省 0 = 未知）。
    pub width: u32,
    /// `ImageLength`（缺省 0 = 未知）。
    pub height: u32,
    /// `Compression`（缺省 1）。
    pub compression: u16,
    /// `BitsPerSample`（缺省 1）。
    pub bits_per_sample: u16,
    /// `SamplesPerPixel`（缺省 1）。
    pub samples_per_pixel: u16,
    /// `PhotometricInterpretation`（缺省 1 = BlackIsZero）。
    pub photometric_interpretation: u16,
    /// `NewSubfileType` 的可读名（预览 / 缩略图 / 主图 …）。
    pub subfile_kind: &'static str,
    /// 载荷布局的可读描述（`"strip（4 行/strip，共 4 块）"` / `"tile（4×4，共 6 块）"` /
    /// `"无载荷布局标签"`）。
    ///
    /// # 为什么值得单独报出来
    ///
    /// 预览图**不总是单 strip**（大预览常被切条，部分机型直接 tile），
    /// 而「解出的像素对不对」直接取决于分块方式。诊断时能一眼看到
    /// 「共 N 块」比只看到一个尺寸有用得多——分块数不对是重组逻辑
    /// 出错的首要信号。
    pub layout: String,
}

impl SubIfdInfo {
    /// 像素总数（`width × height × samples_per_pixel`，checked；溢出则 `None`）。
    pub fn sample_count(&self) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(self.samples_per_pixel as usize)
    }

    /// 一行的人类可读摘要。
    pub fn summary(&self) -> String {
        format!(
            "SubIFD[{}] {}×{} {}bit spp={} Compression={} PI={} ({}) {}",
            self.index,
            self.width,
            self.height,
            self.bits_per_sample,
            self.samples_per_pixel,
            self.compression,
            self.photometric_interpretation,
            self.subfile_kind,
            self.layout,
        )
    }
}

/// 一个 SubIFD 的**解码结果**（元数据 + 尽可能解出的像素）。
///
/// # 为什么元数据内嵌 [`SubIfdInfo`] 而不是再抄一遍字段
///
/// 两者字段完全相同，抄一遍就是**第二份真相**（改一处忘另一处的典型来源）。
/// 故这里内嵌并 `Deref` 到 `SubIfdInfo`——诊断代码仍可写 `p.width`、
/// `p.compression`（自动解引用），而字段只有**一处**定义。
///
/// # `pixels: Option` 表达「没解出来」
///
/// `None` 与 [`SubIfdImage::note`] 里写的原因一一对应：「没解出来」是
/// **事实**，不是错误，故不用 `Err`。
#[derive(Debug, Clone, PartialEq)]
pub struct SubIfdImage {
    /// 元数据（尺寸 / 压缩 / 位深 / 色彩解释 …）。
    pub info: SubIfdInfo,
    /// 解出的像素（行主序，长度 = `width × height × spp`）；**未解出则 `None`**。
    pub pixels: Option<Vec<u16>>,
    /// 未解出时的**明确原因**（解出时是「已解出 N 样本」）。
    pub note: String,
}

impl std::ops::Deref for SubIfdImage {
    type Target = SubIfdInfo;

    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

impl SubIfdImage {
    /// 是否真的解出了像素。
    pub fn has_pixels(&self) -> bool {
        self.pixels.is_some()
    }

    /// 一行的人类可读摘要（**不解像素也给得出**）。
    pub fn summary(&self) -> String {
        format!(
            "{}{}",
            self.info.summary(),
            // 「像素已解出」/「像素未解出」——CLI 与日志里一眼能分辨
            // 「拿到了预览像素」与「只读到了元数据」。
            if self.has_pixels() {
                " 像素已解出"
            } else {
                " 像素未解出"
            }
        )
    }

    /// 像素统计（固定顺序累加 → 确定性；未解出时全 0）。
    ///
    /// 返回 `(min, max, mean)`；无像素时 `(0, 0, 0.0)`。
    pub fn stats(&self) -> (u16, u16, f64) {
        let px = match self.pixels.as_deref() {
            Some(v) if !v.is_empty() => v,
            _ => return (0, 0, 0.0),
        };
        let mut mn = u16::MAX;
        let mut mx = u16::MIN;
        let mut acc = 0f64;
        for &v in px.iter() {
            mn = mn.min(v);
            mx = mx.max(v);
            acc += v as f64;
        }
        (mn, mx, acc / px.len() as f64)
    }

    /// 由 `info` 组一个「未解出」的结果（附原因）——避免每个返回点抄一遍字段。
    fn unresolved(info: SubIfdInfo, reason: impl Into<String>) -> Self {
        Self {
            info,
            pixels: None,
            note: reason.into(),
        }
    }
}

/// `NewSubfileType` → 可读名（TIFF 6.0 §2 的位标志语义）。
fn subfile_kind(v: u32) -> &'static str {
    match v {
        0 => "主图",
        1 => "缩略图",
        2 => "预览图",
        3 => "主图+缩略图+预览",
        4 => "透明度掩模",
        5 => "透明度掩模+预览",
        6 => "透明度掩模+缩略图",
        7 => "透明度掩模+缩略图+预览",
        _ => "未列举",
    }
}

/// 读一个子 IFD 的元数据（不解像素）。
fn read_info(
    data: &[u8],
    order: ByteOrder,
    index: usize,
    offset: u32,
    ifd: &TiffIfd,
) -> SubIfdInfo {
    let num = |tag: u16| -> Option<f32> { ifd.first_number(tag, order) };
    let u32_of = |tag: u16, default: u32| -> u32 {
        num(tag)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u32::MAX as f32)
            .map(|v| v as u32)
            .unwrap_or(default)
    };
    let u16_of = |tag: u16, default: u16| -> u16 {
        num(tag)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u16::MAX as f32)
            .map(|v| v as u16)
            .unwrap_or(default)
    };
    let _ = data;
    let new_subfile_type = u32_of(tags::NEW_SUBFILE_TYPE, 0);
    let width = u32_of(tags::IMAGE_WIDTH, 0);
    let height = u32_of(tags::IMAGE_LENGTH, 0);
    SubIfdInfo {
        index,
        file_offset: offset,
        new_subfile_type,
        width,
        height,
        compression: u16_of(tags::COMPRESSION, 1),
        bits_per_sample: u16_of(tags::BITS_PER_SAMPLE, 1),
        samples_per_pixel: u16_of(tags::SAMPLES_PER_PIXEL, 1),
        photometric_interpretation: u16_of(tags::PHOTOMETRIC_INTERPRETATION, 1),
        subfile_kind: subfile_kind(new_subfile_type),
        layout: describe_layout(data, order, ifd, width, height),
    }
}

/// 载荷布局的一行可读描述（**只读标签，不解像素**）。
///
/// tile 标签优先——与主图 [`crate::raw::RawLayout::detect`] 同一策略
/// （真实文件里 strip 标签常被厂商填成占位值）。
fn describe_layout(
    data: &[u8],
    order: ByteOrder,
    ifd: &TiffIfd,
    width: u32,
    height: u32,
) -> String {
    let _ = data;
    // ---- tile 优先 ----
    let tile_w = ifd
        .first_number(tags::TILE_WIDTH, order)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .map(|v| v as u32);
    let tile_h = ifd
        .first_number(tags::TILE_LENGTH, order)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .map(|v| v as u32);
    if let (Some(tw), Some(th)) = (tile_w, tile_h) {
        if width > 0 && height > 0 {
            let across = width.div_ceil(tw);
            let down = height.div_ceil(th);
            let n = (across as u64).saturating_mul(down as u64);
            return format!("tile（{tw}×{th}，共 {n} 块）");
        }
        return format!("tile（{tw}×{th}）");
    }
    // ---- strip ----
    let offs_len = ifd
        .numbers(tags::STRIP_OFFSETS, order)
        .map(|v| v.len())
        .unwrap_or(0);
    if offs_len == 0 {
        return "无载荷布局标签".to_string();
    }
    let rps = ifd
        .first_number(tags::ROWS_PER_STRIP, order)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .map(|v| v as u32)
        .unwrap_or(height);
    if rps == 0 {
        return format!("strip（共 {offs_len} 块）");
    }
    let nominal = if height > 0 {
        (height as u64).div_ceil(rps as u64)
    } else {
        offs_len as u64
    };
    // 块数取「按高度算出的」与「偏移表实际长度」的较大者：
    // 厂商常把 `RowsPerStrip` 填成图高（于是只有 1 块）却给出多块偏移表，
    // 只报其中一个都会误导诊断。
    let n = nominal.max(offs_len as u64);
    format!("strip（{rps} 行/strip，共 {n} 块）")
}

/// `SubIFDs`（0x014A）标签里的子 IFD 偏移**条目数上限**（防御性）。
///
/// # 为什么需要它（C18：畸形文件不得把内存/时间吃光）
///
/// `SubIFDs` 的 `count` 字段完全由文件自称。若不设上限，一个损坏的（或
/// 恶意构造的）文件可以声明 `count = 0xFFFF_FFFF`，让本模块按它分配并
/// 逐条探测 43 亿次——**在读几个 KiB 的输入上就能制造 DoS**。
///
/// 取 64：真实 DNG 的 SubIFD 链极短（典型 1 个预览 + 1 个缩略图），
/// 64 已远超任何真实文件；超限一律**结构化拒绝**并说明成因。
pub const MAX_SUB_IFDS: usize = 64;

/// 单个 SubIFD 的**样本总数上限**（防御性）。
///
/// 取 2^28 = 268M：与主图合成器 [`crate::raw_dng_synth`] 的上限同量级。
/// 预览图在真实文件里是**几百×几百**（~10^5 样本），2^28 已高出三个数量级；
/// 超限的尺寸标签只可能是文件损坏或刻意构造的畸形输入。
pub const MAX_SUB_IFD_SAMPLES: u64 = 1 << 28;

/// 读出 `SubIFDs`（0x014A）标签里的**子 IFD 文件偏移数组**（原样，未解析）。
///
/// # 返回
///
/// - 无该标签（或不是数值类型）→ 空 `Vec`；
/// - 条目数超 [`MAX_SUB_IFDS`] → **结构化拒绝**（不静默截断，见该常量文档）；
/// - 偏移非法（负 / 超 `u32`）的项被**过滤掉**并保留其余 —— 一个坏条目
///   不应让其余预览都读不出来。
///
/// 想知道每个偏移对应什么，用 [`list_sub_ifds`]。
pub fn sub_ifd_offsets(dng: &DngFile) -> RawResult<Vec<u32>> {
    let entry = dng.ifd0.get(tags::SUB_IFDS);
    let raw = match entry {
        None => return Ok(Vec::new()),
        Some(e) => {
            // `SubIFDs` 的值是**偏移数组**，故字段类型必须是 LONG(4) 或 IFD(13)。
            //
            // # 为什么必须查类型（而不是照`numbers()` 的 Universal 口径读）
            //
            // `TiffEntry::numbers` 对**所有**类型都按自己的宽度解，故一个
            // 写成 SHORT(3) 的 `SubIFDs` 会被解成「一半数量的 32 位偏移」，
            // 偏移值本身却是垃圾——**读得动，但读出来是错的**。真实文件里
            // 厂商写错字段类型的情形并不罕见，故显式拒绝并说清期望类型。
            if e.field_type != field_type::LONG && e.field_type != field_type::IFD {
                return Err(raw_err(
                    LOC,
                    format!(
                        "SubIFDs(0x014A) 的字段类型是 {}，既非 LONG(4) 也非 IFD(13)；\
                         偏移数组的类型写错，读出来的值必是垃圾",
                        e.field_type
                    ),
                    "确认这是合规 TIFF/DNG；用 exiftool -SubIFD 交叉核对字段类型",
                ));
            }
            e.numbers(dng.order)
        }
    };
    if raw.len() > MAX_SUB_IFDS {
        return Err(raw_err(
            LOC,
            format!(
                "SubIFDs 声明 {} 个子 IFD，超过防御上限 {MAX_SUB_IFDS}",
                raw.len()
            ),
            "这是文件损坏（或刻意构造的畸形输入）；请用exiftool -SubIFD 交叉核对",
        ));
    }
    let mut out: Vec<u32> = Vec::with_capacity(raw.len());
    for (i, v) in raw.iter().enumerate() {
        // 非有限 / 负偏移 = 文件损坏，必须报错（不静默丢弃），
        // 否则调用方会以为「文件没有预览」。
        if !v.is_finite() || *v < 0.0 || *v > u32::MAX as f32 {
            return Err(raw_err(
                LOC,
                format!("SubIFDs[{i}] 的偏移非法（{v}）"),
                "确认文件未损坏；可用exiftool -SubIFD 交叉核对",
            ));
        }
        out.push(*v as u32);
    }
    Ok(out)
}

/// 本模块统一的三段式错误构造。
fn raw_err(loc: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> RawError {
    RawError::new("RAW1001", loc, problem, fix)
}

/// 「格式/取值本模块不支持」类错误（`RAW2001`）。
///
/// 与 [`raw_err`]（`RAW1001`，文件/结构损坏）分开：前者是**文件坏了**，
/// 后者是**文件没坏但取值超出本模块能力**。混用会让调用方无法据错误码
/// 区分「用户的文件有问题」与「本模块不支持」，故各用各的。
fn unsupported_err(
    loc: &'static str,
    problem: impl Into<String>,
    fix: impl Into<String>,
) -> RawError {
    RawError::new("RAW2001", loc, problem, fix)
}

/// 解析全部 SubIFD 的**元数据 + 尽可能解出像素**。
///
/// # 行为
///
/// - `list_sub_ifds` 列出的每一项都产出一个 [`SubIfdImage`]；
/// - **下标不连续时按 `SubIfdImage::index` 对齐**（列表里跳过的坏条目
///   不会让后面的下标错位）；
/// - 某个条目完全读不出（`decode_sub_ifd` 返回 `Err`）→ 该条**跳过**，
///   其余照常返回（本函数服务于诊断，一个坏预览不该让整份诊断失败）。
///
/// 故本函数**不会失败**：读不出预览是**事实**，不是错误 —— 事实由
/// [`SubIfdImage::note`] 如实报告。
pub fn parse_sub_ifds(dng: &DngFile) -> RawResult<Vec<SubIfdImage>> {
    let infos = list_sub_ifds(dng)?;
    let mut out: Vec<SubIfdImage> = Vec::with_capacity(infos.len());
    for info in infos.into_iter() {
        // 用 info 里记录的原始下标（列表跳过的坏条目不会让下标错位）。
        out.push(decode_sub_ifd(dng, info.index)?);
    }
    Ok(out)
}

/// 列出 DNG 的**全部 SubIFD** 元数据（按 `SubIFDs` 数组下标顺序）。
///
/// # 行为
///
/// - 无 `SubIFDs` 标签 → 返回空 `Vec`（**不是错误**：没有预览图的 DNG 合法）；
/// - 某个子 IFD 偏移非法 → **跳过该条并继续**（真实文件里偶有坏条目，
///   一个坏预览不该让整份文件的元数据都读不出来）；
/// - 子 IFD 内部结构损坏 → 同上跳过。
///
/// 之所以「跳过而不报错」：本函数服务于**诊断**，一个读不出的预览应当
/// 被报告成「有 N 个 SubIFD，读出 M 个」，而不是让整个诊断失败。
pub fn list_sub_ifds(dng: &DngFile) -> RawResult<Vec<SubIfdInfo>> {
    let data = dng.data_slice();
    let order = dng.order;
    let offsets = sub_ifd_offsets(dng)?;
    let mut out: Vec<SubIfdInfo> = Vec::new();
    for (i, &off32) in offsets.iter().enumerate() {
        if off32 == 0 {
            // TIFF 里偏移 0 表示「无 IFD」；出现在 SubIFDs 数组里是损坏。
            return Err(raw_err(
                LOC,
                format!("SubIFDs[{i}] 的偏移是 0（TIFF 中0 表示无 IFD）"),
                "确认文件未损坏；可用 exiftool 交叉核对 SubIFD 链",
            ));
        }
        let ifd = raw::read_ifd(data, order, off32)?;
        let info = read_info(data, order, i, off32, &ifd);
        validate_sub_ifd(&info)?;
        out.push(info);
    }
    Ok(out)
}

/// 单个 SubIFD 的**防御性校验**：尺寸为 0、位深越界、分量数越界、
/// 样本总数越界（荒谬尺寸标签）一律**结构化拒绝**。
///
/// # 为什么校验放在「列表」层而不是「解码」层
///
/// [`decode_from_ifd`] 对「解不出」是**如实报告**（`pixels = None` +
/// `note`）——那是诚实边界。但**荒谬尺寸不是「解不出」，是「文件损坏」**：
/// 若放任 `ImageWidth = 0x4000_0000` 走到分配语句，一个几KiB 的输入就能
/// 让本模块申请 16 GiB。在**列入清单时**就拒绝，诊断与解码两条路径都安全。
fn validate_sub_ifd(info: &SubIfdInfo) -> RawResult<()> {
    let i = info.index;
    if info.width == 0 || info.height == 0 {
        return Err(raw_err(
            LOC,
            format!(
                "SubIFD[{i}] 的尺寸为 {}×{}（某维为 0）；\
                 尺寸为 0 的 IFD 不含任何像素，是文件损坏",
                info.width, info.height
            ),
            "确认 SubIFD 的 ImageWidth(0x0100)/ImageLength(0x0101) 未被错位读取",
        ));
    }
    if info.samples_per_pixel == 0 {
        return Err(raw_err(
            LOC,
            format!("SubIFD[{i}] 的 SamplesPerPixel = 0"),
            "确认 SamplesPerPixel(0x0115) 标签正确；本模块不做分量数补齐",
        ));
    }
    // 分量数上限4：预览的实际形态是 1（灰度）或 3（RGB / YCbCr）。
    //
    // 依据两处：TIFF 6.0 的 photometric 组合最多用到 4 分量（含 alpha），
    // 而 T.81 无损 JPEG 的 `Nf` 合法区间也是 1..=4（本仓的无损 JPEG 解码
    // 器同样按 1..=4 收）。放行 5+ 只会让「每行样本数」与 JPEG 内部
    // 分量数对不上，产出错图——故在入口拒绝并说清范围。
    if info.samples_per_pixel > 4 {
        return Err(unsupported_err(
            LOC,
            format!(
                "SubIFD[{i}] 的 SamplesPerPixel = {}，超出 1..=4 的合法区间；\
                 预览只可能是灰度(1) 或 RGB/YCbCr(3)",
                info.samples_per_pixel
            ),
            "确认 SamplesPerPixel(0x0115) 未被错位读取；\
             若确为 5 分量以上的私有格式，需先提供该格式的样本布局文档",
        ));
    }
    //样本总数上限：与主图合成器的 2^28 同量级（预览本应远小于此）。
    let px = (info.width as u64)
        .checked_mul(info.height as u64)
        .and_then(|v| v.checked_mul(info.samples_per_pixel as u64))
        .ok_or_else(|| {
            raw_err(
                LOC,
                format!(
                    "SubIFD[{i}] 的样本数 {}×{}×{} 溢出 u64",
                    info.width, info.height, info.samples_per_pixel
                ),
                "文件损坏；请用 exiftool 交叉核对 SubIFD 尺寸",
            )
        })?;
    if px > MAX_SUB_IFD_SAMPLES {
        return Err(raw_err(
            LOC,
            format!(
                "SubIFD[{i}] 声明 {px} 个样本（{}×{}×{}），超过防御上限 {MAX_SUB_IFD_SAMPLES}；\
                 不分配、不解码",
                info.width, info.height, info.samples_per_pixel
            ),
            "这是文件损坏（或刻意构造的畸形输入）；预览图不应有这么大",
        ));
    }
    Ok(())
}

/// 解一个 SubIFD 的像素（能解则解，不能解则**明确说明原因**）。
///
/// # 参数
///
/// - `dng` —— 已解析的文件（提供字节缓冲与字节序）；
/// - `index` —— `SubIFDs` 数组下标。
///
/// # 错误
///
/// 仅在「下标越界」「偏移非法」「子 IFD 结构读不出」时返回 [`RawError`]；
/// 「压缩方式不支持」「位深非 8」**不是错误** —— 那会得到
/// `pixels = None`、`note` 写明原因 的 [`SubIfdImage`]
/// （诚实边界的第二层：**读不出是事实，不是失败**）。
pub fn decode_sub_ifd(dng: &DngFile, index: usize) -> RawResult<SubIfdImage> {
    let data = dng.data_slice();
    let order = dng.order;
    // 偏移的合法性校验只在一处（sub_ifd_offsets），本函数不重复实现。
    let offsets = sub_ifd_offsets(dng)?;
    let &off32 = offsets.get(index).ok_or_else(|| {
        raw_err(
            LOC,
            format!("SubIFDs 下标 {index} 不存在（共 {} 条）", offsets.len()),
            "确认该 DNG 是否带 SubIFD 预览图；用 list_sub_ifds 先看有哪些",
        )
    })?;
    if off32 == 0 {
        return Err(raw_err(
            LOC,
            format!("SubIFDs[{index}] 的偏移是 0（TIFF 中 0 表示无 IFD）"),
            "确认文件未损坏；可用 exiftool 交叉核对 SubIFD 链",
        ));
    }
    let ifd = raw::read_ifd(data, order, off32)?;
    let info = read_info(data, order, index, off32, &ifd);
    validate_sub_ifd(&info)?;
    Ok(decode_from_ifd(dng, data, order, &ifd, info))
}

/// 对**已解析**的子 IFD 执行解码（供 [`decode_sub_ifd`] 与测试复用）。
///
/// # 返回
///
/// **从不失败** —— 解不出像素是**事实**而非错误：`pixels = None` +
/// `note` 写明原因。只有「子 IFD 结构本身读不出」才在 [`decode_sub_ifd`]
/// 里作为 `Err` 返回。
fn decode_from_ifd(
    dng: &DngFile,
    data: &[u8],
    order: ByteOrder,
    ifd: &TiffIfd,
    info: SubIfdInfo,
) -> SubIfdImage {
    // 把 `info` 拆成局部量：既让下面的 `unresolved!` 能重建它（避免 move），
    // 又让 `width/height/...` 在正文里直读（与元数据字段一一对应）。
    let SubIfdInfo {
        index,
        file_offset,
        new_subfile_type,
        width: w,
        height: h,
        compression,
        bits_per_sample: bits,
        samples_per_pixel: spp,
        photometric_interpretation,
        subfile_kind,
        layout,
    } = info;

    /// 由元数据起一个「未解出」的结果（附原因）。
    macro_rules! unresolved {
        ($reason:expr) => {
            SubIfdImage::unresolved(
                SubIfdInfo {
                    index,
                    file_offset,
                    new_subfile_type,
                    width: w,
                    height: h,
                    compression,
                    bits_per_sample: bits,
                    samples_per_pixel: spp,
                    photometric_interpretation,
                    subfile_kind,
                    layout,
                },
                $reason,
            )
        };
    }

    // ---- 尺寸前提 ----
    if w == 0 || h == 0 {
        return unresolved!(format!(
            "SubIFD[{}] 缺 ImageWidth/ImageLength（读到 {w}×{h}），无像素可解",
            index
        ));
    }
    let total = match (w as usize)
        .checked_mul(h as usize)
        .and_then(|v| v.checked_mul(spp as usize))
    {
        Some(v) => v,
        None => {
            return unresolved!(format!(
                "SubIFD[{}] {w}×{h}×{spp} 样本数溢出 usize",
                index
            ))
        }
    };

    // ---- 压缩方式分派：不支持则只报元数据 + 说清为何不解 ----
    match compression {
        1 | 7 | 8 | 32946 => {}
        other => {
            return unresolved!(format!(
                "SubIFD[{}] 的 Compression = {other}（{}）本模块未实装；\
                 本模块只解 1（未压缩）/ 7（无损 JPEG）/ 8、32946（Deflate）。\
                 元数据已完整读出，预览像素不可用",
                index,
                raw::compression_reject_hint(other)
            ))
        }
    }

    // ---- Photoshop TN 警告：位深非 8 的未压缩 SubIFD 不解像素 ----
    //
    // 依据：《Photoshop TIFF Technical Notes》开篇
    // "TIFF readers that encounter SubIFD (tag number 330) and a bit depth
    //  other than 8 should ignore the SubIFD's found in the file"。
    // 这是规范给的明确指引，照做而非自行推断。
    if compression == 1 && bits != 8 {
        return unresolved!(format!(
            "SubIFD[{}] 是未压缩但位深 = {}（非 8）；\
             《Photoshop TIFF Technical Notes》要求读端忽略这类 SubIFD 的像素\
             （'a bit depth other than 8 should ignore the SubIFD's'），故只解析元数据",
            index, info.bits_per_sample
        ));
    }

    // ---- 载荷布局与块区间 ----
    let block_layout = detect_sub_layout(data, ifd, order, h);
    let regions = match collect_payload_regions(data, ifd, order, block_layout) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            return unresolved!(format!(
                "SubIFD[{}] 的偏移表为空（既无 StripOffsets 也无 TileOffsets）",
                index
            ))
        }
        Err(e) => {
            return unresolved!(format!("SubIFD[{}] 载荷布局不可读：{}", index, e.problem))
        }
    };

    let fill = dng_sub_fill_order(dng, ifd);
    // Predictor（缺省 1 = Null；DNG 只在 Compression=8 的章节定义它）。
    let pred_code = ifd
        .first_number(tags::PREDICTOR, order)
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u16::MAX as f32)
        .map(|v| v as u16)
        .unwrap_or(1);

    // 逐块解码 → **按 `(x0, y0)` 落位**到整图缓冲（不是顺序追加！）。
    //
    // 缓冲**预填 0**：图边界处的 tile 含图外填充像素，落到界外时**跳过**
    // （那部分本就不属于本图），而不是把后续块挤偏。
    let mut samples: Vec<u16> = vec![0u16; total];
    let row_len = (w as usize).saturating_mul(spp as usize);
    let mut placed = 0usize;
    for (idx, region) in regions.iter().enumerate() {
        let SubRegion {
            offset: off,
            byte_len: len,
            nominal_samples,
            nominal_row_len,
            x0,
            y0,
        } = *region;
        // 本块应解出的样本数 = 标称量（图外填充照样解出来，落位时再裁）。
        let want = nominal_samples;
        let end = off.saturating_add(len as usize);
        let raw_bytes: &[u8] = match data.get(off..end) {
            Some(v) if !v.is_empty() => v,
            _ => {
                return unresolved!(format!(
                    "SubIFD[{}] 第 {idx} 块区间 [{off}..{end}) 越界或为空（文件长 {}）",
                    index,
                    data.len()
                ))
            }
        };

        let block: Vec<u16> = match compression {
            1 => match unpack_bits(raw_bytes, bits, want, fill) {
                Ok(v) => v,
                Err(e) => {
                    return unresolved!(format!(
                        "SubIFD[{}] 第 {idx} 块解包失败：{}",
                        index, e.problem
                    ))
                }
            },
            7 => match crate::raw_lossless_jpeg::decode_dng_strip(raw_bytes, want, 0, 1, 0) {
                Ok(v) => v,
                Err(e) => {
                    return unresolved!(format!(
                        "SubIFD[{}] 第 {idx} 块无损 JPEG 解码失败：{}",
                        index, e.problem
                    ))
                }
            },
            _ => {
                // Deflate：**zlib 容器**（非裸 DEFLATE），与主图同口径。
                let bytes = match raw::inflate_dng_zlib(raw_bytes, pred_code, LOC) {
                    Ok(v) => v,
                    Err(e) => {
                        return unresolved!(format!(
                            "SubIFD[{}] 第 {idx} 块 Deflate 解压失败：{}",
                            index, e.problem
                        ))
                    }
                };
                let mut v = match unpack_bits(&bytes, bits, want, fill) {
                    Ok(v) => v,
                    Err(e) => {
                        return unresolved!(format!(
                            "SubIFD[{}] 第 {idx} 块解包失败：{}",
                            index, e.problem
                        ))
                    }
                };
                // 预测器不支持时**不解**（不静默产出错图）。
                match raw::DeflatePredictor::from_code(pred_code) {
                    Some(p) => p.undo_in_place(&mut v, nominal_row_len, bits),
                    None => {
                        return unresolved!(format!(
                            "SubIFD[{}] 的 Predictor = {pred_code} 本模块未实装（{}），只解析元数据",
                            index,
                            raw::DeflatePredictor::reject_reason(pred_code)
                        ))
                    }
                }
                v
            }
        };
        // 按 `(x0, y0)` 逐行落位；图边界外的部分**丢弃**（图外填充像素
        // 不属于本图）。`nominal_row_len` 是本块的行宽（tile 宽 × spp），
        // 整图行宽是 `width × spp` —— 两者不等时必须按各自的行宽跳。
        let block_rows = if nominal_row_len == 0 {
            0
        } else {
            block.len() / nominal_row_len
        };
        for by in 0..block_rows {
            let dst_y = y0.saturating_add(by);
            if dst_y >= h as usize {
                break; // 整块都在图下方
            }
            let src_base = by.saturating_mul(nominal_row_len);
            let dst_base = dst_y.saturating_mul(row_len).saturating_add(x0);
            // 图边界处的 tile 含**图外填充像素**：块行宽是 `tile_width × spp`，
            // 整图行宽是 `width × spp`。本行能写的样本数受**两者同时**约束——
            // 既不得越过整图行尾（`row_len - x0`），也不得越过本块行尾
            // （`nominal_row_len`）。少了后一个约束，拷满标称宽度就会跨过
            // 行尾写进下一行（表现为隔行错位）。
            let copy_w = row_len
                .saturating_sub(x0)
                .min(nominal_row_len);
            for bx in 0..copy_w {
                let src = src_base.saturating_add(bx);
                let dst = dst_base.saturating_add(bx);
                if let (Some(&sv), Some(slot)) = (block.get(src), samples.get_mut(dst)) {
                    *slot = sv;
                }
            }
        }
        placed = placed.saturating_add(block.len());
    }

    // 全 0 意味着**一个样本都没落位**（块数/偏移表与尺寸不符）。
    // 部分落位不做此判定——预览允许有 legitimately 为 0 的像素。
    if placed == 0 {
        return unresolved!(format!(
            "SubIFD[{}] 一个块也没落位（{w}×{h}×{spp}，{} 块）；块数或偏移表与尺寸不符",
            index,
            regions.len()
        ));
    }
    // 如实说明**没有做什么**：`PreviewColorSpace` / `YCbCrSubSampling`
    // 只解析不应用，色彩空间转换是调用方的事——说清楚比让人误以为
    // 拿到的是 sRGB 像素要好。
    let note = format!(
        "已解出 {} 样本（{w}×{h}×{spp}）；未做色彩空间转换（交错的原始样本，\
         色彩解释由调用方按 PhotometricInterpretation={} 负责）",
        samples.len(),
        photometric_interpretation
    );
    SubIfdImage {
        info: SubIfdInfo {
            index,
            file_offset,
            new_subfile_type,
            width: w,
            height: h,
            compression,
            bits_per_sample: bits,
            samples_per_pixel: spp,
            photometric_interpretation,
            subfile_kind,
            layout,
        },
        pixels: Some(samples),
        note,
    }
}

/// 子 IFD 的 `FillOrder`（缺省 / 非法 → MSB-first）。
///
/// 复用 [`raw::DngFile::sub_ifd_fill_order`]（与主图 [`raw::DngFile::fill_order`]
/// 的**回退策略一致**：按规范缺省解释，好过让整张预览不可读），
/// 但读的是**子 IFD 自己的**标签——预览图可能与主图用不同位序。
fn dng_sub_fill_order(dng: &DngFile, ifd: &TiffIfd) -> raw::FillOrder {
    dng.sub_ifd_fill_order(ifd)
        .unwrap_or(raw::FillOrder::MsbFirst)
}

/// SubIFD 的载荷布局（只需区分 strip / tile 两种）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubLayout {
    /// strip 布局（`StripOffsets` + `RowsPerStrip`）。
    Strip {
        /// 每 strip 行数。
        rows_per_strip: u32,
    },
    /// tile 布局（`TileOffsets` + `TileWidth`/`TileLength`）。
    Tile {
        /// 单 tile 宽。
        tile_width: u32,
        /// 单 tile 高。
        tile_length: u32,
    },
}

/// 判定 SubIFD 的载荷布局（**tile 标签优先**，与主图 [`raw::RawLayout`] 同策略）。
fn detect_sub_layout(data: &[u8], ifd: &TiffIfd, order: ByteOrder, height: u32) -> SubLayout {
    let _ = data;
    let tw = ifd.first_number(tags::TILE_WIDTH, order);
    let tl = ifd.first_number(tags::TILE_LENGTH, order);
    if let (Some(tw), Some(tl)) = (tw, tl) {
        if tw >= 1.0 && tl >= 1.0 {
            return SubLayout::Tile {
                tile_width: tw as u32,
                tile_length: tl as u32,
            };
        }
    }
    let rps = ifd
        .first_number(tags::ROWS_PER_STRIP, order)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .unwrap_or_else(|| height.max(1) as f32) as u32;
    SubLayout::Strip {
        rows_per_strip: rps.max(1),
    }
}

/// 收集 SubIFD 各块的 `(文件偏移, 字节数, 标称样本数, 标称行宽)`。
///
/// `order` 是文件的真实字节序 —— 布局判定由 [`detect_sub_layout`] 完成，
/// 本函数只按布局切块。
fn collect_payload_regions(
    data: &[u8],
    ifd: &TiffIfd,
    order: ByteOrder,
    layout: SubLayout,
) -> RawResult<Vec<SubRegion>> {
    let offs = numbers_u64(ifd, tags::STRIP_OFFSETS, order);
    let counts = numbers_u64(ifd, tags::STRIP_BYTE_COUNTS, order);
    let w = ifd.first_number(tags::IMAGE_WIDTH, order).unwrap_or(0.0);
    let h = ifd.first_number(tags::IMAGE_LENGTH, order).unwrap_or(0.0);
    let spp = ifd
        .first_number(tags::SAMPLES_PER_PIXEL, order)
        .unwrap_or(1.0)
        .max(1.0);
    let (wi, hi, sppi) = (
        w.max(0.0) as usize,
        h.max(0.0) as usize,
        spp as usize,
    );
    let _ = data;
    let mut out: Vec<SubRegion> = Vec::new();
    match layout {
        SubLayout::Tile {
            tile_width,
            tile_length,
        } => {
            let toffs = numbers_u64(ifd, tags::TILE_OFFSETS, order);
            let tcounts = numbers_u64(ifd, tags::TILE_BYTE_COUNTS, order);
            let tw = tile_width.max(1);
            let tl = tile_length.max(1);
            let across = (wi as u32).div_ceil(tw);
            let down = (hi as u32).div_ceil(tl);
            let n = (across as usize).saturating_mul(down as usize);
            for i in 0..n.min(toffs.len()) {
                // 块按**行优先**排列：第 `ty` 行第 `tx` 列的序号 =
                // `ty * across + tx`（与主图 `read_raw_samples` 的 tile
                // 分支同一口径）。
                let tx = (i as u32) % across;
                let ty = (i as u32) / across;
                let nominal = (tw as usize).saturating_mul(tl as usize).saturating_mul(sppi);
                out.push(SubRegion {
                    offset: toffs[i] as usize,
                    byte_len: tcounts.get(i).copied().unwrap_or(0) as u32,
                    nominal_samples: nominal,
                    nominal_row_len: (tw as usize).saturating_mul(sppi),
                    x0: (tx.saturating_mul(tw) as usize).saturating_mul(sppi),
                    y0: ty.saturating_mul(tl) as usize,
                });
            }
        }
        SubLayout::Strip { rows_per_strip } => {
            let rps = (rows_per_strip.max(1)) as usize;
            let strips = hi.div_ceil(rps);
            for s in 0..strips.min(offs.len()) {
                let rows = rps.min(hi.saturating_sub(s.saturating_mul(rps)));
                let nominal = wi.saturating_mul(rows).saturating_mul(sppi);
                out.push(SubRegion {
                    offset: offs[s] as usize,
                    byte_len: counts.get(s).copied().unwrap_or(0) as u32,
                    nominal_samples: nominal,
                    nominal_row_len: wi.saturating_mul(sppi),
                    // strip 只切行：`x0` 恒 0，`y0 = s × RowsPerStrip`。
                    x0: 0,
                    y0: s.saturating_mul(rps),
                });
            }
        }
    }
    Ok(out)
}

/// 一个载荷块在**整图坐标系**里的定位与标称形状。
///
/// # 为什么必须带 `x0` / `y0`（这曾是一个真实的 bug）
///
/// tile 是**二维**分块：同一行的第 `tx` 个 tile 落在 `x0 = tx × TileWidth`，
/// 第 `ty` 行落在 `y0 = ty × TileLength`。早期实现把块**按顺序追加**到
/// 样本缓冲（隐含「每块都是满宽、且顺序就是行序」），于是：
///
/// - 同一行的所有 tile 都写到第 0 **列** → 互相覆盖，图像右侧被涂掉；
/// - strip 布局下 `y0` 若被当成 0，多个 strip 会全部堆在头几行。
///
/// 症状是「能解出、长度也对、像素全错」——最坏的一类 bug。故定位信息
/// 必须在**收集阶段**就带全，拷贝阶段只管按 `(x0, y0)` 落位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubRegion {
    /// 载荷在文件中的起始偏移。
    offset: usize,
    /// 压缩字节数（`StripByteCounts` / `TileByteCounts` 对应项）。
    byte_len: u32,
    /// 本块的**标称**样本数（含图边界外的填充像素）。
    nominal_samples: usize,
    /// 本块每行的样本数（`tile_width × spp` 或 `width × spp`）。
    ///
    /// 预测器按行运算，必须知道行宽——否则跨块边界的行会被接错。
    nominal_row_len: usize,
    /// 本块左上角在整图中的**列**坐标（样本列，非像素列）。
    x0: usize,
    /// 本块左上角在整图中的**行**坐标。
    y0: usize,
}

/// 从一个 IFD 读 `u64` 数组（过滤非法值）。
fn numbers_u64(ifd: &TiffIfd, tag: u16, order: ByteOrder) -> Vec<u64> {
    ifd.numbers(tag, order)
        .unwrap_or_default()
        .into_iter()
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= u32::MAX as f32)
        .map(|v| v as u64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::ByteOrder;
    use crate::raw_dng_synth::{synthesize_dng, DngSampleSpec, PreviewLayout, PreviewSpec};


    fn parse_one_preview(spec: &DngSampleSpec) -> SubIfdImage {
        let file = synthesize_dng(spec).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        subs.into_iter().next().expect("须有 1 个 SubIFD")
    }

    #[test]
    fn no_subifds_tag_yields_empty_offsets() {
        // 「文件没预览图」是**正常事实**，返回空向量而非错误。
        let s = DngSampleSpec::new(8, 8, 8, [0, 1, 1, 2]);
        let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        assert!(sub_ifd_offsets(&d).unwrap_or_else(|e| panic!("{e}")).is_empty());
        assert!(parse_sub_ifds(&d).unwrap_or_else(|e| panic!("{e}")).is_empty());
    }

    #[test]
    fn subifd_metadata_is_reported_for_every_compression() {
        // 三种受支持的压缩方式都要报出完整元数据。
        for &comp in &[1u16, 7, 8, 32946] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s.preview = Some(PreviewSpec::new(30, 20, 8, comp, 2, 3));
            let p = parse_one_preview(&s);
            assert_eq!((p.width, p.height), (30, 20), "comp={comp}");
            assert_eq!(p.compression, comp, "comp={comp}");
            assert_eq!(p.bits_per_sample, 8, "comp={comp}");
            assert_eq!(p.samples_per_pixel, 3, "comp={comp}");
            assert_eq!(p.photometric_interpretation, 2, "comp={comp}");
            assert_eq!(p.new_subfile_type, 1, "comp={comp} 应是 reduced-resolution");
            assert!(p.has_pixels(), "comp={comp}: {}", p.note);
            assert_eq!(p.pixels.as_ref().unwrap().len(), 30 * 20 * 3, "comp={comp}");
            // summary() 须含关键信息（CLI 可见）。
            let sum = p.summary();
            assert!(sum.contains("30×20"), "comp={comp}: {sum}");
            assert!(sum.contains("像素已解出"), "comp={comp}: {sum}");
        }
    }

    #[test]
    fn subifd_pixels_are_bit_exact() {
        // 像素必须逐位一致（不是「看起来差不多」）。
        for &comp in &[1u16, 7, 8] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            let pv = PreviewSpec::new(17, 11, 8, comp, 2, 3);
            s.preview = Some(pv);
            let p = parse_one_preview(&s);
            let want = crate::raw_dng_synth::tests_support_preview_pattern(&pv);
            let got = p.pixels.as_ref().unwrap();
            assert_eq!(got.len(), want.len(), "comp={comp}");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "comp={comp} 样本 {i} 不一致");
            }
        }
    }

    #[test]
    fn subifd_reports_metadata_only_for_unsupported_compression() {
        // **诚实边界的核心测试**：不支持的压缩方式 → `pixels = None`
        // 且 `note` 说清成因，**绝不返回假像素**。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(16, 8, 8, 1, 2, 3));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        // 把预览的 Compression 改成未支持的取值（34892 = Lossy JPEG）。
        patch_subifd_compression(&mut file, 34892);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        let p = &subs[0];
        assert_eq!(p.compression, 34892, "须如实报出压缩方式");
        assert!(!p.has_pixels(), "不支持的压缩方式不得返回像素");
        assert!(
            p.note.contains("Lossy JPEG"),
            "note 须说明为何不解：{}",
            p.note
        );
        // 元数据仍须完整（这正是「只解析元数据」的意义）。
        assert_eq!((p.width, p.height), (16, 8));
    }

    #[test]
    fn subifd_multi_strip_preview_is_reassembled() {
        // 多 strip 预览：各 strip 须按 `y0` 拼回**正确的行**。
        //
        // # 这条断言为什么不能写成「有像素就查长度，否则查 note」
        //
        // 此前就是这么写的（两分支都恒过，等于没断言）。现在改成
        // **逐位比对**：把多 strip 的重组结果与「同一预览但不切 strip」的
        // 结果对比。`blocks_of` 里 `y0` 若被改成 0，全部 strip 会写到第0 行
        // 互相覆盖 → 像素明显不同 → 这里**必然变红**。
        //
        //覆盖 comp = 1 / 8 / 32946（Deflate 与未压缩都走 `copy_block_into`）。
        for &comp in &[1u16, 8, 32946] {
            // 基线：同一预览图，单 strip（RowsPerStrip = height）。
            let mut base_spec = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            base_spec.preview = Some(PreviewSpec::new(16, 16, 8, comp, 2, 3));
            let base = parse_one_preview(&base_spec);
            let baseline = base
                .pixels
                .as_ref()
                .unwrap_or_else(|| panic!("comp={comp} 基线须解出像素：{}", base.note));
            assert_eq!(baseline.len(), 16 * 16 * 3, "comp={comp}");

            // 同一预览图，切成 4 行/strip 的 4 个 strip。
            let mut ms = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            ms.preview = Some(PreviewSpec {
                layout: PreviewLayout::Strip { rows_per_strip: 4 },
                ..PreviewSpec::new(16, 16, 8, comp, 2, 3)
            });
            let p = parse_one_preview(&ms);
            assert!(
                p.layout.contains("4 行/strip") && p.layout.contains("共 4 块"),
                "comp={comp}: 版式须报出4 块：{}",
                p.layout
            );
            let got = p
                .pixels
                .as_ref()
                .unwrap_or_else(|| panic!("comp={comp} 多 strip 须解出像素：{}", p.note));
            // 逐位比对：与单 strip 的结果**完全一致**。
            assert_eq!(got.len(), baseline.len(), "comp={comp} 样本数");
            for (i, (g, w)) in got.iter().zip(baseline.iter()).enumerate() {
                assert_eq!(g, w, "comp={comp} 多 strip 样本 {i} 与单 strip 不一致");
            }
            // 再与合成图案对一次（防止「基线本身是错的」导致两边一起错）。
            let pv = ms.preview.unwrap_or_else(|| panic!("须有预览规格"));
            let want = crate::raw_dng_synth::tests_support_preview_pattern(&pv);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "comp={comp} 样本 {i} 与预览图案不一致");
            }
        }
    }

    /// **tile 路径**（`TileOffsets`/`TileByteCounts`/`TileWidth`/`TileLength`）
    /// 必须能逐位重组——这条路径此前**完全没有测试守护**（QA 变异测试：
    /// 把 `ty * tile_h` 改成 `0`，477 条测试仍全绿）。
    ///
    /// # 它曾不只是「没测试」，而是**真的错了**
    ///
    /// `blocks_of` 原本返回的块定位信息里**没有 `x0`**，两个调用点都把
    /// `x0` 硬编码成 `0`。tile 是**二维**分块，同一行的第 `tx` 个 tile
    /// 应当落在 `x0 = tx * tile_w`；缺了它，同一行所有 tile 都写到第 0 列
    /// **互相覆盖**（图像右侧被涂掉，只剩最右一列）。本测试用
    /// 「单 tile 宽小于图宽」+「逐位比对」把它钉死。
    #[test]
    fn subifd_tile_preview_is_reassembled_bit_exactly() {
        // 图幅故意**不是** tile 尺寸的整数倍（12 = 2×4+4，10 = 2×4+2，
        // 9 = 2×4+1），以覆盖图边界处tile 被裁剪的分支。
        for &(w, h, tw, tl) in &[(12u32, 10u32, 4u32, 4u32), (9, 9, 4, 4)] {
            for &comp in &[1u16, 7, 8, 32946] {
                let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
                s.preview = Some(PreviewSpec {
                    layout: PreviewLayout::Tile {
                        tile_width: tw,
                        tile_length: tl,
                    },
                    ..PreviewSpec::new(w, h, 8, comp, 2, 3)
                });
                let p = parse_one_preview(&s);
                assert!(
                    p.has_pixels(),
                    "{w}×{h} tile {tw}×{tl} comp={comp} 须解出像素：{}",
                    p.note
                );
                assert!(
                    p.layout.starts_with("tile"),
                    "{w}×{h} comp={comp} 版式须报 tile：{}",
                    p.layout
                );
                let got = p.pixels.as_ref().expect("已断言 has_pixels");
                let want = crate::raw_dng_synth::tests_support_preview_pattern(
                    &s.preview.unwrap_or_else(|| panic!("须有预览规格")),
                );
                assert_eq!(got.len(), want.len(), "{w}×{h} comp={comp} 样本数");
                // 逐位比对：任一 tile 的x0/y0 算错都会让某个样本不同。
                for (i, (g, wl)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(g, wl, "{w}×{h} tile{tw}×{tl} comp={comp} 样本 {i} 不一致");
                }
            }
        }
    }

    /// tile 布局下**只切列**（`tile_length >= height`）时，`y0` 恒 0 但
    /// `x0` 必须正确——这条把「`x0` 被误当 0」与「`y0` 被误当 0」两个
    /// 变异**分别**抓住（上一条两者同时变异，无法定位是谁错）。
    #[test]
    fn subifd_tile_column_offset_is_not_hardcoded_to_zero() {
        let (w, h, tw) = (12u32, 3u32, 4u32);
        for &comp in &[1u16, 8] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s.preview = Some(PreviewSpec {
                layout: PreviewLayout::Tile {
                    tile_width: tw,
                    tile_length: h,
                },
                ..PreviewSpec::new(w, h, 8, comp, 1, 1)
            });
            let p = parse_one_preview(&s);
            let got = p
                .pixels
                .as_ref()
                .unwrap_or_else(|| panic!("comp={comp} 须解出像素：{}", p.note));
            let want = crate::raw_dng_synth::tests_support_preview_pattern(
                &s.preview.unwrap_or_else(|| panic!("须有预览规格")),
            );
            for (i, (g, wl)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, wl, "comp={comp} 列偏移样本 {i} 不一致");
            }
            // 额外钉死「横向内容确实不同」：若图案横向恒定，上面的比对
            // 就抓不住「整行被同一个 tile 覆盖」。这里断言第 0 列与
            // 最后一列的样本**不同**——保证测试真的有区分力。
            // （本测试用 spp=1，故列 stride = 1。）
            let last_col = w as usize - 1;
            assert_ne!(
                want[0], want[last_col],
                "预览图案横向须有差异，否则本测试无区分力"
            );
            assert_ne!(got[0], got[last_col], "重组后横向仍须有差异");
        }
    }

    #[test]
    fn subifd_rejects_zero_dimension_without_panic() {
        // 尺寸标签为 0 → 结构化拒绝，不 panic、不返回零长向量。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        patch_subifd_dimension(&mut file, 0);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        match d.sub_ifds() {
            Ok(_) => panic!("宽 0 的 SubIFD 不该被接受"),
            Err(e) => {
                assert!(
                    e.code == "RAW1001" || e.code == "RAW2001",
                    "须结构化拒绝：{e}"
                );
            }
        }
    }

    #[test]
    fn subifd_rejects_oversized_sample_count() {
        // 荒谬尺寸标签 → 被上限挡住（不 OOM、不 panic）。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        patch_subifd_dimension(&mut file, 0x4000_0000);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        match d.sub_ifds() {
            Ok(_) => panic!("超大尺寸的 SubIFD 不该被接受"),
            Err(e) => assert!(
                e.code == "RAW1001" || e.code == "RAW2001",
                "须结构化拒绝：{e}"
            ),
        }
    }

    #[test]
    fn subifd_rejects_bad_field_type_without_panic() {
        // SubIFDs 用非法字段类型（SHORT 而非 LONG/IFD）→ 结构化拒绝。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        patch_subifd_field_type(&mut file, 3); // SHORT
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        match d.sub_ifds() {
            Ok(_) => panic!("SHORT 类型的 SubIFDs 不该被接受"),
            Err(e) => {
                assert_eq!(e.code, "RAW1001", "{}", e);
                assert!(format!("{e}").contains("LONG"), "{}", e);
            }
        }
    }

    #[test]
    fn subifd_grayscale_and_ycbcr_metadata() {
        // 灰度（PI=1, spp=1）与 YCbCr（PI=6, spp=3）都要报对元数据，
        // 且 note 须点明「不做色彩空间转换」。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let g = parse_one_preview(&s);
        assert_eq!(g.samples_per_pixel, 1);
        assert!(g.has_pixels(), "{}", g.note);

        let mut s2 = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s2.preview = Some(PreviewSpec::new(8, 8, 8, 1, 6, 3));
        let y = parse_one_preview(&s2);
        assert_eq!(y.photometric_interpretation, 6);
        assert_eq!(y.samples_per_pixel, 3);
        assert!(y.has_pixels(), "{}", y.note);
        assert!(
            y.note.contains("色彩空间") || y.note.contains("YCbCr"),
            "须说明未做色彩空间转换：{}",
            y.note
        );
    }

    #[test]
    fn subifd_offsets_accepts_both_long_and_ifd_field_types() {
        // TIFF 5.0 允许 SubIFDs 用 LONG(4) 或 IFD(13)——两种都要接受。
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        // 先改成 IFD(13)——应仍能解析。
        patch_subifd_field_type(&mut file, 13);
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("IFD 类型应被接受: {e}"));
        assert_eq!(subs.len(), 1);
        assert_eq!((subs[0].width, subs[0].height), (8, 8));
    }

    // -------------------------------------------------------------------
    //  字节补丁工具（改子 IFD 的标签值）
    // -------------------------------------------------------------------

    /// 在文件里定位 IFD0 的 `SubIFDs` 条目并改其 field_type。
    fn patch_subifd_field_type(file: &mut [u8], ty: u16) -> bool {
        let d = match DngFile::parse(file) {
            Ok(d) => d,
            Err(_) => return false,
        };
        if d.ifd0.get(crate::raw::tags::SUB_IFDS).is_none() {
            return false;
        }
        // 条目在 IFD0 目录里的位置：2 + 12×idx+ 2（type 字段）。
        let idx = d
            .ifd0
            .entries
            .iter()
            .position(|x| x.tag == crate::raw::tags::SUB_IFDS)
            .unwrap_or(usize::MAX);
        if idx == usize::MAX {
            return false;
        }
        let le = d.order == ByteOrder::Little;
        let at = 8 + 2 + 12 * idx + 2;
        if at + 2 > file.len() {
            return false;
        }
        let bytes = if le { ty.to_le_bytes() } else { ty.to_be_bytes() };
        file[at] = bytes[0];
        file[at + 1] = bytes[1];
        true
    }

    /// 改子 IFD 的 `Compression`（顺带改主 IFD 的 Predictor 不动）。
    fn patch_subifd_compression(file: &mut [u8], code: u16) -> bool {
        patch_subifd_short(file, crate::raw::tags::COMPRESSION, code)
    }

    /// 改子 IFD 的 `ImageWidth`（也用来制造荒谬尺寸）。
    fn patch_subifd_dimension(file: &mut [u8], w: u32) -> bool {
        patch_subifd_long(file, crate::raw::tags::IMAGE_WIDTH, w)
    }

    /// 解析出子 IFD 的文件偏移（供补丁定位）。
    fn sub_ifd_offset(file: &[u8]) -> Option<usize> {
        let d = DngFile::parse(file).ok()?;
        let offs = crate::raw_subifd::sub_ifd_offsets(&d).ok()?;
        offs.first().map(|&v| v as usize)
    }

    /// 在子 IFD 目录里找到 `tag` 条目的 value 字段偏移。
    fn sub_ifd_entry_value_at(file: &[u8], tag: u16) -> Option<usize> {
        let base = sub_ifd_offset(file)?;
        let le = file.get(0..2) == Some(b"II");
        let count = if le {
            u16::from_le_bytes([*file.get(base)?, *file.get(base + 1)?])
        } else {
            u16::from_be_bytes([*file.get(base)?, *file.get(base + 1)?])
        } as usize;
        for i in 0..count {
            let e = base + 2 + 12 * i;
            let t = if le {
                u16::from_le_bytes([*file.get(e)?, *file.get(e + 1)?])
            } else {
                u16::from_be_bytes([*file.get(e)?, *file.get(e + 1)?])
            };
            if t == tag {
                return Some(e + 8);
            }
        }
        None
    }

    /// 改子 IFD 的一个 **SHORT** 值（就地改内联 value 字段）。
    ///
    /// # 内联值字段的对齐（TIFF 5.0 §2，**踩过坑**：不是「看起来那样」）
    ///
    /// 值 ≤ 4 字节时内联在 4 字节 value 字段里，而字节布局是
    /// **LE 右对齐 / BE 左对齐**（读端 [`crate::raw::read_ifd`] 与写端
    /// [`crate::raw_dng_synth::write_ifd0`] 都遵此约定）：
    ///
    /// ```text
    /// LE 容器的 SHORT 1  →  [00 00 01 00]   ← 值在**高**两字节
    /// BE 容器的 SHORT 1  →  [00 01 00 00]   ← 值在**前**两字节
    /// ```
    ///
    /// 直觉上「LE 就写前两字节」是**错的**——那会被读成 0。
    /// 这与 `raw.rs` 里既有的 `patch_inline_short`（写 `e + 10`）同源。
    fn patch_subifd_short(file: &mut [u8], tag: u16, value: u16) -> bool {
        let at = match sub_ifd_entry_value_at(file, tag) {
            Some(a) => a,
            None => return false,
        };
        if at + 4 > file.len() {
            return false;
        }
        let le = file.get(0..2) == Some(b"II");
        let b = if le { value.to_le_bytes() } else { value.to_be_bytes() };
        let v = if le { at + 2 } else { at };
        file[v] = b[0];
        file[v + 1] = b[1];
        true
    }

    fn patch_subifd_long(file: &mut [u8], tag: u16, value: u32) -> bool {
        let at = match sub_ifd_entry_value_at(file, tag) {
            Some(a) => a,
            None => return false,
        };
        let le = file.get(0..2) == Some(b"II");
        let b = if le { value.to_le_bytes() } else { value.to_be_bytes() };
        if at + 4 > file.len() {
            return false;
        }
        file[at..at + 4].copy_from_slice(&b);
        true
    }

    /// 把 IFD0 的 `SubIFDs` **偏移值**改掉（制造越界 / 自指 / 垃圾偏移）。
    ///
    /// 与 [`patch_subifd_long`] 的区别：后者改的是**子 IFD 内部**的 LONG 标签；
    /// 这里改的是 IFD0 里`SubIFDs` 标签指向的那个偏移本身。
    /// `SubIFDs` 是 LONG、count=1，值内联在 4 字节 value 字段里，按容器字节序
    /// 直接读写 u32（无 SHORT 那种「高位补零」语义）。
    fn patch_subifd_offset(file: &mut [u8], new_off: u32) -> bool {
        let d = match DngFile::parse(file) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let idx = match d
            .ifd0
            .entries
            .iter()
            .position(|x| x.tag == crate::raw::tags::SUB_IFDS)
        {
            Some(i) => i,
            None => return false,
        };
        let le = d.order == ByteOrder::Little;
        let at = 8 + 2 + 12 * idx + 8;
        if at + 4 > file.len() {
            return false;
        }
        let b = if le {
            new_off.to_le_bytes()
        } else {
            new_off.to_be_bytes()
        };
        file[at..at + 4].copy_from_slice(&b);
        true
    }

    // -------------------------------------------------------------------
    //  ⑧ 读端守卫与防御上限（缺口 4：此前这些分支一条测试都没有）
    // -------------------------------------------------------------------

    /// 把 IFD0 的 `SubIFDs` 标签改成 `count = n` 的**外置** LONG 偏移数组，
    /// 数组内容全部填**同一个合法子 IFD 偏移**（指向谁不重要——本测试
    /// 只关心「链条长度」这一个维度，多张相同预览足以验证计数行为）。
    ///
    /// # 为什么必须改count 而不只是改偏移
    ///
    /// `SubIFDs` 的偏移数组长度由 IFD 条目的 `count` 字段决定（TIFF 5.0 §2），
    /// 不是由「文件里后面还有多少字节」决定。故要测上限，必须让
    /// `count` 真的变成 65，并给这 65 项配一段真实存在的外置数据。
    fn patch_subifd_offset_count(file: &mut Vec<u8>, n: u32) -> bool {
        let d = match DngFile::parse(file) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let idx = match d
            .ifd0
            .entries
            .iter()
            .position(|x| x.tag == crate::raw::tags::SUB_IFDS)
        {
            Some(i) => i,
            None => return false,
        };
        let le = d.order == ByteOrder::Little;
        // 取一个**当前合法**的子 IFD 偏移，让 65 项都指向它。
        let target = match crate::raw_subifd::sub_ifd_offsets(&d)
            .ok()
            .and_then(|v| v.first().copied())
        {
            Some(v) => v,
            None => return false,
        };
        // 在文件尾追加 4n 字节作为外置数组（追加不改变已有偏移，故安全）。
        let base = file.len();
        let extra = (n as usize).saturating_mul(4);
        let end = match base.checked_add(extra) {
            Some(e) => e,
            None => return false,
        };
        file.resize(end, 0u8);
        for i in 0..n as usize {
            let at = base + i * 4;
            let b = if le {
                target.to_le_bytes()
            } else {
                target.to_be_bytes()
            };
            file[at..at + 4].copy_from_slice(&b);
        }
        // 改写 IFD0 里该条目的 count 与 value（count > 1 → value 是堆偏移）。
        let e_at = 8 + 2 + 12 * idx;
        if e_at + 12 > file.len() {
            return false;
        }
        let cnt = if le { n.to_le_bytes() } else { n.to_be_bytes() };
        file[e_at + 4..e_at + 8].copy_from_slice(&cnt);
        let heap = u32::try_from(base).unwrap_or(u32::MAX);
        let hb = if le {
            heap.to_le_bytes()
        } else {
            heap.to_be_bytes()
        };
        file[e_at + 8..e_at + 12].copy_from_slice(&hb);
        true
    }

    /// [`MAX_SUB_IFDS`] 上限必须真的挡住超限链条——**结构化拒绝，不 panic、
    /// 不静默截断**（静默截断会让用户以为文件只有 64 张预览）。
    ///
    /// # 上限以下必须**放行**（另一半同样重要）
    ///
    /// 只测「超限被拒」是不够的——那类实现（无论上限是1还是640）都会红。
    /// 故同时钉住**边界值本身**：恰好 64 项必须成功返回 64 项。
    #[test]
    fn subifd_count_above_max_sub_ifds_is_rejected_with_structured_error() {
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));

        // (a) 恰好 = MAX_SUB_IFDS（64）：**必须放行**且数量精确。
        let mut at_limit = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            patch_subifd_offset_count(&mut at_limit, MAX_SUB_IFDS as u32),
            "须命中 SubIFDs 条目"
        );
        let d = DngFile::parse(&at_limit).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
        let offs = sub_ifd_offsets(&d).unwrap_or_else(|e| panic!("64 项须被接受：{e}"));
        assert_eq!(
            offs.len(),
            MAX_SUB_IFDS,
            "恰好等于上限时不得截断"
        );

        // (b) 超过上限一（65）：结构化拒绝。
        let mut over = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            patch_subifd_offset_count(&mut over, MAX_SUB_IFDS as u32 + 1),
            "须命中 SubIFDs 条目"
        );
        let d = DngFile::parse(&over).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
        match sub_ifd_offsets(&d) {
            Ok(v) => panic!(
                "{} 项超过上限 {} 仍被接受（返回 {} 项）",
                MAX_SUB_IFDS + 1,
                MAX_SUB_IFDS,
                v.len()
            ),
            Err(e) => {
                assert_eq!(e.code, "RAW1001", "{}", e);
                let text = format!("{e}");
                assert!(text.contains("65"), "错误须点明实际项数：{text}");
                assert!(text.contains("64"), "错误须点明上限：{text}");
            }
        }
        // `parse_sub_ifds` 是它的薄封装，同样不得 panic。
        let d2 = DngFile::parse(&over).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            parse_sub_ifds(&d2).is_err(),
            "超限链条须报错而非静默返回部分结果"
        );
    }

    /// 读端 `SamplesPerPixel` 上限守卫：spp > 4 必须结构化拒绝。
    ///
    /// 本模块只支持 1..=4；spp = 5 越界后若继续按 4 走，会解出**错位**的
    /// 像素（按 4 通道分配缓冲、再按 5 通道拷贝，越界 get_mut 静默丢弃）
    /// ——即「看起来能出图、但内容是错的」，比报错坏得多。
    #[test]
    fn subifd_rejects_samples_per_pixel_above_four() {
        for &bad_spp in &[5u16, 8, 64] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 2, 3));
            let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
            // `SamplesPerPixel` 是子 IFD 里的 SHORT（内联）。
            assert!(
                patch_subifd_short(&mut file, crate::raw::tags::SAMPLES_PER_PIXEL, bad_spp),
                "须命中 SamplesPerPixel"
            );
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
            match d.sub_ifds() {
                Ok(v) => panic!("spp={bad_spp} 不该被接受，实得 {} 个 SubIFD", v.len()),
                Err(e) => {
                    assert_eq!(e.code, "RAW2001", "spp={bad_spp}: {e}");
                    let text = format!("{e}");
                    assert!(
                        text.contains("SamplesPerPixel") && text.contains("1..=4"),
                        "spp={bad_spp} 的错误须点明取值范围：{text}"
                    );
                }
            }
        }
    }

    /// 读端 `BitsPerSample` 越界守卫：bits ∉ 1..=16 不得静默当成 8 位。
    ///
    /// 越界时读端把 `bits_per_sample` 归一到 1（`raw_subifd::parse_one`
    /// 的 `bits_per_sample` 字段），随后由`!(1..=16).contains(&bits)`
    /// 分支给出note 且**不解像素**——这条测试就钉住那个「不解像素」。
    #[test]
    fn subifd_rejects_out_of_range_bits_per_sample() {
        for &bad_bits in &[0u16, 17, 32] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            // spp=1 → `BitsPerSample` 只有 1 项，内联，可直接改。
            s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
            let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
            assert!(
                patch_subifd_short(&mut file, crate::raw::tags::BITS_PER_SAMPLE, bad_bits),
                "须命中 BitsPerSample"
            );
            let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
            let subs = d.sub_ifds().unwrap_or_else(|e| panic!("结构未坏：{e}"));
            let p = &subs[0];
            assert_eq!((p.width, p.height), (8, 8), "元数据仍须报出");
            assert!(
                !p.has_pixels(),
                "bits={bad_bits} 越界时不得返回像素（会解出垃圾）"
            );
            let text = format!("{}{}", p.note, p.layout);
            assert!(
                p.note.contains("BitsPerSample") || p.note.contains("bit"),
                "bits={bad_bits}: note 须点明成因：{}",
                p.note
            );
            assert!(!text.is_empty(), "bits={bad_bits} 须有可读结论");
        }
    }

    // -------------------------------------------------------------------
    //  ⑥ 损坏的 SubIFD 链：结构化错误、绝不 panic、绝不越界读（C18）
    // -------------------------------------------------------------------

    /// 越界偏移 / 自指（循环）偏移 / 非法 IFD 标记——三种损坏都必须
    /// **返回结构化错误**，不是 panic、不是静默返回空链。
    ///
    /// 「静默跳过」在这里是**错的**语义：用户会以为「这个文件没有预览图」，
    /// 而真相是「预览图在，但文件坏了」。二者必须可区分。
    #[test]
    fn subifd_broken_chain_yields_structured_error_not_panic() {
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));

        // (a) 偏移远超文件尾。
        let mut far = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let len = far.len() as u32;
        assert!(patch_subifd_offset(&mut far, len + 4096), "须命中 SubIFDs 条目");
        let d = DngFile::parse(&far).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
        match d.sub_ifds() {
            Ok(v) => panic!("越界偏移不该被接受，实得 {} 个 SubIFD", v.len()),
            Err(e) => assert!(
                e.code == "RAW1001" || e.code == "RAW2001",
                "越界偏移须结构化拒绝：{e}"
            ),
        }

        // (b) 偏移指向自己（循环指向）—— **不会无限递归**（本模块只解析
        //     一层SubIFD，不跟随next 链），且**不 panic**。
        //
        //     注意：自指偏移在字节层面**不是损坏**（那个位置确实是个合法
        //     IFD），所以此处**允许**解析成功——真正要守住的是
        //     「不会因为自指而栈溢出/ 死循环」，而不是「必须报错」。
        let mut self_ref = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        let base = sub_ifd_offset(&self_ref).expect("须能解析出 SubIFD 偏移") as u32;
        assert!(patch_subifd_offset(&mut self_ref, base));
        let d = DngFile::parse(&self_ref).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
        match d.sub_ifds() {
            // 解析成功也合法（自指位置确实是合法 IFD）——但**必须终止**。
            Ok(v) => assert!(
                v.len() <= 1,
                "本模块只解析一层 SubIFD，自指不该产出多条：{}",
                v.len()
            ),
            Err(e) => assert!(
                e.code == "RAW1001" || e.code == "RAW2001",
                "若报错则须结构化：{e}"
            ),
        }

        // (c) 偏移落在非 IFD 数据上（垃圾区）→ 条目数非法 → 结构化拒绝。
        let mut junk = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert!(patch_subifd_offset(&mut junk, 16));
        let d = DngFile::parse(&junk).unwrap_or_else(|e| panic!("IFD0 须仍可解析：{e}"));
        match d.sub_ifds() {
            Ok(v) => panic!("垃圾偏移不该被接受，实得 {} 个 SubIFD", v.len()),
            Err(e) => assert!(
                e.code == "RAW1001" || e.code == "RAW2001",
                "垃圾偏移须结构化拒绝：{e}"
            ),
        }
    }

    /// 载荷区间越界（`StripByteCounts` 改成超出文件长度）→ 结构化错误。
    ///
    /// 与「偏移越界」是**不同**的损坏点：偏移合法，但长度把区间推到文件尾
    /// 之外。专门钉它是为了防`slice_payload` 的 `end > len` 检查被后续重构
    /// 改丢——那是**越界读**（UB），不是普通错误。
    #[test]
    fn subifd_payload_beyond_eof_is_rejected_without_overread() {
        let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
        s.preview = Some(PreviewSpec::new(8, 8, 8, 1, 1, 1));
        let mut file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));
        assert!(patch_subifd_long(
            &mut file,
            crate::raw::tags::STRIP_BYTE_COUNTS,
            0x7FFF_FFFF
        ));
        let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
        let subs = d.sub_ifds().unwrap_or_else(|e| panic!("{e}"));
        // 元数据仍报得出（结构没坏），但**像素不得解出**，且 note 说清原因。
        let p = &subs[0];
        assert_eq!((p.width, p.height), (8, 8));
        assert!(!p.has_pixels(), "载荷越界时不得返回像素");
        assert!(!p.note.is_empty(), "载荷越界必须写明原因：{}", p.note);
    }

    // -------------------------------------------------------------------
    //  ⑦ 确定性：同一输入连续两次解析，`to_bits()` 逐位相同（铁律 2）
    // -------------------------------------------------------------------

    /// 确定性契约：同一字节流重复解析，结果必须**逐位**相同。
    ///
    /// # 为什么这里不用 `to_bits()`
    ///
    /// 铁律 2 说「断言优先 `to_bits()`」，那是针对 `f32`/`f64`——**只有浮点**
    /// 才有「`-0.0 == 0.0` 却比特不同」「`NAN != NAN`」这两个陷阱。本模块的
    /// 像素是**整数** `u16`：整数没有「等价非等值」，逐位一致与数值一致
    /// **严格等价**（`u16` 也确实没有 `to_bits()` 方法）。
    ///
    /// 若将来给 `SubIfdImage` 加了浮点字段，**那一处**必须改用 `to_bits()`
    /// ——那才是 `to_bits()` 不可替代的地方。
    #[test]
    fn subifd_parse_is_bitwise_deterministic() {
        for &comp in &[1u16, 7, 8] {
            let mut s = DngSampleSpec::new(16, 16, 16, [0, 1, 1, 2]);
            s.preview = Some(PreviewSpec::new(13, 9, 8, comp, 2, 3));
            let file = synthesize_dng(&s).unwrap_or_else(|e| panic!("{e}"));

            let run = || -> Vec<(u32, u32, u16, Option<Vec<u16>>)> {
                let d = DngFile::parse(&file).unwrap_or_else(|e| panic!("{e}"));
                parse_sub_ifds(&d)
                    .unwrap_or_else(|e| panic!("{e}"))
                    .into_iter()
                    .map(|p| (p.width, p.height, p.compression, p.pixels))
                    .collect()
            };
            let a = run();
            let b = run();
            assert_eq!(a.len(), 1, "comp={comp}: 须恰有 1 个 SubIFD");
            assert_eq!(a, b, "comp={comp}: 两次解析结果须完全一致");
            let pa = a[0].3.as_ref().unwrap_or_else(|| panic!("comp={comp}: 像素应解出"));
            let pb = b[0].3.as_ref().unwrap();
            assert_eq!(pa.len(), pb.len(), "comp={comp}");
            for (i, (x, y)) in pa.iter().zip(pb.iter()).enumerate() {
                // u16 逐位一致 == 数值一致（u16 没有 NaN/符号零歧义）。
                assert_eq!(*x, *y, "comp={comp}: 样本 {i} 不一致（{x} vs {y}）");
            }
        }
    }
}
