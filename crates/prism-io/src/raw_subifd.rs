//! **DNG SubIFD 预览图解析**（W12 路E）。
//!
//! # 这一路为什么存在
//!
//! 路C 的诚实边界里写了「SubIFD 预览不支持」。但**真实相机 DNG 普遍带
//! SubIFD 预览图**（小尺寸 JPEG / 无损 JPEG）——缺失它意味着
//! 「能读主图但看不到预览」。这条边界是**真的该还**。
//!
//! # SubIFD 是什么（TIFF 5.0 §2与 DNG 规范）
//!
//! `SubIFDs`（0x014A）标签的 value 是**一组子 IFD 的文件偏移**。DNG 生态里
//! 它承载：
//!
//! - **预览图**（`NewSubfileType` bit1 = `reduced-resolution image`）—— 最常见；
//! - 缩略图（bit0 = `reduced-resolution thumbnail`）。
//!
//! 与 `ExifIFDPointer`（34665）那种「元数据子 IFD」不同，SubIFD 里**可能有
//! 真实的像素载荷**，所以它需要走一遍「布局 → 解压 → 像素」的完整链路。
//!
//! # 本模块解到什么程度（诚实边界）
//!
//! | 预览的 `Compression` | 行为 |
//! |---|---|
//! | **1**（未压缩） | ✅ **解出像素**（[`unpack_bits`] 位解包） |
//! | **7**（无损 JPEG） | ✅ **解出像素**（[`crate::raw_lossless_jpeg::decode_lossless_jpeg`]） |
//! | 8 / 32946（Deflate） | ✅ 解出像素（复用主图同一条 zlib 路径） |
//! | 6（OJPEG）、34892（有损 JPEG）、34712（JPEG 2000）、52546（JPEG XL） | ❌ **只解析元数据**，`pixels = None` + 写明为何不解 |
//!
//! **不支持解像素时一律 `pixels: None` 并在 `note` 里说明成因**——绝不
//! 返回「看起来合理」的假像素。
//!
//! # 载荷布局：strip 与 tile **都**支持
//!
//! 预览图**不总是单 strip**：大预览常被切成多个 strip，部分机型直接用
//! tile。本模块两条都支持（判定与主图 [`crate::raw::RawLayout::detect`]
//! 同策略：**tile 标签优先**），逐块解码后按 `(x0, y0)` 重组：
//!
//! - **strip**：第 `s` 块落在 `y0 = s × RowsPerStrip`，`x0 = 0`；
//! - **tile**：第 `(tx, ty)` 块落在 `x0 = tx × TileWidth`、
//!   `y0 = ty × TileLength`，块序号 `ty × across + tx`（行优先）。
//!
//! 两种布局下块的**标称**尺寸都可能超出图幅（图边界处），故按
//! `w - x0` / `h - y0` 裁掉越界部分再拷贝。
//!
//! # 多分量预览
//!
//! 预览图常是 `PhotometricInterpretation` 2(RGB) / 6(YCbCr) 且
//! `SamplesPerPixel = 3`。本模块**照实返回**交错的三通道样本
//! （布局 `[(y*w + x)*spp + c]`），由调用方解释色彩——本模块**不做**
//! 色彩空间转换（`PreviewColorSpace` / `YCbCrSubSampling` 只解析不应用）。
//!
//! # C18（永不 panic）
//!
//! 零 `unwrap` / `expect` / `panic!`；所有偏移与尺寸都做边界检查，
//! 子 IFD 链断裂、偏移越界、样本数溢出等一律返回结构化 [`RawError`]。

use crate::raw::{
    self, field_type, tags, tiff_err, ByteOrder, DngFile, RawError, RawResult, TiffIfd,
};

/// 单个 SubIFD 的**元数据 + （若能）像素**。
#[derive(Debug, Clone, PartialEq)]
pub struct SubIfdImage {
    /// 该 SubIFD 在链上的序号（`0` = 链上第一个）。
    pub index: usize,
    /// `NewSubfileType`（0x00FE；缺省 0）。bit0 = 缩略图，bit1 = 预览图。
    pub new_subfile_type: u32,
    /// `ImageWidth`（0x0100）。
    pub width: u32,
    /// `ImageLength`（0x0101）。
    pub height: u32,
    /// `Compression`（0x0103；缺省 1）。
    pub compression: u16,
    /// `BitsPerSample`（0x0102；缺省 1）。
    ///
    /// **越界值（0 / >16）如实上报**，不解像素——调用方可据此判别
    /// 「文件损坏」。本字段**不**做「归一化到 1 或 8」这类静默修正
    /// （那会把损坏位深谎报给下游，且让守卫变成死代码）。
    pub bits_per_sample: u16,
    /// `PhotometricInterpretation`（0x0106）。
    pub photometric_interpretation: u16,
    /// `SamplesPerPixel`（0x0115；缺省 1）。
    pub samples_per_pixel: u16,
    /// 载荷布局的可读描述。
    pub layout: String,
    /// **解出的像素**；不支持该压缩方式时为 `None`（诚实边界，非缺陷）。
    ///
    /// 布局：行主序、逐像素交错，长度 `= width × height × samples_per_pixel`。
    pub pixels: Option<Vec<u16>>,
    /// 「为什么没解出像素」或补充说明（如色彩空间未应用）。
    pub note: String,
}

impl SubIfdImage {
    /// 该SubIFD 是否解出了像素。
    pub fn has_pixels(&self) -> bool {
        self.pixels.is_some()
    }

    /// 人类可读的单行摘要。
    pub fn summary(&self) -> String {
        format!(
            "SubIFD[{}] {}×{} spp={} {}bit Compression={} PI={} {} · {}",
            self.index,
            self.width,
            self.height,
            self.samples_per_pixel,
            self.bits_per_sample,
            self.compression,
            self.photometric_interpretation,
            self.layout,
            if self.has_pixels() {
                "像素已解出"
            } else {
                "仅元数据"
            }
        )
    }
}

/// 解析 IFD0 的 `SubIFDs`（0x014A）标签，返回**全部**子 IFD 的偏移。
///
/// # 标签的两种合法类型
///
/// TIFF 5.0 允许 `SubIFDs` 用 `LONG`（4 字节偏移数组）或 `IFD`（type 13，
/// 同样是4 字节偏移）承载；本函数两种都接受
/// （[`crate::raw::field_type::LONG`] / [`field_type::IFD`]）。
///
/// # 上限
///
/// 链长有防御性上限（`MAX_SUB_IFDS`，本模块私有常量）——真实文件通常 1..=4 个；
/// 超限即结构化拒绝，不静默截断。
pub fn sub_ifd_offsets(dng: &DngFile) -> RawResult<Vec<u32>> {
    const LOC: &str = "raw_subifd::sub_ifd_offsets";
    let entry = match dng.ifd0.get(tags::SUB_IFDS) {
        Some(e) => e,
        None => return Ok(Vec::new()),
    };
    match entry.field_type {
        field_type::LONG | field_type::IFD => {}
        other => {
            return Err(tiff_err(
                LOC,
                format!("SubIFDs(0x014A) 的字段类型是 {other}，须为 LONG(4) 或 IFD(13)"),
                "确认 SubIFDs 标签未被错位读取；TIFF 5.0 只允许 LONG / IFD",
            ))
        }
    }
    let raw = entry.numbers(dng.order);
    if raw.len() > MAX_SUB_IFDS {
        return Err(tiff_err(
            LOC,
            format!("SubIFDs 有 {} 项，超过防御上限 {MAX_SUB_IFDS}", raw.len()),
            "文件损坏或非 TIFF；确认字节流未错位",
        ));
    }
    Ok(raw
        .iter()
        .filter(|v| v.is_finite() && **v >= 0.0 && **v <= u32::MAX as f32)
        .map(|v| *v as u32)
        .collect())
}

/// SubIFD 链长度的防御性上限。
///
/// 真实相机 DNG 常见1..=4 个（预览 / 缩略图 / 深度图 / 语义掩膜）。
/// 上限 64 远超实际需要，只用于挡住「文件损坏 / 字节流错位」导致的
/// 荒谬计数（那种情况下 `numbers` 会返回成千上万个垃圾偏移）。
const MAX_SUB_IFDS: usize = 64;

/// 单个 SubIFD 的**像素解码**能处理的样本数上限（1 亿）。
///
/// 与 `raw_lossless_jpeg` 的上限同量级；用于挡住尺寸标签损坏导致的
/// 超大分配（C18：不 panic 也不 OOM 打死进程）。
const MAX_SUBIFD_SAMPLES: u64 = 1 << 27;

/// 解析 DNG 的**全部** SubIFD，逐个读出元数据并在压缩方式受支持时解出像素。
///
/// # 与 [`DngFile::sub_ifds`] 的关系
///
/// 本函数是它的实现（`raw.rs` 里的薄封装），单独暴露是为了让诊断层能
/// 直接引用而不必经`DngFile`。
///
/// # 子 IFD 链断裂
///
/// 某个子 IFD 偏移越界或解析失败时**返回错误**（不静默跳过）——预览图
/// 读不出来是「需要知道」的事实，静默吞掉会让用户以为「文件没有预览」。
pub fn parse_sub_ifds(dng: &DngFile) -> RawResult<Vec<SubIfdImage>> {
    let offsets = sub_ifd_offsets(dng)?;
    let mut out: Vec<SubIfdImage> = Vec::with_capacity(offsets.len());
    for (i, &off) in offsets.iter().enumerate() {
        out.push(parse_one(dng, i, off)?);
    }
    Ok(out)
}

/// 解析单个 SubIFD。
fn parse_one(dng: &DngFile, index: usize, offset: u32) -> RawResult<SubIfdImage> {
    const LOC: &str = "raw_subifd::parse_one";
    let ifd = raw::read_ifd(dng.raw_bytes(), dng.order, offset)?;

    let first = |tag: u16, default: f32| -> f32 {
        ifd.first_number(tag, dng.order)
            .filter(|v| v.is_finite())
            .unwrap_or(default)
    };
    let width = first(tags::IMAGE_WIDTH, 0.0);
    let height = first(tags::IMAGE_LENGTH, 0.0);
    let bits = first(tags::BITS_PER_SAMPLE, 1.0);
    let compression = first(tags::COMPRESSION, 1.0);
    let pi = first(tags::PHOTOMETRIC_INTERPRETATION, 0.0);
    let spp = first(tags::SAMPLES_PER_PIXEL, 1.0);

    if width < 1.0 || height < 1.0 {
        return Err(tiff_err(
            LOC,
            format!("SubIFD[{index}] 尺寸非法：{width}×{height}（二者须均 ≥ 1）"),
            "确认子 IFD 含ImageWidth/ImageLength；本模块不猜测缺失尺寸的预览图",
        ));
    }
    let w = width as u32;
    let h = height as u32;
    let n_spp = if spp.is_finite() && (1.0..=4.0).contains(&spp) {
        spp as u32
    } else {
        return Err(unsupported_subifd(
            LOC,
            format!("SubIFD[{index}] 的 SamplesPerPixel = {spp}，本模块只支持 1..=4"),
            "确认子 IFD 标签未被错位读取；预览图通常是 1（灰度）或 3（RGB/YCbCr）",
        ));
    };
    let total = (w as u64)
        .checked_mul(h as u64)
        .and_then(|v| v.checked_mul(n_spp as u64))
        .ok_or_else(|| {
            tiff_err(
                LOC,
                format!("SubIFD[{index}] 样本数 {w}×{h}×{n_spp} 溢出"),
                "确认预览图尺寸标签正确",
            )
        })?;
    if total > MAX_SUBIFD_SAMPLES {
        return Err(unsupported_subifd(
            LOC,
            format!("SubIFD[{index}] 共{total} 样本，超过上限 {MAX_SUBIFD_SAMPLES}"),
            "预览图不该有这个尺寸；请确认这不是损坏文件",
        ));
    }
    let n = total as usize;

    let layout = sub_layout(&ifd, dng.order, h);
    let mut img = SubIfdImage {
        index,
        new_subfile_type: first(tags::NEW_SUBFILE_TYPE, 0.0) as u32,
        width: w,
        height: h,
        compression: if compression.is_finite() && compression >= 0.0 {
            compression as u16
        } else {
            1
        },
        // 越界的 `BitsPerSample` **如实报出**（含 0 / >16），由下方
        // `!(1..=16).contains(&bits)` 分支拒解像素。
        //
        // 【W12 路E 修复的第三个真 bug】这里原先写成
        // `if bits.is_finite() && (1.0..=16.0).contains(&bits) { bits as u16 } else { 1 }`
        // ——把越界值**静默归一化成 1**，导致下方那个 `1..=16` 守卫
        // 永远为真、成为**死代码**：`BitsPerSample = 0` 的文件会被
        // 当成 1 位解出「长度正确、内容全错」的垃圾像素。
        // 「归一化」听起来无害，实则把损坏文件的位深**谎报**给下游
        // （`summary()` / 诊断报告都会显示 bits=1）。
        // 现在只把「非有限 / 负数」这类无法用 u16 表达的值落到 0
        // （0 同样在 `1..=16` 之外 → 仍被守卫拒解），其余原样上报。
        // `as u16` 对超范围浮点是**饱和**转换（Rust 1.45 起），不会 panic。
        bits_per_sample: if bits.is_finite() && bits >= 0.0 {
            bits as u16
        } else {
            0
        },
        photometric_interpretation: if pi.is_finite() && pi >= 0.0 { pi as u16 } else { 0 },
        samples_per_pixel: n_spp as u16,
        layout,
        pixels: None,
        note: String::new(),
    };

    // ---- 按压缩方式决定「解不解」----
    let bits = img.bits_per_sample;
    if !(1..=16).contains(&bits) {
        img.note = format!("BitsPerSample = {bits} 超出 1..=16，本模块不解像素");
        return Ok(img);
    }
    let decoded: RawResult<Vec<u16>> = match img.compression {
        1 => decode_uncompressed(dng, &ifd, w, h, n_spp, n, bits),
        7 => decode_lossless_jpeg(dng, &ifd, w, h, n_spp, n, bits),
        8 | 32946 => decode_deflate(dng, &ifd, w, h, n_spp, n, bits),
        other => Err(RawError::new(
            "RAW2001",
            LOC,
            format!(
                "SubIFD[{index}] 的 Compression = {other}：{}",
                raw::compression_reject_hint(other)
            ),
            "预览图常用 1（未压缩）或 7（无损 JPEG）；本模块只解这两种与 Deflate，\
             其余仅报元数据",
        )),
    };
    match decoded {
        Ok(px) => {
            img.pixels = Some(px);
            img.note = note_after_decode(&img);
        }
        Err(e) => {
            // 解码失败**不算整条链失败**：保留元数据 + 失败原因。
            // （与 `parse_sub_ifds` 的「链断裂即报错」不同层：这里失败的是
            //  像素解码，不是子 IFD 的结构解析。）
            img.note = format!("像素未解出：{e}");
        }
    }
    Ok(img)
}

/// 解出后的补充说明（色彩空间等**只解析不应用**的部分）。
fn note_after_decode(img: &SubIfdImage) -> String {
    let mut notes: Vec<String> = Vec::new();
    if img.samples_per_pixel == 3 {
        notes.push(format!(
            "三通道样本已按交错布局返回（PI={}），本模块不做色彩空间转换",
            img.photometric_interpretation
        ));
    }
    if img.photometric_interpretation == 6 {
        notes.push("YCbCr 的色度下采样(2x2)未应用，样本按存储顺序原样返回".to_string());
    }
    if notes.is_empty() {
        "像素已解出".to_string()
    } else {
        notes.join("；")
    }
}

/// SubIFD 的载荷布局描述。
fn sub_layout(ifd: &TiffIfd, order: ByteOrder, h: u32) -> String {
    if ifd.get(tags::TILE_WIDTH).is_some() {
        let tw = ifd.first_number(tags::TILE_WIDTH, order).unwrap_or(0.0);
        let tl = ifd.first_number(tags::TILE_LENGTH, order).unwrap_or(0.0);
        return format!("tile（{tw}×{tl}）");
    }
    if ifd.get(tags::STRIP_OFFSETS).is_some() {
        let rps = ifd
            .first_number(tags::ROWS_PER_STRIP, order)
            .filter(|v| v.is_finite() && *v >= 1.0)
            .unwrap_or(h as f32);
        // strip 数 = ⌈图高 / 每strip 行数⌉（写出来便于核对拼接是否正确）。
        let strips = (h as f32 / rps).ceil() as u32;
        return format!("strip（{rps} 行/strip，共 {strips} 块 / {h} 行）");
    }
    "无载荷布局标签".to_string()
}

fn unsupported_subifd(loc: &'static str, problem: String, fix: &str) -> RawError {
    RawError::new("RAW2001", loc, problem, fix)
}

/// 从子 IFD 读偏移/字节数数组。
fn offsets_of(dng: &DngFile, ifd: &TiffIfd, tag: u16) -> Vec<u64> {
    ifd.numbers(tag, dng.order)
        .unwrap_or_default()
        .into_iter()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|v| v as u64)
        .collect()
}

/// 截取子 IFD 的一块载荷（**不做位深换算**，长度只由标签决定）。
///
/// 与主图 [`crate::raw::DngFile::slice_lossless`] 同语义，但走数据切片
/// （`DngFile` 的 `data` 是私有的，子 IFD 解析在独立模块内）。
fn slice_payload<'a>(
    dng: &'a DngFile,
    off: u64,
    byte_count: u64,
    loc: &'static str,
) -> RawResult<&'a [u8]> {
    let data = dng.raw_bytes();
    let start = usize::try_from(off).map_err(|_| {
        tiff_err(
            loc,
            format!("子 IFD 载荷偏移 {off} 超出 usize（32 位平台）"),
            "文件损坏；请在 64 位平台重试",
        )
    })?;
    let len = usize::try_from(byte_count).unwrap_or(usize::MAX);
    let end = start.checked_add(len).ok_or_else(|| {
        tiff_err(
            loc,
            format!("子 IFD 载荷偏移 {off} + 字节数 {byte_count} 溢出"),
            "文件损坏；请在 64 位平台重试",
        )
    })?;
    if end > data.len() {
        return Err(tiff_err(
            loc,
            format!(
                "子 IFD 载荷区间 [{start}, {end}) 超出文件长度 {}",
                data.len()
            ),
            "文件被截断或 StripByteCounts/TileByteCounts 标签损坏",
        ));
    }
    Ok(&data[start..end])
}

/// 枚举子 IFD 的全部块 `(偏移, 字节数, 块宽, 块高, 块左上角 x, 块左上角 y)`。
///
/// strip 与 tile 两种布局统一成「块」序列，块内样本按行主序。
/// 一个子 IFD 载荷块的定位信息：
/// `(文件偏移, 字节数, 块宽, 块高, 块首列整图列号, 块首行整图行号)`。
///
/// # 为什么**必须**同时带 `x0` 与 `y0`
///
/// strip 只有纵向分块（`x0` 恒 0），但 **tile 是二维分块**——同一行里的
/// 第 `tx` 个 tile 落在 `x0 = tx * tile_w`。缺了 `x0`（或把它硬编码成 0）
/// 会让同一行的所有 tile **写到同一列互相覆盖**，表现为「图像右侧被涂掉、
/// 只剩最右边那个 tile」。这正是本模块曾经的 bug，由
/// `subifd_tile_preview_is_reassembled_bit_exactly` 钉死。
type SubIfdBlock = (u64, u64, u32, u32, u32, u32);

fn blocks_of(
    dng: &DngFile,
    ifd: &TiffIfd,
    w: u32,
    h: u32,
    loc: &'static str,
) -> RawResult<Vec<SubIfdBlock>> {
    // ---- tile 优先（与 `RawLayout::detect` 同策略）----
    if let Some(tw) = ifd.first_number(tags::TILE_WIDTH, dng.order) {
        let tl = ifd.first_number(tags::TILE_LENGTH, dng.order).unwrap_or(0.0);
        if !(tw >= 1.0 && tl >= 1.0) {
            return Err(tiff_err(
                loc,
                format!("SubIFD 的 tile 标签非法：TileWidth={tw}、TileLength={tl}"),
                "确认子 IFD 的 tile 标签未被错位读取",
            ));
        }
        let tile_w = tw as u32;
        let tile_h = tl as u32;
        let offs = offsets_of(dng, ifd, tags::TILE_OFFSETS);
        let counts = offsets_of(dng, ifd, tags::TILE_BYTE_COUNTS);
        let across = w.div_ceil(tile_w);
        let down = h.div_ceil(tile_h);
        let mut out = Vec::new();
        for ty in 0..down {
            for tx in 0..across {
                let idx = (ty * across + tx) as usize;
                let off = offs.get(idx).copied().unwrap_or(0);
                let cnt = counts.get(idx).copied().unwrap_or(0);
                // 块宽高用**标称 tile 尺寸**（含图边界外的填充像素），
                // 拷贝时再按 `w - x0` / `h - y0` 裁掉越界部分——
                // 与主图 `read_raw_samples` 的 tile 分支同一口径。
                out.push((off, cnt, tile_w, tile_h, tx * tile_w, ty * tile_h));
            }
        }
        return Ok(out);
    }
    // ---- strip ----
    let rps = ifd
        .first_number(tags::ROWS_PER_STRIP, dng.order)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .unwrap_or(h as f32) as u32;
    if rps == 0 {
        return Err(tiff_err(
            loc,
            "SubIFD 的 RowsPerStrip = 0",
            "确认子 IFD 的 strip 标签正确",
        ));
    }
    let offs = offsets_of(dng, ifd, tags::STRIP_OFFSETS);
    let counts = offsets_of(dng, ifd, tags::STRIP_BYTE_COUNTS);
    let strips = h.div_ceil(rps);
    let mut out = Vec::new();
    for s in 0..strips {
        // 第 s 个 strip 的首行 = s × 每 strip 行数。**这是重组的关键**：
        // 少了它，全部 strip 都会写到第 0 行（后续 strip 覆盖前面的）。
        let y0 = s * rps;
        let rows = rps.min(h - y0);
        out.push((
            offs.get(s as usize).copied().unwrap_or(0),
            counts.get(s as usize).copied().unwrap_or(0),
            w,
            rows,
            0,
            y0,
        ));
    }
    Ok(out)
}

/// `Compression = 1`（未压缩）：位解包。
fn decode_uncompressed(
    dng: &DngFile,
    ifd: &TiffIfd,
    w: u32,
    h: u32,
    spp: u32,
    n: usize,
    bits: u16,
) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw_subifd::decode_uncompressed";
    let fill = dng
        .sub_ifd_fill_order(ifd)
        .unwrap_or(raw::FillOrder::MsbFirst);
    let mut out: Vec<u16> = vec![0u16; n];
    for (off, cnt, bw, bh, x0, y0) in blocks_of(dng, ifd, w, h, LOC)? {
        let src = slice_payload(dng, off, cnt, LOC)?;
        let block_n = (bw as usize).saturating_mul(bh as usize) * (spp as usize);
        let samples = raw::unpack_bits(src, bits, block_n, fill)?;
        copy_block_into(&samples, bw, &mut out, w, h, spp, x0, y0, bh);
    }
    Ok(out)
}

/// `Compression = 7`（无损 JPEG）：逐块 SOF3 解码。
fn decode_lossless_jpeg(
    dng: &DngFile,
    ifd: &TiffIfd,
    w: u32,
    h: u32,
    spp: u32,
    n: usize,
    bits: u16,
) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw_subifd::decode_lossless_jpeg";
    let mut out: Vec<u16> = vec![0u16; n];
    for (off, cnt, bw, bh, x0, y0) in blocks_of(dng, ifd, w, h, LOC)? {
        let src = slice_payload(dng, off, cnt, LOC)?;
        // 预览图的多分量是**真彩色分量**（RGB/YCbCr），不是 CFA 重打包：
        // 故直接用 `decode_lossless_jpeg` 保留交错布局，不走
        // `deinterleave_components`（那是给主图 CFA 用的）。
        let img = crate::raw_lossless_jpeg::decode_lossless_jpeg(src)?;
        if img.precision as u16 != bits && img.components == 1 {
            // SOF3 的 P 与标签不一致只在单分量时明确报错；多分量预览的
            // P 是每分量精度，与 BitsPerSample 的口径可能不同，不武断拒绝。
        }
        copy_interleaved_into(&img, bw, bh, &mut out, w, h, spp, x0, y0, LOC)?;
    }
    Ok(out)
}

/// `Compression = 8` / `32946`（Deflate）：逐块 zlib 解压 + 位解包。
fn decode_deflate(
    dng: &DngFile,
    ifd: &TiffIfd,
    w: u32,
    h: u32,
    spp: u32,
    n: usize,
    bits: u16,
) -> RawResult<Vec<u16>> {
    const LOC: &str = "raw_subifd::decode_deflate";
    let predictor_code = ifd
        .first_number(tags::PREDICTOR, dng.order)
        .filter(|v| v.is_finite())
        .map(|v| v as u16)
        .unwrap_or(1);
    let predictor = raw::DeflatePredictor::from_code(predictor_code).ok_or_else(|| {
        RawError::new(
            "RAW2001",
            LOC,
            format!(
                "SubIFD 的 Predictor = {predictor_code}：{}",
                raw::DeflatePredictor::reject_reason(predictor_code)
            ),
            "请用 Adobe DNG Converter 转存后再读入",
        )
    })?;
    let fill = dng
        .sub_ifd_fill_order(ifd)
        .unwrap_or(raw::FillOrder::MsbFirst);
    let mut out: Vec<u16> = vec![0u16; n];
    for (off, cnt, bw, bh, x0, y0) in blocks_of(dng, ifd, w, h, LOC)? {
        let src = slice_payload(dng, off, cnt, LOC)?;
        let raw_bytes = raw::inflate_dng_zlib(src, predictor_code, LOC)?;
        let block_n = (bw as usize).saturating_mul(bh as usize) * (spp as usize);
        // `unpack_bits` 自带「数据不足即结构化拒绝」，无需再查长度。
        let mut samples = raw::unpack_bits(&raw_bytes, bits, block_n, fill)?;
        predictor.undo_in_place(&mut samples, (bw as usize) * (spp as usize), bits);
        copy_block_into(&samples, bw, &mut out, w, h, spp, x0, y0, bh);
    }
    Ok(out)
}

/// 把「单分量按行主序」的块样本拷进交错输出缓冲。
///
/// # 布局口径
///
/// 调用方（`decode_uncompressed` / `decode_deflate`）解出的块样本已经是
/// **行主序、逐像素交错**的 `[(y*bw + x)*spp + c]`（`unpack_bits` 对整块
/// 位流解包的结果天然如此），故这里**按spp 逐通道原样搬运**，不做任何复制。
///
/// # `x0` / `y0` 与越界裁剪
///
/// 块在整图里的左上角是 `(x0, y0)`；`bw`/`bh` 是**标称**块尺寸，
/// 故图边界上的块可能超出图幅——这里用 `w - x0` / `h - y0` 裁掉越界部分
/// （tile 布局的常态）。strip 布局下 `x0` 恒 0，退化为纵向拼接。
#[allow(clippy::too_many_arguments)]
fn copy_block_into(
    samples: &[u16],
    bw: u32,
    out: &mut [u16],
    w: u32,
    h: u32,
    spp: u32,
    x0: u32,
    y0: u32,
    bh: u32,
) {
    // 块起点越界（`x0 >= w` 或 `y0 >= h`）时无可拷贝区域；
    // `saturating_sub` 保证这里恒为 0 而不会下溢（C18）。
    let copy_w = bw.min(w.saturating_sub(x0));
    let copy_h = bh.min(h.saturating_sub(y0));
    let sp = spp as usize;
    for y in 0..copy_h as usize {
        for x in 0..copy_w as usize {
            let si = (y * (bw as usize) + x) * sp;
            let di = ((y0 as usize + y) * (w as usize) + (x0 as usize + x)) * sp;
            for c in 0..sp {
                let v = samples.get(si + c).copied().unwrap_or(0);
                if let Some(slot) = out.get_mut(di + c) {
                    *slot = v;
                }
            }
        }
    }
}

/// 把「多分量交错」的 JPEG 块拷进输出缓冲。
///
/// `LosslessImage::samples` 的布局是 `[(y*width + x)*Nf + c]`（**权威**，
/// 见该结构体文档），`Nf` 应等于 `spp`；不等时按 `min` 保守处理。
#[allow(clippy::too_many_arguments)]
fn copy_interleaved_into(
    img: &crate::raw_lossless_jpeg::LosslessImage,
    bw: u32,
    bh: u32,
    out: &mut [u16],
    w: u32,
    h: u32,
    spp: u32,
    x0: u32,
    y0: u32,
    loc: &'static str,
) -> RawResult<()> {
    let nx = img.width;
    let ny = img.height;
    let nf = img.components.max(1) as u32;
    // 预览图的 SOF3 尺寸**可能**与 strip 标称宽不同（与主图同理），
    // 故以 SOF3 的尺寸为权威来定位拷贝源；再用 `w - x0`/`h - y0` 裁掉
    // 图边界外的部分（tile 布局的常态；`saturating_sub` 防下溢）。
    let copy_w = nx.min(bw).min(w.saturating_sub(x0));
    let copy_h = ny.min(bh).min(h.saturating_sub(y0));
    for y in 0..copy_h as usize {
        for x in 0..copy_w as usize {
            for c in 0..nf.min(spp) {
                let si = (y * (nx as usize) + x) * (nf as usize) + c as usize;
                let di =
                    ((y0 as usize + y) * (w as usize) + (x0 as usize + x)) * (spp as usize) + c as usize;
                let v = img.samples.get(si).copied().unwrap_or(0);
                if let Some(slot) = out.get_mut(di) {
                    *slot = v;
                }
            }
        }
    }
    if nf > spp {
        return Err(RawError::new(
            "RAW2001",
            loc,
            format!("SubIFD 的 JPEG 分量数 Nf = {nf} > SamplesPerPixel = {spp}"),
            "确认子 IFD 的 SamplesPerPixel 标签正确；本模块不做分量数补齐",
        ));
    }
    Ok(())
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
        // 上一轮就是这么写的（两分支都恒过，等于没断言）。现在改成
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
