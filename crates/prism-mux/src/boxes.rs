//! ISO-BMFF 盒原语（C4；路线图 C4 行 / §3.5：ftyp/moov/trak/mdat，手写）。
//!
//! 提供最小的 ISO Base Media File Format 盒构建原语：`FullBox`（版本+标志）、
//! 容器盒（内含子盒）、以及 `mdat` 媒体数据盒以**大盒（64 位尺寸）**形式承载
//! 超过 4 GiB 的可能。
//!
//! # 结构约定
//!
//! - 每个盒 = `size(4) + type(4) + [largesize(8)] + payload`；`size == 1` 时
//!   为 64 位尺寸；`size == 0` 意为"到文件末尾"。
//! - `moov` 内 `stbl` 使用 `stts/stsc/stsz/stco/stss`（MJPEG 逐帧同步样本）。
//! - 所有多字节整数大端（网络字节序）。

/// 写一个基本盒：`type + payload`（32 位尺寸）。
///
/// # 4 GiB 边界
///
/// `size` 字段是 u32，载荷超过 `u32::MAX - 8` 时无法用 32 位尺寸表达。ISO/IEC
/// 14496-12 §4.2 定义了 `size == 1` 的 **64 位 largesize** 扩展形式，本引擎的
/// 大载荷路径（[`write_mdat`]）已走该形式。故此处对超限情形**钳制到 u32::MAX**
/// 并置 `debug_assert`——宁可在 debug 下暴露，也不让 `as u32` 静默回绕成
/// 一个指向错误偏移的小尺寸（那会产出结构错乱但"能播"的文件）。
pub fn write_box(out: &mut Vec<u8>, box_type: &[u8; 4], payload: &[u8]) {
    const HEADER: usize = 8; // size(4) + type(4)
    debug_assert!(
        payload.len() + HEADER <= u32::MAX as usize,
        "盒 {} 载荷 {} 字节超出 32 位 size 表达范围；大载荷应改用 64 位 largesize（见 write_mdat）",
        std::str::from_utf8(box_type).unwrap_or("?"),
        payload.len()
    );
    let size: u32 = (payload.len() + HEADER).min(u32::MAX as usize) as u32;
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
}

/// 写一个 FullBox：`type + version(1) + flags(3) + payload`。
pub fn write_full_box(out: &mut Vec<u8>, box_type: &[u8; 4], version: u8, flags: u32, payload: &[u8]) {
    let mut body: Vec<u8> = Vec::with_capacity(4 + payload.len());
    body.push(version);
    body.extend_from_slice(&(flags & 0x00FF_FFFF).to_be_bytes()[1..]);
    body.extend_from_slice(payload);
    write_box(out, box_type, &body);
}

/// 写一个容器盒：`type + children`（子盒直接拼接）。
pub fn write_container(out: &mut Vec<u8>, box_type: &[u8; 4], children: &[u8]) {
    write_box(out, box_type, children);
}

/// 写 `mdat`：大盒形式（64 位尺寸），承载媒体样本字节。
///
/// `data` 为全部样本的连续字节；返回盒字节流（供 chunk offset 计算）。
pub fn write_mdat(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(16 + data.len());
    out.extend_from_slice(&1_u32.to_be_bytes()); // size = 1 → 大盒
    out.extend_from_slice(b"mdat");
    out.extend_from_slice(&((16 + data.len()) as u64).to_be_bytes());
    out.extend_from_slice(data);
    out
}

/// 写 `ftyp`（major_brand + minor_version + compatible_brands）。
pub fn write_ftyp(major: &[u8; 4], compatible: &[&[u8; 4]]) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(major);
    payload.extend_from_slice(&0_u32.to_be_bytes()); // minor_version
    for c in compatible {
        payload.extend_from_slice(*c);
    }
    let mut out: Vec<u8> = Vec::new();
    write_box(&mut out, b"ftyp", &payload);
    out
}

/// 写 `stts`（Time-to-Sample）：单条目全样本同长。
pub fn stts_uniform(count: u32, delta: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1_u32.to_be_bytes()); // entry_count
    payload.extend_from_slice(&count.to_be_bytes());
    payload.extend_from_slice(&delta.to_be_bytes());
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stts", 0, 0, &payload);
    out
}

/// 写 `stsc`（Sample-to-Chunk）：单条目，每 chunk 一个样本。
pub fn stsc_one_per_chunk(_count: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1_u32.to_be_bytes()); // entry_count
    payload.extend_from_slice(&1_u32.to_be_bytes()); // first_chunk
    payload.extend_from_slice(&1_u32.to_be_bytes()); // samples_per_chunk
    payload.extend_from_slice(&1_u32.to_be_bytes()); // sample_description_index
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stsc", 0, 0, &payload);
    out
}

/// 写 `stsz`（Sample Sizes）：逐样本尺寸。
pub fn stsz(sizes: &[u32]) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u32.to_be_bytes()); // sample_size = 0（逐样本给出）
    payload.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    for s in sizes {
        payload.extend_from_slice(&s.to_be_bytes());
    }
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stsz", 0, 0, &payload);
    out
}

/// 写 `stco`（Chunk Offset 32 位）或 `co64`（64 位）。
pub fn stco(offsets: &[u32]) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&(offsets.len() as u32).to_be_bytes());
    for o in offsets {
        payload.extend_from_slice(&o.to_be_bytes());
    }
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stco", 0, 0, &payload);
    out
}

/// 写 `stss`（Sync Sample Table）：全部样本均为同步样本。
pub fn stss_all(count: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&(count).to_be_bytes());
    for i in 1..=count {
        payload.extend_from_slice(&i.to_be_bytes());
    }
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stss", 0, 0, &payload);
    out
}

/// 写 `mvhd`（Movie Header）。
pub fn mvhd(timescale: u32, duration: u32, next_track: u32) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u32.to_be_bytes()); // creation_time
    payload.extend_from_slice(&0_u32.to_be_bytes()); // modification_time
    payload.extend_from_slice(&timescale.to_be_bytes());
    payload.extend_from_slice(&duration.to_be_bytes());
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes()); // rate = 1.0
    payload.extend_from_slice(&0x0100_u16.to_be_bytes()); // volume = 1.0
    payload.extend_from_slice(&0_u16.to_be_bytes()); // reserved
    payload.extend_from_slice(&[0_u8; 8]); // reserved[2]
    // matrix（单位矩阵，16.16/2.30 定点）。
    let matrix: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
    for m in matrix {
        payload.extend_from_slice(&m.to_be_bytes());
    }
    payload.extend_from_slice(&[0_u8; 24]); // pre_defined[6]
    payload.extend_from_slice(&next_track.to_be_bytes());
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"mvhd", 0, 0, &payload);
    out
}

/// 写 `tkhd`（Track Header）。
pub fn tkhd(track_id: u32, duration: u32, width: u32, height: u32, volume: u16) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u32.to_be_bytes()); // creation_time
    payload.extend_from_slice(&0_u32.to_be_bytes()); // modification_time
    payload.extend_from_slice(&track_id.to_be_bytes());
    payload.extend_from_slice(&0_u32.to_be_bytes()); // reserved
    payload.extend_from_slice(&duration.to_be_bytes());
    payload.extend_from_slice(&[0_u8; 8]); // reserved[2]
    payload.extend_from_slice(&0_u16.to_be_bytes()); // layer
    payload.extend_from_slice(&0_u16.to_be_bytes()); // alternate_group
    payload.extend_from_slice(&volume.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes()); // reserved
    let matrix: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
    for m in matrix {
        payload.extend_from_slice(&m.to_be_bytes());
    }
    payload.extend_from_slice(&(width << 16).to_be_bytes()); // 16.16
    payload.extend_from_slice(&(height << 16).to_be_bytes());
    // flags = 3（enabled + in_movie）。
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"tkhd", 0, 0x0000_0003, &payload);
    out
}

/// 写 `mdhd`（Media Header）。
pub fn mdhd(timescale: u32, duration: u32, language: u16) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(&0_u32.to_be_bytes());
    payload.extend_from_slice(&timescale.to_be_bytes());
    payload.extend_from_slice(&duration.to_be_bytes());
    payload.extend_from_slice(&language.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes()); // pre_defined
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"mdhd", 0, 0, &payload);
    out
}

/// 写 `hdlr`（Handler Reference）。
pub fn hdlr(handler_type: &[u8; 4], name: &str) -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u32.to_be_bytes()); // pre_defined
    payload.extend_from_slice(handler_type);
    payload.extend_from_slice(&[0_u8; 12]); // reserved[3]
    payload.extend_from_slice(name.as_bytes());
    payload.push(0); // NUL 终止
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"hdlr", 0, 0, &payload);
    out
}

/// 写 `vmhd`（视频媒体头）。
pub fn vmhd() -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u16.to_be_bytes()); // graphicsmode
    payload.extend_from_slice(&[0_u8; 6]); // opcolor[3]
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"vmhd", 0, 0x0000_0001, &payload);
    out
}

/// 写 `smhd`（声音媒体头）。
pub fn smhd() -> Vec<u8> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&0_u16.to_be_bytes()); // balance
    payload.extend_from_slice(&0_u16.to_be_bytes()); // reserved
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"smhd", 0, 0, &payload);
    out
}

/// 写 `dinf`/`dref`/`url `（数据引用：自包含）。
pub fn dinf_self() -> Vec<u8> {
    let mut url: Vec<u8> = Vec::new();
    write_full_box(&mut url, b"url ", 0, 0x0000_0001, &[]); // flag 1 = self-contained
    let mut dref_body: Vec<u8> = Vec::new();
    dref_body.extend_from_slice(&1_u32.to_be_bytes()); // entry_count
    dref_body.extend_from_slice(&url);
    let mut dref: Vec<u8> = Vec::new();
    write_full_box(&mut dref, b"dref", 0, 0, &dref_body);
    let mut out: Vec<u8> = Vec::new();
    write_container(&mut out, b"dinf", &dref);
    out
}

/// 写视频 `stsd`（MJPEG）：`mjpg` 视觉样本条目 + `esds`-free 最小负载。
///
/// # 分辨率上限（格式约束，非本实现的缺陷）
///
/// ISO/IEC 14496-12 §8.3.3.1 的 `VisualSampleEntry` 把 width/height 定义为
/// **u16**（本基线未采用 `tkhd` 的 16.16 定点形式），故上限 65535。超过该值
/// 若用 `as u16` 截断会**静默写出错误分辨率**（65536 → 0、100000 → 34464）——
/// 文件能播但画幅错乱，属最恶劣的静默错数据。故此处**入参校验并返回错误**，
/// 不钳制、不 panic（宪章 C18）。
///
/// **已知限制**：本封装不支持 > 65535 的分辨率，超限须由调用方降分辨率或分片。
///
/// # Errors
/// `width`/`height` 为 0 或 > 65535 → [`crate::MuxError`]（`MUX1003`）。
pub fn stsd_mjpeg(width: u32, height: u32) -> crate::MuxResult<Vec<u8>> {
    check_dim_u16("boxes::stsd_mjpeg", width, height)?;
    let mut entry: Vec<u8> = Vec::new();
    entry.extend_from_slice(&[0_u8; 6]); // reserved
    entry.extend_from_slice(&1_u16.to_be_bytes()); // data_reference_index
    entry.extend_from_slice(&0_u16.to_be_bytes()); // pre_defined
    entry.extend_from_slice(&0_u16.to_be_bytes()); // reserved
    entry.extend_from_slice(&[0_u8; 12]); // pre_defined[3]
    entry.extend_from_slice(&(width as u16).to_be_bytes());
    entry.extend_from_slice(&(height as u16).to_be_bytes());
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // horizresolution 72dpi
    entry.extend_from_slice(&0x0048_0000_u32.to_be_bytes()); // vertresolution
    entry.extend_from_slice(&0_u32.to_be_bytes()); // reserved
    entry.extend_from_slice(&1_u16.to_be_bytes()); // frame_count
    entry.extend_from_slice(&[0_u8; 32]); // compressorname
    entry.extend_from_slice(&24_u16.to_be_bytes()); // depth
    entry.extend_from_slice(&0xFFFF_u16.to_be_bytes()); // pre_defined = -1
    // VisualSampleEntry 之后为可选的 config box；mjpg 无需。
    // 实际样本条目需带 8 字节盒头（size+type）。
    let mut entry_box: Vec<u8> = Vec::new();
    write_box(&mut entry_box, b"mjpg", &entry);
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1_u32.to_be_bytes());
    payload.extend_from_slice(&entry_box);
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stsd", 0, 0, &payload);
    Ok(out)
}

/// 校验 width/height 可无损写入 u16 字段（`stsd` / JPEG `SOF` 共用）。
///
/// 单一守卫 = 单一真源：任何"把 u32 尺寸塞进 u16 字段"的路径都必须先过这里，
/// 杜绝 `as u16` 静默截断。合法区间 **1..=65535**（0 视为非法参数）。
pub(crate) fn check_dim_u16(location: &str, width: u32, height: u32) -> crate::MuxResult<()> {
    /// u16 字段上限：同时是 MP4 16.16 定点整数部分与 JPEG SOF0 的格式上限。
    const MAX: u32 = u16::MAX as u32;
    if width == 0 || height == 0 {
        return Err(crate::MuxError::new(
            "MUX1003",
            location,
            format!("宽或高为 0（{width}×{height}）"),
            "提供非零图像尺寸",
        ));
    }
    if width > MAX || height > MAX {
        return Err(crate::MuxError::new(
            "MUX1003",
            location,
            format!("分辨率 {width}×{height} 超出 u16 字段上限 {MAX}"),
            format!("本封装不支持 >{MAX} 分辨率（ISO 14496-12 §8.3.3.1 / JPEG SOF0 格式上限），请降分辨率或分片"),
        ));
    }
    Ok(())
}

/// 写音频 `stsd`（PCM，LPCM 未压缩，16 bit）。
pub fn stsd_pcm(channels: u16, sample_rate: u32, sample_size: u16) -> Vec<u8> {
    let mut entry: Vec<u8> = Vec::new();
    entry.extend_from_slice(&[0_u8; 6]); // reserved
    entry.extend_from_slice(&1_u16.to_be_bytes()); // data_reference_index
    entry.extend_from_slice(&0_u32.to_be_bytes()); // reserved[1] (version/revision/vendor)
    entry.extend_from_slice(&0_u32.to_be_bytes());
    entry.extend_from_slice(&channels.to_be_bytes());
    entry.extend_from_slice(&sample_size.to_be_bytes());
    entry.extend_from_slice(&0_u16.to_be_bytes()); // pre_defined
    entry.extend_from_slice(&0_u16.to_be_bytes()); // reserved
    entry.extend_from_slice(&(sample_rate << 16).to_be_bytes()); // 16.16
    let mut entry_box: Vec<u8> = Vec::new();
    write_box(&mut entry_box, b"lpcm", &entry);
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&1_u32.to_be_bytes());
    payload.extend_from_slice(&entry_box);
    let mut out: Vec<u8> = Vec::new();
    write_full_box(&mut out, b"stsd", 0, 0, &payload);
    out
}

/// 读回：校验字节流以盒结构递归遍历是否合法（供测试用）。
///
/// 返回顶层盒类型列表；结构非法（尺寸越界）返回 `None`。
pub fn inspect_top_level(data: &[u8]) -> Option<Vec<([u8; 4], usize)>> {
    let mut out: Vec<([u8; 4], usize)> = Vec::new();
    let mut pos = 0_usize;
    while pos + 8 <= data.len() {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        let ty: [u8; 4] = data[pos + 4..pos + 8].try_into().ok()?;
        let (box_size, header) = if size == 1 {
            if pos + 16 > data.len() {
                return None;
            }
            let large = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().ok()?) as usize;
            (large, 16)
        } else if size == 0 {
            (data.len() - pos, 8)
        } else {
            (size, 8)
        };
        if box_size < header || pos + box_size > data.len() {
            return None;
        }
        out.push((ty, pos));
        pos += box_size;
    }
    if pos == data.len() {
        Some(out)
    } else {
        None
    }
}

/// 在 `data` 中查找第一个 `box_type` 顶层盒的 (payload 起始, payload 长)。
pub fn find_box(data: &[u8], box_type: &[u8; 4]) -> Option<(usize, usize)> {
    let tops = inspect_top_level(data)?;
    for (ty, pos) in tops {
        if &ty == box_type {
            let size32 = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            if size32 == 1 {
                // 大盒：payload 从第 16 字节开始。
                let large = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().ok()?) as usize;
                return Some((pos + 16, large - 16));
            }
            if size32 == 0 {
                // 延伸到文件末尾。
                return Some((pos + 8, data.len() - (pos + 8)));
            }
            return Some((pos + 8, size32 - 8));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_roundtrip_inspect() {
        let mut out: Vec<u8> = Vec::new();
        write_box(&mut out, b"free", b"hello");
        let ftyp = write_ftyp(b"isom", &[b"isom", b"mp42"]);
        out.extend_from_slice(&ftyp);
        let tops = inspect_top_level(&out).expect("合法盒结构");
        assert_eq!(tops[0].0, *b"free");
        assert_eq!(tops[1].0, *b"ftyp");
    }

    #[test]
    fn mdat_large_box_parses() {
        let data = vec![0xAB_u8; 1000];
        let mdat = write_mdat(&data);
        let tops = inspect_top_level(&mdat).unwrap();
        assert_eq!(tops[0].0, *b"mdat");
        assert_eq!(u32::from_be_bytes(mdat[0..4].try_into().unwrap()), 1, "大盒标志");
        let large = u64::from_be_bytes(mdat[8..16].try_into().unwrap());
        assert_eq!(large as usize, mdat.len());
    }

    #[test]
    fn truncated_box_rejected() {
        let mut out: Vec<u8> = Vec::new();
        write_box(&mut out, b"free", b"12345");
        out.truncate(out.len() - 2);
        assert!(inspect_top_level(&out).is_none());
    }

    #[test]
    fn find_box_locates_payload() {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&write_ftyp(b"isom", &[]));
        out.extend_from_slice(&write_mdat(&[1, 2, 3]));
        let (start, len) = find_box(&out, b"mdat").unwrap();
        assert_eq!(len, 3);
        assert_eq!(&out[start..start + 3], &[1, 2, 3]);
    }
}
