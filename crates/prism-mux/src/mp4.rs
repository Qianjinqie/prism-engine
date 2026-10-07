//! MP4 容器写入（C4；§3.5：MP4(ISO-BMFF) + MJPEG + PCM）。
//!
//! 手写 ISO-BMFF：`ftyp` + `mdat`（全部样本）+ `moov`（`mvhd` + 视频 `trak` +
//! 可选音频 `trak`）。视频轨为 MJPEG（逐帧同步样本），音频轨为 PCM（lpcm）。
//!
//! # 布局策略
//!
//! 采用 **moov 后置**（faststart 的反面，写入简单）：`ftyp → mdat → moov`，
//! 因此 `stco` chunk offset 可在写完 `mdat` 之后据实回填——无需两遍扫描。
//! `ftyp` 尺寸固定，`mdat` 为大盒（64 位），偏移解析确定。
//!
//! # 验收
//!
//! 结构合法性：自写 [`crate::boxes::inspect_top_level`] 递归回读 + 盒字段断言。

use crate::boxes;
use crate::{jpeg, MuxError, MuxResult, JpegConfig};

/// 视频轨规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackSpec {
    /// 画幅宽。
    pub width: u32,
    /// 画幅高。
    pub height: u32,
    /// 帧率（帧/秒）。
    pub fps: u32,
}

/// 写入 MP4（MJPEG 视频 + 可选 PCM 音频）。`gray_frames` 为逐帧灰度样本。
pub fn write_mp4(
    gray_frames: &[Vec<u8>],
    video: TrackSpec,
    audio_pcm16: Option<&[u8]>,
    audio_sample_rate: u32,
    audio_channels: u16,
) -> MuxResult<Vec<u8>> {
    write_mp4_impl(gray_frames, video, audio_pcm16, audio_sample_rate, audio_channels, JpegConfig::default())
}

/// 便捷函数：从 RGBA8 帧写 MP4。
pub fn write_mp4_jpeg(
    rgba_frames: &[Vec<u8>],
    video: TrackSpec,
    audio_pcm16: Option<&[u8]>,
    audio_sample_rate: u32,
    audio_channels: u16,
    jpeg_cfg: JpegConfig,
) -> MuxResult<Vec<u8>> {
    let mut gray: Vec<Vec<u8>> = Vec::with_capacity(rgba_frames.len());
    for f in rgba_frames {
        gray.push(jpeg::rgba_to_gray(f, video.width, video.height)?);
    }
    write_mp4_impl(&gray, video, audio_pcm16, audio_sample_rate, audio_channels, jpeg_cfg)
}

fn write_mp4_impl(
    gray_frames: &[Vec<u8>],
    video: TrackSpec,
    audio_pcm16: Option<&[u8]>,
    audio_sample_rate: u32,
    audio_channels: u16,
    jpeg_cfg: JpegConfig,
) -> MuxResult<Vec<u8>> {
    if video.width == 0 || video.height == 0 || video.fps == 0 {
        return Err(MuxError::new(
            "MUX2001",
            "mp4::write_mp4",
            format!("非法视频参数 {}×{} @{}fps", video.width, video.height, video.fps),
            "宽/高/帧率必须 > 0",
        ));
    }
    // 分辨率上限守卫：stsd/tkhd 的宽高字段均为 u16 语义（ISO 14496-12 §8.3.3.1），
    // 超限必须在入口就报出，而不是让下游 `as u16` 静默截断成错误画幅。
    boxes::check_dim_u16("mp4::write_mp4", video.width, video.height)?;
    if gray_frames.is_empty() {
        return Err(MuxError::new(
            "MUX2001",
            "mp4::write_mp4",
            "帧序列为空",
            "至少提供 1 帧",
        ));
    }
    if let Some(pcm) = audio_pcm16 {
        if pcm.len() % 2 != 0 {
            return Err(MuxError::new(
                "MUX2002",
                "mp4::write_mp4",
                format!("PCM 字节数 {} 为奇数", pcm.len()),
                "16 bit PCM 字节数必须为偶数",
            ));
        }
        if audio_channels == 0 || audio_sample_rate == 0 {
            return Err(MuxError::new(
                "MUX2002",
                "mp4::write_mp4",
                "音频声道或采样率为 0",
                "声道数与采样率必须 > 0",
            ));
        }
    }

    // 编码全部视频帧。
    let n = (video.width as usize) * (video.height as usize);
    let mut jpegs: Vec<Vec<u8>> = Vec::with_capacity(gray_frames.len());
    for (i, f) in gray_frames.iter().enumerate() {
        if f.len() != n {
            return Err(MuxError::new(
                "MUX2001",
                "mp4::write_mp4",
                format!("第 {i} 帧样本 {} 与 {}×{}={n} 不符", f.len(), video.width, video.height),
                "每帧样本数必须等于 width×height",
            ));
        }
        jpegs.push(jpeg::encode_jpeg_gray(f, video.width, video.height, jpeg_cfg)?);
    }

    // 时间基：视频 timescale = fps，每样本 delta = 1 → duration = 帧数。
    let video_timescale: u32 = video.fps;
    let video_duration: u32 = jpegs.len() as u32;

    // 组装 mdat 数据（视频样本 + 音频样本）。
    let mut mdat_data: Vec<u8> = Vec::new();
    let mut video_sizes: Vec<u32> = Vec::with_capacity(jpegs.len());
    for j in &jpegs {
        video_sizes.push(j.len() as u32);
        mdat_data.extend_from_slice(j);
    }
    let audio_byte_len = audio_pcm16.map(|p| p.len()).unwrap_or(0);
    if let Some(pcm) = audio_pcm16 {
        mdat_data.extend_from_slice(pcm);
    }

    // ftyp（固定内容，尺寸可预先算出）。
    let ftyp = boxes::write_ftyp(b"isom", &[b"isom", b"mp41"]);

    // mdat 盒。
    let mdat = boxes::write_mdat(&mdat_data);

    // 计算 chunk offsets（mdat 内每个样本的绝对文件偏移）。
    // 文件布局：ftyp | mdat盒头(16字节) | 样本0 | 样本1 | ...
    let mdat_header = 16_usize; // 大盒头
    let base = ftyp.len() + mdat_header;
    let mut video_offsets: Vec<u32> = Vec::with_capacity(jpegs.len());
    let mut cur = base;
    for s in &video_sizes {
        video_offsets.push(cur as u32);
        cur += *s as usize;
    }
    // 音频：单 chunk 承载全部音频样本（PCM 帧内交错，逐"帧"样本）。
    let audio_frames: u32 = if audio_byte_len > 0 {
        (audio_byte_len / 2) as u32 / audio_channels.max(1) as u32
    } else {
        0
    };
    let audio_offset = cur as u32;

    // 构建 moov。
    let mut tracks: Vec<u8> = Vec::new();
    let next_track = if audio_byte_len > 0 { 3 } else { 2 };
    let movie_duration = video_duration.max(audio_frames);

    // 视频 trak。
    tracks.extend_from_slice(&build_video_trak(
        &video,
        video_timescale,
        video_duration,
        &video_sizes,
        &video_offsets,
    )?);
    // 音频 trak。
    if audio_byte_len > 0 {
        tracks.extend_from_slice(&build_audio_trak(
            audio_channels,
            audio_sample_rate,
            audio_frames,
            audio_byte_len as u32,
            audio_offset,
        ));
    }

    let mut moov_children: Vec<u8> = Vec::new();
    moov_children.extend_from_slice(&boxes::mvhd(video_timescale, movie_duration, next_track));
    moov_children.extend_from_slice(&tracks);
    let mut moov: Vec<u8> = Vec::new();
    boxes::write_container(&mut moov, b"moov", &moov_children);

    // 组装。
    let mut out: Vec<u8> = Vec::with_capacity(ftyp.len() + mdat.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&mdat);
    out.extend_from_slice(&moov);
    Ok(out)
}

/// 构建视频 trak（MJPEG，逐帧同步样本）。
fn build_video_trak(
    video: &TrackSpec,
    timescale: u32,
    duration: u32,
    sizes: &[u32],
    offsets: &[u32],
) -> MuxResult<Vec<u8>> {
    let count = sizes.len() as u32;
    // stbl。
    let mut stbl: Vec<u8> = Vec::new();
    stbl.extend_from_slice(&boxes::stsd_mjpeg(video.width, video.height)?);
    stbl.extend_from_slice(&boxes::stts_uniform(count, 1)); // 每样本 1 tick
    stbl.extend_from_slice(&boxes::stsc_one_per_chunk(count));
    stbl.extend_from_slice(&boxes::stsz(sizes));
    stbl.extend_from_slice(&boxes::stco(offsets));
    stbl.extend_from_slice(&boxes::stss_all(count)); // MJPEG 全同步
    let mut stbl_box: Vec<u8> = Vec::new();
    boxes::write_container(&mut stbl_box, b"stbl", &stbl);

    // minf。
    let mut minf_children: Vec<u8> = Vec::new();
    minf_children.extend_from_slice(&boxes::vmhd());
    minf_children.extend_from_slice(&boxes::dinf_self());
    minf_children.extend_from_slice(&stbl_box);
    let mut minf: Vec<u8> = Vec::new();
    boxes::write_container(&mut minf, b"minf", &minf_children);

    // mdia。
    let mut mdia_children: Vec<u8> = Vec::new();
    // mdhd 语言 "und" = 0x55C4。
    mdia_children.extend_from_slice(&boxes::mdhd(timescale, duration, 0x55C4));
    mdia_children.extend_from_slice(&boxes::hdlr(b"vide", "Prism Video Handler"));
    mdia_children.extend_from_slice(&minf);
    let mut mdia: Vec<u8> = Vec::new();
    boxes::write_container(&mut mdia, b"mdia", &mdia_children);

    // trak。
    let mut trak_children: Vec<u8> = Vec::new();
    trak_children.extend_from_slice(&boxes::tkhd(1, duration, video.width, video.height, 0));
    trak_children.extend_from_slice(&mdia);
    let mut trak: Vec<u8> = Vec::new();
    boxes::write_container(&mut trak, b"trak", &trak_children);
    Ok(trak)
}

/// 构建音频 trak（PCM 16 bit，单 chunk）。
fn build_audio_trak(
    channels: u16,
    sample_rate: u32,
    frames: u32,
    byte_len: u32,
    offset: u32,
) -> Vec<u8> {
    let mut stbl: Vec<u8> = Vec::new();
    stbl.extend_from_slice(&boxes::stsd_pcm(channels, sample_rate, 16));
    stbl.extend_from_slice(&boxes::stts_uniform(1, frames)); // 单"样本"覆盖全部帧
    stbl.extend_from_slice(&boxes::stsc_one_per_chunk(1));
    stbl.extend_from_slice(&boxes::stsz(&[byte_len]));
    stbl.extend_from_slice(&boxes::stco(&[offset]));
    let mut stbl_box: Vec<u8> = Vec::new();
    boxes::write_container(&mut stbl_box, b"stbl", &stbl);

    let mut minf_children: Vec<u8> = Vec::new();
    minf_children.extend_from_slice(&boxes::smhd());
    minf_children.extend_from_slice(&boxes::dinf_self());
    minf_children.extend_from_slice(&stbl_box);
    let mut minf: Vec<u8> = Vec::new();
    boxes::write_container(&mut minf, b"minf", &minf_children);

    let mut mdia_children: Vec<u8> = Vec::new();
    mdia_children.extend_from_slice(&boxes::mdhd(sample_rate, frames, 0x55C4));
    mdia_children.extend_from_slice(&boxes::hdlr(b"soun", "Prism Audio Handler"));
    mdia_children.extend_from_slice(&minf);
    let mut mdia: Vec<u8> = Vec::new();
    boxes::write_container(&mut mdia, b"mdia", &mdia_children);

    let mut trak_children: Vec<u8> = Vec::new();
    trak_children.extend_from_slice(&boxes::tkhd(2, frames, 0, 0, 0x0100));
    trak_children.extend_from_slice(&mdia);
    let mut trak: Vec<u8> = Vec::new();
    boxes::write_container(&mut trak, b"trak", &trak_children);
    trak
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_valid_mp4_structure() {
        let gray = vec![128_u8; 32 * 32];
        let mp4 = write_mp4(&[gray.clone(), gray], TrackSpec { width: 32, height: 32, fps: 30 }, None, 0, 0)
            .unwrap();
        // 顶层盒：ftyp、mdat、moov。
        let tops = boxes::inspect_top_level(&mp4).expect("合法盒结构");
        let types: Vec<[u8; 4]> = tops.iter().map(|(t, _)| *t).collect();
        assert_eq!(types, vec![*b"ftyp", *b"mdat", *b"moov"]);
    }

    #[test]
    fn moov_contains_mvhd_and_trak() {
        let gray = vec![100_u8; 16 * 16];
        let mp4 = write_mp4(&[gray], TrackSpec { width: 16, height: 16, fps: 25 }, None, 0, 0).unwrap();
        let (mstart, mlen) = boxes::find_box(&mp4, b"moov").unwrap();
        let moov = &mp4[mstart..mstart + mlen];
        assert!(contains(moov, b"mvhd"));
        assert!(contains(moov, b"trak"));
        assert!(contains(moov, b"vide"));
        assert!(contains(moov, b"mjpg"));
        assert!(contains(moov, b"stbl"));
        assert!(contains(moov, b"stco"));
    }

    #[test]
    fn with_audio_has_two_traks() {
        let gray = vec![80_u8; 8 * 8];
        let pcm = vec![0_u8; 128]; // 64 samples 16bit
        let mp4 = write_mp4(&[gray], TrackSpec { width: 8, height: 8, fps: 30 }, Some(&pcm), 8000, 1)
            .unwrap();
        let (mstart, mlen) = boxes::find_box(&mp4, b"moov").unwrap();
        let moov = &mp4[mstart..mstart + mlen];
        // 两个 trak。
        let cnt = count_occurrences(moov, b"trak");
        assert_eq!(cnt, 2);
        assert!(contains(moov, b"soun"));
        assert!(contains(moov, b"lpcm"));
    }

    #[test]
    fn chunk_offsets_point_into_mdat() {
        let gray = vec![64_u8; 8 * 8];
        let mp4 = write_mp4(&[gray.clone(), gray], TrackSpec { width: 8, height: 8, fps: 10 }, None, 0, 0)
            .unwrap();
        // 解析 stco 的第一个偏移，应落在 mdat 数据区内。
        let (mdstart, mdlen) = boxes::find_box(&mp4, b"mdat").unwrap();
        let (mstart, mlen) = boxes::find_box(&mp4, b"moov").unwrap();
        let moov = &mp4[mstart..mstart + mlen];
        let p = find(moov, b"stco").unwrap();
        // stco: 4(size)+4(type)+1(ver)+3(flags)+4(count) 后为第一偏移。
        let off = u32::from_be_bytes(moov[p + 16..p + 20].try_into().unwrap()) as usize;
        assert!(off >= mdstart && off < mdstart + mdlen, "偏移 {off} 应在 mdat 数据 [{mdstart},{})", mdstart + mdlen);
    }

    #[test]
    fn rejects_empty_and_bad_params() {
        assert_eq!(
            write_mp4(&[], TrackSpec { width: 8, height: 8, fps: 30 }, None, 0, 0).unwrap_err().code,
            "MUX2001"
        );
        assert_eq!(
            write_mp4(&[vec![0; 64]], TrackSpec { width: 0, height: 8, fps: 30 }, None, 0, 0).unwrap_err().code,
            "MUX2001"
        );
        // 奇数 PCM。
        assert_eq!(
            write_mp4(&[vec![0; 64]], TrackSpec { width: 8, height: 8, fps: 30 }, Some(&[0, 1, 2]), 8000, 1)
                .unwrap_err()
                .code,
            "MUX2002"
        );
    }

    #[test]
    fn inspection_is_self_consistent() {
        // 整个文件以盒结构可完整遍历（无尾部残留）。
        let gray: Vec<u8> = (0..20 * 20).map(|i| (i % 256) as u8).collect();
        let mp4 = write_mp4(&[gray], TrackSpec { width: 20, height: 20, fps: 12 }, None, 0, 0).unwrap();
        assert!(boxes::inspect_top_level(&mp4).is_some());
    }

    fn contains(h: &[u8], n: &[u8]) -> bool {
        find(h, n).is_some()
    }
    fn find(h: &[u8], n: &[u8]) -> Option<usize> {
        h.windows(n.len()).position(|w| w == n)
    }
    fn count_occurrences(h: &[u8], n: &[u8]) -> usize {
        h.windows(n.len()).filter(|w| *w == n).count()
    }
}
