//! AVI 容器写入（C4； 容器路线：AVI(rawvideo) "双击可播"底线）。
//!
//! # 结构（RIFF 容器）
//!
//! `RIFF("AVI ")`
//! - `LIST("hdrl")` — `avih`（主头）+ `LIST("strl")`（`strh` + `strf`）；
//! - `LIST("movi")` — 每帧一个 `00dc`（MJPEG 数据块）；
//! - `idx1` — 每帧索引项（offset/size/flags）。
//!
//! 带音频的变体（[`write_avi_mjpeg_audio`]）在同一 `movi` 内为每视频帧
//! 交错一个 `01wb`（PCM16）音频 chunk，`hdrl` 含第二条 `strl`
//! （`auds` / `01wb`），`idx1` 为两类 chunk 都建索引。
//!
//! 编码：

use crate::jpeg::{self, encode_jpeg_color, JpegConfig};
use crate::{MuxError, MuxResult};
use prism_audio::PcmBuffer;

/// AVI 写入配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AviConfig {
    /// 帧率（帧/秒）。
    pub fps: u32,
    /// 画幅宽（像素）。
    pub width: u32,
    /// 画幅高（像素）。
    pub height: u32,
    /// JPEG 编码配置。
    pub jpeg: JpegConfig,
}

impl AviConfig {
    /// 构造配置（默认质量为 85 的灰度 JPEG）。
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        AviConfig { fps, width, height, jpeg: JpegConfig::default() }
    }
}

/// RIFF 块头辅助：写 FourCC + 小端尺寸。
fn chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0); // RIFF 词对齐
    }
}

/// LIST 块辅助：FourCC + "LIST" + 小端尺寸 + 类型 + 子块。
fn list(out: &mut Vec<u8>, list_type: &[u8; 4], children: &[u8]) {
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&((4 + children.len()) as u32).to_le_bytes());
    out.extend_from_slice(list_type);
    out.extend_from_slice(children);
    if (4 + children.len()) % 2 == 1 {
        out.push(0);
    }
}

/// 写入 AVI（MJPEG 视频轨）。`frames` 为逐帧灰度样本（`width×height` 每帧）。
pub fn write_avi(gray_frames: &[Vec<u8>], cfg: AviConfig) -> MuxResult<Vec<u8>> {
    if cfg.width == 0 || cfg.height == 0 || cfg.fps == 0 {
        return Err(MuxError::new(
            "MUX1001",
            "avi::write_avi",
            format!("非法参数 {}×{} @{}fps", cfg.width, cfg.height, cfg.fps),
            "宽/高/帧率必须 > 0",
        ));
    }
    // 逐帧 JPEG 编码（确定性）。
    let n = (cfg.width as usize) * (cfg.height as usize);
    let mut jpegs: Vec<Vec<u8>> = Vec::with_capacity(gray_frames.len());
    for (i, f) in gray_frames.iter().enumerate() {
        if f.len() != n {
            return Err(MuxError::new(
                "MUX1001",
                "avi::write_avi",
                format!("第 {i} 帧样本 {} 与 {}×{}={n} 不符", f.len(), cfg.width, cfg.height),
                "每帧样本数必须等于 width×height",
            ));
        }
        jpegs.push(jpeg::encode_jpeg_gray(f, cfg.width, cfg.height, cfg.jpeg)?);
    }

    // hdrl。
    let mut hdrl_children: Vec<u8> = Vec::new();
    write_avih(&mut hdrl_children, &cfg, jpegs.len(), jpegs.iter().map(|j| j.len()).sum());
    let strl = build_strl(&cfg);
    list(&mut hdrl_children, b"strl", &strl);

    let mut hdrl: Vec<u8> = Vec::new();
    list(&mut hdrl, b"hdrl", &hdrl_children);

    // movi（记录每帧相对 movi 起始的偏移）。
    let mut movi_children: Vec<u8> = Vec::new();
    let mut frame_offsets: Vec<(usize, usize)> = Vec::with_capacity(jpegs.len());
    for j in &jpegs {
        let off = movi_children.len() + 8; // 跳过 00dc 头
        chunk(&mut movi_children, b"00dc", j);
        frame_offsets.push((off + 4, j.len())); // +4：相对于 "movi" 类型字段后
    }
    let mut movi: Vec<u8> = Vec::new();
    list(&mut movi, b"movi", &movi_children);

    // idx1。
    let mut idx1: Vec<u8> = Vec::new();
    for (off, size) in &frame_offsets {
        idx1.extend_from_slice(b"00dc");
        idx1.extend_from_slice(&0x0000_0010_u32.to_le_bytes()); // AVIIF_KEYFRAME
        idx1.extend_from_slice(&(*off as u32).to_le_bytes());
        idx1.extend_from_slice(&(*size as u32).to_le_bytes());
    }
    let mut idx1_block: Vec<u8> = Vec::new();
    idx1_block.extend_from_slice(b"idx1");
    idx1_block.extend_from_slice(&(idx1.len() as u32).to_le_bytes());
    idx1_block.extend_from_slice(&idx1);

    // 组装 RIFF。
    let mut riff_body: Vec<u8> = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    riff_body.extend_from_slice(&idx1_block);

    let mut out: Vec<u8> = Vec::with_capacity(riff_body.len() + 8);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&riff_body);
    Ok(out)
}

/// 便捷函数：从 RGBA8 帧写 AVI。
pub fn write_avi_jpeg(rgba_frames: &[Vec<u8>], cfg: AviConfig) -> MuxResult<Vec<u8>> {
    let mut gray_frames: Vec<Vec<u8>> = Vec::with_capacity(rgba_frames.len());
    for f in rgba_frames {
        gray_frames.push(jpeg::rgba_to_gray(f, cfg.width, cfg.height)?);
    }
    write_avi(&gray_frames, cfg)
}

// ===========================================================================
// MJPEG-in-AVI（标准 RIFF 容器，流类型 MJPEG）——"双击可播"底线（彩色版）
// ===========================================================================
//
// 与上面的灰度 `write_avi` 不同，本实现：
//   1. 复用已验证的彩色 JPEG 编码器 `encode_jpeg_color`；
//   2. 严格按 OpenDML / AVI 1.0 规范手写容器（流类型 `MJPG`）；
//   3. 所有多字节整数小端写入、chunk 按 RIFF 词对齐（2 字节）；
//   4. 零第三方依赖。
//
// 结构（顶层）：
//   RIFF("AVI ")
//   ├─ LIST("hdrl") ─ avih + LIST("strl") ─ strh + strf
//   ├─ LIST("movi") ─ 每帧一个 "00dc" chunk
//   └─ idx1        ─ 每帧 16 字节索引项

/// 把逐帧 RGBA8 序列封装为标准 **MJPEG-in-AVI**（RIFF 容器，流类型 `MJPG`）。
///
/// - 每帧先经 [`encode_jpeg_color`] 编码为彩色基线 JPEG；
/// - 手写标准 AVI 结构（零第三方 crate，全 Rust）；
/// - 所有多字节整数小端写入，chunk 按 2 字节对齐；
/// - 宽高须为偶数（MJPEG 8×8 块对齐），否则返回错误。
///
/// 返回完整 AVI 字节流，可被 VLC / 浏览器 / QuickTime 直接播放。
pub fn write_avi_mjpeg(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    // 参数校验。
    if width == 0 || height == 0 || fps == 0 {
        return Err(format!(
            "avi::write_avi_mjpeg: 非法参数 {}×{} @{}fps（宽/高/帧率必须 > 0）",
            width, height, fps
        ));
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!(
            "avi::write_avi_mjpeg: MJPEG 要求偶数尺寸，收到 {}×{}",
            width, height
        ));
    }
    if frames.is_empty() {
        return Err("avi::write_avi_mjpeg: 至少需要 1 帧".to_string());
    }
    let frame_pixels = (width as usize) * (height as usize) * 4;

    // 1) 逐帧编码为彩色 JPEG（复用已验证编码器）。
    let mut jpegs: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        if f.len() != frame_pixels {
            return Err(format!(
                "avi::write_avi_mjpeg: 第 {i} 帧 RGBA 字节数 {} 与 {}×{}×4={} 不符",
                f.len(),
                width,
                height,
                frame_pixels
            ));
        }
        let j = encode_jpeg_color(
            f,
            width,
            height,
            JpegConfig {
                quality: quality.clamp(1, 100),
                grayscale: false,
            },
        )
        .map_err(|e| format!("avi::write_avi_mjpeg: 第 {i} 帧 JPEG 编码失败：{e}"))?;
        jpegs.push(j);
    }

    let n = jpegs.len() as u32;
    let total_jpeg: u64 = jpegs.iter().map(|j| j.len() as u64).sum();
    let max_jpeg = jpegs.iter().map(|j| j.len()).max().unwrap_or(0) as u32;
    // 每秒数据率（≈ 总字节 / 时长），写进 avih.dwMaxBytesPerSec 供播放器估算带宽。
    let microsec_per_frame = 1_000_000u32 / fps; // fps >= 1，安全
    let bytes_per_sec = if n > 0 {
        ((total_jpeg * fps as u64) / n as u64) as u32
    } else {
        0
    };

    // 2) hdrl：avih + LIST("strl") [strh + strf]。
    let mut hdrl_children: Vec<u8> = Vec::new();
    write_avih_mjpeg(&mut hdrl_children, n, width, height, microsec_per_frame, bytes_per_sec, 1);
    let strl = build_strl_mjpeg(n, width, height, fps, max_jpeg);
    list(&mut hdrl_children, b"strl", &strl);
    let mut hdrl: Vec<u8> = Vec::new();
    list(&mut hdrl, b"hdrl", &hdrl_children);

    // 3) movi：每帧一个 "00dc" chunk（chunk 含 size+数据，整体按 2 字节对齐）。
    //    索引项的 dwOffset 相对 "movi" 数据起始（即首字节 '00dc' 的位置）。
    let mut movi_children: Vec<u8> = Vec::new();
    let mut frame_offsets: Vec<(u32, u32)> = Vec::with_capacity(jpegs.len()); // (offset, size)
    for j in &jpegs {
        let offset = movi_children.len() as u32; // 指向本帧 '00dc' 四CC
        chunk(&mut movi_children, b"00dc", j); // 自动 RIFF 词对齐（奇数补 1 字节 0）
        frame_offsets.push((offset, j.len() as u32));
    }
    let mut movi: Vec<u8> = Vec::new();
    list(&mut movi, b"movi", &movi_children);

    // 4) idx1：每帧一项 16 字节（dwChunkId / dwFlags / dwOffset / dwSize）。
    let mut idx1: Vec<u8> = Vec::new();
    for (offset, size) in &frame_offsets {
        idx1.extend_from_slice(b"00dc"); // dwChunkId
        idx1.extend_from_slice(&0x0000_0010_u32.to_le_bytes()); // dwFlags = AVIIF_KEYFRAME
        idx1.extend_from_slice(&offset.to_le_bytes()); // dwOffset（相对 movi 数据起始）
        idx1.extend_from_slice(&size.to_le_bytes()); // dwSize（JPEG 长度，不含对齐 padding）
    }
    let mut idx1_block: Vec<u8> = Vec::new();
    chunk(&mut idx1_block, b"idx1", &idx1);

    // 5) 组装顶层 RIFF。
    let mut riff_body: Vec<u8> = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    riff_body.extend_from_slice(&idx1_block);
    // RIFF 顶层也需词对齐（当前各子块均为偶数，恒为偶；防御性保留）。
    if riff_body.len() % 2 == 1 {
        riff_body.push(0);
    }

    let mut out: Vec<u8> = Vec::with_capacity(riff_body.len() + 12);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&riff_body);
    Ok(out)
}

/// 写 `avih` 主头（56 字节），字段严格按 MJPEG-in-AVI 标准。
///
/// `streams` 为流条数（视频-only = 1；视频 + PCM 音频 = 2）。
#[allow(clippy::too_many_arguments)]
fn write_avih_mjpeg(
    out: &mut Vec<u8>,
    total_frames: u32,
    width: u32,
    height: u32,
    microsec_per_frame: u32,
    bytes_per_sec: u32,
    streams: u32,
) {
    let mut body: Vec<u8> = Vec::with_capacity(56);
    body.extend_from_slice(&microsec_per_frame.to_le_bytes()); // dwMicroSecPerFrame
    body.extend_from_slice(&bytes_per_sec.to_le_bytes()); // dwMaxBytesPerSec
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwPaddingGranularity
    body.extend_from_slice(&0x0000_0110_u32.to_le_bytes()); // dwFlags = HASINDEX(0x10)|ISINTERLEAVED(0x100)
    body.extend_from_slice(&total_frames.to_le_bytes()); // dwTotalFrames
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwInitialFrames
    body.extend_from_slice(&streams.to_le_bytes()); // dwStreams
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwSuggestedBufferSize
    body.extend_from_slice(&width.to_le_bytes()); // dwWidth
    body.extend_from_slice(&height.to_le_bytes()); // dwHeight
    body.extend_from_slice(&[0_u8; 16]); // dwReserved[4]
    chunk(out, b"avih", &body);
}

/// 构建 `strl`（`strh` + `strf`：MJPEG 视频流），字段严格按标准。
fn build_strl_mjpeg(total_frames: u32, width: u32, height: u32, fps: u32, max_jpeg: u32) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    // strh（AVISTREAMHEADER，56 字节）。
    let mut strh: Vec<u8> = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids"); // fccType
    strh.extend_from_slice(b"MJPG"); // fccHandler
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwFlags
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wPriority
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wLanguage
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwInitialFrames
    strh.extend_from_slice(&1_u32.to_le_bytes()); // dwScale
    strh.extend_from_slice(&fps.to_le_bytes()); // dwRate（帧率 = dwRate/dwScale = fps）
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwStart
    strh.extend_from_slice(&total_frames.to_le_bytes()); // dwLength（帧数）
    strh.extend_from_slice(&max_jpeg.to_le_bytes()); // dwSuggestedBufferSize
    strh.extend_from_slice(&0xFFFF_FFFF_u32.to_le_bytes()); // dwQuality = -1
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwSampleSize
    strh.extend_from_slice(&0_i16.to_le_bytes()); // rcFrame.left
    strh.extend_from_slice(&0_i16.to_le_bytes()); // top
    strh.extend_from_slice(&0_i16.to_le_bytes()); // right
    strh.extend_from_slice(&0_i16.to_le_bytes()); // bottom（rcFrame 全 0）
    chunk(&mut out, b"strh", &strh);

    // strf（BITMAPINFOHEADER，40 字节）。
    let mut strf: Vec<u8> = Vec::with_capacity(40);
    strf.extend_from_slice(&40_u32.to_le_bytes()); // biSize
    strf.extend_from_slice(&(width as i32).to_le_bytes()); // biWidth
    strf.extend_from_slice(&(height as i32).to_le_bytes()); // biHeight
    strf.extend_from_slice(&1_u16.to_le_bytes()); // biPlanes
    strf.extend_from_slice(&24_u16.to_le_bytes()); // biBitCount
    strf.extend_from_slice(b"MJPG"); // biCompression
    strf.extend_from_slice(&((width * height * 3) as u32).to_le_bytes()); // biSizeImage
    strf.extend_from_slice(&0_i32.to_le_bytes()); // biXPelsPerMeter
    strf.extend_from_slice(&0_i32.to_le_bytes()); // biYPelsPerMeter
    strf.extend_from_slice(&0_u32.to_le_bytes()); // biClrUsed
    strf.extend_from_slice(&0_u32.to_le_bytes()); // biClrImportant
    chunk(&mut out, b"strf", &strf);

    out
}

// ===========================================================================
// MJPEG-in-AVI + PCM 音频流（标准 AVI 1.0 双流，ffmpeg 可解）
// ===========================================================================
//
// 在 [`write_avi_mjpeg`] 的纯视频结构上增加第二条流：
//   RIFF("AVI ")
//   ├─ LIST("hdrl") ─ avih(dwStreams=2)
//   │                ├─ LIST("strl") ─ strh(vids/MJPG) + strf(BITMAPINFOHEADER)
//   │                └─ LIST("strl") ─ strh(auds/01wb) + strf(WAVEFORMATEX 16B)
//   ├─ LIST("movi") ─ 每视频帧后交错一个 "01wb" PCM16 chunk（00dc/01wb 交替）
//   └─ idx1        ─ 两类 chunk 都建索引（共 2×帧数 条）
//
// 音频口径与 `prism-audio::wav::encode_wav(B16)` 完全一致：
// `v = round(clamp(x,-1,1) · 32768)` 后再 clamp 到 i16 域，16-bit 有符号
// 小端——确定性、跨平台逐字节一致。

/// 把逐帧 RGBA8 序列 + 一条 PCM 音轨封装为标准 **MJPEG-in-AVI**
/// （视频流 `MJPG` + 音频流 `pcm_s16le`，双流交错）。
///
/// - 视频编码路径与 [`write_avi_mjpeg`] 完全相同（复用 `encode_jpeg_color`）；
/// - 音频按帧切分：`samples_per_frame = audio.sample_rate() / fps`（必须整除，
///   否则返回 `Err`）；每视频帧后跟一个 `01wb` chunk（偶数长度，RIFF 词对齐）；
/// - 音频总样本不足 `frames × samples_per_frame` 补静音、超出截断；
/// - `avih.dwLength` = 视频帧数；`avih.dwStreams` = 2；`idx1` 为 `00dc` /
///   `01wb` 两类 chunk 都建索引；
/// - 参数校验口径与 [`write_avi_mjpeg`] 一致（空帧 / 奇数尺寸 / 质量越界 /
///   帧尺寸不符 → `Err`，不 panic）。
pub fn write_avi_mjpeg_audio(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u32,
    quality: u8,
    audio: &PcmBuffer,
) -> Result<Vec<u8>, String> {
    // 参数校验（与 write_avi_mjpeg 同口径）。
    if width == 0 || height == 0 || fps == 0 {
        return Err(format!(
            "avi::write_avi_mjpeg_audio: 非法参数 {}×{} @{}fps（宽/高/帧率必须 > 0）",
            width, height, fps
        ));
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!(
            "avi::write_avi_mjpeg_audio: MJPEG 要求偶数尺寸，收到 {}×{}",
            width, height
        ));
    }
    if frames.is_empty() {
        return Err("avi::write_avi_mjpeg_audio: 至少需要 1 帧".to_string());
    }
    let frame_pixels = (width as usize) * (height as usize) * 4;

    // 1) 逐帧编码为彩色 JPEG（复用已验证编码器，与 write_avi_mjpeg 同路径）。
    let mut jpegs: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        if f.len() != frame_pixels {
            return Err(format!(
                "avi::write_avi_mjpeg_audio: 第 {i} 帧 RGBA 字节数 {} 与 {}×{}×4={} 不符",
                f.len(),
                width,
                height,
                frame_pixels
            ));
        }
        let j = encode_jpeg_color(
            f,
            width,
            height,
            JpegConfig { quality: quality.clamp(1, 100), grayscale: false },
        )
        .map_err(|e| format!("avi::write_avi_mjpeg_audio: 第 {i} 帧 JPEG 编码失败：{e}"))?;
        jpegs.push(j);
    }

    // 2) 组装：音频按帧切分 + hdrl + movi 交错 + idx1 + RIFF（与
    //    write_avi_mjpeg_audio_frames 共用同一组装路径）。
    assemble_avi_mjpeg_audio(&jpegs, width, height, fps, audio)
}

/// 把**预编码** JPEG 帧序列 + 一条 PCM 音轨封装为标准 **MJPEG-in-AVI**
/// （视频流 `MJPG` + 音频流 `pcm_s16le`，双流交错）。
///
/// 与 [`write_avi_mjpeg_audio`] 的字节输出**完全一致**（同帧同音频），
/// 差别仅在编码时机：本函数接收调用方已编码的 JPEG 帧（如逐帧流式
/// 编码后仅驻留压缩帧），从而避免全片裸 RGBA8 帧序列同时驻留——
/// 90s 960×540 全片裸帧 ≈5.6GB，超出常见容器内存上限，而 JPEG 帧
/// 序列仅 ≈0.3GB。音频按帧切分 / 补静音 / 截断语义、`idx1` 索引与
/// RIFF 结构与 [`write_avi_mjpeg_audio`] 逐字节相同。
///
/// 参数校验口径一致（偶数尺寸 / 至少 1 帧 / 采样率整除帧率；JPEG 帧
/// 内容合法性由调用方保证）。
pub fn write_avi_mjpeg_audio_frames(
    jpegs: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u32,
    audio: &PcmBuffer,
) -> Result<Vec<u8>, String> {
    assemble_avi_mjpeg_audio(jpegs, width, height, fps, audio)
}

/// 内部：把预编码 JPEG 帧序列 + PCM 音轨组装为 MJPEG-in-AVI（双流交错）。
///
/// 纯字节组装（确定性）：音频按帧切分（不足补静音、超出截断）→ hdrl
/// （avih + 双 strl）→ movi（每视频帧 `00dc` + 紧接 `01wb`）→ idx1 →
/// 顶层 RIFF。由 [`write_avi_mjpeg_audio`]（编码后）与
/// [`write_avi_mjpeg_audio_frames`]（调用方预编码）共用，保证两者输出
/// 位级一致。
fn assemble_avi_mjpeg_audio(
    jpegs: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u32,
    audio: &PcmBuffer,
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || fps == 0 {
        return Err(format!(
            "avi::write_avi_mjpeg_audio: 非法参数 {}×{} @{}fps（宽/高/帧率必须 > 0）",
            width, height, fps
        ));
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!(
            "avi::write_avi_mjpeg_audio: MJPEG 要求偶数尺寸，收到 {}×{}",
            width, height
        ));
    }
    if jpegs.is_empty() {
        return Err("avi::write_avi_mjpeg_audio: 至少需要 1 帧".to_string());
    }

    // 帧 / 样本对齐：每视频帧 samples_per_frame 个采样帧（采样率必须是帧率整数倍）。
    let sample_rate = audio.sample_rate();
    let channels = audio.channels() as usize;
    if sample_rate % fps != 0 {
        return Err(format!(
            "avi::write_avi_mjpeg_audio: 音频采样率 {sample_rate} 不是帧率 {fps} 的整数倍（无法按帧切分音频 chunk）",
        ));
    }
    // sample_rate ≥ 1 且 fps | sample_rate ⇒ samples_per_frame ≥ 1。
    let samples_per_frame = (sample_rate / fps) as usize;
    let block_align = channels * 2; // 16-bit PCM：每采样帧 bytes
    let chunk_bytes = samples_per_frame * block_align;

    // 音频 → 16-bit 有符号小端 PCM 字节流（不足补静音、超出截断）。
    // 口径与 prism-audio wav.rs B16 一致：round(clamp(x,-1,1)·32768) 后 clamp i16。
    let total_samples = jpegs.len() * samples_per_frame;
    let mut pcm: Vec<u8> = Vec::with_capacity(total_samples * block_align);
    let src = audio.as_slice();
    for i in 0..total_samples {
        for c in 0..channels {
            let x = src.get(i * channels + c).copied().unwrap_or(0.0);
            let v = (x.clamp(-1.0, 1.0) * 32_768.0).round() as i32;
            let v = v.clamp(-32_768, 32_767) as i16;
            pcm.extend_from_slice(&v.to_le_bytes());
        }
    }

    let n = jpegs.len() as u32;
    let total_jpeg: u64 = jpegs.iter().map(|j| j.len() as u64).sum();
    let max_jpeg = jpegs.iter().map(|j| j.len()).max().unwrap_or(0) as u32;
    let microsec_per_frame = 1_000_000u32 / fps; // fps >= 1，安全
    // 每秒数据率 ≈ 视频 + 音频（供播放器估算带宽，仅咨询性字段）。
    let bytes_per_sec = ((total_jpeg * fps as u64) / n as u64) as u32
        + sample_rate * block_align as u32;

    // 3) hdrl：avih(dwStreams=2) + 视频 strl + 音频 strl。
    let mut hdrl_children: Vec<u8> = Vec::new();
    write_avih_mjpeg(&mut hdrl_children, n, width, height, microsec_per_frame, bytes_per_sec, 2);
    let strl_video = build_strl_mjpeg(n, width, height, fps, max_jpeg);
    list(&mut hdrl_children, b"strl", &strl_video);
    let strl_audio = build_strl_audio(total_samples, sample_rate, audio.channels());
    list(&mut hdrl_children, b"strl", &strl_audio);
    let mut hdrl: Vec<u8> = Vec::new();
    list(&mut hdrl, b"hdrl", &hdrl_children);

    // 4) movi：每视频帧 "00dc" + 紧接一个 "01wb" 音频 chunk（交错）。
    //    索引项 dwOffset 相对 "movi" 数据起始（即首字节 '00dc' 的位置）。
    let mut movi_children: Vec<u8> = Vec::new();
    let mut idx_entries: Vec<([u8; 4], u32, u32, u32)> = // (fourcc, flags, offset, size)
        Vec::with_capacity(jpegs.len() * 2);
    for (i, j) in jpegs.iter().enumerate() {
        let v_off = movi_children.len() as u32; // 指向本帧 '00dc' 四CC
        chunk(&mut movi_children, b"00dc", j); // 自动 RIFF 词对齐（奇数补 1 字节 0）
        idx_entries.push((*b"00dc", 0x0000_0010, v_off, j.len() as u32));
        let a_off = movi_children.len() as u32; // 指向本帧 '01wb' 四CC
        let a = &pcm[i * chunk_bytes..(i + 1) * chunk_bytes];
        chunk(&mut movi_children, b"01wb", a); // chunk_bytes 恒偶，无需补位
        idx_entries.push((*b"01wb", 0, a_off, a.len() as u32));
    }
    let mut movi: Vec<u8> = Vec::new();
    list(&mut movi, b"movi", &movi_children);

    // 5) idx1：两类 chunk 都建索引（dwChunkId / dwFlags / dwOffset / dwSize）。
    let mut idx1: Vec<u8> = Vec::new();
    for (fourcc, flags, offset, size) in &idx_entries {
        idx1.extend_from_slice(fourcc); // dwChunkId
        idx1.extend_from_slice(&flags.to_le_bytes()); // dwFlags（视频=KEYFRAME，音频=0）
        idx1.extend_from_slice(&offset.to_le_bytes()); // dwOffset（相对 movi 数据起始）
        idx1.extend_from_slice(&size.to_le_bytes()); // dwSize（数据长，不含对齐 padding）
    }
    let mut idx1_block: Vec<u8> = Vec::new();
    chunk(&mut idx1_block, b"idx1", &idx1);

    // 6) 组装顶层 RIFF。
    let mut riff_body: Vec<u8> = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    riff_body.extend_from_slice(&idx1_block);
    if riff_body.len() % 2 == 1 {
        riff_body.push(0);
    }

    let mut out: Vec<u8> = Vec::with_capacity(riff_body.len() + 12);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&riff_body);
    Ok(out)
}

/// 构建音频流 `strl`（`strh` + `strf`：PCM 音频流），字段严格按 AVI 1.0。
///
/// - `strh`：fccType `auds`、fccHandler `01wb`（WAVE_FORMAT_PCM）；
///   `dwScale = 1`、`dwRate = sample_rate`（即 dwLength 以采样帧计）；
/// - `strf`：WAVEFORMATEX（18 字节 = 16 字节基础字段 + `cbSize=0`；
///   `wFormatTag=1`、16 bit）。
fn build_strl_audio(total_sample_frames: usize, sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    // strh（AVISTREAMHEADER，56 字节）。
    let mut strh: Vec<u8> = Vec::with_capacity(56);
    strh.extend_from_slice(b"auds"); // fccType
    strh.extend_from_slice(b"01wb"); // fccHandler = WAVE_FORMAT_PCM
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwFlags
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wPriority
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wLanguage
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwInitialFrames
    strh.extend_from_slice(&1_u32.to_le_bytes()); // dwScale（1 采样帧/单位）
    strh.extend_from_slice(&sample_rate.to_le_bytes()); // dwRate（采样帧/秒）
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwStart
    strh.extend_from_slice(&(total_sample_frames as u32).to_le_bytes()); // dwLength（采样帧总数）
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwSuggestedBufferSize
    strh.extend_from_slice(&0xFFFF_FFFF_u32.to_le_bytes()); // dwQuality = -1
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwSampleSize（0 = 变长，按 chunk 实际大小）
    strh.extend_from_slice(&0_i16.to_le_bytes()); // rcFrame.left
    strh.extend_from_slice(&0_i16.to_le_bytes()); // top
    strh.extend_from_slice(&0_i16.to_le_bytes()); // right
    strh.extend_from_slice(&0_i16.to_le_bytes()); // bottom（rcFrame 全 0）
    chunk(&mut out, b"strh", &strh);

    // strf（WAVEFORMATEX，16 字节）。
    let block_align = channels as usize * 2;
    let mut strf: Vec<u8> = Vec::with_capacity(16);
    strf.extend_from_slice(&1_u16.to_le_bytes()); // wFormatTag = WAVE_FORMAT_PCM
    strf.extend_from_slice(&channels.to_le_bytes()); // nChannels
    strf.extend_from_slice(&sample_rate.to_le_bytes()); // nSamplesPerSec
    strf.extend_from_slice(&((sample_rate as usize * block_align) as u32).to_le_bytes()); // nAvgBytesPerSec
    strf.extend_from_slice(&(block_align as u16).to_le_bytes()); // nBlockAlign
    strf.extend_from_slice(&16_u16.to_le_bytes()); // wBitsPerSample
    strf.extend_from_slice(&0_u16.to_le_bytes()); // cbSize
    chunk(&mut out, b"strf", &strf);

    out
}

/// 写 `avih`（主头，56 字节）。
fn write_avih(out: &mut Vec<u8>, cfg: &AviConfig, frames: usize, total_data: usize) {
    let mut body: Vec<u8> = Vec::with_capacity(56);
    let usec_per_frame = 1_000_000_u32 / cfg.fps.max(1);
    body.extend_from_slice(&usec_per_frame.to_le_bytes()); // dwMicroSecPerFrame
    body.extend_from_slice(&((total_data as u32).max(1)).to_le_bytes()); // dwMaxBytesPerSec（近似）
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwPaddingGranularity
    body.extend_from_slice(&0x10_u32.to_le_bytes()); // dwFlags = AVIF_HASINDEX
    body.extend_from_slice(&(frames as u32).to_le_bytes()); // dwTotalFrames
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwInitialFrames
    body.extend_from_slice(&1_u32.to_le_bytes()); // dwStreams
    body.extend_from_slice(&0_u32.to_le_bytes()); // dwSuggestedBufferSize
    body.extend_from_slice(&cfg.width.to_le_bytes()); // dwWidth
    body.extend_from_slice(&cfg.height.to_le_bytes()); // dwHeight
    body.extend_from_slice(&[0_u8; 16]); // dwReserved[4]
    chunk(out, b"avih", &body);
}

/// 构建 `strl`（`strh` + `strf`：MJPEG 视频流）。
fn build_strl(cfg: &AviConfig) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    // strh（56 字节）。
    let mut strh: Vec<u8> = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids"); // fccType
    strh.extend_from_slice(b"MJPG"); // fccHandler
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwFlags
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wPriority
    strh.extend_from_slice(&0_u16.to_le_bytes()); // wLanguage
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwInitialFrames
    strh.extend_from_slice(&1_000_u32.to_le_bytes()); // dwScale
    strh.extend_from_slice(&cfg.fps.to_le_bytes()); // dwRate
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwStart
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwLength（填充后回写，此处 0）
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwSuggestedBufferSize
    strh.extend_from_slice(&0xFFFF_FFFF_u32.to_le_bytes()); // dwQuality = -1
    strh.extend_from_slice(&0_u32.to_le_bytes()); // dwSampleSize
    strh.extend_from_slice(&0_i16.to_le_bytes()); // rcFrame.left
    strh.extend_from_slice(&0_i16.to_le_bytes()); // top
    strh.extend_from_slice(&(cfg.width as i16).to_le_bytes()); // right
    strh.extend_from_slice(&(cfg.height as i16).to_le_bytes()); // bottom
    chunk(&mut out, b"strh", &strh);

    // strf（BITMAPINFOHEADER，40 字节）。
    let mut strf: Vec<u8> = Vec::with_capacity(40);
    strf.extend_from_slice(&40_u32.to_le_bytes()); // biSize
    strf.extend_from_slice(&(cfg.width as i32).to_le_bytes()); // biWidth
    strf.extend_from_slice(&(cfg.height as i32).to_le_bytes()); // biHeight
    strf.extend_from_slice(&1_u16.to_le_bytes()); // biPlanes
    strf.extend_from_slice(&24_u16.to_le_bytes()); // biBitCount
    strf.extend_from_slice(b"MJPG"); // biCompression
    strf.extend_from_slice(&((cfg.width * cfg.height * 3) as u32).to_le_bytes()); // biSizeImage
    strf.extend_from_slice(&0_i32.to_le_bytes()); // biXPelsPerMeter
    strf.extend_from_slice(&0_i32.to_le_bytes()); // biYPelsPerMeter
    strf.extend_from_slice(&0_u32.to_le_bytes()); // biClrUsed
    strf.extend_from_slice(&0_u32.to_le_bytes()); // biClrImportant
    chunk(&mut out, b"strf", &strf);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_riff_avi_header() {
        let frame = vec![120_u8; 32 * 32];
        let avi = write_avi(&[frame.clone(), frame], AviConfig::new(32, 32, 30)).unwrap();
        assert_eq!(&avi[0..4], b"RIFF");
        assert_eq!(&avi[8..12], b"AVI ");
        // RIFF 尺寸 = 文件长 - 8。
        let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size, avi.len() - 8);
        // 含关键块。
        assert!(contains(&avi, b"hdrl"));
        assert!(contains(&avi, b"movi"));
        assert!(contains(&avi, b"idx1"));
        assert!(contains(&avi, b"MJPG"));
        assert!(contains(&avi, b"00dc"));
    }

    #[test]
    fn frame_count_in_avih() {
        let frame = vec![200_u8; 16 * 16];
        let frames = vec![frame.clone(); 5];
        let avi = write_avi(&frames, AviConfig::new(16, 16, 24)).unwrap();
        // 找 avih，dwTotalFrames 在 body 偏移 16。
        let p = find(&avi, b"avih").unwrap() + 8;
        let total = u32::from_le_bytes(avi[p + 16..p + 20].try_into().unwrap());
        assert_eq!(total, 5);
    }

    #[test]
    fn rejects_bad_frame_size() {
        let err = write_avi(&[vec![0; 10]], AviConfig::new(32, 32, 30)).unwrap_err();
        assert_eq!(err.code, "MUX1001");
    }

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        find(hay, needle).is_some()
    }
    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    // --- 以下为 MJPEG-in-AVI（`write_avi_mjpeg`）测试 ---

    /// 程序化生成 RGBA8 帧序列（平滑渐变 + 随相位移动的色块）。
    fn make_rgba_frames(width: u32, height: u32, count: usize) -> Vec<Vec<u8>> {
        let mut frames = Vec::with_capacity(count);
        let n = (width * height) as usize;
        for fi in 0..count {
            let mut px = vec![0u8; n * 4];
            let phase = fi as f32 / count.max(1) as f32;
            for y in 0..height {
                for x in 0..width {
                    let i = ((y * width + x) as usize) * 4;
                    let fx = x as f32 / width as f32;
                    let fy = y as f32 / height as f32;
                    px[i] = (255.0 * (0.5 + 0.5 * (fx * 0.9 + phase).sin())).clamp(0.0, 255.0) as u8;
                    px[i + 1] =
                        (255.0 * (0.5 + 0.5 * (fy * 0.9 + phase).cos())).clamp(0.0, 255.0) as u8;
                    px[i + 2] =
                        (255.0 * (0.5 + 0.5 * ((fx - fy) * 0.9).sin())).clamp(0.0, 255.0) as u8;
                    px[i + 3] = 255;
                }
            }
            frames.push(px);
        }
        frames
    }

    /// 纯结构校验（不依赖 ffmpeg）：RIFF/AVI 魔数、关键块、avih 帧数。
    #[test]
    fn avi_mjpeg_structure_ok() {
        let (w, h, fps) = (64u32, 48u32, 12u32);
        let frames = make_rgba_frames(w, h, 3);
        let avi = write_avi_mjpeg(&frames, w, h, fps, 80).expect("封装 AVI 失败");

        assert_eq!(&avi[0..4], b"RIFF");
        assert_eq!(&avi[8..12], b"AVI ");
        let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size, avi.len() - 8, "RIFF size 必须等于文件长-8");

        assert!(contains(&avi, b"hdrl"));
        assert!(contains(&avi, b"movi"));
        assert!(contains(&avi, b"idx1"));
        assert!(contains(&avi, b"vids"));
        assert!(contains(&avi, b"MJPG")); // 既作 fccHandler 也作 biCompression
        assert!(contains(&avi, b"00dc"));

        // avih.dwTotalFrames 在 avih body 偏移 16。
        let p = find(&avi, b"avih").unwrap() + 8;
        let total = u32::from_le_bytes(avi[p + 16..p + 20].try_into().unwrap());
        assert_eq!(total, 3);

        // strh.dwLength 在 strh body 偏移 32（fccType4+fccHandler4+dwFlags4+wPriority2+wLanguage2
        // +dwInitialFrames4+dwScale4+dwRate4+dwStart4 = 32），应等于帧数。
        let sp = find(&avi, b"strh").unwrap() + 8;
        let len = u32::from_le_bytes(avi[sp + 32..sp + 36].try_into().unwrap());
        assert_eq!(len, 3);

        // strf.biCompression 应为 "MJPG"。
        let fp = find(&avi, b"strf").unwrap() + 8;
        assert_eq!(&avi[fp + 16..fp + 20], b"MJPG");
    }

    /// 维度 / 帧数不合规应返回 Err，而非 panic。
    #[test]
    fn avi_mjpeg_rejects_bad_dims() {
        let frames = make_rgba_frames(64, 48, 2);
        assert!(write_avi_mjpeg(&frames, 0, 48, 12, 80).is_err());
        assert!(write_avi_mjpeg(&frames, 65, 48, 12, 80).is_err()); // 奇数宽
        assert!(write_avi_mjpeg(&frames, 64, 49, 12, 80).is_err()); // 奇数高
        assert!(write_avi_mjpeg(&[], 64, 48, 12, 80).is_err()); // 空帧
    }

    /// ffmpeg / ffprobe 实时合规校验：能被识别为 mjpeg 流、帧数/时长一致。
    #[test]
    fn avi_mjpeg_identifiable_by_ffmpeg() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg 不可用，跳过实时 AVI 合规校验");
            return;
        }
        let (w, h, fps, count) = (320u32, 240u32, 24u32, 8usize);
        let frames = make_rgba_frames(w, h, count);
        let avi = write_avi_mjpeg(&frames, w, h, fps, 80).expect("封装 AVI 失败");

        let dir = std::env::temp_dir();
        let avi_path = dir.join("prism_test_mjpeg.avi");
        std::fs::write(&avi_path, &avi).expect("写入临时 AVI 失败");

        // 1) ffmpeg 全解（null sink）——必须成功退出且识别 mjpeg 流。
        let out = std::process::Command::new("ffmpeg")
            .args(["-i", avi_path.to_str().unwrap(), "-f", "null", "-"])
            .output()
            .expect("spawn ffmpeg");
        let combined = {
            let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&out.stderr));
            s
        };
        assert!(out.status.success(), "ffmpeg 应成功退出；输出:\n{combined}");
        assert!(
            combined.contains("Video: mjpeg"),
            "ffmpeg 应识别 mjpeg 流；输出:\n{combined}"
        );

        // 2) ffprobe 精确帧数。
        let pf = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                avi_path.to_str().unwrap(),
            ])
            .output()
            .expect("spawn ffprobe");
        let ftext = String::from_utf8_lossy(&pf.stdout);
        let got_frames: u64 = ftext.trim().parse().expect("解析帧数失败");
        assert_eq!(got_frames, count as u64, "ffprobe 帧数应 == {count}");

        // 3) ffprobe 时长 ≈ count/fps。
        let pd = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                avi_path.to_str().unwrap(),
            ])
            .output()
            .expect("spawn ffprobe");
        let dtext = String::from_utf8_lossy(&pd.stdout);
        let got_dur: f64 = dtext.trim().parse().expect("解析时长失败");
        let expected = count as f64 / fps as f64;
        assert!(
            (got_dur - expected).abs() < 0.2,
            "时长 {got_dur} 应≈ {expected}"
        );

        let _ = std::fs::remove_file(&avi_path);
    }

    /// ffmpeg 解码首帧成功，证明标准合规、可被 VLC/浏览器解码。
    #[test]
    fn avi_mjpeg_decode_first_frame() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg 不可用，跳过首帧解码校验");
            return;
        }
        let (w, h, fps, count) = (320u32, 240u32, 24u32, 8usize);
        let frames = make_rgba_frames(w, h, count);
        let avi = write_avi_mjpeg(&frames, w, h, fps, 80).expect("封装 AVI 失败");

        let dir = std::env::temp_dir();
        let avi_path = dir.join("prism_test_mjpeg_dec.avi");
        let png_path = dir.join("prism_test_mjpeg_frame0.png");
        std::fs::write(&avi_path, &avi).expect("写入临时 AVI 失败");

        let out = std::process::Command::new("ffmpeg")
            .args([
                "-i",
                avi_path.to_str().unwrap(),
                "-frames:v",
                "1",
                png_path.to_str().unwrap(),
            ])
            .output()
            .expect("spawn ffmpeg");
        assert!(
            out.status.success(),
            "ffmpeg 解码首帧应成功；stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let meta = std::fs::metadata(&png_path).expect("首帧 PNG 应已生成");
        assert!(meta.len() > 0, "首帧 PNG 应非空");

        let _ = std::fs::remove_file(&avi_path);
        let _ = std::fs::remove_file(&png_path);
    }

    /// 探测本机是否装有 ffmpeg（无则相关测试自动跳过）。
    fn have_ffmpeg() -> bool {
        std::process::Command::new("ffmpeg")
            .args(["-version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    // --- 以下为 MJPEG-in-AVI + PCM 音频流（`write_avi_mjpeg_audio`）测试 ---

    /// 构造一段确定性单声道 PCM（正弦，幅度 0.5）。
    fn make_audio(sample_rate: u32, frames: usize) -> PcmBuffer {
        let data: Vec<f32> =
            (0..frames).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        PcmBuffer::from_mono(sample_rate, data).expect("单声道 PCM 构造应合法")
    }

    /// 纯结构校验（不依赖 ffmpeg）：RIFF/AVI 魔数、双 strl、movi 交错、
    /// idx1 条目数 = 2×帧数、音频 strh/strf 字段。
    #[test]
    fn avi_mjpeg_audio_structure_ok() {
        let (w, h, fps, count) = (64u32, 48u32, 12u32, 4usize);
        let frames = make_rgba_frames(w, h, count);
        // samples_per_frame = 4800/12 = 400 ⇒ 音频 4×400 = 1600 采样帧。
        let audio = make_audio(4_800, count * 400);
        let avi = write_avi_mjpeg_audio(&frames, w, h, fps, 80, &audio)
            .expect("封装带音频 AVI 失败");

        assert_eq!(&avi[0..4], b"RIFF");
        assert_eq!(&avi[8..12], b"AVI ");
        let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size, avi.len() - 8, "RIFF size 必须等于文件长-8");

        // 关键块 / 流标记齐全（视频 + 音频两条流）。
        assert!(contains(&avi, b"hdrl"));
        assert!(contains(&avi, b"movi"));
        assert!(contains(&avi, b"idx1"));
        assert!(contains(&avi, b"vids"));
        assert!(contains(&avi, b"MJPG"));
        assert!(contains(&avi, b"00dc"));
        assert!(contains(&avi, b"auds"));
        assert!(contains(&avi, b"01wb"));

        // avih：dwTotalFrames @16 = 视频帧数；dwStreams @24 = 2。
        let p = find(&avi, b"avih").unwrap() + 8;
        let total = u32::from_le_bytes(avi[p + 16..p + 20].try_into().unwrap());
        assert_eq!(total, count as u32);
        let streams = u32::from_le_bytes(avi[p + 24..p + 28].try_into().unwrap());
        assert_eq!(streams, 2, "avih.dwStreams 必须为 2（视频+音频）");

        // 音频 strh：'auds' 为 strh body 首四CC（fccType）。
        let ap = find(&avi, b"auds").expect("应含 auds 流类型");
        assert_eq!(&avi[ap + 4..ap + 8], b"01wb", "fccHandler 应为 01wb（PCM）");
        // dwScale @20 = 1；dwRate @24 = 采样率；dwLength @32 = 采样帧总数。
        assert_eq!(u32::from_le_bytes(avi[ap + 20..ap + 24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(avi[ap + 24..ap + 28].try_into().unwrap()), 4_800);
        assert_eq!(
            u32::from_le_bytes(avi[ap + 32..ap + 36].try_into().unwrap()),
            (count * 400) as u32,
            "音频 dwLength 应为总采样帧数"
        );

        // 音频 strf（WAVEFORMATEX，18B = 16B 基础字段 + cbSize=0）：
        // strh body(56) + chunk 头(8) 之后。
        let fp = ap + 56 + 8;
        assert_eq!(&avi[fp - 8..fp - 4], b"strf", "strh 之后应紧跟 strf");
        assert_eq!(
            u32::from_le_bytes(avi[fp - 4..fp].try_into().unwrap()),
            18,
            "strf 应为 18 字节 WAVEFORMATEX（16B 基础 + cbSize）"
        );
        assert_eq!(u16::from_le_bytes(avi[fp..fp + 2].try_into().unwrap()), 1, "wFormatTag=PCM");
        assert_eq!(u16::from_le_bytes(avi[fp + 2..fp + 4].try_into().unwrap()), 1, "nChannels");
        assert_eq!(
            u32::from_le_bytes(avi[fp + 4..fp + 8].try_into().unwrap()),
            4_800,
            "nSamplesPerSec"
        );
        assert_eq!(
            u32::from_le_bytes(avi[fp + 8..fp + 12].try_into().unwrap()),
            9_600,
            "nAvgBytesPerSec = 4800×1×2"
        );
        assert_eq!(u16::from_le_bytes(avi[fp + 12..fp + 14].try_into().unwrap()), 2, "nBlockAlign");
        assert_eq!(
            u16::from_le_bytes(avi[fp + 14..fp + 16].try_into().unwrap()),
            16,
            "wBitsPerSample"
        );

        // idx1 条目数 = 2×帧数（00dc + 01wb 各一项）。
        let ip = find(&avi, b"idx1").unwrap();
        let isize = u32::from_le_bytes(avi[ip + 4..ip + 8].try_into().unwrap());
        assert_eq!(isize, (count * 2 * 16) as u32, "idx1 应含 2×帧数 条 16 字节索引");
    }

    /// 音频长度对齐：不足补静音、超出截断，idx1 条目数不变。
    #[test]
    fn avi_mjpeg_audio_pads_and_truncates() {
        let (w, h, fps, count) = (32u32, 32u32, 10u32, 3usize);
        let frames = make_rgba_frames(w, h, count);
        // 短一截：仅 200 采样帧（< 需要的 3×100=300）⇒ 末尾补静音（不 Err）。
        let short = make_audio(1_000, 200);
        let avi = write_avi_mjpeg_audio(&frames, w, h, fps, 80, &short)
            .expect("音频不足应补静音而非报错");
        let ip = find(&avi, b"idx1").unwrap();
        let isize = u32::from_le_bytes(avi[ip + 4..ip + 8].try_into().unwrap());
        assert_eq!(isize, (count * 2 * 16) as u32);
        // 超长：10 帧的音频 ⇒ 截断到 3 帧。
        let long = make_audio(1_000, 10_000);
        let avi = write_avi_mjpeg_audio(&frames, w, h, fps, 80, &long)
            .expect("音频超出应截断而非报错");
        let ip = find(&avi, b"idx1").unwrap();
        let isize = u32::from_le_bytes(avi[ip + 4..ip + 8].try_into().unwrap());
        assert_eq!(isize, (count * 2 * 16) as u32);
        // 音频 strh dwLength 恒 = 3×100 = 300 采样帧（对齐后）。
        let ap = find(&avi, b"auds").unwrap();
        assert_eq!(
            u32::from_le_bytes(avi[ap + 32..ap + 36].try_into().unwrap()),
            300,
            "dwLength 应按帧对齐（补零/截断后）"
        );
    }

    /// 参数不合规 → Err（与 write_avi_mjpeg 同口径 + 采样率必须整除帧率）。
    #[test]
    fn avi_mjpeg_audio_rejects_bad_params() {
        let (w, h, fps) = (64u32, 48u32, 12u32);
        let frames = make_rgba_frames(w, h, 2);
        let audio = make_audio(4_800, 2 * 400);
        assert!(write_avi_mjpeg_audio(&frames, 0, 48, fps, 80, &audio).is_err());
        assert!(write_avi_mjpeg_audio(&frames, 65, 48, fps, 80, &audio).is_err()); // 奇数宽
        assert!(write_avi_mjpeg_audio(&frames, 64, 49, fps, 80, &audio).is_err()); // 奇数高
        assert!(write_avi_mjpeg_audio(&[], 64, 48, fps, 80, &audio).is_err()); // 空帧
        // 帧字节数不符。
        let bad = vec![vec![0u8; 10]; 2];
        assert!(write_avi_mjpeg_audio(&bad, 64, 48, fps, 80, &audio).is_err());
        // 采样率不是帧率整数倍（1000 % 12 = 8）⇒ 无法按帧切分。
        let indiv = make_audio(1_000, 1_000);
        assert!(
            write_avi_mjpeg_audio(&frames, 64, 48, fps, 80, &indiv).is_err(),
            "采样率不整除帧率必须报错"
        );
    }

    // --- 以下为 write_avi_mjpeg_audio_frames（预编码 JPEG 帧入口）测试 ---

    /// 位级对拍：调用方预编码 JPEG 帧后走 frames 入口，必须与
    /// write_avi_mjpeg_audio（内部编码）输出逐字节一致。
    #[test]
    fn avi_mjpeg_audio_frames_matches_encoded_path() {
        let (w, h, fps, count) = (64u32, 48u32, 12u32, 4usize);
        let frames = make_rgba_frames(w, h, count);
        let audio = make_audio(4_800, count * 400);

        let direct = write_avi_mjpeg_audio(&frames, w, h, fps, 80, &audio)
            .expect("write_avi_mjpeg_audio 应成功");
        // 调用方侧用**同一编码器同参数**逐帧预编码（与 write_avi_mjpeg_audio
        // 内部路径逐字节相同的唯一方式）。
        let jpegs: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| {
                encode_jpeg_color(
                    f,
                    w,
                    h,
                    JpegConfig { quality: 80, grayscale: false },
                )
                .expect("预编码应成功")
            })
            .collect();
        let pre = write_avi_mjpeg_audio_frames(&jpegs, w, h, fps, &audio)
            .expect("write_avi_mjpeg_audio_frames 应成功");
        assert_eq!(direct, pre, "两入口输出必须位级一致");
    }

    /// frames 入口结构校验：双流、idx1 条目 = 2×帧数、音频 dwLength 对齐。
    #[test]
    fn avi_mjpeg_audio_frames_structure_ok() {
        let (w, h, fps, count) = (64u32, 48u32, 12u32, 4usize);
        let frames = make_rgba_frames(w, h, count);
        let audio = make_audio(4_800, count * 400);
        let jpegs: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| {
                encode_jpeg_color(
                    f,
                    w,
                    h,
                    JpegConfig { quality: 75, grayscale: false },
                )
                .expect("预编码应成功")
            })
            .collect();
        let avi = write_avi_mjpeg_audio_frames(&jpegs, w, h, fps, &audio)
            .expect("封装应成功");

        assert_eq!(&avi[0..4], b"RIFF");
        assert!(contains(&avi, b"auds"));
        assert!(contains(&avi, b"01wb"));
        let p = find(&avi, b"avih").unwrap() + 8;
        assert_eq!(
            u32::from_le_bytes(avi[p + 16..p + 20].try_into().unwrap()),
            count as u32
        );
        assert_eq!(
            u32::from_le_bytes(avi[p + 24..p + 28].try_into().unwrap()),
            2,
            "avih.dwStreams 必须为 2"
        );
        let ip = find(&avi, b"idx1").unwrap();
        let isize = u32::from_le_bytes(avi[ip + 4..ip + 8].try_into().unwrap());
        assert_eq!(isize, (count * 2 * 16) as u32, "idx1 应含 2×帧数 条索引");
        // 音频 strh dwLength = 4×400 采样帧（按帧对齐）。
        let ap = find(&avi, b"auds").unwrap();
        assert_eq!(
            u32::from_le_bytes(avi[ap + 32..ap + 36].try_into().unwrap()),
            (count * 400) as u32
        );
    }

    /// frames 入口参数校验口径：空序列 / 奇数尺寸 / 采样率不整除帧率 → Err。
    #[test]
    fn avi_mjpeg_audio_frames_rejects_bad_params() {
        let fps = 12u32;
        let audio = make_audio(4_800, 2 * 400);
        let one = vec![vec![0xFFu8; 64]]; // 伪造 JPEG 字节（内容不校验，仅占位）
        assert!(write_avi_mjpeg_audio_frames(&one, 0, 48, fps, &audio).is_err());
        assert!(write_avi_mjpeg_audio_frames(&one, 65, 48, fps, &audio).is_err()); // 奇数宽
        assert!(write_avi_mjpeg_audio_frames(&one, 64, 49, fps, &audio).is_err()); // 奇数高
        assert!(write_avi_mjpeg_audio_frames(&[], 64, 48, fps, &audio).is_err()); // 空序列
        let indiv = make_audio(1_000, 1_000);
        assert!(
            write_avi_mjpeg_audio_frames(&one, 64, 48, fps, &indiv).is_err(),
            "采样率不整除帧率必须报错"
        );
    }

    /// ffmpeg / ffprobe 实时合规校验：双流（mjpeg + pcm_s16le）、视频帧数
    /// 正确、音频样本数（⇒ 时长）与视频一致、整片可解码。
    #[test]
    fn avi_mjpeg_audio_identifiable_by_ffmpeg() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg 不可用，跳过带音频 AVI 合规校验");
            return;
        }
        let (w, h, fps, count) = (320u32, 240u32, 24u32, 8usize);
        let frames = make_rgba_frames(w, h, count);
        // samples_per_frame = 24000/24 = 1000 ⇒ 8×1000 = 8000 样本 = 0.3333s。
        let audio = make_audio(24_000, count * 1_000);
        let avi = write_avi_mjpeg_audio(&frames, w, h, fps, 80, &audio)
            .expect("封装带音频 AVI 失败");

        let dir = std::env::temp_dir();
        let avi_path = dir.join("prism_test_mjpeg_audio.avi");
        std::fs::write(&avi_path, &avi).expect("写入临时 AVI 失败");
        let path = avi_path.to_str().unwrap();

        // 1) ffprobe：2 条流，编码分别为 mjpeg / pcm_s16le。
        let ps = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                path,
            ])
            .output()
            .expect("spawn ffprobe");
        let stext = String::from_utf8_lossy(&ps.stdout);
        let codecs: Vec<&str> =
            stext.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        assert_eq!(codecs.len(), 2, "应报告 2 条流，实测 {codecs:?}");
        assert!(codecs.contains(&"mjpeg"), "应含 mjpeg 视频流，实测 {codecs:?}");
        assert!(codecs.contains(&"pcm_s16le"), "应含 pcm_s16le 音频流，实测 {codecs:?}");

        // 2) ffprobe：视频帧数正确。
        let pf = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                path,
            ])
            .output()
            .expect("spawn ffprobe");
        let ftext = String::from_utf8_lossy(&pf.stdout);
        let got_frames: u64 = ftext.trim().parse().expect("解析帧数失败");
        assert_eq!(got_frames, count as u64, "ffprobe 帧数应 == {count}");

        // 3) 音频样本数（解码到 s16le 后按字节数核对）：时长 ≈ 视频时长。
        let pa = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i", path, "-map", "0:a:0", "-f", "s16le", "-"])
            .output()
            .expect("spawn ffmpeg");
        assert!(pa.status.success(), "ffmpeg 解音频应成功");
        let samples = pa.stdout.len() / 2; // 16-bit 单声道
        assert_eq!(samples, count * 1_000, "解码样本数应为 {count}×1000");
        let audio_dur = samples as f64 / 24_000.0;
        let video_dur = count as f64 / fps as f64;
        assert!(
            (audio_dur - video_dur).abs() < 0.2,
            "音频时长 {audio_dur} 应≈ 视频时长 {video_dur}"
        );

        // 4) ffmpeg 全解（null sink）必须成功退出。
        let out = std::process::Command::new("ffmpeg")
            .args(["-i", path, "-f", "null", "-"])
            .output()
            .expect("spawn ffmpeg");
        assert!(
            out.status.success(),
            "ffmpeg 应成功退出；stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let _ = std::fs::remove_file(&avi_path);
    }
}
