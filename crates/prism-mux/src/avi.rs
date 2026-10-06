//! AVI 容器写入（C4；路线图 C4 行 / §3.5 容器路线：AVI(rawvideo) "双击可播"底线）。
//!
//! # 结构（RIFF 容器）
//!
//! `RIFF("AVI ")`
//! - `LIST("hdrl")` — `avih`（主头）+ `LIST("strl")`（`strh` + `strf`）；
//! - `LIST("movi")` — 每帧一个 `00dc`（MJPEG 数据块）；
//! - `idx1` — 每帧索引项（offset/size/flags）。
//!
//! 编码：

use crate::{jpeg, MuxError, MuxResult, JpegConfig};

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
}
