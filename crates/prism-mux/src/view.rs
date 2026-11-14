//! V3.1 收尾 C —— 编码 IO 的**零拷贝帧视图**（MJPEG / PNG / mux 同源）。
//!
//! # 去掉的那份重复拷贝
//! 旧 `cmd_render` 出片后做 `deliverable.frames.iter().map(|fb| fb.pixels.clone())`
//! 生成 `Vec<Vec<u8>>`，再交给 `write_mp4_jpeg_color(&[Vec<u8>], ...)`：
//! 720p 每帧多 3.7 MB 分配 + 拷贝，3600 帧 = 13.3 GB 无谓 memcpy。
//! 本模块让整条 mux 链只持有 `&[u8]` 借用视图，指针全程不变
//! （[`verify_view_identity`] 用 `as_ptr()` 身份断言，防「声明零拷贝但实际拷贝」）。
//!
//! PNG 出口同样直接借用视图字节，委托 `prism_io::encode_png_view`，
//! 不构造中间 `Framebuffer`、不 clone 像素。

use crate::{JpegConfig, MuxError, MuxResult, TrackSpec};

/// RGBA8 帧的借用视图（不拥有内存，构造零分配）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameView<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba8: &'a [u8],
}

impl<'a> FrameView<'a> {
    /// 从任意 RGBA8 字节切片构造（长度校验，绝不静默截断）。
    pub fn from_bytes(rgba8: &'a [u8], width: u32, height: u32) -> MuxResult<Self> {
        let need: usize = (width as usize).saturating_mul(height as usize).saturating_mul(4);
        if width == 0 || height == 0 || rgba8.len() < need {
            return Err(MuxError::new(
                "MUX2010",
                "view::FrameView::from_bytes",
                format!("RGBA8 缓冲 {} 字节不足 {}×{}×4 = {}", rgba8.len(), width, height, need),
                "提供完整 RGBA8 帧缓冲（width/height 必须为正）",
            ));
        }
        Ok(FrameView { width, height, rgba8 })
    }

    /// 裸字节指针（零拷贝身份判据用）。
    pub fn as_ptr(&self) -> *const u8 {
        self.rgba8.as_ptr()
    }
}

/// 把 `Vec<Vec<u8>>` 帧序列映射成借用视图数组（**不拷贝像素**）。
pub fn views_of(frames: &'_ [Vec<u8>], width: u32, height: u32) -> MuxResult<Vec<FrameView<'_>>> {
    frames
        .iter()
        .map(|f| FrameView::from_bytes(f.as_slice(), width, height))
        .collect()
}

/// 直接对 `Framebuffer` 风格的逐帧字节切片构造视图（CLI 主链路入口）。
pub fn views_of_slices<'a>(frames: &'a [&'a [u8]], width: u32, height: u32) -> MuxResult<Vec<FrameView<'a>>> {
    frames
        .iter()
        .map(|f| FrameView::from_bytes(f, width, height))
        .collect()
}

/// 零拷贝身份断言：视图的 `as_ptr()` 必须与源缓冲一致。
pub fn verify_view_identity(view: &FrameView<'_>, source: &[u8]) {
    assert_eq!(
        view.as_ptr(),
        source.as_ptr(),
        "FrameView 声明零拷贝但发生了拷贝（指针不等）"
    );
}

/// 视图版 MP4(MJPEG) 出口：逐帧只借用 `&[u8]`，容器字节一次产出。
pub fn write_mp4_views(
    views: &[FrameView<'_>],
    video: TrackSpec,
    audio_pcm16: Option<&[u8]>,
    audio_sample_rate: u32,
    audio_channels: u16,
    jpeg_cfg: JpegConfig,
) -> MuxResult<Vec<u8>> {
    let slices: Vec<&[u8]> = views.iter().map(|v| v.rgba8).collect();
    crate::mp4::write_mp4_jpeg_color_views(
        &slices,
        video,
        audio_pcm16,
        audio_sample_rate,
        audio_channels,
        jpeg_cfg,
    )
}

// ---------------------------------------------------------------------------
// V3.1 收尾 C：PNG 零拷贝视图
// ---------------------------------------------------------------------------

/// 单个 PNG 帧：**直接借用** `FrameView.rgba8` 编码，不构造中间帧、不 clone。
///
/// 与 `prism_io::encode_png(&Framebuffer)` 逐字节相同（同一 CRC32/adler32/
/// 存储型 deflate 路径），故 golden 资产与 manifest 哈希零回归。
pub fn png_bytes_of_view(view: &FrameView<'_>) -> Vec<u8> {
    prism_io::encode_png_view(view.rgba8, view.width, view.height)
}

/// 批量 PNG 帧视图（每帧仍只借用字节，零额外像素拷贝）。
pub fn encode_png_views(views: &[FrameView<'_>]) -> Vec<Vec<u8>> {
    views.iter().map(png_bytes_of_view).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_are_borrowed_and_pointer_identical() {
        let frames: Vec<Vec<u8>> = vec![vec![7u8; 4 * 4 * 4], vec![9u8; 4 * 4 * 4]];
        let views = views_of(&frames, 4, 4).unwrap();
        assert_eq!(views.len(), 2);
        for (v, src) in views.iter().zip(frames.iter()) {
            verify_view_identity(v, src.as_slice());
        }
    }

    #[test]
    fn short_buffer_is_structurally_rejected() {
        let short = vec![0u8; 10];
        assert!(FrameView::from_bytes(&short, 8, 8).is_err());
        assert!(FrameView::from_bytes(&short, 0, 0).is_err());
    }

    #[test]
    fn write_mp4_views_matches_owned_frames_bytewise() {
        let frames: Vec<Vec<u8>> = vec![vec![128u8; 8 * 8 * 4], vec![64u8; 8 * 8 * 4]];
        let views = views_of(&frames, 8, 8).unwrap();
        let spec = TrackSpec { width: 8, height: 8, fps: 12 };
        let cfg = JpegConfig { quality: 85, grayscale: false };
        let owned = crate::write_mp4_jpeg_color(&frames, spec, None, 0, 0, cfg).unwrap();
        let borrowed = write_mp4_views(&views, spec, None, 0, 0, cfg).unwrap();
        assert_eq!(owned, borrowed, "零拷贝视图产出必须与 owned 路径逐字节一致");
    }

    #[test]
    fn png_views_match_owned_framebuffer_bytewise() {
        let frames: Vec<Vec<u8>> = vec![vec![200u8; 4 * 4 * 4], vec![33u8; 4 * 4 * 4]];
        let views = views_of(&frames, 4, 4).unwrap();
        let pngs = encode_png_views(&views);
        assert_eq!(pngs.len(), frames.len());
        for (i, f) in frames.iter().enumerate() {
            verify_view_identity(&views[i], f.as_slice());
            let fb = prism_gpu::Framebuffer { width: 4, height: 4, pixels: f.clone() };
            assert_eq!(
                pngs[i],
                prism_io::encode_png(&fb),
                "PNG 视图必须与 owned Framebuffer 逐字节一致"
            );
            assert_eq!(pngs[i], png_bytes_of_view(&views[i]));
        }
    }
}
