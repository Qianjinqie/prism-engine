//! 源抽象（B11；D-17 读入策略）：image / video / audio_source 三类源 + 读入路径标注。
//!
//! 素材"能不能读"如实标注，**永静默不失败**（D-17）：能原生解则 [`ReadKind::Native`]，
//! 需外部协作则 [`ReadKind::ExternalFfmpeg`]，两者皆不可则返回结构化错误。
//!
//! # 三类源
//!
//! - [`MediaSource::Image`] — 静态图（PNG/GIF/APNG 原生；JPEG/WebP 经 ffmpeg）；
//! - [`MediaSource::Video`] — 视频（原生 GIF/APNG 动画；mp4/mov/webm 经 ffmpeg）；
//! - [`MediaSource::AudioSource`] — 音频（WAV 原生；mp3/aac 经 ffmpeg）。
//!
//! 本模块只落**抽象与标注**：具体解码在 [`crate::gif`]/[`crate::apng`]/
//! [`crate::video`]/[`crate::audio`]。

use crate::MediaError;

// ---------------------------------------------------------------------------
// 像素图（RGBA8）
// ---------------------------------------------------------------------------

/// RGBA8 像素图（行主序，每像素 4 字节）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelImage {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// `width × height × 4` 的 RGBA8 缓冲。
    pub pixels: Vec<u8>,
}

impl PixelImage {
    /// 全透明（或指定填充）图。
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, pixels: vec![0u8; (width as usize) * (height as usize) * 4] }
    }

    /// 读一个像素 RGBA（越界返回 `[0,0,0,0]`）。
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        let i = ((y as usize) * self.width as usize + x as usize) * 4;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2], self.pixels[i + 3]]
    }

    /// 写一个像素 RGBA（越界忽略）。
    pub fn set(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = ((y as usize) * self.width as usize + x as usize) * 4;
        self.pixels[i..i + 4].copy_from_slice(&rgba);
    }
}

// ---------------------------------------------------------------------------
// 源种类与读入路径
// ---------------------------------------------------------------------------

/// 源种类（三类源，D-17）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// 静态图。
    Image,
    /// 视频（含帧序列）。
    Video,
    /// 音频源。
    AudioSource,
}

/// 读入路径（D-17：如实标注，永静默不失败）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadKind {
    /// 引擎原生可解（GIF/APNG 动画、PNG、WAV）。
    Native,
    /// 需外部 ffmpeg 协作（mp4/mov/webm/mp3 等）——运行时探测，有则读、无则降级。
    ExternalFfmpeg,
    /// 原生与外部皆不可（如实标注为不可读，调用方显式处理）。
    Unreadable,
}

/// 一个素材源：路径 + 种类 + 读入路径标注。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSource {
    /// 素材路径（相对或绝对；本层只做扩展名与存在性判断，不读取内容）。
    pub path: String,
    /// 源种类。
    pub kind: SourceKind,
    /// 读入路径标注（由 [`MediaSource::classify`] 按扩展名给出）。
    pub read_kind: ReadKind,
}

impl MediaSource {
    /// 按路径与种类构造源，自动标注读入路径。
    pub fn new(path: impl Into<String>, kind: SourceKind) -> Self {
        let path = path.into();
        let read_kind = classify_read_kind(&path, kind);
        Self { path, kind, read_kind }
    }

    /// 按扩展名与种类标注读入路径（纯函数）。
    pub fn classify(path: &str, kind: SourceKind) -> ReadKind {
        classify_read_kind(path, kind)
    }

    /// 扩展名（小写，无点）。
    pub fn extension(&self) -> String {
        extension_of(&self.path)
    }
}

/// 取小写扩展名。
pub fn extension_of(path: &str) -> String {
    match path.rsplit_once('.') {
        Some((_, ext)) => ext.to_ascii_lowercase(),
        None => String::new(),
    }
}

/// 读入路径判定（D-17）：原生可解扩展名 → [`ReadKind::Native`]，ffmpeg 协作 →
/// [`ReadKind::ExternalFfmpeg`]，未知 → [`ReadKind::Unreadable`]。
pub fn classify_read_kind(path: &str, kind: SourceKind) -> ReadKind {
    let ext = extension_of(path);
    match kind {
        SourceKind::Image | SourceKind::Video => match ext.as_str() {
            // JPEG 基线解码已内置（`prism_io::decode_image` 的 L0 路径），
            // 分类从「ffmpeg 协作」修正为「原生可解」。
            "gif" | "apng" | "png" | "jpg" | "jpeg" => ReadKind::Native,
            "webp" | "bmp" => ReadKind::ExternalFfmpeg,
            "mp4" | "mov" | "webm" | "mkv" | "avi" | "m4v" => ReadKind::ExternalFfmpeg,
            _ => ReadKind::Unreadable,
        },
        SourceKind::AudioSource => match ext.as_str() {
            "wav" => ReadKind::Native,
            "mp3" | "aac" | "m4a" | "ogg" | "flac" => ReadKind::ExternalFfmpeg,
            _ => ReadKind::Unreadable,
        },
    }
}

// ---------------------------------------------------------------------------
// 源错误
// ---------------------------------------------------------------------------

/// 源/读入错误（宪章 C18 三段式；错误码 `MD1xxx` 段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】。
    pub location: String,
    /// 三段式二【什么事】。
    pub problem: String,
    /// 三段式三【怎么改】。
    pub fix: String,
}

impl SourceError {
    /// 构造三段式错误。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { code, location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for SourceError {}

impl From<SourceError> for MediaError {
    fn from(e: SourceError) -> Self {
        MediaError::new(e.code, e.location, e.problem, e.fix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_native_vs_external_vs_unreadable() {
        assert_eq!(MediaSource::classify("a.gif", SourceKind::Video), ReadKind::Native);
        assert_eq!(MediaSource::classify("a.apng", SourceKind::Image), ReadKind::Native);
        assert_eq!(MediaSource::classify("a.PNG", SourceKind::Image), ReadKind::Native);
        assert_eq!(MediaSource::classify("a.mp4", SourceKind::Video), ReadKind::ExternalFfmpeg);
        assert_eq!(MediaSource::classify("a.wav", SourceKind::AudioSource), ReadKind::Native);
        assert_eq!(MediaSource::classify("a.mp3", SourceKind::AudioSource), ReadKind::ExternalFfmpeg);
        assert_eq!(MediaSource::classify("a.xyz", SourceKind::Image), ReadKind::Unreadable);
    }

    #[test]
    fn pixel_image_accessors() {
        let mut img = PixelImage::new(2, 2);
        img.set(1, 1, [1, 2, 3, 4]);
        assert_eq!(img.get(1, 1), [1, 2, 3, 4]);
        assert_eq!(img.get(0, 0), [0, 0, 0, 0]);
        assert_eq!(img.get(9, 9), [0, 0, 0, 0], "越界返回全零");
    }

    #[test]
    fn extension_extraction() {
        assert_eq!(extension_of("dir/x.TAR.GZ"), "gz");
        assert_eq!(extension_of("noext"), "");
    }
}
