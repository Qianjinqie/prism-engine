//! 统一图像解码门面。
//!
//! # 分层（L0/L1/L2）与「永不静默失败」
//!
//! | 层 | 含义 | 行为 |
//! |----|------|------|
//! | L0 | 引擎内置解码 | PNG（[`crate::png`]）、JPEG 基线（[`crate::jpeg_baseline`]） |
//! | L1 | 容器/音视频编码 | 返回三段式错误：未内置解码 / ffmpeg 协作命令 / 装 ffmpeg 后重试 |
//! | L2 | 矩阵外编码 | 返回明确错误：不在解码矩阵内，请先转码 |
//!
//! 任何情况下都**不做静默降级**：不认识就报错，错误文案必须能让调用方知道
//! 「在哪、什么事、怎么改」。
//!
//! # 扩展名 vs 魔数
//!
//! 先看魔数（PNG 签名 / JPEG SOI），再退到扩展名——扩展名写错的文件仍能正确解码，
//! 但两者都识别不出时报错而不是猜。

use crate::jpeg_baseline::decode_baseline;
use crate::png::{decode_png, PngImage, PNG_SIGNATURE};

/// 解码结果（按色彩域与位宽分档，避免调用方二次推断）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedImage {
    /// 8-bit 单通道灰度。
    Gray8 {
        /// 宽（像素）。
        w: u32,
        /// 高（像素）。
        h: u32,
        /// 灰度样本（`w*h`，行主序）。
        samples: Vec<u8>,
    },
    /// 16-bit 单通道灰度。
    Gray16 {
        /// 宽（像素）。
        w: u32,
        /// 高（像素）。
        h: u32,
        /// 灰度样本（`w*h`，行主序）。
        samples: Vec<u16>,
    },
    /// 8-bit RGBA（JPEG 恒为该档，alpha = 255）。
    Rgba8 {
        /// 宽（像素）。
        w: u32,
        /// 高（像素）。
        h: u32,
        /// RGBA 像素（`w*h*4`，行主序）。
        pixels: Vec<u8>,
    },
    /// 16-bit RGBA。
    Rgba16 {
        /// 宽（像素）。
        w: u32,
        /// 高（像素）。
        h: u32,
        /// RGBA 像素（`w*h*4`，行主序）。
        pixels: Vec<u16>,
    },
}

impl DecodedImage {
    /// 宽（像素）。
    pub fn width(&self) -> u32 {
        match self {
            DecodedImage::Gray8 { w, .. }
            | DecodedImage::Gray16 { w, .. }
            | DecodedImage::Rgba8 { w, .. }
            | DecodedImage::Rgba16 { w, .. } => *w,
        }
    }

    /// 高（像素）。
    pub fn height(&self) -> u32 {
        match self {
            DecodedImage::Gray8 { h, .. }
            | DecodedImage::Gray16 { h, .. }
            | DecodedImage::Rgba8 { h, .. }
            | DecodedImage::Rgba16 { h, .. } => *h,
        }
    }

    /// 位深（8 或 16）。
    pub fn bit_depth(&self) -> u8 {
        match self {
            DecodedImage::Gray8 { .. } | DecodedImage::Rgba8 { .. } => 8,
            DecodedImage::Gray16 { .. } | DecodedImage::Rgba16 { .. } => 16,
        }
    }

    /// 转为 RGBA8（16-bit 取高字节；灰度扩展三通道复本）。
    pub fn to_rgba8(&self) -> Vec<u8> {
        match self {
            DecodedImage::Rgba8 { pixels, .. } => pixels.clone(),
            DecodedImage::Gray8 { w, h, samples } => {
                let n: usize = (*w as usize) * (*h as usize);
                let mut out: Vec<u8> = vec![255u8; n * 4];
                for i in 0..n {
                    let v: u8 = samples[i];
                    out[i * 4] = v;
                    out[i * 4 + 1] = v;
                    out[i * 4 + 2] = v;
                }
                out
            }
            DecodedImage::Gray16 { w, h, samples } => {
                let n: usize = (*w as usize) * (*h as usize);
                let mut out: Vec<u8> = vec![255u8; n * 4];
                for i in 0..n {
                    let v: u8 = (samples[i] >> 8) as u8;
                    out[i * 4] = v;
                    out[i * 4 + 1] = v;
                    out[i * 4 + 2] = v;
                }
                out
            }
            DecodedImage::Rgba16 { w, h, pixels } => {
                let n: usize = (*w as usize) * (*h as usize);
                let mut out: Vec<u8> = vec![0u8; n * 4];
                for i in 0..n * 4 {
                    out[i] = (pixels[i] >> 8) as u8;
                }
                out
            }
        }
    }
}

/// 容器/音视频扩展名（L1：需要外部 ffmpeg 协作的编码）。
const L1_EXTS: [&str; 17] = [
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "flv", "wmv", "mpg", "mpeg", "ts", "mp3", "aac",
    "m4a", "ogg", "flac", "opus",
];

/// 矩阵外扩展名（L2：既非内置也非 ffmpeg 协作路径）。
const L2_EXTS: [&str; 10] =
    ["webp", "prores", "tiff", "tif", "bmp", "gif", "heic", "heif", "avif", "jxl"];

/// 归一化扩展名：去点、转小写、去查询串残留。
fn normalize_ext(ext: &str) -> String {
    let e: &str = ext.trim();
    let e: &str = e.strip_prefix('.').unwrap_or(e);
    e.to_ascii_lowercase()
}

/// 按魔数判定格式（`None` = 认不出）。
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 8 && bytes[0..8] == PNG_SIGNATURE {
        return Some("png");
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        return Some("jpeg");
    }
    None
}

/// L1 三段式错误文案（在哪 / 什么事 / 怎么改）。
fn l1_error(ext: &str) -> String {
    format!(
        "'{}' 未内置解码（L1：容器/音视频编码不在引擎 L0 范围）；\
         经 ffmpeg 协作可用：ffmpeg -i input.{} frame_%04d.png；\
         安装 ffmpeg 后重试",
        ext, ext
    )
}

/// L2 三段式错误文案（矩阵外，拒绝静默降级）。
fn l2_error(ext: &str) -> String {
    format!(
        "'{}' 不在内置解码矩阵内（L2：既非 L0 内置，也非 ffmpeg 协作路径）；\
         prism 不静默降级为未知格式；\
         请先外部转为 PNG/JPEG（如 ffmpeg -i input.{} out.png）后重试",
        ext, ext
    )
}

/// 从 `PngImage` 选择最贴合的输出分档。
fn png_to_decoded(img: &PngImage) -> DecodedImage {
    let w: u32 = img.width;
    let h: u32 = img.height;
    match (img.color_type, img.bit_depth) {
        // 纯灰度：不浪费 alpha 通道，保留原始样本位宽。
        (0, 8) => DecodedImage::Gray8 { w, h, samples: img.gray_u8() },
        (0, 16) => DecodedImage::Gray16 { w, h, samples: img.gray_u16() },
        (_, 16) => DecodedImage::Rgba16 { w, h, pixels: img.to_rgba16() },
        // 其余 8-bit（真彩 / 调色板 / 带 alpha）统一展开 RGBA8。
        _ => DecodedImage::Rgba8 { w, h, pixels: img.to_rgba8() },
    }
}

/// 统一图像解码入口。
///
/// # 参数
/// * `bytes` — 完整文件字节。
/// * `ext` — 扩展名（带不带点、大小写均可；`""` 表示未知，将只靠魔数识别）。
///
/// # 返回
/// * `Ok(DecodedImage)` — L0 解码成功；
/// * `Err(String)` — L1/L2/损坏，文案含「在哪、什么事、怎么改」。
///
/// # 决策顺序
/// 1. 魔数命中 PNG/JPEG → 直接走内置解码（扩展名仅作日志线索）；
/// 2. 扩展名命中 L1 → ffmpeg 协作三段式错误；
/// 3. 扩展名命中 L2 或未知 → 矩阵外三段式错误。
pub fn decode_image(bytes: &[u8], ext: &str) -> Result<DecodedImage, String> {
    let e: String = normalize_ext(ext);

    // ---- L0：魔数优先 ----
    match sniff(bytes) {
        Some("png") => {
            let img: PngImage = decode_png(bytes)?;
            return Ok(png_to_decoded(&img));
        }
        Some("jpeg") => {
            let img = decode_baseline(bytes)?;
            return Ok(DecodedImage::Rgba8 {
                w: img.width,
                h: img.height,
                pixels: img.to_rgba8(),
            });
        }
        _ => {}
    }

    // ---- L0：扩展名兜底（魔数认不出时按扩展名走内置解码器，失败则透传其错误） ----
    if e == "png" {
        return Err(
            "PNG 解码失败：文件头 8 字节不是 PNG 签名（扩展名为 png 但内容不符）\
             ——请确认文件未损坏，或用 file 命令核对真实格式"
                .to_string(),
        );
    }
    if e == "jpg" || e == "jpeg" {
        return Err(
            "JPEG 解码失败：文件头不是 FFD8（SOI）标记（扩展名为 jpeg 但内容不符）\
             ——请确认文件未损坏，或用 file 命令核对真实格式"
                .to_string(),
        );
    }

    // ---- L1：ffmpeg 协作路径 ----
    if L1_EXTS.contains(&e.as_str()) {
        return Err(l1_error(&e));
    }

    // ---- L2：矩阵外 ----
    if L2_EXTS.contains(&e.as_str()) {
        return Err(l2_error(&e));
    }

    // 未知扩展名 + 魔数不识别：仍然报错（永不静默）。
    if e.is_empty() {
        return Err(
            "无法识别图像格式：扩展名为空且魔数不匹配 PNG/JPEG；\
             请显式传入扩展名（png/jpeg）或确认文件来源；\
             外部转码后再试（ffmpeg -i input out.png）"
                .to_string(),
        );
    }
    Err(l2_error(&e))
}

/// 便捷判定：某扩展名是否由引擎 **内置**（L0）解码。
pub fn is_native_format(ext: &str) -> bool {
    let e: String = normalize_ext(ext);
    matches!(e.as_str(), "png" | "jpg" | "jpeg")
}
