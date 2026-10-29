//! EXR 序列交付路径：把帧序列逐帧落成 `.exr`，并提供读回校验。
//!
//! # 定位
//!
//! 交付矩阵（`docs/交付矩阵.md`）里的 `EXR-seq` 一列就是本模块：**无损、逐帧
//! 独立**的图像序列容器，面向合成/调色/后期（保留浮点线性值、无帧间预测、可任取单帧）。
//! 写入器**完全复用** `prism-io` 的 [`prism_io::exr_write`]（扫描线 flat 通道 +
//! 十种压缩），本模块只负责三件事：
//!
//! 1. **RGBA8 → 4 个浮点通道平面**（`R/G/B/A`，值域 `[0,1]`，`u8 / 255` 精确量化）；
//! 2. **命名与目录**（`{prefix}.{index:04}.exr`，索引从 `start_index` 起）；
//! 3. **三段式错误**（C18：在哪 / 什么事 / 怎么改）。
//!
//! # 确定性
//!
//! - 通道顺序固定 `R → G → B → A`（不依赖 `HashMap`，与 `header_for` 的入参顺序一致）；
//! - 属性顺序、块序由 `exr_write` 固定，同输入两跑字节相同（该性质在 `prism-io` 内
//!   已有断言；本模块不引入任何随机/时间戳字段）；
//! - 无压缩（[`prism_io::Compression::None`]）与 deflate 系列（`Zip`/`Zips`）均为
//!   **逐位无损**，写→读逐通道严格相等。
//!
//! # 与 `Deliverable` 的关系
//!
//! 输入是 `&[Framebuffer]` 而不是 [`crate::movie::Deliverable`]，以免本模块反向耦合
//! 成片管线；调用方传 `deliverable.frames.as_slice()` 即可。

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use prism_gpu::Framebuffer;
use prism_io::exr::{ChannelPlane, ExrImage, PixelType};
use prism_io::exr_write::{header_for, write_scanline_image};
use prism_io::Compression;

/// 序列写入选项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExrSeqOptions {
    /// 压缩方式。`None` / `Rle` / `Zips` / `Zip` / `Piz` / `Pxr24` 中
    /// **无损**的是 `None`/`Rle`/`Zips`/`Zip`/`Piz`（`Pxr24` 对 `Float` 通道会截断到
    /// 24 位定点，`B44` 系列仅对 `Half` 有损）。要"写→读逐通道一致"请用无损档。
    pub compression: Compression,
    /// 起始帧号（文件名补零到 4 位）。`0` → `frame.0000.exr`。
    pub start_index: u32,
}

impl Default for ExrSeqOptions {
    fn default() -> Self {
        // `Zip`（16 扫描线一块的 deflate）：无损、跨实现最通用，体积约为未压缩的
        // 1/3～1/2（实测见 `docs/交付矩阵.md`）。
        ExrSeqOptions { compression: Compression::Zip, start_index: 0 }
    }
}

/// EXR 序列写入 / 读回错误（C18 三段式：在哪 / 什么事 / 怎么改）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrSeqError {
    /// 出错位置（`exr_seq::<函数> @ <path>`）。
    pub location: String,
    /// 发生了什么。
    pub problem: String,
    /// 怎么改。
    pub fix: String,
}

impl ExrSeqError {
    fn new(
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        ExrSeqError { location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl fmt::Display for ExrSeqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}：{}（修复：{}）", self.location, self.problem, self.fix)
    }
}

impl std::error::Error for ExrSeqError {}

/// 一帧的分辨率（宽, 高）——序列内所有帧必须一致。
fn frame_dims(frames: &[Framebuffer]) -> Result<(u32, u32), ExrSeqError> {
    let first = frames.first().ok_or_else(|| {
        ExrSeqError::new(
            "exr_seq::frame_dims",
            "帧序列为空，无法确定分辨率",
            "至少传入 1 帧（`Deliverable::frames` 非空，或先渲染一帧）",
        )
    })?;
    let expected = (first.width, first.height);
    if expected.0 == 0 || expected.1 == 0 {
        return Err(ExrSeqError::new(
            "exr_seq::frame_dims",
            format!("首帧尺寸为 {}×{}（必须非 0）", expected.0, expected.1),
            "检查渲染分辨率参数（`FrameGraph` / `render_frame` 的 w/h）",
        ));
    }
    for (i, f) in frames.iter().enumerate() {
        if (f.width, f.height) != expected {
            return Err(ExrSeqError::new(
                "exr_seq::frame_dims",
                format!(
                    "第 {i} 帧尺寸 {}×{} 与首帧 {}×{} 不一致",
                    f.width, f.height, expected.0, expected.1
                ),
                "EXR 序列要求全序列同分辨率；请逐段导出或先统一重采样",
            ));
        }
        let need = (f.width as usize) * (f.height as usize) * 4;
        if f.pixels.len() != need {
            return Err(ExrSeqError::new(
                "exr_seq::frame_dims",
                format!(
                    "第 {i} 帧像素长度 {} != {}×{}×4={need}",
                    f.pixels.len(),
                    f.width,
                    f.height
                ),
                "帧缓冲被截断（渲染失败或被复用后未重填）；重新渲染该帧",
            ));
        }
    }
    Ok(expected)
}

/// 把一帧 RGBA8 拆成 4 个 `f32` 通道平面（值域 `[0,1]`，顺序 `R,G,B,A`）。
///
/// `u8 → f32` 用 `v as f32 / 255.0`：对每个 `v ∈ 0..=255` 该商在 `f32` 下唯一，
/// 读回时 `(v * 255.0).round() as u8` 精确还原（逐通道一致测试的依据）。
pub fn frame_to_planes(frame: &Framebuffer) -> Vec<ChannelPlane> {
    let n = (frame.width as usize) * (frame.height as usize);
    let names = ["R", "G", "B", "A"];
    let mut planes: Vec<ChannelPlane> = Vec::with_capacity(4);
    for (c, name) in names.iter().enumerate() {
        let mut values: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            values.push(frame.pixels[i * 4 + c] as f32 / 255.0);
        }
        planes.push(ChannelPlane {
            name: (*name).to_string(),
            pixel_type: PixelType::Float,
            width: frame.width,
            height: frame.height,
            values,
        });
    }
    planes
}

/// 序列文件名：`{prefix}.{index:04}.exr`。
pub fn frame_file_name(prefix: &str, index: u32) -> String {
    format!("{prefix}.{index:04}.exr")
}

/// 把帧序列逐帧写成 `.exr`，返回写出的文件路径（按帧序）。
///
/// # 参数
/// - `dir`：输出目录（不存在则创建，含父级）；
/// - `prefix`：文件名前缀（不含扩展名）；
/// - `frames`：帧序列，**必须同分辨率**；
/// - `opts`：压缩与起始帧号。
///
/// # 错误
/// 空序列、尺寸不一致、像素长度不符、EXR 编码失败、目录/文件写入失败均返回
/// [`ExrSeqError`]（三段式）。
pub fn write_exr_sequence(
    dir: &Path,
    prefix: &str,
    frames: &[Framebuffer],
    opts: ExrSeqOptions,
) -> Result<Vec<PathBuf>, ExrSeqError> {
    let (w, h) = frame_dims(frames)?;
    fs::create_dir_all(dir).map_err(|e| {
        ExrSeqError::new(
            format!("exr_seq::write_exr_sequence @ {}", dir.display()),
            format!("创建输出目录失败：{e}"),
            "检查路径权限/磁盘空间，或改用已存在且可写的目录",
        )
    })?;
    // 头部按尺寸与固定通道表构造一次（`dataWindow` 原点为 (0,0)）。
    let header = header_for(
        w,
        h,
        &[("R", PixelType::Float), ("G", PixelType::Float), ("B", PixelType::Float),
          ("A", PixelType::Float)],
        opts.compression,
    );

    let mut paths: Vec<PathBuf> = Vec::with_capacity(frames.len());
    for (i, frame) in frames.iter().enumerate() {
        let planes = frame_to_planes(frame);
        let bytes = write_scanline_image(&header, &planes).map_err(|e| {
            ExrSeqError::new(
                format!("exr_seq::write_exr_sequence @ frame {i}"),
                format!("EXR 编码失败：{e}"),
                "改用 `Compression::None`/`Zip` 重试；若仍失败，报告 pixfmt 与压缩组合",
            )
        })?;
        let index = opts.start_index.wrapping_add(i as u32);
        let path = dir.join(frame_file_name(prefix, index));
        fs::write(&path, &bytes).map_err(|e| {
            ExrSeqError::new(
                format!("exr_seq::write_exr_sequence @ {}", path.display()),
                format!("写入 {} 字节失败：{e}", bytes.len()),
                "检查磁盘空间与目录写权限",
            )
        })?;
        paths.push(path);
    }
    Ok(paths)
}

/// 读回一帧 EXR（[`prism_io::read_image`] 的薄封装，统一成本模块的错误类型）。
pub fn read_exr_frame(path: &Path) -> Result<ExrImage, ExrSeqError> {
    let bytes = fs::read(path).map_err(|e| {
        ExrSeqError::new(
            format!("exr_seq::read_exr_frame @ {}", path.display()),
            format!("读取失败：{e}"),
            "确认文件存在且路径正确",
        )
    })?;
    prism_io::read_image(&bytes).map_err(|e| {
        ExrSeqError::new(
            format!("exr_seq::read_exr_frame @ {}", path.display()),
            format!("EXR 解码失败：{e}"),
            "确认该文件由 EXR 写入器（或官方 OpenEXR）产出，且未被截断",
        )
    })
}

/// 逐通道取回 `u8`：`(value * 255).round().clamp(0, 255)`。
///
/// 写入路径的逆运算（写 `v/255`、读 `round(x*255)`），故无损压缩下严格往返。
pub fn plane_value_to_u8(v: f32) -> u8 {
    let scaled = v * 255.0;
    let rounded = if scaled.is_nan() { 0.0 } else { scaled.round() };
    rounded.clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造确定性的 RGBA8 测试帧（含 0 / 255 端点，覆盖量化边界）。
    fn make_frame(w: u32, h: u32, seed: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = (x * 37 + y * 53 + seed * 11) % 256;
                fb.pixels[i] = v as u8;
                fb.pixels[i + 1] = ((v as u32 * 3) % 256) as u8;
                fb.pixels[i + 2] = if x == 0 { 0 } else { 255 - v as u8 };
                fb.pixels[i + 3] = 255;
            }
        }
        fb
    }

    /// 唯一临时目录（进程号 + 用例名），避免并行测试互相踩踏。
    fn temp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("prism-exr-seq-{}-{tag}", std::process::id()));
        p
    }

    #[test]
    fn exr_sequence_roundtrip_is_lossless_per_channel() {
        // 奇数尺寸（17×9）刻意覆盖非 16/8 对齐的扫描线尾巴。
        let frames = vec![make_frame(17, 9, 0), make_frame(17, 9, 1), make_frame(17, 9, 2)];
        for compression in [Compression::None, Compression::Zip, Compression::Zips, Compression::Rle]
        {
            let dir = temp_dir(&format!("lossless-{}-{:?}", std::process::id(), compression));
            let _ = fs::remove_dir_all(&dir);
            let opts = ExrSeqOptions { compression, start_index: 3 };
            let paths = write_exr_sequence(&dir, "frame", &frames, opts).expect("写序列失败");
            assert_eq!(paths.len(), frames.len(), "写出的帧数应一致");
            // 命名：起始索引 3 起，补零到 4 位。
            assert_eq!(paths[0].file_name().unwrap(), "frame.0003.exr");
            assert_eq!(paths[2].file_name().unwrap(), "frame.0005.exr");

            for (i, path) in paths.iter().enumerate() {
                let img = read_exr_frame(path).expect("读回失败");
                assert_eq!(img.planes.len(), 4, "应读回 4 个通道");
                assert_eq!(
                    img.planes.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
                    vec!["R", "G", "B", "A"],
                    "通道顺序必须为 R,G,B,A"
                );
                for (c, plane) in img.planes.iter().enumerate() {
                    assert_eq!((plane.width, plane.height), (17, 9));
                    assert_eq!(plane.pixel_type, PixelType::Float);
                    let n = (17 * 9) as usize;
                    assert_eq!(plane.values.len(), n);
                    let mut max_err = 0i32;
                    for k in 0..n {
                        let got = plane_value_to_u8(plane.values[k]);
                        let want = frames[i].pixels[k * 4 + c];
                        max_err = max_err.max((got as i32 - want as i32).abs());
                        assert_eq!(
                            got, want,
                            "{compression:?} 第 {i} 帧通道 {c} 像素 {k}：{got} != {want}"
                        );
                    }
                    assert_eq!(max_err, 0, "逐通道最大误差必须为 0");
                }
            }
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn exr_sequence_volume_reference_320x180() {
        // 交付矩阵「体积」列的口径实测：320×180 单帧、4 个 Float 通道。
        let fb = make_frame(320, 180, 5);
        let dir = temp_dir("volume");
        let _ = fs::remove_dir_all(&dir);
        for compression in [Compression::None, Compression::Zip] {
            let opts = ExrSeqOptions { compression, start_index: 0 };
            let paths = write_exr_sequence(&dir, "v", std::slice::from_ref(&fb), opts).unwrap();
            let len = fs::metadata(&paths[0]).unwrap().len();
            eprintln!(
                "[exr-volume] {compression:?}: {len} B / 帧（320×180×4ch×4B 原始 = {} B）",
                320 * 180 * 4 * 4
            );
            assert!(len > 0);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exr_sequence_rejects_mixed_resolutions() {
        let frames = vec![make_frame(8, 8, 0), make_frame(8, 4, 1)];
        let dir = temp_dir("mixed");
        let err = write_exr_sequence(&dir, "f", &frames, ExrSeqOptions::default()).unwrap_err();
        assert!(err.problem.contains("不一致"), "应报尺寸不一致：{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exr_sequence_rejects_empty_and_truncated() {
        let dir = temp_dir("empty");
        let err = write_exr_sequence(&dir, "f", &[], ExrSeqOptions::default()).unwrap_err();
        assert!(err.problem.contains("为空"), "空序列应明确报错：{err}");

        // 像素长度与宽高不符 → 明确报错（不静默写半帧）。
        let mut fb = Framebuffer::new(4, 4);
        fb.pixels.truncate(10);
        let err = write_exr_sequence(&dir, "f", &[fb], ExrSeqOptions::default()).unwrap_err();
        assert!(err.problem.contains("像素长度"), "截断帧应明确报错：{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn frame_to_planes_is_exact_u8_over_255() {
        let fb = make_frame(5, 3, 7);
        let planes = frame_to_planes(&fb);
        assert_eq!(planes.len(), 4);
        for (c, plane) in planes.iter().enumerate() {
            for k in 0..(5 * 3) {
                let v = fb.pixels[k * 4 + c];
                assert_eq!(plane.values[k], v as f32 / 255.0, "u8→f32 必须精确");
                assert_eq!(plane_value_to_u8(plane.values[k]), v, "f32→u8 必须精确");
            }
        }
    }
}
