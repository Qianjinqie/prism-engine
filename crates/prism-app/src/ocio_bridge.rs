//! V2.5-B1：`prism-app` 的 OCIO 产品路径桥（零第三方依赖）。
//!
//! 职责只有三件，全部落在**既有出片链路**上，不另起渲染器：
//! 1. 读 `config.ocio`（`OcioConfig::parse`）；
//! 2. 按 config 目录解析 `FileTransform` 的 LUT，烘成 Q24 定点
//!    1D shaper + 3D cube（`OcioLut::build`）；
//! 3. 在落盘前对 RGBA8 帧应用 LUT（`prism-app render` 与 v1 CLI 共用）。
//!
//! 无 config / 空 config ⇒ `None`，**一字节都不改**（V1 逐字节兼容）。

use std::path::{Path, PathBuf};

use prism_gpu::Framebuffer;
use prism_math::ocio::{OcioConfig, OcioError, OcioResult};
use prism_math::ocio_lut::{
    apply_lut_rgba8_frames, OcioLut, DEFAULT_CUBE_SIZE, DEFAULT_SHAPER_SIZE,
};

/// 一条已装配的 OCIO 帧管线。
#[derive(Debug, Clone)]
pub struct OcioFramePipeline {
    /// `config.ocio` 路径（诊断输出用）。
    pub config_path: String,
    /// 解析后的配置（逐字段保留）。
    pub config: OcioConfig,
    /// 烘焙后的定点 LUT。
    pub lut: OcioLut,
}

impl OcioFramePipeline {
    /// 对 `prism-app render` 的 RGBA8 帧序列原地应用（产品出口）。
    pub fn apply_frames(&self, frames: &mut [Vec<u8>]) {
        apply_lut_rgba8_frames(frames, &self.lut);
    }

    /// 对 v1 CLI 的 `Framebuffer` 原地应用（产品出口）。
    pub fn apply_framebuffer(&self, fb: &mut Framebuffer) {
        self.lut.apply_rgba8(&mut fb.pixels);
    }

    /// 诊断行（打印到 CLI 汇总，证明 config 真被吃到）。
    pub fn summary(&self) -> String {
        format!(
            "{}（version {}，{} colorspaces / {} displays / {} looks，\
             cube {}³ + shaper {}，{} transforms）",
            self.config_path,
            self.config.version,
            self.config.colorspaces.len(),
            self.config.displays.len(),
            self.config.looks.len(),
            self.lut.cube.size,
            self.lut.shaper.size,
            self.lut.transform_count,
        )
    }
}

/// 由 `config.ocio` 路径装配帧管线。
///
/// - 无法解析 / LUT 尺寸错 / 未知 transform → `Err(String)` 三段式原文；
/// - **空 config**（无 colorspaces / displays / looks）→ `Ok(None)`，
///   产品路径据此完全跳过 LUT 分支，逐字节同既有 ACES 路径。
pub fn load_ocio_pipeline(path: &str) -> Result<Option<OcioFramePipeline>, String> {
    let cfg: OcioConfig = OcioConfig::from_path(path).map_err(|e| e.to_string())?;
    if cfg.is_empty() {
        return Ok(None);
    }
    let chain = cfg.view_chain(None, None, None).map_err(|e| e.to_string())?;
    let base: PathBuf = Path::new(path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let search_path: String = cfg.search_path.clone();
    let mut provider = |src: &str| -> OcioResult<Vec<u8>> {
        let candidates: Vec<PathBuf> = {
            let mut v: Vec<PathBuf> = vec![base.join(src)];
            if !search_path.is_empty() {
                v.push(base.join(&search_path).join(src));
            }
            v
        };
        for p in candidates {
            if let Ok(bytes) = std::fs::read(&p) {
                return Ok(bytes);
            }
        }
        Err(OcioError::lut_source(
            1,
            format!("找不到 `src = {src}` 指向的 LUT 文件（基准目录 {}）", base.display()),
        ))
    };
    let lut: Option<OcioLut> = OcioLut::build(
        &chain,
        &mut provider,
        DEFAULT_CUBE_SIZE,
        DEFAULT_SHAPER_SIZE,
    )
    .map_err(|e| e.to_string())?;
    Ok(lut.map(|l| OcioFramePipeline {
        config_path: path.to_string(),
        config: cfg,
        lut: l,
    }))
}
