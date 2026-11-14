//! V3.1 各档 golden 资产的逐帧生成 / 加载 / 摘要 / `manifest.sha256` 校验。
//!
//! # golden 定义（设计 §1.4 / §7）
//! **未启用累积 / 降噪 / 自适应**的 `GOLDEN_SPP = 16`、`GOLDEN_BOUNCES = 6`、
//! 确定性 CPU 路径（`prism_rt::integrator::render_pixel` + `EvalCtx::new(0,0,frame)`）
//! → ACES → sRGB OETF → RGBA8 → 自写 PNG 编码。全链路禁 libm（sRGB 的幂
//! 走 `prism_math::det_pow::det_pow`），保证跨平台逐字节可复现。
//!
//! # 资产布局
//! ```text
//! docs/v3.1-golden/
//!   manifest.sha256                     # "<sha256hex>  <相对路径>" 逐行，路径升序
//!   320x180_preview/frame_0000.png ...  # 各档自比，禁互比、禁自比
//! ```
//! 本批只生成 [`GoldenTier::minimal`] 一个最小可验证档位（T03 后补齐全档位）。
//!
//! # 缺失即硬失败（P0-6）
//! [`GoldenStore::verify_manifest`] 对 manifest 缺失 / 空 / 行格式非法 / 文件
//! 缺失 / 哈希不符一律返回结构化 [`GoldenError`]，**不静默降级**，CI 硬失败。

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use prism_core::node::EvalCtx;
use prism_gpu::Framebuffer;
use prism_math::Vec3;
use prism_math::aces::aces_fitted;
use prism_math::det_pow::det_pow;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    Aabb, Cylinder, Light, MaterialKind, RenderParams, Scene, Sphere, Torus, render_pixel,
};

use crate::v31_config::{GOLDEN_BOUNCES, GOLDEN_SPP, QualityTier, ResTier};

/// golden 资产根目录（相对仓库根；由 `CARGO_MANIFEST_DIR` 编译期定位）。
pub const GOLDEN_ROOT_REL: &str = "docs/v3.1-golden";

/// SHA-256 摘要转小写十六进制（自写，零第三方依赖）。
pub fn sha256_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest: [u8; 32] = prism_io::sha256::sha256(data);
    let mut out: String = String::with_capacity(64);
    for &b in &digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// golden 资产的结构化错误（CI 硬失败时打印三段式定位信息）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoldenError {
    /// 文件 / 目录 IO 失败。
    Io { path: String, detail: String },
    /// PNG 解码失败。
    Decode { path: String, detail: String },
    /// 必需资产缺失（P0-6，不可静默关）。
    Missing { path: String },
    /// 哈希或字节数不符。
    Mismatch { path: String, expected: String, got: String },
    /// manifest 行格式非法。
    InvalidManifest { line: usize, detail: String },
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoldenError::Io { path, detail } => {
                write!(f, "V3.1 golden IO 失败：{path}（{detail}）")
            }
            GoldenError::Decode { path, detail } => {
                write!(f, "V3.1 golden PNG 解码失败：{path}（{detail}）")
            }
            GoldenError::Missing { path } => {
                write!(f, "V3.1 golden 资产缺失：{path}；请先运行 v31_gen_golden 生成（P0-6 不可静默关）")
            }
            GoldenError::Mismatch { path, expected, got } => {
                write!(f, "V3.1 golden 校验不符：{path}，期望 {expected}，实得 {got}")
            }
            GoldenError::InvalidManifest { line, detail } => {
                write!(f, "V3.1 manifest.sha256 第 {line} 行非法：{detail}")
            }
        }
    }
}

impl Error for GoldenError {}

/// 一个 golden 档位（分辨率 × 质量 × 帧数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldenTier {
    pub res: ResTier,
    pub quality: QualityTier,
    pub frames: u32,
}

impl GoldenTier {
    /// 本批最小可验证档位：320×180 / preview / 3 帧。
    pub fn minimal() -> Self {
        GoldenTier {
            res: ResTier::R320x180,
            quality: QualityTier::Preview,
            frames: 3,
        }
    }

    /// 目录名 `<res>_<quality>`。
    pub fn tag(&self) -> String {
        format!("{}_{}", self.res.tag(), self.quality.tag())
    }
}

/// manifest 的一行记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    pub sha256_hex: String,
    /// 相对 [`GOLDEN_ROOT_REL`] 的 POSIX 路径。
    pub rel_path: String,
    pub bytes: u64,
}

/// 单帧摘要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenFrameRecord {
    pub frame: u32,
    pub sha256_hex: String,
    pub bytes: u64,
}

/// 一个档位的摘要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenSummary {
    pub tier_tag: String,
    pub res: ResTier,
    pub quality: QualityTier,
    pub width: u32,
    pub height: u32,
    pub frames: Vec<GoldenFrameRecord>,
}

/// golden 资产仓库（路径策略单点持有，不引入全局可变状态）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenStore {
    root: PathBuf,
}

impl GoldenStore {
    /// 定位到源码树内的 `docs/v3.1-golden/`（编译期绝对路径 + 相对上溯）。
    pub fn at_repo_root() -> Self {
        let manifest_dir: &str = env!("CARGO_MANIFEST_DIR");
        GoldenStore {
            root: PathBuf::from(manifest_dir).join("../..").join(GOLDEN_ROOT_REL),
        }
    }

    /// 指定根目录（CLI / 测试可覆写）。
    pub fn with_root(root: impl AsRef<Path>) -> Self {
        GoldenStore {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tier_dir(&self, res: ResTier, quality: QualityTier) -> PathBuf {
        self.root.join(format!("{}_{}", res.tag(), quality.tag()))
    }

    pub fn frame_path(&self, res: ResTier, quality: QualityTier, frame: u32) -> PathBuf {
        self.tier_dir(res, quality)
            .join(format!("frame_{frame:04}.png"))
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.sha256")
    }

    /// RGBA8 → PNG 字节（`prism_io::encode_png`，存储型 deflate，确定性）。
    fn encode_frame(width: u32, height: u32, rgba: &[[u8; 4]]) -> Vec<u8> {
        let mut fb: Framebuffer = Framebuffer::new(width, height);
        assert!(
            fb.pixels.len() >= rgba.len() * 4,
            "Framebuffer 尺寸不足以承载 golden 帧"
        );
        for (i, p) in rgba.iter().enumerate() {
            fb.pixels[i * 4..i * 4 + 4].copy_from_slice(p);
        }
        prism_io::encode_png(&fb)
    }

    /// 写入一帧并落盘（返回 PNG 字节，供摘要计哈希）。
    pub fn write_frame(
        &self,
        tier: &GoldenTier,
        frame: u32,
        rgba: &[[u8; 4]],
    ) -> Result<Vec<u8>, GoldenError> {
        let (w, h) = tier.res.dims();
        assert_eq!(rgba.len(), (w as usize) * (h as usize), "golden 帧长与档位尺寸不符");
        fs::create_dir_all(self.tier_dir(tier.res, tier.quality)).map_err(|e| GoldenError::Io {
            path: self.tier_dir(tier.res, tier.quality).display().to_string(),
            detail: e.to_string(),
        })?;
        let bytes: Vec<u8> = Self::encode_frame(w, h, rgba);
        let path: PathBuf = self.frame_path(tier.res, tier.quality, frame);
        fs::write(&path, &bytes).map_err(|e| GoldenError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        Ok(bytes)
    }

    /// 从磁盘加载一帧（PNG 解码 → RGBA8）。缺失即 [`GoldenError::Missing`]。
    pub fn load_frame(
        &self,
        res: ResTier,
        quality: QualityTier,
        frame: u32,
    ) -> Result<Vec<[u8; 4]>, GoldenError> {
        let path: PathBuf = self.frame_path(res, quality, frame);
        let bytes: Vec<u8> = fs::read(&path).map_err(|e| GoldenError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let img = prism_io::decode_png(&bytes).map_err(|e| GoldenError::Decode {
            path: path.display().to_string(),
            detail: e,
        })?;
        let raw: Vec<u8> = img.to_rgba8();
        let mut out: Vec<[u8; 4]> = Vec::with_capacity(raw.len() / 4);
        for px in raw.chunks_exact(4) {
            out.push([px[0], px[1], px[2], px[3]]);
        }
        Ok(out)
    }

    /// 逐帧渲染 + 落盘一个档位（全档位 golden 在 T03 后由同一入口补齐）。
    pub fn generate_tier(&self, tier: &GoldenTier) -> Result<GoldenSummary, GoldenError> {
        fs::create_dir_all(&self.root).map_err(|e| GoldenError::Io {
            path: self.root.display().to_string(),
            detail: e.to_string(),
        })?;
        let (w, h) = tier.res.dims();
        let mut records: Vec<GoldenFrameRecord> = Vec::with_capacity(tier.frames as usize);
        for frame in 0..tier.frames {
            let rgba: Vec<[u8; 4]> = render_golden_frame(tier, frame);
            let bytes: Vec<u8> = self.write_frame(tier, frame, &rgba)?;
            records.push(GoldenFrameRecord {
                frame,
                sha256_hex: sha256_hex(&bytes),
                bytes: bytes.len() as u64,
            });
        }
        Ok(GoldenSummary {
            tier_tag: tier.tag(),
            res: tier.res,
            quality: tier.quality,
            width: w,
            height: h,
            frames: records,
        })
    }

    /// 按磁盘上该档位的 PNG 生成摘要（不重渲染）。
    pub fn summary(&self, tier: &GoldenTier) -> Result<GoldenSummary, GoldenError> {
        let (w, h) = tier.res.dims();
        let mut records: Vec<GoldenFrameRecord> = Vec::with_capacity(tier.frames as usize);
        for frame in 0..tier.frames {
            let path: PathBuf = self.frame_path(tier.res, tier.quality, frame);
            let bytes: Vec<u8> = fs::read(&path).map_err(|_| GoldenError::Missing {
                path: path.display().to_string(),
            })?;
            records.push(GoldenFrameRecord {
                frame,
                sha256_hex: sha256_hex(&bytes),
                bytes: bytes.len() as u64,
            });
        }
        Ok(GoldenSummary {
            tier_tag: tier.tag(),
            res: tier.res,
            quality: tier.quality,
            width: w,
            height: h,
            frames: records,
        })
    }

    /// 扫描根目录全部 `frame_*.png`，按相对路径升序重写 `manifest.sha256`。
    ///
    /// 每行格式 `"<64 位小写 hex>  <相对路径>"`，与 `sha256sum -c` 兼容。
    pub fn regenerate_manifest(&self) -> Result<Vec<ManifestEntry>, GoldenError> {
        fs::create_dir_all(&self.root).map_err(|e| GoldenError::Io {
            path: self.root.display().to_string(),
            detail: e.to_string(),
        })?;
        let mut rels: Vec<String> = Vec::new();
        let dirs = fs::read_dir(&self.root).map_err(|e| GoldenError::Io {
            path: self.root.display().to_string(),
            detail: e.to_string(),
        })?;
        for d in dirs {
            let d = d.map_err(|e| GoldenError::Io {
                path: self.root.display().to_string(),
                detail: e.to_string(),
            })?;
            if !d.path().is_dir() {
                continue;
            }
            let files = fs::read_dir(d.path()).map_err(|e| GoldenError::Io {
                path: d.path().display().to_string(),
                detail: e.to_string(),
            })?;
            for f in files {
                let f = f.map_err(|e| GoldenError::Io {
                    path: d.path().display().to_string(),
                    detail: e.to_string(),
                })?;
                let p: PathBuf = f.path();
                let name: String = match p.file_name().and_then(|n| n.to_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                if p.is_file() && name.starts_with("frame_") && name.ends_with(".png") {
                    let rel: String = p
                        .strip_prefix(&self.root)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .replace('\\', "/");
                    rels.push(rel);
                }
            }
        }
        rels.sort(); // 确定性：路径字典序
        let mut entries: Vec<ManifestEntry> = Vec::with_capacity(rels.len());
        for rel in rels {
            let bytes: Vec<u8> = fs::read(self.root.join(&rel)).map_err(|_| GoldenError::Missing {
                path: rel.clone(),
            })?;
            entries.push(ManifestEntry {
                sha256_hex: sha256_hex(&bytes),
                rel_path: rel,
                bytes: bytes.len() as u64,
            });
        }
        let mut text: String = String::with_capacity(entries.len() * 80);
        for e in &entries {
            text.push_str(&e.sha256_hex);
            text.push_str("  ");
            text.push_str(&e.rel_path);
            text.push('\n');
        }
        fs::write(self.manifest_path(), text).map_err(|e| GoldenError::Io {
            path: self.manifest_path().display().to_string(),
            detail: e.to_string(),
        })?;
        Ok(entries)
    }

    /// 校验 `manifest.sha256`：格式、资产存在性、逐文件 SHA-256。
    ///
    /// 任何一项不满足即返回结构化错误（P0-6，CI 硬失败，不静默关）。
    pub fn verify_manifest(&self) -> Result<Vec<ManifestEntry>, GoldenError> {
        let path: PathBuf = self.manifest_path();
        let text: String = fs::read_to_string(&path).map_err(|_| GoldenError::Missing {
            path: path.display().to_string(),
        })?;
        let mut entries: Vec<ManifestEntry> = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            // V3.1 T03：manifest 末尾的 `# v31.sample_frames ...` 注释行承载
            // golden 抽样索引（主理人裁定 1）。`sha256sum -c` 本就忽略 `#`
            // 注释行；此处同样跳过，保持逐条哈希校验语义完全不变。
            if line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let hex: &str = parts.next().ok_or(GoldenError::InvalidManifest {
                line: idx + 1,
                detail: "缺少 SHA-256 字段".to_string(),
            })?;
            let rel: &str = parts.next().ok_or(GoldenError::InvalidManifest {
                line: idx + 1,
                detail: "缺少相对路径字段".to_string(),
            })?;
            if parts.next().is_some() {
                return Err(GoldenError::InvalidManifest {
                    line: idx + 1,
                    detail: "多余字段（路径不得含空格）".to_string(),
                });
            }
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(GoldenError::InvalidManifest {
                    line: idx + 1,
                    detail: format!("SHA-256 必须是 64 位十六进制，实得 {hex}"),
                });
            }
            entries.push(ManifestEntry {
                sha256_hex: hex.to_string(),
                rel_path: rel.to_string(),
                bytes: 0,
            });
        }
        if entries.is_empty() {
            return Err(GoldenError::InvalidManifest {
                line: 0,
                detail: "manifest 为空（golden 资产缺失，不可静默通过）".to_string(),
            });
        }
        for e in &mut entries {
            let bytes: Vec<u8> =
                fs::read(self.root.join(&e.rel_path)).map_err(|_| GoldenError::Missing {
                    path: e.rel_path.clone(),
                })?;
            e.bytes = bytes.len() as u64;
            let got: String = sha256_hex(&bytes);
            if got != e.sha256_hex {
                return Err(GoldenError::Mismatch {
                    path: e.rel_path.clone(),
                    expected: e.sha256_hex.clone(),
                    got,
                });
            }
        }
        Ok(entries)
    }
}

// ===========================================================================
// golden 确定性 CPU 渲染（16 spp / 6 bounces / 无累积 / 无降噪 / 无自适应）
// ===========================================================================

/// 线性 → sRGB OETF（禁 libm：幂函数走 `det_pow`，跨平台逐位一致）。
#[inline]
fn srgb_oetf_det(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * det_pow(x, 1.0 / 2.4) - 0.055
    }
}

/// golden 场景：解析球 / 环 / 柱 + 地台 + 两盏球灯。
///
/// 与 `examples/gallery.rs` 刻意**不同**（禁自比）：独立的场景参数、独立的
/// 相机与帧漂移，但走同一套 `prism-rt` CPU 求交 / NEE / BSDF 内核，保证
/// golden 具备真实图像结构（强边缘、宽亮度直方图），使五判据非恒真。
fn golden_scene(frame: u32) -> Scene {
    let drift: f32 = frame as f32 * 0.02;
    let floor = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.11, 0.105, 0.10),
        0.0,
        0.22,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    let gold = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.92, 0.66, 0.24),
        1.0,
        0.28,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    let porcelain = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.88, 0.86, 0.82),
        0.0,
        0.18,
        1.5,
        0.0,
        0.10,
        Vec3::splat(0.08),
    ));
    let steel = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.60, 0.63, 0.68),
        1.0,
        0.34,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    let matte = MaterialKind::Lambert { albedo: Vec3::new(0.26, 0.24, 0.22) };

    let key_pos: Vec3 = Vec3::new(-4.2, 5.4, 2.6);
    let key_rad: Vec3 = Vec3::new(120.0, 104.0, 82.0);
    let rim_pos: Vec3 = Vec3::new(3.6, 3.8, -4.2);
    let rim_rad: Vec3 = Vec3::new(70.0, 78.0, 92.0);

    Scene {
        boxes: vec![(
            Aabb::new(Vec3::new(-20.0, -0.4, -20.0), Vec3::new(20.0, 0.0, 20.0)),
            floor,
        )],
        spheres: vec![
            (
                Sphere { center: Vec3::new(-0.80 + drift, 0.70, 0.0), radius: 0.70 },
                gold,
            ),
            (
                Sphere { center: Vec3::new(0.62, 0.46, 0.72), radius: 0.46 },
                porcelain,
            ),
            (
                Sphere { center: key_pos, radius: 0.90 },
                MaterialKind::Emissive { radiance: key_rad },
            ),
            (
                Sphere { center: rim_pos, radius: 0.60 },
                MaterialKind::Emissive { radiance: rim_rad },
            ),
        ],
        lights: vec![
            Light::Sphere { center: key_pos, radius: 0.90, radiance: key_rad },
            Light::Sphere { center: rim_pos, radius: 0.60, radiance: rim_rad },
        ],
        toruses: vec![(
            Torus { center: Vec3::new(-1.35, 0.16, 0.95), major: 0.42, minor: 0.11 },
            steel,
        )],
        cylinders: vec![(
            Cylinder {
                center: Vec3::new(1.70, 0.62, -1.35),
                radius: 0.26,
                half_height: 0.62,
            },
            matte,
        )],
        ..Default::default()
    }
}

/// golden 相机（针孔、零快门 ⇒ 无额外随机源；随帧微动产生温和帧间差异）。
fn golden_camera(frame: u32) -> Camera {
    let drift: f32 = frame as f32 * 0.01;
    Camera::new(
        Vec3::new(-0.06 + drift, 1.12, 4.35),
        Vec3::new(0.0, 0.56, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        38.0, // fov_deg（相机竖直视场角，非 PSNR 判据常量）
        0.0,
        4.30,
        0.0,
        0.0,
    )
}

/// 渲染一帧 golden（RGBA8，行主序）。
///
/// 行带并行只切分输出缓冲，每个像素仍是 `(种子, 像素, 样本, 维度)` 的独立
/// 纯函数 ⇒ 与单线程逐位一致（同源纪律见 `examples/gallery.rs`）。
pub fn render_golden_frame(tier: &GoldenTier, frame: u32) -> Vec<[u8; 4]> {
    let (w, h) = tier.res.dims();
    let w_us: usize = w as usize;
    let h_us: usize = h as usize;
    let scene: Scene = golden_scene(frame);
    let cam: Camera = golden_camera(frame);
    let params: RenderParams = RenderParams {
        clamp_indirect: Some(8.0),
        ..RenderParams::new(GOLDEN_SPP, GOLDEN_BOUNCES)
    };
    // 设计 §7：样本流经 EvalCtx::new(RT_SEED=0, 0, frame)。
    let ctx: EvalCtx = EvalCtx::new(0, 0, frame as u64);

    let mut linear: Vec<[f32; 4]> = vec![[0.0; 4]; w_us * h_us];
    let threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(h_us)
        .max(1);
    let band: usize = h_us.div_ceil(threads);
    std::thread::scope(|s| {
        for (bi, chunk) in linear.chunks_mut(band * w_us).enumerate() {
            let scene_ref: &Scene = &scene;
            let cam_ref: &Camera = &cam;
            let ctx_ref: &EvalCtx = &ctx;
            let params_ref: &RenderParams = &params;
            s.spawn(move || {
                for (ly, row) in chunk.chunks_mut(w_us).enumerate() {
                    let y: u32 = (bi * band + ly) as u32;
                    for (x, px) in row.iter_mut().enumerate() {
                        *px = render_pixel(
                            scene_ref,
                            cam_ref,
                            ctx_ref,
                            (x as u32, y),
                            (w, h),
                            params_ref,
                        );
                    }
                }
            });
        }
    });

    let mut out: Vec<[u8; 4]> = Vec::with_capacity(w_us * h_us);
    for c in &linear {
        let toned: [f32; 3] = aces_fitted([c[0], c[1], c[2]]);
        let mut px: [u8; 4] = [255u8; 4];
        for k in 0..3 {
            let v: f32 = srgb_oetf_det(toned[k].clamp(0.0, 1.0));
            px[k] = (v * 255.0 + 0.5) as u8;
        }
        out.push(px);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_fips_empty_vector() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn tier_paths_are_stable() {
        let s = GoldenStore::with_root("/tmp/v31-golden-test");
        let t = GoldenTier::minimal();
        assert_eq!(t.tag(), "320x180_preview");
        assert!(s.frame_path(t.res, t.quality, 7)
            .to_string_lossy()
            .ends_with("320x180_preview/frame_0007.png"));
        assert!(s.manifest_path().to_string_lossy().ends_with("manifest.sha256"));
    }

    #[test]
    fn missing_manifest_is_a_structured_error() {
        let s = GoldenStore::with_root("/tmp/v31-golden-definitely-absent");
        let err = s.verify_manifest().expect_err("缺失 manifest 必须判红");
        assert!(matches!(err, GoldenError::Missing { .. }));
        assert!(err.to_string().contains("不可静默关"));
    }
}
