//! V3.1「全方位性能优化」——档位 / 质量 / spp / 误差判据常量的**单一真源**。
//!
//! # 为什么集中在这里
//! 设计 §7 明确要求：判据阈值（8 / 32 / 38 dB / 0.15 / 90% / 5% / 16 spp）
//! 只允许在**一个地方**定义，其它模块一律 `use crate::v31_config::*`，禁止硬编码。
//! 2026-11-13：判据② 去掉「逐帧 ≥38 且序列均值 ≥42」双 AND，改单一
//! 「序列均值 ≥38 dB、peak 固定 255」；判据③ 由「逐 bin 相对偏差 ≤5%」
//! 改「归一化 L1 总变差 D 序列均值 ≤0.15」。六个数值上限
//! （8 / 32 / 38 dB / 0.15 / 0.90 / 5%）不动，只改度量与统计层级。
//! 本模块因此不依赖 `prism-render` 的任何其它模块（`prism_dsl` 除外，仅为
//! [`V31Config::apply_to`] 的入参类型），保证依赖方向单向、无环。
//!
//! # T03 接线约定（本批只建基础设施，不改出片主链路）
//! - `cmd_render::parse_args` 构造 [`V31Config`] 后调用 [`V31Config::apply_to`]
//!   覆写 `CompileV2Output` 的 `width/height/fps/duration`；**不进 DSL IR**。
//! - `movie::render_movie_v31(compile, device, cfg, sink)` 读本结构决定
//!   spp / 累积 / 降噪 / 线性旁路；旧 `render_movie` 恒委托 [`V31Config::legacy`]。
//! - `TemporalState` 窗口 W 直接取 `cfg.window`，`RtBackend::new(cfg.spp_per_frame,
//!   cfg.bounces)`；`min_spp` 供自适应预算下限（`PRISM_V31_ADAPTIVE=0` 关闭）。
//! - 计数器键统一 `v31.<domain>.<metric>`（域见设计 §7），由 T04/T05 写入。

use prism_dsl::CompileV2Output;

// ===========================================================================
// 判据常量（唯一定义处）
// ===========================================================================

/// 逐帧新增 spp 的自适应预算下限（任何档位不得低于此值）。
pub const MIN_SPP: u32 = 2;
/// `quality final` 的逐帧新增 spp。
pub const FINAL_SPP: u32 = 4;
/// `quality preview` 的逐帧新增 spp。
pub const PREVIEW_SPP: u32 = 2;
/// golden 的等效 spp（也是所有档位 `effective_spp()` 的目标值）。
pub const GOLDEN_SPP: u32 = 16;
/// golden 的弹射深度。
pub const GOLDEN_BOUNCES: u32 = 6;

/// 判据 ①：8-bit 单通道差分的全局硬上限。
pub const DELTA_HARD: u8 = 32;
/// 判据 ①：≥99% 像素必须满足的 max-channel delta 上限。
pub const DELTA_P99: u8 = 8;
/// 判据 ②：逐帧 PSNR 下限（dB）。
pub const PSNR_FRAME_DB: f64 = 38.0;
/// 判据 ②：序列平均 PSNR 下限（dB）。任务书去掉旧「逐帧 ≥38 且均值 ≥42」
/// 双门槛，改单一序列均值 ≥38 dB；此名保留给既有调用点，值与帧下限同源。
pub const PSNR_SEQ_DB: f64 = PSNR_FRAME_DB;
/// 判据 ③：亮度 64-bin 归一化 L1 总变差 D 的序列均值上限。
pub const HIST_BIN_REL: f64 = 0.15;
/// 判据 ④：3×3 梯度强边缘幅值保留率下限。
pub const EDGE_KEEP: f64 = 0.90;
/// 判据 ⑤：相邻帧差分异常像素占比上限（超限触发 `TemporalState::reset`）。
pub const TEMPORAL_OUTLIER: f64 = 0.05;

// ---------------------------------------------------------------------------
// §8.1 分档画质上限（2026-11-14 定版：一套尺子量到底的老口径作废）
// ---------------------------------------------------------------------------
//
// 任务书 §8.1 把画质从「38 dB 神眼标准 + 逐帧一票否决」改成**按档分级**：
// `均衡` 是「平衡档」，`快速` 明确「画质明显降一档」，两档各有一组固定上限，
// 判定仍是确定性布尔。上面那组 `DELTA_HARD / PSNR_FRAME_DB / ...` 保留为
// **最严口径**（`一致` 档自检与历史负例测试仍用它），但出片链路的达标判定
// 一律走本节的按档表——否则「均衡档 max_delta 36」会被 32 的旧硬上限误判红，
// 而 §8.1 给均衡档的上限本来就是 96。
/// 一组画质上限：判据①（p99 / 全局 max）、②（序列 PSNR）、③（L1 总变差 D）、
/// ④（边缘保留率）、⑤（时序异常占比）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JudgeLimits {
    /// 判据①：8-bit 单通道差分的全局硬上限。
    pub max_delta: u8,
    /// 判据①：≥99% 像素必须满足的 max-channel delta 上限。
    pub p99_delta: u8,
    /// 判据②：序列平均 PSNR 下限（dB，peak 固定 255）。
    pub psnr_db: f64,
    /// 判据③：亮度 64-bin 归一化 L1 总变差 D 的序列均值上限。
    pub hist_l1: f64,
    /// 判据④：3×3 梯度强边缘幅值保留率下限。
    pub edge_keep: f64,
    /// 判据⑤：相邻帧差分异常像素占比上限。
    pub temporal: f64,
}

impl JudgeLimits {
    /// `均衡`（= 平衡档）：`p99≤24 / max≤96 / ≥32 dB / D≤0.35 / 边缘≥0.75 / 时序≤15%`。
    pub const BALANCED: JudgeLimits = JudgeLimits {
        max_delta: 96,
        p99_delta: 24,
        psnr_db: 32.0,
        hist_l1: 0.35,
        edge_keep: 0.75,
        temporal: 0.15,
    };
    /// `快速`（画质明显降一档）：`p99≤40 / max≤160 / ≥28 dB / D≤0.50 / 边缘≥0.60 / 时序≤25%`。
    pub const FAST: JudgeLimits = JudgeLimits {
        max_delta: 160,
        p99_delta: 40,
        psnr_db: 28.0,
        hist_l1: 0.50,
        edge_keep: 0.60,
        temporal: 0.25,
    };
    /// 最严口径（历史 38 dB 一套）：`一致` 档自检与负例测试的基线。
    pub const STRICT: JudgeLimits = JudgeLimits {
        max_delta: DELTA_HARD,
        p99_delta: DELTA_P99,
        psnr_db: PSNR_FRAME_DB,
        hist_l1: HIST_BIN_REL,
        edge_keep: EDGE_KEEP,
        temporal: TEMPORAL_OUTLIER,
    };
}

/// 纯 CPU 单帧哈希锚（32 spp @ 1200×675，逐字节不可变）。
pub const GALLERY_SHA256: &str =
    "78366f3e78323ea260a0dfb606d4f497b06913c1b96500a5ddc1018aa9c32da8";
/// 纯 CPU 单帧字节数锚。
pub const GALLERY_BYTES: u64 = 3_240_988;

// ===========================================================================
// 派生常量（口径参数，不是判据阈值；同样集中定义以免散落）
// ===========================================================================

/// 亮度直方图 bin 数（判据 ③）。
pub const HIST_BINS: usize = 64;
/// 3×3 Sobel（L1 幅值）判定「强边缘」的最小幅值，8-bit 亮度域。
pub const STRONG_EDGE_MIN: u32 = 64;
/// `for_tier` 的默认帧率（T03 由 `--fps` 覆写）。
pub const DEFAULT_FPS: u32 = 60;
/// `for_tier` 的默认帧数（T03 由 `--frames` 覆写；3 s @ 60 fps）。
pub const DEFAULT_FRAMES: u32 = 180;

// ===========================================================================
// 档位枚举
// ===========================================================================

/// 质量档（CLI `--quality`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityTier {
    /// 预览：窗口累积 + 保守降噪，逐帧 spp 最低。
    Preview,
    /// 终片：窗口累积 + 保守降噪，逐帧 spp = 4。
    Final,
    /// 快速：映射到既有 `--fast` 平光语义，不累积、不降噪、不自适应。
    Fast,
}

impl QualityTier {
    /// 稳定标签（golden 目录名 / 计数器键后缀）。
    pub fn tag(self) -> &'static str {
        match self {
            QualityTier::Preview => "preview",
            QualityTier::Final => "final",
            QualityTier::Fast => "fast",
        }
    }

    /// CLI 解析（大小写不敏感）。
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "preview" => Some(QualityTier::Preview),
            "final" => Some(QualityTier::Final),
            "fast" => Some(QualityTier::Fast),
            _ => None,
        }
    }
}

/// 分辨率档（CLI `--res`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResTier {
    R320x180,
    R640x360,
    R720p,
    R1080p,
    R4K,
    /// 自定义 `<WxH>`。
    Custom(u32, u32),
}

impl ResTier {
    /// 实际宽高（像素）。`Custom` 的 0 分量按 1 兜底，避免退化尺寸。
    pub fn dims(self) -> (u32, u32) {
        match self {
            ResTier::R320x180 => (320, 180),
            ResTier::R640x360 => (640, 360),
            ResTier::R720p => (1280, 720),
            ResTier::R1080p => (1920, 1080),
            ResTier::R4K => (3840, 2160),
            ResTier::Custom(w, h) => (w.max(1), h.max(1)),
        }
    }

    /// 稳定标签（`docs/v3.1-golden/<res>_<quality>/` 的 `<res>` 段）。
    pub fn tag(self) -> String {
        match self {
            ResTier::R320x180 => "320x180".to_string(),
            ResTier::R640x360 => "640x360".to_string(),
            ResTier::R720p => "720p".to_string(),
            ResTier::R1080p => "1080p".to_string(),
            ResTier::R4K => "4k".to_string(),
            ResTier::Custom(w, h) => format!("{}x{}", w.max(1), h.max(1)),
        }
    }

    /// CLI 解析：固定别名 + `<WxH>` 自定义。
    pub fn parse(s: &str) -> Option<Self> {
        let lower: String = s.to_ascii_lowercase();
        match lower.as_str() {
            "320x180" => return Some(ResTier::R320x180),
            "640x360" => return Some(ResTier::R640x360),
            "720p" | "1280x720" => return Some(ResTier::R720p),
            "1080p" | "1920x1080" => return Some(ResTier::R1080p),
            "4k" | "2160p" | "3840x2160" => return Some(ResTier::R4K),
            _ => {}
        }
        let sep: usize = lower.find('x')?;
        let w: u32 = lower[..sep].parse().ok()?;
        let h: u32 = lower[sep + 1..].parse().ok()?;
        if w == 0 || h == 0 {
            return None;
        }
        Some(ResTier::Custom(w, h))
    }
}

// ===========================================================================
// V31Config
// ===========================================================================

/// V3.1 出片策略对象（设计 §3 类图的全字段，无额外字段）。
///
/// 结构保持 `Copy`，可按值在编排器 / 策略对象间传递；T03 用结构体字面量
/// 覆写单字段时不会因私有字段而编译失败。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V31Config {
    pub res: ResTier,
    pub quality: QualityTier,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames: u32,
    /// 总样本槽位数 / 等效 spp（恒 `GOLDEN_SPP = 16`；送 `RtBackend` 基线）。
    /// 四档一律 16，「档位差异只来自复用比例」⇒ 此值绝不随档位变化。
    pub spp_per_frame: u32,
    /// 逐帧**真实新采**样本数（1..=16）：时域重投影的复用比例控制量。
    /// `[0,new_spp)` 本帧新采，`[new_spp,16)` 由几何校验通过的历史填补；
    /// 总槽位恒 16（重投影组件的自有口径；V3.1 出片链路另按「真实求值的
    /// 像素条目数」计账，见 `rt_tiles::last_frame_sample_accounting`）。
    /// `一致` = 16（零复用）、`均衡` = 8、`快速` = 4、`自定义` 由
    /// `spp_per_frame` 细项覆写。
    pub new_spp: u32,
    /// 时域累积窗口 W（`TemporalState` 直接取用）。
    pub window: u32,
    pub bounces: u32,
    /// 自适应预算下限（T03 接 `adaptive_backend`）。
    pub min_spp: u32,
    pub accumulate: bool,
    pub denoise: bool,
    pub adaptive: bool,
    /// GPU 优先桥开关。本沙箱无 `/dev/dri`，默认关；T04 由 CLI/环境变量打开。
    pub gpu_prefer: bool,
    /// V3.1 收尾：用户命名的大模式（`一致`/`均衡`/`快速`/`自定义`）。
    /// 大模式 = 一组预设值；拨了任一细项即自动变 `自定义`。
    pub mode: crate::v31_modes::V31Mode,
    /// 可选误差校验（**默认关**，不进渲染帧热路径；判据本身完整保留）。
    pub verify: bool,
    /// 自动探测物理 GPU（有真 GPU 自动启用、无则回落 CPU）。
    /// 沙箱仅 lavapipe 软件 ICD ⇒ 恒回落，不进达标链、不测速。
    pub gpu_auto: bool,
}

impl V31Config {
    /// 旧入口专用配置：**所有开关全关 + 规格覆写不启用**。
    ///
    /// `width/height/fps/frames` 全 0 是「不覆写」的显式哨兵（[`Self::is_legacy`]
    /// 据此判定）。`render_movie` / `render_movie_frame` / `--fast` 走本配置，
    /// 输出与 V3.1 之前逐字节一致。
    pub fn legacy() -> Self {
        V31Config {
            res: ResTier::R720p,
            quality: QualityTier::Fast,
            width: 0,
            height: 0,
            fps: 0,
            frames: 0,
            spp_per_frame: GOLDEN_SPP,
            new_spp: GOLDEN_SPP,
            window: 1,
            bounces: GOLDEN_BOUNCES,
            min_spp: MIN_SPP,
            accumulate: false,
            denoise: false,
            adaptive: false,
            gpu_prefer: false,
            mode: crate::v31_modes::V31Mode::Consistent,
            verify: false,
            gpu_auto: false,
        }
    }

    /// 按档位构造配置。
    ///
    /// 硬约束 `effective_spp() == GOLDEN_SPP`：窗口取 `GOLDEN_SPP / spp_per_frame`，
    /// 故 final = 4×4、preview = 8×2、fast = 1×16。所有档位 bounces 与 golden
    /// 一致（6），`min_spp` 恒为 `MIN_SPP`。
    pub fn for_tier(r: ResTier, q: QualityTier) -> Self {
        let (width, height) = r.dims();
        let spp_per_frame: u32 = match q {
            QualityTier::Final => FINAL_SPP,
            QualityTier::Preview => PREVIEW_SPP,
            QualityTier::Fast => GOLDEN_SPP,
        };
        let window: u32 = GOLDEN_SPP / spp_per_frame.max(1);
        let (accumulate, denoise, adaptive) = match q {
            // `--fast` 平光语义：不累积、不降噪、不自适应（与 --adaptive 互斥不变）。
            QualityTier::Fast => (false, false, false),
            _ => (true, true, true),
        };
        V31Config {
            res: r,
            quality: q,
            width,
            height,
            fps: DEFAULT_FPS,
            frames: DEFAULT_FRAMES,
            spp_per_frame,
            new_spp: GOLDEN_SPP,
            window,
            bounces: GOLDEN_BOUNCES,
            min_spp: MIN_SPP,
            accumulate,
            denoise,
            adaptive,
            // 无物理 GPU：能力桥常亮但默认不优先，T04 由 CLI 打开。
            gpu_prefer: false,
            mode: crate::v31_modes::V31Mode::Custom,
            verify: false,
            gpu_auto: false,
        }
    }

    /// 是否为「旧入口 / 不做任何规格覆写」配置。
    ///
    /// 判据：四个可覆写规格字段全 0 且四个能力开关全关。`for_tier` 的 `dims()`
    /// 保证宽高 ≥ 1，故正常档位不会误判为 legacy。
    pub fn is_legacy(&self) -> bool {
        self.width == 0
            && self.height == 0
            && self.fps == 0
            && self.frames == 0
            && !self.accumulate
            && !self.denoise
            && !self.adaptive
            && !self.gpu_prefer
    }

    /// 覆写 DSL 编译产物的规格字段（设计 §1.2：CLI 层覆写，**不进 IR**）。
    ///
    /// legacy 配置直接返回，一个字段都不动 ⇒ 旧入口逐字节不变。
    pub fn apply_to(&self, c: &mut CompileV2Output) {
        if self.is_legacy() {
            return;
        }
        c.width = self.width;
        c.height = self.height;
        c.fps = self.fps;
        c.duration = self.duration_secs();
    }

    /// 旧「窗口累积」等效 spp = `window × spp_per_frame`。四档主链 `window ≡ 1`
    /// ⇒ 恒等于总槽位 16。`--window` 已是保留字段（不参与渲染），拨动它只让
    /// 这个报表值变化；**主链等效 spp 由 `V31RenderDevice::effective_spp()`
    /// 恒定 16 给出**（16 槽位 = 真实新采 + 几何校验复用），与 window 无关。
    pub fn effective_spp(&self) -> u32 {
        self.window.max(1) * self.spp_per_frame.max(1)
    }

    /// `duration = frames / fps`（裁定 6：3600 帧 @60 fps ⇒ 60 s）。
    pub fn duration_secs(&self) -> f32 {
        self.frames as f32 / self.fps.max(1) as f32
    }

    /// golden 资产目录名 `<res>_<quality>`（各档自比，禁互比 / 禁自比）。
    pub fn tier_dir_tag(&self) -> String {
        format!("{}_{}", self.res.tag(), self.quality.tag())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 所有档位的等效 spp 必须恒等于 golden 的 16 spp。
    #[test]
    fn every_tier_effective_spp_is_golden() {
        let resolutions = [
            ResTier::R320x180,
            ResTier::R640x360,
            ResTier::R720p,
            ResTier::R1080p,
            ResTier::R4K,
            ResTier::Custom(200, 100),
        ];
        for &r in &resolutions {
            for &q in &[QualityTier::Preview, QualityTier::Final, QualityTier::Fast] {
                let cfg = V31Config::for_tier(r, q);
                assert_eq!(cfg.effective_spp(), GOLDEN_SPP, "{r:?}/{q:?}");
                assert_eq!(cfg.bounces, GOLDEN_BOUNCES);
                assert_eq!(cfg.min_spp, MIN_SPP);
            }
        }
    }

    #[test]
    fn fast_tier_disables_every_optimization() {
        let cfg = V31Config::for_tier(ResTier::R720p, QualityTier::Fast);
        assert!(!cfg.accumulate && !cfg.denoise && !cfg.adaptive && !cfg.gpu_prefer);
        assert_eq!(cfg.spp_per_frame, GOLDEN_SPP);
        assert_eq!(cfg.window, 1);
    }

    #[test]
    fn legacy_is_all_off_and_does_not_override() {
        let cfg = V31Config::legacy();
        assert!(cfg.is_legacy());
        assert!(!cfg.accumulate && !cfg.denoise && !cfg.adaptive && !cfg.gpu_prefer);
        assert_eq!(cfg.effective_spp(), GOLDEN_SPP);
        assert!(!V31Config::for_tier(ResTier::R720p, QualityTier::Final).is_legacy());
    }

    #[test]
    fn parse_roundtrips_fixed_and_custom_res() {
        assert_eq!(ResTier::parse("720p"), Some(ResTier::R720p));
        assert_eq!(ResTier::parse("4K"), Some(ResTier::R4K));
        assert_eq!(ResTier::parse("320x180"), Some(ResTier::R320x180));
        assert_eq!(ResTier::parse("200X100"), Some(ResTier::Custom(200, 100)));
        assert_eq!(ResTier::parse("0x100"), None);
        assert_eq!(QualityTier::parse("PREVIEW"), Some(QualityTier::Preview));
        assert_eq!(QualityTier::parse("ultra"), None);
    }

    #[test]
    fn duration_follows_frames_over_fps() {
        let mut cfg = V31Config::for_tier(ResTier::R720p, QualityTier::Final);
        cfg.frames = 3600;
        cfg.fps = 60;
        assert_eq!(cfg.duration_secs(), 60.0);
    }
}
