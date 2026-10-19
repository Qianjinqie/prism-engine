//! `prism-ffi-hwenc` — 硬件视频编码器的 **FFI 探测骨架**（V2.0-K）。
//!
//! # 本轮的交付边界（如实声明，不夸大）
//!
//! 本 crate 现在只落**接口形状**：
//!
//! - [`HwCodec`] / [`HwBackend`] / [`ProbeStatus`] / [`HwProbe`] 四个数据类型；
//! - [`HwEncoderProbe`] 探测 trait（每个后端一个实现）；
//! - feature gate：`videotoolbox` / `nvenc` / `vaapi`（默认全关）；
//! - [`probe_all`] 汇总入口。
//!
//! **没有任何真机绑定**：不 `link` 任何平台库、不读任何头文件、不调用
//! `VTCompressionSessionCreate` / `cuNVENC*` / `vaInitialize`。因此**所有**探测在
//! 本轮都**如实返回 [`ProbeStatus::Unsupported`]**，并在 [`HwProbe::reason`] 里说明
//! "真机绑定留 [`PENDING_MILESTONE`]"。不做"看起来可用"的假探测（宪章：绝不静默糊弄）。
//!
//! 真机实现（V2.0-M3）要补的三件事，按后端逐个落：
//!
//! | 后端 | 需要补的东西 | 探测口径 |
//! |---|---|---|
//! | `videotoolbox` | `extern "C"` 声明 + `VTCompressionSessionCreate` 能力查询 | 是否拿到 hardware 加速的 `VTCompressionSession` |
//! | `nvenc` | `libnvidia-encode` 的 `NvEncodeAPIGetMaxSupportedVersion` 等 | `nvEncOpenEncodeSessionEx` 是否返回 SUCCESS |
//! | `vaapi` | `libva` 的 `vaInitialize` + `vaQueryConfigEntrypoints` | 是否存在 `VAEntrypointEncSlice` 的 H.264/HEVC profile |
//!
//! # 主干零依赖不破
//!
//! 本 crate **零依赖**（`[dependencies]` 为空），且**不被** `prism-mux`（或任何主干
//! crate）依赖——它只出现在需要硬件编码的下游的显式依赖里。故：主干构建的依赖图与
//! 编译时间完全不受影响，平台差异被隔离在本 crate 的 feature 之后。
//!
//! # feature 语义
//!
//! | feature | 含义 | 本轮行为 |
//! |---|---|---|
//! | （默认，全关） | 不引入任何后端 | [`enabled_backends`] 返回空，[`probe_all`] 返回空 |
//! | `videotoolbox` | 编入 Apple VideoToolbox 后端 | 返回 `Unsupported`（无绑定） |
//! | `nvenc` | 编入 NVIDIA NVENC 后端 | 返回 `Unsupported`（无绑定） |
//! | `vaapi` | 编入 VA-API 后端 | 返回 `Unsupported`（无绑定） |

use std::fmt;

/// 真机 FFI 绑定尚未落地的里程碑标记（出现在所有 `Unsupported` 理由里）。
pub const PENDING_MILESTONE: &str = "V2.0-M3";

/// 目标编码格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HwCodec {
    /// H.264 / AVC。
    H264,
    /// H.265 / HEVC。
    Hevc,
}

impl HwCodec {
    /// 全部格式（探测覆盖度的基准集合）。
    pub const ALL: [HwCodec; 2] = [HwCodec::H264, HwCodec::Hevc];

    /// 人读名（日志/矩阵文档用）。
    pub fn as_str(self) -> &'static str {
        match self {
            HwCodec::H264 => "h264",
            HwCodec::Hevc => "hevc",
        }
    }
}

impl fmt::Display for HwCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 硬件编码后端（平台/驱动家族的抽象）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HwBackend {
    /// Apple VideoToolbox（macOS / iOS）。
    VideoToolbox,
    /// NVIDIA NVENC。
    Nvenc,
    /// 通用 VA-API（Intel / AMD，Linux）。
    Vaapi,
}

impl HwBackend {
    /// 人读名（与 Cargo feature 同名，便于日志与构建参数对齐）。
    pub fn as_str(self) -> &'static str {
        match self {
            HwBackend::VideoToolbox => "videotoolbox",
            HwBackend::Nvenc => "nvenc",
            HwBackend::Vaapi => "vaapi",
        }
    }

    /// 该后端编译进来需要打开的 Cargo feature 名。
    pub fn feature_name(self) -> &'static str {
        self.as_str()
    }
}

impl fmt::Display for HwBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 探测结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    /// 不可用：**未编译该后端**、**无真机绑定**、**无对应硬件**或**驱动拒绝**之一。
    /// 具体原因见 [`HwProbe::reason`]（本 trait 契约要求理由非空）。
    Unsupported,
    /// 探测到可用的硬件会话（V2.0-M3 起才可能返回）。
    ///
    /// `device` 是后端自报的设备标识（如 `"Apple M2 / hw"`、`"NVIDIA A10G / nvenc"`）。
    Available { device: String },
}

impl ProbeStatus {
    /// 是否可用。
    pub fn is_available(&self) -> bool {
        matches!(self, ProbeStatus::Available { .. })
    }
}

impl fmt::Display for ProbeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeStatus::Unsupported => f.write_str("unsupported"),
            ProbeStatus::Available { device } => write!(f, "available({device})"),
        }
    }
}

/// 单条探测结果：某后端的某格式是否可用，以及理由。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HwProbe {
    /// 后端。
    pub backend: HwBackend,
    /// 编码格式。
    pub codec: HwCodec,
    /// 状态。
    pub status: ProbeStatus,
    /// 理由（**必须非空**：可用时说清设备，不可用时说清缺口）。
    pub reason: String,
}

impl HwProbe {
    /// 构造一条"未支持"探测（理由由调用方给出，契约要求非空）。
    pub fn unsupported(
        backend: HwBackend,
        codec: HwCodec,
        reason: impl Into<String>,
    ) -> Self {
        HwProbe {
            backend,
            codec,
            status: ProbeStatus::Unsupported,
            reason: reason.into(),
        }
    }
}

/// 硬件编码器探测接口。
///
/// # 契约
/// 1. [`probe`](HwEncoderProbe::probe) **不得** panic，也不得做有副作用的初始化；
/// 2. 返回的每条 [`HwProbe`] 的 [`reason`](HwProbe::reason) 必须非空——不可用时说清
///    缺口（哪个 feature / 哪个里程碑 / 哪个设备条件），可用时说清设备标识；
/// 3. 同一个实现返回的 `codec` 不得重复（[`crate::probe_all`] 据此校验）。
pub trait HwEncoderProbe {
    /// 本实现对应的后端。
    fn backend(&self) -> HwBackend;

    /// 逐格式探测（典型返回：每个 [`HwCodec`] 一条）。
    fn probe(&self) -> Vec<HwProbe>;
}

/// 通用"无绑定"探测实现：如实返回 [`ProbeStatus::Unsupported`]。
///
/// 本轮三个后端都用它（差别只在 [`HwBackend`] 与理由文本）。M3 落真机绑定时，
/// 各后端改用自己的结构体实现本 trait，调用方代码（[`probe_all`]）不需要改形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedProbe {
    backend: HwBackend,
}

impl UnsupportedProbe {
    /// 为指定后端构造。
    pub fn new(backend: HwBackend) -> Self {
        UnsupportedProbe { backend }
    }
}

impl HwEncoderProbe for UnsupportedProbe {
    fn backend(&self) -> HwBackend {
        self.backend
    }

    fn probe(&self) -> Vec<HwProbe> {
        let reason = format!(
            "{}: 本构建未做平台探测——真机 FFI 绑定留 {}（当前只有探测骨架，无 extern 声明、\
             不 link 平台库）",
            self.backend.as_str(),
            PENDING_MILESTONE
        );
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(self.backend, codec, reason.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// feature-gated 后端（本轮均为无绑定骨架）
// ---------------------------------------------------------------------------

/// Apple VideoToolbox 后端（feature `videotoolbox`）。
///
/// ⚠️ 本轮**只做探测骨架**：不 `link` VideoToolbox.framework、不声明
/// `VTCompressionSessionCreate`，故 [`probe`](HwEncoderProbe::probe) 如实返回
/// `Unsupported`。真实能力查询留 [`PENDING_MILESTONE`]。
#[cfg(feature = "videotoolbox")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoToolboxProbe;

#[cfg(feature = "videotoolbox")]
impl HwEncoderProbe for VideoToolboxProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::VideoToolbox
    }

    fn probe(&self) -> Vec<HwProbe> {
        // M3 待补：VTCompressionSessionCreate(kVTVideoEncoderSpecification_
        // EnableHardwareAcceleratedVideoEncoder) → 是否拿到 hardware 会话。
        let reason = format!(
            "videotoolbox: feature 已开启，但真机绑定未落地（无 VTCompressionSessionCreate \
             探测）——留 {}",
            PENDING_MILESTONE
        );
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(HwBackend::VideoToolbox, codec, reason.clone()))
            .collect()
    }
}

/// NVIDIA NVENC 后端（feature `nvenc`）。
///
/// ⚠️ 本轮**只做探测骨架**：不 `dlopen` `libnvidia-encode`、不声明
/// `NvEncodeAPIGetMaxSupportedVersion`，故如实返回 `Unsupported`。
#[cfg(feature = "nvenc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvencProbe;

#[cfg(feature = "nvenc")]
impl HwEncoderProbe for NvencProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::Nvenc
    }

    fn probe(&self) -> Vec<HwProbe> {
        // M3 待补：dlopen libnvidia-encode → NvEncodeAPIGetMaxSupportedVersion →
        // nvEncOpenEncodeSessionEx 是否 SUCCESS。
        let reason = format!(
            "nvenc: feature 已开启，但真机绑定未落地（未 dlopen libnvidia-encode）——留 {}",
            PENDING_MILESTONE
        );
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(HwBackend::Nvenc, codec, reason.clone()))
            .collect()
    }
}

/// VA-API 后端（feature `vaapi`）。
///
/// ⚠️ 本轮**只做探测骨架**：不 `link` `libva`、不声明 `vaInitialize`，故如实返回
/// `Unsupported`。
#[cfg(feature = "vaapi")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VaapiProbe;

#[cfg(feature = "vaapi")]
impl HwEncoderProbe for VaapiProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::Vaapi
    }

    fn probe(&self) -> Vec<HwProbe> {
        // M3 待补：vaInitialize → vaQueryConfigEntrypoints 是否存在
        // VAEntrypointEncSlice 的 H.264/HEVC profile。
        let reason = format!(
            "vaapi: feature 已开启，但真机绑定未落地（未 link libva / 未调用 vaInitialize）——\
             留 {}",
            PENDING_MILESTONE
        );
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(HwBackend::Vaapi, codec, reason.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 静态汇入（D-27）：探测集合在此处集中登记，不做运行期注册/反射
// ---------------------------------------------------------------------------

/// 本构建**编入**的后端列表（按 feature 静态决定，默认空）。
pub fn enabled_backends() -> Vec<HwBackend> {
    #[allow(unused_mut)]
    let mut out: Vec<HwBackend> = Vec::new();
    #[cfg(feature = "videotoolbox")]
    out.push(HwBackend::VideoToolbox);
    #[cfg(feature = "nvenc")]
    out.push(HwBackend::Nvenc);
    #[cfg(feature = "vaapi")]
    out.push(HwBackend::Vaapi);
    out
}

/// 构造本构建内所有后端探测器的集合（默认空；按 feature 追加）。
pub fn probes() -> Vec<Box<dyn HwEncoderProbe>> {
    #[allow(unused_mut)]
    let mut out: Vec<Box<dyn HwEncoderProbe>> = Vec::new();
    #[cfg(feature = "videotoolbox")]
    out.push(Box::new(VideoToolboxProbe));
    #[cfg(feature = "nvenc")]
    out.push(Box::new(NvencProbe));
    #[cfg(feature = "vaapi")]
    out.push(Box::new(VaapiProbe));
    out
}

/// 汇总所有后端的探测结果（逐后端 × 逐格式展开）。
///
/// 默认构建（无任何后端 feature）返回空 `Vec`——**这是正确结果**，不是失败：
/// 没有编入任何后端就没有可探测项，调用方应把"空"理解为"未启用硬件编码"。
pub fn probe_all() -> Vec<HwProbe> {
    let mut out: Vec<HwProbe> = Vec::new();
    for p in probes() {
        out.extend(p.probe());
    }
    out
}

/// 是否存在任一可用硬件编码器（本轮恒为 `false`）。
pub fn any_available() -> bool {
    probe_all().iter().any(|p| p.status.is_available())
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_build_has_no_backends() {
        // 默认 feature = []：不编入任何后端。
        if enabled_backends().is_empty() {
            assert!(probes().is_empty(), "无后端时探测器集合必须为空");
            assert!(probe_all().is_empty(), "无后端时探测结果必须为空");
            assert!(!any_available(), "无后端时不可能有可用编码器");
        } else {
            // 带 feature 跑测试时，编入的后端数量必须与 enabled_backends 一致。
            assert_eq!(probes().len(), enabled_backends().len());
        }
    }

    #[test]
    fn every_enabled_probe_is_honestly_unsupported() {
        // 本轮无真机绑定 ⇒ 所有编入的后端都必须如实返回 Unsupported，且理由非空、
        // 指明待办里程碑（不允许"看起来可用"的假探测）。
        for probe in probe_all() {
            assert_eq!(
                probe.status,
                ProbeStatus::Unsupported,
                "{} 的 {} 不应报告可用（本轮无 FFI 绑定）",
                probe.backend,
                probe.codec
            );
            assert!(!probe.reason.is_empty(), "理由不得为空（{}）", probe.backend);
            assert!(
                probe.reason.contains(PENDING_MILESTONE),
                "理由应指明真机绑定留 {}：{}",
                PENDING_MILESTONE,
                probe.reason
            );
        }
        assert!(!any_available(), "本轮 any_available() 必须为 false");
    }

    #[test]
    fn probes_cover_both_codecs_per_backend_without_duplicates() {
        let all = probe_all();
        for backend in enabled_backends() {
            let mut codecs: Vec<HwCodec> =
                all.iter().filter(|p| p.backend == backend).map(|p| p.codec).collect();
            codecs.sort();
            let mut dedup = codecs.clone();
            dedup.dedup();
            assert_eq!(codecs, dedup, "后端 {backend} 的探测结果存在重复格式");
            assert_eq!(
                codecs,
                HwCodec::ALL.to_vec(),
                "后端 {backend} 应覆盖 H264 与 HEVC"
            );
        }
    }

    #[test]
    fn unsupported_probe_is_deterministic() {
        // 同输入两跑逐字节相同（无时间戳/无随机），便于日志比对。
        let a = UnsupportedProbe::new(HwBackend::Vaapi).probe();
        let b = UnsupportedProbe::new(HwBackend::Vaapi).probe();
        assert_eq!(a, b, "探测结果必须确定性");
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].backend, HwBackend::Vaapi);
        assert!(a[0].reason.contains("vaapi"));
    }

    #[test]
    fn feature_names_match_backend_names() {
        // feature 名与后端名同源，避免"文档里写 videotoolbox、构建参数写 vt"的漂移。
        assert_eq!(HwBackend::VideoToolbox.feature_name(), "videotoolbox");
        assert_eq!(HwBackend::Nvenc.feature_name(), "nvenc");
        assert_eq!(HwBackend::Vaapi.feature_name(), "vaapi");
        assert_eq!(HwCodec::H264.as_str(), "h264");
        assert_eq!(HwCodec::Hevc.as_str(), "hevc");
    }
}
