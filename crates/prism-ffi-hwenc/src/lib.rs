//! `prism-ffi-hwenc` — 硬件视频编码器的 **FFI 探测链**（V2.0-M3）。
//!
//! # 本波次交付边界（诚实口径，四条腿缺一不可）
//!
//! | 腿 | 落地形态 |
//! |---|---|
//! | 探测链真探测 | [`crate::env_probe`]：VideoToolbox 平台判定 / NVENC 驱动文件+设备节点 / VAAPI `/dev/dri` 渲染节点，全部**真读本机** |
//! | feature gate 编译矩阵 | `videotoolbox` / `nvenc` / `vaapi` 三 feature 各自 `cargo check -p prism-ffi-hwenc --features <name>` 通过（验证记录见 `docs/v2-m3-ec1-hwenc-capability.md`） |
//! | 降级路径 | 无硬件/无绑定时**明确报错**（[`HwEncoderProbe::probe`] 返回带真实证据的 `Unsupported`；[`require_available`] 直接给 Err），绝不静默失败 |
//! | 文档 | `docs/v2-m3-ec1-hwenc-capability.md` 能力矩阵（哪平台什么状态）+ 环境受限项如实标注 |
//!
//! # 仍不做的（环境受限，如实声明）
//!
//! - **不 link 任何平台库、不声明任何 extern 函数**：`VTCompressionSessionCreate` /
//!   `cuNVENC*` / `vaInitialize` 一概没有——真机 FFI 绑定需要目标平台头文件与真机，
//!   本机（Linux 沙箱）缺 macOS、缺 `/dev/dri`、缺 NVIDIA 设备（证据见
//!   [`crate::env_probe::probe_environment`]）；
//! - 因此**没有任何编码会话被真正建立**：探测结论最好只到"环境证据在/不在"，
//!   "硬件可用且能编码"需要真机，缺硬件不伪装成已做（见文档"环境受限项"）。
//!
//! # feature 语义
//!
//! | feature | 含义 | 本机（Linux 沙箱）行为 |
//! |---|---|---|
//! | （默认，全关） | 不引入任何后端 | [`enabled_backends`] 空，[`probe_all`] 空 |
//! | `videotoolbox` | 编入 Apple VideoToolbox 探测 | 平台不符 ⇒ `Unsupported`（理由带真实平台串） |
//! | `nvenc` | 编入 NVIDIA NVENC 探测 | 无驱动文件/设备节点 ⇒ `Unsupported`（理由带真实证据） |
//! | `vaapi` | 编入 VA-API 探测 | 无 `/dev/dri` 渲染节点 ⇒ `Unsupported`（理由带真实证据） |
//!
//! # 主干零依赖不破
//!
//! 本 crate **零依赖**（`[dependencies]` 为空，仅 `std`），且**不被**任何主干
//! crate 依赖——平台差异完全隔离在本 crate 的 feature 之后。

use std::fmt;

mod env_probe;

pub use env_probe::{
    host_os, join_paths, probe_environment, probe_nvenc, probe_vaapi, videotoolbox_platform_ok,
    EnvironmentReport, NvencEvidence, VaapiEvidence, DRI_DEV_DIR, DRI_RENDER_PREFIX,
    NVIDIA_DEV_PREFIX, NVIDIA_DRIVER_VERSION_FILE,
};

/// 真机 FFI 编码绑定尚未落地的说明标记（出现在"有设备证据但无绑定"的理由里）。
pub const PENDING_BINDING_NOTE: &str =
    "真机编码 FFI 绑定未落地（无平台库 link / 无 extern 声明；缺真机启用项）";

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

    /// 全部后端（能力矩阵行序的权威集合）。
    pub const ALL: [HwBackend; 3] =
        [HwBackend::VideoToolbox, HwBackend::Nvenc, HwBackend::Vaapi];

    /// 该后端 feature 是否在本构建中开启。
    pub fn is_enabled(self) -> bool {
        match self {
            HwBackend::VideoToolbox => cfg!(feature = "videotoolbox"),
            HwBackend::Nvenc => cfg!(feature = "nvenc"),
            HwBackend::Vaapi => cfg!(feature = "vaapi"),
        }
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
    /// 不可用：**平台不符**、**无硬件证据**、**无 FFI 绑定**或**驱动拒绝**之一。
    /// 具体原因见 [`HwProbe::reason`]（本 trait 契约要求理由非空）。
    Unsupported,
    /// 探测到可用的硬件会话（需要真机 + FFI 绑定；本机不可达）。
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
    /// 理由（**必须非空**：不可用时说清真实缺口，可用时说清设备标识）。
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

    /// 是否可用。
    pub fn is_available(&self) -> bool {
        self.status.is_available()
    }
}

/// 硬件编码器探测接口。
///
/// # 契约
/// 1. [`probe`](HwEncoderProbe::probe) **不得** panic，也不得做有副作用的初始化；
/// 2. 返回的每条 [`HwProbe`] 的 [`reason`](HwProbe::reason) 必须非空——不可用时说清
///    真实缺口（哪个平台 / 哪个文件 / 哪个设备条件），可用时说清设备标识；
/// 3. 同一个实现返回的 `codec` 不得重复（[`crate::probe_all`] 据此校验）。
pub trait HwEncoderProbe {
    /// 本实现对应的后端。
    fn backend(&self) -> HwBackend;

    /// 逐格式探测（典型返回：每个 [`HwCodec`] 一条）。
    fn probe(&self) -> Vec<HwProbe>;
}

/// 通用"无绑定"探测实现：如实返回 [`ProbeStatus::Unsupported`]。
///
/// 用途：feature 未开时的占位探测与"环境证据在、但绑定未落地"的后端默认形态。
/// 理由可由调用方给定（默认点明本机探测结论与绑定缺口）。
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
        let reason: String = match self.backend {
            HwBackend::VideoToolbox => format!(
                "videotoolbox: 平台判定 = {}（当前 OS {}{}）",
                if env_probe::videotoolbox_platform_ok() { "相符" } else { "不符" },
                env_probe::host_os(),
                if env_probe::videotoolbox_platform_ok() {
                    format!("；{PENDING_BINDING_NOTE}")
                } else {
                    String::new()
                }
            ),
            HwBackend::Nvenc => {
                format!("nvenc: {}；{PENDING_BINDING_NOTE}", env_probe::probe_nvenc().summary())
            }
            HwBackend::Vaapi => {
                format!("vaapi: {}；{PENDING_BINDING_NOTE}", env_probe::probe_vaapi().summary())
            }
        };
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(self.backend, codec, reason.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// feature-gated 后端（真探测环境；FFI 编码绑定留真机）
// ---------------------------------------------------------------------------

/// Apple VideoToolbox 后端（feature `videotoolbox`）。
///
/// 真探测：平台判定。非 macOS/iOS ⇒ 明确报"平台不符"；平台相符但本波次无
/// FFI 绑定 ⇒ 明确报"绑定未落地"。
#[cfg(feature = "videotoolbox")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoToolboxProbe;

#[cfg(feature = "videotoolbox")]
impl HwEncoderProbe for VideoToolboxProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::VideoToolbox
    }

    fn probe(&self) -> Vec<HwProbe> {
        let reason: String = if env_probe::videotoolbox_platform_ok() {
            format!(
                "videotoolbox: 平台相符（OS = {}），但{PENDING_BINDING_NOTE}——未建 \
                 VTCompressionSession，无法确认真实硬件会话",
                env_probe::host_os()
            )
        } else {
            format!(
                "videotoolbox: 平台不符——当前 OS = {}，VideoToolbox 仅 macOS/iOS；\
                 本机无该平台硬件",
                env_probe::host_os()
            )
        };
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(HwBackend::VideoToolbox, codec, reason.clone()))
            .collect()
    }
}

/// NVIDIA NVENC 后端（feature `nvenc`）。
///
/// 真探测：`/proc/driver/nvidia/version` + `/dev/nvidia*` 设备节点。
#[cfg(feature = "nvenc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvencProbe;

#[cfg(feature = "nvenc")]
impl HwEncoderProbe for NvencProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::Nvenc
    }

    fn probe(&self) -> Vec<HwProbe> {
        let ev: NvencEvidence = env_probe::probe_nvenc();
        let reason: String = if ev.is_absent() {
            format!(
                "nvenc: 未发现 NVIDIA 硬件证据——{}；无驱动即无 NVENC",
                ev.summary()
            )
        } else {
            format!(
                "nvenc: 发现 NVIDIA 硬件证据（{}），但{PENDING_BINDING_NOTE}——未 dlopen \
                 libnvidia-encode，无法确认真实编码会话",
                ev.summary()
            )
        };
        HwCodec::ALL
            .iter()
            .map(|&codec| HwProbe::unsupported(HwBackend::Nvenc, codec, reason.clone()))
            .collect()
    }
}

/// VA-API 后端（feature `vaapi`）。
///
/// 真探测：`/dev/dri` 目录与 `renderD*` 渲染节点。
#[cfg(feature = "vaapi")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VaapiProbe;

#[cfg(feature = "vaapi")]
impl HwEncoderProbe for VaapiProbe {
    fn backend(&self) -> HwBackend {
        HwBackend::Vaapi
    }

    fn probe(&self) -> Vec<HwProbe> {
        let ev: VaapiEvidence = env_probe::probe_vaapi();
        let reason: String = if ev.is_absent() {
            format!(
                "vaapi: 未发现可用渲染节点——{}；无 DRM 设备即无 VA-API",
                ev.summary()
            )
        } else {
            format!(
                "vaapi: 发现渲染节点（{}），但{PENDING_BINDING_NOTE}——未 link libva / \
                 未调 vaInitialize，无法确认真实编码会话",
                ev.summary()
            )
        };
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

/// 是否存在任一可用硬件编码器（本机恒为 `false`，见模块文档的诚实边界）。
pub fn any_available() -> bool {
    probe_all().iter().any(|p| p.status.is_available())
}

// ---------------------------------------------------------------------------
// 能力矩阵（文档同源：哪平台 / 什么 feature 状态 / 什么探测结论）
// ---------------------------------------------------------------------------

/// 能力矩阵的一行（一个后端的完整画像）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// 后端。
    pub backend: HwBackend,
    /// 本构建是否编入（feature 开/关）。
    pub feature_enabled: bool,
    /// 本机环境证据摘要（真探测结论，与 feature 无关）。
    pub environment: String,
    /// 本构建的探测结论（feature 未开 = "未编入"）。
    pub status: String,
}

/// 本机 × 本构建的能力矩阵（三后端全覆盖，含未编入的）。
///
/// 这是文档《EC-1 能力矩阵》的数据源：文档表格由本函数渲染，杜绝"文档手写、
/// 代码另一套"的双源漂移。
pub fn capability_matrix() -> Vec<MatrixRow> {
    let report: EnvironmentReport = probe_environment();
    HwBackend::ALL
        .iter()
        .map(|&backend| {
            let environment: String = match backend {
                HwBackend::VideoToolbox => format!(
                    "平台判定：{}（OS = {}）",
                    if report.videotoolbox_platform { "相符" } else { "不符" },
                    report.os
                ),
                HwBackend::Nvenc => format!("nvenc 证据：{}", report.nvenc.summary()),
                HwBackend::Vaapi => format!("vaapi 证据：{}", report.vaapi.summary()),
            };
            let status: String = if !backend.is_enabled() {
                "未编入（feature 关闭）".to_string()
            } else {
                let reasons: Vec<String> = probe_all()
                    .iter()
                    .filter(|p| p.backend == backend)
                    .map(|p| p.reason.clone())
                    .collect();
                let uniq: usize = reasons.iter().collect::<std::collections::BTreeSet<_>>().len();
                if uniq <= 1 {
                    reasons.first().cloned().unwrap_or_else(|| "已编入".to_string())
                } else {
                    format!("已编入（逐格式 {} 条结论）", reasons.len())
                }
            };
            MatrixRow { backend, feature_enabled: backend.is_enabled(), environment, status }
        })
        .collect()
}

/// 能力矩阵的文本渲染（入档 / 日志用；与文档同源）。
pub fn capability_matrix_text() -> String {
    let mut out: String = String::new();
    out.push_str(&format!("平台: {}\n", env_probe::host_os()));
    for row in capability_matrix() {
        out.push_str(&format!(
            "- {} [feature {}] {}\n  结论：{}\n",
            row.backend,
            if row.feature_enabled { "开" } else { "关" },
            row.environment,
            row.status
        ));
    }
    out
}

/// **降级路径（显式报错）**：调用方需要硬件编码器时调它。
///
/// 有可用后端 ⇒ `Ok(probe)`；没有 ⇒ `Err(HwUnavailableError)`，错误文本**列出
/// 每一条探测结论与理由**（哪平台、哪文件、哪设备缺），绝不静默失败、绝不
/// 假装可用。这是"无硬件时明确报错"的入口；调用方转协议错误时可直接透传。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HwUnavailableError {
    /// 全部探测结论（含未编入后端的环境证据）。
    pub rows: Vec<MatrixRow>,
}

impl fmt::Display for HwUnavailableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "硬件编码器不可用：本机未探测到可用后端（共 {} 条探测结论）",
            self.rows.len()
        )?;
        for row in &self.rows {
            write!(
                f,
                "\n  - {}（feature {}）：{}；{}",
                row.backend,
                if row.feature_enabled { "开" } else { "关" },
                row.environment,
                row.status
            )?;
        }
        write!(
            f,
            "\n  建议：确认目标机有对应硬件（VT 需 macOS、NVENC 需 NVIDIA 驱动+设备、VAAPI 需 /dev/dri）\
             并开启 Cargo feature；无硬件时请改用软件编码路径"
        )
    }
}

impl std::error::Error for HwUnavailableError {}

/// 取第一个可用后端；没有则带全部证据显式报错（见 [`HwUnavailableError`]）。
///
/// # Errors
/// 本机构建内无可用硬件编码器 ⇒ [`HwUnavailableError`]（含逐后端的真实理由）。
pub fn require_available() -> Result<HwProbe, HwUnavailableError> {
    if let Some(p) = probe_all().into_iter().find(|p| p.is_available()) {
        return Ok(p);
    }
    Err(HwUnavailableError { rows: capability_matrix() })
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
    fn every_enabled_probe_carries_real_environment_evidence() {
        // 本轮无 FFI 绑定 ⇒ 所有编入的后端都必须如实返回 Unsupported，且理由
        // 非空、带**真实探测证据**（平台串 / 驱动文件 / 设备节点二选一）。
        let report: EnvironmentReport = probe_environment();
        for probe in probe_all() {
            assert_eq!(
                probe.status,
                ProbeStatus::Unsupported,
                "{} 的 {} 不应报告可用（本机构建无 FFI 编码绑定）",
                probe.backend,
                probe.codec
            );
            assert!(!probe.reason.is_empty(), "理由不得为空（{}）", probe.backend);
            let concrete: bool = match probe.backend {
                HwBackend::VideoToolbox => {
                    probe.reason.contains(report.os)
                }
                HwBackend::Nvenc => {
                    probe.reason.contains(NVIDIA_DRIVER_VERSION_FILE)
                        || probe.reason.contains(NVIDIA_DEV_PREFIX)
                }
                HwBackend::Vaapi => {
                    probe.reason.contains(DRI_DEV_DIR)
                }
            };
            assert!(
                concrete,
                "{} 的理由必须带上真实探测证据（平台/驱动文件/设备节点），实得：{}",
                probe.backend, probe.reason
            );
        }
        assert!(!any_available(), "本机 any_available() 必须为 false（缺硬件+绑定）");
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
        // is_enabled 与 enabled_backends 一致（静态判定不漂移）。
        for b in enabled_backends() {
            assert!(b.is_enabled(), "{b} 在 enabled_backends 里就必须 is_enabled");
        }
    }

    #[test]
    fn capability_matrix_covers_all_three_backends_with_real_rows() {
        let rows: Vec<MatrixRow> = capability_matrix();
        assert_eq!(rows.len(), HwBackend::ALL.len(), "矩阵必须三行（全覆盖）");
        for row in &rows {
            assert!(!row.environment.is_empty(), "环境列不得为空（{}）", row.backend);
            assert!(!row.status.is_empty(), "结论列不得为空（{}）", row.backend);
            // 环境列必须是真探测产物：含 OS 串或探测点路径之一。
            assert!(
                row.environment.contains(env_probe::host_os())
                    || row.environment.contains(NVIDIA_DRIVER_VERSION_FILE)
                    || row.environment.contains(DRI_DEV_DIR),
                "环境列必须是真探测结论（{}）：{}",
                row.backend,
                row.environment
            );
        }
        // 文本渲染非空且含平台行（入档用）。
        let text: String = capability_matrix_text();
        assert!(text.contains("平台:"), "矩阵文本必须带平台行：{text}");
        assert!(text.contains("videotoolbox") && text.contains("nvenc") && text.contains("vaapi"));
    }

    #[test]
    fn require_available_errors_explicitly_without_silent_success() {
        // 本机（Linux 沙箱，无 VT/NVENC/VAAPI 硬件与绑定）必须显式报错，
        // 且错误文本列出全部真实证据 + 建议段（C19）。
        match require_available() {
            Ok(p) => panic!("本机不应有可用硬件编码器，实得 {p:?}"),
            Err(e) => {
                let text: String = e.to_string();
                assert!(text.contains("硬件编码器不可用"), "首段须说清结论：{text}");
                assert!(text.contains("建议："), "必含建议段（C19）：{text}");
                assert!(text.contains("videotoolbox"), "须列出全部后端结论：{text}");
                assert!(text.contains("nvenc") && text.contains("vaapi"));
            }
        }
    }
}
