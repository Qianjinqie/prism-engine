//! `prism-ffi-hwenc` — 硬件视频编码器的 **探测链 + VA-API 绑定层骨架**（零第三方依赖）。
//!
//! # 交付成熟度**四档**（分述口径，禁止混用、禁止越级声称）
//!
//! | 档 | 名称（[`Maturity`]） | 判定标准 | 当前状态 |
//! |---|---|---|---|
//! | 一 | `probe-chain` | 真读平台证据（OS 判定 / 驱动文件 / 设备节点），**不碰任何库里符号** | ✅ 三后端都到 |
//! | 二 | `binding-ready` | `dlopen` 成功 + 核心符号全部 `dlsym` 进函数指针表，**一次都不调用** | ✅ **仅 VA-API**（`vaapi_bind` 模块，`--features vaapi`） |
//! | 三 | `session-established` | 取得 `VADisplay` 并 `vaInitialize` / `vaCreateConfig` / `vaCreateContext` 成功 | ❌ 未做（三后端都未做） |
//! | 四 | `device-encoded` | 真机跑通编码并校验码流 | ❌ 未做（环境受限，登记待验） |
//!
//! **第二档不是"能编码"**：函数指针解析得到 ≠ 设备在、驱动接受、能出码。
//! 本 crate 的对外表述止于"VA-API 绑定就绪（第二档）"；`is_available()` 仍恒 `false`
//! （[`any_available`] 恒 `false`，钉子见各 `tests` 模块）。
//!
//! # 逐后端 × 逐档现状
//!
//! | 后端 | 一 探测链 | 二 绑定就绪 | 三 会话建立 | 四 真机出码 |
//! |---|---|---|---|---|
//! | `videotoolbox` | ✅ 平台判定真读 OS | ❌ 无 `extern` 声明 | ❌ | ❌ |
//! | `nvenc` | ✅ 驱动文件 + 设备节点真读 | ❌ 无 `extern` 声明 | ❌ | ❌ |
//! | `vaapi` | ✅ `/dev/dri` 真读 | ✅ `vaapi_bind`：18 个核心符号（**只解析不调用**） | ❌ 不取 `VADisplay`、不 `vaInitialize` | ❌ |
//!
//! # 不做什么（每条都有理由，不是省事）
//!
//! - **VideoToolbox / NVENC 仍无任何 `extern` 声明**：`VTCompressionSessionCreate` /
//!   `nvEncOpenEncodeSessionEx` 一概没有；两者只到第一档；
//! - **VA-API 只做绑定就绪**：`vaapi_bind` 声明函数表并解析符号，但**不调用**
//!   任何 VA-API 入口，也不 `dlopen` `libva-drm` / `libva-x11`（`VADisplay` 由它们
//!   产出，属第三档）；
//! - **不 `#[link]` 任何库**（含 `libdl`）：`dlopen`/`dlsym` 用裸 `extern "C"`，
//!   符号由 glibc（≥ 2.34 已并入 `libc`）提供；
//! - 因此**没有任何编码会话被真正建立**，"能编码"需要真实设备，环境受限不伪装成
//!   已做（见文档"环境受限项"）。
//!
//! # feature 语义
//!
//! | feature | 含义 | 当前平台行为 |
//! |---|---|---|
//! | （默认，全关） | 不引入任何后端 | [`enabled_backends`] 空，[`probe_all`] 空 |
//! | `videotoolbox` | 编入 Apple VideoToolbox 探测 | 平台不符 ⇒ `Unsupported`（理由带真实平台串） |
//! | `nvenc` | 编入 NVIDIA NVENC 探测 | 无驱动文件/设备节点 ⇒ `Unsupported`（理由带真实证据） |
//! | `vaapi` | 编入 VA-API 探测 + **绑定层**（`vaapi_bind` 模块，仅 `unix`） | 无 `/dev/dri` 渲染节点 ⇒ `Unsupported`（理由带真实证据 + 绑定档结论） |
//!
//! # 主干零依赖不破
//!
//! 本 crate **零依赖**（`[dependencies]` 为空，仅 `std`），且**不被**任何主干
//! crate 依赖——平台差异完全隔离在本 crate 的 feature 之后。VA-API 绑定层用裸
//! `extern "C"` + `dlopen`/`dlsym`，**无 `#[link]`**，故不引入任何原生链接指令。

use std::fmt;

mod env_probe;

#[cfg(all(feature = "vaapi", unix))]
pub mod vaapi_bind;

#[cfg(all(feature = "vaapi", unix))]
pub use vaapi_bind::{
    bind as vaapi_bind_fn_table, bind_from as vaapi_bind_from, bind_report as vaapi_bind_report,
    VaapiBindError, VaapiBindReport, VaapiFnTable, VaapiOpenCause, LIBVA_SONAME, REQUIRED_SYMBOLS,
};

pub use env_probe::{
    host_os, join_paths, probe_environment, probe_nvenc, probe_vaapi, videotoolbox_platform_ok,
    EnvironmentReport, NvencEvidence, VaapiEvidence, DRI_DEV_DIR, DRI_RENDER_PREFIX,
    NVIDIA_DEV_PREFIX, NVIDIA_DRIVER_VERSION_FILE,
};

/// 真机 FFI 编码绑定尚未落地的说明标记。
///
/// 仅用于 **VideoToolbox / NVENC**——这两家连 `extern` 声明都没有，仍停在第一档
/// （探测链）。**VA-API 已不适用**：它有自己的 [`VAAPI_BINDING_TIER_NOTE`] 与
/// 逐次实测的绑定结论（见 `vaapi_bind::bind_report`）。
pub const PENDING_BINDING_NOTE: &str =
    "真机编码 FFI 绑定未落地（无平台库 link / 无 extern 声明；缺真机启用项）";

/// VA-API 已推进到的档位名称（第二档；与 [`Maturity::BindingReady`] 同源）。
pub const VAAPI_BINDING_TIER_NOTE: &str =
    "binding-ready（第二档：函数表已解析为指针，但一次未调用）";

/// 会话建立档缺失的说明（三后端共用；挡住"绑定就绪 ⇒ 能编码"的越级读法）。
pub const SESSION_NOT_ESTABLISHED_NOTE: &str =
    "会话建立未做（第三档）：未取 VADisplay、未调 vaInitialize / vaCreateConfig / vaCreateContext";

/// 真机出码档缺失的说明（环境受限，如实登记）。
pub const DEVICE_ENCODE_NOT_DONE_NOTE: &str =
    "真机出码未做（第四档）：无硬件编码器环境，按环境受限登记";

/// 硬件编码交付成熟度的**四档**（严格递增；分述口径的权威定义）。
///
/// 顺序即档位高低：`ProbeChain < BindingReady < SessionEstablished < DeviceEncoded`。
/// 任何"当前能力"的表述必须能指到具体档位，**禁止混用**（把第二档说成能编码是
/// 越级声称）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Maturity {
    /// 第一档：真读平台证据（OS / 驱动文件 / 设备节点），不碰库里任何符号。
    ProbeChain,
    /// 第二档：`dlopen` + `dlsym` 把核心符号解析成函数指针表，**一次都不调用**。
    BindingReady,
    /// 第三档：真正建会话（`VADisplay` + `vaInitialize` + 配置 + 上下文）。
    SessionEstablished,
    /// 第四档：真机跑通编码并校验码流。
    DeviceEncoded,
}

impl Maturity {
    /// 全部四档（按档位升序；矩阵/测试的行序权威集合）。
    pub const ALL: [Maturity; 4] = [
        Maturity::ProbeChain,
        Maturity::BindingReady,
        Maturity::SessionEstablished,
        Maturity::DeviceEncoded,
    ];

    /// 稳定的机器可判定 token（测试与日志用它，避免措辞漂移）。
    pub fn as_str(self) -> &'static str {
        match self {
            Maturity::ProbeChain => "probe-chain",
            Maturity::BindingReady => "binding-ready",
            Maturity::SessionEstablished => "session-established",
            Maturity::DeviceEncoded => "device-encoded",
        }
    }

    /// 档位序号（1 起）。
    pub fn level(self) -> u8 {
        match self {
            Maturity::ProbeChain => 1,
            Maturity::BindingReady => 2,
            Maturity::SessionEstablished => 3,
            Maturity::DeviceEncoded => 4,
        }
    }

    /// 是否**不超过** `ceiling`（越级声称的机械判据）。
    pub fn is_at_most(self, ceiling: Maturity) -> bool {
        self <= ceiling
    }
}

impl fmt::Display for Maturity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// VA-API 绑定档（feature `vaapi` + unix 才编译绑定层）
// ---------------------------------------------------------------------------

/// VA-API 当前达到的档位（**引擎实测所得，不是常量承诺**）。
///
/// - `--features vaapi`（unix）且 `libva.so.2` 可加载、18 个核心符号全部解析
///   ⇒ [`Maturity::BindingReady`]；
/// - 否则 ⇒ [`Maturity::ProbeChain`]（只剩探测链）。
///
/// **永远不返回第三/第四档**：本 crate 不建会话、不出码。
pub fn vaapi_tier() -> Maturity {
    #[cfg(all(feature = "vaapi", unix))]
    {
        if vaapi_bind::bind_report().is_ready() {
            return Maturity::BindingReady;
        }
    }
    Maturity::ProbeChain
}

/// VA-API 绑定档的一句话结论（拼进探测理由；**保证非空**）。
///
/// 措辞随实测状态变：就绪 ⇒ 列出库名与符号数；未就绪 ⇒ 直接给出结构化证据
/// （路径 + errno + `dlerror`）。
fn vaapi_binding_note() -> String {
    #[cfg(all(feature = "vaapi", unix))]
    {
        let report: &VaapiBindReport = vaapi_bind::bind_report();
        let tier: Maturity = vaapi_tier();
        // 缺档说明并陈：第三/第四档一律以"未做"落笔，杜绝"绑定就绪 ⇒ 能编码"的读法。
        let tail: String = format!(
            "{}；{}",
            SESSION_NOT_ESTABLISHED_NOTE, DEVICE_ENCODE_NOT_DONE_NOTE
        );
        if report.is_ready() {
            return format!("绑定档 = {VAAPI_BINDING_TIER_NOTE}：{}；{}", report.summary(), tail);
        }
        return format!("绑定档 = {tier}（第一档：绑定层未就绪）：{}；{}", report.summary(), tail);
    }
    #[cfg(not(all(feature = "vaapi", unix)))]
    {
        format!(
            "{}（未编入绑定层：需 --features vaapi 且 unix 目标）；{}",
            Maturity::ProbeChain, SESSION_NOT_ESTABLISHED_NOTE
        )
    }
}

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
    /// 探测到可用的硬件会话（需要真实设备与 FFI 绑定）。
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
/// 理由可由调用方给定（默认点明当前平台探测结论与绑定缺口）。
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
                // VA-API 已自带绑定层，不能用"无 extern 声明"的通用话术；
                // 结论由实测绑定状态驱动（见 vaapi_binding_note）。
                format!(
                    "vaapi: {}；{}",
                    env_probe::probe_vaapi().summary(),
                    vaapi_binding_note()
                )
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
/// 真探测：平台判定。非 macOS/iOS ⇒ 明确报"平台不符"；平台相符但当前版本无
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
                 当前平台无该平台硬件",
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
/// 真探测：`/dev/dri` 目录与 `renderD*` 渲染节点（第一档）+ 绑定层实测档位
/// （第二档，见 `vaapi_bind` 模块）。**不建会话、不出码**，故结论恒为 `Unsupported`。
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
        // 两档事实并陈：环境证据（第一档）+ 绑定实测（第二档）。
        // 无论绑定是否就绪，都不越过第二档——"能编码"需要设备与会话。
        let reason: String = if ev.is_absent() {
            format!(
                "vaapi: 未发现可用渲染节点——{}；无 DRM 设备即无 VA-API 硬件面；{}",
                ev.summary(),
                vaapi_binding_note()
            )
        } else {
            format!(
                "vaapi: 发现渲染节点（{}）；{}",
                ev.summary(),
                vaapi_binding_note()
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

/// 是否存在任一可用硬件编码器（当前平台恒为 `false`，见模块文档的诚实边界）。
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
    /// 当前平台环境证据摘要（真探测结论，与 feature 无关）。
    pub environment: String,
    /// 本构建的探测结论（feature 未开 = "未编入"）。
    pub status: String,
}

/// 当前平台 × 本构建的能力矩阵（三后端全覆盖，含未编入的）。
///
/// 这是能力矩阵文档的数据源：文档表格由本函数渲染，杜绝"文档手写、
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
            "硬件编码器不可用：当前平台未探测到可用后端（共 {} 条探测结论）",
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
/// 当前平台构建内无可用硬件编码器 ⇒ [`HwUnavailableError`]（含逐后端的真实理由）。
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
        // 当前版本无 FFI 绑定 ⇒ 所有编入的后端都必须如实返回 Unsupported，且理由
        // 非空、带**真实探测证据**（平台串 / 驱动文件 / 设备节点二选一）。
        let report: EnvironmentReport = probe_environment();
        for probe in probe_all() {
            assert_eq!(
                probe.status,
                ProbeStatus::Unsupported,
                "{} 的 {} 不应报告可用（当前平台构建无 FFI 编码绑定）",
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
        assert!(!any_available(), "当前平台 any_available() 必须为 false（无硬件或绑定）");
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
        // 当前平台（无 VT/NVENC/VAAPI 硬件与绑定）必须显式报错，
        // 且错误文本列出全部真实证据 + 建议段（C19）。
        match require_available() {
            Ok(p) => panic!("当前平台不应有可用硬件编码器，实得 {p:?}"),
            Err(e) => {
                let text: String = e.to_string();
                assert!(text.contains("硬件编码器不可用"), "首段须说清结论：{text}");
                assert!(text.contains("建议："), "必含建议段（C19）：{text}");
                assert!(text.contains("videotoolbox"), "须列出全部后端结论：{text}");
                assert!(text.contains("nvenc") && text.contains("vaapi"));
            }
        }
    }

    // -----------------------------------------------------------------------
    // 四档分述的钉子（验收 ②③：不混用、不越级）
    // -----------------------------------------------------------------------

    #[test]
    fn maturity_tiers_are_strictly_ordered_and_named() {
        // 四档必须严格递增，且 token 固定（机器可判定，防措辞漂移）。
        assert!(Maturity::ProbeChain < Maturity::BindingReady);
        assert!(Maturity::BindingReady < Maturity::SessionEstablished);
        assert!(Maturity::SessionEstablished < Maturity::DeviceEncoded);
        assert_eq!(Maturity::ALL.len(), 4);
        for (idx, tier) in Maturity::ALL.iter().enumerate() {
            assert_eq!(tier.level() as usize, idx + 1, "{tier} 的序号必须是 {}", idx + 1);
            assert!(!tier.as_str().is_empty());
        }
        assert!(Maturity::BindingReady.is_at_most(Maturity::BindingReady));
        assert!(!Maturity::SessionEstablished.is_at_most(Maturity::BindingReady));
    }

    #[test]
    fn tier_notes_are_disjoint_and_nonempty() {
        // 三条"缺档说明"各说各的档，不许互相冒充。
        assert!(PENDING_BINDING_NOTE.contains("未落地"));
        assert!(VAAPI_BINDING_TIER_NOTE.contains(Maturity::BindingReady.as_str()));
        assert!(
            !VAAPI_BINDING_TIER_NOTE.contains(Maturity::SessionEstablished.as_str()),
            "绑定档说明不得混入会话档 token"
        );
        assert!(SESSION_NOT_ESTABLISHED_NOTE.contains("第三档"));
        assert!(DEVICE_ENCODE_NOT_DONE_NOTE.contains("第四档"));
        assert!(!SESSION_NOT_ESTABLISHED_NOTE.contains("第二档"));
    }

    #[test]
    fn no_probe_reason_ever_claims_beyond_binding_ready() {
        // 三后端 × 两格式：任何理由都不许出现第三/第四档 token 的"已完成"式表述，
        // 且必须非空、必须带真实证据（沿用上面的证证据钉子，这里只钉档位）。
        for probe in probe_all() {
            assert!(!probe.reason.is_empty(), "{} 理由不得为空", probe.backend);
            assert_eq!(
                probe.status,
                ProbeStatus::Unsupported,
                "{} 不得声称可用（第二档上限）",
                probe.backend
            );
            // "session-established" / "device-encoded" 只能以"未做"的形式出现：
            // 逐条检查若出现则必须同句带"未做"。
            for forbidden in [Maturity::SessionEstablished.as_str(), Maturity::DeviceEncoded.as_str()]
            {
                if let Some(pos) = probe.reason.find(forbidden) {
                    let tail: &str = &probe.reason[pos..];
                    assert!(
                        tail.contains("未做"),
                        "{} 的理由在提及 {forbidden} 时必须写明未做：{}",
                        probe.backend,
                        probe.reason
                    );
                }
            }
        }
        assert!(!any_available(), "当前构建不存在任何可用硬件编码器");
    }

    #[cfg(all(feature = "vaapi", unix))]
    #[test]
    fn vaapi_probe_reason_reports_binding_tier_from_real_measurement() {
        // 绑定档结论必须来自实测：就绪 ⇒ 理由里带 binding-ready 且带库名；
        // 未就绪 ⇒ 带路径/errno 证据。两条都不得越过第二档。
        let report: &VaapiBindReport = vaapi_bind_report();
        let tier: Maturity = vaapi_tier();
        assert!(
            tier.is_at_most(Maturity::BindingReady),
            "VA-API 档位不得超过第二档，实得 {tier}"
        );
        let reason: String = VaapiProbe
            .probe()
            .first()
            .map(|p| p.reason.clone())
            .unwrap_or_default();
        assert!(reason.contains(DRI_DEV_DIR), "理由必须带 /dev/dri 证据：{reason}");
        if report.is_ready() {
            assert_eq!(tier, Maturity::BindingReady);
            assert!(reason.contains(Maturity::BindingReady.as_str()), "{reason}");
            assert!(reason.contains(LIBVA_SONAME), "理由必须点名库：{reason}");
            assert_eq!(report.resolved.len(), REQUIRED_SYMBOLS.len());
        } else {
            assert_eq!(tier, Maturity::ProbeChain);
            assert!(reason.contains("errno="), "未就绪必须带 errno 证据：{reason}");
        }
        assert_eq!(VaapiProbe.probe()[0].status, ProbeStatus::Unsupported);
    }
}
