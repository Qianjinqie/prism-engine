//! 硬件编码器适配层（C7； 三层策略 ②）。
//!
//! **设计 + 骨架，真机启用**（Phase 4）：VideoToolbox（macOS）/ MediaFoundation
//! （Windows）/ NVENC（Linux + N 卡）/ VAAPI（Linux 通用）。本模块只定义**厂商无关
//! 的统一接口与能力声明**；实际编码调用厂商 API（超出零依赖 crate 范围，由
//! `ffi` 绑定层在真机启用）。
//!
//! # 边界（红线）
//!
//! **不自研 H.264/HEVC 编码器**（D-10）。硬件编码器是"原始厂商 API"边界内，
//! 本层只做适配骨架与降级编排：真机不可用时，编码回退到 [`crate::extprobe`] 的
//! 外部 ffmpeg 路径，再回退到内置 MJPEG。

use crate::extprobe::Codec;

/// 硬件编码器后端。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    /// macOS VideoToolbox。
    VideoToolbox,
    /// Windows MediaFoundation。
    MediaFoundation,
    /// NVIDIA NVENC。
    Nvenc,
    /// Linux VA-API。
    Vaapi,
}

impl HwBackend {
    /// 人类可读名。
    pub fn name(self) -> &'static str {
        match self {
            HwBackend::VideoToolbox => "VideoToolbox",
            HwBackend::MediaFoundation => "MediaFoundation",
            HwBackend::Nvenc => "NVENC",
            HwBackend::Vaapi => "VAAPI",
        }
    }

    /// 该后端在此 OS 上是否**可能**可用（编译期平台判定；真机能力需运行时探测）。
    pub fn platform_possible(self) -> bool {
        match self {
            HwBackend::VideoToolbox => cfg!(target_os = "macos"),
            HwBackend::MediaFoundation => cfg!(target_os = "windows"),
            HwBackend::Nvenc | HwBackend::Vaapi => cfg!(target_os = "linux"),
        }
    }

    /// 当前平台的可能后端清单（确定性序）。
    pub fn for_platform() -> Vec<HwBackend> {
        [HwBackend::VideoToolbox, HwBackend::MediaFoundation, HwBackend::Nvenc, HwBackend::Vaapi]
            .into_iter()
            .filter(|b| b.platform_possible())
            .collect()
    }
}

/// 硬件编码器能力（真机探测结果）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HwCapability {
    /// 后端。
    pub backend: HwBackend,
    /// 支持的编码族。
    pub codecs: &'static [Codec],
    /// 最大宽。
    pub max_width: u32,
    /// 最大高。
    pub max_height: u32,
}

/// 硬件编码器探测接口（真机由厂商 API 实现；沙箱返回空集）。
pub trait HwEncoders {
    /// 枚举当前机器可用的硬件编码器能力（确定性序）。
    fn enumerate(&self) -> Vec<HwCapability>;
}

/// 骨架探测：**无真机 FFI**，仅返回平台"可能"后端的能力声明（未验证）。
///
/// 真机启用时替换为厂商 API 查询；沙箱/无 FFI 环境返回空集，编码自动降级。
#[derive(Debug, Default, Clone, Copy)]
pub struct SkeletonProbe;

impl HwEncoders for SkeletonProbe {
    fn enumerate(&self) -> Vec<HwCapability> {
        // 无 FFI 绑定 → 保守返回空集（真机启用由 Phase 4 的绑定层实现）。
        Vec::new()
    }
}

/// 统一编码器请求（厂商无关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeRequest {
    /// 目标编码族。
    pub codec: Codec,
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 帧率。
    pub fps: u32,
    /// 目标码率（kbps；0 = 恒定质量）。
    pub bitrate_kbps: u32,
}

/// 编码器适配器接口（厂商无关）；真机实现由各后端提供。
///
/// 当前仅定义契约（骨架），沙箱无实现——`available()` 恒 `false`。
pub trait EncoderAdapter {
    /// 本适配器是否在当前机器可用（真机探测）。
    fn available(&self) -> bool;

    /// 底层后端。
    fn backend(&self) -> HwBackend;

    /// 是否支持该请求（编码族 + 尺寸）。
    fn supports(&self, req: &EncodeRequest) -> bool;

    /// 编码一帧（返回压缩字节；骨架/不可用时返回 `None`）。
    fn encode_frame(&mut self, rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>>;

    /// 冲刷（返回尾部压缩字节 + 统计）。
    fn flush(&mut self) -> Vec<u8>;
}

/// 从能力列表挑选满足请求的后端（确定性序：按枚举序取首个）。
pub fn select_encoder(caps: &[HwCapability], req: &EncodeRequest) -> Option<HwBackend> {
    caps.iter()
        .find(|c| {
            c.codecs.contains(&req.codec)
                && req.width <= c.max_width
                && req.height <= c.max_height
        })
        .map(|c| c.backend)
}

// ===========================================================================
// 运行时后端探测 + 交付计划（只读环境检查，无 FFI）
// ===========================================================================

/// 单个硬件后端的运行时可用性（含人类可读证据/原因）。
///
/// 与 [`HwCapability`] 的分工：`HwCapability` 是"厂商 API 声明的能力"，本枚举只是
/// **不调用任何 API 的只读环境探测**结论（设备节点 / 工具是否存在），用于在无 FFI 的
/// 环境下给出诚实的"能/不能"判断与原因链。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendStatus {
    /// 环境证据表明可能可用；`String` 为证据描述。
    Available(String),
    /// 不可用；`String` 为原因（三段式的"什么事"）。
    Unavailable(String),
}

impl BackendStatus {
    /// 是否可用。
    pub fn is_available(&self) -> bool {
        matches!(self, BackendStatus::Available(_))
    }

    /// 证据 / 原因文本。
    pub fn reason(&self) -> &str {
        match self {
            BackendStatus::Available(s) | BackendStatus::Unavailable(s) => s,
        }
    }
}

/// 在 `PATH` 中查找可执行文件（只读文件系统查询，不 spawn、不联网）。
fn find_in_path(name: &str) -> Option<String> {
    let path_env = std::env::var("PATH").unwrap_or_default();
    let sep = if cfg!(windows) { ';' } else { ':' };
    for dir in path_env.split(sep) {
        if dir.is_empty() {
            continue;
        }
        let full = format!("{dir}/{name}");
        if std::path::Path::new(&full).is_file() {
            return Some(full);
        }
    }
    None
}

/// 探测本机各硬件编码后端的可能可用性（**确定性序**：`HwBackend::for_platform` 顺序）。
///
/// 判据（纯只读，无厂商 FFI、无进程启动）：
/// - **NVENC**：`PATH` 中存在 `nvidia-smi`（驱动工具链存在）；
/// - **VAAPI**：`/dev/dri` 存在（DRM 渲染节点目录）；
/// - **VideoToolbox / MediaFoundation**：非本平台 → 不可用（平台判定由
///   [`HwBackend::platform_possible`] 完成）。
///
/// 该判定是"可能可用"的保守下界：真正编码仍需厂商 API 绑定（Phase 4）。
pub fn probe_backends() -> Vec<(HwBackend, BackendStatus)> {
    let mut out: Vec<(HwBackend, BackendStatus)> = Vec::new();
    for b in HwBackend::for_platform() {
        let status = match b {
            HwBackend::Nvenc => match find_in_path("nvidia-smi") {
                Some(p) => BackendStatus::Available(format!("PATH 中 nvidia-smi 存在：{p}")),
                None => {
                    BackendStatus::Unavailable("PATH 中未找到 nvidia-smi（无 NVIDIA 驱动工具链）".to_string())
                }
            },
            HwBackend::Vaapi => {
                if std::path::Path::new("/dev/dri").exists() {
                    BackendStatus::Available("/dev/dri 存在（有 DRM 渲染节点）".to_string())
                } else {
                    BackendStatus::Unavailable("/dev/dri 不存在（无机载/虚拟 GPU 渲染节点）".to_string())
                }
            }
            HwBackend::VideoToolbox => {
                BackendStatus::Unavailable("非 macOS 平台：VideoToolbox 不可用".to_string())
            }
            HwBackend::MediaFoundation => {
                BackendStatus::Unavailable("非 Windows 平台：MediaFoundation 不可用".to_string())
            }
        };
        out.push((b, status));
    }
    out
}

/// 交付计划：最终选定的硬件后端（`None` = 降级软件编码）+ 决策原因链。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryPlan {
    /// 调用方请求的后端。
    pub requested: HwBackend,
    /// 最终选定；`None` 表示无硬件后端可用，降级软件编码（内置 MJPEG / 外部 ffmpeg）。
    pub chosen: Option<HwBackend>,
    /// 决策原因链（按探测顺序逐条记录；永不为空——降级时必须说明为什么）。
    pub reasons: Vec<String>,
}

/// 制定交付计划：请求可用则用之；否则在 [`HwBackend::for_platform`] 里取**首个**可用
/// 后端；全不可用则降级软件编码，并带回完整原因链。
pub fn delivery_plan(requested: HwBackend) -> DeliveryPlan {
    let probed = probe_backends();
    let mut reasons: Vec<String> = Vec::new();
    // 请求后端是否就是可用后端（含"不在本平台列表"的情形）。
    for (b, st) in &probed {
        if *b == requested && st.is_available() {
            reasons.push(format!("{}：{}", b.name(), st.reason()));
            return DeliveryPlan { requested, chosen: Some(*b), reasons };
        }
    }
    match probed.iter().find(|(_, st)| st.is_available()) {
        Some((b, st)) => {
            reasons.push(format!(
                "请求的 {} 不可用（{}）；按平台可能顺序改用 {}：{}",
                requested.name(),
                if requested.platform_possible() {
                    "本机环境探测未通过"
                } else {
                    "非本平台"
                },
                b.name(),
                st.reason()
            ));
            DeliveryPlan { requested, chosen: Some(*b), reasons }
        }
        None => {
            let detail: Vec<String> = probed
                .iter()
                .map(|(b, st)| format!("{}: {}", b.name(), st.reason()))
                .collect();
            reasons.push(format!(
                "请求的 {} 不可用；本平台 {} 个候选后端全不可用 → 降级软件编码（MJPEG/ffmpeg）。详情：{}",
                requested.name(),
                probed.len(),
                if detail.is_empty() { "（本平台无硬件编码候选）".to_string() } else { detail.join("；") }
            ));
            DeliveryPlan { requested, chosen: None, reasons }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_backends_are_os_consistent() {
        let backends = HwBackend::for_platform();
        for b in &backends {
            assert!(b.platform_possible());
        }
        if cfg!(target_os = "linux") {
            assert!(backends.contains(&HwBackend::Nvenc));
            assert!(!backends.contains(&HwBackend::VideoToolbox));
        }
    }

    #[test]
    fn skeleton_probe_empty_then_degrades() {
        let caps = SkeletonProbe.enumerate();
        assert!(caps.is_empty(), "无 FFI 环境应返回空能力集");
        let req = EncodeRequest {
            codec: Codec::H264,
            width: 1920,
            height: 1080,
            fps: 30,
            bitrate_kbps: 8000,
        };
        assert_eq!(select_encoder(&caps, &req), None, "无能力时应降级");
    }

    #[test]
    fn selection_respects_codec_and_size() {
        let caps = [HwCapability {
            backend: HwBackend::Nvenc,
            codecs: &[Codec::H264, Codec::Hevc],
            max_width: 3840,
            max_height: 2160,
        }];
        let ok = EncodeRequest { codec: Codec::H264, width: 1920, height: 1080, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &ok), Some(HwBackend::Nvenc));
        let too_big = EncodeRequest { codec: Codec::H264, width: 8000, height: 4320, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &too_big), None);
        let mjpeg = EncodeRequest { codec: Codec::Mjpeg, width: 640, height: 480, fps: 30, bitrate_kbps: 0 };
        assert_eq!(select_encoder(&caps, &mjpeg), None, "硬件后端不支持 MJPEG");
    }

    #[test]
    fn probe_backends_is_deterministic() {
        // 两次探测必须逐条相同（无随机、无时间依赖；只读文件系统 + 环境变量）。
        let a = probe_backends();
        let b = probe_backends();
        assert_eq!(a, b, "probe_backends 两跑不一致（违反确定性契约）");
        // 只覆盖本平台可能的后端，且顺序与 for_platform 一致。
        let plat = HwBackend::for_platform();
        let got: Vec<HwBackend> = a.iter().map(|(x, _)| *x).collect();
        assert_eq!(got, plat, "探测列表必须等于平台可能后端（确定性序）");
        for (be, st) in &a {
            assert!(!st.reason().is_empty(), "{} 缺少证据/原因文本", be.name());
        }
    }

    #[test]
    fn linux_platform_candidates_are_nvenc_and_vaapi() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let got: Vec<HwBackend> = probe_backends().into_iter().map(|(b, _)| b).collect();
        for b in &got {
            assert!(
                matches!(b, HwBackend::Nvenc | HwBackend::Vaapi),
                "Linux 上不应出现 {} 后端",
                b.name()
            );
        }
        // 平台候选 ⊆ {Nvenc, Vaapi} 且两跑一致。
        assert_eq!(got, HwBackend::for_platform());
    }

    #[test]
    fn delivery_plan_degrades_with_nonempty_reason_chain() {
        // 逐个请求所有后端：计划必须带回非空原因链，且 chosen 若非 None 必为可用后端。
        let probed = probe_backends();
        for req in [
            HwBackend::VideoToolbox,
            HwBackend::MediaFoundation,
            HwBackend::Nvenc,
            HwBackend::Vaapi,
        ] {
            let plan = delivery_plan(req);
            assert_eq!(plan.requested, req);
            assert!(!plan.reasons.is_empty(), "{} 的决策原因链为空", req.name());
            match plan.chosen {
                Some(b) => {
                    let st = probed
                        .iter()
                        .find(|(x, _)| *x == b)
                        .map(|(_, s)| s.clone())
                        .expect("选定的后端必须来自探测列表");
                    assert!(st.is_available(), "选定 {} 却不可用", b.name());
                }
                None => {
                    // 降级软编：原因链必须说明"全不可用"。
                    assert!(
                        plan.reasons.iter().any(|r| r.contains("降级软件编码")),
                        "降级时原因链必须说明降级：{:?}",
                        plan.reasons
                    );
                }
            }
        }
        // 两跑一致。
        assert_eq!(delivery_plan(HwBackend::Vaapi), delivery_plan(HwBackend::Vaapi));
    }

    #[test]
    fn backend_names() {
        assert_eq!(HwBackend::Nvenc.name(), "NVENC");
        assert_eq!(HwBackend::Vaapi.name(), "VAAPI");
    }
}
