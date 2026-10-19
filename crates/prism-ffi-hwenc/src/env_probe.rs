//! 硬件编码器**环境探测**（V2.0-M3 真探测链，纯 `std`）。
//!
//! # 探测什么、不探测什么（诚实边界）
//!
//! 本模块做**真实的环境探测**，但只覆盖"不引平台 FFI 就能拿到证据"的层面：
//!
//! | 后端 | 真探测的层面 | **不**做的层面 |
//! |---|---|---|
//! | `videotoolbox` | 平台判定（`std::env::consts::OS`）：非 macOS 即平台不符 | 不 link VideoToolbox.framework，不建 `VTCompressionSession` |
//! | `nvenc` | PCI/驱动证据：`/proc/driver/nvidia/version`、`/dev/nvidia*` 设备节点 | 不 dlopen `libnvidia-encode`，不调 `nvEncOpenEncodeSessionEx` |
//! | `vaapi` | DRM 设备证据：`/dev/dri` 目录与 `renderD*` 渲染节点 | 不 link `libva`，不调 `vaInitialize` |
//!
//! 所以可能的最好结论是两条：
//! 1. **环境证据缺失** ⇒ `Unsupported`，理由里带上**真实探测到的缺口**（哪个文件 /
//!    哪个目录 / 哪个平台不存在）——不是"留待里程碑"这种占位话术；
//! 2. **环境证据存在但 FFI 绑定未落地** ⇒ 同样 `Unsupported`，理由如实写明
//!    "发现设备证据 X，但真机编码启用项缺硬件/绑定"（环境受限项，不伪装成功）。
//!
//! 两条都不静默失败（C20）：调用方拿到的永远是结构化理由，可以逐条复核。
//!
//! # 为什么不上 FFI（本波次的口径）
//! 真机 FFI 绑定需要目标平台的头文件/链接库与真机，本沙箱为 Linux 且缺
//! `/dev/dri`、NVIDIA 设备与 macOS（证据见 [`probe_environment`] 的返回值）。
//! 缺硬件**不伪装成已做**，也不拿缺硬件当少做理由：探测链、gate 矩阵、
//! 降级路径、文档四件齐交付（见 `docs/v2-m3-ec1-hwenc-capability.md`）。

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// 平台判定
// ---------------------------------------------------------------------------

/// 当前编译目标平台（`std::env::consts::OS`；如 `"linux"` / `"macos"`）。
pub fn host_os() -> &'static str {
    std::env::consts::OS
}

/// VideoToolbox 的平台前提是否满足（仅 macOS / iOS）。
pub fn videotoolbox_platform_ok() -> bool {
    matches!(std::env::consts::OS, "macos" | "ios")
}

// ---------------------------------------------------------------------------
// NVENC：驱动文件 + 设备节点证据
// ---------------------------------------------------------------------------

/// NVIDIA 驱动版本文件（内核模块加载后才存在）。
pub const NVIDIA_DRIVER_VERSION_FILE: &str = "/proc/driver/nvidia/version";

/// NVIDIA 设备节点前缀（`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`）。
pub const NVIDIA_DEV_PREFIX: &str = "/dev/nvidia";

/// NVENC 的环境证据（真实探测结果）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NvencEvidence {
    /// `/proc/driver/nvidia/version` 的首行（驱动已加载时的真实文本）。
    pub driver_version_line: Option<String>,
    /// 存在的 `/dev/nvidia*` 设备节点（升序）。
    pub device_nodes: Vec<String>,
}

impl NvencEvidence {
    /// 是否**完全没有** NVIDIA 证据（驱动文件与设备节点都没有）。
    pub fn is_absent(&self) -> bool {
        self.driver_version_line.is_none() && self.device_nodes.is_empty()
    }

    /// 人读摘要（入档 / 报错用）。
    pub fn summary(&self) -> String {
        if self.is_absent() {
            return format!(
                "无 NVIDIA 证据：{NVIDIA_DRIVER_VERSION_FILE} 不存在，{NVIDIA_DEV_PREFIX}* 无设备节点"
            );
        }
        let mut parts: Vec<String> = Vec::new();
        if let Some(line) = &self.driver_version_line {
            parts.push(format!("驱动版本文件在（{NVIDIA_DRIVER_VERSION_FILE} 首行：{line}）"));
        }
        if !self.device_nodes.is_empty() {
            parts.push(format!("设备节点在：{}", self.device_nodes.join("、")));
        }
        parts.join("；")
    }
}

/// 探测 NVIDIA 环境证据（Linux/Unix 视角；非 Unix 平台同样查一遍，文件不存在
/// 即"无证据"，与 Unix 缺席同口径，不特判、不猜测）。
pub fn probe_nvenc() -> NvencEvidence {
    let driver_version_line: Option<String> = std::fs::read_to_string(NVIDIA_DRIVER_VERSION_FILE)
        .ok()
        .and_then(|text| text.lines().next().map(|l| l.trim().to_string()))
        .filter(|l| !l.is_empty());
    let mut device_nodes: Vec<String> = Vec::new();
    // 设备节点没有通配符 API：按既有命名约定逐个探测，存在的才计入——
    // 这就是"PCI/驱动探测"的本机口径（不猜测、不读取 PCI 配置空间）。
    for node in [
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia0",
        "/dev/nvidia1",
        "/dev/nvidia2",
        "/dev/nvidia3",
        "/dev/nvidia4",
        "/dev/nvidia5",
        "/dev/nvidia6",
        "/dev/nvidia7",
    ] {
        if Path::new(node).exists() {
            device_nodes.push(node.to_string());
        }
    }
    device_nodes.sort();
    NvencEvidence { driver_version_line, device_nodes }
}

// ---------------------------------------------------------------------------
// VAAPI：DRM 渲染节点证据
// ---------------------------------------------------------------------------

/// DRM 设备目录。
pub const DRI_DEV_DIR: &str = "/dev/dri";

/// 渲染节点前缀（`/dev/dri/renderD128` 起）。
pub const DRI_RENDER_PREFIX: &str = "renderD";

/// VA-API 的环境证据（真实探测结果）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VaapiEvidence {
    /// `/dev/dri` 目录是否存在。
    pub dri_dir_present: bool,
    /// `/dev/dri` 下的渲染节点名（升序，如 `renderD128`）。
    pub render_nodes: Vec<String>,
}

impl VaapiEvidence {
    /// 是否没有可用渲染节点。
    pub fn is_absent(&self) -> bool {
        !self.dri_dir_present || self.render_nodes.is_empty()
    }

    /// 人读摘要（入档 / 报错用）。
    pub fn summary(&self) -> String {
        if !self.dri_dir_present {
            return format!("无 DRM 证据：{DRI_DEV_DIR} 目录不存在");
        }
        if self.render_nodes.is_empty() {
            return format!("{DRI_DEV_DIR} 存在但无渲染节点（renderD* 缺席）");
        }
        format!(
            "{} 在，渲染节点：{}",
            DRI_DEV_DIR,
            self.render_nodes.join("、")
        )
    }
}

/// 探测 VA-API 环境证据：`/dev/dri` 目录 + `renderD*` 渲染节点（逐个 read_dir）。
pub fn probe_vaapi() -> VaapiEvidence {
    let mut render_nodes: Vec<String> = Vec::new();
    let dri_dir_present: bool = match std::fs::read_dir(DRI_DEV_DIR) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name: String = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with(DRI_RENDER_PREFIX) {
                    render_nodes.push(format!("{DRI_DEV_DIR}/{name}"));
                }
            }
            true
        }
        Err(_) => false,
    };
    render_nodes.sort();
    VaapiEvidence { dri_dir_present, render_nodes }
}

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------

/// 一机的环境探测汇总（三个后端的真实证据 + 当前平台）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentReport {
    /// 当前平台（`std::env::consts::OS`）。
    pub os: &'static str,
    /// VideoToolbox 平台前提是否满足。
    pub videotoolbox_platform: bool,
    /// NVENC 证据。
    pub nvenc: NvencEvidence,
    /// VA-API 证据。
    pub vaapi: VaapiEvidence,
}

/// 跑一遍本机环境探测（无副作用：只读文件系统与编译期常量）。
pub fn probe_environment() -> EnvironmentReport {
    EnvironmentReport {
        os: host_os(),
        videotoolbox_platform: videotoolbox_platform_ok(),
        nvenc: probe_nvenc(),
        vaapi: probe_vaapi(),
    }
}

impl EnvironmentReport {
    /// 多行人读报告（`--features` 任一形态下都可打印；入档用）。
    pub fn render(&self) -> String {
        let vt: &str = if self.videotoolbox_platform {
            "平台相符（macOS/iOS）"
        } else {
            "平台不符（仅 macOS/iOS）"
        };
        [
            format!("平台 OS = {}", self.os),
            format!("videotoolbox 平台判定：{vt}"),
            format!("nvenc 证据：{}", self.nvenc.summary()),
            format!("vaapi 证据：{}", self.vaapi.summary()),
        ]
        .join("\n")
    }
}

/// 把路径列表渲染为稳定字符串（测试与文档用）。
pub fn join_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("、")
}

// ---------------------------------------------------------------------------
// 测试（最小充分集）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_probe_is_side_effect_free_and_repeatable() {
        // 无副作用 + 可重复：两遍结果相同（只读文件系统，无时钟/随机）。
        let a: EnvironmentReport = probe_environment();
        let b: EnvironmentReport = probe_environment();
        assert_eq!(a, b, "环境探测必须是可重复的纯观测");
        assert!(!a.os.is_empty(), "平台串不得为空");
    }

    #[test]
    fn vaapi_probe_reports_real_dri_evidence() {
        let ev: VaapiEvidence = probe_vaapi();
        let report: EnvironmentReport = probe_environment();
        if ev.dri_dir_present {
            // 有 /dev/dri：摘要必须列出 renderD* 节点（或如实说缺席）。
            assert!(
                ev.summary().contains(DRI_DEV_DIR),
                "有目录时摘要必须带上目录名：{}",
                ev.summary()
            );
        } else {
            // 本沙箱预期路径：目录不存在 ⇒ 明确说"不存在"。
            assert!(
                ev.summary().contains("不存在"),
                "目录缺席时摘要必须如实说不存在：{}",
                ev.summary()
            );
        }
        assert_eq!(ev, report.vaapi, "汇总必须与单项探测一致");
    }

    #[test]
    fn nvenc_probe_reports_real_driver_or_device_evidence() {
        let ev: NvencEvidence = probe_nvenc();
        if ev.is_absent() {
            assert!(
                ev.summary().contains(NVIDIA_DRIVER_VERSION_FILE)
                    && ev.summary().contains(NVIDIA_DEV_PREFIX),
                "缺席时摘要必须点出两个探测点：{}",
                ev.summary()
            );
        } else {
            assert!(
                ev.summary().contains("驱动版本文件在") || ev.summary().contains("设备节点在"),
                "有证据时摘要必须点出证据：{}",
                ev.summary()
            );
        }
    }

    #[test]
    fn videotoolbox_platform_flag_matches_host_os() {
        let expected: bool = matches!(host_os(), "macos" | "ios");
        assert_eq!(videotoolbox_platform_ok(), expected, "平台判定必须与 OS 一致");
    }
}
