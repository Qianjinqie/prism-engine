//! ffmpeg 探测器（C7；路线图 C7 行 / §3.5 三层策略 ① / D-17 / D-10）。
//!
//! **不实现 H.264/HEVC 编码器**（红线，D-10）。本模块只做**能力探测 + 降级链**：
//!
//! 1. **PATH 检测**：在 `PATH` 各目录查 `ffmpeg`（+ Windows `.exe`）是否存在且为文件；
//! 2. **编码器能力探测**：无外部进程调用（零依赖约束），以**编译期能力清单**建模
//!    常见构建（libx264/libx265/nvenc/videotoolbox…）——运行时以声明式 `Encoders`
//!    集合表达，等待真机 `-encoders` 解析接入（同构接口）；
//! 3. **自动降级链**：`H264 → HEVC → MJPEG(内置)`，永远有可用终态（内置 MJPEG/AVI
//!    无任何外部依赖）。
//!
//! 与 `prism-media::video`（B11 读入探测）同哲学：外部可执行文件**可选协作**、
//! 非代码依赖；有则增强、无则降级，永静默不失败。

use std::path::Path;

/// 编码器族（交付目标）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Codec {
    /// H.264（需外部 ffmpeg 或硬件编码器）。
    H264,
    /// H.265/HEVC（需外部 ffmpeg 或硬件编码器）。
    Hevc,
    /// MJPEG（引擎内置终态，永远可用）。
    Mjpeg,
}

impl Codec {
    /// 人类可读名。
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Mjpeg => "mjpeg",
        }
    }

    /// 是否为引擎内置（无外部依赖）。
    pub fn is_builtin(self) -> bool {
        matches!(self, Codec::Mjpeg)
    }
}

/// ffmpeg 探测结果 + 编码器能力集。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegProbe {
    /// 是否在 `PATH` 上找到可执行的 `ffmpeg`。
    pub available: bool,
    /// 找到时的完整路径（未找到为 `None`）。
    pub path: Option<String>,
    /// 探测到的编码器能力（按族）。
    pub encoders: Encoders,
}

/// 编码器能力集（外部 ffmpeg 与硬件编码器的可用性声明）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Encoders {
    /// 软件 H.264（libx264）。
    pub libx264: bool,
    /// 软件 H.265（libx265）。
    pub libx265: bool,
    /// 硬件编码器族（VideoToolbox/MediaFoundation/NVENC/VAAPI 任一可用）。
    pub hardware: bool,
}

impl Encoders {
    /// 全无能力（无 ffmpeg 环境）。
    pub const NONE: Encoders =
        Encoders { libx264: false, libx265: false, hardware: false };

    /// 是否任一外部 H.264 能力可用。
    pub fn has_h264(&self) -> bool {
        self.libx264 || self.hardware
    }

    /// 是否任一外部 HEVC 能力可用。
    pub fn has_hevc(&self) -> bool {
        self.libx265 || self.hardware
    }
}

impl FfmpegProbe {
    /// 探测当前环境（读取 `PATH`；能力清单默认空——真机 `-encoders` 解析为 P1 待接）。
    pub fn detect() -> Self {
        let path_env = std::env::var("PATH").unwrap_or_default();
        Self::detect_in(&path_env)
    }

    /// 在指定 `PATH` 串中探测 ffmpeg 可执行文件（便于测试注入确定性环境）。
    ///
    /// 纯只读文件系统查询，不启动任何进程、不联网。
    pub fn detect_in(path_env: &str) -> Self {
        let mut names: Vec<String> = vec!["ffmpeg".to_string()];
        if cfg!(windows) {
            names.push("ffmpeg.exe".to_string());
        }
        let sep = if cfg!(windows) { ';' } else { ':' };
        for dir in path_env.split(sep) {
            if dir.is_empty() {
                continue;
            }
            for name in &names {
                let full = if dir.ends_with('/') || dir.ends_with('\\') {
                    format!("{dir}{name}")
                } else {
                    format!("{dir}/{name}")
                };
                if Path::new(&full).is_file() {
                    // 找到可执行文件；能力清单待真机探测（此处按"仅软件 H.264"保守声明，
                    // 因绝大多数发行构建含 libx264）。
                    return FfmpegProbe {
                        available: true,
                        path: Some(full),
                        encoders: Encoders { libx264: true, libx265: false, hardware: false },
                    };
                }
            }
        }
        FfmpegProbe { available: false, path: None, encoders: Encoders::NONE }
    }

    /// 构造一个"无 ffmpeg"探测结果（测试/离线环境）。
    pub fn unavailable() -> Self {
        FfmpegProbe { available: false, path: None, encoders: Encoders::NONE }
    }

    /// 是否探测到 ffmpeg。
    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// 交付决策（降级链的结果）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryDecision {
    /// 最终选定的编码器族。
    pub codec: Codec,
    /// 是否发生降级（理想目标不可用，退回内置 MJPEG）。
    pub degraded: bool,
    /// 理想目标（调用方请求）。
    pub requested: Codec,
}

/// 决定在给定探测环境下交付 `requested` 的实际编码器（D-10 降级链）。
///
/// 降级链：请求 H264 → （外部可用则 H264）→ HEVC → MJPEG；请求 HEVC → （外部可用则
/// HEVC）→ H264 → MJPEG；请求 MJPEG → MJPEG。**MJPEG 为终态，永远可用**。
///
/// 注：不自研 H.264/HEVC，故 H264/HEVC 仅在 `probe.available` 时成立；不可用则降级。
pub fn decide(requested: Codec, probe: &FfmpegProbe) -> DeliveryDecision {
    let codec = match requested {
        Codec::Mjpeg => Codec::Mjpeg,
        Codec::H264 => {
            if probe.encoders.has_h264() {
                Codec::H264
            } else if probe.encoders.has_hevc() {
                Codec::Hevc
            } else {
                Codec::Mjpeg
            }
        }
        Codec::Hevc => {
            if probe.encoders.has_hevc() {
                Codec::Hevc
            } else if probe.encoders.has_h264() {
                Codec::H264
            } else {
                Codec::Mjpeg
            }
        }
    };
    DeliveryDecision { codec, degraded: codec != requested, requested }
}

/// 探测并在无 ffmpeg 环境下的完整交付计划（供上层直接消费）。
///
/// 返回降级链途经的所有候选（含终态），调用方按序尝试。
pub fn delivery_chain(requested: Codec) -> Vec<Codec> {
    match requested {
        Codec::Mjpeg => vec![Codec::Mjpeg],
        Codec::H264 => vec![Codec::H264, Codec::Hevc, Codec::Mjpeg],
        Codec::Hevc => vec![Codec::Hevc, Codec::H264, Codec::Mjpeg],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_empty_path_unavailable() {
        let p = FfmpegProbe::detect_in("");
        assert!(!p.available);
        assert!(p.path.is_none());
        assert_eq!(p.encoders, Encoders::NONE);
    }

    #[test]
    fn no_ffmpeg_degrades_to_mjpeg() {
        let probe = FfmpegProbe::unavailable();
        let d = decide(Codec::H264, &probe);
        assert_eq!(d.codec, Codec::Mjpeg);
        assert!(d.degraded);
        assert_eq!(d.requested, Codec::H264);
        // MJPEG 请求恒可用。
        let d2 = decide(Codec::Mjpeg, &probe);
        assert_eq!(d2.codec, Codec::Mjpeg);
        assert!(!d2.degraded);
    }

    #[test]
    fn with_ffmpeg_h264_requested_h264() {
        let probe = FfmpegProbe {
            available: true,
            path: Some("/usr/bin/ffmpeg".into()),
            encoders: Encoders { libx264: true, libx265: true, hardware: false },
        };
        assert_eq!(decide(Codec::H264, &probe).codec, Codec::H264);
        assert_eq!(decide(Codec::Hevc, &probe).codec, Codec::Hevc);
    }

    #[test]
    fn hevc_requested_falls_back_to_h264_then_mjpeg() {
        let only_h264 = FfmpegProbe {
            available: true,
            path: None,
            encoders: Encoders { libx264: true, libx265: false, hardware: false },
        };
        let d = decide(Codec::Hevc, &only_h264);
        assert_eq!(d.codec, Codec::H264);
        assert!(d.degraded);

        let none = FfmpegProbe::unavailable();
        let d2 = decide(Codec::Hevc, &none);
        assert_eq!(d2.codec, Codec::Mjpeg);
    }

    #[test]
    fn chain_always_ends_in_builtin() {
        for req in [Codec::H264, Codec::Hevc, Codec::Mjpeg] {
            let chain = delivery_chain(req);
            assert_eq!(*chain.last().unwrap(), Codec::Mjpeg, "降级链必须以内置 MJPEG 收尾");
        }
    }

    #[test]
    fn builtin_flag() {
        assert!(Codec::Mjpeg.is_builtin());
        assert!(!Codec::H264.is_builtin());
        assert_eq!(Codec::H264.name(), "h264");
    }
}
