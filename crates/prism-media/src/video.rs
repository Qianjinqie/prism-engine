//! ffmpeg 探测读入（B11；D-17：外部路径为主 + 范式终局 + 原生降级）。
//!
//! **不实现解码器**（红线）：自带**最小探测 stub** —— 运行时检查 `ffmpeg` 是否
//! 在 `PATH` 上（存在性检测先行），有则标注可协作、无则降级为原生可解格式。
//!
//! # 设计
//!
//! - 探测 = 在 `PATH` 各目录查 `ffmpeg`（Windows 追加 `.exe`）是否存在且为文件；
//! - **零网络、零外部进程调用**：本层只做存在性判定，实际解码由调用方在具备
//!   ffmpeg 时另启（超出本 crate 范围，D-17 的"代码依赖"硬约束）；
//! - `MediaProbe` 汇总：给定源路径，返回该源**实际可用**的读入路径（原生优先，
//!   否则 ffmpeg 协作，否则如实标注不可读）——"能不能读"如实标注，永静默不失败。

use crate::source::{extension_of, MediaSource, ReadKind, SourceKind};
use crate::MediaResult;

/// ffmpeg 探测结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegProbe {
    /// 是否在 `PATH` 上找到可执行的 `ffmpeg`。
    pub available: bool,
    /// 找到时的完整路径（未找到为 `None`）。
    pub path: Option<String>,
}

impl FfmpegProbe {
    /// 探测当前环境（读取 `PATH` 环境变量，逐目录查 `ffmpeg`/`ffmpeg.exe`）。
    ///
    /// 纯只读文件系统查询，不启动任何进程、不联网。
    pub fn detect() -> Self {
        Self::detect_in(std::env::var("PATH").unwrap_or_default().as_str())
    }

    /// 在指定 `PATH` 串中探测（便于测试注入确定性环境）。
    pub fn detect_in(path_env: &str) -> Self {
        let mut names: Vec<String> = vec!["ffmpeg".to_string()];
        if cfg!(windows) {
            names.push("ffmpeg.exe".to_string());
        }
        // 分隔符：Unix ':'，Windows ';'。
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
                if std::path::Path::new(&full).is_file() {
                    return Self { available: true, path: Some(full) };
                }
            }
        }
        Self { available: false, path: None }
    }

    /// 是否探测到 ffmpeg。
    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// 读入决策（D-17 降级链）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadDecision {
    /// 该源最终生效的读入路径。
    pub effective: ReadKind,
    /// 该源被静态分类出的理想路径（不看环境）。
    pub classified: ReadKind,
    /// 是否发生降级（理想为 ffmpeg 协作但环境无 ffmpeg）。
    pub degraded: bool,
}

/// 决定一个源在当前 ffmpeg 环境下的实际读入路径。
///
/// 降级链（D-17）：`Native` 恒可用；`ExternalFfmpeg` 在探测到 ffmpeg 时可用、
/// 否则降级为 [`ReadKind::Unreadable`]（该源本身非原生可解）并标记 `degraded`。
pub fn decide(source: &MediaSource, probe: &FfmpegProbe) -> ReadDecision {
    match source.read_kind {
        ReadKind::Native => ReadDecision {
            effective: ReadKind::Native,
            classified: ReadKind::Native,
            degraded: false,
        },
        ReadKind::ExternalFfmpeg => {
            if probe.is_available() {
                ReadDecision {
                    effective: ReadKind::ExternalFfmpeg,
                    classified: ReadKind::ExternalFfmpeg,
                    degraded: false,
                }
            } else {
                ReadDecision {
                    effective: ReadKind::Unreadable,
                    classified: ReadKind::ExternalFfmpeg,
                    degraded: true,
                }
            }
        }
        ReadKind::Unreadable => ReadDecision {
            effective: ReadKind::Unreadable,
            classified: ReadKind::Unreadable,
            degraded: false,
        },
    }
}

/// 序列帧目录读入（D-17 原生降级的另一路径）：给定目录，按文件名序返回其中
/// 原生可解图像（`.png`/`.gif`/`.apng`）路径列表。
///
/// 只做目录枚举与扩展名过滤（按名字典序，确定性）；不解码。
pub fn scan_frame_sequence(dir: &str) -> MediaResult<Vec<String>> {
    let mut paths: Vec<String> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        crate::MediaError::new(
            "VID1001",
            "video::scan_frame_sequence",
            format!("目录 `{dir}` 读取失败：{e}"),
            "确认目录存在且可读",
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| {
            crate::MediaError::new(
                "VID1001",
                "video::scan_frame_sequence",
                format!("目录项读取失败：{e}"),
                "确认目录权限",
            )
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = extension_of(&path.to_string_lossy());
        if matches!(ext.as_str(), "png" | "gif" | "apng") {
            paths.push(path.to_string_lossy().into_owned());
        }
    }
    paths.sort(); // 字典序 = 帧序（frame_0001.png…）
    Ok(paths)
}

/// 构造一个视频源（便捷函数）。
pub fn video_source(path: impl Into<String>) -> MediaSource {
    MediaSource::new(path, SourceKind::Video)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_empty_path_is_unavailable() {
        let probe = FfmpegProbe::detect_in("");
        assert!(!probe.available);
        assert!(probe.path.is_none());
        assert!(!probe.is_available());
    }

    #[test]
    fn detect_nonexistent_dir_is_unavailable() {
        let probe = FfmpegProbe::detect_in("/no/such/dir:/also/missing");
        assert!(!probe.available);
    }

    #[test]
    fn decide_native_always_works_without_ffmpeg() {
        // 无 ffmpeg 环境：GIF 原生可解仍可用（降级全流程可跑的前提）。
        let probe = FfmpegProbe { available: false, path: None };
        let gif = MediaSource::new("clip.gif", SourceKind::Video);
        let d = decide(&gif, &probe);
        assert_eq!(d.effective, ReadKind::Native);
        assert!(!d.degraded);

        // mp4 无 ffmpeg → 降级标注不可读（如实，不静默）。
        let mp4 = MediaSource::new("movie.mp4", SourceKind::Video);
        let d = decide(&mp4, &probe);
        assert_eq!(d.effective, ReadKind::Unreadable);
        assert!(d.degraded);
        assert_eq!(d.classified, ReadKind::ExternalFfmpeg);
    }

    #[test]
    fn decide_external_works_with_ffmpeg() {
        let probe = FfmpegProbe { available: true, path: Some("/usr/bin/ffmpeg".into()) };
        let mp4 = MediaSource::new("movie.mp4", SourceKind::Video);
        let d = decide(&mp4, &probe);
        assert_eq!(d.effective, ReadKind::ExternalFfmpeg);
        assert!(!d.degraded);
    }

    #[test]
    fn scan_missing_dir_errors_structured() {
        let err = scan_frame_sequence("/no/such/frames").unwrap_err();
        assert_eq!(err.code, "VID1001");
        assert!(err.to_string().contains("建议："));
    }
}
