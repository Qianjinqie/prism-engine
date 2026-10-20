//! ffmpeg 探测器（C7； 三层策略 ① / D-17 / D-10）。
//!
//! **不实现 H.264/HEVC 编码器**（设计约束，D-10）。本模块只做**能力探测 + 降级链**：
//!
//! 1. **PATH 检测**：在 `PATH` 各目录查 `ffmpeg`（+ Windows `.exe`）是否存在且为文件；
//! 2. **编码器能力探测**：无外部进程调用（零依赖约束），以**编译期能力清单**建模
//!    常见构建（libx264/libx265/nvenc/videotoolbox…）——运行时以声明式 `Encoders`
//!    集合表达，等待真机 `-encoders` 解析接入（同构接口）；
//! 3. **自动降级链**：`H264 → HEVC → MJPEG(内置)`，永远有可用终态（内置 MJPEG/AVI
//!    无任何外部依赖）。
//!
//! 与 `prism-media::video`（读入探测）同哲学：外部可执行文件**可选协作**、
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

// ===========================================================================
// ffmpeg 运行时协作（D-17：外部可执行文件可选协作、永不静默失败）
// ===========================================================================

/// 构造 `ffmpeg` 管道编码参数（**纯函数**，便于离线断言布局）。
///
/// 语义：从 stdin 读 RGBA8 裸流（`-f rawvideo -pix_fmt rgba -s WxH -r FPS`），
/// 出 libx264 + yuv420p 的容器文件，`-y` 覆盖输出。
///
/// 注意：本函数**不**决定是否真正调用 ffmpeg；可用性由 [`FfmpegProbe::detect`] 判定，
/// 实际调用见 [`pipe_frames_to_ffmpeg`]。
pub fn ffmpeg_pipe_args(w: u32, h: u32, fps: u32, out: &str) -> Vec<String> {
    vec![
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgba".to_string(),
        "-s".to_string(),
        format!("{w}x{h}"),
        "-r".to_string(),
        fps.to_string(),
        "-i".to_string(),
        "-".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        out.to_string(),
        "-y".to_string(),
    ]
}

/// 把逐帧 RGBA8 经 stdin 喂给外部 `ffmpeg` 编码为 H.264（`libx264`/`yuv420p`）。
///
/// # 三段式失败（C18/C19：在哪 / 什么事 / 怎么改）
/// 探测不可用时**不启动任何进程**，直接返回含
/// "未检测到 ffmpeg / 已降级 MJPEG / 安装 ffmpeg 可获 H.264" 的说明——调用方据此
/// 走内置 MJPEG 出口，链路永不停摆。
///
/// # 可用时
/// `spawn` 子进程 → 逐帧 `write_all`（BrokenPipe 视为子进程提前退出，报错而非静默）
/// → 关闭 stdin → `wait` 并检查退出码。
pub fn pipe_frames_to_ffmpeg(
    frames: &[Vec<u8>],
    w: u32,
    h: u32,
    fps: u32,
    out: &str,
) -> Result<(), String> {
    let probe = FfmpegProbe::detect();
    let ffmpeg = match probe.path.clone() {
        Some(p) if probe.available => p,
        _ => {
            return Err(format!(
                "未检测到 ffmpeg（在哪：extprobe::pipe_frames_to_ffmpeg；PATH 中无 ffmpeg 可执行文件，未启动任何进程）\n  什么事：无法调用外部 H.264 编码器\n  建议：已降级 MJPEG；安装 ffmpeg 可获 H.264（如 `apt install ffmpeg` / `brew install ffmpeg`）"
            ))
        }
    };
    let expect = (w as usize) * (h as usize) * 4;
    for (i, f) in frames.iter().enumerate() {
        if f.len() != expect {
            return Err(format!(
                "帧 {i} 长度 {} ≠ {w}×{h}×4={expect}（在哪：extprobe::pipe_frames_to_ffmpeg；什么事：裸流尺寸不符；建议：按 RGBA8 补齐或改宽高）",
                f.len()
            ));
        }
    }
    let mut cmd = std::process::Command::new(&ffmpeg);
    cmd.args(ffmpeg_pipe_args(w, h, fps, out));
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "ffmpeg 启动失败（在哪：extprobe::pipe_frames_to_ffmpeg；什么事：{e}；建议：检查 {ffmpeg} 是否可执行，或降级 MJPEG）"
        )
    })?;
    let mut write_err: Option<String> = None;
    {
        use std::io::Write;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "ffmpeg stdin 不可用（管道创建失败）".to_string())?;
        for (i, f) in frames.iter().enumerate() {
            if let Err(e) = stdin.write_all(f) {
                // BrokenPipe 等：子进程已提前退出（常见原因：该 ffmpeg 构建缺 libx264）。
                // 不静默——记录后继续收尾，把子进程 stderr 一并带回。
                write_err = Some(format!("写入第 {i} 帧失败：{e}"));
                break;
            }
        }
        // 作用域结束即关闭 stdin（EOF），让 ffmpeg 正常冲刷尾部。
    }
    child.stdin.take();
    let output = child.wait_with_output().map_err(|e| {
        format!(
            "等待 ffmpeg 结束失败（在哪：extprobe::pipe_frames_to_ffmpeg；什么事：{e}；建议：检查进程是否被系统终止）"
        )
    })?;
    let stderr_tail: String = String::from_utf8_lossy(&output.stderr).chars().take(400).collect();
    if let Some(we) = write_err {
        return Err(format!(
            "{we}（在哪：extprobe::pipe_frames_to_ffmpeg；什么事：ffmpeg 提前退出，退出码 {:?}；建议：确认该 ffmpeg 构建含 libx264（`ffmpeg -encoders | grep libx264`），或降级 MJPEG。stderr 摘要：{stderr_tail}）",
            output.status.code()
        ));
    }
    if !output.status.success() {
        return Err(format!(
            "ffmpeg 退出码非 0（在哪：extprobe::pipe_frames_to_ffmpeg；什么事：{:?}；建议：检查参数与 libx264 是否可用；stderr 摘要：{stderr_tail}）",
            output.status.code()
        ));
    }
    Ok(())
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
    fn ffmpeg_pipe_args_layout() {
        // 纯函数布局断言（不启动任何进程）。
        let a = ffmpeg_pipe_args(96, 54, 10, "/tmp/x.mp4");
        let expect = vec![
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "96x54", "-r", "10", "-i", "-", "-c:v",
            "libx264", "-pix_fmt", "yuv420p", "/tmp/x.mp4", "-y",
        ];
        assert_eq!(a, expect.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        // 关键位点逐项核对（防布局漂移）。
        assert_eq!(a[5], "96x54");
        assert_eq!(a[7], "10");
        assert_eq!(a[9], "-");
        assert_eq!(a[11], "libx264");
        assert_eq!(a[13], "yuv420p");
        assert_eq!(a[15], "-y");
    }

    #[test]
    fn pipe_without_ffmpeg_reports_three_part_error() {
        // 构造"无 ffmpeg"环境：PATH 置空后再探测并走管道编码路径。
        // 注意：`pipe_frames_to_ffmpeg` 内部调用 `FfmpegProbe::detect()`（读真实 PATH），
        // 故本测试只在**确定探测不到**时断言三段式；探测到 ffmpeg 时跳过（不 spawn）。
        let probe = FfmpegProbe::detect();
        if probe.is_available() {
            eprintln!("本机有 ffmpeg：跳过'不可用三段式'断言（可用路径由 example 覆盖）");
            return;
        }
        let frame = vec![0u8; 4 * 4 * 4];
        let err = pipe_frames_to_ffmpeg(&[frame], 4, 4, 1, "/tmp/prism_never_written.mp4")
            .unwrap_err();
        for seg in ["未检测到 ffmpeg", "已降级 MJPEG", "安装 ffmpeg 可获 H.264"] {
            assert!(err.contains(seg), "三段式缺少「{seg}」：{err}");
        }
        // 未 spawn：输出文件不应存在。
        assert!(!Path::new("/tmp/prism_never_written.mp4").exists());
    }

    #[test]
    fn builtin_flag() {
        assert!(Codec::Mjpeg.is_builtin());
        assert!(!Codec::H264.is_builtin());
        assert_eq!(Codec::H264.name(), "h264");
    }
}
