//! `prism-app render` 子命令：DSL v2 成片直达路径。
//!
//! 语义权威：DSL v2 成片直达路径规格。与 v1 的 `prism-app <file.prism>`（帧序列 /
//! 单帧 PNG）不同，本子命令走 **render_movie 直达路径**：
//!
//! ```text
//! 读 .prism 文件 → compile_v2（语法错误带行号）→ render_movie（R5xxx 三段式）
//!   → 写容器（ffmpeg H.264 优先，不可用降级内置 MP4(MJPEG)）
//! ```
//!
//! # 容器降级链（复用 E 项 extprobe）
//! 请求 H.264 → 探测 PATH 中 ffmpeg → 可用则管道编码 → 任一环节失败
//! （含未安装）**不静默**：明确告知已降级，改走内置 MJPEG muxer
//! （`prism-mux::write_mp4_jpeg_color`，零外部依赖、永远可用）。
//!
//! # 错误宪章（C18/C19）
//! 所有失败均为三段式（在哪 / 什么事 / 建议）并以退出码 1 终止：
//! - 文件读取失败：路径 + io 错误 + 建议；
//! - DSL 语法错误：行号 + 消息 + 出错行原文回显（列位置可视化）；
//! - 渲染失败：`RenderError` 的 Display（R5xxx，自带三段式）；
//! - 写容器失败：目标路径 + muxer/io 错误 + 建议。

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use prism_gpu::{CpuRefBackend, GpuDevice};
use prism_mux::extprobe::{self, Codec, FfmpegProbe};
use prism_mux::{write_mp4_jpeg_color, JpegConfig, TrackSpec};
use prism_postfx::adaptive_backend::AdaptiveRtBackend;
use prism_render::{render_movie, RtBackend};

/// render 子命令用法。
pub const RENDER_USAGE: &str = r#"prism-app render — DSL v2 成片直达（整段渲染为 MP4）

用法:
  prism-app render <file.prism> [--out <FILE>] [--codec <h264|mjpeg>]
                  [--fast] [--adaptive <TOTAL>] [--ocio <CONFIG.OCIO>]

参数:
  <file.prism>   DSL v2 场景源文件（首行须为 // prism-dsl 1 版本头）

选项:
  --out <FILE>   输出文件路径（默认 output/render.mp4）
  --codec <C>    容器编码：h264（默认；无 ffmpeg 时自动降级 MJPEG）
                 或 mjpeg（内置 muxer，零外部依赖）
  --fast         快速档：走 CpuRefBackend（单弹射 Lambert 射线投射，
                 无阴影 / 无反射 / 无间接光）。用于出图预览与对拍基线，
                 不是成片质量档
  --adaptive <N> 自适应采样档（R-13）：总预算 N 个样本/帧（npix×8 ≤ N ≤ 2^24），
                 按 pilot σ̂ 逐像素分配 spp（A5），每像素样本连续 ⇒ 与
                 render_pixel(p, spp=n_p) 逐位同语义。与 --fast 互斥
  --ocio <FILE>  V2.5-B1：config.ocio 色彩管理（自研解析 + Q24 定点 1D/3D LUT）。
                 无此开关 / 空 config ⇒ 与既有 ACES 路径逐字节相同；
                 LUT 由 config 同目录的 FileTransform 源装配，零第三方依赖
  -h, --help     显示本帮助

说明:
  本子命令走 render_movie 直达路径（v2 节点图 → 逐帧求值 → 容器），
  与 v1 的帧序列模式（prism-app <file.prism>）是两条独立出口。
  不支持的节点类型（particles / sim / compose）会得到明确的 R5xxx
  报错，而非静默跳过。

  渲染器（D-19 双轨）:
    默认       prism-rt 完整路径追踪（NEE + MIS + RR + 多弹射：
               软阴影 / 镜面高光 / 间接光都在）
    --adaptive prism-postfx::AdaptiveRtBackend（同一路径追踪核 + A5 预算分配）
    --fast     CpuRefBackend 单弹射 Lambert（ambient 0.08 + N·L，平光）

  两档都是确定性实现：同参数两跑逐字节相同。

  例外（既有行为，非本开关引入）: 曲线图元（sdf kind cylinder / cone /
  torus / revolution）在 render_movie 里由 render_curve_frame 直接走
  prism-rt，不经过设备抽象 —— 这类场景 --fast 不生效（两档出图逐字节相同）。

示例:
  prism-app render demo.prism                        # -> output/render.mp4
  prism-app render demo.prism --out out/demo.mp4     # 指定输出文件
  prism-app render demo.prism --codec mjpeg          # 强制内置 MJPEG
  prism-app render demo.prism --fast                 # 快速档（Lambert 对拍基线）"#;

/// render 子命令配置（由参数解析构造）。
struct RenderConfig {
    /// `.prism` v2 源文件路径。
    file: PathBuf,
    /// 输出文件路径（默认 `output/render.mp4`）。
    out: PathBuf,
    /// 请求的容器编码（`h264` 允许降级；`mjpeg` 为终态）。
    codec: Codec,
    /// `--fast`：走 `CpuRefBackend`（Lambert）。`false` = `prism-rt` 路径追踪。
    fast: bool,
    /// `--adaptive <N>`：走 `AdaptiveRtBackend`（总预算 N/帧）。
    /// `None` = 关。与 `fast` 互斥（解析层拒绝）。
    adaptive: Option<u32>,
    /// `--ocio <FILE>`：V2.5-B1 config.ocio 色彩管理路径。
    /// `None` = 无 config ⇒ 既有 ACES 路径逐字节不变。
    ocio: Option<PathBuf>,
}

/// render 子命令默认输出文件（相对运行时工作目录）。
const DEFAULT_OUT: &str = "output/render.mp4";

/// 运行 render 子命令；返回进程退出码（0 成功 / 1 失败）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{RENDER_USAGE}");
        return 0;
    }

    // ---- 1. 参数解析 ------------------------------------------------------
    let cfg: RenderConfig = match parse_args(args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("prism-app render: {msg}");
            eprintln!("{RENDER_USAGE}");
            return 1;
        }
    };

    let started: Instant = Instant::now();

    // ---- 2. 读取 .prism 源文件 ----------------------------------------------
    let file_display: String = cfg.file.display().to_string();
    let source: String = match fs::read_to_string(&cfg.file) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "prism-app render: 在哪：读取场景文件 `{file_display}`\n  什么事：{err}\n  建议：确认路径存在且为 UTF-8 文本（.prism 源码）"
            );
            return 1;
        }
    };

    // ---- 3. DSL v2 编译（语法错误带行号 + 出错行原文回显）--------------------
    let compile = match prism_dsl::compile_v2(&source) {
        Ok(out) => out,
        Err(err) => {
            eprintln!(
                "prism-app render: 在哪：`{file_display}` 第 {} 行\n  什么事：{}",
                err.line, err.message
            );
            if let Some(line_text) = source.lines().nth(err.line.saturating_sub(1)) {
                eprintln!("  {lineno:>4} | {line_text}", lineno = err.line);
            }
            eprintln!("  建议：按上方消息修正语法（v2 首行须为 `// prism-dsl 1` 版本头）");
            return 1;
        }
    };

    // ---- 4. render_movie 直达路径（R5xxx 三段式，绝不静默）-------------------
    // D-19 双轨：默认 prism-rt 完整路径追踪；`--fast` 走 CpuRefBackend（Lambert）。
    // 两档都是 `&mut dyn GpuDevice`，`render_movie` 一行不改。
    // 失败时**不**回退到另一档：要么出片，要么三段式报错。
    let (renderer_name, renderer_detail): (&str, String) = if cfg.fast {
        ("cpu-ref (Lambert)", "单弹射，ambient 0.08 + N·L".to_string())
    } else if let Some(total) = cfg.adaptive {
        let ap = AdaptiveRtBackend::new(total);
        (
            "rt-adaptive",
            format!(
                "自适应采样：N0={}，总预算 {total} 样本/帧（tile {}，精化 {} 轮）",
                ap.ap.n0, ap.ap.tile, ap.ap.rounds
            ),
        )
    } else {
        let rt: RtBackend = RtBackend::default();
        (
            "rt-path-tracing",
            format!("{} spp / {} 弹射", rt.spp, rt.max_bounces),
        )
    };
    // 三个后端都先构造，再按开关取一个 trait object（`render_movie` 只认
    // `&mut dyn GpuDevice`，故此处不需要分支化的调用点）。
    let mut fast_device = CpuRefBackend;
    let mut rt_device: RtBackend = RtBackend::default();
    let mut adaptive_device = AdaptiveRtBackend::new(cfg.adaptive.unwrap_or(0));
    // 自适应档的预算预检：分辨率来自 DSL 编译产物（在 render_movie 之前就知道），
    // 不足时给 R5xxx 式明确报错，而不是让后端产拒绝帧（那会出全黑片）。
    if let Some(total) = cfg.adaptive {
        let npix = (compile.width as usize).saturating_mul(compile.height as usize);
        let need = adaptive_device.min_total(npix);
        if (total as u64) < need {
            eprintln!(
                "prism-app render: 在哪：--adaptive 预算预检（{}×{}）\n  什么事：总预算 {total} 不足（每帧至少 npix×N0 = {need}，N0 = {}）\n  建议：加大 --adaptive 取值，或降低输出分辨率",
                compile.width, compile.height, adaptive_device.ap.n0
            );
            return 1;
        }
    }
    let device: &mut dyn GpuDevice = if cfg.fast {
        &mut fast_device
    } else if cfg.adaptive.is_some() {
        &mut adaptive_device
    } else {
        &mut rt_device
    };
    let deliverable = match render_movie(&compile, device) {
        Ok(d) => d,
        Err(err) => {
            // 绝不回退到另一档：失败就是失败（R5xxx 自带三段式）。
            eprintln!("prism-app render: {err}");
            return 1;
        }
    };
    if deliverable.frames.is_empty() {
        eprintln!(
            "prism-app render: 在哪：render_movie 产物\n  什么事：帧序列为空（0 帧）\n  建议：检查 DSL 的 fps / duration / 场景时长是否为正"
        );
        return 1;
    }

    // ---- 5. 写容器（H.264 优先，失败明确降级内置 MJPEG）----------------------
    let mut frames: Vec<Vec<u8>> = deliverable.frames.iter().map(|fb| fb.pixels.clone()).collect();
    let (w, h, fps) = (deliverable.width, deliverable.height, deliverable.fps);

    // V2.5-B1：config.ocio → Q24 定点 LUT → 落盘前对 RGBA8 帧应用。
    // 无 --ocio / 空 config 时 `load_ocio_pipeline` 返回 None，本段不执行，
    // frames 与改动前逐字节相同（判据 V1）。
    let mut ocio_note: Option<String> = None;
    if let Some(cfg_path) = cfg.ocio.as_ref() {
        let display: String = cfg_path.display().to_string();
        match prism_app::ocio_bridge::load_ocio_pipeline(&display) {
            Ok(Some(pipeline)) => {
                pipeline.apply_frames(&mut frames);
                ocio_note = Some(pipeline.summary());
            }
            Ok(None) => {
                ocio_note = Some(format!("{display}（空 config：沿用既有 ACES 路径）"));
            }
            Err(msg) => {
                eprintln!(
                    "prism-app render: 在哪：--ocio 配置 `{display}`\n  什么事：{msg}\n  建议：核对 config.ocio 语法与 FileTransform 的 LUT 尺寸"
                );
                return 1;
            }
        }
    }

    // 输出文件的父目录确保存在（fs::write 不递归创建）。
    if let Some(parent) = cfg.out.parent() {
        if !parent.as_os_str().is_empty() && fs::create_dir_all(parent).is_err() {
            eprintln!(
                "prism-app render: 在哪：创建输出目录 `{}`\n  什么事：{}\n  建议：换一个有写权限的 --out 路径",
                parent.display(),
                fs::create_dir_all(parent).unwrap_err()
            );
            return 1;
        }
    }

    let out_display: String = cfg.out.display().to_string();
    let codec_used: Codec;
    let mut degraded_note: Option<String> = None;

    match cfg.codec {
        Codec::Mjpeg => {
            codec_used = Codec::Mjpeg;
            if let Err(msg) = write_builtin_mjpeg(&frames, w, h, fps, &cfg.out) {
                eprintln!(
                    "prism-app render: 在哪：内置 MJPEG muxer 写入 `{out_display}`\n  什么事：{msg}\n  建议：检查磁盘空间与 --out 路径可写性"
                );
                return 1;
            }
        }
        Codec::H264 | Codec::Hevc => {
            // 降级链：请求 H264/HEVC → 探测 ffmpeg → 管道编码 → 失败降级 MJPEG。
            let probe: FfmpegProbe = FfmpegProbe::detect();
            let decision = extprobe::decide(cfg.codec, &probe);
            if decision.degraded {
                degraded_note = Some(format!(
                    "未检测到可用的 {} 编码器（PATH 中无 ffmpeg），已降级 {}",
                    decision.requested.name(),
                    decision.codec.name()
                ));
            }
            if decision.codec.is_builtin() {
                // 探测结论本身就是内置 MJPEG：直接走内置出口。
                codec_used = decision.codec;
                if let Err(msg) = write_builtin_mjpeg(&frames, w, h, fps, &cfg.out) {
                    eprintln!(
                        "prism-app render: 在哪：内置 MJPEG muxer 写入 `{out_display}`\n  什么事：{msg}\n  建议：检查磁盘空间与 --out 路径可写性"
                    );
                    return 1;
                }
            } else {
                match extprobe::pipe_frames_to_ffmpeg(&frames, w, h, fps, &out_display) {
                    Ok(()) => codec_used = decision.codec,
                    Err(msg) => {
                        // 外部编码器运行时失败：明确告知后降级内置 MJPEG（永不停摆）。
                        eprintln!(
                            "prism-app render: 外部 {codec} 编码失败，降级内置 MJPEG",
                            codec = decision.codec.name()
                        );
                        eprintln!("  {msg}");
                        degraded_note =
                            Some("外部编码器运行时失败，已降级内置 MJPEG".to_string());
                        codec_used = Codec::Mjpeg;
                        if let Err(msg2) = write_builtin_mjpeg(&frames, w, h, fps, &cfg.out) {
                            eprintln!(
                                "prism-app render: 在哪：内置 MJPEG muxer 写入 `{out_display}`\n  什么事：{msg2}\n  建议：检查磁盘空间与 --out 路径可写性"
                            );
                            return 1;
                        }
                    }
                }
            }
        }
    }

    // ---- 6. 结果输出 ---------------------------------------------------------------
    let elapsed: std::time::Duration = started.elapsed();
    println!("prism-app render 完成");
    println!("  源文件    : {file_display}");
    println!("  标题      : {}", deliverable.title);
    println!("  分辨率    : {w}x{h}");
    println!("  帧率      : {fps} fps");
    println!("  时长      : {} s", deliverable.duration);
    println!("  帧数      : {} 帧", deliverable.frames.len());
    println!("  渲染器    : {renderer_name}（{renderer_detail}）");
    println!("  容器编码  : {}", codec_used.name());
    if let Some(note) = ocio_note {
        println!("  色彩管理  : {note}");
    }
    if let Some(note) = degraded_note {
        println!("  降级说明  : {note}");
    }
    println!("  输出文件  : {out_display}");
    println!("  总耗时    : {elapsed:.3?}");
    0
}

/// 内置 MJPEG 出口：RGBA8 帧序列 → MP4(MJPEG) 字节 → 落盘。
///
/// 复用官方 muxer `write_mp4_jpeg_color`（彩色质量 85；灰度标志关，
/// 与 `prism-author` 的 MP4 出口逐字段一致）。
fn write_builtin_mjpeg(
    frames: &[Vec<u8>],
    w: u32,
    h: u32,
    fps: u32,
    out: &PathBuf,
) -> Result<(), String> {
    let bytes: Vec<u8> = write_mp4_jpeg_color(
        frames,
        TrackSpec { width: w, height: h, fps },
        None,
        0,
        0,
        JpegConfig { quality: 85, grayscale: false },
    )
    .map_err(|err| format!("MP4(MJPEG) 封装失败: {err}"))?;
    fs::write(out, bytes).map_err(|err| format!("写入文件失败: {err}"))
}

/// 解析 render 子命令参数（零依赖手写，`--flag value` 空格分隔形式）。
///
/// 约定：位置参数（`.prism` 文件）恰一个；选项值必须空格分隔（不支持
/// `--out=x`）；同一选项重复出现时后值覆盖前值。
fn parse_args(args: &[String]) -> Result<RenderConfig, String> {
    let mut cfg = RenderConfig {
        file: PathBuf::new(),
        out: PathBuf::from(DEFAULT_OUT),
        codec: Codec::H264,
        fast: false,
        adaptive: None,
        ocio: None,
    };

    let mut i: usize = 0;
    while i < args.len() {
        let arg: &str = args[i].as_str();
        match arg {
            "--fast" => {
                cfg.fast = true;
            }
            "--adaptive" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --adaptive 缺少取值（每帧总预算，正整数）".to_string())?;
                let total: u32 = value.parse().map_err(|_| {
                    format!("参数 --adaptive 的取值 `{value}` 不是合法正整数")
                })?;
                if total == 0 {
                    return Err("参数 --adaptive 的总预算必须 ≥ 1".to_string());
                }
                // render_adaptive 的 assert 上界（f32 舍入会破坏精确分配），
                // CLI 层提前拦，不进 panic 域（宪章 C18）。
                if total > 1 << 24 {
                    return Err(format!(
                        "参数 --adaptive 的总预算 {total} 超过上界 {}（f32 舍入会破坏精确分配）",
                        1 << 24
                    ));
                }
                cfg.adaptive = Some(total);
            }
            "--out" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --out 缺少取值（输出文件路径）".to_string())?;
                if value.is_empty() {
                    return Err("参数 --out 的取值不能为空字符串".to_string());
                }
                cfg.out = PathBuf::from(value);
            }
            "--codec" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --codec 缺少取值（h264 / mjpeg）".to_string())?;
                cfg.codec = match value {
                    "h264" => Codec::H264,
                    "hevc" => Codec::Hevc,
                    "mjpeg" => Codec::Mjpeg,
                    other => {
                        return Err(format!(
                            "参数 --codec 的取值 `{other}` 不支持（允许：h264 / hevc / mjpeg）"
                        ));
                    }
                };
            }
            "--ocio" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --ocio 缺少取值（config.ocio 路径）".to_string())?;
                if value.is_empty() {
                    return Err("参数 --ocio 的取值不能为空字符串".to_string());
                }
                cfg.ocio = Some(PathBuf::from(value));
            }
            other if other.starts_with('-') => {
                return Err(format!("未知参数 `{other}`"));
            }
            other => {
                if !cfg.file.as_os_str().is_empty() {
                    return Err(format!(
                        "多余的源文件参数 `{other}`（只能指定一个 .prism 文件）"
                    ));
                }
                cfg.file = PathBuf::from(other);
            }
        }
        i += 1;
    }

    if cfg.file.as_os_str().is_empty() {
        return Err("缺少 <file.prism> 场景文件参数".to_string());
    }
    if cfg.fast && cfg.adaptive.is_some() {
        return Err("参数 --fast 与 --adaptive 互斥（前者是 Lambert 对拍基线，后者是完整路径追踪 + 自适应预算）".to_string());
    }
    Ok(cfg)
}

// ===========================================================================
//  单元测试（解析逻辑；端到端出片见 tests/ 集成测试与手工 CLI 实跑）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 把 `&str` 列表转成 argv 风格的 `String` 向量。
    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn defaults_apply_when_only_file_given() {
        let cfg: RenderConfig = parse_args(&argv(&["demo.prism"])).expect("单文件应解析成功");
        assert_eq!(cfg.file, PathBuf::from("demo.prism"));
        assert_eq!(cfg.out, PathBuf::from(DEFAULT_OUT));
        assert!(matches!(cfg.codec, Codec::H264));
        // D-19 默认档 = prism-rt 路径追踪（`--fast` 关着）。
        assert!(!cfg.fast, "缺省必须走 prism-rt（fast = false）");
        assert!(cfg.adaptive.is_none(), "缺省必须关自适应（R-13 档是显式 opt-in）");
    }

    #[test]
    fn adaptive_flag_parses_and_validates() {
        // 位置无关 + 数值解析。
        let cfg = parse_args(&argv(&["--adaptive", "4096", "a.prism"]))
            .expect("应解析成功");
        assert_eq!(cfg.adaptive, Some(4096));
        let cfg = parse_args(&argv(&["a.prism", "--adaptive", "1048576"]))
            .expect("应解析成功");
        assert_eq!(cfg.adaptive, Some(1_048_576));
        // 缺值 / 非数 / 零 / 越上界，全部明确报错。
        assert!(parse_args(&argv(&["a.prism", "--adaptive"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--adaptive", "abc"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--adaptive", "0"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--adaptive", &format!("{}", (1 << 24) + 1)])).is_err());
        // 上界本身合法（render_adaptive 的 assert 上界 = 2^24）。
        let cfg = parse_args(&argv(&["a.prism", "--adaptive", &format!("{}", 1 << 24)]))
            .expect("2^24 应是合法上界");
        assert_eq!(cfg.adaptive, Some(1 << 24));
    }

    #[test]
    fn adaptive_and_fast_are_mutually_exclusive() {
        let err = parse_args(&argv(&["a.prism", "--fast", "--adaptive", "4096"]))
            .err()
            .expect("互斥组合必须报错");
        assert!(err.contains("互斥"), "报错要说明互斥：{err}");
    }

    #[test]
    fn fast_flag_parses() {
        // 位置无关：`--fast` 在文件前 / 后都成立。
        assert!(parse_args(&argv(&["--fast", "a.prism"])).expect("应解析成功").fast);
        assert!(parse_args(&argv(&["a.prism", "--fast"])).expect("应解析成功").fast);
        // 与其它选项共存。
        let cfg: RenderConfig =
            parse_args(&argv(&["a.prism", "--fast", "--codec", "mjpeg"])).expect("应解析成功");
        assert!(cfg.fast);
        assert!(matches!(cfg.codec, Codec::Mjpeg));
        // 重复出现幂等。
        assert!(parse_args(&argv(&["a.prism", "--fast", "--fast"])).expect("应解析成功").fast);
    }

    #[test]
    fn out_and_codec_parse() {
        let cfg: RenderConfig =
            parse_args(&argv(&["a.prism", "--out", "out/x.mp4", "--codec", "mjpeg"]))
                .expect("合法参数应解析成功");
        assert_eq!(cfg.out, PathBuf::from("out/x.mp4"));
        assert!(matches!(cfg.codec, Codec::Mjpeg));
    }

    #[test]
    fn unknown_flag_and_bad_codec_rejected() {
        assert!(parse_args(&argv(&["a.prism", "--wat"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--codec", "vp9"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--codec"])).is_err());
    }

    #[test]
    fn missing_or_duplicate_positional_rejected() {
        assert!(parse_args(&argv(&[])).is_err());
        assert!(parse_args(&argv(&["--out", "x.mp4"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "b.prism"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--out", ""])).is_err());
    }
}
