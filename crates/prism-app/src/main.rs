//! `prism-app` — Prism 引擎命令行入口（"代码即视频/图片"的落盘闭环）。
//!
//! 两种运行模式：
//!
//! 1. **内置演示**（无参数）：组装「环绕相机 + 红球 + teal-orange 电影感调色」
//!    场景，用 CPU 参考后端（软件光线追踪）离线渲染 36 帧（480x270 @ 12fps，
//!    时长 3 秒），并将结果落盘为 PNG 帧序列（`output/frames/`）与首帧预览图
//!    （`output/preview.png`）。本模式保持既有行为与产物路径完全不变。
//!
//! 2. **CLI 渲染**（传入 `.prism` 场景文件）：`std::fs` 读入源码 →
//!    `prism_dsl::compile` 编译为 `Scene` + 输出规格 → CLI 覆盖分辨率/帧率 →
//!    `prism_render` 渲染整段或单帧 → `prism_io` 落盘 → 控制台汇总。
//!    任何失败均以明确错误信息退出（退出码 1）。
//!
//! 运行方式（workspace 根目录下）：
//! ```text
//! cargo run -p prism-app                                    # 内置演示
//! cargo run -p prism-app -- demo.prism                      # 整段渲染
//! cargo run -p prism-app -- demo.prism --frame 1.5          # 单帧预览
//! cargo run -p prism-app -- demo.prism --width 1280 --height 720
//! ```
//!
//! 参数解析为手写实现（仅用 `std`，不引入 clap/serde 等第三方依赖）。

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use prism_core::{Aesthetic, Camera, Light, Material, Scene};
use prism_dsl::OutputFormat;
use prism_gpu::CpuRefBackend;
use prism_math::Vec3;

mod cmd_batch;
mod cmd_critique;
mod cmd_http;
mod cmd_mcp;
mod cmd_pack;
mod cmd_preview;
mod cmd_render;
// 编辑器逻辑在库目标（lib.rs）中，binary 复用同一份实现（避免双编译漂移）。

// ===========================================================================
//  内置演示模式常量（与历史版本保持一致的行为与产物路径）
// ===========================================================================

/// 帧序列输出目录（相对运行时工作目录，运行时 cwd 应为 workspace 根）。
const FRAMES_DIR: &str = "output/frames";
/// 首帧预览图输出路径（相对运行时工作目录）。
const PREVIEW_PATH: &str = "output/preview.png";
/// 渲染输出目录的公共父目录，落盘前需确保其存在。
const OUTPUT_ROOT: &str = "output";

/// 渲染参数：分辨率 480x270。
const WIDTH: u32 = 480;
const HEIGHT: u32 = 270;
/// 采样帧率（帧/秒）。
const FPS: u32 = 12;
/// 视频时长（秒）。
const DURATION: f32 = 3.0;

// ===========================================================================
//  CLI 配置与解析
// ===========================================================================

/// CLI 模式默认输出目录。
const DEFAULT_OUT_DIR: &str = "output";

/// 用法说明文本：`-h`/`--help` 时输出到 stdout（退出码 0）；
/// 用法错误时随错误原因输出到 stderr（退出码 1）。
const USAGE: &str = r#"prism-app — 「代码即视频/图片」命令行渲染器

用法:
  prism-app                        无参数：渲染内置演示场景
  prism-app <file.prism> [选项]    编译并渲染 .prism 场景文件
  prism-app batch <manifest.toml>  批量渲染（笛卡尔积展开）
  prism-app mcp                    MCP 服务器（stdio JSON-RPC）
  prism-app http [--addr <ADDR>]   自写 HTTP/1.1 渲染服务
  prism-app pack <add|list|remove> 数据包市场本地库管理
  prism-app render <file.prism>    DSL v2 成片直达（整段渲染为 MP4）
  prism-app preview <file.prism>   预览闭环（HTTP 预览循环 / 单帧直出）

选项:
  --out <DIR>     输出目录（默认 "output"）
  --frame <T>     只渲染时刻 T 秒的单帧 -> <DIR>/preview.png（优先于整段模式）
  --width <N>     覆盖 DSL 分辨率宽
  --height <N>    覆盖 DSL 分辨率高
  --fps <N>       覆盖 DSL 帧率
  -h, --help      显示本帮助

示例:
  prism-app                                     # 内置演示 -> output/
  prism-app demo.prism                          # 整段渲染 -> output/frames/
  prism-app demo.prism --out render             # 整段渲染 -> render/frames/
  prism-app demo.prism --frame 1.5              # 单帧预览 -> output/preview.png
  prism-app demo.prism --width 1280 --height 720 --fps 30
  prism-app render demo.prism                   # v2 成片直达 -> output/render.mp4
  prism-app preview demo.prism                  # 起预览服务于 127.0.0.1:8790
  prism-app preview demo.prism --frame 1.5      # 单帧直出 -> output/preview_frame.png"#;

/// CLI 渲染配置（由 [`parse_args`] 从 argv 构造）。
#[derive(Debug)]
struct CliConfig {
    /// `.prism` 场景源文件路径。
    file: PathBuf,
    /// 输出目录（默认 `DEFAULT_OUT_DIR`）。
    out_dir: PathBuf,
    /// `--frame T`：单帧渲染时刻（秒）；`None` 表示未指定。
    frame_time: Option<f32>,
    /// `--width N`：覆盖 DSL 分辨率宽；`None` 表示沿用 DSL 值。
    width: Option<u32>,
    /// `--height N`：覆盖 DSL 分辨率高；`None` 表示沿用 DSL 值。
    height: Option<u32>,
    /// `--fps N`：覆盖 DSL 帧率；`None` 表示沿用 DSL 值。
    fps: Option<u32>,
}

/// 参数解析结果错误。
///
/// * [`CliError::Help`] — 显式请求帮助，退出码 0；
/// * [`CliError::Unknown`] — 用法错误，携带原因，退出码 1。
#[derive(Debug)]
enum CliError {
    /// 显式 `-h` / `--help` 请求。
    Help,
    /// 用法错误及原因说明。
    Unknown(String),
}

// ===========================================================================
//  入口
// ===========================================================================

/// 入口：无参数 → 内置演示；带参数 → CLI 渲染；`-h`/`--help` → 用法说明。
///
/// 退出码约定：`0` 成功（含帮助）；`1` 用法错误 / 文件读取失败 /
/// DSL 编译失败 / 渲染落盘失败。
fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();

    // ---- 子命令分发（batch / mcp / http / pack；开发者面）-----------
    if let Some(sub) = raw_args.first().map(String::as_str) {
        let rest: &[String] = &raw_args[1..];
        let code: Option<i32> = match sub {
            "batch" => Some(cmd_batch::run(rest)),
            "mcp" => Some(cmd_mcp::run(rest)),
            "http" => Some(cmd_http::run(rest)),
            "pack" => Some(cmd_pack::run(rest)),
            "critique" => Some(cmd_critique::run(rest)),
            "render" => Some(cmd_render::run(rest)),
            "preview" => Some(cmd_preview::run(rest)),
            _ => None,
        };
        if let Some(code) = code {
            std::process::exit(code);
        }
    }

    match parse_args(&raw_args) {
        // ---- 模式一：无参数 → 内置演示场景（历史行为不变）-------------------
        Ok(None) => {
            if let Err(err) = run_demo() {
                eprintln!("prism-app: 运行失败: {err}");
                std::process::exit(1);
            }
        }
        // ---- 模式二：CLI 渲染 ------------------------------------------------
        Ok(Some(cfg)) => {
            if let Err(msg) = run_cli(cfg) {
                eprintln!("prism-app: {msg}");
                std::process::exit(1);
            }
        }
        // ---- 帮助：stdout + 退出码 0 -----------------------------------------
        Err(CliError::Help) => {
            println!("{USAGE}");
        }
        // ---- 用法错误：stderr + 退出码 1 --------------------------------------
        Err(CliError::Unknown(msg)) => {
            eprintln!("prism-app: {msg}");
            eprintln!("{USAGE}");
            std::process::exit(1);
        }
    }
}

// ===========================================================================
//  模式一：内置演示（保持历史行为与产物路径不变）
// ===========================================================================

/// 组装场景 → CPU 后端渲染 → PNG 落盘 → 控制台输出。
///
/// 所有失败以 `io::Error` 透传给 [`main`] 统一处理。
fn run_demo() -> std::io::Result<()> {
    let started: Instant = Instant::now();

    // ---- 1. 场景组装 ------------------------------------------------------
    // 红色球体置于原点，相机在半径 8 的水平圆轨道上环绕；光源为暖色方向光。
    let scene: Scene = Scene {
        radius: 1.0,
        orbit_center: Vec3::new(0.0, 0.0, 0.0),
        orbit_radius: 8.0,
        material: Material {
            albedo: Vec3::new(192.0 / 255.0, 57.0 / 255.0, 43.0 / 255.0), // #c0392b，与黄金示例逐位一致
            roughness: 0.3,
            metalness: 0.05,
        },
        lights: vec![Light {
            direction: Vec3::new(1.0, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 244.0 / 255.0, 230.0 / 255.0), // #fff4e6，与黄金示例逐位一致
            intensity: 3.0,
        }],
        camera: Camera {
            eye: Vec3::new(0.0, 0.0, -8.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 45.0_f32.to_radians(),
        },
        aesthetic: Aesthetic {
            teal_orange: true,
            contrast: 1.12,
            filmic: true,
        },
            attach: Default::default(),
};

    // ---- 2. CPU 参考后端离线渲染 -------------------------------------------
    // 12fps * 3s = 36 帧；逐帧由 prism-core 编译请求、CpuRefBackend 光追成像。
    let mut device = CpuRefBackend;
    let frames: Vec<prism_gpu::Framebuffer> =
        prism_render::render_video(&mut device, &scene, WIDTH, HEIGHT, FPS, DURATION);

    // ---- 3. 落盘 ------------------------------------------------------------
    // 帧序列目录由 save_frames 递归创建；preview.png 的父目录需先行确保存在。
    fs::create_dir_all(OUTPUT_ROOT)?;
    prism_io::save_frames(FRAMES_DIR, &frames, FPS, DURATION)?;

    let preview_frame: &prism_gpu::Framebuffer = frames.first().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "render_video 未产出任何帧，无法导出预览图",
        )
    })?;
    prism_io::save_png(PREVIEW_PATH, preview_frame)?;

    // ---- 4. 结果输出 ------------------------------------------------------------
    let elapsed: std::time::Duration = started.elapsed();
    println!("prism-app PoC 渲染完成");
    println!("  帧数      : {} 帧 ({} fps x {} s)", frames.len(), FPS, DURATION);
    println!("  分辨率    : {}x{}", WIDTH, HEIGHT);
    println!("  帧序列目录: {FRAMES_DIR}/");
    println!("  预览图    : {PREVIEW_PATH}");
    println!("  总耗时    : {elapsed:.3?}");

    Ok(())
}

// ===========================================================================
//  模式二：CLI 渲染
// ===========================================================================

/// CLI 渲染主流程：读源文件 → DSL 编译 → 覆盖参数 → 渲染 → 落盘 → 输出。
///
/// 两类"硬失败"按规范就地处理（明确报错后以退出码 1 终止）：
/// 1. 场景文件读取失败；
/// 2. DSL 编译失败（`DslError` 的 Display 消息原样输出到 stderr）。
///
/// 其余失败（目录创建、落盘、空帧序列）以 `Err(String)` 交给 [`main`]
/// 统一打印并退出。
fn run_cli(cfg: CliConfig) -> Result<(), String> {
    let started: Instant = Instant::now();

    // ---- 1. 读取 .prism 源文件 ----------------------------------------------
    let file_display: String = cfg.file.display().to_string();
    let source: String = match fs::read_to_string(&cfg.file) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("prism-app: 无法读取场景文件 `{file_display}`: {err}");
            std::process::exit(1);
        }
    };

    // ---- 2. DSL 编译 ----------------------------------------------------------
    let compiled: prism_dsl::CompileOutput = match prism_dsl::compile(&source) {
        Ok(output) => output,
        Err(err) => {
            // DslError 实现了 Display："DSL 第 N 行: ..."。
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    // ---- 3. CLI 覆盖（--width/--height/--fps 有值则覆盖 DSL 结果）------------
    let width: u32 = cfg.width.unwrap_or(compiled.width);
    let height: u32 = cfg.height.unwrap_or(compiled.height);
    let fps: u32 = cfg.fps.unwrap_or(compiled.fps);
    let duration: f32 = compiled.duration;

    // 单帧时刻：--frame 优先于整段模式；否则 DSL format=Png 且未给 --frame
    // 时取 t=0；两者都不满足 → 整段模式。超出时长的时刻钳制到 duration。
    let single_frame_t: Option<f32> = match cfg.frame_time {
        Some(t) => Some(if t > duration {
            eprintln!(
                "prism-app: 提示: --frame 时刻 {t}s 超出时长 {duration}s，已按 {duration}s 渲染"
            );
            duration
        } else {
            t
        }),
        None if matches!(compiled.format, OutputFormat::Png) => Some(0.0),
        None => None,
    };

    // ---- 4. 输出目录与产物路径（Path/PathBuf 跨平台拼接）----------------------
    let frames_dir: PathBuf = cfg.out_dir.join("frames");
    let preview_path: PathBuf = cfg.out_dir.join("preview.png");

    // 落盘前先确保输出目录存在（save_png 不会递归创建父目录）。
    fs::create_dir_all(&cfg.out_dir)
        .map_err(|err| format!("无法创建输出目录 `{}`: {err}", cfg.out_dir.display()))?;

    // ---- 5. 渲染 + 落盘 --------------------------------------------------------
    let mut device = CpuRefBackend;
    let (frame_count, mode_line, artifact_line): (usize, String, String) = match single_frame_t {
        // 单帧模式：渲染时刻 t 的一帧 → <out>/preview.png。
        Some(t) => {
            let fb: prism_gpu::Framebuffer = prism_render::render_frame(
                &mut device,
                &compiled.scene,
                t,
                duration,
                width,
                height,
            );
            prism_io::save_png(&path_str(&preview_path), &fb)
                .map_err(|err| format!("写入预览图 `{}` 失败: {err}", preview_path.display()))?;
            (
                1,
                format!("单帧模式, t={t}s"),
                format!("预览图    : {}", preview_path.display()),
            )
        }
        // 整段模式：按 fps 均匀采样 [0, duration] → <out>/frames/。
        None => {
            let frames: Vec<prism_gpu::Framebuffer> = prism_render::render_video(
                &mut device,
                &compiled.scene,
                width,
                height,
                fps,
                duration,
            );
            if frames.is_empty() {
                return Err(
                    "渲染结果为空（0 帧），无法落盘：请检查 DSL 的 fps/duration 是否大于 0"
                        .to_string(),
                );
            }
            prism_io::save_frames(&path_str(&frames_dir), &frames, fps, duration).map_err(
                |err| format!("写入帧序列 `{}` 失败: {err}", frames_dir.display()),
            )?;
            (
                frames.len(),
                "整段模式".to_string(),
                format!("帧序列目录: {}", frames_dir.display()),
            )
        }
    };

    // ---- 6. 结果输出 ---------------------------------------------------------------
    let elapsed: std::time::Duration = started.elapsed();
    println!("prism-app 渲染完成");
    println!("  源文件    : {file_display}");
    println!("  分辨率    : {width}x{height}");
    println!("  帧率      : {fps} fps");
    println!("  时长      : {duration} s");
    println!("  帧数      : {frame_count} 帧 ({mode_line})");
    println!("  {artifact_line}");
    println!("  总耗时    : {elapsed:.3?}");

    Ok(())
}

// ===========================================================================
//  参数解析（手写，仅支持 `--flag value` 空格分隔形式）
// ===========================================================================

/// 解析命令行参数（零依赖手写实现）。
///
/// 返回值语义：
/// * `Ok(None)` — 无任何参数 → 走内置演示场景；
/// * `Ok(Some(_))` — CLI 渲染配置；
/// * `Err(Help)` — 显式请求帮助（`-h`/`--help`）→ main 打印用法后以 0 退出；
/// * `Err(Unknown)` — 未知参数 / 缺值 / 非法取值 → main 打印原因 + 用法后以 1 退出。
///
/// 约定：选项值必须以空格分隔（`--out render`），不支持 `--out=render`；
/// 同一选项多次出现时后值覆盖前值；位置参数（`.prism` 源文件）最多一个。
fn parse_args(args: &[String]) -> Result<Option<CliConfig>, CliError> {
    if args.is_empty() {
        return Ok(None);
    }

    // -h/--help 在任意位置出现都视为帮助请求（优先于其余校验）。
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Err(CliError::Help);
    }

    let mut cfg: CliConfig = CliConfig {
        file: PathBuf::new(),
        out_dir: PathBuf::from(DEFAULT_OUT_DIR),
        frame_time: None,
        width: None,
        height: None,
        fps: None,
    };

    let mut i: usize = 0;
    while i < args.len() {
        let arg: &str = args[i].as_str();
        match arg {
            "--out" => {
                let value: &str = next_value(args, &mut i, arg)?;
                cfg.out_dir = PathBuf::from(value);
            }
            "--frame" => {
                let value: &str = next_value(args, &mut i, arg)?;
                cfg.frame_time = Some(parse_frame_time(value)?);
            }
            "--width" => {
                let value: &str = next_value(args, &mut i, arg)?;
                cfg.width = Some(parse_positive_u32(arg, value)?);
            }
            "--height" => {
                let value: &str = next_value(args, &mut i, arg)?;
                cfg.height = Some(parse_positive_u32(arg, value)?);
            }
            "--fps" => {
                let value: &str = next_value(args, &mut i, arg)?;
                cfg.fps = Some(parse_positive_u32(arg, value)?);
            }
            _ => {
                // 以 '-' 开头的一律视为未知选项（不支持以 '-' 开头的文件名）。
                if arg.starts_with('-') {
                    return Err(CliError::Unknown(format!("未知参数 `{arg}`")));
                }
                // 位置参数 = .prism 源文件，最多一个。
                if !cfg.file.as_os_str().is_empty() {
                    return Err(CliError::Unknown(format!(
                        "多余的源文件参数 `{arg}`（只能指定一个 .prism 文件）"
                    )));
                }
                cfg.file = PathBuf::from(arg);
            }
        }
        i += 1;
    }

    if cfg.file.as_os_str().is_empty() {
        return Err(CliError::Unknown(
            "缺少 <file.prism> 场景文件参数".to_string(),
        ));
    }
    if cfg.out_dir.as_os_str().is_empty() {
        return Err(CliError::Unknown("输出目录不能为空字符串".to_string()));
    }

    Ok(Some(cfg))
}

/// 取走当前 flag 的值：约定 `--flag value`（空格分隔）形式。
///
/// 调用约定：`*i` 当前指向 flag 本身；成功后 `*i` 已推进到值的位置，
/// 由外层循环再 `+= 1` 前进到下一个参数。
fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, CliError> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| CliError::Unknown(format!("参数 {flag} 缺少取值")))
}

/// 解析一个"必须 ≥ 1 的 u32"选项值（--width/--height/--fps 共用）。
fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, CliError> {
    let n: u32 = value
        .parse()
        .map_err(|_| CliError::Unknown(format!("参数 {flag} 的取值 `{value}` 不是合法整数")))?;
    if n == 0 {
        return Err(CliError::Unknown(format!("参数 {flag} 的取值不能为 0")));
    }
    Ok(n)
}

/// 解析 `--frame` 的时刻值：非负有限数（秒）。
fn parse_frame_time(value: &str) -> Result<f32, CliError> {
    let t: f32 = value
        .parse()
        .map_err(|_| CliError::Unknown(format!("参数 --frame 的取值 `{value}` 不是合法数字")))?;
    if !t.is_finite() || t < 0.0 {
        return Err(CliError::Unknown(format!(
            "参数 --frame 的取值 `{value}` 必须是非负有限数（秒）"
        )));
    }
    Ok(t)
}

/// 把路径转成 `save_png`/`save_frames` 契约要求的 `&str`（非 UTF-8 路径按 lossy 处理）。
fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

// ===========================================================================
//  参数解析单元测试（零第三方依赖，仅 std + #[test]）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 把 `&str` 列表转成 argv 风格的 `String` 向量。
    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn no_args_means_demo_mode() {
        assert!(matches!(parse_args(&[]), Ok(None)));
    }

    #[test]
    fn help_flag_yields_help_error_anywhere() {
        for flag in ["-h", "--help"] {
            assert!(matches!(parse_args(&argv(&[flag])), Err(CliError::Help)));
            assert!(matches!(
                parse_args(&argv(&["demo.prism", flag])),
                Err(CliError::Help)
            ));
        }
    }

    #[test]
    fn defaults_apply_when_only_file_given() {
        let cfg: CliConfig = parse_args(&argv(&["demo.prism"]))
            .ok()
            .flatten()
            .expect("单个位置参数应解析成功");
        assert_eq!(cfg.file, PathBuf::from("demo.prism"));
        assert_eq!(cfg.out_dir, PathBuf::from(DEFAULT_OUT_DIR));
        assert!(cfg.frame_time.is_none());
        assert!(cfg.width.is_none());
        assert!(cfg.height.is_none());
        assert!(cfg.fps.is_none());
    }

    #[test]
    fn all_options_parse_and_last_occurrence_wins() {
        let cfg: CliConfig = parse_args(&argv(&[
            "a.prism",
            "--out",
            "render",
            "--width",
            "1280",
            "--height",
            "720",
            "--fps",
            "30",
            "--width",
            "640", // 重复选项：后值覆盖前值
            "--frame",
            "1.5",
        ]))
        .ok()
        .flatten()
        .expect("合法参数组合应解析成功");
        assert_eq!(cfg.out_dir, PathBuf::from("render"));
        assert_eq!(cfg.width, Some(640));
        assert_eq!(cfg.height, Some(720));
        assert_eq!(cfg.fps, Some(30));
        assert_eq!(cfg.frame_time, Some(1.5));
    }

    #[test]
    fn unknown_flags_are_rejected() {
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--wat"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["-x"])),
            Err(CliError::Unknown(_))
        ));
    }

    #[test]
    fn missing_values_and_invalid_numbers_are_rejected() {
        // 缺少取值。
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--out"])),
            Err(CliError::Unknown(_))
        ));
        // 非整数 / 越界。
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--width", "abc"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--width", "0"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--fps", "-1"])),
            Err(CliError::Unknown(_))
        ));
        // 非法时刻：负数 / NaN / 无穷。
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--frame", "-0.5"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--frame", "nan"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--frame", "inf"])),
            Err(CliError::Unknown(_))
        ));
    }

    #[test]
    fn missing_or_duplicate_positional_is_rejected() {
        assert!(matches!(
            parse_args(&argv(&["--out", "x"])),
            Err(CliError::Unknown(_))
        ));
        assert!(matches!(
            parse_args(&argv(&["a.prism", "b.prism"])),
            Err(CliError::Unknown(_))
        ));
    }

    #[test]
    fn empty_out_dir_is_rejected() {
        assert!(matches!(
            parse_args(&argv(&["a.prism", "--out", ""])),
            Err(CliError::Unknown(_))
        ));
    }
}
