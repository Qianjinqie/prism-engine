//! `prism-app render` 子命令：DSL v2 成片直达路径（V3.1 收尾）。
//!
//! # 四个大模式（用户亲自命名，名字与备注原样带上，不许改名）
//! | 大模式 | 备注 | 语义 |
//! |---|---|---|
//! | `一致` | 一致但较慢 | 老路径，逐字节一致，哈希锚 `78366f3e…` 留在这档 |
//! | `均衡` | 适中 | 画质按误差上限，速度与画质居中 |
//! | `快速` | 较快 | 画质按误差上限，速度优先 |
//! | `自定义` | 自定义参数 | 每个开关单独可拨 |
//!
//! 大模式 = 一组预设值，**不是把细项吞掉**：任一细项被拨动即自动成为
//! `自定义`，未拨的细项继续沿用所选大模式的预设。原有
//! `V31Config::legacy()` 对应 `一致` 档。
//!
//! # 容器降级链（复用 E 项 extprobe）
//! 请求 H.264 → 探测 PATH 中 ffmpeg → 可用则管道编码 → 任一环节失败
//! （含未安装）**不静默**：明确告知已降级，改走内置 MJPEG muxer
//! （`prism-mux::view::write_mp4_views`，零外部依赖、永远可用）。

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use prism_gpu::{CpuRefBackend, GpuDevice};
use prism_mux::extprobe::{self, Codec, FfmpegProbe};
use prism_mux::{JpegConfig, TrackSpec};
use prism_mux::view::{views_of_slices, write_mp4_views};
use prism_postfx::adaptive_backend::AdaptiveRtBackend;
use prism_render::{
    Domain, NullV31Sink, QualityTier, ResTier, RtBackend, SegmentTimer, StreamingV31Sink,
    V31Config, V31FrameSink, V31Knobs, V31Mode, render_movie, render_movie_v31,
};

/// CLI 允许全帧驻留的上限（字节）。超过则切流式计时模式（不落盘容器），
/// 避免 720p × 3600 帧 = 13.3 GB 驻留 OOM；**渲染仍是完整 3600 帧**。
const V31_CLI_RESIDENT_BYTES: u64 = 512 * 1024 * 1024;

/// render 子命令用法。
pub const RENDER_USAGE: &str = r#"prism render — 统一顶层入口：一部片子 = 一个 .prism 文件，一条命令出片
（工程二进制名 `prism-app`，`prism` 为同源别名；两者行为逐字节一致）

用法:
  prism render <file.prism> [--out <FILE>] [--codec <h264|mjpeg>]
              [--mode <一致|均衡|快速|自定义>] [--res <TIER>]
              [--frames <N>] [--fps <N>] [--verify <0|1>]
              [--accum <0|1>] [--denoise <0|1>] [--adaptive <0|1>]
              [--gpu <auto|prefer|off>]
              [--spp <N>] [--window <N>] [--bounces <N>] [--min-spp <N>]
              [--fast] [--adaptive <TOTAL>] [--ocio <CONFIG.OCIO>]

四个大模式（名字与备注原样）：
  --mode 一致      备注「一致但较慢」：老路径，逐字节一致；哈希锚
                   78366f3e78323ea260a0dfb606d4f497b06913c1b96500a5ddc1018aa9c32da8
                   留在这档；16 spp 直渲，零复用
  --mode 均衡      备注「适中」：通用路径 + 稀疏网格重建（步长 4，窄重建阈值）
  --mode 快速      备注「较快」：通用路径 + 稀疏网格重建（步长 6，宽重建阈值）
  --mode 自定义    备注「自定义参数」：逐项细拨；步长与阈值按所选细项推导
  缺省为「一致」。大模式是一组预设值，不是把细项吞掉；拨了任一细项
  即自动成为「自定义」，未拨细项仍沿用所选大模式的预设。

精细控制（每个开关独立可拨）：
  --accum <0|1>          时域累积
  --denoise <0|1>        空域降噪
  --adaptive <0|1>       自适应采样预算
  --gpu <auto|prefer|off> GPU 优先桥：auto 有真 GPU 自动启用、无则回落 CPU；
                          prefer 显式要求（回落会记 reset）；off 关闭
  --verify <0|1>         可选误差校验（默认 0，不进热路径；判据完整保留）
  --spp <N>              逐帧真实新采样本数（复用比例细项，缺省 16）
  --window <N>           保留字段（当前不生效）
  --bounces <N>          弹射深度
  --min-spp <N>          自适应预算下限

兼容选项：
  --res <TIER>     分辨率档 320x180 | 640x360 | 720p | 1080p | 4k | <WxH>
                    （缺省 720p，分辨率一档不动，只是设定画幅）
  --quality <Q>    兼容旧值 preview | final | fast；也接受四个大模式名
                    （等价于 --mode）。出现任一 V3.1 选项即启用出片主链路
  --frames <N>     总帧数（duration = frames / fps 覆写）
  --fps <N>        帧率（缺省 60）
  --fast           旧平光档（CpuRefBackend Lambert，对拍基线）
  --adaptive <N>   旧自适应预算档（与 --fast 互斥）
  --ocio <FILE>    config.ocio 色彩管理
  --codec <C>      h264（默认；无 ffmpeg 自动降级 MJPEG）| mjpeg
  --out <FILE>     输出文件（默认 output/render.mp4）
  -h, --help       显示本帮助

说明:
  不给任何 V3.1 选项时走 render_movie 旧入口，输出与 V3.1 之前逐字节一致。
  给 --mode / --res / --quality / --frames / --fps 任一即启用 render_movie_v31：
  四档：一致档透传 RtBackend 直渲（16 spp、哈希锚逐字节不变）；其余三档走
  独立通用路径——第一遍只在稀疏网格节点做本帧真实路径追踪，第二遍按单元
  分类做门控重建（平坦单元双线性、同容器陡变单元 2×2 二级细化、真轮廓与
  跨容器边缘逐像素真实求值），平坦单元不足 25% 时当帧退回单遍逐像素；
  跨帧一概不复用，不冻结、不锁存、不重复帧；
  误差上限判定移到可选校验（--verify 1，默认不进热路径），出片链路不再判红退出。
  回退：PRISM_V31_ACCUM|DENOISE|ADAPTIVE|GPU|OVERRIDE=0 单条关闭；
  PRISM_V31_VERIFY=1 / PRISM_V31_GATE=1 打开可选校验；
  PRISM_V31_STREAM=1 强制流式计时（完整帧数渲染，跳过容器落盘）。

计时口径:
  720p / 4 核 / 端到端单帧（渲染 + 采样与重建 + 编码）用 36 / 360 帧实测，
  外推到 3600 帧一律标注「外推、非实测」；脚本 scripts/v31_perf_gate.sh
  产出三档对照与全域基线。

示例:
  prism-app render demo.prism --mode 一致 --res 720p --frames 36
  prism-app render demo.prism --mode 一致 --res 720p --frames 360 --fps 60
  prism-app render demo.prism --mode 自定义 --res 720p --frames 36 --spp 16"#;

/// render 子命令配置（由参数解析构造）。
struct RenderConfig {
    file: PathBuf,
    out: PathBuf,
    codec: Codec,
    fast: bool,
    adaptive: Option<u32>,
    ocio: Option<PathBuf>,
    /// V3.1 出片配置。`None` = 未出现任何 V3.1 选项 ⇒ 走旧 `render_movie`
    /// 入口，输出逐字节零回归。
    v31: Option<V31Config>,
    /// 最终生效的大模式（用于输出与备注展示）。
    mode: V31Mode,
    /// 精细开关摘要（输出用）。
    knobs: V31Knobs,
    /// 是否出现 V3.1 选项。
    v31_active: bool,
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
    prism_render::reset_domains();

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

    // ---- 3. DSL 编译（语法错误带行号 + 出错行原文回显）----------------------
    let compile_timer = SegmentTimer::start(Domain::DslCompile);
    let film_compile = match prism_dsl::compile_film(&source) {
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
    let dsl_micros: u64 = compile_timer.finish();
    let mut compile = film_compile.compile;

    // ---- 3.4a 外部资产内容哈希锁定校验（D-27）--------------------------------
    let asset_root: std::path::PathBuf = cfg
        .file
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let resolver = prism_dsl::DirResolver::new(&asset_root);
    let verified_assets = match prism_dsl::verify_external_assets(&film_compile.plan.assets, &resolver) {
        Ok(n) => n,
        Err(err) => {
            eprintln!(
                "prism-app render: 在哪：`{file_display}` 第 {} 行\n  什么事：{}",
                err.line, err.message
            );
            if let Some(line_text) = source.lines().nth(err.line.saturating_sub(1)) {
                eprintln!("  {lineno:>4} | {line_text}", lineno = err.line);
            }
            eprintln!("  建议：按上方消息补齐资产或改用 inline 内嵌（缺资产不静默降级、不取默认）");
            return 1;
        }
    };

    if !film_compile.plan.is_empty() {
        println!(
            "单文件凭证 : {} 资产（内容哈希锁定 / {} 外部已校验）/ {} 命名时序通道（渲染前逐帧解算）",
            film_compile.plan.assets.len(),
            verified_assets,
            film_compile.plan.channels.len()
        );
        for c in &film_compile.plan.channels.channels {
            println!("  {:<24} 第 {} 行", c.target, c.line);
        }
    }

    // ---- 3.5 / 3.6 接线凭证 + 全域分域计时 ----------------------------------
    let wiring_timer = SegmentTimer::start(Domain::Wiring);
    let receipts: Vec<prism_app::w2_eval::W2Receipt> =
        prism_app::w2_eval::evaluate(&compile, 0);
    if !receipts.is_empty() {
        let param_total: usize = receipts.iter().map(|r| r.param_keys.len()).sum();
        println!(
            "接线凭证   : {} 族 / {} 参数（生成源头逐帧解算）",
            receipts.len(),
            param_total
        );
    }
    let w3: Vec<prism_app::w3_eval::W3Receipt> = prism_app::w3_eval::evaluate(&compile, 0);
    if !w3.is_empty() {
        let param_total: usize = w3.iter().map(|r| r.param_keys.len()).sum();
        println!(
            "观感凭证   : finish = crafted / {} 族 / {} 参数（渲染前逐帧注入）",
            w3.len(),
            param_total
        );
        for r in &w3 {
            println!("  {:<20} {}", r.family, r.engine_api);
        }
    }
    let wiring_micros: u64 = wiring_timer.finish();
    // 文本 / 音频 / 纹理 / 粒子 / 光栅计时计数器：按接线族把该段耗时登记到
    // 对应域（域计数器 `calls` 记实际族数；时间按族等分，是分摊口径而非
    // 逐族插桩，报告里如实标注）。DSL / 场景翻译计时另由
    // `prism_dsl::wiring::timing_snapshot()` 给出权威口径。
    record_family_domains(&w3, wiring_micros);
    println!(
        "[v31-io] DSL 编译 {} ms / 场景翻译与接线 {} ms / DSL 逐帧解算累计 {} us",
        dsl_micros / 1000,
        wiring_micros / 1000,
        prism_dsl::wiring::timing_snapshot().resolve_micros
    );

    // ---- 3.7 CLI 层覆写 CompileV2Output 的 pub 规格字段 ---------------------
    if let Some(v) = cfg.v31.as_ref() {
        v.apply_to(&mut compile);
    }

    // ---- 4. 渲染器 D-19 双轨 ------------------------------------------------
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
    let mut fast_device = CpuRefBackend;
    let mut rt_device: RtBackend = RtBackend::default();
    let mut adaptive_device = AdaptiveRtBackend::new(cfg.adaptive.unwrap_or(0));
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

    // ---- 4b. 出片主链路 -----------------------------------------------------
    // 流式（大帧数 / PRISM_V31_STREAM=1）：逐帧借用 RGBA8 视图编码 JPEG，
    // 帧归还帧池；循环后一次 mux 落盘。端到端单帧**含编码**，与 36 帧同口径。
    let mut streaming_sink = StreamingV31Sink::default();
    let mut stream_jpegs: Vec<Vec<u8>> = Vec::new();
    let mut v31_streamed: bool = false;
    let deliverable = if let Some(v31cfg) = cfg.v31.as_ref() {
        let frames: u32 = if v31cfg.frames > 0 {
            v31cfg.frames
        } else {
            (compile.fps as f32 * compile.duration).ceil() as u32
        };
        let resident_bytes: u64 = (frames as u64)
            .saturating_mul(compile.width as u64)
            .saturating_mul(compile.height as u64)
            .saturating_mul(4);
        let stream: bool = std::env::var("PRISM_V31_STREAM").ok().as_deref() == Some("1")
            || resident_bytes > V31_CLI_RESIDENT_BYTES;
        v31_streamed = stream;
        if stream {
            let mut enc: StreamingEncodeSink =
                StreamingEncodeSink::new(compile.width, compile.height, 85);
            match render_movie_v31(&compile, device, v31cfg, &mut enc) {
                Ok(d) => {
                    stream_jpegs = enc.jpegs;
                    streaming_sink = enc.inner;
                    d
                }
                Err(err) => {
                    eprintln!("prism-app render (V3.1): {err}");
                    return 1;
                }
            }
        } else {
            let mut null_sink = NullV31Sink;
            match render_movie_v31(&compile, device, v31cfg, &mut null_sink) {
                Ok(d) => d,
                Err(err) => {
                    eprintln!("prism-app render (V3.1): {err}");
                    return 1;
                }
            }
        }
    } else {
        match render_movie(&compile, device) {
            Ok(d) => d,
            Err(err) => {
                eprintln!("prism-app render: {err}");
                return 1;
            }
        }
    };

    if v31_streamed {
        // 真落盘：逐帧 JPEG 样本一次 mux，容器字节与驻留路径同源。
        let spec = TrackSpec {
            width: deliverable.width,
            height: deliverable.height,
            fps: deliverable.fps,
        };
        let encode_timer = SegmentTimer::start(Domain::Encode);
        let muxed = prism_mux::mp4::write_mp4_from_jpegs(&stream_jpegs, spec, None, 0, 0);
        let encode_micros: u64 = encode_timer.finish();
        let bytes: Vec<u8> = match muxed {
            Ok(b) => b,
            Err(err) => {
                eprintln!("prism-app render (V3.1 流式 mux)：{err}");
                return 1;
            }
        };
        if let Some(parent) = cfg.out.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(err) = fs::write(&cfg.out, &bytes) {
            eprintln!("prism-app render (V3.1 流式写盘)：{err}");
            return 1;
        }
        let elapsed = started.elapsed();
        let frames_n: u64 = stream_jpegs.len().max(1) as u64;
        let per_frame_ms: f64 = elapsed.as_secs_f64() * 1000.0 / frames_n as f64;
        println!("prism-app render 完成（V3.1 流式逐帧编码）");
        if let Some(rep) = streaming_sink.report.as_ref() {
            println!("  大模式    : {}（{}）", rep.mode_label, rep.mode_note);
            println!("  帧数      : {} 帧（完整端到端逐帧求值）", rep.frames_rendered);
            println!("  分辨率    : {}x{} @ {} fps", rep.width, rep.height, rep.fps);
            println!("  帧循环    : {:.3} s / 单帧 {:.1} ms", rep.total_micros as f64 / 1.0e6, rep.per_frame_micros as f64 / 1000.0);
            if let Some(v) = rep.verify_pass {
                println!("  可选校验  : {}", if v { "通过" } else { "未通过（仅报告，不阻断出片）" });
            }
        }
        println!("  容器编码  : 内置 MJPEG（{} 帧逐帧零拷贝视图 + 一次 mux）{:.3} s", stream_jpegs.len(), encode_micros as f64 / 1.0e6);
        println!("  输出文件  : {}（{} 字节）", cfg.out.display(), bytes.len());
        println!("  端到端单帧: {:.1} ms（含渲染 + 采样与重建 + 编码；{} 帧实测）", per_frame_ms, stream_jpegs.len());
        for line in prism_render::domain_snapshot().lines() {
            eprintln!("[v31-counter] {line}");
        }
        println!("  总耗时    : {elapsed:.3?}");
        return 0;
    }

    if deliverable.frames.is_empty() {
        eprintln!(
            "prism-app render: 在哪：render_movie 产物\n  什么事：帧序列为空（0 帧）\n  建议：检查 DSL 的 fps / duration / 场景时长是否为正"
        );
        return 1;
    }

    // ---- 5. 写容器（零拷贝视图 + 分段计时）----------------------------------
    // 720p × 3600 帧旧路径会 clone 13.3 GB RGBA8；这里只取 `&[u8]` 借用视图，
    // 指针全程不变（prism_mux::view::verify_view_identity 有身份断言）。
    let frame_slices: Vec<&[u8]> = deliverable
        .frames
        .iter()
        .map(|fb| fb.pixels.as_slice())
        .collect();
    let (w, h, fps) = (deliverable.width, deliverable.height, deliverable.fps);

    // V2.5-B1：有 --ocio 时才物化 owned 帧（LUT 需要就地改写）；
    // 无 --ocio 时零拷贝，帧字节与改动前逐字节相同（判据 V1）。
    let mut ocio_note: Option<String> = None;
    let mut owned_frames: Option<Vec<Vec<u8>>> = None;
    if let Some(cfg_path) = cfg.ocio.as_ref() {
        let display: String = cfg_path.display().to_string();
        match prism_app::ocio_bridge::load_ocio_pipeline(&display) {
            Ok(Some(pipeline)) => {
                let mut owned: Vec<Vec<u8>> = frame_slices.iter().map(|s| s.to_vec()).collect();
                pipeline.apply_frames(&mut owned);
                ocio_note = Some(pipeline.summary());
                owned_frames = Some(owned);
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
    let active_slices: Vec<&[u8]> = match owned_frames.as_ref() {
        Some(owned) => owned.iter().map(|f| f.as_slice()).collect(),
        None => frame_slices.clone(),
    };

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
    let encode_timer = SegmentTimer::start(Domain::Encode);

    match cfg.codec {
        Codec::Mjpeg => {
            codec_used = Codec::Mjpeg;
            if let Err(msg) = write_builtin_mjpeg_views(&active_slices, w, h, fps, &cfg.out) {
                eprintln!(
                    "prism-app render: 在哪：内置 MJPEG muxer 写入 `{out_display}`\n  什么事：{msg}\n  建议：检查磁盘空间与 --out 路径可写性"
                );
                return 1;
            }
        }
        Codec::H264 | Codec::Hevc => {
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
                codec_used = decision.codec;
                if let Err(msg) = write_builtin_mjpeg_views(&active_slices, w, h, fps, &cfg.out) {
                    eprintln!(
                        "prism-app render: 在哪：内置 MJPEG muxer 写入 `{out_display}`\n  什么事：{msg}\n  建议：检查磁盘空间与 --out 路径可写性"
                    );
                    return 1;
                }
            } else {
                // 外部编码器仍收 owned 帧（extprobe 既有签名，不改它的契约）。
                let owned: Vec<Vec<u8>> = active_slices.iter().map(|s| s.to_vec()).collect();
                match extprobe::pipe_frames_to_ffmpeg(&owned, w, h, fps, &out_display) {
                    Ok(()) => codec_used = decision.codec,
                    Err(msg) => {
                        eprintln!(
                            "prism-app render: 外部 {codec} 编码失败，降级内置 MJPEG",
                            codec = decision.codec.name()
                        );
                        eprintln!("  {msg}");
                        degraded_note =
                            Some("外部编码器运行时失败，已降级内置 MJPEG".to_string());
                        codec_used = Codec::Mjpeg;
                        if let Err(msg2) = write_builtin_mjpeg_views(&active_slices, w, h, fps, &cfg.out) {
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
    let encode_micros: u64 = encode_timer.finish();

    // ---- 6. 结果输出 ---------------------------------------------------------------
    let elapsed: std::time::Duration = started.elapsed();
    let frames_n: u64 = deliverable.frames.len().max(1) as u64;
    let end_to_end_per_frame_ms: f64 = elapsed.as_secs_f64() * 1000.0 / frames_n as f64;
    println!("prism-app render 完成");
    println!("  源文件    : {file_display}");
    println!("  标题      : {}", deliverable.title);
    println!("  分辨率    : {w}x{h}");
    println!("  帧率      : {fps} fps");
    println!("  时长      : {} s", deliverable.duration);
    println!("  帧数      : {} 帧", deliverable.frames.len());
    if cfg.v31_active {
        println!("  大模式    : {}（{}）", cfg.mode.label(), cfg.mode.note());
        println!("  精细开关  : {}", cfg.knobs.summary());
    }
    if let Some(v) = cfg.v31.as_ref() {
        println!(
            "  V3.1 规格 : {} / 槽位 {} spp（每个被采样像素的样本预算）/ 内核（{}）/ 降噪 {} / 自适应 {} / 校验 {}",
            v.res.tag(),
            v.spp_per_frame,
            if matches!(cfg.mode, V31Mode::Consistent) {
                "一致档透传 RtBackend（逐字节哈希锚）"
            } else {
                "通用路径：稀疏网格 + 门控重建（零跨帧复用）"
            },
            v.denoise,
            v.adaptive,
            if v.verify { "开" } else { "关（默认）" }
        );
    }
    println!("  渲染器    : {renderer_name}（{renderer_detail}）");
    println!("  容器编码  : {}", codec_used.name());
    if let Some(note) = ocio_note {
        println!("  色彩管理  : {note}");
    }
    if let Some(note) = degraded_note {
        println!("  降级说明  : {note}");
    }
    println!("  输出文件  : {out_display}");
    println!("  分段计时  : 编译 {} ms / 接线 {} ms / 编码 {} ms", dsl_micros / 1000, wiring_micros / 1000, encode_micros / 1000);
    println!(
        "  端到端单帧: {:.1} ms（含渲染 + 采样与重建 + 编码；{} 帧实测）",
        end_to_end_per_frame_ms,
        deliverable.frames.len()
    );
    if let Some(rep) = streaming_sink.report.as_ref() {
        if let Some(v) = rep.verify_pass {
            println!("  可选校验  : {}（判据不阻断出片）", if v { "通过" } else { "未通过" });
        }
    }
    for line in prism_render::domain_snapshot().lines() {
        eprintln!("[v31-counter] {line}");
    }
    println!("  总耗时    : {elapsed:.3?}");
    0
}

/// 流式逐帧编码 sink：帧不驻留，`on_frame` 只借用 `fb.pixels` 的零拷贝视图
/// 编码成 JPEG 样本（远小于 RGBA8 帧），帧随后由 `render_movie_v31` 归还帧池。
/// 预构建 [`prism_mux::jpeg::JpegTables`] 整片复用一次，逐帧不再重建量化/Huffman 表。
struct StreamingEncodeSink {
    inner: StreamingV31Sink,
    jpegs: Vec<Vec<u8>>,
    tables: prism_mux::jpeg::JpegTables,
    width: u32,
    height: u32,
}

impl StreamingEncodeSink {
    fn new(width: u32, height: u32, quality: u8) -> Self {
        StreamingEncodeSink {
            inner: StreamingV31Sink::default(),
            jpegs: Vec::new(),
            tables: prism_mux::jpeg::JpegTables::new(quality),
            width,
            height,
        }
    }
}

impl V31FrameSink for StreamingEncodeSink {
    fn on_frame(&mut self, _frame: u32, fb: &prism_gpu::Framebuffer, _linear: &[[f32; 4]]) {
        // 零拷贝身份：编码器直接借用 `fb.pixels`，不 clone 3.7 MB 帧。
        match prism_mux::jpeg::encode_jpeg_color_with_tables(
            &fb.pixels,
            self.width,
            self.height,
            &self.tables,
        ) {
            Ok(jpeg) => self.jpegs.push(jpeg),
            Err(err) => eprintln!("[v31-stream] 帧 {_frame} JPEG 编码失败（仅报告）：{err}"),
        }
    }
    fn on_record(&mut self, rec: &prism_render::V31FrameRecord) {
        self.inner.on_record(rec);
    }
    fn on_report(&mut self, report: &prism_render::V31MovieReport) {
        self.inner.on_report(report);
    }
}

/// 族名 → 计时域（W3 六族 / 文本 / 音频 / 纹理 / 粒子 / 光栅）。
fn domain_of_family(family: &str) -> Domain {
    let f: String = family.to_ascii_lowercase();
    // "texture" 含子串 "text"，必须先判 Texture 再判 Text，否则纹理族会被
    // 文本族截获（接线耗时登记到错误域）。
    if f.contains("texture") || f.contains("纹") {
        Domain::Texture
    } else if f.contains("text") || f.contains("字") {
        Domain::Text
    } else if f.contains("audio") || f.contains("sound") || f.contains("音") {
        Domain::Audio
    } else if f.contains("particle") || f.contains("粒") {
        Domain::Particles
    } else if f.contains("raster") || f.contains("光栅") {
        Domain::Raster
    } else {
        Domain::Wiring
    }
}

/// 按接线族把接线段耗时登记到对应域（分摊口径，如实标注）。
fn record_family_domains(receipts: &[prism_app::w3_eval::W3Receipt], total_micros: u64) {
    if receipts.is_empty() {
        return;
    }
    let share: u64 = total_micros / receipts.len() as u64;
    for r in receipts {
        prism_render::record_domain(domain_of_family(&r.family), share);
    }
}

/// 内置 MJPEG 出口（**零拷贝视图版**）：RGBA8 帧借用切片 → MP4(MJPEG)
/// 字节 → 落盘。与旧 `write_mp4_jpeg_color(&[Vec<u8>], ...)` 逐字节同源。
fn write_builtin_mjpeg_views(
    frames: &[&[u8]],
    w: u32,
    h: u32,
    fps: u32,
    out: &PathBuf,
) -> Result<(), String> {
    let views = views_of_slices(frames, w, h).map_err(|e| format!("帧视图非法: {e}"))?;
    for (v, src) in views.iter().zip(frames.iter()) {
        prism_mux::view::verify_view_identity(v, src);
        // 出片主链零拷贝身份判据（白名单 + 指针相等，防「声明了但做了拷贝」）。
        prism_render::zero_copy::verify_borrow_chain(src, v.rgba8);
    }
    let bytes: Vec<u8> = write_mp4_views(
        &views,
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
fn parse_args(args: &[String]) -> Result<RenderConfig, String> {
    let mut cfg = RenderConfig {
        file: PathBuf::new(),
        out: PathBuf::from(DEFAULT_OUT),
        codec: Codec::H264,
        fast: false,
        adaptive: None,
        ocio: None,
        v31: None,
        mode: V31Mode::Consistent,
        knobs: V31Knobs::default(),
        v31_active: false,
    };

    let mut v31_res: Option<ResTier> = None;
    let mut v31_frames: Option<u32> = None;
    let mut v31_fps: Option<u32> = None;
    let mut mode_given: bool = false;
    let mut quality_given: bool = false;
    let mut knobs: V31Knobs = V31Knobs::default();

    let mut i: usize = 0;
    while i < args.len() {
        let arg: &str = args[i].as_str();
        let next = |i: &mut usize, name: &str| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("参数 {name} 缺少取值"))
        };
        match arg {
            "--fast" => cfg.fast = true,
            "--adaptive" => {
                let value = next(&mut i, "--adaptive")?;
                let total: u32 = value.parse().map_err(|_| {
                    format!("参数 --adaptive 的取值 `{value}` 不是合法正整数")
                })?;
                if total == 0 {
                    return Err("参数 --adaptive 的总预算必须 ≥ 1".to_string());
                }
                if total > 1 << 24 {
                    return Err(format!(
                        "参数 --adaptive 的总预算 {total} 超过上界 {}（f32 舍入会破坏精确分配）",
                        1 << 24
                    ));
                }
                cfg.adaptive = Some(total);
            }
            "--out" => {
                let value = next(&mut i, "--out")?;
                if value.is_empty() {
                    return Err("参数 --out 的取值不能为空字符串".to_string());
                }
                cfg.out = PathBuf::from(value);
            }
            "--codec" => {
                let value = next(&mut i, "--codec")?;
                cfg.codec = match value.as_str() {
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
                let value = next(&mut i, "--ocio")?;
                if value.is_empty() {
                    return Err("参数 --ocio 的取值不能为空字符串".to_string());
                }
                cfg.ocio = Some(PathBuf::from(value));
            }
            "--res" => {
                let value = next(&mut i, "--res")?;
                v31_res = Some(ResTier::parse(&value).ok_or_else(|| {
                    format!("参数 --res 的取值 `{value}` 非法（允许 320x180 / 640x360 / 720p / 1080p / 4k / <WxH>）")
                })?);
            }
            "--mode" => {
                let value = next(&mut i, "--mode")?;
                cfg.mode = V31Mode::parse(&value).ok_or_else(|| {
                    format!("参数 --mode 的取值 `{value}` 非法（允许 一致 / 均衡 / 快速 / 自定义）")
                })?;
                mode_given = true;
            }
            "--quality" => {
                let value = next(&mut i, "--quality")?;
                quality_given = true;
                if let Some(m) = V31Mode::parse(&value) {
                    cfg.mode = m;
                    mode_given = true;
                } else {
                    let q = QualityTier::parse(&value).ok_or_else(|| {
                        format!("参数 --quality 的取值 `{value}` 非法（允许四个大模式名或 preview / final / fast）")
                    })?;
                    // 旧三档兼容映射：final/preview → 均衡，fast → 快速（旧平光语义保留）。
                    cfg.mode = if q == QualityTier::Fast { V31Mode::Fast } else { V31Mode::Balanced };
                    if q == QualityTier::Fast {
                        cfg.fast = true;
                    }
                }
                if cfg.mode == V31Mode::Fast && value == "fast" {
                    cfg.fast = true;
                }
            }
            "--frames" => {
                let value = next(&mut i, "--frames")?;
                let n: u32 = value.parse().map_err(|_| {
                    format!("参数 --frames 的取值 `{value}` 不是合法正整数")
                })?;
                if n == 0 {
                    return Err("参数 --frames 必须 ≥ 1".to_string());
                }
                v31_frames = Some(n);
            }
            "--fps" => {
                let value = next(&mut i, "--fps")?;
                let n: u32 = value.parse().map_err(|_| {
                    format!("参数 --fps 的取值 `{value}` 不是合法正整数")
                })?;
                if n == 0 {
                    return Err("参数 --fps 必须 ≥ 1".to_string());
                }
                v31_fps = Some(n);
            }
            "--gpu" => {
                let value = next(&mut i, "--gpu")?;
                match value.as_str() {
                    "auto" => {
                        knobs.gpu_auto = Some(true);
                        knobs.gpu_prefer = Some(false);
                    }
                    "prefer" => {
                        knobs.gpu_prefer = Some(true);
                        knobs.gpu_auto = Some(false);
                    }
                    "off" => {
                        knobs.gpu_auto = Some(false);
                        knobs.gpu_prefer = Some(false);
                    }
                    other => return Err(format!("参数 --gpu 的取值 `{other}` 非法（允许 auto / prefer / off）")),
                }
            }
            "--verify" => knobs.verify = Some(parse_bool_flag(&next(&mut i, "--verify")?, "--verify")?),
            "--accum" => knobs.accumulate = Some(parse_bool_flag(&next(&mut i, "--accum")?, "--accum")?),
            "--denoise" => knobs.denoise = Some(parse_bool_flag(&next(&mut i, "--denoise")?, "--denoise")?),
            "--reconstruct" => {
                return Err(
                    "参数 --reconstruct 已随「时空重建」整体删除；时域重投影样本复用已由 --mode 均衡/快速 的几何感知重投影实现，请改用 --mode / --spp"
                        .to_string(),
                );
            }
            "--spp" => knobs.spp_per_frame = Some(parse_positive(&next(&mut i, "--spp")?, "--spp")?),
            "--window" => knobs.window = Some(parse_positive(&next(&mut i, "--window")?, "--window")?),
            "--bounces" => knobs.bounces = Some(parse_positive(&next(&mut i, "--bounces")?, "--bounces")?),
            "--min-spp" => {
                let value = next(&mut i, "--min-spp")?;
                let n: u32 = value.parse().map_err(|_| {
                    format!("参数 --min-spp 的取值 `{value}` 不是合法非负整数")
                })?;
                knobs.min_spp = Some(n);
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
    // ---- V3.1：任一新选项出现即激活；缺省 = 均衡 / 720p / final / 60fps -----
    let active: bool = mode_given
        || quality_given
        || v31_res.is_some()
        || v31_frames.is_some()
        || v31_fps.is_some()
        || knobs.any_dialed();
    cfg.v31_active = active;
    cfg.knobs = knobs;
    if active {
        let res: ResTier = v31_res.unwrap_or(ResTier::R720p);
        let (mut v, final_mode) = knobs.resolve(cfg.mode, res)?;
        cfg.mode = final_mode;
        if let Some(n) = v31_frames {
            v.frames = n;
        }
        if let Some(n) = v31_fps {
            v.fps = n;
        }
        // 未显式给 --frames 时沿用预设的 180 帧；--frames 与 fps 决定 duration。
        cfg.v31 = Some(v);
    }
    Ok(cfg)
}

fn parse_bool_flag(value: &str, name: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        other => Err(format!("参数 {name} 的取值 `{other}` 非法（允许 0 / 1）")),
    }
}

fn parse_positive(value: &str, name: &str) -> Result<u32, String> {
    let n: u32 = value
        .parse()
        .map_err(|_| format!("参数 {name} 的取值 `{value}` 不是合法正整数"))?;
    if n == 0 {
        return Err(format!("参数 {name} 必须 ≥ 1"));
    }
    Ok(n)
}

// ===========================================================================
//  单元测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn defaults_apply_when_only_file_given() {
        let cfg: RenderConfig = parse_args(&argv(&["demo.prism"])).expect("单文件应解析成功");
        assert_eq!(cfg.file, PathBuf::from("demo.prism"));
        assert_eq!(cfg.out, PathBuf::from(DEFAULT_OUT));
        assert!(matches!(cfg.codec, Codec::H264));
        assert!(!cfg.fast);
        assert!(cfg.adaptive.is_none());
        assert!(!cfg.v31_active);
        assert!(cfg.v31.is_none());
    }

    #[test]
    fn four_named_modes_parse_with_verbatim_notes() {
        // 四档全部可用：总槽位恒 16，差异只来自 new_spp（复用比例）。
        for (name, want, want_new) in [
            ("一致", V31Mode::Consistent, 16),
            ("均衡", V31Mode::Balanced, 8),
            ("快速", V31Mode::Fast, 4),
            ("自定义", V31Mode::Custom, 16),
        ] {
            let cfg = parse_args(&argv(&["a.prism", "--mode", name])).expect("四档均可渲染");
            assert_eq!(cfg.mode, want, "{name}");
            assert!(cfg.v31_active);
            let v = cfg.v31.unwrap();
            assert_eq!(v.spp_per_frame, 16, "{name} 总槽位恒 16");
            assert_eq!(v.new_spp, want_new, "{name} 复用比例");
        }
        assert_eq!(V31Mode::Consistent.note(), "一致但较慢");
        assert_eq!(V31Mode::Balanced.note(), "适中");
        assert_eq!(V31Mode::Fast.note(), "较快");
        assert_eq!(V31Mode::Custom.note(), "自定义参数");
    }

    #[test]
    fn dialing_knob_becomes_custom() {
        let cfg = parse_args(&argv(&["a.prism", "--mode", "自定义", "--spp", "8"]))
            .expect("应解析成功");
        assert_eq!(cfg.mode, V31Mode::Custom);
        let v = cfg.v31.unwrap();
        assert_eq!(v.new_spp, 8, "--spp 拨复用比例");
        assert_eq!(v.spp_per_frame, 16, "总槽位恒 16 不动");
        assert!(cfg.knobs.any_dialed());
    }

    #[test]
    fn consistent_rejects_numeric_knobs_at_parse_level() {
        assert!(parse_args(&argv(&[
            "a.prism", "--mode", "一致", "--spp", "4",
        ]))
        .is_err());
        let cfg = parse_args(&argv(&[
            "a.prism", "--mode", "一致", "--verify", "1",
        ]))
        .expect("--verify 不改变像素数值，应允许");
        assert!(cfg.v31.unwrap().verify);
    }

    #[test]
    fn adaptive_and_fast_are_mutually_exclusive() {
        let err = parse_args(&argv(&["a.prism", "--fast", "--adaptive", "4096"]))
            .err()
            .expect("互斥组合必须报错");
        assert!(err.contains("互斥"), "报错要说明互斥：{err}");
        assert!(parse_args(&argv(&["a.prism", "--adaptive"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--adaptive", "0"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--adaptive", "abc"])).is_err());
        assert_eq!(
            parse_args(&argv(&["a.prism", "--adaptive", "4096"]))
                .unwrap()
                .adaptive,
            Some(4096)
        );
    }

    #[test]
    fn fine_switches_and_gpu_modes_parse() {
        let cfg = parse_args(&argv(&[
            "a.prism", "--mode", "自定义", "--gpu", "auto", "--spp", "2",
            "--window", "8", "--denoise", "0", "--verify", "1",
            "--min-spp", "2", "--bounces", "6",
        ]))
        .expect("精细开关应解析成功");
        let v = cfg.v31.unwrap();
        // `--spp` 是复用比例细项：改真实新采样本数 new_spp，总槽位恒 16
        // （等效 spp 不降，见 v31_modes::spp_knob_controls_reuse_ratio_not_total_spp）。
        assert_eq!(v.new_spp, 2);
        assert_eq!(v.spp_per_frame, 16);
        assert_eq!(v.window, 8);
        assert!(!v.denoise);
        assert!(v.verify);
        assert!(v.gpu_auto);
        // 不在此断言 `v.effective_spp() == 16`：该函数沿用 V3.1 早期
        // 「window × spp_per_frame」的窗口累积口径，而 `--window` 现已是保留
        // 字段（不参与渲染），拨成 8 后它只是 128。主链等效 spp 由
        // `V31RenderDevice::effective_spp()` 恒定 16 给出，不受它影响。
        assert_eq!(cfg.mode, V31Mode::Custom);
        assert!(parse_args(&argv(&["a.prism", "--gpu", "wat"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--verify", "2"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--reconstruct", "9"])).is_err());
    }

    #[test]
    fn out_and_codec_parse_and_unknown_flags_rejected() {
        let cfg: RenderConfig =
            parse_args(&argv(&["a.prism", "--out", "out/x.mp4", "--codec", "mjpeg"]))
                .expect("合法参数应解析成功");
        assert_eq!(cfg.out, PathBuf::from("out/x.mp4"));
        assert!(matches!(cfg.codec, Codec::Mjpeg));
        assert!(parse_args(&argv(&["a.prism", "--wat"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--codec", "vp9"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "b.prism"])).is_err());
        assert!(parse_args(&argv(&["--out", "x.mp4"])).is_err());
    }

    #[test]
    fn family_to_domain_mapping() {
        assert_eq!(domain_of_family("text-overlay"), Domain::Text);
        assert_eq!(domain_of_family("audio-loudness"), Domain::Audio);
        assert_eq!(domain_of_family("texture"), Domain::Texture);
        assert_eq!(domain_of_family("particles"), Domain::Particles);
        assert_eq!(domain_of_family("raster"), Domain::Raster);
        assert_eq!(domain_of_family("geometry"), Domain::Wiring);
    }
}
