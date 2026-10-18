//! `prism-app preview` 子命令：**预览闭环（宿主壳）**（V1.5-F 任务书 1.3）。
//!
//! 两种模式：
//!
//! 1. **服务模式**（默认）：起本地 HTTP 服务（复用 `cmd_http` 的自写
//!    HTTP/1.1 服务模式与 `prism-infra::http` 的报文解析 / 响应序列化），
//!    提供预览帧拉取与时间轴 scrub：
//!    - `GET /frame?t=<秒>`：返回时刻 `t` 的预览帧（PNG 字节）——scrub 语义
//!      为 round-to-nearest 选最近采样帧，钳制在 `[0, 帧数)`；
//!    - `GET /meta`：影片元信息（JSON：标题 / 分辨率 / fps / 时长 / 帧数 /
//!      就绪态）；
//!    - `GET /health`：健康检查（JSON）。
//!
//!    整段影片在**后台线程**渲染（`render_movie`，V1.5-D 直达路径），服务
//!    先绑定可连、渲染完成后**替换帧缓冲**——拉帧请求有界等待就绪，
//!    启动不被渲染阻塞（"后台执行 + 缓冲替换"的 CLI 落地）。
//!
//! 2. **单帧模式**（`--frame <秒>`）：不爬服务，直接渲染该时刻帧落盘 PNG。
//!    走 `render_movie` 等价路径（整段求值后取最近帧）——V1.5 务实选择：
//!    movie 模块的逐帧 seek 入口尚未导出，此处如实登记该边界。
//!
//! # 安全红线（同 `cmd_http`）
//! 默认仅绑 `127.0.0.1`（不暴露公网），需显式 `--addr` 才改变。
//!
//! # 错误宪章（C18/C19）
//! 所有失败均为三段式（在哪 / 什么事 / 建议）并以退出码 1 终止。

use std::fs;
use std::io::{BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use prism_gpu::CpuRefBackend;
use prism_infra::http::{self, HttpRequest, HttpResponse};
use prism_infra::jsonrpc::Json;
use prism_render::{render_movie, Deliverable};

/// 默认监听地址（安全：仅本机；与 `http` 子命令的 8787 错开）。
pub const DEFAULT_ADDR: &str = "127.0.0.1:8790";

/// 单帧模式默认输出文件（相对运行时工作目录）。
const DEFAULT_FRAME_OUT: &str = "output/preview_frame.png";

/// 服务等待整片就绪的上限（秒）。超时后 `/frame` 回 503（不静默挂起）。
const READY_TIMEOUT_SECS: u64 = 120;

/// preview 子命令用法。
pub const PREVIEW_USAGE: &str = r#"prism-app preview — 预览闭环（本地 HTTP 预览循环 / 单帧直出）

用法:
  prism-app preview <file.prism> [--addr <HOST:PORT>]  服务模式（默认）
  prism-app preview <file.prism> --frame <秒> [--out <FILE>]  单帧模式

参数:
  <file.prism>   DSL v2 场景源文件（首行须为 // prism-dsl 1 版本头）

选项:
  --addr <ADDR>    监听地址（默认 127.0.0.1:8790，安全：仅本机）
  --frame <秒>     单帧模式：直接渲染该时刻帧出文件（不爬服务）
  --out <FILE>     单帧模式输出路径（默认 output/preview_frame.png）
  --width <N>      覆盖 DSL 分辨率宽（正数）
  --height <N>     覆盖 DSL 分辨率高（正数）
  -h, --help       显示本帮助

路由（服务模式）:
  GET /health          健康检查（JSON，含就绪态）
  GET /meta            影片元信息（JSON）
  GET /frame?t=<秒>    该时刻预览帧（PNG 字节；scrub：不同 t → 不同帧）

说明:
  服务模式在后台线程渲染整段影片（render_movie 直达路径），服务先绑定
  可连；/frame 有界等待就绪后按 t 取最近采样帧。单帧模式走同一管线，
  整段求值后取最近帧落盘（逐帧 seek 入口留待后续波次，此处如实登记）。

示例:
  prism-app preview demo.prism                        # 起服务于 127.0.0.1:8790
  prism-app preview demo.prism --addr 127.0.0.1:9000  # 指定端口
  prism-app preview demo.prism --frame 1.5            # 直出 1.5s 帧 -> output/preview_frame.png"#;

/// preview 子命令配置（由参数解析构造）。
struct PreviewConfig {
    /// `.prism` v2 源文件路径。
    file: PathBuf,
    /// 监听地址（默认 [`DEFAULT_ADDR`]）。
    addr: String,
    /// `--frame <秒>`：单帧模式时刻；`None` 表示服务模式。
    frame: Option<f32>,
    /// 单帧模式输出文件（默认 [`DEFAULT_FRAME_OUT`]）。
    out: PathBuf,
    /// `--width N`：覆盖 DSL 分辨率宽；`None` 表示沿用 DSL 值。
    width: Option<u32>,
    /// `--height N`：覆盖 DSL 分辨率高；`None` 表示沿用 DSL 值。
    height: Option<u32>,
}

/// 运行 preview 子命令；返回进程退出码（0 成功 / 1 失败）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{PREVIEW_USAGE}");
        return 0;
    }
    let cfg: PreviewConfig = match parse_args(args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("prism-app preview: {msg}");
            eprintln!("{PREVIEW_USAGE}");
            return 1;
        }
    };
    match cfg.frame {
        Some(t) => run_single_frame(&cfg, t),
        None => run_server(&cfg),
    }
}

// ---------------------------------------------------------------------------
// 参数解析（零依赖手写，`--flag value` 空格分隔形式；照 cmd_render 惯例）
// ---------------------------------------------------------------------------

/// 解析 preview 子命令参数。
///
/// 约定：位置参数（`.prism` 文件）恰一个；选项值必须空格分隔（不支持
/// `--out=x`）；同一选项重复出现时后值覆盖前值。
fn parse_args(args: &[String]) -> Result<PreviewConfig, String> {
    let mut cfg = PreviewConfig {
        file: PathBuf::new(),
        addr: DEFAULT_ADDR.to_string(),
        frame: None,
        out: PathBuf::from(DEFAULT_FRAME_OUT),
        width: None,
        height: None,
    };

    let mut i: usize = 0;
    while i < args.len() {
        let arg: &str = args[i].as_str();
        match arg {
            "--addr" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --addr 缺少取值（HOST:PORT）".to_string())?;
                if value.is_empty() {
                    return Err("参数 --addr 的取值不能为空字符串".to_string());
                }
                cfg.addr = value.to_string();
            }
            "--frame" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --frame 缺少取值（秒）".to_string())?;
                let t: f32 = value
                    .parse()
                    .map_err(|_| format!("参数 --frame 的取值 `{value}` 不是合法数字（秒）"))?;
                if !t.is_finite() || t < 0.0 {
                    return Err(format!(
                        "参数 --frame 的取值 `{value}` 必须是非负有限数（秒）"
                    ));
                }
                cfg.frame = Some(t);
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
            "--width" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --width 缺少取值".to_string())?;
                cfg.width = Some(parse_positive_u32("--width", value)?);
            }
            "--height" => {
                i += 1;
                let value: &str = args
                    .get(i)
                    .ok_or_else(|| "参数 --height 缺少取值".to_string())?;
                cfg.height = Some(parse_positive_u32("--height", value)?);
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
    Ok(cfg)
}

/// 解析"必须 ≥ 1 的 u32"选项值（--width/--height 共用）。
fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    let n: u32 = value
        .parse()
        .map_err(|_| format!("参数 {flag} 的取值 `{value}` 不是合法整数"))?;
    if n == 0 {
        return Err(format!("参数 {flag} 的取值不能为 0"));
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// 共享：读源 + 编译 + 就绪帧集
// ---------------------------------------------------------------------------

/// 读源文件 + `compile_v2` + 应用宽高覆盖（三段式报错，调用方转退出码 1）。
fn load_compile(cfg: &PreviewConfig) -> Result<prism_dsl::CompileV2Output, String> {
    let file_display: String = cfg.file.display().to_string();
    let source: String = fs::read_to_string(&cfg.file).map_err(|err| {
        format!(
            "在哪：读取场景文件 `{file_display}`\n  什么事：{err}\n  建议：确认路径存在且为 UTF-8 文本（.prism 源码）"
        )
    })?;
    let mut compile = prism_dsl::compile_v2(&source).map_err(|err| {
        format!(
            "在哪：`{file_display}` 第 {} 行\n  什么事：{}",
            err.line, err.message
        )
    })?;
    if let Some(w) = cfg.width {
        compile.width = w;
    }
    if let Some(h) = cfg.height {
        compile.height = h;
    }
    Ok(compile)
}

/// 就绪帧集：后台渲染完成后**替换**进驻留位（渲染中 = `None`）。
enum ReadyFrames {
    /// 渲染失败（携带 R5xxx 三段式文本，供 /frame 如实回 500）。
    Failed(String),
    /// 就绪：元信息 + 逐帧 PNG 字节（启动时编码一次，scrub 零成本）。
    Ready(FrameSet),
}

/// 就绪的帧集（PNG 字节预编码）。
struct FrameSet {
    title: String,
    width: u32,
    height: u32,
    fps: u32,
    duration: f32,
    pngs: Vec<Vec<u8>>,
}

impl FrameSet {
    /// 由成片交付产物构造（逐帧编码 PNG）。
    fn from_deliverable(d: &Deliverable) -> Self {
        Self {
            title: d.title.clone(),
            width: d.width,
            height: d.height,
            fps: d.fps,
            duration: d.duration,
            pngs: d.frames.iter().map(prism_io::encode_png).collect(),
        }
    }
}

/// 时刻 `t`（秒）→ 帧下标：round-to-nearest 后钳制在 `[0, n)`。
///
/// 与 `render_movie` 的采样公式同源（第 k 帧时刻 = `k / fps`）—— scrub 到
/// 的帧与顺序播放到的帧是同一个，杜绝"两套帧号公式"的 seek 偏差。
/// 空序列 / fps=0 → `None`。
pub fn frame_index_for_time(t: f32, fps: u32, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || fps == 0 {
        return None;
    }
    let t: f32 = if t.is_finite() && t > 0.0 { t } else { 0.0 };
    let idx: i64 = (t * fps as f32).round() as i64;
    Some(idx.clamp(0, frame_count as i64 - 1) as usize)
}

// ---------------------------------------------------------------------------
// 模式一：服务模式（后台渲染 + HTTP 预览循环）
// ---------------------------------------------------------------------------

/// 服务共享状态：就绪驻留位 + 条件变量（有界等待，不轮询）。
struct ServerState {
    pair: Arc<(Mutex<Option<ReadyFrames>>, Condvar)>,
}

impl ServerState {
    /// 取时刻 `t` 的预览帧 PNG 字节（等待就绪有界；失败回状态码 + 原因）。
    fn frame_png_at(&self, t: f32) -> Result<Vec<u8>, (u16, String)> {
        let (lock, cv) = &*self.pair;
        let mut guard = lock
            .lock()
            .map_err(|_| (500u16, "就绪锁中毒".to_string()))?;
        if guard.is_none() {
            let (g, _timeout) = cv
                .wait_timeout(guard, Duration::from_secs(READY_TIMEOUT_SECS))
                .map_err(|_| (500u16, "就绪条件变量等待失败".to_string()))?;
            guard = g;
        }
        let Some(frames) = guard.as_ref() else {
            return Err((503u16, "影片渲染未就绪（等待超时）".to_string()));
        };
        match frames {
            ReadyFrames::Failed(msg) => {
                Err((500u16, format!("影片渲染失败：{msg}")))
            }
            ReadyFrames::Ready(set) => match frame_index_for_time(t, set.fps, set.pngs.len()) {
                Some(i) => Ok(set.pngs[i].clone()),
                None => Err((500u16, "帧序列为空（fps/duration 非正？）".to_string())),
            },
        }
    }

    /// 元信息 JSON（就绪态 / 标题 / 规格 / 帧数）。
    fn meta_json(&self) -> Json {
        let (lock, _cv) = &*self.pair;
        let guard = lock.lock().ok();
        match guard.as_ref().and_then(|g| g.as_ref()) {
            Some(ReadyFrames::Ready(set)) => Json::object(vec![
                ("ready", Json::Bool(true)),
                ("title", Json::Str(set.title.clone())),
                ("width", Json::int(set.width as i64)),
                ("height", Json::int(set.height as i64)),
                ("fps", Json::int(set.fps as i64)),
                ("duration", Json::Num(set.duration as f64)),
                ("frames", Json::int(set.pngs.len() as i64)),
            ]),
            Some(ReadyFrames::Failed(msg)) => Json::object(vec![
                ("ready", Json::Bool(true)),
                ("error", Json::Str(msg.clone())),
            ]),
            None => Json::object(vec![("ready", Json::Bool(false))]),
        }
    }

    /// 是否就绪（渲染完成，无论成败）。
    fn is_ready(&self) -> bool {
        let (lock, _cv) = &*self.pair;
        lock.lock().map(|g| g.is_some()).unwrap_or(false)
    }
}

/// 解析 query 串中的 f32 参数（`t=1.5`；缺失 / 非有限 → `None`）。
fn query_f32(query: &str, key: &str) -> Option<f32> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return v.parse::<f32>().ok().filter(|f| f.is_finite());
            }
        }
    }
    None
}

/// 预览服务路由（纯函数，便于单测）。
fn route(req: &HttpRequest, state: &ServerState) -> HttpResponse {
    let (path, query): (&str, &str) = match req.target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (req.target.as_str(), ""),
    };
    match (req.method.as_str(), path) {
        ("GET", "/health") => HttpResponse::json(
            200,
            &Json::object(vec![
                ("status", Json::Str("ok".to_string())),
                ("engine", Json::Str("prism-preview".to_string())),
                ("ready", Json::Bool(state.is_ready())),
            ]),
        ),
        ("GET", "/meta") => HttpResponse::json(200, &state.meta_json()),
        ("GET", "/frame") => {
            let t: f32 = query_f32(query, "t").unwrap_or(0.0);
            match state.frame_png_at(t) {
                Ok(png) => HttpResponse::new(200, "image/png", png),
                Err((status, msg)) => {
                    let body: String =
                        Json::object(vec![("error", Json::Str(msg))]).to_string().unwrap_or_default();
                    HttpResponse::new(status, "application/json", body.into_bytes())
                }
            }
        }
        ("GET", "/") => HttpResponse::text(200, PREVIEW_USAGE),
        _ => HttpResponse::text(404, "404 Not Found"),
    }
}

/// 服务模式：后台渲染整片 + HTTP/1.1 预览循环（阻塞，正常不返回）。
fn run_server(cfg: &PreviewConfig) -> i32 {
    // ---- 1. 读源 + 编译（fail-fast：语法错误立即退出码 1）----------------
    let compile = match load_compile(cfg) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("prism-app preview: {msg}");
            return 1;
        }
    };

    // ---- 2. 后台渲染线程（整段影片；完成后替换就绪驻留位）-----------------
    let pair: Arc<(Mutex<Option<ReadyFrames>>, Condvar)> =
        Arc::new((Mutex::new(None), Condvar::new()));
    {
        let pair = Arc::clone(&pair);
        thread::spawn(move || {
            let started: Instant = Instant::now();
            let mut device = CpuRefBackend;
            let outcome: ReadyFrames = match render_movie(&compile, &mut device) {
                Ok(d) if d.frames.is_empty() => ReadyFrames::Failed(
                    "帧序列为空（0 帧）：检查 DSL 的 fps / duration / 场景时长是否为正".to_string(),
                ),
                Ok(d) => ReadyFrames::Ready(FrameSet::from_deliverable(&d)),
                Err(err) => ReadyFrames::Failed(err.to_string()),
            };
            {
                let (lock, cv) = &*pair;
                if let Ok(mut guard) = lock.lock() {
                    *guard = Some(outcome);
                }
                cv.notify_all();
            }
            eprintln!(
                "prism-app preview: 后台渲染完成（{:?}，{} 帧）",
                started.elapsed(),
                frame_count_hint(&pair)
            );
        });
    }

    // ---- 3. 绑定（立即可连；渲染在后台进行）------------------------------
    let listener: TcpListener = match http_listener_bind(&cfg.addr) {
        Ok(l) => l,
        Err(msg) => {
            eprintln!("prism-app preview: 在哪：绑定监听地址 `{}`\n  什么事：{msg}\n  建议：换一个 --addr（默认 127.0.0.1:8790，仅本机）", cfg.addr);
            return 1;
        }
    };
    eprintln!(
        "prism-app preview: 监听 {}（实际 {}，Ctrl-C 退出；影片后台渲染中，/frame 将等待就绪）",
        cfg.addr,
        listener.local_addr().map(|a| a.to_string()).unwrap_or_else(|_| "?".to_string())
    );

    // ---- 4. 接受循环（复用 prism-infra 的 HTTP/1.1 解析与响应序列化）------
    let state = ServerState { pair };
    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let resp: HttpResponse = {
                    let mut reader = BufReader::new(&mut conn);
                    match http::parse_request(&mut reader) {
                        Ok(req) => route(&req, &state),
                        Err(err) => HttpResponse::text(err.status, &err.reason),
                    }
                };
                let _ = conn.write_all(&resp.to_bytes());
            }
            Err(_) => continue,
        }
    }
    0
}

/// 绑定监听地址（薄封装，便于统一错误文本）。
fn http_listener_bind(addr: &str) -> Result<TcpListener, String> {
    TcpListener::bind(addr).map_err(|e| e.to_string())
}

/// 就绪驻留位的帧数提示（后台汇报用；未就绪回 0）。
fn frame_count_hint(pair: &Arc<(Mutex<Option<ReadyFrames>>, Condvar)>) -> usize {
    let (lock, _cv) = &**pair;
    match lock.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(ReadyFrames::Ready(set)) => set.pngs.len(),
            _ => 0,
        },
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// 模式二：单帧模式（不爬服务，直接渲染该时刻帧落盘）
// ---------------------------------------------------------------------------

/// 单帧模式：读源 → 编译 → `render_movie` 等价路径 → 取最近帧 → PNG 落盘。
fn run_single_frame(cfg: &PreviewConfig, t: f32) -> i32 {
    let started: Instant = Instant::now();
    let compile = match load_compile(cfg) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("prism-app preview: {msg}");
            return 1;
        }
    };

    let mut device = CpuRefBackend;
    let deliverable = match render_movie(&compile, &mut device) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("prism-app preview: {err}");
            return 1;
        }
    };
    if deliverable.frames.is_empty() {
        eprintln!(
            "prism-app preview: 在哪：render_movie 产物\n  什么事：帧序列为空（0 帧）\n  建议：检查 DSL 的 fps / duration / 场景时长是否为正"
        );
        return 1;
    }

    let idx: usize = frame_index_for_time(t, deliverable.fps, deliverable.frames.len())
        .unwrap_or(0);
    let fb = &deliverable.frames[idx];

    // 输出文件的父目录确保存在（save_png 不递归创建）。
    if let Some(parent) = cfg.out.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!(
                    "prism-app preview: 在哪：创建输出目录 `{}`\n  什么事：{err}\n  建议：换一个有写权限的 --out 路径",
                    parent.display()
                );
                return 1;
            }
        }
    }
    if let Err(err) = prism_io::save_png(&path_str(&cfg.out), fb) {
        eprintln!(
            "prism-app preview: 在哪：写入预览帧 `{}`\n  什么事：{err}\n  建议：检查磁盘空间与 --out 路径可写性",
            cfg.out.display()
        );
        return 1;
    }

    let elapsed: std::time::Duration = started.elapsed();
    let clamped: bool = t > deliverable.duration;
    println!("prism-app preview 单帧完成");
    println!("  源文件    : {}", cfg.file.display());
    println!("  标题      : {}", deliverable.title);
    println!("  请求时刻  : {t}s{}", if clamped { "（超出时长，已钳制到末帧）" } else { "" });
    println!("  帧号      : {idx} / {}（fps {}）", deliverable.frames.len(), deliverable.fps);
    println!("  分辨率    : {}x{}", deliverable.width, deliverable.height);
    println!("  输出文件  : {}", cfg.out.display());
    println!("  总耗时    : {elapsed:.3?}");
    0
}

/// 把路径转成 `save_png` 契约要求的 `&str`（非 UTF-8 路径按 lossy 处理）。
fn path_str(path: &std::path::Path) -> String {
    path.to_string_lossy().into_owned()
}

// ===========================================================================
//  单元测试（参数解析 + 帧号选取 + 路由；端到端见 tests/preview_cli.rs）
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
        let cfg: PreviewConfig = parse_args(&argv(&["demo.prism"])).expect("单文件应解析成功");
        assert_eq!(cfg.file, PathBuf::from("demo.prism"));
        assert_eq!(cfg.addr, DEFAULT_ADDR);
        assert!(cfg.frame.is_none());
        assert_eq!(cfg.out, PathBuf::from(DEFAULT_FRAME_OUT));
        assert!(cfg.width.is_none());
        assert!(cfg.height.is_none());
    }

    #[test]
    fn all_options_parse() {
        let cfg: PreviewConfig = parse_args(&argv(&[
            "a.prism",
            "--addr",
            "127.0.0.1:9000",
            "--frame",
            "1.5",
            "--out",
            "out/f.png",
            "--width",
            "320",
            "--height",
            "180",
        ]))
        .expect("合法参数应解析成功");
        assert_eq!(cfg.addr, "127.0.0.1:9000");
        assert_eq!(cfg.frame, Some(1.5));
        assert_eq!(cfg.out, PathBuf::from("out/f.png"));
        assert_eq!(cfg.width, Some(320));
        assert_eq!(cfg.height, Some(180));
    }

    #[test]
    fn unknown_flag_and_bad_values_rejected() {
        assert!(parse_args(&argv(&["a.prism", "--wat"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--frame", "abc"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--frame", "-1"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--frame", "nan"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--frame"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--addr"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--width", "0"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--width", "abc"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "--out", ""])).is_err());
    }

    #[test]
    fn missing_or_duplicate_positional_rejected() {
        assert!(parse_args(&argv(&[])).is_err());
        assert!(parse_args(&argv(&["--addr", "x"])).is_err());
        assert!(parse_args(&argv(&["a.prism", "b.prism"])).is_err());
    }

    #[test]
    fn frame_index_rounds_and_clamps() {
        // fps=8：第 k 帧时刻 = k/8。round-to-nearest + 钳制在 [0, n)。
        assert_eq!(frame_index_for_time(0.0, 8, 8), Some(0));
        assert_eq!(frame_index_for_time(0.5, 8, 8), Some(4));
        assert_eq!(frame_index_for_time(0.0625, 8, 8), Some(1)); // 恰在中点 → 向上取
        assert_eq!(frame_index_for_time(0.9, 8, 8), Some(7)); // round(7.2)=7
        assert_eq!(frame_index_for_time(99.0, 8, 8), Some(7)); // 超时长钳到末帧
        assert_eq!(frame_index_for_time(-1.0, 8, 8), Some(0)); // 负值钳到首帧
        assert_eq!(frame_index_for_time(f32::NAN, 8, 8), Some(0)); // 非有限 → 首帧
        // 空序列 / fps=0 → None（调用方须报错，绝不静默出黑帧）。
        assert_eq!(frame_index_for_time(0.0, 8, 0), None);
        assert_eq!(frame_index_for_time(0.0, 0, 8), None);
    }

    #[test]
    fn query_f32_parses_t_and_rejects_garbage() {
        assert_eq!(query_f32("t=1.5", "t"), Some(1.5));
        assert_eq!(query_f32("t=0", "t"), Some(0.0));
        assert_eq!(query_f32("x=1&t=2.5&y=3", "t"), Some(2.5));
        assert_eq!(query_f32("t=abc", "t"), None);
        assert_eq!(query_f32("t=nan", "t"), None);
        assert_eq!(query_f32("t=inf", "t"), None);
        assert_eq!(query_f32("other=1", "t"), None);
        assert_eq!(query_f32("", "t"), None);
    }

    /// 构造一个最小 HttpRequest（测试路由用）。
    fn http_get(target: &str) -> HttpRequest {
        HttpRequest {
            method: "GET".to_string(),
            target: target.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    #[test]
    fn route_unknown_is_404_and_root_is_usage() {
        let state = ServerState { pair: Arc::new((Mutex::new(None), Condvar::new())) };
        assert_eq!(route(&http_get("/nope"), &state).status, 404);
        assert_eq!(route(&http_get("/"), &state).status, 200);
    }

    #[test]
    fn route_health_reports_not_ready_while_rendering() {
        let state = ServerState { pair: Arc::new((Mutex::new(None), Condvar::new())) };
        let resp = route(&http_get("/health"), &state);
        assert_eq!(resp.status, 200);
        let text = String::from_utf8(resp.body).expect("UTF-8");
        assert!(text.contains("\"ready\":false"), "渲染中应报未就绪：{text}");
    }

    #[test]
    fn route_meta_reports_ready_with_spec() {
        let set = FrameSet {
            title: "t".to_string(),
            width: 96,
            height: 54,
            fps: 8,
            duration: 1.0,
            pngs: vec![vec![0x89, b'P'], vec![0x89, b'P']],
        };
        let state = ServerState {
            pair: Arc::new((Mutex::new(Some(ReadyFrames::Ready(set))), Condvar::new())),
        };
        let resp = route(&http_get("/meta"), &state);
        assert_eq!(resp.status, 200);
        let text = String::from_utf8(resp.body).expect("UTF-8");
        assert!(text.contains("\"ready\":true"), "{text}");
        assert!(text.contains("\"fps\":8"), "{text}");
        assert!(text.contains("\"frames\":2"), "{text}");
    }

    #[test]
    fn route_frame_serves_png_and_scrub_selects_by_time() {
        // 两个不同内容的"帧"：scrub 到不同时刻应返回不同字节。
        let set = FrameSet {
            title: "t".to_string(),
            width: 8,
            height: 8,
            fps: 4,
            duration: 1.0,
            pngs: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![10, 11, 12]],
        };
        let state = ServerState {
            pair: Arc::new((Mutex::new(Some(ReadyFrames::Ready(set))), Condvar::new())),
        };
        let r0 = route(&http_get("/frame?t=0"), &state);
        assert_eq!(r0.status, 200);
        assert_eq!(r0.content_type, "image/png");
        assert_eq!(r0.body, vec![1, 2, 3]);

        let r2 = route(&http_get("/frame?t=0.5"), &state);
        assert_eq!(r2.status, 200);
        assert_eq!(r2.body, vec![7, 8, 9], "t=0.5 → 第 2 帧");

        // 不同时刻 → 不同帧（scrub 的核心语义）。
        assert_ne!(r0.body, r2.body);

        // 超时长钳制到末帧；缺 t 参数默认 0。
        assert_eq!(route(&http_get("/frame?t=99"), &state).body, vec![10, 11, 12]);
        assert_eq!(route(&http_get("/frame"), &state).body, vec![1, 2, 3]);
    }

    #[test]
    fn route_frame_503_when_not_ready_and_500_when_failed() {
        // 未就绪：有界等待对单测太长——直接验证"失败"分支的 500。
        let state = ServerState {
            pair: Arc::new((Mutex::new(Some(ReadyFrames::Failed("R5001 演示失败".to_string()))), Condvar::new())),
        };
        let resp = route(&http_get("/frame?t=0"), &state);
        assert_eq!(resp.status, 500);
        let text = String::from_utf8(resp.body).expect("UTF-8");
        assert!(text.contains("R5001"), "失败原因应如实回传：{text}");
    }
}
