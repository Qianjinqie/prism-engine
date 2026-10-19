//! 引擎门面：把 DSL 编译 + 渲染 + 落盘收敛为可被 MCP / batch / HTTP 复用的
//! 纯 Rust 调用。三个出口共用同一实现，保证三处行为逐字节一致（确定性）。
//!
//! # 语义
//! - [`compile_source`]：DSL v1 编译 → `CompileOutput`（场景 + 输出参数）；
//! - [`render_frame_png`]：编译单帧并编码为 PNG 字节；
//! - [`render_frames`]：按 fps 均匀采样整段，返回帧缓冲序列；
//! - 分辨率 / 帧率可由调用方覆盖（缺省用 DSL 声明）。
//!
//! # 确定性
//! 同输入（源码 + 分辨率 + 帧率 + 时刻）必得同字节输出：无全局 RNG，渲染走
//! [`prism_gpu::CpuRefBackend`] 参考后端（与 CLI 同款）。

use prism_gpu::{CpuRefBackend, Framebuffer};

/// 编译错误（承载 DSL 行号 + 消息，供各出口统一转协议错误）。
#[derive(Debug, Clone, PartialEq)]
pub struct EngineError {
    /// 出错行号（编译错误）或 0（非编译错误）。
    pub line: usize,
    /// 面向人类的消息。
    pub message: String,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(f, "DSL 第 {} 行: {}", self.line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for EngineError {}

/// 渲染请求（各出口解析后的统一形态）。
#[derive(Debug, Clone)]
pub struct RenderRequest {
    /// 分辨率宽度（None = 用 DSL 声明）。
    pub width: Option<u32>,
    /// 分辨率高度（None = 用 DSL 声明）。
    pub height: Option<u32>,
    /// 帧率（None = 用 DSL 声明）。
    pub fps: Option<u32>,
}

impl Default for RenderRequest {
    fn default() -> Self {
        RenderRequest { width: None, height: None, fps: None }
    }
}

/// 编译产物摘要（供 MCP/HTTP 汇报，不进图）。
#[derive(Debug, Clone, PartialEq)]
pub struct CompileSummary {
    /// 分辨率宽度。
    pub width: u32,
    /// 分辨率高度。
    pub height: u32,
    /// 帧率。
    pub fps: u32,
    /// 时长（秒）。
    pub duration: f32,
    /// 输出格式名（"frames" / "png"）。
    pub format: String,
}

/// 编译 DSL 源码为引擎场景（v1 入口，逐字沿用 CLI 路径）。
///
/// # Errors
/// DSL 词法 / 语法 / 语义错误 → [`EngineError`]（含行号）。
pub fn compile_source(source: &str) -> Result<prism_dsl::CompileOutput, EngineError> {
    prism_dsl::compile(source).map_err(|e| EngineError { line: e.line, message: e.message })
}

/// 编译并汇总（不进图，供协议出口汇报元信息）。
///
/// # Errors
/// 同 [`compile_source`]。
pub fn compile_summary(source: &str) -> Result<CompileSummary, EngineError> {
    let out: prism_dsl::CompileOutput = compile_source(source)?;
    Ok(summary_of(&out))
}

/// 已编译产物 → 摘要（不重复编译，供已持有 `CompileOutput` 的出口复用）。
///
/// MCP `scene_check` / `diagnose` 等工具先编译一次、再用同一份产物取摘要，
/// 而不是"取摘要编一遍、渲染再编一遍"——省一次编译，更杜绝两遍编译结果
/// 不一致的理论可能。
pub fn summary_of(out: &prism_dsl::CompileOutput) -> CompileSummary {
    let format: String = match out.format {
        prism_dsl::OutputFormat::Frames => "frames".to_string(),
        prism_dsl::OutputFormat::Png => "png".to_string(),
    };
    CompileSummary {
        width: out.width,
        height: out.height,
        fps: out.fps,
        duration: out.duration,
        format,
    }
}

/// 渲染单帧（不落盘）。
///
/// # Errors
/// 编译失败 → [`EngineError`]；渲染分辨率非法（0）→ [`EngineError`]。
pub fn render_frame(
    source: &str,
    t: f32,
    req: &RenderRequest,
) -> Result<Framebuffer, EngineError> {
    let compiled: prism_dsl::CompileOutput = compile_source(source)?;
    let width: u32 = req.width.unwrap_or(compiled.width);
    let height: u32 = req.height.unwrap_or(compiled.height);
    if width == 0 || height == 0 {
        return Err(EngineError { line: 0, message: "渲染分辨率必须为正（宽/高 > 0）".to_string() });
    }
    let duration: f32 = compiled.duration;
    let mut device = CpuRefBackend;
    Ok(prism_render::render_frame(&mut device, &compiled.scene, t, duration, width, height))
}

/// 渲染单帧并编码为 PNG 字节（内存中，不落盘）。
///
/// `prism_io::save_png` 落盘版走文件；此处复用其 PNG 编码器输出到内存，
/// 供 MCP `render_frame` 直接返回 PNG 字节（验收 6）。
///
/// # Errors
/// 同 [`render_frame`]；PNG 编码失败 → [`EngineError`]。
pub fn render_frame_png(
    source: &str,
    t: f32,
    req: &RenderRequest,
) -> Result<Vec<u8>, EngineError> {
    let fb: Framebuffer = render_frame(source, t, req)?;
    Ok(prism_io::encode_png(&fb))
}

/// 帧数：完全由 `fps × duration` 决定，与任何循环长度无关。
///
/// 抽成公开函数是为了让"渲染 N 帧"与"定位到第 k 帧"共用**同一个**帧号→帧数
/// 公式，杜绝两处各算一遍导致的 off-by-one（那正是 seek 语义缺陷的温床）。
///
/// # 契约
/// - `fps == 0` 或 `duration <= 0` → 0 帧（调用方须自行报错，见 [`render_frames`]）；
/// - 结果钳制在 `usize::MAX`（`fps × duration` 的 f32 值可溢出 u32/u64）。
pub fn frame_count(fps: u32, duration: f32) -> usize {
    if fps == 0 || !(duration > 0.0) {
        return 0;
    }
    let n: f32 = fps as f32 * duration;
    if !n.is_finite() || n >= usize::MAX as f32 {
        return usize::MAX;
    }
    // `ceil` 与 `prism_render::render_video` 的采样数公式一致（末帧不超 duration）。
    n.ceil() as usize
}

/// 帧号 → 时刻 `t`（秒）：**seek 的唯一真源**。
///
/// 第 k 帧的时刻只由 `(frame_index, fps)` 决定，与此前渲染过多少帧无关——这
/// 就是铁律「seek ≡ 顺序推进」的数学表达：批量渲染的第 k 帧与直接 seek 到
/// 第 k 帧走的是同一条 `t = k / fps` 纯函数，因此必然逐字节相同。
///
/// 取值 `t ∈ [0, duration)`：调用方须保证 `frame_index < frame_count(fps, duration)`。
pub fn frame_time(frame_index: u64, fps: u32) -> f32 {
    if fps == 0 {
        return 0.0;
    }
    // u64 → f32 在 ≥ 2²⁴ 时失真，但那种帧号已远超任何真实时长；钳制避免溢出成 inf。
    if frame_index >= (1_u64 << 24) {
        return f32::INFINITY;
    }
    frame_index as f32 / fps as f32
}

/// 渲染整段（按 fps 均匀采样），返回帧序列（不落盘）。
///
/// # 确定性
/// 帧数 = [`frame_count`]（由 fps × duration 决定），第 k 帧时刻 = [`frame_time`]
/// （由帧号派生）。渲染过程无内部可变状态、无按调用序消耗的随机数，故
/// 「渲染 N 帧取第 k 帧」与「seek 到第 k 帧」逐字节相同。
///
/// # Errors
/// 编译失败 / 分辨率非法 / fps*duration 产出 0 帧 → [`EngineError`]。
pub fn render_frames(
    source: &str,
    req: &RenderRequest,
) -> Result<Vec<Framebuffer>, EngineError> {
    let compiled: prism_dsl::CompileOutput = compile_source(source)?;
    let width: u32 = req.width.unwrap_or(compiled.width);
    let height: u32 = req.height.unwrap_or(compiled.height);
    let fps: u32 = req.fps.unwrap_or(compiled.fps);
    if width == 0 || height == 0 {
        return Err(EngineError { line: 0, message: "渲染分辨率必须为正（宽/高 > 0）".to_string() });
    }
    if fps == 0 || compiled.duration <= 0.0 {
        return Err(EngineError { line: 0, message: "fps 与时长必须为正，无法产出帧".to_string() });
    }
    let mut device = CpuRefBackend;
    let frames: Vec<Framebuffer> =
        prism_render::render_video(&mut device, &compiled.scene, width, height, fps, compiled.duration);
    if frames.is_empty() {
        return Err(EngineError { line: 0, message: "渲染结果为空（0 帧）".to_string() });
    }
    Ok(frames)
}

/// **seek 入口**：直接渲染第 `frame_index` 帧，不渲染前序帧。
///
/// 这是铁律「seek ≡ 顺序推进」的门面：seek 与顺序推进必须共用
/// [`frame_time`] 这一条帧号→时刻的纯函数，且渲染本身无跨调用状态，故
/// `render_frame_indexed(k) == render_frames(..)[k]` 逐字节成立——增量渲染、
/// 缓存复用、并行 md5 对拍都以此为前提。
///
/// # Errors
/// 同 [`render_frame`]；`fps == 0` 或 `frame_index` 越界 → [`EngineError`]。
pub fn render_frame_indexed(
    source: &str,
    frame_index: u64,
    req: &RenderRequest,
) -> Result<Framebuffer, EngineError> {
    let compiled: prism_dsl::CompileOutput = compile_source(source)?;
    let fps: u32 = req.fps.unwrap_or(compiled.fps);
    let total: usize = frame_count(fps, compiled.duration);
    if total == 0 {
        return Err(EngineError { line: 0, message: "fps 与时长必须为正，无法定位帧".to_string() });
    }
    if frame_index >= total as u64 {
        return Err(EngineError {
            line: 0,
            message: format!("帧号 {} 越界（总帧数 {}）", frame_index, total),
        });
    }
    render_frame(source, frame_time(frame_index, fps), req)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "scene \"t\" {\n  sphere ball { radius 1.0 material { albedo #c0392b roughness 0.3 metalness 0.05 } }\n  camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }\n  light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }\n}\ntimeline { duration 0.2s fps 10 }\noutput { resolution 64x48 format frames }\n";

    #[test]
    fn compile_summary_matches_dsl() {
        let s: CompileSummary = compile_summary(SRC).expect("应可编译");
        assert_eq!(s.width, 64);
        assert_eq!(s.height, 48);
        assert_eq!(s.fps, 10);
        assert_eq!(s.format, "frames");
    }

    #[test]
    fn render_frame_is_deterministic() {
        let a: Vec<u8> = render_frame_png(SRC, 0.0, &RenderRequest::default()).unwrap();
        let b: Vec<u8> = render_frame_png(SRC, 0.0, &RenderRequest::default()).unwrap();
        assert_eq!(a, b, "同输入应逐字节一致");
        assert_eq!(&a[0..4], &[0x89, b'P', b'N', b'G'], "应为 PNG 魔数");
    }

    #[test]
    fn render_frames_counts_by_fps_times_duration() {
        let frames = render_frames(SRC, &RenderRequest::default()).unwrap();
        assert_eq!(frames.len(), 2, "10fps*0.2s = 2 帧");
    }

    #[test]
    fn invalid_source_reports_line() {
        let err: EngineError = compile_source("bogus command\n").unwrap_err();
        assert!(err.line >= 1, "应带行号");
    }
}
