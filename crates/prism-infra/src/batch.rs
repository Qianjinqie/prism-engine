//! batch 模式：批量渲染任务清单的解析、笛卡尔积展开、并行帧渲染与进度。
//!
//! 语义权威：CLI `batch`与 C6 验收 6（"batch 笛卡尔积
//! 展开"）。
//!
//! # 任务清单格式（最小 TOML 子集，手写解析）
//! ```text
//! [[task]]
//! source = "demo.prism"
//! out = "render/demo"
//! width = [320, 640]        # 数组 = 该维度多变体（笛卡尔积）
//! height = [180]
//! fps = [15, 30]
//! ```
//! 每个 `[[task]]` 块按 `width`/`height`/`fps` 三轴展开为 `N_w * N_h * N_f`
//! 个子任务（缺省单元素）；展开顺序稳定（确定性：外层 width → height → fps）。
//!
//! # 并行
//! 帧渲染用 `std::thread`（标准库）分摊到 `workers` 个工作线程，每个线程独立
//! 构造 `CpuRefBackend`（无共享可变状态，无全局 RNG）→ 结果确定性不依赖调度。
//! 进度以回调上报（`(done, total)`），供 CLI 打印。

use crate::engine::{self, RenderRequest};
use std::sync::Arc;

/// batch 任务轴：三个可变维度（宽 / 高 / 帧率），每个维度是候选值列表。
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// DSL 源文件路径。
    pub source: String,
    /// 输出目录路径。
    pub out: String,
    /// 宽度候选（≥1）。
    pub widths: Vec<u32>,
    /// 高度候选（≥1）。
    pub heights: Vec<u32>,
    /// 帧率候选（≥1）。
    pub fps: Vec<u32>,
}

/// 展开后的单个子任务（笛卡尔积的一格）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubTask {
    /// 源文件路径。
    pub source: String,
    /// 输出目录（含变体后缀）。
    pub out: String,
    /// 该变体的宽度。
    pub width: u32,
    /// 该变体的高度。
    pub height: u32,
    /// 该变体的帧率。
    pub fps: u32,
}

impl Task {
    /// 规范化：空轴补单元素 0（0 = 用 DSL 声明）。
    fn normalized(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let w: Vec<u32> = if self.widths.is_empty() { vec![0] } else { self.widths.clone() };
        let h: Vec<u32> = if self.heights.is_empty() { vec![0] } else { self.heights.clone() };
        let f: Vec<u32> = if self.fps.is_empty() { vec![0] } else { self.fps.clone() };
        (w, h, f)
    }

    /// 笛卡尔积展开为子任务列表（顺序稳定：width → height → fps）。
    pub fn expand(&self) -> Vec<SubTask> {
        let (widths, heights, fps) = self.normalized();
        let mut out: Vec<SubTask> = Vec::with_capacity(widths.len() * heights.len() * fps.len());
        for w in &widths {
            for h in &heights {
                for f in &fps {
                    out.push(SubTask {
                        source: self.source.clone(),
                        out: format!("{}/w{}-h{}-f{}", self.out, w, h, f),
                        width: *w,
                        height: *h,
                        fps: *f,
                    });
                }
            }
        }
        out
    }
}

/// 批量渲染配置。
#[derive(Debug, Clone, PartialEq)]
pub struct BatchConfig {
    /// 任务清单。
    pub tasks: Vec<Task>,
    /// 工作线程数（≥1；1 = 顺序执行）。
    pub workers: usize,
}

/// 子任务渲染结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubTaskResult {
    /// 子任务描述。
    pub subtask: SubTask,
    /// 渲染帧数。
    pub frame_count: usize,
    /// 失败原因（None = 成功）。
    pub error: Option<String>,
}

/// 从最小 TOML 子集解析 batch 任务清单。
///
/// # Errors
/// 语法非法（非 `key = "value"` / `[段]` / `[[task]]` / 数组 / 注释）→ 错误消息。
pub fn parse_manifest(src: &str) -> Result<Vec<Task>, String> {
    let mut tasks: Vec<Task> = Vec::new();
    let mut cur: Option<Task> = None;
    for (idx, raw) in src.lines().enumerate() {
        let line_no: usize = idx + 1;
        let line: &str = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[task]]" {
            if let Some(t) = cur.take() {
                tasks.push(t);
            }
            cur = Some(Task {
                source: String::new(),
                out: String::new(),
                widths: Vec::new(),
                heights: Vec::new(),
                fps: Vec::new(),
            });
            continue;
        }
        let task: &mut Task = cur.as_mut().ok_or_else(|| {
            format!("第 {} 行: 出现键值但不在 [[task]] 块内\n  建议：先用 [[task]] 开始一个任务块", line_no)
        })?;
        let (key, value): (&str, &str) = line.split_once('=').ok_or_else(|| {
            format!("第 {} 行: '{}' 不是 'key = value' 形式", line_no, line)
        })?;
        let key: &str = key.trim();
        let value: &str = value.trim();
        match key {
            "source" => task.source = parse_string(value, line_no)?,
            "out" => task.out = parse_string(value, line_no)?,
            "width" => task.widths = parse_u32_list(value, line_no)?,
            "height" => task.heights = parse_u32_list(value, line_no)?,
            "fps" => task.fps = parse_u32_list(value, line_no)?,
            other => {
                return Err(format!("第 {} 行: 未知键 '{}'（支持 source/out/width/height/fps）", line_no, other));
            }
        }
    }
    if let Some(t) = cur.take() {
        tasks.push(t);
    }
    if tasks.is_empty() {
        return Err("任务清单为空：至少需要一个 [[task]] 块".to_string());
    }
    for (i, t) in tasks.iter().enumerate() {
        if t.source.is_empty() {
            return Err(format!("第 {} 个 [[task]] 缺少 'source'", i + 1));
        }
        if t.out.is_empty() {
            return Err(format!("第 {} 个 [[task]] 缺少 'out'", i + 1));
        }
    }
    Ok(tasks)
}

fn parse_string(value: &str, line_no: usize) -> Result<String, String> {
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        Ok(value[1..value.len() - 1].to_string())
    } else {
        Err(format!("第 {} 行: 值 {} 应为双引号字符串", line_no, value))
    }
}

/// 解析 `[a, b, c]` 或单值 `n` 为 u32 列表。
fn parse_u32_list(value: &str, line_no: usize) -> Result<Vec<u32>, String> {
    let inner: &str = if value.starts_with('[') && value.ends_with(']') {
        &value[1..value.len() - 1]
    } else {
        value
    };
    let mut out: Vec<u32> = Vec::new();
    for part in inner.split(',') {
        let part: &str = part.trim();
        if part.is_empty() {
            continue;
        }
        let n: u32 = part
            .parse()
            .map_err(|_| format!("第 {} 行: '{}' 不是非负整数", line_no, part))?;
        out.push(n);
    }
    if out.is_empty() {
        return Err(format!("第 {} 行: 数值列表为空", line_no));
    }
    Ok(out)
}

/// 执行批量渲染；`on_progress(done, total)` 在每个子任务完成后回调。
///
/// 并行度由 `cfg.workers` 决定：用 `std::thread` 把子任务分片（round-robin），
/// 每线程顺序处理分到的子任务（各自独立 `CpuRefBackend`，确定性不受调度影响）。
/// 结果按子任务原顺序汇总返回。
pub fn run_batch<F>(cfg: &BatchConfig, mut on_progress: F) -> Vec<SubTaskResult>
where
    F: FnMut(usize, usize),
{
    let mut subtasks: Vec<SubTask> = Vec::new();
    for task in &cfg.tasks {
        subtasks.extend(task.expand());
    }
    let total: usize = subtasks.len();
    let workers: usize = cfg.workers.max(1).min(total.max(1));

    // 共享只读的 subtasks（Arc 免拷贝）；结果槽用 Option 预分配，按 index 回填。
    let shared: Arc<Vec<SubTask>> = Arc::new(subtasks);
    let slots: Arc<Vec<std::sync::Mutex<Option<SubTaskResult>>>> =
        Arc::new((0..total).map(|_| std::sync::Mutex::new(None)).collect());

    // 分片执行：把 index 按 round-robin 分给 workers 个线程。
    std::thread::scope(|scope| {
        for worker in 0..workers {
            let shared = Arc::clone(&shared);
            let slots = Arc::clone(&slots);
            scope.spawn(move || {
                let mut idx: usize = worker;
                while idx < total {
                    let result: SubTaskResult = run_subtask(&shared[idx]);
                    *slots[idx].lock().unwrap() = Some(result);
                    idx += workers;
                }
            });
        }
    });

    let mut results: Vec<SubTaskResult> = Vec::with_capacity(total);
    for (i, slot) in slots.iter().enumerate() {
        let r: SubTaskResult = slot.lock().unwrap().take().unwrap_or_else(|| SubTaskResult {
            subtask: shared[i].clone(),
            frame_count: 0,
            error: Some("内部错误：结果槽未回填".to_string()),
        });
        on_progress(i + 1, total);
        results.push(r);
    }
    results
}

fn run_subtask(sub: &SubTask) -> SubTaskResult {
    let source: String = match std::fs::read_to_string(&sub.source) {
        Ok(s) => s,
        Err(e) => {
            return SubTaskResult {
                subtask: sub.clone(),
                frame_count: 0,
                error: Some(format!("无法读取 '{}': {}", sub.source, e)),
            };
        }
    };
    let req: RenderRequest = RenderRequest {
        width: (sub.width > 0).then_some(sub.width),
        height: (sub.height > 0).then_some(sub.height),
        fps: (sub.fps > 0).then_some(sub.fps),
    };
    match engine::render_frames(&source, &req) {
        Ok(frames) => {
            let count: usize = frames.len();
            if let Err(e) = std::fs::create_dir_all(&sub.out) {
                return SubTaskResult {
                    subtask: sub.clone(),
                    frame_count: count,
                    error: Some(format!("无法创建输出目录 '{}': {}", sub.out, e)),
                };
            }
            if let Err(e) = prism_io::save_frames(&sub.out, &frames, req.fps.unwrap_or(30).max(1), 0.0) {
                return SubTaskResult {
                    subtask: sub.clone(),
                    frame_count: count,
                    error: Some(format!("写入帧序列失败: {}", e)),
                };
            }
            SubTaskResult { subtask: sub.clone(), frame_count: count, error: None }
        }
        Err(e) => SubTaskResult {
            subtask: sub.clone(),
            frame_count: 0,
            error: Some(e.to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// 自测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"
# batch demo
[[task]]
source = "demo.prism"
out = "render/demo"
width = [320, 640]
height = [180]
fps = [15, 30]

[[task]]
source = "b.prism"
out = "render/b"
height = 90
fps = 24
"#;

    #[test]
    fn manifest_parses_two_tasks() {
        let tasks = parse_manifest(MANIFEST).expect("应可解析");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].widths, vec![320, 640]);
        assert_eq!(tasks[1].heights, vec![90]);
    }

    #[test]
    fn cartesian_expansion_order_is_stable() {
        let tasks = parse_manifest(MANIFEST).unwrap();
        let sub = tasks[0].expand();
        assert_eq!(sub.len(), 4, "2宽的 × 1高 × 2帧率 = 4");
        assert_eq!(sub[0].out, "render/demo/w320-h180-f15");
        assert_eq!(sub[1].out, "render/demo/w320-h180-f30");
        assert_eq!(sub[2].out, "render/demo/w640-h180-f15");
        assert_eq!(sub[3].out, "render/demo/w640-h180-f30");
    }

    #[test]
    fn missing_source_is_rejected() {
        let bad: &str = "[[task]]\nout = \"x\"\n";
        assert!(parse_manifest(bad).is_err());
    }

    #[test]
    fn unknown_key_is_rejected() {
        let bad: &str = "[[task]]\nsource = \"a\"\nout = \"b\"\nbogus = \"c\"\n";
        assert!(parse_manifest(bad).is_err());
    }

    #[test]
    fn run_batch_reports_progress_and_handles_missing_file() {
        let cfg = BatchConfig {
            tasks: vec![Task {
                source: "does-not-exist.prism".to_string(),
                out: "/tmp/prism-batch-test".to_string(),
                widths: vec![16],
                heights: vec![16],
                fps: vec![2],
            }],
            workers: 2,
        };
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let results = run_batch(&cfg, |d, t| seen.push((d, t)));
        assert_eq!(results.len(), 1);
        assert!(results[0].error.is_some(), "缺失文件应报错而非 panic");
        assert_eq!(seen, vec![(1, 1)]);
    }
}
