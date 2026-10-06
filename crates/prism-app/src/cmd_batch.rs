//! `prism-app batch` 子命令：读任务清单 → 笛卡尔积展开 → 并行渲染 → 进度汇报。
//!
//! 语义权威：《强大版路线图》§3.6（CLI `batch`）、C6（"crism-app/cmd_batch.rs"、
//! "batch matrix"）。核心逻辑在 [`prism_infra::batch`]，本文件只做 CLI 适配：
//! 参数解析、进度打印、退出码。

use prism_infra::batch::{self, BatchConfig};
use std::path::Path;

/// batch 子命令用法。
pub const BATCH_USAGE: &str = r#"prism-app batch — 批量渲染任务清单

用法:
  prism-app batch <manifest.toml> [--workers N]

任务清单格式（最小 TOML 子集）:
  [[task]]
  source = "demo.prism"
  out = "render/demo"
  width = [320, 640]      # 数组 = 该轴多变体（笛卡尔积）
  height = [180]
  fps = [15, 30]

选项:
  --workers <N>   并行工作线程数（默认 4）
  -h, --help      显示本帮助"#;

/// 运行 batch 子命令；返回进程退出码（0 成功 / 1 失败）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{BATCH_USAGE}");
        return 0;
    }
    let mut manifest_path: Option<String> = None;
    let mut workers: usize = 4;
    let mut i: usize = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workers" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n >= 1 => workers = n,
                    _ => {
                        eprintln!("prism-app batch: --workers 需要一个 ≥1 的整数");
                        return 1;
                    }
                }
            }
            other if other.starts_with('-') => {
                eprintln!("prism-app batch: 未知参数 `{other}`");
                eprintln!("{BATCH_USAGE}");
                return 1;
            }
            other => {
                if manifest_path.is_some() {
                    eprintln!("prism-app batch: 多余的清单参数 `{other}`");
                    return 1;
                }
                manifest_path = Some(other.to_string());
            }
        }
        i += 1;
    }
    let manifest_path: String = match manifest_path {
        Some(p) => p,
        None => {
            eprintln!("prism-app batch: 缺少 <manifest.toml> 参数");
            eprintln!("{BATCH_USAGE}");
            return 1;
        }
    };

    let text: String = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("prism-app batch: 无法读取清单 `{manifest_path}`: {e}");
            return 1;
        }
    };
    let tasks = match batch::parse_manifest(&text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("prism-app batch: 清单解析失败: {e}");
            return 1;
        }
    };
    let cfg: BatchConfig = BatchConfig { tasks, workers };

    println!("prism-app batch: 展开子任务中（workers={workers}）...");
    let results = batch::run_batch(&cfg, |done, total| {
        print!("\r  进度 {done}/{total}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    });
    println!();

    let mut failures: usize = 0;
    for r in &results {
        match &r.error {
            None => println!(
                "  ✓ {} ({} 帧)",
                r.subtask.out,
                r.frame_count
            ),
            Some(err) => {
                failures += 1;
                println!("  ✗ {}: {}", r.subtask.out, err);
            }
        }
    }
    // 确保输出目录被创建（有失败时也给出明确报告）。
    if let Some(parent) = Path::new(&manifest_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    println!(
        "prism-app batch: 完成 {} 个子任务，{} 失败",
        results.len(),
        failures
    );
    if failures == 0 {
        0
    } else {
        1
    }
}
