//! `farm-worker` —— 渲染农场 worker 进程入口（V2.0-I）。
//!
//! # 干什么
//!
//! 从标准输入读**任务帧**（长度前缀 + SHA-256 校验，见
//! [`prism_render::farm::transport`]），逐帧渲染，把**逐帧载荷**按分片回写到
//! 标准输出；收到 `Done` 帧或输入 EOF 即收工。
//!
//! 主节点侧用 [`prism_render::farm::transport::LocalProcessTransport`] /
//! [`prism_render::farm::scheduler::dispatch_to_local_workers`] 拉起本进程——即
//! "**本地子进程优先**"：协议、编解码、worker 逻辑与跨机路径同一份代码，只把
//! socket 换成管道。
//!
//! # 用法
//!
//! ```bash
//! # 手工自测：喂一个任务，看回的分片（十六进制）
//! cargo run -p prism-render --bin farm-worker -- --stdio
//!
//! # 带断点续算缓存（进程重启后跳过已完成帧）
//! cargo run -p prism-render --bin farm-worker -- --stdio --cache /tmp/farm.cache
//! ```
//!
//! 参数：
//!
//! | 参数 | 含义 |
//! |---|---|
//! | `--stdio` | 走标准输入/输出（默认；被主节点 spawn 时由它自己加） |
//! | `--cache <path>` | 断点续算缓存文件（`prism_io::CacheManifest` 布局） |
//! | `--cost <n>` | 渲染成本（纯空转迭代数，**不影响输出**，压测用） |
//! | 环境 `PRISM_FARM_COST` | 同上（主节点 spawn 时不便拼参数，只能走环境；`--cost` 优先） |
//! | 环境 `PRISM_FARM_MOVIE_SRC` | 影片 DSL 源码（PE-1）：注入即渲染**真实帧像素**，未设则 32 字节替身；启动时读一次，注入但编译坏 ⇒ 启动失败（不静默退回替身） |
//! | `--help` / `-h` | 打印用法 |
//!
//! 退出码：`0` 正常收工；`1` 出错（错误信息已按三段式打印到 stderr，含建议段）。

use prism_render::farm::{run_stdio_worker, WorkerOptions};
use std::path::PathBuf;

/// 打印用法（`--help`）。
fn print_usage() {
    println!(
        "farm-worker —— 渲染农场 worker（stdio 帧协议）\n\
         \n\
         用法：farm-worker [--stdio] [--cache <path>] [--cost <n>]\n\
         \n\
           --stdio        走标准输入/输出（默认；主节点 spawn 时自动加）\n\
           --cache <path> 断点续算缓存文件（重启后跳过已完成帧）\n\
           --cost <n>     渲染成本（空转迭代数，不影响输出，压测用）\n\
           --help, -h     打印本用法\n\
         \n\
         环境变量：\n\
           PRISM_FARM_COST      渲染成本（--cost 优先）\n\
           PRISM_FARM_MOVIE_SRC 影片 DSL 源码（注入后渲染真实帧像素；不设 = 32 字节替身）\n"
    );
}

/// 解析命令行（不引第三方 CLI 库：零依赖铁律）。
fn parse_args(args: &[String]) -> Result<Option<WorkerOptions>, String> {
    let mut options: WorkerOptions = WorkerOptions::new();
    let mut i: usize = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(None),
            "--stdio" => {}
            "--cache" => {
                i += 1;
                let Some(raw) = args.get(i) else {
                    return Err("--cache 后面必须跟一个文件路径".to_string());
                };
                options = options.with_cache(PathBuf::from(raw));
            }
            "--cost" => {
                i += 1;
                let Some(raw) = args.get(i) else {
                    return Err("--cost 后面必须跟一个非负整数（空转迭代数）".to_string());
                };
                let parsed: u32 = raw
                    .parse()
                    .map_err(|e| format!("--cost 需要 u32 整数，收到 {raw:?}（{e}）"))?;
                options = options.with_cost(parsed);
            }
            other => {
                return Err(format!("未知参数 {other:?}（用 --help 查看用法）"));
            }
        }
        i += 1;
    }
    Ok(Some(options))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options: WorkerOptions = match parse_args(&args) {
        Ok(Some(o)) => o,
        Ok(None) => {
            print_usage();
            return;
        }
        Err(detail) => {
            // 三段式：在哪 / 什么事 / 建议。
            eprintln!(
                "error[PW1001] farm-worker::main\n  \
                 什么事：{detail}\n  \
                 建议：用 `farm-worker --help` 查看可用参数；主节点 spawn 时只需 `--stdio`"
            );
            std::process::exit(1);
        }
    };
    // 渲染成本可由环境变量提供（主节点 spawn 本进程时不便拼参数，只能走环境）；
    // 命令行 `--cost` 优先。成本只影响耗时，**不影响载荷**。
    let options: WorkerOptions = if options.work_iters == 0 {
        match std::env::var("PRISM_FARM_COST").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(cost) => options.with_cost(cost),
            None => options,
        }
    } else {
        options
    };

    match run_stdio_worker(&options) {
        Ok(served) => {
            // 正常收工：把"服务了几个任务"打到 stderr（stdout 是协议通道，不能污染）。
            eprintln!("farm-worker: 收工，已服务 {served} 个任务");
        }
        Err(err) => {
            // FarmError 的 Display 已是三段式（含建议段，C19）。
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
