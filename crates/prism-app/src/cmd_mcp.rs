//! `prism-app mcp` 子命令：MCP stdio 服务器主循环。
//!
//! 语义权威：《强大版路线图》§3.6（"MCP stdio JSON-RPC"）。MCP 用 **行分隔
//! 的 JSON-RPC**（每条消息一行）经 stdin/stdout 收发；本文件只做 I/O 循环，
//! 协议实现见 [`prism_infra::mcp`]。
//!
//! # 约定
//! - 读 stdin 每行作为一条 JSON-RPC 消息；
//! - 写 stdout 每行一条响应（通知不回）；
//! - EOF（stdin 关闭）→ 正常退出 0。

use prism_infra::mcp::McpServer;
use std::io::{BufRead, Write};

/// mcp 子命令用法。
pub const MCP_USAGE: &str = r#"prism-app mcp — MCP 服务器（stdio JSON-RPC）

用法:
  prism-app mcp        从 stdin 读取 JSON-RPC 消息，向 stdout 写响应

工具: compile / render_frame / render_video / critique
（MCP 客户端通常自行启动本进程；手动运行需按行输入 JSON-RPC 消息）"#;

/// 运行 mcp 子命令；返回进程退出码。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{MCP_USAGE}");
        return 0;
    }
    let server = McpServer::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let reader = stdin.lock();
    for line in reader.lines() {
        let line: String = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        match server.handle_text(&line) {
            Ok(Some(resp)) => {
                if writeln!(stdout, "{resp}").is_err() {
                    break;
                }
                let _ = stdout.flush();
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!("prism-app mcp: 序列化响应失败: {err}");
                return 1;
            }
        }
    }
    0
}
