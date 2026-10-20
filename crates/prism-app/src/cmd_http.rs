//! `prism-app http` 子命令：自写 HTTP/1.1 渲染服务。
//!
//! 语义权威：HTTP（自写 HTTP/1.1，局域网渲染服务）。
//! **安全约束：默认绑 `127.0.0.1`**（不暴露公网），需显式 `--addr` 才改变。
//!
//! # 路由
//! - `GET /health`：健康检查（JSON）；
//! - `GET /tools`：工具清单（JSON）；
//! - `POST /compile`：JSON `{source, width?, height?, fps?}` → 编译元信息；
//! - `POST /render_frame`：JSON `{source, t?, width?, height?}` → PNG 字节。

use prism_infra::http::{self, HttpRequest, HttpResponse};

/// 默认监听地址（安全：仅本机）。
pub const DEFAULT_ADDR: &str = "127.0.0.1:8787";

/// http 子命令用法。
pub const HTTP_USAGE: &str = r#"prism-app http — 自写 HTTP/1.1 渲染服务

用法:
  prism-app http [--addr <HOST:PORT>]

选项:
  --addr <HOST:PORT>   监听地址（默认 127.0.0.1:8787，安全：仅本机）
  -h, --help           显示本帮助

路由:
  GET  /health          健康检查（JSON）
  GET  /tools           工具清单（JSON）
  POST /compile         {source,...} → 编译元信息（JSON）
  POST /render_frame    {source,t,...} → PNG 字节"#;

/// 运行 http 子命令；返回进程退出码（服务阻塞，正常不返回）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HTTP_USAGE}");
        return 0;
    }
    let mut addr: String = DEFAULT_ADDR.to_string();
    let mut i: usize = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--addr" => {
                i += 1;
                match args.get(i) {
                    Some(v) => addr = v.clone(),
                    None => {
                        eprintln!("prism-app http: --addr 缺少取值");
                        return 1;
                    }
                }
            }
            other => {
                eprintln!("prism-app http: 未知参数 `{other}`");
                eprintln!("{HTTP_USAGE}");
                return 1;
            }
        }
        i += 1;
    }
    eprintln!("prism-app http: 监听 {addr}（Ctrl-C 退出）");
    match http::serve(&addr, |_req: &HttpRequest, _resp: &HttpResponse| {}) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("prism-app http: 服务失败: {e}");
            1
        }
    }
}
