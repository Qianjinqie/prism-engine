//! `prism-app pack` 子命令：数据包市场本地库管理（add / list / remove）。
//!
//! 语义权威：《强大版路线图》§2.4 / C6（"数据包市场校验（add/list/remove）"）、
//! D-8（插件=数据包，无动态代码加载）。核心逻辑在 [`prism_infra::plugin`]，
//! 本文件只做 CLI 适配（子命令分发 + 库根目录约定）。
//!
//! # 库根目录
//! 默认 `~/.prism/packs`（`$PRISM_HOME` 可覆盖）；`--lib <DIR>` 显式指定。

use prism_infra::plugin::PackLibrary;
use std::path::{Path, PathBuf};

/// pack 子命令用法。
pub const PACK_USAGE: &str = r#"prism-app pack — 数据包市场（本地库管理）

用法:
  prism-app pack add <PACK_DIR> [--lib <LIB_DIR>]
  prism-app pack list [--lib <LIB_DIR>]
  prism-app pack remove <NAME> [--lib <LIB_DIR>]

子命令:
  add     校验包目录的 manifest（semver/许可证/资产清单）后登记到本地库
  list    列出本地库已登记的数据包
  remove  按包名从本地库移除

选项:
  --lib <DIR>   本地库根目录（默认 $PRISM_HOME 或 ~/.prism/packs）
  -h, --help    显示本帮助

包目录须含 manifest.toml（name(kebab-case)/version/license/provides/assets）。"#;

/// 解析库根目录：`--lib` > `$PRISM_HOME/packs` > `~/.prism/packs`。
fn resolve_lib(args: &[String]) -> Result<(PathBuf, Vec<String>), String> {
    let mut lib: Option<PathBuf> = None;
    let mut rest: Vec<String> = Vec::new();
    let mut i: usize = 0;
    while i < args.len() {
        if args[i] == "--lib" {
            i += 1;
            match args.get(i) {
                Some(v) => lib = Some(PathBuf::from(v)),
                None => return Err("--lib 缺少取值".to_string()),
            }
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    let root: PathBuf = match lib {
        Some(p) => p,
        None => {
            let base: PathBuf = match std::env::var("PRISM_HOME") {
                Ok(h) => PathBuf::from(h),
                Err(_) => match std::env::var("HOME") {
                    Ok(h) => PathBuf::from(h).join(".prism"),
                    Err(_) => PathBuf::from(".prism"),
                },
            };
            base.join("packs")
        }
    };
    Ok((root, rest))
}

/// 运行 pack 子命令；返回进程退出码。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        println!("{PACK_USAGE}");
        return if args.is_empty() { 1 } else { 0 };
    }
    let (root, rest) = match resolve_lib(args) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("prism-app pack: {e}");
            return 1;
        }
    };
    let lib = PackLibrary::new(root);
    match rest.first().map(String::as_str) {
        Some("add") => {
            let dir: &str = match rest.get(1) {
                Some(d) => d,
                None => {
                    eprintln!("prism-app pack add: 缺少 <PACK_DIR>");
                    return 1;
                }
            };
            match lib.add(Path::new(dir)) {
                Ok(e) => {
                    println!(
                        "已登记数据包 '{}' v{}.{}.{} ({})",
                        e.name, e.version.0, e.version.1, e.version.2, e.license
                    );
                    0
                }
                Err(e) => {
                    eprintln!("prism-app pack add: {}", e.message);
                    1
                }
            }
        }
        Some("list") => {
            let entries = lib.list();
            if entries.is_empty() {
                println!("（本地库为空：{}）", lib.index_path().display());
            } else {
                for e in &entries {
                    println!(
                        "{}  v{}.{}.{}  [{}]  {}",
                        e.name,
                        e.version.0,
                        e.version.1,
                        e.version.2,
                        e.license,
                        e.path.display()
                    );
                }
            }
            0
        }
        Some("remove") => {
            let name: &str = match rest.get(1) {
                Some(n) => n,
                None => {
                    eprintln!("prism-app pack remove: 缺少 <NAME>");
                    return 1;
                }
            };
            match lib.remove(name) {
                Ok(e) => {
                    println!("已移除数据包 '{}'", e.name);
                    0
                }
                Err(e) => {
                    eprintln!("prism-app pack remove: {}", e.message);
                    1
                }
            }
        }
        Some(other) => {
            eprintln!("prism-app pack: 未知子命令 `{other}`");
            eprintln!("{PACK_USAGE}");
            1
        }
        None => {
            println!("{PACK_USAGE}");
            0
        }
    }
}
