//! `prism-app usd` 子命令：V2.5-B2 产品路径接线。
//!
//! 一个命令吃掉三种容器：`.usda`（文本）/ `.usdc`（二进制 Crate）/
//! `.usdz`（zip 包）。统一走 `prism_io::usd::parse_asset`（内容按 magic 分派），
//! 再经 `usd::build_scene` 出可渲染场景并打印非空摘要。
//!
//! 这就是「DSL / prism-app 能读 .usdc / .usdz」的落点：
//! 运行 DSL v2 场景的同一个 CLI 现在能直接读二进制 USD 资产。
//! DSL 语言面按作者契约 §0 维持现状（V2.5 期间不补表达式/变量/函数/循环）。

use std::path::Path;

/// `usd` 子命令用法。
pub const USD_USAGE: &str = r#"prism-app usd — 读取 USD 资产（.usda / .usdc / .usdz）

用法:
  prism-app usd <FILE>

参数:
  <FILE>   USD 层文件：.usda 文本、.usdc 二进制 Crate、.usdz zip 包

说明:
  三种容器都真解析（不是识别 magic 后拒绝）：.usdc 走 TOC + LZ4 压缩段
  + value rep 重现，.usdz 解包后与裸 .usdc 逐位相同。
  输出 prim 层级 / 属性 / 变换 / 材质绑定的非空摘要。

示例:
  prism-app usd scene.usda
  prism-app usd scene.usdc
  prism-app usd scene.usdz"#;

/// 运行 `usd` 子命令；返回进程退出码（0 成功 / 1 失败）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USD_USAGE}");
        return 0;
    }
    if args.len() != 1 {
        eprintln!("prism-app usd: 需要且只需要一个 <FILE> 参数");
        eprintln!("{USD_USAGE}");
        return 1;
    }
    let path: &str = args[0].as_str();
    if Path::new(path).extension().is_none() {
        eprintln!("prism-app usd: `{path}` 缺扩展名（允许 .usda/.usd/.usdc/.usdz）");
        return 1;
    }
    // 统一产品入口：内容分派在 prism-io::usd::parse_bytes 内完成。
    let parsed = match prism_io::usd::parse_asset(path) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("prism-app usd: 在哪：读取资产 `{path}`\n  什么事：{err}\n  建议：确认文件完整且是 USD 层（.usda/.usd/.usdc/.usdz）");
            return 1;
        }
    };
    let scene = match prism_io::usd::build_scene(&parsed.layer) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("prism-app usd: 在哪：构建场景 `{path}`\n  什么事：{err}\n  建议：检查 prim 的 points / faceVertexCounts / faceVertexIndices 是否自洽");
            return 1;
        }
    };
    let prim_count: usize = parsed
        .layer
        .prims
        .iter()
        .map(|p| p.walk().len())
        .sum();
    if prim_count == 0 {
        eprintln!("prism-app usd: `{path}` 解析出 0 个 prim（空层不视为成功）");
        return 1;
    }
    let bindings: usize = parsed
        .layer
        .prims
        .iter()
        .flat_map(|p| p.walk())
        .map(|p| p.relationships.len())
        .sum();
    println!("prism-app usd 完成");
    println!("  资产      : {path}");
    println!("  层版本    : {}", parsed.layer.version);
    println!("  prim      : {prim_count}");
    println!("  网格      : {}", scene.mesh_count());
    println!("  三角形    : {}", scene.render_triangle_count());
    println!("  关系绑定  : {bindings}");
    println!("  层元数据  : {} 条", parsed.layer.metadata.len());
    0
}
