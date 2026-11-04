//! `prism-app mtlx` 子命令：V2.5-B4 产品路径薄转发。
//!
//! 真实编译、覆盖度统计、逐节点参数映射与摘要文本全部由
//! `prism_io::materialx::compile_mtlx_report` 完成；本文件只做参数校验、
//! 调用与结果打印。应用 crate 受既有静态门禁 `w7_preview` 约束
//! （`src/` 内不得出现散射模型词汇），故相关词汇不下沉到本 crate。

use prism_io::materialx;

/// `mtlx` 子命令用法。
pub const MTLX_USAGE: &str = r#"prism-app mtlx — 编译 MaterialX 节点图 → 特解参数包

用法:
  prism-app mtlx <FILE.mtlx>

说明:
  V2.5-B4 覆盖度判据表：21 种节点 / 45 别名 / 17 个标量参数槽；
  未知节点 MX2004、循环图 MX2007、类型错配 MX2005 结构化报错，不静默降级。

示例:
  prism-app mtlx material.mtlx"#;

/// 运行 `mtlx` 子命令；返回进程退出码（0 成功 / 1 失败）。
pub fn run(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{MTLX_USAGE}");
        return 0;
    }
    if args.len() != 1 {
        eprintln!("prism-app mtlx: 需要且只需要一个 <FILE.mtlx> 参数");
        eprintln!("{MTLX_USAGE}");
        return 1;
    }
    let path: &str = args[0].as_str();
    match materialx::compile_mtlx_report(path) {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(err) => {
            eprintln!(
                "prism-app mtlx: 在哪：编译 `{path}`\n  什么事：{err}\n  建议：检查节点 type / 端口类型与连线端点"
            );
            1
        }
    }
}
