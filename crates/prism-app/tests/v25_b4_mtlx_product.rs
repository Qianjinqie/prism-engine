//! V2.5-B4 产品路径实跑判据：`prism-app mtlx <FILE>` 真进程。
//!
//! 与 `crates/prism-io/tests/v25_b4_materialx.rs` 的库侧判据互补：
//! 这里用 `CARGO_BIN_EXE_prism-app` 起真实二进制，断言 CLI **真的消费**
//! 覆盖度表与 BSDF 映射（不是只加了库函数），双跑 stdout 逐字节一致，
//! 负控走结构化错误且退出码 1。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const EXE: &str = env!("CARGO_BIN_EXE_prism-app");

const MTLX: &str = r#"<materialx version="1.0"><materyialgraph><material name="m"><nodegraph>
<node name="n" type="standard_surface">
 <input name="base_color" type="color3" value="0.2, 0.4, 0.6"/>
 <input name="roughness" type="float" value="0.3"/>
 <input name="metallic" type="float" value="0.7"/>
 <output name="out"/>
</node>
<output name="surface" type="material"><connectfrom node="n" output="out"/></output>
</nodegraph></material></materyialgraph></materialx>"#;

/// 独立临时目录（幂等，不污染仓库）。
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("prism_v25_b4_product").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("创建临时目录应成功");
    dir
}

fn run(path: &str) -> Output {
    Command::new(EXE)
        .args(["mtlx", path])
        .output()
        .expect("启动 prism-app 二进制应成功")
}

#[test]
fn v5_cli_prints_coverage_numbers_and_is_non_empty() {
    let dir = temp_dir("ok");
    let file = dir.join("material.mtlx");
    fs::write(&file, MTLX).unwrap();
    let out = run(file.to_str().unwrap());
    assert_eq!(out.status.code(), Some(0), "合法 .mtlx 必须退出 0：{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("prism-app mtlx 完成"), "成功标识：{stdout}");
    assert!(stdout.contains("材质        : 1"), "材质数非空：{stdout}");
    assert!(stdout.contains("编译通过节点: 1"), "编译通过节点数非空：{stdout}");
    assert!(stdout.contains("21 种节点 / 45 别名 / 17 个 BSDF 标量参数槽"), "覆盖度判据数字：{stdout}");
    assert!(stdout.contains("BSDF 映射   : 1 条"), "BSDF 映射条数非空：{stdout}");
    assert!(stdout.contains("参数签名    : 0x"), "参数签名非空：{stdout}");
    println!("V5 CLI：退出 0，覆盖度 21/45/17 与 BSDF 映射 1 条全部打印");
}

#[test]
fn v5_cli_double_run_stdout_is_byte_identical() {
    let dir = temp_dir("double");
    let file = dir.join("material.mtlx");
    fs::write(&file, MTLX).unwrap();
    let a = run(file.to_str().unwrap());
    let b = run(file.to_str().unwrap());
    assert_eq!(a.status.code(), Some(0));
    assert_eq!(b.status.code(), Some(0));
    assert_eq!(a.stdout, b.stdout, "双跑 stdout 逐字节一致");
    assert_eq!(a.stderr, b.stderr, "双跑 stderr 逐字节一致");
    println!("V5 双跑：stdout {} B / stderr {} B 逐字节一致", a.stdout.len(), a.stderr.len());
}

#[test]
fn v5_cli_negative_control_exits_one_with_structured_error() {
    let dir = temp_dir("bad");
    let file = dir.join("bad.mtlx");
    fs::write(&file, MTLX.replace("standard_surface", "quantum_bsdf")).unwrap();
    let out = run(file.to_str().unwrap());
    assert_eq!(out.status.code(), Some(1), "未知节点必须退出 1");
    assert!(out.stdout.is_empty(), "失败时不打印成功摘要");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("MX2004"), "结构化错误码：{stderr}");
    assert!(stderr.contains("在哪"), "三段式【在哪】：{stderr}");
    assert!(stderr.contains("什么事"), "三段式【什么事】：{stderr}");
    assert!(stderr.contains("建议"), "三段式【怎么改】：{stderr}");
    println!("V5 负控：退出 1 + MX2004 三段式，0 静默降级");
}

#[test]
fn v5_cli_help_lists_three_container_formats_and_usd_command() {
    let out = Command::new(EXE).arg("-h").output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("prism-app mtlx <FILE>"), "用法文本必须列 mtlx：{stdout}");
    assert!(stdout.contains("prism-app usd <FILE>"), "用法文本必须列 usd：{stdout}");
    assert!(stdout.contains(".usda / .usdc / .usdz"), "三种 USD 容器必须明示：{stdout}");
}
