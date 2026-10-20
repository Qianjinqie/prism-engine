//! `prism-app render` 子命令端到端测试。
//!
//! 用 `CARGO_BIN_EXE_prism-app` 起真实二进制进程，验证：
//! 1. 合法 v2 示例 → 退出码 0 + 输出 MP4 文件非空（内置 MJPEG 出口，
//!    不依赖外部 ffmpeg，保证任何环境可跑）；
//! 2. 语法错误 → 退出码 1 + stderr 含行号（P 码消息）；
//! 3. 不支持的节点（particles）→ 退出码 1 + stderr 含 R5010（明确报错，
//!    绝不静默糊弄）。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

/// 最小合法 v2 示例：camera / material / sphere / text 四类已支持节点
/// + title_card 展开形态（opacity 淡入淡出轨道）。1s @ 8fps = 8 帧。
const MINIMAL_PRISM: &str = r#"// prism-dsl 1
movie "cli-min" { duration 1s fps 8 resolution 96x54 format png }
scene "s" {
  duration 1s
  stage3d {
    camera main { orbit { center 0,0,0 radius 6 } fov 45 }
    sdf ball { kind sphere radius 1.0 }
    material hero { albedo #e2b53f roughness 0.35 metalness 0.05 }
  }
  layer text title { content "CLI" x 48 y 24 size 14px color #ffffff opacity { key 0s 0 linear key 0.5s 1 linear key 1s 0 linear } }
}
"#;

/// 起子进程跑 `prism-app render <file> --codec mjpeg --out <out>`。
fn run_render(prism_path: &PathBuf, out_path: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism-app"))
        .args([
            "render",
            prism_path.to_str().expect("路径应可转 str"),
            "--codec",
            "mjpeg",
            "--out",
            out_path.to_str().expect("路径应可转 str"),
        ])
        .output()
        .expect("子进程应可启动")
}

/// 把源码写到临时目录并返回路径（每次调用唯一文件名，避免并行测试互踩）。
fn write_temp_prism(tag: &str, source: &str) -> PathBuf {
    let dir: PathBuf = std::env::temp_dir().join(format!("prism-render-cli-{tag}"));
    fs::create_dir_all(&dir).expect("临时目录应可创建");
    let path: PathBuf = dir.join("case.prism");
    fs::write(&path, source).expect("临时 .prism 应可写");
    path
}

#[test]
fn render_cli_exits_zero_and_writes_nonempty_mp4() {
    let prism: PathBuf = write_temp_prism("ok", MINIMAL_PRISM);
    let out: PathBuf = std::env::temp_dir()
        .join("prism-render-cli-ok")
        .join("out.mp4");

    let output: Output = run_render(&prism, &out);

    assert!(
        output.status.success(),
        "退出码应为 0；stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes: Vec<u8> = fs::read(&out).expect("输出 MP4 应存在");
    assert!(!bytes.is_empty(), "输出 MP4 不应为空");
    // MP4 容器魔数（……ftyp）：前 12 字节内应出现 b"ftyp"。
    assert!(
        bytes.len() >= 12 && &bytes[4..8] == b"ftyp",
        "输出应是 MP4 容器（ftyp box），实际头部: {:?}",
        &bytes[..bytes.len().min(16)]
    );
    // stdout 应输出帧数（8 帧 = 1s @ 8fps）。
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("帧数      : 8 帧"), "stdout 应含帧数输出: {stdout}");
}

#[test]
fn render_cli_syntax_error_reports_line_and_exits_one() {
    let bad: &str = "// prism-dsl 1\nmovie \"x\" { duration 1s fps 8 }\nscene \"s\" { duration bogus }\n";
    let prism: PathBuf = write_temp_prism("syntax", bad);
    let out: PathBuf = std::env::temp_dir()
        .join("prism-render-cli-syntax")
        .join("out.mp4");

    let output: Output = run_render(&prism, &out);

    assert!(!output.status.success(), "语法错误应以退出码 1 终止");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("第 3 行"), "错误应带行号: {stderr}");
    assert!(stderr.contains("[P2102]"), "错误应带 DSL 防呆码: {stderr}");
    assert!(!out.exists(), "失败时不应留下输出文件");
}

#[test]
fn render_cli_unsupported_node_reports_r5010_and_exits_one() {
    let bad: &str = "// prism-dsl 1\nmovie \"x\" { duration 1s fps 8 resolution 96x54 }\nscene \"s\" {\n  duration 1s\n  layer particles { count 50 speed 1 }\n}\n";
    let prism: PathBuf = write_temp_prism("particles", bad);
    let out: PathBuf = std::env::temp_dir()
        .join("prism-render-cli-particles")
        .join("out.mp4");

    let output: Output = run_render(&prism, &out);

    assert!(!output.status.success(), "不支持的节点应以退出码 1 终止");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("R5010") && stderr.contains("粒子"),
        "应明确报出 R5010 与节点类型: {stderr}"
    );
    assert!(!out.exists(), "失败时不应留下输出文件");
}
