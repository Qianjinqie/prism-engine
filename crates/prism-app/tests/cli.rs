//! `prism-app` CLI 端到端集成测试（M1 补充）。
//!
//! 通过 `env!("CARGO_BIN_EXE_prism-app")` 拿到编译产物二进制路径，
//! 用 `std::process::Command` 真实起进程验证退出码、stderr/stderr 与落盘产物。
//!
//! 硬约束：零第三方依赖（仅 `std` + 内置 `#[test]`）；
//! 临时目录一律放在 `std::env::temp_dir()` 下的独立子目录，互不干扰、
//! 重复执行前先清理，保证测试幂等且不污染仓库内 `output/`。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

/// 被测二进制（由 cargo 为集成测试自动构建并注入路径）。
const EXE: &str = env!("CARGO_BIN_EXE_prism-app");

/// PNG 文件签名（8 字节）：`89 50 4E 47 0D 0A 1A 0A`。
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// 黄金示例的绝对路径（CARGO_MANIFEST_DIR = crates/prism-app，向上两级到 workspace 根）。
const GOLDEN_EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/spin_red_ball.prism"
);

// ===========================================================================
//  测试脚手架
// ===========================================================================

/// 在 `std::env::temp_dir()/prism_app_cli_tests/<name>` 创建独立临时目录。
/// 先递归删除同名旧目录再创建，保证测试幂等、互不依赖执行顺序。
fn temp_subdir(name: &str) -> PathBuf {
    let dir: PathBuf = std::env::temp_dir().join("prism_app_cli_tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("创建临时输出目录应成功");
    dir
}

/// 运行 prism-app 二进制（继承测试进程的工作目录；所有传参均为绝对路径，
/// 因此断言不依赖进程 CWD）。
fn run_prism(args: &[&str]) -> Output {
    Command::new(EXE).args(args).output().expect("启动 prism-app 二进制应成功")
}

/// 把输出目录 PathBuf 转成命令行参数可用的 &str。
fn as_arg(path: &PathBuf) -> &str {
    path.to_str().expect("临时路径应为合法 UTF-8")
}

// ===========================================================================
//  1. 黄金路径：整段渲染 → 帧序列 + manifest
// ===========================================================================

#[test]
fn cli_golden_path_renders_frame_sequence() {
    let out_dir: PathBuf = temp_subdir("golden");
    let out_arg: &str = as_arg(&out_dir);

    // 缩小分辨率 / 帧率以加速：64x48 @ 4fps，时长仍为 DSL 的 3s → ceil(4*3)=12 帧。
    let output: Output = run_prism(&[
        GOLDEN_EXAMPLE,
        "--out", out_arg,
        "--width", "64",
        "--height", "48",
        "--fps", "4",
    ]);

    assert!(
        output.status.success(),
        "退出码应为 0，实际 {:?}，stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    // 首帧 PNG 存在且头 8 字节为 PNG 签名。
    let frame1: PathBuf = out_dir.join("frames").join("frame_0001.png");
    assert!(frame1.exists(), "frame_0001.png 应存在: {}", frame1.display());
    let bytes: Vec<u8> = fs::read(&frame1).expect("读回 frame_0001.png 应成功");
    assert!(
        bytes.len() >= 8 && bytes[0..8] == PNG_SIGNATURE,
        "frame_0001.png 头 8 字节应为 PNG 签名，实际 {:?}",
        &bytes[..bytes.len().min(8)]
    );

    // manifest.json 存在，count = 4fps × 3s = 12；宽高应取自 CLI 覆盖值。
    let manifest_path: PathBuf = out_dir.join("frames").join("manifest.json");
    assert!(manifest_path.exists(), "manifest.json 应存在");
    let manifest: String = fs::read_to_string(&manifest_path).expect("读回 manifest 应成功");
    assert!(
        manifest.contains("\"count\":12"),
        "manifest 应包含 \"count\":12（4fps×3s），实际: {}",
        manifest
    );
    assert!(manifest.contains("\"width\":64"), "manifest 应反映 CLI 覆盖宽: {}", manifest);
    assert!(manifest.contains("\"height\":48"), "manifest 应反映 CLI 覆盖高: {}", manifest);
}

// ===========================================================================
//  2. 单帧模式：--frame T → preview.png
// ===========================================================================

#[test]
fn cli_single_frame_mode_writes_preview_png() {
    let out_dir: PathBuf = temp_subdir("single_frame");
    let out_arg: &str = as_arg(&out_dir);

    let output: Output = run_prism(&[
        GOLDEN_EXAMPLE,
        "--out", out_arg,
        "--width", "32",
        "--height", "18",
        "--frame", "0.5",
    ]);

    assert!(
        output.status.success(),
        "单帧模式退出码应为 0，stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let preview: PathBuf = out_dir.join("preview.png");
    assert!(preview.exists(), "preview.png 应存在: {}", preview.display());
    let bytes: Vec<u8> = fs::read(&preview).expect("读回 preview.png 应成功");
    assert!(
        bytes.len() >= 8 && bytes[0..8] == PNG_SIGNATURE,
        "preview.png 头 8 字节应为 PNG 签名，实际 {:?}",
        &bytes[..bytes.len().min(8)]
    );
}

// ===========================================================================
//  3. 错误路径：文件不存在 / DSL 编译失败
// ===========================================================================

#[test]
fn cli_missing_file_exits_1_with_stderr() {
    // 不存在的场景文件：退出码 1，stderr 非空（含可读错误原因）。
    let missing: PathBuf = std::env::temp_dir().join("prism_app_cli_tests_no_such_file.prism");
    let _ = fs::remove_file(&missing); // 确保不存在
    let output: Output = run_prism(&[as_arg(&missing)]);

    assert_eq!(output.status.code(), Some(1), "文件不存在应以退出码 1 结束");
    assert!(
        !output.stderr.is_empty(),
        "文件不存在时 stderr 应包含错误说明"
    );
}

#[test]
fn cli_bad_dsl_exits_1_with_dsl_line_prefix() {
    // 非法 DSL（词法错误 '@'）：退出码 1，stderr 包含带行号前缀「DSL 第」。
    let dir: PathBuf = temp_subdir("bad_dsl");
    let bad: PathBuf = dir.join("bad.prism");
    fs::write(&bad, "scene \"x\" {\n  radius @\n}\n").expect("写入坏 DSL 应成功");

    let output: Output = run_prism(&[as_arg(&bad), "--out", as_arg(&dir.join("out"))]);

    assert_eq!(output.status.code(), Some(1), "坏 DSL 应以退出码 1 结束");
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        stderr.contains("DSL 第"),
        "stderr 应包含「DSL 第」行号前缀，实际: {}",
        stderr
    );
}

// ===========================================================================
//  4. --help：退出码 0 + stdout 用法说明
// ===========================================================================

#[test]
fn cli_help_exits_0_with_usage_keywords() {
    for flag in ["--help", "-h"] {
        let output: Output = run_prism(&[flag]);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{} 应以退出码 0 结束",
            flag
        );
        let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
        assert!(
            stdout.contains("用法") && stdout.contains("<file.prism>"),
            "{} 的 stdout 应包含用法关键字（用法 / <file.prism>），实际: {}",
            flag,
            stdout
        );
        assert!(
            stdout.contains("--frame") && stdout.contains("--out"),
            "用法说明应列出核心选项，实际: {}",
            stdout
        );
    }
}

// ===========================================================================
//  5. 重点探查：duration=0 / fps=0 的退化行为（不得 panic、不得崩溃）
// ===========================================================================

#[test]
fn cli_duration_zero_with_frame_mode_does_not_panic() {
    // 探查点：DSL 当前不校验 `duration 0s`。单帧模式下 t 被钳制为 duration=0，
    // `eval_camera_orbit` 内 0/0 → NaN 相机。验收：进程正常退出（0/1），
    // 绝不因 signal 崩溃，也不出现 Rust panic。
    let dir: PathBuf = temp_subdir("dur_zero_frame");
    let src: PathBuf = dir.join("dur0.prism");
    fs::write(
        &src,
        "scene \"x\" {\n  sphere s {\n    radius 1.0\n  }\n}\ntimeline {\n  duration 0s\n  fps 4\n}\n",
    )
    .expect("写入 duration=0 场景应成功");

    let output: Output = run_prism(&[
        as_arg(&src),
        "--out", as_arg(&dir.join("out")),
        "--width", "32",
        "--height", "18",
        "--frame", "0.5",
    ]);

    let code: Option<i32> = output.status.code();
    assert!(
        code == Some(0) || code == Some(1),
        "进程应正常退出（0/1），不得被 signal 终止，实际 {:?}",
        code
    );
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        !stderr.contains("panicked"),
        "duration=0 + --frame 不应 panic，实际 stderr: {}",
        stderr
    );
}

#[test]
fn cli_fps_zero_full_mode_exits_1_without_panic() {
    // 探查点：`fps 0` + 整段模式。DSL 语义校验直接拒绝（fps ≥ 1），
    // 验收：退出码 1、stderr 非空、无 panic。
    let dir: PathBuf = temp_subdir("fps_zero");
    let src: PathBuf = dir.join("fps0.prism");
    fs::write(&src, "scene \"x\" {\n}\ntimeline {\n  duration 1s\n  fps 0\n}\n")
        .expect("写入 fps=0 场景应成功");

    let output: Output = run_prism(&[as_arg(&src), "--out", as_arg(&dir.join("out"))]);

    assert_eq!(output.status.code(), Some(1), "fps=0 应以退出码 1 结束");
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(!stderr.is_empty(), "fps=0 应在 stderr 给出原因");
    assert!(!stderr.contains("panicked"), "fps=0 不应 panic，实际 stderr: {}", stderr);
}

#[test]
fn cli_duration_zero_is_rejected_at_dsl_compile() {
    // 行为变更（M1 收口）：DSL 层现已校验 duration > 0（见 prism-dsl parse_timeline），
    // `duration 0s` 在编译期即被拒绝，不再走到渲染阶段的空帧路径。
    // 验收：退出码 1、stderr 含行号化 DSL 报错、无 panic。
    let dir: PathBuf = temp_subdir("dur_zero_full");
    let src: PathBuf = dir.join("dur0_full.prism");
    fs::write(&src, "scene \"x\" {\n}\ntimeline {\n  duration 0s\n  fps 4\n}\n")
        .expect("写入 duration=0 场景应成功");

    let output: Output = run_prism(&[as_arg(&src), "--out", as_arg(&dir.join("out"))]);

    assert_eq!(output.status.code(), Some(1), "duration=0 应以退出码 1 结束");
    let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        stderr.contains("DSL 第"),
        "应给出 DSL 行号化报错，实际: {}",
        stderr
    );
    assert!(
        stderr.contains("duration"),
        "报错应指向 duration，实际: {}",
        stderr
    );
    assert!(!stderr.contains("panicked"), "编译期拒绝不应 panic，实际 stderr: {}", stderr);
}
