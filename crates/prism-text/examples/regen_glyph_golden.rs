//! 字形位图 golden 生成器（**工具形态，非常规测试**）。
//!
//! # 为什么是 `examples/` 而不是 `#[ignore]` 测试
//!
//! 本文件原先是 `tests/qa_edge_attack.rs` 里一条 `#[ignore]` 的
//! `regenerate_glyph_golden`。它**不是待修缺陷**，而是一个**写文件的生成器**：
//! 把它挂在 `--ignored` 列表里，等于让"待办"与"缺陷"混淆，且让全仓验收
//! 长期挂着 `ignored` 计数——这与"注释掉测试"是同一类假绿手段。
//! 故改为 `cargo run --example regen_glyph_golden` 的独立入口：
//! - **不进 `cargo test`**，不产生 `ignored` 计数；
//! - 形态与职责自明（生成器就该是可执行程序，不是测试）；
//! - 默认 `--check`：**只比对不写盘**，误跑不会污染 golden。
//!
//! # 用法
//!
//! ```bash
//! # 默认：只比对当前实现与 tests/golden/ 是否一致（CI 友好，退出码非零即不一致）
//! cargo run -p prism-text --example regen_glyph_golden
//!
//! # 显式确认：仍然只比对，不写盘
//! cargo run -p prism-text --example regen_glyph_golden -- --check
//!
//! # 确认实现变更**就是**期望的 golden 变更后才写盘
//! cargo run -p prism-text --example regen_glyph_golden -- --write
//! ```
//!
//! # 纪律
//!
//! `--write` 会覆盖 `tests/golden/*.txt`。**先跑 `--check` 看差异**，
//! 确认是预期的实现变更而非回归，再写盘，并把差异写进提交说明。

use prism_text::font_regular;
use prism_text::raster::render_glyph;

/// golden 用例：`(字符, 字号)`。
///
/// 必须与 `tests/qa_edge_attack.rs` 的 `glyph_rasterization_matches_golden`
/// 完全一致——否则比对会漏项（生成器覆盖不到被测用例）。
const CASES: [(char, f32); 4] = [('中', 32.0), ('文', 32.0), ('A', 24.0), ('g', 24.0)];

/// FNV-1a 32 位（自研哈希——与 `tests/qa_edge_attack.rs` 的实现逐位一致）。
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// 字形位图摘要：`w= h= left= top= fnv=`（golden 文件的正文格式）。
fn glyph_digest(bmp: &prism_text::raster::GlyphBitmap) -> String {
    format!(
        "w={} h={} left={} top={} fnv={:08x}",
        bmp.width,
        bmp.height,
        bmp.left,
        bmp.top,
        fnv1a(&bmp.alpha)
    )
}

/// golden 文件绝对路径（写到 `CARGO_MANIFEST_DIR` 下，与测试同源）。
fn golden_path(ch: char, size: f32) -> std::path::PathBuf {
    format!(
        "{}/tests/golden/glyph_{}_{}.txt",
        env!("CARGO_MANIFEST_DIR"),
        ch as u32,
        size as u32
    )
    .into()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write = args.iter().any(|a| a == "--write");
    for arg in &args {
        assert!(
            arg == "--write" || arg == "--check",
            "未知参数 {arg}（支持：--check（默认）/ --write）"
        );
    }

    let f = font_regular();
    let mut mismatched: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    for (ch, size) in CASES {
        let gid = f.glyph_index(ch).unwrap_or(0);
        let bmp = render_glyph(&f, gid, size);
        let digest = glyph_digest(&bmp);
        let path = golden_path(ch, size);
        let rel = path
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&path)
            .display()
            .to_string();

        match std::fs::read_to_string(&path) {
            Ok(existing) if existing.trim() == digest => {
                println!("[一致] {ch}@{size}px -> {rel}");
            }
            Ok(existing) => {
                println!("[不一致] {ch}@{size}px -> {rel}");
                println!("    golden: {}", existing.trim());
                println!("    实际  : {digest}");
                mismatched.push(rel);
            }
            Err(_) => {
                println!("[缺失] {ch}@{size}px -> {rel}（实际应为 {digest}）");
                missing.push(rel);
            }
        }
    }

    if write {
        if !mismatched.is_empty() || !missing.is_empty() {
            println!("\n--write：写盘 {} 个文件。", mismatched.len() + missing.len());
        }
        for (ch, size) in CASES {
            let gid = f.glyph_index(ch).unwrap_or(0);
            let bmp = render_glyph(&f, gid, size);
        let digest = glyph_digest(&bmp);
            let path = golden_path(ch, size);
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).expect("创建 golden 目录");
            }
            std::fs::write(&path, digest).expect("写 golden 文件");
        }
        println!("已写盘全部 {} 个 golden 文件。", CASES.len());
        return;
    }

    // --check（默认）：有差异即非零退出，便于挂 CI/CI-friendly 门禁。
    if mismatched.is_empty() && missing.is_empty() {
        println!("\n全部 {} 个 golden 与当前实现一致。", CASES.len());
        return;
    }
    eprintln!(
        "\n{} 个不一致、{} 个缺失：{}",
        mismatched.len(),
        missing.len(),
        mismatched.iter().chain(missing.iter()).cloned().collect::<Vec<_>>().join("、")
    );
    eprintln!("确认是预期变更后，用 `--write` 写盘。");
    std::process::exit(1);
}