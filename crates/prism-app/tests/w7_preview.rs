//! W7 / B3 的**静态门禁**（选型报告 §9.1 的 P2）。
//!
//! # 这条门禁在防什么
//!
//! B3 之前的 `preview.rs` 有两套事实源：产品出片走 `prism-rt` 路径追踪，
//! 预览走 `CpuRefBackend`（单弹射 Lambert）+ 一张手工几何图案
//! （`leaf_pattern` / `pattern_family` / `pattern_color`）。后果不是"预览略糙"，
//! 而是**预览与出片可以各自对、合起来错**——用户按预览调好的画面，出片是另一张。
//!
//! W7 的处置是把预览接到产品同一条通路（`prism_render::RtBackend` +
//! `render_frame`），并**删掉整条图案链**。本文件把这件事变成可执行的断言：
//!
//! | 断言 | 拦住的退化 |
//! |---|---|
//! | 全 crate 零 `dot(` / `bsdf` / `hemisphere` / `fresnel` / … | 有人为了"预览快一点"再写一个着色函数 |
//! | 全 crate 零 `leaf_pattern` / `pattern_family` / `pattern_color` / `value_noise` / `fbm` | 图案路径被"临时"恢复 |
//! | `preview.rs` 必须含 `RtBackend::new(` + `render_frame(` | 有人悄悄换掉出帧通路（门禁变成空断言） |
//! | `src/editor/**` 零 `CpuRefBackend` | 预览回退到参考后端 |
//!
//! 这四条一起才成立：只查"没有坏东西"会漏掉"什么都没有"（比如 preview.rs 被
//! 掏空后照样零命中），所以必须有正向接线断言。
//!
//! # 为什么是文本级 grep 而不是类型级
//!
//! 判据原文就是"grep 断言零命中"（§9.1 P2）。文本级门禁的好处是**无法被
//! 类型系统绕过**：任何"我造个新 trait 再实现一遍光照"的路子都会留下
//! `dot(` / `bsdf` 这类词。代价是可能误伤注释——所以注释里也不许出现这些词
//! （见各条 needle 的注释）：门禁代码本身是本文件，不在 `src/` 内，不受影响。

use std::path::{Path, PathBuf};

/// 递归收集 `<crate>/src` 下全部 `.rs` 文件（路径升序，报错信息确定）。
fn collect_sources() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut items: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        items.sort();
        for path in items {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out: Vec<PathBuf> = Vec::new();
    walk(&root, &mut out);
    assert!(!out.is_empty(), "未在 {} 下找到任何 .rs 源文件", root.display());
    out
}

/// 在**全部**源文件里找针，返回 `文件:行号: 行内容` 形式的人类可读命中列表。
///
/// 匹配大小写不敏感（`BSDF` / `Bsdf` 都算），命中列表按文件/行号升序。
fn find_all(needles: &[&str]) -> Vec<String> {
    let mut hits: Vec<String> = Vec::new();
    for path in collect_sources() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let lower: String = text.to_lowercase();
        let lines: Vec<&str> = lower.lines().collect();
        for needle in needles {
            for (i, line) in lines.iter().enumerate() {
                if line.contains(needle) {
                    hits.push(format!(
                        "{}:{}: {}",
                        path.display(),
                        i + 1,
                        text.lines().nth(i).unwrap_or("").trim()
                    ));
                }
            }
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

/// 在单个文件里找针（大小写不敏感）。
fn find_in(path: &Path, needle: &str) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let lower: String = text.to_lowercase();
    let mut hits: Vec<String> = Vec::new();
    for (i, line) in lower.lines().enumerate() {
        if line.contains(needle) {
            hits.push(format!(
                "{}:{}: {}",
                path.display(),
                i + 1,
                text.lines().nth(i).unwrap_or("").trim()
            ));
        }
    }
    hits
}

/// **P2-a：`prism-app` 全 crate 不得出现光照计算。**
///
/// 针表（每条都对应一个"自己写一套着色"的具体形状）：
///
/// - `dot(`：法线点乘是 Lambert/GGX/BRDF 的第一个特征式。本 crate 的画像素
///   全部发生在 `prism-rt`，`prism-app` 侧零几何点乘需求（唯一的方向向量处理
///   是 `normalize()`，那是**场景参数**不是着色）。
/// - `bsdf` / `brdf` / `hemisphere` / `fresnel` / `schlick` / `refract`：
///   散射模型词汇，出现即说明有人在算光。
/// - `radiance` / `throughput` / `cos_theta` / `next_event` / `nee` /
///   `phase(`：路径追踪内部量，属于 `prism-rt`。
/// - `lambert`：参考后端的着色模型名（`--fast` 的 CLI 帮助文本里出现过，
///   但那是 `prism-app` 的**产品**档位说明）。为避免误伤，`lambert` 只在
///   `editor/` 子树里禁（见 P2-c）；本条的针表不含它。
#[test]
fn p2_no_shading_math_in_app_crate() {
    let needles: [&str; 12] = [
        "dot(",
        "bsdf",
        "brdf",
        "hemisphere",
        "fresnel",
        "schlick",
        "refract",
        "radiance",
        "throughput",
        "cos_theta",
        "next_event",
        "phase(",
    ];
    let hits: Vec<String> = find_all(&needles);
    assert!(
        hits.is_empty(),
        "P2 破：prism-app 出现光照计算（预览不得有第二套着色函数），命中 {} 处：\n{}",
        hits.len(),
        hits.join("\n")
    );
}

/// **P2-b：W7 删掉的 Draft 图案链必须保持删除状态。**
///
/// `leaf_pattern` / `pattern_family` / `pattern_color` 是 C2 点名的旧图案路径；
/// `value_noise` / `fbm` / `lattice` / `smoothstep` / `jitter_offset` 是它的
/// 零件。任何一件回来，"预览 ≠ 产品通路"就回来了。
///
/// （`smoothstep` 在 `prism-app` 侧没有合法用途：色调映射在 `prism-aesthetic`，
/// 采样点在 `prism-rt`。）
#[test]
fn p2_draft_pattern_code_is_gone() {
    let needles: [&str; 9] = [
        "leaf_pattern",
        "pattern_family",
        "pattern_color",
        "value_noise",
        "fbm",
        "lattice",
        "smoothstep",
        "jitter_offset",
        "to_rgba",
    ];
    let hits: Vec<String> = find_all(&needles);
    assert!(
        hits.is_empty(),
        "P2 破：旧 Draft 图案链的代码又出现了（§4.4 红线 1：预览不得另起近似路径），命中 {} 处：\n{}",
        hits.len(),
        hits.join("\n")
    );
}

/// **P2-c：预览必须正向接在产品通路上，且不得挂参考后端。**
///
/// 这条是 P2-a/P2-b 的补集：前两条只证明"没写坏东西"，本条证明"确实接对了"。
///
/// 断言：
/// 1. `editor/preview.rs` 含 `RtBackend::new(` 与 `render_frame(`——出帧通路
///    就是 `prism-render` 的 `prism-rt` 适配器 + 产品调度函数；
/// 2. `editor/` 子树零 `CpuRefBackend`——预览不得回退到参考后端；
/// 3. `editor/` 子树零 `lambert`——同上（参考后端的着色模型名）。
#[test]
fn p2_preview_uses_product_rt_pipeline() {
    let manifest: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let preview: PathBuf = manifest.join("src/editor/preview.rs");
    assert!(preview.is_file(), "找不到 {}", preview.display());

    for needle in ["rtbackend::new(", "render_frame("] {
        assert!(
            !find_in(&preview, needle).is_empty(),
            "P2 破：preview.rs 未出现 `{needle}`——出帧通路不在产品管线上"
        );
    }

    let mut editor_files: Vec<PathBuf> = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut items: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        items.sort();
        for path in items {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    walk(&manifest.join("src/editor"), &mut editor_files);
    assert!(!editor_files.is_empty(), "editor 子树为空（结构已变，门禁需重审）");

    let mut hits: Vec<String> = Vec::new();
    for path in &editor_files {
        hits.extend(find_in(path, "cpurefbackend"));
        hits.extend(find_in(path, "lambert"));
    }
    assert!(
        hits.is_empty(),
        "P2 破：编辑器预览挂着参考后端（预览与出片成为两个事实源），命中：\n{}",
        hits.join("\n")
    );
}
