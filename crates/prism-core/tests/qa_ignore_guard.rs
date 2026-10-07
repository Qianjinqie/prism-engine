//! 🔴 防复发守卫：全仓 `#[ignore]` 与 `ignore` doc-test 代码块必须为**零**。
//!
//! # 为什么需要这条门禁
//!
//! 本项目历史上栽过"假绿第二次出现"——模块被注释禁用仍报全绿。
//! `#[ignore]` 与"注释掉测试"是**同一类假绿手段**：它让测试从验收清单上消失，
//! 却让 `cargo test` 继续报 0 failed。本守卫把"消失"这件事本身变成可测量的：
//! **全仓任何 `.rs` 文件里出现一处`#[ignore]`，或任何 `src/` 下的 doc 注释
//! 出现一处 ```` ```ignore ```` 代码块，本测试即判红。**
//!
//! 全仓长期挂着的 4 条已各自处置（见 `docs/` 与各测试文件注释）：
//! - `prism-audio` `exp_f64(-inf)` 挂死 → **真缺陷**，修被测函数 + 测试转正；
//! - `prism-text` 字形 golden 生成器 → **生成器**，移到 `examples/regen_glyph_golden.rs`；
//! - `prism-aesthetic` / `prism-core` 两条 ```` ```ignore ```` → **示例写不成**，
//!   补齐真实可编译可运行的 doctest。
//!
//! # 反向自证（纪律：不许写恒过断言）
//!
//! 本文件带三层非恒真空洞证明，任一失效都会让守卫**自己**判红：
//! 1. `detector_rejects_a_real_ignore` 等合成用例：把已知正例喂给探测器，
//!    断言它**必须**报出来（探测器失效 → 红）。
//! 2. `detector_ignores_prose_mentions` 合成负例：注释里提到 ``#[ignore]`` 字样
//!    **不得**误报（否则守卫会因文档提及而恒红，等于没守）。
//! 3. `scan_covers_the_workspace`：断言实际扫描文件数 ≥ 下界，且本文件
//!    恰好是唯一被排除的文件（防止"扫了 0 个文件所以全绿"）。
//!
//! 人工反向自证：随便加一个 `#[ignore]` → 本守卫必须判红。

use std::path::{Path, PathBuf};

// ===========================================================================
//  一、探测器：把源码里的注释剥掉，只留可编译代码，再找 `#[ignore]`
// ===========================================================================

/// 命中的一处 `#[ignore]`。
#[derive(Debug, PartialEq, Eq)]
struct Hit {
    /// 相对工作区根的路径（如 `crates/prism-audio/tests/x.rs`）。
    file: String,
    /// 1基行号。
    line: usize,
    /// 该行的代码部分（注释已剥离），用于报错定位。
    code: String,
}

/// 逐字符扫描源码，返回**剥离了注释**的代码行与原行号的一一对应表。
///
/// 处理的注释/字面量形态：
/// - 行注释 `//`（含 `///` `//!`）；
/// - 块注释 `/* */`（Rust 允许**嵌套**，用深度计数）；
/// - 普通字符串 `"..."`（含 `\` 转义）与 raw string `r"..."` / `r#"..."#`；
/// - char 字面量 `'x'` / `'\n'`；生命周期 `'a` 不当作 char 字面量。
///
/// 字符串/char 内的内容会被替换成等长空格，故**行号与列宽保持不变**，
/// 报错能直接指回原文。
fn strip_comments(src: &str) -> Vec<String> {
    let b: Vec<char> = src.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut i = 0usize;
    let mut block_depth = 0usize;
    // 原始字符串的哈希数（None 表示不在 raw string 中）。
    let mut raw_hashes: Option<usize> = None;

    let flush = |cur: &mut String, out: &mut Vec<String>| {
        out.push(std::mem::take(cur));
    };

    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();

        if let Some(h) = raw_hashes {
            // raw string 内部：只有 `"` + h 个 `#` 才结束。
            if c == '"' && b[i + 1..].iter().take(h).filter(|&&x| x == '#').count() == h {
                let mut k = i + 1 + h;
                while k < b.len() {
                    cur.push(' ');
                    k += 1;
                }
                i = k;
                raw_hashes = None;
                continue;
            }
            if c == '\n' {
                flush(&mut cur, &mut out);
            } else {
                cur.push(' ');
            }
            i += 1;
            continue;
        }

        if block_depth > 0 {
            if c == '/' && next == Some('*') {
                block_depth += 1;
                cur.push(' ');
                cur.push(' ');
                i += 2;
                continue;
            }
            if c == '*' && next == Some('/') {
                block_depth -= 1;
                cur.push(' ');
                cur.push(' ');
                i += 2;
                continue;
            }
            if c == '\n' {
                flush(&mut cur, &mut out);
            } else {
                cur.push(' ');
            }
            i += 1;
            continue;
        }

        match (c, next) {
            // 行注释
            ('/', Some('/')) => {
                while i < b.len() && b[i] != '\n' {
                    cur.push(' ');
                    i += 1;
                }
            }
            // 块注释
            ('/', Some('*')) => {
                block_depth = 1;
                cur.push(' ');
                cur.push(' ');
                i += 2;
            }
            // raw string 起始：r" / r#" / br##" 等（只处理 r 前缀，b/br 变体
            // 在本仓库不出现真raw 字符串，若出现则 b/r 会被当普通标识符跳过）。
            ('r', Some('"')) => {
                raw_hashes = Some(0);
                cur.push(' ');
                cur.push(' ');
                i += 2;
            }
            ('r', Some('#')) => {
                let mut h = 0usize;
                let mut k = i + 1;
                while b.get(k) == Some(&'#') {
                    h += 1;
                    k += 1;
                }
                if b.get(k) == Some(&'"') {
                    raw_hashes = Some(h);
                    while i <= k {
                        cur.push(' ');
                        i += 1;
                    }
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            // 普通字符串
            ('"', _) => {
                cur.push(' ');
                i += 1;
                while i < b.len() {
                    if b[i] == '\\' && i + 1 < b.len() {
                        cur.push(' ');
                        cur.push(' ');
                        i += 2;
                        continue;
                    }
                    let ch = b[i];
                    cur.push(if ch == '\n' { '\n' } else { ' ' });
                    i += 1;
                    if ch == '"' {
                        break;
                    }
                }
            }
            // char 字面量：'\n' / 'x' / '\\'；生命周期 'a 不匹配此模式，按普通字符处理。
            ('\'', Some('\\')) => {
                if i + 2 < b.len() {
                    cur.push(' ');
                    cur.push(' ');
                    cur.push(' ');
                    i += 3;
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
            ('\'', Some(_)) if b.get(i + 2) == Some(&'\'') => {
                cur.push(' ');
                cur.push(' ');
                cur.push(' ');
                i += 3;
            }
            _ => {
                if c == '\n' {
                    flush(&mut cur, &mut out);
                } else {
                    cur.push(c);
                }
                i += 1;
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// 该代码行是否含 `#[ignore]` 属性。
///
/// 只认三种精确形态（Rust 属性语法），避免 `#[ignored]` 之类误判：
/// `#[ignore]`、`#[ignore =`、行内出现 `#[ignore]` 后紧跟空白/`]`。
fn code_line_has_ignore(code: &str) -> bool {
    let needle = "#[ignore";
    let bytes: Vec<char> = code.chars().collect();
    let n: Vec<char> = needle.chars().collect();
    let mut i = 0usize;
    while i + n.len() <= bytes.len() {
        if bytes[i..i + n.len()] == n[..] {
            // 下一个字符必须是 `]`、`=` 或空白之一。
            match bytes.get(i + n.len()) {
                Some(']') | Some('=') => return true,
                Some(&c) if c.is_whitespace() => return true,
                None => return true,
                _ => {}
            }
        }
        i += 1;
    }
    false
}

/// 该**原始行**是否是 ```` ```ignore ```` doc-test 代码块围栏。
///
/// 围栏总是写在 doc 注释里（rustdoc 只从注释收集 doctest），故先剥掉行首的
/// `///` / `//!` / `*` / 空白，再比对围栏信息串。
///
/// 只在 `src/` 下生效：`tests/` 与 `examples/` 里的 `//!` / `///` 注释
/// **不产生 doctest**（rustdoc 只对库目标收集），故那里的 ```` ```ignore ````
/// 不计入 `cargo test` 的 ignored 计数，不在本守卫管辖内。
fn code_line_has_ignore_fence(raw_line: &str) -> bool {
    let mut t = raw_line.trim();
    for prefix in ["///", "//!", "//", "*"] {
        if let Some(rest) = t.strip_prefix(prefix) {
            t = rest.trim_start();
            break;
        }
    }
    t == "```ignore" || t.starts_with("```ignore,") || t.starts_with("```ignore ")
}

/// 在单个文件里找 `#[ignore]` 与 ignore 围栏。
///
/// 两者**检查的来源不同**，这是关键：
/// - `#[ignore]` 是可编译代码里的属性 → 必须在**剥离注释后**的代码行里找
///   （否则注释里提到 `#[ignore]` 会误报）。
/// - ```` ```ignore ```` 围栏**只存在于 doc 注释内**（rustdoc 从注释里收集
///   doctest），剥离注释会把它一起抹掉→ 必须在**原始行**里找。
fn scan_source(rel: &str, src: &str, check_fences: bool) -> Vec<Hit> {
    let stripped = strip_comments(src);
    let raw: Vec<&str> = src.split('\n').collect();
    let mut hits = Vec::new();
    // 以剥离后的行数为基准（两者行数一致：`strip_comments` 保留换行结构）。
    for (idx, code) in stripped.iter().enumerate() {
        let line = idx + 1;
        if code_line_has_ignore(code) {
            // 报错展示用**原始行**（strip_comments 会把字符串内容抹成空格，
            // 直接展示会让 `#[ignore = "原因"]` 的原因消失，读起来像空属性）。
            let shown = raw.get(line - 1).map_or_else(String::new, |r| r.trim().to_string());
            hits.push(Hit { file: rel.to_string(), line, code: shown });
        } else if check_fences && raw.get(line - 1).is_some_and(|r| code_line_has_ignore_fence(r)) {
            let r = raw[line - 1].trim().to_string();
            hits.push(Hit { file: rel.to_string(), line, code: r });
        }
    }
    hits
}

// ===========================================================================
//  二、工作区扫描
// ===========================================================================

/// 工作区根（本文件在 `crates/prism-core/tests/`，故上溯两级）。
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/prism-core/tests 的上两级应是工作区根")
        .to_path_buf()
}

/// 递归收集目录下的所有 `.rs` 文件（跳过 `target` / `node_modules`）。
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort(); // 确定性遍历（铁律：全链路确定性）
    for p in paths {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "target" || name == "node_modules" || name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// 扫描全仓，返回 `(命中列表, 扫描文件数, 本文件路径)`。
fn scan_workspace() -> (Vec<Hit>, usize, PathBuf) {
    let root = workspace_root();
    let self_path = std::fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")))
        .expect("规范化本文件路径")
        .join("tests/qa_ignore_guard.rs");
    let self_path = std::fs::canonicalize(&self_path).unwrap_or(self_path);

    let mut files = Vec::new();
    // 全仓 Rust 源码树。`tools/` 亦纳入（tools/docs/ci，但 Rust 源码
    // 里的 `#[ignore]` 同样是假绿，必须一视同仁）。
    //
    // 源码根目录按实际存在的目录探测：仓库布局是 `crates/`，单文件源码
    // 分发包布局是 `src/`（同一批文件，不同路径，见 tools/pack_source_txt.mjs）。
    let src_root = if root.join("crates").is_dir() { "crates" } else { "src" };
    for sub in [src_root, "examples", "tools"] {
        let d = root.join(sub);
        if d.is_dir() {
            collect_rs(&d, &mut files);
        }
    }

    let mut hits = Vec::new();
    let mut scanned = 0usize;
    for f in &files {
        let Ok(canon) = std::fs::canonicalize(f) else { continue };
        // 本文件必然含合成正例（反向自证素材），排除自己。
        if canon == self_path {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(f) else { continue };
        scanned += 1;
        let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
        let is_src = rel.contains("/src/") || rel.ends_with("/src/lib.rs");
        hits.extend(scan_source(&rel, &src, is_src));
    }
    (hits, scanned, self_path)
}

/// 把命中渲染成可直接贴进 issue 的报告。
fn render_hits(hits: &[Hit]) -> String {
    let mut s = String::new();
    for h in hits {
        s.push_str(&format!("\n  {}:{}  `{}`", h.file, h.line, h.code));
    }
    s
}

// ===========================================================================
//  三、门禁本体
// ===========================================================================

/// 🔴 主门禁：全仓 `#[ignore]` / ignore doc-test 围栏数必须为 **0**。
///
/// 失败代表有人用 `#[ignore]`（或 ```` ```ignore ````）把测试/示例从验收
/// 清单上抹掉了——**假绿**。合法处置只有三种：修被测函数、改造测试、
/// 或把生成器移出测试（移到 `examples/`）。**不许用 `#[ignore]` 换绿灯。**
#[test]
fn no_ignore_anywhere_in_workspace() {
    let (hits, scanned, _) = scan_workspace();
    assert!(
        hits.is_empty(),
        "🔴 发现 {} 处 `#[ignore]` / ignore doc-test 围栏（假绿手段，与注释掉测试同类）：{}\n\
         处置：修被测函数 / 改造测试 / 把生成器移到 examples/。\
         绝不许加 `#[ignore]` 了事。",
        hits.len(),
        render_hits(&hits)
    );
    assert!(scanned >= 100, "扫描文件数 {scanned} 过低，守卫可能扫空了（应为全仓量级）");
}

// ===========================================================================
//  四、反向自证：探测器本身的非恒空证明
// ===========================================================================

/// 合成正例：探测器**必须**抓到 `#[ignore]`。探测器失效 → 本测试红。
#[test]
fn detector_rejects_a_real_ignore() {
    // 三种合法形态都要抓到。
    for src in [
        "#[ignore]\nfn a() {}\n",
        "#[ignore = \"P0：待修\"]\nfn b() {}\n",
        "#[test]\n#[ignore]\nfn c() {}\n",
        "#[test] #[ignore] fn d() {}\n",
        "    #[ignore = \"x\"]\n    fn e() {}\n",
    ] {
        let hits = scan_source("synthetic.rs", src, false);
        assert!(!hits.is_empty(), "探测器漏抓：{src:?}");
    }
    // ignore doc-test 围栏：只在 src 语义下（check_fences=true）检查时要抓到。
    let fence = "/// ```ignore\n/// let x = 1;\n/// ```\n";
    assert!(
        !scan_source("synthetic.rs", fence, true).is_empty(),
        "探测器漏抓 ignore 围栏"
    );
    assert!(
        scan_source("synthetic.rs", fence, false).is_empty(),
        "非 src 文件不该按围栏判定（tests/ 下的说明不产生 doctest）"
    );
    // 围栏变体：```ignore,ignore_warnings 与带尾随空格都要抓到。
    for f in [
        "/// ```ignore,ignore_warnings\n/// let x = 1;\n/// ```\n",
        "///   ```ignore\n",
    ] {
        assert!(
            !scan_source("synthetic.rs", f, true).is_empty(),
            "探测器漏抓围栏变体：{f:?}"
        );
    }
}

/// 合成负例：注释里**提到** `#[ignore]` 字样不得误报，否则守卫恒红= 没守。
#[test]
fn detector_ignores_prose_mentions() {
    for src in [
        "//! 均已修复，无 `#[ignore]` 占位。\n",
        "/// **守卫断言**（原 `#[ignore]` 已知缺陷取证，缺陷已修）。\n",
        "// 历史：曾经用 #[ignore] 掩盖挂死。\n",
        "/* 块注释里的 #[ignore] 也不算数 */\nfn f() {}\n",
        "/* 嵌套 /* 块注释 */ 里的 #[ignore] */\nfn g() {}\n",
        "const NOTE: &str = \"literal #[ignore] text\";\n",
    ] {
        // 最后一条含字符串字面量，期望被剥成空白 → 不该命中。
        let hits = scan_source("synthetic.rs", src, true);
        assert!(hits.is_empty(), "误报（守卫会因文档提及而恒红）：{src:?} → {hits:?}");
    }
    // 对照：正文里真写一个必须被抓到，证明上一条的负例不是因为探测器失灵。
    let real = "// 说明文字\n#[ignore = \"真缺陷\"]\nfn h() {}\n";
    assert!(
        !scan_source("synthetic.rs", real, false).is_empty(),
        "探测器失灵：注释后的真 ignore 未抓到"
    );
}

/// 扫描覆盖度自证：文件数下界 + 本文件是唯一排除项（防"扫了 0 个所以全绿"）。
#[test]
fn scan_covers_the_workspace() {
    let (hits, scanned, self_path) = scan_workspace();
    assert!(scanned >= 100, "实际扫描 {scanned} 个 .rs 文件，覆盖度不足");
    // 本文件必须在工作区内（否则"排除自己"是空转，排除逻辑没被验证）。
    assert!(
        self_path.starts_with(workspace_root()),
        "本文件路径应位于工作区内：{}",
        self_path.display()
    );
    assert!(
        std::fs::canonicalize(&self_path).is_ok(),
        "本文件应存在：{}",
        self_path.display()
    );
    // 合成正例若放进扫描流程一定会红——这正是人工反向自证的等价形式。
    assert!(
        !scan_source("crates/synthetic_probe.rs", "#[ignore]\nfn z() {}\n", false).is_empty(),
        "合成正例必须被判红，否则守卫恒过"
    );
    let _ = hits;
}