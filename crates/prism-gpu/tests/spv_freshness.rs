//! **D-21 反查机制**：`.spv` 与 `.wgsl`、清单三者不得漂移。
//!
//! # 这个测试在防什么
//!
//! `.spv` 是**提交入库的产物**（不是构建产物）：主仓 `cargo build` 不生成它、
//! `build.rs` 不碰它、release 路径不经过 `tools/`。代价是"改了 WGSL 忘了重生成"
//! 这类漂移不会有任何编译错误——只会在真机上跑出一份**与源码不符**的字节码。
//! 本文件就是那条门禁：
//!
//! 1. 清单里每一行的 `wgsl_sha256` 必须等于**当前** `.wgsl` 的 sha256；
//! 2. 清单里每一行的 `spv_sha256` 必须等于**当前** `.spv` 的 sha256
//!    （防手改二进制 / 防把别的模块的 `.spv` 拷过来）；
//! 3. 清单登记的 `entry` / `naga 版本` 必须与 `tools/wgsl2spv` 的实际一致；
//! 4. 根 `Cargo.toml` 的 `[workspace] exclude` 仍含 `tools`
//!    （把工具踢出 workspace 是 D-21 的结构性保证，被删掉就失效）；
//! 5. `tools/wgsl2spv/Cargo.lock` 存在（自带锁文件并入库，工具可离线复现）。
//!
//! # 纪律
//!
//! * **禁止静默重生成**：本测试只读不写。看见红就重跑清单里的修复命令，
//!   让 diff 落在 `.spv` 上被人 review——而不是测试偷偷改二进制。
//! * **禁止 `#[ignore]`**：全仓禁 `#[ignore]`（`prism-core` 的 `qa_ignore_guard`
//!   硬门禁）。这条门禁一旦可跳过，等于没有。
//! * **只用 std**：本文件**不得**依赖 `naga` 或 `tools/` 里的任何东西。
//!   它要能在"没有任何工具链"的机器上判真伪，否则门禁本身就成了依赖。
//!
//! # 为什么这里有一份自己写的 SHA-256
//!
//! 不能引 `sha256` crate（第三方）、不能调 `wgsl2spv`（那是工具，不存在于主仓
//! 构建图），而"算 sha256"又不是三行能糊过去的事。故此处**独立**实现一份
//! FIPS 180-4（与 `tools/wgsl2spv/src/sha256.rs` 同算法、不同书写，互为交叉
//! 校验），并用公开测试向量 + 与系统 `sha256sum` 的实测一致性钉住。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// 最小 SHA-256（FIPS 180-4；仅 std）
// ---------------------------------------------------------------------------

/// 轮常量。
const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

/// 单块压缩。
fn compress(h: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

/// `sha256(bytes)` 的小写十六进制。
fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut chunks = data.chunks_exact(64);
    for chunk in &mut chunks {
        compress(&mut h, chunk);
    }
    let rest = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let tail_len = if rest.len() < 56 { 64 } else { 128 };
    let bits = (data.len() as u64).wrapping_mul(8);
    tail[tail_len - 8..tail_len].copy_from_slice(&bits.to_be_bytes());
    let mut off = 0;
    while off < tail_len {
        compress(&mut h, &tail[off..off + 64]);
        off += 64;
    }
    let mut s = String::with_capacity(64);
    for v in h {
        for b in v.to_be_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
    }
    s
}

// ---------------------------------------------------------------------------
// 路径 / 清单
// ---------------------------------------------------------------------------

/// 仓库根（`crates/prism-gpu` 的上两级）。
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/prism-gpu 的祖先必含仓库根")
        .to_path_buf()
}

/// 清单路径（相对仓库根）。
const MANIFEST: &str = "crates/prism-gpu/shaders/spv/spv_manifest.txt";

/// 漂移时的修复命令（**唯一**修复方式）。
const FIX_CMD: &str = "cargo run --manifest-path tools/wgsl2spv/Cargo.toml -- --all";

/// 一条清单记录。
struct Row {
    /// 清单声明的 WGSL 摘要。
    wgsl_sha: String,
    /// 清单声明的 SPIR-V 摘要。
    spv_sha: String,
    /// WGSL 相对路径（相对仓库根）。
    wgsl: String,
    /// 入口函数名。
    entry: String,
    /// 产出该 `.spv` 的 naga 版本。
    naga: String,
    /// 行号（1 起，诊断用）。
    line: usize,
}

/// 解析清单（跳过空行与 `#` 注释；字段间任意空白）。
fn parse_manifest(text: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = t.split_whitespace().collect();
        if f.len() != 5 {
            return Err(format!(
                "清单第 {} 行字段数 {} ≠ 5（应为 wgsl_sha256 / spv_sha256 / wgsl路径 / entry / naga版本）：{t}",
                i + 1,
                f.len()
            ));
        }
        rows.push(Row {
            wgsl_sha: f[0].to_string(),
            spv_sha: f[1].to_string(),
            wgsl: f[2].to_string(),
            entry: f[3].to_string(),
            naga: f[4].to_string(),
            line: i + 1,
        });
    }
    Ok(rows)
}

/// 从 `tools/wgsl2spv/Cargo.lock` 取 naga 的解析版本（`None` = 锁文件里没有）。
fn naga_version_from_lock(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("tools/wgsl2spv/Cargo.lock")).ok()?;
    let mut current_is_naga = false;
    for line in text.lines() {
        let t = line.trim();
        if t == "[[package]]" {
            current_is_naga = false;
        } else if t == "name = \"naga\"" {
            current_is_naga = true;
        } else if current_is_naga {
            if let Some(v) = t.strip_prefix("version = \"") {
                return Some(v.trim_end_matches('"').to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

/// 自带 SHA-256 的正确性：公开测试向量（FIPS 180-4 / 常见公开值）。
///
/// 门禁的可信度取决于这里——算法错了会**放过**所有漂移。
#[test]
fn std_sha256_matches_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // 55 / 56 / 64 字节：填充跨块边界（两侧分别走 64 / 128 字节尾块）。
    assert_eq!(
        sha256_hex(&[b'x'; 55]),
        "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072"
    );
    assert_eq!(
        sha256_hex(&[b'x'; 64]),
        "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c"
    );
}

/// **主门禁**：`.wgsl` / `.spv` / 清单三方一致。
///
/// 断言串里写明修复命令——看见红的人不需要再翻文档。
#[test]
fn spv_artifacts_match_manifest_and_sources() {
    let root = repo_root();
    let manifest_path = root.join(MANIFEST);
    let text = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "读不到清单 {}：{e}\n修复：{FIX_CMD}",
            manifest_path.display()
        )
    });
    let rows = parse_manifest(&text).expect("清单格式必须合法");
    assert!(
        !rows.is_empty(),
        "清单为空——`{MANIFEST}` 至少要登记一条 WGSL → SPIR-V 目标。修复：{FIX_CMD}"
    );

    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for row in &rows {
        assert!(
            seen.insert(row.wgsl.clone(), ()).is_none(),
            "清单第 {} 行重复登记 {}",
            row.line,
            row.wgsl
        );

        let wgsl_path = root.join(&row.wgsl);
        let wgsl_bytes = std::fs::read(&wgsl_path)
            .unwrap_or_else(|e| panic!("读不到 {}：{e}", wgsl_path.display()));
        // ---- 判据 1：源没变（变了就必须重生成 .spv）----
        let wgsl_sha = sha256_hex(&wgsl_bytes);
        assert_eq!(
            wgsl_sha, row.wgsl_sha,
            "\n\n**{} 已改动，但入库的 .spv 没有重新生成。**\n\
             清单第 {} 行记录 {:?}，当前源是 {:?}。\n\
             改了 WGSL 必须重跑：\n  {FIX_CMD}\n\
             （本测试**不会**替你重生成：`.spv` 是入库产物，diff 必须被人 review。）\n",
            row.wgsl, row.line, row.wgsl_sha, wgsl_sha
        );

        // ---- 判据 2：二进制没被手改（或拷错文件）----
        // 产物路径的约定：`<wgsl 所在目录>/spv/<wgsl 主名>.spv`（清单只登记源路径，
        // 产物路径由这条约定推出——多一列就多一处可能不一致的地方）。
        let stem = wgsl_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let spv_path = wgsl_path.with_file_name("spv").join(format!("{stem}.spv"));
        let spv_bytes = std::fs::read(&spv_path).unwrap_or_else(|e| {
            panic!(
                "读不到 {}：{e}\n清单登记了它却不存在。修复：{FIX_CMD}",
                spv_path.display()
            )
        });
        let spv_sha = sha256_hex(&spv_bytes);
        assert_eq!(
            spv_sha, row.spv_sha,
            "\n\n**{} 与清单不符（二进制被手改，或换成了别的模块）。**\n\
             清单第 {} 行记录 {:?}，当前文件是 {:?}。\n\
             `.spv` **禁止手改**，只能重生成：\n  {FIX_CMD}\n",
            spv_path.display(),
            row.line,
            row.spv_sha,
            spv_sha
        );

        // ---- 判据 3：SPIR-V 二进制的基本结构（魔数 / 版本 / 长度自洽）----
        assert!(spv_bytes.len() >= 20, "{} 太短，不是 SPIR-V", spv_path.display());
        assert_eq!(
            spv_bytes.len() % 4,
            0,
            "{} 字节数 {} 不是 4 的倍数（SPIR-V 是 u32 字流）",
            spv_path.display(),
            spv_bytes.len()
        );
        let magic = u32::from_le_bytes([spv_bytes[0], spv_bytes[1], spv_bytes[2], spv_bytes[3]]);
        assert_eq!(magic, 0x0723_0203, "{} 魔数不是 SPIR-V", spv_path.display());
        let version = u32::from_le_bytes([spv_bytes[4], spv_bytes[5], spv_bytes[6], spv_bytes[7]]);
        // 版本字：bit16..23 = major，bit8..15 = minor，bit0..7 恒 0。
        let (major, minor) = ((version >> 16) & 0xFF, (version >> 8) & 0xFF);
        assert_eq!(
            (major, minor),
            (1, 0),
            "{} 不是 SPIR-V 1.0（实测 {major}.{minor}）",
            spv_path.display()
        );

        // ---- 判据 4：入口名在字流里（UTF-8 明文出现，SPIR-V 的 OpEntryPoint 约定）----
        let needle = row.entry.as_bytes();
        assert!(
            spv_bytes.windows(needle.len()).any(|w| w == needle),
            "{} 里找不到入口名 {:?}（entry 登记错了？）",
            spv_path.display(),
            row.entry
        );

        // ---- 判据 5：登记的工具链版本 == 锁文件里的实际版本 ----
        let lock_ver = naga_version_from_lock(&root);
        if let Some(v) = lock_ver {
            assert_eq!(
                row.naga, v,
                "清单第 {} 行登记的 naga 版本 {:?} ≠ tools/wgsl2spv/Cargo.lock 的 {:?}；\
                 换版本后必须重生成并更新清单：{FIX_CMD}",
                row.line, row.naga, v
            );
        }
    }
}

/// **结构性反查**：`tools/` 必须仍在 workspace 之外，且工具自带锁文件。
///
/// 这条防的是"工具悄悄长成依赖"：只要有人删掉 `exclude = ["tools"]`，
/// `cargo build` 就会把 naga 拉进主仓构建图——而那样 clone 下来 `cargo build`
/// 就要联网了，D-21 的裁定前提当场失效。
#[test]
fn workspace_still_excludes_tools_and_tool_has_lockfile() {
    let root = repo_root();
    let manifest = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("读不到根 Cargo.toml：{e}"));

    // 只在 `[workspace]` 段内找 `exclude`（避免被其它段落里的同名字样骗过）。
    let mut in_workspace = false;
    let mut exclude_line = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_workspace = t == "[workspace]";
            continue;
        }
        if in_workspace && t.starts_with("exclude") {
            exclude_line = t.to_string();
            break;
        }
    }
    assert!(
        !exclude_line.is_empty(),
        "根 Cargo.toml 的 [workspace] 段里找不到 `exclude`——D-21 的结构性保证被删了。\n\
         必须保留：\n  exclude = [\"tools\"]\n\
         理由：tools/wgsl2spv 依赖 naga，一旦进入 workspace，主仓 `cargo build` \
         就要联网拉 crates.io，与「零第三方依赖的交付物」冲突。"
    );
    assert!(
        exclude_line.contains("tools"),
        "`[workspace] exclude` 不再包含 `tools`（当前：{exclude_line}）——\
         工具会被拉进主仓构建图"
    );

    let lock = root.join("tools/wgsl2spv/Cargo.lock");
    assert!(
        lock.is_file(),
        "{} 不存在——工具的锁文件必须入库（离线可复现，且清单里的 naga 版本要对得上它）",
        lock.display()
    );
    let lock_text = std::fs::read_to_string(&lock).expect("锁文件必须可读");
    assert!(
        lock_text.contains("name = \"naga\""),
        "锁文件里没有 naga——D-21 的工具链已不完整"
    );

    // 工具必须自带 [workspace]（独立根），否则它会被主仓或上层 workspace 吸进去。
    let tool_manifest = root.join("tools/wgsl2spv/Cargo.toml");
    let tool_text = std::fs::read_to_string(&tool_manifest)
        .unwrap_or_else(|e| panic!("读不到 {}：{e}", tool_manifest.display()));
    assert!(
        tool_text.lines().any(|l| l.trim() == "[workspace]"),
        "tools/wgsl2spv/Cargo.toml 必须自带空 `[workspace]` 段（把自己钉成独立根）"
    );
}

/// 用系统 `spirv-val` 复核产物（**不属于 Rust 依赖**：缺了就跳过，不算失败）。
///
/// 与 `wgsl2spv --all` 里的同一道校验重复，但这一份在主仓 `cargo test` 里跑，
/// 是"产物真能过 Khronos 官方校验器"的常驻证据（工具可能很久才被跑一次）。
#[test]
fn spv_passes_khronos_spirv_val_when_available() {
    let root = repo_root();
    let av = root.join("crates/prism-gpu/shaders/spv/scene_frame_v3.spv");
    if !av.is_file() {
        // 清单门禁已经在上面报红；这里不重复 fail（避免同因两报）。
        return;
    }
    let out = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.0")
        .arg(&av)
        .output();
    match out {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!(
                "[skip] 本机未安装 spirv-val：跳过官方校验器复核（它不是 Rust 依赖，\
                 不装不影响主仓构建）"
            );
        }
        Err(e) => panic!("调用 spirv-val 失败：{e}"),
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            assert!(
                o.status.success(),
                "spirv-val 拒绝 {}（exit {:?}）\nstdout: {stdout}\nstderr: {stderr}",
                av.display(),
                o.status.code()
            );
            println!(
                "[val] spirv-val --target-env vulkan1.0 {}: ok",
                av.display()
            );
        }
    }
}

/// **D-21 的第三条反查**：主仓 workspace 仍然**没有第三方依赖**。
///
/// D-21 裁定"开发期借 naga"之所以不破坏"零依赖"，全靠一条结构事实：
/// **交付物（`crates/` 里的 29 个 crate）的每一个依赖都是 `path` 依赖**。
/// 这条一旦破了——比如有人图省事在 `prism-gpu` 里写 `naga = "30"`——那就不是
/// "借工具当尺子"，而是**让交付物依赖 crates.io**：别人 clone 下来 `cargo
/// build` 就要联网。所以这里逐个 crate 扫 `Cargo.toml`，任何一条不带
/// `path =` 的依赖声明都判红。
///
/// 只做**文本级**判断（不解析 TOML）：本仓的依赖声明形式统一为
/// `name = { path = "..." }`，文本判据足够，且不引第三方解析器——否则这条
/// 反查自己就先破了"零依赖"。
#[test]
fn workspace_has_no_third_party_dependencies() {
    let root = repo_root();
    let crates_dir = root.join("crates");
    let mut manifests: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(&crates_dir)
        .unwrap_or_else(|e| panic!("读不到 {}：{e}", crates_dir.display()));
    for entry in entries {
        let entry = entry.expect("目录项必须可读");
        let path = entry.path().join("Cargo.toml");
        if path.is_file() {
            manifests.push(path);
        }
    }
    manifests.sort();
    assert!(
        manifests.len() >= 25,
        "只找到 {} 个 crate 的 Cargo.toml（预期 ≥ 25）——扫描范围不对，这条反查会假绿",
        manifests.len()
    );

    /// 依赖声明所在的段（`workspace.dependencies` 也算）。
    fn is_dep_section(name: &str) -> bool {
        name == "dependencies"
            || name == "dev-dependencies"
            || name == "build-dependencies"
            || name.ends_with(".dependencies")
            || name.ends_with(".dev-dependencies")
            || name.ends_with(".build-dependencies")
    }

    let mut offenders: Vec<String> = Vec::new();
    let mut declared = 0usize;
    for path in &manifests {
        let text = std::fs::read_to_string(path).expect("Cargo.toml 必须可读");
        let mut section = String::new();
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if t.starts_with('[') && t.ends_with(']') {
                section = t.trim_matches(['[', ']'].as_slice()).to_string();
                // `[dependencies.foo]` 子表形式：子表里必须出现 path。
                if let Some(rest) = section.strip_prefix("dependencies.")
                    .or_else(|| section.strip_prefix("dev-dependencies."))
                    .or_else(|| section.strip_prefix("build-dependencies."))
                {
                    offenders.push(format!(
                        "{}:{} 用了 `[dependencies.{rest}]` 子表形式，无法就地判断是否 path 依赖；\
                         本仓统一写法是 `{rest} = {{ path = \"...\" }}`",
                        path.display(),
                        i + 1
                    ));
                }
                continue;
            }
            if t.is_empty() || t.starts_with('#') || !is_dep_section(&section) {
                continue;
            }
            // 段内的顶层键值行（缩进行是内联表续行/数组元素，不算新依赖）。
            if line.starts_with(char::is_whitespace) || !t.contains('=') {
                continue;
            }
            declared += 1;
            if !t.contains("path") {
                offenders.push(format!(
                    "{}:{} 的依赖 `{t}` 不带 `path` —— 这是 crates.io 依赖，\
                     违反「交付物零第三方依赖」（D-21 只允许 `tools/` 借 naga）",
                    path.display(),
                    i + 1
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "主仓 workspace 出现第三方依赖：\n  - {}\n\n\
         修复：把依赖挪进 `tools/` 下独立的 crate（`[workspace] exclude = [\"tools\"]`），\
         产物（如 .spv）提交入库。详见 D-21 裁定。",
        offenders.join("\n  - ")
    );
    assert!(
        declared >= 100,
        "只扫到 {declared} 条依赖声明（预期 ≥ 100）——判据可能失效（假绿）"
    );
    println!(
        "[deps] {} 个 crate、{declared} 条依赖声明，全部为 path 依赖 ✓",
        manifests.len()
    );
}
