//! `wgsl2spv` —— D-21 裁定的**开发期工具**：把 WGSL 翻成 SPIR-V。
//!
//! # 这为什么不算破坏「零第三方依赖」
//!
//! 「零依赖」约束的是**交付物自包含**（clone 下来 `cargo build` 不联网、不拉
//! crates.io、源码里没有第三方代码）。本 crate 在 `tools/` 下、被主仓
//! `[workspace] exclude` 踢出成员表、自带 `Cargo.lock`，主仓
//! `cargo build/test/tree` 完全看不见 naga。它的产物（`.spv`）提交入库，
//! `build.rs` 不生成、不调用，release 路径不经过本目录。
//!
//! # 用法
//!
//! ```text
//! wgsl2spv --in <x.wgsl> --entry <name> --out <x.spv>   # 单文件
//! wgsl2spv --all                                        # 批量重生成 + 重写清单
//! wgsl2spv --check                                      # 只校验（漂移 → exit 1）
//! ```
//!
//! `--check` **绝不写盘**：它把源重新编译到内存，与入库的 `.spv` 逐字节比对，
//! 再按清单重算两侧 sha256。任何一处不一致都以退出码 1 结束并打印修复命令。
//!
//! # 诊断纪律
//!
//! 解析/校验失败时把 naga 的 `ParseError` / `WithSpan<ValidationError>` 原样
//! 打印（含 `emit_to_string` 的源码片段与列号），**不吞、不改写、不降级为
//! "编译失败"四个字**——本工具存在的意义就是让 WGSL 的错误可定位。

mod sha256;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// 本工具所依赖的 naga 版本。
///
/// 与 `Cargo.lock` 逐字对账（[`check_naga_lock`]）：改依赖不更新这里 ⇒ 启动即
/// 失败，避免清单里写着一个版本、实际用的是另一个。
const NAGA_VERSION: &str = "30.0.1";

/// 一个"源 → 产物"条目（相对**仓库根**的路径，跨机器可复现）。
struct Target {
    /// WGSL 源（相对仓库根）。
    wgsl: &'static str,
    /// 入口函数名（`@compute fn` 名，= SPIR-V `OpEntryPoint`）。
    entry: &'static str,
    /// SPIR-V 产物（相对仓库根；入库）。
    spv: &'static str,
}

/// 批量清单。新增 shader 只需在这里加一行（**代码内清单**，不引入 TOML 解析依赖）。
const TARGETS: &[Target] = &[
    Target {
        wgsl: "crates/prism-gpu/shaders/scene_frame_v3.wgsl",
        entry: "scene_main",
        spv: "crates/prism-gpu/shaders/spv/scene_frame_v3.spv",
    },
    // A2 / W2：环境采样语义探针。源文件用 `//!include rt_env.wgsl` 复用
    // **唯一一份** env 实现（不复制粘贴），故本工具需按 `//!include` 展开
    // （见 `expand_includes`）。
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_env_probe.wgsl",
        entry: "env_probe",
        spv: "crates/prism-gpu/shaders/spv/rt_env_probe.spv",
    },
    // ---- T6：整条 wavefront 的 5 个 compute 入口（同一份 WGSL，5 个产物） ----
    //
    // 一个 `.wgsl` 对应多个 `OpEntryPoint` 时**必须一个入口一个 `.spv`**：
    // 驱动侧的 `VkPipelineShaderStageCreateInfo.p_name` 只能选中模块里的一个
    // 入口，而 pipeline layout 的静态资源用量以模块为准；把 5 个入口塞进同一
    // 字流会让每个 pipeline 都背着其余四个入口的死代码（且 naga 的
    // `entry_points.retain` 本来就是单入口语义）。
    //
    // 清单因此从 5 列扩成 6 列（新增 `<spv 相对路径>`）——这是**格式扩展**，
    // 不是重排：前 5 列语义与顺序不变，第 6 列追加在末尾。
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_wavefront.wgsl",
        entry: "generate",
        spv: "crates/prism-gpu/shaders/spv/rt_wavefront_generate.spv",
    },
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_wavefront.wgsl",
        entry: "shade",
        spv: "crates/prism-gpu/shaders/spv/rt_wavefront_shade.spv",
    },
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_wavefront.wgsl",
        entry: "connect",
        spv: "crates/prism-gpu/shaders/spv/rt_wavefront_connect.spv",
    },
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_wavefront.wgsl",
        entry: "extend",
        spv: "crates/prism-gpu/shaders/spv/rt_wavefront_extend.spv",
    },
    Target {
        wgsl: "crates/prism-gpu/shaders/rt_wavefront.wgsl",
        entry: "accumulate",
        spv: "crates/prism-gpu/shaders/spv/rt_wavefront_accumulate.spv",
    },
];

/// 清单文件（相对仓库根）。
const MANIFEST: &str = "crates/prism-gpu/shaders/spv/spv_manifest.txt";

/// 漂移时打印的修复命令（**唯一**修复方式；禁止手改 `.spv`）。
const FIX_CMD: &str = "cargo run --manifest-path tools/wgsl2spv/Cargo.toml -- --all";

/// 仓库根：本 crate 恒在 `<root>/tools/wgsl2spv`，编译期即确定，与 cwd 无关。
fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(2)
        .expect("tools/wgsl2spv 的祖先必含仓库根")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// 编译核心
// ---------------------------------------------------------------------------

/// 递归展开 `//!include <name>.wgsl`（相对 `crates/prism-gpu/shaders/`）。
///
/// 与 `crates/prism-gpu/src/shaders.rs` 的展开器**同一指令**（`//!include`），
/// 保证"浏览器宿主看到的源"与"naga 编译的源"是同一份组合。深度上限 8，
/// 缺文件即报错（不静默跳过——跳过会让 SPIR-V 少一段实现却仍然编译通过）。
///
/// # 为什么必须「同一文件只展开一次」
///
/// `rt_wavefront.wgsl` 的 include 图是菱形（`rt_shade` / `rt_scene` / `rt_env`
/// 各自都 include `rt_common`）。WGSL 没有头文件保护，重复展开会直接
/// `redefinition of PI`。这里的去重语义与 `shaders.rs::expand_into` 一致：
/// **按文件名（相对 shader 目录）首次出现为准**，后续 include 静默跳过。
/// 这不是"容忍错误包含"——真正的循环包含会先撞深度上限（8 层）报错。
fn expand_includes(src: &str, dir: &Path, depth: u32) -> Result<String, String> {
    let mut seen: Vec<String> = Vec::new();
    expand_includes_inner(src, dir, depth, &mut seen)
}

/// [`expand_includes`] 的实现体（`seen` 跨整棵 include 树共享）。
fn expand_includes_inner(
    src: &str,
    dir: &Path,
    depth: u32,
    seen: &mut Vec<String>,
) -> Result<String, String> {
    if depth == 0 {
        return Err("//!include 嵌套超过 8 层（疑似循环包含）".to_string());
    }
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("//!include ") {
            let name = rest.trim();
            // 已展开过：跳过（菱形 include 图的唯一正确语义）。
            if seen.iter().any(|s| s == name) {
                continue;
            }
            seen.push(name.to_string());
            let path = dir.join(name);
            let text = fs::read_to_string(&path)
                .map_err(|e| format!("//!include {} 读取失败：{e}", path.display()))?;
            out.push_str(&format!("// ---- include {name} ----\n"));
            out.push_str(&expand_includes_inner(&text, dir, depth - 1, seen)?);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out)
}

/// WGSL 源 → SPIR-V 字流（parse → validate → SPIR-V backend）。
///
/// 失败时返回**原样**的 naga 诊断（含源码片段/列号），调用方直接打印。
fn compile(wgsl_src: &str, entry: &str) -> Result<Vec<u32>, String> {
    // ---- 1. 解析（语法） ----
    let module = match naga::front::wgsl::parse_str(wgsl_src) {
        Ok(m) => m,
        Err(e) => {
            return Err(format!(
                "WGSL 解析失败（naga {} ParseError）：\n{}",
                NAGA_VERSION,
                e.emit_to_string(wgsl_src)
            ))
        }
    };

    // ---- 2. 入口必须存在（否则 SPIR-V 里没有 OpEntryPoint，驱动建管线才报错）----
    let names: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    if !names.iter().any(|n| *n == entry) {
        return Err(format!(
            "入口 `{entry}` 不存在；本文件的入口有：{names:?}"
        ));
    }

    // 只发请求的入口（一个 .spv 对应一个 kernel；多入口会让驱动侧的选择依赖字流顺序）。
    let mut module = module;
    module.entry_points.retain(|e| e.name == entry);

    // ---- 3. 语义校验 ----
    // `Capabilities::empty()`：只放行 Vulkan 1.0 / WebGPU 核心能力（与交付目标一致）。
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    let info = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            return Err(format!(
                "WGSL 语义校验失败（naga {} ValidationError）：\n{}",
                NAGA_VERSION,
                e.emit_to_string(wgsl_src)
            ))
        }
    };

    // ---- 4. SPIR-V 后端 ----
    let mut options = naga::back::spv::Options::default();
    options.lang_version = (1, 0); // Vulkan 1.0：本仓 loader 声明的目标环境
                                    // 空 flags：默认值里的 ADJUST_COORDINATE_SPACE / CLAMP_FRAG_DEPTH 只与图形
                                    // 阶段有关，DEBUG（debug_assertions 下默认开）会让字节流随构建档位漂移——
                                    // 而 `.spv` 要入库、要跨机器可比，故显式清空。
    options.flags = naga::back::spv::WriterFlags::empty();
    // 缺 @group/@binding 必须报错：`fake_missing_bindings` 默认 true 会**静默**
    // 编造绑定点，让"忘了写 @binding"变成"运行时绑定错位"。
    options.fake_missing_bindings = false;
    // WGSL 的 `@group(g) @binding(b)` → Vulkan 的 `(set=g, binding=b)`，逐条显式
    // 映射（关掉 fake 之后没有兜底，漏一条就是 `MissingBinding` 错误）。
    let mut binding_map = naga::back::spv::BindingMap::default();
    for (_, gv) in module.global_variables.iter() {
        if let Some(rb) = gv.binding {
            binding_map.insert(
                rb,
                naga::back::spv::BindingInfo {
                    descriptor_set: rb.group,
                    binding: rb.binding,
                    binding_array_size: None,
                },
            );
        }
    }
    options.binding_map = binding_map;
    // 越界策略取 `Restrict`（钳位），与 WebGPU 宿主（第三后端）的实际行为一致；
    // 不取 `Unchecked`：让"shader 越界"表现为钳位而不是读驱动垃圾。
    options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Restrict,
        buffer: naga::proc::BoundsCheckPolicy::Restrict,
        image_load: naga::proc::BoundsCheckPolicy::Restrict,
        binding_array: naga::proc::BoundsCheckPolicy::Restrict,
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: entry.to_string(),
    };
    let words = match naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)) {
        Ok(w) => w,
        Err(e) => return Err(format!("SPIR-V 生成失败：{e}")),
    };
    if words.is_empty() {
        return Err("SPIR-V 生成失败：字流为空".to_string());
    }
    // 精度钉死：全部浮点算术加 NoContraction（R-1 逐位判据的前提，见函数文档）。
    let words = pin_no_contraction(&words);
    Ok(words)
}

/// SPIR-V 字流 → 小端字节流（SPIR-V 二进制的标准编码）。
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// 精度钉死（NoContraction post-pass；T-B R-1 逐位判据的前提）
// ---------------------------------------------------------------------------

/// `OpDecorate <id> NoContraction` 的装饰码（SPIR-V 1.0 §3.32）。
const DEC_NO_CONTRACTION: u32 = 42;

/// 会被 FMA 收缩的浮点算术 opcode（SPIR-V 1.0 核心指令码；**逐个对过表**：
/// 128 IAdd / 130 ISub / 132 IMul / 134 UDiv / 135 SDiv / 137-139 整数取模
/// 都**不在**表内）。
const FP_CONTRACTABLE_OPS: &[u32] = &[
    127, // OpFNegate
    129, // OpFAdd
    131, // OpFSub
    133, // OpFMul
    136, // OpFDiv
    140, // OpFRem
    141, // OpFMod
    142, // OpVectorTimesScalar
    143, // OpMatrixTimesScalar
    144, // OpVectorTimesMatrix
    145, // OpMatrixTimesVector
    146, // OpMatrixTimesMatrix
    147, // OpOuterProduct
    148, // OpDot
];

/// 给全部浮点算术结果的 SSA id 加 `NoContraction` 装饰（原位返回新字流）。
///
/// # 为什么必须做（T-B R-1 的根因之一，真机实测抓到）
///
/// Vulkan 的 SPIR-V **默认允许 FMA 收缩**：`a*b + c` 可以被驱动编译成
/// `fma(a,b,c)`，末位与分开乘加不同。WGSL/naga 不生成 `NoContraction`
/// 装饰 ⇒ llvmpipe 的 NIR `fuse_fma` 合法地融合了着色链里的乘加，对拍
/// 出现"单像素 1 ULP"的分歧（CPU 侧 Rust 不收缩）。R-1 的判据是
/// `to_bits()` 全等，收缩必须禁掉——本 pass 给**所有**浮点算术结果加
/// `NoContraction`（保守超集：装饰全部结果 ⇒ 任何乘加都不可能被融合），
/// 把语义钉死为逐指令 IEEE。
///
/// # 为什么是"插 OpDecorate"而不是改 naga
///
/// 装饰不分配 id、不改任何结果编号：把 `OpDecorate` 字插进 annotations 段
/// （位于 debug 段之后、types/constants 段之前，SPIR-V 逻辑布局 §2.4）即可，
/// 其余字流一字节不动。插完后仍过 `spirv-val` 官方校验。
fn pin_no_contraction(words: &[u32]) -> Vec<u32> {
    // 1. 收集浮点算术结果 id（带结果类型的指令：result id 在 word[i+2]）。
    let mut ids: Vec<u32> = Vec::new();
    let mut i = 5usize; // 跳过 magic/version/generator/bound/schema 头
    while i < words.len() {
        let op = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break; // 字流损坏防御（正常不会发生）
        }
        match op {
            12 => {
                // OpExtInst：<id> result type, <id> result, set, inst...
                // GLSL.std450 的浮点件（sqrt/tan/fma/...）结果也纳入保守超集。
                if wc >= 3 {
                    ids.push(words[i + 2]);
                }
            }
            op if FP_CONTRACTABLE_OPS.contains(&op) => {
                if wc >= 3 {
                    ids.push(words[i + 2]);
                }
            }
            _ => {}
        }
        i += wc;
    }
    if ids.is_empty() {
        return words.to_vec();
    }
    // 2. 找插入点：第一条 types/constants 段指令（opcode ≥ 19，排除装饰段
    //    71/72——装饰必须整体位于 types 之前）。**按指令粒度**走（字粒度会把
    //    操作数里的立即数误当 opcode，插进指令中间 = 模块损坏）。naga 输出
    //    的段序保证存在。
    let mut insert_at = words.len();
    let mut i = 5usize;
    while i < words.len() {
        let op = words[i] & 0xFFFF;
        let wc = (words[i] >> 16) as usize;
        if wc == 0 {
            break;
        }
        if op >= 19 && op != 71 && op != 72 {
            insert_at = i;
            break;
        }
        i += wc;
    }
    // 3. 生成 OpDecorate 字（wordcount 3：opcode, target, decoration）。
    let mut out = Vec::with_capacity(words.len() + ids.len() * 3);
    out.extend_from_slice(&words[..insert_at]);
    for &id in &ids {
        out.push((3u32 << 16) | 71);
        out.push(id);
        out.push(DEC_NO_CONTRACTION);
    }
    out.extend_from_slice(&words[insert_at..]);
    out
}

/// 用 `spirv-val` 校验一份产物（**不属于 Rust 依赖**：调系统命令，缺了优雅跳过）。
///
/// 返回 `Ok(Some(说明))` = 校验通过；`Ok(None)` = 本机没有 spirv-val（跳过）；
/// `Err` = 校验失败（带 spirv-val 的原始 stderr）。
fn spirv_val(bytes: &[u8], name: &str) -> Result<Option<String>, String> {
    // 临时文件名带 pid + 自增序号：`--check` 与 `--all` 可能并发，撞名会让
    // 两边互相覆盖（症状是"校验了一个别人的文件"）。
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "wgsl2spv-val-{}-{seq}.spv",
        std::process::id()
    ));
    fs::write(&path, bytes).map_err(|e| format!("写临时文件 {} 失败：{e}", path.display()))?;
    let out = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.0")
        .arg(&path)
        .output();
    let result = match out {
        Err(e) => {
            let _ = fs::remove_file(&path);
            // 只有"命令不存在"才算优雅跳过；其它 IO 错误照实报。
            if e.kind() == std::io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(format!("调用 spirv-val 失败：{e}"));
        }
        Ok(o) => o,
    };
    let _ = fs::remove_file(&path);
    let stderr = String::from_utf8_lossy(&result.stderr);
    if result.status.success() {
        Ok(Some(format!("spirv-val {name}: ok")))
    } else {
        Err(format!(
            "spirv-val 拒绝 {name}（exit {:?}）：\n{}",
            result.status.code(),
            stderr.trim()
        ))
    }
}

/// 读 `Cargo.lock` 确认实际解析到的 naga 版本 == [`NAGA_VERSION`]。
///
/// 不做这件事的话，清单里的"naga 版本"就只是注释——而清单的全部价值在于
/// "看到某个版本产出的字节"。
fn check_naga_lock() -> Result<(), String> {
    let lock = repo_root().join("tools/wgsl2spv/Cargo.lock");
    let text = fs::read_to_string(&lock)
        .map_err(|e| format!("读不到 {}：{e}（先跑一次 cargo fetch）", lock.display()))?;
    let mut in_naga = false;
    for line in text.lines() {
        let t = line.trim();
        if t == "[[package]]" {
            in_naga = false;
        } else if t == "name = \"naga\"" {
            in_naga = true;
        } else if in_naga {
            if let Some(v) = t.strip_prefix("version = \"") {
                let v = v.trim_end_matches('"');
                if v != NAGA_VERSION {
                    return Err(format!(
                        "NAGA_VERSION 常量（{NAGA_VERSION}）与 Cargo.lock 解析到的 naga（{v}）不一致；\
                         改依赖后必须同步本常量并重跑 --all"
                    ));
                }
                return Ok(());
            }
        }
    }
    Err("Cargo.lock 里找不到 naga 包条目".to_string())
}

// ---------------------------------------------------------------------------
// 清单
// ---------------------------------------------------------------------------

/// 一条清单记录。
struct ManifestRow {
    wgsl_sha: String,
    spv_sha: String,
    wgsl: String,
    entry: String,
    naga: String,
    /// SPIR-V 产物相对路径（**第 6 列**；同一 WGSL 多入口时靠它区分）。
    spv: String,
}

impl ManifestRow {
    /// 渲染成一行（分隔符固定两个空格；字段内不含空格）。
    fn line(&self) -> String {
        format!(
            "{}  {}  {}  {}  {}  {}",
            self.wgsl_sha, self.spv_sha, self.wgsl, self.entry, self.naga, self.spv
        )
    }
}

/// 按 `TARGETS` 顺序重算清单（读盘上的 `.wgsl` 与 `.spv`）。
fn build_manifest(root: &Path) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("# spv_manifest.txt —— WGSL → SPIR-V 产物清单（D-21）\n");
    out.push_str("#\n");
    out.push_str("# 由 `tools/wgsl2spv` 生成，**不要手改**；`.spv` 是二进制，手改一定对不上。\n");
    out.push_str(&format!("# 修复/重生成命令：{FIX_CMD}\n"));
    out.push_str("# 校验：cargo test -p prism-gpu --test spv_freshness\n");
    out.push_str("#\n");
    out.push_str("# 格式（每行，字段间两个空格）：\n");
    out.push_str("#   <wgsl_sha256>  <spv_sha256>  <wgsl相对路径>  <entry>  <naga版本>  <spv相对路径>\n");
    for t in TARGETS {
        let wgsl_bytes = fs::read(root.join(t.wgsl))
            .map_err(|e| format!("读不到 {}：{e}", root.join(t.wgsl).display()))?;
        let spv_bytes = fs::read(root.join(t.spv))
            .map_err(|e| format!("读不到 {}：{e}（先跑 --all）", root.join(t.spv).display()))?;
        out.push_str(
            &ManifestRow {
                wgsl_sha: sha256::sha256_hex(&wgsl_bytes),
                spv_sha: sha256::sha256_hex(&spv_bytes),
                wgsl: t.wgsl.to_string(),
                entry: t.entry.to_string(),
                naga: NAGA_VERSION.to_string(),
                spv: t.spv.to_string(),
            }
            .line(),
        );
        out.push('\n');
    }
    Ok(out)
}

/// 解析清单为记录列表（忽略空行与 `#` 注释）。
fn parse_manifest(text: &str) -> Result<Vec<ManifestRow>, String> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = t.split_whitespace().collect();
        if f.len() != 6 {
            return Err(format!(
                "清单第 {} 行字段数 {} ≠ 6：{t}",
                i + 1,
                f.len()
            ));
        }
        rows.push(ManifestRow {
            wgsl_sha: f[0].to_string(),
            spv_sha: f[1].to_string(),
            wgsl: f[2].to_string(),
            entry: f[3].to_string(),
            naga: f[4].to_string(),
            spv: f[5].to_string(),
        });
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// 子命令
// ---------------------------------------------------------------------------

/// `--in/--entry/--out`：单文件。
fn run_single(input: &Path, entry: &str, output: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(input)
        .map_err(|e| format!("读不到 {}：{e}", input.display()))?;
    let src = expand_includes(&raw, input.parent().unwrap_or(Path::new(".")), 8)?;
    let words = compile(&src, entry)?;
    let bytes = words_to_bytes(&words);
    let val = spirv_val(&bytes, &input.display().to_string())?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("建目录 {} 失败：{e}", parent.display()))?;
        }
    }
    fs::write(output, &bytes).map_err(|e| format!("写 {} 失败：{e}", output.display()))?;
    println!(
        "[ok] {} → {}（entry={entry}，{} B = {} word，sha256 {}）",
        input.display(),
        output.display(),
        bytes.len(),
        words.len(),
        &sha256::sha256_hex(&bytes)[..16]
    );
    match val {
        Some(s) => println!("[val] {s}"),
        None => println!("[val] 跳过：本机未安装 spirv-val（不属于 Rust 依赖）"),
    }
    println!("[tip] 清单未更新；改的是入库目标请跑 --all");
    Ok(())
}

/// `--all`：批量重生成 `.spv` + 重写清单。
fn run_all(root: &Path) -> Result<(), String> {
    check_naga_lock()?;
    // **先全部编译 + 官方校验通过，再落盘**：逐个写的话，第二个目标编译失败会
    // 留下"第一个已更新、清单没更新"的半个批次——那正是本工具要消灭的漂移。
    let mut built: Vec<(&'static Target, Vec<u8>, usize)> = Vec::new();
    for t in TARGETS {
        let src_path = root.join(t.wgsl);
        let raw = fs::read_to_string(&src_path)
            .map_err(|e| format!("读不到 {}：{e}", src_path.display()))?;
        let src = expand_includes(&raw, src_path.parent().unwrap_or(Path::new(".")), 8)?;
        let words = compile(&src, t.entry)?;
        let bytes = words_to_bytes(&words);
        match spirv_val(&bytes, t.spv)? {
            Some(s) => println!("[val] {s}"),
            None => println!("[val] 跳过：本机未安装 spirv-val（不属于 Rust 依赖）"),
        }
        built.push((t, bytes, words.len()));
    }
    let mut total = 0usize;
    for (t, bytes, words) in &built {
        let out = root.join(t.spv);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("建目录 {} 失败：{e}", parent.display()))?;
        }
        fs::write(&out, bytes).map_err(|e| format!("写 {} 失败：{e}", out.display()))?;
        total += bytes.len();
        println!(
            "[ok] {} → {}（entry={}，{} B = {} word，sha256 {}）",
            t.wgsl,
            t.spv,
            t.entry,
            bytes.len(),
            words,
            sha256::sha256_hex(bytes)
        );
    }
    let manifest = build_manifest(root)?;
    let mpath = root.join(MANIFEST);
    fs::write(&mpath, manifest).map_err(|e| format!("写 {} 失败：{e}", mpath.display()))?;
    println!(
        "[ok] {}（{} 条，产物共 {} B，naga {}）",
        MANIFEST,
        TARGETS.len(),
        total,
        NAGA_VERSION
    );
    Ok(())
}

/// `--check`：只校验，**不写盘**；漂移 → `Err`（调用方 exit 1）。
fn run_check(root: &Path) -> Result<(), String> {
    check_naga_lock()?;
    let mut problems: Vec<String> = Vec::new();

    for t in TARGETS {
        let src_path = root.join(t.wgsl);
        let raw = fs::read_to_string(&src_path)
            .map_err(|e| format!("读不到 {}：{e}", src_path.display()))?;
        let src = expand_includes(&raw, src_path.parent().unwrap_or(Path::new(".")), 8)?;
        let fresh = words_to_bytes(&compile(&src, t.entry)?);
        let disk = match fs::read(root.join(t.spv)) {
            Ok(b) => b,
            Err(e) => {
                problems.push(format!("{} 读不到（{e}）", t.spv));
                continue;
            }
        };
        if fresh != disk {
            let at = fresh
                .iter()
                .zip(disk.iter())
                .position(|(a, b)| a != b)
                .map(|i| format!("首个不同偏移 {i}"))
                .unwrap_or_else(|| format!("长度 {} vs {}", fresh.len(), disk.len()));
            problems.push(format!(
                "{} 与重新编译的结果不一致（{at}）——WGSL 改了但 .spv 没重生成，或 .spv 被手改",
                t.spv
            ));
        }
        match spirv_val(&fresh, t.spv)? {
            Some(s) => println!("[val] {s}"),
            None => println!("[val] 跳过：本机未安装 spirv-val（不属于 Rust 依赖）"),
        }
    }

    // 清单对账（两侧 sha256 + entry + naga 版本）。
    let mpath = root.join(MANIFEST);
    match fs::read_to_string(&mpath) {
        Err(e) => problems.push(format!("{MANIFEST} 读不到（{e}）")),
        Ok(text) => {
            let rows = parse_manifest(&text)?;
            if rows.len() != TARGETS.len() {
                problems.push(format!(
                    "{MANIFEST} 有 {} 条记录，TARGETS 有 {} 条",
                    rows.len(),
                    TARGETS.len()
                ));
            }
            for t in TARGETS {
                // 同一 WGSL 可能对应多个入口 ⇒ 以 **spv 产物路径**为唯一键。
                let Some(row) = rows.iter().find(|r| r.spv == t.spv) else {
                    problems.push(format!("{MANIFEST} 缺少 {} 的记录", t.spv));
                    continue;
                };
                if row.wgsl != t.wgsl {
                    problems.push(format!(
                        "{} 的 wgsl 源不符（清单 {} / TARGETS {}）",
                        t.spv, row.wgsl, t.wgsl
                    ));
                }
                let wgsl_sha = sha256::sha256_hex(&fs::read(root.join(t.wgsl)).map_err(|e| e.to_string())?);
                let spv_sha = sha256::sha256_hex(&fs::read(root.join(t.spv)).map_err(|e| e.to_string())?);
                if row.wgsl_sha != wgsl_sha {
                    problems.push(format!(
                        "{} 的 wgsl sha256 不符（清单 {} / 实际 {wgsl_sha}）",
                        t.wgsl, row.wgsl_sha
                    ));
                }
                if row.spv_sha != spv_sha {
                    problems.push(format!(
                        "{} 的 spv sha256 不符（清单 {} / 实际 {spv_sha}）",
                        t.spv, row.spv_sha
                    ));
                }
                if row.entry != t.entry {
                    problems.push(format!(
                        "{} 的 entry 不符（清单 {} / TARGETS {}）",
                        t.spv, row.entry, t.entry
                    ));
                }
                if row.naga != NAGA_VERSION {
                    problems.push(format!(
                        "{} 的 naga 版本不符（清单 {} / 本工具 {NAGA_VERSION}）",
                        t.spv, row.naga
                    ));
                }
            }
        }
    }

    if problems.is_empty() {
        println!(
            "[check] ok：{} 个目标的 .spv 与 WGSL 一致，清单对账通过（naga {NAGA_VERSION}）",
            TARGETS.len()
        );
        Ok(())
    } else {
        Err(format!(
            "发现 {} 处漂移：\n  - {}\n\n修复命令：\n  {FIX_CMD}",
            problems.len(),
            problems.join("\n  - ")
        ))
    }
}

const USAGE: &str = "\
wgsl2spv —— WGSL → SPIR-V（D-21 开发期工具，不入主仓构建）

用法：
  wgsl2spv --in <x.wgsl> --entry <name> --out <x.spv>
  wgsl2spv --all        批量重生成 .spv 并重写 spv_manifest.txt
  wgsl2spv --check      只校验不写盘；漂移则 exit 1 并打印修复命令
  wgsl2spv --list       打印批量清单（代码内 TARGETS）
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();

    // 参数解析（手写：本工具只允许 naga 一个第三方依赖，不引 clap）。
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut entry: Option<String> = None;
    let (mut all, mut check, mut list) = (false, false, false);
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => {
                i += 1;
                input = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                output = args.get(i).map(PathBuf::from);
            }
            "--entry" => {
                i += 1;
                entry = args.get(i).cloned();
            }
            "--all" => all = true,
            "--check" => check = true,
            "--list" => list = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("未知参数 `{other}`\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let result = if check {
        run_check(&root)
    } else if all {
        run_all(&root)
    } else if list {
        for t in TARGETS {
            println!("{}  {}  {}", t.wgsl, t.entry, t.spv);
        }
        Ok(())
    } else {
        match (input.as_deref(), entry.as_deref(), output.as_deref()) {
            (Some(i), Some(e), Some(o)) => run_single(i, e, o),
            _ => {
                eprintln!("--in / --entry / --out 三者必须同时给出（或用 --all / --check）\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[error] {e}");
            ExitCode::FAILURE
        }
    }
}
