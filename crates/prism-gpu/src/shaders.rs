//! WGSL 着色器源码管理与结构自检（D2/D3/D4）。
//!
//! # 定位（验收：GPU/CpuRef 双实现互为判对）
//! 真实 GPU 实装待真机（vulkan/metal 绑定 + 手机 WebGPU 测试页，D-14）。
//! 本模块在**无 GPU 环境**下对 WGSL 做两类可执行的工作：
//!
//! 1. **嵌入源码**：`include_str!` 编译期把全部 WGSL 打进二进制（供未来
//!    WebGPU/Vulkan 后端上传；零运行时文件 I/O）。
//! 2. **结构自检**：非完整语法解析（完整 WGSL 解析器待 D1），而是断言每个
//!    shader 的关键结构——`@compute`/`@vertex`/`@fragment` 入口存在、
//!    括号配平、`fn`/`struct` 声明完整、禁 libm 超越函数名（D-29）、
//!    零第三方 include。这能拦截「文件被误改/截断/误删」一类回归，
//!    真机编译错误仍由 harness 错误面板回传（D-14）。
//!
//! # 禁 libm（D-29）
//! WGSL 的 `sin`/`cos`/`tan`/`exp`/`pow`/`log`/`atan` 在 GPU 上实现的精度
//! 因厂商而异——引擎刻意**不使用**它们，改用固定系数 Maclaurin/区间逼近
//! （见各 shader 的 `cos_sin`/`quarter_*`）。自检断言这些内建名不出现。

/// 嵌入的 WGSL 源文件名 → 源码表。
///
/// 必须是 `static` 而非 `const`：`const` 会在**每个引用点**各物化一份副本，
/// 于是 `abi::prism_shader_src(i)` 返回的地址与调用方按 `SHADERS[i].1` 算出的
/// 地址不同（native 单测与 wasm 宿主都会踩到）。`static` 保证全程序唯一实例，
/// 指针跨 FFI 传递才稳定。
pub static SHADERS: &[(&str, &str)] = &[
    ("rt_common.wgsl", include_str!("../shaders/rt_common.wgsl")),
    ("rt_bvh.wgsl", include_str!("../shaders/rt_bvh.wgsl")),
    ("rt_shade.wgsl", include_str!("../shaders/rt_shade.wgsl")),
    ("rt_wavefront.wgsl", include_str!("../shaders/rt_wavefront.wgsl")),
    ("sim_common.wgsl", include_str!("../shaders/sim_common.wgsl")),
    ("sim_particles.wgsl", include_str!("../shaders/sim_particles.wgsl")),
    ("sim_cloth.wgsl", include_str!("../shaders/sim_cloth.wgsl")),
    ("sim_sph.wgsl", include_str!("../shaders/sim_sph.wgsl")),
    ("text.wgsl", include_str!("../shaders/text.wgsl")),
    // GI 的 GPU compute 路径（SDF sphere tracing 遮挡 + SH-L2 投影/重建）。
    ("gi_compute.wgsl", include_str!("../shaders/gi_compute.wgsl")),
    // 背景渲染 GPU compute shader（demo `paint_background` 热点的 GPU 管线兜底）。
    ("background.wgsl", include_str!("../shaders/background.wgsl")),
];

/// 库文件：只被 `//!include` 引用，**不**作为独立测试项上传 GPU。
///
/// 与 [`SHADERS`] 的区别：SHADERS 的每一份都会被 harness 单独交给
/// `createShaderModule`（真机逐份判对），因此它们必须各自含入口；库文件没有
/// 入口，单独编译无意义，只在展开后随宿主一起编译。
///
/// `rt_scene.wgsl`（场景访问层：材质/灯光/求交/可见性）即属此类——它是
/// `rt_shade` / `rt_wavefront` 的实现细节，而非一个可独立运行的 kernel。
///
/// `hair_scatter.wgsl`（毛发深化 B：Marschner R/TT/TRT 三瓣）亦属此类——
/// 它是纯函数库（M / N / A / S 合成），无 `@compute`/`@vertex`/`@fragment` 入口，
/// 由未来的 hair 材质着色 kernel 经 `//!include hair_scatter.wgsl` 复用。
pub static LIBRARIES: &[(&str, &str)] = &[
    ("rt_scene.wgsl", include_str!("../shaders/rt_scene.wgsl")),
    ("hair_scatter.wgsl", include_str!("../shaders/hair_scatter.wgsl")),
];

/// include 指令前缀（D-14 真机实测确立）。
///
/// WGSL **没有** `#include`（`has_include` 判据禁的正是它），也没有传递包含
/// 语义。真机上 6 份 shader 报 `unresolved call target`（`vdot` / `sample_1d`/
/// `cell_key` / `xpbd_distance_project` / `poly6_2d` …）——它们依赖
/// `rt_common.wgsl` / `sim_common.wgsl` 里的公共件，但公共件从不在真机上与它们
/// 一起编译。引擎自研的拼接包含指令形如 `//!include rt_common.wgsl`，由
/// [`expand`] 在**上传 GPU 之前**做文本级展开。这是 CPU 侧结构自检看不见的
/// 缺陷类别（结构自检只查括号/libm/include，不查符号可解析性）。
pub const INCLUDE_DIRECTIVE: &str = "//!include ";

/// 展开一个 shader 的 `//!include` 指令，返回可直接交给
/// `createShaderModule` 的完整 WGSL 源码。
///
/// 展开规则（确定性，同输入必得同输出——全链路确定性铁律）：
/// * 指令形如 `//!include <文件名>`，必须独占一行；
/// * 被包含文件先于包含者输出（依赖序）；
/// * 同一文件在一次展开中**只输出一次**（菱形依赖去重，避免重复定义）；
/// * 循环依赖以 `visited` 集合截断（不死循环、不 panic——宪章 C18）。
///
/// 未知文件名按注释跳过，使新增 shader 在展开器更新前仍可编译——但会被
/// [`unresolved_symbols`] 报出来，故不会静默通过。
pub fn expand(name: &str) -> String {
    let mut out = String::new();
    let mut visited: Vec<&str> = Vec::new();
    expand_into(name, &mut out, &mut visited);
    out
}

/// [`expand`] 的递归实现（`visited` 承担去重与环截断两个职责）。
///
/// `visited` 借用传入的名字，故两者须同生命周期——用 `'s` 显式绑定，避免
/// `&str` 入参被提升为 `'static` 而不必要地约束调用方。
fn expand_into<'s>(name: &'s str, out: &mut String, visited: &mut Vec<&'s str>) {
    if visited.contains(&name) {
        return;
    }
    let Some(src) = source(name) else {
        return;
    };
    visited.push(name);
    // 头注：标明展开来源，真机错误面板可直接定位到具体文件。
    out.push_str("// ===== begin ");
    out.push_str(name);
    out.push_str(" =====\n");
    for line in src.lines() {
        match parse_include(line) {
            Some(dep) => expand_into(dep, out, visited),
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out.push_str("// ===== end ");
    out.push_str(name);
    out.push_str(" =====\n");
}

/// 解析一行 include 指令；非指令返回 [`None`]。
fn parse_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix(INCLUDE_DIRECTIVE)?;
    let dep = rest.trim();
    if dep.is_empty() { None } else { Some(dep) }
}

/// 展开后的完整源码（供 GPU 上传与静态门禁）。
///
/// 与 [`SHADERS`] 一一对应，顺序即 `prism_shader_src(i)` 的下标顺序。
pub fn expanded_sources() -> Vec<(&'static str, String)> {
    SHADERS.iter().map(|(n, _)| (*n, expand(n))).collect()
}

/// 取指定源码（先查 [`SHADERS`]，再查 [`LIBRARIES`]；未知名返回 [`None`]）。
///
/// include 展开器需要同时看到两者——`//!include rt_scene.wgsl` 指向的是库文件。
pub fn source(name: &str) -> Option<&'static str> {
    SHADERS
        .iter()
        .chain(LIBRARIES.iter())
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
}

/// 结构自检结果（每个 shader 一组断言结果）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderCheck {
    /// 文件名。
    pub name: &'static str,
    /// 是否含至少一个入口（`@compute`/`@vertex`/`@fragment`）。
    pub has_entry: bool,
    /// 圆括号/花括号/方括号是否配平。
    pub brackets_balanced: bool,
    /// `fn` 声明数（> 0）。
    pub fn_count: usize,
    /// 是否出现被禁的 libm 超越函数内建。
    pub has_libm: bool,
    /// 是否含第三方 include（应为 false）。
    pub has_include: bool,
}

impl ShaderCheck {
    /// 综合通过判据：有入口、配平、有函数、无 libm、无 include。
    pub fn is_pass(&self) -> bool {
        self.has_entry && self.brackets_balanced && self.fn_count > 0 && !self.has_libm && !self.has_include
    }
}

/// 仅在拼接时前置、本身不含入口的 include-only 文件（豁免入口断言）。
const INCLUDE_ONLY: &[&str] =
    &["rt_common.wgsl", "rt_bvh.wgsl", "rt_shade.wgsl", "sim_common.wgsl"];

/// WGSL 内建函数/类型/属性名白名单（[`unresolved_symbols`] 用）。
///
/// 词表按 WGSL 规范的 builtin function / builtin type / builtin value /
/// address space / access mode / builtin attribute 六类枚举。刻意**不含**
/// 任何超越函数（`sin`/`cos`/…）——它们被 D-29 单独禁掉，若误列入白名单会让
/// 「符号可解析」判据掩盖 D-29 违规。
const WGSL_BUILTINS: &[&str] = &[
    // ---- 数值内建 ----
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil", "clamp", "cos",
    "cosh", "countLeadingZeros", "countOneBits", "countTrailingZeros", "cross", "degrees",
    "determinant", "distance", "dot", "exp", "exp2", "extractBits", "faceForward",
    "firstLeadingBit", "firstTrailingBit", "floor", "fma", "fract", "frexp", "inverseSqrt", "ldexp",
    "length", "log", "log2", "max", "min", "mix", "modf", "normalize", "pow", "quantizeToF16",
    "radians", "reflect", "refract", "reverseBits", "round", "saturate", "select", "sign", "sin",
    "sinh", "smoothstep", "sqrt", "step", "tan", "tanh", "transpose", "trunc", "dpdx", "dpdy",
    "fwidth", "pack4x8snorm", "unpack4x8snorm", "pack2x16snorm", "pack2x16unorm",
    // ---- 原子 ----
    "atomicAdd", "atomicSub", "atomicMax", "atomicMin", "atomicAnd", "atomicOr", "atomicXor",
    "atomicExchange", "atomicLoad", "atomicStore",
    // ---- 布尔聚合 / 同步 / 长度 ----
    "all", "any", "workgroupBarrier", "storageBarrier", "textureBarrier", "workgroupUniformLoad",
    "arrayLength",
    // ---- 纹理（当前无采样，但列入以免误报）----
    "textureLoad", "textureStore", "textureSample", "textureSampleLevel", "textureDimensions",
    "textureNumLevels", "textureNumLayers", "textureNumSamples",
    // ---- 类型 ----
    "bool", "f16", "f32", "i32", "u32", "vec2", "vec3", "vec4", "vec2i", "vec3i", "vec4i",
    "vec2u", "vec3u", "vec4u", "vec2f", "vec3f", "vec4f", "mat2x2", "mat2x3", "mat2x4", "mat3x2",
    "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4", "mat2x2f", "mat3x3f", "mat4x4f", "array",
    "atomic", "ptr",
    // ---- 地址空间 / 访问模式 ----
    "storage", "uniform", "workgroup", "private", "function", "read", "write", "read_write",
    // ---- 属性 ----
    "compute", "vertex", "fragment", "workgroup_size", "location", "group", "binding", "builtin",
    "position", "vertex_index", "instance_index", "global_invocation_id", "local_invocation_id",
    "local_invocation_index", "workgroup_id", "num_workgroups", "interpolate", "invariant",
    // ---- 其它关键字（WGSL 保留字，作为调用目标出现即为语法错）----
    "bitcast", "const_assert", "override", "alias", "requires", "diagnostic", "enable", "if",
    "else", "for", "while", "loop", "break", "continue", "switch", "case", "default", "discard",
    "fallthrough", "return", "let", "var", "const", "struct", "fn", "true", "false", "in", "out",
    "inout", "type", "typedef", "template", "union", "enum", "class", "do", "goto", "static",
    "void",
];

/// 在**已展开**的 WGSL 里找出"调用了但没定义、也不是内建"的标识符。
///
/// 这是 D-14 真机失败经验的直接固化：真 Tint 报 `unresolved call target 'vdot'`，
/// 而 CPU 侧结构自检（括号配平 / D-29 / 零 include）**全部通过**。本函数让
/// 同一类错误在 `cargo test` 阶段就暴露，不必再等手机。
///
/// 判据（保守，只报确定性的错，零假阳性目标）：
/// 1. 扫描所有 `标识符(` 形式的调用目标；
/// 2. 减去本文件内 `fn 标识符(` 定义；
/// 3. 减去 [`WGSL_BUILTINS`] 白名单；
/// 4. 减去同文件内 `struct 标识符` / `let|var|const 标识符` / `@group..@binding.. var… 标识符`
///    声明的**值名与类型名**（含函数形参名——形参在 `(` 后声明，会被步骤 1 采到）；
/// 5. 剩余即为未解析符号。
pub fn unresolved_symbols(expanded: &str) -> Vec<String> {
    // 一次性去注释：后续所有扫描都在干净文本上进行（临时值须绑定，勿内联）。
    let src = cleaned(expanded);

    // ---- 收集已定义 / 已声明的名字 ----
    let mut defined: Vec<String> = Vec::new();
    for pat in ["fn ", "struct "] {
        defined.extend(identifiers_after(&src, pat));
    }
    // 值/类型声明：`let x` / `var x` / `const x` / `var<...> x`（含 storage 变量）
    for kw in ["let ", "var ", "var<", "const "] {
        defined.extend(identifiers_after(&src, kw));
    }
    // 函数形参：`fn f(a: T, b: T)` —— 形参名在调用点也会被采到，故一并声明。
    defined.extend(param_names(&src));

    // ---- 收集调用目标 ----
    let mut missing: Vec<String> = Vec::new();
    for name in call_targets(&src) {
        if defined.iter().any(|d| *d == name) || WGSL_BUILTINS.contains(&name.as_str()) {
            continue;
        }
        if !missing.contains(&name) {
            missing.push(name);
        }
    }
    missing.sort();
    missing
}

/// 去掉行注释与块注释（`//` 与 `/* */`），避免注释里的字样被当符号。
fn cleaned(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut in_block = false;
    while let Some(c) = chars.next() {
        if in_block {
            if c == '*' && chars.peek() == Some(&'/') {
                chars.next();
                in_block = false;
            }
            continue;
        }
        if c == '/' && chars.peek() == Some(&'/') {
            for n in chars.by_ref() {
                if n == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            in_block = true;
            continue;
        }
        out.push(c);
    }
    out
}

/// 取 `pat` 之后紧邻的第一个标识符（用于 `fn ` / `let ` 之类）。
fn identifiers_after(src: &str, pat: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = src[from..].find(pat) {
        let start = from + pos + pat.len();
        let rest = &src[start..];
        let name: String = rest
            .chars()
            .skip_while(|c| !c.is_ascii_alphabetic() && *c != '_')
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push(name);
        }
        from = start;
    }
    out
}

/// 取出所有函数形参名（`fn f(a: T, b: ptr<...>)` 的 `a` / `b`）。
fn param_names(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for seg in split_fn_bodies(src) {
        // 形参表 = 第一对圆括号内
        if let Some(open) = seg.find('(') {
            if let Some(close) = matching_paren(seg, open) {
                for piece in split_top(&seg[open + 1..close]) {
                    if let Some(nm) = piece.split(':').next() {
                        let nm = nm.trim();
                        if !nm.is_empty()
                            && nm.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                            && !nm.chars().next().is_some_and(|c| c.is_ascii_digit())
                        {
                            out.push(nm.to_string());
                        }
                    }
                }
            }
        }
    }
    out
}

/// 按 `fn ` 切出每个函数的头部（`{` 之前的部分）。
fn split_fn_bodies(src: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = src[from..].find("fn ") {
        let start = from + pos;
        let head_end = src[start..].find('{').map(|i| start + i).unwrap_or(src.len());
        out.push(&src[start..head_end]);
        from = head_end.max(start + 3);
    }
    out
}

/// 返回与 `open` 处括号配对的 `close` 下标；无配对返回 [`None`]。
fn matching_paren(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for (i, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// 按顶层逗号切分（忽略括号/尖括号内的逗号）。
fn split_top(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '<' | '[' | '{' => depth += 1,
            ')' | '>' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// 取出所有调用目标（`标识符(`）。
fn call_targets(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let name = &src[start..i];
            // 跳过空白后必须是 `(`，且前面不能紧邻 `.`（方法调用不是 WGSL）
            let mut j = i;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            let prev = if start == 0 { ' ' } else { b[start - 1] as char };
            if j < b.len() && b[j] == b'(' && prev != '.' && !name.is_empty() {
                out.push(name.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// 检查一段 WGSL 源码的结构。
///
/// # D-29 判据必须只看「代码」不看「注释」
/// [`LIBMFN`] 的检测在**去注释后**的文本上进行（[`cleaned`] 与
/// [`unresolved_symbols`] 同口径）。原因：数学/出处注释里天然会出现
/// `sin(`/`cos(`/`exp(` 等**记号**（如「M(θ_i,θ_o) 的高斯核含 sin(θ_o)−sin(θ_i)」
/// 或「本文件写作 det_exp(x)」），它们不是被禁的 libm **调用**。若在原始
/// 源码上直接扫子串，会把带公式注释的合法 shader 误判为 D-29 违规（本轮
/// `hair_scatter.wgsl` 的真机回归即此假阳性）。去注释后只留可执行代码，
/// 判据才与「是否真的调用了 libm 超越函数」一致。
pub fn check_shader(name: &'static str, src: &str) -> ShaderCheck {
    let include_only = INCLUDE_ONLY.contains(&name);
    let has_entry =
        include_only || src.contains("@compute") || src.contains("@vertex") || src.contains("@fragment");
    let brackets_balanced = balanced(src, '(', ')') && balanced(src, '{', '}') && balanced(src, '[', ']');
    let fn_count = count_occurrences(src, "fn ");
    // 去注释后再查 libm：注释里的 `sin(`/`cos(`/`exp(` 是数学记号，非调用。
    let code = cleaned(src);
    let has_libm = LIBMFN.iter().any(|f| contains_word(&code, f));
    let has_include = src.contains("#include") || src.contains("@import");
    ShaderCheck { name, has_entry, brackets_balanced, fn_count, has_libm, has_include }
}

/// 对全部**已展开**的 shader 做结构自检 + 符号解析（真机门禁的 CPU 镜像）。
pub fn check_all_expanded() -> Vec<(ShaderCheck, Vec<String>)> {
    expanded_sources()
        .iter()
        .map(|(name, src)| (check_shader(name, src), unresolved_symbols(src)))
        .collect()
}

/// 对全部嵌入 shader 做结构自检。
pub fn check_all() -> Vec<ShaderCheck> {
    SHADERS
        .iter()
        .map(|(n, s)| check_shader(n, s))
        .collect()
}

/// 被禁的 libm 超越函数内建名（D-29）。
const LIBMFN: &[&str] = &["sin(", "cos(", "tan(", "exp(", "pow(", "log(", "atan(", "asin(", "acos("];

/// 去除注释与字符串后统计括号配平（简化词法；够拦截断类回归）。
fn balanced(src: &str, open: char, close: char) -> bool {
    let mut depth: i64 = 0;
    for line in src.lines() {
        // 去掉行注释（// 之后）。
        let body = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        for ch in body.chars() {
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
        }
    }
    depth == 0
}

/// 计数子串出现次数。
fn count_occurrences(src: &str, needle: &str) -> usize {
    src.match_indices(needle).count()
}

/// 判断 `word` 是否作为「内建函数名」出现（前一位非标识符字符）。
fn contains_word(src: &str, word: &str) -> bool {
    let bytes = src.as_bytes();
    let mut start = 0;
    while let Some(pos) = src[start..].find(word) {
        let abs = start + pos;
        let prev_ok = abs == 0
            || {
                let p = bytes[abs - 1] as char;
                !(p.is_ascii_alphanumeric() || p == '_')
            };
        if prev_ok {
            return true;
        }
        start = abs + word.len();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_shaders_pass_structure_check() {
        let checks = check_all();
        assert_eq!(checks.len(), SHADERS.len());
        for c in &checks {
            println!(
                "{}: entry={} balance={} fns={} libm={} include={} => {}",
                c.name, c.has_entry, c.brackets_balanced, c.fn_count, c.has_libm, c.has_include,
                if c.is_pass() { "PASS" } else { "FAIL" }
            );
            assert!(c.has_entry, "{} 无入口", c.name);
            assert!(c.brackets_balanced, "{} 括号不配平（疑似截断）", c.name);
            assert!(c.fn_count > 0, "{} 无函数声明", c.name);
            assert!(!c.has_libm, "{} 出现被禁 libm 内建（D-29）", c.name);
            assert!(!c.has_include, "{} 含第三方 include", c.name);
            assert!(c.is_pass(), "{} 结构自检未通过", c.name);
        }
    }

    #[test]
    fn shader_lookup_by_name() {
        assert!(source("rt_wavefront.wgsl").is_some());
        assert!(source("text.wgsl").is_some());
        assert!(source("missing.wgsl").is_none());
    }

    #[test]
    fn bracket_check_catches_unbalanced() {
        assert!(balanced("fn a() { b(); }", '{', '}'));
        assert!(!balanced("fn a() { b();", '{', '}')); // 缺右括号
        assert!(!balanced("fn a() { } }", '{', '}')); // 多余右括号
        // 行注释内的括号被忽略。
        assert!(balanced("// )){{}\nfn a() {}", '{', '}'));
    }

    // -----------------------------------------------------------------------
    // D-14 真机门禁：把「只有真 GPU 才暴露」的 WGSL 缺陷固化成 CPU 侧断言
    // -----------------------------------------------------------------------

    #[test]
    fn include_expansion_is_acyclic_and_deduplicated() {
        for (name, src) in expanded_sources() {
            // 每个 begin 标记只出现一次 → 菱形依赖已去重、无循环。
            let begins = src.matches("// ===== begin ").count();
            let ends = src.matches("// ===== end ").count();
            assert_eq!(begins, ends, "{name}: begin/end 标记数不等（展开被截断）");
            let mut seen: Vec<&str> = Vec::new();
            let mut from = 0usize;
            while let Some(p) = src[from..].find("// ===== begin ") {
                let start = from + p + "// ===== begin ".len();
                let end = src[start..].find(' ').map(|i| start + i).unwrap_or(src.len());
                let f = &src[start..end];
                assert!(!seen.contains(&f), "{name}: `{f}` 被包含多次（重复定义）");
                seen.push(f);
                from = end;
            }
            // 展开后不再含指令行（否则 GPU 侧会看到裸指令）。
            assert!(
                !src.lines().any(|l| parse_include(l).is_some()),
                "{name}: 展开后仍残留 //!include 指令"
            );
            // 自身必然是最后一个 begin（依赖序：被包含者在前）。
            assert!(src.contains(name), "{name}: 展开结果不含自身");
        }
    }

    #[test]
    fn include_expansion_is_deterministic() {
        // 全链路确定性铁律：同输入两遍逐字节相同。
        for (name, _) in expanded_sources() {
            assert_eq!(expand(name), expand(name), "{name}: 两次展开结果不同");
        }
    }

    #[test]
    fn every_expanded_shader_has_no_unresolved_symbols() {
        // 这条断言直接对应 D-14 真机 6 份 shader 的 `unresolved call target`。
        for (chk, missing) in check_all_expanded() {
            assert!(
                missing.is_empty(),
                "{}: 展开后仍有未解析符号 {:?}（真机必报 unresolved call target）",
                chk.name,
                missing
            );
        }
    }

    #[test]
    fn unresolved_symbol_detector_catches_real_machine_failures() {
        // 真机原文：`error: unresolved call target 'vdot'`（rt_bvh 依赖 rt_common）
        let missing = unresolved_symbols("@compute @workgroup_size(1) fn k() { let d = vdot(a, b); }");
        assert_eq!(missing, vec!["vdot".to_string()]);

        // 公共件已 include 进来 → 同名调用可解析（本机修复后的形态）。
        let ok = unresolved_symbols("fn vdot(a: vec3<f32>, b: vec3<f32>) -> f32 { return dot(a, b); }\n\
                                     @compute @workgroup_size(1) fn k() { let d = vdot(a, b); }");
        assert!(ok.is_empty(), "本地定义的 vdot 应可解析，实际 {:?}", ok);

        // 内建名不得误报。
        let bi = unresolved_symbols("@compute @workgroup_size(1) fn k() { let s = sin_of(1.0); }");
        assert_eq!(bi, vec!["sin_of".to_string()], "仅未定义名应被报出");

        let all_builtin = unresolved_symbols(
            "@compute @workgroup_size(64) fn k(@builtin(global_invocation_id) g: vec3<u32>) {\n  \
               let m = min(g.x, arrayLength(&buf)); let v = clamp(g.x, 0u, 8u);\n  \
               workgroupBarrier(); _ = select(1u, 2u, g.x > 0u); _ = m; _ = v; }",
        );
        assert!(all_builtin.is_empty(), "内建名被误报为未解析：{:?}", all_builtin);

        // 注释里的字样不得被当符号。
        let commented = unresolved_symbols("// 调用 vdot 与 sample_1d\n/* sample_2d */\n\
                                           @compute @workgroup_size(1) fn k() { let a = 1u; }");
        assert!(commented.is_empty(), "注释内容被误当符号：{:?}", commented);
    }

    #[test]
    fn include_only_files_are_still_compilable_standalone() {
        // rt_scene / rt_shade 等 include-only 文件在真机上也是独立编译项，
        // 故它们自身也必须零未解析符号。
        for name in ["rt_common.wgsl", "rt_scene.wgsl", "rt_shade.wgsl", "sim_common.wgsl"] {
            let missing = unresolved_symbols(&expand(name));
            assert!(missing.is_empty(), "{name} 独立展开后有未解析符号 {:?}", missing);
        }
    }

    // -----------------------------------------------------------------------
    // 毛发深化 B：hair_scatter.wgsl（库件）源码级断言
    // -----------------------------------------------------------------------

    #[test]
    fn hair_scatter_library_is_registered_and_self_contained() {
        // 登记在 LIBRARIES（库件），且不在 SHADERS（kernel，需入口）。
        assert!(source("hair_scatter.wgsl").is_some(), "hair_scatter.wgsl 未登记");
        assert!(
            !SHADERS.iter().any(|(n, _)| *n == "hair_scatter.wgsl"),
            "纯函数库不应登记为 kernel（SHADERS）"
        );
        // 展开后零未解析符号（rt_common 的 cos_sin 等已包含进来）。
        let missing = unresolved_symbols(&expand("hair_scatter.wgsl"));
        assert!(missing.is_empty(), "hair_scatter.wgsl 展开后有未解析符号 {:?}", missing);
    }

    #[test]
    fn hair_scatter_has_no_entry_and_no_libm() {
        let src = source("hair_scatter.wgsl").expect("源码存在");
        // 纯库件：不含任何入口。
        assert!(!src.contains("@compute"), "库件不应含 @compute 入口");
        assert!(!src.contains("@vertex"), "库件不应含 @vertex 入口");
        assert!(!src.contains("@fragment"), "库件不应含 @fragment 入口");
        // D-29：禁 libm 超越函数调用（注释内的字样不算——用 check_shader 口径）。
        let chk = check_shader("hair_scatter.wgsl", src);
        assert!(!chk.has_libm, "hair_scatter.wgsl 出现被禁 libm 内建（D-29）");
        assert!(chk.brackets_balanced, "hair_scatter.wgsl 括号不配平");
        assert!(chk.fn_count > 0, "hair_scatter.wgsl 无函数声明");
    }

    #[test]
    fn hair_scatter_source_contains_required_formulas() {
        // 源码级断言：关键公式/函数名必须在场（防被误删/改名）。
        let src = source("hair_scatter.wgsl").expect("源码存在");
        for needle in [
            "fn marschner_m(",
            "fn marschner_n(",
            "fn a_r(",
            "fn a_tt(",
            "fn a_trt(",
            "fn hair_scatter_s(",
            "fn bessel_i0(",
            "fn det_exp(",
            // M 公式常量 1/(2β_m²)
            "1.0 / (2.0 * bm2)",
            // N 归一化 2π·I₀(k)²
            "HAIR_TWO_PI * bessel_i0(kk) * bessel_i0(kk)",
            // A&S 9.8.1 / 9.8.2 分段判据
            "ax < 3.75",
            // 出处标注（Marschner 2003 / Abramowitz & Stegun）
            "Marschner",
            "Abramowitz",
        ] {
            assert!(src.contains(needle), "hair_scatter.wgsl 缺少关键内容：{needle}");
        }
        // 无循环的确定性要求：不得出现 for/while/loop 循环关键字。
        for kw in ["for (", "while (", "loop {"] {
            assert!(!src.contains(kw), "hair_scatter.wgsl 含循环 `{kw}`——破坏确定性");
        }
    }

    #[test]
    fn hair_scatter_operation_order_is_m_then_n_then_a() {
        // 全链路确定性铁律：hait 散射的合成顺序必须**固定为 M → N → A**，
        // 防「不同驱动重结合浮点」把 `mn * a` 改写成 `m * (n * a)` 之类。
        // 判据（源码级、紧贴实现，非泛化）：
        //   1. 先 `marschner_m` 再 `marschner_n` 的行出现顺序（m 的定义在 n 之前）；
        //   2. `mn = m * n`（先乘 M 与 N）；
        //   3. 三瓣均为 `mn * a_*`（最后乘 A），且 3 行 mn 前缀一致。
        let src = source("hair_scatter.wgsl").expect("源码存在");
        let body = src
            .split("fn hair_scatter_s(")
            .nth(1)
            .expect("hair_scatter_s 定义存在");
        let i_m = body.find("marschner_m(").expect("先取 M");
        let i_n = body.find("marschner_n(").expect("再取 N");
        assert!(i_m < i_n, "散射顺序错：marschner_m 必须在 marschner_n 之前");
        assert!(body.contains("let mn = m * n;"), "缺 M→N 的中间乘积 `mn = m * n`");
        // 三瓣 A：均为 `mn * a_*`，且落在 M/N 之后。
        let i_mn = body.find("let mn = m * n;").expect("mn 定义存在");
        for a in ["mn * a_r(f0)", "mn * a_tt(f0, absorb_tt)", "mn * a_trt(f0, absorb_trt)"] {
            let pos = body.find(a).unwrap_or_else(|| panic!("缺 A 瓣 `{a}`"));
            assert!(pos > i_mn, "A 瓣 `{a}` 必须在 M·N 之后相乘（顺序破坏）");
        }
    }

    #[test]
    fn hair_scatter_i0_is_bezier_polynomial_no_loop() {
        // I₀ 近似必须是 A&S 9.8.1 的**固定系数多项式**（Horner，无循环），
        // 而非级数求和（级数需循环，破坏确定性）。
        let src = source("hair_scatter.wgsl").expect("源码存在");
        let body = src
            .split("fn bessel_i0(")
            .nth(1)
            .expect("bessel_i0 定义存在");
        // 1. 无循环关键字（取到下一个 `fn ` 之前的函数体）。
        let end = body.find("\nfn ").unwrap_or(body.len());
        let fbody = &body[..end];
        for kw in ["for (", "while (", "loop {", "for(", "while("] {
            assert!(!fbody.contains(kw), "bessel_i0 含循环 `{kw}`——I₀ 应为无环多项式");
        }
        // 2. A&S 9.8.1 两段系数必须在场（小参数 6 次偶多项式 / 大参数负幂多项式）。
        for coef in ["3.5156229", "3.0899424", "1.2067492", "0.2659732", "0.0360768",
                     "0.0045813", "0.39894228", "0.00392377", "3.75"] {
            assert!(fbody.contains(coef), "bessel_i0 缺 A&S 系数 `{coef}`");
        }
        // 3. 大参数分支用确定性 exp（det_exp）而非 libm exp。
        assert!(fbody.contains("det_exp("), "I₀ 大参数分支应复用确定性 det_exp");
    }

    #[test]
    fn libm_detection_ignores_comments_but_catches_real_calls() {
        // 去注释口径的**双向**验证：
        //   正向：注释里的 `sin(`/`cos(`/`exp(` 不得被判为 D-29 违规（本轮真机假阳性）。
        let commented = "// sin(θ_i) 与 cos(φ/2) 的记号 + exp 记号\n\
                         fn f() -> f32 { return 1.0; }";
        assert!(!check_shader("comment_only.wgsl", commented).has_libm,
            "注释里的数学记号不应触发 D-29（假阳性回归）");
        //   负向：真实调用仍必须被抓住（保证去注释没有把判据改「瞎」）。
        let real = "fn f(x: f32) -> f32 { return sin(x); }";
        assert!(check_shader("real_call.wgsl", real).has_libm,
            "真实 libm 调用必须仍被 D-29 抓住");
        let real_exp = "fn f(x: f32) -> f32 { return exp(x); }";
        assert!(check_shader("real_exp.wgsl", real_exp).has_libm,
            "真实 exp() 调用必须仍被 D-29 抓住");
    }
}
