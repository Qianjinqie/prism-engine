#!/usr/bin/env node
// ===========================================================================
// wgsl_lint.mjs — Prism Engine WebGPU 静态门禁（D-14 真机缺陷本地化）
//
// 为什么存在：手机真机测试抓出 3 个缺陷，全部**纯静态可查**，本该在提交前
// 被脚本拦下，而不是等真机。逐条对应本文件的规则 A / E / B：
//
//   真机缺陷 1：WGSL uniform 未按 16 字节对齐（scan_block 的 Params 只 8 字节）
//               → createBindGroup 验证失败            ⇒ 规则 A
//   真机缺陷 2：bind group 的 DataView 偏移双重计算   → i≥1 读到错误句柄 ⇒ 规则 E
//   真机缺陷 3：WebAssembly import 被写成 async       → 整个 wasm 链接失败 ⇒ 规则 B
//
// 三条设计准则：
//   1. **零第三方依赖**：只用 node 内置模块（fs / path / url / vm），与工程
//      "零第三方 crate"铁律同源。
//   2. **每条规则必须能自证非恒真**：恒真的门禁比没有门禁更危险（它给人虚假
//      的安全感）。故每条规则都在本文件内自带一份"故意违反"的样例，
//      `--self-test` 会把样例喂给规则并断言规则**必须报错**。
//   3. **不 panic、不猜**：结构体成员出现无法解析的类型时**判FAIL**（而不是
//      静默跳过），因为"解析不了"本身就是门禁的盲区，必须逼人扩规则。
//
// 用法：
//   node wgsl_lint.mjs                 # 跑全部门禁，失败 exit 1
//   node wgsl_lint.mjs --self-test     # 只跑"故意违反"样例，证明规则非恒真
//   node wgsl_lint.mjs --rule=A        # 只跑某条规则
//   node wgsl_lint.mjs --json          # 机器可读输出（host_test.mjs 用）
//
// 作为库使用（host_test.mjs 接入方式）：
//   import { runWgslLint } from "./wgsl_lint.mjs";
//   for (const c of runWgslLint().checks) check(c.name, c.ok, c.detail);
// ===========================================================================

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve, join } from "node:path";

// F18（无入口库件必须登记进真编译链路）需要读真编译链路的**单一真源**。
//
// 【为什么这两个 import 是安全的、不会把静态门禁拖进真 GPU 链路】
//  * `wgsl_lib_manifest.mjs` 是**纯数据**模块，import 它只得到一个数组。
//  * `wgsl_lib_compile.mjs` 的 CLI 段由 `isMain` 守卫，import 本模块**不会**
//    起 Xvfb / Chromium / SwiftShader，只拿到 `RECIPE_FILES` / `INJECTION_FILES`
//    两个常量数组。
// 反过来说，若把配方集合在 F18 里另写一份，两份会随时间漂移，最终出现
// "门禁说 A、脚本编 B"——那比没有门禁更坏。本项目已因"两处各写一份"吃过亏。
import { LIB_MANIFEST, libManifestFiles } from "./wgsl_lib_manifest.mjs";
import { RECIPE_FILES, INJECTION_FILES, RECIPE_BINDINGS } from "./wgsl_lib_compile.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHADER_DIR = resolve(HERE, "../shaders");
const DEVICE_RS = resolve(HERE, "../src/web/device.rs");
const ABI_RS = resolve(HERE, "../src/web/abi.rs");
const INDEX_HTML = resolve(HERE, "index.html");
const BUILD_SH = resolve(HERE, "build.sh");

// ---------------------------------------------------------------------------
// WGSL 类型布局表（WGSL 规范 §Address space layout constraints）
// ---------------------------------------------------------------------------

/** 标量类型的 [align, size]，单位字节。 */
const SCALAR = {
  u32: [4, 4], i32: [4, 4], f32: [4, 4], f16: [2, 2], bool: [4, 4],
};

/** `vecN<T>` 的 [align, size]（N=2/3/4；WGSL 无 vec1）。 */
const VEC = { 2: [8, 8], 3: [16, 12], 4: [16, 16] };

/** `matCxR<f32>` 的 [align, size]（列向量式存储）。 */
const MAT = {
  "2x2": [8, 16], "3x3": [16, 48], "4x4": [16, 64],
  "2x3": [8, 32], "3x2": [16, 32], "2x4": [8, 32], "4x2": [16, 32],
  "3x4": [16, 64], "4x3": [16, 64],
};

/** 向上取整到align 的倍数。 */
const roundUp = (v, a) => Math.ceil(v / a) * a;

/**
 * 求一个 WGSL 类型的 [align, size]。
 *
 * @param {string} type 去掉空白与注释后的类型串，如 `vec3<f32>`。
 * @param {boolean} inUniform 是否位于 uniform 地址空间（决定数组 stride 规则）。
 * @param {Map<string,{align:number,size:number}>} structs 已知结构体表（递归用）。
 * @returns {{align:number,size:number}|null} 无法识别时返回 null（交人工，不猜）。
 */
function typeLayout(type, inUniform, structs) {
  const t = type.trim();
  if (SCALAR[t]) return { align: SCALAR[t][0], size: SCALAR[t][1] };
  const vec = /^vec([234])<(.+)>$/.exec(t);
  if (vec) {
    const [, n, inner] = vec;
    if (!SCALAR[inner]) return null;              // vec3<别名> 之类，本仓不存在
    return { align: VEC[n][0], size: VEC[n][1] };
  }
  const mat = /^mat([234])x([234])<f32>$/.exec(t);
  if (mat) {
    const key = `${mat[1]}x${mat[2]}`;
    return { align: MAT[key][0], size: MAT[key][1] };
  }
  // 数组：元素 stride = roundUp(元素 size, 元素 align)；uniform 地址空间再要求
  // 16 的倍数（规范硬要求）。
  //
  // 【修正】早期实现写的是 `roundUp(el.align, 16)`——对 f32/u32/vec* 这些
  // `max(align,size)` 已被对齐覆盖的元素碰巧正确，但**对 struct 元素错误**：
  // `array<GiPrim, 4>`（GiPrim size=48, align=16）会算成 stride 16 而非 48，
  // 把 uniform size 从 256 压成 128，与宿主写入量脱钩（规则 A3 直接报红）。
  // WGSL 规范的数组 stride 定义以元素 **size** 为基准，故此处取 `max(align,size)`。
  const arr = /^array<(.+),\s*(\d+)>$/.exec(t);
  if (arr) {
    const el = typeLayout(arr[1], inUniform, structs);
    if (!el) return null;
    const natural = roundUp(el.size, el.align); // 规范：数组元素 stride
    const stride = inUniform ? roundUp(natural, 16) : natural;
    // 数组对齐 = 元素对齐（规范）；size = stride × 元素数。
    return { align: el.align, size: stride * Number(arr[2]) };
  }
  // 嵌套结构体：递归求值（本仓 uniform 里未出现，留作扩展位）。
  if (structs && structs.has(t)) {
    const s = structs.get(t);
    return { align: s.align, size: s.size };
  }
  return null;
}

/**
 * 定位 `struct <Name> { ... }` 的体并解析成员表。
 *
 * 用**花括号配对**而非正则找 `\n}`：D0 内核的 `struct Params` 是**单行**声明
 *（`struct Params { a: u32, b: u32, _pad: u32 };`），正则会一路吞到下一个 `}`
 * （即整个函数体），把成员表算错。WGSL 成员类型里不出现花括号（定长数组用
 * `<>`），故裸计数即安全。
 *
 * @returns {{members:Array<{name:string,type:string}>, raw:string}|null}
 */
/**
 * 剥掉 WGSL 注释（`//` 行注释；WGSL 无块注释）。
 *
 * 必要性（真实踩到的坑）：`device.rs` 的 scan 内核里，为解释"为何要补 _pad"
 * 写的注释**本身就包含一句完整的 `struct Params { block: u32, count: u32 }`**。
 * 结构体解析器按 `struct\s+Params\s*\{` 找声明时会先命中注释里那句，于是把
 * 注释中的 2 成员当成真声明 → payload 算成 8B，与真实的 4 成员脱钩。
 * 门禁自身被注释误导时报出来的数就是错的，比不报更坏。故在**收集源时**统一
 * 剥掉，一次修好所有解析路径（struct 解析、var<uniform> 匹配）。
 */
function stripWgslComments(src) {
  return src.split("\n").map((l) => {
    // WGSL 源码中 `//` 只可能起注释作用（无 URL 字符串字面量），直接找首个出现。
    const i = l.indexOf("//");
    return i < 0 ? l : l.slice(0, i);
  }).join("\n");
}

/**
 * 定位 `struct <Name> { ... }` 的体并解析成员表。
 *
 * 用**花括号配对**而非正则找 `\n}`：D0 内核的 `struct Params` 是**单行**声明
 * （`struct Params { a: u32, b: u32, _pad: u32 };`），正则会一路吞到下一个 `}`
 * （即整个函数体），把成员表算错。WGSL 成员类型里不出现花括号（定长数组用
 * `<>`），故裸计数即安全。
 *
 * @returns {{members:Array<{name:string,type:string}>, raw:string}|null}
 */
function parseStruct(src, structName) {
  const head = new RegExp(`struct\\s+${structName}\\s*\\{`).exec(src);
  if (!head) return null;
  const open = head.index + head[0].length - 1;
  let depth = 0;
  let close = -1;
  for (let i = open; i < src.length; i += 1) {
    const c = src[i];
    if (c === "{") depth += 1;
    else if (c === "}") {
      depth -= 1;
      if (depth === 0) { close = i; break; }
    }
  }
  if (close < 0) return null;
  const raw = src.slice(open + 1, close);
  const clean = raw.replace(/\/\/[^\n]*/g, "");
  const members = [];
  // 【必须按顶层逗号切分】WGSL 成员类型可含逗号：定长数组 `array<T, N>`。早期实现
  // 直接 `clean.split(",")`，于是 `prims: array<GiPrim, 4>` 被切成 `prims: array<GiPrim`
  // 与 `4>` 两段 → 成员表解析失败（gi_compute 的 GiParams 即此形态）。改为跟踪
  // `<>`/`()`/`[]` 深度，只在深度 0 处以逗号切分。
  const chunks = [];
  {
    let depth = 0;
    let start = 0;
    for (let i = 0; i < clean.length; i += 1) {
      const c = clean[i];
      if (c === "<" || c === "(" || c === "[") depth += 1;
      else if (c === ">" || c === ")" || c === "]") depth -= 1;
      else if (c === "," && depth === 0) { chunks.push(clean.slice(start, i)); start = i + 1; }
    }
    chunks.push(clean.slice(start));
  }
  for (const chunk of chunks) {
    const piece = chunk.trim();
    if (!piece) continue;
    // WGSL 成员属性（@align/@size/[[...]]）本仓不使用；出现即判"需扩规则"。
    if (piece.includes("@") || piece.includes("[[")) return null;
    const m = /^(\w+)\s*:\s*(.+)$/.exec(piece);
    if (!m) return null;
    members.push({ name: m[1], type: m[2].trim() });
  }
  if (!members.length) return null;
  return { members, raw };
}

/**
 * 按规范算一个 uniform 结构体的对齐与大小。
 *
 *   align = roundUp(16, max(成员 align))
 *   size  = roundUp(align, 末成员末偏移 + 大小)
 *
 * 同时返回 `payload`（末成员末偏移+大小，**不做**对齐上取整）——它与 `size`
 * 的差就是"隐式填充"。规则 A 之所以还要查payload，是因为真机缺陷 1 的本质
 * 不是"对齐算错了"，而是**宿主侧按 payload 写了 8 字节、WGSL 却按 size 要16
 * 字节**，两端对"这个结构体有几个字节"各说各话。
 *
 * @returns {{align:number,size:number,payload:number,error?:string}|null}
 */
function uniformStructLayout(src, structName) {
  const parsed = parseStruct(src, structName);
  if (!parsed) return { error: "成员列表无法解析" };
  const { members } = parsed;
  // 先建全量结构体表，供嵌套成员递归查表。
  const structs = new Map();
  for (const m of src.matchAll(/struct\s+(\w+)\s*\{/g)) {
    const p = parseStruct(src, m[1]);
    if (!p) continue;
    let off = 0, maxA = 1, bad = false;
    for (const mem of p.members) {
      const l = typeLayout(mem.type, false, structs);
      if (!l) { bad = true; break; }
      off = roundUp(off, l.align) + l.size;
      maxA = Math.max(maxA, l.align);
    }
    if (bad) continue;
    structs.set(m[1], { align: maxA, size: off });
  }
  let offset = 0;
  let maxAlign = 1;
  for (const mem of members) {
    const l = typeLayout(mem.type, true, structs);
    if (!l) return { error: `成员 \`${mem.name}: ${mem.type}\` 的类型无法解析（需扩展 typeLayout）` };
    offset = roundUp(offset, l.align) + l.size;   // 成员按自身对齐上取整偏移
    maxAlign = Math.max(maxAlign, l.align);
  }
  const align = roundUp(Math.max(16, maxAlign), 16);
  return { align, size: roundUp(offset, align), payload: offset };
}

// ---------------------------------------------------------------------------
// 输入装载
// ---------------------------------------------------------------------------

/**
 * 抽取 Rust 源里内嵌的 WGSL 原始字符串：`pub static NAME: &str = r#"..."#;`
 * 或 `pub const NAME: &str = r#"..."#;`（两种都见，历史上混用过const/static）。
 */
function extractEmbeddedWgsl(rsSrc) {
  const out = [];
  const re = /(?:pub\s+(?:static|const)\s+(\w+)\s*:\s*&str\s*=\s*r#")([\s\S]*?)("#;)/g;
  for (const m of rsSrc.matchAll(re)) out.push({ label: `device.rs:${m[1]}`, src: m[2] });
  return out;
}

/** 装载全部 WGSL 源：shaders/*.wgsl + device.rs 内嵌的 D0 内核（已剥注释）。 */
export function collectWgslSources() {
  const srcs = [];
  for (const f of readdirSync(SHADER_DIR).filter((n) => n.endsWith(".wgsl")).sort()) {
    srcs.push({ label: f, src: stripWgslComments(readFileSync(join(SHADER_DIR, f), "utf8")) });
  }
  for (const e of extractEmbeddedWgsl(readFileSync(DEVICE_RS, "utf8"))) {
    srcs.push({ label: e.label, src: stripWgslComments(e.src) });
  }
  return srcs;
}

/**
 * 展开 WGSL 的 `//!include <file>`（递归，去重，防环）。
 *
 * 真机（Rust 侧 `shaders::expand`）会把 include 链拼成单体再编译，故 rt_wavefront.wgsl
 * 的**有效**绑定集含 rt_shade.wgsl（sobol_dir）与 rt_scene.wgsl（scene_params…）。
 * 门禁若只看未展开的 rt_wavefront.wgsl，就会把这些被 include 进来的 uniform 绑定
 * 判成"无声明绑定"（假失败），也无法覆盖它们。这里做与 Rust 侧同语义的展开。
 */
function expandShaderIncludes(label, seen = new Set()) {
  if (seen.has(label)) return "";
  seen.add(label);
  const p = join(SHADER_DIR, label);
  if (!existsSync(p)) return "";
  const raw = readFileSync(p, "utf8");
  let out = "";
  for (const line of raw.split("\n")) {
    const m = /^\s*\/\/!include\s+(\S+)\s*$/.exec(line);
    if (m) out += expandShaderIncludes(m[1], seen) + "\n";
    else out += line + "\n";
  }
  return out;
}

/** 找出源里全部 `var<uniform> <name>: <Type>;` 绑定。 */
function uniformBindings(src) {
  const out = [];
  for (const m of src.matchAll(/var\s*<\s*uniform\s*>\s*(\w+)\s*:\s*([\w<>,\s]+?)\s*;/g)) {
    out.push({ varName: m[1], typeName: m[2].trim() });
  }
  return out;
}

/**
 * 判定：给定的若干 entry point 里，哪些**静态引用**了某个 binding 变量。
 *
 * WGSL 的 `layout:"auto"` 按被 dispatch 的 entry 静态引用生成绑定布局，
 * 所以"要不要给这个 binding 建缓冲"取决于引用关系而非声明存在性。做法是
 * 逐个 entry 抽出函数体（花括号配对），再看体内是否出现该变量名——
 * 保守但足够：漏判为"未引用"只会让这条检查变松，不会产生假失败。
 */
function entryReferences(src, entryNames, varName) {
  if (!Array.isArray(entryNames)) return null;
  const hit = [];
  for (const name of entryNames) {
    const m = new RegExp(`fn\\s+${name}\\s*\\([^)]*\\)[^{]*\\{`).exec(src);
    if (!m) continue;
    // 从函数体的首个 `{` 起做括号配对，抽出完整函数体。
    let i = src.indexOf("{", m.index);
    let depth = 0;
    for (let j = i; j < src.length; j++) {
      const c = src[j];
      if (c === "{") depth++;
      else if (c === "}") { depth--; if (depth === 0) { i = j; break; } }
    }
    const body = src.slice(m.index, i + 1);
    if (new RegExp(`\\b${varName}\\b`).test(body)) hit.push(name);
  }
  return hit;
}

/**
 * 从 abi.rs 自动提取每个测试项的 dispatch 链入口清单。
 *
 * 形态是 `let steps: [(&str, u32, &[u32]); N] = [ ("emit", 1, &[…]), … ];`
 * 后跟 `submit_chain("gpu.sim_sph", dev, wgsl, &steps)`。这里只取入口名——
 * 胶水的 `layout:"auto"` 关心的是"哪些 entry 会被 dispatch"。
 *
 * 三种写法都认：元组字面量 `("name", …)`、`step("name", …)`，以及多 bind group
 * 形态的结构体字段 `MultiStep { entry: "name", … }` / `entry: "name"`。
 */
function extractChainEntries(abiSrc) {
  const out = {};
  // 以 `make_buffer("<id>"` 出现的行作为该测试项块的起点，逐块扫到下一个测试项。
  const marks = [...abiSrc.matchAll(/make_buffer\(\s*"([^"]+)"/g)];
  for (let i = 0; i < marks.length; i++) {
    const id = marks[i][1];
    const from = marks[i].index;
    const to = i + 1 < marks.length ? marks[i + 1].index : abiSrc.length;
    const names = new Set();
    const block = abiSrc.slice(from, to);
    for (const m of block.matchAll(/\(\s*"(\w+)"\s*,\s*\d+\s*,/g)) names.add(m[1]);
    for (const m of block.matchAll(/(?:step|Step)\s*\(\s*"(\w+)"/g)) names.add(m[1]);
    // 多 bind group 链：`MultiStep { entry: "generate", workgroups: 2, groups: … }`。
    for (const m of block.matchAll(/\bentry\s*:\s*"(\w+)"/g)) names.add(m[1]);
    // 排除 submit_kernel/submit_chain 自身的名字与 test id（它们不是 entry）。
    names.delete("submit_chain");
    names.delete("submit_kernel");
    if (names.size) out[id] = [...names];
  }
  return out;
}

// ---------------------------------------------------------------------------
// 规则实现：每条返回 { name, ok, detail }
// ---------------------------------------------------------------------------

/**
 * 规则 A：uniform 结构体 16 字节对齐 + 宿主侧缓冲尺寸交叉校验。
 *
 * A1  size % 16 == 0                （规范保证，回归锚点）
 * A1b 成员类型必须全部可解析（解析盲区即失败）
 * A2  宿主 uniform 缓冲字节数 >= WGSL 计算 size（抓真机缺陷 1）
 * A3  WGSL 声明的有效载荷 payload == 宿主写入字节数（抓"两端各说各话"）
 */
function ruleA(ctx) {
  const checks = [];
  const srcs = ctx.wgsl;
  let bindingCount = 0;

  for (const { label, src } of srcs) {
    for (const { varName, typeName } of uniformBindings(src)) {
      bindingCount += 1;
      const bare = typeName.replace(/^array<(.+),\s*\d+>$/, "$1");  // uniform 数组取元素类型
      const lay = uniformStructLayout(src, bare);
      if (!lay) { checks.push({ name: `A1b ${label}: uniform '${varName}: ${typeName}' 结构体存在`, ok: false, detail: "未找到该结构体声明" }); continue; }
      if (lay.error) {
        checks.push({ name: `A1b ${label}: uniform '${varName}: ${typeName}' 成员可解析`, ok: false, detail: lay.error });
        continue;
      }
      checks.push({
        name: `A1 ${label}: uniform '${varName}: ${typeName}' size=${lay.size}B (payload=${lay.payload}B) 16 字节对齐`,
        ok: lay.size % 16 === 0,
        detail: `align=${lay.align}`,
      });
      // 数组型 uniform 的元素 stride 也要 16 对齐。
      if (/^array</.test(typeName)) {
        checks.push({
          name: `A1 ${label}: uniform '${varName}: ${typeName}' 元素 stride 16 字节对齐`,
          ok: lay.size % 16 === 0, detail: "uniform 数组 stride 规范要求 16 的倍数",
        });
      }
    }
  }
  checks.push({
    name: `A1 全部 uniform 绑定均已过 16 字节对齐门禁（${bindingCount} 个绑定）`,
    ok: bindingCount > 0, detail: `${bindingCount} 个`,
  });

  // ---- A2/A3：与 abi.rs 的建缓冲调用交叉校验 ----
  // abiSrc / devSrc 可被自证用例注入改写后的文本（负向验证必须能真的改数字，
  // 不能靠传一个没人读的参数假装测过）。
  const abi = ctx.abiSrc !== undefined ? ctx.abiSrc : readFileSync(ABI_RS, "utf8");
  const dev = ctx.devSrc !== undefined ? ctx.devSrc : readFileSync(DEVICE_RS, "utf8");
  // 常量可能在 device.rs（D0 内核：SCAN_PARAMS_BYTES…），也可能在 abi.rs
  // （D-14 补测：SIM/SPH/CLOTH/WF_PARAMS_BYTES…，声明为 u32 而非 usize）。
  // 两处都要找，且类型后缀不能写死——否则新测试项会被误报成"常量缺失"，
  // 而门禁一旦误报就会被人当成噪声忽略，反而守不住真缺陷。
  const constRe = (n) =>
    new RegExp(`pub const ${n}: (?:u32|u64|usize) = (\\d+);`).exec(dev)
    || new RegExp(`pub const ${n}: (?:u32|u64|usize) = (\\d+);`).exec(abi);
  // 映射：测试项 id → 内核 WGSL 的定位标签。
  // D0 两项是 device.rs 内嵌的 r#"..."# 源；D-14 四项是 shaders/ 下的独立文件。
  const KERNEL_OF = {
    "gpu.scan_block": "device.rs:WGSL_SCAN_BLOCK",
    "gpu.histogram": "device.rs:WGSL_HISTOGRAM",
    "gpu.sim_particles": "sim_particles.wgsl",
    "gpu.sim_sph": "sim_sph.wgsl",
    "gpu.sim_cloth": "sim_cloth.wgsl",
    "gpu.rt_wavefront": "rt_wavefront.wgsl",
    "gpu.rt_wavefront_multi": "rt_wavefront.wgsl",
    "gpu.gi_compute": "gi_compute.wgsl",
    // `gpu.params_roundtrip` 建的就是 `gi_compute.wgsl::GiParams` 那块 uniform
    // （写 → `copyBufferToBuffer` 读回 → 逐字节比对）。登记进来，规则 A 才能继续
    // 核"宿主尺寸 ↔ WGSL 声明"的自洽性——否则新测试项的 uniform 缓冲会落进
    // "未登记内核标签"分支而误报。
    "gpu.params_roundtrip": "gi_compute.wgsl",
  };
  const calls = [...abi.matchAll(/make_buffer\(\s*"([^"]+)"\s*,\s*dev\s*,\s*([^,]+?)\s*,\s*usage::UNIFORM/g)];
  checks.push({
    name: "A2 abi.rs 的 uniform 建缓冲调用可被门禁解析",
    ok: calls.length >= 2, detail: `${calls.length} 处`,
  });
  // 按 testId 归组：一个测试项可能建**多个** uniform 缓冲（如
  // `gpu.rt_wavefront_multi` 建 params + camera + scene_params 三块）。
  // 早期实现逐条 make_buffer 独立校验、并把每条都当"唯一主绑定"，遇到多 uniform
  // 就会把 camera 的尺寸拿去和 WFParams 的 payload 比 → 假失败。改为按 id 归组后
  // 统一做"内置尺寸 ↔ 声明绑定"的双向配对（见下）。
  const byTest = new Map();
  for (const [, testId, sizeExprRaw] of calls) {
    const expr = sizeExprRaw.trim();
    // 先把"字面量尺寸"也解析出来：即使写的是裸数字，也必须继续做**尺寸比对**。
    // 【真实漏洞】此前遇到非标识符实参就`continue`，于是 `make_buffer(..., 8u32, UNIFORM)`
    // 只被报一句"该用常量"就放过——而"缓冲小于 WGSL 最小绑定"恰恰是 D-14 真机
    // 失败的**唯一根因**。规则必须先判对错、再判风格。
    const asLiteral = /^(\d+)(?:\s*(?:u32|u64|usize|as\s+u32|as\s+usize))?$/.exec(expr);
    const cm = /^(\w+)(?:\s+as\s+u32)?$/.exec(expr);
    if (!cm && !asLiteral) {
      checks.push({ name: `A3 ${testId}: uniform size 实参可解析（具名常量或字面量）`, ok: false, detail: `实参 = \`${expr}\`` });
      continue;
    }
    let hostBytes;
    let constName;
    if (asLiteral) {
      hostBytes = Number(asLiteral[1]);
      constName = `<字面量 ${hostBytes}>`;
    } else {
      constName = cm[1];
      const decl = constRe(constName);
      if (!decl) {
        checks.push({
          name: `A3 ${testId}: 常量 ${constName} 在 device.rs 或 abi.rs 有定义`,
          ok: false, detail: "两处均缺失",
        });
        continue;
      }
      hostBytes = Number(decl[1]);
    }
    if (!byTest.has(testId)) byTest.set(testId, []);
    byTest.get(testId).push({ hostBytes, constName, expr, asLiteral });
  }
  const chainEntries = ctx.chainEntries !== undefined
    ? ctx.chainEntries
    : extractChainEntries(abi);
  for (const [testId, built] of byTest) {
    const kernel = KERNEL_OF[testId];
    const kernelSrc = kernel ? ctx.wgsl.find((s) => s.label === kernel) : undefined;
    if (!kernelSrc) { checks.push({ name: `A3 ${testId}: 能定位内核 WGSL 源`, ok: false, detail: `标签 ${kernel} 未在 WGSL 源清单中` }); continue; }
    // 用**展开 include** 后的有效源做绑定/结构体解析（与真机编译口径一致）：
    // 未展开时 rt_wavefront.wgsl 看不到 rt_scene.wgsl 的 scene_params 等绑定。
    // D0 内嵌内核（device.rs:…）无 include，直接用其原始源。
    const effSrc = kernel.startsWith("device.rs:")
      ? kernelSrc.src
      : stripWgslComments(expandShaderIncludes(kernel));
    const binds = uniformBindings(effSrc);
    if (!binds.length) { checks.push({ name: `A3 ${testId}: 内核 WGSL 含 uniform 绑定`, ok: false, detail: "未找到" }); continue; }
    // 逐个声明绑定：算结构体布局（含 16B 对齐门禁），再由"内置尺寸集合"里找配对。
    // 一个绑定必须恰有一个内置尺寸 == 其 payload 或 size（无隐式填充，见下）且 >= payload。
    // 找不到配对 ⇒ 该绑定未被宿主建缓冲；此时要看它是否被本项 dispatch 的入口引用：
    // 被引用却无缓冲 = 真缺陷（layout:auto 会缺绑定）；未被引用 = 合法免建。
    const layouts = binds.map((b) => ({ b, lay: uniformStructLayout(effSrc, b.typeName) }));
    // 两遍匹配（避免"贪心 >= payload"把宽缓冲误配给窄绑定，挤掉真正的宿主）：
    //   第一遍：精确匹配（hostBytes == size 或 == payload），一对一认领。
    //   第二遍：对仍未认领的绑定，才退用 hostBytes >= payload 的宽匹配。
    const claimed = new Set();
    const matchOf = new Map();
    for (let bi = 0; bi < layouts.length; bi++) {
      const { lay } = layouts[bi];
      if (!lay || lay.error) continue;
      const idx = built.findIndex((x, i) => !claimed.has(i) && (x.hostBytes === lay.size || x.hostBytes === lay.payload));
      if (idx >= 0) { claimed.add(idx); matchOf.set(bi, idx); }
    }
    for (let bi = 0; bi < layouts.length; bi++) {
      if (matchOf.has(bi)) continue;
      const { lay } = layouts[bi];
      if (!lay || lay.error) continue;
      const idx = built.findIndex((x, i) => !claimed.has(i) && x.hostBytes >= lay.payload);
      if (idx >= 0) { claimed.add(idx); matchOf.set(bi, idx); }
    }
    for (let bi = 0; bi < layouts.length; bi++) {
      const { b, lay } = layouts[bi];
      if (!lay || lay.error) {
        checks.push({ name: `A3 ${testId}: uniform ${b.varName}:${b.typeName} 结构体可解析`, ok: false, detail: (lay && lay.error) || "解析失败" });
        continue;
      }
      checks.push({
        name: `A1 ${testId}: uniform ${b.varName}:${b.typeName} 16 字节对齐（size ${lay.size}B / payload ${lay.payload}B）`,
        ok: lay.size % 16 === 0,
        detail: lay.size % 16 === 0 ? "" : `size ${lay.size}B 非 16 的倍数`,
      });
      const matchIdx = matchOf.has(bi) ? matchOf.get(bi) : -1;
      const used = chainEntries ? entryReferences(effSrc, chainEntries[testId], b.varName) : null;
      if (matchIdx < 0) {
        // 未建可承载缓冲。先看该绑定是否被本项 dispatch 的入口引用：
        //   * 明确未被引用（used 是空数组）⇒ 合法免建（layout:auto 不为它生成绑定），记 ok 说明。
        //   * 被引用却无缓冲 ⇒ A2 真缺陷（真机必缺绑定）。
        //   * 引用关系不可判定（无 chainEntries，如 submit_kernel 的单入口项）⇒
        //     若本项建了 uniform 缓冲但都不够大 ⇒ A2（D-14 真机缺陷 1）；否则 A3（输入缺失）。
        if (used !== null && used.length === 0) {
          checks.push({ name: `A2 ${testId}: 额外 uniform ${b.varName} 仅被未 dispatch 的入口引用（可免建缓冲）`, ok: true, detail: "本项 dispatch 的入口均不引用它" });
        } else if (used !== null && used.length) {
          checks.push({ name: `A2 ${testId}: uniform ${b.varName} 被 dispatch 的入口 ${used.join("/")} 引用却无宿主缓冲`, ok: false, detail: `需一块 ≥${lay.payload}B 的缓冲` });
        } else if (built.length > 0) {
          checks.push({ name: `A2 ${testId}: uniform ${b.varName} 需一块 ≥${lay.payload}B 的宿主缓冲，但本项所建缓冲均过小`, ok: false, detail: `实建尺寸 ${JSON.stringify(built.map((x) => x.hostBytes))}（需 ≥${lay.size}B 才能通过 WebGPU 绑定校验）` });
        } else {
          checks.push({ name: `A3 ${testId}: 额外 uniform ${b.varName} 的引用关系可判定`, ok: false, detail: "门禁未登记本测试项的 dispatch 链入口清单，且本项未建 uniform 缓冲" });
        }
        continue;
      }
      const m = built[matchIdx];
      // A2：缓冲必须能装下整个 struct（WebGPU 校验 join 最小绑定 = size）。
      checks.push({
        name: `A2 ${testId}: 宿主 uniform 缓冲 ${m.hostBytes}B >= WGSL 最小绑定 ${lay.size}B（常量 ${m.constName}）`,
        ok: m.hostBytes >= lay.size,
        detail: `宿主 ${m.hostBytes}B / WGSL size ${lay.size}B`,
      });
      // A3：宿主写入量必须与 WGSL 声明的有效载荷自洽——写满 struct（==size，含显式尾填充）
      // 或只写 payload（余下由建缓冲零初始化补足）皆可；其它尺寸即"两端各说各话"。
      checks.push({
        name: `A3 ${testId}: WGSL 声明 ${b.varName} payload ${lay.payload}B 与宿主写入 ${m.hostBytes}B 自洽（无隐式填充）`,
        ok: m.hostBytes === lay.payload || m.hostBytes === lay.size,
        detail: `宿主 ${m.hostBytes}B ∈ {size ${lay.size}B, payload ${lay.payload}B}`,
      });
      checks.push({
        name: `A3 ${testId}: uniform size 走具名常量 ${m.constName} 而非字面量`,
        ok: !m.asLiteral,
        detail: m.asLiteral ? `实参是裸字面量 \`${m.expr}\`；改尺寸时容易只改一处` : "",
      });
    }
    // 反向：每个内置 uniform 缓冲都必须被某个已声明绑定认领（防"建了块无绑定的缓冲"）。
    built.forEach((x, i) => {
      if (claimed.has(i)) return;
      checks.push({
        name: `A3 ${testId}: 内置 uniform 缓冲 ${x.constName}（${x.hostBytes}B）对应某个 WGSL uniform 绑定`,
        ok: false, detail: `无声明绑定的 payload/size 等于 ${x.hostBytes}B`,
      });
    });
  }
  return checks;
}

/**
 * 规则 B：JS 胶水不得有 async import，且 wasm 需要的每个 import 都在胶水中。
 *
 * 做法：把 index.html 的 `glue` 对象**真的在 node里求值**（不是正则猜），
 * 再用 `fn.constructor.name === "AsyncFunction"` 判定。这样 `async` 写成
 * `async () =>`、`async function`、shorthand `async foo()` 都会被抓到，
 * 且不依赖"import 名单"这一外部清单——名单从 wasm 二进制
 * （`WebAssembly.Module.imports`）直接读，与 Rust 侧 `extern "C"` 声明同步。
 */
function ruleB(ctx) {
  const checks = [];
  const html = ctx.html;
  // 提取 `const glue = { ... };` 的对象字面量文本（花括号配对）。
  const head = /const\s+glue\s*=\s*\{/.exec(html);
  if (!head) { checks.push({ name: "B1 index.html 中存在 glue 对象", ok: false, detail: "未找到 `const glue = {`" }); return checks; }
  const lit = matchBraces(html, head.index + head[0].length - 1);
  if (!lit) { checks.push({ name: "B1 glue 对象字面量可解析", ok: false, detail: "花括号不配平" }); return checks; }
  const literal = lit;

  // 求值：胶水的成员值全是函数，定义时不触碰闭包里的宿主对象，故可在 node 内构造。
  let glue;
  try {
    glue = new Function(`"use strict"; return (${literal});`)();
  } catch (e) {
    checks.push({ name: "B1 glue 对象字面量可求值", ok: false, detail: String(e && e.message || e) });
    return checks;
  }
  checks.push({ name: "B1 glue 对象字面量可求值", ok: true, detail: `${Object.keys(glue).length} 个成员` });

  // B2：逐个成员判定是否 AsyncFunction。
  let asyncFns = [];
  for (const [name, v] of Object.entries(glue)) {
    if (typeof v !== "function") continue;
    if (v.constructor && v.constructor.name === "AsyncFunction") asyncFns.push(name);
  }
  checks.push({
    name: "B2 glue 中无 async 函数（WebAssembly import 必须同步）",
    ok: asyncFns.length === 0,
    detail: asyncFns.length ? `async: ${asyncFns.join(", ")}` : `${Object.keys(glue).filter((k) => typeof glue[k] === "function").length} 个同步函数`,
  });

  // B3：wasm 二进制要求的每个 env 导入都必须在 glue 里且是函数。
  const imports = WebAssembly.Module.imports(ctx.module).map((i) => i.name);
  const missing = imports.filter((n) => typeof glue[n] !== "function");
  checks.push({
    name: `B3 wasm 的 ${imports.length} 个 env 导入在 glue 中均为同步函数`,
    ok: missing.length === 0,
    detail: missing.length ? `未满足: ${missing.join(", ")}` : "全部满足",
  });
  return checks;
}

/**
 * 规则 C：报告/诊断路径上的 wasm 导出调用必须有 `typeof` 守卫。
 *
 * 只约束**任何状态下都必须渲染成功**的函数（renderReport / renderEnv /
 * renderErrors）：真机缺陷 3 之所以难查，正是因为 wasm 链接失败后
 * renderReport 二次抛错（`wasm.prism_abi_version is not a function`），
 * 把真正的链接诊断盖掉了。`wasm` 为 null 只能挡住"没加载"，挡不住
 * "加载了但导出缺失"，故必须是 `typeof ... === "function"`。
 */
const REPORT_FNS = ["renderReport", "renderEnv", "renderErrors"];

function extractFnBody(src, name) {
  const re = new RegExp(`function\\s+${name}\\s*\\(`);
  const m = re.exec(src);
  if (!m) return null;
  const open = src.indexOf("{", m.index);
  if (open < 0) return null;
  return matchBraces(src, open);
}

/**
 * 从 `open`（`{` 的下标）起做花括号配平，返回含首尾花括号的函数/方法体文本。
 *
 * WGSL/JS 的函数体与对象字面量里都可能嵌花括号，故必须计数而不能正则。
 */
function matchBraces(src, open) {
  let depth = 0;
  for (let j = open; j < src.length; j += 1) {
    if (src[j] === "{") depth += 1;
    else if (src[j] === "}") {
      depth -= 1;
      if (depth === 0) return src.slice(open, j + 1);
    }
  }
  return null;
}

/**
 * 去掉 JS 行注释与块注释，但**保留字符串字面量**。
 *
 * 必要性：index.html 里`prism_create_bind_group` 的注释恰好写着一份"错误写法"
 * 示例（`new DataView(mem.buffer, bufs_ptr + i*4, 4)`）。若不剥注释，规则 E 会
 * 拿注释里的反例去判定真实代码 → **误报**。门禁一旦误报就会被整体关掉，所以
 * 剥注释是正确性问题，不是洁癖。
 */
function stripJsComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i], d = src[i + 1];
    if (c === "/" && d === "/") { while (i < n && src[i] !== "\n") i += 1; continue; }
    if (c === "/" && d === "*") { i += 2; while (i < n && !(src[i] === "*" && src[i + 1] === "/")) i += 1; i += 2; continue; }
    if (c === '"' || c === "'" || c === "`") {
      const q = c; out += c; i += 1;
      while (i < n && src[i] !== q) { if (src[i] === "\\") { out += src[i]; i += 1; } out += src[i]; i += 1; }
      out += q; i += 1; continue;
    }
    out += c; i += 1;
  }
  return out;
}

/**
 * 提取胶水里某个成员的函数体。三种写法都要覆盖（胶水池里三种都出现过）：
 *   `name: (a,b) => {…}` / `name(a,b){…}` / `function name(…){…}`
 */
function extractMemberBody(src, name) {
  const arrow = new RegExp(`\\b${name}\\s*:\\s*(?:async\\s*)?\\([^)]*\\)\\s*=>\\s*\\{`).exec(src);
  if (arrow) return matchBraces(src, src.indexOf("{", arrow.index));
  const method = new RegExp(`\\b${name}\\s*\\(\\s*[^)]*\\)\\s*\\{`).exec(src);
  if (method) return matchBraces(src, src.indexOf("{", method.index));
  return extractFnBody(src, name);
}

/**
 * 取成员体并**剥掉注释**（规则 C/D/E 一律走这个入口）。
 *
 * 规则 D4 统计字面量出现次数时也必须用剥注释后的文本，否则注释里为了说明
 * 历史缺陷而写的 `prism_gpu.wasm` 会把 D4 顶成 FAIL。
 */
function extractMemberCode(src, name) {
  const body = extractMemberBody(src, name);
  return body === null ? null : stripJsComments(body);
}

/**
 * 规则 C 的调用图展开：从roots 出发，把函数体里调用的**本文件内定义的**辅助
 * 函数也纳入被检查集合。深度 2 层足够覆盖当前的 abiVersionText 这类抽取；
 * 再深就改成显式登记，避免无界展开。
 */
function expandCallGraph(html, roots, depth = 2) {
  const seen = new Set();
  const queue = [...roots];
  while (queue.length) {
    const name = queue.shift();
    if (seen.has(name)) continue;
    const body = extractFnBody(html, name) || extractMemberBody(html, name);
    if (!body) continue;          // 调用了但本文件里定义不了（内建/全局）→ 不跟进
    seen.add(name);
    if (depth <= 0) continue;
    const code = stripJsComments(body);
    for (const m of code.matchAll(/\b([A-Za-z_$][\w$]*)\s*\(/g)) {
      const callee = m[1];
      if (JS_BUILTINS.has(callee) || seen.has(callee)) continue;
      if (extractFnBody(html, callee) || extractMemberBody(html, callee)) queue.push(callee);
    }
  }
  return seen;
}

/** 不该被当成"本文件辅助函数"的 JS 内建/全局名。 */
const JS_BUILTINS = new Set([
  "if", "for", "while", "switch", "catch", "return", "typeof", "function", "new",
  "String", "Number", "Boolean", "Object", "Array", "JSON", "Math", "Date", "RegExp",
  "Promise", "Map", "Set", "Error", "Uint8Array", "Uint32Array", "Float32Array",
  "ArrayBuffer", "DataView", "TextDecoder", "TextEncoder", "WebAssembly", "navigator",
  "GPUBufferUsage", "GPUMapMode", "GPUDevice", "document", "window", "fetch",
  "parseInt", "parseFloat", "isNaN", "setTimeout", "require", "import", "super",
]);

function ruleC(ctx) {
  const checks = [];
  const html = ctx.html;
  // 【逐个root 独立展开调用图】守卫**可以**写在被调用的辅助函数里（如
  // abiVersionText）。两个要点：
  //   1. 必须展开，否则 renderReport把wasm 调用挪进 helper 后，本函数退化成
  //      "不调用任何 wasm 导出" → 恒真通过，门禁形同虚设。
  //   2. 必须**逐个 root 独立**展开。若共用一个 reachable 集合，renderReport
  //      的守卫会顺带替 renderEnv 兜底 → renderEnv 真的漏了守卫也照样 PASS。
  for (const root of REPORT_FNS) {
    if (!extractFnBody(html, root)) {
      checks.push({ name: `C ${root}()存在于 index.html`, ok: false, detail: "未找到函数定义" });
      continue;
    }
    const reachable = expandCallGraph(html, [root]);
    const hits = new Map();   // 导出名 -> 调用它的函数名集合
    for (const fn of reachable) {
      const body = stripJsComments(extractFnBody(html, fn) || extractMemberBody(html, fn) || "");
      for (const m of body.matchAll(/\bwasm\.(\w+)\s*\(/g)) {
        if (!hits.has(m[1])) hits.set(m[1], new Set());
        hits.get(m[1]).add(fn);
      }
    }
    if (!hits.size) {
      checks.push({ name: `C ${root}() 报告路径不触碰 wasm 导出`, ok: true, detail: "无调用（合法）" });
      continue;
    }
    for (const [fnName, callers] of hits) {
      // 守卫写在 root 自己或它调到的任一辅助函数体内都算数，但**必须存在**。
      const guardedIn = [...reachable].filter((fn) => {
        const b = stripJsComments(extractFnBody(html, fn) || extractMemberBody(html, fn) || "");
        return new RegExp(`typeof\\s+wasm\\.${fnName}\\s*===\\s*["']function["']`).test(b);
      });
      checks.push({
        name: `C ${root}() 调用 wasm.${fnName}() 有 typeof 守卫`,
        ok: guardedIn.length > 0,
        detail: guardedIn.length
          ? `守卫位于 ${guardedIn.join(", ")}；调用点 ${[...callers].join(", ")}`
          : `调用点 ${[...callers].join(", ")} 均无守卫：wasm 存在但导出缺失时会二次抛错，盖掉真正的错误信息`,
      });
    }
  }
  return checks;
}

/**
 * 规则 D：关键尺寸必须走命名常量，禁止字面量散落。
 *
 * 判据不是"数字很难看"，而是**字面量会与宿主侧常量脱钩**：真机缺陷 1 的
 * 起点就是"8"这个字面量。与 Rust 侧 `*_PARAMS_BYTES` 常量同一思路。
 *   D1 index.html 声明 BYTES_PER_U32
 *   D2 bind group 路径用该常量，不出现裸 `4`字节宽度字面量
 *   D3 wasm 文件名在 index.html 中只出现一次（定义处）
 *   D4 build.sh 中 wasm 文件名只有一处字面量（变量定义处）
 */
function ruleD(ctx) {
  const checks = [];
  const html = stripJsComments(ctx.html);
  checks.push({ name: "D1 index.html 声明 BYTES_PER_U32 常量", ok: /const\s+BYTES_PER_U32\s*=\s*4\s*;/.test(html), detail: "" });

  const bg = extractMemberBody(html, "prism_create_bind_group");
  if (!bg) { checks.push({ name: "D2 bind group 创建函数可被门禁解析", ok: false, detail: "未找到 prism_create_bind_group" }); }
  else {
    const literals = [...bg.matchAll(/(?<![\w.])(?:\*\s*4\b|,\s*4\s*[,)])/g)];
    checks.push({
      name: "D2 prism_create_bind_group 用 BYTES_PER_U32 而非裸 4 字节字面量",
      ok: /BYTES_PER_U32/.test(bg) && literals.length === 0,
      detail: literals.length ? `发现 ${literals.length} 处裸 4：${literals.map((m) => m[0].trim()).join(" / ")}` : "",
    });
  }
  const htmlNameCount = (html.match(/prism_gpu\.wasm/g) || []).length;
  checks.push({ name: "D3 index.html 中 wasm 文件名只出现一次（WASM_URL 常量）", ok: htmlNameCount === 1, detail: `${htmlNameCount} 处` });
  const shNameCount = (stripShellComments(readFileSync(BUILD_SH, "utf8")).match(/prism_gpu\.wasm/g) || []).length;
  checks.push({ name: "D4 build.sh 中 wasm 文件名只出现一次（WASM_NAME 变量）", ok: shNameCount === 1, detail: `${shNameCount} 处` });
  return checks;
}

/** 去掉 shell 的整行 `#` 注释（build.sh 里全是这种），保留 shebang 无关性。 */
function stripShellComments(src) {
  return src.split("\n").filter((l) => !/^\s*#/.test(l)).join("\n");
}

/**
 * 规则 E：bind group 的 DataView 偏移不得双重计算（真机缺陷 2）。
 *
 * 病态写法：`new DataView(mem.buffer, bufs_ptr + i*4, 4)` 之后又
 * `getUint32(i*4)`——view 自身已从第 i 个句柄开始，再加一次 i*4 就偏移了
 * 两倍，i≥1 读到错误句柄，且**不报任何错**（只是结果错）。
 *
 * 判据（两条同时成立才算通过）：
 *   E1 DataView 的 byteOffset 是**裸标识符**（不是 `ptr + i*4` 这类表达式）
 *   E2 读索引相对 view 起点，即 `getUint32(i * BYTES_PER_U32)` 而非在
 *      已偏移的 view 上再加偏移
 */
function ruleE(ctx) {
  const checks = [];
  const html = ctx.html;
  const body = extractMemberCode(html, "prism_create_bind_group");
  if (!body) { checks.push({ name: "E prism_create_bind_group 可被门禁解析", ok: false, detail: "未找到" }); return checks; }

  const dv = /new\s+DataView\(\s*[\w.]+\s*,\s*([^,]+?)\s*,/.exec(body);
  if (!dv) { checks.push({ name: "E1 bind group 用 DataView 读取句柄数组", ok: false, detail: "未找到 DataView 构造" }); return checks; }
  const offExpr = dv[1].trim();
  const offIsBareIdent = /^\w+$/.test(offExpr);
  checks.push({
    name: "E1 DataView 的 byteOffset 是裸标识符（view 覆盖整个句柄数组）",
    ok: offIsBareIdent,
    detail: offIsBareIdent ? `offset = ${offExpr}` : `offset = \`${offExpr}\` 依赖循环变量 ⇒ 偏移会被再加一次`,
  });
  // DataView 必须在循环外构造一次（循环内构造 = 每轮重建，虽不错但白费）。
  const forIdx = body.indexOf("for (");
  const dvIdx = body.indexOf("new DataView");
  checks.push({
    name: "E2 DataView 在循环外构造一次（循环不变量提升）",
    ok: forIdx < 0 || dvIdx < forIdx,
    detail: forIdx < 0 ? "" : (dvIdx > forIdx ? "DataView 位于 for 循环体内" : ""),
  });
  // 读索引必须以循环变量为基准（`i * BYTES_PER_U32`），而不是绝对偏移。
  const reads = [...body.matchAll(/getUint32\(\s*([^,)]+)/g)].map((x) => x[1].trim());
  const allRel = reads.length > 0 && reads.every((r) => /^\w+\s*\*\s*BYTES_PER_U32$/.test(r));
  checks.push({
    name: "E3 getUint32 索引相对 view 起点（i * BYTES_PER_U32）",
    ok: allRel,
    detail: allRel ? reads.join(" / ") : `实际索引: ${reads.join(" / ")}`,
  });
  return checks;
}

/**
 * 剥掉 Rust 源码的注释（行注释 `//`、`//!`，以及两种块注释）。
 *
 * 跨语言同构断言（F16）要读 `hair_scatter.rs`，而该文件的文档注释里大量出现
 * `2.0 * beta * beta` 这类**被讨论过的错误写法**（用作反例说明）。不去注释就会
 * 把反例当成实现 → 假阳性。故 F16 一律走本函数。
 *
 * 【为什么用单遍扫描器而非两次正则替换】正则方案有两种失效模式：
 *   (1) 先剥块注释：`//!` 文档行里若出现块注释起始符（例如讲 C 语法），正则会
 *       一路吞到下一个块注释结束符，把中间的真实代码整段吃掉 → 断言读到空源码。
 *   (2) 先剥行注释：块注释内的双斜杠（例如 doc 链接）会提前截断，
 *       块注释残留的结束符之后进入错误状态。
 * 单遍状态机按"当前位置是否处于注释内"逐字符前进，天然免疫这两类问题。
 * 字符串字面量内的 `//` 不做特殊处理——本文件断言不涉及含 `//` 的字面量。
 *
 * @param {string} src Rust 源码
 * @returns {string} 去掉注释后的源码（注释位置以空格替换，保持行数不变）
 */
function stripRustComments(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    const two = src.slice(i, i + 2);
    if (two === "//") {
      // 行注释：吃掉到行尾（换行符保留，维持行号一致）。
      while (i < n && src[i] !== "\n") i += 1;
      continue;
    }
    if (two === "/*") {
      // 块注释：吃掉到闭合。嵌套块注释在 Rust 里合法（`/* /* */ */`），
      // 故用深度计数而非非贪婪正则。
      let depth = 0;
      while (i < n) {
        if (src.slice(i, i + 2) === "/*") { depth += 1; i += 2; }
        else if (src.slice(i, i + 2) === "*/") { depth -= 1; i += 2; if (depth === 0) break; }
        else { if (src[i] === "\n") out += "\n"; i += 1; }
      }
      continue;
    }
    out += src[i];
    i += 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// 毛发散射的 JS 侧参考实现（模块级，供 F5 与新增的 F8–F15 复用）
// ---------------------------------------------------------------------------
//
// 【为什么提到模块级】F5 需要「WGSL 同式的 JS 复刻」，F9/F10 需要「CPU 同式的
// JS 复刻」。若各规则各写一份，两份会随时间漂移，最终变成"两个都不对但互相
// 印证"的假绿。故此处**单一真源**，规则只引用不重定义。
//
// 【精度口径】这里用 f64（JS number）算。门禁的职责是抓**结构性**回归
// （系数写错、归一化漏因子、阶数不够），f64 复刻的容差取 1e-6 即可；
// 真正的 f32 逐位对拍由 `hair_parity_gpu.mjs` 在真 GPU 上做（1e-5 阈值）。

/** JS 复刻 WGSL `det_exp`：区间归约 + 7 阶 Taylor（f64 口径）。 */
function refDetExp(v) {
  const LN2 = 0.6931471805599453;
  if (v <= -87.0) return 0.0;
  const k = Math.min(Math.max(Math.floor(v / LN2 + 0.5), -126.0), 127.0);
  const r = v - k * LN2;
  // Horner 到 1/7!（与 WGSL / CPU 完全同一排布）。
  const p = 1
    + r * (1
    + r * (0.5
    + r * (1 / 6
    + r * (1 / 24
    + r * (1 / 120
    + r * (1 / 720 + r * (1 / 5040)))))));
  return Math.pow(2, k) * p;
}

/** JS 复刻 WGSL/CPU `bessel_i0`（A&S 9.8.1/9.8.2，f64 口径）。 */
function refI0Wgsl(x) {
  const ax = Math.abs(x);
  if (ax < 3.75) {
    const t = x / 3.75, t2 = t * t;
    return 1 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
      + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
  }
  const t = 3.75 / ax;
  const p = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
    + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
    + t * (-0.01647633 + t * 0.00392377)))))));
  // WGSL 与 CPU 均为 `det_exp(ax) / sqrt(ax)`（**不是** inverseSqrt——后者是
  // 快速近似，精度不受 IEEE 约束，违反确定性铁律）。
  return (refDetExp(ax) / Math.sqrt(ax)) * p;
}

/** 高精度真值：I₀(x) = Σ_{m≥0} (x²/4)^m / (m!)²（仅参照，允许循环）。 */
function refI0Series(x) {
  const y = (x * x) / 4;
  let term = 1, sum = 1;
  for (let m = 1; m < 60; m += 1) {
    term *= y / (m * m);
    sum += term;
    if (term < 1e-20 * sum) break;
  }
  return sum;
}

/**
 * 解析 WGSL 源码里所有 `fn 名(形参表) -> 返回类型` 的**形参个数**。
 *
 * 【为什么需要】本次实测发现 `hair_scatter.wgsl` 的 `a_r`/`a_tt`/`a_trt`
 * 声明形参 2/4/4 却按 1/2/2 实参调用——**整个文件无法通过 createShaderModule**。
 * 该文件登记在 `LIBRARIES`（无入口、不参与 kernel 编译），所以这个编译级缺陷
 * 长期绿灯，规则 F2（只查函数名在场）也照样绿。只有"逐函数核对实参个数"才抓得住。
 *
 * 形参表解析按顶层逗号切分（类型可含 `vec3<f32>` 等带尖括号的复合类型）。
 *
 * @param {string} src 已去注释的 WGSL 源码
 * @returns {Map<string, number>} 函数名 → 形参个数
 */
function parseWgslFnArity(src) {
  const arity = new Map();
  const re = /fn\s+(\w+)\s*\(([^)]*)\)/g;
  for (let m = re.exec(src); m !== null; m = re.exec(src)) {
    const params = m[2].trim();
    if (params.length === 0) { arity.set(m[1], 0); continue; }
    let depth = 0, count = 1;
    for (const ch of params) {
      if (ch === "<" || ch === "(") depth += 1;
      else if (ch === ">" || ch === ")") depth -= 1;
      else if (ch === "," && depth === 0) count += 1;
    }
    arity.set(m[1], count);
  }
  return arity;
}

/**
 * 统计源码里每个**已定义函数**的调用点所传实参个数。
 *
 * 只统计 `name(` 形式的调用，且跳过：
 *   * 定义处本身（`fn name(`）；
 *   * 声明但从未被调用的本地函数之外的一切标识符（避免把 WGSL 内建
 *     `mix`/`clamp`/类型转换 `f32(x)` 当成本地函数调用）。
 *
 * @param {string} src 已去注释的 WGSL 源码
 * @param {Map<string, number>} arity `parseWgslFnArity` 的结果
 * @returns {Array<{caller: string, callee: string, args: number}>} 全部调用点
 */
function collectWgslCalls(src, arity) {
  const calls = [];
  const re = /(\w+)\s*\(/g;
  for (let m = re.exec(src); m !== null; m = re.exec(src)) {
    const callee = m[1];
    if (!arity.has(callee)) continue; // 不是本地函数 → 跳过（内建/类型构造）
    // 回看定义处：`fn name(` —— 跳过。
    const before = src.slice(Math.max(0, m.index - 4), m.index);
    if (/\bfn\s+$/.test(before)) continue;
    // 从左括号起做深度配对，取出实参表。
    let depth = 0, i = m.index + m[0].length - 1, end = -1;
    for (let j = i; j < src.length; j += 1) {
      const c = src[j];
      if (c === "(" || c === "<") depth += 1;
      else if (c === ")" || c === ">") {
        depth -= 1;
        if (depth === 0) { end = j; break; }
      }
    }
    if (end < 0) continue;
    const argText = src.slice(i + 1, end).trim();
    let args = 0;
    if (argText.length > 0) {
      args = 1;
      let d2 = 0;
      for (const ch of argText) {
        if (ch === "(" || ch === "<") d2 += 1;
        else if (ch === ")" || ch === ">") d2 -= 1;
        else if (ch === "," && d2 === 0) args += 1;
      }
    }
    // 调用者名：向上回溯到最近的 `fn <name>(`。
    const head = src.lastIndexOf("fn ", m.index);
    const caller = head < 0 ? "<toplevel>" : (/fn\s+(\w+)\s*\(/.exec(src.slice(head, head + 40)) || [, "?"])[1];
    calls.push({ caller, callee, args });
  }
  return calls;
}

// ---------------------------------------------------------------------------
// 规则 F：毛发散射源码级断言 + I₀ 数值对照
// ---------------------------------------------------------------------------
//
// 定位：`shaders/hair_scatter.wgsl` 是纯函数库（库件），无 uniform 绑定，
// 故规则 A 对它是空过——但它承载 Marschner 三瓣这一**数值契约**，
// 必须有一条规则专门守它：
//   F1  文件存在且为纯库件（无 @compute/@vertex/@fragment 入口）
//   F2  关键函数名在场（marschner_m / marschner_n / a_r / a_tt / a_trt /
//       hair_scatter_s / bessel_i0 / det_exp），且无循环关键字（确定性）
//   F2c bessel_i0 为**无环多项式**：A&S 9.8.1/9.8.2 两段系数齐全且大参数走 det_exp
//   F3  关键公式常量在场（M 的 1/(2β_m²)、N 的 2π·I₀(k)²、A&S 9.8.1 的 3.75 分段）
//   F4  出处标注在场（Marschner / Abramowitz）
//   F5  I₀ 数值对照：JS 独立实现（同式）与 WGSL 内的 A&S 多项式近似在
//       [0,20] 上抽查若干点，相对误差 < 1e-4（要求的界）。
//   F6  散射运算顺序固定 M→N→A（mn = m*n 后各自乘 a_*）
//
// F5 的参考实现**独立于 WGSL 源码**（JS 手写同式），故能真正抓住
// "WGSL 里的系数被改错但注释没改"一类回归。
function ruleF(ctx) {
  const checks = [];
  const p = join(SHADER_DIR, "hair_scatter.wgsl");
  // 自证用例可注入改写后的源码（负向验证必须能真的改内容，不能假测）。
  const raw = ctx && ctx.hairSrc !== undefined ? ctx.hairSrc : (existsSync(p) ? readFileSync(p, "utf8") : null);
  if (raw === null) {
    checks.push({ name: "F1 hair_scatter.wgsl 存在", ok: false, detail: "未找到文件" });
    return checks;
  }
  const src = stripWgslComments(raw);
  checks.push({ name: "F1 hair_scatter.wgsl 为纯函数库（无入口）", ok: !/@(compute|vertex|fragment)\b/.test(src), detail: "" });

  // ---- F2 关键函数名 + 无循环 ----
  const fns = ["marschner_m", "marschner_n", "a_r", "a_tt", "a_trt", "hair_scatter_s", "bessel_i0", "det_exp"];
  const missingFn = fns.filter((f) => !new RegExp(`fn\\s+${f}\\s*\\(`).test(src));
  checks.push({ name: "F2 hair_scatter.wgsl 关键函数齐备", ok: missingFn.length === 0, detail: missingFn.length ? `缺 ${missingFn.join(",")}` : `${fns.length} 个` });
  const loops = [...src.matchAll(/\b(for|while|loop)\b\s*[({]/g)].map((m) => m[1]);
  checks.push({ name: "F2b hair_scatter.wgsl 无循环（确定性：近似用固定系数 Horner）", ok: loops.length === 0, detail: loops.length ? `发现 ${loops.join(",")}` : "" });

  // ---- F3 关键公式常量 ----
  const formulas = [
    ["M 前置系数 1/(2β_m²)", /1\.0\s*\/\s*\(2\.0\s*\*\s*bm2\)/],
    ["N 归一化 2π·I₀(k)²", /HAIR_TWO_PI\s*\*\s*bessel_i0\(kk\)\s*\*\s*bessel_i0\(kk\)/],
    ["I₀ A&S 9.8.1 分段点 3.75", /3\.75/],
    ["I₀ A&S 9.8.1 首系数 3.5156229", /3\.5156229/],
    ["I₀ A&S 9.8.2 首系数 0.39894228", /0\.39894228/],
  ];
  for (const [label, re] of formulas) {
    checks.push({ name: `F3 公式在源码中：${label}`, ok: re.test(src), detail: "" });
  }

  // ---- F4 出处标注 ----
  checks.push({ name: "F4 出处标注 Marschner（2003）", ok: /Marschner/.test(raw), detail: "" });
  checks.push({ name: "F4 出处标注 Abramowitz & Stegun", ok: /Abramowitz/.test(raw), detail: "" });

  // ---- F6 散射运算顺序 M → N → A（全链路确定性铁律）----
  //
  // 判据（源码级、紧贴实现）：在 `hair_scatter_s` 体内，
  //   1. 先 `marschner_m(` 再 `marschner_n(`（取得顺序）；
  //   2. 出现中间乘积 `let mn = m * n;`（先乘 M、再乘 N）；
  //   3. 三瓣均为 `mn * a_*`（最后乘 A）。
  // 防「不同驱动把 `mn * a` 重结合成 `m * (n * a)`」一类顺序漂移。
  const sBody = src.split("fn hair_scatter_s(")[1] || "";
  const iM = sBody.indexOf("marschner_m(");
  const iN = sBody.indexOf("marschner_n(");
  const iMn = sBody.indexOf("let mn = m * n;");
  const failOrder = [];
  if (!(iM >= 0 && iN >= 0 && iM < iN)) failOrder.push("M 未先于 N");
  if (iMn < 0) failOrder.push("缺 `let mn = m * n;`");
  for (const a of ["mn * a_r(f0)", "mn * a_tt(f0, absorb_tt)", "mn * a_trt(f0, absorb_trt)"]) {
    const pos = sBody.indexOf(a);
    if (pos < 0) failOrder.push(`缺 A 瓣 ${a}`);
    else if (pos < iMn) failOrder.push(`A 瓣 ${a} 早于 M·N`);
  }
  checks.push({
    name: "F6 散射运算顺序固定 M→N→A（mn = m*n，再乘 a_*）",
    ok: failOrder.length === 0,
    detail: failOrder.length ? failOrder.join("；") : "",
  });

  // ---- F2c I₀ 为无环贝塞尔多项式（A&S 9.8.1/9.8.2 系数齐备）----
  //
  // F2b 只查「全文无循环」；本项进一步锁定 bessel_i0 体内两段系数齐全，
  // 确保 I₀ 是**固定系数 Horner 多项式**而非级数求和（后者需循环、破坏确定性）。
  const i0Body = (src.split("fn bessel_i0(")[1] || "").split("\nfn ")[0];
  const i0Coefs = ["3.5156229", "3.0899424", "1.2067492", "0.2659732", "0.0360768",
    "0.0045813", "0.39894228", "0.00392377", "3.75"];
  const missingCoef = i0Coefs.filter((c) => !i0Body.includes(c));
  checks.push({
    name: "F2c bessel_i0 为无环多项式（A&S 9.8.1/9.8.2 系数齐备）",
    ok: missingCoef.length === 0 && i0Body.includes("det_exp("),
    detail: missingCoef.length ? `缺系数 ${missingCoef.join(",")}` : (i0Body.includes("det_exp(") ? `${i0Coefs.length} 系数` : "大参数分支未用 det_exp"),
  });

  // ---- F5 I₀ 数值对照（JS 独立参考实现 vs WGSL 系数同式）----
  //
  // 参考真值：Abramowitz & Stegun 9.8.1 / 9.8.2 的级数（`refI0Series`，仅参照，
  // 允许循环、不参与 GPU 路径）；对照对象：`refI0Wgsl` —— 从 WGSL 源码**解析出
  // 的系数**复刻（同式独立书写），故系数被改错时本项会红。
  // 【单一真源】两个参考实现在模块级（`refI0Wgsl` / `refI0Series`），F5 与新增的
  // F9/F14 共享。若各规则各写一份，两份会随时间漂移，最终变成"两个都不对
  // 但互相印证"的假绿。
  let worst = 0, worstAt = 0;
  for (const x of [0, 0.5, 1, 2, 3, 3.74, 3.76, 5, 8, 12, 16, 20]) {
    const approx = refI0Wgsl(x);
    const truth = refI0Series(x);
    const rel = Math.abs(approx - truth) / Math.max(truth, 1e-30);
    if (rel > worst) { worst = rel; worstAt = x; }
  }
  checks.push({
    name: `F5 I₀ WGSL 近似 vs 真值 相对误差 < 1e-4（实际最大 ${worst.toExponential(2)} @ x=${worstAt}）`,
    ok: worst < 1e-4,
    detail: "",
  });

  // =========================================================================
  // 以下 F7–F16 为「Marschner 收口」新增（Agent C）
  // =========================================================================
  //
  // 为什么要加这么多条：F1–F6 守的是「公式在不在」，**守不住「公式对不对」**。
  // 本次实测在 F1–F6 全绿的情况下抓到三个真缺陷：
  //   1. `a_r`/`a_tt`/`a_trt` 声明形参与实参个数不符 → 整个文件编译不过。
  //      因为它在 `LIBRARIES` 里（无入口、不参与 kernel 编译），**长期绿灯**。
  //      F6 甚至把 `mn * a_r(f0)` 这类字面量硬编码进断言，与 Rust 测试
  //      `hair_scatter_operation_order_is_m_then_n_then_a` 一起把 bug 锁死了。
  //   2. `marschner_n` 分母用 `I₀(k)²` 而非 `I₀(k/2)²` —— 归一化常数因子
  //      错了（k=8 时约 1430 倍），头发整体发黑。F3 只查"2π·I₀ ² 在场"，照样绿。
  //   3. `det_exp` 用 `x * (1/ln2)` 而非 `x / ln2`、6 阶而非 7 阶、无下溢守卫；
  //      `bessel_i0` 用 `inverseSqrt` —— 这些与 CPU 侧**不同构**，对拍必然炸。
  // 结论：**光靠"常量在不在"的正向断言是假门禁**。故新增的规则一律走
  // 「去注释后逐字面量 / 逐调用点 / 逐跨语言同构」三条硬判据。
  //
  // 【统一前置：全部基于 `src`（= stripWgslComments(raw)）】
  // 铁律：F2b（无循环）与 F7（无 libm）必须去注释后再扫。本项目已吃过教训——
  // 注释里一句 `// sin(θ_i)` 曾被当成 libm 调用产生**假阳性**，逼着人给注释打
  // 补丁绕过门禁。`src` 已在函数开头由 stripWgslComments 生成，此处直接复用。

  // ---- F7 无 libm 超越函数（去注释后扫；D-29 确定性铁律）----
  //
  // 【为什么必须放行自带件】本文件合法地**定义并调用** det_exp / det_asin /
  // det_atan / bessel_i0 —— 它们是自带的确定性近似件，正是为了替换 libm。
  // 若不精确放行，规则会红；放行得太宽（如只 `replace(/det_/g,"")`）又会漏掉
  // `det_exp(` 里嵌套的真 libm。故按**标识符边界**精确放行这 4 个前缀。
  //
  // 附加严判：`det_exp_neg` 是本文件对 `exp(-x)` 的封装，名字里含 "exp"，
  // 但它不是 libm（`exp` 只是名字的一部分），需一并放行。
  const libmAllow = ["det_exp", "det_exp_neg", "det_asin", "det_atan", "det_atan_small", "det_atan_unit", "bessel_i0", "cos_sin"];
  const libmProbe = src
    .replace(/\b(det_exp_neg|det_exp|det_asin|det_atan_small|det_atan_unit|det_atan|bessel_i0|cos_sin)\b/g, "SAFE");
  // 只禁**超越函数**：sin/cos/tan/asin/acos/atan/exp/exp2/log/log2/pow/sqrt…
  // sqrt/floor/abs/clamp/sign/min/max 是 IEEE 精确或有理运算，不在禁列
  // （CPU 侧 `sss::det_exp` 的 doc 明确写了这条口径），故 sqrt 放行。
  const libmNames = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "exp2", "log", "log2", "pow", "inverseSqrt", "inverseSqrt2"];
  const libmHits = [];
  for (const nm of libmNames) {
    // 负向前瞻排除 SAFE_ 前缀残留；前面加 (?<![\w]) 排除 foo_sin 这类标识符片段。
    const re = new RegExp(`(?<![\\w])${nm}\\s*\\(`, "g");
    for (const m of libmProbe.matchAll(re)) {
      const at = m.index;
      // 排除 `SAFE_` 之类被替换后残留的假阳性（allow 名里含这些子串的情况）。
      if (src.slice(Math.max(0, at - 6), at).includes("SAFE_")) continue;
      libmHits.push(`${nm}() @${src.slice(0, at).split("\n").length} 行`);
    }
  }
  checks.push({
    name: "F7 hair_scatter.wgsl 无 libm 超越函数（D-29；已去注释，自带 det_* 件放行）",
    ok: libmHits.length === 0,
    detail: libmHits.length ? libmHits.join("；") : "",
  });

  // ---- F8 形参/实参个数逐调用点一致（真实编译级缺陷的守门规则）----
  //
  // 这条是 F 组里最重要的一条：它把「函数能不能编译」从"真机才知道"变成
  // 静态可查。判据为 `parseWgslFnArity` 声明表 vs `collectWgslCalls` 调用表。
  const fnArity = parseWgslFnArity(src);
  const allCalls = collectWgslCalls(src, fnArity);
  const arityMismatch = allCalls
    .filter((c) => c.args !== fnArity.get(c.callee))
    .map((c) => `${c.caller}→${c.callee} 声明 ${fnArity.get(c.callee)} 参 / 传 ${c.args} 参`);
  checks.push({
    name: "F8 全部调用点实参个数 == 声明形参个数（WGSL 可编译性）",
    ok: arityMismatch.length === 0,
    detail: arityMismatch.length ? arityMismatch.join("；") : `${allCalls.length} 个调用点全部一致`,
  });
  // 反向补充：只对 a_* 三瓣做**逐个点名**断言。这三个函数正是元凶，
  // 泛化规则（"全部调用点"）若因解析器改动而整体失效，这里仍能独立变红。
  for (const a of ["a_r", "a_tt", "a_trt"]) {
    const d = fnArity.get(a);
    const cs = allCalls.filter((c) => c.callee === a);
    checks.push({
      name: `F8 ${a} 声明形参 ${d} 个且所有调用点实参匹配（${cs.length} 处调用）`,
      ok: d !== undefined && cs.length > 0 && cs.every((c) => c.args === d),
      detail: !d ? "未找到声明" : (cs.length === 0 ? "无调用点" : (cs.every((c) => c.args === d) ? "" : "实参个数不符")),
    });
  }

  // ---- F9 marschner_n 归一化分母必须是 I₀(k/2)²（常数因子守门）----
  //
  // 【真实缺陷】原实现分母是 `I₀(k)²`。正确归一来自
  //   ∫₀^{2π} I₀(k·cos(φ/2)) dφ = 2π·I₀(k/2)²
  // 故分母的自变量是 **k/2**。k=8 时 I₀(k)²/I₀(k/2)² ≈ 1430，发色整体偏差。
  // F3 只查 "2π·I₀² 在场"，抓不到自变量错；本条查**自变量表达式**。
  const nBody = (src.split("fn marschner_n(")[1] || "").split("\nfn ")[0];
  const kHalfOk = /\bkk\s*=\s*0\.5\s*\*\s*kh\b/.test(nBody) || /\bk\s*\*\s*0\.5\b/.test(nBody);
  checks.push({
    name: "F9 marschner_n 归一化自变量是 k/2（kk = 0.5 * kh），非 k",
    ok: kHalfOk,
    detail: kHalfOk ? "" : "未找到 kk = 0.5 * kh —— 分母若用 I₀(k)²，结果差约 1430 倍（k=8）",
  });
  // 数值侧独立复核：∫₀^{2π} N(φ,k) dφ 必须 = 1（用 refI0Wgsl 真算，不看源码）。
  // 这条**不依赖任何源码字面量**，所以哪怕归一化写法换了写法也能抓。
  {
    let worstInt = 0, worstK = 0;
    for (const k of [0.5, 2, 8]) {
      const den = 2 * Math.PI * refI0Wgsl(k * 0.5) ** 2;
      const STEPS = 20000;
      let acc = 0;
      for (let i = 0; i < STEPS; i += 1) {
        const phi = ((i + 0.5) * 2 * Math.PI) / STEPS;
        acc += refI0Wgsl(k * Math.cos(0.5 * phi)) / den;
      }
      const integral = (acc * 2 * Math.PI) / STEPS;
      const dev = Math.abs(integral - 1);
      if (dev > worstInt) { worstInt = dev; worstK = k; }
    }
    checks.push({
      name: `F9 N(φ,k) 方位积分 = 1（实际最大偏差 ${worstInt.toExponential(2)} @ k=${worstK}）`,
      ok: worstInt < 1e-3,
      detail: "",
    });
  }

  // ---- F10 det_exp 与 CPU `sss::det_exp` 逐条同构 ----
  //
  // 对拍实测发现的三处不同构（都会让误差从 1e-7 跳到 1e-3 量级）：
  //   1. `x * (1/ln2)` vs `x / ln2` —— f32 下不是同一个数（1 ulp）；
  //      且"乘倒数"把一次除法的舍入变成两次舍入。
  //   2. 6 阶（到 1/720）vs 7 阶（到 1/5040）—— r ∈ [−0.3466, 0.3466] 上
  //      6 阶余项 r⁷/7! ≈ 1.2e-5，正好在 Marschner 敏感量级上。
  //   3. 缺 `x <= -87` 下溢守卫 —— clamp 会把指数域钉在 −126，破坏归约域，
  //      Taylor 在远离 0 处失灵甚至**翻负**（exp 恒 > 0，产出负值即崩）。
  const expBody = (src.split("fn det_exp(")[1] || "").split("\nfn ")[0];
  checks.push({
    name: "F10 det_exp 用除法归约 x / LN_2（非 x * (1/ln2)）",
    ok: /floor\s*\(\s*x\s*\/\s*LN_2/.test(expBody) && !/x\s*\*\s*INV_LN2|x\s*\*\s*\(?\s*1\s*\/\s*LN_2/.test(expBody),
    detail: "",
  });
  checks.push({
    name: "F10 det_exp 下溢守卫 x <= -87 在场（否则 Taylor 在远端翻负）",
    ok: /x\s*<=\s*-87\.0/.test(expBody),
    detail: "",
  });
  checks.push({
    name: "F10 det_exp 指数域钳位 clamp(kf, -126, 127) 在场",
    ok: /clamp\s*\(\s*kf\s*,\s*-126\.0\s*,\s*127\.0\s*\)/.test(expBody),
    detail: "",
  });
  checks.push({
    name: "F10 det_exp Taylor 到 1/7! = 1/5040（7 阶，与 CPU 同项数）",
    ok: /1\.0\s*\/\s*5040\.0/.test(expBody),
    detail: /1\.0\s*\/\s*5040\.0/.test(expBody) ? "" : "只有 6 阶（缺 1/5040）→ 归约域上余项约 1.2e-5",
  });
  checks.push({
    name: "F10 det_exp 2^k 用位构造 bitcast<f32>((i32(k)+127) << 23)",
    ok: /bitcast<f32>\s*\(\s*\(\s*i32\s*\(\s*k\s*\)\s*\+\s*127\s*\)\s*<<\s*23\s*\)/.test(expBody),
    detail: "",
  });

  // ---- F11 bessel_i0 大参数段不得用 inverseSqrt / 倒数 ----
  //
  // `inverseSqrt` 在 WGSL 里允许是**快速近似**倒数平方根，精度不受 IEEE 约束
  // → 跨驱动不一致 → 违反"同输入逐字节可复现"铁律。同理 `1/(x/3.75)` 里的
  // `x/3.75` 之后再取倒数会引入额外舍入；正确写法是先算 `t = 3.75/ax` 再乘 t。
  const i0Big = i0Body.split("3.75").slice(1).join("3.75") || i0Body;
  checks.push({
    name: "F11 bessel_i0 大参数段用 det_exp(ax) / sqrt(ax)，禁用 inverseSqrt",
    ok: /det_exp\s*\(\s*ax\s*\)\s*\/\s*sqrt\s*\(\s*ax\s*\)/.test(i0Big) && !/inverseSqrt/.test(src),
    detail: "",
  });
  checks.push({
    name: "F11 bessel_i0 大参数段 t = 3.75 / ax 后连乘 t（非 1/(x/3.75)）",
    ok: /let\s+t\s*=\s*3\.75\s*\/\s*ax/.test(i0Big) && !/1\s*\/\s*t\b/.test(i0Big),
    detail: "",
  });

  // ---- F12 Beer–Lambert 吸收结构（三瓣衰减的指数路径）----
  //
  // A 瓣的 T = exp(−σ_a·d) 走 det_exp_neg（`exp(-x)` 的封装，不是 libm）。
  // 路径长度 d_tt = 2r/|cosθ_i|（平板近似）、d_trt = 2·d_tt（多一次穿壁）。
  // 这三处若被改写（例如 d_trt 忘乘 2），三瓣比例就错，而能量仍"看起来守恒"。
  checks.push({
    name: "F12 Beer–Lambert：T = det_exp(−x) 封装 det_exp_neg 在场",
    ok: /fn\s+det_exp_neg\s*\(/.test(src) && /return\s+det_exp\s*\(\s*-\s*x\s*\)/.test(src),
    detail: "",
  });
  const preciseBody = (src.split("fn hair_scatter_s_precise(")[1] || "").split("\nfn ")[0];
  checks.push({
    name: "F12 Beer–Lambert：d_tt = 2r/|cosθ_i|、d_trt = 2·d_tt",
    ok: /d_tt\s*=\s*2\.0\s*\*\s*radius\s*\/\s*csafe/.test(preciseBody)
      && /t_trt\s*=\s*det_exp_neg\s*\(\s*sigma_a\s*\*\s*\(\s*2\.0\s*\*\s*d_tt\s*\)\s*\)/.test(preciseBody),
    detail: "",
  });
  checks.push({
    name: "F12 |cosθ_i| 下界守卫 csafe = max(cos_i, 1e-3)（防掠入射除零）",
    ok: /csafe\s*=\s*max\s*\(\s*cos_i\s*,\s*1e-3\s*\)/.test(preciseBody),
    detail: "",
  });

  // ---- F13 Schlick 角依赖 Fresnel（5 次方 + [0,1] 钳制）----
  //
  // F = f0 + (1−f0)·(1−cosθ)^5。5 次方是 Schlick 的定义本身，写成 4 次或
  // 2 次都能编译、都能量守恒，但掠射反射率会差 10% 以上（发梢边缘发亮程度）。
  const schlickBody = (src.split("fn schlick_fresnel(")[1] || "").split("\nfn ")[0];
  checks.push({
    name: "F13 Schlick 五次方 (1−cosθ)^5（x2*x2*x 排布）",
    ok: /x2\s*=\s*x\s*\*\s*x/.test(schlickBody) && /x5\s*=\s*x2\s*\*\s*x2\s*\*\s*x/.test(schlickBody),
    detail: "",
  });
  checks.push({
    name: "F13 Schlick 结果钳制到 [0,1]（能量守恒：F ∈ [0,1]）",
    ok: /return\s+clamp01\s*\(/.test(schlickBody) && /fn\s+clamp01\s*\(/.test(src),
    detail: "",
  });
  // f0 = ((η−1)/(η+1))²：垂直入射反射率。不在场就无法从 η 推出 F。
  const f0Body = (src.split("fn hair_f0(")[1] || "").split("\nfn ")[0];
  checks.push({
    name: "F13 f0 = ((η−1)/(η+1))² 在场（Born & Wolf，η=1.55 约 0.0469）",
    ok: /r0\s*=\s*\(\s*eta\s*-\s*1\.0\s*\)\s*\/\s*d/.test(f0Body) && /r0\s*\*\s*r0/.test(f0Body),
    detail: "",
  });

  // ---- F14 三瓣齐全 + 三瓣 φ 偏角符号约定（用户点名的漏检高发区）----
  //
  // 【为什么这条最该有】三瓣各自的方位偏角是 φ_R = φ、φ_TT = φ − π、φ_TRT = φ。
  // **符号写错时小角度下误差极小**（|φ| 小时 φ 与 φ−π 的 N 值都接近峰值），
  // 极容易漏过——最初正是靠 `forward_scatter` 与 `backward_scatter` 两组
  // golden vector（φ=0 vs φ=π）的 TT 相差 400 倍（7.6e-3 vs 3.26）才确认符号。
  // 故除源码字面量外，还加一条**数值判据**：φ=π 时 TT 的方位项必须显著小于
  // φ=0 时（判据与具体 k 无关，故门槛取得很松，只抓符号级别的错误）。
  const phiBody = (src.split("fn hair_phi_p(")[1] || "").split("\nfn ")[0];
  checks.push({
    name: "F14 三瓣 φ 偏角：TT 翻转 π（p==1 → phi − PI），R/TRT 不偏移",
    ok: /p\s*==\s*1u\s*\{?\s*return\s+phi\s*-\s*PI/.test(phiBody.replace(/\s+/g, " ").replace(/\{ ?/g, " { ")) || /p\s*==\s*1u[\s\S]{0,40}phi\s*-\s*PI/.test(phiBody),
    detail: "",
  });
  checks.push({
    name: "F14 三瓣 α 偏角：R 取负（p==0 → −a），TT/TRT 取正",
    ok: /p\s*==\s*0u[\s\S]{0,40}return\s+-a/.test((src.split("fn hair_alpha_p(")[1] || "").split("\nfn ")[0]),
    detail: "",
  });
  {
    // 数值判据：N(φ=0,k) / N(φ=π,k) 必须 > 20（|cos(φ/2)| 在两端差 2 倍，
    // I₀ 指数放大后差距极大）。若 TT 的符号写反（用 φ 而非 φ−π），
    // φ=0 的 TT 会落在 N 的谷值上，N(0)/N(π) 立刻塌到 ~1。
    const k = 8, den = 2 * Math.PI * refI0Wgsl(k * 0.5) ** 2;
    const nAt = (phi) => refI0Wgsl(k * Math.cos(0.5 * phi)) / den;
    const ratio = nAt(0) / nAt(Math.PI);
    checks.push({
      name: `F14 N(φ=0)/N(φ=π) = ${ratio.toFixed(1)} > 20（φ 符号约定数值判据）`,
      ok: ratio > 20,
      detail: ratio <= 20 ? "φ 偏角符号可能反了：TT 应落在 N 谷值" : "",
    });
  }
  // 三瓣都必须真正参与求和（防止"三瓣里有一瓣被注释掉/漏乘"）。
  checks.push({
    name: "F14 三瓣 R/TT/TRT 在精确入口 hair_scatter_s_precise 中各被求值一次",
    ok: ["m_r * n_r * a_r", "m_tt * n_tt * a_tt", "m_trt * n_trt * a_trt"]
      .every((lit) => preciseBody.includes(lit)),
    detail: "",
  });

  // ---- F15 无半成品桩（含 B 瓣未实现的诚实标注）----
  //
  // 本项目吃过"目标模块被注释掉却报告 149 passed"的亏（GI 模块被
  // `TEMP-SCOPE-WORKAROUND-DISABLE-GI` 注释掉）。对毛发而言等价的风险是：
  // 三瓣只实现两瓣、或者 B 瓣留一个返回常量的 `a_b` 桩。
  // 故断言：(a) 源码无 TODO/半成品标记；(b) 三个 a_* 都有实打实的函数体
  // （不是空 `return 0.0`）；(c) B 瓣**在文档里被显式声明为未实现**，
  // 而不是在代码里偷偷留桩。
  {
    // (a) 半成品标记（去注释后 —— 文档里写"未实现 B 瓣"是允许的诚实标注）。
    const stubMarks = [...src.matchAll(/\b(TODO|FIXME|XXX|HACK|unimplemented|placeholder|stub)\b/g)]
      .map((m) => `${m[1]} @${src.slice(0, m.index).split("\n").length} 行`);
    checks.push({
      name: "F15 hair_scatter.wgsl 无 TODO/FIXME/unimplemented/占位桩标记",
      ok: stubMarks.length === 0,
      detail: stubMarks.join("；"),
    });
    // (b) 三个 a_* 必须有非平凡函数体（排除 `return 0.0` 这类恒值桩）。
    const trivial = [];
    for (const a of ["a_r", "a_tt", "a_trt"]) {
      const body = (src.split(`fn ${a}(`)[1] || "").split("\nfn ")[0];
      const meaningful = /clamp01/.test(body) && !/^\s*return\s+0\.0\s*;?\s*$/m.test(body);
      if (!meaningful) trivial.push(a);
    }
    checks.push({
      name: "F15 a_r/a_tt/a_trt 均有实打实函数体（非 return 0.0 恒值桩）",
      ok: trivial.length === 0,
      detail: trivial.length ? `${trivial.join(",")} 疑似空桩` : "",
    });
    // (c) B 瓣必须被**文档显式声明为未实现**（诚实边界），而非偷偷留桩函数。
    // 这条把"没做"变成"必须写下来"，杜绝"半成品混进交付"。
    const docDeclaresB = /B\s*瓣/.test(raw) && /未实现|尚未实现|not implemented/i.test(raw);
    checks.push({
      name: "F15 B 瓣在模块文档中被显式声明为未实现（诚实边界）",
      ok: docDeclaresB,
      detail: docDeclaresB ? "" : "B 瓣未实装却无文档声明 —— 属于未记录的半成品",
    });
    // (d) 不许出现 B 瓣的半成品函数名（有桩函数 = 既无实现又无标注）。
    const bStub = [...src.matchAll(/\bfn\s+((?:a_b|b_lobe|marschner_b|hair_b)\w*)\s*\(/g)].map((m) => m[1]);
    checks.push({
      name: "F15 无 B 瓣半成品函数（a_b / b_lobe / marschner_b / hair_b）",
      ok: bStub.length === 0,
      detail: bStub.join("；"),
    });
  }

  // ---- F16 CPU/GPU 跨语言同构锚点（跨仓，只读不改）----
  //
  // 【为什么必须跨语言断言】CPU 侧 `m_longitudinal_full` 写的是
  // `2.0 * beta * beta`（Rust 结合为 `(2β)β`），WGSL 写 `2.0 * bm2`（`2(β²)`）
  // —— f32 下**不是同一个数**（β=0.7 时差 1 ulp）。对拍会被这种实现差异污染，
  // 于是真缺陷与舍入噪声混在一起，无法判定阈值。两侧必须逐位同构。
  //
  // 跨仓只读；若 prism-rt 源码不在（裁剪构建），判 FAIL 而非静默跳过——
  // 与本文件设计准则 3「不猜」一致。
  const cpuRsPath = resolve(SHADER_DIR, "../../prism-rt/src/hair_scatter.rs");
  if (!existsSync(cpuRsPath)) {
    checks.push({ name: "F16 CPU hair_scatter.rs 可读（跨语言同构断言前提）", ok: false, detail: `未找到 ${cpuRsPath}` });
  } else {
    const cpuRaw = readFileSync(cpuRsPath, "utf8");
    const cpu = stripRustComments(cpuRaw);
    // 只取 `m_longitudinal_full`（**对拍基准路径**）的函数体。
    // 【为什么必须收窄到函数体】同文件里的单变量退化形 `m_longitudinal`
    // （保留仅为向后兼容早期调用点）仍写 `2.0 * beta * beta`。它不参与
    // CPU/GPU 对拍（无 GPU 镜像），若全文匹配就会恒红——那就是"假门禁"，
    // 逼人加例外豁免。收窄后判据精确指向真正需要逐位同构的那一处。
    const cpuMFull = (cpu.split("fn m_longitudinal_full(")[1] || "").split("\n    pub fn ")[0].split("\n    fn ")[0];
    const mFullOk = /let\s+bm2\s*=\s*beta\s*\*\s*beta/.test(cpuMFull)
      && /1\.0\s*\/\s*\(\s*2\.0\s*\*\s*bm2\s*\)/.test(cpuMFull);
    checks.push({
      name: "F16 CPU m_longitudinal_full 2.0 * bm2（先平方再乘 2，与 WGSL 同结合序）",
      ok: mFullOk,
      detail: mFullOk ? "" : "m_longitudinal_full 未用 2.0 * bm2 —— 结合序 (2β)β 与 GPU 的 2(β²) 差 1 ulp",
    });
    checks.push({
      name: "F16 CPU N 归一化分母 det_bessel_i0(k * 0.5)²（k/2，与 WGSL 同）",
      ok: /det_bessel_i0\s*\(\s*k\s*\*\s*0\.5\s*\)/.test(cpu),
      detail: "",
    });
    checks.push({
      name: "F16 CPU 三瓣 α 偏角 R 取负（0 => -a），TT/TRT 取正",
      ok: /0\s*=>\s*-a/.test(cpu) && /1\s*=>\s*a/.test(cpu) && /2\s*=>\s*a/.test(cpu),
      detail: "",
    });
    checks.push({
      name: "F16 CPU 三瓣 φ 偏角 TT 翻转 π（1 => phi - PI）",
      ok: /1\s*=>\s*phi\s*-\s*PI/.test(cpu),
      detail: "",
    });
    checks.push({
      name: "F16 CPU I₀ 大参数段 3.75 / ax 后连乘 t（与 WGSL 同式）",
      ok: /let\s+t\s*=\s*3\.75\s*\/\s*ax/.test(cpu),
      detail: "",
    });
    checks.push({
      name: "F16 CPU hair_scatter.rs 亦声明 B 瓣未实现（两侧诚实边界一致）",
      ok: /B\s*瓣/.test(cpuRaw) && /未实现|尚未实现/.test(cpuRaw),
      detail: "",
    });
  }

  // ---- F18 无入口库件必须登记进真实编译链路（Task B 的静态一半）----
  //
  // 【为什么这条规则是 Task B 的另一半，而不是"重复劳动"】
  // 真实编译链路（`wgsl_lib_compile.mjs`）负责"登记在册的库件确实能编译"。
  // 但它**管不了没登记的库件**——清单漏一份，就等于给那份库件开一个新的编译盲区，
  // 而且是**静默**的：脚本照样全绿，只是绿得没有覆盖到它。这正是上一波事故的
  // 形态（LIBRARIES 里的 hair_scatter 从没被编过，却一路绿灯）。
  //
  // 故 F18 的判据取**扫目录所得的无入口库件集合为真值**，反向比对清单：
  //   * 目录里有、清单里没有 → 红（新增库件忘了登记 = 又开一个盲区）
  //   * 清单里有、目录里没有 → 红（清单登记了不存在的文件 = 门禁在骗人）
  //   * 清单项缺合成入口配方 / 缺注入自证 → 红（登记了但没真编，等于没登记）
  //
  // 【为什么必须比对 RECIPE_FILES / INJECTION_FILES 两个集合，而不只是清单】
  // 清单是"意图"，配方/注入是"落实"。三者任一不等，就说明有库件**名义上被管住、
  // 实质上没被编**——那比完全不管更坏，因为它给人虚假的安全感。
  const libManifest = ctx && ctx.libManifest !== undefined ? ctx.libManifest : LIB_MANIFEST;
  // 自证用例需要注入"被删改的清单"，故本规则的所有集合运算都只依赖入参。
  const onDisk = [];
  for (const f of readdirSync(SHADER_DIR).filter((n) => n.endsWith(".wgsl")).sort()) {
    const text = stripWgslComments(readFileSync(join(SHADER_DIR, f), "utf8"));
    // 判据与 shaders.rs 的 LIBRARIES/INCLUDE_ONLY 划分同口径：无任何入口即库件。
    if (!/@(compute|vertex|fragment)\b/.test(text)) onDisk.push(f);
  }
  const inManifest = libManifestFiles(libManifest);
  const missing = onDisk.filter((f) => !inManifest.includes(f));
  const ghost = inManifest.filter((f) => !onDisk.includes(f));
  checks.push({
    name: `F18 无入口 WGSL 库件全部登记进真编译清单（目录 ${onDisk.length} 份 / 清单 ${inManifest.length} 份）`,
    ok: missing.length === 0,
    detail: missing.length ? `未登记（= 编译盲区）：${missing.join(", ")}` : "",
  });
  checks.push({
    name: "F18 真编译清单无幽灵条目（清单里的文件必须真实存在于 shaders/）",
    ok: ghost.length === 0,
    detail: ghost.length ? `清单登记了不存在的文件：${ghost.join(", ")}` : "",
  });

  // 清单项字段契约：缺字段会让真编译链路在运行时才炸（而不是门禁里炸），
  // 那时人已经改完代码准备提交了——门禁必须抢在前面。
  const badItems = [];
  for (const it of libManifest) {
    if (!it.file) { badItems.push("某项缺 file"); continue; }
    if (typeof it.entry !== "string" || it.entry.length === 0) badItems.push(`${it.file} 缺 entry`);
    if (!Array.isArray(it.groups) || it.groups.length === 0) badItems.push(`${it.file} 缺 groups`);
    else {
      const g = it.groups;
      const sorted = [...g].sort((a, b) => a - b);
      if (new Set(g).size !== g.length) badItems.push(`${it.file} 的 groups 有重复`);
      else if (JSON.stringify(g) !== JSON.stringify(sorted)) badItems.push(`${it.file} 的 groups 未升序`);
    }
    if (typeof it.note !== "string" || it.note.trim().length === 0) badItems.push(`${it.file} 缺 note`);
  }
  checks.push({
    name: "F18 清单项字段齐备（file/entry/groups/note；groups 升序去重）",
    ok: badItems.length === 0,
    detail: badItems.length ? badItems.join("；") : `${libManifest.length} 项`,
  });

  // 落实层比对：清单 → 配方（合成入口）→ 注入自证。
  const noRecipe = inManifest.filter((f) => !RECIPE_FILES.includes(f));
  const noInject = inManifest.filter((f) => !INJECTION_FILES.includes(f));
  checks.push({
    name: "F18 清单内每份库件都有合成 compute 入口配方（否则从未真编译）",
    ok: noRecipe.length === 0,
    detail: noRecipe.length ? `缺配方：${noRecipe.join(", ")}` : `${RECIPE_FILES.length} 份`,
  });
  checks.push({
    name: "F18 清单内每份库件都有注入自证（否则无法证明链路对它有效）",
    ok: noInject.length === 0,
    detail: noInject.length ? `缺注入自证：${noInject.join(", ")}` : `${INJECTION_FILES.length} 份`,
  });
  // 反向：配方/注入里出现清单外的文件 = 有人加了库件却没登记进清单真源。
  const recipeGhost = RECIPE_FILES.filter((f) => !inManifest.includes(f));
  const injectGhost = INJECTION_FILES.filter((f) => !inManifest.includes(f));
  checks.push({
    name: "F18 配方/注入无清单外条目（配方与清单必须同一真源，不得各写一份）",
    ok: recipeGhost.length === 0 && injectGhost.length === 0,
    detail: [...recipeGhost, ...injectGhost].length
      ? `清单外：${[...new Set([...recipeGhost, ...injectGhost])].join(", ")}`
      : "",
  });

  // ---- F19 地址空间 ↔ 缓冲用法一致性（本路的核心：静态化本次真机事故）----
  // ---- F20 //!include 依赖图无环 + 引用存在性（静态化"静默丢依赖"）----
  return [...checks, ...ruleF19(ctx), ...ruleF20(ctx)];
}

/**
 * 规则 F19：WGSL 绑定**地址空间** ↔ 宿主配方**缓冲用法**的一致性。
 *
 * # 为什么存在（本路直接教训）
 *
 * v2.8 真机事故：`rt_scene.wgsl:70` 声明
 *   `@group(2) @binding(0) var<uniform> scene_params: SceneParams;`
 * 而合成入口配方却按 **storage** 建缓冲（缺 `uniform: true`）。后果链：
 *   1. `createBindGroup` 报 "Binding usage (...Storage) doesn't match expected
 *      usage (Uniform)" → group 2 变 **Invalid BindGroup**；
 *   2. `SetBindGroup(2, …)` 失效 → `submit` 的 command buffer 非法；
 *   3. **`onSubmittedWorkDone()` 在 submit 无效时仍会 resolve**，回读拿到的是
 *      未被 dispatch 覆写的**初始零值**；
 *   4. 于是「输出全为有限数」这条断言**看起来在测东西，实际测空转**。
 *
 * 第 1 步**只有真机能抓**（这正是事故的暴露方式）。F19 把它静态化：把
 * 「WGSL 声明了什么地址空间」与「配方按什么用法建缓冲」两份数据摆在一起比。
 *
 * # 判据（对每份登记在册的库件）
 *
 *   1. 取该库件**展开 `//!include` 后**的源码 + 合成入口源码 = 真编译的声明集；
 *      逐个 `@group(G) @binding(B) var<ADDR> name: T;` 提取 (G,B,ADDR,name)。
 *      ——【为什么必须展开】`rt_shade.wgsl` 经 `//!include rt_scene.wgsl` 带入
 *        同一处 `var<uniform> scene_params`；只扫单文件会完全漏掉它。
 *   2. 取配方的 (G,B)→用法（`uniform:true` ⇒ UNIFORM，否则 STORAGE）。
 *   3. 逐条断言：
 *      * `var<uniform>`  ⇒ 配方必须是 UNIFORM（否则红 —— 事故 II 本形）
 *      * `var<storage…>` ⇒ 配方必须是 STORAGE（否则红）
 *      * 声明了但配方没有 ⇒ 红（漏绑定 → 真机 createBindGroup 缺 entry）
 *      * 配方有但展开后没声明 ⇒ 红（幽灵绑定 → 配方在自说自话）
 *
 * @param {object} ctx 规则上下文。`ctx.recipeBindings` 可注入被改坏的配方
 *                     （自证非恒真用）；缺省取 `RECIPE_BINDINGS` 单一真源。
 * @returns {Array<{name:string, ok:boolean, detail:string}>}
 */
export function ruleF19(ctx) {
  const checks = [];
  const libManifest = ctx && ctx.libManifest !== undefined ? ctx.libManifest : LIB_MANIFEST;
  const recipes = ctx && ctx.recipeBindings !== undefined ? ctx.recipeBindings : RECIPE_BINDINGS;

  // 提取一份（已展开的）WGSL 源里全部 `@group @binding var<space,…> name` 声明。
  // 地址空间可带访问模式（`var<storage, read_write>`），故 `var<` 后不写死。
  const declsOf = (src) => {
    const out = [];
    for (const m of src.matchAll(
      /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*<\s*([^>]+?)\s*>\s*(\w+)/g,
    )) {
      out.push({ group: Number(m[1]), binding: Number(m[2]), space: m[3].trim(), name: m[4] });
    }
    return out;
  };
  /** WGSL 地址空间 → 期望的缓冲用法（`uniform` / `storage`）。 */
  const expectedUsage = (space) => (/^uniform\b/.test(space) ? "uniform"
    : /^storage\b/.test(space) ? "storage" : null);
  const key = (g, b) => `${g}:${b}`;

  let compared = 0;              // 实际比过的绑定数（用于"非空集"断言）
  const problems = [];           // 逐条问题（detail 串）

  for (const item of libManifest) {
    const file = item.file;
    const recipe = recipes[file];
    if (!recipe) {
      // 配方缺失本应由 F18 抓；F19 不重复判红，但也不静默——记一条可判红项。
      checks.push({
        name: `F19 ${file}：配方存在（否则无法比对绑定用法）`,
        ok: false,
        detail: "RECIPE_BINDINGS 缺该库件",
      });
      continue;
    }
    // 【判据 = 展开后的声明集】库件本体（include 已展开）+ 合成入口源码，
    // 二者共同构成真编译的那份单体源码（与 wgsl_lib_compile 的 buildLibShader 同口径）。
    const expanded = expandShaderIncludes(file);
    const full = `${expanded}\n${recipe.extraWgsl || ""}`;
    const decls = [...new Set([
      ...declsOf(expanded),   // 库件本体（含 include 带入的声明）
      ...declsOf(recipe.extraWgsl || ""), // 合成入口自带的声明
    ].map((d) => `${key(d.group, d.binding)}|${d.space}|${d.name}`))]
      .map((s) => {
        const [kb, space, name] = s.split("|");
        const [g, b] = kb.split(":");
        return { group: Number(g), binding: Number(b), space, name };
      });

    const declByKb = new Map(decls.map((d) => [key(d.group, d.binding), d]));
    const recipeByKb = new Map(
      (recipe.buffers || []).map((b) => [key(b.group, b.binding), b]),
    );

    // (a) 声明 → 配方：用法必须一致；不得漏绑定。
    for (const d of decls) {
      const ent = recipeByKb.get(key(d.group, d.binding));
      if (!ent) {
        problems.push(`${file}: 声明 @group(${d.group}) @binding(${d.binding}) var<${d.space}> ${d.name} 但配方里没有它（漏绑定 → 真机 createBindGroup 缺 entry）`);
        continue;
      }
      const want = expectedUsage(d.space);
      if (want === null) {
        problems.push(`${file}: @group(${d.group}) @binding(${d.binding}) ${d.name} 的地址空间 <${d.space}> 无法判定用法（门禁盲区，须扩规则）`);
        continue;
      }
      const got = ent.uniform ? "uniform" : "storage";
      compared += 1;
      if (want !== got) {
        // 【事故 II 本形】uniform 声明却按 storage 建 → 绑定失效 → submit 无效 → 假绿。
        problems.push(`${file}: @group(${d.group}) @binding(${d.binding}) ${d.name} 声明为 var<${d.space}>（需 ${want.toUpperCase()}），但配方按 ${got.toUpperCase()} 建缓冲 —— createBindGroup 会失败，submit 无效后回读零值即"假绿"`);
      }
    }

    // (b) 配方 → 声明：不得有幽灵绑定。
    for (const b of recipe.buffers || []) {
      if (!declByKb.has(key(b.group, b.binding))) {
        problems.push(`${file}: 配方为 @group(${b.group}) @binding(${b.binding}) 建了缓冲，但展开后的 WGSL 未声明该绑定（幽灵绑定 → 配方在自说自话）`);
      }
    }
  }

  checks.push({
    name: `F19 各库件「地址空间 ↔ 缓冲用法」逐绑定一致（含 //!include 展开；比对 ${compared} 个绑定）`,
    ok: problems.length === 0,
    detail: problems.length ? problems.slice(0, 4).join("；") + (problems.length > 4 ? ` …（共 ${problems.length} 条）` : "") : `${libManifest.length} 份库件`,
  });
  // 非空集断言：F19 若一条都没比，就是恒真门禁（本项目最忌讳的形态）。
  checks.push({
    name: "F19 至少比对到 1 个绑定（防空集恒真）",
    ok: compared > 0,
    detail: `${compared} 个`,
  });
  return checks;
}

// ---------------------------------------------------------------------------
// 规则 F20：`//!include` 依赖图**无环** + 引用文件**真实存在**。
//
// # 为什么存在（本路发现的第三类"静默"缺陷）
//
// 事故 I 的形态是"库件从不进入真编译链路，却一路绿灯"。F18/F19 把那条链路
// 静态化了，但**依赖图本身**仍无人看守，而两处展开器对它的处理都是**静默**的：
//
//   * `wgsl_lint.mjs::expandShaderIncludes` —— `if (!existsSync(p)) return "";`
//   * `shaders.rs::expand_into` —— `let Some(src) = source(name) else { return; };`
//
// 也就是说：**把 `//!include rt_common.wgsl` 改成 `//!include rt_commonn.wgsl`
// （少一个字母），或让 A include B、B include A**，两处展开器都会安安静静地
// 少拼一段源码，编译照过、门禁照绿，而真机拿到的是一份**缺声明**的模块 ——
// 或者一段永远展开不到头的include 链。
//
// 为什么这在真机上**未必**立刻炸：缺声明时报的是 `unresolved_symbols`（规则 A 的
// 符号可解析性），而它只在"缺的符号确实被调用"时才报；若被丢掉的 include 贡献的
// 是常量、struct 或当前路径未用到的函数，**连符号检查都抓不到**。这正是"断言与
// 真实执行脱节"的第四种形态：门禁全绿，而拼出来的模块少了一整块。
//
// # 判据
//
//   1. **引用存在性**：每条 `//!include <f>` 的 `<f>` 必须真实存在于 `shaders/`。
//   2. **无环**：依赖图做三色 DFS，回边即红（环会让 `visited` 集合静默截断，
//      展开结果依赖遍历顺序——直接违反全链路确定性铁律）。
//   3. **指令格式合法**：必须独占一行且只含一个文件名（`expandShaderIncludes`
//      的正则要求整行匹配，写歪了会被当普通行原样拼进 WGSL 源码）。
//   4. **防空集恒真**：至少 1 份 shader、1 条 include 边，否则本规则等于没跑。
//
// @param {object} ctx 规则上下文。`ctx.shaderSrcs` 可注入一份
//                     `{文件名: 源码}`（自证"造环/删文件/改路径"用）；
//                     缺省读 `shaders/` 真实目录。
// @returns {Array<{name:string, ok:boolean, detail:string}>}
export function ruleF20(ctx) {
  const checks = [];
  // 注入模式（自证）：全集 = 注入对象的键。**刻意不回落读盘**——否则"删掉一份
  // 文件"这种反向用例会因为磁盘上还在而被判绿，自证就变成恒真。
  const injected = ctx && ctx.shaderSrcs !== undefined ? ctx.shaderSrcs : null;
  const readOne = (name) => {
    if (injected) return Object.prototype.hasOwnProperty.call(injected, name) ? injected[name] : null;
    const p = join(SHADER_DIR, name);
    return existsSync(p) ? readFileSync(p, "utf8") : null;
  };
  const allNames = (injected
    ? Object.keys(injected)
    : readdirSync(SHADER_DIR)
  ).filter((n) => n.endsWith(".wgsl")).sort();

  // ---- 解析 include 边（口径与 expandShaderIncludes 的正则逐字一致）----
  const edges = new Map();   // 文件名 → 它 include 的文件名列表
  const badFormat = [];      // 以 //!include 开头却不匹配严格式的行
  for (const name of allNames) {
    const src = readOne(name);
    if (src === null) continue;
    const deps = [];
    const lines = src.split("\n");
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const strict = /^\s*\/\/!include\s+(\S+)\s*$/.exec(line);
      if (strict) { deps.push(strict[1]); continue; }
      if (/^\s*\/\/!include/.test(line)) badFormat.push(`${name}:${i + 1} \`${line.trim()}\``);
    }
    edges.set(name, deps);
  }
  let edgeCount = 0;
  for (const deps of edges.values()) edgeCount += deps.length;

  // ---- 判据 1：引用存在性 ----
  const known = new Set(allNames);
  const dangling = [];
  for (const [name, deps] of edges) {
    for (const d of deps) {
      if (!known.has(d)) dangling.push(`${name} → ${d}`);
    }
  }
  checks.push({
    name: `F20 每条 //!include 引用的文件真实存在（共 ${edgeCount} 条边；悬空引用会让展开器静默少拼一段源码）`,
    ok: dangling.length === 0,
    detail: dangling.length ? `悬空：${dangling.join("，")}` : `${allNames.length} 份 shader`,
  });

  // ---- 判据 2：依赖图无环（三色 DFS，回边即环）----
  //
  // 为什么环必须静态拦：两处展开器都靠 `visited` 集合截断环，展开结果因此
  // **依赖遍历顺序**——同一份源码在改动 include 顺序后可能拼出不同模块。
  // 这不是"能跑就行"，而是直接违反全链路确定性铁律，且真机报错往往指向
  // 一个与环毫无关系的符号，排查成本极高。
  const WHITE = 0, GRAY = 1, BLACK = 2;
  const color = new Map();
  const path = [];
  const cycles = [];
  const dfs = (n) => {
    color.set(n, GRAY);
    path.push(n);
    for (const d of edges.get(n) || []) {
      // 悬空引用已由判据 1 判红，此处跳过以免把同一个缺陷报成"环"，制造噪声。
      if (!edges.has(d)) continue;
      const c = color.get(d) || WHITE;
      if (c === GRAY) {
        cycles.push([...path.slice(path.indexOf(d)), d].join(" → "));
      } else if (c === WHITE) {
        dfs(d);
      }
    }
    path.pop();
    color.set(n, BLACK);
  };
  for (const n of allNames) {
    if ((color.get(n) || WHITE) === WHITE) dfs(n);
  }
  checks.push({
    name: "F20 //!include 依赖图无环（环会让 visited 静默截断，展开结果依赖遍历顺序）",
    ok: cycles.length === 0,
    detail: cycles.length ? `发现环：${cycles.slice(0, 3).join("；")}` : "",
  });

  // ---- 判据 3：指令格式合法 ----
  checks.push({
    name: "F20 //!include 指令格式合法（须独占一行、只含一个文件名）",
    ok: badFormat.length === 0,
    detail: badFormat.length ? `格式非法：${badFormat.slice(0, 3).join("；")}` : "",
  });

  // ---- 判据 4：防空集恒真 ----
  checks.push({
    name: "F20 非空集（至少 1 份 shader 且 1 条 include 边，否则本规则等于没跑）",
    ok: allNames.length > 0 && edgeCount > 0,
    detail: `${allNames.length} 份 / ${edgeCount} 条边`,
  });
  return checks;
}

// ---------------------------------------------------------------------------
//
// 定位：`shaders/gi_compute.wgsl` 是 GI 的 compute kernel。它的**数值契约**
// （SH π 约定）与**绑定契约**（layout:"auto" 三条硬约束）都是"错了也能编译过、
// 只有真机或对拍才发现"的隐性缺陷。规则 G 把它们固化成静态断言：
//   G1  三个 compute 入口在场（gi_probe_trace / gi_sh_project / gi_probe_interpolate）
//   G2  三个入口的静态绑定引用集合**各自是连续前缀** {0..n-1}（三条硬约束之二）
//   G3  SH 余弦瓣常数在场且正确（Â_0=π, Â_1=2π/3, Â_2=π/4）——π 约定锚点
//   G4  SH 基函数常数齐备（0.28209479 / 0.48860251 / 1.09254846 / 0.31539157 / 0.54627423）
//   G5  转辐亮度乘 1/π = 0.3183098861837907（Lambert BRDF 锚点）
//   G6  SDF sphere tracing 在场（fn sphere_trace_sdf + 命中阈值判定）
//   G7  单个 uniform（gi_params）且 16 字节对齐（由规则 A 兜底 size，这里断言只 1 个 uniform）
function ruleG(ctx) {
  const checks = [];
  const p = join(SHADER_DIR, "gi_compute.wgsl");
  // 自证用例可注入改写后的源码（负向验证必须能真的改内容）。
  const raw = ctx && ctx.giSrc !== undefined ? ctx.giSrc : (existsSync(p) ? readFileSync(p, "utf8") : null);
  if (raw === null) {
    checks.push({ name: "G1 gi_compute.wgsl 存在", ok: false, detail: "未找到文件" });
    return checks;
  }
  const src = stripWgslComments(raw);
  const eff = stripWgslComments(expandShaderIncludes("gi_compute.wgsl"));

  // ---- G1 三个入口 ----
  const entries = ["gi_probe_trace", "gi_sh_project", "gi_probe_interpolate"];
  const missing = entries.filter((e) => !new RegExp(`fn\\s+${e}\\s*\\(`).test(src));
  checks.push({ name: "G1 gi_compute.wgsl 三个 compute 入口齐备", ok: missing.length === 0, detail: missing.length ? `缺 ${missing.join(",")}` : `${entries.length} 个` });

  // ---- G2 三条 layout:"auto" 约束之二：每入口绑定引用集是连续前缀 ----
  // 解析生效源里所有 `@group(g) @binding(b) var<...> name` 声明（取 binding 号与变量名）。
  // 地址空间可能带访问模式（`var<storage, read_write>`），故 `var<` 后内容不写死。
  const decls = [];
  for (const m of eff.matchAll(/@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*<[^>]+>\s*(\w+)/g)) {
    decls.push({ group: Number(m[1]), binding: Number(m[2]), name: m[3] });
  }
  checks.push({ name: "G2 gi_compute 的 binding 声明可解析", ok: decls.length >= 3, detail: `${decls.length} 个` });
  for (const e of entries) {
    // 找该入口的函数体（花括号配对）。
    const head = new RegExp(`fn\\s+${e}\\s*\\([^)]*\\)[^{]*\\{`).exec(src);
    if (!head) { checks.push({ name: `G2 ${e} 函数体可解析`, ok: false, detail: "未找到" }); continue; }
    let i = src.indexOf("{", head.index), depth = 0, end = src.length;
    for (let j = i; j < src.length; j++) {
      if (src[j] === "{") depth++;
      else if (src[j] === "}") { depth--; if (depth === 0) { end = j; break; } }
    }
    const body = src.slice(head.index, end + 1);
    // 该入口引用的 binding（出现在函数体内的声明名）。
    const used = decls
      .filter((d) => d.group === 0 && new RegExp(`\\b${d.name}\\b`).test(body))
      .map((d) => d.binding)
      .sort((a, b) => a - b);
    // 连续前缀判定：去重后必须等于 [0,1,...,n-1]。
    const uniq = [...new Set(used)];
    const isPrefix = uniq.length > 0 && uniq.every((b, k) => b === k);
    checks.push({
      name: `G2 ${e} 的 group(0) 绑定引用集是连续前缀 {0..${uniq.length - 1}}`,
      ok: isPrefix,
      detail: `引用 binding = [${uniq.join(",")}]`,
    });
  }

  // ---- G3 SH 余弦瓣常数（π 约定锚点）----
  const lobe = [
    ["Â_0 = π", /3\.141592653589793/],
    ["Â_1 = 2π/3", /2\.0943951023931953/],
    ["Â_2 = π/4", /0\.7853981633974483/],
  ];
  for (const [label, re] of lobe) {
    checks.push({ name: `G3 SH 余弦瓣常数在场：${label}`, ok: re.test(src), detail: "" });
  }

  // ---- G4 SH 基函数常数（对齐 sh.rs::sh_basis_l2）----
  for (const c of ["0.28209479", "0.48860251", "1.09254846", "0.31539157", "0.54627423"]) {
    checks.push({ name: `G4 SH 基函数常数在场：${c}`, ok: src.includes(c), detail: "" });
  }

  // ---- G5 Lambert 1/π（对齐 sh.rs::sh_irradiance_to_radiance 的 `1.0 / PI`）----
  checks.push({ name: "G5 转辐亮度乘 1/π（1.0 / 3.141592653589793）", ok: /1\.0\s*\/\s*3\.141592653589793/.test(src), detail: "" });

  // ---- G6 sphere tracing ----
  checks.push({ name: "G6 SDF sphere tracing 函数在场（fn sphere_trace_sdf）", ok: /fn\s+sphere_trace_sdf\s*\(/.test(src), detail: "" });
  checks.push({ name: "G6 命中阈值判定在场（d <= hit_eps 语义）", ok: /hit_eps/.test(src), detail: "" });

  // ---- G7 只 1 个 uniform（多 uniform 逐个校验的"恰好一个"情形）----
  const uniforms = [...src.matchAll(/var\s*<\s*uniform\s*>/g)].length;
  checks.push({ name: "G7 gi_compute 只声明 1 个 uniform（gi_params）", ok: uniforms === 1, detail: `${uniforms} 个` });

  // ---- G8 禁 libm（D-29，兜底 check_shader 之外的同口径复核）----
  const libm = /[^"\w](sin|cos|tan|exp|pow|log|atan)\(/.test(src.replace(/quarter_/g, ""));
  checks.push({ name: "G8 gi_compute 无 libm 超越函数（D-29；三角走 cos_sin）", ok: !libm, detail: "" });

  return checks;
}

// ---------------------------------------------------------------------------
// 自证非恒真：故意违反的样例
// ---------------------------------------------------------------------------

/**
 * 每条规则配一个"最小违反样例"，把样例喂给规则后**必须**报错。
 * 若某条规则对自己的违反样例都 PASS，说明它是恒真的，等于没有门禁。
 */
export function runWgslLintSelfTest() {
  const results = [];
  const glueOk = `const glue = {
  prism_navigator_gpu_available: () => (navigator.gpu ? 1 : 0),
  prism_request_device(){ const t = 1; return t; },
  prism_device_lost: h => 0,
  prism_device_ready: h => 0,
  prism_create_buffer: (h,s,u) => 0,
  prism_write_buffer: (h,b,p,l) => 1,
  prism_create_shader_module: (h,p,l) => 0,
  prism_create_compute_pipeline: (h,m,p,l) => 0,
  prism_create_bind_group: (h,p,ptr,n) => 0,
  prism_dispatch: (h,p,b,x,y,z) => 1,
  prism_read_buffer: (h,b,s) => 0,
  prism_copy_out: (s,d,l) => 1,
  prism_wait_idle: h => 1,
  prism_pop_error: () => 0,
};`;
  const htmlBase = (glueSrc, extra = "") => `<script type="module">
${glueSrc}
const BYTES_PER_U32 = 4;
const WASM_URL = "prism_gpu.wasm";
function renderReport(){
  const abi = (wasm && typeof wasm.prism_abi_version === "function") ? wasm.prism_abi_version() : "n/a";
  return "wasm: " + abi;
}
function renderEnv(){
  const v = (wasm && typeof wasm.prism_abi_version === "function") ? wasm.prism_abi_version() : "-";
  return v;
}
function renderErrors(){ return 0; }
${extra}
</script>`;
  const bgOk = `const glue = {
  prism_create_bind_group: (h, pipe, bufs_ptr, n) => {
    const mem = new Uint8Array(wasm.memory.buffer);
    const view = new DataView(mem.buffer, bufs_ptr, n * BYTES_PER_U32);
    const entries = [];
    for (let i = 0; i < n; i += 1) {
      entries.push({ binding: i, resource: { buffer: get(view.getUint32(i * BYTES_PER_U32, true)) } });
    }
    return 0;
  },
};`;

  const cases = [
    {
      rule: "A",
      desc: "宿主 uniform 缓冲尺寸既非 struct size 也非 payload（两端各说各话）",
      expectFail: "A3",
      run: () => {
        // 真的把 device.rs 的建缓冲常量改成一个"能装下但不对齐声明"的尺寸（16→24）：
        // 既不是 struct size（16）也不是 payload（16），WebGPU 会读错布局。
        const dev = readFileSync(DEVICE_RS, "utf8").replace(
          "pub const SCAN_PARAMS_BYTES: usize = 16;",
          "pub const SCAN_PARAMS_BYTES: usize = 24;");
        return ruleA({ wgsl: collectWgslSources(), html: "", abiSrc: readFileSync(ABI_RS, "utf8"), devSrc: dev })
          .filter((c) => c.name.includes("A3 gpu.scan_block") && !c.ok);
      },
    },
    {
      rule: "A",
      desc: "宿主 uniform 缓冲退化为 8B（真机缺陷 1 原始形态）",
      expectFail: "A2",
      run: () => {
        // 真的把 abi.rs 的建缓冲实参改成 8：这是 D-14 真机失败的原始形态。
        const abi = readFileSync(ABI_RS, "utf8").replace("SCAN_PARAMS_BYTES as u32,", "8u32,");
        return ruleA({ wgsl: collectWgslSources(), html: "", abiSrc: abi, devSrc: readFileSync(DEVICE_RS, "utf8") })
          .filter((c) => c.name.includes("A2 gpu.scan_block") && !c.ok);
      },
    },
    {
      rule: "B",
      desc: "prism_request_device 改回 async",
      expectFail: "B2",
      run: () => {
        const bad = glueOk.replace("prism_request_device(){", "async prism_request_device(){");
        return ruleB({ html: htmlBase(bad), module: ctxModule() }).filter((c) => c.name.startsWith("B2") && !c.ok);
      },
    },
    {
      rule: "B",
      desc: "胶水里删掉一个 wasm 需要的 import",
      expectFail: "B3",
      run: () => {
        const bad = glueOk.replace(/  prism_pop_error: \(\) => 0,\n/, "");
        return ruleB({ html: htmlBase(bad), module: ctxModule() }).filter((c) => c.name.startsWith("B3") && !c.ok);
      },
    },
    {
      rule: "C",
      desc: "renderEnv 去掉 typeof 守卫（退回 wasm 真值判断）",
      expectFail: "C",
      run: () => {
        const bad = htmlBase(glueOk).replace(
          'const v = (wasm && typeof wasm.prism_abi_version === "function") ? wasm.prism_abi_version() : "-";',
          'const v = wasm ? wasm.prism_abi_version() : "-";');
        return ruleC({ html: bad }).filter((c) => !c.ok);
      },
    },
    {
      rule: "D",
      desc: "bind group 路径退回裸 4 字节字面量",
      expectFail: "D2",
      run: () => {
        const bad = bgOk.replace(/BYTES_PER_U32/g, "4");
        return ruleD({ html: htmlBase(glueOk + "\n" + bad) }).filter((c) => c.name.startsWith("D2") && !c.ok);
      },
    },
    {
      rule: "E",
      desc: "DataView 偏移双重计算（真机缺陷 2 原始形态）",
      expectFail: "E1",
      run: () => {
        const bad = bgOk.replace("new DataView(mem.buffer, bufs_ptr, n * BYTES_PER_U32)", "new DataView(mem.buffer, bufs_ptr + i * BYTES_PER_U32, BYTES_PER_U32)")
          .replace("for (let i = 0; i < n; i += 1) {", "for (let i = 0; i < n; i += 1) {\n      const view2 = new DataView(mem.buffer, bufs_ptr, n * BYTES_PER_U32);");
        return ruleE({ html: htmlBase(glueOk + "\n" + bad) }).filter((c) => !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 删掉 marschner_n（关键公式缺失）",
      expectFail: "F2",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("fn marschner_n(", "fn renamed_n(");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F2 ") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 引入循环（破坏确定性）",
      expectFail: "F2b",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("fn hair_scatter_total(s: vec3<f32>) -> f32 {",
          "fn hair_scatter_total(s: vec3<f32>) -> f32 { var acc = 0.0; for (var i = 0; i < 3; i = i + 1) { acc = acc + 0.0; }");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F2b") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 改错 I₀ 系数（A&S 9.8.1 首项 3.5156229→9.9）",
      expectFail: "F3",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("3.5156229", "9.9");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("3.5156229") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 打乱运算顺序（A 提到 M·N 之前）",
      expectFail: "F6",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 把三瓣改成「先算 A，再乘 M·N」——顺序反转仍能编译，但确定性契约被破坏。
        //
        // 【隐患备忘】`let mn = m * n;` 在源码中出现 **2 次**，且第一处在
        // `hair_scatter_s_precise` 的文档注释里（解释为何要写 `let mn = m * n;`）。
        // 故此处 `replace` 只改到注释、不改真实代码；本用例之所以仍能变红，
        // 是**依赖第二个替换**（`let s_r = mn * a_r(f0);` → `let s_r = m * n;`）
        // 删掉了 F6 要求的 A 瓣字面量。改动本用例时务必两个替换一起看。
        const bad = orig
          .replace("let mn = m * n;", "let mn = a_r(f0) * a_tt(f0, absorb_tt);")
          .replace("let s_r = mn * a_r(f0);", "let s_r = m * n;");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F6") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 删掉 I₀ 的 A&S 9.8.2 系数（0.00392377）",
      expectFail: "F2c",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("0.00392377", "0.0");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F2c") && !c.ok);
      },
    },
    // =====================================================================
    // 「Marschner 收口」新增反向用例（Agent C）
    //
    // 【为什么必须逐条贴反向证据】本项目吃过两次"假门禁"的亏：
    //   1. GI 模块被 `TEMP-SCOPE-WORKAROUND-DISABLE-GI` 注释掉，却报告
    //      "149 passed" —— 目标模块根本没参与编译，通过数全是假的。
    //   2. 只写正向断言（"函数名在场"）的门禁，被 `hair_scatter.wgsl` 的
    //      实参/形参不匹配长期骗过 —— 该文件在 LIBRARIES 里不参与编译。
    // 所以每条新规则都必须有一个"故意违反 → 规则必须变红"的样例。
    // 下面 9 例分别针对 F7 / F8 / F9 / F10 / F11 / F12 / F14 / F15 / F16。
    // =====================================================================
    {
      rule: "F",
      desc: "hair_scatter 改用 WGSL 内建 exp() 替代 det_exp（D-29 违规）",
      expectFail: "F7",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 注入一个真正调用内建 exp 的函数（不是 det_exp）。
        const bad = "fn __libm_violation(x: f32) -> f32 { return exp(x); }\n" + orig;
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F7") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 注释里出现 sin(θ)（验证 F7 走 cleaned 去注释，不误报）",
      expectFail: "F7（应 PASS：假阳性反证）",
      run: () => {
        // 【反向的反向：假阳性反证】把一句 libm 写进**注释**里。
        // 正确实现（F7 走 stripWgslComments）必须**仍然全绿**。
        // 本项目历史教训：注释里的 `sin(θ)` 曾被误判为 libm 调用，
        // 逼着人给注释打补丁绕过门禁。这里把该教训固化成自动断言。
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = `${orig}\n// 说明：本行注释里的 sin(theta) / log(x) / pow(a,b) 均非调用。\n`;
        const failing = ruleF({ hairSrc: bad })
          .filter((c) => c.name.startsWith("F7") && !c.ok);
        return failing.length === 0 ? [{ name: "F7-cleaned-no-false-positive", ok: false, detail: "" }] : [];
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 把 a_r 形参改回 2 个但仍按 1 个实参调用（复现真实编译缺陷）",
      expectFail: "F8",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 这正是磁盘上发现的原始形态：`fn a_r(f0: f32, extra: f32)` 却
        // 仍写 `a_r(f0)` —— WGSL 编译期报错，但因该文件不参与 kernel 编译
        // 而长期绿灯。全绿的门禁（F1–F6）对此完全无感。
        const bad = orig.replace("fn a_r(f0: f32) -> f32 {", "fn a_r(f0: f32, absorb_extra: f32) -> f32 {");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F8") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 N 归一化退回 I₀(k)² 而非 I₀(k/2)²（真实缺陷，约 1430 倍）",
      expectFail: "F9",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig
          .replace("let kk = 0.5 * kh;", "let kk = kh;")
          .replace(/HAIR_TWO_PI\s*\*\s*bessel_i0\(kk\)\s*\*\s*bessel_i0\(kk\)/,
            "HAIR_TWO_PI * bessel_i0(kh) * bessel_i0(kh)");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F9 ") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 det_exp 退回 6 阶 Taylor（缺 1/5040，与 CPU 不同构）",
      expectFail: "F10",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("+ r * (1.0 / 5040.0)", "");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("1/7!") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 det_exp 删掉 x <= -87 下溢守卫（Taylor 会翻负）",
      expectFail: "F10",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("if (x <= -87.0) {\n    return 0.0;\n  }", "");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("下溢守卫") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 det_exp 改回 x * (1/ln2)（乘倒数，与 CPU 的 x / ln2 差 1 ulp）",
      expectFail: "F10",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("floor(x / LN_2 + 0.5)", "floor(x * (1.0 / LN_2) + 0.5)");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("除法归约") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 bessel_i0 用 inverseSqrt（快速近似，违反确定性铁律）",
      expectFail: "F11",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 【必须 replaceAll，不能用 replace】该字面量在第 198 行的**注释**里
        // 先于第 224 行的真实代码出现（注释在解释"为何不重结合乘法"）。
        // 单参数 `String.replace(string, ...)` 只替换**第一处** → 改到注释上，
        // 真实代码纹丝不动 → 规则对样例毫无反应 = 假门禁。
        // 这与本项目历史上"注释里的 sin(θ) 被误判为 libm"是同一类陷阱：
        // 注释与代码的耦合会双向出错（误报与漏报）。故此处一律 replaceAll。
        const bad = orig.replaceAll("det_exp(ax) / sqrt(ax)", "det_exp(ax) * inverseSqrt(ax)");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F11") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 TT 瓣 φ 偏角符号写反（用 φ 而非 φ − π，小角度下极难漏过）",
      expectFail: "F14",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 反例形态：TT 瓣不翻转 π（`p == 1u` 分支改成返回 phi）。
        // 后果：φ=0 的前向散射会落在 N 的**谷值**上，TT 能量暴涨数百倍。
        const bad = orig.replace("return phi - PI;", "return phi;");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F14") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 R 瓣 α 偏角取正（应为负，方向反了）",
      expectFail: "F14",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("  if (p == 0u) {\n    return -a;\n  }", "  if (p == 0u) {\n    return a;\n  }");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("α 偏角") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 留一个返回常量的 B 瓣桩函数 a_b（半成品混入交付）",
      expectFail: "F15",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        // 半成品桩：函数名在场（F2 查不到问题）、能编译（F8 也没问题），
        // 但返回恒值 → 物理上完全错误。F15 专门抓这一类。
        const bad = `${orig}\nfn a_b(kk: f32) -> f32 { return 0.5; }\n`;
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F15") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 把 a_tt 换成 return 0.0 恒值桩（三瓣少一瓣，能量丢失）",
      expectFail: "F15",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace(
          /fn a_tt\(f0: f32, absorb_tt: f32\) -> f32 \{[\s\S]*?\n\}/,
          "fn a_tt(f0: f32, absorb_tt: f32) -> f32 {\n  return 0.0;\n}");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.includes("恒值桩") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 TRT 吸收忘乘 2（d_trt = 2·d_tt，能量偏高）",
      expectFail: "F12",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("det_exp_neg(sigma_a * (2.0 * d_tt))", "det_exp_neg(sigma_a * d_tt)");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F12") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "hair_scatter 的 Schlick 五次方写成四次方（掠射反射率偏差 > 10%）",
      expectFail: "F13",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "hair_scatter.wgsl"), "utf8");
        const bad = orig.replace("let x5 = x2 * x2 * x;", "let x5 = x2 * x2;");
        return ruleF({ hairSrc: bad }).filter((c) => c.name.startsWith("F13") && !c.ok);
      },
    },
    {
      rule: "G",
      desc: "gi_compute 删掉 gi_sh_project（关键 kernel 缺失）",
      expectFail: "G1",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "gi_compute.wgsl"), "utf8");
        const bad = orig.replace("fn gi_sh_project(", "fn renamed_project(");
        return ruleG({ giSrc: bad }).filter((c) => c.name.startsWith("G1") && !c.ok);
      },
    },
    {
      rule: "G",
      desc: "gi_compute 改错 SH 余弦瓣 Â_2（π/4 → 0.9）",
      expectFail: "G3",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "gi_compute.wgsl"), "utf8");
        const bad = orig.replace("0.7853981633974483", "0.9");
        return ruleG({ giSrc: bad }).filter((c) => c.name.includes("π/4") && !c.ok);
      },
    },
    {
      rule: "G",
      desc: "gi_compute 让入口引用 {0,2} 跳号（破坏连续前缀约束）",
      expectFail: "G2",
      run: () => {
        // 合成源：函数体引用 gi_probes(0) 与 gi_points(2)，但**不**引用 gi_params(1)
        // → 引用集 {0,2} 非连续前缀，必须被判红。
        const synth =
          "struct GiParams { probe_count: u32 }\n" +
          "struct GiProbe { pos: vec3<f32> }\n" +
          "struct GiPoint { pos: vec3<f32> }\n" +
          "@group(0) @binding(0) var<storage, read_write> gi_probes: array<GiProbe>;\n" +
          "@group(0) @binding(1) var<uniform> gi_params: GiParams;\n" +
          "@group(0) @binding(2) var<storage, read_write> gi_points: array<GiPoint>;\n" +
          "fn gi_probe_trace(x: u32) { let a = gi_probes[x].pos; _ = a; }\n" +
          "fn gi_sh_project(x: u32) { _ = gi_params.probe_count; _ = x; }\n" +
          "fn gi_probe_interpolate(x: u32) { let a = gi_probes[x].pos; let b = gi_points[x].pos; _ = a + b; }\n";
        return ruleG({ giSrc: synth })
          .filter((c) => c.name.includes("gi_probe_interpolate") && c.name.includes("连续前缀") && !c.ok);
      },
    },
    {
      rule: "G",
      desc: "gi_compute 引入 libm sin()（D-29 违规）",
      expectFail: "G8",
      run: () => {
        const orig = readFileSync(join(SHADER_DIR, "gi_compute.wgsl"), "utf8");
        const injected = "fn __bad(x: f32) -> f32 { return sin(x); }\n" + orig;
        return ruleG({ giSrc: injected }).filter((c) => c.name.startsWith("G8") && !c.ok);
      },
    },
    // ---- F18 反向用例：把库件从真编译清单里删掉 → 必须判红 ----
    //
    // 【这条反向用例为什么是 Task B 的核心自证】F18 的正向形态是"6 份库件全部
    // 在册"，它**恒真**的风险极高：只要 F18 的集合运算写错了（比如两边都比同一个
    // 数组、或者干脆没比），它就会在"清单漏了 sim_common.wgsl"的情况下照样全绿。
    // 那正是上一波事故的形态——门禁全绿，而库件从没被编过。
    // 故必须真的把一份库件从清单里删掉，并断言 F18 **必须**报出"未登记"。
    {
      rule: "F",
      desc: "把 sim_common.wgsl 从真编译清单里删掉（新增库件忘登记 = 新开编译盲区）",
      expectFail: "F18",
      run: () => {
        const trimmed = LIB_MANIFEST.filter((e) => e.file !== "sim_common.wgsl");
        // 注意"漏登记"这句话写在 detail 里（name 是固定的门禁描述），
        // 故这里同时看 name 与 detail，否则过滤器会永远匹配不上——那等于
        // 自证用例自己变成恒真，反而掩盖 F18 的真缺陷。
        return ruleF({ libManifest: trimmed })
          .filter((c) => c.name.includes("全部登记进真编译清单") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "把上一波事故主角 hair_scatter.wgsl 从真编译清单里删掉",
      expectFail: "F18",
      run: () => {
        const trimmed = LIB_MANIFEST.filter((e) => e.file !== "hair_scatter.wgsl");
        return ruleF({ libManifest: trimmed })
          .filter((c) => c.name.includes("全部登记进真编译清单") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "清单登记一个不存在的库件（幽灵条目 = 门禁在骗人）",
      expectFail: "F18",
      run: () => {
        const ghost = [...LIB_MANIFEST, { file: "no_such_lib.wgsl", entry: "lib_compile_main", groups: [0], note: "幽灵" }];
        return ruleF({ libManifest: ghost })
          .filter((c) => c.name.includes("F18") && c.name.includes("幽灵") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "清单项 groups 未升序（layout:\"auto\" 的连续前缀契约会被绕过）",
      expectFail: "F18",
      run: () => {
        const bad = LIB_MANIFEST.map((e) => (e.file === "rt_shade.wgsl" ? { ...e, groups: [2, 1, 0] } : e));
        return ruleF({ libManifest: bad })
          .filter((c) => c.name.includes("F18") && c.name.includes("升序") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "清单项缺 note（绑定占用说明丢失，下一任维护者无从判断该不该登记）",
      expectFail: "F18",
      run: () => {
        const bad = LIB_MANIFEST.map((e) => (e.file === "rt_bvh.wgsl" ? { ...e, note: "  " } : e));
        return ruleF({ libManifest: bad })
          .filter((c) => c.name.includes("F18") && c.name.includes("note") && !c.ok);
      },
    },
    // ---- F19 反向用例：地址空间 ↔ 缓冲用法一致性（本路事故 II 的静态复现）----
    //
    // 【这条为什么是本路的"核心自证"】F19 的正向形态是"6 份库件的绑定用法全一致"，
    // 它恒真的风险极高：只要 F19 的比对逻辑写错（比如两边都比配方、或根本没比），
    // 就会在"rt_scene 的 uniform 被按 storage 建"时照样全绿 —— 而那**正是** v2.8
    // 真机事故的形态。故必须真的把配方里的 `uniform: true` 删掉，断言 F19 报红。
    {
      rule: "F",
      desc: "把 rt_scene.wgsl 的 sceneParams 配方 uniform 标志删掉（模拟事故 II：uniform 声明却按 storage 建）",
      expectFail: "F19",
      run: () => {
        // 深拷一份配方，仅把 rt_scene 的 group2/binding0 的 uniform 置 false。
        const broken = {};
        for (const [f, r] of Object.entries(RECIPE_BINDINGS)) {
          broken[f] = {
            extraWgsl: r.extraWgsl,
            buffers: r.buffers.map((b) => ({ ...b })),
          };
        }
        broken["rt_scene.wgsl"].buffers = broken["rt_scene.wgsl"].buffers.map((b) =>
          (b.group === 2 && b.binding === 0 ? { ...b, uniform: false } : b));
        return ruleF({ recipeBindings: broken })
          .filter((c) => c.name.includes("F19") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "删掉 rt_shade.wgsl 经 //!include 带入的 scene_params 配方项（证明 F19 覆盖 include 展开的声明）",
      expectFail: "F19",
      run: () => {
        const broken = {};
        for (const [f, r] of Object.entries(RECIPE_BINDINGS)) {
          broken[f] = {
            extraWgsl: r.extraWgsl,
            buffers: r.buffers.map((b) => ({ ...b })),
          };
        }
        // rt_shade 的 group2/binding0 是由 rt_scene（include 带入）声明的；
        // 删掉配方项 → 漏绑定，必须判红（若 F19 只扫单文件，这条就会漏）。
        broken["rt_shade.wgsl"].buffers = broken["rt_shade.wgsl"].buffers
          .filter((b) => !(b.group === 2 && b.binding === 0));
        return ruleF({ recipeBindings: broken })
          .filter((c) => c.name.includes("F19") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "给某库件加一个 WGSL 未声明的幽灵配方项（配方在自说自话）",
      expectFail: "F19",
      run: () => {
        const broken = {};
        for (const [f, r] of Object.entries(RECIPE_BINDINGS)) {
          broken[f] = {
            extraWgsl: r.extraWgsl,
            buffers: r.buffers.map((b) => ({ ...b })),
          };
        }
        broken["hair_scatter.wgsl"].buffers = [
          ...broken["hair_scatter.wgsl"].buffers,
          { group: 0, binding: 7, uniform: false },
        ];
        return ruleF({ recipeBindings: broken })
          .filter((c) => c.name.includes("F19") && !c.ok);
      },
    },
    // ---- F20 反向用例：//!include 依赖图的悬空引用与环 ----
    //
    // 【为什么这批不可省】F20 的正向形态是"12 条 include 边、无悬空、无环"，
    // 它恒真的风险极高：若解析正则写错、或环检测根本没跑，它在**真出问题时
    // 照样全绿**。而这类缺陷（少一个字母的include / A↔B 互 include）恰恰是
    // 本项目事故 I 的同族形态——静默丢依赖、门禁全绿。故必须真的注入破坏。
    //
    // 用注入的 `ctx.shaderSrcs` 而非改磁盘：既不污染仓库，又能让"删掉一份
    // 文件"这类用例成立（注入模式下全集 = 注入对象的键，不回落读盘）。
    {
      rule: "F",
      desc: "把 //!include rt_common.wgsl 改成悬空路径 no_such_file.wgsl（事故 I 同族：静默丢依赖）",
      expectFail: "F20",
      run: () => {
        const srcs = {};
        for (const [k, v] of Object.entries(Object.fromEntries(
          readdirSync(SHADER_DIR)
            .filter((n) => n.endsWith(".wgsl"))
            .map((n) => [n, readFileSync(join(SHADER_DIR, n), "utf8")])))) srcs[k] = v;
        srcs["rt_scene.wgsl"] = srcs["rt_scene.wgsl"].replace(
          "//!include rt_common.wgsl", "//!include no_such_file.wgsl");
        return ruleF({ shaderSrcs: srcs })
          .filter((c) => c.name.includes("F20") && c.name.includes("真实存在") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "制造 include 环：rt_scene → hair_scatter → rt_scene（visited 会静默截断）",
      expectFail: "F20",
      run: () => {
        const srcs = {};
        for (const [k, v] of Object.entries(Object.fromEntries(
          readdirSync(SHADER_DIR)
            .filter((n) => n.endsWith(".wgsl"))
            .map((n) => [n, readFileSync(join(SHADER_DIR, n), "utf8")])))) srcs[k] = v;
        srcs["rt_scene.wgsl"] = srcs["rt_scene.wgsl"] + "\n//!include hair_scatter.wgsl\n";
        srcs["hair_scatter.wgsl"] = srcs["hair_scatter.wgsl"] + "\n//!include rt_scene.wgsl\n";
        return ruleF({ shaderSrcs: srcs })
          .filter((c) => c.name.includes("F20") && c.name.includes("无环") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "删掉一份被引用的库件（sim_cloth include sim_common，但清单里没有 sim_common）",
      expectFail: "F20",
      run: () => {
        const srcs = {};
        for (const [k, v] of Object.entries(Object.fromEntries(
          readdirSync(SHADER_DIR)
            .filter((n) => n.endsWith(".wgsl"))
            .map((n) => [n, readFileSync(join(SHADER_DIR, n), "utf8")])))) srcs[k] = v;
        delete srcs["sim_common.wgsl"];
        return ruleF({ shaderSrcs: srcs })
          .filter((c) => c.name.includes("F20") && c.name.includes("真实存在") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "把 //!include 写成一行两个文件名（格式非法的 include 会被当普通行拼进 WGSL）",
      expectFail: "F20",
      run: () => {
        const srcs = {};
        for (const [k, v] of Object.entries(Object.fromEntries(
          readdirSync(SHADER_DIR)
            .filter((n) => n.endsWith(".wgsl"))
            .map((n) => [n, readFileSync(join(SHADER_DIR, n), "utf8")])))) srcs[k] = v;
        srcs["rt_scene.wgsl"] = srcs["rt_scene.wgsl"].replace(
          "//!include rt_common.wgsl", "//!include rt_common.wgsl rt_shade.wgsl");
        return ruleF({ shaderSrcs: srcs })
          .filter((c) => c.name.includes("F20") && c.name.includes("格式合法") && !c.ok);
      },
    },
    {
      rule: "F",
      desc: "注入空 shader 集（防空集恒真：F20 必须自己判红，而不是因为没边就绿）",
      expectFail: "F20",
      run: () => ruleF({ shaderSrcs: {} })
        .filter((c) => c.name.includes("F20") && c.name.includes("非空集") && !c.ok),
    },
  ];

  for (const c of cases) {
    const caught = c.run();
    results.push({ rule: c.rule, desc: c.desc, ok: caught.length > 0, detail: caught.length ? `${caught.length} 条断言报错` : "规则对故意违反的样例毫无反应（恒真！）" });
  }
  return results;
}

let _mod = null;
function ctxModule() {
  if (!_mod) _mod = new WebAssembly.Module(readFileSync(join(HERE, "prism_gpu.wasm")));
  return _mod;
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------

const RULES = { A: ruleA, B: ruleB, C: ruleC, D: ruleD, E: ruleE, F: ruleF, G: ruleG };

/** 跑门禁。可只跑指定规则（`only: ["A"]`）。 */
export function runWgslLint({ only = null } = {}) {
  if (!existsSync(INDEX_HTML)) return { checks: [{ name: "门禁前置：index.html 存在", ok: false, detail: INDEX_HTML }] };
  const ctx = {
    html: readFileSync(INDEX_HTML, "utf8"),
    wgsl: collectWgslSources(),
    module: ctxModule(),
  };
  const names = only || Object.keys(RULES);
  const checks = [];
  for (const n of names) {
    if (!RULES[n]) { checks.push({ name: `规则 ${n} 存在`, ok: false, detail: "未知规则" }); continue; }
    for (const c of RULES[n](ctx)) checks.push({ name: `[${n}] ${c.name}`, ok: c.ok, detail: c.detail });
  }
  const failed = checks.filter((c) => !c.ok);
  return { checks, passed: checks.length - failed.length, failed: failed.length, total: checks.length };
}

// ---- CLI ----
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const args = process.argv.slice(2);
  if (args.includes("--self-test")) {
    console.log("== wgsl_lint 自证非恒真：故意违反的样例必须被抓==");
    const rs = runWgslLintSelfTest();
    let bad = 0;
    for (const r of rs) {
      console.log(`  ${r.ok ? "ok  " : "FAIL"}  规则 ${r.rule} ← ${r.desc}  — ${r.detail}`);
      if (!r.ok) bad += 1;
    }
    console.log(`\n自证 ${rs.length} 例，抓到 ${rs.length - bad} 例${bad ? `，${bad} 例规则恒真！` : "（全部规则均非恒真）"}`);
    process.exit(bad ? 1 : 0);
  }
  const only = (args.find((a) => a.startsWith("--rule=")) || "").slice("--rule=".length).split("").filter(Boolean);
  const r = runWgslLint({ only: only.length ? only : null });
  if (!args.includes("--json")) {
    console.log("== Prism WGSL/胶水静态门禁（D-14 真机缺陷本地化）==");
    for (const c of r.checks) console.log(`  ${c.ok ? "ok  " : "FAIL"}  ${c.name}${c.detail ? "  — " + c.detail : ""}`);
    console.log(`\n通过 ${r.passed}/${r.total}`);
  } else {
    console.log(JSON.stringify(r, null, 2));
  }
  if (r.failed) {
    console.error(`\n门禁失败：${r.failed} 项。修完再提交——这三类缺陷都是静态可查的，不该等真机。`);
    process.exit(1);
  }
  console.log("门禁全通过。");
}
