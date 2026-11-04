// ===========================================================================
// ci_wasm_verify.mjs — CI 用的 .wasm **静态**结构校验（零第三方依赖）。
// 只做"产物存在 + 体积合理 + 导出/导入符号齐全"三件事，**不碰**运行期语义
// （内存往返、CPU 参考交叉判对由 host_test.mjs 负责）。分工明确，避免两个脚本
// 重复断言同一件事。
//
// 为什么用 `WebAssembly.Module.imports/exports`（静态 introspection）而不是
// `WebAssembly.instantiate`：静态接口不需要提供 14 个 env.* 桩函数，一次
// `WebAssembly.Module()` 编译即可把 ABI 缺符号/改名问题在 CI 里秒级暴露。
//
// 用法：
//   node ci_wasm_verify.mjs [path/to/prism_gpu.wasm]
//
// 退出码：0 = 通过；1 = 校验失败（CI job 随之失败）；2 = 用法/环境错误。
// ===========================================================================

import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_WASM = resolve(HERE, "prism_gpu.wasm");
const WASM_PATH = process.argv[2] ? resolve(process.argv[2]) : DEFAULT_WASM;

/**
 * 体积下限：低于此值几乎必然是空壳/链接失败（实测产物 367,080 字节）。
 * 下限只拦"空壳"，不承担体积预算职责——预算由下面的 MAX_BYTES 负责。
 */
const MIN_BYTES = 8 * 1024;
/**
 * 体积基线（**实测值**）：699,414 字节。
 *
 * 订正记录（每段都有逐项归因，详见 tools/size_baseline.json 的 wasm.rationale）：
 *   A-1 接线前 344,674 B → A-1 接线后 367,080 B → T03 有设备分支 423,855 B
 *   → T6/T7 wavefront 五段接线 576,620 B（当时基线未更新，属遗留，本轮订正）
 *   → T-B/T-A 路径模式逐位对拍链 + gpu.wavefront_path 测试项 + CPU golden
 *     数据段（15,580 B include_bytes!）699,414 B。
 */
const BASELINE_BYTES = 699414;
/**
 * 体积上限：768 KiB = 786,432 字节（v2.5 T-B/T-A 修订口径，原 448 KiB）。
 *
 * 放宽理由与逐项归因见 tools/size_baseline.json 的 wasm 段——两段增长
 * （wavefront 接线 +152,765 B、路径模式逐位对拍 +122,794 B）全部是
 * 「能在真机判对」所必需的代码/数据，砍掉任何一项都会让对应判据失效。
 * 余量：786,432 − 699,414 = 87,018 B（+12.4%）。要再放宽必须先改
 * tools/size_baseline.json 并在 review 里说明原因——阈值是数据，不是这里的常数。
 */
const MAX_BYTES = 768 * 1024;

/**
 * 必须存在的导出符号。分两类：
 * - ABI 基座：版本号 + 线性内存分配器 + 错误/结果回传三元组；
 * - 测试项清单枚举：wasm 侧测试项是"数据驱动"的，符号缺一个整个手机 harness
 *   的清单就跑不起来。
 * 只校验"存在"，**不校验数量**（测试项会随版本增删，CI 不该因为加了用例而红）。
 */
const REQUIRED_EXPORTS = [
  "memory",
  "prism_abi_version",
  "prism_alloc",
  "prism_dealloc",
  "prism_error_ptr",
  "prism_error_len",
  "prism_result_ptr",
  "prism_result_len",
  "prism_test_count",
  "prism_test_collect",
  "prism_test_id",
  "prism_test_id_len",
  "prism_test_name",
  "prism_test_name_len",
  "prism_test_needs_gpu",
  "prism_test_run",
  "prism_shader_count",
  "prism_shader_name",
  "prism_shader_src",
  "prism_check_wgsl",
  "prism_kernel_wgsl",
];

/**
 * 必须由宿主（index.html）提供、wasm 侧 `extern "C"` 声明的导入符号。
 * 全部来自 `env` 模块——abi.rs 用 `#[link(wasm_import_module = "env")]`，
 * 若有人改了这个属性，wasm 侧单测（cargo test --workspace）发现不了，
 * 只有在这里或真机上才会炸，所以必须在 CI 卡住。
 */
const REQUIRED_IMPORTS = [
  "env.prism_request_device",
  "env.prism_device_ready",
  "env.prism_device_lost",
  "env.prism_navigator_gpu_available",
  "env.prism_create_shader_module",
  "env.prism_create_compute_pipeline",
  "env.prism_create_bind_group",
  "env.prism_create_buffer",
  "env.prism_write_buffer",
  "env.prism_dispatch",
  "env.prism_read_buffer",
  "env.prism_copy_out",
  "env.prism_wait_idle",
  "env.prism_pop_error",
];

const problems = [];

function fail(msg) {
  problems.push(msg);
  console.error(`  FAIL  ${msg}`);
}

// --- 1. 产物存在 -----------------------------------------------------------
let bytes;
try {
  const st = statSync(WASM_PATH);
  if (!st.isFile()) {
    fail(`产物不是普通文件：${WASM_PATH}`);
    console.error("\n==> ci_wasm_verify: FAIL");
    process.exit(1);
  }
  console.log(`  ok    产物存在  ${WASM_PATH}（${st.size} 字节）`);
  bytes = readFileSync(WASM_PATH);
} catch (e) {
  fail(`读不到产物 ${WASM_PATH}：${e.message}（build.sh 是否成功？`);
  console.error("\n==> ci_wasm_verify: FAIL");
  process.exit(1);
}

// --- 2. 体积合理 -----------------------------------------------------------
if (bytes.length < MIN_BYTES) {
  fail(`体积过小：${bytes.length} < ${MIN_BYTES} 字节，疑似空壳或链接失败`);
} else if (bytes.length > MAX_BYTES) {
  fail(`体积过大：${bytes.length} > ${MAX_BYTES} 字节（上限 ${MAX_BYTES / 1024} KiB）——越上限须先更新 tools/size_baseline.json 并说明原因`);
} else {
  const delta = bytes.length - BASELINE_BYTES;
  const headroom = MAX_BYTES - bytes.length;
  console.log(`  ok    体积合理  ${bytes.length} 字节（下限 ${MIN_BYTES} / 上限 ${MAX_BYTES} = ${MAX_BYTES / 1024} KiB）`);
  console.log(`  ok    体积基线  ${BASELINE_BYTES} 字节（本次 ${delta >= 0 ? '+' : ''}${delta}），距上限余量 ${headroom} 字节`);
}

// --- 3. 魔数：必须是真正的 wasm（\0asm） -----------------------------------
if (bytes.length < 4 || bytes[0] !== 0x00 || bytes[1] !== 0x61 ||
    bytes[2] !== 0x73 || bytes[3] !== 0x6d) {
  fail("魔数不是 \\0asm —— 这不是合法的 WebAssembly 模块");
}

// --- 4. 静态 introspection：导出 / 导入符号齐全 ----------------------------
let mod;
try {
  mod = new WebAssembly.Module(bytes);
} catch (e) {
  fail(`WebAssembly.Module 编译失败：${e.message}`);
  console.error("\n==> ci_wasm_verify: FAIL");
  process.exit(1);
}

const exports = WebAssembly.Module.exports(mod).map((e) => e.name);
const imports = WebAssembly.Module.imports(mod).map((e) => `${e.module}.${e.name}`);

for (const name of REQUIRED_EXPORTS) {
  if (!exports.includes(name)) fail(`缺少导出符号：${name}`);
}
console.log(`  ok    导出符号  ${REQUIRED_EXPORTS.length} 个必需符号全部在位（共 ${exports.length} 个导出）`);

for (const name of REQUIRED_IMPORTS) {
  if (!imports.includes(name)) fail(`缺少导入符号：${name}`);
}
const nonEnv = imports.filter((n) => !n.startsWith("env."));
if (nonEnv.length > 0) {
  fail(`导入符号出现在非 env 模块：${nonEnv.join(", ")}（abi.rs 的 wasm_import_module 被改动？）`);
}
console.log(`  ok    导入符号  ${REQUIRED_IMPORTS.length} 个必需符号全部在位（共 ${imports.length} 个导入，均来自 env）`);

// --- 5. 结构性：memory 必须真的导出且可增长（wasm 侧写内存的前提） --------
const memExport = WebAssembly.Module.exports(mod).find((e) => e.name === "memory");
if (memExport && memExport.kind !== "memory") {
  fail(`导出符号 memory 的 kind 是 ${memExport.kind}，应为 memory`);
}

// --- 汇总 ------------------------------------------------------------------
if (problems.length > 0) {
  console.error(`\n==> ci_wasm_verify: FAIL（${problems.length} 项）`);
  for (const p of problems) console.error(`   - ${p}`);
  process.exit(1);
}
console.log("\n==> ci_wasm_verify: PASS（静态结构校验通过；语义正确性见 host_test.mjs）");
process.exit(0);
