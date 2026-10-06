
// ===========================================================================
// host_test.mjs — 最小 WASM 宿主，验证 D-14 手机测试链路的 wasm 侧本身正确。
//
// 用途：在没有浏览器的环境（CI / 开发机）里证明——
//   1. `.wasm` 能被 `WebAssembly.instantiate` 加载（导出/导入签名齐全）；
//   2. JS 能把字节写进 wasm 线性内存、Rust 能读回（`prism_alloc` 往返）；
//   3. Rust 导出的**测试项清单**可枚举，语义正确；
//   4. Rust 侧 CPU 参考（`scan.rs` Blelloch / `sort.rs` radix）与**本文件里的
//      独立 JS 实现**逐元素一致——这是"wasm 侧数值正确"的交叉判对，
//      与 `cargo test --workspace` 的 Rust 内部自比互为独立证据。
//   5. 需要 GPU 的测试项在无 WebGPU 时稳定 SKIP（不 panic、不误报 FAIL）。
//
// 用法：
//   node crates/prism-gpu/web/host_test.mjs [path/to/prism_gpu.wasm]
// 默认路径指向 build.sh 的产物。零第三方依赖（只用 node 内置模块）。
// ===========================================================================

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_WASM = resolve(HERE, "../../../target/wasm32/wasm32-unknown-unknown/wasm/prism_gpu.wasm");
const WASM_PATH = process.argv[2] ? resolve(process.argv[2]) : DEFAULT_WASM;

// ---------------------------------------------------------------------------
// 极简断言 / 统计
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];

function check(name, cond, detail = "") {
  if (cond) {
    passed += 1;
    console.log(`  ok    ${name}${detail ? "  — " + detail : ""}`);
  } else {
    failures.push(name + (detail ? " — " + detail : ""));
    console.log(`  FAIL  ${name}${detail ? "  — " + detail : ""}`);
  }
}

function section(title) {
  console.log(`\n== ${title} ==`);
}

// ---------------------------------------------------------------------------
// JS 侧独立参考实现（与 Rust 的 scan.rs / sort.rs 语义对齐，但代码独立）
// ---------------------------------------------------------------------------

/** 串行 inclusive 前缀和（语义真值口径）。 */
function serialScan(input) {
  const out = new Uint32Array(input.length);
  let acc = 0;
  for (let i = 0; i < input.length; i += 1) {
    acc = (acc + input[i]) >>> 0;
    out[i] = acc;
  }
  return out;
}

/**
 * Blelloch 上下行双扫（补齐到 2 的幂），与 Rust `scan_blelloch_cpu` 同算法。
 * 返回 inclusive 前缀和（Rust 侧 `scan_blelloch_cpu` 的口径）。
 */
function blellochScan(input) {
  const n = input.length;
  if (n === 0) return new Uint32Array(0);
  let width = 1;
  while (width < n) width <<= 1;
  const tree = new Uint32Array(width);
  tree.set(input.subarray(0, n));
  // up-sweep
  for (let step = 1; step < width; step <<= 1) {
    for (let i = step * 2 - 1; i < width; i += step * 2) {
      tree[i] = (tree[i] + tree[i - step]) >>> 0;
    }
  }
  // down-sweep → ExScan
  tree[width - 1] = 0;
  for (let step = width >> 1; step > 0; step >>= 1) {
    for (let i = step * 2 - 1; i < width; i += step * 2) {
      const left = tree[i - step];
      const right = tree[i];
      tree[i - step] = right;
      tree[i] = (right + left) >>> 0;
    }
  }
  // ExScan + 原值 = InclScan
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = (tree[i] + input[i]) >>> 0;
  return out;
}

/** 与 Rust `abi.rs::lcg` 同参数：`(s * 6364136223846793005 + 1442695040888963407) >> 32`。 */
function makeLcg(seed) {
  let s = BigInt(seed);
  const M = (1n << 64n) - 1n;
  return () => {
    s = (s * 6364136223846793005n + 1442695040888963407n) & M;
    return Number(s >> 32n) >>> 0;
  };
}

/** 稳定计数排序单趟（与 Rust `counting_sort_pass` 同口径）。 */
function countingPass(items, shift, buckets) {
  const hist = new Uint32Array(buckets);
  for (const [k] of items) hist[(k >>> shift) & (buckets - 1)] += 1;
  // ExScan：桶起始偏移 = Σ_{d<b} hist[d]
  const offset = new Uint32Array(buckets);
  let acc = 0;
  for (let b = 0; b < buckets; b += 1) {
    offset[b] = acc;
    acc = (acc + hist[b]) >>> 0;
  }
  const out = new Array(items.length);
  const cursor = Uint32Array.from(offset);
  for (const [k, p] of items) {
    const d = (k >>> shift) & (buckets - 1);
    out[cursor[d]] = [k, p];
    cursor[d] += 1;
  }
  return out;
}

/** LSD radix sort（4 趟 × 8 位），与 Rust `radix_sort_u32` 同语义。 */
function radixSort(items) {
  let cur = items.map((x) => [x[0], x[1]]);
  for (let shift = 0; shift < 32; shift += 8) cur = countingPass(cur, shift, 256);
  return cur;
}

/** FNV-1a 32（与 Rust `abi.rs::fnv1a32` 同参数）。 */
function fnv1a32(bytes) {
  let h = 0x811c9dc5 >>> 0;
  for (const b of bytes) {
    h = (h ^ b) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** 指纹：对 u32 数组逐字节（wasm 与 native 均小端）。 */
function hashU32(arr) {
  const bytes = new Uint8Array(arr.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < arr.length; i += 1) view.setUint32(i * 4, arr[i] >>> 0, true);
  return fnv1a32(bytes);
}

/** f32 → u32 位模式（与 Rust `f32::to_bits` 口径一致；JS Number 先收窄到 f32）。 */
function f32Bits(x) {
  const buf = new ArrayBuffer(4);
  const v = new DataView(buf);
  v.setFloat32(0, Math.fround(x), true);
  return v.getUint32(0, true) >>> 0;
}

/** 扫 `key=value` 形式的结果文本，取某个数值。 */
function field(detail, key) {
  const m = new RegExp(key + "=([0-9a-fx]+)").exec(detail);
  return m ? m[1] : null;
}

// ---------------------------------------------------------------------------
// 真机 GI 指纹的「空转特征值」判定（本路加固：把"存在性"升级为"值域"）
// ---------------------------------------------------------------------------
//
// # 为什么必须加这一条（这是本路审计出的**真·弱断言**）
//
// 原断言是：
//     fp !== null && /^[0-9a-f]{8}$/.test(fp)     // "上报了指纹"
// 它只检查**格式合法**，恒可通过——`0x00000000` 也照样绿。而事故 II 的形态
// 恰恰是"submit 无效 → dispatch 没跑 → 回读拿到未被覆写的缓冲内容"。WebGPU
// 新建缓冲是**零初始化**的，于是"没跑"时的指纹就是一个**可计算出来的定值**：
//
//     全零输出指纹 = FNV-1a32( 3 探针 × 9 SH × RGB + 4 查询点 × RGB 个 u32 全 0 )
//                  = FNV-1a32(93 个 u32 全 0) = 0x06a083d5
//
// 于是"指纹等于全零特征值"就是**空转的直接证据**，必须判红。这条把断言从
// 「格式合法」升级为「值域排除」：真正跑过 dispatch 的 GPU 指纹绝不可能等于它。
//
// 【为什么这个判据不会误伤真机】真机实测指纹为 0x87ac3cf3（GPU 允许 FMA 收缩，
// 与 CPU 锚0xaf8429a3 不同），距全零特征值极远。判据只排除"全零"这一个点，
// 不与 CPU 锚比较，故不受 FMA 影响——跨端判对仍由 abi.rs 的 F32_TOL 逐字段负责。

/** GI 输出打包的 u32 元素数（3 探针 × 9 SH × RGB + 4 查询点 × RGB），与 abi.rs `gi_fingerprint_of` 同口径。 */
const GI_FP_U32_COUNT = 3 * 9 * 3 + 4 * 3;

/**
 * 「dispatch 没跑」时回读会得到的指纹——全零缓冲的 FNV-1a32。
 * @returns {string} 8 位小写十六进制（不含 `0x`）
 */
export function giIdleFingerprint() {
  return hashU32(new Uint32Array(GI_FP_U32_COUNT)).toString(16).padStart(8, "0");
}

/**
 * 判定一个真机上报的 GI 指纹是否落在「空转特征值」上。
 *
 * 覆盖三种"断言与执行脱节"的形态：
 *   * `null` / 空串→ 根本没上报（字段缺失）；
 *   * 非 8 位十六进制 → 上报链路坏了；
 *   * 等于全零特征值 → **dispatch 极可能没跑**，回读的是零初始化缓冲。
 *
 * @param {string|null} fp `fingerprint=0x…` 抽出的原始串
 * @returns {boolean} true = 不可信（必须判红）
 */
export function giFingerprintLooksIdle(fp) {
  if (fp === null || fp === undefined) return true;
  const hex = String(fp).replace(/^0x/, "").toLowerCase();
  if (!/^[0-9a-f]{8}$/.test(hex)) return true;
  // 恒等于 0 的指纹同样不可信：真实 GI 结果不可能全为 +0.0。
  if (hex === "00000000") return true;
  return hex === giIdleFingerprint();
}

// ---- 反向自证：故意喂"空转指纹"，必须被判红（证明上面这个判据非恒真）----
//
// 【为什么这条必须存在】`giFingerprintLooksIdle` 若写错成恒返回 false，它在
// 正向跑真机时会一路绿灯，而真机恰好空转时也绿——正是本项目最贵的缺陷形态。
// 故用三个**确定的坏输入**（全零特征值 / 恒零 / 字段缺失）证明它真会判红，
// 再用一个**确定的正常形态**证明它不会把正常指纹也判红。
if (process.argv.includes("--self-test")) {
  const idle = giIdleFingerprint();
  const cases = [
    { desc: `全零缓冲特征值0x${idle}（dispatch 没跑）`, fp: `0x${idle}`, wantIdle: true },
    { desc: "恒零指纹 0x00000000", fp: "0x00000000", wantIdle: true },
    { desc: "指纹字段缺失（null）", fp: null, wantIdle: true },
    { desc: "格式非法（位数不足）", fp: "0x1a2", wantIdle: true },
    { desc: "真机实测指纹 0x87ac3cf3（真跑过的形态）", fp: "0x87ac3cf3", wantIdle: false },
    { desc: "CPU 口径锚0xaf8429a3（真跑过的形态）", fp: "0xaf8429a3", wantIdle: false },
  ];
  console.log("== host_test GI 指纹空转判定自证（故意喂坏值，必须判红）==");
  let bad = 0;
  for (const c of cases) {
    const got = giFingerprintLooksIdle(c.fp);
    const ok = got === c.wantIdle;
    if (!ok) bad += 1;
    console.log(`  ${ok ? "ok  " : "FAIL"}  ${c.desc} → ${got ? "判红(不可信)" : "放行(可信)"}`);
  }
  // 附加：全零特征值必须与真机实测指纹/CPU 锚三者两两不同，否则判据失去区分力。
  const distinct = new Set([idle, "00000000", "87ac3cf3", "af8429a3"]).size === 4;
  if (!distinct) bad += 1;
  console.log(`  ${distinct ? "ok  " : "FAIL"}  空转特征值与真机/CPU 指纹两两不同（判据有区分力）`);
  console.log(`\n自证 ${cases.length + 1} 例，失败 ${bad} 例`);
  process.exit(bad ? 1 : 0);
}

// ---------------------------------------------------------------------------
// 宿主胶水：14 个 `prism_*` 导入。无 WebGPU 的环境下如实报告"不可用"。
// ---------------------------------------------------------------------------
function makeGlue(memoryRef, stats) {
  return {
    prism_navigator_gpu_available: () => 0, // node 无 navigator.gpu
    prism_request_device: () => 0,
    prism_device_lost: () => 0,
    // node 无 GPUDevice：恒为"未就绪"（Rust 侧据此把 GPU 项判 SKIP/FAIL，而非误判就绪）。
    prism_device_ready: () => 0,
    prism_create_buffer: () => {
      stats.glue.create_buffer += 1;
      return 0;
    },
    prism_write_buffer: () => {
      stats.glue.write_buffer += 1;
      return 0;
    },
    prism_read_buffer: () => {
      stats.glue.read_buffer += 1;
      return 0;
    },
    prism_copy_out: () => {
      stats.glue.copy_out += 1;
      return 0;
    },
    prism_create_shader_module: () => {
      stats.glue.create_shader_module += 1;
      return 0;
    },
    prism_create_compute_pipeline: () => {
      stats.glue.create_compute_pipeline += 1;
      return 0;
    },
    prism_create_bind_group: () => {
      stats.glue.create_bind_group += 1;
      return 0;
    },
    prism_dispatch: () => {
      stats.glue.dispatch += 1;
      return 0;
    },
    prism_wait_idle: () => {
      stats.glue.wait_idle += 1;
      return 0;
    },
    prism_pop_error: () => 0,
  };
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
const bytes = readFileSync(WASM_PATH);
const module = new WebAssembly.Module(bytes);
const instance = new WebAssembly.Instance(module, { env: makeGlue(null, { glue: {} }) });
const x = instance.exports;

// `memory` 在实例化后固定；宿主每次访问都重新取 `x.memory.buffer`，
// 因为 Rust 侧分配可能触发 memory.grow 使旧 ArrayBuffer 失效（detached）。
const u8 = () => new Uint8Array(x.memory.buffer);
function readStr(ptr, len) {
  return new TextDecoder().decode(u8().subarray(ptr, ptr + len));
}
const errText = () => readStr(x.prism_error_ptr(), x.prism_error_len());
const resText = () => readStr(x.prism_result_ptr(), x.prism_result_len());

const STAT = { PASS: 0, FAIL: 1, PENDING: 2, SKIP: 3, BAD_ARG: 4 };

section("0. 模块加载");
check("wasm 文件可读", bytes.length > 0, `${bytes.length} 字节`);
check("导出 memory", x.memory instanceof WebAssembly.Memory);
check("ABI 版本 = 1", x.prism_abi_version() === 1, `prism_abi_version()=${x.prism_abi_version()}`);

const REQUIRED_EXPORTS = [
  "prism_abi_version", "prism_alloc", "prism_dealloc",
  "prism_error_ptr", "prism_error_len", "prism_result_ptr", "prism_result_len",
  "prism_test_count", "prism_test_id", "prism_test_id_len",
  "prism_test_name", "prism_test_name_len", "prism_test_needs_gpu",
  "prism_test_run", "prism_test_collect",
  "prism_shader_count", "prism_shader_name", "prism_shader_name_len",
  "prism_shader_src", "prism_shader_src_len",
  "prism_kernel_wgsl", "prism_kernel_wgsl_len",
];
const missing = REQUIRED_EXPORTS.filter((n) => typeof x[n] !== "function");
check("全部 ABI 导出齐备", missing.length === 0, missing.length ? "缺 " + missing.join(",") : `${REQUIRED_EXPORTS.length} 个`);

const IMPORTS = WebAssembly.Module.imports(module).map((i) => i.name).sort();
const GLUE_NAMES = [
  "prism_navigator_gpu_available", "prism_request_device", "prism_device_lost", "prism_device_ready",
  "prism_create_buffer", "prism_write_buffer", "prism_read_buffer", "prism_copy_out",
  "prism_create_shader_module", "prism_create_compute_pipeline", "prism_create_bind_group",
  "prism_dispatch", "prism_wait_idle", "prism_pop_error",
].sort();
check("14 个胶水导入全部被 Rust 引用", JSON.stringify(IMPORTS) === JSON.stringify(GLUE_NAMES), `${IMPORTS.length} 个`);

section("1. 线性内存往返（JS 写 → Rust 读 → JS 校验）");
const LEN = 128;
const ptr = x.prism_alloc(LEN);
check("prism_alloc 返回非 0", ptr !== 0, `ptr=${ptr}`);
check("16 字节对齐", ptr % 16 === 0, `ptr=${ptr}`);
const payload = new Uint8Array(LEN);
for (let i = 0; i < LEN; i += 1) payload[i] = (i * 7 + 3) & 0xff;
u8().set(payload, ptr);
const readback = u8().slice(ptr, ptr + LEN);
check("JS 写入的字节 Rust 侧可见（内存同一性）", readback.every((b, i) => b === payload[i]));
// 独立验证：把同一图样塞进 Rust 内部工作区的读回路径 —— 由 abi.alloc 测试项覆盖。
const d1 = new Uint8Array(u8().buffer, ptr, LEN);
d1[0] = 0xab;
check("写入立即可读（同一 buffer 视图）", u8()[ptr] === 0xab);
check("prism_dealloc 正常返回", x.prism_dealloc(ptr, LEN) === 0);
check("dealloc(0) 幂等", x.prism_dealloc(0, LEN) === 0);

section("2. 测试项清单（Rust 单一真源）");
const count = x.prism_test_count();
check("清单非空", count > 0, `${count} 项`);
const manifest = [];
for (let i = 0; i < count; i += 1) {
  const id = readStr(x.prism_test_id(i), x.prism_test_id_len(i));
  const name = readStr(x.prism_test_name(i), x.prism_test_name_len(i));
  const needsGpu = x.prism_test_needs_gpu(i);
  manifest.push({ i, id, name, needsGpu });
  console.log(`  [${i}] ${id.padEnd(16)} needs_gpu=${needsGpu}  ${name}`);
}
check("每项 id 非空", manifest.every((m) => m.id.length > 0));
check("id 无重复", new Set(manifest.map((m) => m.id)).size === count);
check("越界读返回 0", x.prism_test_id(count) === 0 && x.prism_test_name(count) === 0);
check("越界 run 返回 BAD_ARG", x.prism_test_run(count) === STAT.BAD_ARG, `code=${x.prism_test_run(count)}`);
check("越界 collect 返回 BAD_ARG", x.prism_test_collect(count) === STAT.BAD_ARG);

section("3. CPU 参考测试项（wasm 数值 vs JS 独立实现）");
const byId = Object.fromEntries(manifest.map((m) => [m.id, m]));

function runItem(id) {
  const m = byId[id];
  if (!m) throw new Error("清单缺 " + id);
  let code = x.prism_test_run(m.i);
  if (code === STAT.PENDING) {
    // 无 WebGPU 时不应出现 PENDING；真机上这里 await onSubmittedWorkDone。
    code = x.prism_test_collect(m.i);
  }
  return { code, detail: code === STAT.FAIL ? errText() : resText() };
}

// --- abi.alloc ---
{
  const { code, detail } = runItem("abi.alloc");
  check("abi.alloc PASS", code === STAT.PASS, code === STAT.FAIL ? detail : "");
  check("abi.alloc 报告了对齐", /16B 对齐/.test(detail));
}

// --- cpu.scan：核心交叉判对 ---
{
  const { code, detail } = runItem("cpu.scan");
  check("cpu.scan PASS", code === STAT.PASS, code === STAT.FAIL ? detail : "");
  // 独立复算：128 元素、值 (3i+1)。
  const input = new Uint32Array(128);
  for (let i = 0; i < 128; i += 1) input[i] = (Math.imul(i, 3) + 1) >>> 0;
  const jsBlelloch = blellochScan(input);
  const jsSerial = serialScan(input);
  check("JS 两种 scan 自身一致", jsBlelloch.every((v, i) => v === jsSerial[i]));
  const fp = field(detail, "fingerprint");
  check("wasm fingerprint == JS 独立实现 fingerprint",
    fp !== null && parseInt(fp, 16) === hashU32(jsBlelloch),
    `wasm=${fp} js=0x${hashU32(jsBlelloch).toString(16).padStart(8, "0")}`);
  const last = jsBlelloch[127];
  check("末项 = 输入总和", detail.includes(`out[127]=${last}`), `期望 out[127]=${last}`);
}

// --- cpu.sort ---
{
  const { code, detail } = runItem("cpu.sort");
  check("cpu.scan 之后的 cpu.sort PASS", code === STAT.PASS, code === STAT.FAIL ? detail : "");
  const lcg = makeLcg(7);
  const items = Array.from({ length: 256 }, (_, i) => [lcg(), i >>> 0]);
  const jsSorted = radixSort(items);
  const jsKeys = Uint32Array.from(jsSorted.map(([k]) => k));
  check("JS radix 结果升序", jsKeys.every((v, i) => i === 0 || jsKeys[i - 1] <= v));
  const fp = field(detail, "fingerprint");
  check("wasm keys fingerprint == JS 独立实现",
    fp !== null && parseInt(fp, 16) === hashU32(jsKeys),
    `wasm=${fp} js=0x${hashU32(jsKeys).toString(16).padStart(8, "0")}`);
  // 稳定性：复现 Rust 测试的输入（LCG seed=42，键 = lcg % 4，载荷 = 下标），
  // 按桶收集各键在**输入中的出现序**，再与排序输出逐位对读。
  const lcg2 = makeLcg(42);
  const src = [];
  const buckets = [[], [], [], []];
  for (let i = 0; i < 500; i += 1) {
    const k = lcg2() % 4;
    src.push([k, i >>> 0]);
    buckets[k].push(i >>> 0);
  }
  const jsStable = radixSort(src);
  // 期望输出 = 各桶按序拼接（键升序 → 桶 0,1,2,3）。
  const expect = [];
  for (let b = 0; b < 4; b += 1) for (const p of buckets[b]) expect.push([b, p]);
  check("JS radix 稳定性（逐桶载荷序 == 输入序）",
    jsStable.length === expect.length && jsStable.every(([k, p], i) => k === expect[i][0] && p === expect[i][1]));
}

// --- gpu.gi_compute：GI GPU 路径的 JS 独立实现交叉判对 ---
//
// 【U4 深化 B 新增】`gpu.gi_compute` 无 device 时 SKIP（第 6 节覆盖其 SKIP 语义）。
// 这里再补一个**第三份独立实现**：用 JS（f32 逐式模拟）复算整条 GI 流水线
// （SDF sphere tracing → 半球采样 → SH-L2 投影 → 最近探针重建 + Lambert），
// 与 Rust CPU 参考（abi.rs 的 gi_*_ref 族）应逐位一致，且其 FNV 指纹应与
// Rust 侧真机 dispatch 后上报的 `fingerprint=0x…` 相等。三份实现（WGSL / Rust-CPU
// 参考 / 本 JS）互不共享代码，任何一方的口径漂移都会被这条断言抓住。
{
  const FRAC_PI = Math.fround(3.141592653589793);
  const TWO_PI = Math.fround(6.283185307179586);
  const FRAC_PI_2 = Math.fround(1.5707963267948966);
  const FRAC_PI_4 = Math.fround(0.7853981633974483);

  const f = Math.fround; // 每个算术步骤后收窄到 f32（模拟 WGSL/CPU 参考）
  /** 确定性三角件，与 `rt_common.wgsl::cos_sin` / abi.rs `wf_cos_sin` 同构。 */
  function quarterSin(x) {
    const x2 = f(x * x);
    // x * (1 + x2*(-1/6 + x2*(1/120 + x2*(-1/5040 + x2*(1/362880)))))
    let acc = f(1.0 / 362880.0);
    acc = f(f(-1.0 / 5040.0) + f(x2 * acc));
    acc = f(f(1.0 / 120.0) + f(x2 * acc));
    acc = f(f(-1.0 / 6.0) + f(x2 * acc));
    return f(x * f(1.0 + f(x2 * acc)));
  }
  function quarterCos(x) {
    const x2 = f(x * x);
    let acc = f(1.0 / 40320.0);
    acc = f(f(-1.0 / 720.0) + f(x2 * acc));
    acc = f(f(1.0 / 24.0) + f(x2 * acc));
    acc = f(f(-0.5) + f(x2 * acc));
    return f(1.0 + f(x2 * acc));
  }
  function cosSin(x) {
    let r = f(x - f(TWO_PI * Math.floor(f(x / TWO_PI))));
    const q = (f(r * f(2.0 / FRAC_PI)) | 0) & 3; // as i32 as u32 & 3（x>=0 时等价）
    r = f(r - f(q * FRAC_PI_2));
    let c;
    let s;
    if (r > FRAC_PI_4) {
      const rr = f(FRAC_PI_2 - r);
      c = quarterSin(rr);
      s = quarterCos(rr);
    } else {
      c = quarterCos(r);
      s = quarterSin(r);
    }
    if (q === 0) return [c, s];
    if (q === 1) return [f(-s), c];
    if (q === 2) return [f(-c), f(-s)];
    return [s, f(-c)];
  }
  function normalize(v) {
    // Rust `v0*v0 + v1*v1 + v2*v2` 左结合 → `((a+b)+c)`；此处必须同序。
    const len = f(Math.sqrt(f(f(f(v[0] * v[0]) + f(v[1] * v[1])) + f(v[2] * v[2]))));
    if (len > 0) return [f(v[0] / len), f(v[1] / len), f(v[2] / len)];
    return [0, 0, 0];
  }
  function onb(n) {
    const up = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
    const t = normalize([
      f(f(up[1] * n[2]) - f(up[2] * n[1])),
      f(f(up[2] * n[0]) - f(up[0] * n[2])),
      f(f(up[0] * n[1]) - f(up[1] * n[0])),
    ]);
    const b = [
      f(f(n[1] * t[2]) - f(n[2] * t[1])),
      f(f(n[2] * t[0]) - f(n[0] * t[2])),
      f(f(n[0] * t[1]) - f(n[1] * t[0])),
    ];
    return [t, b, n];
  }
  // 与 abi.rs 的固定测试参数 / 场景逐字一致。
  const P = {
    probes: 3, points: 4, prims: 2, nDirs: 8,
    env: [f(0.9), f(0.8), f(0.7)], albedo: [f(0.8), f(0.8), f(0.8)],
    tMax: 20.0, hitEps: 1e-3, safety: 0.9, maxSteps: 48,
  };
  const PRIMS = [
    { kind: 0, radius: 1.0, center: [2.0, 0.0, 0.0] },
    { kind: 1, radius: 10.0, center: [0.0, -10.0, 0.0] },
  ];
  const PROBE_INPUTS = [
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
  ];
  const POINT_INPUTS = [[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0], [2.5, 1.0, 0.0]];

  function primDist(prim, p) {
    const q = [
      f(f(Math.abs(f(p[0] - prim.center[0])) - prim.radius)),
      f(f(Math.abs(f(p[1] - prim.center[1])) - prim.radius)),
      f(f(Math.abs(f(p[2] - prim.center[2])) - prim.radius)),
    ];
    if (prim.kind === 1) {
      const cx = Math.max(q[0], 0);
      const cy = Math.max(q[1], 0);
      const cz = Math.max(q[2], 0);
      const outside = f(Math.sqrt(f(f(f(cx * cx) + f(cy * cy)) + f(cz * cz))));
      const inside = Math.min(Math.max(Math.max(q[0], q[1]), q[2]), 0);
      return f(outside + inside);
    }
    return q[0];
  }
  function sceneDist(p) {
    let d = 1.0e6;
    for (let i = 0; i < P.prims; i += 1) d = Math.min(d, primDist(PRIMS[i], p));
    return d;
  }
  function sphereTrace(ro, rd) {
    for (let k = 0; k < 3; k += 1) if (!Number.isFinite(rd[k])) return false;
    const rl = f(Math.sqrt(f(f(f(rd[0] * rd[0]) + f(rd[1] * rd[1])) + f(rd[2] * rd[2]))));
    if (rl < 0.5) return false;
    const dir = [f(rd[0] / rl), f(rd[1] / rl), f(rd[2] / rl)];
    let t = 0.0;
    for (let s = 0; s < P.maxSteps; s += 1) {
      const d = sceneDist([f(ro[0] + f(dir[0] * t)), f(ro[1] + f(dir[1] * t)), f(ro[2] + f(dir[2] * t))]);
      if (d <= P.hitEps) return true;
      t = f(t + f(d * P.safety));
      if (t > P.tMax) return false;
    }
    return false;
  }
  const GOLDEN = f(2.3999632); // 黄金角（先收窄 f32，与 Rust 字面量推导一致）
  function hemiDir(d, n) {
    const idx = f(d + 0.5);
    const z = f(idx / n); // cosθ 均匀 ∈ (0,1)
    const r = f(Math.sqrt(Math.max(f(1.0 - f(z * z)), 0)));
    const phi = f(idx * GOLDEN);
    const [c, s] = cosSin(phi);
    return [f(r * c), f(r * s), z];
  }
  function shBasisL2(dir) {
    // 逐式镜像 abi.rs::gi_sh_basis_l2_ref（浮点结合序也必须一致，否则 f32 舍入漂移）：
    //   c1*y / c1*z / c1*x ；c2*x*y 为 ((c2*x)*y) ；c2_0*(3*z*z-1) 等。
    const [x, y, z] = dir;
    const c0 = Math.fround(0.28209479);
    const c1 = Math.fround(0.48860251);
    const c2 = Math.fround(1.09254846);
    const c2b = Math.fround(0.31539157);
    const c2c = Math.fround(0.54627423);
    return [
      c0,
      f(c1 * y),
      f(c1 * z),
      f(c1 * x),
      f(f(c2 * x) * y),
      f(f(c2 * y) * z),
      f(c2b * f(f(f(3.0 * z) * z) - 1.0)),
      f(f(c2 * x) * z),
      f(c2c * f(f(x * x) - f(y * y))),
    ];
  }
  function cosineLobe(band) {
    if (band === 0) return FRAC_PI;
    if (band === 1) return f(f(2.0 * FRAC_PI) / 3.0);
    return f(FRAC_PI / 4.0);
  }
  function bandOf(i) { return i === 0 ? 0 : (i <= 3 ? 1 : 2); }
  function traceProbe(pos, normal) {
    const samples = Array.from({ length: 8 }, () => [0, 0, 0]);
    const nl = f(Math.sqrt(f(f(f(normal[0] * normal[0]) + f(normal[1] * normal[1])) + f(normal[2] * normal[2]))));
    if (nl < 0.5) return { samples, sh: Array.from({ length: 9 }, () => [0, 0, 0]) };
    const n = [f(normal[0] / nl), f(normal[1] / nl), f(normal[2] / nl)];
    const [t, b] = onb(n);
    const nd = Math.min(P.nDirs, 8);
    for (let d = 0; d < nd; d += 1) {
      const local = hemiDir(d, nd);
      const world = normalize([
        f(f(f(t[0] * local[0]) + f(b[0] * local[1])) + f(n[0] * local[2])),
        f(f(f(t[1] * local[0]) + f(b[1] * local[1])) + f(n[1] * local[2])),
        f(f(f(t[2] * local[0]) + f(b[2] * local[1])) + f(n[2] * local[2])),
      ]);
      const ro = [f(pos[0] + f(n[0] * 1e-3)), f(pos[1] + f(n[1] * 1e-3)), f(pos[2] + f(n[2] * 1e-3))];
      if (!sphereTrace(ro, world)) samples[d] = P.env.slice();
    }
    return { samples, sh: Array.from({ length: 9 }, () => [0, 0, 0]) };
  }
  function projectProbe(pos, normal) {
    const r = traceProbe(pos, normal);
    const nl = f(Math.sqrt(f(f(f(normal[0] * normal[0]) + f(normal[1] * normal[1])) + f(normal[2] * normal[2]))));
    if (nl < 0.5 || P.nDirs === 0) return r;
    const n = [f(normal[0] / nl), f(normal[1] / nl), f(normal[2] / nl)];
    const [t, b] = onb(n);
    const nd = Math.min(P.nDirs, 8);
    const dOmega = f(f(2.0 * FRAC_PI) / nd);
    for (let d = 0; d < nd; d += 1) {
      const local = hemiDir(d, nd);
      const world = normalize([
        f(f(f(t[0] * local[0]) + f(b[0] * local[1])) + f(n[0] * local[2])),
        f(f(f(t[1] * local[0]) + f(b[1] * local[1])) + f(n[1] * local[2])),
        f(f(f(t[2] * local[0]) + f(b[2] * local[1])) + f(n[2] * local[2])),
      ]);
      const yb = shBasisL2(world);
      const rad = r.samples[d];
      for (let k = 0; k < 9; k += 1) {
        const w = f(yb[k] * dOmega);
        r.sh[k][0] = f(r.sh[k][0] + f(rad[0] * w));
        r.sh[k][1] = f(r.sh[k][1] + f(rad[1] * w));
        r.sh[k][2] = f(r.sh[k][2] + f(rad[2] * w));
      }
    }
    return r;
  }
  function interpolate(query, refs) {
    let best = 0;
    let bestD2 = Infinity;
    for (let j = 0; j < refs.length; j += 1) {
      const dx = f(refs[j].pos[0] - query[0]);
      const dy = f(refs[j].pos[1] - query[1]);
      const dz = f(refs[j].pos[2] - query[2]);
      const d2 = f(f(dx * dx) + f(f(dy * dy) + f(dz * dz)));
      if (d2 < bestD2) { bestD2 = d2; best = j; }
    }
    const sh = refs[best].sh;
    const yb = shBasisL2([0, 0, 1]); // 重建方向 +Z（与 WGSL 一致）
    const e = [0, 0, 0];
    for (let k = 0; k < 9; k += 1) {
      const cc = f(cosineLobe(bandOf(k)) * yb[k]);
      e[0] = f(e[0] + f(sh[k][0] * cc));
      e[1] = f(e[1] + f(sh[k][1] * cc));
      e[2] = f(e[2] + f(sh[k][2] * cc));
    }
    const invPi = f(1.0 / FRAC_PI);
    return [f(f(P.albedo[0] * e[0]) * invPi), f(f(P.albedo[1] * e[1]) * invPi), f(f(P.albedo[2] * e[2]) * invPi)];
  }
  // 逐探针投影 → 逐查询点插值 → 压成 u32 序列 → FNV 指纹。
  const refs = PROBE_INPUTS.map((pr) => {
    const r = projectProbe([pr[0], pr[1], pr[2]], [pr[3], pr[4], pr[5]]);
    return { pos: [pr[0], pr[1], pr[2]], sh: r.sh };
  });
  const flatU32 = [];
  for (let i = 0; i < refs.length; i += 1) {
    for (let k = 0; k < 9; k += 1) for (let c = 0; c < 3; c += 1) flatU32.push(f32Bits(refs[i].sh[k][c]));
  }
  for (const q of POINT_INPUTS) {
    const e = interpolate(q, refs);
    for (const c of e) flatU32.push(f32Bits(c));
  }
  const jsFp = hashU32(flatU32);

  const r = runItem("gpu.gi_compute");
  // 无 device → SKIP 是预期（真机才会真 dispatch，见 realgpu_check.mjs）。
  check("gpu.gi_compute 无 device 时 SKIP", r.code === STAT.SKIP, `code=${r.code}`);
  // 无论 SKIP（无 device）还是 PASS（真机），JS 独立实现都必须能复算出指纹；
  // 这里断言 JS 自身确定性（两次复算一致），并把 JS 指纹打印出来供真机比对。
  const refs2 = PROBE_INPUTS.map((pr) => projectProbe([pr[0], pr[1], pr[2]], [pr[3], pr[4], pr[5]]).sh);
  let stableFp = true;
  for (let i = 0; i < refs.length; i += 1) {
    for (let k = 0; k < 9; k += 1) for (let c = 0; c < 3; c += 1) {
      if (f32Bits(refs[i].sh[k][c]) !== f32Bits(refs2[i][k][c])) stableFp = false;
    }
  }
  check("JS GI 独立实现自身确定（两次复算逐位一致）", stableFp);
  // 物理合理性：朝上探针（idx 0）应比正对球的探针（idx 1）更亮（遮挡更少）。
  const lum = (v) => f(f(0.2126 * v[0]) + f(f(0.7152 * v[1]) + f(0.0722 * v[2])));
  const eUp = interpolate(PROBE_INPUTS[0].slice(0, 3), refs);
  const eBall = interpolate(PROBE_INPUTS[1].slice(0, 3), refs);
  check("JS GI 遮挡方向性：朝上探针辐照度 > 正对球探针",
    lum(eUp) > lum(eBall), `up=${lum(eUp).toFixed(6)} ball=${lum(eBall).toFixed(6)}`);
  // JS 独立实现的指纹锚定：与 `abi.rs::gi_cpu_reference_fingerprint_matches_independent_js`
  // 里断言的常量必须一致（同一份固定输入的 GI 全链路 CPU 结果）。两边任一漂移都会红。
  // ⚠️ 注意：此指纹是**纯 CPU 口径**（无 FMA）。真机上 GPU 允许 FMA 收缩，逐位指纹会
  // 不同（实测真机 fingerprint=0x87ac3cf3，与 CPU 锚 0xaf8429a3 差在若干 ULP），
  // 故**不以逐位指纹判对 GPU**——GPU vs CPU 的判对在 abi.rs 的 `c_gpu_gi_compute`
  // 里按 F32_TOL 逐字段做（真机实测最大偏差 1.2e-7，mismatch=0）。真机指纹的用途是
  // 「同机两次运行一致」的确定性核对，不参与跨端逐位比较。
  check("JS GI CPU 口径指纹 == Rust CPU 参考锚定值 0xaf8429a3",
    (jsFp >>> 0) === 0xaf8429a3,
    `js_gi_fingerprint=0x${(jsFp >>> 0).toString(16).padStart(8, "0")}`);
  // 真机 PASS 时的指纹断言。**本路加固**：原判据只查"8 位十六进制"（恒可通过），
  // 现升级为值域排除——指纹若等于"全零缓冲特征值"（WebGPU 缓冲零初始化），
  // 说明 **dispatch 极可能没跑**，回读的是初始零值，正是事故 II 的形态。
  // 反向自证见本文件 `--self-test`（7 例，故意喂空转值必须判红）。
  if (r.code === STAT.PASS) {
    const fp = field(r.detail, "fingerprint");
    const idle = giIdleFingerprint();
    check("真机 gpu.gi_compute 上报了格式合法的指纹", fp !== null && /^[0-9a-f]{8}$/.test(String(fp).replace(/^0x/, "")),
      `gpu_fingerprint=${fp}`);
    check(`真机 GI 指纹 ≠ 全零空转特征值 0x${idle}（证明 dispatch 真跑过、输出非初始零值）`,
      !giFingerprintLooksIdle(fp),
      `gpu_fingerprint=${fp}（若等于 0x${idle} 说明回读的是零初始化缓冲 —— submit 无效/dispatch 没跑）`);
  }
}

// --- wgsl.structural / shader.params ---
{
  const a = runItem("wgsl.structural");
  check("wgsl.structural PASS（12 个 WGSL 结构自检）", a.code === STAT.PASS, a.code === STAT.FAIL ? a.detail : "");
  // 【U4 深化 B 改动】新增 `gi_compute.wgsl` 一个 D2 内核（GI GPU 路径），
  // 结构性自检数由 11 增至 12。改断言数值即可，不改语义。
  check("wgsl.structural 覆盖 10 个 D2/D3/D4 + 2 个 D0 内核", (a.detail.match(/PASS/g) || []).length === 12);
  const b = runItem("shader.params");
  check("shader.params PASS", b.code === STAT.PASS, b.code === STAT.FAIL ? b.detail : "");
}

section("4. WGSL 源导出（宿主可直接复用 Rust 侧同一份源码）");
{
  const n = x.prism_shader_count();
  // 【U4 深化 B 改动】新增 `gi_compute.wgsl`，shader 数由 9 增至 10。
  check("shader 数 = 10", n === 10, `${n} 个`);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const nm = readStr(x.prism_shader_name(i), x.prism_shader_name_len(i));
    const src = readStr(x.prism_shader_src(i), x.prism_shader_src_len(i));
    total += src.length;
    if (!nm.endsWith(".wgsl")) check(`${nm} 文件名合法`, false);
    if (!src.includes("@compute") && !src.includes("@vertex") && !src.includes("@fragment")) {
      // include-only 文件（rt_common 等）本就没有入口，属预期。
      if (!["rt_common.wgsl", "rt_bvh.wgsl", "rt_shade.wgsl", "sim_common.wgsl"].includes(nm)) {
        check(`${nm} 含入口`, false);
      }
    }
    if (/[^"]\b(sin|cos|tan|exp|pow|log|atan)\(/.test(src)) check(`${nm} 无 libm 超越函数（D-29）`, false);
  }
  check("10 个 WGSL 源码可读且结构合法", true, `合计 ${total} 字节`);
  const k0 = readStr(x.prism_kernel_wgsl(0), x.prism_kernel_wgsl_len(0));
  const k1 = readStr(x.prism_kernel_wgsl(1), x.prism_kernel_wgsl_len(1));
  check("D0 内核 scan_block 可读", k0.includes("fn scan_block"), `${k0.length} 字节`);
  check("D0 内核 histogram 可读", k1.includes("fn histogram"), `${k1.length} 字节`);
  check("越界内核返回 0", x.prism_kernel_wgsl(2) === 0);
}

section("5. WGSL uniform 布局门禁（D-14 真机教训固化：本地即可发现）");
{
  // ---- 5.1 解析 WGSL 的 uniform struct 声明，校验 16 字节对齐 ----
  // WGSL 规范（§Address space layout constraints）：uniform 地址空间里结构体的
  // 对齐 = roundUp(16, max(成员对齐))，大小 = roundUp(对齐, 末成员末尾)。
  // Tint/Dawn 在 createBindGroup 时按此校验最小绑定大小，缓冲区小于它直接验证失败。
  //
  // 真机教训：`scan_block` 的 `struct Params {block: u32, count: u32}` 只写 8 字节，
  // 而 uniform 对齐把它撑到 16 字节；host 端也只建了 8 字节缓冲 → 真机 FAIL。
  // 这类错误 CPU 侧结构自检（括号/D-29/符号）**完全查不出**，必须单独立门禁。

  /** 标量/向量类型的 (align, size)，单位字节。 */
  const TYPE_LAYOUT = {
    u32: [4, 4], i32: [4, 4], f32: [4, 4],
    vec2: [8, 8], vec3: [16, 12], vec4: [16, 16],
  };

  /** 计算一个 uniform 结构体的字节大小；无法解析时返回 null。 */
  function wgslUniformStructSize(src, structName) {
    // 用括号配对定位结构体体，而不是正则找 `\n}`：D0 内核的 `struct Params` 是
    // **单行**声明（`struct Params { a: u32, b: u32 };`），正则会一路吞到下一个
    // `}`（即整个函数体），把成员表算错。WGSL 成员类型里不出现花括号
    // （定长数组用 `<>`），故裸计数即安全。
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
    const body = src.slice(open + 1, close);
    // 去掉行注释，避免注释里的 `:` 被当成成员分隔。
    const clean = body.replace(/\/\/[^\n]*/g, "");
    // 【修复】按顶层逗号切分：`array<GiPrim, 4>` 里的逗号在 `<>` 内，朴素的
    // `split(",")` 会把它当成员分隔符，导致 `GiParams` 成员表解析失败。
    // 只按 `<>` / `()` / `[]` 深度 == 0 的逗号切。
    const parts = [];
    {
      let depth = 0;
      let cur = "";
      for (const ch of clean) {
        if (ch === "<" || ch === "(" || ch === "[") depth += 1;
        else if (ch === ">" || ch === ")" || ch === "]") depth -= 1;
        if (ch === "," && depth === 0) { parts.push(cur); cur = ""; continue; }
        cur += ch;
      }
      parts.push(cur);
    }
    const members = [];
    for (const raw of parts) {
      const t = raw.trim();
      if (!t) continue;
      const m = /^(\w+)\s*:\s*(.+)$/.exec(t);
      if (!m) return null; // 出现非 `name: type` 的内容 → 交人工，不猜
      members.push({ name: m[1], type: m[2].trim() });
    }
    if (!members.length) return null;
    let offset = 0;
    let maxAlign = 1;
    for (const mem of members) {
      // 【修复】支持定长数组 `array<T, N>`：元素类型 T 可标量/向量/结构体。
      const arr = /^array\s*<\s*([\w<>]+)\s*,\s*(\d+)\s*>$/.exec(mem.type);
      let align;
      let size;
      if (arr) {
        const elType = arr[1].trim();
        const count = Number(arr[2]);
        const elKey = Object.keys(TYPE_LAYOUT).find((k) => elType.startsWith(k));
        let elAlign;
        let elSize;
        if (elKey) {
          [elAlign, elSize] = TYPE_LAYOUT[elKey];
        } else {
          const es = wgslUniformStructSize(src, elType);
          if (es === null) return null;
          // 结构体元素对齐=max(成员对齐)，这里退化为 16（uniform 内所有结构体元素
          // 对齐至少 16）；大小用已算出的字节数。与 wgsl_lint.mjs 的 A 规则一致。
          elAlign = 16;
          elSize = es;
        }
        align = elAlign;
        // uniform 地址空间里数组元素 stride = roundUp(element size, 16)。
        const natural = Math.ceil(elSize / elAlign) * elAlign;
        const stride = Math.ceil(natural / 16) * 16;
        size = stride * count;
      } else {
        const key = Object.keys(TYPE_LAYOUT).find((k) => mem.type.startsWith(k));
        if (!key) return null; // 未知类型：交由人工确认，不猜
        [align, size] = TYPE_LAYOUT[key];
      }
      maxAlign = Math.max(maxAlign, align);
      // 每个成员按自身对齐向上取整偏移（这正是 vec3 后必须补 _pad 的原因）。
      offset = Math.ceil(offset / align) * align + size;
    }
    const structAlign = Math.ceil(Math.max(16, maxAlign) / 16) * 16;
    return Math.ceil(Math.max(offset, structAlign) / structAlign) * structAlign;
  }

  // 收集所有 WGSL 源（Rust 单一真源，经 wasm 导出）。
  const allWgsl = [];
  for (let i = 0; i < x.prism_shader_count(); i += 1) {
    allWgsl.push({
      label: readStr(x.prism_shader_name(i), x.prism_shader_name_len(i)),
      src: readStr(x.prism_shader_src(i), x.prism_shader_src_len(i)),
    });
  }
  for (let i = 0; ; i += 1) {
    const n = x.prism_kernel_wgsl_len(i);
    if (!n) break;
    allWgsl.push({
      label: `[D0] ${i === 0 ? "scan_block" : "histogram"}`,
      src: readStr(x.prism_kernel_wgsl(i), n),
    });
  }

  // 每个源里，找出所有 `var<uniform> x: T;` 并校验 T 的结构体大小。
  let uniformChecked = 0;
  for (const { label, src } of allWgsl) {
    for (const um of src.matchAll(/var<uniform>\s*(\w+)\s*:\s*(\w+)\s*;/g)) {
      const varName = um[1];
      const structName = um[2];
      const size = wgslUniformStructSize(src, structName);
      if (size === null) {
        // 解析不出来的结构体不能静默放过（否则门禁形同虚设）。
        check(`${label}: uniform '${varName}: ${structName}' 结构体可解析`, false, "无法解析成员列表");
        continue;
      }
      uniformChecked += 1;
      check(
        `${label}: uniform '${varName}: ${structName}' = ${size}B 且 16 字节对齐`,
        size % 16 === 0,
        `${size} 字节`,
      );
    }
  }
  check("全部 uniform 绑定均已过 16 字节对齐门禁", uniformChecked >= 9, `${uniformChecked} 个绑定`);

  // ---- 5.2 host 端建 uniform buffer 的 size 必须 >= shader 声明的结构体大小 ----
  // 交叉校验 abi.rs 的建缓冲调用：禁止字面量 8/12 之类，必须用 *_PARAMS_BYTES 常量。
  const abiSrc = readFileSync(resolve(HERE, "../src/web/abi.rs"), "utf8");
  const devSrc = readFileSync(resolve(HERE, "../src/web/device.rs"), "utf8");

  // 5.2a 常量自身必须 16 对齐且 >= 16。
  for (const m of devSrc.matchAll(/pub const (\w*PARAMS_BYTES): usize = (\d+);/g)) {
    const [, name, val] = m;
    const n = Number(val);
    check(`${name} = ${n} 且 16 字节对齐`, n % 16 === 0 && n >= 16, `${n} 字节`);
  }

  // 5.2b D0 两个内核的建缓冲调用必须用常量，且常量值一致。
  const scanUniformBytes = /pub const SCAN_PARAMS_BYTES: usize = (\d+);/.exec(devSrc);
  check("device.rs 导出 SCAN_PARAMS_BYTES = 16", !!scanUniformBytes && Number(scanUniformBytes[1]) === 16,
    scanUniformBytes ? scanUniformBytes[1] : "缺失");

  // 提取 make_buffer(..., <size>, usage::UNIFORM, ...) 调用，校验 size 实参。
  const uniformCalls = [...abiSrc.matchAll(
    /make_buffer\(\s*"([^"]+)"\s*,\s*dev\s*,\s*([^,]+?)\s*,\s*usage::UNIFORM/g,
  )];
  check("abi.rs 中 uniform 缓冲建缓冲调用可被门禁解析", uniformCalls.length >= 2,
    `${uniformCalls.length} 处`);

  for (const [, testName, sizeExprRaw] of uniformCalls) {
    const sizeExpr = sizeExprRaw.trim();
    // 允许 `<CONST> as u32` / `<CONST>`；禁止裸数字（那正是真机缺陷的形态）。
    const cm = /^(\w+)(?:\s+as\s+u32)?$/.exec(sizeExpr);
    if (!cm) {
      check(`${testName}: uniform size 用具名常量而非字面量`, false, `实参 = \`${sizeExpr}\``);
      continue;
    }
    const constName = cm[1];
    // 常量可能在 device.rs（D0 内核），也可能在 abi.rs（D-14 补测，声明为 u32）。
    // 旧实现在这里只查 device.rs 的 usize 声明，于是新测试项一律误报"常量缺失"，
    // 且硬编码"必须 16B"根本覆盖不了 SphParams/ClothParams/WFParams 的 48B。
    // 现改为：两处都查、类型不限，并把尺寸比对交给 wgsl_lint.mjs 的规则 A
    // （那里才真正按 WGSL 结构体算payload/size 并与宿主常量交叉比对）。
    // 本节只保留"size 必须是具名常量"这一条风格断言——它与规则 A 的 A3
    // 覆盖面不同，不算重复。
    const decl =
      new RegExp(`pub const ${constName}: (?:u32|u64|usize) = (\\d+);`).exec(devSrc)
      || new RegExp(`pub const ${constName}: (?:u32|u64|usize) = (\\d+);`).exec(abiSrc);
    if (!decl) {
      check(`${testName}: 常量 ${constName} 在 device.rs 或 abi.rs 有定义`, false);
      continue;
    }
    const bytes = Number(decl[1]);
    check(
      `${testName}: uniform buffer ${bytes}B 是 16 的倍数（常量 ${constName}）`,
      bytes > 0 && bytes % 16 === 0,
      `${constName}=${bytes}B`,
    );
  }
}

// 5b. 静态门禁（wgsl_lint.mjs 的规则 A–G）
//
// 【为什么独立成节而不是并入第 5 节】第 5 节的判据来自 **wasm 导出的 WGSL 文本**
// （能证明"上传 GPU 的那份源码"合规）；本节判据来自 **磁盘上的 .wgsl 文件 +
// device.rs 内嵌字符串 + index.html/build.sh 源码**（能证明"提交进仓库的东西"
// 合规）。两者数据源不同、失效模式也不同：wasm 是旧产物时，第 5 节会绿而本节
// 会红——那正是"改了没重新构建"这类漏检。故并列而非合并。
//
// 编号用 5b 而非重排 6/7：避免与同时在末尾追加新节的同事产生行号冲突。
section("5b. 静态门禁 wgsl_lint（规则 A–G：对齐 / 胶水 / 绑定偏移 / 常量 / 守卫 / 毛发散射 / GI）");
{
  // 动态 import：只在本节范围内引入依赖，不动文件头部 import 区，减少与
  // 并行改动的冲突面。门禁脚本零第三方依赖，只用 node 内置模块。
  const { runWgslLint, runWgslLintSelfTest } = await import("./wgsl_lint.mjs");
  const lint = runWgslLint();
  for (const c of lint.checks) {
    check(`门禁 ${c.name}`, c.ok, c.ok ? c.detail : `未通过：${c.detail || "详见 wgsl_lint.mjs"}`);
  }
  check(
    `静态门禁全通过（${lint.passed}/${lint.total}）`,
    lint.failed === 0,
    `${lint.failed} 项未通过`,
  );
  // 门禁自身的非恒真性：故意违反的样例必须被抓到。若 wgsl_lint 的规则恒真，
  // 上面那批 ok 只是"看起来在检查"，实则什么都没查——必须当场自证。
  const st = runWgslLintSelfTest();
  for (const r of st) {
    check(`门禁自证：规则 ${r.rule} 能抓住「${r.desc}」`, r.ok, r.detail);
  }
  check("门禁自证全部通过（无恒真规则）", st.every((r) => r.ok), `${st.length} 例`);
}

section("6. 无 WebGPU 时的行为（不白屏、不误报、不 panic）");
{
  const device = runItem("device.init");
  check("device.init 在无 navigator.gpu 时 SKIP", device.code === STAT.SKIP, `code=${device.code}`);
  check("SKIP 原因写入结果缓冲", device.detail.includes("navigator.gpu"), device.detail.slice(0, 60));
  for (const id of ["wgsl.compile", "gpu.scan_block", "gpu.histogram"]) {
    const r = runItem(id);
    check(`${id} 在无 device 时 SKIP`, r.code === STAT.SKIP, `code=${r.code}`);
  }
  // 关键：D-16 不留死胶水——SKIP 路径也必须已把 device 句柄写进内部状态，
  // 因此这些项不得依赖任何胶水调用；此处反向断言清单里 GPU 项确实存在。
  const gpuItems = manifest.filter((m) => m.needsGpu).map((m) => m.id);
  // 【D-14 补测改动】原断言写死 `gpuItems.length === 4`，新增 GPU 测试项后必然
  // 转红，且它验的其实不是"GPU 项是否被正确标注"（那由下面两条按 id 逐项查）。
  // 改为**语义断言**：每个 needs_gpu 项都必须能在无 device 环境下跑出 SKIP
  // （真 SKIP，不是被跳过的空跑），且 needs_gpu 项数 >= 4（原有 4 项不得被删）。
  check("GPU 依赖项数量未回退（>= 4）", gpuItems.length >= 4, `${gpuItems.length} 项`);
  for (const id of [
    "device.init", "wgsl.compile", "gpu.scan_block", "gpu.histogram",
    "gpu.sim_particles", "gpu.sim_sph", "gpu.sim_cloth", "gpu.rt_wavefront",
    "gpu.rt_wavefront_multi", "gpu.gi_compute",
  ]) {
    const m = byId[id];
    check(`GPU 项 ${id} 存在且标注 needs_gpu`, !!m && m.needsGpu === 1,
      m ? `needs_gpu=${m.needsGpu}` : "清单缺失");
  }
  for (const id of gpuItems) {
    const r = runItem(id);
    check(`GPU 项 ${id} 无 device 时 SKIP（非静默跳过）`, r.code === STAT.SKIP, `code=${r.code}`);
  }
}

// ===========================================================================
// Marschner CPU/GPU 数值对拍（Marschner 收口）
// ===========================================================================
//
// 【为什么必须并入 host_test 而不是独立脚本】本项目吃过"门禁在仓库里、但没接进
// 主流程"的亏：GI 模块被 `TEMP-SCOPE-WORKAROUND-DISABLE-GI` 注释掉，却报告
// "149 passed"。若对拍只以独立脚本形式存在，总有人会忘记跑它——那等于没有。
// 故这里**真的起子进程**跑 `hair_parity_gpu.mjs`，并把 12 组 × 3 分量的
// 实测误差逐格转成 `check()` 断言：跑不起来就红，超阈值就红。
//
// 【为什么用子进程而不是 import】`hair_parity_gpu.mjs` 顶层 await 真 WebGPU
// device 并在末尾 `process.exit()`，import 进来会直接杀掉宿主进程。
//
// 【GPU 不可用时的口径】**不允许静默跳过**。本节在无 GPU 环境下判 FAIL 并
// 给出明确原因（缺 Xvfb / 缺 Chromium / 无 adapter），而不是记 SKIP——
// 因为"CPU/GPU 一致"这件事只有真机能证明，CI 无 GPU 时静默跳过等于把
// 最关键的回归防线留在了无人看守的地方。CPU 侧 golden vector 仍可离线校验
// （见本节末尾的 fixture 断言），那部分不受 GPU 环境影响。
section("7b. Marschner CPU/GPU 数值对拍（真 WebGPU：12 组向量 × {R, TT, TRT}）");
{
  const { spawnSync } = await import("node:child_process");
  const parityPath = resolve(HERE, "hair_parity_gpu.mjs");
  const goldenPath = resolve(HERE, "hair_golden.json");

  // ---- 8a. CPU 侧 golden vector fixture（不需要 GPU，任何环境都必须成立）----
  let golden = null;
  try {
    golden = JSON.parse(readFileSync(goldenPath, "utf8"));
    check("对拍 fixture hair_golden.json 存在且为合法 JSON", true, goldenPath);
  } catch (e) {
    check("对拍 fixture hair_golden.json 存在且为合法 JSON", false, String(e.message || e));
  }
  if (golden) {
    // 【字段名是 `r_bits` / `tt_bits` / `trt_bits`，不是 `r` / `tt` / `trt`】
    // 后缀 `_bits` 是契约的一部分：它标明该值是 **f32 位模式**（u32 十六进制），
    // 而非 f32 的十进制近似。生成器 `emit_cpu_golden_vectors_for_gpu_parity`
    // 与对拍脚本 `hair_parity_gpu.mjs` 都按 `_bits` 取字段，改名即两边同时失效。
    const BIT_KEYS = ["r_bits", "tt_bits", "trt_bits"];
    check("CPU golden vector 恰好 12 组（对拍门禁按此计数）",
      Array.isArray(golden.vectors) && golden.vectors.length === 12,
      `实际 ${golden.vectors?.length}`);
    const badLen = (golden.vectors || []).filter((v) => !BIT_KEYS.every((k) => typeof v[k] === "string"));
    check("CPU golden vector 每组含 r_bits/tt_bits/trt_bits 三分量（f32 位模式）",
      badLen.length === 0, badLen.length ? `${badLen.length} 组字段缺失` : "36 个分量");
    // 入参字段也必须齐全：缺一个会让 GPU 侧读到 undefined → NaN，
    // 而 NaN 比较恒为 false 会让整节"因失败而红"，掩盖真正的数值结论。
    const IN_KEYS = ["theta_i", "phi_i", "theta_o", "phi_o", "beta_m", "k", "eta", "sigma_a", "radius"];
    const badIn = (golden.vectors || []).filter((v) => !IN_KEYS.every((k) => Number.isFinite(v[k])));
    check("CPU golden vector 每组 9 个物理入参均为有限数", badIn.length === 0,
      badIn.length ? `${badIn.length} 组入参异常：${badIn[0].name}` : IN_KEYS.join("/"));
    // CPU fixture 是**位模式**（无损），不是十进制。若哪天有人改成十进制，
    // 往返舍入会把对拍误差污染到阈值量级 —— 故显式拦住。
    const nonHex = (golden.vectors || []).flatMap((v) => BIT_KEYS.map((k) => v[k]))
      .filter((s) => typeof s === "string" && !/^0x[0-9a-f]{8}$/.test(s));
    check("CPU golden vector 用 f32 位模式无损传递（非十进制往返）",
      nonHex.length === 0, nonHex.length ? `${nonHex.length} 个分量格式异常：${nonHex[0]}` : "0x________");
    // CPU fixture 必须**随源更新**而非陈旧：整组三瓣全为 0 会让对拍恒真
    // （0 误差不代表两侧同构，也可能是 fixture 没写出来）。
    const allZero = (golden.vectors || []).filter((v) => BIT_KEYS.every((k) => v[k] === "0x00000000"));
    check("CPU golden vector 无整组全零向量（避免对拍恒真）", allZero.length === 0,
      allZero.length ? `${allZero.length} 组三瓣全零` : "");
  }

  // ---- 8b. GPU 一半：真起 SwiftShader 跑 12 组向量，与 CPU 逐分量比 ----
  const proc = spawnSync(process.execPath, [parityPath, "--json"], {
    encoding: "utf8",
    timeout: 600000, // 10 分钟：起 Xvfb + Chromium + 编译 WGSL 在慢机上偏慢
    maxBuffer: 64 * 1024 * 1024,
  });
  const stdout = proc.stdout || "";
  let report = null;
  const jsonStart = stdout.indexOf("{");
  if (jsonStart >= 0) {
    try { report = JSON.parse(stdout.slice(jsonStart)); } catch { report = null; }
  }
  if (!report) {
    // 链路失败即判红，并把 stderr 尾部落进 detail，便于定位。
    const errTail = (proc.stderr || "").trim().split("\n").slice(-6).join(" / ");
    check("Marschner CPU/GPU 对拍链路跑通（真 WebGPU dispatch）", false,
      `exit=${proc.status} signal=${proc.signal || "-"} ${errTail || stdout.trim().split("\n").slice(-3).join(" / ")}`);
  } else {
    check("Marschner CPU/GPU 对拍链路跑通（真 WebGPU dispatch）", true,
      `adapter=${report.adapter}，WGSL ${report.shaderBytes}B`);
    check(`对拍向量数 = 12（实际 ${report.rows.length}）`, report.rows.length === 12);
    // 阈值是**实测后**定的，不是拍出来的宽松值。见 hair_parity_gpu.mjs 文件头。
    check(`对拍阈值 abs ≤ ${report.absTol} / rel ≤ ${report.relTol}（实测后判定）`,
      report.absTol <= 1e-5 && report.relTol <= 1e-5,
      "阈值被放宽超过 1e-5 —— 不许放宽，须定位根因");
    // 逐格断言：12 组 × 3 分量 = 36 条。
    let worstAbs = 0, worstRel = 0, cells = 0;
    for (const row of report.rows) {
      for (const c of row.comps) {
        cells += 1;
        worstAbs = Math.max(worstAbs, c.abs);
        worstRel = Math.max(worstRel, c.rel);
        check(`对拍 ${row.name} ${c.name}：|CPU−GPU| = ${c.abs === 0 ? "0" : c.abs.toExponential(2)}`,
          c.ok, c.ok ? `rel=${c.rel.toExponential(2)}` : `rel=${c.rel.toExponential(2)} 超阈值`);
      }
    }
    check("对拍覆盖 12 × 3 = 36 个分量", cells === 36, `实际 ${cells}`);
    // 汇总一条，便于一眼看整体精度。
    check(`对拍最大绝对误差 ${worstAbs.toExponential(3)} ≤ ${report.absTol}（Marschner 敏感线 1e-3）`,
      worstAbs <= report.absTol, `实测 ${worstAbs.toExponential(3)}`);
    check(`对拍最大相对误差 ${worstRel.toExponential(3)} ≤ ${report.relTol}`, worstRel <= report.relTol,
      `实测 ${worstRel.toExponential(3)}`);
    // ---- 7b-submit：submit 有效性（防止"回读零值恰好满足容差"的假绿）----
    //
    // 【为什么单列这条】v2.8 事故的形态是：submit 无效（绑定用法/布局非法）时
    // `onSubmittedWorkDone()` 仍 resolve，mapAsync 回读拿到的是**未被 dispatch
    // 覆写的初始零值**。此时逐格容差断言"看起来在测东西"，实际测的是空转。
    // 对拍脚本本身没有在页面内开 submit validation error scope（它不在本路可改
    // 范围内），故在 host_test 这一层加两条**残留风险兜底**：
    //   (a) 整体判定 report.ok 必须为 true（不是只看逐格）；
    //   (b) GPU 输出**不能全为 0** —— 全 0 是"dispatch 从未写回"的典型指纹。
    //       若将来某次失误让所有期望值都恰为 0，这条会误报，但那种 fixture 本身
    //       就已触发上面"无整组全零向量"的断言，故两条互斥、不会同时失效。
    check("对拍整体判定 report.ok === true（不是只看逐格）", report.ok === true,
      `ok=${report.ok}`);
    const gpuNonZero = report.rows.flatMap((r) => r.comps.map((c) => c.gpu))
      .filter((v) => Number.isFinite(v) && v !== 0).length;
    check("对拍 GPU 输出非全零（若全零 = dispatch 未写回 = submit 无效的假绿指纹）",
      gpuNonZero > 0, `${gpuNonZero} 个非零分量 / 36`);
    const gpuFinite = report.rows.flatMap((r) => r.comps.map((c) => c.gpu))
      .every((v) => Number.isFinite(v));
    check("对拍 GPU 输出全为有限数（无 NaN/Inf）", gpuFinite);
  }
}

// 【本节为什么必须存在，而不是"静态门禁已经查过了"】
// 上一波最严重事故：`hair_scatter.wgsl` 的 a_r/a_tt/a_trt 声明 2/4/4 参却按
// 1/2/2 实参调用 —— 该文件**根本无法通过 createShaderModule**，却因登记为
// LIBRARIES（无入口、不参与任何 kernel 编译）在 F1–F6 全绿下长期存活。
//
// 根因是"编译盲区"：**无入口的库件从不进入任何 createShaderModule**，于是
// 任何静态正则门禁都只能在"文本像是对的"层面打转，永远碰不到真编译器。
// 本节把每一份无入口库件拼一段合成 @compute 入口，送进**真** WebGPU 链路：
//   createShaderModule → getCompilationInfo（断言 0 error）→ createComputePipeline
//   → createBindGroup → 真 dispatchWorkgroups → mapAsync 回读（断言输出全为有限数）
// 静态门禁（wgsl_lint 规则 F18）保证"没有库件被漏登记"，本节保证"登记的都真能跑"。
// 两侧缺一不可：只有静态 → 登记了但没真编；只有真编 → 漏登记的那份永远不跑。
//
// 【GPU 不可用时的口径】与 7b 一致：**不允许静默跳过**。链路失败即判红，
// 因为"库件能不能编译"这件事只有真机能证明。
section("7c. 无入口 WGSL 库件真实编译链路（真 createShaderModule + 真 dispatch）");
{
  const { spawnSync } = await import("node:child_process");
  const libCompilePath = resolve(HERE, "wgsl_lib_compile.mjs");
  const manifestMod = await import("./wgsl_lib_manifest.mjs");
  const expectedFiles = manifestMod.libManifestFiles();

  // 门禁自身的前提：清单与配方集合必须非空，否则本节会"零份库件全绿"——
  // 那是恒真，必须在这里先挡住。
  check("真编译清单非空（否则本节会零份库件恒绿）", expectedFiles.length > 0,
    `${expectedFiles.length} 份`);

  // timeout 给足：起 Xvfb + Chromium + SwiftShader，且要编 2N 份 shader
  // （N 份正向 + N 份注入自证）。与 7b 同量级。
  const proc = spawnSync(process.execPath, [libCompilePath, "--json"], {
    encoding: "utf8",
    timeout: 900000,
    maxBuffer: 64 * 1024 * 1024,
  });
  const stdout = proc.stdout || "";
  let report = null;
  const jsonStart = stdout.indexOf("{");
  if (jsonStart >= 0) {
    try { report = JSON.parse(stdout.slice(jsonStart)); } catch { report = null; }
  }
  if (!report) {
    const errTail = (proc.stderr || "").trim().split("\n").slice(-6).join(" / ");
    check("无入口库件真编译链路跑通（真 createShaderModule + 真 dispatch）", false,
      `exit=${proc.status} signal=${proc.signal || "-"} ${errTail || stdout.trim().split("\n").slice(-3).join(" / ")}`);
  } else {
    check("无入口库件真编译链路跑通（真 createShaderModule + 真 dispatch）", true,
      `adapter=${report.adapter}，每份 dispatch ${report.dispatch} 个 workgroup`);
    check("真编译链路自身报告整体通过", report.ok === true,
      `ok=${report.ok}`);

    // ---- 清单完备性：磁盘上每一份无入口库件都被真编译了 ----
    const gotFiles = (report.results || []).map((r) => r.file).sort();
    const missing = expectedFiles.filter((f) => !gotFiles.includes(f));
    check(`清单内 ${expectedFiles.length} 份无入口库件全部被真编译`,
      missing.length === 0,
      missing.length ? `未编译：${missing.join(", ")}` : gotFiles.join(", "));
    check(`真编译覆盖 ${expectedFiles.length} 份库件（实际 ${gotFiles.length}）`,
      gotFiles.length === expectedFiles.length, `实际 ${gotFiles.length}`);

    // ---- 逐库件：0 error + dispatch 成功 + 输出全为有限数 ----
    for (const r of report.results || []) {
      // error 级诊断必须为 0。这是本节的核心断言：上一波事故正是"有 error 但没人看"。
      check(`库件 ${r.file}：createShaderModule 0 个 error（真编译器判据）`,
        r.errorCount === 0,
        r.errorCount === 0 ? `${r.shaderBytes} 字节` : r.errors.slice(0, 2).join(" | "));
      // warning 记录但不判红：warning 不是缺陷，判红会逼人加豁免，反而变松。
      check(`库件 ${r.file}：warning ${r.warningCount} 条已记录（记录不判红）`,
        Array.isArray(r.warnings), `${r.warningCount} 条`);
      check(`库件 ${r.file}：真 dispatch ${r.dispatched} 个 workgroup 成功（建 pipeline + bind group + 提交）`,
        r.dispatched > 0 && r.stage === "ok",
        `stage=${r.stage} dispatched=${r.dispatched}`);
      // 【为什么 submit 错误要单独断言】dispatch 的 submit 若被验证层判无效
      // （绑定用法/布局非法），`onSubmittedWorkDone()` 仍会 resolve，回读拿到的是
      // 未被覆写的初始零值 —— "输出全为有限数"于是**假绿**。本波真机缺陷
      // （rt_scene / rt_shade 把 uniform 缓冲按 storage 建）正是这样伪装成通过的。
      // 故显式断言"提交本身 0 验证错误"，把"真 dispatch"钉死。
      check(`库件 ${r.file}：dispatch 提交 0 个验证错误（submit 有效性）`,
        !r.submitError,
        r.submitError ? `submit 验证错误：${r.submitError}` : "提交通过验证");
      // 输出全为有限数：NaN/Inf 说明库件在真机上算出了垃圾值，同样是缺陷。
      check(`库件 ${r.file}：回读输出全为有限数（无 NaN/Inf）`,
        (r.nonFinite || []).length === 0,
        (r.nonFinite || []).length ? `非有限：out[${r.nonFinite[0].i}]=${r.nonFinite[0].v}` : `${(r.out || []).length} 个 float`);
    }

    // ---- 非恒真自证：注入一处确定的编译级缺陷，真编译器必须报错 ----
    //
    // 【为什么这一批断言比正向那批更重要】"6 份库件全部 0 error 通过"**不能**
    // 证明这条链路真的能抓错 —— 万一 getCompilationInfo 恒返回空、或
    // createShaderModule 根本没被调用，也会全绿。故必须真的注入缺陷再验证被抓。
    const inj = report.injected || [];
    check(`注入自证覆盖 ${expectedFiles.length} 份库件（实际 ${inj.length}）`,
      inj.length === expectedFiles.length, `实际 ${inj.length}`);
    for (const r of inj) {
      check(`库件 ${r.file}：注入缺陷确已注入源码（注入规则未失效）`,
        r.applied === true, `applied=${r.applied}`);
      check(`库件 ${r.file}：注入后被真编译器拒绝（链路对它有效）`,
        r.caught === true,
        r.caught ? `${r.errorCount} 条 error` : "注入后仍编译通过 —— 这条链路对它无效！");
      // 命中预期：报错必须**指向注入的那处**，而不是碰巧在别处报了个语法错。
      // 只判"报了错"是不够的 —— 抓错了地方等于没抓到目标缺陷。
      check(`库件 ${r.file}：注入报错命中预期位置/符号`,
        r.matched === true,
        r.matched ? (r.errors[0] || "").slice(0, 88) : (r.errors[0] || "无报错").slice(0, 88));
    }
    const injTotal = inj.reduce((n, r) => n + (r.errorCount || 0), 0);
    check(`注入自证共抓到 ${injTotal} 条编译错误（> 0，证明真编译器在工作）`,
      injTotal > 0, `${injTotal} 条`);
  }
}

section("8. 摘要");
console.log(`  通过 ${passed} 项，失败 ${failures.length} 项`);
if (failures.length) {
  console.log("\n失败明细：");
  for (const f of failures) console.log("  - " + f);
  console.log(`\nwasm: ${WASM_PATH}`);
  process.exit(1);
}
console.log(`\n全部通过。wasm: ${WASM_PATH} (${bytes.length} 字节)`);
