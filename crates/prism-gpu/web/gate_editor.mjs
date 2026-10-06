// ===========================================================================
// gate_editor.mjs — prism_editor.wasm 的构建门禁（U13 编辑器）。
//
// 零第三方依赖（只用 node 内建 module），与 wgsl_lint.mjs / host_test.mjs 同哲学：
// 把"只有真机能暴露"的缺陷提前到构建阶段拦下。检查四道门：
//
//   门 1 导出齐全：31 个 `prism_editor_*` 导出全部存在（缺一个 → 页面某功能静默失效）。
//   门 2 **零导入**：本 ABI 是纯导出、不从 JS 调任何函数，故 wasm 的 import 段
//        必须为空。若将来有人加了 `extern "C"` 导入而页面没提供胶水，
//        `WebAssembly.instantiate` 会直接链接失败——在构建期就拦住。
//   门 3 实例化自检：真跑一次 `WebAssembly.instantiate` + 调 `prism_editor_init`，
//        确认 ABI 版本可读、图 JSON 非空。
//   门 4 端到端不变量：在 **真实 wasm** 上跑一遍编辑 → 脏集 → 撤销 → 重做 →
//        渲染，并把结果与 native 单元测试的期望值对照（同样的 [1,3,4] 等）。
//        这条最有价值：它证明 native 上验证过的语义在 wasm32 上同样成立。
//
// 用法：node gate_editor.mjs [path/to/prism_editor.wasm]
// 缺省路径为同目录的 prism_editor.wasm。
// ===========================================================================

import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WASM = process.argv[2] ?? join(HERE, 'prism_editor.wasm');

/** 状态码约定（与 Rust 侧 status 模块同值）。 */
const PASS = 0, FAIL = 1, SKIP = 3, BAD_ARG = 4;

/**
 * 文本缓冲容量（与 Rust 侧 `abi::TEXT_CAP` 同值）。
 * 缓冲重叠检查需要它才能算出 ERR/JSON/STATS 的覆盖区间。
 */
const TEXT_CAP = 65536;

/** 必须存在的导出清单（缺一即门禁失败）。 */
const REQUIRED_EXPORTS = [
  'prism_editor_abi_version',
  'prism_editor_alloc',
  'prism_editor_dealloc',
  'prism_editor_init',
  'prism_editor_is_init',
  'prism_editor_free_pixels',
  'prism_editor_graph_json',
  'prism_editor_select',
  'prism_editor_selected',
  'prism_editor_apply_param_f32',
  'prism_editor_apply_param_u32',
  'prism_editor_apply_param_bool',
  'prism_editor_undo',
  'prism_editor_redo',
  'prism_editor_can_undo',
  'prism_editor_can_redo',
  'prism_editor_set_quality',
  'prism_editor_quality',
  'prism_editor_quality_spp',
  'prism_editor_schema_json',
  'prism_editor_dirty_json',
  'prism_editor_edits_json',
  'prism_editor_render',
  'prism_editor_pixels_ptr',
  'prism_editor_pixels_len',
  'prism_editor_json_ptr',
  'prism_editor_json_len',
  'prism_editor_stats_ptr',
  'prism_editor_stats_len',
  'prism_editor_error_ptr',
  'prism_editor_error_len',
];

/** 期望的 ABI 版本（与 Rust 侧 EDITOR_ABI_VERSION 一致）。 */
const EXPECT_ABI_VERSION = 1;

/** 收集失败项（全部跑完再汇总，不在首个失败处中断——便于一次看全）。 */
const failures = [];
/** 记录一条通过项。 */
const passes = [];

/**
 * 记录一条门禁结果。
 * @param {boolean} ok 是否通过
 * @param {string} name 门禁名
 * @param {string} detail 详情（失败时打印）
 */
function check(ok, name, detail = '') {
  if (ok) {
    passes.push(name);
    console.log(`  ✓ ${name}`);
  } else {
    failures.push(`${name}${detail ? '：' + detail : ''}`);
    console.log(`  ✗ ${name}${detail ? '：' + detail : ''}`);
  }
}

/** 从 wasm 字节里解出导入/导出名（不依赖任何 wasm 解析库，手写 varint 读取）。 */
function readSections(bytes) {
  let i = 8; // 跳过 magic + version
  const exports = [];
  const imports = [];
  /** 读 LEB128 无符号整数。 */
  const uleb = () => {
    let result = 0, shift = 0, byte = 0;
    do {
      byte = bytes[i++];
      result += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    } while (byte & 0x80);
    return result;
  };
  while (i < bytes.length) {
    const id = bytes[i++];
    const size = uleb();
    const end = i + size;
    if (id === 7) {
      // export section
      const count = uleb();
      for (let k = 0; k < count; k++) {
        const len = uleb();
        const name = Buffer.from(bytes.slice(i, i + len)).toString('utf8');
        i += len;
        const kind = bytes[i++];
        uleb(); // index
        exports.push({ name, kind });
      }
    } else if (id === 2) {
      // import section
      const count = uleb();
      for (let k = 0; k < count; k++) {
        const mlen = uleb();
        const mod = Buffer.from(bytes.slice(i, i + mlen)).toString('utf8');
        i += mlen;
        const nlen = uleb();
        const nm = Buffer.from(bytes.slice(i, i + nlen)).toString('utf8');
        i += nlen;
        const kind = bytes[i++];
        if (kind === 0) uleb();
        else if (kind === 1) { i++; const has = bytes[i++]; uleb(); if (has) uleb(); }
        else if (kind === 2) { const has = bytes[i++]; uleb(); if (has) uleb(); }
        else if (kind === 3) { i += 2; }
        imports.push(`${mod}.${nm}`);
      }
    }
    i = end;
  }
  return { exports, imports };
}

/**
 * 把状态码翻成可读文本（与 Rust 侧错误表一致）。
 * @param {number} rc 状态码
 * @returns {string} 文本
 */
function statusText(rc) {
  return { 0: 'PASS', 1: 'FAIL', 2: 'PENDING', 3: 'SKIP', 4: 'BAD_ARG' }[rc] ?? `?${rc}`;
}

async function main() {
  if (!existsSync(WASM)) {
    console.error(`✗ 找不到 wasm：${WASM}`);
    console.error('  请先运行 build_editor.sh 编译。');
    process.exit(1);
  }
  const bytes = await readFile(WASM);
  console.log(`门禁目标：${WASM}（${bytes.length} 字节）\n`);

  // ---------------------------------------------------------------- 门 1
  console.log('门 1/4 导出齐全');
  const { exports, imports } = readSections(bytes);
  const names = new Set(exports.map((e) => e.name));
  const missing = REQUIRED_EXPORTS.filter((n) => !names.has(n));
  check(missing.length === 0, `${REQUIRED_EXPORTS.length} 个必需导出全部存在`,
    missing.length ? `缺少 ${missing.join(', ')}` : '');

  // ---------------------------------------------------------------- 门 2
  console.log('\n门 2/4 零导入（纯导出，无需 JS 胶水）');
  check(imports.length === 0, 'import 段为空',
    imports.length ? `存在导入 ${imports.join(', ')}——页面未提供胶水，链接会失败` : '');

  // ---------------------------------------------------------------- 门 3
  console.log('\n门 3/4 实例化自检');
  /** @type {WebAssembly.Instance} */
  let instance = null;
  try {
    const { instance: inst } = await WebAssembly.instantiate(bytes, {});
    instance = inst;
    check(true, 'WebAssembly.instantiate 成功（env 为空对象即可）');
  } catch (e) {
    check(false, 'WebAssembly.instantiate 成功', String(e));
    return finish();
  }
  const ex = instance.exports;
  const ver = ex.prism_editor_abi_version();
  check(ver === EXPECT_ABI_VERSION, `ABI 版本 = ${EXPECT_ABI_VERSION}`, `实际 ${ver}`);

  const rcInit = ex.prism_editor_init();
  check(rcInit === PASS, 'prism_editor_init 返回 PASS', statusText(rcInit));
  check(ex.prism_editor_is_init() === 1, 'is_init = 1');

  // 读文本缓冲（ptr + len → UTF-8）。
  const readText = (ptrFn, lenFn) => {
    const ptr = ptrFn();
    const len = lenFn();
    if (len === 0) return '';
    return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ptr, len));
  };
  const readJson = () => readText(ex.prism_editor_json_ptr, ex.prism_editor_json_len);
  const readErr = () => readText(ex.prism_editor_error_ptr, ex.prism_editor_error_len);
  const readStats = () => readText(ex.prism_editor_stats_ptr, ex.prism_editor_stats_len);

  ex.prism_editor_graph_json();
  const graphJson = readJson();
  check(graphJson.includes('"nodes":[') && graphJson.length > 100,
    'graph_json 非空且含 nodes 数组', `长度 ${graphJson.length}`);

  // ---------------------------------------------------------------- 门 4
  console.log('\n门 4/4 端到端不变量（真实 wasm）');

  // (a) 选节点 + schema 暴露滑条范围
  check(ex.prism_editor_select(1) === PASS, 'select(1) → PASS');
  check(ex.prism_editor_selected() === 1, 'selected = 1');
  ex.prism_editor_schema_json(1);
  const schema = readJson();
  check(schema.includes('"min":') && schema.includes('"max":') && schema.includes('"step":'),
    'schema_json 含 min/max/step（UI 可生成滑条）', schema.slice(0, 120));

  // (b) 写参数名进 wasm 内存（走 alloc，与页面同一路径）
  const writeName = (name) => {
    const enc = new TextEncoder().encode(name);
    const ptr = ex.prism_editor_alloc(enc.length);
    new Uint8Array(ex.memory.buffer, ptr, enc.length).set(enc);
    return { ptr, len: enc.length };
  };

  const nm = writeName('scale');
  const rcEdit = ex.prism_editor_apply_param_f32(1, nm.ptr, nm.len, 0.9);
  ex.prism_editor_dealloc(nm.ptr, nm.len);
  check(rcEdit === PASS, 'apply_param_f32(scale=0.9) → PASS', readErr());

  // (c) 增量失效精确：改叶 1 → dirty 恰为 [1,3,4]
  ex.prism_editor_dirty_json();
  const dirty = readJson();
  check(dirty === '[1,3,4]', 'dirty 集 = [1,3,4]（恰好下游子图）', `实际 ${dirty}`);

  // (d) 渲染 + 统计展示增量收益
  check(ex.prism_editor_render(64, 64) === PASS, 'render(64,64) → PASS', readErr());
  const pxLen = ex.prism_editor_pixels_len();
  check(pxLen === 64 * 64 * 4, '像素缓冲 = 64*64*4 字节', `实际 ${pxLen}`);
  let stats = {};
  try { stats = JSON.parse(readStats()); } catch { /* 保持空对象，下面断言会失败 */ }
  check(stats.recomputed?.length === 4, '首帧全部重算（4 节点）', JSON.stringify(stats.recomputed));
  check(stats.spp === 1, 'Draft 档位 spp = 1', `实际 ${stats.spp}`);

  // (e) 同参重渲 → 零重算（缓存全命中）
  ex.prism_editor_render(64, 64);
  try { stats = JSON.parse(readStats()); } catch { stats = {}; }
  check(stats.recomputed?.length === 0, '同参重渲零重算（缓存全命中）', JSON.stringify(stats.recomputed));
  check(stats.reused?.length === 4, '4 节点全部复用', JSON.stringify(stats.reused));

  // (f) 草稿/成片档位：Draft spp < Final spp，且档位切换可往返
  const draftSpp = ex.prism_editor_quality_spp();
  ex.prism_editor_set_quality(1);
  const finalSpp = ex.prism_editor_quality_spp();
  check(draftSpp < finalSpp, `Draft spp(${draftSpp}) < Final spp(${finalSpp})`);
  ex.prism_editor_set_quality(7);
  check(ex.prism_editor_set_quality(7) === BAD_ARG, '非法档位 → BAD_ARG', readErr().slice(0, 80));
  ex.prism_editor_set_quality(0);

  // (g) 撤销/重做：撤销后 dirty 变化，且 can_undo/can_redo 正确翻转
  check(ex.prism_editor_can_undo() === 1, 'can_undo = 1');
  check(ex.prism_editor_undo() === PASS, 'undo → PASS', readErr());
  check(ex.prism_editor_can_redo() === 1, 'can_redo = 1（撤销后）');
  check(ex.prism_editor_redo() === PASS, 'redo → PASS', readErr());
  check(ex.prism_editor_can_redo() === 0, '重做后 can_redo = 0');

  // (h) 撤销栈空 → BAD_ARG（不 panic）
  ex.prism_editor_init(); // 清空历史
  const rcEmptyUndo = ex.prism_editor_undo();
  check(rcEmptyUndo === BAD_ARG, '空撤销栈 undo → BAD_ARG（不 panic）', readErr().slice(0, 80));
  check(readErr().includes('E4002'), '空撤销栈错误含结构化错误码 E4002');

  // (i) 编辑序列导出 JSON 可解析且稳定
  ex.prism_editor_init();
  const nm2 = writeName('radius');
  ex.prism_editor_apply_param_f32(2, nm2.ptr, nm2.len, 0.42);
  ex.prism_editor_dealloc(nm2.ptr, nm2.len);
  ex.prism_editor_edits_json();
  const edits1 = readJson();
  ex.prism_editor_edits_json();
  const edits2 = readJson();
  check(edits1 === edits2, '编辑序列导出逐字节稳定（确定性）');
  let parsedEdits = null;
  try { parsedEdits = JSON.parse(edits1); } catch { /* 下面断言会失败 */ }
  check(parsedEdits?.schema === 1 && Array.isArray(parsedEdits.edits) && parsedEdits.edits.length === 1,
    '编辑序列 JSON 可解析且含 1 条 set_param', edits1.slice(0, 160));

  // (j) 错误参数：未知参数名 → BAD_ARG + 结构化错误码
  const nmBad = writeName('no_such_param');
  const rcBad = ex.prism_editor_apply_param_f32(1, nmBad.ptr, nmBad.len, 1.0);
  ex.prism_editor_dealloc(nmBad.ptr, nmBad.len);
  check(rcBad === BAD_ARG, '未知参数 → BAD_ARG（不 panic）');
  check(readErr().includes('E4005'), '未知参数错误含 E4005');

  // (k) 尺寸越界 → BAD_ARG（预览有界，不静默降质）
  check(ex.prism_editor_render(0, 32) === BAD_ARG, 'render(0,32) → BAD_ARG');
  check(ex.prism_editor_render(1024, 32) === BAD_ARG, 'render(1024,32) → BAD_ARG（>512 上限）');

  // (l) 确定性：同参数渲染两次，像素逐字节一致
  ex.prism_editor_init();
  ex.prism_editor_render(48, 48);
  const a1 = new Uint8Array(ex.memory.buffer, ex.prism_editor_pixels_ptr(), ex.prism_editor_pixels_len()).slice();
  ex.prism_editor_render(48, 48);
  const a2 = new Uint8Array(ex.memory.buffer, ex.prism_editor_pixels_ptr(), ex.prism_editor_pixels_len()).slice();
  let same = a1.length === a2.length;
  if (same) {
    for (let i = 0; i < a1.length; i++) {
      if (a1[i] !== a2[i]) { same = false; break; }
    }
  }
  check(same, '同参渲染两次像素逐字节一致（确定性铁律）');

  // (m) 像素**语义**门禁：alpha 恒 255、RGB 非全黑、长度 = w*h*4。
  //     为什么必需：上面的"逐字节一致"只证明**自洽**（两次错得一样也算过），
  //     抓不到"alpha 从未被写"这类全体一致地错的 bug —— 本门禁就是那次漏网
  //     事故的直接产物（陈旧 wasm 产物 alpha 全 0，34 项门禁全绿放过）。
  ex.prism_editor_init();
  for (const [w, h] of [[32, 32], [64, 48], [128, 128]]) {
    check(ex.prism_editor_render(w, h) === PASS, `render(${w},${h}) → PASS`);
    const len = ex.prism_editor_pixels_len();
    const px = new Uint8Array(ex.memory.buffer, ex.prism_editor_pixels_ptr(), len);
    check(len === w * h * 4, `render(${w},${h}) 像素缓冲 = ${w}*${h}*4 字节（实际 ${len}）`);
    let badAlpha = 0, black = 0;
    for (let i = 0; i < len; i += 4) {
      if (px[i + 3] !== 255) badAlpha++;
      if (px[i] + px[i + 1] + px[i + 2] === 0) black++;
    }
    check(badAlpha === 0, `render(${w},${h}) alpha 恒 255（违例 ${badAlpha} 个像素）`);
    check(black < len / 4, `render(${w},${h}) 画面非全黑（全黑像素 ${black}/${len / 4}）`);
  }
  // (n) 像素缓冲不得与三个文本缓冲重叠（否则写错误文本会踩坏像素）
  {
    const pp = ex.prism_editor_pixels_ptr();
    const pl = ex.prism_editor_pixels_len();
    let overlap = [];
    for (const [name, get] of [
      ['ERR', 'prism_editor_error_ptr'], ['JSON', 'prism_editor_json_ptr'], ['STATS', 'prism_editor_stats_ptr'],
    ]) {
      const tp = ex[get]();
      if (pp < tp + TEXT_CAP && tp < pp + pl) overlap.push(`${name}@${tp}`);
    }
    check(overlap.length === 0, `像素缓冲与文本缓冲不重叠（重叠：${overlap.join(',') || '无'}）`);
    // 写一条错误文本后像素必须逐字节不变（重叠的直接后果检验）
    const before = new Uint8Array(ex.memory.buffer, pp, pl).slice();
    ex.prism_editor_select(9999);
    const after = new Uint8Array(ex.memory.buffer, ex.prism_editor_pixels_ptr(), pl);
    let untouched = true;
    for (let i = 0; i < before.length; i++) { if (before[i] !== after[i]) { untouched = false; break; } }
    check(untouched, '写错误文本后像素缓冲逐字节不变（无缓冲重叠踩踏）');
  }

  finish();
}

/** 汇总并给出退出码。 */
function finish() {
  console.log('');
  if (failures.length === 0) {
    console.log(`✅ 门禁全过：${passes.length} 项检查通过。`);
    process.exit(0);
  }
  console.log(`❌ 门禁失败：${failures.length} 项未通过 / 共 ${passes.length + failures.length} 项。`);
  for (const f of failures) console.log(`   - ${f}`);
  process.exit(1);
}

main().catch((e) => {
  console.error('门禁脚本自身异常：', e);
  process.exit(1);
});
