#!/usr/bin/env node
// =============================================================================
// measure_size.mjs —— 包体三指标度量 + 阈值判据（零依赖，仅 node 内建模块）
//
// 三个指标（唯一权威定义在本文件，CI 与本地跑的是同一份）：
//   ① 单文件分发包   dist/prism-engine-v*-source.txt
//   ② WebGPU wasm    crates/prism-gpu/web/prism_gpu.wasm
//   ③ native release target/release/prism-app
//
// 设计要点（改动前请先读）：
//   1. **阈值不写在本文件里**，全部落在 tools/size_baseline.json。基线是实测值
//      且带溯源（measured_on / history / rationale）；阈值是数据，可 review。
//      否则阈值会退化成代码里的魔数，"顺手调大一点让它变绿"就没人看得见。
//   2. **越界即红**：--check 模式下任一指标超过上限 ⇒ exit 1。
//   3. **没测到 ≠ 没问题**：CI 用 --require-all，三个产物缺任何一个都判红；
//      本地默认只报（缺产物按 skip 处理），避免"没构建就红"的噪声。
//   4. **只管体积，不管结构**：产物是不是合法 wasm（魔数/导入导出符号）由
//      crates/prism-gpu/web/ci_wasm_verify.mjs 负责。两个脚本不重复断言同一件事。
//   5. **不碰 Cargo.lock**：本脚本不调 cargo，纯读文件大小，不可能改依赖解析结果。
//
// 用法：
//   node tools/measure_size.mjs                          # 只报，不判红
//   node tools/measure_size.mjs --check                  # 判红：越界 exit 1
//   node tools/measure_size.mjs --check --require-all    # 缺失也判红（CI 用）
//   node tools/measure_size.mjs --json size.json         # 机器可读报表
//   node tools/measure_size.mjs --update-baseline        # 把实测值写回基线（只写 baseline_bytes）
//   node tools/measure_size.mjs --self-test              # 反向自证：造假产物必须被判红
//
// 退出码：0 = 全在阈内（或只报模式）；1 = 越界 / 应有产物缺失；2 = 用法或配置错误。
// =============================================================================

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(HERE, '..');
const DEFAULT_BASELINE = path.join(HERE, 'size_baseline.json');

const EXIT_OK = 0;
const EXIT_RED = 1;
const EXIT_USAGE = 2;

/** 用法错误：区别于"指标越界"，退出码也不同（2 而不是 1）。 */
class UsageError extends Error {}

// ---------------------------------------------------------------------------
// 参数解析（手写，与仓库其余 tools/*.mjs 保持一致；不引第三方 arg 解析器）
// ---------------------------------------------------------------------------

/**
 * 取参数值；缺值即用法错误。
 * @param {string[]} argv 原始参数
 * @param {number} index 值所在下标
 * @param {string} flag 参数名（报错用）
 * @returns {string} 参数值
 */
function needValue(argv, index, flag) {
  const value = argv[index];
  if (value === undefined || value.startsWith('--')) {
    throw new UsageError(`参数 ${flag} 缺值`);
  }
  return value;
}

/**
 * 解析命令行。
 * @param {string[]} argv 不含 node/脚本名的参数表
 * @returns {object} 选项对象
 */
function parseArgs(argv) {
  const opts = {
    check: false,
    requireAll: false,
    updateBaseline: false,
    selfTest: false,
    help: false,
    json: null,
    root: DEFAULT_ROOT,
    baseline: DEFAULT_BASELINE,
    source: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case '--check':
        opts.check = true;
        break;
      case '--require-all':
        opts.requireAll = true;
        break;
      case '--update-baseline':
        opts.updateBaseline = true;
        break;
      case '--self-test':
        opts.selfTest = true;
        break;
      case '--json':
        opts.json = path.resolve(needValue(argv, (i += 1), arg));
        break;
      case '--root':
        opts.root = path.resolve(needValue(argv, (i += 1), arg));
        break;
      case '--baseline':
        opts.baseline = path.resolve(needValue(argv, (i += 1), arg));
        break;
      case '--source':
        opts.source = path.resolve(needValue(argv, (i += 1), arg));
        break;
      case '-h':
      case '--help':
        opts.help = true;
        break;
      default:
        throw new UsageError(`未知参数：${arg}`);
    }
  }
  return opts;
}

const USAGE = `measure_size.mjs —— 包体三指标度量 + 阈值判据（零依赖）

用法:
  node tools/measure_size.mjs [选项]

选项:
  --check             判红模式：任一指标越界即退出码 1（默认只报不判红）
  --require-all       三个产物缺任何一个都判红（CI 用；本地默认按 skip 跳过）
  --json <FILE>       把机器可读报表写到 FILE
  --source <FILE>     指定单文件分发包路径（默认在 dist/ 下按修改时间选最新）
  --root <DIR>        仓库根目录（默认本脚本的上一级；--self-test 用它指向临时树）
  --baseline <FILE>   基线文件（默认 tools/size_baseline.json）
  --update-baseline   把本次实测值写回基线文件的 baseline_bytes（不改已声明的上限）
  --self-test         反向自证：造出越界/缺失的假产物，断言判据确实判红
  -h, --help          本帮助

退出码: 0 通过/只报 · 1 越界或应有产物缺失 · 2 用法或配置错误`;

// ---------------------------------------------------------------------------
// 格式化（确定性：不用 toLocaleString，避免跟随运行环境的 locale 变化）
// ---------------------------------------------------------------------------

/**
 * 千分位分隔的整数串。
 * @param {number} n 非负整数
 * @returns {string} 例如 "18,939,405"
 */
function thousands(n) {
  return String(Math.trunc(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * 人类可读体积（十进制到 2 位，单位按 KiB/MiB 阶跃；用 1024 进制并显式标注）。
 * @param {number} n 字节数
 * @returns {string} 例如 "19,150,204 B（18.26 MiB）"
 */
function humanBytes(n) {
  if (!Number.isFinite(n)) return String(n);
  if (n < 1024) return `${thousands(n)} B`;
  if (n < 1024 * 1024) return `${thousands(n)} B（${(n / 1024).toFixed(2)} KiB）`;
  return `${thousands(n)} B（${(n / 1024 / 1024).toFixed(2)} MiB）`;
}

/**
 * 带符号的字节差 / 百分比（相对基线）。
 * @param {number} delta 字节差
 * @param {number} base 基线
 * @returns {string} 例如 "+22,406 B（+6.50%）"
 */
function signedDelta(delta, base) {
  const sign = delta > 0 ? '+' : delta < 0 ? '-' : '±';
  const abs = Math.abs(delta);
  const pct = base > 0 ? ((delta / base) * 100).toFixed(2) : 'n/a';
  return `${sign}${thousands(abs)} B（${delta > 0 ? '+' : delta < 0 ? '-' : '±'}${pct}%）`;
}

/**
 * 把 root 内的绝对路径压成相对路径（报表里路径要短且可复现）。
 * @param {string} root 仓库根
 * @param {string} abs 绝对路径
 * @returns {string} 相对路径；不在 root 内则原样返回
 */
function relPath(root, abs) {
  const rel = path.relative(root, abs);
  return rel.startsWith('..') || path.isAbsolute(rel) ? abs : rel.split(path.sep).join('/');
}

// ---------------------------------------------------------------------------
// 判据核心（纯函数：给定字节数与基线/上限，返回结论。--self-test 直接对它取反）
// ---------------------------------------------------------------------------

/**
 * 计算生效上限：绝对上限与"基线 × (1+相对带)"取更严者。
 *
 * 为什么要两个口径：
 *   - 相对带（max_growth_ratio）管"渐进膨胀"，随基线滚动，不会因一次放宽而永久失去约束力；
 *   - 绝对上限（max_bytes）管"形态失控"（例如误把 native 产物当 wasm、打包范围含入构建产物）。
 * 只留一个都会留缺口，所以两个都支持，并且**取更严者**。
 *
 * @param {{baselineBytes:number, maxBytes?:number, maxGrowthRatio?:number}} spec 判定规格
 * @returns {{capBytes:number, capSource:string, capDetail:string}} 生效上限及其来源
 */
function resolveCap(spec) {
  const hasAbsolute = typeof spec.maxBytes === 'number' && Number.isFinite(spec.maxBytes);
  const hasRatio = typeof spec.maxGrowthRatio === 'number' && Number.isFinite(spec.maxGrowthRatio);
  const absolute = hasAbsolute ? Math.trunc(spec.maxBytes) : null;
  const relative = hasRatio ? Math.floor(spec.baselineBytes * (1 + spec.maxGrowthRatio)) : null;

  if (absolute === null && relative === null) {
    throw new UsageError('基线文件缺少 max_bytes 与 max_growth_ratio（至少要有一个上限）');
  }
  if (absolute !== null && relative !== null) {
    const capBytes = Math.min(absolute, relative);
    return absolute <= relative
      ? {
          capBytes,
          capSource: 'max_bytes',
          capDetail: `绝对上限 ${thousands(absolute)} B 更严（相对带 = 基线 +${(spec.maxGrowthRatio * 100).toFixed(2)}% = ${thousands(relative)} B）`,
        }
      : {
          capBytes,
          capSource: 'max_growth_ratio',
          capDetail: `相对带 基线 +${(spec.maxGrowthRatio * 100).toFixed(2)}% = ${thousands(relative)} B 更严（绝对上限 ${thousands(absolute)} B）`,
        };
  }
  return absolute !== null
    ? { capBytes: absolute, capSource: 'max_bytes', capDetail: `绝对上限 ${thousands(absolute)} B` }
    : {
        capBytes: relative,
        capSource: 'max_growth_ratio',
        capDetail: `相对带 基线 +${(spec.maxGrowthRatio * 100).toFixed(2)}% = ${thousands(relative)} B`,
      };
}

/**
 * 判一个产物的体积。
 *
 * 边界语义：实测 **等于** 上限算通过（阈值是"上限"而不是"必须小于"），
 * 多 1 字节即越界。这条边界由 --self-test 钉住（cap 与 cap+1 各断言一次）。
 *
 * @param {number} bytes 实测字节数
 * @param {{baselineBytes:number, maxBytes?:number, maxGrowthRatio?:number}} spec 判定规格
 * @returns {object} 判定结果（status ∈ {'ok','over'}）
 */
function judge(bytes, spec) {
  const { capBytes, capSource, capDetail } = resolveCap(spec);
  const deltaBytes = bytes - spec.baselineBytes;
  return {
    status: bytes > capBytes ? 'over' : 'ok',
    bytes,
    baselineBytes: spec.baselineBytes,
    capBytes,
    capSource,
    capDetail,
    deltaBytes,
    deltaRatio: spec.baselineBytes > 0 ? deltaBytes / spec.baselineBytes : null,
    headroomBytes: capBytes - bytes,
  };
}

// ---------------------------------------------------------------------------
// 产物定位
// ---------------------------------------------------------------------------

/** 单文件分发包的命名形态（含历史名 v2.0-urgent_fix-source.txt）。 */
const SOURCE_TXT_RE = /^prism-engine-v.+-source\.txt$/;

/**
 * 源码根目录：仓库布局是 crates/，单文件分发包布局是 src/（与 CI 的
 * detect source root 步骤、tools/hygiene_scan.mjs 的 localize() 同一套约定）。
 * 基线文件里记的是仓库布局（crates/…），落到实际文件路径时按存在的目录重映射——
 * 否则在分发包布局里 wasm 指标会被误判成"产物缺失"。
 * @param {string} root 仓库根
 * @param {string} rel 仓库布局下的相对路径
 * @returns {string} 本地化后的相对路径
 */
function localizeSourcePath(root, rel) {
  if (!rel.startsWith('crates/')) return rel;
  const srcRoot = fs.existsSync(path.join(root, 'crates'))
    ? 'crates'
    : fs.existsSync(path.join(root, 'src'))
      ? 'src'
      : 'crates';
  return `${srcRoot}${rel.slice('crates'.length)}`;
}

/**
 * 选单文件分发包：显式 --source 优先，否则在 dist/ 下按修改时间取最新。
 * 按 mtime 而不是按版本号字符串排序：版本号命名不规范（含 urgent_fix 之类后缀），
 * 字符串比较会选错；"最新打出来的那个"才是本次要量的对象。
 * @param {string} root 仓库根
 * @param {string|null} explicit 显式指定的路径
 * @returns {string|null} 绝对路径；找不到返回 null
 */
function selectSourceTxt(root, explicit) {
  if (explicit) return explicit;
  const distDir = path.join(root, 'dist');
  let names;
  try {
    names = fs.readdirSync(distDir);
  } catch {
    return null;
  }
  const candidates = names
    .filter((name) => SOURCE_TXT_RE.test(name))
    .map((name) => {
      const abs = path.join(distDir, name);
      return { abs, mtime: fs.statSync(abs).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);
  return candidates.length > 0 ? candidates[0].abs : null;
}

/**
 * 由指标定义求产物绝对路径。
 * @param {string} key 指标键
 * @param {object} def 指标定义（来自基线文件）
 * @param {object} opts 选项
 * @returns {string|null} 绝对路径
 */
function resolveTarget(key, def, opts) {
  if (key === 'source_txt') return selectSourceTxt(opts.root, opts.source);
  if (typeof def.path !== 'string') throw new UsageError(`指标 ${key} 的基线定义缺少 path`);
  // 只有 native 可执行产物在 Windows 上带 .exe 后缀；.wasm 不带。
  const exe = key === 'native_release' && process.platform === 'win32' ? '.exe' : '';
  return path.join(opts.root, localizeSourcePath(opts.root, def.path) + exe);
}

// ---------------------------------------------------------------------------
// 度量 + 判红
// ---------------------------------------------------------------------------

/**
 * 量一个指标。
 * @param {string} key 指标键
 * @param {object} def 指标定义
 * @param {object} opts 选项
 * @returns {object} 报表行
 */
function measureMetric(key, def, opts) {
  const row = {
    key,
    label: def.label,
    path: null,
    exists: false,
    status: 'missing',
    bytes: null,
    baselineBytes: def.baseline_bytes,
    measuredOn: def.measured_on ?? null,
    capBytes: null,
    capSource: null,
    capDetail: null,
    deltaBytes: null,
    deltaRatio: null,
    headroomBytes: null,
    note: '',
  };

  let target = null;
  try {
    target = resolveTarget(key, def, opts);
  } catch (err) {
    row.status = 'missing';
    row.note = `路径解析失败：${err.message}`;
    return row;
  }
  if (!target) {
    row.status = opts.requireAll ? 'missing' : 'skip';
    row.note = key === 'source_txt'
      ? 'dist/ 下没有分发包（先跑 node tools/pack_source_txt.mjs <版本>）'
      : '产物路径未解析出来';
    return row;
  }
  row.path = relPath(opts.root, target);

  if (!fs.existsSync(target)) {
    row.status = opts.requireAll ? 'missing' : 'skip';
    row.note = '产物不存在（先构建；CI 里由 --require-all 把缺失判红）';
    return row;
  }
  const st = fs.statSync(target);
  if (!st.isFile()) {
    row.status = opts.requireAll ? 'missing' : 'skip';
    row.note = `不是普通文件：${row.path}`;
    return row;
  }

  const verdict = judge(st.size, {
    baselineBytes: def.baseline_bytes,
    maxBytes: def.max_bytes,
    maxGrowthRatio: def.max_growth_ratio,
  });
  Object.assign(row, {
    exists: true,
    status: verdict.status,
    bytes: verdict.bytes,
    capBytes: verdict.capBytes,
    capSource: verdict.capSource,
    capDetail: verdict.capDetail,
    deltaBytes: verdict.deltaBytes,
    deltaRatio: verdict.deltaRatio,
    headroomBytes: verdict.headroomBytes,
  });
  return row;
}

/**
 * 跑全部指标并给出退出码（main 与 --self-test 共用同一条路径）。
 * @param {object} opts 选项
 * @param {{quiet?:boolean}} [io] quiet=true 时不打印（自证用）
 * @returns {{rows:object[], exitCode:number, verdict:string}}
 */
function runCheck(opts, io = {}) {
  const baselineFile = opts.baseline;
  let baseline;
  try {
    baseline = JSON.parse(fs.readFileSync(baselineFile, 'utf8'));
  } catch (err) {
    throw new UsageError(`读不到基线文件 ${baselineFile}：${err.message}`);
  }
  const defs = baseline.metrics;
  if (!defs || typeof defs !== 'object') throw new UsageError('基线文件缺少 metrics');

  const order = ['source_txt', 'wasm', 'native_release'];
  const keys = [...order.filter((k) => k in defs), ...Object.keys(defs).filter((k) => !order.includes(k))];
  const rows = keys.map((key) => measureMetric(key, defs[key], opts));

  const over = rows.filter((r) => r.status === 'over');
  const missing = rows.filter((r) => r.status === 'missing');
  const measured = rows.filter((r) => r.status === 'ok');
  const notMeasured = rows.length - measured.length;
  const exitCode = opts.check && (over.length > 0 || missing.length > 0) ? EXIT_RED : EXIT_OK;
  // 三态：fail（越界 / 该有却没有）/ incomplete（有指标没量到）/ pass（全部量到且在阈内）。
  // incomplete 与 pass 分开是有意的：0 项实测的"全绿"是报表类门禁最典型的假绿。
  const verdict = over.length > 0 || (opts.check && missing.length > 0)
    ? 'fail'
    : notMeasured > 0
      ? 'incomplete'
      : 'pass';

  if (!io.quiet) {
    printReport(rows, opts, { over, missing, verdict });
  }
  if (opts.json && !io.quiet) writeJsonReport(opts, rows, verdict);
  if (opts.updateBaseline && !io.quiet) updateBaseline(baselineFile, baseline, rows);
  return { rows, exitCode, verdict };
}

/**
 * 打印人读报表。逐指标分块而不是画表格：中文与数字混排时表格列宽必然错位，
 * 分块输出的可读性更稳，也更容易在 CI 日志里 grep（每块以 "①②③" 起头）。
 * @param {object[]} rows 报表行
 * @param {object} opts 选项
 * @param {{over:object[], missing:object[], verdict:string}} summary 汇总
 */
function printReport(rows, opts, summary) {
  const BADGES = ['①', '②', '③', '④', '⑤', '⑥'];
  console.log('=== 包体三指标（基线/阈值数据源：tools/size_baseline.json） ===');
  console.log(`模式：${opts.check ? '判红（越界 exit 1）' : '只报（不判红）'}${opts.requireAll ? ' + 要求全部产物存在' : ''}`);
  console.log('');
  rows.forEach((row, idx) => {
    const badge = BADGES[idx] ?? '·';
    console.log(`${badge} ${row.label}`);
    console.log(`   产物 : ${row.path ?? '（未找到）'}`);
    if (row.status === 'missing' || row.status === 'skip') {
      console.log(`   实测 : —`);
      console.log(`   基线 : ${humanBytes(row.baselineBytes)}`);
      console.log(`   结论 : ${row.status === 'missing' ? '缺失（判红）' : '跳过（只报模式）'} —— ${row.note}`);
      console.log('');
      return;
    }
    console.log(`   实测 : ${humanBytes(row.bytes)}`);
    const baselineNote = row.measuredOn && row.measuredOn !== row.path ? `（基线测得于 ${row.measuredOn}）` : '';
    console.log(`   基线 : ${humanBytes(row.baselineBytes)}${baselineNote}`);
    console.log(`   上限 : ${humanBytes(row.capBytes)}  ← ${row.capDetail}`);
    console.log(`   差值 : ${signedDelta(row.deltaBytes, row.baselineBytes)}`);
    console.log(
      row.headroomBytes >= 0
        ? `   余量 : ${thousands(row.headroomBytes)} B（相对上限 ${((row.headroomBytes / row.capBytes) * 100).toFixed(2)}%）`
        : `   超出 : ${thousands(-row.headroomBytes)} B（相对上限 ${((-row.headroomBytes / row.capBytes) * 100).toFixed(2)}%）`,
    );
    console.log(`   结论 : ${row.status === 'ok' ? '在阈内' : '越界（判红）'}`);
    console.log('');
  });

  const okCount = rows.filter((r) => r.status === 'ok').length;
  const skipCount = rows.filter((r) => r.status === 'skip').length;
  console.log(
    `汇总：${okCount}/${rows.length} 项在阈内，越界 ${summary.over.length} 项，`
    + `缺失 ${summary.missing.length} 项，跳过 ${skipCount} 项 → ${summary.verdict.toUpperCase()}`,
  );
  if (summary.verdict === 'incomplete') {
    console.log(`  注意：0 项实测（产物未构建）——只报模式不判红，但**这不等于"体积没问题"**。`);
    console.log('  CI 用 --check --require-all：缺任何一个产物即判红。');
  }
  if (summary.over.length > 0) {
    for (const row of summary.over) {
      console.log(`  !! ${row.label}：${thousands(row.bytes)} B 超过上限 ${thousands(row.capBytes)} B（${row.capDetail}）`);
    }
    console.log('  处理方式：确认增长是否合理 → 改 tools/size_baseline.json 的 baseline_bytes 并写明原因，或修掉膨胀源头。');
  }
  if (summary.missing.length > 0) {
    for (const row of summary.missing) {
      console.log(`  !! ${row.label}：产物缺失（${row.path ?? '路径未解析'}）`);
    }
  }
}

/**
 * 写机器可读报表（CI 存档 / 版本间 diff 用）。
 * @param {object} opts 选项
 * @param {object[]} rows 报表行
 * @param {string} verdict 结论
 */
function writeJsonReport(opts, rows, verdict) {
  const payload = {
    schemaVersion: 1,
    generated: new Date().toISOString(),
    root: opts.root,
    baselineFile: opts.baseline,
    check: opts.check,
    requireAll: opts.requireAll,
    verdict,
    platform: { platform: process.platform, arch: process.arch, node: process.version },
    metrics: rows,
  };
  fs.mkdirSync(path.dirname(opts.json), { recursive: true });
  fs.writeFileSync(opts.json, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  console.log(`报表已写：${opts.json}`);
}

/**
 * 把实测值写回基线的 baseline_bytes（**只写实测值，不改已声明的上限**）。
 * 不自动改上限是有意的：放宽上限必须是人的决定，而不是脚本的副作用。
 * @param {string} baselineFile 基线文件路径
 * @param {object} baseline 已解析的基线对象
 * @param {object[]} rows 报表行
 */
function updateBaseline(baselineFile, baseline, rows) {
  let changed = 0;
  for (const row of rows) {
    if (row.bytes === null) continue;
    const def = baseline.metrics[row.key];
    if (!def) continue;
    if (def.baseline_bytes !== row.bytes) {
      console.log(`基线更新 ${row.key}：${thousands(def.baseline_bytes)} → ${thousands(row.bytes)} B`);
      def.baseline_bytes = row.bytes;
      def.measured_on = row.path;
      changed += 1;
    }
  }
  if (changed === 0) {
    console.log('基线无需更新（实测值与基线一致）。');
    return;
  }
  fs.writeFileSync(baselineFile, `${JSON.stringify(baseline, null, 2)}\n`, 'utf8');
  console.log(`基线已写：${baselineFile}（${changed} 项）`);
}

// ---------------------------------------------------------------------------
// 反向自证：判据本身必须还有判红能力
// ---------------------------------------------------------------------------

/**
 * 自证。造一棵临时"小仓库"，逐情形断言退出码：
 *   1) 三个产物都在阈内          → 0（否则判据恒红，同样不可信）
 *   2) wasm 超绝对上限           → 1（必须判红）
 *   3) 分发包缺失 + --require-all → 1（"没测到"必须判红）
 *   4) 分发包缺失 + 只报模式      → 0（本地不因未构建而红）
 *   5) 边界：bytes == cap → ok；cap+1 → over（阈值是上限，不是"小于"）
 *   6) 双上限取更严者：两个方向各验一次（绝对上限更严 / 相对带更严）
 * 若有人把判据改成恒过（比如为了让它变绿而删掉 over 分支），本自证立刻失败。
 * @returns {number} 退出码（0 = 自证通过）
 */
function selfTest() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'prism-size-selftest-'));
  const results = [];
  const assert = (name, cond, detail) => {
    results.push({ name, ok: Boolean(cond), detail });
    console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? `  —— ${detail}` : ''}`);
    return Boolean(cond);
  };

  try {
    const root = path.join(tmp, 'repo');
    const baselineFile = path.join(tmp, 'baseline.json');
    fs.mkdirSync(path.join(root, 'dist'), { recursive: true });
    fs.mkdirSync(path.join(root, 'crates/prism-gpu/web'), { recursive: true });
    fs.mkdirSync(path.join(root, 'target/release'), { recursive: true });

    const sourceTxt = path.join(root, 'dist/prism-engine-v9.9-source.txt');
    const wasm = path.join(root, 'crates/prism-gpu/web/prism_gpu.wasm');
    const native = path.join(root, 'target/release/prism-app');

    const baseline = {
      schemaVersion: 1,
      metrics: {
        source_txt: { label: '假分发包', selector: 'dist/prism-engine-v*-source.txt', baseline_bytes: 1000, max_growth_ratio: 0.1 },
        wasm: { label: '假 wasm', path: 'crates/prism-gpu/web/prism_gpu.wasm', baseline_bytes: 2000, max_bytes: 2200 },
        native_release: { label: '假 native', path: 'target/release/prism-app', baseline_bytes: 3000, max_growth_ratio: 0.1 },
      },
    };
    fs.writeFileSync(baselineFile, `${JSON.stringify(baseline, null, 2)}\n`, 'utf8');
    fs.writeFileSync(sourceTxt, Buffer.alloc(1050));
    fs.writeFileSync(wasm, Buffer.alloc(2100));
    fs.writeFileSync(native, Buffer.alloc(3100));

    const baseOpts = { check: true, requireAll: true, json: null, root, baseline: baselineFile, source: null };

    // 情形 1：全部在阈内 ⇒ 不判红
    const inBand = runCheck(baseOpts, { quiet: true });
    assert('在阈内 ⇒ exit 0', inBand.exitCode === EXIT_OK, `exit=${inBand.exitCode}`);

    // 情形 2：wasm 超绝对上限 ⇒ 必须判红
    fs.writeFileSync(wasm, Buffer.alloc(2300));
    const oversize = runCheck(baseOpts, { quiet: true });
    assert('wasm 超上限 ⇒ exit 1', oversize.exitCode === EXIT_RED, `exit=${oversize.exitCode}`);
    assert(
      '越界项被标成 over',
      oversize.rows.find((r) => r.key === 'wasm')?.status === 'over',
      oversize.rows.find((r) => r.key === 'wasm')?.status,
    );

    // 情形 3：分发包缺失 + --require-all ⇒ 必须判红
    // （先把 wasm 还原到阈内，否则情形 2 的越界状态会串味，让这条断言失去意义）
    fs.writeFileSync(wasm, Buffer.alloc(2100));
    fs.rmSync(sourceTxt);
    const missingStrict = runCheck(baseOpts, { quiet: true });
    assert('缺失 + --require-all ⇒ exit 1', missingStrict.exitCode === EXIT_RED, `exit=${missingStrict.exitCode}`);

    // 情形 4：分发包缺失 + 只报模式 ⇒ 不判红，但结论必须是 incomplete（不能是 pass）
    const missingLenient = runCheck({ ...baseOpts, check: false }, { quiet: true });
    assert('缺失 + 只报 ⇒ exit 0', missingLenient.exitCode === EXIT_OK, `exit=${missingLenient.exitCode}`);
    assert('缺失 + 只报 ⇒ 结论为 incomplete 而非 pass', missingLenient.verdict === 'incomplete', missingLenient.verdict);

    // 情形 5：边界精确性
    const atCap = judge(2200, { baselineBytes: 2000, maxBytes: 2200 });
    const overCap = judge(2201, { baselineBytes: 2000, maxBytes: 2200 });
    assert('bytes == cap ⇒ ok（阈值是上限不是小于）', atCap.status === 'ok', atCap.status);
    assert('bytes == cap+1 ⇒ over', overCap.status === 'over', overCap.status);

    // 情形 6：双上限取更严者（两个方向）
    const absStricter = resolveCap({ baselineBytes: 1000, maxBytes: 1100, maxGrowthRatio: 0.5 });
    assert('绝对上限更严时以绝对上限为准', absStricter.capBytes === 1100 && absStricter.capSource === 'max_bytes', `${absStricter.capBytes}/${absStricter.capSource}`);
    const ratioStricter = resolveCap({ baselineBytes: 1000, maxBytes: 999999, maxGrowthRatio: 0.05 });
    assert('相对带更严时以相对带为准', ratioStricter.capBytes === 1050 && ratioStricter.capSource === 'max_growth_ratio', `${ratioStricter.capBytes}/${ratioStricter.capSource}`);

    // 情形 7：配置缺上限 ⇒ 用法错误（不允许"没有上限所以永远绿"）
    let configRejected = false;
    try {
      resolveCap({ baselineBytes: 1000 });
    } catch {
      configRejected = true;
    }
    assert('无上限配置被拒绝', configRejected);

    const failed = results.filter((r) => !r.ok);
    console.log(`\n==> measure_size --self-test: ${failed.length === 0 ? 'PASS' : 'FAIL'}（${results.length - failed.length}/${results.length}）`);
    return failed.length === 0 ? EXIT_OK : EXIT_RED;
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

/**
 * 主流程。
 * @param {string[]} argv 参数表
 * @returns {number} 退出码
 */
function main(argv) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (err) {
    console.error(`measure_size: ${err.message}`);
    console.error(USAGE);
    return EXIT_USAGE;
  }
  if (opts.help) {
    console.log(USAGE);
    return EXIT_OK;
  }
  if (opts.selfTest) return selfTest();

  try {
    const { exitCode, verdict } = runCheck(opts);
    if (exitCode !== EXIT_OK) {
      console.error('\n==> measure_size: FAIL（包体越界或应有产物缺失）');
    } else if (verdict === 'incomplete') {
      console.log('\n==> measure_size: PASS（只报模式，0 项实测——产物未构建；这不等于"体积没问题"）');
    } else {
      console.log('\n==> measure_size: PASS');
    }
    return exitCode;
  } catch (err) {
    console.error(`measure_size: ${err.message}`);
    return err instanceof UsageError ? EXIT_USAGE : EXIT_RED;
  }
}

process.exit(main(process.argv.slice(2)));
