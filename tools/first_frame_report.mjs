#!/usr/bin/env node
// =============================================================================
// first_frame_report.mjs —— 首帧时间趋势报表（零依赖，仅 node 内建模块）
//
// 【口径：报表，不是判据】
//   本脚本**永远不因"变慢"判红**。首帧时间是环境相关的：同一份代码在
//   不同 CPU / 不同负载 / 不同容器配额上差 2 倍是常态，把它做成门禁只会
//   逼人学会忽略红色。故：慢 → 报表可见 + 结论写"变慢"，退出码仍是 0。
//   这条性质由 --self-test 钉住（合成 10 倍回归必须仍然 exit 0）。
//   正确性门禁在别处：cargo test / wasm 结构校验 / 浏览器 harness。
//
// 【为什么固定场景 + 固定尺寸】
//   只有把"输入"钉死，数字才可比。固定项写在本文件的 FIXED 常量里，随报表
//   一起输出；--compare 会先校验两份报表的固定项与机器指纹是否一致——不一致
//   就直接标注"不可直接比较"，而不是给出一个看着像结论的比值。
//
//   网络/磁盘冷缓存不可控，所以每轮都是"新进程"（native）或"新读一遍字节"
//   （wasm），取多轮中位数 + MAD 作噪声带；单轮数字不上报。
//
// 【两个轴】
//   A) native 首帧：全新进程 → 读 DSL → 编译 → 渲染单帧 → 落盘 PNG。
//      量到两段：wall_ms（node 侧端到端，含进程创建/动态链接/退出）与
//      internal_ms（引擎自报的"总耗时"，去掉进程开销，噪声更低）。
//   B) wasm 冷启动代理：读 prism_gpu.wasm → WebAssembly.Module（同步编译）
//      → Instance（自动生成 env 桩）。**这不是渲染首帧**：wasm 侧出首帧需要
//      真实 GPU，沙箱里没有。故本轴只度量可离线复现的"加载+编译+实例化"，
//      标签写清"冷启动代理"，不冒充首帧。
//
// 用法：
//   node tools/first_frame_report.mjs                       # 出报表（人读）
//   node tools/first_frame_report.mjs --json ff-a.json      # 存机器可读报表
//   node tools/first_frame_report.mjs --label A             # 打标签，便于 A/B 配对
//   node tools/first_frame_report.mjs --compare ff-a.json ff-b.json   # A/B 对比
//   node tools/first_frame_report.mjs --self-test           # 自证（含"非门禁"钉子）
//
// 退出码：0 = 至少一个轴测到（无论快慢）；2 = 用法错误；3 = 一个轴都没测到
//         （环境不具备：没有 release 二进制 / 没有 wasm 产物）。
//         **不存在代表"太慢"的退出码。**
// =============================================================================

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(HERE, '..');

const EXIT_OK = 0;
const EXIT_SELF_TEST_FAILED = 1;
const EXIT_USAGE = 2;
const EXIT_NO_SAMPLE = 3;

/** 用法错误（退出码 2，与"没测到"3 区分）。 */
class UsageError extends Error {}

/**
 * 固定测量条件（唯一权威定义；报表与 --compare 都引用它）。
 * 改这里等于改口径：历史报表将标记为"不可直接比较"。
 */
const FIXED = {
  scene: 'examples/spin_red_ball.prism',
  width: 640,
  height: 360,
  frame: 0,
  runs: 15,
  wasmRounds: 5,
};

/**
 * 可调项（--runs 只改采样轮数，不改场景/尺寸/时刻；轮数会记入报表参与可比性校验）。
 * @param {string[]} argv 参数表
 * @returns {object} 选项
 */
function parseArgs(argv) {
  const opts = {
    label: 'local',
    json: null,
    compare: null,
    runs: FIXED.runs,
    root: DEFAULT_ROOT,
    selfTest: false,
    help: false,
  };
  const needValue = (index, flag) => {
    const value = argv[index];
    if (value === undefined || value.startsWith('--')) throw new UsageError(`参数 ${flag} 缺值`);
    return value;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case '--label':
        opts.label = needValue((i += 1), arg);
        break;
      case '--json':
        opts.json = path.resolve(needValue((i += 1), arg));
        break;
      case '--compare': {
        const first = path.resolve(needValue((i += 1), arg));
        const second = path.resolve(needValue((i += 1), arg));
        opts.compare = [first, second];
        break;
      }
      case '--runs': {
        const raw = needValue((i += 1), arg);
        const value = Number(raw);
        if (!Number.isInteger(value) || value < 3 || value > 200) {
          throw new UsageError(`--runs 需为 3..200 的整数（收到 ${raw}）`);
        }
        opts.runs = value;
        break;
      }
      case '--root':
        opts.root = path.resolve(needValue((i += 1), arg));
        break;
      case '--self-test':
        opts.selfTest = true;
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

const USAGE = `first_frame_report.mjs —— 首帧时间趋势报表（只出趋势，不作门禁）

用法:
  node tools/first_frame_report.mjs [选项]

选项:
  --label <NAME>          报表标签（默认 local；A/B 配对时建议 A / B）
  --json <FILE>           写机器可读报表
  --compare <A> <B>       对比两份已存的报表（A=基线，B=本次）；恒不判红
  --runs <N>              采样轮数 3..200（默认 ${FIXED.runs}）
  --root <DIR>            仓库根目录（默认本脚本上一级）
  --self-test             自证：统计函数、输出解析、"慢也不判红"三项
  -h, --help              本帮助

固定条件（改本文件 FIXED 常量等于改口径）:
  场景 ${FIXED.scene} · 尺寸 ${FIXED.width}x${FIXED.height} · 时刻 t=${FIXED.frame}s（单帧）

退出码: 0 已测到（无论快慢） · 2 用法错误 · 3 一个轴都没测到（环境不具备）`;

// ---------------------------------------------------------------------------
// 统计（纯函数，便于自证）
// ---------------------------------------------------------------------------

/**
 * 排序副本（不改入参）。
 * @param {number[]} xs 样本
 * @returns {number[]} 升序副本
 */
function sorted(xs) {
  return [...xs].sort((a, b) => a - b);
}

/**
 * 中位数；偶数个取中间两值的平均。
 * @param {number[]} xs 样本（非空）
 * @returns {number} 中位数
 */
function median(xs) {
  if (xs.length === 0) throw new UsageError('median: 空样本');
  const s = sorted(xs);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * 分位数（线性插值）。p90 用来说明尾巴有多长——中位数相同、尾巴不同，
 * 体验差别很大，所以两个都报。
 * @param {number[]} xs 样本（非空）
 * @param {number} p 0..1
 * @returns {number} 分位数值
 */
function percentile(xs, p) {
  if (xs.length === 0) throw new UsageError('percentile: 空样本');
  if (!(p >= 0 && p <= 1)) throw new UsageError(`percentile: p 越界（${p}）`);
  const s = sorted(xs);
  const pos = (s.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/**
 * 中位绝对偏差（MAD）：比标准差更耐离群点，用来定"噪声带"。
 * @param {number[]} xs 样本（非空）
 * @returns {number} MAD
 */
function mad(xs) {
  const m = median(xs);
  return median(xs.map((x) => Math.abs(x - m)));
}

/**
 * 汇总一组样本。
 * @param {number[]} raw 原始样本（毫秒）
 * @returns {object} 统计量 + 原始样本（原始值一并存，便于事后复算）
 */
function summarize(raw) {
  if (raw.length === 0) return { n: 0 };
  const rounded = raw.map((x) => Math.round(x * 1000) / 1000);
  return {
    n: rounded.length,
    median: Math.round(median(rounded) * 1000) / 1000,
    p90: Math.round(percentile(rounded, 0.9) * 1000) / 1000,
    min: Math.round(Math.min(...rounded) * 1000) / 1000,
    max: Math.round(Math.max(...rounded) * 1000) / 1000,
    mad: Math.round(mad(rounded) * 1000) / 1000,
    raw: rounded,
  };
}

/**
 * 解析引擎自报的"总耗时"行（Rust `Duration` 的 Debug 格式）。
 * 输出格式一旦漂移，这里会返回 null —— 上层据此只报 wall_ms 并标注"内部耗时不可用"，
 * 而不是静默拿不到数还装作测到了。
 * @param {string} stdout 子进程标准输出
 * @returns {number|null} 毫秒；解析不出返回 null
 */
function parseElapsedMs(stdout) {
  if (typeof stdout !== 'string') return null;
  const line = stdout.split('\n').find((l) => l.includes('总耗时'));
  if (!line) return null;
  // 顺序要紧：先匹配 ms，再匹配 s —— 否则 "4.443ms" 会被 s 规则吃掉尾字母。
  const match = /总耗时\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|µs|us|ns|s)\b/.exec(line);
  if (!match) return null;
  const value = Number(match[1]);
  switch (match[2]) {
    case 'ms':
      return value;
    case 'µs':
    case 'us':
      return value / 1000;
    case 'ns':
      return value / 1e6;
    case 's':
      return value * 1000;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// 机器指纹（跨机不可比的判据）
// ---------------------------------------------------------------------------

/**
 * 采集机器指纹。两份报表的指纹不同 ⇒ 时间差异里混进了换机器的因素，
 * --compare 会显式标注"不可直接比较"。
 * @returns {object} 指纹
 */
function fingerprint() {
  const cpus = os.cpus();
  return {
    platform: process.platform,
    arch: process.arch,
    cpuModel: (cpus[0]?.model ?? 'unknown').trim().replace(/\s+/g, ' '),
    cpuCount: cpus.length,
    totalMemGiB: Math.round((os.totalmem() / 1024 ** 3) * 10) / 10,
    node: process.version,
  };
}

// ---------------------------------------------------------------------------
// 轴 A：native 首帧（全新进程）
// ---------------------------------------------------------------------------

/**
 * 量 native 首帧：每轮起一个全新进程，等于"冷启动 + 编译 + 单帧渲染 + 落盘"。
 * 输出目录用临时目录，绝不写进仓库（output/ 不在忽略名单里，写进去会污染工作区）。
 * @param {object} opts 选项
 * @returns {object} 轴记录（status ∈ {'ok','skip','error'}）
 */
function measureNativeAxis(opts) {
  const exe = process.platform === 'win32' ? 'prism-app.exe' : 'prism-app';
  const bin = path.join(opts.root, 'target', 'release', exe);
  const scene = path.join(opts.root, FIXED.scene);
  const base = {
    key: 'native_first_frame',
    label: 'native 首帧（进程冷启动 → DSL 编译 → 单帧渲染 → 落盘）',
    unit: 'ms',
    subject: `target/release/${exe} + ${FIXED.scene} @ ${FIXED.width}x${FIXED.height}, t=${FIXED.frame}s`,
    status: 'skip',
    samples: {},
    note: '每轮都是全新进程（冷启动口径）；首轮含二进制页缓存冷读，故取中位数而非均值',
    environment: {},
  };
  if (!fs.existsSync(bin)) {
    return { ...base, note: '未找到 native release 二进制（先 cargo build --release -p prism-app）' };
  }
  if (!fs.existsSync(scene)) {
    return { ...base, note: `固定场景文件不存在：${FIXED.scene}` };
  }
  if (process.platform === 'win32') {
    base.note = 'Windows 下进程创建开销占比更高，跨平台不要直接比数值';
  }

  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'prism-first-frame-'));
  const wall = [];
  const internal = [];
  try {
    for (let i = 0; i < opts.runs; i += 1) {
      const started = process.hrtime.bigint();
      const res = spawnSync(
        bin,
        [
          FIXED.scene,
          '--frame', String(FIXED.frame),
          '--width', String(FIXED.width),
          '--height', String(FIXED.height),
          '--out', outDir,
        ],
        { cwd: opts.root, encoding: 'utf8', timeout: 300000 },
      );
      const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
      if (res.error) {
        return { ...base, status: 'error', note: `子进程启动失败：${res.error.message}` };
      }
      if (res.status !== 0) {
        const stderr = (res.stderr ?? '').trim().split('\n').slice(0, 3).join(' | ');
        return { ...base, status: 'error', note: `首帧渲染失败（退出码 ${res.status}）：${stderr}` };
      }
      wall.push(wallMs);
      const internalMs = parseElapsedMs(res.stdout);
      if (internalMs !== null) internal.push(internalMs);
    }
  } finally {
    fs.rmSync(outDir, { recursive: true, force: true });
  }

  const samples = { wall_ms: summarize(wall) };
  if (internal.length === opts.runs) {
    samples.internal_ms = summarize(internal);
  } else {
    base.note = `${base.note}${base.note ? '；' : ''}内部耗时行解析失败（${internal.length}/${opts.runs} 轮），只报 wall_ms`;
  }
  return { ...base, status: 'ok', samples };
}

// ---------------------------------------------------------------------------
// 轴 B：wasm 冷启动代理（加载 → 编译 → 实例化）
// ---------------------------------------------------------------------------

/**
 * 由模块的导入清单生成 env 桩：只满足实例化所需的"符号存在"，不被调用。
 * 沙箱无 GPU，本轴刻意不进入渲染路径——它量的是加载侧，不是首帧。
 * @param {WebAssembly.Module} mod 已编译模块
 * @returns {{importObject:object, unsupported:string[]}} 桩与不支持的导入种类
 */
function buildImportObject(mod) {
  const importObject = {};
  const unsupported = [];
  for (const imp of WebAssembly.Module.imports(mod)) {
    importObject[imp.module] = importObject[imp.module] ?? {};
    if (imp.kind === 'function') {
      importObject[imp.module][imp.name] = () => 0;
    } else {
      unsupported.push(`${imp.module}.${imp.name}:${imp.kind}`);
    }
  }
  return { importObject, unsupported };
}

/**
 * 源码根目录：仓库布局是 crates/，单文件分发包布局是 src/（与 CI 的
 * detect source root 步骤、tools/hygiene_scan.mjs 的 localize() 同一套约定）。
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
 * 量 wasm 冷启动代理：每轮重新读字节 + 重新编译 + 重新实例化。
 * 读文件走的是页缓存（不是真磁盘冷读），口径写进 note，避免被当成 I/O 基准。
 * @param {object} opts 选项
 * @returns {object} 轴记录
 */
function measureWasmAxis(opts) {
  const wasmRel = localizeSourcePath(opts.root, 'crates/prism-gpu/web/prism_gpu.wasm');
  const wasmPath = path.join(opts.root, wasmRel);
  const base = {
    key: 'wasm_cold_start',
    label: 'wasm 冷启动代理（读字节 → 编译 → 实例化；**非**渲染首帧）',
    unit: 'ms',
    subject: `${wasmRel}（env 导入自动打桩，不进入渲染路径）`,
    status: 'skip',
    samples: {},
    note: '读文件走页缓存，不是真磁盘冷读；wasm 侧出首帧需真实 GPU，本轴不冒充首帧；丢弃 1 轮预热（V8 首次编译的代码缓存/堆增长会让首轮显著偏慢）',
    environment: {},
  };
  if (!fs.existsSync(wasmPath)) {
    return { ...base, note: '未找到 wasm 产物（先 crates/prism-gpu/web/build.sh）' };
  }

  const compile = [];
  const instantiate = [];
  const total = [];
  let bytes = 0;
  let unsupported = [];
  try {
    const warmup = 1; // 首轮只热，不进样本：否则中位数会被单次冷编译拖偏
    for (let i = 0; i < FIXED.wasmRounds + warmup; i += 1) {
      const t0 = process.hrtime.bigint();
      const buf = fs.readFileSync(wasmPath);
      bytes = buf.length;
      const mod = new WebAssembly.Module(buf); // 同步编译：等价于冷启动里的编译段
      const t1 = process.hrtime.bigint();
      const { importObject, unsupported: bad } = buildImportObject(mod);
      unsupported = bad;
      // eslint-disable-next-line no-new -- 只为触发实例化（量的是构造耗时，不是返回值）
      new WebAssembly.Instance(mod, importObject);
      const t2 = process.hrtime.bigint();
      if (i < warmup) continue;
      compile.push(Number(t1 - t0) / 1e6);
      instantiate.push(Number(t2 - t1) / 1e6);
      total.push(Number(t2 - t0) / 1e6);
    }
  } catch (err) {
    return { ...base, status: 'error', note: `wasm 加载/编译/实例化失败：${err.message}` };
  }

  const note = unsupported.length > 0
    ? `${base.note}；存在非函数导入（${unsupported.join(', ')}），实例化可能不代表真实宿主`
    : base.note;
  return {
    ...base,
    status: 'ok',
    note,
    environment: { wasmBytes: bytes },
    samples: {
      compile_ms: summarize(compile),
      instantiate_ms: summarize(instantiate),
      total_ms: summarize(total),
    },
  };
}

// ---------------------------------------------------------------------------
// 退出码（唯一来源：只与"是否测到"有关，与快慢无关）
// ---------------------------------------------------------------------------

/**
 * 报表退出码：有任一样本 ⇒ 0；全都没有 ⇒ 3（环境不具备）。
 * **刻意不读任何时间数值** —— 这是"不作 Go/No-Go"的落点，自证里会钉住。
 * @param {{axes:object[]}} report 报表
 * @returns {number} 退出码
 */
function reportExitCode(report) {
  const measured = (report.axes ?? []).some((axis) => axis.status === 'ok' && axis.samples
    && Object.values(axis.samples).some((s) => s && s.n > 0));
  return measured ? EXIT_OK : EXIT_NO_SAMPLE;
}

// ---------------------------------------------------------------------------
// A/B 对比（纯函数；恒不判红）
// ---------------------------------------------------------------------------

/**
 * 绝对差下限（毫秒）与"小量级"阈值：**只对亚毫秒/个位数毫秒的样本生效**。
 * 理由：0.15→0.12 ms 是 20%，但对任何决策都无关，把这种数字写成"显著性"会
 * 训练读者忽略报表；而 18→20 ms 这种量级的比较仍应走相对噪声带。
 */
const ABS_FLOOR_MS = 0.5;
const SMALL_SCALE_MS = 5;

/**
 * 对比两份报表：逐轴逐样本给中位数比值 + 噪声带结论。
 *
 * 判定顺序（先绝对后相对，避免把小数字的相对抖动写成显著性）：
 *   1) 小量级样本（两侧中位数都 < 5 ms）且 |B − A| < 0.5 ms → "持平（亚毫秒级）"
 *   2) |B/A − 1| ≤ max(2×MAD/中位数, 2%)                   → "持平（噪声带内）"
 *   3) 其余按方向记 "变慢" / "变快"
 * 比值只做趋势提示，不产生退出码。
 *
 * @param {object} reportA 基线报表（A）
 * @param {object} reportB 本次报表（B）
 * @returns {object} {sameMachine, sameFixed, rows, notes, exitCode}
 */
function compareReports(reportA, reportB) {
  const notes = [];
  const sameMachine = JSON.stringify(reportA.fingerprint) === JSON.stringify(reportB.fingerprint);
  const fixedKey = (r) => JSON.stringify({ ...r.fixed, runs: r.fixed?.runs });
  const sameFixed = fixedKey(reportA) === fixedKey(reportB);
  if (!sameMachine) {
    notes.push('机器指纹不同（CPU/内存/平台/node 版本任一不同）⇒ 差异里混入了换机器因素，不可直接比较');
  }
  if (!sameFixed) {
    notes.push('固定测量条件不同（场景/尺寸/时刻/轮数）⇒ 不可直接比较');
  }

  const rows = [];
  const axesA = new Map((reportA.axes ?? []).map((a) => [a.key, a]));
  for (const axisB of reportB.axes ?? []) {
    const axisA = axesA.get(axisB.key);
    if (!axisA) {
      notes.push(`轴 ${axisB.key} 只出现在 B 报表里，无法对比`);
      continue;
    }
    for (const [sampleKey, sampleB] of Object.entries(axisB.samples ?? {})) {
      const sampleA = axisA.samples?.[sampleKey];
      if (!sampleA || sampleA.n === 0 || sampleB.n === 0) continue;
      const ratio = sampleB.median / sampleA.median;
      const absDelta = Math.abs(sampleB.median - sampleA.median);
      const band = Math.max(
        (2 * Math.max(sampleA.mad ?? 0, sampleB.mad ?? 0)) / Math.min(sampleA.median, sampleB.median),
        0.02,
      );
      let verdict;
      if (Math.min(sampleA.median, sampleB.median) < SMALL_SCALE_MS && absDelta < ABS_FLOOR_MS) {
        verdict = '持平（亚毫秒级）';
      } else if (ratio > 1 + band) {
        verdict = '变慢';
      } else if (ratio < 1 - band) {
        verdict = '变快';
      } else {
        verdict = '持平（噪声带内）';
      }
      rows.push({
        axis: axisB.key,
        axisLabel: axisB.label,
        sample: sampleKey,
        medianA: sampleA.median,
        medianB: sampleB.median,
        absDeltaMs: Math.round(absDelta * 1000) / 1000,
        ratio: Math.round(ratio * 1000) / 1000,
        band: Math.round(band * 10000) / 10000,
        deltaPct: Math.round((ratio - 1) * 10000) / 100,
        verdict,
      });
    }
  }
  return { sameMachine, sameFixed, rows, notes, exitCode: EXIT_OK };
}

/**
 * 打印对比结果。
 * @param {object} cmp compareReports 的返回
 * @param {object} reportA 基线报表
 * @param {object} reportB 本次报表
 */
function printComparison(cmp, reportA, reportB) {
  console.log(`=== 首帧趋势 A/B（A=${reportA.label} / B=${reportB.label}） ===`);
  console.log('（只给趋势，不判红：本节的任何数字都不会改变退出码）');
  console.log('');
  for (const note of cmp.notes) console.log(`  !! ${note}`);
  if (cmp.rows.length === 0) {
    console.log('  两份报表没有可比样本。');
    return;
  }
  console.log(`  ${'轴/样本'.padEnd(34)} ${'A 中位数'.padStart(10)} ${'B 中位数'.padStart(10)} ${'B/A'.padStart(7)} ${'变化'.padStart(9)}  结论`);
  for (const row of cmp.rows) {
    const name = `${row.axis}/${row.sample}`.padEnd(34);
    console.log(
      `  ${name} ${String(row.medianA).padStart(10)} ${String(row.medianB).padStart(10)} `
      + `${row.ratio.toFixed(3).padStart(7)} ${`${row.deltaPct >= 0 ? '+' : ''}${row.deltaPct.toFixed(2)}%`.padStart(9)}  ${row.verdict}`,
    );
  }
  console.log('');
  console.log('判定规则：小量级样本（两侧中位数 < 5 ms）且 |B−A| < 0.5 ms 记"持平（亚毫秒级）"；');
  console.log('          否则 |B/A−1| ≤ max(2×MAD/中位数, 2%) 记"持平（噪声带内）"，其余按方向记"变慢/变快"。');
}

// ---------------------------------------------------------------------------
// 报表输出
// ---------------------------------------------------------------------------

/**
 * 打印人读报表。
 * @param {object} report 报表
 */
function printReport(report) {
  console.log('=== 首帧时间报表（趋势用，非门禁） ===');
  console.log(`标签     : ${report.label}`);
  console.log(`时间     : ${report.generated}`);
  console.log(`机器指纹 : ${report.fingerprint.cpuModel} ×${report.fingerprint.cpuCount}, `
    + `${report.fingerprint.totalMemGiB} GiB, ${report.fingerprint.platform}/${report.fingerprint.arch}, node ${report.fingerprint.node}`);
  console.log(`固定条件 : ${report.fixed.scene} @ ${report.fixed.width}x${report.fixed.height}, `
    + `t=${report.fixed.frame}s, 采样 ${report.fixed.runs} 轮（wasm ${report.fixed.wasmRounds} 轮）`);
  console.log('');
  for (const axis of report.axes) {
    console.log(`· ${axis.label}`);
    console.log(`  对象 : ${axis.subject}`);
    if (axis.status !== 'ok') {
      console.log(`  状态 : ${axis.status === 'skip' ? 'SKIP（未测到）' : 'ERROR（测量失败）'} —— ${axis.note}`);
      console.log('');
      continue;
    }
    if (axis.environment && Object.keys(axis.environment).length > 0) {
      console.log(`  环境 : ${Object.entries(axis.environment).map(([k, v]) => `${k}=${v}`).join(', ')}`);
    }
    for (const [key, stats] of Object.entries(axis.samples)) {
      console.log(`  ${key.padEnd(16)} 中位数 ${String(stats.median).padStart(9)} ${axis.unit}`
        + `  p90 ${String(stats.p90).padStart(9)}  min ${String(stats.min).padStart(9)}`
        + `  max ${String(stats.max).padStart(9)}  MAD ${String(stats.mad).padStart(7)}  n=${stats.n}`);
    }
    if (axis.note) console.log(`  备注 : ${axis.note}`);
    console.log('');
  }
  console.log(`结论：本报表**不作 Go/No-Go**（首帧时间环境相关，同机 A/B 才有意义）。`);
  console.log(`      需要对比时：node tools/first_frame_report.mjs --compare <A.json> <B.json>`);
}

/**
 * 组装报表对象。
 * @param {object} opts 选项
 * @param {object[]} axes 轴记录
 * @returns {object} 报表
 */
function buildReport(opts, axes) {
  return {
    schemaVersion: 1,
    generated: new Date().toISOString(),
    label: opts.label,
    nonGating: true,
    note: '首帧时间只作趋势报表，不作 Go/No-Go：环境相关（CPU/负载/容器配额），同机 A/B 才有意义',
    fingerprint: fingerprint(),
    fixed: { ...FIXED, runs: opts.runs },
    axes,
  };
}

// ---------------------------------------------------------------------------
// 自证
// ---------------------------------------------------------------------------

/**
 * 自证。除统计/解析正确性外，核心是钉住"非门禁"性质：
 *   合成一份 10 倍变慢的 B 报表 → 结论必须是"变慢"，而**退出码必须是 0**。
 * 若日后有人把报表接成门禁（或让慢直接变红），这条断言立刻失败。
 * @returns {number} 退出码
 */
function selfTest() {
  const results = [];
  const assert = (name, cond, detail) => {
    results.push({ name, ok: Boolean(cond) });
    console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined ? `  —— ${detail}` : ''}`);
  };

  // 1) 统计函数
  assert('median 奇数样本', median([3, 1, 2]) === 2, median([3, 1, 2]));
  assert('median 偶数样本取中间均值', median([1, 2, 3, 4]) === 2.5, median([1, 2, 3, 4]));
  assert('percentile p90 线性插值', Math.abs(percentile([1, 2, 3, 4, 5], 0.9) - 4.6) < 1e-9, percentile([1, 2, 3, 4, 5], 0.9));
  assert('mad 抗离群', mad([10, 10, 10, 10, 100]) === 0, mad([10, 10, 10, 10, 100]));

  // 2) 引擎输出解析（格式漂移的钉子）
  assert('解析 ms', Math.abs(parseElapsedMs('  总耗时    : 4.443ms') - 4.443) < 1e-9, parseElapsedMs('  总耗时    : 4.443ms'));
  assert('解析 s（不被 ms 规则吃掉）', Math.abs(parseElapsedMs('总耗时 : 1.25s') - 1250) < 1e-9, parseElapsedMs('总耗时 : 1.25s'));
  assert('解析 µs', Math.abs(parseElapsedMs('总耗时 : 500µs') - 0.5) < 1e-9, parseElapsedMs('总耗时 : 500µs'));
  assert('无该行返回 null（不编造数字）', parseElapsedMs('nothing here') === null);

  // 3) A/B 对比 + 非门禁钉子
  const mkReport = (label, wallMedian, wasmMedian, fp) => ({
    schemaVersion: 1,
    label,
    fingerprint: fp,
    fixed: { ...FIXED, runs: 15 },
    axes: [
      {
        key: 'native_first_frame',
        label: 'native 首帧',
        status: 'ok',
        unit: 'ms',
        samples: { wall_ms: { n: 5, median: wallMedian, p90: wallMedian, min: wallMedian, max: wallMedian, mad: 0.1, raw: [] } },
      },
      {
        key: 'wasm_cold_start',
        label: 'wasm 冷启动代理',
        status: 'ok',
        unit: 'ms',
        samples: { total_ms: { n: 5, median: wasmMedian, p90: wasmMedian, min: wasmMedian, max: wasmMedian, mad: 0.1, raw: [] } },
      },
    ],
  });
  const fpSame = { platform: 'linux', arch: 'x64', cpuModel: 'fake', cpuCount: 4, totalMemGiB: 8, node: 'v22' };
  const baseline = mkReport('A', 20, 10, fpSame);
  const regression = mkReport('B', 200, 100, fpSame); // 刻意 10 倍变慢
  const cmp = compareReports(baseline, regression);
  const slowRows = cmp.rows.filter((r) => r.verdict === '变慢');
  assert('10 倍变慢被标成"变慢"', slowRows.length === cmp.rows.length && cmp.rows.length > 0, `${slowRows.length}/${cmp.rows.length}`);
  assert('>>> 变慢**不**判红（退出码仍为 0）', cmp.exitCode === EXIT_OK, `exit=${cmp.exitCode}`);
  assert('>>> 任意慢的报表退出码仍为 0', reportExitCode(baseline) === EXIT_OK && reportExitCode(regression) === EXIT_OK,
    `${reportExitCode(baseline)}/${reportExitCode(regression)}`);

  // 4) 噪声带内记"持平"
  const jitter = mkReport('B2', 20.2, 10.05, fpSame);
  const cmpJitter = compareReports(baseline, jitter);
  assert('2% 内抖动记"持平（噪声带内）"', cmpJitter.rows.every((r) => r.verdict.startsWith('持平')), cmpJitter.rows.map((r) => r.verdict).join(','));

  // 5) 换机器必须被识别
  const fpOther = { ...fpSame, cpuModel: 'another cpu' };
  const cmpOther = compareReports(baseline, mkReport('B3', 20, 10, fpOther));
  assert('换机器被判为不可直接比较', cmpOther.sameMachine === false && cmpOther.notes.length > 0, cmpOther.notes[0]);
  assert('换机器对比也不判红', cmpOther.exitCode === EXIT_OK, `exit=${cmpOther.exitCode}`);

  // 6) 全轴未测到 ⇒ 3（"没测到"要与"测到了很快"区分开）
  const empty = { axes: [{ key: 'x', status: 'skip', samples: {} }] };
  assert('全轴 SKIP ⇒ 退出码 3', reportExitCode(empty) === EXIT_NO_SAMPLE, reportExitCode(empty));

  const failed = results.filter((r) => !r.ok);
  console.log(`\n==> first_frame_report --self-test: ${failed.length === 0 ? 'PASS' : 'FAIL'}（${results.length - failed.length}/${results.length}）`);
  return failed.length === 0 ? EXIT_OK : EXIT_SELF_TEST_FAILED;
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
    console.error(`first_frame_report: ${err.message}`);
    console.error(USAGE);
    return EXIT_USAGE;
  }
  if (opts.help) {
    console.log(USAGE);
    return EXIT_OK;
  }
  if (opts.selfTest) return selfTest();

  if (opts.compare) {
    let reportA;
    let reportB;
    try {
      reportA = JSON.parse(fs.readFileSync(opts.compare[0], 'utf8'));
      reportB = JSON.parse(fs.readFileSync(opts.compare[1], 'utf8'));
    } catch (err) {
      console.error(`first_frame_report: 读报表失败：${err.message}`);
      return EXIT_USAGE;
    }
    const cmp = compareReports(reportA, reportB);
    printComparison(cmp, reportA, reportB);
    console.log('\n==> 对比完成（报表性质：恒不判红）');
    return cmp.exitCode;
  }

  const axes = [measureNativeAxis(opts), measureWasmAxis(opts)];
  const report = buildReport(opts, axes);
  printReport(report);

  if (opts.json) {
    fs.mkdirSync(path.dirname(opts.json), { recursive: true });
    fs.writeFileSync(opts.json, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
    console.log(`\n报表已写：${opts.json}`);
  }

  const exitCode = reportExitCode(report);
  if (exitCode === EXIT_NO_SAMPLE) {
    console.error('\n==> first_frame_report: SKIP（一个轴都没测到——先构建 release 二进制与 wasm 产物）');
  } else {
    console.log('\n==> first_frame_report: 已采样（趋势报表，不判红）');
  }
  return exitCode;
}

process.exit(main(process.argv.slice(2)));
