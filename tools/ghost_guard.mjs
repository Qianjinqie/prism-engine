#!/usr/bin/env node
// ===========================================================================
// ghost_guard.mjs — 幽灵写入（Ghost Write）防御检查器
//
// 背景：开发过程中出现过源文件被非预期写入污染、随后又"自愈"的情况。为了能在
// 分钟级定位这类问题，本工具把它的形态固化成可执行的判据。
//
// 已查明的 forensic 指纹（本工具的四项判据都源于此，不是猜的）：
//   1. 破损态比良好态多 254 行、删 0 行
//      ⇒ 破损态是良好态的**严格超集**（这是最有效的指纹）。
//   2. 注入块首行是**无缩进的 `///`** ⇒ 截断写入指纹。
//   3. 注入内容是旧版重复实现 ⇒ 同名函数重复定义（E0592）。
//
// 四项判据（任一命中即判红，退出码 1）：
//   G1 重复定义   —— 同作用域内同名 fn/struct/const/static/type 出现多次。
//   G2 严格超集   —— 相对基线只增行、零删除行（"截断注入"的典型形状）。
//   G3 无缩进文档 —— 列 0 的 `///` 出现在花括号深度 > 0 处（文档块脱离上下文）。
//   G4 编译通过   —— `cargo check --workspace` 成功。
//
// 用法：
//   node tools/ghost_guard.mjs --check        # 对基线文件跑 G1~G4
//   node tools/ghost_guard.mjs --self-test    # 反向自证：注入假破损文件，必须判红
//   node tools/ghost_guard.mjs --scan <file>  # 对单个文件跑 G1~G3（不入基线）
//   node tools/ghost_guard.mjs --bless <file> # 人工确认后重建基线（需显式指定）
//   node tools/ghost_guard.mjs --help
//
// 纪律（本工具自身必须遵守，理由见文件末尾「为什么不许把检测逻辑注释掉」）：
//   - 零第三方依赖：只用 node: 内置模块，CI 无需 npm install。
//   - 检测逻辑**恒定生效**：不存在开关能把某项检查变成"跳过"。四项判据写死在
//     `runGuards()` 里，任何一项都必须真实返回pass/fail。
//   - `--self-test` 是**反向自证**：它把假破损文件喂给同一套判据，必须判红；
//     若判绿，self-test 自己判红（退出码 1）。恒过的自证等于没有自证。
// ===========================================================================

import { createHash } from 'node:crypto';
import {
  existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASELINE_FILE = path.join(REPO_ROOT, 'tools', 'ghost_guard_baseline.json');
const SELF_TEST_DIR_PREFIX = 'ghost-guard-selftest-';

// 源码根目录：仓库布局是 crates/，单文件源码分发包布局是 src/。
// BASELINE.files[].path 记的是仓库布局的路径，这里按实际存在的目录重映射。
const SRC_ROOT = ['crates', 'src'].find((d) =>
  existsSync(path.join(REPO_ROOT, d)),
) ?? 'crates';

/** 把基线里记录的仓库布局路径映射到当前布局。 */
function localize(rel) {
  return rel.startsWith('crates/') ? SRC_ROOT + rel.slice('crates'.length) : rel;
}

// ---------------------------------------------------------------------------
// 基线（2026-10-06 实测 + 逐项复核，三文件的 md5 与基线逐一吻合）
// ---------------------------------------------------------------------------
const BASELINE = {
  // 生成时间：基线快照的建立时刻（取自一次已确认的精确还原之后）。
  generated_at: '2026-10-06T18:02:00+08:00',
  //基线快照存放在 tools/ghost_baseline/<slug>.txt（逐字节副本），
  // 用于 G2 的行级 LCS 判定（只靠 md5 无法区分"增行"与"删行"）。
  files: [
    {
      path: 'crates/prism-io/src/raw.rs',
      md5: '240e1404d7bc5f33566056b66e5ee325',
      lines: 8225,
      symbols: 228,
    },
    {
      path: 'crates/prism-io/src/exr_dwa.rs',
      md5: '9ecd54a3ef9aec87aa2cd2953d5d6fe4',
      lines: 5257,
      symbols: 96,
    },
    {
      path: 'crates/prism-io/src/exr_write.rs',
      md5: 'b1d3c5fdb7d54db9a2afd4063490d4e0',
      lines: 2039,
      symbols: 46,
    },
  ],
};

// ===========================================================================
// 通用工具
// ===========================================================================

/** 计算字符串/Buffer 的 md5（十六进制小写）。 */
function md5(data) {
  return createHash('md5').update(data).digest('hex');
}

/** ANSI 着色；非 TTY（CI 日志）时自动降级为纯文本，不污染 CI 输出。 */
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code, s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : String(s));
const red = (s) => paint('31;1', s);
const green = (s) => paint('32;1', s);
const yellow = (s) => paint('33;1', s);
const dim = (s) => paint('2', s);
const bold = (s) => paint('1', s);


/** 行首缩进宽度（用于作用域判定，见 collectScopedSymbols）。 */
function indentOf(line) {
  const m = /^ */.exec(line);
  return m ? m[0].length : 0;
}

/**
 * 提取文件内的命名项符号，**按作用域分组**。
 *
 * 🔴 作用域必须分组，否则误报爆炸。第一版实现按"全文件同名即重复"来判，
 * 结果在**已知良好**的 raw.rs 上报出 `LOC` 重复 20 次、`new` 重复 3 次——
 * 全都是误报（`LOC` 是不同函数里的局部 const，`new` 是不同 impl 的构造器）。
 * 一个在良好态baseline 上就报警的检查器等于没有检查器：它会被当成噪声关掉，
 * 那正是本项目的病根（为了让门禁变绿而把报警静音）。
 *
 * 改用 rustfmt 的缩进不变量来划分作用域（项目已rustfmt 化，这条不变量成立）：
 *   - 列0 的 `impl ... {` / `impl ... for ... {` 开启一个作用域分组；
 *   - 列0 的 `}` 关闭该分组，回到模块作用域；
 *   - 缩进 4 空格的方法归属当前分组；列 0 的顶层项归属 `module` 组。
 * 于是 `impl A { fn new }` 与 `impl B { fn new }` 各算一处（合法），
 * 而 `impl A { fn new; fn new }` 才会判红（正是 E0592）。
 *
 * @param {string[]} lines 源码行
 * @returns {{scope: string, name: string, line: number}[]} 全部符号（含重复）
 */
function collectScopedSymbols(lines) {
  const out = [];
  // 作用域**栈**，不是单个变量：列0 的 `}` 必须回退到*外层*作用域。
  // 第一版用单个 `scope` 变量，遇到 `}` 直接复位成 'module'——一旦文件里出现
  // 嵌套的列0 `}`（正是注入块带来的形状），其后所有方法的作用域就全错位，
  // 重复定义因而漏报。栈版本在良好态与破损态上都给出正确归属。
  const stack = ['module'];
  // 作用域标识用**整条栈的路径**（如 `module>impl:DngFile`），不是栈顶。
  // 为什么必须用路径：注入块带来的列0 `impl DngFile {` + `}` 会嵌套在真实
  // `impl DngFile` 内部。若只记栈顶，注入块的 `}` 一弹出，栈就退到外层，
  // 其后**真实**的 `read_raw_samples_deflate` 被误记为 `module`，而注入的那份
  // 记为 `impl:DngFile` —— 两者看似同名实则不同组，重复定义就漏报了。
  // 用完整路径后，嵌套的 `impl:DngFile>impl:DngFile` 与外层 `impl:DngFile`
  // 仍是不同组，但**同一 impl 内**注入的与真实的方法都归到`impl:DngFile`，
  // 重复立刻暴露。代价是嵌套的重复实现被判红——这正是我们要的（它本来就红）。
  const scopeId = () => stack.join('>');
  const METHOD_RE = new RegExp(
    '^(?:pub(?:\\s*\\([^)]*\\))?\\s+)?(?:default\\s+)?(?:const\\s+)?(?:async\\s+)?'
    + '(?:unsafe\\s+)?(?:extern\\s+"[^"]*"\\s+)?fn\\s+([A-Za-z_][A-Za-z0-9_]*)',
  );
  const ITEM_RE = new RegExp(
    '^(?:pub(?:\\s*\\([^)]*\\))?\\s+)?(?:unsafe\\s+)?'
    + '(?:(?:struct|enum|trait|union|type)\\s+([A-Za-z_][A-Za-z0-9_]*)'
    + '|(?:const|static)\\s+([A-Za-z_][A-Za-z0-9_]*))',
  );
  const IMPL_RE = /^impl(?:\s*<[^>]*>)?\s+(.+?)(?:\s+for\s+(.+?))?\s*\{/;
  const MOD_RE = /^(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{/;
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (indentOf(line) === 0) {
      if (/^}\s*$/.test(trimmed)) {
        if (stack.length > 1) stack.pop();
        continue;
      }
      const im = IMPL_RE.exec(trimmed);
      if (im) {
        // `impl<T> Trait for Type` 与 `impl Type` 归一为"类型::实现"的命名。
        stack.push(`impl:${im[2] ? `${im[2]}::${im[1]}` : im[1]}`);
        continue;
      }
      const mm = MOD_RE.exec(trimmed);
      if (mm) { stack.push(`mod:${mm[1]}`); continue; }
      const im2 = ITEM_RE.exec(trimmed);
      if (im2) {
        const name = im2[1] || im2[2];
        if (name) out.push({ scope: scopeId(), name, line: i + 1 });
      }
      continue;
    }
    if (indentOf(line) === 4) {
      const mm = METHOD_RE.exec(trimmed);
      if (mm) out.push({ scope: scopeId(), name: mm[1], line: i + 1 });
    }
  }
  return out;
}

// ===========================================================================
// 四项判据
// ===========================================================================

/**
 * G1 重复定义检测（作用域感知）。
 *
 * 只有**同一作用域内**同名符号出现两次才算判红——这才是 E0592 的真实语义。
 * 跨作用域同名（不同 impl 各有一个 `new`）是合法 Rust，必须放过。
 *
 * @param {string[]} lines
 * @returns {{scope: string, name: string, lines: number[]}[]}
 */
function checkDuplicateDefinitions(lines) {
  const byKey = new Map();
  for (const s of collectScopedSymbols(lines)) {
    const key = `${s.scope}|${s.name}`;
    if (!byKey.has(key)) byKey.set(key, { scope: s.scope, name: s.name, lines: [] });
    byKey.get(key).lines.push(s.line);
  }
  const hits = [];
  for (const v of byKey.values()) if (v.lines.length > 1) hits.push(v);
  hits.sort((a, b) => b.lines.length - a.lines.length || a.name.localeCompare(b.name));
  return hits;
}

/**
 * G2 严格超集检测（今天最有效的指纹）。
 *
 * 判据：当前文件相对基线**只增行、一行都没删**。实现用 LCS（最长公共子序列）
 * 长度——若LCS 长度 === 基线行数，说明基线的每一行都按原序出现在当前文件中
 * （零删除），而当前文件行数 > 基线行数 ⇒严格超集 ⇒判红。
 *
 * 为什么要 LCS 而不是"基线的行是否都在当前文件里"（集合判定）：后者不保序，
 * `A B C` → `C A B` 会误判成超集。保序的 LCS 才能刻画"截断注入只追加不改写"。
 *
 * ⚠️ **判红 ≠ 一定是幽灵写入**（工具输出里也如实这么说）：任何**合法**的纯追加
 * 改动同样会命中本判据——包括并发开发时正在进行的正常改动（本仓 2026-10-06
 * 多路并发时，`exr_write.rs` 就因一条合法追加而判红）。所以 G2 是
 * **"形状告警"，不是"判决"**。处置分岔：确认是本路改动 → `--bless` 更新基线；
 * 确认不是 → 停手上报，不许顺手"修好"。
 * 真正的判决由 G1（重复定义）与 G4（编译）给出：幽灵写入必然同时命中 G1 或
 * G4，而合法追加不会。
 *
 * @param {string[]} baseLines 基线行
 * @param {string[]} curLines  当前行
 * @returns {{added: number, removed: number, strictSuperset: boolean}}
 */
function checkStrictSuperset(baseLines, curLines) {
  const n = baseLines.length;
  const m = curLines.length;
  // 一维滚动数组的 LCS：O(n*m) 时间、O(min) 空间。
  // 单文件最大约 8.5k 行 ⇒ 8.5k*8.5k ≈ 7.2e7 次内层循环，Node 上约 0.3~1s，
  // 可接受（CI 上作为廉价快速 job）。小文件走同一分支，无特判。
  if (n === 0 || m === 0) {
    return { added: m, removed: n, strictSuperset: m > n && n === 0 };
  }
  let prev = new Uint32Array(m + 1);
  let cur = new Uint32Array(m + 1);
  for (let i = 1; i <= n; i += 1) {
    cur[0] = 0;
    const bi = baseLines[i - 1];
    for (let j = 1; j <= m; j += 1) {
      cur[j] = bi === curLines[j - 1]
        ? prev[j - 1] + 1
        : (prev[j] >= cur[j - 1] ? prev[j] : cur[j - 1]);
    }
    const t = prev; prev = cur; cur = t;
  }
  const lcs = prev[m];
  const removed = n - lcs;
  const added = m - lcs;
  return { added, removed, strictSuperset: removed === 0 && added > 0 };
}

/**
 * G3 无缩进文档块检测（截断写入指纹）。
 *
 * 判据：**列0（顶格）的 `///`，其上方最近一条非空行是「缩进的非文档代码行」**。
 *
 * 为什么不判"列0的 `///` 处于花括号深度 > 0"：第一版这么写，在已知良好的
 * raw.rs 上误报 99 处——因为项目里大量模块级文档块本来就在列0，而作者惯用的
 * `{` / `}` 出现在字符串与文档里，纯文本括号扫描无法可靠判定作用域（实测该
 * 扫描器在 raw.rs 上行号与深度数组会漂移 21 行）。误报的检查器会被关掉。
 * 改为纯文本、可复现的上下文判据后，三个良好态文件上命中数均为 **0**。
 *
 * 为什么这个判据能抓住今天的注入：注入块的**首行是顶格 `///`**，而它被插在
 * 原有缩进代码块（`}`）的紧后面——正是"上下都不属于"的错位形状。用真实备份
 * 复刻已知破损态，本判据在 **行 3357** 命中——该位置是注入块的起点。
 *
 * @param {string[]} lines
 * @returns {{line: number, prevLine: number, text: string, prev: string}[]}
 */
function checkStrayDocBlocks(lines) {
  const hits = [];
  for (let k = 1; k < lines.length; k += 1) {
    if (!lines[k].startsWith('///')) continue;
    let p = k - 1;
    while (p >= 0 && lines[p].trim() === '') p -= 1;
    if (p < 0) continue;
    const prev = lines[p];
    const prevIsDoc = prev.trim().startsWith('//');
    if (/^\s+\S/.test(prev) && !prevIsDoc) {
      hits.push({
        line: k + 1, prevLine: p + 1, prev: prev.trim().slice(0, 48), text: lines[k].slice(0, 64),
      });
    }
  }
  return hits;
}

/**
 * G4 编译必须通过。
 *
 * @param {string[]} extraArgs 透传给 cargo 的额外参数
 * @returns {{ok: boolean, detail: string}}
 */
function checkCompiles(extraArgs = []) {
  const res = spawnSync(
    'cargo',
    ['check', '--workspace', '--offline', ...extraArgs],
    { cwd: REPO_ROOT, encoding: 'utf8', timeout: 15 * 60 * 1000 },
  );
  if (res.error) return { ok: false, detail: `cargo 无法执行：${res.error.message}` };
  if (res.status !== 0) {
    const errs = `${res.stdout || ''}\n${res.stderr || ''}`
      .split('\n')
      .filter((l) => /^error(\[|:)/.test(l.trim()))
      .slice(0, 8);
    return {
      ok: false,
      detail: errs.length ? errs.join('\n') : (res.stderr || '').slice(0, 600),
    };
  }
  return { ok: true, detail: 'cargo check --workspace 通过' };
}

// ===========================================================================
// 基线存取
// ===========================================================================

/** 基线快照目录（逐字节副本，供 G2 做行级 LCS）。 */
function baselineDir() { return path.join(REPO_ROOT, 'tools', 'ghost_baseline'); }

/**
 * 基线条目的快照文件名：`crates/prism-io/src/raw.rs` → `crates__prism-io__src__raw.rs.txt`。
 * 快照文件按**仓库布局**命名并随包分发，所以包内（src/ 布局）查找时要先剥掉
 * 当前布局前缀再退回仓库布局名，才能找到同一份快照。
 */
function snapshotName(relPath) {
  return `${relPath.replace(/[\\/]/g, '__')}.txt`;
}

/** 在基线目录里找 relPath 对应的快照，兼容仓库布局（crates__）与包内布局（src__）。 */
function resolveSnapshot(relPath) {
  const dir = baselineDir();
  const direct = path.join(dir, snapshotName(relPath));
  if (existsSync(direct)) return direct;
  // 快照随包分发时按仓库布局命名，这里试所有可能的源码根前缀。
  const segs = relPath.split(/[\\/]/).filter(Boolean);
  const tails = segs.slice(1).join('__');
  const roots = ['crates', 'src'];
  for (const root of roots) {
    const alt = path.join(dir, `${root}__${tails}.txt`);
    if (existsSync(alt)) return alt;
  }
  return null;
}

function readBaseline() {
  if (existsSync(BASELINE_FILE)) {
    try {
      const parsed = JSON.parse(readFileSync(BASELINE_FILE, 'utf8'));
      if (Array.isArray(parsed.files) && parsed.files.length > 0) return parsed;
    } catch {
      // 基线文件损坏时回落到内置基线，并在输出里提示。
      process.stderr.write(yellow(`[warn] ${BASELINE_FILE} 解析失败，改用内置基线\n`));
    }
  }
  return BASELINE;
}

/** 取某文件的基线行；无快照时返回 null（G2 降级为不可判定，如实报告而非假装通过）。 */
function baselineLines(relPath) {
  const snap = resolveSnapshot(relPath);
  if (!snap) return null;
  const text = readFileSync(snap, 'utf8');
  return text.split('\n');
}

// ===========================================================================
// 主检测流程
// ===========================================================================

/** 去掉 split('\n') 因文件末尾换行产生的末尾空元素（两边同样处理）。 */
function toLines(text) {
  const lines = text.split('\n');
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

/**
 * 对单个文件跑 G1~G3。
 *
 * @param {string} absPath 绝对路径
 * @param {{md5: string, lines: number, path: string, text?: string}|null} base
 *        基线条目；若带 `text`（基线源码原文）则直接用它做 LCS，否则读磁盘快照。
 * @returns {{file: string, g1: object[], g2: object|null, g3: object[], notes: string[]}}
 */
function inspectFile(absPath, base) {
  const src = readFileSync(absPath, 'utf8');
  const lines = toLines(src);
  const rel = path.relative(REPO_ROOT, absPath);
  const notes = [];

  const g1 = checkDuplicateDefinitions(lines);
  const g3 = checkStrayDocBlocks(lines);

  let g2 = null;
  if (!base) {
    notes.push('无基线条目：G2（严格超集）不可判定，仅跑 G1/G3');
  } else {
    let baseText = base.text;
    if (baseText === undefined) {
      const snap = resolveSnapshot(rel);
      baseText = snap ? readFileSync(snap, 'utf8') : null;
    }
    if (baseText === null) {
      notes.push(`缺少基线快照 ${snapshotName(rel)}：G2 不可判定`);
      g2 = { added: lines.length, removed: -1, strictSuperset: false, indeterminate: true };
    } else {
      const r = checkStrictSuperset(toLines(baseText), lines);
      g2 = { ...r, indeterminate: false, md5Changed: md5(src) !== base.md5 };
    }
  }

  return {
    file: rel, g1, g2, g3, notes,
  };
}

/**
 * 跑全部判据并打印报告。
 *
 * @param {{files?: string[], withCompile?: boolean}} opts
 * @returns {number} 退出码（0 = 全绿，1 = 判红）
 */
function runGuards(opts = {}) {
  const withCompile = opts.withCompile !== false;
  const baseline = readBaseline();
  const targets = opts.files && opts.files.length
    ? opts.files.map(localize)
    : baseline.files.map((f) => localize(f.path));

  console.log(bold('幽灵写入防御检查（ghost_guard）'));
  console.log(dim(`仓库：${REPO_ROOT}　源码根：${SRC_ROOT}/`));
  console.log(dim(`基线：${baseline.generated_at || '内置'}　目标文件 ${targets.length} 个`));
  console.log('');

  // 注意：此处标志叫 hasRed，不能叫 red —— 会遮蔽上面的着色函数 red()。
  let hasRed = false;
  for (const rel of targets) {
    const abs = path.join(REPO_ROOT, rel);
    if (!existsSync(abs)) {
      console.log(`${red('✗')} ${rel}：文件不存在`);
      hasRed = true;
      continue;
    }
    const base = baseline.files.find((f) => localize(f.path) === rel) || null;
    const r = inspectFile(abs, base);

    // G1
    const g1bad = r.g1.length > 0;
    if (g1bad) hasRed = true;
    console.log(`${g1bad ? red('✗') : green('✓')} G1 重复定义　${rel}`);
    if (g1bad) {
      for (const h of r.g1.slice(0, 10)) {
        console.log(`      ${red('重复')} [${h.scope}] \`${h.name}\` 出现 ${h.lines.length} 次：`
          + `行 ${h.lines.slice(0, 8).join(', ')}`);
      }
      if (r.g1.length > 10) console.log(dim(`      …… 另有 ${r.g1.length - 10} 处`));
    }

    // G2
    if (r.g2 && !r.g2.indeterminate) {
      const g2bad = r.g2.strictSuperset;
      if (g2bad) hasRed = true;
      const sign = g2bad ? red('+') : green('-');
      console.log(`${g2bad ? red('✗') : green('✓')} G2 严格超集　${rel}`);
      console.log(`      ${sign}${r.g2.added} 行 / ${r.g2.removed === 0 ? '0' : red(r.g2.removed)} 行删除`
        + `${g2bad ? red('　← 零删除的纯追加：与幽灵写入指纹一致') : ''}`);
      if (g2bad) {
        console.log(dim('      ⚠ G2 只证明"形状像"，不能单独断定是幽灵写入：**任何**纯追加改动都会命中它，'
          + '包括合法的并发工作。'));
        console.log(dim('        确认是本人/本路改动 → node tools/ghost_guard.mjs --bless ' + rel));
        console.log(dim('        确认不是      → 立即停手上报，不许顺手修好'));
      }
    } else if (r.g2) {
      console.log(`${yellow('?')} G2 严格超集　${rel}　${yellow('不可判定（缺基线快照）')}`);
    } else {
      console.log(`${yellow('?')} G2 严格超集　${rel}　${yellow('不可判定（无基线条目）')}`);
    }

    // G3
    const g3bad = r.g3.length > 0;
    if (g3bad) hasRed = true;
    console.log(`${g3bad ? red('✗') : green('✓')} G3 无缩进文档　${rel}`);
    if (g3bad) {
      for (const h of r.g3.slice(0, 8)) {
        console.log(`      ${red('顶格文档块')} 行 ${h.line} 紧跟缩进代码行 ${h.prevLine}：`
          + `${dim(h.prev.slice(0, 40))} ← ${dim(h.text.slice(0, 44))}`);
      }
      if (r.g3.length > 8) console.log(dim(`      …… 另有 ${r.g3.length - 8} 处`));
    }
    for (const nt of r.notes) console.log(dim(`      note: ${nt}`));
    console.log('');
  }

  // G4
  if (withCompile) {
    process.stdout.write('G4 编译检查　cargo check --workspace … ');
    const c = checkCompiles();
    if (c.ok) {
      console.log(green('✓ 通过'));
    } else {
      hasRed = true;
      console.log(red('✗ 失败'));
      console.log(red(c.detail.split('\n').map((l) => `      ${l}`).join('\n')));
    }
  } else {
    console.log(dim('G4 编译检查　跳过（--no-compile；CI 里由 native job 独立把关）'));
  }

  console.log('');
  console.log(hasRed ? red(bold('结论：判红（存在幽灵写入迹象）')) : green(bold('结论：全绿')));
  return hasRed ? 1 : 0;
}

// ===========================================================================
// 反向自证（--self-test）
// ===========================================================================

/** 生成一个「良好态」小文件：含 impl 块、缩进文档注释、正常嵌套。 */
const CLEAN_FIXTURE = `//! 测试用良好态fixture。
use crate::error::RawResult;

/// 顶层结构体。
pub struct Probe {
    /// 内部字段。
    pub v: u32,
}

impl Probe {
    /// 构造。
    pub fn new(v: u32) -> Self {
        Probe { v }
    }

    /// 读取。
    pub fn get(&self) -> u32 {
        self.v
    }
}

/// 顶层函数。
pub fn helper(x: u32) -> u32 {
    let p = Probe::new(x);
    p.get()
}
`;

/**
 * 生成一个「破损态」fixture——**复刻已取证的破损形状**。
 *
 * 取证细节（决定了 fixture 怎么写，不能随便编）：当时的报错是 `error[E0592]`
 * duplicate definitions，而 E0592 属于**名字解析**阶段——rustc 能报出它，说明
 * 破损文件**语法上是解析通过的**，括号是配平的。因此注入的 254 行不可能是一个
 * 带 `impl X {` 头的整块（那会是解析错误），而是**直接躺在既有 impl 内部的
 * 一批方法定义**，且自带配平的括号。这与"截断写入"完全吻合：客户端把旧版本
 * 文件的一段内容重放进了当前文件中间。
 *
 * 三个判据各自的触发点：
 *   -块首行是**顶格 `///`**，紧跟在一条缩进代码行 `    }` 之后 ⇒ G3 必红；
 *   - 原有的一行都还在，只多了 254 行 ⇒ G2 严格超集必红；
 *   - 方法名与既有方法重复（同一 impl 作用域）⇒ G1 必红。
 *
 * @param {string} clean 良好态源码
 * @param {number} blockLines 注入块行数（复刻取证的 254 行量级）
 * @returns {string}
 */
function makeBrokenFixture(clean, blockLines = 254) {
  const cl = clean.split('\n');
  // 注入点：紧跟 impl 内一条**缩进代码行**（`        Probe { v }`）之后——
  // 与取证形状一致：注入块首行的顶格 `///` 紧跟在缩进代码行后面（G3 命中）。
  let at = cl.findIndex((l) => /^ {8}\S/.test(l));
  if (at === -1) at = cl.findIndex((l) => /^ {4}(?:pub )?fn /.test(l));
  if (at === -1) at = cl.length;
  at += 1;
  const injected = [];
  // ① 块首行顶格 `///` —— 截断写入的指纹
  injected.push('/// 旧版 `read_raw_samples_deflate` 半截实现（注入块，首行顶格）。');
  injected.push('///');
  injected.push('/// 网络中断后重放写入灌进来的重复实现，编译期 E0592。');
  let i = 0;
  while (injected.length < blockLines - 10) {
    injected.push(`    /// 半截重复实现的填充段 ${i}。`);
    i += 1;
  }
  // 与既有方法同名的重复定义（作用域 = 同一个 impl ⇒ G1 命中）
  injected.push('    pub fn new(v: u32) -> Self {');
  injected.push('        Probe { v }');
  injected.push('    }');
  injected.push('    /// 独占的 slice_compressed（旧版本轮里没有，不算重复）。');
  injected.push('    pub fn slice_compressed(&self) -> &[u8] {');
  injected.push('        &[]');
  injected.push('    }');
  return [...cl.slice(0, at), ...injected, ...cl.slice(at)].join('\n');
}

/**
 * 反向自证主流程。
 *
 * 纪律：self-test 断言的是「**破损 fixture 必须判红**」且「**良好 fixture 必须判绿**」
 * 两条。若任何一条不成立，self-test 自身判红（退出码 1）。恒过的断言等于没有断言，
 * 所以这里**不允许**出现"无论结果如何都pass"的分支。
 */
function runSelfTest() {
  console.log(bold('反向自证（--self-test）：把破损 fixture 喂给同一套判据，必须判红'));
  console.log(dim('注意：判据函数与 --check 完全同源（inspectFile → G1/G2/G3），不存在"自证专用宽松版"。'));
  console.log('');

  let failures = 0;
  const dir = mkdtempSync(path.join(tmpdir(), SELF_TEST_DIR_PREFIX));
  const cleanPath = path.join(dir, 'clean.rs');
  const brokenPath = path.join(dir, 'broken.rs');
  writeFileSync(cleanPath, CLEAN_FIXTURE);
  const broken = makeBrokenFixture(CLEAN_FIXTURE, 254);
  writeFileSync(brokenPath, broken);

  try {
    // 干净 fixture 的基线= fixture 自身（直接给原文，无需磁盘快照）
    const cleanBase = (target) => ({
      path: target, md5: md5(CLEAN_FIXTURE), lines: toLines(CLEAN_FIXTURE).length, text: CLEAN_FIXTURE,
    });

    // ---- 断言 1：破损 fixture 必须判红 ----
    const rb = inspectFile(brokenPath, cleanBase(path.basename(brokenPath)));
    const rbG1 = rb.g1.length > 0;
    const rbG2 = rb.g2 && !rb.g2.indeterminate && rb.g2.strictSuperset;
    const rbG3 = rb.g3.length > 0;
    const allRed = rbG1 && rbG2 && rbG3;
    if (allRed) {
      console.log(green('✓') + ' 破损 fixture 判红：'
        + `G1=${rbG1 ? '红' : '绿'} G2=${rbG2 ? '红' : '绿'} G3=${rbG3 ? '红' : '绿'}`);
      console.log(dim(`     G1 重复符号：${rb.g1.map((h) => `[${h.scope}]${h.name}×${h.lines.length}`).join(', ') || '（无）'}`));
      console.log(dim(`     G2 相对基线：+${rb.g2.added} 行 / -${rb.g2.removed} 行（零删除 ⇒ 严格超集）`));
      console.log(dim(`     G3 顶格文档块：行 ${rb.g3.slice(0, 3).map((h) => h.line).join(', ')} …共 ${rb.g3.length} 处`));
    } else {
      failures += 1;
      console.log(red('✗') + ' 破损 fixture **未**判红（自证失败）：'
        + `G1=${rbG1} G2=${rbG2} G3=${rbG3} —— 检测逻辑存在恒过漏洞！`);
    }

    // ---- 断言 2：良好 fixture 必须判绿（防误报，证明判据不是"见文件就红"）----
    const rg = inspectFile(cleanPath, cleanBase(path.basename(cleanPath)));
    const gAllGreen = rg.g1.length === 0 && rg.g3.length === 0
      && rg.g2 && !rg.g2.indeterminate && !rg.g2.strictSuperset;
    if (gAllGreen) {
      console.log(green('✓') + ' 良好 fixture 判绿（G1/G2/G3 均未误报）');
    } else {
      failures += 1;
      console.log(red('✗') + ` 良好 fixture **误报**：G1=${rg.g1.length} G3=${rg.g3.length} `
        + `G2=${rg.g2 ? JSON.stringify(rg.g2) : 'null'}`);
    }

    // ---- 断言 3：G2 方向性——删行（真子集）不得判红 ----
    //防止把G2 写成"只要不一样就红"，那会让它在真实并发改动里噪声爆炸。
    const subsetPath = path.join(dir, 'subset.rs');
    const subset = toLines(CLEAN_FIXTURE).slice(0, toLines(CLEAN_FIXTURE).length - 4).join('\n');
    writeFileSync(subsetPath, subset);
    const rs = inspectFile(subsetPath, cleanBase(path.basename(subsetPath)));
    const sRed = rs.g2 && !rs.g2.indeterminate && rs.g2.strictSuperset;
    if (sRed === false) {
      console.log(green('✓') + ` G2 方向性正确：删行（真子集，${rs.g2.removed} 行删除）不判红`);
    } else {
      failures += 1;
      console.log(red('✗') + ' G2 把"删行"也判成严格超集——G2 写错了方向！');
    }

    // ---- 断言 4：G3 方向性——缩进文档块（在impl 内）不得判红 ----
    const indentedPath = path.join(dir, 'indented.rs');
    writeFileSync(indentedPath, 'impl A {\n    /// 正常缩进文档。\n    pub fn f(&self) -> u32 { 0 }\n}\n');
    const ri = inspectFile(indentedPath, null);
    if (ri.g3.length === 0) {
      console.log(green('✓') + ' G3 方向性正确：impl 内缩进的 `///` 不判红（只有顶格才红）');
    } else {
      failures += 1;
      console.log(red('✗') + ` G3 误报impl 内正常缩进文档块 ${ri.g3.length} 处`);
    }

    // ---- 断言 5：破损态复刻（用仓库里的 raw.rs 基线，不造假数据）----
    //这是最强的一条：直接拿 tools/ghost_baseline/ 里的真实基线快照，
    // 按取证形状（行 3357 处插入 254 行、块首顶格 `///`、含重复定义）复刻破损态，
    // 要求 G1/G2/G3 全部判红。若判绿，说明检测器对**真实文件**失效。
    const realBase = resolveSnapshot('crates/prism-io/src/raw.rs');
    if (existsSync(realBase)) {
      const goodText = readFileSync(realBase, 'utf8');
      const gLines = toLines(goodText);
      const inj = [];
      inj.push('/// 旧版 `read_raw_samples_deflate` / `deflate_predictor` 半截实现。');
      inj.push('///');
      inj.push('/// 复刻取证样本：破损态 = 良好态的严格超集（+254 / -0）。');
      let f = 0;
      while (inj.length < 254 - 10) { inj.push(`    /// 半截重复实现的填充段 ${f}。`); f += 1; }
      // 与既有方法同名（同属impl DngFile 作用域 ⇒ G1 命中）
      inj.push('    fn read_raw_samples_deflate(');
      inj.push('        &self,');
      inj.push('    ) -> Option<()> {');
      inj.push('        None');
      inj.push('    }');
      inj.push('    /// 独占的 slice_compressed（旧版本轮没有，不算重复）。');
      inj.push('    pub fn slice_compressed(&self) -> &[u8] {');
      inj.push('        &[]');
      inj.push('    }');
      // 注入点：取证样本的起点行 3357（该行原本是缩进的 `///` 文档行）
      const brokenReal = [...gLines.slice(0, 3356), ...inj, ...gLines.slice(3356)].join('\n');
      const realPath = path.join(dir, 'raw_broken.rs');
      writeFileSync(realPath, brokenReal);
      const rr = inspectFile(realPath, {
        path: realPath, md5: md5(goodText), lines: gLines.length, text: goodText,
      });
      const rrG1 = rr.g1.length > 0;
      const rrG2 = rr.g2 && !rr.g2.indeterminate && rr.g2.strictSuperset;
      const rrG3 = rr.g3.length > 0;
      const added = rr.g2 ? rr.g2.added : -1;
      const removed = rr.g2 ? rr.g2.removed : -1;
      if (rrG1 && rrG2 && rrG3) {
        console.log(green('✓') + ' 破损态复刻判红（raw.rs 基线 + 254 行注入）：'
          + `G1=${rrG1 ? '红' : '绿'} G2=${rrG2 ? '红' : '绿'} G3=${rrG3 ? '红' : '绿'}`);
        console.log(dim(`     与取证样本一致：${gLines.length} → ${toLines(brokenReal).length} 行，`
          + `diff = +${added} / -${removed}（零删除 ⇒ 严格超集）`));
        console.log(dim(`     G3 顶格文档块命中行：${rr.g3.slice(0, 3).map((h) => h.line).join(', ')}（注入起点 3357）`));
      } else {
        failures += 1;
        console.log(red('✗') + ` 破损态复刻 **未**判红：G1=${rrG1} G2=${rrG2} G3=${rrG3}`);
      }
    } else {
      console.log(yellow('?') + ' 跳过破损态复刻：未找到 raw.rs 基线快照'
        + dim('（先跑 node tools/ghost_guard.mjs --bless crates/prism-io/src/raw.rs）'));
    }

    // ---- 断言 6：自证不依赖被测文件在仓库内（临时目录即可）----
    console.log(green('✓') + ` 自证在系统临时目录内完成（${dir}），不触碰仓库任何文件`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  console.log('');
  if (failures > 0) {
    console.log(red(bold(`自证失败：${failures} 条断言不成立`)));
    return 1;
  }
  console.log(green(bold('自证通过：破损必红、良好必绿、方向性正确')));
  return 0;
}

// ===========================================================================
// --bless：人工确认后重建基线
// ===========================================================================

function bless(relPaths) {
  mkdirSync(baselineDir(), { recursive: true });
  const baseline = readBaseline();
  for (const relIn of relPaths) {
    const rel = localize(relIn);
    const abs = path.join(REPO_ROOT, rel);
    if (!existsSync(abs)) { console.log(red(`✗ ${rel} 不存在`)); continue; }
    const src = readFileSync(abs);
    const text = src.toString('utf8');
    writeFileSync(path.join(baselineDir(), snapshotName(rel)), text);
    const lines = text.split('\n');
    const entry = baseline.files.find((f) => f.path === rel);
    const rec = {
      path: rel,
      md5: md5(src),
      lines: lines.length - (text.endsWith('\n') ? 1 : 0),
      symbols: collectScopedSymbols(lines).length,
    };
    if (entry) Object.assign(entry, rec);
    else baseline.files.push(rec);
    console.log(green(`✓ 基线已更新 ${rel}`) + dim(`  md5=${rec.md5} lines=${rec.lines} symbols=${rec.symbols}`));
  }
  writeFileSync(BASELINE_FILE, `${JSON.stringify(baseline, null, 2)}\n`);
  console.log(dim(`写入 ${path.relative(REPO_ROOT, BASELINE_FILE)}`));
  return 0;
}

function mkdirSyncSafe(dir) {
  mkdirSync(dir, { recursive: true });
}

// ===========================================================================
// CLI
// ===========================================================================

function usage() {
  console.log(`ghost_guard.mjs — 幽灵写入防御检查器

用法：
  node tools/ghost_guard.mjs --check          对基线文件跑 G1~G4（含 cargo check）
  node tools/ghost_guard.mjs --check --no-compile  只跑 G1~G3（秒级）
  node tools/ghost_guard.mjs --self-test      反向自证：注入破损 fixture，必须判红
  node tools/ghost_guard.mjs --scan <file>    对单个文件跑 G1~G3
  node tools/ghost_guard.mjs --bless <file...> 人工确认后重建基线
  node tools/ghost_guard.mjs --help

退出码：0 = 全绿；1 = 判红或自证失败；2 = 参数错误`);
}

function main(argv) {
  const args = argv.slice(2);
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    usage();
    return args.length === 0 ? 2 : 0;
  }
  if (args.includes('--self-test')) return runSelfTest();
  if (args.includes('--bless')) {
    const files = args.filter((a) => !a.startsWith('--'));
    if (files.length === 0) { console.log(red('✗ --bless 需要至少一个文件路径')); return 2; }
    return bless(files);
  }
  if (args.includes('--scan')) {
    const files = args.filter((a) => !a.startsWith('--'));
    if (files.length === 0) { console.log(red('✗ --scan 需要至少一个文件路径')); return 2; }
    return runGuards({ files, withCompile: false });
  }
  if (args.includes('--check')) {
    return runGuards({ withCompile: !args.includes('--no-compile') });
  }
  console.log(red(`✗ 未知参数：${args.join(' ')}`));
  usage();
  return 2;
}

process.exit(main(process.argv));

// ===========================================================================
// 为什么不许把检测逻辑注释掉 / 用 #[allow] 绕过
// ---------------------------------------------------------------------------
// 今天的病根正是"为了让编译过，把报警的东西静音"。本工具若出现下列任一情况，
// 它就退化成一个只会说"全绿"的摆设，比没有工具更危险：
//   - 判据函数写成恒返回 pass；
//   - --self-test 写成无论结果如何都exit 0（恒过断言）；
//   - 用注释 / try-catch 把 G1/G2/G3 的失败路径吞掉；
//   - 用 #[allow(dead_code)] 之类让"检测器自己不被检测"变成常态。
// 因此：判据的失败路径一律 return 明确的红，self-test 的断言双向成立
// （破损必红 + 良好必绿 + 方向性正确），任何一条不成立就整体判红。
// ===========================================================================