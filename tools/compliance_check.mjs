#!/usr/bin/env node
// ===========================================================================
// compliance_check.mjs — 合规自检器（合规专章的可执行部分）
//
// 存在理由：用户 2026-10-06 明确要求「文件夹里面一定要有一个部分是专门用来
// 合规的」。文档会腐烂，判据不会——所以合规的关键判据写成代码，并接进 CI。
//
// ��查三件事（任一判红 ⇒ 退出码 1）：
//   C1 零第三方代码依赖 —— 26 个 Cargo.toml 的 [dependencies] 只能是本仓
//                          path 依赖；无 build.rs；无原生 C/C++ 链接。
//   C2 资产登记完备—— `docs/合规/assets/资产登记.md` 里每项资产都必须
//                          填全「许可 / 来源 / 自研化状态」。**空字段判红**。
//   C3 归属声明存在—— THIRD-PARTY-NOTICES.md 必须在场且含关键区分句
//                         （数据资产 ≠ 代码依赖）。
//
// 用法：
//   node tools/compliance_check.mjs--self-test   反向自证：故意注入违规，
//                                               必须判红（**不许恒过**）
//   node tools/compliance_check.mjs              对真实仓库跑判据
//   node tools/compliance_check.mjs --help
//
// 退出码：0 = 全绿；1 = 判红或自证失败；2 = 参数错误
//
// 纪律（本工具自身必须遵守，理由见文件末尾「为什么 --self-test 不许恒过」）：
//   - 零第三方依赖：只用 node: 内置模块，CI 无需 npm install。
//   - 检测逻辑**恒定生效**：不存在开关能把某项检查变成"跳过"。
//   - `--self-test` 是**反向自证**：它把违规 fixture 喂给同一套判据，必须判红；
//     若判绿，self-test 自己判红（退出码 1）。恒过的自证等于没有自证。
// ===========================================================================

import {
  existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/**
 * 源码/合规目录探测：同时支持两种布局——
 *   仓库：crates/ + docs/合规/
 *   分发包：src/ + Compliance/
 * 优先取存在者。这样同一份脚本在仓库内与解包后的分发包里都能跑。
 */
function pickDir(...candidates) {
  for (const c of candidates) {
    try { if (existsSync(c)) return c; } catch { /* 继续 */ }
  }
  return candidates[0]; // 都不存在时返回首选，便于报错信息指向期望路径
}
const CRATES_DIR = pickDir(path.join(REPO_ROOT, 'crates'), path.join(REPO_ROOT, 'src'));
const REGISTRY_MD = pickDir(
  path.join(REPO_ROOT, 'docs', '合规', 'assets', '资产登记.md'),
  path.join(REPO_ROOT, 'Compliance', 'assets', '资产登记.md'),
);
const NOTICES_MD = pickDir(
  path.join(REPO_ROOT, 'docs', '合规', 'THIRD-PARTY-NOTICES.md'),
  path.join(REPO_ROOT, 'Compliance', 'THIRD-PARTY-NOTICES.md'),
);
const SELF_TEST_DIR_PREFIX = 'compliance-selftest-';

// 本仓 crate 命名空间。凡依赖名以此开头 → 本仓；否则就是第三方。
const OWN_NAMESPACE = 'prism-';

// C2 要求的资产登记列。**空字段判红**——这是本脚本的核心判据之一。
const REQUIRED_ASSET_COLUMNS = [
  '名称', '来源', '许可', '是否随包分发', '使用位置', '自研化状态',
];

// C3 要求归属声明里必须出现的关键区分句（三条全在才算过）。
// 理由：数据资产与代码依赖是两回事，这个区分必须写在归属声明里，
// 否则读者会误以为"用了字体/方向数表 = 引入第三方库"。
const REQUIRED_NOTICE_PHRASES = [
  '不含第三方代码依赖',
  '数据资产',
  '代码依赖',
];

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

const results = [];

function pass(id, msg, detail = '') {
  results.push({ id, ok: true, msg, detail });
  console.log(`  ok    [${id}] ${msg}${detail ? `  ${DIM}${detail}${RESET}` : ''}`);
}
function fail(id, msg, detail = '') {
  results.push({ id, ok: false, msg, detail });
  console.log(`  ${RED}FAIL${RESET}  [${id}] ${msg}${detail ? `\n         ${detail}` : ''}`);
}
function head(title) {
  console.log(`\n== ${title} ==`);
}

/** 递归列出目录下所有文件（相对路径，POSIX 分隔符），可排除若干目录名。 */
function walk(dir, base = dir, skip = new Set()) {
  if (!existsSync(dir)) return [];
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (skip.has(entry.name)) continue;
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(abs, base, skip));
    else out.push(path.relative(base, abs).split(path.sep).join('/'));
  }
  return out;
}

/** 极简 TOML 段解析：只要 [dependencies] / [dev-dependencies] / [build-dependencies]
 *  下形如 `name = { ... }` 或 `name = "ver"` 的键。不引第三方 TOML 解析器。 */
function parseDependencySections(toml) {
  const out = { deps: [], devDeps: [], buildDeps: [] };
  const lines = toml.split(/\r?\n/);
  let section = null;
  for (const raw of lines) {
    const line = raw.trim();
    const sec = line.match(/^\[([^\]]+)\]/);
    if (sec) {
      const name = sec[1].trim();
      if (name === 'dependencies' || name.startsWith('dependencies.')) section = 'deps';
      else if (name === 'dev-dependencies' || name.startsWith('dev-dependencies.')) section = 'devDeps';
      else if (name === 'build-dependencies' || name.startsWith('build-dependencies.')) section = 'buildDeps';
      else section = null;
      continue;
    }
    if (!section || !line || line.startsWith('#')) continue;
    const kv = line.match(/^([A-Za-z0-9_\-]+)\s*=/);
    if (kv) out[section].push(kv[1]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// C1 · 零第三方代码依赖
// ---------------------------------------------------------------------------
function checkNoThirdPartyCode(repoRoot) {
  head('C1 · 零第三方代码依赖');

  const cratesDir = pickDir(path.join(repoRoot, 'crates'), path.join(repoRoot, 'src'));
  if (!existsSync(cratesDir)) {
    fail('C1.0', '找不到 crates/ 或 src/ 目录', `路径：${cratesDir}`);
    return;
  }
  const crateNames = readdirSync(cratesDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();

  if (crateNames.length === 0) {
    fail('C1.0', 'crates/ 下没有任何 crate');
    return;
  }

  // C1.1 依赖白名单：只允许本仓 path 依赖
  const thirdParty = [];
  let totalDeps = 0;
  for (const crate of crateNames) {
    const tomlPath = path.join(cratesDir, crate, 'Cargo.toml');
    if (!existsSync(tomlPath)) {
      fail('C1.1', `${crate}缺少 Cargo.toml`);
      continue;
    }
    const parsed = parseDependencySections(readFileSync(tomlPath, 'utf8'));
    for (const [bucket, list] of Object.entries(parsed)) {
      for (const dep of list) {
        totalDeps += 1;
        if (!dep.startsWith(OWN_NAMESPACE)) {
          thirdParty.push(`${crate} → ${dep}（${bucket}）`);
        }
      }
    }
  }
  if (thirdParty.length === 0) {
    pass('C1.1', `全部 ${totalDeps} 条依赖均为本仓 path 依赖`, `${crateNames.length} 个 crate，零第三方`);
  } else {
    fail('C1.1', `发现 ${thirdParty.length} 条第三方代码依赖`, thirdParty.join('\n         '));
  }

  // C1.2 无 build.rs（构建期代码生成是供应链面）
  const buildRs = walk(cratesDir, cratesDir, new Set(['target', 'node_modules']))
    .filter((rel) => path.basename(rel) === 'build.rs');
  if (buildRs.length === 0) {
    pass('C1.2', '无 build.rs（无构建期代码生成）');
  } else {
    fail('C1.2', `发现 ${buildRs.length} 个 build.rs`, buildRs.join('\n         '));
  }

  // C1.3 无原生 C/C++ 链接指令（extern "C" 仅允许 wasm32 JS 胶水）
  const rustFiles = walk(cratesDir, cratesDir, new Set(['target', 'node_modules']))
    .filter((rel) => rel.endsWith('.rs'));
  const nativeLink = [];
  for (const rel of rustFiles) {
    const text = readFileSync(path.join(cratesDir, rel), 'utf8');
    for (let i = 0; i < text.split(/\r?\n/).length; i += 1) {
      const line = text.split(/\r?\n/)[i];
      if (/\blink!\s*\(/.test(line) && !/^\s*\/\//.test(line)) {
        nativeLink.push(`${rel}:${i + 1} ${line.trim()}`);
      }
    }
  }
  if (nativeLink.length === 0) {
    pass('C1.3', '无原生链接指令（link!）', `${rustFiles.length} 个 .rs 已扫`);
  } else {
    fail('C1.3', `发现 ${nativeLink.length} 处 link! 指令`, nativeLink.slice(0, 5).join('\n         '));
  }

  // C1.4 Cargo.lock 只含本仓包（独立第二证据，与 C1.1 互为交叉验证）
  const lockPath = path.join(repoRoot, 'Cargo.lock');
  if (!existsSync(lockPath)) {
    fail('C1.4', '缺少 Cargo.lock（无法核验锁定内容）');
  } else {
    const lock = readFileSync(lockPath, 'utf8');
    const names = [...lock.matchAll(/^name = "([^"]+)"/gm)].map((m) => m[1]);
    const foreign = names.filter((n) => !n.startsWith(OWN_NAMESPACE));
    if (foreign.length === 0) {
      pass('C1.4', `Cargo.lock 锁定 ${names.length} 个包，全部为本仓 crate`);
    } else {
      fail('C1.4', `Cargo.lock 含非本仓包：${foreign.join(', ')}`);
    }
  }
}

// ---------------------------------------------------------------------------
// C2 · 资产登记完备（空字段判红）
// ---------------------------------------------------------------------------

/** 解析 Markdown 表格 → 行数组（每行是单元格数组）。 */
function parseTables(md) {
  const tables = [];
  let cur = null;
  for (const raw of md.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line.startsWith('|')) {
      if (cur) { tables.push(cur); cur = null; }
      continue;
    }
    const cells = line.split('|').slice(1, -1).map((c) => c.trim());
    if (cur) cur.rows.push(cells);
    else cur = { header: cells, rows: [] };
  }
  if (cur) tables.push(cur);
  return tables;
}

/** 判定"空"：空串、占位符、待填标记都算空——占位符是最隐蔽的假绿。 */
function isBlankCell(v) {
  if (!v) return true;
  const s = v.replace(/`/g, '').trim();
  if (s === '') return true;
  if (/^(tbd|todo|n\/?a|待定|待填|待补|未知|无|空|-+|\?+)$/i.test(s)) return true;
  return false;
}

function checkAssetRegistry(repoRoot) {
  head('C2 · 资产登记完备（空字段判红）');

  const regPath = pickDir(
    path.join(repoRoot, 'docs', '合规', 'assets', '资产登记.md'),
    path.join(repoRoot, 'Compliance', 'assets', '资产登记.md'),
  );
  if (!existsSync(regPath)) {
    fail('C2.0', '找不到资产登记文件', `期望路径：${regPath}`);
    return { assetCount: 0 };
  }
  const md = readFileSync(regPath, 'utf8');
  const tables = parseTables(md);

  // 找主表：表头必须含全部必需列
  const REQUIRED = REQUIRED_ASSET_COLUMNS;
  const main = tables.find((t) => {
    const h = t.header.join('|');
    return REQUIRED.every((c) => h.includes(c));
  });
  if (!main) {
    fail('C2.0', '资产登记表未找到含全部必需列的表格', `必需列：${REQUIRED.join(' / ')}`);
    return { assetCount: 0 };
  }
  pass('C2.0', `主表列齐全：${REQUIRED.join(' / ')}`);

  // 列索引
  const idx = {};
  main.header.forEach((h, i) => { idx[h.replace(/`/g, '').trim()] = i; });

  const blanks = [];
  let assetCount = 0;
  main.rows.forEach((cells, r) => {
    // 跳过分隔行（形如 --- 或 :---）
    if (cells.every((c) => /^:?-{2,}:?$/.test(c.replace(/`/g, '').trim()))) return;
    // 跳过空行
    if (cells.every((c) => c === '')) return;
    assetCount += 1;
    for (const col of REQUIRED) {
      const i = idx[col];
      if (i === undefined) continue;
      const v = cells[i] ?? '';
      if (isBlankCell(v)) {
        blanks.push(`第 ${r + 1} 行「${cells[idx['名称']] ?? '?'}」的「${col}」为空/占位符`);
      }
    }
  });

  if (assetCount === 0) {
    fail('C2.1', '资产登记表有列头但零数据行', '登记表必须逐项列出外部资产');
  } else if (blanks.length === 0) {
    pass('C2.1', `登记 ${assetCount} 项资产，必需字段全部非空`);
  } else {
    fail('C2.1', `发现 ${blanks.length} 处空/占位符字段`, blanks.join('\n         '));
  }

  // C2.2 每项必须有自研化状态，且取值必须是受控词表（防止"待定"混过）
  const ALLOWED_STATUS = ['已自研', '待自研', '过渡态', '未引入', '已结清'];
  const iStatus = idx['自研化状态'];
  const iName = idx['名称'];
  const badStatus = [];
  if (iStatus !== undefined) {
    main.rows.forEach((cells, r) => {
      if (cells.every((c) => /^:?-{2,}:?$/.test(c.replace(/`/g, '').trim()))) return;
      if (cells.every((c) => c === '')) return;
      const st = (cells[iStatus] ?? '').replace(/`/g, '').trim();
      if (!ALLOWED_STATUS.some((a) => st.includes(a))) {
        badStatus.push(`「${cells[iName] ?? '?'}」的状态是「${st}」，不在受控词表内`);
      }
    });
  }
  if (badStatus.length === 0) {
    pass('C2.2', `自研化状态全部落在受控词表（${ALLOWED_STATUS.join('/')}）`);
  } else {
    fail('C2.2', `${badStatus.length} 项自研化状态不受控`, badStatus.join('\n         '));
  }

  return { assetCount };
}

// ---------------------------------------------------------------------------
// C3 · 归属声明存在且含关键区分
// ---------------------------------------------------------------------------
function checkNotices(repoRoot) {
  head('C3 · 第三方归属声明');

  const p = pickDir(
    path.join(repoRoot, 'docs', '合规', 'THIRD-PARTY-NOTICES.md'),
    path.join(repoRoot, 'Compliance', 'THIRD-PARTY-NOTICES.md'),
  );
  if (!existsSync(p)) {
    fail('C3.0', '找不到 THIRD-PARTY-NOTICES.md', `期望路径：${p}`);
    return;
  }
  pass('C3.0', 'THIRD-PARTY-NOTICES.md 在场');
  const md = readFileSync(p, 'utf8');
  const missing = REQUIRED_NOTICE_PHRASES.filter((s) => !md.includes(s));
  if (missing.length === 0) {
    pass('C3.1', '含关键区分句（数据资产 ≠ 代码依赖）',
      REQUIRED_NOTICE_PHRASES.map((s) => `「${s}」`).join(' '));
  } else {
    fail('C3.1', `归属声明缺少关键区分句：${missing.join('、')}`,
      '必须写清"不含第三方代码依赖"，并把数据/资产类归属单列');
  }
}

// ---------------------------------------------------------------------------
// 汇总
// ---------------------------------------------------------------------------
function summarize() {
  const total = results.length;
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${'='.repeat(64)}`);
  if (failed.length === 0) {
    console.log(`${GREEN}合规自检通过${RESET}：${total} 项判据全部通过`);
    return 0;
  }
  console.log(`${RED}合规自检判红${RESET}：${total} 项中 ${failed.length} 项失败`);
  for (const f of failed) console.log(`  - [${f.id}] ${f.msg}`);
  return 1;
}

function runAll(repoRoot) {
  results.length = 0;
  console.log('compliance_check.mjs — 合规自检器');
  console.log(`${DIM}仓库根：${repoRoot}${RESET}`);
  console.log(`${DIM}判据：C1 零第三方代码依赖 / C2 资产登记完备 / C3 归属声明${RESET}`);
  checkNoThirdPartyCode(repoRoot);
  checkAssetRegistry(repoRoot);
  checkNotices(repoRoot);
  return summarize();
}

// ---------------------------------------------------------------------------
// --self-test · 反向自证（**不许恒过**）
// ---------------------------------------------------------------------------
function selfTest() {
  console.log('compliance_check.mjs --self-test · 反向自证');
  console.log(`${DIM}意图：注入违规 fixture，断言判据必须判红。判据恒真 ⇒ 本自证失败。${RESET}\n`);

  const tmp = mkdtempSync(path.join(tmpdir(), SELF_TEST_DIR_PREFIX));

  /** 在临时仓库里造一个 crate。 */
  const mkCrate = (name, manifest, extra = {}) => {
    const d = path.join(tmp, 'crates', name);
    mkdirSync(d, { recursive: true });
    writeFileSync(path.join(d, 'Cargo.toml'), manifest);
    writeFileSync(path.join(d, 'lib.rs'), extra.libRs ?? 'pub fn f() {}\n');
    if (extra.buildRs !== undefined) writeFileSync(path.join(d, 'build.rs'), extra.buildRs);
    return d;
  };

  // 基线：锁定项里混进一个外部包（场景 1/2/3 都靠它触发 C1.4）
  writeFileSync(
    path.join(tmp, 'Cargo.lock'),
    'version = 4\n\n[[package]]\nname = "libc"\nversion = "0.2"\n',
  );

  // 违规 crate：第三方依赖 + build.rs（触发 C1.1 与 C1.2）
  mkCrate('bad-dep',
    '[package]\nname = "bad-dep"\nversion = "0.1.0"\n\n[dependencies]\nlibc = "0.2"\n',
    { buildRs: 'fn main() {}\n' });

  // 合规 crate：纯本仓 path 依赖（作为对照，证明判据不是恒红）
  mkCrate('good-dep',
    '[package]\nname = "good-dep"\nversion = "0.1.0"\n\n[dependencies]\nprism-math = { path = "../prism-math" }\n');

  const regDir = path.join(tmp, 'docs', '合规', 'assets');
  mkdirSync(regDir, { recursive: true });

  const scenarios = [];

  // 场景 1：第三方代码依赖必须判红
  results.length = 0;
  checkNoThirdPartyCode(tmp);
  scenarios.push({
    name: 'C1 第三方代码依赖（libc）',
    ok: results.some((r) => !r.ok && r.id === 'C1.1'),
    detail: '期望 C1.1 判红',
  });

  // 场景 2：build.rs 必须判红
  results.length = 0;
  checkNoThirdPartyCode(tmp);
  scenarios.push({
    name: 'C1 build.rs 存在',
    ok: results.some((r) => !r.ok && r.id === 'C1.2'),
    detail: '期望 C1.2 判红',
  });

  // 场景 3：Cargo.lock 含外部包必须判红
  results.length = 0;
  checkNoThirdPartyCode(tmp);
  scenarios.push({
    name: 'C1 Cargo.lock 含 libc',
    ok: results.some((r) => !r.ok && r.id === 'C1.4'),
    detail: '期望 C1.4 判红',
  });

  // 场景 4：空许可字段必须判红（**本脚本的核心判据**）
  const mdBlank = [
    '# 资产登记',
    '',
    '| 名称 | 来源 | 许可 | 是否随包分发 | 使用位置 | 自研化状态 |',
    '| --- | --- | --- | --- | --- | --- |',
    '| 测试字体 | 某上游 |  | 是 | crates/x | 过渡态 |',
    '',
  ].join('\n');
  writeFileSync(path.join(regDir, '资产登记.md'), mdBlank);
  results.length = 0;
  checkAssetRegistry(tmp);
  scenarios.push({
    name: 'C2 许可字段为空',
    ok: results.some((r) => !r.ok && r.id === 'C2.1'),
    detail: '期望 C2.1 判红',
  });

  // 场景 5：占位符「待定」也必须判红（最隐蔽的假绿）
  const mdTbd = mdBlank.replace('|  | 是 |', '| 待定 | 是 |');
  writeFileSync(path.join(regDir, '资产登记.md'), mdTbd);
  results.length = 0;
  checkAssetRegistry(tmp);
  scenarios.push({
    name: 'C2 许可字段为占位符「待定」',
    ok: results.some((r) => !r.ok && r.id === 'C2.1'),
    detail: '期望 C2.1 判红（占位符不算已填）',
  });

  // 场景 6：自研化状态不受控必须判红
  const mdBadStatus = [
    '# 资产登记',
    '',
    '| 名称 | 来源 | 许可 | 是否随包分发 | 使用位置 | 自研化状态 |',
    '| --- | --- | --- | --- | --- | --- |',
    '| 测试字体 | 某上游 | OFL 1.1 | 是 | crates/x | 看情况吧 |',
    '',
  ].join('\n');
  writeFileSync(path.join(regDir, '资产登记.md'), mdBadStatus);
  results.length = 0;
  checkAssetRegistry(tmp);
  scenarios.push({
    name: 'C2 自研化状态不受控',
    ok: results.some((r) => !r.ok && r.id === 'C2.2'),
    detail: '期望 C2.2 判红',
  });

  // 场景 7：全字段填全的登记表应当判绿（证明判据不是恒红）
  const mdGood = [
    '# 资产登记',
    '',
    '| 名称 | 来源 | 许可 | 是否随包分发 | 使用位置 | 自研化状态 |',
    '| --- | --- | --- | --- | --- | --- |',
    '| 测试字体 | 某上游 | OFL 1.1 | 是 | crates/x | 过渡态 |',
    '',
  ].join('\n');
  writeFileSync(path.join(regDir, '资产登记.md'), mdGood);
  results.length = 0;
  checkAssetRegistry(tmp);
  const greenOk = results.every((r) => r.ok);
  scenarios.push({
    name: 'C2 字段填全（对照组，应判绿）',
    ok: greenOk,
    detail: '期望全部通过——证明判据不是恒红',
  });

  // 场景 8：归属声明缺关键区分句必须判红
  writeFileSync(path.join(tmp, 'docs', '合规', 'THIRD-PARTY-NOTICES.md'), '# 归属声明\n\n本项目使用了字体。\n');
  results.length = 0;
  checkNotices(tmp);
  scenarios.push({
    name: 'C3 归属声明缺关键区分句',
    ok: results.some((r) => !r.ok && r.id === 'C3.1'),
    detail: '期望 C3.1 判红',
  });

  // ---- 汇总 ----
  console.log('');
  let bad = 0;
  for (const s of scenarios) {
    if (s.ok) {
      console.log(`  ${GREEN}ok${RESET}    ${s.name}${DIM}  ${s.detail}${RESET}`);
    } else {
      bad += 1;
      console.log(`  ${RED}FAIL${RESET}  ${s.name}  ${s.detail}`);
    }
  }
  rmSync(tmp, { recursive: true, force: true });

  console.log('');
  if (bad === 0) {
    console.log(`${GREEN}self-test: ${scenarios.length}/${scenarios.length} 通过${RESET}`);
    console.log('SELF-TEST OK: 违规必红、良好必绿——判据具备双向判别力');
    return 0;
  }
  console.log(`${RED}self-test: ${scenarios.length - bad}/${scenarios.length} 通过${RESET}`);
  console.log('SELF-TEST FAILED: 判据未能判红，它可能已经恒过——**这是最坏的情况**');
  return 1;
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(`compliance_check.mjs — 合规自检器

用法：
  node tools/compliance_check.mjs --self-test   反向自证：注入违规 fixture，必须判红
  node tools/compliance_check.mjs              对真实仓库跑判据

判据：
  C1 零第三方代码依赖：所有 Cargo.toml 的依赖只能是本仓 path 依赖；
                        无 build.rs；无 link!；Cargo.lock 只含本仓包
  C2 资产登记完备：docs/合规/assets/资产登记.md 每项字段非空（空/占位符判红），
                        自研化状态须落在受控词表
  C3 归属声明：THIRD-PARTY-NOTICES.md 在场且含关键区分句

退出码：0 全绿/ 自证通过；1 判红或自证失败；2 参数错误`);
  process.exit(0);
} else if (argv.includes('--self-test')) {
  process.exit(selfTest());
} else if (argv.length > 0) {
  console.error(`未知参数：${argv.join(' ')}（用 --help 看用法）`);
  process.exit(2);
} else {
  process.exit(runAll(REPO_ROOT));
}

// ===========================================================================
// 为什么 --self-test 不许恒过
// ---------------------------------------------------------------------------
// 本脚本的第3 项判据（C2 空字段判红）是一个**否定式判据**：它在"字段为空"
// 时报红。这类判据最危险的失效模式是**恒过**——如果有人为了"让 CI 变绿"
// 把 `blanks.length === 0` 改成 `blanks.length >= 0`，脚本会永远返回 0，
// 而仓库里所有资产的许可字段可以全部空着，CI 依然全绿。这就是**假绿**。
//
// 所以 --self-test 不测"脚本能不能跑"，它测**判据还有没有判别力**：
// 它构造已知违规的 fixture（空许可字段、占位符「待定」、不受控状态、
// 第三方依赖、build.rs、外部包锁定项、缺关键句的归属声明），
// 逐条断言对应判据**必须报红**；同时构造一个全部填好的对照 fixture，
// 断言**必须报绿**——防止判据退化成"恒红"（那样同样无意义）。
//
// 纪律同ghost_guard.mjs：检测逻辑写死在 runAll() 调用的三个函数里，
// **不存在任何开关能把某项检查变成"跳过"**。恒过的自证等于没有自证。
// ===========================================================================
