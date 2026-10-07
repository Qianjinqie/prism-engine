#!/usr/bin/env node
// hygiene_scan.mjs — 开源卫生扫描器（零依赖，仅 node 内建）
//
// 用途：确保对外开源的仓库不含内部过程痕迹——团队内部角色称谓、内部排期编号、
// 内部过程文档引用、以及"提及用户"的表述。
//
// 存在理由：手工 grep 容易漏。曾有一轮清理中，扫描器的 `--include` 集合漏了
// `.toml`，导致 9 处藏在 Cargo.toml 注释里的内部编号整整一轮没被发现。
// 本脚本把「文件类型集合」和「词表」固化在一起，避免同类盲区重演。
//
// 用法：
//   node tools/hygiene_scan.mjs           # 扫描，命中即 exit 1
//   node tools/hygiene_scan.mjs --list    # 列出白名单命中项（不判红）
//   node tools/hygiene_scan.mjs --self-test# 自证：注入 → 判红；移除 → 判绿
//
// 退出码：0 = 干净（或仅白名单命中）；1 = 有内部痕迹；2 = 参数/环境错误

import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, rmSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const SELF = join(ROOT, 'tools/hygiene_scan.mjs');

// 源码根目录：仓库布局是 crates/，单文件源码分发包布局是 src/。
// 词表与白名单里记的是仓库布局路径，按实际存在的目录重映射。
const SRC_ROOT = ['crates', 'src'].find((d) => existsSync(join(ROOT, d))) ?? 'crates';

/** 把词表/白名单里记录的仓库布局路径映射到当前布局。 */
function localize(rel) {
  return rel.startsWith('crates/') ? SRC_ROOT + rel.slice('crates'.length) : rel;
}

/** 扫描覆盖的文件扩展名 —— **这份清单本身就是要固化的核心资产** */
const EXTENSIONS = ['.rs', '.md', '.mjs', '.js', '.toml', '.yml', '.yaml', '.json', '.sh', '.txt'];

/** 从不进入仓库的目录（构建产物 / 本地工作区 / 版本库内部） */
const EXCLUDED_DIRS = new Set([
  'target', 'dist', 'node_modules', '.git', '.codebuddy', 'ghost_baseline',
]);

/**
 * 内部痕迹词表。每项 [正则, 说明]。分类仅用于报告，匹配逻辑统一。
 *
 * 说明：本表是「内部痕迹黑名单」的字面量。base64 存放有两个理由：
 *   1) 扫描器自身不应把被扫的词明文写进源码——否则分发包里会出现内部称谓；
 *   2) 词表对读者无意义，是纯噪音。
 * 维护时用 `node tools/hygiene_scan.mjs --encode '<文本>'` 生成 base64（切勿手写）。
 */
const ENCODED_PATTERNS = [
  // —— 团队内部角色称谓 ——
  ['5Li755CG5Lq6fOm9kOa0u+aelw==', 'A 类 · 角色称谓'],
  ['5a+H6LGG56CBfOmrmOingei/nHzkuKXov4flhbN86K645riF5qWa', 'A 类 · 角色称谓'],

  // —— "提及用户"（对外文档不应出现"这是某个人说的"） ——
  ['55So5oi35Y6f6K+dfOeUqOaIt+WPjemmiHznlKjmiLfmjIfnpLp855So5oi35aeU5omYfOeUqOaIt+S6suiHqnznlKjmiLfmi43mnb9855So5oi35Yaz5a6afOeUqOaIt+imgeaxgg==', 'B 类 · 提及用户'],

  // —— 内部过程文档术语 ——
  ['6ZSA6LSmfOi/mOi0pnzov5jmuIV85qyg6LSmfOW+hei/mHznu5PotKY=', 'C 类 · 过程文档术语'],
  ['6LSmW+S4gOS6jOS4ieWbm+S6lOWFreS4g+WFq+S5neWNgV0=', 'C 类 · 过程文档术语'],
  ['6LSm5pys', 'C 类 · 过程文档术语'],

  // —— 内部工作文档（随仓库不存在，读者无从查证） ——
  ['5Lu75Yqh5Lmm', 'D 类 · 内部文档名'],
  ['5paH5qGj5YWr5Lu25aWXfOW3rui3nei0puacrHzmlr3lt6Xml6XljoZ85pa95bel5omL5YaMfOW8uuWkp+eJiOi3r+e6v+WbvnzojIPlvI/liJvmlrDnoJTnqbZ86Zeo56aB5by65bqm5a6h6K6hfOaJi+acuua1i+ivlQ==', 'D 类 · 内部文档名'],

  // —— 内部项目管理词 ——
  ['5rOi5qyhfOWGsuWIug==', 'E 类 · 内部管理词'],

  // —— 内部排期编号（工作轮次编号与并发分路编号） ——
  ['XGJXWzAtOV17MSwyfVxi', 'F 类 · 内部编号'],
  ['XGJXWzAtOV0tW0EtWjAtOV0rXGI=', 'F 类 · 内部编号'],
  ['6LevW0EtWjAtOV186LevWzEtOV0=', 'F 类 · 内部编号'],

  // —— 排期术语（内部工程管理用语，不进对外文档） ——
  ['5pa95bel', 'G 类 · 排期术语'],
  ['5pel5Y6G', 'G 类 · 排期术语'],
  ['6Lev57q/5Zu+', 'G 类 · 排期术语'],
];

/** base64 → UTF-8 明文字符串（词表解码；与 `--encode` 互逆） */
function decode(b64) {
  return Buffer.from(b64, 'base64').toString('utf8');
}

const PATTERNS = ENCODED_PATTERNS.map(([b64, cat]) => [new RegExp(decode(b64), 'g'), cat]);

/**
 * 白名单：已知且经裁决保留的命中。
 * 每条必须写明保留理由——白名单是审查结果，不是例外开关。
 */
const WHITELIST = [
  {
    file: 'crates/prism-io/src/usd.rs',
    line: 3234,
    match: 'U8',
    reason: 'test-only 局部常量 `U8` = 凹八边形 U 形夹具（8 顶点），与内部编号 U8 字符巧合。禁改标识符。',
  },
  {
    file: 'crates/prism-io/src/usd.rs',
    line: 3266,
    match: 'U8',
    reason: '消费上方 `const U8` 的 Case 构造，同一 test 夹具组（ARROW5 / HEX6 / U8）。',
  },
];

/**
 * 附加白名单：某些词形是技术义，不是内部痕迹。
 * 这些不参与判红，但记录在案以便复核。
 */
const TECH_TERMS = [
  '记账',   // 吞吐记账 / 复用记账（farm/queue.rs、incremental.rs）
  '对账',   // 与参考模型逐点对账（prism-text 独立探针测试语义）
  '批次',   // 渲染批次 / wavefront 批次（技术义；内部管理义一律改写为"轮次"）
];

function* walk(dir) {
  for (const entry of readdirSync(dir)) {
    if (EXCLUDED_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) yield* walk(full);
    else if (EXTENSIONS.some((ext) => entry.endsWith(ext))) yield full;
  }
}

function isWhitelisted(file, line, match) {
  return WHITELIST.some(
    (w) => localize(w.file) === file && w.line === line && w.match === match,
  );
}

function scan() {
  const findings = [];
  for (const full of walk(ROOT)) {
    const rel = relative(ROOT, full);
    if (rel === relative(ROOT, SELF)) continue; // 扫描器自身含词表，不自指
    const lines = readFileSync(full, 'utf8').split('\n');
    lines.forEach((text, idx) => {
      const lineNo = idx + 1;
      for (const [re, category] of PATTERNS) {
        re.lastIndex = 0;
        let m;
        while ((m = re.exec(text)) !== null) {
          if (isWhitelisted(rel, lineNo, m[0])) continue;
          findings.push({ file: rel, line: lineNo, match: m[0], category, text: text.trim().slice(0, 120) });
        }
      }
    });
  }
  return findings;
}

function report(findings) {
  if (findings.length === 0) {
    console.log('开源卫生扫描：干净（0 处内部痕迹）');
    console.log(`  覆盖扩展名 ${EXTENSIONS.length} 种· 白名单 ${WHITELIST.length} 项`);
    return 0;
  }
  const byCat = new Map();
  for (const f of findings) {
    if (!byCat.has(f.category)) byCat.set(f.category, []);
    byCat.get(f.category).push(f);
  }
  console.error(`开源卫生扫描：发现 ${findings.length} 处内部痕迹\n`);
  for (const [cat, items] of byCat) {
    console.error(`── ${cat} · ${items.length} 处 ──`);
    for (const f of items.slice(0, 12)) {
      console.error(`  ${f.file}:${f.line}  「${f.match}」  ${f.text}`);
    }
    if (items.length > 12) console.error(`  … 另有 ${items.length - 12} 处`);
    console.error('');
  }
  console.error(`另：技术义保留词（不判红）${TECH_TERMS.join(' / ')}；白名单 ${WHITELIST.length} 项。`);
  console.error('处置原则：只删「内部叙述」，绝不删「技术信息」。');
  return 1;
}

function selfTest() {
  console.log('自证：注入内部痕迹 → 必须判红；移除 → 必须判绿\n');
  const probe = join(ROOT, `${SRC_ROOT}/prism-core/src/__hygiene_probe.rs`);
  const cases = [
    // 全部探针（label 与 text）均 base64 编码：本文件自身不得含被扫的词面，
    // 否则分发包里会出现内部称谓。维护时用 `--encode '<文本>'` 生成。
    [decode('5aSa5pm66IO95L2T5oiQ5ZGY5ZCN'), decode('Ly8hIOeUseS4u+eQhuS6uum9kOa0u+ael+WNj+iwg++8jOWvh+ixhueggeWunueOsOOAgg==')],
    [decode('5o+Q5Y+K55So5oi3'), decode('Ly8g55So5oi35Y6f6K+d77ya6L+Z5Liq6K6+6K6h5piv5a+555qE44CC')],
    [decode('6L+Y6LSm6K+t5rGH'), decode('Ly8g6LSm5ZubIFU4IOe7k+i0pu+8jOmUgOi0puWujOaIkOOAgg==')],
    [decode('5YaF6YOo5paH5qGj'), decode('Ly8g6K+m6KeBIGRvY3Mv5paH5qGj5YWr5Lu25aWXL+W3rui3nei0puacrC5tZA==')],
    [decode('5bel5L2c5om55qyh57yW5Y+3'), decode('Ly8hIOOAkFcxNCDot68xIOS/ruato+OAkemHj+WMluatpemVvw==')],
    [decode('5o6S5pyf5pyv6K+twrfmlr3lt6U='), decode('Ly8g5pa95bel5pyf5Li05pe25byA5YWz')],
    [decode('5oqA5pyv5LmJwrforrDotKbvvIjkuI3liKTnuqLvvIk='), decode('Ly8hIOWQnuWQkOiusui0puaVsOWumuWHuuWKn+mBruW6j+OAgg==')],
    [decode('5oqA5pyv5LmJwrflr7notKbvvIjkuI3liKTnuqLvvIk='), decode('Ly8hIOS4juWPguiAiOaooeWei+mAlueCueWvuei0puOAgg==')],
  ];
  let pass = 0, fail = 0;
  try {
    for (const [label, text] of cases) {
      const shouldBeRed = !label.startsWith('技术义');
      writeFileSync(probe, text + '\n');
      const red = scan().length > 0;
      const ok = red === shouldBeRed;
      console.log(`  ${ok ? '✓' : '✗'} ${label} → ${red ? '判红' : '判绿'}（期望${shouldBeRed ? '红' : '绿'}）`);
      ok ? pass++ : fail++;
    }
  } finally {
    try { rmSync(probe); } catch { /* 探针可能未创建 */ }
  }
  // 仓库真实状态须判绿
  const real = scan();
  const realOk = real.length === 0;
  console.log(`  ${realOk ? '✓' : '✗'} 真实仓库当前状态 → ${real.length} 处（期望 0）`);
  realOk ? pass++ : fail++;
  console.log(`\n自证${fail === 0 ? '通过' : '失败'}：${pass} 通过 / ${fail} 失败`);
  return fail === 0 ? 0 : 1;
}

function listWhitelist() {
  console.log('白名单（经裁决保留，不判红）：\n');
  for (const w of WHITELIST) {
    console.log(`  ${w.file}:${w.line}  「${w.match}」`);
    console.log(`    理由：${w.reason}\n`);
  }
  console.log('技术义保留词（正则不匹配，不需白名单）：');
  for (const t of TECH_TERMS) console.log(`  ${t}`);
  return 0;
}

// ── 入口 ──
const args = process.argv.slice(2);
try {
  if (args.includes('--encode')) {
    // 维护入口：把明文词表项转成 base64，供 decode() 使用。
    // 词表维护者用它生成 ENCODED_PATTERNS / 自证用例中的 <b64> 串（切勿手写）。
    const idx = args.indexOf('--encode');
    const text = args[idx + 1];
    if (text === undefined || text === '') {
      console.error('用法：--encode <文本>');
      process.exit(2);
    }
    console.log(Buffer.from(text, 'utf8').toString('base64'));
    process.exit(0);
  } else if (args.includes('--self-test')) process.exit(selfTest());
  else if (args.includes('--list')) process.exit(listWhitelist());
  else if (args.includes('--help') || args.includes('-h')) {
    console.log('用法：node tools/hygiene_scan.mjs [--list | --self-test | --encode <文本>]');
    process.exit(0);
  } else if (args.length > 0) {
    console.error('未知参数：' + args.join(' '));
    process.exit(2);
  } else {
    process.exit(report(scan()));
  }
} catch (e) {
  console.error('扫描失败：' + e.message);
  process.exit(2);
}