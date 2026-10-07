#!/usr/bin/env node
// =============================================================================
// pack_source_txt.mjs —— 把应发布内容合并成单文件 TXT 分发包
//
// 格式照抄用户旧引擎（video-engine-source.txt）的成熟格式：
//   1. 头部写"还原成目录树 3 步"
//   2. 内嵌 UNPACK SCRIPT BEGIN/END 之间的可用 Node 脚本（零依赖、Node 18+）
//   3. 后接合并内容，每个文件以 === FILE: 路径 === 分隔
//
// 打包范围（分发包 = 极简三目录 + 构建必需件）：
//   ✅ crates/（→ src/）· docs/合规/（→ Compliance/）· docs/开发者指南.md ·
//      tools/ · .github/ · .cargo/ · examples/ · Cargo.toml · Cargo.lock
//   ❌ 不打包：docs/设计/（设计权威留仓库）· package.json · .gitignore
//      （examples/ 必须带：prism-dsl 的测试用 include_str! 读它，缺了编不过）
//   ❌ 排除：target/ node_modules/ *.bak dist/ .codebuddy/ 及任何中间产物
//
// 用法：node tools/pack_source_txt.mjs [--dry-run]
// =============================================================================

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');
const OUT = path.join(REPO, 'dist', 'prism-engine-v1.0-source.txt');
const VERSION = '1.0';
const PKG_NAME = 'prism-engine-v1.0-source.txt';

// -----------------------------------------------------------------------------
// 打包规则
//
// 分发包结构（极简三目录 + 构建必需件）：
//   src/               ← 由 crates/ 重组
//   Compliance/        ← 由 docs/合规/ 重组
//   开发者指南.md       ← 由 docs/开发者指南.md 重组
//   tools/             ← 门禁脚本
//   .github/workflows/ ← CI 配置
//   examples/           ← prism-dsl 测试用 include_str! 读取，缺了测试编译不过
//   Cargo.toml / Cargo.lock / .cargo/config.toml ← 构建必需件
//
// 不打包：docs/设计/（设计权威，留在仓库，不进分发包）、
//         package.json（tools 全用 node 内建，无 npm 依赖）、.gitignore。
// -----------------------------------------------------------------------------
const INCLUDE_ROOTS = ['crates', 'docs/合规', 'tools', '.github', '.cargo', 'examples'];
const INCLUDE_FILES = ['Cargo.toml', 'Cargo.lock', 'docs/开发者指南.md'];

/**
 * 包内路径重映射：仓库路径 → 分发包路径。
 * 只改**包内**的呈现，仓库目录结构保持原样（crates/ 不动）。
 */
function remap(rel) {
  if (rel === 'docs/开发者指南.md') return '开发者指南.md';
  if (rel.startsWith('docs/合规/')) return rel.slice('docs/合规/'.length).replace(/^/, 'Compliance/');
  if (rel === 'docs/合规') return 'Compliance';
  if (rel.startsWith('crates/')) return 'src/' + rel.slice('crates/'.length);
  return rel;
}

/** 包内内容改写：仅对特定文件做精确替换（仓库原文件不动） */
function rewriteContent(rel, text) {
  // 根 Cargo.toml：workspace members 指向 src/*
  if (rel === 'Cargo.toml') {
    return text.replace(/members\s*=\s*\["crates\/\*"\]/, 'members = ["src/*"]');
  }
  // 源码里的路径注释与 doc 链接：crates/xxx → src/xxx（包内不存在 crates/）。
  // docs/合规/ → Compliance/；docs/开发者指南.md → 开发者指南.md。
  // 只改**路径字面量**，不碰 crates/ 目录名在别处的用法。
  if (rel.startsWith('crates/') && /\.(rs|wgsl|sh|mjs|js|md|html|json|toml|cpp|h)$/i.test(rel)) {
    return text
      .replace(/\bdocs\/合规\//g, 'Compliance/')
      .replace(/\bdocs\/开发者指南\.md\b/g, '开发者指南.md')
      //带路径的：crates/xxx → src/xxx
      .replace(/\bcrates\/(?=[A-Za-z0-9_.-])/g, 'src/')
      // 光目录名（指源码根）：`crates/` → `src/`
      .replace(/(?<=`|"|'|\s|^)crates\/(?=[`"')\s，。；、）)]|$)/g, 'src/');
  }
  return text;
}

/** 硬排除：目录名（任意层级） */
const EXCLUDE_DIRS = new Set([
  'target', // 13G 构建产物
  'node_modules', // 19M 第三方依赖
  'dist', // 分发包自身（自引用）
  '.codebuddy', // 内部工作区
  '.git',
  'backups', // tools/backups 下的 .bak 备份
]);

/** 硬排除：文件名/扩展名模式 */
const EXCLUDE_FILE_RE = [
  /\.bak$/i, // 备份文件
  /\.tmp$/i,
  /\.swp$/i,
  /^\.DS_Store$/,
  /\.md5$/,
  /\.golden\.tmp$/,
  /\.log$/i,
  /\.wasm$/i, // cargo 构建产物（wasm32 目标），CI 自行重建，不进包
];

const EXCLUDE_EXACT = new Set(['thumbs.db']);

// -----------------------------------------------------------------------------
// 收集文件
// -----------------------------------------------------------------------------

/**
 * 判定是否二进制（含 NUL 字节，或不是合法 UTF-8）。
 *
 * **为什么必须判**：打包把每个文件读成 UTF-8 文本写进TXT，还原时再写回。
 * 二进制（字体 .ttf）过这道utf-8 往返会被**损坏**——实测 23,244 B 的
 * prism_multiscript.ttf 变成 28,389 B 且非逐字节一致，字体指纹即失效，
 * include_bytes! 拿到的字节是错的。**假绿比不打包更坏**：它让人以为字体没问题。
 */
function isBinary(buf) {
  if (buf.includes(0)) return true;
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(buf);
    return false;
  } catch {
    return true;
  }
}

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) {
      if (EXCLUDE_DIRS.has(e.name)) continue;
      walk(path.join(dir, e.name), out);
    } else if (e.isFile()) {
      if (EXCLUDE_FILE_RE.some((re) => re.test(e.name))) continue;
      if (EXCLUDE_EXACT.has(e.name)) continue;
      out.push(path.relative(REPO, path.join(dir, e.name)));
    }
  }
  return out;
}

function collect() {
  const files = [];
  for (const f of INCLUDE_FILES) {
    if (fs.existsSync(path.join(REPO, f))) files.push(f);
  }
  for (const root of INCLUDE_ROOTS) {
    const abs = path.join(REPO, root);
    if (!fs.existsSync(abs)) continue;
    const st = fs.statSync(abs);
    if (st.isDirectory()) walk(abs, files);
    else files.push(root); // 单文件根
  }
  // 排序：保证同样输入 → 逐字节同样的输出（确定性打包）
  return files.sort((a, b) => a.localeCompare(b, 'en'));
}

const SEP = '='.repeat(80);
const END = '=== END OF FILE ==='; // 专用结束标记：内容行绝无可能撞上它

// -----------------------------------------------------------------------------
// 内嵌解包脚本（零依赖，Node 18+）
//
// 块格式（每个文件）：
//   === FILE: 路径 ===
//   === BASE64 ===      ← 可选，二进制（字体 .ttf）
//   === NOEOL ===       ← 可选，原文件末尾无换行
//   ==== 横线 ====        （分隔横线，打包侧写）
//   <正文行，逐行原样>
//
// **不使用"空行分隔"技巧**：正文行就是文件按 '\n' 切分的全部行，
// 还原时 join('\\n') 即可逐字节复原，避免多补/少补一个换行。
// -----------------------------------------------------------------------------
const UNPACK_SCRIPT = `const fs = require('fs'), path = require('path');
const SRC = process.argv[2] || '${PKG_NAME}';
const lines = fs.readFileSync(SRC, 'utf-8').split('\\n');
let cur = null, buf = [], count = 0, b64 = false, noEol = false, need = -1, seen = 0;
const flush = () => {
  if (!cur) return;
  // **按打包侧声明的行数精确截取**，不对内容做任何模式判断。
  // 早期版本靠"弹掉尾部 ==== 横线 / 空行"界定范围，会吃掉正文里恰好
  // 长得像分隔符的行（本文件自身即反例：有一行字面就是结束标记），
  // 造成还原物与原文件不逐字节一致。行数法从根上消除这类歧义。
  if (need >= 0 && seen > need) buf = buf.slice(0, need);
  const dest = path.resolve(process.cwd(), cur);
  const root = path.resolve(process.cwd());
  if (dest !== root && !dest.startsWith(root + path.sep)) {
    console.error('拒绝写出目录树之外的路径:', cur); process.exit(1);
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (b64) {
    fs.writeFileSync(dest, Buffer.from(buf.join(''), 'base64'));
  } else {
    // 有尾换行补回，无尾换行原样 —— 还原物与原文件逐字节相同。
    fs.writeFileSync(dest, buf.join('\\n') + (noEol ? '' : '\\n'));
  }
  count++;
};
for (const line of lines) {
  const m = line.match(/^=== FILE: (.+) ===$/);
  if (m) { flush(); cur = m[1]; buf = []; b64 = false; noEol = false; need = -1; seen = 0; continue; }
  if (cur && need < 0) {
    const n = line.match(/^=== LINES: (\\d+) ===$/);
    if (n) { need = parseInt(n[1], 10); continue; }
    if (line.trim() === '=== BASE64 ===') { b64 = true; continue; }
    if (line.trim() === '=== NOEOL ===') { noEol = true; continue; }
  }
  if (cur) { buf.push(line); seen++; }
}
flush();
console.log('还原完成:', count, '个文件');
`;

// -----------------------------------------------------------------------------
// 打包
// -----------------------------------------------------------------------------
const files = collect();
const dryRun = process.argv.includes('--dry-run');

// 统计
let totalBytes = 0;
for (const f of files) totalBytes += fs.statSync(path.join(REPO, f)).size;

if (dryRun) {
  console.log('文件数:', files.length);
  console.log('原始总字节:', totalBytes);
  const byTop = {};
  for (const f of files) {
    const top = f.split('/')[0];
    byTop[top] = (byTop[top] || 0) + 1;
  }
  console.log('按顶层目录:', JSON.stringify(byTop, null, 2));
  process.exit(0);
}

const header = [
  `PRISM ENGINE — 源码合并（${files.length} 个文件）`,
  '',
  '【还原成目录树：3 步】',
  `1. 把本文件保存为 ${PKG_NAME}（UTF-8 编码）`,
  '2. 把下方 UNPACK SCRIPT BEGIN / END 之间的代码，保存为同目录下的 unpack.cjs',
  '3. 运行: node unpack.cjs',
  '   完成后在当前目录得到完整目录树（需 Node 18+）',
  '',
  '【分发包结构】',
  '   src/                源码（26 个 crate）',
  '   Compliance/         合规声明（零依赖核验 / 第三方声明 / 资产登记）',
  '   开发者指南.md        开发者文档（自包含：上手 / 架构 / 数学内核 / 门禁 / 扩展）',
  '   tools/              门禁脚本',
  '   .github/workflows/  CI 配置',
  '   Cargo.toml / Cargo.lock / .cargo/  构建必需件',
  '',
  '【装依赖与跑测试】（还原之后）',
  '   rustup target add wasm32-unknown-unknown   # wasm 目标（可选，手机端验证链路用）',
  '   cargo test -p prism-core --release        # 单 crate 冒烟',
  '   cargo xtest                               # 全仓：= cargo test --workspace --release',
  '   node tools/compliance_check.mjs           # 合规自检（必须 exit 0）',
  '   node tools/compliance_check.mjs --self-test  # 反向自证：必须判红',
  '',
  '【不需要安装】npm install / pip / ffmpeg / OpenEXR —— 本项目零第三方代码依赖。',
  '   字体资产（OFL 1.1）已随包分发，见 Compliance/assets/资产登记.md。',
  '',
  'UNPACK SCRIPT BEGIN',
  UNPACK_SCRIPT,
  'UNPACK SCRIPT END',
  '',
  '以下为合并内容，每个文件以 === FILE: 路径 === 分隔。',
  '',
].join('\n');

const chunks = [header];
let binCount = 0;
let noEolCount = 0;
for (const f of files) {
  const raw = fs.readFileSync(path.join(REPO, f));
  const binary = isBinary(raw);
  if (binary) binCount++;
  const dest = remap(f);
  chunks.push(`${SEP}\n=== FILE: ${dest} ===\n`);
  if (binary) {
    //二进制：整块base64 写成一行，行数恒为 1。
    chunks.push('=== BASE64 ===\n');
    chunks.push(`=== LINES: 1 ===\n`);
    chunks.push(raw.toString('base64'));
    chunks.push('\n');
  } else {
    // 内容改写仅作用于包内（如根 Cargo.toml 的 members 指向 src/*）
    const body = rewriteContent(f, raw.toString('utf-8'));
    // 末尾无换行的文件打 NOEOL 标记，还原时才能精确复原。
    const hasEol = body.endsWith('\n');
    if (!hasEol) { noEolCount++; chunks.push('=== NOEOL ===\n'); }
    // 声明正文行数：解包侧据此精确截取，不靠"长得像分隔符"猜边界。
    const content = hasEol ? body.slice(0, -1) : body;
    const nLines = content === '' ? 0 : content.split('\n').length;
    chunks.push(`=== LINES: ${nLines} ===\n`);
    chunks.push(content);
    chunks.push('\n');
  }
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, chunks.join(''), 'utf-8');

const outSize = fs.statSync(OUT).size;
console.log('打包完成:', path.relative(REPO, OUT));
console.log(`文件数: ${files.length}（二进制 ${binCount} 个走 base64；无尾换行 ${noEolCount} 个）`);
console.log('原始总字节:', totalBytes);
console.log('TXT 大小:', outSize, `(${(outSize / 1048576).toFixed(2)} MiB)`);
