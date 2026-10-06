// ===========================================================================
// ci_wgsl_smoke.mjs — CI 用的 WGSL **基础解析**冒烟（零第三方依赖）。
//
// 范围严格限定在"能被基础解析"：括号/花括号配对、块注释闭合、字符串字面量
// 闭合、文件非空。**不做**深度校验（绑定组布局、符号解析、禁 libm、类型推导
// 等）——那是 wgsl_lint.mjs 的职责，本脚本不重复实现，避免两个门禁给出互相
// 矛盾的结论。
//
// 特别说明：**不要求每个 .wgsl 都有 @compute/@vertex/@fragment 入口**。
// 本目录里 rt_common / rt_scene / rt_shade / rt_bvh / sim_common 是"拼接包含"
// 的公共件（由引擎自研 WGSL 子集汇编器展开 `//!include` 后才成为完整模块），
// 单独看本来就没有 stage 属性——那是设计如此，不是缺陷。是否"能独立编译"
// 属于深度校验范畴，交给 wgsl_lint.mjs 与真机 createShaderModule。
//
// 设计取向：**枚举目录里实际存在的 .wgsl**，不在脚本里写死"9 个"。
// 数量只打印、不作为失败条件；真正失败的条件是"一个都没找到"或"任一份解析不过"。
// 这样"新增一份 shader"不会让 CI 无端变红，而"误删全部 shader"一定会红。
//
// 用法：
//   node ci_wgsl_smoke.mjs [shaders_dir]
//
// 退出码：0 = 全部通过；1 = 有 shader 解析失败；2 = 目录不存在/无 wgsl 文件。
// ===========================================================================

import { readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_DIR = resolve(HERE, "../shaders");
const SHADER_DIR = process.argv[2] ? resolve(process.argv[2]) : DEFAULT_DIR;

// ---------------------------------------------------------------------------
// 基础解析：逐字符扫描，验证注释与字面量正确闭合、括号配对。
// 不用正则（正则表达不了"注释里的 /*"这种），也不用第三方 WGSL parser。
// ---------------------------------------------------------------------------

/**
 * 扫描一份 WGSL 源码，返回发现的结构问题列表。
 * @param {string} src 源码文本
 * @returns {string[]} 问题描述（空数组 = 基础解析通过）
 */
export function basicParse(src) {
  const issues = [];
  const stack = []; // 括号栈：元素为 { ch, line }
  const pairs = { ")": "(", "]": "[", "}": "{" };
  let line = 1;
  let i = 0;

  while (i < src.length) {
    const c = src[i];
    const n = src[i + 1];

    if (c === "\n") { line += 1; i += 1; continue; }

    // 块注释：必须闭合，WGSL 不支持嵌套块注释
    if (c === "/" && n === "*") {
      const end = src.indexOf("*/", i + 2);
      if (end === -1) {
        issues.push(`第 ${line} 行起存在未闭合的块注释 /* ... */`);
        break;
      }
      line += (src.slice(i, end).match(/\n/g) || []).length;
      i = end + 2;
      continue;
    }

    // 行注释
    if (c === "/" && n === "/") {
      const nl = src.indexOf("\n", i);
      i = nl === -1 ? src.length : nl;
      continue;
    }

    // 字符串字面量（WGSL 只有双引号一种）
    if (c === '"') {
      const end = src.indexOf('"', i + 1);
      if (end === -1) {
        issues.push(`第 ${line} 行的字符串字面量未闭合`);
        break;
      }
      line += (src.slice(i, end).match(/\n/g) || []).length;
      i = end + 1;
      continue;
    }

    if (c === "(" || c === "[" || c === "{") {
      stack.push({ ch: c, line });
      i += 1;
      continue;
    }
    if (c === ")" || c === "]" || c === "}") {
      const top = stack.pop();
      if (top === undefined) {
        issues.push(`第 ${line} 行出现多余的 '${c}'`);
      } else if (top.ch !== pairs[c]) {
        issues.push(`第 ${line} 行 '${c}' 与第 ${top.line} 行 '${top.ch}' 不匹配`);
      }
      i += 1;
      continue;
    }

    i += 1;
  }

  for (const open of stack) {
    issues.push(`第 ${open.line} 行的 '${open.ch}' 未闭合`);
  }
  return issues;
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
let entries;
try {
  entries = readdirSync(SHADER_DIR).filter((f) => f.endsWith(".wgsl")).sort();
} catch (e) {
  console.error(`!! 读不到 shader 目录 ${SHADER_DIR}：${e.message}`);
  process.exit(2);
}

if (entries.length === 0) {
  console.error(`!! ${SHADER_DIR} 下没有任何 .wgsl 文件（着色器被误删或路径变了？）`);
  process.exit(2);
}

console.log(`shader 目录：${SHADER_DIR}`);
console.log(`发现 ${entries.length} 份 .wgsl（数量仅记录，不作失败条件）\n`);

let failed = 0;
let standalone = 0;
let partial = 0;

for (const name of entries) {
  const path = join(SHADER_DIR, name);
  const src = readFileSync(path, "utf8");
  const issues = basicParse(src);

  if (src.trim().length === 0) {
    issues.push("文件为空");
  }

  // 含 stage 属性的 = 可独立编译的测试项；否则 = include 公共件（设计如此）。
  const hasEntry = /@(compute|vertex|fragment)\b/.test(src);
  const kind = hasEntry ? "独立入口" : "include 公共件";
  if (hasEntry) standalone += 1; else partial += 1;

  const st = statSync(path);
  const loc = st.size;
  if (issues.length === 0) {
    console.log(`  ok    ${name.padEnd(20)} ${String(loc).padStart(6)} 字节  ${kind}`);
  } else {
    failed += 1;
    console.error(`  FAIL  ${name}  ${kind}`);
    for (const it of issues) console.error(`          - ${it}`);
  }
}

console.log("");
if (failed > 0) {
  console.error(`==> ci_wgsl_smoke: FAIL（${failed}/${entries.length} 份未通过基础解析）`);
  console.error("   注意：本脚本只做基础解析。绑定组/符号/类型层面的问题请跑 wgsl_lint.mjs。");
  process.exit(1);
}
console.log(
  `==> ci_wgsl_smoke: PASS（${entries.length} 份全部通过基础解析：` +
  `${standalone} 份独立入口 + ${partial} 份 include 公共件）`,
);
process.exit(0);
