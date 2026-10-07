#!/usr/bin/env node
// ===========================================================================
// matrix.mjs — DWA 互通矩阵逐格跑批（三方向 × 72 格）
//
// 矩阵定义（2 压缩 × 2 像素类型 × 2 布局 × 3 通道数 × 3 图案 = 72 格）：
//   压缩   dwaa | dwab
//   像素   half | float
//   布局   scan | tile
//   通道   1 | 3 | 4
//   图案   gradient | noise | const
//
// 三个方向：
//   A 官方写 → 我们读   判据 =逐格bit 比对（官方 dump vs 我们 dump）
//   B 我们写 → 官方读   判据 = 官方不抛异常 + 与源数据的误差在同一量级
//   C 两份文件都用官方解码器比差（官方解我们的 vs 官方解它自己的）
//
// 输出 JSON 供 interop_gate.mjs 消费。
// ===========================================================================

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, existsSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const COMPRESSIONS = ["dwaa", "dwab"];
const PIXEL_TYPES = ["half", "float"];
const LAYOUTS = ["scan", "tile"];
const CHANNELS = [1, 3, 4];
const PATTERNS = ["gradient", "noise", "const"];

const WIDTH = 64;
const HEIGHT = 48;
const SEED = 20251006;

export function allSpecs() {
  const out = [];
  for (const c of COMPRESSIONS)
    for (const p of PIXEL_TYPES)
      for (const l of LAYOUTS)
        for (const n of CHANNELS)
          for (const pat of PATTERNS)
            out.push({ comp: c, ptype: p, layout: l, nchan: n, pattern: pat });
  return out;
}

export function specName(s) {
  return `${s.comp}-${s.ptype}-${s.layout}-${s.nchan}-${s.pattern}`;
}

/**
 * 未经压缩的像素载荷字节数（half=2B/样本，float=4B/样本）。
 * 供方向 D（压缩有效性断言）计算压缩比。
 */
export function rawPayloadBytes(s) {
  const bpp = s.ptype === "half" ? 2 : 4;
  return WIDTH * HEIGHT * s.nchan * bpp;
}

/** 文件字节数；不存在返回 -1。 */
function fileSize(p) {
  try {
    return existsSync(p) ? statSync(p).size : -1;
  } catch {
    return -1;
  }
}

/**
 * 跑一格的全部方向。
 * @param {{oexr:string, rs:string}} tools 官方 harness 与 Rust harness 路径
 * @param {string} workDir 产物目录
 * @param {object} s 矩阵格
 * @returns {object} 该格的实测结果
 */
export function runCell(tools, workDir, s) {
  const name = specName(s);
  const rec = { spec: name, ...s, A: null, B: null, C: null, D: null, notes: [] };

  const run = (cmd, args) =>
    execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });

  const tryRun = (cmd, args) => {
    try {
      return { ok: true, out: run(cmd, args) };
    } catch (e) {
      const msg = (e.stdout || "") + (e.stderr || "");
      return { ok: false, out: msg, err: String(e.message).slice(0, 400) };
    }
  };

  // ---- 源数据（两侧共享，消除「源数据不同」引入的假绿/假红）----
  const srcTxt = join(workDir, `src_${name}.txt`);
  const srcGen = tryRun(tools.oexr, ["src", srcTxt, String(WIDTH), String(HEIGHT), name, String(SEED)]);
  if (!srcGen.ok) {
    rec.notes.push(`源数据生成失败: ${srcGen.err}`);
    return rec;
  }

  // ================= 方向 A：官方写 → 我们读 =================
  {
    const exr = join(workDir, `a_${name}.exr`);
    const oDump = join(workDir, `a_${name}.official.txt`);
    const rDump = join(workDir, `a_${name}.ours.txt`);
    const g = tryRun(tools.oexr, ["gen", exr, String(WIDTH), String(HEIGHT), name, String(SEED)]);
    if (!g.ok) {
      rec.A = { ok: false, stage: "official-gen", detail: g.err };
    } else {
      const od = tryRun(tools.oexr, ["dump", exr, oDump]);
      const rd = tryRun(tools.rs, ["dump", exr, rDump]);
      if (!od.ok || !rd.ok) {
        rec.A = {
          ok: false,
          stage: !od.ok ? "official-dump" : "ours-dump",
          detail: (!od.ok ? od.err : "") + (!rd.ok ? rd.err : ""),
        };
      } else {
        const c = tryRun(tools.rs, ["cmpdump", rDump, oDump]);
        rec.A = c.ok
          ? { ok: true, ...parseCmp(c.out) }
          : { ok: false, stage: "cmpdump", detail: (c.out || "") + c.err };
      }
    }
  }

  // ================= 方向 B / C：我们写 → 官方读 =================
  // `exr_write::write_tiled_image` 已实装 flat tiled 写出（十种压缩、
  // ONE_LEVEL、边缘 tile 按裁剪尺寸），故 tiled 与 scanline **走同一条真跑路径**，
  // 不再有 B/C 的显式 SKIP。Rust harness 的 `gen` 按 spec 的 layout 字段切换入口，
  // tile 尺寸与 C++ 侧一致（32×32），保证两侧块边界相同。
  const ourExr = join(workDir, `b_${name}.exr`);
  {
    const g = tryRun(tools.rs, ["gen", ourExr, String(WIDTH), String(HEIGHT), name, srcTxt]);
    if (!g.ok) {
      rec.B = { ok: false, stage: "ours-gen", detail: g.err };
      rec.C = { ok: false, stage: "ours-gen", detail: g.err };
    } else {
      // 官方读我们的文件：必须不抛异常
      const rd = tryRun(tools.oexr, ["read", ourExr]);
      rec.B = rd.ok
        ? { ok: true, read: rd.out.trim() }
        : { ok: false, stage: "official-read-ours", detail: (rd.out || "") + rd.err };

      // 方向 C：官方解码器解两份文件（我们的 / 官方的），比差
      const ourDecodedVsSrc = tryRun(tools.oexr, ["srccmp", ourExr, srcTxt]);
      const offExr = join(workDir, `c_${name}.official.exr`);
      const offDecodedVsSrc = tryRun(tools.oexr, ["gen", offExr, String(WIDTH), String(HEIGHT), name, String(SEED)]);
      if (!ourDecodedVsSrc.ok || !offDecodedVsSrc.ok) {
        rec.C = { ok: false, stage: "srccmp", detail: ourDecodedVsSrc.err || offDecodedVsSrc.err };
      } else {
        const ours = parseCmp(ourDecodedVsSrc.out);
        const off = parseCmp(
          tryRun(tools.oexr, ["srccmp", offExr, srcTxt]).out || ""
        );
        rec.C = {
          ok: true,
          ours_vs_src: ours,
          official_vs_src: off,
          // 判据见 interop_gate.mjs：我们的有损幅度不得比官方自己的劣化 10 倍以上。
          ratio_abs: off.max_abs > 0 ? ours.max_abs / off.max_abs : null,
        };
      }
    }
  }

  // ================= 方向 D：压缩有效性（见 interop_gate.mjs 的 P1 说明） =================
  // 官方写出器的块级「压不小就原样存」回退（ImfOutputFile.cpp:584
  // `if (compSize < _lineBuffer->dataSize)`）**只由官方写出的文件触发**，
  // 全仓无 .exr fixture ⇒ cargo 测试天然覆盖不到。若该回退被改坏
  // （`if false &&`），压缩数据将永不被采用，文件退化成未压缩大小 ——
  // 而**像素值仍然全对**（无损路径），方向 A 的逐位比对抓不到它。
  // 唯一可观测的差别就是**文件体积**，故方向 D 专门断言压缩有效性。
  //
  // 【只对「可压缩图案」断言 —— 这是判据正确性的关键】
  // 高频 **noise** 图案本质不可压缩，实测比值 0.36~1.05（比值≈1 恰恰是那块级
  // 回退**正常生效**的表现：压不小就原样存）。若对 noise 也断言「必须显著
  // 压缩」，就是把**正确行为**判成红 —— 那是假红，不是门禁。
  // 因此方向 D 只覆盖 const / gradient：
  //   实测 const≤ 0.094、gradient ≤ 0.328（36 格实测，见 THRESHOLDS.d_max_ratio）
  //   noise 明确排除，并记skipped 理由，绝不静默。
  {
    const raw = rawPayloadBytes(s);
    const COMPRESSIBLE = new Set(["const", "gradient"]);
    if (s.layout === "tile") {
      // 【tiled 实装的诚实边界】tiled 的官方文件**已存在**（方向 A 就写出来了），
      // 但方向 D 的阈值 0.5 是**按 scanline 的实测分布标定**的
      // （const ≤ 0.094 / gradient ≤ 0.328）。tiled 的实测分布不同：
      // const ≤ 0.154、**gradient ≤ 0.531**（`dwab-half-tile-1-gradient`），
      // 已**越过** 0.5。把 scanline 的 0.5 直接套到 tiled 会让一个**合法**
      // 行为判红；为让 tiled 转绿而把 0.5 放宽到 0.6+ 则是**为转绿放水**
      // ——本项目栽过两次的假绿正是这么来的。
      // 故tiled 的 D 保持显式 SKIP，理由写明「阈值未按 tiled 分布重新标定」
      // （**不是**「本仓未实装tiled 写出」——那已实装，见上）。
      rec.D = { ok: false, skipped: true, reason: "方向 D 的 0.5 阈值按 scanline 分布标定；tiled 实测 gradient ≤ 0.531 已越线，需独立标定 tiled 阈值（不靠放宽 0.5 转绿）" };
    } else if (!COMPRESSIBLE.has(s.pattern)) {
      rec.D = {
        ok: false,
        skipped: true,
        reason: `图案 ${s.pattern} 本质不可压缩（实测比值 0.36~1.05，比值≈1 是块级回退正常生效的表现），断言其压缩率属假红`,
      };
    } else {
      const target = join(workDir, `a_${name}.exr`);
      const sz = fileSize(target);
      rec.D =
        sz < 0
          ? { ok: false, stage: "size", detail: `官方文件不存在: ${target}` }
          : { ok: true, file_bytes: sz, raw_bytes: raw, ratio: sz / raw };
    }
  }

  return rec;
}

/** 解析 `CMPDUMP_OK/SRCCMP_OK` 文本块。 */
export function parseCmp(text) {
  const out = {};
  for (const line of (text || "").split("\n")) {
    const t = line.trim().split(/\s+/);
    if (t.length < 2) continue;
    const k = t[0];
    if (["MAX_ABS", "MAX_REL"].includes(k)) out[k.toLowerCase()] = parseFloat(t[1]);
    else if (k === "BITDIFF" || k === "N") out[k.toLowerCase()] = parseInt(t[1], 10);
    else if (k === "MAX_AT") out.max_at = [parseInt(t[1], 10), parseInt(t[2], 10)];
  }
  return out;
}

// ---------------------------------------------------------------------------
// CLI：node matrix.mjs --oexr <path> --rs <path> [--json out.json] [--only <substr>]
// ---------------------------------------------------------------------------
if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const arg = (k, d) => {
    const i = argv.indexOf(k);
    return i >= 0 ? argv[i + 1] : d;
  };
  const oexr = arg("--oexr");
  const rs = arg("--rs");
  const only = arg("--only", null);
  if (!oexr || !rs) {
    console.error("用法: node matrix.mjs --oexr <harness> --rs <rust_harness> [--json out.json] [--only substr]");
    process.exit(2);
  }
  const workDir = mkdtempSync(join(tmpdir(), "dwa-matrix-"));
  const specs = allSpecs().filter((s) => !only || specName(s).includes(only));
  const results = [];
  for (const s of specs) {
    const r = runCell({ oexr, rs }, workDir, s);
    results.push(r);
    const tag = `${r.A?.ok ? "A✓" : r.A?.skipped ? "A-" : "A✗"} ${r.B?.ok ? "B✓" : r.B?.skipped ? "B-" : "B✗"} ${r.C?.ok ? "C✓" : r.C?.skipped ? "C-" : "C✗"}`;
    console.log(`${tag}  ${r.spec}`);
  }
  const jsonPath = arg("--json", null);
  if (jsonPath) {
    const { writeFileSync } = await import("node:fs");
    writeFileSync(jsonPath, JSON.stringify({ workDir, results }, null, 2));
    console.log(`\nJSON -> ${jsonPath}`);
  }
  console.log(`\n共 ${results.length} 格，产物目录 ${workDir}`);
}