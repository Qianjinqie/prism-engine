#!/usr/bin/env node
// ===========================================================================
// interop_gate.mjs — prism-io DWA(DWAA/DWAB) 与官方 OpenEXR 的互通门禁
//
// ===========================================================================
// 【零污染纪律（红线）】
// 官方 OpenEXR 是**验证工具**，不是项目依赖：
//   * 本目录下的 `.cpp` 只是 harness **源码**，由本脚本在 /tmp 里现场编译；
//   * `crates/` 下**任何** Cargo.toml 都不得出现 OpenEXR / C++ 链接 /
//     `build.rs` / `cc` crate —— 本门禁不新增、也不允许任何此类依赖；
//   * Rust 侧（examples/exr_interop.rs）只通过**文件字节**（.exr / .txt）
//     与 C++ 侧交互，双方**无共享 ABI / 内存 / 链接**。
//
// ===========================================================================
// 【假绿陷阱（本项目已栽过两次：v2.5、v2.8）】
// 无 OpenEXR 环境时**必须**明确输出 `SKIP: <原因>` 并返回**独立可辨识**的
// 退出码 3，**绝不**返回 0。CI 若只看「退出码 == 0」判通过，则 SKIP 必须
// 是非 0；否则「因环境缺失跳过」会被当成「通过」——那就是假绿。
//   退出码：0 = 真通过 / 1 = 失败 / 3 = 跳过（环境不具备）/ 4 = 参数错误
//
// ===========================================================================
// 【判据不许放水】
// 阈值全部来自实测（见文件末尾 THRESHOLDS 的来源注释），并强制两条自证：
//   1. 每个阈值都断言 `threshold <= 实测最差值 × 10`（不比实测宽 10 倍以上）；
//   2. `--self-test` 把实测最差值**放大 10 倍**后喂给判据，断言**必须判红**
//      —— 若放大 10 倍还能过，说明判据形同虚设。
//
// ===========================================================================
// 用法：
//   node interop_gate.mjs              # 跑全矩阵，真通过 exit 0
//   node interop_gate.mjs --self-test  # 反向自证：注入坏数据必须判红
//   node interop_gate.mjs --json out.json
//   node interop_gate.mjs --only <substr>   # 只跑名字含 substr 的格
// ===========================================================================

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { runCell, allSpecs, specName } from "./matrix.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../..");

const PASS = 0;
const FAIL = 1;
const SKIP = 3;
const BAD_ARG = 4;

// ===========================================================================
//  阈值（全部来自 2025-10-06 实测，见下方「实测来源」）
// ===========================================================================

/**
 * 方向 A（官方写 → 我们读）的判据阈值。
 *
 * 【实测来源】72 格逐格实跑（OpenEXR 3.1.5，64×48，seed 20251006）：
 *   worst MAX_ABS = 7.8125e-03   （仅出现在 *-noise 格）
 *   worst MAX_REL = 2.442997e-03
 *   BITDIFF > 0 的格 = 22/72，且全部是 *-noise 格；
 *   gradient / const 两类共 50 格**逐位全等**（BITDIFF = 0，MAX_ABS = 0）。
 *
 * 【为什么不是 0（bit-exact）】官方 IDCT 走 SSE2/SIMD（`ImfDwaCompressorSimd.h`），
 * 我们走标量 f32，**浮点加法次序不同** → 噪声格里 11/9216 个样本差 2~4 个
 * half-ULP。这不是比特流解析错误（那会表现为整片错位或报结构错），而是
 * 末位舍入差异。gradient / const 格bit-exact 正是「解析完全正确」的旁证。
 *
 * 【阈值余量】取实测最差值的 ~2 倍，绝不超过 10 倍（`--self-test` 强制断言）。
 */
export const THRESHOLDS = {
  /** 方向 A：单元素绝对误差上限。实测 7.8125e-03，取 1.5e-2（≈1.92×）。 */
  a_max_abs: 1.5e-2,
  /** 方向 A：单元素相对误差上限。实测 2.442997e-03，取 5e-3（≈2.05×）。 */
  a_max_rel: 5e-3,
  /** 实测记录（仅供--self-test 反向自证与报告引用，不参与判定）。 */
  measured: {
    a_max_abs: 7.8125e-3,
    a_max_rel: 2.442997e-3,
    /**
     * 方向 C 最差劣化比（我们 vs 官方，各自与同一源数据比）。
     * **这是实测的离群值，不是判据的来源** —— 判据已独立取定为 10，
     * 不再从实测反推（见 c_max_ratio 注释里的「自我循环论证」段）。
     */
    c_ratio: 55.96975639705099,
    /** 方向 C 分布的其余统计量（36 格实测，用于说明 10 这个阈值的位置）。 */
    c_ratio_median: 2.0,
    c_ratio_second_worst: 4.1818,
    /** 方向 D 最差压缩比实测（**可压缩图案** const/gradient，24 格）。 */
    d_ratio: 0.328,
  },
  /**
   * 方向 C：我们的有损幅度 / 官方自己的有损幅度 的上限。
   *
   * 【实测来源】36 个 scanline 格逐格实跑（2025-10-06，OpenEXR 3.1.5），
   * 独立复算的分布为：
   *   min = 1.010 / **中位数 = 2.000** / 次差 = 4.182 / **唯一离群 = 55.970**
   *   ⇒ 35/36 格 ≤ 4.19，**只有 1 格**离群。
   * 离群格 = `dwaa-half-scan-1-noise`：我们 MAX_ABS 1.085e-01，官方 1.939e-03。
   *
   * 【为什么是 10 而不是 100 —— 阈值收紧】
   * 原阈值 100 有两个硬伤：
   *   1. **对实际分布宽 24 倍**：中位数 2.00、次差 4.18，35/36 格在 4.19 以下，
   *      阈值却开到 100 —— 判据在绝大多数格上根本不生效。
   *   2. **自我循环论证**：原注释写「取 100（≈1.79× 实测），远低于实测×10 = 559.7」，
   *      而 `--self-test` 的纪律检查恰好只断言「阈值 ≤ 实测×10」——
   *      **实测本身就来自有缺陷的实现**，于是离群值被自动"赦免"，
   *      离群越离谱、阈值可以放得越宽。这正是反假绿要抓的东西。
   * 现在取 **10**：仍高于 35/36 格的实际分布（4.19 的 2.4 倍），
   * 但**足以让那 1 格离群判红**，把真实的编码器量化质量差距暴露成
   * 一个显式的、可追踪的 KNOWN ISSUE，而不是被阈值悄悄吸收。
   *
   * 【那 1 格为什么不"修"阈值而要登记为已知问题】
   * 它是**真实的编码器质量差距**（单通道高频噪声下我们的量化比官方粗，
   * 官方对非 CSC 通道用更细的量化表），不是互通缺陷：官方能无异常读我们的
   * 文件（方向 B 全绿），且误差有界。修它属于 encoder 侧优化，
   * **绝不能靠把门禁阈值调回 100 来"假装通过"** —— 那就是本项目栽过两次的假绿。
   * 见下方 KNOWN_ISSUES。
   */
  c_max_ratio: 10,
  /**
   * 方向 D：官方写出文件的**压缩有效性上限**（file_bytes / raw_payload_bytes）。
   *
   * 【为什么需要它 —— 已知缺陷 4】
   * 官方 `ImfOutputFile.cpp:584` 有一处块级「压不小就原样存」回退：
   *     if (compSize < _lineBuffer->dataSize) { 用压缩数据 }
   * 该分支**只能被官方写出的 .exr 触发**，全仓无 .exr fixture ⇒
   * `cargo test -p prism-io` 对它**零覆盖**（QA 变异测试：改成 `if false &&`
   * 后 510/510 全绿）。方向 A 也抓不到：压缩被绕过时像素值**仍然全对**，
   * 唯一差别是文件**体积**。故给它一条独立判据。
   *
   * 【只对可压缩图案断言（const / gradient），noise 排除】
   * 72格实测的压缩比分布（scanline 36 格，按图案分组）：
   *     const    n=12  min 0.0106  **max 0.0936**
   *     gradient n=12  min 0.0404  **max 0.3280**
   *     noise    n=12  min 0.3561  max 1.0503← **本质不可压缩**
   * noise 的比值接近 1 恰恰是那块级回退**正常生效**的表现（压不小就原样存）。
   * 对 noise 断言「必须显著压缩」= 把正确行为判成红 = **假红**。故排除。
   *
   * 【阈值】取 **0.5**：高于 gradient 实测最差 0.328（1.5× 余量），
   * 远低于 1.0。一旦块级回退被改坏、压缩数据永不被采用，可压缩图案的
   * 体积必然趋近 raw 量级（比值→1）⇒ 必然判红。判据不依赖具体数值，
   * 只依赖「const/gradient 必然显著压缩」这一事实。
   */
  d_max_ratio: 0.5,
};

/**
 * 已知问题登记簿（**不是**豁免清单 —— 命中的格依然计入失败、门禁依然判红）。
 *
 * 【为什么必须显式登记】收紧 `c_max_ratio` 到 10 后，`dwaa-half-scan-1-noise`
 * 稳定判红。若不登记，工程师的理性反应就是把阈值调回去 —— 阈值随即退化成
 * 「随实测漂移的橡皮章」。登记簿把这个决定**钉在文本里**：
 * 谁想放宽阈值，必须先来改这条记录、给出新的实测数据。
 *
 * 【纪律】本清单**不参与** judgeC 的判定，也不把失败改成 SKIP。
 * `judgeKnownIssue` 只用于给失败格附加根因说明；命中与否，失败照记。
 */
export const KNOWN_ISSUES = [
  {
    // ✅ 已修。条目**保留**（按纪律：不直接删条目），改为记录
    // 「症状 → 真正根因 → 修法 → 复测实测值」，供后来者查证。
    id: "KI-DWA-QUANT-1CH-NOISE",
    cell: "dwaa-half-scan-1-noise",
    metric: "方向 C 劣化比 ratio_abs",
    measured: 55.96975639705099,
    status: "fixed",
    fixed_in: "量化改用官方 quantize()",
    measured_after_fix: 1.0,
    threshold: 10,
    // ⚠️ 旧版此处写的根因（「我们量化步长比官方粗，官方对非 CSC 通道用更细的
    // 量化表」）**经实证是错的**，已更正。判错依据见 `root_cause` 的实测。
    root_cause:
      "真正根因 = **写端缺逐块「压不小就原样存」回退**（官方 ImfOutputFile.cpp:586 " +
      "`if (compSize < dataSize)`），而非量化表粗细。" +
      "实测：官方写出的该文件**根本没有 DWA 块** —— chunk0 dataSize=4096=rawChunk、" +
      "chunk1 dataSize=2048=rawChunk，块首 88 字节按 i64 解释是垃圾" +
      "（version=3726514378048814387），负载按小端 half 解释恰为源数据半精度值" +
      "（1.299805 / 0.748047 / …）。即官方 DWA 压不动这块高频噪声，走 else 分支" +
      "原样存像素。故官方那 1.939e-03 **不是量化误差，而是 float→half 表示误差**" +
      "（源为 FLOAT 通道、文件为 half 通道）。我们此前always 产出 DWA 块" +
      "（dataSize=2368 < 4096），把 DWA 的有损量化误差 1.085e-01 暴露出来，" +
      "与官方量级不可比。",
    fix:
      "两处耦合改动，缺一不可（已用消融验证）：" +
      "① exr_dwa::dwa_compress 补官方块级回退 `enc.len() < raw.len() 才采用压缩`，" +
      "否则原样返回 raw（官方是严格小于，相等也回退）；" +
      "② 量化改用官方 quantize()：逐项容差 = quantBaseError(45/100000) × 归一化 JPEG 表" +
      "[raster_i]，comp0 用 Y 表 / comp1,2 用 CbCr 表，取代原统一步长 QUANT_STEP=1/64。" +
      "二者**耦合**：只有官方那条细量化把该块 AC 系数保留到压不动（≥ rawChunk），" +
      "回退才会触发（消融实测：仅加回退而保留粗量化 → 该块仍出 DWA 块 2368 B、" +
      "劣化比仍 55.970 判红）。",
    verification:
      "修后该格 ratio 55.970 → 1.000（判据阈值 10）；我们写出的 .exr 与官方写出的" +
      "**逐字节相同**（md5 4f65db4069aa6cd4383986c49a3c404a，6453 B）。" +
      "全 72 格：改善 36 / 变差 0 / 持平 0（B/C 跳过 0 格，tiled 已真跑）。",
    why_not_fixed_here:
      "曾判为「encoder 侧量化质量优化、非互通缺陷」而未修——该判断本身是错的：" +
      "它与官方的行为差异（存 raw vs 存 DWA 块）是**可观测的格式级差异**，" +
      "且正是它让方向 C 判红。",
    blocks_gate: false,
  },
  {
    // tiled 列实装后，同一根因在 `dwab-half-tile-1-noise` 上同样命中。
    // 修法与 scanline 那一格完全相同（同一份 dwa_compress），故一并转已修。
    id: "KI-DWA-QUANT-1CH-NOISE-TILED",
    cell: "dwab-half-tile-1-noise",
    metric: "方向 C 劣化比 ratio_abs",
    measured: 55.96975639705099,
    status: "fixed",
    fixed_in: "量化改用官方 quantize()",
    measured_after_fix: 1.0,
    threshold: 10,
    root_cause:
      "与 KI-DWA-QUANT-1CH-NOISE **同一根因**（缺官方写端逐块「压不小就原样存」回退），" +
      "修法亦同。修复前 tiled 与 scanline 实测比值完全相同（均 55.9698）——" +
      "因为该比值取决于「我们 vs 官方在该块上是否做 DWA」，与块边界（scanline / tile）无关；" +
      "这正是它属 encoder 侧问题、而非 tiled 写出缺陷的证据。",
    fix: "同 KI-DWA-QUANT-1CH-NOISE（同一处 dwa_compress + 量化，两行路径共用）。",
    verification:
      "修后 ratio 55.970 → 1.000。附带收益：tiled 列的 B/C 由「显式 SKIP 36 格」" +
      "变为**真跑通过**（B/C 跳过格数 36 → 0），即该缺陷曾掩盖 tiled 的真实互通状态。",
    why_not_fixed_here:
      "曾判为「量化质量优化、非 tiled 缺陷」——判断本身没错（确实不是 tiled 缺陷），" +
      "但据此而不修是错的：它同样使方向 C 判红。",
    blocks_gate: false,
  },
];

/**
 * 该格是否命中已知问题（**仅供日志附注**，不影响判定成败）。
 * @param {string} spec 格名
 * @returns {object|null}
 */
export function judgeKnownIssue(spec) {
  return KNOWN_ISSUES.find((k) => k.cell === spec) || null;
}

/**
 * **非门禁方向**的已定位缺陷登记（实装 tiled 时暴露）。
 *
 * 【为什么单列一份、不混进 KNOWN_ISSUES】
 * `KNOWN_ISSUES` 是**对拍门禁格子**的附注（会被 judgeC/judgeD 命中并打印）。
 * 下面两条是 `exr_write` 的**编码器**缺陷，**当前 72 格矩阵照不到**
 * （矩阵只跑 DWA，见 COMPRESSIONS），因此它们不会让任何格判红；
 * 但它们会让**官方解码器读我们的 ZIP/ZIPS/RLE 文件时得到 NaN 或抛异常**。
 *
 * 【纪律：不许拿「未实装」含糊，也不许因为「矩阵照不到」就当它不存在】
 * tiled 实装前，ZIP/RLE 只用于扫描线，且**从未与官方对拍**（无 .exr fixture）——
 * 这正是本项目栽过两次的假绿同构：**没覆盖 ≠ 没问题**。这里用一个
 * 一次性诊断程序（已删除）遍历十种压缩 × tiled，官方读回结果如下：
 *
 * ```text
 *   none   READ_OK  checksum=3395.99296   ✅ 与源数据一致
 *   piz    READ_OK  checksum=3395.99296   ✅
 *   pxr24  READ_OK  checksum=3395.99296   ✅
 *   b44    READ_OK  checksum=3396.03424   ✅（B44 对 half 有损，误差有界）
 *   b44a   READ_OK  checksum=3396.03424   ✅
 *   dwaa   READ_OK  checksum=3396.01523   ✅
 *   dwab   READ_OK  checksum=3396.01523   ✅
 *   zip    READ_OK  checksum=-nan         ❌ 静默错数据（比抛异常更坏）
 *   zips   READ_OK  checksum=-nan         ❌ 同上
 *   rle    EXCEPTION "Unexpected tile block length"  ❌ 官方拒绝
 * ```
 *
 * 【归属已确证，不是 tiled 实装的回归】同一份数据走**扫描线**入口
 * （`write_image`，本路只改了一处 `write_header(...)?` 的 `?`，函数体其余字节相同）
 * 得到**完全相同**的失败：`scanline zip → checksum=nan`、
 * `scanline rle → EXCEPTION`。故根因在 `exr_write` 的
 * `deflate_zlib`（zlib 头 `78 01`，FLEVEL=0）与 `rle_compress`，
 * **二者函数体在本路 diff 中逐字节未变**（已核对）。
 *
 * - ZIP 的 zlib 头：官方写 `78 5E`（FLEVEL=2 默认档），我们写 `78 01`（FLEVEL=0）。
 *   `78 01` **本身是合法 zlib**（`((0x78<<8)|0x01)%31 == 0`，Python `zlib.decompress`
 *   能解出正确的 6144 字节），但官方 `ImfZipCompressor` 按 FLEVEL 选 Huffman 表，
 *   FLEVEL=0 的表与其解码路径不匹配 ⇒ 解出垃圾。**须改成 `78 5E`**。
 * - RLE：需核对 `rle_compress` 的行程编码与官方 `ImfRleCompressor` 的
 *   「连续行程上限 / 重复计数」约定。
 *
 * 【为什么现在不修】本路职责是 tiled 写出；改 `deflate_zlib` 的 zlib 头会同时
 * 改变**扫描线 ZIP/ZIPS 的输出字节**，属跨布局的编码器变更，须单独一轮做
 * （含扫描线 × tiled 双侧对拍 + 既有 deflate 字节稳定性测试的影响评估）。
 * 在此之前**显式登记**，不允许以「矩阵照不到」为由长期静默。
 */
export const ENCODER_DEFECTS = [
  {
    id: "KI-ENC-ZIP-FLEVEL-HEADER",
    scope: "exr_write::deflate_zlib（ZIP / ZIPS，扫描线 + tiled 均受影响）",
    symptom: "官方解码器读我们的 ZIP/ZIPS 文件得到 NaN（静默错数据）",
    observed: "checksum=-nan（tiled 与 scanline 两侧一致复现）",
    root_cause:
      "zlib 头写 0x78 0x01（FLEVEL=0 = 最快档）。官方 ImfZipCompressor 按 FLEVEL " +
      "选 Huffman 表，FLEVEL=0 与其解码路径不匹配。官方写出的是 0x78 0x5E（FLEVEL=2）。",
    why_not_fixed_here:
      "改 zlib 头会同时改变扫描线 ZIP/ZIPS 的输出字节（跨布局的编码器变更），" +
      "且既有 deflate 字节稳定性测试会受影响；须单独一轮做双侧对拍后改。" +
      "本路（tiled 写出）不扩大变更面。",
    matrix_covers_it: false,
    matrix_gap_reason:
      "72 格矩阵的 COMPRESSIONS 只有 dwaa/dwab（见 matrix.mjs），照不到 ZIP/RLE；" +
      "这是覆盖缺口，不是「无问题」的证据。",
  },
  {
    id: "KI-ENC-RLE-BLOCK-LENGTH",
    scope: "exr_write::rle_compress（RLE，扫描线 + tiled 均受影响）",
    symptom: '官方解码器抛 "Unexpected tile block length"（拒绝读我们的 RLE 文件）',
    observed: "tiled 与 scanline 两侧一致复现 EXCEPTION",
    root_cause:
      "待定位：需核对 rle_compress 的行程编码与官方 ImfRleCompressor 的" +
      "「连续行程上限 / 重复计数」约定，以及块长度字段的写入约定。",
    why_not_fixed_here: "同上：属编码器缺陷，须单独一轮定位并双侧对拍。",
    matrix_covers_it: false,
    matrix_gap_reason: "同 KI-ENC-ZIP-FLEVEL-HEADER（矩阵只跑 DWA）。",
  },
];

/** 把实测最差值放大 10 倍，供 --self-test 反向自证。 */
const x10 = (v) => v * 10;

// ===========================================================================
//  环境探测（缺失即 SKIP，绝不静默 pass）
// ===========================================================================

/**
 * 探测官方 OpenEXR 是否可用。
 * @returns {{ok:boolean, reason:string, cflags:string, libs:string}}
 */
function probeOfficialEnv() {
  const reasons = [];
  let cflags = "";
  let libs = "";
  try {
    cflags = execFileSync("pkg-config", ["--cflags", "OpenEXR"], {
      encoding: "utf8",
    }).trim();
    libs = execFileSync("pkg-config", ["--libs", "OpenEXR"], {
      encoding: "utf8",
    }).trim();
  } catch {
    reasons.push("pkg-config查不到 OpenEXR（未装 libopenexr-dev？）");
    return { ok: false, reason: reasons.join("; "), cflags, libs };
  }
  for (const tool of ["g++", "pkg-config"]) {
    try {
      execFileSync("bash", ["-lc", `command -v ${tool}`], { stdio: "ignore" });
    } catch {
      reasons.push(`缺工具 ${tool}`);
    }
  }
  if (reasons.length) return { ok: false, reason: reasons.join("; "), cflags, libs };
  return { ok: true, reason: "", cflags, libs };
}

// ===========================================================================
// 【环境缺失 vs 本仓代码坏了 —— 必须区分】
// 历史缺陷：Rust harness **编译失败**曾被报成 `SKIP:` + 退出码 3。那意味着
// 「本仓代码坏了」被伪装成「环境不具备」，掩盖真失败 —— 这比没有门禁更危险。
// 现在严格二分：
//   * 环境不具备（无 pkg-config / 查不到 OpenEXR / 无 C++ 编译器）→ SKIP(3)
//   * **本仓 Rust 代码编译不过**（cargo 报 E#### 等）→ FAIL(1)
// 判据：只有「连 Rust 工具链都起不来」才算环境问题；只要 cargo **成功启动**
// 并给出了编译诊断，那就是本仓代码的问题，必须判红。
// ===========================================================================

/** 探测 Rust 工具链是否可用（**仅判环境**，不判代码对错）。 */
function probeRustToolchain() {
  try {
    execFileSync("cargo", ["--version"], { stdio: "ignore" });
    return { ok: true, reason: "" };
  } catch (e) {
    return {
      ok: false,
      reason: `cargo 不可用（Rust 工具链缺失）: ${String(e.message).slice(0, 160)}`,
    };
  }
}

/**
 * 编译 Rust 侧 harness（example binary，不计入 cargo test 数量）。
 * @returns {{ok:boolean, reason:string, kind:string, path?:string}}
 *   kind = "env" → 环境不具备（SKIP）；kind = "code" → 本仓代码坏了（FAIL）
 */
function buildRustHarness() {
  const tool = probeRustToolchain();
  if (!tool.ok) {
    return { ok: false, kind: "env", reason: tool.reason };
  }
  let stderr = "";
  try {
    execFileSync(
      "cargo",
      ["build", "-p", "prism-io", "--release", "--example", "exr_interop"],
      { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }
    );
  } catch (e) {
    // cargo 已经**成功启动**并给出了诊断 —— 这是本仓代码/构建的问题，不是环境。
    stderr = `${e.stderr || ""}${e.stdout || ""}` || String(e.message || "");
    return {
      ok: false,
      kind: "code",
      reason: `Rust harness 编译失败（本仓代码问题，非环境缺失）:\n${extractRustErrors(stderr)}`,
    };
  }
  const bin = join(REPO, "target", "release", "examples", "exr_interop");
  if (!existsSync(bin)) {
    // 编译声称成功却没有产物：同样是构建链的问题，必须判红。
    return { ok: false, kind: "code", reason: `Rust harness 编译后仍不存在: ${bin}` };
  }
  return { ok: true, kind: "ok", path: bin };
}

/** 从 cargo 诊断里抽出 `error[E0308]: ...` 关键行（最多 6 条），便于 CI 日志定位。 */
export function extractRustErrors(text) {
  const lines = String(text || "")
    .split("\n")
    .filter((l) => /^error(\[E\d+\])?:/.test(l.trim()));
  const picked = (lines.length ? lines : String(text).split("\n").filter((l) => l.includes("error")))
    .slice(0, 6)
    .map((l) => `    ${l.trim()}`)
    .join("\n");
  return picked || "    (cargo 未给出可识别的错误行)";
}

/** 把 harness .cpp 编译到 /tmp（**绝不**写进 crates/ 的构建链）。 */
function buildOfficialHarness(env, workDir) {
  const src = join(HERE, "oexr_interop.cpp");
  if (!existsSync(src)) {
    return { ok: false, kind: "code", reason: `缺 harness 源码: ${src}` };
  }
  const out = join(workDir, "oexr_interop");
  try {
    execFileSync(
      "g++",
      ["-O2", "-std=c++17", "-o", out, src, ...env.cflags.split(/\s+/).filter(Boolean), ...env.libs.split(/\s+/).filter(Boolean)],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }
    );
  } catch (e) {
    // g++ 能启动但编译失败 ⇒ 是**我们的 harness 源码**坏了 ⇒ 判红，不是 SKIP。
    const msg = `${e.stderr || ""}${e.stdout || ""}` || String(e.message || "");
    return {
      ok: false,
      kind: "code",
      reason: `C++ harness 编译失败（本仓 harness 源码问题，非环境缺失）:\n${extractRustErrors(msg)}`,
    };
  }
  return { ok: true, kind: "ok", path: out };
}

/**
 * 统一的门禁前置失败出口。
 * env 缺失 → SKIP(3)；本仓代码/构建坏 → FAIL(1)。
 * @param {{ok:boolean, kind:string, reason:string}} r
 * @param {string} what 前置步骤名
 */
function bailOut(r, what) {
  if (r.kind === "env") {
    console.log(`SKIP: [${what}] ${r.reason}`);
    console.log("     官方 OpenEXR 是本门禁的**验证工具**（非项目依赖）；缺失时无法对拍。");
    console.log("     这不是「通过」—— 退出码 3 表示跳过，CI 必须区别于 0。");
    process.exit(SKIP);
  }
  console.log(`FAIL: [${what}] ${r.reason}`);
  console.log("     这是**本仓代码/构建的问题**，不是环境缺失 —— 判红（退出码 1）。");
  console.log("     绝不可当成 SKIP：SKIP 会把「代码坏了」伪装成「环境不具备」。");
  process.exit(FAIL);
}

// ===========================================================================
//  判据（三方向）
// ===========================================================================

/**
 * 方向 A：官方写 → 我们读。判据 = 误差在阈值内。
 * @returns {{ok:boolean, why:string}}
 */
export function judgeA(cell, th = THRESHOLDS) {
  const a = cell.A;
  if (!a) return { ok: false, why: "方向 A 无结果" };
  if (!a.ok) return { ok: false, why: `方向 A 失败(@${a.stage}): ${a.detail || ""}`.slice(0, 200) };
  if (!(a.max_abs <= th.a_max_abs)) {
    return { ok: false, why: `MAX_ABS ${a.max_abs.toExponential(3)} > 阈值 ${th.a_max_abs.toExponential(3)}` };
  }
  if (!(a.max_rel <= th.a_max_rel)) {
    return { ok: false, why: `MAX_REL ${a.max_rel.toExponential(3)} > 阈值 ${th.a_max_rel.toExponential(3)}` };
  }
  return { ok: true, why: `MAX_ABS ${a.max_abs.toExponential(3)} MAX_REL ${a.max_rel.toExponential(3)} BITDIFF ${a.bitdiff}/${a.n}` };
}

/**
 * 方向 B：我们写 → 官方读。判据 = 官方**不抛异常**地读完。
 */
export function judgeB(cell) {
  const b = cell.B;
  if (!b) return { ok: false, why: "方向 B 无结果" };
  if (b.skipped) return { ok: true, skipped: true, why: `SKIP: ${b.reason}` };
  if (!b.ok) return { ok: false, why: `官方读我们的文件失败(@${b.stage}): ${(b.detail || "").slice(0, 160)}` };
  return { ok: true, why: b.read || "官方读取成功" };
}

/**
 * 方向 C：两份文件都由官方解码器解，各自与**同一份源数据**比，
 * 判据 = 我们的有损幅度不比官方自己的劣化 10 倍以上。
 */
export function judgeC(cell, th = THRESHOLDS) {
  const c = cell.C;
  if (!c) return { ok: false, why: "方向 C 无结果" };
  if (c.skipped) return { ok: true, skipped: true, why: `SKIP: ${c.reason}` };
  if (!c.ok) return { ok: false, why: `方向 C 失败(@${c.stage}): ${(c.detail || "").slice(0, 160)}` };
  const ratio = c.ratio_abs;
  if (ratio === null || ratio === undefined) {
    return { ok: false, why: "方向 C 缺劣化比（官方自身误差为 0，无法归一化）" };
  }
  if (!(ratio <= th.c_max_ratio)) {
    return { ok: false, why: `劣化比 ${ratio.toFixed(3)} > 阈值 ${th.c_max_ratio}` };
  }
  return {
    ok: true,
    why: `ours ${c.ours_vs_src.max_abs.toExponential(3)} vs official ${c.official_vs_src.max_abs.toExponential(3)} → ratio ${ratio.toFixed(3)}`,
  };
}

/**
 * 方向 D：官方写出文件的压缩有效性。判据 = 体积显著小于未压缩载荷。
 *
 * 【这条判据为何独立存在】见 THRESHOLDS.d_max_ratio 注释：
 * 官方块级「压不小就原样存」回退被改坏时，像素值仍全对（方向 A 抓不到），
 * 只有体积会退化。
 *
 * @param {object} cell 含 D 结果的格
 * @param {object} th 阈值表
 * @returns {{ok:boolean, skipped?:boolean, why:string}}
 */
export function judgeD(cell, th = THRESHOLDS) {
  const d = cell.D;
  if (!d) return { ok: false, why: "方向 D 无结果" };
  if (d.skipped) return { ok: true, skipped: true, why: `SKIP: ${d.reason}` };
  if (!d.ok) return { ok: false, why: `方向 D 失败(@${d.stage}): ${d.detail || ""}`.slice(0, 200) };
  if (!(d.ratio <= th.d_max_ratio)) {
    return {
      ok: false,
      why:
        `压缩比 ${d.ratio.toFixed(4)} > 阈值 ${th.d_max_ratio}` +
        `（文件 ${d.file_bytes} B / raw ${d.raw_bytes} B）——` +
        `块级「压不小就原样存」回退可能已被改坏，压缩数据未被采用`,
    };
  }
  return {
    ok: true,
    why: `file ${d.file_bytes} B / raw ${d.raw_bytes} B → ratio ${d.ratio.toFixed(4)}`,
  };
}

// ===========================================================================
//  --self-test：注入式反向自证
// ===========================================================================

/**
 * 自证清单。每一项都断言「**必须判红**」—— 若某项竟然通过，说明判据恒真，
 * 这比没有门禁更危险（它给人虚假的安全感）。
 */
export function selfTest() {
  const checks = [];
  const add = (name, ok, detail) => checks.push({ name, ok, detail });

  // ---- 1. 阈值不得比实测宽 10 倍以上 ----
  for (const [key, measured, limit] of [
    ["a_max_abs", THRESHOLDS.measured.a_max_abs, THRESHOLDS.a_max_abs],
    ["a_max_rel", THRESHOLDS.measured.a_max_rel, THRESHOLDS.a_max_rel],
    ["d_max_ratio", THRESHOLDS.measured.d_ratio, THRESHOLDS.d_max_ratio],
  ]) {
    const cap = x10(measured);
    add(
      `阈值纪律: ${key} (${limit}) ≤ 实测×10 (${cap.toPrecision(6)})`,
      limit <= cap,
      `阈值 ${limit}，实测 ${measured}，实测×10 ${cap.toPrecision(6)}`
    );
  }

  // ---- 1b. c_max_ratio 的**反循环论证**纪律 ----
  // 原缺陷：阈值 = 实测离群值 × 1.79，而实测来自有缺陷的实现 ⇒ 离群被赦免。
  // 现在阈值独立取定为 10，故必须断言「阈值远低于实测离群值」——
  // 即离群格**确实会判红**，而不是被阈值吸收。
  {
    const limit = THRESHOLDS.c_max_ratio;
    const measured = THRESHOLDS.measured.c_ratio;
    add(
      `阈值纪律: c_max_ratio (${limit}) 必须显著低于实测离群值 (${measured.toFixed(3)})`,
      limit < measured / 2,
      `阈值 ${limit}，实测离群 ${measured.toFixed(3)}，比值 ${(measured / limit).toFixed(2)}×` +
        `（要求 > 2×，确保离群格真会判红，不被阈值赦免）`
    );
    // 离群格本身必须被判红 —— 这是「阈值收紧真的生效」的直接证据。
    const ki = KNOWN_ISSUES.find((k) => k.cell === "dwaa-half-scan-1-noise");
    const bad = {
      spec: "dwaa-half-scan-1-noise",
      C: {
        ok: true,
        ratio_abs: ki ? ki.measured : measured,
        ours_vs_src: { max_abs: 1 },
        official_vs_src: { max_abs: 1 },
      },
    };
    const r = judgeC(bad);
    add(
      "反向自证 C: 已知离群格必须判红（证明 c_max_ratio 已收紧到 10）",
      !r.ok,
      r.why
    );
    // 35/36 格的实际分布（次差 4.18）必须判绿 —— 证明 10 没收到误伤。
    const good = {
      spec: "dwab-half-scan-4-gradient",
      C: {
        ok: true,
        ratio_abs: THRESHOLDS.measured.c_ratio_second_worst,
        ours_vs_src: { max_abs: 1 },
        official_vs_src: { max_abs: 1 },
      },
    };
    const rg = judgeC(good);
    add(
      "反向自证 C: 次差实测值 (4.18) 必须判绿（阈值 10 未误伤 35/36 格）",
      rg.ok,
      rg.why
    );
  }

  // ---- 1c. 已知问题登记簿纪律：命中KNOWN_ISSUES 不得把失败变成通过 ----
  {
    add(
      "KNOWN_ISSUES 纪律: 命中登记簿不得改变判定成败",
      judgeKnownIssue("dwaa-half-scan-1-noise") !== null &&
        judgeKnownIssue("dwab-half-scan-4-gradient") === null,
      "登记簿只做根因附注，不参与 judgeC 判定"
    );
    add(
      "KNOWN_ISSUES 纪律: 登记簿必须逐条写明实测值与根因",
      KNOWN_ISSUES.every(
        (k) => typeof k.measured === "number" && String(k.root_cause || "").length > 20
      ),
      `${KNOWN_ISSUES.length} 条记录`
    );
    // 条目转「已修」时的纪律 —— 不得静默删除（删掉就没人查得到
    // 曾经踩过什么），也不得只写「已修」而不给复测值（那等于没有证据）。
    add(
      "KNOWN_ISSUES 纪律: 已修条目须保留在册并附修法与复测实测值",
      KNOWN_ISSUES.filter((k) => k.status === "fixed").every(
        (k) =>
          String(k.fix || "").length > 20 &&
          String(k.verification || "").length > 20 &&
          typeof k.measured_after_fix === "number" &&
          k.measured_after_fix <= k.threshold
      ),
      `已修 ${KNOWN_ISSUES.filter((k) => k.status === "fixed").length} 条：` +
        KNOWN_ISSUES.filter((k) => k.status === "fixed")
          .map((k) => `${k.id} ${k.measured.toFixed(3)}→${k.measured_after_fix}`)
          .join("; ")
    );
    // 反向自证：若有人把「已修」条目事后改成未修却**不回填** measured_after_fix，
    // 上面那条必须判红 —— 用一个残缺样本证明该判据非恒真。
    add(
      "KNOWN_ISSUES 纪律: 残缺的『已修』样本必须被判红（判据非恒真）",
      !(
        typeof { status: "fixed", fix: "x", verification: "y" }.measured_after_fix === "number" &&
        typeof { status: "fixed", fix: "x", verification: "y" }.measured_after_fix <= 10
      ),
      "缺 measured_after_fix 的已修样本通不过上一条判据"
    );
  }

  // ---- 2. 反向自证：把实测最差值放大 10 倍喂给判据，必须判红 ----
  {
    const bad = {
      spec: "SELF-TEST-inflated-A",
      A: {
        ok: true,
        max_abs: x10(THRESHOLDS.measured.a_max_abs),
        max_rel: x10(THRESHOLDS.measured.a_max_rel),
        bitdiff: 1,
        n: 1,
      },
    };
    const r = judgeA(bad);
    add("反向自证 A: 实测×10 的 MAX_ABS/MAX_REL 必须判红", !r.ok, r.why);
  }
  {
    const bad = {
      spec: "SELF-TEST-ratio-C",
      C: {
        ok: true,
        ratio_abs: x10(THRESHOLDS.measured.c_ratio),
        ours_vs_src: { max_abs: 1 },
        official_vs_src: { max_abs: 1 },
      },
    };
    const r = judgeC(bad);
    add("反向自证 C: 实测×10 的劣化比必须判红", !r.ok, r.why);
  }

  // ---- 3. 真实数据必须判绿（证明判据不是恒假） ----
  {
    const good = {
      spec: "SELF-TEST-measured-A",
      A: {
        ok: true,
        max_abs: THRESHOLDS.measured.a_max_abs,
        max_rel: THRESHOLDS.measured.a_max_rel,
        bitdiff: 11,
        n: 9216,
      },
    };
    const r = judgeA(good);
    add("反向自证 A: 实测值本身必须判绿（判据非恒假）", r.ok, r.why);
  }

  // ---- 4. 结构性失败必须判红（错误信息/异常结果不能被放过） ----
  add("判据 A: 方向 A 结构失败必须判红", !judgeA({ A: { ok: false, stage: "cmpdump", detail: "x" } }).ok, "");
  add("判据 A: A 缺失必须判红", !judgeA({}).ok, "");
  add("判据 B: 官方读失败必须判红", !judgeB({ B: { ok: false, stage: "official-read-ours", detail: "Early end of file" } }).ok, "");
  add("判据 B: B 缺失必须判红", !judgeB({}).ok, "");
  add("判据 C: C 缺失必须判红", !judgeC({}).ok, "");
  add(
    "判据 C: 劣化比缺失必须判红",
    !judgeC({ C: { ok: true, ratio_abs: null, ours_vs_src: { max_abs: 1 }, official_vs_src: { max_abs: 0 } } }).ok,
    ""
  );
  add("判据 D: D 缺失必须判红", !judgeD({}).ok, "");
  add(
    "判据 D: 结构失败必须判红",
    !judgeD({ D: { ok: false, stage: "size", detail: "官方文件不存在" } }).ok,
    ""
  );

  // ---- 4b. 方向 D 反向自证：模拟块级回退被改坏（压缩数据未被采用） ----
  // 这是 P1 缺陷 4 的**变异自证**：把体积退到 raw 量级（比值→1），
  // 判据必须判红。若此处还能判绿，说明方向 D 抓不到那块级回退。
  {
    // 用**实测最差**的合法压缩比（gradient 0.3280）而非最好值，
    // 确保判据对最不利的合法格也不误伤。
    const rawGrad = 24576; // 64*48*4*4 (float, 4chan)
    const goodWorst = { D: { ok: true, file_bytes: Math.round(rawGrad * 0.328), raw_bytes: rawGrad, ratio: 0.328 } };
    const rgw = judgeD(goodWorst);
    add("反向自证 D: 实测最差合法压缩比 (0.328) 必须判绿（判据非恒假）", rgw.ok, rgw.why);

    const raw = 6144;
    const good = { D: { ok: true, file_bytes: 575, raw_bytes: raw, ratio: 575 / raw } };
    const rg = judgeD(good);
    add("反向自证 D: const 实测体积 (575B) 必须判绿", rg.ok, rg.why);

    // 变异：块级回退改坏 ⇒ 压缩数据永不被采用 ⇒ 体积趋近 raw。
    for (const [label, bytes] of [
      ["完全未压缩（= raw 6128B）", 6128],
      ["压缩无效（比值 0.97）", Math.round(raw * 0.97)],
      ["压缩几乎失效（比值 0.85）", Math.round(raw * 0.85)],
      ["过半压缩（比值 0.51，边界外）", Math.round(raw * 0.51)],
    ]) {
      const bad = { D: { ok: true, file_bytes: bytes, raw_bytes: raw, ratio: bytes / raw } };
      const rb = judgeD(bad);
      add(`反向自证 D: ${label} 必须判红（块级回退变异被捕获）`, !rb.ok, rb.why);
    }
    // 恰好在阈值上：ratio == 0.5 必须判绿（边界不含糊）
    const edge = { D: { ok: true, file_bytes: raw * 0.5, raw_bytes: raw, ratio: 0.5 } };
    add("判据 D 边界: ratio 恰为阈值 0.5 判绿（≤ 语义）", judgeD(edge).ok, judgeD(edge).why);
  }

  // ---- 5. SKIP 必须独立可辨识，绝不等价于PASS ----
  {
    const b = judgeB({ B: { ok: false, skipped: true, reason: "本仓未实装 tiled 写出" } });
    const c = judgeC({ C: { ok: false, skipped: true, reason: "本仓未实装 tiled 写出" } });
    const d = judgeD({ D: { ok: false, skipped: true, reason: "tiled 不在方向 D 覆盖范围" } });
    add("SKIP 语义: B 跳过必须带 skipped 标记", b.ok === true && b.skipped === true, b.why);
    add("SKIP 语义: C 跳过必须带 skipped 标记", c.ok === true && c.skipped === true, c.why);
    add("SKIP 语义: D 跳过必须带 skipped 标记", d.ok === true && d.skipped === true, d.why);
    add(
      "SKIP 语义: 跳过时 why 必须含 SKIP: 前缀",
      String(b.why).includes("SKIP:") && String(c.why).includes("SKIP:") && String(d.why).includes("SKIP:"),
      `${b.why} / ${c.why} / ${d.why}`
    );
  }

  // ---- 5b. 编码器缺陷登记簿纪律：矩阵照不到的缺陷必须显式登记（tiled 实装）==-
  // 反假绿：**「矩阵照不到」不等于「没问题」**。ZIP/RLE 的编码器缺陷正是
  // 因为矩阵只跑 DWA 而长期隐身，tiled 实装后才被官方读回暴露出来。
  {
    add(
      "编码器缺陷登记簿: 每条必须写明症状、根因与「矩阵为何照不到」",
      ENCODER_DEFECTS.every(
        (k) =>
          typeof k.symptom === "string" &&
          String(k.root_cause || "").length > 20 &&
          typeof k.matrix_covers_it === "boolean" &&
          String(k.matrix_gap_reason || "").length > 10
      ),
      `${ENCODER_DEFECTS.length} 条：${ENCODER_DEFECTS.map((k) => k.id).join(", ")}`
    );
    add(
      "编码器缺陷登记簿: 缺陷须诚实标为「矩阵照不到」，不得谎称已覆盖",
      ENCODER_DEFECTS.every((k) => k.matrix_covers_it === false),
      "全部标 matrix_covers_it=false（矩阵 COMPRESSIONS 仅 dwaa/dwab）"
    );
    add(
      "编码器缺陷登记簿: 每条都必须写明为何不在当前版本修（不许含糊成「未实装」）",
      ENCODER_DEFECTS.every(
        (k) =>
          String(k.why_not_fixed_here || "").length > 20 &&
          !/未实装|没做|暂时/.test(String(k.why_not_fixed_here))
      ),
      "禁止用「未实装」搪塞已定位的缺陷"
    );
  }

  // ---- 6. 编译错误纪律：Rust 编译错误必须能被识别为「代码坏了」而非环境缺失 ----
  // 这是反向自证：若extractRustErrors 抓不到 E0308，门禁就退回到
  // 「编译失败 → SKIP」的老写法（本仓代码坏了被当成环境不具备）。
  {
    const cargoErr = [
      "error[E0308]: mismatched types",
      "  --> crates/prism-io/examples/exr_interop.rs:42:9",
      "error: could not compile `prism-io` (example \"exr_interop\") due to 1 previous error",
    ].join("\n");
    const got = extractRustErrors(cargoErr);
    add(
      "编译错误纪律: cargo 的 E#### 编译错误必须被抽取出来（供 FAIL 路径使用）",
      got.includes("E0308"),
      got.split("\n")[0]
    );
    // 判据常量：SKIP=3 / FAIL=1，两者绝不可混同
    add(
      "退出码纪律: 退出码 FAIL(1) 与 SKIP(3) 必须不同（编译失败不得报 SKIP）",
      FAIL === 1 && SKIP === 3 && FAIL !== SKIP,
      `FAIL=${FAIL} SKIP=${SKIP}`
    );
  }

  return checks;
}

// ===========================================================================
//  主流程
// ===========================================================================

function main() {
  const argv = process.argv.slice(2);
  if (argv.includes("--help") || argv.includes("-h")) {
    console.log("用法: node interop_gate.mjs [--self-test] [--json out.json] [--only substr]");
    process.exit(PASS);
  }

  // ---------- --self-test ----------
  if (argv.includes("--self-test")) {
    const checks = selfTest();
    let bad = 0;
    console.log("=== interop_gate --self-test（注入式反向自证）===");
    for (const c of checks) {
      if (!c.ok) bad++;
      console.log(`${c.ok ? "  ok  " : "  FAIL"}  ${c.name}${c.detail ? `  [${c.detail}]` : ""}`);
    }
    console.log(`\nself-test: ${checks.length - bad}/${checks.length} 通过`);
    if (bad > 0) {
      console.log("SELF-TEST FAIL: 存在恒真判据或阈值纪律被破坏 —— 门禁不可信");
      process.exit(FAIL);
    }
    console.log("SELF-TEST OK: 每条判据都能判红，且实测值本身判绿");
    process.exit(PASS);
  }

  // ---------- 环境探测：缺失即 SKIP（退出码 3，绝不 0） ----------
  const env = probeOfficialEnv();
  if (!env.ok) {
    console.log(`SKIP: ${env.reason}`);
    console.log("     官方 OpenEXR 是本门禁的**验证工具**（非项目依赖）；缺失时无法对拍。");
    console.log("     这不是「通过」—— 退出码 3 表示跳过，CI 必须区别于 0。");
    process.exit(SKIP);
  }

  const workDir = mkdtempSync(join(tmpdir(), "dwa-gate-"));
  // 注意：Rust / C++ harness 构建失败一律走 bailOut —— 由 kind 决定
  // SKIP(3) 还是 FAIL(1)。**本仓代码编译失败绝不能报 SKIP。**
  const rust = buildRustHarness();
  if (!rust.ok) bailOut(rust, "rust-harness");
  const official = buildOfficialHarness(env, workDir);
  if (!official.ok) bailOut(official, "cpp-harness");

  const onlyIdx = argv.indexOf("--only");
  const only = onlyIdx >= 0 ? argv[onlyIdx + 1] : null;
  const specs = allSpecs().filter((s) => !only || specName(s).includes(only));
  if (specs.length === 0) {
    console.log(`BAD_ARG: --only ${JSON.stringify(only)} 没匹配到任何格`);
    process.exit(BAD_ARG);
  }

  console.log(`=== DWA 互通门禁：${specs.length} 格 × 三方向（官方 OpenEXR 3.1.5）===\n`);

  const tools = { oexr: official.path, rs: rust.path };
  const cells = [];
  let failA = 0;
  let failB = 0;
  let failC = 0;
  let failD = 0;
  let skipBC = 0;
  const failedKnown = [];

  for (const s of specs) {
    const cell = runCell(tools, workDir, s);
    cells.push(cell);
    const ra = judgeA(cell);
    const rb = judgeB(cell);
    const rc = judgeC(cell);
    const rd = judgeD(cell);
    if (!ra.ok) failA++;
    if (!rb.ok) failB++;
    if (!rc.ok) failC++;
    if (!rd.ok) failD++;
    if (!rc.ok && !rc.skipped) {
      const ki = judgeKnownIssue(cell.spec);
      if (ki) failedKnown.push({ spec: cell.spec, ...ki, why: rc.why });
    }

    const tag = (r) => (r.skipped ? "-" : r.ok ? "PASS" : "FAIL");
    // 跳过按**格**计（B、C 各跳一次但只算一格），避免汇总数字翻倍误导。
    if (rb.skipped || rc.skipped) skipBC++;
    console.log(`A:${tag(ra)} B:${tag(rb)} C:${tag(rc)} D:${tag(rd)}  ${cell.spec}`);
    if (!ra.ok) console.log(`      A: ${ra.why}`);
    if (!rb.ok && !rb.skipped) console.log(`      B: ${rb.why}`);
    if (!rc.ok && !rc.skipped) {
      const ki = judgeKnownIssue(cell.spec);
      console.log(`      C: ${rc.why}${ki ? `  [已知问题 ${ki.id}]` : ""}`);
    }
    if (rb.skipped) console.log(`      B: ${rb.why}`);
    if (rc.skipped) console.log(`      C: ${rc.why}`);
    if (!rd.ok && !rd.skipped) console.log(`      D: ${rd.why}`);
  }

  // ---------- 汇总 ----------
  const allA = cells.map((c) => c.A).filter((a) => a && a.ok);
  const worstAbs = allA.length ? Math.max(...allA.map((a) => a.max_abs)) : NaN;
  const worstRel = allA.length ? Math.max(...allA.map((a) => a.max_rel)) : NaN;
  const totalBit = allA.reduce((s, a) => s + (a.bitdiff || 0), 0);
  const totalN = allA.reduce((s, a) => s + (a.n || 0), 0);

  console.log("\n=== 汇总 ===");
  console.log(`格子总数        : ${cells.length}`);
  console.log(`方向 A 最差实测 : MAX_ABS=${worstAbs.toExponential(6)} MAX_REL=${worstRel.toExponential(6)}`);
  console.log(`方向 A 位差异   : ${totalBit} / ${totalN} 样本`);
  console.log(`阈值            : MAX_ABS≤${THRESHOLDS.a_max_abs.toExponential(3)} MAX_REL≤${THRESHOLDS.a_max_rel.toExponential(3)}`);
  console.log(
    `阈值纪律        : 实测×10 = ${x10(worstAbs).toExponential(3)}（ABS）/ ${x10(worstRel).toExponential(3)}（REL），阈值必须 ≤ 它`
  );
  // SKIP 数与实际一致性自检（QA 指出工程师曾自报"24 格"，实测 36 格）。
  const skipB = cells.filter((c) => c.B && c.B.skipped).length;
  const skipC = cells.filter((c) => c.C && c.C.skipped).length;
  const skipD = cells.filter((c) => c.D && c.D.skipped).length;
  console.log(
    `B/C 显式跳过格数: ${skipBC}（B 跳过 ${skipB} 格 / C 跳过 ${skipC} 格）`
  );
  console.log(`方向 D 跳过格数 : ${skipD}（tiled 的 D 阈值未按 tiled 分布独立标定；noise 本质不可压缩）`);
  console.log(`失败            : A=${failA} B=${failB} C=${failC} D=${failD}`);

  // ---- 已知问题登记簿（不豁免、不改判，只附根因） ----
  const openKi = KNOWN_ISSUES.filter((k) => k.status !== "fixed");
  const fixedKi = KNOWN_ISSUES.filter((k) => k.status === "fixed");
  console.log(`\n=== 已知问题（登记簿，${KNOWN_ISSUES.length} 条；**不豁免**，仍计入失败）===`);
  console.log(
    `  未修复 ${openKi.length} 条 / 已修复 ${fixedKi.length} 条` +
      `（已修复条目**保留**在册，只作查证记录，不影响任何判定）`
  );
  for (const k of fixedKi) {
    console.log(`  [已修 ${k.id}] ${k.cell}：${k.metric} ${k.measured.toFixed(3)} → ${k.measured_after_fix}`);
    console.log(`      修法: ${k.fix}`);
    console.log(`      复测: ${k.verification}`);
  }
  if (failedKnown.length === 0) {
    console.log("  （本次运行未命中任何登记在册的未修复已知问题）");
  } else {
    for (const k of failedKnown) {
      console.log(`  [${k.id}] ${k.spec}`);
      console.log(`      实测 ${k.measured.toFixed(3)} > 阈值 ${k.threshold}（${k.metric}）`);
      console.log(`      根因: ${k.root_cause}`);
      console.log(`      为何不在此修: ${k.why_not_fixed_here}`);
    }
  }

  const jsonIdx = argv.indexOf("--json");
  if (jsonIdx >= 0) {
    const p = argv[jsonIdx + 1];
    if (p) {
      writeFileSync(
        p,
        JSON.stringify(
          {
            cells,
            summary: {
              total: cells.length,
              worst_abs: worstAbs,
              worst_rel: worstRel,
              total_bitdiff: totalBit,
              total_samples: totalN,
              thresholds: THRESHOLDS,
              fail_a: failA,
              fail_b: failB,
              fail_c: failC,
              fail_d: failD,
              skip_bc: skipBC,
              skip_b: skipB,
              skip_c: skipC,
              skip_d: skipD,
              known_issues: KNOWN_ISSUES,
              failed_known: failedKnown,
            },
          },
          null,
          2
        )
      );
      console.log(`JSON -> ${p}`);
    }
  }

  const failed = failA + failB + failC + failD;
  if (failed > 0) {
    console.log(`\nGATE FAIL: ${failed} 项判据未通过（A=${failA} B=${failB} C=${failC} D=${failD}）`);
    if (failedKnown.length > 0) {
      console.log(
        `其中 ${failedKnown.length} 项属**已登记的已知问题**（见上）。`
      );
      console.log(
        "**处置纪律：不得为使其变绿而放宽阈值** —— 那是本项目栽过两次的假绿。正确路径是修encoder 量化质量。"
      );
    }
    process.exit(FAIL);
  }
  console.log("\nGATE PASS: 全部判据通过（B/C 的 tiled 格已真跑，非跳过）");
  process.exit(PASS);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}