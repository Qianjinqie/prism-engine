#!/usr/bin/env node
// ===========================================================================
// hair_parity_gpu.mjs — Marschner CPU/GPU 数值对拍：GPU 一半（在真 WebGPU 上跑）
// ===========================================================================
//
// # 为什么存在（本路最核心的产出）
//
// `crates/prism-rt/src/hair_scatter.rs`（Rust）与 `crates/prism-gpu/shaders/
// hair_scatter.wgsl`（WGSL）是**两份独立手写**的同一组 Marschner 公式。在本
// 文件之前，两侧**从未被放在一起比较过**——偏差多大完全未知。Marschner 是
// 能量守恒敏感的物理公式（三瓣之和 ≈ 入射能量），偏差 1e-3 就足以让头发
// "发灰/发黑"。本文件提供 GPU 一半：把 golden vector 里的 12 组入参灌进
// **真实 WebGPU 设备**的 compute pipeline，取回 R/TT/TRT。
//
// 【铁律】这不是"模拟"。
//   1. WGSL 是**真编译**（createShaderModule）+ **真 dispatch**（真 compute
//      pipeline + 真 storage buffer + mapAsync 回读），不是 JS 复刻公式。
//      因此它能抓住"WGSL 根本编译不过"这类缺陷——事实上本轮就抓到了：
//      `hair_scatter.wgsl` 的 `a_r`/`a_tt`/`a_trt` 形参个数与调用点不匹配，
//      该文件**从来无法通过 createShaderModule**；因为它是 `LIBRARIES`
//      （无入口、不参与 kernel 编译），这个缺陷一直绿灯。
//   2. 拿不到 adapter / 编译失败 / 回读失败，**一律非 0 退出**，绝不静默
//      返回"通过"。绝不允许"跑不了就 SKIP 然后报 PASS"——那等于把环境问题
//      伪装成代码没问题，是回归脚本最危险的失败模式。
//
// # 沙箱路径（与 realgpu_check.mjs 同一套既有机制，不新造）
//
//   Xvfb（有头虚拟显示）+ Playwright/Chromium + SwiftShader（软件适配器）。
//   `realgpu_check.mjs` 已验证该路径在本沙箱可跑；本文件复用同样的前置检查
//   顺序（Playwright → Chromium → Xvfb → 静态服务），故不引入新机制。
//
// # 用法
//
//   node hair_parity_gpu.mjs                       # 自跑 golden vector，打印误差表
//   node hair_parity_gpu.mjs --json                # 机器可读（host_test.mjs 用）
//   node hair_parity_gpu.mjs --url http://…/x.html # 复用已起好的服务
//
// # 退出码
//   0 = 全部向量在阈值内   1 = 有向量超阈值 / GPU 链路失败   2 = 用法错误
// ===========================================================================

import { readFileSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve, join, extname, normalize } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHADER_DIR = resolve(HERE, "../shaders");
const GOLDEN = resolve(HERE, "hair_golden.json");

/**
 * 误差阈值（abs）。
 *
 * 【重要】这是**实测后**定的，不是拍脑袋定的宽松值。定阈值的依据：
 *   1. 两侧都是 f32 单精度，且同构到"相同的表达式形状 + 相同结合序"，
 *      故理论上应**逐位一致**；残余误差只来自 exp 归约的舍入与 `sqrt` 的
 *      末位差异（不同后端 sqrt 的最后 1 ulp 允许不同，IEEE 只保证 correctly
 *      rounded，不保证跨实现一致）。
 *   2. 因此阈值只需覆盖"1~2 ulp 的相对量级"。对 f32，1 ulp 的相对量级是
 *      2^-23 ≈ 1.19e-7；本文件取 **1e-5 绝对 + 1e-5 相对**作为门禁线，
 *      比 1 ulp 宽约两个数量级（留给不同 SwiftShader 版本的 exp/sqrt 实现
 *      差异），但**远比"物理上有意义的 1e-3"严 100 倍**。
 *   3. 之所以不用"相对误差"单判据：TT/TRT 常出现 1e-30 量级的下溢值，
 *      相对误差在分母趋零时会爆掉。故**绝对为主、相对为辅**，且对"CPU 与
 *      GPU 都为 0 或都极小"的格子用绝对判据兜底。
 *
 * 任何一条超阈值都**必须定位根因并修**，不许放宽阈值糊过去。
 */
const ABS_TOL = 1e-5;
const REL_TOL = 1e-5;

/** f32 位模式十六进制 → float（无损）。 */
function bitsToF32(hex) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(parseInt(hex, 16) >>> 0, 0);
  return b.readFloatLE(0);
}

/** 展开 `//!include <file>`（递归、去重），与 Rust 侧 `shaders::expand` 同语义。 */
function expandIncludes(label, seen = new Set()) {
  if (seen.has(label)) return "";
  seen.add(label);
  const p = join(SHADER_DIR, label);
  if (!existsSync(p)) return "";
  const raw = readFileSync(p, "utf8");
  let out = "";
  for (const line of raw.split("\n")) {
    const m = /^\s*\/\/!include\s+(\S+)\s*$/.exec(line);
    if (m) out += expandIncludes(m[1], seen) + "\n";
    else out += line + "\n";
  }
  return out;
}

/**
 * 组装 compute shader：`rt_common` + `hair_scatter`（库件，原样 include）
 * + 对拍入口。
 *
 * 【为什么可以直接 include 库件】`hair_scatter.wgsl` 登记在 `LIBRARIES`、
 * 无任何入口，故必须由外部提供入口。这同时验证了"库件自洽"（展开后所有
 * 符号可解析）——若库件里引用了未定义的函数，`createShaderModule` 会报错。
 */
function buildParityShader() {
  // 【只展开 hair_scatter.wgsl 一份】它自身带 `//!include rt_common.wgsl`，
  // 若再单独拼一份 rt_common 会导致 PI / cos_sin 等**重复声明**（WGSL 不允许
  // 顶层重声明，createShaderModule 会报 redeclaration）。展开器本身是递归的，
  // 一份就够。
  const hair = expandIncludes("hair_scatter.wgsl");
  if (!hair.includes("fn cos_sin(")) {
    throw new Error("展开后的 hair_scatter.wgsl 缺少 cos_sin——rt_common 的 include 未生效");
  }
  // 入参结构体补齐到 16 字节倍数（storage 里 struct 的 align 由最大成员决定，
  // 这里显式补 3 个 pad 到 48B，规避任何后端的对齐差异）。
  const harness = `
struct HairIn {
  theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32,
  beta_m: f32, k: f32, eta: f32, sigma_a: f32,
  radius: f32, pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<storage, read> hair_in: array<HairIn>;
@group(0) @binding(1) var<storage, read_write> hair_out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn hair_parity_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = hair_in[gid.x];
  let s = hair_scatter_s_precise(v.theta_i, v.phi_i, v.theta_o, v.phi_o,
                                 v.beta_m, v.k, v.eta, v.sigma_a, v.radius);
  hair_out[gid.x] = vec4<f32>(s.x, s.y, s.z, 0.0);
}
`;
  return { code: `${hair}\n${harness}`, bytes: hair.length };
}

// ---------------------------------------------------------------------------
// 前置检查（与 realgpu_check.mjs 同纪律：任一失败即非 0 退出）
// ---------------------------------------------------------------------------

function die(msg, code = 1) {
  console.error(`\n!! hair_parity_gpu: FATAL — ${msg}`);
  process.exit(code);
}

const PW_ABS = [
  "/usr/lib/node_modules/playwright/index.js",
  "/usr/local/lib/node_modules/playwright/index.js",
  ...(process.env.NVM_BIN ? [`${process.env.NVM_BIN}/../lib/node_modules/playwright`] : []),
  "/root/.nvm/versions/node/v22.13.1/lib/node_modules/playwright",
];

async function loadPlaywright() {
  const tried = [];
  for (const spec of ["playwright"]) {
    try {
      const mod = await import(spec);
      if (mod?.chromium) return { chromium: mod.chromium, spec };
      tried.push(`${spec}（可解析但无 chromium 导出）`);
    } catch (e) {
      tried.push(`${spec}（${e.code || e.message}）`);
    }
  }
  for (const base of PW_ABS) {
    if (!existsSync(base)) { tried.push(`${base}（不存在）`); continue; }
    try {
      const req = createRequire(import.meta.url);
      const mod = req(base);
      if (mod?.chromium) return { chromium: mod.chromium, spec: base };
      tried.push(`${base}（可加载但无 chromium 导出）`);
    } catch (e) {
      tried.push(`${base}（${e.code || e.message}）`);
    }
  }
  return { chromium: null, tried };
}

const BASE_ARGS = [
  "--no-sandbox",
  "--disable-gpu-sandbox",
  "--disable-dev-shm-usage",
  "--enable-unsafe-webgpu",
  "--enable-webgpu-developer-features",
  // SwiftShader 软件光栅：显式要求 Vulkan-on-SwiftShader，否则可能拿不到 adapter。
  "--use-vulkan=swiftshader",
  "--use-angle=swiftshader",
];

function displayAlive(dpy) {
  if (!dpy) return false;
  return spawnSync("xdpyinfo", ["-display", dpy], { stdio: "ignore" }).status === 0;
}

/** 起 Xvfb（若需要）。返回 { dpy, proc }。 */
async function ensureDisplay() {
  if (process.env.DISPLAY && displayAlive(process.env.DISPLAY)) {
    return { dpy: process.env.DISPLAY, proc: null };
  }
  if (spawnSync("which", ["Xvfb"], { stdio: "ignore" }).status !== 0) {
    die("既无 DISPLAY 也无 Xvfb，无法起有头浏览器。安装：apt-get install -y xvfb");
  }
  const dpy = `:${Math.floor(90 + Math.random() * 9)}`;
  const proc = spawn("Xvfb", [dpy, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"], {
    stdio: "ignore",
  });
  for (let i = 0; i < 25; i += 1) {
    await sleep(200);
    if (proc.exitCode !== null) break;
    if (existsSync(`/tmp/.X11-unix/X${dpy.slice(1)}`)) {
      process.env.DISPLAY = dpy;
      return { dpy, proc };
    }
  }
  proc.kill("SIGKILL");
  die(`Xvfb 拉起失败（DISPLAY=${dpy}）。排查：ls -ld /tmp/.X11-unix`);
}

/** 起静态服务（wasm/页面必须走 http，file:// 非安全上下文）。 */
async function ensureServer() {
  const MIME = {
    ".html": "text/html; charset=utf-8", ".wasm": "application/wasm",
    ".mjs": "text/javascript; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
  };
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
      const rel = normalize(urlPath === "/" ? "/index.html" : urlPath).replace(/^(\.\.[/\\])+/, "");
      const file = join(HERE, rel);
      if (!file.startsWith(HERE)) { res.writeHead(403).end("forbidden"); return; }
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", "cache-control": "no-store" });
      res.end(body);
    } catch (e) {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      res.end(`not found: ${e.message}`);
    }
  });
  const port = await new Promise((res, rej) => {
    server.once("error", rej);
    server.listen(0, "127.0.0.1", () => res(server.address().port));
  });
  return { server, origin: `http://127.0.0.1:${port}` };
}

/**
 * 真跑一遍 GPU：编译 WGSL、dispatch、回读。
 * @returns {Promise<Float32Array>} 长度 = vectors.length * 4（vec4，w 未用）
 */
async function runOnGpu(vectors) {
  const { chromium, spec, tried } = await loadPlaywright();
  if (!chromium) {
    die("Playwright 未安装或不可解析。已尝试：\n" + tried.map((t) => `     - ${t}\n`).join("") +
        "   安装：npm i -g playwright && npx playwright install chromium");
  }
  let execPath;
  try { execPath = chromium.executablePath(); } catch (e) { die(`无法定位 Chromium：${e.message}`); }
  if (!execPath || !existsSync(execPath)) {
    const sys = [process.env.CHROME_PATH, "/usr/bin/chromium", "/usr/bin/chromium-browser",
                 "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"].filter(Boolean);
    const found = sys.find((p) => existsSync(p));
    if (found) execPath = found;
  }
  if (!execPath || !existsSync(execPath)) {
    die(`Chromium 未下载且系统 Chromium 缺失。运行：npx playwright install chromium（playwright=${spec}）`);
  }
  const { dpy, proc: xvfbProc } = await ensureDisplay();
  const { server, origin } = await ensureServer();

  let browser;
  try {
    browser = await chromium.launch({ executablePath: execPath, headless: false, args: BASE_ARGS, env: { ...process.env } });
  } catch (e) {
    server.close();
    if (xvfbProc) xvfbProc.kill("SIGKILL");
    die(`Chromium 启动失败：${e.message}`);
  }

  const cleanup = async () => {
    try { if (browser) await browser.close(); } catch { /* ignore */ }
    try { server.close(); } catch { /* ignore */ }
    try { if (xvfbProc) xvfbProc.kill("SIGKILL"); } catch { /* ignore */ }
  };

  try {
    const page = await browser.newPage();
    const pageErrs = [];
    page.on("pageerror", (e) => pageErrs.push(e.message));

    // 页面只需一个安全上下文（http 即是），用来跑 WebGPU。
    await page.goto(`${origin}/index.html`, { waitUntil: "domcontentloaded", timeout: 60000 });

    const { code, bytes } = buildParityShader();

    // 把 shader 源码 + 入参数组交给页面，在**页面内**建 device/pipeline/buffer。
    // 之所以不在 node 侧建：WebGPU 只存在于浏览器上下文。
    const result = await page.evaluate(async ({ code, vectors }) => {
      if (typeof navigator.gpu === "undefined") {
        return { ok: false, why: "navigator.gpu 不存在（浏览器不支持 WebGPU 或非安全上下文）" };
      }
      let adapter = null;
      for (let i = 0; i < 30; i += 1) {
        adapter = await navigator.gpu.requestAdapter();
        if (adapter) break;
        await new Promise((r) => setTimeout(r, 500));
      }
      if (!adapter) return { ok: false, why: "requestAdapter() 返回 null（无可用适配器）" };
      const device = await adapter.requestDevice();
      if (!device) return { ok: false, why: "requestDevice() 返回 null" };

      // 捕获 WGSL 编译/校验错误（uncapturederror）——编译失败必须**显式**报出。
      const gpuErrs = [];
      device.addEventListener?.("uncapturederror", (ev) => {
        gpuErrs.push(String(ev.error?.message || ev.error));
      });

      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo?.();
      const errs = (info?.messages || []).filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
      if (errs.length) {
        return { ok: false, why: `WGSL 编译失败：\n${errs.join("\n")}` };
      }

      const n = vectors.length;
      const inBytes = new Float32Array(n * 12);
      for (let i = 0; i < n; i += 1) {
        const v = vectors[i];
        const o = i * 12;
        inBytes[o + 0] = v.theta_i; inBytes[o + 1] = v.phi_i;
        inBytes[o + 2] = v.theta_o; inBytes[o + 3] = v.phi_o;
        inBytes[o + 4] = v.beta_m;  inBytes[o + 5] = v.k;
        inBytes[o + 6] = v.eta;     inBytes[o + 7] = v.sigma_a;
        inBytes[o + 8] = v.radius;  inBytes[o + 9] = 0; inBytes[o + 10] = 0; inBytes[o + 11] = 0;
      }
      const outBytes = new Float32Array(n * 4);

      const inBuf = device.createBuffer({ size: inBytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const outBuf = device.createBuffer({ size: outBytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      device.queue.writeBuffer(inBuf, 0, inBytes);

      const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "hair_parity_main" } });
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inBuf } },
          { binding: 1, resource: { buffer: outBuf } },
        ],
      });
      // 【防假绿 · 见 D-31】只靠全局 `uncapturederror` 不够：submit 无效
      // （绑定/用法/布局非法）时它未必触发，而 `onSubmittedWorkDone()` 在
      // submit 无效时**仍会 resolve**——回读拿到的是未被 dispatch 覆写的初始
      // 零值。「逐格容差」于是假绿（CPU 恰好为 0 的分量零对零通过）。
      // 故在 dispatch 段精确开一层 validation error scope，作为独立硬判据。
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(n);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      // submit 之后必须 0 个验证错误；非空即说明这次 dispatch 根本没跑。
      const submitErr = await device.popErrorScope();

      // 显式回读（不用 mapAsync 之外的路径，确保值真的落到 JS 可见内存）。
      const readBuf = device.createBuffer({ size: outBytes.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc2 = device.createCommandEncoder();
      enc2.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes.byteLength);
      device.queue.submit([enc2.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();

      const adapterInfo = adapter.info || {};
      return {
        ok: true,
        data: Array.from(got),
        adapter: `${adapterInfo.vendor || "?"}/${adapterInfo.architecture || "?"}/${adapterInfo.description || adapterInfo.device || "?"}`,
        gpuErrs,
        submitErr: submitErr ? String(submitErr.message || submitErr) : "",
      };
    }, { code, vectors });

    if (pageErrs.length) {
      await cleanup();
      die(`页面异常：\n${pageErrs.join("\n")}`);
    }
    if (!result.ok) {
      await cleanup();
      die(result.why);
    }
    if (result.gpuErrs?.length) {
      await cleanup();
      die(`WebGPU 运行时错误：\n${result.gpuErrs.join("\n")}`);
    }
    // 【D-31 防假绿】submit 的有效性独立硬判据——非空即 dispatch 根本没执行。
    if (result.submitErr) {
      await cleanup();
      die(`submit 产生验证错误（dispatch 未执行，回读值为初始零值）：\n${result.submitErr}`);
    }
    // 【D-31 防假绿】输出必须非全零。全零 = dispatch 未覆写输出缓冲 = 假绿。
    if (!result.data.some((x) => x !== 0)) {
      await cleanup();
      die("GPU 输出全为 0——dispatch 未生效（假绿），拒绝按通过处理");
    }
    await cleanup();
    return { data: new Float32Array(result.data), adapter: result.adapter, shaderBytes: bytes };
  } catch (e) {
    await cleanup();
    die(`GPU 执行异常：${e.stack || e.message}`);
  }
}

/** 比对：逐分量算 abs/rel 误差，产出误差表行。 */
function compare(vectors, gpu) {
  const rows = [];
  for (let i = 0; i < vectors.length; i += 1) {
    const v = vectors[i];
    const cpu = [bitsToF32(v.r_bits), bitsToF32(v.tt_bits), bitsToF32(v.trt_bits)];
    const got = [gpu.data[i * 4 + 0], gpu.data[i * 4 + 1], gpu.data[i * 4 + 2]];
    const comps = ["R", "TT", "TRT"].map((name, c) => {
      const cv = cpu[c], gv = got[c];
      const abs = Math.abs(cv - gv);
      // 相对误差：分母取两者较大值（避免 CPU≈0 时分母爆炸）。
      const rel = abs / Math.max(Math.abs(cv), Math.abs(gv), 1e-30);
      return { name, cpu: cv, gpu: gv, abs, rel, ok: abs <= ABS_TOL || rel <= REL_TOL };
    });
    rows.push({ name: v.name, comps, ok: comps.every((c) => c.ok) });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const argv = process.argv.slice(2);
if (argv.includes("--help")) {
  console.log("hair_parity_gpu.mjs — Marschner CPU/GPU 对拍（GPU 一半，真 WebGPU）\n" +
    "用法：node hair_parity_gpu.mjs [--json]\n" +
    "退出码：0 = 全向量在阈值内；1 = 超阈值或链路失败；2 = 用法错误");
  process.exit(2);
}
const AS_JSON = argv.includes("--json");

if (!existsSync(GOLDEN)) {
  die(`CPU golden vector 缺失：${GOLDEN}\n   先跑：cargo test -p prism-rt emit_cpu_golden_vectors_for_gpu_parity`);
}
const golden = JSON.parse(readFileSync(GOLDEN, "utf8"));
if (!Array.isArray(golden.vectors) || golden.vectors.length !== 12) {
  die(`golden vector 数量异常：${golden.vectors?.length}（契约 12）`);
}

const { data, adapter, shaderBytes } = await runOnGpu(golden.vectors);
const rows = compare(golden.vectors, { data });
const allOk = rows.every((r) => r.ok);

if (AS_JSON) {
  console.log(JSON.stringify({
    ok: allOk, adapter, shaderBytes, absTol: ABS_TOL, relTol: REL_TOL,
    rows: rows.map((r) => ({
      name: r.name, ok: r.ok,
      comps: r.comps.map((c) => ({ name: c.name, cpu: c.cpu, gpu: c.gpu, abs: c.abs, rel: c.rel, ok: c.ok })),
    })),
  }, null, 2));
} else {
  console.log(`== Marschner CPU/GPU 对拍（adapter=${adapter}，WGSL ${shaderBytes}B，真 dispatch）==`);
  console.log(`阈值：abs ≤ ${ABS_TOL} 或 rel ≤ ${REL_TOL}（实测后判定，详见文件头注释）\n`);
  console.log("向量".padEnd(20) + "| R abs err".padEnd(13) + "TT abs err".padEnd(13) + "TRT abs err".padEnd(13) + "| 最大 rel");
  console.log("-".repeat(88));
  for (const r of rows) {
    const cells = r.comps.map((c) => (c.abs === 0 ? "0" : c.abs.toExponential(2)).padEnd(11));
    const maxRel = Math.max(...r.comps.map((c) => c.rel));
    console.log(`${r.ok ? " " : "!"}${r.name.padEnd(19)}| ${cells.join(" ")}| ${maxRel.toExponential(2)}`);
  }
  const worstAbs = Math.max(...rows.flatMap((r) => r.comps.map((c) => c.abs)));
  const worstRel = Math.max(...rows.flatMap((r) => r.comps.map((c) => c.rel)));
  console.log("-".repeat(88));
  console.log(`最大绝对误差 = ${worstAbs.toExponential(3)}   最大相对误差 = ${worstRel.toExponential(3)}`);
  console.log(allOk ? "\n==> 全部 12 组向量在阈值内。" : "\n==> FAIL：有分量超阈值（必须定位根因，不许放宽阈值）。");
}
process.exit(allOk ? 0 : 1);
