// ===========================================================================
// realgpu_check.mjs — D-14 真机 / 有头浏览器 WebGPU 回归脚本（入库版）。
//
// 用途：在**真实 GPU 环境**（或 SwiftShader 软件适配器）里，用有头 Chromium
// 打开 harness 页面（index.html），点"全部运行"，收集每个测试项的
// PASS/FAIL/SKIP 与错误面板，判定退出码。这是 CI 覆盖不到的那一层——
// CI 只验"wasm 能加载、数值自洽"，**验不了** `createShaderModule` 在真实
// 驱动上是否接受某份 WGSL、pipeline 是否真能建起来。
//
// 与 host_test.mjs 的分工：
//   host_test.mjs  —— 纯 node，无浏览器，验 ABI + 数值（CI 跑）
//   realgpu_check.mjs —— 有头浏览器 + WebGPU，验真机链路（人工/专项跑）
//
// ===========================================================================
// 【铁律】失败必须显式非 0 退出，绝不静默返回"通过"。
//   Xvfb 起不来 / Playwright 未装 / 浏览器起不来 / adapter 拿不到 /
//   wasm 没实例化 / 有 FAIL 项 —— 任何一条都 exit != 0。
//   尤其禁止"拿不到 adapter 就 SKIP 全部然后报 PASS"：那等于把"环境没配好"
//   伪装成"代码没问题"，是回归脚本最危险的失败模式。
//
// 用法：
//   node realgpu_check.mjs                      # 自动起静态服务 + 找浏览器
//   node realgpu_check.mjs --url http://127.0.0.1:8080/index.html
//   node realgpu_check.mjs --timeout 900         # SwiftShader 很慢，默认 600s
//   node realgpu_check.mjs --allow-skip          # 允许全部 SKIP（仅自检用）
//   node realgpu_check.mjs --headed              # 已有 Xvfb/DISPLAY 时用有头模式
//
// 退出码：
//   0  = 至少 1 项 PASS 且无 FAIL
//   1  = 有 FAIL 项 / 前置检查失败
//   2  = 用法错误
// ===========================================================================

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve, join, extname, normalize } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const HERE = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// 参数解析
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
function arg(name, dflt) {
  const i = argv.indexOf(`--${name}`);
  return i !== -1 && argv[i + 1] !== undefined ? argv[i + 1] : dflt;
}
const HAS = (name) => argv.includes(`--${name}`);

if (HAS("help")) {
  console.log(
    "realgpu_check.mjs — D-14 真机 / 有头浏览器 WebGPU 回归\n" +
    "\n" +
    "用法：\n" +
    "  node realgpu_check.mjs                        # 自动起静态服务 + 自动找浏览器\n" +
    "  node realgpu_check.mjs --url http://127.0.0.1:8080/index.html\n" +
    "  node realgpu_check.mjs --timeout 900           # SwiftShader 很慢，默认 600s\n" +
    "  node realgpu_check.mjs --allow-skip            # 允许全 SKIP（仅环境自检用）\n" +
    "\n" +
    "退出码：0 = 至少 1 项 PASS 且无 FAIL；1 = 有 FAIL 或前置检查失败；2 = 本帮助。\n" +
    "任何环境问题（Xvfb 起不来 / Playwright 未装 / adapter 拿不到）都会非 0 退出，\n" +
    "绝不会静默返回“通过”。",
  );
  process.exit(2);
}

const TIMEOUT_S = Number(arg("timeout", "600"));
const URL_ARG = arg("url", null);
const ALLOW_SKIP = HAS("allow-skip");

const problems = [];
function die(msg, code = 1) {
  console.error(`\n!! realgpu_check: FATAL — ${msg}`);
  process.exit(code);
}

// ---------------------------------------------------------------------------
// 前置 1：Playwright 可用性。动态 import，找不到就明确报错退出。
// 不做静默降级——没有 Playwright 就跑不了有头浏览器，必须让调用方知道。
//
// 注意：ESM 的 `import("playwright")` **不会**去 npm 全局目录找包（全局包装在
// CJS 的 require 解析路径里，ESM 解析器不查）。所以除了裸 import，还要按
// 显式路径重试一次常见安装位置，否则 `npm i -g playwright` 明明装了却报"未安装"。
// ---------------------------------------------------------------------------

/**
 * 常见的 Playwright 安装位置。
 * - 裸说明符：走 ESM 解析（项目内有本地 node_modules 时命中）。
 * - 绝对路径：覆盖 npm 全局安装。**全局包通常是 CommonJS**，ESM 的
 *   `import()` 对 CJS 只能拿到 `default`，拿不到 `chromium` 具名导出——所以
 *   绝对路径一律用 createRequire 加载，才能正确取到 `.chromium`。
 */
const PW_BARE = ["playwright"];
const PW_ABS = [
  "/usr/lib/node_modules/playwright/index.js",
  "/usr/local/lib/node_modules/playwright/index.js",
  // nvm：$NVM_BIN/../lib/node_modules/playwright
  ...(process.env.NVM_BIN ? [`${process.env.NVM_BIN}/../lib/node_modules/playwright`] : []),
  "/root/.nvm/versions/node/v22.13.1/lib/node_modules/playwright",
];

async function loadPlaywright() {
  const tried = [];

  // 1) 裸说明符（ESM）
  for (const spec of PW_BARE) {
    try {
      const mod = await import(spec);
      if (mod?.chromium) return { chromium: mod.chromium, spec };
      // CJS 包走 ESM 导入：具名导出可能缺失，回退 default。
      const viaDefault = mod?.default?.chromium;
      if (viaDefault) return { chromium: viaDefault, spec: `${spec}（CJS default）` };
      tried.push(`${spec}（可解析但无 chromium 导出）`);
    } catch (e) {
      tried.push(`${spec}（${e.code || e.message}）`);
    }
  }

  // 2) 绝对路径（require，覆盖 npm 全局 CJS 安装）
  for (const base of PW_ABS) {
    try {
      if (!existsSync(base)) { tried.push(`${base}（不存在）`); continue; }
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

const { chromium, spec: pwSpec, tried: pwTried } = await loadPlaywright();
if (!chromium) {
  die(
    "Playwright 未安装或不可解析。已尝试：\n" +
      pwTried.map((t) => `     - ${t}\n`).join("") +
      "   安装：npm i -g playwright && npx playwright install chromium\n" +
      "   或用 NODE_PATH=<全局 node_modules 目录> 指定路径。",
  );
}
console.log(`==> 前置检查 1/4：Playwright 可用 OK（${pwSpec}）`);

// ---------------------------------------------------------------------------
// 前置 2：Chromium 可执行文件存在。playwright 的浏览器是独立下载的，
// npm 包装好 ≠ 浏览器二进制在位，这里实测一次。
// ---------------------------------------------------------------------------
let execPath;
try {
  execPath = chromium.executablePath();
} catch (e) {
  die(`无法定位 Chromium 可执行文件：${e.message}`);
}
// playwright 自带版本可能未下载（如精简镜像 / 无网络环境），此时回退到系统
// Chromium。**只在 playwright 的路径确实不存在时**才回退，不静默换版本——
// 系统 Chromium 的 WebGPU 支持度必须由前置 4 实测确认，不能假设。
if (!execPath || !existsSync(execPath)) {
  const sysCandidates = [
    process.env.CHROME_PATH,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
  ].filter(Boolean);
  const found = sysCandidates.find((p) => existsSync(p));
  if (found) {
    console.log(
      `   （playwright 自带 Chromium 未下载，回退系统浏览器：${found}）\n` +
      "   注意：系统浏览器的 WebGPU 支持度由前置 4 实测判定，不假设可用。",
    );
    execPath = found;
  }
}
if (!execPath || !existsSync(execPath)) {
  die(
    `Chromium 未下载（playwright 路径=${chromium.executablePath?.() ?? "<null>"}，\n` +
    "   且系统 Chromium 也未找到）。\n" +
    "   运行：npx playwright install chromium\n" +
    "   或指定环境变量 CHROME_PATH=/path/to/chrome",
  );
}
console.log(`==> 前置检查 2/4：Chromium 存在 OK（${execPath}）`);

// ---------------------------------------------------------------------------
// 前置 3：显示环境。有 DISPLAY 或 Xvfb 才能起有头浏览器；无显示环境时尝试
// 自动拉起 Xvfb。两者都失败 → 明确报错退出。
// ---------------------------------------------------------------------------
let xvfbProc = null;
const BASE_ARGS = [
  "--no-sandbox",
  "--disable-gpu-sandbox",
  "--disable-dev-shm-usage",
  // WebGPU 在无头/软件环境下要的旗标。真机（有独显）时这些旗标无害。
  "--enable-unsafe-webgpu",
  "--enable-webgpu-developer-features",
];

function displayAlive(dpy) {
  if (!dpy) return false;
  const r = spawnSync("xdpyinfo", ["-display", dpy], { stdio: "ignore" });
  return r.status === 0;
}

if (!process.env.DISPLAY || !displayAlive(process.env.DISPLAY)) {
  const hasXvfb = spawnSync("which", ["Xvfb"], { stdio: "ignore" }).status === 0;
  if (!hasXvfb) {
    die(
      "既无 DISPLAY 也无 Xvfb，无法起有头浏览器。\n" +
      "   安装：apt-get install -y xvfb\n" +
      "   或在有显示环境的机器上直接跑本脚本。",
    );
  }
  const dpy = `:${Math.floor(90 + Math.random() * 9)}`;
  console.log(`    无可用 DISPLAY，尝试拉起 Xvfb ${dpy} ...`);
  xvfbProc = spawn("Xvfb", [dpy, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"], {
    stdio: "ignore",
  });
  let xvfbUp = false;
  for (let i = 0; i < 25; i += 1) {
    await sleep(200);
    // Xvfb 可能起来但立刻崩；两种情况都要看到。
    if (xvfbProc.exitCode !== null) break;
    if (existsSync(`/tmp/.X11-unix/X${dpy.slice(1)}`)) { xvfbUp = true; break; }
  }
  if (!xvfbUp) {
    xvfbProc.kill("SIGKILL");
    die(
      `Xvfb 拉起失败（DISPLAY=${dpy}）。\n` +
      "   常见原因：/tmp/.X11-unix 不存在或权限不足。\n" +
      "   排查：ls -ld /tmp/.X11-unix；或改用有显示环境的机器。",
    );
  }
  process.env.DISPLAY = dpy;
  console.log(`    Xvfb 就绪：DISPLAY=${dpy}`);
}
console.log(`==> 前置检查 3/4：显示环境 OK（DISPLAY=${process.env.DISPLAY}）`);

// ---------------------------------------------------------------------------
// 前置 4：被测页面可达。要么用户给了 --url，要么本脚本起一个静态服务
// （wasm 必须走 http，file:// 会被 CORS 拦——这一点在 README 里也标注了）。
// ---------------------------------------------------------------------------
let server = null;
let pageUrl = URL_ARG;

if (!pageUrl) {
  const MIME = {
    ".html": "text/html; charset=utf-8",
    ".wasm": "application/wasm",
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
  };
  server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);

      // 目录穿越防护：规范化后必须仍落在 web/ 内。
      const rel = normalize(urlPath === "/" ? "/index.html" : urlPath).replace(/^(\.\.[/\\])+/, "");
      const file = join(HERE, rel);
      if (!file.startsWith(HERE)) { res.writeHead(403).end("forbidden"); return; }
      const body = await readFile(file);
      res.writeHead(200, {
        "content-type": MIME[extname(file)] || "application/octet-stream",
        // wasm 侧若将来用 SharedArrayBuffer 需要这两行；先写上避免以后踩坑。
        "cache-control": "no-store",
      });
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
  pageUrl = `http://127.0.0.1:${port}/index.html`;
  console.log(`    静态服务已起：${pageUrl}（根目录 ${HERE}）`);
}

try {
  const resp = await fetch(pageUrl, { method: "GET" });
  if (!resp.ok) die(`页面不可达：${pageUrl} → HTTP ${resp.status}`);
  const wasmPath = resolve(HERE, "prism_gpu.wasm");
  if (!existsSync(wasmPath)) {
    die(`wasm 产物不存在：${wasmPath}\n   先跑：cd crates/prism-gpu/web && ./build.sh`);
  }
  const ws = (await stat(wasmPath)).size;
  console.log(`==> 前置检查 4/4：页面与 wasm 就绪 OK（wasm ${ws} 字节）`);

  // =======================================================================
  // 起浏览器 + 驱动页面
  // =======================================================================
  console.log(`\n==> 启动 Chromium（DISPLAY=${process.env.DISPLAY}）`);
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: execPath,
      headless: false, // 有头：真实 GPU 路径与无头不同，这正是本脚本的意义
      args: BASE_ARGS,
      env: { ...process.env },
    });
  } catch (e) {
    die(
      `Chromium 启动失败：${e.message}\n` +
      "   有头模式在容器里常因浏览器沙箱/共享内存限制失败；启动参数里的 --no-sandbox 已带上。\n" +
      "   若在极简容器里，可试 --disable-software-rasterizer 之外的组合。",
    );
  }

  const cleanup = async () => {
    try { if (browser) await browser.close(); } catch { /* 忽略关闭异常 */ }
    try { if (server) server.close(); } catch { /* 忽略 */ }
    try { if (xvfbProc) xvfbProc.kill("SIGKILL"); } catch { /* 忽略 */ }
  };

  let exitCode = 0;
  try {
    // ===================================================================
    // 【设计铁律】跑测试之前，本脚本对 harness 页**只做一件事**：点 #runall。
    //   任何额外的 evaluate() / 另开标签页都会扰动 harness 自己的
    //   requestAdapter → requestDevice → settle 时序，症状是
    //   device.init / wgsl.compile / gpu.scan_block / gpu.histogram
    //   报「预留句柄未绑定到真实 GPUDevice」（实测 4/9 FAIL，且**稳定复现**，
    //   不是抖动）。回归驱动器应当是被动观察者，不该碰被测对象的状态。
    //   因此 navigator.gpu / adapter 等环境诊断一律放在**跑完之后**做。
    // ===================================================================
    const page = await browser.newPage();
    const consoleLines = [];
    page.on("console", (m) => {
      consoleLines.push(`[${m.type()}] ${m.text()}`);
      if (consoleLines.length > 500) consoleLines.shift();
    });
    page.on("pageerror", (e) => consoleLines.push(`[pageerror] ${e.message}`));

    console.log(`==> 打开 harness 页 ${pageUrl}`);
    await page.goto(pageUrl, { waitUntil: "domcontentloaded", timeout: 60000 });

    // --- 等 wasm 实例化（testcount 出现 "Rust"）----------------------
    // 提示：SwiftShader 是纯 CPU 光栅化。若机器正被其它重活占用（如同时在跑
    // cargo test），requestDevice 会因超时而失败，症状是 device.init /
    // wgsl.compile / gpu.* 报「预留句柄未绑定到真实 GPUDevice」。这是**环境
    // 争抢**而非代码缺陷——本脚本如实报 FAIL，不做任何掩盖，但跑之前请先让
    // 机器空闲，否则结论不可信。
    console.log("==> 等待 wasm 实例化（Rust 测试项清单出现）...");
    let manifest = "";
    for (let i = 0; i < 80; i += 1) {
      manifest = await page.evaluate(
        () => document.getElementById("testcount")?.textContent || "",
      ).catch(() => "");
      if (manifest && manifest.includes("Rust")) break;
      await sleep(500);
    }
    if (!manifest.includes("Rust")) {
      problems.push("wasm 未在 40s 内实例化（testcount 未出现 'Rust'）");
    } else {
      console.log(`    wasm OK：${manifest.trim()}`);
    }

    // --- 点"全部运行"并等按钮恢复 ------------------------------------
    console.log(`==> 点击「全部运行」（最长等 ${TIMEOUT_S}s，SwiftShader 慢）...`);
    await page.evaluate(() => {
      const b = document.getElementById("runall");
      if (!b) throw new Error("找不到 #runall 按钮（harness 结构变了？）");
      b.click();
    }).catch((e) => problems.push(`点击 #runall 失败：${e.message}`));

    let finished = false;
    for (let i = 0; i < TIMEOUT_S; i += 1) {
      const enabled = await page.evaluate(
        () => document.getElementById("runall")?.disabled === false,
      ).catch(() => false);
      if (enabled) { finished = true; break; }
      await sleep(1000);
    }
    if (!finished) {
      problems.push(`运行超时：${TIMEOUT_S}s 内 #runall 未恢复可用（可能某项 GPU 测试挂死）`);
    }

    // --- 收集结果 ------------------------------------------------------
    const result = await page.evaluate(() => {
      const items = [...document.querySelectorAll("#tests li")].map((li) => ({
        id: (li.id || "").replace(/^item-/, ""),
        state: li.querySelector(".tstate")?.textContent?.trim() || "?",
        detail: li.querySelector(".tbody")?.textContent?.trim() || "",
      }));
      const errs = [...document.querySelectorAll("#errpanel .e")].map((e) =>
        e.textContent.trim(),
      );
      return {
        items,
        errs,
        env: document.getElementById("env")?.textContent?.trim() || "",
        errcount: document.getElementById("errcount")?.textContent?.trim() || "",
      };
    }).catch((e) => ({ items: [], errs: [`收集结果失败：${e.message}`], env: "", errcount: "" }));

    // --- 判定 ----------------------------------------------------------
    const tally = { PASS: 0, FAIL: 0, SKIP: 0, PENDING: 0, BAD_ARG: 0, "?": 0 };
    for (const it of result.items) {
      tally[it.state] = (tally[it.state] || 0) + 1;
    }

    console.log(`\n${"=".repeat(66)}`);
    console.log(`环境：${result.env || "(harness 未提供)"}`);
    console.log(`错误计数：${result.errcount || "0"}`);
    console.log(`${"=".repeat(66)}\n`);
    for (const it of result.items) {
      console.log(`  [${it.state.padEnd(5)}] ${it.id}`);
      if (it.detail && it.state !== "PASS") console.log(`          ${it.detail}`);
    }
    console.log(
      `\n汇总：PASS=${tally.PASS} FAIL=${tally.FAIL} SKIP=${tally.SKIP}` +
      ` PENDING=${tally.PENDING} BAD_ARG=${tally.BAD_ARG} ?=${tally["?"]}` +
      `（共 ${result.items.length} 项）`,
    );

    if (result.errs.length > 0) {
      console.log(`\n--- 错误面板（${result.errs.length} 条）---`);
      for (const e of result.errs.slice(0, 30)) console.log(`  ${e}`);
    }

    // 判定规则（每一条都可能导致非 0 退出，不存在"静默通过"路径）：
    if (result.items.length === 0) {
      problems.push("未收集到任何测试项（wasm 未加载或选择器失效）");
    }
    if (tally.FAIL > 0) problems.push(`${tally.FAIL} 个测试项 FAIL`);
    if (tally["?"] > 0) problems.push(`${tally["?"]} 个测试项状态未知（运行未正常收尾）`);
    // ---- 错误面板计数必须为 0（submit/绑定/编译类验证错误的兜底断言）----
    //
    // 【为什么必须断言它，而不是只看 PASS/FAIL】harness 的 `prism_dispatch` 会在
    // 页面内开 validation error scope 并把非空错误记进错误面板；Rust 侧 `check_errors`
    // 应在 collect 阶段把这些错误翻成 FAIL。但**两层之间若有一处漏看**，就会出现
    // 「错误面板里有验证错误，而所有测试项仍报 PASS」——那正是"断言存在但与真实
    // 执行脱节"的假绿形态（v2.8 事故即：submit 无效、回读零值、却满足有限数断言）。
    // 故这里把面板计数本身钉成硬判据：非 0 即红，不存在"面板有错但测试全绿"的路径。
    const mErr = /([0-9]+)/.exec(String(result.errcount || ""));
    const errCount = mErr ? Number(mErr[1]) : 0;
    if (errCount > 0) {
      problems.push(
        `harness 错误面板计数 ${errCount} > 0：存在验证/编译错误但未全部翻成 FAIL` +
        `（假绿风险——submit 无效时回读零值会满足"有限数"类断言）`,
      );
    }
    if (tally.PASS === 0) {
      problems.push(
        tally.SKIP > 0
          ? `全部 ${tally.SKIP} 项 SKIP、0 项 PASS —— 环境未提供 GPU，不能视为通过` +
            (ALLOW_SKIP ? "（--allow-skip 已指定，仅作环境自检）" : "")
          : "0 项 PASS",
      );
    }

    // --- 跑完之后：环境诊断（navigator.gpu + adapter）-------------------
    // 全部放在判定之后，且**只作诊断**：拿不到 adapter 不额外改判，因为
    // navigator.gpu 缺失 / 无适配器的后果已由上面 9 项状态如实反映
    // （全 SKIP 或全 FAIL 都会判红）。避免"测试其实过了却因事后诊断报红"。
    console.log("\n==> 环境诊断（跑完后，不影响判定）...");
    let gpuPresent = false;
    let adapterInfo = null;
    try {
      const probe = await browser.newPage();
      try {
        // 复用同一 origin（http(s) 即安全上下文）。必须在 harness 跑完后才做。
        await probe.goto(pageUrl, { waitUntil: "domcontentloaded", timeout: 30000 })
          .catch(() => {});
        gpuPresent = await probe.evaluate(() => typeof navigator.gpu !== "undefined")
          .catch(() => false);
        if (gpuPresent) {
          for (let i = 0; i < 60; i += 1) {
            adapterInfo = await probe.evaluate(async () => {
              const a = await navigator.gpu.requestAdapter();
              if (!a) return { ok: false, why: "requestAdapter() 返回 null（无可用适配器）" };
              const info = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
              return {
                ok: true,
                vendor: info.vendor || "?",
                architecture: info.architecture || "?",
                device: info.device || "?",
              };
            }).catch((e) => ({ ok: false, why: `探针求值异常：${e.message}` }));
            if (adapterInfo.ok) break;
            await sleep(1000);
          }
        }
      } finally {
        await probe.close().catch(() => {});
      }
    } catch (e) {
      console.error(`    （诊断跳过：${e.message}）`);
    }

    if (!gpuPresent) {
      console.error("    navigator.gpu 不存在 —— 该浏览器/构建不支持 WebGPU，或页面非安全上下文。");
      console.error("      排查：① 页面是否走 http(s)（file:// 与 about:blank 非安全上下文）；② 缺 --enable-unsafe-webgpu？");
    } else if (adapterInfo?.ok) {
      console.log(
        `    adapter：vendor=${adapterInfo.vendor} ` +
        `arch=${adapterInfo.architecture} device=${adapterInfo.device}`,
      );
    } else {
      console.error(`    navigator.gpu 存在但申领 adapter 失败：${adapterInfo?.why || "未知"}`);
    }

    await cleanup();

    console.log(`\n${"=".repeat(66)}`);
    if (problems.length > 0) {
      console.error("==> realgpu_check: FAIL");
      for (const p of problems) console.error(`   - ${p}`);
      if (consoleLines.length > 0) {
        console.error("\n--- 页面日志（末 20 条）---");
        for (const l of consoleLines.slice(-20)) console.error(`  ${l}`);
      }
      exitCode = ALLOW_SKIP && tally.FAIL === 0 && tally.PASS > 0 ? 0 : 1;
    } else {
      console.log(
        `==> realgpu_check: PASS（${tally.PASS} 项 PASS` +
        (tally.SKIP > 0 ? `，${tally.SKIP} 项 SKIP（该环境不支持对应能力）` : "") +
        "）",
      );
      exitCode = 0;
    }
  } catch (e) {
    await cleanup();
    die(`执行过程异常：${e.stack || e.message}`);
  }
  process.exit(exitCode);
} catch (e) {
  if (server) server.close();
  if (xvfbProc) xvfbProc.kill("SIGKILL");
  die(e.message);
}
