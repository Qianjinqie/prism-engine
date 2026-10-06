// ===========================================================================
// perf_check.mjs — D-14 桌面 GPU 性能验收脚本（入库版）。
//
// 用途：在**真实桌面 GPU** 上，测量 Phase 4 两个未验收的性能目标：
//   ① 1080p 路径追踪交互帧率（`rt_wavefront` 的 connect/accumulate 链）
//   ② 百万粒子模拟帧率（`sim_particles` 的 emit/integrate/hash_cells 链）
//
// 【本脚本与 realgpu_check.mjs 的分工】
//   realgpu_check.mjs —— 判**正确性**（14 项 PASS/FAIL/SKIP），设备是手机或任意 GPU。
//   perf_check.mjs     —— 判**性能**（每帧耗时/帧率），设备**必须是桌面 GPU**。
//   D-14 明确定义：用户手机 = 正确性测试设备，**不是**性能基准设备。
//   故本脚本测量前会先做**确定性前置门禁**，并检测是否为软件适配器。
//
// ===========================================================================
// 【铁律 1】失败必须显式非 0 退出，绝不静默返回"通过"。
//   任何前置检查失败（Xvfb / Playwright / 浏览器 / adapter / wasm）都 exit != 0。
//   尤其禁止"拿不到 adapter 就 SKIP 全部然后报 PASS"。
//
// 【铁律 2】性能测速前必须先做**确定性校验**：同一输入跑两遍，逐字节比对。
//   若确定性都不成立，性能数字没有意义 —— 脚本**直接失败退出**，不继续测速。
//
// 【铁律 3】软件适配器（swiftshader/lavapipe/llvmpipe/software）只作 smoke，
//   脚本显式警告"此结果不能作为性能验收凭据"，并以**退出码 3**（软件适配器）
//   与真 GPU 的结论区分开，绝不把软件光栅速度包装成验收结论。
//
// 【铁律 4】不伪造数字。测不了就说测不了，外推必须显式标注为外推。
// ===========================================================================
//
// 用法：
//   node perf_check.mjs                          # 自动起静态服务 + 找浏览器
//   node perf_check.mjs --url http://127.0.0.1:8080/perf.html
//   node perf_check.mjs --timeout 900            # SwiftShader 很慢，默认 600s
//   node perf_check.mjs --frames 60              # 每项采样帧数（默认 30）
//   node perf_check.mjs --warmup 5               # 计时前的预热帧数（默认 5）
//   node perf_check.mjs --require-gpu            # 强制要求真 GPU（软件适配器直接判失败）
//   node perf_check.mjs --json                   # 机器可读输出（便于归档验收凭据）
//   node perf_check.mjs --self-check             # 只跑确定性 + adapter，不测速
//
// 退出码：
//   0  = 真 GPU 上测完且确定性成立（结论可用作性能验收）
//   1  = 前置检查失败 / 确定性校验失败 / 页面报错
//   2  = 用法错误（--help）
//   3  = 环境是**软件适配器**：测完了但结果不可作为性能验收凭据（见警告）
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
    "perf_check.mjs — D-14 桌面 GPU 性能验收（1080p 路径追踪 + 百万粒子）\n" +
    "\n" +
    "用法：\n" +
    "  node perf_check.mjs                        # 自动起静态服务 + 自动找浏览器\n" +
    "  node perf_check.mjs --url http://127.0.0.1:8080/perf.html\n" +
    "  node perf_check.mjs --timeout 900           # SwiftShader 很慢，默认 600s\n" +
    "  node perf_check.mjs --frames 60             # 每项采样帧数（默认 30）\n" +
    "  node perf_check.mjs --warmup 5              # 计时前预热帧数（默认 5）\n" +
    "  node perf_check.mjs --require-gpu           # 软件适配器直接判失败\n" +
    "  node perf_check.mjs --json                  # 机器可读输出\n" +
    "  node perf_check.mjs --self-check            # 只跑确定性 + adapter 探测\n" +
    "\n" +
    "退出码：0 = 真 GPU 测完且确定性成立；1 = 前置/确定性/页面失败；\n" +
    "        2 = 本帮助；3 = 软件适配器（结果不可作性能验收凭据）。\n" +
    "任何环境问题都会非 0 退出，绝不会静默返回“通过”。",
  );
  process.exit(2);
}

const TIMEOUT_S = Number(arg("timeout", "600"));
const URL_ARG = arg("url", null);
const FRAMES = Number(arg("frames", "30"));
const WARMUP = Number(arg("warmup", "5"));
const REQUIRE_GPU = HAS("require-gpu");
const JSON_OUT = HAS("json");
const SELF_CHECK = HAS("self-check");

if (!Number.isFinite(FRAMES) || FRAMES < 1) {
  console.error("!! --frames 必须为正整数"); process.exit(2);
}
if (!Number.isFinite(WARMUP) || WARMUP < 0) {
  console.error("!! --warmup 必须为非负整数"); process.exit(2);
}

const problems = [];
// 进度日志统一走 info()：--json 模式下 stdout 必须只含 JSON，故进度改走 stderr。
function info(msg) { if (JSON_OUT) console.error(msg); else console.log(msg); }
function die(msg, code = 1) {
  if (JSON_OUT) {
    console.log(JSON.stringify({ ok: false, verdict: "FATAL", error: msg }, null, 2));
  } else {
    console.error(`\n!! perf_check: FATAL — ${msg}`);
  }
  process.exit(code);
}

// ---------------------------------------------------------------------------
// 前置 1：Playwright 可用性（与 realgpu_check.mjs 同款逻辑，复用不重造）。
// 不做静默降级——没有 Playwright 就跑不了有头浏览器，必须让调用方知道。
// ---------------------------------------------------------------------------
const PW_BARE = ["playwright"];
const PW_ABS = [
  "/usr/lib/node_modules/playwright/index.js",
  "/usr/local/lib/node_modules/playwright/index.js",
  ...(process.env.NVM_BIN ? [`${process.env.NVM_BIN}/../lib/node_modules/playwright`] : []),
  "/root/.nvm/versions/node/v22.13.1/lib/node_modules/playwright",
  // 项目本地 node_modules（本仓库实测装在这里）。
  resolve(HERE, "../../../node_modules/playwright/index.js"),
];

async function loadPlaywright() {
  const tried = [];
  for (const spec of PW_BARE) {
    try {
      const mod = await import(spec);
      if (mod?.chromium) return { chromium: mod.chromium, spec };
      const viaDefault = mod?.default?.chromium;
      if (viaDefault) return { chromium: viaDefault, spec: `${spec}（CJS default）` };
      tried.push(`${spec}（可解析但无 chromium 导出）`);
    } catch (e) {
      tried.push(`${spec}（${e.code || e.message}）`);
    }
  }
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
      "   或在本仓库根目录：npm i playwright（脚本已含 node_modules 回退路径）。",
  );
}
info(`==> 前置检查 1/4：Playwright 可用 OK（${pwSpec}）`);

// ---------------------------------------------------------------------------
// 前置 2：Chromium 可执行文件存在（playwright 自带未下载时回退系统浏览器）。
// ---------------------------------------------------------------------------
let execPath;
try {
  execPath = chromium.executablePath();
} catch (e) {
  die(`无法定位 Chromium 可执行文件：${e.message}`);
}
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
    info(
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
info(`==> 前置检查 2/4：Chromium 存在 OK（${execPath}）`);

// ---------------------------------------------------------------------------
// 前置 3：显示环境。有 DISPLAY 或 Xvfb 才能起有头浏览器；无显示环境时尝试
// 自动拉起 Xvfb。两者都失败 → 明确报错退出。
//
// 【性能测量特别注记】有头模式会占一个真实窗口合成器；性能测量应在**独显直连**
// 的桌面会话里跑（不是远程桌面/虚拟帧缓冲），否则测的是合成器节流后的帧率。
// ---------------------------------------------------------------------------
let xvfbProc = null;
const BASE_ARGS = [
  "--no-sandbox",
  "--disable-gpu-sandbox",
  "--disable-dev-shm-usage",
  "--enable-unsafe-webgpu",
  "--enable-webgpu-developer-features",
  // 关掉后台节流：否则窗口不可见/失焦时 rAF 会被压到 1fps，测出的"帧率"是假的。
  "--disable-background-timer-throttling",
  "--disable-backgrounding-occluded-windows",
  "--disable-renderer-backgrounding",
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
      "   或在有显示环境的桌面机器上直接跑本脚本。",
    );
  }
  const dpy = `:${Math.floor(90 + Math.random() * 9)}`;
  info(`    无可用 DISPLAY，尝试拉起 Xvfb ${dpy} ...`);
  xvfbProc = spawn("Xvfb", [dpy, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"], {
    stdio: "ignore",
  });
  let xvfbUp = false;
  for (let i = 0; i < 25; i += 1) {
    await sleep(200);
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
  info(`    Xvfb 就绪：DISPLAY=${dpy}`);
    info(
    "    ⚠️ Xvfb 是虚拟帧缓冲：此环境下即使拿到真 GPU，帧率也可能被虚拟合成器节流，\n" +
    "       结论仅供参考。桌面 GPU 性能验收请在真实显示会话里跑。",
  );
}
info(`==> 前置检查 3/4：显示环境 OK（DISPLAY=${process.env.DISPLAY}）`);

// ---------------------------------------------------------------------------
// 前置 4：被测页面可达。默认加载专用性能页 perf.html。
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
      const rel = normalize(urlPath === "/" ? "/perf.html" : urlPath).replace(/^(\.\.[/\\])+/, "");
      const file = join(HERE, rel);
      if (!file.startsWith(HERE)) { res.writeHead(403).end("forbidden"); return; }
      const body = await readFile(file);
      res.writeHead(200, {
        "content-type": MIME[extname(file)] || "application/octet-stream",
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
  pageUrl = `http://127.0.0.1:${port}/perf.html`;
  info(`    静态服务已起：${pageUrl}（根目录 ${HERE}）`);
}

try {
  const resp = await fetch(pageUrl, { method: "GET" });
  if (!resp.ok) die(`页面不可达：${pageUrl} → HTTP ${resp.status}`);
  const wasmPath = resolve(HERE, "prism_gpu.wasm");
  if (!existsSync(wasmPath)) {
    die(`wasm 产物不存在：${wasmPath}\n   先跑：cd crates/prism-gpu/web && ./build.sh`);
  }
  const ws = (await stat(wasmPath)).size;
  info(`==> 前置检查 4/4：页面与 wasm 就绪 OK（wasm ${ws} 字节）`);

  // =======================================================================
  // 起浏览器 + 驱动 perf 页
  // =======================================================================
  info(`\n==> 启动 Chromium（DISPLAY=${process.env.DISPLAY}）`);
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: execPath,
      headless: false, // 有头：真实 GPU 路径、可拿到真合成器时序
      args: BASE_ARGS,
      env: { ...process.env },
    });
  } catch (e) {
    die(
      `Chromium 启动失败：${e.message}\n` +
      "   有头模式在容器里常因沙箱/共享内存失败；--no-sandbox 已带上。",
    );
  }

  const cleanup = async () => {
    try { if (browser) await browser.close(); } catch { /* 忽略 */ }
    try { if (server) server.close(); } catch { /* 忽略 */ }
    try { if (xvfbProc) xvfbProc.kill("SIGKILL"); } catch { /* 忽略 */ }
  };

  let exitCode = 1;
  try {
    const page = await browser.newPage();
    const consoleLines = [];
    page.on("console", (m) => {
      consoleLines.push(`[${m.type()}] ${m.text()}`);
      if (consoleLines.length > 500) consoleLines.shift();
    });
    page.on("pageerror", (e) => consoleLines.push(`[pageerror] ${e.message}`));

    info(`==> 打开性能页 ${pageUrl}`);
    await page.goto(pageUrl, { waitUntil: "domcontentloaded", timeout: 60000 });

    // perf.html 暴露 window.__PERF__ = { ready:Promise, run(opts) }。
    // 等它 ready（wasm 实例化 + adapter 申领完成）。
    info("==> 等待 perf.html 就绪（wasm + adapter）...");
    const ready = await page.evaluate(async () => {
      if (!window.__PERF__) return { ok: false, why: "perf.html 未暴露 window.__PERF__（版本不匹配？）" };
      try {
        const r = await window.__PERF__.ready;
        return r;
      } catch (e) {
        return { ok: false, why: `ready 拒绝：${e && e.message || e}` };
      }
    }).catch((e) => ({ ok: false, why: `求值异常：${e.message}` }));

    if (!ready || !ready.ok) {
      await cleanup();
      die(`perf.html 未就绪：${ready?.why || "未知"}\n` +
          "   这属于前置失败（wasm 未加载 / 无 WebGPU adapter），不进行测速。");
    }

    const gpu = ready.gpu;             // { vendor, architecture, device, isSoftware, featureTimestamps }
    const abi = ready.abi;             // { version, simPCount, particlesBytes, wfWidth, wfHeight, wfSpp }
    const software = !!gpu.isSoftware;

    if (!JSON_OUT) {
      info(`\n${"=".repeat(70)}`);
      info(`adapter：vendor=${gpu.vendor} arch=${gpu.architecture} device=${gpu.device}`);
      info(`软件适配器：${software ? "是（!! 不可作性能验收凭据 !!）" : "否"}`);
      info(`ABI：v${abi.version}；sim_particles=${abi.simPCount} 粒子（${abi.particlesBytes}B）；` +
                  `rt_wavefront=${abi.wfWidth}×${abi.wfHeight} 像素 × spp=${abi.wfSpp}`);
      info(`${"=".repeat(70)}\n`);
    }

    if (software) {
      info("⚠️⚠️⚠️  警告：当前是**软件适配器**（SwiftShader/lavapipe/llvmpipe 之类）。");
      info("       它跑在 CPU 上做软件光栅化，帧率与真 GPU 无任何可比性。");
      info("       本次测量结果**不能作为性能验收凭据**，仅可用于验证脚本与链路本身。");
      info("       请在真实桌面 GPU 上重跑以获得验收结论。\n");
    }
    if (REQUIRE_GPU && software) {
      await cleanup();
      die("--require-gpu 指定但当前为软件适配器：拒绝以软件适配器产出性能结论。", 1);
    }

    // ---------------------------------------------------------------------
    // 驱动 perf.html：内部顺序是 ① 确定性门禁 ② 预热 ③ 计时采样。
    //   page 侧任何失败都会以 {ok:false, stage, why} 返回 → 脚本判红。
    // ---------------------------------------------------------------------
    info(`==> 运行测量（确定性门禁 → 预热 ${WARMUP} 帧 → 每项采样 ${FRAMES} 帧）...`);
    const result = await page.evaluate(
      async (opts) => window.__PERF__.run(opts),
      { frames: FRAMES, warmup: WARMUP, selfCheck: SELF_CHECK },
    ).catch((e) => ({ ok: false, stage: "driver", why: `run() 异常：${e.message}` }));

    await cleanup();

    // ---------------------------------------------------------------------
    // 判定 + 输出
    // ---------------------------------------------------------------------
    if (!result || !result.ok) {
      const fatal = {
        ok: false,
        verdict: "FAIL",
        stage: result?.stage || "unknown",
        error: result?.why || "测量未返回结果",
        adapter: gpu,
      };
      if (JSON_OUT) console.log(JSON.stringify(fatal, null, 2));
      else {
        console.error(`\n==> perf_check: FAIL（stage=${fatal.stage}）`);
        console.error(`   - ${fatal.error}`);
        if (consoleLines.length > 0) {
          console.error("\n--- 页面日志（末 20 条）---");
          for (const l of consoleLines.slice(-20)) console.error(`  ${l}`);
        }
      }
      process.exit(1);
    }

    const out = {
      ok: true,
      adapter: gpu,
      abi,
      softwareAdapter: software,
      determinism: result.determinism,
      measurements: result.measurements,   // [{id, scale, framesMs:{min,median,mean,p95}, fps:{...}, gap}]
      capabilityGaps: result.capabilityGaps,
      provenance: result.provenance,
    };

    if (JSON_OUT) {
      console.log(JSON.stringify(out, null, 2));
    } else {
      console.log(`\n${"=".repeat(70)}`);
      console.log("确定性前置门禁");
      console.log(`${"=".repeat(70)}`);
      console.log(`  同输入两遍逐字节比对：${result.determinism.identical ? "一致 ✅" : "不一致 ❌"}`);
      console.log(`  指纹 A=${result.determinism.hashA}  B=${result.determinism.hashB}` +
                  `（FNV-1a32，${result.determinism.bytes}B × ${result.determinism.runs} 遍）`);

      console.log(`\n${"=".repeat(70)}`);
      console.log("测量结果（每帧耗时，取多帧统计）");
      console.log(`${"=".repeat(70)}`);
      for (const m of result.measurements) {
        console.log(`\n  ▸ ${m.id}`);
        console.log(`    规模：${m.scale.human}（${m.scale.detail}）`);
        console.log(`    耗时/帧：min=${fmtMs(m.framesMs.min)} median=${fmtMs(m.framesMs.median)} ` +
                    `mean=${fmtMs(m.framesMs.mean)} p95=${fmtMs(m.framesMs.p95)}`);
        console.log(`    帧率：median=${fmtFps(m.fps.median)} min=${fmtFps(m.fps.max1OverMin)}（1/最长帧）` +
                    `（n=${m.framesMs.n} 帧，预热 ${m.warmup} 帧后计时）`);
        if (m.target) {
          console.log(`    目标：${m.target.human} → ${m.target.met ? "达标 ✅" : "未达标 ❌"}` +
                      (m.target.note ? `（${m.target.note}）` : ""));
        }
        if (m.gap) console.log(`    ⚠️ 规模缺口：${m.gap}`);
      }

      if (result.capabilityGaps && result.capabilityGaps.length) {
        console.log(`\n${"=".repeat(70)}`);
        console.log("表达能力缺口（当前 ABI 无法表达目标规模，结论为外推）");
        console.log(`${"=".repeat(70)}`);
        for (const g of result.capabilityGaps) {
          console.log(`  ▸ ${g.id}`);
          console.log(`    ${g.why}`);
          console.log(`    需要接口：${g.needed}`);
        }
      }

      console.log(`\n${"=".repeat(70)}`);
      console.log(`provenance：${result.provenance}`);
      console.log(`${"=".repeat(70)}`);
    }

    // 退出码判定（每一条都可能非 0，无静默通过路径）：
    if (!result.determinism.identical) {
      problems.push("确定性前置门禁失败：同输入两遍结果不一致 —— 性能数字无意义");
    }
    if (software) {
      // 软件适配器：测完了，但结论不构成性能验收凭据。
      if (!JSON_OUT) {
        console.error("\n==> perf_check: SOFTWARE-ADAPTER（退出码 3）");
        console.error("   测得数字仅证明脚本与链路可用；**不可作为性能验收凭据**。");
      }
      exitCode = 3;
    } else if (problems.length > 0) {
      if (!JSON_OUT) {
        console.error("\n==> perf_check: FAIL");
        for (const p of problems) console.error(`   - ${p}`);
      }
      exitCode = 1;
    } else {
      if (!JSON_OUT) {
        console.log("\n==> perf_check: PASS（真 GPU，确定性成立，结论可用作性能验收）");
      }
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

// ---------------------------------------------------------------------------
// 格式化辅助
// ---------------------------------------------------------------------------
function fmtMs(x) {
  if (!Number.isFinite(x)) return "n/a";
  return x >= 100 ? `${x.toFixed(1)}ms` : x >= 10 ? `${x.toFixed(2)}ms` : `${x.toFixed(3)}ms`;
}
function fmtFps(x) {
  if (!Number.isFinite(x)) return "n/a";
  return `${x.toFixed(2)} fps`;
}
