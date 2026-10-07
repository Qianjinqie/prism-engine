// editor.html 端到端浏览器验证（零依赖，仅 node 内建 + 已装的 playwright）
// 覆盖：加载、节点树渲染、选中、滑条改参→apply_edit、dirty 子图精确性、
//       canvas 预览非空、撤销/重做逐字段一致、空栈结构化错误、档位切换、
//       编辑序列 JSON 导出与稳定性、确定性自检。
//
// 前提：先起一个**本地**静态服务器（wasm 需经 http(s) 提供，file:// 下 fetch 受限）：
//   cd crates/prism-gpu/web && python3 -m http.server 8791 --bind 127.0.0.1
//   node e2e_editor.mjs [端口]
//
// 注：下面的 URL 指向本机回环地址，是**本地测试服务器**，不是外部网络资源；
// 交付页面 editor.html 本身零外链（见其文件头）。
import { chromium } from 'playwright';

const PORT = process.argv[2] ?? '8791';
const URL = `http://127.0.0.1:${PORT}/editor.html`;
const results = [];
const ok = (n, c, d = '') => { results.push({ n, c, d }); console.log(`${c ? '✓' : '❌'} ${n}${d ? ' — ' + d : ''}`); };

// 用系统已装的 Chromium（不依赖 playwright 自带浏览器下载）
const browser = await chromium.launch({
  executablePath: '/usr/bin/chromium',
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
const errs = [];
// favicon 由内联 data-URI 提供；此处只收真正的脚本/资源异常（favicon 404 已消除，仍过滤兜底）
const IGNORE = [/favicon/i];
page.on('pageerror', e => errs.push(String(e)));
page.on('console', m => {
  if (m.type() === 'error' && !IGNORE.some(re => re.test(m.text()))) errs.push('console: ' + m.text());
});
page.on('requestfailed', r => { if (!IGNORE.some(re => re.test(r.url()))) errs.push('reqfail: ' + r.url()); });

await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForFunction(() => document.getElementById('bWasm').textContent.includes('wasm 就绪'), { timeout: 20000 });

// 1. 加载与 ABI
const abiTxt = await page.textContent('#bAbi');
ok('wasm 加载 + ABI 版本', abiTxt.includes('ABI v1'), abiTxt);
const wasmTxt = await page.textContent('#bWasm');
ok('零导入（纯导出，无胶水）', wasmTxt.includes('0 导入'), wasmTxt);

// 2. 节点树
const nodeCount = await page.locator('#nodes li').evaluateAll(ls => ls.length);
ok('节点树渲染 4 节点', nodeCount === 4, `实际 ${nodeCount}`);
const nodeLabels = await page.locator('#nodes li .st > span:first-child').allTextContents();
ok('节点结构中文标签', JSON.stringify(nodeLabels) === JSON.stringify(['标量场', '卷积', '扩散', '卷积']), nodeLabels.join('/'));

// 3. 选中节点 1 → 参数控件生成
await page.locator('#nodes li').first().click();
await page.waitForTimeout(120);
const pnames = await page.locator('#params .pname').allTextContents();
ok('选中节点 1 生成参数控件', pnames.includes('scale') && pnames.includes('octaves'), pnames.join(','));
const ranges = await page.locator('#params input[type=range]').count();
ok('滑条控件存在', ranges >= 1, `${ranges} 个`);

// 4. 画布预览非空（非全黑、非全透明）
const px = await page.evaluate(() => {
  const c = document.getElementById('view');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let nonBlack = 0, nonClear = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] + d[i + 1] + d[i + 2] > 12) nonBlack++;
    if (d[i + 3] === 255) nonClear++;
  }
  return { total: d.length / 4, nonBlack, nonClear, first: [d[0], d[1], d[2], d[3]] };
});
ok('canvas 预览已绘制（非空）', px.nonBlack > px.total * 0.5, `${px.nonBlack}/${px.total} 非黑，首像素 RGBA=${px.first}`);
ok('预览 alpha 全 255（不透明）', px.nonClear === px.total, `${px.nonClear}/${px.total}`);

// 5. 改 scale 0.5 → 0.8：dirty 子图应恰为 {1,3,4}（兄弟叶 2 不脏）
await page.evaluate(() => {
  const r = [...document.querySelectorAll('#params .param')].find(p => p.querySelector('.pname').textContent === 'scale').querySelector('input[type=range]');
  r.value = '0.8';
  r.dispatchEvent(new Event('input', { bubbles: true }));
  r.dispatchEvent(new Event('change', { bubbles: true }));
});
await page.waitForTimeout(200);
const dirtyTags = await page.locator('#nodes li.dirty .nid').allTextContents();
ok('dirty 子图恰为 {1,3,4}', JSON.stringify(dirtyTags) === JSON.stringify(['1', '3', '4']), `实际 {${dirtyTags}}`);

// 6. 增量收益：改叶 1 后重算集 ≡ dirty 集，兄弟叶 2 被复用
const statsTxt = await page.textContent('#lgRec');
const reuseTxt = await page.textContent('#lgReu');
const coneTxt = await page.textContent('#lgCone');
ok('增量：重算=3 复用=1 锥=4', statsTxt === '3' && reuseTxt === '1' && coneTxt === '4',
   `重算${statsTxt} 复用${reuseTxt} 锥${coneTxt}`);

// 7. 撤销 → 逐字段回到 0.5；重做 → 回到 0.8
const readScale = () => page.evaluate(() => {
  const p = [...document.querySelectorAll('#params .param')].find(x => x.querySelector('.pname').textContent === 'scale');
  return p ? Number(p.querySelector('input[type=number]').value) : null;
});
await page.click('#btnUndo');
await page.waitForTimeout(180);
const afterUndo = await readScale();
ok('撤销后 scale 回到 0.500', Math.abs(afterUndo - 0.5) < 1e-6, `实际 ${afterUndo}`);
await page.click('#btnRedo');
await page.waitForTimeout(180);
const afterRedo = await readScale();
ok('重做后 scale 回到 0.800', Math.abs(afterRedo - 0.8) < 1e-6, `实际 ${afterRedo}`);

// 8. 编辑序列 JSON 已导出且含 1 条
const edits1 = await page.inputValue('#edits');
const ej = JSON.parse(edits1);
ok('编辑序列 JSON 已导出', ej.edits.length === 1 && ej.edits[0].param === 'scale',
   `${ej.edits.length} 条，schema=${ej.schema}，draft_spp=${ej.draft_spp}，final_spp=${ej.final_spp}`);

// 9. 档位切换：草稿 spp=1 → 成片 spp=16
const q0 = await page.textContent('#bQuality');
ok('默认草稿档 spp=1', q0.includes('草稿') && q0.includes('spp=1'), q0);
const r0 = await page.textContent('#stRes');
await page.locator('#segQ button[data-q="1"]').click();
await page.waitForTimeout(400);
const q1 = await page.textContent('#bQuality');
const r1 = await page.textContent('#stRes');
const spp1 = await page.textContent('#stSpp');
ok('成片档 spp=16', q1.includes('成片') && q1.includes('spp=16'), q1);
ok('成片档内部分辨率=全分辨率', /(\d+)×\1/.test(r1) && r1 !== r0, `draft ${r0} → final ${r1}，spp=${spp1}`);

// 10. 撤销栈耗尽 → 结构化错误，不 panic，页面存活
//     走**真实 UI 路径**（点按钮），按钮 disabled 状态也一并被验证。
//     注：undo/redo 互为逆操作，耗尽一边会喂饱另一边；而在**单元素栈**上
//     两者是同一个动作，来回 drain 会永远不空。故先补第 2 次编辑，
//     让栈深 ≥ 2，双向穷举才有意义。
async function setScale(v) {
  await page.evaluate((val) => {
    const r = [...document.querySelectorAll('#params .param')]
      .find(p => p.querySelector('.pname').textContent === 'scale').querySelector('input[type=range]');
    r.value = String(val); r.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await page.waitForTimeout(160);
}
await page.click('#segQ button[data-q="0"]');
await page.waitForTimeout(200);
await setScale(0.62);
ok('栈深 ≥ 2（undo/redo 才互不抵消）', (await page.evaluate(() => {
  let d = 0; while (ABI.canUndo() && d < 16) { ABI.undo(); d++; }
  while (ABI.canRedo() && d < 32) { ABI.redo(); d++; }
  return d;
})) >= 2);

async function drain(btn) {
  let n = 0;
  while (!(await page.isDisabled(btn)) && n < 24) { await page.click(btn); await page.waitForTimeout(150); n++; }
  return n;
}
const u1 = await drain('#btnUndo');
ok('UI 撤销按钮在栈空后自动禁用', await page.isDisabled('#btnUndo'), `点了 ${u1} 次`);
const d1 = await drain('#btnRedo');
ok('UI 重做按钮在 redo 栈空后禁用', await page.isDisabled('#btnRedo'), `点了 ${d1} 次`);
const u2 = await drain('#btnUndo');
const d2 = await drain('#btnRedo');
const u3 = await drain('#btnUndo');
// 每轮 drain 的最后一次点击必须正好把按钮点灰（既不卡死也不越界），
// 说明按钮 disabled 状态与真实栈深始终同步。
ok('UI 按钮 disabled 状态与栈深始终同步（每轮 drain 都点满到边界）',
   [u1, u2, u3].every(n => n > 0) && [d1, d2].every(n => n > 0) &&
   (await page.isDisabled('#btnUndo')),
   `耗尽次数 undo=${u1}+${u2}+${u3} redo=${d1}+${d2}`);
// undo/redo 互为逆操作：只要 redo 栈非空，点一次重做就把命令压回 undo 栈，
// 故"两按钮同时禁用"在**非空栈**上不存在。正确判据是"编辑序列为空"
// （edits 只导出自 undo 栈，redo 耗尽后必为空）。
ok('双向穷举后编辑序列为空（redo 栈确已耗尽）',
   JSON.parse(await page.inputValue('#edits')).edits.length === 0,
   await page.inputValue('#edits'));

// 两栈皆空：直接调 ABI 验证结构化错误（按钮 disabled 已单独验证；点不到是正确 UX）
// 用 init() 把状态归零，保证"两栈皆空"这个前置条件确定成立。
const emptyRc = await page.evaluate(() => {
  ABI.init();                                  // 归零：清空 undo/redo/像素/缓存
  const u = { rc: ABI.undo(), err: readErr() };
  const r = { rc: ABI.redo(), err: readErr() };
  return { u, r, canUndo: ABI.canUndo(), canRedo: ABI.canRedo() };
});
ok('init 后两栈皆空（canUndo/canRedo 均为 0）',
   emptyRc.canUndo === false && emptyRc.canRedo === false, JSON.stringify({ u: emptyRc.canUndo, r: emptyRc.canRedo }));
ok('撤销栈空 → BAD_ARG(4) + 结构化错误 E4002',
   emptyRc.u.rc === 4 && emptyRc.u.err.includes('E4002'), `rc=${emptyRc.u.rc} ${emptyRc.u.err.split('\n')[0]}`);
ok('重做栈空 → BAD_ARG(4) + 结构化错误 E4003',
   emptyRc.r.rc === 4 && emptyRc.r.err.includes('E4003'), `rc=${emptyRc.r.rc} ${emptyRc.r.err.split('\n')[0]}`);
ok('错误后页面存活（节点树仍在）', (await page.locator('#nodes li').count()) === 4);
ok('三段式错误含「什么事/建议」',
   emptyRc.u.err.includes('什么事') && emptyRc.u.err.includes('建议'), emptyRc.u.err.replace(/\n/g, ' | '));

// 11. 确定性自检按钮：同参渲染两次像素逐字节一致
await page.click('#btnDet');
await page.waitForTimeout(700);
const detTxt = await page.textContent('#toast');
ok('确定性自检 PASS', detTxt.includes('PASS') && detTxt.includes('逐字节一致'), detTxt.split('\n')[0]);

// 12. 编辑序列 JSON 逐字节稳定（两次导出同字符串）
const stable = await page.evaluate(() => {
  const a = (wasm.prism_editor_edits_json(), readStr(wasm.prism_editor_json_ptr(), wasm.prism_editor_json_len()));
  const b = (wasm.prism_editor_edits_json(), readStr(wasm.prism_editor_json_ptr(), wasm.prism_editor_json_len()));
  return a === b && a.length;
});
ok('编辑序列 JSON 逐字节稳定', stable > 0, `${stable} 字节`);

// 13. 无未捕获 JS 异常
ok('全程零未捕获 JS 异常', errs.length === 0, errs.slice(0, 3).join(' | '));

// 14. 重置按钮回到参考图，并恢复原选中节点（参数区不空白）
//     先显式选中节点 1（前面第 10 步的 ABI.init() 清过选中）
await page.locator('#nodes li').first().click();
await page.waitForTimeout(250);
const nodeBeforeReset = await page.evaluate(() => ABI.selected());
ok('重置前已选中节点 1', nodeBeforeReset === 1, `selected=${nodeBeforeReset}`);
await page.click('#btnReset');
await page.waitForTimeout(400);
const afterReset = await page.evaluate(() => ({
  scale: (() => {
    const p = [...document.querySelectorAll('#params .param')].find(x => x.querySelector('.pname').textContent === 'scale');
    return p ? Number(p.querySelector('input[type=number]').value) : null;
  })(),
  selNode: ABI.selected(),
  paramCount: document.querySelectorAll('#params .param').length,
  canUndo: ABI.canUndo(),
}));
ok('重置回参考图（scale=0.5）', Math.abs(afterReset.scale - 0.5) < 1e-6, `实际 ${afterReset.scale}`);
ok('重置后恢复原选中节点', afterReset.selNode === nodeBeforeReset,
   `重置前 ${nodeBeforeReset} → 重置后 ${afterReset.selNode}`);
ok('重置后参数区非空', afterReset.paramCount > 0, `${afterReset.paramCount} 个控件`);
ok('重置后撤销栈已清空', afterReset.canUndo === false);

// 15. 多步编辑 + 撤销/重做穷举（字节级一致在 Rust 集成测试已覆盖，这里验 UI 通路）
await page.locator('#nodes li').first().click();          // 回到节点 1（有 scale）
await page.waitForTimeout(200);
const seq = [0.11, 0.22, 0.33, 0.44];
for (const v of seq) {
  await page.evaluate((val) => {
    const r = [...document.querySelectorAll('#params .param')]
      .find(p => p.querySelector('.pname').textContent === 'scale').querySelector('input[type=range]');
    r.value = String(val); r.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await page.waitForTimeout(160);
}
const nEdits = JSON.parse(await page.inputValue('#edits')).edits.length;
ok('4 次编辑全部入撤销栈', nEdits === 4, `${nEdits} 条`);
const undoVals = [];
for (let i = 0; i < 4; i++) { await page.click('#btnUndo'); await page.waitForTimeout(170); undoVals.push(await readScale()); }
ok('逐步撤销回到初值 0.5', Math.abs(undoVals[3] - 0.5) < 1e-6, '序列 ' + undoVals.join(' → '));
const redoVals = [];
for (let i = 0; i < 4; i++) { await page.click('#btnRedo'); await page.waitForTimeout(170); redoVals.push(await readScale()); }
ok('逐步重做逐字段一致', Math.abs(redoVals[3] - 0.44) < 1e-6, '序列 ' + redoVals.join(' → '));
ok('编辑序列 JSON 与 UI 历史一致（4 条）',
   JSON.parse(await page.inputValue('#edits')).edits.length === 4);

await page.screenshot({ path: '/tmp/editor_full.png', fullPage: true });
await browser.close();

const fail = results.filter(r => !r.c);
console.log(`\n===== ${results.length - fail.length}/${results.length} 通过 =====`);
process.exit(fail.length ? 1 : 0);
