#!/usr/bin/env node
// ===========================================================================
// wgsl_lib_compile.mjs — 无入口 WGSL 库件的**真实编译链路**（createShaderModule
//                        + getCompilationInfo + 真 dispatch + 回读）
// ===========================================================================
//
// # 为什么存在（本波最核心的产出）
//
// `crates/prism-gpu/shaders/` 下的**库件**（无 `@compute`/`@vertex`/`@fragment`
// 入口）历来**从不进入任何 `createShaderModule`**：
//   * `//!include` 只是**文本拼接**，拼进去的库件随 kernel 一起编译——但只有
//     「被某个有入口的 kernel 真正 include」的那几份才被覆盖；
//   * 登记在 `shaders::LIBRARIES` 的（`rt_scene.wgsl` / `hair_scatter.wgsl`）
//     宿主**永远不会**为它们建 shader module。
// 于是库件里的类型错 / 签名错 / 语义错可以长期存活，在静态门禁全绿下 unnoticed。
//
// 【上一波最严重的一例（本脚本的直接动机）】
//   `hair_scatter.wgsl` 的 `a_r` / `a_tt` / `a_trt` 声明 2 / 4 / 4 个形参，
//   却按 1 / 2 / 2 个实参调用 —— 该文件**根本无法通过 `createShaderModule`**，
//   却因登记为 `LIBRARIES`（无入口、不参与 kernel 编译）一直绿灯。
//
// # 本脚本做什么
//
// 对 `wgsl_lib_manifest.mjs` 里登记的**每一个**库件：
//   1. 展开 `//!include`（递归、去重）得到库件全文；
//   2. 拼一段合成 `@compute @workgroup_size(1)` 入口（该入口**真的调用**库件函数、
//      **真的读**库件自带绑定，并把结果写进 `@group(0)` 的 storage 缓冲）；
//   3. `createShaderModule` → `getCompilationInfo()` → **断言 0 个 error**
//      （warning 记录进 JSON 但**不**判红——warning 不是缺陷）；
//   4. `createComputePipeline` → `createBindGroup`（含库件自带的 group）
//      → 真 `dispatchWorkgroups` → `mapAsync` 回读，断言输出**全为有限数**
//      （NaN/Inf 说明库件在真机上算出了垃圾值，同样是缺陷）。
//
// # 铁律（与 `hair_parity_gpu.mjs` 同纪律）
//
//   1. 这不是"模拟"：WGSL 是**真编译**（createShaderModule）+ **真 dispatch**
//      （真 compute pipeline + 真 storage/uniform buffer + mapAsync 回读）。
//   2. 拿不到 adapter / 编译失败 / 回读失败 / 输出非有限数，**一律非 0 退出**，
//      绝不静默返回"通过"。绝不允许"跑不了就 SKIP 然后报 PASS"——那等于把环境
//      问题伪装成代码没问题，是回归脚本最危险的失败模式。
//   3. **非恒真自证（默认开启，`--no-prove` 关闭）**：光"6 份都编译过了"不能
//      证明这条链路真的能抓错（万一是恒真的呢）。故对每份库件**注入一处确定的
//      编译级缺陷**（形参/实参个数不匹配、builtin 实参个数错），并断言真编译器
//      **必须**报错。任何一份注入后仍然编译通过 = 这条链路对它无效 = 判红。
//
// # 无 GPU 环境路径（复用 hair_parity_gpu.mjs 已验证的机制，不新造）
//
//   Xvfb（有头虚拟显示）+ Playwright/Chromium + SwiftShader（软件适配器）。
//   WebGPU 只存在于浏览器上下文且需安全上下文，故走本地 `createServer` 静态服务
//   （`file://` 不是安全上下文）。
//
// # 用法
//
//   node wgsl_lib_compile.mjs             # 人读：逐份打印编译/dispatch 结果
//   node wgsl_lib_compile.mjs --json      # 机器可读（host_test.mjs 用）
//   node wgsl_lib_compile.mjs --no-prove  # 只跑正向（自证默认开启）
//
// # 退出码
//
//   0 = 全部库件 0 error 编译 + dispatch + 输出有限 + 自证全部抓到
//   1 = 任一库件编译失败 / dispatch 失败 / 输出非有限 / 链路失败 / 自证落空
//   2 = 用法错误
// ===========================================================================

import { readFileSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve, join, extname, normalize } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { LIB_MANIFEST } from "./wgsl_lib_manifest.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SHADER_DIR = resolve(HERE, "../shaders");

/** 合成入口的 workgroup 数（`@workgroup_size(1)`，故 gid.x ∈ [0, DISPATCH)）。 */
const DISPATCH = 8;
/** 合成输出缓冲的 float 数（必须 ≥ DISPATCH）。 */
const OUT_FLOATS = 16;

// ---------------------------------------------------------------------------
// 静态分析：绑定声明 / 函数体 / 静态引用（供"连续前缀"前置校验与报告）
// ---------------------------------------------------------------------------

/**
 * 抽出模块里全部 `@group(g) @binding(b) var<space[,access]> name` 声明。
 *
 * 【为什么要静态算一遍】`layout:"auto"` 只为**被 dispatch 的 entry 静态引用**的
 * binding 生成布局，且要求每个 entry 的引用集是**连续前缀 {0..n-1}**
 * （这正是本项目 D-14 抓到的真机约束之一，见 `wgsl_lint` 规则 G2）。
 * 库件自带 group（`rt_scene` 的 group 2、`rt_shade` 的 group 1+2），
 * 合成入口必须一并满足。GPU 跑之前先静态校验一遍，好把"契约错"与
 * "环境坏"两类失败分开。
 */
function wgslBindingDecls(src) {
  const out = [];
  // 地址空间可带访问模式（`var<storage, read_write>`），故 `var<` 后不写死。
  for (const m of src.matchAll(/@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s*<\s*([^>]+?)\s*>\s*(\w+)/g)) {
    out.push({ group: Number(m[1]), binding: Number(m[2]), space: m[3].trim(), name: m[4] });
  }
  return out;
}

/** 抽出模块里全部 `fn NAME(...) {...}` 的函数体（花括号配对）。 */
function wgslFnBodies(src) {
  const map = new Map();
  const re = /fn\s+(\w+)\s*\(/g;
  let m = re.exec(src);
  while (m !== null) {
    const name = m[1];
    const open = src.indexOf("{", re.lastIndex);
    if (open < 0) break;
    let depth = 0;
    let end = src.length;
    for (let j = open; j < src.length; j += 1) {
      if (src[j] === "{") depth += 1;
      else if (src[j] === "}") { depth -= 1; if (depth === 0) { end = j; break; } }
    }
    map.set(name, src.slice(open, end + 1));
    re.lastIndex = end + 1;
    m = re.exec(src);
  }
  return map;
}

/**
 * 从 `entry` 出发做**传递闭包**，返回该 entry 静态可达的函数名集合。
 *
 * 【为什么必须传递闭包、不能只扫 entry 函数体】库件的函数是层层调用的：
 *   `rt_shade` 的入口调 `bsdf_sample` / `nee_surface`，它们才调
 *   `load_material`（→ `materials` / `prim_mat`）、`sobol_u32`（→ `sobol_dir`）、
 *   `scene_intersect`（→ `tris` / `scene_params`）。只扫入口体会漏掉几乎所有绑定，
 *   判出来的"引用集"是假的。WGSL 无递归，故 visited 集合必然终止。
 */
function entryReachableFns(src, entry) {
  const bodies = wgslFnBodies(src);
  const seen = new Set();
  const stack = [entry];
  while (stack.length) {
    const name = stack.pop();
    if (seen.has(name)) continue;
    seen.add(name);
    const body = bodies.get(name);
    if (body === undefined) continue;
    for (const m of body.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)) {
      if (bodies.has(m[1]) && !seen.has(m[1])) stack.push(m[1]);
    }
  }
  return { seen, bodies };
}

/**
 * 校验「entry 的绑定引用集是连续前缀」，返回 { ok, perGroup }。
 *
 * 判据（与 D-14 真机约束、`wgsl_lint` 规则 G2 同口径）：对每个用到的 group，
 * 被引用的 binding 号去重升序后必须恰为 `[0, 1, ..., n-1]`。
 */
function checkBindingPrefix(src, entry) {
  const decls = wgslBindingDecls(src);
  const { seen, bodies } = entryReachableFns(src, entry);
  let blob = "";
  for (const name of seen) {
    const body = bodies.get(name);
    if (body !== undefined) blob += body;
  }
  const perGroup = new Map();
  for (const d of decls) {
    if (!new RegExp(`\\b${d.name}\\b`).test(blob)) continue;
    if (!perGroup.has(d.group)) perGroup.set(d.group, []);
    perGroup.get(d.group).push(d.binding);
  }
  const report = [];
  let ok = true;
  for (const g of [...perGroup.keys()].sort((a, b) => a - b)) {
    const used = [...new Set(perGroup.get(g))].sort((a, b) => a - b);
    const isPrefix = used.length > 0 && used.every((b, k) => b === k);
    if (!isPrefix) ok = false;
    report.push({ group: g, used, isPrefix });
  }
  return { ok, perGroup: report, declCount: decls.length };
}

// ---------------------------------------------------------------------------
// 库件配方：合成入口 + 真实绑定缓冲
// ---------------------------------------------------------------------------

/**
 * 每个库件的合成入口。
 *
 * 【设计契约（三条，缺一即失去门禁意义）】
 *  1. 入口必须**真的调用**库件函数——否则"库件能否被真机接受"这件事没被测到。
 *     注：WGSL 的 `createShaderModule` 是**全模块验证**（不像 C++ 链接器只实例化
 *     用到的），所以"库件全文进模块"已能抓住函数体错；入口调用是为了额外覆盖
 *     「绑定布局」与「真机能跑」两件事。
 *  2. 入口必须**真的读**每个声明出来的绑定，否则 `layout:"auto"` 会把未引用的
 *     binding 从布局里剔掉，`createBindGroup` 就不再校验它 —— 那等于没测。
 *     `rt_shade.wgsl` 的 `scene_buf` 库件内无函数消费，故入口显式读一次。
 *  3. 所有入参为**编译期常量**（由 `gid.x` 派生），全链路确定性：同一份代码
 *     每次跑出同一批有限数，便于回归比对。
 */
const RECIPES = {
  // ---- rt_common.wgsl：纯常量 + 纯函数，无任何绑定 ----
  "rt_common.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = f32(gid.x) * 0.25 + 0.1;
  let a = v3(i, 1.0 - i, 0.5 + i);
  let b = v3(0.3, -0.2, 0.9);
  var acc = vdot(a, b);
  acc = acc + luminance(hadamard(a, b));
  acc = acc + clampf(acc, -1.0, 1.0);
  acc = acc + quarter_sin(i) + quarter_cos(i);
  let cs = cos_sin(i * 2.0);
  acc = acc + cs.x + cs.y + det_tan(i * 0.5);
  let m = onb(a);
  acc = acc + vdot(m[0], b) + vdot(m[1], b) + vdot(m[2], b);
  let cd = concentric_disk(i * 0.3, 0.6);
  acc = acc + cd.x + cd.y + luminance(cosine_hemisphere(0.3, 0.7));
  acc = acc + luminance(cone_dir(b, 0.5, 0.4, 0.6));
  acc = acc + f32(bounce_dim(1u, SLOT_BSDF)) * 1e-6;
  acc = acc + PI * 1e-6 + FRAC_1_PI * 1e-6 + T_MIN * 1e-3 + SURFACE_EPS * 1e-3
            + MISS_T * 1e-30;
  lib_out[gid.x] = acc;
}
`,
    buffers: [{ group: 0, binding: 0, kind: "out" }],
  },

  // ---- sim_common.wgsl：xpbd_distance_project 形参是 ptr<function,...> ----
  "sim_common.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = f32(gid.x) * 0.1 + 0.05;
  let p = vec3<f32>(i, 1.0 - i, 0.5 + i);
  let h = 0.25;
  var acc = f32(cell_floor(p.y, h)) * 1e-3;
  acc = acc + f32(cell_key(p, h)) * 1e-9;
  acc = acc + poly6_2d(0.1 + i * 0.01, h);
  acc = acc + spiky_grad_scale(0.05, h) * 1e-3;
  acc = acc + poly6_3d(0.08, h) * 1e-3;
  acc = acc + POLY6_3D_COEF * 1e-6;
  // 【ptr<function,...> 形参】必须在**函数作用域**声明 var 再取地址；
  // 模块级 var<private> 虽也可取址，但那样就测不到"函数指针形参"这条路径。
  var pa = vec3<f32>(0.0, 0.0, 0.0);
  var pb = vec3<f32>(1.0, 0.0, 0.0);
  var lambda = 0.0;
  let c = xpbd_distance_project(&pa, &pb, 1.0, 1.0, 1.0, 0.0, 0.016, &lambda);
  acc = acc + c * 1e-3 + lambda * 1e-3 + vdot(pa, pb) * 1e-3;
  acc = acc + PI * 1e-6;
  lib_out[gid.x] = acc;
}
`,
    buffers: [{ group: 0, binding: 0, kind: "out" }],
  },

  // ---- rt_bvh.wgsl：bvh_intersect 形参是 ptr<storage, array<T>, read> ----
  "rt_bvh.wgsl": {
    outBinding: 2,
    extraWgsl: `
@group(0) @binding(0) var<storage, read> lib_nodes: array<BvhNode>;
@group(0) @binding(1) var<storage, read> lib_tris: array<Tri>;
@group(0) @binding(2) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = vec3<f32>(0.0, 0.0, -2.0);
  // 【为什么 rd 各分量都非 0】bvh_intersect 内部算 inv = 1/rd；若某分量为 0，
  // inv 该分量为 inf，随后 (bmin - ro) * inf 可能产出 NaN —— 那是**输入**造成的
  // 垃圾值，不是库件缺陷，会污染"输出全为有限数"这条断言。故用非 0 方向。
  let d = normalize(vec3<f32>(f32(gid.x) * 0.1 - 0.35, 0.05 + f32(gid.x) * 0.01, 1.0));
  let hit = bvh_intersect(&lib_nodes, &lib_tris, arrayLength(&lib_nodes), o, d, T_MIN, 1e30);
  let leaf = tri_ray_hit(lib_tris[0], o, d, T_MIN, 1e30);
  let ab = aabb_ray(vec3<f32>(-1.0), vec3<f32>(1.0), o, vec3<f32>(1.0), T_MIN, 1e30);
  lib_out[gid.x] = hit.x * 1e-3 + f32(hit.y) * 1e-9
                  + leaf * 1e-3 + ab.x * 1e-3 + ab.y * 1e-3;
}
`,
    buffers: [
      { group: 0, binding: 0, kind: "bvhNodes" },
      { group: 0, binding: 1, kind: "bvhTris" },
      { group: 0, binding: 2, kind: "out" },
    ],
  },

  // ---- rt_scene.wgsl：自带 @group(2) @binding(0..4) ----
  "rt_scene.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;
// @group(2) @binding(0..4)（scene_params / materials / prim_mat / tris / lights）
// 由 rt_scene.wgsl 自带，合成入口只负责把它们读一遍。

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = f32(gid.x) * 0.1 + 0.05;
  let o = vec3<f32>(0.0, 0.0, -2.0);
  let d = normalize(vec3<f32>(f - 0.1, 0.05 + f, 1.0));
  let n = vec3<f32>(0.0, 0.0, 1.0);
  let mat = load_material(0u);
  let hit = scene_intersect(o, d, 1e30, 0u, 0u);
  let vis = shadow_ray(o, d, 1e30);
  let ls = sample_light(0u, vec3<f32>(0.0, 0.0, 0.0), f, 0.5);
  let bs = eval_bsdf(mat, -d, d, n);
  var acc = luminance(ambient_background());
  acc = acc + mat.roughness + f32(load_light_count());
  acc = acc + f32(sample_index_seed(0u, 1u, SLOT_BSDF)) * 1e-9;
  acc = acc + luminance(clamp_indirect(vec3<f32>(2.0, 3.0, 4.0), 4u));
  acc = acc + luminance(bs) + vis + ls.dist * 1e-3 + ls.pdf_w * 1e-3;
  acc = acc + luminance(ls.radiance) + vdot(ls.wi, d);
  acc = acc + nee_mis_weight(0u, ls.pdf_w, 0.5);
  acc = acc + emissive_weight(0u, ls.pdf_w, 0u, o);
  acc = acc + hit.t * 1e-3 + luminance(mat.emissive) + emissive_radiance(0u).x;
  acc = acc + luminance(schlick_f0(mat, f)) + pow5_local(f);
  acc = acc + select(0.0, 1.0, should_survive(vec3<f32>(1.0), 7u, 3u, 4u));
  acc = acc + rr_scale(vec3<f32>(0.5, 0.5, 0.5));
  acc = acc + luminance(mnee_caustics(o, n, -d, vec3<f32>(1.0), 1u)) + PI * 1e-6;
  lib_out[gid.x] = acc;
}
`,
    buffers: [
      { group: 0, binding: 0, kind: "out" },
      // 【为什么必须 uniform:true】rt_scene.wgsl 把 scene_params 声明为
      // `@group(2) @binding(0) var<uniform> scene_params: SceneParams`（uniform 地址
      // 空间）。若这里按 storage 建缓冲，真机 CreateBindGroup 会报
      //   "Binding usage (...Storage) doesn't match expected usage (Uniform)"
      // → 整个 group 2 变成 Invalid BindGroup → SetBindGroup(2,…) 失败 → submit 无效。
      // 这正是本波修复的真机缺陷；`kind` 只决定**内容**，`uniform` 决定**用法**。
      { group: 2, binding: 0, kind: "sceneParams", uniform: true },
      { group: 2, binding: 1, kind: "materials" },
      { group: 2, binding: 2, kind: "primMat" },
      { group: 2, binding: 3, kind: "sceneTris" },
      { group: 2, binding: 4, kind: "lights" },
    ],
  },

  // ---- rt_shade.wgsl：自带 @group(1) @binding(0..1) + 经 include 带入 group 2 ----
  "rt_shade.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;
// @group(1) sobol_dir / scene_buf 与 @group(2) 场景绑定分别由 rt_shade.wgsl
// 与 rt_scene.wgsl（//!include 带入）声明。

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = f32(gid.x) * 0.05 + 0.05;
  let n = vec3<f32>(0.0, 0.0, 1.0);
  let w_in = normalize(vec3<f32>(0.2, 0.1, -1.0));
  let o = -w_in;
  let bd = bounce_dim(1u, SLOT_LIGHT);
  let si = sample_index(u32(gid.x), 0u, 1u, 0u);
  let s1 = sample_1d(si, bounce_dim(1u, SLOT_BSDF));
  let s2 = sample_2d(si, bd);
  let aux = aux_1d(si, SLOT_AUX, 1u);
  // 【为什么显式读 scene_buf】该绑定在 rt_shade.wgsl 里声明但**库件内无函数消费**。
  // 若入口也不读它，layout:"auto" 会把它从绑定布局里剔掉，createBindGroup 便不再
  // 校验它 —— 声明与实际用法脱节就查不出来。故显式读一次并计入结果。
  let sb = f32(scene_buf[0]) * 1e-9;
  let bs = bsdf_sample(0u, w_in, n, 1u);
  let nee = nee_surface(vec3<f32>(0.0, 0.0, 0.0), n, w_in, vec3<f32>(1.0), 1u, 0u);
  let h = sample_vndf(o, 0.25, 0.3, 0.7);
  var acc = s1 + s2.x + s2.y + aux + sb;
  acc = acc + f32(si) * 1e-9 + power_heuristic(s1, s2.x);
  acc = acc + ggx_d(0.25, o.z) + smith_g1(0.25, o.z) + pow5(f);
  acc = acc + luminance(schlick(vec3<f32>(0.04), f));
  acc = acc + vdot(refract_dir(w_in, n, 0.5), w_in) + vdot(h, h);
  acc = acc + luminance(bs.weight) + bs.pdf * 1e-3;
  acc = acc + f32(bs.specular) * 1e-3 + f32(bs.entered) * 1e-3;
  acc = acc + vdot(bs.dir, n) + luminance(nee) + f32(bd) * 1e-6;
  lib_out[gid.x] = acc;
}
`,
    buffers: [
      { group: 0, binding: 0, kind: "out" },
      { group: 1, binding: 0, kind: "sobolDir" },
      { group: 1, binding: 1, kind: "sceneBuf" },
      // group 2 的 scene_params 同样经 //!include rt_scene 带入，声明为 var<uniform>。
      { group: 2, binding: 0, kind: "sceneParams", uniform: true },
      { group: 2, binding: 1, kind: "materials" },
      { group: 2, binding: 2, kind: "primMat" },
      { group: 2, binding: 3, kind: "sceneTris" },
      { group: 2, binding: 4, kind: "lights" },
    ],
  },

  // ---- hair_scatter.wgsl：上一波事故现场（a_r/a_tt/a_trt 形参实参不匹配）----
  "hair_scatter.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ti = 0.35 + f32(gid.x) * 0.01;
  let pi = 0.2;
  let to = 0.5 + f32(gid.x) * 0.02;
  let po = 0.9;
  let bm = 0.12;
  let k = 3.0;
  let eta = 1.55;
  let sa = 0.6;
  let rad = 2e-5;
  let sp = hair_scatter_s_precise(ti, pi, to, po, bm, k, eta, sa, rad);
  let ss = hair_scatter_s(ti, to, po - pi, bm, k, hair_f0(eta), 0.9, 0.8);
  var acc = hair_scatter_total(sp) + hair_scatter_total(ss);
  acc = acc + marschner_m(ti, to, bm) + marschner_n(po - pi, k);
  acc = acc + a_r(0.0469) + a_tt(0.0469, 0.9) + a_trt(0.0469, 0.8);
  acc = acc + schlick_fresnel(hair_f0(eta), 0.8) + clamp01(0.5);
  acc = acc + det_exp(0.5) + det_exp(-1.5) + det_exp_neg(0.3);
  acc = acc + bessel_i0(0.5) + bessel_i0(5.0) * 1e-3;
  acc = acc + det_asin(0.3) + det_atan(0.7) + det_atan_unit(0.2) + det_atan_small(0.1);
  acc = acc + hair_phi_p(0.3, 0u) + hair_phi_p(0.3, 1u);
  acc = acc + hair_alpha_p(ti, eta, 0u) + hair_alpha_p(ti, eta, 1u) + hair_alpha_p(ti, eta, 2u);
  acc = acc + sp.x * 1e-3 + ss.y * 1e-3 + PI * 1e-6;
  lib_out[gid.x] = acc;
}
`,
    buffers: [{ group: 0, binding: 0, kind: "out" }],
  },
};

/**
 * 每份库件注入的"确定编译级缺陷"，用于**自证非恒真**。
 *
 * 【为什么这是必须的】"6 份库件全部 0 error 通过"本身**不能**证明这条链路有效
 * —— 万一 `getCompilationInfo()` 恒返回空、或 `createShaderModule` 根本没被调用，
 * 也会全绿。故对每份库件注入一处**编译器必然拒绝**的缺陷（形参/实参个数不匹配），
 * 并要求真编译器报错。任何一份注入后仍编译通过 → 判红。
 *
 * 缺陷形态刻意选成"与上一波事故同类"（签名与调用点不一致），而非随手写个语法错：
 * 语法错太容易抓，抓到了也不能说明"签名错"这类缺陷被覆盖。
 */
const INJECTIONS = {
  "rt_common.wgsl": {
    mustApply: true,
    apply: (s) => s.replace(
      "fn luminance(c: vec3<f32>) -> f32 {",
      "fn luminance(c: vec3<f32>, extra: f32) -> f32 {"),
    expect: /luminance/,
  },
  "sim_common.wgsl": {
    mustApply: true,
    apply: (s) => s.replace(
      "fn poly6_2d(r: f32, h: f32) -> f32 {",
      "fn poly6_2d(r: f32, h: f32, extra: f32) -> f32 {"),
    expect: /poly6_2d/,
  },
  "rt_bvh.wgsl": {
    mustApply: true,
    // 调用点少传一个实参（bvh_intersect 内部对 tri_ray_hit 的调用）。
    apply: (s) => s.replace(
      "let ht = tri_ray_hit((*tris)[slot], ro, rd, t_min, best_t);",
      "let ht = tri_ray_hit((*tris)[slot], ro, rd, t_min);"),
    expect: /tri_ray_hit/,
  },
  "rt_scene.wgsl": {
    mustApply: true,
    // builtin 实参个数错：min(a, b) → min(a, b, c)。
    apply: (s) => s.replace(
      "let n = min(scene_params.prim_count, arrayLength(&tris));",
      "let n = min(scene_params.prim_count, arrayLength(&tris), 1u);"),
    expect: /min|argument/i,
  },
  "rt_shade.wgsl": {
    mustApply: true,
    apply: (s) => s.replace(
      "fn power_heuristic(a: f32, b: f32) -> f32 {",
      "fn power_heuristic(a: f32, b: f32, extra: f32) -> f32 {"),
    expect: /power_heuristic/,
  },
  "hair_scatter.wgsl": {
    mustApply: true,
    // 【上一波事故的原始形态】a_r 声明 2 个形参却按 1 个实参调用。
    apply: (s) => s.replace(
      "fn a_r(f0: f32) -> f32 {",
      "fn a_r(f0: f32, absorb_extra: f32) -> f32 {"),
    expect: /a_r/,
  },
};

/**
 * 有合成入口配方的库件文件集合（升序）。
 *
 * 【为什么导出给门禁】`wgsl_lint.mjs` 规则 F18 要断言"清单里的每份库件都**真的**
 * 有合成入口"，否则清单可以登记一份根本没编的库件——那就是**新的编译盲区**，
 * 比不登记更坏（它看起来被管住了）。静态门禁不能 import 本文件的 CLI 部分
 * （那会起真 GPU），故把配方键集合作为**零副作用的模块级常量**导出。
 * 本文件的 CLI 段由 `isMain` 守卫，import 本模块不会触发任何 GPU 行为。
 */
export const RECIPE_FILES = Object.freeze(Object.keys(RECIPES).sort());

/**
 * 有"注入缺陷"条目的库件文件集合（升序）——自证非恒真的覆盖面。
 *
 * 【为什么必须覆盖全部】只对部分库件自证，等于承认"对剩下的库件，这条链路是否
 * 有效我不知道"。那等于把未知当已知。F18 断言两者相等。
 */
export const INJECTION_FILES = Object.freeze(Object.keys(INJECTIONS).sort());

/**
 * 每份库件的**绑定配方**（供 F19 做「地址空间 ↔ 缓冲用法」静态一致性校验）。
 *
 * 【为什么必须导出它，而不是让门禁自己再解析本文件的 RECIPES 对象】本文件顶层
 * 的 RECIPES 是**代码**（含 `extraWgsl` 模板串）；门禁 `wgsl_lint.mjs` 若要
 * 判「声明了 var<uniform> 的绑定，配方里是不是按 UNIFORM 建的」，就必须知道
 * 配方的 (group, binding) 用法。导出成**纯数据**后：
 *   * 门禁无需 import GPU 逻辑（本文件 CLI 段有 isMain 守卫，import 安全）；
 *   * 反向用例可**注入一份被改坏的配方**（如删掉 `uniform: true`）来证明 F19
 *     非恒真——这正是本次真机事故（rt_scene/rt_shade 的 uniform 绑定按 storage 建）
 *     的静态复现。
 *
 * 【字段契约】
 *   extraWgsl  合成入口源码。判「展开后的声明集」时必须把它一并算进去——
 *              合成入口自己 `@group(0) @binding(0) var<storage, read_write> lib_out`
 *              也是被真编译的声明，只扫库件本体会把 group 0 误判成「幽灵绑定」。
 *   buffers    配方实际建的缓冲 (group, binding, uniform)。`uniform:true` ⇒
 *              用 `GPUBufferUsage.UNIFORM` 建；否则用 `STORAGE`。
 */
export const RECIPE_BINDINGS = Object.freeze(
  Object.fromEntries(
    Object.entries(RECIPES).map(([file, r]) => [
      file,
      Object.freeze({
        extraWgsl: r.extraWgsl,
        buffers: Object.freeze(
          r.buffers.map((b) => Object.freeze({
            group: b.group,
            binding: b.binding,
            // 【用法由声明决定，不由内容种类决定】这里的 `uniform` 标志**就是**
            // 真机建缓冲时的用法来源（见 runOnGpu 里 `b.uniform ? UNIFORM : STORAGE`）。
            // F19 拿它与 WGSL 地址空间比对，不一致即判红。
            uniform: b.uniform === true,
          })),
        ),
      }),
    ]),
  ),
);

// ---------------------------------------------------------------------------
// 绑定缓冲的真实数据（WGSL 布局逐字节对齐；数值全为常量 → 确定性）
// ---------------------------------------------------------------------------

/** 构造一个 u32 数组缓冲。 */
function u32buf(values) {
  const ab = new ArrayBuffer(values.length * 4);
  const dv = new DataView(ab);
  values.forEach((v, i) => dv.setUint32(i * 4, v >>> 0, true));
  return ab;
}

/**
 * 各类绑定缓冲的字节内容。
 *
 * 【布局必须逐字节对齐，否则真机读到的是错位的值】本函数里的偏移量与
 * `shaders/*.wgsl` 的 struct 声明一一对应（vec3 在 WGSL 里 align 16 / size 12，
 * 故 `Tri` = a@0, _p0@12, b@16, _p1@28, c@32, _p2@44 → size 48）。
 */
function buildBuffer(kind) {
  switch (kind) {
    case "out":
      // storage 缓冲在 WebGPU 里必须已初始化；全 0 即可（入口会覆写前 DISPATCH 个）。
      return new ArrayBuffer(OUT_FLOATS * 4);
    // BvhNode: bmin@0 bmax@16 start@28 count@32 right_child@36 _pad@40 → size 48
    //
    // 【踩过的坑：vec3 在 WGSL 里 align 是 16、不是 4】曾按"packed"直觉写成
    // bmax@12 / start@24 / count@28，结果真机上：着色器按 WGSL 规则在**偏移 32**
    // 读 count，而那里放的是 right_child = 0 → 节点被当成**内部节点** → 走
    // else 分支不停把子节点压栈（OOB 读又被钳回节点 0，count 仍是 0）→
    // **死循环**，dispatch 永不返回。
    //
    // 这一条是本脚本存在的价值的最直接证明：布局错位在纯静态门禁下**完全不可见**
    // （正则不看字节偏移），只有真 dispatch 才会暴露成"挂死"。
    case "bvhNodes": {
      const dv = new DataView(new ArrayBuffer(48));
      [-1, -1, -1].forEach((v, i) => dv.setFloat32(i * 4, v, true));   // bmin@0
      [1, 1, 1].forEach((v, i) => dv.setFloat32(16 + i * 4, v, true)); // bmax@16
      dv.setUint32(28, 0, true);   // start
      dv.setUint32(32, 1, true);   // count = 1（叶子，含 1 个三角形）
      dv.setUint32(36, 0, true);   // right_child
      dv.setUint32(40, 0, true);   // _pad
      return dv.buffer;
    }
    // Tri: a@0 _p0@12 b@16 _p1@28 c@32 _p2@44 → size 48（z=0 平面上的三角形）
    case "bvhTris":
    case "sceneTris": {
      const dv = new DataView(new ArrayBuffer(48));
      const put = (off, v) => dv.setFloat32(off, v, true);
      put(0, -1); put(4, -1); put(8, 0); put(12, 0);
      put(16, 1); put(20, -1); put(24, 0); put(28, 0);
      put(32, 0); put(36, 1); put(40, 0); put(44, 0);
      return dv.buffer;
    }
    // SceneParams: 4×u32 + 4×f32 → size 32
    case "sceneParams": {
      const dv = new DataView(new ArrayBuffer(32));
      dv.setUint32(0, 1, true);    // material_count
      dv.setUint32(4, 1, true);    // light_count
      dv.setUint32(8, 1, true);    // prim_count
      dv.setUint32(12, 7, true);   // node_id
      dv.setFloat32(16, 0.1, true); // ambient_r
      dv.setFloat32(20, 0.2, true); // ambient_g
      dv.setFloat32(24, 0.3, true); // ambient_b
      dv.setFloat32(28, 0.0, true); // _pad
      return dv.buffer;
    }
    // Material: base_color@0 roughness@12 f0@16 metalness@28 transmission@32
    //           ior@36 emissive@40 _pad0@52 → size 64
    case "materials": {
      const dv = new DataView(new ArrayBuffer(64));
      const f = (off, v) => dv.setFloat32(off, v, true);
      f(0, 0.8); f(4, 0.8); f(8, 0.8); f(12, 0.5);          // base_color, roughness
      f(16, 0.04); f(20, 0.04); f(24, 0.04); f(28, 0.0);     // f0, metalness
      f(32, 0.0); f(36, 1.5);                                // transmission, ior
      f(40, 0.0); f(44, 0.0); f(48, 0.0); f(52, 0.0);        // emissive, _pad0
      return dv.buffer;
    }
    // Light: position@0 radius@12 color@16 _pad@28 → size 32
    case "lights": {
      const dv = new DataView(new ArrayBuffer(32));
      dv.setFloat32(0, 0.0, true);
      dv.setFloat32(4, 0.0, true);
      dv.setFloat32(8, 5.0, true);  // position.z
      dv.setFloat32(12, 0.5, true); // radius
      dv.setFloat32(16, 1.0, true); // color
      dv.setFloat32(20, 1.0, true);
      dv.setFloat32(24, 1.0, true);
      dv.setFloat32(28, 0.0, true); // _pad
      return dv.buffer;
    }
      // sobol_dir: 方向数矩阵。
      // 【尺寸依据，不是拍脑袋】`sobol_u32(dim, index)` 读 `sobol_dir[dim*32 + k]`，
      // k ≤ 31；本链路用到的最大 dim = bounce_dim(1u, SLOT_LIGHT) = 5 + 0*6 + 2 = 7
      // → 最大下标 7*32 + 31 = 255。故 256 个 u32（1 KiB）已**严格足够**。
      // 给更大值只会白白多占 SwiftShader 的显存与传输时间。
      case "sobolDir": {
        const n = 256;
        const vals = new Array(n);
        for (let i = 0; i < n; i += 1) vals[i] = (Math.imul(i + 1, 2654435761) >>> 8) >>> 0;
        return u32buf(vals);
      }
    case "sceneBuf":
      return u32buf([7, 0, 0, 0]);
    case "primMat":
      return u32buf([0, 0, 0, 0]);
    default:
      throw new Error(`buildBuffer: 未知缓冲种类 ${kind}`);
  }
}

// ---------------------------------------------------------------------------
// //!include 展开（与 Rust 侧 `shaders::expand` 同语义）
// ---------------------------------------------------------------------------

/**
 * 展开 `//!include <file>`（递归、去重）。
 *
 * 【为什么只展开一份、不重复拼 include 目标】WGSL **不允许顶层重声明**。
 * 例如 `hair_scatter.wgsl` 自带 `//!include rt_common.wgsl`，若再单独拼一份
 * `rt_common.wgsl`，`PI` / `cos_sin` / `vdot` 会重复声明 →
 * `createShaderModule` 报 redeclaration。展开器本身是递归的，一份就够。
 */
function expandIncludes(label, seen = new Set()) {
  if (seen.has(label)) return "";
  seen.add(label);
  const p = join(SHADER_DIR, label);
  if (!existsSync(p)) return "";
  return expandSource(readFileSync(p, "utf8"), label, seen);
}

/** 按 `//!include` 展开一份源码；`override` 非空时用 override 代替磁盘内容（注入用）。 */
function expandSource(raw, label, seen) {
  let out = "";
  for (const line of raw.split("\n")) {
    const m = /^\s*\/\/!include\s+(\S+)\s*$/.exec(line);
    if (m) {
      if (seen.has(m[1])) { out += "\n"; continue; }
      seen.add(m[1]);
      const p = join(SHADER_DIR, m[1]);
      const child = existsSync(p) ? readFileSync(p, "utf8") : "";
      out += expandSource(child, m[1], seen) + "\n";
    } else {
      out += line + "\n";
    }
  }
  void label;
  return out;
}

/** 读库件**未展开**的原文（注入缺陷要在原文上做，且原文才是"库件本体"）。 */
function readLibraryRaw(file) {
  const p = join(SHADER_DIR, file);
  if (!existsSync(p)) throw new Error(`库件文件不存在：${p}`);
  return readFileSync(p, "utf8");
}

/** 组装某库件的完整合成 shader 源码。 */
function buildLibShader(item) {
  const recipe = RECIPES[item.file];
  if (!recipe) throw new Error(`清单登记了 ${item.file}，但 RECIPES 里没有它的合成入口`);
  const expanded = expandIncludes(item.file);
  if (expanded.trim().length === 0) {
    throw new Error(`${item.file} 展开后为空——文件缺失或路径不对`);
  }
  return `${expanded}\n${recipe.extraWgsl}`;
}

// ---------------------------------------------------------------------------
// 前置检查（与 realgpu_check.mjs / hair_parity_gpu.mjs 同纪律）
// ---------------------------------------------------------------------------

function die(msg, code = 1) {
  console.error(`\n!! wgsl_lib_compile: FATAL — ${msg}`);
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
  try {
    const mod = await import("playwright");
    if (mod?.chromium) return { chromium: mod.chromium, spec: "playwright" };
    tried.push("playwright（可解析但无 chromium 导出）");
  } catch (e) {
    tried.push(`playwright（${e.code || e.message}）`);
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

// ---------------------------------------------------------------------------
// 真跑：编译 → 建 pipeline → 建 bind group → dispatch → 回读
// ---------------------------------------------------------------------------

/**
 * 在真 WebGPU 设备上编译并执行每个库件的合成 shader，并跑非恒真自证。
 *
 * @param {Array<object>} jobs 正向 job：{ file, code, entry, groups, dispatch, buffers }
 * @param {Array<object>} injectedJobs 注入 job：{ file, code, applied, expectSrc }
 * @returns {Promise<object>} { ok, adapter, results, injected }
 */
async function runOnGpu(jobs, injectedJobs) {
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

    // 【为什么要这条进度通道】SwiftShader 上某些库件的编译/dispatch 可能异常慢，
    // 而整条链路的失败模式是"跑到一半被外层超时杀掉"，此时**看不到卡在哪一步**。
    // 故把页内进度实时打回 Node 侧 stdout：定位卡点靠它，而不是靠猜。
    await page.exposeBinding("__libProgress", (_source, msg) => {
      process.stdout.write(`   · ${msg}\n`);
    });
    // 页面只需一个安全上下文（http 即是），用来跑 WebGPU。
    await page.goto(`${origin}/index.html`, { waitUntil: "domcontentloaded", timeout: 60000 });

    // 之所以在**页面内**建 device/pipeline/buffer：WebGPU 只存在于浏览器上下文。
    const result = await page.evaluate(async ({ jobs: jsJobs, injectedJobs: jsInjected }) => {
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

      // WebGPU 异步错误捕获。
      //
      // 【为什么不能只靠它判红】注入自证阶段那批 shader module 是**故意编译不过**的，
      // WGSL 的编译错误既进 `getCompilationInfo()`，也异步抛 `uncapturederror`。
      // 若一律当致命错误，"注入后仍报错"就会被误判成"链路失败"——恰好把
      // **证明链路有效**的证据当成失败。故这里用 `expectFailures` 开关：注入阶段
      // 期间的 uncaptured error 是**预期产物**，不计失败；它是否真的发生由下面的
      // error scope 精确断言（要求"必须发生"）。
      const gpuErrs = [];
      let expectFailures = false;
      device.addEventListener?.("uncapturederror", (ev) => {
        if (expectFailures) return;
        gpuErrs.push(String(ev.error?.message || ev.error));
      });

      /**
       * 编译一份 shader。
       *
       * @param {string} code WGSL 全文
       * @param {boolean} mustFail true = 期望被驱动拒绝（注入自证用）
       * @returns {Promise<object>} errors / warnings / infos / scopeError
       */
      const compile = async (code, mustFail) => {
        // 用 error scope 精确圈定"这一次的验证错误"，不污染全局异步错误通道。
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code });
        let info = null;
        let fatal = null;
        if (typeof module.getCompilationInfo !== "function") {
          fatal = "该实现不提供 getCompilationInfo()——无法断言 0 error（拒绝降级为'假定通过'）";
        } else {
          info = await module.getCompilationInfo();
        }
        const scopeError = await device.popErrorScope();
        if (fatal) return { fatal };
        const messages = info?.messages || [];
        const pick = (t) => messages.filter((m) => m.type === t)
          .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
        return {
          errors: pick("error"),
          warnings: pick("warning"),
          infos: pick("info"),
          scopeError: scopeError ? String(scopeError.message || scopeError) : "",
          // 「被驱动拒绝」= 拿到 error 级诊断，**或**验证层直接报错。两者都算。
          rejected: pick("error").length > 0 || !!scopeError,
          // 该不该被拒绝（供调用方核对期望）
          mustFail: !!mustFail,
        };
      };

      const results = [];
      for (const job of jsJobs) {
        const rec = { file: job.file, entry: job.entry, shaderBytes: job.code.length };
        await window.__libProgress(`${job.file}：createShaderModule（${job.code.length} 字节）…`);
        const c = await compile(job.code, false);
        await window.__libProgress(`${job.file}：编译诊断 ${c.errors.length} error / ${c.warnings.length} warning`);
        if (c.fatal) return { ok: false, why: `${job.file}: ${c.fatal}` };
        rec.errors = c.errors;
        rec.warnings = c.warnings;
        rec.infos = c.infos;
        // 【正向 job 必须连验证层都干净】只看 getCompilationInfo 不够：部分实现
        // 会把绑定/布局类的验证错误只报在 error scope 上。若此处放过，就等于
        // "编译过但绑定是坏的"被当成通过——正是 D-14 那类真机事故的形态。
        if (c.errors.length || c.scopeError) {
          rec.ok = false;
          rec.stage = "compile";
          if (c.scopeError) rec.errors = [...rec.errors, `validation: ${c.scopeError}`];
          results.push(rec);
          continue;
        }

        try {
          const module = device.createShaderModule({ code: job.code });
          await window.__libProgress(`${job.file}：createComputePipeline…`);
          const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: job.entry } });

          const created = [];
          const entriesByGroup = new Map();
          for (const b of job.buffers) {
            // 【用法由 WGSL 地址空间决定，不由缓冲内容种类决定】scene_params 声明为
            // `var<uniform>` → BufferBindingType::Uniform；若按 storage 建会报
            // "Binding usage ... doesn't match expected usage (Uniform)"，
            // 使整个 group 失效（SetBindGroup(2,…) 失败）。故这里读 job 里显式
            // 携带的 `uniform` 标志，而非靠 kind 猜。
            const usage = b.uniform
              ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
              : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
            const buf = device.createBuffer({ size: b.bytes, usage });
            device.queue.writeBuffer(buf, 0, new Uint8Array(b.data));
            created.push({ ...b, buf });
            if (!entriesByGroup.has(b.group)) entriesByGroup.set(b.group, []);
            entriesByGroup.get(b.group).push({ binding: b.binding, resource: { buffer: buf } });
          }

          const groups = [];
          for (const g of job.groups) {
            const layout = pipe.getBindGroupLayout(g);
            groups.push({ g, bg: device.createBindGroup({ layout, entries: entriesByGroup.get(g) || [] }) });
          }

          // 【为什么要在 dispatch 段再开一层 validation error scope】只靠全局
          // `uncapturederror` 会把"某份库件的 submit 无效"升级成整条链路 FATAL，
          // 既看不出是哪一份、也让 host_test 只能看到一个笼统失败。更糟的是：
          // 无效 submit 时 `onSubmittedWorkDone()` 仍会 resolve，回读拿到的是
          // **未被 dispatch 覆写的初始零值**——"输出全为有限数"于是**假绿**。
          // 这正是本次真机缺陷（rt_scene / rt_shade 的 storage-buffer-当-uniform 用）
          // 的暴露方式。故在此精确圈定本 job 的验证错误，并作为独立断言上报。
          device.pushErrorScope("validation");
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe);
          for (const { g, bg } of groups) pass.setBindGroup(g, bg);
          pass.dispatchWorkgroups(job.dispatch);
          pass.end();
          device.queue.submit([enc.finish()]);
          await window.__libProgress(`${job.file}：dispatch ${job.dispatch} 个 workgroup，等待 GPU…`);
          await device.queue.onSubmittedWorkDone();
          // 【真 dispatch 的硬判据】submit 之后必须有 0 个验证错误。非空即说明
          // 这次 dispatch 根本没跑（绑定/用法/布局非法），"dispatched>0" 是假绿。
          const submitErr = await device.popErrorScope();
          rec.submitError = submitErr ? String(submitErr.message || submitErr) : "";

          // 回读合成输出缓冲（dispatch 个 float），校验全为有限数。
          const outBuf = created.find((b) => b.kind === "out").buf;
          const readFloats = job.dispatch * 4;
          const readBuf = device.createBuffer({
            size: readFloats,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          const enc2 = device.createCommandEncoder();
          enc2.copyBufferToBuffer(outBuf, 0, readBuf, 0, readFloats);
          device.queue.submit([enc2.finish()]);
          await readBuf.mapAsync(GPUMapMode.READ);
          const got = Array.from(new Float32Array(readBuf.getMappedRange().slice(0)));
          readBuf.unmap();

          const nonFinite = [];
          for (let i = 0; i < got.length; i += 1) {
            if (!Number.isFinite(got[i])) nonFinite.push({ i, v: String(got[i]) });
          }
          rec.out = got;
          rec.nonFinite = nonFinite;
          rec.dispatched = job.dispatch;
          // 【submit 验证错误必须让本 job 判负】否则"提交无效 → 回读初始零值 →
          // 全是有限数 → 判绿"会重新变成假绿。优先报 submit 阶段。
          if (rec.submitError) {
            rec.ok = false;
            rec.stage = "submit";
            rec.errors = [...(rec.errors || []), `submit 验证错误：${rec.submitError}`];
          } else {
            rec.ok = nonFinite.length === 0;
            rec.stage = rec.ok ? "ok" : "output";
          }
          await window.__libProgress(`${job.file}：回读 ${got.length} 个 float，非有限数 ${nonFinite.length} 个`);
        } catch (e) {
          rec.ok = false;
          rec.stage = "dispatch";
          rec.errors = [...(rec.errors || []), `dispatch 阶段异常：${e.message}`];
        }
        results.push(rec);
      }

      // ---- 非恒真自证：注入缺陷后必须编译失败 ----
      //
      // 【为什么这一段要开 expectFailures】注入的 module 是**故意编译不过**的，
      // 驱动会同时（a）把 error 级诊断放进 getCompilationInfo，（b）异步抛
      // uncapturederror。若把（b）当致命错误，"注入后确实报错了"这条**期望中的**
      // 证据会被反判成"链路坏了"——恰好把证明链路有效的证据当成失败。
      // 故这里开开关让（b）不计失败，改由 error scope 精确断言「必须被拒绝」
      // （rejected）；断言不成立才判红。
      const injected = [];
      expectFailures = true;
      try {
        for (const job of jsInjected) {
          await window.__libProgress(`${job.file}(注入)：createShaderModule，期望被拒绝…`);
          const c = await compile(job.code, true);
          if (c.fatal) return { ok: false, why: `${job.file}(注入): ${c.fatal}` };
          const expect = new RegExp(job.expectSrc);
          // 验证层拒绝也要并进 errors，否则报告里看不到"被驱动拒绝"的确切理由。
          const allErrs = c.scopeError ? [...c.errors, `validation: ${c.scopeError}`] : c.errors;
          const matched = allErrs.some((m) => expect.test(m));
          injected.push({
            file: job.file,
            applied: job.applied,
            errorCount: allErrs.length,
            errors: allErrs,
            matched,
            // 「抓到」= 拿到 error 级诊断**或**验证层直接拒绝。两者都说明真编译器/
            // 真驱动确实拒绝了它，与用哪个通道报出来无关。
            caught: c.rejected,
          });
          await window.__libProgress(`${job.file}(注入)：error ${allErrs.length} 条，命中预期=${matched}`);
        }
      } finally {
        expectFailures = false;
      }

      const adapterInfo = adapter.info || {};
      return {
        ok: true,
        results,
        injected,
        gpuErrs,
        adapter: `${adapterInfo.vendor || "?"}/${adapterInfo.architecture || "?"}/${adapterInfo.description || adapterInfo.device || "?"}`,
      };
    }, { jobs, injectedJobs });

    if (pageErrs.length) { await cleanup(); die(`页面异常：\n${pageErrs.join("\n")}`); }
    if (!result.ok) { await cleanup(); die(result.why); }
    if (result.gpuErrs?.length) {
      await cleanup();
      die(`WebGPU 运行时错误：\n${result.gpuErrs.join("\n")}`);
    }
    await cleanup();
    return result;
  } catch (e) {
    await cleanup();
    die(`GPU 执行异常：${e.stack || e.message}`);
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
//
// 【为什么整段包在 isMain 守卫里】`wgsl_lint.mjs` 规则 F18 需要静态引用本文件的
// `RECIPE_FILES` / `INJECTION_FILES` 来断言"清单里的每份库件都真的有合成入口 +
// 注入自证"。若 CLI 在 import 时就执行，会把**静态门禁拖进真 GPU 链路**
// （CI 无 GPU 时直接不可用，且门禁的失败原因会被环境问题掩盖）。故 import 本
// 模块只得到纯数据，行为只在直接运行时发生。
const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) await main();

/** 主流程（仅直接运行时调用）。 */
async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes("--help")) {
    console.log("wgsl_lib_compile.mjs — 无入口 WGSL 库件真实编译链路\n" +
      "用法：node wgsl_lib_compile.mjs [--json] [--no-prove]\n" +
      "退出码：0 = 全部库件 0 error 编译 + dispatch + 输出有限 + 自证全抓到；1 = 任一失败；2 = 用法错误");
    process.exit(2);
  }
  const AS_JSON = argv.includes("--json");
  const DO_PROVE = !argv.includes("--no-prove");
  // 调试用：--only=rt_bvh.wgsl 只跑一份库件（定位"哪一份把浏览器压垮"时用）。
  // 门禁与 host_test 一律跑全量——**绝不允许**在正式路径里按份跳过（那等于
  // 把"这份库件编不过"藏起来，正是本任务要消灭的那类盲区）。
  const ONLY = (argv.find((a) => a.startsWith("--only=")) || "").slice("--only=".length).trim();
  const manifest = ONLY ? LIB_MANIFEST.filter((e) => e.file === ONLY) : LIB_MANIFEST;
  if (ONLY && manifest.length === 0) {
    die(`--only=${ONLY} 未匹配到清单里的任何库件`);
  }

  /** 组装正向 job + 静态前缀报告 + 注入 job。任何契约不一致直接致命退出。 */
  function buildJobs() {
    const jobs = [];
    const injectedJobs = [];
    const staticReports = [];
    for (const item of manifest) {
      const code = buildLibShader(item);
      const recipe = RECIPES[item.file];
      const pre = checkBindingPrefix(code, item.entry);
      staticReports.push({
        file: item.file,
        entry: item.entry,
        manifestGroups: item.groups,
        declCount: pre.declCount,
        perGroup: pre.perGroup,
        prefixOk: pre.ok,
      });
      if (!pre.ok) {
        throw new Error(`${item.file}：合成入口 ${item.entry} 的绑定引用集不是连续前缀 ` +
          `（${JSON.stringify(pre.perGroup)}）——layout:"auto" 会在真机验证失败`);
      }
      // 清单登记的 group 必须与实际用到的 group 一致（漏登记 → createBindGroup 缺 group）。
      const actual = pre.perGroup.map((g) => g.group);
      const declared = [...item.groups].sort((a, b) => a - b);
      if (JSON.stringify(actual) !== JSON.stringify(declared)) {
        throw new Error(`${item.file}：清单登记 groups=[${declared}] 与源码实际用到 [${actual}] 不一致`);
      }
      if (OUT_FLOATS < DISPATCH) throw new Error(`OUT_FLOATS(${OUT_FLOATS}) < DISPATCH(${DISPATCH})`);
      // 输出缓冲必须落在清单声明的某个 group 上。
      if (recipe.buffers.find((b) => b.kind === "out").group !== declared[0]) {
        throw new Error(`${item.file}：输出缓冲不在 group ${declared[0]}（合成入口的 @group(0)）`);
      }
      jobs.push({
        file: item.file,
        entry: item.entry,
        code,
        groups: declared,
        dispatch: DISPATCH,
        buffers: recipe.buffers.map((b) => {
          const ab = buildBuffer(b.kind);
          return {
            group: b.group,
            binding: b.binding,
            kind: b.kind,
            // 【用法必须来自声明，不能从内容种类推断】WGSL 的地址空间
            // （var<uniform> vs var<storage>）决定真机的 BufferBindingType，
            // 与缓冲"装什么"（kind）无关。scene_params 装的是计数/环境光，
            // 但它声明成 uniform → 必须建 UNIFORM 缓冲。
            uniform: b.uniform === true,
            bytes: ab.byteLength,
            data: Array.from(new Uint8Array(ab)),
          };
        }),
      });

      if (!DO_PROVE) continue;
      const inj = INJECTIONS[item.file];
      if (!inj) throw new Error(`${item.file}：缺少 INJECTIONS 条目（无法自证非恒真）`);
      const raw = readLibraryRaw(item.file);
      const brokenRaw = inj.apply(raw);
      const applied = brokenRaw !== raw;
      if (inj.mustApply && !applied) {
        throw new Error(`${item.file}：注入未生效（源码字面量已变，注入规则失效——门禁会变恒真）`);
      }
      // 在**展开后的**全文里替换（注入点可能在库件本体，也可能在它 include 的子件里）。
      const brokenExpanded = expandSource(brokenRaw, item.file, new Set([item.file]));
      injectedJobs.push({
        file: item.file,
        code: `${brokenExpanded}\n${recipe.extraWgsl}`,
        applied,
        expectSrc: inj.expect.source,
      });
    }
    return { jobs, injectedJobs, staticReports };
  }

  let built;
  try {
    built = buildJobs();
  } catch (e) {
    die(String(e.message || e));
  }

  const { adapter, results, injected } = await runOnGpu(built.jobs, built.injectedJobs);

  const allOk = results.every((r) => r.ok) && injected.every((r) => r.caught && r.applied);
  const warnCount = results.reduce((n, r) => n + r.warnings.length, 0);
  const injErrTotal = injected.reduce((n, r) => n + r.errorCount, 0);

  if (AS_JSON) {
    console.log(JSON.stringify({
      ok: allOk,
      adapter,
      dispatch: DISPATCH,
      warnCount,
      static: built.staticReports,
      results: results.map((r) => ({
        file: r.file, entry: r.entry, ok: r.ok, stage: r.stage,
        shaderBytes: r.shaderBytes, dispatched: r.dispatched ?? 0,
        submitError: r.submitError ?? "",
        errorCount: r.errors.length, errors: r.errors,
        warningCount: r.warnings.length, warnings: r.warnings,
        nonFinite: r.nonFinite ?? [], out: r.out ?? [],
      })),
      injected: injected.map((r) => ({
        file: r.file, applied: r.applied, caught: r.caught, matched: r.matched,
        errorCount: r.errorCount, errors: r.errors,
      })),
    }, null, 2));
  } else {
    console.log(`== 无入口 WGSL 库件真实编译链路（adapter=${adapter}，真 createShaderModule + 真 dispatch）==`);
    console.log(`清单：${manifest.length} 份库件；每份 dispatch ${DISPATCH} 个 workgroup\n`);
    console.log("库件".padEnd(20) + "入口".padEnd(18) + "error".padEnd(8) + "warn".padEnd(7) + "dispatch".padEnd(10) + "输出");
    console.log("-".repeat(92));
    for (const r of results) {
      const outs = (r.out || []).slice(0, 4).map((v) => (Number.isFinite(v) ? v.toExponential(2) : String(v)));
      console.log(
        `${r.ok ? " " : "!"}${r.file.padEnd(19)} ${r.entry.padEnd(17)} ` +
        `${String(r.errors.length).padEnd(7)} ${String(r.warnings.length).padEnd(6)} ` +
        `${r.dispatched}×      [${outs.join(", ")}${(r.out || []).length > 4 ? ", …" : ""}]`);
      for (const e of r.errors) console.log(`      ERROR  ${e}`);
      if (r.submitError) console.log(`      SUBMIT-INVALID  ${r.submitError}`);
      for (const n of r.nonFinite || []) console.log(`      NON-FINITE out[${n.i}] = ${n.v}`);
    }
    console.log("-".repeat(92));
    console.log(`\n非恒真自证（注入一处确定的编译级缺陷，要求真编译器必须报错）：`);
    for (const r of injected) {
      const first = r.errors[0] ? r.errors[0].slice(0, 96) : "（无报错 —— 门禁恒真！）";
      console.log(`  ${r.caught && r.applied ? "ok  " : "FAIL"}  ${r.file.padEnd(19)} ` +
        `注入=${r.applied} error 数=${r.errorCount} 命中预期=${r.matched}\n        ${first}`);
    }
    console.log(`\n共 ${results.length} 份库件，warning ${warnCount} 条（记录不判红），` +
      `自证注入报错合计 ${injErrTotal} 条。`);
    console.log(allOk
      ? "\n==> 全部库件 0 error 通过 + dispatch 成功 + 输出全有限 + 自证全部抓到。"
      : "\n==> FAIL：见上（编译失败 / 输出非有限 / 注入后仍编译通过 = 链路无效）。");
    process.exit(allOk ? 0 : 1);
  }
}
