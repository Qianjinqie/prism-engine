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

  // ---- rt_scene.wgsl：自带 @group(2) @binding(0..4) + b6 spheres / b7 dir_lights ----
  "rt_scene.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;
// @group(2) @binding(0..4)（scene_params / materials / prim_mat / tris / lights）
// 与 b6 spheres / b7 dir_lights（兼容模式专用表）由 rt_scene.wgsl 自带；
// b5 env_blob 由 wavefront 引用方（rt_wavefront.wgsl）声明——本合成入口按
// 同一布局替它声明一次并真读，让引用集成为连续前缀 {0..7}
//（layout:"auto" 会剔除未引用的绑定，缺读即前缀断裂）。
@group(2) @binding(5) var<storage, read> env_blob: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = f32(gid.x) * 0.1 + 0.05;
  let o = vec3<f32>(0.0, 0.0, -2.0);
  let d = vnorm(vec3<f32>(f - 0.1, 0.05 + f, 1.0));
  let n = vec3<f32>(0.0, 0.0, 1.0);
  let mat = load_material(0u);
  let hit = scene_intersect(o, d, 1e30);
  let vis = shadow_ray(o, d, 1e30);
  let bs = eval_bsdf(mat, -d, d, n);
  var acc = luminance(ambient_background());
  acc = acc + mat.roughness + f32(load_light_count());
  acc = acc + luminance(bs) + vis + hit.t * 1e-3 + luminance(mat.emissive);
  acc = acc + luminance(schlick_f0(mat, f)) + pow5_local(f);
  // b5/b6/b7 显式各读一个字：声明与用法一致（auto 布局剔除 = 前缀断裂）。
  acc = acc + env_blob[0] * 1e-9 + spheres[0] * 1e-9 + dir_lights[0] * 1e-9;
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
      { group: 2, binding: 5, kind: "envBlob" },
      { group: 2, binding: 6, kind: "spheresFlat" },
      { group: 2, binding: 7, kind: "dirLightsFlat" },
    ],
  },

  // ---- rt_shade.wgsl：自带 @group(1) @binding(0..1) + 经 include 带入 group 2 ----
  "rt_shade.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;
// @group(1) sobol_dir / scene_buf 与 @group(2) 场景绑定（0..4 + b6/b7）分别由
// rt_shade.wgsl 与 rt_scene.wgsl（//!include 带入）声明；b5 env_blob 同
// rt_scene 配方：替引用方声明一次并真读（前缀 {0..7}）。
@group(2) @binding(5) var<storage, read> env_blob: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = f32(gid.x) * 0.05 + 0.05;
  let n = vec3<f32>(0.0, 0.0, 1.0);
  let w_in = vnorm(vec3<f32>(0.2, 0.1, -1.0));
  // T-B 路径模式采样流：64 位 PCG 仿真 → Sobol 索引/掩码 → 维度取值。
  let st = wf_stream(3u, 1u, 7u, gid.x, 0u, 0u);
  let s1 = wf_sample_1d(st, 5u);
  let s2 = wf_sample_2d(st, 6u);
  let mat = load_material(0u);
  let bs = bsdf_sample(mat, w_in, n, 1u, st);
  let h = sample_vndf(-w_in, 0.25, s1, s2.x);
  // g2 前缀 {0..7} 要求 b0/b3/b4 也被静态引用：scene_intersect 真读 tris(b3)+
  // scene_params(b0)、load_light_count 真读 scene_params、lights(b4) 显式读一条。
  let o2 = vec3<f32>(0.0, 0.0, -2.0);
  let d2 = vnorm(vec3<f32>(f - 0.1, 0.05 + f, 1.0));
  let hit = scene_intersect(o2, d2, 1e30);
  let nl = load_light_count();
  let l0 = lights[0];
  // 【为什么显式读 scene_buf】该绑定在 rt_shade.wgsl 里声明但**库件内无函数消费**。
  // 若入口也不读它，layout:"auto" 会把它从绑定布局里剔掉，createBindGroup 便不再
  // 校验它 —— 声明与实际用法脱节就查不出来。故显式读一次并计入结果。
  let sb = f32(scene_buf[0]) * 1e-9;
  var acc = s1 + s2.x + s2.y + sb;
  acc = acc + power_heuristic(s1, s2.x);
  acc = acc + ggx_d(0.25, f) + smith_g1(0.25, f) + pow5(f);
  acc = acc + luminance(schlick(vec3<f32>(0.04), f));
  acc = acc + vdot(refract_dir(w_in, n, 0.5), w_in) + vdot(h, h);
  acc = acc + luminance(bs.weight) + bs.pdf * 1e-3;
  acc = acc + f32(bs.specular) * 1e-3 + f32(bs.entered) * 1e-3;
  acc = acc + vdot(bs.dir, n);
  acc = acc + hit.t * 1e-3 + f32(nl) * 1e-6 + luminance(l0.color) * 1e-6
    + l0.radius * 1e-6;
  acc = acc + env_blob[0] * 1e-9 + spheres[0] * 1e-9 + dir_lights[0] * 1e-9;
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
      { group: 2, binding: 5, kind: "envBlob" },
      { group: 2, binding: 6, kind: "spheresFlat" },
      { group: 2, binding: 7, kind: "dirLightsFlat" },
    ],
  },

  // ---- rt_u64.wgsl：64 位 PCG 的 (lo,hi) u32 仿真库（T-B / R-1）----
  // 无绑定无入口；合成入口真调 u64 加/乘/移位/异或与整条 hash 链并回写。
  // 【判据意义】u64_mul 的 16 位 limb 进位传播是 R-1 逐位对拍里修掉的真缺陷
  // （进位加回同一 limb ⇒ 2000/2000 hash 全错）；本配方保证该库件永远过真编译。
  "rt_u64.wgsl": {
    outBinding: 0,
    extraWgsl: `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x + 1u;
  let a = u64_add(0x12345678u, 0x9abcdef0u, g, g << 1u);
  let m = u64_mul(a.x, a.y, 0x4c957f2du, 0x5851f42du);
  let sh = u64_shr(m.x, m.y, 17u);
  let x = u64_xor(sh.x, sh.y, g, 0u);
  let h1 = pcg_hash1(x.x, x.y, g, 0u);
  let hp = pcg_hash2_prefix(x.x, x.y, g, 0u);
  let h2 = pcg_hash2_finish(hp.x, hp.y, g, 0u);
  let h4 = pcg_hash4(x.x, x.y, g, 0u, 1u, 0u, 2u, 0u, 3u, 0u);
  let h3 = pcg_hash3(x.x, x.y, g, 0u, 1u, 0u, 2u, 0u);
  var acc = pcg_unit_f32(h1.x, h1.y);
  acc = acc + pcg_unit_f32(h2.x, h2.y) + pcg_unit_f32(h4.x, h4.y) + pcg_unit_f32(h3.x, h3.y);
  acc = acc + f32(pcg_permute(g, 0u).x) * 2.3283064e-10;
  lib_out[gid.x] = acc;
}
`,
    buffers: [{ group: 0, binding: 0, kind: "out" }],
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

  // ---- rt_env.wgsl：自带 @group(0) @binding(0) 的 CDF blob（只读 storage）----
  //
  // 合成入口必须在**真 CDF 表**上跑一遍采样 / 双线性 / pdf，否则 `env_u(5u) == 0`
  // 会让 `env_sample` 早退成全零，这条链路就变成"编译过但语义没跑"。
  // 故 `envBlob` 种的是一个 2×2 的**最小合法 blob**（头部 + 边际 + 条件 + texels）。
  "rt_env.wgsl": {
    outBinding: 1,
    extraWgsl: `
@group(0) @binding(0) var<storage, read> env_blob: array<f32>;
// 【W2 剩余后的声明面】rt_env.wgsl 的函数体共享、绑定声明**移出**（改由引用方
// 各自声明 env_blob）。本合成入口按 wavefront 引用方的同一布局替它声明一次，
// 让 env_u/env_f 的取数有真缓冲可读（否则整条 CDF/采样链读到的是未定义绑定）。
@group(0) @binding(1) var<storage, read_write> lib_out: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let u1 = 0.05 + f32(gid.x) * 0.1;
  let u2 = 0.3 + f32(gid.x) * 0.05;
  let s = env_sample(u1, u2);
  let r = env_radiance(s.wi);
  let p = env_pdf(s.wi);
  let uv = env_dir_to_uv(s.wi);
  let back = env_uv_to_dir(uv.x, uv.y);
  var acc = s.pdf + p + r.x + r.y + r.z + uv.x + uv.y;
  acc = acc + back.x + back.y + back.z;
  acc = acc + env_row_omega(0u) + env_acos(0.5) + env_atan2(1.0, 1.0);
  acc = acc + env_quarter_atan(0.25) + env_atan_pos(0.3);
  acc = acc + f32(env_w() + env_h()) * 1e-6 + f32(s.texel) * 1e-6;
  acc = acc + f32(env_u(0u) + env_u(5u) + env_u(6u)) * 1e-9 + env_f(4u) * 1e-9;
  // 未命中亮斑的哨兵是 u32::MAX（f32 上 4.29e9），乘 1e-9 后仍是有限数。
  acc = acc + f32(env_spot_index(0u)) * 1e-9;
  acc = acc + f32(env_spot_index(1u)) * 1e-9;
  lib_out[gid.x] = acc;
}
`,
    buffers: [
      { group: 0, binding: 0, kind: "envBlob" },      { group: 0, binding: 1, kind: "out" },
    ],
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
  "rt_env.wgsl": {
    mustApply: true,
    // 形参/实参个数不匹配（与上一波事故同形态）：`env_acos` 多声明一个形参，
    // 而 `env_dir_to_uv` 里仍按 1 个实参调用 → 真编译器必拒。
    apply: (s) => s.replace(
      "fn env_acos(x: f32) -> f32 {",
      "fn env_acos(x: f32, extra: f32) -> f32 {"),
    expect: /env_acos/,
  },
  "rt_u64.wgsl": {
    mustApply: true,
    // 进位传播链断裂（T-B 真实缺陷的注入形态）：u64_mul 的 limb 进位多加一次，
    // 语义错但语法合法 ⇒ 只有"合成入口真调 + 回读有限"之外的**语义判据**能抓；
    // 这里注入的是编译层自证：多传一个实参 → 真编译器必拒。
    apply: (s) => s.replace(
      "fn u64_add(al: u32, ah: u32, bl: u32, bh: u32) -> vec2<u32> {",
      "fn u64_add(al: u32, ah: u32, bl: u32, bh: u32, extra: u32) -> vec2<u32> {"),
    expect: /u64_add/,
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
      dv.setUint32(40, 0, true);                             // kind（T-B：材质判别式）
      f(48, 0.0); f(52, 0.0); f(56, 0.0);                    // emissive（@48 对齐 16）
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
    // env_blob（布局权威 = `crates/prism-rt/tests/qa_w2_env_gpu_parity.rs::build_blob`
    // 与 `rt_env.wgsl` 文件头那张表）：2×2 的最小合法 blob。
    //
    //   [0]=w=2 [1]=h=2 [2]=spot_count=0 [3]=p_spot=0.0 [4]=Ω_total [5]=has_energy=1
    //   [6]=has_residual=1 [7]=cond_len=(w+1)·h=6 [8]=marg_len=h+1=3 [9]=res_cond_len=0
    //   [10]=res_row_off_len=2 [11]=spot_prefix_len=1 [12]=spot_texels_len=0
    //   [13]=texel_words=4·w·h=16 [14..20]=各段偏移
    //
    // 条件/边际表按"每行每列等权"填（`[0, 2^31, MAX]`），texels 给 4 个不同亮度
    // 的 RGBA —— 这样采样、双线性、pdf 三条路径都会产出**非零且有限**的数。
    case "envBlob": {
      const cond = [0, 0x80000000, 0xFFFFFFFF, 0, 0x80000000, 0xFFFFFFFF];
      const marg = [0, 0x80000000, 0xFFFFFFFF];
      const rowOff = [0xFFFFFFFF, 0xFFFFFFFF]; // 无亮斑 ⇒ 两行都读主表
      const spotPrefix = [0xFFFFFFFF];          // 空亮斑列表（den 不被读）
      const texels = [
        4.0, 3.0, 2.0, 1.0,
        0.5, 0.5, 0.5, 1.0,
        0.25, 0.25, 0.25, 1.0,
        0.125, 0.125, 0.125, 1.0,
      ];
      const total = 24 + cond.length + marg.length + rowOff.length + spotPrefix.length + texels.length;
      const ab = new ArrayBuffer(total * 4);
      const dv = new DataView(ab);
      const u = (i, v) => dv.setUint32(i * 4, v >>> 0, true);
      // f32 位模式写进同一个字槽（blob 在 WGSL 侧是 `array<f32>` 承载 u32 位模式）。
      const f = (i, v) => {
        const t = new DataView(new ArrayBuffer(4));
        t.setFloat32(0, v, true);
        dv.setUint32(i * 4, t.getUint32(0, true), true);
      };
      u(0, 2); u(1, 2); u(2, 0); f(3, 0.0); f(4, 4.0 * Math.PI); u(5, 1); u(6, 1);
      u(7, cond.length); u(8, marg.length); u(9, 0);
      u(10, rowOff.length); u(11, spotPrefix.length); u(12, 0); u(13, texels.length);
      let at = 24;
      u(14, at); for (const v of cond) u(at++, v);
      u(15, at); for (const v of marg) u(at++, v);
      u(16, at); // res_cond_len = 0 ⇒ 同址（不读）
      u(17, at); for (const v of rowOff) u(at++, v);
      u(18, at); for (const v of spotPrefix) u(at++, v);
      u(19, at); // spot_texels_len = 0 ⇒ 同址
      u(20, at); for (const v of texels) f(at++, v);
      return ab;
    }
    case "sceneBuf":
      return u32buf([7, 0, 0, 0]);
    case "primMat":
      return u32buf([0, 0, 0, 0]);
    // 兼容模式的球表 / 方向光表（平铺 8 f32/条，rt_scene.wgsl b6/b7）。
    // 全零即可：合成入口只验证"可读且有限"，不消费数值。
    case "spheresFlat":
    case "dirLightsFlat":
      return new ArrayBuffer(8 * 4);
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
// G-8① I2：wavefront 实例层级在真 WebGPU 上的 CPU/GPU 逐位对拍
// ---------------------------------------------------------------------------
//
// 【为什么并进本文件而不是另起一个 harness】上一轮已建好的 Xvfb + Chromium +
// SwiftShader 真机链路就在这里；I2 复用 `runOnGpu`（真 createShaderModule +
// 真 dispatch + validation error scope + mapAsync 回读），只换一份合成入口与
// 一组缓冲数据。判据口径与 `qa_wavefront_parity.rs` 的 I1/I3/I4 一致：按 f32
// 位模式逐字比较，正控 differing_pixels = 0；故意改错一个实例的逆仿射平移，
// 负控自身与「改坏后的 CPU 权威」也 0 差，但与正控参考必须非 0 差（防「根本
// 没消费实例表」的假绿）。
//
// 【为什么能逐位对拍而不是放宽阈值】所有坐标 / 仿射 / 射线方向都取 2 的幂或
// 5/8 这类 f32 精确值；Möller–Trumbore 的 det = -4 或 -1、inv_det = -0.25 或 -1、
// t = 4 全部精确，法线归一化 sqrt(16) = 4 也精确。JS 侧 CPU 权威逐式用
// Math.fround 复刻 WGSL 的运算顺序，位模式可直接对齐，不需要容差、不改尺子。
//
// 【判别力来自哪里】本场景 plain_tri_count = 0：`tris` 里两个三角形**只能**经
// 实例表到达。若 shader 没走实例分支，8 条射线全部 miss，输出必与 CPU 权威
// 非 0 差。故「0 差」本身就证明了实例表被真消费。

/** 实例表尾描述符（与 `vulkan::wavefront::append_instance_table` 同源）。 */
const IP_MAGIC = 0x57384931;
const IP_INST_STRIDE = 28;
const IP_TAIL = 8;
/** I2 的射线网格：16×16 = 256 像素，与 I1 的 16×16 帧同口径。 */
const IP_DISPATCH = 256;
/** 每条射线回写 5 个 f32：t / normal.xyz / prim。 */
const IP_OUT_FLOATS = 5;

/** 合成入口：8 条射线各求一次 `scene_intersect`，把命中记录逐字回写。 */
const IP_WGSL = `
@group(0) @binding(0) var<storage, read_write> lib_out: array<f32>;
@group(2) @binding(5) var<storage, read> env_blob: array<f32>;

@compute @workgroup_size(1)
fn lib_compile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // 16×16 网格：x/y 步长都是 2 的幂（9/32、1/32），f32 精确可表示。
  // 三个实例的世界 x 区间互不重叠（[-0.5,0.5] / [1.5,2.5] / [3.75,4.25]），
  // 故命中能区分 identity / 平移 / 镜像三种变换与两个原型。
  let x = -0.25 + f32(i % 16u) * 0.28125;
  let y = -0.25 + f32(i / 16u) * 0.03125;
  let o = vec3<f32>(x, y, -4.0);
  let d = vec3<f32>(0.0, 0.0, 1.0);
  let h = scene_intersect(o, d, 1e30);
  let base = i * 5u;
  // b1/b4/b5/b6/b7 显式真读：layout:"auto" 只保留被静态引用的绑定，
  // 缺读会让引用集不再是连续前缀 {0..7} → createBindGroup 验证失败。
  let filler = materials[0].roughness * 1e-30 + lights[0].radius * 1e-30
    + env_blob[0] * 1e-30 + spheres[0] * 1e-30 + dir_lights[0] * 1e-30;
  lib_out[base + 0u] = h.t + filler;
  lib_out[base + 1u] = h.normal.x;
  lib_out[base + 2u] = h.normal.y;
  lib_out[base + 3u] = h.normal.z;
  lib_out[base + 4u] = f32(h.prim);
}
`;

/** f32 → 位模式（小端），供逐位比较与 prim_mat 编码共用。 */
function ipBits(v) {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, v, true);
  return dv.getUint32(0, true) >>> 0;
}

/** 位模式 → f32。 */
function ipFloat(w) {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setUint32(0, w >>> 0, true);
  return dv.getFloat32(0, true);
}

/** 两个局部 BLAS 原型：各一个 z=0 平面三角形（Tri: a@0 b@16 c@32，48 B）。 */
function ipGeoms() {
  const geoms = [
    { a: [-1, -1, 0], b: [1, -1, 0], c: [0, 1, 0] },
    { a: [-0.5, -0.5, 0], b: [0.5, -0.5, 0], c: [0, 0.5, 0] },
  ];
  const ab = new ArrayBuffer(geoms.length * 48);
  const dv = new DataView(ab);
  geoms.forEach((g, k) => {
    const put3 = (off, v) => v.forEach((x, j) => dv.setFloat32(off + j * 4, x, true));
    put3(k * 48 + 0, g.a);
    put3(k * 48 + 16, g.b);
    put3(k * 48 + 32, g.c);
  });
  return { bytes: ab, tris: geoms, count: geoms.length };
}

/**
 * 3 个实例：identity / 平移 / 镜像（det<0），2 个原型。
 * 112 B 记录的槽序与 `crates/prism-gpu/src/instance.rs::encode_instances` 一致：
 *   0..12 affine（行主序 3×4） / 12..24 inverse / 24 geom_id / 25 mat_id /
 *   26 flags / 27 node_base。
 */
function ipInstances(tamper) {
  const aff = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0],
  ];
  const inv = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, -2, 0, 1, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0],
  ];
  // I4 同形态负控：把第 1 个实例的 inverse 平移从 -2.0 改成 -1.25。
  if (tamper) inv[1][3] = -1.25;
  const geomId = [0, 0, 1];
  const flags = [1, 0, 2]; // IDENTITY / 普通 / MIRRORED
  const words = [];
  for (let k = 0; k < 3; k += 1) {
    for (const v of aff[k]) words.push(ipBits(v));
    for (const v of inv[k]) words.push(ipBits(v));
    words.push(geomId[k], 0, flags[k], 0);
  }
  return { words, aff, inv, geomId, flags, count: 3 };
}

/** 组装整份实例场景（prim_mat 尾描述符 + tris 局部 BLAS）。 */
function ipScene(tamper) {
  const inst = ipInstances(tamper);
  const geo = ipGeoms();
  const instanceBase = 0;
  const geomBase = inst.words.length;
  const geomRows = [[0, 1, 0, 0], [1, 1, 0, 0]];
  const words = [...inst.words];
  for (const row of geomRows) words.push(...row);
  const blasTriBase = 0;
  const plainTriCount = 0;
  const tail = [
    IP_MAGIC, inst.count, geomRows.length, instanceBase, geomBase,
    plainTriCount, blasTriBase, geo.count,
  ];
  if (tail.length !== IP_TAIL || inst.words.length !== inst.count * IP_INST_STRIDE) {
    throw new Error("I2 实例表槽数与 WF_INST_TAIL / WF_INST_STRIDE 契约不一致");
  }
  words.push(...tail);
  return {
    primMat: u32buf(words),
    tris: geo.bytes,
    scene: {
      inst, geo, geomRows, instanceBase, geomBase, blasTriBase,
      plainTriCount, totalTriCount: geo.count, words,
    },
  };
}

/** 该 job 的 9 个绑定缓冲（内容逐字节对齐 WGSL struct 布局）。 */
function ipBuffers(built) {
  const sp = new ArrayBuffer(32);
  const sd = new DataView(sp);
  sd.setUint32(0, 1, true);                                  // material_count
  sd.setUint32(4, 1, true);                                  // light_count
  sd.setUint32(8, built.scene.totalTriCount, true);          // prim_count
  sd.setUint32(12, 7, true);                                 // node_id
  sd.setFloat32(16, 0.1, true);
  sd.setFloat32(20, 0.2, true);
  sd.setFloat32(24, 0.3, true);
  sd.setUint32(28, 0, true);                                 // sphere_count
  const mats = new ArrayBuffer(64);
  new DataView(mats).setFloat32(12, 0.5, true);              // roughness
  const lights = new ArrayBuffer(32);
  new DataView(lights).setFloat32(12, 0.5, true);            // radius
  return [
    { group: 0, binding: 0, kind: "out", data: new ArrayBuffer(IP_DISPATCH * IP_OUT_FLOATS * 4) },
    { group: 2, binding: 0, kind: "sceneParams", uniform: true, data: sp },
    { group: 2, binding: 1, kind: "materials", data: mats },
    { group: 2, binding: 2, kind: "primMat", data: built.primMat },
    { group: 2, binding: 3, kind: "sceneTris", data: built.tris },
    { group: 2, binding: 4, kind: "lights", data: lights },
    { group: 2, binding: 5, kind: "envBlob", data: new ArrayBuffer(4) },
    { group: 2, binding: 6, kind: "spheresFlat", data: new ArrayBuffer(32) },
    { group: 2, binding: 7, kind: "dirLightsFlat", data: new ArrayBuffer(32) },
  ];
}

/** 组装一个真机 job；`outFloats` 让 `runOnGpu` 回读全部 5×DISPATCH 个 f32。 */
function ipJob(label, built) {
  return {
    file: label,
    entry: "lib_compile_main",
    code: `${expandIncludes("rt_scene.wgsl")}\n${IP_WGSL}`,
    groups: [0, 2],
    dispatch: IP_DISPATCH,
    outFloats: IP_DISPATCH * IP_OUT_FLOATS,
    buffers: ipBuffers(built).map((b) => ({
      group: b.group,
      binding: b.binding,
      kind: b.kind,
      uniform: b.uniform === true,
      bytes: b.data.byteLength,
      data: Array.from(new Uint8Array(b.data)),
    })),
  };
}

/**
 * JS 侧 CPU 权威：逐式复刻 `rt_scene.wgsl::scene_intersect` 的实例遍历，
 * 返回每条射线 5 个 f32 的**位模式**数组。独立实现，不调用页面内 WGSL。
 */
function ipCpuReference(built) {
  const f = Math.fround;
  const s = built.scene;
  const words = s.words;
  const cross = (a, b) => [
    f(f(a[1] * b[2]) - f(a[2] * b[1])),
    f(f(a[2] * b[0]) - f(a[0] * b[2])),
    f(f(a[0] * b[1]) - f(a[1] * b[0])),
  ];
  const dot = (a, b) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
  const sub = (a, b) => [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
  const unit = (n) => {
    const l2 = dot(n, n);
    if (!(l2 > 1e-24)) return [0, 0, 0];
    const k = f(1 / Math.sqrt(l2));
    return [f(n[0] * k), f(n[1] * k), f(n[2] * k)];
  };
  /** 与 `wf_tri_hit` 同树；命中返回 t，否则 null。 */
  const triHit = (g, ro, rd, tMax) => {
    const e1 = sub(g.b, g.a);
    const e2 = sub(g.c, g.a);
    const pvec = cross(rd, e2);
    const det = dot(e1, pvec);
    if (!(Math.abs(det) >= 1e-12)) return null;
    const invDet = f(1 / det);
    const tvec = sub(ro, g.a);
    const u = f(dot(tvec, pvec) * invDet);
    if (!(u >= 0 && u <= 1)) return null;
    const qvec = cross(tvec, e1);
    const v = f(dot(rd, qvec) * invDet);
    if (!(v >= 0 && f(u + v) <= 1)) return null;
    const t = f(dot(e2, qvec) * invDet);
    if (!(t >= 1e-4 && t <= tMax)) return null;
    return t;
  };
  /** 与 `wf_inst_local_point` / `wf_inst_local_dir` 同式。 */
  const invOf = (k) => {
    const b = s.instanceBase + k * IP_INST_STRIDE + 12;
    return words.slice(b, b + 12).map(ipFloat);
  };
  const localPoint = (m, p) => [
    f(f(f(f(m[0] * p[0]) + f(m[1] * p[1])) + f(m[2] * p[2])) + m[3]),
    f(f(f(f(m[4] * p[0]) + f(m[5] * p[1])) + f(m[6] * p[2])) + m[7]),
    f(f(f(f(m[8] * p[0]) + f(m[9] * p[1])) + f(m[10] * p[2])) + m[11]),
  ];
  const localDir = (m, p) => [
    f(f(f(m[0] * p[0]) + f(m[1] * p[1])) + f(m[2] * p[2])),
    f(f(f(m[4] * p[0]) + f(m[5] * p[1])) + f(m[6] * p[2])),
    f(f(f(m[8] * p[0]) + f(m[9] * p[1])) + f(m[10] * p[2])),
  ];
  /** 与 `wf_inst_world_normal` 同式：A⁻ᵀ 用 inverse 3×3 的转置，det<0 反号。 */
  const worldNormal = (m, n, mirrored) => {
    const o = [
      f(f(f(m[0] * n[0]) + f(m[4] * n[1])) + f(m[8] * n[2])),
      f(f(f(m[1] * n[0]) + f(m[5] * n[1])) + f(m[9] * n[2])),
      f(f(f(m[2] * n[0]) + f(m[6] * n[1])) + f(m[10] * n[2])),
    ];
    return mirrored ? [f(-o[0]), f(-o[1]), f(-o[2])] : o;
  };
  const rawNormal = (g) => cross(sub(g.b, g.a), sub(g.c, g.a));

  const out = [];
  for (let i = 0; i < IP_DISPATCH; i += 1) {
    const x = f(-0.25 + f((i % 16) * 0.28125));
    const y = f(-0.25 + f(Math.floor(i / 16) * 0.03125));
    const ro = [x, y, -4];
    const rd = [0, 0, 1];
    let t = 1e30;
    let prim = 0xFFFFFFFF;
    let normal = [0, 0, 0];
    // ---- 普通三角形（本场景 plain_tri_count = 0；保留循环以复刻同序）----
    for (let j = 0; j < s.plainTriCount; j += 1) {
      const ht = triHit(s.geo.tris[j], ro, rd, t);
      if (ht !== null && (ht < t || prim === 0xFFFFFFFF)) {
        t = ht; prim = j; normal = unit(rawNormal(s.geo.tris[j]));
      }
    }
    // ---- 实例族：每条实例只遍历自己 geom_id 指向的 BLAS ----
    for (let k = 0; k < s.inst.count; k += 1) {
      const base = s.instanceBase + k * IP_INST_STRIDE;
      if (base + IP_INST_STRIDE > words.length) continue;
      const flags = words[base + 26];
      const identity = (flags & 1) !== 0;
      const mirrored = (flags & 2) !== 0;
      const m = invOf(k);
      const roL = identity ? ro : localPoint(m, ro);
      const rdL = identity ? rd : localDir(m, rd);
      const gi = s.geomBase + words[base + 24] * 4;
      if (gi + 2 > words.length) continue;
      const tStart = words[gi];
      const tCount = words[gi + 1];
      for (let j = 0; j < tCount; j += 1) {
        const slot = tStart + j;
        if (slot >= s.totalTriCount) continue;
        const ht = triHit(s.geo.tris[slot], roL, rdL, t);
        if (ht !== null && (ht < t || prim === 0xFFFFFFFF)) {
          t = ht;
          prim = slot;
          const nl = rawNormal(s.geo.tris[slot]);
          normal = identity ? unit(nl) : unit(worldNormal(m, nl, mirrored));
        }
      }
    }
    out.push(ipBits(t), ipBits(normal[0]), ipBits(normal[1]), ipBits(normal[2]), ipBits(prim));
  }
  return out;
}

/** GPU 回读的 f32 数组 vs CPU 权威位模式数组 → differing_pixels。 */
function ipDiffering(gpu, cpu) {
  if (!Array.isArray(gpu) || gpu.length < IP_DISPATCH * IP_OUT_FLOATS) return -1;
  let diff = 0;
  for (let i = 0; i < IP_DISPATCH; i += 1) {
    let bad = false;
    for (let c = 0; c < IP_OUT_FLOATS; c += 1) {
      const at = i * IP_OUT_FLOATS + c;
      if (ipBits(gpu[at]) !== cpu[at]) bad = true;
    }
    if (bad) diff += 1;
  }
  return diff;
}

/** I2 主流程：正控 + 负控两个真 dispatch，各自与自己的 CPU 权威比对。 */
async function runInstanceParity(asJson) {
  const pos = ipScene(false);
  const neg = ipScene(true);
  const jobs = [
    ipJob("rt_scene.wgsl [实例正控]", pos),
    ipJob("rt_scene.wgsl [实例负控]", neg),
  ];
  const { adapter, results, gpuErrs } = await runOnGpu(jobs, []);
  if (gpuErrs?.length) {
    if (asJson) console.log(JSON.stringify({ ok: false, why: gpuErrs.join(" | ") }, null, 2));
    else console.error(`WebGPU 运行时错误：${gpuErrs.join(" | ")}`);
    process.exit(1);
  }
  const cpuPos = ipCpuReference(pos);
  const cpuNeg = ipCpuReference(neg);
  const posDiff = ipDiffering(results[0]?.out, cpuPos);
  const negDiff = ipDiffering(results[1]?.out, cpuNeg);
  const tamperVsPos = ipDiffering(results[1]?.out, cpuPos);
  const dispatchOk = results.every((r) => r?.ok === true && r.stage === "ok" &&
    !r.submitError && (r.nonFinite || []).length === 0 && (r.errors || []).length === 0);
  const ok = dispatchOk && posDiff === 0 && negDiff === 0 && tamperVsPos > 0;
  const report = {
    ok,
    // 顶层专属字段：Rust `qa_wavefront_parity::i2_*` 按名取值，不与
    // `results[].ok` 混淆（后者是逐 job 的编译/dispatch 结果）。
    i2_ok: ok,
    adapter,
    dispatch: IP_DISPATCH,
    positive_differing_pixels: posDiff,
    negative_differing_pixels: negDiff,
    tamper_vs_positive_differing_pixels: tamperVsPos,
    instance_count: pos.scene.inst.count,
    geom_count: pos.scene.geomRows.length,
    instance_words: pos.scene.words.length,
    dispatch_ok: dispatchOk,
    results: results.map((r) => ({
      file: r?.file, ok: r?.ok, stage: r?.stage,
      submitError: r?.submitError ?? "", errors: r?.errors ?? [],
      nonFinite: r?.nonFinite ?? [], out: r?.out ?? [],
    })),
  };
  if (asJson) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log("== G-8① I2 浏览器 WebGPU 实例层级逐位对拍（真 dispatch + 回读）==");
    console.log(`adapter = ${adapter}；${pos.scene.inst.count} 实例 / ${pos.scene.geomRows.length} 原型 / 含镜像 det<0；prim_mat ${pos.scene.words.length} 字`);
    console.log(`[I2 正控] CPU 权威 vs WebGPU：differing_pixels = ${posDiff}/${IP_DISPATCH}`);
    console.log(`[I2 负控] 改坏逆仿射后自身 CPU 权威 vs WebGPU：differing_pixels = ${negDiff}/${IP_DISPATCH}`);
    console.log(`[I2 判别力] 负控输出 vs 正控 CPU 权威：differing_pixels = ${tamperVsPos}/${IP_DISPATCH}（必须 > 0）`);
    console.log(`[I2 提交] dispatch 0 验证错误、0 error、0 非有限：${dispatchOk}`);
    console.log(ok ? "==> I2 通过。" : "==> I2 FAIL。");
  }
  process.exit(ok ? 0 : 1);
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
      // 【requiredLimits 为什么必须显式要】rt_shade 展开后的合成单体引用 10 个
      // storage buffer（g0 out + g1 两表 + g2 场景 7 表），超过 WebGPU 默认保证
      // 值 8 —— 不申请则 createComputePipeline 直接 Invalid（本链路真机实测）。
      // 按适配器实际上报值取 min(10, adapter)，低上限硬件自动降级。
      const wantBufs = Math.min(10, adapter.limits?.maxStorageBuffersPerShaderStage ?? 8);
      const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: wantBufs },
      });
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
          // 默认每条 workgroup 回读 1 个 f32；I2 实例对拍 job 用 `outFloats`
          // 指定 5 个 f32/射线（t + normal.xyz + prim），其余库件行为不变。
          const readFloats = (job.outFloats ?? job.dispatch) * 4;
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
  // G-8① I2：复用同一条真机链路，只换成「实例场景 + 逐位对拍」的两个 job。
  // 默认路径（host_test.mjs 的 7c 节）完全不变，不新增任何门禁项或 harness。
  if (argv.includes("--instance-parity")) {
    await runInstanceParity(AS_JSON);
    return;
  }
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
