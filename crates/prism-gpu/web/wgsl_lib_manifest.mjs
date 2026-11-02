// ===========================================================================
// wgsl_lib_manifest.mjs — 无入口 WGSL「库件」真编译清单（单一真源）
// ===========================================================================
//
// # 为什么存在（本波最核心的产出）
//
// `crates/prism-gpu/shaders/` 下有两类 WGSL：
//   * **kernel**（有 `@compute` / `@vertex` / `@fragment` 入口）：由
//     `shaders::SHADERS` 登记，逐份独立编译判对；规则 A/G 守它们。
//   * **库件**（无任何入口）：只被 `//!include` 拼进别的 shader，或登记在
//     `shaders::LIBRARIES`（`rt_scene.wgsl` / `hair_scatter.wgsl`）。
//
// 【本项目的根因事故】库件**从不进入任何 `createShaderModule`**——没有入口，
// 宿主不会为它建 shader module，于是：
//   * 类型错、签名错、语义错可以**长期存活**，在 F1–F6 全绿下 unnoticed。
//   * 上一波最严重的一例：`hair_scatter.wgsl` 的 `a_r` / `a_tt` / `a_trt`
//     声明 2 / 4 / 4 个形参，却按 1 / 2 / 2 个实参调用 —— 该文件**根本无法通过
//     `createShaderModule`**，却因登记为 `LIBRARIES`（无入口、不参与 kernel 编译）
//     一直绿灯。任何"编译过"的断言都没覆盖到它。
//
// # 本文件做什么
//
// 把**每一个无入口库件**登记进一条**真实编译链路**（`wgsl_lib_compile.mjs`）：
//   库件全文（`//!include` 已展开） + 一段合成 `@compute` 入口
//   → `createShaderModule` → `getCompilationInfo()`（断言 0 error）
//   → `createComputePipeline` → 真 `dispatchWorkgroups` → 回读校验。
//
// # 为什么清单要独立成一个模块（架构理由）
//
// 1. **单一真源**：`wgsl_lib_compile.mjs`（执行者）与 `wgsl_lint.mjs` 规则 **F18**
//    （静态门禁）都必须读同一份清单。若各写一份，两份会随时间漂移，最终出现
//    "门禁说 A、脚本编 B"——比没有门禁更坏。本项目已因"两处各写一份"吃过亏。
// 2. **零副作用**：`wgsl_lib_compile.mjs` 顶层会真起 Xvfb + Chromium + SwiftShader。
//    若让 `wgsl_lint.mjs` 去 import 它，静态门禁会被拖进真机链路（CI 无 GPU 时
//    直接不可用）。故清单独立成**纯数据**模块，两侧都只 import 它。
// 3. **F18 才有判据**：规则 F18 的反向用例要"把某库件从清单里删掉 → 判红"，
//    必须能**注入一份被删改的清单**。纯数据模块天然支持（import 后 slice 即可）。
//
// # 清单项契约（`lib_compile` 与 F18 都按此校验）
//
//   file   必填  库件文件名（相对 `shaders/`），必须**真实存在**且**无入口**。
//   entry  必填  合成 compute 入口名（非空）。真实编译链路用它建 pipeline。
//   groups 必填  合成入口会 `setBindGroup` 的 group 号列表，升序、无重复。
//                 —— `layout:"auto"` 的硬约束是"每个 entry 的静态绑定引用集必须是
//                 连续前缀 {0..n-1}"，故 `groups[i]` 对应的绑定必须从 0 起连续。
//                 库件自带的 group（如 `rt_scene` 的 group 2、`rt_shade` 的
//                 group 1）也必须在清单里如实登记，否则 `createBindGroup`
//                 会因缺 group 而验证失败。
//   note   必填  该库件的绑定占用说明（人读；F18 只查非空）。
//
// 【为什么"groups"是清单的一部分而不是编译脚本的内部细节】因为它就是
// `layout:"auto"` 的**契约面**：库件自带 `@group(2) @binding(0..4)`，合成入口
// 必须同时满足 group 0（自己的输出缓冲）与 group 2（库件的绑定）。哪一组的
// 绑定号跳号，真机立刻 `createBindGroup` 验证失败——把它登记成清单字段，
// 静态门禁就能在**不接 GPU** 的前提下先查一遍。
// ===========================================================================

/**
 * 无入口 WGSL 库件的真编译清单。
 *
 * 【完备性契约】`shaders/` 下**每一个**不含 `@compute` / `@vertex` /
 * `@fragment` 的 `.wgsl` 都必须在册。规则 F18 以"实际扫目录得到的集合"为
 * 真值反向比对本清单：少一条即红（新增库件忘了登记 = 又开一个编译盲区）。
 *
 * @type {ReadonlyArray<{file: string, entry: string, groups: readonly number[], note: string}>}
 */
export const LIB_MANIFEST = Object.freeze([
  Object.freeze({
    file: "rt_common.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0]),
    note: "公共常量与 Vec3/采样件；无任何绑定，合成输出独占 group 0。",
  }),
  Object.freeze({
    file: "sim_common.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0]),
    note: "百万粒子公共件（空间哈希 / Poly6 / XPBD 投影）；无绑定。",
  }),
  Object.freeze({
    file: "rt_bvh.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0]),
    note: "BVH 遍历 + Möller–Trumbore；无模块级绑定，但 bvh_intersect 形参是 storage 指针，合成入口须自建 nodes/tris 数组（故仍只用 group 0）。",
  }),
  Object.freeze({
    file: "rt_scene.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0, 2]),
    note: "场景访问层；自带 @group(2) @binding(0..4)（uniform scene_params + materials/prim_mat/tris/lights）。合成输出另占 group 0。",
  }),
  Object.freeze({
    file: "rt_shade.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0, 1, 2]),
    note: "BSDF/NEE/采样流；自带 @group(1) @binding(0..1)（sobol_dir/scene_buf），并经 //!include rt_scene 带入 @group(2) @binding(0..4)。",
  }),
  Object.freeze({
    file: "rt_env.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0]),
    note: "A2 / W2 环境贴图采样（等距柱状 + u32 定点 CDF + 同源 pdf）；自带 @group(0) @binding(0) 的 blob（只读 storage），合成输出另占 group 0 binding 1，故仍在 group 0 的连续前缀内。",
  }),
  Object.freeze({
    file: "hair_scatter.wgsl",
    entry: "lib_compile_main",
    groups: Object.freeze([0]),
    note: "Marschner R/TT/TRT 三瓣；LIBRARIES 成员，无入口。上一波 a_r/a_tt/a_trt 形参/实参不匹配即在本条被真编译器抓住。",
  }),
]);

/** 清单里出现过的库件文件名集合（升序、去重）。供 F18 做集合比对。 */
export function libManifestFiles(manifest = LIB_MANIFEST) {
  return [...new Set(manifest.map((e) => e.file))].sort();
}

/** 取清单里指定库件的那一项；不存在返回 undefined（**不抛**——由调用方判红）。 */
export function libManifestEntry(file, manifest = LIB_MANIFEST) {
  return manifest.find((e) => e.file === file);
}
