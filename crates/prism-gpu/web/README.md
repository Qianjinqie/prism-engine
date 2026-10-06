# Prism GPU Harness（D-14 手机测试路径）

单页 HTML：`index.html`。手机浏览器（Chrome/Edge 113+、Safari 18+）打开即用，
无需本地环境。载体 = wasm（`wasm32-unknown-unknown`）+ WebGPU 手写绑定。

## 四条硬性要求落实位置

| # | 要求 | 实现位置 |
|---|------|----------|
| 1 | 特性检测 + 明确提示（不白屏） | `boot()`：无 `navigator.gpu` 时显示错误横幅并列出浏览器/版本，其余 UI 仍渲染 |
| 2 | 错误全捕获面板（GPU/panic/device lost 带上下文） | `window.error`/`unhandledrejection`/`device.lost`/`uncapturederror`/`pushErrorScope` 全部落 `logError()`，面板逐条含 kind/stage/test/code |
| 3 | 一键复制报告 | 「⧉ 一键复制报告」→ `navigator.clipboard`，报告含 UA/adapter/测试项清单/失败详情 |
| 4 | 测试项可点可重跑 + PASS/FAIL/SKIP | 测试项列表：点状态圆点重跑该项、点其余处展开详情；「全部运行」「重跑失败项」 |

## Rust ↔ JS 互操作

`crates/prism-gpu/src/web/device.rs` 与 `mod.rs` 用 `extern "C"` 手写声明浏览器函数
（零 wasm-bindgen/web-sys）。宿主侧实现见本页 `<script>` 内的 `glue` 对象
（`window.prismGlue`），与 extern 声明一一对应。所有 GPU 对象以 `u32`
句柄传递，宿主维护 `Map<u32, GPUObject>`。

## WGSL 源码来源

`scan_block` / `histogram` 两份 WGSL 与
`crates/prism-gpu/src/web/device.rs` 的 `WGSL_SCAN_BLOCK` / `WGSL_HISTOGRAM`
同源（结构自检 `webgpu_wgsl_sources_pass_structure_check` 覆盖）。
另 9 份 `shaders/*.wgsl` 经 `prism_shader_src(i)` 从 Rust 侧导出，
harness 直接取同一份源码编译，杜绝 JS 侧另抄一份导致不同步。
两者合计 **11 份** WGSL 过结构自检（9 内嵌 + 2 D0 内核，见测试项 #3）。

> **`text.wgsl` 的 `Camera2D` 死代码已删（本轮）**：`Camera2D` struct 与
> `@binding(2) var<uniform> cam2d` 从 `text.wgsl` 删除（**70 → 64 行**）。取证结论：
> 全仓仅 `text.wgsl` 两处出现、`vs_glyph`/`fs_text` 均不引用、宿主从无建缓冲、
> 该文件无 `@compute` 入口故从未 dispatch ⇒ 判决为**纯死代码**——**不是**"宿主漏建
> 缓冲"的真缺陷，也**不是**预留接口。因删除的是专门监测 cam2d 的断言，
> `host_test.mjs` 与 `wgsl_lint.mjs` 的断言数各减了几项，但随后被多 bind group
> 新增的断言覆盖（**最终实测值见下文，以实测为准**）。

### include 展开（D-14 真机实测确立，必读）

**WGSL 没有 include 机制**（`#include` / `@import` 都不是 WGSL 语法；本项目的
D-29 判据本就禁它们）。`rt_bvh` / `rt_shade` / `rt_wavefront` / `sim_*` 依赖
`rt_common.wgsl` / `sim_common.wgsl` / `rt_scene.wgsl` 里的公共件，但 WGSL 无传递
包含语义——真机首次实测时，6 份 shader 直接报 `unresolved call target`
（`vdot` / `sample_1d` / `cell_key` / `xpbd_distance_project` / `poly6_2d` …），
因为公共件从未与它们一起被编译。

引擎自研的拼接包含指令形如 `//!include rt_common.wgsl`（**必须独占一行**），
由 `shaders::expand()` 在**上传 GPU 之前**做文本级展开：

| 规则 | 说明 |
|------|------|
| 依赖序 | 被包含文件先于包含者输出 |
| 去重 | 同一文件在一次展开中只输出一次（菱形依赖不重复定义） |
| 环截断 | `visited` 集合兜底，不死循环、不 panic（C18） |
| 可诊断 | 每段前后加 `// ===== begin/end <文件名> =====` 标记 |
| 缓存 | `prism_shader_src` 走 `OnceLock`，地址一次定死（确定性铁律） |

`prism_shader_src(i)` / `_len(i)` 返回的是**展开后**的完整文本，可直接交给
`createShaderModule`。公共件中 `rt_common` / `rt_bvh` / `rt_shade` /
`sim_common` 同时是独立测试项（它们各自也能编译通过），而 `rt_scene.wgsl`
只被 include，登记在 `shaders::LIBRARIES` 而不在 `shaders::SHADERS`。

### device 就绪契约（两阶段，D-14 真机实测确立）

WebAssembly 的 import **只能是同步函数**——`prism_request_device` 若写成
`async` 会让模块链接直接失败（历史上导致整页"wasm 未加载"）。故它同步返回
**预留句柄**，真正的 `requestAdapter`/`requestDevice` 由宿主 `settle()` 兑现。

由此产生两条硬约定：

| 约定 | 理由 |
|------|------|
| `device.init` 分两阶段：`run` 发起并返回 `PENDING`，`collect` 才校验就绪 | 句柄非 0 只代表"已预约"；提前 PASS 会让**不存在的 device** 被判为就绪 |
| 宿主 `runWasmTest` 对非 PENDING 项**不得清空 `pendingWork`** | 清掉会让 `requestDevice` 的 promise 永不 await，句柄表里恒为 `null` |
| 新增胶水 `prism_device_ready(h)` | 判据 = `get(h) instanceof GPUDevice`；`ready_device()` 据此提前拦下未就绪句柄并给出可操作错误，而不是滑到 `create_buffer 返回 0` |

### CPU 侧门禁（真机问题的本地沉淀）

真机上暴露的 WGSL 缺陷，**绝大部分在 `cargo test` 阶段就能拦住**：

| 门禁 | 拦截的真机错误 |
|------|--------------|
| `shaders::unresolved_symbols` | `unresolved call target`（include 未展开 / 缺定义） |
| `every_expanded_shader_has_no_unresolved_symbols` | 全部 9 份展开后零未解析符号 |
| `include_expansion_is_acyclic_and_deduplicated` | 重复定义 / 循环依赖 / 残留指令 |
| `include_expansion_is_deterministic` | 同输入两遍逐字节相同 |
| `abi::scratch_addr_helpers_do_not_reenter_borrow` | wasm trap `unreachable`（RefCell 重入） |
| `abi::device_init_is_two_phase_and_gates_on_readiness` | device.init 退回单阶段（假通过） |
| `abi::ready_device_rejects_unready_handle_with_actionable_message` | 未就绪句柄滑到不可诊断的失败 |
| `t_wgsl_compile` 的 CPU 前置门禁 | 上传 GPU 前先拦结构/符号问题，报错点名文件 |


### 静态门禁：把真机缺陷拦在提交前（`wgsl_lint.mjs`）

D-14 手机真机测试抓出的 3 个缺陷，**全部是纯静态可查的**，本该在提交前被脚本
拦下。`wgsl_lint.mjs` 就是为此存在的（零第三方依赖，只用 node 内置模块）：

| 规则 | 检查什么 | 解析方式 | 抓哪个真机缺陷 |
|------|----------|----------|----------------|
| **A1** | 每个 `var<uniform>` 结构体按规范算出的 size 是 16 的倍数 | 花括号配平定位 `struct`，逐成员算 align/size；支持单行与多行 struct | 缺陷 1（对齐） |
| **A1b** | 结构体成员类型全部可解析 | 同上；遇未知类型判 FAIL 而非跳过 | 解析盲区 |
| **A2** | 宿主 uniform 缓冲字节数 ≥ WGSL 最小绑定size | 解析 `abi.rs` 的 `make_buffer(..., UNIFORM)` 实参，回查 `device.rs` 常量 | **缺陷 1 根因** |
| **A3** | WGSL 声明 payload == 宿主写入字节数（无隐式填充） | 同A2 + 结构体末成员末偏移 | 缺陷 1 的思维盲区 |
| **B1** | `glue` 对象字面量可求值 | 把对象字面量在 node 里真求值 | 胶水语法错 |
| **B2** | 胶水中**无 async 函数** | `fn.constructor.name === "AsyncFunction"`（覆盖 `async` 箭头/函数/shorthand 三种写法） | **缺陷 3（import 异步）** |
| **B3** | wasm 二进制的每个 env 导入在胶水中存在且是函数 | `WebAssembly.Module.imports()` 读真实 wasm 名单 | 导入缺失/改名 |
| **C** | 报告路径（renderReport/renderEnv/renderErrors）调用 wasm 导出前有 `typeof` 守卫 | 逐 root 展开调用图（深度 2），守卫写在任一层都算数但必须存在 | 二次抛错盖掉真错误 |
| **D1–D4** | 关键尺寸走命名常量：`BYTES_PER_U32`、`WASM_URL`、`WASM_NAME` 各只定义一次 | 剥注释后统计字面量出现次数 | 尺寸脱钩 |
| **E1–E3** | bind group 的 DataView：offset 是裸标识符、构造在循环外、`getUint32(i * BYTES_PER_U32)` 相对 view 起点 | 剥注释 + 花括号配平 | **缺陷 2（偏移双重计算）** |

**关于 A1 与 A2 的区别（重要）**：规范上 `struct Params { block: u32, count: u32 }`
在 uniform 地址空间里的 size **本来就是 16**（align = `roundUp(16, 4)` = 16，
size = `roundUp(16, 8)` = 16）。所以 D-14 真机失败的根因**不是** WGSL 少写了
成员，而是**宿主侧把 uniform 缓冲建成了 8 字节**——A2 抓的正是这个。A3 进一步
要求 payload == 宿主字节数，消除"按成员数×4 直觉算字节数"的错位（scan 的
`Params` 因此显式声明 `_pad0`/`_pad1` 两个占位）。

**门禁自身也要证明非恒真**：恒真的门禁比没有门禁更危险——它给人虚假的安全感。
故 `--self-test` 内置 7 份"故意违反"样例（把上面每个缺陷还原成原始形态），
断言规则**必须报错**；这7 例同时接进 `host_test.mjs`，每次构建都会跑。

```bash
node crates/prism-gpu/web/wgsl_lint.mjs                # 跑全部门禁（63 项），失败 exit 1
node crates/prism-gpu/web/wgsl_lint.mjs --self-test    # 只跑 7 例"故意违反"样例
node crates/prism-gpu/web/wgsl_lint.mjs --rule=A       # 只跑某条规则（可组合，如 --rule=AB）
node crates/prism-gpu/web/wgsl_lint.mjs --json         # 机器可读输出
```

实测：全部门禁 **63/63 通过**；`--self-test` **自证 7 例抓到 7 例**
（`全部规则均非恒真`）。

`build.sh` 会在编译后、`host_test.mjs` 之前自动跑门禁 + 自证，**任一失败即构建
失败**（`exit 1`）。顺序理由：门禁是秒级纯静态检查，放在产物验证之前才能在构建
阶段就红灯，不必等真机。

### 在真机上跑

门禁只覆盖静态可查的部分，**GPU 运行时行为仍必须真机验证**：

```bash
cd crates/prism-gpu/web && python3 -m http.server 8080
# 手机与电脑同一局域网，浏览器打开 http://<电脑IP>:8080/
# 需Chrome/Edge 113+ 或 Safari 18+（iOS：设置→Safari→高级→开启 WebGPU）
```

点「全部运行」，失败项点开看详情，再「一键复制报告」回传。`file://` 会被 CORS
拦住，必须走 http(s)。

## 双向 ABI

JS 调 Rust 的导出定义在 `src/web/abi.rs`（`#[unsafe(no_mangle)] pub extern "C"`，
本工具链 rustc 1.93 要求 `unsafe(no_mangle)` 包裹）。Rust 调 JS 的 14 个胶水
声明在 `src/web/device.rs` / `mod.rs`。

**错误通道**：wasm 只能返回数值，故所有导出返回 `i32` 状态码
（`0=PASS 1=FAIL 2=PENDING 3=SKIP 4=BAD_ARG`），详情写入两块线性内存：
错误缓冲 `prism_error_ptr/len`（UTF-8，NUL 结尾）与结果缓冲
`prism_result_ptr/len`。JS 用 `TextDecoder` 一次性读走。

| 导出 | 签名 | 用途 |
|------|------|------|
| `prism_abi_version` | `() -> u32` | ABI 版本号（宿主兼容性校验） |
| `prism_alloc` / `prism_dealloc` | `(u32) -> usize` / `(usize,u32) -> i32` | 线性内存往返，16 字节对齐；`dealloc(0)` 幂等 |
| `prism_error_ptr` / `prism_error_len` | `() -> usize` / `() -> u32` | 读错误详情 |
| `prism_result_ptr` / `prism_result_len` | `() -> usize` / `() -> u32` | 读结果详情 |
| `prism_test_count` | `() -> u32` | 测试项数量 |
| `prism_test_id` / `_len` / `_name` / `_len` | `(i) -> ptr / u32` | 测试项 id 与中文名（UTF-8） |
| `prism_test_needs_gpu` | `(i) -> u32` | 该项是否依赖 WebGPU（决定 SKIP 还是真跑） |
| `prism_test_run` | `(i) -> i32` | 启动一项；异步项返回 `PENDING(2)` |
| `prism_test_collect` | `(i) -> i32` | 取异步项结果 |
| `prism_shader_count` | `() -> u32` | WGSL 测试项数（库件 `rt_scene` 不计入） |
| `prism_shader_name` / `_len` / `prism_shader_src` / `_len` | `(i) -> ptr / u32` | 导出 shader 源码给 JS 编译（`src` 是**展开后**文本） |
| `prism_kernel_wgsl` / `_len` | `(i) -> ptr / u32` | 导出 D0 内核（`scan_block`/`histogram`）WGSL |
| `prism_check_wgsl` | `(ptr,len) -> i32` | 结构自检（绑定组/入口/禁 libm） |

**多 bind group 相关导出（本轮扩展，胶水实现见 `index.html` 的 `glue`）**：

| 导出/胶水 | 新签名 | 用途 |
|------|------|------|
| `prism_create_bind_group`（胶水） | `(h, pipeline, group, bufs_ptr, n)` | **4 参 → 5 参**：新增 `group` 索引。`bufs_ptr` 指向 `n` 个 buffer 句柄；第 `i` 个硬映射到 `binding: i`，整组绑到 `@group(group)` |
| `prism_dispatch`（胶水） | `(h, pipeline, bgs_ptr, n, gx, gy, gz)` | 旧签名 `(h, pipeline, bg, gx, gy, gz)` 只支持单个 bind group。现 `bgs_ptr` 指向 **`n` 个 bind group 句柄**，宿主循环 `pass.setBindGroup(i, …)`——**下标 `i` 即 `@group(i)`** |

这两处是 `rt_wavefront` 的 `generate` / `shade` / `extend` 能进判对链的前提：它们
静态引用 `@group(1)`（sobol_dir）、`@group(2)`（scene_params/materials/prim_mat/tris/lights）、
`@group(3)`（camera），旧胶水把 group 索引硬编码为 0，故此前只能 SKIP。一次 dispatch
最多设置 **4** 个 bind group（`abi.rs::BIND_GROUP_MAX`）。注意 `layout:"auto"` 仍只为
**被 dispatch 的 entry 静态引用**的 binding 生成布局，故每个 entry 的引用集合必须是
**连续前缀** `{0..n-1}`（与 group 索引无关，详见下文「连续前缀约束」）。

**清单单一真源**：采用方案 A（Rust 导出清单，JS 不硬编码 TESTS 数组），
避免 JS 与 Rust 两处清单不同步。`prism_test_run` 越界返回 `BAD_ARG`，
不 panic（C18）。

## 测试项（15 项，手机可跑完）

清单的**单一真源**是 `src/web/abi.rs` 的 `static TESTS`（`prism_test_count()`
实测 = 15）。下表第 0–8 项的说明取自该数组各 `TestItem` 的 `name` 字段，
第 9–14 项为本轮及上一轮新增。

| # | id | 需要 GPU | 说明（对齐 `TestItem::name`） |
|---|----|---------|--------------------------|
| 0 | `device.init` | 是 | 设备初始化（`navigator.gpu` / `requestDevice` / `lost` 轮询），**两阶段** |
| 1 | `cpu.scan` | 否 | CPU 参考：Blelloch scan 判对（vs 串行参考 + 树形深度） |
| 2 | `cpu.sort` | 否 | CPU 参考：radix sort 判对（vs 稳定插入排序 + 稳定性） |
| 3 | `wgsl.structural` | 否 | WGSL 结构自检：9 内嵌 shader + 库件 + 2 个 D0 内核 = **11 份**（D-29 禁 libm） |
| 4 | `abi.alloc` | 否 | ABI 自检：`alloc`/写/读回/`dealloc` 线性内存往返 |
| 5 | `shader.params` | 否 | 参数校验：ScanConfig 工作组换算 + histogram uniform 打包 |
| 6 | `wgsl.compile` | 是 | 真 WebGPU：**9 份** shader 批量 `createShaderModule`（include 已展开） |
| 7 | `gpu.scan_block` | 是 | 真 WebGPU：`scan_block` 单块 dispatch，与 CPU 参考逐元素比对 |
| 8 | `gpu.histogram` | 是 | 真 WebGPU：histogram 原子计数，与 CPU 参考逐桶比对 |
| 9 | `shader.dispatch_census` | 否 | 覆盖台账：9 份 WGSL 的 compute 入口逐份核对（可 dispatch / 数据竞争 / 多 bind group / 零入口） |
| 10 | `gpu.sim_particles` | 是 | 真 WebGPU：`sim_particles` 的 emit/integrate/hash_cells 三入口链，逐字段判对 |
| 11 | `gpu.sim_sph` | 是 | 真 WebGPU：`sim_sph` 的 sph_predict/sph_density/sph_project，与 CPU 参考（Poly6+XPBD）比对 |
| 12 | `gpu.sim_cloth` | 是 | 真 WebGPU：`sim_cloth` 的 cloth_predict/cloth_project/cloth_velocity（不相交约束）比对 |
| 13 | `gpu.rt_wavefront` | 是 | 真 WebGPU：`rt_wavefront` 的 connect/accumulate，PathSlot 布局与 spp 均值逐字节比对 |
| 14 | `gpu.rt_wavefront_multi` | 是 | 真 WebGPU：`rt_wavefront` 多 bind group 的 generate/shade/extend，group 0/1/2/3 逐字段判对 |

其中 9 项标`needs_gpu = true`（0、6、7、8、10、11、12、13、14），6 项为纯 CPU 项。
**#14（`gpu.rt_wavefront_multi`）是本轮新增**：配合扩展后的多 bind group ABI，把
`rt_wavefront` 的 `generate` / `shade` / `extend` 三个此前 SKIP 的入口真 dispatch 并
与 CPU 参考逐字段判对——实测 GPU fingerprint == CPU fingerprint == `0xc8e907af`，
mismatch 0（64 条路径槽，最大偏差 `5.960e-8`，容差 `2e-6`）。**#9–#13 是上一轮新增**：
#9 把"哪些 kernel 真的被 dispatch 过"从口头承诺变成可执行台账，#10–#13 把 D3 模拟 /
D2 wavefront 从"只编译"升级为"真 dispatch + 与 CPU 参考逐元素判对"。

设计取舍：**不**在手机上跑百万粒子 / 路径 tracing——即使能跑也只会超时，
测不出有效信息。重计算项改为「仅编译 + 参数校验」，真 dispatch 判对一律用
**小构型 + 逐元素比对**（`sim_particles` 32 粒子、`sim_sph` 16、`sim_cloth` 8、
`rt_wavefront` 2×2 像素 × spp=2）。

## 静态层 vs 真机层：两层判别力与覆盖台账

本轮最有价值的认识：**静态层全绿 ≠ 真机层全绿**。二者判的是不同的东西，
必须显式分开记账，否则"CI 绿了"会被误读成"引擎对了"。

| 层 | 谁在判 | 能判什么 | 判不了什么 |
|----|--------|----------|-----------|
| 静态层 | `cargo test` + `wgsl_lint.mjs` + `host_test.mjs` | 结构/符号/对齐/绑定偏移/清单一致性/非恒真性 | 驱动是否接受这份 WGSL、dispatch 结果数值是否对 |
| 真机层 | `realgpu_check.mjs`（有头 Chromium + WebGPU adapter） | 真 `createShaderModule`、真 dispatch、真读回逐字节比对 | —— |

本轮真机层一次性抓出 4 个静态层漏掉的真缺陷（`pendingErrors` 跨测试项泄漏等），
详见 `docs/文档八件套/施工日历.md` v4.7 条目——**这就是"两层必须分开"的直接证据**。

### 16 个 compute 入口的覆盖台账（`shader.dispatch_census`）

`sim_*.wgsl` / `rt_wavefront.wgsl` 共 **16 个 `@compute` 入口**，逐个给出处置：

| shader | 入口数 | 进判对链 | SKIP |
|--------|-------|---------|------|
| `sim_particles.wgsl` | 4 | 3（emit / integrate / hash_cells） | 1（neighbor_scan） |
| `sim_sph.wgsl` | 4 | 3（sph_predict / sph_density / sph_project） | 1（sph_velocity） |
| `sim_cloth.wgsl` | 3 | 3（cloth_predict / cloth_project / cloth_velocity） | 0 |
| `rt_wavefront.wgsl` | 5 | 5（connect / accumulate / **generate / shade / extend**） | 0 |
| **合计** | **16** | **14** | **2** |

> **本轮台账变化**：`rt_wavefront` 的 `generate` / `shade` / `extend` 随多 bind group
> ABI 落地后由 SKIP 转入判对链，`SKIP` 由 6 项中的 5 项降至 **2 项**——即从
> 「11 个进判对链 / 5 个 SKIP」变成「**14 个进判对链 / 2 个 SKIP**」。

台账是**双向**的（`t_shader_dispatch_census` 逐份核对）：

- **正向**：WGSL 里每个 `@compute` 入口都必须在台账里登记——新增 kernel 忘了登记
  ⇒ 直接 FAIL（否则就是"编译过但从未 dispatch 过"）。
- **反向**：`shaders::SHADERS` 里每份都得在台账里出现；台账登记的入口在 WGSL 里
  不存在 ⇒ FAIL；标为 SKIP 却不给原因 ⇒ FAIL。
- 标SKIP 必须写明根因，`result` 面板逐条列出。

`rt_bvh` / `rt_shade` / `rt_common` / `sim_common` / `text` 的 compute 入口数为
**0**（前四者是库件，`text` 只有 `@vertex`/`@fragment`），不参与 dispatch 判对；
其口径是"随展开后的宿主 shader 一起编译"，**已由 `wgsl.compile`（#6）覆盖**——
这也是台账把它们登记为"零入口"而非"漏登记"的原因。

### SKIP 的两类根因（都不是"跑不动"，而是"无法判对"或"胶水表达不了"）

| 类别 | 入口 | 根因 |
|------|------|------|
| **纯占位（现存唯一 SKIP 类别）** | `sim_particles::neighbor_scan`、`sim_sph::sph_velocity` | 函数体只有 `_ = i;` / `_ = sp_vel[i];`，**不写任何 storage**。dispatch 只能证明"不崩"，没有输出就没有可比对的东西——**造判据等于自证恒真**，比不测更危险。其语义等价判对由真正消费这些数据的 kernel 覆盖：`sph_density` 自己做了 O(n²) 全邻域扫描（不读 `cell_range`），`cloth_project` 覆盖了 `neighbor_scan` 预留的"邻域连续消费"语义位 |
| **需多 bind group（历史，已解决）** | `rt_wavefront::generate` / `shade` / `extend` | 曾因宿主胶水只有 `pipe.getBindGroupLayout(0)` + `pass.setBindGroup(0, …)` 而无法 dispatch——这是胶水的表达力缺口。**本轮已扩 ABI**：`prism_create_bind_group` 新增 `group` 索引参数、`prism_dispatch` 接受多 bind group 表，三个入口现由 `gpu.rt_wavefront_multi`（#14）真 dispatch 并判对，**不再 SKIP** |

**必须写清楚**：现存的两个 SKIP **不是"跑不动"，而是"无法判对"**——它们能 dispatch、
不会崩，只是函数体不写任何 storage，没有任何输出可与 CPU 参考比对，硬造一条判据
等于自证恒真。这与"宿主/胶水表达力不足"是完全不同的两类问题，不可混为一谈。

`cloth_project` 只装 4 条**两两不相交**约束（`(0,1) (2,3) (4,5) (6,7)`）也是同一
逻辑的体现：真实 4×2 网格会生成 16 条约束且相邻约束**共享粒子** ⇒ 多线程同址写 ⇒
WebGPU 内存模型下结果**非确定**，无法与任何 CPU 参考逐元素判对。被排除的 12 条
共享粒子约束在 `result` 文本中**显式列出**（不静默缩小覆盖面），并给出根因与
修法方向（按粒子着色 / 按网格行分组）。

### `layout:"auto"` 的连续前缀约束：为什么 `sim_sph`/`sim_cloth` 必须改 AoS

宿主胶水的 `prism_create_bind_group` 把句柄数组第 `i` 个**硬映射**到 `binding: i`，
而 `layout:"auto"` 只为被 dispatch 的 entry **静态引用**的 binding 生成布局。
二者相乘 ⇒每个 entry 的引用集合必须恰好是**连续前缀** `{0,…,n-1}`。

SoA 的 `pos`/`vel`/`inv` 三个独立 binding 会让各 entry 的集合**互不包含**
（如 `{pos,vel,inv,spp}` vs `{pos,density,spp}`），而前缀必嵌套 ⇒
**任何编号方案都无解**。故合并为AoS 状态槽：`pos:vec3@0` + `vel:vec3@16` +
`inv:f32@28`（恰好塞进 `vel` 的对齐填充）→ 对齐 16 → **32 字节**，前缀性质自然成立。

实测绑定前缀（`result` 面板可见）：

| 测试项 | 各入口的绑定前缀 |
|--------|-----------------|
| `gpu.sim_particles` | emit/integrate/hash_cells 均`{0,1}`（particles, params） |
| `gpu.sim_sph` | predict`{0,1}` / density`{0,1,2}` / project`{0,1,2,3}` |
| `gpu.sim_cloth` | predict`{0,1}` / project`{0,1,2,3}` / velocity`{0,1}` |
| `gpu.rt_wavefront` | connect/accumulate 均 `group(0)` 全集 |
| `gpu.rt_wavefront_multi` | generate `g0{0,1} g1{0} g2{0} g3{0}`（4 组）；shade/extend `g0{0,1} g1{0}` + `g2` 各为 5 绑定 `{0..4}`（含 lights）/ 4 绑定 `{0..3}`（不含 lights） |

> **连续前缀对每个 group 独立成立**：多 bind group 只是把「一个前缀」扩成
> 「每个 group 各一个前缀」，**不改变前缀规则本身**。这正是本轮 `rt_wavefront`
> 调整 binding 布局的原因——camera 从原来的 group 移到 `@group(3)`、`rt_scene`
> 重排为 `scene_params(0)…lights(4)`，使每个 group 内的绑定集都成为连续前缀
> `{0..n-1}`。`shade` 引用 `g2{0,1,2,3,4}`（含 lights），`extend` 只引用
> `g2{0,1,2,3}`（不含 lights，NEE 关闭），二者各自都是前缀。

> **踩坑记录**：新加一条 dispatch 链时若报"bind group 的句柄个数/顺序与该 entry
> 静态引用的 binding 不一致"，根因就是引用集合不是连续前缀——**先查这个，不要
> 去调 buffer 尺寸**。错误文本已把该根因写在第一句。

### 冷启动竞态与有限重试（D-14 真机实测）

`requestAdapter` 在页面加载后**立刻**调用时，在 **t≈36~46ms 必定返回 null**
（SwiftShader 6/6 实测）；同一时刻往后推 80ms 再调同一个函数即拿到 adapter
（6/6次），逐 25ms 轮询测得"首次成功"稳定落在 **t≈125~145ms**。同一浏览器、
同一份代码，稍晚 ~100ms 调用就一定成功 ⇒ 这是 **GPU 进程/驱动尚未枚举完适配器的
冷启动窗口**，不是逻辑缺陷。

修法（`index.html` 的 `acquireDevice()`）：**6 次封顶**尝试，退避表
`[50,150,400,800,1500] ms`（累加 **2900ms ≈ 2.9s 有界预算**），退避表长度 =
尝试次数 − 1，无不可达项，**绝不无限重试**。日志分级：重试过程落 `warn`
（冷启动噪声可见但不算故障，避免淹没真问题），预算耗尽才落 `err` 并附排查三条。
**判据不变**——预算耗尽即让 `device.init` 如实判 FAIL，绝不用重试掩盖失败。

> **为什么重试循环不在胶水里**：WebAssembly 的 import 必须是**同步**函数
> （门禁规则 B2 用 `fn.constructor.name === "AsyncFunction"` 实测拦截）。
> 故同步胶水只登记 Promise，重试循环放在 `acquireDevice()` 里由 `settle()` 侧
> await——这正是 `settle()` 的职责。
>
> **为什么必须重试而不是"拿不到就永久失败"**：真实用户会遇到完全相同的窗口
> （手机浏览器冷启动、页面从后台恢复、驱动刚 dlopen 完）。一次性失败会把**全部
> 9 个 GPU 项**带崩，而根因只是"早了 100ms"。

### 真机回归的实测结果（务必连同限制一起读）

Xvfb + SwiftShader（`vendor=google arch=swiftshader`）下 **15/15 全绿**
（本轮多 bind group 落地后实测 `PASS=15 FAIL=0 SKIP=0`）。

> ⚠️ **SwiftShader 是软件适配器，不是真 GPU。** 上述结果只证明"逻辑在 conforming
> 的软件实现上正确"，**不等于真 GPU 全绿**——真GPU 仍需在真实设备（手机）上复测。
> 真实 GPU 与软件适配器的差异集中在：驱动的着色器编译器更严格（含 FMA 收缩
> 差异）、`timestamp-query` / 子组特性可用性、内存模型对同址写的收敛性。
> 换真机前不要把 SwiftShader 的绿灯当作终局结论。

## 编译与本地验证

一条命令（编译 + 静态门禁 + 拷贝 + node 验证）：

```bash
./crates/prism-gpu/web/build.sh              # release 体积优化版（推荐）
./crates/prism-gpu/web/build.sh --debug      # 带调试信息
./crates/prism-gpu/web/build.sh --no-test    # 只编译，跳过门禁与 node 验证
```

流程：`cargo build` →拷贝产物 → **`wgsl_lint.mjs` 门禁 + 自证** →
`node host_test.mjs`。门禁失败 ⇒ **构建失败**。

产物：`crates/prism-gpu/web/prism_gpu.wasm`（实测 **286990 字节 ≈ 280 KB**）。

**必须用独立 `--target-dir`**（脚本已内置 `target/wasm32/`）：历史上踩过
wasm32 产物混进 native `target/`，导致 native 编译报
`can't find crate prism_*`。

### node 宿主验证（无需 WebGPU）

```bash
node crates/prism-gpu/web/host_test.mjs crates/prism-gpu/web/prism_gpu.wasm
```

覆盖 9 个小节共 **154 项断言**（实测 `node host_test.mjs`：通过 154，失败 0）：
线性内存往返、测试项清单、CPU 参考数值与JS 独立实现
交叉比对（fingerprint 必须逐位相等）、WGSL 源导出、uniform 布局、**静态门禁
wgsl_lint 规则 A–E 及其非恒真自证**、无 WebGPU 时不白屏不误报、以及无device
时 GPU 项返回 SKIP。**用户手机反馈前必须先过这条链。**

## 发布到手机

`.wasm` 与 `index.html` 同目录（相对路径 `prism_gpu.wasm`），因此**整个 `web/`
目录一起发布**。三种方式：

1. **发布为应用**（沙箱）：把 `web/` 目录作为应用根目录发布
2. **静态托管**：上传 `web/` 下 2 个文件即可（`index.html` + `prism_gpu.wasm`）
3. **本地起服务调试**：`cd crates/prism-gpu/web && python3 -m http.server 8080`，
   手机与电脑同一局域网访问 `http://<电脑IP>:8080/`

注意：`file://` 直接打开会被 CORS 拦住（fetch 拿不到 wasm），必须走 http(s)。

## 真机 / 有头浏览器回归（realgpu_check.mjs）

前两条链路（`build.sh` + `host_test.mjs`）都是**纯 node、无浏览器**的，验不到
"某份 WGSL 在真实驱动上能不能过 `createShaderModule`"这一层。`realgpu_check.mjs`
补的就是这一层：**有头 Chromium + WebGPU adapter**，自动跑完 15 项测试并
判定退出码。

```bash
# 一条命令：自动起静态服务 + 自动拉 Xvfb + 自动找浏览器
node crates/prism-gpu/web/realgpu_check.mjs

# SwiftShader 很慢，默认等 600s；不够就加 --timeout
node crates/prism-gpu/web/realgpu_check.mjs --timeout 900

# 已有服务/显示环境时
node crates/prism-gpu/web/realgpu_check.mjs --url http://127.0.0.1:8080/index.html
```

前置依赖（缺任何一个都会**明确报错退出**，不会静默"通过"）：

| 依赖 | 缺失时的行为 | 怎么装 |
|---|---|---|
| Playwright | 打印已尝试的解析路径后 exit 1 | `npm i -g playwright` |
| Chromium 二进制 | 打印期望路径后 exit 1 | `npx playwright install chromium` |
| Xvfb 或 `DISPLAY` | 提示装 xvfb 后 exit 1 | `apt-get install -y xvfb` |
| WebGPU adapter | 打印排查三条后 exit 1 | 见下方排查 |

无 GPU 的机器会自动回退到 SwiftShader 软件适配器（实测 `vendor=google
arch=swiftshader`），15 项仍能全跑完——这正是 CI 之外最值得反复跑的回归。
**但 SwiftShader 是软件适配器，不是真 GPU**：它的绿灯只证明逻辑在conforming
的软件实现上正确，真 GPU 仍需在手机上复测（详见上文「真机回归的实测结果」）。

**为什么"拿不到 adapter"必须失败**：如果 adapter 取不到就退化成"全部 SKIP +
报 PASS"，等于把"环境没配好"伪装成"代码没问题"——这是回归脚本最危险的失败
模式，比直接报错有害得多。本脚本对此单独判定，`0 项 PASS` 一律非 0 退出
（除非显式加 `--allow-skip` 做环境自检）。

adapter 拿不到时的排查顺序：① 无 GPU 且未回退软件适配器；② 缺
`--enable-unsafe-webgpu`（脚本已带）；③ 真机缺厂商驱动。

> **踩坑记录（务必先读）**：**不要**在 harness 页上、或在跑测试之前额外调一次
> `navigator.gpu.requestAdapter()`（哪怕在另一个标签页里）。实测这会与 harness
> 自己的 `requestAdapter → requestDevice` 时序抢同一个适配器，症状是
> `device.init` / `wgsl.compile` / `gpu.*` 误报「预留句柄未绑定到真实
> GPUDevice」——**而 15 项状态仍显示 PASS**，即"假 PASS + 面板报错"，比直接失败
> 更难发现。同理，探测页不能用 `about:blank`（非安全上下文，`navigator.gpu`
> 恒为 undefined，会把"有 GPU"误判成"没 GPU"）。
>
> 因此 `realgpu_check.mjs` 的顺序是：① 同源专用探针页**只查**
> `typeof navigator.gpu !== "undefined"`（不申领适配器）→ ② 关掉探针页，开
> **全新** harness 页跑 15 项 → ③ 跑完后再申领一次 adapter，仅作环境诊断。
> 改动这个脚本时请保持该顺序。

配套的 CI 静态校验脚本（都在 `web/` 下，零第三方依赖，CI 会调）：

| 脚本 | 作用 | 谁调它 |
|---|---|---|
| `ci_wasm_verify.mjs` | 产物存在 / 体积合理 / 21 个必需导出（共 26）+ 14 个必需导入符号齐全 | CI `wasm` job |
| `ci_wgsl_smoke.mjs` | `shaders/*.wgsl` 基础解析（注释/字符串闭合、括号配对） | CI `wasm-verify` job |

`ci_wgsl_smoke.mjs` **只做基础解析**，绑定组/符号/类型层面的深度校验由
`wgsl_lint.mjs` 负责，两者不重复实现。注意目录里现为 **10 份** `.wgsl`
（9 份独立测试项 `shaders::SHADERS` + 1 份 `//!include` 库件 `rt_scene`
即 `shaders::LIBRARIES`），脚本按目录实际内容枚举、
不写死数量，所以"新增一份 shader"不会让 CI 无端变红，而"误删全部"一定会红。

## 桌面 GPU 性能验收（perf_check.mjs + perf.html）

`realgpu_check.mjs` 判的是**正确性**（跑完测试项、状态对），**不测吞吐**。桌面
GPU 的**性能验收**由新增的 `perf_check.mjs` + `perf.html` 负责——这是本轮新增的
第三条链路，与正确性回归互补但**不重叠**。

```bash
node crates/prism-gpu/web/perf_check.mjs            # 自动起服务 + Xvfb + 找浏览器
node crates/prism-gpu/web/perf_check.mjs --help     # 完整用法
```

用法：

```
node perf_check.mjs                        # 自动起静态服务 + 自动找浏览器
  --url <URL>  --timeout <s>  --frames <n>  --warmup <n>
  --require-gpu  --json  --self-check  --help
```

`--timeout` 默认 600s（SwiftShader 很慢），`--frames` 默认 30，`--warmup` 默认 5，
`--require-gpu` 令软件适配器直接判失败，`--self-check` 只跑确定性 + adapter 探测。

**退出码**（无静默通过路径）：**0** = 真 GPU 且确定性成立（可作验收）；**1** =
前置/确定性/页面失败；**2** = 用法；**3** = **软件适配器（结果不可作验收凭据）**。

设计要点：

| 要点 | 说明 |
|------|------|
| **确定性前置门禁** | 测速**前**先同输入跑两遍，**逐字节比对 + FNV-1a32 指纹**；不成立直接失败退出，**不测速**——确定性都不成立的性能数字没有意义 |
| **软件适配器显式识别** | 自动识别 `swiftshader` / `lavapipe` / `llvmpipe` 等软件适配器，显式警告 + 退出码 **3**；`--require-gpu` 时直接判失败 |
| **计时口径** | 每帧耗时 = `performance.now()` 差，**包住一次完整 run→settle→同步读回**；预热 5 帧后采样 30 帧，报告 min/median/mean/p95，帧率取 median |
| **绝不伪装 PASS** | **绝不允许"拿不到 adapter 就 SKIP 全部然后报 PASS"**——这正是最危险的失败模式，本脚本单独判定并强制非 0 退出 |

### 能力缺口（当前 ABI 无法表达目标规模，**必读**）

脚本在沙箱只能测**小构型吞吐**，**不可线性外推**——小规模下每帧耗时被固定开销
（dispatch + 一次同步读回）主导，推不出大规模数字。两个目标规模当前**无法表达**：

| 目标 | 为何无法表达 | 需要的接口 |
|------|-------------|-----------|
| `rt_wavefront@1080p` | 构型写死为编译期常量 `WF_WIDTH=WF_HEIGHT=2`、`WF_SPP=2`，JS 侧**无法指定分辨率** | 新增导出 `prism_rt_run(width, height, spp, frames)` |
| `sim_particles@1e6` | 粒子数写死为编译期常量 `SIM_P_COUNT=32`、`PARTICLES_BYTES=1536`，dispatch 工作组数也按它推出 | 新增导出 `prism_sim_particles_run(count, frames)`；百万粒子还需确认 `maxStorageBufferBindingSize ≥ 1e6×48≈48MB`（超出须分块） |

> ⚠️ **两条必须钉死的结论**：
> 1. **SwiftShader 全绿 ≠ 真 GPU 全绿**。沙箱只有 SwiftShader（软件适配器），
>    因此 **`perf_check.mjs` 在沙箱跑出的性能数字不构成验收结论**（脚本会以退出码
>    3 自报这一点）。性能验收必须在**真实桌面 GPU** 上跑，退出码 0 才算数。
> 2. **小构型吞吐 ≠ 目标帧率**。脚本测得的是固定小构型（2×2×spp2 / 32 粒子）的
>    吞吐，**不可线性外推**为 1080p 或百万粒子数字。

## 本页数字的复算方式（防文档再次漂移）

本页所有数字都**可由下面5 条命令当场复算**。改动 harness 后请一并复算，
**不要凭记忆改数字**——文档数字一旦漂移就不再被信任，这是 D-14 踩坑记录的
同源教训（静态层全绿却与真机脱节）。

```bash
cd /workspace/media-engine
cargo test -p prism-gpu --lib 2>&1 | tail -3                  # 85 passed（wasm 侧单测）
node crates/prism-gpu/web/host_test.mjs 2>&1 | tail -3         # 通过 154 项，失败 0 项
node crates/prism-gpu/web/wgsl_lint.mjs 2>&1 | tail -3         # 通过 63/63
node crates/prism-gpu/web/wgsl_lint.mjs --self-test 2>&1 | tail -3  # 自证 7 例，抓到 7 例
cargo test --workspace 2>&1 | grep -c '^test result: ok'       # 138 个测试结果全 ok（lib+集成+doc）
```

测试项数量不必数文档——直接问wasm（清单单一真源，`prism_test_count()` = 15）：

```bash
node -e 'const fs=require("fs");
const m=new WebAssembly.Module(fs.readFileSync("crates/prism-gpu/web/prism_gpu.wasm"));
const env={memory:new WebAssembly.Memory({initial:300,maximum:300})};
for(const i of WebAssembly.Module.imports(m)) if(i.kind==="function") env[i.name]=()=>0;
WebAssembly.instantiate(m,{env}).then(e=>console.log("测试项数 =",e.exports.prism_test_count()));'
```

compute 入口数也不必数文档——`shader.dispatch_census`（#9）会自己核对，
台账与 WGSL 不一致就 FAIL；`result` 面板直接给出 `COVERED 14 个 / SKIP 2 个`。

