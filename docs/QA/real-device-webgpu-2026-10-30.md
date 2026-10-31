# 真机实测记录：WebGPU 浏览器 Harness（2026-10-30）

> **性质**：V2.2-A-1「WebGPU 渲染路径」的**实机验证**档证据（PRD §3.0.1 第二档）。
> 该档只登记、不构成版本 Go/No-Go 条件；本记录是唯一一次真实 GPU 设备的判据产出，
> 用于把「代码就绪」与「实机可用」两件事分开记账。
>
> **执行方式**：手机浏览器打开发布链接 → 点击「全部运行」→ 导出完整报告回传。
> 页面为 `crates/prism-gpu/web/index.html`（V1.0 起的 harness，测试项清单由 Rust `abi.rs::TESTS` 单一真源驱动）。

## 1. 测试环境

| 项 | 值 |
|---|---|
| 时间 | 2026-10-30T12:35:31.704Z |
| UA | `Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/30.1 Chrome/143.0.0.0 Mobile Safari/537.36` |
| 浏览器族 | **Samsung Internet 30.1**（Chromium **143** 内核）——**非 Chrome/Edge 品牌** |
| 操作系统 | Android 10 |
| `navigator.gpu` | `available` |
| adapter info | `{}`（**空对象**：该浏览器不暴露 adapter 详情，无法记录 GPU 型号/厂商） |
| wasm | `423,855 B`，`WebAssembly.instantiateStreaming`，ABI v1 |
| 页面测试项数 | 17（16 项既有 + 本次新增 `gpu.scene_frame`） |

> **为什么 adapter 名记不到**：`GPUAdapter.info` 在该浏览器实现下返回空对象。这是浏览器侧的隐私/实现限制，
> 不是本仓代码问题。**故本记录无法给出 GPU 型号**，只能记「Android 10 + Chromium 143 内核 + 真 GPU adapter 可用」。
> G8 口径纪律要求如实标注，不得因为「反正是真机」就含糊为「Chrome/Edge 已验」。

## 2. 结果汇总：16 PASS / 1 FAIL / 0 SKIP

| # | 测试项 | 结果 | 备注 |
|---|---|---|---|
| 1 | `device.init` | **PASS** | `requestDevice` 成功，句柄=1，`lost=0` |
| 2 | `cpu.scan` | PASS | Blelloch vs 串行参考，树深 7 = ⌈log₂128⌉ |
| 3 | `cpu.sort` | PASS | radix vs 稳定插入排序，稳定性逐桶核 |
| 4 | `wgsl.structural` | PASS | 17 个 WGSL 结构自检全通过（15 shader + 2 个 D0 内核） |
| 5 | `abi.alloc` | PASS | alloc/write/readback/dealloc 64 B 往返 |
| 6 | `shader.params` | PASS | ScanConfig 工作组换算 + histogram uniform 打包 |
| 7 | `wgsl.compile` | PASS | **真机** `createShaderModule` 15/15 编译通过 |
| 8 | `gpu.scan_block` | PASS | 128/128 逐位一致 |
| 9 | `gpu.histogram` | PASS | 256/256 桶一致 |
| 10 | `shader.dispatch_census` | PASS | COVERED 18 / SKIP 6，与 WGSL 逐份一致 |
| 11 | `gpu.sim_particles` | PASS | 3 入口链 32/32 粒子，pos/vel 偏差 = 0.000e0（容差 2e-6） |
| 12 | `gpu.sim_sph` | PASS | 3 入口链，pos/vel/density/lambda 偏差 = 0.000e0 |
| 13 | `gpu.sim_cloth` | PASS | 3 入口链，偏差 = 0.000e0 |
| 14 | `gpu.rt_wavefront` | PASS | connect 768 B **逐字节未变**；accumulate 偏差 = 0.000e0 |
| 15 | `gpu.rt_wavefront_multi` | PASS | 多 bind group（group 0/1/2/3）64 槽逐字段，最大偏差 1.192e-7 |
| 16 | **`gpu.scene_frame`** | **PASS** | **本次新增项，A-1 核心判据 → 见 §3** |
| 17 | `gpu.gi_compute` | **FAIL** | 真机暴露的既有缺陷 → 见 §4 |

**环境健康度结论**：15 个既有项中 14 项 PASS，且包含同样是「多 entry 链 + storage AoS 对拍 + 多 bind group」的
`gpu.sim_*` / `gpu.rt_wavefront_multi`。**故 `gpu.gi_compute` 的失败不是环境问题，是该项自身的孤立缺陷。**

## 3. V2.2-A-1 核心证据：`gpu.scene_frame` 真机 PASS

`scene_frame.wgsl` 的 `scene_main`（栈式 BVH，V2 六绑定）在真机 GPU 上**真 dispatch 并读回**，
产物与 CPU 参考统计对拍：

| 场景 | 读回字节 | 判据 1：`== w*h*4` | 判据 2：非全零 / 色数 | 判据 3：PSNR | 判据 4：双跑逐位 | 判据 5：曲线被消费 |
|---|---|---|---|---|---|---|
| **A**（球+灯，8×8） | 256 B | ✅ `256 == 8*8*4` | ✅ 8 色（max_channel_delta 1，differing 63/64） | ✅ **52.5 dB** | ✅ 一致 | —（A 无曲线） |
| **B**（球+圆柱+灯，16×16） | 1024 B | ✅ `1024 == 16*16*4` | ✅ 7 色（max_channel_delta 1，differing 255/256） | ✅ **52.9 dB** | ✅ 一致 | 与 **BC**（B 去曲线）差 **10/256** 像素，`max_channel_delta=114`；CPU 侧同对照亦 10/256 → **两侧同向** |
| **BC**（B 去曲线） | 1024 B | ✅ | ✅ 5 色 | ✅ 52.9 dB | 对照单跑 | 同上 |

### 3.1 逐条判据对照 PRD 验收第 3 条

| PRD 验收 3 要求 | 实测 |
|---|---|
| 有设备 → 返回 `is_valid()` 帧且长度 `w*h*4` | ✅ 三帧字节数精确等于 `w*h*4` |
| 同设备双跑逐字节一致 | ✅ A、B 各连跑两次，读回**逐位一致** |
| vs CPU 统计对拍（G3 双门禁口径） | ✅ 最低 **52.5 dB** ≫ 门限 40 dB |

### 3.2 PSNR 天花板来源（如实记录，不修饰）

> 真机 PSNR 落在 48–53 dB 区间而非「无限高」，原因是**量化口径差异，非渲染错误**：
> CPU 侧量化用 `(v*255).round()`，WGSL 侧用 `u32(v*255)` **截断**，故多数像素存在 ±1 级系统性差。
> 这是 PSNR 天花板的主要来源，远高于 40 dB 门限。
> **未因此调整门限，也未把 WGSL 改成 `round` 去「刷分」**——真机数据如实保留。

### 3.3 A-1 实机档结论

PRD §3.0.1 对 V2.2-A-1 的实机验证档要求是：**「Chrome/Edge ≥2 设备出 `w*h*4` 有效帧
+ 同设备双跑逐字节一致 + vs CPU 统计对拍」**。

| 子条件 | 状态 |
|---|---|
| 出 `w*h*4` 有效帧 | ✅ 通过 |
| 同设备双跑逐字节一致 | ✅ 通过 |
| vs CPU 统计对拍（≥40 dB） | ✅ 通过（52.5 / 52.9 dB） |
| **≥2 设备** | ❌ **仅 1 台** |
| **Chrome/Edge 品牌** | ⚠️ **非该品牌**：Chromium 143 内核的 Samsung Internet |

> **判定：进度 1/2 设备，且浏览器族为「Chromium 143 内核（非 Chrome/Edge 品牌）」。
> 故 A-1 的实机档整体仍为「已交付待验」，不得改写为已完成。**
>
> ⏭ **本节是当时的判定，保留原样不改写**。最终结论见 **§9**（已推进至 4 个浏览器环境各 19/19 全 PASS，硬件适配器已确认于 §9.10）。
>
> 对外口径维持：「限 Chrome/Edge / 未真机验证」→ 本记录允许追加一句更精确的：
> 「已在 1 台 Chromium 143 内核移动端真机验证（Samsung Internet，Android 10），
> 核心三项判据全通过；≥2 设备与 Chrome/Edge 品牌项仍待验」。
> **不得**简化为「WebGPU 已支持」或「Chrome/Edge 已验证」。

## 4. `gpu.gi_compute` 真机 FAIL（**根因已定位并修复，见 §7**）

### 4.1 原文

```
[FAIL] gpu.gi_compute :: 真 WebGPU：GI 的 GPU 路径 gi_probe_trace/gi_sh_project/gi_probe_interpolate，
                          逐字段与 CPU（prism-rt 同式）判对
      gpu.gi_compute：GI GPU 结果与 CPU 参考不一致——60 个字段超容差 2e-6；
      首个 SH 不一致 probe=0 k=0 ch=0：GPU 0 vs CPU 0.9970052
```

### 4.2 已确认的事实

1. **该项是 V1.0 遗留测试项**，此前**从未在真机跑过**（native 下 `Status::Skip`）。
   本次是它第一次面对真实 GPU。→ 这是「17 项里 16 项第一次一起上真机」暴露的第一个问题。
2. **GPU 侧整块零值**：`samples` / `sh` / `irradiance` 全 0，而 `pos` / `normal` 读回**完好**
   （说明 buffer 句柄身份正确、布局正确）。
3. **`60` 这个数字已逐项对上**：CPU 参考侧 probe 1 的 `sh` **本身就是全零**
   （起点 x=1.001 落在图元内部 → 8/8 全遮挡），故 GPU 全零时的期望不一致数 =
   `probe0(27) + probe1(0) + probe2(27) + point0(3) + point1(3) + point2(0) + point3(0) = 60` ✅
   → **`probe 2` 也是错的，`probe 1` 是「零对零」掩盖**。这不是算术漂移，是整块零。
4. **已静态排除**（附判据，不必重查）：
   - 字节布局三处全对（`GiProbe` 304 / `GiParams` 256 / `GiPrim` 48）；
   - `max_steps`/`hit_eps` 读 0 → 会得 **81** 处不一致，不是 60；
   - `prims` 读全零 → 也是 81；
   - `GiProbe` 步长漂移 → `pos`/`normal` 会读错，与实测「完好」矛盾；
   - pipeline / bind group / 队列顺序 → `gpu.sim_particles` **结构同形**（同 module 三 entry、
     `layout:auto`、`{storage@0, uniform@1}` 连续前缀、entry2/3 读 entry1 写的同一 buffer）且真机 PASS。

### 4.3 剩余两族（宿主读回字节完全相同，无法分辨）

- **(A) 三个入口都没真正写入**：uniform 未生效（`probe_count == 0` → 三入口全早退），
  或 bind group 绑到了别的/全零 buffer（`nlen < 0.5` → 两个入口早退）。
- **(B) 写入生效但全方向判为遮挡**：本场景可静态推导出的必要条件**极窄**——
  probe 0 有 4 个方向 `world.x < 0`（永远远离图元，CPU 侧永不命中），只有「t=0 即命中」才能判遮挡；
  t=0 时三个起点的场景距离是 `1.0 / -1e-3 / 0.999` ⇒ 需 `hit_eps ≥ 1.0`，或需额外图元参与 `min`。
  - `GiParams` 里唯一 ≥1.0 的标量 = `t_max@48 = 20.0` → 疑 `hit_eps` 读到 `t_max`；
  - 唯一能凑成「第 3/4 号图元也参与」的值 = `prim_count == 4` = `point_count@4` 的值，
    而 `prims[2]/[3]` 正是 `radius=1, center=原点`。

### 4.4 处置

- **未修**，也**未降级为 Skip、未放宽 `F32_TOL`、未改判据**（红线）。
- 已在 `c_gpu_gi_compute` 的 FAIL 分支加**零成本诊断块**（纯解析既有读回缓冲，不新增读回）：
  逐 probe 打印 `pos`/`normal`/`|n|`/逐方向 `samples` 模长/可见方向数/`sh[0]`，并对三个 entry
  分列中间态（trace → project → interpolate），末尾给 `[判定]/[研判]/[下一步]`。
  `PASS` 路径不输出，不污染正常报告。诊断块有单测钉住（三种读回形态各落正确分支）。
- **下一轮真机重测**（诊断版已发布，wasm `432,052 B`）即可把 (A)/(B) 分开。
  若仍不可分辨，再允许在 `gi_probe_trace` 入口写 1 行 witness（`probe._pad0 = 1.0`，
  不改 304 B 布局、不动判据）。

## 5. 附带发现：`gi_prim_dist` 球分支语义错误（P1，未修）

`shaders/gi_compute.wgsl::gi_prim_dist` 的 `kind==0`（球）分支 `return q.x;`，
其中 `q = abs(p - center) - radius`，即返回 `|p.x - c.x| - r`。

> **这不是球 SDF，是沿 x 的一张无限大板。**
> 注释声称「q 的三分量相等，任取一分量即 |p−c|」——**该理由不成立**：
> `abs(p-c) - r` 的三分量仅在「只有一个坐标有差」时才相等。

- **不是本次真机 FAIL 的根因**：CPU 参考与 WGSL **同式**，故不构成跨端差异。
- **但它是一个真缺陷**，后果有三：
  1. 测试场景里名义上的「球」实际是板，**`kind==0` 的球 SDF 从未被测到**；
  2. `probe 1` 的 CPU 参考「恰好全零」、以及不一致数「60 而非 81」都直接源于此；
  3. 名称与语义不符（`gi_prim_dist` 声称球/盒 union，实测板/盒 union）。
- **本轮不动**：修它会改变 CPU 参考值与测试语义，属变更判据，须单独一轮并同步
  WGSL + CPU 参考 + 相关断言。
- 另记：`GiPrim._pad: vec3<u32>` 的真实偏移是 **@32**（`vec3` 对齐 16），文档写的 @28 不准确；
  因该字段从未写入，`size` 仍为 48，无害。建议随下次 shader 变更一并订正注释。

## 6. 台账影响

| 台账 | 影响 |
|---|---|
| V2.2-A-1 实机档 | **1/2 设备**：核心三项判据全通过（`w*h*4` / 双跑一致 / 52.5·52.9 dB），但「≥2 设备」与「Chrome/Edge 品牌」未满足 → **仍为「已交付待验」**（→ 已由 **§9** 推进：4 个浏览器环境各 19/19 PASS；品牌项字面满足、「≥2 设备」满足、硬件适配器已直接确认） |
| `gpu.gi_compute` | 真机 FAIL，**新增待修项**。该项属 V1.0 遗留、非 V2.2 新增，但「GI GPU 路径与 CPU 判对」这一能力声称在真机上**当前不成立** |
| `gpu.scene_frame` | 新增项，真机 PASS（本记录 §3 为唯一证据） |
| 其余 14 项既有测试 | 首次真机批量验证，全 PASS（构成「真机环境健康」的旁证） |

---

## 7. 根因定位（同日补测，结论：**V1.0 的 PASS 是巧合，测试本身写错了**）

### 7.1 决定性对照实验

作者指出「V1.0 是全部跑过且全部 PASS 的」——这句话把问题从「两个盲人摸象」变成了
**回归定位**。于是先做了一件最该做的事：**用 V1.0 原件在软件适配器上复现真机故障。**

| 组 | 源码 | wasm | 环境 | `gpu.gi_compute` |
|---|---|---|---|---|
| A | **修复前**（错误球 SDF `q.x`） | 432,052 B | SwiftShader（`realgpu_check.mjs`） | **PASS**（17/17 全绿） |
| B | 与 A **同一份源码**重新构建 | 432,052 B（字节可比） | **真机** Android 10 / Chromium 143 内核 / 真 GPU | **FAIL**，GPU 侧整块零值 |
| C | **修复后**（真球 SDF + 安全填充 + 越界守卫） | 439,539 B | SwiftShader | **PASS**（18/18 全绿，含 `gpu.params_roundtrip`） |

**A 与 B 是同一份代码、同一量级的产物，却在两种 GPU 上给出相反结论。**
这一条就把「跨端基础设施有 bug」整族排除了：绑定、布局、dispatch、读回全部正常。

### 7.2 机制：病态 SDF 的球 tracing

`gi_prim_dist` 的球分支返回 `q.x = |p.x − c.x| − r`，**只与 x 有关**——是沿 x 的无限大板。

- 射线若沿 y/z 前进，该"距离"**不随 t 变化**（梯度恒为 0）。sphere tracing 的
  `t += d · safety` 于是给出恒定步长，能否在 `max_steps=48` 步内跨过 `t_max=20`
  取决于每一步的浮点舍入累积。
- 这是一个**数值病态**的迭代：结果对 `abs`/减法/FMA 收缩的实现细节极度敏感。
- **软件适配器（SwiftShader）的浮点行为接近 CPU 参考**，两端"同错得一致"→ 判据恒绿。
- **真 GPU 的 FMA 收缩策略与精度路径不同**，同一份代码整体翻到"全遮挡"一侧 →
  `samples` 全 0 → `sh` 全 0 → `irradiance` 全 0 → 真机观测到的整块零值。

> **`V1.0` 那次"全部 PASS"是碰巧落在与 CPU 同一侧，不是验对了。**
> 作者的判断——「那说明可能 V1.0 的测试写错了」——是准确的。

### 7.3 连带缺陷：容量填充图元不安全

`GI_PRIMS` 容量为 4，其中 `[2]/[3]` 是「球心在原点、r=1」的填充（`prim_count=2` 时不用）。
一旦 `prim_count` 在任一环节被读成 4，这两个球会把探针起点判成"在球内"
（`|p| − 1 ≈ 5e−7 ≤ hit_eps`）→ 该探针 8/8 全遮挡 → 同样的整块零值形态。

已用 CPU 参考实测复现（回归钉子 `gi_prim_count_misread_does_not_blind_the_scene`）：

| 配置 | `prim_count=2` | `prim_count=4` |
|---|---|---|
| 旧球 SDF + 旧填充 | 可见 `[5, 0, 1]` | **`[0, 0, 0]`（三探针全遮挡、SH 整块零）** |
| 新球 SDF + 旧填充 | — | `[8, 4, 0]`（只杀 probe 2） |
| **新球 SDF + 安全填充（出库）** | `[8, 4, 4]` | `[8, 4, 4]`（逐位相同） |

> 注意一个必须说清的细节：`prim_count=4` 造成"三探针全 8/8 遮挡"的**必要前提包含
> 旧球 SDF**（`|x|−1` 的板同时含住三个起点）。球 SDF 改对后，同一个"原点 r=1"
> 只杀 probe 2。**两处都改是必需的**：球 SDF 保证"球真的被测到"，安全填充保证
> "即使误读容量也无害"。

### 7.4 修复内容

1. **球 SDF 修正**（CPU `gi_prim_dist_ref` 与 WGSL `gi_prim_dist` **同步**）：
   `q.x` → `length(p − c) − r`。两端必须一起改，否则跨端判对失去意义。
2. **容量填充改为不可能命中**：`radius 0` + `center = (0, −1e6, 0)` → SDF 恒 ≈1e6 > `t_max`。
3. **越界守卫**：`gi_scene_dist` / `gi_probe_trace` / `gi_sh_project` /
   `gi_probe_interpolate` 的计数上界全部 `min(count, 容量/arrayLength)`。纯防御，正常路径数值一分不变。
4. **新增 `gpu.params_roundtrip`**：把 `GiParams` 的 256 B 写进 uniform 再读回，
   **逐字节比对 + 4 个关键字段的解释值**（`probe_count@0=3`/`point_count@4=4`/
   `prim_count@8=2`/`n_dirs@12=8`）。这是"参数是否真的到达 GPU"的直接判据——
   若日后再出现"整块零值"，一眼可分是"没传进去"还是"传进去算错了"。
5. **哨兵**：`samples` 预填 `−1`、`sh` 预填 `−2`、`irradiance` 预填 `−3`。
   FAIL 时读回值可三分：`−1` = 入口没执行 / `0` = 执行了但全遮挡 / `env` = 可见。
6. **`WASM_URL` 加内容指纹**（`?v=<sha256 前 8 位>`，`build.sh` 幂等注入）：
   此前无版本参数，浏览器强缓存会让"改了但测不出来"——本轮作者重测仍报同一条错，
   极可能就是这个原因。

### 7.5 方法论结论（写给后来者）

> **「CPU 参考 vs GPU 镜像」的自洽对拍，只有在两端算法本身正确时才有判别力。**
> 两端同错时它恒绿；而**软件适配器（SwiftShader）与 CPU 的浮点行为接近，
> 永远无法暴露这一类错误**——只有真硬件 GPU 能。
>
> 推论：凡是判据建立在"两端一致"上的测试项，**必须至少有一条真机证据**，
> 否则它证明的只是"两份同样的错误实现互相同意"。`gpu.gi_compute` 保留「真机档」、
> 不降级为软件适配器档，正是这个理由。

### 7.6 当前状态

| 项 | 状态 |
|---|---|
| 修复后 SwiftShader（`realgpu_check.mjs`） | **18/18 PASS**，含 `gpu.gi_compute`（可见 16/24）与 `gpu.params_roundtrip`（256/256 字节全等） |
| `cargo xtest` | **4028 passed / 0 failed / 0 ignored / 188 段**（+4 条新回归钉子） |
| `build.sh` | **395 通过 / 0 失败**；wasm 439,539 B（上限 458,752） |
| **真机复测** | ~~待测~~ → **已完成**：`?v=54d66a8e` 19/19 PASS，见 **§9** |


---

## 8. 真机复测（同日第二轮，设备/浏览器更换）—— 暴露第二个根因：`vec3<f32>` 布局错位

### 8.1 测试环境（第二台设备）

| 项 | 值 |
|---|---|
| UA | `Android 12; SM-G9730; Chrome/144.0.7559.86 Quark/10.18.6.1161 Mobile` |
| WebGPU | available（真 GPU） |
| wasm | `439539B`（= W7 修复版，`?v=5c99f623`） |
| 结果 | **17 PASS / 1 FAIL / 0 SKIP（新增 `gpu.params_roundtrip` PASS）** |

**`gpu.params_roundtrip` PASS 是本轮最有价值的一条否定证据**：256/256 字节回环全等、
`probe_count@0=3 / point_count@4=4 / prim_count@8=2 / n_dirs@12=8` 全部正确 ⇒
**uniform 传输链无问题，`prim_count` 误读假设彻底排除**。

### 8.2 诊断块给出的决定性数据

```
probe 0: pos GPU(0.000,1.000,0.000) ✓  normal GPU(-1.000,-1.000,-1.000)   ← Rust 写的是 (0,1,0)
          可见 GPU 2/8 vs CPU 8/8
          GPU |s|[0.000 0.000 1.393 0.000 1.393 0.000 0.000 0.000]
probe 1: pos GPU(1.000,1.000,0.000) ✓  normal GPU(0.900,0.800,0.700)       ← Rust 写的是 (1,0,0)
          GPU |s|[1.393 0.000 1.393 0.000 0.000 0.000 -3.464 -3.464]      ← 末两格 = |(-2,-2,-2)|
probe 2: pos GPU(0.000,1.000,0.000) ✓  normal GPU(-1.000,-1.000,-1.000)
```

三个值互相印证：

| 观测值 | 它是谁 | 出现在哪个字段 |
|---|---|---|
| `(-1,-1,-1)` | samples 哨兵 `GI_SENTINEL_SAMPLE` | `GiProbe.normal`（应 `@16`） |
| `(0.9,0.8,0.7)` | `GiParams.env_radiance`（`@16`） | `GiProbe.normal` |
| `-3.464 = |(-2,-2,-2)|` | sh 哨兵 `GI_SENTINEL_SH` | `GiProbe.samples[6]`（`@128`，而 `sh[0]` 在 `@160`） |

**哨兵值跨字段出现 ⇒ 结构体内部偏移整体重排，不是数值问题。**

### 8.3 判定实验（沙箱内可做，结论确定）

`gi_pack_probes()` 打包后的 buffer 字节经 dump **完全正确**（三探针 normal 分别为
`(0,1,0)` / `(1,0,0)` / `(1,0,0)`，pos 全对，samples 全 `-1`，sh 全 `-2`）。
再用真实 Rust 参考实现跑**候选 normal**，比对**逐方向位掩码**：

| 候选 normal | probe0 位掩码 | 与真机 GPU `[0,0,1,0,1,0,0,0]` |
|---|---|---|
| **`(-1,-1,-1)`** | **`[0,0,1,0,1,0,0,0]`** | **逐位完全相同 ✅** |
| `(0,1,0)`（Rust 写的正确值） | `[1,1,1,1,1,1,1,1]` | 不吻合 |
| `(0.9,0.8,0.7)`（`env`） | `[0,1,0,1,1,0,1,1]` | 不吻合 |
| `(0,0,1)`（紧凑布局假设） | `[0,1,0,1,1,0,1,0]` | 不吻合 |

**结论：GPU 读 `probe.normal` 时读到的是 samples 区的哨兵 `-1`。**

### 8.4 根因

**真机驱动对结构体成员 `vec3<f32>` 的 `size` 处理与 WGSL 规范不一致**（规范
`size=12 / align=16`；真机表现为更大），导致后续字段偏移整体重排：
`normal`（应 `@16`）被推到 `samples` 区（`@32`），`samples`/`sh` 被压缩
（`samples[6]` 落到 `sh[0]` 原应所在的 `@160` 之前的 `@128` 起算位置）。

**旁证（强，且是独立维度的证据）**：

| shader | 存储布局写法 | 真机结果 |
|---|---|---|
| `gi_compute.wgsl` | **全仓唯一**用 `array<vec3<f32>, N>` 做布局 | **唯一 FAIL** |
| `rt_wavefront` / `sim_particles` / `sim_sph` / `sim_cloth` / `scene_frame` | 全部 `array<vec4<f32>>` | **全部 PASS** |

`rt_wavefront.wgsl:79` 的 `array<vec3<f32>, 2>` 是**函数返回值**（值类型，不进 buffer 布局），不受影响。

### 8.5 修复：把所有参与存储布局的 `vec3<f32>` 换成 `vec4<f32>`

`vec4` 的 `align/size` 都恒为 16，与原「`vec3` + 显式/隐式填充」**逐字节等价**，
故 **Rust 侧全部偏移常量一字节未改**（`gi_pack_probes` / `gi_params_bytes` /
`gi_unpack_sh` 的偏移行根本没出现在 diff 里）。四个结构体：

| 结构体 | 变化 | size |
|---|---|---|
| `GiProbe` | `pos`/`normal` → `vec4`；`samples`/`sh` → `array<vec4,N>` | 304（不变） |
| `GiPoint` | `pos`/`irradiance` → `vec4` | 32（不变） |
| `GiParams` | `env_radiance`/`albedo` → `vec4` 并删掉其后 `_pad0`/`_pad1` | 256（不变） |
| `GiPrim` | `center` → `vec4`；`_pad: vec3<u32>` → `_pad1: vec4<u32>` | 48（不变） |

shader 内对 vec3 字段的读取一律加 `.xyz`，写入一律 `vec4(xyz, 0.0)`；`w=0` 不参与任何运算，**数学语义一分不改**。

### 8.6 新增真机判读锚点：`gpu.gi_layout`

新增 shader 入口 `gi_layout_census`（`@workgroup_size(1)`）+ 测试项 `gpu.gi_layout`：
它把 shader **自己读到的** 8 个字段原样写进 `probes[0].samples[0..7]`，宿主读回后与
Rust 打包值**逐格逐分量**比对（含 `w` 必须为 0）。

7 条硬判据，每条对应一种布局失效模式：

| 格 | 内容 | 失效含义 |
|---|---|---|
| `samples[1]` | `probe0.normal` = `(0,1,0)` | **本轮 bug 的直接判据**（读到 `-1` ⇒ vec3 布局仍未修正） |
| `samples[3]` | `probe2.normal` = `(1,0,0)` | `array<GiProbe>` 的 stride 304 未生效 |
| `samples[2]`/`[4]` | `probe2.pos` / `probe1.pos` | 跨元素读取正确 |
| `samples[6]` | `(3, 2, 0.9, 0.8)` | `GiParams` 的 u32 与 vec4 字段混合偏移错 |
| `samples[7]` | `0`（Rust 预填 `-1`） | **证明 8 格确由本 entry 覆写**，排除"没执行却判绿" |

### 8.7 静态门禁（把教训固化成门禁，防回归）

- `web/wgsl_lint.mjs` **规则 H**：struct 体内 `array<vec3<f32>>` 判红；`var<storage|uniform>`
  顶层 `array<vec3<f32>>` 判红；其余命中按「值类型语境」白名单（仅
  `rt_wavefront.wgsl` 3 处：`cast_ray` 返回类型 + 两处 `return array<vec3<f32>,2>(...)`），
  且**白名单处数必须精确相等**（多一处/少一处都判红）。自证 46 例抓到 46 例。
- `web/host_test.mjs` 第 4 节同口径 per-shader 断言。
- `web::abi::tests::gi_wgsl_layout_matches_abi_constants`：**读 WGSL 源码按规范算逐字段
  偏移与 size**，与四个 `GI_*_STRIDE_BYTES` 常量交叉核对（已两次反向演练证明非恒真）。

### 8.8 当前状态

| 项 | 状态 |
|---|---|
| 修复后 SwiftShader（`realgpu_check.mjs`） | **19/19 PASS**（含 `gpu.gi_layout`、`gpu.gi_compute`、`gpu.scene_frame`） |
| `cargo xtest` | **4029 passed / 0 failed / 0 ignored / 188 段** |
| `build.sh` | **429 通过 / 0 失败**；wasm **453,223 B**（上限 458,752，余量 5,529） |
| **真机复测** | **两台设备各 19/19 全 PASS**（`?v=54d66a8e`）→ 见 **§9** |

> **方法论（本轮新增，与 §7.5 并列）**：
> **「本地能跑通」与「规范正确」是两件事。** 软件适配器严格按规范实现，
> 因此它既不能暴露「两端同错」（§7.5），也**不能暴露「实现方对规范的偏离」**（本节）。
> 对存储布局这类"规范有明确定义、但驱动实现可能偏离"的环节，
> 判据必须是**直接读回结构体字段并与宿主打包值逐字节比对**（`gpu.gi_layout`），
> 而不是"跑一遍看结果对不对"——后者只在两端算法都正确时才有判别力。

---

## 9. 真机收尾：**4 个浏览器环境、2 台设备、2 个产物版本，均 19/19 全 PASS** —— 结论收敛

### 9.1 结果

**同一份产物**（wasm `453,223 B`，`?v=54d66a8e`）在这三个浏览器环境各跑一遍，各自全绿：

> **本表记录的是旧产物。** harness 修复后产物变为 `453,239 B` / `?v=8772008b`，
> 新产物的首份真机报告见 **§9.10**（该轮首次拿到 `adapter` 字段，硬件适配器得以确认）。

| 项 | 环境 A | 环境 B | 环境 C |
|---|---|---|---|
| 机型 | 三星 Galaxy S10（SM-G9730，国行） | **荣耀 X40** | 待确认（UA 经 Chromium 缩减，不含机型） |
| SoC / GPU | 骁龙 855（SM8150）/ **Adreno 640** | 骁龙 695 / **Adreno 619** | 待确认（那一轮 harness 未采集 `adapter.info`，见 §9.7） |
| 系统 | Android 12（One UI 4.1） | Magic UI 6.1（基于 Android 12，出厂值；实机版本待确认） | Android 10+（缩减 UA 统一报 `Android 10; K`） |
| 浏览器 | Quark 10.18.6.1161（Chrome 144 内核） | 待补 | **Edge for Android 153.0.0.0**（`EdgA/153`，Chrome/153 内核） |
| wasm | `453,223 B`，`?v=54d66a8e` | 同（同一份字节） | 同（同一份字节） |
| 测试项数 | 19（17 既有 + `gpu.params_roundtrip` + `gpu.gi_layout`） | 19 | 19 |
| 结果 | **19 PASS / 0 FAIL / 0 SKIP** | **19 PASS / 0 FAIL / 0 SKIP** | **19 PASS / 0 FAIL / 0 SKIP** |
| `gpu.gi_layout` | PASS（`samples[1]` = `(0,1,0)`） | PASS | PASS（8 格全等，最大偏差 `0.000e0`） |
| `gpu.gi_compute` | PASS（`visible_total` = `8/8 · 4/8 · 4/8`） | PASS | PASS（16/24 方向可见；fingerprint `0x6959a2a8`） |
| `gpu.scene_frame` | PASS（PSNR 52 dB 档） | PASS | PASS（A 52.5 / B 52.9 dB，双跑逐位一致） |

> **环境 C 的意义与前两个不同：它是 `Chrome/Edge` 品牌。** UA 中的 `EdgA/153.0.0.0`
> 是 **Edge for Android**，内核 Chrome/153（比环境 A 的 144 新 9 个版本）。这补齐了
> A-1 验收条款里一直空着的品牌项 —— 见 §9.6。
>
> **但两点必须同时说清，不能只挑有利的说**：
> 1. 它是**移动版** Edge，不是桌面版。条款字面只要求「Chrome/Edge 品牌」，故字面满足；
>    但条款**目的**是"主流桌面浏览器可用"，而桌面档**至今零实测**（§9.6.3）。
> 2. 该轮 `adapter: {}` **未能确认是否为硬件适配器**。若是软件回退，19/19 只等于
>    第二次 SwiftShader 回归，**不计入真机证据**。harness 侧已修（§9.7）；**该缺口已在
>    另一轮新产物报告中补上**（`isFallbackAdapter:false` + `subgroup 64/128`，见 §9.10），
>    但**本环境自身**仍未采集到 `adapter`，不能追溯确认。

> **⚠️ 这两台的 GPU 是同一家：Qualcomm Adreno 640 与 Adreno 619。**
> 不是「跨 GPU 厂商」验证，是「跨两代 Adreno 驱动 + 跨两个系统」验证。
> 型号来自厂商规格页（三星半导体 / 荣耀官网），非推测。**对外不得说"跨厂商验证过"。**
>
> 残余覆盖缺口（如实登记）：**ARM Mali / Imagination PowerVR 未被覆盖**。
> 本轮 `vec3` 布局偏离最先由 Adreno 640 驱动暴露，说明**"某家驱动有偏离"这件事本身
> 不限于哪一家**；但 Mali 侧是否也有同类偏离、`vec4` 化在 Mali 上是否同样成立，
> **本仓无证据**。若日后能借到一台 Mali 机型（联发科或 Exynos 设备），值得补跑一次
> ——这是 A-1 目前唯一实质性的覆盖缺口，比"Chrome/Edge 品牌"那条更值得补。
>
> 浏览器字段：设备 B 的浏览器未记录，留待补录，**不臆造**。

> **`gpu.gi_layout` PASS 是这一轮的把关项**：它不测"结果像不像"，只测"shader 读到的
> `GiProbe.normal` 是不是宿主写进去的那三个字节"。它在第二轮给出的 `-1` 是**唯一一条
> 不依赖算法正确性的证据**；现在两台设备都读出 `(0,1,0)`，说明结构体偏移按规范重排回来了。
>
> **"换设备仍全绿"这条否掉了"针对性调参"的质疑**：`vec4` 化是**布局层面**的通用改写，
> 不是给某台设备的驱动打补丁。若它是特调，换一台设备（驱动栈必然不同）就会露馅。
> 严格说：本轮只能断言"两台设备、两份驱动栈"，**不能**断言"两家 GPU 厂商"——
> 两台 GPU 型号均未录入，不做超出证据的表述。

### 9.2 结论：代码符合规范，偏离在驱动侧

四轮的完整证据链，把"谁的错"钉死在一个方向上：

| 轮次 | 代码 | 环境 | 结果 | 说明 |
|---|---|---|---|---|
| A（§7.1） | 旧（错误球 SDF + `array<vec3>`） | SwiftShader（**严格按规范**） | PASS | 规范实现不暴露病态 |
| B（§4） | 同上，**字节可比** | 真机（Samsung Internet 30.1 那次跑测） | FAIL，整块零 | 浮点路径差异 |
| C（§8.3） | 修复球 SDF 后 | 三星 / Quark | FAIL，`normal` 读到 `-1` | **布局偏离** |
| D | **`vec4` 化布局** | 三星 / Quark | **19/19 PASS** | 唯一变量是布局写法 |
| **E** | **同上，字节相同** | **荣耀 X40（Adreno 619）** | **19/19 PASS** | **换设备换驱动代际，仍全绿** |

> **判定：`vec3<f32>` 在存储布局上的错位，责任在驱动实现，不在本仓代码。**
> WGSL 规范对 `vec3<f32>` 的规定是 `align=16 / size=12`，本仓 Rust 侧按此打包
> （`gi_pack_probes` 打包字节 dump 逐字节正确，见 §8.3）；shader 侧改成 `vec4` 后
> 同一台设备、同一条链路、同一组数值**立刻全绿**。若是代码算错，改布局不会好；
> **若是"针对某台设备调参"，换一台设备就会露馅——D 与 E 跑的是同一份字节，两台全绿。**
> （但两台同为 Adreno，见 §9.1 的覆盖缺口说明：这证明的是"不挑机型/不挑驱动代际"，
> 不是"不挑 GPU 厂商"。）

**但"驱动不规范"不能作为不修的理由**——这是本轮的工程结论，也是对外口径：

- 驱动偏离规范**不是个案**：同一份 shader 在 SwiftShader 上正确、在这台移动 GPU 上错位，
  说明"规范写了"不等于"设备上成立"。**不能把规范的确定性当作设备的确定性。**
- 因此本仓采取的策略是**主动避开灰区**：凡参与 buffer 布局的结构体，一律 `vec4` 化
  （`vec4` 的 `align/size` 恒为 16，没有解释空间）。**这不是迁就不规范实现，是取规范的交集**：
  按规范 `vec3<f32>` 之后本来就要补 4 字节对齐填充，改成显式 `w` 只是把那 4 字节
  从"编译器填"变成"我填"——**净增 0 字节**。收益是**跨驱动确定性**。
  这个取舍写进 `wgsl_lint.mjs` 规则 H 强制。
- **不修改判据、不降 `F32_TOL`、不把失败项改 Skip** —— 全程未动过这三样。

#### 对严格按规范的驱动：零影响（不是"也兼容"，是"逐字节等价"）

这一条必须写死，否则容易误读成"为了迁就烂驱动牺牲了正确性"。四份独立证据：

| 判据 | 内容 | 结果 |
|---|---|---|
| **字节级** | `const GI_PRIM_STRIDE_BYTES = 48` / `GI_PROBE_STRIDE_BYTES = 304` / `GI_POINT_STRIDE_BYTES = 32` 的**赋值行在修复 diff 里一行都没出现**（diff 只动了 `///` 注释里的布局说明） | 若布局有变，这些常量必须改；没改还全绿 ⇒ 布局等价 |
| **结构体 size** | `GiProbe` 304 / `GiParams` 256 / `GiPrim` 48 / `GiPoint` 32 | 四个**全部不变**（不是"多占 4 字节"） |
| **规范实现端到端** | SwiftShader（**严格按规范**的软件适配器）19/19 PASS，含 `gpu.gi_layout` 逐字段比对与 `gpu.gi_compute` 数值对拍 | 规范驱动不回退 |
| **数值指纹** | `gi_cpu_reference_fingerprint_matches_independent_js` = `0x6adb7c92`，`R-08` 前后**完全没变** | 布局改写不触数学 |

> 指纹这一条要用对：该锚在 `R-07`（球 SDF）时**确实变过**（`0xaf8429a3` → `0x6adb7c92`，
> 因为那次数学真的改了）；到 `R-08`（vec3→vec4）**一次没动**。
> 两次修复一动一静，正好反证：`R-07` 改的是算法，`R-08` 改的只是内存写法。

> 原因很简单：WGSL 里 `vec3<f32>` 的 `align` 也是 16，所以 `struct { vec3 a; vec3 b; }`
> 的 `b` 本来就落在 `@16`，`a` 之后的 `@12..16` 是规范要求存在的填充。
> 写 `vec4` 只是把这 4 字节**显式化**（`w=0`，不参与任何运算）。
> **规范实现下两者的内存映像完全一致，所以"原来规范的驱动"照样跑，且是同一份字节。**
>
> 反过来不成立：`vec3` 的兼容面是 `vec4` 的**真子集**——凡能跑 vec3 布局的实现必然能跑 vec4，
> 反之不然（本轮这台移动 GPU 就是反例）。**改向兼容面更大的写法，永远不会有回退风险。**

### 9.3 一个必须留档的反面教训

> **V1.0 的「全部 PASS」是假阳性。** 不是"以前好现在坏"，是**当时就没验对**：
> §7.2 的 `q.x` 病态 SDF 让两端"同错得一致"，§8 的 `vec3` 布局让 `normal` 一直读到
> 哨兵值——而 V1.0 的 17 项里**没有任何一项直接检查字段偏移**，所以两处错误都能
> 带着绿灯出厂。作者的两句判断（「V1.0 的测试写错了」「有 bug 就要修」）都成立。

这也是 §7.5 与 §8 方法论的可操作版本：**"两端一致"只能证明两份实现相同，
不能证明任一份正确。想让判据有判别力，必须有一条直接观测物理量（字节/偏移/位掩码）
的锚点**——这就是 `gpu.params_roundtrip`（256 B 回环）与 `gpu.gi_layout`（8 格字段）
存在的唯一理由。

### 9.4 台账更新

| 台账 | 更新 |
|---|---|
| **V2.2-A-1** 实机档 | **4 个浏览器环境、2 台设备、2 个产物版本，各 19/19**；三项核心判据（`w*h*4` / 双跑逐位 / ≥40 dB PSNR）均成立。**「≥2 设备」✅**；**「Chrome/Edge 品牌」✅ 字面满足**（Edge for Android 153 一轮 19/19）；**「硬件适配器」✅ 已在一台设备上直接确认**（SamsungBrowser 轮：`isFallbackAdapter:false` + `subgroup 64/128`，§9.10）——但 A/B/C 三轮 `adapter` 未采集，其硬件性仍是推断而非观测；**桌面档仍零实测**（§9.6.3） |
| `gpu.gi_layout` / `gpu.params_roundtrip` | 新增项，两台真机均 PASS，转为常驻回归 |
| `gpu.gi_compute` | 真机 FAIL → **已修复，两台真机验证 PASS**，该项恢复"可声称" |
| 对外口径 | 允许追加：「已在 2 台移动端真机（三星 + 荣耀）+ 1 台 Edge for Android（Chrome 153 内核）上各 19/19 通过」「已确认跑在硬件适配器上（`isFallbackAdapter=false`，subgroup 64/128）」；**仍不得**写「WebGPU 已支持」或「Chrome/Edge 已验证」——**移动版 Edge 通过不等于桌面版可用**（GPU 后端不同：D3D12 / Metal vs Vulkan/GLES）；也**不得**写「跨 GPU 厂商验证」（两台同为 Qualcomm Adreno） |

> **关于「Chrome/Edge 品牌」这条验收条款，建议产品侧重新审视**：它的目的是保证
> "主流浏览器可用"，但真正决定可用性的是**内核版本 + 驱动实现**，不是品牌字符串。
> 本轮拿到的证据比原条款更强——同一份产物在**严格规范实现**（SwiftShader，19/19）与
> **两台移动设备**（三星 + 荣耀，各 19/19）上都正确，覆盖了兼容性的两端。
> 若产品侧同意把条款改为「≥2 台 Chromium ≥ 某版本内核设备，含 1 台非桌面」，
> A-1 即可结项；否则维持待验，**不以"全 PASS"为由自行改写**。
>
> 残余缺口：GPU 厂商只覆盖 Qualcomm（Adreno 640 / 619），**Mali / PowerVR 未覆盖**；荣耀 X40 的实机系统版本与浏览器未记录。

### 9.5 责任边界条款（可引用）

> 结论：**凡本仓按 WGSL 规范输出、且能举证"规范实现上正确 + 实测偏离在驱动"的缺陷，
> 不计为本仓缺陷。** 但"无责"与"可用"是两件事，条款分两段写清，不许只引前半句。

#### 9.5.1 免责成立的三条前提（缺一不可）

| # | 前提 | 本项目的举证物 |
|---|---|---|
| 1 | **输出可证符合规范**，不是"我觉得符合" | `gi_pack_probes` / `gi_params_bytes` 的逐字节 dump；`gi_layout_census` 逐字段偏移自检 |
| 2 | **规范实现上通过**，排除算法与打包错误 | SwiftShader（严格按规范的软件适配器）19/19 PASS |
| 3 | **已尽合理规避**，即选规范交集的写法，而非踩灰区赌运气 | 布局一律 `vec4` 化（`align/size` 恒 16，无解释空间）；`wgsl_lint.mjs` 规则 H 静态强制 |

> 三条缺任何一条，本仓就只剩"口头声明符合规范"，免责不成立。
> **尤其第 3 条**：证明了"规范这么写"却仍选灰区写法，等于明知有分歧还去踩，
> 那就从"驱动的问题"变成"我们的选择"。

#### 9.5.2 三步判别流程（可复用，不依赖直觉）

```
某设备 FAIL
  ├─ ① 同一份 wasm 在 SwiftShader 上 PASS？
  │    否 → 算法/打包可能有错 → 自查，不进本条款
  │    是 ↓
  ├─ ② 布局 census / 字节 dump 与规范一致？
  │    否 → 本仓打包错 → 自查，不进本条款
  │    是 ↓
  └─ ③ 仅在特定设备 FAIL，且换成兼容面更大的写法即 PASS？
       → 归因驱动实现偏离规范 → 适用本条款
```

本轮真机缺陷正是走完 ①②③ 全绿后归因的（§9.2 证据链 A→E），不是跳步结论。

#### 9.5.3 不豁免的部分（必须一起引）

- **可用性责任不转移。** 用户不会因为"驱动有 bug"就接受黑屏。驱动偏离造成的不可用，
  缺陷归属可以不算本仓，**交付风险仍算**——必须登记、必须给降级路径
  （`cpu_fallback` / 适配器能力判定），不能不修也不说。
- **不得以"符合规范"关闭问题单。** 正确处置是：登记为环境兼容项 + 提供规避写法 +
  在边界文档写明，而不是回一句"规范如此"。
- **不得把本条款当"不用测"的理由。** 它约束的是**归因**，不是**验证义务**；
  恰恰相反，能引用本条款的前提就是测过（§9.5.1 三条全靠测试产出）。

#### 9.5.4 一句话版本（对外可直接引）

> 本项目按 WGSL 规范输出，并对规范一致性与设备实测双向举证；经举证确认由驱动偏离规范
> 导致的缺陷，不计为本项目缺陷，但相应设备兼容性风险仍会登记并给出降级路径。

### 9.6 环境 C（Edge）补上了品牌项 —— 但只补了一半

#### 9.6.1 补上的那一半

A-1 的实机档有两个条件：「≥2 台设备」与「Chrome/Edge 品牌」。此前所有跑测
（Samsung Internet、Quark、荣耀那台待补浏览器）**没有一个是 Chrome/Edge 品牌**，
所以台账一直停在"待验"。

环境 C 的 UA 为：

```
Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko)
Chrome/153.0.0.0 Mobile Safari/537.36 EdgA/153.0.0.0
```

`EdgA` = **Edge for Android**，产品名与渲染引擎内核 Chrome/153 均是微软 Edge 品牌。
**按条款字面（品牌字符串），此条满足。**

附带价值：Chrome/153 比环境 A 的 144 新 9 个大版本，说明 `vec4` 化在**更新的内核**上
同样成立，不是只在旧内核上"碰巧过"。

#### 9.6.2 没补上的那一半（必须写在前面，不能只报喜）

| 缺口 | 现状 | 影响 |
|---|---|---|
| **桌面版从未实测** | 三个环境全是移动端 | 条款目的（"主流桌面浏览器"）**未被验证**。移动与桌面即使同为 Chromium，GPU 后端（Metal/D3D12/Vulkan）、驱动栈、扩展支持都不同 |
| **硬件适配器未确认** | 该轮 `adapter: {}`，`isFallbackAdapter` 不可见 | 若为软件回退，本次 19/19 只等于第二次 SwiftShader 回归。**注**：另有 SamsungBrowser 一轮的新版报告给出了 `isFallbackAdapter: false`（§9.10），但**该轮自身**仍未采集到 adapter，不能追溯确认 |

> 结论口径：**可以说「已在一台 Edge（Android 153 内核）上 19/19 全通过」；
> 不可以说「Chrome/Edge 已验证」**（缺桌面、缺硬件适配器确认）。
> 这两句差别很大，前者是事实，后者是承诺。

#### 9.6.3 桌面档为什么必须单列

README 的边界表述是「限 Chrome/Edge」，括号里从来没写"移动版"。用户读到这句话，
默认理解为"我桌面上的 Chrome 能用"。所以：

- **移动 Edge 通过 ≠ 桌面 Edge 通过**。桌面走 D3D12（Windows）或 Metal（macOS），
  与本轮验证的 Vulkan/GLES 路径不是同一条。
- 桌面 Chrome/Edge 是**用户最可能的实际环境**，也是分发说明的承诺对象。
- 补齐成本极低（打开同一个 URL），**收益是让那句边界表述第一次真的有证据**。

> 若本条长期不补，README 那句话应改为「限移动端 Chromium 内核浏览器（Chrome/Edge 品牌已验 1 例）」，
> **不得**保持「限 Chrome/Edge」这种会被读成桌面可用的表述。

### 9.7 harness 缺陷：`adapter: {}` —— 报告丢了硬件指纹

#### 9.7.1 现象与根因

三次真机报告的 `adapter: {}` 都是空的。根因不在采集、在序列化：

```js
// 旧代码（crates/prism-gpu/web/index.html）
lines.push("adapter: " + JSON.stringify(adapterInfo || {}));
```

`GPUAdapterInfo` 的 `vendor` / `architecture` / `device` / `description` /
`isFallbackAdapter` **全部定义在原型 getter 上**，而 `JSON.stringify` 只序列化
**自有可枚举属性** → 恒得 `{}`。对象本身是好的（`adapterInfo?.vendor` 能读到，
摘要表里也能显示），只有进报告文本这一步丢了。

#### 9.7.2 为什么这不只是"显示不好看"

1. **每份报告都缺硬件指纹**：事后只能追问用户"你用的哪台机器"，而 UA 已被
   Chromium 缩减（`Android 10; K`），连机型都拿不到——这就是 §9.1 表里
   环境 C 的"待确认"是怎么来的。
2. **更严重：`isFallbackAdapter` 不可见**。这是"真机证据"与"第二次 SwiftShader
   回归"的唯一分界线。缺了它，一份 19/19 的报告**无法自证是硬件跑的**。
   这与 §9.3 记录的"V1.0 假阳性"是同一类错误：**判据缺少能区分两件事的锚点**。

#### 9.7.3 修复

改为逐字段显式取值，并在报告里加一条**主动守卫**：

```js
const aiFields = {
  vendor: ai.vendor ?? null, architecture: ai.architecture ?? null,
  device: ai.device ?? null, description: ai.description ?? null,
  subgroupMinSize: ai.subgroupMinSize ?? null,
  subgroupMaxSize: ai.subgroupMaxSize ?? null,
  isFallbackAdapter: ai.isFallbackAdapter ?? null,
};
lines.push("adapter: " + JSON.stringify(aiFields));
if (aiFields.isFallbackAdapter === true) {
  lines.push("⚠️ adapter 为 FALLBACK（软件实现）—— 本次结果不计入真机证据…");
} else if (aiFields.isFallbackAdapter === null) {
  lines.push("⚠️ adapter.isFallbackAdapter 不可读，无法排除软件回退：请人工核对 chrome://gpu…");
}
```

设计取舍：**不可读时也要显式告警**，而不是静默当作 `false`。宁可报告多一行噪音，
也不让"无法排除软件回退"这种情况悄悄混进真机证据。

#### 9.7.4 待办

环境 C 需用新页面**重跑一次**，以取得 `adapter` 字段并确认 `isFallbackAdapter`。
重跑前必须强刷（HTML 自身无指纹参数，浏览器可能仍持旧页）。

### 9.8 `--dump-report` 顺带查出的第二个报告缺陷：wasm 大小写成线性内存

#### 9.8.1 现象

给 `realgpu_check.mjs` 加 `--dump-report`（导出 harness 报告全文）后，本机第一份报告写的是：

```
wasm: loaded (instantiateStreaming, 1441792B, ABI v1)
```

而磁盘上的文件是 **453,239 B**。差了 3 倍多。

#### 9.8.2 根因

```js
// 旧代码
return { instance, how, size: total || (instance.exports.memory.buffer.byteLength) };
//                                     ^ total = content-length，静态服务没给时回退到这里
```

回退分支取的是 **wasm 线性内存大小**（约 1.4 MB 量级），与**文件字节数**不是同一个量。

危害不在"数字难看"：A-1 的核心判据之一是「三台设备跑的是**同一份 wasm 字节**」，
而这条判据的观测手段就是报告里的 size + 指纹。**字段说谎，判据就失效。**

#### 9.8.3 修复（两处，都必要）

1. **harness 侧**：size 只认 `content-length`；拿不到就明确写"文件大小未知"，
   并注明"线性内存 N B 不等于文件大小"。**不拿另一个量凑数。**
2. **回归脚本侧**：`realgpu_check.mjs` 的静态服务补 `content-length`——
   本地回归也不能给出错误读数（这一条是根因，页面只是没兜住）。

#### 9.8.4 复验

```
wasm: loaded (instantiateStreaming, 453239B, ABI v1)     ← 与磁盘一致
adapter: {"vendor":"google","architecture":"swiftshader","device":"0xc0de",
          "description":"SwiftShader Device (Subzero)",
          "subgroupMinSize":4,"subgroupMaxSize":4,"isFallbackAdapter":true}
⚠️ adapter 为 FALLBACK（软件实现）—— 本次结果不计入真机证据…
```

**fallback 守卫真的会触发**（SwiftShader 的 `isFallbackAdapter` 确为 `true`），
不是写了不跑的装饰代码。这是本节最有价值的一条：守卫若不触发，等于没有。

### 9.9 修复后的构建与重跑要求

§9.7 / §9.8 改动了 `index.html` 与 `abi.rs`（两处测试 name 字符串），
产物**已不是**设备 A/B/C 跑过的那一份：

| | 旧（设备 A/B/C 实测） | 新（修复后） |
|---|---|---|
| wasm 大小 | `453,223 B` | **`453,239 B`**（+16 B；上限 458,752，余量 5,513） |
| 内容指纹 | `?v=54d66a8e` | **`?v=8772008b`** |
| 实质差异 | — | 仅两处测试 `name` 字符串（去掉会腐烂的硬编码数量） |
| 判据 / 算法 / 布局 | — | **一行未动** |
| 本机 SwiftShader | 19/19 | **19/19**（`build.sh` 429/0） |

> **纪律**：字节变了就不能把旧结论直接挂到新指纹上。正确做法是**重跑一次**，
> 由新报告自带 `adapter` 与 `isFallbackAdapter`——这也是补上环境 C 那个
> "硬件适配器未确认"缺口的唯一途径。
>
> 重跑注意：`index.html` 自身**没有**缓存破坏参数，浏览器可能仍持旧页。
> 必须强刷，或直接访问带查询串的地址（`…/index.html?t=<任意>`）。
>
> → **该重跑已完成**：`?v=8772008b` 首份真机报告见 **§9.10**，`adapter` 字段有值、
> `isFallbackAdapter:false`，硬件适配器确认。

### 9.10 新产物首份真机报告：`adapter` 有值了，硬件适配器确认

#### 9.10.1 报告关键行

```
UA: … SamsungBrowser/30.1 Chrome/143.0.0.0 Mobile Safari/537.36
wasm: loaded (instantiateStreaming, 453239B, ABI v1)
adapter: {"vendor":"qualcomm","architecture":"adreno-6xx","device":"","description":"",
          "subgroupMinSize":64,"subgroupMaxSize":128,"isFallbackAdapter":false}
[wasm-loaded] OK prism_gpu.wasm?v=8772008b via instantiateStreaming；ABI v1；19 个测试项
```

三条判读：

1. **`isFallbackAdapter: false`** —— §9.6.2 列的“没法确认是不是真 GPU”这条被**直接填上**。
   该字段为 `true` 时 harness 会额外打一行告警；本轮没有告警，说明走的是硬件路径。
2. **`?v=8772008b` / `453,239 B`** —— §9.9 要求的新指纹重跑**已完成**，报告自证跑的是修复后
   那份字节，不是浏览器缓存的旧页。
3. **19 PASS / 0 FAIL / 0 SKIP** —— 与旧产物一致；测试名已变为“全部内嵌 shader”
   “全部 WGSL 的 compute”（硬编码数量已移除，§9.7 的连带改动生效）。

#### 9.10.2 `subgroup` 是比 `isFallbackAdapter` 更硬的旁证

| 适配器 | `subgroupMinSize` | `subgroupMaxSize` |
|---|---|---|
| SwiftShader（软件，本机回归） | **4** | **4** |
| 本轮真机（Adreno 6xx） | **64** | **128** |

软件适配器的 subgroup 由 CPU SIMD 宽度决定，恒为小值（SwiftShader 是 4）；
移动 GPU 的 wave 宽度是 64 / 128 这个量级。**一个软件回退的适配器报不出 64/128。**

这条的价值在于它是**独立于 `isFallbackAdapter` 的第二个信号**：
`isFallbackAdapter` 是浏览器**声明**的，subgroup 范围是**驱动实际实现**的。
两者同时指向硬件，才算排除软件回退。只看前者，等于只信一个可能说谎的字段。

**代价说明（诚实登记）**：本仓目前**没有任何 subgroup 操作**（无 `subgroupBallot` /
`subgroupBroadcast` 等，仅 `dispatch.rs` 一处注释提到概念），所以 4/4 与 64/128 的差异
**不影响本次结果**。这条记录对将来有用：若日后引入 subgroup 优化，**不能假设 subgroup
宽度固定** —— `min=64 / max=128` 说明同一设备上两种 wave 宽度都可能出现。

#### 9.10.3 这轮没解决什么

| 仍缺 | 说明 |
|---|---|
| **机型仍无法自动识别** | `device` 与 `description` 都是空串 —— Chrome 在移动端出于隐私不填。UA 又已缩减（`Android 10; K`）。**报告仍不能自证是哪台机器。** |
| **`architecture` 不足以区分那两台设备** | 两台都是 `adreno-6xx`（Adreno 640 与 Adreno 619 同属 6xx）。**看到 `adreno-6xx` 不能推断是哪台。** |
| **桌面档仍零实测** | 本轮是 SamsungBrowser（移动版 Chromium 143 内核），与 §9.6.3 的缺口无关。 |

> **因此 A-1 的正确记法是**：「硬件适配器**已在一台设备上直接确认**」，
> **不是**「所有环境均已确认硬件」。环境 A/B/C 那三轮 `adapter` 字段从未采集到，
> 严格讲它们的硬件性只能靠“同设备、同链路、同一组数值”推断，**不是直接观测**。

#### 9.10.4 一条可能的闭环（需作者确认，未证实）

早前第一轮用的也是 **SamsungBrowser 30.1**（Chromium 143 内核），当时是
**16 PASS / 1 FAIL**（`gpu.gi_compute` FAIL，产物 423,855 B）。本轮同样是
SamsungBrowser 30.1 / Chromium 143，**19 PASS / 0 FAIL**（产物 453,239 B）。

若为同一台设备，则构成**同一设备、同一浏览器、新旧产物直接对比**的证据链：

| 产物 | 大小 | 结果 |
|---|---|---|
| 修复前 | 423,855 B | 16 PASS / 1 FAIL |
| 修复后 | 453,239 B | **19 PASS / 0 FAIL** |

这比“换设备也过”更强：它排除了“换设备碰巧能跑”的解释，把差异直接锁在代码上。
**但 UA 已缩减、`device` 为空**，本轮无法直接证实是同机 —— 记录为“高度可能，未证实”，
不计入证据，只作线索。
