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
| V2.2-A-1 实机档 | **1/2 设备**：核心三项判据全通过（`w*h*4` / 双跑一致 / 52.5·52.9 dB），但「≥2 设备」与「Chrome/Edge 品牌」未满足 → **仍为「已交付待验」** |
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
| **真机复测** | **待测**（需作者用新版本重跑一次，见 §7.4 第 6 条缓存破坏） |

