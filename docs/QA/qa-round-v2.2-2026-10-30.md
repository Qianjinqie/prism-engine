# QA 测试报告 — V2.2 收口全仓检查

- **轮次**：qa-round-v2.2-2026-10-30（W0–W3 全部提交后的**唯一一次**全仓检查，不分波次）
- **日期**：2026-10-30
- **QA 工程师**：严过关（Yan）
- **测试对象**：`/workspace/media-engine`，HEAD = `0f33252`（"V2.2 W3: T07 两目标锚定 + A-2-CURVE-VK 真点亮 + T10 包体度量 + T11 VAAPI 绑定层"），29 个 crate
- **开工时工作区**：干净（`git status --short` 空）
- **基线**：`bf34af1^`（V2.2 开工前）；全量 3929 passed / 0 failed / 0 ignored / 179 段
- **本机环境**：Linux 沙箱；Vulkan ICD 在位（`/usr/share/vulkan/icd.d/lvp_icd.json` + `libvulkan_lvp.so`），实测设备 `llvmpipe (LLVM 20.1.2, 256 bits)`，枚举 1 个物理设备——即 GPU 判据是**真跑**，不是自门控跳过
- **纪律声明**：**未 commit**（按任务书由总监统一收口）；共修 6 处（3 处文案越级/自相矛盾、2 处登记缺失/过期、1 处锚点位移跟随），全部为文字与登记，**未改动任何代码逻辑、未放宽任何阈值、未加 `#[ignore]`、未删任何测试**

---

## 1. 测试范围与命令

| # | 项 | 命令 | 用途 |
|---|----|------|------|
| G1 | 全量回归 | `cargo xtest`（= `test --workspace --release`） | 29 crate 全量；对比 3929 / 179 段基线 |
| G1b | 禁删测核查 | `git log bf34af1^..HEAD --stat`、逐 crate `#[test]` 计数（`git grep -F -c` 对比） | 测试文件删除 / 测试数减少 |
| G2 | 零依赖 | `git diff … -- Cargo.lock Cargo.toml`、`find -name build.rs`、`grep -rn '#\[link'`、`cargo tree --workspace --edges normal --prefix none` | 外部 crate 数 0 |
| G3 | 确定性 | `cargo test -p prism-gpu --test qa_curve_parity/--test qa_render_scene -- --nocapture`、全量中的 golden/MD5 段 | 逐位不漂移 |
| G4 | 不伪造能力 | `grep -rn 'MoltenVK\|moltenvk'`（6 类扩展名）+ 九语 README 逐行核 + 四处（Metal/硬编/WebGPU/加速比）状态标签审查 | 零越级声称 |
| G5 | 卫生 | `cargo test -p prism-core --test qa_ignore_guard`、`grep -rn '#\[ignore\]'`、`grep -rnE 'TODO\|FIXME\|unimplemented!\|todo!'`、`node tools/hygiene_scan.mjs` | 零占位符 |
| G6 | 黄金值登记 | `git diff bf34af1^..HEAD`（`.golden` / `expected` / 常量 / 断言行）→ 对 `CHANGELOG.md` + PRD §7 | 无表外位移 |
| G7 | 两档标注 | PRD §2.1 档位列 + §3.0.1 两档表 + 交付说明 | 只代码就绪档进 Go/No-Go |
| G8 | 对外口径 | README 边界表 8 行 × 9 语言；抽查锚点**实际打开核实** | 每行可指认 |
| G9 | 交叉验证 | `cargo test -p prism-rt --release`（275）+ T12 四钉子单跑 | bsdf 两次改动互不削弱 |
| — | 独立复核 | `cargo test -p prism-ffi-hwenc --release --features vaapi` | 硬编绑定层 feature 路径 31 测试 |

> 注：`cargo xtest -p` 不缩小范围（别名内含 `--workspace`），定向回归一律用 `cargo test -p <crate> --release`。

---

## 2. 全量回归（G1）

```
$ cd /workspace/media-engine && cargo xtest          # 日志 /tmp/xtest_final.log
EXIT=0
```

| 指标 | 基线（V2.2 开工） | 本次（HEAD 0f33252） | 净增 |
|---|---|---|---|
| **passed** | 3929 | **4008** | **+79** |
| failed | 0 | **0** | 0 |
| ignored | 0 | **0** | 0 |
| 测试段数 | 179 | **188** | **+9** |

**段数 +9 的构成**（与新增测试文件一一对应，无遗漏、无合并）：

`qa_dng_support_matrix`(9)、`qa_curve_tessellation`(5)、`qa_curve_parity`(10)、`qa_svgf_phase2`(4)、`qa_perf_counters`(8)、`qa_material_b3`(4)、`qa_material_furnace`(5)、`qa_spectral_degeneracy`(3)、`vaapi_binding`(default 0 / `--features vaapi` 31)。

**净增 +79 与声明 +92 的差额已归因**：`#[test]` 函数从 **4037 → 4129（+92）**，但 `vaapi_binding` 的 17 条测试在默认 feature 下被门禁剔除（`prism-ffi-hwenc` default 14 / `--features vaapi` 31，已实测），另 doc-test 由 4 → 7（+3）。79 + 17 − 3 与 92 的残差为 1，落在 `cfg(test)` 内条件编译的一条辅助用例上，不影响判据。

### 2.1 禁删测核查（红线项）

| 检查 | 结果 |
|---|---|
| `git log bf34af1^..HEAD --stat` 中的 `D`（删除）条目 | **0 个**（改动全为 `M` / `A`） |
| 逐 crate `#[test]` 计数减少 | **0 个 crate 减少**。增加：`prism-ffi-hwenc` 11→31(+20)、`prism-gpu` 289→312(+23)、`prism-rt` 255→275(+20)、`prism-render` 212→221(+9)、`prism-geometry` 75→83(+8)、`prism-dsl` 124→128(+4)、`prism-io` 581→585(+4)、`prism-postfx` 198→202(+4)，其余 21 crate 持平 |
| 既有断言被删除 | 14 行，全部在 `prism-gpu/src/shaders.rs` 的 `scene_kernel` 旧测试段——因 A-2 把绑定集从 V1 四绑定升 V2 六绑定，断言**迁移**到单一真源 `scene_abi.rs`（新增 `bindings_v1_profile_is_continuous_prefix_zero_through_three` / `..._zero_through_five` 双 profile 断言 + 负自证），W3 commit 正文有对应说明（"scene_main 收为两目标锚定 + 从 scene_abi 派生期望值 + 负自证"）。**属判据升级，非放宽** |
| `differing_pixels` / PSNR / `max_channel_delta` / 阈值被修改 | **0 处**（逐行扫描 V2.2 全部 diff 的 `-` 行，仅命中上述 shaders.rs 迁移项） |

**G1 裁决：PASS。**

---

## 3. G1–G9 逐条裁决

| 门 | 项 | 结论 | 关键证据 |
|---|---|---|---|
| **G1** | 全量回归 0 failed / 0 ignored，passed 显著净增，无删测 | **PASS** | 4008/0/0/188，+79；段数 +9 = 9 个新测试文件；无 `D` 条目、无 crate 测试数减少 |
| **G2** | 零第三方依赖 | **PASS** | `Cargo.lock` diff 仅 +`prism-geometry`（仓内 path，设计 Q5 允许）；无 `build.rs`；无新增 `#[link]`；`cargo tree … \| grep -v 'prism-'` 输出**空** |
| **G3** | 确定性 | **PASS** | CPU golden/MD5 段全绿；GPU lavapipe **真跑**：四曲线图元 `max_channel_delta=0 && differing_pixels=0`（`software_icd=true`）+ 双跑逐位一致；`qa_render_scene` 硬判据 0 差异 |
| **G4** | 不允许伪造能力 | **PASS（修 6 处后）** | MoltenVK 指定扩展名 **0 命中**；九语 Metal 行语义一致；四处状态标签逐处核对见 §4.3；修复前模块文档残留「探测链就绪」→ 已改 |
| **G5** | 禁 TODO/占位符/`#[ignore]` | **PASS** | `qa_ignore_guard` 4/4 绿；真实 `#[ignore]` **0**；`TODO/FIXME/unimplemented!/todo!` 5 处命中**全为描述既有修复的散文**（溯源 v1.0/v2.0-M3），V2.2 新增占位符 **0** |
| **G6** | 黄金值位移必登记 | **PASS（补登 1 条 + 订正 1 条后）** | 无 `.golden` 文件改动；发现 1 处未登记行为位移（T08 折射/Fresnel）→ 已补登 `R-2026-xx-06`；PRD §7 A-2 行过期 → 已增行订正 |
| **G7** | 两档标注 | **PASS（附 1 项需裁决）** | PRD §2.1 全 14 行均有档位列；§3.0.1 两档表 7 行齐全；交付说明缺独立「已交付待验（7 项）」章节 → 见 §5 U1 |
| **G8** | 对外口径一致性 | **PASS** | 边界表 8 行 × 9 语言 = 72 行**全部**含 `文件:行号` 锚点；抽查 6 条声称实际打开核实语义相符（含 1 条 off-by-n 隐患复核为正确） |
| **G9** | T12 钉子与 T08 交叉验证 | **PASS** | 四钉子全在且全绿；`prism-rt` 275/0/0；T08 改动只动 `sample` 的 transmission 瓣，未触碰 `eval`/补偿函数；白炉/电介质断言零回归 |

---

## 4. 逐条证据

### 4.1 G2 零依赖（明细）

```
$ git diff bf34af1^..HEAD -- Cargo.lock
+ "prism-geometry"          # 仅此一行，加在 prism-gpu 的依赖列表内（仓内 path）
$ find . -name build.rs -not -path './target/*'   → 空
$ cargo tree --workspace --edges normal --prefix none | grep -v 'prism-' | sort -u
（空输出）
```

`#[link(...)]` 全仓仅 2 处，**均为 V2.2 之前的既存历史**（`git show bf34af1^:` 两文件均已有同位置声明）：

| 位置 | 是否在 Linux 默认构建路径 | 判定 |
|---|---|---|
| `prism-gpu/src/metal/loader.rs:34` | **否**——`metal/mod.rs:68-69` 为 `#[cfg(target_os = "macos")] pub mod loader;`，Linux 上整体剔除 | 既存、且不在默认路径 |
| `prism-gpu/src/vulkan/loader.rs:38` | 是（`#[cfg(not(target_arch = "wasm32"))]`），但 v1.0 起即存在 | 既存、系统 `libdl`，非第三方 |

新增的 `prism-ffi-hwenc/src/vaapi_bind.rs` 用裸 `extern "C"` + `dlopen`/`dlsym`，**无 `#[link]`**（模块文档显式声明并给出理由），与 G2 一致。

### 4.2 G3 确定性（实测输出）

```
[curve-parity:cylinder] pixels=384 max_channel_delta=0 differing_pixels=0 software_icd=true first_diff=None
[curve-parity:cone]     pixels=384 max_channel_delta=0 differing_pixels=0 software_icd=true first_diff=None
[curve-parity:torus]    pixels=384 max_channel_delta=0 differing_pixels=0 software_icd=true first_diff=None
[curve-parity:revolution] pixels=384 max_channel_delta=0 differing_pixels=0 software_icd=true first_diff=None
[curve+sphere] max_channel_delta=0 differing_pixels=0 first_diff=None
test vulkan_curve_render_is_bitwise_deterministic_across_runs ... ok
test curve_render_is_bitwise_deterministic_across_runs ... ok
→ test result: ok. 10 passed; 0 failed; 0 ignored

[device] llvmpipe (LLVM 20.1.2, 256 bits)（枚举 1 个物理设备）
[parity:aesthetic-off] pixels=3072 max_channel_delta=0 differing_pixels=0 differing_gt1=0 mean_abs_delta=0.00000 software_icd=true
[parity:tail] 113×47 max_channel_delta=0 differing_pixels=0
test gpu_scene_render_is_bitwise_deterministic_across_runs ... ok
→ test result: ok. 5 passed; 0 failed; 0 ignored
```

判据是**硬 0**，不是"≤ 阈值"。`A-2-CURVE-VK` 的灯确实亮了（`VULKAN_CURVE_PATH = true`），且失败退路是"翻回 `false` 并留名"而非调阈值（`qa_curve_parity.rs:28-31` 原文纪律）。

### 4.3 G4 四处状态标签审查（逐处）

| 处 | 要求口径 | 现状（修复后） | 结论 |
|---|---|---|---|
| **Metal** | 只可「不提供独立后端 / 不点亮 / `is_available()` 恒 false」 | `metal/mod.rs:6-10` 明写「本仓不提供独立 Metal 后端（不点亮）」；`metal/render.rs:1-8` 同口径并以「历史资产」定位；`is_available()` 在 `metal/mod.rs:192` 恒 `false` 有测试钉住；`metal/` 内「探测链」字样 **0**、「探测链就绪」全仓 **0** | ✅ |
| **硬编** | 只到「绑定就绪」档，绝不称"可编码" | `prism-ffi-hwenc/src/lib.rs:3-12` 四档表：一档三后端 ✅、二档**仅 VA-API** ✅、三档 ❌、四档 ❌；第 14 行显式「**第二档不是"能编码"**」；`any_available()` 恒 `false` | ✅ |
| **WebGPU** | 必须限定 Chrome/Edge；T3 通过 ≠ 可用 | `README.md:324`：「代码就绪（T1/T2/T3 绿），**仅 Chrome/Edge 可行**；渲染路径 T03 落地，**T3 为 SwiftShader 非用户设备**」——未因 T3 通过而改写措辞 | ✅ |
| **GPU 加速比** | 「登记待验」，非"已支持"、不写数字 | `README.md:326`：「**登记待验**：仅在 lavapipe 验证正确性，真实 GPU 加速比不测、**不写数字**（lavapipe 为伪值）」 | ✅ |

**MoltenVK 清零**：`grep -rn 'MoltenVK\|moltenvk' . --include=*.rs --include=*.md --include=*.wgsl --include=*.mjs --include=*.yml --include=*.toml | grep -v target` → **0 命中**。仓内 `dist/prism-engine-v2.2-source.txt`（V2.2 快照，663 文件）同样 0 命中；仅 v2.0 / v2.1 两个**历史未跟踪**快照 `.txt` 各留 1 处旧注释（见 §5 U3）。

**九语 Metal 行一致性（9/9 逐行核）**：`README.md:323`、`docs/README.{en,de,es,fr,ja,ko,pt-BR,ru}.md:322` 全部为「无独立 Metal 后端 + 不点亮 + `is_available()` 恒 false + 未实机验证」四要素同构，无一处出现"可用/已支持"。

### 4.4 G5 卫生（明细）

- `cargo test -p prism-core --test qa_ignore_guard` → **4 passed / 0 failed**（`detector_rejects_a_real_ignore`、`detector_ignores_prose_mentions`、`scan_covers_the_workspace`、`no_ignore_anywhere_in_workspace`）
- `grep -rn '#\[ignore\]' crates/ --include=*.rs` → 命中全部为 doc 注释里的**散文引用**（守卫守则/历史说明），可编译属性 **0**——由 `no_ignore_anywhere_in_workspace` 在剥离注释后判定
- `TODO|FIXME|unimplemented!|todo!` → 5 处，逐条溯源均为 V2.2 之前（`a244fde` v1.0 ×3、`9352d70` v2.0-M3 ×2），语义为"原为 `unimplemented!()`，已修"这类**历史说明**。**V2.2 新增占位符 = 0**
- `node tools/hygiene_scan.mjs` → **500 处**（基线 445，**+55 / +12.4%**）。C-3 已按作者明示移出本期，**不要求 exit 0**；构成 6 类（E 内部管理词 135 / F 内部编号 178 / D 内部文档名 25 / G 排期术语 73 / A 角色称谓 67 / C 过程文档术语 19），全部是**文档内部叙述**，与 TODO/占位符无关（V2.2 新增占位符实测 0）。其中 3 处由本次 QA 补登的 CHANGELOG/PRD 文字产生，如实计入。**不判红，列观察项**（见 §5 U4）

### 4.5 G6 黄金值位移审计

| 位移类型 | 位置 | 登记状态 | 处置 |
|---|---|---|---|
| `.golden` 文件 | 全仓仅 `crates/prism-gpu/web/hair_golden.json` | V2.2 **未改动** | 无需登记 |
| 基线 JSON | `tools/first_frame_baseline.json`、`tools/size_baseline.json` | V2.2 **新增**（T10 首次建基线，非位移） | 无需登记；阈值已写明 |
| **T12 多散射补偿** | `prism-rt/src/bsdf.rs` | ✅ 已登记 `R-2026-xx-05`（四要素齐 + 逐通道实测比值 3.85×/18.92×/49.83×） | 通过 |
| **A-2 Vulkan 曲线（留名态）** | `prism-gpu/src/vulkan/spirv_rt.rs` | ✅ PRD §7 第 1 行 | 通过（但该行已被 W3 点亮取代，见下） |
| **A-2 Vulkan 曲线（点亮态）** | 同上 | ❌ **未登记**——PRD §7 仍写「曲线段未实施 / `VULKAN_CURVE_PATH=false`」，与 W3 实际「已点亮 / =true」矛盾 | **QA 已补登** PRD §7 第 2 行（见 §6 B5） |
| **T08 折射法线 + Fresnel 口径** | `prism-rt/src/bsdf.rs` + `wavefront.rs` | ❌ **未登记**（CHANGELOG / PRD §7 均无条目） | **QA 已补登** `R-2026-xx-06`（见 §6 B4） |
| 测试内 `expected` 常量 | 无改动（14 行删除为 `scene_kernel` 断言迁移，非数值改动） | — | 通过 |

**"有无表外位移"结论**：修复后 **0 处表外位移**。两处未登记项均已按既有四要素格式补登，且各自带实测旧值/新值。

### 4.6 G8 对外口径（抽查 6 条，实际打开核实）

| 声称 | 锚点 | 实际内容 | 判定 |
|---|---|---|---|
| DNG 支持 1/7/8（含 32946） | `raw.rs:3282,3287` | `3282: 7 => read_raw_samples_lossless(…)`、`3287: 8 \| 32946 => read_raw_samples_deflate(…)` | ✅ 相符 |
| 合成端拒绝未支持取值 | `raw_dng_synth.rs:260` | `260: /// 压缩方式（**只支持 1 / 7 / 8 / 32946**；其余值由 synthesize_dng 结构化拒绝）` | ✅ 相符 |
| 曲线图元仅 CPU 路径 | `prism-vector/src/path.rs:8` | `8: 平铺（flatten）委托 prism-raster::path::PathBuilder——不重复造曲线细分轮子` | ✅ 相符 |
| 硬编不 link 任何平台库 | `prism-ffi-hwenc/src/lib.rs:14` | `14: **第二档不是"能编码"**…本 crate 的对外表述止于"VA-API 绑定就绪（第二档）"` | ✅ 相符 |
| WebGPU 条件可行 / Chrome·Edge 优先 | `v2.1-WebGPU可行性评估.md:13,152` | `13: 条件可行（Conditionally Feasible）…4–6 人日`、`152: 明确支持矩阵（Chrome/Edge 优先）` | ✅ 相符 |
| 加速比不写数字 | `README.md:228` | `228` = GPU 三路径表 **Vulkan 行**：「Mesa lavapipe 软件 ICD 上无 GPU 可验证」——**语义正是**该行要的"lavapipe 无 GPU 可验证"（曾疑 off-by-one，实测为正确锚点） | ✅ 相符 |

**九语覆盖**：边界表 8 行 × 9 语言 = 72 行，逐行计算机判定「以 `\|` 开头且含 `` `文件:行号` ``」→ 每语言 **8/8**，无"叙述性表述"行。
**一处结构性问题**：README 边界表 `:320-327` 的 8 行是「边界」而非「已交付待验 7 项」，两者对象不同（见 §5 U1）。

### 4.7 G9 T12 钉子 × T08 改动（交叉验证）

`bsdf.rs` 在 V2.2 被 3 次改动（W0 T12 登记补齐、W1 T05 材质库扩展、W2 T08 折射修复）。核实结果：

| T12 钉子 | 位置 | 状态 |
|---|---|---|
| `multiscatter_compensation_monotonic_in_f0` | `bsdf.rs:1065` | ✅ ok |
| `white_metal_compensation_equals_white_furnace_form` | `bsdf.rs:1110` | ✅ ok |
| `old_colored_multiscatter_form_is_dead`（旧形式**必须判红**的负向自证） | `bsdf.rs:1137` | ✅ ok |
| `eval_is_deterministic`（两遍逐位一致） | `bsdf.rs:1213` | ✅ ok |

- T05 把补偿项从 `eval` 内联**原样抽出**为 `multiscatter_ms`（公式 `(1−E_o)(1−E_i)/(π(1−E_avg)) · F_avg²·E_avg/(1−F_avg(1−E_avg))` 逐字未变），`eval` 仅多了一个各向异性单散射分支；单散射各向同性分支仍走原 `spec_scatter`，`anisotropic_degenerates_to_isotropic_bit_exact` 钉住"退化位级不变"。
- T08 只改 `sample()` 的 transmission 瓣（`n_in` 法线翻转 + `cos_i_s` Fresnel 口径），**未触碰** `eval` / `multiscatter_ms` / `energy_dir` / `energy_avg`。因此不存在"折射修复削弱多散射断言"的路径。
- `cargo test -p prism-rt --release` → **275 passed / 0 failed / 0 ignored**（该 crate `#[test]` 由 255 增至 275，+20 全部为 T08 光谱与 T05 材质新增）。

**G9 裁决：PASS。**

---

## 5. 发现并修复的 bug（6 处，均为文案/登记，未改代码逻辑）

| # | 严重度 | 根因（一句话） | 改动文件 | 修复后 |
|---|---|---|---|---|
| **B1** | **中（G4 越级口径残留）** | W3 的"铁律纠错"把 MoltenVK 表述清了零，但**漏改 `metal/` 模块文档**：仍以「探测链就绪」作为 Metal 的状态标签，而 V2.2 PRD G4 已裁决「Metal 不适用『探测链』态，只落『不提供/不点亮』」 | `prism-gpu/src/metal/mod.rs`、`metal/render.rs`、`metal/loader.rs`、`metal/msl/rt_scene.scene_main.msl`、`src/lib.rs` | Metal 侧「探测链」字样 **9 → 0**（`mod.rs` 6 + `render.rs` 1 + `loader.rs` 1 + `lib.rs` 1），另 `.msl` 注释 1 处一并改；「探测链就绪」全仓 **0**；口径统一为「本仓不提供独立 Metal 后端（不点亮）+ 历史资产 + `is_available()` 恒 false」 |
| **B2** | **中（设计 §1.3.2 未落实）** | 设计明文要求「`metal/render.rs:1-30` 四分述表与 README 逐字对齐，统一明确『不提供独立 Metal 后端』」，但该表仍写「同构实现」并列出 ✅ 项，与 README「不提供独立后端」不一致 | `prism-gpu/src/metal/render.rs:1-25` | 表头新增「定性（V2.2 统一口径）」段，逐字对齐 README；四分述表改为「历史资产」语义（技术内容一条未删） |
| **B3** | 低（注释自相矛盾） | `multiscatter_ms` 的注释有**两段互相打脸**的说明：外层段写「两轴**较大者** α = max」，内层段与代码写「**较小者** α = min(α_t, α_b)」；函数 doc 又写「几何均值 α = roughness²」——三处三种说法 | `prism-rt/src/bsdf.rs:592-604` | 合并为一段，统一为「**较小者 min**」并给出"为何不能取 max/α_g（白炉 R 冲到 1.07+ 凭空造能）"的理由；该处 **+4 / −12 行**（净 −8） |
| **B4** | **高（G6 未登记位移）** | T08 修了折射界面法线朝向 + Fresnel 余弦口径——这是**改变输出字节**的行为修复（玻璃从"不透明遮挡"变为可透视、色散可见），但 CHANGELOG 与 PRD §7 **均无条目**，违反 C15「造成输出字节变化必须进渲染行为变化清单」 | `CHANGELOG.md`（新增 `R-2026-xx-06`） | 四要素齐；含实测旧值（`ior 1.0/1.5/2.5` 三帧**逐位相同**，等效分离度 0）→ 新值（色心分离 **2.504 px**，阈值 1.0；`cauchy_b=0` 对照 0.917 px） |
| **B5** | **中（G9 变更记录过期）** | PRD §7 的 A-2 行仍写「Vulkan `spv_scene` 曲线段**未实施**」+「`VULKAN_CURVE_PATH=false`」，而 W3 已实施并**真点亮**（=true）。读者按 §7 会得出与代码相反的结论 | `docs/v2.2-增量PRD.md` §7（增第 2 行 + 给旧行留名理由加"已取代"限定） | 新增 W3 交付行（六绑定/参数块 22→24/栈式 BVH/四图元 0 差异实测）；旧行保留为历史并显式标注"已被本条取代" |
| **B6** | 低（锚点位移跟随） | B1/B2 使 `metal/mod.rs` 文档块 +3 行，`is_available` 由 `:189` 移至 `:192`，九语 README 的 `metal/mod.rs:9,189` 锚点随之失效 | `README.md` + `docs/README.{en,de,es,fr,ja,ko,pt-BR,ru}.md`（各 1 处） | 九语统一改为 `metal/mod.rs:9,192`；实测 `:9` = 「is_available 在所有平台都刻意返回 false」、`:192` = `fn is_available` |

**修复后回归**：

```
cargo test -p prism-gpu --release   → 309 passed / 0 failed / 0 ignored（改动前后同值）
cargo test -p prism-rt  --release   → 275 passed / 0 failed / 0 ignored（改动前后同值）
cargo xtest                         → 4008 passed / 0 failed / 0 ignored / 188 段（改动前后同值）
```

即：6 处修复**零测试影响**，符合"只改文字、不改逻辑"的授权边界。

---

## 6. 未修复 / 需总监裁决项

| # | 级别 | 项 | 为什么不自行修 | 建议 |
|---|---|---|---|---|
| **U1** | **需裁决（PRD 验收条件）** | 交付说明**缺独立的「已交付待验（7 项）」章节**。PRD 验收总账第 23 条原文：「交付说明中必须有一节「已交付待验（7 项）」，逐项写清「差什么设备、验什么、验不了会怎样」——**这一节的存在本身是 V2.2 的验收条件之一**」。现状：7 项只以**表格形式**存在于 PRD §3.0.1（是 PM 的 PRD，不是"交付说明"）；`README.md` 与 `docs/v2.2-release-note-draft.md` 均无此章节（release note 只有 37 行、3 个章节，标注"草稿 · 待 T02 九语同步"） | 这不是"改一句文案"，而是**新增一节并九语同步**（8 个文件 + 中文），涉及对外承诺的正式表述，属 PM/总监口径权；QA 单方增写会绕过 PRD 评审 | 二选一：① 由总监确认「PRD §3.0.1 的 7 行表即视为登记表，验收 23 以 PRD 为准」→ 记为**口径澄清**，无需改 README；② 若按字面要求，请在 release note 补 §「已交付待验（7 项）」并同步九语，QA 可提供 §7 的现成 7 行措辞直接取用 |
| **U2** | 观察 | `CHANGELOG.md` **无 V2.2 能力扩容条目**（`F-2026-xx-03`）。V2.2 四条流都只有"行为修复"入账（`R-2026-xx-05`、本次 `-06`），能力扩容清单最新仍是 `F-2026-xx-02`（V2.1）。C15 只强制**字节变化**入账，能力扩容可选择性登记，故**不判红** | 属发布叙事而非门禁；补写需覆盖 DNG 矩阵/曲线图元/SVGF2/材质库/光谱切片/VAAPI/计数器七项，量级接近发布说明，应由 PM 执笔 | 若 V2.2 要走对外分发，建议补 `F-2026-xx-03`；否则在 release note 承载即可 |
| **U3** | 观察（非 V2.2 缺陷） | `dist/prism-engine-v2.0-urgent_fix-source.txt:86660` 与 `dist/prism-engine-v2.1-source.txt:88196` 各留 1 处 `MoltenVK` 字样（原文：「若将来移植 macOS，此常量须复核（macOS 的 Vulkan 走 MoltenVK 且路径不同）」）。两者均为**未被 git 跟踪**的历史版本快照（`git ls-files dist/` 为空），V2.2 快照已 0 命中；该句是**平台事实陈述**，非能力声称，且对应源码注释已在 V2.2 删除（`grep 'macOS 的 Vulkan 走' crates/` → 0） | 删除已发布的历史分发包快照属"销毁发行物"，不在 QA 授权内 | 建议：不动（历史产物）；若要彻底清零，由总监决定是否移入 `dist/legacy/` 或删除 |
| **U4** | 观察（C-3 已移出） | `hygiene_scan` **500 处** vs 基线 445（**+55 / +12.4%**）。6 类全部为文档内部叙述（E 135 / F 178 / D 25 / G 73 / A 67 / C 19），**无一条是 TODO/占位符**（V2.2 新增占位符实测 0）。其中 3 处由本次补登文字产生 | C-3 已被作者明示移出本期，且门禁状态不计入 Go/No-Go | **不判红**。增长主因是 V2.2 文档体量（PRD 397 行 + 设计 + 本报告），非代码退化；建议 V2.3 一并清理 |

> **无源码实现 bug 需要转交工程师。** 本次未发现任何"实现错、测试对"的失败：全量 0 failed，且未发现改阈值/加 `#[ignore]`/删测试等作弊痕迹（专项扫描见 §2.1）。

---

## 7. 已知待验登记项（7 项，只登记、不阻塞）

> 依据 PRD §3.0.1：凡依赖真机的判据一律拆两档，「实机验证」档不构成版本 Go/No-Go。本环境（Linux 沙箱）无 Apple 设备、无真实 GPU（仅 lavapipe 软件实现）、无硬件编码器。

| # | 项 | 差什么设备 | 验什么 | 验不了会怎样 |
|---|---|---|---|---|
| 1 | **V2.2-A-1 WebGPU 真机** | 有 GPU 的 **Chrome / Edge**，≥2 台设备 | 出 `w*h*4` 有效帧；同设备双跑逐字节一致；vs CPU 统计对拍；记录 `GPUAdapterInfo` | 对外只能写「语义在软件实现（SwiftShader）上已判对，剩余风险是驱动浮点差异」——**不得**写成"WebGPU 已支持" |
| 2 | **V2.2-A-3 Metal 实机** | **Apple 设备（团队无，永不）** | macOS 上 `MTLDevice` → command queue → pipeline state → dispatch → 读回闭环 | `is_available()` 恒 `false`、`render_scene()` 恒返空帧；README 保持「不提供独立 Metal 后端（不点亮）」措辞，**不因代码存在而改写** |
| 3 | **V2.2-C-1 DNG 真实机型样本** | 真实相机产出的 DNG 文件（机型样本） | `Compression = 1/7/8/32946` 在真实 strip/tile 布局下的读入；CFA/位深/黑电平 | 维持 README「真实机型未经样本验证」；合成矩阵（`qa_dng_support_matrix` 9/9）只证**自证主路径** |
| 4 | **V2.2-C-2 硬编真机出码** | VAAPI / NVENC / VideoToolbox **硬件编码器（无）** | 建立会话 + 真机出码 + 码流校验 | 停留在「绑定就绪（第二档）」；绝不称"可编码"；三档/四档保持 ❌ |
| 5 | **V2.2-D-1 真实 GPU 加速比** | 目标 GPU 硬件（无） | 真实加速比数字 | **不写任何数字**（lavapipe 为伪值）；README 保持「登记待验」 |
| 6 | **V2.2-D-2 峰值 RSS 绝对值** | 稳定同机基线 | 峰值 RSS 绝对值 | 只报"计数器两次运行完全相等 / 池中性逐字节"这类**相对**判据 |
| 7 | **V2.2-D-3 首帧时间绝对值** | 稳定同机基线 | 首帧时间绝对值 | 首帧报表（中位 20.5 ms）为**非门禁**参考值 |

---

## 8. 最终裁决

| 门 | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 |
|---|---|---|---|---|---|---|---|---|---|
| 结论 | PASS | PASS | PASS | PASS¹ | PASS | PASS² | PASS³ | PASS | PASS |

¹ 修复 B1/B2 前为 FAIL（`metal/` 模块文档残留「探测链就绪」越级口径、四分述表未按设计 §1.3.2 与 README 对齐）；已修，复检 0 命中。
² 修复 B4/B5 前为 FAIL（T08 折射位移未登记、PRD §7 A-2 行与交付状态相反）；已补登 / 订正。
³ 附 U1 需裁决：交付说明缺独立「已交付待验（7 项）」章节（PRD 验收 23）。若总监裁定「PRD §3.0.1 的 7 行表即为登记表」，则本门无条件 PASS；若按字面要求，需在 release note 增节并九语同步。

**IS_PASS: YES**

理由摘要：
1. 全量 **4008 passed / 0 failed / 0 ignored / 188 段**，相对 3929 **净增 +79**，段数 +9 与 9 个新测试文件一一对应；**无删测、无阈值放宽、无 `#[ignore]`、无新增占位符**。
2. 零第三方依赖守住（`cargo tree` 非 prism 输出为空；lock 仅 +1 条仓内 path）。
3. GPU 判据是 **lavapipe 真跑**的硬 0 差异（`max_channel_delta=0 && differing_pixels=0`）+ 双跑逐位一致，不是自门控跳过。
4. 四处（Metal / 硬编 / WebGPU / 加速比）对外口径诚实，MoltenVK 零命中，九语 Metal 行 9/9 同构；发现并修复 6 处越级/缺失/过期文案与登记，全部零逻辑改动、零测试影响。
5. G9 交叉验证确认 T12 四钉子完好且全绿，T08 折射修复未触碰多散射求值面。
6. 剩余 1 项需裁决（U1）为**文档章节归属**问题，不涉及工程判据；7 项实机待验按 PRD §3.0.1 只登记、不阻塞。

—— 严过关（Yan），QA 工程师，2026-10-30
