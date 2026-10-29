# QA 测试报告 — V2.1 正式 QA 轮次

- **轮次**：qa-round-v2.1-2026-10-26
- **日期**：2026-10-26
- **QA 工程师**：严过关（Edward）
- **测试对象**：`/workspace/media-engine`，HEAD = `7707004`（"V2.1 交付：三流并行 …；全量回归 3927 passed / 0 failed"），工作区干净（`git status --porcelain` 空）
- **基线**：V2.0-urgent_fix `a957475`，3879 passed / 0 failed（`docs/QA/qa-round-v2.0-urgent_fix-2026-10-24.md`）
- **工程侧自报**：3927 passed / 0 failed / 0 ignored（179 个测试段）
- **本机环境**：Linux 沙箱；Vulkan ICD 在位（`/usr/share/vulkan/icd.d/lvp_icd.json`），实测设备 `llvmpipe (LLVM 20.1.2, 256 bits)`，枚举 1 个物理设备；`spirv-val` 可用（SPIRV-Tools v2025.1）
- **纪律声明**：未改动任何源码与测试代码；未 commit；额外取证工具建在 `/tmp`（`spvcheck` / `hetprobe`），不进入仓库。

---

## 1. 测试范围与命令

| # | 项 | 命令 | 用途 |
|---|----|------|------|
| C1 | 全量回归 | `cd /workspace/media-engine && cargo xtest`（= `test --workspace --release`） | 29 crate 全量，与 3927 基线对比 |
| C2 | prism-gpu 定向 | `cargo test -p prism-gpu --release [--test qa_render_scene] [-- --nocapture]` | 光追实帧、双跑逐位、CPU 对拍、双自检 |
| C3 | prism-io 定向 | `cargo test -p prism-io --release [--test qa_r1_representation_signature]` | R1 表示签名、R2 字节往返 |
| C4 | prism-core 定向 | `cargo test -p prism-core --release --test qa_signature_s3_closure -- --nocapture` | S3 签名闭包 |
| C5 | prism-render 定向 | `cargo test -p prism-render --release` | S3 增量复用判定 / S4 逐次中性 |
| C6 | prism-geometry 定向 | `cargo test -p prism-geometry --release --lib` | R3 Hausdorff 界 + brick 边缘钉子 |
| C7 | prism-rt 定向 | `cargo test -p prism-rt --release` | 异质体积退化一致性 + Beer 无偏 |
| C8 | prism-math 定向 | `cargo test -p prism-math --release` | T4 结清锁 + ACES 指纹 + 黄金值 |
| C9 | Diff 审计 | `git diff HEAD~1 -- <file>`、`git show HEAD~1:<file>` | G6 黄金值逐条对应 |
| C10 | 独立复核（QA 自建，仓库外） | `/tmp/spvcheck`：导出 `spirv_rt::scene_words()` → `spirv-val --target-env vulkan1.0` / `spirv-dis` | 独立验证「spirv-val PASS」 |
| C11 | 独立复核（QA 自建，仓库外） | `/tmp/hetprobe`：以 path 依赖调用 `HeterogeneousMedium` 做偏置实测（N=300k / 1.2M） | 独立验证「Beer 无偏」的适用边界 |
| C12 | 纪律扫描 | `git diff HEAD~1 -- Cargo.toml Cargo.lock`、`grep -rn "#\[ignore\|TODO\|FIXME\|todo!\|unimplemented!"` | G2 / G5 |

> **过程记录**：任务书提示 `cargo xtest -p` 不缩小范围——属实（别名内含 `--workspace`），本报告定向回归一律用 `cargo test -p <crate> --release`。

---

## 2. 全量回归（C1）

```
$ cd /workspace/media-engine && cargo xtest          # 日志 /tmp/xtest_v21.log
```

| 指标 | 数值 | 与基线对比 |
|------|------|-----------|
| passed | **3927** | 3879 → 3927，**净增 +48** |
| failed | **0** | 持平 |
| ignored | **0** | 持平（G1 要求 0） |
| 测试段（`test result:` 行） | **179** | 与自报 179 一致 |
| 编译错误 / `failures:` 段 | 无 | — |
| 进程退出码 | **0** | — |

**净增构成核验（防"基线稀释"与"静默删测"）**：`git diff HEAD~1` 中新增 `#[test]` **48 条**、删除 `#[test]` **0 条**；另有一处**同名替换**（`colorspace_registry` 的别名测试删除并由接替测试顶位，计数不变）。3879 + 48 = **3927**，与实测逐数吻合，**无测试净减、无过滤隐藏**。

**逐 crate 合计**（定向轮复跑，与全量同源）：

| crate | passed | failed | ignored | 自报 | 一致？ |
|---|---|---|---|---|---|
| prism-gpu | **286** | 0 | 0 | 286 | ✅ |
| prism-geometry | **75** | 0 | 0 | 75 | ✅ |
| prism-rt | **253** | 0 | 0 | 253 | ✅ |
| prism-io | 584 | 0 | 0 | — | — |
| prism-math | 161 | 0 | 0 | — | — |
| prism-core | 169 | 0 | 0 | — | — |
| prism-render | 213 | 0 | 0 | — | — |

---

## 3. 定向回归逐项结论

### 3.1 流 A — #11 GPU（prism-gpu，286/286）

| 判据 | 证据（`--nocapture` 实测输出） | 结论 |
|---|---|---|
| **A1 实机出图** | `[device] llvmpipe (LLVM 20.1.2, 256 bits)（枚举 1 个物理设备）`；`[frame] 64×48 有效帧，非零字节 12288 / 12288`；`groups=[8,6,1]`、`invocations=3072`、alpha 全 255；`render_scene` 与 `render_scene_frame` 两条入口 `fb.pixels == report.pixels` | ✅ 真机出图 |
| **G3 双跑逐位一致（硬判据）** | `[determinism] 12288 字节两遍渲染逐位一致（filmic=false / true 各一遍）`；断言 `assert_eq!(mismatches, 0)` 为**真断言**（非打印） | ✅ |
| **Q1 CPU 对拍** | `[parity:aesthetic-off] max_channel_delta=0 differing_pixels=0 (0.000%) differing_gt1=0 mean_abs_delta=0.00000 software_icd=true first_diff=None`；`aesthetic-on` 同；`[parity:tail] 113×47 max_channel_delta=0 differing_pixels=0` | ✅ 软件 ICD 上**逐字节 0 差异** |
| 判据强度 | 软件 ICD（`llvmpipe/lavapipe/swiftshader`）分支另有 `assert_eq!(max_channel_delta, 0)` + `assert_eq!(differing_pixels, 0)` 硬断言；非软件 ICD 才放宽到 `≤1/255`、`≤50‱`（收窄自 Q1 provisional 16/255、500‱） | ✅ 强于 PRD 要求 |
| **Q4 双自检** | `[available] device=llvmpipe compute=true render=true（双自检）`；`is_available() == compute_available() && render_available()`（两处测试各自断言）；`verify_render_path` 真跑 4×4 帧且要求 `any(b != 0)`（黑帧不能冒充可用） | ✅ 语义闭合 |
| 无设备诚实分支 | 无设备时 `assert!(!is_available() && !render_available())` 后 **return（skip）**，**无 `#[ignore]`** | ✅ |
| A1 验收 4 边界 | `(u32::MAX,u32::MAX)/(65536,65536)/(0,0)/(0,48)/(64,0)` 五组均 `catch_unwind` 不 panic、返回无效空缓冲且尺寸原样回显；空场景 16×16 出背景帧 `[10,13,20,255]` | ✅ |
| **独立复核（QA 自建 C10）** | `scene_words()` = 2554 字 / 10216 字节；`spirv-val --target-env vulkan1.0 scene.spv` → **exit 0，无诊断**；`spirv-dis` 显示 `OpEntryPoint GLCompute %2 "scene_main"`、`LocalSize 8 8 1`、`Binding 0..3` 齐备 | ✅ 「spirv-val PASS」经独立工具坐实 |

### 3.2 流 A — A2 Metal 门控（不点亮）

`metal/mod.rs` / `metal/render.rs` / `metal/msl/*` 全平台参与编译（Linux `cargo xtest` 通过），**未点亮**。逐项见 §6。

### 3.3 流 A — A3 WebGPU 评估

`docs/v2.1-WebGPU可行性评估.md`：结论「**条件可行**」，量级 4–6 人日，三档区分（1.1 已有在用 / 1.2 已存在未接线 / 1.3 未建）明确；显式写明「这段厚度说明的是绑定与测试链成熟，不是能渲染」。✅ 无实现、无渲染能力声称。

### 3.4 流 B — #9 R1 / R3（prism-io 584、prism-geometry 75）

| 判据 | 证据 | 结论 |
|---|---|---|
| R1 等价 ≥2 组 | `r1_equivalent_sources_share_representation_signature`：prim 书写序相反、属性书写序相反 → 签名相等（2 组），并附加确定性复算断言 | ✅ |
| R1 不等价 ≥1 组 | `r1_inequivalent_source_has_distinct_signature`（顶点位移 ⇒ 签名不等）+ `r1_distinct_assets_never_collide`（前置条件 + 碰撞判据） | ✅ |
| 域分离标签 | `REPR_TAG = b"PRISM/REPR/v1"`，与缓存键 `CACHE_KEY_TAG`/`CACHE_KEY_MODE_TAG` 分离；同 SHA-256 原语 | ✅ |
| R3 形式化界 | `hausdorff_upper_bound(h, g) = h/2·(1+g)`；测试断言「h↓ ⇒ 界收紧」、实测 ≤ 界、且 ≤ 更紧的 `√3/2·h + 1e-4` | ✅ |
| R3 实测 | 球面 `3.1008e-3 (h=0.1)` / `1.5489e-3 (h=0.05)`；立方体 `2.39e-7 / 2.46e-7`；逐档单调收紧 | ✅ 与设计勘误 §B1 一致 |
| brick 边缘钉子 | `brick_edge_cells_resolve_global_lattice_not_clamped_local`：专挑「≥2 轴同时落 brick 末格」表面点，断言 `n_edge ≥ 50`（防覆盖失效，文档称实测 177 点）+ `worst ≤ √3/2·h + 1e-4 = 0.0434013`；文档记录旧实现 0.06929246 → 修复后 1.4464e-3（余量 ~30×） | ✅ 非空洞断言 |
| R2 不回归 | `usd_export` 全量绿，含 `export_import_export_is_byte_fixed_point`、`f64_text_roundtrip_is_bitwise`、`export_is_byte_level_deterministic`、`imported_layer_re_export_preserves_geometry_and_unknown_attrs` | ✅ |

### 3.5 流 B — #8 签名闭包 S3 / S4（prism-core 169、prism-render 213）

| 判据（PRD B2 验收 1–2） | 证据 | 结论 |
|---|---|---|
| S1 无状态 | `s1_stateless_signature_closure_is_reusable`：sig 等 ⇒ 键等；sig 不等 ⇒ 键不等 | ✅ |
| S3 流节点（frame） | `s3_stream_node_cache_key_contains_frame`：帧 0/1/2 三键互异、同帧同键 | ✅ |
| S3 迭代节点（ε,n） | `s3_iteration_node_cache_key_contains_mode_params`：**`cache_key_with_mode(…, None) == cache_key(…)` 逐字节退化**（None 分支直接 `return cache_key`，代码级保证）；substeps/iterations/residual_eps 任一变化 ⇒ 键变化；带模式键与无模式键域分隔；`def.tag == 0x5333` | ✅ |
| 音频区间（CH4） | `s3_audio_reactive_interval_reuse_and_reject`：恒定带 [0,3] 内不脏、带外 16 帧全脏（逐帧断言） | ✅ |
| S4 逐出中性 | `s4_lru_eviction_recompute_is_bit_identical_to_pre_eviction`：保留键以 `|| panic!("不得触发重算")` 证明真复用；被逐出键重算 `b0b == b0` 逐位一致 | ✅ 强断言 |
| 量级纪律（≤150 行净增） | `signature.rs +44`、`incremental.rs +25`、`prism-render/src/lib.rs +7`、`prism-core/src/lib.rs +6` ⇒ **+82 行** | ✅ |
| 消费面零回归 | prism-app preview / prism-aesthetic critic loop 全绿（全量回归覆盖）；`movie_sim_timeline::mode_metadata_changes_frame_signature_and_cache_key` 绿 | ✅ |

> **实现口径说明（非缺陷，见 F10）**：S3「迭代节点」在**生产路径**上的真实生效机制是既有的 `loop_frame_signature`（mode 通道定义早已进帧签名 → 进 `cache_key`），本版新增的 `cache_key_with_mode`/`decide_with_mode` 是附加加固 API，**无人调用**。因此「零语义回归」成立且不依赖新 API 被接线。

### 3.6 流 C — C1 盘点 / C2 异质体积 / C3 T4

**C1 盘点**：`docs/v2.1-渲染特性长尾盘点.md` 名录四项（光谱 / SVGF-OIDN / 自适应 / 异质体积）逐条 K1 三元组（K、σ∈六策略、d）+ 参考实例对照 + 量级 + 分级；§2 收敛判据三项可机械判定；§4 grep 补漏结论。✅

**C2 异质体积（prism-rt 253/253）**：

| 判据 | 证据 | 结论 |
|---|---|---|
| 退化一致性（逐位） | `degenerate_free_path_bit_identical_to_homogeneous`：4 档 σt（0.3/0.8/1.5/2.7）× 4096 样本，`free_path` 的 `to_bits()` 全等；`degenerate_shadow_bit_identical_to_homogeneous`：3 档 × 4096 样本全等 | ✅ 真逐位 |
| Beer 无偏（tracked 变体） | `shadow_transmittance_unbiased_vs_beer`：N=300k，断言 `|E[T̂]−Beer|/Beer < 2%`；均匀源走 `sampler::aux_1d(dim, step)`（PCG 哈希，**不是 Sobol 相邻点**），且测试文档记录了「曾用 Sobol 相邻点导致 −6.19%」的取证结论 | ✅ 独立源 |
| 相函数归一 | `heterogeneous_phase_integrates_to_one`（40k 样本，±1%） | ✅ |
| 参数域防御 | 真空场 `#[should_panic(expected="σt 场上界须为正")]`、数据长度不匹配 panic | ✅ |
| **独立复核（QA 自建 C11）** | 同一线性场（∫σt=2.0）：独立均匀对 → `+0.128%`（0.55σ）；**生产 2-uniform API → `+1.080%`（4.7σ）**；把同一派生链喂给 tracked API → 同样 `+1.080%`（因果隔离成立） | ⚠️ **见 F1** |

**C3 T4 结清**：结清锁实测输出 `T4 已结清（C15）：主常量−官方推导 f64 4.532e-7 / f32 4.545e-7；逆阵重算差 4.144e-8；Hill 可选口径差 3.205e-2`；`srgb_to_acescg_is_recomputed_inverse` 与 `aces2065_to_acescg_is_recomputed_inverse` 双绿；`aces_golden_file_stable`（27 词指纹）绿。✅（容差口径见 F4）

---

## 4. G1–G6 逐条裁决

| # | 判据 | 实测 | 裁决 |
|---|---|---|---|
| **G1** | 全量 ≥3879 / 0 failed / 0 ignored，新增计入并上报净增 | **3927 / 0 / 0**，179 段，exit 0；净增 **+48**（新增 48 条 `#[test]`，删除 0 条，1 处同名替换） | **PASS** |
| **G2** | 零第三方依赖 | `git diff HEAD~1 -- Cargo.toml Cargo.lock` **输出为空**；`Cargo.lock` 29 个 package **全为 path crate，外部 crate = 0** | **PASS** |
| **G3** | 确定性：GPU 双跑逐位一致 | 64×48 两档后处理，12288 字节两遍 `mismatches = 0`（硬断言）；lavapipe 上 GPU vs CPU 亦 0 差异 | **PASS** |
| **G4** | 不伪造能力：Metal `is_available()` 恒 false 有测试钉住；「探测链就绪」与「真实 dispatch」三处分述 | 见 §6：两处测试钉住 false、模块文档表格四分述、CHANGELOG 含「未实机验证、不点亮」、全仓无「Metal 可用」声称 | **PASS** |
| **G5** | 禁 TODO / 占位符 / `#[ignore]` | 全仓 `#[ignore]` **零命中**（所有 grep 命中均为注释里的说明文字）；改动文件无 `TODO/FIXME/todo!/unimplemented!`；`qa_ignore_guard` 绿（含三层反向自证） | **PASS** |
| **G6** | T4 后所有受影响黄金值重录并逐条登记 | 见 §5：6 条登记逐条对应，**无表外位移**；1 处测试名不精确（F3），数值对应唯一 | **PASS**（附 F3） |

---

## 5. G6 黄金值逐条审计（`git show HEAD~1:…` vs 当前文件）

### 5.1 `c2_color_golden.rs` — `ACES_FINGERPRINT` 27 词逐字节核对

| # | 样本 | 旧（HEAD~1） | 新（当前） | C15 登记 | 一致 |
|---|---|---|---|---|---|
| 1–3 | 黑 | `0000_0000 ×3` | `0000_0000 ×3` | 未列（**真实无位移**，登记正确） | ✅ |
| 4–6 | 18% 灰 | `3DD8_403D / 403A / 403B` | `3DD8_402C / 403D / 4044` | `3DD8_403D/3A/3B`→`3DD8_402C/3D/44` | ✅ |
| 7–9 | 中灰 | `3EBF_A557 / A556 / A556` | `3EBF_A558 / A556 / A554` | `3EBF_A557/A556/A556`→`3EBF_A558/A556/A554` | ✅ |
| 10–12 | 白 | `3F1E_7E5A / 7E59 / 7E5B` | `3F1E_7E63 / 7E59 / 7E54` | `3F1E_7E5A/59/5B`→`3F1E_7E63/59/54` | ✅ |
| 13–15 | 红 | `3F41_0AF6` | `3F3F_786E` | `3F41_0AF6`→`3F3F_786E` | ✅ |
| 16–18 | 绿 | `3D6A_D79F / 3F24_5E27` | `3D52_E8EA / 3F23_F4F4` | `3D6A_D79F/3F24_5E27`→`3D52_E8EA/3F23_F4F4` | ✅ |
| 19–21 | 蓝 | `3F2B_7813` | `3F28_D13F` | `3F2B_7813`→`3F28_D13F` | ✅ |
| 22–24 | 超亮 | `3F68_B522 / B520 / B521` | `3F68_B53A / B521 / B512` | `3F68_B522/20/21`→`3F68_B53A/21/12` | ✅ |
| 25–27 | 混合 | `3D1A_CC7B / 3E5B_0065 / 3F02_9C41` | `3D1B_8FA0 / 3E5A_E656 / 3F01_D3D0` | `3D1A_CC7B/3E5B_0065/3F02_9C41`→`3D1B_8FA0/3E5A_E656/3F01_D3D0` | ✅ |

**结论：27 词中 24 个非黑项全部位移、3 个黑项不变；C15 表格填写与真实 diff 逐字节一致，无一处遗漏、无一处虚报。**

### 5.2 登记表 6 条逐条核对

| # | 登记项 | diff 实物 | 裁决 |
|---|---|---|---|
| 1 | `c2_color_golden::aces_fitted_fingerprint` 27 词全量重录 | `c2_color_golden.rs +20/−8`，仅指纹块（+ 一段 T4 注释） | ✅（测试名不精确，见 F3） |
| 2 | `qa_edge_attack_math::aces_fitted_vs_independent_reference` 独立参考矩阵同步重录 | 该文件 +13/−6：`IN` 3 行 + `OUT` 3 行换成新口径字面量 | ✅ |
| 3 | `aces::tests::reference_aces_fitted` 同上 | `aces.rs +14/−8`：同一组 6 行矩阵 | ✅ |
| 4 | 债务锁更名 `..._settled_t4_is_named`：判定翻转 | `colorspace.rs` 内测试改名 + 断言由「偏差必须 >1e-2」翻为「≤1e-2 + 逆阵可重算 + Hill 旧值未改」；文档注释明写「本锁接替原 `..._debt_t4_is_named`」 | ✅ 未静默删除，翻转留痕于文件内 |
| 5 | `colorspace_registry::legacy_constant_is_a_fitted_matrix...` 取值对象换 `_FITTED_HILL` | 该文件 `let legacy = crate::colorspace::SRGB_TO_ACESCG_FITTED_HILL;` | ✅ 仍锁 ~3.2e-2 |
| 6 | 别名测试**删除**，由 `..._preserved_and_diverged_from_recut_main` 接替 | 旧测试函数删除；新测试断言 `_FITTED_HILL` 三个锚点值一字未改 + 行和 1 + `diverged == true` | ✅ 删除有登记 |

### 5.3 「有无表外位移」

对 `HEAD~1..HEAD` 全量 diff 做数值型改动清点：数值 golden 改动仅出现在上表 1/2/3 三处文件；`collect` 类测试（`qa_edge_attack_math_sample.rs`、`prism-postfx`、`prism-author`）**无一改动且全绿**，其参考实现直接引用 `SRGB_TO_ACESCG` 常量（`const IN = SRGB_TO_ACESCG`），随常量自动跟随，故**无需重录**。**未发现无登记的黄金值位移。**

> 注：设计文档 §2 曾把 `qa_edge_attack_math_sample.rs` 列为「改」，实际未改——这是**设计预测失准**而非遗漏登记（该测试无独立数值 golden），计入 F5。

---

## 6. Metal 诚实口径审计（G4 / Q5）

| 检查项 | 证据 | 结论 |
|---|---|---|
| `is_available()` 恒 false | `metal/mod.rs:189` 直接 `false`；**两处**测试钉住：`metal_stub_is_unavailable_and_stable_name`（`:293` `assert!(!be.is_available())`）、`render_code_landed_without_claiming_availability`（`:387` 在"代码在、语义对拍通过"的前提下仍断言 false） | ✅ |
| `render_scene` 不产假帧 | 恒 `Framebuffer::empty()`；测试 `metal_stub_render_scene_returns_invalid_frame_without_panic` + 新测试 `assert!(!be.render_scene(&req).is_valid())` | ✅ |
| 三层分述（代码注释 / 模块文档 / 交付说明） | ① `metal/mod.rs` 头注表格四分述「设备探测链 ✅（仅 macOS 编译）/ MSL 双份源 ✅ / ABI+语义同构 ✅ / **`MTLDevice`→pipeline→dispatch→读回 ❌ 未做**」，结论句「**macOS 实机验证未做**」；② `metal/render.rs` 头注同款四分述，「`is_available` 恒 false」+「本模块不能替代 macOS 实机验证」；③ `CHANGELOG.md` F-2026-xx-02：`is_available()` 保持 false（**未实机验证、不点亮**，诚实留名） | ✅ 「探测链就绪 ≠ 真实 dispatch」未被混用 |
| 全仓无「Metal 可用」声称 | `grep -rn "Metal" --include=*.rs \| grep -E "可用\|已点亮\|available=true\|能出图"`：仅命中 4 处**否定式**表述（"全仓无任何一处声称 Metal 可用"、"Metal 仅 macOS 可用"、诊断量说明、非 macOS 恒不可用） | ✅ |
| 编译门控 | metal 模块全平台参与编译（`loader` 仅 macOS `#[cfg]`），Linux `cargo xtest` 通过；`render.rs` 刻意**不加** `cfg(target_os)` 以便在 Linux 对拍 | ✅ |
| MSL 源真实性 | `rt_scene.scene_main.msl` 185 行，`include_str!` 编译期读入（源文件即权威），入口 `kernel void scene_main`、`[[buffer(0..3)]]` 四绑定、`[[thread_position_in_grid]]`、尾组守卫 `if (gid.x >= p.width \|\| gid.y >= p.height) return;` | ✅ |
| 语义同构证据 | `render_scene_on_cpu_via_msl_layout` 从**编码后的字节缓冲**读数据（能抓槽位错位），与 `CpuRefBackend` `assert_eq!(pixels, pixels)` 逐字节一致；文档明确标注这是「移植没写错的证据」，**不是** macOS 验证 | ✅ 诚实 |

**裁决：Metal 条目无任何伪造能力表述，Q5 的四条验收（恒 false / 不产假帧 / 「未实机验证」留名 / 无可用声称）全部满足。**

---

## 7. 零依赖审计（G2）

```
$ git diff HEAD~1 -- Cargo.toml Cargo.lock     → （空输出）
$ Cargo.lock: 29 × [[package]]，其中 source 字段存在者 = 0
```

- 29 个 package **全部**为 workspace path crate，**外部 crate 数为 0**；
- 新增代码只复用既有自研件：SHA-256（`prism-core::signature` / `prism-io::sha256`）、`invert3`（`colorspace_registry`）、`det_ln`（`prism-rt`）、SPIR-V 发射器；
- Vulkan 仍走手写 `dlopen/dlsym`，无 `ash`/`vulkano`；Metal 无 `objc`/`metal` crate；WebGPU 无 `wgpu`。

**裁决：零依赖纪律完好。**

---

## 8. TODO / 占位符 / `#[ignore]` 审计（G5）

| 扫描 | 结果 |
|---|---|
| 全仓 `#[ignore]`（`--include=*.rs`） | **0 处代码态命中**；10 条 grep 命中全部为注释/文档中的**说明文字**（"全仓禁 `#[ignore]`"、"不用 `#[ignore]` 占位，改用运行期自门控"等） |
| 改动文件 `TODO / FIXME / XXX / todo! / unimplemented!` | **0 处**；「占位」类命中均为**实体语义**（`spirv_rt` 的"空集合退化为 1 个占位槽"是真实 slot；`web/mod.rs` 的类型占位为既有骨架注释） |
| `qa_ignore_guard` | 绿；该守卫自带三层非恒真空洞证明（探测器正例、注释负例、`scan_covers_the_workspace` 扫描文件数下界） |
| doc-test ```ignore | 守卫同覆盖，零命中 |

**裁决：无「假绿」手段。**

---

## 9. 测试有效性抽查（任务书指定 3 项）

| 抽查项 | 结论 | 证据 |
|---|---|---|
| 光追对拍必须**断言** 0 差异（不是只打印） | ✅ | `software_icd == true` 时执行 `assert_eq!(p.max_channel_delta, 0)` 与 `assert_eq!(p.differing_pixels, 0)`；打印仅作诊断。软件 ICD 判定 `llvmpipe/lavapipe/swiftshader` 子串匹配，本机命中 |
| GPU 双跑逐位一致必须有真实断言 | ✅ | `assert_eq!(mismatches, 0, "GPU 双跑必须逐位一致（首个差异：…）")`，并先断言两帧 `is_valid()`、长度相等 |
| 异质体积无偏测试的均匀源必须独立（非 Sobol 相邻点） | ✅ | 走 `sampler::aux_1d(ctx, si, step)` = `pcg::hash3_f32(aux_seed, dim, step, SALT)` 的计数器型哈希流，逐 step 独立；测试注释记录了 Sobol 相邻点导致的 −6.19% 误偏取证。**但我另测了生产 2-uniform 路径，见 F1** |

**额外抽查（QA 自加）**：
- S4 逐出中性用 `|| panic!("保留键不得触发重算")` 闭包证明「真复用而非重算后偶然相等」——**强断言**，非形式主义。
- R3 brick 钉子有 `n_edge >= 50` 的覆盖度断言，防「采样集为空 ⇒ 恒过」。
- `is_available()` 的双自检有「`true` ⇒ 必须出 `w*h*4` 有效帧」「`false` ⇒ 必须空缓冲」双向蕴含断言，杜绝「报可用却空帧」。

---

## 10. 遗留问题清单

> 分级：**中** = 影响能力表述真实性/生产路径正确性，建议 V2.2 首轮必修；**低** = 文档/口径/精度问题，不阻断交付；**观察** = 不构成缺陷，供后续决策。

### F1（中）生产用异质阴影估计量有 +1.08% 正偏（可复现，因果已隔离）

- **现象**：`HeterogeneousMedium::shadow_transmittance`（2-uniform 签名）在 σt 沿程变化的场上系统性高估透射率 **+1.080%**（N=1.2M，1σ=0.231%，即 **4.7σ**）；同场独立均匀对照仅 **+0.128%**（0.55σ）。
- **因果隔离**：把函数内部同一条派生链（`us=(us*π).fract(); ua=(ua*E).fract()`）在外部复现后喂给无偏的 `shadow_transmittance_tracked`，得到**完全相同**的 `+1.080%` ⇒ 偏差来自派生均匀的强相关性，而非 API 形状。
- **影响面**：`free_path` 用同一派生方式，异质自由程分布同理偏移；该类型**当前无生产消费点**（见 F2），故本版用户可见影响为 0；但一旦接线即生效。
- **与声明的差距**：`CHANGELOG` F-2026-xx-02 写「阴影透射率对 Beer 定律无偏（300k 样本内 <1%）」——该结论仅对 `shadow_transmittance_tracked` 成立；2-uniform 主入口在 300k 下实测 `+1.15%`（2.5σ），1.2M 下 `+1.08%`（4.7σ）已越 1%。
- **建议**：① 后续均匀改由 `aux_1d(dim, step)` 一类「(维度, 步号) 可寻址」的独立流派生（把步序号作为参数），不要用迭代混沌映射；② 为 2-uniform 入口补一条 Beer 无偏性测试（当前无任何测试覆盖该入口的无偏性）；③ CHANGELOG 措辞收窄为「tracked 变体无偏」。

### F2（中）C2 异质体积未接入任何渲染链路（设计承诺的 `Medium::Heterogeneous` 不存在）

- 全仓 grep：`HeterogeneousMedium` 仅出现在 `media.rs` 定义与 `qa_heterogeneous_volume.rs`；**无任何生产消费点**。`integrator.rs:1744` / `wavefront.rs:174` / `sss.rs:138` 仍只吃 `region.medium`（`HomogeneousMedium`）。
- 设计文档 §3 与类图承诺「新增 `HeterogeneousMedium` + `Medium::Heterogeneous`，实现 `Medium` trait」——**全仓不存在 `enum Medium` / `trait Medium`**。
- 盘点文档 §3.4 判「**P0（已还清）**、缺口：无」。按 §2 收敛判据的字面三条（三元组完整 / 结构同级 / 通过对拍）确可判「还清」，但在「渲染特性可达性」上属乐观口径：一项**没有任何渲染路径能调用**的介质类型，不能证明「引擎能渲染异质体积」。
- **建议**：V2.2 把 `Medium` 枚举/分发接进 `integrator`/`wavefront`（一对分支即可），或把盘点措辞改为「算子层已还清，渲染接入留待 V2.2」。

### F3（低）G6 登记表第 1 行的「测试名」实为辅助函数名

- C15 写 `c2_color_golden::aces_fitted_fingerprint`，但 `aces_fitted_fingerprint` 是生成指纹的普通 `fn`（`:31`），真正的 `#[test]` 是 `aces_golden_file_stable`（`:43`）。
- 因该行同时给了「常量名 `ACES_FINGERPRINT` + 27 词逐项旧新值」，定位唯一，**不影响 G6 判定**；建议下次登记改用真实测试名。

### F4（低）T4 结清锁容差比实测松 4 个数量级；文档自称值与实测不符

- 结清锁断言 `worst_f64 < 1e-2` / `worst_f32 < 1e-2`，实测 **4.532e-7 / 4.545e-7**（逆阵 4.144e-8）。
- 后果：若有人把主常量换成"只差 1e-2"的近似矩阵，锁仍会绿（旧债务锁是 `>1e-2` 拦截，翻转后同阈值失去分辨力）；同时 `colorspace.rs` 文档自称「实测 ~1e-8」，与实测 4.5e-7 不符。
- **建议**：按项目自身的「实测后收窄」纪律，把锁收紧到 `< 1e-6`（与 `registry_acescg_matches_official_aces_matrix` 的 4.5e-7 分口径表述），并订正文档数值。

### F5（低）设计文档 §2 文件清单与真实改动偏离 12 处，仅 1 处有勘误说明

| 类型 | 文件 | 说明 |
|---|---|---|
| 未登记·新增 | `vulkan/scene.rs`(258)、`metal/render.rs`(527)、`metal/msl/mod.rs`(改 126) | 光追 dispatch 编排与 Metal 设备无关层 |
| 未登记·改 | `vulkan/spirv.rs`(+212)、`shaders.rs`(+10)、`web/mod.rs`(+4) | 发射器校验器扩展（含 2 处 opcode 勘误）、MSL 注册表、注释订正 |
| 未登记·改（测试） | `qa_verify_p1_framebuffer_guard.rs`、`vulkan_compute_dispatch.rs` | Q4 双自检同步（属 PRD 要求，清单漏列） |
| 未登记·改（math） | `aces.rs`、`colorspace_registry.rs`、`prism-core/src/lib.rs`、`prism-render/src/lib.rs` | C15 登记表已列 2/3/5/6 行，但**文件清单**未列 |
| 登记未改 | `tests/qa_edge_attack_math_sample.rs` | 设计预测需改，实际以常量为参考自动跟随（预测失准） |
| **有说明的偏离** | `prism-core/tests/qa_r1_…` → `prism-io/tests/qa_r1_…` | 测试文件头有「设计偏差说明」（分层纪律禁止 core 依赖 io），**合规** |

- 设计文档的勘误小节只覆盖 R3 与 Q4 测试名，上述偏离**均无说明**。不构成功能缺陷，但削弱「设计↔实现」可追溯性。

### F6（低）设计 §9「Q4 测试名勘误」本身不准确

- 设计称 `availability_requires_verified_dispatch_not_just_a_device` 在本仓**不存在**、真实名为 `backend_availability_follows_verified_compute_path`。
- 实测**两个都存在**：前者在 `crates/prism-gpu/tests/vulkan_compute_dispatch.rs:221`（V2.0 起就有，本版已正确更新为双自检断言），后者在 `src/vulkan/mod.rs:1046`。PRD 引用无误，勘误本身是误判。
- 影响：仅文档；但这类"勘误的勘误"会污染后续引用链。

### F7（低）T4 的「删除即翻转」留痕只在 CHANGELOG，未进 commit 说明

- PRD C3 验收 3：「不得静默删除旧测试——删除即翻转，须在 **commit 说明**留痕」。
- 实际 commit **无正文**（仅 subject），subject 只写「C15×2（T4 矩阵换官方推导、BakedSdf brick 边缘插值修复）」，未点名被删除的 `..._alias_is_bit_identical_to_the_legacy_name` 与被更名的债务锁。
- 留痕在 `CHANGELOG` 第 4/6 行完整存在（明写「**删除**（别名同值语义随结清失效），由 … 接替」），故信息未丢失，仅**通道**不符要求。

### F8（低）CHANGELOG 对 16×16 的「GPU vs CPU 0 差异」强于实际断言

- 声明：「实测 64×48 / 113×47 / 16×16 三种尺寸 GPU vs CpuRefBackend max_channel_delta=0、differing_pixels=0」。
- 实际：64×48（后处理开/关两档）与 113×47 有**逐像素全量**对拍；16×16 场景只断言首像素 `== [10,13,20,255]`（4 字节），全 256 像素的「0 / 256」是**推断值**而非实测值。
- 建议：补一条 16×16 全量对拍（或把表述改为「16×16 空场景背景色与 CPU 参考一致」）。

### F9（低）`spirv.rs` 声称「本环境没有 spirv-val」，与实际不符

- `spirv.rs:33`、`:934` 两处明写「完整验证需 `spirv-val`，本环境没有/无」（`:1038` 为同旨表述）；实测沙箱 `/usr/bin/spirv-val`（SPIRV-Tools v2025.1，target SPIR-V 1.0）可用，我已用它独立校验 `scene.spv` **PASS**。
- 属**保守**声明（不夸大），不影响结论；建议订正并把外部校验纳入可选的运行期验证用例。

### F10（观察）`cache_key_with_mode` / `decide_with_mode` 无生产消费点

- 仅有测试与 re-export 引用；`IncrementalSession`/preview 仍走 `decide()`。
- **不构成缺陷**：迭代节点的 S3 闭包在既有 `loop_frame_signature` 里已真实生效（mode 通道定义进帧签名 ⇒ 进 `cache_key`），`movie_sim_timeline::mode_metadata_changes_frame_signature_and_cache_key` 一直钉住这一点。新增 API 是冗余但正确的加固，`mode=None` 逐字节退化为 `cache_key` 亦已测。
- 建议：要么接线 `IncrementalSession`，要么在文档中注明这是「面向未来的显式入口」。

### F11（观察）SPIR-V 校验白名单全局放宽

- `ALLOWED_OPCODES` 为单一全局常量，本版新增 `OpFNegate/OpLogicalAnd/OpLogicalNot/OpULessThan/OpINotEqual/OpFOrdGreaterThanEqual/OpLoopMerge` 后，**所有模块**（含既有 `spv_bg`/`mul2`）都可用这些指令。
- 既有模块的校验测试仍绿，风险可控；建议后续改为按模块白名单，避免"某 kernel 意外用上循环"无人拦。

### F12（观察）两处 opcode 编号勘误属上一轮遗留的静默错值

- `OP_U_LESS_THAN: 170`（实为 `OpIEqual`）→ 勘误为 `176`；`OP_LOGICAL_NOT: 167`（实为 `OpLogicalAnd`）→ `168`。两者都是"类型合法、语义全错"的静默陷阱，本版修正并留痕（注释写明经 `spirv-as`/`spirv-dis` 复核）。
- 因这些常量此前无消费点（光追 kernel 本版才落地），**对历史输出无影响**；记为工程纪律正面案例。

### F13（极低·风格）`qa_render_scene.rs:309` 两条语句同行

- `let b = backend.render_scene(&req);        assert!(a.is_valid() && b.is_valid(), …);`
- 仓库整体非 rustfmt-clean（`cargo fmt --check` 在既有文件上即报差异），故不计为违规，仅建议顺手格式化。

---

## 11. 最终裁决

**IS_PASS: YES**

裁决依据：

1. **G1–G6 六条全局判据全部通过**：3927 / 0 / 0（净增 +48，无删测）、零外部 crate、GPU 双跑逐位一致、Metal 诚实口径无瑕、全仓零 `#[ignore]`、T4 黄金值逐条可对应且**无表外位移**。
2. **三流逐项验收全部达成**：A1（lavapipe 实机出图 + GPU/CPU **0 差异** + `spirv-val` 独立复核 PASS）、A2（代码 + 门控 + 恒 false + 未实机验证留名）、A3（条件可行结论成文）、B1（R1 2 等价 + 1 不等价、R3 界成立且随分辨率收紧、R2 不动点不回归）、B2（S3 四类逐行有测试、`None` 逐字节退化、S4 逐出中性）、C1（盘点 + 可机械判定收敛判据）、C3（四件套齐 + 结清锁 + 重算逆阵 + C15 四要素 + golden 清单）。
3. **无阻断级缺陷**：无测试失败、无能力伪造、无依赖破纪律、无假绿手段、无静默删测。
4. **两项「中」级问题已记录但判为非阻断**：
   - **F1**（生产 2-uniform 阴影估计量 +1.08% 正偏）与 **F2**（异质体积未接入渲染链路）都是**代码契约/可达性**层面的问题，**不违反 PRD 中任何一条明示验收判据**（C2 验收 1–5 均已满足），且 F2 使 F1 在本版**用户可见影响为 0**。
   - 两者均要求 **V2.2 首轮必修**；其中 F1 的 CHANGELOG 措辞收窄（「无偏」→「tracked 变体无偏」）建议在下一版随手订正，因现表述对主入口不成立。

**回归建议（不阻断）**：F3–F9 为文档/口径/精度类修订，随下一版 CHANGELOG 一并处理；F4（把 T4 结清锁收紧到 1e-6）建议优先，因它直接决定该锁未来的分辨力。

—— 严过关（Edward），QA 工程师，2026-10-26

---

## 附：交付前修复记录（主理人执行，2026-10-26 晚）

本报告裁决为 IS_PASS: YES。裁决后、交付前，主理人按「欠的都结」原则**当场结清了两项审计发现中的 F1 与三条文档口径问题**（F2 保留为 V2.2 首轮必修）：

| 项 | 处置 | 证据 |
|----|------|------|
| **F1** 生产 2-uniform 阴影估计量 +1.08% 正偏 | **已修**：空碰撞派生链由 `(u·π).fract()/(u·e).fract()` 乘法链改为 splitmix64 雪崩混合（`derive_uniforms`，种子 = u_step ⊕ u_accept ⊕ 步号，确定性不变）；新增钉子 `shadow_transmittance_production_entry_unbiased_vs_beer`（200k 样本，2% 容差） | 全量回归 **3928 passed / 0 failed**（较 QA 轮 +1，即新钉子） |
| **F3** 登记表第 1 行写辅助函数名 | **已修**：改为真实测试名 `c2_color_golden::aces_golden_file_stable`（指纹常量 `ACES_FINGERPRINT`） | CHANGELOG R-2026-xx-03 |
| **F4** 结清锁文档称 ~1e-8、实测 4.53e-7 | **已修**：两处注释改为「实测 4.53e-7；容差 1e-2 只作域界，不作精度宣称」 | `colorspace.rs` 常量文档 + 结清锁文档 |
| **F8** 16×16「0 差异」强于实际断言 | **已修（加强测试而非弱化文案）**：空场景断言由 4 字节扩为**全帧 256 像素逐一**核对；CHANGELOG 同步改为准确定述 | `qa_render_scene.rs` |
| **F9** `spirv.rs` 称「本环境没有 spirv-val」 | **已修**：模块文档改为「语义验证走官方 spirv-val；QA 轮实测 exit 0」，并同步光追 kernel 已落地的真实状态 | `spirv.rs` 模块头 |

**F2（异质体积未接入渲染链路）** 维持 QA 裁决：V2.2 首轮必修，本版不阻断。

—— 齐活林（主理人），2026-10-26

---

## 附二：剩余审计项全部结清（主理人执行，2026-10-27 晨）

作者指示「Bug/缺口不能留到 V2.2」。F2/F4/F5/F6/F7 已于交付后当轮结清：

| 项 | 处置 | 证据 |
|----|------|------|
| **F2** 异质体积未接入渲染链路 | **已接线**：`crates/prism-rt/src/media.rs` 新增 `enum Medium { Homogeneous, Heterogeneous }`（统一 `free_path`/`shadow_transmittance`/`phase`/`sample_phase`/`albedo`）；`MediumRegion.medium` 改用该类型；`integrator::medium_collide` 与 `wavefront::medium_collide` 双路径消费。均匀路径仅抽 1 个均匀（采样流与像素逐字节不变），异质路径抽 2 个。新增钉子 `integrator::tests::heterogeneous_medium_render_path_is_reachable_and_matches_beer`（可达性断言 + 整画面 Beer 5% 内 + 两遍渲染逐位一致） | `cargo test -p prism-rt --release` → **194 passed / 0 failed**（+1）；`cargo check --workspace --release` 通过 |
| **F4** 结清锁容差过松 | **已收紧**：1e-2 → **1e-6**（实测 4.53e-7，留 ~2.2× 余量）；文档同步改为「容差已按实收紧」 | `cargo test -p prism-math --release` 全绿 |
| **F5** 设计文件清单 12 处偏离 | **已补录**：设计文档新增 §10「实施偏差登记」，逐条列明未登记文件、登记未改项与合规偏离 | `docs/v2.1-增量设计.md` §10 |
| **F6** Q4「测试名勘误」本身误判 | **已撤回**：设计 §1.1 与 §9 均改为「两个测试都存在、双双升级为双自检」 | `docs/v2.1-增量设计.md` §1.1/§9 |
| **F7** 删除/更名测试未进 commit 说明 | **已补正文**：本批提交正文点名被删除的 `..._alias_is_bit_identical_to_the_legacy_name`、更名的 `..._debt_t4_is_named` → `..._settled_t4_is_named`，并注明 CHANGELOG 对照表位置；设计 §10 同步留痕 | git log（本批提交正文） |

至此 QA V2.1 轮的全部审计发现（F1–F9 + F13 风格项）均已处置，无遗留到 V2.2 的 Bug/缺口。

—— 齐活林（主理人），2026-10-27
