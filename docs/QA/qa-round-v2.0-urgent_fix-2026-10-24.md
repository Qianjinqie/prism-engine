# QA 测试报告 — V2.0-urgent_fix 正式 QA 轮次

- **轮次**：qa-round-v2.0-urgent_fix
- **日期**：2026-10-24
- **QA 工程师**：严过关（Edward）
- **测试对象**：`/workspace/media-engine`，HEAD = `a957475`（"打包工具：版本参数放行 hotfix 后缀；V2.0-urgent_fix 包已出"），工作区干净（`git status` 无变更）
- **基线预期**：全量回归 3879 passed / 0 failed（release，约 1 分 47 秒）

---

## 1. 测试范围与命令

| # | 项 | 命令 | 用途 |
|---|----|------|------|
| C1 | 全量回归 | `cargo xtest`（= `cargo test --workspace --release`，仓库 `.cargo/config.toml` 别名） | 28 crate 全量 |
| C2 | prism-io 定向 | `cargo test -p prism-io --release` | W5 / W4-导出器 |
| C3 | prism-gpu 定向 | `cargo test -p prism-gpu --release` | W4-GPU / W3-Vulkan |
| C4 | prism-core 定向 | `cargo test -p prism-core --release` | W3-fixpoint |
| C5 | prism-math 定向 | `cargo test -p prism-math --release` | W1-Aces2065 |
| C6 | 关键用例单跑 | `cargo test -p <crate> --release [--lib\|--test <file>] <filter> -- --nocapture` | 定向钉子真实验证 |
| C7 | 打包回环 | Node 脚本（复制 `tools/pack_source_txt.mjs` 既有 remap/rewrite/isBinary 规则）+ 内嵌解包脚本解包既有产物逐字节比对 | 非破坏性，不重新打包 |

> **过程记录（重要）**：任务书建议的 `cargo xtest -p prism-io` 等缩小范围写法**实际无效**——`xtest` 别名已内含 `--workspace`，命令行追加的 `-p` 与 `--workspace` 是并集选择，实测跑出的仍是全量（该次运行同样得到 3879 passed / 0 failed，可作全量回归的第二次独立复跑）。真正的单 crate 缩小范围须绕开别名：`cargo test -p <crate> --release`。本报告定向回归均按后者执行。

---

## 2. 全量回归（C1）

```
$ cd /workspace/media-engine && cargo xtest
```

- 首次（含 release 编译）：**1m48s**；复跑（编译缓存命中，第二次独立全量）：**1m24s**
- 汇总（`grep "^test result"` 聚合 174 个测试二进制/文档测试段）：

| 指标 | 数值 |
|------|------|
| passed | **3879** |
| failed | **0** |
| ignored | 0 |
| 编译错误 | 无 |
| `failures:` 段 | 无 |

**结论：3879 passed / 0 failed / 0 ignored，与已知基线完全一致。**

---

## 3. 定向回归逐项结论

### 3.1 W5 — USD UTF-8 字节保真（crate：prism-io，整 crate 581/581 通过）

源码核查：`crates/prism-io/src/usd.rs` `read_string_body`（L948-1007）已为 `Vec<u8>` 逐字节收集、闭合处 `decode_bytes` 整体 UTF-8 解码（`String::from_utf8`，非法序列报语法错误、拒绝 lossy）；`@asset@/<path>` 同类路径同规则；导出侧 `non_ascii_text` 有损类已移除。

关键用例（实跑输出）：

| 用例 | 结果 |
|------|------|
| `usd::tests::utf8_string_literals_are_char_exact`（中文+emoji+组合重音，逐字符+逐字节双断言，六种引号/转义形态） | ok |
| `usd::tests::invalid_utf8_string_body_is_a_structured_error`（公共入口 PU1009 带"第 4 行第 19 列"；词法层兜底 PU1002 带行列；不 lossy） | ok |
| `usd_export::tests::unicode_roundtrip_is_byte_fixed_point`（导入→导出→回读→再导出三轮字节不动点；`@textures/…png@`、`</场景/材质>` 路径保真） | ok |
| `usd_export::tests::non_ascii_text_is_written_verbatim_without_report`（非 ASCII 不再进有损记账） | ok |

**结论：通过。**

### 3.2 W4 — usda 导出器（R2 字节不动点 + ExportReport 八类有损记账；`usd_export` 27/27 通过）

`ExportReport` 八类有损字段与固化测试一一对应：`dropped_time_samples` / `dropped_variant_sets`（`time_samples_and_variants_are_reported_not_written`）、`dropped_layer_metadata`（`sublayers_are_dropped_and_reported`）、`dropped_sheen`（`sheen_is_reported_and_never_silently_dropped`）、`relabeled_interpolation`（`non_vertex_attribute_lengths_are_reported`）、`baked_transforms`（`baked_world_transform_is_reported`）、`renamed_prims`（`name_collisions_are_resolved_and_reported`）、`unused_materials`（`unused_materials_are_reported_and_not_written`）——全部 ok。

R2 字节不动点钉子（全部 ok）：
- `export_import_export_is_byte_fixed_point`（导出→导入→再导出字节级一致）
- `f64_text_roundtrip_is_bitwise` / `export_is_byte_level_deterministic`
- `imported_layer_re_export_preserves_geometry_and_unknown_attrs`（导入资产直导亦字节不动点）

**结论：通过。**

### 3.3 W4 — GPU 首帧（compute 离屏三尺寸 + 三真缺陷固化；crate：prism-gpu，260/260 通过）

三尺寸全像素逐字节 0 差异（真机 llvmpipe 实跑，`--nocapture` 输出）：

```
[device] llvmpipe (LLVM 20.1.2, 256 bits)（枚举 1 个物理设备）
[frame] device=llvmpipe size=64x48  t=1.25  groups=[8,6,1]  bytes=12288 differing_bytes=0 differing_pixels=0 max_channel_delta=0
[frame] device=llvmpipe size=100x70 t=1.25  groups=[13,9,1] bytes=28000 differing_bytes=0 differing_pixels=0 max_channel_delta=0
[frame] device=llvmpipe size=32x24  t=-3    groups=[4,3,1]  bytes=3072  differing_bytes=0 differing_pixels=0 max_channel_delta=0
test result: ok. 1 passed; 0 failed
```

（64×48 整除快路径 / 100×70 带尾组守卫 / 32×24 负时间 `t=-3.0` 饱和口径，三形态覆盖。）

三真缺陷固化测试（逐一实跑，全部 ok）：

| 缺陷 | 固化用例 | 结果 |
|------|----------|------|
| OpAccessChain 越界（lavapipe 真机 SIGSEGV） | `vulkan::spirv::tests::validator_catches_access_chain_through_scalar`（缺陷原样注回模块，静态校验拦下 AccessChainNotComposite / AccessChainMemberOutOfRange） | ok |
| u32 溢出（尺寸守卫） | `vulkan::background::tests::kernel_size_guards_refuse_without_panicking`（`u32::MAX×u32::MAX`、65536²、零轴、超限维一律 `None` 不 panic） | ok |
| usize 溢出（极端尺寸入口） | `vulkan_background_gpu::background_gpu_refuses_extreme_sizes_without_panicking`（catch_unwind 包裹，BufferFailed 且不回像素） | ok |

**结论：通过。**

### 3.4 W3 — Vulkan compute 闭环（真机 dispatch，256 元素 ULP=0）

```
$ cargo test -p prism-gpu --release --test vulkan_compute_dispatch real_dispatch_matches_cpu_reference_bitwise -- --nocapture
[device] llvmpipe (LLVM 20.1.2, 256 bits)（枚举到 1 个物理设备，选偏好最高者）
[spirv] 165 字 / 42 条指令 / entry=mul2 localsize=[64, 1, 1] bindings=[0, 1]
[readback] device=llvmpipe (LLVM 20.1.2, 256 bits) elements=256 bitwise_mismatches=0 max_ulp=0 max_abs_diff=0e0
test real_dispatch_matches_cpu_reference_bitwise ... ok
```

- 256 元素（含 ±0、次正规、极大值×2 溢出为 inf 等边界）vs CPU 参考**逐位相等，ULP=0**； lavapipe ICD（`lvp_icd.json`）在位，为真实 dispatch 非 skip。
- `availability_requires_verified_dispatch_not_just_a_device` ok：`is_available` 严格等同完整 dispatch 自检；无设备时诚实返回 false 的分支在测试内以条件断言覆盖（本机有设备，走真跑分支并打印 `[available] … compute=true render=false`）。

**结论：通过。**

### 3.5 W3 — 环推进驱动 fixpoint.rs（crate：prism-core，165/165 通过）

`fixpoint` 模块 10/10 实跑通过，其中：

| 用例 | 结果 |
|------|------|
| `fixpoint::tests::seek_three_paths_are_bitwise_identical`（顺序延续 / 快照装配 / 冷 seek 三路径字节级一致） | ok |
| `fixpoint::tests::cold_seek_is_structured_p3012_and_writes_nothing` | ok |
| `fixpoint::tests::driver_feeds_frame_residuals_to_divergence_detector`（SimTimeline 上提泛化后残差链） | ok |
| `fixpoint::tests::run_fixed_point_runs_exact_iterations_and_records_residuals` | ok |
| `fixpoint::tests::divergence_detector_needs_full_window_and_ignores_plateau` | ok |
| `fixpoint::tests::expansive_phi_diverges_but_m3_still_runs_full_steps` | ok |
| `fixpoint::tests::fixed_point_mode_rejects_zero_or_nonfinite` | ok |
| `fixpoint::tests::mode_metadata_changes_frame_signature` | ok |
| `fixpoint::tests::solve_mode_channel_def_is_stable_0x5333` | ok |
| `fixpoint::tests::codec_rejects_foreign_bytes` | ok |

prism-core 其余测试（eval / graph / snapshot / seek 语义 P3012 等）随整 crate 运行全过。**结论：通过，零语义回归。**

### 3.6 W1 — Aces2065 矩阵（crate：prism-math，160/160 通过）

- `aces` 模块 6/6：`fitted_matches_reference_within_tolerance`（Stephen Hill 官方拟合对拍 < 0.01）、`fitted_is_monotonic_on_gray_ramp`、`fitted_endpoints_and_midgray`、`acescg_conversion_and_matrix_orthogonality`、`aces_output_bitwise_deterministic`、`fast_style_is_deterministic_identity_at_zero`。
- `colorspace` + `colorspace_registry` 67/67：含 `aces2065_to_acescg_is_recomputed_inverse`（S-2014-004 重录矩阵的可逆性）、`registry_acescg_matches_official_aces_matrix`（新轨推导 vs 官方公布矩阵 < 1e-6，实测 ~4.5e-7）。
- **T4 债务测试** `colorspace::tests::legacy_srgb_acescg_matrices_debt_t4_is_named`：**通过（作为债务锁通过，非失败）**。该测试将 Hill 拟合旧常量 vs 官方 AP1 原色推导的正向偏差锁死在实测 3.2045e-2 @ [2][2] 邻域（断言 `> 3e-2 && < 3.5e-2`、位置 [2][2]），f32/f64 双口径均报。按任务书说明：此为**已知留名债务，单向门不修，不算失败**；测试绿 = 债务仍被钉住，未发生静默统一或漂移。

**结论：通过。**

---

## 4. 打包产物抽检（C7，非破坏性回环）

- 产物：`dist/prism-engine-v2.0-urgent_fix-source.txt`
- 实测：**18,691,086 字节**（与任务书预期一致），头部声明 **633 个文件**（一致）
- sha256：`f2038804602d33e175a773b30071653bf2df33ebcb19d5301287a55c84a76902`
- 回环校验（提取产物内嵌解包脚本解包 → 包内路径按 `remap` 逆映射回仓库路径 → 按打包侧相同 `rewriteContent`/`isBinary` 规则逐字节比对）：

```
解包报告：还原 633 个文件
解包树实际文件数：633
逐字节比对：633 个文件
✓ 回环校验通过：633 个文件与仓库源码逐字节一致
```

未重新打包、未触碰 dist/ 原文件。**结论：通过。**

---

## 5. 路由判定

| 检查项 | 失败数 | 路由 |
|--------|--------|------|
| 全量回归 | 0 | — |
| 定向回归 W5 / W4×2 / W3×2 / W1 | 0 | — |
| 打包回环 | 0 | — |

**无任何失败用例，无需路由。** 测试代码自身无 bug 需要修；源码无 bug 需要反馈工程师。

---

## 6. 遗留问题清单

1. **T4 债务（已知、单向门、不修）**：`legacy_srgb_acescg_matrices_debt_t4_is_named` 留名的 Hill 拟合旧常量 vs 官方 AP1 推导偏差 3.2045e-2 @ [2][2]（逆向 5.9690e-2 @ [0][0]）。结清需四件套（按官方原色重录 + C15 登记 + 存量基线重渲 + 黄金值重录），待作者拍板。本次按任务书口径不计失败。
2. **工具用法提醒（非缺陷）**：`cargo xtest -p <crate>` 因别名内含 `--workspace` 不缩小范围（`-p` 为并集）。建议后续文档注明缩小范围用 `cargo test -p <crate> --release`，或为别名增加窄化变体。本报告已按正确方式执行，不影响结论。
3. **无其他遗留问题。**

---

## 7. 最终裁决

- 全量回归：**3879 passed / 0 failed**（与基线一致）
- 定向回归六项：**全部通过**（每项均有真实命令输出支撑，关键钉子均单独实跑并核对设备/数值输出）
- 打包产物：**633 文件 / 18,691,086 字节 / 回环逐字节一致**

# IS_PASS: YES

—— 严过关（Edward），2026-10-24
