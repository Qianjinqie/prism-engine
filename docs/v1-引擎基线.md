# v1 引擎基线（T01 存档）

> 微积分短片 v2 · 任务 T01 产出。执行时间：2026-10-13；执行人：主理人（齐活林）。
> 本文件是 T02 / T03 / T04 / T05 完成后同命令回归的**对照基线**（架构 §6.1）。
> 原始完整输出存于同目录 `*.log`（未截断、未修饰）。

## 1. 三条验收命令结果

| # | 命令 | 结果 | 数据 |
|---|---|---|---|
| 1 | `cargo test --workspace` | ✅ 退出码 0 | **3342 passed / 0 failed / 0 ignored**，156 个 test result: ok，0 个 FAILED |
| 2 | `cargo build --workspace --all-targets` | ✅ 退出码 0 | 修复后 **warning = 0**（修复前 21 行 / 11 处，见 §3） |
| 3 | `cargo test -p prism-core --test qa_ignore_guard` | ✅ 退出码 0 | 4 passed / 0 failed / 0 ignored（无未登记 `#[ignore]`） |

- 命令 1 完整输出：`cargo-test-workspace.log`（4405 行，236KB）。
- 命令 2 修复前/后输出：`cargo-build-all-targets-before-fix.log` / `-after-fix.log`。
- 命令 3 输出：`cargo-test-qa-ignore-guard.log`。
- 修复后受影响 crate 复跑：`cargo-test-affected-crates-after-fix.log`（6 crate，35 套件全绿、0 警告）。

## 2. 环境事实

- 仓库根：`/workspace/media-engine`（git master，工作区有 v1 遗留未提交改动）。
- 工具链：rustc/cargo 版本见 `target/.rustc_info.json`；CPU 32 核；debug 缓存 7.8G（热）。
- 基线执行顺序：命令 1 → 发现警告 → 修复（§3）→ 命令 2 复验 0 警告 → 命令 3 → 受影响 6 crate 复跑（全绿）。
- 修改面仅触及 6 个 crate 的测试代码 + 1 处死函数删除 + 1 处测试专用导入移位；其余 crate 未动，命令 1 结论对其依然成立（命令 2 的 `--all-targets` 已全量编译含全部测试二进制，0 error 0 warning）。

## 3. 基线警告修复记录（计入 T01 工作量）

修复前 `cargo build --workspace --all-targets` 有 **11 处警告**（21 行含汇总行）。全部为机械性清理，**零断言弱化**，其中 2 处为增强：

| 文件 | 警告 | 修法 |
|---|---|---|
| `crates/prism-core/src/graph.rs:876` | unused variable `n` | 补 `assert_eq!(vals.len(), n)` 把常量钉进向量（增强） |
| `crates/prism-rt/src/gi/probes.rs:46` | unused import `multi_bounce_energy_upper_bound` | 该导入仅服务 `mod tests`，移入 tests 模块（父模块非测试构建不再需要） |
| `crates/prism-rt/src/hair_scatter.rs:483` | dead fn `cos_abs` | 确认全仓无调用后删除（含文档注释） |
| `crates/prism-particles/tests/particles_integration.rs:295` | unnecessary `mut` | 删 `mut` |
| `crates/prism-aesthetic/tests/qa_probe_secondary.rs:40` | unused variable `ec` | 补 `assert!(ef < ec)`（增强：coarse 误差须大于 fine） |
| `crates/prism-ai/tests/qa_edge_attack.rs:19` | unused import `ProviderResponse` | 删导入项 |
| `crates/prism-aesthetic/src/critic_loop.rs:520-521` | unnecessary parentheses ×2 | 去括号 |
| `crates/prism-aesthetic/tests/qa_edge_attack_secondary.rs:4` | unused import `WindowShape` | 删导入项 |
| `crates/prism-physics/src/hair.rs:2308` | unnecessary parentheses | 去括号 |
| `crates/prism-gpu/tests/qa_verify_p1_framebuffer_guard.rs:238` | unnecessary `mut` | 删 `mut` |

回归口径：上述 6 crate（prism-core / prism-rt / prism-aesthetic / prism-ai / prism-particles / prism-gpu）修复后全量测试复跑，35 套件全绿、0 警告；随后全工作区 `--all-targets` 0 warning。

## 4. 基线结论

**T01 通过。** 引擎处于「全绿 + 零警告 + 无未登记 ignore」状态，T02 / T03 / T04 可并行开工（架构 §6.6 依赖图）。
