# EC-3 SDK 稳定化（V2.0-M3）

> 数据同源：本文口径与 `crates/prism-infra/src/sdk.rs`（版本协商）、
> `crates/prism-infra/src/plugin.rs`（市场 add 闸门）、
> `examples/sdk-nodes/cinematic-basic/`（示例包）、
> `crates/prism-infra/tests/sdk_example_node.rs`（端到端判据）逐条对应。
> 复现命令：`cargo test -p prism-infra`。

## 1. 交付面（三件）

| 件 | 落点 | 内容 |
|---|---|---|
| 接口语义化版本 | `prism-infra/src/sdk.rs`（约 430 行，7 测试） | `SdkVersion` / `VersionConstraint` / `Compatibility` / 协商入口 `check_manifest` |
| 版本闸门接入 | `prism-infra/src/plugin.rs` | `PackLibrary::add` 在静态校验 + 资产校验之后跑协商，不兼容 ⇒ 明确拒绝 |
| 示例节点端到端 | `examples/sdk-nodes/cinematic-basic/` + `tests/sdk_example_node.rs`（2 测试） | manifest + presets + subgraphs + assets；可编译 / 可加载 / 版本协商三判据 |

## 2. 版本口径（协商什么、按什么协商）

- **`ENGINE_VERSION`**（`sdk.rs` 常量，当前 `0.1.0`）：引擎**接口**版本，
  pack 契约的协商基准，与 workspace 各 crate 同号，并与 DSL 侧
  `prism_dsl::PackManifest::engine_version()`（取其 crate 版本）对齐——
  同一份 manifest 在市场侧与 DSL 编译侧的协商结论必须一致，否则会出现
  "能 add 的包编译不过"或反之。
- **兼容带**（semver，0.x 特例）：主版本不同 ⇒ 不兼容；主版本为 0 时
  次版本不同 ⇒ 不兼容（0.x 的次版本就是破坏轴）；否则兼容。
  判据：`SdkVersion::is_compatible_with`，测试 `compat_band_follows_semver_zero_x_rule`。
- **约束文法**：`engine` 字段 = 逗号分隔的比较子合取；每条 = 算子
  （`>=` / `>` / `<=` / `<` / `=` / `==`，两字符优先）+ 版本（`x` / `x.y` /
  `x.y.z`，缺段补 0）。**裸版本（无算子）不合法**——与 DSL 侧
  `parse_version_reqs` 同口径。例：`engine = ">=0.1, <0.2"`。

## 3. 不兼容如何报错（明确，不静默）

协商结论三分（`Compatibility`）：

| 结论 | 触发 | add 行为 |
|---|---|---|
| `Unconstrained` | manifest 未声明 `engine` | 放行（隐式"任意版本"） |
| `Compatible` | 当前版本满足全部比较子 | 放行 |
| `Incompatible(conflict)` | 任一比较子不满足 | **拒绝**：`PackError` 三段式（P4001 + 建议段） |
| `Err(SdkError)` | 约束格式非法 | **拒绝**：P4002（坏清单不许进库） |

错误文本形态（C18 三段式 + C19 建议段）：

```text
error[P4001] 插件市场 add 闸门：包 'future-pack'
  什么事：engine 约束 '>=99.0'（规范化 >=99.0.0）不被当前引擎接口版本 0.1.0
           满足——首个失败比较子：>=99.0.0（约束窗口与当前版本不在同一兼容带
           ——面向更高/更低接口版本的包）
  建议：三选一：① 把引擎升级到满足 '>=99.0' 的版本；② 放宽该包 manifest 的
         engine 约束以包含 0.1.0；③ 确认该包面向其他接口版本、暂不入库——
         数据包是纯数据，接口不兼容时其预设/子图可能引用不存在的字段，故显式
         拒绝而非静默放行
```

错误码与 DSL 侧装包校验同码同义（P4001 版本范围 / P4002 清单格式），
码域是宪章级共享的；两侧文法、算子集、0.x 兼容轴口径一致，
测试 `check_manifest_reads_optional_engine_field` / `add_rejects_*` 钉住。

## 4. 示例节点（端到端）

```text
examples/sdk-nodes/cinematic-basic/
  manifest.toml          # name/version/license/engine/provides/assets 全字段
  presets/grade-film.prism    # 参数预设：teal-orange + filmic 调色档
  subgraphs/hero-shot.prism   # 预制子图：主镜头场景模板
  assets/palette.cube         # 2x2x2 恒等 .cube LUT（清单内真实资产）
```

端到端判据（`tests/sdk_example_node.rs`，两步一正一反）：

1. **契约完整**：manifest 解析 + `validate_static` + `validate_assets`；
2. **版本协商**：`check_manifest` ⇒ `Compatible`（约束 `>=0.1, <0.2` 含
   `ENGINE_VERSION` 0.1.0）；
3. **可编译**：`presets/` 与 `subgraphs/` 每个 `.prism` 过
   `prism_dsl::compile`（v1 语法，全字段 DSL 文档）；
4. **可加载**：`PackLibrary::add` 成功 → `list` 恰一条、版本一致 →
   `remove` 往返干净；
5. **反向闸门**：内存中把约束改成 `>=99.0, <99.1`（同目录同文件，只改约束）
   ⇒ `Incompatible`，证明拒绝源于版本协商而非别的校验。

## 5. 边界声明（本波次刻意不做，如实标注）

| 项 | 状态 |
|---|---|
| 组件实例化（子图内联进宿主场景） | Phase 2/3：DSL 侧 `packs.rs` 已留 Phase 2/3 接口；本波次示例包的 subgraph 按"可整文档编译的场景模板"交付 |
| LUT 消费（palette.cube 进调色链） | 资产清单校验只验存在性；消费侧不在本波次 |
| 多包依赖解析（包依赖包） | 未做；manifest 无依赖字段 |
| 包签名 / 校验和分发 | 未做；市场是本地的（add/list/remove） |

## 6. 复现

```bash
cargo test -p prism-infra                      # 全量（lib 69 + 6 个测试床）
cargo test -p prism-infra sdk_                 # 仅示例端到端
cargo test -p prism-infra --lib sdk::          # 仅版本协商单测
```

环境受限项：无（纯逻辑 + 本地文件系统，全部本机实测）。示例包体积极小
（4 文件、480x270 起），渲染判据只到"可编译"（编译是 DSL 契约的完整
校验），端到端成像由既有渲染测试床覆盖，不重复。
