# Prism Engine

> 零依赖、确定性的 Rust 媒体引擎 —— 以**单文件 TXT 分发**：下载一个 14 MB 文本，运行解包脚本即得完整 26-crate 源码树，无需 npm / pip / ffmpeg / OpenEXR。

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## 这是什么

Prism Engine 是一个用 Rust 编写的零第三方代码依赖媒体引擎，覆盖图像、音频、几何、物理、渲染、DSL 等模块。全部 103 条依赖均为本仓 `path` 依赖——不依赖网络、不需要包管理器、没有供应链风险。

采用**单文件源码分发**：整个引擎（481 个文件、26 个 crate）合并进一个 UTF-8 文本 `prism-engine-v1.0-source.txt`。持有该文件即可在本地还原出与仓库完全一致的目录树。

## 为什么是单文件分发

- **可审计**：一个文本文件，人眼可读、可 `diff`、可哈希校验，发布即冻结。
- **可复现**：打包具确定性——相同输入两次打包 SHA-256 完全一致。
- **可携带**：无需 clone 庞大仓库，一个文件经邮件或聊天即可传送，离线亦可还原构建。
- **取舍说明**：单文件分发以让渡"在 GitHub 上直接浏览代码"的便利为代价，换取"一个文件带走全部、且逐字节可验证"的确定性。解包后得到的是普通 Cargo workspace，与常规 Rust 项目无异。

---

## 快速开始

前置：安装 [Rust](https://www.rust-lang.org/)（含 `cargo`）与 [Node.js](https://nodejs.org/) 18+。

```bash
# 1. 下载仓库中的 prism-engine-v1.0-source.txt，存为本地文件
# 2. 用文本编辑器打开，把 "UNPACK SCRIPT BEGIN" 与 "UNPACK SCRIPT END"
#    之间的 Node.js 脚本另存为同目录下的 unpack.cjs
# 3. 运行解包（约数秒，还原 481 个文件）
node unpack.cjs
# 4. 进入还原出的源码树，跑全量测试
cd src/..
cargo test --workspace --release
# 预期：3271 passed / 0 failed
```

解包后得到一个标准 Cargo workspace（`src/` 下 26 个 crate），后续 `cargo` 命令与常规 Rust 项目相同。

> 手机端 wasm 验证链路：`rustup target add wasm32-unknown-unknown`，再 `cargo build -p prism-core --target wasm32-unknown-unknown`。

---

## 仓库结构（解包后）

| 路径 | 说明 |
|------|------|
| `src/` | 26 个 crate 的源码（仓库内原名 `crates/`，分发时重组为 `src/`） |
| `Compliance/` | 合规声明：零依赖核验 / 第三方声明 / 资产登记（仓库内原名 `docs/合规/`） |
| `开发者指南.md` | 自包含开发者文档：上手 / 架构 / 数学内核 / 门禁 / 扩展（仓库内原名 `docs/开发者指南.md`） |
| `tools/` | 门禁脚本（幽灵写入防御、卫生扫描、合规自检） |
| `.github/workflows/` | CI 配置 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | 构建必需件（workspace members 已指向 `src/*`） |
| `examples/` | `prism-dsl` 测试用样例（`include_str!` 读取，缺了编译不过） |

> 设计权威文档（`docs/设计/`）**不随分发**，仅留在发布方的完整仓库中。

---

## 模块概览

26 个 crate 按职责分组：

| 类别 | crate |
|------|-------|
| 内核 | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| 媒体 I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| 渲染 | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| 动画/生成 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| 文本/UI | prism-text · prism-rt · prism-app |
| 高级 | prism-ai · prism-dsl · prism-infra |

---

## 构建与测试

```bash
# 全仓冒烟（release，推荐）
cargo test --workspace --release
# 等价别名
cargo xtest
# 单 crate 快速验证
cargo test -p prism-core --release
```

- 全量测试：**3271 passed / 0 failed**（已在全新解包树上实跑验证）。
- 门禁：14 道，含幽灵写入防御（G1~G4）、开源卫生扫描（0 内部痕迹）、合规自检（9/9）、合规反向自证（8/8，违规必红）、wasm32 构建链路。

本地运行门禁（需 Node 18+）：

```bash
node tools/ghost_guard.mjs --check        # 幽灵写入防御：G1 重复定义/G2 严格超集/G3 顶格文档/G4 编译
node tools/hygiene_scan.mjs               # 开源卫生：应报“干净（0 处内部痕迹）”
node tools/compliance_check.mjs           # 合规自检：应 9/9 通过
node tools/compliance_check.mjs --self-test   # 反向自证：应判红，证明判据具备判别力
```

---

## 合规

引擎承诺**零第三方代码依赖**，并经自检核验：

- **C1 零第三方代码依赖**：全部 103 条依赖均为本仓 `path` 依赖；无 `build.rs`、无原生链接指令、`Cargo.lock` 锁定 26 个本仓包。
- **C2 资产登记完备**：5 项资产主表字段齐全、自研化状态落在受控词表；OFL 1.1 字体许可证全文随字体置于 `src/prism-text/font_data/LICENSE`。
- **C3 第三方归属声明**：`Compliance/THIRD-PARTY-NOTICES.md` 在场，并明确区分"数据资产 ≠ 代码依赖"。

> 注：本仓库分发的是**源码**。字体资产（Prism Sans SC 子集）以 SIL OFL 1.1 许可，**许可证全文随字体置于** `src/prism-text/font_data/LICENSE`；资产来源与许可登记见 `Compliance/assets/资产登记.md`。

---

## 确定性打包（发布方）

如需从完整仓库重新生成分发 TXT：

```bash
node tools/pack_source_txt.mjs
# 产物：dist/prism-engine-v1.0-source.txt
# 连跑两次，输出 SHA-256 一致即证明确定性打包成立
```

打包范围严格限定为：`crates/`（→ `src/`）、`docs/合规/`（→ `Compliance/`）、`docs/开发者指南.md`、`tools/`、`.github/`、`.cargo/`、`examples/`、`Cargo.toml`、`Cargo.lock`。刻意**排除** `docs/设计/`、`target/`、`node_modules/`、`dist/` 本身及一切中间产物。

---

## 贡献

欢迎 Issue 与 Pull Request。提交前请保持：

- `cargo test --workspace --release` 全绿；
- `node tools/hygiene_scan.mjs` 报 0 处内部痕迹；
- `node tools/compliance_check.mjs` 9/9 通过。

更详细的架构与扩展指南见解包后的 `开发者指南.md`。

---

## 许可证

本项目以 **Apache License 2.0** 发布。详见 [`LICENSE`](LICENSE) 文件。

> 合规提示：Apache-2.0 要求保留版权与许可声明；基于本引擎再行分发时，请保留 `Compliance/` 下的归属声明与资产登记。
