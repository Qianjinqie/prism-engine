# Prism Engine

> 零依赖、确定性的 Rust 媒体引擎 —— 以**单文件 TXT 分发**：下载一个 18 MB 文本，运行解包脚本即得完整源码树（28 个 crate），无需 npm / pip / ffmpeg / OpenEXR。

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3929%20passed-brightgreen.svg)
[![Release](https://img.shields.io/badge/release-v2.1-blue.svg)](https://github.com/Qianjinqie/prism-engine/releases)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

（徽章为静态事实徽章，数字口径见下文；`crates-28` = **分发包内**的 crate 数，
仓库里另有 1 个不进包的演示 crate，故仓库数是 29。接入 CI 后可替换为动态徽章。）

---

## 这是什么

Prism Engine 是一个用 Rust 编写的零第三方代码依赖媒体引擎：读图、排版、画画面、
剪时间线、出成片，全部在一个确定性引擎里完成。全部 110 条运行期依赖
（另有 20 条测试依赖）均为本仓 `path` 依赖——不依赖网络、不需要包管理器、
没有供应链风险。

采用**单文件源码分发**：整个引擎（642 个文件、28 个 crate）合并进一个 UTF-8
文本 `prism-engine-v2.1-source.txt`。持有该文件即可在本地还原出与仓库完全一致的
目录树。

---

## 引擎能干什么

一句话总览：**图片是画面在某一时刻的样子，视频是画面在一段时间里一串的样子**——
同一份场景描述，定住某一刻是单帧，连续推进并录下来就是视频，静态图与动态图共用一套能力。
下面按"你要做的事"分五块：**读进来 → 排出来 → 画出来 → 出成片 → 接进你自己的工具链**。

### 读什么（输入）

- **统一解码门面**：一个 `decode_image(bytes, ext)` 入口分发静态图——PNG（非交错 8/16bit）、
  基线 JPEG（定点 IDCT，跨平台逐位确定），**魔数优先、扩展名兜底**（扩展名写错也能解）；
  矩阵外或需外部协作的格式一律三段式报错（在哪 / 什么事 / 怎么改）并给出 ffmpeg 协作建议，
  **永不静默失败**。
- **动图 / 音频 / 容器**：GIF、APNG 原生逐帧解码（`prism-media`）；WAV 内置解码、mp3/aac/FLAC
  走 ffmpeg 探测路径（`prism-audio`）；MP4-MJPEG 回读（`decode_mp4_video`）。
- **EXR 全家桶**：读写、压缩、DWA，且与官方 OpenEXR 做**跨实现比特流对拍**（不是自写自读）。
- **相机 RAW**：基线 RAW 解码。**边界**：真实 DNG 的 `Compression=7`（lossless JPEG）未实现，
  真实相机拍出的 DNG 读不了——写在「诚实的边界」里，不藏。
- **三维资产**：glTF / OBJ 导入，USD（`.usda`）解析与 MaterialX 参数包，mesh 可进渲染场景。
- **外部字体**：`load_font_file` 直接装载第三方 TTF/OTF，回退链逐级解析，
  缺字形给明确诊断而非豆腐块。

### 排什么（创作语法）

- **DSL v2 全块语法**：一套 `.prism` 描述"要什么、怎么摆、怎么动、长什么样、
  导出成什么"，编译产物是节点图 + 属性通道 + 时间线。
- **五组合子**：`title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` 高层组合子展开成节点图，"90 秒成片"从手写场景代码降为拼装。
- **转场与 morph**：Fade / Wipe / Dissolve / Push 全参数化（时长、缓动、方向），
  Morph 折线变形，场景序列任意时刻可采样。
- **公式排版**：`math_ast` + `math_layout` 排上下标 / 分式 / 根号 / 大型算子 /
  矩阵，DSL `formula` 块直排，与正文同基线。

### 画什么（渲染与仿真）

- **离线照片级渲染**：路径追踪 + 物理正确 BSDF（漫反射 / 镜面 / 玻璃 / 金属），
  玻璃与金属走随机逆（Neumann 级数蒙特卡洛），支持体积介质与次表面散射。
- **曲线图元四族**：圆柱 / 圆锥 / 圆环 / 旋转曲面，全部解析求交（四次方程走
  Ferrari 闭式解，精确根）；任意三角 mesh + SAH-BVH，双路径 parity 对拍。
- **GPU compute**：设备无关 compute 语义层 + Vulkan dispatch + 手写最小 SPIR-V；
  Mesa lavapipe 软件 ICD 下无 GPU 平台完整可跑（headless 可验）。
- **物理与粒子**：XPBD 统一求解器（刚体 / 软体 / 布料 / 关节 / 流体 / 破碎）、
  烟雾 FLIP、粒子系统（AudioReactive 跟音频包络联动）。
- **生成艺术与后处理**：流场 / L 系统 / 元胞自动机 / 反应扩散 / 分形 / Voronoi；
  blur / bloom / halation / 胶片颗粒 / NPR 四风格（toon / watercolor / halftone / oil）/ 三合一降噪。
- **调色与矢量**：二级调色（HSL 选区 + RGB 曲线）、cinematic / film / anime 风格预设与
  草稿/标准/电影三档画质（质量是参数，不是玄学）；矢量路径光栅化、路径 draw-on 与 morph。
- **音频处理**：响度归一（loudnorm）、节拍检测、频谱分析、包络提取——
  可与画面联动（粒子 AudioReactive 跟音频包络走）。
- **CPU 参考后端是一等公民**：无 GPU 机器照样出正确图像，同时是 GPU 的判对基准。

### 出什么（交付）

- **`render_movie` 直达**：DSL v2 编译产物一次调用走完"逐帧求值 → tile 渲染 →
  交付"，`.prism` 单命令出成片（`prism-app render demo.prism`）；未覆盖节点如实报错不静默近似。
- **预览闭环**：`prism-app preview` 起预览服务；编辑器 Draft 档（几何图案保帧率）/
  Real 档（低 spp 真实渲染、异步刷新）双档预览。
- **自研编码器 + 率控**：I/P 帧 JPEG 域时序编码器走私有 `PRMV` 容器（v2/v3 两代
  schema），water-filling 率控 + qpel 子像素运动估计（实测 PSNR +2.577 dB、
  体积 −21.6%）。
- **ffmpeg 协作链**：有 ffmpeg → 经 stdin rawvideo 管道转 H.264/HEVC；
  无 ffmpeg → 自动降级 MJPEG，**引擎永远能出片**；不捆绑任何二进制。
- **EXR 序列交付**：帧序列直接写 EXR，进后期管线。
- **分布式农场**：帧级分发 + N 进程 worker + 确定性合并——并行 MD5 == 串行 MD5，
  支持断点续算。

### 怎么集成（生态）

- **MCP 工具面**：stdio JSON-RPC 三工具——`scene_check`（编译 + 试渲 + 帧数预估）、
  `render_submit`（确定性 job id + 逐帧 MD5 摘要）、`diagnose`（三段式错误定位）。
- **节点 SDK**：语义化版本协商（含 0.x 特例），随包附示例节点包
  （`examples/sdk-nodes/cinematic-basic/`，可编译 / 可加载 / 可协商）。
- **硬编探测骨架**：`prism-ffi-hwenc` 三平台（VideoToolbox / NVENC / VAAPI）
  探测链 + feature gate 矩阵，平台不符明确报不支持。

> 详细能力边界（哪些**做不到**）见解包后 `开发者指南.md` 的「诚实的边界」一节——
> 那里记着当前真实状态（曲线图元只走 CPU、GPU 加速比需在目标硬件实测、硬编按
> 目标平台接入…），本文只给全景，不吹。

---

## 引擎内特能力（功能清单之外的那些）

有些能力不适合塞进「读 / 排 / 画 / 出 / 集成」的功能列表——它们是引擎内部的**深水区资产**。
单开这一章，避免主功能列表被细节淹没。

### 音频：一套内置 DAW 内核

`prism-audio` 不是"能读 WAV"的边缘模块，而是从作曲到混音的完整链路，每个函数都是
参数的纯函数（同输入跨平台逐位一致）：

- **合成**：PolyBLEP 防混叠振荡器族 + ADSR 闭式包络 + 4 情绪预设；
- **乐理**：音阶库（大小调/五声/中古调式）、12-TET 音级↔频率、和弦进行**功能语法状态机** + 声部引导；
- **编排**：琶音器（模式/速率/八度）、step 鼓机、小节/节拍网格配乐编排；
- **素材**：手写 SMF 解析（MIDI 导入）、SFZ 采样器、颗粒合成；
- **声音设计**：物理建模、空间声像、效果器链；
- **混音**：多轨线性混合 + 淡入淡出 + 峰值防削波；
- **分析与联动**：节拍/onset 检测、FFT 频谱、包络提取、AudioReactive 烘焙
  （第 N 帧读固定窗，帧级纯函数、seek 等价）、响度归一。

> 边界：AAC/mp3 不自研（专利历史包袱 + 工程量），走外部 ffmpeg 协作。

### 可定义核库（特解注册表）

引擎的每个节点类型 = 一个「核 + 格 + 对偶」三元组**特解**。第三方（或你自己）可以
用 `registry!` 静态注册新特解，**不改引擎核心**：

- 静态汇入制（宏在定义 crate 内展开，禁 `inventory`/proc-macro，零依赖）；
- 注册时**三问校验**（核形式 / 格 / 对偶），答不齐直接拒绝并给三段式报错；
- 官方特解与自定义特解走**同一张注册表**，运行时无差别；
- 数学上对应「算子方程 `f = g + Kf` 的六种求逆策略」——精确 / 随机 / 局部 / 正则化 /
  迭代 / 谱，新特性的接入是**声明三元组**，而不是往渲染循环里打补丁。

### 确定性与可复现基建

- **签名闭包**：节点图 / 缓存键 / 快照键 / 帧签名统一为「四维配置哈希」，改参数
  ⇒ 键必变 ⇒ 不会误复用；迭代节点键含 `(ε, n)`，流节点键含帧号；
- **字节不动点**：USD 导出→导入→导出逐字节一致；WAV 16/24/32bit 往返一致；
  EXR 与官方实现跨实现比特流对拍；
- **单文件分发确定性**：同一源码两次打包 SHA-256 完全一致，解包回环逐文件逐字节校验。

### GPU 后端（诚实分述）

- **Vulkan**：compute + 光追场景双管线**真实运行**（lavapipe 软件 ICD 上无 GPU 可验），
  `render_scene` 出帧与 CPU 参考**逐字节 0 差异**；`is_available()` 是 compute ∧ render 双自检；
- **Metal**：同构移植 + 编译门控就绪，**未实机验证、不点亮**（`is_available()` 恒 false）；
- **WebGPU**：可行性结论已出（迁移量 4–6 人日），渲染路径未实现。

### 反假绿门禁

四个本地门禁脚本（零依赖、只用 `node`）：幽灵写入防御（含反向自证）、开源卫生扫描、
合规自检（9/9 + 反向自证 8/8 判红）、打包回环校验。门禁只有「通过」或「判红」两种结果，
不接受"看起来过了"——**假绿比失败更危险**。

---

## 为什么是单文件分发

- **可审计**：一个文本文件，人眼可读、可 `diff`、可哈希校验，发布即冻结。
- **可复现**：打包具确定性——相同输入两次打包 SHA-256 完全一致。
- **可携带**：无需 clone 庞大仓库，一个文件经邮件或聊天即可传送，离线亦可还原构建。
- **取舍说明**：单文件分发以让渡"在 GitHub 上直接浏览代码"的便利为代价，换取"一个文件带走全部、且逐字节可验证"的确定性。解包后得到的是普通 Cargo workspace，与常规 Rust 项目无异。

---

## 快速开始

前置：安装 [Rust](https://www.rust-lang.org/)（含 `cargo`）与 [Node.js](https://nodejs.org/) 18+。

```bash
# 1. 下载仓库中的 prism-engine-v2.1-source.txt，存为本地文件
# 2. 用文本编辑器打开，把 "UNPACK SCRIPT BEGIN" 与 "UNPACK SCRIPT END"
#    之间的 Node.js 脚本另存为同目录下的 unpack.cjs
# 3. 运行解包（约数秒，还原 642 个文件）
node unpack.cjs
# 4. 在还原出的目录（即 workspace 根）跑全量测试
cargo test --workspace --release
# 预期：3929 passed / 0 failed
```

解包后得到一个标准 Cargo workspace（`src/` 下 28 个 crate），后续 `cargo` 命令与常规 Rust 项目相同。

> 手机端 wasm 验证链路：`rustup target add wasm32-unknown-unknown`，再 `cargo build -p prism-core --target wasm32-unknown-unknown`。

---

## 仓库结构（解包后）

| 路径 | 说明 |
|------|------|
| `src/` | **28 个 crate** 的源码（仓库内 29 个，原名 `crates/`，分发时重组为 `src/`；演示 crate `prism-anim-demo` 不进包） |
| `Compliance/` | 合规声明：零依赖核验 / 第三方声明 / 资产登记（仓库内原名 `docs/合规/`） |
| `开发者指南.md` | 自包含开发者文档：上手 / 架构 / 数学内核 / 门禁 / 扩展（仓库内原名 `docs/开发者指南.md`） |
| `tools/` | 门禁脚本（幽灵写入防御、卫生扫描、合规自检） |
| `.github/workflows/` | CI 配置 |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | 构建必需件（workspace members 已指向 `src/*`） |
| `examples/` | `prism-dsl` 测试用样例与示例节点包（`include_str!` 读取，缺了编译不过） |

> 设计权威文档（`docs/设计/`）**不随分发**，仅留在发布方的完整仓库中。

---

## 模块概览

分发包内 **28 个 crate** 按职责分组（仓库另有演示 crate `prism-anim-demo`，只在仓库里
作为可复现示例，不进包）：

| 类别 | crate |
|------|-------|
| 内核 | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| 媒体 I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| 渲染 | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| 动画/生成 | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| 文本/UI | prism-text · prism-rt · prism-app |
| 高级 | prism-ai · prism-dsl · prism-infra · prism-author |
| 旁路（feature gate） | prism-ffi-hwenc |

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

- 全量测试：**3929 passed / 0 failed / 0 ignored**（仓库内 release 实跑，179 个测试段，
  全核并行 171.5 秒）；分发包与仓库**逐文件逐字节一致**，差异仅演示 crate
  `prism-anim-demo` 不进包（其测试块为空，用例数相同）。
- CI：**6 个 job**——幽灵写入防御（ghost-guard）/ 编译 + 全量测试（native）/ wasm32 构建（wasm）/
  WGSL 静态检查（wasm-verify）/ 与官方 OpenEXR 跨实现比特流对拍（interop）/ node 宿主交叉判对（lint）。
- 本地门禁脚本 3 个（全零依赖、只用 `node`，下）+ 打包脚本 1 个；
  其中合规自检 **9/9 通过**、反向自证 **8/8 判红**。

本地运行门禁（需 Node 18+）：

```bash
node tools/ghost_guard.mjs --check        # 幽灵写入防御：G1 重复定义/G2 严格超集/G3 顶格文档/G4 编译
node tools/hygiene_scan.mjs               # 开源卫生：逐条列出残留的内部用语（文件:行号），命中即 exit 1
node tools/compliance_check.mjs           # 合规自检：应 9/9 通过
node tools/compliance_check.mjs --self-test   # 反向自证：应判红，证明判据具备判别力
```

> **开源卫生扫描当前仍有存量命中**：仓库内残留早期内部用语，其中**随包分发的
> 文档层已清零**（README / 开发者指南 / Compliance），余下在**分发包内源码注释**
> 与**不随包分发的仓库文件**中，已据实分类并列为独立清理项。脚本的存在正是为了让
> 残留**可见、可逐条清**，而不是假装干净——请勿把"命中数不为 0"理解为门禁失效。

---

## 合规

引擎承诺**零第三方代码依赖**，并经自检核验：

- **C1 零第三方代码依赖**：全部 130 条依赖（110 条运行期 + 20 条测试）均为本仓 `path` 依赖；无 `build.rs`、无原生链接指令、`Cargo.lock` 锁定 29 个本仓包。
- **C2 资产登记完备**：5 项资产主表字段齐全、自研化状态落在受控词表；OFL 1.1 字体许可证全文随字体置于 `src/prism-text/font_data/LICENSE`。
- **C3 第三方归属声明**：`Compliance/THIRD-PARTY-NOTICES.md` 在场，并明确区分"数据资产 ≠ 代码依赖"。

> 注：本仓库分发的是**源码**。字体资产（Prism Sans SC 子集）以 SIL OFL 1.1 许可，**许可证全文随字体置于** `src/prism-text/font_data/LICENSE`；资产来源与许可登记见 `Compliance/assets/资产登记.md`。

---

## 确定性打包（发布方）

如需从完整仓库重新生成分发 TXT（版本号可传，默认出版当前版本 v2.1）：

```bash
node tools/pack_source_txt.mjs          # 默认 → dist/prism-engine-v2.1-source.txt
node tools/pack_source_txt.mjs 2.1      # 等价写法（显式传版本号）
node tools/pack_source_txt.mjs 1.0      # 回到 v1.0 时代产物名（在 v1.0 源码上跑即逐字节复现旧包）
node tools/pack_source_txt.mjs --verify # 打包后立刻解包回环校验（逐文件逐字节比对源码树）
# 连跑两次，输出 SHA-256 一致即证明确定性打包成立
```

打包范围严格限定为：`crates/`（→ `src/`）、`docs/合规/`（→ `Compliance/`）、`docs/开发者指南.md`、`tools/`、`.github/`、`.cargo/`、`examples/`、`Cargo.toml`、`Cargo.lock`。刻意**排除** `docs/设计/`、`target/`、`node_modules/`、`dist/` 本身及一切中间产物。

---

## 贡献

欢迎 Issue 与 Pull Request。提交前请保持：

- `cargo test --workspace --release` 全绿；
- `node tools/hygiene_scan.mjs` **不新增**内部用语（脚本会列出全部命中，含存量）；
- `node tools/compliance_check.mjs` 9/9 通过。

更详细的架构与扩展指南见解包后的 `开发者指南.md`。

---

## 许可证

本项目以 **Apache License 2.0** 发布。详见 [`LICENSE`](LICENSE) 文件。

> 合规提示：Apache-2.0 要求保留版权与许可声明；基于本引擎再行分发时，请保留 `Compliance/` 下的归属声明与资产登记。
