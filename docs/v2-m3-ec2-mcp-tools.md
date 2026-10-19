# EC-2 MCP 工具面清单（V2.0-M3）

> 数据同源：本文清单与 `crates/prism-infra/src/mcp.rs` 的 `McpServer::tools()`
> 逐条对应（7 个 `ToolDef`）。判据代码：`crates/prism-infra/src/mcp.rs`
> （工具实现）、`crates/prism-infra/src/engine.rs`（引擎门面）、
> `crates/prism-infra/tests/qa_edge_attack_infra.rs`（QA round-trip）。
> 复现命令：`cargo test -p prism-infra`。

## 1. 协议基本面

stdio 传输上的自写 JSON-RPC 2.0 子集（零第三方 SDK），三类消息：

| 方法 | 语义 |
|---|---|
| `initialize` | 握手：回 `protocolVersion` / `serverInfo` / `capabilities` |
| `tools/list` | 工具清单：7 个 `ToolDef`（name + description + inputSchema） |
| `tools/call` | 调用：参数 `{name, arguments}`，返回 MCP `content` 块数组 |

传输层之外，`prism-infra` 另提供 HTTP 出口（`http.rs`）与 CLI（`cmd_mcp`），
三出口共用同一引擎门面（`engine.rs`），行为逐字节一致。

## 2. 工具清单（7 个）

| # | 工具 | 参数（required = `source`） | 返回 | 失败（RPC error） |
|---|---|---|---|---|
| 1 | `compile` | `source` | 文本：`{w}x{h} @ {fps}fps, {dur}s, format={fmt}` | 编译失败 → `-32602` 带行号 |
| 2 | `render_frame` | `source`, `t?=0`, `w?`, `h?` | 文本说明 + **image 块**（base64 PNG） | 同上 |
| 3 | `render_video` | `source`, `w?`, `h?`, `fps?` | 文本（帧数）+ **image 块**（首帧 PNG） | 同上（含 0 帧判定） |
| 4 | `critique` | `source`, `t?=0`, `w?`, `h?` | 文本：均值 / 最暗 / 最亮 / 动态范围 | 同上 |
| 5 | `scene_check` | `source`, `w?`, `h?`, `fps?` | 文本：元信息 + 预估帧数 + 首帧试渲结论 | 编译失败 / 试渲失败 / 分辨率非法 → `-32602` |
| 6 | `render_submit` | `source`, `w?`, `h?`, `fps?` | 文本：`job=` + 帧数 + 逐帧 MD5 摘要链；**image 块**（首帧 PNG） | 同 `render_video` |
| 7 | `diagnose` | `source`, `t?=0`, `w?`, `h?`, `fps?` | 文本：编译失败 → 三段式定位（**工具本身成功**）；编译成功 → 帧统计 + 病理旗标 | 仅编译通过但渲染失败 → `-32602` |

`w?/h?/fps?` 为可选覆盖参数（`integer`），缺省取 DSL 声明值；`t?` 为
`number`（秒），缺省 0.0。`required` 恒为 `["source"]`。

## 3. EC-2 三新工具语义（本波次落地 / 完善）

### 3.1 `scene_check`（场景检查 = 提交前 preflight）

与 `compile` 的分工：`compile` 只回答"能不能编译"，`scene_check` 回答
"按当前参数能不能真的出片、预计出几帧"：

1. 编译（失败 → `-32602`，携带 DSL 行号——预检不留情）；
2. 生效参数 = `arguments` 覆盖或 DSL 声明；宽/高为 0 → 显式报错
   （"当前参数会产出 0 像素帧"，C20 不静默容错）；
3. 预估帧数 = `engine::frame_count(fps × duration)`（与 CLI 同一公式）；
4. 帧数 > 0 时**真渲一帧**验证出片能力（编译通过 ≠ 能出片：空节点 /
   资产缺失只在渲染期暴露）；
5. 帧数 == 0 时**不报错但警告**：场景合法、提交必败，让调用方在提交前知道。

对拍（`scene_check_roundtrip_passes_scene_and_reports_failure_with_line`）：
成功场景回"场景检查通过 / 32x24 @ 4fps / 预估帧数：1 / 首帧试渲：OK"；
坏源码回 `-32602` 且消息含"第 1 行"。

### 3.2 `render_submit`（渲染提交 = 内容寻址回执）

- **确定性 job id**：`md5_hex("submit|source|w|h|fps")`——提交参数的内容
  寻址（纯 `prism-render` 侧零依赖 MD5），同源码同参 ⇒ 同 id，与提交
  次数 / 顺序无关（自增 id 会依赖提交顺序，破坏确定性，故不用）。
- **逐帧 MD5 摘要链**：按帧号顺序拼接 `md5(RGBA 像素)`——整段产物的内容
  指纹。分布式渲染（farm）合并后可用同一公式对拍，即"并行产物 ==
  串行产物"的机读判据。
- 返回 MCP `content`：文本块（job / 帧数 / 摘要链）+ 首帧 image 块。

对拍（`render_submit_roundtrip_returns_content_addressed_job_and_per_frame_digests`）：
2 帧场景 ⇒ 摘要链恰 2 条 32 位小写十六进制且互不相同；两次提交响应
逐字节相同（job 幂等）；坏源码 → `-32602`。

### 3.3 `diagnose`（错误诊断 = 把坏输入变成诊断结果）

关键语义：**诊断工具不把"坏输入"当协议错误**——调用方（LLM / 编辑器）
要的是"哪里坏、怎么修"的结构化文本，不是一个 `-32602`。

- 编译失败 → RPC **成功**（`result`），文本含：位置（第 N 行 / 引擎级）、
  消息、建议三段（C18 三段式 + C19 必含建议段）。DSL 消息内嵌的
  `建议：` 段优先拆出（`split_dsl_message`）；无内嵌建议时兜底建议
  必带行号（测试钉住两条路径）。
- 编译成功 → 真渲 `t` 帧，回报均值 / 最暗 / 最亮 / 动态范围，并给
  **病理旗标**（`pathology_flags` 纯函数，优先级链：全黑 ⊃ 全白 ⊃
  动态范围坍缩（差 < 16），每帧最多一个旗标）。
- 仅当编译通过但**渲染**失败（分辨率非法等）才回 `-32602`——那是诊断
  能力内的事实，不能伪装成成功文本。

对拍（`diagnose_roundtrip_reports_failure_without_rpc_error_and_stats_on_success`）：
坏源码 ⇒ `error` 字段缺席且文本含"编译失败 / 第 1 行 / 建议："；
好场景 ⇒ 文本含"场景可编译 / 均值 / 病理旗标"。

## 4. Round-trip 对拍测试覆盖（最小充分集）

| 测试 | 对拍内容 |
|---|---|
| `tools_list_contains_seven_tools_including_ec2_faces` | 工具数 4→7，7 个名字齐 |
| `scene_check_roundtrip_…` | 成功报告 + 失败行号 |
| `render_submit_roundtrip_…` | job 幂等 + 摘要链 + image 块 + 失败码 |
| `diagnose_roundtrip_…` | 失败不升级 RPC 错误 + 成功统计 |
| `pathology_flags_classify_flat_frames` | 旗标优先级链 + 阈值边界（差=16 不坍缩 / 15 坍缩） |
| `split_dsl_message_extracts_…` | 内嵌建议拆分 + 兜底建议带行号 |
| QA：`mcp_tools_list_schema_complete` | 7 工具 schema 完整（object + required） |
| QA：`mcp_ec2_tools_roundtrip` | 三工具在 QA 侧再对拍一次（防回归双保险） |

纪律：每工具 1 成功 + 1 失败（diagnose 的"失败"= 坏输入仍成功回诊断，
另有 QA 侧好场景对拍），**不堆料**——不复述已有 4 工具的既有测试。

## 5. 确定性口径

- 工具描述（`inputSchema`）由参数定义静态生成，无 RNG、无时钟、无环境
  读取（除 `CARGO_PKG_VERSION` 编译期常量）；
- `render_frame` / `render_submit` / `diagnose` 同请求同字节（既有测试
  `render_frame_is_deterministic_across_calls` + 本波次 job 幂等对拍）；
- `render_submit` 的 job id 是源码 + 生效参数的纯函数，跨进程 / 跨机器
  可复现——这正是它可被 farm 合并校验复用的原因。

## 6. 复现

```bash
cargo test -p prism-infra                 # 全量（lib 59 + engine_seek 9 + qa_edge 49 + qa_sweep 8）
cargo test -p prism-infra mcp_            # 仅 MCP 相关
tools/incremental_test.sh                 # 增量（按 BASE..HEAD 变更 crate 闭包）
```

环境受限项：本文档所有对拍均在本机（Linux 沙箱，32×24 / 32×24 小图）实测；
无障碍。MCP over HTTP / TCP 传输变体不在本波次范围（stdio 已覆盖
`initialize` / `tools/list` / `tools/call` 三类消息）。
