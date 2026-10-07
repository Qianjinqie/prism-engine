//! `prism-infra` — 开发者面基础设施。
//!
//! 语义权威：开发者面与 C6 行（"MCP stdio JSON-RPC；
//! batch matrix；自写 HTTP；数据包市场校验（add/list/remove）"）。
//!
//! # 四个出口（共用同一引擎门面，行为一致）
//! - [`mcp`]：MCP server（自写 JSON-RPC 2.0，stdio 传输；tools: compile /
//!   render_frame / render_video / critique）；
//! - [`batch`]：批量渲染（任务清单解析 + 笛卡尔积展开 + 多线程帧渲染 + 进度）；
//! - [`http`]：自写 HTTP/1.1 服务器（路由 compile / render_frame，默认绑 localhost）；
//! - [`plugin`]：数据包市场（manifest 校验：semver / 许可证 / 资产清单；本地库
//!   add / list / remove）。
//!
//! # 支撑模块
//! - [`jsonrpc`]：自写 JSON-RPC 2.0 编解码（值模型 + 请求/通知/响应 + 错误码）；
//! - [`engine`]：引擎门面（DSL 编译 + 渲染 + PNG 编码），三个出口共用；
//! - [`farm_wire`]：农场线格式（主节点 ⇄ worker 的任务 / 分片编解码）。
//!
//! # 硬约束
//! - **零第三方依赖**（JSON-RPC / HTTP / TOML 子集 / base64 全部自写，仅 `std`）；
//! - 无全局 RNG（`CpuRefBackend` 参考渲染，同输入必同输出）；
//! - HTTP 默认绑 `127.0.0.1`（安全：不暴露公网）。

#![forbid(unsafe_code)]

pub mod batch;
pub mod engine;
pub mod farm_wire;
pub mod http;
pub mod jsonrpc;
pub mod mcp;
pub mod plugin;
