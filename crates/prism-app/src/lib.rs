//! Prism 引擎 **应用库**（交互式编辑器）。
//!
//! 本库只承载可被集成测试与其它 crate 复用的**逻辑**（编辑器状态机、增量
//! 失效、命令模式）；CLI 入口仍在 `src/main.rs`（binary 目标）。
//!
//! # 与既有 CLI 的关系
//! `main.rs` 里的 `cmd_batch` / `cmd_critique` / `cmd_http` / `cmd_mcp` /
//! `cmd_pack` 与内置演示**保持不变**——本库是**纯新增**，不触碰它们。
//!
//! # 零第三方依赖
//! 仅 `std` + workspace 内 crate（`prism-core` 签名树；Real 档预览复用
//! `prism-render` / `prism-gpu` 既有渲染管线，不另写渲染器）。

pub mod editor;
