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

/// V3.0 W2：七族既有求值器接线（D-28：只接线、不重实现）。
/// 放进库目标是为了让同源对拍集成测试直接调用 `evaluate`，而不是只做源码扫描。
pub mod w2_eval;

/// V2.5-B1：`config.ocio` → 定点 LUT → RGBA8 帧的产品路径桥。
/// 放进库目标是为了让集成测试能直接调用 `load_ocio_pipeline`，
/// 而不是只做源码扫描（判据 V6「产品路径非空断言」）。
pub mod ocio_bridge;

/// V3.0 W3（D-32 / 红线 1）：六族既有 crate 接线，只接线、不重实现。
/// 放进库目标是为了让同源对拍集成测试与 CLI 共用同一份求值凭证，
/// 而不是两端各写一份（§3.5 V5，禁止两端同错）。
pub mod w3_eval;

/// V3.0 W3（§3.5 V3 / 主理人裁定）：确定性可计算的成对盲测协议。
/// 判官不是主观自评、也不是 critic 分数，而是三条可复现「解析性」指标；
/// 样本池 / 种子 / 配对顺序 / 门槛全部冻结，可复现脚本与文档共用本模块。
pub mod w3_blind;
