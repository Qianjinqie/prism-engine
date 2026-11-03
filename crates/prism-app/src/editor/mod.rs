//! 交互式编辑器：预览 = 草稿档位 + 子图签名增量。
//!
//! # 核心洞察
//! 编辑器的实时预览**不该全量重算**。项目已有「子图签名树」
//! （[`prism_core::signature`]，S10 内容↔摘要）：节点签名 = hash(结构, 参数,
//! 通道特征, 各子签名)，自底向上，。改一个节点的参数，只有
//! **该节点到根的祖先链**签名变化，兄弟子树签名逐位不变。
//!
//! 本模块把这一性质落地为编辑器的**精确增量失效**：
//!
//! 1. [`EditorState::apply_edit`] 应用一次编辑（改参数 / 改组合方式）；
//! 2. 用两次 [`prism_core::graph_signatures`]（编辑前 / 编辑后）逐节点比对，
//!    得到签名**实际变化**的节点集合；
//! 3. 把该集合的**下游闭包**（含自身）标记为 `dirty`——即"受影响子图"。
//!
//! 只在签名真的变化时才标记 dirty：若编辑把参数改成与旧值逐位相同的值，
//! 签名不变 ⇒ dirty 集合为空（幂等编辑不触发任何重算）。
//!
//! # 撤销/重做
//! 命令模式：每次 `apply_edit` 把一条 [`EditCommand`] 压入 `undo_stack`；
//! `undo()` 弹出并把命令**逆应用**（restore 旧参数），压入 `redo_stack`；
//! `redo()` 反之。任意位置多步撤销/重做都逐字段还原。
//!
//! # 草稿档位
//! [`PreviewQuality`]：`Draft` 用低采样数（[`DRAFT_SPP`]），`Final` 用高采样
//! 数（[`FINAL_SPP`]）。具体数值经 [`prism_core::CacheDims`] 的 `spp_tier`
//! 进入缓存键——档位切换天然使旧缓存失效（C13），无需手工清缓存。
//!
//! # 质量档位的两档分工（W7 / B3 定稿）
//! 两档是**同一个渲染器**的两个预算点，差异只有"内部分辨率 + spp"：
//! - `Draft`：内部 `⌈R/消4⌉` + 1 spp **真渲染**（`prism-rt`），再经固定整数盒
//!   上采样回 R。交互期用它保延迟；
//! - `Final`：R + 16 spp 真渲染。定稿质量。
//!
//! 两档都走 [`prism_render::RtBackend`] + [`prism_render::render_frame`]，即
//! `prism-app render` 的同一条出片通路——**预览不含任何第二套着色函数**。
//! Draft 也配 [`async_preview`] 的后台刷新（提交即返回、完成后替换缓冲）；
//! ABI 同步路径不变（已冻结）。
//!
//! # 零第三方依赖
//! 仅用 `std` + workspace 内 crate（`prism-core` 图 / 签名 / 节点模型；
//! Real 档复用 `prism-render` / `prism-gpu` / `prism-math` 管线）。
//! 编辑器不引入任何 UI / 序列化库；操作序列导出为手写 JSON 字符串。

pub mod abi;
pub mod async_preview;
pub mod command;
pub mod preview;
pub mod state;

pub use abi::EDITOR_ABI_VERSION;
pub use async_preview::{AsyncFrame, AsyncPreview, PreviewRequest};
pub use command::{EditCommand, EditError, ParamSpec};
pub use preview::{
    PreviewError, PreviewRenderer, RenderStats, DRAFT_DOWNSCALE, MAX_DIM, PREVIEW_BOUNCES,
};
pub use state::{
    DirtySet, EditorState, ExportError, PreviewQuality, EDITOR_SCHEMA_VERSION, DRAFT_SPP, FINAL_SPP,
};

/// 编辑器模块的公开入口再导出（供 `main.rs` 与集成测试使用）。
pub use state::build_reference_graph;
