//! `prism-ai` — AI mock 层（工作流 B10；路线图 B10 行 / D-11 / D-16）。
//!
//! # 边界（D-11，引擎 = 纯渲染执行器）
//!
//! LLM → 旁白 → TTS → cues 的**编排**留在应用侧；引擎内只放**确定性 AI mock
//! 节点**，用于测试与演示。本 crate 不发起任何网络调用、无密钥、无全局 RNG：
//! 所有输出都是输入的纯函数（由 [`hash`](mock::fnv1a64) 派生），因此
//! **同输入跨平台逐位一致**（架构 §4.4）。
//!
//! # 内容
//!
//! - [`provider`] — [`Provider`](provider::Provider) trait：引擎侧对外部 AI
//!   服务/未来小模型的接口抽象（插座式单次调用，不含编排）；
//! - [`mock`] — [`MockProvider`](mock::MockProvider)：确定性 mock（无网络无随机）；
//! - [`timings`] — 词级时间戳 → cues（字幕切分，确定性）；
//! - [`journal`] — 任务流水账 + resume（中断后跳过已完成阶段）；
//! - [`pipeline`] — 草稿渲染流水线（接引擎渲染，低清草稿档位）；
//! - [`assist`] — A8 智能辅助（并入 C5 critic 闭环；确定性 mock 辅助建议）；
//! - [`nodes`] — `registry!`（D-27）静态汇入 prism-core 的确定性 mock 节点。
//!
//! # 确定性契约
//!
//! 一切公开函数都是其参数的纯函数：无内部可变状态、无全局 RNG、无 libm、
//! 无网络。时长/切分以整数格（毫秒）表达，避免浮点漂移。

pub mod assist;
pub mod hash;
pub mod journal;
pub mod mock;
pub mod nodes;
pub mod pipeline;
pub mod provider;
pub mod timings;

pub use assist::{action_id, advise, advise_all, AdviceRequest, AssistAdvice};
pub use journal::{Journal, Stage, StageRecord, StageState};
pub use mock::MockProvider;
pub use nodes::__prism_collect;
pub use pipeline::{DraftConfig, DraftOutput, DraftPipeline};
pub use provider::{Provider, ProviderError, ProviderRequest, ProviderResponse};
pub use timings::{cue_split, Cue, WordTiming};

use std::fmt;

/// AI mock 层错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
///
/// 错误码分段：`AI1xxx` provider、`AI2xxx` timings、`AI3xxx` journal、
/// `AI4xxx` pipeline。
#[derive(Debug, Clone, PartialEq)]
pub struct AiError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl AiError {
    /// 构造三段式错误（C18/C19）。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            code,
            location: location.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for AiError {}
