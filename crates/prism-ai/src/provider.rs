//! [`Provider`] trait — 引擎侧对外部 AI 服务/未来小模型的接口抽象（D-11）。
//!
//! # 定位（D-11 AI 边界）
//!
//! 引擎 = 纯渲染执行器：LLM → 旁白 → TTS → cues 的**编排**在应用侧；本 trait
//! 只是**插座式单次调用**的形态——给定一次请求（提示词/参数），返回一次响应
//! （文本/词级时间戳）。trait 让应用侧可注入真实服务（网络在应用侧实现）或
//! 引擎内置的确定性 [`MockProvider`](crate::mock::MockProvider)（用于测试与
//! 演示）。引擎自身**不做编排、不发网络**。

use crate::{AiError, WordTiming};

/// 一次 Provider 请求（纯数据；应用侧负责填充业务语义）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequest {
    /// 提示词 / 输入文本（编排的语义由应用侧定义）。
    pub prompt: String,
    /// 确定性种子（mock 按键派生输出；真实服务可忽略）。
    pub seed: u64,
    /// 期望的最大输出词数（上限；0 表示不设限）。
    pub max_words: usize,
}

impl ProviderRequest {
    /// 构造请求（默认不设词数上限）。
    pub fn new(prompt: impl Into<String>, seed: u64) -> Self {
        Self { prompt: prompt.into(), seed, max_words: 0 }
    }

    /// 设置最大输出词数（链式）。
    pub fn with_max_words(mut self, max_words: usize) -> Self {
        self.max_words = max_words;
        self
    }
}

/// 一次 Provider 响应：文本 + 词级时间戳（毫秒整数格，逐位可复现）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponse {
    /// 生成的整段文本。
    pub text: String,
    /// 词级时间戳（与 [`text`](Self::text) 的分词一一对应，顺序即出现序）。
    pub words: Vec<WordTiming>,
}

impl ProviderResponse {
    /// 构造响应。
    pub fn new(text: impl Into<String>, words: Vec<WordTiming>) -> Self {
        Self { text: text.into(), words }
    }
}

/// Provider 调用错误（三段式，宪章 C18/C19）。
pub type ProviderError = AiError;

/// AI Provider 接口（引擎侧抽象）。
///
/// 能力 = **一次请求 → 一次响应**（纯函数语义友好：同请求同响应）。实现方
/// 若需网络，网络逻辑在实现体内（应用侧注入）；引擎内置实现
/// [`MockProvider`](crate::mock::MockProvider) 保证零网络零随机。
pub trait Provider {
    /// 完成一次生成：返回文本与词级时间戳。
    ///
    /// 契约：对同一 [`ProviderRequest`]，实现应返回相同 [`ProviderResponse`]
    /// （确定性）；无法满足时返回 [`ProviderError`]（三段式，不 panic）。
    fn complete(&self, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError>;

    /// Provider 名称（诊断 / 日志用；确定性）。
    fn name(&self) -> &str;
}

/// 应用侧注入的**闭包适配器**：把 `Fn` 升格为 [`Provider`]。
///
/// 供测试或轻量接线使用（真实服务在应用侧实现本 trait 或经此闭包注入）。
pub struct FnProvider<F>
where
    F: Fn(&ProviderRequest) -> Result<ProviderResponse, ProviderError>,
{
    label: String,
    f: F,
}

impl<F> FnProvider<F>
where
    F: Fn(&ProviderRequest) -> Result<ProviderResponse, ProviderError>,
{
    /// 用一个函数构造 Provider。
    pub fn new(label: impl Into<String>, f: F) -> Self {
        Self { label: label.into(), f }
    }
}

impl<F> Provider for FnProvider<F>
where
    F: Fn(&ProviderRequest) -> Result<ProviderResponse, ProviderError>,
{
    fn complete(&self, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        (self.f)(req)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_provider_adapter_is_usable() {
        let p = FnProvider::new("inline", |req: &ProviderRequest| {
            Ok(ProviderResponse::new(req.prompt.clone(), Vec::new()))
        });
        assert_eq!(p.name(), "inline");
        let req = ProviderRequest::new("hi", 1).with_max_words(3);
        assert_eq!(req.max_words, 3);
        assert_eq!(p.complete(&req).unwrap().text, "hi");
    }
}
