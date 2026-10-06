//! [`MockProvider`] — 确定性 AI mock（B10；无网络、无随机、无 libm）。
//!
//! # 确定性契约
//!
//! 输出（文本 + 词级时间戳）**完全由 `hash(seed, prompt)` 派生**：
//!
//! - 整段文本 = 由词表按派生索引拼出的短语（词数与语境由派生值决定）；
//! - 词级时间戳 = 整数毫秒格，逐词时长由同一派生序列决定，无浮点漂移。
//!
//! 同一 [`ProviderRequest`] ⇒ 逐字节相同的 [`ProviderResponse`]；跨平台一致。

use crate::hash::{derive, fnv1a64_str, mix64};
use crate::provider::{Provider, ProviderError, ProviderRequest, ProviderResponse};
use crate::timings::WordTiming;

/// mock 词表（确定性短语来源；纯常量，无外部数据）。
const VOCAB: &[&str] = &[
    "prism", "engine", "render", "scene", "light", "shadow", "wave", "field",
    "lattice", "kernel", "camera", "motion", "color", "frame", "sample", "detail",
];

/// 每词时长基准（毫秒整数格）。
const WORD_MS_BASE: u64 = 180;
/// 每词时长的派生调制幅度（毫秒）。
const WORD_MS_SPAN: u64 = 120;
/// 词间停顿（毫秒），在 cue 切分时作为静默间隙基准。
const GAP_MS: u64 = 60;

/// 确定性 AI mock Provider。
///
/// - `salt`：实例级盐，允许同一请求在不同 mock 实例上产出不同（但每实例确定）
///   的结果；默认盐为空串。
#[derive(Debug, Clone)]
pub struct MockProvider {
    salt: String,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self { salt: String::new() }
    }
}

impl MockProvider {
    /// 构造带盐的 mock（盐参与派生，实例级区分）。
    pub fn with_salt(salt: impl Into<String>) -> Self {
        Self { salt: salt.into() }
    }

    /// 由请求派生基础种子。
    fn seed_of(&self, req: &ProviderRequest) -> u64 {
        let mut h = fnv1a64_str(&req.prompt);
        h ^= fnv1a64_str(&self.salt).rotate_left(17);
        mix64(h ^ req.seed)
    }
}

impl Provider for MockProvider {
    fn complete(&self, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        if req.prompt.trim().is_empty() {
            return Err(ProviderError::new(
                "AI1001",
                "MockProvider::complete",
                "请求 prompt 为空，无法派生确定性输出",
                "提供非空 prompt（应用侧负责编排语义）",
            ));
        }

        let base = self.seed_of(req);

        // 词数：1..=6 由派生决定；受 max_words 上限约束（0 = 不限）。
        let mut word_count = (derive(base, "len") % 6) as usize + 1;
        if req.max_words > 0 {
            word_count = word_count.min(req.max_words);
        }

        // 逐词派生：词表索引 + 时长（整数毫秒格）；累计得时间戳。
        let mut text_parts: Vec<String> = Vec::with_capacity(word_count);
        let mut words: Vec<WordTiming> = Vec::with_capacity(word_count);
        let mut cursor_ms: u64 = 0;
        let mut prev_end_ms: u64 = 0;
        for i in 0..word_count {
            let salt_i = format!("w{}", i);
            let pick = derive(base ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15), &salt_i);
            let word = VOCAB[(pick % VOCAB.len() as u64) as usize];
            let dur_ms = WORD_MS_BASE + (derive(pick, "dur") % (WORD_MS_SPAN + 1));

            let start_ms = if i == 0 { 0 } else { prev_end_ms + GAP_MS };
            let end_ms = start_ms + dur_ms;
            words.push(WordTiming { word: word.to_string(), start_ms, end_ms });
            text_parts.push(word.to_string());
            prev_end_ms = end_ms;
            cursor_ms = end_ms;
        }
        let _ = cursor_ms; // 末词结束时刻（供上层取总时长时按 words 计算）

        Ok(ProviderResponse::new(text_parts.join(" "), words))
    }

    fn name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderRequest;

    #[test]
    fn mock_is_deterministic() {
        let p = MockProvider::default();
        let req = ProviderRequest::new("生成一段旁白", 42);
        let a = p.complete(&req).unwrap();
        let b = p.complete(&req).unwrap();
        assert_eq!(a, b, "同请求同响应");
        assert_eq!(a.text, b.text);
        assert_eq!(a.words, b.words);
        let c = p.complete(&ProviderRequest::new("生成一段旁白", 43)).unwrap();
        assert_ne!(a, c, "不同种子应产出不同响应");
    }

    #[test]
    fn mock_word_timings_are_monotonic_and_contiguous() {
        let p = MockProvider::default();
        let resp = p.complete(&ProviderRequest::new("hello world", 7)).unwrap();
        assert!(!resp.words.is_empty());
        let mut prev_end = 0u64;
        for (i, w) in resp.words.iter().enumerate() {
            assert!(w.start_ms >= prev_end, "时间戳非倒退");
            assert!(w.end_ms > w.start_ms, "每词时长为正");
            if i > 0 {
                assert_eq!(w.start_ms, prev_end + GAP_MS, "词间间隙应为固定 GAP_MS");
            }
            prev_end = w.end_ms;
        }
        // 文本词数与词级时间戳数量一致。
        assert_eq!(resp.text.split_whitespace().count(), resp.words.len());
    }

    #[test]
    fn max_words_caps_output() {
        let p = MockProvider::default();
        let resp = p
            .complete(&ProviderRequest::new("long prompt here", 5).with_max_words(2))
            .unwrap();
        assert!(resp.words.len() <= 2);
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let p = MockProvider::default();
        let err = p.complete(&ProviderRequest::new("   ", 1)).unwrap_err();
        assert_eq!(err.code, "AI1001");
        assert!(err.to_string().contains("建议："));
    }

    #[test]
    fn salt_changes_output_deterministically() {
        let req = ProviderRequest::new("x", 1);
        let a = MockProvider::default().complete(&req).unwrap();
        let b = MockProvider::with_salt("s").complete(&req).unwrap();
        assert_ne!(a, b);
        let b2 = MockProvider::with_salt("s").complete(&req).unwrap();
        assert_eq!(b, b2);
    }
}
