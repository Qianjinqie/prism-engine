//! 词级时间戳 → cues（字幕切分，确定性；B10）。
//!
//! # 语义
//!
//! 给定一串 [`WordTiming`]（词 + 起止毫秒整数格），按**确定性切分规则**归并成
//! 若干 [`Cue`]（字幕条目）：每条 cue 覆盖一个连续词段 + 合并后的词表。
//!
//! 切分规则（纯函数，无浮点、无随机）：
//!
//! 1. **最大时长**：cue 时长（末词 end − 首词 start）不得超过 `max_cue_ms`；
//! 2. **最大词数**：cue 词数不得超过 `max_words`；
//! 3. **静默间隙**：相邻词间隙 > `gap_ms` 时在此断开（自然句读）。
//!
//! 规则按上序优先级检查——先看是否应因超限/超词/大间隙而**在加入当前词之前
//! 断开**。cue 的 `start_ms`/`end_ms` 为首/末词边界（不含词间静默的裁剪）。

use crate::AiError;

/// 词级时间戳（整数毫秒格：`start_ms < end_ms`，同 [`crate::mock`] 输出）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordTiming {
    /// 词面。
    pub word: String,
    /// 起始毫秒（含）。
    pub start_ms: u64,
    /// 结束毫秒（不含）。
    pub end_ms: u64,
}

impl WordTiming {
    /// 构造词级时间戳。
    pub fn new(word: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self { word: word.into(), start_ms, end_ms }
    }

    /// 时长（毫秒）；`end < start` 时返回 0（调用方应先行校验）。
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// 一条字幕条目：覆盖连续词段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    /// 起始毫秒 = 首词 `start_ms`。
    pub start_ms: u64,
    /// 结束毫秒 = 末词 `end_ms`。
    pub end_ms: u64,
    /// 该 cue 覆盖的词（顺序即出现序，原样保留）。
    pub words: Vec<WordTiming>,
    /// 拼接文本（词以单空格连接）。
    pub text: String,
}

impl Cue {
    /// 时长（毫秒）。
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// 切分配置（确定性；均为整数格）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueConfig {
    /// 单条 cue 最大时长（毫秒）。
    pub max_cue_ms: u64,
    /// 单条 cue 最大词数。
    pub max_words: usize,
    /// 词间静默间隙阈值（毫秒）：间隙 > 此值即断开。
    pub gap_ms: u64,
}

impl Default for CueConfig {
    /// 默认：单条 ≤ 3 秒、≤ 8 词、间隙 > 400ms 断开。
    fn default() -> Self {
        Self { max_cue_ms: 3000, max_words: 8, gap_ms: 400 }
    }
}

/// 词级时间戳 → cues（确定性切分）。
///
/// 校验：词序列须非倒退（`start < end` 且相对前一词 `start` 不减）；违反返回
/// 三段式错误（`AI2001`），绝不静默（宪章 C20）。空输入返回空 cues。
pub fn cue_split(words: &[WordTiming], cfg: CueConfig) -> Result<Vec<Cue>, AiError> {
    if cfg.max_words == 0 {
        return Err(AiError::new(
            "AI2002",
            "cue_split::cfg",
            "max_words = 0，无法归属任何词",
            "设 max_words ≥ 1（默认 8）",
        ));
    }

    // 前置校验：逐词时长与顺序（整数域，无浮点）。
    let mut prev_end: Option<u64> = None;
    for (i, w) in words.iter().enumerate() {
        if w.end_ms <= w.start_ms {
            return Err(AiError::new(
                "AI2001",
                format!("cue_split::words[{}]", i),
                format!("词的时长非正：start={} end={}", w.start_ms, w.end_ms),
                "保证每词 end_ms > start_ms（毫秒整数格）",
            ));
        }
        if let Some(pe) = prev_end {
            if w.start_ms < pe {
                return Err(AiError::new(
                    "AI2001",
                    format!("cue_split::words[{}]", i),
                    format!("词的时间戳倒退：start={} < 前一 end={}", w.start_ms, pe),
                    "保证词序列按出现序、时间戳不减",
                ));
            }
        }
        prev_end = Some(w.end_ms);
    }

    let mut cues: Vec<Cue> = Vec::new();
    let mut cur: Vec<WordTiming> = Vec::new();

    // 断开判据：把候选词加入当前 cue 后是否越界。
    let would_break = |cur: &[WordTiming], w: &WordTiming| -> bool {
        if cur.is_empty() {
            return false;
        }
        let first = &cur[0];
        let last = cur.last().expect("非空");
        let new_len = w.end_ms.saturating_sub(first.start_ms);
        new_len > cfg.max_cue_ms
            || cur.len() + 1 > cfg.max_words
            || w.start_ms.saturating_sub(last.end_ms) > cfg.gap_ms
    };

    for w in words {
        if would_break(&cur, w) {
            cues.push(close_cue(&cur));
            cur.clear();
        }
        cur.push(w.clone());
    }
    if !cur.is_empty() {
        cues.push(close_cue(&cur));
    }
    Ok(cues)
}

/// 收尾一条 cue（首/末词边界 + 拼接文本）。
fn close_cue(words: &[WordTiming]) -> Cue {
    let start_ms = words.first().map(|w| w.start_ms).unwrap_or(0);
    let end_ms = words.last().map(|w| w.end_ms).unwrap_or(0);
    let text = words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Cue { start_ms, end_ms, words: words.to_vec(), text }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造词序列快捷工具。
    fn w(word: &str, start: u64, end: u64) -> WordTiming {
        WordTiming::new(word, start, end)
    }

    #[test]
    fn empty_input_yields_no_cues() {
        assert!(cue_split(&[], CueConfig::default()).unwrap().is_empty());
    }

    #[test]
    fn golden_split_by_gap_and_limits() {
        // 4 词：前 2 词紧密，第 3 词前有大间隙（> 400ms），第 3-4 词紧密。
        let words = vec![
            w("a", 0, 180),
            w("b", 240, 420),
            w("c", 1000, 1180), // 间隙 580 > 400 → 断
            w("d", 1240, 1420),
        ];
        let cfg = CueConfig { max_cue_ms: 3000, max_words: 8, gap_ms: 400 };
        let cues = cue_split(&words, cfg).unwrap();
        assert_eq!(cues.len(), 2, "应按大间隙切成 2 条");
        assert_eq!(cues[0].text, "a b");
        assert_eq!((cues[0].start_ms, cues[0].end_ms), (0, 420));
        assert_eq!(cues[1].text, "c d");
        assert_eq!((cues[1].start_ms, cues[1].end_ms), (1000, 1420));
    }

    #[test]
    fn golden_split_by_max_words() {
        let words: Vec<WordTiming> =
            (0..5).map(|i| w("w", i * 260, i * 260 + 200)).collect();
        let cfg = CueConfig { max_cue_ms: 10_000, max_words: 2, gap_ms: 400 };
        let cues = cue_split(&words, cfg).unwrap();
        // 5 词，每 cue ≤ 2 词 → 3 条（2/2/1）。
        assert_eq!(cues.len(), 3);
        assert_eq!(cues[0].words.len(), 2);
        assert_eq!(cues[1].words.len(), 2);
        assert_eq!(cues[2].words.len(), 1);
    }

    #[test]
    fn golden_split_by_max_duration() {
        // 每词 600ms、无间隙：max_cue_ms=1500 → 每条最多 2 词（1200ms ok，1800 > 1500）。
        let words: Vec<WordTiming> =
            (0..4).map(|i| w("x", i * 600, i * 600 + 600)).collect();
        let cfg = CueConfig { max_cue_ms: 1500, max_words: 8, gap_ms: 1000 };
        let cues = cue_split(&words, cfg).unwrap();
        assert_eq!(cues.len(), 2);
        assert!(cues.iter().all(|c| c.duration_ms() <= 1500));
    }

    #[test]
    fn non_monotonic_timestamps_rejected() {
        let words = vec![w("a", 0, 100), w("b", 50, 200)];
        let err = cue_split(&words, CueConfig::default()).unwrap_err();
        assert_eq!(err.code, "AI2001");
        assert!(err.to_string().contains("建议："));
    }

    #[test]
    fn zero_max_words_rejected() {
        let cfg = CueConfig { max_cue_ms: 1000, max_words: 0, gap_ms: 100 };
        let err = cue_split(&[w("a", 0, 100)], cfg).unwrap_err();
        assert_eq!(err.code, "AI2002");
    }

    #[test]
    fn split_is_deterministic() {
        let words: Vec<WordTiming> =
            (0..7).map(|i| w("t", i * 300, i * 300 + 250)).collect();
        let cfg = CueConfig::default();
        assert_eq!(cue_split(&words, cfg), cue_split(&words, cfg));
    }
}
