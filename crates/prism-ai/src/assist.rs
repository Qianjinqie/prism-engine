//! A8 智能辅助（并入 C5 critic 闭环；D-11 边界内的**确定性 mock**）。
//!
//! # 定位（D-11）
//!
//! 引擎 = 纯渲染执行器：真正的"AI 编排"（LLM → 旁白 → TTS → cues）留在应用
//! 侧。本模块是 **A8 智能辅助并入 C5** 的落点——给 critic 的诊断结果产出**确定性
//! mock 辅助建议文本**（"智能辅助"的引擎侧形态），用于测试与演示：无网络、无
//! 密钥、无全局 RNG、无 libm，输出 = `hash(输入)` 的纯函数。
//!
//! # 与 critic 的解耦（架构纪律）
//!
//! 本模块**不依赖** `prism-aesthetic`：辅助建议的输入是**原语**
//! （[`AdviceRequest`]：维度名 + 方向 + 置信），由应用侧从 critic 诊断装配。
//! 这样 prism-ai 保持叶子依赖、critic 可独立演进（同构节点哲学，D-3/D-11）。
//!
//! # 产出形态
//!
//! [`AssistAdvice`]：一句话建议（确定性词表拼接）+ 建议动作标识（供应用侧映射
//! 到具体修复）+ 置信度（透传输入）。

use crate::hash::{derive, fnv1a64_str, mix64};
use crate::AiError;

/// 辅助建议请求（原语；应用侧从 critic 诊断装配）。
#[derive(Debug, Clone, PartialEq)]
pub struct AdviceRequest {
    /// 质量维度名（如 "exposure" / "contrast" / "color" / "sharpness" /
    /// "composition"；大小写不敏感，未知维度退化为通用措辞）。
    pub dimension: String,
    /// 修正方向："low" / "high"（其余按通用处理）。
    pub direction: String,
    /// 置信度 `[0,1]`（= critic 诊断 severity；本模块透传，不解释）。
    pub severity: f32,
    /// 确定性种子（同请求同输出；允许应用侧按会话变化）。
    pub seed: u64,
}

impl AdviceRequest {
    /// 构造请求（severity 自动钳到 `[0,1]`）。
    pub fn new(
        dimension: impl Into<String>,
        direction: impl Into<String>,
        severity: f32,
        seed: u64,
    ) -> Self {
        Self {
            dimension: dimension.into(),
            direction: direction.into(),
            severity: crate::assist::clamp01(severity),
            seed,
        }
    }
}

/// `[0,1]` 钳制（局部实现，避免额外依赖；D-16）。
#[inline]
pub(crate) fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// 辅助建议产出（确定性）。
#[derive(Debug, Clone, PartialEq)]
pub struct AssistAdvice {
    /// 一句话建议文本（确定性词表拼接）。
    pub text: String,
    /// 建议动作标识（供应用侧映射到具体修复；确定性 kebab-case）。
    pub action: String,
    /// 置信度（透传输入 severity）。
    pub confidence: f32,
    /// 本建议的确定性指纹（`hash(输入)`；便于缓存/去重）。
    pub fingerprint: u64,
}

/// 维度词表：给每个维度一组动词短语（确定性选取）。
fn dimension_verbs(dim: &str) -> &'static [&'static str] {
    match dim {
        "exposure" => &["lift the exposure", "raise the key light", "open up the shadows"],
        "contrast" => &["stretch the tonal range", "add a contrast curve", "deepen the blacks"],
        "color" => &["nudge the grade warmer", "balance the color cast", "lift saturation"],
        "sharpness" => &["add a light sharpen", "recover micro-detail", "soften the highlights"],
        "composition" => &["rebalance the framing", "shift toward the thirds", "breathe the edges"],
        _ => &["revisit the look", "tune the pipeline", "reconsider the grade"],
    }
}

/// 结合方向细化动词（high 一侧给"反向"短语）。
fn directional_verb(dim: &str, direction: &str, pick: usize) -> String {
    let verbs = dimension_verbs(dim);
    let base = verbs[pick % verbs.len()];
    if direction == "low" {
        base.to_string()
    } else {
        // high：给"收敛/压低"措辞（确定性前缀拼接，不新增词表）。
        format!("ease back: {base}")
    }
}

/// 建议动作标识（确定性 kebab-case：`<dim>-<direction>`；未知维度归一）。
pub fn action_id(dimension: &str, direction: &str) -> String {
    let dim = normalize_dim(dimension);
    let dir = if direction.eq_ignore_ascii_case("high") { "high" } else { "low" };
    format!("{dim}-{dir}")
}

/// 维度名归一（小写 + 未知归 "look"）。
fn normalize_dim(dim: &str) -> &'static str {
    let d = dim.trim();
    if d.eq_ignore_ascii_case("exposure") {
        "exposure"
    } else if d.eq_ignore_ascii_case("contrast") {
        "contrast"
    } else if d.eq_ignore_ascii_case("color") {
        "color"
    } else if d.eq_ignore_ascii_case("sharpness") {
        "sharpness"
    } else if d.eq_ignore_ascii_case("composition") {
        "composition"
    } else {
        "look"
    }
}

/// 确定性智能辅助：由 [`AdviceRequest`] 产出 [`AssistAdvice`]。
///
/// # 契约（D-11）
/// 纯函数：同请求 ⇒ 逐位相同建议；无网络、无随机、无隐藏状态。空维度名 →
/// 结构化错误（`AI5xxx`，三段式），不静默编造。
pub fn advise(req: &AdviceRequest) -> Result<AssistAdvice, AiError> {
    let dim_raw = req.dimension.trim();
    if dim_raw.is_empty() {
        return Err(AiError::new(
            "AI5001",
            "assist::advise",
            "AdviceRequest.dimension 为空",
            "提供维度名（exposure/contrast/color/sharpness/composition）",
        ));
    }
    let dim = normalize_dim(dim_raw);
    let dir = if req.direction.eq_ignore_ascii_case("high") { "high" } else { "low" };

    // 确定性派生：指纹 + 措辞下标。
    let mut h = fnv1a64_str(dim);
    h ^= fnv1a64_str(dir).rotate_left(29);
    h ^= fnv1a64_str(&req.dimension); // 保留原始大小写差异作为熵
    let fingerprint = mix64(h ^ req.seed);
    let pick = (derive(fingerprint, "verb") % 8) as usize;

    let verb = directional_verb(dim, dir, pick);
    let level = severity_word(req.severity);
    let text = format!("{level} — {verb}.");

    Ok(AssistAdvice {
        text,
        action: action_id(dim_raw, &req.direction),
        confidence: req.severity,
        fingerprint,
    })
}

/// 置信度 → 强度措辞（确定性分段，非连续映射以保逐位稳定跨平台）。
fn severity_word(severity: f32) -> &'static str {
    if severity >= 0.75 {
        "strong signal"
    } else if severity >= 0.4 {
        "moderate signal"
    } else if severity > 0.0 {
        "subtle signal"
    } else {
        "no signal"
    }
}

/// 批量辅助：对一组请求逐条产出（顺序 = 输入顺序，确定性）。
pub fn advise_all(reqs: &[AdviceRequest]) -> Result<Vec<AssistAdvice>, AiError> {
    reqs.iter().map(advise).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advise_is_deterministic_same_request_same_advice() {
        let req = AdviceRequest::new("exposure", "low", 0.8, 7);
        let a = advise(&req).unwrap();
        let b = advise(&req).unwrap();
        assert_eq!(a, b, "同请求同建议（逐位）");
        assert_eq!(a.action, "exposure-low");
        assert_eq!(a.confidence, 0.8);
        assert!(a.text.contains("strong signal"), "置信 0.8 → strong: {}", a.text);
    }

    #[test]
    fn advise_distinguishes_dims_and_directions() {
        let low = advise(&AdviceRequest::new("exposure", "low", 0.5, 1)).unwrap();
        let high = advise(&AdviceRequest::new("exposure", "high", 0.5, 1)).unwrap();
        assert_ne!(low.text, high.text, "方向不同 → 措辞不同");
        assert_eq!(high.action, "exposure-high");
        assert!(high.text.contains("ease back"), "high 方向含收敛措辞：{}", high.text);

        let c = advise(&AdviceRequest::new("contrast", "low", 0.5, 1)).unwrap();
        assert_ne!(low.action, c.action);
        assert_eq!(c.action, "contrast-low");
    }

    #[test]
    fn advise_seed_changes_fingerprint() {
        let a = advise(&AdviceRequest::new("color", "low", 0.3, 1)).unwrap();
        let b = advise(&AdviceRequest::new("color", "low", 0.3, 2)).unwrap();
        assert_ne!(a.fingerprint, b.fingerprint, "种子影响指纹");
        // 但动作标识与置信与种子无关。
        assert_eq!(a.action, b.action);
        assert_eq!(a.confidence, b.confidence);
    }

    #[test]
    fn severity_words_are_banded() {
        assert_eq!(severity_word(0.0), "no signal");
        assert_eq!(severity_word(0.2), "subtle signal");
        assert_eq!(severity_word(0.5), "moderate signal");
        assert_eq!(severity_word(0.9), "strong signal");
    }

    #[test]
    fn empty_dimension_is_structured_error() {
        let err = advise(&AdviceRequest::new("  ", "low", 0.5, 1)).unwrap_err();
        assert_eq!(err.code, "AI5001");
        assert!(err.to_string().contains("建议："), "三段式含建议段");
    }

    #[test]
    fn unknown_dimension_normalizes_to_look() {
        let a = advise(&AdviceRequest::new("weirdness", "low", 0.5, 1)).unwrap();
        assert_eq!(a.action, "look-low");
    }

    #[test]
    fn advise_all_preserves_order() {
        let reqs = vec![
            AdviceRequest::new("exposure", "low", 0.6, 1),
            AdviceRequest::new("color", "high", 0.9, 1),
        ];
        let out = advise_all(&reqs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].action, "exposure-low");
        assert_eq!(out[1].action, "color-high");
    }

    #[test]
    fn severity_is_clamped_on_construction() {
        let r = AdviceRequest::new("exposure", "low", 5.0, 1);
        assert_eq!(r.severity, 1.0);
        let r2 = AdviceRequest::new("exposure", "low", -3.0, 1);
        assert_eq!(r2.severity, 0.0);
    }
}
