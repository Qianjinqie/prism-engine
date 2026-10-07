//! critic 引擎命令（C5）：analyze / critique / suggest / apply + `--autofix`。
//!
//! 语义权威：critic 内建为引擎命令且接增量渲染、
//! C5 行（analyze/critique/suggest/apply + `--autofix`（走增量渲染））；CLI
//! 出口 `critique`（开发者面：MCP/CLI/HTTP 三出口之一）。
//!
//! # 命令四相（Deterministic、纯函数）
//!
//! | 相 | 输入 | 输出 |
//! |----|------|------|
//! | analyze | 帧缓冲 | [`Metrics`] |
//! | critique | 帧缓冲 + 规格 | [`Score`] + 诊断 |
//! | suggest | 诊断 | [`Suggestion`] 序列 |
//! | apply | 帧缓冲 + 建议 | 修正后的帧缓冲 |
//!
//! `--autofix` = 把四相串成 [`CriticLoop::autofix`] 的贪心爬山，重渲走 C3 增量
//! 会话（缓存/复用记账）。
//!
//! # 引擎命令形态（D-27 静态汇入的"命令"侧）
//!
//! 本模块同时暴露 [`CritiqueReport`]——应用侧 CLI/MCP/HTTP 三出口的**统一返回
//! 形态**（一份报告：指标 + 评分 + 诊断 + 建议），命令行与 JSON-RPC 都可序列化。

use prism_io::fbuf::Fbuf16F;

use crate::critic::{
    analyze, diagnose, score, Diagnosis, Metrics, Rubric, Score,
};
use crate::critic_loop::{
    apply_suggestion, suggest, Convergence, CriticLoop, Suggestion,
};

/// 命令相（诊断 / 调度用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// 分析：单帧 → 指标。
    Analyze,
    /// 批评：帧 → 评分 + 诊断。
    Critique,
    /// 建议：诊断 → 建议。
    Suggest,
    /// 施加：帧 + 建议 → 修正帧。
    Apply,
}

impl Phase {
    /// 相名（确定性）。
    pub fn name(self) -> &'static str {
        match self {
            Phase::Analyze => "analyze",
            Phase::Critique => "critique",
            Phase::Suggest => "suggest",
            Phase::Apply => "apply",
        }
    }
}

/// critic 引擎命令的统一产出（CLI / MCP / HTTP 三出口共用）。
#[derive(Debug, Clone, PartialEq)]
pub struct CritiqueReport {
    /// 单帧指标（多帧时为代表帧 / 均值帧语义由调用方决定）。
    pub metrics: Metrics,
    /// 评分（含各维分）。
    pub score: Score,
    /// 建议（由诊断派生，顺序确定）。
    pub suggestions: Vec<Suggestion>,
}

/// critic 引擎命令（无状态；所有状态经参数进、经产出出）。
///
/// # 用法
/// ```
/// use prism_aesthetic::{CriticCommand, Rubric};
/// use prism_io::fbuf::Fbuf16F;
///
/// let mut frame = Fbuf16F::new(8, 8);
/// frame.set(4, 4, [0.9, 0.2, 0.2, 1.0]);
///
/// let cmd = CriticCommand::new(Rubric::default());
/// let report = cmd.critique(&frame);
/// let fixed = cmd.apply(&frame, &report.suggestions);
///
/// // 四相皆为纯函数：同输入必得同产出（无隐藏状态）。
/// assert_eq!(cmd.critique(&frame).suggestions, report.suggestions);
/// assert_eq!(fixed.width, frame.width);
/// ```
#[derive(Debug, Clone)]
pub struct CriticCommand {
    rubric: Rubric,
}

impl Default for CriticCommand {
    /// 默认规格（[`Rubric::default`]）。
    fn default() -> Self {
        Self { rubric: Rubric::default() }
    }
}

impl CriticCommand {
    /// 以规格构造。
    pub fn new(rubric: Rubric) -> Self {
        Self { rubric }
    }

    /// 规格（只读）。
    pub fn rubric(&self) -> &Rubric {
        &self.rubric
    }

    /// `analyze`：单帧 → 指标。
    pub fn analyze(&self, fb: &Fbuf16F) -> Metrics {
        analyze(fb)
    }

    /// `critique`：单帧 → 评分 + 诊断（含建议派生）。
    pub fn critique(&self, fb: &Fbuf16F) -> CritiqueReport {
        let metrics = analyze(fb);
        let sc = score(&metrics, &self.rubric);
        let suggestions = suggest(&sc.diagnoses);
        CritiqueReport { metrics, score: sc, suggestions }
    }

    /// `suggest`：诊断 → 建议（纯映射）。
    pub fn suggest(&self, diagnoses: &[Diagnosis]) -> Vec<Suggestion> {
        suggest(diagnoses)
    }

    /// `apply`：把建议施加到一帧（逐像素；纯函数）。返回新帧缓冲。
    pub fn apply(&self, fb: &Fbuf16F, suggestions: &[Suggestion]) -> Fbuf16F {
        let mut out = Fbuf16F::new(fb.width, fb.height);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let mut c = fb.get(x, y);
                for s in suggestions {
                    c = apply_suggestion(c, s);
                }
                out.set(x, y, c);
            }
        }
        out
    }

    /// `--autofix`：贪心爬山闭环（接 C3 增量渲染）。返回（收敛态，最终评分，
    /// 轮次轨迹）。`render` / `fix` 契约见 [`CriticLoop::autofix`]。
    pub fn autofix<F, G>(
        &self,
        total_frames: u64,
        max_rounds: u32,
        render: F,
        fix: G,
    ) -> (Convergence, Option<f32>, Vec<crate::critic_loop::RoundReport>)
    where
        F: FnMut(u64) -> Fbuf16F,
        G: FnMut(Fbuf16F, &[Suggestion]) -> Fbuf16F,
    {
        let mut lp = CriticLoop::new(total_frames, self.rubric, 0);
        let (conv, best) = lp.autofix(max_rounds, render, fix);
        (conv, best, lp.rounds().to_vec())
    }

    /// 诊断（单独出口，供 CLI `--diagnose` 直取）。
    pub fn diagnose(&self, fb: &Fbuf16F) -> Vec<Diagnosis> {
        let m = analyze(fb);
        diagnose(&m, &self.rubric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, c);
            }
        }
        fb
    }

    #[test]
    fn critique_report_is_deterministic() {
        let cmd = CriticCommand::default();
        let fb = solid(16, 16, [0.3, 0.1, 0.1, 1.0]);
        let a = cmd.critique(&fb);
        let b = cmd.critique(&fb);
        assert_eq!(a, b, "同帧同报告");
    }

    #[test]
    fn analyze_phase_returns_metrics() {
        let cmd = CriticCommand::default();
        let m = cmd.analyze(&solid(8, 8, [0.5, 0.5, 0.5, 1.0]));
        assert_eq!(m.pixels, 64);
        // 中灰亮度的 log 均值 ≈ ln(0.5)。
        assert!(m.log_mean_luma < 0.0);
    }

    #[test]
    fn apply_phase_is_pure_and_reversible_when_empty() {
        let cmd = CriticCommand::default();
        let fb = solid(8, 8, [0.4, 0.2, 0.1, 1.0]);
        let out = cmd.apply(&fb, &[]);
        assert_eq!(out, fb, "无建议 → 恒等（逐位）");
    }

    #[test]
    fn apply_phase_brightens_underexposed() {
        let cmd = CriticCommand::default();
        let fb = solid(8, 8, [0.02, 0.02, 0.02, 1.0]);
        let rep = cmd.critique(&fb);
        let fixed = cmd.apply(&fb, &rep.suggestions);
        assert!(fixed.get(0, 0)[0] >= fb.get(0, 0)[0], "提亮或不变");
    }

    #[test]
    fn autofix_improves_and_records_rounds() {
        let cmd = CriticCommand::default();
        let (conv, score, rounds) = cmd.autofix(
            2,
            2,
            |_| solid(24, 24, [0.02, 0.02, 0.02, 1.0]),
            |fb, sug| {
                let mut out = Fbuf16F::new(fb.width, fb.height);
                for y in 0..fb.height {
                    for x in 0..fb.width {
                        let mut c = fb.get(x, y);
                        for s in sug {
                            if s.dim == crate::critic::Dim::Exposure {
                                c = apply_suggestion(c, s);
                            }
                        }
                        out.set(x, y, c);
                    }
                }
                out
            },
        );
        assert_eq!(conv, Convergence::Improving);
        assert!(score.unwrap() > 0.0);
        assert!(rounds.len() >= 2);
    }

    #[test]
    fn phase_names_are_stable() {
        assert_eq!(Phase::Analyze.name(), "analyze");
        assert_eq!(Phase::Critique.name(), "critique");
        assert_eq!(Phase::Suggest.name(), "suggest");
        assert_eq!(Phase::Apply.name(), "apply");
    }
}
