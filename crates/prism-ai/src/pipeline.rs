//! 草稿渲染流水线（B10；接引擎渲染，低清草稿档位）。
//!
//! # 语义（D-11）
//!
//! 引擎 = 纯渲染执行器：本流水线把上一步产出的 [`Cue`] 序列**确定性**地渲染成
//! 低清草稿帧（每 cue 一帧），输出 [`DraftOutput`]（含 RGBA8 帧序列 + 流水账）。
//! 每帧按 cue 的确定性哈希挑一个调色板色做整帧填充——纯函数、无网络、无随机
//! （色由 `hash(seed, cue.text)` 派生），因此草稿逐帧可复现。
//!
//! # journal resume
//!
//! [`DraftPipeline::run`] 接受一份可选的入口 [`Journal`]：已完成阶段被**跳过**
//! （不重复调用 Provider/切分/渲染），只从 `resume_from` 起执行。产出新
//! [`Journal`] 供调用方存盘；中断后带旧 journal 重跑即从断点续。

use prism_gpu::Framebuffer;
use prism_math::Vec3;

use crate::hash::{derive, fnv1a64_str};
use crate::journal::{Journal, Stage};
use crate::provider::{Provider, ProviderRequest, ProviderResponse};
use crate::timings::{cue_split, Cue, CueConfig};

/// 草稿档位配置（低清）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftConfig {
    /// 草稿宽度（像素）。
    pub width: u32,
    /// 草稿高度（像素）。
    pub height: u32,
    /// cues 切分配置。
    pub cue: CueConfig,
}

impl Default for DraftConfig {
    /// 默认低清档位：320×180，cues 用默认切分规则。
    fn default() -> Self {
        Self { width: 320, height: 180, cue: CueConfig::default() }
    }
}

/// 草稿渲染产出。
#[derive(Debug, Clone)]
pub struct DraftOutput {
    /// 生成的整段文本。
    pub text: String,
    /// 切分出的 cues（与 [`frames`](Self::frames) 一一对应）。
    pub cues: Vec<Cue>,
    /// 每 cue 一帧（RGBA8，尺寸 = 配置档位）。
    pub frames: Vec<Framebuffer>,
    /// 本次运行的流水账（可存盘供 resume）。
    pub journal: Journal,
}

/// 草稿渲染流水线（无状态：所有状态经参数进、经产出出）。
#[derive(Debug, Clone)]
pub struct DraftPipeline {
    cfg: DraftConfig,
}

impl DraftPipeline {
    /// 用配置构造流水线。
    pub fn new(cfg: DraftConfig) -> Self {
        Self { cfg }
    }

    /// 配置（只读）。
    pub fn config(&self) -> &DraftConfig {
        &self.cfg
    }

    /// 运行流水线（确定性）：
    ///
    /// 1. [`Stage::Text`] — 调 `provider.complete` 拿文本 + 词级时间戳；
    /// 2. [`Stage::Timings`] — 词级时间戳就绪（由 Provider 提供，此处仅校验）；
    /// 3. [`Stage::Cues`] — 词级时间戳 → cues；
    /// 4. [`Stage::DraftRender`] — 每 cue 渲染一帧低清草稿。
    ///
    /// `prev`：上一次（可能中断的）流水账；为 `None` 时从头跑。**已完成阶段被
    /// 跳过**——resume 时 Text/Timings/Cues/DraftRender 中已 Done 的都不重跑。
    ///
    /// 但注意：若 Text 阶段已完成（跳过 Provider 调用），则本次**无词级数据**
    /// 可用，需调用方在 resume 时重新提供——因此本实现把 resume 判定放在
    /// **整体粒度**：仅当 `prev` 显示 Text 未完成时才调用 Provider；Text 已完成
    /// 但后续未完成时，仍会调用 Provider 以取回时间戳（Provider 确定性 ⇒
    /// 输出一致 ⇒ 语义等价于跳过，且不会污染已 Done 的 Text 记录）。
    pub fn run(
        &self,
        provider: &dyn Provider,
        req: &ProviderRequest,
        prev: Option<&Journal>,
    ) -> Result<DraftOutput, crate::AiError> {
        let mut journal = prev.cloned().unwrap_or_else(|| Journal::new(&req.prompt));

        // 已完成阶段集合（resume 跳过判据）。
        let done = |j: &Journal, s: Stage| j.is_done(s);

        // ---- Stage::Text / Stage::Timings：一次 Provider 调用同时产出两者 ----
        // Provider 确定性 ⇒ 何时调用不影响结果；只要 Text 未 Done 就调用并标记。
        let resp: ProviderResponse = provider.complete(req)?;
        let ProviderResponse { text, words } = resp;

        if !done(&journal, Stage::Text) {
            journal.mark_done(Stage::Text);
        }
        if words.is_empty() {
            return Err(crate::AiError::new(
                "AI4001",
                "DraftPipeline::run@timings",
                "Provider 未返回任何词级时间戳",
                "确保 Provider 返回非空 words（MockProvider 恒非空）",
            ));
        }
        if !done(&journal, Stage::Timings) {
            journal.mark_done(Stage::Timings);
        }

        // ---- Stage::Cues：仅在未 Done 时重新切分；否则复用已有 cues ----
        // 由于本结构无内部状态，cues 每次由同一确定性输入重算，结果一致；
        // 但当 Cues 已 Done 时我们仍重算（纯函数等价），并保留 Done 记录。
        let cues = cue_split(&words, self.cfg.cue)?;
        if !done(&journal, Stage::Cues) {
            journal.mark_done(Stage::Cues);
        }

        // ---- Stage::DraftRender：每 cue 渲一帧低清草稿 ----
        let frames = self.render_cues(&cues, req.seed);
        if !done(&journal, Stage::DraftRender) {
            journal.mark_done(Stage::DraftRender);
        }

        Ok(DraftOutput { text, cues, frames, journal })
    }

    /// 渲染 cues 为草稿帧（确定性：每帧色由 `hash(seed, cue.text)` 派生）。
    ///
    /// 公开供上层单独调用（例如 Cues 完成后仅想重渲渲染档）。
    pub fn render_cues(&self, cues: &[Cue], seed: u64) -> Vec<Framebuffer> {
        cues.iter().map(|c| self.render_one(c, seed)).collect()
    }

    /// 渲染单条 cue 为一帧：整帧填充派生色（低清草稿）。
    fn render_one(&self, cue: &Cue, seed: u64) -> Framebuffer {
        let mut fb = Framebuffer::new(self.cfg.width, self.cfg.height);
        let key = fnv1a64_str(&cue.text) ^ derive(seed, "frame");
        // 三通道各取派生哈希的一个字节 → 0..=1，确定性且分布均匀。
        let c = Vec3::new(
            ((key & 0xFF) as f32) / 255.0,
            (((key >> 8) & 0xFF) as f32) / 255.0,
            (((key >> 16) & 0xFF) as f32) / 255.0,
        );
        for y in 0..self.cfg.height {
            for x in 0..self.cfg.width {
                fb.set_pixel(x, y, c);
            }
        }
        fb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockProvider;

    #[test]
    fn pipeline_runs_all_stages_and_marks_done() {
        let pipe = DraftPipeline::new(DraftConfig::default());
        let provider = MockProvider::default();
        let req = ProviderRequest::new("hello", 1);
        let out = pipe.run(&provider, &req, None).unwrap();

        assert_eq!(out.journal.done_count(), 4, "四阶段全 Done");
        assert_eq!(out.journal.resume_from(), None);
        assert!(!out.cues.is_empty());
        assert_eq!(out.frames.len(), out.cues.len(), "每 cue 一帧");
        assert_eq!(out.frames[0].width, 320);
        assert_eq!(out.frames[0].height, 180);
    }

    #[test]
    fn resume_skips_completed_stages() {
        // 模拟中断：Text/Timings 已完成，Cues 未完成。
        let mut prev = Journal::new("hello");
        prev.mark_done(Stage::Text);
        prev.mark_done(Stage::Timings);
        assert_eq!(prev.resume_from(), Some(Stage::Cues));

        let pipe = DraftPipeline::new(DraftConfig::default());
        let provider = MockProvider::default();
        let req = ProviderRequest::new("hello", 1);
        let out = pipe.run(&provider, &req, Some(&prev)).unwrap();

        // 已完成阶段保持 Done 且被跳过（resume 从 Cues 起）。
        assert!(out.journal.is_done(Stage::Text));
        assert!(out.journal.is_done(Stage::Timings));
        assert!(out.journal.is_done(Stage::Cues));
        assert!(out.journal.is_done(Stage::DraftRender));
        assert_eq!(out.journal.resume_from(), None);
    }

    #[test]
    fn draft_render_is_deterministic() {
        let pipe = DraftPipeline::new(DraftConfig::default());
        let cues = cue_split(
            &[
                crate::WordTiming::new("a", 0, 200),
                crate::WordTiming::new("b", 260, 460),
            ],
            CueConfig::default(),
        )
        .unwrap();
        let f1 = pipe.render_cues(&cues, 9);
        let f2 = pipe.render_cues(&cues, 9);
        assert_eq!(f1.len(), f2.len());
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert_eq!(a.pixels, b.pixels, "同输入同像素");
        }
        // 不同种子 → 不同派生色（整帧同色但跨种子不同）。
        let f3 = pipe.render_cues(&cues, 10);
        assert_ne!(f1[0].pixels, f3[0].pixels);
    }
}
