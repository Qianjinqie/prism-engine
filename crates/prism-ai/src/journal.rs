//! 任务流水账 + resume（B10；路线图 Phase 2 验收第 6 条）。
//!
//! # 语义
//!
//! AI 草稿流水线按**阶段**（[`Stage`]）串行推进；每完成一个阶段就把该阶段
//! 标为 [`StageState::Done`] 存入 [`Journal`]。若中途中断（进程退出/错误），
//! 重启时 `resume_from` 返回**首个未完成阶段**，因此已完成阶段被**跳过**，
//! 不重复执行——这是 Phase 2 验收第 6 条「ai-run mock 中断 resume 跳过已完成
//! 阶段」的机制实现。
//!
//! # 确定性
//!
//! 流水账是纯数据结构（阶段 → 状态映射 + 记录序）；序列化用自实现的确定性
//! 文本格式（无第三方、无 HashMap 迭代序依赖）。

use std::collections::BTreeMap;

use crate::AiError;

/// 流水线阶段（固定顺序 = 声明序）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// 生成文本（Provider 调用）。
    Text,
    /// 生成词级时间戳（承接 Text）。
    Timings,
    /// 词级时间戳 → cues（字幕切分）。
    Cues,
    /// 草稿渲染（接引擎渲染，低清档位）。
    DraftRender,
}

impl Stage {
    /// 全阶段固定顺序（resume 按此序找首个未完成阶段）。
    pub const ALL: [Stage; 4] = [Stage::Text, Stage::Timings, Stage::Cues, Stage::DraftRender];

    /// 阶段名（确定性字符串，序列化/诊断用）。
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Text => "text",
            Stage::Timings => "timings",
            Stage::Cues => "cues",
            Stage::DraftRender => "draft_render",
        }
    }

    /// 由名字解析阶段（读档路径；未知名字返回 `None`，调用方结构化报错）。
    pub fn from_name(name: &str) -> Option<Stage> {
        match name {
            "text" => Some(Stage::Text),
            "timings" => Some(Stage::Timings),
            "cues" => Some(Stage::Cues),
            "draft_render" => Some(Stage::DraftRender),
            _ => None,
        }
    }
}

/// 阶段状态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageState {
    /// 未开始。
    Pending,
    /// 已完成（resume 时被跳过）。
    Done,
    /// 失败（含确定性三段式原因摘要）。
    Failed(String),
}

/// 单条阶段记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRecord {
    /// 阶段。
    pub stage: Stage,
    /// 状态。
    pub state: StageState,
}

/// 任务流水账（阶段 → 状态 + 有序记录）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Journal {
    /// 阶段状态映射（BTreeMap：迭代序确定，序列化可复现）。
    states: BTreeMap<Stage, StageState>,
    /// 有序记录（首次置位的顺序，诊断/审计用）。
    records: Vec<StageRecord>,
    /// 任务标识（诊断用；不参与 resume 逻辑）。
    task: String,
}

impl Journal {
    /// 新建空流水账（`task` 为任务标识）。
    pub fn new(task: impl Into<String>) -> Self {
        Self { states: BTreeMap::new(), records: Vec::new(), task: task.into() }
    }

    /// 任务标识。
    pub fn task(&self) -> &str {
        &self.task
    }

    /// 标记阶段完成（幂等：重复标记覆盖为 Done，并首次追加记录）。
    pub fn mark_done(&mut self, stage: Stage) {
        let first = !self.states.contains_key(&stage);
        self.states.insert(stage, StageState::Done);
        if first {
            self.records.push(StageRecord { stage, state: StageState::Done });
        }
    }

    /// 标记阶段失败（原因摘要为确定性文本）。
    pub fn mark_failed(&mut self, stage: Stage, reason: impl Into<String>) {
        let st = StageState::Failed(reason.into());
        let first = !self.states.contains_key(&stage);
        self.states.insert(stage, st.clone());
        if first {
            self.records.push(StageRecord { stage, state: st });
        }
    }

    /// 查询阶段状态（未记录 = [`StageState::Pending`]）。
    pub fn state(&self, stage: Stage) -> StageState {
        self.states.get(&stage).cloned().unwrap_or(StageState::Pending)
    }

    /// 阶段是否已完成（resume 跳过判据）。
    pub fn is_done(&self, stage: Stage) -> bool {
        matches!(self.state(stage), StageState::Done)
    }

    /// resume 起点：返回**首个未完成阶段**（`Stage::ALL` 顺序）。
    ///
    /// 全部完成返回 `None`（任务已完成，无需 resume）。
    pub fn resume_from(&self) -> Option<Stage> {
        Stage::ALL.into_iter().find(|s| !self.is_done(*s))
    }

    /// 完成阶段数。
    pub fn done_count(&self) -> usize {
        self.states.values().filter(|s| matches!(s, StageState::Done)).count()
    }

    /// 序列化为确定性文本（每行 `stage=state[:reason]`，按 `Stage::ALL` 序）。
    ///
    /// 状态字：`pending` / `done` / `failed:原因`（原因中的换行被替换为空格，
    /// 保证单行可解析）。
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("task={}\n", sanitize(&self.task)));
        for stage in Stage::ALL {
            let state = match self.state(stage) {
                StageState::Pending => "pending".to_string(),
                StageState::Done => "done".to_string(),
                StageState::Failed(r) => format!("failed:{}", sanitize(&r)),
            };
            out.push_str(&format!("{}={}\n", stage.name(), state));
        }
        out
    }

    /// 反序列化（读档路径）。格式非法返回三段式错误（`AI3001`）。
    pub fn deserialize(text: &str) -> Result<Journal, AiError> {
        let mut task = String::new();
        let mut j = Journal::default();
        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (k, v) = line.split_once('=').ok_or_else(|| {
                AiError::new(
                    "AI3001",
                    format!("Journal::deserialize 行 {}", lineno + 1),
                    format!("缺少 `=` 分隔：`{}`", line),
                    "每行形如 `stage=state`（stage ∈ text/timings/cues/draft_render）",
                )
            })?;
            if k == "task" {
                task = v.to_string();
                continue;
            }
            let stage = Stage::from_name(k).ok_or_else(|| {
                AiError::new(
                    "AI3001",
                    format!("Journal::deserialize 行 {}", lineno + 1),
                    format!("未知阶段名 `{}`", k),
                    "stage ∈ text / timings / cues / draft_render",
                )
            })?;
            let state = if v == "pending" {
                StageState::Pending
            } else if v == "done" {
                StageState::Done
            } else if let Some(reason) = v.strip_prefix("failed:") {
                StageState::Failed(reason.to_string())
            } else {
                return Err(AiError::new(
                    "AI3001",
                    format!("Journal::deserialize 行 {}", lineno + 1),
                    format!("未知状态 `{}`", v),
                    "state ∈ pending / done / failed:<原因>",
                ));
            };
            j.states.insert(stage, state.clone());
            j.records.push(StageRecord { stage, state });
        }
        j.task = task;
        Ok(j)
    }
}

/// 把多行文本压成单行（序列化安全）。
fn sanitize(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_skips_completed_stages() {
        // 模拟中断：Text 已完成、Timings 失败后退、后两阶段未动。
        let mut j = Journal::new("demo");
        j.mark_done(Stage::Text);
        assert_eq!(j.resume_from(), Some(Stage::Timings), "应从未完成的 timings 起");

        j.mark_done(Stage::Timings);
        assert_eq!(j.resume_from(), Some(Stage::Cues));

        j.mark_done(Stage::Cues);
        j.mark_done(Stage::DraftRender);
        assert_eq!(j.resume_from(), None, "全部完成无需 resume");
        assert_eq!(j.done_count(), 4);
    }

    #[test]
    fn failed_stage_is_not_done_and_resumes_there() {
        let mut j = Journal::new("t");
        j.mark_done(Stage::Text);
        j.mark_failed(Stage::Timings, "mock 词序列倒退");
        assert!(!j.is_done(Stage::Timings));
        assert_eq!(j.resume_from(), Some(Stage::Timings));
    }

    #[test]
    fn serialize_roundtrip_is_deterministic() {
        let mut j = Journal::new("task-1");
        j.mark_done(Stage::Text);
        j.mark_done(Stage::Timings);
        j.mark_failed(Stage::Cues, "切分失败\n第二行");
        let s1 = j.serialize();
        let s2 = j.serialize();
        assert_eq!(s1, s2, "序列化确定性");
        assert!(!s1.contains('\n') || s1.lines().count() == 5, "task + 4 阶段行");

        let back = Journal::deserialize(&s1).unwrap();
        assert_eq!(back.task(), "task-1");
        assert_eq!(back.state(Stage::Text), StageState::Done);
        assert_eq!(back.resume_from(), Some(Stage::Cues), "失败阶段应从该阶段续跑");
    }

    #[test]
    fn malformed_serialization_is_rejected() {
        assert_eq!(Journal::deserialize("garbage").unwrap_err().code, "AI3001");
        assert_eq!(Journal::deserialize("nope=done").unwrap_err().code, "AI3001");
        assert_eq!(Journal::deserialize("text=weird").unwrap_err().code, "AI3001");
    }
}
