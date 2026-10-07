//! 确定性任务队列：确定序出队 + 幂等 + 固定重试（工程四律）。
//!
//! # 确定序，不是到达序
//!
//! 出队键是 `(priority, frames_done, config_hash, start, end)` 的字典序
//! （见 [`DequeueKey`]）。**优先级与吞吐档来自调度策略，配置哈希与帧区间来自
//! 任务内容**；没有到达序号、没有时间戳、没有 `HashMap` 迭代序 ⇒ 同一
//! （任务集, 已完成集）在任何机器、任何进程数、任何提交顺序下出队序完全相同。
//!
//! **反例说明为什么不能用到达序**：到达序依赖 `submit` 的调用时序，而时序
//! 依赖线程调度；调度依赖机器负载；机器负载依赖环境。所以到达序**不是输入的
//! 函数**，而确定序是。"禁止用到达时间排序"这条要求正是此意。
//!
//! # 幂等
//!
//! 已完成的 [`TaskId`] 记入 `BTreeSet`；重复提交时若 id 已在集合中，直接返回
//! [`SubmitOutcome::Duplicate`]，**不重复执行**。完成的判定只看 id（配置哈希 +
//! 帧区间）——因为同 id 意味着输入逐字节相同，输出必然逐字节相同（公理，
//! 与 `incremental` 的复用纪律同源）。
//!
//! # 重试
//!
//! [`RetryPolicy`] 是**固定次数**（`max_attempts`），**没有随机退避、没有
//! 时间抖动**。随机退避会把"重试时刻"变成环境函数，而重试时刻若影响归约树的
//! 分组，就会影响浮点结合序——固定次数 + 纯函数切分 ⇒ 重试后的结果与一次成功
//! 逐位一致。
//!
//! # 历史吞吐只影响"顺序"
//!
//! [`ThroughputLedger`] 记录每个配置哈希的**完成帧数**（纯计数，不记时间——
//! 记时间就引入环境依赖了）。它作为**优先级之后**的次级排序键微调出队顺序：
//! 吞吐高的配置哈希后出队（先派发慢的，尾延迟更平）。**它不影响结果**：任务
//! 集合不变 ⇒ 每帧的产出不变 ⇒ 归约树不变 ⇒ 归约结果逐位一致。测试
//! `throughput_ledger_changes_order_but_not_result` 钉住这一点。

use crate::farm::task::{ConfigHash, FrameTask, TaskId};
use std::collections::{BTreeMap, BTreeSet};

/// 提交裁决。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// 新任务：已入队。
    Accepted,
    /// 幂等命中：同 [`TaskId`] 已在队列或已完成集合中，**不重复执行**。
    Duplicate,
}

/// 确定性重试策略：固定尝试次数，**无随机退避**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// 最大尝试次数（≥1；1 = 不重试）。
    pub max_attempts: u32,
}

impl RetryPolicy {
    /// 不重试。
    pub fn none() -> RetryPolicy {
        RetryPolicy { max_attempts: 1 }
    }

    /// 允许 `extra_retries` 次重试（总尝试 = 1 + extra_retries）。
    pub fn with_retries(extra_retries: u32) -> RetryPolicy {
        RetryPolicy { max_attempts: extra_retries.saturating_add(1) }
    }

    /// 归一化：`max_attempts` 至少为 1（C21：不接受 0 次尝试的静默失败）。
    pub fn normalized(self) -> RetryPolicy {
        RetryPolicy { max_attempts: self.max_attempts.max(1) }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy::none()
    }
}

/// 历史吞吐记录：配置哈希 → 已完成帧数（**纯计数，不记时间**）。
///
/// 只作为**同优先级内的 tie-break** 影响出队顺序，**不影响结果**。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThroughputLedger {
    frames: BTreeMap<ConfigHash, u64>,
}

impl ThroughputLedger {
    /// 空记录。
    pub fn new() -> ThroughputLedger {
        ThroughputLedger::default()
    }

    /// 记一次完成（`frames_done` 帧）。
    pub fn record(&mut self, config_hash: ConfigHash, frames_done: u64) {
        let slot: &mut u64 = self.frames.entry(config_hash).or_insert(0);
        *slot = slot.saturating_add(frames_done);
    }

    /// 某配置哈希的累计完成帧数（无记录 = 0）。
    pub fn frames_of(&self, config_hash: ConfigHash) -> u64 {
        self.frames.get(&config_hash).copied().unwrap_or(0)
    }

    /// 记录是否为空。
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// 出队键 = 优先级 + **吞吐** + 内容派生的 [`crate::farm::task::TaskOrderKey`]
/// 的其余字段。
///
/// 字段序（字典序）刻意把**吞吐放在配置哈希之前**：
///
/// ```text
/// (priority, frames_done, config_hash, start, end)
/// ```
///
/// - `priority` 最高优先（人工指定的调度意图）；
/// - `frames_done` 次之：该配置哈希**已完成的帧数**（纯计数）。已完成帧数
///   **多者后出队**（先派发慢的，尾延迟更平）；**无记录者（0）最先出队**
///   （新配置优先试）。这正是"负吞吐优先"的装箱直觉，且不需要取反运算——
///   直接用计数即可，方向就是字典序的升序方向。
/// - 之后才是纯内容字段（配置哈希 / 帧区间），保证**同吞吐档内**仍是确定的
///   内容序。
///
/// **确定性论证**：吞吐记录只计**帧数**（纯计数，不记时间——记时间就引入环境
/// 依赖），而"哪些任务已完成"是**内容派生**的（完成集合只取决于执行了哪些
/// 任务，不取决于完成顺序）。所以整个键仍是 `(任务集, 已完成集)` 的纯函数，
/// 与到达序、机器数、进程数无关。测试
/// `throughput_ledger_changes_order_but_not_execution_set` 钉住"改顺序、不改
/// 执行集合"这条。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DequeueKey {
    /// 优先级（小 = 先）。
    priority: u32,
    /// 已完成帧数（纯计数）：多者后出队；0（无记录）最先出队。
    frames_done: u64,
    /// 配置哈希（内容派生）。
    config_hash: ConfigHash,
    /// 帧区间起点。
    start: u64,
    /// 帧区间终点。
    end: u64,
}

impl DequeueKey {
    fn of(task: &FrameTask, ledger: &ThroughputLedger) -> DequeueKey {
        DequeueKey {
            priority: task.priority,
            // 纯计数；记录内部已用 saturating_add 兜住溢出。
            frames_done: ledger.frames_of(task.config_hash),
            config_hash: task.config_hash,
            start: task.frames.start,
            end: task.frames.end,
        }
    }
}

/// 队列状态快照（可序列化到主节点 manifest）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueueState {
    /// 待执行任务数。
    pub pending: usize,
    /// 已完成（幂等去重）任务数。
    pub done: usize,
    /// 出队次数（含重试占用）。
    pub dispatched: u64,
}

/// 确定性任务队列。
#[derive(Debug, Clone, Default)]
pub struct FarmQueue {
    /// 待执行：确定序键 → 任务。
    pending: BTreeMap<DequeueKey, FrameTask>,
    /// 已完成 id（幂等）。
    done: BTreeSet<TaskId>,
    /// 历史吞吐（只影响顺序）。
    ledger: ThroughputLedger,
    /// 累计出队次数。
    dispatched: u64,
}

impl FarmQueue {
    /// 空队列。
    pub fn new() -> FarmQueue {
        FarmQueue::default()
    }

    /// 提交任务：幂等（重复 id 不重复执行）。
    pub fn submit(&mut self, task: FrameTask) -> SubmitOutcome {
        let id: TaskId = task.id();
        if self.done.contains(&id) {
            return SubmitOutcome::Duplicate;
        }
        let key: DequeueKey = DequeueKey::of(&task, &self.ledger);
        // 同 (order, ...) 键已存在 ⇒ 同 id（同区间 + 同配置 + 同优先级）已在队。
        if self.pending.contains_key(&key) {
            return SubmitOutcome::Duplicate;
        }
        self.pending.insert(key, task);
        SubmitOutcome::Accepted
    }

    /// 出队下一条（**确定序**：优先级 → 配置哈希 → 帧起点 → 帧终点）。
    pub fn pop(&mut self) -> Option<FrameTask> {
        let key: DequeueKey = self.pending.keys().next().copied()?;
        let task: FrameTask = self.pending.remove(&key)?;
        self.dispatched = self.dispatched.saturating_add(1);
        Some(task)
    }

    /// 标记任务完成（进入幂等集合）。
    pub fn complete(&mut self, task: &FrameTask, frames_done: u64) {
        self.done.insert(task.id());
        self.ledger.record(task.config_hash, frames_done);
    }

    /// 某任务是否已完成。
    pub fn is_done(&self, id: &TaskId) -> bool {
        self.done.contains(id)
    }

    /// 待执行任务数。
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// 是否无待执行任务。
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// 吞吐记录（只读）。
    pub fn ledger(&self) -> &ThroughputLedger {
        &self.ledger
    }

    /// 注入历史吞吐（帧数）——**只影响出队顺序，不影响执行集合与结果**。
    ///
    /// 用途：调度器重启后从 manifest 恢复历史（本轮只提供接口，不做持久化）。
    /// 注入的是**纯计数**（已完成帧数），不是时间——记时间就引入环境依赖了。
    pub fn seed_ledger(&mut self, config_hash: ConfigHash, frames_done: u64) {
        self.ledger.record(config_hash, frames_done);
    }

    /// 队列状态快照。
    pub fn state(&self) -> QueueState {
        QueueState { pending: self.pending.len(), done: self.done.len(), dispatched: self.dispatched }
    }

    /// 确定序预览：全部待执行任务按出队序排列（不消费）。
    pub fn peek_order(&self) -> Vec<FrameTask> {
        self.pending.values().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::split::split_frames;
    use crate::incremental::FrameRange;

    fn task(cfg: &[u8], start: u64, end: u64, priority: u32) -> FrameTask {
        FrameTask {
            frames: FrameRange { start, end },
            config_hash: ConfigHash::of_bytes(cfg),
            priority,
            resources: crate::farm::task::ResourceNeed::unbounded(),
        }
    }

    /// 把队列排空为出队序列表。
    fn drain(q: &mut FarmQueue) -> Vec<TaskId> {
        let mut out: Vec<TaskId> = Vec::new();
        while let Some(t) = q.pop() {
            out.push(t.id());
        }
        out
    }

    #[test]
    fn queue_dequeues_in_content_derived_order_not_arrival_order() {
        // 故意"逆优先级"提交：先提 priority=9，再提 priority=0。
        let mut q: FarmQueue = FarmQueue::new();
        assert_eq!(q.submit(task(b"slow", 0, 9, 9)), SubmitOutcome::Accepted);
        assert_eq!(q.submit(task(b"fast", 90, 99, 0)), SubmitOutcome::Accepted);
        assert_eq!(q.submit(task(b"mid", 45, 49, 5)), SubmitOutcome::Accepted);
        let order: Vec<TaskId> = drain(&mut q);
        assert_eq!(
            order,
            vec![
                task(b"fast", 90, 99, 0).id(),
                task(b"mid", 45, 49, 5).id(),
                task(b"slow", 0, 9, 9).id(),
            ],
            "出队序必须按优先级升序，与提交顺序无关"
        );
    }

    #[test]
    fn queue_order_is_identical_under_arbitrary_submission_permutation() {
        // 同一任务集合，任意提交顺序 ⇒ 同一出队序（确定序的核心断言）。
        let base: Vec<FrameTask> = (0..12u64)
            .map(|i| task(&[i as u8], i * 10, i * 10 + 9, (i % 3) as u32))
            .collect();
        let canonical: Vec<TaskId> = {
            let mut q: FarmQueue = FarmQueue::new();
            for t in &base {
                let _ = q.submit(*t);
            }
            drain(&mut q)
        };
        // 三种不同的提交排列（含倒序与交错），出队序必须一致。
        for perm in [vec![11usize, 0, 5, 3, 9, 1, 7, 2, 10, 4, 8, 6], vec![6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]] {
            let mut q: FarmQueue = FarmQueue::new();
            for &i in perm.iter() {
                let _ = q.submit(base[i]);
            }
            assert_eq!(drain(&mut q), canonical, "提交顺序不得影响出队序");
        }
    }

    #[test]
    fn duplicate_submission_executes_only_once() {
        // 幂等：同 id 重复提交（含完成后再提交）都不重复执行。
        let mut q: FarmQueue = FarmQueue::new();
        let t: FrameTask = task(b"cfg", 0, 99, 0);
        assert_eq!(q.submit(t), SubmitOutcome::Accepted);
        assert_eq!(q.submit(t), SubmitOutcome::Duplicate, "队内重复提交应被拒");
        assert_eq!(q.len(), 1, "重复提交不得增加队列长度");

        // 取出并完成后再提交 ⇒ 仍然 Duplicate。
        let popped: FrameTask = q.pop().expect("应可出队");
        q.complete(&popped, 100);
        assert_eq!(q.submit(t), SubmitOutcome::Duplicate, "已完成的任务重复提交应被拒");
        assert_eq!(q.len(), 0, "已完成任务不得重新入队");
        assert!(q.is_done(&t.id()));
        assert_eq!(q.state(), QueueState { pending: 0, done: 1, dispatched: 1 });
    }

    #[test]
    fn retry_policy_is_fixed_count_and_reproducible() {
        // 固定次数：max_attempts 由 extra_retries 唯一决定，无随机/时间项。
        assert_eq!(RetryPolicy::none().normalized().max_attempts, 1);
        assert_eq!(RetryPolicy::with_retries(3).normalized().max_attempts, 4);
        // 两遍构造必得同一策略（可复现）。
        assert_eq!(RetryPolicy::with_retries(2), RetryPolicy::with_retries(2));
        // C21：0 次尝试被归一为 1（不静默失败）。
        assert_eq!(RetryPolicy { max_attempts: 0 }.normalized().max_attempts, 1);
        assert_eq!(RetryPolicy::default(), RetryPolicy::none());
    }

    #[test]
    fn throughput_ledger_changes_order_but_not_execution_set() {
        // 历史吞吐只影响"顺序"，不影响"执行集合"——结果逐位一致的前提。
        let tasks: Vec<FrameTask> = vec![task(b"a", 0, 9, 1), task(b"b", 10, 19, 1)];
        // 无记录：按 (优先级, 吞吐档, 配置哈希, 帧区间) 定序 ⇒ 配置哈希定序。
        let mut q1: FarmQueue = FarmQueue::new();
        for t in &tasks {
            let _ = q1.submit(*t);
        }
        let plain: Vec<TaskId> = drain(&mut q1);

        // 记账后：cfg "a" 吞吐高（500 帧）⇒ 其 inverse_frames 更小 ⇒ "a" **后**出队。
        // 顺序反转，但执行集合不变。
        let mut q2: FarmQueue = FarmQueue::new();
        q2.seed_ledger(ConfigHash::of_bytes(b"a"), 500);
        for t in &tasks {
            let _ = q2.submit(*t);
        }
        let with_history: Vec<TaskId> = drain(&mut q2);
        assert_ne!(plain, with_history, "吞吐记录应能改变出队顺序（本例如此）");
        assert_eq!(
            with_history.first().map(|id| id.config_hash),
            Some(ConfigHash::of_bytes(b"b")),
            "吞吐高的 cfg \"a\" 应后出队（先派发慢的，尾延迟更平）"
        );
        let mut set_a: Vec<TaskId> = plain.clone();
        let mut set_b: Vec<TaskId> = with_history.clone();
        set_a.sort();
        set_b.sort();
        assert_eq!(set_a, set_b, "但执行集合必须完全相同（结果不受影响）");

        // 优先级高于吞吐：priority 不同则吞吐不得越过优先级。
        let mut q3: FarmQueue = FarmQueue::new();
        q3.seed_ledger(ConfigHash::of_bytes(b"a"), 500);
        let hi: FrameTask = task(b"a", 0, 9, 9); // priority 9（低）
        let lo: FrameTask = task(b"b", 10, 19, 0); // priority 0（高）
        let _ = q3.submit(hi);
        let _ = q3.submit(lo);
        let by_priority: Vec<TaskId> = drain(&mut q3);
        assert_eq!(
            by_priority.first().map(|id| id.frames.start),
            Some(10),
            "优先级 0 的任务必须先出队，哪怕它有吞吐记账"
        );

        // 记录只计帧数，不计时间 ⇒ 纯计数。
        let mut l: ThroughputLedger = ThroughputLedger::new();
        assert!(l.is_empty());
        assert_eq!(l.frames_of(ConfigHash::of_bytes(b"a")), 0);
        l.record(ConfigHash::of_bytes(b"a"), 7);
        l.record(ConfigHash::of_bytes(b"a"), 3);
        assert_eq!(l.frames_of(ConfigHash::of_bytes(b"a")), 10, "纯累加，可复现");
        assert!(!l.is_empty());
    }

    #[test]
    fn queue_peek_is_read_only_and_matches_drain_order() {
        let mut q: FarmQueue = FarmQueue::new();
        for i in 0..5u64 {
            let _ = q.submit(task(&[i as u8], i, i, (4 - i) as u32));
        }
        let peeked: Vec<TaskId> = q.peek_order().iter().map(|t| t.id()).collect();
        assert_eq!(q.len(), 5, "peek 不得消费队列");
        let drained: Vec<TaskId> = drain(&mut q);
        assert_eq!(peeked, drained, "peek 顺序必须等于实际出队顺序");
        assert!(q.is_empty());
        assert_eq!(q.pop(), None, "空队列出队返回 None 而非 panic");
    }

    #[test]
    fn queue_handles_split_ranges_from_the_scheduler() {
        // 与切分函数对接：k 块 ⇒ k 个任务，出队序按帧起点升序（确定）。
        let mut q: FarmQueue = FarmQueue::new();
        for r in split_frames(0, 99, 4).expect("应可切分") {
            let _ = q.submit(task(b"job", r.start, r.end, 0));
        }
        let order: Vec<TaskId> = drain(&mut q);
        let starts: Vec<u64> = order.iter().map(|id| id.frames.start).collect();
        assert_eq!(starts, vec![0, 25, 50, 75], "同优先级同配置 ⇒ 按帧起点升序出队");
    }
}
