//! 调度器：本地多线程执行 + 多机任务分发接口（P2）。
//!
//! # 本地执行：`LocalScheduler`
//!
//! [`LocalScheduler::run`] 用 [`std::thread::scope`] 起 scoped 线程，**按下标
//! round-robin 分派帧区间**，结果**按下标回填**（不是按完成顺序 append）——
//! 这个模式与 `prism-infra::batch` 的槽位回填同源，但本模块**不复用**
//! `batch::SubTask`：它的语义是"配置变体"（source/out/width/height/fps），与农场
//! "帧号区间"不同源，硬套会把两个语义层搅在一起（D-16 Mini Code：能写清楚比
//! 能复用更要紧）。
//!
//! # 确定性的三层保证
//!
//! 1. **切分确定**：[`crate::farm::split::split_frames`] 是输入规模的纯函数；
//! 2. **执行确定**：分派按**下标**（帧区间序号），与哪个线程抢到无关；
//! 3. **归位确定**：[`crate::farm::merge::merge_shards`] 按**帧号**（内容）而非
//!    完成顺序归位，然后走**唯一一次** [`prism_core::graph::reduce`]。
//!
//! 线程数、机器数、完成顺序**不出现在任何一个因子里** ⇒ 结果逐位一致。
//!
//! # 重试：固定次数，无随机退避
//!
//! [`LocalScheduler::run_shards_retry`] 按 [`RetryPolicy::max_attempts`] 重试，
//! **没有随机退避、没有时间抖动**。理由：重试的内容是同一个纯函数（帧号 → 值），
//! 所以重试 `k` 次与成功 1 次给出的值逐位相同——重试只影响"算了几遍"，不影响
//! "算出什么"。若引入随机退避，退避时长会成为环境函数，让**执行轨迹**（哪些帧
//! 被重算过几次）不可复现，那就违背了本任务"全链路确定性"的要求。
//!
//! # 多机：`plan_dispatch` + `FarmNode`
//!
//! 跨机本轮**只做接口与序列化，不真跑**（明确许可）。[`FarmNode`]
//! 描述一个节点（id + 容量 + 是否在线），[`plan_dispatch`] 把任务**按确定序**
//! 分配到节点上：
//!
//! - 节点按 `id` 升序遍历（**不是**按负载、不是按到达序——那不确定）；
//! - 任务按 [`crate::farm::task::TaskOrderKey`] 升序遍历（即队列的出队序）；
//! - 资源需求装不进节点容量则**跳过**该节点继续找（`ResourceNeed::fits`）；
//! - 每个任务落在**第一个**装得下的节点 ⇒ 分配是 `(任务集, 节点集)` 的纯函数。
//!
//! 传输层留给 [`crate::farm::FarmError::MalformedEnvelope`] 与
//! `prism-infra::farm_wire` 的线格式；本模块只产出**可序列化的分配计划**。

use crate::farm::merge::{FrameShard, FrameValue};
use crate::farm::queue::RetryPolicy;
use crate::farm::split::split_frames;
use crate::farm::task::{ConfigHash, FrameTask, ResourceNeed};
use crate::incremental::FrameRange;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// 一个农场节点（机器）的静态描述。本轮只做接口，不做真实传输。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FarmNode {
    /// 节点 id（内容派生：跨机对齐与日志追踪用）。
    pub id: String,
    /// 节点容量（0 = 不限）。
    pub capacity: ResourceNeed,
    /// 是否在线（离线节点不参与分配）。
    pub online: bool,
}

impl FarmNode {
    /// 以 id + 容量构造（在线）。
    pub fn new(id: impl Into<String>, capacity: ResourceNeed) -> FarmNode {
        FarmNode { id: id.into(), capacity, online: true }
    }

    /// 标记离线（链式）。
    pub fn offline(mut self) -> FarmNode {
        self.online = false;
        self
    }
}

/// 一个任务在某个节点上的执行结果：`T` 是**逐帧**部分值。
///
/// **刻意不提供"块内汇总值"字段**（见 [`crate::farm::merge`] 模块文档的反面
/// 纪律）：块内预归约会改变结合序，使浮点结果随切分漂移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardOutcome<T> {
    /// 执行该分片的节点 id。
    pub node_id: String,
    /// 分片覆盖的帧区间（= 分派计划里给它的那个，纯函数产物）。
    pub frames: FrameRange,
    /// 实际尝试次数（1 = 一次成功；> 1 = 发生过重试）。
    pub attempts: u32,
    /// 逐帧部分值，长度 = `frames.len()`。
    pub values: Vec<T>,
}

impl<T> ShardOutcome<T> {
    /// 以节点 id + 帧区间 + 逐帧值构造（`attempts = 1`）。
    pub fn new(node_id: impl Into<String>, frames: FrameRange, values: Vec<T>) -> ShardOutcome<T> {
        ShardOutcome { node_id: node_id.into(), frames, attempts: 1, values }
    }

    /// 记录实际尝试次数。
    pub fn with_attempts(mut self, attempts: u32) -> ShardOutcome<T> {
        self.attempts = attempts;
        self
    }

    /// 转成合并入口的输入（形状转换，**不做任何块内归约**）。
    pub fn to_shard(&self, config_hash: ConfigHash) -> FrameShard<T>
    where
        T: FrameValue,
    {
        FrameShard::new(self.frames, self.values.clone(), config_hash)
    }
}

/// 本地多线程调度器：帧级并行 + 固定重试 + 槽位回填。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalScheduler {
    /// 工作线程数（只影响吞吐，不影响结果；构造时归一为 ≥ 1）。
    pub workers: usize,
    /// 固定重试策略（构造时归一为 ≥ 1 次尝试）。
    pub retry: RetryPolicy,
}

impl LocalScheduler {
    /// 构造调度器（`workers` 归一为 ≥ 1；`retry` 归一为 ≥ 1 次尝试）。
    pub fn new(workers: usize, retry: RetryPolicy) -> LocalScheduler {
        LocalScheduler { workers: workers.max(1), retry: retry.normalized() }
    }

    /// 对闭区间 `[start, end]` 并行计算逐帧值，返回**按帧号升序**的序列。
    ///
    /// # 确定性
    /// 返回序列是帧号 `[start, end]` 的严格升序函数，**与 `workers` 无关**：
    /// 帧 `f` 的值只由 `f` 与 `render_frame` 决定。
    ///
    /// # Errors
    /// 区间非法 / 帧数溢出 → [`crate::farm::FarmError`]（P5001 / P5003）。
    pub fn run<F>(&self, start: u64, end: u64, render_frame: F) -> Result<Vec<u64>, crate::farm::FarmError>
    where
        F: Fn(u64) -> u64 + Send + Sync,
    {
        let parts: Vec<FrameRange> = split_frames(start, end, self.workers)?;
        let outcomes: Vec<ShardOutcome<u64>> =
            self.run_shards(&parts, "local", &render_frame);
        Ok(flatten_in_frame_order(&outcomes))
    }

    /// 对切好的帧区间并行计算，返回每个分片的逐帧值（**按下标升序**）。
    ///
    /// 与 [`LocalScheduler::run`] 的区别：不重新切分（切分由调用方给，可来自
    /// [`crate::farm::split::split_frames_shape_aligned`] 的形状对齐切分，
    /// 也可来自**远端机器**下发的区间）。返回序 = 分片输入序，**不是完成序**。
    pub fn run_shards<F>(
        &self,
        parts: &[FrameRange],
        node_id: &str,
        render_frame: &F,
    ) -> Vec<ShardOutcome<u64>>
    where
        F: Fn(u64) -> u64 + Send + Sync,
    {
        let infallible = |f: u64| Ok::<u64, String>(render_frame(f));
        self.dispatch(parts, node_id, &infallible)
    }

    /// **确定性重试**版：渲染可失败，失败后按 [`RetryPolicy::max_attempts`]
    /// 固定次数重试，**无随机退避、无时间抖动**。
    ///
    /// 用尽次数仍失败 ⇒ 该分片**不产出结果**（`None`），由调用方决定报错还是
    /// 重派。**绝不**用单位元、上一帧值或任何默认值悄悄补齐（C20 禁静默容错：
    /// 漏帧必须让上层看见）。
    pub fn run_shards_retry<F>(
        &self,
        parts: &[FrameRange],
        node_id: &str,
        render_frame: &F,
    ) -> Vec<Option<ShardOutcome<u64>>>
    where
        F: Fn(u64) -> Result<u64, String> + Send + Sync,
    {
        let max_attempts: u32 = self.retry.max_attempts.max(1);
        let mut out: Vec<Option<ShardOutcome<u64>>> = Vec::with_capacity(parts.len());
        for part in parts {
            out.push(render_with_retry(*part, node_id, max_attempts, render_frame));
        }
        out
    }

    /// 内部：分派 + 槽位回填（确定序按下标，不看完成顺序）。
    ///
    /// 单次尝试（无重试）；重试在 [`LocalScheduler::run_shards_retry`] 里串行
    /// 编排——因为"第几次尝试成功"是**该分片自己的**局部事实，与其它分片无关，
    /// 放进线程反而要额外同步。
    fn dispatch<F>(
        &self,
        parts: &[FrameRange],
        node_id: &str,
        render_frame: &F,
    ) -> Vec<ShardOutcome<u64>>
    where
        F: Fn(u64) -> Result<u64, String> + Send + Sync,
    {
        let workers: usize = self.workers;
        if parts.is_empty() {
            return Vec::new();
        }
        // 单分片或单线程：直接顺序算（无线程开销，且结果必然相同）。
        if workers <= 1 || parts.len() == 1 {
            return parts.iter().map(|p| compute_part(*p, node_id, render_frame)).collect();
        }
        // 槽位回填：第 i 个分片写第 i 个槽（与完成顺序无关 ⇒ 确定性）。
        let slots: Vec<Mutex<Option<ShardOutcome<u64>>>> =
            parts.iter().map(|_| Mutex::new(None)).collect();
        let slot_refs: Vec<&Mutex<Option<ShardOutcome<u64>>>> = slots.iter().collect();
        let part_refs: &[FrameRange] = parts;
        // `&Vec<...>` / `&[T]` 是 Copy：循环内 move 的是引用本身，不是容器，
        // 因此不会 E0382。
        let slots_ref: &Vec<&Mutex<Option<ShardOutcome<u64>>>> = &slot_refs;
        let parts_ref: &[FrameRange] = part_refs;
        std::thread::scope(|scope| {
            for w in 0..workers {
                let taken: Vec<usize> = (0..parts.len()).filter(|i| i % workers == w).collect();
                scope.spawn(move || {
                    for i in taken {
                        let Some(part) = parts_ref.get(i).copied() else { continue };
                        let value: ShardOutcome<u64> =
                            compute_part(part, node_id, render_frame);
                        // 锁中毒时留空：上层按下标升序收集，缺槽会显式重算
                        // （宁可多算一次，也不产出错误结果）。
                        if let Some(slot) = slots_ref.get(i) {
                            if let Ok(mut guard) = slot.lock() {
                                *guard = Some(value);
                            }
                        }
                    }
                });
            }
        });
        // 按下标升序收集（确定性）。锁中毒 / 未写入的槽在此同步重算。
        let mut out: Vec<ShardOutcome<u64>> = Vec::with_capacity(parts.len());
        for (i, slot) in slots.into_iter().enumerate() {
            match slot.into_inner().ok().flatten() {
                Some(v) => out.push(v),
                None => {
                    let part: FrameRange =
                        parts.get(i).copied().unwrap_or(FrameRange { start: 0, end: 0 });
                    out.push(compute_part(part, node_id, render_frame));
                }
            }
        }
        out
    }
}

impl Default for LocalScheduler {
    fn default() -> Self {
        LocalScheduler::new(1, RetryPolicy::none())
    }
}

/// 计算一个帧区间的逐帧值（`u64 → u64` 的纯映射）。
///
/// 渲染失败（`Err`）时返回 `values: Vec::new()`——**不静默补默认值**，由合并点
/// 的覆盖校验（P5005/P5006）显式报错。
fn compute_part<F>(part: FrameRange, node_id: &str, render_frame: &F) -> ShardOutcome<u64>
where
    F: Fn(u64) -> Result<u64, String> + Send + Sync,
{
    let mut values: Vec<u64> = Vec::new();
    let mut f: u64 = part.start;
    loop {
        match render_frame(f) {
            Ok(v) => values.push(v),
            Err(_) => return ShardOutcome::new(node_id, part, Vec::new()),
        }
        if f == part.end {
            break;
        }
        f = f.saturating_add(1);
    }
    ShardOutcome::new(node_id, part, values)
}

/// 单个分片的确定性重试：最多 `max_attempts` 次，**固定次数无退避**。
///
/// 返回 `None` = 用尽次数仍失败（调用方必须显式处理，C20 禁静默补齐）。
fn render_with_retry<F>(
    part: FrameRange,
    node_id: &str,
    max_attempts: u32,
    render_frame: &F,
) -> Option<ShardOutcome<u64>>
where
    F: Fn(u64) -> Result<u64, String> + Send + Sync,
{
    let attempts: u32 = max_attempts.max(1);
    for i in 0..attempts {
        let out: ShardOutcome<u64> = compute_part(part, node_id, render_frame);
        if out.values.len() as u64 == part.len() {
            return Some(out.with_attempts(u32::try_from(i).unwrap_or(u32::MAX).saturating_add(1)));
        }
        // 固定次数重试：没有 sleep、没有 jitter、没有 attempt 计数器差异。
    }
    None
}

/// 把若干分片结果**按帧号**摊平成升序序列（跨机合并的共用地板）。
///
/// 归位键是**帧号**（数据内容），不是分片到达序 / 完成序（律三"消序用
/// 排序"）：分片先按 `(帧起点, 帧终点)` 升序排列，再逐帧展开。分片集合不重不漏
/// 由切分保证（[`crate::farm::split::split_frames`]）。
pub fn flatten_in_frame_order<T: Clone>(outcomes: &[ShardOutcome<T>]) -> Vec<T> {
    // 消序用排序：按帧号（内容）而非完成顺序排列。
    let mut sorted: Vec<&ShardOutcome<T>> = outcomes.iter().collect();
    sorted.sort_by_key(|o| (o.frames.start, o.frames.end));
    let mut out: Vec<T> = Vec::new();
    for o in sorted {
        out.extend(o.values.iter().cloned());
    }
    out
}

/// 分发计划：任务 → 节点 id（**确定序**产物，可直接序列化下发）。
///
/// 结构是 `BTreeMap<节点 id, 任务列表>`：节点 id 升序、每个节点内任务按
/// [`crate::farm::task::TaskOrderKey`] 升序 ⇒ 计划文本逐字节确定。
pub type DispatchPlan = BTreeMap<String, Vec<FrameTask>>;

/// 把任务**按确定序**分配到节点。
///
/// # 确定性
/// 分配只依赖 `(任务集, 节点集, 资源需求)`，**不依赖**到达序、负载、网络
/// 时延。节点按 `id` 升序（`BTreeMap` 保证），任务按 [`crate::farm::task::TaskOrderKey`] 升序，
/// 每个任务落到**第一个**装得下的在线节点。
///
/// 离线节点与装不下任何任务的节点**不出现在结果里**（`BTreeMap` 无空值条目）。
/// 一个都装不下时该任务**不被分配**（不静默丢弃——调用方可对比计划与任务集
/// 检出漏派）。
pub fn plan_dispatch(tasks: &[FrameTask], nodes: &[FarmNode]) -> DispatchPlan {
    // 节点按 id 升序（去重：同 id 后者覆盖前者，id 相同即同一台机器）。
    let mut by_id: BTreeMap<&str, &FarmNode> = BTreeMap::new();
    for n in nodes {
        by_id.insert(n.id.as_str(), n);
    }
    // 任务按确定序升序（内容派生，非到达序）。
    let mut sorted: Vec<&FrameTask> = tasks.iter().collect();
    sorted.sort_by_key(|t| t.order_key());

    let mut plan: DispatchPlan = BTreeMap::new();
    for task in sorted {
        for (id, node) in &by_id {
            if !node.online {
                continue;
            }
            if !task.resources.fits(&node.capacity) {
                continue;
            }
            plan.entry((*id).to_string()).or_default().push(*task);
            break;
        }
    }
    plan
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::merge::merge_shards;
    use crate::farm::split::slot_counts;
    use crate::farm::split::slot_ranges;
    use crate::farm::split::split_frames;
    use crate::farm::split::split_frames_shape_aligned;
    use crate::farm::task::TaskOrderKey;
    use prism_core::graph::reduce;

    /// 确定性 LCG（Numerical Recipes 参数）——禁 `rand` crate（零第三方依赖）。
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        /// Fisher–Yates 洗牌（确定性）。
        fn shuffle<T>(&mut self, items: &mut [T]) {
            if items.len() < 2 {
                return;
            }
            for i in (1..items.len()).rev() {
                let j: usize = (self.next_u64() % (i as u64 + 1)) as usize;
                items.swap(i, j);
            }
        }
    }

    /// 纯整数"渲染"：帧号 → 载荷（splitmix64 风格，纯整数、确定性）。
    fn frame_value(f: u64) -> u64 {
        let mut z: u64 = f.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn serial(n: u64) -> Vec<u64> {
        (0..n).map(frame_value).collect()
    }

    #[test]
    fn local_scheduler_output_is_independent_of_worker_count() {
        for n in [1u64, 2, 3, 7, 8, 100, 1000, 1024] {
            let reference: Vec<u64> = serial(n);
            for workers in [1usize, 2, 3, 5, 8, 16] {
                let sched: LocalScheduler = LocalScheduler::new(workers, RetryPolicy::none());
                let got: Vec<u64> = sched.run(0, n - 1, frame_value).expect("合法区间应可调度");
                assert_eq!(
                    got, reference,
                    "n={n} workers={workers}：逐帧值必须与串行参考逐项相同（帧级是天然安全区）"
                );
            }
        }
    }

    #[test]
    fn run_shards_output_is_independent_of_worker_count_and_split_kind() {
        for n in [3u64, 7, 8, 100, 1000, 1024] {
            let reference: Vec<u64> = serial(n);
            for workers in [1usize, 3, 8, 17] {
                for aligned in [false, true] {
                    let parts: Vec<FrameRange> = if aligned {
                        split_frames_shape_aligned(0, n - 1, workers).expect("形状对齐切分应成功")
                    } else {
                        split_frames(0, n - 1, workers).expect("均分切分应成功")
                    };
                    let sched: LocalScheduler = LocalScheduler::new(workers, RetryPolicy::none());
                    let outcomes: Vec<ShardOutcome<u64>> =
                        sched.run_shards(&parts, "local", &frame_value);
                    assert_eq!(outcomes.len(), parts.len(), "每个分片都必须有结果");
                    // 按下标升序摊平（分片输入序 = 确定序）。
                    let flat: Vec<u64> = flatten_in_frame_order(&outcomes);
                    assert_eq!(
                        flat, reference,
                        "n={n} workers={workers} aligned={aligned}：摊平后必须逐项等于串行"
                    );
                }
            }
        }
    }

    #[test]
    fn worker_count_zero_and_malformed_range_are_structured_errors() {
        let sched: LocalScheduler = LocalScheduler::new(0, RetryPolicy::with_retries(0));
        assert_eq!(sched.workers, 1, "workers 必须归一为 ≥ 1");
        // 空区间 / 负区间。
        assert!(sched.run(5, 4, frame_value).is_err(), "end < start 必须报 P5001");
        // 帧数溢出（start=0, end=u64::MAX）。
        assert!(sched.run(0, u64::MAX, frame_value).is_err(), "帧数溢出必须报 P5003");
        // RetryPolicy 归一：0 次重试 ⇒ 仍 1 次尝试。
        assert_eq!(LocalScheduler::new(4, RetryPolicy { max_attempts: 0 }).retry.max_attempts, 1);
    }

    #[test]
    fn retry_is_fixed_count_and_reports_actual_attempts() {
        let parts: Vec<FrameRange> = split_frames(0, 9, 2).expect("可切分");

        // 一次成功 ⇒ attempts = 1。
        let ok_sched: LocalScheduler = LocalScheduler::new(2, RetryPolicy::with_retries(2));
        for o in ok_sched.run_shards(&parts, "n", &frame_value) {
            assert_eq!(o.attempts, 1, "无失败时不得重试");
            assert_eq!(o.node_id, "n");
        }

        // 前 2 次失败、第 3 次成功 ⇒ attempts = 3（固定次数，可复现）。
        let calls: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let flaky = |_f: u64| -> Result<u64, String> {
            let n: u32 = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < 2 { Err("注入的瞬时故障".to_string()) } else { Ok(42) }
        };
        let sched: LocalScheduler = LocalScheduler::new(2, RetryPolicy::with_retries(2));
        let retried: Vec<Option<ShardOutcome<u64>>> =
            sched.run_shards_retry(&parts, "n", &flaky);
        // 第一个分片耗尽 3 次（前 2 次失败）→ attempts = 3；第二个分片首次即成功。
        let first: Option<&ShardOutcome<u64>> = retried.first().and_then(|o| o.as_ref());
        assert_eq!(first.map(|o| o.attempts), Some(3), "前 2 次失败 ⇒ 第 3 次成功，attempts=3");
        assert!(retried.iter().all(|o| o.is_some()), "两个分片最终都应成功");

        // 永久失败 ⇒ 显式 None（不静默补默认值，C20）。
        let always_fail = |_f: u64| -> Result<u64, String> { Err("永久故障".to_string()) };
        let failed: Vec<Option<ShardOutcome<u64>>> =
            sched.run_shards_retry(&parts, "n", &always_fail);
        assert!(failed.iter().all(|o| o.is_none()), "永久失败必须返回 None，不得补默认值");
    }

    #[test]
    fn dispatch_plan_is_pure_function_of_tasks_and_nodes() {
        let cfg_a: ConfigHash = ConfigHash::of_bytes(b"cfg-a");
        let cfg_b: ConfigHash = ConfigHash::of_bytes(b"cfg-b");
        let tasks: Vec<FrameTask> = (0..12u64)
            .map(|i| FrameTask {
                frames: FrameRange { start: i * 10, end: i * 10 + 9 },
                config_hash: if i % 2 == 0 { cfg_a } else { cfg_b },
                priority: (i % 3) as u32,
                resources: ResourceNeed::unbounded(),
            })
            .collect();
        let nodes: Vec<FarmNode> = vec![
            FarmNode::new("node-c", ResourceNeed { cpu_cores: 4, memory_mb: 0, vram_mb: 0 }),
            FarmNode::new("node-a", ResourceNeed { cpu_cores: 4, memory_mb: 0, vram_mb: 0 }),
            FarmNode::new("node-b", ResourceNeed::unbounded()),
        ];
        let plan_a: DispatchPlan = plan_dispatch(&tasks, &nodes);
        // 乱序提交 + 乱序节点 + 多跑几遍 ⇒ 计划逐字段相同。
        let mut rng: Lcg = Lcg::new(0x5eed);
        for _ in 0..50 {
            let mut shuffled_tasks: Vec<FrameTask> = tasks.clone();
            let mut shuffled_nodes: Vec<FarmNode> = nodes.clone();
            rng.shuffle(&mut shuffled_tasks);
            rng.shuffle(&mut shuffled_nodes);
            let plan_b: DispatchPlan = plan_dispatch(&shuffled_tasks, &shuffled_nodes);
            assert_eq!(plan_a, plan_b, "分发计划必须是 (任务集, 节点集) 的纯函数");
        }
        // 节点 id 升序 ⇒ 第一个任务落 node-a（id 升序首个）。
        assert_eq!(plan_a.keys().next().map(|s| s.as_str()), Some("node-a"), "节点必须按 id 升序遍历");
        let all: usize = plan_a.values().map(|v| v.len()).sum();
        assert_eq!(all, tasks.len(), "容量不限时每个任务都必须被分配（不静默丢弃）");
        // 节点内任务按确定序升序。
        for list in plan_a.values() {
            let mut expect: Vec<TaskOrderKey> = list.iter().map(|t| t.order_key()).collect();
            let got: Vec<TaskOrderKey> = list.iter().map(|t| t.order_key()).collect();
            expect.sort();
            assert_eq!(got, expect, "节点内任务必须按确定序升序");
        }
    }

    #[test]
    fn dispatch_skips_offline_and_undersized_nodes() {
        let cfg: ConfigHash = ConfigHash::of_bytes(b"cfg");
        let need4: ResourceNeed = ResourceNeed { cpu_cores: 4, memory_mb: 0, vram_mb: 0 };
        let need8: ResourceNeed = ResourceNeed { cpu_cores: 8, memory_mb: 0, vram_mb: 0 };
        let tasks: Vec<FrameTask> = vec![
            FrameTask {
                frames: FrameRange { start: 0, end: 9 },
                config_hash: cfg,
                priority: 0,
                resources: need4,
            },
            FrameTask {
                frames: FrameRange { start: 10, end: 19 },
                config_hash: cfg,
                priority: 1,
                resources: need8,
            },
        ];
        let nodes: Vec<FarmNode> =
            vec![FarmNode::new("small", need4).offline(), FarmNode::new("mid", need4)];
        let plan: DispatchPlan = plan_dispatch(&tasks, &nodes);
        assert!(!plan.contains_key("small"), "离线节点不得出现在计划里");
        let mid: Vec<FrameTask> = plan.get("mid").cloned().unwrap_or_default();
        assert_eq!(mid.len(), 1, "只有 4 核需求的任务能装进 mid");
        assert_eq!(mid.first().map(|t| t.frames.start), Some(0));
        // 8 核任务装不进任何在线节点 ⇒ 不出现在计划里（调用方可对比任务集检出漏派）。
        assert_eq!(plan.values().flatten().count(), 1, "装不下的任务不被分配，也不静默丢弃到别处");
    }

    #[test]
    fn parallel_merge_of_scheduler_output_equals_serial_reference_bitwise() {
        // 端到端：切分 → 本地多线程 → 分片 → 乱序到达 → 合并归约 ⇒ 与串行逐位一致。
        for n in [1u64, 2, 3, 7, 8, 100, 1000, 1024] {
            let parts: Vec<FrameRange> = split_frames(0, n - 1, 8).expect("可切分");
            let sched: LocalScheduler = LocalScheduler::new(8, RetryPolicy::none());
            let outcomes: Vec<ShardOutcome<u64>> = sched.run_shards(&parts, "local", &frame_value);
            let cfg: ConfigHash = ConfigHash::of_bytes(b"end-to-end");
            let shards: Vec<FrameShard<u64>> =
                outcomes.iter().map(|o| o.to_shard(cfg)).collect();
            // 串行参考：全帧值走一次 reduce（唯一归约入口）。
            let serial_ref: u64 = reduce(serial(n), |a, b| a.wrapping_add(b), 0);
            // 并行：分片乱序到达后合并。
            let mut rng: Lcg = Lcg::new(0xC0FFEE);
            let mut shuffled: Vec<FrameShard<u64>> = shards.clone();
            rng.shuffle(&mut shuffled);
            let parallel: u64 =
                merge_shards(&shuffled, FrameRange { start: 0, end: n - 1 }, |a, b| a.wrapping_add(b), 0).expect("覆盖完整应可合并");
            assert_eq!(parallel, serial_ref, "n={n}：并行归约必须逐位等于串行参考");
        }
    }

    #[test]
    fn shape_aligned_blocks_are_reduction_tree_subtrees() {
        // 断言"块边界落在归约树槽边界上"——分布式同树的归纳基础。
        //
        // 精确的不变式：设切分选中的对齐深度为 d（最深且槽数 >= k 的层），则
        // **每个块边界都必须是深度 d 以及所有更浅深度（0..=d）的槽边界**。
        // 更浅层的槽边界集合是更深层的**超集**（浅层的合并边必然也是深层边界），
        // 所以"落在 0..=d 全部层的槽边界上"就是"落在对齐层的槽边界上"。
        // 反过来，**更深**层的槽更少、边界更少，块边界不必落在那里——断言它
        // 反而是错的（那是过度约束）。
        for n in [7u64, 8, 100, 1000, 1024] {
            for workers in [2usize, 3, 5, 8] {
                let blocks: Vec<FrameRange> =
                    split_frames_shape_aligned(0, n - 1, workers).expect("可切分");
                // 复算对齐深度：最深且 slot_counts[d] >= k 的层。
                let k: u64 = workers.min(n as usize) as u64;
                let counts: Vec<usize> = slot_counts(n);
                let mut align_depth: usize = 0;
                for (d, c) in counts.iter().enumerate() {
                    if *c as u64 >= k {
                        align_depth = d;
                    }
                }
                // 收集 0..=align_depth 全部层的槽边界。
                let mut boundaries: Vec<u64> = vec![0, n];
                for depth in 0..=align_depth {
                    for s in slot_ranges(n, depth).expect("槽可推导") {
                        boundaries.push(s.start);
                        boundaries.push(s.end.saturating_add(1));
                    }
                }
                boundaries.sort_unstable();
                boundaries.dedup();
                for b in &blocks {
                    for edge in [b.start, b.end.saturating_add(1)] {
                        assert!(
                            boundaries.contains(&edge),
                            "n={n} workers={workers} 对齐深度={align_depth}：\
                             块边界 {edge} 必须落在归约树槽边界上（块={:?} 全部层边界={boundaries:?}）",
                            (b.start, b.end)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn to_shard_carries_values_verbatim_without_local_reduction() {
        let outcome: ShardOutcome<u64> =
            ShardOutcome::new("n", FrameRange { start: 4, end: 7 }, vec![10, 20, 30, 40]);
        let cfg: ConfigHash = ConfigHash::of_bytes(b"x");
        let shard: FrameShard<u64> = outcome.to_shard(cfg);
        assert_eq!(shard.values, vec![10, 20, 30, 40], "逐帧值原样带上，不做块内归约");
        assert_eq!(shard.frames, FrameRange { start: 4, end: 7 });
        assert_eq!(shard.config_hash, cfg);
        assert_eq!(shard.values.len(), shard.frames.len() as usize, "契约：逐帧齐备");
    }

    #[test]
    fn flatten_sorts_by_frame_number_not_arrival_order() {
        // 值刻意不等于帧号（用 1000 + 帧号），以便断言"摊平后第 i 项对应帧 i"。
        let a: ShardOutcome<u64> = ShardOutcome::new(
            "n2",
            FrameRange { start: 50, end: 59 },
            (50..60u64).map(|f| 1000 + f).collect(),
        );
        let b: ShardOutcome<u64> = ShardOutcome::new(
            "n1",
            FrameRange { start: 0, end: 49 },
            (0..50u64).map(|f| 1000 + f).collect(),
        );
        // 故意让"后完成的"排在前面：摊平结果必须仍按帧号升序。
        let forward: Vec<u64> = flatten_in_frame_order(&[a.clone(), b.clone()]);
        let reversed: Vec<u64> = flatten_in_frame_order(&[b.clone(), a.clone()]);
        assert_eq!(forward, reversed, "分片到达顺序不得影响摊平结果");
        assert_eq!(forward.first().copied(), Some(1000), "摊平后必须从最小帧号开始");
        assert_eq!(forward.len(), 60);
        // 第 i 项必须恰好是帧 i 的值（值 = 1000 + 帧号）。
        for (i, v) in forward.iter().enumerate() {
            assert_eq!(*v, 1000 + i as u64, "摊平后第 {i} 项必须是帧 {i} 的值");
        }
    }
}
