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
//! 跨机当前版本**只做接口与序列化，不真跑**（明确许可）。[`FarmNode`]
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
use crate::farm::transport::{spawn_local_workers, FramePayloadShard, ShardTransport};
use crate::farm::worker::{render_frame_payload_cost, run_task_with_cost, FrameCache, WorkerStats};
use crate::incremental::FrameRange;
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;

/// 一个农场节点（机器）的静态描述。当前版本只做接口，不做真实传输。
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
// 并行逐帧载荷 + 断点续算 + 本地子进程分发
// ---------------------------------------------------------------------------

/// 续算版分片结果：分片产物 + 执行统计（复用帧数 / 真算帧数）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedShardOutcome {
    /// 分片产物（逐帧载荷，无块内汇总）。
    pub shard: FramePayloadShard,
    /// 执行统计。
    pub stats: WorkerStats,
}

impl LocalScheduler {
    /// **并行渲染逐帧载荷**（`workers` 个线程，槽位回填，结果按帧号升序）。
    ///
    /// 返回序列第 `i` 项恒为帧 `frames.start + i` 的载荷——**与 `workers` 无关**
    /// （帧级天然并行 + 槽位回填）。
    ///
    /// `work_iters` 是**渲染成本**（空转迭代数），只影响耗时、不影响载荷
    /// （见 [`crate::farm::worker::render_frame_payload_cost`]）——所以吞吐实验
    /// 与正确性判据可以同时成立。
    ///
    /// # Errors
    /// 区间非法 / 帧数溢出 → [`crate::farm::FarmError`]（P5001 / P5003）。
    pub fn run_payloads(
        &self,
        frames: FrameRange,
        config_hash: ConfigHash,
        work_iters: u32,
    ) -> Result<Vec<Vec<u8>>, crate::farm::FarmError> {
        let parts: Vec<FrameRange> = split_frames(frames.start, frames.end, self.workers)?;
        let render = |f: u64| render_frame_payload_cost(config_hash, f, work_iters);
        let workers: usize = self.workers;
        if parts.is_empty() || workers <= 1 || parts.len() == 1 {
            let mut out: Vec<Vec<u8>> = Vec::new();
            for p in &parts {
                let mut f: u64 = p.start;
                loop {
                    out.push(render(f));
                    if f == p.end {
                        break;
                    }
                    f = f.saturating_add(1);
                }
            }
            return Ok(out);
        }
        // 槽位回填：第 i 块写第 i 槽（与完成顺序无关 ⇒ 确定性）。
        let slots: Vec<Mutex<Option<Vec<Vec<u8>>>>> =
            parts.iter().map(|_| Mutex::new(None)).collect();
        let slots_ref: &Vec<Mutex<Option<Vec<Vec<u8>>>>> = &slots;
        let parts_ref: &[FrameRange] = &parts;
        let render_ref: &(dyn Fn(u64) -> Vec<u8> + Sync) = &render;
        std::thread::scope(|scope| {
            for w in 0..workers {
                let taken: Vec<usize> = (0..parts.len()).filter(|i| i % workers == w).collect();
                scope.spawn(move || {
                    for i in taken {
                        let Some(part) = parts_ref.get(i).copied() else { continue };
                        let mut local: Vec<Vec<u8>> = Vec::new();
                        let mut f: u64 = part.start;
                        loop {
                            local.push(render_ref(f));
                            if f == part.end {
                                break;
                            }
                            f = f.saturating_add(1);
                        }
                        if let Some(slot) = slots_ref.get(i) {
                            if let Ok(mut guard) = slot.lock() {
                                *guard = Some(local);
                            }
                        }
                    }
                });
            }
        });
        // 按下标升序收集（确定性）；缺槽在此同步重算（宁可多算，不出错值）。
        let mut out: Vec<Vec<u8>> = Vec::new();
        for (i, slot) in slots.into_iter().enumerate() {
            match slot.into_inner().ok().flatten() {
                Some(v) => out.extend(v),
                None => {
                    if let Some(part) = parts.get(i).copied() {
                        let mut f: u64 = part.start;
                        loop {
                            out.push(render(f));
                            if f == part.end {
                                break;
                            }
                            f = f.saturating_add(1);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// **断点续算版**分片渲染：命中 [`FrameCache`] 的帧**跳过不重算**。
    ///
    /// 顺序遍历分片（续算场景下重算率低，线程收益小于缓存查询开销；且顺序路径
    /// 让"哪帧复用、哪帧重算"可逐帧断言）。返回序 = 输入分片序（**不是**完成序）。
    ///
    /// 每个分片的 `stats` 直接暴露"复用几帧 / 真算几帧"——这就是断点续算的
    /// 可观测证据，而不是"感觉快了"。
    pub fn run_shards_cached(
        &self,
        parts: &[FrameRange],
        config_hash: ConfigHash,
        cache: &mut FrameCache,
    ) -> Vec<CachedShardOutcome> {
        let mut out: Vec<CachedShardOutcome> = Vec::with_capacity(parts.len());
        for part in parts {
            if part.is_empty() {
                out.push(CachedShardOutcome {
                    shard: FramePayloadShard::new("local", *part, config_hash, Vec::new()),
                    stats: WorkerStats::default(),
                });
                continue;
            }
            let task: FrameTask = FrameTask::new(*part, config_hash);
            let (mut shard, stats) = run_task_with_cost(&task, cache, 0);
            shard.node_id = "local".to_string();
            out.push(CachedShardOutcome { shard, stats });
        }
        out
    }
}

/// 一次本地子进程分发的报告（帧集合 + 吞吐证据）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchReport {
    /// 各 worker 回传的分片，**按帧号升序**（与完成顺序无关）。
    pub shards: Vec<FramePayloadShard>,
    /// 实际使用的 worker 进程数。
    pub workers: usize,
    /// 下发的任务数。
    pub tasks: usize,
    /// 总帧数（= 各分片帧数之和，已与覆盖校验对齐）。
    pub frames: u64,
}

impl DispatchReport {
    /// 帧集合的 MD5（按帧号顺序拼字节流）——与单机顺序渲染的判据**同一算法**。
    pub fn digest(&self) -> String {
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        for s in &self.shards {
            payloads.extend(s.payloads.iter().cloned());
        }
        crate::farm::evidence::manifest_digest(&payloads)
    }

    /// 参与过计算的节点 id（升序、去重；证据用）。
    pub fn node_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.shards.iter().map(|s| s.node_id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// 各分片的逐帧载荷按帧号摊平（供合并点使用）。
    pub fn payloads_in_frame_order(&self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for s in &self.shards {
            out.extend(s.payloads.iter().cloned());
        }
        out
    }
}

/// **任务分发到本地 worker 子进程**（本地子进程优先路径）。
///
/// 流程：`spawn N 个 farm-worker` → 按**下标 round-robin**交错下发/回收 →
/// `Done` 收工 → **按帧号归位** → 覆盖校验。返回序是帧号序，不是完成序。
///
/// # 确定性
/// 任务是"帧区间 + 配置哈希"，渲染是 `(配置, 帧)` 的纯函数，回收后按帧号排序
/// ⇒ worker 数、完成顺序、哪个进程抢到哪块**都不进结果**。
///
/// # Errors
/// - 找不到 / 起不动 worker 可执行文件 → P5009；
/// - worker 报错、帧被篡改 → P5007 / P5008；
/// - 分片未恰好覆盖任务帧集 → P5005。
pub fn dispatch_to_local_workers(
    tasks: &[FrameTask],
    workers: usize,
) -> Result<DispatchReport, crate::farm::FarmError> {
    dispatch_with_worker_cache(tasks, workers, None, None)
}

/// 同上，但每个 worker 一个**独立的断点续算缓存文件**（目录内 `farm-worker-<i>.cache`）。
///
/// 独立文件而非共享：并发写同一清单没有原子性保证（`CacheManifest` 是整文件
/// 替换格式），共享会互相覆盖。分片缓存各管一段帧，语义上与"每台机器自己的
/// 本地缓存"一致——真实农场也是这么做的。
pub fn dispatch_to_local_workers_with_cache(
    tasks: &[FrameTask],
    workers: usize,
    cache_dir: &Path,
) -> Result<DispatchReport, crate::farm::FarmError> {
    dispatch_with_worker_cache(tasks, workers, Some(cache_dir), None)
}

/// 影片分发变体（PE-1）：把 DSL 源码经 [`crate::farm::movie::MOVIE_SRC_ENV`]
/// 注入 worker 子进程，worker 渲染**真实帧像素**（而非 32 字节替身）。
///
/// 源码注入发生在进程级**分发互斥锁内**（`SUBPROCESS_DISPATCH_LOCK`，私有
/// 实现 `dispatch_with_worker_cache` 的锁说明），用完即删——绝不把环境变量
/// 泄漏到其它并发分发路径。
///
/// # Errors
/// 同 [`dispatch_to_local_workers`]。
pub fn dispatch_to_local_workers_movie(
    tasks: &[FrameTask],
    workers: usize,
    source: &str,
) -> Result<DispatchReport, crate::farm::FarmError> {
    dispatch_with_worker_cache(tasks, workers, None, Some(source))
}

/// 影片分发 + 断点续算缓存变体（源码注入与缓存目录同时生效）。
///
/// # Errors
/// 同 [`dispatch_to_local_workers_with_cache`]。
pub fn dispatch_to_local_workers_movie_cached(
    tasks: &[FrameTask],
    workers: usize,
    cache_dir: &Path,
    source: &str,
) -> Result<DispatchReport, crate::farm::FarmError> {
    dispatch_with_worker_cache(tasks, workers, Some(cache_dir), Some(source))
}

/// 子进程分发的进程级互斥锁。
///
/// 为什么需要：worker 的载荷模式由**进程级环境变量**（`PRISM_FARM_MOVIE_SRC`）
/// 选择，而环境变量是全局态——若两条分发路径并发 spawn worker，后者的
/// 子进程会读到前者的影片源码（或相反），载荷模式串味。锁把"源码注入 +
/// spawn + 收发 + 收工"收敛为临界区：任一刻至多一条分发现在置环境变量。
/// 进程内调度（`LocalScheduler`）不走环境变量，不受此锁约束。
static SUBPROCESS_DISPATCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// [`dispatch_to_local_workers`] / `_with_cache` / 影片变体的共同实现。
///
/// `movie_source = Some(src)` 时在锁内注入 / 清理影片源码环境变量。
fn dispatch_with_worker_cache(
    tasks: &[FrameTask],
    workers: usize,
    cache_dir: Option<&Path>,
    movie_source: Option<&str>,
) -> Result<DispatchReport, crate::farm::FarmError> {
    let location: &str = "farm::scheduler::dispatch_to_local_workers";
    // 互斥锁：中毒（持锁线程 panic）时不返回 Err——临界区只含 spawn/收发包，
    //  panic 时子进程由 Drop 回收，锁数据本身没被破坏；取 inner 继续用。
    let _dispatch_guard: std::sync::MutexGuard<'_, ()> =
        SUBPROCESS_DISPATCH_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if tasks.is_empty() {
        return Ok(DispatchReport { shards: Vec::new(), workers: workers.max(1), tasks: 0, frames: 0 });
    }
    let n: usize = workers.max(1).min(tasks.len());
    // 影片源码注入（锁内；失败也只是 spawn 出天然替身 worker——任务配置哈希
    // 与替身不匹配时 worker 会显式报 P5007，不会静默产错产物）。
    if let Some(src) = movie_source {
        std::env::set_var(crate::farm::movie::MOVIE_SRC_ENV, src);
    }
    let dispatched: Result<DispatchReport, crate::farm::FarmError> =
        dispatch_locked(tasks, n, cache_dir, location);
    if movie_source.is_some() {
        std::env::remove_var(crate::farm::movie::MOVIE_SRC_ENV);
    }
    dispatched
}

/// 锁内的实际分发逻辑（spawn → round-robin 收发 → 收工 → 归位校验）。
fn dispatch_locked(
    tasks: &[FrameTask],
    n: usize,
    cache_dir: Option<&Path>,
    location: &str,
) -> Result<DispatchReport, crate::farm::FarmError> {
    let mut transports = spawn_local_workers(n, cache_dir)?;

    // 下标 round-robin：确定的分派（与任务内容无关，只决定"谁算哪块"）。
    let mut queues: Vec<VecDeque<FrameTask>> = (0..n).map(|_| VecDeque::new()).collect();
    for (i, t) in tasks.iter().enumerate() {
        queues[i % n].push_back(*t);
    }

    // 交错轮次：每轮先给各 worker 发一个任务，再各收一个分片。
    // 这样既让 N 个进程真并行，又不会因为一方写满管道而互等（背压友好）。
    let mut shards: Vec<FramePayloadShard> = Vec::with_capacity(tasks.len());
    loop {
        let mut dispatched: bool = false;
        for w in 0..n {
            if let Some(task) = queues[w].front() {
                transports[w].send_task(task)?;
                dispatched = true;
            }
        }
        if !dispatched {
            break;
        }
        for w in 0..n {
            if queues[w].pop_front().is_some() {
                shards.push(transports[w].recv_shard()?);
            }
        }
    }
    for transport in transports.iter_mut() {
        transport.shutdown()?;
    }

    // 按帧号归位（律三"消序用排序"），再校验不重不漏。
    shards.sort_by_key(|s| (s.frames.start, s.frames.end));
    let frames: u64 = shards.iter().map(|s| s.frames.len()).sum();
    let mut cursor: u64 = shards.first().map(|s| s.frames.start).unwrap_or(0);
    for s in &shards {
        if s.frames.start != cursor {
            return Err(crate::farm::FarmError::FrameCoverageBroken {
                location: location.to_string(),
                frame: s.frames.start,
                expected: frames,
                got: s.payloads.len(),
            });
        }
        cursor = s.frames.end.saturating_add(1);
    }
    let expected_frames: u64 = tasks.iter().map(|t| t.frames.len()).sum();
    if frames != expected_frames {
        return Err(crate::farm::FarmError::FrameCoverageBroken {
            location: location.to_string(),
            frame: cursor,
            expected: expected_frames,
            got: shards.len(),
        });
    }
    Ok(DispatchReport { shards, workers: n, tasks: tasks.len(), frames })
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::evidence::manifest_digest;
    use crate::farm::merge::merge_shards;
    use crate::farm::worker::render_frame_payload;
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
        assert_eq!(mid.len(), 1, "只有 cpu_cores = 4 的任务能装进 mid");
        assert_eq!(mid.first().map(|t| t.frames.start), Some(0));
        // cpu_cores = 8 的任务装不进任何在线节点 ⇒ 不出现在计划里（调用方可对比任务集检出漏派）。
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

    // ---------------------------------------------------------------------
    // 传输 + worker + 断点续算
    // ---------------------------------------------------------------------

    /// 唯一临时缓存路径（避免测试间互相踩）。
    fn temp_cache(tag: &str) -> std::path::PathBuf {
        let mut p: std::path::PathBuf = std::env::temp_dir();
        p.push(format!("prism-farm-sched-{}-{tag}.cache", std::process::id()));
        p
    }

    /// 摊平分片的逐帧载荷（按帧号序 —— 分片已按帧号升序）。
    fn flatten_payloads(outcomes: &[CachedShardOutcome]) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for o in outcomes {
            out.extend(o.shard.payloads.iter().cloned());
        }
        out
    }

    #[test]
    fn parallel_payloads_equal_serial_bitwise_and_md5_identical() {
        // 验收线：N worker 并行渲染的帧集合与单机顺序渲染 **MD5 一致**。
        let cfg: ConfigHash = ConfigHash::of_bytes(b"parallel-md5");
        for n in [1u64, 2, 3, 7, 8, 33, 100, 1000, 1024] {
            let serial: Vec<Vec<u8>> = (0..n).map(|f| render_frame_payload(cfg, f)).collect();
            let serial_md5: String = manifest_digest(&serial);
            for workers in [1usize, 2, 3, 4, 8, 16] {
                let sched: LocalScheduler = LocalScheduler::new(workers, RetryPolicy::none());
                let parallel: Vec<Vec<u8>> = sched
                    .run_payloads(FrameRange { start: 0, end: n - 1 }, cfg, 0)
                    .expect("合法区间应可并行");
                assert_eq!(
                    parallel, serial,
                    "n={n} workers={workers}：逐帧载荷必须与顺序渲染逐字节相同"
                );
                assert_eq!(
                    manifest_digest(&parallel),
                    serial_md5,
                    "n={n} workers={workers}：MD5 必须一致（确定性合并）"
                );
            }
        }
        // 非零起点也要一致（帧号进载荷，不能把"第几块"当帧号）。
        let sched: LocalScheduler = LocalScheduler::new(3, RetryPolicy::none());
        let shifted: Vec<Vec<u8>> = sched
            .run_payloads(FrameRange { start: 500, end: 549 }, cfg, 0)
            .expect("非零起点应可并行");
        let reference: Vec<Vec<u8>> = (500..550u64).map(|f| render_frame_payload(cfg, f)).collect();
        assert_eq!(shifted, reference, "非零起点必须逐帧对齐（不是从 0 数）");
    }

    #[test]
    fn resume_after_interrupt_skips_completed_frames() {
        let path: std::path::PathBuf = temp_cache("resume");
        let _ = std::fs::remove_file(&path);
        let cfg: ConfigHash = ConfigHash::of_bytes(b"resume");
        let parts: Vec<FrameRange> = split_frames(0, 39, 4).expect("可切分"); // 4×10 帧
        let sched: LocalScheduler = LocalScheduler::new(4, RetryPolicy::none());

        // ---- 第一轮：只做前两块（10+10 帧），模拟中断 ----
        {
            let mut cache: FrameCache = FrameCache::open(&path).expect("首次打开");
            let first: Vec<CachedShardOutcome> = sched.run_shards_cached(&parts[..2], cfg, &mut cache);
            let rendered: u64 = first.iter().map(|o| o.stats.rendered).sum();
            assert_eq!(rendered, 20, "中断前只算了前 20 帧");
            cache.flush().expect("已完成帧必须落盘");
        } // 进程"死"在这里

        // ---- 第二轮：重启，从磁盘缓存派生完成集 ----
        let mut cache: FrameCache = FrameCache::open(&path).expect("重启应能读到缓存");
        assert_eq!(cache.len(), 20, "重启后缓存里有 20 帧");
        let mut queue: crate::farm::queue::FarmQueue = crate::farm::queue::FarmQueue::new();
        let seeded: usize =
            queue.seed_from_cache(cfg, &cache, FrameRange { start: 0, end: 39 });
        assert_eq!(seeded, 20, "完成集从缓存派生");

        let second: Vec<CachedShardOutcome> = sched.run_shards_cached(&parts, cfg, &mut cache);
        let reused: u64 = second.iter().map(|o| o.stats.reused).sum();
        let rendered: u64 = second.iter().map(|o| o.stats.rendered).sum();
        assert_eq!(reused, 20, "已完成的 20 帧必须被跳过");
        assert_eq!(rendered, 20, "只剩 20 帧需要真算（不是全量 40）");
        assert_eq!(second.iter().map(|o| o.stats.frames).sum::<u64>(), 40, "覆盖仍是全 40 帧");

        // ---- 与"一次算完"逐字节一致（续算不改变结果） ----
        let mut fresh: FrameCache = FrameCache::new();
        let reference: Vec<CachedShardOutcome> = sched.run_shards_cached(&parts, cfg, &mut fresh);
        assert_eq!(
            manifest_digest(&flatten_payloads(&second)),
            manifest_digest(&flatten_payloads(&reference)),
            "续算与全算的 MD5 必须一致"
        );
        assert_eq!(flatten_payloads(&second), flatten_payloads(&reference));

        // ---- 第三轮：全命中 ⇒ 零渲染（续算收敛） ----
        let third: Vec<CachedShardOutcome> = sched.run_shards_cached(&parts, cfg, &mut cache);
        assert!(
            third.iter().all(|o| o.stats.rendered == 0),
            "全部已完成后不得再渲染任何帧"
        );
        assert_eq!(third.iter().map(|o| o.stats.reused).sum::<u64>(), 40);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn local_worker_subprocess_end_to_end_single_and_multi_frame() {
        // 真起子进程：单帧任务 + 多帧多任务，产物必须与顺序渲染逐字节相同。
        //
        // 前置：`farm-worker` 可执行文件必须存在。`cargo test -p prism-render`
        // 会先构建 bin，故正常门禁下必然满足；仅当用 `--lib` 单独跑且从未构建过
        // bin 时会缺——此时**显式跳过并大声说明**（不伪装成功）。
        if crate::farm::transport::discover_worker_exe().is_none() {
            eprintln!(
                "[跳过] 未找到 {WORKER_BIN_NAME}：请先 `cargo build -p prism-render --bin {WORKER_BIN_NAME}`\
                 （`cargo test -p prism-render` 会自动构建），或设 PRISM_FARM_WORKER 指向它。\
                 进程内传输路径已由 transport/worker 单测覆盖。",
                WORKER_BIN_NAME = crate::farm::transport::WORKER_BIN_NAME
            );
            return;
        }
        let cfg: ConfigHash = ConfigHash::of_bytes(b"e2e-subprocess");

        // 单帧。
        let single: FrameTask = FrameTask::new(FrameRange::single(7), cfg);
        let report: DispatchReport =
            dispatch_to_local_workers(&[single], 1).expect("单 worker 单帧任务应跑通");
        assert_eq!(report.tasks, 1);
        assert_eq!(report.frames, 1, "单帧任务只产 1 帧");
        assert_eq!(report.shards.len(), 1);
        assert_eq!(
            report.payloads_in_frame_order(),
            vec![render_frame_payload(cfg, 7)],
            "子进程回传的载荷必须等于纯函数值"
        );
        assert_eq!(report.node_ids().len(), 1);

        // 多帧 + 多 worker。
        let tasks: Vec<FrameTask> = split_frames(0, 49, 4)
            .expect("可切分")
            .into_iter()
            .map(|r| FrameTask::new(r, cfg))
            .collect();
        let multi: DispatchReport =
            dispatch_to_local_workers(&tasks, 3).expect("三 worker 多任务应跑通");
        assert_eq!(multi.tasks, 4);
        assert_eq!(multi.frames, 50, "4 块共 50 帧，不重不漏");
        assert_eq!(multi.shards.len(), 4, "每个任务一个分片");
        let serial: Vec<Vec<u8>> = (0..50u64).map(|f| render_frame_payload(cfg, f)).collect();
        assert_eq!(
            multi.payloads_in_frame_order(),
            serial,
            "子进程并行的帧集合必须与单机顺序渲染逐字节相同"
        );
        assert_eq!(
            multi.digest(),
            manifest_digest(&serial),
            "MD5 必须一致（确定性合并）"
        );
        assert!(multi.node_ids().len() >= 2, "多进程应出现多个节点 id");

        // 空任务列表：不报错、不起进程、帧数为 0（幂等）。
        let empty: DispatchReport = dispatch_to_local_workers(&[], 4).expect("空任务不应报错");
        assert_eq!((empty.tasks, empty.frames), (0, 0));
        assert!(empty.shards.is_empty());

        // 本地子进程路径的吞吐实测（如实记录两种口径）：
        // 1) 零渲染成本：耗时 = 进程启动 + 协议往返 ⇒ 加速比 < 1（并行无收益，
        //    这正是"农场收益来自帧成本远大于传输成本"的反面证据）；
        // 2) 有渲染成本（PRISM_FARM_COST 注入空转，**不改载荷**）⇒ 加速比随
        //    worker 数上升，且 MD5 与单 worker 一致。
        let bench: Vec<FrameTask> = split_frames(0, 95, 8)
            .expect("可切分")
            .into_iter()
            .map(|r| FrameTask::new(r, cfg))
            .collect();
        let t0: std::time::Instant = std::time::Instant::now();
        let one: DispatchReport = dispatch_to_local_workers(&bench, 1).expect("单 worker 应跑通");
        let one_time: std::time::Duration = t0.elapsed();
        let t1: std::time::Instant = std::time::Instant::now();
        let four: DispatchReport = dispatch_to_local_workers(&bench, 4).expect("四 worker 应跑通");
        let four_time: std::time::Duration = t1.elapsed();
        assert_eq!(one.digest(), four.digest(), "子进程并行度不得改变 MD5");
        let speedup: f64 = if four_time.as_secs_f64() > 0.0 {
            one_time.as_secs_f64() / four_time.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "[farm 子进程吞吐·零成本] 96 帧 / 8 任务：1 worker {:?}；4 worker {:?}；加速比 {speedup:.2}×；MD5 {}",
            one_time, four_time, one.digest()
        );

        // ---- 有渲染成本（子进程继承 PRISM_FARM_COST）----
        std::env::set_var("PRISM_FARM_COST", "80000");
        let t2: std::time::Instant = std::time::Instant::now();
        let cost_one: DispatchReport =
            dispatch_to_local_workers(&bench, 1).expect("单 worker（有成本）应跑通");
        let cost_one_time: std::time::Duration = t2.elapsed();
        let t3: std::time::Instant = std::time::Instant::now();
        let cost_four: DispatchReport =
            dispatch_to_local_workers(&bench, 4).expect("四 worker（有成本）应跑通");
        let cost_four_time: std::time::Duration = t3.elapsed();
        std::env::remove_var("PRISM_FARM_COST");
        assert_eq!(
            cost_one.digest(),
            cost_four.digest(),
            "成本注入不得改变 MD5（成本只影响耗时）"
        );
        assert_eq!(cost_one.digest(), one.digest(), "有成本与零成本必须同 MD5");
        let cost_speedup: f64 = if cost_four_time.as_secs_f64() > 0.0 {
            cost_one_time.as_secs_f64() / cost_four_time.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "[farm 子进程吞吐·有成本 cost=80000] 96 帧 / 8 任务：1 worker {:?}；4 worker {:?}；加速比 {cost_speedup:.2}×；MD5 {}",
            cost_one_time, cost_four_time, cost_one.digest()
        );
    }

    #[test]
    fn throughput_speedup_is_measured_and_digest_is_unchanged() {
        // 吞吐实测：同一（成本非零的）作业，单线程 vs N 线程。
        // 断言的是"MD5 不变"（硬门槛）；加速比只打印记录（数值随机器波动，
        // 不做脆断言——那是把测试变成性能跑分）。
        const N: u64 = 96;
        const COST: u32 = 120_000;
        let cfg: ConfigHash = ConfigHash::of_bytes(b"bench");
        let frames: FrameRange = FrameRange { start: 0, end: N - 1 };

        let serial_sched: LocalScheduler = LocalScheduler::new(1, RetryPolicy::none());
        let t0: std::time::Instant = std::time::Instant::now();
        let serial: Vec<Vec<u8>> = serial_sched.run_payloads(frames, cfg, COST).expect("串行应成功");
        let serial_time: std::time::Duration = t0.elapsed();

        let parallel_sched: LocalScheduler = LocalScheduler::new(4, RetryPolicy::none());
        let t1: std::time::Instant = std::time::Instant::now();
        let parallel: Vec<Vec<u8>> =
            parallel_sched.run_payloads(frames, cfg, COST).expect("并行应成功");
        let parallel_time: std::time::Duration = t1.elapsed();

        assert_eq!(serial, parallel, "吞吐实验不得改变结果（成本不泄漏进载荷）");
        let serial_md5: String = manifest_digest(&serial);
        assert_eq!(manifest_digest(&parallel), serial_md5, "MD5 必须一致");
        let speedup: f64 = if parallel_time.as_secs_f64() > 0.0 {
            serial_time.as_secs_f64() / parallel_time.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "[farm 吞吐实测] 单机顺序 {N} 帧 × cost={COST}: {:?}；4 worker: {:?}；加速比 {speedup:.2}×；MD5 {}",
            serial_time, parallel_time, serial_md5
        );
        // 硬门槛只有确定性；加速比不设阈值（单核平台上并行反而可能更慢）。
        assert!(speedup > 0.0, "加速比必须可计算（时间非零）");
    }
}
