//! 农场任务描述：帧号区间 + 配置哈希 + 优先级 + 资源需求
//! （路线图 §2.6 P2 设计："任务 = 帧号区间 + 配置哈希"）。
//!
//! # 幂等键
//!
//! [`TaskId`] = (配置哈希, 帧区间)。**同 id 重复提交不重复执行**——这是农场
//! 幂等的唯一依据：重试、网络重发、调度器重复入队都收敛到同一 id。
//!
//! # 确定序
//!
//! [`TaskOrderKey`] = (优先级, 配置哈希, 帧起点, 帧终点) 的字典序。**不含
//! 到达时间**——到达序不确定，确定序必须只由任务自身的字段决定（架构 §4.4
//! 求值顺序确定性的推广：消序用排序，且排序键必须是内容而非时序）。

use crate::incremental::FrameRange;

/// 配置哈希：渲染配置的内容指纹（路线图 §2.6"任务 = 帧号区间 + 配置哈希"）。
///
/// 由配置字节串单遍 FNV-1a 64 位折叠而来——**纯整数运算、零依赖、确定性**
/// （同配置字节 ⇒ 同哈希；不同配置 ⇒ 哈希碰撞概率可忽略但**不作为正确性
/// 依据**：正确性只依赖 [`TaskId`] 的区间相等）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ConfigHash(u64);

impl ConfigHash {
    /// FNV-1a 64 位偏移基（零依赖、确定性）。
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    /// FNV-1a 64 位素数。
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// 由配置字节串计算配置哈希（纯函数）。
    pub fn of_bytes(bytes: &[u8]) -> ConfigHash {
        let mut h: u64 = ConfigHash::OFFSET;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(ConfigHash::PRIME);
        }
        ConfigHash(h)
    }

    /// 裸哈希值。
    pub fn value(self) -> u64 {
        self.0
    }

    /// 从裸值构造（供线格式解码用）。
    pub fn from_raw(v: u64) -> ConfigHash {
        ConfigHash(v)
    }
}

/// 资源需求：worker 申报的硬件占用（用于装箱与准入，不影响结果）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ResourceNeed {
    /// CPU 核数需求（0 = 不限）。
    pub cpu_cores: u32,
    /// 内存需求（MiB；0 = 不限）。
    pub memory_mb: u32,
    /// 显存需求（MiB；0 = 不限）。
    pub vram_mb: u32,
}

impl ResourceNeed {
    /// 全部不限。
    pub fn unbounded() -> ResourceNeed {
        ResourceNeed { cpu_cores: 0, memory_mb: 0, vram_mb: 0 }
    }

    /// 资源需求是否被某节点容量满足（0 视为不限，故恒满足）。
    pub fn fits(&self, capacity: &ResourceNeed) -> bool {
        let ok = |need: u32, have: u32| need == 0 || (have != 0 && need <= have);
        ok(self.cpu_cores, capacity.cpu_cores)
            && ok(self.memory_mb, capacity.memory_mb)
            && ok(self.vram_mb, capacity.vram_mb)
    }
}

/// 农场任务：帧号区间 + 配置哈希 + 优先级 + 资源需求。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTask {
    /// 帧号闭区间。
    pub frames: FrameRange,
    /// 配置哈希（同配置的任务产物可互相复用）。
    pub config_hash: ConfigHash,
    /// 优先级（**小 = 优先级高**，先出队）。
    pub priority: u32,
    /// 资源需求。
    pub resources: ResourceNeed,
}

impl FrameTask {
    /// 以帧区间 + 配置哈希构造（优先级 0、资源不限）。
    pub fn new(frames: FrameRange, config_hash: ConfigHash) -> FrameTask {
        FrameTask { frames, config_hash, priority: 0, resources: ResourceNeed::unbounded() }
    }

    /// 幂等键：(配置哈希, 帧区间)。
    pub fn id(&self) -> TaskId {
        TaskId { config_hash: self.config_hash, frames: self.frames }
    }

    /// 确定序排序键（**内容派生，不含到达时间**）。
    pub fn order_key(&self) -> TaskOrderKey {
        TaskOrderKey {
            priority: self.priority,
            config_hash: self.config_hash,
            start: self.frames.start,
            end: self.frames.end,
        }
    }
}

/// 任务幂等键：(配置哈希, 帧区间)。同 id 重复提交不重复执行。
///
/// **手写 `Ord` / `Hash`（不用 derive）**：[`FrameRange`] 本身只 derive 了
/// `PartialEq, Eq`，没有 `Ord` / `Hash`，而本类型要进 `BTreeSet`（幂等去重）
/// 与 `BTreeMap` 键。任务书要求"`incremental.rs` 原则上不碰"，所以这里
/// **不回去给 `FrameRange` 加 derive**，而是在本类型上按 `(config_hash,
/// start, end)` 自行实现——语义与 derive 完全一致（`FrameRange` 是
/// `start` 在前 `end` 在后的结构体，字段序即字典序），但零侵入。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskId {
    /// 配置哈希。
    pub config_hash: ConfigHash,
    /// 帧号闭区间。
    pub frames: FrameRange,
}

impl Ord for TaskId {
    /// 字典序：先配置哈希，再帧起点，再帧终点（与结构体字段序一致）。
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.config_hash
            .cmp(&other.config_hash)
            .then_with(|| self.frames.start.cmp(&other.frames.start))
            .then_with(|| self.frames.end.cmp(&other.frames.end))
    }
}

impl PartialOrd for TaskId {
    /// 与 [`Ord::cmp`] 一致（`Eq` 已保证自反性）。
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for TaskId {
    /// 与 [`Ord::cmp`] 的比较字段严格一致（`config_hash`, `start`, `end`）。
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.config_hash.hash(state);
        self.frames.start.hash(state);
        self.frames.end.hash(state);
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 显示形式稳定（便于日志与跨机对齐）：cfg=<hex>/frames=<start>-<end>。
        write!(f, "cfg={:016x}/frames={}-{}", self.config_hash.value(), self.frames.start, self.frames.end)
    }
}

/// 任务确定序键：(优先级, 配置哈希, 帧起点, 帧终点)。
///
/// 字典序即出队序。**四个字段全部来自任务内容**——没有到达序号、没有时间戳、
/// 没有 `HashMap` 迭代序，所以同一任务集合的出队序在任何机器、任何进程、
/// 任何提交顺序下都相同。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskOrderKey {
    /// 优先级（小 = 先）。
    pub priority: u32,
    /// 配置哈希。
    pub config_hash: ConfigHash,
    /// 帧区间起点。
    pub start: u64,
    /// 帧区间终点。
    pub end: u64,
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_hash_is_deterministic_pure_function_of_bytes() {
        // 纯函数：同字节 → 同哈希（两遍一致）。
        let a: ConfigHash = ConfigHash::of_bytes(b"width=1920;height=1080;fps=30");
        let b: ConfigHash = ConfigHash::of_bytes(b"width=1920;height=1080;fps=30");
        assert_eq!(a, b, "同配置字节必得同哈希");
        // 内容不同 → 哈希不同。
        let c: ConfigHash = ConfigHash::of_bytes(b"width=1920;height=1080;fps=60");
        assert_ne!(a, c, "配置字节不同必得不同哈希");
        // 空配置也有确定值。
        assert_eq!(ConfigHash::of_bytes(b""), ConfigHash::of_bytes(b""));
        // from_raw 往返。
        assert_eq!(ConfigHash::from_raw(a.value()), a);
    }

    #[test]
    fn task_id_is_content_derived_and_used_for_idempotency() {
        let t1: FrameTask = FrameTask::new(FrameRange { start: 0, end: 99 }, ConfigHash::of_bytes(b"a"));
        let t2: FrameTask = FrameTask::new(FrameRange { start: 0, end: 99 }, ConfigHash::of_bytes(b"a"));
        assert_eq!(t1.id(), t2.id(), "同配置同区间 ⇒ 同 id（幂等依据）");
        let t3: FrameTask = FrameTask::new(FrameRange { start: 0, end: 98 }, ConfigHash::of_bytes(b"a"));
        assert_ne!(t1.id(), t3.id(), "区间不同 ⇒ id 不同");
        let t4: FrameTask = FrameTask::new(FrameRange { start: 0, end: 99 }, ConfigHash::of_bytes(b"b"));
        assert_ne!(t1.id(), t4.id(), "配置不同 ⇒ id 不同");
        // id 的显示形式稳定（跨机日志对齐用）。
        assert_eq!(t1.id().to_string(), t2.id().to_string());
        assert!(t1.id().to_string().contains("frames=0-99"));
    }

    #[test]
    fn order_key_is_pure_content_derivation_with_priority_first() {
        // 优先级小者先；同优先级比配置哈希；再比帧起点。
        let mk = |prio: u32, cfg: &[u8], start: u64| FrameTask {
            frames: FrameRange { start, end: start + 9 },
            config_hash: ConfigHash::of_bytes(cfg),
            priority: prio,
            resources: ResourceNeed::unbounded(),
        };
        let hi: FrameTask = mk(0, b"z", 100);
        let lo: FrameTask = mk(5, b"a", 0);
        assert!(hi.order_key() < lo.order_key(), "优先级 0 必须先于优先级 5（与帧号无关）");

        let a1: FrameTask = mk(1, b"a", 50);
        let a2: FrameTask = mk(1, b"b", 0);
        assert!(a1.order_key() < a2.order_key(), "同优先级比配置哈希（与帧号无关）");

        let s1: FrameTask = mk(1, b"a", 0);
        let s2: FrameTask = mk(1, b"a", 10);
        assert!(s1.order_key() < s2.order_key(), "同优先级同配置比帧起点");

        // 排序键与构造顺序无关（内容派生）。
        let mut keys: Vec<TaskOrderKey> = vec![a2.order_key(), s2.order_key(), hi.order_key()];
        keys.sort();
        let mut keys2: Vec<TaskOrderKey> = keys.clone();
        keys2.reverse();
        keys2.sort();
        assert_eq!(keys, keys2, "排序键的排序结果与初始排列无关");
    }

    #[test]
    fn resource_need_fits_respects_zero_as_unbounded() {
        let unbounded: ResourceNeed = ResourceNeed::unbounded();
        let node: ResourceNeed = ResourceNeed { cpu_cores: 8, memory_mb: 16384, vram_mb: 0 };
        assert!(unbounded.fits(&node), "0 需求视为不限，恒可容纳");
        let big: ResourceNeed = ResourceNeed { cpu_cores: 64, memory_mb: 1, vram_mb: 0 };
        assert!(!big.fits(&node), "超需求必须拒绝");
        // 节点容量为 0 时有界需求不可满足（0 需求侧才不限）。
        let zero_node: ResourceNeed = ResourceNeed { cpu_cores: 0, memory_mb: 0, vram_mb: 0 };
        assert!(!big.fits(&zero_node), "节点某项容量为 0 ⇒ 该项不可满足");
        assert!(ResourceNeed { cpu_cores: 4, memory_mb: 0, vram_mb: 0 }.fits(&node));
    }
}
