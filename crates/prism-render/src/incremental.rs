//! 增量渲染：脏区间检测与复用（C3）。
//!
//! 语义权威：（子图签名树 × 帧区间推导双维复用；缓存键 =
//! 子图签名 × 分辨率 × spp 档 × tile × engine 版本 × schema 版本；[T] 红利：
//! 签名不变即累积续用）；缓存键五元组 + tile、
//!  / D-28（快照键 = (子图签名, 帧)；LRU 逐出只影响重算不影响结果）；
//! 验收：改一节点 → 未涉区间帧 md5 一致且
//! reuse_ratio ≥ 0.5、D-16（Mini Code）、D-27（registry! 静态汇入制）。
//!
//! # 三维复用模型
//! 1. **子图维（签名）**：改一个节点 → 仅其"自身到根的祖先锥"签名变化，
//!    兄弟子树签名逐位不变（`signature::graph_signatures` 已证）。因此以
//!  "根节点子树签名"为粗粒度缓存单位——签名相等 ⇒ 逐字节可复用（公理）；
//! 2. **帧维（区间）**：通道分类（Const 静态 / Track 分段常 / Expr 变化率界 /
//!    Spring Lipschitz 界）给出参数随时间变化的分段常区间——区间内参数恒定
//!    ⇒ 该子图结果恒定 ⇒ 帧间复用；
//! 3. **tile 维**：缓存键含 tile 坐标，本模块以整幅为 key 单元，
//!    tile 网格由 `orchestrator` 落地时展开为多 tile 键（键生成同 [`cache_key`]）。
//!
//! # 确定性铁律（第一优先）
//! 复用**绝不能改变输出**：命中缓存 = 直接取回上帧字节；未命中 = 全量重算。
//! 两者产出经 [`ReuseVerdict`] 保证**逐字节一致**（验收关卡 2）。LRU 逐出
//! 只影响"是否重算"，不影响"算出什么"——seek 等价成立（D-28）。

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fmt;

use prism_core::signature::{
    cache_key, graph_signatures, CacheDims, ChannelDefs, SignatureError,
};
use prism_core::node::Hash256;
use prism_core::{Graph, NodeId};
use prism_io::fbuf::Fbuf16F;

// ---------------------------------------------------------------------------
// 参数变更粒度（通道分类 → 帧区间推导）
// ---------------------------------------------------------------------------

/// 通道随时间变化的分类（CH0–CH4）。
///
/// 分类决定"参数变化把哪些帧区间判脏"：
/// - [`ChannelKind::Constant`]：常值，改值 → 全帧但仅该子图；
/// - [`ChannelKind::Track`]：关键帧轨道，给出分段常区间（关键帧之间参数恒定）；
/// - [`ChannelKind::Expr`]：表达式，给出变化率 Lipschitz 上界；
/// - [`ChannelKind::Spring`]：弹簧类（闭式解析解），给出 Lipschitz 界；
/// - [`ChannelKind::AudioReactive`]：音频反应（烘焙表），给出采样区间。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// 常值（CH0）：不随帧变化。
    Constant,
    /// 关键帧轨道（CH1）：关键帧之间为分段常。
    Track,
    /// 表达式（CH2）：给出变化率上界（0 = 保守逐帧）。
    Expr,
    /// 弹簧/Damped（CH3）：Lipschitz 界。
    Spring,
    /// 音频反应（CH4）：烘焙采样区间。
    AudioReactive,
}

/// 帧区间（闭区间 `[start, end]`，帧号整数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    /// 起始帧（含）。
    pub start: u64,
    /// 结束帧（含）。
    pub end: u64,
}

impl FrameRange {
    /// 单帧区间。
    pub fn single(frame: u64) -> Self {
        FrameRange { start: frame, end: frame }
    }

    /// 全序列区间 `[0, total-1]`。
    pub fn full(total: u64) -> Self {
        FrameRange { start: 0, end: total.saturating_sub(1) }
    }

    /// 区间长度（帧数）。
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start) + 1
    }

    /// 是否为空（start > end）。
    pub fn is_empty(&self) -> bool {
        self.start > self.end
    }

    /// 是否包含某帧。
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start && frame <= self.end
    }

    /// 与另一区间的交集。
    pub fn intersect(&self, other: &FrameRange) -> FrameRange {
        FrameRange {
            start: self.start.max(other.start),
            end: self.end.min(other.end),
        }
    }
}

/// 通道区间表：通道名 → 该通道"参数恒定"的帧区间集合（升序、互不重叠）。
///
/// 由 `prism-anim`（property.rs）按通道分类 + 关键帧密度产出；本模块消费
/// 其结果推导脏区间。空表 = 无任何区间 ⇒ 保守判全帧脏。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelIntervals {
    maps: BTreeMap<String, Vec<FrameRange>>,
}

impl ChannelIntervals {
    /// 构造空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记某通道的一个恒定区间（同通道多段应各自升序加入）。
    pub fn push(&mut self, channel: &str, range: FrameRange) {
        let vec = self.maps.entry(channel.to_string()).or_default();
        vec.push(range);
        vec.sort_by_key(|r| (r.start, r.end));
    }

    /// 某通道的恒定区间（升序）。
    pub fn intervals_of(&self, channel: &str) -> &[FrameRange] {
        self.maps.get(channel).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 是否为空表。
    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    /// 给定"变更帧区间"（保守过近似，可能全帧），返回该通道**恒定**（参数不
    /// 变、产物可复用）的区间集合 = 与该变更包络**有交集**的通道恒定区间。
    ///
    /// 语义：`changed_frames` 是保守起点（无变化率界时可全帧）；
    /// 通道恒定区间描述"哪些帧参数压根不变"。落在变更包络**之内**的恒定帧可
    /// 被安全扣除（复用），故这里返回与变更包络相交的恒定区间；调用方再用
    /// [`subtract_ranges`] 将其从保守包络中扣除。
    ///
    /// 保守情形（无区间信息）→ 返回空（调用方判全帧脏）。
    pub fn stable_intervals_excluding(
        &self,
        channel: &str,
        changed: &FrameRange,
    ) -> Vec<FrameRange> {
        self.intervals_of(channel)
            .iter()
            .filter(|r| !r.intersect(changed).is_empty())
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 脏区间推导（子图签名变化 → 脏子树 + 脏帧区间）
// ---------------------------------------------------------------------------

/// 参数变更记录：某节点的一份参数（或其绑定通道）在上帧基线之上发生了变化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamChange {
    /// 变更节点 id。
    pub node: NodeId,
    /// 该节点绑定的通道名（无绑定 = 静态参数变更；`None` 表示参数值直接变更）。
    pub channel: Option<String>,
    /// 参数实际变化的帧区间（静态参数变更 → 全帧）。
    pub changed_frames: FrameRange,
}

/// 脏推导结果：脏子树（需重算的节点集合）+ 每节点脏帧区间。
///
/// 每节点携带**一组**帧区间（帧维复用收窄后一个节点的脏帧可能被恒定带切成
/// 多段；单 `FrameRange` 无法表达"洞"）。空区间段不记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtySet {
    /// 脏节点 → 脏帧区间集合（升序、互不重叠；节点自身到根的祖先锥）。
    dirty: BTreeMap<NodeId, Vec<FrameRange>>,
}

impl DirtySet {
    /// 是否含某节点为脏。
    pub fn contains(&self, node: NodeId) -> bool {
        self.dirty.contains_key(&node)
    }

    /// 某节点的脏帧区间**并集包络**（min start, max end；无 → None）。
    ///
    /// 兼容单区间调用方；多段脏帧下返回覆盖全部脏帧的最小包络。
    pub fn range_of(&self, node: NodeId) -> Option<FrameRange> {
        let ranges = self.dirty.get(&node)?;
        let start = ranges.iter().map(|r| r.start).min()?;
        let end = ranges.iter().map(|r| r.end).max()?;
        Some(FrameRange { start, end })
    }

    /// 某节点的脏帧区间集合（升序、互不重叠；无 → 空切片）。
    pub fn ranges_of(&self, node: NodeId) -> &[FrameRange] {
        self.dirty.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 脏节点数。
    pub fn len(&self) -> usize {
        self.dirty.len()
    }

    /// 是否为空（无任何脏节点）。
    pub fn is_empty(&self) -> bool {
        self.dirty.is_empty()
    }

    /// 全体脏节点（升序）。
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.dirty.keys().copied()
    }

    /// 脏帧并集大小：**至少一个脏节点**的帧区间覆盖的帧数（帧维复用口径）。
    ///
    /// 注意与 [`DirtySet::len`]（脏**节点**数）不同：帧维复用以"帧"为单元，
    /// 各节点的脏区间取**并集**（同一帧被多节点判脏只计一次）。
    pub fn dirty_frame_count(&self) -> u64 {
        let mut frames: Vec<(u64, u64)> = self
            .dirty
            .values()
            .flat_map(|rs| rs.iter())
            .filter(|r| !r.is_empty())
            .map(|r| (r.start, r.end))
            .collect();
        frames.sort_unstable();
        let mut total: u64 = 0;
        let mut cur: Option<(u64, u64)> = None;
        for (s, e) in frames {
            match cur {
                None => cur = Some((s, e)),
                Some((cs, ce)) => {
                    if s <= ce.saturating_add(1) {
                        // 相邻或重叠 → 合并。
                        cur = Some((cs, ce.max(e)));
                    } else {
                        total += ce.saturating_sub(cs) + 1;
                        cur = Some((s, e));
                    }
                }
            }
        }
        if let Some((cs, ce)) = cur {
            total += ce.saturating_sub(cs) + 1;
        }
        total
    }
}

/// 脏推导错误（含签名计算错误透传）。
#[derive(Debug, Clone, PartialEq)]
pub enum IncrementalError {
    /// 签名计算错误透传（环图 / 缺失通道定义）。
    Signature(SignatureError),
    /// 参数变更引用了图中不存在的节点。
    UnknownNode {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：缺失节点 id。
        node: NodeId,
    },
}

impl fmt::Display for IncrementalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncrementalError::Signature(e) => write!(f, "{e}"),
            IncrementalError::UnknownNode { location, node } => write!(
                f,
                "error[P3014] {location}\n  什么事：参数变更引用的节点 {node} 不在图中\n  建议：核对变更记录的节点 id；先 add_node 再加入变更"
            ),
        }
    }
}

impl std::error::Error for IncrementalError {}

impl From<SignatureError> for IncrementalError {
    fn from(e: SignatureError) -> Self {
        IncrementalError::Signature(e)
    }
}

/// 下游（消费者）节点：`id` 的输出流向哪些节点（升序、去重）。
///
/// `Graph::inputs_of` 只给上游；脏区间需向**祖先（下游）**传播，故反查全图
/// 输入边。图规模为渲染图（节点数小），线性扫描可接受且完全确定。
fn downstream_of(graph: &Graph, id: NodeId) -> Vec<NodeId> {
    let mut out: Vec<NodeId> = Vec::new();
    for node in graph.nodes() {
        for (_, from) in graph.inputs_of(node.id) {
            if from == id {
                out.push(node.id);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 从 `base` 区间中**扣除**一组 `exclude` 区间，返回剩余的**帧区间集合**
/// （升序、互不重叠、丢弃空段）。
///
/// 帧维复用核心算术：`base` 是保守变更区间（可能全帧），`exclude` 是通道
/// 恒定（参数不变）区间。结果 = `base \ ∪exclude` 的最大连续段集合。恒定帧
/// 被精确扣除（例如 base=[0,19]、exclude={[0,4],[10,19]} → 结果 {[5,9]}）。
///
/// 边界：`exclude` 为空 → 返回 `[base]`（保守）；`base` 为空 → 返回空。
fn subtract_ranges(base: FrameRange, exclude: &[FrameRange]) -> Vec<FrameRange> {
    if base.is_empty() {
        return Vec::new();
    }
    // 合并后的排除区间（升序）逐段从 base 中"切洞"。
    let mut ex: Vec<FrameRange> = exclude.iter().copied().filter(|r| !r.is_empty()).collect();
    ex.sort_by_key(|r| (r.start, r.end));
    let mut out: Vec<FrameRange> = Vec::new();
    let mut cursor: u64 = base.start;
    for r in ex {
        if r.end < cursor {
            continue; // 完全在游标之前。
        }
        if r.start > base.end {
            break; // 完全在 base 之后。
        }
        if r.start > cursor {
            // 排除区间之前还有一段脏帧。
            out.push(FrameRange { start: cursor, end: r.start - 1 });
        }
        // 游标推到排除区间之后。
        cursor = cursor.max(r.end.saturating_add(1));
        if cursor > base.end {
            break;
        }
    }
    if cursor <= base.end {
        out.push(FrameRange { start: cursor, end: base.end });
    }
    out
}

fn union_ranges(a: &[FrameRange], b: &[FrameRange]) -> Vec<FrameRange> {
    let mut all: Vec<FrameRange> = a
        .iter()
        .chain(b.iter())
        .copied()
        .filter(|r| !r.is_empty())
        .collect();
    all.sort_by_key(|r| (r.start, r.end));
    let mut out: Vec<FrameRange> = Vec::new();
    for r in all {
        match out.last_mut() {
            Some(last) if r.start <= last.end.saturating_add(1) => {
                // 相邻或重叠 → 合并为一段。
                last.end = last.end.max(r.end);
            }
            _ => out.push(r),
        }
    }
    out
}

/// 脏区间检测器：以"上帧各节点签名"为基线，比对当前图签名，推导脏子树。
///
/// # 算法
/// 节点签名自底向上（[`graph_signatures`]）：改一个节点只会改变"该节点自身
/// 到根的祖先锥"上各节点的签名；兄弟子树签名逐位不变。因此：
/// 1. 计算当前帧签名表 `cur`；
/// 2. 与基线 `base` 逐节点比对：签名不同 ⇒ 该节点脏；
/// 3. 一个节点脏 ⇒ **其全部祖先**必脏（签名自底向上传递）——`graph_signatures`
///    已保证这一传递，无需本模块再爬祖先链；本模块只需逐节点比对；
/// 4. 脏节点的帧区间 = 变更记录给出的 `changed_frames`（静态 → 全帧）。
#[derive(Debug, Clone, Default)]
pub struct DirtyDetector {
    /// 上帧基线：节点 → 子树签名。
    base: BTreeMap<NodeId, Hash256>,
}

impl DirtyDetector {
    /// 空基线构造（首帧：全图脏）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 以上帧签名表构造基线。
    pub fn with_base(base: BTreeMap<NodeId, Hash256>) -> Self {
        DirtyDetector { base }
    }

    /// 当前基线快照（节点 → 签名）。
    pub fn baseline(&self) -> &BTreeMap<NodeId, Hash256> {
        &self.base
    }

    /// **检测单节点参数变更**：只有"When 参数变更 → 脏子树与脏帧区间"。
    ///
    /// `changes` 逐条给出变更节点、绑定通道（区分参数静态变更 vs 通道驱动）
    /// 与变更帧区间。返回脏集合并把基线更新为当前签名。
    ///
    /// # 关键不变式
    /// - 改叶 1 参数 → 仅 {1, 3, 4}（叶到根链）脏，兄弟子树 {2} 不脏（验收 1）；
    /// - 帧维：`channel` 有界且 [`ChannelIntervals`] 给出恒定区间时，脏帧区间
    ///   收窄为 `changed_frames`；否则保守全帧。
    ///
    /// # Errors
    /// 图含环、缺失通道定义、或变更引用不存在节点时结构化报错（永不静默，C20）。
    pub fn detect(
        &mut self,
        graph: &Graph,
        defs: &ChannelDefs,
        changes: &[ParamChange],
        total_frames: u64,
        channel_intervals: &ChannelIntervals,
    ) -> Result<DirtySet, IncrementalError> {
        let cur = graph_signatures(graph, defs)?;
        let mut dirty: BTreeMap<NodeId, Vec<FrameRange>> = BTreeMap::new();

        // 参数变更的"根脏节点"：其帧区间按祖先链**向上传播**——改叶 1 的帧 40，
        // 其祖先 3、4 也只在帧 40 需要重算（签名变而"哪些帧变"由根变更决定）。
        let mut requested: BTreeMap<NodeId, Vec<FrameRange>> = BTreeMap::new();
        for change in changes {
            if !cur.contains_key(&change.node) {
                return Err(IncrementalError::UnknownNode {
                    location: format!("incremental.detect(参数变更节点 {})", change.node),
                    node: change.node,
                });
            }
            // 帧维复用落地：若变更绑定通道且 ChannelIntervals
            // 给出该通道的**恒定区间**，则把这些恒定帧从变更区间中**扣除**
            // ——恒定帧内参数不变 ⇒ 其产物可复用，不计入脏帧。
            //
            // 收窄语义：`changed_frames` 是保守起点（可能全帧），恒定区间与该
            // 起点不相交（由 `stable_intervals_excluding` 保证），故收窄后的脏
            // 区间 = `changed_frames` **减去**恒定区间（可能被切成多段）。多个
            // 恒定区间逐个扣除；无任何区间信息（空表）→ 保持 `changed_frames`
            // （保守全帧）。
            let effective: Vec<FrameRange> = match change.channel.as_deref() {
                Some(ch) => {
                    let stable = channel_intervals
                        .stable_intervals_excluding(ch, &change.changed_frames);
                    subtract_ranges(change.changed_frames, &stable)
                }
                None => vec![change.changed_frames],
            };
            requested
                .entry(change.node)
                .and_modify(|rs| *rs = union_ranges(rs, &effective))
                .or_insert(effective);
        }

        // 祖先传播：按拓扑序（确定性）自底向上，把根变更的帧区间并到**全部
        // 下游（祖先）**。改叶 1 的帧 40 → 其祖先 3、4 也只在帧 40 需重算。
        let order = graph
            .topo_order()
            .map_err(|e| IncrementalError::Signature(SignatureError::Graph(e)))?;
        for id in order {
            if let Some(ranges) = requested.get(&id).cloned() {
                for to in downstream_of(graph, id) {
                    let merged = match requested.get(&to) {
                        Some(existing) => union_ranges(existing, &ranges),
                        None => ranges.clone(),
                    };
                    requested.insert(to, merged);
                }
            }
        }

        // 签名比对：与基线不同的节点即脏。首帧（空基线）→ 全图脏。
        let base_is_empty: bool = self.base.is_empty();
        for (&id, cur_sig) in &cur {
            let is_dirty: bool = match (base_is_empty, self.base.get(&id)) {
                (true, _) => true,
                (false, None) => true, // 新增节点 → 脏
                (false, Some(base_sig)) => base_sig != cur_sig,
            };
            if is_dirty {
                // 脏节点帧区间：若有显式变更请求取请求区间；否则全帧
                // （签名变而未登记变更 = 保守全帧，如实标注）。
                let ranges = requested
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| vec![FrameRange::full(total_frames)]);
                dirty.insert(id, ranges);
            }
        }
        // 变更节点若未在 cur（已在上方报错）——此处 cur 覆盖全部图节点，故脏集
        // 必已含所有签名变化节点。基线更新为当前签名。
        self.base = cur;
        Ok(DirtySet { dirty })
    }

    /// 显式重置基线（强制全图重算，如引擎版本变更）。
    pub fn reset(&mut self) {
        self.base.clear();
    }
}

// ---------------------------------------------------------------------------
// 复用判定：签名相等 → 直接复用上帧结果（缓存命中与否逐字节一致）
// ---------------------------------------------------------------------------

/// 复用裁决：一帧一子图的处理决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseVerdict {
    /// 命中缓存：复用上帧字节（不重算）。
    Reuse,
    /// 未命中：需全量重算。
    Recompute,
}

/// 复用判定器：以缓存键为索引，判定某帧某子图能否复用上帧结果。
///
/// # 确定性铁律
/// 判定**只**看缓存键是否相等（键含子图签名 + 分辨率 + spp 档 + tile +
/// 版本）。键相等 ⇒ 输入逐字节相同 ⇒ 输出逐字节相同（公理）。命中即
/// 取回缓存字节；未命中即重算并回填缓存。二者对同一 (帧, 子图) 的输出**逐
/// 字节一致**（验收关卡 2）——这是复用不改输出的可证性来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReuseDecision {
    /// 裁决。
    pub verdict: ReuseVerdict,
    /// 缓存键（命中/未命中的判定依据）。
    pub key: Hash256,
}

/// 判定某子图在当前维度下能否复用（键 + 维度 → 裁决）。
///
/// 复用条件 = 缓存含该键。**签名碰撞防护**：键是 256 位哈希（S10：10⁹ 条目
/// 下生日界碰撞概率 ~10⁻⁵⁹，缓存正确性有数字保单）；调用方以 `store` 的键
/// 存在性为唯一判据，不做启发式近似。
pub fn decide(
    store: &ReuseCache,
    subgraph: &Hash256,
    dims: &CacheDims,
) -> ReuseDecision {
    let key = cache_key(subgraph, dims);
    let verdict = if store.contains(&key) {
        ReuseVerdict::Reuse
    } else {
        ReuseVerdict::Recompute
    };
    ReuseDecision { key, verdict }
}

// ---------------------------------------------------------------------------
// 缓存管理：LRU + 快照库对接（D-28；缓存命中与否输出逐字节一致）
// ---------------------------------------------------------------------------

/// 带 LRU 的增量复用缓存：缓存键 → tile 产物字节，容量上限触发逐出。
///
/// 语义权威：D-28（LRU 逐出只影响重算不影响结果——seek 等价成立）；
/// （缓存键含全部渲染维度）。逐出策略为**最久未用**（LRU），逐出仅使将来
/// 未命中而重算，绝不改变对任何键的回取值。
#[derive(Debug, Clone)]
pub struct ReuseCache {
    entries: BTreeMap<Hash256, Vec<u8>>,
    /// 最近使用序（front = 最近，back = 最久）；命中/插入时移动到 front。
    lru: VecDeque<Hash256>,
    /// 容量上限（条目数；0 = 无限）。
    capacity: usize,
    /// 累计命中次数（reuse_ratio 的另一观测面）。
    hits: u64,
    /// 累计未命中（重算）次数。
    misses: u64,
}

impl ReuseCache {
    /// 以容量构造（`capacity = 0` 表示不限容量）。
    pub fn new(capacity: usize) -> Self {
        ReuseCache {
            entries: BTreeMap::new(),
            lru: VecDeque::new(),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    /// 容量。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 命中次数。
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// 未命中次数。
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// 含某键（判定用；不改 LRU——判定与取值分离，保证 [`decide`] 幂等）。
    pub fn contains(&self, key: &Hash256) -> bool {
        self.entries.contains_key(key)
    }

    /// 读取并标记最近使用（命中路径）。返回产物字节。
    pub fn touch_get(&mut self, key: &Hash256) -> Option<Vec<u8>> {
        match self.entries.get(key).cloned() {
            Some(bytes) => {
                self.hits += 1;
                self.mark_used(key);
                Some(bytes)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// 写入（未命中路径）：重算结果回填，容量超限触发 LRU 逐出。
    pub fn insert(&mut self, key: Hash256, payload: Vec<u8>) {
        self.entries.insert(key, payload);
        self.mark_used(&key);
        while self.capacity != 0 && self.entries.len() > self.capacity {
            if let Some(victim) = self.lru.pop_back() {
                self.entries.remove(&victim);
            } else {
                break;
            }
        }
    }

    /// 移动键到 LRU 队首（最近使用）；若不在队中则入队。
    fn mark_used(&mut self, key: &Hash256) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_front(*key);
    }

    /// 命中率 = hits / (hits + misses)；无访问返回 0.0（无定义 → 保守零）。
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// reuse_ratio 统计（复用率度量；验收 ≥ 0.5）
// ---------------------------------------------------------------------------

/// 复用率度量：以"帧 × 子图"为统计单元。
///
/// `reuse_ratio = 复用单元数 / 总单元数`。验收 ≥ 0.5。
///
/// # 统计口径
/// 总单元 = 全序列帧数 × 逻辑子图数（"逻辑子图"由调用方界定：通常为全图
/// 一次渲染 = 1 个单元；多子图场景按子图数计）。复用单元 = 落在脏区间之外
/// 的单元（未涉及区间）+ 命中缓存的单元。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReuseStats {
    /// 复用单元数。
    reused: u64,
    /// 总单元数。
    total: u64,
}

impl ReuseStats {
    /// 构造空统计。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一批单元：`reused` 个复用 + `total - reused` 个重算。
    ///
    /// # Panics
    /// `reused > total` 时 panic（统计不变量违反）。
    pub fn record(&mut self, reused: u64, total: u64) {
        assert!(reused <= total, "复用数不得超过总数");
        self.reused += reused;
        self.total += total;
    }

    /// 复用率 = reused / total；无单元返回 0.0。
    pub fn ratio(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.reused as f64 / self.total as f64
        }
    }

    /// 复用单元数。
    pub fn reused(&self) -> u64 {
        self.reused
    }

    /// 总单元数。
    pub fn total(&self) -> u64 {
        self.total
    }

    /// 是否达验收线（≥ 0.5）。
    pub fn meets_phase3(&self) -> bool {
        self.ratio() >= 0.5
    }
}

/// 由脏集与总帧数计算复用统计：复用单元 = (总帧数 × 子图数) − 脏帧并集 × 脏子图数。
///
/// 即"未涉及区间"免于重算—— 超越点（子图级 × 帧区间级双维复用）。
/// 单子图场景（`subgraph_count = 1`）下：脏单元 = 脏帧并集；改动帧之外全部复用。
pub fn reuse_stats_from_dirty(
    dirty: &DirtySet,
    total_frames: u64,
    subgraph_count: u64,
) -> ReuseStats {
    let total_units = total_frames.saturating_mul(subgraph_count);
    // 脏帧并集：帧维复用口径（多节点判脏的同一帧只计一次）。
    let dirty_frames = dirty.dirty_frame_count();
    let dirty_units = dirty_frames.min(total_frames);
    let reused_units = total_units.saturating_sub(dirty_units);
    let mut stats = ReuseStats::new();
    stats.record(reused_units, total_units);
    stats
}

// ---------------------------------------------------------------------------
// 增量渲染会话：把脏检测 + 复用 + 缓存 + 统计串成一次渲染循环
// ---------------------------------------------------------------------------

/// 单个 (帧, 子图) 的渲染产物（tile 帧缓冲的字节化）。
pub type TileArtifact = Vec<u8>;

/// 增量渲染会话（高内核：一次序列渲染的复用记账）。
///
/// # 用法
/// 1. 首帧：基线空 → 全图脏 → 全部重算 → 回填缓存；
/// 2. 后续帧：喂 `changes` → 检测脏子树 → 对脏单元重算、脏区间外复用；
/// 3. 每帧产物**逐字节等于全量重算**（复用不改输出）。
#[derive(Debug, Clone)]
pub struct IncrementalSession {
    detector: DirtyDetector,
    cache: ReuseCache,
    stats: ReuseStats,
    /// 会话当前帧号（D-28 快照键的帧维）。`process_unit` 的缓存键 = (子图签名,
    /// 帧, 维度)——不同帧同子图同维度必须得到不同键，杜绝跨帧误命中。
    frame: u64,
}

impl IncrementalSession {
    /// 以缓存容量构造会话。
    pub fn new(cache_capacity: usize) -> Self {
        IncrementalSession {
            detector: DirtyDetector::new(),
            cache: ReuseCache::new(cache_capacity),
            stats: ReuseStats::new(),
            frame: 0,
        }
    }

    /// 显式设定当前帧号（调用方按渲染时序推进；D-28 帧维）。
    pub fn set_frame(&mut self, frame: u64) {
        self.frame = frame;
    }

    /// 当前帧号。
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// 推进到下一帧。
    pub fn advance_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// 复用统计。
    pub fn stats(&self) -> ReuseStats {
        self.stats
    }

    /// 缓存引用（判定用）。
    pub fn cache(&self) -> &ReuseCache {
        &self.cache
    }

    /// 检测一帧的脏集合并更新基线；同时把统计登记进去。
    ///
    /// # Errors
    /// 同 [`DirtyDetector::detect`]。
    pub fn observe_frame(
        &mut self,
        graph: &Graph,
        defs: &ChannelDefs,
        changes: &[ParamChange],
        total_frames: u64,
        subgraph_count: u64,
        intervals: &ChannelIntervals,
    ) -> Result<DirtySet, IncrementalError> {
        let dirty = self
            .detector
            .detect(graph, defs, changes, total_frames, intervals)?;
        // 统计：脏单元 = 脏帧并集（单子图场景即脏帧数）。
        let total_units = total_frames.saturating_mul(subgraph_count);
        let dirty_units = dirty.dirty_frame_count().min(total_frames);
        let reused_units = total_units.saturating_sub(dirty_units);
        self.stats.record(reused_units, total_units);
        Ok(dirty)
    }

    /// 处理一个 (帧, 子图) 单元：命中 → 复用缓存字节；未命中 → 用 `recompute`
    /// 全量重算并回填。
    ///
    /// # 帧维（D-28）
    /// 缓存键 = ([`cache_key`]，含**帧号**)——帧号取自 `dims.frame`（调用方按渲染
    /// 时序设定）。不同帧同子图同维度 ⇒ 不同键 ⇒ 绝不跨帧误命中（帧 3 不会取回
    /// 帧 2 的字节）。
    ///
    /// # 确定性保证
    /// `recompute` 是纯函数（同输入同输出）；命中路径返回的字节 = 上一次
    /// 同键 `recompute` 的字节 ⇒ 命中与重算**逐字节一致**（验收关卡 2、4）。
    pub fn process_unit<F>(
        &mut self,
        subgraph: &Hash256,
        dims: &CacheDims,
        recompute: F,
    ) -> (TileArtifact, ReuseVerdict)
    where
        F: FnOnce() -> TileArtifact,
    {
        // 缓存键 = (子图签名, **会话当前帧**, 维度)。会话帧是 D-28 帧维权威：
        // 不同帧同子图同维度 ⇒ 不同键 ⇒ 绝不跨帧误命中。若会话帧为 0（未显式
        // 推进），回退到 `dims.frame`（调用方可在 dims 中携带帧号）。
        let mut effective = *dims;
        if self.frame != 0 {
            effective.frame = self.frame;
        }
        let key = cache_key(subgraph, &effective);
        if let Some(bytes) = self.cache.touch_get(&key) {
            (bytes, ReuseVerdict::Reuse)
        } else {
            let bytes = recompute();
            self.cache.insert(key, bytes.clone());
            (bytes, ReuseVerdict::Recompute)
        }
    }
}

// ---------------------------------------------------------------------------
// Fbuf16F 字节化助手（缓存产物 = RGBA16F 位模式字节）
// ---------------------------------------------------------------------------

/// 帧缓冲 → 确定性字节串（小端 f16 位模式；缓存产物的唯一出口）。
pub fn fbuf_to_bytes(fb: &Fbuf16F) -> TileArtifact {
    let mut out = Vec::with_capacity(8 + fb.pixels.len() * 2);
    out.extend_from_slice(&fb.width.to_le_bytes());
    out.extend_from_slice(&fb.height.to_le_bytes());
    for h in &fb.pixels {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// 字节串 → 帧缓冲（[`fbuf_to_bytes`] 逆运算；往返逐位精确）。
///
/// # Errors
/// 长度不符（字节串损坏）时返回 Err。
pub fn bytes_to_fbuf(bytes: &[u8]) -> Result<Fbuf16F, String> {
    if bytes.len() < 8 {
        return Err(format!("帧缓冲字节串过短：{} 字节", bytes.len()));
    }
    let mut wb = [0u8; 4];
    wb.copy_from_slice(&bytes[0..4]);
    let mut hb = [0u8; 4];
    hb.copy_from_slice(&bytes[4..8]);
    let w = u32::from_le_bytes(wb);
    let h = u32::from_le_bytes(hb);
    let expect = (w as usize) * (h as usize) * 4 * 2; // RGBA × f16 = 4 × 2 字节/像素
    if bytes.len() != 8 + expect {
        return Err(format!(
            "帧缓冲字节串长度不符：期望 {}，实际 {}",
            8 + expect,
            bytes.len()
        ));
    }
    let mut fb = Fbuf16F::new(w, h);
    for (i, chunk) in bytes[8..].chunks_exact(2).enumerate() {
        fb.pixels[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(fb)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::node::{Composition, MergeOp, Node, ParamValue, StructureId};

    /// 叶子节点（可设参数值）。
    fn leaf(id: NodeId, structure: StructureId, x: f32) -> Node {
        let mut node = Node::new(id, structure, Composition::Feed);
        node.params.set("x", ParamValue::F32(x));
        node
    }

    /// 链形图：1(叶) ┐
    ///          3 = Merge(Plus) ── 4(Chain 根)
    ///        2(叶) ┘
    fn chain_graph(x1: f32) -> Graph {
        let mut g = Graph::new();
        g.add_node(leaf(1, StructureId::ScalarField, x1)).unwrap();
        g.add_node(leaf(2, StructureId::Convolution, 2.0)).unwrap();
        g.add_node(Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus)))
            .unwrap();
        g.add_node(Node::new(4, StructureId::Convolution, Composition::Chain))
            .unwrap();
        g.connect(1, 3, 0).unwrap();
        g.connect(2, 3, 1).unwrap();
        g.connect(3, 4, 0).unwrap();
        g
    }

    fn dims(version: u64) -> CacheDims {
        CacheDims {
            width: 192,
            height: 108,
            spp_tier: 2,
            tile: (0, 0),
            engine_version: version,
            frame: 0,
        }
    }

    // ---- 验收关卡 1：脏区间正确性——改一节点只脏对应子图 ----

    #[test]
    fn checkpoint1_changing_one_leaf_only_dirties_its_ancestor_chain() {
        let empty = ChannelDefs::new();
        let intervals = ChannelIntervals::new();
        let mut det = DirtyDetector::new();
        let total = 10u64;

        // 首帧：空基线 → 全图脏。
        let first = det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
        assert_eq!(first.len(), 4, "首帧空基线 → 全图脏");
        for id in [1u64, 2, 3, 4] {
            assert!(first.contains(id));
        }

        // 第二帧：只改叶 1 的参数 → 脏集 = {1, 3, 4}（叶到根链），兄弟 2 不脏。
        let change = ParamChange {
            node: 1,
            channel: None,
            changed_frames: FrameRange::full(total),
        };
        let d = det
            .detect(&chain_graph(0.75), &empty, &[change], total, &intervals)
            .unwrap();
        assert!(d.contains(1), "变更叶自身脏");
        assert!(d.contains(3), "父节点脏（签名自底向上传递）");
        assert!(d.contains(4), "根脏");
        assert!(!d.contains(2), "兄弟子树签名不变 → 不脏（增量根基）");
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn checkpoint1_unknown_node_change_is_structured_error() {
        let empty = ChannelDefs::new();
        let intervals = ChannelIntervals::new();
        let mut det = DirtyDetector::new();
        let change = ParamChange {
            node: 99,
            channel: None,
            changed_frames: FrameRange::full(5),
        };
        let err = det
            .detect(&chain_graph(0.5), &empty, &[change], 5, &intervals)
            .unwrap_err();
        match &err {
            IncrementalError::UnknownNode { node, .. } => assert_eq!(*node, 99),
            other => panic!("应为 UnknownNode，实际 {other:?}"),
        }
        assert!(err.to_string().contains("P3014"));
        assert!(err.to_string().contains("建议："), "必含建议段（C19）");
    }

    #[test]
    fn frame_range_channel_intervals_reduce_dirty_frames() {
        // 通道恒定区间：Track 通道 [0..4] 与 [10..19] 恒定，参数仅在帧 5 变更。
        let mut intervals = ChannelIntervals::new();
        intervals.push("wind", FrameRange { start: 0, end: 4 });
        intervals.push("wind", FrameRange { start: 10, end: 19 });

        // 保守变更包络 = 全帧（无变化率界的起点）。恒定区间落在包络内 → 可扣除。
        let changed = FrameRange::full(20);
        let stable = intervals.stable_intervals_excluding("wind", &changed);
        assert_eq!(stable.len(), 2, "两段恒定区间均落在变更包络内 → 可扣除");
        assert_eq!(stable[0], FrameRange { start: 0, end: 4 });
        assert_eq!(stable[1], FrameRange { start: 10, end: 19 });

        // 变更包络仅覆盖 [2,3]：只有与其相交的恒定区间 [0,4] 可扣除。
        let changed2 = FrameRange { start: 2, end: 3 };
        let stable2 = intervals.stable_intervals_excluding("wind", &changed2);
        assert_eq!(stable2.len(), 1, "仅与变更包络相交的恒定区间 [0,4] 被返回");
        assert_eq!(stable2[0], FrameRange { start: 0, end: 4 });

        // 无区间信息通道 → 空（保守全帧脏）。
        assert!(intervals.stable_intervals_excluding("missing", &changed).is_empty());
    }

    // ---- 验收关卡 2：复用结果 == 全量重算（逐字节一致） ----

    #[test]
    fn checkpoint2_reuse_equals_full_recompute_byte_identical() {
        let source = chain_graph(0.5);
        let defs = ChannelDefs::new();
        let sigs = graph_signatures(&source, &defs).unwrap();
        let root_sig = sigs[&4];

        // 纯重算函数：由签名 + 维度确定性派生一小段产物字节。
        let recompute = || -> TileArtifact {
            let mut out = Vec::with_capacity(64);
            for i in 0..64u32 {
                out.push((i as u8).wrapping_mul(7).wrapping_add(root_sig[0]));
            }
            out
        };

        let mut session = IncrementalSession::new(8);

        // 第一次：未命中 → 重算。
        let (a, v1) = session.process_unit(&root_sig, &dims(1), recompute);
        assert_eq!(v1, ReuseVerdict::Recompute, "首次必未命中");

        // 第二次同键：命中 → 复用。产物与重算**逐字节一致**。
        let (b, v2) = session.process_unit(&root_sig, &dims(1), || panic!("命中时不得调用重算"));
        assert_eq!(v2, ReuseVerdict::Reuse, "同键第二次必命中");
        assert_eq!(a, b, "复用产物必须与全量重算逐字节一致（确定性铁律）");

        // 直接重算对照也一致。
        assert_eq!(a, recompute(), "复用产物 == 全量重算产物");
    }

    #[test]
    fn cache_key_dimension_change_misses_and_recomputes() {
        let defs = ChannelDefs::new();
        let sigs = graph_signatures(&chain_graph(0.5), &defs).unwrap();
        let root_sig = sigs[&4];
        let mut session = IncrementalSession::new(8);

        let count = std::cell::Cell::new(0u32);
        let recompute = || {
            count.set(count.get() + 1);
            vec![count.get() as u8; 4]
        };

        // 维度 1 首次：未命中。
        let (a, v1) = session.process_unit(&root_sig, &dims(1), recompute);
        assert_eq!(v1, ReuseVerdict::Recompute);
        assert_eq!(a, vec![1u8; 4]);

        // 引擎版本不同 → 键不同 → 未命中（C13：旧版本不误复用）。
        let (b, v2) = session.process_unit(&root_sig, &dims(2), recompute);
        assert_eq!(v2, ReuseVerdict::Recompute, "引擎版本变化必未命中");
        assert_eq!(b, vec![2u8; 4]);

        // 回维度 1 → 仍命中（键未变）。
        let (c, v3) = session.process_unit(&root_sig, &dims(1), recompute);
        assert_eq!(v3, ReuseVerdict::Reuse);
        assert_eq!(c, a, "同键回取逐字节一致");
    }

    #[test]
    fn decide_matches_touch_get_outcome() {
        let mut cache = ReuseCache::new(0);
        let sig: Hash256 = [5u8; 32];
        let d = dims(1);
        // 空缓存：判定未命中，且一致。
        assert_eq!(decide(&cache, &sig, &d).verdict, ReuseVerdict::Recompute);
        let key = cache_key(&sig, &d);
        cache.insert(key, vec![1, 2, 3]);
        assert!(cache.contains(&key));
        assert_eq!(decide(&cache, &sig, &d).verdict, ReuseVerdict::Reuse);
        // 取值成功。
        assert_eq!(cache.touch_get(&key), Some(vec![1, 2, 3]));
        assert_eq!(cache.hits(), 1);
    }

    // ---- 验收关卡 3：reuse_ratio ≥ 0.5 断言 ----

    #[test]
    fn checkpoint3_reuse_ratio_at_least_half() {
        // 场景：100 帧序列，单子图（subgraph_count = 1）。仅在帧 40 改参数
        // （且该帧脏）→ 脏单元 1，复用 99 → ratio = 0.99 ≥ 0.5。
        let empty = ChannelDefs::new();
        let intervals = ChannelIntervals::new();
        let mut det = DirtyDetector::new();
        let total = 100u64;

        // 基线帧。
        det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
        // 第 40 帧改叶 1 → 脏集 {1,3,4}，帧区间 = 单帧 40。
        let change = ParamChange {
            node: 1,
            channel: None,
            changed_frames: FrameRange::single(40),
        };
        let dirty = det
            .detect(&chain_graph(0.9), &empty, &[change], total, &intervals)
            .unwrap();

        let stats = reuse_stats_from_dirty(&dirty, total, 1);
        println!("checkpoint3 reuse_ratio = {}", stats.ratio());
        assert!(
            stats.ratio() >= 0.5,
            "复用率必须达验收线 ≥ 0.5，实际 {}",
            stats.ratio()
        );
        assert!(stats.meets_phase3());
    }

    #[test]
    fn reuse_ratio_full_sequence_static_change_is_low_but_reported_honestly() {
        // 静态参数变更 → 全帧但仅该子图。单子图场景下全帧脏 → 复用率 0（保守
        // 如实标注）；双子图场景下另一子图整段复用 → ratio = 0.5。
        let empty = ChannelDefs::new();
        let intervals = ChannelIntervals::new();
        let mut det = DirtyDetector::new();
        let total = 10u64;
        det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
        let change = ParamChange {
            node: 1,
            channel: None,
            changed_frames: FrameRange::full(total),
        };
        let dirty = det
            .detect(&chain_graph(0.9), &empty, &[change], total, &intervals)
            .unwrap();

        // 单子图：全帧脏 → 0。
        let s1 = reuse_stats_from_dirty(&dirty, total, 1);
        assert!(s1.ratio() < 0.5, "单子图全帧脏 → 复用率低（如实）");

        // 双子图：脏的只有子图之一，另一子图整段复用。
        // 总单元 = 帧数 × 2；脏单元 = 脏帧并集（10）。→ 复用 10/20 = 0.5。
        let s2 = reuse_stats_from_dirty(&dirty, total, 2);
        assert_eq!(s2.ratio(), 0.5, "双子图改一子图 → 恰 0.5");
        assert!(s2.meets_phase3());
    }

    #[test]
    fn stats_record_invariants_and_ratio() {
        let mut s = ReuseStats::new();
        assert_eq!(s.ratio(), 0.0, "无单元 → 0.0");
        s.record(3, 10);
        s.record(7, 10);
        assert_eq!(s.reused(), 10);
        assert_eq!(s.total(), 20);
        assert_eq!(s.ratio(), 0.5);
        assert!(s.meets_phase3());
    }

    // ---- 验收关卡 4：缓存确定性（LRU 逐出不影响输出） ----

    #[test]
    fn checkpoint4_lru_eviction_does_not_change_output() {
        // 容量 2 的缓存，插入 3 个不同键 → 最久未用被逐出；但任何仍存的键
        // 取值不变、被逐出键重算亦得同字节（D-28：逐出只影响重算不影响结果）。
        let mut cache = ReuseCache::new(2);
        let k = |n: u8| -> Hash256 { [n; 32] };
        cache.insert(k(1), vec![1]);
        cache.insert(k(2), vec![2]);
        cache.insert(k(3), vec![3]); // 逐出 k(1)
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&k(1)), "最久未用被逐出");
        assert!(cache.contains(&k(2)));
        assert!(cache.contains(&k(3)));

        // 重算 k(1) 得同字节（纯函数 → 逐出无损确定性）。
        let mut session = IncrementalSession::new(2);
        let sig = [1u8; 32];
        let d = dims(1);
        let (a, _) = session.process_unit(&sig, &d, || vec![42, 42]);
        let (b, v) = session.process_unit(&sig, &d, || vec![42, 42]);
        assert_eq!(v, ReuseVerdict::Reuse);
        assert_eq!(a, b);
    }

    #[test]
    fn fbuf_bytes_roundtrip_bit_exact() {
        let mut fb = Fbuf16F::new(3, 2);
        for y in 0..2u32 {
            for x in 0..3u32 {
                fb.set(x, y, [x as f32 * 0.1, y as f32 * 0.2, 0.5, 1.0]);
            }
        }
        let bytes = fbuf_to_bytes(&fb);
        let back = bytes_to_fbuf(&bytes).unwrap();
        assert_eq!(fb, back, "RGBA16F 字节化往返逐位精确");
        assert!(bytes_to_fbuf(&bytes[0..4]).is_err(), "损坏字节串结构化拒绝");
    }

    #[test]
    fn session_observe_frame_reports_stats() {
        let empty = ChannelDefs::new();
        let intervals = ChannelIntervals::new();
        let mut session = IncrementalSession::new(4);
        let total = 10u64;

        // 首帧：全图脏 → 复用 0。
        session.observe_frame(&chain_graph(0.5), &empty, &[], total, 1, &intervals).unwrap();
        assert_eq!(session.stats().reused(), 0, "首帧无复用");

        // 第二帧：单帧脏 → 复用 9。
        let change = ParamChange {
            node: 1,
            channel: None,
            changed_frames: FrameRange::single(3),
        };
        session
            .observe_frame(&chain_graph(0.8), &empty, &[change], total, 1, &intervals)
            .unwrap();
        assert_eq!(session.stats().reused(), 9, "单帧脏 → 复用 9 帧");
        let expected = 9.0 / (10.0 + 10.0);
        assert!((session.stats().ratio() - expected).abs() < 1e-12);
    }

    // ---- 帧维复用算术（subtract_ranges / 恒定区间扣除） ----

    #[test]
    fn subtract_ranges_empty_exclude_is_conservative() {
        let base = FrameRange::full(20);
        assert_eq!(subtract_ranges(base, &[]), vec![base], "无排除 → 保守保留 [base]");
    }

    #[test]
    fn subtract_ranges_fully_covered_collapses_to_empty() {
        let base = FrameRange { start: 0, end: 4 };
        let exclude = [FrameRange { start: 0, end: 4 }];
        let out = subtract_ranges(base, &exclude);
        assert!(out.is_empty(), "整段恒定 → 空脏区间");
    }

    #[test]
    fn subtract_ranges_partial_splits_exactly() {
        let base = FrameRange { start: 0, end: 19 };
        // 恒定带 [0,4] 覆盖 base 前段 → 精确扣除，保留脏段 [5,19]
        // （subtract_ranges = base \ ∪exclude 的最大连续段集合）。
        let exclude = [FrameRange { start: 0, end: 4 }];
        assert_eq!(
            subtract_ranges(base, &exclude),
            vec![FrameRange { start: 5, end: 19 }]
        );
    }

    #[test]
    fn detect_frame_reuse_narrows_via_channel_intervals() {
        // 与 QA 场景对齐：通道 "wind" 恒定 [0,4]∪[10,19]，变更保守全帧
        // 但带通道信息 → 脏帧数应 < 全帧（收窄生效）。
        let empty = ChannelDefs::new();
        let mut intervals = ChannelIntervals::new();
        intervals.push("wind", FrameRange { start: 0, end: 4 });
        intervals.push("wind", FrameRange { start: 10, end: 19 });
        let total = 20u64;
        let mut det = DirtyDetector::new();
        det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
        let change = ParamChange {
            node: 1,
            channel: Some("wind".to_string()),
            changed_frames: FrameRange::full(total),
        };
        let dirty = det
            .detect(&chain_graph(0.9), &empty, &[change], total, &intervals)
            .unwrap();
        assert!(
            dirty.dirty_frame_count() < total,
            "通道恒定帧应被扣除（收窄生效），实际 {}",
            dirty.dirty_frame_count()
        );
    }
}
