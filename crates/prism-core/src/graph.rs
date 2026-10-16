//! 图结构与固定形状归约代数。
//!
//! 语义权威：（固定形状归约：`reduction_shape(n)` 形状唯一 +
//! 工程四律：形状唯一 / 安全区优先 / 消序用排序 / 分布式同树）、
//! 六种组合方式与类图、D-26（归约代数立项）。
//!
//! - [`Graph`]：节点集合 + 连线，键序容器固定遍历序（同图同遍历序）。
//! - [`reduction_shape`] / [`ReductionShape`]：归约树形状是输入规模 n 的纯
//!   函数——**prism-core 单点定义**，任何归约点不得私定结合序（工程四律 1）。
//! - [`reduce`]：固定树形 + 固定结合序的通用归约入口，即
//!   "串行参考"——对同一形状树的固定顺序求值；并行/分布式实现必须按同一
//!   形状函数切分（工程四律 4），同 n 同树 ⇒ 任意执行拓扑逐位一致。
//! - [`reduce_unordered`] / [`reduce_sorted_by_key`]：工程四律 3"消序用排序"
//!   的入口——无序集合（含 `HashMap` 迭代序）先按稳定键排序再归约。
//! - [`ceil_log2`] / [`error_bound_coeff`]：形状契约与误差界的整数辅助件。
//!
//! 补充：正文**未**要求误差上界证明，但"树深 = ⌈log₂ n⌉"
//! 是规格明文，其数值推论（pairwise 误差 O(log n)）由 [`error_bound_coeff`]
//! 显式化并测试；实现覆盖全部 `n`（含 0 / 1 / 非 2 的幂 / 极大值）。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::node::{Composition, Node, NodeId};

// ---------------------------------------------------------------------------
// 固定形状归约代数
// ---------------------------------------------------------------------------

/// `ceil(log₂ n)`：大于等于 n 的最小 2 的幂的指数（n ≤ 1 时为 0）。
///
/// 规格明文"树深 ⌈log₂ n⌉"，本函数是该量的**整数纯函数**实现（无浮点、
/// 无 libm—— 格锚定要求判别量走整数运算，确定性是定理而非承诺）。
/// 供 [`ReductionShape::depth`] 的契约断言与误差界上界推导使用。
pub fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    // 位宽（不含符号位）；2 的幂精确等于其指数。
    usize::BITS as usize - (n - 1).leading_zeros() as usize
}

/// 固定形状归约树。层结构：每层相邻成对（pairwise），奇数尾元素直接进位
/// 下一层；层 k 的 pair 下标指向层 k-1 的槽位。形状是 n 的纯函数——同 n
/// 必得同树、同浮点结合序（核心构造）。
///
/// # 形状契约（核心构造的可验证面）
///
/// 规格把"形状 = n 的纯函数（与数据无关）"作为双保证的根基。本类型把该
/// 契约暴露为**可被测试机关直接断言**的结构面，而不仅是 `reduce` 的内部
/// 细节：
///
/// - [`ReductionShape::depth`] ≡ [`ceil_log2`]`(n)`（树深 ⌈log₂ n⌉）；
/// - [`ReductionShape::leaf_slots`] ≡ `n`（叶槽数 = 输入规模）；
/// - [`ReductionShape::carry_counts`] 逐层给出"奇数尾进位"的个数；
/// - [`ReductionShape::combine_count`] = 归约发生的结合次数 = `n - 1`（单株
///   二叉树的内部结点数，与树形无关的守恒量——可用于交叉校验）；
/// - [`ReductionShape::shape_key`] 把整个层结构编码为**确定性整数键**，供跨
///   规模一致性对拍（键只依赖 n，永不依赖数据）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionShape {
    /// 输入规模。
    n: usize,
    /// 各层结合对（`layers[k]` 的下标指向层 k-1 的槽位；进位尾元素不占对）。
    layers: Vec<Vec<[u32; 2]>>,
}

impl ReductionShape {
    /// 以输入规模构造归约树形状（同 n 必得同树）。
    pub fn new(n: usize) -> Self {
        let mut layers: Vec<Vec<[u32; 2]>> = Vec::new();
        let mut len = n;
        while len > 1 {
            let mut pairs = Vec::with_capacity(len / 2);
            let mut i = 0;
            while i + 1 < len {
                pairs.push([i as u32, i as u32 + 1]);
                i += 2;
            }
            layers.push(pairs);
            len = (len + 1) / 2; // 奇数尾元素进位下一层
        }
        Self { n, layers }
    }

    /// 输入规模 n。
    pub fn n(&self) -> usize {
        self.n
    }

    /// 树深 = ⌈log₂ n⌉（n ≤ 1 时为 0）。
    ///
    /// 契约：返回值恒等于 [`ceil_log2`]`(self.n())`（见单元测试
    /// `shape_depth_equals_ceil_log2`）。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 叶槽数 = 输入规模 n（形状第 0 层槽位数的形式化）。
    pub fn leaf_slots(&self) -> usize {
        self.n
    }

    /// 归约结合次数 = `n - 1`（n ≥ 1）。
    ///
    /// 单株二叉树的守恒量：每个非叶层级把若干对合并（每对恰一次 combine），
    /// 逐层求和恒为 `n - 1`，与**树形无关**——可作形状自洽的交叉校验：
    /// `Σ_layers(pairs[k].len()) == n - 1`。
    pub fn combine_count(&self) -> usize {
        self.n.saturating_sub(1)
    }

    /// 逐层"奇数尾进位"计数：第 `k` 层的输入长度若为奇数则该层有 1 个尾元素
    /// 直接进位（不参与本层 combine）。返回长度 = [`depth`](Self::depth)。
    ///
    /// 第 0 层输入长度 = n；第 k 层输入长度 = 层 k-1 的输出长度
    /// = `ceil(prev / 2)`。奇数 ⇒ 进位 1，偶数 ⇒ 进位 0。
    pub fn carry_counts(&self) -> Vec<usize> {
        let mut counts: Vec<usize> = Vec::with_capacity(self.layers.len());
        let mut len: usize = self.n;
        while len > 1 {
            counts.push(len & 1); // 奇数尾进位 1，偶数 0
            len = len.div_ceil(2); // (len + 1) / 2，与 new 的进位规则一致
        }
        counts
    }

    /// 形状的**确定性整数键**：把整棵层结构编码为一个 u64，只依赖 n。
    ///
    /// 供跨规模一致性对拍：同 n 必得同键（形状纯函数），不同树形必得不同键
    /// （碰撞概率极低且非语义依赖——键由逐对下标经 splitmix64 混洗累积，
    /// 只是形状的**指纹**，不是哈希表的键）。空形状（n ≤ 1）键为 `n`。
    pub fn shape_key(&self) -> u64 {
        // FNV-1a 风格 + splitmix64 终结：纯整数、无浮点、跨平台逐位一致。
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        let mix = |mut z: u64| -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        hash ^= mix(self.n as u64);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3); // FNV prime
        for (k, pairs) in self.layers.iter().enumerate() {
            hash ^= mix(k as u64);
            hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            for p in pairs {
                let packed: u64 = ((p[0] as u64) << 32) | (p[1] as u64);
                hash ^= mix(packed);
                hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        hash
    }

    /// 各层结合对（层 k 的下标指向层 k-1 的槽位）。
    pub fn layers(&self) -> &[Vec<[u32; 2]>] {
        &self.layers
    }
}

/// `reduction_shape(n)`：全引擎唯一归约树形状函数（
/// 格锚定规则表"求和/归约"行）。任何归约点一律经 [`reduce`] 走此形状。
pub fn reduction_shape(n: usize) -> ReductionShape {
    ReductionShape::new(n)
}

/// 固定形状归约的**误差上界**（浮点给证， 树深的直接推论）。
///
///  规定归约树是平衡 pairwise 二叉树、树深 ⌈log₂ n⌉。据此 Higham 的经典
/// 结果给出：n 项浮点求和的向前误差满足
///
/// ```text
/// |ŝ − s| ≤ (n − 1) · u · Σ|aᵢ| / (1 − (n − 1) · u)   （一般界）
/// |ŝ − s| ≤ ⌈log₂ n⌉ · u · Σ|aᵢ| / (1 − ⌈log₂ n⌉ · u)（pairwise 树界）
/// ```
///
/// 其中 `u` = 单位舍入（f32 为 `2⁻²⁴`，即 `f32::EPSILON / 2`）。pairwise 树的
/// 界从 `O(n)` 降到 `O(log n)`——这正是选平衡树的数值理由。本函数返回
/// **pairwise 界的分子系数** `⌈log₂ n⌉`（供调用方乘 `u · Σ|aᵢ|`），把规格
/// 的结构推论显式化、可测试（见 `pairwise_error_bound_holds_empirically`）。
///
/// n ≤ 1 时系数为 0（无归约、无误差）。
pub fn error_bound_coeff(n: usize) -> usize {
    ceil_log2(n)
}

/// **律三"消序用排序"的归约入口**（工程四律 3）。
///
/// 无序集合（碰撞对 / 邻居列表 / `HashMap` 迭代序）必须先按稳定键排序再处理，
/// 否则归约项的**顺序**随执行环境漂移，即使形状固定，喂给树的序列也不同
/// ⇒ 结合序被打乱 ⇒ 结果漂移。本函数把该纪律固化为 API：调用方提供
/// `(key, value)` 对，函数**按键升序稳定排序**后交给 [`reduce`]——键相同时
/// 保持原相对序（稳定排序），保证"排序是输入多重集的纯函数"。
///
/// `key` 的类型 `K: Ord` 决定排序；`combine` / `unit` 语义同 [`reduce`]。
pub fn reduce_sorted_by_key<T, K, F>(
    keyed: Vec<(K, T)>,
    key: impl Fn(&K) -> K,
    combine: F,
    unit: T,
) -> T
where
    K: Ord,
    F: Fn(T, T) -> T,
{
    let mut pairs: Vec<(K, T)> = keyed;
    // 稳定排序：键相同保持原相对序 ⇒ 排序结果 = 输入多重集的纯函数。
    pairs.sort_by(|a, b| key(&a.0).cmp(&key(&b.0)));
    let items: Vec<T> = pairs.into_iter().map(|(_, v)| v).collect();
    reduce(items, combine, unit)
}

/// 律三的**集合入口**：直接吃无序 `HashMap`/迭代器产物，按键排序后归约。
///
/// 入参 `items` 为 `(key, value)` 序列，其**顺序不受信任**（可能来自
/// `HashMap::iter`）。本函数先按键升序稳定排序——这是明文禁止"以
/// `HashMap` 迭代顺序决定归约顺序"的直接落地。
pub fn reduce_unordered<T, K, F>(items: impl IntoIterator<Item = (K, T)>, combine: F, unit: T) -> T
where
    K: Ord,
    F: Fn(T, T) -> T,
{
    let mut pairs: Vec<(K, T)> = items.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let ordered: Vec<T> = pairs.into_iter().map(|(_, v)| v).collect();
    reduce(ordered, combine, unit)
}

/// 固定形状归约（通用入口）：树形由 [`reduction_shape`] 唯一
/// 给定，结合序固定——层序推进、层内左到右、左操作数在前。
///
/// 这就是"串行参考"：对同一形状树的固定顺序求值。并行调度只是让层内
/// 各对独立执行（互不改变结合序），因此任意执行拓扑与串行逐位一致。
/// `unit` 仅在 `items` 为空时返回（空集归约 = 单位元）。
pub fn reduce<T>(items: Vec<T>, combine: impl Fn(T, T) -> T, unit: T) -> T {
    let shape = reduction_shape(items.len());
    let mut layer = items;
    let mut shape_layers = shape.layers().iter();
    while layer.len() > 1 {
        let pairs = shape_layers.next().expect("归约进度与形状层数一致");
        debug_assert_eq!(pairs.len(), layer.len() / 2, "层尺寸与形状一致");
        let mut next: Vec<T> = Vec::with_capacity((layer.len() + 1) / 2);
        let mut rest = layer.into_iter();
        while let Some(a) = rest.next() {
            match rest.next() {
                Some(b) => next.push(combine(a, b)),
                // 奇数尾元素直接进位下一层（形状规则）。
                None => next.push(a),
            }
        }
        layer = next;
    }
    layer.pop().unwrap_or(unit)
}

// ---------------------------------------------------------------------------
// 图结构（类图 Graph 侧）
// ---------------------------------------------------------------------------

/// 图结构错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含建议段）。
/// 错误码取 P3 段（图结构，宪章 C18 分段）。
#[derive(Debug, Clone, PartialEq)]
pub struct GraphError {
    /// 错误码（P3xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】：发生位置。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for GraphError {}

/// 节点图：节点集合 + 连线，全部 BTreeMap 固定键序（同图必得同遍历序）。
///
/// 连线以 `(目标节点, 输入槽位) → 源节点` 建索引：槽位号只定多输入归约的
/// 顺序（升序），同一槽位只允许一条入边。连线端点在 [`Graph::connect`] 时
/// 即时校验（悬空连线结构化报错，永不静默，宪章 C20）。
#[derive(Debug, Clone, Default)]
pub struct Graph {
    nodes: BTreeMap<NodeId, Node>,
    edges: BTreeMap<(NodeId, u8), NodeId>,
}

impl Graph {
    /// 构造空图。
    pub fn new() -> Self {
        Self::default()
    }

    /// 加入节点。重复 id 拒绝（P3008，三段式）。
    pub fn add_node(&mut self, node: Node) -> Result<(), GraphError> {
        if self.nodes.contains_key(&node.id) {
            return Err(GraphError {
                code: "P3008",
                location: format!("graph.add_node(节点 {})", node.id),
                problem: format!("节点 id {} 已存在，重复加入", node.id),
                fix: format!(
                    "换用未占用的节点 id；当前已用 id：{:?}",
                    self.nodes.keys().collect::<Vec<_>>()
                ),
            });
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// 连线 `from → (to 的第 input 号输入槽位)`。
    ///
    /// 即时校验：端点必须已加入图（悬空 → P3006）、自环 → P3005、槽位
    /// 重复占用 → P3007、Feed 节点多入边 → P3007（Feed 是单进单出连线，
    /// 多输入请用 `Merge(op)`）。
    pub fn connect(&mut self, from: NodeId, to: NodeId, input: u8) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&from) {
            return Err(GraphError {
                code: "P3006",
                location: format!("graph.connect({from} → {to})"),
                problem: format!("源节点 {from} 不存在：连线悬空（端点必须先加入图）"),
                fix: format!("先 add_node 加入节点 {from}，再执行连线"),
            });
        }
        if !self.nodes.contains_key(&to) {
            return Err(GraphError {
                code: "P3006",
                location: format!("graph.connect({from} → {to})"),
                problem: format!("目标节点 {to} 不存在：连线悬空（端点必须先加入图）"),
                fix: format!("先 add_node 加入节点 {to}，再执行连线"),
            });
        }
        if from == to {
            return Err(GraphError {
                code: "P3005",
                location: format!("graph.connect({from} → {to})"),
                problem: "自环连线：节点不允许连接自身".to_string(),
                fix: "检查连线方向；环依赖请拆解为多个节点".to_string(),
            });
        }
        if let Some(&old) = self.edges.get(&(to, input)) {
            return Err(GraphError {
                code: "P3007",
                location: format!("graph.connect({from} → ({to}, 槽位 {input}))"),
                problem: format!("目标槽位已被占用：节点 {to} 的输入槽位 {input} 已连自节点 {old}"),
                fix: format!("改用空闲槽位，或先断开槽位 {input} 的既有连线"),
            });
        }
        if self.nodes.get(&to).map(|n| &n.composition) == Some(&Composition::Feed)
            && self.inputs_of(to).len() >= 1
        {
            return Err(GraphError {
                code: "P3007",
                location: format!("graph.connect({from} → {to})"),
                problem: "Feed 节点已有一条入边：Feed 是单进单出连线，不接受第二条入边"
                    .to_string(),
                fix: "多输入请把目标节点的组合方式改为 Composition::Merge(op)".to_string(),
            });
        }
        self.edges.insert((to, input), from);
        Ok(())
    }

    /// 目标节点 `to` 的上游列表（输入槽位升序）。
    pub fn inputs_of(&self, to: NodeId) -> Vec<(u8, NodeId)> {
        self.edges
            .range((to, u8::MIN)..=(to, u8::MAX))
            .map(|(&(_, input), &from)| (input, from))
            .collect()
    }

    /// 拓扑序（Kahn 算法）：就绪集按 node_id 升序取——**稳定 tie-break 按
    /// node_id**（求值顺序确定性）。环 → P3005，报错列出无法
    /// 入序的节点（参与环或被环阻塞）。
    pub fn topo_order(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut indegree: BTreeMap<NodeId, usize> =
            self.nodes.keys().map(|&id| (id, 0)).collect();
        let mut downstream: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for (&(to, _), &from) in &self.edges {
            *indegree.get_mut(&to).expect("连线端点已在图内（connect 校验）") += 1;
            downstream.entry(from).or_default().push(to);
        }

        let mut ready: BTreeSet<NodeId> = indegree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(&id) = ready.iter().next() {
            ready.remove(&id);
            order.push(id);
            if let Some(tos) = downstream.get(&id) {
                for &to in tos {
                    let d = indegree.get_mut(&to).expect("下游节点已在图内");
                    *d -= 1;
                    if *d == 0 {
                        ready.insert(to);
                    }
                }
            }
        }

        if order.len() < self.nodes.len() {
            let stuck: Vec<String> = indegree
                .iter()
                .filter(|(_, &d)| d > 0)
                .map(|(id, _)| id.to_string())
                .collect();
            return Err(GraphError {
                code: "P3005",
                location: "graph.topo_order（拓扑校验）".to_string(),
                problem: format!(
                    "检测到环：{} 个节点无法入序（参与环或被环阻塞）：{}",
                    stuck.len(),
                    stuck.join(", ")
                ),
                fix: "检查图中连线，移除构成环的一条连线后重新求拓扑序".to_string(),
            });
        }
        Ok(order)
    }

    /// 按节点 id 读取节点。
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// 全部节点（id 升序）。
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// 节点数。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否为空图。
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 是否包含节点 id。
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// 连线数。
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{MergeOp, ParamValue, StructureId};

    #[test]
    fn reduction_shape_is_pure_function_of_n() {
        // 同 n 同树（形状唯一，工程四律 1）。
        assert_eq!(reduction_shape(5), reduction_shape(5));
        assert_eq!(reduction_shape(8), reduction_shape(8));
        assert_ne!(reduction_shape(5), reduction_shape(6));

        // n=5：层 0 成对 (0,1)(2,3)，尾 4 进位；层 1 (0,1)，尾 2 进位；层 2 (0,1)。
        let s5 = reduction_shape(5);
        assert_eq!(s5.depth(), 3, "⌈log₂ 5⌉ = 3");
        assert_eq!(s5.layers()[0], vec![[0, 1], [2, 3]]);
        assert_eq!(s5.layers()[1], vec![[0, 1]]);
        assert_eq!(s5.layers()[2], vec![[0, 1]]);

        // n=8：完全平衡二叉树，4 对 → 2 对 → 1 对。
        let s8 = reduction_shape(8);
        assert_eq!(s8.depth(), 3);
        assert_eq!(s8.layers()[0], vec![[0, 1], [2, 3], [4, 5], [6, 7]]);
        assert_eq!(s8.layers()[1], vec![[0, 1], [2, 3]]);
        assert_eq!(s8.layers()[2], vec![[0, 1]]);

        // 边界：n=1 无归约层；n=0 同理（空集归约 = 单位元，见 reduce）。
        assert_eq!(reduction_shape(1).depth(), 0);
        assert_eq!(reduction_shape(0).depth(), 0);
        assert_eq!(reduction_shape(2).layers(), vec![vec![[0, 1]]]);
    }

    #[test]
    fn reduce_follows_pairwise_tree_association_order() {
        // 结构断言：n=5 的结合序 = pairwise 树形（尾元素最后进位合入）。
        let items: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tree = reduce(items, |a, b| format!("({a}+{b})"), String::new());
        assert_eq!(tree, "(((a+b)+(c+d))+e)");

        // n=8：完全平衡。
        let items8: Vec<String> = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tree8 = reduce(items8, |a, b| format!("({a}+{b})"), String::new());
        assert_eq!(tree8, "(((a+b)+(c+d))+((e+f)+(g+h)))");

        // 空集归约 = 单位元；单元素原样返回。
        assert_eq!(reduce(Vec::<String>::new(), |a, b| a + &b, "unit".into()), "unit");
        assert_eq!(reduce(vec![7.5f32], |a, b| a + b, 0.0), 7.5);
    }

    #[test]
    fn parallel_scheduling_of_same_shape_matches_serial_bitwise() {
        // 浮点求和场景（结合序敏感）：37 个值覆盖奇数尾进位路径。
        let vals: Vec<f32> = (0..37).map(|i| 0.1 * (i as f32) + 0.05).collect();
        let serial = reduce(vals.clone(), |a, b| a + b, 0.0_f32);

        // 按 reduction_shape 分层并行：层内各对相互独立，spawn 顺序刻意倒序
        // 模拟任意执行时序——形状固定 ⇒ 与串行参考逐位一致。
        let shape = reduction_shape(vals.len());
        let mut layer: Vec<f32> = vals;
        for pairs in shape.layers() {
            let mut slots: Vec<Option<f32>> = vec![None; pairs.len()];
            std::thread::scope(|s| {
                let handles: Vec<_> = pairs
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(i, &[a, b])| {
                        let (x, y) = (layer[a as usize], layer[b as usize]);
                        s.spawn(move || (i, x + y))
                    })
                    .collect();
                for h in handles {
                    let (i, v) = h.join().expect("并行归约线程不 panic");
                    slots[i] = Some(v);
                }
            });
            let carry_from = pairs.len() * 2;
            let mut next: Vec<f32> = slots.into_iter().map(|v| v.expect("每对必产出")).collect();
            next.extend_from_slice(&layer[carry_from..]);
            layer = next;
        }

        assert_eq!(
            serial.to_bits(),
            layer[0].to_bits(),
            "并行 ≡ 串行参考，逐位一致（f32 位模式）"
        );
    }

    #[test]
    fn topo_order_is_deterministic_with_id_tiebreak() {
        let mut g = Graph::new();
        for (id, structure) in [
            (10u64, StructureId::ScalarField),
            (2u64, StructureId::Diffusion),
            (7u64, StructureId::Convolution),
        ] {
            g.add_node(Node::new(
                id,
                structure,
                Composition::Merge(MergeOp::Plus),
            ))
            .unwrap();
        }
        // 插入序 10 在前；就绪集 tie-break 按 id 升序 ⇒ 先出 2。
        g.connect(10, 7, 0).unwrap();
        g.connect(2, 7, 1).unwrap();
        assert_eq!(g.topo_order().unwrap(), vec![2, 10, 7]);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.inputs_of(7), vec![(0, 10), (1, 2)], "上游按槽位升序");
        assert_eq!(g.inputs_of(2), vec![]);
    }

    #[test]
    fn cycle_detection_reports_cyclic_nodes_with_advice() {
        let mut g = Graph::new();
        for id in [1u64, 2u64, 3u64] {
            g.add_node(Node::new(id, StructureId::ScalarField, Composition::Feed))
                .unwrap();
        }
        g.connect(1, 2, 0).unwrap();
        g.connect(2, 3, 0).unwrap();
        g.connect(3, 1, 0).unwrap();

        let err = g.topo_order().unwrap_err();
        assert_eq!(err.code, "P3005");
        let text = err.to_string();
        assert!(text.starts_with("error[P3005] "), "首段应在哪：{}", text);
        assert!(text.contains("检测到环"), "什么事段应指出环：{}", text);
        assert!(text.contains('1') && text.contains('2') && text.contains('3'),
            "应列出环中节点：{}", text);
        assert!(text.contains("建议："), "必须含建议段（C19）：{}", text);
    }

    #[test]
    fn dangling_and_conflicting_edges_are_rejected() {
        let mut g = Graph::new();
        g.add_node(Node::new(1, StructureId::ScalarField, Composition::Feed))
            .unwrap();
        g.add_node(Node::new(
            2,
            StructureId::EntityRepresentation,
            Composition::Merge(MergeOp::Plus),
        ))
        .unwrap();

        // 悬空连线：任一端点未加入图 → P3006。
        assert_eq!(g.connect(99, 1, 0).unwrap_err().code, "P3006");
        assert_eq!(g.connect(1, 99, 0).unwrap_err().code, "P3006");
        assert_eq!(g.edge_count(), 0);

        // 自环 → P3005。
        assert_eq!(g.connect(1, 1, 0).unwrap_err().code, "P3005");

        // 槽位重复占用 → P3007。
        g.connect(1, 2, 0).unwrap();
        assert_eq!(g.connect(1, 2, 0).unwrap_err().code, "P3007");

        // Feed 节点第二条入边 → P3007（单进单出）。
        let mut g2 = Graph::new();
        for id in [1u64, 2u64, 3u64] {
            g2.add_node(Node::new(id, StructureId::ScalarField, Composition::Feed))
                .unwrap();
        }
        g2.connect(1, 3, 0).unwrap();
        let err = g2.connect(2, 3, 1).unwrap_err();
        assert_eq!(err.code, "P3007");
        assert!(err.problem.contains("单进单出"), "应指出 Feed 语义：{}", err.problem);
    }

    #[test]
    fn duplicate_node_id_is_rejected() {
        let mut g = Graph::new();
        g.add_node(Node::new(4, StructureId::ScalarField, Composition::Feed))
            .unwrap();
        let err = g
            .add_node(Node::new(4, StructureId::Diffusion, Composition::Feed))
            .unwrap_err();
        assert_eq!(err.code, "P3008");
        assert_eq!(g.len(), 1);
        assert!(g.contains(4));
        assert!(g.node(4).is_some());
        assert_eq!(g.node(4).unwrap().params.len(), 0);
        // 参数包仍可在节点上装配（节点三元组完整可用）。
        let mut with_params = Node::new(5, StructureId::ScalarField, Composition::Feed);
        with_params.params.set("value", ParamValue::F32(1.5));
        g.add_node(with_params).unwrap();
        assert_eq!(g.nodes().count(), 2);
        assert!(!g.is_empty());
    }

    // -----------------------------------------------------------------------
    //  形状契约 + 缺口补测
    // -----------------------------------------------------------------------

    #[test]
    fn ceil_log2_is_exact_integer_bound() {
        // 规格："树深 ⌈log₂ n⌉"。逐点对拍整数纯函数。
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(7), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(1024), 10);
        assert_eq!(ceil_log2(1025), 11);
        // 极大值不溢出（usize::MAX 的 ⌈log₂⌉ = 位宽）。
        assert_eq!(ceil_log2(usize::MAX), usize::BITS as usize);
    }

    #[test]
    fn shape_depth_equals_ceil_log2_across_scales() {
        // 形状契约 1：depth() ≡ ceil_log2(n)（规格明文，逐点成立）。
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 100, 1000, 1024, 4096, 65536] {
            assert_eq!(
                reduction_shape(n).depth(),
                ceil_log2(n),
                "n={n}：树深必须等于 ⌈log₂ n⌉"
            );
        }
    }

    #[test]
    fn shape_leaf_slots_and_combine_count_conserved() {
        // 形状契约 2/3：叶槽 = n；结合次数 = n−1 且 = Σ 各层对数（守恒量）。
        for n in [1usize, 2, 3, 5, 8, 37, 100, 1000, 1024] {
            let s = reduction_shape(n);
            assert_eq!(s.leaf_slots(), n, "n={n}：叶槽数须为 n");
            assert_eq!(s.combine_count(), n - 1, "n={n}：结合次数须为 n−1");
            let summed: usize = s.layers().iter().map(|v| v.len()).sum();
            assert_eq!(summed, n - 1, "n={n}：Σ各层对数须为 n−1（守恒量）");
            // 各层对数 = ⌊该层输入长度 / 2⌋。
            let mut len = n;
            for (k, pairs) in s.layers().iter().enumerate() {
                assert_eq!(pairs.len(), len / 2, "n={n} 层 {k}：对数须为 ⌊len/2⌋");
                len = len.div_ceil(2);
            }
        }
    }

    #[test]
    fn carry_counts_match_odd_tail_layers() {
        // 形状契约 4：进位计数逐层 = 该层输入长度的奇偶（规格"奇数尾进位"）。
        // n=5：层0 len5 奇→1；层1 len3 奇→1；层2 len2 偶→0。
        assert_eq!(reduction_shape(5).carry_counts(), vec![1, 1, 0]);
        // n=8：全偶，无进位。
        assert_eq!(reduction_shape(8).carry_counts(), vec![0, 0, 0]);
        // n=1 / n=0：无层。
        assert_eq!(reduction_shape(1).carry_counts(), Vec::<usize>::new());
        assert_eq!(reduction_shape(0).carry_counts(), Vec::<usize>::new());
        // 一般校验：进位计数长度 = depth，且每位 ∈ {0,1}。
        for n in [3usize, 6, 7, 13, 37, 100] {
            let s = reduction_shape(n);
            let cc = s.carry_counts();
            assert_eq!(cc.len(), s.depth(), "n={n}：进位计数长度须等于树深");
            assert!(cc.iter().all(|&c| c <= 1), "n={n}：进位计数须为 0/1");
        }
    }

    #[test]
    fn shape_is_data_independent_and_key_is_pure_function_of_n() {
        // 缺口补测（规格核心构造）：形状完全由 n 决定、与数据无关。
        // 同一 n 多次构造（"不同数据"映射到同一 n）必得同键；不同 n 树形不同。
        for n in [1usize, 2, 3, 5, 8, 37, 1000, 1024] {
            let a = reduction_shape(n);
            let b = reduction_shape(n);
            assert_eq!(a, b, "n={n}：同 n 必得同树（形状唯一）");
            assert_eq!(a.shape_key(), b.shape_key(), "n={n}：同 n 必得同形状键");
        }
        // 不同 n 的形状键不同（键是形状的可靠指纹；n=5 vs 6 树形不同）。
        assert_ne!(reduction_shape(5).shape_key(), reduction_shape(6).shape_key());
        assert_ne!(reduction_shape(7).shape_key(), reduction_shape(8).shape_key());
        // ⌈log₂⌉ 相同但 n 不同（4 vs 5/6/7/8）——键仍区分（键抓的是具体树形，
        // 不只是深度），保证"跨规模一致性"对拍能识别不同树。
        assert_ne!(reduction_shape(4).shape_key(), reduction_shape(5).shape_key());
        assert_ne!(reduction_shape(5).shape_key(), reduction_shape(6).shape_key());
        // n ≤ 1 退化为空形状：无层。
        assert_eq!(reduction_shape(0).layers().len(), 0);
        assert_eq!(reduction_shape(1).layers().len(), 0);
    }

    #[test]
    fn same_n_different_data_yields_same_tree_and_same_float_association() {
        // 缺口补测（关键）：两个**不同数据**但**同 n** 的归约，必须走同一棵树
        // ⇒ 相同结合序。用"求和结果 == 按树的固定顺序手工求值"证明结合序一致。
        // 构造两组完全不同的值，各自手工按 reduction_shape(5) 树求值，二者
        // 与 reduce 结果逐位一致（形状与数据解耦）。
        let set_a: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let set_b: Vec<f32> = vec![1e7, -1e7, 3.5, 1e-7, 2.25];
        let n = 5usize;
        // 手工按 shape 树驱动一遍（与 reduce 同序）：层0 (0+1)(2+3) 尾4进位；
        // 层1 (0+1) 尾2进位；层2 (0+1)。
        let manual = |v: &[f32]| -> f32 {
            let s = reduction_shape(n);
            let mut layer: Vec<f32> = v.to_vec();
            for pairs in s.layers() {
                let mut next: Vec<f32> = Vec::new();
                for p in pairs {
                    next.push(layer[p[0] as usize] + layer[p[1] as usize]);
                }
                let carry_from = pairs.len() * 2;
                next.extend_from_slice(&layer[carry_from..]);
                layer = next;
            }
            layer[0]
        };
        assert_eq!(
            reduce(set_a.clone(), |a, b| a + b, 0.0_f32).to_bits(),
            manual(&set_a).to_bits(),
            "组 A：reduce 结合序须等于 shape 树固定序"
        );
        assert_eq!(
            reduce(set_b.clone(), |a, b| a + b, 0.0_f32).to_bits(),
            manual(&set_b).to_bits(),
            "组 B：reduce 结合序须等于 shape 树固定序"
        );
        // 形状与数据无关：同 n ⇒ 同键（两组数据喂同一棵树）。
        assert_eq!(reduction_shape(set_a.len()).shape_key(), reduction_shape(set_b.len()).shape_key());
    }

    #[test]
    fn pairwise_error_bound_holds_empirically() {
        // 缺口补测（规格树深的数值推论）：pairwise 树的求和误差满足
        // |ŝ − s| ≤ ⌈log₂ n⌉ · u · Σ|aᵢ| / (1 − ⌈log₂ n⌉ · u)。
        // 用 f64 高精度参考值对拍 f32 reduce，验证上界成立（浮点给证）。
        let u_f32: f64 = (f32::EPSILON as f64) / 2.0;
        for n in [3usize, 5, 8, 37, 100, 1000] {
            // 结合序敏感值：大正大负交替 + 小数，制造消去误差。
            let vals: Vec<f32> = (0..n)
                .map(|i| {
                    let s = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                    s * (1.0 + (i as f32) * 0.001) - 1.0 + (i as f32) * 1e-4
                })
                .collect();
            let got: f32 = reduce(vals.clone(), |a, b| a + b, 0.0_f32);
            // f64 精确参考（f64 累加误差远低于 f32 上界，可作真值近似）。
            let exact: f64 = vals.iter().map(|&v| v as f64).sum();
            let abs_sum: f64 = vals.iter().map(|&v| (v as f64).abs()).sum();
            let err: f64 = (got as f64 - exact).abs();
            let coeff = error_bound_coeff(n) as f64;
            let bound: f64 = coeff * u_f32 * abs_sum / (1.0 - coeff * u_f32);
            assert!(
                err <= bound,
                "n={n}：|err|={err:e} 超过 pairwise 误差上界 {bound:e}（coeff=⌈log₂n⌉={coeff}）"
            );
        }
    }

    #[test]
    fn reduce_unordered_sorts_key_before_association() {
        // 律三"消序用排序"：无序输入的归约顺序须由**键排序**决定，
        // 而非迭代顺序。同一多重集任意枚举顺序 ⇒ 同结果（逐位）。
        let items: Vec<(u32, f32)> =
            vec![(3, 0.3), (0, 0.1), (4, 0.5), (1, 0.2), (2, 0.4)];
        let canonical = reduce_unordered(items.clone(), |a, b| a + b, 0.0_f32);
        // 打乱（模拟 HashMap 迭代序不稳定）多次，结果必须逐位相同。
        let shuffles: Vec<Vec<(u32, f32)>> = vec![
            vec![(4, 0.5), (3, 0.3), (2, 0.4), (1, 0.2), (0, 0.1)],
            vec![(0, 0.1), (2, 0.4), (4, 0.5), (1, 0.2), (3, 0.3)],
            vec![(2, 0.4), (0, 0.1), (3, 0.3), (4, 0.5), (1, 0.2)],
        ];
        for sh in shuffles {
            assert_eq!(
                reduce_unordered(sh, |a, b| a + b, 0.0_f32).to_bits(),
                canonical.to_bits(),
                "无序输入的归约必须按键排序后走同一棵树（逐位）"
            );
        }
        // 键排序等价于先按键有序再 reduce。
        let sorted_vals: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(
            canonical.to_bits(),
            reduce(sorted_vals, |a, b| a + b, 0.0_f32).to_bits(),
            "按键排序后的归约须等于有序 reduce"
        );
    }

    #[test]
    fn reduce_sorted_by_key_matches_manual_sort() {
        // reduce_sorted_by_key 与"手工排序 + reduce"逐位一致。
        let keyed: Vec<(u32, f32)> = vec![(10, 1.0), (2, 2.0), (7, 3.0), (1, 4.0)];
        let via_helper = reduce_sorted_by_key(keyed.clone(), |k| *k, |a, b| a + b, 0.0_f32);
        let mut manual = keyed.clone();
        manual.sort_by_key(|(k, _)| *k);
        let manual_vals: Vec<f32> = manual.into_iter().map(|(_, v)| v).collect();
        assert_eq!(
            via_helper.to_bits(),
            reduce(manual_vals, |a, b| a + b, 0.0_f32).to_bits()
        );
    }

    #[test]
    fn block_prereduction_changes_result_versus_global_tree() {
        // ===== 反向测试（本路核心证据）=====
        // 证明纪律"禁止 worker 块内预归约"**真的有效**，不是空话：
        // 让 worker 在块内先把部分和归约掉 → 主节点看到的是 k 个块值 ⇒
        // 走 reduction_shape(k) 而非 reduction_shape(n) ⇒ **结果不同**。
        //
        // 取明文的 n=5 反例场景（规格："n=5 即有反例"）。
        let n = 5usize;
        // 该组值经实测确为反例：全局 pairwise 树（保序）得 9.0000010e7，
        // 块内预归约（前 2 块 + 后 3 块）在 4e7 处吞掉末位小数 → 9.0000000e7，
        // 逐位相差 2 个 ulp（差值 10，远超舍入噪声）。
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0e7, 5.0e7];
        assert_eq!(vals.len(), n, "反例场景须恰为 n=5 个明文");

        // 正确路径：全局一次 reduce(n=5)。
        let correct = reduce(vals.clone(), |a, b| a + b, 0.0_f32);

        // 违规路径：把 5 项切成 2 块（前 2 项、后 3 项），各块内先预归约成 1 个
        // 值，再对 2 个块值做 reduce(n=2)。
        let block0 = reduce(vals[0..2].to_vec(), |a, b| a + b, 0.0_f32);
        let block1 = reduce(vals[2..5].to_vec(), |a, b| a + b, 0.0_f32);
        let prereduced = reduce(vec![block0, block1], |a, b| a + b, 0.0_f32);

        // 两条路径树形不同（n=5 vs n=2）⇒ 结果**必须不同**（逐位）。
        assert_ne!(
            correct.to_bits(),
            prereduced.to_bits(),
            "反向测试失败：块内预归约竟与全局树同结果——纪律将形同虚设"
        );
        // 且证明形状确实不同（键不同），把"为什么不同"钉在形状层。
        assert_ne!(
            reduction_shape(5).shape_key(),
            reduction_shape(2).shape_key(),
            "n=5 与 n=2 必为不同树（反向测试的形状根据）"
        );
        // 精确值对照：全局 pairwise 保住了小数（9.0000010e7），块内预归约
        // 提前吞掉末位（9.0000000e7）——结合序差异被放大成逐位可见的漂移。
        assert_eq!(correct.to_bits(), (9.0000010e7_f32).to_bits());
        assert_eq!(prereduced.to_bits(), (9.0000000e7_f32).to_bits());
        // 差值 10（≈ 2 ulp）——远大于"数值巧合相等"的可能。
        assert!((correct - prereduced).abs() >= 8.0, "反例差值应远大于舍入噪声");
    }

    #[test]
    fn layered_parallel_equals_serial_for_all_topologies() {
        // 跨规模一致性 + 任意并行度：分层并行（模拟任意执行拓扑）与串行参考
        // 逐位一致。含非 2 的幂（3/5/7/37/100）与 2 的幂（8/1024）。
        for n in [1usize, 2, 3, 5, 7, 8, 37, 100, 1024] {
            let vals: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32) + 0.05).collect();
            let serial = reduce(vals.clone(), |a, b| a + b, 0.0_f32);
            // 按 shape 层"并行"：层内各对逆序 spawn（模拟任意时序）。
            let shape = reduction_shape(vals.len());
            let mut layer: Vec<f32> = vals;
            for pairs in shape.layers() {
                let mut slots: Vec<Option<f32>> = vec![None; pairs.len()];
                std::thread::scope(|s| {
                    let handles: Vec<_> = pairs
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(i, &[a, b])| {
                            let (x, y) = (layer[a as usize], layer[b as usize]);
                            s.spawn(move || (i, x + y))
                        })
                        .collect();
                    for h in handles {
                        let (i, v) = h.join().expect("并行归约线程不 panic");
                        slots[i] = Some(v);
                    }
                });
                let carry_from = pairs.len() * 2;
                let mut next: Vec<f32> = slots.into_iter().map(|v| v.expect("每对必产出")).collect();
                next.extend_from_slice(&layer[carry_from..]);
                layer = next;
            }
            assert_eq!(
                serial.to_bits(),
                layer[0].to_bits(),
                "n={n}：任意并行拓扑须与串行参考逐位一致"
            );
        }
    }
}
