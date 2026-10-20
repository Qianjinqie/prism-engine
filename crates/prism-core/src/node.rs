//! 节点模型 v2 数据结构。
//!
//! 语义权威：节点三元组 / 三问接口 / 类图与
//! D-22（数学结构第一性）、D-27（registry! 静态汇入制）、D-28（EvalCtx 快照库
//! + SampleIndex 像素域）。
//!
//! 本文件只落数据结构与接口：节点三元组（结构 StructureId、特解参数 Params、
//! 组合方式 Composition）、三问元数据 StructureMeta、求值器统一接口
//! `MathStructure`（作用/求逆/采样三操作）、求值上下文 `EvalCtx` 与
//! `SampleIndex`。图与求值调度（graph.rs / eval.rs）属第二批。

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use prism_math::Vec3;

/// 256 位哈希值（S10 层次签名的摘要类型）。哈希算法本体（SHA-256，A6）在
/// `signature.rs` 落地；本批仅定型类型供快照键使用。
pub type Hash256 = [u8; 32];

/// 节点唯一标识。节点 id 生活在格 Zᵏ 上，直接用整数。
pub type NodeId = u64;

// ---------------------------------------------------------------------------
// 结构标识 StructureId（官方段 S1–S12 + 用户段 S13+）
// ---------------------------------------------------------------------------

/// 数学结构标识（D-22：节点三元组的"结构"位）。
///
/// 官方段 1–12 对应十二结构总表；用户段 `Custom` 自 S13 起
/// （L4），`crate_name` 由 `registry!` 宏在定义所在 crate 的展开点
/// 自动捕获（`env!("CARGO_CRATE_NAME")`），同编号跨 crate 天然隔离冲突。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructureId {
    /// S1 标量场（协方差核随机场：噪声族 / SDF 场 / 密度场）
    ScalarField,
    /// S2 卷积（调和分析核：blur/bloom / 插值 / shutter 核）
    Convolution,
    /// S3 约束梯度流（近端算子不动点：XPBD 全家）
    ConstraintFlow,
    /// S4 确定性抽样（样本↔测度配对：Sobol+Owen / PCG）
    DeterministicSampling,
    /// S5 重参数化（变量替换+方程求根：MNEE / 时间映射 / UV warp）
    Reparameterization,
    /// S6 谱分解（傅里叶对角化：FFT 节拍 / DCT）
    SpectralDecomposition,
    /// S7 闭式线性响应（格林函数：Spring 解析解 / 包络）
    ClosedFormResponse,
    /// S8 辐射传输积分（转移核：路径追踪 / SSS / 体积介质）
    RadiativeTransfer,
    /// S9 分段多项式（再生核：track/缓动 / ACES 拟合 / Catmull-Rom）
    PiecewisePolynomial,
    /// S10 层次+签名（内容↔摘要：BVH / brick / 签名树）
    HierarchySignature,
    /// S11 扩散（热核：降噪 / halation 渗光链）
    Diffusion,
    /// S12 实体表示（边界↔体对偶：mesh/SDF/glTF / bake）
    EntityRepresentation,
    /// 用户段 S13+：`n` 由贡献者声明（必须 ≥13），`crate_name` 隔离跨 crate 冲突。
    Custom {
        /// 自定义编号（S13 起算，即 n ≥ 13）。
        n: u16,
        /// 定义所在 crate 名（宏展开点自动捕获）。
        crate_name: &'static str,
    },
}

impl StructureId {
    /// 官方段结构总数（S1–S12）。
    pub const OFFICIAL_COUNT: u16 = 12;

    /// 构造用户段结构标识（供 `registry!` 展开与贡献者代码使用）。
    pub const fn custom(n: u16, crate_name: &'static str) -> Self {
        Self::Custom { n, crate_name }
    }

    /// 官方段序号（S1–S12 → `Some(1..=12)`）；用户段返回 `None`。
    pub fn ordinal(&self) -> Option<u16> {
        match self {
            Self::ScalarField => Some(1),
            Self::Convolution => Some(2),
            Self::ConstraintFlow => Some(3),
            Self::DeterministicSampling => Some(4),
            Self::Reparameterization => Some(5),
            Self::SpectralDecomposition => Some(6),
            Self::ClosedFormResponse => Some(7),
            Self::RadiativeTransfer => Some(8),
            Self::PiecewisePolynomial => Some(9),
            Self::HierarchySignature => Some(10),
            Self::Diffusion => Some(11),
            Self::EntityRepresentation => Some(12),
            Self::Custom { .. } => None,
        }
    }
}

impl fmt::Display for StructureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Custom { n, crate_name } => write!(f, "S{}@{}", n, crate_name),
            other => match other.ordinal() {
                Some(k) => write!(f, "S{}", k),
                None => unreachable!("官方段结构必有序号"),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// 三问元数据（核/对偶/格）
// ---------------------------------------------------------------------------

/// 问一：核是什么（五种化身 + 无核服务）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// 未回答（注册汇入时被三问校验拒绝）。
    Unspecified,
    /// 卷积核（调和分析）。
    Convolution,
    /// 格林函数（PDE/物理，冲激响应）。
    Green,
    /// 转移核（概率论，随机游走）。
    Transfer,
    /// 协方差核（高斯过程）。
    Covariance,
    /// 再生核（逼近论/RKHS）。
    Reproducing,
    /// 无核服务（层次/表示等不需要核的结构）。
    NoKernel,
}

/// 问二：配对哪一侧（五对偶）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualityPair {
    /// 未回答（注册汇入时被三问校验拒绝）。
    Unspecified,
    /// 边界 ↔ 体（mesh vs SDF）。
    BoundaryVolume,
    /// 样本 ↔ 测度（蒙特卡洛）。
    SampleMeasure,
    /// 时域 ↔ 频域（傅里叶）。
    TimeFrequency,
    /// 原问题 ↔ 乘子（拉格朗日对偶）。
    PrimalMultiplier,
    /// 内容 ↔ 摘要（哈希/Merkle）。
    ContentDigest,
}

/// 问三：格锚了吗（哪些量锚 Zᵏ、哪些量允许格间插值）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticePlan {
    /// 未回答（注册汇入时被三问校验拒绝）。
    Unspecified,
    /// 全量锚定 Zᵏ：整数运算，确定性是定理而非承诺。
    LatticeAnchored,
    /// 格锚 + 格点间局部浮点插值（端点皆格点），插值核显式命名。
    LatticeAnchoredInterp {
        /// 插值核名（如 "quintic"、"catmull-rom"）。
        kernel: &'static str,
    },
}

/// 能力面声明（三操作接口全集，能力缺失是类型级事实）。
/// `apply` 为必答主操作恒可用；`invert` / `sample` 按能力显式声明。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpCapabilities {
    /// 是否声明求逆能力（未声明者不会被调度，含该操作的组合方式注册表拒绝）。
    pub invert: bool,
    /// 是否声明采样能力。
    pub sample: bool,
}

/// 三问注册元数据。由特解作者（内核贡献者）填写，
/// 使用者无感；注册汇入时校验，问不齐注册表拒绝（三段式结构化报错，宪章 C18）。
#[derive(Debug, Clone, PartialEq)]
pub struct StructureMeta {
    /// 结构标识（官方段或用户段）。
    pub id: StructureId,
    /// 问一：核是什么。
    pub kernel: KernelKind,
    /// 问二：配对哪一侧。
    pub duality: DualityPair,
    /// 问三：格锚了吗。
    pub lattice: LatticePlan,
    /// 能力面（invert / sample 支持与否， L3）。
    pub capabilities: OpCapabilities,
}

impl StructureMeta {
    /// 以三问答案与能力面构造元数据。
    pub fn new(
        id: StructureId,
        kernel: KernelKind,
        duality: DualityPair,
        lattice: LatticePlan,
        capabilities: OpCapabilities,
    ) -> Self {
        Self { id, kernel, duality, lattice, capabilities }
    }
}

// ---------------------------------------------------------------------------
// 数据包裹与求值上下文（类图；D-28）
// ---------------------------------------------------------------------------

/// 结构间传递的数据包裹（域/陪域按结构定义，Payload 统一包裹）。
/// 变体随各结构特解落地按需扩充（第二批求值引擎接续）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Payload {
    /// 标量值。
    Scalar(f32),
    /// 三维向量。
    Vec3(Vec3),
    /// 无值（单位元 / 占位）。
    Empty,
}

/// 格上下文：确定性在格 Zᵏ 上的约定载体（/ 格锚定规则表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeCtx {
    /// 定点相位的小数位数（Q32.32 约定 = 32； 采样回放）。
    pub fixed_point_frac_bits: u32,
}

impl Default for LatticeCtx {
    fn default() -> Self {
        Self { fixed_point_frac_bits: 32 }
    }
}

/// 模拟快照键 = (子图签名, 帧)（D-28；签名含通道定义哈希的语义在 A6 落地）。
pub type SnapshotKey = (Hash256, u64);

/// 快照库：有状态模拟的纯数据状态仓（D-28）。
///
/// 模拟类节点（S3 约束族）的 apply 语义 = 调度器自快照库装配（上一帧状态，
/// 当前帧参数）后投影一步。LRU 逐出只影响重算、不影响结果——seek 等价成立。
/// LRU 上限策略随求值接线（eval.rs）落地，当前版本定型存取接口。
#[derive(Debug, Clone, Default)]
pub struct SnapshotStore {
    entries: HashMap<SnapshotKey, Vec<u8>>,
}

impl SnapshotStore {
    /// 构造空快照库。
    pub fn new() -> Self {
        Self::default()
    }

    /// 写入一条快照。
    pub fn insert(&mut self, key: SnapshotKey, data: Vec<u8>) {
        self.entries.insert(key, data);
    }

    /// 读取一条快照。
    pub fn get(&self, key: &SnapshotKey) -> Option<&Vec<u8>> {
        self.entries.get(key)
    }

    /// 当前快照条数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// 求值上下文（类图）：全局种子 / 节点 id / 帧号由求值器注入，
/// 自定义特解不得私开 RNG（护栏：SampleIndex hash 派生 = 唯一随机源）。
#[derive(Debug, Clone)]
pub struct EvalCtx {
    /// 全局种子（格 Zᵏ）。
    pub global_seed: u64,
    /// 当前求值节点 id。
    pub node_id: NodeId,
    /// 当前帧号（格 Zᵏ）。
    pub frame: u64,
    /// 格上下文。
    pub lattice: LatticeCtx,
    /// 快照库引用（D-28：有状态模拟经此装配，状态在纯数据资产而非隐藏全局）。
    pub snapshots: SnapshotStore,
}

impl EvalCtx {
    /// 以种子 / 节点 / 帧构造上下文，格约定与快照库取默认。
    pub fn new(global_seed: u64, node_id: NodeId, frame: u64) -> Self {
        Self {
            global_seed,
            node_id,
            frame,
            lattice: LatticeCtx::default(),
            snapshots: SnapshotStore::new(),
        }
    }
}

/// 样本索引（D-28 定型）：随机求积的纯函数派生键。
///
/// 样本流 = hash(全局种子, 节点id, 帧, pixel, 样本索引[, 维度])——同节点不同
/// 像素必不同流（防网格伪影）；任何结构的 sample 不得私开 RNG。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleIndex {
    /// 帧号。
    pub frame: u64,
    /// 像素坐标（D-28 像素域）。
    pub pixel: (u32, u32),
    /// 样本序号。
    pub sample: u32,
    /// 维度序号（Sobol 维度 / 弹射深度编码）。
    pub dimension: u16,
    /// 节点 id。
    pub node_id: u64,
}

// ---------------------------------------------------------------------------
// 求值器统一接口：作用 / 求逆 / 采样三操作
// ---------------------------------------------------------------------------

/// 12 结构统一求值器接口。每个特解 = 本 trait 的一个实现 + 一份三问元数据
/// + 能力面声明。三操作是接口全集；能力面在 [`StructureMeta::capabilities`]
/// 显式声明——`invert` / `sample` 的默认实现为 panic，注册表保证未声明能力
/// 不会被调度（能力缺失从静默缺陷变为类型级事实）。
pub trait MathStructure {
    /// 三问元数据（作者填写；注册汇入时校验，问不齐注册表拒绝）。
    fn meta(&self) -> &StructureMeta;

    /// 作用：正向求值（必答主操作）。
    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload;

    /// 求逆：按 meta 声明的策略解算子方程。仅当 `capabilities.invert` 为真时
    /// 会被调度，否则默认实现 panic（编程错误，贡献者通道内显式暴露）。
    fn invert(&self, y: &Payload, ctx: &EvalCtx) -> Payload {
        let _ = (y, ctx);
        panic!(
            "invert 被调用，但结构 {} 的 meta 未声明 invert 能力",
            self.meta().id
        );
    }

    /// 采样：随机求积。样本流 = hash(seed, node_id, frame, pixel, sample[, dim])，
    /// 纯函数。仅当 `capabilities.sample` 为真时会被调度。
    fn sample(&self, si: SampleIndex, ctx: &EvalCtx) -> Payload {
        let _ = (si, ctx);
        panic!(
            "sample 被调用，但结构 {} 的 meta 未声明 sample 能力",
            self.meta().id
        );
    }
}

// ---------------------------------------------------------------------------
// 节点三元组：结构 / 特解参数 / 组合方式
// ---------------------------------------------------------------------------

/// 节点端口方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortDirection {
    /// 输入端口。
    Input,
    /// 输出端口。
    Output,
}

/// 节点端口：组合算子的输入槽位或输出（图连线语义项在 graph.rs 第二批落地，
/// 本批仅定端口标识类型）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Port {
    /// 端口方向。
    pub direction: PortDirection,
    /// 同方向内的槽位序号。
    pub index: u8,
}

/// 合并归约算子（`Merge(op)`：Compose 树、混音）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOp {
    /// over（常规覆盖合成）。
    Over,
    /// plus（加算）。
    Plus,
    /// screen（滤色）。
    Screen,
    /// multiply（正片叠底）。
    Multiply,
    /// mask（蒙版）。
    Mask,
}

/// CSG 布尔变体（`Csg(op, k)`：min/max/smooth）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsgOp {
    /// 并（min）。
    Union,
    /// 交（max）。
    Intersect,
    /// 减。
    Subtract,
    /// 平滑并（k 控制融合半径）。
    SmoothUnion,
    /// 平滑交。
    SmoothIntersect,
    /// 平滑减。
    SmoothSubtract,
}

/// 组合方式（六种归约算子）。
#[derive(Debug, Clone, PartialEq)]
pub enum Composition {
    /// 默认图边（单进单出）：一切连线。
    Feed,
    /// 串行 f∘g∘h：post 链、色彩节点链。
    Chain,
    /// 多输入归约：over/plus/screen/multiply/mask。
    Merge(MergeOp),
    /// CSG 布尔变体，`k` 为 smooth 融合半径。
    Csg { op: CsgOp, k: f32 },
    /// power heuristic 多估计量合并，`w` 为权重。
    Mis(f32),
    /// 表示切换（单向声明：mesh→SDF bake）。
    Dual,
}

/// 属性通道引用：按名记录（存档语义，D-23；通道求值语义属 A2 / prism-anim）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelRef {
    /// 通道定义名（anim/expr 块命名通道）。
    pub name: String,
}

/// 参数绑定：把节点的一个参数接到一条属性通道（类图 `Node --> PropertyChannel`，
/// "参数绑定 Node.bindings，非六种组合方式"， 对照表）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeBinding {
    /// 被绑定的参数名（Params 的键）。
    pub param: String,
    /// 通道引用。
    pub channel: ChannelRef,
}

/// 特解参数值：schema 化参数的最小值集（类型/范围/默认值的 schema 本体随
/// A4 校验件落地；本批承接类图 `Node.params` 槽位的值形态）。
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// 浮点参数。
    F32(f32),
    /// 整数参数。
    U32(u32),
    /// 布尔参数。
    Bool(bool),
    /// 字符串参数。
    Str(String),
}

/// 特解参数包（节点三元组的"特解参数"位）：有序键值集合。
///
/// 内部用 `BTreeMap` 固定键序，序列化输出逐字节确定（节点签名与存档的输入，
///  / 宪章 C13）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params {
    entries: BTreeMap<String, ParamValue>,
}

impl Params {
    /// 构造空参数包。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置一个参数（同名覆盖）。
    pub fn set(&mut self, name: impl Into<String>, value: ParamValue) {
        self.entries.insert(name.into(), value);
    }

    /// 读取一个参数。
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.entries.get(name)
    }

    /// 参数个数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按字典序遍历全部 (参数名, 参数值)——键序固定，遍历序确定（编辑器
    /// 参数 schema 生成用；只读访问器，不改变 canonical 字节语义）。
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParamValue)> {
        self.entries.iter()
    }

    /// 参数名有序切片（键序升序，编辑器 UI 控件生成用）。
    pub fn names(&self) -> Vec<&String> {
        self.entries.keys().collect()
    }

    /// 序列化为 canonical 字节（键按字典序固定；大端编码；值带类型 tag）。
    /// 供节点签名与存档序列化消费，同参数集必得同字节。
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, value) in &self.entries {
            out.extend_from_slice(&(name.len() as u16).to_be_bytes());
            out.extend_from_slice(name.as_bytes());
            match value {
                ParamValue::F32(v) => {
                    out.push(0);
                    out.extend_from_slice(&v.to_bits().to_be_bytes());
                }
                ParamValue::U32(v) => {
                    out.push(1);
                    out.extend_from_slice(&v.to_be_bytes());
                }
                ParamValue::Bool(v) => {
                    out.push(2);
                    out.push(*v as u8);
                }
                ParamValue::Str(v) => {
                    out.push(3);
                    out.extend_from_slice(&(v.len() as u16).to_be_bytes());
                    out.extend_from_slice(v.as_bytes());
                }
            }
        }
        out
    }
}

/// 图节点（类图 `Node`）：节点三元组的实例形态。
///
/// `signature()` 方法属签名树（signature.rs，第二批），本批不落。
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// 节点唯一标识。
    pub id: NodeId,
    /// 结构位：本节点是哪个数学结构的特解。
    pub structure: StructureId,
    /// 特解参数位：schema 化参数包（可被 `bindings` 的通道按帧解算覆盖）。
    pub params: Params,
    /// 组合方式位：本节点如何与图连接。
    pub composition: Composition,
    /// 参数绑定：哪些参数接属性通道。
    pub bindings: Vec<NodeBinding>,
}

impl Node {
    /// 构造节点：id / 结构 / 组合方式必填，参数与绑定从空开始。
    pub fn new(id: NodeId, structure: StructureId, composition: Composition) -> Self {
        Self { id, structure, params: Params::new(), composition, bindings: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_index_has_pixel_domain() {
        // D-28：SampleIndex 增 pixel 字段，同节点不同像素必不同流（防网格伪影）。
        let a = SampleIndex { frame: 3, pixel: (0, 0), sample: 1, dimension: 2, node_id: 9 };
        let b = SampleIndex { frame: 3, pixel: (1, 0), sample: 1, dimension: 2, node_id: 9 };
        assert_ne!(a, b, "同节点不同像素的样本索引必须可区分");
        assert_eq!(a.pixel, (0, 0));
    }

    #[test]
    fn params_canonical_bytes_are_insertion_order_independent() {
        let mut p1 = Params::new();
        p1.set("roughness", ParamValue::F32(0.25));
        p1.set("metalness", ParamValue::F32(0.9));
        p1.set("tint", ParamValue::Str("steel".into()));

        let mut p2 = Params::new();
        p2.set("tint", ParamValue::Str("steel".into()));
        p2.set("metalness", ParamValue::F32(0.9));
        p2.set("roughness", ParamValue::F32(0.25));

        // 插入顺序不同，canonical 字节相同（BTreeMap 键序）。
        assert_eq!(p1.to_canonical_bytes(), p2.to_canonical_bytes());
        assert!(!p1.to_canonical_bytes().is_empty());

        // 值不同则字节不同。
        let mut p3 = p2.clone();
        p3.set("roughness", ParamValue::F32(0.5));
        assert_ne!(p1.to_canonical_bytes(), p3.to_canonical_bytes());
    }

    #[test]
    fn structure_id_official_ordinals_and_custom_isolation() {
        assert_eq!(StructureId::RadiativeTransfer.ordinal(), Some(8));
        assert_eq!(StructureId::EntityRepresentation.ordinal(), Some(12));
        assert_eq!(
            StructureId::ScalarField.to_string(),
            "S1"
        );

        // 按 crate 名隔离冲突：同编号不同 crate 是不同结构。
        let a = StructureId::custom(13, "my_fx");
        let b = StructureId::custom(13, "other_fx");
        assert_ne!(a, b);
        assert_eq!(a.to_string(), "S13@my_fx");
        assert_eq!(StructureId::custom(42, "x").ordinal(), None);
    }

    #[test]
    fn eval_ctx_and_snapshot_store_roundtrip() {
        let ctx = EvalCtx::new(7, 1, 10);
        assert_eq!(ctx.global_seed, 7);
        assert_eq!(ctx.node_id, 1);
        assert_eq!(ctx.frame, 10);
        // Q32.32 默认格约定。
        assert_eq!(ctx.lattice.fixed_point_frac_bits, 32);

        let mut ctx2 = EvalCtx::new(7, 1, 10);
        let key: SnapshotKey = ([0xAB; 32], 9);
        assert!(ctx2.snapshots.get(&key).is_none());
        ctx2.snapshots.insert(key, vec![1, 2, 3]);
        assert_eq!(ctx2.snapshots.get(&key), Some(&vec![1, 2, 3]));
        assert_eq!(ctx2.snapshots.len(), 1);

        // 快照键含帧号：不同帧互不误取（seek 等价的数据侧前提）。
        assert!(ctx2.snapshots.get(&([0xAB; 32], 10)).is_none());
    }

    #[test]
    fn node_triple_fields_compose() {
        let mut node = Node::new(5, StructureId::RadiativeTransfer, Composition::Feed);
        node.params.set("ior", ParamValue::F32(1.5));
        node.bindings.push(NodeBinding {
            param: "roughness".into(),
            channel: ChannelRef { name: "mat_flow".into() },
        });

        assert_eq!(node.structure, StructureId::RadiativeTransfer);
        assert_eq!(node.params.get("ior"), Some(&ParamValue::F32(1.5)));
        assert_eq!(node.bindings.len(), 1);
        assert_eq!(node.composition, Composition::Feed);

        // 其余组合方式构造可用。
        let merge = Composition::Merge(MergeOp::Over);
        let csg = Composition::Csg { op: CsgOp::SmoothSubtract, k: 0.2 };
        let mis = Composition::Mis(0.5);
        assert_eq!(merge, Composition::Merge(MergeOp::Over));
        assert_eq!(csg, Composition::Csg { op: CsgOp::SmoothSubtract, k: 0.2 });
        assert_eq!(mis, Composition::Mis(0.5));
    }
}
