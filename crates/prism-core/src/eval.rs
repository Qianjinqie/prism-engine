//! Graph(t) 纯函数求值调度。
//!
//! 语义权威：一帧求值时序：Graph(t) 纯函数调度、
//! （确定性操作化：随机性 = hash(种子, 节点id, 帧, pixel, 样本[, 维度])
//! 唯一入口；有状态模拟 = EvalCtx 快照库装配后投影一步，D-28）、
//! （任何归约点一律走 `reduction_shape` 固定形状）。
//!
//! 求值契约：`eval` 无隐藏状态——同 (图, 注册表, 采样面, 子图签名, EvalCtx)
//! 两遍求值，全帧输出逐字节一致（关卡 1）；求值顺序 = 拓扑序
//! + 按 node_id 稳定 tie-break。签名缓存当前只落 eval 侧键类型
//! （`SnapshotKey`，缓存存储归签名模块）。
//!
//! 帧循环接线（架构 §4.4「有状态模拟 = EvalCtx 快照库接线」）：`EvalCtx`
//! 携带 `SnapshotStore` 持有值，帧循环驱动必须让快照库跨帧保留——同一
//! EvalCtx 逐帧推进，或每帧新建 EvalCtx 时由驱动把上一帧库注入新 ctx
//! （两种形态逐位等价，测试固化）。快照键 = (子图签名, 帧)，签名位来自
//! `snapshot_signature()`（含通道定义哈希链）。seek 等价成立条件：跳第 N 帧
//! ≡ 顺序推进到第 N 帧，逐字节；冷 seek（frame>0 且 (sig, frame-1) 缺失）
//! → P3012 结构化报错，绝不静默 fallback（C20）。LRU 逐出只影响重算、
//! 不影响结果，前提是依赖跨度 ≤ 保留窗口。

use std::collections::BTreeMap;
use std::fmt;

use prism_math::Vec3;

use crate::graph::{reduce, Graph, GraphError};
use crate::node::{
    Composition, EvalCtx, Hash256, MathStructure, MergeOp, NodeId, Payload, SampleIndex,
    SnapshotKey, StructureId,
};
use crate::registry::NodeRegistry;

// ---------------------------------------------------------------------------
// 求值错误（宪章 C18 三段式；C19：必含建议段）
// ---------------------------------------------------------------------------

/// 图求值错误。错误码分段（宪章 C18）：P1xxx 防呆参数 / P3xxx 图结构。
#[derive(Debug, Clone, PartialEq)]
pub struct EvalError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：发生位置。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for EvalError {}

impl From<GraphError> for EvalError {
    fn from(e: GraphError) -> Self {
        EvalError { code: e.code, location: e.location, problem: e.problem, fix: e.fix }
    }
}

// ---------------------------------------------------------------------------
// 求值入口与产物
// ---------------------------------------------------------------------------

/// 帧内采样面：当前求值像素（D-28 像素域）与每像素样本数 spp。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplePlane {
    /// 像素坐标。样本流经 `SampleIndex.pixel` 派生——同节点不同像素必不同
    /// 流（防网格伪影，D-28；tile 编排接入后坐标换算归位）。
    pub pixel: (u32, u32),
    /// 每像素样本数（≥1）。样本合并走固定形状归约（归约点）。
    pub spp: u32,
}

/// 一帧求值产物：节点 id → 输出值（BTreeMap 固定 id 序）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalOutput {
    values: BTreeMap<NodeId, Payload>,
}

impl EvalOutput {
    /// 读取节点输出。
    pub fn get(&self, id: NodeId) -> Option<Payload> {
        self.values.get(&id).copied()
    }

    /// 按 id 升序遍历全部输出。
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, Payload)> + '_ {
        self.values.iter().map(|(&id, &p)| (id, p))
    }

    /// 输出节点数。
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// canonical 字节（全帧输出指纹的输入）：节点 id 大端 + payload 字节，
    /// 按 id 升序。两遍求值逐字节一致 ⇔ canonical 字节逐字节一致（关卡 1）。
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (&id, value) in &self.values {
            out.extend_from_slice(&id.to_be_bytes());
            out.extend_from_slice(&payload_to_bytes(value));
        }
        out
    }
}

/// Graph(t) 一帧求值：拓扑序 + 按 node_id 稳定 tie-break 逐
/// 节点调度特解三操作；无隐藏状态。
///
/// 签名口径：`sig` 必须来自 snapshot_signature()（签名计算），与 session.rs 口径一致。
///
/// 调度语义（当前范围）：
/// - 组合归约：`Feed` 单进单出；`Merge(Plus | Multiply)` 在标量/向量域按
///  [`reduction_shape`] 固定形状归约（任何归约点不得私定结合
///   序）。其余组合方式需要图像/几何域 Payload，结构化报错不静默（C20）。
/// - 采样节点（meta.capabilities.sample）：spp 个 `SampleIndex`（dimension
///   = 0，样本流由特解按 D-28 纯函数派生）经固定形状归约取均值。
/// - 模拟节点（S3，D-28）：自快照库按 (sig, frame-1) 装配上一帧状态后投影
///   一步，产物写回 (sig, frame)；首帧（frame = 0 或无快照）以图输入作初值。
///   同帧重求值幂等（写键 = 当前帧，读键 = 上一帧，互不污染）。
pub fn eval(
    graph: &Graph,
    registry: &NodeRegistry,
    plane: &SamplePlane,
    sig: Hash256,
    ctx: &mut EvalCtx,
) -> Result<EvalOutput, EvalError> {
    if plane.spp == 0 {
        return Err(EvalError {
            code: "P1001",
            location: "eval（帧求值参数）".to_string(),
            problem: format!("spp = 0：每像素样本数必须 ≥1（pixel = {:?}）", plane.pixel),
            fix: "把 SamplePlane::spp 设为 ≥1，如 SamplePlane { pixel, spp: 1 }".to_string(),
        });
    }
    let order = graph.topo_order()?;

    let mut values: BTreeMap<NodeId, Payload> = BTreeMap::new();
    for id in order {
        let node = graph.node(id).expect("拓扑序内的节点必在图内");
        let structure = node.structure.clone();

        let meta = registry.meta_of(&structure).ok_or_else(|| EvalError {
            code: "P3009",
            location: format!("eval（节点 {id}）"),
            problem: format!("结构 {structure} 未注册：图引用了缺定义的结构"),
            fix: format!(
                "先向 NodeRegistry 注册结构 {structure}（registry! 静态汇入或 collect）"
            ),
        })?;
        let sample_capable = meta.capabilities.sample;
        let structure_impl = registry
            .resolve(&structure)
            .expect("meta_of 命中则 resolve 必命中");

        // 节点级上下文：种子 / 帧 / 格约定来自 ctx，节点 id 按本节点注入
        // （全局种子 / 节点id / 帧号由求值器注入）。
        let node_ctx = EvalCtx {
            global_seed: ctx.global_seed,
            node_id: id,
            frame: ctx.frame,
            lattice: ctx.lattice,
            snapshots: ctx.snapshots.clone(),
        };

        // 1) 图输入装配（组合方式归约语义）。
        let upstream: Vec<Payload> = graph
            .inputs_of(id)
            .into_iter()
            .map(|(_, src)| values.get(&src).copied().expect("拓扑序保证上游已求值"))
            .collect();

        // 2) 节点求值（采样 / 模拟装配 / 普通 apply）。
        let output = if sample_capable {
            sample_mean(structure_impl.as_ref(), plane, id, &node_ctx)?
        } else {
            let mut input = reduce_by_composition(&node.composition, upstream)?;
            if structure == StructureId::ConstraintFlow {
                // D-28 快照装配：按 (sig, frame-1) 取上一帧状态。
                // frame = 0：图输入作初值（首帧语义，合法）。
                // frame > 0 且快照缺失：绝不静默 fallback 到初值——那会把
                // seek 场景（时间线拖动 / LRU 逐出 / 缓存清空）变成语义级
                // 静默错误：输出与首帧逐位相同、错误状态写回快照库并沿
                // 轨道传播（qa_seek_snapshot_semantics 四个探针记录）。
                // C20：结构化报错，修复责任交还调用方（顺序求值或显式初态）。
                if ctx.frame > 0 {
                    match snapshot_read(ctx, sig, ctx.frame - 1) {
                        Some(prev) => input = prev,
                        None => return Err(EvalError {
                            code: "P3012",
                            location: format!("eval（节点 {id}，帧 {}）", ctx.frame),
                            problem: format!(
                                "帧间状态缺失：节点 {id} 在帧 {} 求值，但快照 (sig, {}) 不存在——\
                                 seek 断档或缓存未顺序预热",
                                ctx.frame,
                                ctx.frame - 1
                            ),
                            fix: format!(
                                "从 0 帧起顺序求值（跨帧复用同一 EvalCtx，快照库随帧累积），\
                                 或为该节点显式提供帧 {} 的初态快照",
                                ctx.frame - 1
                            ),
                        }),
                    }
                }
            }
            let y = structure_impl.apply(&input, &node_ctx);
            if structure == StructureId::ConstraintFlow {
                snapshot_write(ctx, sig, ctx.frame, &y);
            }
            y
        };
        values.insert(id, output);
    }

    Ok(EvalOutput { values })
}

// ---------------------------------------------------------------------------
// 组合归约（归约点唯一形状入口）
// ---------------------------------------------------------------------------

/// 组合方式的图输入归约。当前范围：标量/向量代数子集。
fn reduce_by_composition(
    composition: &Composition,
    inputs: Vec<Payload>,
) -> Result<Payload, EvalError> {
    match composition {
        Composition::Feed => match inputs.len() {
            0 => Ok(Payload::Empty),
            1 => Ok(inputs.into_iter().next().expect("len == 1")),
            n => Err(EvalError {
                code: "P3006",
                location: "eval（Feed 输入装配）".to_string(),
                problem: format!(
                    "Feed 节点有 {n} 条入边：Feed 是单进单出连线，多输入属图结构错误"
                ),
                fix: "多输入请把节点组合方式改为 Composition::Merge(op)，或拆掉多余连线"
                    .to_string(),
            }),
        },
        Composition::Merge(MergeOp::Plus) => try_reduce_items(
            inputs.into_iter().map(Ok).collect(),
            payload_add,
        ),
        Composition::Merge(MergeOp::Multiply) => try_reduce_items(
            inputs.into_iter().map(Ok).collect(),
            payload_mul,
        ),
        other => Err(EvalError {
            code: "P3010",
            location: format!("eval（组合方式 {other:?}）"),
            problem: "该组合方式的归约语义需要图像/几何域 Payload（当前仅标量/向量代数域）"
                .to_string(),
            fix: "标量/向量域请用 Merge(Plus) 或 Merge(Multiply)；图像域合成（over/screen/\
                  mask）、链式（Chain）、CSG、MIS、表示切换（Dual）随对应宿主 crate 接入"
                .to_string(),
        }),
    }
}

/// 把"逐项 Result"序列按固定形状归约，任一 Err 短路（唯一形状入口 = reduce）。
fn try_reduce_items(
    items: Vec<Result<Payload, EvalError>>,
    combine: impl Fn(&Payload, &Payload) -> Result<Payload, EvalError>,
) -> Result<Payload, EvalError> {
    if items.len() <= 1 {
        return items
            .into_iter()
            .next()
            .unwrap_or_else(|| Ok(Payload::Empty));
    }
    reduce(
        items,
        |a, b| match (a, b) {
            (Ok(x), Ok(y)) => combine(&x, &y),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        },
        Ok(Payload::Empty),
    )
}

/// 二元合并：Plus（标量加 / 向量逐分量加）。
fn payload_add(a: &Payload, b: &Payload) -> Result<Payload, EvalError> {
    match (a, b) {
        (Payload::Scalar(x), Payload::Scalar(y)) => Ok(Payload::Scalar(x + y)),
        (Payload::Vec3(x), Payload::Vec3(y)) => Ok(Payload::Vec3(*x + *y)),
        _ => Err(domain_mismatch_p3011(a, b)),
    }
}

/// 二元合并：Multiply（标量乘 / 向量逐分量乘）。
fn payload_mul(a: &Payload, b: &Payload) -> Result<Payload, EvalError> {
    match (a, b) {
        (Payload::Scalar(x), Payload::Scalar(y)) => Ok(Payload::Scalar(x * y)),
        (Payload::Vec3(x), Payload::Vec3(y)) => Ok(Payload::Vec3(Vec3::new(
            x.x * y.x,
            x.y * y.y,
            x.z * y.z,
        ))),
        _ => Err(domain_mismatch_p3011(a, b)),
    }
}

/// 归约输入域不匹配（P3011，三段式）。
fn domain_mismatch_p3011(a: &Payload, b: &Payload) -> EvalError {
    EvalError {
        code: "P3011",
        location: "eval（组合归约）".to_string(),
        problem: format!(
            "归约输入域不匹配：{a:?} 与 {b:?}（Plus/Multiply 要求同为标量或同为向量，且非 Empty）"
        ),
        fix: "检查上游输出域一致；Empty（无值）上游不能参与归约".to_string(),
    }
}

// ---------------------------------------------------------------------------
// 采样路径（SampleIndex 纯函数派生 = 唯一随机性入口）
// ---------------------------------------------------------------------------

/// 采样节点求值：spp 个样本（dimension = 0）→ 固定形状归约求和 → 均值。
fn sample_mean(
    structure: &dyn MathStructure,
    plane: &SamplePlane,
    node_id: NodeId,
    ctx: &EvalCtx,
) -> Result<Payload, EvalError> {
    let items: Vec<Result<Payload, EvalError>> = (0..plane.spp)
        .map(|s| {
            let si = SampleIndex {
                frame: ctx.frame,
                pixel: plane.pixel,
                sample: s,
                dimension: 0,
                node_id,
            };
            match structure.sample(si, ctx) {
                Payload::Empty => Err(EvalError {
                    code: "P3011",
                    location: format!("eval（采样节点 {node_id}）"),
                    problem: "sample 返回 Empty（无值）：采样节点必须产出标量/向量样本"
                        .to_string(),
                    fix: "在特解 sample() 中返回 Payload::Scalar 或 Payload::Vec3".to_string(),
                }),
                other => Ok(other),
            }
        })
        .collect();
    let summed = try_reduce_items(items, payload_add)?;
    match summed {
        Payload::Scalar(x) => Ok(Payload::Scalar(x / plane.spp as f32)),
        Payload::Vec3(v) => Ok(Payload::Vec3(v / plane.spp as f32)),
        Payload::Empty => Ok(Payload::Empty), // spp ≥ 1 时不可达（items 非空且已滤 Empty）
    }
}

// ---------------------------------------------------------------------------
// 快照读写 API（D-28：有状态模拟的纯数据状态仓装配路径）
// ---------------------------------------------------------------------------

/// 快照读：按 (子图签名, 帧) 读一帧状态并解码为 [`Payload`]。
///
/// 字节不被识别时返回 `None`（绝不猜测，宪章 C20）。
///
/// **seek 等价的保障方式（第 4 轮研究后修订）**：`eval` 对 ConstraintFlow 在
/// `frame > 0` 且快照缺失时**结构化报错（P3012）**，绝不静默 fallback 到初值——
/// 旧路径（`unwrap_or(input)`）会把 seek 断档变成语义级静默错误：输出与首帧
/// 逐位相同，且错误状态写回快照库沿轨道传播（`qa_seek_snapshot_semantics`
/// 四探针记录）。LRU 逐出策略落地时，逐出必须遵守「依赖跨度 ≤ 保留窗口」，
/// 否则同样触发 P3012——逐出影响重算的前提是**先保证不断档**。
pub fn snapshot_read(ctx: &EvalCtx, sig: Hash256, frame: u64) -> Option<Payload> {
    let key: SnapshotKey = (sig, frame);
    ctx.snapshots.get(&key).and_then(|bytes| payload_from_bytes(bytes))
}

/// 快照写：求值产物按 (子图签名, 帧) 写回快照库。
///
/// 同帧重写覆盖同值（求值纯函数 ⇒ 幂等）；跨帧键互不污染——同帧重求值
/// 读上一帧键，装配结果不变。
pub fn snapshot_write(ctx: &mut EvalCtx, sig: Hash256, frame: u64, value: &Payload) {
    let key: SnapshotKey = (sig, frame);
    ctx.snapshots.insert(key, payload_to_bytes(value));
}

// ---------------------------------------------------------------------------
// Payload 确定性字节编码（快照库与全帧指纹共用）
// ---------------------------------------------------------------------------

/// Payload 的确定性字节编码：tag 0 = 标量 / 1 = Vec3 / 2 = Empty，数值一律
/// 大端位模式（宪章 C24 平台中立；同值必得同字节）。
pub fn payload_to_bytes(value: &Payload) -> Vec<u8> {
    let mut out = Vec::new();
    match value {
        Payload::Scalar(v) => {
            out.push(0);
            out.extend_from_slice(&v.to_bits().to_be_bytes());
        }
        Payload::Vec3(v) => {
            out.push(1);
            for c in [v.x, v.y, v.z] {
                out.extend_from_slice(&c.to_bits().to_be_bytes());
            }
        }
        Payload::Empty => out.push(2),
    }
    out
}

/// [`payload_to_bytes`] 的逆向解码；格式不识别返回 `None`（绝不静默猜测）。
pub fn payload_from_bytes(bytes: &[u8]) -> Option<Payload> {
    match bytes.first()? {
        &0 => {
            if bytes.len() != 5 {
                return None;
            }
            let bits = u32::from_be_bytes(bytes[1..5].try_into().ok()?);
            Some(Payload::Scalar(f32::from_bits(bits)))
        }
        &1 => {
            if bytes.len() != 13 {
                return None;
            }
            let mut comps = [0.0f32; 3];
            for (i, c) in comps.iter_mut().enumerate() {
                let bits = u32::from_be_bytes(bytes[1 + i * 4..5 + i * 4].try_into().ok()?);
                *c = f32::from_bits(bits);
            }
            Some(Payload::Vec3(Vec3::new(comps[0], comps[1], comps[2])))
        }
        &2 => {
            if bytes.len() != 1 {
                return None;
            }
            Some(Payload::Empty)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{
        ChannelRef, DualityPair, KernelKind, LatticePlan, Node, NodeBinding, OpCapabilities,
        ParamValue, SnapshotStore, StructureMeta,
    };
    use crate::signature::{
        channel_definition_hash, node_signature, snapshot_signature, ChannelDef, ChannelDefs,
    };

    /// 测试用子图签名（signature.rs 落地前由调用方提供的占位签名）。
    const TEST_SIG: Hash256 = [0xCD; 32];

    // —— 测试特解 ——

    /// 常量源：apply(Empty) → 标量 MILLI/1000（特解常量按类型编码——resolve
    /// 缺 SolvedParams 参数位，缺口在册，参数装配随属性通道落地）。
    #[derive(Debug)]
    struct ConstSourceSpec<const MILLI: u32> {
        meta: StructureMeta,
    }

    impl<const MILLI: u32> Default for ConstSourceSpec<MILLI> {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::PiecewisePolynomial,
                    KernelKind::Reproducing,
                    DualityPair::ContentDigest,
                    LatticePlan::LatticeAnchoredInterp { kernel: "constant" },
                    OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl<const MILLI: u32> MathStructure for ConstSourceSpec<MILLI> {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Empty => Payload::Scalar(MILLI as f32 / 1000.0),
                other => *other,
            }
        }
    }

    /// 恒等透传（归约汇点特解）。
    #[derive(Debug)]
    struct IdentitySpec {
        meta: StructureMeta,
    }

    impl Default for IdentitySpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::EntityRepresentation,
                    KernelKind::NoKernel,
                    DualityPair::ContentDigest,
                    LatticePlan::LatticeAnchored,
                    OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl MathStructure for IdentitySpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            *x
        }
    }

    /// 常量向量源（域不匹配用例）。
    #[derive(Debug)]
    struct ConstVecSpec {
        meta: StructureMeta,
    }

    impl Default for ConstVecSpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::HierarchySignature,
                    KernelKind::NoKernel,
                    DualityPair::ContentDigest,
                    LatticePlan::LatticeAnchored,
                    OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl MathStructure for ConstVecSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Empty => Payload::Vec3(Vec3::new(1.0, 2.0, 3.0)),
                other => *other,
            }
        }
    }

    /// 采样特解（S4）：样本流 = splitmix 纯函数派生（唯一随机性
    /// 入口的测试实例）——同 (frame, pixel, sample, dim, node_id) 同值。
    #[derive(Debug)]
    struct SampleSpec {
        meta: StructureMeta,
    }

    impl Default for SampleSpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::DeterministicSampling,
                    KernelKind::Transfer,
                    DualityPair::SampleMeasure,
                    LatticePlan::LatticeAnchored,
                    OpCapabilities { invert: false, sample: true },
                ),
            }
        }
    }

    /// splitmix64 终化函数（整数运算跨平台逐位一致）。
    fn hash_mix(seed: u64, x: u64) -> u64 {
        let mut z = seed ^ x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    impl MathStructure for SampleSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            *x
        }
        fn sample(&self, si: SampleIndex, ctx: &EvalCtx) -> Payload {
            let mut h = hash_mix(ctx.global_seed, si.node_id);
            h = hash_mix(h, si.frame);
            h = hash_mix(h, ((si.pixel.0 as u64) << 32) | si.pixel.1 as u64);
            h = hash_mix(h, si.sample as u64);
            h = hash_mix(h, si.dimension as u64);
            // u64 高 24 位 → f32 ∈ [0, 1)（格点间局部插值域）。
            Payload::Scalar((h >> 40) as f32 / 16_777_216.0)
        }
    }

    /// 模拟投影特解（S3）：状态 +1 一步（Empty = 初值 0）。
    #[derive(Debug)]
    struct StepSpec {
        meta: StructureMeta,
    }

    impl Default for StepSpec {
        fn default() -> Self {
            Self {
                meta: StructureMeta::new(
                    StructureId::ConstraintFlow,
                    KernelKind::Green,
                    DualityPair::PrimalMultiplier,
                    LatticePlan::LatticeAnchored,
                    OpCapabilities { invert: false, sample: false },
                ),
            }
        }
    }

    impl MathStructure for StepSpec {
        fn meta(&self) -> &StructureMeta {
            &self.meta
        }
        fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
            match x {
                Payload::Empty => Payload::Scalar(0.0),
                Payload::Scalar(v) => Payload::Scalar(v + 1.0),
                other => *other,
            }
        }
    }

    // —— 装配辅助 ——

    fn make<T: MathStructure + Default + 'static>() -> Box<dyn MathStructure> {
        Box::new(T::default())
    }

    fn spec_meta(
        id: StructureId,
        kernel: KernelKind,
        duality: DualityPair,
        lattice: LatticePlan,
        sample: bool,
    ) -> StructureMeta {
        StructureMeta::new(
            id,
            kernel,
            duality,
            lattice,
            OpCapabilities { invert: false, sample },
        )
    }

    fn test_registry() -> NodeRegistry {
        let mut reg = NodeRegistry::new();
        reg.collect(
            "const.1500",
            spec_meta(
                StructureId::PiecewisePolynomial,
                KernelKind::Reproducing,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchoredInterp { kernel: "constant" },
                false,
            ),
            make::<ConstSourceSpec<1500>>,
        )
        .unwrap();
        reg.collect(
            "const.2250",
            spec_meta(
                StructureId::ScalarField,
                KernelKind::Covariance,
                DualityPair::SampleMeasure,
                LatticePlan::LatticeAnchored,
                false,
            ),
            make::<ConstSourceSpec<2250>>,
        )
        .unwrap();
        reg.collect(
            "const.750",
            spec_meta(
                StructureId::Diffusion,
                KernelKind::Convolution,
                DualityPair::BoundaryVolume,
                LatticePlan::LatticeAnchored,
                false,
            ),
            make::<ConstSourceSpec<750>>,
        )
        .unwrap();
        reg.collect(
            "identity",
            spec_meta(
                StructureId::EntityRepresentation,
                KernelKind::NoKernel,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchored,
                false,
            ),
            make::<IdentitySpec>,
        )
        .unwrap();
        reg.collect(
            "const_vec3",
            spec_meta(
                StructureId::HierarchySignature,
                KernelKind::NoKernel,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchored,
                false,
            ),
            make::<ConstVecSpec>,
        )
        .unwrap();
        reg.collect(
            "sampler",
            spec_meta(
                StructureId::DeterministicSampling,
                KernelKind::Transfer,
                DualityPair::SampleMeasure,
                LatticePlan::LatticeAnchored,
                true,
            ),
            make::<SampleSpec>,
        )
        .unwrap();
        reg.collect(
            "step",
            spec_meta(
                StructureId::ConstraintFlow,
                KernelKind::Green,
                DualityPair::PrimalMultiplier,
                LatticePlan::LatticeAnchored,
                false,
            ),
            make::<StepSpec>,
        )
        .unwrap();
        reg
    }

    /// 关卡 1 图：3 常量源 → Merge(Plus) 归约节点（n4）+ 采样节点（n5）。
    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new(
            1,
            StructureId::PiecewisePolynomial,
            Composition::Feed,
        ))
        .unwrap();
        g.add_node(Node::new(2, StructureId::ScalarField, Composition::Feed))
            .unwrap();
        g.add_node(Node::new(3, StructureId::Diffusion, Composition::Feed))
            .unwrap();
        g.add_node(Node::new(
            4,
            StructureId::EntityRepresentation,
            Composition::Merge(MergeOp::Plus),
        ))
        .unwrap();
        g.add_node(Node::new(
            5,
            StructureId::DeterministicSampling,
            Composition::Feed,
        ))
        .unwrap();
        g.connect(1, 4, 0).unwrap();
        g.connect(2, 4, 1).unwrap();
        g.connect(3, 4, 2).unwrap();
        g
    }

    // —— RFC 1321 MD5（仅测试指纹资产：关卡 1 断言贴出全帧输出 md5）——

    mod md5 {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
            20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
            16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
            0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
            0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
            0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
            0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
            0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
            0xeb86d391,
        ];

        /// RFC 1321 MD5 摘要（测试内嵌指纹件；签名哈希本体归 SHA-256）。
        pub fn digest(data: &[u8]) -> [u8; 16] {
            let bit_len = (data.len() as u64).wrapping_mul(8);
            let mut msg = data.to_vec();
            msg.push(0x80);
            while msg.len() % 64 != 56 {
                msg.push(0);
            }
            msg.extend_from_slice(&bit_len.to_le_bytes());

            let (mut a0, mut b0, mut c0, mut d0) =
                (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
            for chunk in msg.chunks_exact(64) {
                let mut m = [0u32; 16];
                for (i, word) in m.iter_mut().enumerate() {
                    *word = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
                }
                let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
                for i in 0..64 {
                    let (f, g) = match i / 16 {
                        0 => ((b & c) | (!b & d), i),
                        1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                        2 => (b ^ c ^ d, (3 * i + 5) % 16),
                        _ => (c ^ (b | !d), (7 * i) % 16),
                    };
                    let f2 = f
                        .wrapping_add(a)
                        .wrapping_add(K[i])
                        .wrapping_add(m[g]);
                    a = d;
                    d = c;
                    c = b;
                    b = b.wrapping_add(f2.rotate_left(S[i]));
                }
                a0 = a0.wrapping_add(a);
                b0 = b0.wrapping_add(b);
                c0 = c0.wrapping_add(c);
                d0 = d0.wrapping_add(d);
            }
            let mut out = [0u8; 16];
            out[0..4].copy_from_slice(&a0.to_le_bytes());
            out[4..8].copy_from_slice(&b0.to_le_bytes());
            out[8..12].copy_from_slice(&c0.to_le_bytes());
            out[12..16].copy_from_slice(&d0.to_le_bytes());
            out
        }
    }

    fn md5_hex(data: &[u8]) -> String {
        md5::digest(data).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn md5_fingerprint_self_check() {
        // RFC 1321 官方测试向量：指纹件本身正确，贴出的 md5 才可信。
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    // —— 关卡 1：两遍求值全帧输出 md5 逐字节一致 ——

    #[test]
    fn eval_twice_is_bit_identical_with_md5_fingerprint() {
        let graph = sample_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (13, 29), spp: 4 };

        let mut ctx_a = EvalCtx::new(42, 0, 7);
        let out_a = eval(&graph, &registry, &plane, TEST_SIG, &mut ctx_a).expect("首遍求值");
        let mut ctx_b = EvalCtx::new(42, 0, 7);
        let out_b = eval(&graph, &registry, &plane, TEST_SIG, &mut ctx_b).expect("次遍求值");

        // 归约节点：1.5 + 2.25 + 0.75（三者皆 2^-k 倍数，加法精确）。
        assert_eq!(out_a.get(4), Some(Payload::Scalar(4.5)), "Merge(Plus) 归约值");
        // 采样节点：spp=4 固定形状均值，产出标量。
        assert!(matches!(out_a.get(5), Some(Payload::Scalar(_))), "采样节点输出");

        let bytes_a = out_a.to_canonical_bytes();
        let bytes_b = out_b.to_canonical_bytes();
        assert_eq!(bytes_a, bytes_b, "两遍求值全帧输出逐字节一致（关卡 1）");
        let fingerprint = md5_hex(&bytes_a);
        assert_eq!(fingerprint, md5_hex(&bytes_b));
        println!(
            "eval_twice md5 = {fingerprint}（frame=7, pixel=(13,29), spp=4, bytes={}）",
            bytes_a.len()
        );
    }

    // —— 关卡 3：Graph(t) 纯函数断言 + 采样可复现 ——

    #[test]
    fn graph_eval_is_pure_function_of_frame() {
        let graph = sample_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (13, 29), spp: 4 };

        let mut ctx5a = EvalCtx::new(42, 0, 5);
        let out5a = eval(&graph, &registry, &plane, TEST_SIG, &mut ctx5a).unwrap();
        let mut ctx5b = EvalCtx::new(42, 0, 5);
        let out5b = eval(&graph, &registry, &plane, TEST_SIG, &mut ctx5b).unwrap();
        let mut ctx6 = EvalCtx::new(42, 0, 6);
        let out6 = eval(&graph, &registry, &plane, TEST_SIG, &mut ctx6).unwrap();

        // 同 frame：两遍逐位一致（纯函数）。
        assert_eq!(
            out5a.to_canonical_bytes(),
            out5b.to_canonical_bytes(),
            "同 frame 两遍逐字节一致"
        );
        // 常量子图不随 frame 变；采样节点样本流含 frame → 随 frame 变化。
        assert_eq!(out5a.get(4), out6.get(4), "常量子图与 frame 无关");
        assert_ne!(out5a.get(5), out6.get(5), "采样节点随 frame 变化（Graph(t)）");
    }

    #[test]
    fn sampler_node_is_reproducible_and_pixel_sensitive() {
        let graph = sample_graph();
        let registry = test_registry();

        let mut ctx_a = EvalCtx::new(7, 0, 3);
        let a = eval(&graph, &registry, &SamplePlane { pixel: (0, 0), spp: 2 }, TEST_SIG, &mut ctx_a)
            .unwrap();
        let mut ctx_b = EvalCtx::new(7, 0, 3);
        let b = eval(&graph, &registry, &SamplePlane { pixel: (0, 0), spp: 2 }, TEST_SIG, &mut ctx_b)
            .unwrap();
        let mut ctx_c = EvalCtx::new(7, 0, 3);
        let c = eval(&graph, &registry, &SamplePlane { pixel: (1, 0), spp: 2 }, TEST_SIG, &mut ctx_c)
            .unwrap();

        // 相同 EvalCtx/SampleIndex 下可复现（关卡 3）。
        assert_eq!(a.get(5), b.get(5), "同采样面重求值可复现");
        // D-28：同节点不同像素必不同流（防网格伪影）。
        assert_ne!(a.get(5), c.get(5), "不同像素样本流必不同");
    }

    // —— 关卡 5：EvalCtx 快照装配路径（写入 → 读出 → 装配） ——

    #[test]
    fn snapshot_assembly_write_read_and_project() {
        let mut g = Graph::new();
        g.add_node(Node::new(9, StructureId::ConstraintFlow, Composition::Feed))
            .unwrap();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        let mut ctx = EvalCtx::new(11, 0, 0);
        let o0 = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap();
        assert_eq!(
            o0.get(9),
            Some(Payload::Scalar(0.0)),
            "首帧（frame=0）无上一帧快照：以图输入（Empty）装配初值，投影一步"
        );
        assert_eq!(
            snapshot_read(&ctx, TEST_SIG, 0),
            Some(Payload::Scalar(0.0)),
            "产物写回 (sig, frame=0)"
        );

        ctx.frame = 1;
        let o1 = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap();
        assert_eq!(
            o1.get(9),
            Some(Payload::Scalar(1.0)),
            "frame=1：装配上一帧状态 (sig,0) 后投影一步"
        );
        ctx.frame = 2;
        let o2 = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap();
        assert_eq!(o2.get(9), Some(Payload::Scalar(2.0)), "逐帧推进：0 → 1 → 2");
        assert_eq!(ctx.snapshots.len(), 3, "三帧各一条快照");

        // 同帧重求值幂等：读键 = 上一帧，写键 = 当前帧，互不污染。
        ctx.frame = 1;
        let o1b = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap();
        assert_eq!(o1b.get(9), Some(Payload::Scalar(1.0)), "同帧重求值幂等");
        assert_eq!(ctx.snapshots.len(), 3, "重求值不新增快照条目");
    }

    #[test]
    fn snapshot_bytes_roundtrip() {
        // 快照载荷字节编码往返（写入 → 读出 的格式契约）。
        for value in [
            Payload::Scalar(0.25),
            Payload::Vec3(Vec3::new(1.0, -2.5, 3.25)),
            Payload::Empty,
        ] {
            let bytes = payload_to_bytes(&value);
            assert_eq!(payload_from_bytes(&bytes), Some(value), "{value:?} 往返一致");
        }
        // 非法字节 → None（绝不静默猜测，C20）。
        assert_eq!(payload_from_bytes(&[9]), None);
        assert_eq!(payload_from_bytes(&[0, 1, 2]), None);
        assert_eq!(payload_from_bytes(&[2, 0]), None);
        // 同值同字节（确定性编码）。
        assert_eq!(
            payload_to_bytes(&Payload::Scalar(1.0)),
            payload_to_bytes(&Payload::Scalar(1.0))
        );
    }

    // —— 关卡 4（eval 侧）：错误路径三段式 ——

    #[test]
    fn eval_errors_are_actionable_with_advice() {
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 0 };

        // spp = 0 → P1001（防呆参数）。
        let mut g = Graph::new();
        g.add_node(Node::new(1, StructureId::ScalarField, Composition::Feed))
            .unwrap();
        let mut ctx = EvalCtx::new(1, 0, 0);
        let err = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap_err();
        assert_eq!(err.code, "P1001");
        assert!(err.to_string().contains("建议："), "必须含建议段（C19）：{}", err);

        // 图引用未注册结构 → P3009（绝不静默降级，C20/D-23）。
        let mut g = Graph::new();
        g.add_node(Node::new(
            1,
            StructureId::SpectralDecomposition,
            Composition::Feed,
        ))
        .unwrap();
        let mut ctx = EvalCtx::new(1, 0, 0);
        let err = eval(
            &g,
            &registry,
            &SamplePlane { pixel: (0, 0), spp: 1 },
            TEST_SIG,
            &mut ctx,
        )
        .unwrap_err();
        assert_eq!(err.code, "P3009");
        assert!(err.problem.contains("未注册"), "{}", err.problem);
        assert!(err.fix.contains("registry!"), "建议应给出注册路径：{}", err.fix);
    }

    #[test]
    fn composition_reduction_domain_errors() {
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };
        let mut ctx = EvalCtx::new(1, 0, 0);

        // 图像域组合方式（Over）→ P3010：域未支持，如实报错不静默。
        let mut g = Graph::new();
        g.add_node(Node::new(1, StructureId::PiecewisePolynomial, Composition::Feed))
            .unwrap();
        g.add_node(Node::new(
            2,
            StructureId::EntityRepresentation,
            Composition::Merge(MergeOp::Over),
        ))
        .unwrap();
        g.connect(1, 2, 0).unwrap();
        let err = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap_err();
        assert_eq!(err.code, "P3010");
        assert!(err.fix.contains("Merge(Plus)"), "建议应给可行动替代：{}", err.fix);

        // 归约输入域不匹配（Scalar + Vec3）→ P3011。
        let mut g = Graph::new();
        g.add_node(Node::new(
            1,
            StructureId::HierarchySignature,
            Composition::Feed,
        ))
        .unwrap();
        g.add_node(Node::new(
            2,
            StructureId::EntityRepresentation,
            Composition::Merge(MergeOp::Plus),
        ))
        .unwrap();
        g.add_node(Node::new(
            3,
            StructureId::PiecewisePolynomial,
            Composition::Feed,
        ))
        .unwrap();
        g.connect(1, 2, 0).unwrap();
        g.connect(3, 2, 1).unwrap();
        let err = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap_err();
        assert_eq!(err.code, "P3011");
        assert!(
            err.problem.contains("域不匹配"),
            "应指出事实：{}",
            err.problem
        );
    }

    #[test]
    fn multiply_merge_reduces_scalar_and_vec3() {
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };
        let mut ctx = EvalCtx::new(1, 0, 0);

        // Scalar：1.5 × 2.25 = 3.375（2^-k 倍数，乘法精确）。
        let mut g = Graph::new();
        g.add_node(Node::new(
            1,
            StructureId::PiecewisePolynomial,
            Composition::Feed,
        ))
        .unwrap();
        g.add_node(Node::new(2, StructureId::ScalarField, Composition::Feed))
            .unwrap();
        g.add_node(Node::new(
            4,
            StructureId::EntityRepresentation,
            Composition::Merge(MergeOp::Multiply),
        ))
        .unwrap();
        g.connect(1, 4, 0).unwrap();
        g.connect(2, 4, 1).unwrap();
        let out = eval(&g, &registry, &plane, TEST_SIG, &mut ctx).unwrap();
        assert_eq!(out.get(4), Some(Payload::Scalar(3.375)));

        // Vec3 逐分量乘：三个常量向量源逐位断言。
        //（ConstVecSpec 经域不匹配测试已验证，此处以双 Scalar 为代表；Vec3
        // 逐分量乘的单元行为由 payload_add/payload_mul 直测覆盖。）
        let a = Payload::Vec3(Vec3::new(2.0, 4.0, 6.0));
        let b = Payload::Vec3(Vec3::new(3.0, 5.0, 7.0));
        assert_eq!(
            payload_mul(&a, &b).unwrap(),
            Payload::Vec3(Vec3::new(6.0, 20.0, 42.0))
        );
        assert_eq!(
            payload_add(&a, &b).unwrap(),
            Payload::Vec3(Vec3::new(5.0, 9.0, 13.0))
        );
    }

    #[test]
    fn eval_output_canonical_bytes_are_id_ordered() {
        // canonical 字节按节点 id 升序拼接：id 排布无关，逐字节指纹稳定。
        let mut output = EvalOutput::default();
        let mut values = BTreeMap::new();
        values.insert(2u64, Payload::Scalar(2.0));
        values.insert(1u64, Payload::Scalar(1.0));
        output.values = values;

        let direct = {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&1u64.to_be_bytes());
            bytes.extend_from_slice(&payload_to_bytes(&Payload::Scalar(1.0)));
            bytes.extend_from_slice(&2u64.to_be_bytes());
            bytes.extend_from_slice(&payload_to_bytes(&Payload::Scalar(2.0)));
            bytes
        };
        assert_eq!(output.to_canonical_bytes(), direct);
        assert_eq!(output.len(), 2);
        assert_eq!(output.iter().collect::<Vec<_>>().len(), 2);
        assert!(!output.is_empty());
    }

    // —— 参数绑定与节点参数在图上装配（节点三元组贯通） ——

    #[test]
    fn node_params_flow_through_graph_construction() {
        // 节点参数（Params）随节点进图；绑定列表就位（通道解算的图侧载体）。
        let mut node = Node::new(6, StructureId::PiecewisePolynomial, Composition::Feed);
        node.params.set("value", ParamValue::F32(1.5));
        node.bindings.push(crate::node::NodeBinding {
            param: "value".to_string(),
            channel: crate::node::ChannelRef { name: "rise".to_string() },
        });
        let mut g = Graph::new();
        g.add_node(node).unwrap();
        let stored = g.node(6).unwrap();
        assert_eq!(stored.params.get("value"), Some(&ParamValue::F32(1.5)));
        assert_eq!(stored.bindings.len(), 1);
        assert_eq!(stored.bindings[0].channel.name, "rise");
    }

    // —— 帧循环接线（架构 §4.4：快照库跨帧保留 / seek 等价 / 冷 seek 报错） ——

    /// 流节点图：单 ConstraintFlow 节点（模拟投影特解 +1 一步）。
    fn flow_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new(9, StructureId::ConstraintFlow, Composition::Feed))
            .unwrap();
        g
    }

    #[test]
    fn frame_loop_retains_snapshot_store_across_frames() {
        // 帧循环接线的两种驱动形态必须逐位等价：
        // A) 单个 EvalCtx 逐帧推进（库随帧累积）；
        // B) 每帧新建 EvalCtx（movie.rs 帧循环形态），由驱动把上一帧库注入。
        let g = flow_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        let mut retained = EvalCtx::new(11, 0, 0);
        let mut out_a = Vec::new();
        for frame in 0..=4u64 {
            retained.frame = frame;
            out_a.push(
                eval(&g, &registry, &plane, TEST_SIG, &mut retained)
                    .unwrap()
                    .get(9),
            );
        }

        let mut carried = SnapshotStore::new();
        let mut out_b = Vec::new();
        for frame in 0..=4u64 {
            let mut per_frame = EvalCtx::new(11, 0, frame);
            per_frame.snapshots = std::mem::take(&mut carried);
            out_b.push(
                eval(&g, &registry, &plane, TEST_SIG, &mut per_frame)
                    .unwrap()
                    .get(9),
            );
            // 驱动把累积后的库搬回，供下一帧注入。
            carried = per_frame.snapshots;
        }

        assert_eq!(
            out_a, out_b,
            "单 ctx 跨帧推进 ≡ 每帧新建 ctx 注入保留库（逐位一致）"
        );
        assert_eq!(
            out_a,
            vec![
                Some(Payload::Scalar(0.0)),
                Some(Payload::Scalar(1.0)),
                Some(Payload::Scalar(2.0)),
                Some(Payload::Scalar(3.0)),
                Some(Payload::Scalar(4.0)),
            ],
            "0 → 1 → 2 → 3 → 4 逐帧投影"
        );
        assert_eq!(retained.snapshots.len(), 5, "五帧各一条快照");
        assert_eq!(carried.len(), 5, "搬运后的库同样五条");
        assert_eq!(
            retained.snapshots.get(&(TEST_SIG, 4)),
            carried.get(&(TEST_SIG, 4)),
            "两种驱动的终态库逐位一致"
        );
    }

    #[test]
    fn seek_after_sequential_warmup_is_bit_identical() {
        // 架构 §4.4 seek 等价成立条件：跳第 N 帧 ≡ 顺序推进到第 N 帧，逐字节。
        let g = flow_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        let mut sequential = EvalCtx::new(11, 0, 0);
        let mut sequential_bytes = Vec::new();
        for frame in 0..=6u64 {
            sequential.frame = frame;
            sequential_bytes.push(
                eval(&g, &registry, &plane, TEST_SIG, &mut sequential)
                    .unwrap()
                    .to_canonical_bytes(),
            );
        }

        // seek：注入预热后的快照库，直接跳第 6 帧求值。
        let mut seeked = EvalCtx::new(11, 0, 6);
        seeked.snapshots = sequential.snapshots.clone();
        let seeked_bytes = eval(&g, &registry, &plane, TEST_SIG, &mut seeked)
            .unwrap()
            .to_canonical_bytes();

        assert_eq!(
            seeked_bytes, sequential_bytes[6],
            "跳第 6 帧 ≡ 顺序推进到第 6 帧（逐字节）"
        );
        assert_eq!(seeked.snapshots.len(), 7, "seek 求值覆写同键，不新增条目");
    }

    #[test]
    fn cold_seek_gap_is_structured_p3012_not_silent() {
        // C20：冷 seek（frame>0 且 (sig, frame-1) 缺失）结构化报错，
        // 绝不静默 fallback 到初值（那会把 seek 断档变成语义级静默错误）。
        let g = flow_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        let mut cold = EvalCtx::new(11, 0, 5);
        let err = eval(&g, &registry, &plane, TEST_SIG, &mut cold).unwrap_err();

        assert_eq!(err.code, "P3012");
        assert!(err.location.contains("节点 9"), "三段式一【在哪】：{}", err.location);
        assert!(err.problem.contains("快照"), "三段式二【什么事】：{}", err.problem);
        assert!(err.fix.contains("顺序求值"), "三段式三【怎么改】：{}", err.fix);
        let rendered = err.to_string();
        assert!(rendered.contains("error[P3012]"), "{}", rendered);
        assert!(rendered.contains("建议："), "必含建议段（C19）：{}", rendered);

        // 错误路径不写快照（绝不把错误状态写回库沿轨道传播）。
        assert!(
            cold.snapshots.is_empty(),
            "P3012 报错路径不得写入任何快照"
        );

        // frame=0 合法初值：无上一帧快照时以图输入（Empty）作初态（首帧语义）。
        let mut first = EvalCtx::new(11, 0, 0);
        let out0 = eval(&g, &registry, &plane, TEST_SIG, &mut first).unwrap();
        assert_eq!(out0.get(9), Some(Payload::Scalar(0.0)), "frame=0 首帧初值合法");
        assert_eq!(first.snapshots.len(), 1);
    }

    #[test]
    fn snapshot_store_truncation_recompute_is_bit_identical() {
        // S4 LRU 逐出中性：逐出只影响重算、不影响结果——窗口内续算与全量
        // 重放逐位一致；逐出跨过依赖跨度则 P3012（前提是跨度 ≤ 保留窗口）。
        let g = flow_graph();
        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        // 全量重放 0..=7，记录每帧 canonical 字节与终态快照库。
        let mut full = EvalCtx::new(11, 0, 0);
        let mut full_bytes = Vec::new();
        for frame in 0..=7u64 {
            full.frame = frame;
            full_bytes.push(
                eval(&g, &registry, &plane, TEST_SIG, &mut full)
                    .unwrap()
                    .to_canonical_bytes(),
            );
        }

        // 截断：只保留窗口 [1,7]（帧 0 状态被逐出）——续算帧 2..=7 所需
        // 的上一帧键均在窗口内（帧 2 依赖帧 1 状态，其后逐帧刚重算写出）。
        let mut windowed = SnapshotStore::new();
        for frame in 1..=7u64 {
            windowed.insert(
                (TEST_SIG, frame),
                payload_to_bytes(&Payload::Scalar(frame as f32)),
            );
        }
        let mut resume = EvalCtx::new(11, 0, 0);
        resume.snapshots = windowed;
        for frame in 2..=7u64 {
            resume.frame = frame;
            let bytes = eval(&g, &registry, &plane, TEST_SIG, &mut resume)
                .unwrap()
                .to_canonical_bytes();
            assert_eq!(
                bytes, full_bytes[frame as usize],
                "窗口 [1,7] 续算与全量重放逐位一致（帧 {frame}）"
            );
        }
        assert_eq!(
            resume.snapshots.get(&(TEST_SIG, 7)),
            full.snapshots.get(&(TEST_SIG, 7)),
            "终态快照逐位一致"
        );

        // 逐出跨过依赖跨度：窗口 [5,7] 求帧 5 → 上一帧键 (sig,4) 缺失。
        let mut gap = SnapshotStore::new();
        for frame in 5..=7u64 {
            gap.insert(
                (TEST_SIG, frame),
                payload_to_bytes(&Payload::Scalar(frame as f32)),
            );
        }
        let mut cold_resume = EvalCtx::new(11, 0, 5);
        cold_resume.snapshots = gap;
        let err = eval(&g, &registry, &plane, TEST_SIG, &mut cold_resume).unwrap_err();
        assert_eq!(
            err.code, "P3012",
            "窗口不含帧 4 时求帧 5 必报错，绝不静默续错"
        );
    }

    #[test]
    fn stream_snapshot_key_includes_channel_definition_chain() {
        // 流层键闭包（S3）：快照键 = (snapshot_signature(子图签名, 通道定义链),
        // 帧)——通道定义变更 → 链变更 → 键变更 → 旧快照绝不误用。
        let mut g = Graph::new();
        let mut n = Node::new(9, StructureId::ConstraintFlow, Composition::Feed);
        n.bindings.push(NodeBinding {
            param: "drive".to_string(),
            channel: ChannelRef { name: "drive".to_string() },
        });
        g.add_node(n).unwrap();

        let defs_v1: ChannelDefs = [(
            "drive".to_string(),
            ChannelDef { tag: 3, def_bytes: b"expr:sin(t)*2".to_vec() },
        )]
        .into_iter()
        .collect();
        let defs_v2: ChannelDefs = [(
            "drive".to_string(),
            ChannelDef { tag: 3, def_bytes: b"expr:sin(t)*3".to_vec() },
        )]
        .into_iter()
        .collect();

        let registry = test_registry();
        let plane = SamplePlane { pixel: (0, 0), spp: 1 };

        // 子图签名 + 通道定义链 → 快照签名位（D-28 键的签名位来源）。
        let sub_v1 = node_signature(&g, 9, &defs_v1).unwrap();
        let sub_v2 = node_signature(&g, 9, &defs_v2).unwrap();
        assert_ne!(sub_v1, sub_v2, "节点签名已含通道变化特征");
        let chain_v1 = [channel_definition_hash(&defs_v1["drive"])];
        let chain_v2 = [channel_definition_hash(&defs_v2["drive"])];
        let sig_v1 = snapshot_signature(&sub_v1, &chain_v1);
        let sig_v2 = snapshot_signature(&sub_v2, &chain_v2);
        assert_ne!(
            sig_v1, sig_v2,
            "通道定义变更必须改变快照签名（否则旧快照误复用）"
        );

        // v1 顺序预热 0..=2。
        let mut warm = EvalCtx::new(11, 0, 0);
        for frame in 0..=2u64 {
            warm.frame = frame;
            eval(&g, &registry, &plane, sig_v1, &mut warm).unwrap();
        }
        assert_eq!(warm.snapshots.len(), 3);

        // 通道定义改版后以 v2 签名 seek 帧 2：旧键不在 v2 命名空间
        // → P3012 结构化报错，绝不跨版本误用旧状态。
        let mut cold_v2 = EvalCtx::new(11, 0, 2);
        cold_v2.snapshots = warm.snapshots.clone();
        let err = eval(&g, &registry, &plane, sig_v2, &mut cold_v2).unwrap_err();
        assert_eq!(
            err.code, "P3012",
            "通道定义变更后旧快照不得误用：{}",
            err
        );

        // v2 从 0 帧重新预热：键命名空间全新，与 v1 历史互不污染。
        let mut redo_v2 = EvalCtx::new(11, 0, 0);
        let out0 = eval(&g, &registry, &plane, sig_v2, &mut redo_v2).unwrap();
        assert_eq!(out0.get(9), Some(Payload::Scalar(0.0)));
        redo_v2.frame = 1;
        let out1 = eval(&g, &registry, &plane, sig_v2, &mut redo_v2).unwrap();
        assert_eq!(out1.get(9), Some(Payload::Scalar(1.0)), "v2 命名空间独立演进");
    }
}
