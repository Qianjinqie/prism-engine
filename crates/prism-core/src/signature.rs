//! 子图签名树（S10 内容↔摘要，A6）。
//!
//! 语义权威：（节点签名 = hash(StructureId, 特解参数
//! 序列化, 各子签名, 属性通道变化特征)；"属性通道变化特征" = 通道定义哈希，
//! 非瞬时值；缓存键五元组 + tile 维度；256 位哈希以无碰撞为公理）、/D-28
//! （模拟快照键 = (子图签名, 帧)，签名含通道定义哈希）、（子树哈希
//! 自底向上：改一个节点只影响其自身到根的祖先链）、宪章 C13（缓存键含
//! schema_version 与 engine_version，旧版本缓存只失效重建、永不误复用）。
//!
//! # 本批实现
//! - **canonical 序列化**：图 → 规范字节序列（节点按 id 升序、字段定长/长度
//!   前缀、大端位模式——与 [`crate::node::Params::to_canonical_bytes`] 同风格）；
//! - **子树哈希**：自底向上，处理序 = [`Graph::topo_order`]（id 稳定 tie-break，
//!   环图结构化报错不静默）。叶子 = hash(StructureId, params canonical, 通道
//!   定义哈希)；内部节点在此基础上追加各子签名（槽位升序）——**改一个节点只
//!   影响其自身到根的祖先链，兄弟子树签名逐位不变**（C3 增量渲染的根基）；
//! - **通道定义哈希**：本批定义输入接口 [`ChannelDef`]（枚举 tag + 定义
//!   canonical 字节），property.rs（并行轮次）落地后由其通道枚举直接产出；
//!   本批不 import property.rs（任务红线）；
//! - **缓存键**：hash(schema_version, engine_version, 子图签名, 分辨率, spp 档,
//!  tile)（五元组 + tile；C13 版本入键）；
//! - **快照咬合**：签名类型即 [`crate::node::Hash256`]（[u8; 32]），直接充当
//!   [`crate::node::SnapshotKey`] 的签名位，键类型零改动（D-28）。
//!
//! # 与公式的偏差声明（有意为之，评审留档）
//! 公式列举四要素未含 Composition 与 NodeId；两者均进入求值输出（`Merge`
//! 算子决定归约语义；`node_id` 经 `EvalCtx`/`SampleIndex` 进入随机样本流），
//! 漏项将破坏"同签名 ⇒ 逐字节可复用"公理，故按公理补全。补全不影响
//! 增量性：id 与组合方式在参数编辑期间不变，祖先链影响域不变。
//!
//! # SHA-256 双实现说明（规范缺口，上报项）
//!  将 SHA-256 安放在 prism-io，而依赖方向（io∈L4 → core∈L2）
//! 禁止 prism-core 依赖 prism-io。故本文件内置模块私有的同一 FIPS 180-4 实现
//! （整数运算，零依赖；与 prism-io::sha256 以同一组 FIPS 官方向量分别验收）。
//! 后续轮次建议把 SHA-256 下沉 prism-core（或 prism-math）单一实现，prism-io
//! 改为依赖复用——涉及 Cargo.toml，超出本批文件红线，仅上报不私改。

use std::collections::BTreeMap;
use std::fmt;

use crate::graph::{Graph, GraphError};
use crate::node::{Composition, CsgOp, Hash256, MergeOp, Node, NodeBinding, NodeId, StructureId};

// ---------------------------------------------------------------------------
// 模块私有 SHA-256（FIPS 180-4；与 prism-io::sha256 同算法同向量，见模块头说明）
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

/// 处理一个 512 位消息块（FIPS 180-4 §6.2.2；仅整数运算，零 libm）。
fn compress(h: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
        *word = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h7] = *h;
    for i in 0..64 {
        let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h7
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = big_s0.wrapping_add(maj);
        h7 = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    let rounds_out = [a, b, c, d, e, f, g, h7];
    for (slot, value) in h.iter_mut().zip(rounds_out.iter()) {
        *slot = slot.wrapping_add(*value);
    }
}

/// SHA-256（FIPS 180-4）：模块私有，供本文件的签名/键计算使用。
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
    let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        compress(&mut h, block);
    }
    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

// ---------------------------------------------------------------------------
// schema 版本与域分隔标签
// ---------------------------------------------------------------------------

/// canonical 序列化 / 签名 schema 版本（宪章 C13/R3：版本变更 → 全部缓存
/// 签名失效重建，永不误复用）。版本演进纪律：任何 canonical 字节布局或签名
/// 输入要素的变更都必须递增本值。
pub const SIGNATURE_SCHEMA_VERSION: u32 = 1;

/// canonical 字节头（宪章 C8：内部格式带版本头；静默失效重建 ≠ 用户数据丢失）。
const CANON_MAGIC: &[u8; 10] = b"PRISMCANON";

/// 域分隔标签：不同用途的哈希输入各挂唯一前缀，防跨域拼接歧义。
const NODE_TAG: &[u8; 14] = b"PRISM-SIG-NODE";
const GRAPH_TAG: &[u8; 15] = b"PRISM-SIG-GRAPH";
const CACHE_KEY_TAG: &[u8; 15] = b"PRISM-CACHE-KEY";
const SNAPSHOT_TAG: &[u8; 14] = b"PRISM-SNAPSHOT";

// ---------------------------------------------------------------------------
// 通道定义哈希（"属性通道变化特征"的输入接口）
// ---------------------------------------------------------------------------

/// 属性通道定义（签名侧视角的最小形状）。
///
/// "属性通道变化特征 = 通道定义（规格/AST）哈希，非瞬时值"：
/// 瞬时值相同而定义不同的通道必须不同签名，否则模拟快照误复用。枚举本体
/// （Const/Track/Spring/Expr/Wiggle/Remap/Loop …）与判别值归 property.rs
/// （并行签名）；当前版本只按 `tag + def_bytes` 消费字节，不 import property.rs。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelDef {
    /// 通道种类枚举 tag（通道枚举的判别值；u16 大端进哈希）。
    pub tag: u16,
    /// 通道定义参数的 canonical 字节（枚举 tag 之后的部分，由 A2 侧按其
    /// canonical 风格产出；内容变更 → 哈希变更 → 祖先链签名失效）。
    pub def_bytes: Vec<u8>,
}

/// 通道定义表：通道名 → 定义（节点绑定的 `ChannelRef` 按名解析至此）。
pub type ChannelDefs = BTreeMap<String, ChannelDef>;

/// 通道定义哈希：hash(枚举 tag 大端 ++ 定义 canonical 字节)。
pub fn channel_definition_hash(def: &ChannelDef) -> Hash256 {
    let mut buf = Vec::with_capacity(2 + def.def_bytes.len());
    buf.extend_from_slice(&def.tag.to_be_bytes());
    buf.extend_from_slice(&def.def_bytes);
    sha256(&buf)
}

// ---------------------------------------------------------------------------
// 签名错误（宪章 C18 三段式；C19：必含建议段）
// ---------------------------------------------------------------------------

/// 签名计算错误。错误码取 P3 段（图结构/悬空引用族，宪章 C18 分段）。
#[derive(Debug, Clone, PartialEq)]
pub enum SignatureError {
    /// 图结构错误透传（环等，来自 `Graph::topo_order`）。
    Graph(GraphError),
    /// 绑定引用的通道定义不存在（P3012，悬空引用；永不静默，宪章 C20）。
    UnknownChannel {
        /// 三段式一【在哪】。
        location: String,
        /// 三段式二【什么事】：缺失定义的通道名。
        channel: String,
    },
}

impl fmt::Display for SignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignatureError::Graph(e) => write!(f, "{e}"),
            SignatureError::UnknownChannel { location, channel } => write!(
                f,
                "error[P3012] {location}\n  什么事：绑定引用的通道 \"{channel}\" 在通道定义表中不存在\n  建议：在 ChannelDefs 表中补齐该通道的定义（tag + def_bytes），或移除该绑定"
            ),
        }
    }
}

impl std::error::Error for SignatureError {}

impl From<GraphError> for SignatureError {
    fn from(e: GraphError) -> Self {
        SignatureError::Graph(e)
    }
}

// ---------------------------------------------------------------------------
// canonical 序列化（节点按 id 序、字段定长/长度前缀、大端位模式）
// ---------------------------------------------------------------------------

/// 结构位 canonical 字节：官方段 = tag 0 + 序号 u16 大端；用户段 = tag 1 +
/// 编号 u16 大端 + 长度前缀 crate 名（跨 crate 隔离进签名， L4）。
fn structure_bytes(id: &StructureId) -> Vec<u8> {
    let mut out = Vec::new();
    match id.ordinal() {
        Some(ordinal) => {
            out.push(0);
            out.extend_from_slice(&ordinal.to_be_bytes());
        }
        None => match id {
            StructureId::Custom { n, crate_name } => {
                out.push(1);
                out.extend_from_slice(&n.to_be_bytes());
                out.extend_from_slice(&(crate_name.len() as u16).to_be_bytes());
                out.extend_from_slice(crate_name.as_bytes());
            }
            _ => unreachable!("有 ordinal 的官方结构不会进入 None 分支"),
        },
    }
    out
}

/// 组合方式位 canonical 字节：变体 tag u8 + 变体负载；浮点按位模式大端
/// （与 `Params::to_canonical_bytes` 的 `to_bits().to_be_bytes()` 同风格）。
fn composition_bytes(c: &Composition) -> Vec<u8> {
    let mut out = Vec::new();
    match c {
        Composition::Feed => out.push(0),
        Composition::Chain => out.push(1),
        Composition::Merge(op) => {
            out.push(2);
            out.push(match op {
                MergeOp::Over => 0,
                MergeOp::Plus => 1,
                MergeOp::Screen => 2,
                MergeOp::Multiply => 3,
                MergeOp::Mask => 4,
            });
        }
        Composition::Csg { op, k } => {
            out.push(3);
            out.push(match op {
                CsgOp::Union => 0,
                CsgOp::Intersect => 1,
                CsgOp::Subtract => 2,
                CsgOp::SmoothUnion => 3,
                CsgOp::SmoothIntersect => 4,
                CsgOp::SmoothSubtract => 5,
            });
            out.extend_from_slice(&k.to_bits().to_be_bytes());
        }
        Composition::Mis(w) => {
            out.push(4);
            out.extend_from_slice(&w.to_bits().to_be_bytes());
        }
        Composition::Dual => out.push(5),
    }
    out
}

/// 单节点 canonical 字节：id + 结构 + 组合方式 + 参数包 + 参数绑定。
///
/// 绑定按 (param, channel) 字典序排序——绑定集合语义无序，canonical 输出
/// 与插入序无关。
pub fn node_canonical_bytes(node: &Node) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&node.id.to_be_bytes());
    out.extend_from_slice(&structure_bytes(&node.structure));
    out.extend_from_slice(&composition_bytes(&node.composition));

    let params = node.params.to_canonical_bytes();
    out.extend_from_slice(&(params.len() as u32).to_be_bytes());
    out.extend_from_slice(&params);

    let mut bindings: Vec<(&str, &str)> = node
        .bindings
        .iter()
        .map(|b| (b.param.as_str(), b.channel.name.as_str()))
        .collect();
    bindings.sort_unstable();
    out.extend_from_slice(&(bindings.len() as u32).to_be_bytes());
    for (param, channel) in bindings {
        out.extend_from_slice(&(param.len() as u16).to_be_bytes());
        out.extend_from_slice(param.as_bytes());
        out.extend_from_slice(&(channel.len() as u16).to_be_bytes());
        out.extend_from_slice(channel.as_bytes());
    }
    out
}

/// 全图 canonical 字节：头（magic + schema 版本）+ 节点数 + 各节点 canonical
/// 字节（按 id 升序，BTreeMap 键序）+ 边数 + 各边（按 (目标节点, 槽位) 升序）。
///
/// 绑定在此按名记录（存档语义 D-23：定义跟着代码走）；通道定义**内容**不进
/// canonical 字节、只进签名（见 [`graph_signature`]）。
pub fn graph_canonical_bytes(graph: &Graph) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(CANON_MAGIC);
    out.extend_from_slice(&SIGNATURE_SCHEMA_VERSION.to_be_bytes());
    out.extend_from_slice(&(graph.len() as u64).to_be_bytes());
    for node in graph.nodes() {
        out.extend_from_slice(&node_canonical_bytes(node));
    }

    let mut edge_count: u64 = 0;
    for node in graph.nodes() {
        edge_count += graph.inputs_of(node.id).len() as u64;
    }
    out.extend_from_slice(&edge_count.to_be_bytes());
    for node in graph.nodes() {
        // inputs_of 已按槽位升序；外层节点 id 升序 ⇒ (to, input) 全序。
        for (input, from) in graph.inputs_of(node.id) {
            out.extend_from_slice(&node.id.to_be_bytes());
            out.push(input);
            out.extend_from_slice(&from.to_be_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 子树哈希（自底向上）
// ---------------------------------------------------------------------------

/// 全图各节点签名：节点 id → 子树签名（自底向上，处理序 = 拓扑序）。
///
/// 节点签名 = hash(域分隔 tag, schema 版本, node_id, StructureId, Composition,
/// params canonical, 通道变化特征, 各子签名槽位升序)。改一个节点 → 仅其自身
/// 到根的祖先链签名变化，兄弟子树签名逐位不变。
pub fn graph_signatures(
    graph: &Graph,
    defs: &ChannelDefs,
) -> Result<BTreeMap<NodeId, Hash256>, SignatureError> {
    // 拓扑序：环图 → GraphError 透传（P3005），永不静默/死循环（C20）。
    let order = graph.topo_order()?;
    let mut sigs: BTreeMap<NodeId, Hash256> = BTreeMap::new();

    for id in order {
        let node = graph.node(id).expect("拓扑序覆盖全部节点");

        // 属性通道变化特征：绑定按 (param, channel) 排序，逐条拼入
        // (param 名, channel 名, 通道定义哈希)；缺失定义结构化报错（C20）。
        let mut bindings: Vec<&NodeBinding> = node.bindings.iter().collect();
        bindings.sort_by(|a, b| {
            (a.param.as_str(), a.channel.name.as_str())
                .cmp(&(b.param.as_str(), b.channel.name.as_str()))
        });
        let mut feature = Vec::new();
        feature.extend_from_slice(&(bindings.len() as u32).to_be_bytes());
        for binding in &bindings {
            let def = defs.get(&binding.channel.name).ok_or_else(|| {
                SignatureError::UnknownChannel {
                    location: format!("signature.节点 {}（绑定参数 {}）", node.id, binding.param),
                    channel: binding.channel.name.clone(),
                }
            })?;
            feature.extend_from_slice(&(binding.param.len() as u16).to_be_bytes());
            feature.extend_from_slice(binding.param.as_bytes());
            feature.extend_from_slice(&(binding.channel.name.len() as u16).to_be_bytes());
            feature.extend_from_slice(binding.channel.name.as_bytes());
            feature.extend_from_slice(&channel_definition_hash(def));
        }

        // 各子签名（输入槽位升序；拓扑序保证子节点已算完）。
        let inputs = graph.inputs_of(id);
        let mut child_bytes = Vec::with_capacity(inputs.len() * 32 + 4);
        child_bytes.extend_from_slice(&(inputs.len() as u32).to_be_bytes());
        for (_, from) in &inputs {
            let child_sig = sigs
                .get(from)
                .expect("拓扑序保证上游节点先于下游节点处理");
            child_bytes.extend_from_slice(child_sig);
        }

        let params = node.params.to_canonical_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(NODE_TAG);
        buf.extend_from_slice(&SIGNATURE_SCHEMA_VERSION.to_be_bytes());
        buf.extend_from_slice(&node.id.to_be_bytes());
        buf.extend_from_slice(&structure_bytes(&node.structure));
        buf.extend_from_slice(&composition_bytes(&node.composition));
        buf.extend_from_slice(&(params.len() as u32).to_be_bytes());
        buf.extend_from_slice(&params);
        buf.extend_from_slice(&feature);
        buf.extend_from_slice(&child_bytes);
        sigs.insert(id, sha256(&buf));
    }
    Ok(sigs)
}

/// 单节点（子图祖先锥）签名。子图签名即该节点子树签名的 Merle 摘要——
/// 缓存键与模拟快照键（D-28）的"子图签名"位。
///
/// # Errors
/// 图含环或绑定引用未定义通道时结构化报错。
pub fn node_signature(
    graph: &Graph,
    id: NodeId,
    defs: &ChannelDefs,
) -> Result<Hash256, SignatureError> {
    let mut sigs = graph_signatures(graph, defs)?;
    sigs.remove(&id).ok_or_else(|| SignatureError::Graph(GraphError {
        code: "P3013",
        location: format!("signature.node_signature(节点 {id})"),
        problem: format!("节点 id {id} 不在图中，无法计算签名"),
        fix: "先 add_node 加入节点，再请求其签名".to_string(),
    }))
}

/// 全图签名：hash(域分隔 tag, schema 版本, 节点数, Σ(节点 id ++ 节点签名))，
/// 节点按 id 升序。边结构已由各节点签名的子签名序列（槽位序）完全承载，
/// 不重复入哈希。
pub fn graph_signature(graph: &Graph, defs: &ChannelDefs) -> Result<Hash256, SignatureError> {
    let sigs = graph_signatures(graph, defs)?;
    let mut buf = Vec::with_capacity(graph.len() * 40 + 20);
    buf.extend_from_slice(GRAPH_TAG);
    buf.extend_from_slice(&SIGNATURE_SCHEMA_VERSION.to_be_bytes());
    buf.extend_from_slice(&(graph.len() as u64).to_be_bytes());
    for node in graph.nodes() {
        buf.extend_from_slice(&node.id.to_be_bytes());
        buf.extend_from_slice(&sigs[&node.id]);
    }
    Ok(sha256(&buf))
}

// ---------------------------------------------------------------------------
// 缓存键与快照键咬合
// ---------------------------------------------------------------------------

/// 缓存键渲染维度（五元组 + tile + 帧）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheDims {
    /// 渲染分辨率宽（像素）。
    pub width: u32,
    /// 渲染分辨率高（像素）。
    pub height: u32,
    /// spp 档位（草稿/标准/电影 → 档号；逼近阶显式标价）。
    pub spp_tier: u32,
    /// tile 坐标（tile 维度，与 query 一致）。
    pub tile: (u32, u32),
    /// 引擎版本（调用方把语义化版本打包为 u64，如 major<<32 | minor<<16 | patch）。
    pub engine_version: u64,
    /// 帧号（D-28 快照键 = (子图签名, 帧)；增量复用的**帧维**必须可区分）。
    ///
    /// 缺此字段时不同帧同子图同维度 → 同键 → 跨帧误命中（帧 3 取回帧 2 的
    /// 字节），确定性铁律帧维击穿。故帧必须纳入 cache_key 摘要。
    pub frame: u64,
}

/// 增量渲染缓存键 = hash(域分隔 tag, schema_version, engine_version, 子图签名,
/// 分辨率, spp 档, tile, 帧)（+ 宪章 C13 + D-28：旧 schema / 旧版本 /
/// 旧维度 / **旧帧**的缓存永不误复用，只会静默失效重建）。
pub fn cache_key(subgraph: &Hash256, dims: &CacheDims) -> Hash256 {
    let mut buf = Vec::with_capacity(8 + 32 + 28);
    buf.extend_from_slice(CACHE_KEY_TAG);
    buf.extend_from_slice(&SIGNATURE_SCHEMA_VERSION.to_be_bytes());
    buf.extend_from_slice(&dims.engine_version.to_be_bytes());
    buf.extend_from_slice(subgraph);
    buf.extend_from_slice(&dims.width.to_be_bytes());
    buf.extend_from_slice(&dims.height.to_be_bytes());
    buf.extend_from_slice(&dims.spp_tier.to_be_bytes());
    buf.extend_from_slice(&dims.tile.0.to_be_bytes());
    buf.extend_from_slice(&dims.tile.1.to_be_bytes());
    buf.extend_from_slice(&dims.frame.to_be_bytes());
    sha256(&buf)
}

/// 模拟节点快照键的签名位（/ D-28）：hash(子图签名 ++ 自起点以来
/// 通道定义签名链)。链为空时输出仍区别于裸子图签名（域分隔）。与
/// [`crate::node::SnapshotKey`] = (本函数输出, 帧) 咬合使用。
pub fn snapshot_signature(subgraph: &Hash256, channel_chain: &[Hash256]) -> Hash256 {
    let mut buf = Vec::with_capacity(32 + channel_chain.len() * 32);
    buf.extend_from_slice(SNAPSHOT_TAG);
    buf.extend_from_slice(subgraph);
    for link in channel_chain {
        buf.extend_from_slice(link);
    }
    sha256(&buf)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{NodeBinding, ChannelRef, ParamValue};

    /// 以 (tag, 字节) 对快速构造通道定义表。
    fn defs(pairs: &[(&str, u16, &[u8])]) -> ChannelDefs {
        pairs
            .iter()
            .map(|(name, tag, bytes)| {
                (
                    name.to_string(),
                    ChannelDef { tag: *tag, def_bytes: bytes.to_vec() },
                )
            })
            .collect()
    }

    /// 构造叶子节点。
    fn leaf(id: NodeId, structure: StructureId, x: f32) -> Node {
        let mut node = Node::new(id, structure, Composition::Feed);
        node.params.set("x", ParamValue::F32(x));
        node
    }

    /// 汉明距离（雪崩 smoke 用）。
    fn hamming(a: &Hash256, b: &Hash256) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    // ------------------------------------------------------------------
    // 关卡 1：SHA-256 FIPS 官方向量（prism-core 私有实现同向量验收）
    // ------------------------------------------------------------------

    #[test]
    fn sha256_matches_fips_vectors() {
        let hex = |d: &Hash256| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&sha256(&million)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    // ------------------------------------------------------------------
    // 关卡 2：canonical 序列化——同图同字节，插入序无关
    // ------------------------------------------------------------------

    /// 等价构建 A：插入序 1 → 2 → 3 → 4，参数/绑定按一种顺序装配。
    fn build_graph_a() -> Graph {
        let mut g = Graph::new();
        let mut n1 = leaf(1, StructureId::ScalarField, 0.5);
        n1.params.set("octaves", ParamValue::U32(4));
        let mut n2 = leaf(2, StructureId::Convolution, 2.0);
        n2.params.set("radius", ParamValue::F32(3.0));
        let mut n3 = Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
        n3.params.set("strength", ParamValue::F32(0.7));
        n3.bindings.push(NodeBinding {
            param: "bias".into(),
            channel: ChannelRef { name: "wind".into() },
        });
        n3.bindings.push(NodeBinding {
            param: "gain".into(),
            channel: ChannelRef { name: "pulse".into() },
        });
        let mut n4 = Node::new(4, StructureId::Convolution, Composition::Chain);
        n4.params.set("radius", ParamValue::F32(1.5));

        g.add_node(n1).unwrap();
        g.add_node(n2).unwrap();
        g.add_node(n3).unwrap();
        g.add_node(n4).unwrap();
        g.connect(1, 3, 0).unwrap();
        g.connect(2, 3, 1).unwrap();
        g.connect(3, 4, 0).unwrap();
        g
    }

    /// 等价构建 B：插入序倒置（4 → 3 → 2 → 1），参数/绑定装配序不同，
    /// 内容与连线与 A 完全等价。
    fn build_graph_b() -> Graph {
        let mut g = Graph::new();
        let mut n4 = Node::new(4, StructureId::Convolution, Composition::Chain);
        n4.params.set("radius", ParamValue::F32(1.5));
        let mut n3 = Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
        n3.params.set("strength", ParamValue::F32(0.7));
        n3.bindings.push(NodeBinding {
            param: "gain".into(),
            channel: ChannelRef { name: "pulse".into() },
        });
        n3.bindings.push(NodeBinding {
            param: "bias".into(),
            channel: ChannelRef { name: "wind".into() },
        });
        let mut n2 = leaf(2, StructureId::Convolution, 2.0);
        n2.params.set("radius", ParamValue::F32(3.0));
        let mut n1 = leaf(1, StructureId::ScalarField, 0.5);
        n1.params.set("octaves", ParamValue::U32(4));

        g.add_node(n4).unwrap();
        g.add_node(n3).unwrap();
        g.add_node(n2).unwrap();
        g.add_node(n1).unwrap();
        g.connect(3, 4, 0).unwrap();
        g.connect(2, 3, 1).unwrap();
        g.connect(1, 3, 0).unwrap();
        g
    }

    #[test]
    fn canonical_bytes_identical_for_equivalent_builds() {
        let a = graph_canonical_bytes(&build_graph_a());
        let b = graph_canonical_bytes(&build_graph_b());
        assert_eq!(a, b, "同图等价构建必得同 canonical 字节（插入序无关）");
        assert_eq!(&a[0..10], b"PRISMCANON", "canonical 字节应带版本头（C8）");

        // 内容变化 → 字节变化（canonical 字节是内容的单射指纹）。
        let mut g = build_graph_a();
        let mut n1 = leaf(1, StructureId::ScalarField, 0.9);
        n1.params.set("octaves", ParamValue::U32(4));
        g.add_node(n1).unwrap_err(); // id 冲突按预期被拒（图内容未被改动）
        let mut g_changed = Graph::new();
        let mut n1c = leaf(1, StructureId::ScalarField, 0.9);
        n1c.params.set("octaves", ParamValue::U32(4));
        g_changed.add_node(n1c).unwrap();
        g_changed.add_node(leaf(2, StructureId::Convolution, 2.0)).unwrap();
        let mut n3 = Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
        n3.params.set("strength", ParamValue::F32(0.7));
        n3.bindings.push(NodeBinding {
            param: "bias".into(),
            channel: ChannelRef { name: "wind".into() },
        });
        n3.bindings.push(NodeBinding {
            param: "gain".into(),
            channel: ChannelRef { name: "pulse".into() },
        });
        let mut n4 = Node::new(4, StructureId::Convolution, Composition::Chain);
        n4.params.set("radius", ParamValue::F32(1.5));
        g_changed.add_node(n3).unwrap();
        g_changed.add_node(n4).unwrap();
        g_changed.connect(1, 3, 0).unwrap();
        g_changed.connect(2, 3, 1).unwrap();
        g_changed.connect(3, 4, 0).unwrap();
        assert_ne!(
            graph_canonical_bytes(&g_changed),
            a,
            "节点参数值不同 → canonical 字节不同"
        );
    }

    #[test]
    fn node_canonical_bytes_insertion_order_independent() {
        let mut p1 = Node::new(7, StructureId::RadiativeTransfer, Composition::Csg {
            op: CsgOp::SmoothUnion,
            k: 0.25,
        });
        p1.params.set("ior", ParamValue::F32(1.5));
        p1.params.set("rough", ParamValue::F32(0.1));
        p1.bindings.push(NodeBinding {
            param: "a".into(),
            channel: ChannelRef { name: "ch1".into() },
        });
        p1.bindings.push(NodeBinding {
            param: "b".into(),
            channel: ChannelRef { name: "ch2".into() },
        });

        let mut p2 = Node::new(7, StructureId::RadiativeTransfer, Composition::Csg {
            op: CsgOp::SmoothUnion,
            k: 0.25,
        });
        p2.params.set("rough", ParamValue::F32(0.1));
        p2.params.set("ior", ParamValue::F32(1.5));
        p2.bindings.push(NodeBinding {
            param: "b".into(),
            channel: ChannelRef { name: "ch2".into() },
        });
        p2.bindings.push(NodeBinding {
            param: "a".into(),
            channel: ChannelRef { name: "ch1".into() },
        });

        assert_eq!(
            node_canonical_bytes(&p1),
            node_canonical_bytes(&p2),
            "参数与绑定的装配序不影响 canonical 字节"
        );

        // 组合方式差异必须反映在字节中（Merge 算子是求值语义的一部分）。
        let over = Node::new(7, StructureId::Diffusion, Composition::Merge(MergeOp::Over));
        let plus = Node::new(7, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
        assert_ne!(node_canonical_bytes(&over), node_canonical_bytes(&plus));
    }

    // ------------------------------------------------------------------
    // 关卡 3（核心）：子树哈希增量性——改叶子只影响其到根链
    // ------------------------------------------------------------------

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

    #[test]
    fn leaf_param_change_only_affects_ancestor_chain() {
        let empty = ChannelDefs::new();
        let base = graph_signatures(&chain_graph(0.5), &empty).unwrap();
        let changed = graph_signatures(&chain_graph(0.75), &empty).unwrap();

        // 改叶 1 参数：仅 {1, 3, 4}（叶到根链）签名变化。
        assert_ne!(base[&1], changed[&1], "叶自身签名必须变化");
        assert_ne!(base[&3], changed[&3], "父节点签名必须变化");
        assert_ne!(base[&4], changed[&4], "根签名必须变化");

        // 兄弟子树逐位不变（增量渲染的根基）。
        assert_eq!(
            base[&2], changed[&2],
            "兄弟子树签名必须逐位不变"
        );

        // 还原参数 → 全部签名逐位还原（同输入同输出，确定性）。
        let restored = graph_signatures(&chain_graph(0.5), &empty).unwrap();
        assert_eq!(base, restored);
    }

    #[test]
    fn channel_definition_change_propagates_up_the_chain() {
        // 通道定义（非瞬时值）变更 → 仅引用该通道的节点及其祖先链失效。
        // 正确姿势：重建含绑定的图（1、2 为叶，3 绑定 wind 通道）。
        let mut g = Graph::new();
        g.add_node(leaf(1, StructureId::ScalarField, 0.5)).unwrap();
        g.add_node(leaf(2, StructureId::Convolution, 2.0)).unwrap();
        let mut n3 = Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus));
        n3.bindings.push(NodeBinding {
            param: "strength".into(),
            channel: ChannelRef { name: "wind".into() },
        });
        g.add_node(n3).unwrap();
        g.connect(1, 3, 0).unwrap();
        g.connect(2, 3, 1).unwrap();

        let d1 = defs(&[("wind", 3, b"expr:sin(t*3)*8")]);
        let d2 = defs(&[("wind", 3, b"expr:sin(t*4)*8")]); // 同 tag，定义字节不同
        let s1 = graph_signatures(&g, &d1).unwrap();
        let s2 = graph_signatures(&g, &d2).unwrap();
        assert_ne!(s1[&3], s2[&3], "通道定义变更必须改变引用节点的签名");
        assert_ne!(s1[&3], s2[&3]);
        assert_eq!(s1[&1], s2[&1], "未引用该通道的兄弟子树签名不变");
        assert_eq!(s1[&2], s2[&2], "未引用该通道的兄弟子树签名不变");

        // 定义字节相同、枚举 tag 不同 → 定义哈希不同（瞬时值相同≠定义相同）。
        let d3 = defs(&[("wind", 4, b"expr:sin(t*3)*8")]);
        let s3 = graph_signatures(&g, &d3).unwrap();
        assert_ne!(s1[&3], s3[&3], "枚举 tag 属通道定义，进哈希");
    }

    #[test]
    fn unknown_channel_is_structured_error_not_silent() {
        let mut g = Graph::new();
        let mut n = leaf(1, StructureId::ScalarField, 1.0);
        n.bindings.push(NodeBinding {
            param: "x".into(),
            channel: ChannelRef { name: "missing".into() },
        });
        g.add_node(n).unwrap();

        let err = graph_signatures(&g, &ChannelDefs::new()).unwrap_err();
        match &err {
            SignatureError::UnknownChannel { channel, .. } => {
                assert_eq!(channel, "missing");
            }
            other => panic!("应为 UnknownChannel 错误，实际 {other:?}"),
        }
        let text = err.to_string();
        assert!(text.starts_with("error[P3012] "), "三段式首段【在哪】：{text}");
        assert!(text.contains("missing"), "应指出缺失通道名：{text}");
        assert!(text.contains("建议："), "必含建议段（C19）：{text}");
    }

    #[test]
    fn cyclic_graph_is_rejected_not_looped() {
        let mut g = Graph::new();
        for id in [1u64, 2u64] {
            g.add_node(Node::new(id, StructureId::ScalarField, Composition::Feed))
                .unwrap();
        }
        g.connect(1, 2, 0).unwrap();
        g.connect(2, 1, 0).unwrap();

        let err = graph_signatures(&g, &ChannelDefs::new()).unwrap_err();
        match &err {
            SignatureError::Graph(ge) => assert_eq!(ge.code, "P3005", "环图透传拓扑错误"),
            other => panic!("应为 Graph 错误，实际 {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // 关卡 4：同签名同输出前提 / 异签名必异（雪崩 smoke）
    // ------------------------------------------------------------------

    #[test]
    fn equivalent_graphs_have_identical_signatures() {
        let defs_table = defs(&[("wind", 3, b"w"), ("pulse", 5, b"p")]);
        let sigs_a = graph_signatures(&build_graph_a(), &defs_table).unwrap();
        let sigs_b = graph_signatures(&build_graph_b(), &defs_table).unwrap();
        assert_eq!(sigs_a, sigs_b, "等价图全节点签名一致");
        assert_eq!(
            graph_signature(&build_graph_a(), &defs_table).unwrap(),
            graph_signature(&build_graph_b(), &defs_table).unwrap(),
            "等价图全图签名一致"
        );
    }

    #[test]
    fn single_byte_param_change_avalanches() {
        // 单字节改动：F32 位模式 0x3F800000 → 0x3F800001（canonical 尾字节）。
        let g1 = chain_graph(f32::from_bits(0x3F80_0000));
        let g2 = chain_graph(f32::from_bits(0x3F80_0001));
        let empty = ChannelDefs::new();
        let s1 = graph_signatures(&g1, &empty).unwrap();
        let s2 = graph_signatures(&g2, &empty).unwrap();

        let d = hamming(&s1[&1], &s2[&1]);
        assert!(
            d > 64 && d < 192,
            "单字节改动后签名汉明距离应落在随机分布合理区间，实际 {d}/256"
        );
        // 全图签名同样雪崩。
        let gd = hamming(
            &graph_signature(&g1, &empty).unwrap(),
            &graph_signature(&g2, &empty).unwrap(),
        );
        assert!(gd > 64 && gd < 192, "全图签名雪崩距离应合理，实际 {gd}/256");
    }

    #[test]
    fn composition_id_and_structure_changes_alter_signature() {
        let empty = ChannelDefs::new();

        // 组合方式不同（Over vs Plus）→ 签名不同（Merge 算子决定归约语义）。
        let mut ga = Graph::new();
        let mut gb = Graph::new();
        ga.add_node(Node::new(1, StructureId::Diffusion, Composition::Merge(MergeOp::Over)))
            .unwrap();
        gb.add_node(Node::new(1, StructureId::Diffusion, Composition::Merge(MergeOp::Plus)))
            .unwrap();
        assert_ne!(
            node_signature(&ga, 1, &empty).unwrap(),
            node_signature(&gb, 1, &empty).unwrap(),
            "组合方式进签名（同签名⇒同输出公理的补全项）"
        );

        // 节点 id 不同 → 签名不同（node_id 经 EvalCtx/SampleIndex 进入随机流）。
        let mut gc = Graph::new();
        gc.add_node(leaf(9, StructureId::ScalarField, 1.0)).unwrap();
        let mut gd = Graph::new();
        gd.add_node(leaf(10, StructureId::ScalarField, 1.0)).unwrap();
        assert_ne!(
            node_signature(&gc, 9, &empty).unwrap(),
            node_signature(&gd, 10, &empty).unwrap(),
            "node_id 进签名（同签名⇒同输出公理的补全项）"
        );

        // 结构不同：官方段 vs 用户段（crate 名隔离进签名， L4）。
        let mut ge = Graph::new();
        let mut gf = Graph::new();
        ge.add_node(leaf(1, StructureId::ScalarField, 1.0)).unwrap();
        gf.add_node(leaf(1, StructureId::custom(13, "my_fx"), 1.0)).unwrap();
        assert_ne!(
            node_signature(&ge, 1, &empty).unwrap(),
            node_signature(&gf, 1, &empty).unwrap()
        );
        // 同编号不同 crate 的用户段互异。
        let mut gg = Graph::new();
        gg.add_node(leaf(1, StructureId::custom(13, "other_fx"), 1.0)).unwrap();
        assert_ne!(
            node_signature(&gf, 1, &empty).unwrap(),
            node_signature(&gg, 1, &empty).unwrap()
        );
    }

    #[test]
    fn node_signature_matches_graph_signatures_entry() {
        let defs_table = defs(&[("wind", 3, b"w")]);
        let g = chain_graph(0.5);
        let sigs = graph_signatures(&g, &defs_table).unwrap();
        for id in [1u64, 2, 3, 4] {
            assert_eq!(
                node_signature(&g, id, &defs_table).unwrap(),
                sigs[&id],
                "单节点签名入口与全表一致（节点 {id}）"
            );
        }
        // 缺失节点 → P3013 结构化报错。
        let err = node_signature(&g, 99, &defs_table).unwrap_err();
        assert!(err.to_string().contains("P3013"));
    }

    #[test]
    fn channel_definition_hash_distinguishes_inputs() {
        let a = ChannelDef { tag: 1, def_bytes: b"abc".to_vec() };
        let same = ChannelDef { tag: 1, def_bytes: b"abc".to_vec() };
        let diff_tag = ChannelDef { tag: 2, def_bytes: b"abc".to_vec() };
        let diff_bytes = ChannelDef { tag: 1, def_bytes: b"abd".to_vec() };

        assert_eq!(channel_definition_hash(&a), channel_definition_hash(&same));
        assert_ne!(channel_definition_hash(&a), channel_definition_hash(&diff_tag));
        assert_ne!(channel_definition_hash(&a), channel_definition_hash(&diff_bytes));
    }

    // ------------------------------------------------------------------
    // 缓存键（五元组 + tile；C13 版本入键）
    // ------------------------------------------------------------------

    #[test]
    fn cache_key_varies_across_all_dimensions() {
        let sub: Hash256 = [7u8; 32];
        let base = CacheDims {
            width: 1920,
            height: 1080,
            spp_tier: 2,
            tile: (3, 5),
            engine_version: (0 << 32) | (1 << 16) | 0,
            frame: 0,
        };
        let k = cache_key(&sub, &base);
        assert_eq!(k, cache_key(&sub, &base), "同维度同键");

        let mut d;
        d = base; d.width = 1280; assert_ne!(k, cache_key(&sub, &d), "分辨率宽变化 → 键变化");
        d = base; d.height = 720; assert_ne!(k, cache_key(&sub, &d), "分辨率高变化 → 键变化");
        d = base; d.spp_tier = 1; assert_ne!(k, cache_key(&sub, &d), "spp 档变化 → 键变化");
        d = base; d.tile = (4, 5); assert_ne!(k, cache_key(&sub, &d), "tile x 变化 → 键变化");
        d = base; d.tile = (3, 6); assert_ne!(k, cache_key(&sub, &d), "tile y 变化 → 键变化");
        d = base; d.engine_version += 1; assert_ne!(k, cache_key(&sub, &d), "引擎版本变化 → 键变化（C13）");
        d = base; d.frame = 1; assert_ne!(k, cache_key(&sub, &d), "帧号变化 → 键变化（D-28）");

        let other_sub: Hash256 = [8u8; 32];
        assert_ne!(k, cache_key(&other_sub, &base), "子图签名变化 → 键变化");
    }

    // ------------------------------------------------------------------
    // 快照键咬合（D-28：SnapshotKey = (签名位, 帧)）
    // ------------------------------------------------------------------

    #[test]
    fn snapshot_signature_feeds_snapshot_key() {
        let sub: Hash256 = [1u8; 32];
        let chain_a = [[2u8; 32], [3u8; 32]];
        let chain_b = [[2u8; 32], [4u8; 32]];

        let s_empty = snapshot_signature(&sub, &[]);
        let s_a = snapshot_signature(&sub, &chain_a);
        let s_b = snapshot_signature(&sub, &chain_b);

        assert_eq!(s_empty, snapshot_signature(&sub, &[]), "同链同签名");
        assert_ne!(s_a, s_b, "通道定义签名链不同 → 快照签名不同（链语义）");
        assert_ne!(s_empty, sub, "域分隔：快照签名 ≠ 裸子图签名");

        // 咬合 node.rs 的 SnapshotStore：键类型零改动。
        let mut store = crate::node::SnapshotStore::new();
        let key: crate::node::SnapshotKey = (s_a, 42);
        store.insert(key, vec![9, 9]);
        assert_eq!(store.get(&(s_a, 42)), Some(&vec![9, 9]));
        assert!(store.get(&(s_b, 42)).is_none(), "签名位不同互不误取");
        assert!(store.get(&(s_a, 43)).is_none(), "帧位不同互不误取（seek 等价前提）");
    }

    #[test]
    fn empty_graph_has_defined_signature() {
        let g = Graph::new();
        let bytes = graph_canonical_bytes(&g);
        // 头（10B magic + 4B schema）+ 节点数 8B + 边数 8B。
        assert_eq!(bytes.len(), 10 + 4 + 8 + 8);
        let sig = graph_signature(&g, &ChannelDefs::new()).unwrap();
        assert_eq!(sig, graph_signature(&g, &ChannelDefs::new()).unwrap());
    }
}
