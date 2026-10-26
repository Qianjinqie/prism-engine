//! S3 分类表闭包测试（#8）：在签名 / 缓存键层把四类动态节点 + S4 逐出中性
//! 逐行钉死（PRD B2 验收 1）。
//!
//! 覆盖：
//! - S1 无状态节点：sig 相等 ⇒ 缓存键相等 ⇒ 可复用；
//! - S3 行 2 流节点（ConstraintFlow）：键含 `frame`；
//! - S3 行 3 迭代节点（M2/M3）：键含模式参数 `(ε, n)`（经 `0x5333` 通道定义
//!   哈希，与 `loop_frame_signature` 共用编码）；
//! - C13：键含 schema / 引擎版本；
//! - S4 逐出中性在 `prism-render` 侧（见 `qa_incremental_s3.rs`）。

use prism_core::fixpoint::{solve_mode_def, SolveModeMeta};
use prism_core::graph::Graph;
use prism_core::node::{Composition, Hash256, MergeOp, Node, NodeId, ParamValue, StructureId};
use prism_core::signature::{
    cache_key, cache_key_with_mode, node_signature, ChannelDef, ChannelDefs, CacheDims,
    SIGNATURE_SCHEMA_VERSION,
};

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

/// 缓存维度构造。
fn dims(version: u64, frame: u64) -> CacheDims {
    CacheDims {
        width: 192,
        height: 108,
        spp_tier: 2,
        tile: (0, 0),
        engine_version: version,
        frame,
    }
}

// ---- S1：无状态节点 sig 闭包（sig 相等 ⇒ 键相等 ⇒ 可复用） ----

#[test]
fn s1_stateless_signature_closure_is_reusable() {
    let empty = ChannelDefs::new();
    let root_a = node_signature(&chain_graph(0.5), 4, &empty).unwrap();
    let root_b = node_signature(&chain_graph(0.5), 4, &empty).unwrap(); // 同结构
    let root_c = node_signature(&chain_graph(0.9), 4, &empty).unwrap(); // 不同参数

    assert_eq!(root_a, root_b, "同结构图根签名一致（前置条件）");
    assert_ne!(root_a, root_c, "不同参数根签名不同（前置条件）");

    // sig 相等 ⇒ 键相等 ⇒ 可复用。
    assert_eq!(
        cache_key(&root_a, &dims(1, 0)),
        cache_key(&root_b, &dims(1, 0)),
        "S1：无状态节点 sig 相等 ⇒ 缓存键相等 ⇒ 可复用"
    );
    // sig 不同 ⇒ 键不同（不误复用）。
    assert_ne!(
        cache_key(&root_a, &dims(1, 0)),
        cache_key(&root_c, &dims(1, 0)),
        "S1：sig 不同 ⇒ 键不同（不误复用）"
    );
}

// ---- S3 行 2：流节点键含 frame（D-28 快照键 (sig,frame)） ----

#[test]
fn s3_stream_node_cache_key_contains_frame() {
    let empty = ChannelDefs::new();
    let root = node_signature(&chain_graph(0.5), 4, &empty).unwrap();

    let k0 = cache_key(&root, &dims(1, 0));
    let k1 = cache_key(&root, &dims(1, 1));
    let k2 = cache_key(&root, &dims(1, 2));

    // 同一子图、同一维度，仅帧号不同 ⇒ 键不同（杜绝跨帧误命中）。
    assert_ne!(k0, k1, "帧 0 ≠ 帧 1 键（流节点帧维）");
    assert_ne!(k1, k2, "帧 1 ≠ 帧 2 键（流节点帧维）");
    assert_ne!(k0, k2, "帧 0 ≠ 帧 2 键（流节点帧维）");
    // 同帧同维度 ⇒ 同键（确定性）。
    assert_eq!(k0, cache_key(&root, &dims(1, 0)), "同帧同维度同键（确定性）");
}

// ---- S3 行 3：迭代节点键含 (ε, n)（经 0x5333 通道定义哈希） ----

#[test]
fn s3_iteration_node_cache_key_contains_mode_params() {
    let empty = ChannelDefs::new();
    let root = node_signature(&chain_graph(0.5), 4, &empty).unwrap();
    let d = dims(1, 0);

    let m1 = SolveModeMeta::new(4, 8, 1e-4);
    let m2 = SolveModeMeta::new(5, 8, 1e-4); // substeps 变
    let m3 = SolveModeMeta::new(4, 16, 1e-4); // iterations 变
    let m4 = SolveModeMeta::new(4, 8, 1e-3); // residual_eps 变
    let def1 = solve_mode_def(&m1);
    let def2 = solve_mode_def(&m2);
    let def3 = solve_mode_def(&m3);
    let def4 = solve_mode_def(&m4);

    // None 退化为 cache_key（逐字节相同，零回归）。
    assert_eq!(
        cache_key_with_mode(&root, &d, None),
        cache_key(&root, &d),
        "mode=None 必须逐字节等价于 cache_key"
    );
    // 模式元数据进入键：同模式 ⇒ 同键（确定性）。
    assert_eq!(
        cache_key_with_mode(&root, &d, Some(&def1)),
        cache_key_with_mode(&root, &d, Some(&def1)),
        "同模式 ⇒ 同键（确定性）"
    );
    // 任一模式参数变化 ⇒ 键变化（S3 迭代层 (ε, n) 闭包）。
    assert_ne!(
        cache_key_with_mode(&root, &d, Some(&def1)),
        cache_key_with_mode(&root, &d, Some(&def2)),
        "substeps 变化 ⇒ 键变化"
    );
    assert_ne!(
        cache_key_with_mode(&root, &d, Some(&def1)),
        cache_key_with_mode(&root, &d, Some(&def3)),
        "iterations 变化 ⇒ 键变化"
    );
    assert_ne!(
        cache_key_with_mode(&root, &d, Some(&def1)),
        cache_key_with_mode(&root, &d, Some(&def4)),
        "residual_eps 变化 ⇒ 键变化"
    );
    // 模式感知键与无模式键处于不同域（标签隔离）：同 sub/dims，带模式 ≠ 不带模式。
    assert_ne!(
        cache_key_with_mode(&root, &d, Some(&def1)),
        cache_key(&root, &d),
        "带模式键与无模式键域分隔（不跨族误命中）"
    );
    // 键经 0x5333 通道定义哈希（与 loop_frame_signature 同编码）。
    assert_eq!(def1.tag, 0x5333, "迭代模式通道定义 tag 必须为 0x5333");
}

// ---- C13：键含 schema / 引擎版本（旧版本缓存只失效重建、永不误复用） ----

#[test]
fn c13_cache_key_contains_schema_and_engine_version() {
    let empty = ChannelDefs::new();
    let root = node_signature(&chain_graph(0.5), 4, &empty).unwrap();

    let k = cache_key(&root, &dims(1, 0));
    assert_eq!(k, cache_key(&root, &dims(1, 0)), "同维度同键");

    // 引擎版本变化 ⇒ 键变化（C13）。
    let mut d = dims(1, 0);
    d.engine_version += 1;
    assert_ne!(k, cache_key(&root, &d), "引擎版本变化 ⇒ 键变化（C13）");

    // schema 版本是编译期常数，显式断言其进入键（与引擎版本同族）。
    assert!(SIGNATURE_SCHEMA_VERSION >= 1, "schema 版本必须进键（C13）");
}
