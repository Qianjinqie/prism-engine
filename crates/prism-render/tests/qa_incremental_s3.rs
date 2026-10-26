//! S3 分类表复用判定测试（#8）：在 `ReuseVerdict` / `DirtyDetector` 层把四类动态
//! 节点 + S4 逐出中性逐行钉死（PRD B2 验收 1–2）。
//!
//! 覆盖：
//! - S3 行 2 流节点（ConstraintFlow）：帧维分离 ⇒ 帧 0 的字节绝不回取帧 1；
//! - S3 行 3 迭代节点（M2/M3）：模式参数 `(ε, n)` 进键 ⇒ 跨模式绝不误复用；
//! - 音频反应（CH4）：烘焙表区间内复用、区间外拒绝；
//! - S4 逐出中性：LRU 逐出只影响重算量、不影响结果（逐出后重算逐位一致）。

use prism_core::fixpoint::{solve_mode_def, SolveModeMeta};
use prism_core::graph::Graph;
use prism_core::node::{
    Composition, Hash256, MergeOp, Node, NodeBinding, NodeId, ParamValue, StructureId, ChannelRef,
};
use prism_core::signature::{cache_key, node_signature, ChannelDef, ChannelDefs, CacheDims};
use prism_render::incremental::{
    decide, decide_with_mode, ChannelIntervals, DirtyDetector, FrameRange, IncrementalSession,
    ParamChange, ReuseCache, ReuseVerdict,
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

fn frame_dims(f: u64) -> CacheDims {
    CacheDims {
        frame: f,
        ..dims(1)
    }
}

// ---- S3 行 2：流节点（ConstraintFlow）帧维分离（D-28） ----

#[test]
fn s3_stream_node_reuse_verdict_frame_separated() {
    let empty = ChannelDefs::new();
    let root = node_signature(&chain_graph(0.5), 4, &empty).unwrap();
    let mut store = ReuseCache::new(0);

    let d0 = frame_dims(0);
    let d1 = frame_dims(1);

    // 两帧是不同键 ⇒ 初始都未命中（绝不跨帧误命中）。
    assert_eq!(decide(&store, &root, &d0).verdict, ReuseVerdict::Recompute);
    assert_eq!(decide(&store, &root, &d1).verdict, ReuseVerdict::Recompute);

    // 回填帧 0 ⇒ 帧 0 命中、帧 1 仍未命中（D-28 帧维权威）。
    store.insert(cache_key(&root, &d0), vec![9]);
    assert_eq!(
        decide(&store, &root, &d0).verdict,
        ReuseVerdict::Reuse,
        "帧 0 回填后命中"
    );
    assert_eq!(
        decide(&store, &root, &d1).verdict,
        ReuseVerdict::Recompute,
        "帧 1 绝不取回帧 0 的字节（跨帧误命中）"
    );
}

// ---- S3 行 3：迭代节点（M2/M3）模式参数进键（0x5333） ----

#[test]
fn s3_iteration_node_reuse_verdict_mode_separated() {
    let empty = ChannelDefs::new();
    let root = node_signature(&chain_graph(0.5), 4, &empty).unwrap();
    let d = frame_dims(0);

    let m1 = SolveModeMeta::new(4, 8, 1e-4);
    let m2 = SolveModeMeta::new(4, 16, 1e-4); // 不同 iterations
    let def1 = solve_mode_def(&m1);
    let def2 = solve_mode_def(&m2);

    let mut store = ReuseCache::new(0);
    let v1 = decide_with_mode(&store, &root, &d, Some(&def1));
    let v2 = decide_with_mode(&store, &root, &d, Some(&def2));

    // 模式不同 ⇒ 键不同 ⇒ 初始都未命中（绝不跨模式误复用）。
    assert_ne!(v1.key, v2.key, "模式不同 ⇒ 键不同");
    assert_eq!(v1.verdict, ReuseVerdict::Recompute);
    assert_eq!(v2.verdict, ReuseVerdict::Recompute);

    // 回填 m1 ⇒ m1 命中、m2 仍未命中（S3 迭代层 (ε, n) 闭包）。
    store.insert(v1.key, vec![7]);
    assert_eq!(
        decide_with_mode(&store, &root, &d, Some(&def1)).verdict,
        ReuseVerdict::Reuse,
        "模式 m1 回填后命中"
    );
    assert_eq!(
        decide_with_mode(&store, &root, &d, Some(&def2)).verdict,
        ReuseVerdict::Recompute,
        "跨模式绝不误复用（不同 (ε, n) 是不同键）"
    );

    // None 退化为 decide（同键族）。
    assert_eq!(
        decide_with_mode(&store, &root, &d, None).key,
        decide(&store, &root, &d).key,
        "mode=None 与 decide 同键"
    );
}

// ---- 音频反应（CH4）：烘焙表区间内复用、区间外拒绝 ----

/// 单节点图，节点 1 绑定音频反应通道 "audio"。
fn audio_graph(x: f32) -> Graph {
    let mut g = Graph::new();
    let mut n1 = leaf(1, StructureId::ScalarField, x);
    n1.bindings.push(NodeBinding {
        param: "x".into(),
        channel: ChannelRef { name: "audio".into() },
    });
    g.add_node(n1).unwrap();
    g
}

/// 音频通道定义表（tag = 4 标记 CH4 音频反应；定义字节仅占位，签名只消费相等性）。
fn audio_defs() -> ChannelDefs {
    let mut m = ChannelDefs::new();
    m.insert(
        "audio".to_string(),
        ChannelDef {
            tag: 4,
            def_bytes: b"baked-segment".to_vec(),
        },
    );
    m
}

#[test]
fn s3_audio_reactive_interval_reuse_and_reject() {
    let defs = audio_defs();
    let total = 20u64;
    let mut det = DirtyDetector::new();
    // 基线（同图，无变更）。
    det.detect(&audio_graph(0.5), &defs, &[], total, &ChannelIntervals::new())
        .unwrap();

    // 音频通道 "audio" 烘焙表恒定带 [0,3]：带内参数恒定 ⇒ 可复用。
    let mut intervals = ChannelIntervals::new();
    intervals.push("audio", FrameRange { start: 0, end: 3 });

    // 第二帧：节点 1 参数变更（x: 0.5 → 0.9），绑定 audio 通道，变更包络 = 全帧。
    let change = ParamChange {
        node: 1,
        channel: Some("audio".to_string()),
        changed_frames: FrameRange::full(total),
    };
    let dirty = det
        .detect(&audio_graph(0.9), &defs, &[change], total, &intervals)
        .unwrap();

    // 区间外（帧 4..19，共 16 帧）脏（拒绝/重算）；区间内（0..3）不脏（复用）。
    assert_eq!(
        dirty.dirty_frame_count(),
        16,
        "音频恒定带 [0,3] 被扣除，脏帧 = [4,19] 共 16 帧"
    );
    for f in 0..4u64 {
        assert!(
            !dirty.ranges_of(1).iter().any(|r| r.contains(f)),
            "帧 {f} 在恒定带内 ⇒ 复用（不脏）"
        );
    }
    for f in 4..total {
        assert!(
            dirty.ranges_of(1).iter().any(|r| r.contains(f)),
            "帧 {f} 在恒定带外 ⇒ 重算（脏）"
        );
    }
}

// ---- S4：LRU 逐出中性（逐出只影响重算量、不影响结果） ----

#[test]
fn s4_lru_eviction_recompute_is_bit_identical_to_pre_eviction() {
    let mut session = IncrementalSession::new(2);
    let sig: Hash256 = [7u8; 32];
    let artifact = |f: u64| -> Vec<u8> { vec![f as u8; 4] };

    let (b0, v0) = session.process_unit(&sig, &frame_dims(0), || artifact(0));
    assert_eq!(v0, ReuseVerdict::Recompute, "帧 0 首轮未命中");
    let (b1, _) = session.process_unit(&sig, &frame_dims(1), || artifact(1));
    let (b2, _) = session.process_unit(&sig, &frame_dims(2), || artifact(2));

    // 容量 2 ⇒ 最久未用的帧 0 键被逐出。
    assert_eq!(session.cache().len(), 2, "容量 2：仅留最近 2 帧键");
    assert!(
        !session.cache().contains(&cache_key(&sig, &frame_dims(0))),
        "帧 0 键被逐出"
    );

    // 保留键（帧 1）仍命中，回取逐位不变。
    let (b1b, v1) = session.process_unit(&sig, &frame_dims(1), || panic!("保留键不得触发重算"));
    assert_eq!(v1, ReuseVerdict::Reuse);
    assert_eq!(b1b, b1, "逐出不改变保留键的回取值（S4 中性）");

    // 被逐出键（帧 0）重算：与逐出前逐位一致（纯函数 ⇒ 逐出无损）。
    let (b0b, v0b) = session.process_unit(&sig, &frame_dims(0), || artifact(0));
    assert_eq!(v0b, ReuseVerdict::Recompute, "被逐出键必未命中");
    assert_eq!(b0b, b0, "逐出后重算与逐出前逐位一致（S4 中性）");
    assert_eq!(b2, artifact(2));
}
