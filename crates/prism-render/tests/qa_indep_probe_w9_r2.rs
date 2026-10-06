//! W9 Round-2 独立复核探针（QA 严过关 Edward）。
//!
//! 目的：不依赖"绿了"，独立复核 4 个缺陷修复 + 新攻击面。
//! - P0-A: stable_intervals_excluding 新语义（取与变更包络相交）是否正确；
//!         多通道不同恒定区间 / 零恒定区间 / 全覆盖变更 的收窄结果是否精确；
//!         脏帧实测数 vs 全量。
//! - P0-B: CacheDims.frame 是否真进摘要；跨帧不误命中；同帧同子图仍命中。
//! - reuse_ratio: 帧维收窄后是否仍 ≥0.5 且相对全量提升（红利）。
//! 只读探针，不改源码。

use prism_core::node::{Composition, MergeOp, Node, ParamValue, StructureId};
use prism_core::signature::{cache_key, graph_signatures, CacheDims, ChannelDef, ChannelDefs};
use prism_core::{Graph, NodeId};

use prism_render::incremental::{
    ChannelIntervals, DirtyDetector, FrameRange, IncrementalSession, ParamChange,
};

fn leaf(id: NodeId, structure: StructureId, x: f32) -> Node {
    let mut node = Node::new(id, structure, Composition::Feed);
    node.params.set("x", ParamValue::F32(x));
    node
}

fn chain_graph(x1: f32) -> Graph {
    let mut g = Graph::new();
    g.add_node(leaf(1, StructureId::ScalarField, x1)).unwrap();
    g.add_node(leaf(2, StructureId::Convolution, 2.0)).unwrap();
    g.add_node(Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus))).unwrap();
    g.add_node(Node::new(4, StructureId::Convolution, Composition::Chain)).unwrap();
    g.connect(1, 3, 0).unwrap();
    g.connect(2, 3, 1).unwrap();
    g.connect(3, 4, 0).unwrap();
    g
}

fn cdef(tag: u16, bytes: &[u8]) -> ChannelDef {
    ChannelDef { tag, def_bytes: bytes.to_vec() }
}


/// 双叶可变图：同时可改叶 1、叶 2 参数（用于多通道脏传播探针）。
fn chain_graph2(x1: f32, x2: f32) -> Graph {
    let mut g = Graph::new();
    g.add_node(leaf(1, StructureId::ScalarField, x1)).unwrap();
    g.add_node(leaf(2, StructureId::Convolution, x2)).unwrap();
    g.add_node(Node::new(3, StructureId::Diffusion, Composition::Merge(MergeOp::Plus))).unwrap();
    g.add_node(Node::new(4, StructureId::Convolution, Composition::Chain)).unwrap();
    g.connect(1, 3, 0).unwrap();
    g.connect(2, 3, 1).unwrap();
    g.connect(3, 4, 0).unwrap();
    g
}

fn defs_with(channel: &str) -> ChannelDefs {
    let mut d = ChannelDefs::new();
    d.insert(channel.to_string(), cdef(2, &[1]));
    d
}

/// 把脏集展开为逐帧布尔向量（帧 0..total），验证"不多算/不少算"。
fn dirty_frame_vector(ds: &prism_render::incremental::DirtySet, total: u64, node: NodeId) -> Vec<bool> {
    let ranges = ds.ranges_of(node);
    (0..total).map(|f| ranges.iter().any(|r| r.contains(f))).collect()
}

// ===========================================================================
// P0-A 语义复核：stable_intervals_excluding 取"与变更包络相交"
// ===========================================================================

/// 攻击 1：变更包络与恒定区间**部分相交**——只有交集内的帧可扣除。
/// total=20，changed=[0,19]（全帧保守）。
/// 通道恒定区间 = {[0,4],[10,14]}。
/// 正确脏帧 = 全帧减去两段恒定 = [5,9] ∪ [15,19] = 5+5 = 10 帧。
#[test]
fn p0a_partial_intersect_narrows_exactly() {
    let defs = defs_with("wind");
    let mut iv = ChannelIntervals::new();
    iv.push("wind", FrameRange { start: 0, end: 4 });
    iv.push("wind", FrameRange { start: 10, end: 14 });

    let mut det = DirtyDetector::new();
    // 首帧建立基线。
    det.detect(&chain_graph(1.0), &defs, &[], 20, &iv).unwrap();
    // 第二帧：叶 1 参数实际变化（签名变 → 脏），变更保守登记全帧。
    let changes =
        [ParamChange { node: 1, channel: Some("wind".into()), changed_frames: FrameRange::full(20) }];
    let ds = det.detect(&chain_graph(1.5), &defs, &changes, 20, &iv).unwrap();

    // 脏节点 1 的脏帧应恰为 [5,9] ∪ [15,19]。
    assert_eq!(dirty_frame_vector(&ds, 20, 1), {
        let mut v = vec![false; 20];
        for f in 5..=9 { v[f] = true; }
        for f in 15..=19 { v[f] = true; }
        v
    }, "脏帧必须精确 = 全帧减恒定交集；多算=白算，少算=漏渲染");
    assert_eq!(ds.dirty_frame_count(), 10, "脏帧实测应为 10（全量 20）");
}

/// 攻击 2：零恒定区间（空表）——必须保守全帧，不能漏算。
#[test]
fn p0a_no_intervals_is_conservative_full() {
    let defs = defs_with("wind");
    let iv = ChannelIntervals::new(); // 空表
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(1.0), &defs, &[], 20, &iv).unwrap();
    let changes =
        [ParamChange { node: 1, channel: Some("wind".into()), changed_frames: FrameRange::full(20) }];
    let ds = det.detect(&chain_graph(1.5), &defs, &changes, 20, &iv).unwrap();
    assert!(dirty_frame_vector(&ds, 20, 1).iter().all(|b| *b), "无区间信息必须全帧脏（保守，绝不漏渲染）");
    assert_eq!(ds.dirty_frame_count(), 20);
}

/// 攻击 3：全覆盖变更——恒定区间完全覆盖变更包络 → 脏帧应为 0。
/// changed=[5,9]，恒定={[0,9]}（含 [5,9]）→ 变更帧全在恒定带内 → 可 100% 复用。
#[test]
fn p0a_full_cover_yields_zero_dirty() {
    let defs = defs_with("wind");
    let mut iv = ChannelIntervals::new();
    iv.push("wind", FrameRange { start: 0, end: 9 });
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(1.0), &defs, &[], 10, &iv).unwrap();
    let changes =
        [ParamChange { node: 1, channel: Some("wind".into()), changed_frames: FrameRange { start: 5, end: 9 } }];
    let ds = det.detect(&chain_graph(1.5), &defs, &changes, 10, &iv).unwrap();
    assert_eq!(ds.dirty_frame_count(), 0, "变更全落在恒定带内 → 零脏帧（全复用）");
}

/// 攻击 4：恒定区间**完全在变更包络之外**（不相交）→ 不得扣除（否则漏算）。
/// changed=[0,4]；恒定={[10,14]}（不相交）→ 脏帧 = [0,4] = 5 帧（不因无关恒定带而变）。
#[test]
fn p0a_disjoint_stable_not_subtracted() {
    let defs = defs_with("wind");
    let mut iv = ChannelIntervals::new();
    iv.push("wind", FrameRange { start: 10, end: 14 });
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(1.0), &defs, &[], 20, &iv).unwrap();
    let changes = [ParamChange {
        node: 1,
        channel: Some("wind".into()),
        changed_frames: FrameRange { start: 0, end: 4 },
    }];
    let ds = det.detect(&chain_graph(1.5), &defs, &changes, 20, &iv).unwrap();
    assert_eq!(dirty_frame_vector(&ds, 20, 1), {
        let mut v = vec![false; 20];
        for f in 0..=4 { v[f] = true; }
        v
    }, "不相交的恒定带不得影响脏帧（否则漏渲染）");
}

/// 攻击 5：多通道不同恒定区间——各自独立扣除。
/// 节点 1 绑 wind（恒定 {[0,3]}），节点 2 绑 rain（恒定 {[15,19]}），均为全帧变更。
/// 节点 1 脏 = [4,19]（16 帧）；节点 2 脏 = [0,14]（15 帧）；并集 = 全 20 帧。
#[test]
fn p0a_multi_channel_distinct_intervals() {
    let mut defs = ChannelDefs::new();
    defs.insert("wind".into(), cdef(2, &[1]));
    defs.insert("rain".into(), cdef(2, &[2]));
    let mut iv = ChannelIntervals::new();
    iv.push("wind", FrameRange { start: 0, end: 3 });
    iv.push("rain", FrameRange { start: 15, end: 19 });

    let mut det = DirtyDetector::new();
    det.detect(&chain_graph2(1.0, 2.0), &defs, &[], 20, &iv).unwrap();
    let changes = [
        ParamChange { node: 1, channel: Some("wind".into()), changed_frames: FrameRange::full(20) },
        ParamChange { node: 2, channel: Some("rain".into()), changed_frames: FrameRange::full(20) },
    ];
    let ds = det.detect(&chain_graph2(1.5, 2.5), &defs, &changes, 20, &iv).unwrap();
    assert_eq!(dirty_frame_vector(&ds, 20, 1), {
        let mut v = vec![true; 20];
        for f in 0..=3 { v[f] = false; }
        v
    }, "wind 通道：恒定 [0,3] 扣除后脏 [4,19]");
    assert_eq!(dirty_frame_vector(&ds, 20, 2), {
        let mut v = vec![true; 20];
        for f in 15..=19 { v[f] = false; }
        v
    }, "rain 通道：恒定 [15,19] 扣除后脏 [0,14]");
    // 两通道并集 = 全帧（互补）。
    assert_eq!(ds.dirty_frame_count(), 20, "两通道并集覆盖全帧");
}

// ===========================================================================
// P0-B：CacheDims.frame 进摘要 + 跨帧行为
// ===========================================================================

/// 跨帧不同键（不误命中）。
#[test]
fn p0b_frame_participates_in_summary() {
    let sub = graph_signatures(&chain_graph(1.0), &defs_with("wind")).unwrap();
    let root = sub[&4];
    let base = CacheDims { width: 64, height: 64, spp_tier: 2, tile: (0, 0), engine_version: 1, frame: 0 };
    let k0 = cache_key(&root, &base);
    let mut f1 = base;
    f1.frame = 1;
    let k1 = cache_key(&root, &f1);
    assert_ne!(k0, k1, "帧号必须进摘要：跨帧不同键（否则误命中）");
    // 同帧同子图同维度 → 必须命中（别改过头导致永不命中）。
    assert_eq!(k0, cache_key(&root, &base), "同帧同维度必须稳定命中");
}

/// 会话级：同帧同子图命中、跨帧不命中（缓存容量充足）。
#[test]
fn p0b_session_same_frame_hits_cross_frame_misses() {
    let mut s = IncrementalSession::new(64);
    let sub = graph_signatures(&chain_graph(1.0), &defs_with("wind")).unwrap();
    let root = sub[&4];
    let d = CacheDims { width: 32, height: 32, spp_tier: 2, tile: (0, 0), engine_version: 1, frame: 0 };

    s.set_frame(3);
    let (_, v1) = s.process_unit(&root, &d, || vec![1, 2, 3]);
    assert!(matches!(v1, prism_render::incremental::ReuseVerdict::Recompute));
    // 同帧再处理 → 命中。
    let (_, v2) = s.process_unit(&root, &d, || vec![9, 9, 9]);
    assert!(matches!(v2, prism_render::incremental::ReuseVerdict::Reuse), "同帧同子图必须命中");
    // 推进到帧 4 → 不得命中帧 3 的字节（否则跨帧误命中）。
    s.advance_frame();
    let (bytes3, v3) = s.process_unit(&root, &d, || vec![4, 4, 4]);
    assert!(matches!(v3, prism_render::incremental::ReuseVerdict::Recompute), "跨帧必须不命中");
    assert_eq!(bytes3, vec![4, 4, 4], "跨帧命中会取回上帧字节 = 铁律击穿");
}

// ===========================================================================
// reuse_ratio：帧维收窄红利
// ===========================================================================

/// 帧维收窄后复用率应仍 ≥0.5，且相对全量提升。
/// 场景：total=20，恒定 [0,14]（15 帧），全帧保守变更 → 脏 5 帧 → 复用 15/20 = 0.75。
#[test]
fn reuse_ratio_bonus_after_narrowing() {
    let defs = defs_with("wind");
    let mut iv = ChannelIntervals::new();
    iv.push("wind", FrameRange { start: 0, end: 14 });
    let mut s = IncrementalSession::new(64);
    // 首帧（冷启动）：建基线，首帧全脏属预期，不计入稳态红利。
    s.observe_frame(&chain_graph(1.0), &defs, &[], 20, 1, &iv).unwrap();
    let after_cold = s.stats();
    // 第二帧：参数实际变化 + 全帧保守变更，恒定 [0,14]（15 帧）→ 脏仅 5 帧。
    let changes =
        [ParamChange { node: 1, channel: Some("wind".into()), changed_frames: FrameRange::full(20) }];
    let ds = s.observe_frame(&chain_graph(2.0), &defs, &changes, 20, 1, &iv).unwrap();
    assert_eq!(ds.dirty_frame_count(), 5, "帧维收窄后脏帧应为 5（全量 20）");

    // 第二帧（稳态单帧）复用率：本帧新增 reused = 15 / 本帧 total = 20 → 0.75。
    let after_warm = s.stats();
    let frame_reused = after_warm.reused() - after_cold.reused();
    let frame_total = after_warm.total() - after_cold.total();
    let frame_ratio = frame_reused as f64 / frame_total as f64;
    assert!(frame_ratio >= 0.5, "帧维收窄后单帧 reuse_ratio 须 ≥0.5，实测 {frame_ratio}");

    // 红利对照：若帧维未收窄（全量），本帧 reused 应为 0；
    // 实测本帧 reused = 15（脏 5），证明收窄真正兑现。
    assert_eq!(frame_reused, 15, "收窄红利：本帧应复用 15 帧（否则帧维复用未落地）");
}
