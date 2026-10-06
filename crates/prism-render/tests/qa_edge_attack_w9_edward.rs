//! W9-C3 攻击性测试（QA 严过关 Edward）。
//!
//! 焦点：两条线索验证 + 全面攻击三面。
//! 每个测试注明"失败代表什么缺陷"。

use prism_core::node::{Composition, Hash256, MergeOp, Node, ParamValue, StructureId};
use prism_core::signature::{cache_key, graph_signatures, CacheDims, ChannelDef, ChannelDefs};
use prism_core::{Graph, NodeId};
use prism_io::fbuf::Fbuf16F;

use prism_render::incremental::{
    bytes_to_fbuf, decide, fbuf_to_bytes, reuse_stats_from_dirty, ChannelIntervals, DirtyDetector,
    FrameRange, IncrementalSession, ParamChange, ReuseCache, ReuseStats, ReuseVerdict,
};

// ---------------------------------------------------------------------------
// 图夹具
// ---------------------------------------------------------------------------

fn leaf(id: NodeId, structure: StructureId, x: f32) -> Node {
    let mut node = Node::new(id, structure, Composition::Feed);
    node.params.set("x", ParamValue::F32(x));
    node
}

/// 链形图：1(叶) ┐
///              3 = Merge(Plus) ── 4(Chain 根)
///            2(叶) ┘
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

fn dims(version: u64, w: u32, h: u32, tile: (u32, u32)) -> CacheDims {
    CacheDims { width: w, height: h, spp_tier: 2, tile, engine_version: version, frame: 0 }
}

// ===========================================================================
// 线索 1（P0）：DirtyDetector::detect 是否使用 channel_intervals？
// ===========================================================================

/// 缺陷面：帧维复用未落地——`detect` 完全忽略 `channel_intervals`（源码 362
/// 行 `let _ = channel_intervals;`）。
///
/// 判别设计：变更**保守登记为全帧**（`changed_frames = full`，如通道为 Expr
/// 无变化率界时的保守起点），但该通道在 ChannelIntervals 中给出**大部分帧恒定**
/// 的区间。正确的帧维复用应把脏区间收窄到"实际变化帧"之外，恒定区间内的帧免
/// 于重算（reuse_ratio 上升）。若 `detect` 忽略 channel_intervals，脏区间将保持
/// 全帧。
///
/// 期望：dirty_frame_count < total（恒定帧被排除）。
/// 失败 = 帧维复用未落地（线索 1 坐实）。
#[test]
fn detect_should_use_channel_intervals_for_frame_reuse() {
    let empty = ChannelDefs::new();
    // 通道 "wind"：帧 [0,4] 与 [10,19] 恒定，仅帧 5..9 变化带。
    let mut intervals = ChannelIntervals::new();
    intervals.push("wind", FrameRange { start: 0, end: 4 });
    intervals.push("wind", FrameRange { start: 10, end: 19 });

    let total = 20u64;
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();

    // 保守：变更登记为"全帧"（通道驱动但未知精确变化帧）。
    let change = ParamChange {
        node: 1,
        channel: Some("wind".to_string()),
        changed_frames: FrameRange::full(total),
    };
    let dirty = det.detect(&chain_graph(0.9), &empty, &[change], total, &intervals).unwrap();

    // 帧维复用的判据：恒定区间 [0,4] 与 [10,19]（共 15 帧）应免于重算。
    // 预期脏帧数 = 变化带 5..9 = 5 帧；忽略 channel_intervals 则 = 20 帧。
    let frames = dirty.dirty_frame_count();
    assert!(
        frames < total,
        "channel_intervals 应把脏帧收窄（恒定帧免重算），实际脏帧 {frames}/20（帧维复用未落地，detect 忽略 channel_intervals）"
    );
    assert_eq!(frames, 5, "恒定帧 [0,4]+[10,19] 应免重算，脏帧应为 5，实际 {frames}");
}

/// 缺陷面：变更绑定了通道，但该通道**无任何恒定区间信息**（空表）→ 保守全帧。
/// 这是正确退化路径，本测试锁定保守语义不被误收窄。
#[test]
fn detect_conservative_full_frame_when_no_interval_info() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new(); // 空表
    let total = 20u64;
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
    let change = ParamChange {
        node: 1,
        channel: Some("wind".to_string()),
        changed_frames: FrameRange::full(total),
    };
    let dirty = det.detect(&chain_graph(0.9), &empty, &[change], total, &intervals).unwrap();
    let d1 = dirty.range_of(1).expect("节点 1 应脏");
    assert_eq!((d1.start, d1.end), (0, total - 1), "无区间信息 → 保守全帧");
}

// ===========================================================================
// 线索 2（P0）：cache_key 是否含帧字段？（D-28 快照键 = (子图签名, 帧)）
// ===========================================================================

/// 缺陷面：`cache_key` 缺 frame 字段。D-28 明确快照键 = (子图签名, 帧)；
/// 增量复用的"帧间复用"必须能区分不同帧。
///
/// 判别设计：`cache_key(subgraph, dims)` 的输入**只有** (子图签名, 维度)，
/// `CacheDims` 无任何帧字段。因此无论"当前帧"是多少，键恒等——无法按帧区分。
/// 本测试以行为断言锁定该接口缺口：跨帧用同一键判定 → 得到同一缓存命中
/// （即跨帧误复用），这是 D-28 的反例。
///
/// 失败 = cache_key 缺帧字段（线索 2 坐实，D-28 违反）。
#[test]
fn cache_key_must_differ_across_frames() {
    let defs = ChannelDefs::new();
    let sigs = graph_signatures(&chain_graph(0.5), &defs).unwrap();
    let root_sig = sigs[&4];
    let d = dims(1, 192, 108, (0, 0));

    // 用会话演示：同子图同维度，两"帧"产物不同 → 第二帧被误判命中。
    let mut session = IncrementalSession::new(8);
    session.set_frame(2);
    let (frame2, v2) = session.process_unit(&root_sig, &d, || vec![0x22u8; 6]);
    assert_eq!(v2, ReuseVerdict::Recompute, "帧 2 首次重算");
    // 帧 3 内容不同（产物本应不同），但键不含帧 → 命中帧 2 缓存。
    session.set_frame(3);
    let (frame3, v3) = session.process_unit(&root_sig, &d, || vec![0x33u8; 6]);

    // 期望（D-28）：不同帧应各得其产物 → frame3 应为 0x33。
    // 若键缺帧，frame3 = frame2 = 0x22（跨帧误复用）。
    assert_ne!(
        frame3, frame2,
        "不同帧必须产出不同字节（键应含帧字段）；实际两帧同字节 → cache_key 缺帧（D-28 违反）"
    );
    assert_eq!(frame3, vec![0x33u8; 6], "帧 3 应产出自身字节");
    let _ = v3;
}

/// 缺陷面：真实用例——IncrementalSession 在帧 2 与帧 3 用同一子图签名 but
/// 不同输入产物时，会返回**同一键**的缓存字节（帧 3 拿到帧 2 的产物）。
/// 这是"缓存键缺帧"的端到端可复现后果。
#[test]
fn session_reuses_across_frames_due_to_missing_frame_in_key() {
    let mut session = IncrementalSession::new(8);
    let sig: Hash256 = [7u8; 32];
    let d = dims(1, 64, 64, (0, 0));

    // 帧 2：产物字节 = [2]。
    session.set_frame(2);
    let (a, va) = session.process_unit(&sig, &d, || vec![2u8; 4]);
    assert_eq!(va, ReuseVerdict::Recompute);
    assert_eq!(a, vec![2u8; 4]);

    // 帧 3：内容不同（帧号变化 → 产物应含帧号 3），但同子图签名同维度。
    // 实现会命中帧 2 的缓存（键相同），返回 [2;4] 而非 [3;4] → 错帧。
    session.set_frame(3);
    let (b, vb) = session.process_unit(&sig, &d, || vec![3u8; 4]);
    assert_eq!(
        b,
        vec![3u8; 4],
        "帧 3 应产出帧 3 的字节；实际 {:?}（用根因：cache_key 缺帧字段，跨帧误命中）",
        b
    );
    let _ = vb;
}

// ===========================================================================
// 确定性命门：复用产物 == 全量重算（逐字节）
// ===========================================================================

/// 命门：命中缓存复用 vs 全量重算，对同一 (帧,子图) 必须逐字节一致。
/// 失败 = 复用改变输出（确定性铁律破裂）。
#[test]
fn reuse_equals_full_recompute_over_frames() {
    let defs = ChannelDefs::new();
    let sigs = graph_signatures(&chain_graph(0.5), &defs).unwrap();
    let root_sig = sigs[&4];
    let d = dims(3, 128, 72, (1, 1));

    let mut session = IncrementalSession::new(16);
    // 首帧重算。
    let (recomputed, v1) = session.process_unit(&root_sig, &d, || vec![9u8; 32]);
    assert_eq!(v1, ReuseVerdict::Recompute);
    // 复命命中 → 不得调用 recompute。
    let (reused, v2) = session.process_unit(&root_sig, &d, || panic!("命中不得重算"));
    assert_eq!(v2, ReuseVerdict::Reuse);
    assert_eq!(recomputed, reused, "复用产物必须与全量重算逐字节一致");
}

/// 命门：LRU 逐出后，对仍存 / 被逐出的键，输出都必须与首次重算逐字节一致。
#[test]
fn lru_eviction_preserves_output_bytes() {
    let mut cache = ReuseCache::new(2);
    let k = |n: u8| -> Hash256 { [n; 32] };
    let payload = |n: u8| -> Vec<u8> { vec![n; 8] };
    cache.insert(k(1), payload(1));
    cache.insert(k(2), payload(2));
    cache.insert(k(3), payload(3)); // 逐出 k(1)
    assert!(!cache.contains(&k(1)), "k(1) 应被逐出");
    assert_eq!(cache.touch_get(&k(3)).unwrap(), payload(3), "未被逐出键取值不变");
    // 被逐出键重算（纯函数）→ 同字节。
    let mut session = IncrementalSession::new(2);
    let sig = [1u8; 32];
    let d = dims(1, 32, 32, (0, 0));
    let (a, _) = session.process_unit(&sig, &d, || vec![42u8; 4]);
    let (b, v) = session.process_unit(&sig, &d, || vec![42u8; 4]);
    assert_eq!(v, ReuseVerdict::Reuse);
    assert_eq!(a, b, "逐出后重算不得改变输出");
}

/// 命门：fbuf 字节化往返逐位精确（缓存产物唯一出口）。
#[test]
fn fbuf_bytes_roundtrip_bit_exact() {
    let mut fb = Fbuf16F::new(5, 3);
    for y in 0..3u32 {
        for x in 0..5u32 {
            fb.set(x, y, [x as f32 * 0.13, y as f32 * 0.7, 0.25, 1.0]);
        }
    }
    let bytes = fbuf_to_bytes(&fb);
    let back = bytes_to_fbuf(&bytes).unwrap();
    assert_eq!(fb, back, "往返逐位精确");
    assert!(bytes_to_fbuf(&bytes[0..6]).is_err(), "过短结构拒绝");
}

// ===========================================================================
// 脏区间正确性 + reuse_ratio
// ===========================================================================

/// 脏区间正确性：改叶 1 → 仅 {1,3,4} 脏，兄弟 2 不脏。
#[test]
fn dirty_only_ancestor_chain() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let total = 10u64;
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
    let change = ParamChange {
        node: 1,
        channel: None,
        changed_frames: FrameRange::full(total),
    };
    let d = det.detect(&chain_graph(0.9), &empty, &[change], total, &intervals).unwrap();
    assert!(d.contains(1) && d.contains(3) && d.contains(4));
    assert!(!d.contains(2), "兄弟 2 不得脏");
    assert_eq!(d.len(), 3);
}

/// 边界：未知节点变更 → 结构化错误 P3014（不静默）。
#[test]
fn unknown_node_change_structured_error() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let mut det = DirtyDetector::new();
    let change = ParamChange { node: 99, channel: None, changed_frames: FrameRange::full(5) };
    let err = det.detect(&chain_graph(0.5), &empty, &[change], 5, &intervals).unwrap_err();
    assert!(err.to_string().contains("P3014"));
}

/// reuse_ratio ≥ 0.5：100 帧单子图仅帧 40 变 → 复用 99/100。
#[test]
fn reuse_ratio_at_least_half() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let total = 100u64;
    let mut det = DirtyDetector::new();
    det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
    let change = ParamChange {
        node: 1,
        channel: None,
        changed_frames: FrameRange::single(40),
    };
    let dirty = det.detect(&chain_graph(0.9), &empty, &[change], total, &intervals).unwrap();
    let stats = reuse_stats_from_dirty(&dirty, total, 1);
    assert!(stats.ratio() >= 0.5, "复用率 {} < 0.5", stats.ratio());
    assert!(stats.meets_phase3());
}

/// 统计不变量：reused > total 必须 panic（而非静默错误）。
#[test]
#[should_panic]
fn stats_record_rejects_overcount() {
    let mut s = ReuseStats::new();
    s.record(11, 10);
}

// ===========================================================================
// 签名碰撞 / decide 一致性
// ===========================================================================

/// decide 与 touch_get 结果一致（判定与取值同源）。
#[test]
fn decide_consistent_with_touch_get() {
    let mut cache = ReuseCache::new(0);
    let sig: Hash256 = [5u8; 32];
    let d = dims(1, 192, 108, (0, 0));
    assert_eq!(decide(&cache, &sig, &d).verdict, ReuseVerdict::Recompute);
    let key = cache_key(&sig, &d);
    cache.insert(key, vec![1, 2, 3]);
    assert_eq!(decide(&cache, &sig, &d).verdict, ReuseVerdict::Reuse);
    assert_eq!(cache.touch_get(&key), Some(vec![1, 2, 3]));
}

/// 分辨率变化 → 键变 → 未命中（维度区分；不得误复用）。
#[test]
fn dims_resolution_change_misses() {
    let sig: Hash256 = [1u8; 32];
    let k1 = cache_key(&sig, &dims(1, 192, 108, (0, 0)));
    let k2 = cache_key(&sig, &dims(1, 384, 216, (0, 0)));
    let k3 = cache_key(&sig, &dims(1, 192, 108, (1, 0)));
    assert_ne!(k1, k2, "分辨率变化应换键");
    assert_ne!(k1, k3, "tile 变化应换键");
}

/// 子图签名不同 → 键不同（子图维区分）。
#[test]
fn subgraph_signature_change_swaps_key() {
    let d = dims(1, 64, 64, (0, 0));
    let a: Hash256 = [1u8; 32];
    let b: Hash256 = [2u8; 32];
    assert_ne!(cache_key(&a, &d), cache_key(&b, &d));
}

/// 通道定义不同 → 签名不同（防止瞬时值相同误复用，§4.2）。
/// 用法：构造同一图不同 ChannelDef，验证 graph_signatures 变化。
#[test]
fn channel_def_change_invalidates_signature() {
    use prism_core::node::{ChannelRef, NodeBinding};

    let mut g = Graph::new();
    let mut n = leaf(1, StructureId::ScalarField, 0.5);
    n.bindings.push(NodeBinding {
        param: "x".to_string(),
        channel: ChannelRef { name: "wind".to_string() },
    });
    g.add_node(n).unwrap();

    let mut defs_a: ChannelDefs = ChannelDefs::new();
    defs_a.insert("wind".to_string(), ChannelDef { tag: 1, def_bytes: vec![1, 2, 3] });
    let mut defs_b: ChannelDefs = ChannelDefs::new();
    defs_b.insert("wind".to_string(), ChannelDef { tag: 1, def_bytes: vec![9, 9, 9] });

    let sa = graph_signatures(&g, &defs_a).unwrap();
    let sb = graph_signatures(&g, &defs_b).unwrap();
    assert_ne!(sa[&1], sb[&1], "通道定义变化必须换签名（瞬时值无关）");
}

/// 退化：空图签名表为空、detect 空变更 → 空脏集（无 panic）。
#[test]
fn empty_graph_and_no_changes() {
    let g = Graph::new();
    let defs = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let mut det = DirtyDetector::new();
    let d = det.detect(&g, &defs, &[], 0, &intervals).unwrap();
    assert!(d.is_empty(), "空图 + 空变更 → 空脏集");
    assert_eq!(d.dirty_frame_count(), 0);
}

/// 退化：total_frames = 0 时 FrameRange::full 的饱和语义（不 panic）。
#[test]
fn zero_total_frames_saturating() {
    let g = chain_graph(0.5);
    let defs = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let mut det = DirtyDetector::new();
    // total = 0：首帧全图脏，range = full(0) = [0, sat_sub(0)-... ]。
    let d = det.detect(&g, &defs, &[], 0, &intervals).unwrap();
    assert_eq!(d.len(), 4, "首帧全图脏");
    let r = d.range_of(1).unwrap();
    assert!(r.start <= r.end, "full(0) 不得产生 start > end（饱和）");
}

/// 边界：空基线（首帧）→ 全图脏。
#[test]
fn first_frame_all_dirty() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let mut det = DirtyDetector::new();
    let d = det.detect(&chain_graph(0.5), &empty, &[], 5, &intervals).unwrap();
    for id in [1u64, 2, 3, 4] {
        assert!(d.contains(id), "首帧应全图脏：缺 {id}");
    }
}

/// 边界：两遍逐位一致（同输入两次 detect 结果一致）。
#[test]
fn detect_deterministic_two_passes() {
    let empty = ChannelDefs::new();
    let intervals = ChannelIntervals::new();
    let total = 20u64;
    let run = || {
        let mut det = DirtyDetector::new();
        det.detect(&chain_graph(0.5), &empty, &[], total, &intervals).unwrap();
        let change = ParamChange {
            node: 1,
            channel: None,
            changed_frames: FrameRange::single(7),
        };
        det.detect(&chain_graph(0.8), &empty, &[change], total, &intervals).unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "两遍脏推导必须一致");
}
