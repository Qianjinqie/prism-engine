//! 交互式编辑器 —— 集成测试（硬性检查项集中在此）。
//!
//! 覆盖"硬性判据"四条：
//!
//! * **确定断言**：编辑序列重放后图状态**字节级一致**（`replay_is_byte_identical`、
//!   `undo_then_redo_field_by_field_identical`）；debug + release 双端一致由双 profile 测试保证。
//! * **增量失效精确**：dirty 集合**恰好**是下游子图（`incremental_dirty_is_exact_downstream`、
//!   `incremental_recompute_set_equals_dirty_set`）。
//! * **撤销/重做多步任意位置正确**（`multi_step_undo_redo_arbitrary_positions`）。
//! * **撤销栈空返回结构化错误不 panic**（`empty_undo_stack_is_structured_error`）。
//! * **Draft 采样数低于 Final**（`draft_spp_strictly_lower_than_final`）。
//!
//! # 零第三方依赖
//! 仅用 `std` + workspace 内 `prism-app` / `prism-core`。

use std::collections::BTreeSet;

use prism_app::editor::{
    build_reference_graph, EditCommand, EditError, EditorState, PreviewQuality, PreviewRenderer,
    DRAFT_SPP, FINAL_SPP,
};
use prism_core::node::ParamValue;
use prism_core::signature::ChannelDefs;

// ---------------------------------------------------------------------------
// 参考图上的增量失效
// ---------------------------------------------------------------------------

#[test]
fn incremental_dirty_is_exact_downstream() {
    // 参考图：1 叶 ┐
    //            3 = Merge(Plus) ── 4 根
    //          2 叶 ┘
    // 改叶 1 → 受影响子图恰为 {1,3,4}；兄弟叶 2 逐位不受影响。
    let mut st = EditorState::with_reference_graph();
    let dirty = st.apply_edit(1, "scale", ParamValue::F32(0.9)).unwrap();
    assert_eq!(dirty.to_vec(), vec![1, 3, 4], "dirty 必须是下游闭包，不多不少");
    assert!(!dirty.contains(2), "兄弟叶 2 不得被标记 dirty");
    assert!(!dirty.is_structural(), "参数编辑不属结构失效");
    assert_eq!(dirty.len(), 3);
}

#[test]
fn incremental_dirty_on_root_edit_is_root_only() {
    // 改根节点自身 → 只有根脏（无下游）。
    let mut st = EditorState::with_reference_graph();
    let dirty = st.apply_edit(4, "radius", ParamValue::F32(0.8)).unwrap();
    assert_eq!(dirty.to_vec(), vec![4], "根节点无下游，dirty 仅自身");
}

#[test]
fn idempotent_edit_yields_empty_dirty() {
    // 幂等编辑：参数改成逐位相同的值 → 签名不变 → 零重算。
    let mut st = EditorState::with_reference_graph();
    let before = st.canonical_bytes();
    let dirty = st.apply_edit(1, "scale", ParamValue::F32(0.5)).unwrap();
    assert!(dirty.is_empty(), "幂等编辑不得触发任何重算，实际 {:?}", dirty.to_vec());
    assert_eq!(st.canonical_bytes(), before, "幂等编辑后图字节不变");
    // 注意：命令仍入撤销栈（是"发生过的一次编辑"），故 can_undo 为真。
    assert!(st.can_undo());
}

#[test]
fn dirty_set_is_independent_of_edit_order_but_content_is_not() {
    // 编辑顺序不同 → dirty 集合各自正确；最终图内容按最后一次编辑而定。
    let mut a = EditorState::with_reference_graph();
    let da1 = a.apply_edit(1, "scale", ParamValue::F32(0.7)).unwrap();
    let da2 = a.apply_edit(2, "radius", ParamValue::F32(0.6)).unwrap();
    // 改叶 2 → {2,3,4}（叶 1 不在下游）。
    assert_eq!(da1.to_vec(), vec![1, 3, 4]);
    assert_eq!(da2.to_vec(), vec![2, 3, 4]);

    let mut b = EditorState::with_reference_graph();
    b.apply_edit(2, "radius", ParamValue::F32(0.6)).unwrap();
    b.apply_edit(1, "scale", ParamValue::F32(0.7)).unwrap();
    // 同终态 → canonical 字节一致（与顺序无关，canonical 是内容单射）。
    assert_eq!(a.canonical_bytes(), b.canonical_bytes(), "同终态必得同字节");
}

#[test]
fn incremental_recompute_set_equals_dirty_set() {
    // **本设计的核心不变式**：预览实际重算的节点集合 ≡ 签名比对得出的 dirty 集合。
    // 二者同由"子树签名是否变化"决定，故必然相等——这条断言把"签名增量"与
    // "真实省下的算力"钉在一起，防止将来某处改动让二者悄悄漂移。
    let mut st = EditorState::with_reference_graph();
    let mut r = PreviewRenderer::new();
    let defs = ChannelDefs::new();

    // 建立初始缓存。
    let g0 = st.model().to_graph().unwrap();
    r.render(&g0, &defs, PreviewQuality::Final, 64, 64, 1, 0).unwrap();

    // 依次改两个叶子，每次都比对 dirty 集与重算集。
    for (node, param, val) in [
        (1u64, "scale", ParamValue::F32(0.9)),
        (2, "radius", ParamValue::F32(0.55)),
        (3, "strength", ParamValue::F32(0.11)),
    ] {
        let dirty: BTreeSet<u64> = st.apply_edit(node, param, val).unwrap().to_vec().into_iter().collect();
        let g = st.model().to_graph().unwrap();
        let out = r.render(&g, &defs, PreviewQuality::Final, 64, 64, 1, 0).unwrap();
        let recomputed: BTreeSet<u64> = out.stats.recomputed.iter().copied().collect();
        let reused: BTreeSet<u64> = out.stats.reused.iter().copied().collect();
        assert_eq!(recomputed, dirty, "重算集必须恰等于 dirty 集（编辑节点 {node}）");
        assert!(
            recomputed.is_disjoint(&reused),
            "重算集与复用集必须不相交（编辑节点 {node}）"
        );
        // 锥内全部节点 = 重算 ∪ 复用，一个不多一个不少。
        assert_eq!(
            recomputed.len() + reused.len(),
            out.stats.cone_size,
            "锥内节点应恰好被分为重算或复用（编辑节点 {node}）"
        );
    }
}

// ---------------------------------------------------------------------------
// 撤销 / 重做
// ---------------------------------------------------------------------------

#[test]
fn empty_undo_stack_is_structured_error() {
    // 硬性判据：撤销栈空返回**结构化错误**，绝不 panic。
    let mut st = EditorState::with_reference_graph();
    assert!(!st.can_undo());
    assert_eq!(st.undo_depth(), 0);
    let err = st.undo().unwrap_err();
    match &err {
        EditError::EmptyUndoStack { location } => assert!(!location.is_empty()),
        other => panic!("应为 EmptyUndoStack，实际 {other:?}"),
    }
    // 三段式 + 建议段（宪章 C18/C19）。
    let text = err.to_string();
    assert!(text.starts_with("error[E4002] "), "首段【在哪】：{text}");
    assert!(text.contains("什么事："), "必含【什么事】：{text}");
    assert!(text.contains("建议："), "必含建议段（C19）：{text}");
    // 状态未被破坏。
    assert_eq!(st.undo_depth(), 0);
}

#[test]
fn empty_redo_stack_is_structured_error() {
    let mut st = EditorState::with_reference_graph();
    assert!(!st.can_redo());
    let err = st.redo().unwrap_err();
    assert!(matches!(err, EditError::EmptyRedoStack { .. }), "实际 {err:?}");
    assert!(err.to_string().contains("E4003"));
}

#[test]
fn multi_step_undo_redo_arbitrary_positions() {
    // 4 步编辑后，穷举撤销深度 0..=4 与重放深度，验证任意位置逐字段一致。
    let edits: [(u64, &str, ParamValue); 4] = [
        (1, "scale", ParamValue::F32(0.11)),
        (2, "radius", ParamValue::F32(0.22)),
        (3, "strength", ParamValue::F32(0.33)),
        (4, "radius", ParamValue::F32(0.44)),
    ];
    let mut st = EditorState::with_reference_graph();
    let initial = st.canonical_bytes();

    // 记录每步之后的字节（= 各历史位置的"真值"）。
    let mut expected: Vec<Vec<u8>> = vec![initial.clone()];
    for (node, param, val) in &edits {
        st.apply_edit(*node, *param, val.clone()).unwrap();
        expected.push(st.canonical_bytes());
    }
    assert_eq!(st.undo_depth(), 4);

    // 逐一验证：先撤销 k 步到每个位置，再逐步重做回满，每个位置都要对。
    //
    // 索引约定：`expected[i]` = 施加前 i 次编辑后的字节。施加全部 n 次编辑后
    // 状态为 `expected[n]`；撤销 k 步 → `expected[n - k]`；再重做 r 步 →
    // `expected[n - k + r]`。
    let n = edits.len();
    for k in 0..=n {
        let mut s = EditorState::with_reference_graph();
        for (node, param, val) in &edits {
            s.apply_edit(*node, *param, val.clone()).unwrap();
        }
        for _ in 0..k {
            s.undo().unwrap();
        }
        assert_eq!(
            s.canonical_bytes(),
            expected[n - k],
            "撤销 {k} 步后应逐字节等于施加 {} 次编辑的状态", n - k
        );
        assert_eq!(s.undo_depth(), n - k);
        // 再逐步重做回去，每一步都要对。
        for r in 1..=k {
            s.redo().unwrap();
            assert_eq!(
                s.canonical_bytes(),
                expected[n - k + r],
                "从撤销 {k} 步处重做第 {r} 步应逐字节一致"
            );
        }
        assert_eq!(s.undo_depth(), n, "重做回满后撤销栈深度应复原");
        assert!(!s.can_redo(), "重做回满后重做栈应为空");
    }
}

#[test]
fn undo_then_redo_field_by_field_identical() {
    // 硬性判据：撤销后重做与撤销前**逐字段**一致（含组合方式）。
    let mut st = EditorState::with_reference_graph();
    st.apply_edit(1, "scale", ParamValue::F32(0.77)).unwrap();
    st.apply_edit(1, "octaves", ParamValue::U32(7)).unwrap();
    let before_nodes: Vec<(u64, String, Vec<String>)> = snapshot(&st);
    let before_bytes = st.canonical_bytes();

    st.undo().unwrap();
    st.undo().unwrap();
    assert_eq!(st.undo_depth(), 0, "两次撤销后应回到初始");
    st.redo().unwrap();
    st.redo().unwrap();

    assert_eq!(snapshot(&st), before_nodes, "重做后逐字段应与撤销前一致");
    assert_eq!(st.canonical_bytes(), before_bytes, "重做后字节级一致");
    assert_eq!(st.undo_depth(), 2);
    assert_eq!(st.redo_depth(), 0);
}

#[test]
fn new_edit_clears_redo_stack() {
    // 分支语义：撤销后再编辑 → 重做栈清空（不能"重做"到被覆盖的历史）。
    let mut st = EditorState::with_reference_graph();
    st.apply_edit(1, "scale", ParamValue::F32(0.6)).unwrap();
    st.undo().unwrap();
    assert!(st.can_redo());
    st.apply_edit(1, "scale", ParamValue::F32(0.4)).unwrap();
    assert!(!st.can_redo(), "新编辑必须清空重做栈");
    assert_eq!(st.redo_depth(), 0);
    assert!(st.redo().is_err(), "重做栈空应结构化报错");
}

#[test]
fn composition_edit_is_undoable() {
    use prism_core::node::{Composition, MergeOp};
    let mut st = EditorState::with_reference_graph();
    let before = st.canonical_bytes();
    let dirty = st
        .apply_composition_edit(3, Composition::Merge(MergeOp::Multiply))
        .unwrap();
    assert_eq!(dirty.to_vec(), vec![3, 4], "改节点 3 组合 → {{3,4}} 脏");
    assert_ne!(st.canonical_bytes(), before);
    st.undo().unwrap();
    assert_eq!(st.canonical_bytes(), before, "组合方式撤销后字节还原");
}

#[test]
fn composition_edit_rejects_cross_variant() {
    use prism_core::node::Composition;
    let mut st = EditorState::with_reference_graph();
    // 节点 3 是 Merge，不能改成 Chain（变体不同）。
    let err = st.apply_composition_edit(3, Composition::Chain).unwrap_err();
    assert!(matches!(err, EditError::CompositionMismatch { .. }), "实际 {err:?}");
    assert!(err.to_string().contains("E4006"));
    // 状态未变。
    assert_eq!(st.undo_depth(), 0, "被拒的编辑不得入历史");
}

// ---------------------------------------------------------------------------
// 草稿档位
// ---------------------------------------------------------------------------

#[test]
fn draft_spp_strictly_lower_than_final() {
    // 硬性判据：Draft 采样数低于 Final。
    assert!(DRAFT_SPP < FINAL_SPP, "DRAFT_SPP({DRAFT_SPP}) 必须 < FINAL_SPP({FINAL_SPP})");
    assert!(PreviewQuality::Draft.spp() < PreviewQuality::Final.spp());
    // 档号也不同 → 缓存键不同 → 切档自然失效旧缓存（C13）。
    assert_ne!(PreviewQuality::Draft.spp_tier(), PreviewQuality::Final.spp_tier());
    assert_eq!(PreviewQuality::Draft.name(), "draft");
    assert_eq!(PreviewQuality::Final.name(), "final");
}

#[test]
fn quality_switch_does_not_touch_history() {
    // 切档不是编辑，不得污染撤销栈。
    let mut st = EditorState::with_reference_graph();
    st.apply_edit(1, "scale", ParamValue::F32(0.9)).unwrap();
    let depth = st.undo_depth();
    let bytes = st.canonical_bytes();
    st.set_preview_quality(PreviewQuality::Final);
    assert_eq!(st.undo_depth(), depth, "切档不得改变撤销栈深度");
    assert_eq!(st.canonical_bytes(), bytes, "切档不得改变图内容");
    assert_eq!(st.preview_quality(), PreviewQuality::Final);
}

// ---------------------------------------------------------------------------
// 确定性：编辑序列重放
// ---------------------------------------------------------------------------

#[test]
fn replay_is_byte_identical() {
    // 硬性判据：编辑序列重放后图状态**字节级**一致。
    let commands = vec![
        EditCommand::SetParam {
            node: 1,
            param: "scale".into(),
            old: ParamValue::F32(0.5),
            new: ParamValue::F32(0.83),
        },
        EditCommand::SetParam {
            node: 2,
            param: "radius".into(),
            old: ParamValue::F32(0.3),
            new: ParamValue::F32(0.41),
        },
        EditCommand::SetParam {
            node: 1,
            param: "octaves".into(),
            old: ParamValue::U32(4),
            new: ParamValue::U32(9),
        },
    ];

    // 原路径：逐条 apply。
    let mut st = EditorState::with_reference_graph();
    st.apply_edit(1, "scale", ParamValue::F32(0.83)).unwrap();
    st.apply_edit(2, "radius", ParamValue::F32(0.41)).unwrap();
    st.apply_edit(1, "octaves", ParamValue::U32(9)).unwrap();
    let original = st.canonical_bytes();

    // 重放路径：把导出的命令序列重放到一张全新图。
    let replayed = EditorState::replay_onto_fresh(build_reference_graph(), &commands).unwrap();
    assert_eq!(replayed, original, "编辑序列重放后必须字节级一致");
    assert!(!original.is_empty(), "canonical 字节不应为空");
}

#[test]
fn replay_is_idempotent_on_same_sequence() {
    // 同一序列重放两次 → 同一结果（确定性铁律：重放本身无副作用地幂等）。
    let commands = vec![EditCommand::SetParam {
        node: 3,
        param: "strength".into(),
        old: ParamValue::F32(0.7),
        new: ParamValue::F32(0.15),
    }];
    let a = EditorState::replay_onto_fresh(build_reference_graph(), &commands).unwrap();
    let b = EditorState::replay_onto_fresh(build_reference_graph(), &commands).unwrap();
    assert_eq!(a, b, "同序列重放必得同字节");
}

#[test]
fn inverse_command_restores_previous_state() {
    // 逆命令（交换 old/new）重放 → 回到编辑前的字节。
    let mut st = EditorState::with_reference_graph();
    let before = st.canonical_bytes();
    st.apply_edit(1, "scale", ParamValue::F32(0.95)).unwrap();
    let after = st.canonical_bytes();

    let cmd = st.undo().unwrap();
    assert_eq!(st.canonical_bytes(), before);
    // redo 重放原命令 → 回到编辑后（逆向验证 invert 的对称性）。
    st.redo().unwrap();
    let _ = cmd;
    assert_eq!(st.canonical_bytes(), after, "重做后应回到编辑后字节");
}

#[test]
fn export_edits_json_is_deterministic_and_ordered() {
    let mut st = EditorState::with_reference_graph();
    st.apply_edit(1, "scale", ParamValue::F32(0.31)).unwrap();
    st.apply_edit(2, "radius", ParamValue::F32(0.32)).unwrap();
    let a = st.export_edits_json();
    let b = st.export_edits_json();
    assert_eq!(a, b, "重复导出必须逐字节稳定");
    // 顺序 = 时间正向（撤销栈自底向上）。
    let i1 = a.find("\"node\":1").expect("应含节点 1 编辑");
    let i2 = a.find("\"node\":2").expect("应含节点 2 编辑");
    assert!(i1 < i2, "编辑应按时间正向排列：{a}");
    assert!(a.contains("\"schema\":1"), "应带 schema 版本头：{a}");
    assert!(a.contains("\"draft_spp\":1"), "应带草稿采样数：{a}");
    assert!(a.contains("\"final_spp\":16"), "应带成片采样数：{a}");
}

// ---------------------------------------------------------------------------
// 错误路径（永不 panic）
// ---------------------------------------------------------------------------

#[test]
fn unknown_node_and_param_are_structured_errors() {
    let mut st = EditorState::with_reference_graph();
    let e1 = st.apply_edit(99, "scale", ParamValue::F32(0.5)).unwrap_err();
    assert!(matches!(e1, EditError::UnknownNode { .. }), "实际 {e1:?}");
    assert!(e1.to_string().contains("E4001"));

    let e2 = st.apply_edit(1, "no_such_param", ParamValue::F32(0.5)).unwrap_err();
    assert!(matches!(e2, EditError::UnknownParam { .. }), "实际 {e2:?}");
    assert!(e2.to_string().contains("E4005"));

    // 状态未被污染。
    assert_eq!(st.undo_depth(), 0);
}

#[test]
fn param_type_mismatch_is_structured_error() {
    let mut st = EditorState::with_reference_graph();
    // octaves 是 u32，写 f32 应被拒。
    let e = st.apply_edit(1, "octaves", ParamValue::F32(1.5)).unwrap_err();
    match &e {
        EditError::ParamTypeMismatch { expected, got, .. } => {
            assert_eq!(*expected, "u32");
            assert_eq!(*got, "f32");
        }
        other => panic!("应为 ParamTypeMismatch，实际 {other:?}"),
    }
    assert!(e.to_string().contains("E4004"));
    assert_eq!(st.undo_depth(), 0);
}

#[test]
fn select_unknown_node_is_structured_error() {
    let mut st = EditorState::with_reference_graph();
    assert!(st.select(42).is_err());
    assert_eq!(st.selected(), None, "失败的 select 不得改变选中态");
    st.select(3).unwrap();
    assert_eq!(st.selected(), Some(3));
    st.clear_selection();
    assert_eq!(st.selected(), None);
}

// ---------------------------------------------------------------------------
// 导出 JSON
// ---------------------------------------------------------------------------

#[test]
fn graph_json_is_valid_shaped_and_deterministic() {
    let st = EditorState::with_reference_graph();
    let a = st.export_graph_json();
    let b = st.export_graph_json();
    assert_eq!(a, b, "图 JSON 导出必须逐字节稳定");
    assert!(a.starts_with('{') && a.ends_with('}'), "应为单个 JSON 对象：{a}");
    assert!(a.contains("\"nodes\":["));
    assert!(a.contains("\"params\":["));
    assert!(a.contains("\"inputs\":["));
    // 花括号配平（粗筛 JSON 结构完整性）。
    let opens = a.matches('{').count();
    let closes = a.matches('}').count();
    assert_eq!(opens, closes, "花括号应配平：{a}");
}

#[test]
fn schema_json_covers_all_params_of_selected_node() {
    let mut st = EditorState::with_reference_graph();
    st.select(1).unwrap();
    let js = st.export_schema_json(1);
    // 节点 1 有 scale + octaves 两个参数。
    assert!(js.contains("\"name\":\"scale\""), "{js}");
    assert!(js.contains("\"name\":\"octaves\""), "{js}");
    assert!(js.contains("\"t\":\"f32\""), "scale 应为 f32：{js}");
    assert!(js.contains("\"t\":\"u32\""), "octaves 应为 u32：{js}");
    // 不存在的节点 → 空 params，不 panic。
    let empty = st.export_schema_json(999);
    assert!(empty.contains("\"params\":[]"), "未知节点应给空 schema：{empty}");
}

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------

/// 快照：逐节点的 (id, 结构, 参数键集合)，用于"逐字段一致"断言。
fn snapshot(st: &EditorState) -> Vec<(u64, String, Vec<String>)> {
    st.model()
        .nodes()
        .map(|n| {
            (
                n.id,
                n.structure.to_string(),
                n.params.names().into_iter().cloned().collect(),
            )
        })
        .collect()
}
