//! 快照装配的 seek 语义（QA，第 4 轮研究后定稿）。
//!
//! 背景：`eval.rs` 的 ConstraintFlow 快照路径曾用 `unwrap_or(input)` 静默
//! fallback——seek 断档（时间线拖动 / LRU 逐出 / 缓存清空）会让输出与首帧
//! 逐位相同，错误状态写回快照库并沿轨道传播，全程无信号（C20 违反）。
//! 研究定位见《docs/设计/研究/求值台的数学.md》命题 4.1。
//!
//! 定稿行为：`frame > 0` 且 `(sig, frame−1)` 缺失 ⟹ 结构化报错 P3012
//! （三段式：在哪 / 什么事 / 怎么改）。`frame == 0` 维持「图输入作初值」。

use prism_core::eval::{eval, SamplePlane};
use prism_core::graph::Graph;
use prism_core::node::{
    Composition, DualityPair, EvalCtx, Hash256, KernelKind, LatticePlan, MathStructure, Node,
    OpCapabilities, Payload, StructureId, StructureMeta,
};
use prism_core::registry::NodeRegistry;

const TEST_SIG: Hash256 = [0xCD; 32];

/// 状态 +1 一步投影特解（Empty = 初值 0）。
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

fn registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.collect(
        "step",
        StructureMeta::new(
            StructureId::ConstraintFlow,
            KernelKind::Green,
            DualityPair::PrimalMultiplier,
            LatticePlan::LatticeAnchored,
            OpCapabilities { invert: false, sample: false },
        ),
        || Box::new(StepSpec::default()),
    )
    .unwrap();
    reg
}

fn graph() -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::new(9, StructureId::ConstraintFlow, Composition::Feed))
        .unwrap();
    g
}

fn plane() -> SamplePlane {
    SamplePlane { pixel: (0, 0), spp: 1 }
}

// —— 合法路径：首帧初值 + 顺序轨道 ——

#[test]
fn first_frame_uses_graph_input_as_initial_state() {
    let reg = registry();
    let g = graph();
    let mut ctx = EvalCtx::new(11, 0, 0);
    let out = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap();
    assert_eq!(out.get(9), Some(Payload::Scalar(0.0)), "首帧：图输入作初值");
}

#[test]
fn sequential_frames_produce_orbit_0_1_2() {
    let reg = registry();
    let g = graph();
    let mut ctx = EvalCtx::new(11, 0, 0);
    for expect in 0..=2u64 {
        ctx.frame = expect;
        let out = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap();
        assert_eq!(
            out.get(9),
            Some(Payload::Scalar(expect as f32)),
            "frame {expect} 应得轨道值 {expect}"
        );
    }
}

// —— 修复验证：seek 断档 → P3012 结构化报错（三段式）——

#[test]
fn seek_to_frame_without_snapshot_reports_p3012() {
    let reg = registry();
    let g = graph();
    let mut ctx = EvalCtx::new(11, 0, 5); // frame = 5，快照库空 = seek 场景
    let err = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx)
        .expect_err("seek 断档必须报错，绝不静默（C20）");

    assert_eq!(err.code, "P3012");
    assert!(err.location.contains("帧 5"), "在哪：{}", err.location);
    assert!(err.problem.contains("帧间状态缺失"), "什么事：{}", err.problem);
    assert!(
        err.fix.contains("从 0 帧起顺序求值"),
        "怎么改（建议段，C19）：{}",
        err.fix
    );
    // 无写回：报错的帧不污染快照库。
    assert_eq!(ctx.snapshots.len(), 0, "报错帧不得写回快照");
}

#[test]
fn seek_with_gap_reports_p3012() {
    let reg = registry();
    let g = graph();

    let mut ctx = EvalCtx::new(11, 0, 0);
    eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap(); // 写入 (sig, 0)
    let snapshots_after_first = ctx.snapshots.len();

    // 跳到 frame = 3：(sig, 2) 缺失（库里只有 (sig, 0)）。
    ctx.frame = 3;
    let err = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).expect_err("断档 seek 必须报错");
    assert_eq!(err.code, "P3012");
    assert_eq!(
        ctx.snapshots.len(),
        snapshots_after_first,
        "报错帧不得写回快照（旧行为会把错误状态固化为合法快照）"
    );
}

#[test]
fn same_frame_reevaluation_still_idempotent() {
    // 回归保护：同帧重求值幂等（读键 = frame−1，写键 = frame，互不污染）。
    let reg = registry();
    let g = graph();
    let mut ctx = EvalCtx::new(11, 0, 0);
    eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap(); // (sig,0)=0.0
    ctx.frame = 1;
    let a = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap();
    let b = eval(&g, &reg, &plane(), TEST_SIG, &mut ctx).unwrap();
    assert_eq!(a.get(9), b.get(9), "同帧重求值幂等");
    assert_eq!(ctx.snapshots.len(), 2, "两帧各一条快照，重求值不新增");
}
