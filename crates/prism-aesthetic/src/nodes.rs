//! critic 节点注册（D-27 静态汇入制）。
//!
//! 语义权威：节点注册表、DSL 是一切的声明入口、
//! D-22（数学结构第一性）、D-23（自定义光谱）、D-27（registry! 静态汇入制）。
//!
//! # 节点语义
//!
//! [`QualityScoreNode`]（S13 @ prism-aesthetic）：给定输入 `Payload::Scalar`
//! （应用侧把某帧的质量指标汇总为一个标量代理，如总评分），返回**确定性**
//! 的评分映射结果（原样透传并 clamp 到 `[0,1]`）——用于演示"critic 评分作为
//! 普通节点输出进图、可被任意下游节点消费"（铁律：硬实力节点化）。
//!
//! 三问：核 = 无核（纯映射，不做卷积）、对偶 = 内容摘要、格锚 = 整数格锚
//! （同输入同值，零浮点漂移）。与 prism-ai 的 CueColorNode 同构、不同编号段
//! （同 crate 内编号唯一即可；跨 crate 同编号按 crate 名天然隔离）。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

/// QualityScore 节点三问元数据（S13 @ prism-aesthetic）。
fn quality_meta() -> StructureMeta {
    StructureMeta::new(
        StructureId::custom(13, env!("CARGO_CRATE_NAME")),
        KernelKind::NoKernel,
        DualityPair::ContentDigest,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: false },
    )
}

/// QualityScore 节点（S13 @ prism-aesthetic）：质量评分透传 + 钳制到 `[0,1]`。
///
/// 输入 `Payload::Scalar(v)` → 输出 `Payload::Scalar(clamp(v, 0, 1))`；非标量
/// 输入 → `Payload::Empty`（单位元，不静默编造数值）。
#[derive(Debug)]
pub struct QualityScoreNode {
    meta: StructureMeta,
}

impl Default for QualityScoreNode {
    fn default() -> Self {
        Self { meta: quality_meta() }
    }
}

impl MathStructure for QualityScoreNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        match x {
            Payload::Scalar(v) => Payload::Scalar(prism_math::clamp(*v, 0.0, 1.0)),
            _ => Payload::Empty,
        }
    }
}

// D-27 静态汇入：可执行 crate 启动时调用 __prism_collect(reg) 一次即注册本节点。
prism_core::registry! {
    structure QualityScoreNode => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
}

// 注：`prism-aesthetic` 与 `prism-ai` 均用编号 13，但 `StructureId::custom`
// 携带 crate 名 → 跨 crate 天然隔离（NodeRegistry 以 (n, crate) 为键）。
// 若同 crate 内再注册另一结构需错开编号。

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_collects_and_clamps() {
        let mut reg = prism_core::NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 1);

        let id = StructureId::custom(13, env!("CARGO_CRATE_NAME"));
        let inst = reg.resolve(&id).expect("静态汇入后应可 resolve");
        assert_eq!(inst.meta().kernel, KernelKind::NoKernel);

        let ctx = EvalCtx::new(0, 1, 0);
        // 透传 + 钳制。
        assert_eq!(inst.apply(&Payload::Scalar(0.42), &ctx), Payload::Scalar(0.42));
        assert_eq!(inst.apply(&Payload::Scalar(2.0), &ctx), Payload::Scalar(1.0));
        assert_eq!(inst.apply(&Payload::Scalar(-1.0), &ctx), Payload::Scalar(0.0));
        // 确定性：同输入同输出。
        let a = inst.apply(&Payload::Scalar(0.7), &ctx);
        let b = inst.apply(&Payload::Scalar(0.7), &ctx);
        assert_eq!(a, b);
        // 非标量 → Empty。
        assert_eq!(inst.apply(&Payload::Empty, &ctx), Payload::Empty);
    }
}
