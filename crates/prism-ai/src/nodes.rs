//! AI mock 节点（→ prism-core 注册表；D-27 静态汇入制）。
//!
//! 本文件实现一个确定性 [`MathStructure`] 特解，经 `registry!`（宏展开为
//! `__prism_collect(reg)` 注册函数）静态汇入 [`NodeRegistry`]——可执行 crate
//! 启动时汇入一次，无动态加载、无脚本层（D-27）。
//!
//! # 节点语义
//!
//! [`CueColorNode`]（S13 @ prism-ai）：给定输入 `Payload::Scalar(h)`，返回
//! 由 `hash(h, 语境)` 派生的确定性标量色相（0..=1）——用于演示 AI 草稿的
//! "调色由文本摘要派生"这一确定性管线。无网络、无随机、无隐藏状态。
//!
//! # 编号口径
//!
//! 走用户段 `StructureId::custom(13, "prism-ai")`（官方段留给 alias.rs A4）。
//! 三问：核 = 无核服务（纯查表派生，不做卷积）、对偶 = 内容摘要、格锚 =
//! 整数格锚（同输入同值，零浮点漂移）。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

use crate::hash::mix64;

/// CueColor 节点三问元数据。
fn cue_color_meta() -> StructureMeta {
    StructureMeta::new(
        StructureId::custom(13, env!("CARGO_CRATE_NAME")),
        KernelKind::NoKernel,
        DualityPair::ContentDigest,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: false },
    )
}

/// CueColor 节点（S13 @ prism-ai）：确定性色相派生。
///
/// 输入 `Payload::Scalar(h)`：伪摘要值（应用侧把文本摘要化为标量）；输出
/// `Payload::Scalar` ∈ `[0, 1]`。非标量输入 → `Payload::Empty`（单位元，
/// 不静默编造数值）。
#[derive(Debug)]
pub struct CueColorNode {
    meta: StructureMeta,
}

impl Default for CueColorNode {
    fn default() -> Self {
        Self { meta: cue_color_meta() }
    }
}

impl MathStructure for CueColorNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        match x {
            Payload::Scalar(h) => {
                // 派生：混洗整数位 → [0,1]；确定性、无 libm。
                let bits = mix64(f64::to_bits(*h as f64));
                let v = ((bits >> 11) as f32) / ((1u64 << 53) as f32);
                Payload::Scalar(v)
            }
            _ => Payload::Empty,
        }
    }
}

// D-27 静态汇入：可执行 crate 启动时调用 __prism_collect(reg) 一次即注册本节点。
prism_core::registry! {
    structure CueColorNode => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_collects_and_is_deterministic() {
        let mut reg = prism_core::NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 1);

        let id = StructureId::custom(13, env!("CARGO_CRATE_NAME"));
        let inst = reg.resolve(&id).expect("静态汇入后应可 resolve");
        assert_eq!(inst.meta().kernel, KernelKind::NoKernel);
        assert_eq!(inst.meta().duality, DualityPair::ContentDigest);

        let ctx = EvalCtx::new(0, 1, 0);
        let a = inst.apply(&Payload::Scalar(1.25), &ctx);
        let b = inst.apply(&Payload::Scalar(1.25), &ctx);
        assert_eq!(a, b, "同输入同输出");
        match a {
            Payload::Scalar(v) => assert!((0.0..=1.0).contains(&v), "色相应落在 [0,1]：{}", v),
            other => panic!("应为 Scalar，得 {:?}", other),
        }
        // 非标量输入 → Empty。
        assert_eq!(inst.apply(&Payload::Empty, &ctx), Payload::Empty);
    }
}
