//! C1 合成域节点注册（D-27 静态汇入制）：深度合成与 Compose 归约结构的
//! 特解注册 + 三问元数据。
//!
//! 语义权威：《强大版路线图》C1 行（"节点注册（D-27 静态汇入）"）；《架构总设计》
//! v2.4 §2.6.2（registry! 双通道汇入同一注册表）、D-27（静态汇入制）、D-22
//! （数学结构第一性：新功能 = 新特解）。
//!
//! # 两个结构
//!
//! - [`DepthCompositeNode`]：深度合成（S5 重参数化族——按 z 差容差重映射
//!   合成权重），配对 SampleMeasure 对偶，格锚浮点；
//! - [`ComposeMergeNode`]：Compose 树归约（S9 分段多项式 / 组合代数），
//!   三问按"再生核 + 内容摘要 + 格锚"给出。
//!
//! 全部经 `registry!` 声明，由可执行 crate 启动静态汇入（[`collect`]）。

use crate::compose::blend_pixel;
use crate::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

/// 用户段起始编号（D-27：用户段自 S13 起；本 crate 用 S13/S14）。
const DEPTH_COMPOSITE_N: u16 = 13;
/// Compose 归约结构编号。
const COMPOSE_MERGE_N: u16 = 14;

/// 深度合成结构（S13@prism-core）：把两个标量深度按容差软权重融合
/// （Payload::Scalar 承载深度；完整多层合成见 `depth::DepthComposite`）。
#[derive(Debug)]
pub struct DepthCompositeNode {
    meta: StructureMeta,
}

impl Default for DepthCompositeNode {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(DEPTH_COMPOSITE_N, env!("CARGO_CRATE_NAME")),
                // 问一：深度的软权重重映射是变量替换（z → 权重），取无核服务。
                KernelKind::NoKernel,
                // 问二：配对样本↔测度（深度合成是逐像素确定性合并，无随机）。
                DualityPair::SampleMeasure,
                // 问三：全部格锚浮点（z 差容差比较，无非格点插值）。
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for DepthCompositeNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    /// 作用：把 `x` 视为"近层深度"，与默认远层深度融合——返回软权重的
    /// 归一化结果（[0,1]）。真实多层合成由 `depth::DepthComposite` 承载，
    /// 本结构提供单值软权重（tolerance = 0.1 缺省）。
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        match x {
            Payload::Scalar(z) => {
                // 相对参考深度 0.5 的软权重（tolerance 0.1）。
                let w = prism_math::clamp(0.5 + (*z - 0.5) / (2.0 * 0.1), 0.0, 1.0);
                Payload::Scalar(w)
            }
            other => *other,
        }
    }
}

/// Compose 归约结构（S14@prism-core）：over 混合的逐像素双输入算子
/// （色彩节点链里的合成算子特解）。
#[derive(Debug)]
pub struct ComposeMergeNode {
    meta: StructureMeta,
}

impl Default for ComposeMergeNode {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(COMPOSE_MERGE_N, env!("CARGO_CRATE_NAME")),
                // 问一：over 是覆盖核（无卷积支撑，取无核服务）。
                KernelKind::NoKernel,
                // 问二：配对内容↔摘要（合成结果由其输入层与层序唯一摘要）。
                DualityPair::ContentDigest,
                // 问三：全量格锚（逐像素有限项四则运算）。
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for ComposeMergeNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    /// 作用：标量 over——`a over b = a·(1−αb) + b·αb`，此处把 `x` 当 alpha，
    /// 与默认色 (1,1,1) 做 over（返回合成的 alpha 分量）。真实多层合成由
    /// `compose::ComposeNode` 承载。
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        match x {
            Payload::Scalar(alpha) => {
                let a = [1.0f32, 1.0, 1.0, 0.0]; // 下层不透明白，alpha=0
                let b = [1.0f32, 1.0, 1.0, prism_math::clamp(*alpha, 0.0, 1.0)];
                let m = blend_pixel(crate::node::MergeOp::Over, a, b);
                Payload::Scalar(m[3])
            }
            other => *other,
        }
    }
}

// registry! 静态汇入（D-27）：展开为 __prism_collect(reg) 注册函数。
crate::registry! {
    structure DepthCompositeNode => StructureId::custom(DEPTH_COMPOSITE_N, env!("CARGO_CRATE_NAME"));
    structure ComposeMergeNode => StructureId::custom(COMPOSE_MERGE_N, env!("CARGO_CRATE_NAME"));
}

/// 静态汇入入口：把 C1 合成域的两个结构注册进给定注册表（可执行 crate
/// 启动时调用一次；同表二次调用返回 P3003 错误——重复注册永不静默）。
pub fn collect(reg: &mut crate::NodeRegistry) -> Result<(), crate::RegistryError> {
    reg.collect(
        "DepthCompositeNode",
        DepthCompositeNode::default().meta,
        || Box::new(DepthCompositeNode::default()),
    )?;
    reg.collect(
        "ComposeMergeNode",
        ComposeMergeNode::default().meta,
        || Box::new(ComposeMergeNode::default()),
    )?;
    Ok(())
}

/// 供测试/调用方快速构造一个含 C1 结构的注册表。
pub fn default_registry() -> crate::NodeRegistry {
    let mut reg = crate::NodeRegistry::new();
    collect(&mut reg).expect("C1 结构汇入应成功");
    reg
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_macro_expands_collect_function() {
        // registry! 展开出 __prism_collect（D-27 静态汇入入口）。
        let mut reg = crate::NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 2, "C1 应注册两个结构");
    }

    #[test]
    fn c1_structures_resolve_with_three_question_meta() {
        let reg = default_registry();
        let id = StructureId::custom(DEPTH_COMPOSITE_N, env!("CARGO_CRATE_NAME"));
        let inst = reg.resolve(&id).expect("深度合成结构应可解析");
        // 三问齐全（非 Unspecified）。
        assert_ne!(inst.meta().kernel, KernelKind::Unspecified);
        assert_ne!(inst.meta().duality, DualityPair::Unspecified);
        assert!(matches!(inst.meta().lattice, LatticePlan::LatticeAnchored));

        // 按 id 查元数据；同名重复注册被拒。
        assert!(reg.contains(&id));
        assert!(reg.meta_of(&id).is_some());
    }

    #[test]
    fn depth_composite_apply_is_soft_weight() {
        let n = DepthCompositeNode::default();
        let ctx = EvalCtx::new(0, 0, 0);
        let val = |z: f32| match n.apply(&Payload::Scalar(z), &ctx) {
            Payload::Scalar(v) => v,
            _ => panic!("应返回标量"),
        };
        // z = 0.5（参考深度）→ 权重 0.5（f32 舍入允许小误差）。
        assert!((val(0.5) - 0.5).abs() < 1e-6, "z=0.5 → 0.5");
        // z = 0.6 → 0.5 + 0.1/0.2 = 1.0（钳制）。
        assert!((val(0.6) - 1.0).abs() < 1e-6, "z=0.6 → 1.0");
        // z = 0.4 → 0.5 − 0.5 = 0.0（f32 舍入后近 0）。
        assert!(val(0.4) < 1e-6, "z=0.4 → 0.0，实际 {}", val(0.4));
    }

    #[test]
    fn compose_merge_apply_returns_over_alpha() {
        let n = ComposeMergeNode::default();
        let ctx = EvalCtx::new(0, 0, 0);
        // alpha over：下层 alpha 0 → 合成 alpha = 上层 alpha。
        assert_eq!(n.apply(&Payload::Scalar(0.5), &ctx), Payload::Scalar(0.5));
        assert_eq!(n.apply(&Payload::Scalar(1.0), &ctx), Payload::Scalar(1.0));
    }

    #[test]
    fn collect_rejects_double_registration() {
        let mut reg = crate::NodeRegistry::new();
        collect(&mut reg).expect("首次汇入应成功");
        // 同表二次汇入 → 三问已通过但重复注册（P3003）。
        let err = collect(&mut reg).unwrap_err();
        assert_eq!(err.code, "P3003");
    }

    #[test]
    fn c1_node_signatures_are_stable() {
        // 结构 id 打印稳定（存档按名记录，D-23）。
        let id = StructureId::custom(DEPTH_COMPOSITE_N, env!("CARGO_CRATE_NAME"));
        assert_eq!(id.to_string(), format!("S13@{}", env!("CARGO_CRATE_NAME")));
    }
}
