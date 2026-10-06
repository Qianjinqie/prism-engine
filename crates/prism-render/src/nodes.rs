//! prism-render 节点注册（D-27 静态汇入制）。
//!
//! `DirtyRange` 是增量渲染（C3）的脏区间闭包特解：给定"参数实际变化的帧
//! 区间"，`apply` 返回同一区间在祖先链上的**闭包**语义等价表示（此处以
//! `(start, end, span)` 标量三元组表达）——确定性纯函数，把"哪些帧需要
//! 重算"从工程数据结构提升为可注册、可签名、可复用的数学特解（D-22：
//! 新功能 = 新特解；D-27：宏展开为 `__prism_collect`，静态汇入单一注册表）。
//!
//! 语义边界：脏区间**推导**（签名比对 + 祖先传播）在 [`crate::incremental`]；
//! 本特解只承载"区间闭包"这一可复用内核，供调度器按 StructureId 求值。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

/// S15@prism_render：脏帧区间闭包特解。
///
/// `apply(Vec3(start, end, child_span)) → Vec3(start, end, span)`：
/// - 输入 x = 变更起始帧，y = 变更结束帧，z = 子变更跨度（用于取闭包并集）；
/// - 输出 = 归一化后的闭区间三元组（start ≤ end，span = end − start + 1）。
///
/// 确定性：纯整数/浮点运算，无隐藏状态；非法输入（start > end）安全归一化
/// 为单帧区间（取 start）。
#[derive(Debug)]
pub struct DirtyRange {
    meta: StructureMeta,
}

impl Default for DirtyRange {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(15, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::ContentDigest,
                LatticePlan::LatticeAnchored,
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for DirtyRange {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    /// `Vec3(start, end, child_span) → Vec3(start, end, span)`（闭区间归一化）。
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        let Payload::Vec3(v) = x else { return Payload::Empty };
        let start = v.x;
        let end = v.y;
        // 归一化：start ≤ end；非法取 start 单帧。
        let (s, e) = if end >= start { (start, end) } else { (start, start) };
        Payload::Vec3(prism_math::Vec3::new(s, e, e - s + 1.0))
    }
}

prism_core::registry! {
    structure DirtyRange => StructureId::custom(15, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::NodeRegistry;
    use prism_math::Vec3;

    #[test]
    fn dirty_range_closure_is_deterministic_and_normalized() {
        let spec = DirtyRange::default();
        let ctx = EvalCtx::new(0, 1, 0);
        // 区间 [40, 40] → span 1。
        let out = spec.apply(&Payload::Vec3(Vec3::new(40.0, 40.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(40.0, 40.0, 1.0)));
        // 区间 [10, 19] → span 10。
        let out = spec.apply(&Payload::Vec3(Vec3::new(10.0, 19.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(10.0, 19.0, 10.0)));
        // 非法（start > end）→ 归一化为单帧 start。
        let out = spec.apply(&Payload::Vec3(Vec3::new(9.0, 3.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Vec3(Vec3::new(9.0, 9.0, 1.0)));
        // 非 Vec3 输入 → Empty（类型安全，不 panic）。
        assert_eq!(spec.apply(&Payload::Empty, &ctx), Payload::Empty);
    }

    #[test]
    fn registry_collects_dirty_range() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        assert_eq!(reg.len(), 1);
        let id = StructureId::custom(15, "prism_render");
        assert!(reg.contains(&id), "DirtyRange 应按 id 汇入注册表");
        let inst = reg.resolve_by_name("DirtyRange").expect("应按名解析");
        let ctx = EvalCtx::new(0, 1, 0);
        assert_eq!(
            inst.apply(&Payload::Vec3(Vec3::new(0.0, 4.0, 0.0)), &ctx),
            Payload::Vec3(Vec3::new(0.0, 4.0, 5.0))
        );
    }
}
