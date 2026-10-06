//! prism-text 节点注册（D-27 静态汇入制）。
//!
//! `GlyphAdvance` 是文本管线的度量内核特解：`apply(Vec3(gid, size,
//! spacing)) → advance(px)`——水平排版步进的解析解（hmtx/upem 线性映射）。
//!
//! 参数装配说明：求值器 SolvedParams 注入位随 A2 批次落地（eval.rs 已
//! 记账）；本特解的字体源为**常量资产**（内嵌 OFL 子集，非运行时状态），
//! `Default` 实例即给出真实 advance——数学诚实，非桩。图接线随 C1
//! Compose 波次接入。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

/// S14@prism_text：字形步进宽特解（hmtx 线性度量）。
#[derive(Debug)]
pub struct GlyphAdvance {
    meta: StructureMeta,
}

impl Default for GlyphAdvance {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(14, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::BoundaryVolume,
                LatticePlan::LatticeAnchoredInterp { kernel: "ttf-hmtx-advance" },
                OpCapabilities { invert: false, sample: false },
            ),
        }
    }
}

impl MathStructure for GlyphAdvance {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    /// `Vec3(gid, px_size, letter_spacing) → Scalar(advance_px)`。
    ///
    /// 约定：gid 编码于 x 通道（f32 精确整数域；`as u16` 饱和转换，
    /// 非法输入安全落到 .notdef/0 语义）；y 通道 = 字号 px；z 通道 =
    /// 追踪 px。advance = hmtx[gid]/upem × size + spacing。
    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        let Payload::Vec3(v) = x else { return Payload::Empty };
        let font = crate::font_regular();
        let gid = v.x as u16; // f32→u16 饱和：负→0，超界→65535（安全不 panic）。
        let scale = v.y / font.units_per_em() as f32;
        Payload::Scalar(font.advance(gid) as f32 * scale + v.z)
    }
}

prism_core::registry! {
    structure GlyphAdvance => StructureId::custom(14, env!("CARGO_CRATE_NAME"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::NodeRegistry;
    use prism_math::Vec3;

    #[test]
    fn advance_solves_full_width_cjk() {
        // Noto Sans SC 全角字符 hmtx = 1000 upem → size 24 → 24px。
        let spec = GlyphAdvance::default();
        let ctx = EvalCtx::new(0, 1, 0);
        let font = crate::font_regular();
        let gid = font.glyph_index('中').expect("中 在子集内");
        let out = spec.apply(&Payload::Vec3(Vec3::new(gid as f32, 24.0, 0.0)), &ctx);
        assert_eq!(out, Payload::Scalar(24.0));
        // 追踪叠加。
        let out = spec.apply(&Payload::Vec3(Vec3::new(gid as f32, 24.0, 1.5)), &ctx);
        assert_eq!(out, Payload::Scalar(25.5));
    }

    #[test]
    fn registry_static_import_resolves_by_name() {
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        let id = StructureId::custom(14, "prism_text");
        assert!(reg.contains(&id));
        let inst = reg.resolve_by_name("GlyphAdvance").expect("按名解析");
        assert_eq!(inst.meta().id, id);
        // Default 实例即真实度量（常量资产），非桩。
        let ctx = EvalCtx::new(0, 1, 0);
        let font = crate::font_regular();
        let gid = font.glyph_index('A').expect("A 在子集内");
        let adv = font.advance(gid) as f32;
        let out = inst.apply(
            &Payload::Vec3(Vec3::new(gid as f32, font.units_per_em() as f32, 0.0)),
            &ctx,
        );
        assert_eq!(out, Payload::Scalar(adv), "upem 字号下 advance = hmtx 原值");
        // 非 Vec3 载荷 → Empty（通道契约）。
        assert_eq!(inst.apply(&Payload::Scalar(1.0), &ctx), Payload::Empty);
    }
}
