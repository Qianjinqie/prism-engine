//! prism-raster 节点注册（D-27 静态汇入制）。
//!
//! `PolygonWinding` 是扫描线光栅的数学内核特解：任意点对多边形的
//! winding 数（水平射线法）——光栅覆盖率正是该谓词在像素面积上的积分。
//!
//! 参数装配说明：求值器 SolvedParams 注入位随 A2 批次落地（eval.rs 已
//! 记账）；本特解以实例状态携带路径（`Default` = 空路径，winding 恒 0，
//! 数学上诚实非桩），`with_path` 供直接构造。图接线（fbuf 载荷）随 C1
//! Compose 波次接入。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};
use prism_core::node::Params;

use crate::geom::Point;

/// S13@prism_raster：多边形 winding 数特解（非零规则）。
#[derive(Debug)]
pub struct PolygonWinding {
    meta: StructureMeta,
    contours: Vec<Vec<Point>>,
}

impl Default for PolygonWinding {
    fn default() -> Self {
        Self {
            meta: StructureMeta::new(
                StructureId::custom(13, env!("CARGO_CRATE_NAME")),
                KernelKind::NoKernel,
                DualityPair::BoundaryVolume,
                LatticePlan::LatticeAnchoredInterp { kernel: "analytic-coverage" },
                OpCapabilities { invert: false, sample: false },
            ),
            contours: Vec::new(),
        }
    }
}

impl PolygonWinding {
    /// 以轮廓集合构造实例（每轮廓为顶点环，首尾自动闭合）。
    pub fn with_path(contours: Vec<Vec<Point>>) -> Self {
        let mut s = Self::default();
        s.contours = contours;
        s
    }

    /// 水平射线法 winding 数：右向射线与边相交的带权和
    /// （边纵跨 `min_y ≤ y < max_y` 半开区间、交点在右侧记 ±1）。
    pub fn winding_at(contours: &[Vec<Point>], px: f32, py: f32) -> f32 {
        let mut w = 0.0f32;
        for ring in contours {
            let n = ring.len();
            for i in 0..n {
                let a = ring[i];
                let b = ring[(i + 1) % n];
                let (y0, y1) = if a.y <= b.y { (a.y, b.y) } else { (b.y, a.y) };
                if py >= y0 && py < y1 {
                    let xi = a.x + (py - a.y) / (b.y - a.y) * (b.x - a.x);
                    if xi > px {
                        w += if b.y > a.y { 1.0 } else { -1.0 };
                    }
                }
            }
        }
        w
    }
}

impl MathStructure for PolygonWinding {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, _ctx: &EvalCtx) -> Payload {
        let Payload::Vec3(v) = x else { return Payload::Empty };
        Payload::Scalar(Self::winding_at(&self.contours, v.x, v.y))
    }
}

prism_core::registry! {
    structure PolygonWinding => StructureId::custom(13, env!("CARGO_CRATE_NAME"));
}

// 供序列化消费方引用（保持 prism-core 参数面可见性）。
#[allow(unused)]
fn _params_type_anchor(_: &Params) {}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::NodeRegistry;

    fn unit_square() -> Vec<Vec<Point>> {
        vec![vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ]]
    }

    #[test]
    fn winding_number_matches_containment() {
        let spec = PolygonWinding::with_path(unit_square());
        let ctx = EvalCtx::new(0, 1, 0);
        // 内部点 winding = ±1（非零即内）；外部点 = 0。
        let inside = spec.apply(&Payload::Vec3(prism_math::Vec3::new(1.0, 1.0, 0.0)), &ctx);
        let outside = spec.apply(&Payload::Vec3(prism_math::Vec3::new(5.0, 1.0, 0.0)), &ctx);
        assert_eq!(inside, Payload::Scalar(1.0));
        assert_eq!(outside, Payload::Scalar(0.0));
        // 与光栅覆盖率互证：winding ≠ 0 的点必在覆盖区内。
        let cov = crate::rasterize(&unit_square(), crate::FillRule::Winding, 4, 4);
        for (i, v) in cov.data.iter().enumerate() {
            let x = (i % 4) as f32 + 0.5;
            let y = (i / 4) as f32 + 0.5;
            let w = PolygonWinding::winding_at(&unit_square(), x, y);
            assert_eq!(
                *v > 0.0,
                w != 0.0,
                "点 ({x},{y}) 覆盖 {v} 与 winding {w} 判定不一致"
            );
        }
    }

    #[test]
    fn registry_static_import_resolves_by_name() {
        // D-27：registry! 展开的 __prism_collect 由可执行 crate 静态汇入；
        // 这里直接调汇入入口验证外部 crate 通道可用。
        let mut reg = NodeRegistry::new();
        __prism_collect(&mut reg);
        let id = StructureId::custom(13, "prism_raster");
        assert!(reg.contains(&id));
        let inst = reg.resolve_by_name("PolygonWinding").expect("按名解析");
        assert_eq!(inst.meta().id, id);
        // Default 实例 = 空路径 → winding 恒 0（数学诚实，非桩）。
        let ctx = EvalCtx::new(0, 1, 0);
        assert_eq!(
            inst.apply(&Payload::Vec3(prism_math::Vec3::new(1.0, 1.0, 0.0)), &ctx),
            Payload::Scalar(0.0)
        );
    }
}
