//! CSG 树：图元 + 布尔/平滑/位移组合（S12 CSG 特解）。
//!
//! 树求值直接调用 [`crate::sdf`] 的域算子函数——CSG 与域算子单实现（D-16）。
//! [`Csg::dist_lod`] 支持 octaves 参数沿树传播到 displace 节点，供 [`crate::lod`] 按距离衰减。

use crate::primitive::{Box3, Capsule, Cylinder, Plane, Sdf, Sphere, Torus};
use crate::sdf;
use prism_math::Vec3;

/// CSG 树节点。叶子为图元，内部节点为域算子。
#[derive(Debug, Clone, PartialEq)]
pub enum Csg {
    // —— 图元叶子 ——
    /// 球。
    Sphere(Sphere),
    /// 轴对齐盒。
    Box(Box3),
    /// 平面（半空间，法线侧为正/外部）。
    Plane(Plane),
    /// 圆柱（+Y 轴向）。
    Cylinder(Cylinder),
    /// 圆环（XZ 主环平面）。
    Torus(Torus),
    /// 胶囊（线段扫掠球）。
    Capsule(Capsule),
    // —— 布尔与平滑变体（复用 sdf 域算子）——
    /// 并集 `min(a, b)`。
    Union(Box<Csg>, Box<Csg>),
    /// 交集 `max(a, b)`。
    Intersection(Box<Csg>, Box<Csg>),
    /// 差集 `a − b`。
    Subtraction(Box<Csg>, Box<Csg>),
    /// 平滑并集（k 控制融合半径）。
    SmoothUnion(Box<Csg>, Box<Csg>, f32),
    /// 平滑交集。
    SmoothIntersection(Box<Csg>, Box<Csg>, f32),
    /// 平滑差集。
    SmoothSubtraction(Box<Csg>, Box<Csg>, f32),
    // —— 域算子 ——
    /// 平移子场。
    Translate(Box<Csg>, Vec3),
    /// 噪声位移子场（amp 幅度 / freq 频率 / seed 种子，噪声来自 prism-noise fbm）。
    Displace(Box<Csg>, f32, f32, u64),
}

impl Csg {
    /// 带 LOD 的求值：`octaves` 传播到 displace 节点（距离越远档位越低）。
    pub fn dist_lod(&self, p: Vec3, octaves: u32) -> f32 {
        match self {
            Csg::Sphere(s) => s.dist(p),
            Csg::Box(b) => b.dist(p),
            Csg::Plane(pl) => pl.dist(p),
            Csg::Cylinder(c) => c.dist(p),
            Csg::Torus(t) => t.dist(p),
            Csg::Capsule(c) => c.dist(p),
            Csg::Union(a, b) => sdf::union(a.dist_lod(p, octaves), b.dist_lod(p, octaves)),
            Csg::Intersection(a, b) => {
                sdf::intersection(a.dist_lod(p, octaves), b.dist_lod(p, octaves))
            }
            Csg::Subtraction(a, b) => sdf::subtraction(a.dist_lod(p, octaves), b.dist_lod(p, octaves)),
            Csg::SmoothUnion(a, b, k) => {
                sdf::smooth_min(a.dist_lod(p, octaves), b.dist_lod(p, octaves), *k)
            }
            Csg::SmoothIntersection(a, b, k) => {
                sdf::smooth_intersection(a.dist_lod(p, octaves), b.dist_lod(p, octaves), *k)
            }
            Csg::SmoothSubtraction(a, b, k) => {
                sdf::smooth_subtraction(a.dist_lod(p, octaves), b.dist_lod(p, octaves), *k)
            }
            Csg::Translate(child, off) => child.dist_lod(p - *off, octaves),
            Csg::Displace(child, amp, freq, seed) => {
                let base = child.dist_lod(p, octaves);
                sdf::displace_lod(base, p, *amp, *freq, *seed, octaves)
            }
        }
    }
}

impl Sdf for Csg {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        self.dist_lod(p, sdf::DEFAULT_DISPLACE_OCTAVES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::TraceParams;

    const SEED: u64 = 0xC5C;

    fn sphere(center: Vec3, r: f32) -> Csg {
        Csg::Sphere(Sphere { center, radius: r })
    }

    /// 关卡 2a：并/交/差在采样点上与域算子公式逐点一致（性质级解析对照）。
    #[test]
    fn boolean_ops_match_domain_formulas() {
        let a = sphere(Vec3::new(0.0, 0.0, -5.0), 2.0);
        let b = sphere(Vec3::new(1.0, 1.0, -5.0), 1.5);
        let (uni, inter, diff) = (
            Csg::Union(Box::new(a.clone()), Box::new(b.clone())),
            Csg::Intersection(Box::new(a.clone()), Box::new(b.clone())),
            Csg::Subtraction(Box::new(a.clone()), Box::new(b.clone())),
        );
        for i in 0..200 {
            // 覆盖两球重叠区域的采样点。
            let p = Vec3::new(
                3.0 * (i % 17) as f32 / 8.0 - 2.0,
                3.0 * (i % 13) as f32 / 6.0 - 1.5,
                -5.0 + 3.0 * (i % 11) as f32 / 5.0 - 1.5,
            );
            let (da, db) = (a.dist(p), b.dist(p));
            assert_eq!(uni.dist(p), da.min(db), "并集 @ {p:?}");
            assert_eq!(inter.dist(p), da.max(db), "交集 @ {p:?}");
            assert_eq!(diff.dist(p), da.max(-db), "差集 @ {p:?}");
        }
    }

    /// 关卡 2b：平滑最小解析对照——已知名值、k→0 退化、界约束；平滑差集挖圆角。
    #[test]
    fn smooth_ops_analytic_reference() {
        // 已知名值：两球表面相切点（各距 0）→ smin = −k/4；等距点 → d − k/4。
        let s1 = sphere(Vec3::splat(0.0), 1.0);
        let s2 = sphere(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let sm = Csg::SmoothUnion(Box::new(s1.clone()), Box::new(s2.clone()), 0.5);
        let mid = Vec3::new(1.0, 0.0, 0.0); // 两球表面相切点，各距 0
        let got = sm.dist(mid);
        let want = -0.5 / 4.0; // smin(0, 0, k) = −k/4
        assert!((got - want).abs() < 1e-5, "smin(d,d,k)={got} vs {want}");
        // 等距点 (1,1,0)：各距 √2−1 → smin = (√2−1) − k/4。
        let eq = Vec3::new(1.0, 1.0, 0.0);
        let de = s1.dist(eq); // == s2.dist(eq)
        assert!((s2.dist(eq) - de).abs() < 1e-6, "该点应等距");
        let want_eq = de - 0.5 / 4.0;
        assert!((sm.dist(eq) - want_eq).abs() < 1e-5, "smin(d,d,k)={} vs {want_eq}", sm.dist(eq));
        // 融合使相切点下沉为负——表面融合语义生效。
        assert!(got < 0.0);
        // k → 0 极限：平滑并集退化为精确 min。
        let tiny = Csg::SmoothUnion(Box::new(s1.clone()), Box::new(s2.clone()), 1e-6);
        let uni = Csg::Union(Box::new(s1.clone()), Box::new(s2.clone()));
        for i in 0..20 {
            let p = Vec3::new(0.1 * i as f32 - 1.0, 0.05, -0.2);
            assert!(
                (tiny.dist(p) - uni.dist(p)).abs() < 1e-4,
                "k→0 应退化 @ {p:?}: {} vs {}",
                tiny.dist(p),
                uni.dist(p)
            );
        }
        // 平滑差集：挖出圆角孔——孔缘点比硬差集更近（融合内收）。
        let boxy = Csg::Box(Box3 { center: Vec3::splat(0.0), half: Vec3::splat(1.0) });
        let drill = sphere(Vec3::new(0.0, 0.0, -3.0), 0.8);
        let hard = Csg::Subtraction(Box::new(boxy.clone()), Box::new(drill.clone()));
        let soft = Csg::SmoothSubtraction(Box::new(boxy), Box::new(drill), 0.3);
        let p = Vec3::new(0.0, 0.0, -1.2); // 孔正后方盒面附近
        assert!(soft.dist(p) >= hard.dist(p) - 1e-6, "平滑差不应越过硬差表面");
    }

    /// Translate 与 Displace 节点语义：平移等价于对子场取反平移点；amp=0 退化；确定性。
    #[test]
    fn translate_and_displace_nodes() {
        let s = sphere(Vec3::splat(0.0), 1.0);
        let moved = Csg::Translate(Box::new(s.clone()), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(moved.dist(Vec3::new(2.0, 0.0, 0.0)), s.dist(Vec3::splat(0.0)));
        assert_eq!(moved.dist(Vec3::new(3.0, 0.0, 0.0)), s.dist(Vec3::new(1.0, 0.0, 0.0)));
        // amp = 0：displace 节点与子场逐位一致。
        let flat = Csg::Displace(Box::new(s.clone()), 0.0, 2.0, SEED);
        for i in 0..10 {
            let p = Vec3::new(0.2 * i as f32 - 1.0, 0.3, -0.1);
            assert_eq!(flat.dist(p), s.dist(p));
        }
        // amp > 0：确定性（同参两遍一致）且确实改变场值。
        let dis = Csg::Displace(Box::new(s.clone()), 0.3, 2.0, SEED);
        let mut changed = false;
        for i in 0..24 {
            let p = Vec3::new(0.3 * i as f32 - 3.0, 0.13, -0.7);
            let (d1, d2) = (dis.dist(p), dis.dist(p));
            assert_eq!(d1, d2, "displace 必须逐位确定");
            changed |= (d1 - s.dist(p)).abs() > 1e-6;
        }
        assert!(changed, "amp>0 应实际位移场值");
    }

    /// CSG 场景 sphere tracing 冒烟：Translate + Union 组合场的命中与解析一致。
    #[test]
    fn csg_scene_trace() {
        let s1 = sphere(Vec3::splat(0.0), 1.0);
        let moved = Csg::Translate(Box::new(s1), Vec3::new(0.0, 0.0, -4.0));
        let params = TraceParams { hit_eps: 1e-5, ..TraceParams::default() };
        let got = crate::sdf::sphere_trace(&moved, Vec3::splat(0.0), Vec3::new(0.0, 0.0, -1.0), &params)
            .expect("应命中平移后的球");
        assert!((got.t - 3.0).abs() < 1e-4, "平移球正面 t={} vs 3.0", got.t);
    }
}
