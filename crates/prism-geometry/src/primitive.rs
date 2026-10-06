//! 解析图元族闭式 SDF（白皮书 §6 / B3；架构 v2.4 S12 解析图元特解）。
//!
//! 全部图元实现 [`Sdf`] trait：输入世界坐标点，输出有符号距离——**负在内、正在外、表面为零**。
//! 公式为标准闭式（Inigo Quilez 系），连续可微，供 sphere tracing 与域算子直接消费。

use crate::{vabs, vmax, vmaxc, xz_len};
use prism_math::Vec3;

/// 有符号距离场：点 → 距离（内负外正）。
pub trait Sdf {
    /// 点 `p` 到表面的有符号距离。
    fn dist(&self, p: Vec3) -> f32;
}

/// 球体：`dist(p) = |p - center| - radius`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    /// 球心。
    pub center: Vec3,
    /// 半径（> 0）。
    pub radius: f32,
}

impl Sdf for Sphere {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        (p - self.center).length() - self.radius
    }
}

/// 轴对齐长方体：`dist` 为精确欧氏距离（内部点到面距离）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box3 {
    /// 盒中心。
    pub center: Vec3,
    /// 半尺寸（各分量 > 0）。
    pub half: Vec3,
}

impl Sdf for Box3 {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        // q = |p - c| - h；外部：|max(q, 0)|；内部：min(max(q.x, q.y, q.z), 0)。
        let q = vabs(p - self.center) - self.half;
        let outside = vmax(q, Vec3::splat(0.0));
        outside.length() + vmaxc(q).min(0.0)
    }
}

/// 平面：`dist(p) = n·p - offset`，n 为单位法线（构造时归一化）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// 单位法线。
    pub normal: Vec3,
    /// 原点到平面的有符号距离（沿法线方向）。
    pub offset: f32,
}

impl Plane {
    /// 构造平面；法线按长度归一，`offset` 同步换算，保证 `dist` 公式直接成立。
    pub fn new(normal: Vec3, offset: f32) -> Self {
        let len = normal.length();
        debug_assert!(len > 1e-8, "plane normal must be non-degenerate");
        let n = normal / len;
        Plane { normal: n, offset: offset / len }
    }
}

impl Sdf for Plane {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        self.normal.dot(&p) - self.offset
    }
}

/// 圆柱体（轴向 +Y，带上下端盖）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    /// 圆柱中心（轴中点）。
    pub center: Vec3,
    /// 半径（> 0）。
    pub radius: f32,
    /// 半高（> 0）。
    pub half_height: f32,
}

impl Sdf for Cylinder {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        // 2D 子空间 (r, y) 上做"圆角盒"式闭式：q = (|r| - R, |y| - H)。
        // 外部：|max(q, 0)|；内部：min(max(q.r, q.y), 0)（注意不含第三分量）。
        let q = p - self.center;
        let dr = xz_len(q) - self.radius;
        let dy = q.y.abs() - self.half_height;
        let m = Vec3::new(dr.max(0.0), dy.max(0.0), 0.0);
        m.length() + dr.max(dy).min(0.0)
    }
}

/// 圆环体（主环位于 XZ 平面，轴向 +Y）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    /// 环心。
    pub center: Vec3,
    /// 主半径（环心到管心，> 0）。
    pub major: f32,
    /// 管半径（> 0，须 < major 才不自交）。
    pub minor: f32,
}

impl Sdf for Torus {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        let q = p - self.center;
        let d = Vec3::new(xz_len(q) - self.major, q.y, 0.0);
        d.length() - self.minor
    }
}

/// 胶囊体：线段 `a-b` 的扫掠球。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Capsule {
    /// 线段端点 A。
    pub a: Vec3,
    /// 线段端点 B。
    pub b: Vec3,
    /// 半径（> 0）。
    pub radius: f32,
}

impl Sdf for Capsule {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        let ab = self.b - self.a;
        let ap = p - self.a;
        // t = clamp(投影参数)，退化线段（|ab|≈0）时 t=0 安全。
        let t = if ab.length() < 1e-8 {
            0.0
        } else {
            (ap.dot(&ab) / ab.dot(&ab)).clamp(0.0, 1.0)
        };
        let closest = self.a + ab * t;
        (p - closest).length() - self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 球：中心距语义（内负外正、表面零）。
    #[test]
    fn sphere_known_values() {
        let s = Sphere { center: Vec3::new(1.0, 0.0, 0.0), radius: 2.0 };
        assert!((s.dist(Vec3::new(4.0, 0.0, 0.0)) - 1.0).abs() < 1e-6, "外部 +1");
        assert!((s.dist(Vec3::new(1.0, 0.0, 0.0)) + 2.0).abs() < 1e-6, "球心 -r");
        assert!(s.dist(Vec3::new(3.0, 0.0, 0.0)).abs() < 1e-6, "表面零");
        // 内部点到表面距离：|(2.5,0,0) 到球面| = 0.5 → dist = -0.5。
        assert!((s.dist(Vec3::new(2.5, 0.0, 0.0)) + 0.5).abs() < 1e-6);
    }

    /// 盒：外部角区欧氏、内部面距、表面零。
    #[test]
    fn box_known_values() {
        let b = Box3 { center: Vec3::splat(0.0), half: Vec3::new(1.0, 1.0, 1.0) };
        assert!(b.dist(Vec3::new(1.0, 0.0, 0.0)).abs() < 1e-6, "面点零");
        assert!((b.dist(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6, "面外 +1");
        // 角区外 (2,2,2)：最近点 (1,1,1)，距离 √3。
        assert!((b.dist(Vec3::new(2.0, 2.0, 2.0)) - 3.0_f32.sqrt()).abs() < 1e-6, "角区欧氏");
        // 内部 (0.5, 0, 0)：到最近面 0.5 → -0.5。
        assert!((b.dist(Vec3::new(0.5, 0.0, 0.0)) + 0.5).abs() < 1e-6, "内部面距");
        // 盒内中心：-1。
        assert!((b.dist(Vec3::splat(0.0)) + 1.0).abs() < 1e-6);
    }

    /// 平面：法线归一化与偏移换算。
    #[test]
    fn plane_normalized_construction() {
        // 非单位法线 (0,0,2)，offset 4 → 等价 z = 2 平面。
        let p = Plane::new(Vec3::new(0.0, 0.0, 2.0), 4.0);
        assert!((p.normal.z - 1.0).abs() < 1e-6);
        assert!((p.dist(Vec3::new(9.0, 9.0, 5.0)) - 3.0).abs() < 1e-6);
        assert!((p.dist(Vec3::new(0.0, 0.0, 0.0)) + 2.0).abs() < 1e-6, "平面下侧为负");
    }

    /// 圆柱：侧面/端盖/内部语义。
    #[test]
    fn cylinder_known_values() {
        let c = Cylinder { center: Vec3::splat(0.0), radius: 1.0, half_height: 2.0 };
        assert!((c.dist(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6, "侧向外");
        assert!((c.dist(Vec3::new(0.0, 3.0, 0.0)) - 1.0).abs() < 1e-6, "端盖外");
        // 侧+端对角外 (2,3,0)：最近点 (1,2,0)，距离 √2。
        assert!((c.dist(Vec3::new(2.0, 3.0, 0.0)) - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((c.dist(Vec3::new(0.0, 0.0, 0.0)) + 1.0).abs() < 1e-6, "轴心 -R");
    }

    /// 圆环：主环半径与管半径语义。
    #[test]
    fn torus_known_values() {
        let t = Torus { center: Vec3::splat(0.0), major: 2.0, minor: 0.5 };
        assert!((t.dist(Vec3::new(2.0, 0.0, 0.0)) + 0.5).abs() < 1e-6, "环心管内");
        assert!(t.dist(Vec3::new(2.5, 0.0, 0.0)).abs() < 1e-6, "管表面零");
        assert!((t.dist(Vec3::new(0.0, 0.0, 0.0)) - 1.5).abs() < 1e-6, "环心空洞区");
    }

    /// 胶囊：线段扫掠语义与退化线段。
    #[test]
    fn capsule_known_values() {
        let cap = Capsule { a: Vec3::new(-1.0, 0.0, 0.0), b: Vec3::new(1.0, 0.0, 0.0), radius: 0.5 };
        assert!((cap.dist(Vec3::new(0.0, 0.0, 0.0)) + 0.5).abs() < 1e-6, "线段中点 -r");
        // 端球外：距端点 a 距离 - r。
        assert!((cap.dist(Vec3::new(-2.0, 0.0, 0.0)) - 0.5).abs() < 1e-6);
        // 退化线段（a == b）应为球。
        let deg = Capsule { a: Vec3::splat(1.0), b: Vec3::splat(1.0), radius: 1.0 };
        assert!((deg.dist(Vec3::splat(1.0)) + 1.0).abs() < 1e-6);
    }
}
