//! 2D 几何基元：点与仿射变换（平移/缩放/旋转的 2×3 矩阵）。
//!
//! 旋转在 y 向下的像素坐标里以标准矩阵给出；光栅全程只用 ±×÷。

/// 2D 点（像素或字体单位）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// 横坐标。
    pub x: f32,
    /// 纵坐标（光栅空间 y 向下；字体空间 y 向上，由变换翻转）。
    pub y: f32,
}

impl Point {
    /// 构造点。
    pub const fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }
}

/// 2D 仿射变换：`x' = a·x + c·y + tx`，`y' = b·x + d·y + ty`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// x' 对 x 的系数。
    pub a: f32,
    /// y' 对 x 的系数。
    pub b: f32,
    /// x' 对 y 的系数。
    pub c: f32,
    /// y' 对 y 的系数。
    pub d: f32,
    /// x 平移。
    pub tx: f32,
    /// y 平移。
    pub ty: f32,
}

impl Transform {
    /// 恒等变换。
    pub const IDENTITY: Transform = Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };

    /// 构造平移。
    pub const fn translation(tx: f32, ty: f32) -> Self {
        Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx, ty }
    }

    /// 构造均匀缩放（含绕原点；文本常用 `scale` + 平移组合）。
    pub const fn scale(s: f32) -> Self {
        Transform { a: s, b: 0.0, c: 0.0, d: s, tx: 0.0, ty: 0.0 }
    }

    /// 旋转（弧度，y 向下坐标系里正角为顺时针视觉方向）。
    ///
    /// sin/cos 由调用方以自研超越件（D-29 多项式）或常量提供——本 crate
    /// 不引入任何 libm 调用。
    pub const fn rotation(cos_a: f32, sin_a: f32) -> Self {
        Transform { a: cos_a, b: sin_a, c: -sin_a, d: cos_a, tx: 0.0, ty: 0.0 }
    }

    /// 复合：先应用 `self`，再应用 `outer`（即 `outer ∘ self`）。
    pub fn then(&self, outer: &Transform) -> Transform {
        Transform {
            a: outer.a * self.a + outer.c * self.b,
            b: outer.b * self.a + outer.d * self.b,
            c: outer.a * self.c + outer.c * self.d,
            d: outer.b * self.c + outer.d * self.d,
            tx: outer.a * self.tx + outer.c * self.ty + outer.tx,
            ty: outer.b * self.tx + outer.d * self.ty + outer.ty,
        }
    }

    /// 变换一个点。
    #[inline]
    pub fn apply(&self, p: Point) -> Point {
        Point::new(self.a * p.x + self.c * p.y + self.tx, self.b * p.x + self.d * p.y + self.ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_identity_and_scale() {
        let p = Point::new(3.0, -4.0);
        assert_eq!(Transform::IDENTITY.apply(p), p);
        let s = Transform::scale(2.0);
        let q = s.apply(p);
        assert_eq!((q.x, q.y), (6.0, -8.0));
    }

    #[test]
    fn transform_then_composes_matmul() {
        // 平移后缩放 vs 复合矩阵：同点同结果（位模式一致）。
        let t = Transform::translation(10.0, 20.0);
        let s = Transform::scale(2.0);
        let p = Point::new(1.5, 2.5);
        let direct = s.apply(t.apply(p));
        let composed = t.then(&s).apply(p);
        assert_eq!(direct.x.to_bits(), composed.x.to_bits());
        assert_eq!(direct.y.to_bits(), composed.y.to_bits());
    }

    #[test]
    fn rotation_quarter_turn_matches_constants() {
        // 90°：cos=0, sin=1 → (x,y) → (−y, x)。
        let r = Transform::rotation(0.0, 1.0);
        let q = r.apply(Point::new(2.0, 5.0));
        assert_eq!((q.x, q.y), (-5.0, 2.0));
    }
}
