//! `prism-math` — minimal 3D linear-algebra primitives for the Prism engine.
//!
//! Zero external dependencies. Only the Rust standard library is used.
//! All math is performed with `f32`. Matrices are stored in **column-major**
//! order: the flat `[f32; 16]` array is laid out as four consecutive columns,
//! i.e. `self.0[col * 4 + row]` is element `(row, col)`.
//!
//! # 色彩与 HDR 子模块（C2）
//! - [`det_pow`] — D-29 确定性幂/指/对数件（无 libm）；
//! - [`pq`] — PQ（SMPTE ST 2084）HDR 传输函数；
//! - [`hlg`] — HLG（ITU-R BT.2100）传输函数；
//! - [`colorspace`] — 线性 sRGB ↔ ACEScg（AP1）矩阵与色彩空间枚举；
//! - [`aces`] — ACES 输出变换（RRT+ODT 拟合两档）；
//! - [`ocio`] — V2.5-B1 自研 `config.ocio` 解析（零第三方依赖、结构化报错）；
//! - [`ocio_lut`] — config → Q24 定点 1D/3D LUT 生成与整数求值（零 libm）。

use std::ops::{Add, Div, Mul, Neg, Sub};

pub mod aces;
pub mod colorspace;
pub mod colorspace_registry;
pub mod det_pow;
pub mod det_trig;
pub mod hlg;
pub mod ocio;
pub mod ocio_lut;
pub mod pq;

/// A three-dimensional vector with `x`, `y` and `z` components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Constructs a vector from its three components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Constructs a vector whose three components are all equal to `v`.
    #[inline]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Dot (scalar) product with `o`.
    #[inline]
    pub fn dot(&self, o: &Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product with `o`.
    #[inline]
    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    /// Euclidean length of the vector.
    #[inline]
    pub fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns a unit-length vector pointing in the same direction.
    ///
    /// If the length is below `1e-8` (numerically degenerate) a zero vector is
    /// returned to avoid division by (nearly) zero.
    #[inline]
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len < 1e-8 {
            Vec3::splat(0.0)
        } else {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        }
    }

    /// Linear interpolation between `self` and `o` by factor `t`.
    ///
    /// `t == 0` yields `self`, `t == 1` yields `o`.
    #[inline]
    pub fn lerp(&self, o: &Vec3, t: f32) -> Vec3 {
        Vec3::new(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    /// Scales every component by the scalar `s`.
    #[inline]
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    /// Scales every component by the reciprocal of `s`.
    #[inline]
    fn div(self, s: f32) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

/// A 4x4 matrix stored in column-major order.
///
/// The single field is the flat array `m` where `m[col * 4 + row]` is the
/// element at row `row`, column `col` (0-indexed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    /// The identity matrix.
    #[inline]
    pub fn identity() -> Self {
        Mat4([
            1.0, 0.0, 0.0, 0.0, // column 0
            0.0, 1.0, 0.0, 0.0, // column 1
            0.0, 0.0, 1.0, 0.0, // column 2
            0.0, 0.0, 0.0, 1.0, // column 3
        ])
    }

    /// A translation matrix that moves points by `v`.
    #[inline]
    pub fn translation(v: Vec3) -> Self {
        Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            v.x, v.y, v.z, 1.0,
        ])
    }

    /// A rotation of `a` radians about the Y axis.
    #[inline]
    pub fn rotation_y(a: f32) -> Self {
        let (c, s) = crate::det_trig::cos_sin(a);
        Mat4([
            c, 0.0, -s, 0.0, // column 0
            0.0, 1.0, 0.0, 0.0, // column 1
            s, 0.0, c, 0.0, // column 2
            0.0, 0.0, 0.0, 1.0, // column 3
        ])
    }

    /// A right-handed look-at view matrix.
    ///
    /// The camera looks from `eye` toward `center`; the view-space +Z axis
    /// points from `center` back to `eye` (i.e. `z = normalize(eye - center)`).
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        // Forward (toward the scene), right, and recomputed up.
        let f = (center - eye).normalize();
        let s = f.cross(&up).normalize();
        let u = s.cross(&f);

        let dot_s = s.dot(&eye);
        let dot_u = u.dot(&eye);
        let dot_f = f.dot(&eye);

        Mat4([
            s.x, s.y, s.z, -dot_s, // column 0
            u.x, u.y, u.z, -dot_u, // column 1
            -f.x, -f.y, -f.z, dot_f, // column 2
            0.0, 0.0, 0.0, 1.0, // column 3
        ])
    }

    /// A standard OpenGL-style perspective projection matrix.
    ///
    /// Depth is mapped to the clip-space range `[-1, 1]`. `fovy` is the vertical
    /// field-of-view in radians; `near` and `far` must be positive.
    pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / crate::det_trig::tan(fovy / 2.0);
        let nf = 1.0 / (near - far);
        Mat4([
            f / aspect, 0.0, 0.0, 0.0, // column 0
            0.0, f, 0.0, 0.0, // column 1
            0.0, 0.0, (far + near) * nf, -1.0, // column 2
            0.0, 0.0, (2.0 * far * near) * nf, 0.0, // column 3
        ])
    }

    /// Matrix product `self * o` (treating both as column-major).
    pub fn mul(&self, o: &Mat4) -> Mat4 {
        let a = &self.0;
        let b = &o.0;
        let mut m = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    // a[row, k] = a[k*4+row]; b[k, col] = b[col*4+k]
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                m[col * 4 + row] = sum;
            }
        }
        Mat4(m)
    }

    /// Transforms a point `v` (implicit `w = 1`) by this matrix, then performs
    /// the perspective divide by the resulting `w`.
    pub fn transform_point(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        let x = m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12];
        let y = m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13];
        let z = m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14];
        let w = m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15];
        if w.abs() < 1e-8 {
            Vec3::splat(0.0)
        } else {
            Vec3::new(x / w, y / w, z / w)
        }
    }
}

/// Clamps `x` into the inclusive range `[lo, hi]`.
#[inline]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(&b), 32.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(b / 2.0, Vec3::new(2.0, 2.5, 3.0));
    }

    #[test]
    fn vec3_cross_and_normalize() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(&y), Vec3::new(0.0, 0.0, 1.0));
        let n = Vec3::new(0.0, 3.0, 4.0).normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        // Degenerate normalize returns zero vector.
        assert_eq!(Vec3::splat(0.0).normalize(), Vec3::splat(0.0));
    }

    #[test]
    fn mat4_identity_and_translation() {
        let t = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.transform_point(Vec3::splat(0.0)), Vec3::new(1.0, 2.0, 3.0));
        let i = Mat4::identity();
        assert_eq!(i.transform_point(Vec3::new(5.0, 6.0, 7.0)), Vec3::new(5.0, 6.0, 7.0));
    }

    #[test]
    fn mat4_mul_associativity_with_identity() {
        let t = Mat4::translation(Vec3::new(1.0, 0.0, 0.0));
        let i = Mat4::identity();
        assert_eq!(t.mul(&i), t);
        assert_eq!(i.mul(&t), t);
    }

    #[test]
    fn look_at_basis() {
        // Looking down -Z, up = +Y. Camera at origin.
        let v = Mat4::look_at(
            Vec3::splat(0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // The origin should map to the origin in view space.
        assert_eq!(v.transform_point(Vec3::splat(0.0)), Vec3::splat(0.0));
    }

    #[test]
    fn vec3_dot_and_cross_known_values() {
        // 标准正交基叉积：(1,0,0) × (0,1,0) = (0,0,1)。
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(x.cross(&y), z);
        // 叉积反交换律：y × x = -z。
        assert_eq!(y.cross(&x), -z);
        // 平行向量叉积为零向量；正交向量点积为零。
        assert_eq!(x.cross(&x), Vec3::splat(0.0));
        assert_eq!(x.dot(&y), 0.0);
        // 数值点积：(1,2,3)·(4,5,6) = 4 + 10 + 18 = 32。
        assert_eq!(
            Vec3::new(1.0, 2.0, 3.0).dot(&Vec3::new(4.0, 5.0, 6.0)),
            32.0
        );
        // 数值叉积：(2,3,4)×(5,6,7) = (-3, 6, -3)。
        let c = Vec3::new(2.0, 3.0, 4.0).cross(&Vec3::new(5.0, 6.0, 7.0));
        assert_eq!(c, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn vec3_normalize_direction_and_length() {
        // (3,0,-4) 归一化后长度≈1 且方向保持：(0.6, 0, -0.8)。
        let n = Vec3::new(3.0, 0.0, -4.0).normalize();
        assert!((n.length() - 1.0).abs() < 1e-6, "归一化后长度应为 1");
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!(n.y.abs() < 1e-6);
        assert!((n.z + 0.8).abs() < 1e-6);
    }

    #[test]
    fn mat4_rotation_y_quarter_turn() {
        // 右手系绕 +Y 旋转 90°：+X → -Z，+Z → +X，+Y 保持不动。
        let r = Mat4::rotation_y(std::f32::consts::FRAC_PI_2);
        let got_x = r.transform_point(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (got_x - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5,
            "+X 旋转 90° 后应到达 -Z，实际 {:?}",
            got_x
        );
        let got_z = r.transform_point(Vec3::new(0.0, 0.0, 1.0));
        assert!(
            (got_z - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5,
            "+Z 旋转 90° 后应到达 +X，实际 {:?}",
            got_z
        );
        let got_y = r.transform_point(Vec3::new(0.0, 1.0, 0.0));
        assert!(
            (got_y - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5,
            "旋转轴 +Y 应保持不动，实际 {:?}",
            got_y
        );
    }

    #[test]
    fn mat4_identity_leaves_arbitrary_matrix_invariant() {
        // 用一个非对称、非平凡的矩阵验证恒等矩阵左右乘都不改变它。
        let m = Mat4([
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ]);
        let i = Mat4::identity();
        assert_eq!(i.mul(&m), m, "identity * M 应等于 M");
        assert_eq!(m.mul(&i), m, "M * identity 应等于 M");
    }

    #[test]
    fn clamp_boundary_values() {
        assert_eq!(clamp(-1.0, 0.0, 1.0), 0.0, "低于下界应取下界");
        assert_eq!(clamp(2.0, 0.0, 1.0), 1.0, "高于上界应取上界");
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5, "区间内应原样返回");
        assert_eq!(clamp(0.0, 0.0, 1.0), 0.0, "恰在下界");
        assert_eq!(clamp(1.0, 0.0, 1.0), 1.0, "恰在上界");
    }
}
