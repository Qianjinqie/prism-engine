//! 透视相机 + 薄透镜景深 + 运动模糊。
//!
//! # 语义
//! - 构造期一次性三角计算：焦距标度 = `sampler::tan(fov_rad/2)`（自研 tan
//!   从简声明见 sampler 模块文档；fov 单位为度——宪章 C17 行业豁免，构造
//!   断言 (0°,180°) 开区间）。
//! - 正交基用叉积构造（格锚纪律：禁浮点三角）；运行期抽样路径零 libm
//!   超越函数——圆盘同心映射 + 一次 sqrt，角度项走 sampler 的 det 三角件
//!   （D-29 合规路径，非"无三角计算"）。
//! - [`Camera::cast_ray`] 纯函数：消费 SampleStream 维度槽 0,1（光圈圆盘）、
//!   2（快门时间）、3,4（像素内抖动）——维度预算表见 integrator 模块文档。
//!
//! # 薄透镜模型
//! 先在焦平面锁定目标点 P（与光圈抽样无关——同一样本的针孔光线与 DOF 光线
//! 汇聚于同一焦平面点，这是薄透镜的不变量；两条光线的舍入路径不同，测试按
//! 舍入内一致断言 <1e-4），再在光圈圆盘上取视点：
//! `dir = normalize(P − O_lens)`。运动模糊：快门时间在 [open, close] 内抽样，
//! B1 场景静态，time 仅为下游快照接口预留（区间内保证）。

use std::f32::consts::PI;

use prism_math::Vec3;

use crate::sampler::{concentric_disk, tan, SampleStream};

/// 相机段维度槽（弹射段预算表见 integrator 模块文档）。
pub const DIM_LENS: u16 = 0; // 槽 0,1：光圈圆盘
/// 快门时间维度槽。
pub const DIM_SHUTTER: u16 = 2; // 槽 2
/// 像素内抖动维度槽。
pub const DIM_PIXEL: u16 = 3; // 槽 3,4

/// 光线（dir 单位化；time ∈ [shutter_open, shutter_close]）。
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// 起点。
    pub origin: Vec3,
    /// 方向（单位化）。
    pub dir: Vec3,
    /// 快门时刻（秒）。
    pub time: f32,
}

impl Ray {
    /// 线上点 origin + t·dir（t 为几何距离）。
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// 透视相机（薄透镜 + 快门）。
#[derive(Debug, Clone)]
pub struct Camera {
    /// 视点。
    pub eye: Vec3,
    /// 前向单位向量（指向场景）。
    pub forward: Vec3,
    /// 右向单位向量。
    pub right: Vec3,
    /// 上向单位向量。
    pub up: Vec3,
    /// 竖直半视角正切 tan(fov/2)（构造期一次 `sampler::tan`）。
    pub tan_half_fov: f32,
    /// 光圈半径（world 单位；0 = 针孔快路径）。
    pub aperture_radius: f32,
    /// 对焦距离（world 单位，沿 forward 轴度量）。
    pub focus_distance: f32,
    /// 快门开（秒）。
    pub shutter_open: f32,
    /// 快门关（秒）。
    pub shutter_close: f32,
}

impl Camera {
    /// 构造相机。fov 为竖直视场角（度，C17 豁免）。
    ///
    /// # Panics
    /// fov 越界、光圈为负、对焦距离非正、快门区间为负、forward 与 world_up
    /// 共线（正交基退化）时 panic；信息含建议段（宪章 C19）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eye: Vec3,
        look_at: Vec3,
        world_up: Vec3,
        fov_deg: f32,
        aperture_radius: f32,
        focus_distance: f32,
        shutter_open: f32,
        shutter_close: f32,
    ) -> Self {
        assert!(
            fov_deg > 0.0 && fov_deg < 180.0,
            "fov_deg 必须在 (0,180) 开区间（C17/C18），收到 {fov_deg}；建议：常用竖直 fov 30–90 度，超广角 ≤ 120 度，接近 180 会数值退化"
        );
        assert!(
            aperture_radius >= 0.0,
            "光圈半径不得为负，收到 {aperture_radius}；建议：景深从小光圈（0.05–0.5 世界单位）起步，0 表示针孔（无景深）"
        );
        assert!(
            focus_distance > 0.0,
            "对焦距离必须为正，收到 {focus_distance}；建议：取视点到对焦主体的实际距离（世界单位），沿 forward 轴度量"
        );
        assert!(
            shutter_close >= shutter_open,
            "快门关时刻不得早于开时刻（{shutter_open} > {shutter_close}）；建议：静态场景用零长快门（open == close），运动模糊按位移量设定区间长度"
        );
        let forward = (look_at - eye).normalize();
        let side = forward.cross(&world_up);
        assert!(
            side.length() > 1e-6,
            "forward 与 world_up 共线，正交基退化；请更换 world_up（如 (0,0,1)）"
        );
        let right = side.normalize();
        let up = right.cross(&forward);
        let fov_rad = fov_deg * PI / 180.0;
        Camera {
            eye,
            forward,
            right,
            up,
            tan_half_fov: tan(fov_rad * 0.5),
            aperture_radius,
            focus_distance,
            shutter_open,
            shutter_close,
        }
    }

    /// 亚像素坐标 (px, py) 穿过近像平面的方向（未单位化，forward 分量恒 1）。
    /// fov 锚定竖直方向，水平按宽高比展开；NDC：x 向右、y 向上（图像行号向下）。
    pub fn pixel_dir(&self, px: f32, py: f32, resolution: (u32, u32)) -> Vec3 {
        let w = resolution.0 as f32;
        let h = resolution.1 as f32;
        let u = (px / w) * 2.0 - 1.0;
        let v = 1.0 - (py / h) * 2.0;
        let aspect = w / h;
        self.forward
            + self.right * (u * self.tan_half_fov * aspect)
            + self.up * (v * self.tan_half_fov)
    }

    /// 投射一条相机光线（纯函数）：像素内抖动（槽 3,4）→ 焦平面目标点 →
    /// 光圈圆盘视点（槽 0,1）→ 快门时间（槽 2）。
    pub fn cast_ray(&self, stream: &SampleStream<'_>, resolution: (u32, u32)) -> Ray {
        let (jx, jy) = stream.sample_2d(DIM_PIXEL);
        let d = self.pixel_dir(
            stream.pixel().0 as f32 + jx,
            stream.pixel().1 as f32 + jy,
            resolution,
        );
        // 零长快门（静态场景的常态）：`time = open + u·(close−open) = open + u·0 = open`
        // 与 `u` 无关 ⇒ 这一抽可以不抽，结果**逐位不变**。
        //
        // 为什么不是"改画质"：`ray.time` 在本仓除被拷贝外没有任何消费者（静态
        // 场景无运动模糊），故跳过它连 ray 的位模式都不变。Sobol 维度是**绝对
        // 寻址**（`dim_value(seed, dim, index)`），少抽一维不会让后续维度错位，
        // 所以 NEE / BSDF 的抽样序列不受影响。
        //
        // 唯一位差异是病态输入 `shutter_open == -0.0`：改前算得 `-0.0 + 0.0 = +0.0`，
        // 改后取 `-0.0`。两者数值相等、且 `time` 无消费者 ⇒ 像素输出仍逐字节相同。
        let time: f32 = if self.shutter_close == self.shutter_open {
            self.shutter_open
        } else {
            self.shutter_open
                + stream.sample_1d(DIM_SHUTTER) * (self.shutter_close - self.shutter_open)
        };
        if self.aperture_radius == 0.0 {
            // 针孔快路径：视点 = eye。
            return Ray { origin: self.eye, dir: d.normalize(), time };
        }
        // 薄透镜：焦平面目标点与光圈抽样无关（薄透镜不变量）。
        let d_unit = d.normalize();
        let focal = self.eye + d_unit * (self.focus_distance / d_unit.dot(&self.forward));
        let (du, dv) = stream.sample_2d(DIM_LENS);
        let (lx, ly) = concentric_disk(du, dv);
        let origin = self.eye
            + self.right * (lx * self.aperture_radius)
            + self.up * (ly * self.aperture_radius);
        Ray { origin, dir: (focal - origin).normalize(), time }
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 4：相机几何断言）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::EvalCtx;

    fn cam(aperture: f32) -> Camera {
        Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            aperture,
            3.0,
            0.0,
            0.5,
        )
    }

    fn stream<'a>(ctx: &'a EvalCtx, k: u32) -> SampleStream<'a> {
        SampleStream::new(ctx, (4, 4), k)
    }

    #[test]
    fn center_pixel_dir_is_optical_axis() {
        let c = cam(0.0);
        // (9,9) 中心像素的几何中心 → NDC (0,0) → 方向 = forward。
        let d = c.pixel_dir(4.5, 4.5, (9, 9));
        assert!((d.x - c.forward.x).abs() < 1e-6);
        assert!((d.y - c.forward.y).abs() < 1e-6);
        assert!((d.z - c.forward.z).abs() < 1e-6);
    }

    #[test]
    fn pinhole_center_ray_stays_near_axis() {
        let c = cam(0.0);
        let ctx = EvalCtx::new(7, 7, 3);
        // 中心像素抖动限制在半个像素内 → 与光轴夹角极小；针孔视点 = eye。
        for k in 0..64u32 {
            let r = c.cast_ray(&stream(&ctx, k), (9, 9));
            let cos = r.dir.dot(&c.forward);
            assert!(cos > 0.99, "中心像素光线偏离光轴过多：cos = {cos}");
            assert_eq!(r.origin.x.to_bits(), c.eye.x.to_bits());
            assert_eq!(r.origin.y.to_bits(), c.eye.y.to_bits());
            assert_eq!(r.origin.z.to_bits(), c.eye.z.to_bits());
        }
    }

    #[test]
    fn dof_origins_within_aperture_and_share_focal_point() {
        let c = cam(0.2);
        let pinhole = cam(0.0);
        let ctx = EvalCtx::new(7, 7, 3);
        let res = (9, 9);
        for k in 0..256u32 {
            let r = c.cast_ray(&stream(&ctx, k), res);
            // ① 视点在光圈圆盘内。
            let off = r.origin - c.eye;
            assert!(
                off.length() <= c.aperture_radius + 1e-6,
                "视点越出光圈：{}",
                off.length()
            );
            // ② 薄透镜不变量（精确）：DOF 光线与焦平面的交点 = 同一样本的
            // 针孔光线与焦平面的交点（抖动目标逐样本不同，必须逐样本配对）。
            let r0 = pinhole.cast_ray(&stream(&ctx, k), res);
            let t = (c.focus_distance - r.origin.dot(&c.forward)) / r.dir.dot(&c.forward);
            let p_dof = r.at(t);
            let t0 = (c.focus_distance - r0.origin.dot(&c.forward)) / r0.dir.dot(&c.forward);
            let p_pin = r0.at(t0);
            let err = (p_dof - p_pin).length();
            assert!(err < 1e-4, "DOF 与针孔焦平面目标不一致：err = {err}");
        }
    }

    #[test]
    fn shutter_time_within_interval_and_degenerate_is_exact() {
        let ctx = EvalCtx::new(7, 7, 3);
        let c = cam(0.0);
        for k in 0..128u32 {
            let r = c.cast_ray(&stream(&ctx, k), (9, 9));
            assert!(
                (0.0..=0.5).contains(&r.time),
                "快门时间越界：{}",
                r.time
            );
        }
        // 零长快门：time 精确等于开时刻（逐位）。
        let c0 = Camera::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            45.0,
            0.0,
            5.0,
            0.25,
            0.25,
        );
        let r = c0.cast_ray(&stream(&ctx, 3), (9, 9));
        assert_eq!(r.time.to_bits(), 0.25f32.to_bits());
    }

    #[test]
    fn cast_ray_is_deterministic() {
        let c = cam(0.15);
        let ctx = EvalCtx::new(7, 7, 3);
        let a = c.cast_ray(&stream(&ctx, 11), (9, 9));
        let b = c.cast_ray(&stream(&ctx, 11), (9, 9));
        assert_eq!(a.origin.x.to_bits(), b.origin.x.to_bits());
        assert_eq!(a.origin.y.to_bits(), b.origin.y.to_bits());
        assert_eq!(a.origin.z.to_bits(), b.origin.z.to_bits());
        assert_eq!(a.dir.x.to_bits(), b.dir.x.to_bits());
        assert_eq!(a.dir.z.to_bits(), b.dir.z.to_bits());
        assert_eq!(a.time.to_bits(), b.time.to_bits());
    }

    #[test]
    #[should_panic(expected = "fov_deg")]
    fn fov_bounds_enforced() {
        let _ = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            180.0,
            0.0,
            1.0,
            0.0,
            1.0,
        );
    }
}
