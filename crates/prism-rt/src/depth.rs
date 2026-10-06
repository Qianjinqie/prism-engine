//! 深度 AOV 输出（C1）：路径追踪主射线命中距离 → 逐像素深度图。
//!
//! 语义权威：《强大版路线图》§3.4（"深度缓冲保留（本域的地基）：Stage3D 输出
//! 从 RGBA8 底图升级为 RGBA16F + 线性深度 f32 通道——路径追踪顺手产出（命中
//! 距离），零额外成本"）与 C1 行（"Stage3D 输出附深度"）。
//!
//! # 确定性
//!
//! 主射线方向用与颜色渲染**同一** `SampleStream`（D-28 像素域，Sobol+Owen）——
//! 深度与颜色对同一像素取同一相机光线（抖动一致），且深度是相机的纯函数：
//! 同种子同像素同帧必得同深度（无随机、无隐藏状态）。

use prism_core::{DepthMap, EvalCtx};

use crate::camera::Camera;
use crate::integrator::Scene;
use crate::sampler::SampleStream;

/// 深度 AOV 渲染参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthParams {
    /// 近平面（世界单位；> 0）。
    pub near: f32,
    /// 远平面（世界单位；> near）。
    pub far: f32,
}

impl Default for DepthParams {
    fn default() -> Self {
        Self { near: 0.1, far: 100.0 }
    }
}

/// 渲染一整帧的深度图（逐像素主射线命中距离）。
///
/// 无命中像素在 [`DepthMap`] 中记为 -1（背景/天空）。相机抖动经 SampleStream
/// （样本 0）与颜色渲染一致。
pub fn render_depth(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    width: u32,
    height: u32,
    params: DepthParams,
) -> DepthMap {
    let mut dm = DepthMap::new(width, height, params.near, params.far);
    for y in 0..height {
        for x in 0..width {
            let stream = SampleStream::new(ctx, (x, y), 0);
            let ray = cam.cast_ray(&stream, (width, height));
            let d = scene.primary_hit_distance(ray.origin, ray.dir, params.far);
            dm.set(x, y, d.unwrap_or(-1.0));
        }
    }
    dm
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Aabb, MaterialKind, Sphere};
    use prism_core::SampleIndex;
    use prism_math::Vec3;

    fn test_ctx() -> EvalCtx {
        EvalCtx::new(7, 1, 0)
    }

    /// 最小场景：一个位于 (0,0,2)、半径 0.5 的球（相机在原点看向 +Z）。
    fn ball_scene() -> Scene {
        Scene {
            spheres: vec![(
                Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 0.5 },
                MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
            )],
            boxes: Vec::<(Aabb, MaterialKind)>::new(),
            lights: Vec::new(),
            medium: None,
        }
    }

    #[test]
    fn depth_map_records_hit_distance() {
        let scene = ball_scene();
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            0.0,
            2.0,
            0.0,
            0.0,
        );
        let ctx = test_ctx();
        let dm = render_depth(&scene, &cam, &ctx, 4, 4, DepthParams::default());
        // 中心像素（看向球）应有命中，距离接近 1.5（球前表面 z = 2 − 0.5）。
        let center = dm.get(2, 2);
        assert!(center > 0.0, "中心像素应有命中，实际 {center}");
        assert!((center - 1.5).abs() < 0.2, "中心命中距离应 ≈ 1.5，实际 {center}");
    }

    #[test]
    fn depth_is_deterministic_across_runs() {
        let scene = ball_scene();
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            0.0,
            2.0,
            0.0,
            0.0,
        );
        let ctx = test_ctx();
        let a = render_depth(&scene, &cam, &ctx, 8, 8, DepthParams::default());
        let b = render_depth(&scene, &cam, &ctx, 8, 8, DepthParams::default());
        assert_eq!(a.depth, b.depth, "同输入深度必须逐位一致");
    }

    #[test]
    fn depth_aov_from_rt_is_in_unit_interval() {
        let scene = ball_scene();
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            0.0,
            2.0,
            0.0,
            0.0,
        );
        let ctx = test_ctx();
        let dm = render_depth(&scene, &cam, &ctx, 8, 8, DepthParams::default());
        let aov = dm.to_aov(false);
        assert_eq!(aov.len(), 64);
        for v in &aov {
            assert!(*v >= 0.0 && *v <= 1.0, "AOV 必须在 [0,1]，实际 {v}");
        }
        // 无样本派生依赖（确定性）：SampleIndex 未直接使用但保持接口一致。
        let _ = SampleIndex { frame: 0, pixel: (0, 0), sample: 0, dimension: 0, node_id: 0 };
    }
}
