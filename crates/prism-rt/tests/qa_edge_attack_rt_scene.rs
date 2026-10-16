//! prism-rt depth AOV 攻击测试。
//!
//! 面：crates/prism-rt/src/depth.rs。找真 Bug，非复读既有断言。

use prism_core::{DepthMap, EvalCtx};
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::depth::{render_depth, DepthParams};
use prism_rt::integrator::{Aabb, MaterialKind, Scene, Sphere};

fn ball_scene() -> Scene {
    Scene {
        spheres: vec![(
            Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 0.5 },
            MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
        )],
        boxes: Vec::<(Aabb, MaterialKind)>::new(),
        lights: Vec::new(),
        medium: None,
        paraboloids: vec![],
        planes: vec![],
    }
}

fn cam() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        0.0,
        2.0,
        0.0,
        0.0,
    )
}

/// 缺陷面：深度渲染逐位确定（同种子同帧两遍一致）。
#[test]
fn rt_depth_bitwise_deterministic() {
    let scene = ball_scene();
    let ctx = EvalCtx::new(7, 1, 0);
    let a = render_depth(&scene, &cam(), &ctx, 16, 12, DepthParams::default());
    let b = render_depth(&scene, &cam(), &ctx, 16, 12, DepthParams::default());
    assert_eq!(a.depth, b.depth, "深度必须逐位确定");
}

/// 缺陷面：不同种子 → 不同抖动 → 可能存在差异，但同种子恒等（确定性地基）。
#[test]
fn rt_depth_same_seed_stable() {
    let scene = ball_scene();
    let ctx = EvalCtx::new(42, 1, 0);
    let a = render_depth(&scene, &cam(), &ctx, 8, 8, DepthParams::default());
    let b = render_depth(&scene, &cam(), &ctx, 8, 8, DepthParams::default());
    assert_eq!(a.depth, b.depth, "同种子必须稳定");
}

/// 缺陷面：空场景（无物体）→ 全无命中（-1），AOV 应全 1.0。
#[test]
fn rt_depth_empty_scene_all_miss() {
    let scene = Scene {
        spheres: Vec::new(),
        boxes: Vec::new(),
        lights: Vec::new(),
        medium: None,
        paraboloids: vec![],
        planes: vec![],
    };
    let ctx = EvalCtx::new(1, 1, 0);
    let dm = render_depth(&scene, &cam(), &ctx, 4, 4, DepthParams::default());
    for d in &dm.depth {
        assert!(*d < 0.0, "空场景应无命中（-1），得 {d}");
    }
    let aov = dm.to_aov(false);
    for v in &aov {
        assert_eq!(*v, 1.0, "无命中 AOV 应全 1.0");
    }
}

/// 缺陷面：AOV 值域契约（始终 [0,1]）。
#[test]
fn rt_depth_aov_in_unit_interval() {
    let scene = ball_scene();
    let ctx = EvalCtx::new(3, 1, 0);
    let dm = render_depth(&scene, &cam(), &ctx, 12, 12, DepthParams::default());
    let aov = dm.to_aov(false);
    assert_eq!(aov.len(), 144);
    for v in &aov {
        assert!(*v >= 0.0 && *v <= 1.0, "AOV 越界 {v}");
    }
}

/// 缺陷面：near/far 参数影响深度图边界（far 拉近 → 命中距离仍为几何距离，
/// 超过 far 的命中被截断为无命中）。
#[test]
fn rt_depth_far_plane_clips() {
    let scene = ball_scene();
    let ctx = EvalCtx::new(3, 1, 0);
    // far=1.0 < 球前表面 1.5 → 应全部无命中。
    let dm = render_depth(&scene, &cam(), &ctx, 4, 4, DepthParams { near: 0.1, far: 1.0 });
    for d in &dm.depth {
        assert!(*d < 0.0, "far 截断后应无命中，得 {d}");
    }
}

/// 缺陷面：退化尺寸（1×1）不 panic。
#[test]
fn rt_depth_one_pixel() {
    let scene = ball_scene();
    let ctx = EvalCtx::new(1, 1, 0);
    let dm = render_depth(&scene, &cam(), &ctx, 1, 1, DepthParams::default());
    assert_eq!(dm.depth.len(), 1);
    assert_eq!(dm.width, 1);
    assert_eq!(dm.height, 1);
}

/// 保留 DepthMap 引用（公有契约可编译）。
#[test]
fn rt_depth_map_contract() {
    let dm = DepthMap::new(2, 2, 1.0, 10.0);
    assert_eq!(dm.depth.len(), 4);
    assert!(dm.depth.iter().all(|d| *d < 0.0), "构造为全无命中");
}
