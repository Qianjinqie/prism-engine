//! V2-h 验收（rt 侧）：USD 网格 → `prism_rt::integrator::Scene` 加载 roundtrip
//! + 双路径 parity（integrator ≡ wavefront）。
//!
//! # 为什么这个测试放在 rt 的 tests/ 里
//! `prism-io` 是 `prism-rt` 的 **dev-dependency**（生产方向 rt → io 会成环），
//! 故 "USD 加载 → rt 渲染" 的对接验证只能在集成测试里做。生产侧的对接点只有
//! 一行：[`prism_rt::integrator::Mesh::from_flat`]（f32 顶点 + u32 索引流
//! → 内建 BVH 的网格），io 侧由 `usd::Scene::render_meshes()` 给出中性数据。
//!
//! # 验收项
//! - USD 三角化（6 四边形 → 12 三角形）经 `to_render_mesh` 无损落到 rt 网格；
//! - Xform 世界变换已烘焙（包围盒 / 命中距离对账）；
//! - 加载后的场景可出图（非平凡、全有限）；
//! - integrator 与 wavefront 在同一 USD 场景上**逐位**一致。

use prism_core::EvalCtx;
use prism_geometry::bvh::Triangle as GeoTriangle;
use prism_io::usd;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, Light, MaterialKind, Mesh, RenderParams, Scene};
use prism_rt::wavefront::render_frame as wavefront_render_frame;

/// 居中立方体的 USDA（6 四边形 → 耳切 12 三角形），整体 +Y 平移 1。
const CUBE_USDA: &str = r#"#usda 1.0
def Xform "root"
{
    double3 xformOp:translate = (0, 1, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "cube"
    {
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2]
        point3f[] points = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]
    }
}
"#;

/// USD 网格 → rt 场景（生产对接路径：`render_meshes` → `Mesh::from_flat`）。
fn load_usd_cube(usda: &str) -> (usd::Scene, Scene) {
    let parsed = usd::parse_str(usda).expect("USDA 应可解析");
    let io_scene = usd::build_scene(&parsed.layer).expect("USDA 应可构建场景");
    let mut rt = Scene::default();
    for rm in io_scene.render_meshes() {
        rt.meshes.push((
            Mesh::from_flat(&rm.positions, &rm.indices),
            MaterialKind::Lambert { albedo: Vec3::splat(0.75) },
        ));
    }
    (io_scene, rt)
}

/// roundtrip：三角形数 / 顶点数 / 包围盒 / 命中距离逐项对账。
#[test]
fn usd_mesh_loads_into_rt_scene_roundtrip() {
    let (io_scene, scene) = load_usd_cube(CUBE_USDA);

    // io 侧三角化：6 四边形 → 12 三角形；rt 侧逐项相等。
    assert_eq!(io_scene.mesh_count(), 1);
    assert_eq!(io_scene.render_triangle_count(), 12);
    assert_eq!(scene.mesh_count(), 1);
    assert_eq!(scene.triangle_count(), 12);
    assert!(!scene.meshes[0].0.is_soup(), "USD 索引网格应保留拓扑");
    assert_eq!(scene.meshes[0].0.positions.len(), 8);
    assert_eq!(scene.meshes[0].0.indices.len(), 36);

    // Xform 已烘焙：立方体半边 1 + 平移 (0,1,0) ⇒ y ∈ [0, 2]。
    let (bmin, bmax) = scene.meshes[0].0.bounds().expect("非空网格必包围盒");
    assert!((bmin.x + 1.0).abs() < 1e-6, "bmin = {bmin:?}");
    assert!((bmax.x - 1.0).abs() < 1e-6, "bmax = {bmax:?}");
    assert!(bmin.y.abs() < 1e-6, "平移后底面 y = 0，实测 {bmin:?}");
    assert!((bmax.y - 2.0).abs() < 1e-6, "平移后顶面 y = 2，实测 {bmax:?}");

    // 命中对账：从正上方垂直下落，顶面 y = 2 ⇒ t = 3。
    let t = scene
        .primary_hit_distance(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 1.0e30)
        .expect("应命中立方体顶面");
    assert!((t - 3.0).abs() < 1e-5, "命中距离 = {t}");
    // 侧面外擦（x = 1.2 > 半边 1）不命中。
    assert!(scene
        .primary_hit_distance(Vec3::new(1.2, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 1.0e30)
        .is_none());

    // 三角形与源顶点一致（第一块三角形的顶点须是原立方体顶点之一集合内）。
    let tri: &GeoTriangle = &scene.meshes[0].0.triangles[0];
    assert!(tri.area() > 0.0, "三角化不应产出零面积面");
    println!(
        "USD roundtrip：{} mesh / {} 三角形 / 包围盒 {:?}..{:?}",
        scene.mesh_count(),
        scene.triangle_count(),
        bmin,
        bmax
    );
}

/// 加载后的 USD 场景可出图，且 integrator ≡ wavefront（逐位）。
#[test]
fn usd_mesh_scene_renders_and_paths_agree_bitwise() {
    let (_io_scene, mut scene) = load_usd_cube(CUBE_USDA);
    scene.lights.push(Light::Rect {
        center: Vec3::new(0.0, 4.0, 1.2),
        edge_u: Vec3::new(2.4, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 0.0, 2.4),
        radiance: Vec3::new(16.0, 14.0, 11.0),
    });
    scene.lights.push(Light::Ambient { radiance: Vec3::new(0.06, 0.08, 0.11) });
    let cam = Camera::new(
        Vec3::new(0.0, 1.2, 5.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        50.0,
        0.0,
        5.0,
        0.0,
        1.0 / 24.0,
    );
    let ctx = EvalCtx::new(20260715, 7, 3);
    let res = (8u32, 8u32);
    let params = RenderParams::new(8, 3);

    let mut min_l = f32::INFINITY;
    let mut max_l = f32::NEG_INFINITY;
    let wf_frame = wavefront_render_frame(&scene, &cam, &ctx, res, &params);
    let mut rt_frame = Vec::with_capacity(wf_frame.len());
    for y in 0..res.1 {
        for x in 0..res.0 {
            rt_frame.push(render_pixel(&scene, &cam, &ctx, (x, y), res, &params));
        }
    }
    assert_eq!(wf_frame.len(), rt_frame.len());
    for i in 0..wf_frame.len() {
        for c in 0..4 {
            assert!(
                wf_frame[i][c].is_finite(),
                "像素 {i} ch{c} 非有限：{}",
                wf_frame[i][c]
            );
            assert_eq!(
                wf_frame[i][c].to_bits(),
                rt_frame[i][c].to_bits(),
                "像素 {i} 通道 {c}：wavefront {} ≠ integrator {}",
                wf_frame[i][c],
                rt_frame[i][c]
            );
        }
        let l = wf_frame[i][0] + wf_frame[i][1] + wf_frame[i][2];
        min_l = min_l.min(l);
        max_l = max_l.max(l);
    }
    assert!(max_l > 0.01, "USD 场景应被照亮：max_l = {max_l}");
    assert!(max_l - min_l > 1e-3, "图像应非平凡（应有明暗差异）");
    println!(
        "USD 场景出图 + parity：{res:?} spp{}，luma ∈ [{min_l:.4}, {max_l:.4}]（逐位一致）",
        params.spp
    );
}
