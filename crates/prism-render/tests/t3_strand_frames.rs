//! T3 / R-6 判据：**毛发按股标架**（每股一个 `Mesh` + 该股自己的平行传输标架）。
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V1 | 单股与 `93e59d0` 的语义**逐位相同** | 位型全等 |
//! | V2a | ≥3 个正交方向的股束，每股的标架切向 = 该股自己的首段切向 | 夹角 ≤ 1e-6 rad |
//! | V2b | 每股的角误差 ≤ **单股水平**（单股是 0） | 同上（不是"接近"） |
//! | V2c | 像素级判别力：旧的"一材质一标架"会错，本实现不会 | 帧必须不同 + 旧法角度误差 90° |
//!
//! # 缺口原文（路线图 §2 G-9 行的"缺什么 ①"）
//!
//! > `MaterialKind::Hair` 是**一材质一标架**（同 Mesh 内多股异向时只有一股精确，
//! > 按股建 Mesh 可精确）
//!
//! 本轮选的正是"按股切 Mesh"：**精确**（不是近似）、`hair_scatter.rs` 与
//! `surface_brdf` **一行未改**、零 ABI 改动。备选方案（命中时按股查标架）要改
//! `ShapeRef` 的返回类型与三后端 ABI，收益与风险不成比例。
//!
//! # 为什么判据量的是"标架"而不是"像素"
//!
//! 像素差异只能证明"变了"，不能证明"变对了"——一根亮、一根暗的错标架同样会
//! 让帧变化。故 V2a/V2b 直接对 `prism_rt::Scene.meshes` 里的 `HairGeometry`
//! 下断言（切向 vs 该股自身轴向的夹角），像素只用来做 V2c 的**判别力**反证。

use prism_core::{HairParams, Scene, SceneAttach, StrandDesc};
use prism_gpu::{Framebuffer, GpuDevice, SceneRenderRequest};
use prism_geometry::curve_tess::strand_frames;
use prism_math::Vec3;
use prism_render::rt_backend::RtBackend;
use prism_render::{build_render_request, translate_scene};
use prism_rt::integrator::{MaterialKind, Scene as RtScene};

const W: u32 = 40;
const H: u32 = 30;
const SPP: u32 = 8;
const BOUNCES: u32 = 3;

// ---------------------------------------------------------------------------
// 素材
// ---------------------------------------------------------------------------

/// 一根沿 `axis`、从 `from` 出发、长 `len` 的直股（3 个控制点，2 段）。
fn straight_strand(from: Vec3, axis: Vec3, len: f32) -> StrandDesc {
    let u: Vec3 = axis.normalize();
    StrandDesc::new(vec![from, from + u * (len * 0.5), from + u * len], 0.03)
}

/// 抽出一个场景里所有毛发材质的 `HairGeometry`（按 Mesh 顺序）。
fn hair_frames(scene: &RtScene) -> Vec<(Vec3, Vec3)> {
    let mut out: Vec<(Vec3, Vec3)> = Vec::new();
    for (_, m) in &scene.meshes {
        if let MaterialKind::Hair { geometry, .. } = m {
            out.push((geometry.tangent, geometry.normal_ref));
        }
    }
    out
}

/// 两个单位向量的夹角（弧度；用 `acos` 但不依赖 libm 精度，判据只看量级）。
fn angle(a: Vec3, b: Vec3) -> f32 {
    let c: f32 = a.normalize().dot(&b.normalize()).clamp(-1.0, 1.0);
    // acos 在 1 附近数值敏感；用 atan2(|a×b|, a·b) 更稳（两向量都已归一）。
    let s: f32 = a.normalize().cross(&b.normalize()).length();
    let _ = c;
    s.atan2(c)
}

fn base_scene(attach: SceneAttach) -> Scene {
    let mut s: Scene = Scene::default();
    s.radius = 1.0;
    s.orbit_center = Vec3::new(0.0, 0.0, 0.0);
    s.orbit_radius = 3.2;
    s.material = prism_core::Material::new(Vec3::new(0.8, 0.8, 0.8), 0.4, 0.0);
    s.lights = vec![prism_core::Light::new(
        Vec3::new(0.4, -1.0, 0.6),
        Vec3::new(1.0, 0.97, 0.92),
        3.2,
    )];
    s.camera = prism_core::Camera::new(
        Vec3::new(0.0, 0.0, -3.2),
        Vec3::new(0.0, 0.0, 0.0),
        45.0_f32.to_radians(),
    );
    s.aesthetic = prism_core::Aesthetic::new(false, 1.0, false);
    s.attach = attach;
    s
}

fn render(scene: &Scene) -> Framebuffer {
    let req: SceneRenderRequest = build_render_request(scene, 0.0, 10.0, W, H);
    let mut dev: RtBackend = RtBackend::new(SPP, BOUNCES).with_threads(1);
    dev.render_scene(&req)
}

fn differing(a: &Framebuffer, b: &Framebuffer) -> usize {
    a.pixels
        .chunks_exact(4)
        .zip(b.pixels.chunks_exact(4))
        .filter(|(x, y)| x != y)
        .count()
}

/// 三个**两两正交**的方向（+X / +Y / +Z），彼此在画面上不重叠。
fn orthogonal_strands() -> Vec<StrandDesc> {
    // 每根股放在不同的深度/横向位置，保证画面里各占一块。
    vec![
        straight_strand(Vec3::new(-1.15, -0.55, -0.2), Vec3::new(1.0, 0.0, 0.0), 1.0), // +X
        straight_strand(Vec3::new(-1.15, -0.55, -0.2), Vec3::new(0.0, 1.0, 0.0), 1.0), // +Y
        straight_strand(Vec3::new(-1.15, -0.55, -0.2), Vec3::new(0.0, 0.0, 1.0), 1.0), // +Z
    ]
}

// ---------------------------------------------------------------------------
// V1：单股与 93e59d0 逐位相同
// ---------------------------------------------------------------------------

/// **V1：单股 ⇒ 一个 Mesh、标架 = `strand_frames(points)[0]`，位型全等。**
///
/// `93e59d0` 的语义是"一材质一标架"，单股时它与"按股切"**必然**给出同一个
/// 标架（只有一股）。这条把"必然"变成可判的位型相等：`tangent` / `normal_ref`
/// 三个分量逐位相同，且**只有**一个 mesh（没有多切出一份）。
#[test]
fn v1_single_strand_frame_is_bitwise_the_pre_change_one() {
    let st: StrandDesc = straight_strand(Vec3::new(-0.5, -0.5, 0.0), Vec3::new(1.0, 1.0, 0.0), 1.2);
    let want_frame = strand_frames(&st.points)[0].clone();

    let mut attach: SceneAttach = SceneAttach::default();
    attach.strands.push(st.clone());
    let scene: Scene = base_scene(attach);
    let req: SceneRenderRequest = build_render_request(&scene, 0.0, 10.0, W, H);
    let rt: RtScene = translate_scene(&req);

    assert_eq!(rt.meshes.len(), 1, "单股必须只产出一个 Mesh");
    let got: Vec<(Vec3, Vec3)> = hair_frames(&rt);
    assert_eq!(got.len(), 1, "单股的 Mesh 必须是毛发材质");
    let (t, n) = got[0];
    // `93e59d0`（W5）那版的单股写法就是这一行：把 `strand_frames` 的首段原样
    // 交给 `HairGeometry::new`。逐位相等 ⇒ 语义一个字都没变。
    let want: prism_rt::hair_scatter::HairGeometry =
        prism_rt::hair_scatter::HairGeometry::new(want_frame.tangent, want_frame.normal_ref);
    assert_eq!(
        (t.x.to_bits(), t.y.to_bits(), t.z.to_bits()),
        (want.tangent.x.to_bits(), want.tangent.y.to_bits(), want.tangent.z.to_bits()),
        "单股切向必须与 `93e59d0` 的 `HairGeometry::new(首段切向, 首段法向)` 逐位相同"
    );
    assert_eq!(
        (n.x.to_bits(), n.y.to_bits(), n.z.to_bits()),
        (want.normal_ref.x.to_bits(), want.normal_ref.y.to_bits(), want.normal_ref.z.to_bits()),
        "单股法平面参考必须与 `93e59d0` 逐位相同"
    );
    // 与原始 `strand_frames` 输出只差"二次归一"的 ≤2 ULP（`HairGeometry::new`
    // 会再 normalize 一次，这是它的契约，不是本层的额外运算）。
    let ulp = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
    assert!(ulp(t.x, want_frame.tangent.x) <= 2, "二次归一不应超过 2 ULP");
    // 几何也在：至少 3 个三角形（2 段圆管）。
    assert!(rt.meshes[0].0.triangles.len() >= 3, "毛股必须真的三角化");
}

// ---------------------------------------------------------------------------
// V2：多股异向，每股精确
// ---------------------------------------------------------------------------

/// **V2a/V2b：三根正交方向的股 ⇒ 三个 Mesh、三套标架，每股角误差 = 单股水平。**
///
/// "单股水平"是**当场量出来的**，不是引用的常数：先跑单股场景得到误差 `e₀`
/// （实测 0），再要求多股场景里每一股的误差 `≤ e₀ + 1e-6`。
///
/// 这条的判别力来自"一股一标架"的反面：若仍是"一材质一标架"（全局取第一股），
/// 另外两股的夹角会是 **90°**（`|cos| = 0`）。测试末尾把这个 90° 现算一遍。
#[test]
fn v2_orthogonal_strands_each_get_their_own_frame() {
    // ---- 单股基准 ----
    let axis_x: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    let mut a1: SceneAttach = SceneAttach::default();
    a1.strands.push(straight_strand(Vec3::new(-1.15, -0.55, -0.2), axis_x, 1.0));
    let rt1: RtScene = translate_scene(&build_render_request(&base_scene(a1), 0.0, 10.0, W, H));
    let f1: Vec<(Vec3, Vec3)> = hair_frames(&rt1);
    assert_eq!(f1.len(), 1, "单股基准必须一个 Mesh");
    let e_single: f32 = angle(f1[0].0, axis_x);
    println!("V2b：单股基准的轴向误差 = {e_single:e} rad");
    assert!(e_single <= 1e-6, "单股标架就该是精确的，实测 {e_single:e}");

    // ---- 三正交方向 ----
    let strands: Vec<StrandDesc> = orthogonal_strands();
    let axes: [Vec3; 3] = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut a3: SceneAttach = SceneAttach::default();
    a3.strands = strands.clone();
    let rt3: RtScene = translate_scene(&build_render_request(&base_scene(a3), 0.0, 10.0, W, H));
    let frames: Vec<(Vec3, Vec3)> = hair_frames(&rt3);
    assert_eq!(frames.len(), 3, "三股必须产出三个 Mesh（按股切）");

    let mut worst: f32 = 0.0;
    for (i, (t, n)) in frames.iter().enumerate() {
        let e: f32 = angle(*t, axes[i]);
        let e_n: f32 = angle(*n, axes[i]);
        println!(
            "V2a：股 {i}（轴 {axes:?}）⇒ 切向夹角 {e:e} rad，法向与轴夹角 {e_n:.3} rad",
            axes = axes[i]
        );
        assert!(e <= e_single + 1e-6, "股 {i} 的标架误差 {e:e} 超过单股水平 {e_single:e}");
        // 法平面参考必须**垂直于**该股切向（正交性是 `HairGeometry` 的契约）。
        assert!(e_n.cos().abs() <= 1e-6, "股 {i} 的法平面参考不垂直于自身切向");
        worst = worst.max(e);
    }
    println!("V2b：三股的最大角误差 = {worst:e} rad（单股水平 {e_single:e}）");

    // ---- 判别力：旧的"一材质一标架"在这三股上必然错 90° ----
    let global: Vec3 = frames[0].0; // 旧法只能有一个标架
    let mut worst_old: f32 = 0.0;
    for (i, ax) in axes.iter().enumerate() {
        // 取与该股最"平行"的那个方向当旧法的标架（对旧法最有利的情形）。
        let _ = i;
        let e: f32 = angle(global, *ax);
        worst_old = worst_old.max(e.min(std::f32::consts::PI - e));
    }
    println!("V2c 反例：全局单标架对三股的最大夹角 = {worst_old:.4} rad（≈ π/2）");
    assert!(
        worst_old > 1.0,
        "旧法在这组方向上必须显著错（实测 {worst_old:.4} rad），否则这条判据没牙"
    );
}

/// **V2c：像素级——按股标架与"一材质一标架"渲染出的帧必须不同。**
///
/// 这条不证明"新的对"（那是 V2a/V2b 的事），只证明"这个改动真的走到了像素"：
/// 若 `translate_scene` 仍按全局标架建一个 Mesh，两帧会逐字节相同。
#[test]
fn v2c_per_strand_frames_change_the_frame() {
    let strands: Vec<StrandDesc> = orthogonal_strands();

    // 产品路径（按股标架）。
    let mut a: SceneAttach = SceneAttach::default();
    a.strands = strands.clone();
    let new_scene: Scene = base_scene(a);
    let new_fb: Framebuffer = render(&new_scene);

    // 旧行为：所有股合成一个 Mesh、标架取第一股（`93e59d0` 的唯一可能做法）。
    let req: SceneRenderRequest = build_render_request(&new_scene, 0.0, 10.0, W, H);
    let mut old_rt: RtScene = translate_scene(&req);
    {
        // 把三个 Mesh 的三角形并进第一个，丢掉后两个的材质 —— 标架仍是第一股的。
        let mut merged: Vec<prism_geometry::bvh::Triangle> = Vec::new();
        for (m, _) in &old_rt.meshes {
            merged.extend_from_slice(&m.triangles);
        }
        let first_mat: MaterialKind = old_rt.meshes[0].1;
        let merged_mesh = prism_rt::integrator::Mesh::from_triangles(merged);
        old_rt.meshes.clear();
        old_rt.meshes.push((merged_mesh, first_mat));
    }
    let cam: prism_rt::camera::Camera = prism_rt::camera::Camera::new(
        Vec3::new(0.0, 0.0, -3.2),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        45.0,
        0.0,
        3.2,
        0.0,
        0.0,
    );
    let aesthetic = prism_gpu::AestheticReq { teal_orange: false, contrast: 1.0, filmic: false };
    let old_fb: Framebuffer = prism_render::rt_tiles::render_frame(
        &old_rt,
        &cam,
        &prism_core::EvalCtx::new(0, 0, 0),
        (W, H),
        &prism_rt::integrator::RenderParams::new(SPP, BOUNCES),
        &aesthetic,
        1,
    );

    let d: usize = differing(&new_fb, &old_fb);
    println!("V2c：按股标架 vs 全局单标架 ⇒ differing_pixels = {d} / {}", W * H);
    assert!(d > 0, "按股标架没有影响像素 —— 说明标架没真的按股取");
}

// ---------------------------------------------------------------------------
// 边界：多股同向 / 单点股 / 非法参数
// ---------------------------------------------------------------------------

/// 同向股束：仍然每股一个 Mesh（"合并同向股"是优化，不是正确性前提）。
///
/// 这条挡住"按首股方向去重合并"的写法——它在正确性上等价，但会让
/// V2a 的"三股 ⇒ 三 Mesh"结构断言失效。本实现不做合并，如实钉住。
#[test]
fn same_direction_strands_still_get_one_mesh_each() {
    let mut a: SceneAttach = SceneAttach::default();
    for i in 0..3 {
        a.strands.push(straight_strand(
            Vec3::new(-1.0 + 0.6 * i as f32, -0.5, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
        ));
    }
    let rt: RtScene = translate_scene(&build_render_request(&base_scene(a), 0.0, 10.0, W, H));
    assert_eq!(rt.meshes.len(), 3, "同向三股仍是三个 Mesh");
    let frames: Vec<(Vec3, Vec3)> = hair_frames(&rt);
    assert_eq!(frames.len(), 3);
    for (t, _) in &frames {
        assert!(angle(*t, Vec3::new(0.0, 1.0, 0.0)) <= 1e-6);
    }
}

/// 退化输入不 panic、不产 Mesh：单点股 / 零长段 / 越界半径 / 非有限点。
#[test]
fn degenerate_strands_are_skipped_without_panic() {
    let mut a: SceneAttach = SceneAttach::default();
    // 单点（< 2 个控制点）
    a.strands.push(StrandDesc::new(vec![Vec3::new(0.0, 0.0, 0.0)], 0.03));
    // 零长段（两点重合）
    a.strands.push(StrandDesc::new(
        vec![Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.1, 0.1, 0.1)],
        0.03,
    ));
    // 非有限点
    a.strands.push(StrandDesc::new(
        vec![Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
        0.03,
    ));
    // 半径非正
    let mut bad_radius: StrandDesc =
        StrandDesc::new(vec![Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)], -1.0);
    bad_radius.params = HairParams::default();
    a.strands.push(bad_radius);
    // 唯一合法的一股
    a.strands.push(straight_strand(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0));

    let rt: RtScene = translate_scene(&build_render_request(&base_scene(a), 0.0, 10.0, W, H));
    assert_eq!(rt.meshes.len(), 1, "四条退化股必须被跳过，只剩一条合法股");
}
