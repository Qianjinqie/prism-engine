//! W5 毛发判据：V4（帧不连续 ≤ 5°）+ `MaterialKind::Hair` 出图。
//!
//! `hair_scatter.rs` **一行不改**（R-5 的 1544 行测试是已对齐资产）——本文件的
//! V3 由"`cargo test -p prism-rt --lib hair_scatter` 全绿"承担，不在这里重复。
//!
//! 判据 V1（与曲线同路，`differing_pixels = 0`）在
//! `tests/w4_w5_gpu_parity.rs`（要走 llvmpipe）。

use prism_core::EvalCtx;
use prism_geometry::curve_tess::{strand_frames, tessellate, CurveReq};
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::hair_scatter::{HairGeometry, HairScatter};
use prism_rt::integrator::{render_pixel, Light, MaterialKind, RenderParams, Scene};

/// 一条空间螺旋毛股（20 段）。
///
/// 段数是判据 V4 的**采样条件**：判据说的是"相邻段夹角 ≤ 5°"，而相邻段的
/// 夹角由中心线本身的转折角决定（平行传输不会额外加扭转）。故股必须采到
/// 每段转折 ≤ 5° 才谈得上这条判据——20 段螺旋的实测转折角约 1.2°。
fn helix_strand() -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 0..=20 {
        let a = 0.35 * (i as f32) * 0.25;
        let (c, s) = prism_rt::sampler::cos_sin(a);
        pts.push(Vec3::new(c * 0.4, i as f32 * 0.12, s * 0.4));
    }
    pts
}

/// **V4：沿股采样 `HairGeometry::longitudinal / azimuth`，相邻段夹角 ≤ 5°。**
///
/// 两件事一起测，缺一条都说明不了"帧连续"：
///
/// 1. **帧本身**：相邻段的 `HairGeometry` 基（`tangent` / `normal_ref`）夹角必须
///    ≤ 5°。这就是判据里的"相邻段夹角"——平行传输的语义是"帧只跟着切向转，
///    不额外绕轴打转"，所以相邻帧的相对旋转量就等于段间转折角。
/// 2. **采样函数自洽**：把一个固定世界方向投影成 `(θ, φ)`，再用
///    `d' = u·sinθ + (v·cosφ + w·sinφ)·cosθ` 反投影回世界方向。`d'` 必须还原
///    `d`（误差只来自 `asin`/`atan2` 的数值精度）。这条排除"两个函数各自能跑
///    但基向量取错"的情形（例如 `azimuth` 的 w 用了 `v × u` 而不是 `u × v`，
///    反投影会把它镜像回来，但 φ 的符号就反了——反投影恰能抓住这类错）。
#[test]
fn v4_frame_rotation_along_strand_within_5_degrees() {
    let pts = helix_strand();
    let frames = strand_frames(&pts);
    assert_eq!(frames.len(), pts.len() - 1, "每股应有 (n−1) 个段标架");

    // 固定世界方向：不与任何一段切向平行（否则 φ 退化）。
    let d = Vec3::new(0.3, 0.5, 0.81).normalize();
    const COS_5DEG: f32 = 0.996_194_7;

    let geos: Vec<HairGeometry> = frames
        .iter()
        .map(|f| HairGeometry::new(f.tangent, f.normal_ref))
        .collect();
    // (1) 帧夹角。
    let mut min_n = f32::INFINITY;
    let mut min_u = f32::INFINITY;
    // 诊断量：相邻帧 normal_ref 的相对旋转量与切向的相对旋转量之差。
    //
    // **它不是"额外扭转"的度量**（我一度这么以为，实测把它否了）：最小旋转的
    // 转轴 `k = u₀ × u₁` 与 `u₀` 正交，但 `normal_ref` 一般**不**与 `k` 正交，
    // 于是 `dot(n₀, n₁) = cos²α + sin²α·cosθ ≥ cosθ`（α = n₀ 与 k 的夹角）。
    // 两者天然不等，差值随 α 增大而增大。故只打印、不断言。
    let mut max_twist = 0.0_f32;
    for w in geos.windows(2) {
        let dn = w[0].normal_ref.dot(&w[1].normal_ref);
        let du = w[0].tangent.dot(&w[1].tangent);
        min_n = min_n.min(dn);
        min_u = min_u.min(du);
        max_twist = max_twist.max((dn - du).abs());
    }
    // (2) 采样函数自洽。
    let mut worst_roundtrip = 0.0_f32;
    for g in &geos {
        let th = g.longitudinal(d);
        let ph = g.azimuth(d);
        let u = g.tangent;
        let v = g.normal_ref;
        let w = u.cross(&v);
        let (cph, sph) = prism_rt::sampler::cos_sin(ph);
        let (cth, sth) = prism_rt::sampler::cos_sin(th);
        let back = u * sth + (v * cph + w * sph) * cth;
        worst_roundtrip = worst_roundtrip.max((back.dot(&d) - 1.0).abs());
    }
    println!(
        "V4：{} 段；相邻帧 normal_ref 最小点积 {min_n:.8}、tangent 最小点积 {min_u:.8}\
         （cos5° = {COS_5DEG:.8}）；帧角差 {max_twist:e}（诊断量，见注释）；\
         (θ,φ) 反投影 vs 原方向最大偏差 {worst_roundtrip:e}",
        geos.len()
    );
    let _ = max_twist;
    assert!(min_n >= COS_5DEG, "相邻帧 normal_ref 夹角超过 5°：{min_n} < {COS_5DEG}");
    assert!(min_u >= COS_5DEG, "相邻帧 tangent 夹角超过 5°：{min_u} < {COS_5DEG}");
    assert!(worst_roundtrip < 1e-5, "longitudinal/azimuth 反投影不自洽：{worst_roundtrip:e}");
}

/// 毛发材质出图：`MaterialKind::Hair` 走 `HairScatter::eval_full`，帧非平凡。
///
/// 同时验证"接线没有把毛发退化成普通 Lambert"：同一几何换 Lambert 材质，
/// 两者的像素**必须不同**（否则说明 `surface_brdf` 的 Hair 臂没生效）。
#[test]
fn hair_material_renders_and_differs_from_lambert() {
    let pts = helix_strand();
    let radii = vec![0.06_f32];
    let curve = CurveReq::Strand { points: pts.clone(), radii: radii.clone() };
    let tris = tessellate(&curve, 0.01);
    assert!(!tris.is_empty());

    let frame = strand_frames(&pts);
    let geo = HairGeometry::new(frame[0].tangent, frame[0].normal_ref);
    let scatter = HairScatter::blond();
    let hair = MaterialKind::Hair { scatter, geometry: geo };

    let render = |mat: MaterialKind| -> Vec<[f32; 4]> {
        let mut scene = Scene::default();
        scene.add_bvh(tris.clone(), mat);
        scene.lights.push(Light::Rect {
            center: Vec3::new(2.0, 2.0, 2.0),
            edge_u: Vec3::new(1.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::new(20.0, 18.0, 15.0),
        });
        scene.lights.push(Light::Ambient { radiance: Vec3::new(0.1, 0.1, 0.12) });
        let cam = Camera::new(
            Vec3::new(0.0, 1.2, 2.6),
            Vec3::new(0.0, 1.2, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            40.0,
            0.0,
            4.0,
            0.0,
            1.0 / 24.0,
        );
        let ctx = EvalCtx::new(20261102, 21, 9);
        let res = (12u32, 12u32);
        let params = RenderParams::new(6, 3);
        let mut out = Vec::with_capacity((res.0 * res.1) as usize);
        for y in 0..res.1 {
            for x in 0..res.0 {
                out.push(render_pixel(&scene, &cam, &ctx, (x, y), res, &params));
            }
        }
        out
    };

    let a = render(hair);
    let b = render(MaterialKind::Lambert { albedo: Vec3::splat(0.55) });
    for (i, px) in a.iter().enumerate() {
        for c in px {
            assert!(c.is_finite(), "像素 {i} 非有限");
        }
    }
    let lit = a.iter().filter(|p| p[0] + p[1] + p[2] > 0.02).count();
    assert!(lit > 0, "毛发场景必须被照亮");
    let diff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| {
            (x[0] - y[0]).abs() > 1e-6 || (x[1] - y[1]).abs() > 1e-6 || (x[2] - y[2]).abs() > 1e-6
        })
        .count();
    assert!(diff > 0, "Hair 与 Lambert 的像素完全相同 ⇒ 毛发臂没生效");
    println!("Hair 出图：{} 三角，{lit} 个被照亮像素，与 Lambert 差 {diff} 个像素", tris.len());
}
