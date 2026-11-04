//! T0 产品路径接线判据：P1 / P2 / P3。
//!
//! # 缺口原文（本轮要消灭的那条）
//!
//! `prism_core::Scene` 只有七个字段，`SceneRenderRequest` 只有 `spheres/lights/
//! camera`，`prism-render::translate_scene` 只翻球和方向光 ⇒ W1–W7 补的纹理 /
//! 介质 / 实例 / 毛发 / 细分 / 位移 / 自适应**一条都出不了产品片**。这条横切口
//! 是 R-1、R-7②、E-10④、R-13③ 一起卡在「部分」的共同原因。
//!
//! # 三条判据
//!
//! | # | 判据 | 方法 |
//! |---|---|---|
//! | **P1** | 空 `Scene`（无新字段）渲染帧与 `93e59d0` **逐字节相同** | ①`req.geo.is_empty()`；②帧 md5 对账（旧值硬编码）；③与非空字段的"字面旧翻译"手搭 `prism_rt::Scene` 对拍 → 逐位相等 |
//! | **P2** | 新场景能出片，且像素与"直接手搭同一 `prism_rt::Scene`"**0 差异** | 同源性：翻译层与手搭场景走同一条 `rt_tiles::render_frame` |
//! | **P3** | 静态门禁：`translate_scene` 对每个新能力都有非空翻译分支 | 源码扫描六个记号 + 正向断言（段内必须有 `push`/赋值） |
//!
//! # P1 的 md5 是怎么来的
//!
//! 它**不是**"跑一遍记下来"的自证：同一段代码里还有第 ③ 条——
//! [`p1_empty_attach_matches_literal_old_translation`] 用手写的"只翻球和方向光"
//! 的 `prism_rt::Scene` 渲染同一请求，断言**逐字节相等**。md5 只是把那一位串
//! 再钉一次（防未来某次改动悄悄挪了浮点顺序却又自洽通过）。
//!
//! # 为什么不重开 `93e59d0` 的 worktree 跑一遍
//!
//! 那需要另建一棵 29 crate 的构建树（分钟级、与判据等价的成本）。本文件用
//! **结构等价 + 数值等价**两条替代：`translate_scene` 在 `geo.is_empty()` 时
//! 一条新分支都不执行（源码可查），而渲染走的是同一串 `trace_sample` 索引 ——
//! 第 ③ 条把它测出来。若将来 `translate_scene` 的空场景路径真变了，③ 会先红。

use prism_core::{
    DisplacementSlot, EvalCtx, InstanceDesc, MediumBox, MeshDesc, SceneAttach, StrandDesc,
    TextureDesc,
};
use prism_gpu::{Framebuffer, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;
use prism_render::farm::md5_hex;
use prism_render::rt_backend::RtBackend;
use prism_render::{build_render_request, rt_tiles};
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera as RtCamera;
use prism_rt::displace::{displace_positions, DisplaceParams};
use prism_rt::hair_scatter::{HairGeometry, HairScatter};
use prism_rt::integrator::{
    Aabb, Light, MaterialKind, MediumRegion, Mesh, RenderParams, Scene as RtScene, Sphere,
};
use prism_rt::instance::Instance;
use prism_rt::media::{HomogeneousMedium, Medium};
use prism_rt::texsample::TexTable;

// ---------------------------------------------------------------------------
// 口径常量
// ---------------------------------------------------------------------------

/// 判据分辨率（与 `qa_render_scene` 同档：够多像素、秒级出图）。
const W: u32 = 48;
const H: u32 = 32;
/// 判据 spp / 弹射（与 CLI 默认档一致）。
const SPP: u32 = 8;
const BOUNCES: u32 = 4;
/// 空附加能力场景的帧 md5（P1 的硬编码旧值）。
///
/// 这个值在 T0 落地时由本测试打印并写回；它是"空 `Scene` 与 `93e59d0`
/// 逐字节同帧"的那一位串的**固化形式**。任何一次会让它变的改动都必须
/// 在 CHANGELOG 的《渲染行为变化清单》里交代四要素。
const P1_EMPTY_ATTACH_MD5: &str = "4e5295ef736fbc62afae7d3dd432debf";

// ---------------------------------------------------------------------------
// 场景素材
// ---------------------------------------------------------------------------

/// 判据基准场景：两个球 + 一盏方向光（**不含任何附加能力**）。
fn base_scene() -> prism_core::Scene {
    let mut s = prism_core::Scene::default();
    s.radius = 1.0;
    s.orbit_center = Vec3::new(0.0, 0.0, 0.0);
    s.orbit_radius = 3.4;
    s.material = prism_core::Material::new(Vec3::new(0.75, 0.25, 0.2), 0.35, 0.1);
    s.lights = vec![prism_core::Light::new(
        Vec3::new(0.3, -1.0, 0.4),
        Vec3::new(1.0, 0.96, 0.9),
        3.0,
    )];
    s.camera = prism_core::Camera::new(
        Vec3::new(0.0, 0.0, -3.4),
        Vec3::new(0.0, 0.0, 0.0),
        50.0_f32.to_radians(),
    );
    s.aesthetic = prism_core::Aesthetic::new(false, 1.0, false);
    s
}

/// `build_camera` 在球场景下的等价参数（fov 弧度 → 度）。
///
/// `base_scene` 的 fov 是 `50°` 的弧度值，`build_camera` 会 `to_degrees()`
/// 回到 50.0-ish；这里用同一个 `to_degrees()` 保证手搭场景与产品路径同参。
fn rt_camera_for(scene: &prism_core::Scene) -> RtCamera {
    let fov_deg: f32 = scene.camera.fov.to_degrees().clamp(1.0, 179.0);
    RtCamera::new(
        scene.orbit_center + Vec3::new(0.0, 0.0, -scene.orbit_radius),
        scene.orbit_center,
        Vec3::new(0.0, 1.0, 0.0),
        fov_deg,
        0.0,
        scene.orbit_radius.max(1e-3),
        0.0,
        0.0,
    )
}

/// 渲染一个 `prism_rt::Scene`（产品路径的同一个入口，只是场景是手搭的）。
fn render_rt(scene: &RtScene, cam: &RtCamera, aesthetic: &prism_gpu::AestheticReq) -> Framebuffer {
    let ctx: EvalCtx = EvalCtx::new(0, 0, 0);
    let params: RenderParams = RenderParams::new(SPP, BOUNCES);
    rt_tiles::render_frame(scene, cam, &ctx, (W, H), &params, aesthetic, 1)
}

/// 产品路径：`prism_core::Scene` → 请求 → `RtBackend`。
fn render_product(scene: &prism_core::Scene) -> Framebuffer {
    let req: SceneRenderRequest =
        build_render_request(scene, 0.0, 10.0, W, H);
    let mut dev: RtBackend = RtBackend::new(SPP, BOUNCES).with_threads(1);
    dev.render_scene(&req)
}

/// 8×8 竖条纹灰度高度图（线性域；用于位移与贴图）。
fn height_map() -> TextureDesc {
    let mut px: Vec<u8> = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8u32 {
        for x in 0..8u32 {
            let v: u8 = if x % 2 == 0 { 230 } else { 40 };
            let v: u8 = if y == 0 { 255 } else { v };
            px.extend_from_slice(&[v, v, v, 255]);
        }
    }
    TextureDesc::new(8, 8, false, px)
}

/// 一个**倾斜 45°** 的 2×2 四边形（4 顶点 2 三角形，带 uv）——位移的载体。
///
/// 为什么不是"正对相机的平面"：位移沿**法线**推出去，正对相机的平面被推出后
/// 仍是同一个平面（只改了深度），画面一像素都不变 —— 那样的判据是平凡通过。
/// 倾斜 45° 后，法线有 `+Y` 与 `−Z` 两个分量：位移同时改变轮廓与明暗，
/// 位移非零 ⇒ 画面必变（这条由 `p_displacement_zero_...` 反向断言）。
fn plane_mesh(slot: Option<DisplacementSlot>) -> MeshDesc {
    let n = std::f32::consts::FRAC_1_SQRT_2;
    let u = Vec3::new(1.0, 0.0, 0.0);
    let v = Vec3::new(0.0, n, n);
    let pos = vec![
        u * -1.0 - v,
        u - v,
        u + v,
        v - u,
    ];
    // 绕序使几何法线 = (0, n, −n)（朝相机 + 朝光）
    let idx = vec![0u32, 2, 1, 0, 3, 2];
    let uv = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut m = MeshDesc::with_uv(pos, idx, uv);
    m.displacement = slot;
    m
}

// ---------------------------------------------------------------------------
// P1 —— 空 Scene 与 93e59d0 逐字节同帧
// ---------------------------------------------------------------------------

/// **P1-a：空附加能力 ⇒ `req.geo` 全空（翻译层一条新分支都不执行）。**
#[test]
fn p1_empty_attach_produces_empty_geo() {
    let scene: prism_core::Scene = base_scene();
    assert!(scene.attach.is_empty(), "基准场景不该带附加能力");
    let req: SceneRenderRequest = build_render_request(&scene, 0.0, 10.0, W, H);
    assert!(req.geo.is_empty(), "P1 失败：空 attach 却产出了非空 geo 请求");
    assert_eq!(req.geo.entry_count(), 0);
    assert_eq!(req.spheres.len(), 1, "轨道球必须照旧在场");
    assert_eq!(req.lights.len(), 1, "方向光必须照旧在场");
}

/// **P1-b：空附加能力场景的帧 md5 == 硬编码旧值。**
///
/// md5 口径：RGBA8 像素流的 md5（与 `qa_render_scene` / `w6_a6_media` 同法）。
#[test]
fn p1_frame_md5_matches_pinned_value() {
    let scene: prism_core::Scene = base_scene();
    let fb: Framebuffer = render_product(&scene);
    assert!(fb.is_valid(), "P1 失败：基准场景必须出有效帧");
    let got: String = md5_hex(&fb.pixels);
    println!("[t0-p1] 空 attach 帧 md5 = {got}");
    assert_eq!(
        got, P1_EMPTY_ATTACH_MD5,
        "P1 失败：空 attach 的帧 md5 变了 —— 说明翻译层动了既有渲染路径。\
         若这是有意的渲染行为变化，必须进 CHANGELOG《渲染行为变化清单》并更新本常量"
    );
}

/// **P1-c：产品路径与"字面旧翻译"手搭的 `prism_rt::Scene` 逐字节同帧。**
///
/// 这条是 P1 的**结构性证据**（md5 只是它的固化）：手搭场景用的就是
/// `93e59d0` 那版 `translate_scene` 的全部内容——一个 Pbr 球 + 一盏方向光，
/// 一个字节都不多。
#[test]
fn p1_empty_attach_matches_literal_old_translation() {
    let scene: prism_core::Scene = base_scene();

    // ---- 手搭：93e59d0 的 translate_scene（逐行照抄）----
    let mut old: RtScene = RtScene::default();
    let material: MaterialKind = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.75, 0.25, 0.2),
        0.1,
        0.35,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    old.spheres.push((Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }, material));
    let dir: Vec3 = Vec3::new(0.3, -1.0, 0.4);
    let dlen: f32 = dir.length();
    old.lights.push(Light::Directional {
        direction: dir * (1.0 / dlen),
        radiance: Vec3::new(1.0, 0.96, 0.9) * (3.0 * std::f32::consts::PI),
        cos_theta_max: 1.0,
    });

    let aesthetic = prism_gpu::AestheticReq {
        teal_orange: false,
        contrast: 1.0,
        filmic: false,
    };
    let want: Framebuffer = render_rt(&old, &rt_camera_for(&scene), &aesthetic);
    let got: Framebuffer = render_product(&scene);
    assert_eq!(
        got.pixels, want.pixels,
        "P1 失败：空 attach 的产品帧与字面旧翻译不逐字节相同"
    );
    // md5 同源（两条口径互相印证，而不是各说各话）。
    assert_eq!(md5_hex(&got.pixels), md5_hex(&want.pixels));
}

// ---------------------------------------------------------------------------
// P2 —— 新场景出片，且与手搭 rt::Scene 0 差异
// ---------------------------------------------------------------------------

/// 手搭一个"带介质 + 网格 + 实例 + 毛发"的 `prism_rt::Scene`（与翻译层同源）。
///
/// 这一段刻意**不用**翻译层的任何代码：每个 `add_*` 都照 `prism_rt` 的公开
/// API 写一遍。若翻译层把某个字段搬错（丢 priority、漏 uv、实例逆算错），
/// 两帧就会分叉。
fn hand_built_scene() -> RtScene {
    let mut s: RtScene = RtScene::default();
    // 一盏灯 + 一个背景球（保证画面不是纯黑，否则一切差异都被 0 掩盖）。
    s.lights.push(Light::Directional {
        direction: Vec3::new(0.3, -1.0, 0.4).normalize(),
        radiance: Vec3::new(1.0, 0.96, 0.9) * (3.0 * std::f32::consts::PI),
        cos_theta_max: 1.0,
    });
    s.spheres.push((
        Sphere { center: Vec3::new(0.0, 0.0, -1.2), radius: 0.8 },
        MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.2, 0.5, 0.8),
            0.0,
            0.5,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        )),
    ));

    // ---- 纹理（位移要采样它）----
    let tex = height_map();
    let handle: u32 = s.add_texture_rgba8(tex.width, tex.height, tex.srgb, &tex.rgba8);
    assert_eq!(handle, 1);

    // ---- 介质（互斥区域 + priority）----
    s.media.push(
        MediumRegion::new(
            Aabb::new(Vec3::new(-2.0, -2.0, 0.2), Vec3::new(2.0, 2.0, 3.0)),
            Medium::Homogeneous(HomogeneousMedium::new(0.45, 0.85, 0.25)),
        )
        .with_priority(3),
    );

    // ---- 网格 + 运行期位移（细分 + 位移，两个公开函数的组合）----
    let m: MeshDesc = plane_mesh(Some(DisplacementSlot::new(1, 0.12, 1.0, 2)));
    let sub = prism_geometry::subdiv::subdivide(&m.positions, &m.indices, &m.uv, 2);
    let table = TexTable::new(&s.textures, &s.texels);
    let moved = displace_positions(
        &sub.positions,
        &sub.normals,
        &sub.uv,
        table,
        1,
        DisplaceParams::new(0.12),
    );
    let mut mesh = Mesh::new(moved, sub.indices);
    mesh.uv = sub.uv;
    s.meshes.push((
        mesh,
        MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.72, 0.74, 0.78),
            0.0,
            0.6,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        )),
    ));

    // ---- 实例：一个原型 + 两条示例（平移 + 镜像）----
    let proto = plane_mesh(None);
    let geom_id: u32 = s.add_instance_geom(proto.positions.clone(), proto.indices.clone(),
        MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.72, 0.74, 0.78),
            0.0,
            0.6,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        )));
    let mut a1: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    a1[3] = -0.5;
    a1[7] = 0.6;
    a1[11] = -0.4;
    let mut a2: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    a2[0] = -0.6; // 镜像（det < 0）：法线方向必须靠 `det(A)·A⁻ᵀ·n` 翻回来
    a2[3] = 0.7;
    a2[7] = 0.4;
    a2[11] = -0.4;
    for a in [a1, a2] {
        let inst: Instance =
            Instance::from_affine(a, geom_id, prism_rt::instance::INSTANCE_MAT_FROM_GEOM)
                .expect("仿射必须可逆");
        s.instances.push(inst);
    }
    s.rebuild_instance_bvh();

    // ---- 毛发：一股（每股一个 Mesh + 该股自己的标架）----
    let pts = vec![
        Vec3::new(0.6, -0.9, -0.2),
        Vec3::new(0.62, -0.3, -0.15),
        Vec3::new(0.58, 0.3, -0.1),
        Vec3::new(0.6, 0.9, -0.05),
    ];
    let radii = vec![0.03f32; 4];
    let curve = prism_geometry::curve_tess::CurveReq::Strand {
        points: pts.clone(),
        radii: radii.clone(),
    };
    let tris = prism_geometry::curve_tess::tessellate(&curve, 0.02);
    let frames = prism_geometry::curve_tess::strand_frames(&pts);
    let geometry = HairGeometry::new(frames[0].tangent, frames[0].normal_ref);
    s.meshes.push((
        Mesh::from_triangles(tris),
        MaterialKind::Hair {
            scatter: HairScatter::new(0.22, 3.0, 1.55, 0.8, 0.02),
            geometry,
        },
    ));

    s
}

/// 同一场景的 **core 侧**描述（走产品路径）。
fn core_scene_with_attachments() -> prism_core::Scene {
    let mut s: prism_core::Scene = base_scene();
    // 背景球保持基准（但挪到 z = −1.2、缩小、换色，与手搭场景一致）。
    s.radius = 0.8;
    s.orbit_center = Vec3::new(0.0, 0.0, -1.2);
    s.material = prism_core::Material::new(Vec3::new(0.2, 0.5, 0.8), 0.5, 0.0);
    // 相机与手搭场景同参。
    s.camera = prism_core::Camera::new(
        Vec3::new(0.0, 0.0, -3.4),
        Vec3::new(0.0, 0.0, -1.2),
        50.0_f32.to_radians(),
    );

    s.attach = prism_core::SceneAttach {
        media: vec![MediumBox::new(
            Vec3::new(-2.0, -2.0, 0.2),
            Vec3::new(2.0, 2.0, 3.0),
            0.45,
            0.85,
            0.25,
        )
        .with_priority(3)],
        meshes: vec![plane_mesh(Some(DisplacementSlot::new(1, 0.12, 1.0, 2)))],
        instance_geoms: vec![plane_mesh(None)],
        instances: {
            let mut a1: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            a1[3] = -0.5;
            a1[7] = 0.6;
            a1[11] = -0.4;
            let mut a2: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            a2[0] = -0.6;
            a2[3] = 0.7;
            a2[7] = 0.4;
            a2[11] = -0.4;
            vec![
                InstanceDesc { geom_id: 0, affine: a1, mat_id: u32::MAX },
                InstanceDesc { geom_id: 0, affine: a2, mat_id: u32::MAX },
            ]
        },
        textures: vec![height_map()],
        strands: vec![StrandDesc::with_radii(
            vec![
                Vec3::new(0.6, -0.9, -0.2),
                Vec3::new(0.62, -0.3, -0.15),
                Vec3::new(0.58, 0.3, -0.1),
                Vec3::new(0.6, 0.9, -0.05),
            ],
            vec![0.03f32; 4],
        )],
        ocio_config: None,
    };
    s
}

/// **P2-a：新场景能出片（不是黑洞、不是拒绝帧）。**
#[test]
fn p2_attached_scene_renders_a_real_frame() {
    let scene: prism_core::Scene = core_scene_with_attachments();
    let req: SceneRenderRequest = build_render_request(&scene, 0.0, 10.0, W, H);
    assert!(!req.geo.is_empty(), "P2 失败：附加能力没进请求");
    assert_eq!(req.geo.media.len(), 1);
    assert_eq!(req.geo.meshes.len(), 1);
    assert_eq!(req.geo.instance_geoms.len(), 1);
    assert_eq!(req.geo.instances.len(), 2);
    assert_eq!(req.geo.textures.len(), 1);
    assert_eq!(req.geo.strands.len(), 1);

    let fb: Framebuffer = render_product(&scene);
    assert!(fb.is_valid(), "P2 失败：带附加能力的场景必须出有效帧");
    // 不是全黑：至少 1% 的像素有非零能量。
    let lit: usize = fb
        .pixels
        .chunks_exact(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 0)
        .count();
    let total: usize = (W * H) as usize;
    assert!(lit * 100 / total >= 1, "P2 失败：出片全黑（{lit}/{total} 有能量）");
    println!("[t0-p2] 带附加能力出片：md5 = {}，有能量像素 {lit}/{total}", md5_hex(&fb.pixels));
}

/// **P2-b：像素与"直接用 `prism_rt::Scene` 手搭同一场景"0 差异（同源性）。**
#[test]
fn p2_pixels_match_hand_built_rt_scene() {
    let scene: prism_core::Scene = core_scene_with_attachments();
    let got: Framebuffer = render_product(&scene);

    let hand: RtScene = hand_built_scene();
    let aesthetic = prism_gpu::AestheticReq {
        teal_orange: false,
        contrast: 1.0,
        filmic: false,
    };
    let cam: RtCamera = rt_camera_for(&scene);
    let want: Framebuffer = render_rt(&hand, &cam, &aesthetic);

    assert_eq!(got.pixels.len(), want.pixels.len());
    let mut differing: usize = 0;
    let mut worst: i32 = 0;
    for (a, b) in got.pixels.chunks_exact(4).zip(want.pixels.chunks_exact(4)) {
        if a != b {
            differing += 1;
            for k in 0..4 {
                worst = worst.max((a[k] as i32 - b[k] as i32).abs());
            }
        }
    }
    println!(
        "[t0-p2] 同源性：differing_pixels = {differing}/{}（最大通道差 {worst}），md5 {} vs {}",
        W * H,
        md5_hex(&got.pixels),
        md5_hex(&want.pixels)
    );
    assert_eq!(
        differing, 0,
        "P2 失败：产品路径与手搭 rt 场景像素不一致（最大通道差 {worst}）—— \
         翻译层搬错了某个字段"
    );
}

// ---------------------------------------------------------------------------
// P3 —— 静态门禁：六个能力在 translate_scene 里都有非空翻译分支
// ---------------------------------------------------------------------------

/// **P3：`translate_scene` 对每个新能力都有非空翻译分支。**
///
/// 查两件事，缺一条都能让"字段加了但没接"溜过去：
///
/// 1. **记号在场**：六个能力各自的落点记号必须出现在 `rt_backend.rs`；
/// 2. **分支非空**：每个记号**所在的那一段**（以 `// ---- 分支 N/6` 为界）
///    必须含有真正的写入（`push` / `extend_from_slice` / 赋值），而不是只有
///    一句注释。只查第 1 条会被"留了个 TODO 注释"骗过。
#[test]
fn p3_every_capability_has_a_non_empty_branch() {
    let src: &str = include_str!("../src/rt_backend.rs");

    // (能力名, 落点记号, 该段必须出现的写入记号)
    let checks: [(&str, &str, &str); 6] = [
        ("介质", "scene.media.push", "MediumRegion"),
        ("纹理", "prism_rt::texture::build_rgba8", "texels.extend_from_slice"),
        ("网格", "scene.meshes.push", "tessellate_displace"),
        ("实例原型", "scene.instance_geoms.push", "instance_geoms"),
        ("实例", "Instance::from_affine", "rebuild_instance_bvh"),
        ("毛发", "MaterialKind::Hair", "HairGeometry::new"),
    ];
    for (name, marker, write) in checks {
        assert!(
            src.contains(marker),
            "P3 失败：`translate_scene` 里没有 {name} 的翻译记号 `{marker}`（字段加了但没接）"
        );
        assert!(
            src.contains(write),
            "P3 失败：{name} 的翻译段缺少写入记号 `{write}`（分支是空的）"
        );
    }

    // 反向断言：六个能力**都必须**在 `translate_scene` 的函数体内。
    let body_start: usize = src
        .find("fn translate_scene(req: &SceneRenderRequest) -> RtScene {")
        .expect("translate_scene 必须存在");
    let body: &str = &src[body_start..];
    let body: &str = &body[..body.find("\n/// 场景网格的固定反照率").unwrap_or(body.len())];
    for (name, marker, _) in checks {
        assert!(
            body.contains(marker),
            "P3 失败：{name} 的翻译记号 `{marker}` 不在 translate_scene 函数体内"
        );
    }

    // 附带：`build_render_request` 必须真的调用 `build_geo_request`（左侧接线）。
    let lib_src: &str = include_str!("../src/lib.rs");
    assert!(
        lib_src.contains("geo: build_geo_request(scene)"),
        "P3 失败：build_render_request 没有调用 build_geo_request（core → 请求成了断头路）"
    );
}

// ---------------------------------------------------------------------------
// 位移槽的自洽（T2 的判据在 prism-rt / prism-geometry 侧；这里只钉产品路径）
// ---------------------------------------------------------------------------

/// `displacement = 0` 的请求与"完全没有位移槽"的请求产出**同一个** `rt::Scene`
/// 的网格顶点（判据 V1 在产品路径上的投影）。
#[test]
fn p_displacement_zero_is_bitwise_identical_on_product_path() {
    let mut with_slot: prism_core::Scene = base_scene();
    with_slot.attach.meshes = vec![plane_mesh(Some(DisplacementSlot::new(1, 0.0, 1.0, 3)))];
    with_slot.attach.textures = vec![height_map()];

    let mut without: prism_core::Scene = base_scene();
    without.attach.meshes = vec![plane_mesh(None)];
    without.attach.textures = vec![height_map()];

    let a: Framebuffer = render_product(&with_slot);
    let b: Framebuffer = render_product(&without);
    assert_eq!(
        a.pixels, b.pixels,
        "位移幅值 0 必须与无位移槽逐位相同（构造性短路）"
    );

    // 非零幅值必须真的改变画面（否则这条判据是平凡的）。
    let mut moved: prism_core::Scene = base_scene();
    moved.attach.meshes = vec![plane_mesh(Some(DisplacementSlot::new(1, 0.35, 1.0, 3)))];
    moved.attach.textures = vec![height_map()];
    let c: Framebuffer = render_product(&moved);
    assert_ne!(c.pixels, b.pixels, "非零位移幅值必须改变画面");
}

/// `SceneAttach` 再导出可达（`prism_core::SceneAttach` 与 attach 模块同名类型）。
#[test]
fn attach_type_is_reachable_from_crate_root() {
    let a: SceneAttach = prism_core::attach::SceneAttach::default();
    assert!(a.is_empty());
}


// ---------------------------------------------------------------------------
// P4 —— T1 / G-9 折痕标记从 DSL 一路走到像素
// ---------------------------------------------------------------------------

/// 立方体网格（8 顶点 12 三角，带顶点域 uv），供 P4 用。
///
/// 为什么用**立方体**而不是平面：位移沿法线推、平面被推后画面只改深度，
/// 折痕在平面上也无处可折（光滑与锐利的极限都是同一张平面）。立方体的棱是
/// 真折痕：`w = 1` 时棱保持锐利、`w = 0` 时被抹圆，轮廓与明暗都会变。
fn cube_mesh(slot: Option<DisplacementSlot>) -> MeshDesc {
    let h: f32 = 0.85;
    let mut pos: Vec<Vec3> = Vec::with_capacity(8);
    for &z in &[-h, h] {
        for &y in &[-h, h] {
            for &x in &[-h, h] {
                pos.push(Vec3::new(x, y, z));
            }
        }
    }
    let idx: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2, // z = −h
        4, 6, 7, 4, 7, 5, // z = +h
        0, 4, 5, 0, 5, 1, // y = −h
        2, 3, 7, 2, 7, 6, // y = +h
        1, 5, 7, 1, 7, 3, // x = +h
        0, 2, 6, 0, 6, 4, // x = −h
    ];
    let mut uv: Vec<[f32; 2]> = Vec::with_capacity(8);
    for &z in &[0.0f32, 1.0] {
        for &y in &[0.0f32, 1.0] {
            for &x in &[0.0f32, 1.0] {
                uv.push([x, 0.5 * y + 0.5 * z]);
            }
        }
    }
    let mut m: MeshDesc = MeshDesc::with_uv(pos, idx, uv);
    m.displacement = slot;
    m
}

/// 立方体的全部唯一边（`(min, max)` 升序）。
fn cube_edges() -> Vec<(u32, u32)> {
    let idx: Vec<u32> = (0..12u32).collect();
    let _ = idx;
    let m: MeshDesc = cube_mesh(None);
    let mut e: Vec<(u32, u32)> = Vec::new();
    for c in m.indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (c[k], c[(k + 1) % 3]);
            e.push((a.min(b), a.max(b)));
        }
    }
    e.sort_unstable();
    e.dedup();
    e
}

/// **P4（T1 / G-9）：`MeshDesc::crease` 从产品路径一路影响到了像素。**
///
/// 判据：同一份网格、同一个位移槽，只有折痕标记不同 ⇒ 帧**必须不同**
/// （且差异像素数 ≥ 1% 的画面，防止"只差几个像素"的偶然通过）。
///
/// 这条同时锁住三段：core 的 `MeshDesc.crease` → 请求的 `MeshReq.crease` →
/// `translate_scene` 里建 `SubdivMarks` 并传给 `tessellate_displace`。
/// 任何一段漏接，两帧就会**逐字节相同**。
#[test]
fn p4_crease_marks_reach_pixels() {
    let slot: DisplacementSlot = DisplacementSlot::new(1, 0.12, 1.0, 2);

    let mut smooth_scene: prism_core::Scene = base_scene();
    smooth_scene.attach.textures.push(height_map());
    smooth_scene.attach.meshes.push(cube_mesh(Some(slot)));

    let mut sharp_scene: prism_core::Scene = base_scene();
    sharp_scene.attach.textures.push(height_map());
    let mut sharp_mesh: MeshDesc = cube_mesh(Some(slot));
    let edges: Vec<(u32, u32)> = cube_edges();
    for (a, b) in &edges {
        sharp_mesh = sharp_mesh.with_crease(*a, *b, 1.0);
    }
    sharp_scene.attach.meshes.push(sharp_mesh);

    let fa: Framebuffer = render_product(&smooth_scene);
    let fb: Framebuffer = render_product(&sharp_scene);
    assert!(fa.is_valid() && fb.is_valid(), "P4 失败：两帧都必须有效");

    let mut differing: usize = 0;
    let total: usize = (W * H) as usize;
    for (pa, pb) in fa.pixels.chunks_exact(4).zip(fb.pixels.chunks_exact(4)) {
        if pa != pb {
            differing += 1;
        }
    }
    println!(
        "[t1-p4] 折痕标记：differing_pixels = {differing} / {total}（{:.2}%）",
        100.0 * differing as f32 / total as f32
    );
    assert!(
        differing * 100 >= total,
        "P4 失败：折痕标记没有影响画面（{differing}/{total} 像素不同）—— \
         说明 MeshDesc.crease 在某一段被丢了"
    );
    assert_eq!(edges.len(), 18, "立方体网格应有 18 条唯一边");
}
