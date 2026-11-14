//! 通用渲染路径的「内容无关」回归（2026-11-14 裁定三）。
//!
//! 裁定原文：介质 / 位移网格 / 实例 / 毛发 / 贴图这些附加能力路径，必须走和
//! 普通场景**同一条**通用渲染路径，禁止按场景内容分流回退——不许出现
//! 「无介质无贴图才走快路径、带这些就退回老路」的判定。
//!
//! 本文件把这条裁定变成可执行的布尔：
//! 1. 同一档位下，带全套附加能力的请求与不带任何附加能力的请求，
//!    都必须走**同一条**通用路径（`last_fast_path == true`，即
//!    `rt_tiles` 的稀疏网格 + 门控重建内核），且都不被拒绝；
//! 2. 路线只由**档位**决定，与场景内容无关（`一致` 档两请求都 false）；
//! 3. 带能力的请求必须真出画面（非黑、非单色、逐帧推进、两次运行逐字节
//!    一致），不存在"带能力 ⇒ 返回空帧/静默降级"的后门。

use prism_core::attach::{
    DisplacementSlot, InstanceDesc, MediumBox, MeshDesc, StrandDesc, TextureDesc,
};
use prism_core::Scene;
use prism_gpu::{Framebuffer, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;
use prism_render::rt_backend::RtBackend;
use prism_render::{build_render_request, V31Config, V31Mode, V31RenderDevice};

/// 8×8 棋盘高度图（RGBA8，线性域）——只驱动位移几何，不当材质贴图用。
fn height_map() -> TextureDesc {
    let (w, h): (u32, u32) = (8, 8);
    let mut rgba: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let on: bool = ((x / 2) + (y / 2)) % 2 == 0;
            let v: u8 = if on { 255 } else { 0 };
            rgba.extend_from_slice(&[v, v, v, 255]);
        }
    }
    TextureDesc::new(w, h, false, rgba)
}

/// 一张 2×2 的 XZ 平面（4 顶点 / 2 三角形），带 uv。
fn plane_mesh(displace: Option<DisplacementSlot>) -> MeshDesc {
    MeshDesc {
        positions: vec![
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        uv: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        displacement: displace,
        crease: Vec::new(),
        corners: Vec::new(),
    }
}

/// 带全套附加能力的 core 场景（贴图 + 位移网格 + 实例 + 介质 + 毛发）。
fn capable_scene() -> Scene {
    let mut s: Scene = Scene::default();
    s.radius = 0.45;
    s.lights.push(prism_core::Light::new(
        Vec3::new(0.4, -1.0, -0.6).normalize(),
        Vec3::new(1.0, 1.0, 1.0),
        3.2,
    ));
    // 贴图：位移要采样它（句柄 = 1，声明序 + 1）。
    s.attach.textures.push(height_map());
    // 介质：一块薄雾（互斥区域 + priority）。
    s.attach.media.push(MediumBox {
        bounds_min: Vec3::new(-2.5, -2.5, -1.0),
        bounds_max: Vec3::new(2.5, 0.6, 2.5),
        sigma_t: 0.35,
        albedo: 0.85,
        g: 0.2,
        priority: 1,
    });
    // 位移网格（细分 2 级后按贴图亮度位移）。
    s.attach
        .meshes
        .push(plane_mesh(Some(DisplacementSlot::new(1, 0.12, 1.0, 2))));
    // 实例：同一份原型几何平移两份。
    s.attach.instance_geoms.push(plane_mesh(None));
    for dx in [-1.4_f32, 1.4] {
        s.attach.instances.push(InstanceDesc {
            geom_id: 0,
            affine: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            mat_id: u32::MAX,
        });
        let last = s.attach.instances.len() - 1;
        // 平移写入第 3 列（行主序 3×4 的 tx/ty/tz）。
        s.attach.instances[last].affine[3] = dx;
        s.attach.instances[last].affine[7] = 0.9;
        s.attach.instances[last].affine[11] = 0.0;
    }
    // 毛发：两股同向纤维。
    for dx in [-0.25_f32, 0.25] {
        s.attach.strands.push(StrandDesc::new(
            vec![
                Vec3::new(dx, -0.5, 0.2),
                Vec3::new(dx, 0.0, 0.2),
                Vec3::new(dx, 0.5, 0.2),
            ],
            0.02,
        ));
    }
    s
}

/// 同一相机 / 灯 / 主体，但清空全部附加能力。
fn plain_scene() -> Scene {
    let mut s: Scene = capable_scene();
    s.attach = Default::default();
    s
}

/// 渲染是**单帧串行**的生产语义，但本文件的多条用例会被 cargo 并行调度。
/// 重建阈值 / 阴影档 / 介质档是帧级作用域的**进程静态量**（第二遍跨线程必须
/// 共享同一组阈值），两条渲染交错就会互相看到对方的阈值。这里把渲染串行化：
/// 争的是测试调度，不是渲染确定性——生产链路一次只渲染一帧。
static RENDER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn render(
    mode: V31Mode,
    req: &SceneRenderRequest,
    w: u32,
    h: u32,
) -> (Framebuffer, bool) {
    let _serial: std::sync::MutexGuard<'_, ()> =
        RENDER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut inner: RtBackend = RtBackend::new(16, 6);
    let mut cfg: V31Config = mode.preset(prism_render::ResTier::R320x180).expect("档位预设");
    cfg.width = w;
    cfg.height = h;
    let mut dev: V31RenderDevice<'_> = V31RenderDevice::new(&mut inner, cfg, w, h);
    let fb: Framebuffer = dev.render_scene(req);
    (fb, dev.last_fast_path())
}

const W: u32 = 96;
const H: u32 = 54;

#[test]
fn capability_scene_takes_the_same_general_path_as_plain_scene() {
    let capable: SceneRenderRequest = build_render_request(&capable_scene(), 0.0, 1.0, W, H);
    let plain: SceneRenderRequest = build_render_request(&plain_scene(), 0.0, 1.0, W, H);
    assert_eq!(capable.geo.media.len(), 1, "介质槽必须真的搬进请求");
    assert_eq!(capable.geo.textures.len(), 1, "贴图槽必须真的搬进请求");
    assert!(!capable.geo.meshes.is_empty(), "网格槽必须真的搬进请求");
    assert!(!capable.geo.instances.is_empty(), "实例槽必须真的搬进请求");
    assert!(!capable.geo.strands.is_empty(), "毛发槽必须真的搬进请求");
    assert!(plain.geo.media.is_empty() && plain.geo.strands.is_empty());

    for mode in [V31Mode::Balanced, V31Mode::Fast] {
        let (fb_cap, fast_cap) = render(mode, &capable, W, H);
        let (fb_plain, fast_plain) = render(mode, &plain, W, H);
        assert!(
            fast_cap,
            "{mode:?}：带全套附加能力也必须走通用路径（禁止按内容回退老路）"
        );
        assert!(
            fast_plain,
            "{mode:?}：无附加能力同样走通用路径（两条请求同一路线）"
        );
        assert!(fb_cap.is_valid(), "{mode:?}：带能力请求不得被拒绝（返回空帧）");
        assert!(fb_plain.is_valid(), "{mode:?}：普通请求不得被拒绝");
        assert_eq!((fb_cap.width, fb_cap.height), (W, H));
    }
}

#[test]
fn route_is_decided_by_mode_only_not_by_scene_content() {
    let capable: SceneRenderRequest = build_render_request(&capable_scene(), 0.0, 1.0, W, H);
    let plain: SceneRenderRequest = build_render_request(&plain_scene(), 0.0, 1.0, W, H);
    // `一致` 档两请求都透传（false），`均衡/快速` 两请求都走通用路径（true）。
    for (mode, want) in [
        (V31Mode::Consistent, false),
        (V31Mode::Balanced, true),
        (V31Mode::Fast, true),
    ] {
        let (_, fast_cap) = render(mode, &capable, W, H);
        let (_, fast_plain) = render(mode, &plain, W, H);
        assert_eq!(fast_cap, want, "{mode:?}：带能力请求的路线");
        assert_eq!(fast_plain, want, "{mode:?}：普通请求的路线");
    }
}

#[test]
fn capability_scene_really_renders_a_moving_non_trivial_frame() {
    // 能力回归（裁定二）：出得了片、非全黑、非单色、逐帧推进、两次跑逐字节一致。
    let req: SceneRenderRequest = build_render_request(&capable_scene(), 0.0, 1.0, W, H);
    let (a, _) = render(V31Mode::Balanced, &req, W, H);
    let (b, _) = render(V31Mode::Balanced, &req, W, H);
    let diff: usize = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .filter(|(x, y)| x != y)
        .count();
    eprintln!("[route] 两次渲染差异字节数 = {diff}");
    assert_eq!(a.pixels, b.pixels, "同帧两次渲染必须逐字节一致");

    let distinct: std::collections::BTreeSet<[u8; 3]> = a
        .pixels
        .chunks_exact(4)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    assert!(distinct.len() > 8, "带能力场景不得退化成单色/纯黑：{} 种颜色", distinct.len());
    assert!(
        a.pixels.chunks_exact(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0),
        "带能力场景必须至少有一个非黑像素"
    );

    // 时间推进：相机绕行后画面必须变（不是把一帧复制成整段）。
    let later: SceneRenderRequest = build_render_request(&capable_scene(), 0.4, 1.0, W, H);
    let (c, _) = render(V31Mode::Balanced, &later, W, H);
    assert_ne!(a.pixels, c.pixels, "相机推进后画面必须变化（禁止重复帧冒充逐帧实渲）");
}

// ---------------------------------------------------------------------------
// 能力回归（裁定二）：`examples/t0_media_mesh.prism` 必须出片
// ---------------------------------------------------------------------------
//
// 裁定一已把 t0 移出 8.2 速度门槛与 8.1 像素比对（它的 16 spp golden 在体积
// 介质里是噪声解），但裁定二保留它为**能力回归项**：退出码 0、帧数正确、
// 画面非全黑且非全同、不 panic、不判红。本用例把这条钉住。

#[test]
fn t0_media_mesh_renders_through_the_general_path() {
    let src: &str = include_str!("../../../examples/t0_media_mesh.prism");
    let compile: prism_dsl::CompileV2Output =
        prism_dsl::compile_v2(src).expect("t0_media_mesh 应编译成功");
    let (w, h): (u32, u32) = (compile.width, compile.height);
    let frames: u32 = (compile.fps as f32 * compile.duration).ceil() as u32;
    assert!(frames >= 2, "t0 至少要 2 帧才能验证时间推进");

    let mut cfg: V31Config = V31Mode::Balanced
        .preset(prism_render::ResTier::Custom(w, h))
        .expect("均衡档预设");
    cfg.width = w;
    cfg.height = h;
    cfg.fps = compile.fps;
    cfg.frames = frames;

    let mut inner: RtBackend = RtBackend::new(16, 6);
    let mut sink: prism_render::NullV31Sink = prism_render::NullV31Sink;
    let _serial: std::sync::MutexGuard<'_, ()> =
        RENDER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = prism_render::render_movie_v31(&compile, &mut inner, &cfg, &mut sink)
        .expect("t0 出片不得报错（能力回归项）");
    drop(_serial);

    assert_eq!(out.frames.len(), frames as usize, "帧数必须正确");
    let mut any_lit: bool = false;
    let mut max_colors: usize = 0;
    for (i, fb) in out.frames.iter().enumerate() {
        assert_eq!((fb.width, fb.height), (w, h), "第 {i} 帧尺寸");
        assert_eq!(fb.pixels.len(), (w * h * 4) as usize, "第 {i} 帧长度");
        let colors: std::collections::BTreeSet<[u8; 3]> = fb
            .pixels
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        max_colors = max_colors.max(colors.len());
        any_lit |= fb
            .pixels
            .chunks_exact(4)
            .any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    }
    assert!(any_lit, "t0 不得全黑");
    assert!(max_colors > 8, "t0 不得退化成单色：最多 {} 种颜色", max_colors);
    assert_ne!(
        out.frames[0].pixels,
        out.frames[out.frames.len() - 1].pixels,
        "t0 首末帧必须不同（禁止重复帧冒充逐帧实渲）"
    );
}
