//! **T-B / R-1 验收测试**：路径模式 WGSL wavefront ↔ CPU 参考**逐位**对拍。
//!
//! # 判据（比统计更严：`to_bits` 全等）
//!
//! ```text
//! CPU 权威：prism_rt::wavefront::render_frame（SampleStream 真 Joe–Kuo Sobol
//!           + 64 位 PCG + Möller–Trumbore + Sphere NEE/MIS + RR）
//! GPU 被测：rt_wavefront.wgsl 五段 compute（llvmpipe / lavapipe 真机派发，
//!           同表同流：sobol_table 由 sobol_u32(d, 1<<k) 生成上传）
//! 判据：  逐像素 RGB `to_bits()` 全等（differing_pixels == 0）
//! ```
//!
//! # 场景类（R-1 判据范围，如实声明）
//!
//! 三角网格（两个 Lambert 材质）+ Sphere 灯（**不注册为几何** ⇒ `light_skip`
//! 为 None、发光命中段不触发）+ 针孔相机（零长快门）+ 无环境图 / 介质 / 折射体
//! / 间接钳位（`clamp_indirect = None`）。`MAT_PBR` 的 Disney 续弹是方向提案
//! 近似，不进本判据（shader 内已逐处标注）。
//!
//! # 环境自门控（禁 `#[ignore]`）
//!
//! 无 Vulkan 设备时打印探测报告后返回（不 fail、不假装跑过）。

use prism_core::EvalCtx;
use prism_gpu::vulkan::wavefront::{
    WavefrontLight, WavefrontMaterial, WavefrontMode, WavefrontRequest, WavefrontTri,
};
use prism_gpu::vulkan::VulkanBackend;
use prism_gpu::AestheticReq;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{MaterialKind, RenderParams, Scene};
use prism_rt::light::Light;

/// 设备自门控（与 `qa_wavefront.rs::device_ready` 同口径）。
pub fn device_ready(backend: &VulkanBackend) -> bool {
    let report = VulkanBackend::probe();
    if !report.device_created {
        println!(
            "[skip] 本机无可用 Vulkan 设备：reason={:?} → 跳过 R-1 真机判据（不 panic）",
            report.reason
        );
        return false;
    }
    println!(
        "[device] {}（枚举 {} 个物理设备）",
        backend.device_name().unwrap_or("<unnamed>"),
        report.device_count
    );
    true
}

/// 真 Joe–Kuo 方向数表（64×32 u32 小端字节流）。
///
/// `direction_table()[d][k] == sobol_u32(d, 1<<k)`（GF(2) 异或的单置位退化），
/// 故由公开的 [`prism_rt::sampler::sobol_u32`] 生成即与 CPU 采样器**同表**——
/// 这是逐位对拍的前提（内置 PCG 混洗表与 CPU 不同流）。
pub fn sobol_table_bytes() -> Vec<u8> {
    let mut out = vec![0u8; 64 * 32 * 4];
    for d in 0..64u32 {
        for k in 0..32u32 {
            let v = prism_rt::sampler::sobol_u32(d as u16, 1u32 << k);
            let o = ((d * 32 + k) * 4) as usize;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// 把 `prism_rt::Camera` 的公开字段打包成 96 字节相机 uniform（逐位同源）。
///
/// 槽位 = `rt_wavefront.wgsl::Camera`：eye@0 / forward@16 / right@32 / up@48 /
/// tan_half_fov@64 / aperture@68 / focus@72 / shutter_open@76 / shutter_close@80 /
/// aspect@84。基向量与 `tan_half_fov`（`sampler::tan` 多项式）直接取 CPU 值的
/// 位型——不经过 `det_trig::tan` 第二套三角。
pub fn camera_uniform(cam: &Camera, aspect: f32) -> [u8; 96] {
    let mut out = [0u8; 96];
    let put = |out: &mut [u8], off: usize, v: f32| {
        out[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    };
    for (k, c) in [cam.eye, cam.forward, cam.right, cam.up].iter().enumerate() {
        let off = k * 16;
        put(&mut out, off, c.x);
        put(&mut out, off + 4, c.y);
        put(&mut out, off + 8, c.z);
    }
    put(&mut out, 64, cam.tan_half_fov);
    put(&mut out, 68, cam.aperture_radius);
    put(&mut out, 72, cam.focus_distance);
    put(&mut out, 76, cam.shutter_open);
    put(&mut out, 80, cam.shutter_close);
    put(&mut out, 84, aspect);
    out
}

/// R-1 场景（CPU 侧）：起伏地板 + 后墙（两个 Lambert 网格）+ Sphere 灯
/// （不注册为几何）。高度用整数算术生成（确定性、无 libm）。
pub fn parity_scene() -> Scene {
    let mut scene = Scene::default();
    // 地板：y = 0 的两三角形，法线 +y（绕序 a→b→c 使 cross(e1,e2) = +y）。
    let floor_pts = vec![
        Vec3::new(-1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(-1.0, 0.0, -1.0),
    ];
    let floor_idx = vec![0u32, 1, 2, 0, 2, 3];
    let m0 = scene.add_mesh(
        floor_pts,
        floor_idx,
        MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) },
    );
    // 后墙：z = 1.5 的两三角形，法线 −z（朝向房间内侧）。
    let wall_pts = vec![
        Vec3::new(-1.0, 0.0, 1.5),
        Vec3::new(1.0, 2.0, 1.5),
        Vec3::new(1.0, 0.0, 1.5),
        Vec3::new(-1.0, 2.0, 1.5),
    ];
    let wall_idx = vec![0u32, 1, 2, 0, 3, 1];
    let m1 = scene.add_mesh(
        wall_pts,
        wall_idx,
        MaterialKind::Lambert { albedo: Vec3::new(0.2, 0.5, 0.3) },
    );
    assert_eq!((m0, m1), (0, 1), "网格序 = 三角形全局序的前提");
    // Sphere 灯：只进灯光表（NEE 用），**不注册为几何**（light_skip = None，
    // 发光命中段不触发——R-1 场景类契约）。
    scene.lights = vec![Light::Sphere {
        center: Vec3::new(0.0, 1.6, 0.3),
        radius: 0.3,
        radiance: Vec3::new(14.0, 12.0, 9.0),
    }];
    scene
}

/// 与 [`parity_scene`] 同一几何的 GPU 请求（三角形/材质/灯逐位同源）。
pub fn parity_request(w: u32, h: u32, spp: u32, max_bounces: u32) -> WavefrontRequest {
    let lam = |r: f32, g: f32, b: f32| WavefrontMaterial {
        base_color: Vec3::new(r, g, b),
        roughness: 0.0,
        f0: Vec3::splat(0.04),
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.5,
        emissive: Vec3::splat(0.0),
        kind: 0, // Lambert
    };
    let tri = |a: Vec3, b: Vec3, c: Vec3| WavefrontTri { a, b, c };
    WavefrontRequest {
        width: w,
        height: h,
        spp,
        max_bounces,
        seed: 2026,
        mode: WavefrontMode::Path,
        camera: prism_gpu::CameraReq {
            eye: Vec3::new(0.0, 0.8, -2.5),
            target: Vec3::new(0.0, 0.4, 0.5),
            fov: 55.0_f32 * core::f32::consts::PI / 180.0,
            aspect: w as f32 / h as f32,
        },
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        spheres: Vec::new(),
        triangles: vec![
            // 地板（与 CPU add_mesh 的三角形序一致：索引流顺序）
            tri(Vec3::new(-1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, -1.0)),
            tri(Vec3::new(-1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(-1.0, 0.0, -1.0)),
            // 后墙
            tri(Vec3::new(-1.0, 0.0, 1.5), Vec3::new(1.0, 2.0, 1.5), Vec3::new(1.0, 0.0, 1.5)),
            tri(Vec3::new(-1.0, 0.0, 1.5), Vec3::new(-1.0, 2.0, 1.5), Vec3::new(1.0, 2.0, 1.5)),
        ],
        materials: vec![lam(0.72, 0.71, 0.69), lam(0.2, 0.5, 0.3)],
        prim_mat: vec![0, 0, 1, 1],
        lights: vec![WavefrontLight {
            position: Vec3::new(0.0, 1.6, 0.3),
            radius: 0.3,
            color: Vec3::new(14.0, 12.0, 9.0),
        }],
        ambient: Vec3::splat(0.0),
        env_blob: Vec::new(),
        sobol_table: Some(sobol_table_bytes()),
        node_id: 7,
        frame: 3,
        camera_uniform: Some(camera_uniform(
            &Camera::new(
                Vec3::new(0.0, 0.8, -2.5),
                Vec3::new(0.0, 0.4, 0.5),
                Vec3::new(0.0, 1.0, 0.0),
                55.0,
                0.0,
                3.1,
                0.0,
                0.0,
            ),
            w as f32 / h as f32,
        )),
        stage_stop: None,
        readback_slots: false,
    }
}

/// CPU 参考帧（`prism_rt::wavefront::render_frame`；clamp_indirect 关闭）。
pub fn cpu_reference(w: u32, h: u32, spp: u32, max_bounces: u32) -> Vec<[f32; 4]> {
    let scene = parity_scene();
    let cam = Camera::new(
        Vec3::new(0.0, 0.8, -2.5),
        Vec3::new(0.0, 0.4, 0.5),
        Vec3::new(0.0, 1.0, 0.0),
        55.0,
        0.0,
        3.1,
        0.0,
        0.0,
    );
    let ctx = EvalCtx::new(2026, 7, 3);
    let mut params = RenderParams::new(spp, max_bounces);
    params.clamp_indirect = None; // R-1 场景类：无间接钳位（GPU 侧同关）
    prism_rt::wavefront::render_frame(&scene, &cam, &ctx, (w, h), &params)
}

/// 逐位比对 + 差异定位（首差异像素的 RGB 位型）。
pub fn assert_bitwise_equal(w: u32, h: u32, cpu: &[[f32; 4]], gpu: &[[f32; 4]], label: &str) {
    assert_eq!(cpu.len(), (w * h) as usize, "{label}：CPU 帧长不符");
    assert_eq!(gpu.len(), (w * h) as usize, "{label}：GPU 帧长不符");
    let mut differing = 0usize;
    let mut first: Option<(u32, u32, [u32; 3], [u32; 3])> = None;
    for i in 0..cpu.len() {
        let a = [cpu[i][0].to_bits(), cpu[i][1].to_bits(), cpu[i][2].to_bits()];
        let b = [gpu[i][0].to_bits(), gpu[i][1].to_bits(), gpu[i][2].to_bits()];
        if a != b {
            differing += 1;
            if first.is_none() {
                first = Some((
                    (i as u32) % w,
                    (i as u32) / w,
                    a,
                    b,
                ));
            }
        }
    }
    println!("[{label}] {w}×{h} differing_pixels = {differing}/{}", cpu.len());
    assert_eq!(
        differing, 0,
        "{label}：R-1 逐位对拍失败，首差异像素 {:?}（cpu bits vs gpu bits）",
        first
    );
}

// ---------------------------------------------------------------------------
// R-6 缺口⑤闭合：曲线图元经三角化进入 wavefront 路径模式
// ---------------------------------------------------------------------------

/// 曲线场景（R-6）：柱状 `CurveReq::Cylinder`（场景中央竖柱）+ parity 地板与
/// 后墙。柱子经 [`prism_geometry::curve_tess::tessellate`] 三角化后**作为三角
/// 网格**进两侧（CPU `Scene::add_mesh` / GPU `WavefrontRequest.triangles`）——
/// 同一份三角形流（本函数一次生成，两侧共享），对拍由此成立。
///
/// 这条闭合的是「wavefront 路径模式 kernel 的 scene_intersect 只吃三角形，
/// 曲线段不在本 kernel」：等价路径 = 曲线先三角化（产品路径 `translate_scene`
/// 对曲线做的正是这一步），wavefront 吃三角化结果与 CPU 逐位一致。
pub fn curve_scene_tris(tol: f32) -> Vec<prism_geometry::bvh::Triangle> {
    use prism_geometry::curve_tess::{tessellate, CurveReq};
    let cyl = CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.7, 0.2),
        radius: 0.35,
        height: 1.4,
        axis: Vec3::new(0.0, 1.0, 0.0),
    };
    tessellate(&cyl, tol)
}

/// 与 [`parity_request`] 同基座、追加曲线三角的 GPU 请求。
///
/// 材质表追加第 3 个 Lambert（柱子），`prim_mat` 追加对应的逐三角形映射。
pub fn curve_request(w: u32, h: u32, spp: u32, max_bounces: u32, tol: f32) -> WavefrontRequest {
    let mut req = parity_request(w, h, spp, max_bounces);
    let tris = curve_scene_tris(tol);
    let mat_base = req.materials.len() as u32;
    req.materials.push(WavefrontMaterial {
        base_color: Vec3::new(0.85, 0.55, 0.15),
        roughness: 0.0,
        f0: Vec3::splat(0.04),
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.5,
        emissive: Vec3::splat(0.0),
        kind: 0, // Lambert
    });
    for t in &tris {
        req.triangles.push(WavefrontTri { a: t.a, b: t.b, c: t.c });
        req.prim_mat.push(mat_base);
    }
    req
}

/// 曲线场景的 CPU 参考帧（同几何同材质，`prism_rt::wavefront::render_frame`）。
pub fn curve_cpu_reference(w: u32, h: u32, spp: u32, max_bounces: u32, tol: f32) -> Vec<[f32; 4]> {
    use prism_core::EvalCtx;
    use prism_rt::camera::Camera;
    use prism_rt::integrator::{MaterialKind, RenderParams};
    use prism_math::Vec3 as V3;

    // 从 parity_scene 起步（地板 + 后墙 + Sphere 灯），追加柱子网格。
    let mut scene = parity_scene();
    let tris = curve_scene_tris(tol);
    let mut positions: Vec<V3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for t in &tris {
        let base = positions.len() as u32;
        positions.push(t.a);
        positions.push(t.b);
        positions.push(t.c);
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    scene.add_mesh(
        positions,
        indices,
        MaterialKind::Lambert { albedo: V3::new(0.85, 0.55, 0.15) },
    );
    let cam = Camera::new(
        V3::new(0.0, 0.8, -2.5),
        V3::new(0.0, 0.4, 0.5),
        V3::new(0.0, 1.0, 0.0),
        55.0,
        0.0,
        3.1,
        0.0,
        0.0,
    );
    let ctx = EvalCtx::new(2026, 7, 3);
    let mut params = RenderParams::new(spp, max_bounces);
    params.clamp_indirect = None;
    prism_rt::wavefront::render_frame(&scene, &cam, &ctx, (w, h), &params)
}
