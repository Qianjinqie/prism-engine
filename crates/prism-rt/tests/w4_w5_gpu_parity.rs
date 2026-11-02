//! 三后端对拍：W5 的 V1（毛股 ≡ 圆柱，`differing_pixels = 0`）与 W4 的
//! "细分 + 位移"烘焙帧（`differing_pixels = 0`）。
//!
//! # 判据纪律
//!
//! 软件 ICD（llvmpipe / lavapipe / SwiftShader）上 `max_channel_delta == 0`
//! **且** `differing_pixels == 0` 是硬判据，一个字都不放宽。无设备时打印一句
//! 并跳过（实机项永不进门禁）。
//!
//! # 为什么 W4 的帧能走这条通路（零 ABI 改动）
//!
//! 两个后端消费的是**同一条** `TRIANGLE_SLOTS` 通路：`CurveReq` →
//! `prism_geometry::curve_tess::tessellate_all` → 三角形 + SAH-BVH →
//! 扁平化成 `array<f32>` 槽位。细分/位移的产物用一个 `CurveReq::TriSoup`
//! 变体原样进入（该变体的 `tessellate` 就是恒等），**没有新绑定、没有新的
//! 遍历代码、没有着色器改动**（`scene_abi.rs` 的 V2 布局一个字节都没动）。

use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, CurveReq, Framebuffer, GpuDevice, LightReq,
    SceneRenderRequest, SphereReq,
};
use prism_geometry::subdiv::subdivide;
use prism_math::Vec3;
use prism_rt::displace::{bake_triangles, displace_positions, DisplaceParams};
use prism_rt::integrator::Scene;
use prism_rt::texsample::TexTable;

/// 是否软件 ICD（此时判据是逐位一致）。
fn is_software_icd(device: &str) -> bool {
    let d = device.to_ascii_lowercase();
    d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
}

/// 打开 Vulkan 后端；无可用设备 ⇒ `None`（打印后跳过）。
fn gpu_backend() -> Option<prism_gpu::vulkan::VulkanBackend> {
    let be = prism_gpu::vulkan::VulkanBackend::new();
    if !be.render_available() {
        println!("无可用设备：0 差异判据跳过（实机项 = 永不进门禁）");
        return None;
    }
    Some(be)
}

/// 帧差统计（与 `qa_curve_parity.rs` 同口径）。
struct Parity {
    total_pixels: usize,
    max_channel_delta: u32,
    differing_pixels: usize,
    first: Option<(usize, [u8; 4], [u8; 4])>,
}

fn parity(a: &Framebuffer, b: &Framebuffer) -> Parity {
    assert_eq!((a.width, a.height), (b.width, b.height), "帧尺寸必须一致");
    let mut p = Parity {
        total_pixels: (a.width * a.height) as usize,
        max_channel_delta: 0,
        differing_pixels: 0,
        first: None,
    };
    for i in 0..p.total_pixels {
        let pa = &a.pixels[i * 4..i * 4 + 4];
        let pb = &b.pixels[i * 4..i * 4 + 4];
        let mut d = 0u32;
        for c in 0..4 {
            d = d.max((pa[c] as i32 - pb[c] as i32).unsigned_abs());
        }
        if d > 0 {
            p.differing_pixels += 1;
            if p.first.is_none() {
                p.first = Some((i, [pa[0], pa[1], pa[2], pa[3]], [pb[0], pb[1], pb[2], pb[3]]));
            }
        }
        p.max_channel_delta = p.max_channel_delta.max(d);
    }
    p
}

fn base_request(w: u32, h: u32) -> SceneRenderRequest {
    SceneRenderRequest {
        width: w,
        height: h,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.0, -4.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![LightReq {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }],
        spheres: vec![SphereReq {
            center: Vec3::new(0.0, 2.6, 0.0),
            radius: 0.3,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4,
            metalness: 0.1,
            maps: prism_gpu::NO_MAPS,
        }],
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        curves: Vec::new(),
    }
}

fn is_background(px: [u8; 4]) -> bool {
    px[0] <= 12 && px[1] <= 15 && px[2] <= 22
}

/// **W5 / V1：单股（等半径直管）与等价的既有曲线族逐位一致。**
///
/// 判据原文写 `Cone`，那是笔误——`Cone` 的半径从 0 线性长到 `radius`，与等半径
/// 直管几何上不等价；等半径直管的对应物是 `Cylinder`。本测试按**几何等价**选
/// `Cylinder`（同一中心、同一半径、同一高度、同一轴），理由写在报告 §21 的 V1 行。
///
/// 三层断言，从强到弱：
/// 1. `CpuRefBackend` 渲两版帧 → **逐字节相同**（三角化逐位一致的直接结果）；
/// 2. Vulkan(llvmpipe) vs `CpuRefBackend` → `differing_pixels = 0`（判据本体）；
/// 3. 帧非平凡（中心像素被覆盖）——防止"两边都空"也能拿 0 差异。
#[test]
fn v1_strand_matches_cylinder_bitwise_on_three_backends() {
    // 轴取 +Z（正对相机），半径 0.9、高 3.0，与 `qa_curve_parity` 的圆柱用例同参。
    let cyl = CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.9,
        height: 3.0,
        axis: Vec3::new(0.0, 0.0, 1.0),
    };
    let strand = CurveReq::Strand {
        points: vec![Vec3::new(0.0, 0.0, -1.5), Vec3::new(0.0, 0.0, 1.5)],
        radii: vec![0.9],
    };
    let mut req_c = base_request(24, 16);
    req_c.curves = vec![cyl];
    let mut req_s = base_request(24, 16);
    req_s.curves = vec![strand];

    // (1) CPU 参考：两版帧逐字节相同。
    let mut cpu = CpuRefBackend;
    let fc = cpu.render_scene(&req_c);
    let fs = cpu.render_scene(&req_s);
    assert!(fc.is_valid() && fs.is_valid());
    let p = parity(&fc, &fs);
    println!(
        "V1[CPU]：毛股 vs 圆柱 pixels={} max_channel_delta={} differing_pixels={}",
        p.total_pixels, p.max_channel_delta, p.differing_pixels
    );
    assert_eq!(p.max_channel_delta, 0, "CPU 参考上两版帧必须逐字节相同");
    assert_eq!(p.differing_pixels, 0, "CPU 参考上两版帧必须逐字节相同");
    // (3) 非平凡：中心被几何覆盖。
    let center = fs.get_pixel(12, 8).expect("中心像素可读");
    assert!(!is_background(center), "毛股应覆盖画面中心，实读 {center:?}");

    // (2) Vulkan（软件 ICD）vs CPU 参考。
    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    for (tag, req) in [("cylinder", &req_c), ("strand", &req_s)] {
        let gpu = backend.render_scene(req);
        assert!(gpu.is_valid(), "[{tag}] GPU 必须产出有效帧");
        let cpu_fb = cpu.render_scene(req);
        let p = parity(&cpu_fb, &gpu);
        println!(
            "[{tag}] pixels={} max_channel_delta={} differing_pixels={} software_icd={} \
             first_diff={:?}",
            p.total_pixels, p.max_channel_delta, p.differing_pixels, software, p.first
        );
        if software {
            assert_eq!(p.max_channel_delta, 0, "[{tag}] 软件 ICD 上必须逐字节一致");
            assert_eq!(p.differing_pixels, 0, "[{tag}] 软件 ICD 上必须逐字节一致");
        } else {
            assert!(
                p.max_channel_delta <= 1 && p.differing_pixels * 200 <= p.total_pixels,
                "[{tag}] 真机档位不达标：max={} diff={}",
                p.max_channel_delta,
                p.differing_pixels
            );
        }
    }
}

/// 线性域纹理（16×16 渐变 + 黑块），用于位移幅值。
fn make_texture(scene: &mut Scene) -> u32 {
    let mut px = Vec::with_capacity(16 * 16);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let on = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.0 };
            let g = (x as f32 + y as f32) / 30.0;
            px.push([(0.25 + 0.75 * g) * on, (0.25 + 0.75 * g) * on, (0.25 + 0.75 * g) * on, 1.0]);
        }
    }
    scene.add_texture_linear(16, 16, &px)
}

/// 二十面体三角笼（20 面）。
fn icosahedron() -> (Vec<Vec3>, Vec<u32>) {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut pos = vec![
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    for p in pos.iter_mut() {
        *p = p.normalize();
    }
    let idx = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6,
        7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6,
        7, 9, 8, 1,
    ];
    (pos, idx)
}

/// **W4：细分 + 位移烘焙成三角形后，三后端逐位一致。**
///
/// 场景只装一个 `CurveReq::TriSoup`（烘焙产物）+ 一盏灯：把几何差异隔离出来，
/// 不让球体/曲线的着色分支混进差异归因。
#[test]
fn w4_subdivided_displaced_bake_matches_on_three_backends() {
    // 烘焙：Loop k=3 → 位移（幅值 0.12，亮度通道）→ 三角形 soup。
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let (p, t) = icosahedron();
    let uv: Vec<[f32; 2]> = p
        .iter()
        .map(|v| [0.5 + 0.5 * v.z.atan2(v.x) / std::f32::consts::PI, 0.5 + 0.5 * v.y])
        .collect();
    let sub = subdivide(&p, &t, &uv, 3);
    let amp = DisplaceParams::new(0.12);
    let disp = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);
    let tris = bake_triangles(&disp, &sub.indices);
    assert_eq!(tris.len(), sub.tri_count());
    assert!(tris.len() > 1000, "烘焙产物应有足够的三角形：{}", tris.len());

    let mut req = base_request(32, 24);
    req.spheres.clear();
    req.curves = vec![CurveReq::TriSoup { triangles: tris.clone() }];

    let mut cpu = CpuRefBackend;
    let cpu_fb = cpu.render_scene(&req);
    assert!(cpu_fb.is_valid());
    // 非平凡：球正对相机，按几何算应覆盖 ~85 px（半径 1、距 4、fov 60°、24 px 高），
    // 断言取 40（约一半，留位移/着色的余量）。这条只为排除"两边都是空帧也能
    // 拿 0 差异"，不是成像质量的判据。
    let lit = (0..(32 * 24))
        .filter(|i| !is_background(cpu_fb.get_pixel(i % 32, i / 32).unwrap()))
        .count();
    assert!(lit >= 40, "烘焙帧应覆盖画面，实测 {lit} 个非背景像素");
    println!("W4 烘焙帧：{} 三角，{lit} / {} 个非背景像素", tris.len(), 32 * 24);

    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    let gpu_fb = backend.render_scene(&req);
    assert!(gpu_fb.is_valid(), "GPU 必须产出有效帧");
    let p = parity(&cpu_fb, &gpu_fb);
    println!(
        "[w4-bake] pixels={} max_channel_delta={} differing_pixels={} software_icd={} \
         first_diff={:?}",
        p.total_pixels, p.max_channel_delta, p.differing_pixels, software, p.first
    );
    if software {
        assert_eq!(p.max_channel_delta, 0, "软件 ICD 上必须逐字节一致（首个差异 {:?}）", p.first);
        assert_eq!(p.differing_pixels, 0, "软件 ICD 上必须逐字节一致（首个差异 {:?}）", p.first);
    } else {
        assert!(
            p.max_channel_delta <= 1 && p.differing_pixels * 200 <= p.total_pixels,
            "真机档位不达标：max={} diff={}",
            p.max_channel_delta,
            p.differing_pixels
        );
    }
}
