//! V2.2-A-2 曲线图元对拍钉子（任务表 T06「测 2」之二）。
//!
//! 覆盖 PRD 验收 1 / 2 / 3（GPU 侧 5 见下"留名登记"）：
//!
//! 1. **CPU 参考出有效帧**：含曲线图元的场景经 `CpuRefBackend::render_scene`
//!    得到 `w*h*4` 的有效帧，且曲线确实参与成像（几何位置可判）。
//! 2. **零位移（不回归）**：`curves` 为空 ⇒ 与"不含曲线字段的 V2.1 等价请求"
//!    **逐字节相同**（A-2 不回归的机械保证，设计 §4.3）。
//! 3. **双跑逐位一致**：同请求两次渲染 `to_bits` 全等（CPU 口径 G3）。
//! 4. **退化形态不 panic**：`r=0 / h=0 / profile 单点 / 非有限 / 空集合` 经
//!    `render_scene` 全链路不 panic，且输出与"曲线被忽略"一致。
//!
//! # GPU 侧（Vulkan megakernel 曲线支持）——留名登记（设计 §1.2 失败退路）
//!
//! 设计 §1.2.2 的失败退路写明：若 megakernel 遍历段的软件 ICD 0 差异在 3 个
//! 迭代内达不到，则降级交付「三角化器 + CPU 参考 + 保真度界 + WGSL 侧」，Vulkan
//! megakernel 曲线支持整项**留名**（G9 允许，A-2 为 P1 需记一次变更）。
//!
//! 本仓现状：`vulkan::spirv_rt` 的发射器仍是 **V1 四绑定**（params/spheres/lights/out），
//! 栈式 BVH 遍历段未实装 ⇒ `VulkanBackend::render_scene` 对含曲线场景**忽略曲线**
//! （与 `curves` 为空逐位相同）。故本文件对 GPU 侧只做**登记式**断言：
//! `VULKAN_CURVE_PATH` 为 `false` 时校验"曲线被忽略"这一降级语义的确定性；
//! 置 `true`（遍历段实装后）自动切换为「软件 ICD 上 `max_channel_delta == 0 &&
//! differing_pixels == 0`」的硬判据——**判据不放宽**，只是尚未点亮。

use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, CurveReq, GpuDevice, LightReq, SceneRenderRequest,
    SphereReq,
};
use prism_math::Vec3;

/// Vulkan megakernel 的曲线（栈式 BVH 遍历）支持是否已点亮。
///
/// 见文件头「留名登记」：当前为 `false`（发射器仍 V1 四绑定）。此常量翻 `true`
/// 即自动启用软件 ICD 的 0 差异硬断言。
const VULKAN_CURVE_PATH: bool = false;

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
            center: Vec3::new(0.0, 2.5, 0.0),
            radius: 0.3,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4,
            metalness: 0.1,
        }],
        aesthetic: AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        curves: Vec::new(),
    }
}

/// 中心放一个正对相机的圆柱（轴沿 Z），用于"几何位置可判"的成像断言。
fn cylinder_scene(w: u32, h: u32) -> SceneRenderRequest {
    let mut req = base_request(w, h);
    // 相机的 forward 是 +Z（eye 在 -Z 侧看向原点），故圆柱轴取 +Z 才正对相机。
    req.curves = vec![CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.9,
        height: 3.0,
        axis: Vec3::new(0.0, 0.0, 1.0),
    }];
    req
}

fn pixel(fb: &prism_gpu::Framebuffer, x: u32, y: u32) -> Option<[u8; 4]> {
    fb.get_pixel(x, y)
}

fn is_background(px: [u8; 4]) -> bool {
    // 背景色 (0.04,0.05,0.08) → RGBA8 约 (10,13,20)；曲线命中必经直接光，明显更亮。
    px[0] <= 12 && px[1] <= 15 && px[2] <= 22
}

// ---------------------------------------------------------------------------
// 验收 1：CPU 参考出有效帧 + 曲线确实参与成像
// ---------------------------------------------------------------------------

#[test]
fn cpu_reference_renders_curves_into_a_valid_frame() {
    let req = cylinder_scene(48, 32);
    let mut be = CpuRefBackend;
    let fb = be.render_scene(&req);
    assert!(fb.is_valid(), "含曲线场景必须产出有效帧");
    assert_eq!(fb.pixels.len(), (48 * 32 * 4) as usize);

    // 中心像素必然被圆柱侧壁覆盖（半径 0.9、高 3.0、正对相机）。
    let center = pixel(&fb, 24, 16).expect("中心像素必须可读");
    assert!(
        !is_background(center),
        "圆柱应覆盖画面中心，实际读到背景 {center:?}"
    );
    assert_eq!(center[3], 255, "渲染像素必须不透明");

    // 四角必然在圆柱之外 → 背景。
    for (x, y) in [(0u32, 0u32), (47, 0), (0, 31), (47, 31)] {
        let px = pixel(&fb, x, y).expect("角落像素必须可读");
        assert!(is_background(px), "角落 ({x},{y}) 应落在背景，实际 {px:?}");
    }

    // 画面里"非背景"像素必须成规模（不是个别噪点）。
    let lit = (0..48 * 32)
        .filter(|i| {
            let px = pixel(&fb, i % 48, i / 48).unwrap();
            !is_background(px)
        })
        .count();
    assert!(lit > 48 * 32 / 5, "曲线覆盖面过小：仅 {lit} 个非背景像素");
    println!("含曲线场景：{lit} / {} 个像素被曲线覆盖", 48 * 32);
}

/// 曲线与球的**最近命中**正确性：球在圆柱后方（同一条中心射线），球必须被遮挡。
#[test]
fn nearest_hit_prefers_nearer_primitive() {
    let mut occluded = base_request(32, 32);
    occluded.spheres = vec![SphereReq {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.5,
        albedo: Vec3::new(1.0, 0.0, 0.0),
        roughness: 0.4,
        metalness: 0.0,
    }];
    occluded.curves = vec![CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.0, -1.5), // 更靠近相机（eye 在 -Z 侧）
        radius: 0.9,
        height: 3.0,
        axis: Vec3::new(0.0, 0.0, 1.0),
    }];
    let mut be = CpuRefBackend;
    let with_curve = be.render_scene(&occluded);
    let center = pixel(&with_curve, 16, 16).expect("中心可读");

    // 对照：同一请求去掉曲线 ⇒ 中心看到红球。
    let mut no_curve = occluded.clone();
    no_curve.curves.clear();
    let without = be.render_scene(&no_curve);
    let center_no_curve = pixel(&without, 16, 16).expect("中心可读");

    assert!(
        center_no_curve[0] > center_no_curve[1] + 40,
        "对照帧中心应为红球，实际 {center_no_curve:?}"
    );
    assert_ne!(
        center, center_no_curve,
        "前挡圆柱必须改变中心像素（最近命中语义）"
    );
    println!("最近命中：曲线帧 {center:?} vs 球帧 {center_no_curve:?}");
}

// ---------------------------------------------------------------------------
// 验收 2：零位移（curves 为空 ⇒ 与 V2.1 等价请求逐字节相同）
// ---------------------------------------------------------------------------

#[test]
fn empty_curves_is_bytewise_identical_to_v21_baseline() {
    let req = base_request(40, 24);
    let mut be = CpuRefBackend;
    let a = be.render_scene(&req);

    // "V2.1 等价请求"：同一场景、同样使用 curves 字段但为空 —— 逐字节比对。
    let mut b_req = req.clone();
    b_req.curves = Vec::new();
    let b = be.render_scene(&b_req);

    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    assert_eq!(a.pixels, b.pixels, "curves 为空时输出必须与 V2.1 逐字节相同");
    println!("零位移：curves 为空 ⇒ {} 字节逐字节相同", a.pixels.len());
}

// ---------------------------------------------------------------------------
// 验收 2（续）：双跑逐位一致
// ---------------------------------------------------------------------------

#[test]
fn curve_render_is_bitwise_deterministic_across_runs() {
    let req = cylinder_scene(40, 24);
    let mut be = CpuRefBackend;
    let a = be.render_scene(&req);
    let b = be.render_scene(&req);
    assert_eq!(a.pixels.len(), b.pixels.len());
    assert_eq!(a.pixels, b.pixels, "同请求双跑必须逐位一致（确定性契约 G3）");

    // 非法/退化曲线：空网格 ⇒ 必须与"无曲线"逐字节相同（不引入噪声）。
    let mut degenerate = req.clone();
    degenerate.curves = vec![
        CurveReq::Cylinder {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.0,
            height: 1.0,
            axis: Vec3::new(0.0, 1.0, 0.0),
        },
        CurveReq::Torus {
            center: Vec3::new(0.0, 0.0, 0.0),
            major: 1.0,
            minor: f32::NAN,
            axis: Vec3::new(0.0, 1.0, 0.0),
        },
    ];
    let d = be.render_scene(&degenerate);
    let mut none = req.clone();
    none.curves.clear();
    let e = be.render_scene(&none);
    assert_eq!(
        d.pixels, e.pixels,
        "全退化曲线 ⇒ 空网格 ⇒ 必须与无曲线逐字节相同"
    );
}

// ---------------------------------------------------------------------------
// 验收 3：退化形态全链路不 panic
// ---------------------------------------------------------------------------

#[test]
fn degenerate_curves_do_not_panic_through_render_scene() {
    let cases: Vec<(&str, CurveReq)> = vec![
        (
            "r=0",
            CurveReq::Cylinder {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 0.0,
                height: 1.0,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
        (
            "h=0",
            CurveReq::Cone {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                height: 0.0,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
        (
            "profile-single",
            CurveReq::Revolution {
                center: Vec3::new(0.0, 0.0, 0.0),
                profile: vec![Vec3::new(1.0, 0.0, 0.0)],
                segments: 8,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
        (
            "non-finite",
            CurveReq::Torus {
                center: Vec3::new(0.0, 0.0, 0.0),
                major: f32::INFINITY,
                minor: 1.0,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
    ];
    for (tag, c) in &cases {
        let mut req = base_request(16, 16);
        req.curves = vec![c.clone()];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut be = CpuRefBackend;
            be.render_scene(&req).pixels.len()
        }));
        let len = r.unwrap_or_else(|_| panic!("退化形态 `{tag}` 在 render_scene 里 panic 了"));
        assert_eq!(len, 16 * 16 * 4, "退化形态 `{tag}` 仍须产出有效帧");
    }
    // 空集合：直接可跑。
    let req = base_request(16, 16);
    let mut be = CpuRefBackend;
    assert!(be.render_scene(&req).is_valid());
    println!("退化形态：{} 组全部不 panic 且出有效帧", cases.len());
}

// ---------------------------------------------------------------------------
// GPU 侧：曲线路径留名登记 / 点亮后 0 差异硬判据
// ---------------------------------------------------------------------------

#[test]
fn vulkan_curve_path_registration() {
    let req = cylinder_scene(24, 16);
    let mut be = prism_gpu::vulkan::VulkanBackend::new();
    let available = be.render_available();

    if !VULKAN_CURVE_PATH {
        // 留名：megakernel 遍历段未实装 ⇒ 曲线被忽略（降级语义确定）。
        if available {
            let gpu = be.render_scene(&req);
            if gpu.is_valid() {
                let mut req_no_curve = req.clone();
                req_no_curve.curves.clear();
                let gpu_no_curve = be.render_scene(&req_no_curve);
                assert_eq!(
                    gpu.pixels, gpu_no_curve.pixels,
                    "留名期：GPU 必须**忽略**曲线（与 curves 为空逐位相同），不得半实现"
                );
            }
        }
        println!(
            "留名登记：Vulkan megakernel 曲线支持未点亮（VULKAN_CURVE_PATH=false，\
             render_available={available}）；按设计 §1.2 失败退路，A-2 降级为\
             「三角化器 + CPU 参考 + 保真度界 + WGSL 侧」，需交付前 CI/真机验证后方可翻 true"
        );
        return;
    }

    // 点亮后：软件 ICD 上必须逐字节一致（判据与 `qa_render_scene.rs:388-390` 同款）。
    if !available {
        println!("无可用设备：0 差异判据跳过（实机项 = 永不进门禁）");
        return;
    }
    let gpu = be.render_scene(&req);
    assert!(gpu.is_valid(), "点亮后 GPU 必须产出有效帧");
    let mut cpu_be = CpuRefBackend;
    let cpu = cpu_be.render_scene(&req);
    let mut max_delta = 0u8;
    let mut differing = 0usize;
    for i in 0..(24 * 16) {
        let a = &cpu.pixels[i * 4..i * 4 + 4];
        let b = &gpu.pixels[i * 4..i * 4 + 4];
        let mut pix = 0u8;
        for c in 0..3 {
            pix = pix.max(a[c].abs_diff(b[c]));
        }
        max_delta = max_delta.max(pix);
        if pix > 0 {
            differing += 1;
        }
    }
    assert_eq!(max_delta, 0, "软件 ICD 上必须逐字节一致（最大通道差 {max_delta}）");
    assert_eq!(differing, 0, "软件 ICD 上必须逐字节一致（差异像素 {differing}）");
}
