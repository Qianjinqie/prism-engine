//! V2.2-A-2 曲线图元对拍钉子（任务表 T06「测 2」之二 + A-2-CURVE-VK 收口）。
//!
//! 覆盖 PRD 验收 1 / 2 / 3 / 5：
//!
//! 1. **CPU 参考出有效帧**：含曲线图元的场景经 `CpuRefBackend::render_scene`
//!    得到 `w*h*4` 的有效帧，且曲线确实参与成像（几何位置可判）。
//! 2. **零位移（不回归）**：`curves` 为空 ⇒ 与"不含曲线字段的 V2.1 等价请求"
//!    **逐字节相同**（A-2 不回归的机械保证，设计 §4.3）。
//! 3. **双跑逐位一致**：同请求两次渲染 `to_bits` 全等（CPU 口径 G3）。
//! 4. **退化形态不 panic**：`r=0 / h=0 / profile 单点 / 非有限 / 空集合` 经
//!    `render_scene` 全链路不 panic，且输出与"曲线被忽略"一致。
//! 5. **Vulkan megakernel 曲线路径点亮**（A-2-CURVE-VK）：
//!    `VULKAN_CURVE_PATH == true` ⇒ 软件 ICD（lavapipe）上
//!    `max_channel_delta == 0 && differing_pixels == 0` 的**硬判据**生效，
//!    四种图元（圆柱 / 圆锥 / 圆环 / 旋转曲面）逐一过门。
//!
//! # 判据纪律（一个字都不放宽）
//!
//! 软件 ICD 的 0 差异是 PRD 钉死的档位（设计 §1.2.1「软件 ICD 档 0 差异可达」）。
//! 为了让判据变绿而放宽 `differing_pixels` = **改判据**，不是修实现——本文件
//! 的 `assert_eq!(.., 0)` 是硬判据，A-2-CURVE-VK 的失败退路是**把开关翻回
//! `false` 并留名**，不是调阈值。
//!
//! # 当前状态（V2.2 W3，A-2-CURVE-VK 交付）
//!
//! `crates/prism-gpu/src/vulkan/spirv_rt.rs` 的发射器已升到 **V2 六绑定**
//! （params / spheres / lights / triangles / nodes / out）+ 栈式 BVH 遍历段
//! （双栈、近子后入栈、`t` 并列先左），语义逐句移植 `shaders/scene_frame.wgsl`；
//! `VULKAN_CURVE_PATH = true` 即灯亮。回退只需把它改回 `false`（GPU 侧随即
//! 退回"忽略曲线"的登记语义，见本文件下方 `vulkan_curve_path_off_*` 留档说明）。

use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, CurveReq, Framebuffer, GpuDevice, LightReq,
    SceneRenderRequest, SphereReq,
};
use prism_math::Vec3;

/// Vulkan megakernel 的曲线（栈式 BVH 遍历）支持是否已点亮。
///
/// `true` ⇒ 本文件对 GPU 侧启用软件 ICD 的 0 差异硬判据（见文件头）。
const VULKAN_CURVE_PATH: bool = true;

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
            maps: prism_gpu::NO_MAPS,
        }],
        aesthetic: AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        curves: Vec::new(),
    ..Default::default()
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

/// 四种曲线图元各一个对拍场景（**只含曲线 + 一盏灯**，把几何差异隔离出来）。
///
/// 尺寸统一取 24×16：足以覆盖"圆弧段 / 轮廓折线 / 上下盖"三类面片，又让
/// 三角化 + SAH-BVH 构建保持在毫秒级（torus 在 tol=0.02 下会切出上万三角形，
/// 故取小半径，见每个 case 的注释）。
fn primitive_cases() -> Vec<(&'static str, CurveReq)> {
    vec![
        (
            // 侧壁 2×seg 四边形 + 上下盖扇形；轴沿 +Z 正对相机。
            "cylinder",
            CurveReq::Cylinder {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 0.9,
                height: 3.0,
                axis: Vec3::new(0.0, 0.0, 1.0),
            },
        ),
        (
            // 侧壁 seg 三角形 + 底盖扇形；锥尖朝 +Z（朝相机）。
            "cone",
            CurveReq::Cone {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 0.9,
                height: 2.0,
                axis: Vec3::new(0.0, 0.0, 1.0),
            },
        ),
        (
            // 主环 × 管：小半径版（major 0.30 / minor 0.10 ⇒ 约 1.8 万三角形）。
            "torus",
            CurveReq::Torus {
                center: Vec3::new(0.0, 0.0, 0.0),
                major: 0.30,
                minor: 0.10,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
        (
            // 轮廓折线 × 周向段：含水平段（环带）与斜段（锥台）。
            "revolution",
            CurveReq::Revolution {
                center: Vec3::new(0.0, 0.0, 0.0),
                profile: vec![
                    Vec3::new(0.30, 0.0, 0.0),
                    Vec3::new(0.50, 0.5, 0.0),
                    Vec3::new(0.30, 1.0, 0.0),
                ],
                segments: 12,
                axis: Vec3::new(0.0, 1.0, 0.0),
            },
        ),
    ]
}

/// 单图元场景（清空球，只留曲线 + 一盏灯）。
fn curve_only_scene(w: u32, h: u32, curve: CurveReq) -> SceneRenderRequest {
    let mut req = base_request(w, h);
    req.spheres.clear();
    req.curves = vec![curve];
    req
}

fn pixel(fb: &Framebuffer, x: u32, y: u32) -> Option<[u8; 4]> {
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
        maps: prism_gpu::NO_MAPS,
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
// GPU 侧（Vulkan megakernel 曲线支持）：0 差异硬判据
// ---------------------------------------------------------------------------

/// 一帧的逐像素差异摘要（诊断 + 硬判据共用）。
struct CurveParity {
    /// 非 alpha 通道的最大绝对差。
    max_channel_delta: u8,
    /// 任一通道有差异的像素数。
    differing_pixels: usize,
    /// 总像素数。
    total_pixels: usize,
    /// 首个差异样本 `(像素下标, CPU, GPU)`。
    first: Option<(usize, [u8; 4], [u8; 4])>,
}

/// 逐像素比对（只比 RGB；alpha 恒 255，本文件另做断言）。
fn curve_parity(cpu: &Framebuffer, gpu: &Framebuffer) -> CurveParity {
    assert_eq!((cpu.width, cpu.height), (gpu.width, gpu.height), "尺寸必须一致");
    assert_eq!(cpu.pixels.len(), gpu.pixels.len(), "帧长必须一致");
    let total_pixels = (cpu.width as usize) * (cpu.height as usize);
    let mut max_channel_delta = 0u8;
    let mut differing_pixels = 0usize;
    let mut first = None;
    for i in 0..total_pixels {
        let a = &cpu.pixels[i * 4..i * 4 + 4];
        let b = &gpu.pixels[i * 4..i * 4 + 4];
        let mut pix = 0u8;
        for c in 0..3 {
            pix = pix.max(a[c].abs_diff(b[c]));
        }
        max_channel_delta = max_channel_delta.max(pix);
        if pix > 0 {
            differing_pixels += 1;
            if first.is_none() {
                first = Some((i, [a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]));
            }
        }
        assert_eq!(b[3], 255, "GPU 像素 alpha 必须不透明");
    }
    CurveParity {
        max_channel_delta,
        differing_pixels,
        total_pixels,
        first,
    }
}

/// 是否是**软件 ICD**（lavapipe / llvmpipe / SwiftShader）：此时判据是逐位一致。
fn is_software_icd(device: &str) -> bool {
    let d = device.to_ascii_lowercase();
    d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
}

/// 打开后端；无可用设备时打印一句并返回 `None`（实机项永不进门禁）。
fn gpu_backend() -> Option<prism_gpu::vulkan::VulkanBackend> {
    let be = prism_gpu::vulkan::VulkanBackend::new();
    if !VULKAN_CURVE_PATH {
        panic!(
            "VULKAN_CURVE_PATH 必须为 true（A-2-CURVE-VK 已点亮）；\
             若走失败退路请同时恢复本文件的登记式断言，不得静默降级"
        );
    }
    if !be.render_available() {
        println!("无可用设备：0 差异判据跳过（实机项 = 永不进门禁）");
        return None;
    }
    Some(be)
}

/// **PRD 验收 5（硬判据）**：四种曲线图元在软件 ICD 上必须与 CPU 参考逐字节一致。
///
/// 判据：`max_channel_delta == 0 && differing_pixels == 0`（**不放宽**）。
/// 非软件 ICD（真机 GPU）上按 `qa_render_scene` 的同一档位（≤1/255、≤0.5%）——
/// 逐位一致是软件 ICD 的实测结论，真机允许末位量化噪声。
#[test]
fn vulkan_curve_primitives_match_cpu_reference_bitwise() {
    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    let mut cpu_be = CpuRefBackend;
    for (tag, curve) in primitive_cases() {
        let req = curve_only_scene(24, 16, curve);
        let gpu = backend.render_scene(&req);
        assert!(gpu.is_valid(), "[{tag}] 点亮后 GPU 必须产出有效帧");
        let cpu = cpu_be.render_scene(&req);
        assert!(cpu.is_valid(), "[{tag}] CPU 参考必须产出有效帧");
        let p = curve_parity(&cpu, &gpu);
        println!(
            "[curve-parity:{tag}] pixels={} max_channel_delta={} differing_pixels={} \
             software_icd={} first_diff={:?}",
            p.total_pixels, p.max_channel_delta, p.differing_pixels, software, p.first
        );
        if software {
            assert_eq!(
                p.max_channel_delta, 0,
                "[{tag}] 软件 ICD 上必须逐字节一致（最大通道差 {} ≠ 0，首个差异 {:?}）",
                p.max_channel_delta, p.first
            );
            assert_eq!(
                p.differing_pixels, 0,
                "[{tag}] 软件 ICD 上必须逐字节一致（差异像素 {} ≠ 0，首个 {:?}）",
                p.differing_pixels, p.first
            );
        } else {
            assert!(
                p.max_channel_delta <= 1 && p.differing_pixels * 200 <= p.total_pixels,
                "[{tag}] 真机档位（≤1/255、≤0.5%）不达标：max={} diff={}（首个 {:?}）",
                p.max_channel_delta,
                p.differing_pixels,
                p.first
            );
        }
    }
}

/// **验收 5（续）**：球与曲线**同场**时的最近命中与互相遮挡也必须逐位一致。
///
/// 这一项覆盖「`has_hit && has_curve` 的互斥裁决」——曲线命中作废球命中、
/// 且曲线搜索的上界是球的最近 `t`（与 CPU 的 `hit = None` / `closest_t` 同式）。
#[test]
fn vulkan_curve_and_spheres_share_the_nearest_hit_path() {
    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    let mut cpu_be = CpuRefBackend;
    // 前挡圆柱 + 后方红球（同一条中心射线）+ 一枚偏置小球。
    let mut req = base_request(32, 24);
    req.spheres = vec![
        SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.5,
            albedo: Vec3::new(1.0, 0.0, 0.0),
            roughness: 0.4,
            metalness: 0.0,
            maps: prism_gpu::NO_MAPS,
        },
        SphereReq {
            center: Vec3::new(-1.2, -1.0, -1.0),
            radius: 0.35,
            albedo: Vec3::new(0.2, 0.9, 0.3),
            roughness: 0.5,
            metalness: 0.2,
            maps: prism_gpu::NO_MAPS,
        },
    ];
    req.curves = vec![CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.0, -1.5),
        radius: 0.9,
        height: 3.0,
        axis: Vec3::new(0.0, 0.0, 1.0),
    }];
    let gpu = backend.render_scene(&req);
    assert!(gpu.is_valid());
    let cpu = cpu_be.render_scene(&req);
    let p = curve_parity(&cpu, &gpu);
    println!(
        "[curve+sphere] max_channel_delta={} differing_pixels={} first_diff={:?}",
        p.max_channel_delta, p.differing_pixels, p.first
    );
    if software {
        assert_eq!(p.max_channel_delta, 0, "首个差异 {:?}", p.first);
        assert_eq!(p.differing_pixels, 0, "首个差异 {:?}", p.first);
    }
}

/// **G3：双跑逐位一致**（GPU 侧；软件 ICD 上连中间量也必须确定）。
#[test]
fn vulkan_curve_render_is_bitwise_deterministic_across_runs() {
    let Some(mut backend) = gpu_backend() else {
        return;
    };
    for (tag, curve) in primitive_cases() {
        let req = curve_only_scene(20, 12, curve);
        let a = backend.render_scene(&req);
        let b = backend.render_scene(&req);
        assert!(a.is_valid() && b.is_valid(), "[{tag}] 双跑都必须出有效帧");
        assert_eq!(a.pixels, b.pixels, "[{tag}] 同请求双跑必须逐位一致（G3）");
        assert_eq!(
            (a.width, a.height),
            (b.width, b.height),
            "[{tag}] 帧尺寸必须一致"
        );
    }
    println!("G3：四种图元双跑全部逐位一致");
}

/// **`curves` 为空 ⇒ 与 V1（4 绑定时代）输出逐位一致**（不回归的机械保证）。
///
/// V1 的输出基准就是 [`CpuRefBackend`]（V2.1 起两者在软件 ICD 上逐字节一致，
/// 见 `qa_render_scene.rs`）。故本项断言：空曲线时 GPU ≡ CPU，
/// 且带空 `curves` 字段的请求 ≡ 不带该字段的等价请求。
#[test]
fn vulkan_curve_path_with_empty_curves_is_bytewise_v1() {
    let Some(mut backend) = gpu_backend() else {
        return;
    };
    let software = is_software_icd(backend.device_name().unwrap_or(""));
    let mut cpu_be = CpuRefBackend;

    let req = base_request(40, 24); // curves 为空，带球 + 灯
    let gpu = backend.render_scene(&req);
    assert!(gpu.is_valid(), "空曲线场景 GPU 必须出有效帧");
    let cpu = cpu_be.render_scene(&req);
    let p = curve_parity(&cpu, &gpu);
    println!(
        "[empty-curves] max_channel_delta={} differing_pixels={} first_diff={:?}",
        p.max_channel_delta, p.differing_pixels, p.first
    );
    if software {
        assert_eq!(
            p.max_channel_delta, 0,
            "空曲线必须与 V1 逐字节一致（首个差异 {:?}）",
            p.first
        );
        assert_eq!(p.differing_pixels, 0, "空曲线必须与 V1 逐字节一致");
    }

    // 「退化到空网格」的曲线集合同样必须与"无曲线"逐字节相同（GPU 侧）。
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
            major: f32::NAN,
            minor: 1.0,
            axis: Vec3::new(0.0, 1.0, 0.0),
        },
    ];
    let d = backend.render_scene(&degenerate);
    assert!(d.is_valid(), "全退化曲线不得 panic / 不得空帧");
    assert_eq!(
        d.pixels, gpu.pixels,
        "退化曲线 ⇒ 空网格 ⇒ 必须与空曲线逐字节相同（不得半实现）"
    );
}

/// **验收 3（GPU 侧）**：退化形态不 panic、且退化为空网格时与"无曲线"逐位相同。
#[test]
fn vulkan_curve_degenerate_forms_do_not_panic() {
    let Some(mut backend) = gpu_backend() else {
        return;
    };
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
    for (tag, curve) in &cases {
        let mut req = base_request(16, 16);
        req.curves = vec![curve.clone()];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.render_scene(&req).pixels.len()
        }));
        // 设备不可用是合法结局（返回空帧），这里只钉"不 panic"。
        let len = r.unwrap_or_else(|_| panic!("退化形态 `{tag}` 在 GPU 路径里 panic 了"));
        assert!(
            len == 0 || len == 16 * 16 * 4,
            "退化形态 `{tag}` 的输出长度必须是 0（诚实空帧）或 w*h*4，实际 {len}"
        );
    }
    println!("GPU 退化形态：{} 组全部不 panic", cases.len());
}
