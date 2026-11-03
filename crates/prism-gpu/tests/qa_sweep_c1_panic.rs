//! QA 全仓缺陷普查 · 审计 1（panic 源）攻击测试。
//!
//! 范围：`prism-gpu` 公开入口的极端参数穷举 + 已知高危点实证。
//! 全部用例均以「不 panic」为断言（宪章 C18：永不 panic）。
//! 失败即活缺陷（红），通过即守卫已存在且有效（绿）。
//!
//! 零第三方依赖，仅 std + #[test]。

use prism_gpu::{CpuRefBackend, Framebuffer, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// Framebuffer::new 尺寸乘法溢出 / 巨额分配
// ---------------------------------------------------------------------------

/// `Framebuffer::new(w, h)` 内部算 `(w as usize) * (h as usize) * 4`。
/// 契约未声明尺寸上界 ⇒ 需验证不 panic（debug 下 usize 乘法溢出即 panic）。
#[test]
fn a01_framebuffer_new_extreme_dims_do_not_panic() {
    let dims: [(u32, u32); 10] = [
        (0, 0),
        (1, 1),
        (0, u32::MAX),
        (u32::MAX, 0),
        (1, u32::MAX),
        (u32::MAX, 1),
        (65535, 65535),
        (65536, 65536),
        (1 << 20, 1 << 20),
        (u32::MAX, u32::MAX),
    ];
    for (w, h) in dims {
        // 只要求：不 panic、不 abort。允许返回 Err/空/被钳制。
        let fb = Framebuffer::new(w, h);
        assert_eq!(fb.width, w, "width 应原样回显（钳制需另有契约）");
        assert_eq!(fb.height, h, "height 应原样回显");
    }
}

/// 同上，但要求「要么成功分配、要么干净拒绝」——不允许进程 abort。
/// 用 `catch_unwind` 捕捉 panic 并转为显式失败证据。
#[test]
fn a02_framebuffer_new_no_panic_via_catch_unwind() {
    let r = std::panic::catch_unwind(|| Framebuffer::new(u32::MAX, u32::MAX));
    assert!(
        r.is_ok(),
        "Framebuffer::new(u32::MAX, u32::MAX) panic 了（usize 乘法溢出或分配失败）——P1 缺陷"
    );
}

// ---------------------------------------------------------------------------
// VulkanBackend::render_scene 的 unimplemented!()
// ---------------------------------------------------------------------------

/// `vulkan::VulkanBackend` 是 `pub` 且实现 `pub trait GpuDevice`，
/// `render_scene` 现返回空帧（原为 `unimplemented!()`，已修）。D1 实装使该类型带探测上下文，构造走 `VulkanBackend::new()`（仍无参、仍
/// 可经公开 API 触达）。
#[test]
fn a03_vulkan_backend_render_scene_does_not_panic() {
    let mut backend = prism_gpu::vulkan::VulkanBackend::new();
    let req = SceneRenderRequest {
        camera: prism_gpu::CameraReq {
            eye: Vec3::splat(0.0),
            target: Vec3::splat(0.0),
            fov: 1.0,
            aspect: 1.0,
        },
        lights: Vec::new(),
        spheres: Vec::new(),
        aesthetic: prism_gpu::AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        width: 4,
        height: 4,
    curves: Vec::new(),
    ..Default::default()
};
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let fb = backend.render_scene(&req);
        fb.pixels.len()
    }));
    assert!(
        r.is_ok(),
        "VulkanBackend::render_scene panic（unimplemented!）——P0：pub trait 实现含 panic 宏，违反宪章 C18"
    );
}

// ---------------------------------------------------------------------------
// set_pixel 越界索引
// ---------------------------------------------------------------------------

/// `set_pixel` 内部 `(y*width + x)*4` 为 usize 运算；`x/y` 为任意 u32。
/// 契约称越界即 return，需实证在极端 x/y 下不 panic（乘法溢出）。
#[test]
fn a04_set_pixel_extreme_coords_do_not_panic() {
    let mut fb = Framebuffer::new(4, 4);
    let coords: [(u32, u32); 8] = [
        (0, 0),
        (3, 3),
        (4, 4),
        (u32::MAX, u32::MAX),
        (u32::MAX, 0),
        (0, u32::MAX),
        (1 << 31, 1 << 31),
        (u32::MAX - 1, u32::MAX - 1),
    ];
    for (x, y) in coords {
        for c in [
            Vec3::splat(0.0),
            Vec3::splat(1.0),
            Vec3::splat(-1.0), // 越界 clamp 下界
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
        ] {
            fb.set_pixel(x, y, c);
        }
    }
    assert_eq!(fb.pixels.len(), 4 * 4 * 4, "越界写不应改变缓冲长度");
}

/// `channel_to_u8` 为 `(v.clamp(0,1)*255).round() as u8`。
/// NaN 经 `clamp` 后在 Rust 中返回 NaN ⇒ `NaN as u8` = 0（饱和）。
/// 本用例锁定「不 panic」并记录实际字节，NaN 行为交由断言 2 观测。
#[test]
fn a05_channel_to_u8_nan_inf_via_set_pixel_no_panic() {
    let mut fb = Framebuffer::new(2, 2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fb.set_pixel(0, 0, Vec3::splat(f32::NAN));
        fb.set_pixel(1, 0, Vec3::splat(f32::INFINITY));
        fb.set_pixel(0, 1, Vec3::splat(f32::NEG_INFINITY));
        fb.set_pixel(1, 1, Vec3::splat(-0.5));
    }));
    assert!(r.is_ok(), "set_pixel 对 NaN/±Inf 不应 panic");
    // 观测记录：NaN 落到 0（饱和转换），±Inf 被 clamp 到 0/255。
    assert_eq!(fb.pixels[0], 0, "NaN as u8 应饱和为 0（记录行为，非缺陷）");
    assert_eq!(fb.pixels[5], 255, "+Inf 应 clamp 到 1.0 ⇒ 255");
    assert_eq!(fb.pixels[10], 0, "-Inf 应 clamp 到 0.0 ⇒ 0");
}

// ---------------------------------------------------------------------------
// CpuRefBackend 在极端请求下的不变量
// ---------------------------------------------------------------------------

fn base_req(w: u32, h: u32) -> SceneRenderRequest {
    SceneRenderRequest {
        camera: prism_gpu::CameraReq {
            eye: Vec3::splat(0.0),
            target: Vec3::splat(0.0),
            fov: 1.0,
            aspect: 1.0,
        },
        lights: vec![prism_gpu::LightReq {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::splat(1.0),
            intensity: 1.0,
        }],
        spheres: vec![prism_gpu::SphereReq {
            center: Vec3::splat(0.0),
            radius: 1.0,
            albedo: Vec3::splat(1.0),
            roughness: 0.5,
            metalness: 0.0,
            maps: prism_gpu::NO_MAPS,
        }],
        aesthetic: prism_gpu::AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        width: w,
        height: h,
    curves: Vec::new(),
    ..Default::default()
}
}

/// 各种退化/极端渲染请求不得 panic。
#[test]
fn a06_cpu_ref_backend_degenerate_requests_do_not_panic() {
    let reqs: Vec<SceneRenderRequest> = vec![
        base_req(0, 0),
        base_req(1, 1),
        base_req(0, 8),
        base_req(8, 0),
    ];
    for mut req in reqs {
        // fov 极端值（perspective 除零 / NaN 传播）
        for fov in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e30] {
            req.camera.fov = fov;
            let mut be = CpuRefBackend;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                be.render_scene(&req).pixels.len()
            }));
            assert!(r.is_ok(), "CpuRefBackend fov={fov:?} panic 了");
        }
    }
}

/// 光源/球体携带 NaN/Inf 坐标时不得 panic，且不得写出与尺寸不符的缓冲。
#[test]
fn a07_cpu_ref_backend_nan_geometry_does_not_panic() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30] {
        let mut req = base_req(4, 4);
        req.spheres[0].center = Vec3::splat(bad);
        req.spheres[0].radius = bad;
        req.lights[0].intensity = bad;
        let mut be = CpuRefBackend;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.render_scene(&req)
        }));
        assert!(r.is_ok(), "球/光含 {bad:?} 时 panic 了");
        let fb = r.unwrap();
        assert_eq!(
            fb.pixels.len(),
            4 * 4 * 4,
            "NaN 几何下缓冲长度仍须 == w*h*4（尺寸不变量）"
        );
    }
}

// ---------------------------------------------------------------------------
// 汇总：像素缓冲长度不变量（尺寸自洽）
// ---------------------------------------------------------------------------

/// 帧缓冲 `pixels.len() == width*height*4` 是文档契约。
/// 对全部可成功构造的尺寸组合验证之。
#[test]
fn a08_framebuffer_len_invariant_holds() {
    for (w, h) in [(1u32, 1u32), (2, 3), (7, 5), (16, 16), (64, 1)] {
        let fb = Framebuffer::new(w, h);
        assert_eq!(fb.pixels.len(), (w as usize) * (h as usize) * 4);
    }
}
