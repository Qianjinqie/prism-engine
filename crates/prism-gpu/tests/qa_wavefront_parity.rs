//! **T-B / R-1 验收测试**：路径模式 WGSL wavefront ↔ CPU 参考**逐位**对拍。
//!
//! 场景构造 / 相机 uniform 打包 / Sobol 表字节 / CPU 参考帧的公共实现
//! 在 [`mod@common::r1_parity`]（golden 生成器 `qa_wavefront_golden.rs`
//! 与本文件共用同一份，杜绝"两份几何"）。
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

#[path = "common/r1_parity.rs"]
mod r1_parity;

use r1_parity::{
    assert_bitwise_equal, camera_uniform, cpu_reference, curve_cpu_reference, curve_request,
    device_ready, parity_request,
};
use prism_gpu::vulkan::wavefront::WavefrontMode;
use prism_gpu::vulkan::VulkanBackend;
use prism_rt::camera::Camera;
use prism_math::Vec3;

/// **R-1 主判据**：spp=2、max_bounces=3 的全路径逐位对拍（覆盖相机抖动、
/// NEE/MIS、Lambert 续弹、RR 存活、逃逸背景、spp 固定形状归约全链）。
#[test]
fn r1_wavefront_path_bitwise_parity_cpu_vs_gpu() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (16u32, 16u32, 2u32, 3u32);
    let cpu = cpu_reference(w, h, spp, bounces);

    // 场景有效性前置：CPU 参考必须非平凡（有光、有明暗），防"两边全零"假 parity。
    let max_l = cpu
        .iter()
        .map(|p| p[0] + p[1] + p[2])
        .fold(0.0f32, f32::max);
    assert!(max_l > 0.01, "CPU 参考无能量（max_luma={max_l}）：场景构造失效");

    let req = parity_request(w, h, spp, bounces);
    let rep = backend.render_wavefront_frame(&req);
    assert_eq!(rep.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", rep.detail);
    assert_eq!(rep.hdr.len(), (w * h) as usize);
    // 段数 = generate + 3×bounces + accumulate。
    assert_eq!(rep.dispatches, 1 + 3 * bounces + 1);

    assert_bitwise_equal(w, h, &cpu, &rep.hdr, "R-1 spp2 bounces3");
    println!(
        "[R-1] 设备={} 缓冲={} B",
        backend.device_name().unwrap_or("<unnamed>"),
        rep.buffer_bytes
    );
}

/// **R-1 第二配置**：spp=1、max_bounces=1（单事件：直接光 + 逃逸；RR 与
/// Sobol 弹射段不触发——把"直接光路径"与"多弹射路径"的判责分开）。
#[test]
fn r1_wavefront_path_bitwise_parity_single_bounce() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (12u32, 10u32, 1u32, 1u32);
    let cpu = cpu_reference(w, h, spp, bounces);
    let req = parity_request(w, h, spp, bounces);
    let rep = backend.render_wavefront_frame(&req);
    assert_eq!(rep.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", rep.detail);
    assert_bitwise_equal(w, h, &cpu, &rep.hdr, "R-1 spp1 bounces1");
}

/// **R-1 第三配置**：spp=5（非 2 的幂次树形归约形状：奇数尾直进路径）、
/// max_bounces=2。专打 accumulate 固定形状树的结合序。
#[test]
fn r1_wavefront_path_bitwise_parity_odd_tree_accumulate() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (8u32, 8u32, 5u32, 2u32);
    let cpu = cpu_reference(w, h, spp, bounces);
    let req = parity_request(w, h, spp, bounces);
    let rep = backend.render_wavefront_frame(&req);
    assert_eq!(rep.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", rep.detail);
    assert_bitwise_equal(w, h, &cpu, &rep.hdr, "R-1 spp5 bounces2");
}

/// **GPU 双跑确定性**：同请求连跑两次逐字节相同（路径模式 HDR 位型）。
#[test]
fn r1_gpu_path_mode_two_pass_bitwise_identical() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let req = parity_request(16, 16, 2, 3);
    let a = backend.render_wavefront_frame(&req);
    let b = backend.render_wavefront_frame(&req);
    assert_eq!(a.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", a.detail);
    assert_eq!(b.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", b.detail);
    assert_eq!(a.hdr.len(), b.hdr.len());
    for (i, (x, y)) in a.hdr.iter().zip(b.hdr.iter()).enumerate() {
        assert_eq!(
            x.map(f32::to_bits),
            y.map(f32::to_bits),
            "第 {i} 像素两次派发不一致"
        );
    }
    println!("[R-1] GPU 双跑逐位一致（{} 像素）", a.hdr.len());
}

/// **R-1 场景常量冻结**：相机 uniform 打包、Sobol 表、场景请求的字节级指纹。
///
/// 浏览器侧（`web/abi.rs` 的 `gpu.wavefront_path`）以 `include_bytes!` 内嵌
/// golden 并用**同一份常量**构造请求；本测试把两侧共用的标量（相机基向量、
/// tan_half_fov、aspect）钉在这里，任何一侧改动都会被对拍判红。
#[test]
fn r1_scene_constants_are_frozen() {
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
    let uni = camera_uniform(&cam, 16.0 / 16.0);
    // 相机 96 字节的关键位型（与 parity_request 逐位同源）。
    assert_eq!(uni.len(), 96);
    let f32_at = |off: usize| {
        f32::from_bits(u32::from_le_bytes([uni[off], uni[off + 1], uni[off + 2], uni[off + 3]]))
    };
    assert_eq!(f32_at(64), cam.tan_half_fov);
    assert_eq!(f32_at(84), 1.0);
    // 请求形状：seed/node_id/frame 三元组决定整条采样流（golden 的权威口径）。
    let req = parity_request(16, 16, 2, 3);
    assert_eq!(req.seed, 2026);
    assert_eq!(req.node_id, 7);
    assert_eq!(req.frame, 3);
    assert!(matches!(req.mode, WavefrontMode::Path));
    println!(
        "[R-1] 常量冻结：seed=2026 node_id=7 frame=3 tan={:e}（位型 0x{:08x}）",
        cam.tan_half_fov,
        cam.tan_half_fov.to_bits()
    );
}

// ---------------------------------------------------------------------------
// R-6 缺口⑤：曲线图元经三角化进入 wavefront 路径模式（逐位对拍）
// ---------------------------------------------------------------------------

/// **R-6 等价路径判据**：曲线 `CurveReq::Cylinder` 经 `tessellate` 三角化后，
/// wavefront 路径模式与 CPU 参考逐位一致（differing_pixels = 0）。
///
/// 这条闭合矩阵 R-6 缺口⑤「wavefront 路径模式 kernel 的 scene_intersect 只吃
/// 三角形，曲线段不在本 kernel」：产品路径（`translate_scene`）对曲线图元做的
/// 正是三角化，wavefront 吃**同一份**三角形流（本测试一次生成、两侧共享），
/// 与 CPU 参考逐位相等 ⇒ 「曲线进 wavefront」的等价路径成立。
#[test]
fn r6_curve_via_tessellation_wavefront_parity() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (16u32, 16u32, 2u32, 3u32);
    let tol = 0.004f32;
    let cpu = curve_cpu_reference(w, h, spp, bounces, tol);

    // 场景有效性前置：柱子必须真的挡光（与无柱场景有差异），防退化场景假绿。
    let plain = cpu_reference(w, h, spp, bounces);
    let diff_px = plain
        .iter()
        .zip(cpu.iter())
        .filter(|(a, b)| {
            a[0].to_bits() != b[0].to_bits()
                || a[1].to_bits() != b[1].to_bits()
                || a[2].to_bits() != b[2].to_bits()
        })
        .count();
    assert!(
        diff_px > 0,
        "柱子场景与无柱场景逐位相同：曲线三角化没有改变几何（测试构造失效）"
    );

    let req = curve_request(w, h, spp, bounces, tol);
    let rep = backend.render_wavefront_frame(&req);
    assert_eq!(rep.status, prism_gpu::vulkan::compute::DispatchStatus::Ok, "{}", rep.detail);
    assert_eq!(rep.hdr.len(), (w * h) as usize);
    assert_bitwise_equal(w, h, &cpu, &rep.hdr, "R-6 曲线（三角化）spp2 bounces3");
    println!(
        "[R-6] 曲线三角化（{} 三角，tol={tol}）进 wavefront：differing_pixels = 0/{}",
        req.triangles.len(),
        w * h
    );
}
