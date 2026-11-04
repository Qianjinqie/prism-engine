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

// G-8① 判据 I1–I4 的额外导入（模块级；与上面的 R-1 判据共用同一份场景基座）。
use prism_core::EvalCtx;
use prism_gpu::instance::{InstanceReq, INSTANCE_MAT_FROM_GEOM as GPU_MAT_FROM_GEOM};
use prism_gpu::vulkan::wavefront::{
    append_instance_table, validate_instance_tail, WavefrontTri,
};
use prism_rt::instance::{Instance, INSTANCE_MAT_FROM_GEOM as CPU_MAT_FROM_GEOM};
use prism_rt::integrator::{MaterialKind, RenderParams, Scene};
use prism_rt::light::Light;

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

// ---------------------------------------------------------------------------
// G-8① 收口：wavefront 腿几何实例层级（I1 / I3 / I4）
// ---------------------------------------------------------------------------
//
// CPU 权威 = `prism_rt::wavefront::render_frame(&Scene-with-instances, ...)`：
// `Scene::instance_nearest` 做「世界射线 → 逆仿射 → 局部 BLAS → 世界 t」，
// `normal_at` 的 Instance 臂用 `det(A)·A⁻ᵀ·n`。GPU 被测 =
// `rt_scene.wgsl::scene_intersect` 的实例族（同一套算术、同一条 112 B 记录）。
// 判据仍是逐像素 `to_bits()` 全等，不放宽任何尺子。
//
// 实例数据经现有 group(2) b2 `prim_mat` / b3 `tris` 承载（零绑定变更），
// 由 `append_instance_table` 复用 `crate::instance::encode_instances` 编码；
// Vulkan 与 WebGPU 消费同一张 `build_resources` 资源表。

/// 3 个实例的仿射（行主序 3×4）：纯平移 / **镜像 det<0** + 平移 / 恒等。
///
/// 整数平移与轴向镜像在 f32 下逐位精确（`-1.0` 与整数坐标都不引入舍入），
/// 这是 CPU 权威与 GPU 实例遍历能给出 `differing_pixels = 0` 的数值条件——
/// 选的是条件数好的场景，不是把阈值放宽。
const I_AFFINES: [[f32; 12]; 3] = [
    [1.0, 0.0, 0.0, -0.9, 0.0, 1.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.2],
    [-1.0, 0.0, 0.0, 0.9, 0.0, 1.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.4],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
];
/// 每个实例的 `geom_id`（2 个原型 ⇒ 多原型 BLAS 索引真的被遍历）。
const I_GEOMS: [u32; 3] = [0, 1, 0];

/// 2 个原型的局部三角形（+Z 朝向方板；尺寸不同 ⇒ 多原型可辨）。
fn i_geom_tris() -> Vec<Vec<WavefrontTri>> {
    let board = |s: f32| -> Vec<WavefrontTri> {
        let p0 = Vec3::new(-s, -s, 0.0);
        let p1 = Vec3::new(s, -s, 0.0);
        let p2 = Vec3::new(s, s, 0.0);
        let p3 = Vec3::new(-s, s, 0.0);
        vec![
            WavefrontTri { a: p0, b: p1, c: p2 },
            WavefrontTri { a: p0, b: p2, c: p3 },
        ]
    };
    vec![board(0.45), board(0.30)]
}

/// CPU 侧实例（与 GPU 侧**同一组仿射**，分别过两个 crate 的 `from_affine`）。
fn i_cpu_instances() -> Vec<Instance> {
    I_AFFINES
        .iter()
        .zip(I_GEOMS)
        .map(|(a, g)| {
            Instance::from_affine(*a, g, CPU_MAT_FROM_GEOM).expect("可逆仿射")
        })
        .collect()
}

/// GPU 侧实例（112 B 编码的唯一事实源是 `crate::instance`）。
fn i_gpu_instances() -> Vec<InstanceReq> {
    I_AFFINES
        .iter()
        .zip(I_GEOMS)
        .map(|(a, g)| {
            InstanceReq::from_affine(*a, g, GPU_MAT_FROM_GEOM).expect("可逆仿射")
        })
        .collect()
}

/// CPU 权威场景：2 原型 + 3 实例（含镜像）+ Sphere 灯（不注册为几何）。
fn i_cpu_scene() -> Scene {
    let mut scene = Scene::default();
    let albedos = [Vec3::new(0.72, 0.71, 0.69), Vec3::new(0.2, 0.5, 0.3)];
    for (g, tris) in i_geom_tris().iter().enumerate() {
        let mut pos: Vec<Vec3> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        for t in tris {
            let b = pos.len() as u32;
            pos.extend_from_slice(&[t.a, t.b, t.c]);
            idx.extend_from_slice(&[b, b + 1, b + 2]);
        }
        let gid = scene.add_instance_geom(
            pos,
            idx,
            MaterialKind::Lambert { albedo: albedos[g] },
        );
        assert_eq!(gid, g as u32, "原型注册序必须等于 geom_id");
    }
    scene.add_instances(i_cpu_instances());
    scene.lights = vec![Light::Sphere {
        center: Vec3::new(0.0, 1.6, 0.3),
        radius: 0.3,
        radiance: Vec3::new(14.0, 12.0, 9.0),
    }];
    scene
}

/// 与 `parity_request` 逐位同源的相机（seed/node_id/frame 亦同源）。
fn i_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.8, -2.5),
        Vec3::new(0.0, 0.4, 0.5),
        Vec3::new(0.0, 1.0, 0.0),
        55.0,
        0.0,
        3.1,
        0.0,
        0.0,
    )
}

fn i_cpu_reference(w: u32, h: u32, spp: u32, bounces: u32) -> Vec<[f32; 4]> {
    let scene = i_cpu_scene();
    let ctx = EvalCtx::new(2026, 7, 3);
    let mut params = RenderParams::new(spp, bounces);
    params.clamp_indirect = None;
    prism_rt::wavefront::render_frame(&scene, &i_camera(), &ctx, (w, h), &params)
}

/// 与 `parity_request` 同一基座（相机 uniform / Sobol 表 / 灯 / 2 个 Lambert
/// 材质逐位同源），把普通三角形换成「2 原型 + 3 实例」的实例表。
fn i_request(
    w: u32,
    h: u32,
    spp: u32,
    bounces: u32,
) -> prism_gpu::vulkan::wavefront::WavefrontRequest {
    let mut req = parity_request(w, h, spp, bounces);
    req.triangles.clear();
    req.prim_mat.clear();
    let geoms = i_geom_tris();
    let instances = i_gpu_instances();
    append_instance_table(&mut req.triangles, &mut req.prim_mat, &instances, &geoms, &[0, 1])
        .expect("实例表编码必须成功");
    validate_instance_tail(&req).expect("实例描述符必须自洽");
    req
}

fn i_differing(cpu: &[[f32; 4]], gpu: &[[f32; 4]]) -> usize {
    cpu.iter()
        .zip(gpu.iter())
        .filter(|(a, b)| {
            a[0].to_bits() != b[0].to_bits()
                || a[1].to_bits() != b[1].to_bits()
                || a[2].to_bits() != b[2].to_bits()
        })
        .count()
}

/// **I1**：3 实例（含镜像 `det<0`）+ 2 原型，llvmpipe 上 wavefront ≡ CPU 权威，
/// `differing_pixels = 0`。
#[test]
fn i1_wavefront_instances_match_cpu_authority_bitwise() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (16u32, 16u32, 2u32, 3u32);
    let cpu = i_cpu_reference(w, h, spp, bounces);
    let max_l = cpu
        .iter()
        .map(|p| p[0] + p[1] + p[2])
        .fold(0.0f32, f32::max);
    assert!(max_l > 0.01, "CPU 参考无能量（max_luma={max_l}）：实例场景构造失效");
    assert!(i_gpu_instances()[1].is_mirrored(), "判据必须含 det<0 镜像实例");
    assert_eq!(i_gpu_instances().len(), 3, "2~3 个实例");

    let req = i_request(w, h, spp, bounces);
    let rep = backend.render_wavefront_frame(&req);
    assert_eq!(
        rep.status,
        prism_gpu::vulkan::compute::DispatchStatus::Ok,
        "{}",
        rep.detail
    );
    let diff = i_differing(&cpu, &rep.hdr);
    println!(
        "[I1] llvmpipe 3 实例/2 原型/含镜像：differing_pixels = {diff}/{}（{} 三角，{} 实例字）",
        w * h,
        req.triangles.len(),
        req.prim_mat.len()
    );
    assert_bitwise_equal(w, h, &cpu, &rep.hdr, "I1 wavefront 实例 vs CPU 权威");
}

/// 从 `--instance-parity --json` 报告里按顶层字段名取整数（纯 ASCII 数值，
/// 无需引入 JSON 库；顶层字段排在 `results` 之前，故 `find` 命中顶层）。
fn i2_json_i64(report: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\":");
    let at = report
        .find(&pat)
        .unwrap_or_else(|| panic!("[I2] 报告缺少字段 {key}：\n{report}"))
        + pat.len();
    let rest = report[at..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse::<i64>().unwrap_or_else(|e| {
        panic!("[I2] 字段 {key} 不是整数：{e}；片段 `{}`", &rest[..end])
    })
}

/// **I2**：浏览器 WebGPU（复用上一轮既有的 Xvfb + 系统 Chromium +
/// SwiftShader 真机 harness，零新建 harness）上，同一份 112 B 实例表驱动的
/// `rt_scene.wgsl::scene_intersect` 与 CPU 权威逐位一致，`differing_pixels = 0`；
/// 故意改错一个实例逆仿射的负控与正控参考必须非 0 差（判别力前置）。
#[test]
fn i2_browser_webgpu_instances_match_cpu_authority_bitwise() {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/web/wgsl_lib_compile.mjs");
    let proc = std::process::Command::new("node")
        .args([script, "--instance-parity", "--json"])
        .output()
        .unwrap_or_else(|e| panic!("[I2] 无法启动 node 复用既有 WebGPU harness：{e}"));
    let stdout = String::from_utf8_lossy(&proc.stdout);
    let stderr = String::from_utf8_lossy(&proc.stderr);
    let at = stdout.find('{').unwrap_or(0);
    let report = &stdout[at..];
    assert!(
        proc.status.success() && report.contains("\"i2_ok\": true"),
        "[I2] 浏览器 WebGPU 实例对拍失败（exit={:?}）：\n{report}\n{stderr}",
        proc.status.code()
    );
    let total = i2_json_i64(report, "dispatch");
    let pos = i2_json_i64(report, "positive_differing_pixels");
    let neg = i2_json_i64(report, "negative_differing_pixels");
    let tamper = i2_json_i64(report, "tamper_vs_positive_differing_pixels");
    println!(
        "[I2] 浏览器 WebGPU（SwiftShader）3 实例/2 原型/含镜像：differing_pixels = {pos}/{total}"
    );
    println!(
        "[I2] 负控自身 differing_pixels = {neg}/{total}；负控 vs 正控 = {tamper}/{total}（必须 > 0）"
    );
    assert_eq!(pos, 0, "[I2] 浏览器 WebGPU 实例场景必须 0 差");
    assert_eq!(
        neg, 0,
        "[I2] 负控与改坏后的 CPU 权威也必须 0 差（证明 GPU 真消费实例表）"
    );
    assert!(tamper > 0, "[I2] 改错逆仿射必须产生非 0 差，否则判据无判别力");
}

/// **I3**：实例帧双跑逐字节一致。
#[test]
fn i3_instance_frame_two_pass_bitwise_identical() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let req = i_request(16, 16, 2, 3);
    let a = backend.render_wavefront_frame(&req);
    let b = backend.render_wavefront_frame(&req);
    assert_eq!(
        a.status,
        prism_gpu::vulkan::compute::DispatchStatus::Ok,
        "{}",
        a.detail
    );
    assert_eq!(
        b.status,
        prism_gpu::vulkan::compute::DispatchStatus::Ok,
        "{}",
        b.detail
    );
    let diff = a
        .hdr
        .iter()
        .zip(b.hdr.iter())
        .filter(|(x, y)| x.map(f32::to_bits) != y.map(f32::to_bits))
        .count();
    println!("[I3] 实例帧双跑 differing_pixels = {diff}/{}", a.hdr.len());
    assert_eq!(diff, 0, "实例帧两次派发必须逐位一致");
}

/// **I4 判别力前置**：故意改错一个实例的逆仿射（shader 真消费的
/// `inverse[3]` 平移字）必须 FAIL；正控必须 `differing_pixels = 0`。
/// 这条排除「根本没消费实例表」的假绿。
#[test]
fn i4_tampered_instance_affine_must_fail() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h, spp, bounces) = (16u32, 16u32, 2u32, 3u32);
    let cpu = i_cpu_reference(w, h, spp, bounces);

    // 正控：正确实例表 ⇒ 0 差。
    let ok_rep = backend.render_wavefront_frame(&i_request(w, h, spp, bounces));
    assert_eq!(
        ok_rep.status,
        prism_gpu::vulkan::compute::DispatchStatus::Ok,
        "{}",
        ok_rep.detail
    );
    let ok_diff = i_differing(&cpu, &ok_rep.hdr);
    assert_eq!(ok_diff, 0, "I4 正控必须 0 差");

    // 负控：篡改第 1 条实例的逆平移字（inverse 起点 12 + 平移槽 3）。
    let mut bad = i_request(w, h, spp, bounces);
    let words = bad.prim_mat.len();
    let ib = bad.prim_mat[words - 5] as usize; // 描述符 [3] = instance_base
    let off = ib + 12 + 3;
    let v = f32::from_bits(bad.prim_mat[off]);
    bad.prim_mat[off] = (v + 0.75).to_bits();
    let bad_rep = backend.render_wavefront_frame(&bad);
    assert_eq!(
        bad_rep.status,
        prism_gpu::vulkan::compute::DispatchStatus::Ok,
        "{}",
        bad_rep.detail
    );
    let bad_diff = i_differing(&cpu, &bad_rep.hdr);
    println!(
        "[I4] 正控 differing_pixels = {ok_diff}/{}；改错一个实例逆仿射 → {bad_diff}/{}",
        w * h,
        w * h
    );
    assert!(
        bad_diff > 0,
        "改错仿射必须 FAIL：仍是 0 差说明实例表根本没被消费（假绿）"
    );
}
