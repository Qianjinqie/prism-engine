//! **R-10 设备采数**：HDR 环境 / IBL 的白炉与无偏判据在 **llvmpipe 设备上**跑通。
//!
//! # 判据口径（照抄 `docs/v2.5-W2-光照报告.md` §3.1 / §3.3，阈值不放宽）
//!
//! * **V1 白炉**：常数环境 `L = 1` + Lambert `albedo = 1` 闭合盒，8 弹射
//!   ⇒ 像素 = 1.0，相对误差 ≤ 1e-2；
//! * **V3 无偏**：MIS 全开 vs 均匀球面暴力积分（参考与 CDF 采样完全解耦），
//!   相对误差 ≤ 0.5%。
//!
//! # 设备形态（与 CPU 版判据的差异，如实声明）
//!
//! * 设备 spp 上限 64（wavefront accumulate 的固定形状归约树），CPU 版 V1 用
//!   2048 spp / V3 用 8192 spp——设备版用**多 frame 累积**补样本数
//!   （`frame` 进采样流基种子，批次独立同分布，与 w2 测试的 `pixel_batches`
//!   同一机理），测试侧对逐 frame 的 HDR 帧做**固定顺序** f32 累加平均；
//! * 每条判据都带 **GPU ↔ CPU 逐位对拍**（R-1 机制在 env 场景的扩展）：
//!   `prism_rt::wavefront::render_frame` 与 WGSL wavefront 同 env 同流；
//! * **V4 互校的设备形态** = 两条逐位链之和：环境腿（本文件 V1/V3 的 env
//!   场景逐位）+ 球灯腿（`qa_wavefront_parity.rs` 的 Sphere 灯场景逐位）——
//!   两种光源形态都与 CPU 同流，"互校"的设备判据由此闭合（球灯 + 发光几何
//!   的解析对拍形态在 CPU 通路已有 0.023% 数字，见 w2_a2_ibl）。
//!
//! # 环境自门控（禁 `#[ignore]`）
//!
//! 无 Vulkan 设备时打印探测报告后返回（不 fail、不假装跑过）。

#[path = "common/r1_parity.rs"]
#[allow(dead_code)]
mod r1_parity;

use r1_parity::{camera_uniform, device_ready, sobol_table_bytes};
use prism_core::EvalCtx;
use prism_gpu::vulkan::wavefront::{
    WavefrontMaterial, WavefrontMode, WavefrontRequest, WavefrontTri,
};
use prism_gpu::vulkan::VulkanBackend;
use prism_gpu::AestheticReq;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::env::{uv_to_dir, EnvCdf};
use prism_rt::integrator::{MaterialKind, RenderParams, Scene};
use prism_rt::texsample::TexTable;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// 共享夹具
// ---------------------------------------------------------------------------

/// u32 blob → `WavefrontRequest.env_blob`（f32 位模式承载 u32，与 WGSL 的
/// `bitcast` 读取约定一致）。
fn blob_to_f32(words: &[u32]) -> Vec<f32> {
    words.iter().map(|&w| f32::from_bits(w)).collect()
}

/// 立方盒（中心 `c`、边长 `s`）的 12 个三角形（面法线朝外，绕序 a→b→c 右手）。
fn box_tris(c: Vec3, s: f32) -> Vec<(Vec3, Vec3, Vec3)> {
    let h = s * 0.5;
    let p = |ix: i32, iy: i32, iz: i32| {
        Vec3::new(
            c.x + h * ix as f32,
            c.y + h * iy as f32,
            c.z + h * iz as f32,
        )
    };
    // 8 顶点
    let v = [
        p(-1, -1, -1), p(1, -1, -1), p(1, 1, -1), p(-1, 1, -1),
        p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1),
    ];
    // 6 面 × 2 三角（绕序保证外法线）
    let q = [
        [0, 3, 2, 1], // -z
        [4, 5, 6, 7], // +z
        [0, 1, 5, 4], // -y
        [3, 7, 6, 2], // +y
        [0, 4, 7, 3], // -x
        [1, 2, 6, 5], // +x
    ];
    let mut out = Vec::new();
    for f in q {
        out.push((v[f[0]], v[f[1]], v[f[2]]));
        out.push((v[f[0]], v[f[2]], v[f[3]]));
    }
    out
}

/// R-10 GPU 请求骨架（路径模式 + 真 Sobol 表 + 相机 uniform + env blob）。
fn r10_request(
    w: u32,
    h: u32,
    spp: u32,
    bounces: u32,
    frame: u32,
    env_blob: &[f32],
    tris: Vec<(Vec3, Vec3, Vec3)>,
    albedo: Vec3,
    cam_uniform: [u8; 96],
) -> WavefrontRequest {
    let mat = WavefrontMaterial {
        base_color: albedo,
        roughness: 0.0,
        f0: Vec3::splat(0.04),
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.5,
        emissive: Vec3::splat(0.0),
        kind: 0, // Lambert
    };
    WavefrontRequest {
        width: w,
        height: h,
        spp,
        max_bounces: bounces,
        seed: 2026,
        mode: WavefrontMode::Path,
        camera: prism_gpu::CameraReq {
            eye: Vec3::new(0.0, 0.0, -3.0),
            target: Vec3::splat(0.0),
            fov: 45.0 * PI / 180.0,
            aspect: w as f32 / h as f32,
        },
        aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        spheres: Vec::new(),
        triangles: tris
            .iter()
            .map(|&(a, b, c)| WavefrontTri { a, b, c })
            .collect(),
        materials: vec![mat],
        prim_mat: vec![0; tris.len()],
        lights: Vec::new(), // 灯 = 环境（V1/V3 场景无解析灯）
        ambient: Vec3::splat(0.0),
        env_blob: env_blob.to_vec(),
        sobol_table: Some(sobol_table_bytes()),
        node_id: 11,
        frame,
        camera_uniform: Some(cam_uniform),
        stage_stop: None,
        readback_slots: false,
    }
}

/// R-10 场景的 CPU 侧构造（与 GPU 请求同一几何/env，逐位同源）。
fn r10_cpu_scene(
    env_w: u32,
    env_h: u32,
    env_pix: &[[f32; 4]],
    tris: &[(Vec3, Vec3, Vec3)],
    albedo: Vec3,
) -> Scene {
    let mut scene = Scene::default();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for &(a, b, c) in tris {
        let base = positions.len() as u32;
        positions.push(a);
        positions.push(b);
        positions.push(c);
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    scene.add_mesh(
        positions,
        indices,
        MaterialKind::Lambert { albedo },
    );
    scene.add_environment(env_w, env_h, env_pix);
    scene.assert_layout();
    scene
}

/// R-10 CPU 参考帧（`prism_rt::wavefront::render_frame`，clamp_indirect 关）。
/// 相机由调用方传入（V1 正视盒 / V3 俯视地板不同视点，逐位同源由
/// `camera_uniform` 保障 GPU 侧同一相机）。
fn r10_cpu_frame(
    scene: &Scene,
    cam: &Camera,
    frame: u32,
    w: u32,
    h: u32,
    spp: u32,
    bounces: u32,
) -> Vec<[f32; 4]> {
    let ctx = EvalCtx::new(2026, 11, frame as u64);
    let mut params = RenderParams::new(spp, bounces);
    params.clamp_indirect = None;
    prism_rt::wavefront::render_frame(scene, cam, &ctx, (w, h), &params)
}

/// 多 frame 固定顺序平均（测试侧判据算术；顺序固定 ⇒ GPU/CPU 均值逐位可比）。
fn frame_mean(frames: &[Vec<[f32; 4]>]) -> Vec<[f32; 4]> {
    let n = frames.len() as f32;
    let npix = frames[0].len();
    let mut acc = vec![[0.0f32; 4]; npix];
    for f in frames {
        for (a, s) in acc.iter_mut().zip(f.iter()) {
            for c in 0..4 {
                a[c] += s[c];
            }
        }
    }
    acc.iter_mut().for_each(|a| {
        for c in 0..4 {
            a[c] /= n;
        }
    });
    acc
}

/// 逐位比对（4 通道 `to_bits` 全等）+ 首差异定位。
fn assert_bitwise(w: u32, h: u32, cpu: &[[f32; 4]], gpu: &[[f32; 4]], label: &str) {
    assert_eq!(cpu.len(), (w * h) as usize, "{label}：CPU 帧长不符");
    assert_eq!(gpu.len(), (w * h) as usize, "{label}：GPU 帧长不符");
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits()],
            [b[0].to_bits(), b[1].to_bits(), b[2].to_bits(), b[3].to_bits()],
            "{label}：第 {i} 像素逐位不符（cpu bits vs gpu bits）"
        );
    }
    println!("[{label}] {w}×{h} 逐位一致（{} 像素）", cpu.len());
}

// ---------------------------------------------------------------------------
// V1 白炉（设备）：常数环境 + 白盒 ⇒ 像素 = 1.0（≤1e-2）+ GPU↔CPU 逐位
// ---------------------------------------------------------------------------

#[test]
fn r10_v1_white_furnace_on_device() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    // 场景：L=1 常数环境（8×4 位图）+ 白 Lambert 立方盒（边长 1，原点）。
    let (ew, eh) = (8u32, 4u32);
    let env_pix: Vec<[f32; 4]> = vec![[1.0, 1.0, 1.0, 1.0]; (ew * eh) as usize];
    let tris = box_tris(Vec3::splat(0.0), 1.0);
    let albedo = Vec3::splat(1.0);

    let scene = r10_cpu_scene(ew, eh, &env_pix, &tris, albedo);
    let cdf: &EnvCdf = scene.env().expect("环境 CDF 必须已建");
    let blob = blob_to_f32(&prism_rt::env::blob_words(cdf, &env_pix));

    // 设备采样预算：spp=32 × 32 frame = 1024 样本/像素（spp 上限 64 的约束下，
    // 多 frame 累积补样本；白炉的直接光 NEE 对常数环境是常数，方差在间接段）。
    let (w, h, spp, bounces, frames_n) = (8u32, 8u32, 32u32, 8u32, 32u32);

    // 相机：正视盒（CPU/GPU 逐位同源：Camera 位型直接进 camera_uniform）。
    let cam = Camera::new(
        Vec3::new(0.0, 0.0, -3.0),
        Vec3::splat(0.0),
        Vec3::new(0.0, 1.0, 0.0),
        45.0,
        0.0,
        3.0,
        0.0,
        0.0,
    );
    let cam_uniform = camera_uniform(&cam, w as f32 / h as f32);

    let mut gpu_frames = Vec::new();
    let mut cpu_frames = Vec::new();
    for f in 0..frames_n {
        let req = r10_request(w, h, spp, bounces, f, &blob, tris.clone(), albedo, cam_uniform);
        let rep = backend.render_wavefront_frame(&req);
        assert_eq!(
            rep.status,
            prism_gpu::vulkan::compute::DispatchStatus::Ok,
            "{}",
            rep.detail
        );
        gpu_frames.push(rep.hdr.clone());
        cpu_frames.push(r10_cpu_frame(&scene, &cam, f, w, h, spp, bounces));
    }

    // 判据 A：每个 frame 的 GPU ↔ CPU 逐位一致（env 采样的设备确定性）。
    for (f, (g, c)) in gpu_frames.iter().zip(cpu_frames.iter()).enumerate() {
        assert_bitwise(w, h, c, g, &format!("V1 白炉 frame{f} GPU↔CPU"));
    }

    // 判据 B：白炉绝对判据 |L − 1| ≤ 1e-2（对**累积均值**）。
    let mean = frame_mean(&cpu_frames);
    let mut worst = 0.0f32;
    for p in &mean {
        for ch in [p[0], p[1], p[2]] {
            worst = worst.max((ch - 1.0).abs());
        }
    }
    let mut nonzero = 0usize;
    for p in &mean {
        if p[0] > 0.5 && p[1] > 0.5 && p[2] > 0.5 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero >= (w * h) as usize / 2,
        "白炉场景过半像素亮度 ≤ 0.5：环境没被消费（场景构造失效）"
    );
    println!(
        "[R-10 V1 设备白炉] {w}×{h} spp{spp}×{frames_n}帧 b{bounces}：\
         最差 |L−1| = {worst:.3e}（阈值 1e-2；CPU 通路 2048spp 版 1.932e-3）"
    );
    assert!(worst <= 1e-2, "设备白炉最差相对误差 {worst} 超过 1e-2");
}

// ---------------------------------------------------------------------------
// V3 无偏（设备）：MIS 全开 vs 均匀球面暴力积分（≤0.5%）+ GPU↔CPU 逐位
// ---------------------------------------------------------------------------

/// 基 2 radical inverse（与 `w2_a2_ibl.rs` 同式；无 RNG、无依赖）。
fn radical_inverse_base2(mut k: u32) -> f32 {
    k = (k << 16) | (k >> 16);
    k = ((k & 0x5555_5555) << 1) | ((k & 0xAAAA_AAAA) >> 1);
    k = ((k & 0x3333_3333) << 2) | ((k & 0xCCCC_CCCC) >> 2);
    k = ((k & 0x0F0F_0F0F) << 4) | ((k & 0xF0F0_F0F0) >> 4);
    k = ((k & 0x00FF_00FF) << 8) | ((k & 0xFF00_FF00) >> 8);
    k as f32 * 2.328_306_4e-10
}

/// 辐照度 `E(n) = ∫ L(ω)·max(0, n·ω) dω` 的暴力参考（均匀球面 QMC，1e6 点）。
///
/// 与 CDF 采样完全解耦：方向来自等面积映射，只借 `env.radiance`（查表 =
/// "环境图长什么样"的输入定义）。
fn brute_force_irradiance(env: &EnvCdf, table: TexTable<'_>, n: Vec3, samples: u32) -> Vec3 {
    let mut acc = Vec3::splat(0.0);
    let inv = 1.0 / samples as f32;
    for k in 0..samples {
        let u1 = (k as f32 + 0.5) * inv;
        let u2 = radical_inverse_base2(k);
        let z = 1.0 - 2.0 * u1;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * PI * u2;
        let d = Vec3::new(r * phi.cos(), z, r * phi.sin());
        let c = n.dot(&d);
        if c > 0.0 {
            acc = acc + env.radiance(table, d) * c;
        }
    }
    acc * (4.0 * PI * inv)
}

/// 平滑非均匀环境（宽瓣，64×32；与 `w2_a2_ibl.rs::smooth_env` 同式同轴）。
fn smooth_env(w: u32, h: u32) -> Vec<[f32; 4]> {
    let up = Vec3::new(0.3, 1.0, 0.2).normalize();
    let mut out = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let u = (i as f32 + 0.5) / w as f32;
            let v = (j as f32 + 0.5) / h as f32;
            let d = uv_to_dir(u, v);
            let lobe = d.dot(&up).max(0.0).powi(3);
            let c = Vec3::new(0.25 + 2.0 * lobe, 0.18 + 1.4 * lobe, 0.12 + 0.9 * lobe);
            out.push([c.x, c.y, c.z, 1.0]);
        }
    }
    out
}

#[test]
fn r10_v3_unbiased_on_device() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    // 场景：平滑环境（64×32）+ 大 Lambert 地板三角（albedo=0.8，法线 +y）。
    // 命中点法线 +y ⇒ 暴力参考对 n=(0,1,0) 积分。
    let (ew, eh) = (64u32, 32u32);
    let env_pix = smooth_env(ew, eh);
    let albedo = Vec3::splat(0.8);
    let y = 0.0f32;
    let tris = vec![
        (
            Vec3::new(-2.0, y, -2.0),
            Vec3::new(-2.0, y, 2.0),
            Vec3::new(2.0, y, 2.0),
        ),
        (
            Vec3::new(-2.0, y, -2.0),
            Vec3::new(2.0, y, 2.0),
            Vec3::new(2.0, y, -2.0),
        ),
    ];
    let scene = r10_cpu_scene(ew, eh, &env_pix, &tris, albedo);
    let cdf: &EnvCdf = scene.env().expect("环境 CDF 必须已建");
    let blob = blob_to_f32(&prism_rt::env::blob_words(cdf, &env_pix));

    // 相机俯视地板中心；中心像素命中 (0,0,0)，法线 +y。
    // （分辨率 1×1：整帧就是那个"点"；fov 0.2° 与 CPU V3 同口径——像素足迹
    // 远小于被积函数的变化尺度，点值对拍才成立。）
    let (w, h) = (1u32, 1u32);
    let eye = Vec3::new(0.0, 3.0, 0.0);
    let cam_fov = 0.2f32;
    let cam = Camera::new(eye, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), cam_fov, 0.0, 3.0, 0.0, 0.0);

    // 设备采样预算：spp=64 × 128 frame = 8192 样本（与 CPU 通路 8192spp 同预算）。
    // 注意收敛形态：多 frame 是**独立批次**平均（每帧一个 64 样本 Sobol 批次，
    // 误差 ~1/√M），拿不到单条 8192 长 Sobol 序列的 ~1/N 低偏差——实测 32 帧
    // 最差 7.3e-3（≈1.8e-2/√32），128 帧进 5e-3 阈内且有余量。
    let (spp, bounces, frames_n) = (64u32, 2u32, 128u32);
    let mut gpu_frames = Vec::new();
    let mut cpu_frames = Vec::new();
    for f in 0..frames_n {
        let mut req = r10_request(
            w,
            h,
            spp,
            bounces,
            f,
            &blob,
            tris.clone(),
            albedo,
            camera_uniform(&cam, 1.0),
        );
        req.camera = prism_gpu::CameraReq {
            eye,
            target: Vec3::splat(0.0),
            fov: cam_fov * PI / 180.0,
            aspect: 1.0,
        };
        let rep = backend.render_wavefront_frame(&req);
        assert_eq!(
            rep.status,
            prism_gpu::vulkan::compute::DispatchStatus::Ok,
            "{}",
            rep.detail
        );
        gpu_frames.push(rep.hdr.clone());
        cpu_frames.push(r10_cpu_frame(&scene, &cam, f, w, h, spp, bounces));
    }

    // 判据 A：每个 frame 的 GPU ↔ CPU 逐位一致。
    for (f, (g, c)) in gpu_frames.iter().zip(cpu_frames.iter()).enumerate() {
        assert_bitwise(w, h, c, g, &format!("V3 无偏 frame{f} GPU↔CPU"));
    }

    // 判据 B：累积均值 vs 暴力积分 ≤ 0.5%（Lambert：L_out = albedo/π · E）。
    let table = TexTable::new(&scene.textures, &scene.texels);
    let e = brute_force_irradiance(cdf, table, Vec3::new(0.0, 1.0, 0.0), 1_000_000);
    let want = Vec3::new(
        e.x * albedo.x,
        e.y * albedo.y,
        e.z * albedo.z,
    ) * (1.0 / PI);
    let mean = frame_mean(&cpu_frames)[0];
    let mut worst = 0.0f32;
    for (g, t) in [(mean[0], want.x), (mean[1], want.y), (mean[2], want.z)] {
        worst = worst.max(((g - t) / t).abs());
    }
    println!(
        "[R-10 V3 设备无偏] 1×1 spp{spp}×{frames_n}帧（{} 样本）b{bounces}：\
         渲染 [{:.6},{:.6},{:.6}] vs 暴力参考 [{:.6},{:.6},{:.6}]，\
         最差相对误差 {worst:.3e}（阈值 0.5%；CPU 通路 8192spp 版 1.116e-4）",
        spp * frames_n,
        mean[0], mean[1], mean[2],
        want.x, want.y, want.z,
    );
    assert!(worst <= 5e-3, "设备 V3 三通道最差相对误差 {worst:.3e} 超过 0.5%");
}
