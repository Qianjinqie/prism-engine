//! **T6 验收测试**：WGSL wavefront 五段派发（R-13 / A6）。
//!
//! # 闭环链（真机执行，非模拟）
//!
//! ```text
//! 5 份入库 .spv（tools/wgsl2spv --all 产物，D-21）
//!   → 5 条 VkComputePipeline（同一 pipeline layout，4 个 descriptor set）
//!   → 一条 command buffer：generate → [shade → connect → extend]×N → accumulate
//!   → 段间 vkCmdPipelineBarrier（memory barrier）→ 一次 submit → fence
//!   → 读回 accum（兼容模式 = 打包 RGBA8；路径模式 = HDR vec4）
//! ```
//!
//! # 三条判据
//!
//! * **W1 逐位等价**：同一场景 / 同 spp / 同 seed，wavefront（[`WavefrontMode::CpuRefCompat`]）
//!   与 megakernel（[`prism_gpu::vulkan::scene`]，`spirv_rt` 那条）在软件 ICD 上
//!   **`differing_pixels == 0`**。差异若出现，本文件把它**定位到像素 + 阶段**
//!   （`generate` 还是 `shade`），不是"统计等价"。
//! * **W2 确定性**：同请求连跑两次逐字节相同（两种模式都测）。
//! * **W3 弹射深度**：路径模式在**设备上**实测 4 vs 8 弹射的 PSNR / 相对 RMSE
//!   （替换／并置报告里既有的"估计量级"数字）。
//!
//! # 环境自门控（禁 `#[ignore]`）
//!
//! 无 Vulkan 设备时打印探测报告后返回（不 fail、不假装跑过）。本沙箱装有
//! lavapipe，故真机用例**真实执行**；设备名与全部统计量打进测试输出。

use prism_gpu::vulkan::compute::DispatchStatus;
use prism_gpu::vulkan::spirv_rt::camera_basis;
use prism_gpu::vulkan::wavefront::{
    WavefrontLight, WavefrontMaterial, WavefrontMode, WavefrontRequest, WavefrontTri, STAGES,
    PATH_SLOT_BYTES,
};
use prism_gpu::vulkan::VulkanBackend;
use prism_gpu::{
    AestheticReq, CameraReq, LightReq, SceneRenderRequest, SphereReq,
};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 自门控 / 公共件
// ---------------------------------------------------------------------------

/// 无设备时打印探测报告并返回 `false`（调用方据此 skip，而不是 fail）。
fn device_ready(backend: &VulkanBackend) -> bool {
    let report = VulkanBackend::probe();
    if !report.device_created {
        println!(
            "[skip] 本机无可用 Vulkan 设备：supported_platform={} loader_loaded={} \
             instance_created={} device_count={} reason={:?} → 跳过真机判据（不 panic）",
            report.supported_platform,
            report.loader_loaded,
            report.instance_created,
            report.device_count,
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

/// 是否软件 ICD（lavapipe / llvmpipe / swiftshader）：W1 在此收紧到逐位一致。
fn is_software_icd(device: &str) -> bool {
    let d = device.to_ascii_lowercase();
    d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
}

/// 一帧 RGBA8 的逐像素差异摘要。
struct Diff {
    /// 任一通道有差异的像素数。
    differing: usize,
    /// 加权重差：`Σ|Δ|`（诊断）。
    sum_abs: u64,
    /// 首个差异像素 `(x, y, megakernel, wavefront)`。
    first: Option<(u32, u32, [u8; 4], [u8; 4])>,
}

/// 逐像素比对两帧 RGBA8（同尺寸；只比 RGB，alpha 单独断言）。
fn diff_rgba(w: u32, h: u32, a: &[u8], b: &[u8]) -> Diff {
    assert_eq!(a.len(), (w as usize) * (h as usize) * 4, "帧 A 长度不符");
    assert_eq!(b.len(), (w as usize) * (h as usize) * 4, "帧 B 长度不符");
    let total = (w as usize) * (h as usize);
    let mut differing = 0usize;
    let mut sum_abs = 0u64;
    let mut first: Option<(u32, u32, [u8; 4], [u8; 4])> = None;
    for i in 0..total {
        let pa = &a[i * 4..i * 4 + 4];
        let pb = &b[i * 4..i * 4 + 4];
        assert_eq!(pa[3], 255, "megakernel 像素 {i} 的 alpha 必须为 255");
        assert_eq!(pb[3], 255, "wavefront 像素 {i} 的 alpha 必须为 255");
        let mut pix_diff = false;
        for c in 0..3 {
            let d = pa[c].abs_diff(pb[c]);
            sum_abs += d as u64;
            pix_diff |= d != 0;
        }
        if pix_diff {
            differing += 1;
            if first.is_none() {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                first = Some((
                    x,
                    y,
                    [pa[0], pa[1], pa[2], pa[3]],
                    [pb[0], pb[1], pb[2], pb[3]],
                ));
            }
        }
    }
    Diff {
        differing,
        sum_abs,
        first,
    }
}

/// aesthetic 全关（`AestheticReq` 没有 `Default`：三个开关的"关"必须显式写）。
fn aesthetic_off() -> AestheticReq {
    AestheticReq {
        teal_orange: false,
        contrast: 1.0,
        filmic: false,
    }
}

/// W1 的主场景：4 球 2 灯（与 `qa_render_scene.rs` 同构图口径，覆盖
/// 球内/球外/球间遮挡/多光源叠加四类像素）。
fn sphere_scene(w: u32, h: u32, aesthetic: AestheticReq) -> SceneRenderRequest {
    SceneRenderRequest {
        width: w,
        height: h,
        camera: CameraReq {
            eye: Vec3::new(0.6, 0.9, -5.2),
            target: Vec3::new(0.0, 0.1, 0.0),
            fov: 55.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![
            LightReq {
                direction: Vec3::new(0.3, -0.6, 1.0),
                color: Vec3::new(1.0, 0.95, 0.85),
                intensity: 1.4,
            },
            LightReq {
                direction: Vec3::new(-0.8, 0.4, 0.6),
                color: Vec3::new(0.35, 0.55, 0.95),
                intensity: 0.9,
            },
        ],
        spheres: vec![
            SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                albedo: Vec3::new(0.85, 0.25, 0.2),
                roughness: 0.35,
                metalness: 0.1,
                maps: prism_gpu::NO_MAPS,
            },
            SphereReq {
                center: Vec3::new(1.75, -0.25, 1.2),
                radius: 0.75,
                albedo: Vec3::new(0.2, 0.75, 0.4),
                roughness: 0.6,
                metalness: 0.0,
                maps: prism_gpu::NO_MAPS,
            },
            SphereReq {
                center: Vec3::new(-1.6, 0.35, 1.8),
                radius: 0.6,
                albedo: Vec3::new(0.25, 0.35, 0.9),
                roughness: 0.2,
                metalness: 0.7,
                maps: prism_gpu::NO_MAPS,
            },
            // 远球：验证「最近命中」而不是「首个命中」。
            SphereReq {
                center: Vec3::new(0.25, 0.15, 6.5),
                radius: 3.0,
                albedo: Vec3::new(0.55, 0.55, 0.5),
                roughness: 0.9,
                metalness: 0.0,
                maps: prism_gpu::NO_MAPS,
            },
        ],
        aesthetic,
        curves: Vec::new(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 静态门（无设备也执行）
// ---------------------------------------------------------------------------

/// 5 份入库 `.spv` 都在、都非空、都以 SPIR-V magic 开头。
#[test]
fn five_committed_spv_are_present_and_well_formed() {
    assert_eq!(STAGES.len(), 5, "wavefront 恒为 5 段");
    let entries: Vec<&str> = STAGES.iter().map(|(e, _)| *e).collect();
    assert_eq!(
        entries,
        ["generate", "shade", "connect", "extend", "accumulate"],
        "段序必须与 CPU 版 `prism-rt::wavefront` 的派发循环一致"
    );
    for (entry, rel) in STAGES {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("{entry} 的产物 {rel} 读不到：{e}"));
        assert!(
            bytes.len() >= 20,
            "{entry} 的产物只有 {} 字节（连 SPIR-V 头都不够）",
            bytes.len()
        );
        assert_eq!(bytes.len() % 4, 0, "{entry} 的产物长度必须是 4 的倍数");
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, 0x0723_0203, "{entry} 不是 SPIR-V 模块（magic 不符）");
        println!("[artefact] {entry:<10} {rel}  {} B", bytes.len());
    }
}

/// 派发计划的段数：`generate + 3×bounces + accumulate`。
#[test]
fn dispatch_plan_matches_cpu_wavefront_shape() {
    for bounces in 1u32..=8 {
        let stages = 1 + 3 * (bounces as usize) + 1;
        assert_eq!(stages, 2 + 3 * bounces as usize);
    }
    // 兼容模式固定 1 弹射 ⇒ 5 段（`compat_from_scene` 的契约）。
    let req = sphere_scene(16, 16, aesthetic_off());
    let wreq = WavefrontRequest::compat_from_scene(&req);
    assert_eq!(wreq.max_bounces, 1);
    assert_eq!(wreq.spp, 1, "兼容模式每像素 1 样本（与 megakernel 同口径）");
    assert_eq!(wreq.mode, WavefrontMode::CpuRefCompat);
    assert_eq!(PATH_SLOT_BYTES, 96, "PathSlot 的 std430 步长");
}

// ---------------------------------------------------------------------------
// W1：逐位等价（真机）
// ---------------------------------------------------------------------------

/// **W1 主判据**：wavefront vs megakernel，软件 ICD 上 `differing_pixels == 0`。
///
/// 三组场景各跑一次：aesthetic 全关 / 全开 / 非 8 倍数宽度（尾组守卫）。
#[test]
fn w1_wavefront_matches_megakernel_bit_for_bit() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let software = is_software_icd(backend.device_name().unwrap_or(""));

    let cases: [(&str, u32, u32, AestheticReq); 3] = [
        ("aesthetic-off", 64, 48, aesthetic_off()),
        (
            "aesthetic-on",
            64,
            48,
            AestheticReq {
                teal_orange: true,
                contrast: 1.25,
                filmic: true,
            },
        ),
        ("tail-group", 113, 47, aesthetic_off()),
    ];

    for (name, w, h, aesthetic) in cases {
        let scene = sphere_scene(w, h, aesthetic);
        let mega = backend.render_scene_frame(&scene);
        assert_eq!(
            mega.status,
            DispatchStatus::Ok,
            "[{name}] megakernel 未出图：{}",
            mega.detail
        );
        let wave = backend.render_wavefront_frame(&WavefrontRequest::compat_from_scene(&scene));
        assert_eq!(
            wave.status,
            DispatchStatus::Ok,
            "[{name}] wavefront 未出图：{}",
            wave.detail
        );
        assert_eq!(
            wave.dispatches,
            5,
            "[{name}] 兼容模式的段数必须是 5（generate + shade/connect/extend + accumulate）"
        );

        let d = diff_rgba(w, h, &mega.pixels, &wave.pixels);
        println!(
            "[W1 {name}] {w}×{h} differing={}/{} Σ|Δ|={} first={:?}",
            d.differing,
            w * h,
            d.sum_abs,
            d.first
        );
        if software {
            assert_eq!(
                d.differing, 0,
                "[{name}] 软件 ICD 上 W1 要求逐位一致，实测 {} 个像素不同：{:?}",
                d.differing, d.first
            );
        } else {
            // 硬件 GPU 的除法/开方允许 ≤2.5 ULP：只承诺视觉等价（C15 口径）。
            assert!(
                d.differing * 10_000 / ((w * h) as usize).max(1) <= 50,
                "[{name}] 差异像素占比超 0.5%：{} / {}",
                d.differing,
                w * h
            );
        }
    }
}

/// **W1 定位件**：把任一像素的差异定位到**阶段**（`generate` / `shade`）。
///
/// 手法：`stage_stop` 把派发停在指定段，读回该像素的槽位，与 Rust 侧按
/// **同一表达式树**重算的值逐位比对。
///
/// * 停在第 1 段（`generate`）→ 比射线（origin/dir）。不等 ⇒ 差异是生成段引入。
/// * 停在第 2 段（`shade`）→ 把 `radiance` 按 `clamp01*255+0.5 floor` 量化，
///   与 megakernel 的同像素字节比。不等 ⇒ 差异是着色段引入。
///
/// 本用例对**每一个**像素跑这套定位（不是抽样）：全图像素量 64×48 足够小。
#[test]
fn w1_localises_any_difference_to_a_stage() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h) = (64u32, 48u32);
    let scene = sphere_scene(w, h, aesthetic_off());
    let mega = backend.render_scene_frame(&scene);
    assert_eq!(mega.status, DispatchStatus::Ok, "{}", mega.detail);

    let base = WavefrontRequest::compat_from_scene(&scene);

    // ---- 第 1 段：generate → 比射线 ----
    let mut gen_only = base.clone();
    gen_only.stage_stop = Some(1);
    gen_only.readback_slots = true;
    let gen = backend.render_wavefront_frame(&gen_only);
    assert_eq!(gen.status, DispatchStatus::Ok, "{}", gen.detail);
    assert_eq!(gen.slot_bytes.len(), (w as usize) * (h as usize) * PATH_SLOT_BYTES);

    // ---- 第 2 段：generate + shade → 比量化后的颜色 ----
    let mut up_to_shade = base.clone();
    up_to_shade.stage_stop = Some(2);
    up_to_shade.readback_slots = true;
    let sh = backend.render_wavefront_frame(&up_to_shade);
    assert_eq!(sh.status, DispatchStatus::Ok, "{}", sh.detail);

    let basis = camera_basis(&scene.camera);
    let mut bad_gen = 0usize;
    let mut bad_shade = 0usize;
    let mut first_gen: Option<(u32, u32)> = None;
    let mut first_shade: Option<(u32, u32)> = None;

    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) as usize;

            // 期望射线：与 `rt_wavefront.wgsl::generate` 的兼容分支同表达式树。
            let nx = (2.0 * (px as f32 + 0.5) / (w as f32) - 1.0)
                * basis.tan_half_fov
                * basis.aspect;
            let ny = (1.0 - 2.0 * (py as f32 + 0.5) / (h as f32)) * basis.tan_half_fov;
            let d = Vec3::new(
                basis.forward[0] + basis.right[0] * nx + basis.up[0] * ny,
                basis.forward[1] + basis.right[1] * nx + basis.up[1] * ny,
                basis.forward[2] + basis.right[2] * nx + basis.up[2] * ny,
            )
            .normalize();

            let slot = &gen.slot_bytes[idx * PATH_SLOT_BYTES..(idx + 1) * PATH_SLOT_BYTES];
            let o = read_vec3(slot, 0);
            let dv = read_vec3(slot, 16);
            let ray_ok = o == [basis.eye[0], basis.eye[1], basis.eye[2]]
                && dv == [d.x, d.y, d.z];
            if !ray_ok {
                bad_gen += 1;
                first_gen.get_or_insert((px, py));
            }

            // 期望颜色：shade 写出的 radiance，按 megakernel 的量化式取字节。
            let sslot = &sh.slot_bytes[idx * PATH_SLOT_BYTES..(idx + 1) * PATH_SLOT_BYTES];
            let rad = read_vec3(sslot, 48);
            let q = [
                quantize(rad[0]),
                quantize(rad[1]),
                quantize(rad[2]),
                255u8,
            ];
            let got = &mega.pixels[idx * 4..idx * 4 + 4];
            if q != [got[0], got[1], got[2], got[3]] {
                bad_shade += 1;
                first_shade.get_or_insert((px, py));
            }
        }
    }

    println!(
        "[W1-localise] 64×48：generate 段不符 {bad_gen} 像素{:?}；shade 段不符 {bad_shade} 像素{:?}",
        first_gen, first_shade
    );
    assert_eq!(bad_gen, 0, "generate 段与 Rust 重算的射线不逐位一致");
    assert_eq!(
        bad_shade, 0,
        "shade 段量化结果与 megakernel 像素不逐位一致"
    );
}

/// 读槽位里 `off` 起的三个 f32。
fn read_vec3(slot: &[u8], off: usize) -> [f32; 3] {
    let rd = |o: usize| f32::from_le_bytes([slot[o], slot[o + 1], slot[o + 2], slot[o + 3]]);
    [rd(off), rd(off + 4), rd(off + 8)]
}

/// `clamp01 * 255 + 0.5` 取整（与 `spirv_rt::quantize` / `rt_wavefront::quantize_compat`
/// 同一式：floor(x+0.5)，非负域等价于四舍五入）。
fn quantize(v: f32) -> u8 {
    let t = v.clamp(0.0, 1.0) * 255.0 + 0.5;
    t.floor() as u8
}

// ---------------------------------------------------------------------------
// W2：确定性
// ---------------------------------------------------------------------------

/// **W2**：同请求连跑两次，`pixels` / `hdr` / `dispatches` 逐字节相同。
#[test]
fn w2_wavefront_is_bitwise_deterministic() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let scene = sphere_scene(32, 24, aesthetic_off());

    // 兼容模式
    let req = WavefrontRequest::compat_from_scene(&scene);
    let a = backend.render_wavefront_frame(&req);
    let b = backend.render_wavefront_frame(&req);
    assert_eq!(a.status, DispatchStatus::Ok, "{}", a.detail);
    assert_eq!(b.status, DispatchStatus::Ok, "{}", b.detail);
    assert_eq!(a.pixels, b.pixels, "W2：兼容模式两次像素不一致");
    assert_eq!(a.dispatches, b.dispatches);

    // 路径模式（HDR）
    let preq = path_scene(32, 24, 4);
    let c = backend.render_wavefront_frame(&preq);
    let d = backend.render_wavefront_frame(&preq);
    assert_eq!(c.status, DispatchStatus::Ok, "{}", c.detail);
    assert_eq!(d.status, DispatchStatus::Ok, "{}", d.detail);
    assert_eq!(c.hdr.len(), d.hdr.len());
    for (i, (x, y)) in c.hdr.iter().zip(d.hdr.iter()).enumerate() {
        assert_eq!(
            x.map(f32::to_bits),
            y.map(f32::to_bits),
            "W2：路径模式第 {i} 像素两次不一致（{:?} vs {:?}）",
            x,
            y
        );
    }
    println!(
        "[W2] 兼容模式 {} 段 / 路径模式 {} 段，两次逐字节一致",
        a.dispatches, c.dispatches
    );
}

// ---------------------------------------------------------------------------
// W3：4 vs 8 弹射（设备实测）
// ---------------------------------------------------------------------------

/// 路径模式的三角形场景：地面 2 三角 + 后墙 2 三角 + 一盏球面灯。
///
/// 为什么是三角形：路径模式的 `scene_intersect` 只吃 `tris`（球表只服务兼容
/// 模式）。地面/后墙给多次弹射留出真实的间接光路面。
fn path_scene(w: u32, h: u32, max_bounces: u32) -> WavefrontRequest {
    let floor_mat = WavefrontMaterial {
        base_color: Vec3::new(0.75, 0.72, 0.68),
        roughness: 0.55,
        f0: Vec3::splat(0.04),
        metalness: 0.0,
        transmission: 0.0,
        ior: 1.5,
        emissive: Vec3::splat(0.0),
        kind: 0, // Lambert（T-B：路径模式按 kind 选逐位对拍分支）
    };
    let wall_mat = WavefrontMaterial {
        base_color: Vec3::new(0.35, 0.42, 0.62),
        roughness: 0.7,
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
        spp: 1,
        max_bounces,
        seed: 0x5EED_1234,
        mode: WavefrontMode::Path,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.35, -3.2),
            target: Vec3::new(0.0, 0.0, 0.4),
            fov: 50.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        aesthetic: aesthetic_off(),
        spheres: Vec::new(),
        triangles: vec![
            // 地面（y = -1，**法线朝上**：绕序使 cross(b-a, c-a) = +y，
            // 否则 NEE 的 cos_l < 0 会把直接光整段跳过——那是场景错了，不是内核错了）
            tri(
                Vec3::new(-2.0, -1.0, -2.0),
                Vec3::new(2.0, -1.0, 3.0),
                Vec3::new(2.0, -1.0, -2.0),
            ),
            tri(
                Vec3::new(-2.0, -1.0, -2.0),
                Vec3::new(-2.0, -1.0, 3.0),
                Vec3::new(2.0, -1.0, 3.0),
            ),
            // 后墙（z = 3，**法线朝 -z**：朝向房间内侧）
            tri(
                Vec3::new(-2.0, -1.0, 3.0),
                Vec3::new(2.0, 2.0, 3.0),
                Vec3::new(2.0, -1.0, 3.0),
            ),
            tri(
                Vec3::new(-2.0, -1.0, 3.0),
                Vec3::new(-2.0, 2.0, 3.0),
                Vec3::new(2.0, 2.0, 3.0),
            ),
        ],
        materials: vec![floor_mat, wall_mat],
        prim_mat: vec![0, 0, 1, 1],
        lights: vec![WavefrontLight {
            position: Vec3::new(0.0, 1.1, 0.2),
            radius: 0.35,
            color: Vec3::new(9.0, 8.6, 7.8),
        }],
        ambient: Vec3::splat(0.0),
        env_blob: Vec::new(),
        sobol_table: None,
        node_id: 0,
        frame: 0,
        camera_uniform: None,
        stage_stop: None,
        readback_slots: false,
    }
}

/// 两张 HDR 图的 `(PSNR dB, 相对 RMSE)`；以 `a` 的均值光强为参照。
fn psnr(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len().max(1);
    let mut sse = 0.0f64;
    let mut ref_energy = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        for c in 0..3 {
            let d = (x[c] as f64) - (y[c] as f64);
            sse += d * d;
            ref_energy += (x[c] as f64) * (x[c] as f64);
        }
    }
    let mse = sse / (n * 3) as f64;
    let rms_ref = (ref_energy / (n * 3) as f64).sqrt();
    let rel = if rms_ref > 0.0 {
        mse.sqrt() / rms_ref
    } else {
        0.0
    };
    // HDR 已 clamp01 到 [0,1]（读回时只做预览量化，hdr 原样），峰值取 1.0。
    let peak = 1.0f64;
    let p = if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    };
    (p, rel, rms_ref)
}

/// **W3**：设备上实测 4 vs 8 弹射的 PSNR / 相对 RMSE。
///
/// 判据（有牙）：8 弹射相对 4 弹射必须是**小幅修正**——PSNR ≥ 20 dB 且相对
/// RMSE ≤ 10%，同时两图都必须有非零能量（不是"两张黑图相等"式的假通过）。
#[test]
fn w3_bounce_depth_measured_on_device() {
    let backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let (w, h) = (64u32, 48u32);
    let r4 = backend.render_wavefront_frame(&path_scene(w, h, 4));
    let r8 = backend.render_wavefront_frame(&path_scene(w, h, 8));
    for (tag, r) in [("4b", &r4), ("8b", &r8)] {
        assert_eq!(r.status, DispatchStatus::Ok, "[{tag}] {}", r.detail);
    }
    assert_eq!(r4.hdr.len(), (w as usize) * (h as usize));
    assert_eq!(r8.hdr.len(), r4.hdr.len());

    let (p, rel, rms) = psnr(&r4.hdr, &r8.hdr);
    // 非零能量：否则 PSNR 的两张黑图会给出 ∞，判据就失去意义。
    assert!(rms > 1e-4, "4 弹射图没有能量（rms={rms}）：场景/内核无效");
    let lit = r4
        .hdr
        .iter()
        .filter(|c| c[0] + c[1] + c[2] > 1e-3)
        .count();
    println!(
        "[W3] {w}×{h} max_bounces 4 vs 8：PSNR={p:.2} dB  相对RMSE={:.4}  参照RMS={rms:.5}  \
         有光像素={lit}/{}  段数 4b={} 8b={}",
        rel,
        w * h,
        r4.dispatches,
        r8.dispatches
    );
    assert_eq!(r4.dispatches, 1 + 3 * 4 + 1, "4 弹射的段数");
    assert_eq!(r8.dispatches, 1 + 3 * 8 + 1, "8 弹射的段数");
    assert!(
        lit * 100 / ((w * h) as usize) >= 5,
        "有光像素过少（{lit}/{}）：场景构图失效",
        w * h
    );
    assert!(p >= 20.0, "8 弹射相对 4 弹射的修正过大（PSNR={p:.2} dB）");
    assert!(rel <= 0.10, "相对 RMSE 超 10%（{rel:.4}）");
}

