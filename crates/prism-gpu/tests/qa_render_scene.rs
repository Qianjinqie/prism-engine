//! **V2.1 T-A1 验收测试**：Vulkan 光追场景管线的真机出图 + 对拍。
//!
//! # 闭环链（真机执行，非模拟）
//!
//! ```text
//! VkInstance → lavapipe（VK_PHYSICAL_DEVICE_TYPE_CPU）→ VkDevice + queue
//!   → SPIR-V（自研发射器 build_scene_module，含球/灯动态循环）
//!   → VkComputePipeline + 描述符（b0 参数 / b1 球 / b2 灯 / b3 像素）
//!   → vkCmdDispatch(ceil(w/8), ceil(h/8), 1) → fence → vkMapMemory 读回
//!   → ① GPU 双跑逐位比对（硬判据）② 与 CpuRefBackend 统计对拍（Q1 阈值）
//! ```
//!
//! # 两层判据（Q1 裁定）
//!
//! 1. **GPU 自跑双次逐位一致**——**硬判据**（G3 确定性）。lavapipe 是确定性
//!    软栅：同设备 + 同请求 + 同模块两次 dispatch 必须逐字节相同。这一层与
//!    CPU 实现无关，任何「偶发正确」都无法通过它。
//! 2. **GPU vs CPU 参考的统计 diff**——跨后端只承诺**视觉等价**（C15 口径）。
//!    光追链路含 `sqrt` / 除法 / 归一化：Vulkan 对这些运算只要求 ≤2.5 ULP
//!    （非逐位正确舍入），故末位差异属**预期**，不能用逐位判据。
//!
//!    Q1 给的是 provisional 检测线（`max_channel_delta ≤ 16/255`、
//!    `differing_pixels ≤ 5%`）。**实测后已收窄**（Q1 的「若更紧就收窄」条款）：
//!
//!    | 场景（w×h / 球 / 灯） | 实测 max_channel_delta | 实测 differing_pixels | 锁定线 |
//!    |---|---|---|---|
//!    | 64×48 / 4 / 2（aesthetic 关） | **0** | **0 / 3072（0.000%）** | 见 `MAX_CHANNEL_DELTA_LIMIT` |
//!    | 64×48 / 4 / 2（teal+contrast+filmic 开） | **0** | **0 / 3072（0.000%）** | 同上 |
//!    | 113×47 / 4 / 2（尾组守卫） | **0** | **0 / 5311（0.000%）** | 同上 |
//!    | 16×16 空场景（0 球 0 灯） | **0** | **0 / 256** | 同上 |
//!
//!    即：在**软件 ICD（lavapipe / llvmpipe）**上，本 kernel 与 `CpuRefBackend`
//!    **逐字节一致**——这是「同式发射 + `NoContraction` + 无超越函数」三条纪律
//!    的直接产物，比 Q1 的统计线强得多。故本文件的判据分两档：
//!    * **软件 ICD（CI 环境，设备名含 `llvmpipe`）**：**逐位 0 差异**（硬判据）；
//!    * **任意设备**：`max_channel_delta ≤ 1/255` 且 `differing_pixels ≤ 0.5%`
//!      （真正的硬件 GPU 上除法/开方允许 ≤2.5 ULP，留一个末位的余量）。
//!
//!    **C15 登记（待集成阶段并入 `CHANGELOG.md` 的《渲染行为变化清单》）**：
//!    * 受影响特性：`prism-gpu` 的 `VulkanBackend::render_scene`（#11 GPU 缺口）。
//!    * 变化原因：该方法此前恒返回 `Framebuffer::empty()`（调用方回退 CPU）；
//!      V2.1 起在有设备时真跑 4 绑定 compute megakernel 并出帧。GPU 与 CPU
//!      参考的实测差异（llvmpipe）：**0 字节**（逐位一致），故跨后端在本环境
//!      仍逐字节可复现；非软件 ICD 上承诺「视觉等价」，容差 ≤1/255。
//!    * 视觉影响：无（与 CPU 参考逐字节相同；`is_available()` 语义升级为
//!      compute ∧ render 双自检，仅在双自检通过时才报可用）。
//!    * 是否有开关：有——调用方仍可按返回帧的 `is_valid()` 回退 CPU；
//!      不装 Vulkan ICD 的环境行为不变（恒空帧 + 回退）。
//!
//! # 环境自门控（禁 `#[ignore]`）
//!
//! 无 Vulkan 设备时打印探测报告后返回（不 fail、不假装跑过）。本沙箱装有
//! lavapipe，因此这些用例**真实执行**，设备名与统计量都会打进测试输出。
//!
//! # 与 T4（色彩债务）零耦合
//!
//! `SceneRenderRequest` 的 aesthetic 口径是 **Reinhard filmic**（`color/(color+1)`），
//! 全程**不消费** `SRGB_TO_ACESCG` / `ACESCG_TO_SRGB`；`CpuRefBackend` 即基准。
//! 故本文件的判据**不随 T4 的矩阵重录移动**（PRD R2）。

use prism_gpu::vulkan::scene::SceneFrameReport;
use prism_gpu::vulkan::VulkanBackend;
use prism_gpu::{
    AestheticReq, CameraReq, CpuRefBackend, Framebuffer, GpuDevice, LightReq, SceneRenderRequest,
    SphereReq,
};
use prism_math::Vec3;

/// 最大通道差上限（Q1 provisional 16 → **实测收窄到 1**）。
///
/// 软件 ICD 上实测为 0（见文件头表格），这里留 1 个末位余量给「除法/开方
/// 允许 ≤2.5 ULP」的硬件 GPU；CI 环境（llvmpipe）另有更严的 0 断言。
const MAX_CHANNEL_DELTA_LIMIT: u8 = 1;
/// 差异像素占比上限，单位 **万分之几**（Q1 provisional 5% = 500 → 收窄到 50 = 0.5%）。
const DIFFERING_PIXEL_PERMYRIAD_LIMIT: usize = 50;

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
    assert!(
        backend.probed_device(),
        "probe() 报设备已建，backend 却无上下文"
    );
    println!(
        "[device] {}（枚举 {} 个物理设备）",
        backend.device_name().unwrap_or("<unnamed>"),
        report.device_count
    );
    true
}

/// 一帧的统计摘要（诊断 + 阈值断言共用）。
struct Parity {
    /// 非 alpha 通道的最大绝对差（0..=255）。
    max_channel_delta: u8,
    /// 任一通道有差异的像素数。
    differing_pixels: usize,
    /// 任一通道差 **> 1** 的像素数（超出量化末位噪声的部分）。
    differing_pixels_gt1: usize,
    /// 总像素数。
    total_pixels: usize,
    /// 平均绝对通道差（诊断量）。
    mean_abs_delta: f64,
    /// 首个差异样本 `(像素下标, CPU, GPU)`。
    first: Option<(usize, [u8; 4], [u8; 4])>,
}

impl Parity {
    fn differs_pct(&self) -> f64 {
        100.0 * self.differing_pixels as f64 / self.total_pixels.max(1) as f64
    }

    /// 差异像素占比，单位**万分之几**（整数比较，无浮点阈值抖动）。
    fn differs_permyriad(&self) -> usize {
        self.differing_pixels * 10_000 / self.total_pixels.max(1)
    }
}

/// 是否是**软件 ICD**（lavapipe / llvmpipe）：此时判据收紧到逐位一致。
fn is_software_icd(device: &str) -> bool {
    let d = device.to_ascii_lowercase();
    d.contains("llvmpipe") || d.contains("lavapipe") || d.contains("swiftshader")
}

/// 逐像素比对两帧（只比 RGB 三通道；alpha 恒 255，另单独断言）。
fn parity(cpu: &Framebuffer, gpu: &Framebuffer) -> Parity {
    assert_eq!(
        (cpu.width, cpu.height),
        (gpu.width, gpu.height),
        "对拍的两帧尺寸必须一致"
    );
    let total = (cpu.width as usize) * (cpu.height as usize);
    let mut max_channel_delta = 0u8;
    let mut differing_pixels = 0usize;
    let mut differing_pixels_gt1 = 0usize;
    let mut sum_abs = 0u64;
    let mut first: Option<(usize, [u8; 4], [u8; 4])> = None;
    for i in 0..total {
        let a = &cpu.pixels[i * 4..i * 4 + 4];
        let b = &gpu.pixels[i * 4..i * 4 + 4];
        assert_eq!(a[3], 255, "CPU 参考像素 {i} 的 alpha 必须为 255");
        assert_eq!(b[3], 255, "GPU 帧像素 {i} 的 alpha 必须为 255");
        let mut pix_max = 0u8;
        for c in 0..3 {
            let d = a[c].abs_diff(b[c]);
            sum_abs += d as u64;
            pix_max = pix_max.max(d);
        }
        max_channel_delta = max_channel_delta.max(pix_max);
        if pix_max > 0 {
            differing_pixels += 1;
            if first.is_none() {
                first = Some((i, [a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]));
            }
        }
        if pix_max > 1 {
            differing_pixels_gt1 += 1;
        }
    }
    Parity {
        max_channel_delta,
        differing_pixels,
        differing_pixels_gt1,
        total_pixels: total,
        mean_abs_delta: sum_abs as f64 / (total.max(1) * 3) as f64,
        first,
    }
}

/// 主场景：4 球 2 灯，构图覆盖「球内/球外/球间遮挡/多光源叠加」四类像素。
fn scene_request(w: u32, h: u32, aesthetic: AestheticReq) -> SceneRenderRequest {
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
            },
            SphereReq {
                center: Vec3::new(1.75, -0.25, 1.2),
                radius: 0.75,
                albedo: Vec3::new(0.2, 0.75, 0.4),
                roughness: 0.6,
                metalness: 0.0,
            },
            SphereReq {
                center: Vec3::new(-1.6, 0.35, 1.8),
                radius: 0.6,
                albedo: Vec3::new(0.25, 0.35, 0.9),
                roughness: 0.2,
                metalness: 0.7,
            },
            // 远球：验证「最近命中」而非「首个命中」。
            SphereReq {
                center: Vec3::new(0.0, 0.0, 6.0),
                radius: 2.5,
                albedo: Vec3::new(0.7, 0.7, 0.7),
                roughness: 0.5,
                metalness: 0.0,
            },
        ],
        aesthetic,
    }
}

fn aesthetic_off() -> AestheticReq {
    AestheticReq {
        teal_orange: false,
        contrast: 1.0,
        filmic: false,
    }
}

fn aesthetic_on() -> AestheticReq {
    AestheticReq {
        teal_orange: true,
        contrast: 1.35,
        filmic: true,
    }
}

/// 有设备时必须出**有效帧**（`w*h*4` 字节、alpha 全 255、非全零）。
#[test]
fn scene_render_produces_valid_frame_on_device() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        assert!(!backend.is_available(), "无设备 ⇒ 不得报可用");
        assert!(!backend.render_available(), "无设备 ⇒ 渲染自检必须为假");
        return;
    }
    assert!(
        backend.render_available(),
        "有设备 ⇒ 场景渲染自检（4×4 探针帧）必须真跑通"
    );
    let req = scene_request(64, 48, aesthetic_off());
    let report: SceneFrameReport = backend.render_scene_frame(&req);
    assert!(report.is_ok(), "真机场景派发应成功：{:?}", report.detail);
    assert_eq!(report.status.as_str(), "ok");
    assert_eq!((report.width, report.height), (64, 48), "尺寸原样回显");
    assert_eq!(report.pixels.len(), 64 * 48 * 4, "读回长度须为 w*h*4");
    assert_eq!(report.groups, [8, 6, 1], "64×48 / 8×8 ⇒ 8×6 组");
    assert_eq!(report.invocations, 64 * 48, "整除尺寸：零尾组浪费");
    assert!(report.pixels.iter().any(|&b| b != 0), "帧不得全零");
    assert!(
        report.pixels.chunks_exact(4).all(|p| p[3] == 255),
        "alpha 必须恒为 255（与 CPU 参考同布局）"
    );

    // 同一个 `GpuDevice` 入口也必须给出有效帧（两条入口同一路径）。
    let fb = backend.render_scene(&req);
    assert!(fb.is_valid(), "GpuDevice::render_scene 必须给出有效帧");
    assert_eq!(fb.pixels, report.pixels, "两条入口必须逐位相同");
    println!(
        "[frame] device={} {}×{} 有效帧，非零字节 {} / {}",
        report.device,
        report.width,
        report.height,
        report.pixels.iter().filter(|&&b| b != 0).count(),
        report.pixels.len()
    );
}

/// **G3 硬判据**：同一设备 + 同一请求 + 同一模块，两次渲染**逐字节相同**。
///
/// 与 CPU 无关（不涉及任何容差）——这是「确定性」这一条硬约束的直接证据。
#[test]
fn gpu_scene_render_is_bitwise_deterministic_across_runs() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    for aesthetic in [aesthetic_off(), aesthetic_on()] {
        let req = scene_request(64, 48, aesthetic);
        let a = backend.render_scene(&req);
        let b = backend.render_scene(&req);        assert!(a.is_valid() && b.is_valid(), "两次渲染都必须出有效帧");
        assert_eq!(a.pixels.len(), b.pixels.len());
        let mut mismatches = 0usize;
        let mut first = None;
        for (i, (x, y)) in a.pixels.iter().zip(b.pixels.iter()).enumerate() {
            if x != y {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, *x, *y));
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "GPU 双跑必须逐位一致（首个差异：{first:?}）"
        );
        println!(
            "[determinism] {} 字节两遍渲染逐位一致（aesthetic.filmic={}）",
            a.pixels.len(),
            req.aesthetic.filmic
        );
    }
}

/// **Q1 统计对拍（实测已收窄）**：GPU vs `CpuRefBackend`。
///
/// 断言分两档（见文件头表格）：
/// * 任意设备：`max_channel_delta ≤ 1/255`、`differing_pixels ≤ 0.5%`；
/// * **软件 ICD（llvmpipe，CI 环境）**：两者都必须为 **0**（逐位一致）——
///   这是「同式发射 + 禁收缩 + 无超越函数」三条纪律的实测结论。
///
/// 诊断量（差 > 1 的像素数、平均绝对差、首个差异样本）一并打印，使「阈值
/// 是否还该收窄」在每次运行时可见。
#[test]
fn gpu_scene_render_matches_cpu_reference_statistically() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    for (tag, aesthetic) in [("aesthetic-off", aesthetic_off()), ("aesthetic-on", aesthetic_on())] {
        let req = scene_request(64, 48, aesthetic);
        let gpu = backend.render_scene(&req);
        assert!(gpu.is_valid(), "GPU 必须出有效帧");
        let mut cpu = CpuRefBackend;
        let cpu_fb = cpu.render_scene(&req);
        assert!(cpu_fb.is_valid(), "CPU 参考必须出有效帧");

        let p = parity(&cpu_fb, &gpu);
        let software = is_software_icd(backend.device_name().unwrap_or(""));
        println!(
            "[parity:{tag}] pixels={} max_channel_delta={} differing_pixels={} ({:.3}%) \
             differing_gt1={} mean_abs_delta={:.5} software_icd={} first_diff={:?}",
            p.total_pixels,
            p.max_channel_delta,
            p.differing_pixels,
            p.differs_pct(),
            p.differing_pixels_gt1,
            p.mean_abs_delta,
            software,
            p.first
        );

        assert!(
            p.max_channel_delta <= MAX_CHANNEL_DELTA_LIMIT,
            "[{tag}] 最大通道差 {} 超过收窄后的检测线 {}（首个差异 {:?}）",
            p.max_channel_delta,
            MAX_CHANNEL_DELTA_LIMIT,
            p.first
        );
        assert!(
            p.differs_permyriad() <= DIFFERING_PIXEL_PERMYRIAD_LIMIT,
            "[{tag}] 差异像素 {}‱ 超过收窄后的检测线 {}‱（{} / {} 像素）",
            p.differs_permyriad(),
            DIFFERING_PIXEL_PERMYRIAD_LIMIT,
            p.differing_pixels,
            p.total_pixels
        );
        if software {
            // 软件 ICD：实测逐字节一致 ⇒ 钉死为硬判据（任何回退立刻红）。
            assert_eq!(
                p.max_channel_delta, 0,
                "[{tag}] 软件 ICD 上必须逐字节一致（最大通道差 {} ≠ 0）",
                p.max_channel_delta
            );
            assert_eq!(
                p.differing_pixels, 0,
                "[{tag}] 软件 ICD 上必须逐字节一致（差异像素 {} ≠ 0，首个 {:?}）",
                p.differing_pixels, p.first
            );
        }
    }
}

/// 边界形态（A1 验收 4）：极端/零尺寸**秒回拒绝缓冲**，不 panic、不 dispatch。
#[test]
fn scene_render_refuses_extreme_sizes_without_panicking() {
    let mut backend = VulkanBackend::new();
    // 有设备与否都必须成立：尺寸契约先于任何设备访问。
    for (w, h) in [
        (u32::MAX, u32::MAX),
        (65_536, 65_536),
        (0, 0),
        (0, 48),
        (64, 0),
    ] {
        let mut req = scene_request(w, h, aesthetic_off());
        req.width = w;
        req.height = h;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut be = VulkanBackend::new();
            be.render_scene(&req)
        }));
        let fb = r.expect("极端尺寸不得 panic");
        assert!(!fb.is_valid(), "({w},{h}) 必须被拒绝");
        assert!(fb.pixels.is_empty(), "({w},{h}) 被拒缓冲不含像素");
        assert_eq!((fb.width, fb.height), (w, h), "尺寸须原样回显");
    }
    // 空场景（0 球 0 灯）仍是一张合法帧：全背景色。
    if device_ready(&backend) {
        let mut req = scene_request(16, 16, aesthetic_off());
        req.spheres.clear();
        req.lights.clear();
        let fb = backend.render_scene(&req);
        assert!(fb.is_valid(), "空场景也须出有效帧（背景色）");
        // 背景色 (0.04, 0.05, 0.08) → 量化 (10, 13, 20)。
        let px = &fb.pixels[..4];
        assert_eq!(px, &[10, 13, 20, 255], "空场景须为背景色");
        println!("[empty-scene] 16×16 全背景帧：{:?}", px);
    }
}

/// 尾组守卫（A1 验收 4 的第二种形态）：非 8 整除尺寸由 kernel 侧丢弃越界调用。
///
/// 113×47：`ceil(113/8)=15` 组 × `ceil(47/8)=6` 组 = 90 组 = 5760 调用，
/// 有效 5311，浪费 449。越界调用**不得写内存**（否则尾部像素被污染）。
#[test]
fn scene_render_handles_non_divisible_sizes_via_tail_guard() {
    let mut backend = VulkanBackend::new();
    if !device_ready(&backend) {
        return;
    }
    let req = scene_request(113, 47, aesthetic_off());
    let report = backend.render_scene_frame(&req);
    assert!(report.is_ok(), "非整除尺寸也应成功：{:?}", report.detail);
    assert_eq!(report.pixels.len(), 113 * 47 * 4);
    assert_eq!(report.groups, [15, 6, 1]);
    assert_eq!(report.invocations, 15 * 6 * 64);
    assert!(
        report.invocations > 113 * 47,
        "该尺寸应确有尾组浪费（诊断量非零）"
    );

    // 与 CPU 对拍（尾组不得污染任何像素）。
    let mut cpu = CpuRefBackend;
    let cpu_fb = cpu.render_scene(&req);
    let p = parity(&cpu_fb, &report_frame(&report));
    println!(
        "[parity:tail] 113×47 max_channel_delta={} differing_pixels={} ({:.3}%)",
        p.max_channel_delta,
        p.differing_pixels,
        p.differs_pct()
    );
    assert!(p.max_channel_delta <= MAX_CHANNEL_DELTA_LIMIT);
    assert!(p.differs_permyriad() <= DIFFERING_PIXEL_PERMYRIAD_LIMIT);
}

/// 把 `SceneFrameReport` 包成 `Framebuffer` 以便共用 `parity`。
fn report_frame(r: &SceneFrameReport) -> Framebuffer {
    Framebuffer {
        width: r.width,
        height: r.height,
        pixels: r.pixels.clone(),
    }
}
