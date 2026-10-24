//! **W4 收口测试**：GPU 出第一张与 CPU 参考一致的帧缓冲。
//!
//! 闭环链（真机执行，非模拟）：
//!
//! ```text
//! VkInstance → lavapipe（VK_PHYSICAL_DEVICE_TYPE_CPU）→ VkDevice + queue
//!   → SPIR-V（自研发射器 build_background_module）
//!   → VkComputePipeline + 描述符（b0 参数 / b1 像素）
//!   → vkCmdDispatch(ceil(w/8), ceil(h/8), 1) → fence → vkMapMemory 读回
//!   → 与 render_background_cpu 全像素逐字节比对
//! ```
//!
//! # 判等口径（§6 命题 P1）
//!
//! 背景算子是逐点算子（支撑半径 `r = 0`）：`halo ≥ 0` 恒成立，故 GPU 与 CPU
//! 必须**逐字节**相同。本测试断言 `mismatches == 0`，并同时打印
//! 「若有差」的分布（差异字节数 / 差异像素数 / 最大通道差），使失败可诊断。
//!
//! # 环境自门控
//!
//! 无 Vulkan 设备时打印探测报告后 skip（不 fail、不假装跑过）。本沙箱装有
//! lavapipe，因此这些用例**真实执行**，设备名与图像尺寸都会打进输出。

use prism_gpu::background::{render_background, render_background_cpu, render_background_gpu};
use prism_gpu::vulkan::background::{background_kernel, BackgroundFrameReport};
use prism_gpu::vulkan::spirv::{module, validate, SpvExpect};
use prism_gpu::vulkan::spirv_bg::{ENTRY_BACKGROUND, MODULE_BACKGROUND, WORKGROUP_BACKGROUND};
use prism_gpu::vulkan::{compute::DispatchStatus, VulkanBackend};

/// 无设备时打印探测报告并返回 `false`（调用方据此 skip，而不是 fail）。
fn device_ready(backend: &VulkanBackend) -> bool {
    let report = VulkanBackend::probe();
    if !report.device_created {
        println!(
            "[skip] 本机无可用 Vulkan 设备：supported_platform={} loader_loaded={} \
             instance_created={} device_count={} reason={:?}",
            report.supported_platform,
            report.loader_loaded,
            report.instance_created,
            report.device_count,
            report.reason
        );
        return false;
    }
    assert!(backend.probed_device(), "probe() 报设备已建，backend 却无上下文");
    println!(
        "[device] {}（枚举 {} 个物理设备）",
        backend.device_name().unwrap_or("<unnamed>"),
        report.device_count
    );
    true
}

/// 两帧的逐字节差异统计（差异字节数 / 差异像素数 / 最大通道差 / 首个差异位置）。
struct Diff {
    differing_bytes: usize,
    differing_pixels: usize,
    max_channel_delta: u8,
    first: Option<(usize, u8, u8)>,
}

fn diff_bytes(gpu: &[u8], cpu: &[u8]) -> Diff {
    let mut d = Diff {
        differing_bytes: 0,
        differing_pixels: 0,
        max_channel_delta: 0,
        first: None,
    };
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if g != c {
            d.differing_bytes += 1;
            let delta = g.max(c) - g.min(c);
            d.max_channel_delta = d.max_channel_delta.max(delta);
            if d.first.is_none() {
                d.first = Some((i, g, c));
            }
        }
    }
    // 以「像素」为单位再数一遍（4 字节一组；不足一组时按剩余字节数算）。
    for px in 0..cpu.len() / 4 {
        let base = px * 4;
        if gpu[base..base + 4] != cpu[base..base + 4] {
            d.differing_pixels += 1;
        }
    }
    d
}

/// 主判对：GPU 背景帧 vs CPU 参考，全像素逐字节。
#[test]
fn gpu_background_frame_matches_cpu_reference_bytewise() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    assert!(
        backend.frame_available(),
        "有设备但背景帧路径自检失败 —— W4 收口不成立"
    );

    // 64×48：两轴都整除工作组（8×8）——纯快路径；
    // 100×70：两轴都带尾组（100 = 12×8+4，70 = 8×8+6）——守卫生效；
    // 32×24 / t=-3.0：噪声格点坐标为负——锁住「负值 floor → u32」的饱和口径
    // （Rust `as u32` 饱和到 0，shader 侧先 `FMax(x, 0)` 再 `OpConvertFToU`）。
    for (w, h, t) in [
        (64u32, 48u32, 1.25f32),
        (100, 70, 1.25),
        (32, 24, -3.0),
    ] {
        let seed = 0x9E37_79B1u64;
        let report = backend.render_background_frame(w, h, t, seed);
        assert_eq!(
            report.status,
            DispatchStatus::Ok,
            "GPU 出帧失败：{}（{}）",
            report.status.as_str(),
            report.detail
        );
        assert_eq!(report.pixels.len(), (w * h * 4) as usize, "帧字节数应为 w*h*4");
        assert!(!report.device.is_empty(), "成功报告必须带设备名");

        let cpu = render_background_cpu(w, h, t, seed);
        assert_eq!(cpu.len(), report.pixels.len());

        let d = diff_bytes(&report.pixels, &cpu);
        println!(
            "[frame] device={} size={}x{} t={} groups={:?} bytes={} differing_bytes={} \
             differing_pixels={} max_channel_delta={}",
            report.device,
            w,
            h,
            t,
            report.groups,
            report.pixels.len(),
            d.differing_bytes,
            d.differing_pixels,
            d.max_channel_delta
        );

        assert_eq!(
            d.differing_bytes,
            0,
            "GPU 帧须与 CPU 参考逐字节一致（P1：逐点算子 r=0）；\
             差异像素 {}/{}，最大通道差 {}，首个差异 {:?}",
            d.differing_pixels,
            w * h,
            d.max_channel_delta,
            d.first
        );

        // 图像本身是「像样的图」：不透明、非全黑、上下有明显渐变。
        assert!(report.pixels.chunks_exact(4).all(|p| p[3] == 255), "alpha 必须恒 255");
        assert!(report.pixels.iter().any(|&b| b != 0), "不应全黑");
        let row_luma = |y: u32| -> f32 {
            let mut sum = 0.0f32;
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                sum += 0.299 * report.pixels[i] as f32
                    + 0.587 * report.pixels[i + 1] as f32
                    + 0.114 * report.pixels[i + 2] as f32;
            }
            sum / w as f32
        };
        assert!(
            (row_luma(0) - row_luma(h - 1)).abs() > 1.0,
            "垂直渐变应可测（顶 {:.3} / 底 {:.3}）",
            row_luma(0),
            row_luma(h - 1)
        );
    }
}

/// 确定性：同参数重复出帧逐字节一致（不是「偶尔对一次」）。
#[test]
fn gpu_background_frame_is_deterministic_across_runs() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    let a = backend.render_background_frame(37, 23, 2.5, 0xDEAD_BEEF);
    let b = backend.render_background_frame(37, 23, 2.5, 0xDEAD_BEEF);
    assert_eq!(a.status, DispatchStatus::Ok, "{}", a.detail);
    assert_eq!(b.status, DispatchStatus::Ok, "{}", b.detail);
    assert_eq!(a.pixels, b.pixels, "两遍出帧必须逐字节一致");
    println!(
        "[determinism] device={} 37x23 两遍逐字节一致（{} 字节）",
        a.device,
        a.pixels.len()
    );
}

/// 引擎入口：`render_background` 与 CPU 参考逐字节一致（GPU 或回退都成立），
/// 且**真的走了 GPU**（有设备时 `render_background_gpu` 必须返回 `Some`）。
#[test]
fn engine_entry_is_byte_identical_and_uses_gpu_when_available() {
    let cpu = render_background_cpu(48, 24, 3.7, 0x1234_5678);
    let engine = render_background(48, 24, 3.7, 0x1234_5678);
    assert_eq!(engine.len(), cpu.len(), "入口长度须为 w*h*4");
    assert_eq!(engine, cpu, "引擎入口必须与 CPU 参考逐字节一致");

    let gpu = render_background_gpu(48, 24, 3.7, 0x1234_5678);
    if VulkanBackend::probe().device_created {
        let gpu = gpu.expect("有设备时入口必须走 GPU（不得静默退回 CPU）");
        assert_eq!(gpu, cpu, "GPU 出帧须与 CPU 参考一致");
        println!("[entry] 有设备：render_background 走 GPU，48x24 逐字节一致");
    } else {
        assert!(gpu.is_none(), "无设备时 GPU 入口必须如实返回 None");
        println!("[entry] 无设备：render_background 优雅回退 CPU");
    }
}

/// 尺寸守卫与降级：零轴 / 超限尺寸不 panic、不返回假帧。
#[test]
fn background_gpu_refuses_extreme_sizes_without_panicking() {
    let mut backend = VulkanBackend::new();
    for (w, h) in [
        (0u32, 0u32),
        (0, 16),
        (16, 0),
        (u32::MAX, u32::MAX),
        (65_536, 65_536),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.render_background_frame(w, h, 0.0, 0)
        }));
        let report = r.expect("背景出帧不得 panic（C18）");
        assert_eq!(
            report.status,
            DispatchStatus::BufferFailed,
            "({w},{h}) 应被尺寸守卫拒绝，实际 {}（{}）",
            report.status.as_str(),
            report.detail
        );
        assert!(report.pixels.is_empty(), "被拒时不得返回像素");
        // 引擎入口同样不 panic，并回退到 CPU 参考（空 vec）。
        let engine = render_background(w, h, 0.0, 0);
        assert!(engine.is_empty(), "超限尺寸的入口输出应为空");
    }
    println!("[guard] 极端尺寸全部被拒且不 panic");
}

/// 非有限时间参数：GPU 侧是 UB（`OpConvertFToU` 越界），必须**回退 CPU**，
/// 且任何路径都不 panic。
#[test]
fn non_finite_time_falls_back_to_cpu_without_panicking() {
    for t in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0e12] {
        let out = std::panic::catch_unwind(|| render_background(16, 12, t, 7))
            .expect("非有限 t 不得 panic");
        let cpu = render_background_cpu(16, 12, t, 7);
        assert_eq!(out, cpu, "t={t} 必须回退 CPU 参考（GPU 侧未定义）");

        let mut backend = VulkanBackend::new();
        let r = backend.render_background_frame(16, 12, t, 7);
        assert_eq!(
            r.status,
            DispatchStatus::InvalidArguments,
            "t={t} 应由契约守卫拒绝，实际 {}",
            r.status.as_str()
        );
        assert!(r.pixels.is_empty());
    }
    println!("[contract] 非有限/超界 t 在 FFI 之前被拒并回退 CPU");
}

/// SPIR-V 结构门禁（无设备也执行）：模块结构合法 + 与 kernel 描述一致。
#[test]
fn background_spirv_matches_kernel_description() {
    let words = module(MODULE_BACKGROUND).expect("背景 SPIR-V 模块必须已注册");
    let kernel = background_kernel(80, 60).expect("80×60 合法");
    let summary = validate(
        words,
        &SpvExpect {
            entry: kernel.entry,
            local_size: [
                kernel.workgroup.x,
                kernel.workgroup.y,
                kernel.workgroup.z,
            ],
            bindings: &[0, 1],
        },
    )
    .expect("背景 SPIR-V 须通过结构验证");
    assert_eq!(summary.entry, ENTRY_BACKGROUND);
    assert_eq!(summary.local_size, WORKGROUP_BACKGROUND);
    assert_eq!(summary.bindings, vec![0, 1]);
    println!(
        "[spirv] {} 字 / {} 条指令 / entry={} localsize={:?}",
        words.len(),
        summary.instruction_count,
        summary.entry,
        summary.local_size
    );
}

/// 报告类型契约（无设备也可断言）：失败态的字段语义稳定。
#[test]
fn failed_report_shape_is_honest() {
    let mut backend = VulkanBackend::new();
    let r: BackgroundFrameReport = backend.render_background_frame(0, 0, 0.0, 0);
    assert!(!r.is_ok());
    assert_eq!((r.width, r.height), (0, 0), "尺寸原样回显");
    assert!(r.pixels.is_empty(), "失败不得带像素（不返回假帧）");
    assert!(!r.detail.is_empty(), "失败必须说明原因");
}
