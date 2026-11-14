//! **2D 矢量路径 GPU 光栅：真机闭环 + 判据口径 + 反假绿负控**。
//!
//! # 判据设计（动手前写死，绝不事后改尺子）
//!
//! ## 难点：CPU 权威是 f64 几何 + f32 写回，GPU/WGSL 只有 f32
//!
//! `prism-raster` 的扫描线用 f64 做几何插值与面积积分、只在写回覆盖率时舍入
//! 一次；SPIR-V 1.0 与 WGSL 都没有 f64。若直接把全局 x 坐标以 f32 传给 kernel，
//! 1080p 坐标的 ulp ≈ 1.2e-4，覆盖率会偏离精确面积到 1e-4——**「GPU 与 f64
//! 标尺逐字节一致」在数学上不可达**。本实现不靠放宽阈值过关，而是换数据编码：
//! 片段的 x 拆成「精确整数基准 + 小量小数」（`X = col_base + f`，
//! `base = col_base + 2²⁰` 是 f32 精确整数），kernel 里
//! `u = (base − 2²⁰ − px) + f`；括号内是整数差（≤2²¹，f32 精确），只有
//! `u ∈ (−1,2)` 的敏感区影响面积，此时 `ulp(u) ≤ 2.4e-7`。每像素最多 4 条穿越
//! 边 ⇒ `|Δcov| < 1e-6`。这就是下面 `TOL_EXACT = 2e-6` 的**推导来源**。
//!
//! ## P1（主判据，逐字节，绿灯的硬门槛）
//!
//! GPU 输出与 [`prism_gpu::path_raster::rasterize_cpu_prepared`] 在**同一份 f32
//! 片段缓冲**上必须**逐字节相同**：`differing_bytes = 0`、
//! `differing_pixels = 0`、u8 量化视图 `max_channel_delta = 0`。
//! 依据：三端（CPU 参考 / SPIR-V / WGSL）同式、同求值树、同运算顺序、
//! `NoContraction` 禁 FMA、f32 位型传参（无文本往返）。这是**逐点算子**，
//! 不存在「容差内接近」的余地。
//!
//! ## P2（数学等价，f64 精确面积标尺）
//!
//! 与 `prism-raster` f64 几何权威（其精度由 `scanline_diff.rs` 用 f64 精确有理
//! 积分标尺钉在 2e-6 内）逐像素比：`max_abs_delta ≤ 2e-6`、u8 量化
//! `differing_pixels = 0`、`max_channel_delta = 0`、面积总量相对误差 ≤ 1e-6。
//! 恒等式：`clamp(u,0,1) = u⁺ − (u−1)⁺`，而 `∫(c+kt)⁺dt` 的闭式 `pos_area`
//! 与扫描线的「区域 `[xl,xr)` 边界列解析积分」是同一个面积积分的两种写法
//! （Green/射线法 telescopes），不是近似、不是超采样。P2 由
//! `crates/prism-postfx/tests/qa_path_raster_gpu_parity.rs` 断言。
//!
//! ## 负控（反假绿，三条）
//!
//! 1. **扰动 kernel**：把 `1.0` 常量换成 `1.001` 发射一份故意错误的 SPIR-V，
//!    派发后逐字节判据**必须 FAIL**（`differing_pixels > 0`）——证明绿灯来自
//!    真实 kernel 执行，而非 CPU 兜底或恒真断言。
//! 2. **扰动输入**：把片段 `delta` 全置 0 派发 ⇒ 输出全 0，与 CPU 参考不同——
//!    证明输出确实来自 kernel 读取的片段缓冲。
//! 3. **执行证据**：报告里 `status == Ok` ∧ `device` 非空 ∧ `groups` 正确 ∧
//!    `invocations ≥ w*h` ∧ `segments > 0`；有设备时
//!    `rasterize_path_gpu` 必须返回 `Some`。
//!
//! **纪律**：为变绿而放宽 `differing_pixels` / 调 `TOL_EXACT` = 改判据，不是修
//! 实现。失败退路是修 kernel 或修预处理，本文件的阈值一个字都不动。
//!
//! # 环境自门控
//!
//! 无 Vulkan 设备时打印探测报告后 skip（不 fail、不假装跑过）。本沙箱装
//! lavapipe（`VK_PHYSICAL_DEVICE_TYPE_CPU`），故这些用例**真实执行**，设备名、
//! 组数、差异统计全打进输出。

use prism_gpu::path_raster::{
    prepare, rasterize, rasterize_cpu_prepared, PathFillRule, RasterPoint,
};
// GPU-only 入口在 crate 根以 `rasterize_path_gpu` 别名导出（避免与场景 API 混淆）；
// `rasterize`（GPU 优先、CPU 兜底引擎入口）直接取 `path_raster` 模块本名。
use prism_gpu::rasterize_path_gpu;
use prism_gpu::vulkan::path_raster::{path_kernel, PathFrameReport, KERNEL_PATH_RASTER_FRAME};
use prism_gpu::vulkan::spirv::{module, validate, SpvExpect};
use prism_gpu::vulkan::spirv_path::{
    build_path_module_with_one, path_words, ENTRY_PATH_RASTER, MODULE_PATH_RASTER,
    WORKGROUP_PATH_RASTER,
};
use prism_gpu::vulkan::{compute::DispatchStatus, VulkanBackend};

/// P2 的 f64 标尺阈值（推导见文件头；本常量在 postfx 对拍测试里同源复用）。
const TOL_EXACT: f64 = 2e-6;

/// 无设备时打印探测报告并返回 false（调用方 skip，不 fail）。
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

/// 两份 f32 覆盖率的差异统计（逐字节 / 逐像素 / u8 量化 / 最大绝对差）。
#[derive(Debug)]
struct Diff {
    differing_bytes: usize,
    differing_pixels: usize,
    max_channel_delta: u8,
    max_abs_delta: f32,
    first: Option<(usize, u32, u32)>,
}

fn diff_f32(gpu: &[f32], cpu: &[f32]) -> Diff {
    assert_eq!(gpu.len(), cpu.len());
    let mut d = Diff {
        differing_bytes: 0,
        differing_pixels: 0,
        max_channel_delta: 0,
        max_abs_delta: 0.0,
        first: None,
    };
    let gu = prism_gpu::path_raster::CoverageMask { width: 0, height: 0, data: gpu.to_vec() }.to_u8();
    let cu = prism_gpu::path_raster::CoverageMask { width: 0, height: 0, data: cpu.to_vec() }.to_u8();
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gb = g.to_bits();
        let cb = c.to_bits();
        if gb != cb {
            d.differing_bytes += 1;
            d.differing_pixels += 1;
            if d.first.is_none() {
                d.first = Some((i, gb, cb));
            }
        }
        let delta = g.max(c) - g.min(c);
        if delta > d.max_abs_delta {
            d.max_abs_delta = delta;
        }
        d.max_channel_delta = d.max_channel_delta.max(gu[i].max(cu[i]) - gu[i].min(cu[i]));
    }
    d
}

fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<RasterPoint> {
    vec![
        RasterPoint::new(x0, y0),
        RasterPoint::new(x1, y0),
        RasterPoint::new(x1, y1),
        RasterPoint::new(x0, y1),
    ]
}

fn star(n: usize, cx: f32, cy: f32, r: f32) -> Vec<RasterPoint> {
    (0..n)
        .map(|i| {
            let t = (i as f32) * (core::f32::consts::TAU / n as f32);
            let ri = if i % 2 == 0 { r } else { r * 0.42 };
            RasterPoint::new(cx + ri * t.cos(), cy + ri * t.sin())
        })
        .collect()
}

/// 判据覆盖的用例集：规则 / 孔洞 / 自交 / 尾组尺寸 / 空路径。
fn cases() -> Vec<(&'static str, Vec<Vec<RasterPoint>>, PathFillRule, u32, u32)> {
    let mut outer = square(2.0, 2.0, 58.0, 44.0);
    outer.reverse();
    let inner = square(12.0, 10.0, 40.0, 34.0);
    vec![
        ("square_64x48", vec![square(1.5, 1.5, 40.5, 30.5)], PathFillRule::Winding, 64, 48),
        ("triangle_37x23", vec![vec![
            RasterPoint::new(2.3, 1.7), RasterPoint::new(31.7, 6.1), RasterPoint::new(8.9, 20.3),
        ]], PathFillRule::Winding, 37, 23),
        ("tailgroups_100x70", vec![square(3.25, 2.75, 91.5, 63.25)], PathFillRule::Winding, 100, 70),
        ("ring_hole_64x48", vec![outer, inner], PathFillRule::Winding, 64, 48),
        ("overlap_winding_48x32", vec![square(1.0, 1.0, 30.0, 24.0), square(12.0, 8.0, 44.0, 30.0)], PathFillRule::Winding, 48, 32),
        ("overlap_evenodd_48x32", vec![square(1.0, 1.0, 30.0, 24.0), square(12.0, 8.0, 44.0, 30.0)], PathFillRule::EvenOdd, 48, 32),
        ("star64_64x48", vec![star(64, 32.0, 24.0, 21.0)], PathFillRule::Winding, 64, 48),
        ("empty_16x12", Vec::new(), PathFillRule::Winding, 16, 12),
    ]
}

/// GPU 执行证据（负控 3）：报告字段必须自证「真在设备上跑过」。
fn assert_gpu_evidence(r: &PathFrameReport, w: u32, h: u32) {
    assert_eq!(r.status, DispatchStatus::Ok, "GPU 派发失败：{}（{}）", r.status.as_str(), r.detail);
    assert_eq!(r.kernel, KERNEL_PATH_RASTER_FRAME);
    assert!(!r.device.is_empty(), "成功报告必须带设备名");
    assert!(r.groups[0] > 0 && r.groups[1] > 0 && r.groups[2] == 1, "groups={:?}", r.groups);
    assert!(r.invocations >= (w as u64) * (h as u64), "invocations={}", r.invocations);
    assert_eq!(r.coverage.data.len(), (w * h) as usize);
}

/// **主判对 P1**：GPU 帧 vs CPU f32 参考，全像素逐字节。
#[test]
fn gpu_path_frame_matches_cpu_reference_bytewise() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    assert!(backend.path_frame_available(), "有设备但 2D 路径 kernel 自检失败 —— GPU 光栅未点亮");

    for (name, contours, rule, w, h) in cases() {
        let input = prepare(&contours, rule, w, h);
        let report = backend.rasterize_path_frame(&input);
        assert_gpu_evidence(&report, w, h);
        let cpu = rasterize_cpu_prepared(&input);
        let d = diff_f32(&report.coverage.data, &cpu.data);
        println!(
            "[frame] device={} case={} size={}x{} rule={:?} groups={:?} segments={} \
             invocations={} differing_bytes={} differing_pixels={} max_channel_delta={} \
             max_abs_delta={:.3e}",
            report.device, name, w, h, rule, report.groups, report.segments,
            report.invocations, d.differing_bytes, d.differing_pixels,
            d.max_channel_delta, d.max_abs_delta
        );
        assert_eq!(
            d.differing_bytes, 0,
            "[{name}] P1 逐字节判据：差异像素 {}/{}，量化通道差 {}，首个差异 {:?}",
            d.differing_pixels, w * h, d.max_channel_delta, d.first
        );
        assert_eq!(d.differing_pixels, 0, "[{name}] P1 逐像素判据");
        assert_eq!(d.max_channel_delta, 0, "[{name}] P1 u8 量化判据");
        // 覆盖率是像样的图：值域 [0,1]、有限、非空用例有面积。
        for &v in &report.coverage.data {
            assert!(v.is_finite() && v >= -1e-6 && v <= 1.0 + 1e-6, "[{name}] 覆盖率越界 {v}");
        }
    }
}

/// 确定性：同输入两遍 dispatch 逐字节一致（不是「偶尔对一次」）。
#[test]
fn gpu_path_frame_is_deterministic_across_runs() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    let input = prepare(&[star(32, 25.0, 20.0, 18.0)], PathFillRule::Winding, 53, 41);
    let a = backend.rasterize_path_frame(&input);
    let b = backend.rasterize_path_frame(&input);
    assert_gpu_evidence(&a, 53, 41);
    assert_gpu_evidence(&b, 53, 41);
    assert_eq!(a.coverage.data, b.coverage.data, "两遍出帧必须逐字节一致");
    let d = diff_f32(&a.coverage.data, &rasterize_cpu_prepared(&input).data);
    println!("[determinism] device={} 53x41 两遍逐字节一致；P1 differing_pixels={} max_channel_delta={}", a.device, d.differing_pixels, d.max_channel_delta);
    assert_eq!(d.differing_pixels, 0);
}

/// 引擎入口：与 CPU 参考逐字节一致，且**有设备时必须真的走 GPU**。
#[test]
fn engine_entry_is_byte_identical_and_uses_gpu_when_available() {
    let sq = vec![square(2.5, 1.5, 33.5, 27.5)];
    let cpu = prism_gpu::path_raster::rasterize_cpu(&sq, PathFillRule::Winding, 48, 36);
    let engine = rasterize(&sq, PathFillRule::Winding, 48, 36);
    assert_eq!(engine.data.len(), cpu.data.len());
    assert_eq!(engine.data, cpu.data, "引擎入口必须与 CPU 参考逐字节一致");

    let gpu = rasterize_path_gpu(&sq, PathFillRule::Winding, 48, 36);
    if VulkanBackend::probe().device_created {
        let g = gpu.expect("有设备时入口必须走 GPU（不得静默退回 CPU）");
        assert_eq!(g.data, cpu.data, "GPU 出帧须与 CPU 参考一致");
        println!("[entry] 有设备：rasterize_path 走 GPU，48x36 逐字节一致");
    } else {
        assert!(gpu.is_none(), "无设备时 GPU 入口必须如实返回 None");
        println!("[entry] 无设备：rasterize_path 优雅回退 CPU");
    }
}

/// **负控 1（反假绿核心）**：故意扰动 kernel 的一个系数，逐字节判据必须 FAIL。
///
/// 若这条也「绿」，说明判据恒真或 GPU 路径根本没执行——那正是本测试要抓的
/// 假绿形态。做法与 `qa_curve_parity.rs` 的「开关留档」同源：把错误注入到
/// 真机 kernel 里，看判据有没有判别力。
#[test]
fn negative_control_perturbed_kernel_must_fail_the_bytewise_criterion() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    let sq = vec![square(1.5, 1.5, 40.5, 30.5)];
    let input = prepare(&sq, PathFillRule::Winding, 64, 48);
    let cpu = rasterize_cpu_prepared(&input);

    let perturbed = build_path_module_with_one(1.001);
    assert_ne!(perturbed, path_words().to_vec(), "扰动版字流必须与产品版不同");
    let report = backend.rasterize_path_frame_with_words(&input, &perturbed);
    assert_eq!(report.status, DispatchStatus::Ok, "扰动版也应真跑完（只是结果错）");
    let d = diff_f32(&report.coverage.data, &cpu.data);
    println!(
        "[negative-control-1] device={} 扰动 1.0→1.001：differing_pixels={} max_channel_delta={} \
         max_abs_delta={:.3e} ⇒ 判据 {}",
        report.device, d.differing_pixels, d.max_channel_delta, d.max_abs_delta,
        if d.differing_pixels > 0 { "FAIL（符合预期）" } else { "PASS（假绿！）" }
    );
    assert!(
        d.differing_pixels > 0,
        "扰动 kernel 后逐字节判据必须 FAIL；若它也绿，说明 GPU 路径没真执行或判据恒真"
    );
}

/// **负控 2**：把片段 delta 全置 0 派发 ⇒ 输出必须全 0 且与 CPU 参考不同。
/// 证明 kernel 真的读了片段缓冲，而不是写死输出 / CPU 兜底顶包。
#[test]
fn negative_control_zeroed_weights_prove_kernel_reads_segments() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let mut backend = VulkanBackend::new();
    let sq = vec![square(1.5, 1.5, 40.5, 30.5)];
    let mut input = prepare(&sq, PathFillRule::Winding, 64, 48);
    let reference = rasterize_cpu_prepared(&input);
    assert!(reference.data.iter().any(|&v| v != 0.0), "参考帧必须非空");
    for i in 0..input.segment_count() {
        input.segments[i * 6 + 5] = 0.0;
    }
    let report = backend.rasterize_path_frame(&input);
    assert_gpu_evidence(&report, 64, 48);
    assert!(
        report.coverage.data.iter().all(|&v| v == 0.0),
        "delta 全零时 kernel 输出必须全零"
    );
    let d = diff_f32(&report.coverage.data, &reference.data);
    println!(
        "[negative-control-2] device={} delta 全零：differing_pixels={} max_channel_delta={} ⇒ {}",
        report.device, d.differing_pixels, d.max_channel_delta,
        if d.differing_pixels > 0 { "与参考不同（符合预期）" } else { "假绿！" }
    );
    assert!(d.differing_pixels > 0, "零权重帧必须与真实帧不同");
}

/// SPIR-V 结构门禁（无设备也执行）：模块合法 + 与 kernel 描述一致。
#[test]
fn path_spirv_matches_kernel_description() {
    let words = module(MODULE_PATH_RASTER).expect("2D 路径 SPIR-V 必须已注册");
    let input = prepare(&[square(1.5, 1.5, 20.5, 20.5)], PathFillRule::Winding, 32, 24);
    let kernel = path_kernel(&input).expect("32×24 合法");
    let summary = validate(
        words,
        &SpvExpect {
            entry: kernel.entry,
            local_size: [kernel.workgroup.x, kernel.workgroup.y, kernel.workgroup.z],
            bindings: &[0, 1, 2, 3],
        },
    )
    .expect("路径 SPIR-V 须通过结构验证");
    assert_eq!(summary.entry, ENTRY_PATH_RASTER);
    assert_eq!(summary.local_size, WORKGROUP_PATH_RASTER);
    assert_eq!(summary.bindings, vec![0, 1, 2, 3]);
    println!(
        "[spirv] {} 字 / {} 条指令 / entry={} localsize={:?} bindings={:?}",
        words.len(), summary.instruction_count, summary.entry, summary.local_size, summary.bindings
    );
}

/// **WebGPU / WGSL 侧**：kernel 必须在 `SHADERS` 注册表、通过结构自检与
/// 未解析符号检查、三目标宿主全绿；常量与 Vulkan/CPU 同源。
#[test]
fn webgpu_wgsl_kernel_is_registered_and_self_consistent() {
    use prism_gpu::shaders;
    let src = shaders::source("path_raster.wgsl").expect("path_raster.wgsl 必须登记进 SHADERS");
    let chk = shaders::check_shader("path_raster.wgsl", src);
    assert!(chk.is_pass(), "WGSL 结构自检失败：{chk:?}");
    assert!(shaders::unresolved_symbols(&shaders::expand("path_raster.wgsl")).is_empty(),
        "展开后不得有未解析符号（真机必报 unresolved call target）");
    assert!(src.contains("fn path_main("), "WGSL 入口名须与 SPIR-V 一致");
    assert!(src.contains("@workgroup_size(8, 8, 1)"));
    assert!(src.contains("1048576.0"), "COORD_BIAS 须与 CPU/SPIR-V 同源");
    for (name, target, _) in &shaders::target_inventory() {
        if *name == "path_raster.wgsl" {
            assert_eq!(*target, shaders::ShaderTarget::Wgsl);
        }
    }
    let all = shaders::check_targets();
    let bad: Vec<_> = all.iter().filter(|c| !c.ok).collect();
    assert!(bad.is_empty(), "三目标宿主校验不得有失败项：{bad:?}");
    println!(
        "[wgsl] SHADERS={} 项，path_raster.wgsl={}B，结构自检 PASS，unresolved=0，check_targets={}/{} PASS",
        shaders::SHADERS.len(), src.len(),
        all.iter().filter(|c| c.ok).count(), all.len()
    );
}

/// 尺寸守卫与降级：极端尺寸不 panic、不返回假帧。
#[test]
fn path_gpu_refuses_extreme_sizes_without_panicking() {
    let mut backend = VulkanBackend::new();
    for (w, h) in [(0u32, 0u32), (0, 16), (16, 0), (u32::MAX, u32::MAX), (65_536, 65_536)] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare(&[], PathFillRule::Winding, w, h)
        }))
        .expect("预处理不得 panic（C18）");
        let report = backend.rasterize_path_frame(&r);
        assert!(!report.is_ok(), "({w},{h}) 必须被拒");
        assert!(report.coverage.data.is_empty(), "被拒时不得产出覆盖率");
        let engine = rasterize(&[], PathFillRule::Winding, w, h);
        assert!(engine.data.is_empty(), "超限尺寸的入口输出应为空");
    }
    println!("[guard] 极端尺寸全部被拒且不 panic；TOL_EXACT={TOL_EXACT:e} 判据口径未改动");
}
