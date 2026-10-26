//! W3 最小闭环：**真实 Vulkan compute dispatch ↔ CPU 参考逐值比对**。
//!
//! 闭环链（每一步都在真机上执行，不是模拟）：
//!
//! ```text
//! VkInstance → VkPhysicalDevice（lavapipe/llvmpipe 软栅或硬件 GPU）
//!   → VkDevice + graphics queue
//!   → 两个 SSBO（binding 0 输入 / binding 1 输出）
//!   → VkShaderModule（自研 SPIR-V 发射器的 mul2 words）
//!   → VkComputePipeline + VkDescriptorSet
//!   → vkCmdDispatch → vkQueueSubmit → vkWaitForFences
//!   → vkMapMemory 读回
//!   → 与 spirv::cpu_reference_mul2 逐元素比对（bit 相等 + ULP 统计）
//! ```
//!
//! # 环境自门控（skip 而非 fail）
//!
//! * 无 Vulkan 设备（未装 ICD / 无 libvulkan）：打印探测报告后 **skip**——
//!   本机不存在可跑的设备，失败没有意义，也不能假装跑过；
//! * 有设备：**必须真跑**并把设备名打进输出。本仓库 CI 沙箱装有 lavapipe
//!   （`lvp_icd.json`），因此这条测试在此环境**真实执行**。
//!
//! # 判等口径
//!
//! `mul2` 是纯乘法（`x * 2.0`）：二进制乘 2 只改指数位，正常数/次正规数
//! 都**无舍入**，故期望值与 `cpu_reference_mul2` **逐位相同**（ULP = 0）。
//! 断言写成 `to_bits()` 相等而不是 `==`，是为了把 `-0.0` 与 `+0.0` 的差别
//! 也抓住——那正是「逐位一致」这一主张的边界。若某设备刷次正规数为零，
//! 本测试会如实报出 ULP 与绝对差，而不是用容差把差异盖掉。

use prism_gpu::compute::spirv_probe_kernel;
use prism_gpu::vulkan::compute::DispatchStatus;
use prism_gpu::vulkan::spirv::{cpu_reference_mul2, module, validate, SpvExpect};
use prism_gpu::vulkan::VulkanBackend;
use prism_gpu::GpuDevice;

/// 探针输入（256 个 f32 = 四个工作组，正好铺满 kernel 的定长绑定）。
///
/// 覆盖：±0、整数、负值、非二进制小数、极大值（×2 溢出为 inf）、
/// 极小数（次正规，×2 仍为次正规）。全部值 ×2 都可被 f32 精确表示或
/// 按 IEEE 溢出为 ±inf，因此「逐位一致」是可达的硬标准。
fn probe_input() -> Vec<f32> {
    const SPECIAL: [f32; 10] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -2.5,
        1.0e-30,
        1.0e-45, // 次正规数（最小正 subnormal 量级）
        3.402_823_5e38,
        -3.402_823_5e38,
    ];
    let mut input = Vec::with_capacity(256);
    input.extend_from_slice(&SPECIAL);
    for i in SPECIAL.len()..256 {
        // 确定性构造：整数 + 1/8 步进，落在 [-16, 14] 的稠密区间。
        input.push(i as f32 * 0.125 - 16.0);
    }
    input
}

/// IEEE-754 有序键（`-0.0` 与 `+0.0` 视为相邻，NaN 落在正端之外）。
fn ordered_key(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// 两个 f32 之间的 ULP 距离（无符号，跨零也正确）。
fn ulp_distance(a: f32, b: f32) -> u64 {
    ordered_key(a).abs_diff(ordered_key(b)) as u64
}

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
    assert!(
        backend.probed_device(),
        "probe() 说 device 已建，backend 却没有探测上下文"
    );
    let name = backend.device_name().unwrap_or("<unnamed>");
    println!(
        "[device] {name}（枚举到 {} 个物理设备，选偏好最高者）",
        report.device_count
    );
    true
}

/// **主判对**：真机 dispatch 的读回值 vs `cpu_reference_mul2`，逐位比对。
#[test]
fn real_dispatch_matches_cpu_reference_bitwise() {
    let backend_probe = VulkanBackend::new();
    if !device_ready(&backend_probe) {
        return;
    }

    let kernel = spirv_probe_kernel();
    let words = module(kernel.spv_module).expect("探针 SPIR-V 模块必须已嵌入");
    let summary = validate(
        words,
        &SpvExpect {
            entry: kernel.entry,
            local_size: [kernel.workgroup.x, kernel.workgroup.y, kernel.workgroup.z],
            bindings: &[0, 1],
        },
    )
    .expect("SPIR-V 结构验证须通过（上机前的静态门禁）");
    println!(
        "[spirv] {} 字 / {} 条指令 / entry={} localsize={:?} bindings={:?}",
        words.len(),
        summary.instruction_count,
        summary.entry,
        summary.local_size,
        summary.bindings
    );

    let mut backend = VulkanBackend::new();
    // 自检 = 完整路径真跑一次；这一步失败等价于「设备在但 dispatch 不可用」。
    assert!(
        backend.compute_available(),
        "compute_available() 为 false：设备存在但完整 dispatch 路径不可用"
    );

    let input = probe_input();
    let extent = [input.len() as u32, 1, 1];
    let report = backend.compute_dispatch(&kernel, extent, &input);
    assert_eq!(
        report.status,
        DispatchStatus::Ok,
        "真机派发失败：{} [{}]",
        report.status.as_str(),
        report.detail
    );
    assert_eq!(report.groups, [4, 1, 1], "256 / 64 → 4 个工作组");
    assert_eq!(report.invocations, 256);
    assert_eq!(report.wasted_invocations, 0, "整除形态不应有尾组浪费");
    assert!(
        !report.device.is_empty(),
        "成功报告必须带设备名（不许无名成功）"
    );
    assert_eq!(
        report.output.len(),
        input.len(),
        "读回元素数 = 输出绑定容量（256）"
    );

    // CPU 参考：与 kernel 宣称的语义同源（不是测试里另写一遍乘法）。
    let mut expected = vec![0.0f32; input.len()];
    cpu_reference_mul2(&input, &mut expected);

    let mut mismatches = 0usize;
    let mut max_ulp = 0u64;
    let mut max_abs = 0.0f32;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for i in 0..input.len() {
        let got = report.output[i];
        let want = expected[i];
        if got.to_bits() != want.to_bits() {
            mismatches += 1;
            max_ulp = max_ulp.max(ulp_distance(got, want));
            let d = (got - want).abs();
            if d.is_finite() {
                max_abs = max_abs.max(d);
            }
            if first_bad.is_none() {
                first_bad = Some((i, got, want));
            }
        }
    }

    println!(
        "[readback] device={} elements={} bitwise_mismatches={} max_ulp={} max_abs_diff={:e}",
        report.device,
        report.output.len(),
        mismatches,
        max_ulp,
        max_abs
    );

    assert_eq!(
        mismatches, 0,
        "GPU 读回须与 cpu_reference_mul2 逐位一致；首个不符：{:?}",
        first_bad
    );
    assert_eq!(max_ulp, 0, "最大 ULP 差应为 0");
    assert_eq!(
        report.output[3].to_bits(),
        (-2.0f32).to_bits(),
        "定点抽查：-1.0 × 2 = -2.0"
    );
}

/// **可知性**：`is_available()` = **compute 自检 ∧ 渲染自检**（V2.1 Q4 裁定）。
///
/// 这条测试锁住三件事：
/// 1. 有设备 ⇒ 两条自检都真跑通（本机必须走到 `true` 分支）；
/// 2. `is_available()` **严格等同** `compute_available() && render_available()`——
///    既不是「探测到设备」，也不是任一单项；
/// 3. **可用性与出图能力不脱钩**：`render_available()` 为真 ⇒ `render_scene`
///    必须给出 `w*h*4` 的有效帧；为假 ⇒ 必须是无效空缓冲。
///    （V2.1 之前 `render_scene` 恒空帧，本条曾是「可用 ⇒ 仍空帧」；
///    T-A1 实装后语义翻转为「可用 ⇔ 出图」，这正是 Q4 要闭合的组合。）
#[test]
fn availability_requires_verified_dispatch_not_just_a_device() {
    let mut backend = VulkanBackend::new();
    let report = VulkanBackend::probe();

    // 双自检口径（与设备有无无关，恒成立）。
    assert_eq!(
        backend.is_available(),
        backend.compute_available() && backend.render_available(),
        "is_available 必须严格等同 compute + render 双自检"
    );

    if !report.device_created {
        assert!(
            !backend.is_available(),
            "无设备时 is_available() 必须为 false（不得靠探测撒谎）"
        );
        assert!(!backend.render_available(), "无设备 ⇒ 渲染自检必须为假");
        assert!(backend.device_name().is_none());
        println!("[skip] 无设备：is_available() 诚实为 false");
        return;
    }

    assert!(backend.probed_device(), "设备已创建 ⇒ probed_device 为真");
    let name = backend.device_name().unwrap_or("<unnamed>").to_string();
    assert!(!name.is_empty(), "探测成功却拿不到设备名");

    // 本机有设备：两条自检都必须真跑通，否则闭环不成立。
    assert!(
        backend.compute_available(),
        "设备 {name} 存在但 compute 自检失败 —— 闭环未打通"
    );
    assert!(
        backend.render_available(),
        "设备 {name} 存在但场景渲染自检失败 —— 光追闭环未打通"
    );
    assert_eq!(backend.backend_name(), "vulkan-compute");

    // 可用 ⇔ 出图：4×4 请求必须给出有效帧（不再允许「可用却空帧」）。
    let req = prism_gpu::SceneRenderRequest {
        width: 4,
        height: 4,
        camera: prism_gpu::CameraReq {
            eye: prism_math::Vec3::new(0.0, 0.0, -4.0),
            target: prism_math::Vec3::new(0.0, 0.0, 0.0),
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
    };
    let fb = backend.render_scene(&req);
    assert!(
        fb.is_valid(),
        "渲染自检为真 ⇒ render_scene 必须出图（不得再返回无效帧）"
    );
    assert_eq!(fb.pixels.len(), 4 * 4 * 4, "有效帧须铺满 w*h*4 字节");

    println!("[available] device={name} compute=true render=true（双自检）");
}

/// 同一设备上重复派发逐位稳定（确定性：不是「偶尔对一次」）。
#[test]
fn repeated_dispatch_is_bitwise_deterministic() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let kernel = spirv_probe_kernel();
    let input = probe_input();
    let mut backend = VulkanBackend::new();

    let a = backend.compute_dispatch(&kernel, [256, 1, 1], &input);
    let b = backend.compute_dispatch(&kernel, [256, 1, 1], &input);
    assert_eq!(a.status, DispatchStatus::Ok, "{}", a.detail);
    assert_eq!(b.status, DispatchStatus::Ok, "{}", b.detail);
    assert_eq!(a.output.len(), b.output.len());
    for i in 0..a.output.len() {
        assert_eq!(
            a.output[i].to_bits(),
            b.output[i].to_bits(),
            "第 {i} 个元素两遍派发不逐位一致"
        );
    }
    println!(
        "[determinism] device={} 两遍派发 {} 元素逐位一致",
        a.device,
        a.output.len()
    );
}

/// 尾组未写区域恒为 `0.0`（不把未初始化设备内存当结果读回）。
///
/// extent=100 ⇒ 2 个工作组 = 128 次调用（尾组浪费 28）。读回长度仍是输出
/// 绑定的容量 256；`[100, 256)` 区间没有任何合法调用写过（尾组调用写的是
/// 输入缓冲的零填充区），必须逐位等于 `+0.0`。
#[test]
fn unwritten_tail_reads_back_as_zero() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let kernel = spirv_probe_kernel();
    let input = vec![3.0f32; 100];
    let mut backend = VulkanBackend::new();
    let report = backend.compute_dispatch(&kernel, [100, 1, 1], &input);
    assert_eq!(report.status, DispatchStatus::Ok, "{}", report.detail);
    assert_eq!(report.groups, [2, 1, 1]);
    assert_eq!(
        report.wasted_invocations, 28,
        "100 / 64 → 128 调用，浪费 28"
    );
    assert_eq!(report.output.len(), 256, "读回长度 = 绑定容量");
    for (i, v) in report.output.iter().enumerate().take(100) {
        assert_eq!(v.to_bits(), 6.0f32.to_bits(), "第 {i} 个元素应为 3×2");
    }
    for (i, v) in report.output.iter().enumerate().skip(100) {
        assert_eq!(
            v.to_bits(),
            0.0f32.to_bits(),
            "第 {i} 个元素未被合法调用写过，应为 +0.0（实际 {v:e}）"
        );
    }
    println!(
        "[tail] 100/256 有效，尾区 {} 个元素全为 +0.0",
        report.output.len() - 100
    );
}

/// 越界守卫：extent 超过写绑定容量时**拒绝上机**（不越界写设备内存）。
#[test]
fn oversized_extent_is_refused_before_dispatch() {
    let probe = VulkanBackend::new();
    if !device_ready(&probe) {
        return;
    }
    let kernel = spirv_probe_kernel(); // 写绑定 = 256 个 f32
    let mut backend = VulkanBackend::new();
    let input = vec![1.0f32; 16];
    let r = backend.compute_dispatch(&kernel, [512, 1, 1], &input);
    assert_eq!(
        r.status,
        DispatchStatus::BufferFailed,
        "超出输出容量的 extent 必须被拒，实际：{}（{}）",
        r.status.as_str(),
        r.detail
    );
    assert!(r.output.is_empty(), "被拒时不得返回读回数据");
    assert!(
        r.detail.contains("超出写绑定容量"),
        "拒绝原因须可诊断，实际：{}",
        r.detail
    );
    println!("[guard] 512 > 256 已被拒：{}", r.detail);
}
