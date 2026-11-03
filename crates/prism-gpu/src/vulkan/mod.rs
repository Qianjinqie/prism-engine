//! Vulkan 后端（D1，Linux/Windows 原生 GPU 路径）。
//!
//! # ⚠️ 诚实边界（必读）
//!
//! **compute 路径已跑通**：instance → physical device → device+queue → 两个
//! SSBO → shader module（自研 SPIR-V 发射器的 `mul2` words）→ compute
//! pipeline → descriptor set → `vkCmdDispatch` → fence 等待 → map 读回 →
//! 与 [`spirv::cpu_reference_mul2`] 逐位比对。这条闭环由
//! [`VulkanBackend::compute_available`] **真跑一次自检**证明（不是靠探测推测）。
//!
//! **背景帧已能出图（W4）**：[`background`] 用 compute kernel 离屏渲染背景，
//! 读回的 RGBA8 与 [`crate::background::render_background_cpu`] **逐字节一致**
//! （`tests/vulkan_background_gpu.rs` 全像素对拍）。无窗口、无 swapchain、
//! 无 graphics pipeline——本沙箱无显示器，也不需要一个逐点算子走光栅路。
//!
//! **光追场景帧已能出图（V2.1 T-A1）**：[`scene`] 用 4 绑定的 compute
//! megakernel（[`spirv_rt`] 发射，含球/灯两层**动态循环**）离屏渲染
//! [`crate::SceneRenderRequest`]，读回 RGBA8；与
//! [`crate::CpuRefBackend::render_scene`] 的对拍判据分两层：① GPU **自跑双次
//! 逐位一致**（确定性硬判据）；② 与 CPU 参考的**统计 diff**（光追含
//! sqrt/除法/归一化，末位差异属预期，故跨后端只承诺视觉等价——C15 口径）。
//! 仍然没有 swapchain / render pass / graphics pipeline：这是一个逐像素独立
//! 算子，走光栅路只会引入格式转换与插值误差，对正确性毫无收益。
//!
//! [`VulkanBackend::is_available`] 的口径 = **compute 自检 ∧ 渲染自检**：
//! 探测到设备但任一自检失败（无 ICD / 管线被拒 / 读回不符）一律 `false`。
//! 三条独立能力位各有自检、互不冒充：
//! * [`VulkanBackend::compute_available`]：mul2 探针闭环；
//! * [`VulkanBackend::frame_available`]：背景帧真跑一帧 16×16；
//! * [`VulkanBackend::render_available`]：场景帧真跑一帧 4×4（V2.1 新增）。
//!
//! # 零第三方依赖（D-15）
//!
//! 不引入 `ash` / `vulkano` / `libloading` 等任何 crate。加载走手写
//! `dlopen`/`dlsym`（见 [`loader`] 子模块），所有 Vulkan `struct` /
//! 函数指针均 `#[repr(C)]` 手写，字段布局与 `<vulkan/vulkan_core.h>` 逐一对齐。
//! Vulkan 是**平台原生图形 API**，调用它不构成第三方依赖。
//!
//! # 编译期不依赖 `libvulkan`
//!
//! 本模块**不链接** `libvulkan`（无 `#[link(name = "vulkan")]`）。loader 库
//! 在**运行期**经 `dlopen` 探测；拿不到 `libvulkan.so.1` / `libvulkan.so`
//! 就判定不可用。故即便当前平台没有 Vulkan（如未装 mesa 的系统），本 crate
//! 仍可正常编译、链接、测试。
//!
//! # 运行环境说明
//!
//! 安装 `mesa-vulkan-drivers` 后可获得
//! **lavapipe（llvmpipe）软件 ICD**（`lvp_icd.json` →
//! `/usr/share/vulkan/icd.d/`），无需 X/Wayland 即可枚举到物理设备
//! （`VK_PHYSICAL_DEVICE_TYPE_CPU`）。设备可用时 compute 派发
//! （SPIR-V → pipeline → dispatch → 读回）真实执行，由
//! `compute::tests::real_device_compute_dispatch_smoke` 运行期自门控
//! 验证；当前平台无可用设备时按不支持处理并如实报告。llvmpipe 是 CPU 软栅：
//! 只证明 GPU 代码路径功能正确，**不构成硬件加速比证据**。
//!
//! # 平台门禁
//!
//! * **Linux / Windows（native）**：真实探测链编译（[`loader`]）。
//! * **wasm32**：`loader` 子模块被整体剔除、探测链置空（wasm 无 `libdl`），
//!   [`VulkanBackend`] 仍可见但恒报不可用。
//!   门禁：`#[cfg(not(target_arch = "wasm32"))]`。

use crate::{Framebuffer, GpuDevice, SceneRenderRequest};

// ---------------------------------------------------------------------------
// 探测链实现（仅非 wasm 目标）
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub mod loader;

// ---------------------------------------------------------------------------
// compute 管线：SPIR-V 供给 + dispatch / queue 提交
// ---------------------------------------------------------------------------
//
// `spirv` 是纯数据（手写 SPIR-V 字流），**所有平台**都编译——wasm 侧也要能
// 报「本 kernel 的 SPIR-V 是什么」；`compute` 含真机 FFI 装配路径，wasm 上
// 只剩「如实报 unsupported」的入口（见其 `dispatch` 的 cfg 分支）。

/// 手写 / 嵌入的 SPIR-V 模块表与结构校验。
pub mod spirv;
/// A2 / W2：`rt_env_probe.spv` 的编译期字流（生成物，见文件头）。
#[cfg(not(target_arch = "wasm32"))]
pub mod spirv_env_probe;

/// 背景渲染 kernel 的 **SPIR-V 生成**（自研发射器：同式、禁收缩、无 libm）。
///
/// 与 [`spirv`] 的分工：`spirv` 是「模块是什么」（表 + 结构校验），本模块是
/// 「模块怎么被造出来」（id 分配/段序/常量去重 + 逐式对应
/// [`crate::background`] 的语义发射）。产物经 [`spirv::module`] 统一取用。
pub mod spirv_bg;

/// 光追场景 kernel 的 **SPIR-V 生成**（含**动态循环**的 megakernel）。
///
/// 与 [`spirv_bg`] 同级：同一套发射器纪律，区别在本模块含两层 `OpLoopMerge`
/// （球/灯循环，上界来自参数块，不展开）与两个 `Function` 局部变量块。
/// 语义逐式对应 [`crate::CpuRefBackend::render_scene`]（球面求交 → 直接光 →
/// aesthetic），产物经 [`spirv::module`] 以 `spv_scene` 取用。
pub mod spirv_rt;

/// compute pipeline 装配 + 队列提交（真机路径；有设备真跑，无设备如实报
/// `DeviceUnavailable`，不假装跑过）。
pub mod compute;

/// **背景帧的 GPU 出图**：compute 离屏 → 字节读回 RGBA8（无窗口/swapchain）。
///
/// SPIR-V（[`spirv_bg`]）→ compute pipeline → dispatch → fence → 读回，
/// 输出与 [`crate::background::render_background_cpu`] 逐字节对拍。
pub mod background;

/// **光追场景帧的 GPU 出图**：同一套 compute 闭环，4 个绑定（参数/球/灯/输出）。
///
/// SPIR-V（[`spirv_rt`]）→ compute pipeline → dispatch → fence → 读回 RGBA8；
/// 输出与 [`crate::CpuRefBackend::render_scene`] 做**统计对拍**（GPU 自跑双次
/// 逐位一致是硬判据，跨后端视觉等价走统计阈值——见 `tests/qa_render_scene.rs`）。
pub mod scene;

/// **V3 十一绑定场景 kernel 的真机派发 harness**（D-21 / V6 第三条腿）。
///
/// 只在**native 的测试构建**里存在：它读的是**入库产物**
/// `shaders/spv/scene_frame_v3.spv`（由 `tools/wgsl2spv` 用 naga 生成），
/// release 二进制一个字节都不多背。判据与 `tests/qa_render_scene.rs` 逐字同口径
/// （软件 ICD 上 `differing_pixels == 0`、`max_channel_delta == 0`）。
///
/// wasm32 下也排除：本 crate 的 wasm 分支没有 `VulkanContext`（走 WebGPU 宿主），
/// 而 `compute::dispatch_inputs_with_words` 本来就是 `#[cfg(not(wasm32))]`。
#[cfg(all(test, not(target_arch = "wasm32")))]
mod v3_frame;

// ---------------------------------------------------------------------------
// 探测结果（跨平台可见的诊断结构）
// ---------------------------------------------------------------------------

/// 一次 Vulkan 能力探测的结构化结果。
///
/// 用于诊断「探测链走到了哪一步、在哪一步失败」——这比一个裸 `bool` 更有
/// 价值，也便于测试对探测链的**真实状态**做断言（而非恒真断言）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeReport {
    /// 本次探测是否在受支持平台（非 wasm）执行。
    pub supported_platform: bool,
    /// 是否成功 `dlopen` 到 Vulkan loader 库。
    pub loader_loaded: bool,
    /// 是否成功创建 `VkInstance`。
    pub instance_created: bool,
    /// 枚举到的物理设备数量（未走到该步则为 `0`）。
    pub device_count: usize,
    /// 是否成功创建 `VkDevice`（含 graphics 队列）。
    pub device_created: bool,
    /// 失败 / 降级原因（成功时为 `None`）。
    pub reason: Option<ProbeFailure>,
}

impl ProbeReport {
    /// 非 wasm 平台、探测链在最早处即不可用的报告。
    #[cfg(not(target_arch = "wasm32"))]
    fn unavailable(reason: ProbeFailure) -> Self {
        ProbeReport {
            supported_platform: true,
            loader_loaded: false,
            instance_created: false,
            device_count: 0,
            device_created: false,
            reason: Some(reason),
        }
    }

    /// wasm32 平台的标准报告（本平台不存在原生 Vulkan）。
    #[cfg(target_arch = "wasm32")]
    fn unsupported_platform() -> Self {
        ProbeReport {
            supported_platform: false,
            loader_loaded: false,
            instance_created: false,
            device_count: 0,
            device_created: false,
            reason: Some(ProbeFailure::UnsupportedPlatform),
        }
    }
}

/// 探测链降级原因（逐步定位）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeFailure {
    /// wasm32：本平台不存在原生 Vulkan。
    UnsupportedPlatform,
    /// `dlopen("libvulkan.so.1")` 与 `dlopen("libvulkan.so")` **均失败**
    /// （未安装 Vulkan loader）。
    LoaderLoadFailed,
    /// loader 已加载但 `dlsym("vkGetInstanceProcAddr")` 返回空
    /// （符号缺失，异常情况）。
    GetInstanceProcAddrMissing,
    /// `vkGetInstanceProcAddr` 未给出 `vkCreateInstance`
    /// （loader 异常，理论上不应发生）。
    CreateInstanceMissing,
    /// `vkCreateInstance` 返回非 `VK_SUCCESS`（如无可用 ICD：
    /// `VK_ERROR_INCOMPATIBLE_DRIVER`）。
    InstanceCreationFailed,
    /// `vkEnumeratePhysicalDevices` 失败或无设备。
    NoPhysicalDevice,
    /// 物理设备没有 `VK_QUEUE_GRAPHICS_BIT` 队列族。
    NoGraphicsQueueFamily,
    /// `vkCreateDevice` 返回非 `VK_SUCCESS`。
    DeviceCreationFailed,
    /// 设备已创建但 `vkGetDeviceQueue` 未给出队列句柄。
    QueueFetchFailed,
}

impl ProbeFailure {
    /// 稳定短名，用于诊断日志。
    pub fn as_str(self) -> &'static str {
        match self {
            ProbeFailure::UnsupportedPlatform => "unsupported-platform",
            ProbeFailure::LoaderLoadFailed => "loader-load-failed",
            ProbeFailure::GetInstanceProcAddrMissing => "get-instance-proc-addr-missing",
            ProbeFailure::CreateInstanceMissing => "create-instance-missing",
            ProbeFailure::InstanceCreationFailed => "instance-creation-failed",
            ProbeFailure::NoPhysicalDevice => "no-physical-device",
            ProbeFailure::NoGraphicsQueueFamily => "no-graphics-queue-family",
            ProbeFailure::DeviceCreationFailed => "device-creation-failed",
            ProbeFailure::QueueFetchFailed => "queue-fetch-failed",
        }
    }
}

// ---------------------------------------------------------------------------
// 已建立的 Vulkan 上下文（探测成功后缓存；`Drop` 时释放）
// ---------------------------------------------------------------------------

/// 探测成功后缓存的 Vulkan 句柄集合。
///
/// 持有它即代表「探测链成功」。销毁顺序与创建顺序相反（device 先于
/// instance），且全部在 [`VulkanContext::drop`] 中释放——**不 panic**。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct VulkanContext {
    /// `VkInstance`。
    instance: loader::VkInstance,
    /// 选中的 `VkPhysicalDevice`（compute 管线查内存属性用）。
    physical_device: loader::VkPhysicalDevice,
    /// `VkDevice`。
    device: loader::VkDevice,
    /// graphics 队列句柄。
    queue: loader::VkQueue,
    /// graphics 队列族索引（诊断用）。
    graphics_queue_family: u32,
    /// 物理设备数量（诊断用）。
    device_count: usize,
    /// `VkPhysicalDeviceProperties::deviceName`（如 `llvmpipe (LLVM 20.1.2, 256 bits)`）。
    ///
    /// 报告与诊断日志的唯一设备标识来源——`is_available()` 为真时必须能说出来
    /// 「在哪个设备上真跑过」，故探测期就取好、不在报告里现编。
    device_name: String,
    /// 从实例取到的函数表（`Drop` 时需要 `vkDestroyDevice`）。
    instance_fns: loader::InstanceFnTable,
    /// 从设备取到的函数表（`vkDestroyDevice`）。
    device_fns: loader::DeviceFnTable,
    /// loader 库句柄：**最后**释放者，`Drop` 字段顺序保证其活到所有 FFI 调用之后。
    ///
    /// Rust 按字段声明序 **正向** drop，故把 `_loader` 放最后：它之后不再有
    /// 字段需要 FFI（`instance`/`device` 只是裸指针，无 `Drop`），安全。
    _loader: loader::VulkanLoader,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for VulkanContext {
    /// 先销毁 device 再销毁 instance（Vulkan 规范要求的逆序），最后 `dlclose`。
    ///
    /// 任一步失败仅忽略返回码——**不 panic**（C18）。
    fn drop(&mut self) {
        // SAFETY: 句柄均由本结构体创建、非空且尚未销毁；函数指针在创建时已
        // 判空（None 则不调用）。所有调用都是「销毁」语义，忽略错误码。
        unsafe {
            if let Some(destroy_device) = self.device_fns.destroy_device {
                if !self.device.is_null() {
                    destroy_device(self.device, std::ptr::null());
                    self.device = std::ptr::null_mut();
                }
            }
            if let Some(destroy_instance) = self.instance_fns.destroy_instance {
                if !self.instance.is_null() {
                    destroy_instance(self.instance, std::ptr::null());
                    self.instance = std::ptr::null_mut();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Vulkan 后端把手。
///
/// 这是 **compute 路径的真机装配 + 真实设备探测**：它能真跑 kernel，并且
/// **两条出图路径都已实装**——背景帧（[`background`]）与光追场景帧
/// （[`scene`]，V2.1 T-A1）。没有图形管线（swapchain / render pass /
/// graphics pipeline），因为这两条路径都是**逐点/逐像素独立算子**，走光栅路
/// 只会引入格式转换与插值误差。
///
/// # 手写 FFI + 运行期探测（硬要求直面）
///
/// 探测链（每一步失败都优雅返回，**不 unwrap / expect / panic**）：
///
/// 1. `dlopen("libvulkan.so.1")` → 失败再试 `"libvulkan.so"` → 都失败即不可用；
/// 2. `dlsym("vkGetInstanceProcAddr")`（取不到即不可用）；
/// 3. `vkCreateInstance`（手写 `VkApplicationInfo` + `VkInstanceCreateInfo`；
///    返回码须为 `VK_SUCCESS`，失败则清理并返回不可用）；
/// 4. `vkEnumeratePhysicalDevices`（两段式：先问数量再取列表；选偏好设备）；
/// 5. `vkCreateDevice`（选一个 graphics 队列族，队列数 1，优先级 1.0）；
/// 6. 成功 → 缓存 instance/device/queue/deviceName，`probed_device()` 返回 `true`。
///
/// # `is_available()` 的口径：**compute 自检 ∧ 渲染自检**，不是「有设备」
///
/// 探测到设备只说明可枚举，不说明 SPIR-V 能被接受、描述符能绑、队列能提交、
/// 读回值正确。V2.1（Q4 裁定）起本标志是**两项真跑自检的合取**：
///
/// * [`VulkanBackend::compute_available`]：mul2 探针闭环（首次调用真跑一次）；
/// * [`VulkanBackend::render_available`]：`spv_scene` 真渲染一帧 4×4 并读回。
///
/// 任一环节失败 ⇒ `false`，无设备平台同样 `false`（优雅降级，不 panic）。
///
/// ⚠️ 语义边界：`is_available() == true` **蕴含** `render_scene` 能出有效帧
/// （这正是 Q4 要闭合的组合：不再允许「报可用却出空帧」）。背景帧是另一条
/// 独立能力位（[`VulkanBackend::frame_available`]），一条通过不替另一条背书。
pub struct VulkanBackend {
    /// 探测会话结果（成功后持有 `VulkanContext`，`Drop` 时释放句柄）。
    ///
    /// 非 wasm：`Option<Box<VulkanContext>>`。
    #[cfg(not(target_arch = "wasm32"))]
    context: Option<Box<VulkanContext>>,
    /// 诊断量：是否探测到物理设备（与 `context.is_some()` 基本一致，独立保留
    /// 便于 wasm 分支统一字段布局，并明确「探测到设备」与「可用」的区别）。
    probed_device: bool,
    /// **完整 dispatch 路径**自检结果缓存（`None` = 尚未自检）。
    ///
    /// 惰性求值：只有被问 [`VulkanBackend::compute_available`] 时才真跑一次
    /// mul2 探针（建管线 + 提交 + 读回 + 与 CPU 参考逐位比对）。探测到设备
    /// 但自检失败（SPIR-V 被拒 / 队列提交失败 / 读回不符）⇒ 缓存 `false`，
    /// 此后恒定 `false`——不重试、不撒谎。
    ///
    /// wasm32 无原生 Vulkan，`compute_available()` 直接返回 `false`，无需缓存
    /// （故该字段在 wasm 下不参与布局，避免 dead_code）。
    #[cfg(not(target_arch = "wasm32"))]
    compute_self_test: std::cell::Cell<Option<bool>>,
    /// **背景帧路径**自检结果缓存（`None` = 尚未自检）。
    ///
    /// 与 `compute_self_test` 的区别：后者只证明「最小 dispatch 装配可用」
    /// （mul2 探针，16 字节 I/O），前者真跑一次 16×16 的**背景帧**并确认
    /// 读回长度 = `w*h*4`。二者都过才说明「GPU 能出这张图」。
    #[cfg(not(target_arch = "wasm32"))]
    frame_self_test: std::cell::Cell<Option<bool>>,
    /// **光追场景帧路径**自检结果缓存（`None` = 尚未自检，V2.1 新增）。
    ///
    /// 语义：首次调用时真渲染一帧 4×4 单球（SPIR-V `spv_scene` → pipeline →
    /// dispatch → 读回），确认读回长度恰为 `4*4*4` **且至少有一个非零字节**
    /// （驱动若返回全零/半截缓冲，自检判失败——不把「空跑一趟」当能力）。
    /// 与 `frame_self_test` 并列：背景帧与场景帧是两条**独立的** kernel 路径，
    /// 一条通过不能替另一条背书。
    #[cfg(not(target_arch = "wasm32"))]
    render_self_test: std::cell::Cell<Option<bool>>,
}

impl VulkanBackend {
    /// 构造一个 Vulkan 后端把手，并立即尝试探测设备。
    ///
    /// 任何失败（无 loader / 无 ICD / 无设备 …）都安全降级为「不可用」，
    /// **永不 panic**（C18）。
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (context, probed_device) = Self::establish_context();
            VulkanBackend {
                context,
                probed_device,
                compute_self_test: std::cell::Cell::new(None),
                frame_self_test: std::cell::Cell::new(None),
                render_self_test: std::cell::Cell::new(None),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // wasm32 无 libdl / libvulkan，直接判为未探测到设备。
            VulkanBackend {
                probed_device: false,
            }
        }
    }

    /// 完整探测（诊断用）：返回本次探测的结构化结果，含失败环节。
    ///
    /// * 非 wasm：执行真实探测链并报告每一步状态。
    /// * wasm32：返回 [`ProbeReport::unsupported_platform`]。
    pub fn probe() -> ProbeReport {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match Self::try_build_context() {
                Ok(ctx) => ProbeReport {
                    supported_platform: true,
                    loader_loaded: true,
                    instance_created: true,
                    device_count: ctx.device_count,
                    device_created: true,
                    reason: None,
                    // ctx 在此 drop：device/instance 被正确销毁（探测不泄漏）。
                },
                Err((stage, reason)) => {
                    let mut report = ProbeReport::unavailable(reason);
                    report.loader_loaded = stage.loader_loaded;
                    report.instance_created = stage.instance_created;
                    report.device_count = stage.device_count;
                    report
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            ProbeReport::unsupported_platform()
        }
    }

    /// 是否在启动时探测到 Vulkan 设备（**诊断量，非可用性**）。
    ///
    /// 请勿用此值代替 [`GpuDevice::is_available`] 做渲染路径选择——二者语义
    /// 不同（见 [`VulkanBackend`] 类型文档）。
    pub fn probed_device(&self) -> bool {
        self.probed_device
    }

    /// **完整 compute dispatch 路径**是否可用（本后端 `is_available` 的唯一依据）。
    ///
    /// 「可用」的定义不是「探测到设备」，而是**真的把一条 kernel 跑完并读回**：
    /// 首次调用时在设备上装配 mul2 探针（SPIR-V → shader module → 描述符 →
    /// pipeline → 命令缓冲 → 队列提交 → fence 等待 → map 读回），并把读回值
    /// 与 [`spirv::cpu_reference_mul2`] 做**逐位**比对。
    ///
    /// * 全链路通过且逐位一致 ⇒ `true`；
    /// * 无设备上下文 / 任一步失败 / 读回不一致 ⇒ `false`（不 panic）；
    /// * 结果缓存：失败后不再重试（不让「偶发成功」污染可用性口径）。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_available(&self) -> bool {
        if let Some(v) = self.compute_self_test.get() {
            return v;
        }
        let v = match self.context.as_deref() {
            Some(ctx) => verify_compute_path(ctx),
            None => false,
        };
        self.compute_self_test.set(Some(v));
        v
    }

    /// wasm32：无原生 Vulkan，恒 `false`（不假装跑过）。
    #[cfg(target_arch = "wasm32")]
    pub fn compute_available(&self) -> bool {
        false
    }

    /// 选中设备的名称（`VkPhysicalDeviceProperties::deviceName`）。
    ///
    /// 仅当探测链成功时为 `Some`；报告里出现设备名 = 真的在这个设备上跑过。
    pub fn device_name(&self) -> Option<&str> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.context.as_deref().map(|c| c.device_name.as_str())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }

    /// 已建立的设备上下文（`None` = 本平台无可用 Vulkan 设备）。
    ///
    /// 只给**同一 crate 内的测试 harness** 用（D-21 的 V3 场景 kernel 真机对拍
    /// 需要把入库的 `.spv` 直接喂给 `compute::dispatch_inputs_with_words`），
    /// 故与 `v3_frame` 同一条 `cfg`：非测试构建里连符号都不存在。
    ///
    /// 不对外部调用方暴露：`VulkanContext` 的生命周期与销毁顺序由本类型独占，
    /// 第三方拿到裸引用就会绕过 [`GpuDevice::is_available`] 的能力口径。
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn context(&self) -> Option<&VulkanContext> {
        self.context.as_deref()
    }

    /// 探测成功时选中的 graphics 队列族索引；否则 `None`（诊断量）。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn graphics_queue_family(&self) -> Option<u32> {
        self.context.as_ref().map(|c| c.graphics_queue_family)
    }

    /// 探测成功时枚举到的物理设备数量；否则 `0`（诊断量）。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn device_count(&self) -> usize {
        self.context.as_ref().map_or(0, |c| c.device_count)
    }

    /// 探测成功时缓存的 `VkQueue` 句柄（以整数形式返回，便于诊断打印）；
    /// 否则 `None`。**不用于图形渲染**：本后端的两条出图路径都走 compute
    /// （同一个 compute 队列），没有 swapchain / graphics pipeline 需要它。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn graphics_queue_handle(&self) -> Option<usize> {
        self.context
            .as_ref()
            .map(|c| c.queue as usize)
            .filter(|&p| p != 0)
    }

    /// **compute 派发入口**。
    ///
    /// 走 [`compute::dispatch`]：有设备上下文则在设备上装配 SPIR-V → compute
    /// pipeline → 命令缓冲 → 队列提交 → 读回；无上下文（当前平台无
    /// ICD）则如实返回 [`compute::DispatchStatus::DeviceUnavailable`]，
    /// **不 panic、不返回假输出**。
    ///
    /// 注意：本方法与 [`GpuDevice::is_available`] 语义不同——后者刻画的是
    /// 「能否 service `render_scene`」（compute 自检 ∧ 渲染自检），而本方法是
    /// **任意 kernel 的通用 compute 派发**入口（探针 / 自定义 kernel）。
    pub fn compute_dispatch(
        &mut self,
        kernel: &crate::compute::KernelDesc,
        extent: [u32; 3],
        input: &[f32],
    ) -> compute::ComputeRunReport {
        #[cfg(not(target_arch = "wasm32"))]
        {
            compute::dispatch(self.context.as_deref(), kernel, extent, input)
        }
        #[cfg(target_arch = "wasm32")]
        {
            compute::dispatch((), kernel, extent, input)
        }
    }

    /// **背景帧路径**是否已在本设备上真跑通（`is_available` **不**覆盖它）。
    ///
    /// 语义：首次调用时真渲染一帧 16×16 背景（SPIR-V → pipeline → dispatch →
    /// 读回），并检查读回长度恰为 `16*16*4`。结果缓存，失败不重试。
    ///
    /// 三条能力位的分工（V2.1 起）：
    /// * `is_available()` ≡ [`Self::compute_available`] ∧ [`Self::render_available`]
    ///   ——「能否 service `render_scene`」；
    /// * `frame_available()`：**能出背景帧**（真跑 16×16 kernel + 读回长度自洽）；
    /// * [`Self::render_available`]：**能出光追场景帧**（真跑 4×4 探针帧）。
    ///
    /// 三条各自独立自检、互不背书：背景 kernel 与场景 kernel 是两份不同的
    /// SPIR-V，绑定数与循环结构都不同。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn frame_available(&self) -> bool {
        if let Some(v) = self.frame_self_test.get() {
            return v;
        }
        let v = match self.context.as_deref() {
            Some(ctx) => {
                let report = background::dispatch(Some(ctx), 16, 16, 1.25, 0x9E37_79B1);
                report.is_ok() && report.pixels.len() == 16 * 16 * 4
            }
            None => false,
        };
        self.frame_self_test.set(Some(v));
        v
    }

    /// wasm32：无原生 Vulkan，恒 `false`。
    #[cfg(target_arch = "wasm32")]
    pub fn frame_available(&self) -> bool {
        false
    }

    /// **光追场景帧路径**是否已在本设备上真跑通（V2.1 T-A1；`frame_available`
    /// 的**场景侧**对应物）。
    ///
    /// 语义：首次调用时真渲染一帧 4×4 单球（`spv_scene` → pipeline → dispatch →
    /// 读回），并检查两件事——① 读回长度恰为 `4*4*4`；② 至少有一个非零字节
    /// （否则说明 kernel 根本没写出东西，不能算「能出图」）。结果缓存，失败不重试。
    ///
    /// 与 [`GpuDevice::is_available`] 的关系：`is_available()` =
    /// `compute_available() && render_available()`（Q4 裁定的双自检），因此
    /// 「报可用」等价于「compute 闭环通过**且**这一帧真的出得来」——
    /// 杜绝「`is_available()==true` 但 `render_scene` 恒空帧」的组合。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_available(&self) -> bool {
        if let Some(v) = self.render_self_test.get() {
            return v;
        }
        let v = match self.context.as_deref() {
            Some(ctx) => verify_render_path(ctx),
            None => false,
        };
        self.render_self_test.set(Some(v));
        v
    }

    /// wasm32：无原生 Vulkan，恒 `false`。
    #[cfg(target_arch = "wasm32")]
    pub fn render_available(&self) -> bool {
        false
    }

    /// **出场景帧**（compute 离屏 → 读回 RGBA8；诊断/测试入口）。
    ///
    /// 有设备上下文时真跑 kernel；无设备 / wasm / 超限尺寸时如实返回非 `Ok`
    /// 报告（**不 panic、不返回假帧**）。[`GpuDevice::render_scene`] 是本方法
    /// 的 `Framebuffer` 视图：失败时它返回空缓冲，让调用方回退 CPU。
    pub fn render_scene_frame(&mut self, req: &SceneRenderRequest) -> scene::SceneFrameReport {
        #[cfg(not(target_arch = "wasm32"))]
        {
            scene::dispatch(self.context.as_deref(), req)
        }
        #[cfg(target_arch = "wasm32")]
        {
            scene::dispatch((), req)
        }
    }

    /// **出背景帧**（compute 离屏 → 读回 RGBA8）。
    ///
    /// 有设备上下文时真跑 kernel；无设备 / wasm / 超限尺寸时如实返回
    /// 非 `Ok` 报告（**不 panic、不返回假帧**）。调用方应按
    /// [`background::BackgroundFrameReport::is_ok`] 判断是否回退 CPU 参考。
    pub fn render_background_frame(
        &mut self,
        w: u32,
        h: u32,
        t: f32,
        seed: u64,
    ) -> background::BackgroundFrameReport {
        #[cfg(not(target_arch = "wasm32"))]
        {
            background::dispatch(self.context.as_deref(), w, h, t, seed)
        }
        #[cfg(target_arch = "wasm32")]
        {
            background::dispatch((), w, h, t, seed)
        }
    }

    /// 构造后端时建立探测上下文。
    ///
    /// 成功返回缓存的 [`VulkanContext`]，失败返回 `None`；两种情况下都**不
    /// panic**。内部只调用一次 [`VulkanBackend::try_build_context`]（单一权威
    /// 实现），不做重复探测。
    #[cfg(not(target_arch = "wasm32"))]
    fn establish_context() -> (Option<Box<VulkanContext>>, bool) {
        match Self::try_build_context() {
            Ok(ctx) => (Some(Box::new(ctx)), true),
            Err(_) => (None, false),
        }
    }

    /// 真正构建 [`VulkanContext`]（探测链的唯一权威实现）。
    ///
    /// 返回 `Ok(ctx)` 表示链路全程成功；`Err((stage, reason))` 表示在
    /// `stage` 标明的环节失败（`reason` 给出具体失败原因）。已获取的部分
    /// 资源在各自 `Drop` 中释放，**不 panic**（C18）。
    #[cfg(not(target_arch = "wasm32"))]
    fn try_build_context() -> Result<VulkanContext, (ProbeStage, ProbeFailure)> {
        use loader::{self as vk, PhysicalDeviceInfo, VK_API_VERSION_1_0, VK_SUCCESS};

        let mut stage = ProbeStage::default();

        // --- 步骤 1：dlopen loader ----------------------------------------
        let loader = vk::open().ok_or((stage, ProbeFailure::LoaderLoadFailed))?;
        stage.loader_loaded = true;

        // --- 步骤 2：dlsym(vkGetInstanceProcAddr) --------------------------
        let get_proc = unsafe { loader.load_get_instance_proc_addr() }
            .ok_or((stage, ProbeFailure::GetInstanceProcAddrMissing))?;

        // --- 步骤 3：vkCreateInstance -------------------------------------
        let create_instance = unsafe {
            let sym = get_proc(std::ptr::null_mut(), c"vkCreateInstance".as_ptr());
            if sym.is_null() {
                return Err((stage, ProbeFailure::CreateInstanceMissing));
            }
            std::mem::transmute_copy::<*const std::ffi::c_void, vk::PfnVkCreateInstance>(&sym)
        };
        let (result, instance) = unsafe { vk::create_instance(create_instance, VK_API_VERSION_1_0) };
        if result != VK_SUCCESS || instance.is_null() {
            // instance 创建失败 ⇒ loader 的 Drop 会 dlclose。
            return Err((stage, ProbeFailure::InstanceCreationFailed));
        }
        stage.instance_created = true;

        // 实例级函数表。
        let instance_fns = unsafe { vk::InstanceFnTable::bind_all(get_proc, instance) };
        let destroy_instance = instance_fns
            .destroy_instance
            .ok_or((stage, ProbeFailure::CreateInstanceMissing))?;
        let enumerate = instance_fns
            .enumerate_physical_devices
            .ok_or((stage, ProbeFailure::NoPhysicalDevice))?;
        let get_queue_families = instance_fns
            .get_queue_family_properties
            .ok_or((stage, ProbeFailure::NoGraphicsQueueFamily))?;
        let create_device_fn = instance_fns
            .create_device
            .ok_or((stage, ProbeFailure::DeviceCreationFailed))?;
        // 注意 `get_physical_device_properties` 允许缺失（仅用于诊断与偏好排序），
        // 缺失时退化为枚举顺序第一个设备。

        // --- 步骤 4：vkEnumeratePhysicalDevices（两段式）-------------------
        let mut count: u32 = 0;
        let enumerate_result = unsafe { enumerate(instance, &mut count, std::ptr::null_mut()) };
        if enumerate_result != VK_SUCCESS || count == 0 {
            // 无设备 / 枚举失败：销毁 instance 后返回。
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }
        stage.device_count = count as usize;

        // 上限保护：恶意 / 异常 loader 返回天文数字时避免巨额分配。
        // 取合理上界（远大于现实中的设备数，同时杜绝 `count * size` 溢出）。
        const MAX_DEVICES: u32 = 64;
        let take = count.min(MAX_DEVICES);
        let mut handles: Vec<vk::VkPhysicalDevice> = Vec::new();
        if handles.try_reserve_exact(take as usize).is_err() {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }
        handles.resize(take as usize, std::ptr::null_mut());
        let mut actual: u32 = take;
        let fill_result = unsafe { enumerate(instance, &mut actual, handles.as_mut_ptr()) };
        if fill_result != VK_SUCCESS {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }
        let actual = actual.min(take) as usize;
        handles.truncate(actual);
        if handles.is_empty() {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }

        // 为每个设备收集摘要（类型 + 名称 + graphics 队列族）。
        let mut infos: Vec<PhysicalDeviceInfo> = Vec::new();
        if infos.try_reserve_exact(handles.len()).is_err() {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }
        for &handle in &handles {
            let device_type = if let Some(get_props) = instance_fns.get_physical_device_properties {
                let mut props = vk::VkPhysicalDevicePropertiesStorage::zeroed();
                unsafe { get_props(handle, &mut props) };
                let ty = unsafe { vk::device_type_of(&props) };
                let name = unsafe { vk::device_name_of(&props) };
                infos.push(PhysicalDeviceInfo {
                    handle,
                    device_type: ty,
                    name,
                    graphics_queue_family: None,
                    queue_family_count: 0,
                });
                ty
            } else {
                // 无该入口：按枚举顺序保留，类型记为 Unknown。
                infos.push(PhysicalDeviceInfo {
                    handle,
                    device_type: vk::PhysicalDeviceType::Unknown(-1),
                    name: String::new(),
                    graphics_queue_family: None,
                    queue_family_count: 0,
                });
                vk::PhysicalDeviceType::Unknown(-1)
            };
            let _ = device_type; // 摘要已把类型存进 infos；此处仅为分支可读性保留。

            // 查询队列族：两段式。
            let mut fam_count: u32 = 0;
            unsafe { get_queue_families(handle, &mut fam_count, std::ptr::null_mut()) };
            if fam_count > 0 {
                const MAX_FAMILIES: u32 = 64;
                let fam_take = fam_count.min(MAX_FAMILIES);
                let mut fams: Vec<vk::VkQueueFamilyProperties> = Vec::new();
                if fams.try_reserve_exact(fam_take as usize).is_ok() {
                    fams.resize(fam_take as usize, vk::VkQueueFamilyProperties::default());
                    let mut fam_actual = fam_take;
                    unsafe { get_queue_families(handle, &mut fam_actual, fams.as_mut_ptr()) };
                    let fam_actual = fam_actual.min(fam_take) as usize;
                    fams.truncate(fam_actual);
                    if let Some(info) = infos.last_mut() {
                        info.graphics_queue_family = vk::first_graphics_queue_family(&fams);
                        info.queue_family_count = fam_actual as u32;
                    }
                }
            }
        }

        // 选设备：偏好离散 GPU，且必须有一个 graphics 队列族。
        // 先过滤出「有 graphics 族」的候选，再按偏好排序。
        let mut candidates: Vec<usize> = Vec::new();
        if candidates.try_reserve_exact(infos.len()).is_err() {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoPhysicalDevice));
        }
        for (idx, info) in infos.iter().enumerate() {
            if info.graphics_queue_family.is_some() {
                candidates.push(idx);
            }
        }
        if candidates.is_empty() {
            // 有设备但都无 graphics 队列族 ⇒ 无法建立渲染设备。
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::NoGraphicsQueueFamily));
        }
        let best_idx = {
            let candidate_infos: Vec<PhysicalDeviceInfo> = candidates
                .iter()
                .map(|&i| infos[i].clone())
                .collect();
            match vk::pick_best_device(&candidate_infos) {
                Some(rel) => candidates[rel],
                None => candidates[0],
            }
        };
        let chosen = &infos[best_idx];
        let graphics_queue_family = chosen
            .graphics_queue_family
            .ok_or((stage, ProbeFailure::NoGraphicsQueueFamily))?;
        let physical_device = chosen.handle;
        let device_name = chosen.name.clone();

        // --- 步骤 5：vkCreateDevice ---------------------------------------
        let priority: f32 = 1.0;
        let queue_create_info = vk::VkDeviceQueueCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: graphics_queue_family,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let device_create_info = vk::VkDeviceCreateInfo {
            s_type: vk::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
        };
        let mut device: vk::VkDevice = std::ptr::null_mut();
        let create_result = unsafe {
            create_device_fn(
                physical_device,
                &device_create_info,
                std::ptr::null(),
                &mut device,
            )
        };
        if create_result != VK_SUCCESS || device.is_null() {
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::DeviceCreationFailed));
        }

        // --- 步骤 6：vkGetDeviceQueue -------------------------------------
        let device_fns = unsafe { vk::DeviceFnTable::bind_all(get_proc, instance) };
        let Some(get_device_queue) = device_fns.get_device_queue else {
            // 取不到队列入口：销毁 device + instance 后返回。
            if let Some(destroy_device) = device_fns.destroy_device {
                unsafe { destroy_device(device, std::ptr::null()) };
            }
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::QueueFetchFailed));
        };
        let mut queue: vk::VkQueue = std::ptr::null_mut();
        unsafe { get_device_queue(device, graphics_queue_family, 0, &mut queue) };
        if queue.is_null() {
            if let Some(destroy_device) = device_fns.destroy_device {
                unsafe { destroy_device(device, std::ptr::null()) };
            }
            unsafe { destroy_instance(instance, std::ptr::null()) };
            return Err((stage, ProbeFailure::QueueFetchFailed));
        }

        Ok(VulkanContext {
            instance,
            physical_device,
            device,
            queue,
            graphics_queue_family,
            device_count: actual,
            device_name,
            instance_fns,
            device_fns,
            _loader: loader,
        })
    }
}

// ---------------------------------------------------------------------------
// 可用性自检：完整 compute dispatch 路径（真跑一次，不靠推测）
// ---------------------------------------------------------------------------

/// 在已建立的上下文中真跑一遍 mul2 探针，并与 CPU 参考**逐位**比对。
///
/// 这是 [`VulkanBackend::compute_available`] 的唯一判据：实例/设备探测成功
/// 只说明「有设备」，本函数才回答「这条 dispatch 路径真能用」。任一环节
/// 失败（建管线/提交/读回）或读回值不符 ⇒ `false`，**不 panic**。
#[cfg(not(target_arch = "wasm32"))]
fn verify_compute_path(ctx: &VulkanContext) -> bool {
    let kernel = crate::compute::spirv_probe_kernel();
    // 64 个值 = 1 个工作组（与探针 LocalSize 64 对齐，无尾组浪费）。
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let report = compute::dispatch(Some(ctx), &kernel, [input.len() as u32, 1, 1], &input);
    if !report.is_ok() || report.output.len() < input.len() {
        return false;
    }
    let mut expected = vec![0.0f32; input.len()];
    spirv::cpu_reference_mul2(&input, &mut expected);
    expected
        .iter()
        .zip(report.output.iter())
        .all(|(e, g)| e.to_bits() == g.to_bits())
}

/// 在已建立的上下文中真跑一遍**场景帧探针**（4×4 单球），确认渲染路径可用。
///
/// 这是 [`VulkanBackend::render_available`] 的唯一判据。两条断言缺一不可：
/// 1. **读回长度自洽**：恰为 `4*4*4` 字节（绑定/布局/读回口径对得上）；
/// 2. **确有内容**：至少一个字节非零——`spv_scene` 会把未命中像素写成背景色
///    `(0.04, 0.05, 0.08)`（量化后 (10, 13, 20)），全零只可能意味着 kernel
///    根本没执行（或驱动返回了未写入的缓冲），那不是「能出图」。
///
/// 任一不满足 ⇒ `false`（上层回退 CPU），**不 panic**。
#[cfg(not(target_arch = "wasm32"))]
fn verify_render_path(ctx: &VulkanContext) -> bool {
    let req = scene::self_test_request();
    let expected = (req.width as usize) * (req.height as usize) * crate::BYTES_PER_PIXEL;
    let report = scene::dispatch(Some(ctx), &req);
    report.is_ok() && report.pixels.len() == expected && report.pixels.iter().any(|&b| b != 0)
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}
impl GpuDevice for VulkanBackend {
    /// **光追场景渲染**（V2.1 T-A1：从「恒空帧」改为真实出图）。
    ///
    /// 路径：[`scene::dispatch`]（4 绑定 compute megakernel → fence → 读回 RGBA8），
    /// 输出与 [`crate::CpuRefBackend::render_scene`] 同构算法（球面求交 + 直接光
    /// + Reinhard filmic，**全程不消费 ACES 矩阵**，故对拍基准不随 T4 移动）。
    ///
    /// # 诚实边界（三态，互不冒充）
    ///
    /// * **有设备且派发成功** ⇒ 返回 `is_valid()` 帧（`len == w*h*4`）；
    /// * **无设备 / wasm** ⇒ [`Framebuffer::empty`]（`is_valid() == false`），
    ///   调用方据此回退 [`crate::CpuRefBackend`]——**不返回全黑假帧**；
    /// * **尺寸被拒**（零轴、维度 > [`crate::MAX_FRAMEBUFFER_DIM`]、字节 >
    ///   [`crate::MAX_FRAMEBUFFER_BYTES`]）⇒ 原样回显尺寸的**拒绝缓冲**
    ///   （零分配、秒回，与 CPU 参考同一 `Framebuffer` 契约）。
    ///
    /// 本函数在任何平台、任何输入下都不 panic（C18）。
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer {
        // 先按 Framebuffer 的尺寸契约分配：被拒时立即返回（不发起 dispatch，
        // 也不按 u32::MAX² 空转）。
        let mut fb = Framebuffer::new(req.width, req.height);
        if !fb.is_valid() {
            return fb;
        }
        let report = self.render_scene_frame(req);
        if !report.is_ok() || report.pixels.len() != fb.pixels.len() {
            // 无设备 / 派发失败 / 读回长度不符：诚实空帧（调用方回退 CPU）。
            return Framebuffer::empty();
        }
        fb.pixels = report.pixels;
        fb
    }

    /// **如实**报告「**compute 可用 ∧ 场景渲染可用**」（Q4 裁定的双自检）。
    ///
    /// # 契约（V2.1 T-A1 起）
    ///
    /// 本值的语义是「**能否 service [`GpuDevice::render_scene`]**」：
    /// * [`Self::compute_available`]：mul2 探针的完整 dispatch 闭环真跑通过；
    /// * [`Self::render_available`]：`spv_scene` 真渲染一帧 4×4 并读回长度自洽。
    ///
    /// 两者都过才 `true`。这消灭了「`is_available()==true` 但 `render_scene`
    /// 出空帧」的组合：渲染路径一旦被驱动拒绝（管线不兼容 / 读回不符），
    /// 本标志立即塌成 `false`，上层据此回退 CPU。
    ///
    /// 无设备平台的每个分支都是 `false`（不 panic、不假装）。
    fn is_available(&self) -> bool {
        self.compute_available() && self.render_available()
    }

    /// 稳定标识：`vulkan-compute`（compute 路径 + 背景帧 + 光追场景帧均已真跑）。
    fn backend_name(&self) -> &'static str {
        "vulkan-compute"
    }
}

// ---------------------------------------------------------------------------
// 探测链中间态（非 wasm 专用）
// ---------------------------------------------------------------------------

/// 探测链已走到的最远状态（供 [`ProbeReport`] 还原失败环节）。
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default, Clone, Copy)]
struct ProbeStage {
    loader_loaded: bool,
    instance_created: bool,
    device_count: usize,
}

// ---------------------------------------------------------------------------
// 辅助：properties 承接缓冲的零初始化
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
impl loader::VkPhysicalDevicePropertiesStorage {
    /// 全零初始化（驱动会覆写；未写到的尾部字节保持 0，可安全读前缀）。
    fn zeroed() -> Self {
        // SAFETY: 结构体为 POD（全整型/字节数组），全零是合法位模式。
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// 单元测试（有 Vulkan 真跑真机路径，无 Vulkan 验证降级语义与探测链边界）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AestheticReq, CameraReq, LightReq, SphereReq};
    use prism_math::Vec3;

    /// 最小渲染请求，用于验证降级路径不 panic。
    fn tiny_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 8,
            height: 8,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                fov: 60.0_f32.to_radians(),
                aspect: 1.0,
            },
            lights: vec![LightReq {
                direction: Vec3::new(0.0, 0.0, -1.0),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 1.0,
            }],
            spheres: vec![SphereReq {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 1.0,
                albedo: Vec3::new(0.8, 0.2, 0.2),
                roughness: 0.4,
                metalness: 0.1,
                maps: crate::NO_MAPS,
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

    /// 后端标识跨平台稳定；可用性 = **compute 自检 ∧ 渲染自检**（Q4 裁定）。
    ///
    /// **非恒真自证**：无设备的机器断言三者全 `false`（诚实降级），有设备且两条
    /// 闭环都跑通的机器断言全 `true`（本沙箱的 lavapipe 即此分支）。两个分支都是
    /// 对当前环境真实能力的刻画——若把自检换成「探测到设备就算可用」，无 ICD
    /// 的机器上仍会红；若把渲染自检去掉，`is_available` 与 `render_available`
    /// 的一致性断言会红。
    #[test]
    fn backend_availability_follows_verified_compute_path() {
        let be = VulkanBackend::new();
        assert_eq!(be.backend_name(), "vulkan-compute", "后端标识须跨平台稳定");
        // 双自检：is_available ≡ compute ∧ render（不是任一单项，也不是「有设备」）。
        assert_eq!(
            be.is_available(),
            be.compute_available() && be.render_available(),
            "is_available 必须严格等同 compute + render 双自检"
        );
        assert!(
            !be.render_available() || be.probed_device(),
            "渲染自检不可能在无设备时为真（防御性自洽检查）"
        );
        if be.probed_device() {
            assert!(
                be.compute_available(),
                "有设备 ⇒ mul2 探针闭环必须跑通（否则 W3 闭环不成立）"
            );
            assert!(
                be.render_available(),
                "有设备 ⇒ 场景帧探针（4×4 单球）必须真跑通（否则 T-A1 闭环不成立）"
            );
            assert!(be.is_available(), "双自检通过 ⇒ 必须报可用");
            assert!(
                be.device_name().is_some_and(|n| !n.is_empty()),
                "可用却拿不到设备名 = 无法证明在哪个设备上跑过"
            );
        } else {
            assert!(!be.compute_available(), "无设备 ⇒ compute 自检必须为假");
            assert!(!be.render_available(), "无设备 ⇒ 渲染自检必须为假");
            assert!(!be.is_available(), "无设备 ⇒ 不得报可用");
            assert!(be.device_name().is_none(), "无设备 ⇒ 无设备名");
        }
    }

    /// **可用性语义闭合**：`render_scene` 的帧有效性与渲染自检**严格一致**。
    ///
    /// 禁止的组合是「`render_available() == true` 但 `render_scene` 出空帧」
    /// （PRD A1 验收 3）。两个分支都被真实执行到：
    /// * 自检通过（本沙箱 lavapipe）⇒ 必须出 `w*h*4` 有效帧；
    /// * 自检失败 / 无设备 ⇒ 必须返回无效空缓冲，**不得冒充成功**。
    #[test]
    fn render_scene_validity_matches_render_self_test() {
        let req = tiny_request();
        let mut be = VulkanBackend::new();
        let verified = be.render_available();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| be.render_scene(&req)));
        let fb = r.expect("Vulkan render_scene 不得 panic");
        assert_eq!(
            fb.is_valid(),
            verified,
            "帧有效性必须严格等同渲染自检结论（可用性与出图能力不得脱钩）"
        );
        if verified {
            assert_eq!(
                fb.pixels.len(),
                (req.width * req.height) as usize * 4,
                "有效帧必须铺满 w*h*4 字节"
            );
            assert!((fb.width, fb.height) == (req.width, req.height));
        } else {
            assert_eq!(fb.pixels.len(), 0, "空缓冲不含像素");
            assert_eq!((fb.width, fb.height), (0, 0), "empty() 为 0x0");
        }
    }

    /// 极端尺寸**秒回拒绝缓冲**（不 dispatch、不 panic、尺寸原样回显）。
    ///
    /// 与 `CpuRefBackend` 同一 `Framebuffer` 尺寸契约（A1 验收 4）。
    #[test]
    fn render_scene_refuses_extreme_sizes_without_panicking() {
        let mut be = VulkanBackend::new();
        for (w, h) in [(u32::MAX, u32::MAX), (0, 8), (8, 0), (65_536, 65_536)] {
            let mut req = tiny_request();
            req.width = w;
            req.height = h;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| be.render_scene(&req)));
            let fb = r.expect("超限尺寸不得 panic");
            assert!(!fb.is_valid(), "({w},{h}) 应被拒绝");
            assert!(fb.pixels.is_empty(), "被拒缓冲不含像素");
            assert_eq!((fb.width, fb.height), (w, h), "尺寸须原样回显");
        }
    }

    /// 反复构造 / 析构后端不 panic、不泄漏（`Drop` 释放 loader 与句柄）。
    ///
    /// 本测试同时是「`dlopen`/`dlclose` 配对」的烟雾证据：构造 N 次后仍能
    /// 正常返回（若句柄泄漏到耗尽则后续 `open` 会失败，但计数不变）。
    /// 每次构造后都问一次 `is_available()`：有设备时这会把
    /// 「建 instance → 建 device → 建/拆 compute 管线 → 提交 → 读回」
    /// 整套真机资源生命周期跑 64 遍，任何一处漏释放/二次释放都会暴露。
    #[test]
    fn repeated_construction_is_stable_and_leak_free() {
        for _ in 0..64 {
            let be = VulkanBackend::new();
            assert_eq!(
                be.is_available(),
                be.compute_available() && be.render_available(),
                "可用性口径（compute ∧ render）在反复构造后不得漂移"
            );
        }
    }

    /// **环境自证（非恒真）**：`is_available()` 必须与探测报告的**真实**状态
    /// 自洽，而不是恒真/恒假。
    ///
    /// 有设备 ⇒ `probed_device()` 为真且自检必须跑通（本沙箱 lavapipe 走此分支）；
    /// 无设备 ⇒ 二者同为假（诚实降级）。两条分支都不是空转断言。
    #[test]
    fn availability_is_consistent_with_probe_report() {
        let be = VulkanBackend::new();
        let report = VulkanBackend::probe();

        if report.device_created {
            assert!(
                be.probed_device(),
                "probe() 报设备已创建，则 probed_device() 必须为 true"
            );
            assert!(
                be.compute_available() && be.render_available(),
                "设备已创建且两条自检都跑通 ⇒ 双自检必须为 true"
            );
            assert!(
                be.is_available(),
                "设备已创建且双自检跑通 ⇒ is_available() 须为 true"
            );
            assert!(
                be.device_name().is_some_and(|n| !n.is_empty()),
                "设备已创建 ⇒ 必须有设备名"
            );
        } else {
            assert!(
                !be.probed_device(),
                "probe() 报不可用，则 probed_device() 不得为 true"
            );
            assert!(
                !be.is_available(),
                "无设备 ⇒ is_available() 不得为 true"
            );
        }
    }

    /// `probe()` 报告的字段必须自洽（后续阶段为真 ⇒ 前序阶段为真）。
    ///
    /// **本测试在无可用设备的平台与有 GPU 的平台上均成立且有区分度**：
    /// 前者走 `device_count == 0` 分支，后者走 `device_created == true` 分支。
    #[test]
    fn probe_report_fields_are_monotonic() {
        let report = VulkanBackend::probe();
        if report.instance_created {
            assert!(report.loader_loaded, "instance 已建 ⇒ loader 必已加载");
        }
        if report.device_created {
            assert!(report.instance_created, "device 已建 ⇒ instance 必已建");
            assert!(report.device_count > 0, "device 已建 ⇒ 必有物理设备");
            assert!(report.reason.is_none(), "成功时不应有失败原因");
        }
    }

    /// 探测失败原因字符串稳定且非空（诊断日志契约）。
    #[test]
    fn probe_failure_names_are_stable() {
        for reason in [
            ProbeFailure::UnsupportedPlatform,
            ProbeFailure::LoaderLoadFailed,
            ProbeFailure::GetInstanceProcAddrMissing,
            ProbeFailure::CreateInstanceMissing,
            ProbeFailure::InstanceCreationFailed,
            ProbeFailure::NoPhysicalDevice,
            ProbeFailure::NoGraphicsQueueFamily,
            ProbeFailure::DeviceCreationFailed,
            ProbeFailure::QueueFetchFailed,
        ] {
            assert!(!reason.as_str().is_empty(), "失败原因短名不得为空");
        }
        assert_eq!(ProbeFailure::LoaderLoadFailed.as_str(), "loader-load-failed");
    }

    /// `Default` 与 `new()` 行为一致（均不 panic、能力口径相同）。
    #[test]
    fn default_matches_new() {
        let by_default = VulkanBackend::default();
        let by_new = VulkanBackend::new();
        assert_eq!(by_default.backend_name(), by_new.backend_name());
        assert_eq!(by_default.is_available(), by_new.is_available());
        assert_eq!(by_default.device_name(), by_new.device_name());
    }

    /// **加载器边界（非 wasm）**：`loader::open()` 对不存在的库名返回 `None`。
    ///
    /// 通过间接证据刻画：`VulkanLoader` 的 `open` 只可能返回「有效句柄」或
    /// `None`。此处断言「若 open 成功，则必然能取到 `vkGetInstanceProcAddr`」
    /// ——这是 loader 库的固有契约；而若当前平台无 `libvulkan`，`open` 返回
    /// `None`，分支被跳过。两种情形都有意义。
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn loader_open_contract_holds_or_is_absent() {
        match loader::open() {
            Some(l) => {
                // 成功打开 ⇒ `vkGetInstanceProcAddr` 必须可得（loader 的导出表
                // 里必含该符号）。若取不到，说明库异常，本断言会如实抓出。
                assert!(
                    unsafe { l.load_get_instance_proc_addr() }.is_some(),
                    "libvulkan 已打开但缺 vkGetInstanceProcAddr —— 库异常"
                );
                // l 在此 drop → dlclose，验证析构不 panic。
            }
            None => {
                // 当前平台无 loader：跳过（不写 `assert!(false)` 之类恒真式）。
            }
        }
    }
}
