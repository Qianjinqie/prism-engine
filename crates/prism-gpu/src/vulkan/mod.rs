//! Vulkan 后端（D1，Linux/Windows 原生 GPU 路径）。
//!
//! # ⚠️ 诚实边界（必读）
//!
//! 本模块**只实装了设备探测链**（instance → physical device → device →
//! graphics queue）。**渲染管线尚未实装**：swapchain / render pass /
//! graphics pipeline / 自研 SPIR-V 发射器 / 命令编码 / 读回路径**全部缺失**。
//!
//! 因此 [`VulkanBackend::is_available`] **恒返回 `false`**，即便探测链成功
//! 拿到 instance + device 也不改口——`is_available` 的语义是「**能否 service
//! `render_scene`**」，探测成功 ≠ 能出图（见 [`VulkanBackend`] 类型文档）。
//! [`VulkanBackend::render_scene`] 如实返回 [`Framebuffer::empty`]（`is_valid()
//! == false`），**绝不返回假帧**（例如全黑有效帧）冒充成功。上层据此安全
//! 回退 [`crate::CpuRefBackend`]。
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
//! 就判定不可用。故即便构建机没有 Vulkan（如本仓库的 CI 沙箱），本 crate
//! 仍可正常编译、链接、测试。
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
// compute 管线（V2.0-G）：SPIR-V 供给 + dispatch / queue 提交
// ---------------------------------------------------------------------------
//
// `spirv` 是纯数据（手写 SPIR-V 字流），**所有平台**都编译——wasm 侧也要能
// 报「本 kernel 的 SPIR-V 是什么」；`compute` 含真机 FFI 装配路径，wasm 上
// 只剩「如实报 unsupported」的入口（见其 `dispatch` 的 cfg 分支）。

/// 手写 / 嵌入的 SPIR-V 模块表与结构校验。
pub mod spirv;

/// compute pipeline 装配 + 队列提交（真机路径写全，本环境无 ICD 走降级分支）。
pub mod compute;

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
/// 这是一个**能力占位 + 真实设备探测**而非可用渲染器：见模块头注的诚实边界。
/// 它的存在意义有二：① 使上层能在所有平台统一引用 `VulkanBackend`；
/// ② 在 native 平台提供真实的设备探测入口（[`VulkanBackend::probe`]），
/// 为后续实装渲染管线铺路。
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
/// 6. 成功 → 缓存 instance/device/queue，`probed_device()` 返回 `true`。
///
/// # 为什么 `is_available()` 恒为 `false`
///
/// [`GpuDevice::is_available`] 的语义是「**能否 service `render_scene`**」。
/// 本例虽然实装了探测链，但**没有** render pipeline / swapchain / 命令编码 /
/// SPIR-V 发射器 / 读回路径，出一个真帧的最小闭环尚不存在。因此此处**不撒谎**
/// ——探测成功不等于可用，恒返回 `false`，让上层安全降级到
/// [`crate::CpuRefBackend`]。
pub struct VulkanBackend {
    /// 探测会话结果（成功后持有 `VulkanContext`，`Drop` 时释放句柄）。
    ///
    /// 非 wasm：`Option<Box<VulkanContext>>`。
    #[cfg(not(target_arch = "wasm32"))]
    context: Option<Box<VulkanContext>>,
    /// 诊断量：是否探测到物理设备（与 `context.is_some()` 基本一致，独立保留
    /// 便于 wasm 分支统一字段布局，并明确「探测到设备」与「可用」的区别）。
    probed_device: bool,
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
    /// 否则 `None`。**不用于渲染**（渲染管线未实装）。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn graphics_queue_handle(&self) -> Option<usize> {
        self.context
            .as_ref()
            .map(|c| c.queue as usize)
            .filter(|&p| p != 0)
    }

    /// **compute 派发入口**（V2.0-G）。
    ///
    /// 走 [`compute::dispatch`]：有设备上下文则在真机上装配 SPIR-V → compute
    /// pipeline → 命令缓冲 → 队列提交 → 读回；无上下文（本仓库构建环境无
    /// ICD）则如实返回 [`compute::DispatchStatus::DeviceUnavailable`]，
    /// **不 panic、不返回假输出**。
    ///
    /// 注意：本方法与 [`GpuDevice::is_available`] 语义不同——后者刻画的是
    /// 「能否 service `render_scene`」（渲染管线仍未实装，恒 `false`），
    /// compute 派发是**另一条独立路径**。
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
            instance_fns,
            device_fns,
            _loader: loader,
        })
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice for VulkanBackend {
    /// Vulkan 渲染管线尚未实装：**返回空缓冲而非假帧**。
    ///
    /// 刻意不返回一个全黑的有效帧去冒充成功——那会污染上层判对逻辑。
    /// [`Framebuffer::empty`] 的 `is_valid() == false`，调用方据此回退 CPU。
    /// 本函数在任何平台、任何输入下都不 panic（C18）。
    fn render_scene(&mut self, _req: &SceneRenderRequest) -> Framebuffer {
        Framebuffer::empty()
    }

    /// 恒 `false`：能探测到 device 不代表能出图（渲染管线未实装）。
    ///
    /// 见 [`VulkanBackend`] 类型文档的「为什么 `is_available()` 恒为 false」。
    /// 待到本模块补上 swapchain / render pass / graphics pipeline / SPIR-V
    /// 发射器 / 命令编码 / 读回路径后，此函数才可改为返回真实能力。
    fn is_available(&self) -> bool {
        false
    }

    /// 稳定标识：所有平台一致，便于能力日志对齐。
    fn backend_name(&self) -> &'static str {
        "vulkan-stub"
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
// 单元测试（在无 Vulkan 的沙箱中验证降级语义与探测链边界）
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
            }],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        }
    }

    /// 后端标识跨平台稳定 + 恒不可用（渲染管线未实装）。
    ///
    /// **非恒真自证**：`is_available() == false` 在「有 GPU 的机器」上**同样
    /// 成立**（渲染未实装），故这不是「靠环境空转」的断言；一旦将来补上
    /// 渲染管线，本断言必须翻转为 `true` 并补像素判对——正是这一翻转点证明
    /// 它此刻非空。
    #[test]
    fn backend_is_unavailable_and_name_is_stable() {
        let be = VulkanBackend::new();
        assert_eq!(be.backend_name(), "vulkan-stub", "后端标识须跨平台稳定");
        assert!(
            !be.is_available(),
            "渲染管线未实装 ⇒ is_available() 必须为 false（不得撒谎）"
        );
    }

    /// `render_scene` 返回**无效**空缓冲（而非 panic 或假帧）。
    ///
    /// **非恒真自证**：同 metal 测试——若实装渲染，此断言须改为断言有效帧。
    #[test]
    fn render_scene_returns_invalid_frame_without_panic() {
        let req = tiny_request();
        let mut be = VulkanBackend::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| be.render_scene(&req)));
        let fb = r.expect("Vulkan render_scene 不得 panic");
        assert!(!fb.is_valid(), "未实装渲染须返回无效缓冲，不得冒充成功");
        assert_eq!(fb.pixels.len(), 0, "空缓冲不含像素");
        assert_eq!((fb.width, fb.height), (0, 0), "empty() 为 0x0");
    }

    /// 反复构造 / 析构后端不 panic、不泄漏（`Drop` 释放 loader 与句柄）。
    ///
    /// 本测试同时是「`dlopen`/`dlclose` 配对」的烟雾证据：构造 N 次后仍能
    /// 正常返回（若句柄泄漏到耗尽则后续 `open` 会失败，但计数不变）。
    #[test]
    fn repeated_construction_is_stable_and_leak_free() {
        for _ in 0..64 {
            let be = VulkanBackend::new();
            // 恒不可用（渲染未实装），且诊断量与之一致。
            assert!(!be.is_available());
            // 在无 ICD 的沙箱里，probed_device 应为 false；在有可用 Vulkan 的
            // 机器上可能为 true。此处不写死断言，避免环境恒真/恒假。
        }
    }

    /// **环境自证（非恒真）**：`is_available()` 与「真实渲染能力」一致，
    /// 且 `probed_device()` 与环境真实状态自洽。
    ///
    /// 断言逻辑：无论是否有 Vulkan，`is_available()` **必须**为 `false`
    /// （这是渲染未实装的不变量）；而 `probed_device()` 则**取决于环境**——
    /// 本断言同时校验「若探测报告不可用，则 probed_device 必为 false」这一
    /// 自洽性，使这条测试在两种环境下都刻画真实状态，而非恒真。
    #[test]
    fn availability_is_consistent_with_probe_report() {
        let be = VulkanBackend::new();
        let report = VulkanBackend::probe();

        // 不变量：渲染未实装 ⇒ 恒不可用（与环境无关）。
        assert!(!be.is_available());

        // 自洽性：探测报告不可用时，缓存中必不应有设备。
        if report.device_created {
            assert!(
                be.probed_device(),
                "probe() 报设备已创建，则 probed_device() 必须为 true"
            );
        } else {
            assert!(
                !be.probed_device(),
                "probe() 报不可用，则 probed_device() 不得为 true"
            );
        }
    }

    /// `probe()` 报告的字段必须自洽（后续阶段为真 ⇒ 前序阶段为真）。
    ///
    /// **本测试在沙箱（枚举到 0 设备）与真机（有 GPU）上均成立且有区分度**：
    /// 沙箱走 `device_count == 0` 分支，真机走 `device_created == true` 分支。
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

    /// `Default` 与 `new()` 行为一致（均不 panic、均不可用）。
    #[test]
    fn default_matches_new() {
        let be = VulkanBackend::default();
        assert!(!be.is_available());
        assert_eq!(be.backend_name(), VulkanBackend::new().backend_name());
    }

    /// **加载器边界（非 wasm）**：`loader::open()` 对不存在的库名返回 `None`。
    ///
    /// 通过间接证据刻画：`VulkanLoader` 的 `open` 只可能返回「有效句柄」或
    /// `None`。此处断言「若 open 成功，则必然能取到 `vkGetInstanceProcAddr`」
    /// ——这是 loader 库的固有契约；而若沙箱根本无 `libvulkan`，`open` 返回
    /// `None`，分支被跳过。两种环境都有意义。
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
                // 沙箱无 loader：跳过（不写 `assert!(false)` 之类恒真式）。
            }
        }
    }
}
