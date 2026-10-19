//! Vulkan 加载器 + 函数指针表（手写 FFI，**零第三方 crate** —— D-15）。
//!
//! # 为什么手写 `dlopen` / `dlsym`
//!
//! Rust 标准库**不提供**运行时动态加载封装（`libloading` 是第三方 crate，
//! 铁律禁用）。故本模块直接用 `#[link(name = "dl")]` 声明系统 `libdl` 的
//! 三个入口：`dlopen` / `dlsym` / `dlclose`。`libdl` 是操作系统原生组件
//! （`libc` 的一部分），调用它属于「调用平台 API」，不构成第三方依赖。
//!
//! # 编译期不依赖 `libvulkan`
//!
//! **本模块编译期不链接任何 Vulkan 库**：所有 Vulkan 符号都在运行期经
//! `dlopen("libvulkan.so.1")` + `dlsym` 取得。因此即便构建机上完全没有
//! `libvulkan.so`，本 crate 依旧可以正常编译、链接、测试。
//!
//! # 平台门禁
//!
//! 整模块仅在 `not(target_arch = "wasm32")` 下编译。wasm32 既无 `libdl`
//! 也无 `libvulkan`，故 [`super`] 在 wasm 上提供等价空实现（恒不可用）。

use std::ffi::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// libdl FFI（系统库，非第三方 crate）
// ---------------------------------------------------------------------------

/// `RTLD_LAZY` —— glibc `<dlfcn.h>` 定义。
///
/// ⚠️ **平台常量，不可臆测**：POSIX 只规定 `RTLD_LAZY` / `RTLD_NOW` 存在，
/// **不规定其数值**。Linux/glibc 上分别为 `1` / `2`；macOS 上 `RTLD_LAZY`
/// 也是 `0x1`。本模块的目标平台是 Linux（Vulkan 原生后端），故取 glibc 值。
/// 若将来移植 macOS，此常量须复核（macOS 的 Vulkan 走 MoltenVK 且路径不同）。
///
/// 选 `RTLD_LAZY`（而非 `RTLD_NOW`）的理由：`libvulkan.so.1` 体量大且依赖
/// ICD 加载器；`LAZY` 推迟未用符号重定位，探测失败时的开销更低。
const RTLD_LAZY: c_int = 1;

#[link(name = "dl")]
extern "C" {
    /// `void *dlopen(const char *filename, int flag);`
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    /// `void *dlsym(void *handle, const char *symbol);`
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    /// `int dlclose(void *handle);`
    fn dlclose(handle: *mut c_void) -> c_int;
}

// ---------------------------------------------------------------------------
// Vulkan 基础类型（与 `<vulkan/vulkan_core.h>` 逐字段对齐）
// ---------------------------------------------------------------------------

/// `VkResult`：`0`（`VK_SUCCESS`）表示成功，负值为错误码。
///
/// 本模块只关心 `VK_SUCCESS`，其余一律按「失败」处理（错误码不逐一分派，
/// 因为探测链上任一失败都收敛到「不可用」）。
pub type VkResult = i32;

/// `VK_SUCCESS`（Vulkan 头文件定义）。
pub const VK_SUCCESS: VkResult = 0;

/// 不透明 Vulkan 句柄（`dispatchable` 与非 `dispatchable` 在 C 层都是指针）。
pub type VkInstance = *mut c_void;
/// `VkPhysicalDevice`。
pub type VkPhysicalDevice = *mut c_void;
/// `VkDevice`。
pub type VkDevice = *mut c_void;
/// `VkQueue`。
pub type VkQueue = *mut c_void;
/// `VkQueueFamilyProperties` 的数量类型（`uint32_t`）。
pub type U32 = u32;

/// `VkStructureType` 中本次用到的取值（`<vulkan/vulkan_core.h>`）。
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: i32 = 0;
/// `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: i32 = 1;
/// `VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: i32 = 2;
/// `VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: i32 = 3;

/// `VK_API_VERSION_1_0` = `(major<<22) | (minor<<12) | 0` = `1.0.0`。
///
/// 写成 `(1 << 22) | (0 << 12)` 会触发 clippy `identity_op`；此处直接给出
/// 打包值 `1 << 22`，并把完整打包公式留在文档里（major=1, minor=0, patch=0）。
pub const VK_API_VERSION_1_0: u32 = 1 << 22;

/// `VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT` = `0x00000001`。
pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x0000_0001;

/// `VkPhysicalDeviceType` —— 本模块用于「偏好离散 GPU」的排序权重。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalDeviceType {
    /// `VK_PHYSICAL_DEVICE_TYPE_OTHER`。
    Other,
    /// `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`。
    IntegratedGpu,
    /// `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`。
    DiscreteGpu,
    /// `VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU`。
    VirtualGpu,
    /// `VK_PHYSICAL_DEVICE_TYPE_CPU`。
    Cpu,
    /// 未识别的枚举值（保留原码，不 panic）。
    Unknown(i32),
}

impl PhysicalDeviceType {
    /// 从 `VkPhysicalDeviceProperties::deviceType` 原始码映射。
    ///
    /// 未识别码返回 [`PhysicalDeviceType::Unknown`]，**不 panic**（C18）。
    fn from_raw(code: i32) -> Self {
        match code {
            0 => PhysicalDeviceType::Other,
            1 => PhysicalDeviceType::IntegratedGpu,
            2 => PhysicalDeviceType::DiscreteGpu,
            3 => PhysicalDeviceType::VirtualGpu,
            4 => PhysicalDeviceType::Cpu,
            other => PhysicalDeviceType::Unknown(other),
        }
    }

    /// 选择偏好权重（越大越优先）。离散 GPU > 虚拟 GPU > 集成 GPU > 其他。
    ///
    /// 与常见 Vulkan 样例一致：优先独立显卡以获得最大吞吐。
    fn preference_weight(self) -> u8 {
        match self {
            PhysicalDeviceType::DiscreteGpu => 4,
            PhysicalDeviceType::VirtualGpu => 3,
            PhysicalDeviceType::IntegratedGpu => 2,
            PhysicalDeviceType::Cpu => 1,
            PhysicalDeviceType::Other | PhysicalDeviceType::Unknown(_) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan `struct` 定义（`#[repr(C)]`，字段顺序/类型与头文件逐一对齐）
// ---------------------------------------------------------------------------

/// `VkApplicationInfo`。
///
/// 字段布局对齐 `vulkan_core.h`（64 位 ABI）：
/// `sType, pNext, pApplicationName, applicationVersion, pEngineName,
///  engineVersion, apiVersion`。
#[repr(C)]
pub(crate) struct VkApplicationInfo {
    pub(crate) s_type: i32,
    pub(crate) p_next: *const c_void,
    pub(crate) p_application_name: *const c_char,
    pub(crate) application_version: u32,
    pub(crate) p_engine_name: *const c_char,
    pub(crate) engine_version: u32,
    pub(crate) api_version: u32,
}

/// `VkInstanceCreateInfo`。
#[repr(C)]
pub(crate) struct VkInstanceCreateInfo {
    pub(crate) s_type: i32,
    pub(crate) p_next: *const c_void,
    pub(crate) flags: u32,
    pub(crate) p_application_info: *const VkApplicationInfo,
    pub(crate) enabled_layer_count: u32,
    pub(crate) pp_enabled_layer_names: *const *const c_char,
    pub(crate) enabled_extension_count: u32,
    pub(crate) pp_enabled_extension_names: *const *const c_char,
}

/// `VkQueueFamilyProperties`。
///
/// 布局：`VkQueueFlags queueFlags; uint32_t queueCount; uint32_t timestampValidBits;
/// VkExtent3D minImageTransferGranularity;`（`VkExtent3D` = 3×`uint32_t`）。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VkQueueFamilyProperties {
    /// `VkQueueFlags` 位掩码。
    pub queue_flags: u32,
    /// 该队列族支持的队列数量。
    pub queue_count: u32,
    /// 时间戳有效位。
    pub timestamp_valid_bits: u32,
    /// 最小图像传输粒度（`VkExtent3D`）。
    pub min_image_transfer_granularity: [u32; 3],
}

/// `VkDeviceQueueCreateInfo`。
#[repr(C)]
pub(crate) struct VkDeviceQueueCreateInfo {
    pub(crate) s_type: i32,
    pub(crate) p_next: *const c_void,
    pub(crate) flags: u32,
    pub(crate) queue_family_index: u32,
    pub(crate) queue_count: u32,
    pub(crate) p_queue_priorities: *const f32,
}

/// `VkDeviceCreateInfo`。
#[repr(C)]
pub(crate) struct VkDeviceCreateInfo {
    pub(crate) s_type: i32,
    pub(crate) p_next: *const c_void,
    pub(crate) flags: u32,
    pub(crate) queue_create_info_count: u32,
    pub(crate) p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub(crate) enabled_layer_count: u32,
    pub(crate) pp_enabled_layer_names: *const *const c_char,
    pub(crate) enabled_extension_count: u32,
    pub(crate) pp_enabled_extension_names: *const *const c_char,
    pub(crate) p_enabled_features: *const c_void,
}

/// `VkPhysicalDeviceProperties` 的**前缀**。
///
/// ⚠️ 完整结构体尾部还有 `limits` / `sparseProperties` 等大量字段。本模块
/// **只读前若干字段**（`deviceType` / `deviceName`），因此声明一个前缀切面
/// 即可——但**尾部切片必须原样保留其内存**：我们用一个足够大的字节缓冲
/// 承接驱动写回的整块结构，再从前缀读出所需字段（见
/// [`physical_device_type`]）。这样既避免手写全部 `limits` 字段，又不越界。
///
/// `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE` = 256。
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;

/// 前缀字段（`apiVersion, driverVersion, vendorID, deviceID, deviceType,
/// deviceName[256], pipelineCacheUUID[16]` 之后还有 `limits`…）。
#[repr(C)]
pub(crate) struct VkPhysicalDevicePropertiesPrefix {
    pub(crate) api_version: u32,
    pub(crate) driver_version: u32,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) device_type: i32,
    pub(crate) device_name: [u8; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub(crate) pipeline_cache_uuid: [u8; 16],
}

/// `VkPhysicalDeviceProperties` 的保守总尺寸上界（字节）。
///
/// `limits` 是约 50 个字段、多为 `u32`/`f32`/`u64` 的结构；连同前缀，
/// `sizeof` 实测约 800 字节量级。取 4096 字节（对齐到 `VkPhysicalDeviceProperties`
/// 的 8 字节对齐）作为承接缓冲，**远超**实际需要，确保驱动写回不越界。
/// 缓冲以 `#[repr(C)]` 结构体 + 显式 `_pad` 承载，并 `align(8)` 保证对齐。
#[repr(C, align(8))]
pub(crate) struct VkPhysicalDevicePropertiesStorage {
    pub(crate) prefix: VkPhysicalDevicePropertiesPrefix,
    /// 尾部余量（`limits` / `sparseProperties` 等），内容不解释。
    pub(crate) tail: [u8; 4096 - std::mem::size_of::<VkPhysicalDevicePropertiesPrefix>()],
}

// ---------------------------------------------------------------------------
// Vulkan 函数指针类型（从 loader 直接取，或经 vkGetInstanceProcAddr 取）
// ---------------------------------------------------------------------------

/// `PFN_vkGetInstanceProcAddr`：
/// `PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance, const char*);`
pub(crate) type PfnVkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const c_char) -> *const c_void;

/// `PFN_vkCreateInstance`：
/// `VkResult vkCreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*);`
pub(crate) type PfnVkCreateInstance = unsafe extern "system" fn(
    *const VkInstanceCreateInfo,
    *const c_void,
    *mut VkInstance,
) -> VkResult;

/// `PFN_vkDestroyInstance`：`void vkDestroyInstance(VkInstance, const VkAllocationCallbacks*);`
pub(crate) type PfnVkDestroyInstance = unsafe extern "system" fn(VkInstance, *const c_void);

/// `PFN_vkEnumeratePhysicalDevices`：
/// `VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*);`
pub(crate) type PfnVkEnumeratePhysicalDevices =
    unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;

/// `PFN_vkGetPhysicalDeviceProperties`：
/// `void vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties*);`
pub(crate) type PfnVkGetPhysicalDeviceProperties = unsafe extern "system" fn(
    VkPhysicalDevice,
    *mut VkPhysicalDevicePropertiesStorage,
);

/// `PFN_vkGetPhysicalDeviceQueueFamilyProperties`：
/// `void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);`
pub(crate) type PfnVkGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);

/// `PFN_vkCreateDevice`：
/// `VkResult vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice*);`
pub(crate) type PfnVkCreateDevice = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> VkResult;

/// `PFN_vkDestroyDevice`：`void vkDestroyDevice(VkDevice, const VkAllocationCallbacks*);`
pub(crate) type PfnVkDestroyDevice = unsafe extern "system" fn(VkDevice, *const c_void);

/// `PFN_vkGetDeviceQueue`：
/// `void vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue*);`
pub(crate) type PfnVkGetDeviceQueue = unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue);

// ---------------------------------------------------------------------------
// 加载器句柄 + 函数指针表
// ---------------------------------------------------------------------------

/// `libvulkan` 动态库句柄，`Drop` 时 `dlclose`。
///
/// 只负责「库句柄的打开/关闭」。符号绑定由 [`VulkanFnTable`] 承担，
/// 二者组合成一次可用的探测会话。
pub(crate) struct VulkanLoader {
    /// `dlopen` 返回的句柄；构造成功时保证非空。
    handle: *mut c_void,
}

/// 尝试打开 Vulkan loader 共享库。
///
/// 依次尝试 `libvulkan.so.1`（带 SONAME 版本，绝大多数发行版的标准名）
/// 与 `libvulkan.so`（开发符号链接，仅装了 `libvulkan-dev` 时才有）。
/// 两者皆失败返回 `None`。**不 panic**。
pub(crate) fn open() -> Option<VulkanLoader> {
    // 候选库名。用 Rust 内建 `c"..."` C 字符串字面量（`std::ffi::CStr` 家族，
    // 非第三方 crate），天然以 NUL 结尾且类型即 `*const c_char`。
    const CANDIDATES: [&core::ffi::CStr; 2] = [c"libvulkan.so.1", c"libvulkan.so"];
    for name in CANDIDATES {
        // SAFETY: `name` 合法 C 字符串常量；`dlopen` 对失败返回空指针，我们
        // 随即判空，不做任何解引用。
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_LAZY) };
        if !handle.is_null() {
            return Some(VulkanLoader { handle });
        }
    }
    None
}

impl VulkanLoader {
    /// 取 `vkGetInstanceProcAddr`（实例创建前唯一的全局入口）。
    ///
    /// `vkGetInstanceProcAddr(instance = null, name)` 是 Vulkan 规定的
    /// 「在不持有实例时取得全局函数」方式；但 loader 共享库里该符号本身
    /// 也可直接 `dlsym`，本模块走后者（更简单，少一层调用）。
    ///
    /// # Safety
    /// 需要 `handle` 仍是有效库句柄（由类型不变量保证，`handle` 非空且
    /// 在 `self` 存活期间不 `dlclose`）。
    pub unsafe fn load_get_instance_proc_addr(&self) -> Option<PfnVkGetInstanceProcAddr> {
        load_symbol::<PfnVkGetInstanceProcAddr>(self.handle, c"vkGetInstanceProcAddr")
    }
}

impl Drop for VulkanLoader {
    /// 释放库句柄。`dlclose` 失败仅表示引用计数非零（其他模块仍在用），
    /// 对本进程无副作用，忽略返回值即可——**不 panic**。
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `handle` 来自 `dlopen`，只在此处关闭一次；`self` 即将析构，
            // 不会再有调用方使用它。忽略返回码（见上）。
            let _ = unsafe { dlclose(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// 通用符号解析：`dlsym` → 空检 → `transmute_copy` 成 `F`。
///
/// # Safety
/// 调用方须保证 `handle` 有效，且 `name` 指向符号的真实 ABI 与 `F` 匹配。
unsafe fn load_symbol<F: Copy>(handle: *mut c_void, name: &core::ffi::CStr) -> Option<F> {
    let sym = dlsym(handle, name.as_ptr());
    if sym.is_null() {
        return None;
    }
    // 函数指针与数据指针在目标 ABI 上同宽；`transmute_copy` 避免
    // `mem::transmute` 对「目标类型大小未知」的编译期断言。
    Some(std::mem::transmute_copy::<*mut c_void, F>(&sym))
}

// ---------------------------------------------------------------------------
// 实例级函数指针表
// ---------------------------------------------------------------------------

/// 从实例（或 loader）取到的 Vulkan 函数指针集合。
///
/// 全部经 `vkGetInstanceProcAddr` 取得；任取返回空则对应字段为 `None`，
/// 探测链在下一步判空后优雅失败。
pub(crate) struct InstanceFnTable {
    /// `vkDestroyInstance`。
    pub destroy_instance: Option<PfnVkDestroyInstance>,
    /// `vkEnumeratePhysicalDevices`。
    pub enumerate_physical_devices: Option<PfnVkEnumeratePhysicalDevices>,
    /// `vkGetPhysicalDeviceProperties`。
    pub get_physical_device_properties: Option<PfnVkGetPhysicalDeviceProperties>,
    /// `vkGetPhysicalDeviceQueueFamilyProperties`。
    pub get_queue_family_properties: Option<PfnVkGetPhysicalDeviceQueueFamilyProperties>,
    /// `vkCreateDevice`。
    pub create_device: Option<PfnVkCreateDevice>,
    /// `vkGetPhysicalDeviceMemoryProperties`（compute 管线选 host 可见内存用）。
    pub get_physical_device_memory_properties: Option<PfnVkGetPhysicalDeviceMemoryProperties>,
}

impl InstanceFnTable {
    /// 用 `get_proc` 逐个绑定实例级入口。
    ///
    /// `instance` 为 `null` 时取「全局」函数（Vulkan 规范允许对部分名称这样做；
    /// 这些名称恰好都是本表所需）。
    ///
    /// # Safety
    /// 调用方须保证 `get_proc` 有效且 `instance` 与之匹配。
    pub unsafe fn bind_all(
        get_proc: PfnVkGetInstanceProcAddr,
        instance: VkInstance,
    ) -> InstanceFnTable {
        // 辅助：取符号并空检，空则 `None`（`transmute_copy` 前的空指针会被
        // 后续调用方判空拦截）。`$name` 为 `c"..."` C 字符串字面量。
        macro_rules! bind {
            ($name:literal, $ty:ty) => {{
                let sym = get_proc(instance, $name.as_ptr());
                if sym.is_null() {
                    None
                } else {
                    Some(std::mem::transmute_copy::<*const c_void, $ty>(&sym))
                }
            }};
        }
        InstanceFnTable {
            destroy_instance: bind!(c"vkDestroyInstance", PfnVkDestroyInstance),
            enumerate_physical_devices: bind!(
                c"vkEnumeratePhysicalDevices",
                PfnVkEnumeratePhysicalDevices
            ),
            get_physical_device_properties: bind!(
                c"vkGetPhysicalDeviceProperties",
                PfnVkGetPhysicalDeviceProperties
            ),
            get_queue_family_properties: bind!(
                c"vkGetPhysicalDeviceQueueFamilyProperties",
                PfnVkGetPhysicalDeviceQueueFamilyProperties
            ),
            create_device: bind!(c"vkCreateDevice", PfnVkCreateDevice),
            get_physical_device_memory_properties: bind!(
                c"vkGetPhysicalDeviceMemoryProperties",
                PfnVkGetPhysicalDeviceMemoryProperties
            ),
        }
    }
}

/// 从设备取到的函数指针集合。
pub(crate) struct DeviceFnTable {
    /// `vkDestroyDevice`。
    pub destroy_device: Option<PfnVkDestroyDevice>,
    /// `vkGetDeviceQueue`。
    pub get_device_queue: Option<PfnVkGetDeviceQueue>,
    /// `vkCreateShaderModule`（SPIR-V 载入）。
    pub create_shader_module: Option<PfnVkCreateShaderModule>,
    /// `vkDestroyShaderModule`。
    pub destroy_shader_module: Option<PfnVkDestroyShaderModule>,
    /// `vkCreateDescriptorSetLayout`。
    pub create_descriptor_set_layout: Option<PfnVkCreateDescriptorSetLayout>,
    /// `vkDestroyDescriptorSetLayout`。
    pub destroy_descriptor_set_layout: Option<PfnVkDestroyDescriptorSetLayout>,
    /// `vkCreatePipelineLayout`。
    pub create_pipeline_layout: Option<PfnVkCreatePipelineLayout>,
    /// `vkDestroyPipelineLayout`。
    pub destroy_pipeline_layout: Option<PfnVkDestroyPipelineLayout>,
    /// `vkCreateComputePipelines`。
    pub create_compute_pipelines: Option<PfnVkCreateComputePipelines>,
    /// `vkDestroyPipeline`。
    pub destroy_pipeline: Option<PfnVkDestroyPipeline>,
    /// `vkCreateBuffer`。
    pub create_buffer: Option<PfnVkCreateBuffer>,
    /// `vkDestroyBuffer`。
    pub destroy_buffer: Option<PfnVkDestroyBuffer>,
    /// `vkGetBufferMemoryRequirements`。
    pub get_buffer_memory_requirements: Option<PfnVkGetBufferMemoryRequirements>,
    /// `vkAllocateMemory`。
    pub allocate_memory: Option<PfnVkAllocateMemory>,
    /// `vkFreeMemory`。
    pub free_memory: Option<PfnVkFreeMemory>,
    /// `vkBindBufferMemory`。
    pub bind_buffer_memory: Option<PfnVkBindBufferMemory>,
    /// `vkCreateDescriptorPool`。
    pub create_descriptor_pool: Option<PfnVkCreateDescriptorPool>,
    /// `vkDestroyDescriptorPool`。
    pub destroy_descriptor_pool: Option<PfnVkDestroyDescriptorPool>,
    /// `vkAllocateDescriptorSets`。
    pub allocate_descriptor_sets: Option<PfnVkAllocateDescriptorSets>,
    /// `vkUpdateDescriptorSets`。
    pub update_descriptor_sets: Option<PfnVkUpdateDescriptorSets>,
    /// `vkCreateCommandPool`。
    pub create_command_pool: Option<PfnVkCreateCommandPool>,
    /// `vkDestroyCommandPool`。
    pub destroy_command_pool: Option<PfnVkDestroyCommandPool>,
    /// `vkAllocateCommandBuffers`。
    pub allocate_command_buffers: Option<PfnVkAllocateCommandBuffers>,
    /// `vkFreeCommandBuffers`。
    pub free_command_buffers: Option<PfnVkFreeCommandBuffers>,
    /// `vkBeginCommandBuffer`。
    pub begin_command_buffer: Option<PfnVkBeginCommandBuffer>,
    /// `vkEndCommandBuffer`。
    pub end_command_buffer: Option<PfnVkEndCommandBuffer>,
    /// `vkCmdBindPipeline`。
    pub cmd_bind_pipeline: Option<PfnVkCmdBindPipeline>,
    /// `vkCmdBindDescriptorSets`。
    pub cmd_bind_descriptor_sets: Option<PfnVkCmdBindDescriptorSets>,
    /// `vkCmdDispatch`。
    pub cmd_dispatch: Option<PfnVkCmdDispatch>,
    /// `vkQueueSubmit`。
    pub queue_submit: Option<PfnVkQueueSubmit>,
    /// `vkQueueWaitIdle`。
    pub queue_wait_idle: Option<PfnVkQueueWaitIdle>,
    /// `vkCreateFence`。
    pub create_fence: Option<PfnVkCreateFence>,
    /// `vkDestroyFence`。
    pub destroy_fence: Option<PfnVkDestroyFence>,
    /// `vkWaitForFences`。
    pub wait_for_fences: Option<PfnVkWaitForFences>,
    /// `vkMapMemory`。
    pub map_memory: Option<PfnVkMapMemory>,
    /// `vkUnmapMemory`。
    pub unmap_memory: Option<PfnVkUnmapMemory>,
}

impl DeviceFnTable {
    /// 绑定设备级入口（经 `vkGetInstanceProcAddr(instance, name)`）。
    ///
    /// # Safety
    /// 调用方须保证 `get_proc` 有效且 `instance` 是承载 `device` 的实例。
    pub unsafe fn bind_all(
        get_proc: PfnVkGetInstanceProcAddr,
        instance: VkInstance,
    ) -> DeviceFnTable {
        macro_rules! bind {
            ($name:literal, $ty:ty) => {{
                let sym = get_proc(instance, $name.as_ptr());
                if sym.is_null() {
                    None
                } else {
                    Some(std::mem::transmute_copy::<*const c_void, $ty>(&sym))
                }
            }};
        }
        DeviceFnTable {
            destroy_device: bind!(c"vkDestroyDevice", PfnVkDestroyDevice),
            get_device_queue: bind!(c"vkGetDeviceQueue", PfnVkGetDeviceQueue),
            create_shader_module: bind!(c"vkCreateShaderModule", PfnVkCreateShaderModule),
            destroy_shader_module: bind!(c"vkDestroyShaderModule", PfnVkDestroyShaderModule),
            create_descriptor_set_layout: bind!(
                c"vkCreateDescriptorSetLayout",
                PfnVkCreateDescriptorSetLayout
            ),
            destroy_descriptor_set_layout: bind!(
                c"vkDestroyDescriptorSetLayout",
                PfnVkDestroyDescriptorSetLayout
            ),
            create_pipeline_layout: bind!(c"vkCreatePipelineLayout", PfnVkCreatePipelineLayout),
            destroy_pipeline_layout: bind!(c"vkDestroyPipelineLayout", PfnVkDestroyPipelineLayout),
            create_compute_pipelines: bind!(
                c"vkCreateComputePipelines",
                PfnVkCreateComputePipelines
            ),
            destroy_pipeline: bind!(c"vkDestroyPipeline", PfnVkDestroyPipeline),
            create_buffer: bind!(c"vkCreateBuffer", PfnVkCreateBuffer),
            destroy_buffer: bind!(c"vkDestroyBuffer", PfnVkDestroyBuffer),
            get_buffer_memory_requirements: bind!(
                c"vkGetBufferMemoryRequirements",
                PfnVkGetBufferMemoryRequirements
            ),
            allocate_memory: bind!(c"vkAllocateMemory", PfnVkAllocateMemory),
            free_memory: bind!(c"vkFreeMemory", PfnVkFreeMemory),
            bind_buffer_memory: bind!(c"vkBindBufferMemory", PfnVkBindBufferMemory),
            create_descriptor_pool: bind!(
                c"vkCreateDescriptorPool",
                PfnVkCreateDescriptorPool
            ),
            destroy_descriptor_pool: bind!(
                c"vkDestroyDescriptorPool",
                PfnVkDestroyDescriptorPool
            ),
            allocate_descriptor_sets: bind!(
                c"vkAllocateDescriptorSets",
                PfnVkAllocateDescriptorSets
            ),
            update_descriptor_sets: bind!(c"vkUpdateDescriptorSets", PfnVkUpdateDescriptorSets),
            create_command_pool: bind!(c"vkCreateCommandPool", PfnVkCreateCommandPool),
            destroy_command_pool: bind!(c"vkDestroyCommandPool", PfnVkDestroyCommandPool),
            allocate_command_buffers: bind!(
                c"vkAllocateCommandBuffers",
                PfnVkAllocateCommandBuffers
            ),
            free_command_buffers: bind!(c"vkFreeCommandBuffers", PfnVkFreeCommandBuffers),
            begin_command_buffer: bind!(c"vkBeginCommandBuffer", PfnVkBeginCommandBuffer),
            end_command_buffer: bind!(c"vkEndCommandBuffer", PfnVkEndCommandBuffer),
            cmd_bind_pipeline: bind!(c"vkCmdBindPipeline", PfnVkCmdBindPipeline),
            cmd_bind_descriptor_sets: bind!(
                c"vkCmdBindDescriptorSets",
                PfnVkCmdBindDescriptorSets
            ),
            cmd_dispatch: bind!(c"vkCmdDispatch", PfnVkCmdDispatch),
            queue_submit: bind!(c"vkQueueSubmit", PfnVkQueueSubmit),
            queue_wait_idle: bind!(c"vkQueueWaitIdle", PfnVkQueueWaitIdle),
            create_fence: bind!(c"vkCreateFence", PfnVkCreateFence),
            destroy_fence: bind!(c"vkDestroyFence", PfnVkDestroyFence),
            wait_for_fences: bind!(c"vkWaitForFences", PfnVkWaitForFences),
            map_memory: bind!(c"vkMapMemory", PfnVkMapMemory),
            unmap_memory: bind!(c"vkUnmapMemory", PfnVkUnmapMemory),
        }
    }
}

// ---------------------------------------------------------------------------
// 探测请求构造（`VkApplicationInfo` / `VkInstanceCreateInfo`）
// ---------------------------------------------------------------------------

/// 构造 `VkApplicationInfo` + `VkInstanceCreateInfo` 并调用 `vkCreateInstance`。
///
/// * `api_version`：请求的 Vulkan API 版本（`VK_API_VERSION_1_0`）。
/// * 不启用任何 layer / extension（探测阶段无需 present / surface 支持）。
///
/// 返回 `(result, instance)`；调用方须先判 `result == VK_SUCCESS`。
///
/// # Safety
/// 调用方须保证 `create_instance` 指针有效。
pub(crate) unsafe fn create_instance(
    create_instance: PfnVkCreateInstance,
    api_version: u32,
) -> (VkResult, VkInstance) {
    // 静态 C 字符串常量，'static 生命周期覆盖整个调用。
    let app_name = c"Prism Engine";
    let engine_name = c"prism-gpu";

    let app_info = VkApplicationInfo {
        s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: app_name.as_ptr(),
        application_version: 0,
        p_engine_name: engine_name.as_ptr(),
        engine_version: 0,
        api_version,
    };

    let create_info = VkInstanceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
    };

    let mut instance: VkInstance = std::ptr::null_mut();
    let result = create_instance(&create_info, std::ptr::null(), &mut instance);
    if result != VK_SUCCESS {
        // 失败时不信任驱动写回的句柄，一律返回空。
        return (result, std::ptr::null_mut());
    }
    (result, instance)
}

// ---------------------------------------------------------------------------
// 设备探测与队列族选择
// ---------------------------------------------------------------------------

/// 已枚举到的物理设备摘要（诊断用 + 选择用）。
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// 原始句柄。
    pub handle: VkPhysicalDevice,
    /// 设备类型。
    pub device_type: PhysicalDeviceType,
    /// 设备名（UTF-8 lossy 化的前缀）。
    pub name: String,
    /// 首个支持 graphics 的队列族索引；无则为 `None`。
    pub graphics_queue_family: Option<u32>,
    /// 队列族数量。
    pub queue_family_count: u32,
}

/// 读取 `deviceType`（只读前缀，尾部不解释）。
///
/// # Safety
/// `props` 须由 `vkGetPhysicalDeviceProperties` 写回完整结构。
pub(crate) unsafe fn device_type_of(
    props: &VkPhysicalDevicePropertiesStorage,
) -> PhysicalDeviceType {
    PhysicalDeviceType::from_raw(props.prefix.device_type)
}

/// 读设备名（截断到首个 NUL，UTF-8 lossy）。
///
/// # Safety
/// 同 [`device_type_of`]。
pub(crate) unsafe fn device_name_of(props: &VkPhysicalDevicePropertiesStorage) -> String {
    let bytes = &props.prefix.device_name;
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

/// 在给定的队列族数组里选第一个支持 `VK_QUEUE_GRAPHICS_BIT` 的族。
///
/// 无 graphics 族返回 `None`。
pub fn first_graphics_queue_family(families: &[VkQueueFamilyProperties]) -> Option<u32> {
    families
        .iter()
        .enumerate()
        .find(|(_, p)| (p.queue_flags & VK_QUEUE_GRAPHICS_BIT) != 0)
        .map(|(idx, _)| idx as u32)
}

/// 按 [`PhysicalDeviceType::preference_weight`] 选最优设备索引。
///
/// 并列时取枚举顺序靠前者（稳定、确定性）。
pub fn pick_best_device(infos: &[PhysicalDeviceInfo]) -> Option<usize> {
    infos
        .iter()
        .enumerate()
        .max_by_key(|(idx, info)| (info.device_type.preference_weight(), std::cmp::Reverse(*idx)))
        .map(|(idx, _)| idx)
}

// ---------------------------------------------------------------------------
// 单元测试（纯逻辑：无 GPU、无 dlopen）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 队列族选择：跳过无 graphics 位的族，命中第一个有 graphics 位的族。
    #[test]
    fn graphics_queue_family_selection_skips_non_graphics() {
        let families = [
            VkQueueFamilyProperties {
                queue_flags: 0, // 无任何位
                queue_count: 1,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [1, 1, 1],
            },
            VkQueueFamilyProperties {
                queue_flags: 0x10, // VK_QUEUE_TRANSFER_BIT，非 graphics
                queue_count: 2,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [1, 1, 1],
            },
            VkQueueFamilyProperties {
                queue_flags: VK_QUEUE_GRAPHICS_BIT,
                queue_count: 4,
                timestamp_valid_bits: 64,
                min_image_transfer_granularity: [1, 1, 1],
            },
        ];
        assert_eq!(
            first_graphics_queue_family(&families),
            Some(2),
            "应跳过队列族 0/1，命中队列族 2"
        );
    }

    /// 无 graphics 族 ⇒ `None`（不得 panic、不得越界）。
    #[test]
    fn graphics_queue_family_absent_returns_none() {
        let families = [VkQueueFamilyProperties {
            queue_flags: 0x2,
            queue_count: 1,
            timestamp_valid_bits: 0,
            min_image_transfer_granularity: [1, 1, 1],
        }];
        assert_eq!(first_graphics_queue_family(&families), None);
        assert_eq!(first_graphics_queue_family(&[]), None, "空数组应返回 None");
    }

    /// 设备偏好：离散 GPU 优先于集成/虚拟/CPU。
    #[test]
    fn pick_best_device_prefers_discrete() {
        let mk = |t: PhysicalDeviceType, q: Option<u32>| PhysicalDeviceInfo {
            handle: std::ptr::null_mut(),
            device_type: t,
            name: String::new(),
            graphics_queue_family: q,
            queue_family_count: 1,
        };
        let infos = [
            mk(PhysicalDeviceType::IntegratedGpu, Some(0)),
            mk(PhysicalDeviceType::DiscreteGpu, Some(0)),
            mk(PhysicalDeviceType::Cpu, Some(0)),
        ];
        assert_eq!(pick_best_device(&infos), Some(1), "应选中离散 GPU（索引 1）");
        assert_eq!(pick_best_device(&[]), None, "空列表应返回 None");
    }

    /// 设备类型映射：识别码逐一对齐，未识别码收敛到 `Unknown`（不 panic）。
    #[test]
    fn physical_device_type_mapping_is_total() {
        assert_eq!(PhysicalDeviceType::from_raw(0), PhysicalDeviceType::Other);
        assert_eq!(
            PhysicalDeviceType::from_raw(1),
            PhysicalDeviceType::IntegratedGpu
        );
        assert_eq!(
            PhysicalDeviceType::from_raw(2),
            PhysicalDeviceType::DiscreteGpu
        );
        assert_eq!(PhysicalDeviceType::from_raw(3), PhysicalDeviceType::VirtualGpu);
        assert_eq!(PhysicalDeviceType::from_raw(4), PhysicalDeviceType::Cpu);
        assert_eq!(
            PhysicalDeviceType::from_raw(99),
            PhysicalDeviceType::Unknown(99),
            "未识别码不得 panic"
        );
    }

    /// `VkApplicationInfo` / `VkInstanceCreateInfo` 的内存布局自证。
    ///
    /// **非恒真自证**：若有人在结构体里插了一个字段或改了字段顺序，这些
    /// `size_of` / `align_of` 断言会立刻失败——它们把「与 Vulkan 头文件一致」
    /// 这一契约钉死。数值按 LP64 ABI 计算（`rustc` 实测）：
    /// * `VkApplicationInfo` = 48（指针 8 字节对齐在 `sType` 后插入 4 字节 pad）。
    /// * `VkInstanceCreateInfo` = 64。
    /// * `VkDeviceQueueCreateInfo` = 40。
    /// * `VkDeviceCreateInfo` = 72。
    #[test]
    fn ffi_struct_layouts_match_vulkan_abi() {
        assert_eq!(
            std::mem::size_of::<VkApplicationInfo>(),
            48,
            "VkApplicationInfo 布局"
        );
        assert_eq!(
            std::mem::size_of::<VkInstanceCreateInfo>(),
            64,
            "VkInstanceCreateInfo 布局"
        );
        assert_eq!(
            std::mem::size_of::<VkDeviceQueueCreateInfo>(),
            40,
            "VkDeviceQueueCreateInfo 布局"
        );
        assert_eq!(
            std::mem::size_of::<VkDeviceCreateInfo>(),
            72,
            "VkDeviceCreateInfo 布局"
        );
        // 前缀尺寸：4×4 + 4 + 256 + 16 = 292，无额外 pad（全 4 字节对齐）。
        assert_eq!(
            std::mem::size_of::<VkPhysicalDevicePropertiesPrefix>(),
            292,
            "VkPhysicalDeviceProperties 前缀布局"
        );
        // 承接缓冲须容纳前缀 + 充分余量（≥ 前缀，且对齐 8）。
        assert!(
            std::mem::size_of::<VkPhysicalDevicePropertiesStorage>() >= 4096,
            "properties 承接缓冲须足够大"
        );
        assert_eq!(
            std::mem::align_of::<VkPhysicalDevicePropertiesStorage>(),
            8,
            "properties 缓冲须 8 字节对齐"
        );
    }

    /// `VK_API_VERSION_1_0` 打包正确（`1.0.0` = `1<<22`）。
    #[test]
    fn api_version_packing_is_correct() {
        assert_eq!(VK_API_VERSION_1_0, 1 << 22);
    }

    /// `RTLD_LAZY` 取 glibc 值 —— **注明平台依赖**：此断言刻画 Linux/glibc。
    ///
    /// **非恒真自证**：若将来为 macOS 改写该常量（`RTLD_LAZY` 也是 1），或
    /// 改为 `RTLD_NOW`（glibc = 2），本断言会失败——它不是永真空断言。
    #[test]
    fn rtld_lazy_is_glibc_value_on_linux() {
        assert_eq!(RTLD_LAZY, 1, "glibc 的 RTLD_LAZY 为 1（见常量注释）");
    }
}

// ---------------------------------------------------------------------------
// compute 管线扩展：SPIR-V 模块 / compute pipeline / 描述符 / 命令缓冲 /
// 队列提交（V2.0-G）
// ---------------------------------------------------------------------------
//
// 这一段的每个 `struct` 都与 `<vulkan/vulkan_core.h>` **逐字段对齐**
// （字段顺序、`size_t`/`uint64_t` 宽度、指针对齐），并有编译期尺寸断言
// （见本文件 `tests::v2g_struct_layouts`）。不加任何第三方 crate（D-15）。
//
// 诚实边界：本仓库构建环境**无 ICD**（见 `super` 模块头注），因此这段代码
// 只能被「结构校验 + CPU 侧装配计划」测试覆盖；真机派发由
// `crate::vulkan::compute` 的 `#[ignore]` 测试占位（探针如实报无设备）。

/// 非分派句柄（Vulkan 规范：64 位不透明值，可为 0 表示空）。
pub type VkShaderModule = u64;
/// `VkBuffer`。
pub type VkBuffer = u64;
/// `VkDeviceMemory`。
pub type VkDeviceMemory = u64;
/// `VkPipeline`。
pub type VkPipeline = u64;
/// `VkPipelineLayout`。
pub type VkPipelineLayout = u64;
/// `VkDescriptorSetLayout`。
pub type VkDescriptorSetLayout = u64;
/// `VkDescriptorPool`。
pub type VkDescriptorPool = u64;
/// `VkDescriptorSet`。
pub type VkDescriptorSet = u64;
/// `VkCommandPool`。
pub type VkCommandPool = u64;
/// `VkFence`。
pub type VkFence = u64;
/// 可分派句柄（`VkCommandBuffer` 是指针，与 `VkDevice` 同类）。
pub type VkCommandBuffer = *mut c_void;

/// `VK_STRUCTURE_TYPE_SUBMIT_INFO`。
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: i32 = 4;
/// `VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO`。
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: i32 = 5;
/// `VK_STRUCTURE_TYPE_FENCE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: i32 = 8;
/// `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: i32 = 12;
/// `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO`。
///
/// V2.0-W4 lavapipe 实测修正：旧值 15 实为 `PIPELINE_CACHE_CREATE_INFO`，
/// 错误 sType 使 `vkCreateShaderModule` 行为未定义（验证层报
/// VUID-VkShaderModuleCreateInfo-sType-sType）。
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: i32 = 16;
/// `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: i32 = 18;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO`。
///
/// V2.0-W4 lavapipe 实测修正：旧值 20 实为
/// `PIPELINE_VIEWPORT_STATE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: i32 = 32;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO`。
///
/// V2.0-W4 lavapipe 实测修正：旧值 22 实为
/// `PIPELINE_MULTISAMPLE_STATE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: i32 = 33;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO`。
///
/// V2.0-W4 lavapipe 实测修正：旧值 23 实为
/// `PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: i32 = 34;
/// `VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: i32 = 29;
/// `VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: i32 = 30;
/// `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET`。
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: i32 = 35;
/// `VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO`。
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: i32 = 39;
/// `VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO`。
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: i32 = 40;
/// `VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO`。
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: i32 = 42;

/// `VK_SHADER_STAGE_COMPUTE_BIT`。
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0020;
/// `VK_PIPELINE_BIND_POINT_COMPUTE`。
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
/// `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`。
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
/// `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`。
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0020;
/// `VK_SHARING_MODE_EXCLUSIVE`。
pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
/// `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`。
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0000_0002;
/// `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`。
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0000_0004;
/// `VK_COMMAND_BUFFER_LEVEL_PRIMARY`。
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
/// `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`。
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x0000_0002;
/// `VK_WHOLE_SIZE`（`VK_WHOLE_SIZE` = `(~0ULL)`，此处以 `u64::MAX` 表达）。
pub const VK_WHOLE_SIZE: u64 = u64::MAX;

/// `VkShaderModuleCreateInfo`。
#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}

/// `VkPipelineShaderStageCreateInfo`。
#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}

/// `VkComputePipelineCreateInfo`。
#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

/// `VkDescriptorSetLayoutBinding`。
#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub p_immutable_samplers: *const c_void,
}

/// `VkDescriptorSetLayoutCreateInfo`。
#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

/// `VkPipelineLayoutCreateInfo`。
#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const c_void,
}

/// `VkBufferCreateInfo`。
#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: u64,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

/// `VkMemoryRequirements`。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VkMemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
}

/// `VkMemoryAllocateInfo`。
#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub allocation_size: u64,
    pub memory_type_index: u32,
}

/// `VkMemoryType`（`VkPhysicalDeviceMemoryProperties` 的内嵌元素）。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VkMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

/// `VkMemoryHeap`（同上）。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VkMemoryHeap {
    pub size: u64,
    pub flags: u32,
    pub _pad: u32,
}

/// `VkPhysicalDeviceMemoryProperties`（定长内嵌数组，POD）。
#[repr(C)]
pub struct VkPhysicalDeviceMemoryPropertiesStorage {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 32],
}

impl VkPhysicalDeviceMemoryPropertiesStorage {
    /// 全零初始化（驱动覆写；未写到的尾部保持 0，可安全读前缀）。
    pub fn zeroed() -> Self {
        // SAFETY: 结构体为 POD（全整型/数组），全零是合法位模式。
        unsafe { std::mem::zeroed() }
    }

    /// 选一个满足 `required` 且 host 可见 + coherent 的内存类型下标。
    ///
    /// 返回 `None` 表示无匹配（调用方据此降级为「不可派发」，不 panic）。
    pub fn pick_host_visible(&self, required: u32) -> Option<u32> {
        let want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let count = self.memory_type_count.min(32);
        for i in 0..count {
            let flags = self.memory_types[i as usize].property_flags;
            if (required & (1 << i)) != 0 && (flags & want) == want {
                return Some(i);
            }
        }
        None
    }
}

/// `VkCommandPoolCreateInfo`。
#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

/// `VkCommandBufferAllocateInfo`。
#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: u32,
    pub command_buffer_count: u32,
}

/// `VkCommandBufferBeginInfo`。
#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}

/// `VkSubmitInfo`。
#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const u64,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const u64,
}

/// `VkFenceCreateInfo`。
#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
}

/// `VkDescriptorPoolSize`。
#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub ty: u32,
    pub descriptor_count: u32,
}

/// `VkDescriptorPoolCreateInfo`。
#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

/// `VkDescriptorSetAllocateInfo`。
#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

/// `VkDescriptorBufferInfo`。
#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: u64,
    pub range: u64,
}

/// `VkWriteDescriptorSet`。
#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

// --- compute 相关函数指针类型 ------------------------------------------------

/// `vkCreateShaderModule`。
pub type PfnVkCreateShaderModule = unsafe extern "system" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> VkResult;
/// `vkDestroyShaderModule`。
pub type PfnVkDestroyShaderModule = unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void);
/// `vkCreateDescriptorSetLayout`。
pub type PfnVkCreateDescriptorSetLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> VkResult;
/// `vkDestroyDescriptorSetLayout`。
pub type PfnVkDestroyDescriptorSetLayout =
    unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
/// `vkCreatePipelineLayout`。
pub type PfnVkCreatePipelineLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> VkResult;
/// `vkDestroyPipelineLayout`。
pub type PfnVkDestroyPipelineLayout =
    unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void);
/// `vkCreateComputePipelines`。
pub type PfnVkCreateComputePipelines = unsafe extern "system" fn(
    VkDevice,
    u64,
    u32,
    *const VkComputePipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> VkResult;
/// `vkDestroyPipeline`。
pub type PfnVkDestroyPipeline = unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void);
/// `vkCreateBuffer`。
pub type PfnVkCreateBuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> VkResult;
/// `vkDestroyBuffer`。
pub type PfnVkDestroyBuffer = unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);
/// `vkGetBufferMemoryRequirements`。
pub type PfnVkGetBufferMemoryRequirements =
    unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
/// `vkAllocateMemory`。
pub type PfnVkAllocateMemory = unsafe extern "system" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> VkResult;
/// `vkFreeMemory`。
pub type PfnVkFreeMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
/// `vkBindBufferMemory`。
pub type PfnVkBindBufferMemory = unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> VkResult;
/// `vkCreateDescriptorPool`。
pub type PfnVkCreateDescriptorPool = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> VkResult;
/// `vkDestroyDescriptorPool`。
pub type PfnVkDestroyDescriptorPool =
    unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void);
/// `vkAllocateDescriptorSets`。
pub type PfnVkAllocateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> VkResult;
/// `vkUpdateDescriptorSets`。
pub type PfnVkUpdateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const VkWriteDescriptorSet,
    u32,
    *const c_void,
);
/// `vkCreateCommandPool`。
pub type PfnVkCreateCommandPool = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> VkResult;
/// `vkDestroyCommandPool`。
pub type PfnVkDestroyCommandPool = unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void);
/// `vkAllocateCommandBuffers`。
pub type PfnVkAllocateCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> VkResult;
/// `vkFreeCommandBuffers`。
pub type PfnVkFreeCommandBuffers =
    unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer);
/// `vkBeginCommandBuffer`。
pub type PfnVkBeginCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult;
/// `vkEndCommandBuffer`。
pub type PfnVkEndCommandBuffer = unsafe extern "system" fn(VkCommandBuffer) -> VkResult;
/// `vkCmdBindPipeline`。
pub type PfnVkCmdBindPipeline = unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline);
/// `vkCmdBindDescriptorSets`。
pub type PfnVkCmdBindDescriptorSets = unsafe extern "system" fn(
    VkCommandBuffer,
    u32,
    VkPipelineLayout,
    u32,
    u32,
    *const VkDescriptorSet,
    u32,
    *const u32,
);
/// `vkCmdDispatch`。
pub type PfnVkCmdDispatch = unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32);
/// `vkQueueSubmit`。
pub type PfnVkQueueSubmit =
    unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult;
/// `vkQueueWaitIdle`。
pub type PfnVkQueueWaitIdle = unsafe extern "system" fn(VkQueue) -> VkResult;
/// `vkCreateFence`。
pub type PfnVkCreateFence = unsafe extern "system" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> VkResult;
/// `vkDestroyFence`。
pub type PfnVkDestroyFence = unsafe extern "system" fn(VkDevice, VkFence, *const c_void);
/// `vkWaitForFences`。
pub type PfnVkWaitForFences = unsafe extern "system" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult;
/// `vkMapMemory`。
pub type PfnVkMapMemory =
    unsafe extern "system" fn(VkDevice, VkDeviceMemory, u64, u64, u32, *mut *mut c_void) -> VkResult;
/// `vkUnmapMemory`。
pub type PfnVkUnmapMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory);
/// `vkGetPhysicalDeviceMemoryProperties`。
pub type PfnVkGetPhysicalDeviceMemoryProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryPropertiesStorage);

// ---------------------------------------------------------------------------
// 编译期布局断言（真机 FFI 的**唯一**可在此环境验证的正确性证据）
// ---------------------------------------------------------------------------

const _: () = {
    // 只断言本环境能验证的东西：尺寸 + 对齐。字段偏移由 `size` + `repr(C)`
    // 的 C ABI 规则共同决定（见各 struct 上方的逐字段推导注释）。
    assert!(std::mem::size_of::<VkShaderModuleCreateInfo>() == 40);
    assert!(std::mem::size_of::<VkPipelineShaderStageCreateInfo>() == 48);
    assert!(std::mem::size_of::<VkComputePipelineCreateInfo>() == 96);
    assert!(std::mem::size_of::<VkDescriptorSetLayoutBinding>() == 24);
    assert!(std::mem::size_of::<VkDescriptorSetLayoutCreateInfo>() == 32);
    assert!(std::mem::size_of::<VkPipelineLayoutCreateInfo>() == 48);
    assert!(std::mem::size_of::<VkBufferCreateInfo>() == 56);
    assert!(std::mem::size_of::<VkMemoryRequirements>() == 24);
    assert!(std::mem::size_of::<VkMemoryAllocateInfo>() == 32);
    assert!(std::mem::size_of::<VkCommandPoolCreateInfo>() == 24);
    assert!(std::mem::size_of::<VkCommandBufferAllocateInfo>() == 32);
    assert!(std::mem::size_of::<VkCommandBufferBeginInfo>() == 32);
    assert!(std::mem::size_of::<VkSubmitInfo>() == 72);
    assert!(std::mem::size_of::<VkFenceCreateInfo>() == 24);
    assert!(std::mem::size_of::<VkDescriptorPoolCreateInfo>() == 40);
    assert!(std::mem::size_of::<VkDescriptorSetAllocateInfo>() == 40);
    assert!(std::mem::size_of::<VkDescriptorBufferInfo>() == 24);
    assert!(std::mem::size_of::<VkWriteDescriptorSet>() == 64);
    assert!(std::mem::size_of::<VkPhysicalDeviceMemoryPropertiesStorage>() == 776);
};
