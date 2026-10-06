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
        }
    }
}

/// 从设备取到的函数指针集合。
pub(crate) struct DeviceFnTable {
    /// `vkDestroyDevice`。
    pub destroy_device: Option<PfnVkDestroyDevice>,
    /// `vkGetDeviceQueue`。
    pub get_device_queue: Option<PfnVkGetDeviceQueue>,
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
