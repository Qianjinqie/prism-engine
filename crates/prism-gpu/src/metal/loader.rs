//! Metal 设备探测链（**仅 `target_os = "macos"` 编译**）。
//!
//! # 为何走 `dlopen` + `dlsym` 而非 `objc_msgSend`
//!
//! Metal 的对象（`MTLDevice` 等）是 Objective-C 对象，与之交互要 `objc_msgSend`。
//! 但**设备发现**这一层入口是纯 C 函数：
//! * `id<MTLDevice> MTLCreateSystemDefaultDevice(void);`
//! * `NSArray<id<MTLDevice>> *MTLCopyAllDevices(void);`
//!
//! 因此探测层用 `dlopen` + `dlsym` 拿这两个符号即可，无需 Objective-C runtime
//! FFI 声明（更少 cast、更稳、更少 panic 面）。**拿到 `MTLDevice` 指针之后**，
//! 若要读 `device.name` 或建 pipeline，就必须进入 `objc_msgSend` 消息发送——
//! 这一层**本模块刻意不做**（见模块末尾"下一步"注释），故 `is_available` 恒 false。
//!
//! # 零第三方 crate
//!
//! `dlopen`/`dlsym`/`dlclose` 来自系统 `libdl`（`#[link(name = "dl")]`），
//! `MTLCreateSystemDefaultDevice` 等来自系统 Metal 框架。二者都是 OS 原生 API，
//! 不构成第三方依赖（D-15）。

use super::{ProbeFailure, ProbeReport};

// ---------------------------------------------------------------------------
// libdl FFI（系统库，非第三方 crate）
// ---------------------------------------------------------------------------

/// `RTLD_LAZY | RTLD_LOCAL`（macOS `<dlfcn.h>` 常量）。
///
/// 取字面量而非 `dlsym` 常量，避免依赖 `libc` crate（零第三方）。
/// macOS 上 `RTLD_LAZY == 0x1`，`RTLD_LOCAL == 0x4`。
const RTLD_LAZY: i32 = 0x1;
const RTLD_LOCAL: i32 = 0x4;

#[link(name = "dl")]
extern "C" {
    /// `void *dlopen(const char *path, int mode);`
    fn dlopen(path: *const i8, mode: i32) -> *mut core::ffi::c_void;
    /// `void *dlsym(void *handle, const char *symbol);`
    fn dlsym(handle: *mut core::ffi::c_void, symbol: *const i8) -> *mut core::ffi::c_void;
    /// `int dlclose(void *handle);`
    fn dlclose(handle: *mut core::ffi::c_void) -> i32;
}

// ---------------------------------------------------------------------------
// Metal C 入口签名
// ---------------------------------------------------------------------------

/// `MTLCreateSystemDefaultDevice()` → `id<MTLDevice>`（可空）。
type MtlCreateSystemDefaultDevice = unsafe extern "C" fn() -> *mut core::ffi::c_void;

/// `MTLCopyAllDevices()` → `NSArray<id<MTLDevice>> *`（可空）。
///
/// 返回的 `NSArray` 是 Objective-C 对象；本模块只做"是否存在"判断，不遍历其
/// 元素（遍历需 `objc_msgSend`，超出本层职责）。故此处用"非空即视为至少一块
/// 设备"的保守判据，并在 [`full_probe`] 的注释里标注其局限。
type MtlCopyAllDevices = unsafe extern "C" fn() -> *mut core::ffi::c_void;

/// 系统 Metal 框架路径（macOS 固定位置）。
const METAL_FRAMEWORK_PATH: &[u8] = b"/System/Library/Frameworks/Metal.framework/Metal\0";

/// 探测结果（macOS）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceProbe {
    /// 拿到默认设备；`device_ptr` 为 `MTLDevice` 的裸指针（不透明，仅供诊断打印）。
    Present { device_ptr: usize },
    /// 不可用，附原因。
    Unavailable { reason: ProbeFailure },
}

// ---------------------------------------------------------------------------
// 探测实现
// ---------------------------------------------------------------------------

/// 打开 Metal 框架，返回句柄；失败返回 `None`。
///
/// # Safety
/// 纯 FFI 调用壳；`dlopen` 对错误路径返回空指针，本函数把它映射为 `None`。
unsafe fn open_metal_framework() -> Option<*mut core::ffi::c_void> {
    let handle = dlopen(
        METAL_FRAMEWORK_PATH.as_ptr() as *const i8,
        RTLD_LAZY | RTLD_LOCAL,
    );
    if handle.is_null() {
        None
    } else {
        Some(handle)
    }
}

/// 取符号并转成目标函数指针；符号缺失返回 `None`。
///
/// # Safety
/// 调用方须保证 `handle` 来自 [`open_metal_framework`] 且符号 ABI 与 `F` 匹配。
unsafe fn load_symbol<F: Copy>(handle: *mut core::ffi::c_void, name: &[u8]) -> Option<F> {
    debug_assert!(name.last() == Some(&0), "符号名须以 NUL 结尾");
    let sym = dlsym(handle, name.as_ptr() as *const i8);
    if sym.is_null() {
        None
    } else {
        // 函数指针与数据指针同宽于目标 ABI；`transmute_copy` 避免 `mem::transmute`
        // 对大小约束的编译期断言。
        Some(core::mem::transmute_copy::<*mut core::ffi::c_void, F>(&sym))
    }
}

/// 默认设备探测：走完整链条，返回设备有无。
///
/// 链条：`dlopen(Metal)` → `dlsym(MTLCreateSystemDefaultDevice)` → 调用。
pub fn default_device_probe() -> DeviceProbe {
    // SAFETY: 以下 FFI 调用均有空指针检查；`handle` 仅在成功时非空并传给 dlsym。
    unsafe {
        let Some(handle) = open_metal_framework() else {
            return DeviceProbe::Unavailable {
                reason: ProbeFailure::FrameworkLoadFailed,
            };
        };
        let Some(create) =
            load_symbol::<MtlCreateSystemDefaultDevice>(handle, b"MTLCreateSystemDefaultDevice\0")
        else {
            let _ = dlclose(handle);
            return DeviceProbe::Unavailable {
                reason: ProbeFailure::SymbolMissing,
            };
        };
        let device = create();
        let result = if device.is_null() {
            DeviceProbe::Unavailable {
                reason: ProbeFailure::NoDefaultDevice,
            }
        } else {
            DeviceProbe::Present {
                device_ptr: device as usize,
            }
        };
        // 不 dlclose：Metal 框架在整个进程生命周期内保持加载是标准做法，过早
        // 关闭可能使已缓存的函数指针悬空。
        result
    }
}

/// 完整探测：在默认设备之外，再尝试 `MTLCopyAllDevices` 统计设备数量。
///
/// # 设备计数的局限（诚实标注）
/// `MTLCopyAllDevices` 返回一个 Objective-C `NSArray`。要读其 `count` 需
/// `objc_msgSend`，超出本层范围。故此处以"默认设备存在"计 `1`，并以数组
/// 非空作为"≥1"的交叉旁证；**这不是精确计数**，仅用于诊断报告。
pub fn full_probe() -> ProbeReport {
    match default_device_probe() {
        DeviceProbe::Present { .. } => {
            // 旁证：`MTLCopyAllDevices` 是否也返回非空数组。
            let extra_present = all_devices_nonempty();
            ProbeReport {
                supported_platform: true,
                device_present: true,
                // 默认设备已确认存在；数组非空作为交叉印证。取 1 为下限。
                device_count: usize::from(extra_present).max(1),
                reason: None,
            }
        }
        DeviceProbe::Unavailable { reason } => ProbeReport {
            supported_platform: true,
            device_present: false,
            device_count: 0,
            reason: Some(reason),
        },
    }
}

/// 调用 `MTLCopyAllDevices`，判断其返回数组指针是否非空（仅"有无"旁证）。
fn all_devices_nonempty() -> bool {
    // SAFETY: 与默认设备探测同一套 dlopen/dlsym；调用前逐个空检。
    unsafe {
        let Some(handle) = open_metal_framework() else {
            return false;
        };
        let Some(copy) = load_symbol::<MtlCopyAllDevices>(handle, b"MTLCopyAllDevices\0") else {
            return false;
        };
        !copy().is_null()
    }
}

// ---------------------------------------------------------------------------
// 下一步（真渲染闭环所需的 Objective-C 层，本模块未实现）
// ---------------------------------------------------------------------------
//
// 要真正出图，还须（均需 `objc_msgSend`，此处刻意留白）：
//   1. `device.newCommandQueue` 取命令队列；
//   2. `device.newLibraryWithSource:options:error:` 或预编译 `.metallib`
//      加载 MSL（见 `src/metal/msl/`）；
//   3. 建 `MTLRenderPipelineState`（或 compute pipeline）；
//   4. 建纹理 / buffer，编码 draw/dispatch，`commit` + `waitUntilCompleted`；
//   5. `getBytes:bytesPerRow:` 读回 RGBA8 到 `Framebuffer`。
// 完成 1~5 之后，`MetalBackend::is_available` 才可改为返回 `true`。
