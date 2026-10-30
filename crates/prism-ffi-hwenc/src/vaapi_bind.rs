//! VA-API **绑定就绪层**（第二档）：`extern "C"` 函数表声明 + 手写 `dlopen`/`dlsym` 解析。
//!
//! # 四档分述（本模块只站在**第二档**，一寸不多）
//!
//! | 档 | 名称（[`crate::Maturity`]） | 判定标准 | 本模块 |
//! |---|---|---|---|
//! | 一 | `probe-chain` | 真读平台证据（`/dev/dri` 渲染节点） | 由 `crate::env_probe` 承担，**不是**本模块 |
//! | 二 | `binding-ready` | `dlopen` 成功 + **全部**核心符号 `dlsym` 解析进函数指针表 | ✅ **本模块的全部产出** |
//! | 三 | `session-established` | 拿到 `VADisplay`、`vaInitialize`、`vaCreateConfig`、`vaCreateContext` 成功 | ❌ 不做（见下"三个不做"） |
//! | 四 | `device-encoded` | 真机跑通一次编码并校验码流 | ❌ 不做（环境受限） |
//!
//! **第二档 ≠ 能编码**：解析到函数指针只证明"符号在库里找得到、ABI 形状对得上"，
//! 不证明设备在、不证明驱动接受、更不证明出得了码。任何对外表述止于
//! "绑定就绪（第二档）"。
//!
//! # 三个不做（各自都有明确理由，不是省事）
//!
//! 1. **不调用任何 VA-API 入口**：函数表只解析、不调用。`vaInitialize` 需要先有
//!    合法 `VADisplay`，而 `VADisplay` 由 `vaGetDisplayDRM`（在 `libva-drm.so.2`）
//!    或 `vaGetDisplay`（`libva-x11` / `libva-wayland`）产出——那是**第三档**的事。
//!    本模块连 `libva-drm` 都不 `dlopen`，避免"看起来在建会话"。
//! 2. **不 `#[link]` 任何库**（含 `libdl`）：见下节。
//! 3. **不缓存活句柄**：`bind()` 返回的函数表持有 `dlopen` 句柄并在 `Drop` 时
//!    `dlclose`；探测复用的 [`bind_report`] **只缓存事实字符串**（库名 / 符号数 /
//!    错误证据），不缓存句柄，避免探测把库长驻在进程里。
//!
//! # 为什么不写 `#[link]`（铁律）
//!
//! 本仓 `prism-gpu` 的两处 loader 用 `#[link(name = "dl")]` 声明系统 `libdl`。
//! **本模块刻意不照抄**：`dlopen` / `dlsym` / `dlclose` / `dlerror` 一律用**裸
//! `extern "C"`**，符号由 glibc 提供——glibc ≥ 2.34（Ubuntu 22.04+）已把 `libdl`
//! 并入 `libc`，而 Rust 的 `std` 必然链接 `libc`，故裸声明即可解析，**无需任何
//! 原生链接指令**。
//!
//! 已知边界（如实登记，不改方案）：目标机 glibc < 2.34 且未装 `libdl.so` 时，
//! 链接期会报 `undefined reference to dlopen`。此时**不得**改用 `#[link]`
//! （那会破"无原生链接"的交付硬约束），按"该平台不支持本档"处理并登记——本模块
//! 的降级路径不因此改变：拿不到句柄就是结构化 `Unsupported`，不 panic、不静默。
//!
//! # 失败路径的形状（验收要求：路径 + errno/错误证据）
//!
//! ```text
//! bind_from("libva-absent.so.2")
//!   ⇒ Err(LibraryNotLoadable {
//!        library: "libva-absent.so.2",
//!        errno:   Some(2),                       // dlopen 返回 NULL 后**立即**捕获
//!        errno_text: "No such file or directory (os error 2)",
//!        detail:  "libva-absent.so.2: cannot open shared object file: ..."  // dlerror()
//!      })
//! ```
//!
//! 顺序有讲究：`errno` 必须在 `dlopen` 返回后、`dlerror()` 之前读——`dlerror()`
//! 自身可能改写 `errno`。读取 errno 用 `std::io::Error::last_os_error()`
//! （`std` 已有能力，**不需要** `__errno_location` 的额外 `extern`）。

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// libdl：裸 `extern "C"`（**无 `#[link]`**，见模块文档）
// ---------------------------------------------------------------------------

/// `RTLD_LAZY`（glibc `<dlfcn.h>` 取值 `0x0001`）。
///
/// POSIX 不规定数值，只规定存在；本模块目标平台是 Linux，取 glibc 值。
/// 选 `LAZY` 而非 `NOW`：本档只要符号可解析，不想为一个用不到的符号付
/// 立即重定位的代价（探测失败时也更快返回）。
const RTLD_LAZY: c_int = 0x0001;

/// `RTLD_LOCAL`（glibc `<dlfcn.h>` 取值 `0x0000`）：`libva` 的符号**不**污染
/// 全局符号表，避免与调用方自己的 VA-API 绑定串味。
const RTLD_LOCAL: c_int = 0x0000;

extern "C" {
    /// `void *dlopen(const char *filename, int flag);`
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    /// `void *dlsym(void *handle, const char *symbol);`
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    /// `int dlclose(void *handle);`
    fn dlclose(handle: *mut c_void) -> c_int;
    /// `char *dlerror(void);`
    fn dlerror() -> *const c_char;
}

// ---------------------------------------------------------------------------
// VA-API 基础类型（与 `<va/va.h>` 对齐）
// ---------------------------------------------------------------------------

/// `VAStatus`：`0`（`VA_STATUS_SUCCESS`）表示成功。
pub type VaStatus = c_int;

/// `VADisplay`：`struct _VADisplayContext *`，在 FFI 层是不透明指针。
///
/// 本档**从不构造**它：`VADisplay` 由 `vaGetDisplayDRM`（`libva-drm`）产出，
/// 属第三档（会话建立）。
pub type VaDisplay = *mut c_void;

/// `VAProfile`（`<va/va.h>` 枚举，C 层为 `int`）。
pub type VaProfile = c_int;

/// `VAEntrypoint`（`<va/va.h>` 枚举，C 层为 `int`）。
pub type VaEntrypoint = c_int;

/// `VAConfigID`（`<va/va.h>`：`unsigned int`）。
pub type VaConfigId = c_uint;

/// `VAContextID`（`<va/va.h>`：`unsigned int`）。
pub type VaContextId = c_uint;

/// `VASurfaceID`（`<va/va.h>`：`unsigned int`）。
pub type VaSurfaceId = c_uint;

/// `VABufferID`（`<va/va.h>`：`unsigned int`）。
pub type VaBufferId = c_uint;

/// `VA_STATUS_SUCCESS`（成功码）。
pub const VA_STATUS_SUCCESS: VaStatus = 0;

/// `VAProfileH264High`（`<va/va.h>` 枚举序；本档**不调用**，供第三档取用）。
pub const VA_PROFILE_H264_HIGH: VaProfile = 7;

/// `VAProfileHEVCMain`（`<va/va.h>` 枚举序；本档**不调用**，供第三档取用）。
pub const VA_PROFILE_HEVC_MAIN: VaProfile = 17;

/// `VAEntrypointEncSlice`（硬件切片编码入口；本档**不调用**）。
pub const VA_ENTRYPOINT_ENC_SLICE: VaEntrypoint = 6;

/// `VAEntrypointEncSliceLP`（低功耗编码入口；本档**不调用**）。
pub const VA_ENTRYPOINT_ENC_SLICE_LP: VaEntrypoint = 8;

/// `VAConfigAttrib`（`<va/va.h>`）。
///
/// 布局：`VAConfigAttribType type;`（`unsigned int`）+ `unsigned int value;`。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VaConfigAttrib {
    /// 属性类型（`VAConfigAttribType`）。
    pub type_: c_uint,
    /// 属性值。
    pub value: c_uint,
}

/// `VAGenericValue` 的载荷联合体（`<va/va.h>` 的匿名 `union`）。
#[repr(C)]
#[derive(Clone, Copy)]
pub union VaGenericValuePayload {
    /// `int i;`
    pub i: c_int,
    /// `float f;`
    pub f: f32,
    /// `void *p;`
    pub p: *mut c_void,
}

/// `VAGenericValue`（`<va/va.h>`）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaGenericValue {
    /// `VAGenericValueType type;`
    pub type_: c_uint,
    /// `union { int i; float f; void *p; } value;`
    pub value: VaGenericValuePayload,
}

/// `VASurfaceAttrib`（`<va/va.h>`）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VaSurfaceAttrib {
    /// `VASurfaceAttribType type;`
    pub type_: c_uint,
    /// `unsigned int flags;`
    pub flags: c_uint,
    /// `VAGenericValue value;`
    pub value: VaGenericValue,
}

// ---------------------------------------------------------------------------
// 函数指针类型（`unsafe extern "C" fn`：调用即 unsafe）
// ---------------------------------------------------------------------------

/// `VAStatus vaInitialize(VADisplay dpy, int *major_version, int *minor_version);`
pub type PfnVaInitialize = unsafe extern "C" fn(
    VaDisplay,
    *mut c_int,
    *mut c_int,
) -> VaStatus;
/// `VAStatus vaTerminate(VADisplay dpy);`
pub type PfnVaTerminate = unsafe extern "C" fn(VaDisplay) -> VaStatus;
/// `VAStatus vaQueryConfigProfiles(VADisplay dpy, VAProfile *list, int *num);`
pub type PfnVaQueryConfigProfiles = unsafe extern "C" fn(
    VaDisplay,
    *mut VaProfile,
    *mut c_int,
) -> VaStatus;
/// `VAStatus vaQueryConfigEntrypoints(VADisplay, VAProfile, VAEntrypoint *, int *);`
pub type PfnVaQueryConfigEntrypoints = unsafe extern "C" fn(
    VaDisplay,
    VaProfile,
    *mut VaEntrypoint,
    *mut c_int,
) -> VaStatus;
/// `VAStatus vaCreateConfig(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib *, int, VAConfigID *);`
pub type PfnVaCreateConfig = unsafe extern "C" fn(
    VaDisplay,
    VaProfile,
    VaEntrypoint,
    *mut VaConfigAttrib,
    c_int,
    *mut VaConfigId,
) -> VaStatus;
/// `VAStatus vaDestroyConfig(VADisplay dpy, VAConfigID config_id);`
pub type PfnVaDestroyConfig = unsafe extern "C" fn(VaDisplay, VaConfigId) -> VaStatus;
/// `VAStatus vaCreateContext(VADisplay, VAConfigID, int, int, int, VASurfaceID *, int, VAContextID *);`
pub type PfnVaCreateContext = unsafe extern "C" fn(
    VaDisplay,
    VaConfigId,
    c_int,
    c_int,
    c_int,
    *mut VaSurfaceId,
    c_int,
    *mut VaContextId,
) -> VaStatus;
/// `VAStatus vaDestroyContext(VADisplay dpy, VAContextID context);`
pub type PfnVaDestroyContext = unsafe extern "C" fn(VaDisplay, VaContextId) -> VaStatus;
/// `VAStatus vaCreateSurfaces(VADisplay, unsigned int, unsigned int, unsigned int,
///                            VASurfaceID *, unsigned int, VASurfaceAttrib *, unsigned int);`
pub type PfnVaCreateSurfaces = unsafe extern "C" fn(
    VaDisplay,
    c_uint,
    c_uint,
    c_uint,
    *mut VaSurfaceId,
    c_uint,
    *mut VaSurfaceAttrib,
    c_uint,
) -> VaStatus;
/// `VAStatus vaDestroySurfaces(VADisplay, VASurfaceID *, int);`
pub type PfnVaDestroySurfaces = unsafe extern "C" fn(
    VaDisplay,
    *mut VaSurfaceId,
    c_int,
) -> VaStatus;
/// `VAStatus vaBeginPicture(VADisplay, VAContextID, VASurfaceID);`
pub type PfnVaBeginPicture = unsafe extern "C" fn(VaDisplay, VaContextId, VaSurfaceId) -> VaStatus;
/// `VAStatus vaRenderPicture(VADisplay, VAContextID, VABufferID *, int);`
pub type PfnVaRenderPicture = unsafe extern "C" fn(
    VaDisplay,
    VaContextId,
    *mut VaBufferId,
    c_int,
) -> VaStatus;
/// `VAStatus vaEndPicture(VADisplay dpy, VAContextID context);`
pub type PfnVaEndPicture = unsafe extern "C" fn(VaDisplay, VaContextId) -> VaStatus;
/// `VAStatus vaCreateBuffer(VADisplay, VAContextID, VABufferType, unsigned int,
///                          unsigned int, void *, VABufferID *);`
pub type PfnVaCreateBuffer = unsafe extern "C" fn(
    VaDisplay,
    VaContextId,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut VaBufferId,
) -> VaStatus;
/// `VAStatus vaDestroyBuffer(VADisplay dpy, VABufferID buffer_id);`
pub type PfnVaDestroyBuffer = unsafe extern "C" fn(VaDisplay, VaBufferId) -> VaStatus;
/// `VAStatus vaMapBuffer(VADisplay dpy, VABufferID buf_id, void **pbuf);`
pub type PfnVaMapBuffer = unsafe extern "C" fn(VaDisplay, VaBufferId, *mut *mut c_void) -> VaStatus;
/// `VAStatus vaUnmapBuffer(VADisplay dpy, VABufferID buf_id);`
pub type PfnVaUnmapBuffer = unsafe extern "C" fn(VaDisplay, VaBufferId) -> VaStatus;
/// `VAStatus vaSyncSurface(VADisplay dpy, VASurfaceID render_target);`
pub type PfnVaSyncSurface = unsafe extern "C" fn(VaDisplay, VaSurfaceId) -> VaStatus;

// ---------------------------------------------------------------------------
// 符号清单
// ---------------------------------------------------------------------------

/// 核心符号名（顺序与 [`VaapiFnTable`] 字段一致，也是证据串里的呈现顺序）。
///
/// 这 18 个都是 `libva` 2.x **自身导出**的入口（`libva.so.2`）；`vaGetDisplay`
/// 不在此列——它在 `libva-x11` / `libva-wayland` / `libva-drm` 里各有一份实现，
/// 属第三档（会话建立）的依赖，本档刻意不碰。
pub const REQUIRED_SYMBOLS: [&str; 18] = [
    "vaInitialize",
    "vaTerminate",
    "vaQueryConfigProfiles",
    "vaQueryConfigEntrypoints",
    "vaCreateConfig",
    "vaDestroyConfig",
    "vaCreateContext",
    "vaDestroyContext",
    "vaCreateSurfaces",
    "vaDestroySurfaces",
    "vaBeginPicture",
    "vaRenderPicture",
    "vaEndPicture",
    "vaCreateBuffer",
    "vaDestroyBuffer",
    "vaMapBuffer",
    "vaUnmapBuffer",
    "vaSyncSurface",
];

/// 默认加载的库：`libva.so.2`（`libva` 2.x 的 SONAME，各发行版一致）。
pub const LIBVA_SONAME: &str = "libva.so.2";

// ---------------------------------------------------------------------------
// 错误（结构化，永远带证据）
// ---------------------------------------------------------------------------

/// `dlopen` 失败的成因分类。
///
/// **由 `dlerror()` 原文判读**（不是猜 errno）：实测 glibc 的 `dlopen` 失败时
/// `errno` 常保持 `0`（loader 只在 `dlerror` 里给细节），因此 errno 只能当"有则更好"
/// 的旁证，真正的证据是 `dlerror` 文本。分类只是把这段文本变成可判定 token，
/// 原文永远一并保留。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VaapiOpenCause {
    /// 库文件找不到（`cannot open shared object file: No such file or directory`）。
    NotFound,
    /// 无读取权限（`Permission denied`）。
    PermissionDenied,
    /// ELF 架构/格式不符（`wrong ELF class` / `invalid ELF header`）。
    WrongElfClass,
    /// 依赖库缺失（`error while loading shared libraries` / `undefined symbol`）。
    MissingDependency,
    /// 其它（保留 `dlerror` 原文，不做过度判读）。
    Other,
}

impl VaapiOpenCause {
    /// 稳定的机器可判定 token。
    pub fn as_str(self) -> &'static str {
        match self {
            VaapiOpenCause::NotFound => "not-found",
            VaapiOpenCause::PermissionDenied => "permission-denied",
            VaapiOpenCause::WrongElfClass => "wrong-elf-class",
            VaapiOpenCause::MissingDependency => "missing-dependency",
            VaapiOpenCause::Other => "other",
        }
    }

    /// 从 `dlerror()` 原文判读成因（大小写不敏感；无文本 ⇒ [`VaapiOpenCause::Other`]）。
    ///
    /// 判读顺序有讲究：glibc 的"依赖缺失"消息里**也**含 `cannot open shared object
    /// file`，故必须先认 `error while loading shared libraries` 这个更具体的前缀，
    /// 否则会把"依赖缺"误报成"本库不存在"。
    pub fn classify(dlerror_text: &str) -> VaapiOpenCause {
        let low: String = dlerror_text.to_ascii_lowercase();
        if low.contains("error while loading shared libraries") || low.contains("undefined symbol") {
            return VaapiOpenCause::MissingDependency;
        }
        if low.contains("permission denied") {
            return VaapiOpenCause::PermissionDenied;
        }
        if low.contains("wrong elf class")
            || low.contains("invalid elf header")
            || low.contains("file too short")
        {
            return VaapiOpenCause::WrongElfClass;
        }
        if low.contains("no such file or directory") || low.contains("cannot open shared object file")
        {
            // glibc 把"文件不存在"与"打不开"都归到这条消息；对运维是同一处理。
            return VaapiOpenCause::NotFound;
        }
        VaapiOpenCause::Other
    }
}

impl core::fmt::Display for VaapiOpenCause {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 绑定失败的结构化理由。**每个变体都带可复核的证据**（路径 / 成因 / errno / 符号名）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaapiBindError {
    /// `dlopen` 返回空指针（库不存在 / 无权限 / 架构不符 / 依赖缺失）。
    LibraryNotLoadable {
        /// 实际尝试的库名或路径（原样回显，便于复核）。
        library: String,
        /// `dlopen` 失败后**立即**读到的 errno；为 `0` 时归一化为 `None`
        /// （glibc 的 loader 常不设置它，见 [`VaapiOpenCause`] 的说明）。
        errno: Option<i32>,
        /// errno 的人读文本（errno 缺席时为空串）。
        errno_text: String,
        /// `dlerror()` 的原文（可能为空：glibc 之外不保证有文本）。
        detail: String,
    },
    /// 库已加载但某个核心符号解析失败（版本过老 / 非 va 库 / ABI 不符）。
    SymbolMissing {
        /// 已成功 `dlopen` 的库名。
        library: String,
        /// 缺失的符号名。
        symbol: &'static str,
        /// `dlerror()` 的原文（可能为空）。
        detail: String,
    },
}

impl VaapiBindError {
    /// 结构化的**一行证据串**（保证非空：`dlerror` 无文本时也写明"无输出"）。
    pub fn evidence(&self) -> String {
        match self {
            VaapiBindError::LibraryNotLoadable { library, errno, errno_text, detail } => {
                let cause: VaapiOpenCause = VaapiOpenCause::classify(detail);
                let errno_part: String = match errno {
                    Some(code) => {
                        let text: &str =
                            if errno_text.is_empty() { "无文本" } else { errno_text.as_str() };
                        format!("{code}（{text}）")
                    }
                    // 归一路径：errno 为 0 / 缺席时如实说"未设置"，不编数字。
                    None => "未设置（loader 只在 dlerror 里给细节）".to_string(),
                };
                let detail: &str =
                    if detail.is_empty() { "<dlerror 无输出>" } else { detail.as_str() };
                format!(
                    "dlopen(\"{library}\") 失败：cause={cause}；errno={errno_part}；dlerror={detail}"
                )
            }
            VaapiBindError::SymbolMissing { library, symbol, detail } => {
                let detail: &str =
                    if detail.is_empty() { "<dlerror 无输出>" } else { detail.as_str() };
                format!("dlopen(\"{library}\") 成功，但 dlsym(\"{symbol}\") 失败：{detail}")
            }
        }
    }

    /// 失败的库名/路径（原样）。
    pub fn library(&self) -> &str {
        match self {
            VaapiBindError::LibraryNotLoadable { library, .. }
            | VaapiBindError::SymbolMissing { library, .. } => library.as_str(),
        }
    }

    /// 是否属于"库都打不开"（相对"打开了但符号缺"）。
    pub fn is_library_not_loadable(&self) -> bool {
        matches!(self, VaapiBindError::LibraryNotLoadable { .. })
    }

    /// 库打不开时的成因分类；符号缺失情形返回
    /// [`VaapiOpenCause::Other`]（那不属于 `dlopen` 失败面）。
    pub fn cause(&self) -> VaapiOpenCause {
        match self {
            VaapiBindError::LibraryNotLoadable { detail, .. } => {
                VaapiOpenCause::classify(detail)
            }
            VaapiBindError::SymbolMissing { .. } => VaapiOpenCause::Other,
        }
    }
}

impl core::fmt::Display for VaapiBindError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.evidence())
    }
}

impl std::error::Error for VaapiBindError {}

// ---------------------------------------------------------------------------
// 动态库句柄
// ---------------------------------------------------------------------------

/// `dlopen` 得到的库句柄，`Drop` 时 `dlclose`。
///
/// 不实现 `Clone`：句柄的关闭责任必须**唯一**，否则会出现二次 `dlclose`。
struct VaapiLibrary {
    /// 库名/路径（证据用）。
    name: String,
    /// `dlopen` 句柄；构造成功时保证非空。
    handle: *mut c_void,
}

impl VaapiLibrary {
    /// 打开库。失败 ⇒ 带路径 + errno + `dlerror` 的结构化错误。
    fn open(name: &str) -> Result<VaapiLibrary, VaapiBindError> {
        // 先清一次错误状态：`dlerror` 有"自上次调用以来"的状态语义，
        // 不清会让随后的失败原因归错到别处。
        let _ = dl_error_text();

        let cname: CString = match CString::new(name) {
            Ok(v) => v,
            Err(_) => {
                // 库名里含内嵌 NUL：这是调用方给的非法 C 字符串，按"打不开"处理，
                // 不 panic。
                return Err(VaapiBindError::LibraryNotLoadable {
                    library: name.to_string(),
                    errno: None,
                    errno_text: format!("库名含内嵌 NUL（非法 C 字符串，长度 {}）", name.len()),
                    detail: String::new(),
                });
            }
        };

        // SAFETY: `cname.as_ptr()` 是合法 NUL 结尾 C 字符串，在本次调用期间有效；
        // `dlopen` 失败返回空指针，我们随即判空，不做解引用。
        let handle: *mut c_void = unsafe { dlopen(cname.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };

        if handle.is_null() {
            // 顺序关键：errno 必须在 dlerror() **之前**读（dlerror 可能改 errno）。
            // 实测：glibc 的 loader 失败时并不保证设置 errno（此处常为 0），故把
            // 0 归一化为 None——宁可如实说"未设置"，也不把 `Success (os error 0)`
            // 这种自相矛盾的串当证据；真正的证据是下面的 dlerror 原文。
            let os_err: std::io::Error = std::io::Error::last_os_error();
            let errno: Option<i32> = os_err.raw_os_error().filter(|code| *code != 0);
            let errno_text: String = match errno {
                Some(_) => os_err.to_string(),
                None => String::new(),
            };
            let detail: String = dl_error_text();
            return Err(VaapiBindError::LibraryNotLoadable {
                library: name.to_string(),
                errno,
                errno_text,
                detail,
            });
        }

        Ok(VaapiLibrary { name: name.to_string(), handle })
    }
}

impl Drop for VaapiLibrary {
    /// 释放句柄。`dlclose` 的返回值只反映引用计数，对本进程无副作用，忽略即可
    /// ——**不 panic**。
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: 句柄来自 dlopen 且只在此处关闭一次（类型不实现 Clone，
            // 关闭责任唯一）；`self` 即将析构，不会再有调用方使用它。
            let _ = unsafe { dlclose(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// 读一次 `dlerror()` 并清空错误状态（空串表示"无悬置错误"）。
fn dl_error_text() -> String {
    // SAFETY: `dlerror` 返回线程局部缓冲区的指针（或空）；只在本次调用内转成
    // `String`，不跨调用保存。
    let raw: *const c_char = unsafe { dlerror() };
    if raw.is_null() {
        return String::new();
    }
    // SAFETY: 非空即指向 NUL 结尾字符串，且在下一次 dlerror 调用前保持有效。
    unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

/// `dlsym` 一个符号并转成函数指针 `F`；符号缺失返回 `None`（不 panic）。
///
/// # Safety
/// 调用方须保证 `handle` 是有效库句柄，且 `name` 指向符号的真实 ABI 与 `F` 一致。
unsafe fn load_symbol<F: Copy>(handle: *mut c_void, name: &str) -> Option<F> {
    let cname: CString = CString::new(name).ok()?;
    let _ = dl_error_text();
    let sym: *mut c_void = dlsym(handle, cname.as_ptr());
    if sym.is_null() {
        return None;
    }
    // 函数指针与数据指针在目标 ABI 上同宽；`transmute_copy` 避免
    // `mem::transmute` 对"目标类型大小未知"的编译期断言。
    Some(std::mem::transmute_copy::<*mut c_void, F>(&sym))
}

// ---------------------------------------------------------------------------
// 函数表（第二档的全部产出）
// ---------------------------------------------------------------------------

/// VA-API 核心函数表。**只解析，不调用**（见模块文档"三个不做"）。
///
/// 持有 `dlopen` 句柄：表活着的期间库保持加载，表 `Drop` 时 `dlclose`。
/// 任何字段都在 `Drop` 后失效，故表不可 `Clone`、不可跨线程随意传递——
/// 正常用法是"用完即走"。
pub struct VaapiFnTable {
    /// 库句柄（**字段序在前**：保证先于函数指针析构）。
    library: VaapiLibrary,
    /// `VAStatus vaInitialize(VADisplay, int *, int *);`
    pub va_initialize: PfnVaInitialize,
    /// `VAStatus vaTerminate(VADisplay);`
    pub va_terminate: PfnVaTerminate,
    /// `VAStatus vaQueryConfigProfiles(VADisplay, VAProfile *, int *);`
    pub va_query_config_profiles: PfnVaQueryConfigProfiles,
    /// `VAStatus vaQueryConfigEntrypoints(VADisplay, VAProfile, VAEntrypoint *, int *);`
    pub va_query_config_entrypoints: PfnVaQueryConfigEntrypoints,
    /// `VAStatus vaCreateConfig(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib *, int, VAConfigID *);`
    pub va_create_config: PfnVaCreateConfig,
    /// `VAStatus vaDestroyConfig(VADisplay, VAConfigID);`
    pub va_destroy_config: PfnVaDestroyConfig,
    /// `VAStatus vaCreateContext(VADisplay, VAConfigID, int, int, int, VASurfaceID *, int, VAContextID *);`
    pub va_create_context: PfnVaCreateContext,
    /// `VAStatus vaDestroyContext(VADisplay, VAContextID);`
    pub va_destroy_context: PfnVaDestroyContext,
    /// `VAStatus vaCreateSurfaces(VADisplay, unsigned int, unsigned int, unsigned int, VASurfaceID *, unsigned int, VASurfaceAttrib *, unsigned int);`
    pub va_create_surfaces: PfnVaCreateSurfaces,
    /// `VAStatus vaDestroySurfaces(VADisplay, VASurfaceID *, int);`
    pub va_destroy_surfaces: PfnVaDestroySurfaces,
    /// `VAStatus vaBeginPicture(VADisplay, VAContextID, VASurfaceID);`
    pub va_begin_picture: PfnVaBeginPicture,
    /// `VAStatus vaRenderPicture(VADisplay, VAContextID, VABufferID *, int);`
    pub va_render_picture: PfnVaRenderPicture,
    /// `VAStatus vaEndPicture(VADisplay, VAContextID);`
    pub va_end_picture: PfnVaEndPicture,
    /// `VAStatus vaCreateBuffer(VADisplay, VAContextID, VABufferType, unsigned int, unsigned int, void *, VABufferID *);`
    pub va_create_buffer: PfnVaCreateBuffer,
    /// `VAStatus vaDestroyBuffer(VADisplay, VABufferID);`
    pub va_destroy_buffer: PfnVaDestroyBuffer,
    /// `VAStatus vaMapBuffer(VADisplay, VABufferID, void **);`
    pub va_map_buffer: PfnVaMapBuffer,
    /// `VAStatus vaUnmapBuffer(VADisplay, VABufferID);`
    pub va_unmap_buffer: PfnVaUnmapBuffer,
    /// `VAStatus vaSyncSurface(VADisplay, VASurfaceID);`
    pub va_sync_surface: PfnVaSyncSurface,
}

impl VaapiFnTable {
    /// 本表来自哪个库（原样库名/路径）。
    pub fn library_name(&self) -> &str {
        self.library.name.as_str()
    }

    /// 已解析的符号数（恒定等于 [`REQUIRED_SYMBOLS`] 的长度——缺一个就构造不出表）。
    pub fn resolved_count(&self) -> usize {
        REQUIRED_SYMBOLS.len()
    }

    /// 表覆盖的符号名（顺序固定，证据/日志用）。
    pub fn resolved_symbols(&self) -> &'static [&'static str] {
        &REQUIRED_SYMBOLS
    }
}

impl core::fmt::Debug for VaapiFnTable {
    /// 只打印库名与符号数——**不打印函数指针地址**（避免日志里出现无法复核的
    /// 随机地址，也让双跑输出可逐字节比对）。
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VaapiFnTable")
            .field("library", &self.library.name)
            .field("resolved_count", &self.resolved_count())
            .finish()
    }
}

/// 打开 `library` 并解析全部核心符号。失败 ⇒ 结构化错误（不 panic）。
fn open_and_resolve(library: &str) -> Result<VaapiFnTable, VaapiBindError> {
    let lib: VaapiLibrary = VaapiLibrary::open(library)?;
    let handle: *mut c_void = lib.handle;

    // 逐个解析；任一缺失 ⇒ 带库名 + 符号名 + dlerror 的错误。
    // `need!` 展开成"取符号 / 缺失即早返回"，避免长串 `?` 丢失符号名。
    macro_rules! need {
        ($name:literal, $ty:ty) => {
            // SAFETY: `handle` 是刚 dlopen 成功的有效句柄，`$name` 是核心符号的
            // 真实名字，`$ty` 是 `<va/va.h>` 里该符号的精确签名。
            match unsafe { load_symbol::<$ty>(handle, $name) } {
                Some(f) => f,
                None => {
                    let detail: String = dl_error_text();
                    return Err(VaapiBindError::SymbolMissing {
                        library: library.to_string(),
                        symbol: $name,
                        detail,
                    });
                }
            }
        };
    }

    Ok(VaapiFnTable {
        va_initialize: need!("vaInitialize", PfnVaInitialize),
        va_terminate: need!("vaTerminate", PfnVaTerminate),
        va_query_config_profiles: need!("vaQueryConfigProfiles", PfnVaQueryConfigProfiles),
        va_query_config_entrypoints: need!(
            "vaQueryConfigEntrypoints",
            PfnVaQueryConfigEntrypoints
        ),
        va_create_config: need!("vaCreateConfig", PfnVaCreateConfig),
        va_destroy_config: need!("vaDestroyConfig", PfnVaDestroyConfig),
        va_create_context: need!("vaCreateContext", PfnVaCreateContext),
        va_destroy_context: need!("vaDestroyContext", PfnVaDestroyContext),
        va_create_surfaces: need!("vaCreateSurfaces", PfnVaCreateSurfaces),
        va_destroy_surfaces: need!("vaDestroySurfaces", PfnVaDestroySurfaces),
        va_begin_picture: need!("vaBeginPicture", PfnVaBeginPicture),
        va_render_picture: need!("vaRenderPicture", PfnVaRenderPicture),
        va_end_picture: need!("vaEndPicture", PfnVaEndPicture),
        va_create_buffer: need!("vaCreateBuffer", PfnVaCreateBuffer),
        va_destroy_buffer: need!("vaDestroyBuffer", PfnVaDestroyBuffer),
        va_map_buffer: need!("vaMapBuffer", PfnVaMapBuffer),
        va_unmap_buffer: need!("vaUnmapBuffer", PfnVaUnmapBuffer),
        va_sync_surface: need!("vaSyncSurface", PfnVaSyncSurface),
        library: lib,
    })
}

// ---------------------------------------------------------------------------
// 公共入口
// ---------------------------------------------------------------------------

/// 打开默认库 [`LIBVA_SONAME`]（`libva.so.2`）并解析全部核心符号。
///
/// 成功 ⇒ [`VaapiFnTable`]（**只解析，不调用**）；失败 ⇒ [`VaapiBindError`]
/// （带库名/路径 + errno + `dlerror` 文本）。**任何路径都不 panic**。
///
/// # Errors
/// 库无法 `dlopen`，或某个核心符号 `dlsym` 不到。
pub fn bind() -> Result<VaapiFnTable, VaapiBindError> {
    open_and_resolve(LIBVA_SONAME)
}

/// 同 [`bind`]，但可指定库名或**绝对路径**（老发行版的其它 SONAME、测试用的
/// 不存在路径）。失败路径的实测证据就靠这个入口取得。
///
/// # Errors
/// 同 [`bind`]。
pub fn bind_from(library: &str) -> Result<VaapiFnTable, VaapiBindError> {
    open_and_resolve(library)
}

/// 绑定探测的**事实报告**（可 `Clone`、可比较、可入档）：不持有句柄。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaapiBindReport {
    /// 尝试加载的库名/路径。
    pub library: String,
    /// `dlopen` 是否成功（符号缺失时仍为 `true`）。
    pub loaded: bool,
    /// 已解析的符号名（成功时等于 [`REQUIRED_SYMBOLS`]；失败时为空）。
    pub resolved: Vec<&'static str>,
    /// 失败理由（成功时为 `None`）。
    pub error: Option<VaapiBindError>,
}

impl VaapiBindReport {
    /// 是否达成**第二档**（绑定就绪）：库打开 + 全部核心符号解析成功。
    ///
    /// 注意：这是"绑定就绪"，**不是**"能编码"（见模块文档）。
    pub fn is_ready(&self) -> bool {
        self.error.is_none() && self.resolved.len() == REQUIRED_SYMBOLS.len()
    }

    /// 一行事实摘要（**保证非空**，可直接拼进探测理由/日志）。
    pub fn summary(&self) -> String {
        if self.is_ready() {
            format!(
                "{} 已 dlopen，{}/{} 核心符号已解析（未调用任何入口）",
                self.library,
                self.resolved.len(),
                REQUIRED_SYMBOLS.len()
            )
        } else {
            match &self.error {
                Some(err) => err.evidence(),
                // 理论上不可达（失败必有 error）；仍然给非空文本，杜绝静默。
                None => format!(
                    "{} 绑定未就绪：已解析 {}/{} 符号（无错误对象，疑为内部状态不一致）",
                    self.library,
                    self.resolved.len(),
                    REQUIRED_SYMBOLS.len()
                ),
            }
        }
    }

    /// 报告里所有已解析符号名（失败时为空切片）。
    pub fn resolved_symbols(&self) -> &[&'static str] {
        &self.resolved
    }
}

/// 探测默认库的绑定状态，结果**进程内只算一次**并缓存（缓存的是事实字符串，
/// 不缓存句柄）。
///
/// 用 `OnceLock` 而不是每次重算：`probe()` 契约要求可重复、无副作用；反复
/// `dlopen`/`dlclose` 虽无持久副作用，但没有理由让"读状态"变成"反复加载库"。
pub fn bind_report() -> &'static VaapiBindReport {
    static CACHE: OnceLock<VaapiBindReport> = OnceLock::new();
    CACHE.get_or_init(|| {
        let report: VaapiBindReport = match bind() {
            Ok(table) => VaapiBindReport {
                library: table.library_name().to_string(),
                loaded: true,
                resolved: table.resolved_symbols().to_vec(),
                error: None,
            },
            Err(err) => VaapiBindReport {
                library: err.library().to_string(),
                loaded: !err.is_library_not_loadable(),
                resolved: Vec::new(),
                error: Some(err),
            },
        };
        // `match` 结束时 `table` 已析构 ⇒ 句柄已 dlclose：报告只留事实，不长驻句柄。
        report
    })
}

// ---------------------------------------------------------------------------
// 测试（单元层：纯逻辑 + 结构布局 + 失败路径；跑法见 crate 根文档）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_symbol_list_is_unique_and_nonempty() {
        assert_eq!(REQUIRED_SYMBOLS.len(), 18, "核心符号应为 18 个（与函数表字段一一对应）");
        for name in REQUIRED_SYMBOLS {
            assert!(name.starts_with("va"), "VA-API 符号一律以 va 开头：{name}");
            assert!(!name.is_empty());
        }
        let mut sorted: Vec<&str> = REQUIRED_SYMBOLS.to_vec();
        sorted.sort_unstable();
        let before: usize = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "符号名不得重复");
    }

    #[test]
    fn required_symbol_list_covers_the_authoritative_core_api() {
        // 逐条钉住：这 14 个是绑定层必须覆盖的核心面（会话建立起点的最小集）。
        for must in [
            "vaInitialize",
            "vaTerminate",
            "vaQueryConfigProfiles",
            "vaCreateConfig",
            "vaDestroyConfig",
            "vaCreateContext",
            "vaDestroyContext",
            "vaCreateSurfaces",
            "vaDestroySurfaces",
            "vaBeginPicture",
            "vaRenderPicture",
            "vaEndPicture",
            "vaMapBuffer",
            "vaUnmapBuffer",
        ] {
            assert!(
                REQUIRED_SYMBOLS.contains(&must),
                "核心符号 {must} 必须在表里（当前：{REQUIRED_SYMBOLS:?}）"
            );
        }
        // `vaGetDisplay` 不在 libva.so.2 里（在 libva-drm / libva-x11），刻意不列。
        assert!(
            !REQUIRED_SYMBOLS.contains(&"vaGetDisplay"),
            "vaGetDisplay 属 libva-drm/x11，第二档不得声称已解析"
        );
    }

    #[test]
    fn vaapi_struct_layouts_match_the_c_header_on_64bit() {
        // `<va/va.h>` 的 ABI 形状：VAConfigAttrib = 2×u32；VAGenericValue =
        // u32 + padding + (u32/f32/void*) 联合；VASurfaceAttrib = 2×u32 + VAGenericValue。
        if core::mem::size_of::<usize>() != 8 {
            return; // 32 位目标的指针宽度不同，本节不适用（不猜、不硬套）。
        }
        assert_eq!(core::mem::size_of::<VaConfigAttrib>(), 8);
        assert_eq!(core::mem::size_of::<VaGenericValue>(), 16);
        assert_eq!(core::mem::size_of::<VaSurfaceAttrib>(), 24);
        assert_eq!(core::mem::size_of::<VaStatus>(), 4);
        assert_eq!(core::mem::size_of::<VaConfigId>(), 4);
        assert_eq!(core::mem::size_of::<VaSurfaceId>(), 4);
    }

    #[test]
    fn missing_library_produces_structured_error_with_path_and_errno() {
        // 失败路径的核心钉子：不 panic、不静默，且**带路径 + errno/错误证据**。
        let missing: &str = "libva-absent-for-binding-test.so.2";
        let err: VaapiBindError = match bind_from(missing) {
            Ok(_) => panic!("库 {missing} 不该存在；失败路径没被走到"),
            Err(e) => e,
        };
        let text: String = err.evidence();
        assert!(!text.is_empty(), "证据串不得为空");
        assert!(text.contains(missing), "证据串必须带库名/路径：{text}");
        assert!(text.contains("errno="), "证据串必须带 errno 位：{text}");
        assert!(text.contains("dlerror="), "证据串必须带 dlerror 原文位：{text}");
        assert!(text.contains("cause=not-found"), "证据串必须给出成因分类：{text}");
        assert_eq!(err.library(), missing, "库名必须原样回显");
        assert_eq!(err.cause(), VaapiOpenCause::NotFound, "不存在的库应判为 not-found");
        match err {
            VaapiBindError::LibraryNotLoadable { errno, errno_text, detail, .. } => {
                // glibc 的 loader 不保证设 errno（实测常为 0，已归一化为 None）；
                // 若它给了值，就必须是 ENOENT。绝不允许出现自相矛盾的 0。
                assert!(
                    errno.is_none() || errno == Some(2),
                    "errno 只能是未设置或 ENOENT，实得 {errno:?}"
                );
                assert_ne!(errno, Some(0), "errno=0 是 loader 没设置，必须归一化为 None");
                if errno.is_some() {
                    assert!(!errno_text.is_empty(), "有 errno 就必须有文本");
                }
                assert!(
                    detail.contains("No such file or directory")
                        || detail.contains("cannot open shared object file"),
                    "dlerror 原文必须说清打不开的原因：{detail:?}"
                );
            }
            VaapiBindError::SymbolMissing { symbol, .. } => {
                panic!("不存在的库不可能解析到符号 {symbol}——这是 dlopen 失败路径")
            }
        }
    }

    #[test]
    fn open_cause_classification_covers_the_glibc_message_families() {
        // 纯文本判读：不猜 errno，只把 dlerror 的既有消息族映射成 token。
        assert_eq!(
            VaapiOpenCause::classify(
                "libva.so.2: cannot open shared object file: No such file or directory"
            ),
            VaapiOpenCause::NotFound
        );
        assert_eq!(
            VaapiOpenCause::classify("libva.so.2: cannot open shared object file: Permission denied"),
            VaapiOpenCause::PermissionDenied
        );
        assert_eq!(
            VaapiOpenCause::classify("libva.so.2: wrong ELF class: ELFCLASS32"),
            VaapiOpenCause::WrongElfClass
        );
        // 依赖缺失的消息里也含 "cannot open shared object file"：必须先认更具体的前缀。
        assert_eq!(
            VaapiOpenCause::classify(
                "libva.so.2: error while loading shared libraries: libdrm.so.2: cannot open \
                 shared object file: No such file or directory"
            ),
            VaapiOpenCause::MissingDependency
        );
        assert_eq!(VaapiOpenCause::classify(""), VaapiOpenCause::Other);
        assert_eq!(VaapiOpenCause::NotFound.as_str(), "not-found");
        assert_eq!(VaapiOpenCause::Other.to_string(), "other");
    }

    #[test]
    fn missing_library_absolute_path_is_echoed_verbatim() {
        // 绝对路径同样原样回显（便于运维对照 ldconfig / LD_LIBRARY_PATH）。
        let path: &str = "/nonexistent/vaapi-binding-probe/libva.so.2";
        match bind_from(path) {
            Ok(_) => panic!("路径 {path} 不该存在"),
            Err(err) => {
                assert_eq!(err.library(), path);
                assert!(err.is_library_not_loadable());
                assert!(err.to_string().contains(path), "Display 也必须带路径");
            }
        }
    }

    #[test]
    fn bind_report_is_cached_repeatable_and_structurally_sound() {
        let a: &VaapiBindReport = bind_report();
        let b: &VaapiBindReport = bind_report();
        assert_eq!(a, b, "同一进程内两次探测必须逐字段相同（缓存 + 无随机）");
        assert!(!a.summary().is_empty(), "摘要不得为空（失败也要说清理由）");
        assert_eq!(a.library, LIBVA_SONAME, "默认探测对象是 libva.so.2");
        if a.is_ready() {
            assert_eq!(a.resolved.len(), REQUIRED_SYMBOLS.len());
            assert!(a.error.is_none());
            assert!(a.loaded);
            // 再走一次真实绑定：表可构造，且字段数与清单一致。
            let table: VaapiFnTable = bind().unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(table.resolved_count(), REQUIRED_SYMBOLS.len());
            assert_eq!(table.library_name(), LIBVA_SONAME);
            assert_eq!(table.resolved_symbols(), &REQUIRED_SYMBOLS);
            // Debug 不带随机地址（双跑可逐字节比对）。
            let dbg: String = format!("{table:?}");
            assert!(dbg.contains(LIBVA_SONAME), "{dbg}");
            assert!(dbg.contains("resolved_count"), "{dbg}");
        } else {
            assert!(a.error.is_some(), "未就绪必须有结构化理由");
            assert!(a.resolved.is_empty());
        }
    }

    #[test]
    fn report_summary_never_empty_even_with_blank_dlerror() {
        // 构造一个"无 dlerror 文本"的错误，确认摘要仍非空（不静默）。
        let err = VaapiBindError::LibraryNotLoadable {
            library: "libva.so.2".to_string(),
            errno: None,
            errno_text: String::new(),
            detail: String::new(),
        };
        let text: String = err.evidence();
        assert!(text.contains("未设置"), "errno 缺席必须写明未设置：{text}");
        assert!(text.contains("<dlerror 无输出>"), "dlerror 空必须写明：{text}");
        assert!(text.contains("cause=other"), "无文本时成因归 other：{text}");
        let report = VaapiBindReport {
            library: "libva.so.2".to_string(),
            loaded: false,
            resolved: Vec::new(),
            error: Some(err),
        };
        assert!(!report.is_ready());
        assert!(!report.summary().is_empty());
    }

    #[test]
    fn symbol_missing_error_names_the_symbol_and_library() {
        let err = VaapiBindError::SymbolMissing {
            library: "libva.so.2".to_string(),
            symbol: "vaCreateConfig",
            detail: String::new(),
        };
        let text: String = err.evidence();
        assert!(text.contains("vaCreateConfig"), "必须点名缺失符号：{text}");
        assert!(text.contains("libva.so.2"), "必须点名库：{text}");
        assert!(!err.is_library_not_loadable(), "符号缺失 ≠ 库打不开");
    }

    #[test]
    fn rld_constants_match_glibc_definitions() {
        // 平台常量不可臆测：这里钉住本模块采用的 glibc 取值，改动必须显式过审。
        assert_eq!(RTLD_LAZY, 0x0001);
        assert_eq!(RTLD_LOCAL, 0x0000);
    }
}
