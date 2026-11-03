//! 编辑器 wasm C-ABI（手写，零第三方 crate）。
//!
//! # 为什么 ABI 在 prism-app 而不是 prism-gpu
//!
//! D-14 的 harness ABI 落在 `prism-gpu/src/web/abi.rs`，但**编辑器不能放那里**：
//! 编辑器的增量失效依赖 `prism-core` 的**子图签名树**（`graph_signatures` /
//! `cache_key`），而 `prism-gpu` 只依赖 `prism-math`——反向依赖 `prism-core`
//! 会成环（`prism-gpu/src/scan.rs` 模块头明确禁止："故此处**不得反向依赖
//! prism-core**（避免循环依赖）"）。故本 ABI 落在 `prism-app`（已依赖
//! `prism-core`），**对 `abi.rs` 零改动**，也顺带避开与并行轮次的编辑冲突。
//!
//! # 与既有 ABI 同一套约定（照搬 `prism-gpu/src/web/abi.rs` 的哲学）
//!
//! * **零第三方 crate**：不使用 `wasm-bindgen`；导出全部手写
//!   `#[unsafe(no_mangle)] extern "C"`。
//! * **状态码判成败 + 文本缓冲给细节**：`i32` 返回码 + 线性内存 NUL 结尾文本。
//!   WebAssembly 的 JS 侧只能返回数值，无法返回字符串，故沿用同一约定。
//! * **状态码**：`0=PASS 1=FAIL 2=PENDING 3=SKIP 4=BAD_ARG`（与 D-14 一致）。
//! * **裸分配**：[`prism_editor_alloc`] / [`prism_editor_dealloc`]，16 字节对齐，
//!   两侧必须用同一 `len` 释放。
//! * **文本缓冲**：[`prism_editor_json_ptr`] / [`_len`] 等，**下一次调用会覆写**
//!   ——宿主须在调用后立刻读走。
//!
//! # 内存模型
//!
//! 全局状态（[`EditorState`] + [`PreviewRenderer`]）放在
//! [`thread_local`] 的 [`RefCell`] 中：wasm 单线程，无需锁；native 下每个线程
//! 各持一份，测试可并行不互相污染。
//!
//! # 平台行为
//!
//! native 目标下本模块**照常编译并可测**（wasm 导出在 native 上是无害的
//! `extern "C"` 符号），故 `cargo test -p prism-app` 覆盖了全部 ABI 逻辑；
//! 浏览器侧只需 `WebAssembly.instantiate`（无 `env` 导入——本 ABI 是纯导出，
//! 不从 JS 调任何函数，故编辑器页面**零胶水**）。

use std::cell::RefCell;

use prism_core::node::ParamValue;

use super::preview::PreviewRenderer;
use super::state::{EditorState, PreviewQuality};

/// 本 ABI 版本号（宿主加载后先比对，不匹配即拒绝运行）。
pub const EDITOR_ABI_VERSION: u32 = 1;

/// 状态码（与 D-14 `prism-gpu` 的 `status` 模块同值，便于宿主复用同一套判定）。
pub mod status {
    /// 成功。
    pub const PASS: i32 = 0;
    /// 失败；细节见错误缓冲（[`super::prism_editor_error_ptr`]）。
    pub const FAIL: i32 = 1;
    /// 已提交需宿主等待（本 ABI 无异步项，恒不使用；保留以对齐状态码空间）。
    pub const PENDING: i32 = 2;
    /// 跳过（前置条件不满足）。
    pub const SKIP: i32 = 3;
    /// 参数错误（越界 / 空图 / 非法档位）。
    pub const BAD_ARG: i32 = 4;
}

/// 裸分配对齐字节数（与 `prism-gpu` 的 `PRISM_ALLOC_ALIGN` 同值）。
const ALLOC_ALIGN: usize = 16;

/// 文本缓冲容量（含结尾 NUL）。
const TEXT_CAP: usize = 65536;

/// 预览缓冲最大字节数（`MAX_DIM` 见 `preview` 模块）。
///
/// 512x512 RGBA = 1 MiB；此处留 2 MiB 上限，与 [`super::preview::MAX_DIM`]
/// 的单边上限配套（`512*512*4 = 1048576` < 2 MiB，有余量）。
const MAX_PREVIEW_BYTES: usize = 2 * 1024 * 1024;

/// 编辑器 ABI 版本号（宿主加载后先比对）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_abi_version() -> u32 {
    EDITOR_ABI_VERSION
}

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

thread_local! {
    /// 编辑器状态（`None` = 未初始化；宿主须先调 [`prism_editor_init`]）。
    static STATE: RefCell<Option<EditorState>> = const { RefCell::new(None) };
    /// 预览渲染器（跨帧中间缓冲缓存，承载"子图签名增量"）。
    /// 非 `const` 初始化：`BTreeMap::new()` 非 const（见 `PreviewRenderer`）。
    static RENDERER: RefCell<PreviewRenderer> = RefCell::new(PreviewRenderer::new());
    /// 错误文本（NUL 结尾，按 `char` 边界截断，绝不 panic）。
    static ERR: RefCell<[u8; TEXT_CAP]> = const { RefCell::new([0u8; TEXT_CAP]) };
    /// 错误文本有效长度（不含 NUL）。
    static ERR_LEN: RefCell<usize> = const { RefCell::new(0) };
    /// JSON 文本缓冲（节点树 / schema / 编辑序列 / dirty 集共用）。
    static JSON: RefCell<[u8; TEXT_CAP]> = const { RefCell::new([0u8; TEXT_CAP]) };
    /// JSON 文本有效长度（不含 NUL）。
    static JSON_LEN: RefCell<usize> = const { RefCell::new(0) };
    /// 预览像素缓冲（RGBA8；由 [`prism_editor_render`] 填充）。
    static PIXELS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// 上次渲染的统计 JSON（recomputed / reused / spp …，供 UI 展示增量收益）。
    static STATS: RefCell<[u8; TEXT_CAP]> = const { RefCell::new([0u8; TEXT_CAP]) };
    /// 上次渲染统计 JSON 的有效长度。
    static STATS_LEN: RefCell<usize> = const { RefCell::new(0) };
}

/// 写文本到定长缓冲（按 UTF-8 `char` 边界截断，末尾补 NUL）。
///
/// 返回写入的字节数（不含 NUL）。绝不 panic：超长文本被安全截断。
fn write_capped<const N: usize>(
    buf: &'static std::thread::LocalKey<RefCell<[u8; N]>>,
    len: &'static std::thread::LocalKey<RefCell<usize>>,
    s: &str,
) -> usize {
    let mut used = 0usize;
    buf.with(|b| {
        let mut v = b.borrow_mut();
        v.fill(0);
        for ch in s.chars() {
            let mut tmp = [0u8; 4];
            let encoded = ch.encode_utf8(&mut tmp);
            // 留 1 字节给 NUL。
            if used + encoded.len() + 1 > N {
                break;
            }
            v[used..used + encoded.len()].copy_from_slice(encoded.as_bytes());
            used += encoded.len();
        }
    });
    len.with(|l| *l.borrow_mut() = used);
    used
}

/// 写错误文本并返回 [`status::FAIL`]。
fn fail(s: impl AsRef<str>) -> i32 {
    write_capped(&ERR, &ERR_LEN, s.as_ref());
    status::FAIL
}

/// 写错误文本并返回 [`status::BAD_ARG`]。
fn bad_arg(s: impl AsRef<str>) -> i32 {
    write_capped(&ERR, &ERR_LEN, s.as_ref());
    status::BAD_ARG
}

/// 写 JSON 文本（成功路径用）。
fn set_json(s: &str) -> i32 {
    write_capped(&JSON, &JSON_LEN, s);
    status::PASS
}

/// 写统计 JSON。
fn set_stats(s: &str) {
    write_capped(&STATS, &STATS_LEN, s);
}

/// 错误文本地址（NUL 结尾，用 [`prism_editor_error_len`] 取长度）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_error_ptr() -> usize {
    ERR.with(|b| b.borrow().as_ptr() as usize)
}

/// 错误文本字节数（不含 NUL）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_error_len() -> u32 {
    ERR_LEN.with(|l| *l.borrow() as u32)
}

/// JSON 文本地址（用 [`prism_editor_json_len`] 取长度）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_json_ptr() -> usize {
    JSON.with(|b| b.borrow().as_ptr() as usize)
}

/// JSON 文本字节数（不含 NUL）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_json_len() -> u32 {
    JSON_LEN.with(|l| *l.borrow() as u32)
}

/// 渲染统计 JSON 地址（用 [`prism_editor_stats_len`] 取长度）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_stats_ptr() -> usize {
    STATS.with(|b| b.borrow().as_ptr() as usize)
}

/// 渲染统计 JSON 字节数（不含 NUL）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_stats_len() -> u32 {
    STATS_LEN.with(|l| *l.borrow() as u32)
}

/// 预览像素缓冲地址（RGBA8，用 [`prism_editor_pixels_len`] 取字节数）。
///
/// # 必须写 `b.borrow().as_ptr()` 而非 `b.as_ptr()`
///
/// `RefCell::as_ptr()`（Rust 1.86 稳定）经自动解引用**遮蔽**了 `Vec::as_ptr()`：
/// 前者返回指向 `Vec` **结构体本身**（ptr/len/cap 三个字）的指针，后者才返回
/// 像素数据的首地址。二者相差 16 字节且**紧邻** `ERR` 文本缓冲——实测会把
/// `Vec` 的三个字误当 RGBA 读出（alpha 恒 0），且任何一次错误文本写入都会
/// 顺带**踩坏**相邻缓冲。显式 `borrow()` 才能取到 `Vec` 的元素指针。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_pixels_ptr() -> usize {
    PIXELS.with(|b| b.borrow().as_ptr() as usize)
}

/// 预览像素缓冲字节数（= `width * height * 4`）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_pixels_len() -> u32 {
    PIXELS.with(|b| b.borrow().len() as u32)
}

// ---------------------------------------------------------------------------
// 裸分配（与 prism-gpu 同约定：16 字节对齐，两侧同一 len）
// ---------------------------------------------------------------------------

/// 在 wasm 线性内存中分配 `len` 字节，返回地址；失败（`len == 0` 或布局非法）返回 0。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_alloc(len: u32) -> usize {
    let len = len as usize;
    if len == 0 {
        return 0;
    }
    let layout = match std::alloc::Layout::from_size_align(len, ALLOC_ALIGN) {
        Ok(l) => l,
        Err(_) => return 0,
    };
    // SAFETY: layout 非零大小且对齐合法；返回的指针交由宿主用同一 len 释放。
    (unsafe { std::alloc::alloc(layout) }) as usize
}

/// 释放 [`prism_editor_alloc`] 分配的内存。`ptr == 0` 或 `len == 0` 为无操作。
///
/// # Safety
/// `ptr`/`len` 必须来自同一次 [`prism_editor_alloc`] 调用，且未重复释放。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prism_editor_dealloc(ptr: usize, len: u32) -> i32 {
    if ptr == 0 || len == 0 {
        return status::PASS;
    }
    let len = len as usize;
    let layout = match std::alloc::Layout::from_size_align(len, ALLOC_ALIGN) {
        Ok(l) => l,
        Err(_) => return bad_arg("prism_editor_dealloc：非法 len，无法重建布局"),
    };
    // SAFETY: 由本函数契约保证 ptr/len 来自且仅来自一次 prism_editor_alloc。
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    status::PASS
}

// ---------------------------------------------------------------------------
// 生命周期
// ---------------------------------------------------------------------------

/// 初始化（或重置）编辑器为默认参考图，并清空预览缓存与像素缓冲。
///
/// 返回 [`status::PASS`]；宿主应在每次打开页面 / 换图时调用。
///
/// # 幂等契约
///
/// 本函数重置**全部** ABI 线程局部状态：`STATE`（图 + 选中 + 撤销栈）、
/// `RENDERER`（中间缓冲缓存）、`PIXELS`（预览像素）与 `STATS`。其中 `PIXELS`
/// 必须一并清空：否则上一次渲染留下的像素会跨"重新初始化"残留，使
/// `pixels_len()` 与宿主手里的像素内容不属于同一次渲染。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_init() -> i32 {
    STATE.with(|s| *s.borrow_mut() = Some(EditorState::with_reference_graph()));
    RENDERER.with(|r| r.borrow_mut().clear_cache());
    PIXELS.with(|p| p.borrow_mut().clear());
    set_stats("{}");
    status::PASS
}

/// 编辑器是否已初始化（1 = 已初始化）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_is_init() -> u32 {
    STATE.with(|s| u32::from(s.borrow().is_some()))
}

/// 释放预览像素缓冲（宿主不再显示时可调用以回收 wasm 内存）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_free_pixels() -> i32 {
    PIXELS.with(|p| p.borrow_mut().clear());
    status::PASS
}

// ---------------------------------------------------------------------------
// 图与选择
// ---------------------------------------------------------------------------

/// 把当前图导出为 JSON 节点树（用 [`prism_editor_json_ptr`] / [`prism_editor_json_len`] 取）。
///
/// # Errors
/// 未初始化 → [`status::FAIL`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_graph_json() -> i32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => fail("error[E4000] prism_editor_graph_json\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
        Some(st) => set_json(&st.export_graph_json()),
    })
}

/// 选中一个节点（UI 高亮）。节点不存在 → [`status::BAD_ARG`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_select(node: u32) -> i32 {
    STATE.with(|s| {
        let mut slot = s.borrow_mut();
        let st = match slot.as_mut() {
            None => {
                return fail("error[E4000] prism_editor_select\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init")
            }
            Some(st) => st,
        };
        match st.select(u64::from(node)) {
            Ok(()) => status::PASS,
            Err(e) => bad_arg(e.to_string()),
        }
    })
}

/// 当前选中节点（`u32::MAX` = 未选中）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_selected() -> u32 {
    STATE.with(|s| match s.borrow().as_ref().and_then(|st| st.selected()) {
        Some(id) => id as u32,
        None => u32::MAX,
    })
}

// ---------------------------------------------------------------------------
// 编辑
// ---------------------------------------------------------------------------

/// 应用一次参数编辑：把 `node` 的参数 `name`（UTF-8 文本）设为 `value`。
///
/// `name_ptr`/`name_len` 指向宿主经 [`prism_editor_alloc`] 写入的 UTF-8 字节。
/// 成功返回 [`status::PASS`]，并把本次受影响子图写入 dirty 集
/// （[`prism_editor_dirty_json`] 可取）。参数不存在 / 类型不匹配 → [`status::BAD_ARG`]。
///
/// # 为什么指针参数是 `usize` 而不是 `u32`
///
/// wasm32 上指针宽 32 位，`usize` 即 `i32`，与 JS 侧 ABI 完全一致；而在 64 位
/// native 上 `u32` 会**截断**地址，导致解引用崩溃（`cargo test` SIGSEGV）。
/// 这正是 `prism-gpu/src/web/abi.rs` 模块头登记的同款约束（其
/// `prism_alloc -> usize` / `prism_dealloc(ptr: usize, ..)` 同理）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_apply_param_f32(node: u32, name_ptr: usize, name_len: u32, value: f32) -> i32 {
    apply_param(node, name_ptr, name_len, ParamValue::F32(value))
}

/// 应用一次参数编辑（`u32` 值）。语义同 [`prism_editor_apply_param_f32`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_apply_param_u32(node: u32, name_ptr: usize, name_len: u32, value: u32) -> i32 {
    apply_param(node, name_ptr, name_len, ParamValue::U32(value))
}

/// 应用一次参数编辑（`bool` 值；`value != 0` 为真）。语义同 [`prism_editor_apply_param_f32`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_apply_param_bool(node: u32, name_ptr: usize, name_len: u32, value: u32) -> i32 {
    apply_param(node, name_ptr, name_len, ParamValue::Bool(value != 0))
}

/// `apply_param` 的共同实现：校验 UTF-8 参数名 → 施加编辑。
fn apply_param(node: u32, name_ptr: usize, name_len: u32, value: ParamValue) -> i32 {
    // SAFETY: 宿主契约——ptr/len 指向 prism_editor_alloc 分配的有效区间。
    // 越界由 wasm 边界 trap 兜底；此处再校验 UTF-8 以免把坏字节塞进图。
    let name = if name_len == 0 || name_ptr == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => {
                return bad_arg(format!(
                    "error[E4008] prism_editor_apply_param(节点 {node})\n  什么事：参数名不是合法 UTF-8（{name_len} 字节）\n  建议：宿主须以 UTF-8 编码写入参数名后再调用"
                ))
            }
        }
    };
    if name.is_empty() {
        return bad_arg(format!(
            "error[E4005] prism_editor_apply_param(节点 {node})\n  什么事：参数名为空\n  建议：传入非空 UTF-8 参数名"
        ));
    }

    STATE.with(|s| {
        let mut slot = s.borrow_mut();
        let st = match slot.as_mut() {
            None => {
                return fail("error[E4000] prism_editor_apply_param\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init")
            }
            Some(st) => st,
        };
        match st.apply_edit(u64::from(node), name, value) {
            Ok(_dirty) => status::PASS,
            Err(e) => bad_arg(e.to_string()),
        }
    })
}

/// 撤销一次编辑。撤销栈空 → [`status::BAD_ARG`]（结构化错误，不 panic）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_undo() -> i32 {
    STATE.with(|s| {
        let mut slot = s.borrow_mut();
        let st = match slot.as_mut() {
            None => return fail("error[E4000] prism_editor_undo\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
            Some(st) => st,
        };
        match st.undo() {
            Ok(_) => status::PASS,
            Err(e) => bad_arg(e.to_string()),
        }
    })
}

/// 重做一次编辑。重做栈空 → [`status::BAD_ARG`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_redo() -> i32 {
    STATE.with(|s| {
        let mut slot = s.borrow_mut();
        let st = match slot.as_mut() {
            None => return fail("error[E4000] prism_editor_redo\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
            Some(st) => st,
        };
        match st.redo() {
            Ok(_) => status::PASS,
            Err(e) => bad_arg(e.to_string()),
        }
    })
}

/// 是否可撤销（1 = 可撤销）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_can_undo() -> u32 {
    STATE.with(|s| s.borrow().as_ref().map_or(0, |st| u32::from(st.can_undo())))
}

/// 是否可重做（1 = 可重做）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_can_redo() -> u32 {
    STATE.with(|s| s.borrow().as_ref().map_or(0, |st| u32::from(st.can_redo())))
}

// ---------------------------------------------------------------------------
// 草稿档位
// ---------------------------------------------------------------------------

/// 设置质量档位（`0` = Draft，`1` = Final；其它 → [`status::BAD_ARG`]）。
///
/// 档位经缓存键的 `spp_tier` + 内部分辨率生效，切换后旧缓存自然失效（C13）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_set_quality(tier: u32) -> i32 {
    let q = match tier {
        0 => PreviewQuality::Draft,
        1 => PreviewQuality::Final,
        other => {
            return bad_arg(format!(
                "error[E4009] prism_editor_set_quality({other})\n  什么事：档位号非法（仅 0=Draft / 1=Final）\n  建议：传入 0 或 1"
            ))
        }
    };
    STATE.with(|s| {
        if let Some(st) = s.borrow_mut().as_mut() {
            st.set_preview_quality(q);
            status::PASS
        } else {
            fail("error[E4000] prism_editor_set_quality\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init")
        }
    })
}

/// 当前档位号（0 = Draft，1 = Final；`u32::MAX` = 未初始化）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_quality() -> u32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => u32::MAX,
        Some(st) => st.preview_quality().spp_tier(),
    })
}

/// 当前档位的每像素样本数（`u32::MAX` = 未初始化）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_quality_spp() -> u32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => u32::MAX,
        Some(st) => st.preview_quality().spp(),
    })
}

// ---------------------------------------------------------------------------
// JSON 导出
// ---------------------------------------------------------------------------

/// 把选中节点的参数 schema 导出为 JSON（UI 生成滑条 / 数字控件用）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_schema_json(node: u32) -> i32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => fail("error[E4000] prism_editor_schema_json\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
        Some(st) => set_json(&st.export_schema_json(u64::from(node))),
    })
}

/// 把最近一次编辑的受影响子图（dirty 集）导出为 JSON 数组。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_dirty_json() -> i32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => fail("error[E4000] prism_editor_dirty_json\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
        Some(st) => {
            let ids = st.dirty_subgraph().to_vec();
            let mut out = String::from("[");
            for (i, id) in ids.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&id.to_string());
            }
            out.push(']');
            set_json(&out)
        }
    })
}

/// 把**编辑操作序列**导出为 JSON（可复制保存 / 回放）。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_edits_json() -> i32 {
    STATE.with(|s| match s.borrow().as_ref() {
        None => fail("error[E4000] prism_editor_edits_json\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init"),
        Some(st) => set_json(&st.export_edits_json()),
    })
}

// ---------------------------------------------------------------------------
// 预览渲染
// ---------------------------------------------------------------------------

/// 渲染一帧预览到内部像素缓冲（用 [`prism_editor_pixels_ptr`] 取地址）。
///
/// `width`/`height` 为输出尺寸（1..=512）；渲染完成后统计写入 stats JSON
/// （[`prism_editor_stats_ptr`]），含 `recomputed` / `reused`——**增量收益的
/// 直接可视化**。未初始化 → [`status::FAIL`]；尺寸非法 → [`status::BAD_ARG`]。
#[unsafe(no_mangle)]
pub extern "C" fn prism_editor_render(width: u32, height: u32) -> i32 {
    STATE.with(|s| {
        let slot = s.borrow();
        let st = match slot.as_ref() {
            None => {
                return fail("error[E4000] prism_editor_render\n  什么事：编辑器未初始化\n  建议：先调用 prism_editor_init")
            }
            Some(st) => st,
        };
        let graph = match st.model().to_graph() {
            Some(g) => g,
            None => {
                return fail("error[E4007] prism_editor_render\n  什么事：编辑器模型无法构造 core 图\n  建议：检查图是否存在悬空连线")
            }
        };
        let bytes = (width as usize) * (height as usize) * 4;
        if bytes > MAX_PREVIEW_BYTES {
            return bad_arg(format!(
                "error[E4102] prism_editor_render({width}x{height})\n  什么事：预览缓冲 {bytes} 字节超过上限 {MAX_PREVIEW_BYTES}\n  建议：把宽高降到 512 以内"
            ));
        }
        let quality = st.preview_quality();
        let out = RENDERER.with(|r| {
            r.borrow_mut().render(&graph, &Default::default(), quality, width, height, ENGINE_VERSION, FRAME)
        });
        match out {
            Ok(rendered) => {
                set_stats(&stats_json(&rendered));
                PIXELS.with(|p| *p.borrow_mut() = rendered.rgba);
                status::PASS
            }
            Err(e) => bad_arg(e.to_string()),
        }
    })
}

/// 缓存键用的引擎版本（编辑器预览固定值；改动须递增以失效旧缓存，C13）。
const ENGINE_VERSION: u64 = (0 << 32) | (13 << 16) | 0;

/// 缓存键用的帧号（编辑器为单帧静态预览，固定 0）。
const FRAME: u64 = 0;

/// 把渲染统计编为 JSON（recomputed / reused / spp / 内部分辨率 / 锥大小）。
fn stats_json(r: &super::preview::Rendered) -> String {
    let s = &r.stats;
    let ids = |v: &[u64]| {
        let mut out = String::from("[");
        for (i, id) in v.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&id.to_string());
        }
        out.push(']');
        out
    };
    format!(
        "{{\"width\":{},\"height\":{},\"spp\":{},\"internal_width\":{},\"internal_height\":{},\"cone_size\":{},\"recomputed\":{},\"reused\":{}}}",
        r.width,
        r.height,
        s.spp,
        s.internal_width,
        s.internal_height,
        s.cone_size,
        ids(&s.recomputed),
        ids(&s.reused)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 把 JSON 文本缓冲读回 Rust String。
    fn json_text() -> String {
        let ptr = prism_editor_json_ptr();
        let len = prism_editor_json_len() as usize;
        if len == 0 {
            return String::new();
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// 把错误文本缓冲读回 Rust String。
    fn err_text() -> String {
        let ptr = prism_editor_error_ptr();
        let len = prism_editor_error_len() as usize;
        if len == 0 {
            return String::new();
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// 把统计 JSON 读回 Rust String。
    fn stats_text() -> String {
        let ptr = prism_editor_stats_ptr();
        let len = prism_editor_stats_len() as usize;
        if len == 0 {
            return String::new();
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// 在 wasm 线性内存里造一个 C 字符串，返回 (ptr, len)。调用方须自行释放。
    fn cstr(s: &str) -> (usize, u32) {
        let bytes = s.as_bytes();
        let ptr = prism_editor_alloc(bytes.len() as u32);
        assert_ne!(ptr, 0, "分配不应失败");
        // SAFETY: ptr 指向刚分配的 len 字节，可写。
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const u8, ptr as *mut u8, bytes.len()) };
        (ptr, bytes.len() as u32)
    }

    /// ABI 测试的全局串行锁。
    ///
    /// 根因（曾观测到 `alloc_dealloc_round_trip` 偶发 SIGSEGV，单跑必过）：
    /// `STATE` / `ERR` 是 [`thread_local`]，而 `cargo test` 默认多线程且**复用
    /// 工作线程**。两个 ABI 测试若落在同一线程，第二个的 [`prism_editor_init`]
    /// 会重置第一个已建好的 `STATE`（含其 `Vec` / `String` 缓冲），而第一个的
    /// 断言与 `cstr` 缓冲读取仍在进行 ⇒ 悬垂引用 ⇒ SIGSEGV。
    ///
    /// wasm 侧单线程无此问题（见模块头"零锁"约定）；此锁**仅存在于测试代码**，
    /// 不进wasm 产物。修法选择串行化而非"每测试独立线程"：后者会改变
    /// `thread_local` 的语义前提（`uninitialized_calls_fail_gracefully`
    /// 正是**依赖**独立线程才成立）。
    static ABI_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 取测试锁守卫（RAII），确保 ABI 测试逐个串行执行。
    fn abi_guard() -> std::sync::MutexGuard<'static, ()> {
        // 上一测试若 panic 释放了毒，此处恢复后继续（不二次 panic）。
        ABI_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn abi_version_is_stable() {
        assert_eq!(prism_editor_abi_version(), EDITOR_ABI_VERSION);
    }

    #[test]
    fn init_then_graph_json_lists_four_nodes() {
        let _g = abi_guard();
        assert_eq!(prism_editor_init(), status::PASS);
        assert_eq!(prism_editor_is_init(), 1);
        assert_eq!(prism_editor_graph_json(), status::PASS);
        let js = json_text();
        assert!(js.contains("\"nodes\":["), "应含 nodes 数组：{js}");
        // 参考图 4 节点：`"id":` 恰好 4 次（`selected` 为 null，不含 `"id":`）。
        assert_eq!(js.matches("\"id\":").count(), 4, "参考图应恰好 4 个节点：{js}");
        assert!(js.contains("\"id\":1"), "应含节点 1：{js}");
        assert!(js.contains("\"id\":4"), "应含节点 4：{js}");
        assert!(js.contains("\"selected\":null"), "初始未选中：{js}");
    }

    #[test]
    fn select_and_query_selected_node() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_selected(), u32::MAX, "初始未选中");
        assert_eq!(prism_editor_select(2), status::PASS);
        assert_eq!(prism_editor_selected(), 2);
        // 不存在的节点 → BAD_ARG + 结构化错误。
        assert_eq!(prism_editor_select(999), status::BAD_ARG);
        assert!(err_text().contains("E4001"), "应给结构化错误码：{}", err_text());
    }

    #[test]
    fn apply_param_marks_exact_dirty_subgraph() {
        let _g = abi_guard();
        prism_editor_init();
        let (p, n) = cstr("scale");
        let rc = prism_editor_apply_param_f32(1, p, n, 0.9);
        // SAFETY: 由 cstr 分配、同一 len 释放。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(rc, status::PASS, "错误：{}", err_text());

        assert_eq!(prism_editor_dirty_json(), status::PASS);
        let dirty = json_text();
        assert_eq!(dirty, "[1,3,4]", "改叶 1 → 受影响子图恰为 {{1,3,4}}");
        assert_eq!(prism_editor_can_undo(), 1);
        assert_eq!(prism_editor_can_redo(), 0);
    }

    #[test]
    fn undo_redo_round_trip_via_abi() {
        let _g = abi_guard();
        prism_editor_init();
        let (p, n) = cstr("scale");
        prism_editor_apply_param_f32(1, p, n, 0.9);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(prism_editor_undo(), status::PASS);
        assert_eq!(prism_editor_can_redo(), 1);
        assert_eq!(prism_editor_redo(), status::PASS);
        assert_eq!(prism_editor_can_undo(), 1);
        assert_eq!(prism_editor_can_redo(), 0);
    }

    #[test]
    fn undo_on_empty_stack_is_bad_arg_not_panic() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_undo(), status::BAD_ARG);
        assert!(err_text().contains("E4002"), "应含 EmptyUndoStack 错误码：{}", err_text());
        assert_eq!(prism_editor_redo(), status::BAD_ARG);
        assert!(err_text().contains("E4003"), "应含 EmptyRedoStack 错误码：{}", err_text());
    }

    #[test]
    fn quality_tier_switches_and_rejects_bad_value() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_quality(), 0, "默认 Draft");
        let draft_spp = prism_editor_quality_spp();
        assert_eq!(prism_editor_set_quality(1), status::PASS);
        assert_eq!(prism_editor_quality(), 1, "切到 Final");
        let final_spp = prism_editor_quality_spp();
        assert!(draft_spp < final_spp, "Draft spp({draft_spp}) 必须低于 Final({final_spp})");
        assert_eq!(prism_editor_set_quality(7), status::BAD_ARG);
        assert!(err_text().contains("E4009"), "应含档位非法错误码：{}", err_text());
    }

    #[test]
    fn render_fills_pixels_and_stats_show_incremental_gain() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_render(64, 64), status::PASS, "错误：{}", err_text());
        assert_eq!(prism_editor_pixels_len(), 64 * 64 * 4);
        let st = stats_text();
        assert!(st.contains("\"recomputed\":[1,2,3,4]"), "首帧全部重算：{st}");
        assert!(st.contains("\"spp\":1"), "Draft 默认 1 spp：{st}");

        // 第二帧同参数 → 全部命中缓存。
        assert_eq!(prism_editor_render(64, 64), status::PASS);
        let st2 = stats_text();
        assert!(st2.contains("\"recomputed\":[]"), "同帧再渲染零重算：{st2}");
        assert!(st2.contains("\"reused\":[1,2,3,4]"), "全部复用：{st2}");

        // 改叶 1 → 只重算 {1,3,4}。
        let (p, n) = cstr("scale");
        prism_editor_apply_param_f32(1, p, n, 0.9);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(prism_editor_render(64, 64), status::PASS);
        let st3 = stats_text();
        assert!(st3.contains("\"recomputed\":[1,3,4]"), "增量：只重算祖先链：{st3}");
        assert!(st3.contains("\"reused\":[2]"), "兄弟叶复用：{st3}");
    }

    #[test]
    fn render_rejects_oversize_and_zero() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_render(0, 32), status::BAD_ARG);
        assert!(err_text().contains("E4102"), "应含尺寸非法错误码：{}", err_text());
        assert_eq!(prism_editor_render(1024, 32), status::BAD_ARG);
    }

    #[test]
    fn edits_json_export_is_stable_and_replayable() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_edits_json(), status::PASS);
        let empty = json_text();
        assert!(empty.contains("\"edits\":[]"), "初始无编辑：{empty}");
        assert!(empty.contains("\"schema\":1"), "应带 schema 版本头：{empty}");

        let (p, n) = cstr("scale");
        prism_editor_apply_param_f32(1, p, n, 0.9);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(prism_editor_edits_json(), status::PASS);
        let js = json_text();
        assert!(js.contains("\"kind\":\"set_param\""), "应含 set_param 命令：{js}");
        assert!(js.contains("\"node\":1"), "应含目标节点：{js}");
        // 确定性：重复导出逐字节一致。
        assert_eq!(prism_editor_edits_json(), status::PASS);
        assert_eq!(json_text(), js, "导出必须逐字节稳定（确定性铁律）");
    }

    #[test]
    fn schema_json_exposes_slider_range() {
        let _g = abi_guard();
        prism_editor_init();
        assert_eq!(prism_editor_schema_json(1), status::PASS);
        let js = json_text();
        assert!(js.contains("\"name\":\"scale\""), "应含参数名：{js}");
        assert!(js.contains("\"min\":"), "应含滑条下界：{js}");
        assert!(js.contains("\"max\":"), "应含滑条上界：{js}");
        assert!(js.contains("\"step\":"), "应含步进：{js}");
    }

    #[test]
    fn apply_param_rejects_unknown_param_and_bad_utf8() {
        let _g = abi_guard();
        prism_editor_init();
        let (p, n) = cstr("nonexistent_param");
        let rc = prism_editor_apply_param_f32(1, p, n, 1.0);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(rc, status::BAD_ARG);
        assert!(err_text().contains("E4005"), "应含 UnknownParam 错误码：{}", err_text());

        // 类型不匹配：u32 参数写入 f32。
        let (p2, n2) = cstr("octaves");
        let rc2 = prism_editor_apply_param_f32(1, p2, n2, 1.5);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p2, n2) };
        assert_eq!(rc2, status::BAD_ARG, "u32 参数不接受 f32");
        assert!(err_text().contains("E4004"), "应含类型不匹配错误码：{}", err_text());
    }

    #[test]
    fn apply_param_u32_and_bool_paths() {
        let _g = abi_guard();
        prism_editor_init();
        let (p, n) = cstr("octaves");
        assert_eq!(prism_editor_apply_param_u32(1, p, n, 6), status::PASS, "{}", err_text());
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p, n) };
        assert_eq!(prism_editor_can_undo(), 1);

        // bool 路径：用字符串参数验证类型 tag 分派（本参考图无 bool 参数，
        // 故预期 BAD_ARG 而非 panic——这正是要验的：不 panic）。
        let (p2, n2) = cstr("octaves");
        let rc = prism_editor_apply_param_bool(1, p2, n2, 1);
        // SAFETY: 同一 cstr 分配/释放配对。
        unsafe { prism_editor_dealloc(p2, n2) };
        assert_eq!(rc, status::BAD_ARG, "u32 参数不接受 bool");
    }

    #[test]
    fn uninitialized_calls_fail_gracefully() {
        // 注意：thread_local 每线程独立，本测试用独立线程确保"未初始化"。
        std::thread::spawn(|| {
            // 该线程内 STATE 初始为 None（未被 init 触碰）。
            assert_eq!(prism_editor_is_init(), 0);
            assert_eq!(prism_editor_graph_json(), status::FAIL);
            assert!(err_text().contains("E4000"), "应含未初始化错误码：{}", err_text());
            assert_eq!(prism_editor_undo(), status::FAIL);
            assert_eq!(prism_editor_selected(), u32::MAX);
            assert_eq!(prism_editor_quality(), u32::MAX);
        })
        .join()
        .expect("子线程应正常结束（无 panic）");
    }

    #[test]
    fn pixels_ptr_points_at_pixels_not_at_vec_struct() {
        let _g = abi_guard();
        // 回归测试：`RefCell::as_ptr()`（Rust 1.86 稳定）会自动解引用**遮蔽**
        // `Vec::as_ptr()`，返回指向 Vec **结构体**（ptr/len/cap）的指针，而非像素
        // 首地址。二者仅隔 16 字节且紧邻 ERR 缓冲——真实 wasm 上表现为：
        //   1) 读出的是 Vec 的三个字，alpha 恒 0（真像素缓冲的 alpha 恒 255）；
        //   2) 任何一次错误文本写入都会踩坏相邻缓冲；
        //   3) 指针在 Vec 重新分配后失效 → 两次渲染像素"随机"不同（假性非确定）。
        // 本测试从**可观测后果**入手，不依赖实现细节。
        prism_editor_init();
        assert_eq!(prism_editor_render(32, 32), status::PASS, "错误：{}", err_text());
        let len = prism_editor_pixels_len() as usize;
        assert_eq!(len, 32 * 32 * 4);

        let ptr = prism_editor_pixels_ptr();
        assert_ne!(ptr, 0, "像素指针不应为空");
        // SAFETY: 契约保证 ptr 指向 len 字节的 RGBA 缓冲。
        let buf = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        // 关键断言：alpha 恒 255（`prism-rt` 出帧时对每个像素写 255）。
        // 若指针误指向 Vec 结构体，这里读到的会是 ptr/len/cap 三个字，alpha 必为 0。
        // 注意取 alpha 的偏移是 3（RGBA 排列），不能写成 `step_by(4)`——
        // 那从偏移 0 起跳，取到的是**红通道**（本项目首像素 R=173，正是曾经的误报值）。
        for (i, px) in buf.chunks_exact(4).enumerate() {
            assert_eq!(px[3], 255, "第 {i} 个像素的 alpha 应为 255，实际 {}（像素指针是否指错？）", px[3]);
        }
        // 像素缓冲不得与文本缓冲重叠（否则错误写入会踩坏像素）。
        for (name, tptr) in [
            ("ERR", prism_editor_error_ptr()),
            ("JSON", prism_editor_json_ptr()),
            ("STATS", prism_editor_stats_ptr()),
        ] {
            let overlap = ptr < tptr + TEXT_CAP && tptr < ptr + len;
            assert!(!overlap, "像素缓冲与 {name} 缓冲重叠（像素 {ptr}..{}, {name} {tptr}..{}）", ptr + len, tptr + TEXT_CAP);
        }

        // 错误写入不得改动像素（重叠时的第二个可观测后果）。
        let before: Vec<u8> = buf.to_vec();
        assert_eq!(prism_editor_select(999), status::BAD_ARG, "应写错误文本");
        assert!(err_text().contains("E4001"), "错误文本应已写入：{}", err_text());
        // SAFETY: 同上，读同一缓冲。
        let after = unsafe { std::slice::from_raw_parts(prism_editor_pixels_ptr() as *const u8, len) };
        assert_eq!(before, after, "写错误文本不得改动像素缓冲（说明二者未重叠）");
    }

    #[test]
    fn render_twice_is_byte_identical_through_abi() {
        let _g = abi_guard();
        // 经 ABI 连续两次渲染 → 像素逐字节一致（确定性铁律）。
        // 这条同时覆盖上面那条回归：指针错位时两次结果会不同。
        prism_editor_init();
        assert_eq!(prism_editor_render(48, 48), status::PASS, "错误：{}", err_text());
        let len = prism_editor_pixels_len() as usize;
        // SAFETY: 契约保证 ptr/len 为有效 RGBA 缓冲。
        let first = unsafe { std::slice::from_raw_parts(prism_editor_pixels_ptr() as *const u8, len) }.to_vec();
        assert_eq!(prism_editor_render(48, 48), status::PASS);
        let second = unsafe { std::slice::from_raw_parts(prism_editor_pixels_ptr() as *const u8, len) };
        assert_eq!(first, second, "同参连续渲染两次，像素必须逐字节一致");
    }

    #[test]
    fn alloc_dealloc_round_trip() {
        let _g = abi_guard();
        let ptr = prism_editor_alloc(64);
        assert_ne!(ptr, 0);
        // SAFETY: 刚分配的 64 字节，写入合法。
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0xAB, 64) };
        // SAFETY: 同一 alloc 分配、同一 len 释放。
        let rc = unsafe { prism_editor_dealloc(ptr, 64) };
        assert_eq!(rc, status::PASS);
        // 零长度 / 空指针为无操作。
        // SAFETY: len=0 或 ptr=0 时不解引用，契约要求不触碰内存。
        unsafe {
            assert_eq!(prism_editor_dealloc(0, 64), status::PASS);
            assert_eq!(prism_editor_dealloc(ptr, 0), status::PASS);
        }
        assert_eq!(prism_editor_alloc(0), 0);
    }
}
