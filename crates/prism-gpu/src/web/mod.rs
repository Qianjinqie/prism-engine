//! WebGPU **手写绑定**（零第三方 crate）。
//!
//! 本模块不依赖 `wasm-bindgen` / `web-sys`；它用最原始的 `extern "C"`
//! 声明直接触达浏览器注入的 JS 全局，与项目"手写 PNG/手写平台 API"同哲学
//! （D-15：调用平台原生 API 不算第三方依赖）。
//!
//! # 互操作约束（诚实说明）
//!
//! `wasm32-unknown-unknown` 目标下 Rust 无法直接调用 JS。要调用 `navigator.gpu`
//! 这类 JS 对象，规范上有两条零 crate 路径：
//!
//! 1. **导入函数胶水**：宿主（harness 页面的 JS）用 `WebAssembly.instantiate`
//!    时传入 `env` 导入对象，把"发起 requestAdapter / 提交命令 / 读回 buffer"
//!    包装成少量 JS 函数；Rust 侧用 `extern "C" { fn ... }` 声明并调用。
//! 2. **手写 JS 值句柄表**：Rust 只持有整数句柄，JS 侧维护 `Map<id, GPUObject>`。
//!
//! 本模块采用 **路径 1 的最小自研胶水**：JS 胶水仅做"无法用纯 WASM 表达的
//! 异步/对象操作"，不含任何渲染逻辑；WGSL shader、绑定布局决策、dispatch
//! 规模等**全部逻辑留在 Rust 侧**。这是把"必须的 JS 互操作"降到理论最小
//! （约十余个转发函数）的工程取舍，已在此显式登记。
//!
//! 在非 wasm 目标（native）上，本模块仍会编译：extern 块被 `#[cfg]` 屏蔽，
//! 提供纯 Rust 的**能力探测桩**，保证 `cargo build`/`cargo test` 在桌面通过。

// ---------------------------------------------------------------------------
// 子模块
// ---------------------------------------------------------------------------

pub mod abi;
pub mod device;

// ---------------------------------------------------------------------------
// 平台分支：仅在 wasm32 目标下生成真实的 JS 导入声明
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub(crate) mod js {
    //! 浏览器 WebGPU 互操作胶水的 `extern "C"` 声明。
    //!
    //! 所有函数由 harness 页面（`web/index.html`）在实例化 WASM 时通过
    //! `env` 导入对象提供。约定所有 GPU 对象以 `u32` 句柄传递，宿主侧维持
    //! `Map<handle, GPUObject>`。返回 `0` 表示失败/空。

    // 全部 13 个函数都由 [`abi`] 模块的导出测试项实际调用（D-16：不留死胶水）：
    // 能力探测 3 个由 `detect_support`/`WebGpuBackend` 使用，资源创建与执行 9 个
    // 由 `abi` 的 `submit_chain` 路径使用（`prism_create_bind_group` 支持 group
    // 索引 / `prism_dispatch` 支持多 bind group），错误上报由测试项收尾阶段轮询。
    extern "C" {
        // --- 能力探测 ---------------------------------------------------
        /// 返回 1 表示 `navigator.gpu` 存在（即 WebGPU 可用）。
        pub fn prism_navigator_gpu_available() -> u32;

        /// 请求 adapter + device 并登记，成功返回 device 句柄，失败返回 0。
        pub fn prism_request_device() -> u32;

        /// 返回 device 的 `lost` 是否已触发（1 = lost）。
        pub fn prism_device_lost(device: u32) -> u32;

        /// 返回该句柄是否已绑定到一个**真实的 GPUDevice**（1 = 就绪）。
        ///
        /// # 为什么必须单独问"就绪"而不能看句柄非 0
        /// `prism_request_device` 是**同步**返回预留句柄的（WebAssembly 的 import
        /// 只能是同步函数，返回 Promise 会让模块链接直接失败），真正的
        /// `requestAdapter`/`requestDevice` 在宿主 `settle()` 时才 await。因此
        /// 「句柄 ≠ 0」只代表"已预约"，不代表"device 已到手"——此时宿主句柄表里
        /// 该键的值仍是 `null`，后续 `createBuffer` 等一律失败。
        ///
        /// D-14 真机实测踩过：宿主 `runWasmTest` 对非 PENDING 项会清空
        /// `pendingWork`，`requestDevice` 的 promise 因此从未被 await，而
        /// `device.init` 只查句柄非 0 就算 PASS —— 于是一个**根本不存在的 device**
        /// 被判为通过，后续 3 个 GPU 项全部 `prism_create_buffer 失败`。
        /// 本函数让 `device.init` 的 PASS 真正等价于"device 已到手"。
        pub fn prism_device_ready(device: u32) -> u32;

        // --- 资源创建 ---------------------------------------------------
        /// 创建 buffer，返回句柄。`usage` 位：1 = STORAGE（附 COPY_SRC/COPY_DST），
        /// 2 = UNIFORM（附 COPY_DST）——与 `abi::usage` 约定一致。
        pub fn prism_create_buffer(device: u32, size: u32, usage: u32) -> u32;

        /// 写入 buffer：`ptr`/`len` 指向 WASM 线性内存中的字节。
        pub fn prism_write_buffer(device: u32, buffer: u32, ptr: u32, len: u32) -> u32;

        /// 读回 buffer 到一个 WASM 侧可读的暂存区（宿主登记），返回暂存句柄。
        pub fn prism_read_buffer(device: u32, buffer: u32, size: u32) -> u32;

        /// 把宿主暂存区的数据搬到 WASM 线性内存 `dst_ptr` 处（`len` 字节）。
        pub fn prism_copy_out(staging: u32, dst_ptr: u32, len: u32) -> u32;

        /// 用 WGSL 源创建 shader module，返回句柄。
        pub fn prism_create_shader_module(device: u32, src_ptr: u32, src_len: u32) -> u32;

        /// 用 compute entry point 创建 compute pipeline，返回句柄。
        pub fn prism_create_compute_pipeline(
            device: u32,
            module: u32,
            entry_ptr: u32,
            entry_len: u32,
        ) -> u32;

        /// 创建 bind group（绑定 N 个 storage/uniform buffer），`bufs_ptr` 为句柄数组。
        ///
        /// `group` 指定 WGSL 的 `@group(group)` 索引。宿主用
        /// `pipe.getBindGroupLayout(group)` 取该组的布局；句柄数组第 `i` 个硬映射到
        /// `binding: i`（见 `abi.rs` 的 `bind_group_of` 与 `web/index.html` 胶水）。
        /// 多 bind group 能力（group 0/1/2/3）由本参数解锁——generate/shade/extend
        /// 静态引用 `@group(1)`/`@group(2)`/`@group(3)` 资源，旧签名硬编码 group 0
        /// 无法为它们建 bind group（D2 的"待决策"缺口）。
        pub fn prism_create_bind_group(
            device: u32,
            pipeline: u32,
            group: u32,
            bufs_ptr: u32,
            n: u32,
        ) -> u32;

        // --- 执行 -------------------------------------------------------
        /// 运行一次 compute：绑定 `n` 个 bind group 并 dispatch `gx*gy*gz` 工作组。
        ///
        /// `bgs_ptr` 指向 `n` 个 bind group 句柄；第 `i` 个绑到 `@group(i)`
        /// （group 索引即为数组下标——与 `layout:"auto"` 下"每个 group 的绑定集是
        /// 连续前缀"同理，被 pipeline 静态使用的 group 集合同样从 0 起连续）。
        pub fn prism_dispatch(
            device: u32,
            pipeline: u32,
            bgs_ptr: u32,
            n: u32,
            gx: u32,
            gy: u32,
            gz: u32,
        ) -> u32;

        /// 等待队列完成（`onSubmittedWorkDone`），返回 1 表示成功。
        pub fn prism_wait_idle(device: u32) -> u32;

        // --- 错误上报 ---------------------------------------------------
        /// 弹出并返回最近一条已捕获的 GPU 错误码（0 = 无错误）。
        pub fn prism_pop_error(device: u32) -> u32;
    }
}

// ---------------------------------------------------------------------------
// 公开的手写绑定外观（native 侧提供可编译的桩实现）
// ---------------------------------------------------------------------------
/// WebGPU 句柄类型：宿主 JS 侧 `Map<u32, GPUObject>` 的键。
pub type GpuHandle = u32;

/// 创建失败/空句柄的约定值。
pub const INVALID_HANDLE: GpuHandle = 0;

/// WebGPU 可用性探测结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuSupport {
    /// `navigator.gpu` 存在，WebGPU 可用。
    Available,
    /// 浏览器无 `navigator.gpu`（特性检测失败）。
    NoNavigatorGpu,
    /// 非 wasm 目标（native 编译期，永远不可用）。
    NotWasmTarget,
}

/// 探测当前运行环境的 WebGPU 支持情况。
///
/// 非 wasm 目标直接返回 [`WebGpuSupport::NotWasmTarget`]，保证 native 编译
/// 与测试通过；wasm 目标查询宿主注入的 `navigator.gpu` 探测函数。
pub fn detect_support() -> WebGpuSupport {
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: 该函数由 harness 页面在实例化时提供，无副作用、无参数。
        let available = unsafe { js::prism_navigator_gpu_available() };
        if available != 0 {
            WebGpuSupport::Available
        } else {
            WebGpuSupport::NoNavigatorGpu
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        WebGpuSupport::NotWasmTarget
    }
}

/// WebGPU 基础特性/限额摘要，用于 harness 环境报告。
#[derive(Debug, Clone, Default)]
pub struct AdapterSummary {
    /// 适配器名称（如 `apple m1` / `adreno`），失败时为空串。
    pub adapter_name: String,
    /// 是否支持 `shader-f16`（可选特性）。
    pub has_shader_f16: bool,
    /// 是否支持 `timestamp-query`（可选特性）。
    pub has_timestamp_query: bool,
}

/// 读取当前 device 的适配器摘要。
///
/// native 目标返回默认值（全 false、空名），仅用于编译期接口一致性。
pub fn adapter_summary(_device: GpuHandle) -> AdapterSummary {
    // 摘要信息由 harness JS 侧直接从 GPUAdapterInfo 组装后回传文本；
    // Rust 侧目前只需提供类型占位，保证接口稳定（D1 骨架）。
    AdapterSummary::default()
}

// ---------------------------------------------------------------------------
// 单元测试：仅验证平台无关的能力探测语义（零 GPU 依赖）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_support_on_native_is_not_wasm_target() {
        // native 编译下探测必须稳定返回 NotWasmTarget，不得 panic。
        assert_eq!(detect_support(), WebGpuSupport::NotWasmTarget);
    }

    #[test]
    fn adapter_summary_default_is_empty() {
        let s = adapter_summary(INVALID_HANDLE);
        assert!(s.adapter_name.is_empty());
        assert!(!s.has_shader_f16);
        assert!(!s.has_timestamp_query);
    }
}
