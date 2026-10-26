//! Metal 后端（D1，macOS/iOS 独有技术）。
//!
//! # ⚠️ 诚实边界（必读，G4/Q5）
//!
//! Metal 是 Apple 平台的私有图形 API，**只在 macOS/iOS 上存在**；其他平台
//! （Linux / wasm）上本模块编译为零或仅保留结构面。本模块的
//! 收尾标准因此是：*在非 macOS 上编译为零，在 macOS 上编译为结构正确、
//! 逻辑完备的探测链*。**渲染路径已按 A1 的 kernel 结构同构移植代码，但未点亮**：
//! [`MetalBackend::is_available`] 在**所有平台**都刻意返回 `false`。
//!
//! ## 「探测链就绪」与「真实 dispatch」是两件事（不得混用）
//!
//! | 层 | 状态 | 位置 |
//! |---|---|---|
//! | 设备探测链（`dlopen` Metal 框架 + `MTLCreateSystemDefaultDevice`） | ✅ 代码就绪（**仅 macOS 编译**，Linux 上被 `#[cfg]` 剔除） | [`loader`] |
//! | MSL 双份源（场景整帧 kernel） | ✅ 代码就绪（文本，全平台可见） | [`msl::RT_SCENE_MSL`] |
//! | kernel 描述 + 参数 ABI 编码 | ✅ 代码就绪，偏移有单测钉死 | [`render`] |
//! | 算法的 CPU 同构实现（与 `CpuRefBackend` 逐字节对拍的证据） | ✅ 已证明 | [`render::render_scene_msl_semantics`] |
//! | **`MTLDevice` → queue → pipeline state → dispatch → 读回** | ❌ **未做**（需 `objc_msgSend` 层） | — |
//!
//! 结论：**macOS 实机验证未做**。V2.1 对 Metal 的交付口径是「代码 + 编译门控，
//! 不点亮」（PRD A2 / Q5）。因此：
//!
//! * [`MetalBackend::is_available`] **恒 `false`**（所有平台）；
//! * [`MetalBackend::render_scene`] **恒返回无效空缓冲**（绝不产假帧）；
//! * 全仓无任何一处声称 Metal 可用——「有源码」≠「能编译」≠「能出图」。
//!
//! # 零第三方依赖（D-15）
//!
//! 不引入 `metal` / `objc` / `core-foundation` 等任何 crate。macOS 上的设备
//! 探测走 **`dlopen` + `dlsym` + `extern "C"`**（见 [`loader`] 子模块），
//! `dlopen`/`dlsym` 来自系统 `libdl`，这是调用平台原生 API，不是第三方依赖。
//!
//! # 平台门禁
//!
//! * **macOS**：`loader` 子模块编译，探测链可用（`MTLCreateSystemDefaultDevice`
//!   → `MTLCopyAllDevices`）。
//! * **Linux / wasm**：`loader` 被 `#[cfg]` 整体剔除；[`MetalBackend`] 仍可见
//!   （供 `lib.rs` 引用与测试），但恒报不可用、`render_scene` 返回
//!   [`Framebuffer::empty`]，不 panic（宪章 C18）。
//!
//! # MSL「双份源」组织约定
//!
//! 定案：*Metal 侧 MSL 手写双份源*（Vulkan 侧走自研 SPIR-V
//! 发射器；Metal 侧不走发射器，直接手写 MSL）。约定见 [`msl`] 子模块：
//!
//! * **目录**：`crates/prism-gpu/src/metal/msl/`，与 WGSL 源
//!   `crates/prism-gpu/shaders/` 平行。
//! * **命名**：`<wgsl 文件名去扩展>.<entry>.msl`。例：WGSL `sim_cloth.wgsl`
//!   的 `cloth_predict` 入口 → `msl/sim_cloth.cloth_predict.msl`。
//! * **绑定映射**：WGSL `@group(g) @binding(b)` → MSL
//!   `[[buffer(<index>)]]`，`index = g * 31 + b`（与 Metal 常规单级
//!   `[[buffer(i)]]` 命名空间对齐；具体编号由后端实装时统一，见各 `.msl`
//!   头注）。`@builtin(global_invocation_id)` → `[[thread_position_in_grid]]`。
//! * **样本**：`cloth_predict`（首样例）、`primary_ray_gen`、`scene_main`
//!   （V2.1 T-A2，`include_str!` 从真实 `.msl` 文件读入）。其余 kernel 待后续
//!   逐一手写，**不建空文件充数**（D-16）。

use crate::{Framebuffer, GpuDevice, SceneRenderRequest};

// ---------------------------------------------------------------------------
// 平台门禁：仅 macOS 编译真实探测链
// ---------------------------------------------------------------------------
//
// 注意：即使 macOS 上探测链真的拿到了 `MTLDevice`，本模块也**没有** command
// queue / pipeline state / 读回路径——`is_available()` 仍恒 `false`。

#[cfg(target_os = "macos")]
pub mod loader;

// ---------------------------------------------------------------------------
// MSL「双份源」注册表（所有平台可见：源是纯文本，无平台依赖）
// ---------------------------------------------------------------------------

pub mod msl;

// ---------------------------------------------------------------------------
// render 路径的**设备无关**部分（所有平台编译：编码 + 语义同构 + 描述）
// ---------------------------------------------------------------------------
//
// 刻意**不用** `#[cfg(target_os = "macos")]` 包住：这部分是纯数据变换
// （参数/球/灯编码 + MSL 语义的 CPU 同构实现），在 Linux 上也能编译与对拍，
// 从而保证「移植没写错」这件事在沙箱内可验证。真正需要 macOS 的只有
// `objc_msgSend` 那一层（**未实现**，见 `metal/loader.rs` 末尾清单）。

pub mod render;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Metal 后端把手。
///
/// 这是一个**能力占位**而非可用渲染器：见模块头注的诚实边界。它的存在意义
/// 有三：① 使 `lib.rs` / 上层能在所有平台统一引用 `MetalBackend`；② 在
/// macOS 上提供真实的设备探测入口（[`MetalBackend::probe`]）；③ 承载 V2.1
/// 的同构移植（MSL 双份源 + 参数 ABI 编码 + 语义对拍，见 [`render`]）。
///
/// # 为什么 `is_available()` 恒为 `false`
///
/// [`GpuDevice::is_available`] 的语义是「**能否 service `render_scene`**」。
/// 本后端即便在 macOS 上探测到 `MTLDevice`，也**没有** pipeline state /
/// command buffer 编码 / 读回路径——出一个真帧的最小闭环尚不存在
/// （`objc_msgSend` 层未写，见 `loader.rs` 末尾清单）。因此此处**不撒谎**：
/// 探测成功不等于可用，恒返回 `false`，让上层安全降级到
/// [`crate::CpuRefBackend`]。**macOS 实机验证未做**（V2.1 非目标，Q5）。
pub struct MetalBackend {
    /// 是否已在 macOS 上探测到 Metal 设备（仅供诊断；**不影响** `is_available`）。
    ///
    /// 非 macOS 平台恒为 `false`。
    probed_device: bool,
}

impl MetalBackend {
    /// 构造一个 Metal 后端把手，并立即尝试探测设备（macOS）或直接置为未探测
    /// （其他平台）。任何失败都安全降级，**永不 panic**。
    pub fn new() -> Self {
        MetalBackend {
            probed_device: Self::probe_device_present(),
        }
    }

    /// 平台派发的设备探测：macOS 走真实 dlopen 链，其余平台恒 `false`。
    fn probe_device_present() -> bool {
        #[cfg(target_os = "macos")]
        {
            match loader::probe::default_device_probe() {
                loader::DeviceProbe::Present { .. } => true,
                loader::DeviceProbe::Unavailable { .. } => false,
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Metal 仅 macOS 可用；Linux/wasm 上没有任何 Metal 运行时。
            false
        }
    }

    /// 完整探测（诊断用）：返回本次探测的结构化结果。
    ///
    /// * macOS：返回 [`ProbeReport`]，含设备有无、设备数量与失败原因。
    /// * 非 macOS：返回 [`ProbeReport::unsupported_platform`]。
    pub fn probe() -> ProbeReport {
        #[cfg(target_os = "macos")]
        {
            loader::probe::full_probe()
        }
        #[cfg(not(target_os = "macos"))]
        {
            ProbeReport::unsupported_platform()
        }
    }

    /// 是否在启动时探测到 Metal 设备（**诊断量，非可用性**）。
    ///
    /// 非 macOS 恒 `false`。请勿用此值代替 [`GpuDevice::is_available`] 做
    /// 渲染路径选择——二者语义不同（见 [`MetalBackend`] 类型文档）。
    pub fn device_present(&self) -> bool {
        self.probed_device
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice for MetalBackend {
    /// Metal 渲染管线**代码就绪但未点亮**：**返回空缓冲而非假帧**。
    ///
    /// V2.1 T-A2 交付的是「同构代码 + 编译门控」：MSL 源（[`msl::RT_SCENE_MSL`]）、
    /// 参数 ABI（[`render::encode_scene_params`]）与语义同构实现
    /// （[`render::render_scene_msl_semantics`]）都已落地并与 `CpuRefBackend`
    /// 逐字节对拍通过；但**没有** `MTLDevice` → pipeline → dispatch 的闭环，
    /// 所以这里仍如实返回空缓冲（`is_valid() == false`），调用方回退 CPU。
    ///
    /// 刻意不返回一个全黑的有效帧去冒充成功——那会污染上层判对逻辑。
    /// 本函数在任何平台、任何输入下都不 panic（C18）。
    fn render_scene(&mut self, _req: &SceneRenderRequest) -> Framebuffer {
        Framebuffer::empty()
    }

    /// 恒 `false`：能探测到设备不代表能出图（render pipeline 未点亮）。
    ///
    /// V2.1 T-A2 只交付代码 + 编译门控（Q5）：MSL/ABI/语义同构都已落地，
    /// 但 **macOS 实机验证未做**，`MTLDevice` → dispatch → 读回的闭环不存在。
    /// 见 [`MetalBackend`] 类型文档的「为什么恒为 false」。
    fn is_available(&self) -> bool {
        false
    }

    /// 稳定标识：所有平台一致，便于能力日志对齐。
    fn backend_name(&self) -> &'static str {
        "metal-stub"
    }
}

// ---------------------------------------------------------------------------
// 探测报告（跨平台可见的诊断结构）
// ---------------------------------------------------------------------------

/// 一次 Metal 能力探测的结构化结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeReport {
    /// 探测是否在受支持平台（macOS）执行。
    pub supported_platform: bool,
    /// 是否拿到至少一个 `MTLDevice`。
    pub device_present: bool,
    /// 枚举到的设备数量（`MTLCopyAllDevices` 计数；未拿到时 `0`）。
    pub device_count: usize,
    /// 失败/降级原因（成功时为 `None`）。
    pub reason: Option<ProbeFailure>,
}

impl ProbeReport {
    /// 非 macOS 平台的标准报告。
    pub fn unsupported_platform() -> Self {
        ProbeReport {
            supported_platform: false,
            device_present: false,
            device_count: 0,
            reason: Some(ProbeFailure::UnsupportedPlatform),
        }
    }
}

/// 探测降级原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeFailure {
    /// 非 macOS：本平台不存在 Metal。
    UnsupportedPlatform,
    /// `dlopen("/System/Library/Frameworks/Metal.framework/Metal")` 失败
    /// （框架缺失或路径变更）。
    FrameworkLoadFailed,
    /// 框架已加载但 `dlsym` 未找到 `MTLCreateSystemDefaultDevice` 符号。
    SymbolMissing,
    /// 符号存在但调用返回空指针 ⇒ 无默认 GPU。
    NoDefaultDevice,
}

// ---------------------------------------------------------------------------
// 单元测试（零 GPU 依赖；在 Linux 上验证 stub 语义）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CameraReq, LightReq, SceneRenderRequest, SphereReq, AestheticReq};
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

    /// 非 macOS 平台：Metal 恒不可用。
    ///
    /// **非恒真自证**：在 macOS 上，本断言的前半部分**会继续成立**（`is_available`
    /// 仍恒 `false`，因为渲染管线未实装），但 [`MetalBackend::probe`] 的
    /// `supported_platform` 与 `device_present` 将变为 `true`（若无 GPU 则
    /// `device_present` 为 `false` 且 `reason == Some(NoDefaultDevice)`）。此处
    /// 只断言平台无关的部分，故在两种平台上都**不会**变成恒真的空断言。
    #[test]
    fn metal_stub_is_unavailable_and_stable_name() {
        let be = MetalBackend::new();
        assert!(!be.is_available(), "探测到设备也不等于可渲染，须恒 false");
        assert_eq!(be.backend_name(), "metal-stub", "后端标识须跨平台稳定");
    }

    /// `render_scene` 返回**无效**空缓冲，绝不产假帧、不 panic。
    ///
    /// **非恒真自证**：`is_valid() == false` 的断言在 macOS 上同样成立（渲染
    /// 未实装）；若未来 macOS 实装渲染，本测试**必须**改为断言
    /// `is_valid() == true` 与像素内容——这正是本断言此刻为"非恒真"的证据：
    /// 它把一个真实契约（未实装 ⇒ 无有效帧）钉死，而非永真。
    #[test]
    fn metal_stub_render_scene_returns_invalid_frame_without_panic() {
        let req = tiny_request();
        let mut be = MetalBackend::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.render_scene(&req)
        }));
        let fb = r.expect("Metal stub render_scene 不得 panic");
        assert!(!fb.is_valid(), "未实装渲染须返回无效缓冲，不得冒充成功");
        assert_eq!(fb.pixels.len(), 0, "空缓冲不含像素");
        assert_eq!((fb.width, fb.height), (0, 0), "empty() 为 0x0");
    }

    /// `probe()` 在非 macOS 上给出 `unsupported_platform` 报告。
    ///
    /// **非恒真自证**：macOS 上 `supported_platform == true`，故断言
    /// `!report.supported_platform` 会**失败**——这不是恒真断言，而是对
    /// "本机平台" 的真实刻画。
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn probe_on_non_macos_reports_unsupported() {
        let report = MetalBackend::probe();
        assert!(!report.supported_platform);
        assert!(!report.device_present);
        assert_eq!(report.device_count, 0);
        assert_eq!(report.reason, Some(ProbeFailure::UnsupportedPlatform));
    }

    /// `device_present()` 非 mac 恒 false（诊断量与可用性分离）。
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn device_present_false_on_non_macos() {
        let be = MetalBackend::new();
        assert!(!be.device_present());
    }

    /// MSL 首样例存在且非空，入口名与组织约定一致（跨平台可查）。
    ///
    /// 该断言在非 macOS 平台与 macOS 上均成立且**有意义**：它证明「双份源」的第一个
    /// 手写样例已被登记，而非仅有目录约定。
    #[test]
    fn msl_first_sample_is_registered() {
        assert!(
            msl::FIRST_SAMPLE_ENTRY.eq("cloth_predict"),
            "首样例入口应为 sim_cloth.wgsl 的 cloth_predict"
        );
        assert!(
            !msl::CLOTH_PREDICT_MSL.trim().is_empty(),
            "MSL 源码不得为空"
        );
        assert!(
            msl::CLOTH_PREDICT_MSL.contains("kernel void cloth_predict"),
            "MSL 须声明 kernel void cloth_predict 入口"
        );
    }

    /// **V2.1 T-A2 收口**：同构代码已落地，但可用性**不得**因此改变（G4/Q5）。
    ///
    /// 这条测试同时锁住两件相反方向的断言，避免「写了一堆代码就顺手点亮」：
    /// * 代码确实在（MSL 源 / 入口 / 4 绑定 / CPU 语义实现可跑且与参考一致）；
    /// * `is_available()` 仍恒 `false`、`render_scene` 仍返回无效帧
    ///   （没有 `MTLDevice` → dispatch 闭环，就不能声称可用）。
    #[test]
    fn render_code_landed_without_claiming_availability() {
        let mut be = MetalBackend::new();
        // ① 代码在。
        assert!(
            msl::RT_SCENE_MSL.contains("kernel void scene_main"),
            "MSL 场景 kernel 必须已登记"
        );
        assert_eq!(render::ENTRY_SCENE_MAIN, msl::RT_SCENE_ENTRY);
        assert_eq!(render::BINDINGS.len(), 4, "四绑定（params/spheres/lights/out）");
        // ② 语义同构真的能跑（纯 CPU，跨平台），且与 CpuRefBackend 一致。
        let req = tiny_request();
        let msl_fb = render::render_scene_on_cpu_via_msl_layout(&req);
        assert!(msl_fb.is_valid(), "MSL 语义实现应能出帧（纯 CPU）");
        let mut cpu = crate::CpuRefBackend;
        let cpu_fb = cpu.render_scene(&req);
        assert_eq!(
            msl_fb.pixels, cpu_fb.pixels,
            "MSL 语义实现必须与 CPU 参考逐字节一致"
        );
        // ③ 但能力位不因此改变：没有 dispatch 闭环就不许报可用。
        assert!(
            !be.is_available(),
            "有源码/有语义对拍 ≠ 有 dispatch 闭环 ⇒ 不得谎报可用（G4）"
        );
        assert!(!be.render_scene(&req).is_valid(), "不得产假帧");
        assert_eq!(be.backend_name(), "metal-stub");
    }
}
