//! `prism-gpu` — GPU/CPU rendering abstraction for the Prism engine.
//!
//! This crate defines the frame-buffer type, the low-level scene render
//! request structs (constructed by `prism-core`), and the [`GpuDevice`]
//! trait that abstracts over rendering backends.
//!
//! Backends provided:
//! * [`CpuRefBackend`] — a real, runnable software ray-tracer (CPU reference).
//! * [`vulkan::VulkanBackend`] — a native Vulkan backend with a real
//!   run-time device-probe chain (hand-written `dlopen`/`dlsym` FFI, zero
//!   third-party crates). The **compute dispatch path is real and verified**
//!   (SPIR-V → pipeline → dispatch → fence → readback → bitwise CPU parity),
//!   and [`GpuDevice::is_available`] reports `true` only when **both** the
//!   compute self-test **and** the scene-render self-test pass
//!   (`compute_available() && render_available()`). Since W4 the backend also
//!   **produces real frames**: [`background::render_background_gpu`] runs the
//!   background kernel offscreen and reads back RGBA8 that is byte-identical
//!   to the CPU reference. Since V2.1 **the ray-traced scene pipeline is
//!   real too**: [`vulkan::scene`] dispatches a 4-binding compute megakernel
//!   (emitted by [`vulkan::spirv_rt`], with dynamic sphere/light loops) and
//!   reads back RGBA8 for [`GpuDevice::render_scene`]; GPU-vs-CPU parity is
//!   statistical (bitwise identity is guaranteed *across GPU runs*, not
//!   across backends — see `tests/qa_render_scene.rs`). When no device is
//!   present both paths degrade honestly: `render_scene` returns an invalid
//!   [`Framebuffer::empty`] and callers fall back to [`CpuRefBackend`].
//! * [`web::device::WebGpuBackend`] — the WebGPU path (hand-written browser
//!   bindings) used by the wasm/mobile test link.
//!
//! Only `std` is used. `Vec3`/`Mat4` come from the locked `prism-math` API.

use prism_math::Vec3;

/// 曲线图元请求（A-2）：几何权威在叶子 crate `prism-geometry`（`curve_tess`）。
///
/// `prism-gpu → prism-geometry` 为单向 path 依赖（设计 §9-Q5 已拍板）；本 crate
/// **不重写**三角化/BVH，只做扁平化编码（[`scene_abi::encode_curve_buffers`]）与
/// CPU 参考的求交接线。
pub use prism_geometry::curve_tess::CurveReq;

// ----------------------------------------------------------------------------
// WGSL shader sources + structural self-check (D2/D3/D4)
// ----------------------------------------------------------------------------

pub mod shaders;
pub mod scene_abi;
pub mod instance;
pub mod cpu_tex;

// ----------------------------------------------------------------------------
// 背景渲染（GPU 管线兜底）：CPU 参考 + 引擎入口
// ----------------------------------------------------------------------------
//
// `background` 把 demo 的逐像素 `paint_background` 热点迁移为无 GPU 平台完整可运行的
// CPU 后端，并作为 GPU 实现的数值判对基准。W4 起 GPU 路径**真能出图**：
// `render_background` 走 Vulkan compute（`vulkan::background` + 自研发射器
// `vulkan::spirv_bg`），读回的 RGBA8 与 CPU 参考逐字节一致；无可用设备时
// 静默回退 CPU（不 panic）。

pub mod background;

// ----------------------------------------------------------------------------
// WebGPU 手写绑定后端（D1；零第三方 crate — D-15）
// ----------------------------------------------------------------------------
//
// `web` 模块用 `extern "C"` 手写触达浏览器 `navigator.gpu`，不依赖
// wasm-bindgen/web-sys；`web::device::WebGpuBackend` 实现 [`GpuDevice`]，
// 把 D0 的 scan/sort 承载为 WGSL compute 内核（源见 `WGSL_SCAN_BLOCK` /
// `WGSL_HISTOGRAM`）。native 目标下提供可编译桩并安全降级 CPU。

pub mod web;

// ----------------------------------------------------------------------------
// Vulkan 原生后端
// ----------------------------------------------------------------------------
//
// 手写 `dlopen`/`dlsym` + `#[repr(C)]` Vulkan struct 声明，**零第三方 crate**
// （不引 `ash`/`vulkano`/`libloading`；D-15）。编译期不链接 `libvulkan`——
// loader 在**运行期**探测，拿不到就不可用。compute 闭环已真跑通：SPIR-V →
// compute pipeline → 描述符 → dispatch → fence → 读回 → 与 CPU 参考逐位比对；
// `is_available()` 只在**compute 自检 ∧ 场景渲染自检**都为真时为 `true`
// （探测到设备不够，单条闭环通过也不够）。两条出图路径都已实装：背景帧
// （与 CPU 逐字节一致）与光追场景帧（与 CPU 统计等价、GPU 双跑逐位一致）。
// 无设备时 `render_scene` 返回无效空缓冲（不撒谎），调用方按返回帧的
// `is_valid()` 回退 CPU。wasm32 下 `loader` 被门禁剔除，`VulkanBackend` 恒报
// 不可用。

pub mod vulkan;

// ----------------------------------------------------------------------------
// Metal 后端（D1，macOS/iOS 独有；非 macOS 编译为空 stub）
// ----------------------------------------------------------------------------
//
// [`metal::MetalBackend`] 在所有平台可见，但 `is_available()` 恒 `false`：
// V2.1 T-A2 只交付「同构代码 + 编译门控」——MSL 源、参数 ABI 与语义同构实现
// 都已落地（并与 CPU 参考逐字节对拍通过），但 `MTLDevice` → pipeline →
// dispatch → 读回的闭环**未接**（缺 `objc_msgSend` 层），**macOS 实机验证未做**。
// 故「有源码」≠「能出图」：`render_scene` 恒返回无效空缓冲，调用方回退 CPU。
// macOS 上的设备发现走 `dlopen` + `dlsym`（零第三方 crate）；Linux/wasm 上
// 该子模块被 `#[cfg]` 整体剔除，仅保留可编译的降级 stub，绝不影响 Linux 与
// wasm 构建。

pub mod metal;

/// 便捷导出：Metal 后端把手（各平台一致可见）。
pub use metal::MetalBackend;

/// 背景渲染入口与 CPU 参考（GPU 优先、CPU 兜底）。
pub use background::{render_background, render_background_cpu, render_background_gpu};

// ----------------------------------------------------------------------------
// D0 GPU 基础设施（设备无关算法层）
// ----------------------------------------------------------------------------
//
// 承接既有骨架（本 crate 既有的 `GpuDevice` / `Framebuffer` / 后端占位）：
//
// * **设备无关算法**（本层纯数据变换，CPU 可跑可测，真机 backend 只需提供
//   层内并行 dispatch）：
//     - [`scan`]：前缀和（Blelloch work-efficient 上下行扫描 + 串行参考）
//     - [`sort`]：radix sort（LSD 多趟稳定计数排序，复用 scan 做 ExScan）
// * **设备相关语义模型**（形态封闭，backend 依此派生物理指令）：
//     - [`sync`]：五类 pass 形态 + 屏障/同步推导
//
// 依赖边界（D-15）：零第三方 crate 承诺延伸至 GPU 侧；本 crate 仅依赖
// `prism-math`（数学）。注意 `prism-core` 依赖本 crate（骨架方向），
// 故此处不得反向依赖 prism-core；`scan` 用本地 `ScanShape` 复刻
// `reduction_shape`（D-26 单点定义）的树形语义，二者口径由测试对齐。
// GPU 真机路径（Vulkan/WebGPU/Metal 手写绑定）标注待接入。

pub mod scan;
pub mod sort;
pub mod sync;

// **compute 语义层**（GPU compute-first 管线的基础件）：
// 设备无关的 kernel 描述 / dispatch 形状 / buffer 布局 + 真实 kernel 语义的
// CPU 参考实现（主射线发射、wavefront 求交一步）。真机执行经
// [`vulkan::compute`] 的装配路径；本层保证「语义与设备无关」这一条硬约束。
pub mod compute;

// ----------------------------------------------------------------------------
// Framebuffer
// ----------------------------------------------------------------------------

/// Maximum accepted value for a single frame-buffer axis (`width`/`height`).
///
/// Frames larger than this on either axis are refused by [`Framebuffer::new`].
/// 65536 is the largest power-of-two edge that keeps `MAX_FRAMEBUFFER_BYTES`
/// reachable for square frames while bounding every downstream index
/// computation to 32-bit-friendly magnitudes.
pub const MAX_FRAMEBUFFER_DIM: u32 = 65_536;

/// Maximum accepted size of the RGBA8 pixel buffer, in bytes (1 GiB).
///
/// Rationale: `u32::MAX * u32::MAX * 4` is ~73 EiB — a request that can only
/// ever be a bug or a denial-of-service. Anything above this ceiling is refused.
pub const MAX_FRAMEBUFFER_BYTES: usize = 1 << 30;

/// Bytes per pixel in the RGBA8 layout (`[r, g, b, a]`).
pub const BYTES_PER_PIXEL: usize = 4;

/// A simple RGBA8 frame buffer backed by a flat `Vec<u8>`.
///
/// The pixel buffer is laid out row-major with [`BYTES_PER_PIXEL`] bytes per
/// pixel (`[r, g, b, a]`), so `pixels.len() == width * height * 4`.
///
/// # Size contract
///
/// [`Framebuffer::new`] **never panics and never aborts**. A requested size is
/// *accepted* only when both axes are `<= [`MAX_FRAMEBUFFER_DIM`]` and the total
/// byte size is `<= [`MAX_FRAMEBUFFER_BYTES`]` (and the allocation succeeds).
///
/// * **Accepted** → `pixels.len() == width * height * 4` holds strictly.
/// * **Refused** (dimension/byte ceiling exceeded, `checked_mul` overflow, or
///   allocation failure) → `pixels` is **empty** while `width`/`height` still
///   echo the requested values verbatim, so a caller can report the refusal.
///   The `pixels.len() == width * height * 4` invariant then does not hold, by
///   design: it is guaranteed only for accepted sizes. Every read/write path is
///   bounds-checked, so a refused buffer is inert (see [`Framebuffer::is_valid`]).
#[derive(Debug, Clone)]
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Framebuffer {
    /// Allocate a zeroed frame buffer of the given size (all pixels transparent black).
    ///
    /// Over-limit requests are refused rather than panicking or aborting: see the
    /// type-level *Size contract*. Use [`Framebuffer::is_valid`] to distinguish a
    /// real frame from a refused one.
    pub fn new(width: u32, height: u32) -> Self {
        let Some(len) = Self::checked_pixel_len(width, height) else {
            // Refused: record the requested size, allocate nothing.
            return Framebuffer {
                width,
                height,
                pixels: Vec::new(),
            };
        };
        // `vec![0u8; len]` aborts the process when the allocator refuses, which
        // would be an abort (not a panic) — still a C18 violation. Reserve
        // fallibly and degrade to a refused buffer instead.
        let mut pixels: Vec<u8> = Vec::new();
        if pixels.try_reserve_exact(len).is_err() {
            return Framebuffer {
                width,
                height,
                pixels: Vec::new(),
            };
        }
        pixels.resize(len, 0u8);
        Framebuffer {
            width,
            height,
            pixels,
        }
    }

    /// A refused / empty frame buffer (`0 x 0`, no pixels).
    ///
    /// Returned by backends that cannot service a request, so that callers always
    /// get a value and never a panic.
    pub fn empty() -> Self {
        Framebuffer {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        }
    }

    /// `Some(width * height * 4)` when the size is within both ceilings and the
    /// multiplication does not overflow `usize`; `None` when it must be refused.
    fn checked_pixel_len(width: u32, height: u32) -> Option<usize> {
        if width > MAX_FRAMEBUFFER_DIM || height > MAX_FRAMEBUFFER_DIM {
            return None;
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(BYTES_PER_PIXEL))?;
        if len > MAX_FRAMEBUFFER_BYTES {
            return None;
        }
        Some(len)
    }

    /// Whether this buffer actually holds a renderable image.
    ///
    /// `false` for buffers produced by a refused [`Framebuffer::new`] or by
    /// [`Framebuffer::empty`]; such buffers are safe to pass around and index
    /// into (all writes are dropped), but carry no pixels.
    pub fn is_valid(&self) -> bool {
        !self.pixels.is_empty() && self.width > 0 && self.height > 0
    }

    /// Byte offset of pixel `(x, y)`, or `None` if the index computation
    /// overflows `usize` (possible only for absurd `x`/`y` on huge frames).
    #[inline]
    fn pixel_offset(&self, x: u32, y: u32) -> Option<usize> {
        (y as usize)
            .checked_mul(self.width as usize)
            .and_then(|row| row.checked_add(x as usize))
            .and_then(|idx| idx.checked_mul(BYTES_PER_PIXEL))
    }

    /// Write a single pixel. `c` components are expected in `0.0..=1.0`; they are
    /// clamped, scaled by 255 and stored as RGB with a fully-opaque alpha (255).
    ///
    /// Out-of-range coordinates (including `u32::MAX`) are silently dropped; the
    /// index math is overflow-checked, so this never panics on any input.
    pub fn set_pixel(&mut self, x: u32, y: u32, c: Vec3) {
        // 行宽守卫：`pixel_offset` 只算 `y*width + x` 并按 `pixels.len()` 做**全局**
        // 末尾检查，不查 `x < width`。故 `x == width`（越界列）会回绕到**下一行**的
        // 第 0 列 —— 静默写错像素，违反上方"越界坐标被丢弃"的契约。
        if x >= self.width || y >= self.height {
            return;
        }
        let Some(idx) = self.pixel_offset(x, y) else {
            return;
        };
        let Some(last) = idx.checked_add(BYTES_PER_PIXEL - 1) else {
            return;
        };
        if last >= self.pixels.len() {
            return;
        }
        self.pixels[idx] = channel_to_u8(c.x);
        self.pixels[idx + 1] = channel_to_u8(c.y);
        self.pixels[idx + 2] = channel_to_u8(c.z);
        self.pixels[idx + 3] = 255;
    }

    /// Read a single pixel as `[r, g, b, a]`, or `None` when `(x, y)` is out of
    /// range or the index computation overflows.
    ///
    /// The safe counterpart of the raw `pixels[y * width + x]` indexing used in
    /// tests; returns `None` instead of panicking.
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        // 与 `set_pixel` 同一行宽守卫：`x == width` 会回绕读下一行第 0 列，
        // 属于静默返回错误像素。
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = self.pixel_offset(x, y)?;
        let last = idx.checked_add(BYTES_PER_PIXEL - 1)?;
        if last >= self.pixels.len() {
            return None;
        }
        Some([
            self.pixels[idx],
            self.pixels[idx + 1],
            self.pixels[idx + 2],
            self.pixels[idx + 3],
        ])
    }

    /// Reallocate to `width x height`, discarding contents (zero-filled).
    ///
    /// Behaves like [`Framebuffer::new`]: over-limit sizes yield a refused
    /// (empty-pixels) buffer instead of panicking.
    pub fn resize(&mut self, width: u32, height: u32) {
        *self = Framebuffer::new(width, height);
    }
}

/// Convert a `0.0..=1.0` float channel to an 8-bit value.
#[inline]
fn channel_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ----------------------------------------------------------------------------
// Low-level render requests (built by `prism-core`)
// ----------------------------------------------------------------------------

/// Camera placement and projection parameters for one frame.
#[derive(Debug, Clone)]
pub struct CameraReq {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub aspect: f32,
}

/// A single directional light source.
#[derive(Debug, Clone)]
pub struct LightReq {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

/// 球材质四张贴图句柄的槽数（= [`scene_abi::TEXMAP_SLOTS`]）。
///
/// 与 `prism_rt::bsdf::MAP_SLOTS` 同值；两份常量不能互相 import
/// （`prism-gpu` 不得反引 `prism-rt`），逐格对拍在 `prism-rt` 侧的测试里做。
pub const SPHERE_MAP_SLOTS: usize = 4;

/// 「没有贴图」句柄（与 `prism_rt::texture::NO_TEXTURE` 同值）。
///
/// **0 不是"第 0 张贴图"**：句柄 = 纹理下标 + 1，故 1 才是第一张。
pub const NO_TEXTURE_MAP: u32 = 0;

/// 四槽全无贴图的默认句柄组（每个构造点的显式初值）。
///
/// 存在的理由：`SphereReq` 有 20+ 处构造点，全写 `[0; 4]` 看不出"这是有意的
/// 无贴图"还是"随手填的 0"；具名常量让每处都是**一次显式声明**。
pub const NO_MAPS: [u32; SPHERE_MAP_SLOTS] = [NO_TEXTURE_MAP; SPHERE_MAP_SLOTS];

/// A ray-traced sphere with physically-inspired shading parameters.
///
/// # 材质贴图（V3 剖面）
///
/// [`SphereReq::maps`] 的四个槽与 `prism_rt::bsdf::Bsdf::maps` **逐格同义**：
/// 槽序 `0 = base color` / `1 = metal-rough` / `2 = normal` / `3 = emissive`，
/// `0 = NO_TEXTURE`，句柄 `k` 指向 `textures[k-1]`。V2 剖面的球块只有 8 槽
/// （[`scene_abi::SPHERE_SLOTS_V2`]）放不下它们，故 V2 编码器**不写**
/// （[`scene_abi::encode_spheres`]）；V3 编码器写进槽 8..11
/// （[`scene_abi::encode_spheres_v3`]）。
#[derive(Debug, Clone)]
pub struct SphereReq {
    pub center: Vec3,
    pub radius: f32,
    pub albedo: Vec3,
    pub roughness: f32,
    pub metalness: f32,
    /// 四张贴图句柄（槽序同 `Bsdf::maps`；`0` = 无贴图）。
    ///
    /// 缺省 [`NO_MAPS`] ⇒ 与 W1-B 之前的渲染输出**逐位相同**（没有贴图时
    /// CPU 参考与 GPU 都不进纹理分支）。
    pub maps: [u32; SPHERE_MAP_SLOTS],
}

impl Default for SphereReq {
    /// 全零球（`maps` = [`NO_MAPS`]）。只为让"新增字段"不改变既有语义；
    /// **构造点仍须逐字段显式写出**（不许用 `..Default::default()` 蒙过）。
    fn default() -> Self {
        SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.0,
            albedo: Vec3::new(0.0, 0.0, 0.0),
            roughness: 0.0,
            metalness: 0.0,
            maps: NO_MAPS,
        }
    }
}

/// Post-processing / look-development flags applied per pixel.
#[derive(Debug, Clone)]
pub struct AestheticReq {
    pub teal_orange: bool,
    pub contrast: f32,
    pub filmic: bool,
}

/// A fully-specified scene render request handed to a [`GpuDevice`].
#[derive(Debug, Clone)]
pub struct SceneRenderRequest {
    pub width: u32,
    pub height: u32,
    pub camera: CameraReq,
    pub lights: Vec<LightReq>,
    pub spheres: Vec<SphereReq>,
    pub aesthetic: AestheticReq,
    /// 曲线图元（V2.2-A-2 方案乙）。
    ///
    /// **缺省空 ⇒ 与 V2.1 输出逐位相同**（这是 A-2 不回归的机械保证，见设计
    /// §4.3）：空集时 CPU 参考不建 BVH、GPU 侧遍历段整段跳过。
    pub curves: Vec<CurveReq>,
}

/// 曲线命中面的着色反照率（A-2 切片口径）。
///
/// `CurveReq` 只携带**几何**（中心/半径/轴/轮廓），不带材质——A-2 只承诺"曲线
/// 图元进入既有 Möller–Trumbore + 直接光管线"，材质参数化留 V2.3。中性灰
/// 反照率使曲线与球的直接光响应可对拍（球带 albedo，曲线固定此值）。
pub const CURVE_ALBEDO: [f32; 3] = [0.72, 0.74, 0.78];

/// 曲线求交的 `t_min`（避免自交；与三角化弦高无关，纯数值守卫）。
pub const CURVE_T_MIN: f32 = 1e-4;

// ----------------------------------------------------------------------------
// Device abstraction
// ----------------------------------------------------------------------------

/// Abstraction over any rendering backend (CPU software or real GPU).
pub trait GpuDevice {
    /// Render one frame described by `req` and return an RGBA8 frame buffer.
    ///
    /// Implementations must never panic (宪章 C18). A backend that cannot service
    /// the request returns a refused buffer (see [`Framebuffer::empty`] and
    /// [`Framebuffer::is_valid`]) instead.
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer;

    /// Whether this backend can actually perform **verified real-device work**.
    ///
    /// # 语义（V2.1 T-A1 收口）
    ///
    /// 本标志回答的是「**能否 service [`GpuDevice::render_scene`]**」——
    /// 对 [`vulkan::VulkanBackend`] 而言就是 **compute 自检 ∧ 渲染自检**：
    /// `true` 仅当 mul2 探针的完整闭环（SPIR-V → pipeline → submit → fence →
    /// 读回 → 与 CPU 逐位一致）**和**场景 kernel 的 4×4 探针帧（读回长度自洽
    /// 且确有内容）都真跑通过。
    ///
    /// 两个方向都被钉死（不再有中间态）：
    /// * `true` ⇒ [`GpuDevice::render_scene`] 会给出 `w*h*4` 的有效帧；
    /// * `false` ⇒ `render_scene` 返回无效空缓冲，调用方回退
    ///   [`CpuRefBackend`]——**不存在**「报可用却出空帧」的组合。
    ///
    /// 背景帧是另一条可查询的独立能力位（`VulkanBackend::frame_available`，
    /// 真跑一次背景 kernel 并确认读回长度），一条通过不替另一条背书。
    ///
    /// 之所以保留「能力位」而非恒 `true`/恒 `false`：本 crate 的 backend 是
    /// compute-first 的（scan/sort/kernel 语义层都建立在 dispatch 之上），
    /// 谎报 `false` 会让所有 compute 消费方失去能力信息；谎报 `true` 则会让
    /// 上层拿到假帧。
    ///
    /// 默认 `true` 表示「纯软件后端恒可用」（对 [`CpuRefBackend`] 正确）。
    fn is_available(&self) -> bool {
        true
    }

    /// Stable backend identifier, for diagnostics and capability logging.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }
}

// ----------------------------------------------------------------------------
// 美学后处理链（两档渲染器共用，单一事实源）
// ----------------------------------------------------------------------------

/// 逐像素后处理链：teal-orange → contrast → filmic(Reinhard) → clamp01。
///
/// # 为什么要单独成函数（而不是留在 `CpuRefBackend` 里内联）
///
/// D-19 双轨有**两个**产品渲染器（[`CpuRefBackend`] 与 `prism-render` 的
/// `RtBackend`）。若两条路径各自内联一份后处理，微调一处（比如换 tone-map
/// 曲线）就会让"两档差异"里混进后期差异，A/B 归因立刻失效——这正是
/// `docs/v2.5-v3.0-路线图.md` §4.4「不得另起近似路径」要防的第二个事实源。
/// 故本函数是**唯一**实现：`CpuRefBackend` 与 `RtBackend` 都调它。
///
/// # 语义
///
/// 顺序与常量逐字固定（改任何一个都会改变 `--fast` 的既有输出）：
/// 1. `teal_orange`：亮度分流的双色补偿；
/// 2. `contrast != 1.0`：以 0.5 为中心的线性对比度；
/// 3. `filmic`：Reinhard `c / (1 + c)`；
/// 4. 逐分量 clamp 到 `0.0..=1.0`（`Framebuffer::set_pixel` 随后 ×255 取整）。
pub fn apply_aesthetic(color: Vec3, aesthetic: &AestheticReq) -> Vec3 {
    let mut color = color;
    if aesthetic.teal_orange {
        let luma = 0.299 * color.x + 0.587 * color.y + 0.114 * color.z;
        let shadow_tint = Vec3::new(0.10, 0.35, 0.45);
        let high_tint = Vec3::new(1.0, 0.7, 0.3);
        let shadow_comp = vmul(shadow_tint, color) * (1.0 - luma) * 0.5;
        let high_comp = vmul(high_tint, color) * luma * 0.5;
        color = color * 0.6 + shadow_comp + high_comp;
    }

    if aesthetic.contrast != 1.0 {
        color = (color - Vec3::splat(0.5)) * aesthetic.contrast + Vec3::splat(0.5);
    }

    if aesthetic.filmic {
        // Simple Reinhard tone-mapping.
        color = vdiv(color, color + Vec3::splat(1.0));
    }

    clamp01(color)
}

// ----------------------------------------------------------------------------
// CPU reference backend (real, runnable software ray tracer)
// ----------------------------------------------------------------------------

/// A pure-CPU reference renderer that ray-traces spheres per-pixel.
///
/// This backend is fully functional and needs no GPU; it exists both as a
/// testable ground-truth implementation and as a fallback renderer.
pub struct CpuRefBackend;

/// CPU 参考侧的实例场景（已解码的实例记录 + TLAS 节点）。
///
/// 字段私有：唯一构造入口是 [`CpuRefBackend::render_scene_instanced`]（它从
/// `encode_v3_instanced` 的字节流解码），`Default` = 空实例（= 无实例路径）。
#[derive(Debug, Clone, Default)]
struct InstanceScene {
    /// 实例记录（已按 TLAS 叶子序重排）。
    records: Vec<scene_abi::InstanceReq>,
    /// TLAS 扁平节点（**与 GPU 读同一份字节**解码而来）。
    nodes: Vec<crate::instance::TlasNode>,
}

impl CpuRefBackend {
    /// 带纹理的 CPU 参考渲染（W1-C 段 3）：**V3 对拍的基准帧**。
    ///
    /// # 为什么是"新增方法"而不是给 `SceneRenderRequest` 加字段
    ///
    /// 请求结构体是**三后端共用的 ABI 载体**，给它加一个 `Vec<TextureUpload>`
    /// 会让 20+ 处 `render_scene` 调用点全部要动（并且把 `'a` 生命周期传染到
    /// `GpuDevice::render_scene` 的签名上，连带 trait object 也不再成立）。
    /// 纹理是**设备侧资源**、不是"这一帧要画什么"，把它按次传入更贴近真实
    /// 后端（GPU 侧同样是 `encode_v3(req, textures, env)` 分开收）。
    ///
    /// # 不回归
    ///
    /// `textures` 为空 ⇒ 与 [`GpuDevice::render_scene`] **逐字节相同**（同一条
    /// 代码路径，纹理分支整段跳过：`sphere.maps` 全 0 时不进采样）。
    ///
    /// # 语义（与 `scene_frame_v3.wgsl` 逐式相同）
    ///
    /// * 读 `maps[MAP_BASE]`（槽 0）：命中点 uv 处的三线性采样**替换** albedo；
    /// * 读 `maps[MAP_METAL_ROUGH]`（槽 1）：G = roughness、B = metalness（glTF），
    ///   置位即启用两瓣 PBR（diffuse 按 (1−m) 衰减 + F0=0.04 镜面瓣 NdotH⁴）；
    /// * 读 `maps[MAP_NORMAL]`（槽 2）：切线空间法线经 `plane_tangent_basis` 旋到
    ///   世界空间扰动着色法线；
    /// * 句柄 `0` 或越界 ⇒ 按"无贴图"处理（保持球自带 albedo / 纯 Lambert 基线）；
    /// * 足迹走解析足迹（三线性 + mip + 各向异性），见 [`cpu_tex`]。
    pub fn render_scene_with_textures(
        &mut self,
        req: &SceneRenderRequest,
        textures: &[scene_abi::TextureUpload<'_>],
    ) -> Framebuffer {
        self.render_scene_impl(req, textures, &InstanceScene::default())
    }

    /// **W3：带实例的 CPU 参考渲染**（GPU 侧 V3 实例通路的判据基准）。
    ///
    /// # 三个后端读同一份字节（这里是"CPU 也读那份字节"）
    ///
    /// 本函数**先编码再解码**：`SceneAbi::encode_v3_instanced(...)` 产出实例 +
    /// TLAS 的字节流，随后用 [`crate::instance::decode_instances`] /
    /// [`crate::instance::decode_nodes`] 读回来做遍历。这样 CPU 参考与 GPU
    /// 消费的是**同一份字节**（而不是"两边各自从 `InstanceReq` 算一遍"）——
    /// 布局错一位，两边会一起错，但**判据会红**（因为参考帧与 GPU 帧来自同一
    /// 份错误字节时，`differing_pixels` 反而为 0）。故字节布局本身另有单测
    /// （`instance::tests` 的逐槽断言与 prism-rt 的跨 crate 逐字节对拍）。
    ///
    /// # 语义
    ///
    /// * `instances` 为空 ⇒ 与 [`CpuRefBackend::render_scene_with_textures`]
    ///   **逐字节相同**（同一代码路径，实例段整段跳过）；
    /// * 非空 ⇒ `triangles`/`nodes`（来自 `req.curves`）是**共享 BLAS**，
    ///   不再作为世界几何绘制（与 `scene_frame_v3.wgsl` 同一条判断）。
    ///
    /// # 错误（不 panic）
    ///
    /// 实例编码失败（`geom_id != 0` / 原型几何为空）⇒ `Err`。
    pub fn render_scene_instanced(
        &mut self,
        req: &SceneRenderRequest,
        textures: &[scene_abi::TextureUpload<'_>],
        instances: &[scene_abi::InstanceReq],
    ) -> Result<Framebuffer, String> {
        if instances.is_empty() {
            return Ok(self.render_scene_with_textures(req, textures));
        }
        let buffers = scene_abi::SceneAbi::encode_v3_instanced(req, textures, None, instances)?;
        let scene = InstanceScene {
            records: crate::instance::decode_instances(&buffers.instances),
            nodes: crate::instance::decode_nodes(&buffers.instance_nodes),
        };
        Ok(self.render_scene_impl(req, textures, &scene))
    }

    /// [`CpuRefBackend::render_scene_with_textures`] 与
    /// [`CpuRefBackend::render_scene_instanced`] 的共同实现。
    ///
    /// `inst` 为空 ⇒ 与 W1-C 的渲染逐字节相同（实例段整段跳过，零额外分支
    /// 成本以外的行为变化）。
    fn render_scene_impl(
        &mut self,
        req: &SceneRenderRequest,
        textures: &[scene_abi::TextureUpload<'_>],
        inst: &InstanceScene,
    ) -> Framebuffer {
        let mut fb = Framebuffer::new(req.width, req.height);
        // 尺寸被拒（见 Framebuffer 的size contract）⇒ 没有任何像素可写，
        // 且继续按 req.width/height 循环会在超大请求上空转（u32::MAX² 迭代），
        // 因此立即返回拒绝缓冲。
        if !fb.is_valid() {
            return fb;
        }
        let cam = &req.camera;
        // 纹理表：offset 按纹素前缀累加，与 `encode_v3` 同口径（[`cpu_tex`]）。
        let tex_table = cpu_tex::RefTextureTable::new(textures);
        // 无纹理 ⇒ 与 `render_scene` 完全同路径（下面所有纹理判断都短路）。
        let has_textures = !tex_table.is_empty();

        // --- A-2 曲线：三角化 → SAH-BVH（一次性，逐帧复用）-----------------
        // 权威在 `prism-geometry`（叶子 crate）：本 crate 只接线，不重写。
        // 曲线为空 ⇒ `None` ⇒ 下面整段跳过，输出与 V2.1 逐位相同（不回归保证）。
        let curve_tris = if req.curves.is_empty() {
            Vec::new()
        } else {
            prism_geometry::curve_tess::tessellate_all(&req.curves, scene_abi::CURVE_TESS_TOL)
        };
        let curve_bvh = if curve_tris.is_empty() {
            None
        } else {
            Some(prism_geometry::bvh::Bvh::build(&curve_tris))
        };

        // --- Camera orthonormal basis -------------------------------------
        let forward = (cam.target - cam.eye).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = cross(forward, world_up).normalize();
        let up2 = cross(right, forward);

        // D-29：相机焦距标度必须走自研 det_trig（`(fov/2).tan()` 会落到平台
        // libm，跨平台逐位不一致）。与 prism_math::perspective 同一实现。
        let tan_f = prism_math::det_trig::tan(cam.fov * 0.5);
        let aspect = cam.aspect;

        for py in 0..req.height {
            for px in 0..req.width {
                // Map pixel to normalized device-ish screen coordinates.
                let nx = (2.0 * (px as f32 + 0.5) / req.width as f32 - 1.0) * tan_f * aspect;
                let ny = (1.0 - 2.0 * (py as f32 + 0.5) / req.height as f32) * tan_f;

                let dir = (forward + right * nx + up2 * ny).normalize();
                let origin = cam.eye;

                // --- Ray/sphere intersection (nearest hit) -----------------
                let mut closest_t = f32::INFINITY;
                let mut hit: Option<&SphereReq> = None;
                for s in &req.spheres {
                    let oc = origin - s.center;
                    let b = dot(oc, dir);
                    let c = dot(oc, oc) - s.radius * s.radius;
                    let disc = b * b - c;
                    if disc >= 0.0 {
                        let t = -b - disc.sqrt();
                        if t > 0.0 && t < closest_t {
                            closest_t = t;
                            hit = Some(s);
                        }
                    }
                }

                // --- A-2 曲线：扁平 BVH 遍历最近命中（覆盖球命中若更近）----
                // 求交语义与 `prism-geometry/src/bvh.rs::ray_intersect` 逐式一致
                // （Möller–Trumbore + 近子先入栈 + `(t, tri)` 字典序 tie-break），
                // 故 CPU 参考 ≡ BVH 权威件，GPU 侧照抄同式即可对齐。
                let mut curve_hit: Option<Vec3> = None;
                // W3：实例表非空时 `triangles`/`nodes` 是共享 BLAS（局部坐标），
                // 只能经实例到达 ⇒ 不作为世界几何绘制（与 shader 同判据）。
                let inst_active = !inst.records.is_empty() && !inst.nodes.is_empty();
                if let (false, Some(bvh)) = (inst_active, &curve_bvh) {
                    if let Some(h) = bvh.ray_intersect(origin, dir, CURVE_T_MIN, closest_t) {
                        closest_t = h.t;
                        let tri = &curve_tris[h.tri as usize];
                        let mut n = tri.normal_unnormalized().normalize();
                        // 法线朝向来光侧（与球面 `(p - center).normalize()` 同口径：
                        // 球面法线恒朝外，这里按射线侧翻转，保证漫反射不出现负半球）。
                        if dot(n, dir) > 0.0 {
                            n = n * -1.0;
                        }
                        curve_hit = Some(n);
                        hit = None;
                    }
                }

                // --- W3 实例：TLAS 剪枝 + 共享 BLAS 局部求交（最近命中覆盖球/曲线）---
                //
                // 与 `scene_frame_v3.wgsl` 的实例段逐句对应：同一份字节（编码后
                // 解码）、同一 `CURVE_T_MIN`、同一 1e30 空界、同一 world t。
                let mut inst_hit: Option<crate::instance::InstanceHit> = None;
                if inst_active {
                    if let Some(bvh) = &curve_bvh {
                        let t_max_inst = if hit.is_some() { closest_t } else { 1e30 };
                        if let Some(h) = crate::instance::instance_nearest(
                            &inst.nodes,
                            &inst.records,
                            bvh,
                            origin,
                            dir,
                            CURVE_T_MIN,
                            t_max_inst,
                        ) {
                            closest_t = h.t;
                            inst_hit = Some(h);
                            hit = None;
                            curve_hit = None;
                        }
                    }
                }

                // --- Shading -------------------------------------------------
                let mut color = Vec3::new(0.04, 0.05, 0.08); // background
                if let Some(s) = hit {
                    let p = origin + dir * closest_t;
                    let n = (p - s.center).normalize();

                    // ---- 纹理（W1 段 1 + 段 2）：命中点 uv 处解析采样 ----
                    // 判据与 `scene_frame_v3.wgsl` 逐字相同：句柄非 0 **且**在表内，
                    // 否则按"无贴图"处理（越界句柄不 panic、不采样成白）。
                    // 无纹理集时 `has_textures == false` ⇒ 整段短路（V2 行为不回归）。
                    //
                    // 采样走**解析足迹**（与 GPU 同一 `spread0`、`footprint_at`、
                    // SVD 约定）⇒ 三线性 + mip + 各向异性，llvmpipe 上逐位对齐。
                    let mut albedo = s.albedo;
                    let view = -dir;                       // 出射（视线）方向，单位化
                    let mut n_shade = n;                   // 着色法线（默认 = 几何法线）
                    let mut metalness = 0.0f32;
                    let mut roughness = 0.0f32;
                    let mut use_pbr = false;
                    if has_textures {
                        let uv = cpu_tex::sphere_uv(p, s.center, s.radius);
                        let spread0 = 2.0 * tan_f / req.height as f32;
                        let (foot_c0, foot_c1) =
                            cpu_tex::footprint_at(p, n, dir, closest_t, spread0, s.center, s.radius);

                        // MAP_BASE（W1-C）：替换 albedo。
                        let base_map = s.maps[scene_abi::MAP_BASE_SLOT];
                        if tex_table.has(base_map) {
                            if let Some(t) =
                                tex_table.sample_with_footprint(base_map, uv, foot_c0, foot_c1)
                            {
                                albedo = Vec3::new(t[0], t[1], t[2]);
                            }
                        }

                        // MAP_METAL_ROUGH（段 2）：G = roughness、B = metalness（glTF）。
                        // 置位即启用两瓣 PBR（diffuse 按 (1−m) 衰减、加 F0=0.04 镜面瓣）。
                        // 句柄清 0（或越界）⇒ `use_pbr = false` ⇒ 纯 Lambert 基线。
                        let mr_map = s.maps[scene_abi::MAP_METAL_ROUGH_SLOT];
                        if tex_table.has(mr_map) {
                            if let Some(c) =
                                tex_table.sample_with_footprint(mr_map, uv, foot_c0, foot_c1)
                            {
                                roughness = c[1].clamp(0.0, 1.0);
                                metalness = c[2].clamp(0.0, 1.0);
                                use_pbr = true;
                            }
                        }

                        // MAP_NORMAL（段 2）：切线空间法线 → 世界空间扰动着色法线。
                        // `+Z` 朝外、B 已含手性；退化（零长）⇒ 退回几何法线，绝不 NaN。
                        // 句柄清 0 ⇒ `n_shade = n`，与无法线贴图逐字节相同。
                        let nm_map = s.maps[scene_abi::MAP_NORMAL_SLOT];
                        if tex_table.has(nm_map) {
                            if let Some(c) =
                                tex_table.sample_with_footprint(nm_map, uv, foot_c0, foot_c1)
                            {
                                let ns = Vec3::new(
                                    2.0 * c[0] - 1.0,
                                    2.0 * c[1] - 1.0,
                                    2.0 * c[2] - 1.0,
                                );
                                let (tb0, tb1) = cpu_tex::plane_tangent_basis(n);
                                let shaded = tb0 * ns.x + tb1 * ns.y + n * ns.z;
                                if dot(shaded, shaded) > 1e-12 {
                                    n_shade = shaded.normalize();
                                }
                            }
                        }
                    }

                    let ambient = Vec3::splat(0.08);
                    let mut acc = Vec3::splat(0.0);
                    for light in &req.lights {
                        let ldir = light.direction.normalize();
                        let lint = light.intensity;
                        let lcolor = light.color;
                        let diff = dot(n_shade, ldir).max(0.0);
                        // albedo * (ambient + diff * intensity * color)
                        let term = ambient + lcolor * (diff * lint);
                        let mut contrib = vmul(albedo, term);
                        if use_pbr {
                            let h = (ldir + view).normalize();
                            let ndh = dot(n_shade, h).max(0.0);
                            let x2 = ndh * ndh;
                            let spec_lobe = x2 * x2; // NdotH⁴（无超越函数，与 GPU 逐位一致）
                            let f0 = Vec3::new(0.04, 0.04, 0.04)
                                + (albedo - Vec3::new(0.04, 0.04, 0.04)) * metalness;
                            let rough_factor = 1.0 - 0.75 * roughness;
                            let spec = f0 * (spec_lobe * rough_factor);
                            let diffuse = albedo * (1.0 - metalness);
                            let diff_contrib = vmul(diffuse, term);
                            let spec_contrib = vmul(spec, lcolor * (diff * lint));
                            contrib = diff_contrib + spec_contrib;
                        }
                        acc = acc + contrib;
                    }
                    color = acc;
                } else if let Some(h) = inst_hit {
                    // 实例命中：世界法线（`det(A)·A⁻ᵀ·n`，identity 走展平版同一条
                    // 归一化）+ 反照率（`mat_id` 覆盖 → 球表 albedo，哨兵 →
                    // [`CURVE_ALBEDO`]）。直接光项与曲线/球分支**逐式相同**。
                    let n = h.normal;
                    let mat = inst.records[h.inst as usize].mat_id;
                    let albedo = if mat != scene_abi::INSTANCE_MAT_FROM_GEOM
                        && (mat as usize) < req.spheres.len()
                    {
                        req.spheres[mat as usize].albedo
                    } else {
                        Vec3::new(CURVE_ALBEDO[0], CURVE_ALBEDO[1], CURVE_ALBEDO[2])
                    };
                    let ambient = Vec3::splat(0.08);
                    let mut acc = Vec3::splat(0.0);
                    for light in &req.lights {
                        let diff = dot(n, light.direction.normalize()).max(0.0);
                        let term = ambient + light.color * (diff * light.intensity);
                        acc = acc + vmul(albedo, term);
                    }
                    color = acc;
                } else if let Some(n) = curve_hit {
                    // 曲线命中：几何法线 + 固定中性反照率（[`CURVE_ALBEDO`]），
                    // 直接光项与球分支**逐式相同**（同 ambient、同 light 循环）。
                    let albedo = Vec3::new(CURVE_ALBEDO[0], CURVE_ALBEDO[1], CURVE_ALBEDO[2]);
                    let ambient = Vec3::splat(0.08);
                    let mut acc = Vec3::splat(0.0);
                    for light in &req.lights {
                        let diff = dot(n, light.direction.normalize()).max(0.0);
                        let term = ambient + light.color * (diff * light.intensity);
                        acc = acc + vmul(albedo, term);
                    }
                    color = acc;
                }

                // --- Aesthetic post-processing -----------------------------
                fb.set_pixel(px, py, apply_aesthetic(color, &req.aesthetic));
            }
        }

        fb
    }
}

impl GpuDevice for CpuRefBackend {
    /// 无纹理请求（V2 及以前的所有调用点）：委托给
    /// [`CpuRefBackend::render_scene_with_textures`] 并传空纹理集。
    ///
    /// 传空集 ⇒ `RefTextureTable` 为空 ⇒ 纹理分支整段短路 ⇒ 与 W1-B 之前的
    /// 输出**逐字节相同**（既有调用点与既有断言都不动）。
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer {
        self.render_scene_with_textures(req, &[])
    }

    /// CPU reference渲染恒可用（纯软件，无外部依赖）。
    fn is_available(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "cpu-ref"
    }
}

// ----------------------------------------------------------------------------
// Window abstraction (minimal)
// ----------------------------------------------------------------------------

/// A minimal windowing abstraction so renderers can be driven by a host loop.
pub trait Window {
    fn new(title: &str, w: u32, h: u32) -> Self
    where
        Self: Sized;
    fn poll(&mut self);
}

/// A no-op window used for headless/offline rendering.
pub struct HeadlessWindow;

impl Window for HeadlessWindow {
    fn new(_: &str, _: u32, _: u32) -> Self {
        HeadlessWindow
    }
    fn poll(&mut self) {}
}

// ----------------------------------------------------------------------------
// Private math helpers (kept within the locked `prism-math` Vec3 contract)
// ----------------------------------------------------------------------------

#[inline]
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.dot(&b)
}

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(&b)
}

/// Component-wise multiply (not a locked `Vec3` operator, hence a helper).
#[inline]
fn vmul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// Component-wise divide (not a locked `Vec3` operator, hence a helper).
#[inline]
fn vdiv(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x / b.x, a.y / b.y, a.z / b.z)
}

/// Clamp every component of the vector into `0.0..=1.0`.
#[inline]
fn clamp01(c: Vec3) -> Vec3 {
    Vec3::new(
        c.x.clamp(0.0, 1.0),
        c.y.clamp(0.0, 1.0),
        c.z.clamp(0.0, 1.0),
    )
}

// ----------------------------------------------------------------------------
// 单元测试（零第三方依赖，仅 std + #[test]）
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个 32×18 红球渲染请求：球心在原点、相机在 -Z 侧正视球体，
    /// 主光从相机侧打向球面（方向 (0,0,-1)），关闭全部后期特效以获得确定性输出。
    fn red_sphere_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 32,
            height: 18,
            camera: CameraReq {
                eye: Vec3::new(0.0, 0.0, -4.0),
                target: Vec3::new(0.0, 0.0, 0.0),
                // 运行时契约：fov 以弧度计（参见 prism-app/main.rs 的 to_radians
                // 用法与 CpuRefBackend 中 (fov/2).tan() 的弧度消费）。
                fov: 60.0_f32.to_radians(),
                aspect: 32.0 / 18.0,
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
                maps: NO_MAPS,
            }],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        curves: Vec::new(),
        }
    }

    /// 以 `[r, g, b, a]` 形式读回指定像素（行主序 RGBA8）。
    fn pixel(fb: &Framebuffer, x: u32, y: u32) -> [u8; 4] {
        let idx = (y as usize * fb.width as usize + x as usize) * 4;
        [
            fb.pixels[idx],
            fb.pixels[idx + 1],
            fb.pixels[idx + 2],
            fb.pixels[idx + 3],
        ]
    }

    #[test]
    fn framebuffer_set_pixel_roundtrip_and_clamp() {
        let mut fb = Framebuffer::new(4, 3);
        // 常规写入读回一致（含 alpha 强制不透明）。
        fb.set_pixel(1, 2, Vec3::new(1.0, 0.5, 0.0));
        assert_eq!(
            pixel(&fb, 1, 2),
            [255, 128, 0, 255],
            "set_pixel 写入后应可按行主序读回"
        );

        // 超出 [0,1] 的分量应被钳制到 0/255。
        fb.set_pixel(3, 0, Vec3::new(2.0, -1.0, 0.5));
        assert_eq!(
            pixel(&fb, 3, 0),
            [255, 0, 128, 255],
            "通道值应被钳制到 [0,255]"
        );
    }

    #[test]
    fn framebuffer_set_pixel_out_of_bounds_is_safe() {
        let mut fb = Framebuffer::new(4, 3);
        // 越界写入不得 panic，也不得影响既有像素。
        fb.set_pixel(1, 1, Vec3::new(1.0, 1.0, 1.0));
        fb.set_pixel(100, 100, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(pixel(&fb, 1, 1), [255, 255, 255, 255]);
    }

    #[test]
    fn cpu_ref_backend_center_pixel_is_red() {
        let mut device = CpuRefBackend;
        let fb = device.render_scene(&red_sphere_request());

        // 正中心像素 (16,9) 应明显偏红。
        let [r, g, b, a] = pixel(&fb, 16, 9);
        assert!(r > g, "中心像素应偏红：R({}) 应大于 G({})", r, g);
        assert!(r > b, "中心像素应偏红：R({}) 应大于 B({})", r, b);
        assert!(r > 100, "中心像素红色应足够亮：R({}) <= 100", r);
        assert_eq!(a, 255, "渲染像素应为不透明");
    }

    #[test]
    fn cpu_ref_backend_corners_are_dark_background() {
        let mut device = CpuRefBackend;
        let fb = device.render_scene(&red_sphere_request());

        // 四角射线不与球体相交，应落在深色背景上（RGB 各分量 < 40）。
        for (x, y) in [(0u32, 0u32), (31, 0), (0, 17), (31, 17)] {
            let [r, g, b, _] = pixel(&fb, x, y);
            assert!(
                r < 40 && g < 40 && b < 40,
                "角落 ({},{}) 应为深色背景，实际 R={} G={} B={}",
                x,
                y,
                r,
                g,
                b
            );
        }
    }
}
// ----------------------------------------------------------------------------
// 边界守卫：Framebuffer 尺寸上限语义（P1）+ 后端能力查询（P2）
// ----------------------------------------------------------------------------
//
// P1 选定语义：`Framebuffer::new` **永不 panic、永不 abort**；超限即「拒绝」——
// `pixels` 为空但 `width`/`height` 原样回显。不变量
// `pixels.len() == width * height * 4` 仅对**被接受**的尺寸严格成立。

#[cfg(test)]
mod framebuffer_bounds {
    use super::*;

    /// 构造一个最小可用的渲染请求（复用顶层测试的球体场景参数）。
    fn tiny_request() -> SceneRenderRequest {
        SceneRenderRequest {
            width: 4,
            height: 4,
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
                maps: NO_MAPS,
            }],
            aesthetic: AestheticReq {
                teal_orange: false,
                contrast: 1.0,
                filmic: false,
            },
        curves: Vec::new(),
        }
    }

    /// 穷举极端尺寸组合：`new` 不 panic、不 abort，且尺寸原样回显。
    #[test]
    fn framebuffer_new_extreme_dims_never_panics() {
        let dims: [(u32, u32); 14] = [
            (0, 0),
            (1, 1),
            (0, u32::MAX),
            (u32::MAX, 0),
            (1, u32::MAX),
            (u32::MAX, 1),
            (65535, 65535),
            (65536, 65536),
            (1 << 20, 1 << 20),
            (1 << 31, 1 << 31),
            (u32::MAX, u32::MAX),
            (u32::MAX - 1, u32::MAX - 1),
            (MAX_FRAMEBUFFER_DIM, 1),
            (1, MAX_FRAMEBUFFER_DIM),
        ];
        for (w, h) in dims {
            let fb = Framebuffer::new(w, h);
            assert_eq!(fb.width, w, "width 应原样回显");
            assert_eq!(fb.height, h, "height 应原样回显");
        }
    }

    /// `catch_unwind` 证据：`u32::MAX * u32::MAX` 曾溢出 panic（P1 缺陷）。
    #[test]
    fn framebuffer_new_extreme_dims_via_catch_unwind() {
        for (w, h) in [(u32::MAX, u32::MAX), (1 << 20, 1 << 20), (65536, 65536)] {
            let r = std::panic::catch_unwind(move || Framebuffer::new(w, h));
            assert!(r.is_ok(), "Framebuffer::new({w}, {h}) panic 了 —— P1 未修复");
        }
    }

    /// 选定语义断言：超限 → 拒绝（空 pixels + 原样尺寸 + is_valid()==false）。
    #[test]
    fn framebuffer_over_limit_is_refused_not_clamped() {
        // 字节超限：65536*65536*4 = 16 GiB > MAX_FRAMEBUFFER_BYTES(1 GiB)。
        let fb = Framebuffer::new(65_536, 65_536);
        assert_eq!(
            (fb.width, fb.height),
            (65_536, 65_536),
            "尺寸不回写为钳制值"
        );
        assert!(
            fb.pixels.is_empty(),
            "字节超限应拒绝分配（否则 16 GiB 巨额分配）"
        );
        assert!(!fb.is_valid(), "被拒绝的缓冲须 is_valid()==false");

        // 维度超限：单轴 > MAX_FRAMEBUFFER_DIM。
        let fb2 = Framebuffer::new(MAX_FRAMEBUFFER_DIM + 1, 1);
        assert!(fb2.pixels.is_empty(), "维度超限应拒绝分配");

        // 乘法溢出路径：checked_mul 返回 None。
        let fb3 = Framebuffer::new(u32::MAX, u32::MAX);
        assert!(fb3.pixels.is_empty(), "溢出尺寸应走 checked_mul 拒绝路径");

        // 对照：未超限则有效。
        let ok = Framebuffer::new(64, 64);
        assert!(ok.is_valid(), "64x64 应是有效缓冲");
        assert_eq!(Framebuffer::empty().pixels.len(), 0);
        assert!(!Framebuffer::empty().is_valid());
    }

    /// 不变量：`pixels.len() == width * height * 4` 对**所有非超限**输入成立。
    #[test]
    fn framebuffer_len_invariant_for_all_accepted_sizes() {
        let dims: [(u32, u32); 11] = [
            (1, 1),
            (2, 3),
            (7, 5),
            (16, 16),
            (64, 1),
            (1, 64),
            (255, 255),
            (1920, 1080),
            (0, 0),
            (0, 12345),
            (MAX_FRAMEBUFFER_DIM, 0),
        ];
        for (w, h) in dims {
            let fb = Framebuffer::new(w, h);
            let expect = (w as usize)
                .checked_mul(h as usize)
                .and_then(|px| px.checked_mul(BYTES_PER_PIXEL))
                .expect("测试用例本身不应溢出");
            assert!(
                expect <= MAX_FRAMEBUFFER_BYTES
                    && w <= MAX_FRAMEBUFFER_DIM
                    && h <= MAX_FRAMEBUFFER_DIM,
                "用例 ({w},{h}) 应属于非超限集合"
            );
            assert_eq!(
                fb.pixels.len(),
                expect,
                "非超限尺寸 ({w},{h}) 的长度不变量须严格成立"
            );
        }
    }

    /// 边界线精确性：恰好等于上限 ⇒ 接受；上限 +1 ⇒ 拒绝。
    #[test]
    fn framebuffer_limit_boundary_is_exact() {
        // 单轴恰好等于 MAX_FRAMEBUFFER_DIM、另一轴为 0 ⇒ 0 字节，接受。
        let at = Framebuffer::new(MAX_FRAMEBUFFER_DIM, 0);
        assert!(at.pixels.is_empty() && at.width == MAX_FRAMEBUFFER_DIM);
        // 超一 ⇒ 拒绝（尺寸仍原样回显）。
        let over = Framebuffer::new(MAX_FRAMEBUFFER_DIM + 1, 0);
        assert_eq!(over.width, MAX_FRAMEBUFFER_DIM + 1, "超限不回写尺寸");
        assert!(over.pixels.is_empty());
    }

    /// 被拒绝的缓冲必须完全惰性：越界写不得 panic，也不得凭空长出像素。
    #[test]
    fn refused_framebuffer_is_inert_under_writes() {
        let mut fb = Framebuffer::new(u32::MAX, u32::MAX);
        assert!(fb.pixels.is_empty());
        for (x, y) in [(0, 0), (u32::MAX, u32::MAX), (1 << 31, 1 << 31)] {
            fb.set_pixel(x, y, Vec3::splat(1.0));
            assert!(fb.get_pixel(x, y).is_none(), "被拒绝缓冲不得接受写入");
        }
        assert!(fb.pixels.is_empty(), "越界写不应改变缓冲长度");
    }

    /// `set_pixel` / `get_pixel` 在极端坐标下不 panic，且不写越界。
    #[test]
    fn pixel_access_extreme_coords_never_panics() {
        let mut fb = Framebuffer::new(4, 4);
        let coords: [(u32, u32); 10] = [
            (0, 0),
            (3, 3),
            (4, 4),
            (u32::MAX, u32::MAX),
            (u32::MAX, 0),
            (0, u32::MAX),
            (1 << 31, 1 << 31),
            (u32::MAX - 1, u32::MAX - 1),
            (1, 1 << 31),
            (1 << 31, 1),
        ];
        for (x, y) in coords {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fb.set_pixel(x, y, Vec3::splat(1.0));
                fb.get_pixel(x, y)
            }));
            assert!(r.is_ok(), "set/get_pixel({x},{y}) panic 了");
            // 合法坐标仍应写透；越界坐标须返回 None。
            if x < 4 && y < 4 {
                assert_eq!(r.unwrap(), Some([255, 255, 255, 255]));
            } else {
                assert_eq!(r.unwrap(), None, "越界坐标须返回 None");
            }
        }
        assert_eq!(fb.pixels.len(), 4 * 4 * 4, "越界写不应改变缓冲长度");
    }

    /// `resize` 复用 `new` 的拒绝语义，超限不 panic。
    #[test]
    fn framebuffer_resize_over_limit_does_not_panic() {
        let mut fb = Framebuffer::new(4, 4);
        fb.resize(8, 8);
        assert_eq!(fb.pixels.len(), 8 * 8 * 4);
        assert!(fb.is_valid());

        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fb.resize(u32::MAX, u32::MAX);
        }));
        assert!(r.is_ok(), "resize 到溢出尺寸 panic 了");
        assert_eq!((fb.width, fb.height), (u32::MAX, u32::MAX));
        assert!(
            fb.pixels.is_empty() && !fb.is_valid(),
            "resize 超限应拒绝"
        );
    }

    /// 后端能力查询：Vulkan 的可用性 = **compute 自检 ∧ 渲染自检**（双自检），
    /// 且 `render_scene` 的有效性与渲染自检严格一致。
    ///
    /// 语义（V2.1 T-A1 起，Q4 裁定）：`is_available()` 只有在「mul2 探针闭环
    /// 跑通」**且**「`spv_scene` 真渲染一帧 4×4 并读回自洽」时才为 `true`。
    /// 无论真假，`render_scene` 都不得 panic；自检没通过时返回无效缓冲
    /// （调用方回退 CPU），自检通过时必须给出 `w*h*4` 的有效帧。
    #[test]
    fn vulkan_backend_capability_is_verified_and_render_follows_self_test() {
        use crate::GpuDevice;
        let mut be = crate::vulkan::VulkanBackend::new();
        assert_eq!(
            be.is_available(),
            be.compute_available() && be.render_available(),
            "is_available 必须严格等同 compute + render 双自检"
        );
        assert_eq!(be.backend_name(), "vulkan-compute");
        if !be.probed_device() {
            assert!(!be.is_available(), "无设备时须诚实报不可用");
            assert!(be.device_name().is_none());
        }

        let req = tiny_request();
        let verified = be.render_available();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.render_scene(&req).pixels.len()
        }));
        let len = r.expect("render_scene 不得 panic");
        if verified {
            assert_eq!(
                len,
                (req.width * req.height) as usize * 4,
                "渲染自检通过 ⇒ 必须出 w*h*4 字节的真帧"
            );
        } else {
            assert_eq!(len, 0, "渲染路径不可用 ⇒ 返回空缓冲（不冒充成功）");
        }

        // CpuRefBackend 恒可用。
        let cpu = CpuRefBackend;
        assert!(cpu.is_available());
        assert_eq!(cpu.backend_name(), "cpu-ref");
    }

    /// 被拒绝尺寸下的 `CpuRefBackend::render_scene` 不 panic，且不产出错图。
    #[test]
    fn cpu_ref_backend_huge_request_does_not_panic() {
        let mut req = tiny_request();
        req.width = u32::MAX;
        req.height = u32::MAX;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut be = CpuRefBackend;
            be.render_scene(&req)
        }));
        let fb = r.expect("超限渲染请求 panic 了");
        assert!(fb.pixels.is_empty(), "超限请求应得到拒绝缓冲");
        assert!(!fb.is_valid());
    }
}