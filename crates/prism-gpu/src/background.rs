//! 背景渲染：CPU 参考实现（GPU 管线兜底）。
//!
//! 本模块把 demo 的逐像素 `paint_background` 热点迁移为无 GPU 平台完整可运行的
//! CPU 后端，并作为 GPU 路径的数值判对基准。**三条实现严格同式**
//! （hash value noise + 手写 smoothstep + 手写正弦逼近）：
//!
//! * [`render_background_cpu`]：CPU 参考（本文件，权威定义）；
//! * `shaders/background.wgsl`：WGSL 源（WebGPU 侧，结构自检覆盖）；
//! * [`crate::vulkan::spirv_bg`]：Vulkan compute 的 SPIR-V（自研发射器，
//!   W4 起真机执行，与 CPU 逐字节对拍）。
//!
//! 接口：
//! - [`render_background_cpu`]：纯 CPU 实现，返回 RGBA8（`Vec<u8>`，行主序，len = w*h*4）。
//! - [`render_background_gpu`]：Vulkan compute 真机出帧（W4 接通），失败返回 `None`。
//! - [`render_background`]：引擎入口——**GPU 优先，失败回退 CPU 参考**（不 panic）。
//!
//! # GPU 路径（W4）
//!
//! SPIR-V 由 [`crate::vulkan::spirv_bg`] 的自研发射器生成，逐式对应本文件的
//! CPU 参考（同求值树 + `NoContraction` + 无 libm）。判据是 §6 命题 P1：
//! 逐点算子（`r = 0`）⟹ GPU 与 CPU 必须**逐字节相同**，不是「容差内接近」。
//! 无设备/无 ICD 的平台上 [`render_background_gpu`] 返回 `None`，引擎入口
//! 静默回退 CPU——这条降级路径与 GPU 路径一样有测试覆盖。

use std::f32::consts::PI;

/// 背景色顶端（垂直渐变）。与 demo `BG_TOP` 一致。
pub(crate) const BG_TOP: [f32; 3] = [0.035, 0.055, 0.130];
/// 背景色底端（垂直渐变）。与 demo `BG_BOTTOM` 一致。
pub(crate) const BG_BOTTOM: [f32; 3] = [0.058, 0.090, 0.205];
/// 辉光冷青调系数。与 demo 一致。
pub(crate) const GLOW: [f32; 3] = [0.030, 0.070, 0.120];
/// 2π。
pub(crate) const TAU: f32 = 2.0 * PI;
/// `1 / 2^32`，与 WGSL `hash2i` 的归一化系数一致（均按 f32 计算）。
pub(crate) const INV_U32: f32 = 1.0 / 4_294_967_296.0;

/// 确定性 u32 哈希（与 WGSL `hash_u32` 位级一致）。
#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// 整数格点 + seed → [0,1) 伪随机（与 WGSL `hash2i` 同公式）。
#[inline]
fn hash2i(ix: u32, iy: u32, seed: u32) -> f32 {
    let mut h: u32 = ix
        .wrapping_mul(0x9e37_79b1)
        .wrapping_add(iy.wrapping_mul(0x85eb_ca77))
        .wrapping_add(seed.wrapping_mul(0xc2b2_ae3d));
    h = hash_u32(h);
    (h as f32) * INV_U32
}

/// 2D value noise（与 WGSL `value_noise` 同公式），返回 [0,1)。
#[inline]
fn value_noise(px: f32, py: f32, seed: u32) -> f32 {
    let ix = px.floor() as u32;
    let iy = py.floor() as u32;
    let fx = px - px.floor();
    let fy = py - py.floor();
    // smoothstep 权重，手写（与 WGSL 一致）。
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let x0 = ix;
    let y0 = iy;
    // `wrapping_add`：`px` 为 +inf 时 `floor() as u32` 饱和到 `u32::MAX`，
    // 朴素 `+ 1` 会在 debug 下 panic（C18）、release 下静默回绕。GPU 侧
    // `OpIAdd` 本就是模 2^32 回绕，故这里显式回绕才是两端同式且不 panic 的写法。
    let x1 = ix.wrapping_add(1);
    let y1 = iy.wrapping_add(1);
    let v00 = hash2i(x0, y0, seed);
    let v10 = hash2i(x1, y0, seed);
    let v01 = hash2i(x0, y1, seed);
    let v11 = hash2i(x1, y1, seed);
    let a = v00 + (v10 - v00) * ux;
    let b = v01 + (v11 - v01) * ux;
    a + (b - a) * uy
}

/// demo `smooth`：clamp 到 [0,1] 再 t*t*(3-2t)。
#[inline]
fn smooth01(e: f32) -> f32 {
    let t = e.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 手写正弦逼近（Maclaurin 5 阶，与 WGSL `sin_approx` 同公式）。
#[inline]
fn sin_approx(x: f32) -> f32 {
    let mut a = x - TAU * (x / TAU).floor();
    if a > PI {
        a -= TAU;
    }
    if a < -PI {
        a += TAU;
    }
    let xx = a * a;
    let a3 = a * xx;
    let a5 = a3 * xx;
    let a7 = a5 * xx;
    let a9 = a7 * xx;
    a - a3 / 6.0 + a5 / 120.0 - a7 / 5040.0 + a9 / 362_880.0
}

/// 按像素坐标计算背景色（RGB，[0,1]）。与 WGSL `background_at` 同公式。
#[inline]
fn background_at(px: u32, py: u32, w: u32, h: u32, t: f32, seed: u32) -> [f32; 3] {
    let vx = px as f32 / (w as f32 - 1.0).max(1.0);
    let vy = py as f32 / (h as f32 - 1.0).max(1.0);

    // 1) 垂直渐变（lerp top/bottom，smoothstep）。
    let g = smooth01(vy);
    let mut col = [
        BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * g,
        BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * g,
        BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * g,
    ];

    // 2) 冷青辉光（二次衰减）。
    let gx = vx - 0.5;
    let gy = vy - 0.40;
    let d2 = gx * gx + gy * gy;
    let mut glow = 1.0 - d2 * 3.0;
    if glow < 0.0 {
        glow = 0.0;
    }
    let glow = glow * glow;
    col[0] += GLOW[0] * glow;
    col[1] += GLOW[1] * glow;
    col[2] += GLOW[2] * glow;

    // 3) 极低频噪声（hash value noise，振幅 ±3%，随时间极慢漂移）。
    let nx = vx * 2.2 + t * 0.05;
    let ny = vy * 1.3 - t * 0.035;
    let n = value_noise(nx, ny, seed) * 2.0 - 1.0; // [0,1) -> [-1,1)
    let modulation = 1.0 + 0.03 * n;

    // 4) 暗角。
    let cx = vx - 0.5;
    let cy = vy - 0.5;
    let rad = (cx * cx + cy * cy).sqrt();
    let vig = 1.0 - smooth01((rad - 0.45) / 0.55) * 0.55;

    // 5) 呼吸（正弦逼近，±1.5%）。
    let breath = 1.0 + 0.015 * sin_approx(t * (TAU / 5.0));

    let k = modulation * breath * vig;
    [col[0] * k, col[1] * k, col[2] * k]
}

/// CPU 参考实现：逐像素渲染背景为 RGBA8。
///
/// 返回行主序 `Vec<u8>`，每像素 4 字节 `[r, g, b, a]`，`len == w * h * 4`，
/// alpha 恒为 255。无 GPU 环境下这是当前可用后端（兜底）。
///
/// 公式与 `shaders/background.wgsl` 一致，供数值一致性对拍。
pub fn render_background_cpu(w: u32, h: u32, t: f32, seed: u64) -> Vec<u8> {
    // 尺寸守卫（与 `Framebuffer::new` 同一拒绝语义）：`u32::MAX × u32::MAX × 4`
    // 在 64 位上会溢出 `usize`，在 32 位上会尝试分配 16 EiB。真实缺陷：
    // 早期实现直接 `vec![0u8; w*h*4]`，debug 下 panic、release 下 abort，
    // 都违反 C18。超限一律返回空 vec（调用方按长度判断，不 panic）。
    let Some(len) = (w as usize)
        .checked_mul(h as usize)
        .and_then(|px| px.checked_mul(crate::BYTES_PER_PIXEL))
    else {
        return Vec::new();
    };
    if w > crate::MAX_FRAMEBUFFER_DIM
        || h > crate::MAX_FRAMEBUFFER_DIM
        || len > crate::MAX_FRAMEBUFFER_BYTES
    {
        return Vec::new();
    }
    let seed_u = seed as u32; // 高位折入低位，确定性；与 GPU uniform(u32) 同口径。
    let mut out = vec![0u8; len];
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = background_at(x, y, w, h, t, seed_u);
            let idx = ((y as usize) * (w as usize) + (x as usize)) * 4;
            out[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

/// GPU 背景出帧（Vulkan compute 离屏 → 字节读回）。
///
/// * 有可用设备（本沙箱 = lavapipe）⇒ 返回 `w*h*4` 字节的 RGBA8，与
///   [`render_background_cpu`] **逐字节相同**（由 `tests/vulkan_background_gpu.rs`
///   全像素对拍）；
/// * 无设备 / 无 ICD / wasm / 尺寸超限 / 参数超契约（`t` 非有限）⇒ `None`
///   （调用方回退 CPU，不 panic）。
///
/// 设备句柄按**线程**缓存（`thread_local`）：同一线程的连续帧复用同一个
/// `VkDevice`，不会每帧重建 instance/device。
///
/// 降级分两级，成本都可预测：
/// * **初始化级**：首次调用跑 `mul2` 探针（compute 自检）。无设备或自检失败
///   ⇒ 该线程永久标记为「无 GPU 路径」，后续调用直接 `None`，不再碰驱动；
/// * **单帧级**：某帧派发失败（如超大尺寸分配不到缓冲）只让这一帧回退
///   CPU，不永久关闭 GPU 路径——偶发失败不应把能力位钉死。
pub fn render_background_gpu(w: u32, h: u32, t: f32, seed: u64) -> Option<Vec<u8>> {
    if w == 0 || h == 0 {
        // 空图：CPU 参考返回空 vec，GPU 无需上机。
        return None;
    }
    GPU_SLOT.with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            // 重入（同一线程的嵌套调用）：保守回退 CPU，不 panic。
            return None;
        };
        if matches!(*slot, GpuSlot::Uninit) {
            let backend = crate::vulkan::VulkanBackend::new();
            // 双门：探测到设备 且 最小 dispatch 闭环自检通过。
            *slot = if backend.probed_device() && backend.compute_available() {
                GpuSlot::Ready(Box::new(backend))
            } else {
                GpuSlot::Unavailable
            };
        }
        match &mut *slot {
            GpuSlot::Ready(backend) => {
                let report = backend.render_background_frame(w, h, t, seed);
                if report.is_ok() {
                    Some(report.pixels)
                } else {
                    None
                }
            }
            _ => None,
        }
    })
}

/// 本线程的 GPU 设备槽（探测一次，结果缓存；失败后不再重试）。
///
/// `Uninit`/`Unavailable` 两态是必要的：把「还没试」与「试过，不行」分开，
/// 才能既惰性探测、又不让无设备平台每帧白跑一次 `VulkanBackend::new()`。
enum GpuSlot {
    Uninit,
    Unavailable,
    Ready(Box<crate::vulkan::VulkanBackend>),
}

thread_local! {
    static GPU_SLOT: std::cell::RefCell<GpuSlot> = std::cell::RefCell::new(GpuSlot::Uninit);
}

/// 引擎背景渲染入口：**GPU 优先，CPU 兜底**。
///
/// 有可用 GPU 路径时返回 GPU 帧（与 CPU 参考逐字节一致，见
/// [`render_background_gpu`]）；否则返回 [`render_background_cpu`]。
/// 两条路径同输入同输出，调用方无需区分——这正是「GPU 从能算到能出图」
/// 之后引擎入口应有的形态：**能力提升对上层透明，结果不变**。
///
/// 任何条件下都不 panic：尺寸超限时 CPU 参考同样返回拒绝缓冲（空 vec）。
pub fn render_background(w: u32, h: u32, t: f32, seed: u64) -> Vec<u8> {
    if let Some(pixels) = render_background_gpu(w, h, t, seed) {
        return pixels;
    }
    render_background_cpu(w, h, t, seed)
}
