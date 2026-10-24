//! 背景帧的 GPU 出图路径：compute 离屏 → storage image 读回 RGBA8。
//!
//! # 为什么是 compute 而不是图形管线
//!
//! 背景算子是**逐点函数**（§6 命题 P1：支撑半径 `r = 0`），没有顶点/光栅
//! 语义可依赖；本沙箱又**无显示器**，swapchain/窗口表面不可用。用 compute
//! 直写 storage buffer 再读回，是唯一同时满足「零窗口」「零第三方 crate」
//! 「与 CPU 参考逐位可对拍」的路径。图形管线（render pass + framebuffer +
//! descriptor image）会引入格式转换与片元插值——两处都能把末位改掉，却对
//! 一个逐点算子毫无收益。
//!
//! # 契约
//!
//! * 输入：`(w, h, t, seed)`，与 [`crate::background::render_background_cpu`] 同参；
//! * 输出：行主序 RGBA8，`len == w*h*4`，alpha 恒 255（与 CPU 参考同布局）；
//! * 尺寸上限沿用 [`crate::MAX_FRAMEBUFFER_BYTES`]（1 GiB）与
//!   [`crate::MAX_FRAMEBUFFER_DIM`]；越限**不 panic**，如实报 `BufferFailed`；
//! * 无设备 / 无 ICD / wasm：`DeviceUnavailable` / `UnsupportedPlatform`，
//!   调用方（[`crate::background::render_background`]）据此回退 CPU 参考。

use super::compute::DispatchStatus;
#[cfg(not(target_arch = "wasm32"))]
use super::compute::{self as compute_mod, ByteRunReport};
use super::spirv_bg::{ENTRY_BACKGROUND, MODULE_BACKGROUND, WORKGROUP_BACKGROUND};
#[cfg(not(target_arch = "wasm32"))]
use super::VulkanContext;
use crate::compute::{BindingAccess, BufferSpec, ElementType, KernelDesc, WorkgroupSize};
use crate::{MAX_FRAMEBUFFER_BYTES, MAX_FRAMEBUFFER_DIM};

/// 背景帧 kernel 的**引擎侧**名字（与 SPIR-V 模块名区分：
/// `spv_background` 是二进制标识，`bg_frame` 是这个 dispatch 的描述标识）。
pub const KERNEL_BACKGROUND_FRAME: &str = "bg_frame";

/// `t` 的契约上界（秒）。
///
/// 噪声格点坐标 `|vx*2.2 + t*0.05|` 必须落在 `u32` 可表示范围内：Rust 的
/// `f32 as u32` 对越界值**饱和**（定义良好），而 SPIR-V `OpConvertFToU` 对
/// 越界是 **UB**（不同驱动可给不同像素）。`8e10 s ≈ 2536 年`，远超任何媒体
/// 时间线，留 6% 余量。超契约的输入不上机，交回 CPU 参考——结果依然确定。
pub const MAX_ABS_TIME: f32 = 8.0e10;

/// 一次背景帧 GPU 渲染的结果（诊断可断言，不靠裸 bool）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundFrameReport {
    /// 结果状态（`Ok` 时 `pixels` 非空）。
    pub status: DispatchStatus,
    /// 请求宽度（原样回显）。
    pub width: u32,
    /// 请求高度（原样回显）。
    pub height: u32,
    /// 派发的工作组个数。
    pub groups: [u32; 3],
    /// 设备名（有设备上下文时非空；失败也带出，便于定位「在哪个设备上失败」）。
    pub device: String,
    /// 失败原因（成功时为空串）。
    pub detail: String,
    /// RGBA8 像素（行主序；`len == width * height * 4`；非 `Ok` 时为空）。
    pub pixels: Vec<u8>,
}

impl BackgroundFrameReport {
    /// 是否成功出图。
    pub fn is_ok(&self) -> bool {
        self.status == DispatchStatus::Ok
    }

    fn failed(status: DispatchStatus, w: u32, h: u32, detail: &str) -> Self {
        BackgroundFrameReport {
            status,
            width: w,
            height: h,
            groups: [0, 0, 0],
            device: String::new(),
            detail: detail.to_string(),
            pixels: Vec::new(),
        }
    }
}

/// 背景帧 kernel 的描述（**按帧尺寸构造**：输出绑定的元素数 = 像素数）。
///
/// 返回 `None` 表示尺寸非法/超限（零轴、维度超 [`MAX_FRAMEBUFFER_DIM`]、
/// 字节超 [`MAX_FRAMEBUFFER_BYTES`]）——**不上机、不 panic**。
pub fn background_kernel(w: u32, h: u32) -> Option<KernelDesc> {
    if w == 0 || h == 0 || w > MAX_FRAMEBUFFER_DIM || h > MAX_FRAMEBUFFER_DIM {
        return None;
    }
    let pixels = (w as u64).checked_mul(h as u64)?;
    let bytes = pixels.checked_mul(crate::BYTES_PER_PIXEL as u64)?;
    if bytes > MAX_FRAMEBUFFER_BYTES as u64 {
        return None;
    }
    Some(KernelDesc::new(
        KERNEL_BACKGROUND_FRAME,
        ENTRY_BACKGROUND,
        WorkgroupSize::new(
            WORKGROUP_BACKGROUND[0],
            WORKGROUP_BACKGROUND[1],
            WORKGROUP_BACKGROUND[2],
        ),
        vec![
            // b0: Params { w, h, t_bits, seed }（4 × u32 = 16 字节）
            BufferSpec::new(0, ElementType::U32, 4, BindingAccess::ReadOnly),
            // b1: 每像素一个 little-endian 打包 RGBA8（u32）
            BufferSpec::new(1, ElementType::U32, pixels as u32, BindingAccess::WriteOnly),
        ],
        MODULE_BACKGROUND,
        "background.wgsl",
    ))
}

/// 参数块：`[w, h, t.to_bits(), seed as u32]`，little-endian 16 字节。
///
/// `t` 走**位型**而非十进制文本：shader 用 `OpBitcast` 还原，两端拿到的是
/// 同一个 f32 值（无文本往返、无第二套舍入口径）。
pub fn background_params(w: u32, h: u32, t: f32, seed: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&w.to_le_bytes());
    out[4..8].copy_from_slice(&h.to_le_bytes());
    out[8..12].copy_from_slice(&t.to_bits().to_le_bytes());
    out[12..16].copy_from_slice(&(seed as u32).to_le_bytes());
    out
}

/// 把字节报告转成背景帧报告（`pixels` 只在 `Ok` 且长度恰好为 `w*h*4` 时保留）。
#[cfg(not(target_arch = "wasm32"))]
fn frame_from_bytes(w: u32, h: u32, run: ByteRunReport) -> BackgroundFrameReport {
    let expected = (w as usize) * (h as usize) * crate::BYTES_PER_PIXEL;
    if run.status == DispatchStatus::Ok && run.output.len() != expected {
        // 长度不符 = 驱动/绑定异常：宁可报失败，也不把半截缓冲当帧。
        return BackgroundFrameReport {
            status: DispatchStatus::ReadbackFailed,
            width: w,
            height: h,
            groups: run.groups,
            device: run.device,
            detail: format!(
                "读回长度 {} != 期望 {expected}（w={w} h={h}）",
                run.output.len()
            ),
            pixels: Vec::new(),
        };
    }
    BackgroundFrameReport {
        status: run.status,
        width: w,
        height: h,
        groups: run.groups,
        device: run.device,
        detail: run.detail,
        pixels: run.output,
    }
}

/// **真机路径**：在已建立的上下文上出背景帧。
///
/// 无上下文时如实返回 `DeviceUnavailable`（不 panic、不返回假帧）。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn dispatch(
    ctx: Option<&VulkanContext>,
    w: u32,
    h: u32,
    t: f32,
    seed: u64,
) -> BackgroundFrameReport {
    if !t.is_finite() || t.abs() > MAX_ABS_TIME {
        return BackgroundFrameReport::failed(
            DispatchStatus::InvalidArguments,
            w,
            h,
            "t 非有限或超出格点契约（|t| ≤ 8e10 s）：GPU 侧 OpConvertFToU 越界属 UB",
        );
    }
    let Some(kernel) = background_kernel(w, h) else {
        return BackgroundFrameReport::failed(
            DispatchStatus::BufferFailed,
            w,
            h,
            "尺寸非法或超限（零轴 / 维度 > MAX_FRAMEBUFFER_DIM / 字节 > 1 GiB）",
        );
    };
    let params = background_params(w, h, t, seed);
    let run = compute_mod::dispatch_bytes(ctx, &kernel, [w, h, 1], &params);
    frame_from_bytes(w, h, run)
}

/// wasm32：无原生 Vulkan，如实报不支持（不 panic）。
#[cfg(target_arch = "wasm32")]
pub(crate) fn dispatch(
    _ctx: (),
    w: u32,
    h: u32,
    _t: f32,
    _seed: u64,
) -> BackgroundFrameReport {
    // wasm 下 `_t` 仍参与契约检查（参数校验与平台无关）。
    if background_kernel(w, h).is_none() {
        return BackgroundFrameReport::failed(
            DispatchStatus::BufferFailed,
            w,
            h,
            "尺寸非法或超限（零轴 / 维度 > MAX_FRAMEBUFFER_DIM / 字节 > 1 GiB）",
        );
    }
    if !_t.is_finite() || _t.abs() > MAX_ABS_TIME {
        return BackgroundFrameReport::failed(
            DispatchStatus::InvalidArguments,
            w,
            h,
            "t 非有限或超出格点契约（|t| ≤ 8e10 s）",
        );
    }
    BackgroundFrameReport::failed(
        DispatchStatus::UnsupportedPlatform,
        w,
        h,
        "wasm32 无原生 Vulkan（背景帧走 WebGPU 后端）",
    )
}

// ---------------------------------------------------------------------------
// 测试（无设备即可执行：kernel 描述 / 参数编码 / 尺寸守卫）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::kernel_spv_status;

    /// kernel 描述合法，且 SPIR-V 是真的嵌入的（不是待工具链的空串）。
    #[test]
    fn background_kernel_desc_is_valid_and_has_embedded_spv() {
        let k = background_kernel(64, 48).expect("64×48 合法");
        assert_eq!(k.name, KERNEL_BACKGROUND_FRAME);
        assert_eq!(k.entry, ENTRY_BACKGROUND);
        assert_eq!((k.workgroup.x, k.workgroup.y, k.workgroup.z), (8, 8, 1));
        k.validate().expect("kernel 描述须通过布局校验");
        assert_eq!(
            kernel_spv_status(&k),
            crate::compute::SpvStatus::Embedded,
            "背景 kernel 的 SPIR-V 必须已嵌入（发射器产物）"
        );
        assert_eq!(k.spv_module, MODULE_BACKGROUND);
        // 布局：b0 = 16 字节参数，b1 = w*h*4 字节像素。
        let layout = k.layout();
        assert_eq!(layout.len(), 2);
        assert_eq!(layout[0].bytes, 16);
        assert_eq!(layout[1].bytes, 64 * 48 * 4);
        assert_eq!(layout[1].offset % crate::compute::BINDING_ALIGNMENT, 0);
    }

    /// 尺寸守卫：零轴 / 维度超限 / 字节超限一律 `None`，且不 panic。
    #[test]
    fn kernel_size_guards_refuse_without_panicking() {
        assert!(background_kernel(0, 10).is_none());
        assert!(background_kernel(10, 0).is_none());
        assert!(background_kernel(MAX_FRAMEBUFFER_DIM + 1, 1).is_none());
        // 65536×65536×4 = 16 GiB > 1 GiB。
        assert!(background_kernel(65_536, 65_536).is_none());
        assert!(background_kernel(u32::MAX, u32::MAX).is_none());
        // 边界内可用（1920×1080 = 8.3 MiB）。
        assert!(background_kernel(1920, 1080).is_some());
        assert!(background_kernel(1, 1).is_some());
    }

    /// `t` 契约守卫**先于**设备访问：非有限/超界的 `t` 在任何 FFI 之前被拒。
    ///
    /// 这一条在无设备环境下同样可断言（`ctx = None` 也走同一守卫），因此
    /// 「为什么没有上机」不依赖环境。
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn non_finite_time_is_refused_before_device_access() {
        for t in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0e12, -1.0e12] {
            let r = dispatch(None, 8, 8, t, 0);
            assert_eq!(
                r.status,
                DispatchStatus::InvalidArguments,
                "t={t} 应被契约守卫拒绝，实际 {}（{}）",
                r.status.as_str(),
                r.detail
            );
            assert!(r.pixels.is_empty(), "被拒时不得产出像素");
            assert_eq!((r.width, r.height), (8, 8), "尺寸原样回显");
        }
        // 契约内的 t 不在守卫处被拦：无 ctx ⇒ 走到设备分支如实报不可用。
        let ok = dispatch(None, 8, 8, 0.0, 0);
        assert_eq!(ok.status, DispatchStatus::DeviceUnavailable);
        // 上界本身（8e10）仍在契约内。
        let edge = dispatch(None, 8, 8, MAX_ABS_TIME, 0);
        assert_eq!(edge.status, DispatchStatus::DeviceUnavailable);
    }

    /// 参数块：四字段 little-endian，`t` 走位型（bitcast 语义）。
    #[test]
    fn params_are_little_endian_and_bit_exact() {
        let p = background_params(1920, 1080, 1.25, 0x9E37_79B1);
        assert_eq!(u32::from_le_bytes([p[0], p[1], p[2], p[3]]), 1920);
        assert_eq!(u32::from_le_bytes([p[4], p[5], p[6], p[7]]), 1080);
        assert_eq!(
            u32::from_le_bytes([p[8], p[9], p[10], p[11]]),
            1.25f32.to_bits(),
            "t 必须以 f32 位型传输（无十进制往返）"
        );
        assert_eq!(
            u32::from_le_bytes([p[12], p[13], p[14], p[15]]),
            0x9E37_79B1_u64 as u32,
            "seed 取低 32 位（与 CPU 参考同一截断口径）"
        );
        // 高位 seed 不影响参数块（CPU 参考也只取 as u32）。
        let q = background_params(1, 1, 0.0, 0xFFFF_FFFF_0000_0001);
        assert_eq!(&q[12..16], &1u32.to_le_bytes());
    }
}
