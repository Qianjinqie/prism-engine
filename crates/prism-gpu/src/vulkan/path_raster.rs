//! 2D 路径光栅帧的 GPU 出图路径：compute 离屏 → readback f32 覆盖率。
//!
//! # 契约
//!
//! * 输入：[`RasterInput`]（`prism_gpu::path_raster::prepare` 的产物，CPU/GPU 同一份
//!   字节）；
//! * 输出：行主序 f32 覆盖率，`len == w*h`（与 `CoverageMask` 等价）；
//! * 四个 storage 绑定全部声明为 `array<u32>`：f32 走位型（`OpBitcast`），
//!   避免文本往返与 std430 混合类型布局的第二套口径；
//! * 无设备 / wasm / 超限尺寸：`DeviceUnavailable` / `UnsupportedPlatform` /
//!   `BufferFailed`，调用方（[`crate::path_raster::rasterize`]）据此回退 CPU。
//!
//! # 与背景帧路径的同构性
//!
//! `render_background_frame` 的九步装配（shader module → descriptor → pipeline →
//! command buffer → dispatch → fence → map）由 `compute::dispatch_inputs_with_words`
//! 统一承载，本文件只提供 kernel 描述与字节解释——**不复制第二套 Vulkan 代码**。

use super::compute::DispatchStatus;
#[cfg(not(target_arch = "wasm32"))]
use super::compute::{self as compute_mod, ByteRunReport};
use super::spirv_path::{ENTRY_PATH_RASTER, MODULE_PATH_RASTER, WORKGROUP_PATH_RASTER};
#[cfg(not(target_arch = "wasm32"))]
use super::VulkanContext;
use crate::compute::{BindingAccess, BufferSpec, ElementType, KernelDesc, WorkgroupSize};
use crate::path_raster::{CoverageMask, RasterInput, MAX_GPU_SEGMENT_BYTES};
use crate::{MAX_FRAMEBUFFER_BYTES, MAX_FRAMEBUFFER_DIM};

/// 路径光栅 kernel 的引擎侧名字。
pub const KERNEL_PATH_RASTER_FRAME: &str = "path_raster_frame";

/// 一次路径光栅 GPU 派发的结构化报告（负控要断言的「真跑过 GPU」证据都在此）。
#[derive(Debug, Clone, PartialEq)]
pub struct PathFrameReport {
    /// 结果状态。
    pub status: DispatchStatus,
    /// kernel 名。
    pub kernel: &'static str,
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 派发的工作组个数。
    pub groups: [u32; 3],
    /// 调用总数（含尾组浪费）。
    pub invocations: u64,
    /// 片段数。
    pub segments: usize,
    /// 设备名（有上下文时非空；成功与失败都带出）。
    pub device: String,
    /// 失败原因（成功时为空）。
    pub detail: String,
    /// f32 覆盖率（行主序，`Ok` 时 `len == width*height`）。
    pub coverage: CoverageMask,
}

impl PathFrameReport {
    /// 是否成功出图。
    pub fn is_ok(&self) -> bool {
        self.status == DispatchStatus::Ok
    }

    fn failed(
        status: DispatchStatus,
        w: u32,
        h: u32,
        detail: &str,
        segments: usize,
        device: String,
    ) -> Self {
        PathFrameReport {
            status,
            kernel: KERNEL_PATH_RASTER_FRAME,
            width: w,
            height: h,
            groups: [0, 0, 0],
            invocations: 0,
            segments,
            device,
            detail: detail.to_string(),
            coverage: CoverageMask::new(0, 0),
        }
    }
}

/// 按输入构造 kernel 描述（**每次派发按片段数/尺寸现算**）。
///
/// 返回 `None` 表示尺寸非法或超限——不上机、不 panic。
pub fn path_kernel(input: &RasterInput) -> Option<KernelDesc> {
    let w = input.width;
    let h = input.height;
    if w == 0 || h == 0 || w > MAX_FRAMEBUFFER_DIM || h > MAX_FRAMEBUFFER_DIM {
        return None;
    }
    let pixels = (w as u64).checked_mul(h as u64)?;
    let out_bytes = pixels.checked_mul(4)?;
    if out_bytes > MAX_FRAMEBUFFER_BYTES as u64 {
        return None;
    }
    if input.segments.len() * 4 > MAX_GPU_SEGMENT_BYTES {
        return None;
    }
    let seg_words = input.segments.len().max(1) as u32;
    let row_words = input.rows.len().max(1) as u32;
    Some(KernelDesc::new(
        KERNEL_PATH_RASTER_FRAME,
        ENTRY_PATH_RASTER,
        WorkgroupSize::new(
            WORKGROUP_PATH_RASTER[0],
            WORKGROUP_PATH_RASTER[1],
            WORKGROUP_PATH_RASTER[2],
        ),
        vec![
            // b0: Params [w, h, segments_f32_len, rows_u32_len]（4 × u32）
            BufferSpec::new(0, ElementType::U32, 4, BindingAccess::ReadOnly),
            // b1: 片段（每 f32 一个 u32 位型槽，6 个一组）
            BufferSpec::new(1, ElementType::U32, seg_words, BindingAccess::ReadOnly),
            // b2: 每行 [start, count]（2*h 个 u32）
            BufferSpec::new(2, ElementType::U32, row_words, BindingAccess::ReadOnly),
            // b3: 输出覆盖率（每像素一个 f32 位型 u32）
            BufferSpec::new(3, ElementType::U32, pixels as u32, BindingAccess::WriteOnly),
        ],
        MODULE_PATH_RASTER,
        "path_raster.wgsl",
    ))
}

/// 字节报告 → 帧报告：把 u32 位型读回解释成 f32 覆盖率。
#[cfg(not(target_arch = "wasm32"))]
fn frame_from_bytes(input: &RasterInput, run: ByteRunReport) -> PathFrameReport {
    let w = input.width;
    let h = input.height;
    let expected = (w as usize) * (h as usize) * 4;
    let segments = input.segment_count();
    if run.status != DispatchStatus::Ok {
        return PathFrameReport::failed(run.status, w, h, &run.detail, segments, run.device);
    }
    if run.output.len() != expected {
        return PathFrameReport::failed(
            DispatchStatus::ReadbackFailed,
            w,
            h,
            &format!("读回长度 {} != 期望 {expected}（w={w} h={h}）", run.output.len()),
            segments,
            run.device,
        );
    }
    let mut data: Vec<f32> = Vec::with_capacity(w as usize * h as usize);
    for c in run.output.chunks_exact(4) {
        data.push(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
    }
    PathFrameReport {
        status: DispatchStatus::Ok,
        kernel: KERNEL_PATH_RASTER_FRAME,
        width: w,
        height: h,
        groups: run.groups,
        invocations: run.invocations,
        segments,
        device: run.device,
        detail: run.detail,
        coverage: CoverageMask { width: w, height: h, data },
    }
}

/// **真机路径**：在已有/缺失设备上走同一条报告口径。
///
/// `words = None` 用注册表里的产品 SPIR-V；`Some` 用显式字流（负控注入扰动版）。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn dispatch(
    ctx: Option<&VulkanContext>,
    input: &RasterInput,
    words: Option<&[u32]>,
) -> PathFrameReport {
    let w = input.width;
    let h = input.height;
    let segments = input.segment_count();
    let Some(kernel) = path_kernel(input) else {
        return PathFrameReport::failed(
            DispatchStatus::BufferFailed,
            w,
            h,
            "尺寸非法/超限（零轴 / 维度 > MAX_FRAMEBUFFER_DIM / 字节 > 1 GiB / 片段 > 64 MiB）",
            segments,
            String::new(),
        );
    };
    let params = input.params_bytes();
    let segs = input.segments_bytes();
    let rows = input.rows_bytes();
    let run = compute_mod::dispatch_inputs_with_words(
        ctx,
        &kernel,
        [w, h, 1],
        &[&params, &segs, &rows],
        words,
    );
    frame_from_bytes(input, run)
}

/// wasm32：无原生 Vulkan，如实报不支持（WGSL kernel 由 WebGPU harness 消费）。
#[cfg(target_arch = "wasm32")]
pub fn dispatch(_input: &RasterInput, words: Option<&[u32]>) -> PathFrameReport {
    let _ = words;
    let segments = _input.segment_count();
    if path_kernel(_input).is_none() {
        return PathFrameReport::failed(
            DispatchStatus::BufferFailed,
            _input.width,
            _input.height,
            "尺寸非法或超限",
            segments,
            String::new(),
        );
    }
    PathFrameReport::failed(
        DispatchStatus::UnsupportedPlatform,
        _input.width,
        _input.height,
        "wasm32 无原生 Vulkan（2D 路径走 WebGPU kernel path_raster.wgsl）",
        segments,
        String::new(),
    )
}

// ---------------------------------------------------------------------------
// 测试（无设备即可执行）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::kernel_spv_status;
    use crate::path_raster::{prepare, PathFillRule, RasterPoint};

    fn sample_input() -> RasterInput {
        let sq = vec![
            RasterPoint::new(1.5, 1.5),
            RasterPoint::new(9.5, 1.5),
            RasterPoint::new(9.5, 9.5),
            RasterPoint::new(1.5, 9.5),
        ];
        prepare(&[sq], PathFillRule::Winding, 16, 12)
    }

    #[test]
    fn kernel_desc_has_four_bindings_and_embedded_spv() {
        let input = sample_input();
        let k = path_kernel(&input).expect("16×12 合法");
        assert_eq!(k.name, KERNEL_PATH_RASTER_FRAME);
        assert_eq!(k.entry, ENTRY_PATH_RASTER);
        assert_eq!((k.workgroup.x, k.workgroup.y, k.workgroup.z), (8, 8, 1));
        k.validate().expect("kernel 描述须通过布局校验");
        assert_eq!(kernel_spv_status(&k), crate::compute::SpvStatus::Embedded);
        assert_eq!(k.spv_module, MODULE_PATH_RASTER);
        assert_eq!(k.wgsl_source, "path_raster.wgsl");
        let layout = k.layout();
        assert_eq!(layout.len(), 4);
        assert_eq!(layout[0].bytes, 16);
        assert_eq!(layout[1].bytes, input.segments.len() as u32 * 4);
        assert_eq!(layout[2].bytes, 2 * 12 * 4);
        assert_eq!(layout[3].bytes, 16 * 12 * 4);
        for l in &layout {
            assert_eq!(l.offset % crate::compute::BINDING_ALIGNMENT, 0);
        }
    }

    #[test]
    fn size_guards_refuse_without_panicking() {
        for (w, h) in [(0u32, 10u32), (10, 0), (MAX_FRAMEBUFFER_DIM + 1, 1), (65_536, 65_536)] {
            let input = prepare(&[], PathFillRule::Winding, w, h);
            assert!(path_kernel(&input).is_none(), "({w},{h}) 应被拒");
        }
        assert!(path_kernel(&sample_input()).is_some());
    }

    #[test]
    fn params_are_little_endian_and_bit_exact() {
        let input = sample_input();
        let p = input.params_bytes();
        assert_eq!(u32::from_le_bytes([p[0], p[1], p[2], p[3]]), 16);
        assert_eq!(u32::from_le_bytes([p[4], p[5], p[6], p[7]]), 12);
        assert_eq!(u32::from_le_bytes([p[8], p[9], p[10], p[11]]), input.segments.len() as u32);
        assert_eq!(u32::from_le_bytes([p[12], p[13], p[14], p[15]]), input.rows.len() as u32);
        assert_eq!(input.segments_bytes().len(), input.segments.len() * 4);
        assert_eq!(input.rows_bytes().len(), input.rows.len() * 4);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn no_context_reports_device_unavailable_not_a_fake_frame() {
        let input = sample_input();
        let r = dispatch(None, &input, None);
        assert_eq!(r.status, DispatchStatus::DeviceUnavailable);
        assert!(r.coverage.data.is_empty(), "失败不得返回假帧");
        assert!(!r.detail.is_empty());
        assert_eq!(r.kernel, KERNEL_PATH_RASTER_FRAME);
    }
}
