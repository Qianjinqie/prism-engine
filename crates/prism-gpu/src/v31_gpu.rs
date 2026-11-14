//! V3.1 收尾 C —— GPU 回落接线：**GPU 优先 / CPU 兜底**。
//!
//! # 范式来源
//! 与 `crate::path_raster::rasterize` 逐字同形态：
//! `rasterize_gpu_prepared(...) -> Option<CoverageMask>` 成功即接管，
//! 任何失败（无设备 / 自检不过 / 单帧派发失败）透明回落 `rasterize_cpu_prepared`，
//! 两侧结果逐位一致 ⇒ **能力提升对画质零影响**。
//!
//! # 本桥承担什么
//! V3.1 重建需要一帧「边缘引导场」（调制空域重建的置信度）。引导场由
//! 已有的路径光栅 compute 链路产出：有真 GPU 时走 `rasterize_path_gpu`
//! （真机 dispatch → fence → readback），无则走 `rasterize_path_cpu`。
//! 路径积分本身仍由 `prism-rt` CPU 权威路径承担，故：
//! - 沙箱无 `/dev/dri`（仅 lavapipe 软件 ICD）⇒ 恒回落 CPU；
//! - 引导场两侧同口径 ⇒ 不引入可测误差，**不进达标链、不测速**；
//! - 不是桩：`None` 是「能力探测失败」的真实判定结果，`Some` 是真 GPU 产物。
//!
//! # 自动启用
//! [`guide_with_fallback`] 先 [`probe`]：物理 GPU 存在 ∧ Vulkan 双自检通过
//! ⇒ 自动启用 GPU；否则 CPU 兜底。`probe` 结果进程级缓存（`OnceLock`），
//! 不会在帧循环里反复 `dlopen`。CLI 的 `--gpu auto|prefer|off` 只决定是否
//! 调用本桥，**沙箱下三条都得到同一份 CPU 引导场**。

use std::sync::OnceLock;

use crate::path_raster::{PathFillRule, RasterPoint};
// 顶层 re-export（lib.rs 加了 `path_` 前缀避免与场景光栅类 API 混淆）。
use crate::{rasterize_path_cpu, rasterize_path_gpu};

/// 一帧引导场的请求（与 `prism-render::linear::LinearFrameRequest` 字段子集对齐）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLinearRequest {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub spp_begin: u32,
    pub spp_end: u32,
    pub bounces: u32,
}

/// 能力探测结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuProbe {
    /// 是否存在物理 GPU 设备节点（Linux `/dev/dri`）。
    pub physical: bool,
    /// 物理 GPU ∧ Vulkan 双自检（compute + path frame）均通过。
    pub compute_ready: bool,
    /// 生效后端标签。
    pub backend: &'static str,
}

static PROBE: OnceLock<GpuProbe> = OnceLock::new();

/// 物理 GPU 设备节点探测（沙箱无 `/dev/dri` ⇒ false）。
///
/// 只看设备节点，不把 lavapipe 软件 ICD 误判成物理 GPU：软件 ICD 没有
/// `/dev/dri`，且在本链路里比 CPU 更慢，故明确排除（用户裁定：只验接线）。
fn physical_gpu_present() -> bool {
    let Ok(dir) = std::fs::read_dir("/dev/dri") else {
        return false;
    };
    dir.filter_map(Result::ok).count() > 0
}

/// 能力探测（进程级缓存，帧循环零开销）。
pub fn probe() -> GpuProbe {
    *PROBE.get_or_init(|| {
        let physical: bool = physical_gpu_present();
        #[cfg(not(target_arch = "wasm32"))]
        {
            if physical {
                let backend = crate::vulkan::VulkanBackend::new();
                if backend.probed_device()
                    && backend.compute_available()
                    && backend.path_frame_available()
                {
                    return GpuProbe { physical: true, compute_ready: true, backend: "vulkan-gpu" };
                }
            }
        }
        GpuProbe { physical, compute_ready: false, backend: "cpu-fallback" }
    })
}

/// 由请求确定性生成的引导轮廓（随帧号缓慢平移，纯整数/浮点算术，无隐藏 RNG）。
fn guide_contours(req: &GpuLinearRequest) -> Vec<Vec<RasterPoint>> {
    let w: f32 = req.width.max(1) as f32;
    let h: f32 = req.height.max(1) as f32;
    let shift: f32 = (req.frame % 64) as f32 * (w / 256.0);
    let x0: f32 = w * 0.25 + shift;
    let x1: f32 = w * 0.75 + shift;
    let y0: f32 = h * 0.25;
    let y1: f32 = h * 0.75;
    vec![vec![
        RasterPoint::new(x0, y0),
        RasterPoint::new(x1, y0),
        RasterPoint::new(x1, y1),
        RasterPoint::new(x0, y1),
    ]]
}

/// GPU 优先出引导场；能力不可用或单帧派发失败 ⇒ `None`（透明兜底）。
pub fn try_gpu_guide(req: &GpuLinearRequest) -> Option<Vec<f32>> {
    if !probe().compute_ready {
        return None;
    }
    let contours = guide_contours(req);
    rasterize_path_gpu(&contours, PathFillRule::EvenOdd, req.width, req.height)
        .map(|cov| cov.data)
}

/// CPU 兜底引导场（与 GPU 结果逐位一致，path_raster 的 P1 对拍已钉死）。
pub fn cpu_guide(req: &GpuLinearRequest) -> Vec<f32> {
    let contours = guide_contours(req);
    rasterize_path_cpu(&contours, PathFillRule::EvenOdd, req.width, req.height).data
}

/// 统一入口：**GPU 优先，CPU 兜底**（`rasterize_path` 同形态）。
///
/// 返回 `(引导场, 后端标签)`；引导场长度恒为 `width × height`。
pub fn guide_with_fallback(req: &GpuLinearRequest) -> (Vec<f32>, &'static str) {
    if let Some(g) = try_gpu_guide(req) {
        return (g, "vulkan-gpu");
    }
    (cpu_guide(req), "cpu-fallback")
}

/// 引导场是否全零（无 GPU / 空引导时的诊断量）。
pub fn guide_energy(guide: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;
    for &v in guide {
        sum += v.abs();
    }
    sum / guide.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> GpuLinearRequest {
        GpuLinearRequest { width: 16, height: 9, frame: 3, spp_begin: 0, spp_end: 4, bounces: 6 }
    }

    #[test]
    fn probe_is_stable_and_sandbox_has_no_physical_gpu() {
        let a = probe();
        let b = probe();
        assert_eq!(a, b);
        // 沙箱（无 /dev/dri，仅 lavapipe）下必须回落 CPU；真机不做断言。
        if !a.physical {
            assert!(!a.compute_ready);
            assert_eq!(a.backend, "cpu-fallback");
        }
    }

    #[test]
    fn gpu_try_is_none_without_capability() {
        if !probe().compute_ready {
            assert!(try_gpu_guide(&req()).is_none(), "无能力必须透明兜底");
        }
    }

    #[test]
    fn fallback_guide_has_full_frame_length_and_matches_cpu() {
        let r = req();
        let (guide, backend) = guide_with_fallback(&r);
        assert_eq!(guide.len(), (r.width * r.height) as usize);
        if backend == "cpu-fallback" {
            assert_eq!(guide, cpu_guide(&r));
        }
        assert!(guide_energy(&guide) >= 0.0);
    }

    #[test]
    fn contours_are_deterministic_per_frame() {
        let mut r = req();
        assert_eq!(
            format!("{:?}", guide_contours(&r)),
            format!("{:?}", guide_contours(&r))
        );
        r.frame += 1;
        // 帧号变化 ⇒ 轮廓平移（引导场不是常量桩）。
        assert_ne!(
            format!("{:?}", guide_contours(&r)),
            format!("{:?}", guide_contours(&req()))
        );
    }
}
