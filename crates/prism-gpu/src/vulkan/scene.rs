//! 光追场景帧的 GPU 出图路径：compute 离屏 → 字节读回 RGBA8。
//!
//! # 为什么光追也走 compute（而不是图形管线）
//!
//! `render_scene` 是一个**逐像素独立**的算子：每个像素的射线只依赖该像素的
//! NDC 与全局场景数据，像素之间没有任何归约依赖（支撑半径 `r = 0`）。这与
//! 背景帧（[`super::background`]）同一个理由：没有顶点/光栅语义可依赖，
//! 本沙箱又没有显示器（swapchain / render pass 不可用），而 compute 直写
//! storage buffer 再读回是唯一同时满足「零窗口」「零第三方 crate」
//! 「与 CPU 参考可对拍」的路径。
//!
//! # 契约
//!
//! * 输入：整个 [`SceneRenderRequest`]（相机 / 灯 / 球 / aesthetic）；
//! * 输出：行主序 RGBA8，`len == w*h*4`，alpha 恒 255（与
//!   [`crate::CpuRefBackend`] 同布局）；
//! * 尺寸上限沿用 [`crate::MAX_FRAMEBUFFER_BYTES`] / [`crate::MAX_FRAMEBUFFER_DIM`]；
//!   越限**不 panic**，如实报 `BufferFailed`；
//! * 无设备 / 无 ICD / wasm：`DeviceUnavailable` / `UnsupportedPlatform`，
//!   调用方（[`super::VulkanBackend::render_scene`] 的调用方）据此回退 CPU 参考。
//!
//! # 诚实边界
//!
//! 本模块只负责「把请求变成一次真实 dispatch 并读回」。**能不能跑**由
//! [`super::VulkanBackend::render_available`]（真跑一张 4×4 探针帧）单独回答；
//! 「探测到设备」不构成出图能力（G4 口径，同 compute / background 两条路径）。

use super::compute::{self as compute_mod, ByteRunReport, DispatchStatus};
use super::spirv_rt::scene_kernel;
#[cfg(not(target_arch = "wasm32"))]
use super::VulkanContext;
#[cfg(any(test, not(target_arch = "wasm32")))]
use crate::{AestheticReq, CameraReq, LightReq, SphereReq};
use crate::SceneRenderRequest;

/// 槽位个数（= 字节数 / 步长）；供 [`scene_kernel`] 推导缓冲容量。
fn tri_slot_count(bytes: &[u8]) -> usize {
    bytes.len() / (crate::scene_abi::TRIANGLE_SLOTS as usize * 4)
}

/// 节点槽位个数（同上）。
fn node_slot_count(bytes: &[u8]) -> usize {
    bytes.len() / (crate::scene_abi::NODE_SLOTS as usize * 4)
}

/// 场景 kernel 的**引擎侧**名字（与 SPIR-V 模块名 `spv_scene` 区分：
/// 后者是二进制标识，本名是这个 dispatch 的描述标识）。
pub const KERNEL_SCENE_RENDER: &str = "rt_scene_render";

/// 一次场景帧 GPU 渲染的结果（诊断可断言，不靠裸 bool）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneFrameReport {
    /// 结果状态（`Ok` 时 `pixels` 非空）。
    pub status: DispatchStatus,
    /// 请求宽度（原样回显）。
    pub width: u32,
    /// 请求高度（原样回显）。
    pub height: u32,
    /// 派发的工作组个数。
    pub groups: [u32; 3],
    /// 调用总数（含尾组浪费；诊断量）。
    pub invocations: u64,
    /// 设备名（有设备上下文时非空；失败也带出，便于定位「在哪个设备上失败」）。
    pub device: String,
    /// 失败原因（成功时为空串）。
    pub detail: String,
    /// RGBA8 像素（行主序；`len == width * height * 4`；非 `Ok` 时为空）。
    pub pixels: Vec<u8>,
}

impl SceneFrameReport {
    /// 是否成功出图。
    pub fn is_ok(&self) -> bool {
        self.status == DispatchStatus::Ok
    }

    fn failed(status: DispatchStatus, w: u32, h: u32, detail: &str) -> Self {
        SceneFrameReport {
            status,
            width: w,
            height: h,
            groups: [0, 0, 0],
            invocations: 0,
            device: String::new(),
            detail: detail.to_string(),
            pixels: Vec::new(),
        }
    }
}

/// 把字节报告转成场景帧报告（`pixels` 只在 `Ok` 且长度恰好为 `w*h*4` 时保留）。
fn frame_from_bytes(w: u32, h: u32, run: ByteRunReport) -> SceneFrameReport {
    let expected = (w as usize) * (h as usize) * crate::BYTES_PER_PIXEL;
    if run.status == DispatchStatus::Ok && run.output.len() != expected {
        // 长度不符 = 驱动/绑定异常：宁可报失败，也不把半截缓冲当帧。
        return SceneFrameReport {
            status: DispatchStatus::ReadbackFailed,
            width: w,
            height: h,
            groups: run.groups,
            invocations: run.invocations,
            device: run.device,
            detail: format!(
                "读回长度 {} != 期望 {expected}（w={w} h={h}）",
                run.output.len()
            ),
            pixels: Vec::new(),
        };
    }
    SceneFrameReport {
        status: run.status,
        width: w,
        height: h,
        groups: run.groups,
        invocations: run.invocations,
        device: run.device,
        detail: run.detail,
        pixels: run.output,
    }
}

/// **真机路径**：在已建立的上下文上出场景帧。
///
/// 无上下文时如实返回 `DeviceUnavailable`（不 panic、不返回假帧）。
///
/// 输入缓冲走 V2 剖面（6 绑定 → 5 块输入 + 输出）：
/// `params(96B) / spheres / lights / triangles / nodes`，编码权威在
/// [`crate::scene_abi::SceneAbi::encode_v2`]——与 CPU 参考同一份字节。
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn dispatch(ctx: Option<&VulkanContext>, req: &SceneRenderRequest) -> SceneFrameReport {
    let (params, spheres, lights, tris, nodes) = crate::scene_abi::SceneAbi::encode_v2(req);
    let Some(kernel) = scene_kernel(
        req.width,
        req.height,
        req.spheres.len(),
        req.lights.len(),
        tri_slot_count(&tris),
        node_slot_count(&nodes),
    ) else {
        return SceneFrameReport::failed(
            DispatchStatus::BufferFailed,
            req.width,
            req.height,
            "尺寸/实体数非法或超限（零轴 / 维度 > MAX_FRAMEBUFFER_DIM / 字节 > 1 GiB / 实体数超上限）",
        );
    };
    let run = compute_mod::dispatch_scene(
        ctx,
        &kernel,
        [req.width, req.height, 1],
        &[&params, &spheres, &lights, &tris, &nodes],
    );
    frame_from_bytes(req.width, req.height, run)
}

/// wasm32：无原生 Vulkan，如实报不支持（不 panic）。
///
/// 参数校验（尺寸/实体数）与**报告形状**都与真机路径一致：仍先构造 kernel
/// 描述（布局推导 + 容量守卫），再交给同一条 `dispatch_scene` 判定分支——
/// 「为什么没有上机」的答案与报告字段不随平台漂移。
#[cfg(target_arch = "wasm32")]
pub(crate) fn dispatch(_ctx: (), req: &SceneRenderRequest) -> SceneFrameReport {
    let (tris, nodes) = crate::scene_abi::encode_curve_buffers(&req.curves);
    let Some(kernel) = scene_kernel(
        req.width,
        req.height,
        req.spheres.len(),
        req.lights.len(),
        tri_slot_count(&tris),
        node_slot_count(&nodes),
    ) else {
        return SceneFrameReport::failed(
            DispatchStatus::BufferFailed,
            req.width,
            req.height,
            "尺寸/实体数非法或超限（零轴 / 维度 > MAX_FRAMEBUFFER_DIM / 字节 > 1 GiB / 实体数超上限）",
        );
    };
    let run = compute_mod::dispatch_scene((), &kernel, [req.width, req.height, 1], &[]);
    frame_from_bytes(req.width, req.height, run)
}

/// 可用性自检用的小请求：4×4、单球、单灯、关闭全部后期。
///
/// 尺寸刻意取极小（16 像素）：自检要「真跑一次」，但不能让可用性查询变成
/// 一次昂贵的渲染。单球保证**至少有一个像素命中**（不是全零帧）。
#[cfg(any(test, not(target_arch = "wasm32")))]
pub(crate) fn self_test_request() -> SceneRenderRequest {
    SceneRenderRequest {
        width: 4,
        height: 4,
        camera: CameraReq {
            eye: prism_math::Vec3::new(0.0, 0.0, -4.0),
            target: prism_math::Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: 1.0,
        },
        lights: vec![LightReq {
            direction: prism_math::Vec3::new(0.0, 0.0, -1.0),
            color: prism_math::Vec3::new(1.0, 1.0, 1.0),
            intensity: 1.0,
        }],
        spheres: vec![SphereReq {
            center: prism_math::Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            albedo: prism_math::Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.4,
            metalness: 0.1,
            maps: crate::NO_MAPS,
        }],
        aesthetic: AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
    curves: Vec::new(),
    ..Default::default()
}
}

// ---------------------------------------------------------------------------
// 测试（无设备即可执行：守卫口径 + 报告形状 + 编码链路）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 尺寸守卫**先于**设备访问：超限请求在任何环境都得到同一判决。
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn oversized_request_is_refused_before_device_access() {
        let mut req = self_test_request();
        req.width = u32::MAX;
        req.height = u32::MAX;
        let r = dispatch(None, &req);
        assert_eq!(r.status, DispatchStatus::BufferFailed);
        assert!(r.pixels.is_empty(), "被拒时不得产出像素");
        assert_eq!((r.width, r.height), (u32::MAX, u32::MAX), "尺寸原样回显");

        // 零轴同上。
        let mut zero = self_test_request();
        zero.width = 0;
        let z = dispatch(None, &zero);
        assert_eq!(z.status, DispatchStatus::BufferFailed);

        // 契约内的请求：无 ctx ⇒ 走到设备分支如实报不可用（不 panic、不假帧）。
        let ok = dispatch(None, &self_test_request());
        assert_eq!(ok.status, DispatchStatus::DeviceUnavailable);
        assert!(ok.pixels.is_empty());
        assert_eq!((ok.width, ok.height), (4, 4));
    }

    /// 自检请求本身合法（尺寸/实体数都过守卫），且带一个球一盏灯。
    #[test]
    fn self_test_request_is_valid() {
        let req = self_test_request();
        assert!(scene_kernel(req.width, req.height, req.spheres.len(), req.lights.len(), 0, 0)
            .is_some());
        assert_eq!(req.spheres.len(), 1);
        assert_eq!(req.lights.len(), 1);
        assert_eq!(req.width * req.height * 4, 64);
    }

    /// 实体数超限 ⇒ 同样在 FFI 之前被拒（与尺寸守卫同一口径）。
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn excessive_entity_counts_are_refused() {
        let mut req = self_test_request();
        req.spheres = (0..(super::super::spirv_rt::MAX_SCENE_SPHERES + 1))
            .map(|i| SphereReq {
                center: prism_math::Vec3::new(i as f32, 0.0, 0.0),
                radius: 0.1,
                albedo: prism_math::Vec3::new(1.0, 1.0, 1.0),
                roughness: 0.5,
                metalness: 0.0,
                maps: crate::NO_MAPS,
            })
            .collect();
        let r = dispatch(None, &req);
        assert_eq!(r.status, DispatchStatus::BufferFailed);
        assert!(r.detail.contains("超限"), "原因须可诊断：{}", r.detail);
    }
}
