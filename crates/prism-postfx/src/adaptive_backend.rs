//! **R-13 产品路径接线**：把 A5/C2 自适应采样接进 `GpuDevice` 后端面。
//!
//! # 为什么是本 crate 里的一层包装（而不是改 `prism-render`）
//!
//! 依赖方向是 `prism-postfx → prism-render`（`render_adaptive` 消费
//! `translate_scene` 的产物），Rust 不允许循环依赖，故 `prism-render::RtBackend`
//! 无法反向调用 [`render_adaptive`]。包装器复用 `prism-render` 的**公开**原语
//! （`RtBackend` / `translate_scene` / `build_camera` / `black_frame` / `RT_SEED`），
//! 自己只做「分配 + 归约 + aesthetic」三步——与均匀档共用同一份实现的地方
//! 全部走原函数，不存在第二份几何/相机/量化代码。
//!
//! # 语义（与均匀档的关系，逐条如实）
//!
//! * `render_adaptive` 的判据 V1（`crates/prism-postfx/tests/`）钉死：
//!   `adaptive[p] == render_pixel(p, spp = n_p)[p]` **逐位相等**——即本后端
//!   的每像素样本数与"以 `n_p` 均匀渲染该像素"完全一致，只是预算按
//!   `σ̂` 重新分配了；
//! * `EvalCtx::new(RT_SEED, 0, 0)` 与 [`RtBackend`] 同值（采样流三要素同源）；
//! * aesthetic 链与量化走 `prism_gpu::apply_aesthetic` + `Framebuffer::set_pixel`
//!   —— 与 `rt_tiles::render_tile` 同一实现（两档出片同一后处理）。
//!
//! # 拒绝服务语义（不 panic、不静默降级，宪章 C18）
//!
//! * 请求含曲线 ⇒ [`Framebuffer::empty`]（与 [`RtBackend`] 同判据同理由）；
//! * 相机退化 ⇒ 全黑有效帧（同 [`RtBackend`]）；
//! * **预算不足**（`npix × N0 > total`）⇒ [`Framebuffer::empty`]。CLI 侧
//!   （`prism-app render --adaptive`）在渲染前用 [`AdaptiveRtBackend::min_total`]
//!   预检并给出明确报错，正常不会走到这里；直接用库的调用方拿到的是
//!   拒绝缓冲（`is_valid() == false`），与"后端拒了这帧"的既有契约一致。

use prism_gpu::{Framebuffer, GpuDevice, SceneRenderRequest};
use prism_math::Vec3;
use prism_render::rt_backend::{black_frame, build_camera, translate_scene, RtBackend, RT_SEED};
use prism_core::EvalCtx;
use prism_rt::integrator::RenderParams;

use crate::adaptive::{render_adaptive, AdaptiveParams};

/// 自适应采样后端：`RtBackend` 的语义 + A5/C2 的预算分配。
///
/// `rt.spp` 在本后端**不参与**每像素样本数（样本数由 `σ̂` 分配决定）；
/// 它仍作为 `RenderParams.spp` 传入（供 `clamp_indirect` / `nee_only` 等
/// 与 spp 无关的字段保持同源），`max_bounces` 照常生效。
#[derive(Debug, Clone)]
pub struct AdaptiveRtBackend {
    /// 基础 RT 参数（max_bounces / 线程数；spp 字段语义见结构体文档）。
    pub rt: RtBackend,
    /// 自适应分配参数（N0 / tile / mantissa 桶位 / 精化轮数）。
    pub ap: AdaptiveParams,
}

impl AdaptiveRtBackend {
    /// 默认 RT 参数 + 指定总预算的自适应参数。
    pub fn new(total: u32) -> Self {
        AdaptiveRtBackend {
            rt: RtBackend::default(),
            ap: AdaptiveParams::new(total),
        }
    }

    /// 显式两段参数构造。
    pub fn with_params(rt: RtBackend, ap: AdaptiveParams) -> Self {
        AdaptiveRtBackend { rt, ap }
    }

    /// `npix` 像素一帧所需的最低总预算（`npix × N0`）。
    pub fn min_total(&self, npix: usize) -> u64 {
        (npix as u64) * (self.ap.n0 as u64)
    }
}

impl GpuDevice for AdaptiveRtBackend {
    fn render_scene(&mut self, req: &SceneRenderRequest) -> Framebuffer {
        // 与 RtBackend::render_scene 同口径的前置（尺寸 / 曲线 / 相机）。
        let fb: Framebuffer = Framebuffer::new(req.width, req.height);
        if !fb.is_valid() {
            return fb;
        }
        if !req.curves.is_empty() {
            return Framebuffer::empty();
        }
        let Some(cam) = build_camera(&req.camera) else {
            return black_frame(req.width, req.height);
        };
        let scene = translate_scene(req);
        scene.assert_layout();

        // 预算不足 ⇒ 拒绝服务（见模块文档；CLI 侧有 min_total 预检）。
        let npix = (req.width as usize) * (req.height as usize);
        if (self.min_total(npix)) > self.ap.total as u64 {
            return Framebuffer::empty();
        }

        let params: RenderParams = RenderParams::new(self.rt.spp, self.rt.max_bounces);
        // 采样流三要素与均匀档同源（RT_SEED / 0 / 0）。
        let ctx: EvalCtx = EvalCtx::new(RT_SEED, 0, 0);
        let img = render_adaptive(&scene, &cam, &ctx, (req.width, req.height), &params, &self.ap);

        let mut out: Framebuffer = Framebuffer::new(req.width, req.height);
        let w = req.width as usize;
        for y in 0..req.height {
            for x in 0..req.width {
                let p = img.pixels[(y as usize) * w + (x as usize)];
                let linear: Vec3 = Vec3::new(p[0], p[1], p[2]);
                // 与 rt_tiles::render_tile 同一后处理链（唯一实现在 prism-gpu）。
                let mapped: Vec3 = prism_gpu::apply_aesthetic(linear, &req.aesthetic);
                out.set_pixel(x, y, mapped);
            }
        }
        out
    }

    /// 纯软件路径追踪：无外部依赖 ⇒ 恒可用。
    fn is_available(&self) -> bool {
        true
    }

    fn backend_name(&self) -> &'static str {
        "rt-adaptive"
    }
}

// ---------------------------------------------------------------------------
// 单元测试（零 GPU 依赖；语义钉子）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gpu::{AestheticReq, CameraReq};

    /// 后端名与能力位（诊断与能力日志的契约）。
    #[test]
    fn backend_identity_is_stable() {
        let mut b = AdaptiveRtBackend::new(4096);
        assert_eq!(b.backend_name(), "rt-adaptive");
        assert!(b.is_available());
        assert_eq!(b.ap.n0, 8, "默认 N0 = 8（A5 判据档）");
        // min_total：线性于像素数。
        assert_eq!(b.min_total(100), 800);
        b.ap = b.ap.with_rounds(2);
        assert_eq!(b.min_total(100), 800, "轮数不影响最低预算");
    }

    /// 曲线请求 ⇒ 拒绝服务（与 RtBackend 同判据）。
    #[test]
    fn curve_requests_are_refused() {
        let mut b = AdaptiveRtBackend::new(1 << 20);
        let mut req = SceneRenderRequest::default();
        req.width = 8;
        req.height = 8;
        req.curves.push(prism_gpu::CurveReq::Cylinder {
            center: Vec3::splat(0.0),
            radius: 0.1,
            height: 1.0,
            axis: Vec3::new(0.0, 1.0, 0.0),
        });
        let fb = b.render_scene(&req);
        assert!(!fb.is_valid(), "曲线请求必须拿拒绝缓冲");
    }

    /// 预算不足 ⇒ 拒绝服务（不 panic；CLI 预检的正常兜底）。
    #[test]
    fn insufficient_budget_is_refused_not_panicked() {
        let mut b = AdaptiveRtBackend::new(64); // 64 < 8×8×8
        let req = SceneRenderRequest {
            width: 8,
            height: 8,
            ..Default::default()
        };
        let fb = b.render_scene(&req);
        assert!(!fb.is_valid(), "npix×N0 > total 必须拒绝，不得 panic");
    }

    /// 相机退化 ⇒ 全黑有效帧（与 RtBackend 同判据）。
    #[test]
    fn degenerate_camera_is_black_valid_frame() {
        let mut b = AdaptiveRtBackend::new(1 << 20);
        let req = SceneRenderRequest {
            width: 4,
            height: 4,
            camera: CameraReq {
                eye: Vec3::splat(0.0),
                target: Vec3::splat(0.0), // eye == target ⇒ 退化
                fov: 60.0,
                aspect: 1.0,
            },
            aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
            ..Default::default()
        };
        let fb = b.render_scene(&req);
        assert!(fb.is_valid());
        // set_pixel 把 alpha 置 255（Framebuffer 契约）；"黑"指 RGB 三通道全 0。
        assert!(
            fb.pixels.chunks_exact(4).all(|px| px[0] == 0 && px[1] == 0 && px[2] == 0),
            "退化相机 ⇒ RGB 全黑（alpha 恒 255）"
        );
    }

    /// **确定性**：同参数两跑逐字节相同（render_movie 的铁律在本后端成立）。
    #[test]
    fn two_passes_are_bitwise_identical() {
        let mut req = SceneRenderRequest::default();
        req.width = 12;
        req.height = 9;
        req.camera = CameraReq {
            eye: Vec3::new(0.0, 0.0, -4.0),
            target: Vec3::splat(0.0),
            fov: 55.0,
            aspect: 12.0 / 9.0,
        };
        req.aesthetic = AestheticReq { teal_orange: true, contrast: 1.1, filmic: true };
        // 一个真实可命中的球 + 方向光（预算给足：108 px × 8 = 864 ≤ 4096）。
        req.spheres.push(prism_gpu::SphereReq {
            center: Vec3::splat(0.0),
            radius: 1.0,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.0,
            metalness: 0.0,
            maps: prism_gpu::NO_MAPS,
        });
        let a = {
            let mut b = AdaptiveRtBackend::new(4096);
            b.render_scene(&req).pixels.clone()
        };
        let c = {
            let mut b = AdaptiveRtBackend::new(4096);
            b.render_scene(&req).pixels.clone()
        };
        assert_eq!(a.len(), (12 * 9 * 4) as usize);
        assert_eq!(a, c, "同参数两跑必须逐字节相同");
        assert!(a.iter().any(|&v| v != 0), "全零帧 = 场景构造失效的假绿指纹");
    }
}
