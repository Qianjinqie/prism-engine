//! V3.1 T03 —— f32 线性帧旁路：把 tone-map 延后到时域累积之后。
//!
//! # 为什么必须线性（设计 §1.1 / 裁定 3）
//! 判据 ① 的预算是 8-bit `p99 ≤ 8 / 全局 ≤ 32`。若在 RGBA8 域做窗口
//! 平均，量化误差（±0.5 LSB 的逐帧舍入）会先吃掉一部分预算，且多个 u8
//! 帧的均值不可还原线性光能 ⇒ 累积在数学上就不等价。故 3D 基帧必须
//! 以 `Vec<[f32;4]>` 进入 [`crate::v31_accum::TemporalState`]，只在出帧时
//! 做一次 `apply_aesthetic` + 8-bit 量化。
//!
//! # 同源纪律（与旧路径逐位可对照）
//! - 逐像素样本走 `prism_rt::integrator::render_pixel_range`，与
//!   `rt_tiles::render_frame` 的 `render_pixel` **同一 `trace_sample` /
//!   同一 `mean_samples_fill` 固定形状树**；仅不调用 `apply_aesthetic`
//!   与 `Framebuffer::set_pixel`（tone-map 延后）。
//! - 样本区间 `[k·spp, (k+1)·spp)` 的前缀语义（W7/B3）保证：W 个窗口
//!   pass 覆盖的 16 个样本 ≡ 一次 `spp=16` 直渲的 16 个样本（逐位相同），
//!   差异只来自 `running_mean` 与 `mean_samples_fill` 的浮点结合序。
//! - `CpuRefBackend`（`--fast` / `--quality fast`）**不进本旁路**（由
//!   `V31RenderDevice` 透传），零回归。
//!
//! # 降噪的依赖方向裁定
//! 设计 §6 固定 `prism-postfx → prism-render` 单向，`prism-render` 不能
//! 反向 import `prism_postfx::denoise::AtrousSpatial`。故本模块内联一份
//! **同核 / 同步距 / 同 σ** 的保守 à-trous（f32 计算域，避免 f16 量化
//! 吃掉 8-bit 判据预算）；`prism-postfx/src/denoise.rs` 另提供复用
//! `AtrousSpatial::eval_window` 的 `CpuDenoiseBackend` 作为公共接线位。
//! 两处口径一致（KERNEL `[1,4,6,4,1]/256`、3 趟 step `1,2,4`、σ=0.1）。

use prism_core::Scene;
use prism_gpu::{apply_aesthetic, AestheticReq, Framebuffer, GpuDevice};
use prism_math::Vec3;

use crate::v31_config::{QualityTier, V31Config};

/// 保守 à-trous 的 5 tap B3 样条核（与 `prism-postfx::denoise::KERNEL` 同值）。
/// `pub` 供 [`crate::v31_reconstruct`] 的可分离空域重建复用同一份核常量。
pub const ATROUS_KERNEL: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
/// 保守档迭代趟数（3 趟 / σ=0.1，设计 §5 T03 任务 5）。
pub const ATROUS_ITERATIONS: u32 = 3;
/// 色差边缘停止带宽 σ。
pub const ATROUS_SIGMA: f32 = 0.1;

/// 一帧线性渲染请求（设计 §3 `LinearFrameRequest`；T04 GPU 桥的消费面）。
///
/// 字段全部为值语义，便于 T04 在不改本 crate 的前提下实现
/// `v31_gpu::try_render_linear(req) -> Option<Vec<[f32;4]>>`。
#[derive(Debug, Clone)]
pub struct LinearFrameRequest {
    pub width: u32,
    pub height: u32,
    pub frame: u32,
    pub spp_begin: u32,
    pub spp_end: u32,
    pub bounces: u32,
    pub aesthetic: AestheticReq,
}

/// GPU 优先桥的 V3.1 收尾接线：把 `LinearFrameRequest` 翻成
/// `prism_gpu::v31_gpu::GpuLinearRequest`，向 GPU 侧请求**边缘引导场**。
///
/// # 真 GPU 自动启用 / 无则透明回落 CPU
/// 与 `rasterize_path_gpu` 同形态：物理 GPU ∧ Vulkan 双自检通过 ⇒ 真机
/// dispatch → readback 得到引导场；否则 `None`，编排层继续走 CPU 权威路径
/// （CPU 侧的空域重建本就同核，不需要再算一份引导场，故**不做 CPU 光栅**，
/// 热路径零开销）。沙箱无 `/dev/dri`、仅 lavapipe ⇒ 恒 `None`。
///
/// 返回 `None` 不是桩，而是「能力探测失败」的真实判定结果。
pub fn try_render_linear(req: &LinearFrameRequest) -> Option<Vec<f32>> {
    prism_gpu::v31_gpu::try_gpu_guide(&gpu_request_of(req))
}

/// `LinearFrameRequest` → GPU 侧请求（字段子集，值语义一对一）。
pub fn gpu_request_of(req: &LinearFrameRequest) -> prism_gpu::v31_gpu::GpuLinearRequest {
    prism_gpu::v31_gpu::GpuLinearRequest {
        width: req.width,
        height: req.height,
        frame: req.frame,
        spp_begin: req.spp_begin,
        spp_end: req.spp_end,
        bounces: req.bounces,
    }
}

/// 统一入口的「GPU 优先 / CPU 兜底」引导场（**离线 / 接线验证用，不进帧循环**）。
///
/// 无 GPU 时才做 CPU 光栅兜底，且两侧 coverage 逐位一致 ⇒ 误差不变。
pub fn linear_guide_with_fallback(req: &LinearFrameRequest) -> (Vec<f32>, &'static str) {
    prism_gpu::v31_gpu::guide_with_fallback(&gpu_request_of(req))
}

/// 设计 §3 公共 API：把 core `Scene` 在时刻 `t` 渲染为 f32 线性帧。
///
/// 曲线图元（DSL `sdf kind ∈ {cylinder,cone,torus,revolution}`）由
/// `movie.rs::render_curve_frame` 走 `wavefront::render_frame`，不经过本
/// 函数的设备请求面；本函数覆盖普通球体 / 网格 / 介质场景（600 s 达标链）。
pub fn render_frame_linear(
    _device: &mut dyn GpuDevice,
    scene: &Scene,
    t: f32,
    duration: f32,
    w: u32,
    h: u32,
    cfg: &V31Config,
) -> Vec<[f32; 4]> {
    let req = crate::build_render_request(scene, t, duration, w, h);
    render_request_linear(&req, cfg, 0, cfg.spp_per_frame, 0)
}

/// 从 `SceneRenderRequest` 渲染 f32 线性帧（样本区间形式）。
///
/// 供 `V31RenderDevice` 在窗口内逐 pass 调用；`s_begin/s_end` 决定本帧
/// 新增的 spp，`threads` 只影响墙钟、不进结果（D-20）。
pub fn render_request_linear(
    req: &prism_gpu::SceneRenderRequest,
    cfg: &V31Config,
    s_begin: u32,
    s_end: u32,
    frame: u32,
) -> Vec<[f32; 4]> {
    use prism_core::EvalCtx;
    use prism_rt::integrator::RenderParams;

    let (w, h): (u32, u32) = (req.width, req.height);
    let n: usize = (w as usize).saturating_mul(h as usize);
    if n == 0 || w > prism_gpu::MAX_FRAMEBUFFER_DIM || h > prism_gpu::MAX_FRAMEBUFFER_DIM {
        return Vec::new();
    }
    if !req.curves.is_empty() {
        return Vec::new();
    }
    let Some(cam) = crate::rt_backend::build_camera(&req.camera) else {
        return vec![[0.0; 4]; n];
    };
    let rt_scene = crate::rt_backend::translate_scene(req);
    rt_scene.assert_layout();
    let params: RenderParams = RenderParams::new(cfg.spp_per_frame.max(1), cfg.bounces.max(1));
    // 采样流三要素与 RtBackend 同源（RT_SEED=0 / node 0 / frame）。
    // frame 进 ctx 使不同窗口的样本流可区分；窗口内锁存同一 req ⇒ 同一场景。
    let ctx: EvalCtx = EvalCtx::new(crate::rt_backend::RT_SEED, 0, frame as u64);
    crate::rt_tiles::render_frame_linear(
        &rt_scene,
        &cam,
        &ctx,
        (w, h),
        &params,
        s_begin,
        s_end,
        crate::rt_tiles::auto_threads(),
    )
}

/// f32 线性帧 → RGBA8（tone-map 唯一出口）。
///
/// 与 `rt_tiles::render_tile` / `RtBackend::render_scene` 的后处理链逐式
/// 相同：`apply_aesthetic`（teal_orange → contrast → filmic → clamp01）
/// 再 `(v × 255).round()` 量化、alpha 恒 255。
pub fn rgba8_of(
    linear: &[[f32; 4]],
    aesthetic: &AestheticReq,
    w: u32,
    h: u32,
) -> Vec<[u8; 4]> {
    let mut out: Vec<[u8; 4]> = Vec::with_capacity(linear.len());
    for p in linear {
        let mapped: Vec3 = apply_aesthetic(Vec3::new(p[0], p[1], p[2]), aesthetic);
        out.push([
            channel_to_u8(mapped.x),
            channel_to_u8(mapped.y),
            channel_to_u8(mapped.z),
            255,
        ]);
    }
    // 尺寸一致性由调用方保证；此处只做防御性截断，绝不静默糊弄。
    let want: usize = (w as usize).saturating_mul(h as usize);
    if out.len() != want {
        out.resize(want, [0, 0, 0, 255]);
    }
    out
}

/// 与 `prism_gpu::Framebuffer::set_pixel` 的 `channel_to_u8` 逐式一致。
#[inline]
fn channel_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// f32 线性帧 → `Framebuffer`（RGBA8 裸字节）。
pub fn tone_map_linear(
    linear: &[[f32; 4]],
    aesthetic: &AestheticReq,
    w: u32,
    h: u32,
) -> Framebuffer {
    let rgba: Vec<[u8; 4]> = rgba8_of(linear, aesthetic, w, h);
    let mut pixels: Vec<u8> = Vec::with_capacity(rgba.len() * 4);
    for p in &rgba {
        pixels.extend_from_slice(p);
    }
    Framebuffer { width: w, height: h, pixels }
}

/// 保守 à-trous 空域降噪（3 趟 / σ=0.1，f32 计算域）。
///
/// 与 `prism_postfx::denoise::AtrousSpatial::eval_window` 同口径：
/// 第 `i` 趟步距 `2^i`，权重 `K[k]·K[l]/256 / (1 + d²/σ²)`，`d²` 为
/// RGB 色差平方，邻域坐标按边缘钳制复制。中心 tap 权重下界 `36/256`
/// ⇒ 权重和恒正、除法安全。常数场卷积恒等 ⇒ 平坦区不改值，强边缘
/// 因 `d²` 大被边缘停止函数保护 ⇒ 不糊边缘（判据 ④）。
pub fn denoise_conservative(frame: &[[f32; 4]], w: u32, h: u32) -> Vec<[f32; 4]> {
    let n: usize = (w as usize).saturating_mul(h as usize);
    if frame.len() != n || n == 0 {
        return frame.to_vec();
    }
    let sigma2: f32 = (ATROUS_SIGMA * ATROUS_SIGMA).max(1.0e-12);
    let mut buf: Vec<[f32; 4]> = frame.to_vec();
    for iter in 0..ATROUS_ITERATIONS {
        let step: i64 = 1i64 << iter;
        let mut next: Vec<[f32; 4]> = vec![[0.0; 4]; n];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let c: [f32; 4] = buf[(y as usize) * (w as usize) + (x as usize)];
                let mut acc: [f32; 4] = [0.0; 4];
                let mut wsum: f32 = 0.0;
                for l in 0..5i64 {
                    let sy: usize = (y + (l - 2) * step).clamp(0, h as i64 - 1) as usize;
                    for k in 0..5i64 {
                        let sx: usize = (x + (k - 2) * step).clamp(0, w as i64 - 1) as usize;
                        let q: [f32; 4] = buf[sy * (w as usize) + sx];
                        let dr: f32 = q[0] - c[0];
                        let dg: f32 = q[1] - c[1];
                        let db: f32 = q[2] - c[2];
                        let d2: f32 = dr * dr + dg * dg + db * db;
                        let weight: f32 = ATROUS_KERNEL[k as usize]
                            * ATROUS_KERNEL[l as usize]
                            / 256.0
                            / (1.0 + d2 / sigma2);
                        for i in 0..4 {
                            acc[i] += weight * q[i];
                        }
                        wsum += weight;
                    }
                }
                let inv: f32 = 1.0 / wsum;
                next[(y as usize) * (w as usize) + (x as usize)] = [
                    acc[0] * inv,
                    acc[1] * inv,
                    acc[2] * inv,
                    acc[3] * inv,
                ];
            }
        }
        buf = next;
    }
    buf
}

/// 是否应对该档位启用保守降噪（preview / final 默认开，fast 不开）。
pub fn denoise_enabled(cfg: &V31Config) -> bool {
    cfg.denoise && cfg.quality != QualityTier::Fast
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v31_config::{QualityTier, ResTier};

    #[test]
    fn constant_field_is_invariant_under_atrous() {
        let f: Vec<[f32; 4]> = vec![[0.3, 0.4, 0.5, 1.0]; 16 * 12];
        let out = denoise_conservative(&f, 16, 12);
        for (a, b) in f.iter().zip(out.iter()) {
            for c in 0..4 {
                assert!((a[c] - b[c]).abs() < 1.0e-6, "常数场必须恒等：{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn hard_edge_is_preserved_by_color_stop() {
        let (w, h) = (8u32, 8u32);
        let mut f: Vec<[f32; 4]> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v: f32 = if x < 4 { 0.05 } else { 0.95 };
                f.push([v, v, v, 1.0]);
            }
        }
        let out = denoise_conservative(&f, w, h);
        // 边缘两侧的中心像素应各自保持接近原值（σ=0.1 ⇒ d²=0.81 衰减 ~9 倍）。
        let left: [f32; 4] = out[3];
        let right: [f32; 4] = out[4];
        assert!((left[0] - 0.05).abs() < 0.12, "左边缘被抹平：{left:?}");
        assert!((right[0] - 0.95).abs() < 0.12, "右边缘被抹平：{right:?}");
    }

    #[test]
    fn tone_map_matches_framebuffer_set_pixel_chain() {
        let linear: Vec<[f32; 4]> = vec![[0.2, 0.5, 0.9, 1.0]; 4];
        let aes = AestheticReq { teal_orange: true, contrast: 1.1, filmic: true };
        let mut reference = Framebuffer::new(2, 2);
        for (i, p) in linear.iter().enumerate() {
            let mapped = apply_aesthetic(Vec3::new(p[0], p[1], p[2]), &aes);
            reference.set_pixel((i % 2) as u32, (i / 2) as u32, mapped);
        }
        let got = tone_map_linear(&linear, &aes, 2, 2);
        assert_eq!(got.pixels, reference.pixels, "tone-map 必须与 set_pixel 链逐字节一致");
    }

    #[test]
    fn denoise_switch_follows_quality_tier() {
        assert!(denoise_enabled(&V31Config::for_tier(ResTier::R720p, QualityTier::Final)));
        assert!(denoise_enabled(&V31Config::for_tier(ResTier::R720p, QualityTier::Preview)));
        assert!(!denoise_enabled(&V31Config::for_tier(ResTier::R720p, QualityTier::Fast)));
        let mut off = V31Config::for_tier(ResTier::R720p, QualityTier::Final);
        off.denoise = false;
        assert!(!denoise_enabled(&off));
    }

    #[test]
    fn gpu_bridge_is_transparently_unavailable() {
        let req = LinearFrameRequest {
            width: 8,
            height: 8,
            frame: 0,
            spp_begin: 0,
            spp_end: 4,
            bounces: 6,
            aesthetic: AestheticReq { teal_orange: false, contrast: 1.0, filmic: false },
        };
        assert!(try_render_linear(&req).is_none(), "lavapipe 下必须透明兜底 CPU");
    }
}
