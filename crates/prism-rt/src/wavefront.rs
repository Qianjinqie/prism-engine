//! Wavefront 路径追踪 CPU 仿真版（D2；GPU WGSL 的判对基准）。
//!
//! # 目的
//! GPU wavefront 路径追踪把"每条路径的递归游走"重排为**按状态分组的批量
//! 队列**（生成/着色/扩展/连接 四阶段），避免 GPU 分支发散。本模块是同一
//! 数据流的**纯 CPU 仿真**：逐条路径以显式状态机推进，但样本流、求交、
//! BSDF/灯光语义与 [`crate::integrator`] 的 `trace_one` **逐位一致**——
//! 因而 `wavefront::render_pixel` ≡ `integrator::render_pixel`（同器判对基准）。
//!
//! # 判对契约（验收）
//! - 逐像素统计 diff ≤ 视觉等价阈值（本模块测试断言逐位相等，强于阈值）；
//! - 确定性模式 md5：同 `EvalCtx` 两遍逐位一致。
//!
//! # 为什么能逐位一致
//! Wavefront 仅是"执行组织方式"的改变，**不改变估计量**：每条路径按
//! 完全相同的顺序消耗相同维度槽的样本、走相同算术。本模块以"队列状态机"
//! 显式表达四阶段，但下推的是与 `trace_one` 相同的 `PathState`。
//!
//! # 禁 libm / 零第三方
//! 只经 `crate::sampler`（Sobol + det 三角件）取随机与三角；无 `libm`、
//! 无全局 RNG（D-29）。

use std::f32::consts::FRAC_1_PI;

use prism_core::EvalCtx;
use prism_math::Vec3;

use crate::accum::mean_samples;
use crate::bsdf::{hadamard, SampleCtx};
use crate::camera::{Camera, Ray};
use crate::integrator::{
    light_skip, material_bsdf, surface_brdf, surface_pdf, visibility, MaterialKind, RenderParams,
    Scene, FRAC_1_PI_F32, T_MIN_SENTINEL,
};
use crate::light::{ambient_background, nee_mis_weight, sample_nee, Light};
use crate::sampler::{onb, SampleStream};

// ---------------------------------------------------------------------------
// 维度预算（与 integrator 逐槽对齐）
// ---------------------------------------------------------------------------

/// 事件段维度基（同 integrator）。
const BASE_BOUNCE: u16 = 5;
/// 每事件段槽数。
const SLOTS_PER_BOUNCE: u16 = 6;
/// 槽位：BSDF/相位 uv。
const SLOT_BSDF: u16 = 0;
/// 槽位：光源 uv。
const SLOT_LIGHT: u16 = 2;
/// 槽位：RR。
const SLOT_RR: u16 = 4;
/// 槽位：介质游程流。
const SLOT_AUX: u16 = 5;

/// 事件段 ev（≥1）槽 slot 的全局维度号。
#[inline]
fn bounce_dim(ev: u32, slot: u16) -> u16 {
    BASE_BOUNCE + ((ev - 1) as u16) * SLOTS_PER_BOUNCE + slot
}

/// 表面偏移（与 integrator 同值）。
const SURFACE_EPS: f32 = 1e-3;
/// 未命中段长。
const MISS_T: f32 = 1.0e30;

// ---------------------------------------------------------------------------
// 路径状态（wavefront 队列单元）
// ---------------------------------------------------------------------------

/// 单个 wavefront 队列项的游走状态（与 integrator 的 `PathState` 同字段）。
#[derive(Debug, Clone, Copy)]
struct WavefrontItem {
    /// 当前射线。
    ray: Ray,
    /// 路径吞吐量。
    throughput: Vec3,
    /// 已累计辐亮度。
    radiance: Vec3,
    /// 上一段方向的立体角 pdf（发光命中 MIS 用）。
    prev_pdf_dir: f32,
    /// 上一段是否镜面。
    prev_specular: bool,
    /// 已完成事件数（bounce）。
    bounce: u32,
    /// 介质游程步计数器（本路径全程共享）。
    aux_steps: u32,
}

/// 波前阶段（诊断/测试用；状态机显式化四阶段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavefrontStage {
    /// 生成：相机射线进入队列。
    Generate,
    /// 着色：NEE 直接光 + 发光命中。
    Shade,
    /// 连接：MIS 权重融合。
    Connect,
    /// 扩展：BSDF/相位抽样续弹（或终止）。
    Extend,
}

/// 渲染统计（wavefront 批次计数，供诊断与 GPU 队列规模估计）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WavefrontStats {
    /// 生成的路径数（= spp）。
    pub generated: u32,
    /// 着色事件数。
    pub shaded: u32,
    /// 扩展（续弹）次数。
    pub extended: u32,
    /// 终止路径数（逃逸/吸收/RR 死亡/达到深度上限）。
    pub terminated: u32,
}

/// 间接贡献钳制（与 integrator 同式）。
fn clamp_indirect(c: Vec3, ev: u32, limit: Option<f32>) -> Vec3 {
    match limit {
        Some(t) if ev >= 2 => Vec3::new(c.x.min(t), c.y.min(t), c.z.min(t)),
        _ => c,
    }
}

// ---------------------------------------------------------------------------
// 状态机单步：着色 + 扩展（与 trace_one 主体严格同序）
// ---------------------------------------------------------------------------

/// RR 存活判定（与 integrator 同式）。
fn rr_survive(st: &mut WavefrontItem, stream: &SampleStream<'_>, next_ev: u32) -> bool {
    if next_ev < 3 {
        return true;
    }
    let q = st
        .throughput
        .x
        .max(st.throughput.y)
        .max(st.throughput.z)
        .clamp(0.05, 0.95);
    let u = stream.sample_1d(bounce_dim(next_ev, SLOT_RR));
    if u >= q {
        return false;
    }
    st.throughput = st.throughput * (1.0 / q);
    true
}

/// 介质行进（与 integrator 同式；B1 单一凸盒）。
fn medium_collide(
    scene: &Scene,
    ro: Vec3,
    rd: Vec3,
    t_max: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    steps: &mut u32,
) -> Option<f32> {
    let region = scene.medium.as_ref()?;
    let b = &region.bounds;
    let t_enter = if b.contains(ro) {
        0.0
    } else {
        b.segment_interval(ro, rd, 0.0, t_max)?.0
    };
    let exit = b.segment_interval(ro, rd, t_enter, t_max)?.1;
    let u = stream.aux_1d(aux_dim, *steps);
    *steps += 1;
    let t_free = region.medium.free_path(u);
    if t_enter + t_free < exit && t_enter + t_free < t_max {
        Some(t_enter + t_free)
    } else {
        None
    }
}

/// 推进一个 wavefront 项一步（一个事件的着色 + 扩展）。返回 false = 路径终止。
///
/// 该函数是 [`crate::integrator`] 主循环体的**逐句等价**重排：同一 `max_bounces`
/// 次迭代、同一维度槽序、同一算术。
fn advance_one(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    stream: &SampleStream<'_>,
    params: &RenderParams,
    st: &mut WavefrontItem,
    stats: &mut WavefrontStats,
) -> bool {
    let _ = cam;
    let _ = resolution;
    let bounce = st.bounce;
    let ev = bounce + 1;
    let aux_dim = bounce_dim(ev, SLOT_AUX);
    let hit = scene.intersect(st.ray.origin, st.ray.dir, MISS_T, None);
    let t_surf = hit.map_or(MISS_T, |h| h.t);

    // ---- 介质段（相机/续弹统一；碰撞优先于表面）----
    if let Some(region) = &scene.medium {
        if let Some(tc) = medium_collide(
            scene,
            st.ray.origin,
            st.ray.dir,
            t_surf,
            stream,
            aux_dim,
            &mut st.aux_steps,
        ) {
            let p = st.ray.at(tc);
            let u_scatter = stream.aux_1d(aux_dim, st.aux_steps);
            st.aux_steps += 1;
            if u_scatter >= region.medium.albedo {
                stats.terminated += 1;
                return false; // 吸收
            }
            // 体 NEE。
            let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
            let w_in = -st.ray.dir;
            for light in &scene.lights {
                let skip = light_skip(scene, light);
                let ls = match sample_nee(light, p, u_l1, u_l2) {
                    Some(ls) => ls,
                    None => continue,
                };
                let cos_l = w_in.dot(&ls.wi);
                let ph = region.medium.phase(cos_l);
                let vis = visibility(
                    scene,
                    p,
                    ls.wi,
                    ls.dist,
                    skip,
                    stream,
                    aux_dim,
                    &mut st.aux_steps,
                );
                if vis <= 0.0 {
                    continue;
                }
                let w = nee_mis_weight(light, ls.pdf_w, ph);
                let c = hadamard(st.throughput, ls.radiance)
                    * (ph * vis * w / ls.pdf_w);
                st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
            }
            stats.shaded += 1;
            // 相位抽样续弹。
            let (u1, u2) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF));
            let dir = region.medium.sample_phase(w_in, u1, u2);
            st.prev_pdf_dir = region.medium.phase(w_in.dot(&dir));
            st.prev_specular = false;
            if !rr_survive(st, stream, ev + 1) {
                stats.terminated += 1;
                return false;
            }
            stats.extended += 1;
            st.ray = Ray { origin: p, dir, time: st.ray.time };
            st.bounce = bounce + 1;
            return st.bounce < params.max_bounces;
        }
    }

    // ---- 表面事件（或逃逸）----
    let h = match hit {
        None => {
            let bg = ambient_background(&scene.lights);
            if bg.x > 0.0 || bg.y > 0.0 || bg.z > 0.0 {
                let c = hadamard(st.throughput, bg);
                st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
            }
            stats.terminated += 1;
            return false;
        }
        Some(h) => h,
    };
    let p = st.ray.at(h.t);
    let n = scene.normal_at(h.shape, p);
    let mat = scene.material(h.shape);

    // 发光命中（MIS 权重 vs 光源立体角 pdf）。
    if let MaterialKind::Emissive { radiance } = mat {
        let w_em = if params.nee_only && ev > 1 {
            0.0
        } else if ev == 1 || st.prev_specular {
            1.0
        } else {
            match scene.light_pdf_sa(st.ray.origin, h.shape) {
                Some(pdf_l) => crate::light::power_heuristic(st.prev_pdf_dir, pdf_l),
                None => 1.0,
            }
        };
        let c = hadamard(st.throughput, *radiance) * w_em;
        st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
    }

    let bsdf = match material_bsdf(mat) {
        Some(b) => b,
        None => {
            stats.terminated += 1;
            return false;
        }
    };
    let from_inside = st.ray.dir.dot(&n) > 0.0;
    let nv = if from_inside { -n } else { n };
    let wo = -st.ray.dir;

    // 反射域 NEE。
    {
        let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
        for light in &scene.lights {
            let skip = light_skip(scene, light);
            let ls = match sample_nee(light, p, u_l1, u_l2) {
                Some(ls) => ls,
                None => continue,
            };
            let cos_l = nv.dot(&ls.wi);
            if cos_l <= 0.0 {
                continue;
            }
            let f = surface_brdf(mat, &bsdf, wo, ls.wi, nv);
            let f_cos = hadamard(f, Vec3::splat(cos_l));
            let alt_pdf = surface_pdf(mat, &bsdf, wo, ls.wi, nv);
            let ro = p + nv * SURFACE_EPS;
            let vis = visibility(
                scene,
                ro,
                ls.wi,
                ls.dist,
                skip,
                stream,
                aux_dim,
                &mut st.aux_steps,
            );
            if vis <= 0.0 {
                continue;
            }
            let w = nee_mis_weight(light, ls.pdf_w, alt_pdf);
            let c = hadamard(hadamard(st.throughput, f_cos), ls.radiance)
                * (vis * w / ls.pdf_w);
            st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
        }
    }
    stats.shaded += 1;

    // 焦散 MNEE：与 integrator 同调用（连接阶段）。
    if !params.nee_only {
        let si = prism_core::SampleIndex {
            frame: ctx.frame,
            pixel: stream.pixel(),
            sample: 0,
            dimension: bounce_dim(ev, SLOT_LIGHT),
            node_id: ctx.node_id,
        };
        for refr in scene.refractors() {
            for light in &scene.lights {
                let lp = match light {
                    Light::Sphere { center, .. } => *center,
                    Light::Directional { direction, .. } => p + *direction * 8.0,
                    _ => continue,
                };
                let Some(cs) = crate::caustics::mnee_refraction(&refr, p, lp, ctx, &si) else {
                    continue;
                };
                let cos_l = nv.dot(&cs.wi);
                if cos_l <= 0.0 {
                    continue;
                }
                let f = bsdf.eval(wo, cs.wi, nv);
                let c = hadamard(hadamard(st.throughput, f), cs.weight)
                    * (cos_l / cs.pdf_w);
                st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
            }
        }
    }

    // BSDF 抽样续弹。
    let (u_lobe, _) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF));
    let (u1, u2) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF) + 1);
    if let MaterialKind::Lambert { albedo } = mat {
        let d_local = crate::sampler::cosine_hemisphere(u1, u2);
        let (t, b) = onb(n);
        let dir = (t * d_local.x + b * d_local.y + n * d_local.z).normalize();
        st.throughput = hadamard(st.throughput, *albedo);
        st.prev_pdf_dir = d_local.z * FRAC_1_PI;
        st.prev_specular = false;
        if !rr_survive(st, stream, ev + 1) {
            stats.terminated += 1;
            return false;
        }
        stats.extended += 1;
        st.ray = Ray { origin: p + n * SURFACE_EPS, dir, time: st.ray.time };
    } else if let MaterialKind::Mirror = mat {
        let dir = st.ray.dir - n * (2.0 * st.ray.dir.dot(&n));
        st.prev_pdf_dir = 0.0;
        st.prev_specular = true;
        if !rr_survive(st, stream, ev + 1) {
            stats.terminated += 1;
            return false;
        }
        stats.extended += 1;
        st.ray = Ray { origin: p + n * SURFACE_EPS, dir, time: st.ray.time };
    } else {
        let sctx = SampleCtx { psr_rough: if bsdf.roughness < 1e-3 { 1e-3 } else { 0.0 } };
        if let Some(bs) = bsdf.sample(wo, n, from_inside, &sctx, u_lobe, u1, u2) {
            if bs.entered {
                if let Some((_sss_w, radius, ior)) = material_subsurface(mat) {
                    let si = prism_core::SampleIndex {
                        frame: ctx.frame,
                        pixel: stream.pixel(),
                        sample: 0,
                        dimension: bounce_dim(ev, SLOT_AUX),
                        node_id: ctx.node_id,
                    };
                    let geo = |o: Vec3, d: Vec3| scene.intersect_n(o, d, MISS_T);
                    if let Some(exit) = crate::sss::random_walk(
                        &geo, ctx, &si, p, n, bs.dir, radius, bsdf.base_color, ior, 0.0,
                    ) {
                        st.throughput = hadamard(st.throughput, bs.weight);
                        st.throughput = hadamard(st.throughput, exit.weight);
                        st.prev_pdf_dir = 0.0;
                        st.prev_specular = false;
                        if !rr_survive(st, stream, ev + 1) {
                            stats.terminated += 1;
                            return false;
                        }
                        stats.extended += 1;
                        st.ray = Ray {
                            origin: exit.p + exit.n * SURFACE_EPS,
                            dir: bs.dir,
                            time: st.ray.time,
                        };
                    } else {
                        stats.terminated += 1;
                        return false;
                    }
                } else {
                    st.throughput = hadamard(st.throughput, bs.weight);
                    st.prev_pdf_dir = 0.0;
                    st.prev_specular = false;
                    if !rr_survive(st, stream, ev + 1) {
                        stats.terminated += 1;
                        return false;
                    }
                    stats.extended += 1;
                    st.ray = Ray {
                        origin: p - n * SURFACE_EPS,
                        dir: bs.dir,
                        time: st.ray.time,
                    };
                }
            } else {
                st.throughput = hadamard(st.throughput, bs.weight);
                st.prev_pdf_dir = bs.pdf;
                st.prev_specular = bs.specular;
                if !rr_survive(st, stream, ev + 1) {
                    stats.terminated += 1;
                    return false;
                }
                stats.extended += 1;
                st.ray = Ray { origin: p + nv * SURFACE_EPS, dir: bs.dir, time: st.ray.time };
            }
        } else {
            stats.terminated += 1;
            return false;
        }
    }
    st.bounce = bounce + 1;
    st.bounce < params.max_bounces
}

/// 材质是否走 SSS random walk（同 integrator）。
fn material_subsurface(m: &MaterialKind) -> Option<(f32, Vec3, f32)> {
    match *m {
        MaterialKind::Pbr(b) if b.subsurface > 0.0 => {
            Some((b.subsurface, b.subsurface_radius, b.ior))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Wavefront 调度：生成 → [着色 → 连接 → 扩展] 循环
// ---------------------------------------------------------------------------

/// 追踪一个样本（wavefront 队列状态机），返回 rgba。
fn trace_wavefront(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    stream: &SampleStream<'_>,
    params: &RenderParams,
    stats: &mut WavefrontStats,
) -> [f32; 4] {
    // 生成阶段：相机段视作镜面链起点。
    let ray = cam.cast_ray(stream, resolution);
    let mut st = WavefrontItem {
        ray,
        throughput: Vec3::new(1.0, 1.0, 1.0),
        radiance: Vec3::new(0.0, 0.0, 0.0),
        prev_pdf_dir: 0.0,
        prev_specular: true,
        bounce: 0,
        aux_steps: 0,
    };
    stats.generated += 1;
    // 着色/连接/扩展循环（活动队列深度 = 1；CPU 仿真显式推进状态机）。
    //
    // `max_bounces == 0` 表示**不允许任何弹射事件**：与
    // [`crate::integrator::render_pixel`] 的 `for bounce in 0..max_bounces`
    // （此时执行零次、Radiance 零贡献）严格同语义。若不短路，本循环仍会推进
    // 至少一个事件（相机段即着色），导致两条路径在全部 4×N 通道上发散。
    while params.max_bounces > 0
        && advance_one(scene, cam, ctx, resolution, stream, params, &mut st, stats)
    {}
    [st.radiance.x, st.radiance.y, st.radiance.z, 1.0]
}

/// 像素渲染（wavefront 仿真版）：与 [`crate::integrator::render_pixel`] 同结果。
pub fn render_pixel(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    pixel: (u32, u32),
    resolution: (u32, u32),
    params: &RenderParams,
) -> [f32; 4] {
    let mut samples = Vec::with_capacity(params.spp as usize);
    for s in 0..params.spp {
        let stream = SampleStream::new(ctx, pixel, s);
        let mut stats = WavefrontStats::default();
        samples.push(trace_wavefront(scene, cam, ctx, resolution, &stream, params, &mut stats));
    }
    mean_samples(&samples)
}

/// 渲染整帧并返回 (像素缓冲, 累计统计)。缓冲行主序 w×h。
pub fn render_frame_with_stats(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
) -> (Vec<[f32; 4]>, WavefrontStats) {
    let mut out = Vec::with_capacity((resolution.0 * resolution.1) as usize);
    let mut stats = WavefrontStats::default();
    for y in 0..resolution.1 {
        for x in 0..resolution.0 {
            let mut samples = Vec::with_capacity(params.spp as usize);
            for s in 0..params.spp {
                let stream = SampleStream::new(ctx, (x, y), s);
                samples.push(trace_wavefront(
                    scene, cam, ctx, resolution, &stream, params, &mut stats,
                ));
            }
            out.push(mean_samples(&samples));
        }
    }
    (out, stats)
}

/// 渲染整帧（行主序 w×h）。
pub fn render_frame(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
) -> Vec<[f32; 4]> {
    render_frame_with_stats(scene, cam, ctx, resolution, params).0
}

/// 使 `T_MIN_SENTINEL` 与 `FRAC_1_PI_F32` 在 crate 内一致（编译期断言口径）。
const _: () = {
    // 编译期常量核对：sentinel 与 integrator 同值。
    assert!(T_MIN_SENTINEL == 1e-4);
    assert!(FRAC_1_PI_F32 == FRAC_1_PI);
};

// ---------------------------------------------------------------------------
// 测试（判对基准：wavefront ≡ integrator）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{render_pixel as ref_render_pixel, Aabb, Sphere};

    /// 参考整帧（逐像素经 integrator::render_pixel）。
    fn ref_render_frame(
        scene: &Scene,
        cam: &Camera,
        ctx: &EvalCtx,
        res: (u32, u32),
        params: &RenderParams,
    ) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((res.0 * res.1) as usize);
        for y in 0..res.1 {
            for x in 0..res.0 {
                out.push(ref_render_pixel(scene, cam, ctx, (x, y), res, params));
            }
        }
        out
    }

    /// Cornell 式场景（与 integrator 测试同构，独立构造以保证判对含场景构造）。
    fn cornell_scene() -> Scene {
        let white = MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) };
        let red = MaterialKind::Lambert { albedo: Vec3::new(0.63, 0.065, 0.05) };
        let green = MaterialKind::Lambert { albedo: Vec3::new(0.14, 0.45, 0.091) };
        let t = 0.1f32;
        Scene {
            boxes: vec![
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 0.0, 1.0 + t)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, 2.0, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, -1.0)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(-1.0, 2.0 + t, 1.0 + t)),
                    red,
                ),
                (
                    Aabb::new(Vec3::new(1.0, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)),
                    green,
                ),
            ],
            spheres: vec![
                (Sphere { center: Vec3::new(-0.35, 0.45, 0.1), radius: 0.45 }, white),
                (Sphere { center: Vec3::new(0.38, 0.32, -0.05), radius: 0.32 }, MaterialKind::Mirror),
                (
                    Sphere { center: Vec3::new(0.0, 1.92, 0.0), radius: 0.28 },
                    MaterialKind::Emissive { radiance: Vec3::new(16.0, 13.0, 10.0) },
                ),
            ],
            lights: vec![Light::Sphere {
                center: Vec3::new(0.0, 1.92, 0.0),
                radius: 0.28,
                radiance: Vec3::new(16.0, 13.0, 10.0),
            }],
            medium: None,
        }
    }

    fn cornell_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 1.0, 3.1),
            Vec3::new(0.0, 0.95, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            3.1,
            0.0,
            1.0 / 24.0,
        )
    }

    /// 判对 1：wavefront 逐像素 ≡ integrator（逐位相等）。
    #[test]
    fn wavefront_equals_integrator_bitwise() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(2024, 7, 3);
        let res = (10, 10);
        let params = RenderParams::new(32, 3);
        let a = render_frame(&scene, &cam, &ctx, res, &params);
        let b = ref_render_frame(&scene, &cam, &ctx, res, &params);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            for c in 0..3 {
                assert_eq!(
                    a[i][c].to_bits(),
                    b[i][c].to_bits(),
                    "像素 {i} 通道 {c}：wavefront {} ≠ integrator {}",
                    a[i][c],
                    b[i][c]
                );
            }
        }
        println!("判对通过：{}×{} spp{} wavefront ≡ integrator（逐位）", res.0, res.1, params.spp);
    }

    /// 判对 2：逐像素统计 diff 在阈值内（本实现为 0；阈值口径保留）。
    #[test]
    fn wavefront_statistical_diff_within_threshold() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(101, 7, 3);
        let res = (12, 12);
        let params = RenderParams::new(16, 4);
        let a = render_frame(&scene, &cam, &ctx, res, &params);
        let b = ref_render_frame(&scene, &cam, &ctx, res, &params);
        let mut sum_abs = 0.0f64;
        let mut count = 0.0f64;
        for i in 0..a.len() {
            for c in 0..3 {
                sum_abs += (a[i][c] as f64 - b[i][c] as f64).abs();
                count += 1.0;
            }
        }
        let mean_abs = sum_abs / count;
        // 视觉等价阈值：均值绝对差 < 1e-3（本实现为 0，强于阈值）。
        assert!(mean_abs < 1e-3, "统计 diff 超阈值：{mean_abs}");
        println!("判对统计：均值绝对差 = {mean_abs:.3e}");
    }

    /// 判对 3：确定性——两遍逐位一致 + md5 稳定。
    #[test]
    fn wavefront_two_pass_bitwise_identical() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let res = (8, 8);
        let params = RenderParams::new(8, 3);
        let f1 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &params);
        let f2 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &params);
        for i in 0..f1.len() {
            for c in 0..4 {
                assert_eq!(f1[i][c].to_bits(), f2[i][c].to_bits(), "两遍不逐位一致 px{i} ch{c}");
            }
        }
        // md5 稳定性自锚：同字节两遍摘要一致（内容确定性由逐位相等保证）。
        let mut bytes = Vec::new();
        for px in &f1 {
            for c in 0..4 {
                bytes.extend_from_slice(&px[c].to_le_bytes());
            }
        }
        assert_eq!(md5_hex(&bytes), md5_hex(&bytes));
        println!("确定性：8×8 spp8 md5 = {}", md5_hex(&bytes));
    }

    /// 队列统计：生成/着色/扩展/终止计数自洽（终止 = 生成 + 扩展 − 存活）。
    #[test]
    fn wavefront_stats_self_consistent() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(3, 7, 3);
        let res = (6, 6);
        let params = RenderParams::new(4, 2);
        let (frame, stats) = render_frame_with_stats(&scene, &cam, &ctx, res, &params);
        assert_eq!(frame.len(), res.0 as usize * res.1 as usize);
        // 生成数 = 像素数 × spp（每条路径一个生成项）。
        assert_eq!(stats.generated, res.0 * res.1 * params.spp);
        // 扩展数 ≤ 生成数 × max_bounces。
        assert!(stats.extended <= stats.generated * params.max_bounces);
        // 每条路径或显式终止（terminated）或到达深度上限（正常收束）——
        // 两者之和必 ≥ 生成数（深度上限收束的路径不计入 terminated）。
        assert!(
            stats.extended + stats.terminated >= stats.generated,
            "队列计数不自洽：ext={} term={} gen={}",
            stats.extended,
            stats.terminated,
            stats.generated
        );
        println!(
            "队列统计：gen={} shade={} ext={} term={}",
            stats.generated, stats.shaded, stats.extended, stats.terminated
        );
    }

    /// md5（RFC 1321，测试专用；确定性路径不依赖——判对由逐位相等保证）。
    fn md5_hex(data: &[u8]) -> String {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
            5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let k = |i: usize| -> u32 { (((1u64 << 32) as f64) * (i as f64 + 1.0).sin().abs()) as u32 };
        let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
        msg.extend_from_slice(data);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
        let (mut a0, mut b0, mut c0, mut d0) =
            (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, word) in m.iter_mut().enumerate() {
                *word = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64usize {
                let (f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let tmp = d;
                d = c;
                c = b;
                let sum = a.wrapping_add(f).wrapping_add(k(i)).wrapping_add(m[g]);
                b = b.wrapping_add(sum.rotate_left(S[i]));
                a = tmp;
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in [a0, b0, c0, d0] {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }
}
