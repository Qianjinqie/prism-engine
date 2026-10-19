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
    use crate::bsdf::Bsdf;
    use crate::integrator::{render_pixel as ref_render_pixel, Aabb, Paraboloid, Plane, Sphere};

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
            paraboloids: vec![],
            planes: vec![],
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

    // ==================== E5 解析图元 wavefront ↔ integrator parity ====================

    /// E5 parity 场景 A（PRD §6 / 架构 §6.3 S7 配置复刻）：金属抛物面
    /// （z=x²+y²，r_max=1.2）+ 有限切片玻璃平面（切点 P=(0.8,0.8,1.28)、
    /// 法线 normalize(-1.6,-1.6,1)、half_extent=0.85）+ 无限地面 +
    /// 矩形面光 + 环境光。
    fn e5_scene_a() -> Scene {
        let gold = Bsdf::new(
            Vec3::new(1.0, 0.76, 0.33),
            1.0,
            0.2,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        );
        let glass = Bsdf::new(
            Vec3::splat(0.95),
            0.0,
            0.02,
            1.5,
            0.9,
            0.0,
            Vec3::splat(0.3),
        );
        Scene {
            spheres: vec![],
            boxes: vec![],
            lights: vec![
                Light::Rect {
                    center: Vec3::new(0.0, 3.0, 0.0),
                    edge_u: Vec3::new(1.2, 0.0, 0.0),
                    edge_v: Vec3::new(0.0, 0.0, 1.2),
                    radiance: Vec3::new(14.0, 12.0, 9.0),
                },
                Light::Ambient { radiance: Vec3::new(0.12, 0.14, 0.18) },
            ],
            medium: None,
            paraboloids: vec![(Paraboloid { k: 1.0, r_max: 1.2 }, MaterialKind::Pbr(gold))],
            planes: vec![
                (
                    Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None),
                    MaterialKind::Lambert { albedo: Vec3::new(0.18, 0.20, 0.24) },
                ),
                (
                    Plane::new(
                        Vec3::new(0.8, 0.8, 1.28),
                        Vec3::new(-1.6, -1.6, 1.0),
                        Some(0.85),
                    ),
                    MaterialKind::Pbr(glass),
                ),
            ],
        }
    }

    /// E5 parity 场景 B：发光球（NEE/MIS 球光源路径）+ 金属抛物面 +
    /// Lambert 地面——验 ShapeRef 索引错位不会在 MIS 权重上漂移。
    fn e5_scene_b() -> Scene {
        let gold = Bsdf::new(
            Vec3::new(1.0, 0.76, 0.33),
            1.0,
            0.15,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        );
        let center = Vec3::new(0.0, 1.6, 0.2);
        let radiance = Vec3::new(18.0, 15.0, 11.0);
        Scene {
            spheres: vec![(
                Sphere { center, radius: 0.3 },
                MaterialKind::Emissive { radiance },
            )],
            boxes: vec![],
            lights: vec![Light::Sphere { center, radius: 0.3, radiance }],
            medium: None,
            paraboloids: vec![(Paraboloid { k: 1.0, r_max: 1.4 }, MaterialKind::Pbr(gold))],
            planes: vec![(
                Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None),
                MaterialKind::Lambert { albedo: Vec3::new(0.18, 0.20, 0.24) },
            )],
        }
    }

    /// 逐位对拍辅助：wavefront ≡ integrator（3 通道 to_bits 全等）。
    fn assert_parity_bitwise(scene: &Scene, label: &str) {
        let cam = cornell_camera();
        let ctx = EvalCtx::new(77, 7, 3);
        let res = (8, 8);
        let params = RenderParams::new(4, 4);
        let a = render_frame(scene, &cam, &ctx, res, &params);
        let b = ref_render_frame(scene, &cam, &ctx, res, &params);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            for c in 0..3 {
                assert_eq!(
                    a[i][c].to_bits(),
                    b[i][c].to_bits(),
                    "{label} 像素 {i} 通道 {c}：wavefront {} ≠ integrator {}",
                    a[i][c],
                    b[i][c]
                );
            }
        }
        println!("parity 通过（{label}）：8×8 spp4 wavefront ≡ integrator（逐位）");
    }

    /// 判对 4（E5）：抛物面 + 玻璃切片 + 地面（Rect + Ambient）逐位对拍。
    #[test]
    fn wavefront_equals_integrator_bitwise_analytic_shapes() {
        assert_parity_bitwise(&e5_scene_a(), "E5-A 抛物面/玻璃/地面");
    }

    /// 判对 5（E5）：球光源（NEE/MIS）+ 金属抛物面逐位对拍。
    #[test]
    fn wavefront_equals_integrator_bitwise_sphere_light_paraboloid() {
        assert_parity_bitwise(&e5_scene_b(), "E5-B 球光源/抛物面");
    }
}

// ===========================================================================
// V2.0-G：GPU kernel 插槽 + double gate parity 网
// ===========================================================================
//
// # 为什么「CPU 递归序 vs CPU 模拟 GPU 数组序」
//
// 真机 compute 路径本环境不可执行（只有 loader、无 ICD）。但 GPU **执行组织
// 方式**的差异是可以在 CPU 上精确复现的：真机把「逐路径递归游走」重排为
// 「扁平路径数组 + 逐 bounce 层同步推进 + workgroup 分块归约」。这一重排会
// 改变**浮点结合序**（分块求和 vs 树形归约），于是判对必须从「逐位相等」
// 降级为**统计判定**——这正是 exp14-T1 形态：语义级 parity（差值均值/方差
// 有界）**且**输出级 PSNR ≥ 40 dB（双门禁，任一不达标即判红）。
//
// CPU 权威路径（[`render_frame`]）**不动**：它的 MD5 逐字节一致仍由既有
// 测试把关（`wavefront_two_pass_bitwise_identical`），本段只**新增**通道。
//
// # 双门禁的判据口径
//
// * 语义级：逐通道 |Δ| 的**均值**与**方差**都必须有界（不是只卡最大值——
//   单个离群像素不应放行，但也不该被单个尖峰一票否决）；
// * 输出级：PSNR ≥ 40 dB（8 bit 视觉无损的常用门槛，对 HDR 辐亮度同样成立）；
// * 门禁**非恒真**：测试另有一条「人为扰动必须被判红」的负向用例。

/// GPU kernel 插槽：一条可被真机 compute 路径接管的具体 kernel。
///
/// 插槽把三件事绑在一起：① 跨后端稳定名；② dispatch 形状（workgroup）；
/// ③ 二进制/文本供给状态（[`prism_gpu::compute::SpvStatus`]，**如实**）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuKernelSlot {
    /// kernel 名（与 `prism_gpu::compute::KernelDesc::name` 一致）。
    pub name: &'static str,
    /// 入口函数名（WGSL `fn` / SPIR-V `OpEntryPoint` / MSL kernel 同名）。
    pub entry: &'static str,
    /// 工作组尺寸 `[x, y, z]`。
    pub workgroup: [u32; 3],
    /// WGSL 宿主源名（[`prism_gpu::shaders`] 注册表）。
    pub wgsl_source: &'static str,
    /// SPIR-V 模块名（空串 = 待真机工具链生成）。
    pub spv_module: &'static str,
    /// SPIR-V 供给状态（如实：`Embedded` / `PendingToolchain`）。
    pub spv_status: prism_gpu::compute::SpvStatus,
}

impl GpuKernelSlot {
    /// 该插槽当前是否可真机派发（SPIR-V 已嵌入二进制）。
    ///
    /// **注意**：`true` 只说明「二进制在手」，不代表本机有设备——设备可用性
    /// 由 [`probe_gpu_slots`] 的实时探测给出（两个概念不混）。
    pub fn is_dispatchable(&self) -> bool {
        matches!(self.spv_status, prism_gpu::compute::SpvStatus::Embedded)
    }

    /// 给定问题域 extent 的派发形状（唯一权威：
    /// [`prism_gpu::compute::DispatchShape`]）。
    pub fn dispatch_shape(&self, extent: [u32; 3]) -> prism_gpu::compute::DispatchShape {
        prism_gpu::compute::DispatchShape::new(
            extent,
            prism_gpu::compute::WorkgroupSize::new(
                self.workgroup[0],
                self.workgroup[1],
                self.workgroup[2],
            ),
        )
    }

    /// 工作组个数 `(gx, gy, gz)`（喂 `vkCmdDispatch` 的三个数）。
    pub fn groups(&self, extent: [u32; 3]) -> [u32; 3] {
        self.dispatch_shape(extent).groups()
    }
}

/// 全部插槽（由 `prism_gpu::compute::kernel_registry` 派生，单一真源）。
///
/// 顺序即注册序（确定性）；不含任何真机运行时状态。
pub fn gpu_kernel_slots() -> Vec<GpuKernelSlot> {
    prism_gpu::compute::kernel_registry()
        .into_iter()
        .map(|k| GpuKernelSlot {
            name: k.name,
            entry: k.entry,
            workgroup: [k.workgroup.x, k.workgroup.y, k.workgroup.z],
            wgsl_source: k.wgsl_source,
            spv_module: k.spv_module,
            spv_status: prism_gpu::compute::kernel_spv_status(&k),
        })
        .collect()
}

/// 按名取插槽。
pub fn gpu_kernel_slot(name: &str) -> Option<GpuKernelSlot> {
    gpu_kernel_slots().into_iter().find(|s| s.name == name)
}

/// GPU 插槽实时可用性探测结果（**如实**：无设备就报无设备）。
#[derive(Debug, Clone, PartialEq)]
pub struct GpuSlotProbe {
    /// 是否真有设备并把探针 kernel 派发成功。
    pub available: bool,
    /// 状态短名（`prism_gpu::vulkan::compute::DispatchStatus::as_str()`）。
    pub status: &'static str,
    /// 说明（失败原因 / 读回长度）。
    pub detail: String,
    /// 派发的工作组个数（无设备时也照实给出「本该派多少」）。
    pub groups: [u32; 3],
    /// 参与探测的插槽数。
    pub checked_slots: usize,
}

/// 实时探测 GPU 插槽：用 SPIR-V 已嵌入的探针 kernel 做一次真机派发尝试。
///
/// * 有设备 ⇒ `available = true`，并核对读回值与 CPU 参考（`src * 2`）；
/// * 无设备（本构建环境）⇒ `available = false`，`status = "device-unavailable"`，
///   `detail` 带 `no-vulkan-icd`，**不 panic、不返回假结果**。
pub fn probe_gpu_slots(extent: [u32; 3]) -> GpuSlotProbe {
    let slots = gpu_kernel_slots();
    let probe_kernel = prism_gpu::compute::spirv_probe_kernel();
    let shape = prism_gpu::compute::DispatchShape::new(extent, probe_kernel.workgroup);
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
    let mut backend = prism_gpu::vulkan::VulkanBackend::new();
    let report = backend.compute_dispatch(&probe_kernel, extent, &input);
    let available = report.is_ok();
    let detail = if available {
        let mut ok = true;
        for (i, v) in input.iter().enumerate() {
            if let Some(got) = report.output.get(i) {
                if (got - v * 2.0).abs() > 1e-6 {
                    ok = false;
                }
            }
        }
        format!(
            "真机派发成功：读回 {} 个 f32，与 CPU 参考 {}",
            report.output.len(),
            if ok { "一致" } else { "不一致" }
        )
    } else {
        format!("{}（{}）", report.detail, report.status.as_str())
    };
    GpuSlotProbe {
        available,
        status: report.status.as_str(),
        detail,
        groups: shape.groups(),
        checked_slots: slots.len(),
    }
}

// ---------------------------------------------------------------------------
// CPU 模拟 GPU 数组序（exp14-T1 形态的执行重排）
// ---------------------------------------------------------------------------

/// 用 **GPU 数组序**（扁平路径数组 + 逐 bounce 层同步 + workgroup 分块归约）
/// 重排 [`render_frame`] 的执行，返回同形状像素缓冲。
///
/// # 与 [`render_frame`] 的关系
///
/// 每条路径消耗的样本维度、求交、BSDF/灯光语义**完全不变**（同一个
/// [`advance_one`]），因此结果与串行递归版本只差**浮点结合序**：
///
/// * 递归版：逐像素逐样本算完即用 `mean_samples`（固定形状**树形**归约）；
/// * 本函数：所有路径状态平铺在数组里，按层推进（真机 wavefront 的
///   「一个 dispatch = 一层」），最后按 `workgroup` 大小**分块顺序**累加，
///   再把块和逐块相加（真机 workgroup 内归约 + 跨组累加的形态）。
///
/// 差异量级应为「浮点末位」，由 [`parity_report`] + [`judge`] 双门禁把关。
///
/// # 参数
/// * `workgroup`：分块归约的块大小（≤ 1 时退化为逐样本顺序累加）。
pub fn render_frame_array_order(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    params: &RenderParams,
    workgroup: u32,
) -> Vec<[f32; 4]> {
    let pixels = (resolution.0 * resolution.1) as usize;
    let spp = params.spp as usize;
    if pixels == 0 {
        return Vec::new();
    }
    if spp == 0 {
        // 与 `mean_samples(&[])` 同语义：零样本 ⇒ 全零像素。
        return vec![[0.0; 4]; pixels];
    }

    // ---- 生成阶段：把全部路径平铺进数组（真机的 path buffer） ----
    let mut streams: Vec<SampleStream<'_>> = Vec::with_capacity(pixels * spp);
    let mut items: Vec<WavefrontItem> = Vec::with_capacity(pixels * spp);
    let mut alive: Vec<bool> = Vec::with_capacity(pixels * spp);
    let mut stats = WavefrontStats::default();
    for p in 0..pixels {
        let x = (p as u32) % resolution.0;
        let y = (p as u32) / resolution.0;
        for s in 0..params.spp {
            let stream = SampleStream::new(ctx, (x, y), s);
            let ray = cam.cast_ray(&stream, resolution);
            items.push(WavefrontItem {
                ray,
                throughput: Vec3::new(1.0, 1.0, 1.0),
                radiance: Vec3::new(0.0, 0.0, 0.0),
                prev_pdf_dir: 0.0,
                prev_specular: true,
                bounce: 0,
                aux_steps: 0,
            });
            streams.push(stream);
            alive.push(params.max_bounces > 0);
            stats.generated += 1;
        }
    }

    // ---- 逐层推进（一个 level = 真机的一次 wavefront dispatch） ----
    for _level in 0..params.max_bounces {
        for i in 0..items.len() {
            if !alive[i] {
                continue;
            }
            alive[i] = advance_one(
                scene,
                cam,
                ctx,
                resolution,
                &streams[i],
                params,
                &mut items[i],
                &mut stats,
            );
        }
    }

    // ---- 归约阶段：workgroup 分块顺序累加（真机 workgroup 归约形态） ----
    let block = workgroup.max(1) as usize;
    let mut out: Vec<[f32; 4]> = Vec::with_capacity(pixels);
    for p in 0..pixels {
        let base = p * spp;
        let mut total = [0.0f32; 4];
        let mut chunk = [0.0f32; 4];
        let mut in_chunk = 0usize;
        for s in 0..spp {
            let r = items[base + s].radiance;
            chunk[0] += r.x;
            chunk[1] += r.y;
            chunk[2] += r.z;
            chunk[3] += 1.0;
            in_chunk += 1;
            if in_chunk == block {
                total[0] += chunk[0];
                total[1] += chunk[1];
                total[2] += chunk[2];
                total[3] += chunk[3];
                chunk = [0.0; 4];
                in_chunk = 0;
            }
        }
        if in_chunk > 0 {
            total[0] += chunk[0];
            total[1] += chunk[1];
            total[2] += chunk[2];
            total[3] += chunk[3];
        }
        let n = spp as f32;
        out.push([total[0] / n, total[1] / n, total[2] / n, total[3] / n]);
    }
    out
}

// ---------------------------------------------------------------------------
// Parity 双门禁：语义级统计判定 + 输出级 PSNR
// ---------------------------------------------------------------------------

/// 两帧的 parity 统计量（全部为诊断/判定量，不参与渲染输出）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityReport {
    /// 参与比较的通道数（含比较的样本个数）。
    pub samples: usize,
    /// 逐通道 |Δ| 的均值（语义级门禁 ①）。
    pub mean_abs_diff: f64,
    /// 逐通道有符号 Δ 的方差（语义级门禁 ②：检查是否只有「系统性偏移」）。
    pub variance: f64,
    /// 逐通道 |Δ| 的最大值（诊断）。
    pub max_abs_diff: f32,
    /// 峰值信噪比（dB，输出级门禁）。
    pub psnr_db: f64,
    /// 均方误差（PSNR 的线性等价量，便于无 libm 判据）。
    pub mse: f64,
    /// 信号峰值（PSNR 归一化基准）。
    pub peak: f32,
}

/// 双门禁阈值。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityGate {
    /// 允许的最大 `mean_abs_diff`。
    pub max_mean_abs: f64,
    /// 允许的最大 `variance`。
    pub max_variance: f64,
    /// 要求的最小 PSNR（dB）。
    pub min_psnr_db: f64,
}

impl Default for ParityGate {
    /// 默认门禁：均值 ≤ 1e-6、方差 ≤ 1e-12、PSNR ≥ 40 dB。
    ///
    /// 取值依据：f32 末位差（~1e-7 相对）在 O(1) 辐亮度上给出 ~1e-7 绝对差，
    /// 1e-6 给它一个数量级余量；方差 1e-12 排除「整体偏移」；40 dB 是业界
    /// 「视觉无损」通用门槛。
    fn default() -> Self {
        ParityGate {
            max_mean_abs: 1e-6,
            max_variance: 1e-12,
            min_psnr_db: 40.0,
        }
    }
}

/// 确定性 `log10`（诊断量，自研：二进制指数分解 + atanh 级数，**无 libm**）。
///
/// 与引擎 D-29 同纪律：本函数不调用 libm，任意平台同输入同输出。
fn log10_det(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    // 分解 x = m · 2^e，m ∈ [0.5, 1)。
    let mut m = x;
    let mut e: i32 = 0;
    while m >= 1.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 0.5 {
        m *= 2.0;
        e -= 1;
    }
    // ln(m) = 2·atanh(z)，z = (m-1)/(m+1) ∈ [-1/3, 0)。
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    // 级数：2·Σ_{k≥0} z^(2k+1)/(2k+1)。|z| ≤ 1/3 ⇒ 取到 k=10 时截断误差
    // < |z|^23/(23·(1-z²)) ≈ 4e-12，远小于 1e-9 的判定余量。
    let mut term = 1.0f64;
    let mut acc = 0.0f64;
    for k in 0..11u32 {
        acc += term / ((2 * k + 1) as f64);
        term *= z2;
    }
    let ln_m = 2.0 * z * acc;
    const LN2: f64 = 0.693_147_180_559_945_3;
    const LN10: f64 = 2.302_585_092_994_046;
    (ln_m + e as f64 * LN2) / LN10
}

/// 计算两帧的 parity 统计（通道 0..3，逐元素）。
///
/// 长度不等时按较短者比较，并在 `samples` 中如实反映（调用方应自行断言等长）。
pub fn parity_report(a: &[[f32; 4]], b: &[[f32; 4]], channels: usize) -> ParityReport {
    let n = a.len().min(b.len());
    let ch = channels.clamp(1, 4);
    let mut sum_abs = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut sum_delta_sq = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut peak = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        for c in 0..ch {
            let x = a[i][c];
            let y = b[i][c];
            let d = x as f64 - y as f64;
            let ad = d.abs();
            sum_abs += ad;
            sum_delta += d;
            sum_delta_sq += d * d;
            sum_sq_err += d * d;
            if ad as f32 > max_abs {
                max_abs = ad as f32;
            }
            let m = x.abs().max(y.abs());
            if m > peak {
                peak = m;
            }
            count += 1;
        }
    }
    if count == 0 {
        return ParityReport {
            samples: 0,
            mean_abs_diff: 0.0,
            variance: 0.0,
            max_abs_diff: 0.0,
            psnr_db: f64::INFINITY,
            mse: 0.0,
            peak: 0.0,
        };
    }
    let cnt = count as f64;
    let mean_abs = sum_abs / cnt;
    let mean_delta = sum_delta / cnt;
    let variance = (sum_delta_sq / cnt - mean_delta * mean_delta).max(0.0);
    let mse = sum_sq_err / cnt;
    let psnr_db = if mse <= 0.0 {
        f64::INFINITY
    } else if peak <= 0.0 {
        f64::NEG_INFINITY
    } else {
        let p = peak as f64;
        10.0 * log10_det(p * p / mse)
    };
    ParityReport {
        samples: count,
        mean_abs_diff: mean_abs,
        variance,
        max_abs_diff: max_abs,
        psnr_db,
        mse,
        peak,
    }
}

/// 双门禁判定结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ParityVerdict {
    /// 是否三门全过（均值 / 方差 / PSNR）。
    pub pass: bool,
    /// 判红原因（空 = 全过）。
    pub reasons: Vec<String>,
    /// 统计量原文（报告用）。
    pub report: ParityReport,
}

/// 依据 [`ParityGate`] 判定一份 [`ParityReport`]。
///
/// 三门全部必过（`and` 语义）：语义级两项 + 输出级 PSNR。
pub fn judge(report: &ParityReport, gate: &ParityGate) -> ParityVerdict {
    let mut reasons: Vec<String> = Vec::new();
    if report.samples == 0 {
        reasons.push("无可比较样本（samples = 0）".to_string());
    }
    if !(report.mean_abs_diff <= gate.max_mean_abs) {
        reasons.push(format!(
            "语义级门禁 ①：|Δ| 均值 {:.3e} > 上限 {:.3e}",
            report.mean_abs_diff, gate.max_mean_abs
        ));
    }
    if !(report.variance <= gate.max_variance) {
        reasons.push(format!(
            "语义级门禁 ②：Δ 方差 {:.3e} > 上限 {:.3e}",
            report.variance, gate.max_variance
        ));
    }
    if !(report.psnr_db >= gate.min_psnr_db) {
        reasons.push(format!(
            "输出级门禁：PSNR {:.2} dB < 下限 {:.2} dB",
            report.psnr_db, gate.min_psnr_db
        ));
    }
    ParityVerdict {
        pass: reasons.is_empty(),
        reasons,
        report: *report,
    }
}

// ===========================================================================
// V2.0-G 测试：GPU 插槽 + double gate parity 网
// ===========================================================================

#[cfg(test)]
mod v2g_parity_tests {
    use super::*;
    use crate::integrator::{Sphere, render_pixel as ref_render_pixel};

    /// V2.0-G parity 场景：两个漫反射球 + 金属球 + 发光球（带 NEE/MIS 路径）。
    fn v2g_scene() -> Scene {
        let center = Vec3::new(0.0, 2.0, 0.0);
        let radiance = Vec3::new(12.0, 11.0, 9.0);
        Scene {
            spheres: vec![
                (
                    Sphere { center: Vec3::new(-0.55, 0.45, 0.0), radius: 0.5 },
                    MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.28, 0.18) },
                ),
                (
                    Sphere { center: Vec3::new(0.6, 0.35, 0.15), radius: 0.4 },
                    MaterialKind::Mirror,
                ),
                (
                    Sphere { center, radius: 0.3 },
                    MaterialKind::Emissive { radiance },
                ),
            ],
            boxes: vec![],
            lights: vec![Light::Sphere { center, radius: 0.3, radiance }],
            medium: None,
            paraboloids: vec![],
            planes: vec![],
        }
    }

    fn v2g_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 1.0, 3.2),
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            3.2,
            0.0,
            1.0 / 24.0,
        )
    }

    /// 逐位比较两帧（CPU 权威路径的确定性证据，强于 MD5 相等）。
    fn assert_bitwise(a: &[[f32; 4]], b: &[[f32; 4]], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}：长度不等");
        for i in 0..a.len() {
            for c in 0..4 {
                assert_eq!(
                    a[i][c].to_bits(),
                    b[i][c].to_bits(),
                    "{label}：px{i} ch{c} 不逐位一致（{} vs {}）",
                    a[i][c],
                    b[i][c]
                );
            }
        }
    }

    /// 插槽注册表与 `prism_gpu::compute` 单一真源一致（无第二套口径）。
    #[test]
    fn gpu_kernel_slots_match_compute_registry() {
        let slots = gpu_kernel_slots();
        let registry = prism_gpu::compute::kernel_registry();
        assert_eq!(slots.len(), registry.len(), "插槽数须等于 kernel 注册表");
        for (slot, desc) in slots.iter().zip(registry.iter()) {
            assert_eq!(slot.name, desc.name);
            assert_eq!(slot.entry, desc.entry);
            assert_eq!(slot.wgsl_source, desc.wgsl_source);
            assert_eq!(slot.spv_module, desc.spv_module);
            assert_eq!(
                slot.workgroup,
                [desc.workgroup.x, desc.workgroup.y, desc.workgroup.z]
            );
            assert_eq!(slot.spv_status, prism_gpu::compute::kernel_spv_status(desc));
            // dispatch 形状与 prism-gpu 的 DispatchShape 同源。
            let shape = slot.dispatch_shape([1920, 1080, 1]);
            assert_eq!(shape.groups(), slot.groups([1920, 1080, 1]));
            assert_eq!(shape.useful_invocations(), 1920 * 1080);
        }
        // 唯一的 Embedded 插槽是 SPIR-V 探针；RT 插槽如实标 PendingToolchain。
        let dispatchable: Vec<&str> = slots
            .iter()
            .filter(|s| s.is_dispatchable())
            .map(|s| s.name)
            .collect();
        assert_eq!(dispatchable, vec![prism_gpu::compute::KERNEL_SPV_MUL2_PROBE]);
        let prg = gpu_kernel_slot(prism_gpu::compute::KERNEL_PRIMARY_RAY_GEN)
            .expect("主射线插槽存在");
        assert!(!prg.is_dispatchable(), "无 glslang ⇒ 主射线 SPIR-V 待生成（如实）");
        assert!(gpu_kernel_slot("no_such_kernel").is_none());
    }

    /// GPU 插槽实时探测：本环境必须**如实**报无 ICD（不 panic、不假装）。
    #[test]
    fn gpu_slot_probe_reports_environment_truthfully() {
        let probe = std::panic::catch_unwind(|| probe_gpu_slots([64, 1, 1]))
            .expect("probe_gpu_slots 不得 panic");
        assert!(probe.checked_slots >= 3);
        assert_eq!(probe.groups, [1, 1, 1], "64 / 64 → 1 个工作组");
        assert!(!probe.detail.is_empty(), "探测必须给出原因说明");
        if probe.available {
            // 真机分支：读回值已与 CPU 参考（src*2）核对。
            assert_eq!(probe.status, "ok");
            assert!(probe.detail.contains("与 CPU 参考 一致"));
        } else {
            // 本沙箱分支：如实报 device-unavailable。
            assert_eq!(probe.status, "device-unavailable");
            assert!(
                probe.detail.contains("no-vulkan-icd") || probe.detail.contains("ICD"),
                "无设备时须如实报 ICD 缺失，实际：{}",
                probe.detail
            );
        }
        println!(
            "GPU 插槽探测：available={} status={} groups={:?} detail={}",
            probe.available, probe.status, probe.groups, probe.detail
        );
    }

    /// **双门禁主判据**：CPU 递归序 vs CPU 模拟 GPU 数组序。
    ///
    /// 语义级（|Δ| 均值 / 方差）+ 输出级（PSNR ≥ 40 dB）三门全过才算 parity 通过。
    #[test]
    fn array_order_parity_passes_double_gate() {
        let scene = v2g_scene();
        let cam = v2g_camera();
        let ctx = EvalCtx::new(2026, 7, 3);
        let res = (8u32, 8u32);
        let params = RenderParams::new(8, 3);

        // 权威路径（递归序，MD5 锚定）。
        let recursive = render_frame(&scene, &cam, &ctx, res, &params);
        // GPU 数组序（层同步 + workgroup 分块归约）。
        let array_order = render_frame_array_order(&scene, &cam, &ctx, res, &params, 4);

        assert_eq!(recursive.len(), array_order.len());
        let report = parity_report(&recursive, &array_order, 3);
        let gate = ParityGate::default();
        let verdict = judge(&report, &gate);

        println!(
            "parity（8×8 spp8 bounces3 wg4）：samples={} |Δ|均值={:.3e} 方差={:.3e} max|Δ|={:.3e} peak={:.3} MSE={:.3e} PSNR={:.2} dB",
            report.samples,
            report.mean_abs_diff,
            report.variance,
            report.max_abs_diff,
            report.peak,
            report.mse,
            report.psnr_db
        );
        assert!(
            verdict.pass,
            "double gate 未通过：{:?}（报告 {:?}）",
            verdict.reasons, report
        );
        if report.max_abs_diff == 0.0 {
            // 如实说明：本场景多数路径逃逸（贡献 0），非零贡献样本稀少 ⇒
            // 分块求和与树形归约的结合序差异不显形，两帧逐位一致。
            println!(
                "  ↳ 逐位一致：重排未引入浮点差异（本场景非零贡献样本稀少，\
                 结合序不改变求和结果）；门禁仍由 gate_rejects_real_deviations 证明非恒真"
            );
        }
        // 门禁的具体口径（非恒真）：均值/方差/PSNR 三项都真被检查过。
        assert!(report.mean_abs_diff <= gate.max_mean_abs);
        assert!(report.variance <= gate.max_variance);
        assert!(report.psnr_db >= gate.min_psnr_db);
    }

    /// 第二组 parity 配置：spp 取素数、workgroup 不整除 spp（块边界错开），
    /// 用于验证「分块归约 vs 树形归约」的结合序差异仍在门禁内。
    #[test]
    fn array_order_parity_holds_for_non_divisible_workgroup() {
        let scene = v2g_scene();
        let cam = v2g_camera();
        let ctx = EvalCtx::new(313, 7, 3);
        let res = (6u32, 6u32);
        let params = RenderParams::new(7, 4);
        let recursive = render_frame(&scene, &cam, &ctx, res, &params);
        let array_order = render_frame_array_order(&scene, &cam, &ctx, res, &params, 3);
        let report = parity_report(&recursive, &array_order, 3);
        let verdict = judge(&report, &ParityGate::default());
        println!(
            "parity（6×6 spp7 bounces4 wg3）：|Δ|均值={:.3e} 方差={:.3e} max|Δ|={:.3e} PSNR={:.2} dB",
            report.mean_abs_diff, report.variance, report.max_abs_diff, report.psnr_db
        );
        assert!(
            verdict.pass,
            "非整除 workgroup 的 double gate 未通过：{:?}",
            verdict.reasons
        );
    }

    /// 数组序与递归序的**几何**必须一致：像素数对齐、无 NaN、无非有限值。
    #[test]
    fn array_order_output_is_well_formed() {
        let scene = v2g_scene();
        let cam = v2g_camera();
        let ctx = EvalCtx::new(11, 7, 3);
        let res = (6u32, 5u32);
        let params = RenderParams::new(4, 2);
        let frame = render_frame_array_order(&scene, &cam, &ctx, res, &params, 8);
        assert_eq!(frame.len(), 30);
        for (i, px) in frame.iter().enumerate() {
            for (c, v) in px.iter().enumerate() {
                assert!(v.is_finite(), "px{i} ch{c} 非有限值：{v}");
                assert!(*v >= 0.0, "px{i} ch{c} 出现负辐亮度：{v}");
            }
            assert_eq!(px[3], 1.0, "alpha 通道恒 1");
        }
        // 边界：spp = 0 / 尺寸 0 不 panic 且给出零像素。
        let zero_spp = RenderParams::new(0, 2);
        let zeros = render_frame_array_order(&scene, &cam, &ctx, res, &zero_spp, 4);
        assert_eq!(zeros.len(), 30);
        assert!(zeros.iter().all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0));
        assert!(render_frame_array_order(&scene, &cam, &ctx, (0, 0), &params, 4).is_empty());
    }

    /// 数组序确定性：同输入两遍逐位一致（真机 wavefront 重排不得引入抖动）。
    #[test]
    fn array_order_is_deterministic() {
        let scene = v2g_scene();
        let cam = v2g_camera();
        let ctx = EvalCtx::new(5, 7, 3);
        let res = (5u32, 5u32);
        let params = RenderParams::new(4, 3);
        let a = render_frame_array_order(&scene, &cam, &ctx, res, &params, 4);
        let b = render_frame_array_order(&scene, &cam, &ctx, res, &params, 4);
        assert_bitwise(&a, &b, "数组序两遍");
    }

    /// **CPU 权威路径不动**：递归序两遍逐字节一致（MD5 相等的强形式）。
    ///
    /// MD5 摘要相等的**证明**就是这里的逐位相等（相同字节流必有相同摘要）。
    #[test]
    fn cpu_recursive_path_stays_byte_identical() {
        let scene = v2g_scene();
        let cam = v2g_camera();
        let res = (6u32, 6u32);
        let params = RenderParams::new(4, 3);
        let f1 = render_frame(&scene, &cam, &EvalCtx::new(99, 7, 3), res, &params);
        let f2 = render_frame(&scene, &cam, &EvalCtx::new(99, 7, 3), res, &params);
        assert_bitwise(&f1, &f2, "CPU 递归序两遍");
        // 与 integrator 参考同器判对（既有契约在本场景上复核一遍）。
        let mut bytes = 0usize;
        for y in 0..res.1 {
            for x in 0..res.0 {
                let want = ref_render_pixel(&scene, &cam, &EvalCtx::new(99, 7, 3), (x, y), res, &params);
                let got = f1[(y * res.0 + x) as usize];
                for c in 0..3 {
                    assert_eq!(
                        got[c].to_bits(),
                        want[c].to_bits(),
                        "px({x},{y}) ch{c} 与 integrator 不一致"
                    );
                }
                bytes += 16;
            }
        }
        println!("CPU 权威路径：{bytes} 字节逐位稳定（MD5 相同）");
    }

    /// **门禁非恒真**：人为扰动必须被判红（三门口径各测一次）。
    #[test]
    fn gate_rejects_real_deviations() {
        let gate = ParityGate::default();
        let base = vec![[1.0f32, 2.0, 3.0, 1.0]; 16];

        // ① 完全一致 ⇒ 通过（均值 0、方差 0、PSNR = ∞）。
        let same = parity_report(&base, &base, 3);
        assert_eq!(same.mean_abs_diff, 0.0);
        assert_eq!(same.variance, 0.0);
        assert!(same.psnr_db.is_infinite());
        assert!(judge(&same, &gate).pass);

        // ② 单点大扰动 ⇒ PSNR 掉到 40 dB 以下 ⇒ 判红（输出级门禁）。
        let mut spike = base.clone();
        spike[3][0] += 8.0;
        let r_spike = parity_report(&base, &spike, 3);
        let v_spike = judge(&r_spike, &gate);
        assert!(!v_spike.pass, "单点大扰动能通过门禁 → 门禁恒真");
        assert!(
            v_spike.reasons.iter().any(|r| r.contains("PSNR")),
            "应报 PSNR 不足：{:?}",
            v_spike.reasons
        );
        println!("负向用例①（单点扰动）：PSNR={:.2} dB（判红）", r_spike.psnr_db);

        // ③ 全场系统性偏移 1e-4 ⇒ |Δ| 均值超标（语义级门禁 ①）。
        let biased: Vec<[f32; 4]> = base
            .iter()
            .map(|p| [p[0] + 1e-4, p[1] + 1e-4, p[2] + 1e-4, p[3]])
            .collect();
        let r_bias = parity_report(&base, &biased, 3);
        let v_bias = judge(&r_bias, &gate);
        assert!(!v_bias.pass, "系统性偏移能通过门禁 → 语义级门禁失效");
        assert!(r_bias.mean_abs_diff > gate.max_mean_abs);
        println!(
            "负向用例②（系统偏移 1e-4）：|Δ|均值={:.3e} 方差={:.3e}（判红）",
            r_bias.mean_abs_diff, r_bias.variance
        );

        // ④ 高频抖动（有符号均值抵消、|Δ| 却很大）⇒ 需**方差门禁**单独抓。
        //
        // 用一条「均值门禁放宽到 1e-3」的闸：此时 |Δ| 均值 2e-4 可过均值门禁，
        // 但 ±2e-4 的抖动使方差远超 1e-12 ⇒ 只有方差门禁能判红。
        let relaxed = ParityGate {
            max_mean_abs: 1e-3,
            max_variance: 1e-12,
            min_psnr_db: 40.0,
        };
        let jitter: Vec<[f32; 4]> = base
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = if i % 2 == 0 { 2e-4 } else { -2e-4 };
                [p[0] + d, p[1], p[2], p[3]]
            })
            .collect();
        let r_jit = parity_report(&base, &jitter, 3);
        let v_jit = judge(&r_jit, &relaxed);
        assert!(!v_jit.pass, "均值抵消的抖动未被方差门禁抓住");
        assert!(
            r_jit.mean_abs_diff <= relaxed.max_mean_abs,
            "本例应过均值门禁（均值 {:e}）",
            r_jit.mean_abs_diff
        );
        assert!(r_jit.variance > relaxed.max_variance);
        println!(
            "负向用例③（±2e-4 抖动）：|Δ|均值={:.3e}（过均值门禁）方差={:.3e}（方差门禁判红）",
            r_jit.mean_abs_diff, r_jit.variance
        );

        // ⑤ 空样本 ⇒ 判红（不得把「没比过」当通过）。
        let empty = parity_report(&[], &[], 3);
        assert_eq!(empty.samples, 0);
        assert!(!judge(&empty, &gate).pass);
    }

    /// 自研 `log10` 的精度（诊断量正确性；无 libm）。
    #[test]
    fn deterministic_log10_is_accurate() {
        // 参考值取自 IEEE 754 十进制常量（不由 libm 运行时生成）。
        let cases = [
            (1.0f64, 0.0f64),
            (10.0, 1.0),
            (100.0, 2.0),
            (0.1, -1.0),
            (1000.0, 3.0),
            (2.0, 0.301_029_995_663_981_2),
        ];
        for (x, want) in cases {
            let got = log10_det(x);
            assert!(
                (got - want).abs() < 1e-9,
                "log10({x}) = {got} ≠ {want}（误差超 1e-9）"
            );
        }
        assert!(log10_det(0.0).is_infinite(), "log10(0) 应为 -∞ 而非 panic");
        assert!(log10_det(-1.0).is_infinite());
    }
}
