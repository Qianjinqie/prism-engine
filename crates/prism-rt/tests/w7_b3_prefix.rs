//! W7 / B3 判据 P3：**前缀**——预览的 N 个样本是终帧 M 个样本的前 N 个。
//!
//! # 这条判据在检验什么（与"同义反复"的区别）
//!
//! M4 原文的"预览帧与终帧在同一 spp 下逐位一致"是同义反复：同一函数同一参数
//! 当然是同一个调用。真正有内容的性质是**样本流可截断**：
//!
//! ```text
//! render_pixel_range(p, 0, N) ≡ mean( trace_sample(p, s) for s in 0..N )
//!                           ≡ 终帧 spp=M 那条流的前 N 项（N ≤ M）
//! ```
//!
//! 三条子断言：
//!
//! 1. `sample_stream_is_spp_independent`：第 `s` 个样本与 `params.spp` 无关
//!    ——这是"前缀"成立的前提（否则终帧换个 spp，前 N 个样本就变了）；
//! 2. `range_mean_matches_accumulator`：把 `M` 个单样本 pass 推进
//!    [`prism_rt::accum::AccumBuffer`]，其 `canonical_mean` 在 `N` 处等于
//!    `render_pixel_range(0, N)`——**逐位**（复用 `accum.rs` 已有的
//!    "push 8 个单样本 pass == 一次 spp=8" 方法学，这里是它的区间版）；
//! 3. `render_pixel_is_range_special_case`：`render_pixel(p, spp=M)` ≡
//!    `render_pixel_range(0, M)`——预览与终帧共用同一个出口。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::accum::AccumBuffer;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    render_pixel, render_pixel_range, trace_sample, Aabb, Light, MaterialKind, RenderParams, Scene,
    Sphere,
};

/// 一个"有话说"的场景：球 + 地板 + 区域光 + 中灰材质（既非全黑也非纯平）。
fn scene() -> Scene {
    Scene {
        spheres: vec![(
            Sphere { center: Vec3::new(0.0, 1.0, 0.0), radius: 1.0 },
            MaterialKind::Pbr(Bsdf::new(
                Vec3::new(0.75, 0.45, 0.35),
                0.2,
                0.35,
                1.5,
                0.0,
                0.0,
                Vec3::splat(0.3),
            )),
        )],
        boxes: vec![(
            Aabb::new(Vec3::new(-8.0, -0.01, -8.0), Vec3::new(8.0, 0.0, 8.0)),
            MaterialKind::Lambert { albedo: Vec3::splat(0.6) },
        )],
        lights: vec![Light::Directional {
            direction: Vec3::new(0.35, 1.0, 0.2).normalize(),
            radiance: Vec3::splat(5.0),
            cos_theta_max: 0.35,
        }],
        ..Default::default()
    }
}

fn cam() -> Camera {
    Camera::new(
        Vec3::new(0.0, 1.4, -5.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        45.0,
        0.0,
        5.0,
        0.0,
        0.5,
    )
}

/// 逐位比较四个通道。
#[track_caller]
fn assert_bits_eq(a: [f32; 4], b: [f32; 4], what: &str) {
    for c in 0..4 {
        assert_eq!(
            a[c].to_bits(),
            b[c].to_bits(),
            "{what}：通道 {c} 不逐位相等（{} vs {}）",
            a[c],
            b[c]
        );
    }
}

#[test]
fn sample_stream_is_spp_independent() {
    let scene = scene();
    let camera = cam();
    let ctx = EvalCtx::new(0, 0, 0);
    let res = (24u32, 16u32);
    let pixel = (11u32, 6u32);
    let p1 = RenderParams::new(1, 5);
    let p16 = RenderParams::new(16, 5);
    let p64 = RenderParams::new(64, 5);
    for s in 0..16u32 {
        let a = trace_sample(&scene, &camera, &ctx, res, pixel, s, &p1);
        let b = trace_sample(&scene, &camera, &ctx, res, pixel, s, &p16);
        let c = trace_sample(&scene, &camera, &ctx, res, pixel, s, &p64);
        assert_bits_eq(a, b, &format!("spp=1 vs spp=16 的第 {s} 个样本"));
        assert_bits_eq(a, c, &format!("spp=1 vs spp=64 的第 {s} 个样本"));
    }
    println!("P3-a 样本流与 spp 无关：16 个样本 × 3 档 spp 逐位相同");
}

#[test]
fn range_mean_matches_accumulator() {
    let scene = scene();
    let camera = cam();
    let ctx = EvalCtx::new(0, 0, 0);
    let res = (24u32, 16u32);
    let pixel = (11u32, 6u32);
    let m = 16u32;
    let params = RenderParams::new(m, 5);

    // 终帧的 M 个样本，逐个推进累计缓冲（方法学同 `accum.rs` 的
    // "push 8 个单样本 pass == 一次 spp=8"；这里关心的是**区间**等价）。
    let mut buf = AccumBuffer::new(1, 1);
    let per_sample: Vec<[f32; 4]> = (0..m)
        .map(|s| trace_sample(&scene, &camera, &ctx, res, pixel, s, &params))
        .collect();
    for v in &per_sample {
        buf.push_pass(std::slice::from_ref(v));
    }

    for n in [1u32, 2, 4, 8, 16] {
        let want = buf.canonical_mean(0, 0); // 注意：下面按 n 截断后重算
        let _ = want;
        // 用只含前 n 个 pass 的缓冲（"预览在第 n 个样本处停下"）。
        let mut part = AccumBuffer::new(1, 1);
        for v in per_sample.iter().take(n as usize) {
            part.push_pass(std::slice::from_ref(v));
        }
        let accumulated = part.canonical_mean(0, 0);
        let ranged = render_pixel_range(&scene, &camera, &ctx, pixel, res, &params, 0, n);
        assert_bits_eq(ranged, accumulated, &format!("N={n}：区间均值 vs 累计均值"));
        // 前缀性：前 n 个样本的均值必须只由这 n 个样本决定 —— 与 M 无关。
        let p_n = RenderParams::new(n, 5);
        let direct = render_pixel(&scene, &camera, &ctx, pixel, res, &p_n);
        assert_bits_eq(ranged, direct, &format!("N={n}：区间均值 vs render_pixel(spp={n})"));
    }
    println!("P3-b 区间均值 == 累计缓冲前 N 个 pass 的规范均值（N=1/2/4/8/16，逐位）");
}

#[test]
fn render_pixel_is_range_special_case() {
    let scene = scene();
    let camera = cam();
    let ctx = EvalCtx::new(3, 1, 2);
    let res = (20u32, 15u32);
    for spp in [1u32, 3, 8] {
        let params = RenderParams::new(spp, 4);
        for pixel in [(3u32, 4u32), (10, 7), (19, 14)] {
            let a = render_pixel(&scene, &camera, &ctx, pixel, res, &params);
            let b = render_pixel_range(&scene, &camera, &ctx, pixel, res, &params, 0, spp);
            assert_bits_eq(a, b, &format!("spp={spp} pixel={pixel:?}：render_pixel vs range(0,spp)"));
        }
    }
    // 空区间 = 单位元（不 panic）。非空子区间必须等于用同样本数的独立调用。
    let params = RenderParams::new(8, 4);
    let empty = render_pixel_range(&scene, &camera, &ctx, (5, 5), res, &params, 4, 4);
    assert_eq!(empty, [0.0f32; 4], "空区间必须返回单位元");
    let mid = render_pixel_range(&scene, &camera, &ctx, (5, 5), res, &params, 3, 5);
    let p2 = RenderParams::new(2, 4);
    let seq = {
        // [3,5) 的均值 == 单个渲染的 2 个样本（用 s=3,4 —— 由 trace_sample 直接取）。
        let a = trace_sample(&scene, &camera, &ctx, res, (5, 5), 3, &p2);
        let b = trace_sample(&scene, &camera, &ctx, res, (5, 5), 4, &p2);
        prism_rt::accum::mean_samples(&[a, b])
    };
    assert_bits_eq(mid, seq, "子区间 [3,5) 的均值");
    println!("P3-c render_pixel ≡ render_pixel_range(0, spp)，且子区间/空区间语义正确");
}
