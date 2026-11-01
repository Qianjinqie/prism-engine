//! `rt_microbench` —— `prism-rt` 各原语的单位成本（W0-C L2 的量尺）。
//!
//! 为什么需要它：本机没有 `perf` / `valgrind`，而 `rdtsc` 插桩在
//! `lfence` 串行化下会把流水线抽干（实测探针合计比墙钟多 64%），段间比例
//! 不可信。故 L2 改为**两件互不污染的事**：
//!
//! 1. **计数**（插桩，只数事件不取时间）：每样本弹射数、求交数、BSDF 数；
//! 2. **单位成本**（本文件，大 N 紧循环）：每个原语多少纳秒。
//!
//! 两者相乘即得"单样本 322 ns 花在哪几段"，且计时开销被大 N 摊成 0。
//!
//! ```text
//! cargo run --release -p prism-rt --example rt_microbench
//! ```

use std::hint::black_box;
use std::time::Instant;

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::accum::mean_samples;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{MaterialKind, RenderParams, Scene, Sphere};
use prism_rt::sampler::{sample_2d, sobol_u32, SampleStream};

/// 紧循环计时：返回每次调用的纳秒。
fn ns_per_call<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    // 热身：让 i-cache / 分支预测器进稳态。
    for _ in 0..(iters / 16).max(1) {
        f();
    }
    let t0: Instant = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dt: f64 = t0.elapsed().as_secs_f64();
    black_box(());
    dt * 1e9 / iters as f64
}

fn main() {
    let iters: u32 = 4_000_000;
    let ctx: EvalCtx = EvalCtx::new(0, 0, 0);
    let cam: Camera = Camera::new(
        Vec3::new(0.0, 0.0, -4.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        0.0,
        3.0,
        0.0,
        0.5,
    );
    let res: (u32, u32) = (1920, 1080);

    // 场景同 `rt_backend::tests::red_sphere_request`（单球 + 方向光）。
    let mut scene: Scene = Scene::default();
    scene.spheres.push((
        Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
        MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.8, 0.2, 0.2),
            0.05,
            0.35,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        )),
    ));

    let bsdf: Bsdf = Bsdf::new(
        Vec3::new(0.8, 0.2, 0.2),
        0.05,
        0.35,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    );
    let n: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    let wo: Vec3 = Vec3::new(0.0, 0.0, 1.0).normalize();
    let wi: Vec3 = Vec3::new(0.36, 0.48, 0.8).normalize();

    // ---- 采样器 ----
    let t_sobol: f64 = ns_per_call(iters, || {
        black_box(sobol_u32(black_box(7), black_box(0x1234_5678)));
    });
    let t_s2d: f64 = ns_per_call(iters, || {
        black_box(sample_2d(
            black_box(&ctx),
            &prism_core::SampleIndex {
                frame: 0,
                pixel: (960, 540),
                sample: 3,
                dimension: 7,
                node_id: 0,
            },
        ));
    });
    let stream: SampleStream<'_> = SampleStream::new(&ctx, (960, 540), 3);
    let t_s2d_stream: f64 = ns_per_call(iters, || {
        black_box(black_box(&stream).sample_2d(black_box(7)));
    });
    let t_ray: f64 = ns_per_call(iters / 4, || {
        black_box(cam.cast_ray(black_box(&stream), res));
    });

    // ---- 求交（主射线 + 阴影射线两种几何）----
    let t_primary: f64 = ns_per_call(iters, || {
        black_box(scene.primary_hit_distance(
            black_box(Vec3::new(0.02, 0.01, -4.0)),
            black_box(Vec3::new(0.0, 0.0, 1.0)),
            1.0e30,
        ));
    });
    // 命中面的阴影射线（起点在球面上，射向光源，几乎必逃逸）。
    let t_shadow: f64 = ns_per_call(iters, || {
        black_box(scene.primary_hit_distance(
            black_box(Vec3::new(0.0, 0.0, -1.0)),
            black_box(Vec3::new(0.3, 0.4, 0.866).normalize()),
            1.0e30,
        ));
    });

    // 空场景：隔离 `intersect` 的固定开销（族循环 + 曲线判活）。
    let empty: Scene = Scene::default();
    let t_empty: f64 = ns_per_call(iters, || {
        black_box(empty.primary_hit_distance(
            black_box(Vec3::new(0.02, 0.01, -4.0)),
            black_box(Vec3::new(0.0, 0.0, 1.0)),
            1.0e30,
        ));
    });

    // ---- BSDF ----
    let t_eval: f64 = ns_per_call(iters, || {
        black_box(bsdf.eval(black_box(wo), black_box(wi), black_box(n)));
    });
    let t_sample: f64 = ns_per_call(iters, || {
        black_box(bsdf.sample(
            black_box(wo),
            black_box(n),
            false,
            &prism_rt::bsdf::SampleCtx { psr_rough: 0.0 },
            black_box(0.3),
            black_box(0.6),
            black_box(0.7),
        ));
    });

    // ---- 归约 ----
    let params: RenderParams = RenderParams::new(16, 6);
    let mut s16: Vec<[f32; 4]> = Vec::with_capacity(params.spp as usize);
    for i in 0..params.spp {
        s16.push([i as f32 * 0.01, 0.5, 0.25, 1.0]);
    }
    let t_mean: f64 = ns_per_call(iters / 4, || {
        black_box(mean_samples(black_box(&s16)));
    });
    let t_alloc: f64 = ns_per_call(iters / 4, || {
        let v: Vec<[f32; 4]> = Vec::with_capacity(16);
        black_box(v.capacity());
    });

    let rows: [(&str, f64, &str); 10] = [
        ("sobol_u32（一次 Sobol 坐标）", t_sobol, "次"),
        ("sample_2d（自由函数）", t_s2d, "次"),
        ("SampleStream::sample_2d", t_s2d_stream, "次"),
        ("Camera::cast_ray（针孔）", t_ray, "条"),
        ("Scene 求交（主射线，命中）", t_primary, "条"),
        ("Scene 求交（阴影射线，逃逸）", t_shadow, "条"),
        ("Scene 求交（空场景=固定开销）", t_empty, "条"),
        ("Bsdf::eval（Disney 全式）", t_eval, "次"),
        ("Bsdf::sample（三瓣 + VNDF）", t_sample, "次"),
        ("mean_samples（spp=16）", t_mean, "像素"),
    ];
    println!("---- prism-rt 原语单位成本（{} 次紧循环）----", iters);
    for (name, ns, unit) in rows {
        println!("{name:<30} {:>9.2} ns/{unit}", ns);
    }
    println!("{:<30} {:>9.2} ns/次", "Vec::with_capacity(16)", t_alloc);

    // ---- 合成：单样本各段预算（计数来自 L2 插桩，见报告 §14.2）----
    //
    // 计数口径（该场景 8.8% 主射线命中球）：
    //   每样本 1 条主射线、1.088 次求交（0.088 条阴影射线）、
    //   0.088 次 BSDF eval+pdf、0.088 次 BSDF sample、
    //   5.26 次 sample_2d（1 像素抖动 + 1 快门 + 3×1.088）、
    //   1/16 次 mean_samples。
    let per_sample: f64 = 1.0 * t_ray
        + 1.0 * t_primary
        + 0.088 * t_shadow
        + 0.088 * t_eval
        + 0.088 * t_sample
        + 4.26 * t_s2d_stream
        + t_mean / 16.0;
    println!("\n---- 合成预算（ns/样本）----");
    println!("cast_ray          {:>8.2}", t_ray);
    println!("求交（主）        {:>8.2}", t_primary);
    println!("求交（阴影）      {:>8.2}", 0.088 * t_shadow);
    println!("BSDF eval         {:>8.2}", 0.088 * t_eval);
    println!("BSDF sample       {:>8.2}", 0.088 * t_sample);
    println!("sample_2d ×4.26   {:>8.2}", 4.26 * t_s2d_stream);
    println!("归约              {:>8.2}", t_mean / 16.0);
    println!("合计              {:>8.2}", per_sample);
}
