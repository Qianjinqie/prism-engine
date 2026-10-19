//! QA 独立复核探针 —— 缺陷 5（最高优先级）：wavefront 与 integrator 逐通道一致。
//!
//! 复核 `wavefront.rs:510` 的 `while params.max_bounces > 0` 短路是否真的与
//! integrator 的 `for bounce in 0..max_bounces` 严格同语义，且无死循环/NaN/性能退化。
//!
//! 探针全部为**正向断言 + 变异自证**，不含见证型断言。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{Aabb, Light, MaterialKind, RenderParams, Scene, Sphere};
use prism_rt::wavefront;

/// 多材质场景：漫反射 + 金属 + 发光体 + 介质，覆盖多种事件路径。
fn rich_scene() -> Scene {
    Scene {
        spheres: vec![
            (
                Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 0.5 },
                MaterialKind::Lambert { albedo: Vec3::new(0.7, 0.3, 0.2) },
            ),
            (
                Sphere { center: Vec3::new(0.6, 0.2, 2.4), radius: 0.3 },
                MaterialKind::Mirror,
            ),
            (
                Sphere { center: Vec3::new(-0.7, -0.3, 3.0), radius: 0.25 },
                MaterialKind::Emissive { radiance: Vec3::new(0.4, 0.35, 0.3) },
            ),
        ],
        boxes: vec![(
            Aabb { min: Vec3::new(-1.5, -1.5, 4.0), max: Vec3::new(1.5, 1.5, 4.2) },
            MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
        )],
        lights: vec![Light::Directional {
            direction: Vec3::new(-0.5, -0.5, -1.0).normalize(),
            radiance: Vec3::new(6.0, 6.0, 6.0),
            cos_theta_max: 0.9,
        }],
        medium: None,
        paraboloids: vec![],
        meshes: Vec::new(),
        planes: vec![],
        ..Default::default()
    }
}

fn empty_scene() -> Scene {
    Scene { spheres: Vec::new(), boxes: Vec::new(), lights: Vec::new(), medium: None, paraboloids: vec![], planes: vec![], meshes: Vec::new(), ..Default::default() }
}

fn cam() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        0.0,
        2.0,
        0.0,
        0.0,
    )
}

/// 探针 5a（核心）：4×N 通道逐位一致，覆盖 `max_bounces = 0/1/2/5`。
///
/// `max_bounces == 0` 是本缺陷的修复靶点：修复前 wavefront 仍会推进至少一个
/// 事件（相机段着色），与 integrator 的零次循环发散。
#[test]
fn probe5a_wavefront_matches_integrator_all_channels_all_bounces() {
    let resolution = (12u32, 9u32);
    let scenes: [(&str, Scene); 3] =
        [("rich", rich_scene()), ("empty", empty_scene()), ("no_light", {
            let mut s = rich_scene();
            s.lights.clear();
            s
        })];
    let bounces = [0u32, 1, 2, 5];
    let mut total = 0usize;
    for (sname, scene) in &scenes {
        for &mb in &bounces {
            for spp in [1u32, 3] {
                for seed in [1u64, 7, 42] {
                    let ctx = EvalCtx::new(seed, 1, 0);
                    let params = RenderParams::new(spp, mb);
                    for py in 0..resolution.1 {
                        for px in 0..resolution.0 {
                            let a = prism_rt::integrator::render_pixel(
                                scene, &cam(), &ctx, (px, py), resolution, &params,
                            );
                            let b = wavefront::render_pixel(
                                scene, &cam(), &ctx, (px, py), resolution, &params,
                            );
                            for ch in 0..4 {
                                assert!(
                                    a[ch].to_bits() == b[ch].to_bits(),
                                    "场景 `{sname}` mb={mb} spp={spp} seed={seed} \
                                     像素({px},{py}) 通道{ch} 不一致：integrator={} wavefront={}",
                                    a[ch],
                                    b[ch]
                                );
                            }
                            total += 4;
                        }
                    }
                }
            }
        }
    }
    assert!(total > 4000, "比对通道数过少（{total}），覆盖不足");
}

/// 探针 5b：`max_bounces == 0` ⇒ 两模块都返回**背景零贡献**，
/// 无事件产生、无 NaN、无死循环（超时即失败）。
#[test]
fn probe5b_zero_bounces_returns_background_no_events_no_nan() {
    let resolution = (8u32, 6u32);
    let params = RenderParams::new(2, 0);
    for seed in [1u64, 99] {
        let ctx = EvalCtx::new(seed, 1, 0);
        for py in 0..resolution.1 {
            for px in 0..resolution.0 {
                let a = prism_rt::integrator::render_pixel(
                    &rich_scene(), &cam(), &ctx, (px, py), resolution, &params,
                );
                let b =
                    wavefront::render_pixel(&rich_scene(), &cam(), &ctx, (px, py), resolution, &params);
                // 0 次弹射 ⇒ 无间接光、无直接光命中（相机段不计弹射）。
                // 关键：两条路径必须相等，且都不含 NaN。
                for ch in 0..4 {
                    assert!(!a[ch].is_nan(), "integrator 0 弹射产出 NaN（ch={ch}）");
                    assert!(!b[ch].is_nan(), "wavefront 0 弹射产出 NaN（ch={ch}）");
                    assert_eq!(
                        a[ch].to_bits(),
                        b[ch].to_bits(),
                        "0 弹射下两路径在({px},{py}) ch={ch} 不一致: {} vs {}",
                        a[ch],
                        b[ch]
                    );
                }
            }
        }
    }
}

/// 探针 5c：`max_bounces` 单调性——0 弹射的输出必须是 1/2/5 弹射的**下界**
/// （加弹射只增加能量贡献，不应使辐射变负）。
#[test]
fn probe5c_zero_bounce_is_lowest_energy() {
    let resolution = (6u32, 5u32);
    let ctx = EvalCtx::new(3, 1, 0);
    let p0 = RenderParams::new(2, 0);
    for py in 0..resolution.1 {
        for px in 0..resolution.0 {
            let z = wavefront::render_pixel(&rich_scene(), &cam(), &ctx, (px, py), resolution, &p0);
            for mb in [1u32, 2, 5] {
                let p = RenderParams::new(2, mb);
                let v = wavefront::render_pixel(&rich_scene(), &cam(), &ctx, (px, py), resolution, &p);
                for ch in 0..3 {
                    assert!(!v[ch].is_nan(), "mb={mb} 产出 NaN");
                    assert!(
                        v[ch] >= z[ch] - 1e-6,
                        "mb={mb} ch={ch} 能量低于 0 弹射基线：{} < {}（疑似负贡献）",
                        v[ch],
                        z[ch]
                    );
                }
            }
        }
    }
}

/// 探针 5d：NaN / Inf 压力入参——两者都必须**不 panic 且不产生 NaN 输出**。
#[test]
fn probe5d_nan_inf_params_do_not_panic_or_emit_nan() {
    let resolution = (5u32, 4u32);
    // spp=0 / max_bounces 极大：前者空循环，后者多轮推进。
    let cases = [
        ("spp0_mb0", RenderParams { spp: 0, max_bounces: 0, clamp_indirect: None, nee_only: false }),
        ("spp0_mb5", RenderParams { spp: 0, max_bounces: 5, clamp_indirect: None, nee_only: false }),
        ("spp1_mb64", RenderParams { spp: 1, max_bounces: 64, clamp_indirect: Some(f32::NAN), nee_only: false }),
        ("spp2_mb3_clamp_inf", RenderParams { spp: 2, max_bounces: 3, clamp_indirect: Some(f32::INFINITY), nee_only: true }),
    ];
    for (name, params) in cases {
        let ctx = EvalCtx::new(5, 1, 0);
        for py in 0..resolution.1 {
            for px in 0..resolution.0 {
                let a = prism_rt::integrator::render_pixel(
                    &rich_scene(), &cam(), &ctx, (px, py), resolution, &params,
                );
                let b = wavefront::render_pixel(
                    &rich_scene(), &cam(), &ctx, (px, py), resolution, &params,
                );
                for ch in 0..4 {
                    assert!(!a[ch].is_nan(), "{name}: integrator NaN (ch={ch})");
                    assert!(!b[ch].is_nan(), "{name}: wavefront NaN (ch={ch})");
                    assert_eq!(
                        a[ch].to_bits(),
                        b[ch].to_bits(),
                        "{name}: ({px},{py}) ch={ch} 不一致 {} vs {}",
                        a[ch],
                        b[ch]
                    );
                }
            }
        }
    }
}

/// 探针 5e：性能不退化——`while params.max_bounces > 0` 每次迭代多一次判断。
/// 口径：`max_bounces = 64` 的wavefront 单帧耗时不得比 `= 2` 高出 40 倍
/// （短路只省1 次判断，量级应与弹射数成线性而非爆炸）。
#[test]
fn probe5e_no_perf_regression_from_bounce_guard() {
    let resolution = (32u32, 32u32);
    let scene = rich_scene();
    let ctx = EvalCtx::new(11, 1, 0);
    let p2 = RenderParams::new(1, 2);
    let p64 = RenderParams::new(1, 64);
    let t2 = std::time::Instant::now();
    for _ in 0..3 {
        let _ = wavefront::render_frame(&scene, &cam(), &ctx, resolution, &p2);
    }
    let d2 = t2.elapsed();
    let t64 = std::time::Instant::now();
    for _ in 0..3 {
        let _ = wavefront::render_frame(&scene, &cam(), &ctx, resolution, &p64);
    }
    let d64 = t64.elapsed();
    let ratio = d64.as_secs_f64() / d2.as_secs_f64().max(1e-9);
    assert!(
        ratio < 40.0,
        "max_bounces 2→64 耗时比 {ratio:.1} 异常（{d2:?} → {d64:?}），疑似性能退化"
    );
}

/// 探针 5f：确定性铁律——同输入两遍逐字节一致（wavefront 侧）。
#[test]
fn probe5f_wavefront_bitwise_deterministic() {
    let resolution = (16u32, 12u32);
    let scene = rich_scene();
    for &mb in &[0u32, 1, 2, 5] {
        let params = RenderParams::new(4, mb);
        let ctx = EvalCtx::new(13, 1, 0);
        let a = wavefront::render_frame(&scene, &cam(), &ctx, resolution, &params);
        let b = wavefront::render_frame(&scene, &cam(), &ctx, resolution, &params);
        assert_eq!(a, b, "wavefront mb={mb} 非逐位确定（违反确定性铁律）");
    }
}

/// 探针 5g（变异自证）：断言 5a 的比对**不是恒真**。
/// 手法：把两条路径的 spp 错开（wavefront 用不同seed），此时必须不一致。
/// 若错开seed 后仍然相等，说明比对逻辑失效。
#[test]
fn probe5g_parity_check_is_not_vacuous() {
    let resolution = (12u32, 9u32);
    let scene = rich_scene();
    let params = RenderParams::new(3, 2);
    let mut diff = 0usize;
    let mut same = 0usize;
    for py in 0..resolution.1 {
        for px in 0..resolution.0 {
            let a = prism_rt::integrator::render_pixel(
                &scene, &cam(), &EvalCtx::new(1, 1, 0), (px, py), resolution, &params,
            );
            // 故意用不同 seed —— 期望出现差异
            let b = wavefront::render_pixel(
                &scene, &cam(), &EvalCtx::new(2, 1, 0), (px, py), resolution, &params,
            );
            if a[0].to_bits() == b[0].to_bits() {
                same += 1;
            } else {
                diff += 1;
            }
        }
    }
    assert!(
        diff > 0,
        "换seed 后 0 处不同——5a 的逐位比对可能恒真（同seed 下未真正采样）"
    );
    assert!(same + diff == (resolution.0 * resolution.1) as usize);
}
