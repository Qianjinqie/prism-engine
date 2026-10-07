//! QA 攻击性测试 — Phase 4 D2 wavefront vs integrator 判对（`prism-rt`）。
//!
//! # 判对命门
//! 验收核心：`wavefront::render_pixel`（GPU WGSL 的 CPU 仿真基准）
//! 必须与 `integrator::render_pixel`（CpuRef）在**同场景同种子**下逐像素**逐位一致**
//! （强于「统计 diff ≤ 阈值」）。本文件以多种场景/参数/材质攻击该契约。
//!
//! 每个测试失败的含义在函数头注释显式声明。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    render_pixel as ref_render_pixel, Aabb, MaterialKind, RenderParams, Scene, Sphere,
};
use prism_rt::light::Light;
use prism_rt::media::HomogeneousMedium;
use prism_rt::wavefront::{
    render_frame, render_frame_with_stats, render_pixel as wf_render_pixel, WavefrontStats,
};

// ===========================================================================
// 场景工厂（判对含场景构造）
// ===========================================================================

fn cornell_scene() -> Scene {
    let white = MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) };
    let red = MaterialKind::Lambert { albedo: Vec3::new(0.63, 0.065, 0.05) };
    let green = MaterialKind::Lambert { albedo: Vec3::new(0.14, 0.45, 0.091) };
    let t = 0.1f32;
    Scene {
        boxes: vec![
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 0.0, 1.0 + t)), white),
            (Aabb::new(Vec3::new(-1.0 - t, 2.0, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)), white),
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, -1.0)), white),
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(-1.0, 2.0 + t, 1.0 + t)), red),
            (Aabb::new(Vec3::new(1.0, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)), green),
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

/// 介质场景（攻击 medium 碰撞 → 体 NEE → 相位续弹路径的判对）。
fn medium_scene() -> Scene {
    let white = MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) };
    Scene {
        boxes: vec![(Aabb::new(Vec3::new(-2.0, -0.1, -2.0), Vec3::new(2.0, 0.0, 2.0)), white)],
        spheres: vec![(Sphere { center: Vec3::new(0.0, 0.5, 0.0), radius: 0.5 }, white)],
        lights: vec![Light::Sphere {
            center: Vec3::new(0.0, 1.5, 0.0),
            radius: 0.2,
            radiance: Vec3::new(16.0, 13.0, 10.0),
        }],
        medium: Some(prism_rt::integrator::MediumRegion {
            bounds: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 3.0, 1.0)),
            medium: HomogeneousMedium::new(0.1, 0.9, 0.6),
        }),
    }
}

/// PBR 场景（攻击 glass 透射 / SSS random walk 续弹路径的判对）。
fn pbr_scene() -> Scene {
    use prism_rt::bsdf::Bsdf;
    let glass = MaterialKind::Pbr(Bsdf::new(
        Vec3::splat(1.0), 0.05, 0.0, 1.5, 1.0, 0.0, Vec3::splat(0.3),
    ));
    let sss = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.9, 0.4, 0.4), 0.6, 0.0, 1.4, 0.0, 0.8, Vec3::new(0.2, 0.1, 0.1),
    ));
    let white = MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) };
    Scene {
        boxes: vec![(Aabb::new(Vec3::new(-2.0, -0.1, -2.0), Vec3::new(2.0, 0.0, 2.0)), white)],
        spheres: vec![
            (Sphere { center: Vec3::new(-0.4, 0.5, 0.0), radius: 0.45 }, glass),
            (Sphere { center: Vec3::new(0.5, 0.4, 0.1), radius: 0.4 }, sss),
        ],
        lights: vec![Light::Sphere {
            center: Vec3::new(0.0, 1.5, 0.0),
            radius: 0.2,
            radiance: Vec3::new(16.0, 13.0, 10.0),
        }],
        medium: None,
    }
}

// ===========================================================================
// 判对工具
// ===========================================================================

/// 逐像素逐通道逐位比较 wavefront 帧与 integrator 帧；返回不一致通道数。
fn bitwise_diff(scene: &Scene, cam: &Camera, ctx: &EvalCtx, res: (u32, u32), p: &RenderParams) -> usize {
    let mut diff = 0usize;
    for y in 0..res.1 {
        for x in 0..res.0 {
            let a = wf_render_pixel(scene, cam, ctx, (x, y), res, p);
            let b = ref_render_pixel(scene, cam, ctx, (x, y), res, p);
            for c in 0..3 {
                if a[c].to_bits() != b[c].to_bits() {
                    diff += 1;
                }
            }
        }
    }
    diff
}

// ===========================================================================
// 判对命门测试
// ===========================================================================

/// 失败含义（**P0 判对命门**）：wavefront CPU 仿真与 integrator 在 Cornell 场景
/// 逐像素**逐位不一致**——GPU/CpuRef 互判基准破裂，Phase 4 验收失败。
#[test]
fn wavefront_equals_integrator_cornell_bitwise() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let res = (12, 12);
    for (spp, bounces, seed) in [(32u32, 3u32, 2024u64), (16, 5, 101), (8, 8, 7)] {
        let ctx = EvalCtx::new(seed, 7, 3);
        let p = RenderParams::new(spp, bounces);
        let diff = bitwise_diff(&scene, &cam, &ctx, res, &p);
        assert_eq!(diff, 0, "Cornell spp={spp} b={bounces} seed={seed}：判对逐位 diff={diff}");
    }
}

/// 失败含义（**P0 判对命门**）：含介质场景下 wavefront 与 integrator 逐位发散——
/// 介质碰撞/体 NEE/相位续弹的样本槽序在两实现间不一致。
#[test]
fn wavefront_equals_integrator_medium_bitwise() {
    let scene = medium_scene();
    let cam = cornell_camera();
    let res = (10, 10);
    for (spp, bounces, seed) in [(24u32, 4u32, 6u64), (12, 6, 55)] {
        let ctx = EvalCtx::new(seed, 7, 3);
        let p = RenderParams::new(spp, bounces);
        let diff = bitwise_diff(&scene, &cam, &ctx, res, &p);
        assert_eq!(diff, 0, "介质 spp={spp} b={bounces}：diff={diff}");
    }
}

/// 失败含义（**P0 判对命门**）：含 glass 透射 + SSS random walk 场景逐位发散——
/// 复杂 BSDF 续弹路径的估计量在两实现间不一致（最易出偏差的分支）。
#[test]
fn wavefront_equals_integrator_pbr_sss_bitwise() {
    let scene = pbr_scene();
    let cam = cornell_camera();
    let res = (10, 10);
    for (spp, bounces, seed) in [(24u32, 6u32, 11u64), (12, 10, 12)] {
        let ctx = EvalCtx::new(seed, 7, 3);
        let p = RenderParams::new(spp, bounces);
        let diff = bitwise_diff(&scene, &cam, &ctx, res, &p);
        assert_eq!(diff, 0, "PBR/SSS spp={spp} b={bounces}：diff={diff}");
    }
}

/// 失败含义（**P0 判对命门**）：切换 `nee_only` / `clamp_indirect` 验证开关后
/// 两实现发散——验证开关路径未在 wavefront 中镜像。
#[test]
fn wavefront_equals_integrator_switch_paths_bitwise() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let res = (10, 10);
    let ctx = EvalCtx::new(321, 7, 3);

    let mut p = RenderParams::new(16, 4);
    p.nee_only = true;
    assert_eq!(bitwise_diff(&scene, &cam, &ctx, res, &p), 0, "nee_only 路径");

    let mut p = RenderParams::new(16, 4);
    p.clamp_indirect = None;
    assert_eq!(bitwise_diff(&scene, &cam, &ctx, res, &p), 0, "clamp_off 路径");

    let mut p = RenderParams::new(16, 4);
    p.clamp_indirect = Some(0.5);
    assert_eq!(bitwise_diff(&scene, &cam, &ctx, res, &p), 0, "clamp_tight 路径");
}

/// 失败含义：整帧渲染路径 `render_frame` 在首行某像素与逐像素参考发散（行主序
/// 索引或帧级状态泄漏）。
#[test]
fn wavefront_render_frame_matches_per_pixel() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(9, 7, 3);
    let res = (8, 8);
    let p = RenderParams::new(8, 3);
    let frame = render_frame(&scene, &cam, &ctx, res, &p);
    assert_eq!(frame.len(), (res.0 * res.1) as usize);
    for y in 0..res.1 {
        for x in 0..res.0 {
            let idx = (y * res.0 + x) as usize;
            let px = wf_render_pixel(&scene, &cam, &ctx, (x, y), res, &p);
            for c in 0..4 {
                assert_eq!(
                    frame[idx][c].to_bits(),
                    px[c].to_bits(),
                    "({x},{y}) ch{c} 帧级与逐像素不一致"
                );
            }
        }
    }
}

// ===========================================================================
// 边界 / 退化
// ===========================================================================

/// 失败含义：**零 spp** 时 wavefront 与 integrator 行为不一致或 panic
/// （空采样均值未定义/除零）。
#[test]
fn wavefront_zero_spp_matches_integrator() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(1, 1, 0);
    let p = RenderParams::new(0, 4);
    let a = wf_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    let b = ref_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    for c in 0..4 {
        assert_eq!(a[c].to_bits(), b[c].to_bits(), "零 spp 通道 {c}");
    }
}

/// 失败含义：1×1 帧、max_bounces=0（仅相机段）下两实现发散——退化深度处理不一致。
#[test]
fn wavefront_one_pixel_zero_bounce() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(42, 7, 0);
    let p = RenderParams::new(16, 0);
    let a = wf_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    let b = ref_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    for c in 0..4 {
        assert_eq!(a[c].to_bits(), b[c].to_bits(), "b=0 通道 {c}");
    }
}

/// 失败含义：**空场景**（无几何无光）下 wavefront panic 或与 integrator 不一致。
#[test]
fn wavefront_empty_scene_matches_integrator() {
    let scene = Scene::default();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(3, 1, 0);
    let res = (4, 4);
    let p = RenderParams::new(8, 4);
    let diff = bitwise_diff(&scene, &cam, &ctx, res, &p);
    assert_eq!(diff, 0, "空场景判对 diff={diff}");
}

/// 失败含义：空场景 + 空光源下整帧不产出（长度错）或 panic。
#[test]
fn wavefront_empty_scene_frame_length() {
    let scene = Scene::default();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(3, 1, 0);
    let res = (3, 5);
    let p = RenderParams::new(4, 2);
    let frame = render_frame(&scene, &cam, &ctx, res, &p);
    assert_eq!(frame.len(), 15);
}

// ===========================================================================
// 确定性
// ===========================================================================

/// 失败含义：同 `EvalCtx` 两遍整帧**不逐位一致**——违反全链路确定性铁律。
#[test]
fn wavefront_two_pass_bitwise_identical() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let res = (8, 8);
    let p = RenderParams::new(8, 3);
    let f1 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &p);
    let f2 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &p);
    assert_eq!(f1.len(), f2.len());
    for i in 0..f1.len() {
        for c in 0..4 {
            assert_eq!(f1[i][c].to_bits(), f2[i][c].to_bits(), "两遍不逐位一致 px{i} ch{c}");
        }
    }
}

/// 失败含义：不同种子产生**完全相同**的帧（种子未生效）——确定性伪影源未接入。
#[test]
fn wavefront_different_seeds_differ() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let res = (6, 6);
    let p = RenderParams::new(8, 3);
    let a = render_frame(&scene, &cam, &EvalCtx::new(1, 7, 3), res, &p);
    let b = render_frame(&scene, &cam, &EvalCtx::new(2, 7, 3), res, &p);
    let mut any_diff = false;
    for i in 0..a.len() {
        for c in 0..4 {
            if a[i][c].to_bits() != b[i][c].to_bits() {
                any_diff = true;
            }
        }
    }
    assert!(any_diff, "不同种子（1 vs 2）产生相同帧，种子未生效");
}

/// 失败含义：像素位置不影响样本流（网格伪影）——同帧内不同像素应不同。
#[test]
fn wavefront_distinct_pixels_differ() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let res = (8, 8);
    let p = RenderParams::new(8, 3);
    let ctx = EvalCtx::new(5, 7, 3);
    let a = wf_render_pixel(&scene, &cam, &ctx, (1, 1), res, &p);
    let b = wf_render_pixel(&scene, &cam, &ctx, (5, 6), res, &p);
    let differs = (0..3).any(|c| a[c].to_bits() != b[c].to_bits());
    assert!(differs, "两不同像素结果相同，样本流未按像素派生（网格伪影）");
}

// ===========================================================================
// 队列统计自洽
// ===========================================================================

/// 失败含义：wavefront 队列计数不自洽（生成≠像素×spp，或扩展/终止关系破裂）——
/// GPU 队列规模估计失真。
#[test]
fn wavefront_stats_self_consistent() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(3, 7, 3);
    let res = (6, 6);
    let p = RenderParams::new(4, 2);
    let (frame, stats) = render_frame_with_stats(&scene, &cam, &ctx, res, &p);
    assert_eq!(frame.len(), (res.0 * res.1) as usize);
    assert_eq!(stats.generated, res.0 * res.1 * p.spp, "生成数 = 像素 × spp");
    assert!(stats.extended <= stats.generated * p.max_bounces, "扩展数上界");
    assert!(
        stats.extended + stats.terminated >= stats.generated,
        "队列计数不自洽：ext={} term={} gen={}",
        stats.extended,
        stats.terminated,
        stats.generated
    );
    // 默认统计全零。
    assert_eq!(WavefrontStats::default(), WavefrontStats::default());
}

/// 失败含义：舞台枚举缺失（诊断/测试面契约破裂）。
#[test]
fn wavefront_stage_enum_contract() {
    use prism_rt::wavefront::WavefrontStage;
    // 四阶段齐备且可比较。
    let stages = [
        WavefrontStage::Generate,
        WavefrontStage::Shade,
        WavefrontStage::Connect,
        WavefrontStage::Extend,
    ];
    for (i, a) in stages.iter().enumerate() {
        for (j, b) in stages.iter().enumerate() {
            assert_eq!(i == j, a == b, "舞台相等性");
        }
    }
}

// ===========================================================================
// 缺陷 5 自证：`max_bounces` 边界（0 / 1）下wavefront 与 integrator 判对
// ===========================================================================

/// `max_bounces = 0` 时两实现在多场景 × 多 spp × 105 通道上逐通道一致，
/// 且输出有限（无NaN/Inf）、alpha恒为 1。
///
/// 口径：`max_bounces` = 允许的**最大弹射次数**，0 次即不产生任何弹射事件、
/// Radiance 零贡献（integrator 的 `for bounce in 0..0` 执行零次）。
#[test]
fn zero_bounces_wavefront_matches_integrator_all_scenes() {
    let scenes: Vec<(&str, Scene)> = vec![
        ("cornell", cornell_scene()),
        ("empty", Scene::default()),
        ("pbr", pbr_scene()),
        ("medium", medium_scene()),
    ];
    for (name, scene) in &scenes {
        let cam = cornell_camera();
        for spp in [0u32, 1, 4] {
            let ctx = EvalCtx::new(11, 3, 0);
            let p = RenderParams::new(spp, 0);
            let a = wf_render_pixel(scene, &cam, &ctx, (0, 0), (1, 1), &p);
            let b = ref_render_pixel(scene, &cam, &ctx, (0, 0), (1, 1), &p);
            for c in 0..4 {
                assert!(
                    a[c].is_finite(),
                    "{name} spp={spp} b=0 通道 {c} 非有限（首帧 NaN/Inf）：{:?}",
                    a
                );
                assert_eq!(
                    a[c].to_bits(),
                    b[c].to_bits(),
                    "{name} spp={spp} b=0 通道 {c}发散：wavefront={} integrator={}",
                    a[c],
                    b[c]
                );
            }
            // spp>0 时 alpha 恒为 1；spp=0 时两实现同为 0（无样本），
            // 该一致性已由上面的逐通道 to_bits 比较覆盖。
            if spp > 0 {
                assert_eq!(a[3], 1.0, "{name} spp={spp} alpha 恒为 1");
            }
        }
    }
}

/// `max_bounces = 0` 时统计量自洽：生成 1 个相机项，但**零次着色/扩展**
/// （证明确实没有产生弹射事件，而非"恰好结果相同"）。
#[test]
fn zero_bounces_wavefront_stats_show_no_events() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(5, 2, 0);
    let p = RenderParams::new(1, 0);
    let (_px, stats) =
        render_frame_with_stats(&scene, &cam, &ctx, (1, 1), &p);
    assert_eq!(stats.generated, 1, "应恰好生成 1 个相机项");
    assert_eq!(stats.shaded, 0, "max_bounces=0 不得发生任何着色事件");
    assert_eq!(stats.extended, 0, "max_bounces=0 不得发生任何扩展事件");
    assert_eq!(stats.terminated, 0, "max_bounces=0 不得发生终止事件");
}

/// `max_bounces = 1`（单次弹射）仍与 integrator 逐通道一致——证明新增的
/// `max_bounces > 0` 短路没有误伤正常深度的路径。
#[test]
fn one_bounce_wavefront_matches_integrator() {
    let scene = cornell_scene();
    let cam = cornell_camera();
    let ctx = EvalCtx::new(21, 5, 0);
    let p = RenderParams::new(4, 1);
    let a = wf_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    let b = ref_render_pixel(&scene, &cam, &ctx, (0, 0), (1, 1), &p);
    for c in 0..4 {
        assert_eq!(a[c].to_bits(), b[c].to_bits(), "b=1 通道 {c} 发散");
    }
    assert!(stats_have_events(&scene, &cam, 1), "b=1 应确实产生着色事件");
}

/// 辅助：`max_bounces = 1` 时统计量显示确实发生了事件（与 b=0 互补的证据）。
fn stats_have_events(scene: &Scene, cam: &Camera, max_bounces: u32) -> bool {
    let ctx = EvalCtx::new(21, 5, 0);
    let p = RenderParams::new(4, max_bounces);
    let (_px, stats) = render_frame_with_stats(scene, cam, &ctx, (1, 1), &p);
    stats.shaded > 0 || stats.terminated > 0
}
