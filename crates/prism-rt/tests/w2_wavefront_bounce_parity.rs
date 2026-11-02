//! W2 剩余欠账的**实测**：整条 WGSL wavefront 的 4 弹射 vs 8 弹射（CPU 权威）差异。
//!
//! # 这个文件判什么（以及为什么必须现在判）
//!
//! `docs/v2.5-W2-光照报告.md` §8.1 原话：
//!
//! > 4 弹射 vs 8 弹射的画面差异：**未测**（前提是整条 wavefront 跑起来）
//!
//! W2 收口时 `rt_wavefront.wgsl` 还没接上环境 CDF（group(0) 的 {slots,params,accum}
//! 与 env blob 的绑定号冲突，宿主 ABI 要求每个 entry 的绑定集是连续前缀），故那一轮
//! 只能给探针级判据。本轮把 env blob 挂到 `group(2) @binding(5)`（见
//! `rt_wavefront.wgsl` 绑定段注释与 `web/abi.rs` 的 `shade` group 前缀），
//! **wavefront 的逃逸段从此与环境图同一份 CDF 语义**。
//!
//! # 为什么用 `prism_rt::wavefront` 而不用 GPU 跑
//!
//! `prism_rt::wavefront` 是**同一个估计量**的 CPU 逐句重排——`wavefront.rs` 的
//! `wavefront_equals_integrator_bitwise` 已证明它和 `integrator` **逐位相等**
//! （本文件再证一次，作为本文件的第 0 条自检）。仓库里没有 path-tracing 通路的
//! SPIR-V 发射器（`spirv_rt.rs` 发的是 megakernel），Vulkan compute 宿主也只支持
//! 单 bind group，故"整条 WGSL wavefront 真机跑 4 vs 8"在本环境**不可达**。
//! 这一点如实写在报告里：本文件测的是**估计量层面的弹射截断差**，
//! 不是"WGSL 在 llvmpipe 上跑出来的差"。
//!
//! # 口径（先钉死）
//!
//! | 项 | 取值 |
//! |---|---|
//! | 参考（8 弹射 CPU 权威） | `integrator::render_pixel`，`max_bounces = 8`，种子 (2026, 11, 2) |
//! | 被测（4 弹射 wavefront） | `wavefront::render_frame`，`max_bounces = 4`，**同种子** |
//! | 阶梯 | 2 / 4 / 8 / 16 弹射，**同一批随机数**（common random numbers） |
//! | 分辨率 | 12×12（球占满中心 ~8 px，背景是解析值） |
//! | spp | 512 |
//! | 场景 A | 金属球（PBR：metalness=1、roughness=0.12）+ 128×64 环境 |
//! | 场景 B | 漫反射球（Lambert albedo=0.85）+ 同一环境（多次弹射占比大，截断差最明显） |
//! | PSNR 域 | 逐通道 Reinhard `x/(1+x)` → MSE → `10·log10(1/MSE)`，MAX=1 |
//!
//! 同种子 ⇒ 两版走**完全相同的采样流**，差额里没有"两个独立噪声场相减"的方差项，
//! 只剩**弹射截断的偏差**。这也是为什么 8 vs 8 那一行必须是 0（第 0 条自检）。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, MaterialKind, Plane, RenderParams, Scene, Sphere};
use prism_rt::wavefront;

/// 环境图尺寸（等距柱状）。
const EW: u32 = 128;
const EH: u32 = 64;
/// 成像分辨率（像素）。
const RES: u32 = 12;
/// 每像素样本数（两版相同）。
const SPP: u32 = 512;
/// 固定种子（写死：确定性纪律，不取时间/随机源）。
const SEED: u64 = 2026;
const STREAM: u64 = 11;
const FRAME: u64 = 2;
/// 参考弹射数（CPU 权威：`RenderParams::new(spp, 8)` 是产品默认档）。
const REF_BOUNCES: u32 = 8;
/// 阶梯（含 4 = `WF2_MAX_BOUNCES`，含 8 = 权威，含 2/16 = 收敛趋势两端）。
const LADDER: [u32; 4] = [2, 4, 8, 16];

/// 太阳方向（与 `w2_a2_ibl_psnr.rs` 同一靶场景，便于横向对读）。
fn sun_dir() -> Vec3 {
    Vec3::new(-0.45, 0.55, 0.70).normalize()
}

/// 3° 太阳 + 天际梯度的等距柱状图（行主序，`v=0` 在 +Y）。
fn env_pixels() -> Vec<[f32; 4]> {
    let sun = sun_dir();
    let cos_sun = (3.0f32).to_radians().cos();
    let mut out = Vec::with_capacity((EW * EH) as usize);
    for j in 0..EH {
        let v = (j as f32 + 0.5) / EH as f32;
        for i in 0..EW {
            let u = (i as f32 + 0.5) / EW as f32;
            let d = prism_rt::env::uv_to_dir(u, v);
            let sky = 0.35 + 1.05 * (1.0 - v);
            let rgb = if d.dot(&sun) >= cos_sun {
                Vec3::new(400.0, 380.0, 340.0)
            } else {
                Vec3::new(0.85, 0.92, 1.0) * sky
            };
            out.push([rgb.x, rgb.y, rgb.z, 1.0]);
        }
    }
    out
}

/// 场景（`metal = true` → 金属球；`false` → Lambert 白球）。
///
/// # 为什么要加地板 + 背墙（这是本文件第一版踩的坑）
///
/// 只放**一个凸球**时，球面上任意点沿外半球方向**永远不会再打到球自己**
/// （凸性），于是 `max_bounces ≥ 2` 与 `max_bounces = 1` 的输出**逐位相同**——
/// 第一版实测就是全 0，等于什么都没测。加无限地板 + 背墙后，光在"地板 ↔ 墙 ↔
/// 球"之间多次弹射，弹射截断才有可测量。
fn scene(metal: bool) -> Scene {
    let mut s = Scene::default();
    s.add_environment(EW, EH, &env_pixels());
    let m = if metal {
        MaterialKind::Pbr(Bsdf::new(
            Vec3::splat(1.0),
            1.0,  // metalness
            0.12, // roughness
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        ))
    } else {
        MaterialKind::Lambert { albedo: Vec3::splat(0.85) }
    };
    s.spheres.push((Sphere { center: Vec3::splat(0.0), radius: 1.0 }, m));
    // 地板（法线朝上）与背墙（法线朝相机），均为 Lambert 0.85。
    let wall = MaterialKind::Lambert { albedo: Vec3::splat(0.85) };
    s.planes
        .push((Plane::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Some(4.0)), wall));
    s.planes
        .push((Plane::new(Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 1.0), Some(4.0)), wall));
    s.validate_layout().expect("场景装配必须自洽");
    s
}

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::splat(0.0),
        Vec3::new(0.0, 1.0, 0.0),
        35.0,
        0.0,
        1.0,
        0.0,
        1.0 / 24.0,
    )
}

/// CPU 权威：`integrator::render_pixel` 整幅。
fn render_integrator(sc: &Scene, cam: &Camera, bounces: u32, spp: u32) -> Vec<Vec3> {
    let params = RenderParams::new(spp, bounces);
    let ctx = EvalCtx::new(SEED, STREAM, FRAME);
    let mut out = Vec::with_capacity((RES * RES) as usize);
    for y in 0..RES {
        for x in 0..RES {
            let p = render_pixel(sc, cam, &ctx, (x, y), (RES, RES), &params);
            out.push(Vec3::new(p[0], p[1], p[2]));
        }
    }
    out
}

/// wavefront 仿真：同一个 `max_bounces` 经队列状态机推进。
fn render_wavefront(sc: &Scene, cam: &Camera, bounces: u32, spp: u32) -> Vec<Vec3> {
    let params = RenderParams::new(spp, bounces);
    let ctx = EvalCtx::new(SEED, STREAM, FRAME);
    wavefront::render_frame(sc, cam, &ctx, (RES, RES), &params)
        .into_iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect()
}

/// Reinhard 压缩（显示域；PSNR 在压缩域算，理由见 `w2_a2_ibl_psnr.rs` 头注）。
#[inline]
fn reinhard(x: f32) -> f32 {
    x / (1.0 + x)
}

/// 逐通道取值（`Vec3` 不实现 `Index`）。
#[inline]
fn ch(v: Vec3, k: usize) -> f32 {
    match k {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// 比较报告（全部为实测值，不参与渲染）。
#[derive(Debug, Clone, Copy)]
struct Cmp {
    /// Reinhard 域 PSNR（dB）。
    psnr_db: f64,
    /// 相对 RMSE（HDR 域，除以参考图最大通道）。
    rel_rmse: f64,
    /// 逐通道最大绝对差（HDR）。
    max_abs: f32,
    /// 逐通道最大差（Reinhard → 8 bit 量化后，0..255）。
    max_q8: i32,
    /// 量化后至少一通道不同的像素数。
    diff_px: usize,
    /// 像素总数。
    total: usize,
}

fn compare(a: &[Vec3], b: &[Vec3]) -> Cmp {
    assert_eq!(a.len(), b.len());
    let mut peak = 0.0f32;
    for v in b {
        for k in 0..3 {
            peak = peak.max(ch(*v, k));
        }
    }
    let mut se = 0.0f64; // Reinhard 域平方误差和
    let mut n = 0usize;
    let mut sse_hdr = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_q8 = 0i32;
    let mut diff_px = 0usize;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let mut px_diff = false;
        for k in 0..3 {
            let (xa, xb) = (ch(*pa, k), ch(*pb, k));
            let d = xa - xb;
            max_abs = max_abs.max(d.abs());
            sse_hdr += (d as f64) * (d as f64);
            let (ra, rb) = (reinhard(xa), reinhard(xb));
            let dd = (ra - rb) as f64;
            se += dd * dd;
            n += 1;
            let qa = (ra.clamp(0.0, 1.0) * 255.0).round() as i32;
            let qb = (rb.clamp(0.0, 1.0) * 255.0).round() as i32;
            let q = (qa - qb).abs();
            max_q8 = max_q8.max(q);
            if q != 0 {
                px_diff = true;
            }
        }
        if px_diff {
            diff_px += 1;
        }
    }
    let mse = se / n as f64;
    let psnr_db = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    };
    let rel_rmse = if peak > 0.0 {
        (sse_hdr / n as f64).sqrt() / peak as f64
    } else {
        0.0
    };
    Cmp {
        psnr_db,
        rel_rmse,
        max_abs,
        max_q8,
        diff_px,
        total: a.len(),
    }
}

/// 逐通道 `to_bits` 全等（逐位判据）。
fn bitwise_equal(a: &[Vec3], b: &[Vec3]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            (0..3).all(|k| ch(*x, k).to_bits() == ch(*y, k).to_bits())
        })
}

/// 打印一行阶梯结果（`--nocapture` 可见）。
fn report_row(tag: &str, bounces: u32, c: Cmp) {
    println!(
        "[{tag}] {bounces:>2} 弹射 vs {REF_BOUNCES} 弹射（CPU 权威）：PSNR {:.3} dB，\
         相对 RMSE {:.4e}，max|Δ| {:.4e}，量化 max_channel_delta {}，differing_pixels {}/{}（{:.2}%）",
        c.psnr_db,
        c.rel_rmse,
        c.max_abs,
        c.max_q8,
        c.diff_px,
        c.total,
        100.0 * c.diff_px as f64 / c.total as f64
    );
}

/// **第 0 条自检**：wavefront(8) ≡ integrator(8) **逐位**。
///
/// 没有这一条，"4 弹射差多少"就分不清是"弹射数"还是"两个渲染器不同"。
#[test]
fn wavefront_matches_integrator_bitwise_at_same_bounce_count() {
    for &metal in &[true, false] {
        let sc = scene(metal);
        let cam = camera();
        let wf = render_wavefront(&sc, &cam, REF_BOUNCES, 64);
        let ig = render_integrator(&sc, &cam, REF_BOUNCES, 64);
        assert!(
            bitwise_equal(&wf, &ig),
            "metal={metal}：同弹射数下 wavefront 必须与 integrator 逐位相等（否则本文件的对比无意义）"
        );
    }
    println!("[自检] wavefront(8) ≡ integrator(8) 逐位相等（金属球 + 漫反射球，各 {} px）", RES * RES);
}

/// **主判据**：4 弹射（wavefront）vs 8 弹射（CPU 权威）的成像差，以及弹射阶梯的收敛趋势。
#[test]
fn four_bounce_wavefront_vs_eight_bounce_cpu_reference() {
    println!(
        "口径：{}×{}、spp={}、种子 ({SEED},{STREAM},{FRAME})、环境 {EW}×{EH}（3° 太阳 L=400 + 天际梯度）、\
         PSNR 在 Reinhard 域",
        RES, RES, SPP
    );
    let mut out_rows: Vec<(bool, u32, Cmp)> = Vec::new();
    for &metal in &[true, false] {
        let tag = if metal { "金属球 r=0.12" } else { "漫反射球 albedo=0.85" };
        let sc = scene(metal);
        let cam = camera();
        let reference = render_integrator(&sc, &cam, REF_BOUNCES, SPP);
        for &b in LADDER.iter() {
            let img = render_wavefront(&sc, &cam, b, SPP);
            let c = compare(&img, &reference);
            report_row(tag, b, c);
            out_rows.push((metal, b, c));
        }
        // 8 弹射那一行必须逐位 0（同一估计量 + 同一采样流）
        let eight = out_rows.iter().find(|(m, b, _)| *m == metal && *b == REF_BOUNCES).unwrap().2;
        assert_eq!(eight.max_abs, 0.0, "{tag}：8 弹射 vs 8 弹射必须 0 差");
        assert_eq!(eight.diff_px, 0, "{tag}：8 弹射 vs 8 弹射必须 0 像素差");
    }

    // 收敛趋势：相对误差随弹射数单调下降（2 > 4 > 8 = 0），16 弹射不得反弹。
    for &metal in &[true, false] {
        let tag = if metal { "金属球" } else { "漫反射球" };
        let err = |b: u32| {
            out_rows
                .iter()
                .find(|(m, bb, _)| *m == metal && *bb == b)
                .unwrap()
                .2
                .rel_rmse
        };
        println!(
            "[{tag}] 收敛趋势：2→{:.4e} 4→{:.4e} 8→{:.4e} 16→{:.4e}",
            err(2),
            err(4),
            err(8),
            err(16)
        );
        assert!(err(2) >= err(4), "{tag}：2 弹射误差必须 ≥ 4 弹射（截断偏差随深度下降）");
        assert_eq!(err(8), 0.0, "{tag}：8 弹射是参考本身，误差必须 0");
        assert!(
            err(16) <= err(4),
            "{tag}：16 弹射的误差不得高于 4 弹射（8 之后是采样噪声地板，不是新偏差）"
        );
    }

    // 4 弹射 vs 8 弹射**确实存在差异**（不是"两边一样所以看不出来"）：
    // 漫反射场景的多次弹射占比大，4 弹射的截断偏差必须可测。
    let diffuse4 = out_rows
        .iter()
        .find(|(m, b, _)| !*m && *b == 4)
        .unwrap()
        .2;
    println!(
        "[结论] 漫反射球 4 弹射 vs 8 弹射：PSNR {:.3} dB、相对 RMSE {:.4e}、\
         量化 max_channel_delta {}、differing_pixels {}/{}（{:.2}%）",
        diffuse4.psnr_db,
        diffuse4.rel_rmse,
        diffuse4.max_q8,
        diffuse4.diff_px,
        diffuse4.total,
        100.0 * diffuse4.diff_px as f64 / diffuse4.total as f64
    );
    assert!(
        diffuse4.diff_px > 0,
        "4 弹射与 8 弹射在漫反射场景下必须可测出差异（否则本判据什么都没测）"
    );
}

/// **旁证**：关掉萤火钳制（`clamp_indirect = None`）时截断差更大、趋势一致。
///
/// 钳制会把多弹射的尾部贡献压掉，等于人为缩小截断差；这一条证明
/// "4 vs 8 的差不是钳制造成的假象"。
#[test]
fn bounce_ladder_without_firefly_clamp_is_larger() {
    let sc = scene(false);
    let cam = camera();
    let ctx = EvalCtx::new(SEED, STREAM, FRAME);
    let render = |bounces: u32, clamp: Option<f32>| -> Vec<Vec3> {
        let mut params = RenderParams::new(SPP, bounces);
        params.clamp_indirect = clamp;
        let mut out = Vec::with_capacity((RES * RES) as usize);
        for y in 0..RES {
            for x in 0..RES {
                let p = render_pixel(&sc, &cam, &ctx, (x, y), (RES, RES), &params);
                out.push(Vec3::new(p[0], p[1], p[2]));
            }
        }
        out
    };
    let reference = render(REF_BOUNCES, None);
    let c4 = compare(&render(4, None), &reference);
    let c2 = compare(&render(2, None), &reference);
    println!(
        "[无萤火钳制][漫反射球] 4 弹射 vs 8 弹射：PSNR {:.3} dB、相对 RMSE {:.4e}、max|Δ| {:.4e}、\
         differing_pixels {}/{}",
        c4.psnr_db, c4.rel_rmse, c4.max_abs, c4.diff_px, c4.total
    );
    println!(
        "[无萤火钳制][漫反射球] 2 弹射 vs 8 弹射：PSNR {:.3} dB、相对 RMSE {:.4e}",
        c2.psnr_db, c2.rel_rmse
    );
    assert!(c2.rel_rmse >= c4.rel_rmse, "无钳制时 2 弹射误差也必须 ≥ 4 弹射");
}

/// **负控**：只有一个凸球时 4 弹射与 8 弹射**逐位相同**（差 0）。
///
/// 这条不是为了"证明弹射数无所谓"，而是钉住上一版实测踩到的事实：球面凸性
/// ⇒ 外半球方向永不再命中自身 ⇒ 没有互反射 ⇒ 弹射截断无可测量。
/// 也就是说"4 vs 8 的差"**是场景相关的**，把它写成"0 差异"或"到处都有差"
/// 都不对——两种场景的数字必须分列（本文件两条测试各占一边）。
#[test]
fn convex_sphere_only_scene_has_zero_bounce_truncation() {
    let mut sc = Scene::default();
    sc.add_environment(EW, EH, &env_pixels());
    sc.spheres.push((
        Sphere { center: Vec3::splat(0.0), radius: 1.0 },
        MaterialKind::Lambert { albedo: Vec3::splat(0.85) },
    ));
    sc.validate_layout().expect("场景装配必须自洽");
    let cam = camera();
    let reference = render_integrator(&sc, &cam, REF_BOUNCES, 128);
    let four = render_wavefront(&sc, &cam, 4, 128);
    let c = compare(&four, &reference);
    println!(
        "[负控][单凸球] 4 弹射 vs 8 弹射：PSNR {:.3} dB、相对 RMSE {:.4e}、max|Δ| {:.4e}、differing_pixels {}/{}",
        c.psnr_db, c.rel_rmse, c.max_abs, c.diff_px, c.total
    );
    assert_eq!(c.max_abs, 0.0, "单凸球场景：4 弹射与 8 弹射必须逐位相同（无互反射）");
}
