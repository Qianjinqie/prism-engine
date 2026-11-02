//! W2 / A2 **成像级**判据（`docs/v2.5-v3.0-路线图.md` §4.2 W2 行原话）：
//!
//! > HDR 环境球在金属球上成像与参考 PSNR ≥ 40 dB
//!
//! # 为什么单开一个文件，而不是塞进 V1–V6
//!
//! W0 §3.2 的 V1–V6 判的是**估计器性质**（无偏、pdf 一致、MIS 互补、逐位对拍）。
//! "成像 PSNR" 判的是**别的东⻄**：给定预算下的画面与收敛画面的接近程度。
//! 它可以全绿而估计器仍有偏（两张图一起偏），也可以全红而估计器无偏（预算太低）。
//! 两者都要，故分列，不互相替代。
//!
//! # 口径先钉死（否则 PSNR 可以随便调）
//!
//! | 项 | 取值 | 依据 |
//! |---|---|---|
//! | 环境 | 128×64 等距柱状：天际梯度 `0.35~1.40`，3° 太阳 `L=400`（对比 ~1000:1） | W0 §3 定稿 C1 的靶场景（高对比） |
//! | 材质 | 半径 1 的金属球：`metalness=1`、`roughness=0.12`、`base_color=1` | "金属球" |
//! | 相机 | `eye=(0,0,4)`、`look=原点`、`fov=35°`、32×32 整幅 | 球占 ~26 px 高 |
//! | 参考图 | **4096 spp**，种子 `(2026, 11, 3)` | W0 §1.1「固定参考图（≥4096 spp 均匀）」 |
//! | 被测图 | **256 spp**，种子 `(2026, 11, 2)` | 预算 1/16；两图种子不同 ⇒ 噪声独立 |
//! | PSNR 域 | 逐通道 Reinhard `x/(1+x)` → MSE → `10·log10(1/MSE)`，MAX=1 | HDR 域 PSNR 无统一定义（见下） |
//! | 判据窗口 | **球心 16×16 窗口**（全部像素命中球面） | 背景像素是解析值、零方差，计入会把 PSNR 抬成"自证" |
//!
//! ## 为什么 PSNR 必须在显示域算
//!
//! HDR 直接算 PSNR 时 `MAX` 取谁都能吵：取峰值 400 ⇒ 40 dB 允许 RMSE=4.0
//! （相对峰值 1%），而暗部真值 0.35 的像素错 4.0 早已不是"成像一致"；取 1.0 ⇒
//! 亮斑像素的量化台阶就足以把 PSNR 压到 20 dB 以下。Reinhard 压缩后两档都落在
//! 同一把尺子上，`40 dB` 才对应一句人话：**压缩域 RMSE ≤ 1%**。
//!
//! HDR 域的相对 RMSE（除以参考图最大通道）同时打印，**不参与判据**，只做透明度。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{render_pixel, MaterialKind, RenderParams, Scene, Sphere};

/// 环境图尺寸（等距柱状）。
const EW: u32 = 128;
const EH: u32 = 64;
/// 成像分辨率。
const RES: u32 = 32;
/// 参考 / 被测样本数。
const REF_SPP: u32 = 4096;
const TEST_SPP: u32 = 256;
/// 判据窗口边长（以球心为中心的正方形）。
const WINDOW: u32 = 16;

/// 太阳方向（单位向量）：偏左上前方，镜面球正对相机的那一片能反射到它。
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
                Vec3::new(400.0, 380.0, 340.0) // 略暖的太阳
            } else {
                Vec3::new(0.85, 0.92, 1.0) * sky
            };
            out.push([rgb.x, rgb.y, rgb.z, 1.0]);
        }
    }
    out
}

/// 金属球场景（环境图经 `Scene::add_environment` 一步到位装灯 + 建 CDF）。
fn scene() -> Scene {
    let mut s = Scene::default();
    s.add_environment(EW, EH, &env_pixels());
    s.spheres.push((
        Sphere { center: Vec3::splat(0.0), radius: 1.0 },
        MaterialKind::Pbr(Bsdf::new(
            Vec3::splat(1.0), // base_color
            1.0,              // metalness
            0.12,             // roughness
            1.5,              // ior
            0.0,              // transmission
            0.0,              // subsurface
            Vec3::splat(0.3),
        )),
    ));
    s.validate_layout().expect("场景装配必须自洽");
    s
}

/// 整幅渲染（`render_pixel` 逐像素；这里是"成像"而非"单点"）。
fn render(spp: u32, seed: u64, frame: u64) -> Vec<Vec3> {
    let sc = scene();
    let cam = Camera::new(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::splat(0.0),
        Vec3::new(0.0, 1.0, 0.0),
        35.0,
        0.0,
        1.0,
        0.0,
        1.0 / 24.0,
    );
    let params = RenderParams::new(spp, 8);
    let ctx = EvalCtx::new(seed, 11, frame);
    let mut out = Vec::with_capacity((RES * RES) as usize);
    for y in 0..RES {
        for x in 0..RES {
            let p = render_pixel(&sc, &cam, &ctx, (x, y), (RES, RES), &params);
            out.push(Vec3::new(p[0], p[1], p[2]));
        }
    }
    out
}

/// 球心 16×16 窗口内的像素下标。
fn window_pixels() -> Vec<usize> {
    let lo = (RES - WINDOW) / 2;
    let mut v = Vec::with_capacity((WINDOW * WINDOW) as usize);
    for y in lo..lo + WINDOW {
        for x in lo..lo + WINDOW {
            v.push((y * RES + x) as usize);
        }
    }
    v
}

/// 按通道取值（`Vec3` 不实现 `Index`）。
#[inline]
fn ch(v: Vec3, k: usize) -> f32 {
    match k {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// 显示域 PSNR（Reinhard 压缩 → MSE → `10·log10(1/MSE)`，MAX = 1）。
fn psnr_display(a: &[Vec3], b: &[Vec3], idx: &[usize]) -> f64 {
    let tm = |c: f32| (c / (1.0 + c)) as f64;
    let mut mse = 0.0f64;
    for &i in idx {
        for k in 0..3 {
            let x = tm(ch(a[i], k));
            let y = tm(ch(b[i], k));
            mse += (x - y) * (x - y);
        }
    }
    mse /= (idx.len() * 3) as f64;
    -10.0 * mse.log10()
}

/// HDR 域相对 RMSE（除以参考图最大通道）——只做透明度，不参与判据。
fn rel_rmse_hdr(a: &[Vec3], b: &[Vec3], idx: &[usize]) -> f64 {
    let mut max = 1e-9f64;
    for &i in idx {
        for k in 0..3 {
            max = max.max(ch(b[i], k).abs() as f64);
        }
    }
    let mut se = 0.0f64;
    for &i in idx {
        for k in 0..3 {
            let d = ch(a[i], k) as f64 - ch(b[i], k) as f64;
            se += d * d;
        }
    }
    (se / (idx.len() * 3) as f64).sqrt() / max
}

#[test]
fn hdr_env_on_metal_sphere_matches_reference_at_40db() {
    let reference = render(REF_SPP, 2026, 3);
    let test = render(TEST_SPP, 2026, 2);
    // 负控：同一度量在 16 spp 下必须**掉到 40 dB 以下**。没有这一条，
    // "40 dB" 可能只是这把尺子的地板（例如 Reinhard 压缩把噪声压平了），
    // 判据就变成恒真。16 spp 是刻意选的"明显不够"的预算。
    let control = render(16, 2026, 2);

    let all: Vec<usize> = (0..(RES * RES) as usize).collect();
    let win = window_pixels();

    let psnr_win = psnr_display(&test, &reference, &win);
    let psnr_all = psnr_display(&test, &reference, &all);
    let psnr_ctrl = psnr_display(&control, &reference, &win);
    let rrmse_win = rel_rmse_hdr(&test, &reference, &win);

    // 画面本身的参照量：参考图窗口内的辐亮度范围（说明这确实是一张 HDR 图）。
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for &i in &win {
        for k in 0..3 {
            lo = lo.min(ch(reference[i], k) as f64);
            hi = hi.max(ch(reference[i], k) as f64);
        }
    }

    println!(
        "PSNR（成像）：{TEST_SPP} spp vs {REF_SPP} spp 参考，{RES}×{RES} 整幅；\
         窗口 {WINDOW}×{WINDOW}（全命中球面）PSNR = {psnr_win:.3} dB，\
         整幅 PSNR = {psnr_all:.3} dB（背景为解析值，仅参考）；\
         负控 16 spp = {psnr_ctrl:.3} dB；\
         窗口内 HDR 辐亮度 {lo:.4}~{hi:.4}，HDR 相对 RMSE = {rrmse_win:.3e}"
    );

    assert!(
        psnr_win >= 40.0,
        "球面窗口 PSNR {psnr_win:.3} dB 低于路线图 §4.2 的 40 dB 门槛"
    );
    assert!(
        psnr_ctrl < 40.0,
        "负控失败：16 spp 也到 {psnr_ctrl:.3} dB ⇒ 这把尺子分不出预算，40 dB 判据恒真"
    );
}
