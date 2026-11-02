//! W2 / A2 验收：HDR 环境光照（IBL）——V1~V5 + 超越判据 #1。
//!
//! 判据口径逐条照抄 `docs/v2.5-W0-算法选型报告.md` §3.2（阈值不放宽）。
//! 本文件只比公开 API（`Scene` / `render_pixel` / `EnvCdf`）：
//! 参考实现也走同一套场景与相机，避免"两份几何"。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::bsdf::luminance;
use prism_rt::camera::Camera;
use prism_rt::env::{uv_to_dir, EnvCdf};
use prism_rt::integrator::{render_pixel, Light, MaterialKind, RenderParams, Scene, Sphere};
use prism_rt::light::{nee_mis_weight, power_heuristic};
use prism_rt::texsample::TexTable;
use std::f32::consts::{FRAC_1_PI, PI};

// ---------------------------------------------------------------------------
// 公共夹具
// ---------------------------------------------------------------------------

/// 常数环境图 `L = radiance`。
fn constant_env(w: u32, h: u32, radiance: Vec3) -> Vec<[f32; 4]> {
    vec![[radiance.x, radiance.y, radiance.z, 1.0]; (w * h) as usize]
}

/// 环境图 `f(u,v) -> rgb`。
fn env_map(w: u32, h: u32, f: impl Fn(f32, f32) -> Vec3) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let u = (i as f32 + 0.5) / w as f32;
            let v = (j as f32 + 0.5) / h as f32;
            let c = f(u, v);
            out.push([c.x, c.y, c.z, 1.0]);
        }
    }
    out
}

fn cam(eye: Vec3, look: Vec3, fov_deg: f32) -> Camera {
    Camera::new(eye, look, Vec3::new(0.0, 1.0, 0.0), fov_deg, 0.0, 1.0, 0.0, 1.0 / 24.0)
}

fn sphere_scene(mat: MaterialKind, radius: f32) -> Scene {
    Scene {
        spheres: vec![(Sphere { center: Vec3::splat(0.0), radius }, mat)],
        ..Default::default()
    }
}

/// 像素均值（`render_pixel` 的公开口径）。
fn px(scene: &Scene, camera: &Camera, x: u32, y: u32, res: (u32, u32), spp: u32, bounces: u32) -> Vec3 {
    let params = RenderParams::new(spp, bounces);
    let ctx = EvalCtx::new(2026, 11, 2);
    let p = render_pixel(scene, camera, &ctx, (x, y), res, &params);
    Vec3::new(p[0], p[1], p[2])
}

// ---------------------------------------------------------------------------
// V2 方向锚定（先测：它最快，且能立刻暴露 CDF/映射的方向错位）
// ---------------------------------------------------------------------------

/// 环境中置一个 4×4 texel 的亮斑（值 10.0），镜面球反射必须命中它。
///
/// 判据（§3.2 V2）：峰值像素位置精确；峰值辐亮度 == 亮斑值（±1 LSB）。
#[test]
fn v2_direction_anchoring_mirror_hits_bright_spot() {
    let (w, h) = (128u32, 64u32);
    let spot = Vec3::splat(10.0);
    let bg = Vec3::splat(0.001);
    // 亮斑中心取**纹素中心**（不是 uv 网格点）：这样 `target` 恰是亮斑的几何
    // 中心，反射命中的就是中心本身，而不是"中心 + 半纹素"——否则质心天然带
    // 半纹素偏置，判据会被实现细节污染。
    let ci = 64u32; // u = (64 + ½)/128 = 0.50390625
    let cj = 16u32; // v = (16 + ½)/32  = 0.515625
    let cu = (ci as f32 + 0.5) / w as f32;
    let cv = (cj as f32 + 0.5) / h as f32;
    let pix = env_map(w, h, |u, v| {
        let i = (u * w as f32) as u32;
        let j = (v * h as f32) as u32;
        if i.abs_diff(ci) <= 1 && j.abs_diff(cj) <= 1 {
            spot
        } else {
            bg
        }
    });
    let target = uv_to_dir(cu, cv); // 亮斑中心方向

    let mut scene = sphere_scene(MaterialKind::Mirror, 1.0);
    scene.add_environment(w, h, &pix);
    scene.assert_layout();

    // 镜面球（半径 1）在原点；相机沿 `target` 方向退到距离 3 处看向原点。
    // 于是球面命中点的法线恰为 `target`、入射方向为 `−target`，
    // 反射方向 `d − 2(d·n)n` 精确等于 `target` ⇒ 中心像素命中亮斑中心。
    let eye = target * 3.0;
    let camera = cam(eye, Vec3::splat(0.0), 2.0);
    let res = (9u32, 9u32);
    let spp = 512u32;
    let v = px(&scene, &camera, 4, 4, res, spp, 4);
    // 峰值辐亮度应当就是亮斑值（镜面 + 块内双线性恒等 ⇒ 逐位 10.0）。
    assert!(
        (v.x - spot.x).abs() <= 1.0 / 255.0,
        "峰值辐亮度 {} 与亮斑值 {} 差超过 1 LSB",
        v.x,
        spot.x
    );
    // 位置：亮区必须**以中心像素为质心**。
    //
    // 为什么不是"argmax == (4,4)"：镜面球把像素方向映射成反射方向时是 2:1
    // 压缩，3×3 texel 的亮斑在 9×9 图上是一片**平台**（多个像素都恰好 10.0，
    // 见下面打印的 9×9 值）。平台的 argmax 取决于扫描顺序，不构成判据；
    // "亮度的加权质心落在中心 ±1 px" 才是"方向锚定正确"的可判定形式。
    let mut sum = 0.0f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut dark_corner = 0.0f32;
    for y in 0..res.1 {
        for x in 0..res.0 {
            let c = px(&scene, &camera, x, y, res, 64, 4);
            let l = luminance(c);
            sum += l;
            cx += l * x as f32;
            cy += l * y as f32;
            if (x, y) == (0, 0) {
                dark_corner = l;
            }
        }
    }
    let (ccx, ccy) = (cx / sum, cy / sum);
    assert!(
        (ccx - 4.0).abs() <= 1.0 && (ccy - 4.0).abs() <= 1.0,
        "亮区质心 ({ccx:.2}, {ccy:.2}) 偏离中心像素 (4,4) 超过 1 px"
    );
    // 反向证据：远离亮斑的角像素必须是背景，否则"处处都是 10"也能过上面两条。
    assert!(
        dark_corner < 0.01 * spot.x,
        "左上角像素 {dark_corner} 不是背景（亮斑没有方向选择性）"
    );
    println!(
        "V2 方向锚定：中心像素 L = {:.6}（亮斑 {:.1}，差 {:.3e} ≤ 1 LSB = {:.6}），\
         亮区质心 = ({ccx:.3}, {ccy:.3}) 期望 (4, 4)，左上角 L = {dark_corner:.3e}",
        v.x,
        spot.x,
        (v.x - spot.x).abs(),
        1.0 / 255.0
    );
}

// ---------------------------------------------------------------------------
// V1 白炉
// ---------------------------------------------------------------------------

/// 常数环境 L=1 + 白漫反射 albedo=1，8 弹射 2048 spp ⇒ 像素 = 1.0（≤1e-2）。
#[test]
fn v1_white_furnace_converges_to_one() {
    let (w, h) = (64u32, 32u32);
    let mut scene = sphere_scene(MaterialKind::Lambert { albedo: Vec3::splat(1.0) }, 1.0);
    let env = constant_env(w, h, Vec3::splat(1.0));
    scene.add_environment(w, h, &env);
    scene.assert_layout();
    let camera = cam(Vec3::new(0.0, 0.0, 3.0), Vec3::splat(0.0), 45.0);
    let res = (8u32, 8u32);
    let mut worst = 0.0f32;
    for y in 0..res.1 {
        for x in 0..res.0 {
            let c = px(&scene, &camera, x, y, res, 2048, 8);
            for ch in [c.x, c.y, c.z] {
                worst = worst.max((ch - 1.0).abs());
            }
        }
    }
    assert!(worst <= 1e-2, "白炉最差相对误差 {worst} 超过 1e-2");
    println!("V1 白炉：64 像素 × 2048 spp，最差 |L−1| = {worst:.3e}");
}

// ---------------------------------------------------------------------------
// V5 pdf 一致性（全 1024 行）
// ---------------------------------------------------------------------------

/// 常数环境下全图每个 texel 的 `pdf_w·4π` 偏差 ≤ 1e-3。
#[test]
fn v5_pdf_consistency_over_all_rows() {
    let (w, h) = (2048u32, 1024u32);
    let env = constant_env(w, h, Vec3::splat(1.0));
    let mut scene = Scene::default();
    scene.add_environment(w, h, &env);
    let cdf: &EnvCdf = scene.env().expect("环境 CDF 必须已建");
    assert_eq!(cdf.dims(), (w, h));
    let omega = cdf.omega_total();
    let mut worst = 0.0f32;
    let mut checked = 0u64;
    for j in 0..h {
        for i in 0..w {
            let d = uv_to_dir((i as f32 + 0.5) / w as f32, (j as f32 + 0.5) / h as f32);
            let p = cdf.pdf_w(d);
            worst = worst.max((p * omega - 1.0).abs());
            checked += 1;
        }
    }
    assert!(worst <= 1e-3, "全图 pdf·4π 最大偏差 {worst} 超过 1e-3");
    println!("V5：{checked} 个 texel（1024 行全覆盖），max|pdf·Ω−1| = {worst:.3e}");
}

// ===========================================================================
// 暴力参考：均匀球面 QMC（与 CDF 采样完全解耦）
// ===========================================================================
//
// V3/V4 的参考值**不许**复用被测对象（CDF 反演 / MIS 权重）。这里只借
// `EnvCdf::radiance`（双线性查表）——那是"环境图长什么样"的定义，属于输入
// 而非采样策略。方向本身由等面积映射的均匀球面点给出，与 CDF 无关。

/// 基 2 radical inverse（Hammersley 第二维；无 RNG、无依赖）。
fn radical_inverse_base2(mut k: u32) -> f32 {
    k = (k << 16) | (k >> 16);
    k = ((k & 0x5555_5555) << 1) | ((k & 0xAAAA_AAAA) >> 1);
    k = ((k & 0x3333_3333) << 2) | ((k & 0xCCCC_CCCC) >> 2);
    k = ((k & 0x0F0F_0F0F) << 4) | ((k & 0xF0F0_F0F0) >> 4);
    k = ((k & 0x00FF_00FF) << 8) | ((k & 0xFF00_FF00) >> 8);
    k as f32 * 2.328_306_4e-10 // 2^-32
}

/// 辐照度 `E(n) = ∫_{S²} L(ω)·max(0, n·ω) dω` 的暴力参考。
///
/// * 均匀球面 QMC：`cosθ = 1 − 2u₁`（等面积）、`φ = 2πu₂`，`N` 点；
/// * 收敛 `O(1/N)`（光滑被积函数 + Hammersley 低差异序列），1e6 点时
///   相对误差实测 ≪ 0.1%（见 V3 打印）。
fn brute_force_irradiance(env: &EnvCdf, table: TexTable<'_>, n: Vec3, samples: u32) -> Vec3 {
    let mut acc = Vec3::splat(0.0);
    let inv = 1.0 / samples as f32;
    for k in 0..samples {
        let u1 = (k as f32 + 0.5) * inv;
        let u2 = radical_inverse_base2(k);
        let z = 1.0 - 2.0 * u1;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = 2.0 * PI * u2;
        let d = Vec3::new(r * phi.cos(), z, r * phi.sin());
        let c = n.dot(&d);
        if c > 0.0 {
            acc = acc + env.radiance(table, d) * c;
        }
    }
    acc * (4.0 * PI * inv)
}

/// 平滑非均匀环境（宽瓣，故意不与坐标轴重合）。
fn smooth_env(w: u32, h: u32) -> Vec<[f32; 4]> {
    let up = Vec3::new(0.3, 1.0, 0.2).normalize();
    env_map(w, h, |u, v| {
        let d = uv_to_dir(u, v);
        let lobe = d.dot(&up).max(0.0).powi(3);
        Vec3::new(0.25 + 2.0 * lobe, 0.18 + 1.4 * lobe, 0.12 + 0.9 * lobe)
    })
}

/// 单像素多批次渲染 → (均值, 标准误)。只取 R 通道（本文件的灯都是灰的）。
///
/// 批次间用不同 `frame` 派生独立样本流（`SampleStream` 是
/// `hash(seed, node, frame, pixel, sample)` 的纯函数）⇒ 批次独立同分布，
/// 标准误可算，V4 的 3σ 判据才有意义。
fn pixel_batches(
    scene: &Scene,
    camera: &Camera,
    batches: u32,
    spp: u32,
    bounces: u32,
) -> (f32, f32) {
    let params = RenderParams {
        spp,
        max_bounces: bounces,
        clamp_indirect: None, // 参考对拍不许钳制（会引入偏置）
        nee_only: false,
    };
    let mut xs = Vec::with_capacity(batches as usize);
    for b in 0..batches {
        let ctx = EvalCtx::new(2026, 11, 1000 + b as u64);
        xs.push(render_pixel(scene, camera, &ctx, (0, 0), (1, 1), &params)[0]);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

// ===========================================================================
// V3 无偏：MIS 全开 vs 1e6 点均匀球面暴力积分
// ===========================================================================

/// `max_bounces = 2` 恰好让**两条策略都落地**：第 1 段走表面 NEE（CDF 策略，
/// 权重 `w_nee`），第 2 段的逃逸分支走 BSDF 策略（权重 `1 − w_nee`）。
/// 两者相加才是无偏的直接光照；只截一段会让 MIS 权重变成"少算"。
#[test]
fn v3_mis_matches_uniform_sphere_brute_force() {
    let (w, h) = (256u32, 128u32);
    let albedo = Vec3::splat(0.8);
    let mut scene = sphere_scene(MaterialKind::Lambert { albedo }, 1.0);
    scene.add_environment(w, h, &smooth_env(w, h));
    scene.assert_layout();

    // 相机在 +z 轴上看向原点：1×1 分辨率的中心像素命中 (0,0,1)，法线 +z。
    //
    // **fov 必须很小（0.2°）**：1×1 像素的抖动覆盖整个 fov——视场 45° 时
    // 这个"像素"实际是覆盖全半球的一束射线（一半直接逃逸到环境），
    // 参考点积分与它毫无关系（实测差 2.2×）。0.2° 下足迹侧向 3.5e-3、
    // 法线倾角 2e-3 rad，二阶效应 ~1e-5，才谈得上"逐点对拍"。
    let camera = cam(Vec3::new(0.0, 0.0, 3.0), Vec3::splat(0.0), 0.2);
    let n = Vec3::new(0.0, 0.0, 1.0);

    let params = RenderParams {
        spp: 8192,
        max_bounces: 2,
        clamp_indirect: None,
        nee_only: false,
    };
    let ctx = EvalCtx::new(2026, 11, 2);
    let got = render_pixel(&scene, &camera, &ctx, (0, 0), (1, 1), &params);

    let env = scene.env().expect("环境 CDF 必须已建");
    let table = TexTable::new(&scene.textures, &scene.texels);
    let e = brute_force_irradiance(env, table, n, 1_000_000);
    // Lambert：L_out = (albedo/π)·E
    let want = Vec3::new(e.x * albedo.x, e.y * albedo.y, e.z * albedo.z) * FRAC_1_PI;

    let mut worst = 0.0f32;
    for (g, t) in [(got[0], want.x), (got[1], want.y), (got[2], want.z)] {
        worst = worst.max(((g - t) / t).abs());
    }
    assert!(
        worst <= 5e-3,
        "V3 三通道最差相对误差 {worst:.3e} 超过 0.5%\
         （渲染 {got:?} vs 暴力参考 [{:.6}, {:.6}, {:.6}]）",
        want.x,
        want.y,
        want.z
    );
    println!("V3 无偏：MIS 8192 spp vs 1e6 点均匀球面暴力积分，最差相对误差 {worst:.3e}");
}

// ===========================================================================
// V4 互校：环境 ≡ 同辐亮度同立体角的 Light::Sphere（远距小锥）
// ===========================================================================

/// 环境亮帽 vs 球光源：同一着色点、同一辐亮度，立体角由**实测**等效角对齐。
///
/// # 为什么要反解 `θ_eff` 而不是直接写 15°
///
/// 亮帽边界在等距柱状上是一条**离散行界**：双线性取值把边界摊开约一个纹素，
/// 等效立体角与连续锥差 ~1e-2 相对量。那条差异是"环境图离散化"的固有属性，
/// 不是积分器的偏差。判据要求"同辐亮度同立体角"，故球光的 `R/D` 取
/// **实测** `sinθ_eff = √(E_env/(πL))`——两侧的目标辐照度因此逐位一致，
/// 剩下的任何差异只能来自采样/权重实现。
#[test]
fn v4_environment_matches_equivalent_sphere_light() {
    let (w, h) = (2048u32, 1024u32);
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let cos0 = 15.0f32.to_radians().cos();
    let cap = env_map(w, h, |u, v| {
        let d = uv_to_dir(u, v);
        if d.dot(&axis) >= cos0 {
            Vec3::splat(1.0)
        } else {
            Vec3::splat(0.0)
        }
    });
    let albedo = Vec3::splat(0.8);
    // fov 0.2°：理由同 V3（1×1 像素的抖动必须远小于"点"的邻域）。
    let camera = cam(Vec3::new(0.0, 0.0, 3.0), Vec3::splat(0.0), 0.2);
    let n = Vec3::new(0.0, 0.0, 1.0);

    let mut env_scene = sphere_scene(MaterialKind::Lambert { albedo }, 1.0);
    env_scene.add_environment(w, h, &cap);
    env_scene.assert_layout();

    let e_env = {
        let env = env_scene.env().expect("环境 CDF 必须已建");
        let table = TexTable::new(&env_scene.textures, &env_scene.texels);
        brute_force_irradiance(env, table, n, 4_000_000)
    };
    let sin_eff = (e_env.x / PI).clamp(0.0, 1.0).sqrt(); // L = 1

    // 球光：着色点 (0,0,1)，球心 (0,0,10) ⇒ 距离 9；R/D = sinθ_eff。
    let center = Vec3::new(0.0, 0.0, 10.0);
    let radius = 9.0f32 * sin_eff;
    let radiance = Vec3::splat(1.0);
    // 发光球必须**同时**是几何：否则 BSDF 策略取不到它，MIS 权重
    // `ph(pdf_light, pdf_bsdf)` 的另一半永远缺席 ⇒ 系统性欠估 ~0.5%
    // （那正是 V4 要测的 1% 量级）。这也是引擎的既有约定（发光球 +
    // `Light::Sphere` 成对注册）。
    let sph_scene = Scene {
        spheres: vec![
            (
                Sphere {
                    center: Vec3::splat(0.0),
                    radius: 1.0,
                },
                MaterialKind::Lambert { albedo },
            ),
            (
                Sphere { center, radius },
                MaterialKind::Emissive { radiance },
            ),
        ],
        lights: vec![Light::Sphere {
            center,
            radius,
            radiance,
        }],
        ..Default::default()
    };

    let (m_env, se_env) = pixel_batches(&env_scene, &camera, 16, 4096, 2);
    let (m_sph, se_sph) = pixel_batches(&sph_scene, &camera, 16, 4096, 2);
    let diff = (m_env - m_sph).abs();
    let rel = diff / (0.5 * (m_env + m_sph));
    let sigma = (se_env * se_env + se_sph * se_sph).sqrt();
    println!(
        "V4 互校：环境 {m_env:.6} ± {se_env:.2e} vs 球光 {m_sph:.6} ± {se_sph:.2e}\
         （θ_eff = {:.3}°，Δ = {diff:.3e} = {:.3}%，z = {:.2}）",
        sin_eff.asin().to_degrees(),
        100.0 * rel,
        if sigma > 0.0 { diff / sigma } else { 0.0 }
    );
    assert!(
        rel <= 1e-2,
        "V4 环境与球光相差 {:.3}%（{m_env} vs {m_sph}）超过 1%",
        100.0 * rel
    );
    if sigma > 0.0 {
        assert!(
            diff <= 3.0 * sigma,
            "V4 差异 {diff:.3e} 超出 3σ = {:.3e}（z = {:.2}）⇒ 不是统计涨落",
            3.0 * sigma,
            diff / sigma
        );
    } else {
        assert!(diff <= 1e-3 * m_env, "V4 零方差下仍有差异 {diff:.3e}");
    }
}

// ===========================================================================
// V1 第二条：`w_nee + w_bsdf ≡ 1` 逐点比特级
// ===========================================================================

/// MIS 互补性的**比特级**验证。
///
/// # 为什么能保证逐位等于 1.0（不是"约等于"）
///
/// * `w_nee = ph(a,b) = a²/(a²+b²)`；逃逸分支写的是 `1 − ph(a,b)`——
///   与 NEE 侧**同一个函数、同一组实参、同一顺序**（`power_heuristic`
///   对调实参也逐位相同：分母 `a²+b²` 与 `b²+a²` 是 IEEE 加法交换律下的
///   同一比特，分子各自精确平方）。
/// * 令 `u = fl(a²/(a²+b²)) ∈ [0,1]`、`v = fl(1−u)`，则
///   `fl(u+v) = 1.0`：`1−u` 的舍入误差 `|δ| ≤ ½ulp(1−u) ≤ 2⁻²⁵ = ½ulp(1)`，
///   故 `u+v = 1+δ` 落在 `[1−2⁻²⁵, 1+2⁻²⁵]`，最近舍入唯一落到 1.0
///   （恰好半个 ulp 的平局按"取偶"也落到 1.0）。
///
/// 断言不抽样：**全部 texel × 65 档备择 pdf** 逐对比较比特。
#[test]
fn v1_mis_weights_are_bitwise_complementary() {
    let (w, h) = (128u32, 64u32);
    let mut scene = Scene::default();
    scene.add_environment(w, h, &smooth_env(w, h)); // 非均匀 ⇒ pdf 逐 texel 不同
    let env = scene.env().expect("环境 CDF 必须已建");
    let light = Light::Environment { map: 1 };

    let one = 1.0f32.to_bits();
    let mut checked = 0u64;
    let (mut min_w, mut max_w) = (1.0f32, 0.0f32);
    for j in 0..h {
        for i in 0..w {
            let d = uv_to_dir((i as f32 + 0.5) / w as f32, (j as f32 + 0.5) / h as f32);
            let p_env = env.pdf_w(d);
            // 备择 pdf 取 Lambert 余弦半球族（真实路径上的 pdf_bsdf 形态），
            // 覆盖 0（半球边缘）与 1/π（正入射）两端。
            for k in 0..=64u32 {
                let p_bsdf = (k as f32 / 64.0) * FRAC_1_PI;
                let w_nee = nee_mis_weight(&light, p_env, p_bsdf);
                let w_bsdf = 1.0 - power_heuristic(p_env, p_bsdf); // 逃逸分支同表达式
                let sum = w_nee + w_bsdf;
                assert_eq!(
                    sum.to_bits(),
                    one,
                    "p_env={p_env:e} p_bsdf={p_bsdf:e} w_nee={w_nee:e} w_bsdf={w_bsdf:e}"
                );
                checked += 1;
                min_w = min_w.min(w_nee);
                max_w = max_w.max(w_nee);
            }
        }
    }
    // 平凡性反证：权重必须真的在动（全 1 或全 0.5 时上面的等式是废话）。
    assert!(
        min_w < 0.2 && max_w > 0.8,
        "w_nee 覆盖范围 [{min_w:.3e}, {max_w:.3e}] 太平凡，判据失去意义"
    );
    println!(
        "V1 MIS：{checked} 对 (pdf_env, pdf_bsdf)，w_nee+w_bsdf 逐对比特级 == 1.0；\
         w_nee ∈ [{min_w:.3e}, {max_w:.3e}]"
    );
}

// 说明：超越判据 #1（7 档 roughness × 3 档对比度 vs 4096 spp 暴力参考，
// 同表 C2 split-sum 实测）见 `w2_a2_ibl_beyond.rs`。
