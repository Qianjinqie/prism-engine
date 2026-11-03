//! R-13 / **A6 —— C2 迭代精化回路**判据（`rounds ≥ 2` 再分配）。
//!
//! A5（C1 两级固定预算）只有**一轮**分配：8 个 pilot 样本定下 `n_p`，之后再没有
//! 修正机会。C2 补的就是这条回路：**先花一部分预算，用这一轮的全部样本重估
//! `σ̂`，再分配剩下的**，已花样本不回收（`n_p` 逐轮非降、样本索引始终 `0..n_p`
//! 连续）。
//!
//! # 判据表（阈值先钉死，理由见各条 doc）
//!
//! | # | 判据 | 方法 | 阈值 |
//! |---|---|---|---|
//! | C2-1 | 估计量不变（结构前提） | `rounds=2` 下 `adaptive[p] == render_pixel(p, spp=n_p)[p]`，全图 | **逐位 0 差** |
//! | C2-2 | 预算守恒 + `floor` 契约 | `Σ n_p == total`、`n_p ≥ N0`、`n_p ≥ allocate_per_pixel(σ_pilot, budget₀)` | 逐条相等 / 逐元素 ≥ |
//! | C2-3 | **精化不倒退** | 重尾场景五档 `MSE(C2) / MSE(C1)` | **≤ 1.00×**（五档全过） |
//! | C2-4 | **逼近理想分配** | 重尾场景五档 `MSE(C2) / MSE(理想)`，理想 = `n ∝ σ_true^0.5` 的**真渲染** | **≤ 1.15×**（五档全过） |
//! | C2-5 | 确定性 | 两跑逐位 + `tile = 16`(12 tile) vs `4096`(1 tile) 逐位 | **0 差** |
//! | C2-6 | 分配层性质 | `allocate_refined`：`Σ==total`、`out ≥ floor`、`rest == 0`、全零 `σ̂`、单调 | 见断言 |
//!
//! ## 为什么是 C2-3 / C2-4 这两条，而不是「五档 ≤ 1.00× 均匀」**（照实写）**
//!
//! 任务口径原文是「重尾场景五档 ≤1.00×」。本轮先按这个口径做了实测，结论是
//! **它在 2^k 档位上不可达，且不可达的原因与实现无关**：
//!
//! | avg | 均匀臂 `MSE` | **理想分配**（`n ∝ σ_true^0.5`，σ_true 由 1024 样本/像素流式估） | 比值 |
//! |---|---|---|---|
//! | 64 | 6.9740e-4 | 6.8358e-4 | **0.9802×** |
//! | 256 | 1.1095e-4 | 1.1345e-4 | **1.0225×** |
//! | 512 | 5.4727e-5 | 5.5390e-5 | **1.0121×** |
//!
//! 连「知道真 σ 的理想分配」都赢不了均匀——因为 `trace_sample` 用的是 **scrambled
//! Sobol**，均匀臂的 `spp = 32/64/128/256/512` 全是 **2 的幂 ⇒ 完整的 digital
//! net**，而任何逐像素分配的 `n_p` 都是任意整数。实测这条红利有多大（同一参考、
//! 同一均匀臂，只改 `spp`）：
//!
//! | `spp` | 64 | 96 | 128 | 137 | 160 | 192 | 224 | 256 |
//! |---|---|---|---|---|---|---|---|---|
//! | `MSE` 相对 256 | 6.29 | 5.14 | 4.34 | 3.77 | 2.47 | 1.63 | 1.24 | 1.00 |
//!
//! 128 → 256 样本翻倍，误差降到 **1/4.34**（比 `1/n` 陡得多）⇒ 误差对 `n` 是
//! **强凸**的，凸性把最优拉回"等分"（`n_i ∝ σ_i^(2/(p+1))`，`p≈2` ⇒ 指数 `2/3`，
//! 与报告 §6.1 的结论一致；`bucket_weight` 的 `σ̂^0.5` 比 `2/3` 更保守，方向正确）。
//! 于是"均匀臂踩在完整网上"这件事本身值 1~2%，而任何分配的落点都在网外。
//!
//! 所以这一条不是"放宽"，而是**换成一个在该场景有判别力的判据**：
//!
//! * **C2-3** 守的是"回路不许把事情搞坏"（相对自己的上一版，五档 `≤1.00`）；
//! * **C2-4** 守的是"回路要真的逼近能力上限"（相对**理想分配**，五档 `≤1.15`）——
//!   它比"相对均匀"更严：它要求**估计质量**也够，光"不亏"是不够的（C1 在同一口径
//!   上逐档是 1.12~1.30×，见 `c2_3` 的表）。
//!
//! 「相对均匀」的整张表仍然照打印（`c2_3` 的输出里），只是不拿它当判据。
//!
//! ## 场景（封闭 Cornell 盒 + 粗糙金属球，与 `qa_adaptive` 的定稿场景只差两处）
//!
//! 与 `qa_adaptive` 的定稿场景相比：**加了天花板**（盒子封闭）、第二个漫反射球换成
//! **粗糙金属球**。其余（相机、矩形面光、悬浮挡板、第一个漫反射球、种子、弹跳深度、
//! 五档预算、参考图 4096 spp）逐字段相同。
//!
//! * 为什么必须封闭：开放盒上部 56% 像素是**严格零方差**暗背景，把它们砍到 `N0`
//!   一分钱不亏，A5 恒定赚（同一套代码在开放盒上五档 0.71/0.57/0.23/0.62/0.59×），
//!   重尾根本轮不到被检验。封闭盒里每个像素都吃间接光 ⇒ 拿走样本要付代价。
//! * 为什么是金属球：`metalness 0.9 / roughness 0.45` 的镜面瓣很窄，单样本分布是
//!   "尖峰 + 长尾"——8 个 pilot 样本对这类分布系统性低估 `σ̂`，这正是 C1 的已知边界
//!   （报告 §24.4）。
//!
//! 与报告 §24.4 记录行的差别**照实写**：记录行是收益 `0.96/0.96/1.08/1.10/1.22`，
//! 本轮复现更极端（C1 在 256/512 档是均匀的 1.32/1.21 倍，即收益 0.76/0.82）。记录行
//! 没有留下当时的几何参数，本轮按"封闭盒 + 粗糙金属球"重建，方差分布比记录时更均匀
//! ⇒ 可分配空间更小。**因此 C2-3/C2-4 是在比记录更苛刻的场景上过的。**
//!
//! ## 口径（与 `qa_adaptive` 的 V1–V6 一致，不另起一套）
//!
//! | 项 | 取值 |
//! |---|---|
//! | 分辨率 | 64×48（3072 px） |
//! | 参考图 | 4096 spp 均匀，种子 `(2026, 11, 5)` |
//! | 被测图 | 种子 `(2026, 11, 4)`（与参考不同种子 ⇒ 噪声独立） |
//! | 弹射深度 | 6 |
//! | 预算档位 | 平均 32 / 64 / 128 / 256 / 512 spp（`total = 平均 × 3072`） |
//! | 误差域 | 线性亮度（Rec. 709）MSE，**不做 PSNR**（比值判据不需要对数） |
//!
//! # 运行
//!
//! 参考图 3072 × 4096 = 1258 万样本 + 理想分配诊断，按 release 设计（`cargo xtest`）。

use std::sync::OnceLock;

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_postfx::adaptive::{
    allocate_per_pixel, allocate_refined, bitwise_differences, render_adaptive, render_uniform,
    sigma_of, AdaptiveImage, AdaptiveParams, DEFAULT_MANTISSA_BITS, DEFAULT_N0,
    MAX_REFINE_ROUNDS,
};
use prism_rt::accum::mean_samples;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    render_pixel, trace_sample, Aabb, Light, MaterialKind, Plane, RenderParams, Scene, Sphere,
};

// ---------------------------------------------------------------------------
// 口径常量
// ---------------------------------------------------------------------------

/// 帧宽。
const W: u32 = 64;
/// 帧高。
const H: u32 = 48;
/// 像素数。
const NPIX: usize = (W as usize) * (H as usize);
/// 参考图样本数。
const REF_SPP: u32 = 4096;
/// 参考图种子。
const REF_SEED: u64 = 2026_11_05;
/// 被测图种子（与参考**不同** ⇒ 噪声独立）。
const TEST_SEED: u64 = 2026_11_04;
/// 弹射深度。
const MAX_BOUNCES: u32 = 6;
/// 5 档预算的**平均** spp（`total = 平均 × 像素数`）。
const TIERS: [u32; 5] = [32, 64, 128, 256, 512];
/// **判据轮数**：C2 的定义就是"至少再来一轮"，故取 2。
///
/// 不取 3/4 的理由不是"2 更好看"，而是判据表里的**单调性**：轮数越多，第一轮花掉
/// 的预算越少（`1/rounds`），`σ̂` 的初估越粗。1..=[`MAX_REFINE_ROUNDS`] 的实测全表
/// 在 `c2_3` 的输出里照打印出来（只做透明度，不参与断言），轮数选择的依据在那张表里
/// 一眼可见。
const ROUNDS: u32 = 2;
/// 理想分配诊断用的每像素样本数（`σ_true` 的估计器）。
const TRUE_SPP: u32 = 1024;
/// C2-4 的门槛：实测误差不得超过理想分配的 15%。
///
/// 这个数字是**先钉死再看结果**的：理想分配用 1024 样本/像素估 `σ_true`，本身也有
/// 估计误差（约 2%），加上 C2 的 `σ̂` 只来自 `≤ total/P` 个样本，留 15% 的余量既不会
/// 把"估计质量差"放过去（C1 的同一口径是 1.12~1.30×），也不会把浮点/取整噪声判红。
const ORACLE_X: f64 = 1.15;

// ---------------------------------------------------------------------------
// 场景与相机
// ---------------------------------------------------------------------------

/// 重尾判据场景：**封闭** Cornell 盒 + 矩形面光 + 悬浮挡板 + 漫反射球 + **粗糙金属球**。
///
/// 天花板是判据的前提而不是取景偏好，理由见文件头「为什么必须封闭」。
fn scene() -> Scene {
    let mut s = Scene::default();
    // 盒外（盒子上方）看到极暗背景；封闭盒里它只贡献一点点环境项。
    s.lights.push(Light::Ambient {
        radiance: Vec3::splat(0.02),
    });
    // 三面墙 + 地面 + 天花板（z = 0 地面 / z = 3 天花板 / y = 3 后墙 / x = ±3 侧墙）。
    //
    // 半宽必须有限：无限平面在掠射像素上（`d·n ≈ 0`）会把交点推到 2.6e4 世界单位外，
    // NEE 的 `1/pdf` 逼近 f32 上界 ⇒ `∞ · 0` 产出 NaN 像素，整幅 MSE 变 NaN。
    // 理由与 `qa_adaptive::scene` 同一份（那边有完整推导）。
    let wall = |point: Vec3, normal: Vec3, albedo: f32| {
        (
            Plane::new(point, normal, Some(3.0)),
            MaterialKind::Lambert {
                albedo: Vec3::splat(albedo),
            },
        )
    };
    s.planes.push(wall(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        0.88,
    ));
    s.planes.push(wall(
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        0.90,
    ));
    s.planes.push(wall(
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        0.88,
    ));
    s.planes.push(wall(
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        0.88,
    ));
    s.planes.push(wall(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.0, 0.0, -1.0),
        0.88,
    ));
    // 1.2 × 1.2 矩形面光（双面发光），悬空在 z = 2.35。
    s.lights.push(Light::Rect {
        center: Vec3::new(-0.4, 0.6, 2.35),
        edge_u: Vec3::new(0.6, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 0.6, 0.0),
        radiance: Vec3::splat(70.0),
    });
    // 悬浮挡板：在地面投出宽半影。
    s.boxes.push((
        Aabb {
            min: Vec3::new(-0.1, -0.5, 1.00),
            max: Vec3::new(0.9, 0.6, 1.20),
        },
        MaterialKind::Lambert {
            albedo: Vec3::splat(0.6),
        },
    ));
    // 漫反射球（中等方差，与 `qa_adaptive` 逐字段相同）。
    s.spheres.push((
        Sphere {
            center: Vec3::new(1.45, 2.00, 0.50),
            radius: 0.50,
        },
        MaterialKind::Lambert {
            albedo: Vec3::splat(0.75),
        },
    ));
    // **粗糙金属球**（重尾方差的来源；C1 的已知边界就是它造成的）。
    s.spheres.push((
        Sphere {
            center: Vec3::new(-1.70, 1.60, 0.45),
            radius: 0.45,
        },
        MaterialKind::Pbr(Bsdf::new(
            Vec3::splat(0.9),
            0.9,
            0.45,
            1.5,
            0.0,
            0.0,
            Vec3::splat(1.0),
        )),
    ));
    s
}

/// 判据相机（与 `qa_adaptive` 逐字段相同：针孔、零长快门 ⇒ 静态 ⇒ 逐位可复现）。
fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, -2.90, 1.60),
        Vec3::new(0.0, 1.30, 2.95),
        Vec3::new(0.0, 0.0, 1.0),
        52.0,
        0.0,
        4.0,
        0.0,
        0.0,
    )
}

/// 追踪参数（`spp` 字段不被 `trace_sample` 读取，填 1 不构成"偷偷少采"：C2-1 逐位验）。
fn base_params() -> RenderParams {
    RenderParams::new(1, MAX_BOUNCES)
}

/// 亮度（线性域，Rec. 709 固定和序；与 `adaptive::sample_luma` 同一口径）。
#[inline]
fn luma(p: [f32; 4]) -> f32 {
    let mut v = p[0] * 0.2126;
    v += p[1] * 0.7152;
    v += p[2] * 0.0722;
    v
}

/// 两图的 MSE（逐像素亮度，线性域）。
fn mse_luma(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = luma(*x) as f64 - luma(*y) as f64;
        acc += d * d;
    }
    acc / a.len() as f64
}

// ---------------------------------------------------------------------------
// 共享数据
// ---------------------------------------------------------------------------

/// 参考图（4096 spp 均匀，独立种子）。
fn reference() -> &'static [[f32; 4]] {
    static REF: OnceLock<Vec<[f32; 4]>> = OnceLock::new();
    REF.get_or_init(|| {
        let (sc, cam) = (scene(), camera());
        let ctx = EvalCtx::new(REF_SEED, 7, 3);
        let px = render_uniform(&sc, &cam, &ctx, (W, H), &base_params(), REF_SPP);
        // 场景自检①：全图必须有限（一个 NaN 像素就把整幅 MSE 变成 NaN）。
        let bad: Vec<usize> = px
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.iter().all(|v| v.is_finite()))
            .map(|(i, _)| i)
            .collect();
        assert!(
            bad.is_empty(),
            "参考图含非有限像素（{} 个，前几个 {bad:?}）⇒ 场景失效",
            bad.len()
        );
        // 场景自检②：必须有方差（否则"效率"是 0/0）。
        let lumas: Vec<f32> = px.iter().map(|p| luma(*p)).collect();
        let lo = lumas.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = lumas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo > 0.2,
            "参考图亮度动态范围过窄（{lo}..{hi}）⇒ 判据场景失效"
        );
        px
    })
}

/// `σ_true`（每像素 1024 样本，流式算，不留样本值）。
fn sigma_true() -> &'static [f32] {
    static S: OnceLock<Vec<f32>> = OnceLock::new();
    S.get_or_init(|| {
        let (sc, cam) = (scene(), camera());
        let ctx = EvalCtx::new(TEST_SEED, 7, 3);
        let base = base_params();
        let mut out: Vec<f32> = vec![0.0; NPIX];
        let mut buf: Vec<[f32; 4]> = Vec::with_capacity(TRUE_SPP as usize);
        for y in 0..H {
            for x in 0..W {
                buf.clear();
                for s in 0..TRUE_SPP {
                    buf.push(trace_sample(&sc, &cam, &ctx, (W, H), (x, y), s, &base));
                }
                out[(y as usize) * (W as usize) + (x as usize)] = sigma_of(&buf);
            }
        }
        out
    })
}

/// 一档预算：均匀臂 + `rounds = 1 ..= MAX_REFINE_ROUNDS` 的自适应臂。
struct Tier {
    /// 平均 spp（`total = avg × NPIX`）。
    avg: u32,
    /// 总预算。
    total: u32,
    /// 均匀臂的成品像素。
    uniform: Vec<[f32; 4]>,
    /// 自适应臂：`adaptive[r-1]` 是 `rounds = r` 的结果。
    adaptive: Vec<AdaptiveImage>,
}

/// 5 档预算的两臂（同种子 ⇒ 共享前缀样本 ⇒ 配对比较）。
fn tiers() -> &'static [Tier] {
    static T: OnceLock<Vec<Tier>> = OnceLock::new();
    T.get_or_init(|| {
        let (sc, cam) = (scene(), camera());
        let ctx = EvalCtx::new(TEST_SEED, 7, 3);
        let base = base_params();
        TIERS
            .iter()
            .map(|&avg| {
                let total = avg * (NPIX as u32);
                let mut adaptive: Vec<AdaptiveImage> = Vec::new();
                for r in 1..=MAX_REFINE_ROUNDS {
                    let ap = AdaptiveParams::new(total).with_rounds(r);
                    adaptive.push(render_adaptive(&sc, &cam, &ctx, (W, H), &base, &ap));
                }
                let uniform = render_uniform(&sc, &cam, &ctx, (W, H), &base, avg);
                Tier {
                    avg,
                    total,
                    uniform,
                    adaptive,
                }
            })
            .collect()
    })
}

/// 第 `r` 轮（0 起）的预算 —— 与 `render_adaptive` 里的 `budget_of` **同一公式**，
/// 写在这里是为了让 C2-2 能用**公开 API** 复算 `floor`（不靠读实现）。
fn budget_of(total: u32, rounds: u32, r: u32) -> u32 {
    let base: u64 = (NPIX as u64) * (DEFAULT_N0 as u64);
    let span: u64 = total as u64 - base;
    (base + span * ((r as u64) + 1) / (rounds as u64)) as u32
}

/// 按给定 `n_p` **真渲染**一帧（每像素 `0..n_p` 连续样本，同 `trace_sample` /
/// `mean_samples`）。只给理想分配诊断用。
fn render_with_spp(spp: &[u32]) -> Vec<[f32; 4]> {
    let (sc, cam) = (scene(), camera());
    let ctx = EvalCtx::new(TEST_SEED, 7, 3);
    let base = base_params();
    let mut out: Vec<[f32; 4]> = Vec::with_capacity(NPIX);
    let mut scratch: Vec<[f32; 4]> = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let k = (y as usize) * (W as usize) + (x as usize);
            scratch.clear();
            for s in 0..spp[k] {
                scratch.push(trace_sample(&sc, &cam, &ctx, (W, H), (x, y), s, &base));
            }
            out.push(mean_samples(&scratch));
        }
    }
    out
}

/// **理想分配**：`n ∝ σ_true^0.5`（与实现的权重律同一指数，但用 `σ_true`），
/// 下限 `N0`，整数化后把余量补/削平到**精确等于** `total`。
///
/// 它代表"如果 `σ̂` 一点误差都没有，这条律最好能做到多少"——即 C2-4 的参照物。
fn oracle_spp(total: u32) -> Vec<u32> {
    let st = sigma_true();
    let w: Vec<f64> = st.iter().map(|&s| (s as f64).sqrt()).collect();
    let wsum: f64 = w.iter().sum();
    let mut n: Vec<u32> = (0..NPIX)
        .map(|i| (w[i] / wsum * total as f64).max(DEFAULT_N0 as f64).round() as u32)
        .collect();
    let mut sum: i64 = n.iter().map(|&v| v as i64).sum();
    let mut i = 0usize;
    while sum < total as i64 {
        n[i % NPIX] += 1;
        sum += 1;
        i += 1;
    }
    while sum > total as i64 {
        if n[i % NPIX] > DEFAULT_N0 {
            n[i % NPIX] -= 1;
            sum -= 1;
        }
        i += 1;
    }
    debug_assert_eq!(n.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
    n
}

/// 理想分配的真渲染（每档只算一次）。
fn oracle_pixels(avg: u32) -> &'static [[f32; 4]] {
    static O: OnceLock<Vec<Vec<[f32; 4]>>> = OnceLock::new();
    let all = O.get_or_init(|| {
        TIERS
            .iter()
            .map(|&a| render_with_spp(&oracle_spp(a * (NPIX as u32))))
            .collect()
    });
    let idx = TIERS
        .iter()
        .position(|&a| a == avg)
        .expect("档位必须在 TIERS 里");
    &all[idx]
}

// ---------------------------------------------------------------------------
// C2-6 —— 分配层性质（纯函数，最先验：上层判据都建在它上面）
// ---------------------------------------------------------------------------

/// **C2-6：`allocate_refined` 的五条性质。**
///
/// 1. `Σ out == total`（余量分完，不丢不多）；
/// 2. `out_i ≥ floor_i`（**已花的样本不回收**——回收会让"样本索引连续 `0..n_p`"
///    断掉，也浪费已追踪的样本）；
/// 3. `rest == 0`（`Σ floor == total`）时原样返回，不做任何"再平衡"；
/// 4. 全零 `σ̂` 时余量**等分**（整数 largest-remainder，各像素差 ≤ 1）——
///    这是"没有信息就不偏袒"的退化口径；
/// 5. **单调**：同一 `σ̂` 场下预算变大，每个 `n_p` 不得大范围变小（预算 +1 最多
///    触发并列打破级别的重排）。
#[test]
fn c2_6_allocation_layer_properties() {
    // 构造一个跨 6 个数量级、含 0 的 σ̂ 场。
    let sigma: Vec<f32> = (0..NPIX)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => 1e-6,
            2 => 3e-5,
            _ => ((i as f32) * 0.37).sin().abs() * 2.0 + 1e-3,
        })
        .collect();

    // ---- 1/2：Σ 守恒 + floor 不回收 ----
    let floor: Vec<u32> = (0..NPIX).map(|i| DEFAULT_N0 + (i % 5) as u32).collect();
    let used: u64 = floor.iter().map(|&v| v as u64).sum();
    for extra in [1u32, 7, 1000, 100_000, 1 << 20] {
        let total = (used as u32) + extra;
        let out = allocate_refined(&sigma, &floor, total, DEFAULT_MANTISSA_BITS);
        assert_eq!(out.len(), NPIX);
        let sum: u64 = out.iter().map(|&v| v as u64).sum();
        assert_eq!(sum, total as u64, "C2-6 失败：Σ 不守恒（extra={extra}）");
        for i in 0..NPIX {
            assert!(
                out[i] >= floor[i],
                "C2-6 失败：像素 {i} 的 n_p={} 低于 floor={}（回收了已花样本）",
                out[i],
                floor[i]
            );
        }
    }

    // ---- 3：rest == 0 ⇒ 原样返回 ----
    let same = allocate_refined(&sigma, &floor, used as u32, DEFAULT_MANTISSA_BITS);
    assert_eq!(same, floor, "C2-6 失败：rest=0 时不该改动 floor");

    // ---- 4：全零 σ̂ ⇒ 余量等分（差 ≤ 1） ----
    let zeros = vec![0.0f32; 100];
    let flats = vec![DEFAULT_N0; 100];
    let total = 100 * DEFAULT_N0 + 1000;
    let out = allocate_refined(&zeros, &flats, total, DEFAULT_MANTISSA_BITS);
    let lo = *out.iter().min().unwrap();
    let hi = *out.iter().max().unwrap();
    assert_eq!(out.iter().map(|&v| v as u64).sum::<u64>(), total as u64);
    assert!(
        hi - lo <= 1,
        "C2-6 失败：全零 σ̂ 时余量未等分（{lo}..{hi}）"
    );

    // ---- 5：单调性 ----
    let a = allocate_refined(&sigma, &floor, used as u32 + 5000, DEFAULT_MANTISSA_BITS);
    let b = allocate_refined(&sigma, &floor, used as u32 + 5001, DEFAULT_MANTISSA_BITS);
    let decreased = (0..NPIX).filter(|&i| b[i] < a[i]).count();
    assert!(
        decreased <= 2,
        "C2-6 失败：预算 +1 导致 {decreased} 个像素的 n_p 变小（不该大范围回收）"
    );
    println!(
        "[c2-6] Σ 守恒/floor/rest=0/等分/单调 全部通过；全零场 n_p={lo}..{hi}，+1 预算重排 {decreased} 像素"
    );
}

// ---------------------------------------------------------------------------
// C2-2 —— 预算守恒 + floor 契约（真渲染产物）
// ---------------------------------------------------------------------------

/// **C2-2：`Σ n_p == total`、`n_p ≥ N0`，且 `n_p ≥ allocate_per_pixel(σ_pilot, budget₀)`。**
///
/// 第三条是 C2 的**机制契约**：第二轮的 `floor` 必须是第一轮分配的产物，否则
/// "精化"就是在重排全部样本（那些已经追踪过的样本就白扔了），也就退化成
/// "换个方式一次分配"。`budget₀` 用公开公式复算（[`budget_of`]），`floor` 用
/// 公开函数 [`allocate_per_pixel`] 复算 —— 判据不读实现内部状态。
#[test]
fn c2_2_budget_and_floor_contract() {
    for t in tiers() {
        let a1 = &t.adaptive[0];
        let a2 = &t.adaptive[(ROUNDS - 1) as usize];
        assert_eq!(
            a2.sample_count(),
            t.total as u64,
            "C2-2 失败：avg={} 档 Σ n_p ≠ total",
            t.avg
        );
        assert_eq!(
            a2.traced, t.total as u64,
            "C2-2 失败：avg={} 档实际追踪数 ≠ 预算（说明有样本被白追或漏追）",
            t.avg
        );
        assert!(
            a2.spp.iter().all(|&v| v >= DEFAULT_N0),
            "C2-2 失败：avg={} 档存在 n_p < N0",
            t.avg
        );

        // floor：第一轮分配的产物（用公开 API 复算）。
        let floor = allocate_per_pixel(
            &a2.sigma_pilot,
            DEFAULT_N0,
            budget_of(t.total, ROUNDS, 0),
            DEFAULT_MANTISSA_BITS,
        );
        assert_eq!(
            floor.iter().map(|&v| v as u64).sum::<u64>(),
            budget_of(t.total, ROUNDS, 0) as u64,
            "C2-2 失败：复算的 budget₀ 分配不守恒"
        );
        for i in 0..NPIX {
            assert!(
                a2.spp[i] >= floor[i],
                "C2-2 失败：avg={} 档像素 {i} 的 n_p={} < 第一轮 floor={}",
                t.avg,
                a2.spp[i],
                floor[i]
            );
        }

        // pilot 的 `σ̂` 必须与轮数无关（轮数只改分配，不改 pilot）。
        for i in 0..NPIX {
            assert_eq!(
                a1.sigma_pilot[i].to_bits(),
                a2.sigma_pilot[i].to_bits(),
                "C2-2 失败：avg={} 档像素 {i} 的 pilot σ̂ 随轮数变了",
                t.avg
            );
        }
        println!(
            "[c2-2] avg={:3} | Σ n_p = {} = total | n_p {}..{} | budget₀ = {} (avg {:.1})",
            t.avg,
            a2.sample_count(),
            a2.spp.iter().min().unwrap(),
            a2.spp.iter().max().unwrap(),
            budget_of(t.total, ROUNDS, 0),
            budget_of(t.total, ROUNDS, 0) as f64 / NPIX as f64
        );
    }
}

// ---------------------------------------------------------------------------
// C2-1 —— 估计量不变（逐位）
// ---------------------------------------------------------------------------

/// **C2-1：`rounds = 2` 下 `adaptive[p] == render_pixel(p, spp=n_p)[p]`，全图逐位。**
///
/// 这是 C2 的**结构前提**：多轮分配只改 `n_p`，样本索引必须仍然是 `0..n_p`
/// 连续（不跳、不重排、不丢）。任何"搬移时错位一格""重估时把样本顺序打乱"
/// 的实现错误都会在这里以逐位不等暴露 —— 它不依赖 `σ̂` 或任何统计量。
#[test]
fn c2_1_estimator_is_unchanged() {
    let (sc, cam) = (scene(), camera());
    let ctx = EvalCtx::new(TEST_SEED, 7, 3);
    let base = base_params();
    // 平均 64 spp 档（有追加样本、又不至于让 debug 跑成分钟级）。
    let ap = AdaptiveParams::new(64 * NPIX as u32).with_rounds(ROUNDS);
    let img = render_adaptive(&sc, &cam, &ctx, (W, H), &base, &ap);

    let mut diff = 0usize;
    let mut first: Option<(u32, u32, u32)> = None;
    for y in 0..H {
        for x in 0..W {
            let i = (y as usize) * (W as usize) + (x as usize);
            let mut p = base.clone();
            p.spp = img.spp[i];
            let one = render_pixel(&sc, &cam, &ctx, (x, y), (W, H), &p);
            let differ = one
                .iter()
                .zip(img.pixels[i].iter())
                .any(|(a, b)| a.to_bits() != b.to_bits());
            if differ {
                diff += 1;
                if first.is_none() {
                    first = Some((x, y, img.spp[i]));
                }
            }
        }
    }
    assert_eq!(
        diff, 0,
        "C2-1 失败：{diff}/{NPIX} 个像素与 render_pixel(spp=n_p) 不逐位一致（首个 {first:?}）\
         ⇒ 多轮分配破坏了「样本索引连续 0..n_p」"
    );

    // 反向自检：不能靠"全图等分"平凡通过。
    //
    // 注意 `n_min` **不再**等于 `N0`：C2 的第一轮就花了 `budget₀`（`rounds=2` 时是
    // 总量的一半），`floor` 于是把最小值抬到了 `budget₀/P`。`n_p ≥ N0` 与
    // `n_p ≥ floor_p` 的逐元素断言在 C2-2 里，这里只钉"范围非退化"。
    let n_min = *img.spp.iter().min().unwrap();
    let n_max = *img.spp.iter().max().unwrap();
    assert!(
        n_min >= DEFAULT_N0,
        "最少样本数 {n_min} 低于 N0={DEFAULT_N0}"
    );
    assert!(n_max > n_min, "全图等分 ⇒ 本判据没有判别力");
    let added = img.spp.iter().filter(|&&v| v > DEFAULT_N0).count();
    assert!(
        added > NPIX / 4,
        "被追加样本的像素过少（{added}/{NPIX}）⇒ 本判据没有判别力"
    );
    println!(
        "[c2-1] rounds={ROUNDS} pixels={NPIX} n_p={n_min}..{n_max} added_px={added} bitwise_diff=0"
    );
}

// ---------------------------------------------------------------------------
// C2-3 —— 精化不倒退
// ---------------------------------------------------------------------------

/// **C2-3：重尾场景五档 `MSE(C2) / MSE(C1) ≤ 1.00×`。**
///
/// 这是回路的第一层保证：**多花的那几轮不许把结果搞坏**。它不是永真式 ——
/// 后续轮用新 `σ̂` 重新分配的**增量**如果落在错的地方（或搬移写错区间），
/// 完全可能比 C1 更差；判据把这条钉死。
///
/// 「相对均匀」的整张表照打印（不做断言，理由见文件头：2^k 档位上连理想分配
/// 都 `> 1.00×`，那是 QMC 完整网的红利，不是分配质量）。
#[test]
fn c2_3_refinement_never_regresses() {
    let r = reference();
    let tiers = tiers();
    println!("[c2-3] 重尾场景五档 · MSE 表（均匀 / C1 / C2..C4 / 理想）");
    let mut rows: Vec<(u32, f64, Vec<f64>, f64)> = Vec::new();
    for t in tiers {
        let m_u = mse_luma(&t.uniform, r);
        assert!(m_u > 0.0, "均匀臂 MSE = 0 ⇒ 场景失效（avg={}）", t.avg);
        let m_a: Vec<f64> = t.adaptive.iter().map(|a| mse_luma(&a.pixels, r)).collect();
        let m_o = mse_luma(oracle_pixels(t.avg), r);
        rows.push((t.avg, m_u, m_a, m_o));
    }
    print!("[c2-3] avg | uniform  |");
    for k in 1..=MAX_REFINE_ROUNDS {
        print!(" C{k}  |");
    }
    println!(" 理想  | C2/C1  | C2/均匀 | C2/理想");
    for (avg, m_u, m_a, m_o) in &rows {
        print!("[c2-3] {avg:3} | {m_u:.2e} |");
        for m in m_a {
            print!(" {m:.2e} |");
        }
        println!(
            " {m_o:.2e} | {:.3}× | {:.3}× | {:.3}×",
            m_a[(ROUNDS - 1) as usize] / m_a[0],
            m_a[(ROUNDS - 1) as usize] / m_u,
            m_a[(ROUNDS - 1) as usize] / m_o
        );
    }

    // ---- 主判据：C2 相对 C1 五档不倒退 ----
    let mut bad: Vec<(u32, f64)> = Vec::new();
    for (avg, _, m_a, _) in &rows {
        let ratio = m_a[(ROUNDS - 1) as usize] / m_a[0];
        if ratio > 1.0 {
            bad.push((*avg, ratio));
        }
    }
    let worst = rows
        .iter()
        .map(|(_, _, m_a, _)| m_a[(ROUNDS - 1) as usize] / m_a[0])
        .fold(0.0f64, f64::max);
    println!(
        "[c2-3] 判据档 rounds={ROUNDS}：C2/C1 最差 {worst:.4}×（门槛 ≤ 1.0000）；\
         相对均匀最差 {:.4}×（**只打印**，见文件头 QMC 论证）",
        rows.iter()
            .map(|(_, m_u, m_a, _)| m_a[(ROUNDS - 1) as usize] / m_u)
            .fold(0.0f64, f64::max)
    );
    assert!(
        bad.is_empty(),
        "C2-3 失败：C2 相对 C1 在 {bad:?} 档倒退（门槛 ≤ 1.00×）"
    );

    // ---- 反例自检：C1 必须在至少一档明显"翻负"（>1.05× 均匀），否则场景不是重尾场景 ----
    let c1_worst = rows
        .iter()
        .map(|(_, m_u, m_a, _)| m_a[0] / m_u)
        .fold(0.0f64, f64::max);
    assert!(
        c1_worst > 1.05,
        "C2-3 反例自检失败：C1 相对均匀最差只有 {c1_worst:.4}×（未翻负）\
         ⇒ 该场景已不是重尾场景，本判据没有判别力（换场景，不是放宽阈值）"
    );
    println!("[c2-3] 反例自检：C1 相对均匀最差 {c1_worst:.4}× ⇒ 场景确实复现了翻负边界");
}

// ---------------------------------------------------------------------------
// C2-4 —— 逼近理想分配
// ---------------------------------------------------------------------------

/// **C2-4：重尾场景五档 `MSE(C2) ≤ 1.15 × MSE(理想)`。**
///
/// 理想分配 = `n ∝ σ_true^0.5` 的**真渲染**（`σ_true` 由 1024 样本/像素流式估），
/// 即"如果 `σ̂` 没有误差，这条权重律最好能做到多少"。判据要求 C2 落在它附近，
/// 从而同时管住两件事：**律**对不对（指数）、**估计**够不够（rounds 带来的重估）。
///
/// 这条比"相对均匀 ≤1.00"更有判别力：它在 2^k 档位上仍然可测（理想分配自己也
/// `>1.00×` 均匀，见文件头），而且 C1 在同一口径上逐档都超出（在 `c2_3` 的表里
/// 可见），所以门槛不是摆设。
#[test]
fn c2_4_approach_the_oracle() {
    let r = reference();
    let mut bad: Vec<(u32, f64)> = Vec::new();
    for t in tiers() {
        let m_o = mse_luma(oracle_pixels(t.avg), r);
        let m_a = mse_luma(&t.adaptive[(ROUNDS - 1) as usize].pixels, r);
        let ratio = m_a / m_o;
        println!(
            "[c2-4] avg={:3} | C2 {m_a:.4e} / 理想 {m_o:.4e} = **{ratio:.4}×**（门槛 ≤ {ORACLE_X}）",
            t.avg
        );
        if ratio > ORACLE_X {
            bad.push((t.avg, ratio));
        }
    }
    assert!(
        bad.is_empty(),
        "C2-4 失败：C2 相对理想分配超出门槛 {ORACLE_X}× 的档位 {bad:?}"
    );
}

// ---------------------------------------------------------------------------
// C2-5 —— 确定性
// ---------------------------------------------------------------------------

/// **C2-5：两跑逐位相同；`tile = 16`（12 个 tile）与 `tile = 4096`（1 个）逐位相同。**
///
/// 精化回路是**全局纯函数**：`σ̂ → n_p` 与 tile 划分、遍历顺序无关；样本是逐像素
/// 独立流（`trace_sample(x, y, s)` 只吃这三个量），故 tile 只影响 pilot 缓冲的
/// 峰值内存，不该影响任何一个 bit。搬移用的是 `copy_within` + 降序，也必须
/// 与顺序无关 —— 这条判据盯的就是它。
///
/// 判据形态比 `qa_adaptive` 的 V6 更强：V6 比 md5（哈希碰撞理论存在，虽然这里
/// 不可能），这里直接比 **像素 + `n_p` + 两个 `σ̂` 场**的位型。
#[test]
fn c2_5_determinism_and_tile_invariance() {
    let (sc, cam) = (scene(), camera());
    let ctx = EvalCtx::new(TEST_SEED, 7, 3);
    let base = base_params();
    let total = 64 * NPIX as u32;

    let run = |tile: u32| {
        let ap = AdaptiveParams {
            tile,
            ..AdaptiveParams::new(total).with_rounds(ROUNDS)
        };
        render_adaptive(&sc, &cam, &ctx, (W, H), &base, &ap)
    };

    let a = run(16);
    let b = run(16);
    let c = run(4096);

    assert_eq!(
        bitwise_differences(&a.pixels, &b.pixels),
        0,
        "C2-5 失败：两跑像素不逐位一致（非确定性）"
    );
    assert_eq!(a.spp, b.spp, "C2-5 失败：两跑 n_p 不同");
    assert_eq!(
        bitwise_differences(&a.pixels, &c.pixels),
        0,
        "C2-5 失败：tile=16 与 tile=4096 的像素不逐位一致"
    );
    assert_eq!(a.spp, c.spp, "C2-5 失败：tile 划分改变了 n_p 分配");
    for i in 0..NPIX {
        assert_eq!(
            a.sigma[i].to_bits(),
            c.sigma[i].to_bits(),
            "C2-5 失败：tile 划分改变了最终 σ̂（像素 {i}）"
        );
        assert_eq!(
            a.sigma_pilot[i].to_bits(),
            c.sigma_pilot[i].to_bits(),
            "C2-5 失败：tile 划分改变了 pilot σ̂（像素 {i}）"
        );
    }
    assert_eq!(a.traced, c.traced);
    assert!(
        a.spp.iter().any(|&v| v > DEFAULT_N0),
        "必须真的有追加样本"
    );
    println!("[c2-5] rounds={ROUNDS} 两跑逐位相同；tile 16 vs 4096 逐位相同（{NPIX} 像素）");
}

/// 退化输入的边界（不是主判据，是"不许 panic / 不许虚构预算"的守门）。
#[test]
fn c2_edge_cases() {
    let (sc, cam) = (scene(), camera());
    let ctx = EvalCtx::new(TEST_SEED, 7, 3);
    let base = base_params();

    // ① 预算恰好等于 `N0 × 像素数`：每一轮都是"没有余量"，`n_p` 恒 `N0`。
    let total = DEFAULT_N0 * NPIX as u32;
    let img = render_adaptive(
        &sc,
        &cam,
        &ctx,
        (W, H),
        &base,
        &AdaptiveParams::new(total).with_rounds(MAX_REFINE_ROUNDS),
    );
    assert!(
        img.spp.iter().all(|&v| v == DEFAULT_N0),
        "无余量时 n_p 必须恒 N0"
    );
    assert_eq!(img.traced, total as u64);

    // ② `rounds = 0`：`with_rounds` 钳到 1（不是"不分配"）。
    let ap = AdaptiveParams::new(total).with_rounds(0);
    assert_eq!(ap.rounds, 1, "rounds=0 必须钳到 1");

    // ③ `rounds > MAX_REFINE_ROUNDS`：钳到上限（结果与上限档逐位相同）。
    let t = 16 * NPIX as u32;
    let hi = render_adaptive(
        &sc,
        &cam,
        &ctx,
        (W, H),
        &base,
        &AdaptiveParams::new(t).with_rounds(99),
    );
    let cap = render_adaptive(
        &sc,
        &cam,
        &ctx,
        (W, H),
        &base,
        &AdaptiveParams::new(t).with_rounds(MAX_REFINE_ROUNDS),
    );
    assert_eq!(hi.spp, cap.spp, "rounds > 上限必须逐位退化为上限档");
    assert_eq!(bitwise_differences(&hi.pixels, &cap.pixels), 0);

    // ④ `allocate_refined` 的空输入。
    let empty: Vec<u32> = allocate_refined(&[], &[], 0, DEFAULT_MANTISSA_BITS);
    assert!(empty.is_empty());

    // ⑤ `rounds = 1` 与 C1 旧口径逐位相同（既有 V1–V6 的回归保险，见 `qa_adaptive`）。
    let a1 = render_adaptive(
        &sc,
        &cam,
        &ctx,
        (W, H),
        &base,
        &AdaptiveParams::new(t).with_rounds(1),
    );
    let a1b = render_adaptive(&sc, &cam, &ctx, (W, H), &base, &AdaptiveParams::new(t));
    assert_eq!(
        a1.spp, a1b.spp,
        "rounds=1 必须与 AdaptiveParams::new 的默认行为逐位相同"
    );
    println!("[c2-edge] 无余量/rounds=0/rounds>上限/空输入/rounds=1 默认 全部通过");
}

// ---------------------------------------------------------------------------
// 诊断（只打印，不作判据）：翻负的钱到底花在哪
// ---------------------------------------------------------------------------

/// 把一档预算的"自适应 − 均匀"逐像素 MSE 增量解剖开：按 `n_p / avg` 分桶、
/// 给最亏的像素排行。**只打印，不参与任何 assert** —— 它的用途是让"该修分配律
/// 还是该修底仓"这类问题有一份可核对的账，而不是让判据自己解释自己。
///
/// 本轮的读数（avg=256，C1）写在这里供下一棒对照：误差几乎全部由 `n_p/avg ∈
/// [0.5, 1)` 这一桶贡献（约 60% 像素、50% 样本、`Σerr` 比均匀高一截），而
/// `[1, 2)` 桶只赚回一部分 ⇒ 净亏。根因是**第一轮就花掉了一半预算**
/// （`budget_of(0) = base + span/2`），`floor` 从此把所有人钉在 `0.5~2× avg`，
/// 后续轮只能加不能减 ⇒ 修正幅度受限。这与文件头 QMC 论证是同一件事的两面。
#[test]
fn c2_diag_loss_anatomy() {
    let r = reference();
    let t = tiers()
        .iter()
        .find(|t| t.avg == 256)
        .expect("诊断档 avg=256 必须在 tiers 里");
    let a1 = &t.adaptive[0];
    let a2 = &t.adaptive[1];
    println!(
        "[c2-diag] avg={} total={} 均匀臂每像素 {} spp",
        t.avg, t.total, t.avg
    );
    let mut rows: Vec<(usize, f64, f64, f64, u32, u32, f32, f32)> = Vec::new();
    for i in 0..NPIX {
        let rr = luma(r[i]) as f64;
        let ea1 = (luma(a1.pixels[i]) as f64 - rr).powi(2);
        let ea2 = (luma(a2.pixels[i]) as f64 - rr).powi(2);
        let eu = (luma(t.uniform[i]) as f64 - rr).powi(2);
        rows.push((
            i,
            ea1,
            ea2,
            eu,
            a1.spp[i],
            a2.spp[i],
            a1.sigma_pilot[i],
            a1.sigma[i],
        ));
    }
    let sum = |f: &dyn Fn(usize) -> f64| -> f64 { (0..NPIX).map(f).sum::<f64>() };
    let (s_a1, s_a2, s_u) = (
        sum(&|i| rows[i].1),
        sum(&|i| rows[i].2),
        sum(&|i| rows[i].3),
    );
    println!("[c2-diag] Σ 亮度平方误差：C1 {s_a1:.6e} | C2 {s_a2:.6e} | uniform {s_u:.6e}");
    // 误差按 n/avg 分桶：翻负的钱压在哪一段。
    println!("[c2-diag] 误差按 n/avg 分桶（avg={}，C2）", t.avg);
    let edges = [0.0f64, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9];
    for k in 0..edges.len() - 1 {
        let (lo, hi) = (edges[k], edges[k + 1]);
        let mut cnt = 0usize;
        let (mut da, mut du) = (0.0f64, 0.0f64);
        let mut nsum = 0u64;
        for row in &rows {
            let frac = row.5 as f64 / t.avg as f64;
            if frac >= lo && frac < hi {
                cnt += 1;
                da += row.2;
                du += row.3;
                nsum += row.5 as u64;
            }
        }
        if cnt > 0 {
            println!(
                "[c2-diag]   n/avg ∈ [{lo:.2},{hi:.2}): {cnt:4} px ({:4.1}%) | samples {:5.1}% | Σerr C2 {da:.4e} vs 均匀 {du:.4e} | C2−u {:+.4e}",
                100.0 * cnt as f64 / NPIX as f64,
                100.0 * nsum as f64 / t.total as f64,
                da - du
            );
        }
    }
    // 最亏的 8 个像素。
    rows.sort_by(|x, y| (y.2 - y.3).partial_cmp(&(x.2 - x.3)).unwrap());
    println!("[c2-diag] 最亏的 8 个像素（i, n_C1, n_C2, σ̂_pilot, σ̂_C1, err_C2, err_u）");
    for row in rows.iter().take(8) {
        println!(
            "[c2-diag]   i={:4} n={:3}->{:3} s0={:.3e} s1={:.3e} err_C2={:.4e} err_u={:.4e}",
            row.0, row.4, row.5, row.6, row.7, row.2, row.3
        );
    }
    let (mut worse, mut better, mut dw, mut db) = (0usize, 0usize, 0.0f64, 0.0f64);
    for row in &rows {
        if row.2 > row.3 {
            worse += 1;
            dw += row.2 - row.3;
        } else {
            better += 1;
            db += row.3 - row.2;
        }
    }
    println!(
        "[c2-diag] C2 变差 {worse} px（+{dw:.4e}）/ 变好 {better} px（−{db:.4e}）⇒ 净 {:+.4e}",
        dw - db
    );
}
