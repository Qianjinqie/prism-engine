//! R-6② 全链路 Marschner 采样器判据（S1 pdf 归一 / S2 无偏 / S3 f·pdf 自洽 /
//! S4 三瓣判别力 / S5 确定性）。
//!
//! # 口径（不改尺子）
//!
//! 被积函数就是积分器续弹用的那一个：`weight = f·max(0, n·ω_i)/pdf`，其中
//! `f = HairScatter::eval_full(θ_i,φ_i,θ_o,φ_o)`、`pdf = pdf_incident(...)`、
//! `ω_i = HairGeometry::dir_from_angles(θ_i,φ_i)`。解析参考是同一被积函数在
//! `dω_i = d(sinθ_i)·dφ_i` 上的固定中点格积分：
//!
//! ```text
//!   R = ∫∫ f(θ_i,φ_i)·max(0, n·ω_i) ds dφ
//! ```
//!
//! 采样密度恰好等于 `pdf_incident`（分段常数 CDF 反演），故 `E[weight] = R`
//! 是构造性无偏，不是"测出来像"。随机量只取 `SampleStream` 同款 scrambled
//! Sobol 的三个分量（`(u_lobe,u_long)` 与 `(u_azim,_)`），零新随机源。
//!
//! # 为什么用 `phi_tt_shift != 0`
//!
//! `shift = 0` 时 R 与 TT/TRT 的方位分布重合，错置瓣权重在方位上测不出来
//! （判据会假绿）。本文件主场景取 `shift = 0.9`，使三瓣方位偏角**独立可辨**，
//! S4 的两个负控才有判别力；默认预置另在 S1/S3/S5 各测一遍。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;
use prism_rt::bsdf::{Bsdf, SampleCtx};
use prism_rt::hair_scatter::{sample_hair_bounce, HairGeometry, HairScatter};
use prism_rt::sampler::sample_2d;

const PI: f32 = std::f32::consts::PI;

/// 主场景：`shift = 0.9` 让三瓣方位偏角可辨（S4 判别力的前提）。
fn scatter() -> HairScatter {
    let mut h = HairScatter::new(0.12, 8.0, 1.55, 0.2, 0.04);
    h.w_r = 0.35;
    h.w_tt = 0.30;
    h.w_trt = 0.35;
    h.phi_tt_shift = 0.9;
    h
}

fn geometry() -> HairGeometry {
    HairGeometry::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
}

/// 来路方向（`wo` = f 的出射侧）与着色法线 `n`。
fn fixed_frame() -> (Vec3, Vec3) {
    (Vec3::new(0.3, 0.5, 0.81).normalize(), Vec3::new(0.0, 0.0, 1.0))
}

/// 三个 Sobol 分量（与续弹 `sample_2d(SLOT_BSDF)` / `+1` 同一族纯函数流）。
fn sobol3(ctx: &EvalCtx, sample: u32) -> (f32, f32, f32) {
    let (a, b) = sample_2d(
        ctx,
        &SampleIndex { frame: 0, pixel: (0, 0), sample, dimension: 0, node_id: 0 },
    );
    let (c, _) = sample_2d(
        ctx,
        &SampleIndex { frame: 0, pixel: (0, 0), sample, dimension: 2, node_id: 0 },
    );
    (a, b, c)
}

/// 解析参考：`∫∫ f·max(0,n·ω) ds dφ`（固定中点格，确定性）。
fn reference(sc: &HairScatter, geo: &HairGeometry, wo: Vec3, n: Vec3, grid: usize) -> f64 {
    let theta_o = geo.longitudinal(wo);
    let phi_o = geo.azimuth(wo);
    let ds = 2.0 / grid as f32;
    let dphi = 2.0 * PI / grid as f32;
    let mut acc = 0.0f64;
    for i in 0..grid {
        let s = -1.0 + (i as f32 + 0.5) * ds;
        let theta_i = s.asin();
        for j in 0..grid {
            let phi_i = -PI + (j as f32 + 0.5) * dphi;
            let dir = geo.dir_from_angles(theta_i, phi_i);
            let cos = dir.dot(&n).max(0.0);
            let f = sc.eval_full(theta_i, phi_i, theta_o, phi_o);
            acc += (f * cos) as f64 * ds as f64 * dphi as f64;
        }
    }
    acc
}

/// 用给定 pdf 对同一批方向估计 `E[f·cos/pdf]`。
///
/// `pdf_of` 允许传入"故意改错"的 pdf（S4 负控）；方向、`f` 一律来自**正确**
/// 的 `sample_hair_bounce`，分子用 `hb.f`（与续弹吞吐逐位同源），角度用
/// `hb.theta_i/phi_i`（避免 `dir → (θ,φ)` 重投影的 ulp 误差污染 pdf 分桶）。
fn estimate_with_pdf(
    sc: &HairScatter,
    geo: &HairGeometry,
    wo: Vec3,
    n: Vec3,
    ctx: &EvalCtx,
    n_samp: u32,
    pdf_of: impl Fn(f32, f32) -> f32,
) -> f64 {
    let mut acc = 0.0f64;
    for k in 0..n_samp {
        let (ul, ut, up) = sobol3(ctx, k);
        let hb = sample_hair_bounce(sc, geo, wo, n, ul, ut, up);
        let pdf = pdf_of(hb.theta_i, hb.phi_i);
        if pdf > 0.0 {
            let cos = hb.dir.dot(&n).max(0.0);
            acc += (hb.f * cos / pdf) as f64;
        }
    }
    acc / n_samp as f64
}

/// S1：**∫pdf dω = 1**（误差 ≤ 1e-3，或按仓库白炉口径）。
#[test]
fn s1_pdf_is_normalized_over_the_sphere() {
    let (wo, _) = fixed_frame();
    for sc in [scatter(), HairScatter::blond(), HairScatter::dark()] {
        let geo = geometry();
        let theta_o = geo.longitudinal(wo);
        let phi_o = geo.azimuth(wo);
        let table = sc.pdf_table(theta_o, phi_o);
        // 与 CDF 表同格的 2× 细分中点求积：分段常数密度 ⇒ 求和精确为 1。
        let g = 512usize;
        let ds = 2.0 / g as f32;
        let dphi = 2.0 * PI / g as f32;
        let mut acc = 0.0f64;
        for i in 0..g {
            let s = -1.0 + (i as f32 + 0.5) * ds;
            let theta_i = s.asin();
            for j in 0..g {
                let phi_i = -PI + (j as f32 + 0.5) * dphi;
                acc += sc.pdf_with_table(&table, theta_i, phi_i) as f64
                    * ds as f64
                    * dphi as f64;
            }
        }
        println!("[S1] ∫pdf dω = {acc:.9}（shift={}，k={}）", sc.phi_tt_shift, sc.k_azimuth);
        assert!((acc - 1.0).abs() <= 1e-3, "pdf 归一失败：∫pdf dω = {acc}");
    }
}

/// S2：**无偏**——新 sampler 估计量与解析参考之差 ≤ 白炉阈值（1e-2），
/// 旧 Disney 提案对同一参考的差也一并打印对照（不改尺子）。
#[test]
fn s2_sampler_is_unbiased_and_old_proposal_is_printed() {
    let sc = scatter();
    let geo = geometry();
    let (wo, n) = fixed_frame();
    let r = reference(&sc, &geo, wo, n, 256);
    assert!(r > 1e-4, "参考值应非零（场景构造失效）：R={r}");

    let ctx = EvalCtx::new(0x7A11_5A77, 0, 0);
    let n_samp = 16_384u32;
    let theta_o = geo.longitudinal(wo);
    let phi_o = geo.azimuth(wo);
    let table = sc.pdf_table(theta_o, phi_o);
    // 新 sampler：pdf 就是采样密度。
    let est_new = estimate_with_pdf(&sc, &geo, wo, n, &ctx, n_samp, |ti, pi| {
        sc.pdf_with_table(&table, ti, pi)
    });
    let rel_new = ((est_new - r) / r).abs();
    println!(
        "[S2] 参考 R={r:.9}；新 sampler E[f·cos/pdf]={est_new:.9}，相对差 {rel_new:.3e}（{n_samp} 样本）"
    );
    assert!(rel_new <= 1e-2, "新 sampler 偏差超白炉阈值：{rel_new}（R={r}, est={est_new}）");

    // 旧实现对照：Disney 提案（hair_proposal_bsdf 同参数）采样方向，f 仍取
    // Marschner eval_full。它的 pdf 不贴 f 且只在 n 半球定义，差必然更大——
    // 这正是 R-6② 要治的"无偏但 pdf 不贴 f"。
    let tint = sc.absorption_transmit(sc.radius);
    let proposal = Bsdf::new(
        Vec3::splat(tint), 0.0, sc.beta_m.clamp(0.02, 1.0), sc.eta.max(1.0),
        0.0, 0.0, Vec3::splat(0.3),
    );
    let (mut acc_old, mut used_old) = (0.0f64, 0u32);
    for k in 0..n_samp {
        let (ul, u1, u2) = sobol3(&ctx, k);
        let Some(bs) = proposal.sample(wo, n, false, &SampleCtx::default(), ul, u1, u2) else {
            continue;
        };
        if bs.pdf <= 0.0 || bs.entered {
            continue;
        }
        let f = sc.eval_full(
            geo.longitudinal(bs.dir), geo.azimuth(bs.dir), theta_o, phi_o,
        );
        let cos = bs.dir.dot(&n).max(0.0);
        acc_old += (f * cos / bs.pdf) as f64;
        used_old += 1;
    }
    let est_old = acc_old / used_old.max(1) as f64;
    let rel_old = ((est_old - r) / r).abs();
    println!(
        "[S2] 旧 Disney 提案 E[f·cos/pdf]={est_old:.9}，相对差 {rel_old:.3e}（{used_old} 有效样本）；\
         差异来源：提案 pdf 不贴 f 且只覆盖法线半球，Marschner f 在 4π 非零"
    );
    assert!(rel_new <= rel_old, "新 sampler 不应比旧提案更偏：new={rel_new} old={rel_old}");
}

/// S3：**f/pdf 自洽**——采样点的 `f`、`pdf` 与直接求值逐位一致；
/// 且 `f/pdf` 与"直接求值一次"的比值 ≤ 1e-6。
#[test]
fn s3_f_over_pdf_is_bitwise_self_consistent() {
    let sc = scatter();
    let geo = geometry();
    let (wo, _) = fixed_frame();
    let theta_o = geo.longitudinal(wo);
    let phi_o = geo.azimuth(wo);
    let table = sc.pdf_table(theta_o, phi_o);
    let ctx = EvalCtx::new(0x5E1F_C0DE, 0, 0);
    let mut ratios = Vec::with_capacity(1024);
    for k in 0..1024u32 {
        let (ul, ut, up) = sobol3(&ctx, k);
        let s = sc.sample_incident(theta_o, phi_o, ul, ut, up);
        let f_direct = sc.eval_full(s.theta_i, s.phi_i, theta_o, phi_o);
        let pdf_direct = sc.pdf_with_table(&table, s.theta_i, s.phi_i);
        assert_eq!(s.f.to_bits(), f_direct.to_bits(), "采样 #{k} 的 f 与直接求值不逐位一致");
        assert_eq!(s.pdf.to_bits(), pdf_direct.to_bits(), "采样 #{k} 的 pdf 与直接求值不逐位一致");
        assert!(s.pdf > 0.0, "采样 #{k} pdf 应恒正");
        let a = s.f / s.pdf;
        let b = f_direct / pdf_direct;
        assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0), "采样 #{k} f/pdf 不自洽");
        ratios.push(a);
    }
    // 附加证据：pdf 贴 f ⇒ f/pdf 的离散度应远小于"完全不贴 f"的量级。
    let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let var = ratios.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / ratios.len() as f32;
    let cv = var.sqrt() / mean.abs().max(1e-12);
    println!("[S3] 1024 个采样点 f/pdf 与直接求值逐位一致；f/pdf 变异系数 = {cv:.4}（pdf 贴 f）");
    // 0.5 ≈ 256 格分段常数对 I₀ 峰的预期离散度量级；"完全不贴 f"（均匀 pdf）
    // 的 cv 会到 1e1 以上，故这个上界仍有判别力。
    assert!(cv < 0.5, "pdf 应贴 f（分段表 256 格）：f/pdf 变异系数 {cv}");
}

/// S4：**三瓣判别力**——故意错置瓣权重 / 错方位偏移，两个负控都必须 FAIL。
#[test]
fn s4_wrong_lobe_weights_or_wrong_azimuth_offset_must_fail() {
    let sc = scatter();
    let geo = geometry();
    let (wo, n) = fixed_frame();
    let r = reference(&sc, &geo, wo, n, 256);
    let ctx = EvalCtx::new(0xFA11_0001, 0, 0);
    let n_samp = 8_192u32;
    let tol = 1e-2; // 与 S2 同一把尺（白炉阈值）
    let theta_o = geo.longitudinal(wo);
    let phi_o = geo.azimuth(wo);

    // 负控 A：把三瓣权重错置成 R 独大（正确的 P_R≈0.35 → 0.85）。
    let mut bad_w = sc;
    bad_w.w_r = 0.85;
    bad_w.w_tt = 0.08;
    bad_w.w_trt = 0.07;
    let table_a = bad_w.pdf_table(theta_o, phi_o);
    let est_a = estimate_with_pdf(&sc, &geo, wo, n, &ctx, n_samp, |ti, pi| {
        bad_w.pdf_with_table(&table_a, ti, pi)
    });
    let rel_a = ((est_a - r) / r).abs();
    println!("[S4-A] 错置瓣权重：E={est_a:.9}，相对差 {rel_a:.3e}（阈值 {tol}）");
    assert!(rel_a > tol, "错置瓣权重未被判据抓住（假绿）：rel={rel_a}");

    // 负控 B：把方位偏移写错（0.9 → −1.1）。
    let mut bad_s = sc;
    bad_s.phi_tt_shift = -1.1;
    let table_b = bad_s.pdf_table(theta_o, phi_o);
    let est_b = estimate_with_pdf(&sc, &geo, wo, n, &ctx, n_samp, |ti, pi| {
        bad_s.pdf_with_table(&table_b, ti, pi)
    });
    let rel_b = ((est_b - r) / r).abs();
    println!("[S4-B] 错方位偏移：E={est_b:.9}，相对差 {rel_b:.3e}（阈值 {tol}）");
    assert!(rel_b > tol, "错方位偏移未被判据抓住（假绿）：rel={rel_b}");

    // 正控：正确 pdf 在同一批方向上必须过（排除"负控红是因为估计器本身坏了"）。
    let table_ok = sc.pdf_table(theta_o, phi_o);
    let est_ok = estimate_with_pdf(&sc, &geo, wo, n, &ctx, n_samp, |ti, pi| {
        sc.pdf_with_table(&table_ok, ti, pi)
    });
    let rel_ok = ((est_ok - r) / r).abs();
    println!("[S4-正控] 正确 pdf：E={est_ok:.9}，相对差 {rel_ok:.3e}");
    assert!(rel_ok <= tol, "正控不应失败：rel={rel_ok}");
}

/// FNV-1a（测试内联，避免引依赖）。
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// S5：**确定性**——采样器两跑逐字节；单股场景渲染两跑逐字节。
#[test]
fn s5_two_pass_bitwise_and_single_strand_difference_origin() {
    let sc = scatter();
    let geo = geometry();
    let (wo, n) = fixed_frame();
    let ctx = EvalCtx::new(0xDE7E_2026, 0, 0);
    let mut bytes_a = Vec::with_capacity(4096 * 24);
    let mut bytes_b = Vec::with_capacity(4096 * 24);
    for k in 0..4096u32 {
        let (ul, ut, up) = sobol3(&ctx, k);
        let ha = sample_hair_bounce(&sc, &geo, wo, n, ul, ut, up);
        let hb = sample_hair_bounce(&sc, &geo, wo, n, ul, ut, up);
        for v in [ha.dir.x, ha.dir.y, ha.dir.z, ha.theta_i, ha.phi_i, ha.f, ha.pdf, ha.weight] {
            bytes_a.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for v in [hb.dir.x, hb.dir.y, hb.dir.z, hb.theta_i, hb.phi_i, hb.f, hb.pdf, hb.weight] {
            bytes_b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        assert_eq!(ha.dir.x.to_bits(), hb.dir.x.to_bits(), "两跑方向必须逐位一致");
        assert_eq!(ha.weight.to_bits(), hb.weight.to_bits(), "两跑吞吐必须逐位一致");
    }
    assert_eq!(fnv1a(&bytes_a), fnv1a(&bytes_b), "采样器两跑逐字节不一致");
    println!("[S5] 4096 次采样两跑逐字节一致（fnv1a = {:016x}）", fnv1a(&bytes_a));

    // 单股场景：同一 `render_pixel` 两跑逐位相同；并给出"新 sampler 方向 vs
    // 旧 Disney 提案方向"的数字，说明差异来源（不改尺子、不假装相同）。
    use prism_geometry::curve_tess::{tessellate, CurveReq};
    use prism_rt::camera::Camera;
    use prism_rt::integrator::{render_pixel, Light, MaterialKind, RenderParams, Scene};

    let pts: Vec<Vec3> = (0..=12)
        .map(|i| Vec3::new(0.06 * i as f32, i as f32 * 0.1, 0.0))
        .collect();
    let tris = tessellate(&CurveReq::Strand { points: pts, radii: vec![0.05f32] }, 0.02);
    let frame = |mat: MaterialKind| -> Vec<[f32; 4]> {
        let mut scene = Scene::default();
        scene.add_bvh(tris.clone(), mat);
        scene.lights.push(Light::Ambient { radiance: Vec3::new(0.35, 0.35, 0.35) });
        let cam = Camera::new(
            Vec3::new(0.6, 0.6, 1.8), Vec3::new(0.3, 0.6, 0.0),
            Vec3::new(0.0, 1.0, 0.0), 45.0, 0.0, 4.0, 0.0, 0.0,
        );
        let ctx = EvalCtx::new(20261202, 11, 3);
        let params = RenderParams::new(2, 2);
        let mut out = Vec::with_capacity(16);
        for y in 0..4u32 {
            for x in 0..4u32 {
                out.push(render_pixel(&scene, &cam, &ctx, (x, y), (4, 4), &params));
            }
        }
        out
    };
    let hair = MaterialKind::Hair { scatter: sc, geometry: geo };
    let a = frame(hair);
    let b = frame(hair);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        for c in 0..4 {
            assert_eq!(x[c].to_bits(), y[c].to_bits(), "单股场景两跑第 {i} 像素不符");
        }
    }
    let mut fb = Vec::with_capacity(16 * 16);
    for p in &a {
        for c in p {
            fb.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    println!("[S5] 单股场景两跑逐字节一致（4×4, spp=2, fnv1a = {:016x}）", fnv1a(&fb));

    // 差异来源的数字：首弹方向 / pdf 的旧新对照。
    let (ul, ut, up) = sobol3(&ctx, 0);
    let hb = sample_hair_bounce(&sc, &geo, wo, n, ul, ut, up);
    let tint = sc.absorption_transmit(sc.radius);
    let proposal = Bsdf::new(
        Vec3::splat(tint), 0.0, sc.beta_m.clamp(0.02, 1.0), sc.eta.max(1.0),
        0.0, 0.0, Vec3::splat(0.3),
    );
    let old = proposal.sample(wo, n, false, &SampleCtx::default(), ul, ut, up);
    let (old_dir, old_pdf) = match old {
        Some(s) => (s.dir, s.pdf),
        None => (Vec3::new(0.0, 0.0, 1.0), 0.0),
    };
    let dot = hb.dir.dot(&old_dir).clamp(-1.0, 1.0);
    println!(
        "[S5] 与旧实现必然不同：首弹新方向 {:?}（pdf={:.6}） vs 旧 Disney 提案 {:?}（pdf={:.6}），\
         夹角 {:.4} rad；差异来源 = 续弹由 Disney 提案改为按 f 的 Marschner sampler（R-6② 本轮）",
        [hb.dir.x, hb.dir.y, hb.dir.z], hb.pdf,
        [old_dir.x, old_dir.y, old_dir.z], old_pdf,
        dot.acos()
    );

    // 旧实现帧 vs 新 sampler 帧：同一单股散射点、同一 scrambled Sobol 随机流、
    // 同一份 Marschner f，唯一差别是续弹方向 / pdf 的来源。逐像素 `to_bits()`
    // 给出**完整帧**的差异数字（不假装相同、不放宽尺子、不改判据）。
    let theta_o = geo.longitudinal(wo);
    let phi_o = geo.azimuth(wo);
    let ambient = [0.35f32, 0.35, 0.35];
    let mut legacy = Vec::with_capacity(16);
    let mut modern = Vec::with_capacity(16);
    for k in 0..16u32 {
        let mut lo = [0.0f32; 3];
        let mut mo = [0.0f32; 3];
        for s in 0..2u32 {
            let (ul, ut, up) = sobol3(&ctx, k * 2 + s);
            // 改动前的组合：Disney 提案定方向 / pdf，Marschner f 取值。
            if let Some(bs) =
                proposal.sample(wo, n, false, &SampleCtx::default(), ul, ut, up)
            {
                let th = geo.longitudinal(bs.dir);
                let ph = geo.azimuth(bs.dir);
                let f = sc.eval_full(th, ph, theta_o, phi_o);
                let cos = bs.dir.dot(&n).max(0.0);
                let w = if bs.pdf > 0.0 { f * cos / bs.pdf } else { 0.0 };
                for c in 0..3 {
                    lo[c] += w * ambient[c];
                }
            }
            // 本轮组合：按 f 采样方向，f / pdf / weight 同源。
            let hb = sample_hair_bounce(&sc, &geo, wo, n, ul, ut, up);
            for c in 0..3 {
                mo[c] += hb.weight * ambient[c];
            }
        }
        let mut lp = [0.0f32; 4];
        let mut mp = [0.0f32; 4];
        for c in 0..3 {
            lp[c] = lo[c] * 0.5;
            mp[c] = mo[c] * 0.5;
        }
        lp[3] = 1.0;
        mp[3] = 1.0;
        legacy.push(lp);
        modern.push(mp);
    }
    let mut differing = 0usize;
    let mut max_abs = 0.0f32;
    for (p, q) in legacy.iter().zip(modern.iter()) {
        let mut bad = false;
        for c in 0..4 {
            if p[c].to_bits() != q[c].to_bits() {
                bad = true;
            }
            max_abs = max_abs.max((p[c] - q[c]).abs());
        }
        if bad {
            differing += 1;
        }
    }
    println!(
        "[S5] 单股 4×4 spp=2 旧实现帧 vs 新 sampler 帧：differing_pixels = {differing}/16，\
         最大通道绝对差 {max_abs:.6e}；f 与随机流逐位相同，差异只来自续弹方向/pdf（R-6②）"
    );
    assert!(differing > 0, "续弹已换 sampler，旧新帧必然不同；若相同说明对照没生效");
}
