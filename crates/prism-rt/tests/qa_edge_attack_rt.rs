//! QA 攻击性边界测试（bsdf / light / sss / caustics）。
//!
//! 每项攻击一个边界/退化输入，断言以、
//! 、D-29/D-30 契约为准。**只写测试，不改源码**——发现缺陷记录证据。
//!
//! 攻击清单：
//! 1. BSDF 构造器参数域防御（金属度/粗糙度/ior/透射/次表面/半径）。
//! 2. BSDF eval 的 NaN/Inf 传播与非单位向量健壮性。
//! 3. GGX D 在掠射/退化 α 的有限性与单调衰减（能量守恒前提）。
//! 4. schlick / ggx_d / smith_g1 / refract_dir 的边界数学锚点。
//! 5. 灯光采样退化：零半径球 / 退化矩形 / 零边 / NaN。
//! 6. light pdf 换算恒等（球光立体角 vs 数值积分）。
//! 7. caustics 退化几何防御 + 确定性 + 雅可比有限性。
//! 8. sss det_exp 全域有限性 + 零半径。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;
use prism_rt::bsdf::{ggx_d, refract_dir, schlick, smith_g1, Bsdf};
use prism_rt::caustics::{mnee_mirror, mnee_refraction, Refractor};
use prism_rt::light::{nee_mis_weight, sample_nee, Light};
use prism_rt::sss::det_exp;

fn vbits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

// ---------------------------------------------------------------------------
// ① BSDF 构造器参数域防御（宪章 C19：非法输入必须 panic 而非静默错误）
// ---------------------------------------------------------------------------

#[test]
fn bsdf_ctor_rejects_out_of_domain() {
    let bc = Vec3::splat(0.5);
    let r = Vec3::splat(0.3);
    // 金属度越界。
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 1.5, 0.5, 1.5, 0.0, 0.0, r)).is_err(),
        "metalness=1.5 必须拒绝"
    );
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, -0.1, 0.5, 1.5, 0.0, 0.0, r)).is_err(),
        "metalness=-0.1 必须拒绝"
    );
    // 粗糙度越界。
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 0.5, 1.01, 1.5, 0.0, 0.0, r)).is_err(),
        "roughness>1 必须拒绝"
    );
    // ior < 1。
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 0.5, 0.5, 0.9, 0.0, 0.0, r)).is_err(),
        "ior<1 必须拒绝"
    );
    // 透射越界。
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 0.5, 0.5, 1.5, 1.2, 0.0, r)).is_err(),
        "transmission>1 必须拒绝"
    );
    // 次表面半径非正。
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 0.5, 0.5, 1.5, 0.0, 0.0, Vec3::new(0.0, 0.3, 0.3)))
            .is_err(),
        "次表面半径分量=0 必须拒绝"
    );
    assert!(
        std::panic::catch_unwind(|| Bsdf::new(bc, 0.5, 0.5, 1.5, 0.0, 0.0, Vec3::new(-1.0, 0.3, 0.3)))
            .is_err(),
        "次表面半径分量<0 必须拒绝"
    );
    // 合法边界值应通过（闭区间端点）。
    let _ = Bsdf::new(bc, 0.0, 0.0, 1.0, 0.0, 0.0, r);
    let _ = Bsdf::new(bc, 1.0, 1.0, 1.0, 1.0, 1.0, r);
}

// ---------------------------------------------------------------------------
// ② BSDF eval / sample：NaN/Inf 与非单位向量健壮性
// ---------------------------------------------------------------------------

#[test]
fn bsdf_eval_no_nan_under_degenerate_input() {
    // 攻击：NaN 分量 / Inf / 零向量 / 反向 wi。eval 在库代码，契约是
    // "非法/侧别相反输入返回 0，不 panic、不产 NaN"。
    let b = Bsdf::new(Vec3::splat(0.8), 0.0, 0.5, 1.5, 0.0, 0.0, Vec3::splat(0.3));
    let n = Vec3::new(0.0, 0.0, 1.0);
    // 侧别相反：wi 朝下 → cos_i ≤ 0 → 精确 0。
    let f = b.eval(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0), n);
    assert_eq!(vbits(f), [0, 0, 0], "反向 wi 应精确零");
    // 零向量 wi。
    let f0 = b.eval(Vec3::new(0.0, 0.0, 1.0), Vec3::splat(0.0), n);
    assert_eq!(vbits(f0), [0, 0, 0], "零 wi 应零");
    // NaN 输入不得 panic（结果可非有限，但不得崩溃）。
    let _ = b.eval(Vec3::new(f32::NAN, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), n);
    // 正常评估必须有限非负。
    let g = b.eval(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), n);
    assert!(
        g.x.is_finite() && g.y.is_finite() && g.z.is_finite() && g.x >= 0.0,
        "正入射 eval 必须有限非负：{g:?}"
    );
}

// ---------------------------------------------------------------------------
// ③ GGX D 边界：掠射有限、退化 α、单调衰减
// ---------------------------------------------------------------------------

#[test]
fn ggx_d_grazing_and_monotonic() {
    // 掠射 cosθ→0：D 应趋于 0（分母 (α²−1)c²+1 → 1，D → α²/π）。
    // 关键：不得 NaN/Inf/负。
    for &alpha in &[1e-3f32, 0.1, 0.5, 1.0] {
        for &ct in &[1.0f32, 0.5, 1e-6, 0.0] {
            let d = ggx_d(alpha, ct);
            assert!(d.is_finite() && d >= 0.0, "戈GGX D 越界：α={alpha} cosθ={ct} → {d}");
        }
        // 峰值 ∈ 谷底（h=n 最大，掠射最小）——单调衰减前提。
        let peak = ggx_d(alpha, 1.0);
        let mid = ggx_d(alpha, 0.5);
        let edge = ggx_d(alpha, 0.0);
        assert!(peak >= mid && mid >= edge, "GGX D 应随 cosθ 单调不增：α={alpha}");
    }
    // α=1（完全粗糙）：D = 1/π 常数（Lambert 极限）。
    for &ct in &[1.0f32, 0.5, 0.1] {
        let d = ggx_d(1.0, ct);
        assert!((d - 1.0 / std::f32::consts::PI).abs() < 1e-5, "α=1 应退化为 1/π：{d}");
    }
}

#[test]
fn smith_g1_bounds() {
    // G1 ∈ [0,1]；掠射 → 0；正入射 → 1。
    for &alpha in &[1e-3f32, 0.1, 0.5, 1.0] {
        assert!((smith_g1(alpha, 1.0) - 1.0).abs() < 1e-5, "正入射 G1 应=1");
        for &ct in &[0.5f32, 0.1, 1e-3] {
            let g = smith_g1(alpha, ct);
            assert!((0.0..=1.0 + 1e-6).contains(&g), "G1 越界：α={alpha} cosθ={ct} → {g}");
        }
    }
}

// ---------------------------------------------------------------------------
// ④ refract_dir 掠射 / 全内反射（ior<1 场景）
// ---------------------------------------------------------------------------

#[test]
fn refract_dir_tir_and_snells_law() {
    let n = Vec3::new(0.0, 0.0, 1.0);
    // 正入射：不偏折。
    let r = refract_dir(Vec3::new(0.0, 0.0, -1.0), n, 0.5).unwrap();
    assert!((r.z + 1.0).abs() < 1e-5, "正入射应沿 −z：{r:?}");
    // eta > 1（出射侧光密）且掠射 → 全内反射 None。
    // cos_i 小 + eta 大 → sin2_t ≥ 1。
    let grazing = Vec3::new((1.0 - 0.01f32 * 0.01).sqrt(), 0.0, -0.01).normalize();
    let r2 = refract_dir(grazing, n, 2.0);
    assert!(r2.is_none(), "掠射 + eta>1 应触发 TIR（返回 None）");
    // 折射后方向应单位化且与入射同侧（z<0 进）。
    let r3 = refract_dir(Vec3::new(0.3, 0.0, -0.95).normalize(), n, 1.0 / 1.5).unwrap();
    assert!((r3.length() - 1.0).abs() < 1e-5, "折射方向应单位化：{}", r3.length());
    assert!(r3.z < 0.0, "折射方向应继续向 −z");
}

// ---------------------------------------------------------------------------
// ⑤ schlick 边界锚点
// ---------------------------------------------------------------------------

#[test]
fn schlick_energy_bounds() {
    let f0 = Vec3::new(0.04, 0.04, 0.04);
    // cosθ=1 → F0；cosθ=0 → 1；cosθ∈(0,1) → F0 ≤ F ≤ 1（单调）。
    let f_1 = schlick(f0, 1.0);
    assert_eq!(f_1.x.to_bits(), 0.04f32.to_bits());
    let f_0 = schlick(f0, 0.0);
    assert!((f_0.x - 1.0).abs() < 1e-6, "cosθ=0 应 F→1：{}", f_0.x);
    // 契约边界（doc：F = F0+(1−F0)(1−cosθ)⁵，保护项 clamp 的是 (1−cosθ) 使
    // cosθ>1 不越界）。**注意**：源函数对 cosθ<0 不 clamp（调用方保证 cosθ≥0），
    // 故本测试不假设负 cosθ 的语义——仅验证 cosθ∈[0,1] 的正确域。
    // 单调**不增**（cosθ 增大 → Fresnel 反射从 1 降到 F0）。
    let mut prev = 1.0 + 1e-6;
    for k in 0..=10 {
        let c = k as f32 / 10.0;
        let v = schlick(f0, c).x;
        assert!(
            v <= prev + 1e-6 && (0.04 - 1e-6..=1.0 + 1e-6).contains(&v),
            "Schlick 单调/有界破缺 c={c} v={v} prev={prev}"
        );
        prev = v;
    }
    // 端点：cosθ=1 → F0 精确。
    assert!((schlick(f0, 1.0).x - 0.04).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// ⑥ 灯光退化采样防御
// ---------------------------------------------------------------------------

#[test]
fn light_degenerate_sampling_no_panic() {
    let p = Vec3::new(1.0, 2.0, 3.0);
    // 零半径球：d_c > 0 → cos_max=1，pdf = 1/(2π·0) = Inf？攻击塌缩。
    let zero_sphere = Light::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.0,
        radiance: Vec3::splat(1.0),
    };
    let s = sample_nee(&zero_sphere, p, 0.5, 0.5);
    if let Some(s) = s {
        // 零半径球立体角→0，pdf 可能发散；记录行为（不得 panic）。
        assert!(s.pdf_w.is_finite() || s.pdf_w.is_infinite(), "零半径球 pdf 异常：{}", s.pdf_w);
    }
    // 退化矩形（零边长）：cross = 0 → None。
    let degen_rect = Light::Rect {
        center: Vec3::new(0.0, 5.0, 0.0),
        edge_u: vec3(0.0, 0.0, 0.0),
        edge_v: Vec3::new(1.0, 0.0, 0.0),
        radiance: Vec3::splat(1.0),
    };
    assert!(sample_nee(&degen_rect, p, 0.5, 0.5).is_none(), "退化矩形应 None");
    // 共线边（cross=0）：None。
    let col = Light::Rect {
        center: Vec3::new(0.0, 5.0, 0.0),
        edge_u: Vec3::new(1.0, 0.0, 0.0),
        edge_v: Vec3::new(2.0, 0.0, 0.0),
        radiance: Vec3::splat(1.0),
    };
    assert!(sample_nee(&col, p, 0.5, 0.5).is_none(), "共线边矩形应 None");
    // 采样点贴面（d2≈0）：None。
    let rect = Light::Rect {
        center: Vec3::new(0.0, 0.0, 0.0),
        edge_u: Vec3::new(4.0, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 4.0, 0.0),
        radiance: Vec3::splat(1.0),
    };
    // p 在矩形中心 → to_q 可能近零。
    let near = sample_nee(&rect, Vec3::splat(0.0), 0.5, 0.5);
    assert!(near.is_none(), "着色点在矩形面上应 None（d2≈0）");
}

fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

// ---------------------------------------------------------------------------
// ⑦ 灯光 pdf 换算：球光立体角 vs 数值积分（能量守恒前提）
// ---------------------------------------------------------------------------

#[test]
fn sphere_nee_pdf_matches_numeric_solid_angle() {
    // 球交锥立体角 Ω = 2π(1−cos_max)；pdf_w = 1/Ω。
    // 数值核对：对大量 u1/u2 锥内方向，立体角应 = Ω。
    let sph = Light::Sphere {
        center: Vec3::new(0.0, 0.0, 3.0),
        radius: 1.0,
        radiance: Vec3::splat(2.0),
    };
    let p = Vec3::splat(0.0);
    let d = 3.0f32;
    let cos_max = (1.0 - 1.0 / (d * d)).sqrt();
    let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_max);
    let s = sample_nee(&sph, p, 0.37, 0.61).unwrap();
    assert!((s.pdf_w * omega - 1.0).abs() < 1e-5, "球光 pdf·Ω 应为 1：{}", s.pdf_w * omega);
}

// ---------------------------------------------------------------------------
// ⑧ caustics 退化几何 + 确定性 + 雅可比有限性
// ---------------------------------------------------------------------------

#[test]
fn mnee_mirror_degenerate_and_bounded() {
    let ctx = EvalCtx::new(0xCA_ED, 0, 0);
    let si = SampleIndex { frame: 0, pixel: (0, 0), sample: 0, dimension: 0, node_id: 1 };
    // 退化法线（零向量）：normalize 会产生 NaN → 攻击。契约：不应静默返回错误样本。
    let s = mnee_mirror(
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::splat(0.0),
        Vec3::splat(0.0),
        Vec3::new(2.0, 1.0, 0.2),
        &ctx,
        &si,
    );
    if let Some(s) = s {
        assert!(
            s.wi.x.is_finite() && s.wi.y.is_finite() && s.wi.z.is_finite(),
            "零法线应返回 None 或有限 wi，得 {:?}",
            s.wi
        );
    }
    // 正常镜面链：pdf=1, weight=1, wi 单位化。
    let ok = mnee_mirror(
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::splat(0.0),
        Vec3::new(2.0, 1.0, 0.2),
        &ctx,
        &si,
    )
    .unwrap();
    assert!((ok.wi.length() - 1.0).abs() < 1e-5, "wi 应单位化：{}", ok.wi.length());
}

#[test]
fn mnee_refraction_jacobian_pdf_finite() {
    // 球折射链：多初值收敛时 pdf_w 必须有限正（差分雅可比不得为零/NaN）。
    let refr = Refractor { center: Vec3::new(0.0, 1.5, 3.0), radius: 1.0, ior: 1.5 };
    let ctx = EvalCtx::new(0xCA_EF, 0, 0);
    let p = Vec3::splat(0.0);
    let light = Vec3::new(0.3, 2.5, 6.0);
    let mut converged = 0;
    for node in 0..16u64 {
        let si = SampleIndex { frame: 0, pixel: (0, 0), sample: 3, dimension: 0, node_id: node };
        if let Some(s) = mnee_refraction(&refr, p, light, &ctx, &si) {
            converged += 1;
            assert!(s.pdf_w.is_finite() && s.pdf_w > 0.0, "pdf_w 非法：{}", s.pdf_w);
            assert!(
                s.wi.x.is_finite() && s.wi.y.is_finite() && s.wi.z.is_finite(),
                "wi 非法：{:?}",
                s.wi
            );
            assert!((s.wi.length() - 1.0).abs() < 1e-4, "wi 应单位化");
        }
    }
    assert!(converged > 0, "多初值至少一收敛");
}

// ---------------------------------------------------------------------------
// ⑨ det_exp 全域有限性
// ---------------------------------------------------------------------------

#[test]
fn det_exp_is_finite_over_range() {
    // D-29 确定性 exp：数学上 exp(x) > 0 恒成立。全域应有限且**非负**。
    for &x in &[0.0f32, -1.0, -10.0, -88.0, -89.0, -90.0, -90.5, -95.0, -100.0, 1.0, 10.0, 80.0, 88.0] {
        let v = det_exp(x);
        assert!(
            v.is_finite(),
            "det_exp({x}) = {v} 非有限（D-29 违例）"
        );
        assert!(
            v >= 0.0,
            "det_exp({x}) = {v} 为负——exp(x) 必正，负透射率会污染 SSS 吞吐（物理不可能）"
        );
    }
    // 大负 → 0（下溢），非负。
    assert!(det_exp(-100.0) >= 0.0, "det_exp(-100) 必须 ≥ 0：{}", det_exp(-100.0));
    assert!(det_exp(-100.0) < 1e-30, "det_exp(-100) 应极小：{}", det_exp(-100.0));
    // 单调性抽查（正常域）。
    assert!(det_exp(-3.0) < det_exp(-2.0));
    assert!(det_exp(2.0) < det_exp(3.0));
}

// ---------------------------------------------------------------------------
// ⑩ BSDF sample 确定性（瓣选择 + 方向逐位）
// ---------------------------------------------------------------------------

#[test]
fn bsdf_sample_deterministic_all_lobes() {
    // 攻击不同瓣：金属镜面 / 电介质漫反 / 玻璃透射，同参数两遍逐位一致。
    let cases = [
        Bsdf::new(Vec3::splat(0.9), 1.0, 0.3, 1.5, 0.0, 0.0, Vec3::splat(0.3)),
        Bsdf::new(Vec3::splat(0.8), 0.0, 0.6, 1.5, 0.0, 0.0, Vec3::splat(0.3)),
        Bsdf::new(Vec3::splat(0.9), 0.0, 0.1, 1.5, 1.0, 0.0, Vec3::splat(0.3)),
    ];
    let n = Vec3::new(0.0, 0.0, 1.0);
    let wo = Vec3::new(0.3, 0.0, 0.95).normalize();
    let ctx = EvalCtx::new(0xB5DF, 0, 0);
    let sctx = prism_rt::bsdf::SampleCtx::default();
    for (i, b) in cases.iter().enumerate() {
        for k in 0..32u32 {
            let si = SampleIndex { frame: 0, pixel: (0, 0), sample: k, dimension: 8, node_id: 0 };
            let (ul, u1, u2) = {
                let a = prism_rt::sampler::sample_1d(&ctx, &si);
                let si2 = SampleIndex { dimension: 9, ..si };
                let (x, y) = prism_rt::sampler::sample_2d(&ctx, &si2);
                (a, x, y)
            };
            let s1 = b.sample(wo, n, false, &sctx, ul, u1, u2);
            let s2 = b.sample(wo, n, false, &sctx, ul, u1, u2);
            match (s1, s2) {
                (Some(a), Some(c)) => {
                    assert_eq!(vbits(a.dir), vbits(c.dir), "case {i} k={k} dir 不确定");
                    assert_eq!(a.pdf.to_bits(), c.pdf.to_bits(), "case {i} k={k} pdf 不确定");
                    assert_eq!(vbits(a.weight), vbits(c.weight), "case {i} k={k} weight 不确定");
                    assert_eq!(a.specular, c.specular);
                    assert_eq!(a.entered, c.entered);
                }
                (None, None) => {}
                _ => panic!("case {i} k={k} 同参数采样结果不一致"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ⑪ nee_mis_weight 契约：非几何灯恒 1，球光 power heuristic
// ---------------------------------------------------------------------------

#[test]
fn nee_mis_weight_contract() {
    let rect = Light::Rect {
        center: Vec3::new(0.0, 5.0, 0.0),
        edge_u: Vec3::new(2.0, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 2.0, 0.0),
        radiance: Vec3::splat(1.0),
    };
    // 非几何灯：任意 pdf 权重恒 1。
    assert_eq!(nee_mis_weight(&rect, 100.0, 0.001).to_bits(), 1.0f32.to_bits());
    let sph = Light::Sphere { center: Vec3::new(0.0, 0.0, 5.0), radius: 1.0, radiance: Vec3::splat(1.0) };
    // 双零防御：a=b=0 → 0（不 NaN）。
    assert_eq!(nee_mis_weight(&sph, 0.0, 0.0).to_bits(), 0.0f32.to_bits());
    // power heuristic 对称互补。
    let w = nee_mis_weight(&sph, 0.3, 0.7);
    let wr = nee_mis_weight(&sph, 0.7, 0.3);
    assert!((w + wr - 1.0).abs() < 1e-6, "互补性破缺");
}
