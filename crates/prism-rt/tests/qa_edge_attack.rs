//! QA 边界反例攻击（路径追踪五项）。
//!
//! 每项攻击一个边界，断言以/文档契约为准：
//! 1. Sobol 超 64 维兜底：确定性 + [0,1) + 维度互异（lib.rs "超表兜底"声明）。
//! 2. 数字平移修正版语义：同像素前 2^m 样本保持 (0,2)-net 分层——hash 只作
//!  平移（index = i ⊕ c， ①），不得直接充当 Sobol 索引；c = 0 时
//!    index = i（恒等路径）是该性质的特例。
//! 3. RR 全黑 albedo = 0：能量精确为零（无 NaN、无泄漏， RR 无偏）。
//! 4. 介质 σt 边界：σt=0 构造拒绝；近真空 / σt→∞ 行为合理（Beer 定律极限）。
//! 5. 相机 fov 边界：0/180 拒绝、1°/179° 合法；光圈 = 0 针孔快路径与
//!    focus_distance 无关（DOF 退化恒等）。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;
use prism_rt::camera::{Camera, DIM_PIXEL};
use prism_rt::integrator::{Light, MaterialKind, RenderParams, Scene, Aabb as SceneAabb};
use prism_rt::media::HomogeneousMedium;
use prism_rt::sampler::{sample_1d, sample_2d, SampleStream};

// ---------------------------------------------------------------------------
// ① Sobol 超 64 维兜底
// ---------------------------------------------------------------------------

#[test]
fn sobol_beyond_64_dims_fallback_is_deterministic() {
    let ctx = EvalCtx::new(0xABCD_1234, 7, 3);
    let pixel = (11, 17);

    // 超 64 维路径：dim = 64/65/100 → PCG hash 兜底。
    for &dim in &[64_u16, 65, 100, 255] {
        let si = SampleIndex { frame: 3, pixel, sample: 42, dimension: dim, node_id: 7 };
        let a = sample_1d(&ctx, &si);
        let b = sample_1d(&ctx, &si);
        assert_eq!(a.to_bits(), b.to_bits(), "dim {dim} 兜底必须同参数同输出");
        assert!((0.0..1.0).contains(&a), "dim {dim} 兜底值 {a} 越出 [0,1)");
        assert!(a.is_finite(), "dim {dim} 兜底值必须有限");
    }

    // 不同超表维度互不串流（盐分离）。
    let s64 = sample_1d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 42, dimension: 64, node_id: 7 },
    );
    let s65 = sample_1d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 42, dimension: 65, node_id: 7 },
    );
    assert_ne!(s64.to_bits(), s65.to_bits(), "dim 64 与 65 不得同值");

    // 2D 兜底：两分量都合法且不同。
    let (u, v) = sample_2d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 42, dimension: 64, node_id: 7 },
    );
    assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v), "2D 兜底越界");
    assert_ne!(u.to_bits(), v.to_bits(), "2D 兜底两分量同值可疑");

    // 表内最后一维（63）仍走 Sobol：与超表兜底值不同（不同派生路径）。
    let s63 = sample_1d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 42, dimension: 63, node_id: 7 },
    );
    assert_ne!(s63.to_bits(), s64.to_bits(), "dim 63（Sobol）与 dim 64（hash 兜底）应不同");

    // 超表兜底跨样本仍变化（流不退化）。
    let s0 = sample_1d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 0, dimension: 64, node_id: 7 },
    );
    let s1 = sample_1d(
        &ctx,
        &SampleIndex { frame: 3, pixel, sample: 1, dimension: 64, node_id: 7 },
    );
    assert_ne!(s0.to_bits(), s1.to_bits(), "兜底流跨样本退化");
}

// ---------------------------------------------------------------------------
// ② 数字平移修正版：(0,2)-net 分层保持（含 c = 0 恒等特例）
// ---------------------------------------------------------------------------

#[test]
fn digital_shift_preserves_net_strata() {
    // 256 样本 × 16×16 对齐格：每格恰 1 点（(0,2)-net，m=8, t=0）。
    // 该性质对任意平移常量 c 成立（前 2^m 样本 i ⊕ c 是 [0,2^m) 的置换）——
    // hash 若被错误地直接当索引用，点集不再是 Sobol 前 2^m 点，必有空格/重格。
    let ctx = EvalCtx::new(0xDEAD_BEEF, 1, 0);
    let pixel = (3, 7);
    let n = 256_u32;
    let grid = 16;
    let mut bins = vec![0_u32; grid * grid];
    for i in 0..n {
        let si = SampleIndex { frame: 0, pixel, sample: i, dimension: 0, node_id: 1 };
        let (u, v) = sample_2d(&ctx, &si);
        assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v), "样本 {i} 越界");
        let ju = (u * grid as f32) as usize;
        let jv = (v * grid as f32) as usize;
        bins[ju * grid + jv] += 1;
    }
    for (k, &c) in bins.iter().enumerate() {
        assert_eq!(c, 1, "16×16 格 {k} 含 {c} 点（(0,2)-net 要求恰 1）");
    }

    // 换种子（不同 c 与掩码）性质不变——证明修正版对任意 hash 结果稳健。
    let ctx2 = EvalCtx::new(0x1234_5678_9ABC_DEF0, 2, 5);
    let mut bins2 = vec![0_u32; grid * grid];
    for i in 0..n {
        let si = SampleIndex { frame: 5, pixel, sample: i, dimension: 0, node_id: 2 };
        let (u, v) = sample_2d(&ctx2, &si);
        let ju = (u * grid as f32) as usize;
        let jv = (v * grid as f32) as usize;
        bins2[ju * grid + jv] += 1;
    }
    for (k, &c) in bins2.iter().enumerate() {
        assert_eq!(c, 1, "第二种子下 16×16 格 {k} 含 {c} 点");
    }

    // c = 0 恒等特例的语义锚：index = i ⊕ 0 = i，序列首样本（i=0）必须是
    // Sobol 原点 (0,0) ⊕ 掩码——若实现把 hash 掺进样本序号本身，i=0 的点
    // 不再对应 Sobol(0)=0。这里只断言可从契约推出的弱形式：i=0 与 i=1 的
    // dim-0 坐标不同（Sobol 相邻索引异或非零方向数）。
    let s0 = sample_1d(&ctx, &SampleIndex { frame: 0, pixel, sample: 0, dimension: 0, node_id: 1 });
    let s1 = sample_1d(&ctx, &SampleIndex { frame: 0, pixel, sample: 1, dimension: 0, node_id: 1 });
    assert_ne!(s0.to_bits(), s1.to_bits(), "Sobol 相邻样本 dim0 不得相同");
}

// ---------------------------------------------------------------------------
// ③ RR 全黑 albedo = 0 → 精确零能量
// ---------------------------------------------------------------------------

#[test]
fn rr_black_albedo_exact_zero_energy() {
    // 黑 Lambert 大地板 + 正上方方向光。所有光线命中地板（fov 60 向下看，
    // 最大偏角 30°，全部向下），NEE 贡献含 albedo=0 因子 → 每项精确 0；
    // RR 以 q=0.05 存活继续弹射，但 throughput 恒 0 → 能量恒 0。
    // 无偏性契约：期望值 0（不 panic、无 NaN、无负光）。
    let scene = Scene {
        spheres: vec![],
        boxes: vec![(
            SceneAabb::new(Vec3::new(-10.0, -1.0, -10.0), Vec3::new(10.0, 0.0, 10.0)),
            MaterialKind::Lambert { albedo: Vec3::splat(0.0) },
        )],
        lights: vec![Light::Directional {
            direction: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::splat(10.0),
            cos_theta_max: 1.0,
        }],
        medium: None,
        paraboloids: vec![],
        meshes: Vec::new(),
        planes: vec![],
        ..Default::default()
    };
    let cam = Camera::new(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        60.0,
        0.0,
        5.0,
        0.0,
        0.0,
    );
    let ctx = EvalCtx::new(42, 1, 0);
    let params = RenderParams::new(16, 8);
    let res = (8_u32, 8_u32);
    for py in 0..8_u32 {
        for px in 0..8_u32 {
            let px_rgba = prism_rt::integrator::render_pixel(&scene, &cam, &ctx, (px, py), res, &params);
            assert_eq!(
                px_rgba,
                [0.0, 0.0, 0.0, 1.0],
                "全黑 albedo 像素 ({px},{py}) 必须精确零能量，得 {px_rgba:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ④ 介质 σt 边界
// ---------------------------------------------------------------------------

#[test]
fn medium_sigma_t_zero_is_rejected() {
    let e = std::panic::catch_unwind(|| HomogeneousMedium::new(0.0, 0.5, 0.0));
    assert!(e.is_err(), "σt=0 必须被构造器拒绝（契约：σt 必须为正）");
    let e = std::panic::catch_unwind(|| HomogeneousMedium::new(-1.0, 0.5, 0.0));
    assert!(e.is_err(), "σt<0 必须被拒绝");
    let e = std::panic::catch_unwind(|| HomogeneousMedium::new(1.0, -0.1, 0.0));
    assert!(e.is_err(), "albedo<0 必须被拒绝");
    let e = std::panic::catch_unwind(|| HomogeneousMedium::new(1.0, 0.5, 1.0));
    assert!(e.is_err(), "g=1 必须被拒绝（开区间）");
    // 边界内合法值：albedo 0 与 1（纯吸收/纯散射）。
    let _ = HomogeneousMedium::new(1.0, 0.0, 0.0);
    let _ = HomogeneousMedium::new(1.0, 1.0, -0.999);
}

#[test]
fn medium_sigma_t_extremes_behave_like_beer_law() {
    // σt → ∞：任意 u 的自由程 ≈ 0 → 距离 > 0 的透射率 = 0；dist = 0 → 1。
    let dense = HomogeneousMedium::new(1.0e30, 0.5, 0.0);
    let fp = dense.free_path(0.5);
    assert!(fp.is_finite() && fp >= 0.0, "超大 σt 自由程 {fp} 必须有限非负");
    assert_eq!(dense.shadow_transmittance(0.5, 1.0), 0.0, "σt→∞ 时 dist>0 必须不透明");
    assert_eq!(dense.shadow_transmittance(0.5, 0.0), 1.0, "dist=0 必须全透");

    // 近真空：σt → 0⁺ → 自由程巨大 → 有限距离内恒透射。
    let vacuum = HomogeneousMedium::new(1.0e-30, 0.5, 0.0);
    let fpv = vacuum.free_path(0.5);
    assert!(fpv > 1.0e28, "近真空自由程 {fpv} 应巨大");
    assert_eq!(vacuum.shadow_transmittance(0.5, 1.0e6), 1.0, "近真空 1e6 距离内必须透射");

    // σs/σa 一致性在两个极端下不产生 NaN/Inf。
    for m in [&dense, &vacuum] {
        assert!(m.sigma_s().is_finite(), "σs 溢出");
        assert!(m.sigma_a().is_finite(), "σa 溢出");
    }

    // det_ln 次正规下界不 panic、结果有限（det 件自检）。
    let tiny = f32::from_bits(1); // 最小正次正规
    let l = prism_rt::media::det_ln(tiny);
    assert!(l.is_finite() && l < 0.0, "det_ln(最小次正规) = {l} 应为有限负值");
}

// ---------------------------------------------------------------------------
// ⑤ 相机 fov 边界 + 光圈 = 0 针孔恒等
// ---------------------------------------------------------------------------

/// Vec3 逐位指纹（f32 无 Vec3::to_bits，逐分量取）。
fn vbits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

#[test]
fn camera_fov_bounds_enforced() {
    let ok = |fov: f32| {
        Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            fov,
            0.0,
            1.0,
            0.0,
            0.0,
        )
    };
    assert!(std::panic::catch_unwind(|| ok(0.0)).is_err(), "fov=0 必须拒绝");
    assert!(std::panic::catch_unwind(|| ok(180.0)).is_err(), "fov=180 必须拒绝");
    assert!(std::panic::catch_unwind(|| ok(-30.0)).is_err(), "fov<0 必须拒绝");
    // 1° / 179° 合法（C17 豁免以度为单位的行业约定仅限 fov）。
    let near = ok(1.0);
    let wide = ok(179.0);
    // 中心方向恒 forward；tan_half_fov 单调。
    assert!(near.tan_half_fov > 0.0 && near.tan_half_fov < 0.01, "fov 1° 半角正切应极小");
    assert!(wide.tan_half_fov > 57.0, "fov 179° 半角正切应巨大（tan(89.5°)≈114.6）");
    assert!(wide.tan_half_fov > near.tan_half_fov * 1000.0);
}

#[test]
fn pinhole_aperture_zero_bitwise_identity() {
    // 光圈 = 0：射线与 focus_distance / shutter 采样完全无关（针孔快路径）。
    // 仅 focus_distance 不同的两台相机必须产生逐位一致的相机光线。
    let mk = |focus: f32| {
        Camera::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            focus,
            0.0,
            0.0,
        )
    };
    let a = mk(3.0);
    let b = mk(100.0);
    let ctx = EvalCtx::new(7, 1, 0);
    let res = (64_u32, 48_u32);
    for sample in 0..8_u32 {
        let stream = SampleStream::new(&ctx, (5, 9), sample);
        let ra = prism_rt::integrator::cast_ray(&a, &stream, res);
        let rb = prism_rt::integrator::cast_ray(&b, &stream, res);
        assert_eq!(vbits(ra.origin), vbits(rb.origin), "针孔 origin 逐位一致");
        assert_eq!(vbits(ra.dir), vbits(rb.dir), "针孔方向逐位一致");
        assert_eq!(ra.time.to_bits(), rb.time.to_bits(), "针孔 time 逐位一致");
        // 针孔视点恒 eye（逐位）。
        assert_eq!(vbits(ra.origin), vbits(a.eye), "针孔 origin == eye");
        // 方向单位化。
        assert!((ra.dir.length() - 1.0).abs() < 1e-5, "针孔方向应单位化");
        // 抖动维度正确性：中心像素的抖动应小（不越出 ±0.5 像素格）。
        let (jx, jy) = stream.sample_2d(DIM_PIXEL);
        assert!((0.0..1.0).contains(&jx) && (0.0..1.0).contains(&jy), "像素抖动越界");
    }
}
