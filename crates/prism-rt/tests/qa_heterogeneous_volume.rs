//! 异质体积介质攻击性测试（C2，#10 实施子集）。
//!
//! 核心判据（PRD C2 验收 1）= **退化一致性**：σt 恒定场下，
//! `HeterogeneousMedium` 的 `free_path` / `shadow_transmittance` 与
//! `HomogeneousMedium` **逐位一致**（同 `det_ln` / 同 `clamp` / 同 `U_EPS`）。
//! 另含无偏性（Woodcock delta tracking 估计量 E[T̂] = Beer 定律）与
//! 参数域防御。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;
use prism_rt::media::{Grid3, HeterogeneousMedium, HomogeneousMedium};
use prism_rt::sampler;

/// 退化一致性：恒定 σt 场下自由程与均匀介质逐位相同。
#[test]
fn degenerate_free_path_bit_identical_to_homogeneous() {
    let o = Vec3::new(0.0, 0.0, 0.0);
    let d = Vec3::new(1.0, 0.0, 0.0);
    let ctx = EvalCtx::new(31, 7, 3);
    // 多个 σt 档位都应逐位一致（majorant = σt ⇒ 首步即 -ln(u)/σt）。
    for &s in &[0.3f32, 0.8, 1.5, 2.7] {
        let homo = HomogeneousMedium::new(s, 0.9, 0.2);
        let hetero = HeterogeneousMedium::constant_sigma_t(s, 0.9, 0.2);
        for k in 0..4096u32 {
            let si = SampleIndex {
                frame: 3,
                pixel: (5, 6),
                sample: k,
                dimension: 40,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            let h = homo.free_path(u).to_bits();
            let he = hetero.free_path(o, d, u, 0.5).to_bits();
            assert_eq!(h, he, "退化自由程失配 s={s} k={k} u={u}");
        }
    }
}

/// 退化一致性：恒定 σt 场下阴影透射率与均匀介质逐位相同（二值估计）。
#[test]
fn degenerate_shadow_bit_identical_to_homogeneous() {
    let o = Vec3::new(0.0, 0.0, 0.0);
    let d = Vec3::new(1.0, 0.0, 0.0);
    let dist = 1.3f32;
    let ctx = EvalCtx::new(97, 7, 3);
    for &s in &[0.3f32, 0.7, 1.5] {
        let homo = HomogeneousMedium::new(s, 1.0, 0.0);
        let hetero = HeterogeneousMedium::constant_sigma_t(s, 1.0, 0.0);
        for k in 0..4096u32 {
            let si = SampleIndex {
                frame: 3,
                pixel: (1, 1),
                sample: k,
                dimension: 41,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            let h = homo.shadow_transmittance(u, dist).to_bits();
            let he = hetero.shadow_transmittance(o, d, dist, u, 0.5).to_bits();
            assert_eq!(h, he, "退化阴影失配 s={s} k={k} u={u}");
        }
    }
}

/// 无偏性：沿 +x 线性变化 σt 场的阴影透射率均值 = Beer 定律（解析积分）。
///
/// Woodcock delta tracking 估计量 E[T̂] = exp(−∫σt ds)；用**逐步独立**均匀对
/// 喂多步跟踪，均值应贴合 Beer，误差 < 2%。
///
/// # 逐步均匀的来源（取证结论，勿回退）
/// 必须走 [`sampler::aux_1d`]——采样器模块正是为「介质 delta tracking 步进等
/// 变长随机序列」定义的入口（PCG 纯函数流，按 (维度槽位, 步序号) 区分）。
/// 曾用 `sample_1d` 取 Sobol 维 50/51 且样本序号跨步 12（`k*12+j`）：
/// Sobol 相邻点按构造负相关，同一条路径的 12 步因此互相锁相，「逐步独立」
/// 前提被破坏——实测同场同网格下 E[T̂]=0.126953（Beer=0.135335，**−6.19%**），
/// 而独立 PRNG 同条件为 0.135010（−0.24%）、aux_1d 为 0.136140（N=3e6 收敛
/// 至 +0.09%）。即偏差来自喂入源违规，不是估计量有偏。
#[test]
fn shadow_transmittance_unbiased_vs_beer() {
    // 沿 +x 的线性变化 σt 场：8 格、每格 0.25、x∈[0,2]，σt 由 s0→s1。
    let nx = 8usize;
    let s0 = 0.4f32;
    let s1 = 1.6f32;
    let mut data = Vec::with_capacity(nx);
    for i in 0..nx {
        let frac = (i as f32 + 0.5) / nx as f32;
        data.push(s0 + (s1 - s0) * frac);
    }
    let grid = Grid3::new(
        nx,
        1,
        1,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.25, 1.0, 1.0),
        data.clone(),
    );
    let m = HeterogeneousMedium::new(grid, 0.8, 0.0);
    let o = Vec3::new(0.0, 0.0, 0.0);
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let dist = 2.0f32;

    // 解析 ∫σt ds = Σ σt_i · cell_w：格**心**口径下射线 x∈[0,2] 恰好整格贯穿
    // 8 格（跨度 nx·cell_w = 2.0），线性场在中点取样 ⇒ 积分精确。
    let cell_w = 0.25f32;
    let integral: f64 = data.iter().map(|&v| v as f64 * cell_w as f64).sum();
    let beer = (-integral).exp();

    // MC：每样本用 12 对独立均匀（majorant·dist ≈ 3.2，12 步充裕）。
    let ctx = EvalCtx::new(123, 7, 3);
    let n = 300_000u32;
    let mut ones = 0u32;
    for k in 0..n {
        let si = SampleIndex {
            frame: 3,
            pixel: (7, 7),
            sample: k,
            dimension: 50,
            node_id: 7,
        };
        let mut uni = Vec::with_capacity(12);
        for j in 0..12u32 {
            // 逐步独立游程均匀（见测试文档「逐步均匀的来源」）。
            uni.push((
                sampler::aux_1d(&ctx, &si, 2 * j),
                sampler::aux_1d(&ctx, &si, 2 * j + 1),
            ));
        }
        let t = m.shadow_transmittance_tracked(o, dir, dist, &uni);
        assert!(t == 0.0 || t == 1.0, "二值估计出现非二值 {t}");
        ones += t as u32;
    }
    let est = ones as f64 / n as f64;
    assert!(
        (est - beer).abs() / beer < 0.02,
        "异质体积阴影透射率无偏性失败：E[T̂]={est}，Beer={beer}（∫σt={integral}）"
    );
}

/// 相函数归一化（异质介质 g 空间不变，与均匀同款 HG）。
#[test]
fn heterogeneous_phase_integrates_to_one() {
    let m = HeterogeneousMedium::constant_sigma_t(1.0, 1.0, 0.3);
    let ctx = EvalCtx::new(11, 7, 3);
    let n = 40_000u32;
    let mut sum_p = 0.0f64;
    for k in 0..n {
        let si = SampleIndex {
            frame: 3,
            pixel: (2, 2),
            sample: k,
            dimension: 42,
            node_id: 7,
        };
        let (u1, u2) = sampler::sample_2d(&ctx, &si);
        // 均匀球面方向。
        let z = 1.0 - 2.0 * u1;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let (c, s) = prism_rt::sampler::cos_sin(2.0 * std::f32::consts::PI * u2);
        let dir = Vec3::new(r * c, r * s, z);
        let cos = dir.z as f64;
        sum_p += 4.0 * std::f64::consts::PI * m.phase(cos as f32) as f64;
    }
    let norm = sum_p / n as f64;
    assert!((norm - 1.0).abs() < 0.01, "异质 HG ∫p dΩ = {norm}");
}

/// 参数域防御：σt 场全真空（原始上界 = 0）必须 panic，而非静默错误。
///
/// 注意 `majorant()` 带 U_EPS 地板（除零安全），恒 > 0，**不能**作真空判据；
/// 守卫用 `raw_max()`。见 `grid3_raw_max_excludes_eps_floor`。
#[test]
#[should_panic(expected = "σt 场上界须为正")]
fn heterogeneous_rejects_zero_majorant_field() {
    let grid = Grid3::new(2, 2, 2, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), vec![0.0; 8]);
    let _ = HeterogeneousMedium::new(grid, 0.8, 0.0);
}

/// 参数域防御：Grid3 数据长度必须匹配 nx*ny*nz。
#[test]
#[should_panic]
fn grid3_rejects_mismatched_data_length() {
    let _ = Grid3::new(2, 2, 2, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), vec![0.5; 7]);
}
