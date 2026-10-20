//! random walk SSS：—— 次表面随机游走。
//!
//! 路径：Fresnel 折射进入介质 → σ̄ 自由程随机游走（RGB 平均自由程
//! `radius` 参数）→ HG 相函数改向（复用 [`crate::media`]）→ 表面事件按
//! Fresnel 概率逃逸/内反射 → 逃逸点作为新着色点（Lambert 漫反 + NEE，
//! 由 integrator 处理）。**约束：逃逸点禁二次 SSS 入射**（防路径树爆炸，
//! ）——integrator 挂接时以 `from_sss` 标志保证。
//!
//! # 出处（数据资产清单 #4 注释义务）
//! - random walk BSSRDF 路径框架：E. Veach, "Robust Monte Carlo Methods for
//!   Light Transport Simulation", PhD thesis, Stanford 1997（参与介质路径
//!   积分）；工程参照：Pixar RenderMan 22+ / Blender Cycles 的 random walk
//!   SSS（工业收敛设计：σ̄ 采样 + 逐通道吞吐）。
//! - `radius` = RGB 平均自由程参数语义：P. Christensen, B. Burley,
//!   "Approximate Reflectance Profiles for Efficient Subsurface Scattering",
//!   Pixar Technical Memo 15-04, 2015。
//! - HG 相函数：L. G. Henyey, J. L. Greenstein, "Diffuse reflection by
//!   large particles", ApJ 93, 1941（实现复用 [`crate::media`]）。
//!
//! # 无偏性与确定性
//! - σ̄ = 三通道 σt 均值做自由程采样，逐通道吞吐补
//!   exp(−(σc−σ̄)·s) 修正（Cycles 同款取舍；谱域逐通道独立游走列为
//!   spectral 扩展位）。
//! - 内部散射吞吐 ×= 反照率（单次散射着色 σs/σt = base_color）。
//! - 全部随机数走 aux 游程流（(槽位, step) 纯函数寻址，不占 Sobol 维度
//!  预算； hash 派生）。
//! - [`det_exp`]：位分解 + 泰勒级数，D-29 合规（与 [`crate::media::det_ln`]
//!   同风格自足件，后续随 prism-math D-29 数学件上收）。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;

use crate::bsdf::{hadamard, reflect_dir, refract_dir, schlick};
use crate::media::HomogeneousMedium;
use crate::sampler::aux_1d;

/// 游走步数上限（防退化几何死循环；超出按吸收处理）。
pub const SSS_MAX_STEPS: u32 = 16;
/// 每步 aux 预算（0 自由程 / 1,2 相函数 / 3 RR / 4 Fresnel，余量 3）。
const AUX_PER_STEP: u32 = 8;
/// 表面推进防自交偏移（相对平均自由程）。
const PUSH_EPS: f32 = 1e-4;

/// SSS 逃逸结果：逃逸点作为新着色点。
#[derive(Debug, Clone, Copy)]
pub struct SssExit {
    /// 逃逸点（世界空间）。
    pub p: Vec3,
    /// 逃逸面外向法线（与出射方向同侧）。
    pub n: Vec3,
    /// 内部全程吞吐（含逐通道透射衰减、散射反照率、出射 Fresnel 透过；
    /// **不含入射 Fresnel**——由调用方的 BsdfSample.weight 承担）。
    pub weight: Vec3,
}

/// 确定性指数 exp(x)（D-29）：x = k·ln2 + r（k 由 floor 精确归约，
/// r ∈ [−ln2/2, ln2/2]）→ exp(r) 泰勒 7 阶（余项 r⁸/8! < 6e-9）× 2^k 位注入。
/// 只含 ±×÷ 与整数位运算；floor/sqrt 为 IEEE 精确舍入运算，不在禁列。
#[inline]
pub fn det_exp(x: f32) -> f32 {
    // D-30：原硬编码 0.693_147_180_559_945_3 收窄值，与 `std::f32::consts::LN_2`
    // `to_bits()` 逐位相同；着色器侧 `det_exp` 用同一数值，**改值需同步改 WGSL**。
    const LN_2: f32 = std::f32::consts::LN_2;
    // 下溢到零（exp 恒 > 0，绝不产出负值）：x ≤ −87 时单精度 exp 必下溢，
    // 直接返回 0 以保证输出 ≥ 0——否则 clamp 会把 k 钉在 −126，破坏 Taylor
    // 归约域 r ∈ [−ln2/2, ln2/2]，泰勒在远离 0 处失灵甚至翻负。
    if x <= -87.0 {
        return 0.0;
    }
    let kf = (x / LN_2 + 0.5).floor();
    let k = kf.clamp(-126.0, 127.0);
    let r = x - k * LN_2;
    // exp(r) = Σ rⁿ/n!，Horner 排布到 1/7!。
    let e = 1.0
        + r
            * (1.0
                + r * (0.5
                    + r * (1.0 / 6.0
                        + r * (1.0 / 24.0
                            + r * (1.0 / 120.0
                                + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let scale = f32::from_bits(((k as i32 + 127) as u32) << 23);
    e * scale
}

/// random walk 次表面散射。
///
/// - `geo`：场景求交闭包（o, d) → (t, 外向法线)；integrator 侧由
///   `Scene::intersect_hit` 适配。
/// - `p_in`：入射点（表面）；`n`：入射面外向法线；`w_in`：介质内初始行进
///   方向（指向内部，来自折射分支）。
/// - `radius`：RGB 平均自由程；`albedo`：内部单次散射反照率（逐通道 σs/σt，
///   取材质 base_color）；`ior`：介质折射率；`g`：HG 各向异性。
/// - `si`：样本流基准索引（**aux 槽位放 `si.dimension`**，sampler aux 流
///   设计约定；step 由游走内部递增）。
///
/// 返回 None = 被吸收 / 步数耗尽 / 内部脱靶。
#[allow(clippy::too_many_arguments)]
pub fn random_walk<F>(
    geo: &F,
    ctx: &EvalCtx,
    si: &SampleIndex,
    p_in: Vec3,
    n: Vec3,
    w_in: Vec3,
    radius: Vec3,
    albedo: Vec3,
    ior: f32,
    g: f32,
) -> Option<SssExit>
where
    F: Fn(Vec3, Vec3) -> Option<(f32, Vec3)>,
{
    let _ = n; // 入射面法线仅用于调用方入射权重；游走从 p_in 沿 w_in 起步
    let sigma_t = Vec3::new(1.0 / radius.x, 1.0 / radius.y, 1.0 / radius.z);
    let sigma_bar = (sigma_t.x + sigma_t.y + sigma_t.z) * (1.0 / 3.0);
    // 复用 media 的自由程/HG 件（albedo=1：透射衰减与散射着色手动逐通道处理）。
    let med = HomogeneousMedium::new(sigma_bar, 1.0, g);
    let push = PUSH_EPS * (radius.x + radius.y + radius.z) * (1.0 / 3.0);
    let mut p = p_in + w_in * push; // 推进入射面，防 t=0 自命中
    let mut w = w_in;
    let mut tp = Vec3::splat(1.0);
    // RR 存活概率 = 反照率上界（能量上界，Veach 1997 RR 语义）。
    let p_survive = albedo.x.max(albedo.y).max(albedo.z).max(1e-3);
    // 电介质 F0（出射 Fresnel 用）。
    let r0 = (ior - 1.0) / (ior + 1.0);
    let f0 = Vec3::splat(r0 * r0);

    for step in 0..SSS_MAX_STEPS {
        let base = step * AUX_PER_STEP;
        let aux = |k: u32| aux_1d(ctx, si, base + k);
        // 沿行进方向的最近表面。
        let (t_surf, n_surf) = match geo(p, w) {
            Some(h) => h,
            None => return None, // 内部脱靶（退化场景）：吸收
        };
        // σ̄ 自由程采样。
        let s = med.free_path(aux(0));
        let travel = s.min(t_surf);
        // 逐通道透射修正：σc 通道相对 σ̄ 采样的偏差衰减。
        tp = Vec3::new(
            tp.x * det_exp(-(sigma_t.x - sigma_bar) * travel),
            tp.y * det_exp(-(sigma_t.y - sigma_bar) * travel),
            tp.z * det_exp(-(sigma_t.z - sigma_bar) * travel),
        );
        if s < t_surf {
            // ---- 内部散射事件（吞吐 ×= 反照率；T 与 pdf 相消，无额外权重）----
            tp = hadamard(tp, albedo);
            w = med.sample_phase(w, aux(1), aux(2));
            p = p + w * s;
            // 俄式轮盘（存活概率 = 能量上界）。
            if aux(3) > p_survive {
                return None;
            }
        } else {
            // ---- 表面事件：Fresnel 概率逃逸 / 内反射（概率与权重相消）----
            let p_edge = p + w * t_surf;
            let cos_i = w.dot(&n_surf).max(0.0); // 朝外余弦
            let refr = refract_dir(w, -n_surf, ior); // 内 → 外：n 取内法线，eta = ior
            let f_refl = schlick(f0, cos_i).x;
            if refr.is_none() || aux(4) < f_refl {
                // TIR / Fresnel 内反射：吞吐不变（F 概率与权重相消）。
                w = reflect_dir(w, n_surf);
                p = p_edge + w * (push.max(1e-6));
            } else {
                // 逃逸：出射 Fresnel 透过，逃逸点作为新着色点。
                tp = tp * (1.0 - f_refl);
                return Some(SssExit {
                    p: p_edge,
                    n: n_surf,
                    weight: tp,
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 测试（det_exp 锚点 / 平板穿透单调 / 确定性 / 吸收）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// z ∈ [0, T] 无限平板求交闭包（两面解析求交）。
    fn slab(t: f32) -> impl Fn(Vec3, Vec3) -> Option<(f32, Vec3)> {
        move |o: Vec3, d: Vec3| {
            if d.z.abs() <= 1e-9 {
                return None;
            }
            let mut best: Option<(f32, Vec3)> = None;
            // z = 0 面（外法线 −z）。
            let t0 = (0.0 - o.z) / d.z;
            if t0 > 1e-6 {
                best = Some((t0, Vec3::new(0.0, 0.0, -1.0)));
            }
            // z = T 面（外法线 +z）。
            let t1 = (t - o.z) / d.z;
            if t1 > 1e-6 && best.map_or(true, |(tb, _)| t1 < tb) {
                best = Some((t1, Vec3::new(0.0, 0.0, 1.0)));
            }
            best
        }
    }

    #[test]
    fn det_exp_matches_f64_reference() {
        // 锚点（测试参考值用 f64 exp——libm 禁令不约束测试，media 同惯例）。
        for &x in &[0.0f32, -0.5, -2.0, -5.0, -13.8, 0.7, 3.0] {
            let got = det_exp(x);
            let want = (x as f64).exp() as f32;
            let rel = (got - want).abs() / want;
            assert!(rel < 1e-5, "exp({x}) = {got}，参考 {want}，rel={rel}");
        }
        assert_eq!(det_exp(0.0).to_bits(), 1.0f32.to_bits());
        // 确定性：两遍逐位一致。
        assert_eq!(det_exp(-1.37).to_bits(), det_exp(-1.37).to_bits());
        // 单调性（数学性质抽查）。
        assert!(det_exp(-2.0) < det_exp(-1.0));
        assert!(det_exp(-1.0) < det_exp(-0.5));
    }

    #[test]
    fn sss_slab_transmission_is_monotonic_in_thickness() {
        // 平板穿透率随厚度单调递减（衰减随厚度单调；各向同性 g=0）。
        let ctx = EvalCtx::new(0x5A5_001, 0, 0);
        let radius = 1.0f32;
        let albedo = Vec3::splat(0.9);
        let ior = 1.4;
        let ns = 20_000u32;
        let mut prev_rate = f32::INFINITY;
        for &t in &[0.25f32, 0.5, 1.0, 2.0, 4.0] {
            let geo = slab(t);
            let si0 = SampleIndex {
                frame: 0,
                pixel: (0, 0),
                sample: 0,
                dimension: 0,
                node_id: 0xBEEF,
            };
            // 入射：从 z<0 侧打到 z=0 面（外法线 −z），正入射折射进入 +z。
            let w_in = refract_dir(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, -1.0),
                1.0 / ior,
            )
            .expect("正入射必折射");
            let mut escaped = 0u32;
            for k in 0..ns {
                // 每样本独立 aux 流：sample 字段区分（纯函数寻址，无状态）。
                let si_k = SampleIndex {
                    sample: k,
                    ..si0
                };
                if random_walk(&geo, &ctx, &si_k, Vec3::splat(0.0), Vec3::new(0.0, 0.0, -1.0), w_in, Vec3::splat(radius), albedo, ior, 0.0).is_some() {
                    escaped += 1;
                }
            }
            let rate = escaped as f32 / ns as f32;
            println!("平板 T={t}·r: 穿透率={rate:.4}");
            assert!(
                rate < prev_rate,
                "穿透率必须随厚度单调递减：T={t} rate={rate} prev={prev_rate}"
            );
            prev_rate = rate;
        }
        // 量级锚点：厚板（T=4r）几乎不透射，薄板（T=0.25r）多数透射。
        assert!(prev_rate < 0.25, "T=4r 穿透率应很小：{prev_rate}");
    }

    #[test]
    fn sss_is_deterministic() {
        // 同参数两遍逐位一致（纯函数 [D]）。
        let geo = slab(1.0);
        let ctx = EvalCtx::new(0xD11_CE, 0, 0);
        let si = SampleIndex { frame: 2, pixel: (3, 4), sample: 5, dimension: 0, node_id: 99 };
        let w_in = Vec3::new(0.1, 0.0, 0.99).normalize();
        let walk = |si: &SampleIndex| {
            random_walk(
                &geo, &ctx, si,
                Vec3::splat(0.0),
                Vec3::new(0.0, 0.0, 1.0),
                w_in,
                Vec3::splat(1.0),
                Vec3::splat(0.85),
                1.4,
                0.3,
            )
        };
        match (walk(&si), walk(&si)) {
            (Some(x), Some(y)) => {
                assert_eq!(x.p.x.to_bits(), y.p.x.to_bits());
                assert_eq!(x.p.z.to_bits(), y.p.z.to_bits());
                assert_eq!(x.n.z.to_bits(), y.n.z.to_bits());
                assert_eq!(x.weight.y.to_bits(), y.weight.y.to_bits());
            }
            (None, None) => {} // 一致吸收也算确定
            _ => panic!("同参数游走结果必须一致"),
        }
    }

    #[test]
    fn sss_absorbs_with_dark_albedo() {
        // 反照率 → 0：绝大多数游走被吸收（None），透射权重保持 ≤ 1。
        let geo = slab(2.0);
        let ctx = EvalCtx::new(0xB1A_C0, 0, 0);
        let si0 = SampleIndex { frame: 0, pixel: (0, 0), sample: 0, dimension: 0, node_id: 7 };
        let w_in = Vec3::new(0.0, 0.0, 1.0);
        let mut absorbed = 0u32;
        let n = 8_000u32;
        for k in 0..n {
            let si = SampleIndex { sample: k, ..si0 };
            if let Some(e) = random_walk(&geo, &ctx, &si, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), w_in, Vec3::splat(1.0), Vec3::splat(0.05), 1.4, 0.0) {
                assert!(e.weight.x <= 1.0 + 1e-5, "吞吐越界：{e:?}");
            } else {
                absorbed += 1;
            }
        }
        let rate = absorbed as f32 / n as f32;
        assert!(rate > 0.6, "深色材质吸收率应高：{rate}");
    }
}
