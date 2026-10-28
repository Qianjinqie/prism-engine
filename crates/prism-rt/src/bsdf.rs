//! Disney BSDF：specular / diffuse / transmission 三瓣 + 多散射补偿。
//!
//! # 论文出处（数据资产清单 #4：代码注释注明出处义务）
//! - Disney 参数化（metalness / 感知 roughness / FD90 漫反修正）：B. Burley,
//!   "Physically-Based Shading at Disney", SIGGRAPH 2012 Course: Practical
//!   Physically Based Shading in Film and Game Production。
//! - GGX 微表面分布与 Smith 遮蔽项：B. Walter, S. Marschner, H. Li, K. Torrance,
//!   "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007
//!   （D 项 Eq. 33、G1 项 Eq. 34）。
//! - Schlick Fresnel 近似：C. Schlick, "An Inexpensive BRDF Model for
//!   Physically-based Lighting", Computer Graphics Forum 13(3), 1994。
//! - VNDF 采样：E. Heitz, "Sampling the GGX Distribution of Visible Normals",
//!   JCGT 7(4), 2018——[`sample_vndf`] 逐步对应官方参考实现
//!   `sampleGGXVNDF.h`（含 2019-06-17 errata 的零叉积特例）。
//! - 多散射能量补偿框架：C. Kulla, A. Conty, "Revisiting Physically Based
//!   Shading at Imageworks", SIGGRAPH 2017 Course——取其"单散射反照率预积分 +
//!   亏损回填"结构；f0 影响按两端点（f0=0 / f0=1）线性插值、回填密度方向
//!  解耦，"解析近似按比例回填"的字面执行（简化声明：非完整
//!   Kulla-Conty 双线性形式）。
//!
//! # glTF 同构（`gltf.rs` PbrMaterial 字段一一对应，零转换）
//! `base_color` ↔ baseColorFactor、`metalness` ↔ metallicFactor、`roughness` ↔
//! roughnessFactor、`ior` ↔ KHR_materials_ior。α = roughness²（Burley 感知映射）；
//! 电介质 F0 = ((ior−1)/(ior+1))²（ior=1.5 → 0.04，与 glTF 默认 specular 常数一致）。
//!
//! # 确定性（D-29 / 格锚表）
//! 全部公式只含 ±×÷√（sqrt 为 IEEE 精确舍入运算，不在禁列）；方位角三角走
//! [`crate::sampler`] 的 det 三角件；整数幂一律连乘（禁 pow）；能量表由 Sobol
//! 纯函数流一次性构造（OnceLock，样本流 = hash 派生，无全局 RNG）。

use std::f32::consts::{FRAC_1_PI, PI};
use std::sync::OnceLock;

use prism_core::{EvalCtx, SampleIndex};
use prism_math::{clamp, Vec3};

use crate::sampler::{cos_sin, cosine_hemisphere, onb, sample_2d};

// ---------------------------------------------------------------------------
// 公共向量件（integrator 复用；B1 曾在 integrator 私有定义，B2 上收至此）
// ---------------------------------------------------------------------------

/// 分量乘（Hadamard 积；prism-math Vec3 未提供）。
#[inline]
pub fn hadamard(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// 亮度（Rec.709 权重）。
#[inline]
pub fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// x⁵ 连乘（D-29：整数幂禁 pow，一律连乘）。
#[inline]
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

// ---------------------------------------------------------------------------
// 微表面基础件（Walter 2007 / Schlick 1994）
// ---------------------------------------------------------------------------

/// Schlick Fresnel 近似（Schlick 1994）：F = F0 + (1−F0)(1−cosθ)⁵。
#[inline]
pub fn schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let m5 = pow5((1.0 - cos_theta).max(0.0));
    f0 + (Vec3::splat(1.0) - f0) * m5
}

/// GGX 法线分布 D（Walter et al. 2007, Eq. 33）：α²/(π·((n·h)²(α²−1)+1)²)。
#[inline]
pub fn ggx_d(alpha: f32, cos_theta_m: f32) -> f32 {
    let c2 = cos_theta_m * cos_theta_m;
    let den = c2 * (alpha * alpha - 1.0) + 1.0;
    alpha * alpha / (PI * den * den)
}

/// GGX 单散射核心（Walter 2007）：F·D·G/(4·cosθo·cosθi)。
/// eval 与能量表预积分共用（保证两处严格同式）。
#[inline]
fn spec_scatter(wo: Vec3, wi: Vec3, n: Vec3, alpha: f32, f0: Vec3) -> Vec3 {
    let cos_o = wo.dot(&n).max(0.0);
    let cos_i = wi.dot(&n).max(0.0);
    if cos_o <= 0.0 || cos_i <= 0.0 {
        return Vec3::splat(0.0);
    }
    let h = (wo + wi).normalize();
    let cos_h = h.dot(&n);
    if cos_h <= 0.0 {
        return Vec3::splat(0.0);
    }
    let g = smith_g1(alpha, cos_o) * smith_g1(alpha, cos_i);
    let f_h = schlick(f0, wo.dot(&h).max(0.0));
    f_h * (ggx_d(alpha, cos_h) * g / (4.0 * cos_o * cos_i))
}

/// Smith 单向遮蔽 G1（GGX 形式，Walter et al. 2007, Eq. 34）。
#[inline]
pub fn smith_g1(alpha: f32, cos_theta: f32) -> f32 {
    let c2 = cos_theta * cos_theta;
    let den = cos_theta + (alpha * alpha + (1.0 - alpha * alpha) * c2).sqrt();
    2.0 * cos_theta / den
}

/// Snell 折射（确定性；eta = 入射侧 ior / 出射侧 ior）。
/// 返回 None = 全内反射（TIR）。w_in 指向表面，n 与出射侧同向。
#[inline]
pub fn refract_dir(w_in: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = -(w_in.dot(&n));
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some(w_in * eta + n * (eta * cos_i - cos_t))
}

/// 镜面反射方向。
#[inline]
pub fn reflect_dir(w_in: Vec3, n: Vec3) -> Vec3 {
    w_in - n * (2.0 * w_in.dot(&n))
}

// ---------------------------------------------------------------------------
// GGX VNDF 采样（Heitz 2018，官方 sampleGGXVNDF.h 逐步对应）
// ---------------------------------------------------------------------------

/// 在局部坐标系（+z = 几何法线）内采样 GGX VNDF 法线。
/// 输入 `ve_local` = 局部视线方向（z > 0）；返回微表面法线。
fn sample_vndf(ve_local: Vec3, alpha: f32, u1: f32, u2: f32) -> Vec3 {
    // 拉伸到半球配置。
    let vh = Vec3::new(alpha * ve_local.x, alpha * ve_local.y, ve_local.z).normalize();
    // 正交基（lensq = 0 零叉积特例，errata 2019-06-17）。
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) * (1.0 / lensq.sqrt())
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t2 = vh.cross(&t1);
    // 投影圆盘参数化。
    let r = u1.sqrt();
    let (c, s) = cos_sin(2.0 * PI * u2);
    let t1p = r * c;
    let mut t2p = r * s;
    let sc = 0.5 * (1.0 + vh.z);
    t2p = (1.0 - sc) * (1.0 - t1p * t1p).max(0.0).sqrt() + sc * t2p;
    // 重投影回半球。
    let nz = (1.0 - t1p * t1p - t2p * t2p).max(0.0).sqrt();
    let nh = t1 * t1p + t2 * t2p + vh * nz;
    // 反拉伸回椭球配置。
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize()
}

// ---------------------------------------------------------------------------
// 多散射能量补偿表（Kulla-Conty 2017 框架的简化形式）
// ---------------------------------------------------------------------------

/// 能量表的 μ（cosθ）格数（均匀格）。
const E_MU: usize = 8;
/// 能量表的 α 格点（固定数学格点，非外部数据资产）。
const E_ALPHA: [f32; 8] = [0.05, 0.1, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0];
/// 每格预积分样本数（Sobol 纯函数流；相对误差 ~2%，守恒断言容差 3%）。
const E_SAMPLES: u32 = 2048;

/// 单散射方向反照率表：E_dir(μ, α) 在 f0 = 0 与 f0 = 1 两端点的预积分。
#[derive(Debug, Clone, Copy)]
struct EnergyTable {
    /// E(f0=0)：F = (1−cosθh)⁵（grazing 仍有反射）。
    e0: [[f32; E_ALPHA.len()]; E_MU],
    /// E(f0=1)：F ≡ 1。
    e1: [[f32; E_ALPHA.len()]; E_MU],
    /// 方向平均反照率 E_avg = (1/π)∫E_dir cosμ dΩ（μ 格梯形积分）。
    avg0: [f32; E_ALPHA.len()],
    avg1: [f32; E_ALPHA.len()],
}

/// 用 GGX+Smith 单散射白炉积分填表：E_dir = ∫f_spec·cos dωi。
/// 采样走 **VNDF 重要性采样**（E[weight] = ∫f·cos，无偏）——余弦采样在
/// 低 α（近镜面 lobe 占半球 <0.2%）时严重欠采样，VNDF 对全部 α 低方差。
/// ωo 取仰角 μ、方位 0——GGX 各向同性，方位对称性保证方位无关。
fn build_energy_table() -> EnergyTable {
    let ctx = EvalCtx::new(0xB2C4_0001, 0, 0);
    let mut t = EnergyTable {
        e0: [[0.0; E_ALPHA.len()]; E_MU],
        e1: [[0.0; E_ALPHA.len()]; E_MU],
        avg0: [0.0; E_ALPHA.len()],
        avg1: [0.0; E_ALPHA.len()],
    };
    let n = Vec3::new(0.0, 0.0, 1.0);
    for (ai, &alpha) in E_ALPHA.iter().enumerate() {
        for mi in 0..E_MU {
            let mu_o = ((mi as f32) + 0.5) / E_MU as f32;
            let sin_o = (1.0 - mu_o * mu_o).sqrt();
            let wo = Vec3::new(sin_o, 0.0, mu_o);
            for f0_case in 0..2 {
                let f0 = if f0_case == 0 {
                    Vec3::splat(0.0)
                } else {
                    Vec3::splat(1.0)
                };
                let mut sum = 0.0f64;
                for k in 0..E_SAMPLES {
                    let si = SampleIndex {
                        frame: 0,
                        pixel: (0, 0),
                        sample: k,
                        dimension: (ai * 16 + mi * 2 + f0_case) as u16,
                        node_id: 0,
                    };
                    let (u1, u2) = sample_2d(&ctx, &si);
                    // VNDF 采样法线 → 镜面反射方向；pdf = D·G1(v)·(v·h)/(4|v·h|·cos_o)。
                    let h = sample_vndf(wo, alpha, u1, u2);
                    let v_h = wo.dot(&h).max(1e-9);
                    let wi = reflect_dir(-wo, h);
                    let cos_i = wi.dot(&n);
                    if cos_i <= 0.0 {
                        continue; // 采样翻到表面下方：贡献 0
                    }
                    let f = spec_scatter(wo, wi, n, alpha, f0).x;
                    let pdf = ggx_d(alpha, h.dot(&n)) * smith_g1(alpha, mu_o) * v_h
                        / (4.0 * v_h * mu_o.max(1e-6));
                    if pdf <= 0.0 {
                        continue;
                    }
                    sum += (f * cos_i / pdf) as f64;
                }
                let e = (sum / E_SAMPLES as f64) as f32;
                if f0_case == 0 {
                    t.e0[mi][ai] = e;
                } else {
                    t.e1[mi][ai] = e;
                }
            }
        }
        // E_avg = (1/π)∫₀^{2π}∫₀^{π/2} E_dir(μ)·cosμ·sinμ dθ dφ = 2∫E_dir(μ)·μ·dμ。
        let (mut a0, mut a1) = (0.0f64, 0.0f64);
        for mi in 0..E_MU {
            let mu = ((mi as f32) + 0.5) / E_MU as f32;
            a0 += (t.e0[mi][ai] * mu) as f64;
            a1 += (t.e1[mi][ai] * mu) as f64;
        }
        t.avg0[ai] = (2.0 * a0 / E_MU as f64) as f32;
        t.avg1[ai] = (2.0 * a1 / E_MU as f64) as f32;
    }
    t
}

fn energy_table() -> &'static EnergyTable {
    static TABLE: OnceLock<EnergyTable> = OnceLock::new();
    TABLE.get_or_init(build_energy_table)
}

/// α 维格插值（端点 clamp，双格点线性）；返回 (格索引, 格内比例)。
fn alpha_interp(alpha: f32) -> (usize, f32) {
    if alpha <= E_ALPHA[0] {
        return (0, 0.0);
    }
    if alpha >= E_ALPHA[E_ALPHA.len() - 1] {
        return (E_ALPHA.len() - 2, 1.0);
    }
    let mut i = 0usize;
    while i + 2 < E_ALPHA.len() && E_ALPHA[i + 1] < alpha {
        i += 1;
    }
    let span = E_ALPHA[i + 1] - E_ALPHA[i];
    (i, (alpha - E_ALPHA[i]) / span)
}

/// 表插值：E_dir(μ, α, f0v)。
///
/// 返回**域约束到 [0,1]**：方向反照率按定义是能量比，必然 ∈ [0,1]。
/// LUT 由有限样本蒙特卡洛建成，在 α 小、f0≈1 的格点可能估计出略大于 1 的
/// 值（镜面瓣极窄时估计量方差大）；越界会让调用方的 `1−E` 变负，使多散射
/// 补偿**反向抽走**该通道能量——表现是高饱和金属（金/铜）的强通道被抵消，
/// 整体偏成补色（青绿）。域约束在此统一执行，不依赖 LUT 精度。
fn energy_dir(mu: f32, alpha: f32, f0v: f32) -> f32 {
    let t = energy_table();
    let mx = clamp(mu, 0.0, 1.0) * (E_MU as f32 - 1.0);
    let mi = (mx as usize).min(E_MU - 2);
    let mf = mx - mi as f32;
    let (ai, af) = alpha_interp(alpha);
    let e0 = (t.e0[mi][ai] * (1.0 - af) + t.e0[mi][ai + 1] * af) * (1.0 - mf)
        + (t.e0[mi + 1][ai] * (1.0 - af) + t.e0[mi + 1][ai + 1] * af) * mf;
    let e1 = (t.e1[mi][ai] * (1.0 - af) + t.e1[mi][ai + 1] * af) * (1.0 - mf)
        + (t.e1[mi + 1][ai] * (1.0 - af) + t.e1[mi + 1][ai + 1] * af) * mf;
    (e0 * (1.0 - f0v) + e1 * f0v).clamp(0.0, 1.0)
}

/// 表插值：E_avg(α, f0v)。域约束理由同 [`energy_dir`]。
fn energy_avg(alpha: f32, f0v: f32) -> f32 {
    let t = energy_table();
    let (ai, af) = alpha_interp(alpha);
    let a0 = t.avg0[ai] * (1.0 - af) + t.avg0[ai + 1] * af;
    let a1 = t.avg1[ai] * (1.0 - af) + t.avg1[ai + 1] * af;
    (a0 * (1.0 - f0v) + a1 * f0v).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// BSDF 主体
// ---------------------------------------------------------------------------

/// Disney BSDF 参数包（与 glTF PbrMaterial 字段同构零转换，B3 确认）。
#[derive(Debug, Clone, Copy)]
pub struct Bsdf {
    /// 反照率 / 透射着色（线性域）。
    pub base_color: Vec3,
    /// 金属度 0–1（1 = 金属反射着色）。
    pub metalness: f32,
    /// 感知粗糙度 0–1（α = roughness²）。
    pub roughness: f32,
    /// 折射率（KHR_materials_ior；驱动电介质 F0 与透射偏折）。
    pub ior: f32,
    /// 透射权重 0–1（玻璃 / 水）。
    pub transmission: f32,
    /// 次表面权重 0–1（> 0 时透射分支进 [`crate::sss`] 随机游走）。
    pub subsurface: f32,
    /// RGB 平均自由程（random walk radius 参数；逐分量 > 0）。
    pub subsurface_radius: Vec3,
}

impl Bsdf {
    /// 构造（参数域断言，宪章 C19）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_color: Vec3,
        metalness: f32,
        roughness: f32,
        ior: f32,
        transmission: f32,
        subsurface: f32,
        subsurface_radius: Vec3,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&metalness),
            "metalness 必须在 [0,1]，收到 {metalness}；建议：0 = 电介质，1 = 金属"
        );
        assert!(
            (0.0..=1.0).contains(&roughness),
            "roughness 必须在 [0,1]，收到 {roughness}；建议：0 = 镜面，1 = 完全粗糙"
        );
        assert!(
            ior >= 1.0,
            "ior 必须 ≥ 1，收到 {ior}；建议：玻璃 1.5、水 1.33、空气 1.0"
        );
        assert!(
            (0.0..=1.0).contains(&transmission),
            "transmission 必须在 [0,1]，收到 {transmission}"
        );
        assert!(
            (0.0..=1.0).contains(&subsurface),
            "subsurface 必须在 [0,1]，收到 {subsurface}；建议：玉石/皮肤 0.3–1.0"
        );
        assert!(
            subsurface_radius.x > 0.0
                && subsurface_radius.y > 0.0
                && subsurface_radius.z > 0.0,
            "subsurface_radius 必须逐分量为正，收到 {subsurface_radius:?}；建议：RGB 平均自由程，玉石约 0.2–0.6 世界单位"
        );
        Bsdf {
            base_color,
            metalness,
            roughness,
            ior,
            transmission,
            subsurface,
            subsurface_radius,
        }
    }

    /// GGX α（Burley 感知粗糙度平方映射）。
    #[inline]
    pub fn alpha(&self) -> f32 {
        (self.roughness * self.roughness).max(1e-6)
    }

    /// F0（Schlick 反射率）：电介质由 ior 导出，金属取 base_color（glTF 惯例）。
    #[inline]
    pub fn f0(&self) -> Vec3 {
        let n = self.ior;
        let r = (n - 1.0) / (n + 1.0);
        Vec3::splat(r * r).lerp(&self.base_color, self.metalness)
    }

    /// 瓣选择概率（样本流哈希驱动的 lobe 选择 [D]；返回 (spec, diff, trans)）。
    /// `from_inside` 时 diffuse 瓣关闭（介质内无漫反）。
    fn lobe_probs(&self, cos_v: f32, from_inside: bool) -> (f32, f32, f32) {
        let f_v = schlick(self.f0(), cos_v);
        let one_metal = 1.0 - self.metalness;
        let one_trans = 1.0 - self.transmission;
        let w_spec = luminance(f_v) + 1e-4;
        let w_diff = if from_inside {
            0.0
        } else {
            luminance(self.base_color) * one_metal * one_trans
        };
        let w_trans = self.transmission * one_metal;
        let sum = w_spec + w_diff + w_trans;
        if sum <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (w_spec / sum, w_diff / sum, w_trans / sum)
    }

    /// 反射域求值 f(wo, wi)：diffuse（Burley FD90）+ specular（GGX+Smith+Schlick）
    /// + 多散射补偿。wo/wi 都在法线 n 同侧（调用方保证；侧别翻转在调用方）。
    pub fn eval(&self, wo: Vec3, wi: Vec3, n: Vec3) -> Vec3 {
        let cos_o = wo.dot(&n).max(0.0);
        let cos_i = wi.dot(&n).max(0.0);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::splat(0.0);
        }
        let f0 = self.f0();
        let f_v = schlick(f0, cos_o);
        let mut f = Vec3::splat(0.0);

        // diffuse 瓣：电介质漫反（Burley 2012 FD90 修正），逐通道乘 (1−F) 让位镜面。
        let one_metal = 1.0 - self.metalness;
        let one_trans = 1.0 - self.transmission;
        if one_metal * one_trans > 0.0 {
            let h = (wo + wi).normalize();
            let cos_d = h.dot(&n).max(0.0);
            let fd90 = 0.5 + 2.0 * self.roughness * cos_d * cos_d;
            let term = (1.0 + (fd90 - 1.0) * pow5(1.0 - cos_i))
                * (1.0 + (fd90 - 1.0) * pow5(1.0 - cos_o));
            let diff = hadamard(
                hadamard(self.base_color, Vec3::splat(FRAC_1_PI * term)),
                Vec3::splat(1.0) - f_v,
            );
            f = f + diff * (one_metal * one_trans);
        }

        // specular 瓣：GGX D × Smith G × Schlick F / (4 cos cos)。
        let alpha = self.alpha();
        f = f + spec_scatter(wo, wi, n, alpha, f0);

        // 多散射补偿（彩色 F0 形式）：幅度必须取**白炉表**（F0=1）——
        // 那是「被单散射漏掉多少能量」的唯一正确度量；色相由平均 Fresnel
        //   F_avg = f0 + (1−f0)/21（Schlick 的半球平均），
        //   f_ms = (1−E_o)(1−E_i)/(π(1−E_avg)) · F_avg²·E_avg/(1−F_avg(1−E_avg))
        // 承担。F_avg = 1 时逐字退化为白炉标准形（白金属能量严格回 1）。
        //
        // 为什么不能逐通道用 E(f0_ch) 线性回填：低 f0 通道的 (1−E) 更大，
        // 回填量 ∝ f0·(1−f0) 在 f0→0 侧被放大 —— 高饱和金属的低通道被
        // 过量补能，整球在掠射角偏成补色（金球发青）。此处按 Fdez-Agüera
        // 2019 的彩色导体形式，补偿色相跟随 F0。
        let e_o = energy_dir(cos_o, alpha, 1.0);
        let e_i = energy_dir(cos_i, alpha, 1.0);
        let e_avg = energy_avg(alpha, 1.0);
        let ms_base = (1.0 - e_o) * (1.0 - e_i) / (PI * (1.0 - e_avg).max(1e-3));
        let f_avg = f0 + (Vec3::splat(1.0) - f0) * (1.0 / 21.0);
        let comp = |fa: f32| ms_base * fa * fa * e_avg / (1.0 - fa * (1.0 - e_avg)).max(1e-3);
        let ms = Vec3::new(comp(f_avg.x), comp(f_avg.y), comp(f_avg.z));
        f + ms
    }

    /// 反射域立体角 pdf（BSDF 采样策略 vs NEE/MNEE 的 MIS 备择策略）。
    /// 瓣混合 pdf：Σ p_lobe·pdf_lobe（与 [`Self::sample`] 的瓣选择一致）。
    pub fn pdf(&self, wo: Vec3, wi: Vec3, n: Vec3) -> f32 {
        let cos_o = wo.dot(&n).max(0.0);
        let cos_i = wi.dot(&n).max(0.0);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let (p_spec, p_diff, _) = self.lobe_probs(cos_o, false);
        let mut pdf = 0.0;
        if p_diff > 0.0 {
            pdf += p_diff * cos_i * FRAC_1_PI;
        }
        if p_spec > 0.0 {
            let h = (wo + wi).normalize();
            let cos_h = h.dot(&n);
            if cos_h > 0.0 {
                let alpha = self.alpha();
                let v_h = wo.dot(&h).max(0.0);
                let pdf_h =
                    smith_g1(alpha, cos_o) * ggx_d(alpha, cos_h) * v_h / cos_o.max(1e-6);
                let denom = 4.0 * v_h.max(1e-6);
                pdf += p_spec * pdf_h / denom;
            }
        }
        pdf
    }
}

/// BSDF 采样结果。
#[derive(Debug, Clone, Copy)]
pub struct BsdfSample {
    /// 世界空间出射方向（单位化；折射分支指向表面内侧）。
    pub dir: Vec3,
    /// 立体角 pdf（delta 瓣按积分器约定记 0）。
    pub pdf: f32,
    /// 吞吐乘子（= f·|cosθ|/(pdf·p_lobe)；delta 瓣 = 贡献/p_lobe）。
    pub weight: Vec3,
    /// 镜面瓣（true → 积分器命中发光权重 1、不参与 MIS 配对）。
    pub specular: bool,
    /// 折射进入介质内部（SSS 随机游走 / 玻璃内部行进触发位）。
    pub entered: bool,
}

/// 采样上下文（PSR 路径空间正则化的 roughness 下限；0 = 不正则化）。
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleCtx {
    /// PSR 注入的最小 roughness（镜面 roughness 注入种子派生微抖动）。
    pub psr_rough: f32,
}

impl Bsdf {
    /// BSDF 采样：u_lobe 选瓣（样本流哈希 [D]），u1/u2 定方向。
    /// `from_inside` = 视点在介质内（eta = ior；diffuse 瓣关闭）。
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        wo: Vec3,
        n: Vec3,
        from_inside: bool,
        ctx: &SampleCtx,
        u_lobe: f32,
        u1: f32,
        u2: f32,
    ) -> Option<BsdfSample> {
        let cos_v = wo.dot(&n).max(0.0);
        let (p_spec, p_diff, p_trans) = self.lobe_probs(cos_v, from_inside);
        if p_spec + p_diff + p_trans <= 0.0 {
            return None; // 纯黑材质：吸收
        }

        // ---- transmission 瓣（u_lobe ∈ [0, p_trans)）----
        if u_lobe < p_trans {
            let eta = if from_inside { self.ior } else { 1.0 / self.ior };
            let w_in = -wo; // 指向表面
            let refr = refract_dir(w_in, n, eta);
            let p_reflect = refr.map_or(1.0, |_| luminance(schlick(self.f0(), cos_v)).clamp(1e-3, 1.0 - 1e-3));
            if refr.is_none() || u1 < p_reflect {
                // 镜面反射（δ；PSR 打开时方向加切平面微抖动挤过链）。
                let dir = reflect_dir(w_in, n).normalize();
                let dir = if ctx.psr_rough > 0.0 {
                    jitter_dir(dir, n, ctx.psr_rough, u2)
                } else {
                    dir
                };
                let weight = if refr.is_none() {
                    schlick(self.f0(), cos_v) // TIR：全反射（概率 1）
                } else {
                    schlick(self.f0(), cos_v) * (1.0 / p_reflect)
                };
                return Some(BsdfSample {
                    dir,
                    pdf: 0.0,
                    weight,
                    specular: ctx.psr_rough == 0.0,
                    entered: false,
                });
            }
            // 折射进入介质（SSS 或玻璃内部行进）。
            let dir = refr.unwrap().normalize();
            let dir = if ctx.psr_rough > 0.0 {
                jitter_dir(dir, n, ctx.psr_rough, u2)
            } else {
                dir
            };
            let weight = hadamard(self.base_color, Vec3::splat(1.0) - schlick(self.f0(), cos_v))
                * (1.0 / (1.0 - p_reflect));
            return Some(BsdfSample {
                dir,
                pdf: 0.0,
                weight,
                specular: ctx.psr_rough == 0.0,
                entered: true,
            });
        }

        // ---- diffuse 瓣（u_lobe ∈ [p_trans, p_trans+p_diff)）----
        if u_lobe < p_trans + p_diff {
            let d_local = cosine_hemisphere(u1, u2);
            let (t, b) = onb(n);
            let dir_w = (t * d_local.x + b * d_local.y + n * d_local.z).normalize();
            // 混合采样无偏性：weight 除以混合 pdf（Σ p_i·pdf_i，pbrt 同款）。
            let pdf_mix = self.pdf(wo, dir_w, n);
            if pdf_mix <= 0.0 {
                return None;
            }
            let f = self.eval(wo, dir_w, n);
            let cos_i = dir_w.dot(&n).max(0.0);
            return Some(BsdfSample {
                dir: dir_w,
                pdf: pdf_mix,
                weight: f * (cos_i / pdf_mix),
                specular: false,
                entered: false,
            });
        }

        // ---- specular 瓣（GGX VNDF；roughness < 1e-3 退化 δ 镜像）----
        let rough_used = self.roughness.max(ctx.psr_rough);
        let alpha_used = (rough_used * rough_used).max(1e-6);
        // VNDF 采样器要求**局部系**视线（+z = n）。必须先把世界 wo 投影到
        // onb 基，再把输出的局部 h 变换回世界——否则 D_vis(ωo) 的视线相关
        // 分布会被 onb 的基旋转破坏（曾致 E[weight] 系统性偏低 ~4%）。
        let (t, b) = onb(n);
        let wo_local = Vec3::new(wo.dot(&t), wo.dot(&b), cos_v);
        let h_local = sample_vndf(wo_local, alpha_used, u1, u2);
        let h = t * h_local.x + b * h_local.y + n * h_local.z;
        let dir = reflect_dir(-wo, h).normalize();
        let cos_l = dir.dot(&n);
        if cos_l <= 0.0 {
            return None; // 采样翻到表面下方：样本作废（罕见，无偏性不受影响）
        }
        if rough_used < 1e-3 {
            // δ 镜像：weight = F_v / p_spec（δ 瓣无 pdf，瓣选择概率归一）。
            return Some(BsdfSample {
                dir,
                pdf: 0.0,
                weight: schlick(self.f0(), cos_v) * (1.0 / p_spec.max(1e-4)),
                specular: true,
                entered: false,
            });
        }
        let pdf_mix = self.pdf(wo, dir, n);
        if pdf_mix <= 0.0 {
            return None;
        }
        let f = self.eval(wo, dir, n);
        Some(BsdfSample {
            dir,
            pdf: pdf_mix,
            weight: f * (cos_l / pdf_mix),
            specular: false,
            entered: false,
        })
    }
}

/// 局部（z 轴 = n）向量 → 世界向量（复用 [`crate::sampler::onb`]，格锚纪律禁浮点三角）。
pub fn local_to_world(v_local: Vec3, n: Vec3) -> Vec3 {
    let (t, b) = onb(n);
    t * v_local.x + b * v_local.y + n * v_local.z
}

/// PSR 方向微抖动：切平面注入 ε 量级偏移（PSR；u 为样本流哈希
/// 坐标，确定性纯函数）。
fn jitter_dir(dir: Vec3, n: Vec3, eps: f32, u: f32) -> Vec3 {
    let (t, b) = onb(n);
    let (c, s) = cos_sin(2.0 * PI * u);
    (dir + (t * c + b * s) * eps).normalize()
}

// ---------------------------------------------------------------------------
// 测试（能量守恒 / 分布锚点 / 采样一致性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::sample_1d;
    use prism_core::SampleIndex;

    #[test]
    fn schlick_and_f0_anchors() {
        // Schlick 端点：cos=1 → F0；cos→0 → 1。
        let f0 = Vec3::splat(0.04);
        let f1 = schlick(f0, 1.0);
        assert_eq!(f1.x.to_bits(), 0.04f32.to_bits());
        let f2 = schlick(f0, 1e-4);
        assert!(f2.x > 0.99, "grazing Fresnel 应接近 1：{}", f2.x);
        // ior → F0 锚点：ior 1.5 → 0.04（公开物理常量 ((1.5−1)/(1.5+1))²）。
        let b = Bsdf::new(Vec3::splat(0.5), 0.0, 0.5, 1.5, 0.0, 0.0, Vec3::splat(0.3));
        let got = b.f0().x;
        assert!((got - 0.04).abs() < 1e-6, "ior 1.5 的 F0 应为 0.04，实得 {got}");
        // 金属 F0 = base_color。
        let m = Bsdf::new(Vec3::new(0.9, 0.7, 0.4), 1.0, 0.2, 1.5, 0.0, 0.0, Vec3::splat(0.3));
        let f0m = m.f0();
        assert_eq!(f0m.x.to_bits(), 0.9f32.to_bits());
        assert_eq!(f0m.z.to_bits(), 0.4f32.to_bits());
    }

    #[test]
    fn ggx_d_peak_and_hemisphere_normalization() {
        // 峰值锚点：h = n 时 D = 1/(πα²)。
        for &alpha in &[0.1f32, 0.5] {
            let d = ggx_d(alpha, 1.0);
            let want = 1.0 / (PI * alpha * alpha);
            assert!((d - want).abs() / want < 1e-5, "GGX 峰值 {d} vs {want}");
        }
        // 半球归一 ∫D·cosθ dΩ = 1（余弦采样 MC：估计量 = D·π）。
        let ctx = EvalCtx::new(0xD_00D, 0, 0);
        let n = 100_000u32;
        for &alpha in &[0.2f32, 0.6] {
            let mut sum = 0.0f64;
            for k in 0..n {
                let si = SampleIndex { frame: 0, pixel: (0, 0), sample: k, dimension: 5, node_id: 0 };
                let (u1, u2) = sample_2d(&ctx, &si);
                let d = cosine_hemisphere(u1, u2);
                sum += (ggx_d(alpha, d.z) * PI) as f64;
            }
            let norm = (sum / n as f64) as f32;
            assert!((norm - 1.0).abs() < 0.03, "α={alpha}: ∫D·cos dΩ = {norm}，期望 1");
        }
    }

    #[test]
    fn energy_conservation_by_sampling() {
        // 单方向能量守恒（sample-based）：固定 ωo，E[weight] = ∫f(ωo,ωi)·cosθi dωi
        // = E(ωo)（方向反照率）。混合瓣采样的 weight 除以混合 pdf，联合分布下无偏。
        // 补偿结构（彩色 F0 形式）→ 白金属（f0=1）每方向精确 = 1（最强守恒断言，
        // 对任意 α 成立，因为 F_avg=1 时补偿项逐字退化为白炉标准形）；
        // 彩色电介质每通道 ≤ 1（diffuse 让位 + 补偿色相跟随 F_avg 保证上界）。
        // 旧双向白炉（余弦采样 ωi）在低 α 严重欠采样，故弃用。
        let ctx = EvalCtx::new(0xF0_1D, 0, 0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        // 400k/方向：VNDF 采样的 weight 方差来自 G1(l) 因子未入 pdf，
        // 掠射方向（μ→0.25）收敛慢，40k 时 μ=0.6 噪声 ±6% 会误报。
        let ns = 400_000u32;

        // 无效样本（VNDF 偶发翻面 → None）贡献 0、分母仍为 ns：等价于
        // weight=0，估计无偏（cos_i ≤ 0 区域 f ≡ 0）。
        let mean_weight = |b: &Bsdf, wo: Vec3, dim: u16| -> Vec3 {
            let (mut r, mut g, mut bl) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..ns {
                let s1 = SampleIndex { frame: 0, pixel: (0, 0), sample: k, dimension: dim, node_id: 0 };
                let s2 = SampleIndex { frame: 0, pixel: (0, 0), sample: k, dimension: dim + 1, node_id: 0 };
                let u_lobe = sample_1d(&ctx, &s1);
                let (u1, u2) = sample_2d(&ctx, &s2);
                if let Some(s) = b.sample(wo, n, false, &SampleCtx::default(), u_lobe, u1, u2) {
                    if s.pdf > 0.0 {
                        r += s.weight.x as f64;
                        g += s.weight.y as f64;
                        bl += s.weight.z as f64;
                    }
                }
            }
            let inv = 1.0 / ns as f32;
            Vec3::new((r * inv as f64) as f32, (g * inv as f64) as f32, (bl * inv as f64) as f32)
        };

        for (i, &mu) in [1.0f32, 0.6, 0.25].iter().enumerate() {
            let b = Bsdf::new(Vec3::splat(1.0), 1.0, 0.4, 1.5, 0.0, 0.0, Vec3::splat(0.3));
            let wo = Vec3::new((1.0 - mu * mu).sqrt(), 0.0, mu);
            let e = mean_weight(&b, wo, 30 + i as u16 * 4);
            println!("白金属 μ={mu}: E=({:.4},{:.4},{:.4})", e.x, e.y, e.z);
            for ch in [e.x, e.y, e.z] {
                assert!((ch - 1.0).abs() < 0.03, "白金属 μ={mu} 方向反照率 {ch} 偏离 1");
            }
        }
        // 彩色电介质：每通道 ≤ 1.02（能量上界守恒）。
        let d = Bsdf::new(Vec3::new(0.8, 0.3, 0.1), 0.0, 0.5, 1.5, 0.0, 0.0, Vec3::splat(0.3));
        let e = mean_weight(&d, Vec3::new(0.0, 0.0, 1.0), 44);
        println!("电介质 μ=1: E=({:.4},{:.4},{:.4})", e.x, e.y, e.z);
        for ch in [e.x, e.y, e.z] {
            assert!(ch <= 1.02, "电介质通道方向反照率越界：{ch}");
        }
    }

    #[test]
    fn vndf_sample_matches_energy_table() {
        // 单散射一致性：VNDF 采样路径下 E[spec_scatter·cos/pdf] = E_dir 表值。
        // （sample() 的完整 weight 含 ms 补偿项 → 方向反照率恒等式 E_o+(1−E_o)·f0，
        // 守恒性由 energy_conservation_by_sampling 单独覆盖。）
        let b = Bsdf::new(Vec3::splat(1.0), 1.0, 0.35, 1.5, 0.0, 0.0, Vec3::splat(0.3));
        let ctx = EvalCtx::new(0x9E_7D, 0, 0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let ns = 30_000u32;
        let alpha = b.alpha();
        for &mu in &[1.0f32, 0.5] {
            let wo = Vec3::new((1.0 - mu * mu).sqrt(), 0.0, mu);
            let mut acc = 0.0f64;
            for k in 0..ns {
                let si = SampleIndex { frame: 0, pixel: (0, 0), sample: k, dimension: 7, node_id: 0 };
                let (u1, u2) = sample_2d(&ctx, &si);
                // metal=1 → p_spec=1，u_lobe=0.5 必落 spec 瓣。
                if let Some(s) = b.sample(wo, n, false, &SampleCtx::default(), 0.5, u1, u2) {
                    if s.pdf > 0.0 {
                        let f_spec = spec_scatter(wo, s.dir, n, alpha, Vec3::splat(1.0)).x;
                        acc += (f_spec * (s.dir.dot(&n) / s.pdf)) as f64;
                    }
                }
            }
            let mc = (acc / ns as f64) as f32;
            let table = energy_dir(mu, b.alpha(), 1.0);
            let rel = (mc - table).abs() / table;
            println!("μ={mu}: E[spec]={mc:.4} 表值={table:.4} rel={rel:.4}");
            assert!(rel < 0.05, "VNDF 采样与能量表不一致：μ={mu} mc={mc} table={table}");
        }
    }

    #[test]
    fn bsdf_sample_is_deterministic() {
        // 同参数两遍逐位一致（纯函数 [D]）。
        let b = Bsdf::new(Vec3::splat(0.7), 0.2, 0.4, 1.45, 0.3, 0.1, Vec3::splat(0.4));
        let n = Vec3::new(0.1, 0.9, 0.2).normalize();
        let wo = Vec3::new(0.2, 0.7, 0.5).normalize();
        let s1 = b.sample(wo, n, false, &SampleCtx::default(), 0.42, 0.13, 0.77);
        let s2 = b.sample(wo, n, false, &SampleCtx::default(), 0.42, 0.13, 0.77);
        match (s1, s2) {
            (Some(a), Some(c)) => {
                assert_eq!(a.dir.x.to_bits(), c.dir.x.to_bits());
                assert_eq!(a.dir.z.to_bits(), c.dir.z.to_bits());
                assert_eq!(a.weight.y.to_bits(), c.weight.y.to_bits());
                assert_eq!(a.pdf.to_bits(), c.pdf.to_bits());
            }
            _ => panic!("有效参数下采样不应失败"),
        }
    }
}
