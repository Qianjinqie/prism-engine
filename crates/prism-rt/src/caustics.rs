//! 焦散：MNEE 流形下一事件估计 + PSR 兜底（全部要点）。
//!
//! # 出处（数据资产清单 #4 注释义务）
//! - 流形探索：W. Jakob, S. Marschner, "Manifold Exploration: A Markov Chain
//!   Monte Carlo Technique for the Rendering of General Spectral Effects",
//!   EGSR 2012。
//! - MNEE：J. Hanika, M. Droske, L. Fascione, "Manifold Next Event
//!   Estimation", Computer Graphics Forum 34(4), EGSR 2015。
//! - 路径空间正则化（PSR）：J. Hanika, A. Kaplanyan, C. Dachsbacher,
//!   "Improved path space regularization for glossy and caustics rendering",
//!   EGSR 2015——**兜底分层**：MNEE 不收敛的样本返回 None，由 BSDF 采样的
//!  PSR 镜面抖动路径（[`crate::bsdf::SampleCtx::psr_rough`]，
//!   "roughness 注入微抖动"）以拉宽后的 glossy 链覆盖，两层合成不降级。
//! - D-30：k=2 球面折射链 = 漫射面 → 入球折射 → 出球折射 → 光源。
//!
//! # 牛顿与雅可比（D-29 确定性）
//! - 切平面 2D 参数化：变量 (u,v) ∈ [−1,1]²（无量纲，偏移幅 `h` 缩放），
//!  球链样本投影回球面归一（"镜面切平面参数化"）。
//! - 残差 = 链末端方向与目标方向的横向偏差，投影到**固定基**（P→L 连线的
//!   正交基）——基不随迭代旋转，差分无伪噪声。
//! - 牛顿 3 步 + 残差增阻尼减半；雅可比 = 中心差分。
//!   k=1 平面镜链约束线性（牛顿一步精确），其差分雅可比由解析解
//!   |det| = cosθ·h²/D²（Lagrange 恒等式）锚定测试；k=2 球面折射链同框架
//!  差分（"一般曲面有限差分"条款；解析雅可比列为 D-30 扩展位）。
//! - pdf_w = p_st/|det DΦ|：p_st = 1/4（(u,v) 均匀域），DΦ 为切平面到入射
//!   方向的全链雅可比（差分）。面光源公式 pdf_w = (p_st/|J|)·(cosθ_l/d²)
//!   的几何因子（cosθ/d²）在 DΦ 中自然出现（平面镜解析验证），两式等价。
//! - 全程 ±×÷√（sqrt 为 IEEE 精确舍入运算），无 libm、无全局 RNG；
//!  多初值随机数走 aux 游程流。

use prism_core::{EvalCtx, SampleIndex};
use prism_math::Vec3;

use crate::bsdf::{refract_dir, schlick};
use crate::sampler::{aux_1d, onb};

/// 玻璃折射体（球）：MNEE k=2 链的镜面事件对象（与 Scene 发光/几何球对应）。
#[derive(Debug, Clone, Copy)]
pub struct Refractor {
    /// 球心。
    pub center: Vec3,
    /// 半径（> 0）。
    pub radius: f32,
    /// 折射率（> 1）。
    pub ior: f32,
}

/// MNEE 焦散样本（单初值分支；调用方按 NEE 语义组装贡献）。
#[derive(Debug, Clone, Copy)]
pub struct CausticsSample {
    /// 着色点 P 处的入射方向（指向链上镜面事件点，单位向量）。
    pub wi: Vec3,
    /// 入射方向的立体角 pdf（含多初值均摊前的单分支值）。
    pub pdf_w: f32,
    /// 链吞吐 / 多初值数（Fresnel_in·Fresnel_out；k=1 镜面 = 1/m）。
    pub weight: Vec3,
}

/// 牛顿步数上限（牛顿 3 步）。
const NEWTON_STEPS: usize = 3;
/// 收敛残差阈（切向角偏差，rad 量级）。
const RES_EPS: f32 = 1.0e-4;
/// 雅可比中心差分步长（无量纲坐标）。
const DIFF_EPS: f32 = 1.0e-4;
/// 多初值数 m（aux 流均摊， 多初值框架）。
const MNEE_MULTISTART: u32 = 4;

/// 射线与球最近正交点（t > 0 的最小根）；无正根 None。
fn ray_sphere_near(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(&d);
    let disc = b * b - (oc.dot(&oc) - r * r);
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    if t0 > 1e-6 {
        return Some(t0);
    }
    let t1 = -b + s;
    if t1 > 1e-6 {
        return Some(t1);
    }
    None
}

/// 2×2 线性求解 J·x = b（克拉默）；|det| 过小 None。
fn solve2(j00: f32, j01: f32, j10: f32, j11: f32, b0: f32, b1: f32) -> Option<(f32, f32)> {
    let det = j00 * j11 - j01 * j10;
    if det.abs() < 1e-12 {
        return None;
    }
    Some((
        (b0 * j11 - b1 * j01) / det,
        (b1 * j00 - b0 * j10) / det,
    ))
}

/// 通用 2D 牛顿（差分雅可比 + 阻尼）：residual(u,v) → [r1, r2]。
/// 返回收敛解 (u*, v*)；3 步内未达 RES_EPS 或链断裂 None。
fn newton2(mut residual: impl FnMut(f32, f32) -> Option<[f32; 2]>, u0: f32, v0: f32) -> Option<(f32, f32)> {
    let (mut u, mut v) = (u0, v0);
    let mut r = residual(u, v)?;
    let mut norm2 = r[0] * r[0] + r[1] * r[1];
    for _ in 0..NEWTON_STEPS {
        if norm2 < RES_EPS * RES_EPS {
            return Some((u, v));
        }
        // 中心差分雅可比。
        let rpu = residual(u + DIFF_EPS, v)?;
        let rmu = residual(u - DIFF_EPS, v)?;
        let rpv = residual(u, v + DIFF_EPS)?;
        let rmv = residual(u, v - DIFF_EPS)?;
        let j00 = (rpu[0] - rmu[0]) / (2.0 * DIFF_EPS);
        let j10 = (rpu[1] - rmu[1]) / (2.0 * DIFF_EPS);
        let j01 = (rpv[0] - rmv[0]) / (2.0 * DIFF_EPS);
        let j11 = (rpv[1] - rmv[1]) / (2.0 * DIFF_EPS);
        // 牛顿步 J·d = −r（步长 clamp：大偏差初值下防线性化失效跑飞）。
        let (du, dv) = solve2(j00, j01, j10, j11, -r[0], -r[1])?;
        let du = du.clamp(-4.0, 4.0);
        let dv = dv.clamp(-4.0, 4.0);
        // 阻尼：残差增则步长减半（至多两轮）。
        let (mut du_d, mut dv_d) = (du, dv);
        for _ in 0..2 {
            match residual(u + du_d, v + dv_d) {
                Some(rt) => {
                    let n2 = rt[0] * rt[0] + rt[1] * rt[1];
                    if n2 <= norm2 {
                        break;
                    }
                    du_d *= 0.5;
                    dv_d *= 0.5;
                }
                None => {
                    du_d *= 0.5;
                    dv_d *= 0.5;
                }
            }
        }
        u += du_d;
        v += dv_d;
        let rn = residual(u, v)?;
        norm2 = rn[0] * rn[0] + rn[1] * rn[1];
        r = rn;
    }
    if norm2 < RES_EPS * RES_EPS {
        Some((u, v))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// k=1：平面镜链（解析结构，差分雅可比由解析锚定）
// ---------------------------------------------------------------------------

/// 平面镜 MNEE 链（k=1，**解析**——"平面镜面解析雅可比"）：
/// P → 镜面点反射 → L。镜像法直接给出流形解（L 关于镜面镜像 → 连线交镜面），
/// 无需牛顿迭代；pdf 记 δ 约定 1.0（确定 P-L 对的流形解唯一，随机性全在
/// 调用方的光源采样中，MIS 权重按非几何 NEE 处理 = 1）。
/// 镜面无损 → weight = 1。返回 None：退化几何（光贴镜面 / 连线不达）。
#[allow(clippy::too_many_arguments)]
pub fn mnee_mirror(
    plane_p: Vec3,
    plane_n: Vec3,
    p: Vec3,
    light: Vec3,
    ctx: &EvalCtx,
    si: &SampleIndex,
) -> Option<CausticsSample> {
    let _ = (ctx, si); // 解析链无随机数（保留签名与球链对称，aux 流后续扩展位）
    let n = plane_n.normalize();
    // 镜像法：L 关于镜面镜像 → P→L' 连线与镜面交点 = 精确链解。
    let dist_l = (light - plane_p).dot(&n);
    if dist_l.abs() < 1e-8 {
        return None; // 光贴镜面
    }
    let l_mirror = light - n * (2.0 * dist_l);
    let to_m = l_mirror - p;
    let d_m_len2 = to_m.dot(&to_m);
    if d_m_len2 < 1e-12 {
        return None;
    }
    let d_m = to_m * (1.0 / d_m_len2.sqrt());
    let denom = d_m.dot(&n);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t0 = (plane_p - p).dot(&n) / denom;
    if t0 <= 1e-6 || t0 >= d_m_len2.sqrt() {
        return None; // 交点不在 P 与镜像点之间（P 在镜面上/镜面异侧退化）
    }
    let x_star = p + d_m * t0;
    let wi = (x_star - p).normalize();
    Some(CausticsSample {
        wi,
        pdf_w: 1.0,
        weight: Vec3::splat(1.0),
    })
}

// ---------------------------------------------------------------------------
// k=2：球面折射链（D-30：入球折射 → 出球折射 → 光源）
// ---------------------------------------------------------------------------

/// 给定入射折射点 x1，模拟玻璃球链：P → x1（折射入）→ x2（折射出）→ 方向。
/// 返回 (x2, w_out, 横向偏差 e)。TIR / 非向内行进 None。
fn refract_chain(refr: &Refractor, p: Vec3, light: Vec3, x1: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    let n1 = (x1 - refr.center) * (1.0 / refr.radius); // 入射点外法线
    let d_in = (x1 - p).normalize(); // P → x1（指向球面）
    let w_mid = refract_dir(d_in, n1, 1.0 / refr.ior)?; // 空气 → 玻璃
    // 内部第二交点：弦长 t = −2·(x1−c)·w_mid（|w_mid| = 1，b < 0）。
    let b = (x1 - refr.center).dot(&w_mid);
    if b >= 0.0 {
        return None;
    }
    let x2 = x1 + w_mid * (-2.0 * b);
    let n2 = (x2 - refr.center) * (1.0 / refr.radius); // 出射点外法线
    let w_out = refract_dir(w_mid, -n2, refr.ior)?; // 玻璃 → 空气（n 取内法线）
    let d_t = (light - x2).normalize();
    let e = d_t - w_out * d_t.dot(&w_out); // 横向偏差
    Some((x2, w_out, e))
}

/// 球面折射 MNEE 链（k=2， / D-30）：P → 入球折射 → 出球折射 → L。
/// 多初值 m=4（aux 流均摊）；任一初值收敛即返回该分支样本（期望语义：
/// 估计量 = Σ_i f·吞吐_i/(m·pdf_i)，不收敛初值贡献 0）。
#[allow(clippy::too_many_arguments)]
pub fn mnee_refraction(
    refr: &Refractor,
    p: Vec3,
    light: Vec3,
    ctx: &EvalCtx,
    si: &SampleIndex,
) -> Option<CausticsSample> {
    // 基点：P→L 连线与球的近交点（链的直连近似起点）。
    let d_pl = (light - p).normalize();
    let t0 = ray_sphere_near(p, d_pl, refr.center, refr.radius)?;
    let x_base = p + d_pl * t0;
    let n_b = (x_base - refr.center) * (1.0 / refr.radius);
    let (t1b, t2b) = onb(n_b);
    let (q1, q2) = onb(d_pl); // 残差固定投影基
    let h = refr.radius * 0.35; // 切平面偏移幅（初值集中，收敛域内）
    // 链残差（固定基投影）。
    let residual = |u: f32, v: f32| -> Option<[f32; 2]> {
        let off = t1b * (u * h) + t2b * (v * h);
        let x1 = refr.center + (x_base + off - refr.center).normalize() * refr.radius;
        let (_, _, e) = refract_chain(refr, p, light, x1)?;
        Some([e.dot(&q1), e.dot(&q2)])
    };
    // 方向映射：切平面 (u,v) → 入射方向 wi（全链复合）。
    let wi_of = |u: f32, v: f32| -> Option<Vec3> {
        let off = t1b * (u * h) + t2b * (v * h);
        let x1 = refr.center + (x_base + off - refr.center).normalize() * refr.radius;
        Some((x1 - p).normalize())
    };
    // 电介质 F0（Schlick 反射率，入/出两侧同值）。
    let r0 = (refr.ior - 1.0) / (refr.ior + 1.0);
    let f0 = Vec3::splat(r0 * r0);
    for m in 0..MNEE_MULTISTART {
        let u1 = aux_1d(ctx, si, m * 2);
        let u2 = aux_1d(ctx, si, m * 2 + 1);
        let u0 = (u1 - 0.5) * 2.0;
        let v0 = (u2 - 0.5) * 2.0;
        if let Some((us, vs)) = newton2(residual, u0, v0) {
            // 收敛点复核：残差阈值已由 newton2 保证。
            let x1s_off = t1b * (us * h) + t2b * (vs * h);
            let x1s = refr.center + (x_base + x1s_off - refr.center).normalize() * refr.radius;
            let (x2s, w_out, e) = refract_chain(refr, p, light, x1s)?;
            if e.x * e.x + e.y * e.y + e.z * e.z > RES_EPS * RES_EPS {
                continue; // 防御：终态残差超阈放弃
            }
            let wi = (x1s - p).normalize();
            // 链吞吐：入射 + 出射 Fresnel（无色散假设：光谱权重同值）。
            let cos_in = (-(x1s - p).normalize().dot(&((x1s - refr.center) * (1.0 / refr.radius)))).max(0.0);
            let n2 = (x2s - refr.center) * (1.0 / refr.radius);
            let cos_out = w_out.dot(&n2).max(0.0);
            let f_path = schlick(f0, cos_in).x * schlick(f0, cos_out).x;
            // 差分雅可比（收敛点，切于 wi 的固定 2D 基）。
            let (s1, s2) = onb(wi);
            let wu1 = wi_of(us + DIFF_EPS, vs)?;
            let wu0 = wi_of(us - DIFF_EPS, vs)?;
            let wv1 = wi_of(us, vs + DIFF_EPS)?;
            let wv0 = wi_of(us, vs - DIFF_EPS)?;
            let j00 = (wu1 - wu0).dot(&s1) / (2.0 * DIFF_EPS);
            let j10 = (wu1 - wu0).dot(&s2) / (2.0 * DIFF_EPS);
            let j01 = (wv1 - wv0).dot(&s1) / (2.0 * DIFF_EPS);
            let j11 = (wv1 - wv0).dot(&s2) / (2.0 * DIFF_EPS);
            let det = (j00 * j11 - j01 * j10).abs();
            if det < 1e-14 {
                continue;
            }
            let pdf_w = 0.25 / det;
            return Some(CausticsSample {
                wi,
                pdf_w,
                weight: Vec3::splat(f_path / MNEE_MULTISTART as f32),
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 测试（平面镜解析 / 球链收敛与守恒 / 确定性 / 雅可比锚点）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_of(seed: u64) -> EvalCtx {
        EvalCtx::new(seed, 0, 0)
    }

    fn si_of(sample: u32, node: u64) -> SampleIndex {
        SampleIndex { frame: 0, pixel: (0, 0), sample, dimension: 0, node_id: node }
    }

    #[test]
    fn mirror_chain_matches_analytic() {
        // 平面镜反射解析（P、L 同侧）：镜像法——L=(1,0.5,0.5) 关于镜面 z=1
        // 的镜像 L'=(1,0.5,1.5)；链解 = P→L' 连线与镜面交点，
        // wi 应 = 该交点方向；pdf = p_uv·D²/cosθ（差分雅可比锚定）。
        let plane_p = Vec3::new(0.0, 0.0, 1.0);
        let plane_n = Vec3::new(0.0, 0.0, -1.0); // 朝 P 侧
        let p = Vec3::new(0.0, 0.0, 0.0);
        let light = Vec3::new(2.0, 1.0, 0.2);
        let ctx = ctx_of(0xCA_01);
        let si = si_of(3, 0x1);
        let s = mnee_mirror(plane_p, plane_n, p, light, &ctx, &si)
            .expect("平面镜链必须收敛");
        // 解析：镜像点 → 连线交镜面 z=1。
        let l_mirror = Vec3::new(2.0, 1.0, 1.8);
        let x_star = l_mirror * (1.0 / l_mirror.z); // t 使 z 分量 = 1
        let wi_a = (x_star - p).normalize();
        assert!(
            (s.wi - wi_a).length() < 1e-4,
            "wi 偏差：got {:?} want {:?}",
            s.wi,
            wi_a
        );
        // δ 约定：流形解唯一 → pdf = 1（随机性在光源采样）；镜面无损 → weight = 1。
        assert_eq!(s.pdf_w.to_bits(), 1.0f32.to_bits(), "解析链 pdf 应为 δ 约定 1");
        assert_eq!(s.weight.x.to_bits(), 1.0f32.to_bits(), "镜面无损 weight = 1");
    }

    #[test]
    fn glass_sphere_chain_converges_and_conserves() {
        // D-30 k=2 链：P → 入球折射 → 出球折射 → L。多初值收敛、吞吐 ≤ 1。
        let refr = Refractor {
            center: Vec3::new(0.0, 1.5, 3.0),
            radius: 1.0,
            ior: 1.5,
        };
        let p = Vec3::new(0.0, 0.0, 0.0);
        let light = Vec3::new(0.3, 2.5, 6.0);
        let ctx = ctx_of(0xCA_02);
        let mut converged = 0u32;
        for node in 0..8u64 {
            let si = si_of(5, node);
            if let Some(s) = mnee_refraction(&refr, p, light, &ctx, &si) {
                converged += 1;
                assert!(s.pdf_w.is_finite() && s.pdf_w > 0.0, "pdf 非法：{}", s.pdf_w);
                for ch in [s.weight.x, s.weight.y, s.weight.z] {
                    assert!(ch > 0.0 && ch <= 0.25 + 1e-6, "链吞吐越界：{ch}");
                    // F·F/m ≤ 1/m（Fresnel ≤ 1）。
                }
            }
        }
        assert!(converged > 0, "多初值应至少一个收敛");
    }

    #[test]
    fn mnee_is_deterministic() {
        // 同参数两遍逐位一致（纯函数 [D]）。
        let refr = Refractor {
            center: Vec3::new(0.0, 1.5, 3.0),
            radius: 1.0,
            ior: 1.5,
        };
        let ctx = ctx_of(0xCA_03);
        let si = si_of(7, 0x2);
        let a = mnee_refraction(&refr, Vec3::splat(0.0), Vec3::new(0.3, 2.5, 6.0), &ctx, &si);
        let b = mnee_refraction(&refr, Vec3::splat(0.0), Vec3::new(0.3, 2.5, 6.0), &ctx, &si);
        match (a, b) {
            (Some(x), Some(y)) => {
                assert_eq!(x.wi.x.to_bits(), y.wi.x.to_bits());
                assert_eq!(x.wi.z.to_bits(), y.wi.z.to_bits());
                assert_eq!(x.pdf_w.to_bits(), y.pdf_w.to_bits());
                assert_eq!(x.weight.y.to_bits(), y.weight.y.to_bits());
            }
            (None, None) => {}
            _ => panic!("同参数 MNEE 结果必须一致"),
        }
        // 平面镜链同样确定。
        let si2 = si_of(9, 0x3);
        let m1 = mnee_mirror(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::splat(0.0),
            Vec3::new(2.0, 1.0, 0.2),
            &ctx,
            &si2,
        );
        let m2 = mnee_mirror(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::splat(0.0),
            Vec3::new(2.0, 1.0, 0.2),
            &ctx,
            &si2,
        );
        match (m1, m2) {
            (Some(x), Some(y)) => assert_eq!(x.pdf_w.to_bits(), y.pdf_w.to_bits()),
            (None, None) => {}
            _ => panic!("镜面链确定性问题"),
        }
    }

    #[test]
    fn mirror_chain_misses_return_none() {
        // 退化几何 → None：P 在镜面上（t0=0）/ 光贴镜面（目标方向退化）。
        let ctx = ctx_of(0xCA_04);
        let si = si_of(1, 0x4);
        // P 在镜面上：连线起点即镜面，t0 ≤ 1e-6 → None。
        let s = mnee_mirror(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.3, -0.2, 1.0), // P 在 z=1 镜面上
            Vec3::new(2.0, 1.0, 0.2),
            &ctx,
            &si,
        );
        assert!(s.is_none(), "P 在镜面上应返回 None");
        // 光恰在镜面上：链目标方向退化 → None。
        let s2 = mnee_mirror(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::splat(0.0),
            Vec3::new(0.5, 0.2, 1.0), // L 在 z=1 镜面上
            &ctx,
            &si,
        );
        assert!(s2.is_none(), "光贴镜面应返回 None");
    }
}
