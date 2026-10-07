//! SDF 域算子与 sphere tracing（/ B3； S1 SDF 场 + S12 域算子）。
//!
//! 域算子是自由函数（输入/输出均为距离值），[`crate::csg::Csg`] 树直接复用它们组合子场
//! ——CSG 与域算子只有这一份实现（D-16）。displace 接 [`prism_noise`] 的 simplex fbm。

use crate::primitive::Sdf;
use prism_math::Vec3;
use prism_noise::fbm::Fbm;

/// 并集：`min(a, b)`。
#[inline]
pub fn union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// 交集：`max(a, b)`。
#[inline]
pub fn intersection(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// 差集 `a - b`：`max(a, -b)`（a ∩ 补b）。
#[inline]
pub fn subtraction(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// 多项式平滑最小（k > 0 控制融合半径）。
///
/// `k → 0` 时退化为精确 `min`；两表面距离小于 `k` 时产生圆角融合，
/// 下沉量有界：`smin ≥ min(a, b) − k/4`。
#[inline]
pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    debug_assert!(k > 0.0, "smooth_min requires k > 0");
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

/// 平滑交集：`−smooth_min(−a, −b, k)`。
#[inline]
pub fn smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// 平滑差集 `a − b`：`smooth_intersection(a, −b, k)`。
#[inline]
pub fn smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    smooth_intersection(a, -b, k)
}

/// displace 用的 fbm 参数（octaves 4–6 为推荐档，取 4）。
const DISPLACE_FBM: Fbm = Fbm { octaves: 4, lacunarity: 2.0, gain: 0.5 };

/// 默认 displace octaves（与 LOD 满档一致，见 [`crate::lod`]）。
pub const DEFAULT_DISPLACE_OCTAVES: u32 = 4;

/// 域位移：基场距离叠加 simplex fbm 噪声（——分形岩石/地形的来源）。
///
/// `amp` 为位移幅度，`freq` 为噪声空间频率，`seed` 接引擎全局种子派生（确定性）。
#[inline]
pub fn displace(base: f32, p: Vec3, amp: f32, freq: f32, seed: u64) -> f32 {
    displace_lod(base, p, amp, freq, seed, DEFAULT_DISPLACE_OCTAVES)
}

/// displace 的 LOD 版：octaves 由 [`crate::lod::lod_octaves`] 按距离衰减。
#[inline]
pub fn displace_lod(base: f32, p: Vec3, amp: f32, freq: f32, seed: u64, octaves: u32) -> f32 {
    let n = Fbm { octaves, lacunarity: DISPLACE_FBM.lacunarity, gain: DISPLACE_FBM.gain }
        .simplex3(seed, p.x * freq, p.y * freq, p.z * freq);
    base + amp * n
}

/// sphere tracing 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceParams {
    /// 射线参数上限（超过即未命中）。
    pub t_max: f32,
    /// 命中阈值：`dist ≤ hit_eps` 即视为触及表面。
    pub hit_eps: f32,
    /// 最大步进次数（步数固定 → 确定性）。
    pub max_steps: u32,
    /// 步长安全系数（SDF Lipschitz = 1，取 0.9 保证单步不穿越表面）。
    pub safety: f32,
}

impl Default for TraceParams {
    fn default() -> Self {
        TraceParams { t_max: 100.0, hit_eps: 1e-4, max_steps: 256, safety: 0.9 }
    }
}

/// sphere tracing 命中：射线参数 `t` 处触及表面。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceHit {
    /// 命中点射线参数（命中点 = `ro + rd * t`）。
    pub t: f32,
}

/// sphere tracing：沿射线按 SDF 值步进求交（B3；`rd` 必须为单位向量）。
///
/// 命中语义：
/// - 起点已在体内或表面上（`dist(ro) ≤ hit_eps`）→ 命中 `t = 0`（渲染侧自遮挡约定）；
/// - `t` 超过 `t_max` 或步数耗尽 → `None`（未命中，含穿透出 `t_max` 边界的情形）。
///
/// 命中点位置误差 ≤ `hit_eps`（Lipschitz 1 保证了表面必在 `dist` 半径球内）。
pub fn sphere_trace(field: &dyn Sdf, ro: Vec3, rd: Vec3, params: &TraceParams) -> Option<TraceHit> {
    debug_assert!((rd.length() - 1.0).abs() < 1e-4, "sphere_trace requires unit rd");
    let mut t = 0.0_f32;
    for _ in 0..params.max_steps {
        let d = field.dist(ro + rd * t);
        if d <= params.hit_eps {
            return Some(TraceHit { t });
        }
        t += d * params.safety;
        if t > params.t_max {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{Box3, Sphere};
    use crate::{vmax, vmaxc, vmin, vminc};

    const SEED: u64 = 0x6E01_5EED; // 任意常量种子

    /// 域算子与逐点公式对照 + 几何语义（球 A 减球 B）。
    #[test]
    fn domain_operators_match_formulas() {
        let a = -0.7_f32;
        let b = 0.3_f32;
        assert_eq!(union(a, b), a.min(b));
        assert_eq!(intersection(a, b), a.max(b));
        assert_eq!(subtraction(a, b), a.max(-b));
        // 几何语义：subtraction(a, b) = a ∩ ¬b —— 点在 b 深内部（db = −2 → −b = +2）
        // 即被挖空，差集距离由 −b 决定。
        let (da, db) = (-0.1_f32, -2.0_f32);
        assert_eq!(subtraction(da, db), 2.0, "点在 b 深内部 → 被挖空");
        let (da, db) = (-0.1_f32, 0.5_f32); // b 外部：-b = -0.5，交集仍取 a
        assert_eq!(subtraction(da, db), -0.1);
        let (da, db) = (0.1_f32, -0.05_f32); // a 在 b 内：差集为正（被挖空）
        assert_eq!(subtraction(da, db), 0.1);
    }

    /// smooth_min：k → 0 退化为 min、已知名值、下沉界。
    #[test]
    fn smooth_min_limits_and_bounds() {
        let a = 0.8_f32;
        let b = -0.3_f32;
        // k 很小时逐点逼近 min（误差 O(k)）。
        let tiny = smooth_min(a, b, 1e-5);
        assert!((tiny - a.min(b)).abs() < 1e-4, "k→0 退化：{tiny} vs {}", a.min(b));
        // 已知名值：smin(0, 0, k) = −k/4。
        assert!((smooth_min(0.0, 0.0, 0.5) + 0.125).abs() < 1e-6);
        // 下沉界：smin ≥ min − k/4（在重叠区采样）。
        for i in 0..50 {
            let t = i as f32 / 49.0;
            let (x, y) = (-0.2 + 0.4 * t, 0.2 - 0.4 * t); // x、y 交错穿过 0
            let s = smooth_min(x, y, 0.4);
            assert!(s >= x.min(y) - 0.4 / 4.0 - 1e-6, "smin={s} 越界");
            assert!(s <= x.min(y) + 1e-6, "smin 不应高于 min");
        }
        // 平滑交/差是 smin 的对偶变换。
        let (x, y, k) = (0.3_f32, -0.1_f32, 0.25_f32);
        assert_eq!(smooth_intersection(x, y, k), -smooth_min(-x, -y, k));
        assert_eq!(smooth_subtraction(x, y, k), smooth_intersection(x, -y, k));
    }

    /// displace：amp = 0 等价纯基场；确定性（同参两遍逐位一致）。
    #[test]
    fn displace_zero_amp_and_determinism() {
        let sph = Sphere { center: Vec3::splat(0.0), radius: 1.0 };
        for i in 0..8 {
            let p = Vec3::new(0.1 * i as f32, -0.3, 0.2);
            assert_eq!(displace(sph.dist(p), p, 0.0, 1.0, SEED), sph.dist(p), "amp=0 恒等");
            let d1 = displace(sph.dist(p), p, 0.2, 3.0, SEED);
            let d2 = displace(sph.dist(p), p, 0.2, 3.0, SEED);
            assert_eq!(d1, d2, "displace 必须逐位确定");
        }
        // amp > 0 时场值确实被噪声推动（抽样点至少一处不为零位移）。
        let mut moved = false;
        for i in 0..16 {
            let p = Vec3::new(0.3 * i as f32 - 2.0, 0.11, -0.7);
            if (displace(sph.dist(p), p, 0.2, 3.0, SEED) - sph.dist(p)).abs() > 1e-6 {
                moved = true;
            }
        }
        assert!(moved, "amp>0 应产生实际位移");
    }

    /// 解析射线-球交点（测试参照实现）。
    fn ray_sphere(ro: Vec3, rd: Vec3, c: Vec3, r: f32) -> Option<f32> {
        let oc = ro - c;
        let b = oc.dot(&rd);
        let disc = b * b - (oc.dot(&oc) - r * r);
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        let t0 = -b - s;
        let t1 = -b + s;
        if t0 > 1e-6 {
            Some(t0)
        } else if t1 > 1e-6 {
            Some(t1)
        } else {
            None
        }
    }

    /// 解析射线-AABB 交点（slab 法，测试参照实现）。
    fn ray_box(ro: Vec3, rd: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
        let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
        // 逐分量乘（Vec3 只定义了 Mul<f32>）。
        let a = min - ro;
        let b = max - ro;
        let t1 = Vec3::new(a.x * inv.x, a.y * inv.y, a.z * inv.z);
        let t2 = Vec3::new(b.x * inv.x, b.y * inv.y, b.z * inv.z);
        let tmin = vmin(t1, t2);
        let tmax = vmax(t1, t2);
        let enter = vmaxc(tmin).max(0.0);
        let exit = vminc(tmax);
        if enter <= exit {
            Some(enter)
        } else {
            None
        }
    }

    fn trace_params() -> TraceParams {
        TraceParams { hit_eps: 1e-5, ..TraceParams::default() }
    }

    /// 关卡 1a：sphere tracing vs 解析射线求交（球/盒），误差 < 1e-4。
    #[test]
    fn sphere_trace_matches_analytic_ray() {
        let params = trace_params();
        // —— 球（r=1 @ 距离 5 → 命中半角 sinθ ≤ 0.2；dy 扫描保持在命中锥内）——
        let sph = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 1.0 };
        for i in 0..25 {
            let dy = -0.008 * i as f32; // |dy| ≤ 0.192 → sinθ ≤ 0.189，全命中
            let ro = Vec3::splat(0.0);
            let rd = Vec3::new(0.0, dy, -1.0).normalize();
            let got = sphere_trace(&sph, ro, rd, &params).expect("应命中");
            let want = ray_sphere(ro, rd, sph.center, sph.radius).expect("解析应命中");
            assert!(
                (got.t - want).abs() < 1e-4,
                "球 hit {}/25: traced {} vs analytic {}",
                i, got.t, want
            );
        }
        // —— 盒（z∈[−5,−3]，x∈[−0.2,0.8]；dx 扫描保证整段射线穿盒）——
        let bx = Box3 { center: Vec3::new(0.3, -0.2, -4.0), half: Vec3::new(0.5, 0.8, 1.0) };
        for i in 0..25 {
            let dx = 0.008 * i as f32 - 0.035; // x(t)=t·dx 全程在盒 x 范围内
            let ro = Vec3::splat(0.0);
            let rd = Vec3::new(dx, 0.1, -1.0).normalize();
            let got = sphere_trace(&bx, ro, rd, &params).expect("应命中");
            let want = ray_box(ro, rd, bx.center - bx.half, bx.center + bx.half).expect("解析应命中");
            assert!(
                (got.t - want).abs() < 1e-4,
                "盒 hit {}/25: traced {} vs analytic {}",
                i, got.t, want
            );
        }
    }

    /// 关卡 1b：未命中（脱靶 / t_max 边界穿透）与内部起点（穿透边界）用例。
    #[test]
    fn sphere_trace_miss_and_boundary() {
        let params = trace_params();
        let sph = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 1.0 };
        // 脱靶：射线偏离球心 2 倍半径以上 → None。
        let rd = Vec3::new(0.0, 3.0, -1.0).normalize();
        assert!(sphere_trace(&sph, Vec3::splat(0.0), rd, &params).is_none(), "脱靶应未命中");
        // t_max 边界穿透：命中点在 t_max 之外 → None。
        let tight = TraceParams { t_max: 2.0, ..params }; // 球面在 t≈4 之外
        let rd = Vec3::new(0.0, 0.0, -1.0);
        assert!(sphere_trace(&sph, Vec3::splat(0.0), rd, &tight).is_none(), "出界应未命中");
        // 穿透边界：起点在体内 → 命中 t = 0（自遮挡约定）。
        let inside = Vec3::new(0.0, 0.0, -5.0);
        assert_eq!(
            sphere_trace(&sph, inside, Vec3::new(0.0, 0.0, -1.0), &params),
            Some(TraceHit { t: 0.0 })
        );
        // 起点 = 表面 → t = 0。
        let on_surface = Vec3::new(0.0, 1.0, -5.0);
        assert_eq!(
            sphere_trace(&sph, on_surface, Vec3::new(0.0, 1.0, -1.0).normalize(), &params),
            Some(TraceHit { t: 0.0 })
        );
    }

    /// 关卡 1c：CSG 场景 sphere tracing vs 解析对照（并/交/差已知交点）。
    #[test]
    fn sphere_trace_matches_analytic_csg() {
        use crate::csg::Csg;
        let params = trace_params();
        // —— 并集：两分离球，射线命中较近者正面（t = 5 − 1 = 4）。
        let s1 = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 1.0 };
        let s2 = Sphere { center: Vec3::new(0.0, 0.0, -8.0), radius: 1.0 };
        let uni = Csg::Union(Box::new(Csg::Sphere(s1)), Box::new(Csg::Sphere(s2)));
        let rd = Vec3::new(0.0, 0.0, -1.0);
        let got_u = sphere_trace(&uni, Vec3::splat(0.0), rd, &params).unwrap();
        assert!((got_u.t - 4.0).abs() < 1e-4, "并集近球正面 t={} vs 4.0", got_u.t);
        // —— 交集：两重叠球沿连心轴，交区间 = [max(进入), min(离开)] = [4, 7]。
        let a = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 2.0 };
        let b = Sphere { center: Vec3::new(0.0, 0.0, -6.0), radius: 2.0 };
        let inter = Csg::Intersection(Box::new(Csg::Sphere(a)), Box::new(Csg::Sphere(b)));
        let got_i = sphere_trace(&inter, Vec3::splat(0.0), rd, &params).unwrap();
        assert!((got_i.t - 4.0).abs() < 1e-4, "交集进入面 t={} vs 4.0", got_i.t);
        // —— 差集：盒 [-1,1]³ 中心挖球 r=0.5。射线沿 +Z 打盒正面（正面在挖孔区外）t = 3 − 1 = 2。
        let bx = Box3 { center: Vec3::splat(0.0), half: Vec3::splat(1.0) };
        let hole = Sphere { center: Vec3::splat(0.0), radius: 0.5 };
        let diff = Csg::Subtraction(Box::new(Csg::Box(bx)), Box::new(Csg::Sphere(hole)));
        let rd_z = Vec3::new(0.0, 0.0, 1.0);
        let got_d = sphere_trace(&diff, Vec3::new(0.0, 0.0, -3.0), rd_z, &params).unwrap();
        assert!((got_d.t - 2.0).abs() < 1e-4, "差集正面 t={} vs 2.0", got_d.t);
        // 同起点打纯盒：同为 t = 2（挖孔不影响此面）。
        let got_plain = sphere_trace(&Csg::Box(bx), Vec3::new(0.0, 0.0, -3.0), rd_z, &params).unwrap();
        assert!((got_plain.t - 2.0).abs() < 1e-4);
        // —— 差集空腔内出发：起点 = 挖孔中心（被挖空，d = 0.5 > 0），沿 +X 命中挖孔内壁背面 x = 0.5 → t = 0.5。
        let got_cav = sphere_trace(&diff, Vec3::splat(0.0), Vec3::new(1.0, 0.0, 0.0), &params).unwrap();
        assert!((got_cav.t - 0.5).abs() < 1e-4, "空腔内壁背面 t={} vs 0.5", got_cav.t);
        // 沿 +Y 对称命中。
        let got_cav_y = sphere_trace(&diff, Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), &params).unwrap();
        assert!((got_cav_y.t - 0.5).abs() < 1e-4);
        _ = SEED;
    }
}
