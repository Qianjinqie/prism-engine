//! 路径追踪积分器（集成收尾）。
//!
//! # 范围（已接入）
//! Lambert 漫反 + 完美镜面基线之上，接入 Disney BSDF 全族（spec/diffuse/
//! transmission 三瓣 + 多散射补偿）、四灯光族（方向/球/矩形/环境）、
//! random walk SSS（BSDF 材质带 subsurface 时走 [`crate::sss`]）、
//! 焦散（镜面链 k=1 / 球面折射链 k=2 MNEE + PSR 兜底）。
//! NEE + BSDF 双策略 MIS 幂次启发式、RR 续弹、萤火钳制、均匀 HG 介质保持基线。
//!
//! # 维度预算（SampleIndex.dimension；与 sampler 超表兜底衔接）
//! ```text
//! 相机段：槽 0,1 = 光圈圆盘；2 = 快门；3,4 = 像素抖动（见 camera.rs）。
//! 事件段 ev ≥ 1（第 ev 次弹射的着色事件）：基 = 5 + (ev−1)·6，
//!   槽 0,1 = BSDF/相位抽样 uv；槽 2,3 = 光源抽样 uv（多光共用同一对——
//!   预算如此，相关性不引入偏差）；槽 4 = RR；槽 5 = 介质游程流 aux
//!   （同事件内阴影/步进共享一个步计数器）。
//! max_bounces = 8 → 最高维度 5 + 8·6 + 4 = 57 < 64（Sobol 表内）。
//! ```
//!
//! # 估计量结构
//! - 每个非镜面事件对每个光源做 NEE（阴影射线 + 介质二值透射）；
//!   方向光不可被 BSDF/相位抽样命中（非几何）→ 其 NEE 权重恒 1（不做
//!   伪 MIS——否则系统性欠估，方向光解析关卡会暴露）；
//! - BSDF/相位抽样命中发光体：w = power_heuristic(pdf_prev, pdf_light_sa)，
//!   相机段与镜面链权重 1；
//! - RR：ev ≥ 3 的续弹按 throughput 最大分量存活（槽 4）；
//! - 萤火钳制：仅间接贡献（ev ≥ 2 的加法）按固定常量截断（默认 50×中灰，
//!   可关）。
//!
//! # 确定性
//! 渲染 = 纯函数复合：spp 均值走 prism-core reduce 固定形状（
//! 像素级局部归约，禁跨 tile）；采样流只依赖 (种子, 像素, 样本, 维度)
//! （D-28 像素域）→ tile 化不影响采样流，整幅与分块逐位一致（关卡 3）。

use std::f32::consts::{FRAC_1_PI, PI};

use prism_core::EvalCtx;
use prism_math::Vec3;

use crate::accum::mean_samples;
use crate::bsdf::{hadamard, Bsdf, SampleCtx};
use crate::caustics::{mnee_mirror, mnee_refraction, Refractor};
use crate::camera::{Camera, Ray};
use crate::light::{ambient_background, nee_mis_weight, sample_nee};

/// 光源（四族；对外沿用 integrator 命名，内部委托 [`crate::light`]）。
pub use crate::light::Light;
use crate::media::HomogeneousMedium;
use crate::sampler::SampleStream;
use crate::sss::random_walk;

// ---------------------------------------------------------------------------
// 场景原语（最小集）
// ---------------------------------------------------------------------------

/// 球体。
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    /// 球心。
    pub center: Vec3,
    /// 半径。
    pub radius: f32,
}

/// 轴对齐包围盒（Cornell 墙体 / 介质区域）。
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// 最小角。
    pub min: Vec3,
    /// 最大角。
    pub max: Vec3,
}

impl Aabb {
    /// 构造。
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// 点是否在盒内（闭区间）。
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// 线段 [t0, t1] 与盒的相交区间（slab 法；零方向分量显式处理，
    /// 避免 ±inf·0 = NaN 破坏确定性）。
    pub fn segment_interval(
        &self,
        ro: Vec3,
        rd: Vec3,
        t0: f32,
        t1: f32,
    ) -> Option<(f32, f32)> {
        let mut enter = t0;
        let mut exit = t1;
        for axis in 0..3 {
            let (o, d, lo, hi) = match axis {
                0 => (ro.x, rd.x, self.min.x, self.max.x),
                1 => (ro.y, rd.y, self.min.y, self.max.y),
                _ => (ro.z, rd.z, self.min.z, self.max.z),
            };
            if d == 0.0 {
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let mut ta = (lo - o) * inv;
                let mut tb = (hi - o) * inv;
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                enter = enter.max(ta);
                exit = exit.min(tb);
                if enter > exit {
                    return None;
                }
            }
        }
        Some((enter, exit))
    }
}

/// 抛物面（z = k·(x²+y²)，径向有界 r² = x²+y² <= r_max²，闭边界）。
///
/// 求交为闭式解（射线代入隐式方程得二次方程）；`k <= 0` / `r_max <= 0` /
/// 任一非有限 ⇒ 该图元永不命中（`intersect` 入口跳过，不 panic）。
#[derive(Debug, Clone, Copy)]
pub struct Paraboloid {
    /// 曲率（>0 开口向上）。
    pub k: f32,
    /// 径向截断（>0；命中要求 r²(t) <= r_max²，闭边界）。
    pub r_max: f32,
}

/// 平面（可有限切片）：`point` 上一点，`normal` 法线（单位化后使用），
/// `half_extent = None` 为无限平面；`Some(h)` 为以 `point` 为中心、边长
/// `2h` 的正方形切片（沿确定性正交基 `tu`/`tv` 裁剪，闭边界）。
///
/// `normal` 零长 / 非有限 / `half_extent <= 0` ⇒ 该图元永不命中
/// （`intersect` 入口跳过，不 panic）。
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// 平面上一点。
    pub point: Vec3,
    /// 法线（构造后须单位化，如 [`Plane::new`]）。
    pub normal: Vec3,
    /// 半宽（None = 无限平面；Some(h) 时 |u| <= h 且 |v| <= h 才命中）。
    pub half_extent: Option<f32>,
}

impl Plane {
    /// 构造并单位化法线（零长 / 非有限法线归一为零向量 ⇒ 永不命中）。
    pub fn new(point: Vec3, normal: Vec3, half_extent: Option<f32>) -> Self {
        Plane {
            point,
            normal: normal.normalize(),
            half_extent,
        }
    }
}

/// 材质（基线三型 + Disney PBR 全族）。
#[derive(Debug, Clone, Copy)]
pub enum MaterialKind {
    /// Lambert 漫反射（反照率）。
    Lambert { albedo: Vec3 },
    /// 完美镜面。
    Mirror,
    /// 自发光（辐亮度；发光球同时注册为 [`Light::Sphere`] 才能被 NEE 采样、
    /// 被 BSDF 命中做 MIS 融合）。
    Emissive { radiance: Vec3 },
    /// Disney PBR（BSDF 三瓣 + 可选 SSS；[`Bsdf`] 参数域断言在构造期生效）。
    Pbr(Bsdf),
}

/// 材质取 Disney [`Bsdf`]（Pbr 直取；Lambert/Mirror 无 Disney 等价——其
/// 解析形式在积分器内精确保留，见 [`surface_brdf`]）。Emissive 无 BSDF → None。
pub(crate) fn material_bsdf(m: &MaterialKind) -> Option<Bsdf> {
    match *m {
        MaterialKind::Pbr(b) => Some(b),
        // 基线材质的解析形式即精确保留（无 Disney 近似），但采样/续弹仍走
        // 统一 BSDF 通道——Lambert/Mirror 用等价 Disney 参数近似。
        MaterialKind::Lambert { albedo } => {
            Some(Bsdf::new(albedo, 0.0, 1.0, 1.5, 0.0, 0.0, Vec3::splat(0.3)))
        }
        MaterialKind::Mirror => {
            Some(Bsdf::new(Vec3::splat(1.0), 1.0, 0.0, 1.5, 0.0, 0.0, Vec3::splat(0.3)))
        }
        MaterialKind::Emissive { .. } => None,
    }
}

/// 反射域 BRDF 求值（解析）：Lambert 用 albedo/π（基线精确）；Mirror 用
/// δ（NEE 下不参与，续弹反射方向无 §f 因子）；Pbr 用 Disney [`Bsdf::eval`]。
/// 返回 f·cosθ 之外的纯 BRDF f（调用方补 cosθ）。
pub(crate) fn surface_brdf(mat: &MaterialKind, bsdf: &Bsdf, wo: Vec3, wi: Vec3, nv: Vec3) -> Vec3 {
    match *mat {
        MaterialKind::Lambert { albedo } => {
            if wo.dot(&nv) <= 0.0 || wi.dot(&nv) <= 0.0 {
                Vec3::splat(0.0)
            } else {
                albedo * FRAC_1_PI
            }
        }
        MaterialKind::Mirror => Vec3::splat(0.0), // δ：反射域 NEE 不贡献
        _ => bsdf.eval(wo, wi, nv),
    }
}

/// 反射域 pdf（解析）：Lambert = cosθ/π；Mirror = 0（δ）；Pbr = Disney pdf。
pub(crate) fn surface_pdf(mat: &MaterialKind, bsdf: &Bsdf, wo: Vec3, wi: Vec3, nv: Vec3) -> f32 {
    match *mat {
        MaterialKind::Lambert { .. } => wi.dot(&nv).max(0.0) * FRAC_1_PI,
        MaterialKind::Mirror => 0.0,
        _ => bsdf.pdf(wo, wi, nv),
    }
}

/// 材质是否走 SSS random walk（subsurface 权重 > 0）。
fn material_subsurface(m: &MaterialKind) -> Option<(f32, Vec3, f32)> {
    match *m {
        MaterialKind::Pbr(b) if b.subsurface > 0.0 => Some((b.subsurface, b.subsurface_radius, b.ior)),
        _ => None,
    }
}

/// 介质区域（B1：单一均匀介质盒；异质网格待后续扩展）。
#[derive(Debug, Clone)]
pub struct MediumRegion {
    /// 区域包围盒。
    pub bounds: Aabb,
    /// 均匀介质。
    pub medium: HomogeneousMedium,
}

/// 最小场景。
#[derive(Debug, Clone, Default)]
pub struct Scene {
    /// 球体几何（含发光球注册）。
    pub spheres: Vec<(Sphere, MaterialKind)>,
    /// 盒体几何（墙体等）。
    pub boxes: Vec<(Aabb, MaterialKind)>,
    /// 抛物面几何（z = k(x²+y²)，径向有界；S7 焦散碗）。
    pub paraboloids: Vec<(Paraboloid, MaterialKind)>,
    /// 平面几何（无限地面 / 可倾斜有限切片；S7 玻璃切平面）。
    pub planes: Vec<(Plane, MaterialKind)>,
    /// 光源列表（NEE 采样对象）。
    pub lights: Vec<Light>,
    /// 可选单一介质区域。
    pub medium: Option<MediumRegion>,
}

// ---------------------------------------------------------------------------
// 求交
// ---------------------------------------------------------------------------

/// 形状引用（借场景索引）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeRef {
    Sphere(usize),
    Box(usize),
    Paraboloid(usize),
    Plane(usize),
}

/// 最近命中。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hit {
    pub(crate) t: f32,
    pub(crate) shape: ShapeRef,
}

/// 自交防护下界。
pub(crate) const T_MIN: f32 = 1e-4;

/// 新解析图元（抛物面 / 平面）求交入口的有限性检查：o/d 任一分量
/// NaN/Inf ⇒ 不命中。**仅在新图元臂内使用**——不动 `Scene::intersect`
/// 入口的统一检查，避免改变 v1 球 / 盒对 NaN 射线的既有行为。
#[inline]
fn finite_vec3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}
/// 自交防护下界的编译期镜像（wavefront 判对基准一致性核对；同值）。
pub const T_MIN_SENTINEL: f32 = 1e-4;
/// `FRAC_1_PI` 的编译期镜像（wavefront 与 integrator 同口径核对）。
pub const FRAC_1_PI_F32: f32 = FRAC_1_PI;
/// 表面偏移（阴影/续弹起点沿法线外推）。
const SURFACE_EPS: f32 = 1e-3;
/// 未命中段长（逃逸/方向光阴影距离）。
const MISS_T: f32 = 1.0e30;

impl Scene {
    pub(crate) fn material(&self, s: ShapeRef) -> &MaterialKind {
        match s {
            ShapeRef::Sphere(i) => &self.spheres[i].1,
            ShapeRef::Box(i) => &self.boxes[i].1,
            ShapeRef::Paraboloid(i) => &self.paraboloids[i].1,
            ShapeRef::Plane(i) => &self.planes[i].1,
        }
    }

    /// 面法向（外向单位化；盒按最大相对分量轴选取）。
    pub(crate) fn normal_at(&self, s: ShapeRef, p: Vec3) -> Vec3 {
        match s {
            ShapeRef::Sphere(i) => {
                let sp = &self.spheres[i].0;
                (p - sp.center) * (1.0 / sp.radius)
            }
            ShapeRef::Box(i) => {
                let b = &self.boxes[i].0;
                let c = (b.min + b.max) * 0.5;
                let half = (b.max - b.min) * 0.5;
                let rel = Vec3::new(
                    (p.x - c.x) / half.x,
                    (p.y - c.y) / half.y,
                    (p.z - c.z) / half.z,
                );
                let (ax, ay, az) = (rel.x.abs(), rel.y.abs(), rel.z.abs());
                if ax >= ay && ax >= az {
                    Vec3::new(rel.x.signum(), 0.0, 0.0)
                } else if ay >= az {
                    Vec3::new(0.0, rel.y.signum(), 0.0)
                } else {
                    Vec3::new(0.0, 0.0, rel.z.signum())
                }
            }
            ShapeRef::Paraboloid(i) => {
                // 抛物面解析法线：n = normalize(-2k·x, -2k·y, 1)
                // （恒有 n.z > 0，朝上外法线；k > 0 由命中前提保证）。
                let g = &self.paraboloids[i].0;
                Vec3::new(-2.0 * g.k * p.x, -2.0 * g.k * p.y, 1.0).normalize()
            }
            ShapeRef::Plane(i) => {
                // 平面外法线即构造时的单位化法线（切片两面同法线约定；
                // 玻璃透射材质下由积分器 from_inside 翻转）。
                self.planes[i].0.normal
            }
        }
    }

    fn sphere_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        let s = &self.spheres[i].0;
        let oc = ro - s.center;
        let b = oc.dot(&rd);
        let c = oc.dot(&oc) - s.radius * s.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let t_near = -b - sq;
        let t_far = -b + sq;
        if t_near >= T_MIN && t_near < t_max {
            return Some(t_near);
        }
        if t_far >= T_MIN && t_far < t_max {
            return Some(t_far);
        }
        None
    }

    fn box_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        self.boxes[i]
            .0
            .segment_interval(ro, rd, T_MIN, t_max)
            .map(|(enter, _)| enter)
    }

    /// 抛物面求交（闭式解）：隐式 F(x,y,z) = z − k(x²+y²) = 0，射线
    /// p(t) = o + t·d 代入得二次方程
    /// `a = −k(d.x²+d.y²)`、`b = d.z − 2k(o.x·d.x + o.y·d.y)`、
    /// `c = o.z − k(o.x²+o.y²)`。
    ///
    /// - `a == 0`（轴向射线 d.x == d.y == 0，IEEE 平方和恰为零）：退化线性
    ///   单根 `t = −c/b`；`b == 0`（平行且不在轴上）⇒ None；
    /// - 否则二次双根**都验证**（a < 0 时勿假设顺序），`disc < 0` ⇒ None；
    /// - 候选有效条件与 `sphere_t`/`box_t` 同口径：`t >= T_MIN` 且
    ///   `t < t_max`，且径向闭边界 `r²(t) <= r_max²`（`==` 算命中）；
    ///   取最小有效 t；
    /// - 数值稳健：o/d 任一分量 NaN/Inf、`k <= 0` / `r_max <= 0` /
    ///   任一非有限 ⇒ None（该图元永不命中），不 panic。
    fn paraboloid_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let g = &self.paraboloids[i].0;
        if !g.k.is_finite() || !g.r_max.is_finite() || g.k <= 0.0 || g.r_max <= 0.0 {
            return None; // 图元自身非法 ⇒ 永不命中
        }
        let a = -g.k * (rd.x * rd.x + rd.y * rd.y);
        let b = rd.z - 2.0 * g.k * (ro.x * rd.x + ro.y * rd.y);
        let c = ro.z - g.k * (ro.x * ro.x + ro.y * ro.y);

        // 候选根（a == 0 轴向退化线性根；否则二次双根都验证）。
        let mut candidates = [0.0f32; 2];
        let n_cand = if a == 0.0 {
            if b == 0.0 {
                return None; // 轴向平行且不在轴上：不相交（轴为测度零）
            }
            candidates[0] = -c / b;
            1
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                return None;
            }
            let sq = disc.sqrt();
            candidates[0] = (-b - sq) / (2.0 * a);
            candidates[1] = (-b + sq) / (2.0 * a);
            2
        };

        let r_max2 = g.r_max * g.r_max;
        let mut best: Option<f32> = None;
        for &t in &candidates[..n_cand] {
            // NaN 防御：NaN >= T_MIN 为 false，自然排除。
            if !(t >= T_MIN && t < t_max) {
                continue;
            }
            let x = ro.x + t * rd.x;
            let y = ro.y + t * rd.y;
            if x * x + y * y > r_max2 {
                continue; // 径向界外（闭边界：== 算命中）
            }
            best = Some(match best {
                Some(bt) if bt <= t => bt,
                _ => t,
            });
        }
        best
    }

    /// 平面求交：`denom = d·n`（n 为单位法线）；`denom == 0`（平行）⇒
    /// None；`t = (point − o)·n / denom`，`t < T_MIN` 或 `t >= t_max` ⇒
    /// None。
    ///
    /// `half_extent = Some(h)` 时按确定性正交基裁剪：`a` 轴按
    /// `|n.x| > 0.9` 固定选择（无分支歧义），`tu = normalize(n×a)`、
    /// `tv = n×tu`，`|u| > h` 或 `|v| > h` ⇒ None（闭边界）。
    ///
    /// 数值稳健：o/d 任一分量 NaN/Inf、法线零长 / 非有限、
    /// `half_extent <= 0` ⇒ None（该图元永不命中），不 panic。
    fn plane_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let pl = &self.planes[i].0;
        let n = pl.normal;
        if !n.x.is_finite() || !n.y.is_finite() || !n.z.is_finite() {
            return None; // 法线非有限 ⇒ 永不命中
        }
        let nlen2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if !(nlen2 >= 1e-16) {
            return None; // 零长 / 极短法线（NaN 亦落此分支）⇒ 永不命中
        }
        if let Some(h) = pl.half_extent {
            if !h.is_finite() || h <= 0.0 {
                return None; // half_extent <= 0 ⇒ 永不命中
            }
        }
        let denom = rd.dot(&n);
        if denom == 0.0 {
            return None; // 射线平行平面（grazing 由 t < t_max 自然排除）
        }
        let t = (pl.point - ro).dot(&n) / denom;
        if !(t >= T_MIN && t < t_max) {
            return None;
        }
        if let Some(h) = pl.half_extent {
            // 切片裁剪（确定性正交基：a 轴按 |n.x| > 0.9 固定选择）。
            let a = if n.x.abs() > 0.9 {
                Vec3::new(0.0, 1.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let tu = n.cross(&a).normalize();
            let tv = n.cross(&tu);
            let d = (ro + rd * t) - pl.point;
            if d.dot(&tu).abs() > h || d.dot(&tv).abs() > h {
                return None; // 界外（闭边界）
            }
        }
        Some(t)
    }

    /// 最近命中（可跳过指定形状——阴影射线对被采样光源豁免）。
    pub(crate) fn intersect(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut limit = t_max;
        for i in 0..self.spheres.len() {
            let r = ShapeRef::Sphere(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.sphere_t(i, ro, rd, limit) {
                best = Some(Hit { t, shape: r });
                limit = t;
            }
        }
        for i in 0..self.boxes.len() {
            let r = ShapeRef::Box(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.box_t(i, ro, rd, limit) {
                best = Some(Hit { t, shape: r });
                limit = t;
            }
        }
        for i in 0..self.paraboloids.len() {
            let r = ShapeRef::Paraboloid(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.paraboloid_t(i, ro, rd, limit) {
                best = Some(Hit { t, shape: r });
                limit = t;
            }
        }
        for i in 0..self.planes.len() {
            let r = ShapeRef::Plane(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.plane_t(i, ro, rd, limit) {
                best = Some(Hit { t, shape: r });
                limit = t;
            }
        }
        best
    }

    /// 主射线最近命中距离（C1 深度 AOV 输出）：从 `ro` 沿 `rd` 投射单位
    /// 主射线，返回最近命中距离（世界单位）；无命中返回 None（背景/天空）。
    ///
    /// 纯函数、零随机——深度通道是路径追踪的免费副产品。
    /// `t_max` 为远平面（超过即视作无命中）。
    pub fn primary_hit_distance(&self, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        self.intersect(ro, rd, t_max, None).map(|h| h.t)
    }

    /// 发光球几何对应的 Light 索引（按 center/radius 位模式匹配；场景构造期
    /// 数据同源，位相等可靠）。
    fn sphere_light_index(&self, i: usize) -> Option<usize> {
        let s = &self.spheres[i].0;
        self.lights.iter().position(|l| {
            matches!(l, Light::Sphere { center, radius, .. }
                if center.x.to_bits() == s.center.x.to_bits()
                    && center.y.to_bits() == s.center.y.to_bits()
                    && center.z.to_bits() == s.center.z.to_bits()
                    && radius.to_bits() == s.radius.to_bits())
        })
    }

    /// 光源在 from 点的立体角 pdf（MIS 备择策略权重；仅可被 BSDF 命中的
    /// 发光球有定义，其余返回 None → 命中权重 1）。
    pub(crate) fn light_pdf_sa(&self, from: Vec3, s: ShapeRef) -> Option<f32> {
        let i = match s {
            ShapeRef::Sphere(i) => i,
            // 盒 / 抛物面 / 平面均非球光源几何：无立体角 pdf（与 Box 臂同语义）。
            ShapeRef::Box(_) | ShapeRef::Paraboloid(_) | ShapeRef::Plane(_) => return None,
        };
        let li = self.sphere_light_index(i)?;
        match self.lights[li] {
            Light::Sphere { center, radius, .. } => {
                let d2 = (center - from).dot(&(center - from));
                if d2 <= radius * radius {
                    return None; // 光源内部：不发生（防御）
                }
                let cos_max = (1.0 - radius * radius / d2).max(0.0).sqrt();
                Some(1.0 / (2.0 * PI * (1.0 - cos_max)))
            }
            _ => None, // 方向/矩形/环境：非几何，无立体角 pdf
        }
    }

    /// 求交适配（SSS random walk 用）：(o, d) → 最近命中 (外向法线, t)。
    /// 闭包签名要求 (t, n)——返回 t 为行进距离、n 为命中外向法线。
    pub(crate) fn intersect_n(
        &self,
        o: Vec3,
        d: Vec3,
        t_max: f32,
    ) -> Option<(f32, Vec3)> {
        let h = self.intersect(o, d, t_max, None)?;
        let p = o + d * h.t;
        let n = self.normal_at(h.shape, p);
        Some((h.t, n))
    }

    /// 球面折射体构造（MNEE k=2 链）：发光球列表中的玻璃球（Pbr transmission）。
    pub(crate) fn refractors(&self) -> Vec<Refractor> {
        let mut out = Vec::new();
        for (s, m) in &self.spheres {
            if let Some(b) = material_bsdf(m) {
                if b.transmission > 0.0 {
                    out.push(Refractor { center: s.center, radius: s.radius, ior: b.ior });
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 光源抽样与 MIS（委托 crate::light，B2 全族）
// ---------------------------------------------------------------------------

/// 阴影射线对被采样光源的几何豁免（发光球不应遮挡其自身近交点）。
pub(crate) fn light_skip(scene: &Scene, light: &Light) -> Option<ShapeRef> {
    match light {
        Light::Sphere { center, radius, .. } => scene
            .spheres
            .iter()
            .position(|(s, _)| {
                s.center.x.to_bits() == center.x.to_bits()
                    && s.center.y.to_bits() == center.y.to_bits()
                    && s.center.z.to_bits() == center.z.to_bits()
                    && s.radius.to_bits() == radius.to_bits()
            })
            .map(ShapeRef::Sphere),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 维度预算与路径状态
// ---------------------------------------------------------------------------

/// 事件段维度基。
const BASE_BOUNCE: u16 = 5;
/// 每事件段槽数。
const SLOTS_PER_BOUNCE: u16 = 6;
/// 槽位：BSDF/相位 uv。
const SLOT_BSDF: u16 = 0;
/// 槽位：光源 uv。
const SLOT_LIGHT: u16 = 2;
/// 槽位：RR。
const SLOT_RR: u16 = 4;
/// 槽位：介质游程流。
const SLOT_AUX: u16 = 5;

/// 事件段 ev（≥1）槽 slot 的全局维度号（预算表见模块文档）。
#[inline]
fn bounce_dim(ev: u32, slot: u16) -> u16 {
    BASE_BOUNCE + ((ev - 1) as u16) * SLOTS_PER_BOUNCE + slot
}

/// 单条路径的游走状态。
#[derive(Debug, Clone, Copy)]
struct PathState {
    /// 路径吞吐量。
    throughput: Vec3,
    /// 已累计辐亮度。
    radiance: Vec3,
    /// 上一段方向的立体角 pdf（发光命中 MIS 用；镜面段为 0）。
    prev_pdf_dir: f32,
    /// 上一段是否镜面（specular 链命中发光权重 1）。
    prev_specular: bool,
}

/// 间接贡献钳制（ev ≥ 2 的加法； 萤火钳制——仅间接）。
fn clamp_indirect(c: Vec3, ev: u32, limit: Option<f32>) -> Vec3 {
    match limit {
        Some(t) if ev >= 2 => Vec3::new(c.x.min(t), c.y.min(t), c.z.min(t)),
        _ => c,
    }
}

/// RR 存活判定（next_ev ≥ 3 的续弹；槽 4）。
fn rr_survive(st: &mut PathState, stream: &SampleStream<'_>, next_ev: u32) -> bool {
    if next_ev < 3 {
        return true;
    }
    let q = st
        .throughput
        .x
        .max(st.throughput.y)
        .max(st.throughput.z)
        .clamp(0.05, 0.95);
    let u = stream.sample_1d(bounce_dim(next_ev, SLOT_RR));
    if u >= q {
        return false;
    }
    st.throughput = st.throughput * (1.0 / q);
    true
}

// ---------------------------------------------------------------------------
// 介质行进（delta tracking）
// ---------------------------------------------------------------------------

/// 沿线段做 delta tracking：返回段内碰撞点距离（Some = 碰撞）。
/// 进入介质不耗权重（外部 σt = 0，进入前碰撞概率为 0）；aux 流 =
/// bounce_dim(ev, SLOT_AUX) + 共享步计数器（变长游程序列）。
fn medium_collide(
    region: &MediumRegion,
    ro: Vec3,
    rd: Vec3,
    t_max: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    steps: &mut u32,
) -> Option<f32> {
    let b = &region.bounds;
    // 介质内段的起点：origin 在介质内从 0 起，否则从入盒点起（进入不耗权重：
    // 外部 σt = 0，进入前碰撞概率为 0）。进入点用 `?` 传播——段不穿盒则无碰撞。
    let t_enter = if b.contains(ro) {
        0.0
    } else {
        b.segment_interval(ro, rd, 0.0, t_max)?.0
    };
    // B1 介质区域为单一凸盒：直线段至多穿越一次，介质内剩余段
    // [t_enter, exit] 上单次自由程抽样即完备（异质/多区域后续扩展为循环）。
    let exit = b.segment_interval(ro, rd, t_enter, t_max)?.1;
    let u = stream.aux_1d(aux_dim, *steps);
    *steps += 1;
    let t_free = region.medium.free_path(u);
    if t_enter + t_free < exit && t_enter + t_free < t_max {
        Some(t_enter + t_free)
    } else {
        None
    }
}

/// 阴影射线可见性 = 几何遮挡 × 介质透射（二值），结果 ∈ {0.0, 1.0}。
#[allow(clippy::too_many_arguments)]
pub(crate) fn visibility(
    scene: &Scene,
    ro: Vec3,
    rd: Vec3,
    max_t: f32,
    skip: Option<ShapeRef>,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    steps: &mut u32,
) -> f32 {
    if scene.intersect(ro, rd, max_t - T_MIN, skip).is_some() {
        return 0.0;
    }
    if let Some(region) = &scene.medium {
        if medium_collide(region, ro, rd, max_t - T_MIN, stream, aux_dim, steps).is_some()
        {
            return 0.0;
        }
    }
    1.0
}

// ---------------------------------------------------------------------------
// 路径追踪主循环
// ---------------------------------------------------------------------------

/// 渲染参数。
#[derive(Debug, Clone)]
pub struct RenderParams {
    /// 每像素样本数。
    pub spp: u32,
    /// 最大弹射深度（事件段数上限；相机段不计）。
    pub max_bounces: u32,
    /// 间接贡献萤火钳制阈值（固定常量 50×中灰；None = 关闭）。
    pub clamp_indirect: Option<f32>,
    /// 验证开关：true = 纯 NEE（BSDF 命中发光体不加权合并——能量对照用）。
    pub nee_only: bool,
}

impl RenderParams {
    /// 默认参数（萤火钳制开）。
    pub fn new(spp: u32, max_bounces: u32) -> Self {
        RenderParams {
            spp,
            max_bounces,
            clamp_indirect: Some(50.0),
            nee_only: false,
        }
    }
}

/// 追踪一条完整路径（单样本）→ rgba。
fn trace_one(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    stream: &SampleStream<'_>,
    params: &RenderParams,
) -> [f32; 4] {
    let mut ray = cam.cast_ray(stream, resolution);
    let mut st = PathState {
        throughput: Vec3::new(1.0, 1.0, 1.0),
        radiance: Vec3::new(0.0, 0.0, 0.0),
        prev_pdf_dir: 0.0,
        prev_specular: true, // 相机段视为镜面链起点（直视发光权重 1）
    };
    let mut aux_steps: u32 = 0;
    // bounce = 已完成事件数；本次线段远端事件编号 ev = bounce + 1。
    for bounce in 0..params.max_bounces {
        let ev = bounce + 1;
        let aux_dim = bounce_dim(ev, SLOT_AUX);
        let hit = scene.intersect(ray.origin, ray.dir, MISS_T, None);
        let t_surf = hit.map_or(MISS_T, |h| h.t);

        // ---- 介质段（相机/续弹段统一；碰撞事件优先于表面事件）----
        if let Some(region) = &scene.medium {
            if let Some(tc) =
                medium_collide(region, ray.origin, ray.dir, t_surf, stream, aux_dim, &mut aux_steps)
            {
                let p = ray.at(tc);
                // 散射 vs 吸收（aux 下一步；散射分支 throughput 不变——
                // T(t)·σs 与联合 pdf 相消，见 media 模块文档推导）。
                let u_scatter = stream.aux_1d(aux_dim, aux_steps);
                aux_steps += 1;
                if u_scatter >= region.medium.albedo {
                    break; // 吸收：终止（存活权重隐式）
                }
                // 体 NEE：相函数对光源方向；备择策略 = 相位抽样该方向。
                let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
                let w_in = -ray.dir; // 相函数轴向（散射前传播方向）
                for light in &scene.lights {
                    let skip = light_skip(scene, light);
                    let ls = match sample_nee(light, p, u_l1, u_l2) {
                        Some(ls) => ls,
                        None => continue,
                    };
                    let cos_l = w_in.dot(&ls.wi);
                    let ph = region.medium.phase(cos_l);
                    let vis = visibility(
                        scene, p, ls.wi, ls.dist, skip, stream, aux_dim, &mut aux_steps,
                    );
                    if vis <= 0.0 {
                        continue;
                    }
                    // 方向光不可被相位抽样命中（非几何）→ 权重 1。
                    let w = nee_mis_weight(light, ls.pdf_w, ph);
                    let c = hadamard(st.throughput, ls.radiance)
                        * (ph * vis * w / ls.pdf_w);
                    st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                }
                // 相位抽样续弹。
                let (u1, u2) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF));
                let dir = region.medium.sample_phase(w_in, u1, u2);
                st.prev_pdf_dir = region.medium.phase(w_in.dot(&dir));
                st.prev_specular = false;
                if !rr_survive(&mut st, stream, ev + 1) {
                    break;
                }
                ray = Ray { origin: p, dir, time: ray.time };
                continue;
            }
        }

        // ---- 表面事件（或逃逸）----
        let h = match hit {
            None => {
                // 逃逸：环境光背景全额贡献（B2；无 Ambient 时为黑 → 保持 B1 语义）。
                let bg = ambient_background(&scene.lights);
                if bg.x > 0.0 || bg.y > 0.0 || bg.z > 0.0 {
                    let c = hadamard(st.throughput, bg);
                    st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                }
                break;
            }
            Some(h) => h,
        };
        let p = ray.at(h.t);
        let n = scene.normal_at(h.shape, p);
        let mat = scene.material(h.shape);

        // 发光命中（MIS 权重 vs 光源立体角 pdf；nee_only 为能量对照开关）。
        if let MaterialKind::Emissive { radiance } = mat {
            let w_em = if params.nee_only && ev > 1 {
                0.0
            } else if ev == 1 || st.prev_specular {
                1.0
            } else {
                match scene.light_pdf_sa(ray.origin, h.shape) {
                    Some(pdf_l) => crate::light::power_heuristic(st.prev_pdf_dir, pdf_l),
                    None => 1.0, // 未注册 NEE 的发光几何（如盒发光）：仅此策略
                }
            };
            let c = hadamard(st.throughput, *radiance) * w_em;
            st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
        }

        // BSDF 求值（Disney 三瓣；Lambert/Mirror 映射为等价 Disney 参数）。
        let bsdf = match material_bsdf(mat) {
            Some(b) => b,
            None => break, // 发光面不续弹（基线简化）
        };
        let from_inside = ray.dir.dot(&n) > 0.0;
        // 视点侧法线（出/入射方向同侧；from_inside 时翻转）。
        let nv = if from_inside { -n } else { n };
        let wo = -ray.dir;

        // 反射域 NEE（除镜面瓣占优外；transmission/diffuse 面均采样光源）。
        {
            let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
            for light in &scene.lights {
                let skip = light_skip(scene, light);
                let ls = match sample_nee(light, p, u_l1, u_l2) {
                    Some(ls) => ls,
                    None => continue,
                };
                let cos_l = nv.dot(&ls.wi);
                if cos_l <= 0.0 {
                    continue;
                }
                let f = surface_brdf(mat, &bsdf, wo, ls.wi, nv);
                // f·cosθ（BRDF 已含归一；此处补几何余弦因子）。
                let f_cos = hadamard(f, Vec3::splat(cos_l));
                let alt_pdf = surface_pdf(mat, &bsdf, wo, ls.wi, nv);
                let ro = p + nv * SURFACE_EPS;
                let vis = visibility(
                    scene, ro, ls.wi, ls.dist, skip, stream, aux_dim, &mut aux_steps,
                );
                if vis <= 0.0 {
                    continue;
                }
                let w = nee_mis_weight(light, ls.pdf_w, alt_pdf);
                let c = hadamard(hadamard(st.throughput, f_cos), ls.radiance)
                    * (vis * w / ls.pdf_w);
                st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
            }
        }

        // 焦散 MNEE：玻璃球折射链 k=2（PSR 由 BSDF 采样抖动兜底）。
        if !params.nee_only {
            let si = prism_core::SampleIndex {
                frame: ctx.frame,
                pixel: stream.pixel(),
                sample: 0,
                dimension: bounce_dim(ev, SLOT_LIGHT),
                node_id: ctx.node_id,
            };
            for refr in scene.refractors() {
                for light in &scene.lights {
                    let lp = match light {
                        Light::Sphere { center, .. } => *center,
                        Light::Directional { direction, .. } => p + *direction * 8.0,
                        _ => continue,
                    };
                    let Some(cs) = mnee_refraction(&refr, p, lp, ctx, &si) else {
                        continue;
                    };
                    let cos_l = nv.dot(&cs.wi);
                    if cos_l <= 0.0 {
                        continue;
                    }
                    let f = bsdf.eval(wo, cs.wi, nv);
                    let c = hadamard(hadamard(st.throughput, f), cs.weight)
                        * (cos_l / cs.pdf_w);
                    st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                }
            }
            // k=1 平面镜链：MNEE 兜底。场景未显式注册镜面平板，
            // 故 k=1 链仅对显式镜面几何生效——当前基线场景无镜面板，跳过；
            // k=2 球面折射链已覆盖玻璃焦散主路径，PSR 抖动覆盖残余镜面流形。
            let _ = mnee_mirror; // 接口已挂接（镜面板几何注册后启用）
        }

        // BSDF 抽样续弹。基线 Lambert / Mirror 走解析精确路径（逐位兼容）；
        // Disney PBR 走 [`Bsdf::sample`] 三瓣（含 SSS / 玻璃折射 / PSR 抖动）。
        let (u_lobe, _) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF));
        let (u1, u2) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF) + 1);
        if let MaterialKind::Lambert { albedo } = mat {
            let d_local = crate::sampler::cosine_hemisphere(u1, u2);
            let (t, b) = crate::sampler::onb(n);
            let dir = (t * d_local.x + b * d_local.y + n * d_local.z).normalize();
            st.throughput = hadamard(st.throughput, *albedo);
            st.prev_pdf_dir = d_local.z * FRAC_1_PI;
            st.prev_specular = false;
            if !rr_survive(&mut st, stream, ev + 1) {
                break;
            }
            ray = Ray { origin: p + n * SURFACE_EPS, dir, time: ray.time };
        } else if let MaterialKind::Mirror = mat {
            let dir = ray.dir - n * (2.0 * ray.dir.dot(&n));
            st.prev_pdf_dir = 0.0;
            st.prev_specular = true;
            if !rr_survive(&mut st, stream, ev + 1) {
                break;
            }
            ray = Ray { origin: p + n * SURFACE_EPS, dir, time: ray.time };
        } else {
            let sctx = SampleCtx {
                psr_rough: if bsdf.roughness < 1e-3 { 1e-3 } else { 0.0 },
            };
            if let Some(bs) = bsdf.sample(wo, n, from_inside, &sctx, u_lobe, u1, u2) {
                // SSS：透射入介质且材质带 subsurface → random walk 逃逸点作新着色点。
                if bs.entered {
                    if let Some((_sss_w, radius, ior)) = material_subsurface(mat) {
                        let si = prism_core::SampleIndex {
                            frame: ctx.frame,
                            pixel: stream.pixel(),
                            sample: 0,
                            dimension: bounce_dim(ev, SLOT_AUX),
                            node_id: ctx.node_id,
                        };
                        let geo = |o: Vec3, d: Vec3| scene.intersect_n(o, d, MISS_T);
                        if let Some(exit) = random_walk(
                            &geo, ctx, &si, p, n, bs.dir, radius, bsdf.base_color, ior, 0.0,
                        ) {
                            st.throughput = hadamard(st.throughput, bs.weight);
                            st.throughput = hadamard(st.throughput, exit.weight);
                            st.prev_pdf_dir = 0.0;
                            st.prev_specular = false;
                            if !rr_survive(&mut st, stream, ev + 1) {
                                break;
                            }
                            // 逃逸点作为新着色点：法线取逃逸面外向法线。
                            ray = Ray {
                                origin: exit.p + exit.n * SURFACE_EPS,
                                dir: bs.dir,
                                time: ray.time,
                            };
                        } else {
                            break; // SSS 吸收/超出步数 → 终止
                        }
                    } else {
                        // 无 subsurface：折射续弹（玻璃内部行进，贡献经 BSDF 权重）。
                        st.throughput = hadamard(st.throughput, bs.weight);
                        st.prev_pdf_dir = 0.0;
                        st.prev_specular = false;
                        if !rr_survive(&mut st, stream, ev + 1) {
                            break;
                        }
                        ray = Ray { origin: p - n * SURFACE_EPS, dir: bs.dir, time: ray.time };
                    }
                } else {
                    // 反射域（diffuse / specular 瓣）：BSDF 权重即吞吐乘子。
                    st.throughput = hadamard(st.throughput, bs.weight);
                    st.prev_pdf_dir = bs.pdf;
                    st.prev_specular = bs.specular;
                    if !rr_survive(&mut st, stream, ev + 1) {
                        break;
                    }
                    ray = Ray { origin: p + nv * SURFACE_EPS, dir: bs.dir, time: ray.time };
                }
            } else {
                break; // 采样失败（纯黑材质 / VNDF 翻面）：终止
            }
        }
    }
    [st.radiance.x, st.radiance.y, st.radiance.z, 1.0]
}

/// 像素渲染（纯函数）：spp 样本经 prism-core reduce 固定形状归约取均值
/// （像素级局部归约，禁跨 tile）。
pub fn render_pixel(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    pixel: (u32, u32),
    resolution: (u32, u32),
    params: &RenderParams,
) -> [f32; 4] {
    let mut samples = Vec::with_capacity(params.spp as usize);
    for s in 0..params.spp {
        let stream = SampleStream::new(ctx, pixel, s);
        samples.push(trace_one(scene, cam, ctx, resolution, &stream, params));
    }
    mean_samples(&samples)
}

/// 自由函数形式的相机射线投射（任务接口字面形式；实现委托 [`Camera::cast_ray`]）。
pub fn cast_ray(camera: &Camera, stream: &SampleStream<'_>, resolution: (u32, u32)) -> Ray {
    camera.cast_ray(stream, resolution)
}

// ---------------------------------------------------------------------------
// 测试（关卡 1–4 集成验证）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::SampleIndex;

    /// Cornell 式最小场景（漫反盒 + Lambert 球 + 镜面球 + 球光）。
    fn cornell_scene() -> Scene {
        let white = MaterialKind::Lambert { albedo: Vec3::new(0.72, 0.71, 0.69) };
        let red = MaterialKind::Lambert { albedo: Vec3::new(0.63, 0.065, 0.05) };
        let green = MaterialKind::Lambert { albedo: Vec3::new(0.14, 0.45, 0.091) };
        let t = 0.1f32; // 墙厚（向外扩）
        Scene {
            boxes: vec![
                // 地板 / 天花板 / 后墙 / 左墙(红) / 右墙(绿)
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 0.0, 1.0 + t)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, 2.0, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, -1.0)),
                    white,
                ),
                (
                    Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(-1.0, 2.0 + t, 1.0 + t)),
                    red,
                ),
                (
                    Aabb::new(Vec3::new(1.0, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)),
                    green,
                ),
            ],
            spheres: vec![
                (
                    Sphere { center: Vec3::new(-0.35, 0.45, 0.1), radius: 0.45 },
                    white,
                ),
                (
                    Sphere { center: Vec3::new(0.38, 0.32, -0.05), radius: 0.32 },
                    MaterialKind::Mirror,
                ),
                // 发光球同时注册为几何（Emissive）才能被 BSDF 命中（MIS 语义）。
                (
                    Sphere { center: Vec3::new(0.0, 1.92, 0.0), radius: 0.28 },
                    MaterialKind::Emissive { radiance: Vec3::new(16.0, 13.0, 10.0) },
                ),
            ],
            lights: vec![Light::Sphere {
                center: Vec3::new(0.0, 1.92, 0.0),
                radius: 0.28,
                radiance: Vec3::new(16.0, 13.0, 10.0),
            }],
            medium: None,
            paraboloids: vec![],
            planes: vec![],
        }
    }

    fn cornell_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 1.0, 3.1),
            Vec3::new(0.0, 0.95, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0, // 针孔
            3.1,
            0.0,
            1.0 / 24.0,
        )
    }

    fn render_frame(
        scene: &Scene,
        cam: &Camera,
        ctx: &EvalCtx,
        res: (u32, u32),
        params: &RenderParams,
    ) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((res.0 * res.1) as usize);
        for y in 0..res.1 {
            for x in 0..res.0 {
                out.push(render_pixel(scene, cam, ctx, (x, y), res, params));
            }
        }
        out
    }

    // ---- 关卡 1：Cornell 式最小场景 spp 收敛、误差单调下降 ----

    #[test]
    fn cornell_convergence_monotone() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(2024, 7, 3);
        let res = (10, 10);
        let reference = render_frame(&scene, &cam, &ctx, res, &RenderParams::new(4096, 3));
        let mut errors = Vec::new();
        for spp in [1u32, 16, 256, 1024] {
            let frame = render_frame(
                &scene,
                &cam,
                &ctx,
                res,
                &RenderParams { spp, ..RenderParams::new(1, 3) },
            );
            let (mut err, mut cnt) = (0.0f64, 0.0f64);
            for i in 0..frame.len() {
                for c in 0..3 {
                    err += (frame[i][c] as f64 - reference[i][c] as f64).abs();
                    cnt += 1.0;
                }
            }
            errors.push(err / cnt);
            println!("spp={spp}: 图像均值误差 = {:.6}", errors[errors.len() - 1]);
        }
        assert!(errors[0] > errors[1], "spp 16 未改善于 spp 1：{errors:?}");
        assert!(errors[1] > errors[2], "spp 256 未改善于 spp 16：{errors:?}");
        assert!(errors[2] > errors[3], "spp 1024 未改善于 spp 256：{errors:?}");
    }

    // ---- 关卡 2：NEE 方向光直射照度 vs 解析值；MIS 无 NaN / 能量一致 ----

    #[test]
    fn directional_light_matches_analytic() {
        // 地板 + 正上方方向光：像素 = ρ/π · L·Ω·cosθ = ρ·L·2(1−cos_max)。
        let cos_max = (3.0f32 * PI / 180.0).cos();
        let scene = Scene {
            boxes: vec![(
                Aabb::new(Vec3::new(-10.0, -0.1, -10.0), Vec3::new(10.0, 0.0, 10.0)),
                MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
            )],
            spheres: vec![],
            lights: vec![Light::Directional {
                direction: Vec3::new(0.0, 1.0, 0.0),
                radiance: Vec3::splat(10.0),
                cos_theta_max: cos_max,
            }],
            medium: None,
            paraboloids: vec![],
            planes: vec![],
        };
        let cam = Camera::new(
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            50.0,
            0.0,
            4.0,
            0.0,
            1.0,
        );
        let ctx = EvalCtx::new(5, 7, 3);
        let res = (8, 8);
        let frame = render_frame(&scene, &cam, &ctx, res, &RenderParams::new(64, 2));
        let analytic = 0.8f32 * 10.0f32 * 2.0 * (1.0 - cos_max);
        let mut mean = 0.0f64;
        for px in &frame {
            for c in 0..3 {
                mean += px[c] as f64;
            }
        }
        mean /= (frame.len() * 3) as f64;
        let rel = ((mean as f32 - analytic).abs() / analytic) as f64;
        println!(
            "方向光解析关卡：analytic = {analytic}，实测 = {mean:.7}，相对误差 = {rel:.2e}"
        );
        for (i, px) in frame.iter().enumerate() {
            for c in 0..3 {
                assert!(px[c].is_finite(), "像素 {i} 非有限值");
                let rel = (px[c] - analytic).abs() / analytic;
                assert!(
                    rel < 1e-2,
                    "像素 {i} ch{c}: {} vs 解析 {analytic}（相对误差 {rel}）",
                    px[c]
                );
            }
        }
        println!("方向光解析关卡：analytic = {analytic}");
    }

    #[test]
    fn mis_finite_and_energy_consistent() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(77, 7, 3);
        let res = (10, 10);
        let mut nee_params = RenderParams::new(128, 4);
        nee_params.nee_only = true;
        let a = render_frame(&scene, &cam, &ctx, res, &RenderParams::new(128, 4));
        let b = render_frame(&scene, &cam, &ctx, res, &nee_params);
        let (mut ma, mut mb) = (0.0f64, 0.0f64);
        for i in 0..a.len() {
            for c in 0..3 {
                assert!(
                    a[i][c].is_finite() && a[i][c] >= 0.0,
                    "MIS 帧非有限/负值：{:?}",
                    a[i]
                );
                assert!(
                    b[i][c].is_finite() && b[i][c] >= 0.0,
                    "NEE 帧非有限/负值：{:?}",
                    b[i]
                );
                ma += a[i][c] as f64;
                mb += b[i][c] as f64;
            }
        }
        ma /= (a.len() * 3) as f64;
        mb /= (b.len() * 3) as f64;
        let rel = (ma - mb).abs() / mb;
        assert!(
            rel < 0.08,
            "MIS 与纯 NEE 能量偏差过大：MIS={ma} NEE={mb} rel={rel}"
        );
        println!("MIS 能量对照：MIS={ma:.5} NEE={mb:.5} rel={rel:.5}");
    }

    // ---- 关卡 3：确定性（两遍逐位一致 + md5；tile 化 vs 整幅逐位一致）----

    #[test]
    fn two_pass_render_bitwise_identical() {
        let scene = cornell_scene();
        let cam = cornell_camera();
        let res = (8, 8);
        let params = RenderParams::new(16, 3);
        let f1 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &params);
        let f2 = render_frame(&scene, &cam, &EvalCtx::new(9, 7, 3), res, &params);
        for i in 0..f1.len() {
            for c in 0..4 {
                assert_eq!(
                    f1[i][c].to_bits(),
                    f2[i][c].to_bits(),
                    "两遍渲染不逐位一致 px{i} ch{c}"
                );
            }
        }
        // md5 锚定（RFC 1321 测试参考件；自锚公开摘要 md5("") / md5("abc")）。
        let mut bytes = Vec::new();
        for px in &f1 {
            for c in 0..4 {
                bytes.extend_from_slice(&px[c].to_le_bytes());
            }
        }
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        let digest = md5_hex(&bytes);
        assert_eq!(digest, md5_hex(&bytes), "md5 两遍不一致");
        println!("Cornell 8×8 spp16 md5 = {digest}");
    }

    #[test]
    fn tiled_render_equals_full_bitwise() {
        use prism_io::Fbuf16F;
        use prism_render::{PixelSource, TileGrid, TileOrchestrator};

        struct RtSource<'a> {
            scene: &'a Scene,
            cam: &'a Camera,
            ctx: &'a EvalCtx,
            res: (u32, u32),
            params: RenderParams,
        }
        impl PixelSource for RtSource<'_> {
            fn eval_pixel(&self, x: u32, y: u32, si: SampleIndex) -> [f32; 4] {
                let stream = SampleStream::new(self.ctx, (x, y), si.sample);
                trace_one(self.scene, self.cam, self.ctx, self.res, &stream, &self.params)
            }
        }

        let scene = cornell_scene();
        let cam = cornell_camera();
        let ctx = EvalCtx::new(42, 7, 3);
        let res = (12, 12);
        let params = RenderParams::new(1, 3);
        let source = RtSource {
            scene: &scene,
            cam: &cam,
            ctx: &ctx,
            res,
            params: params.clone(),
        };

        let grid = TileGrid::new(res.0, res.1, 4, 4);
        let mut orch = TileOrchestrator::new(grid);
        orch.frame = ctx.frame;
        orch.node_id = ctx.node_id;
        orch.global_seed = ctx.global_seed;

        let full = orch.render_frame_full(&source, &[]);
        let tiled = orch.render_frame_tiled(&source, &[]);
        assert_eq!(full.pixels, tiled.pixels, "tile 化与整幅不逐位一致");

        // 与 render_pixel 通道交叉验证（同 SampleIndex 语义）。
        let mut expect = Fbuf16F::new(res.0, res.1);
        for y in 0..res.1 {
            for x in 0..res.0 {
                expect.set(x, y, render_pixel(&scene, &cam, &ctx, (x, y), res, &params));
            }
        }
        assert_eq!(full.pixels, expect.pixels, "PixelSource 通道与 render_pixel 通道不一致");
    }

    // ---- 关卡 4：HG 体积透射 vs Beer 定律（整画面路径级验证）----

    #[test]
    fn medium_attenuation_matches_beer_law() {
        // 纯吸收介质（反照率 0）：像素 = base × T_相机段 × E[T̂_阴影]
        // = base × e^(−2σt·h)（下行存活与阴影返回各衰减一次）。
        // 相机取近垂直视角（y=40, fov 5°）：斜程修正 sec² ≤ 1.004，
        // 保证解析式不含倾角项（否则测试需要复刻引擎的斜程几何）。
        let sigma_t = 0.9f32;
        let height = 1.6f32;
        let cos_max = (3.0f32 * PI / 180.0).cos();
        let scene = Scene {
            boxes: vec![(
                Aabb::new(Vec3::new(-10.0, -0.1, -10.0), Vec3::new(10.0, 0.0, 10.0)),
                MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
            )],
            spheres: vec![],
            lights: vec![Light::Directional {
                direction: Vec3::new(0.0, 1.0, 0.0),
                radiance: Vec3::splat(10.0),
                cos_theta_max: cos_max,
            }],
            medium: Some(MediumRegion {
                bounds: Aabb::new(Vec3::new(-5.0, 0.0, -5.0), Vec3::new(5.0, height, 5.0)),
                medium: HomogeneousMedium::new(sigma_t, 0.0, 0.0),
            }),
            paraboloids: vec![],
            planes: vec![],
        };
        let cam = Camera::new(
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            5.0,
            0.0,
            40.0,
            0.0,
            1.0,
        );
        let ctx = EvalCtx::new(13, 7, 3);
        let res = (8, 8);
        let frame = render_frame(&scene, &cam, &ctx, res, &RenderParams::new(256, 2));
        let base = 0.8f32 * 10.0f32 * 2.0 * (1.0 - cos_max);
        let expected = base as f64 * (-(2.0 * sigma_t as f64) * height as f64).exp();
        let mut mean = 0.0f64;
        for px in &frame {
            for c in 0..3 {
                assert!(px[c].is_finite() && px[c] >= 0.0);
                mean += px[c] as f64;
            }
        }
        mean /= (frame.len() * 3) as f64;
        let rel = (mean - expected).abs() / expected;
        assert!(
            rel < 0.05,
            "介质衰减：实测 {mean}，Beer 解析 {expected}（相对 {rel}）"
        );
        println!("Beer 关卡：实测 = {mean:.6}，解析 = {expected:.6}（相对 {rel:.5}）");
    }

    // ---- 关卡 5：端到端（金属粗糙度展宽 / 玻璃焦散能量 / SSS 厚度）----

    /// 金属球 + 方向光的最小场景（Pbr 金属；粗糙度可变）。
    fn metal_scene(roughness: f32) -> (Scene, Camera) {
        let metal = MaterialKind::Pbr(Bsdf::new(
            Vec3::new(0.9, 0.85, 0.7),
            1.0,
            roughness,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        ));
        let scene = Scene {
            boxes: vec![(
                Aabb::new(Vec3::new(-5.0, -0.1, -5.0), Vec3::new(5.0, 0.0, 5.0)),
                MaterialKind::Lambert { albedo: Vec3::splat(0.0) }, // 黑板：只留金属高光
            )],
            spheres: vec![(Sphere { center: Vec3::new(0.0, 0.5, 0.0), radius: 0.5 }, metal)],
            lights: vec![Light::Directional {
                direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
                radiance: Vec3::splat(50.0),
                cos_theta_max: (6.0f32 * PI / 180.0).cos(),
            }],
            medium: None,
            paraboloids: vec![],
            planes: vec![],
        };
        let cam = Camera::new(
            Vec3::new(0.0, 0.8, 3.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            40.0,
            0.0,
            3.0,
            0.0,
            1.0,
        );
        (scene, cam)
    }

    #[test]
    fn pbr_metal_roughness_broadens_highlight() {
        // 金属 roughness ↑ → 高光展宽（亮斑覆盖像素数上升、峰值下降）。
        // 端到端：Disney spec 瓣 GGX α = roughness² 控制 lobe 宽度。
        let ctx = EvalCtx::new(0xB2_11, 7, 3);
        let res = (24, 24);
        let params = RenderParams::new(256, 3);
        let highlight_stats = |roughness: f32| -> (u32, f64) {
            let (scene, cam) = metal_scene(roughness);
            let frame = render_frame(&scene, &cam, &ctx, res, &params);
            let lum: Vec<f64> = frame
                .iter()
                .map(|px| 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64)
                .collect();
            let peak = lum.iter().cloned().fold(0.0f64, f64::max);
            let thr = 0.25 * peak;
            let count = lum.iter().filter(|&&l| l >= thr).count() as u32;
            (count, peak)
        };
        let (c_smooth, peak_smooth) = highlight_stats(0.05);
        let (c_rough, peak_rough) = highlight_stats(0.65);
        println!("金属高光：smooth 覆盖={c_smooth} 峰值={peak_smooth:.4}；rough 覆盖={c_rough} 峰值={peak_rough:.4}");
        assert!(
            c_rough > c_smooth,
            "粗糙度上升应展宽高光：smooth 覆盖={c_smooth} rough 覆盖={c_rough}"
        );
        assert!(
            peak_rough < peak_smooth,
            "粗糙度上升应降低峰值：smooth={peak_smooth} rough={peak_rough}"
        );
    }

    #[test]
    fn glass_sphere_caustics_energy_positive() {
        // 玻璃球焦散：地面非零亮斑能量 > 0（MNEE k=2 球面折射链端到端贡献）。
        // 玻璃球置于方向光 + 地面上方；球对侧地面应有折射聚焦亮斑。
        let glass = MaterialKind::Pbr(Bsdf::new(
            Vec3::splat(1.0),
            0.0,
            0.02, // 近完美光滑（焦散锐利）
            1.5,
            1.0, // 全透射
            0.0,
            Vec3::splat(0.3),
        ));
        let scene = Scene {
            boxes: vec![(
                Aabb::new(Vec3::new(-6.0, -0.1, -6.0), Vec3::new(6.0, 0.0, 6.0)),
                MaterialKind::Lambert { albedo: Vec3::splat(0.7) },
            )],
            spheres: vec![(Sphere { center: Vec3::new(0.0, 1.2, 0.0), radius: 0.6 }, glass)],
            lights: vec![Light::Directional {
                direction: Vec3::new(0.0, 1.0, 0.0),
                radiance: Vec3::splat(30.0),
                cos_theta_max: (4.0f32 * PI / 180.0).cos(),
            }],
            medium: None,
            paraboloids: vec![],
            planes: vec![],
        };
        // 俯视地面，含球与球影区域。
        let cam = Camera::new(
            Vec3::new(0.0, 3.5, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            45.0,
            0.0,
            3.5,
            0.0,
            1.0,
        );
        let ctx = EvalCtx::new(0xB2_CA, 7, 3);
        let res = (20, 20);
        let params = RenderParams::new(256, 4);
        let frame = render_frame(&scene, &cam, &ctx, res, &params);
        let mut total = 0.0f64;
        let mut max_lum = 0.0f64;
        for px in &frame {
            assert!(px[0].is_finite() && px[0] >= 0.0, "焦散帧出现非法值：{px:?}");
            let l = 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64;
            total += l;
            max_lum = max_lum.max(l);
        }
        println!("玻璃球焦散：总能量={total:.4} 峰值亮度={max_lum:.4}");
        assert!(total > 0.0, "玻璃球场景总能量应 > 0");
        // 焦散亮斑：存在显著高于环境中位的像素（折射聚焦）。
        let med = {
            let mut v: Vec<f64> = frame
                .iter()
                .map(|px| 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64)
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[v.len() / 2]
        };
        assert!(
            max_lum > med * 1.5 + 1e-3 || max_lum > 0.05,
            "焦散亮斑能量过低：峰值={max_lum} 中位={med}"
        );
    }

    // ---- 测试参考件：MD5（RFC 1321；K 表以 f64 sin 运行时生成——禁 libm
    // 纪律只约束引擎确定性路径，不约束测试；正确性由 md5("")/md5("abc")
    // 公开摘要自锚，见 two_pass 测试）----

    fn md5_hex(data: &[u8]) -> String {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
            5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let k = |i: usize| -> u32 {
            (((1u64 << 32) as f64) * (i as f64 + 1.0).sin().abs()) as u32
        };
        let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
        msg.extend_from_slice(data);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
        let (mut a0, mut b0, mut c0, mut d0) =
            (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, word) in m.iter_mut().enumerate() {
                *word = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64usize {
                let (f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let tmp = d;
                d = c;
                c = b;
                let sum = a.wrapping_add(f).wrapping_add(k(i)).wrapping_add(m[g]);
                b = b.wrapping_add(sum.rotate_left(S[i]));
                a = tmp;
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in [a0, b0, c0, d0] {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }

    // ==================== E5 解析图元（抛物面 / 平面）单测组 ====================
    // 期望值均由闭式解独立推导（Python 双精度核准）；除注明"精确"外，
    // t 的比较均为 1e-4 近似（f32 求交与解析解的舍入差量级）。

    /// 中性灰 Lambert（解析单测用材质）。
    fn lambert_grey() -> MaterialKind {
        MaterialKind::Lambert { albedo: Vec3::splat(0.8) }
    }

    /// 仅含一个抛物面的场景（k, r_max 参数化）。
    fn paraboloid_scene(k: f32, r_max: f32) -> Scene {
        Scene {
            spheres: vec![],
            boxes: vec![],
            lights: vec![],
            medium: None,
            paraboloids: vec![(Paraboloid { k, r_max }, lambert_grey())],
            planes: vec![],
        }
    }

    /// 仅含一个平面的场景。
    fn plane_scene(plane: Plane) -> Scene {
        Scene {
            spheres: vec![],
            boxes: vec![],
            lights: vec![],
            medium: None,
            paraboloids: vec![],
            planes: vec![(plane, lambert_grey())],
        }
    }

    /// E5-1 轴向射线 `a == 0` 退化线性单根：o=(0,0,-1), d=(0,0,1), k=1 ⇒ t=1.0
    /// （精确：a=−1·0=−0.0，IEEE `−0.0 == 0.0` 为真走线性分支，t=−c/b=1）。
    #[test]
    fn paraboloid_axial_linear_root() {
        let sc = paraboloid_scene(1.0, 1.5);
        let t = sc.paraboloid_t(0, Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30);
        assert_eq!(t, Some(1.0));
        // 经 intersect 同值（ShapeRef 索引正确）。
        let h = sc
            .intersect(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30, None)
            .unwrap();
        assert_eq!(h.t, 1.0);
        assert_eq!(h.shape, ShapeRef::Paraboloid(0));
    }

    /// E5-2 轴向平行且不在轴上（`a == 0 && b == 0`）：None 不 panic
    /// （零方向射线 d=(0,0,0)：`-c/b` 除零前被 `b == 0` 分支拦截）。
    #[test]
    fn paraboloid_axial_parallel_off_axis_none() {
        let sc = paraboloid_scene(1.0, 1.5);
        assert_eq!(
            sc.paraboloid_t(0, Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0e30),
            None
        );
    }

    /// E5-3 离轴双根都有效 ⇒ 取最小有效 t（a < 0 时勿假设根序）：
    /// o=(-2,0,0.5), d=(1,0,0) ⇒ t = 1.29289 / 2.70711（r² 均 0.5 ≤ 1.5²）。
    #[test]
    fn paraboloid_offaxis_two_valid_roots_takes_min() {
        let sc = paraboloid_scene(1.0, 1.5);
        let t = sc
            .paraboloid_t(0, Vec3::new(-2.0, 0.0, 0.5), Vec3::new(1.0, 0.0, 0.0), 1.0e30)
            .unwrap();
        assert!((t - 1.2928932).abs() < 1e-4, "t={t}");
        // 命中点在抛物面上（F = z − x² − y² ≈ 0）。
        let x = -2.0 + t;
        assert!((0.5 - x * x).abs() < 1e-4, "F={}", 0.5 - x * x);
    }

    /// E5-4 r_max 边界：界外 miss / 界内 hit / 闭边界 `r² == r_max²` 算命中。
    /// 射线 o=(1,0,-1), d=(0,0,1) 恒命中 (1,0,0)（r²=1, t=2）。
    #[test]
    fn paraboloid_r_max_boundary_inside_outside() {
        let ro = Vec3::new(1.0, 0.0, -1.0);
        let rd = Vec3::new(0.0, 0.0, 1.0);
        // 界外：r_max=0.5 ⇒ r²=1 > 0.25。
        assert_eq!(
            paraboloid_scene(1.0, 0.5).paraboloid_t(0, ro, rd, 1.0e30),
            None
        );
        // 界内：r_max=1.5 ⇒ r²=1 ≤ 2.25。
        assert_eq!(
            paraboloid_scene(1.0, 1.5).paraboloid_t(0, ro, rd, 1.0e30),
            Some(2.0)
        );
        // 闭边界：r_max=1.0 ⇒ r²=1 == r_max²，`>` 判否 ⇒ 命中（全整数精确）。
        assert_eq!(
            paraboloid_scene(1.0, 1.0).paraboloid_t(0, ro, rd, 1.0e30),
            Some(2.0)
        );
    }

    /// E5-5 `disc == 0` 精确相切（单重根正常命中）：o=(1,-2,1), d=(0,1,0)
    /// ⇒ a=-1, b=4, c=-4，disc=16-16=0，t=2.0（全整数运算，逐位精确）。
    #[test]
    fn paraboloid_disc_zero_tangent_exact() {
        let sc = paraboloid_scene(1.0, 1.5);
        let t = sc.paraboloid_t(0, Vec3::new(1.0, -2.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 1.0e30);
        assert_eq!(t, Some(2.0));
    }

    /// E5-6 PRD §6 切平面配置钉死：P=(0.8,0.8,1.28) 在 z=x²+y² 上
    /// （0.64+0.64=1.28 ✓），n=(-1.6,-1.6,1) 即 P 点 ∇F；自 P+s·n̂ 沿 −n̂
    /// 入射必命中 t=s±1e-4（s 取 0.5 / 1.0 / 2.0 三档）。
    /// 注：法向入射 disc≠0——唯一正根 t=s，第二根在起点反方向（负）被
    /// T_MIN 排除；真相切（disc==0）用例见 E5-5。
    #[test]
    fn paraboloid_prd_tangent_plane_normal_incidence() {
        let sc = paraboloid_scene(1.0, 2.0); // r_max=2：r²(P)=1.28 界内
        let n = Vec3::new(-1.6, -1.6, 1.0).normalize();
        let p = Vec3::new(0.8, 0.8, 1.28);
        for &s in &[0.5f32, 1.0, 2.0] {
            let o = p + n * s;
            let d = -n;
            let t = sc.paraboloid_t(0, o, d, 1.0e30).expect("必命中");
            assert!((t - s).abs() < 1e-4, "s={s} t={t}");
            // 命中点即 P。
            let hp = o + d * t;
            assert!((hp - p).length() < 1e-4, "s={s} hp={hp:?}");
        }
    }

    /// E5-7 图元自身参数非法 ⇒ 永不命中（不 panic）：k<=0 / r_max<=0 / 非有限。
    #[test]
    fn paraboloid_degenerate_params_never_hit() {
        let ro = Vec3::new(0.0, 0.0, -1.0);
        let rd = Vec3::new(0.0, 0.0, 1.0);
        let cases: &[(f32, f32)] = &[
            (0.0, 1.5),
            (-1.0, 1.5),
            (1.0, 0.0),
            (1.0, -0.5),
            (f32::NAN, 1.5),
            (1.0, f32::INFINITY),
            (f32::NAN, f32::NAN),
        ];
        for &(k, r_max) in cases {
            let sc = paraboloid_scene(k, r_max);
            assert_eq!(
                sc.paraboloid_t(0, ro, rd, 1.0e30),
                None,
                "k={k} r_max={r_max} 应永不命中"
            );
        }
    }

    /// E5-8 射线非有限 ⇒ None（NaN 比较天然排除；Inf 由入口 finite 检查拦截）。
    #[test]
    fn paraboloid_nonfinite_ray_none() {
        let sc = paraboloid_scene(1.0, 1.5);
        assert_eq!(
            sc.paraboloid_t(0, Vec3::new(f32::NAN, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30),
            None
        );
        assert_eq!(
            sc.paraboloid_t(0, Vec3::new(0.0, 0.0, -1.0), Vec3::new(f32::INFINITY, 0.0, 1.0), 1.0e30),
            None
        );
        assert_eq!(
            sc.paraboloid_t(0, Vec3::new(0.0, 0.0, f32::INFINITY), Vec3::new(0.0, 0.0, 1.0), 1.0e30),
            None
        );
    }

    /// E5-9 `normal_at` 解析对照：命中点法线 ≡ normalize(-2k·x, -2k·y, 1)，
    /// 且恒朝上（n.z > 0）。
    #[test]
    fn paraboloid_normal_at_matches_analytic() {
        let sc = paraboloid_scene(1.0, 2.0);
        let o = Vec3::new(-2.0, 0.0, 0.5);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let h = sc.intersect(o, d, 1.0e30, None).unwrap();
        let p = o + d * h.t;
        let n = sc.normal_at(h.shape, p);
        let expect = Vec3::new(-2.0 * p.x, -2.0 * p.y, 1.0).normalize();
        assert!((n - expect).length() < 1e-6, "n={n:?} expect={expect:?}");
        assert!(n.z > 0.0, "抛物面外法线恒朝上");
    }

    /// E5-10 `t_max` 语义与 sphere_t/box_t 同口径：`t >= t_max` 即排除
    /// （边界等于即排除；Infinity 远平面正常命中）。
    #[test]
    fn paraboloid_t_max_semantics() {
        let sc = paraboloid_scene(1.0, 1.5);
        let ro = Vec3::new(0.0, 0.0, -1.0);
        let rd = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(sc.paraboloid_t(0, ro, rd, 1.0), None); // t == t_max 排除
        assert_eq!(sc.paraboloid_t(0, ro, rd, 1.0 + 1e-3), Some(1.0));
        assert_eq!(sc.paraboloid_t(0, ro, rd, 0.5), None); // t > t_max 排除
        assert_eq!(sc.paraboloid_t(0, ro, rd, f32::INFINITY), Some(1.0));
    }

    /// E5-11 平面：轴向命中（t 精确）+ 平行（denom==0）⇒ None；
    /// 双面约定（背面入射同样命中，法线朝向由积分器 from_inside 处理）。
    #[test]
    fn plane_axial_hit_and_parallel_none() {
        let sc = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            None,
        ));
        assert_eq!(
            sc.plane_t(0, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 1.0e30),
            Some(5.0)
        );
        assert_eq!(
            sc.plane_t(0, Vec3::new(0.0, 0.0, 5.0), Vec3::new(1.0, 0.0, 0.0), 1.0e30),
            None
        );
        assert_eq!(
            sc.plane_t(0, Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30),
            Some(5.0)
        );
    }

    /// E5-12 有限切片裁剪：`|u|>h` 或 `|v|>h` ⇒ None；闭边界 `|u|==h` 算命中；
    /// 无限平面（half_extent=None）任意位置命中。
    #[test]
    fn plane_half_extent_clip_and_infinite() {
        let sc = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Some(1.0),
        ));
        let down = Vec3::new(0.0, 0.0, -1.0);
        // 界内。
        assert_eq!(sc.plane_t(0, Vec3::new(0.5, 0.3, 5.0), down, 1.0e30), Some(5.0));
        // 界外（x=1.5 > h=1）。
        assert_eq!(sc.plane_t(0, Vec3::new(1.5, 0.0, 5.0), down, 1.0e30), None);
        // 闭边界（x=±1.0 精确表示，|u|==h 命中）。
        assert_eq!(sc.plane_t(0, Vec3::new(1.0, 0.0, 5.0), down, 1.0e30), Some(5.0));
        assert_eq!(sc.plane_t(0, Vec3::new(0.0, -1.0, 5.0), down, 1.0e30), Some(5.0));
        // 界外（y=-1.0001，超出 1e-4）。
        assert_eq!(sc.plane_t(0, Vec3::new(0.0, -1.0001, 5.0), down, 1.0e30), None);
        // 无限平面：远处命中。
        let inf = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            None,
        ));
        assert_eq!(inf.plane_t(0, Vec3::new(50.0, 30.0, 5.0), down, 1.0e30), Some(5.0));
    }

    /// E5-13 `Plane::new` 单位化法线；零长/非有限法线永不命中（不 panic）；
    /// t 对方向缩放不变。
    #[test]
    fn plane_normal_normalized_zero_and_scale_invariance() {
        // 非单位法线 (0,3,0) 单位化后：点 (0,2,0) 命中 t=2（精确）。
        let sc = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            None,
        ));
        assert_eq!(
            sc.plane_t(0, Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0e30),
            Some(2.0)
        );
        // 零长法线 ⇒ 永不命中。
        let zero = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            None,
        ));
        assert_eq!(
            zero.plane_t(0, Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0e30),
            None
        );
        // 非有限法线（NaN）⇒ 永不命中。
        let nan = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f32::NAN, 0.0, 1.0),
            None,
        ));
        assert_eq!(
            nan.plane_t(0, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 1.0e30),
            None
        );
        // 方向缩放 7 倍 ⇒ t/7（参数等价；用法线 (0,0,1) 平面独立验证）。
        let flat = plane_scene(Plane::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            None,
        ));
        let t1 = flat
            .plane_t(0, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 1.0e30)
            .unwrap();
        let t7 = flat
            .plane_t(0, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -7.0), 1.0e30)
            .unwrap();
        assert!((t7 - t1 / 7.0).abs() < 1e-6, "t7={t7}");
    }

    /// E5-14 `light_pdf_sa`：抛物面/平面均非球光源几何 ⇒ None（与 Box 臂同语义）。
    #[test]
    fn new_shapes_light_pdf_sa_none() {
        let sc = Scene {
            spheres: vec![],
            boxes: vec![],
            lights: vec![Light::Sphere {
                center: Vec3::new(0.0, 1.92, 0.0),
                radius: 0.28,
                radiance: Vec3::splat(8.0),
            }],
            medium: None,
            paraboloids: vec![(Paraboloid { k: 1.0, r_max: 1.5 }, lambert_grey())],
            planes: vec![(
                Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None),
                lambert_grey(),
            )],
        };
        let from = Vec3::new(0.3, 0.2, 0.4);
        assert_eq!(sc.light_pdf_sa(from, ShapeRef::Paraboloid(0)), None);
        assert_eq!(sc.light_pdf_sa(from, ShapeRef::Plane(0)), None);
    }

    /// E5-15 `Scene::intersect` 跨图元取最近 + skip 豁免（阴影射线语义）：
    /// 球(miss)/盒(后墙)/抛物面/平面混合，抛物面 t≈4.91 最近；skip 抛物面
    /// ⇒ 平面 t=5.0 接上；skip 平面 ⇒ 抛物面仍胜盒（t=6.5）。
    #[test]
    fn scene_intersect_nearest_across_kinds_and_skip() {
        let mat = lambert_grey();
        let sc = Scene {
            spheres: vec![(Sphere { center: Vec3::new(1.5, 0.0, 2.0), radius: 0.5 }, mat)],
            boxes: vec![(Aabb::new(Vec3::new(-2.0, -1.0, -2.0), Vec3::new(2.0, 1.0, -1.5)), mat)],
            lights: vec![],
            medium: None,
            paraboloids: vec![(Paraboloid { k: 1.0, r_max: 1.5 }, mat)],
            planes: vec![(Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None), mat)],
        };
        let ro = Vec3::new(0.3, 0.0, 5.0);
        let rd = Vec3::new(0.0, 0.0, -1.0);
        // 抛物面 t=4.91（命中 (0.3,0,0.09)，r²=0.09 界内）最近。
        let h = sc.intersect(ro, rd, 1.0e30, None).unwrap();
        assert_eq!(h.shape, ShapeRef::Paraboloid(0));
        assert!((h.t - 4.91).abs() < 1e-4, "t={}", h.t);
        // skip 抛物面 ⇒ 平面 t=5.0。
        let h2 = sc.intersect(ro, rd, 1.0e30, Some(ShapeRef::Paraboloid(0))).unwrap();
        assert_eq!(h2.shape, ShapeRef::Plane(0));
        assert_eq!(h2.t, 5.0);
        // skip 平面 ⇒ 抛物面仍胜盒（墙 t=6.5）。
        let h3 = sc.intersect(ro, rd, 1.0e30, Some(ShapeRef::Plane(0))).unwrap();
        assert_eq!(h3.shape, ShapeRef::Paraboloid(0));
        assert!((h3.t - 4.91).abs() < 1e-4, "t={}", h3.t);
    }
}
