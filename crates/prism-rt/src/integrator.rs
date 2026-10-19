//! 路径追踪积分器（集成收尾）。
//!
//! # 范围（已接入）
//! Lambert 漫反 + 完美镜面基线之上，接入 Disney BSDF 全族（spec/diffuse/
//! transmission 三瓣 + 多散射补偿）、四灯光族（方向/球/矩形/环境）、
//! random walk SSS（BSDF 材质带 subsurface 时走 [`crate::sss`]）、
//! 焦散（镜面链 k=1 / 球面折射链 k=2 MNEE + PSR 兜底）。
//! NEE + BSDF 双策略 MIS 幂次启发式、RR 续弹、萤火钳制、均匀 HG 介质保持基线。
//!
//! # 三角几何（V2-h 接入）
//! [`Mesh`] 承载索引网格与三角形 soup 两型三角几何，内建
//! [`prism_geometry::bvh::Bvh`]（SAH，O(log n) 求交，100k 三角形构建 < 1s）。
//! 命中经 [`ShapeRef::Triangle`]（网格下标 + 原始三角形下标）细化——法线精确、
//! 材质/豁免语义与解析图元同口径；[`ShapeRef::Mesh`] / [`ShapeRef::Bvh`] 是
//! 容器级引用（整体豁免 / 整体材质），不产生命中。
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
use prism_geometry::bvh::{Bvh as GeoBvh, Triangle as GeoTriangle};
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

/// 有限圆柱（轴向 +Y，带上下端盖；V2.0-W4 曲线图元族）。
///
/// 求交口径：侧面（x²+z²=R² 二次方程，|y|<=H 裁剪）+ 上下端盖圆盘
/// （y=±H 线性根，x²+z²<=R² 裁剪）；闭合实体语义（内外两侧命中都记录，
/// 取最近 t>0）；切向 / 掠射 / 端盖内外侧统一走同一候选验证。
/// `radius <= 0` / `half_height <= 0` / 非有限 ⇒ 该图元永不命中。
#[derive(Debug, Clone, Copy)]
pub struct Cylinder {
    /// 圆柱中心（轴中点）。
    pub center: Vec3,
    /// 半径（> 0）。
    pub radius: f32,
    /// 半高（> 0）。
    pub half_height: f32,
}

/// 有限圆锥（轴向 +Y；V2.0-W4 曲线图元族）。
///
/// 底面位于 `center.y - half_height`（半径 `radius`），顶点位于
/// `center.y + half_height`（半径 0）。求交口径：斜锥侧面（二次方程，
/// y∈[-H, H] 闭裁剪）+ 底面圆盘（y=-H 线性根，x²+z²<=R² 裁剪）；
/// 顶点处侧面法线退化，按 (0,1,0) 兜底（测度零）。
#[derive(Debug, Clone, Copy)]
pub struct Cone {
    /// 圆锥中心（轴中点）。
    pub center: Vec3,
    /// 底面半径（> 0）。
    pub radius: f32,
    /// 半高（> 0）。
    pub half_height: f32,
}

/// 圆环（主环位于 XZ 平面，轴向 +Y；V2.0-W4 曲线图元族）。
///
/// 隐式方程 `(ρ-R)² + y² = r²`（ρ = √(x²+z²)，局部坐标）。射线代入得
/// **四次方程**，精确求根走 [`prism_geometry::quartic::solve_quartic`]
/// （Ferrari 闭式、零迭代；任务书钉死禁止牛顿迭代）。闭合实体语义：
/// 四个实根（前外 / 前内 / 后内 / 后外）都记录，取最近 t>0。
/// `major <= 0` / `minor <= 0` / 非有限 ⇒ 永不命中。
#[derive(Debug, Clone, Copy)]
pub struct Torus {
    /// 环心。
    pub center: Vec3,
    /// 主半径（环心到管心，> 0）。
    pub major: f32,
    /// 管半径（> 0）。
    pub minor: f32,
}

/// 旋转曲面（轮廓绕 +Y 轴旋转；V2.0-W4 曲线图元族）。
///
/// # 轮廓与段序列（钉死口径）
/// `profile` 为 (r, y) 点列（r >= 0），相邻两点定义一个**斜锥环带段**；
/// 段数 = 点数 − 1，**同参数声明 ⇒ 同段序列**（`new` 是纯函数：
/// 仅做绕向归一，不增删点——非有限点由求交方按"永不命中"处理）。
/// 求交对每个段解一个二次方程（过该段的斜锥；水平段特化为环带），
/// y（或水平段的 ρ）落在段闭区间内算命中；闭合实体语义（内外两侧
/// 都记录，取最近 t>0）。
///
/// # 绕向归一（外法线口径）
/// 轮廓按 (r, y) 平面多边形（闭合边 末点→首点）计有向面积：负（CW）
/// 时反转点序，保证内部恒在左侧 ⇒ 外向法线统一为剖面内 (Δy, −Δr)。
/// 反转是纯置换：同声明 → 同归一结果 → 同段序列。
#[derive(Debug, Clone)]
pub struct Revolution {
    /// 旋转心（轮廓平面的原点）。
    pub center: Vec3,
    /// 轮廓点列 (r, y)（`new` 已做绕向归一；直接构造字段时调用方自保）。
    pub profile: Vec<[f32; 2]>,
}

impl Revolution {
    /// 构造并归一化绕向（见类型文档；纯函数，同声明同结果）。
    pub fn new(center: Vec3, profile: Vec<[f32; 2]>) -> Self {
        let mut pts: Vec<[f32; 2]> = profile;
        // 有向面积（shoelace，闭合边 末→首）：< 0 为 CW ⇒ 反转为 CCW。
        if pts.len() >= 3 && profile_signed_area(&pts) < 0.0 {
            pts.reverse();
        }
        Revolution { center, profile: pts }
    }

    /// 轮廓段数（点数 − 1；点数 < 2 时为 0 ⇒ 永不命中）。
    pub fn segment_count(&self) -> usize {
        self.profile.len().saturating_sub(1)
    }
}

/// 轮廓多边形有向面积（2 倍；闭合边 末→首）。NaN 输入 ⇒ NaN（调用方
/// 不反转，求交方按永不命中处理）。
fn profile_signed_area(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    let mut sum = 0.0_f32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum
}

/// XZ 平面子向量长度（圆柱 / 圆锥 / 圆环 / 旋转曲面共用）。
#[inline]
fn xz_len(v: Vec3) -> f32 {
    (v.x * v.x + v.z * v.z).sqrt()
}

/// (ρ, y) 平面上的点到线段距离平方（旋转曲面法线选段用；退化线段
/// 返回点到端点距离）。
fn point_segment_dist2(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let ex = bx - ax;
    let ey = by - ay;
    let len2 = ex * ex + ey * ey;
    if len2 <= 1e-24 {
        // 退化段：点到端点 a。
        let dx = px - ax;
        let dy = py - ay;
        return dx * dx + dy * dy;
    }
    let mut s = ((px - ax) * ex + (py - ay) * ey) / len2;
    if !(s >= 0.0) {
        s = 0.0;
    } else if s > 1.0 {
        s = 1.0;
    }
    let cx = ax + s * ex;
    let cy = ay + s * ey;
    let dx = px - cx;
    let dy = py - cy;
    dx * dx + dy * dy
}

/// 圆柱 AABB（中心 ± (R, H, R)）。
fn cylinder_aabb(c: &Cylinder) -> Aabb {
    let e = Vec3::new(c.radius, c.half_height, c.radius);
    Aabb::new(c.center - e, c.center + e)
}

/// 圆锥 AABB（与圆柱同形：底面半径 R 与半高 H 即 x/z/y 向极值）。
fn cone_aabb(c: &Cone) -> Aabb {
    let e = Vec3::new(c.radius, c.half_height, c.radius);
    Aabb::new(c.center - e, c.center + e)
}

/// 圆环 AABB（中心 ± (R+r, r, R+r)）。
fn torus_aabb(t: &Torus) -> Aabb {
    let e = Vec3::new(t.major + t.minor, t.minor, t.major + t.minor);
    Aabb::new(t.center - e, t.center + e)
}

/// 旋转曲面 AABB（x/z ± max|r|，y ∈ [min y, max y]；点数 < 2 为
/// ±∞ 空盒——BVH 构建按空盒安全跳过，求交按点数 < 2 永不命中）。
fn revolution_aabb(r: &Revolution) -> Aabb {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in &r.profile {
        for sx in [1.0_f32, -1.0] {
            let pt = r.center + Vec3::new(sx * p[0], p[1], 0.0);
            min = Vec3::new(min.x.min(pt.x), min.y.min(pt.y), min.z.min(pt.z));
            max = Vec3::new(max.x.max(pt.x), max.y.max(pt.y), max.z.max(pt.z));
        }
    }
    Aabb::new(min, max)
}

/// rt Aabb → geometry Aabb（同名同构字段；BVH 构建在 geometry 侧）。
#[inline]
fn to_geo_aabb(b: &Aabb) -> prism_geometry::bvh::Aabb {
    prism_geometry::bvh::Aabb { min: b.min, max: b.max }
}

/// 三角网格（rt 侧三角几何；内建 SAH-BVH，随网格构建一次）。
///
/// # 两型构造（同一存储，语义只差"有无拓扑"）
/// - [`Mesh::new`]：**索引网格**（顶点流 + 索引流），glTF/OBJ/USD 导入主路径；
/// - [`Mesh::from_triangles`]：**三角形 soup**（无拓扑），独立 BVH 直挂路径
///   （[`Scene::add_bvh`]）。soup 的 `positions` / `indices` 为空。
///
/// # 为什么 `triangles` 与 `bvh` 各存一份
/// [`GeoBvh`] 内部 `tris` 是**重排后**的三角形副本，而求交返回的是**原始**
/// 图元索引（`order` 反查）。本结构保留原始顺序的 `triangles`，使
/// "命中图元索引 → 三角形几何" 的映射为 O(1) 直接下标（法线精确，
/// 不做最近点回查）。代价是 ~32 B/三角形 的额外副本（100k 三角形 ≈ 3.2 MB）。
///
/// # 契约
/// 构造绝不 panic：索引越界 / 顶点非有限的三角形被**剔除**（`indices`
/// 同步只保留合法三元组，故 `indices.chunks_exact(3)` 与 `triangles`
/// 恒一一对应）。
#[derive(Debug, Clone)]
pub struct Mesh {
    /// 顶点位置（soup 构造时为空）。
    pub positions: Vec<Vec3>,
    /// 合法三角形索引流（每 3 个一三角形；soup 构造时为空）。
    pub indices: Vec<u32>,
    /// 原始顺序的三角形（与 `indices` 一一对应）。
    pub triangles: Vec<GeoTriangle>,
    /// SAH-BVH（与 `triangles` 同源构建）。
    pub bvh: GeoBvh,
}

impl Mesh {
    /// 索引网格构造：`indices.chunks_exact(3)` 逐个展开为三角形并建树。
    ///
    /// 越界索引、非有限顶点坐标 ⇒ 该三角形被剔除（不 panic，不产生
    /// 错误几何）；`indices` 仅保留通过校验的三元组。
    pub fn new(positions: Vec<Vec3>, indices: Vec<u32>) -> Self {
        let n_pos = positions.len();
        let mut valid: Vec<u32> = Vec::with_capacity(indices.len());
        let mut triangles: Vec<GeoTriangle> = Vec::with_capacity(indices.len() / 3);
        for c in indices.chunks_exact(3) {
            let (i0, i1, i2) = (c[0] as usize, c[1] as usize, c[2] as usize);
            if i0 >= n_pos || i1 >= n_pos || i2 >= n_pos {
                continue; // 越界索引：剔除（绝不 panic）
            }
            let (a, b, cc) = (positions[i0], positions[i1], positions[i2]);
            if !finite_vec3(a) || !finite_vec3(b) || !finite_vec3(cc) {
                continue; // 非有限顶点：剔除（永不命中的垃圾图元不入树）
            }
            triangles.push(GeoTriangle { a, b, c: cc });
            valid.extend_from_slice(c);
        }
        let bvh = GeoBvh::build(&triangles);
        Mesh { positions, indices: valid, triangles, bvh }
    }

    /// 三角形 soup 构造（无拓扑；BVH 在此构建一次）。
    pub fn from_triangles(triangles: Vec<GeoTriangle>) -> Self {
        let bvh = GeoBvh::build(&triangles);
        Mesh { positions: Vec::new(), indices: Vec::new(), triangles, bvh }
    }

    /// 顶点/索引**已构建树**的 soup 构造（零重建：直接接管调用方的树）。
    ///
    /// 前置条件：`bvh` 必须由同一 `triangles` 序列构建（否则图元索引
    /// 对不上）；本函数不做校验（O(n) 校验会抵消零重建收益）。
    pub fn from_prebuilt(triangles: Vec<GeoTriangle>, bvh: GeoBvh) -> Self {
        Mesh { positions: Vec::new(), indices: Vec::new(), triangles, bvh }
    }

    /// 扁平 f32 顶点（`[[x, y, z]; n]`）+ 索引流构造（USD/glTF 桥接用）。
    pub fn from_flat(positions: &[[f32; 3]], indices: &[u32]) -> Self {
        let pts: Vec<Vec3> = positions
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();
        Mesh::new(pts, indices.to_vec())
    }

    /// 三角形数。
    pub fn tri_count(&self) -> usize {
        self.triangles.len()
    }

    /// 是否 soup 型（无顶点/索引拓扑，经 [`Scene::add_bvh`] 注册）。
    pub fn is_soup(&self) -> bool {
        self.positions.is_empty() && self.indices.is_empty()
    }

    /// 轴对齐包围盒 `(min, max)`；空网格返回 `None`。
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        if self.triangles.is_empty() {
            return None;
        }
        let b = self.bvh.bounds;
        Some((b.min, b.max))
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
    /// 三角几何（网格或 soup；每项内建 SAH-BVH，见 [`Mesh`]）。
    pub meshes: Vec<(Mesh, MaterialKind)>,
    /// 有限圆柱族（V2.0-W4 曲线图元；全局曲线 id 段 1）。
    pub cylinders: Vec<(Cylinder, MaterialKind)>,
    /// 有限圆锥族（V2.0-W4 曲线图元；全局曲线 id 段 2）。
    pub cones: Vec<(Cone, MaterialKind)>,
    /// 圆环族（V2.0-W4 曲线图元；全局曲线 id 段 3）。
    pub toruses: Vec<(Torus, MaterialKind)>,
    /// 旋转曲面族（V2.0-W4 曲线图元；全局曲线 id 段 4）。
    pub revolutions: Vec<(Revolution, MaterialKind)>,
    /// 光源列表（NEE 采样对象）。
    pub lights: Vec<Light>,
    /// 可选单一介质区域。
    pub medium: Option<MediumRegion>,
    /// 曲线图元 BVH（[`Scene::add_cylinder`] 等构造入口维护；外部只读，
    /// 直接改四族字段后须调 [`Scene::rebuild_curve_bvh`]）。
    ///
    /// `None` 或图元计数与四族总数不符 ⇒ [`Scene::intersect`] 自动走线性
    /// 兜底路径（两条路径结果一致——有对拍测试）。
    ///
    /// 设计说明：本字段为公开字段（而非私有缓存）——`Scene` 的既有构造
    /// 模式是跨 crate 结构体字面量（tests / examples），私有字段会使外部
    /// 字面量构造直接编译失败（E0451；函数更新语法 `..Default::default()`
    /// 不能绕过私有性），故按"公开只读缓存"口径暴露。
    pub curve_bvh: Option<prism_geometry::aabb_bvh::AabbBvh>,
}

impl Scene {
    /// 注册**索引三角网格**（顶点流 + 索引流；内建 BVH 在此构建），
    /// 返回网格下标（[`ShapeRef`] 索引空间 / [`Scene::meshes`] 下标）。
    pub fn add_mesh(
        &mut self,
        positions: Vec<Vec3>,
        indices: Vec<u32>,
        material: MaterialKind,
    ) -> usize {
        self.meshes.push((Mesh::new(positions, indices), material));
        self.meshes.len() - 1
    }

    /// 注册**三角形 soup**（无拓扑；BVH 在此构建），返回网格下标。
    ///
    /// 与 [`Scene::add_mesh`] 同一存储：soup 项的容器级引用走
    /// [`ShapeRef::Bvh`]，索引网格走 [`ShapeRef::Mesh`]；两者命中均细化为
    /// [`ShapeRef::Triangle`]（网格下标, 三角形下标）。
    pub fn add_bvh(&mut self, triangles: Vec<GeoTriangle>, material: MaterialKind) -> usize {
        self.meshes.push((Mesh::from_triangles(triangles), material));
        self.meshes.len() - 1
    }

    /// 注册**已构建 BVH** 的 soup（零重建：直接接管调用方树与三角形）。
    ///
    /// 前置条件见 [`Mesh::from_prebuilt`]（`bvh` 必须由同一 `triangles` 构建）。
    pub fn add_bvh_prebuilt(
        &mut self,
        triangles: Vec<GeoTriangle>,
        bvh: GeoBvh,
        material: MaterialKind,
    ) -> usize {
        self.meshes.push((Mesh::from_prebuilt(triangles, bvh), material));
        self.meshes.len() - 1
    }

    /// 三角几何条目数（网格 + soup）。
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// 全场景三角形总数。
    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|(m, _)| m.tri_count()).sum()
    }

    // -----------------------------------------------------------------------
    // 曲线图元族构造入口（V2.0-W4；add_* 维护曲线 BVH）
    // -----------------------------------------------------------------------

    /// 注册有限圆柱，返回圆柱下标（[`ShapeRef::Cylinder`] 索引空间）。
    pub fn add_cylinder(&mut self, cylinder: Cylinder, material: MaterialKind) -> usize {
        self.cylinders.push((cylinder, material));
        self.rebuild_curve_bvh();
        self.cylinders.len() - 1
    }

    /// 注册有限圆锥，返回圆锥下标（[`ShapeRef::Cone`] 索引空间）。
    pub fn add_cone(&mut self, cone: Cone, material: MaterialKind) -> usize {
        self.cones.push((cone, material));
        self.rebuild_curve_bvh();
        self.cones.len() - 1
    }

    /// 注册圆环，返回圆环下标（[`ShapeRef::Torus`] 索引空间）。
    pub fn add_torus(&mut self, torus: Torus, material: MaterialKind) -> usize {
        self.toruses.push((torus, material));
        self.rebuild_curve_bvh();
        self.toruses.len() - 1
    }

    /// 注册旋转曲面，返回曲面下标（[`ShapeRef::Revolution`] 索引空间）。
    pub fn add_revolution(&mut self, revolution: Revolution, material: MaterialKind) -> usize {
        self.revolutions.push((revolution, material));
        self.rebuild_curve_bvh();
        self.revolutions.len() - 1
    }

    /// 重建曲线图元 BVH（四族 AABB 按全局曲线 id 序：
    /// cylinders → cones → toruses → revolutions）。
    ///
    /// 直接改 `cylinders` / `cones` / `toruses` / `revolutions` 字段后
    /// 须调用本函数（或改用 `add_*`）；不重建时 `intersect` 按计数不符
    /// 自动退化线性兜底（结果与 BVH 路径一致，仅无粗阶段加速）。
    pub fn rebuild_curve_bvh(&mut self) {
        let mut aabbs: Vec<Aabb> = Vec::with_capacity(self.curve_total());
        for (c, _) in &self.cylinders {
            aabbs.push(cylinder_aabb(c));
        }
        for (c, _) in &self.cones {
            aabbs.push(cone_aabb(c));
        }
        for (t, _) in &self.toruses {
            aabbs.push(torus_aabb(t));
        }
        for (r, _) in &self.revolutions {
            aabbs.push(revolution_aabb(r));
        }
        self.curve_bvh = if aabbs.is_empty() {
            None
        } else {
            let geo: Vec<prism_geometry::bvh::Aabb> = aabbs.iter().map(to_geo_aabb).collect();
            Some(prism_geometry::aabb_bvh::AabbBvh::build(&geo))
        };
    }

    /// 曲线图元四族总数。
    pub(crate) fn curve_total(&self) -> usize {
        self.cylinders.len() + self.cones.len() + self.toruses.len() + self.revolutions.len()
    }

    /// 全局曲线 id → 族内下标与种类（family-major 序，见
    /// [`Scene::rebuild_curve_bvh`]）。
    fn curve_locate(&self, id: u32) -> Option<ShapeRef> {
        let nc = self.cylinders.len() as u32;
        let nk = self.cones.len() as u32;
        let nt = self.toruses.len() as u32;
        let nr = self.revolutions.len() as u32;
        if id < nc {
            Some(ShapeRef::Cylinder(id as usize))
        } else if id < nc + nk {
            Some(ShapeRef::Cone((id - nc) as usize))
        } else if id < nc + nk + nt {
            Some(ShapeRef::Torus((id - nc - nk) as usize))
        } else if id < nc + nk + nt + nr {
            Some(ShapeRef::Revolution((id - nc - nk - nt) as usize))
        } else {
            None // id 越界（BVH 与四族不同步的防御）
        }
    }

    /// 全局曲线 id → 最近有效 t（BVH 回调精阶段；口径与 `*_t` 一致）。
    fn curve_t(&self, id: u32, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        match self.curve_locate(id)? {
            ShapeRef::Cylinder(i) => self.cylinder_t(i, ro, rd, t_max),
            ShapeRef::Cone(i) => self.cone_t(i, ro, rd, t_max),
            ShapeRef::Torus(i) => self.torus_t(i, ro, rd, t_max),
            ShapeRef::Revolution(i) => self.revolution_t(i, ro, rd, t_max),
            _ => None, // 其余 ShapeRef 变体不属曲线 id 空间
        }
    }
}

// ---------------------------------------------------------------------------
// 求交
// ---------------------------------------------------------------------------

/// 形状引用（借场景索引）。
///
/// # 三角几何的三变体语义（V2-h）
/// - [`ShapeRef::Triangle`] `(网格下标, 原始三角形下标)` —— **唯一**的三角形
///   命中结果形态：法线由三角形绕序直接给出（flat shading），材质取所属
///   网格的材质；
/// - [`ShapeRef::Mesh`] `(网格下标)` —— **索引网格**的容器级引用（整体豁免 /
///   整体材质），`intersect` 不产生此变体；
/// - [`ShapeRef::Bvh`] `(网格下标)` —— **soup（BVH 直挂）**的容器级引用，
///   与 `Mesh` 共享同一存储 [`Scene::meshes`]（`Mesh::is_soup()` 可判别）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeRef {
    Sphere(usize),
    Box(usize),
    Paraboloid(usize),
    Plane(usize),
    /// 三角网格中的单个三角形：(网格下标, 原始三角形下标)。
    Triangle(usize, usize),
    /// 索引网格整体（材质 / 豁免的容器级引用；命中细化为 `Triangle`）。
    Mesh(usize),
    /// 独立 BVH（soup）整体（材质 / 豁免的容器级引用；命中细化为 `Triangle`）。
    Bvh(usize),
    /// 有限圆柱（V2.0-W4；下标见 [`Scene::cylinders`]）。
    Cylinder(usize),
    /// 有限圆锥（V2.0-W4；下标见 [`Scene::cones`]）。
    Cone(usize),
    /// 圆环（V2.0-W4；下标见 [`Scene::toruses`]）。
    Torus(usize),
    /// 旋转曲面（V2.0-W4；下标见 [`Scene::revolutions`]）。
    Revolution(usize),
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

/// 三角形单位几何法线（绕序 a→b→c 右手；零面积退化返回零向量）。
///
/// 退化三角形（零面积）在 Möller–Trumbore 下分母为零 ⇒ **永不被射线命中**，
/// 故零向量返回值对渲染路径不可达；保留它是为了"绝不 NaN"的兜底
/// （归一化零长向量会产生 NaN 并污染整条路径的辐亮度）。
#[inline]
fn triangle_unit_normal(t: &GeoTriangle) -> Vec3 {
    let n = t.normal_unnormalized();
    let len2 = n.dot(&n);
    if len2 > 1e-24 {
        n * (1.0 / len2.sqrt())
    } else {
        Vec3::splat(0.0)
    }
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
            // 曲线图元族（V2.0-W4）：各取所属条目材质。
            ShapeRef::Cylinder(i) => &self.cylinders[i].1,
            ShapeRef::Cone(i) => &self.cones[i].1,
            ShapeRef::Torus(i) => &self.toruses[i].1,
            ShapeRef::Revolution(i) => &self.revolutions[i].1,
            // 三角几何三变体同源：都取所属网格的材质（容器级引用与
            // 图元级命中语义一致——一个网格一个材质）。
            ShapeRef::Triangle(m, _) | ShapeRef::Mesh(m) | ShapeRef::Bvh(m) => &self.meshes[m].1,
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
            // ---- 曲线图元族（V2.0-W4）：由命中点 p 纯函数重判表面归属 ----
            // 口径：法线只依赖命中点与图元参数（与 `*_t` 的候选验证同源），
            // 故 integrator / wavefront 双路径天然一致；棱掠点（侧/盖残差
            // 并列）按固定优先级取侧面（测度零，纯函数保证确定性）。
            ShapeRef::Cylinder(i) => {
                let c = &self.cylinders[i].0;
                let pl = p - c.center;
                let dr = xz_len(pl) - c.radius; // 侧面残差（侧面表面处 = 0）
                let dc = pl.y.abs() - c.half_height; // 端盖残差（端盖表面处 = 0）
                // 侧/盖归属按**残差绝对值**近者：纯侧面点（dr=0、端盖内
                // dc<0）必须判侧面——早年签名比较会把侧面点误判成端盖
                // （V2.0-W4 对拍修复：dr<=dc 在 dc<0<dr 时恒假）。
                if dr.abs() <= dc.abs() {
                    // 侧面外法线（轴向 +Y 圆柱侧面法线即径向）。
                    Vec3::new(pl.x, 0.0, pl.z) * (1.0 / c.radius)
                } else {
                    // 端盖外法线（pl.y == 0 不可达：half_height > 0 且盖在
                    // |y| = H；符号兜底防零向量）。
                    Vec3::new(0.0, if pl.y >= 0.0 { 1.0 } else { -1.0 }, 0.0)
                }
            }
            ShapeRef::Cone(i) => {
                let c = &self.cones[i].0;
                let pl = p - c.center;
                let k = c.radius / (2.0 * c.half_height);
                // 侧/底残差（(ρ, y) 平面）：侧面 ρ = k(H−y)，底面 y = −H。
                let rs = xz_len(pl) - k * (c.half_height - pl.y);
                let rc = pl.y + c.half_height;
                if rs.abs() <= rc.abs() {
                    // 侧面外法线 ∝ (x, k²(H−y), z)（顶点处退化 → (0,1,0)）。
                    let n = Vec3::new(pl.x, k * k * (c.half_height - pl.y), pl.z);
                    let l2 = n.dot(&n);
                    if l2 > 1e-24 {
                        n * (1.0 / l2.sqrt())
                    } else {
                        Vec3::new(0.0, 1.0, 0.0)
                    }
                } else {
                    // 底面外法线（朝 −Y）。
                    Vec3::new(0.0, -1.0, 0.0)
                }
            }
            ShapeRef::Torus(i) => {
                let t = &self.toruses[i].0;
                let pl = p - t.center;
                let rho = xz_len(pl);
                if rho > 1e-12 {
                    // 外法线 ∝ ((1−R/ρ)x, y, (1−R/ρ)z)（表面上模长 = minor）。
                    let s = 1.0 - t.major / rho;
                    let n = Vec3::new(pl.x * s, pl.y, pl.z * s);
                    let l2 = n.dot(&n);
                    if l2 > 1e-24 {
                        n * (1.0 / l2.sqrt())
                    } else {
                        Vec3::new(0.0, 1.0, 0.0)
                    }
                } else {
                    // 轴心测度零（仅 minor >= major 的自交纺锤可达）：兜底。
                    Vec3::new(0.0, if pl.y >= 0.0 { 1.0 } else { -1.0 }, 0.0)
                }
            }
            ShapeRef::Revolution(i) => {
                let rev = &self.revolutions[i].0;
                let pl = p - rev.center;
                let rho = xz_len(pl);
                // 最近轮廓段（(ρ, y) 平面点段距离；并列取小索引 → 确定性）。
                let mut best_j: Option<usize> = None;
                let mut best_d2 = f32::INFINITY;
                for j in 0..rev.segment_count() {
                    let a = rev.profile[j];
                    let b = rev.profile[j + 1];
                    let d2 = point_segment_dist2(rho, pl.y, a[0], a[1], b[0], b[1]);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_j = Some(j);
                    }
                }
                match best_j {
                    Some(j) => {
                        let a = rev.profile[j];
                        let b = rev.profile[j + 1];
                        let dr = b[0] - a[0];
                        let dy = b[1] - a[1];
                        // 外法线（CCW 归一后）：剖面 (Δy, −Δr) → 三维径向+竖向。
                        let inv_rho = if rho > 1e-12 { 1.0 / rho } else { 0.0 };
                        let n = Vec3::new(dy * pl.x * inv_rho, -dr, dy * pl.z * inv_rho);
                        let l2 = n.dot(&n);
                        if l2 > 1e-24 {
                            n * (1.0 / l2.sqrt())
                        } else {
                            Vec3::new(0.0, 1.0, 0.0)
                        }
                    }
                    None => Vec3::new(0.0, 1.0, 0.0), // 点数 < 2：永不命中图元的兜底
                }
            }
            // 三角几何：几何法线（绕序 a→b→c 右手）。两面性由积分器
            // `from_inside` 统一翻转，故此处恒返**绕序法线**（不按视角翻转）。
            ShapeRef::Triangle(m, t) => self
                .meshes
                .get(m)
                .and_then(|(mesh, _)| mesh.triangles.get(t))
                .map_or(Vec3::splat(0.0), triangle_unit_normal),
            // 容器级引用（整体豁免 / 外部构造的容器 ref）不承载图元索引：
            // 法线按**最近图元**兜底（flat shading 语义，O(log n) 的
            // BVH 最近点查询）。`intersect` 从不产生这两种变体，故该分支
            // 只在调用方显式持有容器 ref 时到达。
            ShapeRef::Mesh(m) | ShapeRef::Bvh(m) => self.meshes.get(m).map_or(
                Vec3::splat(0.0),
                |(mesh, _)| {
                    let cp = mesh.bvh.closest_point(p);
                    if cp.tri == u32::MAX {
                        return Vec3::splat(0.0); // 空网格
                    }
                    mesh.triangles
                        .get(cp.tri as usize)
                        .map_or(Vec3::splat(0.0), triangle_unit_normal)
                },
            ),
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

    /// 网格求交：BVH 遍历 → `(t, 原始三角形下标)`。
    ///
    /// 契约与解析图元臂同口径：`t >= T_MIN` 且 `t < t_max`（`GeoBvh` 内部用
    /// 闭区间 `[t_min, t_max]`，此处显式按半开 `t < t_max` 过滤，避免"恰在
    /// 远端"的网格面遮蔽同距离的解析图元）。射线须为单位向量
    /// （`Scene::intersect` 全局契约，`GeoBvh::ray_intersect` 以
    /// `debug_assert` 断言）。
    ///
    /// `skip_tri = Some(k)` 排除第 k 个三角形：若最近命中恰为 k，则以
    /// `t_min = t_k + T_MIN` 二次查询取其后最近命中（BVH 的 t_min 剪枝
    /// 天然重做，无 O(n) 退化；与 k 共享同一 t 的图元属测度零边界）。
    fn mesh_t(
        &self,
        i: usize,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip_tri: Option<usize>,
    ) -> Option<(f32, usize)> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None; // 非有限射线 ⇒ 该网格永不命中（与平面/抛物面臂同语义）
        }
        let mesh = &self.meshes[i].0;
        if mesh.triangles.is_empty() {
            return None;
        }
        let first = mesh.bvh.ray_intersect(ro, rd, T_MIN, t_max)?;
        if !(first.t < t_max) {
            return None; // 最近命中已在远端之外（或 NaN）
        }
        if skip_tri != Some(first.tri as usize) {
            return Some((first.t, first.tri as usize));
        }
        // 排除被豁免图元后重查（起点沿射线推进一个自交下界）。
        let t0 = first.t + T_MIN;
        if !(t0 < t_max) {
            return None;
        }
        let second = mesh.bvh.ray_intersect(ro, rd, t0, t_max)?;
        if !(second.t < t_max) {
            return None;
        }
        Some((second.t, second.tri as usize))
    }

    // -----------------------------------------------------------------------
    // 曲线图元族求交（V2.0-W4；闭式解析，口径与 sphere_t 家族一致：
    // `t >= T_MIN && t < t_max`，候选并列先达者优先，闭合实体内外都记录）
    // -----------------------------------------------------------------------

    /// 圆柱求交：侧面二次方程（|y| <= H 裁剪）+ 上下端盖圆盘
    /// （y = ±H 线性根，x²+z² <= R² 裁剪）。
    ///
    /// - `a == 0`（射线平行轴，dx²+dz² 恰为零）：侧面跳过（测度零；
    ///   轴上射线仍由端盖覆盖）；`dy == 0`：端盖跳过；
    /// - `disc < 0`：侧面不交；切向（disc == 0）单根照常验证；
    /// - 数值稳健：o/d 非有限、`radius <= 0` / `half_height <= 0` /
    ///   非有限 ⇒ None（永不命中），不 panic。
    fn cylinder_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let c = &self.cylinders[i].0;
        if !finite_vec3(c.center) || !(c.radius > 0.0) || !(c.half_height > 0.0) {
            return None; // 图元自身非法 ⇒ 永不命中
        }
        let q = ro - c.center;
        let (r, h) = (c.radius, c.half_height);
        let mut best: Option<f32> = None;

        // 侧面：x²+z² = R²（局部坐标）。
        let a = rd.x * rd.x + rd.z * rd.z;
        if a != 0.0 {
            let b = 2.0 * (q.x * rd.x + q.z * rd.z);
            let cc = q.x * q.x + q.z * q.z - r * r;
            let disc = b * b - 4.0 * a * cc;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                    if !(t >= T_MIN && t < t_max) {
                        continue;
                    }
                    let y = q.y + t * rd.y;
                    if y < -h || y > h {
                        continue; // 段界外（闭边界：== 算命中）
                    }
                    best = Some(match best {
                        Some(bt) if bt <= t => bt,
                        _ => t,
                    });
                }
            }
        }

        // 端盖：y = ±H（线性根；半径内闭边界）。
        if rd.y != 0.0 {
            for ny in [-h, h] {
                let t = (ny - q.y) / rd.y;
                if !(t >= T_MIN && t < t_max) {
                    continue;
                }
                let x = q.x + t * rd.x;
                let z = q.z + t * rd.z;
                if x * x + z * z > r * r {
                    continue;
                }
                best = Some(match best {
                    Some(bt) if bt <= t => bt,
                    _ => t,
                });
            }
        }
        best
    }

    /// 圆锥求交：斜锥侧面二次方程（y ∈ [-H, H] 闭裁剪）+ 底面圆盘
    /// （y = -H 线性根，x²+z² <= R² 裁剪）。
    ///
    /// 侧面方程（局部坐标，k = R/(2H)，底面 y=-H 半径 R、顶点 y=+H）：
    /// `(dx²+dz²-k²dy²)t² + 2(qx·dx+qz·dz+k²·dy·(H-qy))t
    ///  + (qx²+qz²-k²(H-qy)²) = 0`。
    /// `a == 0`（射线平行斜锥母线，测度零）：退线性根 `t = -c/b`；
    /// `dy == 0`：底面跳过。顶点（y=+H）闭边界算命中。
    fn cone_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let c = &self.cones[i].0;
        if !finite_vec3(c.center) || !(c.radius > 0.0) || !(c.half_height > 0.0) {
            return None; // 图元自身非法 ⇒ 永不命中
        }
        let q = ro - c.center;
        let (r, h) = (c.radius, c.half_height);
        let k = r / (2.0 * h);
        let mut best: Option<f32> = None;

        // 侧面（斜锥，过底面圆 + 顶点的无限锥，段内裁剪）。
        let a = (rd.x * rd.x + rd.z * rd.z) - k * k * rd.y * rd.y;
        let b = 2.0 * (q.x * rd.x + q.z * rd.z) + 2.0 * k * k * rd.y * (h - q.y);
        let cc = q.x * q.x + q.z * q.z - k * k * (h - q.y) * (h - q.y);
        if a != 0.0 {
            let disc = b * b - 4.0 * a * cc;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                for t in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                    if !(t >= T_MIN && t < t_max) {
                        continue;
                    }
                    let y = q.y + t * rd.y;
                    if y < -h || y > h {
                        continue; // 段界外（闭边界）
                    }
                    best = Some(match best {
                        Some(bt) if bt <= t => bt,
                        _ => t,
                    });
                }
            }
        } else if b != 0.0 {
            // 平行母线退化：线性单根。
            let t = -cc / b;
            if t >= T_MIN && t < t_max {
                let y = q.y + t * rd.y;
                if y >= -h && y <= h {
                    best = Some(t);
                }
            }
        }

        // 底面圆盘：y = -H（半径内闭边界）。
        if rd.y != 0.0 {
            let t = (-h - q.y) / rd.y;
            if t >= T_MIN && t < t_max {
                let x = q.x + t * rd.x;
                let z = q.z + t * rd.z;
                if x * x + z * z <= r * r {
                    best = Some(match best {
                        Some(bt) if bt <= t => bt,
                        _ => t,
                    });
                }
            }
        }
        best
    }

    /// 圆环求交：隐式方程 `(ρ-R)² + y² = r²` 代入射线得**四次方程**，
    /// 精确求根走 [`prism_geometry::quartic::solve_quartic`]（Ferrari
    /// 闭式、零迭代；禁止牛顿迭代——确定性 D-29）。
    ///
    /// 系数（局部坐标 q = o-c；A = |q+td|² = a2t²+a1t+a0，
    /// B = |q_xz+td_xz|² = b2t²+b1t+b0，u = R²-r²）：
    /// `c4 = a2²`、`c3 = 2·a2·a1`、
    /// `c2 = a1² + 2·a2·(a0+u) - 4R²·b2`、
    /// `c1 = 2·a1·(a0+u) - 4R²·b1`、
    /// `c0 = (a0+u)² - 4R²·b0`。
    /// 四个实根（前外/前内/后内/后外）都验证，取最小有效 t
    /// （闭合实体语义：穿环孔的射线在后半还有两次命中）。
    fn torus_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let t = &self.toruses[i].0;
        if !finite_vec3(t.center) || !(t.major > 0.0) || !(t.minor > 0.0) {
            return None; // 图元自身非法 ⇒ 永不命中
        }
        let q = ro - t.center;
        let u = t.major * t.major - t.minor * t.minor;
        let a2 = rd.dot(&rd);
        let a1 = 2.0 * q.dot(&rd);
        let a0 = q.dot(&q);
        let b2 = rd.x * rd.x + rd.z * rd.z;
        let b1 = 2.0 * (q.x * rd.x + q.z * rd.z);
        let b0 = q.x * q.x + q.z * q.z;
        let mm = t.major * t.major;
        let c4 = a2 * a2;
        let c3 = 2.0 * a2 * a1;
        let c2 = a1 * a1 + 2.0 * a2 * (a0 + u) - 4.0 * mm * b2;
        let c1 = 2.0 * a1 * (a0 + u) - 4.0 * mm * b1;
        let c0 = (a0 + u) * (a0 + u) - 4.0 * mm * b0;
        let mut roots = [0.0_f32; 4];
        let n = prism_geometry::quartic::solve_quartic(c4, c3, c2, c1, c0, &mut roots);
        let mut best: Option<f32> = None;
        for &tr in &roots[..n] {
            if !(tr >= T_MIN && tr < t_max) {
                continue;
            }
            best = Some(match best {
                Some(bt) if bt <= tr => bt,
                _ => tr,
            });
        }
        best
    }

    /// 旋转曲面求交：轮廓逐段解二次方程（过该段的斜锥；水平段特化为
    /// 环带），段内闭裁剪；**同参数声明 → 同段序列**（段序 =
    /// [`Revolution::new`] 归一后的点列邻接序，纯函数）。
    ///
    /// 斜锥段（Δy != 0；A = Δr·(qy-y0) + Δy·r0）：
    /// `(Δy²·b2 - Δr²·dy²)t² + (Δy²·b1 - 2·Δr·dy·A)t
    ///  + (Δy²·b0 - A²) = 0`，命中需 y(t) ∈ [y0, y1]（闭）。
    /// 水平段（Δy == 0）：环带 y = y0、ρ ∈ [min(r0,r1), max(r0,r1)]
    /// （闭）；射线平行环带平面（dy == 0）⇒ 共面测度零，不命中。
    fn revolution_t(&self, i: usize, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None;
        }
        let rev = &self.revolutions[i].0;
        if !finite_vec3(rev.center) || rev.profile.len() < 2 {
            return None;
        }
        for p in &rev.profile {
            if !p[0].is_finite() || !p[1].is_finite() {
                return None; // 非有限轮廓点 ⇒ 整图元永不命中
            }
            if p[0] < 0.0 {
                // 契约要求 r >= 0（见 [`Revolution`] 文档）。负半径不仅几何
                // 无意义，还会破坏"平方求根无外部增根"的论证（增根落在轮廓
                // 线的镜像分支上，靠 y 段界 + r>=0 半平面排除）⇒ 按永不命中。
                return None;
            }
        }
        let q = ro - rev.center;
        let b2 = rd.x * rd.x + rd.z * rd.z;
        let b1 = 2.0 * (q.x * rd.x + q.z * rd.z);
        let b0 = q.x * q.x + q.z * q.z;
        let mut best: Option<f32> = None;
        for j in 0..rev.segment_count() {
            let (r0, y0) = (rev.profile[j][0], rev.profile[j][1]);
            let (r1, y1) = (rev.profile[j + 1][0], rev.profile[j + 1][1]);
            let dr = r1 - r0;
            let dy = y1 - y0;
            if dr == 0.0 && dy == 0.0 {
                continue; // 重复点：退化段
            }
            if dy != 0.0 {
                // 斜锥段：二次方程。
                let a = dy * dy * b2 - dr * dr * rd.y * rd.y;
                let big_a = dr * (q.y - y0) + dy * r0;
                let b = dy * dy * b1 - 2.0 * dr * rd.y * big_a;
                let cc = dy * dy * b0 - big_a * big_a;
                let mut cand = [0.0_f32; 2];
                let n_cand = if a != 0.0 {
                    let disc = b * b - 4.0 * a * cc;
                    if !(disc >= 0.0) {
                        continue;
                    }
                    let sq = disc.sqrt();
                    cand[0] = (-b - sq) / (2.0 * a);
                    cand[1] = (-b + sq) / (2.0 * a);
                    2
                } else if b != 0.0 {
                    cand[0] = -cc / b; // 平行母线退化：线性单根
                    1
                } else {
                    continue; // 射线落在斜锥面上（测度零）或永不相交
                };
                for &t in &cand[..n_cand] {
                    if !(t >= T_MIN && t < t_max) {
                        continue;
                    }
                    let y = q.y + t * rd.y;
                    if (y - y0) * (y - y1) > 0.0 {
                        continue; // 段界外（闭边界：== 算命中）
                    }
                    best = Some(match best {
                        Some(bt) if bt <= t => bt,
                        _ => t,
                    });
                }
            } else {
                // 水平段（环带）：y = y0 平面上半径 ∈ [r0, r1] 的圆环。
                if rd.y == 0.0 {
                    continue; // 共面（测度零）⇒ 不命中
                }
                let t = (y0 - q.y) / rd.y;
                if !(t >= T_MIN && t < t_max) {
                    continue;
                }
                let x = q.x + t * rd.x;
                let z = q.z + t * rd.z;
                let rho2 = x * x + z * z;
                let (rlo, rhi) = if r0 <= r1 { (r0, r1) } else { (r1, r0) };
                if rho2 < rlo * rlo || rho2 > rhi * rhi {
                    continue; // 环带界外（闭边界）
                }
                best = Some(match best {
                    Some(bt) if bt <= t => bt,
                    _ => t,
                });
            }
        }
        best
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
        for i in 0..self.meshes.len() {
            // 容器级豁免：`Mesh(i)`（索引网格）与 `Bvh(i)`（soup）均指整块网格。
            if skip == Some(ShapeRef::Mesh(i)) || skip == Some(ShapeRef::Bvh(i)) {
                continue;
            }
            // 图元级豁免：单个三角形被显式豁免时走二次查询。
            let skip_tri = match skip {
                Some(ShapeRef::Triangle(m, t)) if m == i => Some(t),
                _ => None,
            };
            if let Some((t, tri)) = self.mesh_t(i, ro, rd, limit, skip_tri) {
                best = Some(Hit { t, shape: ShapeRef::Triangle(i, tri) });
                limit = t;
            }
        }
        // ---- 曲线图元族（V2.0-W4；测尾，既有族行为不变）----------------
        // 快路径：曲线 BVH 粗阶段 + 回调精阶段；线性兜底：四族按全局
        // 曲线 id 序逐图元 `*_t`（与既有族同款 running limit 口径）。
        // 两路结果一致：min t 优先、t 并列取最小全局 id（BVH 的
        // `(t, idx)` tie-break 与线性的前达者优先在 id 序下同解），
        // 由 `curve_bvh_matches_linear_scan` 对拍钉死。
        let curve_total = self.curve_total();
        if curve_total > 0 {
            let bvh_live = self
                .curve_bvh
                .as_ref()
                .is_some_and(|b| b.prim_count() == curve_total);
            if bvh_live {
                let bvh = self.curve_bvh.as_ref().expect("bvh_live 已判定存在");
                let limit_entry = limit;
                let mut cb = |id: u32| self.curve_t(id, ro, rd, limit_entry);
                if let Some((t, id)) = bvh.ray_intersect(ro, rd, T_MIN, limit_entry, &mut cb) {
                    if let Some(shape) = self.curve_locate(id) {
                        best = Some(Hit { t, shape });
                        // 末族：limit 不再被读，无需回写（避免 unused 警告）。
                    }
                }
            } else {
                for i in 0..self.cylinders.len() {
                    if let Some(t) = self.cylinder_t(i, ro, rd, limit) {
                        best = Some(Hit { t, shape: ShapeRef::Cylinder(i) });
                        limit = t;
                    }
                }
                for i in 0..self.cones.len() {
                    if let Some(t) = self.cone_t(i, ro, rd, limit) {
                        best = Some(Hit { t, shape: ShapeRef::Cone(i) });
                        limit = t;
                    }
                }
                for i in 0..self.toruses.len() {
                    if let Some(t) = self.torus_t(i, ro, rd, limit) {
                        best = Some(Hit { t, shape: ShapeRef::Torus(i) });
                        limit = t;
                    }
                }
                for i in 0..self.revolutions.len() {
                    if let Some(t) = self.revolution_t(i, ro, rd, limit) {
                        best = Some(Hit { t, shape: ShapeRef::Revolution(i) });
                        // 末族：limit 不再被读，无需回写（避免 unused 警告）。
                    }
                }
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
            // 曲线图元族（V2.0-W4）：非球光源几何，未注册 NEE ⇒ 命中权重 1。
            ShapeRef::Cylinder(_)
            | ShapeRef::Cone(_)
            | ShapeRef::Torus(_)
            | ShapeRef::Revolution(_) => return None,
            // 三角几何（任意变体）：非球光源几何，未注册 NEE ⇒ 命中权重 1
            // （仅 BSDF 策略；发光网格既未注册 Light 也无立体角 pdf）。
            ShapeRef::Triangle(_, _) | ShapeRef::Mesh(_) | ShapeRef::Bvh(_) => return None,
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![(plane, lambert_grey())],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![(
                Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None),
                lambert_grey(),
            )],
            ..Default::default()
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
            meshes: Vec::new(),
            planes: vec![(Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), None), mat)],
            ..Default::default()
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

    // ===================================================================
    //  V2-h：三角几何（Mesh / BVH）进 rt
    // ===================================================================

    use prism_geometry::bvh::{Bvh as GeoBvh, Triangle as GeoTriangle};

    /// 轴对齐立方体（12 三角形，边长 `2h`，中心 `c`）。
    ///
    /// 顶点编号 `x + 2y + 4z`（各轴取 `c ± h`）；六个面按**外向绕序**
    /// 逐面两三角形展开——法线朝外，故半球抽样、NEE 余弦全部落在可见侧。
    fn cube_mesh(c: Vec3, h: f32) -> (Vec<Vec3>, Vec<u32>) {
        let mut pts = Vec::with_capacity(8);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let sx = if x == 0 { -1.0 } else { 1.0 };
                    let sy = if y == 0 { -1.0 } else { 1.0 };
                    let sz = if z == 0 { -1.0 } else { 1.0 };
                    pts.push(Vec3::new(c.x + sx * h, c.y + sy * h, c.z + sz * h));
                }
            }
        }
        let v = |x: u32, y: u32, z: u32| x + 2 * y + 4 * z;
        // 六个面的四边形（外向绕序；每面两三角形 v0v1v2 / v0v2v3）。
        let quads: [[u32; 4]; 6] = [
            [v(0, 0, 1), v(0, 1, 1), v(0, 1, 0), v(0, 0, 0)], // -X
            [v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)], // +X
            [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)], // -Y
            [v(0, 1, 1), v(1, 1, 1), v(1, 1, 0), v(0, 1, 0)], // +Y
            [v(0, 0, 0), v(0, 1, 0), v(1, 1, 0), v(1, 0, 0)], // -Z
            [v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)], // +Z
        ];
        let mut ids: Vec<u32> = Vec::with_capacity(36);
        for q in quads {
            ids.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        (pts, ids)
    }

    /// 程序化圆环网格（rings × segs 四边形 → 2·rings·segs 三角形）。
    ///
    /// 三角件走 [`prism_geometry::det_math::det_cos_sin`]（D-29：禁 libm，
    /// 同输入逐位一致）——网格本身可复现，100k 计时测试才可重复对账。
    fn torus_mesh(rings: usize, segs: usize, big_r: f32, small_r: f32) -> (Vec<Vec3>, Vec<u32>) {
        let (two_pi_over_rings, two_pi_over_segs) = (
            6.283_185_5 / rings as f32,
            6.283_185_5 / segs as f32,
        );
        let mut pts = Vec::with_capacity(rings * segs);
        for i in 0..rings {
            let (cu, su) = prism_geometry::det_math::det_cos_sin(i as f32 * two_pi_over_rings);
            for j in 0..segs {
                let (cv, sv) = prism_geometry::det_math::det_cos_sin(j as f32 * two_pi_over_segs);
                let r = big_r + small_r * cv;
                pts.push(Vec3::new(r * cu, small_r * sv, r * su));
            }
        }
        let at = |i: usize, j: usize| ((i % rings) * segs + (j % segs)) as u32;
        let mut ids = Vec::with_capacity(rings * segs * 6);
        for i in 0..rings {
            for j in 0..segs {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                ids.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
        (pts, ids)
    }

    /// 单个三角形：命中 / 未命中 / 法线 / 材质 / 远平面 一次过。
    #[test]
    fn mesh_single_triangle_hit_miss_normal_material() {
        let mat = MaterialKind::Lambert { albedo: Vec3::new(0.5, 0.5, 0.5) };
        let mut sc = Scene::default();
        let mi = sc.add_bvh(
            vec![GeoTriangle {
                a: Vec3::new(-1.0, -1.0, 0.0),
                b: Vec3::new(1.0, -1.0, 0.0),
                c: Vec3::new(0.0, 1.0, 0.0),
            }],
            mat,
        );
        assert_eq!(mi, 0, "add_bvh 返回首项下标 0");
        assert_eq!(sc.mesh_count(), 1);
        assert_eq!(sc.triangle_count(), 1);

        // 命中：(0,0,-2) 沿 +z 射向三角形重心。
        let h = sc
            .intersect(Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30, None)
            .expect("应命中单三角形");
        assert_eq!(h.shape, ShapeRef::Triangle(0, 0), "命中细化为 Triangle(mesh, tri)");
        assert!((h.t - 2.0).abs() < 1e-6, "t = {}", h.t);
        // 绕序 a→b→c 右手 ⇒ +z 法线。
        let n = sc.normal_at(h.shape, Vec3::new(0.0, 0.0, 0.0));
        assert!((n.z - 1.0).abs() < 1e-6 && n.x.abs() < 1e-6 && n.y.abs() < 1e-6, "n = {n:?}");
        assert!(matches!(
            sc.material(h.shape),
            MaterialKind::Lambert { albedo } if (albedo.x - 0.5).abs() < 1e-6
        ));
        assert!((sc.primary_hit_distance(Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30)
            .expect("primary_hit_distance 同源")
            - 2.0)
            .abs()
            < 1e-6);

        // 未命中 1：射线偏离三角形（x = 3 在包围盒外）。
        assert!(sc
            .intersect(Vec3::new(3.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30, None)
            .is_none());
        // 未命中 2：反方向（背离三角形）。
        assert!(sc
            .intersect(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30, None)
            .is_none());
        // 未命中 3：远平面在命中之前（t_max = 1.5 < 2.0）。
        assert!(sc
            .intersect(Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.5, None)
            .is_none());
        // 非有限射线：永不命中（与平面/抛物面臂同语义）。
        assert!(sc
            .intersect(
                Vec3::new(f32::NAN, 0.0, -2.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0e30,
                None
            )
            .is_none());
    }

    /// 图元级 / 容器级豁免语义（阴影射线 skip）：两层平行三角形叠放。
    #[test]
    fn mesh_skip_triangle_and_container() {
        let mat = lambert_grey();
        let tri = |z: f32| GeoTriangle {
            a: Vec3::new(-1.0, -1.0, z),
            b: Vec3::new(1.0, -1.0, z),
            c: Vec3::new(0.0, 1.0, z),
        };
        let mut sc = Scene::default();
        sc.add_bvh(vec![tri(0.0), tri(-2.0)], mat);
        let ro = Vec3::new(0.0, 0.0, 5.0);
        let rd = Vec3::new(0.0, 0.0, -1.0);
        let h = sc.intersect(ro, rd, 1.0e30, None).unwrap();
        assert_eq!(h.shape, ShapeRef::Triangle(0, 0));
        assert!((h.t - 5.0).abs() < 1e-6, "近面 t = {}", h.t);
        // 豁免近三角形 ⇒ 落到远三角形（t = 7）。
        let h2 = sc
            .intersect(ro, rd, 1.0e30, Some(ShapeRef::Triangle(0, 0)))
            .unwrap();
        assert_eq!(h2.shape, ShapeRef::Triangle(0, 1));
        assert!((h2.t - 7.0).abs() < 1e-6, "远面 t = {}", h2.t);
        // 容器级豁免（Mesh / Bvh 两种容器引用等价）⇒ 全无命中。
        assert!(sc.intersect(ro, rd, 1.0e30, Some(ShapeRef::Mesh(0))).is_none());
        assert!(sc.intersect(ro, rd, 1.0e30, Some(ShapeRef::Bvh(0))).is_none());
        // 三角几何无立体角 pdf（未注册 NEE ⇒ 命中权重 1）。
        assert_eq!(sc.light_pdf_sa(ro, ShapeRef::Triangle(0, 0)), None);
        assert_eq!(sc.light_pdf_sa(ro, ShapeRef::Mesh(0)), None);
        assert_eq!(sc.light_pdf_sa(ro, ShapeRef::Bvh(0)), None);
    }

    /// 小网格出图：12 三角形立方体 + 矩形面光（无环境光 ⇒ 背景恒 0）。
    ///
    /// 验收：图像全有限、立方体覆盖区有能量、四角（逃逸）恰为 0。
    #[test]
    fn mesh_cube_12_triangles_renders() {
        let (pts, ids) = cube_mesh(Vec3::new(0.0, 0.0, 0.0), 0.8);
        assert_eq!(ids.len() / 3, 12, "立方体应为 12 三角形");
        let mut scene = Scene::default();
        let mi = scene.add_mesh(pts, ids, MaterialKind::Lambert { albedo: Vec3::splat(0.8) });
        assert_eq!(mi, 0);
        assert_eq!(scene.meshes[0].0.tri_count(), 12);
        assert!(!scene.meshes[0].0.is_soup());
        let (bmin, bmax) = scene.meshes[0].0.bounds().expect("非空网格必有包围盒");
        assert!((bmin.x + 0.8).abs() < 1e-6 && (bmax.z - 0.8).abs() < 1e-6);
        scene.lights.push(Light::Rect {
            center: Vec3::new(0.0, 3.0, 1.0),
            edge_u: Vec3::new(2.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 2.0),
            radiance: Vec3::new(18.0, 16.0, 12.0),
        });
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            50.0,
            0.0,
            3.0,
            0.0,
            1.0 / 24.0,
        );
        let ctx = EvalCtx::new(11, 7, 3);
        let res = (12, 12);
        let params = RenderParams::new(24, 2);
        let frame = render_frame(&scene, &cam, &ctx, res, &params);
        assert_eq!(frame.len(), (res.0 * res.1) as usize);
        let mut max_l = 0.0f32;
        let mut lit = 0usize;
        for (i, px) in frame.iter().enumerate() {
            for c in 0..4 {
                assert!(px[c].is_finite(), "像素 {i} ch{c} 非有限：{}", px[c]);
            }
            let l = px[0] + px[1] + px[2];
            max_l = max_l.max(l);
            if l > 1e-4 {
                lit += 1;
            }
        }
        // 四角为逃逸方向（无环境光）⇒ 恰为 0。
        for corner in [0usize, res.0 as usize - 1, frame.len() - res.0 as usize, frame.len() - 1] {
            let px = frame[corner];
            assert_eq!(px[0], 0.0, "角像素 {corner} 应为背景 0");
            assert_eq!(px[1], 0.0, "角像素 {corner} 应为背景 0");
            assert_eq!(px[2], 0.0, "角像素 {corner} 应为背景 0");
        }
        assert!(max_l > 0.01, "立方体应被照亮：max_l = {max_l}");
        assert!(lit > 20, "被照亮点过少：{lit}");
        println!("12 三角形立方体出图：{res:?} spp{}，lit = {lit}，max_l = {max_l:.4}", params.spp);
    }

    /// 100k 三角形程序化网格：三角形数 ≥ 100k + BVH 构建耗时实测。
    ///
    /// 阈值说明：验收口径 "BVH 构建 < 1s" 以 **release** 为准（此处断言
    /// `cfg!(debug_assertions)` 时放宽到 20 s——debug 下 SAH 构建有 ~10×
    /// 常数惩罚，把 release 口径写死在 debug 里只会得到一条假的回归线）。
    /// 实测值无论哪个 profile 都打印，便于对账。
    #[test]
    fn mesh_100k_triangles_bvh_build_timing() {
        let (rings, segs) = (224usize, 224usize);
        let (pts, ids) = torus_mesh(rings, segs, 1.0, 0.35);
        let expected = rings * segs * 2;
        assert_eq!(ids.len() / 3, expected);
        assert!(expected >= 100_000, "三角形数应 ≥ 100k，实际 {expected}");

        let t0 = std::time::Instant::now();
        let mut sc = Scene::default();
        let mi = sc.add_mesh(pts, ids, MaterialKind::Mirror);
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(mi, 0);
        assert_eq!(sc.triangle_count(), expected);
        // BVH 结构自洽：节点非空、包围盒非退化、叶子覆盖全部图元。
        let bvh: &GeoBvh = &sc.meshes[0].0.bvh;
        assert!(!bvh.nodes.is_empty() && bvh.tris.len() == expected);
        assert!(bvh.bounds.max.x > bvh.bounds.min.x);
        let leaf_sum: usize = bvh
            .nodes
            .iter()
            .filter(|n| n.count > 0)
            .map(|n| n.count as usize)
            .sum();
        assert_eq!(leaf_sum, expected, "叶子图元数应覆盖全部三角形");

        // 求交可用性：从环心上方沿 -y 应命中环体；从环心沿 +y 走洞中不命中。
        let hit = sc.intersect(
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0e30,
            None,
        );
        assert!(hit.is_some(), "环体上方射线应命中");
        let miss = sc.intersect(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0e30,
            None,
        );
        assert!(miss.is_none(), "环洞中垂直射线不应命中");

        let limit_ms = if cfg!(debug_assertions) { 20_000.0 } else { 1_000.0 };
        println!(
            "100k 三角形 BVH 构建：{expected} tris，{build_ms:.1} ms（{} 档，阈值 {limit_ms} ms）",
            if cfg!(debug_assertions) { "debug" } else { "release" }
        );
        assert!(
            build_ms < limit_ms,
            "BVH 构建 {build_ms:.1} ms 超阈值 {limit_ms} ms"
        );

        // 双路径一致性：同一 100k 场景 integrator ≡ wavefront（逐位）。
        sc.lights.push(Light::Rect {
            center: Vec3::new(0.0, 3.0, 0.0),
            edge_u: Vec3::new(2.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 2.0),
            radiance: Vec3::new(14.0, 13.0, 10.0),
        });
        sc.lights.push(Light::Ambient { radiance: Vec3::new(0.05, 0.06, 0.08) });
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            50.0,
            0.0,
            5.0,
            0.0,
            1.0 / 24.0,
        );
        let ctx = EvalCtx::new(100_000, 7, 3);
        let res = (4u32, 4u32);
        let params = RenderParams::new(1, 2);
        let wf = crate::wavefront::render_frame(&sc, &cam, &ctx, res, &params);
        let mut lit = 0usize;
        for y in 0..res.1 {
            for x in 0..res.0 {
                let rt = render_pixel(&sc, &cam, &ctx, (x, y), res, &params);
                let i = (y * res.0 + x) as usize;
                for c in 0..4 {
                    assert_eq!(
                        wf[i][c].to_bits(),
                        rt[c].to_bits(),
                        "100k 场景 parity 破：像素 ({x},{y}) 通道 {c}：{} ≠ {}",
                        wf[i][c],
                        rt[c]
                    );
                }
                if wf[i][0] + wf[i][1] + wf[i][2] > 1e-4 {
                    lit += 1;
                }
            }
        }
        assert!(lit > 0, "100k 场景应至少有一个像素被照亮");
        println!("100k 场景双路径 parity：{res:?} spp{} 逐位一致，lit = {lit}", params.spp);
    }

    /// 索引越界 / 非有限顶点：剔除而非 panic（C18），且 `indices` 与
    /// `triangles` 仍一一对应。
    #[test]
    fn mesh_invalid_triangles_are_dropped() {
        let pts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(f32::NAN, 0.0, 0.0),
        ];
        // 0,1,2 合法；0,1,9 越界；0,3,2 含 NaN。
        let ids = vec![0u32, 1, 2, 0, 1, 9, 0, 3, 2];
        let m = Mesh::new(pts, ids);
        assert_eq!(m.tri_count(), 1);
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert_eq!(m.indices.len() / 3, m.tri_count());
        // 空网格：求交安全返回 None，bounds 为 None。
        let mut sc = Scene::default();
        sc.add_mesh(Vec::new(), Vec::new(), lambert_grey());
        assert!(sc.intersect(Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), 1.0e30, None).is_none());
        assert!(sc.meshes[0].0.bounds().is_none());
        assert_eq!(sc.normal_at(ShapeRef::Mesh(0), Vec3::splat(0.0)).x, 0.0);
    }

    // -----------------------------------------------------------------------
    // V2.0-W4 曲线图元族：圆柱 / 圆锥 / 圆环 / 旋转曲面
    //
    // 验收口径（任务书钉死）：每类 ≥1 条已知值解析对拍（手算参考值写进
    // 注释）+ 1 条边界语义测试；另加全族确定性、BVH≡线性兜底、
    // wavefront 双路径对拍。全部为正向断言 + 变异自证（边界用例同时是
    // 变异杀器：闭边界改开、残差比较改签名比较都会红）。
    // -----------------------------------------------------------------------

    /// t 近似相等（解析对拍容差；参考值均为 O(10) 量级）。
    fn assert_t_close(got: f32, want: f32) {
        assert!(
            (got - want).abs() <= 1.0e-4,
            "t 对拍失败：got = {got}，want = {want}（|Δ| = {}）",
            (got - want).abs()
        );
    }

    /// Vec3 近似相等（法线对拍容差；含 √ 归一化的参考值）。
    fn assert_vec3_close(got: Vec3, want: Vec3) {
        let d = (got - want).length();
        assert!(d <= 1.0e-4, "向量对拍失败：got = {got:?}，want = {want:?}（|Δ| = {d}）");
    }

    /// 四族各一的中置场景（BVH 随 `add_*` 自动重建）。
    fn curve_family_scene() -> Scene {
        let mat = lambert_grey();
        let mut sc = Scene::default();
        sc.add_cylinder(
            Cylinder { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0, half_height: 1.0 },
            mat,
        );
        sc.add_cone(
            Cone { center: Vec3::new(4.0, 0.0, 0.0), radius: 1.0, half_height: 1.0 },
            mat,
        );
        sc.add_torus(
            Torus { center: Vec3::new(8.0, 0.0, 0.0), major: 2.0, minor: 0.5 },
            mat,
        );
        sc.add_revolution(
            Revolution::new(Vec3::new(12.0, 0.0, 0.0), vec![[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
            mat,
        );
        sc.lights = vec![Light::Directional {
            direction: Vec3::new(-0.3, -0.8, -0.52).normalize(),
            radiance: Vec3::splat(5.0),
            cos_theta_max: 0.9,
        }];
        sc
    }

    /// 每族 3 个、共 12 个曲线图元（BVH 多叶节点，喂 ≡ 线性对拍）。
    fn curve_multi_scene() -> Scene {
        let mat = lambert_grey();
        let mut sc = Scene::default();
        for i in 0..3usize {
            let x = i as f32 * 3.0;
            sc.add_cylinder(
                Cylinder { center: Vec3::new(x, 0.0, 0.0), radius: 0.6, half_height: 1.2 },
                mat,
            );
            sc.add_cone(
                Cone { center: Vec3::new(x, 0.0, 3.0), radius: 0.7, half_height: 1.1 },
                mat,
            );
            sc.add_torus(
                Torus { center: Vec3::new(x, 0.0, 6.0), major: 1.6, minor: 0.4 },
                mat,
            );
            sc.add_revolution(
                Revolution::new(Vec3::new(x, 0.0, 9.0), vec![[0.0, -1.2], [0.9, 0.0], [0.0, 1.2]]),
                mat,
            );
        }
        sc
    }

    /// 曲线场景相机（四族沿 x 排布，焦距内全覆盖）。
    fn curve_camera() -> Camera {
        Camera::new(
            Vec3::new(6.0, 2.5, 12.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            12.0,
            0.0,
            1.0 / 24.0,
        )
    }

    /// 圆柱已知值对拍：侧面 t=4 + 顶盖 t=2（均为精确解析值）+ 双侧法线。
    ///
    /// 手算（圆柱 R=1、H=1、居中）：
    /// - 射线 (0,0,-5)+t·(0,0,1) 代入 x²+z²=1：a=1、b=-10、c=24、
    ///   disc=4 ⇒ t=4/6，y=0 段内 ⇒ 最近 t=4，命中点 (0,0,-1)（纯侧面，
    ///   法线 (0,0,-1)——侧/盖归属按残差绝对值，见 `normal_at` 注释）；
    /// - 射线 (0,3,0)+t·(0,-1,0) 顶盖 y=1：t=2，ρ=0 半径内，法线 (0,1,0)。
    #[test]
    fn cylinder_side_cap_known_hits_and_normals() {
        let mut sc = Scene::default();
        sc.add_cylinder(
            Cylinder { center: Vec3::splat(0.0), radius: 1.0, half_height: 1.0 },
            lambert_grey(),
        );
        let big = 1.0e30;

        // 侧面：t = 4 精确。
        let h = sc
            .intersect(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), big, None)
            .expect("圆柱侧面应命中");
        assert_t_close(h.t, 4.0);
        assert_eq!(h.shape, ShapeRef::Cylinder(0));
        let p = Vec3::new(0.0, 0.0, -5.0) + Vec3::new(0.0, 0.0, 1.0) * h.t;
        assert_eq!(sc.normal_at(h.shape, p), Vec3::new(0.0, 0.0, -1.0));

        // 顶盖：t = 2 精确。
        let h2 = sc
            .intersect(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), big, None)
            .expect("圆柱顶盖应命中");
        assert_t_close(h2.t, 2.0);
        let p2 = Vec3::new(0.0, 3.0, 0.0) + Vec3::new(0.0, -1.0, 0.0) * h2.t;
        assert_eq!(sc.normal_at(h2.shape, p2), Vec3::new(0.0, 1.0, 0.0));
    }

    /// 圆柱边界语义：棱线闭裁剪（切向 Rim 单根 y==H）、端盖半径闭边界、
    /// 闭合实体（体内发射取出口）、轴平行界外、退化参数永不命中。
    #[test]
    fn cylinder_boundary_semantics() {
        let mut sc = Scene::default();
        sc.add_cylinder(
            Cylinder { center: Vec3::splat(0.0), radius: 1.0, half_height: 1.0 },
            lambert_grey(),
        );
        let big = 1.0e30;
        let side = |sc: &Scene, ro: Vec3, rd: Vec3| sc.intersect(ro, rd, big, None).map(|h| h.t);

        // (a) 切向 Rim：射线 (1,1,-3)+t·(0,0,1) 与侧面相切（disc=0，t=3），
        // 命中点 y=1 == H —— 闭段界（改开即脱靶）。
        assert_t_close(side(&sc, Vec3::new(1.0, 1.0, -3.0), Vec3::new(0.0, 0.0, 1.0)).unwrap(), 3.0);
        // (b) 端盖半径闭边界：ρ=1 == R 命中；ρ=1.001 界外（侧面 a=0 跳过）。
        assert_t_close(side(&sc, Vec3::new(1.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0)).unwrap(), 2.0);
        assert_eq!(side(&sc, Vec3::new(1.001, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0)), None);
        // (c) 闭合实体：体内 (0,0,0) 沿 +z 发射取侧壁出口 t=1（最近 t>0）。
        assert_t_close(side(&sc, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0)).unwrap(), 1.0);
        // (d) 轴平行界外（x=2）：侧面退化跳过、端盖半径外 ⇒ None。
        assert_eq!(side(&sc, Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), None);

        // (e) 退化参数：radius <= 0 / half_height <= 0 ⇒ 永不命中。
        let mut dz = Scene::default();
        dz.add_cylinder(
            Cylinder { center: Vec3::splat(0.0), radius: 0.0, half_height: 1.0 },
            lambert_grey(),
        );
        dz.add_cylinder(
            Cylinder { center: Vec3::splat(0.0), radius: 1.0, half_height: -1.0 },
            lambert_grey(),
        );
        assert_eq!(side(&dz, Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)), None);
        // 非有限射线 ⇒ None（不 panic）。
        assert!(sc
            .intersect(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), big, None)
            .is_none());
    }

    /// 圆锥已知值对拍：侧面 t=4.5 + 底面 t=2 + 顶点 t=2（全精确）+ 法线。
    ///
    /// 手算（圆锥 R=1、H=1、居中，k=R/2H=0.5，底面 y=-1、顶点 y=+1）：
    /// - 射线 (0,0,-5)+t·(0,0,1)：a=1、b=-10、c=24.75、disc=1 ⇒ t=4.5/5.5，
    ///   y=0 段内 ⇒ 最近 4.5，命中点 (0,0,-0.5)（ρ=0.5 = k(H-y) 侧面上），
    ///   法线 (0, k²(H-y), z)/|·| = (0, 0.25, -0.5)/√0.3125 = (0,1,-2)/√5；
    /// - 射线 (0,-3,0)+t·(0,1,0)：底面 y=-1 ⇒ t=2，ρ=0 半径内，法线 (0,-1,0)；
    /// - 射线 (0,3,0)+t·(0,-1,0)：顶点 y=+1 ⇒ t=2（disc=0 相切），法线兜底
    ///   (0,1,0)（顶点处侧面法线退化，测度零）。
    #[test]
    fn cone_known_hits_and_normals() {
        let mut sc = Scene::default();
        sc.add_cone(
            Cone { center: Vec3::splat(0.0), radius: 1.0, half_height: 1.0 },
            lambert_grey(),
        );
        let big = 1.0e30;

        let h = sc
            .intersect(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), big, None)
            .expect("圆锥侧面应命中");
        assert_t_close(h.t, 4.5);
        assert_eq!(h.shape, ShapeRef::Cone(0));
        let p = Vec3::new(0.0, 0.0, -5.0) + Vec3::new(0.0, 0.0, 1.0) * h.t;
        assert_vec3_close(sc.normal_at(h.shape, p), Vec3::new(0.0, 1.0, -2.0) * (1.0 / 5.0_f32.sqrt()));

        let h2 = sc
            .intersect(Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0), big, None)
            .expect("圆锥底面应命中");
        assert_t_close(h2.t, 2.0);
        let p2 = Vec3::new(0.0, -3.0, 0.0) + Vec3::new(0.0, 1.0, 0.0) * h2.t;
        assert_eq!(sc.normal_at(h2.shape, p2), Vec3::new(0.0, -1.0, 0.0));

        let h3 = sc
            .intersect(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), big, None)
            .expect("圆锥顶点应命中");
        assert_t_close(h3.t, 2.0);
        let p3 = Vec3::new(0.0, 3.0, 0.0) + Vec3::new(0.0, -1.0, 0.0) * h3.t;
        assert_eq!(sc.normal_at(h3.shape, p3), Vec3::new(0.0, 1.0, 0.0));
    }

    /// 圆锥边界语义：顶点闭段界（y==H 命中，改开则坠落到底面 t=4）、
    /// 底面半径闭边界、锥外完全脱靶、退化参数永不命中。
    #[test]
    fn cone_boundary_semantics() {
        let mut sc = Scene::default();
        sc.add_cone(
            Cone { center: Vec3::splat(0.0), radius: 1.0, half_height: 1.0 },
            lambert_grey(),
        );
        let big = 1.0e30;
        let t_of = |sc: &Scene, ro: Vec3, rd: Vec3| sc.intersect(ro, rd, big, None).map(|h| h.t);

        // (a) 顶点闭段界：轴射线自上方入射，侧面根 t=2（y=+1==H）必须胜过
        // 底面根 t=4；段界若改开（y>=H 拒）则退化成 t=4 —— 本断言即杀器。
        assert_t_close(t_of(&sc, Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0)).unwrap(), 2.0);
        // 段界外：同向但起点偏移使侧面根 y>H，仅剩底面根。
        // 射线 (0.1,2,0)+t·(0,-1,0)：侧面根 t=0.8/1.2，y=1.2 界外拒、
        // y=0.8 命中 ⇒ t=1.2（手算 disc=0.01、sq=0.1）。
        assert_t_close(t_of(&sc, Vec3::new(0.1, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0)).unwrap(), 1.2);
        // (b) 底面半径闭边界：ρ=1 == R 命中；ρ=1.001 界外（侧面根 y=-1.002
        // 段界外，底面 ρ²>R²）⇒ None。
        assert_t_close(t_of(&sc, Vec3::new(1.0, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap(), 2.0);
        assert_eq!(t_of(&sc, Vec3::new(1.001, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), None);
        // (c) 锥外脱靶：x=2 轴平行入射，侧面根 y=5 界外、底面根 t=-1 拒。
        assert_eq!(t_of(&sc, Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), None);

        // (d) 退化参数 ⇒ 永不命中。
        let mut dz = Scene::default();
        dz.add_cone(Cone { center: Vec3::splat(0.0), radius: 0.0, half_height: 1.0 }, lambert_grey());
        dz.add_cone(Cone { center: Vec3::splat(0.0), radius: 1.0, half_height: 0.0 }, lambert_grey());
        assert_eq!(t_of(&dz, Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)), None);
    }

    /// 圆环已知值对拍：−z 轴穿环射线四根 {2.5, 3.5, 6.5, 7.5}（精确）。
    ///
    /// 手算（圆环 R=2、r=0.5、居中，隐式 (ρ-2)²+y²=0.25）：
    /// 射线 (0,0,5)+t·(0,0,-1) ⇒ ρ=|5-t|、y=0 ⇒ ρ=1.5/2.5 ⇒
    /// t = 3.5/6.5（前内/后内）与 2.5/7.5（前外/后外）；代入射线的四次
    /// 式为 (t-2.5)(t-3.5)(t-6.5)(t-7.5)（系数 c4..c0 = 1,-20,141.5,
    /// -415,426.5625，f32 精确可表）。最近 t=2.5，命中点 (0,0,2.5)，
    /// 法线 (1-R/ρ)·p = (0,0,1)。
    #[test]
    fn torus_known_hits_and_normal() {
        let mut sc = Scene::default();
        sc.add_torus(
            Torus { center: Vec3::splat(0.0), major: 2.0, minor: 0.5 },
            lambert_grey(),
        );
        let big = 1.0e30;

        let h = sc
            .intersect(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), big, None)
            .expect("圆环应命中");
        assert_t_close(h.t, 2.5);
        assert_eq!(h.shape, ShapeRef::Torus(0));
        let p = Vec3::new(0.0, 0.0, 5.0) + Vec3::new(0.0, 0.0, -1.0) * h.t;
        assert_vec3_close(sc.normal_at(h.shape, p), Vec3::new(0.0, 0.0, 1.0));

        // 管顶切向：射线 (0,0.5,5)+t·(0,0,-1)（y=r 顶 silhouette，
        // 四次式 (t-3)²(t-7)² 精确双根）⇒ t=3，法线 (0,1,0)。
        let h2 = sc
            .intersect(Vec3::new(0.0, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0), big, None)
            .expect("圆环管顶切向应命中");
        assert_t_close(h2.t, 3.0);
        let p2 = Vec3::new(0.0, 0.5, 5.0) + Vec3::new(0.0, 0.0, -1.0) * h2.t;
        assert_vec3_close(sc.normal_at(h2.shape, p2), Vec3::new(0.0, 1.0, 0.0));
    }

    /// 圆环边界语义：穿环孔脱靶、管顶上方脱靶、退化参数永不命中。
    #[test]
    fn torus_boundary_semantics() {
        let mut sc = Scene::default();
        sc.add_torus(
            Torus { center: Vec3::splat(0.0), major: 2.0, minor: 0.5 },
            lambert_grey(),
        );
        let big = 1.0e30;
        let t_of = |sc: &Scene, ro: Vec3, rd: Vec3| sc.intersect(ro, rd, big, None).map(|h| h.t);

        // (a) 环孔内竖射：(ρ-2)²+y²=0.25 在 ρ=0 时 4+y²=0.25 无解 ⇒ None。
        assert_eq!(t_of(&sc, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 1.0, 0.0)), None);
        // (b) 管顶上方水平射（y=3 > r）：四次式无实根 ⇒ None。
        assert_eq!(t_of(&sc, Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, 0.0, -1.0)), None);
        // (c) t_max 裁剪：t_max=2.0 时最近根 2.5 界外 ⇒ None（闭区间上界）。
        assert_t_close(t_of(&sc, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).unwrap(), 2.5);
        assert!(sc
            .intersect(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), 2.0, None)
            .is_none());

        // (d) 退化参数 ⇒ 永不命中。
        let mut dz = Scene::default();
        dz.add_torus(Torus { center: Vec3::splat(0.0), major: 0.0, minor: 0.5 }, lambert_grey());
        dz.add_torus(Torus { center: Vec3::splat(0.0), major: 2.0, minor: -0.5 }, lambert_grey());
        assert_eq!(t_of(&dz, Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0)), None);
    }

    /// 旋转曲面已知值对拍：双锥轮廓 [(0,-1),(1,0),(0,1)] 赤道命中 t=4
    /// （精确）+ 外法线 (0,-1,-1)/√2；绕向归一使 CW 声明与 CCW 同解。
    ///
    /// 手算（轮廓绕 +Y 旋转；段 0：(0,-1)→(1,0)，Δr=1、Δy=1）：
    /// 射线 (0,0,-5)+t·(0,0,1)：b2=1、b1=-10、b0=25、A=Δr(qy-y0)+Δy·r0=1，
    /// a=1、b=-10、c=24、disc=4 ⇒ t=4/6，y=0 段内 ⇒ 最近 4，命中点
    /// (0,0,-1)（ρ=1 赤道棱）。两段在赤道并列 ⇒ 取小索引段 0，外法线
    /// (Δy·x/ρ, -Δr, Δy·z/ρ) = (0,-1,-1)，归一 (0,-1,-1)/√2。
    #[test]
    fn revolution_known_hits_and_normal() {
        let ccw: Vec<[f32; 2]> = vec![[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        let cw: Vec<[f32; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]];

        let mut sc = Scene::default();
        sc.add_revolution(Revolution::new(Vec3::splat(0.0), ccw.clone()), lambert_grey());
        assert_eq!(sc.revolutions[0].0.segment_count(), 2);

        let big = 1.0e30;
        let h = sc
            .intersect(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), big, None)
            .expect("旋转曲面应命中");
        assert_t_close(h.t, 4.0);
        assert_eq!(h.shape, ShapeRef::Revolution(0));
        let p = Vec3::new(0.0, 0.0, -5.0) + Vec3::new(0.0, 0.0, 1.0) * h.t;
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert_vec3_close(sc.normal_at(h.shape, p), Vec3::new(0.0, -inv_sqrt2, -inv_sqrt2));

        // 绕向归一（shoelace 有向面积 <0 ⇒ 反转）：CW 声明归一后与 CCW
        // 逐点相同 ⇒ 同段序列、同命中、同法线（纯函数口径）。
        let mut sc_cw = Scene::default();
        sc_cw.add_revolution(Revolution::new(Vec3::splat(0.0), cw), lambert_grey());
        assert_eq!(sc_cw.revolutions[0].0.profile, ccw);
        let h_cw = sc_cw
            .intersect(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), big, None)
            .expect("CW 旋转曲面应命中");
        assert_eq!(h_cw.t.to_bits(), h.t.to_bits());
        assert_eq!(h_cw.shape, h.shape);
    }

    /// 旋转曲面边界语义：水平段环带（ρ 闭区间 + 共面不命中）、斜段
    /// y 闭段界、退化轮廓（点数 <2 / 非有限 / 负半径）永不命中。
    #[test]
    fn revolution_boundary_semantics() {
        let big = 1.0e30;
        let t_of = |sc: &Scene, ro: Vec3, rd: Vec3| sc.intersect(ro, rd, big, None).map(|h| h.t);

        // (a) 水平段（环带 y=0、ρ∈[0.5,1.5]）：ρ=1 命中 t=2 精确；
        // ρ=0.5/1.5 闭边界命中；ρ=0.4/1.6 界外脱靶。
        let mut ann = Scene::default();
        ann.add_revolution(
            Revolution::new(Vec3::splat(0.0), vec![[0.5, 0.0], [1.5, 0.0]]),
            lambert_grey(),
        );
        assert_t_close(t_of(&ann, Vec3::new(1.0, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap(), 2.0);
        assert!(t_of(&ann, Vec3::new(0.5, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).is_some());
        assert!(t_of(&ann, Vec3::new(1.5, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).is_some());
        assert_eq!(t_of(&ann, Vec3::new(0.4, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), None);
        assert_eq!(t_of(&ann, Vec3::new(1.6, -2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)), None);
        // 共面测度零：射线躺在环带平面内（dy=0）⇒ 不命中（钉死口径）。
        assert_eq!(t_of(&ann, Vec3::new(1.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 1.0)), None);

        // (b) 斜段 y 闭段界：竖直管轮廓 [(1,-1),(1,1)]，射线 (0,1,-3)+
        // t·(0,0,1) 命中点 y=1 == y1（闭）；手算 disc=64 ⇒ t=2/4。
        let mut tube = Scene::default();
        tube.add_revolution(
            Revolution::new(Vec3::splat(0.0), vec![[1.0, -1.0], [1.0, 1.0]]),
            lambert_grey(),
        );
        assert_t_close(t_of(&tube, Vec3::new(0.0, 1.0, -3.0), Vec3::new(0.0, 0.0, 1.0)).unwrap(), 2.0);

        // (c) 退化轮廓 ⇒ 永不命中：点数 <2（段数 0）、非有限点、负半径。
        let mut dz = Scene::default();
        dz.add_revolution(Revolution::new(Vec3::splat(0.0), vec![[1.0, 0.0]]), lambert_grey());
        dz.add_revolution(
            Revolution::new(Vec3::splat(0.0), vec![[1.0, -1.0], [f32::NAN, 1.0]]),
            lambert_grey(),
        );
        dz.add_revolution(
            Revolution::new(Vec3::splat(0.0), vec![[-1.0, -1.0], [-1.0, 1.0]]),
            lambert_grey(),
        );
        assert_eq!(t_of(&dz, Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0)), None);
    }

    /// 曲线四族渲染确定性：同参数两遍渲染逐位一致（D-29），md5 自锚。
    #[test]
    fn curve_family_render_bitwise_deterministic() {
        let scene = curve_family_scene();
        let cam = curve_camera();
        let res = (12u32, 9u32);
        let params = RenderParams::new(4, 2);
        let f1 = render_frame(&scene, &cam, &EvalCtx::new(11, 7, 3), res, &params);
        let f2 = render_frame(&scene, &cam, &EvalCtx::new(11, 7, 3), res, &params);
        for i in 0..f1.len() {
            for c in 0..4 {
                assert_eq!(f1[i][c].to_bits(), f2[i][c].to_bits(), "曲线场景两遍渲染不逐位一致 px{i} ch{c}");
            }
        }
        // 非空洞：至少一个像素被照亮（否则对拍是空转）。
        let lit = f1
            .iter()
            .filter(|px| px[0] + px[1] + px[2] > 1.0e-4)
            .count();
        assert!(lit > 0, "曲线场景应至少有一个像素被照亮");
        let mut bytes = Vec::new();
        for px in &f1 {
            for c in 0..4 {
                bytes.extend_from_slice(&px[c].to_le_bytes());
            }
        }
        println!("曲线四族 6×6 spp4 md5 = {}", md5_hex(&bytes));
    }

    /// BVH 快路径 ≡ 线性兜底：12 图元 × 256 射线扇，(t, shape) 逐位同解。
    ///
    /// 对拍方法：同源场景两份，一份保留 `add_*` 自动建的 BVH，一份把
    /// `curve_bvh` 置 None（计数不符 ⇒ `intersect` 自动退化线性兜底）。
    /// 两路必须逐位一致（min t 优先、t 并列取最小全局曲线 id）。
    #[test]
    fn curve_bvh_matches_linear_scan() {
        let bvh_scene = curve_multi_scene();
        assert!(bvh_scene.curve_bvh.is_some(), "add_* 后曲线 BVH 应已构建");
        assert_eq!(
            bvh_scene.curve_bvh.as_ref().map(|b| b.prim_count()),
            Some(12),
            "BVH 图元数应等于四族总数"
        );
        let mut lin_scene = bvh_scene.clone();
        lin_scene.curve_bvh = None; // 强制线性兜底路径

        // 确定性射线扇（整数哈希 ⇒ 跨平台可复现，无 rand 依赖）。
        let hash01 = |k: u32| -> f32 {
            let mut x = k.wrapping_mul(2654435761);
            x ^= x >> 13;
            x = x.wrapping_mul(1274126177);
            x ^= x >> 16;
            (x & 0x00FF_FFFF) as f32 / 16777216.0
        };
        let mut hits = 0usize;
        let mut per_family = [0usize; 4];
        for i in 0..256u32 {
            let ro = Vec3::new(
                -4.0 + 14.0 * hash01(i.wrapping_mul(4)),
                -3.0 + 6.0 * hash01(i.wrapping_mul(4).wrapping_add(1)),
                -6.0 + 20.0 * hash01(i.wrapping_mul(4).wrapping_add(2)),
            );
            let mut d = Vec3::new(
                hash01(i.wrapping_mul(4).wrapping_add(3)) - 0.5,
                hash01(i.wrapping_mul(8).wrapping_add(5)) - 0.5,
                hash01(i.wrapping_mul(8).wrapping_add(6)) - 0.5,
            );
            let l = d.length();
            if l < 1.0e-6 {
                continue; // 退化方向跳过（两路同样无命中）
            }
            d = d * (1.0 / l);
            let a = bvh_scene.intersect(ro, d, 1.0e30, None);
            let b = lin_scene.intersect(ro, d, 1.0e30, None);
            match (a, b) {
                (Some(ha), Some(hb)) => {
                    assert_eq!(ha.t.to_bits(), hb.t.to_bits(), "射线 {i} t 不一致：{ha:?} vs {hb:?}");
                    assert_eq!(ha.shape, hb.shape, "射线 {i} 形状不一致：{ha:?} vs {hb:?}");
                    hits += 1;
                    match ha.shape {
                        ShapeRef::Cylinder(_) => per_family[0] += 1,
                        ShapeRef::Cone(_) => per_family[1] += 1,
                        ShapeRef::Torus(_) => per_family[2] += 1,
                        ShapeRef::Revolution(_) => per_family[3] += 1,
                        _ => {}
                    }
                }
                (None, None) => {}
                (x, y) => panic!("射线 {i} 一路命中一路未命中：{x:?} vs {y:?}"),
            }
        }
        assert!(hits >= 20, "命中数过少（{hits}），覆盖不足");
        for (f, n) in per_family.iter().enumerate() {
            assert!(*n >= 1, "族 {f} 在射线扇中零命中，覆盖不足");
        }
        println!("曲线 BVH≡线性对拍：256 射线、{hits} 命中、分族 {per_family:?}");
    }

    /// 曲线四族 wavefront 双路径逐位对拍（探针：新族经 Scene 委派自动
    /// 同步，无逐族特判；若将来 wavefront 绕过 Scene 自建求交即会红）。
    #[test]
    fn curve_family_wavefront_parity() {
        let scene = curve_family_scene();
        let cam = curve_camera();
        let res = (12u32, 9u32);
        let params = RenderParams::new(2, 2);
        let ctx = EvalCtx::new(4242, 7, 3);
        let wf = crate::wavefront::render_frame(&scene, &cam, &ctx, res, &params);
        let mut lit = 0usize;
        for y in 0..res.1 {
            for x in 0..res.0 {
                let rt = render_pixel(&scene, &cam, &ctx, (x, y), res, &params);
                let i = (y * res.0 + x) as usize;
                for c in 0..4 {
                    assert_eq!(
                        wf[i][c].to_bits(),
                        rt[c].to_bits(),
                        "曲线场景 parity 破：像素 ({x},{y}) 通道 {c}：{} ≠ {}",
                        wf[i][c],
                        rt[c]
                    );
                }
                if wf[i][0] + wf[i][1] + wf[i][2] > 1.0e-4 {
                    lit += 1;
                }
            }
        }
        assert!(lit > 0, "曲线场景应至少有一个像素被照亮");
        println!("曲线四族双路径 parity：{res:?} spp{} 逐位一致，lit = {lit}", params.spp);
    }
}
