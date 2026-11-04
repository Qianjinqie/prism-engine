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
use prism_math::{det_trig, Vec3};

use crate::accum::mean_samples;
use crate::bsdf::{hadamard, Bsdf, SampleCtx};
use crate::caustics::{mnee_mirror, mnee_refraction, Refractor};
use crate::camera::{Camera, Ray};
use crate::instance::{Instance, INSTANCE_MAT_FROM_GEOM};
use crate::light::{ambient_background, nee_mis_weight, sample_nee};

/// 光源（四族；对外沿用 integrator 命名，内部委托 [`crate::light`]）。
pub use crate::light::Light;
use crate::media::Medium;
use crate::sampler::SampleStream;
use crate::spectral::{
    cauchy_ior, flat_spectral_value, hero_wavelength, spectral_to_rgb, SpectralConfig,
};
use crate::sss::random_walk;
use crate::texsample::{Footprint, TexTable};
use crate::texture::TextureRef;

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

/// 有限圆柱（轴向 +Y，带上下端盖；曲线图元族）。
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

/// 有限圆锥（轴向 +Y；曲线图元族）。
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

/// 圆环（主环位于 XZ 平面，轴向 +Y；曲线图元族）。
///
/// 隐式方程 `(ρ-R)² + y² = r²`（ρ = √(x²+z²)，局部坐标）。射线代入得
/// **四次方程**，精确求根走 [`prism_geometry::quartic::solve_quartic`]
/// （Ferrari 闭式、零迭代；规格钉死禁止牛顿迭代）。闭合实体语义：
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

/// 旋转曲面（轮廓绕 +Y 轴旋转；曲线图元族）。
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
    /// 顶点域 uv（W1-A 定桩；`[u, v]` 与 `positions` **同长同序**）。
    ///
    /// # 为什么是顶点域而不是角点域
    ///
    /// 选型报告 §2.2 的决定：`uv` 定义在顶点上，则相邻面共享同一顶点 ⇒
    /// 共享同一 `uv` ⇒ 位移/法线在缝处**逐位相等**，缝在结构上不存在
    /// （W4 的判据 V4 靠这条成立）。角点域能表达硬边缝合，代价是每个
    /// 角点都要独立 uv，而缝处"两个 uv 本该相等"变成需要校验的约定。
    ///
    /// 空 `Vec` = 该网格无 uv（现有全部网格都是这样）；
    /// 非空时长度**必须**等于 `positions.len()`（W1-B 的校验，本段只定字段）。
    ///
    /// 用 `[f32; 2]` 而非任何三元组：uv 是二元量，且 `[f32; N]` 是保守类型
    /// 集合里的写法（禁 `vec3<f32>`，见 [`crate::layout`]）。
    pub uv: Vec<[f32; 2]>,
    /// 顶点域切线（W1-A 定桩；`[tx, ty, tz, w]`，`w` = 副切线手性 ±1）。
    ///
    /// 由 uv 偏导在 **CPU 烘焙**（Lengyel 法），运行期零成本，且避免屏幕空间
    /// 求导——后者按 2×2 quad 求导，跨后端/跨 tile 边界**不可复现**
    /// （选型报告 §2.1 第 3 条）。本段只存字段，烘焙是 **W1-B**。
    ///
    /// 空 `Vec` = 无切线；非空时长度**必须**等于 `positions.len()`。
    ///
    /// 用 `[f32; 4]` 承载三元组 + 手性：定长 16 B、4 B 对齐，与
    /// `TRIANGLE_SLOTS` 的"`vec3` 后补 pad"同款保守写法。
    pub tangent: Vec<[f32; 4]>,
}

impl Mesh {
    /// 网格堆内存估算（字节；`size_of::<Mesh>()` 之外的部分）。
    ///
    /// 逐 `Vec` 按容量计，外加 BVH 的三个扁平数组。用途同
    /// [`Scene::memory_bytes`]：多区域介质 / 实例化之后，"场景多大"必须有
    /// 一处统一口径，否则内存问题只能靠事后 RSS 猜。
    pub fn memory_bytes(&self) -> usize {
        let mut n: usize = 0;
        n += self.positions.capacity() * core::mem::size_of::<Vec3>();
        n += self.indices.capacity() * core::mem::size_of::<u32>();
        n += self.triangles.capacity() * core::mem::size_of::<GeoTriangle>();
        n += self.uv.capacity() * core::mem::size_of::<[f32; 2]>();
        n += self.tangent.capacity() * core::mem::size_of::<[f32; 4]>();
        n += self.bvh.nodes.capacity() * core::mem::size_of::<prism_geometry::bvh::BvhNode>();
        n += self.bvh.tris.capacity() * core::mem::size_of::<GeoTriangle>();
        n += self.bvh.order.capacity() * core::mem::size_of::<u32>();
        n
    }

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
        Mesh {
            positions,
            indices: valid,
            triangles,
            bvh,
            uv: Vec::new(),
            tangent: Vec::new(),
        }
    }

    /// 三角形 soup 构造（无拓扑；BVH 在此构建一次）。
    pub fn from_triangles(triangles: Vec<GeoTriangle>) -> Self {
        let bvh = GeoBvh::build(&triangles);
        Mesh {
            positions: Vec::new(),
            indices: Vec::new(),
            triangles,
            bvh,
            uv: Vec::new(),
            tangent: Vec::new(),
        }
    }

    /// 顶点/索引**已构建树**的 soup 构造（零重建：直接接管调用方的树）。
    ///
    /// 前置条件：`bvh` 必须由同一 `triangles` 序列构建（否则图元索引
    /// 对不上）；本函数不做校验（O(n) 校验会抵消零重建收益）。
    pub fn from_prebuilt(triangles: Vec<GeoTriangle>, bvh: GeoBvh) -> Self {
        Mesh {
            positions: Vec::new(),
            indices: Vec::new(),
            triangles,
            bvh,
            uv: Vec::new(),
            tangent: Vec::new(),
        }
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

/// 材质（基线三型 + Disney PBR 全族 + 毛发）。
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
    /// 毛发纤维（W5 / A7 定稿 C1）：Marschner R/TT/TRT 三分量 + 纤维标架。
    ///
    /// # 为什么两个字段按值放（而不是 `Arc` / 引用）
    ///
    /// `MaterialKind` 必须是 `Copy`：`Scene` 以 `(几何, 材质)` 值对存材质
    /// （19 处结构体字面量），命中结果也按值带出材质。而
    /// [`crate::hair_scatter::HairScatter`]（9×f32）与
    /// [`crate::hair_scatter::HairGeometry`]（2×Vec3）**本来就都派生了 `Copy`**
    /// —— 它们是无堆分配的纯参数包。故按值放进枚举：不引入 `Arc`、
    /// 不引入生命周期、`MaterialKind` 保持 `Copy`。
    ///
    /// 代价：`size_of::<MaterialKind>()` 从 `Pbr` 的 72 B 档位增到本变体的
    /// 64 B 档位（两个 `Copy` 小程序相加不超过 `Bsdf`），**没有 ABI 影响**：
    /// 材质枚举从不上 GPU（GPU 只见三角形/球/光/纹理，见 `scene_abi`）。
    Hair {
        /// 散射参数（三瓣权重、纵向/方位粗糙度、纤维折射率与吸收）。
        scatter: crate::hair_scatter::HairScatter,
        /// 纤维局部标架：切向 + 法平面参考方向。
        ///
        /// # 一材质一标架的边界（如实标注）
        ///
        /// 本变体给的是**每个 `Mesh` 一个**标架，因此"一个 Mesh 内多股不同朝向"
        /// 时只有一股的切向是精确的。精确做法是每股一个 `Mesh`（也就每股一个
        /// 材质）——毛股几何本身就按股三角化，故按股注册是自然的；把**同向**
        /// 股束合并成一个 Mesh 也仍然精确。这一限制在报告 §21 里记为已知边界。
        geometry: crate::hair_scatter::HairGeometry,
    },
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
        // 毛发：续弹已改走 [`crate::hair_scatter::sample_hair_bounce`]（按 f
        // 抽样，R-6②），NEE 的 f / pdf 由 `surface_brdf` / `surface_pdf` 的
        // Hair 臂用 `eval_full` / `pdf_incident` 覆盖。这里保留 Disney 等价形
        // 只供非续弹的 Disney 侧求值（如 MNEE 焦散），**不再**决定方向或 pdf。
        MaterialKind::Hair { scatter, .. } => Some(hair_proposal_bsdf(&scatter)),
    }
}

/// 毛发材质的 Disney 等价形（**仅非续弹路径使用**，见下）。
///
/// # 历史与现状（R-6② 已治本）
///
/// R-5/T6 时期 Marschner 没有按 f 的抽样器，续弹方向、pdf、吞吐都从这个
/// Disney 提案取，BRDF 取值另用 `eval_full` —— 无偏但 pdf 不贴 f。R-6② 起
/// `hair_scatter.rs` 新增 [`crate::hair_scatter::sample_hair_bounce`]：按三瓣
/// 权重选瓣 → 纵向高斯采 θ → 方位偏移采 φ → 组装方向，pdf 与 f 同源；
/// 续弹与 NEE 的 pdf 都改走 `pdf_incident`。本函数因此**只**保留给
/// `material_bsdf_at` 的 Disney 侧求值（MNEE 焦散等），不参与毛发续弹。
///
/// # 参数映射（固定，不随场景变）
///
/// * `base_color = splat(absorption_transmit(radius))`：纤维的 Beer 透过率，
///   落在 (0,1]，是"这根纤维能反多少光"的标量代理；
/// * `metalness = 0`（纤维是电介质）；
/// * `roughness = clamp(beta_m, 0.02, 1)`：纵向粗糙度直接当感知粗糙度用
///   （`beta_m` 的语义就是"多散"，与 roughness 同向）；下界 0.02 是为了不落
///   进 δ 瓣分支（那会让续弹退化成镜面，毛发立刻变成塑料丝）；
/// * `ior = eta`（纤维折射率，TT/TRT 的物理量）；
/// * 无透射/无 SSS（毛发不是玻璃，`transmission` 置 0 让 Disney 走不透明瓣；
///   TT/TRT 的能量已经由 `eval_full` 的取值承担）。
fn hair_proposal_bsdf(scatter: &crate::hair_scatter::HairScatter) -> Bsdf {
    let tint = scatter.absorption_transmit(scatter.radius);
    Bsdf::new(
        Vec3::splat(tint),
        0.0,
        scatter.beta_m.clamp(0.02, 1.0),
        scatter.eta.max(1.0),
        0.0,
        0.0,
        Vec3::splat(0.3),
    )
}

// ---------------------------------------------------------------------------
// W1-B：贴图材质（`material_bsdf(m)` 的 uv 版）
// ---------------------------------------------------------------------------

/// 材质是否挂了任何贴图（决定是否需要算 uv / 足迹——无贴图场景零成本）。
#[inline]
pub(crate) fn material_has_maps(m: &MaterialKind) -> bool {
    match m {
        MaterialKind::Pbr(b) => b.maps.iter().any(|&h| h != crate::texture::NO_TEXTURE),
        _ => false,
    }
}

/// 材质是否挂了**法线贴图**（决定是否需要算切线基——比算基础贴图贵）。
#[inline]
pub(crate) fn has_normal_map(m: &MaterialKind) -> bool {
    match m {
        MaterialKind::Pbr(b) => b.maps[crate::bsdf::MAP_NORMAL] != crate::texture::NO_TEXTURE,
        _ => false,
    }
}

/// 单张贴图的采样（句柄 0 / 越界 → `None`）。
#[inline]
fn sample_map(
    tex: TexTable<'_>,
    handle: u32,
    uv: [f32; 2],
    foot: Footprint,
) -> Option<[f32; 4]> {
    if handle == crate::texture::NO_TEXTURE {
        return None;
    }
    crate::texsample::sample(&tex, handle, uv, foot)
}

/// 按 uv 解析材质 → Disney [`Bsdf`]（A1 的"纹理在命中点解析成标量"）。
///
/// # 与 [`material_bsdf`] 的关系
///
/// 先取标量基线（`material_bsdf`），再按四张图的句柄覆盖对应字段：
///
/// | 槽 | 覆盖 | 约定 |
/// |---|---|---|
/// | [`MAP_BASE`](crate::bsdf::MAP_BASE) | `base_color`（rgb） | glTF `baseColorTexture` |
/// | [`MAP_METAL_ROUGH`](crate::bsdf::MAP_METAL_ROUGH) | `roughness` ← **G**、`metalness` ← **B** | glTF `metallicRoughnessTexture`（绿=粗糙度、蓝=金属度） |
/// | [`MAP_NORMAL`](crate::bsdf::MAP_NORMAL) | 不在此处：法线是**方向**，由 [`material_normal_at`] 改着色法线 | 切线空间，+Z 朝外 |
/// | [`MAP_EMISSIVE`](crate::bsdf::MAP_EMISSIVE) | 不在此处：`Bsdf` 无自发光字段 | 由 [`material_emission_at`] 返回 |
///
/// 返回**仍然是 `Copy` 的 `Bsdf`**（选型报告 §2.2 的硬要求：不引入生命周期、
/// 不让 `Bsdf` 变成引用包）。
///
/// # 逐位兼容
///
/// 四个句柄全 0（现有全部场景）时，本函数与 `material_bsdf` 的返回值**逐位
/// 相同**（同一路径、同一表达式，只在句柄非 0 时才改写字段）。
pub(crate) fn material_bsdf_at(
    m: &MaterialKind,
    uv: [f32; 2],
    foot: Footprint,
    tex: TexTable<'_>,
) -> Option<Bsdf> {
    let mut b = material_bsdf(m)?;
    if !matches!(m, MaterialKind::Pbr(_)) {
        return Some(b); // 基线三材质的解析形式精确保留（无贴图槽）
    }
    if let Some(c) = sample_map(tex, b.maps[crate::bsdf::MAP_BASE], uv, foot) {
        b.base_color = Vec3::new(c[0], c[1], c[2]);
    }
    if let Some(c) = sample_map(tex, b.maps[crate::bsdf::MAP_METAL_ROUGH], uv, foot) {
        // glTF：G = roughness、B = metalness（R/A 忽略）。钳到参数域：
        // HDR 线性源可能超出 [0,1]，而 `Bsdf` 的域断言只在构造期跑。
        b.roughness = c[1].clamp(0.0, 1.0);
        b.metalness = c[2].clamp(0.0, 1.0);
        // 逐纹素粗糙度是各向同性的：α_t = α_b = roughness²（Burley 映射）。
        // 材质本身的各向异性（alpha_t≠alpha_b）只在无 metal-rough 贴图时生效
        // —— glTF 的 metallicRoughness 只有一个标量粗糙度，没有方向。
        b.alpha_t = b.roughness * b.roughness;
        b.alpha_b = b.alpha_t;
    }
    Some(b)
}

/// 自发光贴图在命中点的辐亮度（无该贴图 → 零）。
///
/// 与 [`MaterialKind::Emissive`] 的关系：那个变体是**几何级**发光（可注册为
/// `Light` 参与 NEE）；本函数是**材质级**发光（`Bsdf::maps[MAP_EMISSIVE]`），
/// 只对命中生效、不注册 NEE。故它不进 MIS——与"未注册 NEE 的发光几何"
/// 同一口径（`light_pdf_sa` 返回 `None` 的臂）。
pub(crate) fn material_emission_at(
    m: &MaterialKind,
    uv: [f32; 2],
    foot: Footprint,
    tex: TexTable<'_>,
) -> Vec3 {
    let MaterialKind::Pbr(b) = m else {
        return Vec3::splat(0.0);
    };
    match sample_map(tex, b.maps[crate::bsdf::MAP_EMISSIVE], uv, foot) {
        Some(c) => Vec3::new(c[0], c[1], c[2]),
        None => Vec3::splat(0.0),
    }
}

/// 法线贴图 → 扰动后的着色法线（无该贴图 → 原法线，逐位相同）。
///
/// 切线空间约定（glTF / OpenGL）：`+X = T`、`+Y = B`、`+Z = n`；
/// 贴图值 `[0,1]³` 映射到 `[-1,1]³`。`B` 已含手性（`n × T · w`），
/// 故此处不再翻转。
///
/// 扰动只改**着色法线**（决定 BRDF 的入射/出射半球），不改几何法线
/// （决定射线偏移方向与 `from_inside` 判定）——两者混淆会让掠射表面自交。
pub(crate) fn material_normal_at(
    m: &MaterialKind,
    uv: [f32; 2],
    foot: Footprint,
    tex: TexTable<'_>,
    n: Vec3,
    t: Vec3,
    b: Vec3,
) -> Vec3 {
    let MaterialKind::Pbr(bsdf) = m else {
        return n;
    };
    let Some(c) = sample_map(tex, bsdf.maps[crate::bsdf::MAP_NORMAL], uv, foot) else {
        return n;
    };
    let ns = Vec3::new(2.0 * c[0] - 1.0, 2.0 * c[1] - 1.0, 2.0 * c[2] - 1.0);
    let shaded = t * ns.x + b * ns.y + n * ns.z;
    let l2 = shaded.dot(&shaded);
    if l2 > 1e-12 {
        shaded * (1.0 / l2.sqrt())
    } else {
        n // 退化（零长切向组合）：退回几何法线，绝不产生 NaN
    }
}

/// 反射域 BRDF 求值（解析）：Lambert 用 albedo/π（基线精确）；Mirror 用
/// δ（NEE 下不参与，续弹反射方向无 §f 因子）；Pbr 用 Disney [`Bsdf::eval`]；
/// 毛发用 `HairScatter::eval_full`（R/TT/TRT 三瓣完整版）。
///
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
        // 毛发：把两个世界方向投影到纤维局部系，交给 Marschner 三分量核。
        //
        // 方向约定：`wi` 指向光源（入射）、`wo` 指向视点（出射），与
        // `HairScatter::eval_full(theta_i, phi_i, theta_o, phi_o)` 的
        // (i = 入射, o = 出射) 一一对应。`HairGeometry::longitudinal` 给的是
        // "相对法平面的仰角"（= `asin(d·u)`），正是 Marschner 的 θ；
        // `azimuth` 给的是绕纤维轴的方位 φ。整个转换只走
        // `sampler::{cos_sin, sqrt, atan}` 的确定性件（D-29），无 libm。
        MaterialKind::Hair { scatter, geometry } => {
            let theta_i = geometry.longitudinal(wi);
            let phi_i = geometry.azimuth(wi);
            let theta_o = geometry.longitudinal(wo);
            let phi_o = geometry.azimuth(wo);
            Vec3::splat(scatter.eval_full(theta_i, phi_i, theta_o, phi_o))
        }
        _ => bsdf.eval(wo, wi, nv),
    }
}

/// 确定性切基（平面 uv 用）：取与法线**最不平行**的坐标轴做叉乘。
///
/// 为什么不用 `up`：法线接近 ±Y 时 `n × up` 会退化成长度 0 的向量，
/// `normalize` 后是 NaN，而 NaN 会顺着 uv 污染整张贴图（表现为整片像素
/// 变成纹素 (0,0)）。取"最不平行的轴"把叉乘长度下界钉在 `1/√3`。
pub(crate) fn plane_tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let helper = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = n.cross(&helper).normalize();
    let v = n.cross(&u);
    (u, v)
}

/// 周期量（u）的最短环绕差：`d ∈ (−½, ½]`。
///
/// uv 的 u 轴是周期 1 的（球/柱/锥/环/旋转面的方位角，以及平铺重复）：
/// 直接相减会在接缝处得到 ±1 的跳变，被误读成"这里 uv 变化极快" ⇒ 足迹
/// 变成一整张图 ⇒ mip 级被拉到最低 ⇒ 接缝处糊成一条。环绕差是缝处理的
/// 第一道（第二道是采样器的 repeat 取模）。
#[inline]
pub(crate) fn wrap_delta(d: f32) -> f32 {
    d - (d + 0.5).floor()
}

/// 反射域 pdf（解析）：Lambert = cosθ/π；Mirror = 0（δ）；Pbr = Disney pdf；
/// 毛发 = [`HairScatter::pdf_incident`]（与续弹 `prev_pdf_dir` 同一把尺）。
pub(crate) fn surface_pdf(mat: &MaterialKind, bsdf: &Bsdf, wo: Vec3, wi: Vec3, nv: Vec3) -> f32 {
    match *mat {
        MaterialKind::Lambert { .. } => wi.dot(&nv).max(0.0) * FRAC_1_PI,
        MaterialKind::Mirror => 0.0,
        // 全链路 Marschner：NEE 的备择 pdf 必须等于续弹 sampler 的 pdf，
        // 否则 power heuristic 两侧不同尺 ⇒ MIS 有偏。参数顺序与
        // `surface_brdf` 的 `eval_full(theta_i(wi), …, theta_o(wo), …)` 一致。
        MaterialKind::Hair { scatter, geometry } => {
            let theta_i = geometry.longitudinal(wi);
            let phi_i = geometry.azimuth(wi);
            let theta_o = geometry.longitudinal(wo);
            let phi_o = geometry.azimuth(wo);
            scatter.pdf_incident(theta_o, phi_o, theta_i, phi_i)
        }
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

/// 介质区域（单一介质盒：均匀 σt 或异质 σt 场，统一 [`Medium`] 入口）。
#[derive(Debug, Clone)]
pub struct MediumRegion {
    /// 区域包围盒。
    pub bounds: Aabb,
    /// 介质（均匀 / 异质）。
    pub medium: Medium,
    /// 优先级（A6 / W6，C1「区域列表 + 优先级（不重叠）」）。
    ///
    /// # 语义（**只在重叠时才说话**，这是规格的一部分）
    ///
    /// A6 的规格是「**区域不得重叠**」：物理上互斥的介质盒各占一块空间。
    /// 在合规场景里，优先级**永远不参与判定** —— 一段射线在任一 `t` 上至多
    /// 属于一个区域，所以选哪个区域由几何唯一决定（见 [`select_region`]：
    /// 先按入点 `t_enter` 升序，`t_enter` 相同才比 `priority`）。
    ///
    /// 于是本字段的作用只有两个：
    ///
    /// 1. **违规场景的确定性兜底**：用户若摆出重叠区域，`select_region` 仍须
    ///    返回**唯一定值**（逐位），不能依赖 `Vec` 顺序之外的任何东西——
    ///    这条由 `a6_select_region_is_total_order_on_overlap` 钉死；
    /// 2. **未来的表达力预留**：C2（介质栈）解除重叠约束后，"谁遮谁"要有
    ///    一个引擎侧可声明的量，而不是靠列表顺序（列表顺序是构造细节）。
    ///
    /// 值大者胜。缺省 0（与 `MediumRegion::new` 一致）。
    pub priority: u32,
}

impl MediumRegion {
    /// 构造一个优先级为 0 的区域（单区域场景 / 既有测试的最小写法）。
    pub fn new(bounds: Aabb, medium: Medium) -> Self {
        MediumRegion { bounds, medium, priority: 0 }
    }

    /// 指定优先级（链式；`priority` 大者胜，见字段文档）。
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
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
    /// 有限圆柱族（曲线图元；全局曲线 id 段 1）。
    pub cylinders: Vec<(Cylinder, MaterialKind)>,
    /// 有限圆锥族（曲线图元；全局曲线 id 段 2）。
    pub cones: Vec<(Cone, MaterialKind)>,
    /// 圆环族（曲线图元；全局曲线 id 段 3）。
    pub toruses: Vec<(Torus, MaterialKind)>,
    /// 旋转曲面族（曲线图元；全局曲线 id 段 4）。
    pub revolutions: Vec<(Revolution, MaterialKind)>,
    /// 光源列表（NEE 采样对象）。
    pub lights: Vec<Light>,
    /// 可选单一介质区域（**旧路径**；A6 多区域介质到来后仍保留）。
    ///
    /// 语义不变：`media` 为空时按本字段走原分支（逐位兼容，见 [`Scene::media`]）。
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

    // -----------------------------------------------------------------
    // W1-A 定桩（三个字段一次加完；空 ⇒ 行为与加字段前逐位相同）
    //
    // 这三块**本段只存不读**：没有任何渲染分支查询它们。所以「默认档与
    // `--fast` 定桩前后逐字节相同」不是运气，是结构性结论。
    // -----------------------------------------------------------------
    /// 位图纹理表（A1 / W1）。元素只描述"像素在哪儿、多大、什么格式"，
    /// 像素本体在 GPU 侧的扁平 `array<vec4<f32>>` 里（[`TextureRef::offset`]）。
    ///
    /// 空 ⇒ 无纹理（现有全部场景）。长度与 `scene_abi` 的 `P_NUM_TEXTURES`
    /// 同源，禁止各后端各自数一遍。
    pub textures: Vec<TextureRef>,
    /// 几何实例表（A3 / W3）。空 ⇒ 走现有"几何已展平"路径。
    ///
    /// 槽宽 112 B 由 [`Instance`] 的编译期断言钉住。每条记录引用
    /// [`Scene::instance_geoms`] 里**一份共享几何**（`geom_id`），并携带世界
    /// 仿射与其逆（CPU 算一次，三后端读同一份字节）。
    ///
    /// # 语义（W3 定）
    ///
    /// * `geom_id` ⇒ `instance_geoms[geom_id]`（共享几何，BLAS 只建一次）；
    /// * `mat_id == u32::MAX` ⇒ 材质取自共享几何自身，否则取
    ///   `meshes[mat_id].1`（**材质覆盖**；引擎里材质与几何同表存放，
    ///   故覆盖源只能是 `meshes` 的某一条目）；
    /// * 命中细化为 [`ShapeRef::Instance`]`(实例下标, 三角形下标)`。
    pub instances: Vec<Instance>,
    /// **共享几何表**（A3 / W3）：实例原型，`Vec<(Mesh, 自带材质)>`。
    ///
    /// # 为什么必须与 [`Scene::meshes`] 分开（这条不是洁癖，是正确性）
    ///
    /// 原型是**局部坐标系**里的几何，本身不该出现在场景里。若把它塞进
    /// `meshes`，`intersect` 的网格臂会把它当普通几何在原点命中一次 ——
    /// 于是"一个 glTF 节点沿 x 轴摆 10 份"会变成"10 份正确地摆在 x 轴上
    /// **外加**1 份莫名其妙钉在原点"。分开存放后，原型只能经实例到达。
    ///
    /// 附带好处：`meshes` 的既有语义（[`Scene::triangle_count`]、
    /// [`Scene::mesh_count`]、GPU 编码器）**一个字节都不变** —— 现有全部
    /// 场景与后端不受 W3 影响。
    pub instance_geoms: Vec<(Mesh, MaterialKind)>,
    /// 实例 TLAS（A3 / **C1 两级 BVH** 的上层）。
    ///
    /// 复用 [`prism_geometry::aabb_bvh::AabbBvh`]：叶子是"世界空间实例
    /// AABB"，回调精阶段把射线变换进实例局部空间后查下层 BLAS
    /// （[`Mesh::bvh`]，SAH 已确定性）。
    ///
    /// # 为什么是公开字段（而非私有缓存）
    ///
    /// 与 [`Scene::curve_bvh`] 同一理由：`Scene` 的既有构造模式是跨 crate
    /// 结构体字面量（tests / examples），私有字段会让字面量构造直接
    /// E0451。口径同为"公开只读缓存"。
    ///
    /// # 失效与兜底
    ///
    /// [`Scene::add_instance`] / [`Scene::add_instances`] 会把它置 `None`
    /// （**不在每次 push 后重建** —— 那样 1 万实例是 O(n² log n)）；
    /// 批量装载后调一次 [`Scene::rebuild_instance_bvh`]。`None` 或
    /// `prim_count() != instances.len()` ⇒ [`Scene::intersect`] 自动走
    /// 线性兜底（结果与 TLAS 路径一致，只是 O(n)）。
    pub instance_bvh: Option<prism_geometry::aabb_bvh::AabbBvh>,
    /// 位图纹理的**像素本体**（A1 / W1-B）：扁平的线性域 `[f32; 4]` 纹素流。
    ///
    /// `textures[k]` 的纹素从 `textures[k].offset` 起紧接排布（含 mip 链，
    /// 级序递增）。**恒为线性域**：sRGB 源在 [`Scene::add_texture_rgba8`]
    /// 里一次性解码（走 [`crate::texture::SRGB_DECODE_LUT`]），采样器不解码
    /// —— 这样三后端上传的是同一份线性位型，滤波也发生在线性域（物理正确）。
    ///
    /// # 为什么这块存储不在 W1-A 的定桩里
    ///
    /// §11 只定了 `textures` 描述表（GPU 绑定 `texmeta`），没有定"CPU 侧像素
    /// 存在哪"。纹理像素必须有一块与 `array<vec4<f32>>` 逐格对应的扁平缓冲，
    /// 而它的类型（`[f32; 4]`）正是 §11 指定的 GPU 元素类型——**没有引入任何
    /// 新的 ABI 形态**，只是把 GPU 绑定的 CPU 侧 backing store 显式化。
    /// 见 `docs/v2.5-W0-算法选型报告.md` §16.1。
    pub texels: Vec<[f32; 4]>,
    /// 环境贴图的立体角 2D CDF（A2 / W2）——`Light::Environment` 的**只读缓存**。
    ///
    /// # 为什么放在 `Scene` 上而不是每次渲染重建
    ///
    /// 构建一次要扫全图（2048×1024 = 2M texel，量化为 u64 整数前缀和），
    /// 单线程 3~8 ms。逐像素 [`render_pixel`](crate::integrator::render_pixel)
    /// 重建会让一张 512×512 的图从"几秒"变成"几小时"。
    ///
    /// # 为什么是公开字段（而非私有缓存）
    ///
    /// 与 [`Scene::curve_bvh`] / [`Scene::instance_bvh`] 同一口径：`Scene` 的
    /// 既有构造模式是跨 crate 结构体字面量，私有字段会让字面量构造直接
    /// E0451。**只读**：由 [`Scene::add_environment`] /
    /// [`Scene::rebuild_env_cdf`] 维护，其余地方不得写。
    ///
    /// `None` 且存在 [`Light::Environment`] ⇒ 场景装配漏了一步，
    /// [`Scene::validate_layout`] 报错（不会静默变黑）。
    pub env_cdf: Option<crate::env::EnvCdf>,
    /// 多区域介质表（A6 / W6，C1「区域列表 + 优先级（不重叠）」）。
    ///
    /// # 逐位兼容的硬化要求（选型报告 §7）
    ///
    /// `media.is_empty()` 时**必须**走 [`Scene::medium`] 的原分支、一行不改；
    /// 否则旧场景的 aux 步计数会变（每个区域独立计数器一加，采样流就平移）。
    /// 故本字段存在本身不改变任何输出——真正的切换在 W6。
    pub media: Vec<MediumRegion>,
}

// ---------------------------------------------------------------------------
// W1-A 定桩：Mesh / Scene 的布局断言
// ---------------------------------------------------------------------------

/// W1-A 之前的 `Mesh` 布局（**影子结构**：字段与加字段前逐字相同）。
///
/// 用它做"增量断言"而不是写死 `216`：本仓要出 wasm32（32 位，
/// `Vec` 头 12 B），写死字节数会让 wasm 构建挂在一句与它无关的断言上。
/// 影子结构随目标平台一起变，故断言在两种指针宽度下都成立。
#[allow(dead_code)] // 只用于 size_of：字段从不被读
struct MeshBeforeW1A {
    positions: Vec<Vec3>,
    indices: Vec<u32>,
    triangles: Vec<GeoTriangle>,
    bvh: GeoBvh,
}

/// W1-A 之前的 `Scene` 布局（影子结构）。
#[allow(dead_code)] // 只用于 size_of：字段从不被读
struct SceneBeforeW1A {
    spheres: Vec<(Sphere, MaterialKind)>,
    boxes: Vec<(Aabb, MaterialKind)>,
    paraboloids: Vec<(Paraboloid, MaterialKind)>,
    planes: Vec<(Plane, MaterialKind)>,
    meshes: Vec<(MeshBeforeW1A, MaterialKind)>,
    cylinders: Vec<(Cylinder, MaterialKind)>,
    cones: Vec<(Cone, MaterialKind)>,
    toruses: Vec<(Torus, MaterialKind)>,
    revolutions: Vec<(Revolution, MaterialKind)>,
    lights: Vec<Light>,
    medium: Option<MediumRegion>,
    curve_bvh: Option<prism_geometry::aabb_bvh::AabbBvh>,
}

/// `Mesh` 只多了两个 `Vec` 头，其余一个字节都没多。
///
/// 这条比"`size_of == 216`"强：写死数字只能抓住"总长变了"，抓不住
/// "总量没变但字段换了"（比如把 `triangles` 从 `Vec` 换成 `Box<[…]>`）。
const _: () = assert!(
    core::mem::size_of::<Mesh>()
        == core::mem::size_of::<MeshBeforeW1A>()
            + core::mem::size_of::<Vec<[f32; 2]>>()
            + core::mem::size_of::<Vec<[f32; 4]>>()
);

/// `Scene` = W1-A 之前的布局 + 四个 `Vec` 头（`textures` / `instances` /
/// `media` / `texels`）+ W3 的共享几何表与实例 TLAS 缓存。
///
/// 同一条纪律：旧字段一个都没增删、类型一个都没变（影子结构若与真实结构
/// 的字段集不一致，两者尺寸就不相等）。`medium` 仍在（旧路径保留）。
///
/// `texels` 是 W1-B 补的第四个头（§15.1 的偏离记录）：它是 `textures` 的
/// 像素 backing store，元素类型就是 §11 指定的 `array<vec4<f32>>` 的 CPU 侧
/// 对应物，故不构成新的 ABI 形态——断言在此显式记账，而不是悄悄少一项。
const _: () = assert!(
    core::mem::size_of::<Scene>()
        == core::mem::size_of::<SceneBeforeW1A>()
            + core::mem::size_of::<Vec<TextureRef>>()
            + core::mem::size_of::<Vec<Instance>>()
            + core::mem::size_of::<Vec<MediumRegion>>()
            + core::mem::size_of::<Vec<[f32; 4]>>()
            + core::mem::size_of::<Vec<(Mesh, MaterialKind)>>()
            + core::mem::size_of::<Option<prism_geometry::aabb_bvh::AabbBvh>>()
            + core::mem::size_of::<Option<crate::env::EnvCdf>>()
);


impl Scene {
    /// **W1-A 定桩：场景布局自检**（构造期断言的实际落点）。
    ///
    /// 返回 `Ok(())` 或第一条违规的可读描述。纯读、无分配、无浮点——
    /// 每帧调一次也不影响确定性（不进 `per-sample` 路径）。
    ///
    /// # 查什么
    ///
    /// 1. **[`Light::Environment`] 与 [`Light::Ambient`] 不可并存**
    ///    —— 两者都描述"逃逸射线看到的背景"，同时生效即双计
    ///    （选型报告 §3.1 第 4 条末句）。双计的表现是背景偏亮，
    ///    不会报错、只会被对拍抓到，所以必须在构造期拦。
    /// 2. `Environment` 的句柄必须落在 `textures` 范围内（或为 0 = 无）。
    /// 3. `Mesh::uv` / `Mesh::tangent` 非空时必须与 `positions` **同长**
    ///    —— 短一寸就是越界读，而越界读在 CPU 上是 panic、在 GPU 上是垃圾像素。
    /// 4. `Bsdf::maps` 的每个非 0 句柄必须落在 `textures` 范围内。
    ///
    /// # 为什么不 panic
    ///
    /// 调用方（[`Scene::assert_layout`] / 调度层）自己决定"报错还是继续"。
    /// 本函数只回答"合法吗"，与 `Scene` 的其余 API 同为纯函数。
    pub fn validate_layout(&self) -> Result<(), String> {
        let n_tex = self.textures.len() as u32;
        // 局部小工具：材质是否挂了贴图（只用于 3b 的判定）。
        fn has_maps_in(mat: &MaterialKind) -> bool {
            matches!(mat, MaterialKind::Pbr(b) if b.maps.iter().any(|&h| h != crate::texture::NO_TEXTURE))
        }
        // ---- 1/2：环境光互斥 + 句柄范围 ----
        let mut has_ambient = false;
        let mut has_env = false;
        for (i, l) in self.lights.iter().enumerate() {
            match l {
                Light::Ambient { .. } => has_ambient = true,
                Light::Environment { map } => {
                    has_env = true;
                    if !crate::texture::handle_in_range(*map, n_tex) {
                        return Err(format!(
                            "lights[{i}] 是 Light::Environment，map 句柄 {map} 越界\
                             （Scene::textures 有 {n_tex} 条，合法句柄为 1..={n_tex}，\
                             0 表示无贴图）；句柄 k 映射到 textures[k-1]"
                        ));
                    }
                }
                _ => {}
            }
        }
        if has_ambient && has_env {
            return Err(
                "Light::Ambient 与 Light::Environment 不可并存：两者都描述逃逸背景，\
                 同时生效会让背景双计（ambient_background 给常数 + 环境图给采样值）。\
                 二选一：要方向结构用 Environment，要平光用 Ambient"
                    .to_string(),
            );
        }
        // ---- 1b：Environment 灯必须带上已构建的 CDF ----
        //
        // 漏建的表现是"背景全黑、IBL 完全没生效"——不会报错、不会 panic，
        // 只会让人以为是场景没配好。故在构造期把它变成一条可读的错误。
        // 句柄 0（[`crate::texture::NO_TEXTURE`]）= 未配置贴图 ⇒ 惰性环境灯，
        // 不要求 CDF（与 W1-A 定桩期的"本变体不参与计算"口径兼容）。
        if has_env {
            let env_map = self
                .lights
                .iter()
                .find_map(|l| match l {
                    Light::Environment { map } => Some(*map),
                    _ => None,
                })
                .unwrap_or(crate::texture::NO_TEXTURE);
            if env_map != crate::texture::NO_TEXTURE {
                match &self.env_cdf {
                    None => {
                        return Err(
                            "Light::Environment 存在但 Scene::env_cdf 为空：CDF 未构建。\
                             用 Scene::add_environment(...)（一步到位）或 \
                             Scene::rebuild_env_cdf()（手工装配场景后补建）"
                                .to_string(),
                        );
                    }
                    Some(c) => {
                        if c.handle() != env_map {
                            return Err(format!(
                                "Scene::env_cdf 的句柄 {} 与 Light::Environment{{map: {env_map}}} \
                                 不一致（CDF 与灯光指向了不同的贴图）；\
                                 改灯后须重跑 Scene::rebuild_env_cdf()",
                                c.handle()
                            ));
                        }
                    }
                }
            }
        }
        // ---- 3：uv / tangent 与 positions 同长 ----
        for (i, (m, _)) in self.meshes.iter().enumerate() {
            if !m.uv.is_empty() && m.uv.len() != m.positions.len() {
                return Err(format!(
                    "meshes[{i}].uv 有 {} 条，但 positions 有 {} 条；\
                     顶点域 uv 必须与 positions 同长同序（否则按顶点下标取值会越界）",
                    m.uv.len(),
                    m.positions.len()
                ));
            }
            if !m.tangent.is_empty() && m.tangent.len() != m.positions.len() {
                return Err(format!(
                    "meshes[{i}].tangent 有 {} 条，但 positions 有 {} 条（同 uv 的约束）",
                    m.tangent.len(),
                    m.positions.len()
                ));
            }
        }
        // ---- 3b：有贴图的材质 + 无 uv 的网格 = 采样无定义 ----
        //
        // `Mesh::uv` 为空时 `uv_at` 只能返回 `[0,0]`（把整张贴图采样成
        // 单点），图像上看是"这张贴图被抹成纯色"。这不是运行时能修的，
        // 是场景装配错误 ⇒ 构造期拒绝。soup（`add_bvh`）同理。
        for (i, (m, mat)) in self.meshes.iter().enumerate() {
            if m.uv.is_empty() && has_maps_in(mat) {
                return Err(format!(
                    "meshes[{i}] 的材质挂了贴图，但该网格没有顶点 uv\
                     （positions 有 {} 条、uv 有 0 条）：采样点无定义。\
                     给 `Mesh::uv` 填上与 positions 同长的 uv，或改用无贴图材质",
                    m.positions.len()
                ));
            }
        }
        // ---- 4：Bsdf::maps 句柄范围 ----
        let check_maps = |mat: &MaterialKind, where_: &str| -> Result<(), String> {
            if let MaterialKind::Pbr(b) = mat {
                for slot in 0..crate::bsdf::MAP_SLOTS {
                    let h = b.maps[slot];
                    if !crate::texture::handle_in_range(h, n_tex) {
                        return Err(format!(
                            "{where_} 的 Bsdf::maps[{slot}] 句柄 {h} 越界\
                             （Scene::textures 有 {n_tex} 条，合法句柄为 1..={n_tex}，\
                             0 表示无贴图）；句柄 k 映射到 textures[k-1]"
                        ));
                    }
                }
            }
            Ok(())
        };
        for (i, (_, m)) in self.spheres.iter().enumerate() {
            check_maps(m, &format!("spheres[{i}]"))?;
        }
        for (i, (_, m)) in self.boxes.iter().enumerate() {
            check_maps(m, &format!("boxes[{i}]"))?;
        }
        for (i, (_, m)) in self.planes.iter().enumerate() {
            check_maps(m, &format!("planes[{i}]"))?;
        }
        for (i, (_, m)) in self.meshes.iter().enumerate() {
            check_maps(m, &format!("meshes[{i}]"))?;
        }
        // ---- 5：实例表（W3 落地，定桩时留的位置）----
        self.validate_instances()?;
        Ok(())
    }

    /// [`Scene::validate_layout`] 的 panic 版（构造期断言的调用形态）。
    ///
    /// # Panics
    ///
    /// 布局违规时 panic。违规是**编程错误**（场景构造错），不是用户输入错误，
    /// 与 `Bsdf::new` 的参数域断言同风格：早炸比晚出脏帧好。
    pub fn assert_layout(&self) {
        if let Err(msg) = self.validate_layout() {
            panic!("场景布局违规（W1-A 定桩断言）：{msg}");
        }
    }

    /// 场景**堆内存**估算（字节；`size_of::<Scene>()` 之外的部分）。
    ///
    /// # 覆盖口径
    ///
    /// 逐字段按 `Vec::capacity() × size_of::<T>()` 累加（`capacity` 而非 `len`
    /// ——容量才是真实占用的那块堆），外加 [`Scene::texels`] 与
    /// [`Scene::media`] 的像素/网格荷载。
    ///
    /// **A6 记账**：`media` 是 `Vec<MediumRegion>`，每个区域自带一个
    /// [`Medium`]；异质介质的 σt 场（[`crate::media::Grid3`]）是堆上的
    /// `Vec<f32>`，不数进去会低估几个数量级（256³ 的场是 64 MB）。故本函数
    /// 逐区域把 `medium.heap_bytes()` 加上。
    ///
    /// # 为什么需要它
    ///
    /// 多区域介质把"场景里有多少介质"从 0/1 变成 N；没有一处统一的记账，
    /// "10 个 256³ 的异质场"会静默吃掉 640 MB 而没人知道。本函数是唯一的
    /// 口径出口（测试 `a6_memory_bytes_counts_media` 钉住它随 `media` 增长）。
    ///
    /// # 不是什么
    ///
    /// 不是 RSS：`Vec` 的 capacity 可能大于实际使用，且分配器/碎片/共享几何
    /// 的引用计数都不在此列。是个**上界式估算**，用于"量级对不对"的判定。
    pub fn memory_bytes(&self) -> usize {
        // 单条 `Vec` 的堆占用（容量 × 元素大小；元素内若还含 `Vec`，由调用方另计）。
        fn vec_bytes<T>(v: &Vec<T>) -> usize {
            v.capacity() * core::mem::size_of::<T>()
        }
        let mut n: usize = 0;
        n += vec_bytes(&self.spheres);
        n += vec_bytes(&self.boxes);
        n += vec_bytes(&self.paraboloids);
        n += vec_bytes(&self.planes);
        n += vec_bytes(&self.cylinders);
        n += vec_bytes(&self.cones);
        n += vec_bytes(&self.toruses);
        n += vec_bytes(&self.revolutions);
        n += vec_bytes(&self.lights);
        n += vec_bytes(&self.textures);
        n += vec_bytes(&self.texels);
        n += vec_bytes(&self.instances);
        n += vec_bytes(&self.instance_geoms);
        // 网格：顶点 / 索引 / 三角形 / uv / tangent / BVH 节点都不是本函数能
        // 逐项数清的（`Mesh` 的内部布局是私有实现），按 `Mesh::memory_bytes`
        // 的口径取一次——若该口径不存在，则退化为"元素大小 × 数量"。
        for (m, _) in &self.meshes {
            n += m.memory_bytes();
        }
        for (m, _) in &self.instance_geoms {
            n += m.memory_bytes();
        }
        // 介质：单区域旧路径 + 多区域新路径都要记（A6）。
        if let Some(r) = &self.medium {
            n += core::mem::size_of::<MediumRegion>();
            n += r.medium.heap_bytes();
        }
        n += vec_bytes(&self.media);
        for r in &self.media {
            n += r.medium.heap_bytes();
        }
        n
    }

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
    // 几何实例构造入口（W3 / A3；C1 两级 BVH 的 TLAS 在这里维护）
    // -----------------------------------------------------------------------

    /// 注册**共享几何原型**（实例的 BLAS 源），返回 `geom_id`。
    ///
    /// 原型只经实例到达，**不**进 [`Scene::meshes`]（否则它会在局部坐标系
    /// 里被直接渲染一份，见 [`Scene::instance_geoms`] 的说明）。BVH 在这里
    /// 构建**一次** —— 之后无论多少条实例引用它，都不会再建。
    pub fn add_instance_geom(
        &mut self,
        positions: Vec<Vec3>,
        indices: Vec<u32>,
        material: MaterialKind,
    ) -> u32 {
        self.instance_geoms.push((Mesh::new(positions, indices), material));
        self.instance_geoms.len() as u32 - 1
    }

    /// 注册**已构建好的**共享几何原型（零重建；如 glTF/USD 导入器已建树）。
    ///
    /// 前置条件同 [`Mesh::from_prebuilt`]（`bvh` 必须由同一 `triangles` 构建）。
    pub fn add_instance_geom_mesh(&mut self, mesh: Mesh, material: MaterialKind) -> u32 {
        self.instance_geoms.push((mesh, material));
        self.instance_geoms.len() as u32 - 1
    }

    /// 共享几何原型条数。
    pub fn instance_geom_count(&self) -> usize {
        self.instance_geoms.len()
    }

    /// 共享几何三角形总数（**与实例数无关**的那部分内存的规模）。
    pub fn instance_geom_triangle_count(&self) -> usize {
        self.instance_geoms.iter().map(|(m, _)| m.tri_count()).sum()
    }

    /// 注册一条实例，返回实例下标（[`ShapeRef::Instance`] 的索引空间）。
    ///
    /// # 为什么不在这里重建 TLAS
    ///
    /// 逐条重建是 O(n² log n)：1 万实例 = 1 万次构建。故本函数只 push +
    /// **作废**缓存（置 `None`），批量装载请用 [`Scene::add_instances`]
    /// （一次重建）或事后调一次 [`Scene::rebuild_instance_bvh`]。
    /// 缓存缺失时 [`Scene::intersect`] 走线性兜底：结果一致，只是 O(n)。
    ///
    /// 前置条件：`instance.geom_id`（与可选的 `mat_id`）须落在
    /// `meshes` 范围内 —— 由 [`Scene::validate_layout`] 在构造期检查。
    pub fn add_instance(&mut self, instance: Instance) -> usize {
        self.instances.push(instance);
        self.instance_bvh = None; // 作废（不逐条重建）
        self.instances.len() - 1
    }

    /// 批量注册实例（**一次** TLAS 重建），返回下标区间。
    ///
    /// 1 万实例的构建时间是 O(n log n) 的毫秒级 —— "构建与实例数解耦"
    /// 指的是**几何**（BLAS 只建一次，见 [`Mesh::bvh`]），TLAS 仍是
    /// 每实例一个叶子的 n log n。
    pub fn add_instances(
        &mut self,
        batch: impl IntoIterator<Item = Instance>,
    ) -> std::ops::Range<usize> {
        let start = self.instances.len();
        for inst in batch {
            self.instances.push(inst);
        }
        self.rebuild_instance_bvh();
        start..self.instances.len()
    }

    /// 重建实例 TLAS（世界空间 AABB，SAH 确定性，与 `curve_bvh` 同款缓存）。
    ///
    /// 空实例表 ⇒ `instance_bvh = None`（与"从未构建"同义，`intersect`
    /// 不会走空树查询）。
    pub fn rebuild_instance_bvh(&mut self) {
        if self.instances.is_empty() {
            self.instance_bvh = None;
            return;
        }
        let mut aabbs: Vec<prism_geometry::bvh::Aabb> = Vec::with_capacity(self.instances.len());
        for inst in &self.instances {
            // 几何缺失/为空的实例给空盒（＝永不命中）。**不能**给退化点盒：
            // 那会让射线在错误的位置被剪枝。
            let aabb = match self.instance_geoms.get(inst.geom_id as usize) {
                Some((mesh, _)) => match mesh.bounds() {
                    Some((mn, mx)) => inst.world_aabb(mn, mx),
                    None => prism_geometry::bvh::Aabb::empty(),
                },
                None => prism_geometry::bvh::Aabb::empty(),
            };
            aabbs.push(aabb);
        }
        self.instance_bvh = Some(prism_geometry::aabb_bvh::AabbBvh::build(&aabbs));
    }

    /// 实例表专用校验（`validate_layout` 的子过程；纯读、无分配）。
    pub fn validate_instances(&self) -> Result<(), String> {
        let n_geom = self.instance_geoms.len() as u32;
        let n_mesh = self.meshes.len() as u32;
        for (i, inst) in self.instances.iter().enumerate() {
            if inst.geom_id >= n_geom {
                return Err(format!(
                    "instances[{i}].geom_id = {} 越界（Scene::instance_geoms 有 {n_geom} 条）；\
                     实例只**引用**共享几何，原型必须先经 Scene::add_instance_geom 注册",
                    inst.geom_id
                ));
            }
            if inst.mat_id != INSTANCE_MAT_FROM_GEOM && inst.mat_id >= n_mesh {
                return Err(format!(
                    "instances[{i}].mat_id = {} 越界（Scene::meshes 有 {n_mesh} 条，\
                     合法值为 0..{n_mesh} 或 u32::MAX = 材质取自共享几何）；\
                     材质覆盖源只能是 meshes 的条目",
                    inst.mat_id
                ));
            }
            if !inst.affine.iter().all(|v| v.is_finite())
                || !inst.inverse.iter().all(|v| v.is_finite())
            {
                return Err(format!(
                    "instances[{i}] 的 affine/inverse 含非有限分量（NaN/Inf）——\
                     射线进去就是 NaN 分支，不命中且污染累加缓冲"
                ));
            }
            if inst.is_identity() && inst.affine != crate::instance::IDENTITY_AFFINE {
                return Err(format!(
                    "instances[{i}] 打了 INSTANCE_FLAG_IDENTITY 但 affine 不是恒等 —— \
                     flags 与数据不符（identity 快路径会跳过变换，画面表现为几何错位）"
                ));
            }
            if inst.pad != 0 {
                return Err(format!(
                    "instances[{i}].pad = {}（保留字必须为 0；非 0 说明有人在槽里 \
                     偷偷塞了第 29 个字，112 B 的步长语义已被破坏）",
                    inst.pad
                ));
            }
            // 逆矩阵自洽（廉价版）：det(A)·det(A⁻¹) ≈ 1。
            // 抓不住逐元素转置/错位，但抓得住"忘了填逆"与"填了别人的逆"。
            let d = inst.affine_det() * {
                let b = &inst.inverse;
                b[0] * (b[5] * b[10] - b[6] * b[9])
                    + b[1] * (b[6] * b[8] - b[4] * b[10])
                    + b[2] * (b[4] * b[9] - b[5] * b[8])
            };
            if !(d.is_finite() && (d - 1.0).abs() <= 1e-2) {
                return Err(format!(
                    "instances[{i}] 的 affine 与 inverse 不自洽：det(A)·det(A⁻¹) = {d}（应 ≈ 1）\
                     —— 逆矩阵必须由 Instance::from_affine 生成，不许手填"
                ));
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 曲线图元族构造入口（add_* 维护曲线 BVH）
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
    /// 有限圆柱（下标见 [`Scene::cylinders`]）。
    Cylinder(usize),
    /// 有限圆锥（下标见 [`Scene::cones`]）。
    Cone(usize),
    /// 圆环（下标见 [`Scene::toruses`]）。
    Torus(usize),
    /// 旋转曲面（下标见 [`Scene::revolutions`]）。
    Revolution(usize),
    /// **几何实例**中的单个三角形：(实例下标, 共享几何的原始三角形下标)。
    ///
    /// W3（A3 实例化）新增。与 [`ShapeRef::Triangle`] 的区别只在**空间**：
    /// `Triangle` 的顶点已是世界坐标，`Instance` 的顶点在共享几何的局部
    /// 坐标系里，世界值要过实例的仿射。故两者不能合并——材质/豁免的解析
    /// 路径相同，法线路径不同（`Triangle` 取绕序法线，`Instance` 取
    /// **逆转置**后的绕序法线）。
    Instance(usize, usize),
}

/// 最近命中。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hit {
    /// 沿射线的命中距离。
    pub(crate) t: f32,
    /// 命中图元。
    pub(crate) shape: ShapeRef,
    /// 命中点的**表面参数坐标**（A1 / W1-B；[`Scene::uv_at`] 的口径）。
    ///
    /// # 谁填它
    ///
    /// 只有 [`Scene::intersect_uv`] 填真值；[`Scene::intersect`] 恒填
    /// `[0, 0]`（阴影射线 / 介质穿越不需要 uv，省一次 `uv_at`）。
    /// 两个入口的 `t` / `shape` **逐位相同**（`intersect_uv_matches_intersect`
    /// 钉住），差别只有这一个字段。着色路径必须用 `intersect_uv`。
    ///
    /// 只加 2 个 f32（选型报告 §2.2「不新增每图元状态」）：uv 由
    /// `(shape, p)` 纯函数重算，不在求交里逐候选维护。
    pub(crate) uv: [f32; 2],
}

impl Hit {
    /// 构造（uv 由 `intersect_uv` 事后填；此处恒 0 起点）。
    #[inline]
    pub(crate) const fn new(t: f32, shape: ShapeRef) -> Self {
        Hit { t, shape, uv: [0.0, 0.0] }
    }
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
            // 曲线图元族（本族）：各取所属条目材质。
            ShapeRef::Cylinder(i) => &self.cylinders[i].1,
            ShapeRef::Cone(i) => &self.cones[i].1,
            ShapeRef::Torus(i) => &self.toruses[i].1,
            ShapeRef::Revolution(i) => &self.revolutions[i].1,
            // 三角几何三变体同源：都取所属网格的材质（容器级引用与
            // 图元级命中语义一致——一个网格一个材质）。
            ShapeRef::Triangle(m, _) | ShapeRef::Mesh(m) | ShapeRef::Bvh(m) => &self.meshes[m].1,
            // 实例：材质来源由 `mat_id` 决定（哨兵 ⇒ 共享几何自带的材质）。
            // 越界不在这里兜——`validate_layout` 已在构造期拦掉；运行期越界
            // 会 panic，与 `Triangle(m,_)` 的越界行为一致。
            ShapeRef::Instance(i, _) => {
                let inst = &self.instances[i];
                match inst.material_override() {
                    Some(m) => &self.meshes[m as usize].1,
                    None => &self.instance_geoms[inst.geom_id as usize].1,
                }
            }
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
            // ---- 曲线图元族（本族）：由命中点 p 纯函数重判表面归属 ----
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
                // （对拍修复：dr<=dc 在 dc<0<dr 时恒假）。
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
            // 实例三角：局部绕序法线 → **逆转置** → 与 `triangle_unit_normal`
            // 同一条归一化公式。
            //
            // identity 走**完全相同的代码**（`triangle_unit_normal`）：
            // 逆转置后二次归一化不是逐位幂等的（`n·n` 归一化一次后不一定
            // 恰好等于 1.0，再归一化会挪末位），而判据要求 identity 与展平版
            // 逐位一致 —— 故 identity 这一支必须原样走，不做任何多余运算。
            ShapeRef::Instance(i, t) => {
                let Some(inst) = self.instances.get(i) else {
                    return Vec3::splat(0.0);
                };
                let Some(tri) = self
                    .instance_geoms
                    .get(inst.geom_id as usize)
                    .and_then(|(mesh, _)| mesh.triangles.get(t))
                else {
                    return Vec3::splat(0.0);
                };
                if inst.is_identity() {
                    return triangle_unit_normal(tri);
                }
                // 未归一化绕序法线 → 逆转置（含镜像反号）→ 归一化。
                let n = inst.inv_transpose_normal(tri.normal_unnormalized());
                let len2 = n.dot(&n);
                if len2 > 1e-24 {
                    n * (1.0 / len2.sqrt())
                } else {
                    Vec3::splat(0.0) // 退化（零面积或极端缩放）：绝不返回 NaN
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // W1-B：uv 参数化 / 解析足迹 / 贴图材质（A1 纹理管线的 CPU 侧）
    // -----------------------------------------------------------------
    //
    // 三个函数构成纹理管线的"几何侧"，全是 (shape, p) 的纯函数：
    //   `uv_at`        —— → uv（9 族 + 三角形重心插值）
    //   `uv_gradient`  —— uv 对世界切向的雅可比（有限差分，固定 ε）
    //   `footprint_at` —— 射线锥 × 雅可比 → uv 空间椭圆足迹
    // 纯函数 ⇒ integrator / wavefront / 三后端可共用同一份口径（不存在"两个
    // 各算一遍 uv"的第二事实源）。

    /// 表面参数坐标 `uv ∈ [0,1]²`（顶点域口径；选型报告 §2.2）。
    ///
    /// # 九族参数化（只依赖命中点与图元参数 ⇒ 纯函数）
    ///
    /// | 族 | u | v |
    /// |---|---|---|
    /// | 球 | `atan2(z,x)/2π + ½`（经度） | `acos(y/r)/π`（纬度，+Y 端 = 0） |
    /// | 盒 | 按主导轴选面（与 `normal_at` 同判据），面内两坐标 → `[0,1]` | 同左 |
    /// | 抛物面 | 方位角 | `r / r_max` |
    /// | 平面 | `½ + (p−p₀)·t_u / 2h`（无限平面取世界单位） | 同左 |
    /// | 柱 / 锥 | 方位角 | `(y+H)/2H` |
    /// | 环 | 主环方位角 | 管周角 `atan2(y, ρ−R)/2π + ½` |
    /// | 旋转面 | 方位角 | 轮廓**归一化弧长**（段内线性插值） |
    /// | 三角形 | 顶点 uv 的重心插值 | 同左 |
    ///
    /// # 退化约定
    ///
    /// - 极点 / 轴心：方位角测度零，`det_trig::atan2(0,0) = 0` ⇒ 极点落在
    ///   `u = ½`，不产生 NaN。
    /// - **无 uv 的网格**（soup / 未烘焙顶点 uv）返回 `[0,0]`。这种组合
    ///   （有贴图材质 + 无 uv 网格）由 [`Scene::validate_layout`] 在构造期
    ///   拒绝，故运行期不可达。
    pub(crate) fn uv_at(&self, s: ShapeRef, p: Vec3) -> [f32; 2] {
        /// 1/(2π)：方位角 → `[0,1]`。
        const FRAC_1_TAU: f32 = 0.5 * FRAC_1_PI;
        match s {
            ShapeRef::Sphere(i) => {
                let sp = &self.spheres[i].0;
                let d = (p - sp.center) * (1.0 / sp.radius);
                let u = det_trig::atan2(d.z, d.x) * FRAC_1_TAU + 0.5;
                let v = det_trig::acos(d.y.clamp(-1.0, 1.0)) * FRAC_1_PI;
                [u, v]
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
                // 与 `normal_at` 同一"主导轴"判据 ⇒ 法线所在面与 uv 所在面恒同。
                let (u, v) = if ax >= ay && ax >= az {
                    if rel.x >= 0.0 {
                        (-rel.z, rel.y)
                    } else {
                        (rel.z, rel.y)
                    }
                } else if ay >= az {
                    if rel.y >= 0.0 {
                        (rel.x, -rel.z)
                    } else {
                        (rel.x, rel.z)
                    }
                } else if rel.z >= 0.0 {
                    (rel.x, rel.y)
                } else {
                    (-rel.x, rel.y)
                };
                [0.5 + 0.5 * u, 0.5 + 0.5 * v]
            }
            ShapeRef::Paraboloid(i) => {
                let g = &self.paraboloids[i].0;
                let u = det_trig::atan2(p.y, p.x) * FRAC_1_TAU + 0.5;
                let r = (p.x * p.x + p.y * p.y).max(0.0).sqrt();
                let v = if g.r_max > 0.0 {
                    (r / g.r_max).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                [u, v]
            }
            ShapeRef::Plane(i) => {
                let pl = &self.planes[i].0;
                let (tu, tv) = plane_tangent_basis(pl.normal);
                let d = p - pl.point;
                let (a, b) = (d.dot(&tu), d.dot(&tv));
                match pl.half_extent {
                    // 有限切片：半宽 h ⇒ 边长 2h 铺满 [0,1]。
                    Some(h) if h > 0.0 => [0.5 + a / (2.0 * h), 0.5 + b / (2.0 * h)],
                    // 无限平面：世界 1 单位 = 1 次平铺。
                    _ => [a, b],
                }
            }
            ShapeRef::Cylinder(i) => {
                let c = &self.cylinders[i].0;
                let pl = p - c.center;
                let u = det_trig::atan2(pl.z, pl.x) * FRAC_1_TAU + 0.5;
                let v = if c.half_height > 0.0 {
                    (pl.y + c.half_height) / (2.0 * c.half_height)
                } else {
                    0.0
                };
                [u, v]
            }
            ShapeRef::Cone(i) => {
                let c = &self.cones[i].0;
                let pl = p - c.center;
                let u = det_trig::atan2(pl.z, pl.x) * FRAC_1_TAU + 0.5;
                let v = if c.half_height > 0.0 {
                    (pl.y + c.half_height) / (2.0 * c.half_height)
                } else {
                    0.0
                };
                [u, v]
            }
            ShapeRef::Torus(i) => {
                let t = &self.toruses[i].0;
                let pl = p - t.center;
                let rho = xz_len(pl);
                let u = det_trig::atan2(pl.z, pl.x) * FRAC_1_TAU + 0.5;
                // 管周角：(ρ−R, y) 平面内的极角。
                let v = det_trig::atan2(pl.y, rho - t.major) * FRAC_1_TAU + 0.5;
                [u, v]
            }
            ShapeRef::Revolution(i) => {
                let rev = &self.revolutions[i].0;
                let pl = p - rev.center;
                let rho = xz_len(pl);
                let u = det_trig::atan2(pl.z, pl.x) * FRAC_1_TAU + 0.5;
                let n_seg = rev.segment_count();
                if n_seg == 0 {
                    return [u, 0.0];
                }
                // 单遍：累计弧长 + 最近段的段内投影 ⇒ v = 归一化弧长。
                let mut arc = 0.0f32;
                let mut best_d2 = f32::INFINITY;
                let mut best_s = 0.0f32;
                for j in 0..n_seg {
                    let a = rev.profile[j];
                    let b = rev.profile[j + 1];
                    let (dr, dy) = (b[0] - a[0], b[1] - a[1]);
                    let len = (dr * dr + dy * dy).max(0.0).sqrt();
                    let d2 = point_segment_dist2(rho, pl.y, a[0], a[1], b[0], b[1]);
                    // 严格小于 ⇒ 并列取小索引（确定性）。
                    if d2 < best_d2 {
                        best_d2 = d2;
                        let tseg = if len > 1e-12 {
                            (((rho - a[0]) * dr + (pl.y - a[1]) * dy) / (len * len))
                                .clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        best_s = arc + len * tseg;
                    }
                    arc += len;
                }
                let v = if arc > 1e-12 { (best_s / arc).clamp(0.0, 1.0) } else { 0.0 };
                [u, v]
            }
            ShapeRef::Triangle(m, t) => self.mesh_uv_at(m, t, p),
            ShapeRef::Mesh(m) | ShapeRef::Bvh(m) => {
                let cp = self.meshes[m].0.bvh.closest_point(p);
                if cp.tri == u32::MAX {
                    [0.0, 0.0]
                } else {
                    self.mesh_uv_at(m, cp.tri as usize, p)
                }
            }
            // W3 实例（A3 × A1 接缝）：世界点先回局部，再按原型几何插值。
            ShapeRef::Instance(i, t) => {
                let Some(inst) = self.instances.get(i) else {
                    return [0.0, 0.0];
                };
                let Some((mesh, _)) = self.instance_geoms.get(inst.geom_id as usize) else {
                    return [0.0, 0.0];
                };
                // identity 原样走（与 `normal_at` 同纪律）：`M⁻¹·p ≡ p` 逐位，
                // 不做多余运算 —— W3 的「identity 与展平版逐位一致」判据依赖这点。
                let lp = if inst.is_identity() { p } else { inst.inv_point(p) };
                Self::mesh_local_uv(mesh, t, lp)
            }
        }
    }

    /// 三角形上按**重心坐标**插值顶点 uv（`Mesh::uv` 未烘焙 ⇒ `[0,0]`）。
    ///
    /// 重心坐标由命中点重算（与 `normal_at` 同款"纯函数重判"口径）——
    /// 不在 `Hit` 里存重心，避免给每个候选求交加 3 个 f32。
    fn mesh_uv_at(&self, m: usize, t: usize, p: Vec3) -> [f32; 2] {
        let Some((mesh, _)) = self.meshes.get(m) else {
            return [0.0, 0.0];
        };
        Self::mesh_local_uv(mesh, t, p)
    }

    /// 在**给定网格**上按重心坐标插值顶点 uv（`p` 为该网格的局部坐标）。
    ///
    /// 与 [`Scene::mesh_uv_at`] 的唯一差别是网格来源：实例的原型几何不在
    /// [`Scene::meshes`] 里（存于 [`Scene::instance_geoms`]），但插值口径
    /// 必须是同一份 —— 否则 glTF 场景在「展平」与「实例化」两条路径上
    /// 贴图会错位，而 W3 的 identity 判据正是要求两者逐位一致。
    fn mesh_local_uv(mesh: &Mesh, t: usize, p: Vec3) -> [f32; 2] {
        if mesh.uv.is_empty() || mesh.indices.len() < 3 {
            return [0.0, 0.0]; // 无 uv：由 validate_layout 拦在构造期
        }
        let Some(tri) = mesh.triangles.get(t) else {
            return [0.0, 0.0];
        };
        let base = t * 3;
        let (i0, i1, i2) = (
            mesh.indices[base] as usize,
            mesh.indices[base + 1] as usize,
            mesh.indices[base + 2] as usize,
        );
        let (Some(&u0), Some(&u1), Some(&u2)) = (
            mesh.uv.get(i0),
            mesh.uv.get(i1),
            mesh.uv.get(i2),
        ) else {
            return [0.0, 0.0];
        };
        let (a, b, c) = (tri.a, tri.b, tri.c);
        // 重心坐标（与 Möller–Trumbore 同一投影，退化面积 ⇒ 取首顶点）。
        let v0 = b - a;
        let v1 = c - a;
        let v2 = p - a;
        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let d20 = v2.dot(&v0);
        let d21 = v2.dot(&v1);
        let den = d00 * d11 - d01 * d01;
        if den.abs() <= 1e-30 {
            return u0;
        }
        let w1 = (d11 * d20 - d01 * d21) / den;
        let w2 = (d00 * d21 - d01 * d20) / den;
        let w0 = 1.0 - w1 - w2;
        [
            u0[0] * w0 + u1[0] * w1 + u2[0] * w2,
            u0[1] * w0 + u1[1] * w1 + u2[1] * w2,
        ]
    }

    /// uv 对世界切向的雅可比：返回 `(∇_T1 uv, ∇_T2 uv)`（每项 `∂u/∂s, ∂v/∂s`）。
    ///
    /// # 为什么用有限差分
    ///
    /// `uv_at` 对九族是解析可导的，但逐族写导数 = 九份必须与 `uv_at` 同步维护
    /// 的第二实现（改一处漏一处就是"贴图错位"这类难查 bug）。有限差分把
    /// "导数"重新变回 `uv_at` 的单点调用：**只有一份 uv 定义**。代价是
    /// 每个命中多 4 次 `uv_at`（着色路径才付，阴影射线不付）。
    ///
    /// `ε = 1e-3·max(1, |p|)`：与场景尺度同步的相对步长。差分只影响 mip 级
    /// 选择（log2 尺度，对 1e-3 的相对误差完全不敏感）。
    fn uv_gradient(&self, s: ShapeRef, p: Vec3, t1: Vec3, t2: Vec3) -> ([f32; 2], [f32; 2]) {
        let eps = 1e-3 * p.length().max(1.0);
        let g = |t: Vec3| -> [f32; 2] {
            let a = self.uv_at(s, p + t * eps);
            let b = self.uv_at(s, p - t * eps);
            // u 是周期量（跨接缝时 atan2 会跳 1.0）：差分化前先取最短环绕差。
            let du = wrap_delta(a[0] - b[0]);
            let dv = a[1] - b[1];
            [du / (2.0 * eps), dv / (2.0 * eps)]
        };
        (g(t1), g(t2))
    }

    /// 命中点的 uv 空间**椭圆足迹**（A1 各向异性的全部输入）。
    ///
    /// # 模型：射线锥 × 表面投影 × uv 雅可比
    ///
    /// 1. 该样本携带的射线锥半角 `spread`（主射线 = 一像素的角宽；非镜面弹射
    ///    后按瓣宽增长），在命中距离 `t` 处张开成半径 `r = t·spread` 的圆盘；
    /// 2. 圆盘投影到切平面：沿"视线投影方向"拉长 `1/cosθ_i`（掠射端就是
    ///    各向异性的来源，ρ = 1/cosθ_i），垂直方向仍是 `r`；
    /// 3. 两个世界半轴经 uv 雅可比映射到 uv 空间，得到两列向量
    ///    （[`Footprint`]）；真正的长短轴由采样器做 2×2 奇异值分解。
    ///
    /// # 为什么不用 Igehy 式完整射线微分
    ///
    /// 完整微分要在每次弹射传播 `dP/dx, dD/dx`（4 个 `Vec3`/射线），且反射
    /// Jacobian 需要 `dn/du`——三角形面上它**恒为 0**、在棱边**不连续**，
    /// 而棱边正是 mip 选择最要紧的地方。锥模型只需要 1 个 `f32`，且在
    /// 面上 C⁰ 连续。二弹射之后的 `spread` 是启发式（见 `trace_one` 的
    /// 弹射处），对主射线（纹理判据全部走主射线）是精确的。
    pub(crate) fn footprint_at(
        &self,
        s: ShapeRef,
        p: Vec3,
        n: Vec3,
        view: Vec3,
        t_hit: f32,
        spread: f32,
    ) -> Footprint {
        // T1 = 视线在切平面上的投影（足迹长轴方向）；退化时取任意切向。
        let vt = view - n * view.dot(&n);
        let vl = vt.length();
        let t1 = if vl > 1e-6 {
            vt * (1.0 / vl)
        } else {
            plane_tangent_basis(n).0
        };
        let t2 = n.cross(&t1);
        // 入射余弦：掠射端钳到 0.02（ρ ≤ 50），避免除零把足迹炸成 inf。
        let cos_i = view.dot(&n).abs().clamp(0.02, 1.0);
        let r = (t_hit * spread).max(0.0);
        let a = r / cos_i; // 长轴半宽（沿 T1）
        let b = r; // 短轴半宽（沿 T2）
        let (j1, j2) = self.uv_gradient(s, p, t1, t2);
        // 世界 → uv 的链式法则：列向量 = J·T_k × 半宽。
        Footprint::new([j1[0] * a, j1[1] * a], [j2[0] * b, j2[1] * b])
    }

    /// 着色切基 `(T, B)`（法线贴图用；`B = n × T`）。
    ///
    /// 网格带烘焙切线（`Mesh::tangent`，Lengyel 法，W0-A 定桩）时取重心插值；
    /// 否则用 uv 雅可比反解 `dP/du = J⁻¹·(1,0)`——两条路径都给出与 uv 一致的
    /// 切向，故同一个法线贴图在解析图元与网格上指向同一方向。
    pub(crate) fn shading_frame(&self, s: ShapeRef, p: Vec3, n: Vec3) -> (Vec3, Vec3) {
        // 优先：网格烘焙切线（美术权威，且与 uv 缝一致）。
        if let ShapeRef::Triangle(m, t) = s {
            if let Some((mesh, _)) = self.meshes.get(m) {
                if !mesh.tangent.is_empty() && mesh.indices.len() >= (t + 1) * 3 {
                    let base = t * 3;
                    let (i0, i1, i2) = (
                        mesh.indices[base] as usize,
                        mesh.indices[base + 1] as usize,
                        mesh.indices[base + 2] as usize,
                    );
                    if let (Some(&g0), Some(&g1), Some(&g2)) = (
                        mesh.tangent.get(i0),
                        mesh.tangent.get(i1),
                        mesh.tangent.get(i2),
                    ) {
                        // 重心权重与 uv 同一投影（复用 `mesh_uv_at` 的算子）。
                        let (w0, w1, w2) = self.barycentric(m, t, p);
                        let tv = Vec3::new(
                            g0[0] * w0 + g1[0] * w1 + g2[0] * w2,
                            g0[1] * w0 + g1[1] * w1 + g2[1] * w2,
                            g0[2] * w0 + g1[2] * w1 + g2[2] * w2,
                        );
                        let handed = if g0[3] >= 0.0 { 1.0 } else { -1.0 };
                        let t_ortho = tv - n * tv.dot(&n);
                        if t_ortho.length() > 1e-6 {
                            let t_unit = t_ortho.normalize();
                            return (t_unit, n.cross(&t_unit) * handed);
                        }
                    }
                }
            }
        }
        // 兜底：从 uv 雅可比反解 dP/du。
        let vt = if n.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let t1 = (vt - n * vt.dot(&n)).normalize();
        let t2 = n.cross(&t1);
        let (j1, j2) = self.uv_gradient(s, p, t1, t2);
        // J = [[j1.x, j2.x],[j1.y, j2.y]]（列 = 沿 T1/T2 的 uv 变化率）
        let det = j1[0] * j2[1] - j1[1] * j2[0];
        if det.abs() < 1e-12 {
            return (t1, t2);
        }
        // J⁻¹·(1,0) = (j2.y, −j2.x)/det ⇒ dP/du = (j2.y·T1 − j2.x·T2)/det。
        let dpdu = t1 * (j2[1] / det) - t2 * (j2[0] / det);
        if dpdu.length() > 1e-6 {
            let tu = dpdu.normalize();
            (tu, n.cross(&tu))
        } else {
            (t1, t2)
        }
    }

    /// 三角形重心权重 `(w0, w1, w2)`（与 `mesh_uv_at` 同一投影）。
    fn barycentric(&self, m: usize, t: usize, p: Vec3) -> (f32, f32, f32) {
        let Some((mesh, _)) = self.meshes.get(m) else {
            return (1.0, 0.0, 0.0);
        };
        let Some(tri) = mesh.triangles.get(t) else {
            return (1.0, 0.0, 0.0);
        };
        let (a, b, c) = (tri.a, tri.b, tri.c);
        let v0 = b - a;
        let v1 = c - a;
        let v2 = p - a;
        let d00 = v0.dot(&v0);
        let d01 = v0.dot(&v1);
        let d11 = v1.dot(&v1);
        let d20 = v2.dot(&v0);
        let d21 = v2.dot(&v1);
        let den = d00 * d11 - d01 * d01;
        if den.abs() <= 1e-30 {
            return (1.0, 0.0, 0.0);
        }
        let w1 = (d11 * d20 - d01 * d21) / den;
        let w2 = (d00 * d21 - d01 * d20) / den;
        (1.0 - w1 - w2, w1, w2)
    }

    // ---- 纹理构造入口（CPU 侧像素 → Scene::textures + Scene::texels）----

    /// 注册一张 8 位 RGBA 位图（sRGB 源在**此处**解码），返回**句柄**（从 1 起）。
    ///
    /// mip 链自动生成（盒式，固定求和序）。句柄 → `textures[handle-1]`，
    /// 由 [`crate::texture::handle_index`] 单点定义。
    pub fn add_texture_rgba8(&mut self, w: u32, h: u32, srgb: bool, pixels: &[u8]) -> u32 {
        let offset = self.texels.len() as u32;
        let (r, texels) = crate::texture::build_rgba8(w, h, srgb, offset, pixels);
        self.textures.push(r);
        self.texels.extend_from_slice(&texels);
        self.textures.len() as u32
    }

    /// 注册一张线性 f32 位图（HDR / 程序化贴图路径），返回句柄（从 1 起）。
    pub fn add_texture_linear(&mut self, w: u32, h: u32, pixels: &[[f32; 4]]) -> u32 {
        let offset = self.texels.len() as u32;
        let (r, texels) = crate::texture::build_linear_f32(w, h, false, offset, pixels);
        self.textures.push(r);
        self.texels.extend_from_slice(&texels);
        self.textures.len() as u32
    }

    /// **A2 的唯一推荐入口**：注册 HDR 等距柱状环境图 + 装灯 + 建 CDF（一步到位）。
    ///
    /// 三步必须原子完成，否则会出现三种"看起来能跑"的错误配置：
    /// 有灯无图（黑背景）、有图无灯（IBL 不生效）、灯图句柄错位（采到别的贴图）。
    /// 本函数把三者绑成一次调用，并顺手调 [`Scene::validate_layout`] 自检。
    ///
    /// `pixels` 为行主序 `W×H` 的线性域 `[r,g,b,a]`（`v = 0` 在 +Y 北极）。
    /// 返回环境图句柄（≥ 1）。
    ///
    /// # Panics
    ///
    /// `pixels.len() != w·h`（形状不符是编程错误，同 [`crate::texture::build_linear_f32`]）。
    pub fn add_environment(&mut self, w: u32, h: u32, pixels: &[[f32; 4]]) -> u32 {
        let handle = self.add_texture_linear(w, h, pixels);
        self.lights.push(Light::Environment { map: handle });
        // 构建失败只可能是"刚注册的贴图取不到"——那是内部一致性错误，
        // 用 assert 早炸（validate_layout 也会兜住）。
        self.rebuild_env_cdf()
            .unwrap_or_else(|e| panic!("刚注册的环境图无法建 CDF：{e}"));
        handle
    }

    /// 重建环境图 CDF（手工装配 `Scene::lights` / `textures` 之后补调）。
    ///
    /// * 没有 [`Light::Environment`] ⇒ 清空缓存并返回 `Ok(())`（幂等）；
    /// * 有灯但句柄取不到贴图 ⇒ `Err`（不静默变黑）。
    pub fn rebuild_env_cdf(&mut self) -> Result<(), String> {
        let Some(handle) = self.lights.iter().find_map(|l| match l {
            Light::Environment { map } => Some(*map),
            _ => None,
        }) else {
            self.env_cdf = None;
            return Ok(());
        };
        let table = TexTable::new(&self.textures, &self.texels);
        match crate::env::EnvCdf::build(handle, table) {
            Some(c) => {
                self.env_cdf = Some(c);
                Ok(())
            }
            None => Err(format!(
                "Light::Environment 的句柄 {handle} 取不到可用的等距柱状贴图\
                 （textures 有 {} 条、texels 有 {} 个纹素）",
                self.textures.len(),
                self.texels.len()
            )),
        }
    }

    /// 环境 CDF 视图（`None` = 场景没有环境灯）。
    #[inline]
    pub fn env(&self) -> Option<&crate::env::EnvCdf> {
        self.env_cdf.as_ref()
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
    // 曲线图元族求交（闭式解析，口径与 sphere_t 家族一致：
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

    /// 最近命中（可跳过指定形状——阴影射线对被采样光源豁免），**不填 uv**。
    ///
    /// 阴影射线 / 介质穿越 / 可见性判定的入口：`Hit.uv` 恒 `[0, 0]`。
    /// 需要 uv 的着色路径用 [`Scene::intersect_uv`]（两者 `t`/`shape` 逐位相同）。
    #[inline]
    pub(crate) fn intersect(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
    ) -> Option<Hit> {
        self.intersect_impl(ro, rd, t_max, skip, false)
    }

    /// 最近命中 + **填 uv**（着色路径入口；多一次 [`Scene::uv_at`] 调用）。
    #[inline]
    pub(crate) fn intersect_uv(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
    ) -> Option<Hit> {
        self.intersect_impl(ro, rd, t_max, skip, true)
    }

    /// 两个公开入口的公共实现（`want_uv` 只在末尾分支，不进候选循环）。
    #[inline]
    fn intersect_impl(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
        want_uv: bool,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut limit = t_max;
        for i in 0..self.spheres.len() {
            let r = ShapeRef::Sphere(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.sphere_t(i, ro, rd, limit) {
                best = Some(Hit::new(t, r));
                limit = t;
            }
        }
        for i in 0..self.boxes.len() {
            let r = ShapeRef::Box(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.box_t(i, ro, rd, limit) {
                best = Some(Hit::new(t, r));
                limit = t;
            }
        }
        for i in 0..self.paraboloids.len() {
            let r = ShapeRef::Paraboloid(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.paraboloid_t(i, ro, rd, limit) {
                best = Some(Hit::new(t, r));
                limit = t;
            }
        }
        for i in 0..self.planes.len() {
            let r = ShapeRef::Plane(i);
            if skip == Some(r) {
                continue;
            }
            if let Some(t) = self.plane_t(i, ro, rd, limit) {
                best = Some(Hit::new(t, r));
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
                best = Some(Hit::new(t, ShapeRef::Triangle(i, tri)));
                limit = t;
            }
        }
        // ---- 曲线图元族（测尾，既有族行为不变）----------------
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
                        best = Some(Hit::new(t, shape));
                        // 末族：limit 不再被读，无需回写（避免 unused 警告）。
                    }
                }
            } else {
                for i in 0..self.cylinders.len() {
                    if let Some(t) = self.cylinder_t(i, ro, rd, limit) {
                        best = Some(Hit::new(t, ShapeRef::Cylinder(i)));
                        limit = t;
                    }
                }
                for i in 0..self.cones.len() {
                    if let Some(t) = self.cone_t(i, ro, rd, limit) {
                        best = Some(Hit::new(t, ShapeRef::Cone(i)));
                        limit = t;
                    }
                }
                for i in 0..self.toruses.len() {
                    if let Some(t) = self.torus_t(i, ro, rd, limit) {
                        best = Some(Hit::new(t, ShapeRef::Torus(i)));
                        limit = t;
                    }
                }
                for i in 0..self.revolutions.len() {
                    if let Some(t) = self.revolution_t(i, ro, rd, limit) {
                        best = Some(Hit::new(t, ShapeRef::Revolution(i)));
                        // 末族：limit 不再被读，无需回写（避免 unused 警告）。
                    }
                }
            }
        }
        // ---- 几何实例（W3 / A3，末族：空表时本段一行不执行）----
        //
        // 位置放在最后不是随手排的：`limit` 是"running best"，同 `t` 下
        // **先到的族胜**。把实例族排在末位 ⇒ 实例与既有族同距时既有族优先，
        // 于是「往场景里加实例」不会改变既有几何的命中语义（只在实例确实
        // 更近时才接管）。
        if !self.instances.is_empty() {
            if let Some((t, i, tri)) = self.instance_nearest(ro, rd, limit, skip) {
                // 末族：`limit` 不再被读（无需回写，避免 unused 警告）。
                best = Some(Hit::new(t, ShapeRef::Instance(i, tri)));
            }
        }
        match best {
            Some(mut h) if want_uv => {
                h.uv = self.uv_at(h.shape, ro + rd * h.t);
                Some(h)
            }
            other => other,
        }
    }

    /// 实例族最近命中：`(t, 实例下标, 三角形下标)`。
    ///
    /// 两条路径（与曲线族同款设计）：
    /// * **TLAS 快路径**：`AabbBvh` 粗阶段（世界 AABB）+ 回调精阶段
    ///   （射线变换进局部空间 → 下层 BLAS）。命中按 `(t, 实例下标)`
    ///   字典序，故同 `t` 取最小实例下标。
    /// * **线性兜底**：`instance_bvh` 缺失或与实例数不符时逐条扫，
    ///   升序 + 严格更优 ⇒ 与 TLAS 路径同解（对拍测试钉住）。
    fn instance_nearest(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
    ) -> Option<(f32, usize, usize)> {
        if !finite_vec3(ro) || !finite_vec3(rd) {
            return None; // 非有限射线 ⇒ 实例族永不命中（与网格臂同语义）
        }
        let n = self.instances.len();
        let bvh_live = self.instance_bvh.as_ref().is_some_and(|b| b.prim_count() == n);
        if bvh_live {
            let bvh = self.instance_bvh.as_ref().expect("bvh_live 已判定存在");
            let mut cb = |id: u32| self.instance_t(id as usize, ro, rd, t_max, skip).map(|(t, _)| t);
            let (t, id) = bvh.ray_intersect(ro, rd, T_MIN, t_max, &mut cb)?;
            // 回调只回了 `t`（`AabbBvh` 的签名如此），三角形下标用**同参数
            // 再查一次**取回：同一实例、同一射线下 BVH 的返回值是确定的，
            // 且最小值不受 `t_max` 放松影响 ⇒ 二次查询必得同一个 `(t, tri)`。
            let (_, tri) = self.instance_t(id as usize, ro, rd, t_max, skip)?;
            return Some((t, id as usize, tri));
        }
        let mut best: Option<(f32, usize, usize)> = None;
        let mut limit = t_max;
        for i in 0..n {
            if let Some((t, tri)) = self.instance_t(i, ro, rd, limit, skip) {
                best = Some((t, i, tri));
                limit = t;
            }
        }
        best
    }

    /// 单条实例求交：世界射线 → 局部射线（**`t` 不变**）→ 下层 BLAS。
    ///
    /// 契约与 [`Scene::mesh_t`] 逐条对齐：`t >= T_MIN` 且 `t < t_max`，
    /// 被豁免三角形走 `t + T_MIN` 二次查询。区别只有两点：
    ///
    /// 1. 射线先过 `M⁻¹`（identity 实例原样返回，故 identity 路径与展平
    ///    路径**同码同值**）；
    /// 2. `Bvh::ray_intersect_raw` 不做"单位方向"断言 —— 局部方向故意不
    ///    归一化，`t` 才与世界参数共享（见 [`Instance::local_ray`]）。
    fn instance_t(
        &self,
        i: usize,
        ro: Vec3,
        rd: Vec3,
        t_max: f32,
        skip: Option<ShapeRef>,
    ) -> Option<(f32, usize)> {
        let inst = self.instances.get(i)?;
        let (mesh, _) = self.instance_geoms.get(inst.geom_id as usize)?;
        if mesh.triangles.is_empty() {
            return None;
        }
        let skip_tri = match skip {
            Some(ShapeRef::Instance(si, st)) if si == i => Some(st),
            _ => None,
        };
        let (lo, ld) = inst.local_ray(ro, rd);
        let first = mesh.bvh.ray_intersect_raw(lo, ld, T_MIN, t_max)?;
        if !(first.t < t_max) {
            return None; // 最近命中已在远端之外（或 NaN）
        }
        if skip_tri != Some(first.tri as usize) {
            return Some((first.t, first.tri as usize));
        }
        // 排除被豁免图元后重查：起点沿射线推进一个自交下界。
        // `t` 与世界参数共享 ⇒ 这一步在世界单位上推进 `T_MIN`，与
        // `mesh_t` 的 `t0 = first.t + T_MIN` 是同一个量。
        let t0 = first.t + T_MIN;
        if !(t0 < t_max) {
            return None;
        }
        let second = mesh.bvh.ray_intersect_raw(lo, ld, t0, t_max)?;
        if !(second.t < t_max) {
            return None;
        }
        Some((second.t, second.tri as usize))
    }

    /// 实例条数。
    pub fn instance_count(&self) -> usize {
        self.instances.len()
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
            // 曲线图元族（本族）：非球光源几何，未注册 NEE ⇒ 命中权重 1。
            ShapeRef::Cylinder(_)
            | ShapeRef::Cone(_)
            | ShapeRef::Torus(_)
            | ShapeRef::Revolution(_) => return None,
            // 三角几何（任意变体）：非球光源几何，未注册 NEE ⇒ 命中权重 1
            // （仅 BSDF 策略；发光网格既未注册 Light 也无立体角 pdf）。
            ShapeRef::Triangle(_, _) | ShapeRef::Mesh(_) | ShapeRef::Bvh(_) => return None,
            // 实例三角：同理（材质可能是 Emissive，但没有对应 `Light`
            // 注册 ⇒ 只有 BSDF 策略命中，MIS 权重 1）。
            ShapeRef::Instance(_, _) => return None,
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
    /// 当前射线锥的半角（弧度；A1 足迹的输入）。
    ///
    /// 主射线初值 = 一像素的角宽 `2·tan(fov/2)/height`；非镜面弹射后按瓣宽
    /// 增长（漫反射 ≈ 0.6 rad，粗糙瓣 ≈ α = roughness²），镜面/折射不变。
    /// 只有挂了贴图的材质才读它（无贴图场景零成本）。
    spread: f32,
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
///
/// 本函数只做**单区域**（旧路径 `Scene::medium`）的区间推导；自由程本体在
/// [`medium_collide_span`] 里，多区域（A6）复用同一份。两次 `segment_interval`
/// 的语义（含 `contains` 时 `t_enter = 0`）逐字保留，故旧场景的采样流与
/// 算术顺序一个字节都没变（V1 由 md5 钉死）。
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
    // B1 介质区域为单一凸盒：直线段至多穿越一次。均匀介质单次自由程抽样即完备
    // （一次均匀）；异质介质走 Woodcock delta tracking（步进 + 接受两个均匀，
    // 内部可能多步），故按 `needs_accept_uniform` 决定是否多抽一个——均匀
    // 路径的采样流消耗与接线前**逐字节一致**。
    let exit = b.segment_interval(ro, rd, t_enter, t_max)?.1;
    medium_collide_span(region, ro, rd, t_enter, exit, t_max, stream, aux_dim, steps)
}

/// 在**已算好**的子区间 `[t_enter, exit]` 内做一次自由程抽样（A6 的共用原语）。
///
/// 抽出这个函数的唯一理由是**多区域介质**：区域列表路径要把一条线段拆成
/// 若干子区间（每个区域一段），区间推导在 [`select_region`] 里完成，自由程
/// 判定必须与旧路径**逐位同式**——同一份 `needs_accept_uniform` 分支、同一
/// 顺序的 `aux_1d` 抽取、同一个 `t_enter + t_free < exit && < t_max` 判据。
///
/// `steps` 由调用方拥有：旧路径传路径级共享计数器（逐位兼容），多区域路径
/// 传**该区域自己的**计数器（A6 硬化要求，见 [`region_aux_dim`]）。
fn medium_collide_span(
    region: &MediumRegion,
    ro: Vec3,
    rd: Vec3,
    t_enter: f32,
    exit: f32,
    t_max: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    steps: &mut u32,
) -> Option<f32> {
    let t_free = if region.medium.needs_accept_uniform() {
        let us = stream.aux_1d(aux_dim, *steps);
        *steps += 1;
        let ua = stream.aux_1d(aux_dim, *steps);
        *steps += 1;
        region.medium.free_path(ro, rd, us, ua)
    } else {
        let u = stream.aux_1d(aux_dim, *steps);
        *steps += 1;
        region.medium.free_path(ro, rd, u, 0.0)
    };
    if t_enter + t_free < exit && t_enter + t_free < t_max {
        Some(t_enter + t_free)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// A6 / W6：多区域介质（C1「区域列表 + 优先级（不重叠）」）
// ---------------------------------------------------------------------------
//
// # 规格（选型报告 §7 定稿 C1）
//
// * 区域**不得重叠**。合规场景里每一段射线的每一个 `t` 至多属于一个区域，
//   故"选哪个区域"由几何唯一决定；`priority` 只在违规（重叠）时破同入点。
// * 一条线段可能穿过**多个不相交**的区域（串联板），因此不是"每段挑一个就完"，
//   而是**按入点升序逐子区间行进**：在每个子区间里各做一次自由程抽样，遇到
//   第一个碰撞事件即返回。这正是 V3（`exp(−σt₁d₁−σt₂d₂)`）能成立的机制。
// * **逐区域独立的 aux 步计数器 + 逐区域独立的 aux 维**：区域 k 的抽样流是
//   `aux_1d(base + k, j)`，`j` 从 0 起。于是"新增一个区域"**不会平移**任何
//   既有区域的抽样流（否则 V1 那一类逐位兼容在 media 场景里立刻失效）。

/// 多区域介质行进结果（A6）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MediaHit {
    /// 命中的区域下标（`Scene::media` 的下标）。
    pub region: usize,
    /// 碰撞点沿射线的参数（从 `ro` 起算）。
    pub t: f32,
    /// 该区域自己的 aux 步计数器在本段消耗到哪儿（散事件从这里继续抽）。
    pub steps: u32,
}

/// 区域 `k` 的 aux 维：`base + k`（**逐区域独立**，A6 硬化要求）。
///
/// `base` 是事件段 aux 维（`bounce_dim(ev, SLOT_AUX)`）或阴影臂基址（`+32`）。
/// 区域数 ≤ [`SHADOW_BASE_OFFSET`] 时，各区域的维互不相同、且与阴影臂不重叠。
/// 超过该数会与阴影臂别名——那是**统计相关性**（同一 `hash3` 的两个维被复用），
/// 不是确定性风险：同一场景两跑仍是逐字节相同。
///
/// `k` 大于 `u16` 表示范围的部分按 `u16::MAX` 截断（实际不可能：区域表长度
/// 受场景内存限制，且此处不引入任何 panic 路径）。
#[inline]
pub(crate) fn region_aux_dim(base: u16, k: usize) -> u16 {
    let kk: u16 = if k > usize::from(u16::MAX) { u16::MAX } else { k as u16 };
    base.wrapping_add(kk)
}

/// 阴影臂 / 体 NEE 的 aux 维相对散事件基址的偏移（见 [`region_aux_dim`]）。
pub(crate) const SHADOW_BASE_OFFSET: u16 = 32;

/// 选出 `[t0, t1]` 上与射线相交的**下一个**区域：返回 `(下标, 入点, 出点)`。
///
/// # 全序（这条是 V5 的判据本体）
///
/// 排序键：`t_enter` 升序 → `priority` 降序 → 下标升序。三个键都是整数/浮点
/// 的**严格比较**（无 epsilon、无哈希遍历序），故同一输入必得同一结果——这也
/// 是 [`Scene::media`] 允许"违规重叠"却不允许"结果不定"的兜底。
///
/// # 退化区间
///
/// `exit <= enter`（切向掠过 / 已被游标走过）不参与选择：它不构成任何物理
/// 子区间，选它会让行进循环空转。
///
/// # 公开性
///
/// `pub` 是为了让 V5（重叠区域下的唯一定值）能从集成测试直接探针，而不是
/// 只能间接从像素反推。
pub fn select_region(
    scene: &Scene,
    ro: Vec3,
    rd: Vec3,
    t0: f32,
    t1: f32,
) -> Option<(usize, f32, f32)> {
    let mut best: Option<(usize, f32, f32)> = None;
    for (k, region) in scene.media.iter().enumerate() {
        let Some((enter, exit)) = region.bounds.segment_interval(ro, rd, t0, t1) else {
            continue;
        };
        if exit <= enter {
            continue; // 退化：零长度子区间
        }
        let better: bool = match best {
            None => true,
            Some((bk, benter, _)) => {
                if enter < benter {
                    true
                } else if enter > benter {
                    false
                } else {
                    // 同一入点（合规场景里不存在，只有重叠才会出现）：
                    // 优先级高者胜；再相同则保留下标更小者（best 先到者）。
                    region.priority > scene.media[bk].priority
                }
            }
        };
        if better {
            best = Some((k, enter, exit));
        }
    }
    best
}

/// 沿 `[t_lo, t_hi]` 依次走遍所有介质区域的子区间，返回**第一个**碰撞事件。
///
/// 循环上界 `media.len() + 1`：每轮必然消费掉一个区域（游标前进到该区域出点，
/// 该区域下一次查询即为退化区间被跳过），故正常输入下轮数 = 穿过的区域数 +
/// 1。上界是防御性的——即使区间推导出现意料之外的几何，也只退化为"少走几段"，
/// 不会挂死（宪章 C18：不 panic、不死循环）。
///
/// `aux_dim` 是散事件段的 aux 维基址；区域 `k` 用 `aux_dim + k`，每个区域
/// 自带从 0 起的步计数器。
pub(crate) fn advance_multi(
    scene: &Scene,
    ro: Vec3,
    rd: Vec3,
    t_lo: f32,
    t_hi: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
) -> Option<MediaHit> {
    let mut cursor: f32 = t_lo;
    let budget: usize = scene.media.len() + 1;
    for _ in 0..budget {
        let (k, enter, exit) = select_region(scene, ro, rd, cursor, t_hi)?;
        let region: &MediumRegion = &scene.media[k];
        let mut steps: u32 = 0;
        let dim: u16 = region_aux_dim(aux_dim, k);
        if let Some(t) =
            medium_collide_span(region, ro, rd, enter, exit, t_hi, stream, dim, &mut steps)
        {
            return Some(MediaHit { region: k, t, steps });
        }
        if exit >= t_hi {
            return None; // 已到段末
        }
        cursor = exit;
    }
    None
}

/// 阴影射线可见性 = 几何遮挡 × 介质透射（二值），结果 ∈ {0.0, 1.0}。
///
/// # 多区域（A6）：连乘各区域的**独立**二值估计
///
/// `scene.media` 非空时走新分支：对**每一个**区域各做一次独立的二值估计
/// （各自从 `steps = 0` 起、各自用 `aux_dim + k` 这个维），再连乘。
///
/// 两条硬化要求写在这里而不是散在代码里：
///
/// 1. **连乘而非取一**：阴影射线穿过的每个区域都在衰减，漏掉任一区域就是
///    系统性漏光（V3 的串联板会直接抓到）；
/// 2. **逐区域独立**：区域 k 的抽样流只依赖 `(aux_dim + k, j)`。加一个区域
///    不会平移任何既有区域的流 —— 若共用一个步计数器，"前面多插一个区域"
///    就会把后面所有区域的 `u` 全部错位，逐位兼容立刻失效。
///
/// 调用方的 `steps`（路径级共享计数器）在本分支**不被推进**：多区域路径的
/// 全部抽样都在区域自己的计数器上，故区域数与场景其余部分的采样流无关。
/// 旧分支（`scene.medium`）一行未改，仍用调用方的 `steps`。
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
    if !scene.media.is_empty() {
        let t_end: f32 = max_t - T_MIN;
        let mut vis: f32 = 1.0;
        for (k, region) in scene.media.iter().enumerate() {
            // 区间推导与 `medium_collide` 同式（一次 `segment_interval`，
            // 端点已含 `max(0, t_near)` / `min(t_far, t_end)` 钳制）。
            let Some((enter, exit)) = region.bounds.segment_interval(ro, rd, 0.0, t_end) else {
                continue; // 该区域不在这条阴影射线上
            };
            if exit <= enter {
                continue; // 退化：切向掠过，不构成遮挡
            }
            // 阴影臂维基址：`aux_dim + SHADOW_BASE_OFFSET`。**不是**可选优化：
            // 若用 `aux_dim + k`，同一个事件的"主射线自由程"（`advance_multi`
            // 的区域 k，步 0）与"阴影射线自由程"会取到**同一个** `u`，而两者
            // 之间存在条件关系（主射线没碰撞 ⇔ u 偏大）⇒ 阴影透射率有偏。
            // 实测：单区域 σt=1、albedo=0 时 E[vis] 应为 0.108，复用同一 `u`
            // 会给出 0.30（偏亮 2.7×）。
            let dim: u16 = region_aux_dim(aux_dim.wrapping_add(SHADOW_BASE_OFFSET), k);
            let mut ksteps: u32 = 0;
            if medium_collide_span(region, ro, rd, enter, exit, t_end, stream, dim, &mut ksteps)
                .is_some()
            {
                vis = 0.0;
            }
        }
        return vis;
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
// A2 / W2：环境贴图的 NEE（CDF 策略）与逃逸权重
// ---------------------------------------------------------------------------

/// 环境灯的表面 NEE：按 CDF 采一个方向 → 阴影射线 → MIS 加权贡献。
///
/// # 与球光/矩形光 NEE 的同构关系
///
/// `sample_nee` 对 `Light::Environment` 返回 `None`（它拿不到 CDF），
/// 故环境这一臂由本函数承担：**采样**换成 CDF 反演，**pdf** 换成
/// `EnvCdf::pdf_w`，其余（BSDF 求值、备择 pdf、可见性、MIS 权重）逐项同构。
///
/// `w = nee_mis_weight(light, pdf_env, pdf_bsdf)` —— 与 [`Light::Sphere`]
/// **同臂**（power heuristic）：环境是"BSDF 采样可达的几何"。
///
/// 返回**未钳制**的贡献（调用方按 `clamp_indirect` 口径处理）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn nee_env_surface(
    scene: &Scene,
    env: &crate::env::EnvCdf,
    table: TexTable<'_>,
    light: &Light,
    p: Vec3,
    nv: Vec3,
    wo: Vec3,
    mat: &MaterialKind,
    bsdf: &Bsdf,
    throughput: Vec3,
    u1: f32,
    u2: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    aux_steps: &mut u32,
) -> Vec3 {
    let Some(s) = env.sample(u1, u2) else {
        return Vec3::splat(0.0); // 全黑环境
    };
    let cos_l = nv.dot(&s.wi);
    if cos_l <= 0.0 || !(s.pdf_w > 0.0) {
        return Vec3::splat(0.0);
    }
    let f = surface_brdf(mat, bsdf, wo, s.wi, nv);
    let f_cos = hadamard(f, Vec3::splat(cos_l));
    let alt_pdf = surface_pdf(mat, bsdf, wo, s.wi, nv);
    let radiance = env.radiance(table, s.wi);
    if !(radiance.x > 0.0 || radiance.y > 0.0 || radiance.z > 0.0) {
        return Vec3::splat(0.0);
    }
    let ro = p + nv * SURFACE_EPS;
    let vis = visibility(scene, ro, s.wi, f32::INFINITY, None, stream, aux_dim, aux_steps);
    if vis <= 0.0 {
        return Vec3::splat(0.0);
    }
    let w = nee_mis_weight(light, s.pdf_w, alt_pdf);
    hadamard(hadamard(throughput, f_cos), radiance) * (vis * w / s.pdf_w)
}

/// 环境灯的体积 NEE（相函数版）：介质内散射点的直接照明。
///
/// 语义与表面版同构，只把 `f·cosθ` 换成相函数 `ph`、备择 pdf 换成 `ph`。
/// 不接这一臂会让"介质 + 环境"的 MIS 权重永远拿不到 NEE 那一半 ⇒ 系统性欠估。
#[allow(clippy::too_many_arguments)]
pub(crate) fn nee_env_volume(
    scene: &Scene,
    env: &crate::env::EnvCdf,
    table: TexTable<'_>,
    light: &Light,
    p: Vec3,
    w_in: Vec3,
    throughput: Vec3,
    region: &MediumRegion,
    u1: f32,
    u2: f32,
    stream: &SampleStream<'_>,
    aux_dim: u16,
    aux_steps: &mut u32,
) -> Vec3 {
    let Some(s) = env.sample(u1, u2) else {
        return Vec3::splat(0.0);
    };
    if !(s.pdf_w > 0.0) {
        return Vec3::splat(0.0);
    }
    let cos_l = w_in.dot(&s.wi);
    let ph = region.medium.phase(cos_l);
    let radiance = env.radiance(table, s.wi);
    if !(radiance.x > 0.0 || radiance.y > 0.0 || radiance.z > 0.0) {
        return Vec3::splat(0.0);
    }
    let vis = visibility(scene, p, s.wi, f32::INFINITY, None, stream, aux_dim, aux_steps);
    if vis <= 0.0 {
        return Vec3::splat(0.0);
    }
    let w = nee_mis_weight(light, s.pdf_w, ph);
    hadamard(throughput, radiance) * (ph * vis * w / s.pdf_w)
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
    // 一像素的角宽（竖直方向）：主射线的锥半角（A1 足迹的初值）。
    // 与 `Camera::pixel_dir` 同口径 —— 该函数把 ndc_y 乘 `tan_half_fov`，
    // 故相邻像素的方向差 ≈ `2·tan_half_fov / height`（小角近似即角宽本身）。
    let spread0 = 2.0 * cam.tan_half_fov / (resolution.1.max(1) as f32);
    let mut st = PathState {
        throughput: Vec3::new(1.0, 1.0, 1.0),
        radiance: Vec3::new(0.0, 0.0, 0.0),
        prev_pdf_dir: 0.0,
        prev_specular: true, // 相机段视为镜面链起点（直视发光权重 1）
        spread: spread0,
    };
    let mut aux_steps: u32 = 0;
    // bounce = 已完成事件数；本次线段远端事件编号 ev = bounce + 1。
    for bounce in 0..params.max_bounces {
        let ev = bounce + 1;
        let aux_dim = bounce_dim(ev, SLOT_AUX);
        let hit = scene.intersect_uv(ray.origin, ray.dir, MISS_T, None);
        let t_surf = hit.map_or(MISS_T, |h| h.t);

        // ---- 介质段（相机/续弹段统一；碰撞事件优先于表面事件）----
        //
        // 两条通路**严格互斥**（A6 硬化要求，选型报告 §7）：
        // `media` 非空 ⇒ 新路径；`media.is_empty()` ⇒ 原分支、**一行不改**。
        // 不允许"空列表也走新路径"——那会把旧场景的共享 aux 步计数器换成
        // 逐区域计数器，采样流立刻平移，逐位兼容当场失效（V1）。
        if !scene.media.is_empty() {
            // 逐子区间行进：区域序由入点决定（`select_region` 全序），
            // 一段射线可以顺次穿过多个不相交区域（V3 的串联板）。
            if let Some(hit) =
                advance_multi(scene, ray.origin, ray.dir, T_MIN, t_surf, stream, aux_dim)
            {
                let region: &MediumRegion = &scene.media[hit.region];
                let p: Vec3 = ray.at(hit.t);
                // 该区域**自己的** aux 流：维 = aux_dim + 区域下标，步从这里续。
                let rdim: u16 = region_aux_dim(aux_dim, hit.region);
                let rsteps: u32 = hit.steps;
                // 散射 vs 吸收（同一区域的下一步）。这是本区域本段的**最后一次**
                // 抽取：散射之后射线已离开该区域的子区间，后续抽样全在阴影臂的
                // 独立基址上，故不必回写计数器。
                let u_scatter = stream.aux_1d(rdim, rsteps);
                if u_scatter >= region.medium.albedo() {
                    break; // 吸收：终止（存活权重隐式）
                }
                let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
                let w_in = -ray.dir; // 相函数轴向（散射前传播方向）
                for (li, light) in scene.lights.iter().enumerate() {
                    // 阴影臂基址：按灯序号错开（`visibility` 内部再加
                    // `SHADOW_BASE_OFFSET`）—— 同一散点的多盏灯不共用同一串均匀量。
                    let sdim: u16 = aux_dim.wrapping_add(li as u16);
                    let mut ssteps: u32 = 0;
                    // ---- A2：环境贴图（CDF 策略）----
                    if let Light::Environment { .. } = light {
                        if let Some(env) = scene.env() {
                            let tex_v = TexTable::new(&scene.textures, &scene.texels);
                            let c = nee_env_volume(
                                scene, env, tex_v, light, p, w_in, st.throughput, region,
                                u_l1, u_l2, stream, sdim, &mut ssteps,
                            );
                            st.radiance =
                                st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                        }
                        continue;
                    }
                    let skip = light_skip(scene, light);
                    let ls = match sample_nee(light, p, u_l1, u_l2) {
                        Some(ls) => ls,
                        None => continue,
                    };
                    let cos_l = w_in.dot(&ls.wi);
                    let ph = region.medium.phase(cos_l);
                    let vis =
                        visibility(scene, p, ls.wi, ls.dist, skip, stream, sdim, &mut ssteps);
                    if vis <= 0.0 {
                        continue;
                    }
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
        } else if let Some(region) = &scene.medium {
            if let Some(tc) =
                medium_collide(region, ray.origin, ray.dir, t_surf, stream, aux_dim, &mut aux_steps)
            {
                let p = ray.at(tc);
                // 散射 vs 吸收（aux 下一步；散射分支 throughput 不变——
                // T(t)·σs 与联合 pdf 相消，见 media 模块文档推导）。
                let u_scatter = stream.aux_1d(aux_dim, aux_steps);
                aux_steps += 1;
                if u_scatter >= region.medium.albedo() {
                    break; // 吸收：终止（存活权重隐式）
                }
                // 体 NEE：相函数对光源方向；备择策略 = 相位抽样该方向。
                let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
                let w_in = -ray.dir; // 相函数轴向（散射前传播方向）
                for light in &scene.lights {
                    // ---- A2：环境贴图（CDF 策略）----
                    // 与表面同构。少了这一臂，介质内的 MIS 权重只有 BSDF 那一半，
                    // 会系统性欠估（不报错、只是偏暗）。
                    if let Light::Environment { .. } = light {
                        if let Some(env) = scene.env() {
                            let tex_v = TexTable::new(&scene.textures, &scene.texels);
                            let c = nee_env_volume(
                                scene, env, tex_v, light, p, w_in, st.throughput, region,
                                u_l1, u_l2, stream, aux_dim, &mut aux_steps,
                            );
                            st.radiance =
                                st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                        }
                        continue;
                    }
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
                // ---- A2：环境**贴图**（BSDF 策略命中环境）----
                //
                // 逃逸射线看到的辐亮度由环境图给出，权重与 NEE 侧的 CDF 策略
                // 配对（选型报告 §3.1 第 4 条）：
                //   * `prev_specular`（相机段 / 镜面链）⇒ 权重 1，延续镜面语义
                //     （镜面方向是 δ 分布，CDF 策略对它测度为零）；
                //   * 否则 `w = 1 − power_heuristic(pdf_env, pdf_bsdf)`，
                //     与 `nee_mis_weight` 的 `Environment` 臂**逐位互补**
                //     （V1 的 `w_nee + w_bsdf ≡ 1` 由此结构性成立）。
                if let Some(env) = scene.env() {
                    let tex_bg = TexTable::new(&scene.textures, &scene.texels);
                    let r = env.radiance(tex_bg, ray.dir);
                    if r.x > 0.0 || r.y > 0.0 || r.z > 0.0 {
                        let w = if params.nee_only && ev > 1 {
                            0.0 // 能量对照：BSDF 策略整体关掉
                        } else if st.prev_specular {
                            1.0
                        } else {
                            let p_env = env.pdf_w(ray.dir);
                            1.0 - crate::light::power_heuristic(p_env, st.prev_pdf_dir)
                        };
                        let c = hadamard(st.throughput, r) * w;
                        st.radiance =
                            st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                    }
                }
                // 逃逸：环境光背景全额贡献（B2；无 Ambient 时为黑 → 保持 B1 语义）。
                // `Light::Ambient` 与 `Light::Environment` 互斥（validate_layout 拦），
                // 故这一段不会与上面的环境图双计。
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

        // ---- W1-B：uv + 解析足迹（只有挂了贴图的材质才付这笔成本）----
        //
        // `mapped == false`（现有全部场景）⇒ uv/足迹不参与任何计算，
        // 下方的材质求值走 `material_bsdf_at` 的"句柄全 0"分支，与
        // `material_bsdf` 逐位相同 ⇒ 定桩前的输出逐字节不变。
        let tex = TexTable::new(&scene.textures, &scene.texels);
        let mapped = material_has_maps(mat);
        let (uv, foot) = if mapped {
            (h.uv, scene.footprint_at(h.shape, p, n, ray.dir, h.t, st.spread))
        } else {
            ([0.0f32, 0.0f32], Footprint::ZERO)
        };
        // 自发光贴图：材质级发光，不注册 NEE（与"未注册 NEE 的发光几何"同口径）。
        if mapped {
            let em = material_emission_at(mat, uv, foot, tex);
            if em.x > 0.0 || em.y > 0.0 || em.z > 0.0 {
                let c = hadamard(st.throughput, em);
                st.radiance = st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
            }
        }

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
        // uv 版：四张图的句柄全 0 时与 `material_bsdf(mat)` 逐位相同。
        let bsdf = match material_bsdf_at(mat, uv, foot, tex) {
            Some(b) => b,
            None => break, // 发光面不续弹（基线简化）
        };
        // 着色法线：法线贴图只改它（BRDF 半球），不改几何法线 `n`
        // （射线偏移与 `from_inside` 判定仍用 `n`，否则掠射面会自交）。
        let n_shade = if mapped && has_normal_map(mat) {
            let (t_ax, b_ax) = scene.shading_frame(h.shape, p, n);
            material_normal_at(mat, uv, foot, tex, n, t_ax, b_ax)
        } else {
            n
        };
        let from_inside = ray.dir.dot(&n) > 0.0;
        // 视点侧法线（出/入射方向同侧；from_inside 时翻转）。
        let nv = if from_inside { -n_shade } else { n_shade };
        let wo = -ray.dir;

        // 反射域 NEE（除镜面瓣占优外；transmission/diffuse 面均采样光源）。
        {
            let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
            for light in &scene.lights {
                // ---- A2：环境贴图走 CDF 策略（`sample_nee` 拿不到 CDF）----
                if let Light::Environment { .. } = light {
                    if let Some(env) = scene.env() {
                        let c = nee_env_surface(
                            scene, env, tex, light, p, nv, wo, mat, &bsdf, st.throughput,
                            u_l1, u_l2, stream, aux_dim, &mut aux_steps,
                        );
                        st.radiance =
                            st.radiance + clamp_indirect(c, ev, params.clamp_indirect);
                    }
                    continue;
                }
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
            // 漫反射把锥角撑到半球量级（A1 的启发式；对二次弹射的 mip 选择生效）。
            st.spread = st.spread.max(0.6);
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
        } else if let MaterialKind::Hair { scatter, geometry } = mat {
            // 全链路 Marschner（R-6②）：续弹方向按 f 采样，pdf 进 MIS /
            // throughput。不再走 Disney 提案；`wo` 是来路（f 的出射侧），
            // 采样结果 `hb.dir` 是 f 的入射侧，与 `surface_brdf` 逐式同源。
            let hb = crate::hair_scatter::sample_hair_bounce(
                scatter, geometry, wo, n, u_lobe, u1, u2,
            );
            st.throughput = hadamard(st.throughput, Vec3::splat(hb.weight));
            st.prev_pdf_dir = hb.pdf;
            st.prev_specular = false;
            // 瓣宽 = 纵向粗糙度（与 Disney 的 roughness² 锥角增量同量级）。
            let alpha = scatter.beta_m;
            st.spread = (st.spread * st.spread + alpha * alpha).sqrt().max(st.spread);
            if !rr_survive(&mut st, stream, ev + 1) {
                break;
            }
            let eps = if hb.dir.dot(&n) > 0.0 {
                SURFACE_EPS
            } else {
                -SURFACE_EPS
            };
            ray = Ray { origin: p + n * eps, dir: hb.dir, time: ray.time };
        } else {
            let sctx = SampleCtx {
                psr_rough: if bsdf.roughness < 1e-3 { 1e-3 } else { 0.0 },
            };
            if let Some(bs) = bsdf.sample(wo, n_shade, from_inside, &sctx, u_lobe, u1, u2) {
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
                        // 折射是**δ（镜面）事件**：后续发光命中必须走「权重 1」分支而不是
                        // MIS。原先置 `prev_specular = false` + `prev_pdf_dir = 0`，使
                        // `power_heuristic(0, pdf) ≡ 0` —— 透过玻璃看到的发光体贡献被乘成
                        // **精确 0**，玻璃在图像上与不透明遮挡体无差别（T08 实测根因）。
                        st.prev_pdf_dir = 0.0;
                        st.prev_specular = true;
                        if !rr_survive(&mut st, stream, ev + 1) {
                            break;
                        }
                        // 折射续弹的落点侧别：`dir·n > 0` ⇒ 出射（表面外侧），
                        // `< 0` ⇒ 入射（表面内侧）。仅靠 `entered` 标志无法区分二者
                        // （BSDF 的折射瓣两个方向都回 `entered = true`），一律取
                        // `p - n·ε` 会让出射射线从**内侧**起步、立刻自交玻璃，
                        // 路径在表面上反复弹射直至 bounce 耗尽 → 玻璃全黑、色散不可见。
                        let eps = if bs.dir.dot(&n) > 0.0 {
                            SURFACE_EPS
                        } else {
                            -SURFACE_EPS
                        };
                        ray = Ray { origin: p + n * eps, dir: bs.dir, time: ray.time };
                    }
                } else {
                    // 反射域（diffuse / specular 瓣）：BSDF 权重即吞吐乘子。
                    st.throughput = hadamard(st.throughput, bs.weight);
                    st.prev_pdf_dir = bs.pdf;
                    st.prev_specular = bs.specular;
                    // 瓣宽即锥角增量（α = roughness²，Burley 映射）；镜面瓣不动。
                    if !bs.specular {
                        let alpha = bsdf.roughness * bsdf.roughness;
                        st.spread = (st.spread * st.spread + alpha * alpha).sqrt().max(st.spread);
                    }
                    if !rr_survive(&mut st, stream, ev + 1) {
                        break;
                    }
                    // 出射侧判定：`dir·n > 0` ⇒ 新射线在表面**外侧**（外部反射 / 介质
                    // 出射折射）；`< 0` ⇒ 仍在**内侧**（介质内全反射）。`entered` 一个
                    // 标志区分不出这两种「非入射」情形（原先一律用 `nv`，介质出射折射
                    // 会被放回介质内侧 → 出射面自交、路径夭折）。
                    let eps = if bs.dir.dot(&n) > 0.0 {
                        SURFACE_EPS
                    } else {
                        -SURFACE_EPS
                    };
                    ray = Ray { origin: p + n * eps, dir: bs.dir, time: ray.time };
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
///
/// 等价于 `render_pixel_range(..., 0, params.spp)`——**不是"另一条路"，是它的
/// 特例**（W7 / B3 的同源要求：预览只降 `(spp, res)`，不换函数）。
pub fn render_pixel(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    pixel: (u32, u32),
    resolution: (u32, u32),
    params: &RenderParams,
) -> [f32; 4] {
    render_pixel_range(scene, cam, ctx, pixel, resolution, params, 0, params.spp)
}

/// 像素渲染（**样本区间**形式）：只取样本序号 `[s_begin, s_end)`，返回其均值。
///
/// # 前缀语义（W7 / B3 判据 P3 的实现基础）
///
/// 第 `s` 个样本只由 `(种子, 像素, s, 维度)` 决定（[`trace_sample`]，D-28
/// 像素域），**与 `params.spp` 无关**。于是
///
/// ```text
/// render_pixel_range(p, 0, N) 的样本序列  ==  render_pixel(p, spp=M).的样本序列 的前 N 项
/// ```
///
/// 对任意 `N ≤ M` 逐位成立。这就是"预览的 N 个样本是终帧 M 个样本的前缀"的
/// 可判定形式：它不是"同一 spp 下两处调用相等"（那是同义反复），而是**样本
/// 流本身可截断**——预览停下来时手里那 N 个样本，等于终帧同一位置的前 N 个。
///
/// # 归约口径
///
/// 区间内取 [`mean_samples`]（prism-core `reduce` 固定形状树），与
/// [`render_pixel`] 完全同一个出口；`count == 1` 时直接返回该样本（`reduce`
/// 对单项是恒等、`/1.0` 是精确运算，故**逐位等价**，同时省掉两次 `Vec`
/// 分配——480×270 / spp=1 的视口路径正是这一档）。
///
/// # 空区间
///
/// `s_end <= s_begin` ⇒ 返回单位元 `[0,0,0,0]`（与 [`mean_samples`] 的空集
/// 语义一致）。调用方不该依赖它，但它不会 panic。
pub fn render_pixel_range(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    pixel: (u32, u32),
    resolution: (u32, u32),
    params: &RenderParams,
    s_begin: u32,
    s_end: u32,
) -> [f32; 4] {
    if s_end <= s_begin {
        return [0.0; 4];
    }
    if s_end - s_begin == 1 {
        return trace_sample(scene, cam, ctx, resolution, pixel, s_begin, params);
    }
    let mut samples = Vec::with_capacity((s_end - s_begin) as usize);
    for s in s_begin..s_end {
        samples.push(trace_sample(scene, cam, ctx, resolution, pixel, s, params));
    }
    mean_samples(&samples)
}

/// 单样本求值（`render_pixel` 的**逐样本形式**，两条通路共用同一个入口）。
///
/// A5 自适应采样按这个入口取样本：`s` 连续地从 `0` 走到 `n_p − 1` ⇒ 与
/// `render_pixel(p, spp = n_p)` 喂给归约的是**同一串**值（逐位相同）。
#[inline]
pub fn trace_sample(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    resolution: (u32, u32),
    pixel: (u32, u32),
    s: u32,
    params: &RenderParams,
) -> [f32; 4] {
    let stream = SampleStream::new(ctx, pixel, s);
    trace_one(scene, cam, ctx, resolution, &stream, params)
}

/// 自由函数形式的相机射线投射（任务接口字面形式；实现委托 [`Camera::cast_ray`]）。
pub fn cast_ray(camera: &Camera, stream: &SampleStream<'_>, resolution: (u32, u32)) -> Ray {
    camera.cast_ray(stream, resolution)
}

// ---------------------------------------------------------------------------
// 光谱渲染切片（V2.2-B-1 / T08）：标量路径 + 退化一致性
// ---------------------------------------------------------------------------
//
// 本组与 [`trace_one`]（RGB Vec3）同构，但吞吐/辐亮度为标量 `f32`：每个彩色
// 量经 [`flat_spectral_value`] 降维为标量（灰反照率口径），末端经
// [`spectral_to_rgb`] 升采样回 RGB（退化=单位复制 `K=(1,1,1)`，色散=CMF 加权）。
//
// # 退化一致性（验收 1，核心）
// 退化配置下 `spectral_to_rgb` 为单位复制 `Vec3::splat(s)`、且 `cauchy_b=0`
// （ior 恒 = `cauchy_a`，不被覆盖），故被积函数与 λ 无关；标量算术 `a·b` /
// `a+b` / `a/q` 与其在灰 RGB 路径单通道上的位模式**逐位一致**（IEEE f32
// 乘法/加法/除法不随载体是标量还是 Vec3 分量而改变）。于是：**对灰材质场景，
// 光谱退化路径与 RGB 路径输出逐字节相等**（由 `qa_spectral_degeneracy` 钉死）。

/// 光谱单路径游走状态（标量切片）。
#[derive(Debug, Clone, Copy)]
struct PathStateSpectral {
    /// 路径吞吐（标量）。
    throughput: f32,
    /// 已累计光谱辐亮度（标量）。
    radiance: f32,
    /// 上一段方向立体角 pdf（发光命中 MIS 用）。
    prev_pdf_dir: f32,
    /// 上一段是否镜面。
    prev_specular: bool,
}

/// 间接贡献标量钳制（与 [`clamp_indirect`] 逐位同口径：灰输入下
/// `Vec3::splat(c).min(t)` = `Vec3::splat(c.min(t))`）。
#[inline]
fn clamp_indirect_scalar(c: f32, ev: u32, limit: Option<f32>) -> f32 {
    match limit {
        Some(t) if ev >= 2 => c.min(t),
        _ => c,
    }
}

/// RR 存活判定（标量版；与 [`rr_survive`] 同口径，灰度下 `clamp(0.05,0.95)`
/// 单通道值 == 标量值，决策位一致）。
fn rr_survive_spectral(
    st: &mut PathStateSpectral,
    stream: &SampleStream<'_>,
    next_ev: u32,
) -> bool {
    if next_ev < 3 {
        return true;
    }
    let q = st.throughput.max(0.05).min(0.95);
    let u = stream.sample_1d(bounce_dim(next_ev, SLOT_RR));
    if u >= q {
        return false;
    }
    st.throughput = st.throughput / q;
    true
}

/// 追踪一条完整光谱路径（单样本）→ rgba。
///
/// 与 [`trace_one`] 同构：几何（射线/命中/法线）与 Sobol 维度完全一致，仅
/// 颜色以标量传递；末端 [`spectral_to_rgb`] 升采样回 RGB（退化=单位复制，
/// 色散=CMF 加权）。退化配置下与灰 RGB [`trace_one`] 逐字节相等。
fn trace_one_spectral(
    scene: &Scene,
    cam: &Camera,
    _ctx: &EvalCtx,
    resolution: (u32, u32),
    stream: &SampleStream<'_>,
    params: &RenderParams,
    cfg: &SpectralConfig,
) -> [f32; 4] {
    let lambda = hero_wavelength(stream.sample_lambda(), cfg);
    let mut ray = cam.cast_ray(stream, resolution);
    let mut st = PathStateSpectral {
        throughput: 1.0,
        radiance: 0.0,
        prev_pdf_dir: 0.0,
        prev_specular: true, // 相机段视为镜面链起点
    };
    let mut aux_steps: u32 = 0;
    for bounce in 0..params.max_bounces {
        let ev = bounce + 1;
        let aux_dim = bounce_dim(ev, SLOT_AUX);
        let hit = scene.intersect(ray.origin, ray.dir, MISS_T, None);

        // 表面事件（或逃逸）。本切片不接入体积光谱（介质内散射留 V2.3），
        // 直接走表面事件——与退化一致性的简单场景口径一致，不引入回归。
        let h = match hit {
            None => {
                // A2：环境贴图（光谱切片与 RGB 路径同权重口径；辐亮度经
                // `flat_spectral_value` 降维——退化配置下与 RGB 单通道逐位一致）。
                if let Some(env) = scene.env() {
                    let tex_bg = TexTable::new(&scene.textures, &scene.texels);
                    let r = flat_spectral_value(env.radiance(tex_bg, ray.dir));
                    if r > 0.0 {
                        let w = if params.nee_only && ev > 1 {
                            0.0
                        } else if st.prev_specular {
                            1.0
                        } else {
                            let p_env = env.pdf_w(ray.dir);
                            1.0 - crate::light::power_heuristic(p_env, st.prev_pdf_dir)
                        };
                        let c = st.throughput * r * w;
                        st.radiance =
                            st.radiance + clamp_indirect_scalar(c, ev, params.clamp_indirect);
                    }
                }
                let bg = ambient_background(&scene.lights);
                let bg_s = flat_spectral_value(bg);
                if bg_s > 0.0 {
                    let c = st.throughput * bg_s;
                    st.radiance = st.radiance + clamp_indirect_scalar(c, ev, params.clamp_indirect);
                }
                break;
            }
            Some(h) => h,
        };
        let p = ray.at(h.t);
        let n = scene.normal_at(h.shape, p);
        let mat = scene.material(h.shape);

        // 发光命中（MIS 权重 vs 光源立体角 pdf）。
        if let MaterialKind::Emissive { radiance } = mat {
            let w_em = if params.nee_only && ev > 1 {
                0.0
            } else if ev == 1 || st.prev_specular {
                1.0
            } else {
                match scene.light_pdf_sa(ray.origin, h.shape) {
                    Some(pdf_l) => crate::light::power_heuristic(st.prev_pdf_dir, pdf_l),
                    None => 1.0,
                }
            };
            let c = st.throughput * flat_spectral_value(*radiance) * w_em;
            st.radiance = st.radiance + clamp_indirect_scalar(c, ev, params.clamp_indirect);
        }

        // BSDF（Pbr；Lambert/Mirror 映射等价 Disney 参数）。
        let mut bsdf = match material_bsdf(mat) {
            Some(b) => b,
            None => break, // 发光面不续弹（基线简化）
        };
        // 色散：用 Cauchy 折射率覆盖材质 ior（退化模式保持原 ior，确保与
        // RGB 路径逐字节一致——见模块文档 §退化一致性）。
        if !cfg.degenerate {
            bsdf.ior = cauchy_ior(cfg, lambda);
        }
        let from_inside = ray.dir.dot(&n) > 0.0;
        let nv = if from_inside { -n } else { n };
        let wo = -ray.dir;

        // 反射域 NEE（除镜面瓣占优外）。
        {
            let (u_l1, u_l2) = stream.sample_2d(bounce_dim(ev, SLOT_LIGHT));
            for light in &scene.lights {
                // A2：环境贴图走 CDF 策略（标量版，与 RGB 路径同构）。
                if let Light::Environment { .. } = light {
                    if let Some(env) = scene.env() {
                        let tex_v = TexTable::new(&scene.textures, &scene.texels);
                        let c = nee_env_surface(
                            scene, env, tex_v, light, p, nv, wo, mat, &bsdf, Vec3::splat(1.0),
                            u_l1, u_l2, stream, aux_dim, &mut aux_steps,
                        );
                        // 标量口径：RGB 贡献经 `flat_spectral_value` 降维后再乘吞吐
                        // （`nee_env_surface` 传吞吐 1 是为了拿到未乘吞吐的辐亮度）。
                        let c_s = flat_spectral_value(c) * st.throughput;
                        st.radiance =
                            st.radiance + clamp_indirect_scalar(c_s, ev, params.clamp_indirect);
                    }
                    continue;
                }
                let skip = light_skip(scene, light);
                let ls = match sample_nee(light, p, u_l1, u_l2) {
                    Some(ls) => ls,
                    None => continue,
                };
                let cos_l = nv.dot(&ls.wi);
                if cos_l <= 0.0 {
                    continue;
                }
                let f = flat_spectral_value(surface_brdf(mat, &bsdf, wo, ls.wi, nv));
                let f_cos = f * cos_l;
                let alt_pdf = surface_pdf(mat, &bsdf, wo, ls.wi, nv);
                let ro = p + nv * SURFACE_EPS;
                let vis = visibility(
                    scene, ro, ls.wi, ls.dist, skip, stream, aux_dim, &mut aux_steps,
                );
                if vis <= 0.0 {
                    continue;
                }
                let w = nee_mis_weight(light, ls.pdf_w, alt_pdf);
                let ls_rad = flat_spectral_value(ls.radiance);
                let c = st.throughput * f_cos * ls_rad * (vis * w / ls.pdf_w);
                st.radiance = st.radiance + clamp_indirect_scalar(c, ev, params.clamp_indirect);
            }
        }

        // BSDF 抽样续弹（Lambert / Mirror / Pbr 三型）。
        let (u_lobe, _) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF));
        let (u1, u2) = stream.sample_2d(bounce_dim(ev, SLOT_BSDF) + 1);
        if let MaterialKind::Lambert { albedo } = mat {
            let d_local = crate::sampler::cosine_hemisphere(u1, u2);
            let (t, b) = crate::sampler::onb(n);
            let dir = (t * d_local.x + b * d_local.y + n * d_local.z).normalize();
            st.throughput = st.throughput * flat_spectral_value(*albedo);
            st.prev_pdf_dir = d_local.z * FRAC_1_PI;
            st.prev_specular = false;
            if !rr_survive_spectral(&mut st, stream, ev + 1) {
                break;
            }
            ray = Ray { origin: p + n * SURFACE_EPS, dir, time: ray.time };
        } else if let MaterialKind::Mirror = mat {
            let dir = ray.dir - n * (2.0 * ray.dir.dot(&n));
            st.prev_pdf_dir = 0.0;
            st.prev_specular = true;
            if !rr_survive_spectral(&mut st, stream, ev + 1) {
                break;
            }
            ray = Ray { origin: p + n * SURFACE_EPS, dir, time: ray.time };
        } else {
            let sctx = SampleCtx {
                psr_rough: if bsdf.roughness < 1e-3 { 1e-3 } else { 0.0 },
            };
            if let Some(bs) = bsdf.sample(wo, n, from_inside, &sctx, u_lobe, u1, u2) {
                if bs.entered {
                    if material_subsurface(mat).is_some() {
                        // 切片范围：SSS 随机游走不在本切片（V2.3 留名）。
                        break;
                    }
                    // 无 subsurface：折射续弹（玻璃内部行进）。
                    st.throughput = st.throughput * flat_spectral_value(bs.weight);
                    // 同上（[`trace_one`] 的 δ 事件口径）：折射后续发光命中权重 = 1。
                    st.prev_pdf_dir = 0.0;
                    st.prev_specular = true;
                    if !rr_survive_spectral(&mut st, stream, ev + 1) {
                        break;
                    }
                    // 落点侧别同 [`trace_one`]：`dir·n > 0` ⇒ 出射（表面外侧）。
                    let eps = if bs.dir.dot(&n) > 0.0 {
                        SURFACE_EPS
                    } else {
                        -SURFACE_EPS
                    };
                    ray = Ray { origin: p + n * eps, dir: bs.dir, time: ray.time };
                } else {
                    // 反射域（diffuse / specular 瓣）。
                    st.throughput = st.throughput * flat_spectral_value(bs.weight);
                    st.prev_pdf_dir = bs.pdf;
                    st.prev_specular = bs.specular;
                    if !rr_survive_spectral(&mut st, stream, ev + 1) {
                        break;
                    }
                    // 出射侧判定（与 [`trace_one`] 逐字同一规则）：`dir·n > 0` ⇒ 外侧。
                    let eps = if bs.dir.dot(&n) > 0.0 {
                        SURFACE_EPS
                    } else {
                        -SURFACE_EPS
                    };
                    ray = Ray { origin: p + n * eps, dir: bs.dir, time: ray.time };
                }
            } else {
                break; // 采样失败：终止
            }
        }
    }
    // 末端升采样：退化=单位复制（Vec3::splat），色散=CMF 加权。
    let rgb = spectral_to_rgb(st.radiance, lambda, cfg);
    [rgb.x, rgb.y, rgb.z, 1.0]
}

/// 像素渲染（光谱切片，纯函数）：spp 样本经 [`mean_samples`] 同口径归约
/// （与 [`render_pixel`] 共用，保证退化模式与 RGB 路径逐字节一致）。
pub fn render_pixel_spectral(
    scene: &Scene,
    cam: &Camera,
    ctx: &EvalCtx,
    pixel: (u32, u32),
    resolution: (u32, u32),
    params: &RenderParams,
    cfg: &SpectralConfig,
) -> [f32; 4] {
    let mut samples = Vec::with_capacity(params.spp as usize);
    for s in 0..params.spp {
        let stream = SampleStream::new(ctx, pixel, s);
        samples.push(trace_one_spectral(
            scene, cam, ctx, resolution, &stream, params, cfg,
        ));
    }
    mean_samples(&samples)
}

// ---------------------------------------------------------------------------
// 测试（关卡 1–4 集成验证）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::HomogeneousMedium;
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
                medium: Medium::Homogeneous(HomogeneousMedium::new(sigma_t, 0.0, 0.0)),
                priority: 0,
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

    /// QA-F2 回归钉子：异质介质必须**真实可达**渲染路径，且整画面衰减仍贴 Beer。
    ///
    /// 用 σt 恒定场（退化情形）走 `Medium::Heterogeneous`：渲染路径必须选择
    /// 「步进 + 接受」两均匀分支（`needs_accept_uniform`），整画面均值与
    /// 均匀介质同解析式（`base × e^(−2σt·h)`）在 MC 误差内一致。同场景两遍
    /// 渲染还须逐位一致（确定性）。
    #[test]
    fn heterogeneous_medium_render_path_is_reachable_and_matches_beer() {
        use crate::media::HeterogeneousMedium;
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
                medium: Medium::Heterogeneous(HeterogeneousMedium::constant_sigma_t(
                    sigma_t, 0.0, 0.0,
                )),
                priority: 0,
            }),
            paraboloids: vec![],
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
        };
        // 可达性：渲染路径确实走异质分支（两均匀）。
        assert!(
            scene.medium.as_ref().unwrap().medium.needs_accept_uniform(),
            "Medium::Heterogeneous 必须触发两均匀分支（否则异质从未被渲染路径消费）"
        );
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
        let frame2 = render_frame(&scene, &cam, &ctx, res, &RenderParams::new(256, 2));
        assert_eq!(frame, frame2, "异质介质渲染必须确定性（两遍逐位一致）");

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
            "异质介质衰减：实测 {mean}，Beer 解析 {expected}（相对 {rel}）"
        );
        println!("异质 Beer 关卡：实测 = {mean:.6}，解析 = {expected:.6}（相对 {rel:.5}）");
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

    /// 轴向射线 `a == 0` 退化线性单根：o=(0,0,-1), d=(0,0,1), k=1 ⇒ t=1.0
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

    /// 轴向平行且不在轴上（`a == 0 && b == 0`）：None 不 panic
    /// （零方向射线 d=(0,0,0)：`-c/b` 除零前被 `b == 0` 分支拦截）。
    #[test]
    fn paraboloid_axial_parallel_off_axis_none() {
        let sc = paraboloid_scene(1.0, 1.5);
        assert_eq!(
            sc.paraboloid_t(0, Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0e30),
            None
        );
    }

    /// 离轴双根都有效 ⇒ 取最小有效 t（a < 0 时勿假设根序）：
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

    /// r_max 边界：界外 miss / 界内 hit / 闭边界 `r² == r_max²` 算命中。
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

    /// `disc == 0` 精确相切（单重根正常命中）：o=(1,-2,1), d=(0,1,0)
    /// ⇒ a=-1, b=4, c=-4，disc=16-16=0，t=2.0（全整数运算，逐位精确）。
    #[test]
    fn paraboloid_disc_zero_tangent_exact() {
        let sc = paraboloid_scene(1.0, 1.5);
        let t = sc.paraboloid_t(0, Vec3::new(1.0, -2.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 1.0e30);
        assert_eq!(t, Some(2.0));
    }

    /// 切平面配置钉死：P=(0.8,0.8,1.28) 在 z=x²+y² 上
    /// （0.64+0.64=1.28 ✓），n=(-1.6,-1.6,1) 即 P 点 ∇F；自 P+s·n̂ 沿 −n̂
    /// 入射必命中 t=s±1e-4（s 取 0.5 / 1.0 / 2.0 三档）。
    /// 注：法向入射 disc≠0——唯一正根 t=s，第二根在起点反方向（负）被
    /// T_MIN 排除；真相切（disc==0）用例见上一项。
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

    /// 图元自身参数非法 ⇒ 永不命中（不 panic）：k<=0 / r_max<=0 / 非有限。
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

    /// 射线非有限 ⇒ None（NaN 比较天然排除；Inf 由入口 finite 检查拦截）。
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

    /// `normal_at` 解析对照：命中点法线 ≡ normalize(-2k·x, -2k·y, 1)，
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

    /// `t_max` 语义与 sphere_t/box_t 同口径：`t >= t_max` 即排除
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

    /// 平面：轴向命中（t 精确）+ 平行（denom==0）⇒ None；
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

    /// 有限切片裁剪：`|u|>h` 或 `|v|>h` ⇒ None；闭边界 `|u|==h` 算命中；
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

    /// `Plane::new` 单位化法线；零长/非有限法线永不命中（不 panic）；
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

    /// `light_pdf_sa`：抛物面/平面均非球光源几何 ⇒ None（与 Box 臂同语义）。
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

    /// `Scene::intersect` 跨图元取最近 + skip 豁免（阴影射线语义）：
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
    /// 判据：图像全有限、立方体覆盖区有能量、四角（逃逸）恰为 0。
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
    /// 阈值说明：构建耗时上限 "BVH 构建 < 1 s" 以 **release** 为准（此处断言
    /// `cfg!(debug_assertions)` 时放宽到 20 s——debug 下 SAH 构建有 ~10×
    /// 常数惩罚，把 release 上限写死在 debug 里只会得到一条假的回归线）。
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
    // 曲线图元族：圆柱 / 圆锥 / 圆环 / 旋转曲面
    //
    // 覆盖要求（规格钉死）：每类 ≥1 条已知值解析对拍（手算参考值写进
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

// ---------------------------------------------------------------------------
// W1-A 定桩：布局不变式测试（断言"真的会拦"）
// ---------------------------------------------------------------------------
//
// 这一组测试的存在理由与 `crate::layout` 相同：断言写完不验等于没写。
// 每条都构造一个**违规**场景并要求被拒绝，而不是只测"合法场景通过"
// ——后者在断言被删掉时仍然全绿。

#[cfg(test)]
mod w1a_layout_tests {
    use super::*;
    use crate::texture::{cdf_len_bytes, NO_TEXTURE, TEX_FMT_RGBA16F, SRGB_LINEAR};

    #[test]
    fn default_scene_passes_layout_check() {
        // `Scene::default()` 必须合法：它是所有字面量的补全项
        // （`..Default::default()`），若默认值非法，全部既有场景都会被拦。
        let s = Scene::default();
        assert!(s.media.is_empty(), "media 缺省为空 ⇒ 走旧 medium 分支");
        assert!(s.textures.is_empty() && s.instances.is_empty());
        assert_eq!(s.validate_layout(), Ok(()));
        s.assert_layout(); // 不 panic 即为通过
    }

    #[test]
    fn ambient_and_environment_cannot_coexist() {
        let mut s = Scene::default();
        s.lights.push(Light::Ambient { radiance: Vec3::splat(0.05) });
        s.textures.push(crate::texture::TextureRef::new(
            2048,
            1024,
            1,
            0,
            TEX_FMT_RGBA16F,
            SRGB_LINEAR,
        ));
        // 先验"只有 Ambient"合法。
        assert_eq!(s.validate_layout(), Ok(()));
        // 加一盏环境图灯 ⇒ 必须被拒（双计）。
        s.lights.push(Light::Environment { map: 0 });
        let err = s.validate_layout().expect_err("Ambient + Environment 必须被拒");
        assert!(err.contains("不可并存"), "错误消息须点明互斥，实际：{err}");
    }

    #[test]
    #[should_panic(expected = "场景布局违规")]
    fn assert_layout_panics_on_dual_background() {
        let mut s = Scene::default();
        s.lights.push(Light::Ambient { radiance: Vec3::splat(0.1) });
        s.lights.push(Light::Environment { map: 0 });
        s.assert_layout();
    }

    #[test]
    fn environment_handle_must_be_in_range() {
        let mut s = Scene::default();
        // 句柄 0 = 无贴图 ⇒ 合法（不引用任何 texture）。
        s.lights.push(Light::Environment { map: NO_TEXTURE });
        assert_eq!(s.validate_layout(), Ok(()));
        // 非 0 句柄越界 ⇒ 拒。
        s.lights[0] = Light::Environment { map: 3 };
        let err = s.validate_layout().expect_err("越界句柄必须被拒");
        assert!(err.contains("越界"), "实际：{err}");
        // 补一张纹理后，句柄 1（= textures[0]）合法；句柄 2 仍越界。
        //
        // W2 起"合法"还多一条：非 0 句柄必须带上已构建的 CDF（否则 IBL
        // 静默失效）。`rebuild_env_cdf` 是该缓存的唯一维护入口。
        s.textures.push(crate::texture::TextureRef::new(
            8,
            4,
            1,
            0,
            TEX_FMT_RGBA16F,
            SRGB_LINEAR,
        ));
        s.texels = vec![[1.0, 1.0, 1.0, 1.0]; 8 * 4];
        s.lights[0] = Light::Environment { map: 1 };
        s.rebuild_env_cdf().expect("8×4 常数环境图应可建 CDF");
        assert_eq!(s.validate_layout(), Ok(()));
        s.lights[0] = Light::Environment { map: 2 };
        assert!(s.validate_layout().is_err(), "句柄 2 超出 1 条纹理");
    }

    #[test]
    fn uv_and_tangent_must_match_positions_len() {
        let mut s = Scene::default();
        let mut m = Mesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        );
        // 空 = 无 uv ⇒ 合法（现有全部网格都是这样）。
        assert_eq!(s.validate_layout(), Ok(()));
        // 长度不符 ⇒ 拒。
        m.uv = vec![[0.0, 0.0]; 2];
        s.meshes.push((m, MaterialKind::Mirror));
        let err = s.validate_layout().expect_err("uv 与 positions 不同长必须被拒");
        assert!(err.contains("uv"), "实际：{err}");
        // 补齐 ⇒ 合法。
        s.meshes[0].0.uv = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(s.validate_layout(), Ok(()));
        // tangent 同理。
        s.meshes[0].0.tangent = vec![[1.0, 0.0, 0.0, 1.0]; 1];
        let err = s.validate_layout().expect_err("tangent 与 positions 不同长必须被拒");
        assert!(err.contains("tangent"), "实际：{err}");
        s.meshes[0].0.tangent = vec![[1.0, 0.0, 0.0, 1.0]; 3];
        assert_eq!(s.validate_layout(), Ok(()));
    }

    #[test]
    fn bsdf_map_handle_must_be_in_range() {
        let mut s = Scene::default();
        let b = Bsdf::new(Vec3::splat(0.8), 0.0, 0.5, 1.5, 0.0, 0.0, Vec3::splat(0.3));
        // 全 0 句柄 ⇒ 合法（无贴图）。
        assert!(!b.has_any_map());
        s.spheres.push((
            Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
            MaterialKind::Pbr(b),
        ));
        assert_eq!(s.validate_layout(), Ok(()));
        // 句柄 1 但没有纹理 ⇒ 拒。
        s.spheres[0].1 = MaterialKind::Pbr(b.with_maps([1, 0, 0, 0]));
        let err = s.validate_layout().expect_err("越界贴图句柄必须被拒");
        assert!(err.contains("maps[0]"), "实际：{err}");
        // 补纹理 ⇒ 合法。
        s.textures.push(crate::texture::TextureRef::new(
            8,
            8,
            4,
            0,
            TEX_FMT_RGBA16F,
            SRGB_LINEAR,
        ));
        assert_eq!(s.validate_layout(), Ok(()));
    }

    /// CDF 长度口径 `(W+1)·H·4`：编译期常量与运行时断言两条路径都对，
    /// 且**少一行会被拦住**（off-by-one-row 是最容易犯的错）。
    #[test]
    fn cdf_len_formula_and_off_by_one_row() {
        assert_eq!(cdf_len_bytes(2048, 1024), 8_392_704);
        crate::texture::assert_cdf_len(2048, 1024, 8_392_704);
        let bad = std::panic::catch_unwind(|| {
            crate::texture::assert_cdf_len(2048, 1024, cdf_len_bytes(2048, 1023))
        });
        assert!(bad.is_err(), "少一行必须 panic");
    }

    /// **行为零变化的机械保证**（W3 后语义已更新，见下）。
    ///
    /// 三个 V2.5 定桩字段里，`textures` / `media` 至今**没有任何渲染分支读**；
    /// `instances` 在 W3 之后**确实被读了**（`intersect` 末族 + TLAS），
    /// 所以这条测试不再依赖"没人读"，而是依赖更弱也更真的性质：
    /// **一条引用了不存在的原型几何（`geom_id = 0`，而 `instance_geoms` 为空）
    /// 的实例记录，既不产生命中也不遮挡阴影**——`world_aabb` 退化为空盒、
    /// TLAS 里不占叶、线性回退里 `instance_geoms.get()` 直接 `None`。
    /// 于是"填满定桩字段"与"空场景"的同一像素 `render_pixel` 输出仍**逐位相同**。
    ///
    /// 这条测试因此从"惰性字段不干扰"升级成"**悬空实例不干扰**"的回归闸：
    /// 若哪天有人把 `instance_geoms` 的越界访问改成 panic 或改成"用 0 号
    /// 几何兜底"，这里第一个红。
    ///
    /// # W6 / A6 改写（原注释预言的那一天）
    ///
    /// `media` **已被点亮**（多区域介质），故它不再是"未读字段"，本测试不再
    /// 把它塞进 `stubbed`——原样保留会让"填了 media 就改像素"变成失败，而
    /// 那恰恰是新语义的正确行为。`media` 的新语义判据在
    /// `tests/w6_a6_media.rs`（V1–V5），这里只守 `textures` / `instances`
    /// 这两个仍然惰性的字段。
    #[test]
    fn dormant_stub_fields_do_not_change_pixel_output() {
        let white = MaterialKind::Pbr(Bsdf::new(
            Vec3::splat(0.72),
            0.0,
            0.5,
            1.5,
            0.0,
            0.0,
            Vec3::splat(0.3),
        ));
        let mut base = Scene::default();
        base.spheres.push((
            Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 },
            white,
        ));
        base.lights
            .push(Light::Directional {
                direction: Vec3::new(0.3, -1.0, 0.2).normalize(),
                radiance: Vec3::splat(3.0),
                cos_theta_max: 1.0,
            });

        let mut stubbed = base.clone();
        stubbed.textures.push(crate::texture::TextureRef::new(
            16, 16, 5, 0, TEX_FMT_RGBA16F, SRGB_LINEAR,
        ));
        stubbed.instances.push(crate::instance::Instance::identity(0, 0));
        // `media` 自 W6 起被消费，不再是定桩字段（见上方"W6 / A6 改写"）。

        let cam = Camera::new(
            Vec3::new(0.0, 0.0, -4.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            0.0,
            4.0,
            0.0,
            0.0,
        );
        let ctx = prism_core::EvalCtx::new(0, 0, 0);
        let params = RenderParams::new(4, 2);
        let res = (8u32, 8u32);
        for y in 0..res.1 {
            for x in 0..res.0 {
                let a = render_pixel(&base, &cam, &ctx, (x, y), res, &params);
                let b = render_pixel(&stubbed, &cam, &ctx, (x, y), res, &params);
                for c in 0..4 {
                    assert_eq!(
                        a[c].to_bits(),
                        b[c].to_bits(),
                        "定桩字段改变了像素 ({x},{y}) 通道 {c}：{} ≠ {}",
                        a[c],
                        b[c]
                    );
                }
            }
        }
    }

    // =====================================================================
    // W3 / A3：几何实例化（C1 两级 BVH）
    // =====================================================================

    /// 局部立方体 `[-h, h]³`（12 三角形、绕序朝外）。
    fn w3_cube(h: f32) -> (Vec<Vec3>, Vec<u32>) {
        let v = [
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        // 每面两三角形，绕序逆时针（外法线）。
        let idx = [
            0, 3, 2, 0, 2, 1, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        (v.to_vec(), idx.to_vec())
    }

    /// 立方体网格（局部坐标；供"共享几何"用）。
    fn w3_cube_mesh(h: f32) -> Mesh {
        let (p, i) = w3_cube(h);
        Mesh::new(p, i)
    }

    fn w3_lambert(c: f32) -> MaterialKind {
        MaterialKind::Lambert { albedo: Vec3::splat(c) }
    }

    fn w3_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 0.6, -4.2),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            4.2,
            0.0,
            0.0,
        )
    }

    /// 整幅渲染（`render_pixel` 通道，行主序 RGBA）。
    fn w3_render(scene: &Scene, res: (u32, u32), spp: u32, bounces: u32) -> Vec<[f32; 4]> {
        let cam = w3_camera();
        let ctx = EvalCtx::new(7, 3, 11);
        let params = RenderParams::new(spp, bounces);
        let mut out = Vec::with_capacity((res.0 * res.1) as usize);
        for y in 0..res.1 {
            for x in 0..res.0 {
                out.push(render_pixel(scene, &cam, &ctx, (x, y), res, &params));
            }
        }
        out
    }

    fn w3_light() -> Light {
        // 注意：`Light::Directional::direction` 是**指向光**的方向
        // （light.rs:34），不是光传播方向。写成传播方向会让所有面 cos < 0，
        // 整幅全黑 —— 本测试撞过这个坑。
        Light::Directional {
            direction: Vec3::new(-0.35, 1.0, -0.25).normalize(),
            radiance: Vec3::splat(4.0),
            cos_theta_max: 1.0,
        }
    }

    /// 确定性 LCG（不引第三方、不用 libm）。
    fn w3_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
        }
    }

    /// 判据 ①：**identity 实例 vs 展平版逐位一致**（选型报告 §4.1 第一条）。
    ///
    /// 为什么可证而不是"测出来像"：`Instance::local_ray` 对 identity 实例
    /// 原样返回世界射线（一次乘法都不做），法线也走同一条
    /// `triangle_unit_normal`，于是下游每个浮点运算的输入位型完全相同。
    #[test]
    fn w3_identity_instance_is_bitwise_identical_to_flattened() {
        let (pos, idx) = w3_cube(1.0);
        let m = w3_lambert(0.75);
        // A：展平（顶点即世界坐标）。
        let mut flat = Scene {
            lights: vec![w3_light()],
            ..Default::default()
        };
        flat.add_mesh(pos.clone(), idx.clone(), m);
        // B：共享几何 + identity 实例。
        let mut inst = Scene {
            lights: vec![w3_light()],
            ..Default::default()
        };
        let g = inst.add_instance_geom(pos, idx, m);
        inst.add_instance(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        inst.rebuild_instance_bvh();
        inst.assert_layout();

        let res = (40, 40);
        let a = w3_render(&flat, res, 3, 2);
        let b = w3_render(&inst, res, 3, 2);
        assert_eq!(a.len(), b.len());
        let mut nonzero = 0usize;
        for i in 0..a.len() {
            if a[i][0] > 0.0 || a[i][1] > 0.0 || a[i][2] > 0.0 {
                nonzero += 1;
            }
            for c in 0..4 {
                assert_eq!(
                    a[i][c].to_bits(),
                    b[i][c].to_bits(),
                    "identity 实例与展平版像素 {} 通道 {c} 不逐位：{} ≠ {}",
                    i,
                    a[i][c],
                    b[i][c]
                );
            }
        }
        assert!(nonzero > 100, "画面几乎全黑（{nonzero} 个非零像素），判据没有鉴别力");

        // 反向对照：把实例挪走 ⇒ 像素必须**变**（证明上面对比的是真有几何的画面）。
        let mut moved = inst.clone();
        moved.instances[0] = Instance::from_affine(
            [1.0, 0.0, 0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            g,
            INSTANCE_MAT_FROM_GEOM,
        )
        .unwrap();
        moved.rebuild_instance_bvh();
        let c = w3_render(&moved, res, 3, 2);
        assert_ne!(a, c, "实例被挪到 50 单位外，像素竟然没变 —— 渲染根本没走实例路径");
    }

    /// 掠射门槛：`|cos|` 低于此值的命中，`t` 的条件数发散到 1/cos，
    /// 相对误差 ≤ 1e-6 在 f32 下**不可达**（见测试末的说明与实录 §16）。
    const W3_NON_GRAZING_COS: f32 = 0.4;

    /// 判据 ②：非 identity 逐命中 `t` 相对误差 ≤ 1e-6（1e5 条射线抽样）。
    ///
    /// 非掠射（`|cos| ≥ 0.4`）硬判 1e-6；掠射段单独记账（相对误差的
    /// 条件数是 `1/cos`，两个**都正确**的实现也会在 |cos|→0 处拉开）。
    #[test]
    fn w3_instance_t_matches_flattened_within_1e_6() {
        // 非均匀缩放 + 剪切 + 行近似旋转（用有理数，避免 libm 影响可复现性）。
        let c = 0.6f32;
        let s = 0.8f32; // c² + s² = 0.36 + 0.64 = 1
        let inst = Instance::from_affine(
            [
                2.0 * c, -2.0 * s, 0.25, 0.3, //
                1.5 * s, 1.5 * c, 0.0, -0.2, //
                0.0, 0.3, 0.5, 1.1,
            ],
            0,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("仿射应可逆");
        let (pos, idx) = w3_cube(1.0);
        let pos_flat: Vec<Vec3> = pos.iter().map(|p| inst.apply_point(*p)).collect();
        let m = w3_lambert(0.75);
        let mut flat = Scene::default();
        flat.add_mesh(pos_flat, idx.clone(), m);
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(pos, idx, m);
        assert_eq!(g, inst.geom_id, "原型下标应与实例引用一致");
        scene.add_instance(inst);
        scene.rebuild_instance_bvh();

        let mut next = w3_rng(0x2545_F491_4F6C_DD1D);
        let mut hits = 0usize;
        let mut worst = 0.0f32;
        let mut worst_cos = 1.0f32;
        let mut over = 0usize;
        let mut over_nongrazing = 0usize;
        let mut min_cos_over = 1.0f32;
        let mut worst_abs = 0.0f32;
        let mut mismatched = 0usize;
        let mut rels: Vec<f32> = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            let theta = next() * 2.0 * PI;
            let phi = (next() - 0.5) * 1.4;
            let dist = 6.0 + next() * 2.0;
            let o = Vec3::new(
                dist * phi.cos() * theta.cos(),
                dist * phi.sin() * 0.6,
                dist * phi.cos() * theta.sin(),
            );
            let target = Vec3::new(
                (next() - 0.5) * 3.0,
                (next() - 0.5) * 3.0,
                (next() - 0.5) * 3.0,
            );
            let d = (target - o).normalize();
            let a = flat.intersect(o, d, MISS_T, None);
            let b = scene.intersect(o, d, MISS_T, None);
            match (a, b) {
                (Some(x), Some(y)) => {
                    hits += 1;
                    let abs = (x.t - y.t).abs();
                    let rel = abs / x.t.abs().max(1e-12);
                    rels.push(rel);
                    if abs > worst_abs {
                        worst_abs = abs;
                    }
                    let p = o + d * x.t;
                    let cos = d.dot(&flat.normal_at(x.shape, p)).abs();
                    if rel > worst {
                        worst = rel;
                        worst_cos = cos;
                    }
                    if rel > 1e-6 {
                        over += 1;
                        if cos < min_cos_over {
                            min_cos_over = cos;
                        }
                        if cos >= W3_NON_GRAZING_COS {
                            over_nongrazing += 1;
                        }
                    }
                }
                (None, None) => {}
                _ => mismatched += 1, // 命中性都不一致：比数值误差严重
            }
        }
        rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p999 = rels[(rels.len() as f64 * 0.999) as usize];
        let median = rels[rels.len() / 2];
        println!(
            "t 相对误差：hits={hits} median={median:e} p99.9={p999:e} worst={worst:e}\
             (cos={worst_cos:.4}) 超1e-6={over}（其中 |cos|≥{W3_NON_GRAZING_COS} 的 {over_nongrazing} 条，\
             掠射段最小 |cos|={min_cos_over:.4}）最大绝对偏差={worst_abs:e}"
        );
        assert_eq!(mismatched, 0, "有 {mismatched} 条射线命中性不一致（比数值误差严重）");
        assert!(hits >= 20_000, "抽样命中数太少（{hits}），判据没有鉴别力");
        assert!(median <= 1e-7, "中位相对误差 {median:e} 超过 1e-7：这不是舍入级差异，是算法差异");
        assert_eq!(
            over_nongrazing, 0,
            "有 {over_nongrazing} 条 |cos| ≥ {W3_NON_GRAZING_COS} 的命中相对误差超 1e-6（掠射解释不成立）"
        );
        // 绝对偏差硬界：世界单位下 1e-3（物体尺度 ~2，即 5e-4 相对；
        // 折算到 64×64 画面是 1e-4 像素量级 —— 远小于判据③的 1/255）。
        assert!(
            worst_abs <= 1e-3,
            "最大绝对 t 偏差 {worst_abs:e} 超 1e-3 世界单位（像素尺度下应远小于 1）"
        );
    }

    /// 判据 ③：像素 `max_channel_delta ≤ 1`、`differing_pixels` 占比 ≤ 0.5%。
    #[test]
    fn w3_instance_pixels_match_flattened_within_1_delta() {
        let c = 0.6f32;
        let s = 0.8f32;
        let inst = Instance::from_affine(
            [
                1.6 * c, -1.6 * s, 0.0, 0.25, //
                1.2 * s, 1.2 * c, 0.0, -0.15, //
                0.0, 0.0, 1.4, 0.9,
            ],
            0,
            INSTANCE_MAT_FROM_GEOM,
        )
        .unwrap();
        let (pos, idx) = w3_cube(1.0);
        let pos_flat: Vec<Vec3> = pos.iter().map(|p| inst.apply_point(*p)).collect();
        let m = w3_lambert(0.75);
        let mut flat = Scene {
            lights: vec![w3_light()],
            ..Default::default()
        };
        flat.add_mesh(pos_flat, idx.clone(), m);
        let mut scene = Scene {
            lights: vec![w3_light()],
            ..Default::default()
        };
        let g = scene.add_instance_geom(pos, idx, m);
        assert_eq!(g, inst.geom_id, "原型下标应与实例引用一致");
        scene.add_instance(inst);
        scene.rebuild_instance_bvh();

        let res = (64, 64);
        let a = w3_render(&flat, res, 4, 2);
        let b = w3_render(&scene, res, 4, 2);
        let mut max_delta = 0i32;
        let mut differing = 0usize;
        for i in 0..a.len() {
            let mut pix_diff = false;
            for ch in 0..3 {
                let x = (a[i][ch] * 255.0).clamp(0.0, 255.0).round() as i32;
                let y = (b[i][ch] * 255.0).clamp(0.0, 255.0).round() as i32;
                let d = (x - y).abs();
                if d > 0 {
                    pix_diff = true;
                }
                if d > max_delta {
                    max_delta = d;
                }
            }
            if pix_diff {
                differing += 1;
            }
        }
        let ratio = differing as f64 / a.len() as f64;
        println!("像素：max_channel_delta={max_delta} differing={differing}/{} ({:.4}%)", a.len(), ratio * 100.0);
        assert!(max_delta <= 1, "max_channel_delta = {max_delta} 超 1");
        assert!(ratio <= 0.005, "differing_pixels 占比 {:.4}% 超 0.5%", ratio * 100.0);
    }

    /// TLAS 快路径与线性兜底必须同解（含 `t` 并列的实例下标 tie-break）。
    #[test]
    fn w3_tlas_matches_linear_scan() {
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(w3_cube_mesh(0.5).triangles.iter().flat_map(|t| [t.a, t.b, t.c]).collect(), (0..36).collect(), w3_lambert(0.6));
        // 8 个实例，其中两个完全重合（考 tie-break）。
        let mut batch = Vec::new();
        for i in 0..6 {
            let x = i as f32 * 0.9 - 2.0;
            batch.push(
                Instance::from_affine(
                    [0.8, 0.0, 0.0, x, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
                    g,
                    INSTANCE_MAT_FROM_GEOM,
                )
                .unwrap(),
            );
        }
        batch.push(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        batch.push(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        scene.add_instances(batch);
        let mut linear = scene.clone();
        linear.instance_bvh = None; // 走线性兜底

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut n = 0usize;
        for _ in 0..20_000 {
            let o = Vec3::new((next() - 0.5) * 8.0, (next() - 0.5) * 4.0, -6.0);
            let d = Vec3::new((next() - 0.5) * 0.5, (next() - 0.5) * 0.5, 1.0).normalize();
            let a = scene.intersect(o, d, MISS_T, None);
            let b = linear.intersect(o, d, MISS_T, None);
            match (a, b) {
                (Some(x), Some(y)) => {
                    n += 1;
                    assert_eq!(x.t.to_bits(), y.t.to_bits(), "TLAS 与线性兜底的 t 不逐位");
                    assert_eq!(x.shape, y.shape, "TLAS 与线性兜底的形状引用不一致");
                }
                (None, None) => {}
                _ => panic!("TLAS 与线性兜底命中性不一致"),
            }
        }
        assert!(n > 1000, "命中数太少（{n}）");
    }

    /// 判据 ④⑤：1 万实例 —— 共享几何只建一次 BLAS；内存 112 B/实例；
    /// 构建 O(n log n) 毫秒级。
    #[test]
    fn w3_ten_thousand_instances_share_one_blas() {
        // 一个 32×32 网格 ≈ 2048 三角形的原型。
        let (mut pos, mut idx) = (Vec::new(), Vec::new());
        let n = 32u32;
        for j in 0..=n {
            for i in 0..=n {
                pos.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend_from_slice(&[a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let tri = idx.len() / 3;
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(pos, idx, w3_lambert(0.5));
        let mut batch = Vec::with_capacity(10_000);
        for k in 0..10_000u32 {
            let x = (k % 100) as f32 * 0.25 - 12.5;
            let y = (k / 100) as f32 * 0.25 - 12.5;
            batch.push(
                Instance::from_affine(
                    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, -1.0],
                    g,
                    INSTANCE_MAT_FROM_GEOM,
                )
                .unwrap(),
            );
        }
        let t0 = std::time::Instant::now();
        scene.add_instances(batch);
        let build = t0.elapsed();
        assert_eq!(scene.instance_geom_count(), 1, "共享几何必须只有 1 份");
        assert_eq!(scene.instance_geom_triangle_count(), tri);
        assert_eq!(scene.mesh_count(), 0, "原型不得进 meshes（否则会在原点被直接渲染）");
        assert_eq!(scene.triangle_count(), 0);
        assert_eq!(scene.instance_count(), 10_000);
        assert_eq!(
            scene.instance_bvh.as_ref().map(|b| b.prim_count()),
            Some(10_000),
            "TLAS 叶子数必须等于实例数"
        );
        let inst_bytes = 10_000 * core::mem::size_of::<Instance>();
        assert_eq!(inst_bytes, 1_120_000, "实例表必须恰好 1.12 MB（112 B/条）");
        let tlas_bytes = scene.instance_bvh.as_ref().map_or(0, |b| b.mem_bytes());
        let tlas_nodes = scene.instance_bvh.as_ref().map_or(0, |b| b.node_count());
        println!(
            "1 万实例：BLAS={tri} 三角形（1 份）、实例表={:.2} MB、TLAS={} MB/{} 节点、构建={:?}",
            inst_bytes as f64 / (1024.0 * 1024.0),
            tlas_bytes as f64 / (1024.0 * 1024.0),
            tlas_nodes,
            build
        );
        assert!(
            build.as_millis() < 2_000,
            "10k 实例 TLAS 构建 {build:?} 超过 2 s（实测应在毫秒级）"
        );
        // 真的能命中：从 -z 方向打一条射线穿整片实例。
        let h = scene
            .intersect(Vec3::new(0.1, 0.1, -5.0), Vec3::new(0.0, 0.0, 1.0), MISS_T, None)
            .expect("网格阵列上必有命中");
        assert!(matches!(h.shape, ShapeRef::Instance(_, _)));
    }

    /// TLAS 剪枝真的在起作用：同一 10k 实例场景，BVH 路径 vs 线性兜底的
    /// 单射线耗时（打印，不断言绝对时间 —— 只断言两路结果一致）。
    #[test]
    fn w3_tlas_prunes_ten_thousand_instances() {
        let (mut pos, mut idx) = (Vec::new(), Vec::new());
        let n = 16u32;
        for j in 0..=n {
            for i in 0..=n {
                pos.push(Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend_from_slice(&[a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(pos, idx, w3_lambert(0.5));
        let mut batch = Vec::with_capacity(10_000);
        for k in 0..10_000u32 {
            let x = (k % 100) as f32 * 0.25 - 12.5;
            let y = (k / 100) as f32 * 0.25 - 12.5;
            batch.push(
                Instance::from_affine(
                    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, -1.0],
                    g,
                    INSTANCE_MAT_FROM_GEOM,
                )
                .unwrap(),
            );
        }
        scene.add_instances(batch);
        let mut linear = scene.clone();
        linear.instance_bvh = None;

        // 512 条射线（方向确定，命中率约一半）。
        let mut rays = Vec::new();
        let mut next = w3_rng(0xDEAD_BEEF_1234_5678);
        for _ in 0..512 {
            let o = Vec3::new((next() - 0.5) * 24.0, (next() - 0.5) * 24.0, -6.0);
            let d = Vec3::new((next() - 0.5) * 0.4, (next() - 0.5) * 0.4, 1.0).normalize();
            rays.push((o, d));
        }
        let run = |sc: &Scene| -> (usize, std::time::Duration) {
            let t0 = std::time::Instant::now();
            let mut hits = 0usize;
            for &(o, d) in &rays {
                if sc.intersect(o, d, MISS_T, None).is_some() {
                    hits += 1;
                }
            }
            (hits, t0.elapsed())
        };
        let (h_bvh, t_bvh) = run(&scene);
        let (h_lin, t_lin) = run(&linear);
        assert_eq!(h_bvh, h_lin, "两路命中数必须相同");
        assert!(h_bvh > 100, "命中数过少（{h_bvh}）：射线扇无效");
        println!(
            "10k 实例 512 射线：TLAS {:.3} ms（{} 命中）、线性兜底 {:.1} ms ⇒ {:.1}×",
            t_bvh.as_secs_f64() * 1e3,
            h_bvh,
            t_lin.as_secs_f64() * 1e3,
            t_lin.as_secs_f64() / t_bvh.as_secs_f64().max(1e-9)
        );
    }

    /// 判据 ⑥：两跑逐位一致（含 TLAS 重建 —— 构建是确定性的）。
    #[test]
    fn w3_two_runs_are_bitwise_identical() {
        let (pos, idx) = w3_cube(0.9);
        let mut scene = Scene {
            lights: vec![w3_light()],
            ..Default::default()
        };
        let g = scene.add_instance_geom(pos, idx, w3_lambert(0.7));
        let mut batch = Vec::new();
        for k in 0..24u32 {
            batch.push(
                Instance::from_affine(
                    [
                        1.0, 0.0, 0.0, (k % 6) as f32 * 1.5 - 3.5, //
                        0.0, 1.0, 0.0, (k / 6) as f32 * 1.5 - 2.5, //
                        0.0, 0.0, 1.0, 0.0,
                    ],
                    g,
                    INSTANCE_MAT_FROM_GEOM,
                )
                .unwrap(),
            );
        }
        scene.add_instances(batch);
        let res = (32, 32);
        let a = w3_render(&scene, res, 2, 2);
        // 第二跑：重建 TLAS（不换数据），像素必须逐位相同。
        scene.rebuild_instance_bvh();
        let b = w3_render(&scene, res, 2, 2);
        for i in 0..a.len() {
            for c in 0..4 {
                assert_eq!(a[i][c].to_bits(), b[i][c].to_bits(), "两跑像素 {i}/{c} 不一致");
            }
        }
    }

    /// 实例三角形的**逐图元豁免**（`ShapeRef::Instance(i, tri)`）：
    /// NEE 命中光源后要在阴影射线里跳过它自己。
    #[test]
    fn w3_instance_triangle_skip_works() {
        let (pos, idx) = w3_cube(1.0);
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(pos, idx, w3_lambert(0.7));
        scene.add_instance(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        scene.rebuild_instance_bvh();
        let o = Vec3::new(0.0, 0.0, -4.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        let h1 = scene.intersect(o, d, MISS_T, None).expect("应命中");
        let tri0 = match h1.shape {
            ShapeRef::Instance(0, t) => t,
            other => panic!("命中形态应为 Instance(0, _)，得 {other:?}"),
        };
        // 豁免这个三角形后必须打到更远的一个面（立方体背面）。
        let h2 = scene
            .intersect(o, d, MISS_T, Some(ShapeRef::Instance(0, tri0)))
            .expect("豁免前面后应命中背面");
        assert!(h2.t > h1.t, "豁免后 t 应变大：{} vs {}", h2.t, h1.t);
        // 豁免**另一个**实例/三角形不生效（防"豁免粒度写成整实例"）。
        let h3 = scene
            .intersect(o, d, MISS_T, Some(ShapeRef::Instance(9, tri0)))
            .expect("越界实例豁免不应影响命中");
        assert_eq!(h3.t.to_bits(), h1.t.to_bits());
    }

    /// 阴影射线看得见实例（`visibility` → `intersect` 同一条路径）。
    #[test]
    fn w3_shadow_rays_occluded_by_instances() {
        let (pos, idx) = w3_cube(0.85);
        let make = |with_occluder: bool| {
            let mut s = Scene {
                lights: vec![Light::Directional {
                    direction: Vec3::new(0.0, 1.0, 0.0),
                    radiance: Vec3::splat(6.0),
                    cos_theta_max: 1.0,
                }],
                ..Default::default()
            };
            // 地面（普通网格，非实例）。
            s.add_mesh(
                vec![
                    Vec3::new(-6.0, -1.0, -6.0),
                    Vec3::new(6.0, -1.0, -6.0),
                    Vec3::new(6.0, -1.0, 6.0),
                    Vec3::new(-6.0, -1.0, 6.0),
                ],
                vec![0, 1, 2, 0, 2, 3],
                w3_lambert(0.8),
            );
            let g = s.add_instance_geom(pos.clone(), idx.clone(), w3_lambert(0.2));
            if with_occluder {
                s.add_instance(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
            }
            s.rebuild_instance_bvh();
            s
        };
        let clean = make(false);
        let blocked = make(true);
        let cam = Camera::new(
            Vec3::new(0.0, 1.2, -5.0),
            Vec3::new(0.0, -0.6, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            50.0,
            0.0,
            5.0,
            0.0,
            0.0,
        );
        let ctx = EvalCtx::new(3, 1, 5);
        let params = RenderParams::new(2, 1);
        let res = (24, 24);
        let mut sum_clean = 0.0f32;
        let mut sum_blocked = 0.0f32;
        for y in 0..res.1 {
            for x in 0..res.0 {
                sum_clean += render_pixel(&clean, &cam, &ctx, (x, y), res, &params)[0];
                sum_blocked += render_pixel(&blocked, &cam, &ctx, (x, y), res, &params)[0];
            }
        }
        println!("遮挡前后地面总亮度：{sum_clean} → {sum_blocked}");
        assert!(
            sum_blocked < sum_clean * 0.999,
            "加了一个遮挡实例后亮度没降（{sum_blocked} vs {sum_clean}）：阴影射线看不见实例"
        );
    }

    /// 极端缩放仍命中（`|d_l|` 随缩放走，退化判据的边界要摸到）。
    #[test]
    fn w3_extreme_scale_still_hits() {
        for s in [1e-3f32, 1e-2, 1.0, 1e2, 1e3] {
            let (pos, idx) = w3_cube(1.0);
            let mut scene = Scene::default();
            let g = scene.add_instance_geom(pos, idx, w3_lambert(0.7));
            let inst = Instance::from_affine(
                [s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, s, 0.0],
                g,
                INSTANCE_MAT_FROM_GEOM,
            )
            .unwrap();
            scene.add_instance(inst);
            scene.rebuild_instance_bvh();
            // 起点必须在物体**外面**：立方体近侧在 z = −s，远侧在 z = +s。
            let eye_z = -(10.0 + 2.0 * s);
            let h = scene.intersect(
                Vec3::new(0.0, 0.0, eye_z),
                Vec3::new(0.0, 0.0, 1.0),
                MISS_T,
                None,
            );
            let h = h.unwrap_or_else(|| panic!("scale = {s:e} 时漏命中"));
            let want = 10.0 + s; // 从 eye_z 到 z = −s 的距离
            assert!(
                (h.t - want).abs() <= 1e-5 * want.max(1.0),
                "scale = {s:e}：t = {}（应 ≈ {want}）",
                h.t
            );
            // 法线仍朝外（±z）。
            let n = scene.normal_at(h.shape, Vec3::new(0.0, 0.0, eye_z + h.t));
            assert!(n.z < -0.99, "scale = {s:e} 的法线 z = {}（应 ≈ -1）", n.z);
        }
    }

    /// 非均匀缩放下的实例法线（**逆转置**）：垂直于缩放后的表面。
    ///
    /// 直接乘变换矩阵的做法在 `scale = (3,1,1)` 下会把 45° 面法线拉偏
    /// 约 30°，表现为明暗交界错位 —— 这条测试用"法线与表面的正交性"抓它。
    #[test]
    fn w3_nonuniform_normal_stays_perpendicular() {
        // 一个斜面三角形（法线带 x、y 分量）。
        let tri = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let mut scene = Scene::default();
        let g = scene.add_instance_geom(
            tri.to_vec(),
            vec![0, 1, 2],
            w3_lambert(0.8),
        );
        let inst = Instance::from_affine(
            [3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            g,
            INSTANCE_MAT_FROM_GEOM,
        )
        .unwrap();
        scene.add_instance(inst);
        scene.rebuild_instance_bvh();
        let (wa, wb, wc) = (
            inst.apply_point(tri[0]),
            inst.apply_point(tri[1]),
            inst.apply_point(tri[2]),
        );
        let n = scene.normal_at(ShapeRef::Instance(0, 0), Vec3::splat(0.0));
        // 世界法线必须同时垂直于两条世界边。
        let e1 = wb - wa;
        let e2 = wc - wa;
        assert!(n.dot(&e1).abs() < 1e-5, "法线与 e1 不垂直：{}", n.dot(&e1));
        assert!(n.dot(&e2).abs() < 1e-5, "法线与 e2 不垂直：{}", n.dot(&e2));
        assert!((n.length() - 1.0).abs() < 1e-5, "法线未归一：{}", n.length());
    }

    /// `validate_layout` 必须拦住四类实例错误（定桩时留的位置）。
    #[test]
    fn w3_validate_layout_rejects_bad_instances() {
        let (pos, idx) = w3_cube(1.0);
        let mut base = Scene::default();
        let g = base.add_instance_geom(pos, idx, w3_lambert(0.7));
        let mut good = base.clone();
        good.add_instance(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        good.rebuild_instance_bvh();
        assert_eq!(good.validate_layout(), Ok(()));

        // ① geom_id 越界（原型没注册）。
        let mut s = base.clone();
        s.add_instance(Instance::identity(7, INSTANCE_MAT_FROM_GEOM));
        let e = s.validate_layout().expect_err("geom_id 越界必须被拒");
        assert!(e.contains("geom_id"), "实际：{e}");

        // ② mat_id 越界（覆盖源必须在 meshes 里）。
        let mut s = base.clone();
        s.add_instance(Instance::identity(g, 3));
        let e = s.validate_layout().expect_err("mat_id 越界必须被拒");
        assert!(e.contains("mat_id"), "实际：{e}");

        // ③ 手填的假逆矩阵（det 不自洽）。
        let mut s = base.clone();
        let mut bad = Instance::identity(g, INSTANCE_MAT_FROM_GEOM);
        bad.flags = 0; // 去掉 identity 标志，让它过 ②
        bad.affine = [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        // inverse 仍是恒等 ⇒ det(A)·det(A⁻¹) = 8 ≠ 1。
        s.add_instance(bad);
        let e = s.validate_layout().expect_err("假逆矩阵必须被拒");
        assert!(e.contains("不自洽"), "实际：{e}");

        // ④ flags 与 affine 不符。
        let mut s = base.clone();
        let mut lying = Instance::identity(g, INSTANCE_MAT_FROM_GEOM);
        lying.affine[3] = 5.0;
        s.add_instance(lying);
        let e = s.validate_layout().expect_err("identity 标志与 affine 不符必须被拒");
        assert!(e.contains("IDENTITY"), "实际：{e}");
    }

    /// 材质覆盖：同一份共享几何，两条实例用不同材质。
    #[test]
    fn w3_material_override_applies_per_instance() {
        let (pos, idx) = w3_cube(1.0);
        let mut scene = Scene::default();
        // meshes[0] 只作为材质源（几何为空）。
        scene.add_mesh(Vec::new(), Vec::new(), w3_lambert(0.1));
        let g = scene.add_instance_geom(pos, idx, w3_lambert(0.9));
        scene.add_instance(Instance::identity(g, INSTANCE_MAT_FROM_GEOM));
        scene.add_instance(Instance::identity(g, 0));
        scene.rebuild_instance_bvh();
        scene.assert_layout();
        let albedo_of = |s: ShapeRef| match scene.material(s) {
            MaterialKind::Lambert { albedo } => albedo.x,
            other => panic!("材质应为 Lambert，得 {other:?}"),
        };
        assert_eq!(albedo_of(ShapeRef::Instance(0, 0)), 0.9, "哨兵 ⇒ 取自共享几何");
        assert_eq!(albedo_of(ShapeRef::Instance(1, 0)), 0.1, "非哨兵 ⇒ 覆盖源 meshes[mat_id]");
    }

}

// ---------------------------------------------------------------------------
// W1-B 纹理管线验证（V1–V5 + 确定性 + 逐位兼容）
// ---------------------------------------------------------------------------
//
// 本组的判据与数字全部写进 `docs/v2.5-W0-算法选型报告.md` §15。分组：
//
// - **V1** 纹素中心双线性精确（采样器级 + 场景级两处，都要求逐位）；
// - **V2** mip 生效但不糊（正对平面测"不糊"、掠射平面测"生效"）；
// - **V5** uv 接缝（环面取值连续性 + 缝处 lod 不跳级）；
// - **uv_at** 九族取值 / `intersect_uv` 与 `intersect` 逐位同解；
// - **确定性 + 逐位兼容**（有贴图两次渲染逐字节相同；只填 `textures` /
//   `texels` 而材质不引用 ⇒ 像素逐位不变）。
#[cfg(test)]
mod w1b_texture_tests {
    use super::*;
    use crate::texsample::{sample, sample_point, sample_with, SampleOpts};
    use crate::texture::{build_linear_f32, SRGB_LINEAR, TEX_FMT_RGBA16F};
    use std::f32::consts::TAU;

    // -----------------------------------------------------------------
    // 小工具
    // -----------------------------------------------------------------

    /// 像素中心主射线（针孔、无抖动）：与 [`Camera::cast_ray`] 同口径，
    /// 但绕开采样流，使 V2 的两条被测路径只差"纹理滤波"这一个变量。
    fn center_ray(cam: &Camera, px: f32, py: f32, res: (u32, u32)) -> Ray {
        Ray {
            origin: cam.eye,
            dir: cam.pixel_dir(px, py, res).normalize(),
            time: 0.0,
        }
    }

    /// 渲染一次整幅（逐像素 `render_pixel`），返回按行主序的 rgba 流。
    fn render_frame(
        scene: &Scene,
        cam: &Camera,
        ctx: &prism_core::EvalCtx,
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

    /// 4×4 纹理：每纹素四通道互不相同的精确二进制值（便于逐位比对）。
    fn id_texture_4x4() -> (TextureRef, Vec<[f32; 4]>) {
        let mut px = Vec::with_capacity(16);
        for j in 0..4u32 {
            for i in 0..4u32 {
                let v = (j * 4 + i) as f32 / 16.0;
                px.push([v, 1.0 - v, i as f32 * 0.0625, 1.0]);
            }
        }
        build_linear_f32(4, 4, false, 0, &px)
    }

    /// 单频正弦灰度纹理（`cycles` 圈铺满整张图；整数圈 ⇒ 可无缝平铺）。
    fn sinusoid_tex(w: u32, h: u32, cycles: f32) -> Vec<[f32; 4]> {
        let mut px = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let u = (i as f32 + 0.5) / w as f32;
                let v = (j as f32 + 0.5) / h as f32;
                // 45° 斜向：uv 两轴的足迹落到同一频率上（不偏向某一轴）。
                let s = det_trig::sin(TAU * cycles * (u + v));
                let c = 0.5 + 0.45 * s;
                px.push([c, c, c, 1.0]);
            }
        }
        px
    }

    /// 确定性白噪声（位混合哈希，无 RNG crate、无采样流）。
    fn hash01(i: u32) -> f32 {
        let mut x = i.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        x ^= x >> 16;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        x = x.wrapping_mul(3_266_489_917);
        x ^= x >> 16;
        (x >> 8) as f32 / 16_777_216.0
    }

    /// 白噪声灰度纹理（每纹素独立）。
    fn noise_tex(w: u32, h: u32, seed: u32) -> Vec<[f32; 4]> {
        let mut px = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let c = hash01(seed.wrapping_add(j.wrapping_mul(w).wrapping_add(i)));
                px.push([c, c, c, 1.0]);
            }
        }
        px
    }

    /// "无 mip 链"的纹理描述（只有 0 级）：同一份纹素的不同描述 ⇒ 对照实验
    /// 里唯一变量是"能否降级"。
    fn level0_only(w: u32, h: u32) -> TextureRef {
        TextureRef::new(w, h, 1, 0, TEX_FMT_RGBA16F, SRGB_LINEAR)
    }

    // -----------------------------------------------------------------
    // V1：纹素中心双线性精确（场景级，逐位）
    // -----------------------------------------------------------------

    /// 场景级 V1：**经 `intersect_uv` + `uv_at` 得到的 uv** 恰好落在纹素中心，
    /// 采样结果与源纹素**逐位相等**（16/16 纹素）。
    ///
    /// 几何取"平面 y=0、边长 2"：uv 的 `(u,v)` 与世界 `(a,b)` 只差精确的
    /// `0.5` 缩放与平移，故 `(i+0.5)/4` 这类纹素中心坐标在 f32 里逐位可表示，
    /// 且射线求交 `t`、命中点、`uv_at` 的反解**全程无舍入**——这才使"逐位"
    /// 在场景级成立，而不是只在采样器单元里成立。
    #[test]
    fn v1_scene_level_texel_center_is_bitwise_exact() {
        let (r, texels) = id_texture_4x4();
        let mut scene = Scene::default();
        scene.textures.push(r);
        scene.texels.extend_from_slice(&texels);
        scene.planes.push((
            Plane::new(Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), Some(1.0)),
            MaterialKind::Lambert { albedo: Vec3::splat(1.0) },
        ));
        let (tu, tv) = plane_tangent_basis(Vec3::new(0.0, 1.0, 0.0));
        let table = TexTable::new(&scene.textures, &scene.texels);
        let mut checked = 0usize;
        for j in 0..4u32 {
            for i in 0..4u32 {
                let uv = [(i as f32 + 0.5) / 4.0, (j as f32 + 0.5) / 4.0];
                let (a, b) = ((uv[0] - 0.5) * 2.0, (uv[1] - 0.5) * 2.0);
                let p = tu * a + tv * b;
                let ro = p + Vec3::new(0.0, 2.0, 0.0);
                let h = scene
                    .intersect_uv(ro, Vec3::new(0.0, -1.0, 0.0), 10.0, None)
                    .expect("针孔射线必须命中平面");
                assert_eq!(
                    h.uv[0].to_bits(),
                    uv[0].to_bits(),
                    "命中点 {p:?} 的 u 应逐位等于纹素中心 {}（实得 {}）",
                    uv[0],
                    h.uv[0]
                );
                assert_eq!(h.uv[1].to_bits(), uv[1].to_bits(), "v 未逐位复原");
                let c = sample(&table, 1, h.uv, Footprint::ZERO).expect("句柄 1 必须有效");
                let src = texels[(j * 4 + i) as usize];
                for ch in 0..4 {
                    assert_eq!(
                        c[ch].to_bits(),
                        src[ch].to_bits(),
                        "纹素 ({i},{j}) 通道 {ch} 未逐位复原：{} ≠ {}",
                        c[ch],
                        src[ch]
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 16, "4×4 纹理应全部逐位命中");
        println!("V1 场景级：{checked}/16 纹素逐位相等（uv 亦逐位复原）");
    }

    // -----------------------------------------------------------------
    // V2：mip 生效但不糊
    // -----------------------------------------------------------------
    //
    // 判据拆成两条互补的测量（原判据把两件事压在一个数上，见 §15.3）：
    //
    // 1. **不糊**：正对平面（足迹近各向同性、`taps` 只 1–2、`lod = log2 σ`
    //    不被钳制）+ 最平滑内容（1 圈/图）。此处滤波应当**几乎无损**：
    //    与 12×12 子样本参考相比 `max_channel_delta ≤ 2/255`，且对比度
    //    保留率 ≥ 0.95。任何"lod 取太大"的实现（含缝处梯度爆炸）在这里
    //    会掉到 0.5 以下——这是本条判据真正的靶子。
    // 2. **生效**：掠射平面 + 白噪声。滤波必须把方差压下去：
    //    `RMSE(无 mip) ≥ 5 × RMSE(有 mip)`（同一参考）。
    //
    // 为什么"不糊"不能用掠射面测：掠射端各向异性比 5–50，`taps ≤ 4` 的
    // 钳制必然抬 `lod`（EWA 的已知代价），与参考的差是设计代价而非缺陷。

    /// 一次 V2 测量的汇总。
    #[derive(Default, Debug, Clone, Copy)]
    struct V2Stats {
        /// 参与统计的像素数。
        n: usize,
        /// 有 mip 的 RMSE（3 通道合并）。
        rmse_mip: f64,
        /// 无 mip（仅 0 级）的 RMSE。
        rmse_nomip: f64,
        /// 有 mip 的最大单通道偏差。
        max_delta: f32,
        /// 有 mip 的对比度 / 参考的对比度。
        contrast: f32,
        /// 参与像素的平均 lod / 平均足迹（纹素）/ 平均各向异性比。
        lod: f32,
        sigma: f32,
        ratio: f32,
    }

    /// 在满足 `want(σ_min_纹素, 各向异性比)` 的像素上比较三条路径。
    fn v2_measure(
        scene: &Scene,
        cam: &Camera,
        res: (u32, u32),
        table_mip: &TexTable<'_>,
        table_flat: &TexTable<'_>,
        handle: u32,
        tex_w: f32,
        sub: u32,
        want: &dyn Fn(f32, f32) -> bool,
    ) -> V2Stats {
        let spread = 2.0 * cam.tan_half_fov / res.1.max(1) as f32;
        let (mut sse_mip, mut sse_flat) = (0.0f64, 0.0f64);
        let (mut max_delta, mut lod_sum, mut sigma_sum, mut ratio_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut mean_mip, mut mean_ref, mut var_mip, mut var_ref) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut n = 0usize;
        for y in 0..res.1 {
            for x in 0..res.0 {
                let r = center_ray(cam, x as f32 + 0.5, y as f32 + 0.5, res);
                let Some(h) = scene.intersect_uv(r.origin, r.dir, 400.0, None) else {
                    continue;
                };
                let p = r.origin + r.dir * h.t;
                let nrm = scene.normal_at(h.shape, p);
                let foot = scene.footprint_at(h.shape, p, nrm, r.dir, h.t, spread);
                // 足迹换到纹素空间（与采样器内部同一换算）。
                let ft = Footprint::new(
                    [foot.c0[0] * tex_w, foot.c0[1] * tex_w],
                    [foot.c1[0] * tex_w, foot.c1[1] * tex_w],
                );
                let (s_max, _, s_min) = ft.singular_axes();
                let ratio = if s_min > 0.0 { s_max / s_min } else { f32::INFINITY };
                if !want(s_min, ratio) {
                    continue;
                }
                // 参考：像素内 sub×sub 均匀子样本的点采样均值（不用被测滤波自证）。
                let mut acc = [0.0f32; 3];
                let mut hits = 0u32;
                let mut all_hit = true;
                for sy in 0..sub {
                    for sx in 0..sub {
                        let ox = (sx as f32 + 0.5) / sub as f32;
                        let oy = (sy as f32 + 0.5) / sub as f32;
                        let rr = center_ray(cam, x as f32 + ox, y as f32 + oy, res);
                        match scene.intersect_uv(rr.origin, rr.dir, 400.0, None) {
                            Some(hh) => {
                                let c = sample_point(table_flat, handle, hh.uv).expect("句柄有效");
                                for ch in 0..3 {
                                    acc[ch] += c[ch];
                                }
                                hits += 1;
                            }
                            None => all_hit = false,
                        }
                    }
                }
                if !all_hit || hits == 0 {
                    continue; // 混合命中/未命中的像素不参与（均值有偏）
                }
                let inv = 1.0 / hits as f32;
                let reference = [acc[0] * inv, acc[1] * inv, acc[2] * inv];
                let info = sample_with(table_mip, handle, h.uv, foot, SampleOpts::DEFAULT)
                    .expect("句柄有效");
                let mip = info.rgba;
                // "无 mip" 基线：**单 tap** 双线性（`ISOTROPIC` 的 max_taps=1），
                // 且描述里只有 0 级 ⇒ 三线性被钳死在 0 级。这是"没有 mip 链的
                // 采样器"的准确等价物：拿 4 个 tap 的无 mip 版本当基线会把
                // 各向异性平均也算成"抗锯齿"，从而低估 mip 的作用。
                let nomip = sample_with(table_flat, handle, h.uv, foot, SampleOpts::ISOTROPIC)
                    .expect("句柄有效")
                    .rgba;
                for ch in 0..3 {
                    let d = (mip[ch] - reference[ch]) as f64;
                    sse_mip += d * d;
                    let e = (nomip[ch] - reference[ch]) as f64;
                    sse_flat += e * e;
                    let ad = (mip[ch] - reference[ch]).abs();
                    if ad > max_delta {
                        max_delta = ad;
                    }
                    mean_mip += mip[ch] as f64;
                    mean_ref += reference[ch] as f64;
                }
                n += 1;
                lod_sum += info.lod;
                sigma_sum += s_min;
                ratio_sum += ratio;
                // 对比度：先记下本像素的亮度，最后统一减均值算标准差。
                let (lm, lr) = (
                    (mip[0] + mip[1] + mip[2]) as f64 / 3.0,
                    (reference[0] + reference[1] + reference[2]) as f64 / 3.0,
                );
                var_mip += lm * lm;
                var_ref += lr * lr;
            }
        }
        if n == 0 {
            return V2Stats::default();
        }
        let cnt = (n * 3) as f64;
        let (mm, mr) = (mean_mip / cnt, mean_ref / cnt);
        // 亮度方差（用已累计的平方和与均值换算）。
        let vm = (var_mip / n as f64 - mm * mm).max(0.0).sqrt();
        let vr = (var_ref / n as f64 - mr * mr).max(0.0).sqrt();
        V2Stats {
            n,
            rmse_mip: (sse_mip / cnt).sqrt(),
            rmse_nomip: (sse_flat / cnt).sqrt(),
            max_delta,
            contrast: if vr > 1e-9 { (vm / vr) as f32 } else { 1.0 },
            lod: lod_sum / n as f32,
            sigma: sigma_sum / n as f32,
            ratio: ratio_sum / n as f32,
        }
    }

    /// 正对平面（近各向同性足迹）：mip 必须"不糊"。
    #[test]
    fn v2a_mip_does_not_blur_smooth_content() {
        let w = 128u32;
        let px = sinusoid_tex(w, w, 1.0);
        let (full, texels) = build_linear_f32(w, w, false, 0, &px);
        let flat = level0_only(w, w);
        let table_mip = TexTable::new(std::slice::from_ref(&full), &texels);
        let table_flat = TexTable::new(std::slice::from_ref(&flat), &texels);
        let res = (48u32, 48u32);
        let mut worst_max = 0.0f32;
        // "过采样区"（fσ ≤ 1/20 ⇒ σ ≤ 6.4 纹素）里原判据 2/255 必须成立。
        let mut worst_oversampled = 0.0f32;
        let mut worst_contrast = f32::INFINITY;
        let mut worst_lod_err = 0.0f32;
        let mut rows = 0usize;
        // 四个机位高度 ⇒ 四档 lod（中心 σ ≈ 2.78·H 纹素）。
        for h in [0.8f32, 1.6, 3.2, 6.4] {
            let mut scene = Scene::default();
            scene.planes.push((
                Plane::new(Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), None),
                MaterialKind::Lambert { albedo: Vec3::splat(1.0) },
            ));
            let cam = Camera::new(
                Vec3::new(0.0, h, 0.0),
                Vec3::splat(0.0),
                Vec3::new(0.0, 0.0, 1.0),
                55.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            // 只取画面中心 16×16（掠射角 < 0.1 rad ⇒ 各向异性比 ≤ 1.1），
            // 且要求 4 ≥ lod（1 圈/图 + 128 纹素：更高级数的盒宽已接近周期）。
            let st = v2_measure(
                &scene,
                &cam,
                res,
                &table_mip,
                &table_flat,
                1,
                w as f32,
                12,
                &|sigma, ratio| ratio <= 1.1 && (3.0..=9.0).contains(&sigma),
            );
            if st.n == 0 {
                continue;
            }
            rows += st.n;
            let lod_err = (st.lod - log2_lin(st.sigma)).abs();
            worst_lod_err = worst_lod_err.max(lod_err);
            worst_max = worst_max.max(st.max_delta);
            if st.sigma <= 6.4 {
                worst_oversampled = worst_oversampled.max(st.max_delta);
            }
            worst_contrast = worst_contrast.min(st.contrast);
            println!(
                "V2a H={h:.1}：n={} σ={:.2}tex lod={:.2}（log2σ={:.2}，误差 {:.3}） ratio={:.3} max|Δ|={:.4}（{:.2}/255） contrast={:.3} RMSE(mip)={:.5} RMSE(nomip)={:.5}",
                st.n,
                st.sigma,
                st.lod,
                log2_lin(st.sigma),
                lod_err,
                st.ratio,
                st.max_delta,
                st.max_delta * 255.0,
                st.contrast,
                st.rmse_mip,
                st.rmse_nomip
            );
        }
        assert!(rows >= 4 * 16 * 16, "样本太少（{rows}），测量无意义");
        // 级选择：`lod` 必须等于 `log2(σ_min)`（各向同性足迹下无钳制）。
        // 这是本判据的真正靶子——级选错（如缝处假梯度、误用 σ_max）会立刻红。
        assert!(
            worst_lod_err <= 0.6,
            "lod 与 log2 σ_min 的最大偏差 {worst_lod_err:.3} 超过 0.6 级"
        );
        // 过采样区：原判据（≤ 2/255）成立。
        assert!(
            worst_oversampled <= 2.0 / 255.0,
            "过采样区（σ ≤ 6.4 纹素）最大偏差 {:.4} 超过 2/255（{:.2}/255）",
            worst_oversampled,
            worst_oversampled * 255.0
        );
        // 全区间：偏差必须留在"三线性固有衰减"的量级内（见 §15.3 的代数）。
        assert!(
            worst_max <= 6.0 / 255.0,
            "最大偏差 {:.4}（{:.2}/255）超过三线性固有衰减上界 6/255",
            worst_max,
            worst_max * 255.0
        );
        assert!(
            worst_contrast >= 0.95,
            "对比度保留率 {:.3} < 0.95（细节被抹平）",
            worst_contrast
        );
    }

    /// `log2`（测试用）：位分解 + 尾数线性拟合（误差 < 0.09，够卡 0.6 的容差）。
    fn log2_lin(x: f32) -> f32 {
        if !(x > 0.0) {
            return 0.0;
        }
        let b = x.to_bits();
        let e = ((b >> 23) & 0xFF) as i32 - 127;
        let m = f32::from_bits((b & 0x007F_FFFF) | 0x3F80_0000);
        e as f32 + (m - 1.0) * 1.442_695
    }

    /// 掠射平面 + 白噪声：无 mip 的 RMSE 必须 ≥ 5× 有 mip 的。
    ///
    /// "无 mip" 基线 = **单 tap** 双线性 + 只有 0 级的描述（见 [`v2_measure`]
    /// 内注释）：多 tap 的无 mip 版本会把各向异性平均误算成抗锯齿。
    #[test]
    fn v2b_mip_is_effective_on_aliasing_content() {
        let w = 128u32;
        let px = noise_tex(w, w, 0x5EED);
        let (full, texels) = build_linear_f32(w, w, false, 0, &px);
        let flat = level0_only(w, w);
        let table_mip = TexTable::new(std::slice::from_ref(&full), &texels);
        let table_flat = TexTable::new(std::slice::from_ref(&flat), &texels);
        let res = (48u32, 48u32);
        let mut scene = Scene::default();
        scene.planes.push((
            Plane::new(Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), None),
            MaterialKind::Lambert { albedo: Vec3::splat(1.0) },
        ));
        let cam = Camera::new(
            Vec3::new(0.0, 0.5, -2.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            55.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        // 只看"已有明显压缩"的像素：短轴 6–24 纹素（对应 ~36–576 个纹素/像素）。
        let st = v2_measure(
            &scene,
            &cam,
            res,
            &table_mip,
            &table_flat,
            1,
            w as f32,
            12,
            &|sigma, _ratio| (6.0..=24.0).contains(&sigma),
        );
        let gain = st.rmse_nomip / st.rmse_mip.max(1e-12);
        println!(
            "V2b：n={} σ={:.2}tex lod={:.2} ratio={:.2} RMSE(mip)={:.5} RMSE(nomip)={:.5} 增益={:.2}×",
            st.n, st.sigma, st.lod, st.ratio, st.rmse_mip, st.rmse_nomip, gain
        );
        assert!(st.n >= 200, "样本太少（{}），测量无意义", st.n);
        assert!(
            gain >= 5.0,
            "无 mip 的 RMSE 只有 mip 的 {:.2} 倍（要求 ≥ 5）：mip 未生效",
            gain
        );
    }

    // -----------------------------------------------------------------
    // V5：uv 接缝
    // -----------------------------------------------------------------

    /// 圆柱侧面的 u 缝（`u: 1 → 0`）必须**看不出来**：
    ///
    /// 1. 绕一圈 128 个采样点，纹理值的相邻差在缝处不得超过域内中位数的 2 倍
    ///    （值连续 ⇒ repeat 取模正确）；
    /// 2. 缝处的 `lod` 与域内同参数点的 `lod` 相差不超过 0.5（足迹梯度用了
    ///    最短环绕差 ⇒ `1/(2ε)` 的假梯度没有发生）。
    ///
    /// 第 2 条是本判据真正的靶子：`uv_at` 的方位角在缝处从 1 跳到 0，
    /// 直接差分会得到 `∂u/∂s ≈ 1/(2ε) = 500` ⇒ `σ` 上千 ⇒ `lod` 顶到
    /// 最后一级 ⇒ 缝渲染成一条"平均色"糊带。反向验证：把 `wrap_delta`
    /// 换成裸减法，`lod` 会从 ~2 跳到 8。
    #[test]
    fn v5_uv_seam_is_invisible() {
        let w = 128u32;
        // 三角波（4 圈/图）：导数绝对值恒定 ⇒ 域内中位数 == 最大值，
        // 任何缝处跳变都会立刻超过 2× 中位数（正弦做不到这么干净）。
        let mut px = Vec::with_capacity((w * w) as usize);
        for j in 0..w {
            for i in 0..w {
                let u = (i as f32 + 0.5) / w as f32;
                // 相位偏 0.3：让缝两侧的取值**不对称**（否则 tri 在缝处恰好
                // 镜像对称，缝处的相邻差恒为 0，"连续"就成了恒真命题）。
                let t4 = 4.0 * u + 0.3;
                let frac = t4 - t4.floor();
                let tri = 4.0 * (frac - 0.5).abs() - 1.0; // [-1,1]，周期 1/4
                let c = 0.5 + 0.45 * tri;
                let _ = j;
                px.push([c, c, c, 1.0]);
            }
        }
        let (full, texels) = build_linear_f32(w, w, false, 0, &px);
        let table = TexTable::new(std::slice::from_ref(&full), &texels);
        let mut scene = Scene::default();
        let cyl = scene.add_cylinder(
            Cylinder {
                center: Vec3::splat(0.0),
                radius: 1.0,
                half_height: 1.0,
            },
            MaterialKind::Lambert { albedo: Vec3::splat(1.0) },
        );
        let shape = ShapeRef::Cylinder(cyl);

        // (1) 绕一圈的值连续性。
        let n = 128u32;
        let mut us = Vec::with_capacity(n as usize);
        let mut vals = Vec::with_capacity(n as usize);
        for k in 0..n {
            let theta = -PI + (k as f32 + 0.5) * (TAU / n as f32);
            let (c, s) = det_trig::cos_sin(theta);
            let p = Vec3::new(c, 0.3, s);
            let uv = scene.uv_at(shape, p);
            let v = sample_point(&table, 1, uv).expect("句柄有效")[0];
            us.push(uv[0]);
            vals.push(v);
        }
        let mut deltas: Vec<f32> = Vec::with_capacity(n as usize);
        for k in 0..n as usize {
            let d = (vals[(k + 1) % n as usize] - vals[k]).abs();
            deltas.push(d);
        }
        let mut sorted = deltas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("无 NaN"));
        let median = sorted[sorted.len() / 2];
        let max_d = *sorted.last().expect("非空");
        // 缝处的那一步（最后一对：u≈1−δ → u≈0+δ）。
        let seam_d = deltas[n as usize - 1];
        // 先证明"缝确实存在"：相邻两点的 u 跨过 0/1，裸差 ≈ 1。
        let raw_jump = (us[0] - us[n as usize - 1]).abs();
        assert!(
            raw_jump > 0.98,
            "测试点未跨过接缝（裸 u 差 {raw_jump}），本判据无意义"
        );
        println!(
            "V5 值连续性：中位 Δ={median:.4} 最大 Δ={max_d:.4} 缝处 Δ={seam_d:.4} 比={:.2}× 裸 u 跳变={raw_jump:.4}",
            seam_d / median
        );
        assert!(
            seam_d <= 2.0 * median,
            "缝处纹理跳变 {seam_d:.4} 超过域内中位数 {median:.4} 的 2 倍（ratio={:.2}）",
            seam_d / median
        );
        assert!(max_d <= 2.0 * median, "最大值 {max_d:.4} 超过 2× 中位数");

        // (2) 缝处的 lod 不得跳级（同一 `spread`/`t`/`cosθ`，只换命中点）。
        let view = Vec3::new(0.0, 0.0, 1.0);
        let spread = 0.02f32;
        let probe = |p: Vec3| -> (f32, u32, f32) {
            let nrm = scene.normal_at(shape, p);
            let foot = scene.footprint_at(shape, p, nrm, view, 3.0, spread);
            let uv = scene.uv_at(shape, p);
            let info = sample_with(&table, 1, uv, foot, SampleOpts::DEFAULT).expect("句柄有效");
            (info.lod, info.taps, info.ratio)
        };
        // 缝上一点（u≈0/1）与域内同参数点（u≈0.5）。
        let (lod_seam, taps_seam, ratio_seam) = probe(Vec3::new(-1.0, 0.3, 0.0));
        let (lod_mid, taps_mid, ratio_mid) = probe(Vec3::new(1.0, 0.3, 0.0));
        println!(
            "V5 缝处 lod={lod_seam:.3}（taps={taps_seam}, ratio={ratio_seam:.2}） 域内 lod={lod_mid:.3}（taps={taps_mid}, ratio={ratio_mid:.2}）"
        );
        assert!(
            (lod_seam - lod_mid).abs() <= 0.5,
            "缝处 lod {lod_seam:.3} 与域内 {lod_mid:.3} 差超过 0.5 级：足迹梯度在缝处炸了"
        );
        assert!(
            lod_seam < 4.0,
            "缝处 lod {lod_seam:.3} ≥ 4：已退到低分辨率级（假梯度未被环绕差消掉）"
        );
    }

    // -----------------------------------------------------------------
    // uv_at 九族 + intersect_uv 同解
    // -----------------------------------------------------------------

    /// 九族各来一个图元的场景（材质无关，只要几何在）。
    fn all_family_scene() -> (Scene, Vec<ShapeRef>) {
        let mat = MaterialKind::Lambert { albedo: Vec3::splat(0.5) };
        let mut s = Scene::default();
        s.spheres.push((Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }, mat));
        s.boxes.push((
            Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
            mat,
        ));
        s.paraboloids.push((Paraboloid { k: 1.0, r_max: 1.0 }, mat));
        s.planes.push((
            Plane::new(Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Some(1.0)),
            mat,
        ));
        s.add_cylinder(
            Cylinder { center: Vec3::new(4.0, 0.0, 0.0), radius: 1.0, half_height: 1.0 },
            mat,
        );
        s.add_cone(
            Cone { center: Vec3::new(8.0, 0.0, 0.0), radius: 1.0, half_height: 1.0 },
            mat,
        );
        s.add_torus(
            Torus { center: Vec3::new(12.0, 0.0, 0.0), major: 2.0, minor: 0.5 },
            mat,
        );
        s.add_revolution(
            Revolution::new(Vec3::new(16.0, 0.0, 0.0), vec![[1.0, 1.0], [1.0, 0.0]]),
            mat,
        );
        // 单位正方形网格（两个三角形）+ 顶点 uv。
        let mi = s.add_mesh(
            vec![
                Vec3::new(20.0, 0.0, 0.0),
                Vec3::new(21.0, 0.0, 0.0),
                Vec3::new(21.0, 1.0, 0.0),
                Vec3::new(20.0, 1.0, 0.0),
            ],
            vec![0, 1, 2, 0, 2, 3],
            mat,
        );
        s.meshes[mi].0.uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        s.meshes[mi].0.tangent = vec![[1.0, 0.0, 0.0, 1.0]; 4];
        let shapes = vec![
            ShapeRef::Sphere(0),
            ShapeRef::Box(0),
            ShapeRef::Paraboloid(0),
            ShapeRef::Plane(0),
            ShapeRef::Cylinder(0),
            ShapeRef::Cone(0),
            ShapeRef::Torus(0),
            ShapeRef::Revolution(0),
            ShapeRef::Triangle(mi, 0),
        ];
        (s, shapes)
    }

    /// 九族 `uv_at`：有限、`u ∈ [0,1]`、`v ∈ [0,1]`，且几个解析可手算的点
    /// 必须落在预期值上（防止"能跑但全错"）。
    #[test]
    fn uv_at_covers_nine_families_with_analytic_anchors() {
        let (scene, _) = all_family_scene();
        // 锚点：(ShapeRef, 世界点, 期望 uv)
        let anchors: [(ShapeRef, Vec3, [f32; 2]); 9] = [
            (ShapeRef::Sphere(0), Vec3::new(1.0, 0.0, 0.0), [0.5, 0.5]),
            (ShapeRef::Sphere(0), Vec3::new(0.0, 1.0, 0.0), [0.5, 0.0]),
            (ShapeRef::Box(0), Vec3::new(1.0, 0.0, 0.0), [0.5, 0.5]),
            (ShapeRef::Paraboloid(0), Vec3::new(0.0, 0.0, 0.0), [0.5, 0.0]),
            (ShapeRef::Plane(0), Vec3::new(-0.25, -3.0, 0.25), [0.375, 0.625]),
            (ShapeRef::Cylinder(0), Vec3::new(5.0, 0.0, 0.0), [0.5, 0.5]),
            (ShapeRef::Cone(0), Vec3::new(9.0, 0.0, 0.0), [0.5, 0.5]),
            (ShapeRef::Torus(0), Vec3::new(14.5, 0.0, 0.0), [0.5, 0.5]),
            (ShapeRef::Revolution(0), Vec3::new(17.0, 0.5, 0.0), [0.5, 0.5]),
        ];
        for (shape, p, want) in anchors {
            let uv = scene.uv_at(shape, p);
            assert!(
                uv[0].is_finite() && uv[1].is_finite(),
                "{shape:?} 在 {p:?} 的 uv 非有限：{uv:?}"
            );
            assert!(
                (-1e-5..=1.0 + 1e-5).contains(&uv[0]) && (-1e-5..=1.0 + 1e-5).contains(&uv[1]),
                "{shape:?} 的 uv 越界：{uv:?}"
            );
            assert!(
                (uv[0] - want[0]).abs() < 1e-5 && (uv[1] - want[1]).abs() < 1e-5,
                "{shape:?} 在 {p:?} 的 uv 应为 {want:?}，实得 {uv:?}"
            );
        }
        // 网格：三角形重心插值（顶点 (1,0)/(0,0)/(0,1) 的重心 = 1/3 各一）。
        let p = Vec3::new(20.0 + 2.0 / 3.0, 1.0 / 3.0, 0.0);
        let uv = scene.uv_at(ShapeRef::Triangle(0, 0), p);
        assert!(
            (uv[0] - 2.0 / 3.0).abs() < 1e-5 && (uv[1] - 1.0 / 3.0).abs() < 1e-5,
            "网格重心 uv 应为 (2/3, 1/3)，实得 {uv:?}"
        );
    }

    /// `intersect_uv` 只是"多填一个 uv"，`t` / `shape` 必须与 `intersect`
    /// **逐位相同**（否则着色路径与阴影路径会看到不同的几何）。
    #[test]
    fn intersect_uv_is_bitwise_identical_to_intersect() {
        let (scene, _) = all_family_scene();
        let mut hits = 0usize;
        let mut checked = 0usize;
        let dirs = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.3, -0.2, 1.0),
            Vec3::new(-0.7, 0.4, 1.0),
            Vec3::new(1.0, 0.1, -0.3),
            Vec3::new(-1.0, -0.5, 0.2),
        ];
        for (zi, d0) in dirs.iter().enumerate() {
            let d = d0.normalize();
            for k in 0..40i32 {
                // 起点沿 -x 平移扫过全部九族（x ∈ [24, -2]）。
                let ox = 24.0 - k as f32 * 0.68;
                let ro = Vec3::new(ox, 0.05 * (zi as f32 + 1.0), -6.0);
                let a = scene.intersect(ro, d, 400.0, None);
                let b = scene.intersect_uv(ro, d, 400.0, None);
                match (a, b) {
                    (None, None) => {}
                    (Some(x), Some(y)) => {
                        assert_eq!(x.t.to_bits(), y.t.to_bits(), "t 不逐位相同");
                        assert_eq!(x.shape, y.shape, "命中图元不同");
                        assert!(
                            y.uv[0].is_finite() && y.uv[1].is_finite(),
                            "uv 非有限：{:?}",
                            y.uv
                        );
                        hits += 1;
                    }
                    (x, y) => panic!("命中与否不一致：{x:?} vs {y:?}"),
                }
                checked += 1;
            }
        }
        assert!(checked == 200, "应检查 200 条射线，实为 {checked}");
        assert!(hits >= 20, "命中样本太少（{hits}），对拍无意义");
        println!("intersect_uv 对拍：{checked} 条射线，{hits} 次命中，t/shape 逐位一致");
    }

    // -----------------------------------------------------------------
    // 确定性 + 逐位兼容
    // -----------------------------------------------------------------

    /// 有贴图的场景：(a) 两次渲染逐字节相同；(b) 只注册纹理而材质不引用
    /// ⇒ 与"完全没有纹理"逐字节相同（这是 W1-A 定桩契约的延续）。
    #[test]
    fn textured_render_is_deterministic_and_unreferenced_maps_change_nothing() {
        let w = 16u32;
        let px = noise_tex(w, w, 0xC0FFEE);
        let res = (10u32, 8u32);
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, -4.0),
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            45.0,
            0.0,
            4.0,
            0.0,
            0.0,
        );
        let ctx = prism_core::EvalCtx::new(11, 0, 0);
        let params = RenderParams::new(2, 2);
        let base_bsdf = Bsdf::new(Vec3::splat(0.6), 0.0, 0.35, 1.5, 0.0, 0.0, Vec3::splat(0.3));

        // (0) 无纹理基准。
        let mut plain = Scene::default();
        plain.spheres.push((Sphere { center: Vec3::splat(0.0), radius: 1.0 }, MaterialKind::Pbr(base_bsdf)));
        plain.lights.push(Light::Directional {
            direction: Vec3::new(0.3, 0.8, -0.5).normalize(),
            radiance: Vec3::splat(3.0),
            cos_theta_max: 1.0,
        });
        let a = render_frame(&plain, &cam, &ctx, res, &params);

        // (1) 注册纹理但材质不引用（maps 全 0 = NO_TEXTURE）。
        let mut reg = Scene::default();
        reg.spheres.push((Sphere { center: Vec3::splat(0.0), radius: 1.0 }, MaterialKind::Pbr(base_bsdf)));
        reg.lights = plain.lights.clone();
        let _h = reg.add_texture_rgba8(w, w, true, &{
            let mut b = Vec::with_capacity((w * w * 4) as usize);
            for c in &px {
                let v = (c[0] * 255.0 + 0.5) as u8;
                b.extend_from_slice(&[v, v, v, 255]);
            }
            b
        });
        assert_eq!(reg.textures.len(), 1);
        assert!(!reg.texels.is_empty(), "纹素必须真的铺进 Scene::texels");
        let b = render_frame(&reg, &cam, &ctx, res, &params);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            for ch in 0..4 {
                assert_eq!(
                    x[ch].to_bits(),
                    y[ch].to_bits(),
                    "只注册纹理改变了像素 {i} 通道 {ch}"
                );
            }
        }

        // (2) 材质引用 base map：结果必须变化，且两次渲染逐字节相同。
        let mut mapped = reg.clone();
        let h = 1u32;
        if let MaterialKind::Pbr(ref mut bsdf) = mapped.spheres[0].1 {
            bsdf.maps[crate::bsdf::MAP_BASE] = h;
        }
        let c1 = render_frame(&mapped, &cam, &ctx, res, &params);
        let c2 = render_frame(&mapped, &cam, &ctx, res, &params);
        let mut differing = 0usize;
        for (i, (x, y)) in c1.iter().zip(c2.iter()).enumerate() {
            for ch in 0..4 {
                assert_eq!(x[ch].to_bits(), y[ch].to_bits(), "两次渲染不同：像素 {i} 通道 {ch}");
            }
        }
        for (x, y) in a.iter().zip(c1.iter()) {
            if (0..4).any(|ch| x[ch].to_bits() != y[ch].to_bits()) {
                differing += 1;
            }
        }
        assert!(
            differing > 0,
            "引用 base map 后画面完全没变：贴图没有接进积分器"
        );
        let lit = c1.iter().filter(|p| p[0] > 0.0).count();
        println!(
            "有贴图渲染：{} 像素，{} 与无贴图基准不同，{} 非黑；两次渲染逐字节相同",
            c1.len(),
            differing,
            lit
        );
        // 数值健全性：全有限。
        for p in &c1 {
            for ch in 0..4 {
                assert!(p[ch].is_finite(), "渲染出现非有限值：{p:?}");
            }
        }
    }
}
