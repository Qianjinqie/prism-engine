//! SDF-based GI：sphere tracing 遮挡查询 + 环境光遮蔽 + 一步间接光。
//!
//! # 架构红利（D-21）：为什么用 SDF 而非 BVH
//! Lumen 的遮挡查询走 BVH/表面缓存 + 屏幕空间探针层次。Prism 的 SDF 生态让
//! 遮挡查询变成**沿射线按 SDF 值步进**（sphere tracing）：单步 `dist(p)` 即知
//! "距最近表面多远"，无需遍历树、无栈、无 SAH 节点访问。对**光滑程序化几何**
//! 与"粗略遮挡"（GI 只要"挡住没有"，不要精确交点）而言，这正是 D-21 说的
//! "好实现"路径：
//!
//! - **无数据结构**：只要求一个 `f(p) -> 距离`（[`SdfField`] trait），
//!   由 [`crate::gi::SceneSdf`] 从既有的 [`crate::integrator::Scene`] 组合而成；
//! - **步数即预算**：`max_steps` 直接是质量-成本旋钮（GI 遮挡可早停）；
//! - **Lipschitz=1 保证不漏**：单步 `d·safety`（safety≤1）保证不越过表面。
//!
//! # 本模块不做重复造轮子
//! sphere tracing 本体在 `prism-geometry::sdf::sphere_trace`，此处**复用**它
//! （D-16 去繁、单实现）。`SceneSdf` 则把 RT 场景的球/盒转成 `Csg` 树（S12
//! 实体表示）——球→`Csg::Sphere`，盒→`Csg::Box`，多体→`Union`。
//!
//! # 与路径追踪的关系
//! 路径追踪（[`crate::integrator`]）的 GI 来自弹射，质量高但慢；本模块是**实时
//! 档位**：用固定步数 SDF 遮挡 + 探针/屏幕空间缓存，牺牲精度换速度，两者互补。
//! 本模块**不改 integrator**，是并行的第二条 GI 路径（CpuRef 可跑）。

use prism_geometry::csg::Csg;
use prism_geometry::primitive::{Box3, Sdf, Sphere as SdfSphere};
use prism_geometry::sdf::{sphere_trace, TraceParams};
use prism_math::Vec3;

use crate::integrator::Scene;
use crate::gi::sh::hadamard;
use crate::sampler::{cosine_hemisphere, onb};

/// 距离场抽象：点 → 到最近表面的有符号距离（内负外正）。
///
/// 与 `prism-geometry::primitive::Sdf` 同构，但此处**重复声明为薄 trait**，
/// 以便 GI 模块可注入任意参考场（测试用解析场、生产用 [`SceneSdf`]），
/// 且不强制调用方持有 `prism-geometry` 类型。任何实现了该 trait 的类型都可
/// 直接喂给 [`sphere_trace`]（本模块与之通过 blanket impl 桥接）。
pub trait SdfField {
    /// 点 `p` 的有符号距离。
    fn dist(&self, p: Vec3) -> f32;
}

/// 把任意 [`SdfField`] 桥到 `prism-geometry` 的 `Sdf`（供 [`sphere_trace`] 使用）。
struct FieldAdapter<'a>(&'a dyn SdfField);

impl Sdf for FieldAdapter<'_> {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        self.0.dist(p)
    }
}

/// GI 遮挡预算：步数、t 上限、命中阈值、安全系数（质量-成本旋钮）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcclusionBudget {
    /// sphere tracing 最大步数（**核心成本旋钮**：步数 ≈ 查询成本）。
    pub max_steps: u32,
    /// 射线参数上限（世界单位；超过即未命中）。
    pub t_max: f32,
    /// 命中阈值：`dist ≤ hit_eps` 视为触及遮挡体。
    pub hit_eps: f32,
    /// 步长安全系数（≤1；1 为理论极限，取 0.9 稳健）。
    pub safety: f32,
}

impl Default for OcclusionBudget {
    fn default() -> Self {
        Self { max_steps: 48, t_max: 20.0, hit_eps: 1e-3, safety: 0.9 }
    }
}

impl OcclusionBudget {
    /// 低档（快速预览）：步数少、t 短。
    pub fn low() -> Self {
        Self { max_steps: 16, t_max: 8.0, hit_eps: 5e-3, safety: 0.9 }
    }
    /// 标准档（默认）。
    pub fn standard() -> Self {
        Self::default()
    }
    /// 高档（收敛）：步数多、t 长。
    pub fn high() -> Self {
        Self { max_steps: 128, t_max: 50.0, hit_eps: 1e-4, safety: 0.9 }
    }

    /// 参数防呆（C18）：非法值收束到合法域。
    fn sanitized(self) -> Self {
        let max_steps = if self.max_steps == 0 { 1 } else { self.max_steps.min(4096) };
        let t_max = if self.t_max.is_finite() && self.t_max > 0.0 { self.t_max } else { 20.0 };
        let hit_eps =
            if self.hit_eps.is_finite() && self.hit_eps > 0.0 { self.hit_eps } else { 1e-3 };
        let safety = if self.safety.is_finite() && self.safety > 0.0 {
            self.safety.min(1.0)
        } else {
            0.9
        };
        Self { max_steps, t_max, hit_eps, safety }
    }
}

/// 从 RT [`Scene`] 构造的距离场（S12 实体表示：球/盒 → `Csg` 树）。
///
/// 场景为空时返回"空前向无遮挡"的场（恒为大正距离）。
#[derive(Debug, Clone)]
pub struct SceneSdf {
    /// 组合后的 CSG 树（None = 空场）。
    root: Option<Csg>,
    /// 场景包围半径（供空场/退化场返回大距离）。
    bound_radius: f32,
}

/// 空场的"无限远"距离（远大于任何真实场景尺度）。
const EMPTY_FAR: f32 = 1.0e6;

impl SceneSdf {
    /// 从 RT 场景构建 SDF（球→Sphere，盒→Box，多体→Union）。确定性：按
    /// 场景索引固定序组合（tie-break 为索引序）。
    pub fn from_scene(scene: &Scene) -> Self {
        let mut root: Option<Csg> = None;
        let mut bound_radius = 1.0f32;
        let union_with = |node: Csg, root: &mut Option<Csg>| {
            *root = Some(match root.take() {
                None => node,
                Some(prev) => Csg::Union(Box::new(prev), Box::new(node)),
            });
        };
        for (s, _m) in &scene.spheres {
            let r = if s.radius.is_finite() && s.radius > 0.0 { s.radius } else { 1e-4 };
            union_with(
                Csg::Sphere(SdfSphere { center: s.center, radius: r }),
                &mut root,
            );
            bound_radius = bound_radius.max(s.center.length() + r);
        }
        for (b, _m) in &scene.boxes {
            let center = (b.min + b.max) * 0.5;
            let mut half = (b.max - b.min) * 0.5;
            // 退化盒（零/负尺寸）→ 给最小半尺寸，避免除零/负距离异常。
            half = Vec3::new(
                half.x.max(1e-4),
                half.y.max(1e-4),
                half.z.max(1e-4),
            );
            union_with(Csg::Box(Box3 { center, half }), &mut root);
            bound_radius = bound_radius.max(center.length() + half.length());
        }
        SceneSdf { root, bound_radius }
    }

    /// 场景包围半径（诊断用）。
    pub fn bound_radius(&self) -> f32 {
        self.bound_radius
    }

    /// 是否为空场（无几何）。
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
}

impl SdfField for SceneSdf {
    #[inline]
    fn dist(&self, p: Vec3) -> f32 {
        match &self.root {
            None => EMPTY_FAR,
            Some(csg) => csg.dist(p),
        }
    }
}

/// 单调球面遮挡查询：沿 `rd` 从 `origin` 投射，被遮挡返回 true。
///
/// 这是 GI 的**唯一原语**（AO / 间接光 / 探针遮挡都调它）。空场恒返回 false。
/// 起点先沿法线外推一个小量避免自遮挡（调用方传已外推的点即可，本函数不自作主张）。
pub fn is_occluded(
    field: &dyn SdfField,
    origin: Vec3,
    rd: Vec3,
    budget: OcclusionBudget,
) -> bool {
    let b = budget.sanitized();
    // 方向合法性：非有限分量 → 视为无遮挡（C18 不 panic；也避免破坏
    // prism-geometry sphere_trace 的单位向量 debug_assert）。
    if !rd.x.is_finite() || !rd.y.is_finite() || !rd.z.is_finite() {
        return false;
    }
    let rd = rd.normalize();
    if rd.length() < 0.5 {
        return false; // 退化方向（零向量）→ 视为无遮挡，不 panic
    }
    let params = TraceParams {
        t_max: b.t_max,
        hit_eps: b.hit_eps,
        max_steps: b.max_steps,
        safety: b.safety,
    };
    sphere_trace(&FieldAdapter(field), origin, rd, &params).is_some()
}

/// 环境光遮蔽（AO）：在法线半球内做 `n_dirs` 条确定性遮挡射线，返回 `\[0,1\]`
/// 可见度（1 = 全可见，0 = 全遮蔽）。
///
/// 方向用固定低差异序列（球面 Fibonacci，无随机）→ 同输入逐位一致。
///
/// # 确定性
/// 方向生成只用常量乘法与 sqrt/atan2 的确定性替身（此处只需方向序，用
/// 黄金角螺旋即可，无超越函数依赖测试稳定）。累加按方向序固定。
pub fn ambient_occlusion(field: &dyn SdfField, p: Vec3, n: Vec3, n_dirs: u32, budget: OcclusionBudget) -> f32 {
    let n_dirs = n_dirs.max(1);
    let n = n.normalize();
    if n.length() < 0.5 {
        return 1.0; // 退化法线 → 全可见（C18 不 panic）
    }
    let (t, b) = onb(n);
    let mut visible = 0.0f32;
    for i in 0..n_dirs {
        let (xi, yi) = fib_hemi_dir(i, n_dirs);
        let dir = (t * xi + b * yi + n * (1.0 - xi * xi - yi * yi).max(0.0).sqrt()).normalize();
        // 起点沿法线外推（防自遮挡）。
        let o = p + n * 1e-3;
        if !is_occluded(field, o, dir, budget) {
            visible += 1.0;
        }
    }
    (visible / n_dirs as f32).clamp(0.0, 1.0)
}

/// 半球内确定性低差异方向（Fibonacci 螺旋投影到单位圆盘）。
#[inline]
fn fib_hemi_dir(i: u32, n: u32) -> (f32, f32) {
    let idx = i as f32 + 0.5;
    let r = (idx / n as f32).sqrt(); // 面积均匀圆盘半径
    let phi = idx * 2.399_963_2; // 黄金角（常量）
    (r * phi.cos(), r * phi.sin())
}

/// 一步间接光：在 `p` 的半球内取 `n_samples` 条余弦加权方向，未被遮挡者
/// 累加 `albedo · env_radiance`（用 SDF 遮挡 + 环境常量近似命中点辐亮度）。
///
/// # 返回语义（诚实说明）
/// 返回的是 **可见性加权的环境辐亮度均值 × albedo**（`V·L_env·albedo` 的
/// 方向平均），**不是**严格意义上的辐照度 `E = ∫L·cosθ dω`。余弦分布 MC 的
/// 无偏估计量应为 `L·π`，本实现刻意**省略 π 因子**（与 Lambert `albedo/π`
/// 的 π 在两处约定下互相抵消，见 [`crate::gi::sh`] 模块文档的 π 约定）。
/// 若需严格辐照度，请乘 π，或改用 [`crate::gi::sh`] 的 SH 投影路径。
///
/// 这是"实时档位"的间接光估计：只做**一步**（不递归弹射），命中点按环境
/// 常量近似其辐亮度（Lumen 的表面缓存角色在此被"环境常量 + SDF 遮挡"替代）。
///
/// # 确定性
/// 方向用固定余弦半球序列（[`cosine_hemisphere`]，低差异），累加序固定。
pub fn one_bounce_indirect(
    field: &dyn SdfField,
    p: Vec3,
    n: Vec3,
    albedo: Vec3,
    env_radiance: Vec3,
    n_samples: u32,
    budget: OcclusionBudget,
) -> Vec3 {
    let n_samples = n_samples.max(1);
    let nn = n.normalize();
    if nn.length() < 0.5 {
        return Vec3::splat(0.0);
    }
    let mut e = Vec3::splat(0.0);
    for i in 0..n_samples {
        // 低差异 2D（黄金角螺旋的 [0,1)² 映射）。
        let u = (i as f32 + 0.5) / n_samples as f32;
        let v = (i as f32 + 0.5) * 0.618_034; // 黄金比小数部分
        let v = v - v.floor();
        let dir = cosine_hemisphere(u, v);
        // 把余弦半球方向（以 +Z 为极）旋到 n。
        let (t, b) = onb(nn);
        let world = (t * dir.x + b * dir.y + nn * dir.z).normalize();
        let o = p + nn * 1e-3;
        if !is_occluded(field, o, world, budget) {
            // 未遮挡：命中点辐亮度近似 = 环境常量 × 命中反照率（单步）。
            let c = hadamard(albedo, env_radiance);
            e = e + c;
        }
    }
    // 余弦加权蒙特卡洛：E ≈ (1/N)·Σ L·albedo（pdf=cos/π 已被估量吸收）。
    e * (1.0 / n_samples as f32)
}

// ---------------------------------------------------------------------------
// 多步间接光（深化 A：迭代式多弹射）
// ---------------------------------------------------------------------------

/// 多弹射求解预算（质量-成本旋钮）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiBounceBudget {
    /// 每个弹射点的间接方向采样数（与 [`one_bounce_indirect`] 同义）。
    pub n_samples: u32,
    /// 最大弹射次数（≥1；第 0 弹 = 直接光，第 1..k 弹 = 间接）。
    pub max_bounces: u32,
    /// 收敛阈值：相邻迭代辐亮度变化的 L∞ 范数（分量绝对值最大者）小于此值即停。
    pub tolerance: f32,
    /// 每次间接弹射的衰减系数（能量守恒因子，`[0,1]`）。
    ///
    /// 物理含义：Lambert 面每弹射后辐亮度被反照率 `≤1` 与 1/π 余弦归一共同衰减。
    /// 本实现把"局部反照率"抽象为**全局衰减**（实时档位简化：探针处无材质）。
    /// 取 `≤1` 保证几何级数收敛（能量单调有界）。
    pub decay: f32,
    /// SDF 遮挡预算。
    pub occlusion: OcclusionBudget,
}

impl Default for MultiBounceBudget {
    fn default() -> Self {
        Self {
            n_samples: 64,
            max_bounces: 8,
            tolerance: 1e-3,
            decay: 0.8,
            occlusion: OcclusionBudget::standard(),
        }
    }
}

impl MultiBounceBudget {
    /// 低档（快速预览）：少采样、少弹射。
    pub fn low() -> Self {
        Self {
            n_samples: 16,
            max_bounces: 3,
            tolerance: 5e-3,
            decay: 0.8,
            occlusion: OcclusionBudget::low(),
        }
    }
    /// 标准档。
    pub fn standard() -> Self {
        Self::default()
    }
    /// 高档（收敛）：多采样、多弹射、严阈值。
    pub fn high() -> Self {
        Self {
            n_samples: 256,
            max_bounces: 16,
            tolerance: 1e-4,
            decay: 0.9,
            occlusion: OcclusionBudget::high(),
        }
    }

    /// 参数防呆（C18）：非法值收束到合法域。
    fn sanitized(self) -> Self {
        let n_samples = if self.n_samples == 0 { 1 } else { self.n_samples.min(65536) };
        let max_bounces = if self.max_bounces == 0 { 1 } else { self.max_bounces.min(256) };
        let tolerance = if self.tolerance.is_finite() && self.tolerance > 0.0 {
            self.tolerance
        } else {
            1e-3
        };
        let decay = if self.decay.is_finite() { self.decay.clamp(0.0, 1.0) } else { 0.8 };
        Self { n_samples, max_bounces, tolerance, decay, occlusion: self.occlusion.sanitized() }
    }
}

/// 多弹射求解结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiBounceResult {
    /// 累积到收敛（或最大弹射）的**总辐亮度**：`L = Σ_k L_k`（L_k = 第 k 弹贡献）。
    pub total: Vec3,
    /// 实际执行的弹射轮数（含第 0 弹直接光；≥1）。
    pub bounces: u32,
    /// 收敛标志：true = 因变化量低于阈值提前收敛；false = 达最大弹射仍未达阈。
    pub converged: bool,
    /// 最后一次迭代的变化量（L∞ 范数），诊断用。
    pub last_delta: f32,
}

/// 迭代式多弹射间接光（深化 A）。
///
/// # 弹射语义
/// - **第 0 弹** = 直接光（环境常量 `env_radiance` 直接被"看见"一次：`V·env`）；
/// - **第 k 弹**（k ≥ 1）以第 k−1 弹的辐亮度作为**源辐亮度**再乘一次
///   `decay`，并按 SDF 可见性加权半球平均（复用 [`one_bounce_indirect`] 的
///   余弦半球确定性采样）。
///
/// 数学形式（每弹一次"可见性加权半球平均 × albedo × decay"算子 T）：
///
/// ```text
/// L_0 = albedo · env · V̄                        （直接光，V̄ = 半球可见比）
/// L_k = decay · bounces 的上一弹作为源 × T(·)     （T = 可见性加权余弦平均）
/// L   = Σ_{k=0}^{K} L_k                          （几何级数求和）
/// ```
///
/// # 收敛判据（公式 + 阈值依据）
/// 取相邻两次**累积和** `A_k = Σ_{j≤k} L_j` 的 **L∞ 范数差**
/// `Δ_k = max_c |A_k,c − A_{k−1},c|`（分量绝对值最大者）。当 `Δ_k < tolerance`
/// 即停。阈值默认 `1e-3`：因为 albedo/辐亮度量级为 O(1)，`1e-3` 相当于
/// **相对收敛 0.1%**，肉眼与逐位复现都不敏感；低档放宽到 `5e-3`（省几次迭代），
/// 高档收紧到 `1e-4`。
///
/// 由于 `decay ≤ 1`（且 albedo ≤1 隐含下调），该级数是**收敛几何级数**，
/// 因此 `A_k` 单调不减且有上界，`Δ_k` 单调递减——这为物理断言（单调 + 差值递减）
/// 提供保证。
///
/// # 确定性
/// 采样方向序固定（[`one_bounce_indirect`] 内为低差异黄金角序），每弹累加序固定，
/// 无 RNG、无并发。同输入逐位一致。
pub fn solve_multi_bounce(
    field: &dyn SdfField,
    p: Vec3,
    n: Vec3,
    albedo: Vec3,
    env_radiance: Vec3,
    budget: MultiBounceBudget,
) -> MultiBounceResult {
    let b = budget.sanitized();
    let nn = n.normalize();
    if nn.length() < 0.5 {
        // 退化法线 → 无间接光（C18 不 panic）。
        return MultiBounceResult {
            total: Vec3::splat(0.0),
            bounces: 0,
            converged: true,
            last_delta: 0.0,
        };
    }
    // 第 0 弹：直接光（复用一步间接光原语，语义即"可见性加权环境辐亮度 × albedo"）。
    let mut accum = one_bounce_indirect(
        field,
        p,
        nn,
        albedo,
        env_radiance,
        b.n_samples,
        b.occlusion,
    );
    let mut bounces = 1u32;
    let mut converged = false;
    let mut last_delta = 0.0f32;

    // 后续弹射：以上一弹的**增量**作为源辐亮度，衰减 + 可见性加权半球平均。
    // 源 = 本次新增的辐亮度（末弹增量），保证几何级数求和语义。
    let mut source = accum;
    let mut k = 1u32;
    while k < b.max_bounces {
        // 本弹贡献 = decay · T(source)，其中 T 复用一步间接光语义，但把
        // "环境常量"替换为"上一弹作为方向无关源辐亮度"（实时档位近似）。
        let bounce = one_bounce_indirect(
            field,
            p,
            nn,
            albedo,
            source,
            b.n_samples,
            b.occlusion,
        ) * b.decay;
        let new_accum = accum + bounce;
        // 收敛判据：累积和的 L∞ 变化量。
        let diff = new_accum - accum;
        let delta = diff.x.abs().max(diff.y.abs()).max(diff.z.abs());
        accum = new_accum;
        source = bounce;
        last_delta = delta;
        bounces = k + 1;
        if delta < b.tolerance {
            converged = true;
            break;
        }
        // 防止源辐亮度退化为 0（全遮挡）后无意义迭代：若 bounce 已恒 0，提前收敛。
        if bounce.x == 0.0 && bounce.y == 0.0 && bounce.z == 0.0 {
            converged = true;
            break;
        }
        k += 1;
    }
    MultiBounceResult { total: accum, bounces, converged, last_delta }
}

/// 多弹射的**总辐射能量**辅助函数（便于测试与诊断）。
///
/// 定义：`energy = (R + G + B)/3 · 4π`（RGB 均值 × 全立体角），用于"全漫反白体
/// 理论上界"对照。这里的 `4π` 是几何因子约定（把"方向平均辐亮度"还原为
/// "全空间积分能量"），与 [`crate::gi::sh`] 的 π 约定相容（不重复计 π）。
///
/// # 上界（物理断言依据）
/// 每次弹射的辐亮度 ≤ `albedo·env·decay^k`（可见比 ≤1），故
/// `L_total ≤ albedo·env·Σ decay^k = albedo·env/(1−decay)`（decay<1）。
/// [`MultiBounceBudget::decay`] 默认 0.8 时上界 = 5·albedo·env。
#[inline]
pub fn total_radiant_energy(l: Vec3) -> f32 {
    if !l.x.is_finite() || !l.y.is_finite() || !l.z.is_finite() {
        return 0.0;
    }
    ((l.x + l.y + l.z) / 3.0) * (4.0 * std::f32::consts::PI)
}

/// 多弹射的**全漫反白体理论能量上界**（closed-form 几何级数）。
///
/// `upper = (albedo·env 的 RGB 均值) · Σ_{k=0}^{max_bounces−1} decay^k · 4π`。
/// 这是"可见比恒为 1、方向完全各向同性收敛"的理想上界，任何实际几何（遮挡
/// 使可见比 <1）的总能量都不应超过它。用于物理断言"能量不超上界"。
pub fn multi_bounce_energy_upper_bound(
    albedo: Vec3,
    env_radiance: Vec3,
    budget: MultiBounceBudget,
) -> f32 {
    let b = budget.sanitized();
    // 几何级数 Σ decay^k（k=0..max_bounces-1）；decay=1 时退化为 max_bounces。
    let decay = b.decay;
    let sum = if (1.0 - decay).abs() < 1e-6 {
        b.max_bounces as f32
    } else {
        (1.0 - decay.powi(b.max_bounces as i32)) / (1.0 - decay)
    };
    // 每弹贡献上界 = albedo·env（可见比 ≤1）。取 RGB 均值 × 4π。
    let per = (albedo.x * env_radiance.x + albedo.y * env_radiance.y + albedo.z * env_radiance.z) / 3.0;
    per * sum * (4.0 * std::f32::consts::PI)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Aabb, MaterialKind, Sphere};

    /// 解析场：单个闭合球（用于对照解析解）。
    struct SphereField {
        c: Vec3,
        r: f32,
    }
    impl SdfField for SphereField {
        fn dist(&self, p: Vec3) -> f32 {
            (p - self.c).length() - self.r
        }
    }

    /// 解析场：单个 AABB（用于遮挡对照）。
    struct BoxField {
        min: Vec3,
        max: Vec3,
    }
    impl SdfField for BoxField {
        fn dist(&self, p: Vec3) -> f32 {
            let center = (self.min + self.max) * 0.5;
            let half = (self.max - self.min) * 0.5;
            let q = Vec3::new(
                (p.x - center.x).abs() - half.x,
                (p.y - center.y).abs() - half.y,
                (p.z - center.z).abs() - half.z,
            );
            let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
            outside.length() + q.x.max(q.y).max(q.z).min(0.0)
        }
    }

    fn budget() -> OcclusionBudget {
        OcclusionBudget::standard()
    }

    /// 遮挡正确性：两 AABB 之间的可见性与解析一致。
    #[test]
    fn occlusion_matches_analytic_boxes() {
        // 一个盒子在 (0,0,1) 处，尺寸 1³（min=(−0.5,−0.5,0.5), max=(0.5,0.5,1.5)）。
        let field = BoxField {
            min: Vec3::new(-0.5, -0.5, 0.5),
            max: Vec3::new(0.5, 0.5, 1.5),
        };
        // 从 (0,0,0) 沿 +Z 打 → 命中盒子（遮挡）。
        assert!(
            is_occluded(&field, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), budget()),
            "正对盒子的射线应被遮挡"
        );
        // 从 (0,0,0) 沿 −Z 打 → 背向，不命中。
        assert!(
            !is_occluded(&field, Vec3::splat(0.0), Vec3::new(0.0, 0.0, -1.0), budget()),
            "背向盒子的射线不应被遮挡"
        );
        // 从 (3,3,1) 沿 −Y 打：x=3 远在盒横向范围 [−0.5,0.5] 之外 → 不命中。
        assert!(
            !is_occluded(&field, Vec3::new(3.0, 3.0, 1.0), Vec3::new(0.0, -1.0, 0.0), budget()),
            "偏离盒子横向的射线不应被遮挡"
        );
        // 正上方（x=0,z=1 在盒范围内）沿 −Y → 命中盒顶（对照，证明横向确实关键）。
        assert!(
            is_occluded(&field, Vec3::new(0.0, 3.0, 1.0), Vec3::new(0.0, -1.0, 0.0), budget()),
            "正上方射线应命中盒顶"
        );
    }

    /// 遮挡正确性：球体 vs 解析可见性。
    #[test]
    fn occlusion_matches_analytic_sphere() {
        let field = SphereField { c: Vec3::new(0.0, 0.0, 5.0), r: 1.0 };
        let origin = Vec3::splat(0.0);
        // 正对球心 → 遮挡。
        assert!(is_occluded(&field, origin, Vec3::new(0.0, 0.0, 1.0), budget()));
        // 偏离球（横向 2.0 > 半径 1.0）→ 不遮挡。
        assert!(!is_occluded(
            &field,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            budget()
        ));
        // 相切边缘附近：横向 1.5（球在 z=5，半径 1 → 横向在 [4,6] 才命中；
        // 从 x=1.5 沿 +Z 打，最近点 x 距离 1.5 > 1）→ 不遮挡。
        assert!(!is_occluded(
            &field,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            budget()
        ));
    }

    /// 确定性：同输入两遍遮挡结果一致（逐样本 AO 也逐位一致）。
    #[test]
    fn occlusion_and_ao_bitwise_deterministic() {
        let field = BoxField {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let p = Vec3::new(0.0, 0.0, 3.0);
        let n = Vec3::new(0.0, 0.0, -1.0);
        let a = ambient_occlusion(&field, p, n, 32, budget());
        let b = ambient_occlusion(&field, p, n, 32, budget());
        assert_eq!(a.to_bits(), b.to_bits(), "AO 必须逐位确定");
    }

    /// AO 正确性：无穷大平面上的点，正上方开口 → AO ≈ 1；被大盒包裹 → AO < 1。
    #[test]
    fn ao_open_vs_enclosed() {
        // 场景：一个大盒在点上方（-Y 朝向点）。
        let field = BoxField {
            min: Vec3::new(-10.0, 0.5, -10.0),
            max: Vec3::new(10.0, 5.0, 10.0),
        };
        // 点 P=(0,0,0)，法线 +Y（朝盒）→ 半球全被盒遮挡 → AO 接近 0。
        let ao_top = ambient_occlusion(&field, Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), 64, budget());
        assert!(ao_top < 0.2, "上方有大盒 AO 应很低，实际 {ao_top}");
        // 同点法线 −Y（背向盒）→ 半球全开 → AO = 1。
        let ao_bottom =
            ambient_occlusion(&field, Vec3::splat(0.0), Vec3::new(0.0, -1.0, 0.0), 64, budget());
        assert!(ao_bottom > 0.95, "背向盒 AO 应接近 1，实际 {ao_bottom}");
    }

    /// 空场：AO 恒 1、遮挡恒 false；间接光 = 全方向环境（物理正确的"天光"项，
    /// 非 0）。空场无表面 → 每个方向都未遮挡 → E ≈ albedo·env。
    #[test]
    fn empty_scene_is_fully_visible() {
        let field = SceneSdf::from_scene(&Scene::default());
        assert!(field.is_empty());
        assert!(!is_occluded(&field, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), budget()));
        let ao = ambient_occlusion(&field, Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), 16, budget());
        assert!((ao - 1.0).abs() < 1e-6, "空场 AO 应为 1");
        let e = one_bounce_indirect(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::splat(0.8),
            Vec3::splat(1.0),
            16,
            budget(),
        );
        // 全方向未遮挡 → E = albedo·env = 0.8（环境天光项，确定性单步估计）。
        assert!((e.x - 0.8).abs() < 1e-4, "空场间接光应为 albedo·env=0.8，实际 {}", e.x);
    }

    /// SceneSdf 从 RT 场景构建：球/盒组合，遮挡行为正确。
    #[test]
    fn scene_sdf_from_rt_scene() {
        let scene = Scene {
            spheres: vec![(
                Sphere { center: Vec3::new(0.0, 0.0, 2.0), radius: 0.5 },
                MaterialKind::Lambert { albedo: Vec3::splat(0.8) },
            )],
            boxes: vec![(
                Aabb::new(Vec3::new(-2.0, -0.1, -2.0), Vec3::new(2.0, 0.0, 2.0)),
                MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
            )],
            lights: Vec::new(),
            medium: None,
            paraboloids: vec![],
            meshes: Vec::new(),
            planes: vec![],
            ..Default::default()
        };
        let field = SceneSdf::from_scene(&scene);
        assert!(!field.is_empty());
        // 正对球 → 遮挡。
        assert!(is_occluded(&field, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), budget()));
        // 向下打地板 → 遮挡。
        assert!(is_occluded(
            &field,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            budget()
        ));
        // 向上打 → 无遮挡。
        assert!(!is_occluded(
            &field,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            budget()
        ));
    }

    /// 间接光：有遮挡体的点比开阔点间接光低（遮挡起作用）。
    #[test]
    fn indirect_light_reduced_by_occlusion() {
        let env = Vec3::splat(1.0);
        let albedo = Vec3::splat(0.8);
        // 开阔场（无几何）：全方向天光 → E = albedo·env = 0.8。
        let open = SceneSdf::from_scene(&Scene::default());
        let e_open = one_bounce_indirect(
            &open,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            albedo,
            env,
            64,
            budget(),
        );
        assert!((e_open.x - albedo.x * env.x).abs() < 1e-4, "开阔场应为 albedo·env");
        // 上方有大遮挡体 → 部分方向被挡 → 间接光应低于开阔场。
        let field = BoxField {
            min: Vec3::new(-5.0, 0.3, -5.0),
            max: Vec3::new(5.0, 3.0, 5.0),
        };
        let e_occl = one_bounce_indirect(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            albedo,
            env,
            64,
            budget(),
        );
        assert!(
            e_occl.x < e_open.x,
            "被遮挡点间接光应低于开阔场：{} vs {}",
            e_occl.x,
            e_open.x
        );
        assert!(e_occl.x >= 0.0 && e_occl.x <= albedo.x * env.x + 1e-4, "间接光越界 {}", e_occl.x);
        // 确定性。
        let e2 = one_bounce_indirect(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            albedo,
            env,
            64,
            budget(),
        );
        assert_eq!(e_occl.x.to_bits(), e2.x.to_bits());
    }

    /// 预算防呆：0 步、NaN t_max、负 hit_eps → 收束合法，不 panic。
    #[test]
    fn budget_sanitization_no_panic() {
        let field = SphereField { c: Vec3::new(0.0, 0.0, 2.0), r: 0.5 };
        let bad = OcclusionBudget {
            max_steps: 0,
            t_max: f32::NAN,
            hit_eps: -1.0,
            safety: 2.0,
        };
        let s = bad.sanitized();
        assert!(s.max_steps >= 1 && s.t_max > 0.0 && s.hit_eps > 0.0 && s.safety <= 1.0);
        // 用该预算查询不 panic。
        let _ = is_occluded(&field, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), bad);
        // NaN 方向 → 不 panic。
        let _ = is_occluded(
            &field,
            Vec3::splat(0.0),
            Vec3::new(f32::NAN, 0.0, 1.0),
            budget(),
        );
    }

    /// 测试自证非恒真：把遮挡阈值故意设大到"什么都算命中"，结果显示过于保守
    /// 的反例——证明遮挡查询确实依赖几何而非恒返回某值。
    #[test]
    fn occlusion_query_not_vacuous() {
        let field = SphereField { c: Vec3::new(0.0, 0.0, 100.0), r: 0.5 };
        // 球远在 100 外，t_max=20 → 不命中。
        assert!(!is_occluded(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 0.0, 1.0),
            OcclusionBudget { max_steps: 64, t_max: 20.0, hit_eps: 1e-4, safety: 0.9 }
        ));
        // 同方向，t_max=200 → 命中（证明 t_max 真在起作用）。
        assert!(is_occluded(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 0.0, 1.0),
            OcclusionBudget { max_steps: 256, t_max: 200.0, hit_eps: 1e-4, safety: 0.9 }
        ));
    }

    // -----------------------------------------------------------------------
    // 多弹射（深化 A）测试
    // -----------------------------------------------------------------------

    /// 多弹射预算防呆：0 采样/0 弹射/NaN 阈值/越界 decay → 收束合法，不 panic。
    #[test]
    fn multi_bounce_budget_sanitization() {
        let bad = MultiBounceBudget {
            n_samples: 0,
            max_bounces: 0,
            tolerance: f32::NAN,
            decay: 2.5,
            occlusion: OcclusionBudget::default(),
        };
        let s = bad.sanitized();
        assert!(s.n_samples >= 1 && s.max_bounces >= 1);
        assert!(s.tolerance > 0.0 && s.tolerance.is_finite());
        assert!((0.0..=1.0).contains(&s.decay));
        // 负 decay → 钳 0；用其求解不 panic（探测 0 弹射退化）。
        let neg = MultiBounceBudget { decay: -1.0, ..MultiBounceBudget::default() }.sanitized();
        assert_eq!(neg.decay, 0.0);
    }

    /// 确定性：同输入两遍多弹射结果逐位一致（total 逐位 + bounces/delta 一致）。
    #[test]
    fn multi_bounce_bitwise_deterministic() {
        let field = BoxField { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        let p = Vec3::new(0.0, 0.0, 3.0);
        let n = Vec3::new(0.0, 0.0, -1.0);
        let budget = MultiBounceBudget {
            n_samples: 32,
            max_bounces: 8,
            tolerance: 1e-4,
            decay: 0.8,
            occlusion: budget(),
        };
        let a = solve_multi_bounce(&field, p, n, Vec3::splat(0.7), Vec3::splat(1.0), budget);
        let b = solve_multi_bounce(&field, p, n, Vec3::splat(0.7), Vec3::splat(1.0), budget);
        assert_eq!(a.total.x.to_bits(), b.total.x.to_bits(), "多弹射 X 必须逐位确定");
        assert_eq!(a.total.y.to_bits(), b.total.y.to_bits());
        assert_eq!(a.total.z.to_bits(), b.total.z.to_bits());
        assert_eq!(a.bounces, b.bounces);
        assert_eq!(a.last_delta.to_bits(), b.last_delta.to_bits());
    }

    /// 物理：多弹射总辐亮度单调增加（每弹非负累加），且相邻步差值递减（几何收敛）。
    /// 用开阔场（可见比=1）保证纯几何级数，断言严格。
    #[test]
    fn multi_bounce_monotone_and_converging() {
        let field = SceneSdf::from_scene(&Scene::default()); // 空场：全可见
        let albedo = Vec3::splat(0.8);
        let env = Vec3::splat(1.0);
        // 逐弹累积：固定 max_bounces=12，逐次记录 A_k（用 bounces 递增的方式）。
        // 直接用一次求解的逐弹增量：通过不断加大 max_bounces 并固定 tolerance 极小，
        // 收集每次的 total（A_k）。因采样序固定，A_k 是同一序列的前缀和。
        let mut totals: Vec<f32> = Vec::new();
        for mb in 1..=10u32 {
            let budget = MultiBounceBudget {
                n_samples: 64,
                max_bounces: mb,
                tolerance: f32::MIN_POSITIVE, // 近似"不提前收敛"
                decay: 0.8,
                occlusion: budget(),
            };
            let r = solve_multi_bounce(
                &field,
                Vec3::splat(0.0),
                Vec3::new(0.0, 1.0, 0.0),
                albedo,
                env,
                budget,
            );
            totals.push(r.total.x);
        }
        // 单调不减。
        for w in totals.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "总辐亮度应单调不减：{} vs {}", w[1], w[0]);
        }
        // 相邻步差值（增量）递减：Δ_k = A_k − A_{k-1} 应满足 Δ_{k+1} ≤ Δ_k + eps。
        let deltas: Vec<f32> = totals.windows(2).map(|w| w[1] - w[0]).collect();
        for w in deltas.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-4,
                "相邻步差值应递减：增量 {} 后于 {}",
                w[1],
                w[0]
            );
        }
    }

    /// 物理：多弹射总辐亮度不超过"全漫反白体"理论上界（几何级数 closed-form）。
    #[test]
    fn multi_bounce_within_white_furnace_bound() {
        let field = SceneSdf::from_scene(&Scene::default()); // 空场（最亮，可见比=1）
        let albedo = Vec3::splat(0.7);
        let env = Vec3::splat(1.0);
        let budget = MultiBounceBudget {
            n_samples: 64,
            max_bounces: 12,
            tolerance: 1e-5,
            decay: 0.8,
            occlusion: budget(),
        };
        let r = solve_multi_bounce(&field, Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), albedo, env, budget);
        let e = total_radiant_energy(r.total);
        let bound = multi_bounce_energy_upper_bound(albedo, env, budget);
        assert!(e <= bound + 1e-3, "多弹射能量 {} 超过上界 {}", e, bound);
        assert!(e > 0.0, "空场多弹射能量应为正");
    }

    /// 物理：多步结果 ≥ 一步结果（多弹射只会加强间接光，绝不减弱）。
    #[test]
    fn multi_bounce_ge_one_bounce() {
        let field = BoxField { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        let p = Vec3::new(0.0, 0.0, 2.5);
        let n = Vec3::new(0.0, 0.0, -1.0);
        let albedo = Vec3::splat(0.8);
        let env = Vec3::splat(1.0);
        let one = one_bounce_indirect(&field, p, n, albedo, env, 64, budget());
        let multi = solve_multi_bounce(
            &field,
            p,
            n,
            albedo,
            env,
            MultiBounceBudget { n_samples: 64, max_bounces: 8, ..MultiBounceBudget::default() },
        );
        assert!(
            multi.total.x >= one.x - 1e-6,
            "多步 {} 应 ≥ 一步 {}",
            multi.total.x,
            one.x
        );
        // 空场下多步严格 > 一步（多一次弹射带来正增量）。
        let open = SceneSdf::from_scene(&Scene::default());
        let one_open = one_bounce_indirect(&open, Vec3::splat(0.0), Vec3::new(0.0, 1.0, 0.0), albedo, env, 64, budget());
        let multi_open = solve_multi_bounce(
            &open,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            albedo,
            env,
            MultiBounceBudget { n_samples: 64, max_bounces: 8, ..MultiBounceBudget::default() },
        );
        assert!(multi_open.total.x > one_open.x, "空场多步应严格大于一步");
    }

    /// 收敛标志：低阈值高弹射 → converged=true；极严阈值 + 仅 2 弹 → false。
    #[test]
    fn multi_bounce_convergence_flag() {
        let field = SceneSdf::from_scene(&Scene::default());
        let easy = solve_multi_bounce(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::splat(0.8),
            Vec3::splat(1.0),
            MultiBounceBudget { n_samples: 32, max_bounces: 32, tolerance: 1e-2, decay: 0.5, occlusion: budget() },
        );
        assert!(easy.converged, "低 toleranc/低 decay 应在 max_bounces 内收敛");
        assert!(easy.bounces >= 1);
        // 严阈值 + 少弹射 → 未收敛（除非恰好极快收敛，用 decay=1 保不收敛）。
        let hard = solve_multi_bounce(
            &field,
            Vec3::splat(0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::splat(0.8),
            Vec3::splat(1.0),
            MultiBounceBudget { n_samples: 32, max_bounces: 2, tolerance: 1e-9, decay: 1.0, occlusion: budget() },
        );
        assert_eq!(hard.bounces, 2, "decay=1 严阈值应耗尽弹射预算");
        assert!(!hard.converged, "严阈值 + 少弹射应报未收敛");
    }
}
