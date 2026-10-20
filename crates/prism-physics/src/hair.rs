//! 毛发系统：发丝几何（Groom）+ 丝束物理（XPBD 约束族扩展）。
//!
//! 语义权威：的分层约定（"发丝=曲线几何（S9 已有）+ 丝束物理（S3
//! XPBD 约束族加一类约束）+ 散射（介质传输核复用， 同款路径）——同为
//! 特解组合，内核零改动"）；万物皆粒子+约束；XPBD 统一求解。
//!
//! # 设计边界（D-16 Mini Code，宁少勿假）
//!
//! 本模块只落**物理侧**：发丝曲线的表达与 Groom 插值、以及毛发特有的三类
//! XPBD 约束。散射另在 `prism-rt::hair_scatter`（要求的"介质传输核复用"
//! 路径）。内核零改动——[`crate::xpbd::World`]、[`crate::xpbd::Constraint`]
//! 语义不改，毛发约束族以**新增**实现接入（本模块不修改任何既有约束）。
//!
//! # 发丝几何（"复用 S9 曲线结构"的落地口径）
//!
//! S9 的曲线表达在 `prism-vector::path`（TTF 惯例 points/tags + 二次贝塞尔）。
//! 但那是**二维路径光栅**数据面，且 `prism-physics` 不依赖 `prism-vector`
//! （依赖面纪律：物理 crate 只依赖 core/math）。据此，发丝曲线在此表达为
//! **三维控制多边形 + Catmull–Rom 细分**——与 S9 同构的"控制点链 → 细分曲线"
//! 结构，在 3D 世界空间自足承载（与 [`crate::cloth`] 自建网格同一口径：布料
//! 亦不复用外部网格结构）。此选择在报告"发丝几何"节如实标注。
//!
//! - 引导发丝（guide hair）= 用户/工具给出的**控制点链**（锚点序列），
//!   经 Catmull–Rom 细分成模拟用的稠密折线（每段一个 XPBD 粒子）；
//! - 从属发丝（interpolated hair）= 从若干引导发丝按**固定权重**插值出的
//!   发丝（Groom 核心操作），插值顺序固定，不引入 HashMap。
//!
//! # 确定性
//!
//! - 控制点遍历、细分、插值全部按**升序下标**（无 HashMap 迭代序）；
//! - 只用 ±×÷√（含节点参数化的固定除法）；无浮点三角、无 libm；
//! - 同输入同输出逐位一致（测试把关）。
//!
//! # C18 永不 panic
//!
//! 所有公共入口先做参数域/结构校验并返回 [`PhysicsError`]（三段式）；
//! 细分段长、权重和被校验为有限正数；求解循环内的除零/开方有守卫。

use prism_math::Vec3;

use crate::xpbd::{Constraint, Particle, StepReport, World};
use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 参数域护栏（数值稳定）
// ---------------------------------------------------------------------------

/// 段长下界（低于此值的段在弯曲约束雅可比中退化，进入保护分支）。
pub const MIN_SEGMENT: f32 = 1e-6;
/// 发丝速度上限（世界单位/秒）：超过则拒绝该子步的速度写入（防御发散）。
pub const MAX_HAIR_SPEED: f32 = 1.0e4;

/// 判定 f32 是否有限（NaN/Inf 同拒）。
#[inline]
fn is_finite3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

// ---------------------------------------------------------------------------
// 发丝曲线（Groom）
// ---------------------------------------------------------------------------

/// 发丝控制曲线：一根引导发丝的 3D 控制点链（根 → 梢）。
///
/// 控制点数量 ≥ 2。[`HairCurve::subdivide`] 用 Catmull–Rom 样条把控制点链
/// 细分成模拟折线（每段一个质粒子）。
#[derive(Debug, Clone, PartialEq)]
pub struct HairCurve {
    /// 控制点（世界空间；`points[0]` = 发根，`points[last]` = 发梢）。
    points: Vec<Vec3>,
}

impl HairCurve {
    /// 构造引导发丝曲线（控制点 ≥ 2；拒绝 NaN/Inf 控制点）。
    pub fn new(points: Vec<Vec3>) -> Result<Self, PhysicsError> {
        if points.len() < 2 {
            return Err(PhysicsError {
                code: "P4001",
                location: "HairCurve::new".to_string(),
                problem: format!(
                    "控制点数 = {}：发丝曲线至少需要 2 个控制点（根与梢）",
                    points.len()
                ),
                fix: "传 ≥ 2 个控制点，首点为发根、末点为发梢".to_string(),
            });
        }
        if points.iter().any(|p| !is_finite3(*p)) {
            return Err(PhysicsError {
                code: "P4002",
                location: "HairCurve::new".to_string(),
                problem: "控制点含 NaN/Inf：发丝几何必须有限".to_string(),
                fix: "剔除非有限控制点".to_string(),
            });
        }
        Ok(Self { points })
    }

    /// 控制点数。
    #[inline]
    pub fn control_count(&self) -> usize {
        self.points.len()
    }

    /// 控制点切片（根 → 梢）。
    #[inline]
    pub fn control_points(&self) -> &[Vec3] {
        &self.points
    }

    /// 发丝曲线的**整条折线长度**（控制点折线和，用于长度守恒检验基准）。
    pub fn polyline_length(&self) -> f32 {
        let mut len = 0.0f32;
        for w in self.points.windows(2) {
            len += (w[1] - w[0]).length();
        }
        len
    }

    /// Catmull–Rom 细分：把控制点链细分为 `segments_per_span` 段/控制点区间
    /// 的折线（世界空间）。
    ///
    /// Catmull–Rom（均匀张力，标准公开形式）：对每个区间 [P1, P2]，控制点
    /// P0/P1/P2/P3（端点用镜像延拓），t ∈ [0,1]：
    /// `C(t) = ½(2P1 + (−P0+P2)t + (2P0−5P1+4P2−P3)t² + (−P0+3P1−3P2+P3)t³)`。
    ///
    /// 端点镜像：P0 = 2P1 − P2（P1 为根时）、P3 = 2P2 − P1（P2 为梢时）——
    /// 保证根/梢处切线自然，且不引入额外状态。返回点序列长度 =
    /// `(n−1)·segments_per_span + 1`（含首尾控制点）。
    ///
    /// `segments_per_span` 必须 ≥ 1（0 会产出单点退化）。
    pub fn subdivide(&self, segments_per_span: usize) -> Result<Vec<Vec3>, PhysicsError> {
        if segments_per_span == 0 {
            return Err(PhysicsError {
                code: "P4003",
                location: "HairCurve::subdivide".to_string(),
                problem: "segments_per_span = 0：每区间至少细分为 1 段".to_string(),
                fix: "传 segments_per_span ≥ 1（发丝常用 4–8）".to_string(),
            });
        }
        let p = &self.points;
        let n = p.len();
        let inv_seg = 1.0 / segments_per_span as f32;
        let mut out: Vec<Vec3> = Vec::with_capacity((n - 1) * segments_per_span + 1);
        out.push(p[0]);
        for i in 0..n - 1 {
            let p1 = p[i];
            let p2 = p[i + 1];
            // 端点镜像延拓（根 / 梢处无相邻控制点）。
            let p0 = if i == 0 {
                p1 * 2.0 - p2
            } else {
                p[i - 1]
            };
            let p3 = if i + 2 < n {
                p[i + 2]
            } else {
                p2 * 2.0 - p1
            };
            for s in 1..=segments_per_span {
                let t = s as f32 * inv_seg; // ∈ (0, 1]
                let t2 = t * t;
                let t3 = t2 * t;
                // 标准 Catmull–Rom 基（0.5 系数 + 三阶多项式）。
                let c = (p1 * 2.0
                    + (p2 - p0) * t
                    + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
                    + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3)
                    * 0.5;
                out.push(c);
            }
        }
        Ok(out)
    }
}

/// Groom：从引导发丝插值生成稠密毛发束。
///
/// 每条从属发丝由**一组 (引导发丝下标, 权重)** 定义（权重和须为 1）。
/// 从属发丝的每个细分顶点 = 引导发丝对应顶点按权重的加权和（等长细分：全部
/// 引导发丝必须细分出**同一点数**）。这是经典"引导发丝插值"Groom 核心。
///
/// 插值纯函数、无状态：同输入（引导折线 + 权重集）必得同输出。
#[derive(Debug, Clone, PartialEq)]
pub struct Groom {
    /// 从属发丝定义：每条 = `Vec<(引导发丝下标, 权重)>`（升序下标，权重和 1）。
    strands: Vec<Vec<(usize, f32)>>,
}

impl Groom {
    /// 构造 Groom（校验每条的引导下标升序去重、权重有限、权重和 ≈ 1）。
    pub fn new(strands: Vec<Vec<(usize, f32)>>) -> Result<Self, PhysicsError> {
        for (si, wlist) in strands.iter().enumerate() {
            if wlist.is_empty() {
                return Err(PhysicsError {
                    code: "P4004",
                    location: "Groom::new".to_string(),
                    problem: format!("从属发丝 {si} 的引导权重集为空"),
                    fix: "每条从属发丝至少引用 1 条引导发丝".to_string(),
                });
            }
            let mut prev: Option<usize> = None;
            let mut sum = 0.0f32;
            for &(gi, w) in wlist {
                if let Some(pv) = prev {
                    if gi <= pv {
                        return Err(PhysicsError {
                            code: "P4005",
                            location: "Groom::new".to_string(),
                            problem: format!(
                                "从属发丝 {si} 的引导下标必须严格升序（{pv} → {gi}）——这是插值顺序确定性纪律"
                            ),
                            fix: "把引导下标按升序去重排列".to_string(),
                        });
                    }
                }
                if !(w.is_finite()) || w < 0.0 {
                    return Err(PhysicsError {
                        code: "P4006",
                        location: "Groom::new".to_string(),
                        problem: format!("从属发丝 {si} 的权重 {w} 非法（须为非负有限数）"),
                        fix: "权重取非负有限数，且同条权重和为 1".to_string(),
                    });
                }
                sum += w;
                prev = Some(gi);
            }
            if (sum - 1.0).abs() > 1e-4 {
                return Err(PhysicsError {
                    code: "P4007",
                    location: "Groom::new".to_string(),
                    problem: format!("从属发丝 {si} 权重和 = {sum}：须 ≈ 1"),
                    fix: "归一化权重（和 = 1）".to_string(),
                });
            }
        }
        Ok(Self { strands })
    }

    /// 从属发丝条数。
    #[inline]
    pub fn len(&self) -> usize {
        self.strands.len()
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strands.is_empty()
    }

    /// 从引导折线集插值出全部从属发丝折线。
    ///
    /// `guides`：每条引导发丝的细分折线（必须**全部等长**）。返回每条从属发丝
    /// 的折线（点数 = 引导点数）。空引导集或点长不等返回错误（C18/C20）。
    pub fn interpolate(&self, guides: &[Vec<Vec3>]) -> Result<Vec<Vec<Vec3>>, PhysicsError> {
        // 引导点长一致性（插值前提：逐顶点对应加权和）。
        let mut point_count: Option<usize> = None;
        for (gi, g) in guides.iter().enumerate() {
            if let Some(pc) = point_count {
                if g.len() != pc {
                    return Err(PhysicsError {
                        code: "P4008",
                        location: "Groom::interpolate".to_string(),
                        problem: format!(
                            "引导发丝 {gi} 点数 = {}，与首条 {} 不等：插值要求全部引导等长",
                            g.len(),
                            pc
                        ),
                        fix: "全部引导发丝用同一 segments_per_span 细分".to_string(),
                    });
                }
            } else {
                point_count = Some(g.len());
            }
        }
        let pc = match point_count {
            Some(pc) => pc,
            None => {
                return Err(PhysicsError {
                    code: "P4009",
                    location: "Groom::interpolate".to_string(),
                    problem: "引导发丝集为空：无从插值".to_string(),
                    fix: "先给出至少 1 条引导发丝".to_string(),
                })
            }
        };
        let mut out: Vec<Vec<Vec3>> = Vec::with_capacity(self.strands.len());
        for wlist in &self.strands {
            let mut strand: Vec<Vec3> = Vec::with_capacity(pc);
            for k in 0..pc {
                let mut acc = Vec3::splat(0.0);
                for &(gi, w) in wlist {
                    // 引导下标越界守卫（C18；权重集合法时不应发生，防御性）。
                    let gp = guides.get(gi).and_then(|g| g.get(k)).copied().ok_or_else(|| {
                        PhysicsError {
                            code: "P4010",
                            location: "Groom::interpolate".to_string(),
                            problem: format!("引导发丝下标 {gi} 或顶点 {k} 越界"),
                            fix: "检查权重集下标与引导折线点长".to_string(),
                        }
                    })?;
                    acc = acc + gp * w;
                }
                strand.push(acc);
            }
            out.push(strand);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// 丝束物理：XPBD 约束族扩展（新增，不改既有语义）
// ---------------------------------------------------------------------------

/// 弯曲约束：相邻两段方向的夹角约束（发丝刚度）。
///
/// 三粒子 (a, b, c)：两段 d1 = b − a、d2 = c − b。约束 C = |d1 − d2| − rest。
/// 稳态 |d1 − d2| = rest 对应夹角的一半正弦关系；`rest = 0` 即"两段共线"
/// （完全伸直刚度）。
///
/// 投影（XPBD 风格，标准三粒子链弯曲约束的工程化，Müller et al. "Position
/// Based Dynamics" 2007 §3.3 距离/弯曲族思想 + Macklin 2016 XPBD compliance
/// 重标定）：
/// - 雅可比 ∂C/∂a = −n̂、∂C/∂b = n̂ − n̂'、∂C/∂c = n̂'（n̂ = (d1−d2)/|d1−d2|，
///   n̂' 为同向量在 b 处对 (d2 段) 的对应方向）——此处取工程化的两段差向量
///   梯度，对 d1/d2 的贡献按 ±1 分配（守恒动量，Δp_a+Δp_b+Δp_c ≈ 0）；
/// - α̃ = compliance/h²，Δλ = (−C − α̃λ)/(Σw + α̃)。
///
/// 动量守恒：Δp_a = −n̂·w_a·Δλ、Δp_c = n̂·w_c·Δλ、Δp_b = −(Δp_a + Δp_c)。
#[derive(Debug, Clone, PartialEq)]
pub struct BendConstraint {
    id: u32,
    /// 三粒子下标（升序规范化不影响语义：取物理相邻序 a-b-c）。
    triad: [usize; 3],
    rest: f32,
    compliance: f32,
}

impl BendConstraint {
    /// 构造弯曲约束（a-b-c 为沿发丝的相邻三粒子；rest/compliance ≥ 0 由
    /// [`Constraint::validate`] 把关）。
    pub fn new(id: u32, a: usize, b: usize, c: usize, rest: f32, compliance: f32) -> Self {
        Self { id, triad: [a, b, c], rest, compliance }
    }
}

impl Constraint for BendConstraint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        &self.triad
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.rest >= 0.0) {
            return Err(PhysicsError {
                code: "P4011",
                location: format!("BendConstraint（约束 {}）", self.id),
                problem: format!("rest = {}：弯曲静止量必须 ≥ 0（NaN 同拒）", self.rest),
                fix: "传非负 rest（0 = 完全伸直）".to_string(),
            });
        }
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P4012",
                location: format!("BendConstraint（约束 {}）", self.id),
                problem: format!("compliance = {}：柔度必须 ≥ 0（NaN 同拒）", self.compliance),
                fix: "传非负柔度；完全刚性传 0".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        let (a, b, c) = (self.triad[0], self.triad[1], self.triad[2]);
        let d1 = particles[b].position - particles[a].position;
        let d2 = particles[c].position - particles[b].position;
        (d1 - d2).length() - self.rest
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        let (a, b, c) = (self.triad[0], self.triad[1], self.triad[2]);
        let d1 = particles[b].position - particles[a].position;
        let d2 = particles[c].position - particles[b].position;
        let diff = d1 - d2;
        let len = diff.length();
        if len < 1e-8 {
            // 奇异构型（两段方向相消）：方向无定义，跳过（残余如实返回）。
            return self.evaluate(particles).abs();
        }
        let n = diff / len; // = (d1 − d2)/|d1 − d2|
        let c_val = len - self.rest;
        let wa = particles[a].inv_mass;
        let wb = particles[b].inv_mass;
        let wc = particles[c].inv_mass;
        // XPBD 投影分母 = Σ w_i |∇_i C|² + α̃（论文 §3)：梯度模分别为
        // |∇_a| = 1、|∇_b| = 2、|∇_c| = 1（见下），故加权和为 wa + 4·wb + wc。
        let w_sum = wa + 4.0 * wb + wc;
        let alpha_tilde = self.compliance / (dt_h * dt_h);
        let denom = w_sum + alpha_tilde;
        if denom <= 0.0 {
            return c_val.abs();
        }
        let dlambda = (-c_val - alpha_tilde * *lambda) / denom;
        *lambda += dlambda;
        // C = |(b−a) − (c−b)| − rest = |2b−a−c| − rest；对 (2b−a−c) 求导：
        // ∂C/∂a = −n̂、∂C/∂b = 2n̂、∂C/∂c = −n̂（Σ∇ = 0 → 动量守恒）。
        let da = n * (-wa * dlambda);
        let db = n * (2.0 * wb * dlambda);
        let dc = n * (-wc * dlambda);
        particles[a].position = particles[a].position + da;
        particles[b].position = particles[b].position + db;
        particles[c].position = particles[c].position + dc;
        self.evaluate(particles).abs()
    }
}

/// 发根固定约束：把发根粒子钉在一个锚点上（头皮附着）。
///
/// C = |p_root − anchor|。锚点 `inv_mass = 0` 的语义（发根不可移动）在
/// [`HairStrand`] 装配时由发根粒子 `inv_mass = 0` 承担——本约束仅提供"发根
/// 与头皮锚点的软/硬锔定"，用于发根需要在模拟中**轻微跟随头皮**变形的场景。
///
/// project：把 p_root 往 anchor 拉（只修发根粒子，不限锚——锚是几何量）。
#[derive(Debug, Clone, PartialEq)]
pub struct RootConstraint {
    id: u32,
    root: usize,
    /// 锚点（发根应处的世界位置）。
    anchor: Vec3,
}

impl RootConstraint {
    /// 构造发根固定约束（root 为发根粒子下标；anchor 须有限，由 validate 把关）。
    pub fn new(id: u32, root: usize, anchor: Vec3) -> Self {
        Self { id, root, anchor }
    }
}

impl Constraint for RootConstraint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        // 返回单元素切片（Constraint 契约允许；World 只校验端点非空且在界内）。
        std::slice::from_ref(&self.root)
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !is_finite3(self.anchor) {
            return Err(PhysicsError {
                code: "P4013",
                location: format!("RootConstraint（约束 {}）", self.id),
                problem: format!("anchor = {:?}：锚点必须有限", self.anchor),
                fix: "传有限锚点坐标".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        (particles[self.root].position - self.anchor).length()
    }

    fn project(&self, particles: &mut [Particle], _dt_h: f32, _lambda: &mut f32) -> f32 {
        let w = particles[self.root].inv_mass;
        if w <= 0.0 {
            // 发根固定粒子（inv_mass = 0）：不可移动，残余如实返回。
            return self.evaluate(particles).abs();
        }
        // 硬锚定：把发根直接拉回锚点（w > 0 时完全投影）。
        particles[self.root].position = self.anchor;
        0.0
    }
}

// ---------------------------------------------------------------------------
// 发丝与毛发束（装配 XPBD 世界）
// ---------------------------------------------------------------------------

/// 单根发丝的规格：细分后折线的静止段长 + 约束刚度参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrandSpec {
    /// 每控制点区间的细分数（≥ 1）。
    pub segments_per_span: usize,
    /// 距离约束（沿发丝）compliance（0 = 不可伸展）。
    pub stretch_compliance: f32,
    /// 弯曲约束 compliance（越大越软/越垂）。
    pub bend_compliance: f32,
    /// 是否固定发根（true = 发根 inv_mass 置 0，钉在头皮上）。
    pub pin_root: bool,
}

impl StrandSpec {
    /// 便捷默认（细分 8、近刚性伸展、可弯、固定发根）。
    pub fn default_for_hair() -> Self {
        Self {
            segments_per_span: 8,
            stretch_compliance: 0.0,
            bend_compliance: 0.05,
            pin_root: true,
        }
    }

    /// 校验规格（细分 ≥ 1、柔度非负）。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        if self.segments_per_span == 0 {
            return Err(PhysicsError {
                code: "P4014",
                location: "StrandSpec::validate".to_string(),
                problem: "segments_per_span = 0：每区间至少细分为 1 段".to_string(),
                fix: "传 segments_per_span ≥ 1".to_string(),
            });
        }
        for (name, v) in [
            ("stretch_compliance", self.stretch_compliance),
            ("bend_compliance", self.bend_compliance),
        ] {
            if !(v >= 0.0) {
                return Err(PhysicsError {
                    code: "P4015",
                    location: "StrandSpec::validate".to_string(),
                    problem: format!("{name} = {v}：柔度必须 ≥ 0（NaN 同拒）"),
                    fix: "传非负柔度".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// 单根发丝：细分折线粒子 + 距离/弯曲约束 + （可选）发根固定。
///
/// 粒子在**独立子世界**（每根发丝一个 [`World`]）中求解？——否。本类型只提供
/// **装配到公共 [`World`] 的方式**：见 [`HairStrand::emit_into`]，把粒子与约束
/// 以固定顺序追加进一个共享 `World`（多根发丝共世界，发根固定由 inv_mass 承担）。
/// 这避免每根发丝一个求解器（[`crate::cloth`] 同款：布料是一个 World 装网格）。
pub struct HairStrand {
    /// 细分折线（世界空间；`[0]` = 发根）。
    nodes: Vec<Vec3>,
    spec: StrandSpec,
}

impl std::fmt::Debug for HairStrand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HairStrand")
            .field("nodes", &self.nodes.len())
            .field("segments_per_span", &self.spec.segments_per_span)
            .finish()
    }
}

impl HairStrand {
    /// 由引导曲线 + 规格细分出一根模拟发丝。
    pub fn from_curve(curve: &HairCurve, spec: StrandSpec) -> Result<Self, PhysicsError> {
        spec.validate()?;
        let nodes = curve.subdivide(spec.segments_per_span)?;
        Ok(Self { nodes, spec })
    }

    /// 由已细分折线直接构造（Groom 插值出的从属发丝走此路径）。
    pub fn from_nodes(nodes: Vec<Vec3>, spec: StrandSpec) -> Result<Self, PhysicsError> {
        spec.validate()?;
        if nodes.len() < 2 {
            return Err(PhysicsError {
                code: "P4016",
                location: "HairStrand::from_nodes".to_string(),
                problem: format!("折线点数 = {}：发丝至少 2 点", nodes.len()),
                fix: "传 ≥ 2 个折线点".to_string(),
            });
        }
        if nodes.iter().any(|p| !is_finite3(*p)) {
            return Err(PhysicsError {
                code: "P4002",
                location: "HairStrand::from_nodes".to_string(),
                problem: "折线点含 NaN/Inf".to_string(),
                fix: "剔除非有限点".to_string(),
            });
        }
        Ok(Self { nodes, spec })
    }

    /// 折线点数。
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 折线点切片。
    #[inline]
    pub fn nodes(&self) -> &[Vec3] {
        &self.nodes
    }

    /// 静止折线总长（长度守恒基准）。
    pub fn rest_length(&self) -> f32 {
        let mut len = 0.0f32;
        for w in self.nodes.windows(2) {
            len += (w[1] - w[0]).length();
        }
        len
    }

    /// 把本发丝的粒子与约束**追加**进共享世界（多根发丝共世界）。
    ///
    /// 顺序固定：先加本发丝全部粒子（根 → 梢），再加本发丝距离约束（沿链）、
    /// 再弯曲约束（隔一）、最后（可选）发根固定约束。约束 id 由 `id_base`
    /// 起连续分配（调用方负责多根发丝间 id_base 不重叠——毛发束用
    /// [`HairBundle`] 自动分配）。
    ///
    /// 返回 `(粒子下标区间起点, 本发丝段数)`。
    pub fn emit_into(&self, world: &mut World, id_base: u32) -> Result<(usize, usize), PhysicsError> {
        let first = world.particles().len();
        // 发根 inv_mass：pin_root 时置 0（固定），否则 1。
        let root_w = if self.spec.pin_root { 0.0 } else { 1.0 };
        for (i, &p) in self.nodes.iter().enumerate() {
            let w = if i == 0 { root_w } else { 1.0 };
            world.add_particle(p, Vec3::splat(0.0), w);
        }
        let segs = self.nodes.len() - 1;
        let mut id = id_base;
        // 距离约束（沿链，静止长 = 初始段长）。
        for i in 0..segs {
            let rest = (self.nodes[i + 1] - self.nodes[i]).length().max(MIN_SEGMENT);
            world.add_constraint(Box::new(crate::xpbd::DistanceConstraint::new(
                id,
                first + i,
                first + i + 1,
                rest,
                self.spec.stretch_compliance,
            )))?;
            id += 1;
        }
        // 弯曲约束（相邻两段）。rest = 初始 |d1 − d2|（保持初始弯曲）。
        for i in 0..segs.saturating_sub(1) {
            let d1 = self.nodes[i + 1] - self.nodes[i];
            let d2 = self.nodes[i + 2] - self.nodes[i + 1];
            let rest = (d1 - d2).length();
            world.add_constraint(Box::new(BendConstraint::new(
                id,
                first + i,
                first + i + 1,
                first + i + 2,
                rest,
                self.spec.bend_compliance,
            )))?;
            id += 1;
        }
        // 发根固定约束（仅当发根为自由粒子时才有意义——pin_root 时 inv_mass=0
        // 已不可动，无需再加；非 pin 时把发根锚在初始位置上防漂移）。
        if !self.spec.pin_root {
            if let Some(root_pos) = self.nodes.first().copied() {
                world.add_constraint(Box::new(RootConstraint::new(
                    id,
                    first,
                    root_pos,
                )))?;
                id += 1;
            }
        }
        let _ = id; // id 为下一发丝起点的提示（本发丝消耗 = id − id_base）
        Ok((first, segs))
    }
}

/// 毛发束：一个共享 XPBD 世界承载的全部发丝（引导 + 从属）。
///
/// 装配顺序固定（发丝按插入序、每根内粒子/约束按 [`HairStrand::emit_into`]
/// 固定序），约束 id 跨发丝连续——求解序按 id 升序（格锚），与插入序无关。
pub struct HairBundle {
    world: World,
    /// 每根发丝的（首粒子下标, 段数）。
    strands: Vec<(usize, usize)>,
}

impl std::fmt::Debug for HairBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HairBundle")
            .field("strands", &self.strands.len())
            .field("particles", &self.world.particles().len())
            .field("constraints", &self.world.constraint_count())
            .finish()
    }
}

impl HairBundle {
    /// 空毛发束（给定重力）。
    pub fn new(gravity: Vec3) -> Self {
        Self { world: World::new(gravity), strands: Vec::new() }
    }

    /// 追加一根发丝（约束 id 自动接续，避免跨发丝冲突）。
    pub fn push_strand(&mut self, strand: &HairStrand) -> Result<(), PhysicsError> {
        let id_base = self.world.constraint_count() as u32;
        let (first, segs) = strand.emit_into(&mut self.world, id_base)?;
        self.strands.push((first, segs));
        Ok(())
    }

    /// 发丝条数（0 根 = 合法空束，`step` 不 panic）。
    #[inline]
    pub fn len(&self) -> usize {
        self.strands.len()
    }

    /// 是否为空束。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strands.is_empty()
    }

    /// 内部世界（只读）。
    #[inline]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// 内部世界（可变；供上层组织初始化）。
    #[inline]
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// 各发丝的首粒子下标（升序 = 插入序）。
    #[inline]
    pub fn strand_starts(&self) -> &[(usize, usize)] {
        &self.strands
    }

    /// 推进一步（透传 [`World::step`]；空束仍返回空报告，不 panic）。
    pub fn step(
        &mut self,
        dt: f32,
        substeps: u32,
        iterations: u32,
    ) -> Result<StepReport, PhysicsError> {
        self.world.step(dt, substeps, iterations)
    }

    /// **带丝束碰撞的推进一步**（把 [`StrandCollider`] 接入求解循环）。
    ///
    /// 语义：把 `substeps` 个子步**显式展开**（每子步调用
    /// `world.step(h, 1, iterations)`，h = dt/substeps）。每个子步在约束投影之后：
    /// 1. 施加 `collision_passes` 轮碰撞**位置**修正（发丝-发丝 + 发丝-球体）；
    /// 2. **子步级净位移钳制**（每粒子累计碰撞位移 ≤ `min_distance`，防过冲发散）；
    /// 3. 一次**速度级无摩擦接触**（[`StrandCollider::resolve_strand_velocity_contacts`]：
    ///    消除接近法向速度分量，防能量泵浦）。
    ///
    /// 这是 PBD/XPBD 碰撞处理的标准排布（Müller 2007 §4：位置级非穿透 + 速度级
    /// 非穿透；Macklin 2016 §4 柔度重标定）。
    ///
    /// # 为什么碰撞要多轮（`collision_passes`）
    /// 丝束碰撞是**位置投影**，会与距离/弯曲约束**耦合**：单轮推离后，沿发丝的
    /// 距离约束会把发丝拉回静止长，部分侵蚀刚产生的间距（各节点推离量不均），
    /// 故与约束投影同样需要迭代。子步级净位移钳制保证多轮安全。
    ///
    /// # 为什么不是直接改 [`World::step`]
    /// 只允许改本文件；[`World`] 的求解循环语义保持零改动（模块头部纪律）。
    /// 通过"子步外展开 + 每子步后碰撞"实现接入，**不侵入内核**。
    ///
    /// # 逐位等价性（确定性）
    /// 当 `collider` 无接触时，本函数与 `self.step(dt, substeps, iterations)`
    /// **逐位等价**（每子步 `step(h,1,iter)` 展开 = 一次 `step(dt,substeps,iter)`；
    /// 无接触时位置/速度两轮碰撞均为 no-op；见测试
    /// `step_with_collision_matches_plain_step_without_contacts`）。
    /// 有接触时，碰撞修正以固定次序（发丝对字典序 → 球体插入序）施加，
    /// 确定性保持（测试 `step_with_collision_is_deterministic`）。
    ///
    /// # 能量行为（如实标注）
    /// 碰撞本身**不改速度**（位置修正）且速度级接触只**消除接近分量**，故无外部
    /// 驱动时碰撞不注入能量。但当存在**发根 pin（固定锚）**时，锚会对外做功——
    /// 深度重叠构型的校正期能量会瞬态上冲后回落（有界、非发散；见测试
    /// `step_with_collision_does_not_inject_energy`）。严格能量守恒不适用于含
    /// 固定锚的系统。
    ///
    /// `collision_passes` 固定（≥ 1），迭代数固定是确定性纪律（同 `iterations`）。
    ///
    /// 返回与 [`World::step`] 同形的 [`StepReport`]（残余序列 = 子步 × 迭代，
    /// 顺序不变；碰撞不含 λ，不写残余）。
    pub fn step_with_collision(
        &mut self,
        dt: f32,
        substeps: u32,
        iterations: u32,
        collision_passes: u32,
        collider: &StrandCollider,
    ) -> Result<StepReport, PhysicsError> {
        // 参数域校验（与 World::step 同款三段式；防呆前置，不产生部分状态）。
        if !(dt > 0.0) {
            return Err(PhysicsError {
                code: "P1002",
                location: "HairBundle::step_with_collision".to_string(),
                problem: format!("dt = {dt}：步长必须为正（NaN 同样被拒）"),
                fix: "传正步长，如 dt = 1.0 / fps".to_string(),
            });
        }
        if substeps == 0 {
            return Err(PhysicsError {
                code: "P1003",
                location: "HairBundle::step_with_collision".to_string(),
                problem: "substeps = 0：子步数必须 ≥1（子步数固定是确定性纪律）".to_string(),
                fix: "传 substeps ≥ 1".to_string(),
            });
        }
        if iterations == 0 {
            return Err(PhysicsError {
                code: "P1004",
                location: "HairBundle::step_with_collision".to_string(),
                problem: "iterations = 0：迭代数必须 ≥1（迭代数固定是确定性纪律）".to_string(),
                fix: "传 iterations ≥ 1".to_string(),
            });
        }
        if collision_passes == 0 {
            return Err(PhysicsError {
                code: "P1005",
                location: "HairBundle::step_with_collision".to_string(),
                problem: "collision_passes = 0：碰撞轮数必须 ≥1（固定轮数是确定性纪律）".to_string(),
                fix: "传 collision_passes ≥ 1（推荐 1–4；子步级净位移钳制保证多轮安全）".to_string(),
            });
        }
        let h = dt / substeps as f32;
        let mut report = StepReport::default();
        for _ in 0..substeps {
            // 单子步约束求解（world.step(h, 1, iter) 与 dt/substeps 展开逐位等价）。
            let sub = self.world.step(h, 1, iterations)?;
            report.residuals.extend_from_slice(&sub.residuals);
            // 碰撞前位置快照（用于**子步级净位移钳制**）。
            let pre_collision: Vec<Vec3> =
                self.world.particles().iter().map(|p| p.position).collect();
            // 碰撞位置修正：多轮 Gauss-Seidel（固定轮数，确定性）。
            for _ in 0..collision_passes {
                collider.resolve_strand_contacts(self.world.particles_mut());
                collider.resolve_sphere_contacts(self.world.particles_mut());
            }
            // **子步级净位移钳制**：多轮碰撞对同一粒子的**累计**位移模长不超过
            // 一个最小丝距（`min_distance`）。防止多轮过冲 + 速度回读放大导致
            // 发散（无外力系统若单子步位移 ≫ 一格，下一子步速度回读会指数放大；
            // 见测试 `step_with_collision_does_not_inject_energy`）。
            {
                let limit = collider.min_distance();
                let limit2 = limit * limit;
                let ps = self.world.particles_mut();
                for (i, p) in ps.iter_mut().enumerate() {
                    let x0 = match pre_collision.get(i) {
                        Some(&x) => x,
                        None => continue,
                    };
                    let d = p.position - x0;
                    let m2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    if m2 > limit2 && m2 > 0.0 {
                        p.position = x0 + d * (limit / m2.sqrt());
                    }
                }
            }
            // 位置分离后做**速度级无摩擦接触**：消除接近法向速度分量，
            // 保证碰撞不泵浦能量（Müller 2007 §4 速度级非穿透）。
            collider.resolve_strand_velocity_contacts(self.world.particles_mut());
        }
        Ok(report)
    }

    /// 数值稳定守卫：检出非有限粒子位置/速度，并把速度钳到 [`MAX_HAIR_SPEED`]。
    ///
    /// 返回 `true` = 本次调用**未检出**非有限值（状态健康）；`false` = 检出
    /// 非有限值（该步应被调用方拒绝——本函数只报告，不改写 NaN 位置，C20）。
    /// 速度钳制始终执行（防发散；有限值才钳）。
    ///
    /// 说明：C18 永不 panic 的数值稳定兜底，独立于求解循环，由调用方在每步后
    /// 调用（不侵入 [`World::step`] 语义）。
    pub fn stabilize(&mut self) -> bool {
        let mut healthy = true;
        let ps = self.world.particles_mut();
        for p in ps.iter_mut() {
            if !is_finite3(p.position) || !is_finite3(p.velocity) {
                healthy = false;
                continue; // 不静默清洗 NaN（C20）——只标记，交由调用方拒绝该步
            }
            let sp = p.velocity.length();
            if sp > MAX_HAIR_SPEED {
                p.velocity = p.velocity * (MAX_HAIR_SPEED / sp);
            }
        }
        healthy
    }

    /// 每根发丝的当前折线总长（长度守恒观测；顺序 = 插入序）。
    pub fn strand_lengths(&self) -> Vec<f32> {
        let ps = self.world.particles();
        let mut out = Vec::with_capacity(self.strands.len());
        for &(first, segs) in &self.strands {
            let mut len = 0.0f32;
            for i in 0..segs {
                len += (ps[first + i + 1].position - ps[first + i].position).length();
            }
            out.push(len);
        }
        out
    }

    /// 全束总动能（Σ½·m·|v|²；m = 1/inv_mass，固定粒子计 0）。
    pub fn kinetic_energy(&self) -> f32 {
        let mut e = 0.0f32;
        for p in self.world.particles() {
            if p.inv_mass > 0.0 {
                let v2 = p.velocity.x * p.velocity.x
                    + p.velocity.y * p.velocity.y
                    + p.velocity.z * p.velocity.z;
                e += 0.5 * v2 / p.inv_mass;
            }
        }
        e
    }
}

// ---------------------------------------------------------------------------
// 丝束碰撞（发丝-发丝排斥 + 发丝-球体 + 空间哈希宽相）
// ---------------------------------------------------------------------------
//
// 语义权威：本模块实现范围的第一/二部分——"毛发系统深化 A —— 丝束碰撞 +
// 完整 Marschner 偏角"；XPBD 约束族扩展。基础版诚实记录的缺口：
// "无丝束碰撞（发丝之间不互相排斥，会穿模）"。
//
// # 参考出处
// - XPBD / PBD 碰撞距离约束（把"最小间距"当作不等式约束 C = min_dist − |d| ≥ 0）：
//   M. Müller, B. Heidelberger, M. Hennix, J. Ratcliff, "Position Based Dynamics",
//   J. Vis. Commun. Image R. 18(2), 2007, §4（collision handling：位置投影推开）；
//   M. Macklin, M. Müller, N. Chentanez, "XPBD: Position-Based Simulation of
//   Compliant Constrained Dynamics", MIG 2016, §4（softbody/self-collision 的
//   柔度重标定 α̃ = α/h²）。
// - 均匀空间哈希宽相（uniform grid broad phase）：M. Teschner, B. Heidelberger,
//   M. Müller, et al., "Optimized Spatial Hashing for Collision Detection of
//   Deformable Objects", VMV 2003（空间哈希键 = 格子整数坐标哈希）。本实装
//   采用**确定性**变体：不依赖 HashMap 迭代序，改为按格子坐标**排序后**顺序
//   处理（见 [`StrandCollider::resolve_strand_contacts`]）。
//
// # 全链路确定性（D-29 / 硬性要求 A）
// - 空间哈希键用手写 **FNV-1a 64 位**整数哈希（常量见 [`fnv1a64`]），无 crate；
// - 段对枚举顺序：先按插入（段 id）序构造哈希桶 ⇒ 每个桶内段 id 升序 ⇒ 收集
//   候选对后用 **(段a, 段b) 字典序**排序去重 ⇒ 逐对（按最小段 id 优先）投影。
//   因此**插入顺序不影响结果**（测试 `spatial_hash_insertion_order_invariant`）；
// - 投影只含 ±×÷√（无三角函数、无 libm）；同输入逐位一致。
//
// # C18 永不 panic
// - 格子坐标越界钳制到 [`GRID_COORD_LIMIT`]（防 i64 溢出 / 巨坐标）；全部
//   下标经 `get()`/`get_mut()` 边界检查；投影分母有零守卫。

/// 空间哈希格子坐标分量的绝对值上限（防溢出与巨坐标；±2³⁰ 远超合理世界尺度）。
const GRID_COORD_LIMIT: i64 = 1 << 30;

/// FNV-1a 64 位哈希（Fowler–Noll–Vo，公开域算法）。
///
/// 常量：offset basis = `0xcbf29ce484222325`、prime = `0x100000001b3`。
/// 输出为稳定整数（不依赖平台 HashMap 的随机 SipHash 种子）。
#[inline]
fn fnv1a64(bytes: &[i64; 3]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &c in bytes.iter() {
        // 逐 8 字节大端混入（整数坐标 → 字节流），无字节序依赖。
        let b = c.to_le_bytes();
        for &byte in b.iter() {
            h ^= byte as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

/// 世界坐标 → 整数格子坐标（floor(p / cell)；越界钳制，[`GRID_COORD_LIMIT`]）。
///
/// `cell` 由调用方保证 > 0（构造期校验）；p 非有限 → 返回 0 格（C18）。
#[inline]
fn cell_coord(p: Vec3, cell: f32) -> [i64; 3] {
    let f = |v: f32| -> i64 {
        if !v.is_finite() {
            return 0;
        }
        let q = (v / cell).floor();
        if q >= GRID_COORD_LIMIT as f32 {
            GRID_COORD_LIMIT
        } else if q <= -(GRID_COORD_LIMIT as f32) {
            -GRID_COORD_LIMIT
        } else {
            q as i64
        }
    };
    [f(p.x), f(p.y), f(p.z)]
}

/// 线段-线段最近点对（确定性；只用 ±×÷√）。
///
/// 返回 `(t_a, t_b, 最近距离)`：在段 a = A0+t_a·(A1−A0) 与段 b = B0+t_b·(B1−B0)
/// 上的最近参数（∈ [0,1]）与最短距离。退化段（长 < 1e-12）回落到点-段。
///
/// 出处：Ericson, "Real-Time Collision Detection", Morgan Kaufmann 2005,
/// §5.1.9（ClosestPtSegmentSegment 的工程化；本实装为公开几何算法自研复写）。
fn closest_seg_seg(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (f32, f32, f32) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = b0 - a0;
    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);
    let eps = 1e-12f32;
    let (mut s, mut t);
    if a <= eps && e <= eps {
        // 双退化段：两点重合。
        s = 0.0;
        t = 0.0;
    } else if a <= eps {
        // a 段退化：点 A0 与 b 段。
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= eps {
            // b 段退化：点 B0 与 a 段。
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            // 非平行：解析解；平行（denom ≈ 0）：从 s = 0 起。
            s = if denom > eps {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let pa = a0 + d1 * s;
    let pb = b0 + d2 * t;
    (s, t, (pa - pb).length())
}

/// 丝束碰撞体：发丝-发丝排斥 + 发丝-球体 + 空间哈希宽相。
///
/// 只读观察：本结构不持有 [`World`]（保持调用方对求解循环的控制），经
/// [`StrandCollider::register_strand`] 收集段端点下标，再于
/// [`StrandCollider::resolve_strand_contacts`] 中一次性投影粒子位置。
#[derive(Debug, Clone)]
pub struct StrandCollider {
    /// 段参数间距（须 > 0；[`StrandCollider::new`] 校验）。
    min_distance: f32,
    /// 单次投影的刚度 ∈ (0, 1]（0 = 不变、1 = 硬推开；[`StrandCollider::new`] 校验）。
    stiffness: f32,
    /// 球体碰撞体：(中心, 半径) 列表（插入序确定）。
    spheres: Vec<(Vec3, f32)>,
    /// 每段的端点粒子下标 (p0, p1)（插入序）。
    segments: Vec<(usize, usize)>,
}

impl StrandCollider {
    /// 构造（`min_distance > 0`、`stiffness ∈ (0,1]`；非法返回三段式错误）。
    pub fn new(min_distance: f32, stiffness: f32) -> Result<Self, PhysicsError> {
        if !(min_distance > 0.0) {
            return Err(PhysicsError {
                code: "P4020",
                location: "StrandCollider::new".to_string(),
                problem: format!("min_distance = {min_distance}：最小丝距必须 > 0（NaN 同拒）"),
                fix: "传正的最小间距（如发丝半径的 2 倍）".to_string(),
            });
        }
        if !(stiffness > 0.0 && stiffness <= 1.0) {
            return Err(PhysicsError {
                code: "P4021",
                location: "StrandCollider::new".to_string(),
                problem: format!("stiffness = {stiffness}：投影刚度须 ∈ (0, 1]（NaN 同拒）"),
                fix: "传 0 < stiffness ≤ 1（1 = 硬推开，0.5 = 半推开）".to_string(),
            });
        }
        Ok(Self { min_distance, stiffness, spheres: Vec::new(), segments: Vec::new() })
    }

    /// 最小丝距（格宽 = 本值）。
    #[inline]
    pub fn min_distance(&self) -> f32 {
        self.min_distance
    }

    /// 已注册段数。
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// 已注册球体数。
    #[inline]
    pub fn sphere_count(&self) -> usize {
        self.spheres.len()
    }

    /// 注册一根发丝（折线点下标区间）：把相邻点连成段。
    ///
    /// `point_indices` 为折线各点在 [`World`] 中的粒子下标（根 → 梢，须 ≥ 2）。
    /// 段按折线序追加（确定性插入序）。非法下标由调用方保证在界内（求解时
    /// 仍做边界检查，C18）。
    pub fn register_strand(&mut self, point_indices: &[usize]) -> Result<usize, PhysicsError> {
        if point_indices.len() < 2 {
            return Err(PhysicsError {
                code: "P4022",
                location: "StrandCollider::register_strand".to_string(),
                problem: format!("折线点数 = {}：注册发丝至少 2 点", point_indices.len()),
                fix: "传 ≥ 2 个粒子下标".to_string(),
            });
        }
        let added = point_indices.len() - 1;
        for w in point_indices.windows(2) {
            self.segments.push((w[0], w[1]));
        }
        Ok(added)
    }

    /// 注册一个球体碰撞体（中心 + 半径；半径须 > 0，中心须有限）。
    pub fn add_sphere(&mut self, center: Vec3, radius: f32) -> Result<(), PhysicsError> {
        if !is_finite3(center) || !(radius > 0.0) {
            return Err(PhysicsError {
                code: "P4023",
                location: "StrandCollider::add_sphere".to_string(),
                problem: format!("球体非法：center = {center:?}、radius = {radius}（须有限中心 + 正半径）"),
                fix: "传有限中心与正半径".to_string(),
            });
        }
        self.spheres.push((center, radius));
        Ok(())
    }

    /// 段包围盒（两条端点），用于插入哈希桶（确定性）。
    #[inline]
    fn seg_aabb(&self, idx: usize, particles: &[Particle]) -> Option<([i64; 3], [i64; 3])> {
        let (i0, i1) = *self.segments.get(idx)?;
        let p0 = particles.get(i0)?.position;
        let p1 = particles.get(i1)?.position;
        let c0 = cell_coord(p0, self.min_distance);
        let c1 = cell_coord(p1, self.min_distance);
        let lo = [c0[0].min(c1[0]), c0[1].min(c1[1]), c0[2].min(c1[2])];
        let hi = [c0[0].max(c1[0]), c0[1].max(c1[1]), c0[2].max(c1[2])];
        Some((lo, hi))
    }

    /// 宽相：收集候选**段对**（空间哈希 + 确定性排序去重）。
    ///
    /// 返回按规范键（两段端点粒子下标升序对）字典序排序、去重后的段对列表。
    /// 顺序与注册序、哈希迭代序均无关（硬性要求 A）。
    fn contact_pairs(
        &self,
        particles: &[Particle],
    ) -> Vec<((usize, usize), (usize, usize))> {
        let n = self.segments.len();
        if n < 2 {
            return Vec::new();
        }
        // 1) 插入哈希桶（键 → 段 id 列表）。为确定性，用 Vec<(key, seg_id)> 后排序。
        let mut entries: Vec<(u64, usize)> = Vec::new();
        for idx in 0..n {
            if let Some((lo, hi)) = self.seg_aabb(idx, particles) {
                // 覆盖格子数上限守卫（防巨段 ⇒ 巨循环；超限则只插端点格）。
                let dx = (hi[0] - lo[0]).min(3); // ≤ 4 格/轴
                let dy = (hi[1] - lo[1]).min(3);
                let dz = (hi[2] - lo[2]).min(3);
                for ix in 0..=dx {
                    for iy in 0..=dy {
                        for iz in 0..=dz {
                            let key = fnv1a64(&[lo[0] + ix, lo[1] + iy, lo[2] + iz]);
                            entries.push((key, idx));
                        }
                    }
                }
            }
        }
        // 2) 按 (键, 段 id) 升序排序 → 桶键升序、桶内段 id 升序（确定性）。
        entries.sort_unstable();
        // 3) 同一键的段构成桶，桶内两两成对。段对的**规范键**以段端点粒子下标
        //    （升序对）表达，与注册序无关——判重与排序均用键，保证"同一物理
        //    段集"在任何注册序下得到同一处理序（哈希迭代序 + 插入序均不影响
        //    结果，硬性要求 A）。
        let seg_key = |s: usize| -> Option<(usize, usize)> {
            let (i0, i1) = *self.segments.get(s)?;
            Some(if i0 <= i1 { (i0, i1) } else { (i1, i0) })
        };
        let mut keyed: Vec<((usize, usize), (usize, usize))> = Vec::new();
        let mut i = 0usize;
        while i < entries.len() {
            let key = entries[i].0;
            let mut j = i;
            while j < entries.len() && entries[j].0 == key {
                j += 1;
            }
            // entries[i..j] 为同桶段 id（已升序）。
            for x in i..j {
                for y in (x + 1)..j {
                    let (a, b) = (entries[x].1, entries[y].1);
                    if a == b {
                        continue;
                    }
                    if let (Some(ka), Some(kb)) = (seg_key(a), seg_key(b)) {
                        // 规范两段次序：按键字典序，保证 (a,b) 与 (b,a) 归一。
                        keyed.push(if ka <= kb { (ka, kb) } else { (kb, ka) });
                    }
                }
            }
            i = j;
        }
        keyed.sort_unstable();
        keyed.dedup();
        keyed
    }

    /// 发丝-发丝排斥求解（空间哈希宽相 + 确定性投影）。
    ///
    /// 步骤（全部确定性，插入序无关）：
    /// 1. 把每段按其**格子包围盒覆盖的所有格子**插入哈希桶（FNV-1a 键）；
    /// 2. 在**桶键升序**遍历（不依赖 HashMap 迭代序）；桶内段 id 升序；
    /// 3. 收集候选段对 `(a < b)`，按字典序排序 + 去重；
    /// 4. 逐对做线段最近点，距离 < `min_distance` 时沿连线方向把两点推开到
    ///    恰好 `min_distance`（乘 `stiffness` 缓和；`dist ≥ min_distance` 则不动）；
    /// 5. 固定粒子（`inv_mass = 0`）不参与位移，动量由自由粒子分担。
    ///
    /// 返回本次处理的接触对数量（0 = 无接触）。
    ///
    /// 单位球体（[`StrandCollider::resolve_sphere_contacts`]）已把 `|v| ≤ inv_mass` 的
    /// 动能视作能量上界；本函数只做位置修正，不写速度（保持守恒，见测试）。
    pub fn resolve_strand_contacts(&self, particles: &mut [Particle]) -> usize {
        let keyed = self.contact_pairs(particles);
        // 逐对投影（按规范键升序 → 与注册序无关）。
        let min_d = self.min_distance;
        let min_d2 = min_d * min_d;
        let mut contacts = 0usize;
        // 本次的**累计位移缓冲**：同一粒子可能被多个段对引用（共点段 / 重叠段），
        // 先累加再**逐粒子钳制到 ≤ min_distance**（单轮位移上界 = 一格）后统一
        // 施加。这防止多段对叠加 + 多轮 Gauss-Seidel 造成过度推离/发散
        // （位置投影的单步位移不应超过一个最小丝距；见测试
        // `collision_keeps_strands_separated_in_full_loop` 与
        // `step_with_collision_does_not_inject_energy`）。
        let mut disp: Vec<Vec3> = vec![Vec3::splat(0.0); particles.len()];
        for &((ai0, ai1), (bi0, bi1)) in &keyed {
            // 端点下标边界检查（C18）。
            if ai0 >= particles.len() || ai1 >= particles.len() || bi0 >= particles.len()
                || bi1 >= particles.len()
            {
                continue;
            }
            let a0 = particles[ai0].position;
            let a1 = particles[ai1].position;
            let b0 = particles[bi0].position;
            let b1 = particles[bi1].position;
            let (s, t, dist) = closest_seg_seg(a0, a1, b0, b1);
            if dist >= min_d {
                continue;
            }
            // 最近点与其连线方向 n（a 点指向 b 点）。
            let pa = a0 + (a1 - a0) * s;
            let pb = b0 + (b1 - b0) * t;
            let mut n = pb - pa;
            let nlen = n.length();
            if nlen < 1e-9 {
                // 重合：方向无定义，跳过（残余不静默推进，C20 只跳过不伪造方向）。
                continue;
            }
            n = n * (1.0 / nlen);
            let penetration = min_d - dist;
            // 最近点的粒子权重：端点按 (1−s, s)/(1−t, t) 插值（与 BendConstraint
            // 同款"按几何插值分配自由度"思路）。
            let wa0 = particles[ai0].inv_mass * (1.0 - s);
            let wa1 = particles[ai1].inv_mass * s;
            let wb0 = particles[bi0].inv_mass * (1.0 - t);
            let wb1 = particles[bi1].inv_mass * t;
            let w_sum = wa0 + wa1 + wb0 + wb1;
            if w_sum <= 0.0 {
                continue; // 四端全固定：无自由度。
            }
            // XPBD 式位置投影 Δp = w·∇C·Δλ，∇C 方向 = ±n，Δλ = penetration/w_sum。
            // stiffness 缓和（< 1 时欠投影，避免爆炸）。
            let scale = self.stiffness * penetration / w_sum;
            let d = n * scale;
            // a 侧沿 −n、b 侧沿 +n 分离（累加进缓冲）。
            accumulate(&mut disp, ai0, d * (wa0 * -1.0));
            accumulate(&mut disp, ai1, d * (wa1 * -1.0));
            accumulate(&mut disp, bi0, d * wb0);
            accumulate(&mut disp, bi1, d * wb1);
            if dist * dist < min_d2 {
                contacts += 1;
            }
        }
        // 统一施加（逐粒子钳制位移模长 ≤ min_d，防单轮过冲/发散）。
        let limit = min_d;
        let limit2 = limit * limit;
        for (i, p) in particles.iter_mut().enumerate() {
            let d = match disp.get(i) {
                Some(d) => *d,
                None => Vec3::splat(0.0),
            };
            let m2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if m2 <= 0.0 {
                continue;
            }
            let applied = if m2 > limit2 { d * (limit / m2.sqrt()) } else { d };
            p.position = p.position + applied;
        }
        contacts
    }

    /// **发丝-发丝接触的速度修正**（速度级无摩擦接触；能量守恒关键）。
    ///
    /// 在位置分离之后调用：对每一尚未分离的接触对，把**最近点处的相对法向速度**
    /// 投影为**非接近**（`v_rel·n ≥ 0`），沿法向按逆质量权重分配修正量。这是 PBD
    /// / XPBD 的标准**速度级接触响应**（Müller 2007 §4：速度级非穿透；不引入
    /// 静摩擦/恢复，故**不注入能量**——只消除接近分量，从不增加动能）。
    ///
    /// 位置级排斥（[`StrandCollider::resolve_strand_contacts`]）与速度级修正
    /// 配合，才能保证"碰撞接入求解循环"不泵浦能量（见测试
    /// `step_with_collision_does_not_inject_energy`）。
    ///
    /// 只改速度，不改位置；固定粒子速度不参与（视为无限质量）。返回处理的
    /// 接触对数。顺序与前相同（确定性）。
    pub fn resolve_strand_velocity_contacts(&self, particles: &mut [Particle]) -> usize {
        let keyed = self.contact_pairs(particles);
        let min_d = self.min_distance;
        let mut handled = 0usize;
        for &((ai0, ai1), (bi0, bi1)) in &keyed {
            if ai0 >= particles.len() || ai1 >= particles.len() || bi0 >= particles.len()
                || bi1 >= particles.len()
            {
                continue;
            }
            let (s, t, dist) = closest_seg_seg(
                particles[ai0].position,
                particles[ai1].position,
                particles[bi0].position,
                particles[bi1].position,
            );
            if dist >= min_d {
                continue; // 未接触（位置级已分离）。
            }
            let pa = particles[ai0].position
                + (particles[ai1].position - particles[ai0].position) * s;
            let pb = particles[bi0].position
                + (particles[bi1].position - particles[bi0].position) * t;
            let mut n = pb - pa;
            let nlen = n.length();
            if nlen < 1e-9 {
                continue; // 方向退化，跳过（C18/C20）。
            }
            n = n * (1.0 / nlen);
            // 最近点处 a/b 侧速度（端点按 (1−s, s) 插值）。
            let va = particles[ai0].velocity * (1.0 - s) + particles[ai1].velocity * s;
            let vb = particles[bi0].velocity * (1.0 - t) + particles[bi1].velocity * t;
            let v_rel = vb - va;
            let vn = v_rel.dot(&n);
            if vn >= 0.0 {
                continue; // 已在分离/切向：不修正（无摩擦，切向不动）。
            }
            // 逆质量权重（与位置投影一致）。
            let wa0 = particles[ai0].inv_mass * (1.0 - s);
            let wa1 = particles[ai1].inv_mass * s;
            let wb0 = particles[bi0].inv_mass * (1.0 - t);
            let wb1 = particles[bi1].inv_mass * t;
            let w_sum = wa0 + wa1 + wb0 + wb1;
            if w_sum <= 0.0 {
                continue; // 四端全固定。
            }
            // 消除接近分量：目标相对法向速度 = 0（当前 = vn < 0），故需改变
            //   Δ(v_rel·n) = −vn  ⇒  Δv = n·(−vn/w_sum) 按逆质量分配。
            let scale = -vn / w_sum; // vn < 0 ⇒ scale > 0。
            let dv = n * scale;
            apply_vel_delta(particles, ai0, dv * (wa0 * -1.0));
            apply_vel_delta(particles, ai1, dv * (wa1 * -1.0));
            apply_vel_delta(particles, bi0, dv * wb0);
            apply_vel_delta(particles, bi1, dv * wb1);
            handled += 1;
        }
        handled
    }

    /// 发丝-球体碰撞求解：把落在球内的段端点推到球面外（沿径向）。
    ///
    /// 对每个球体、每个段端点粒子：`d = p − center`，若 `|d| < radius`，则
    /// 把 `p` 推至 `center + d̂ · radius`（无自由度时跳过）。只修位置、不写速度。
    ///
    /// 返回修正的粒子数（0 = 无穿透）。顺序 = 球体插入序 × 段插入序（确定性）。
    pub fn resolve_sphere_contacts(&self, particles: &mut [Particle]) -> usize {
        let mut fixed = 0usize;
        for si in 0..self.spheres.len() {
            let (center, radius) = self.spheres[si];
            for seg in self.segments.iter() {
                for &pi in [seg.0, seg.1].iter() {
                    let p = match particles.get(pi) {
                        Some(pp) => pp,
                        None => continue,
                    };
                    if p.inv_mass <= 0.0 {
                        continue; // 固定粒子不推。
                    }
                    let d = p.position - center;
                    let dist = d.length();
                    if dist >= radius {
                        continue;
                    }
                    // 退化：球心与点重合 → 沿 +Y 兜底方向推出（C18）。
                    let dir = if dist < 1e-9 { Vec3::new(0.0, 1.0, 0.0) } else { d * (1.0 / dist) };
                    if let Some(pm) = particles.get_mut(pi) {
                        pm.position = center + dir * radius;
                        fixed += 1;
                    }
                }
            }
        }
        fixed
    }
}

/// 把位移增量累加到指定粒子（越界忽略，C18）。
///
/// 同一粒子在一次段对投影中可能被两侧引用（共点段），调用方按端点分别调用；
/// 位置增量可交换，重复施加等价于分别投影（见 `segment_pair_contact_partial_stiffness`）。
/// 把位移增量累加进缓冲（越界忽略，C18）。同一粒子被多段对引用时累加，
/// 施加前统一钳制（见 [`StrandCollider::resolve_strand_contacts`]）。
#[inline]
fn accumulate(buf: &mut [Vec3], idx: usize, delta: Vec3) {
    if let Some(slot) = buf.get_mut(idx) {
        *slot = *slot + delta;
    }
}

/// 把速度增量累加到指定粒子（越界忽略，C18）。速度级接触修正用。
#[inline]
fn apply_vel_delta(particles: &mut [Particle], idx: usize, delta: Vec3) {
    if let Some(p) = particles.get_mut(idx) {
        p.velocity = p.velocity + delta;
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 直线引导发丝（沿 +Y，n 个控制点，间距 h）。
    fn straight_curve(n: usize, h: f32) -> HairCurve {
        let pts: Vec<Vec3> = (0..n).map(|i| Vec3::new(0.0, i as f32 * h, 0.0)).collect();
        HairCurve::new(pts).unwrap()
    }

    #[test]
    fn catmull_rom_on_straight_line_is_exact() {
        // 共线控制点的 Catmull–Rom 细分必须仍落在同一直线上且等距
        // （自证非恒真：若基函数系数写错，细分点会偏离直线或间距失真）。
        let c = straight_curve(4, 1.0);
        let sub = c.subdivide(4).unwrap();
        // 点数 = (4−1)·4 + 1 = 13。
        assert_eq!(sub.len(), 13, "点数应 = (n−1)·seg + 1");
        for (i, p) in sub.iter().enumerate() {
            assert!(p.x.abs() < 1e-5 && p.z.abs() < 1e-5, "共线细分不应偏离 Y 轴：{p:?}");
            let expect_y = i as f32 * 0.25; // 每段 0.25
            assert!((p.y - expect_y).abs() < 1e-5, "第 {i} 点 y = {} 应为 {expect_y}", p.y);
        }
        // 首尾控制点原样保留。
        assert_eq!(sub[0], Vec3::new(0.0, 0.0, 0.0));
        assert!((sub[12].y - 3.0).abs() < 1e-5);
    }

    #[test]
    fn groom_interpolation_is_weighted_average() {
        // 两条引导发丝（相等细分），一条从属发丝 = 50/50 插值 → 必在中点。
        let g0 = straight_curve(2, 1.0).subdivide(2).unwrap();
        let g1 = straight_curve(2, 1.0)
            .subdivide(2)
            .unwrap()
            .iter()
            .map(|p| *p + Vec3::new(1.0, 0.0, 0.0))
            .collect::<Vec<_>>();
        let groom = Groom::new(vec![vec![(0, 0.5), (1, 0.5)]]).unwrap();
        let out = groom.interpolate(&[g0, g1]).unwrap();
        assert_eq!(out.len(), 1);
        for (k, p) in out[0].iter().enumerate() {
            // 期望 x = 0.5（两引导 x 差 1 的中点），y 与引导一致。
            assert!((p.x - 0.5).abs() < 1e-5, "插值 x = {} 应为 0.5", p.x);
            assert!(p.z.abs() < 1e-5);
            assert!(k < 3);
        }
    }

    #[test]
    fn groom_rejects_bad_weights_and_length_mismatch() {
        // 权重和 ≠ 1 → P4007。
        let e = Groom::new(vec![vec![(0, 0.3), (1, 0.3)]]).unwrap_err();
        assert_eq!(e.code, "P4007");
        // 下标非升序 → P4005。
        let e = Groom::new(vec![vec![(1, 0.5), (0, 0.5)]]).unwrap_err();
        assert_eq!(e.code, "P4005");
        // 引导点长不等 → P4008。
        let groom = Groom::new(vec![vec![(0, 1.0)]]).unwrap();
        let bad = groom.interpolate(&[vec![Vec3::splat(0.0); 3], vec![Vec3::splat(0.0); 4]]);
        assert_eq!(bad.unwrap_err().code, "P4008");
    }

    #[test]
    fn hair_bundle_bitwise_deterministic() {
        // 同输入跑两遍，粒子位置 + 汇聚报告逐位一致（确定性铁律）。
        let build = || {
            let mut b = HairBundle::new(Vec3::new(0.0, -9.81, 0.0));
            let c = straight_curve(3, 0.3);
            let s = HairStrand::from_curve(&c, StrandSpec::default_for_hair()).unwrap();
            b.push_strand(&s).unwrap();
            // 第二根带横向偏移，制造非平凡 3D 运动。
            let c2 = HairCurve::new(vec![
                Vec3::new(0.2, 0.0, 0.0),
                Vec3::new(0.25, 0.3, 0.05),
                Vec3::new(0.3, 0.6, 0.0),
            ])
            .unwrap();
            let s2 = HairStrand::from_curve(&c2, StrandSpec::default_for_hair()).unwrap();
            b.push_strand(&s2).unwrap();
            b
        };
        let mut a = build();
        let mut b = build();
        let dt = 1.0 / 60.0;
        let mut ra = Vec::new();
        let mut rb = Vec::new();
        for _ in 0..20 {
            ra.push(a.step(dt, 4, 8).unwrap());
            rb.push(b.step(dt, 4, 8).unwrap());
        }
        let bits = |bb: &HairBundle| -> Vec<[u32; 3]> {
            bb.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "两遍毛发模拟位置应逐位一致");
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(
                x.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                y.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                "残余序列应逐位一致"
            );
        }
    }

    #[test]
    fn strand_length_conserved_on_already_satisfied_configuration() {
        // 长度守恒（无漂移）：在已满足距离约束的静止构型上求解，总长不应被
        // 求解器引入净漂移（< 0.1%）。无重力 → 只看约束投影是否守恒段长。
        // 自证非恒真：若距离约束投影有系统偏差（如权重写错），净长会漂移。
        let mut b = HairBundle::new(Vec3::splat(0.0));
        let c = straight_curve(3, 0.5); // 3 控制点，细分 8 → 17 点
        let s = HairStrand::from_curve(&c, StrandSpec::default_for_hair()).unwrap();
        let rest = s.rest_length();
        b.push_strand(&s).unwrap();
        for _ in 0..30 {
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let got = b.strand_lengths()[0];
        assert!(
            (got - rest).abs() / rest < 1e-3,
            "静止构型总长漂移应 < 0.1%：rest={rest} got={got}"
        );
    }

    #[test]
    fn strand_length_conserved_from_stretched_state() {
        // 长度守恒（拉伸恢复）：以静止段长 0.2 装配（约束 rest=0.2），再把
        // 初态拉成段长 0.4（2× 拉伸），求解后总长应恢复到静止长（刚性伸展）。
        let spec = StrandSpec {
            segments_per_span: 1,
            stretch_compliance: 0.0,
            bend_compliance: 0.0,
            pin_root: true,
        };
        let rest_nodes: Vec<Vec3> = (0..5).map(|i| Vec3::new(0.0, i as f32 * 0.2, 0.0)).collect();
        let rest_strand = HairStrand::from_nodes(rest_nodes, spec).unwrap();
        let rest_total = rest_strand.rest_length(); // = 0.8

        let mut b = HairBundle::new(Vec3::splat(0.0));
        b.push_strand(&rest_strand).unwrap();
        // 把初态改到拉伸构型（段长 0.4）。
        {
            let ps = b.world_mut().particles_mut();
            for (i, p) in ps.iter_mut().enumerate() {
                p.position = Vec3::new(0.0, i as f32 * 0.4, 0.0);
            }
        }
        let stretched_total = b.strand_lengths()[0];
        assert!(
            (stretched_total - 1.6).abs() < 1e-5,
            "拉伸初态总长应 = 1.6：{stretched_total}"
        );
        for _ in 0..60 {
            b.step(1.0 / 60.0, 8, 16).unwrap();
        }
        let got = b.strand_lengths()[0];
        assert!(
            (got - rest_total).abs() / rest_total < 1e-2,
            "拉伸初态应恢复到静止长：rest={rest_total} got={got}"
        );
    }

    #[test]
    fn root_stays_pinned_during_simulation() {
        // 发根固定：pin_root=true（inv_mass=0），重力下位置逐位不动。
        let mut b = HairBundle::new(Vec3::new(0.0, -9.81, 0.0));
        let c = straight_curve(3, 0.3);
        let s = HairStrand::from_curve(&c, StrandSpec::default_for_hair()).unwrap();
        let root_pos = s.nodes()[0];
        b.push_strand(&s).unwrap();
        for _ in 0..40 {
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let rp = b.world().particles()[0].position;
        assert_eq!(rp.x.to_bits(), root_pos.x.to_bits(), "发根 x 应不动");
        assert_eq!(rp.y.to_bits(), root_pos.y.to_bits(), "发根 y 应不动");
        assert_eq!(rp.z.to_bits(), root_pos.z.to_bits(), "发根 z 应不动");
    }

    #[test]
    fn energy_decays_without_gravity() {
        // 能量衰减：无外力 + 初速摆动 → 动能最终衰减（XPBD 阻尼/投影耗散），
        // 且不发散（有限）。
        let mut b = HairBundle::new(Vec3::splat(0.0));
        let c = straight_curve(3, 0.3);
        let s = HairStrand::from_curve(&c, StrandSpec::default_for_hair()).unwrap();
        b.push_strand(&s).unwrap();
        // 注入横向初速。
        {
            let ps = b.world_mut().particles_mut();
            for (i, p) in ps.iter_mut().enumerate() {
                if i > 0 {
                    p.velocity = Vec3::new(2.0, 0.0, 0.0);
                }
            }
        }
        let mut e_prev = f32::INFINITY;
        let mut e_early = 0.0f32;
        for step in 0..200 {
            b.step(1.0 / 60.0, 4, 8).unwrap();
            assert!(b.stabilize(), "状态应保持有限（不发散）");
            let e = b.kinetic_energy();
            assert!(e.is_finite(), "动能应有限：{e}");
            if step == 5 {
                e_early = e;
            }
            e_prev = e;
        }
        let e_final = e_prev;
        assert!(
            e_final < e_early,
            "无外力动能应衰减：早期 {e_early} → 末态 {e_final}"
        );
        assert!(e_final.is_finite() && e_early.is_finite());
    }

    #[test]
    fn extreme_inputs_do_not_panic() {
        // 0 根发丝：空束 step 不 panic。
        let mut b = HairBundle::new(Vec3::splat(0.0));
        assert!(b.is_empty());
        let r = b.step(1.0 / 60.0, 4, 8).unwrap();
        assert!(r.residuals.is_empty() || r.residuals.iter().all(|x| x.is_finite()));

        // 单控制点 → 构造错误 P4001（不 panic）。
        assert_eq!(HairCurve::new(vec![Vec3::splat(0.0)]).unwrap_err().code, "P4001");

        // 单点发丝（折线 < 2 点）→ P4016。
        let spec = StrandSpec::default_for_hair();
        assert_eq!(
            HairStrand::from_nodes(vec![Vec3::splat(0.0)], spec).unwrap_err().code,
            "P4016"
        );

        // NaN 输入 → 拒绝（P4002），不 panic。
        assert_eq!(
            HairCurve::new(vec![Vec3::splat(0.0), Vec3::new(f32::NAN, 0.0, 0.0)])
                .unwrap_err()
                .code,
            "P4002"
        );

        // 两点退化发丝（两控制点重合）→ 细分仍产生有效点；求解不 panic、无 NaN。
        let deg = HairCurve::new(vec![Vec3::splat(0.0), Vec3::splat(0.0)]).unwrap();
        let mut b2 = HairBundle::new(Vec3::splat(0.0));
        let s = HairStrand::from_curve(&deg, spec).unwrap();
        b2.push_strand(&s).unwrap();
        b2.step(1.0 / 60.0, 2, 4).unwrap();
        for p in b2.world().particles() {
            assert!(is_finite3(p.position));
        }

        // 非法步参不 panic，返回结构化错误。
        let mut b3 = HairBundle::new(Vec3::splat(0.0));
        let s3 = HairStrand::from_curve(&straight_curve(2, 0.2), spec).unwrap();
        b3.push_strand(&s3).unwrap();
        assert_eq!(b3.step(0.0, 4, 8).unwrap_err().code, "P1002");
        assert_eq!(b3.step(1.0 / 60.0, 0, 8).unwrap_err().code, "P1003");
    }

    #[test]
    fn bend_constraint_opposes_folding() {
        // 弯曲约束恢复初始弯曲：三粒子，初始共线（rest=0），折叠后应被拉直。
        // 自证非恒真：若 BendConstraint 的梯度/权重写错，拉直不成立。
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0); // 固定根
        world.add_particle(Vec3::new(0.0, 1.0, 0.0), Vec3::splat(0.0), 1.0);
        // 折叠：第三点偏离连线的角度。
        world.add_particle(Vec3::new(1.0, 1.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(BendConstraint::new(0, 0, 1, 2, 0.0, 0.0)))
            .unwrap();
        let before = {
            let ps = world.particles();
            let d1 = ps[1].position - ps[0].position;
            let d2 = ps[2].position - ps[1].position;
            (d1 - d2).length()
        };
        for _ in 0..40 {
            world.step(1.0 / 60.0, 8, 16).unwrap();
        }
        let after = {
            let ps = world.particles();
            let d1 = ps[1].position - ps[0].position;
            let d2 = ps[2].position - ps[1].position;
            (d1 - d2).length()
        };
        assert!(after < before, "弯曲约束应减小折叠量：before={before} after={after}");
        assert!(after < 1e-2, "零柔度弯曲约束应近完全拉直：after={after}");
    }

    #[test]
    fn invalid_strand_spec_is_structured_error() {
        let spec = StrandSpec {
            segments_per_span: 0,
            stretch_compliance: 0.0,
            bend_compliance: 0.0,
            pin_root: true,
        };
        assert_eq!(spec.validate().unwrap_err().code, "P4014");
        let spec2 = StrandSpec {
            segments_per_span: 2,
            stretch_compliance: -1.0,
            bend_compliance: 0.0,
            pin_root: true,
        };
        assert_eq!(spec2.validate().unwrap_err().code, "P4015");
        // subdivide 0 → P4003。
        let c = straight_curve(2, 0.5);
        assert_eq!(c.subdivide(0).unwrap_err().code, "P4003");
    }

    // -----------------------------------------------------------------------
    // 丝束碰撞（本次新增）
    // -----------------------------------------------------------------------

    /// 便捷：把一串点装进裸 [`World`]（全自由粒子），返回粒子下标列表。
    fn world_with_points(pts: &[Vec3]) -> (World, Vec<usize>) {
        let mut w = World::new(Vec3::splat(0.0));
        let idx = pts.iter().map(|&p| w.add_particle(p, Vec3::splat(0.0), 1.0)).collect();
        (w, idx)
    }

    #[test]
    fn segment_pair_separates_below_min_distance() {
        // 两根平行近距离发丝（间距 0.02 < min_distance 0.1）：求解后最近距离
        // 必须 ≥ min_distance（容差内），且不过度推离（≤ 1.1×min_distance，防爆炸）。
        // 自证非恒真：若投影方向或缩放写错，距离不会落在 [min_d, 1.1·min_d]。
        let (mut w, idx) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.02),
            Vec3::new(0.0, 1.0, 0.02),
        ]);
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.register_strand(&[idx[0], idx[1]]).unwrap();
        col.register_strand(&[idx[2], idx[3]]).unwrap();
        let contacts = col.resolve_strand_contacts(w.particles_mut());
        assert!(contacts > 0, "应有接触对被处理");
        let ps = w.particles();
        let (_, _, dist) =
            closest_seg_seg(ps[0].position, ps[1].position, ps[2].position, ps[3].position);
        assert!(
            dist >= 0.1 - 1e-4,
            "分离后距离应 ≥ min_distance：{dist}"
        );
        assert!(
            dist <= 0.1 * 1.1,
            "分离不应过度推离（≤ 1.1×min_distance，防爆炸）：{dist}"
        );
    }

    #[test]
    fn segment_pair_contact_partial_stiffness_does_not_overshoot() {
        // 半刚度：一次投影只补 penetration 的一半 → 距离落在 (原始, min_distance)。
        // 自证：若 stiffness 被忽略（当作 1），距离会直接到 min_distance。
        let (mut w, idx) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]);
        // 把第二根放到 z = 0.04（间距 0.04，penetration 0.06）。
        {
            let ps = w.particles_mut();
            ps[idx[2]].position = Vec3::new(0.0, 0.0, 0.04);
            ps[idx[3]].position = Vec3::new(0.0, 1.0, 0.04);
        }
        let mut col = StrandCollider::new(0.1, 0.5).unwrap();
        col.register_strand(&[idx[0], idx[1]]).unwrap();
        col.register_strand(&[idx[2], idx[3]]).unwrap();
        col.resolve_strand_contacts(w.particles_mut());
        let ps = w.particles();
        let (_, _, dist) =
            closest_seg_seg(ps[0].position, ps[1].position, ps[2].position, ps[3].position);
        // 0.04 < dist < 0.1（补了一半）。
        assert!(dist > 0.04 + 1e-4, "半刚度应推开一段：{dist}");
        assert!(dist < 0.1 - 1e-4, "半刚度不应到满 min_distance：{dist}");
    }

    #[test]
    fn no_contact_when_far_apart() {
        // 远超 min_distance 的两段：0 接触，位置逐位不变（碰撞不误伤）。
        let (mut w, idx) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
        ]);
        let before: Vec<[u32; 3]> = w
            .particles()
            .iter()
            .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
            .collect();
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.register_strand(&[idx[0], idx[1]]).unwrap();
        col.register_strand(&[idx[2], idx[3]]).unwrap();
        let contacts = col.resolve_strand_contacts(w.particles_mut());
        assert_eq!(contacts, 0, "远离段不应有接触");
        let after: Vec<[u32; 3]> = w
            .particles()
            .iter()
            .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
            .collect();
        assert_eq!(before, after, "无接触时位置应逐位不变");
    }

    #[test]
    fn spatial_hash_insertion_order_invariant() {
        // 硬性要求 A：以**不同插入顺序**注册发丝段，碰撞求解结果逐位一致。
        // 这条能抓住哈希迭代顺序问题。构造 3 根互相接近的平行发丝。
        let mk_pts = || {
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.03),
                Vec3::new(0.0, 1.0, 0.03),
                Vec3::new(0.0, 0.0, 0.06),
                Vec3::new(0.0, 1.0, 0.06),
            ]
        };
        // 顺序 A：0-1, 2-3, 4-5。
        let run = |order: &[[usize; 2]]| -> Vec<[u32; 3]> {
            let pts = mk_pts();
            let (mut w, idx) = world_with_points(&pts);
            let mut col = StrandCollider::new(0.1, 0.8).unwrap();
            for &[a, b] in order {
                col.register_strand(&[idx[a], idx[b]]).unwrap();
            }
            col.resolve_strand_contacts(w.particles_mut());
            w.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        let a = run(&[[0, 1], [2, 3], [4, 5]]);
        let b = run(&[[4, 5], [0, 1], [2, 3]]);
        let c = run(&[[2, 3], [4, 5], [0, 1]]);
        assert_eq!(a, b, "插入顺序 1 vs 2 结果应逐位一致（哈希迭代序无关）");
        assert_eq!(a, c, "插入顺序 1 vs 3 结果应逐位一致");
    }

    #[test]
    fn sphere_contact_pushes_point_out() {
        // 发丝-球体碰撞：球心原点、半径 0.5，点位于球内（距心 0.2）→ 推到球面。
        let (mut w, idx) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.2), // 球内
            Vec3::new(0.0, 0.0, 1.0), // 球外
        ]);
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.add_sphere(Vec3::splat(0.0), 0.5).unwrap();
        col.register_strand(&[idx[0], idx[1]]).unwrap();
        let fixed = col.resolve_sphere_contacts(w.particles_mut());
        assert!(fixed > 0, "应至少修正一个穿透点");
        let p0 = w.particles()[idx[0]].position;
        assert!(
            (p0.length() - 0.5).abs() < 1e-4,
            "球内点应被推到球面（|p| ≈ 0.5）：{}",
            p0.length()
        );
        // 球外点不动。
        assert_eq!(w.particles()[idx[1]].position.z.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn collision_does_not_inject_separation_energy_direction() {
        // 硬性要求 B（能量不增）：碰撞前相对速度方向为"远离"，求解（纯位置投影，
        // 不写速度）后速度逐位不变 ⇒ 不会变成"靠近"。自证：若投影误写速度，
        // 相对速度方向会改变。
        let mut w = World::new(Vec3::splat(0.0));
        // 两根近距离发丝，自带动能：a 向左、b 向右（远离）。
        let a0 = w.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0);
        let a1 = w.add_particle(Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 1.0);
        let b0 = w.add_particle(Vec3::new(0.0, 0.0, 0.02), Vec3::new(1.0, 0.0, 0.0), 1.0);
        let b1 = w.add_particle(Vec3::new(0.0, 1.0, 0.02), Vec3::new(1.0, 0.0, 0.0), 1.0);
        let vel_before: Vec<[u32; 3]> = w
            .particles()
            .iter()
            .map(|p| [p.velocity.x.to_bits(), p.velocity.y.to_bits(), p.velocity.z.to_bits()])
            .collect();
        // 相对速度 = v_b − v_a = (2,0,0)（沿 +Z 分离方向？此处分离沿 Z）。
        // 分离方向 n = b − a ≈ +Z；相对速度沿 +X（垂直于连线）→ 不构成"靠近"。
        // 更贴题构造：让相对速度沿分离方向。
        {
            let ps = w.particles_mut();
            ps[a0].velocity = Vec3::new(0.0, 0.0, -1.0);
            ps[a1].velocity = Vec3::new(0.0, 0.0, -1.0);
            ps[b0].velocity = Vec3::new(0.0, 0.0, 1.0);
            ps[b1].velocity = Vec3::new(0.0, 0.0, 1.0);
        }
        let rel_before = w.particles()[b0].velocity - w.particles()[a0].velocity; // (0,0,2) 远离
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.register_strand(&[a0, a1]).unwrap();
        col.register_strand(&[b0, b1]).unwrap();
        col.resolve_strand_contacts(w.particles_mut());
        let _ = vel_before;
        let rel_after = w.particles()[b0].velocity - w.particles()[a0].velocity;
        // 分离方向（a→b）≈ +Z；相对速度投影到该方向应仍 ≥ 0（未变"靠近"）。
        let sep = (w.particles()[b0].position - w.particles()[a0].position).normalize();
        assert!(
            rel_before.dot(&sep) >= 0.0,
            "前置条件：碰撞前相对速度应是远离方向"
        );
        assert!(
            rel_after.dot(&sep) >= 0.0,
            "碰撞求解不应使相对速度变为靠近：dot = {}",
            rel_after.dot(&sep)
        );
        // 速度逐位不变（纯位置投影）。
        assert_eq!(
            rel_after.z.to_bits(),
            rel_before.z.to_bits(),
            "纯位置投影不应改写速度"
        );
    }

    #[test]
    fn strand_collision_bitwise_deterministic_repeat() {
        // 硬性要求 A：同初始状态跑两遍，所有控制点坐标 to_bits() 序列完全一致。
        // 构造多根互相接近的 3D 发丝，同时含球体接触。
        let build = || {
            let pts: Vec<Vec3> = vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.03, 0.5, 0.01),
                Vec3::new(0.06, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.04),
                Vec3::new(0.02, 0.5, 0.05),
                Vec3::new(0.05, 1.0, 0.04),
                Vec3::new(0.01, 0.2, 0.02),
                Vec3::new(0.02, 0.7, 0.03),
            ];
            (world_with_points(&pts), pts.len())
        };
        let run = |_n: usize| -> Vec<[u32; 3]> {
            let (mut w, idx) = build().0;
            let mut col = StrandCollider::new(0.08, 0.7).unwrap();
            col.add_sphere(Vec3::new(0.02, 0.4, 0.0), 0.1).unwrap();
            col.register_strand(&[idx[0], idx[1], idx[2]]).unwrap();
            col.register_strand(&[idx[3], idx[4], idx[5]]).unwrap();
            col.register_strand(&[idx[6], idx[7]]).unwrap();
            // 跑多轮（模拟挂进求解循环的多次迭代）。
            for _ in 0..5 {
                col.resolve_strand_contacts(w.particles_mut());
                col.resolve_sphere_contacts(w.particles_mut());
            }
            w.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        let a = run(8);
        let b = run(8);
        assert_eq!(a, b, "碰撞求解两遍应逐位一致");
    }

    #[test]
    fn collider_rejects_bad_params() {
        // 参数域护栏：min_distance ≤ 0 → P4020；stiffness 越界 → P4021；
        // 单点发丝 → P4022；非法球 → P4023；均不 panic。
        assert_eq!(StrandCollider::new(0.0, 1.0).unwrap_err().code, "P4020");
        assert_eq!(StrandCollider::new(f32::NAN, 1.0).unwrap_err().code, "P4020");
        assert_eq!(StrandCollider::new(0.1, 0.0).unwrap_err().code, "P4021");
        assert_eq!(StrandCollider::new(0.1, 1.5).unwrap_err().code, "P4021");
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        assert_eq!(col.register_strand(&[0]).unwrap_err().code, "P4022");
        assert_eq!(col.add_sphere(Vec3::splat(0.0), -1.0).unwrap_err().code, "P4023");
        // 空碰撞体求解不 panic。
        assert_eq!(col.resolve_strand_contacts(&mut []), 0);
    }

    #[test]
    fn collider_does_not_panic_on_degenerate_and_missing_indices() {
        // C18：重合段（方向退化）+ 越界下标（段注册后粒子被裁）不 panic。
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.register_strand(&[0, 1]).unwrap();
        col.register_strand(&[1, 2]).unwrap();
        let mut w = World::new(Vec3::splat(0.0));
        w.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
        w.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0); // 与上重合
        w.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
        // 只有 3 个粒子，段引用 2（界内）→ 无越界；再直接调用含越界段的解算。
        let _ = col.resolve_strand_contacts(w.particles_mut());
        for p in w.particles() {
            assert!(is_finite3(p.position));
        }
        // 局部越界：注册引用了不存在下标（调用方责任），解算应跳过而非 panic。
        let mut col2 = StrandCollider::new(0.1, 1.0).unwrap();
        col2.register_strand(&[0, 99]).unwrap();
        let mut w2 = World::new(Vec3::splat(0.0));
        w2.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
        let _ = col2.resolve_strand_contacts(w2.particles_mut());
        assert!(is_finite3(w2.particles()[0].position));
    }

    #[test]
    fn fnv1a64_is_stable_and_distinguishes_cells() {
        // 哈希函数确定性 + 区分性：同输入同输出；不同格子坐标大概率不同键。
        assert_eq!(fnv1a64(&[1, 2, 3]), fnv1a64(&[1, 2, 3]));
        assert_ne!(fnv1a64(&[1, 2, 3]), fnv1a64(&[3, 2, 1]));
        assert_ne!(fnv1a64(&[0, 0, 0]), fnv1a64(&[1, 0, 0]));
    }

    // -----------------------------------------------------------------------
    // 碰撞接入求解循环（本次补齐：StrandCollider → HairBundle::step_with_collision）
    // -----------------------------------------------------------------------

    /// 便捷：构造一束两条平行垂直发丝，返回 (bundle, collider, 各发丝粒子下标)。
    ///
    /// 两条发丝沿 +Y、在 Z 向间隔 `separation` 放置；发根固定（pin_root=true，
    /// 模拟头皮附着——真实毛发发根不可动），其余节点自由。
    fn two_parallel_strands(
        separation: f32,
        min_distance: f32,
    ) -> (HairBundle, StrandCollider, Vec<usize>, Vec<usize>) {
        let spec = StrandSpec {
            segments_per_span: 1,
            stretch_compliance: 0.0,
            bend_compliance: 0.0,
            pin_root: true, // 发根钉住（头皮附着）；自由段由碰撞分离
        };
        let mut b = HairBundle::new(Vec3::splat(0.0));
        // 每条 6 点折线（段长 0.2，共 5 段），两条互相接近。
        let nodes_a: Vec<Vec3> =
            (0..6).map(|i| Vec3::new(0.0, i as f32 * 0.2, 0.0)).collect();
        let nodes_b: Vec<Vec3> = (0..6)
            .map(|i| Vec3::new(0.0, i as f32 * 0.2, separation))
            .collect();
        let sa = HairStrand::from_nodes(nodes_a, spec).unwrap();
        let sb = HairStrand::from_nodes(nodes_b, spec).unwrap();
        b.push_strand(&sa).unwrap();
        b.push_strand(&sb).unwrap();
        let (start_a, segs_a) = b.strand_starts()[0];
        let (start_b, segs_b) = b.strand_starts()[1];
        let idx_a: Vec<usize> = (0..=segs_a).map(|k| start_a + k).collect();
        let idx_b: Vec<usize> = (0..=segs_b).map(|k| start_b + k).collect();
        let mut col = StrandCollider::new(min_distance, 1.0).unwrap();
        col.register_strand(&idx_a).unwrap();
        col.register_strand(&idx_b).unwrap();
        (b, col, idx_a, idx_b)
    }

    #[test]
    fn step_with_collision_matches_plain_step_without_contacts() {
        // 无接触时：step_with_collision 与 step 逐位等价（子步展开不改内核语义）。
        // 自证非恒真：若子步展开或 h 计算写错（如 h=dt 而非 dt/substeps），
        // 有 / 无碰撞两路的粒子轨迹会分叉。
        let build = || {
            let mut b = HairBundle::new(Vec3::new(0.0, -9.81, 0.0));
            let c = straight_curve(3, 0.3);
            let s = HairStrand::from_curve(&c, StrandSpec::default_for_hair()).unwrap();
            b.push_strand(&s).unwrap();
            b
        };
        let mut a = build();
        let mut b = build();
        // min_distance 极小（无接触）+ 远离球体。
        let col = StrandCollider::new(1e-4, 1.0).unwrap();
        let dt = 1.0 / 60.0;
        let ra = {
            let mut v = Vec::new();
            for _ in 0..20 {
                v.push(a.step(dt, 4, 8).unwrap());
            }
            v
        };
        let rb = {
            let mut v = Vec::new();
            for _ in 0..20 {
                v.push(b.step_with_collision(dt, 4, 8, 4, &col).unwrap());
            }
            v
        };
        let bits = |bb: &HairBundle| -> Vec<[u32; 3]> {
            bb.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "无接触时两路位置应逐位一致");
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(
                x.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                y.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                "无接触时残余序列应逐位一致"
            );
        }
    }

    #[test]
    fn step_with_collision_is_deterministic() {
        // 硬性要求 A：带碰撞的完整求解循环，同初始状态两遍逐位一致。
        let build = || {
            let (b, col, ia, ib) = two_parallel_strands(0.05, 0.1);
            (b, col, ia, ib)
        };
        let run = || -> Vec<[u32; 3]> {
            let (mut b, col, _ia, _ib) = build();
            for _ in 0..12 {
                b.step_with_collision(1.0 / 60.0, 2, 4, 4, &col).unwrap();
            }
            b.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "带碰撞求解循环两遍应逐位一致");
    }

    #[test]
    fn collision_keeps_strands_separated_in_full_loop() {
        // 物理断言：把碰撞接入求解循环后，两条互相接近的平行发丝的**自由段**
        // （非发根）应保持间距 ≥ min_distance（容差内），且不过度（≤ 1.1×）。
        // 自证非恒真：若碰撞未接入循环，自由段间距会收缩到 < min_distance。
        // 注：发根 pin（inv_mass=0）按头皮附着语义不可动，故只检查自由节点。
        let (mut b, col, idx_a, idx_b) = two_parallel_strands(0.03, 0.1);
        // 无外力；初始重叠（间距 0.03 < min_distance 0.1）+ 约束投影耦合驱动碰撞。
        for _ in 0..30 {
            b.step_with_collision(1.0 / 60.0, 2, 4, 1, &col).unwrap();
            assert!(b.stabilize(), "求解应保持有限");
        }
        let ps = b.world().particles();
        assert_eq!(idx_a.len(), idx_b.len());
        let mut min_sep = f32::INFINITY;
        for k in 1..idx_a.len() {
            let pa = ps[idx_a[k]].position;
            let pb = ps[idx_b[k]].position;
            let d = (pb - pa).length();
            if d < min_sep {
                min_sep = d;
            }
            assert!(
                d >= 0.1 - 1e-3,
                "自由节点 {k} 间距应 ≥ min_distance：{d}"
            );
        }
        assert!(min_sep.is_finite() && min_sep >= 0.1 - 1e-3);
        // 对比基线：不接碰撞时自由节点仍保持初始重叠（0.03 < min_distance）。
        // 这证明"分离"确实由碰撞接入循环带来（自证非恒真）。
        let (mut base, _col2, _, _) = two_parallel_strands(0.03, 0.1);
        for _ in 0..30 {
            base.step(1.0 / 60.0, 2, 4).unwrap();
        }
        let bps = base.world().particles();
        let d_base = (bps[idx_b[1]].position - bps[idx_a[1]].position).length();
        assert!(
            d_base < 0.06,
            "无碰撞基线应仍重叠（< 0.06）：{d_base}（证明分离来自碰撞）"
        );
    }

    #[test]
    fn step_with_collision_does_not_inject_energy() {
        // 物理断言（两部分）：
        // (1) **精确不变量**：碰撞位置修正**不触碰速度**——求解前后每个自由粒子
        //     速度逐位不变 ⇒ 动能逐位不变。这是碰撞"不注入能量"的最强可证形式。
        //     自证非恒真：若碰撞误写速度/增益，任一分量 to_bits 会变。
        // (2) **有界性**：把碰撞接入求解循环后，长时程动能不爆炸（有限且不发散）。
        //     注意：发根 pin 是外部"电机"（钉住不动），与运动碰撞相互作用会做功
        //     ——严格能量守恒在含固定锚的系统中不成立，故此处只断言**有界**
        //     （真实的能量界由 (1) 的精确不变量 + 无外力时约束耗散保证）。
        let (mut b, col, ia, ib) = two_parallel_strands(0.03, 0.1);
        // 给自由粒子一个非零初速。
        {
            let ps = b.world_mut().particles_mut();
            for p in ps.iter_mut() {
                if p.inv_mass > 0.0 {
                    p.velocity = Vec3::new(0.3, 0.1, -0.2);
                }
            }
        }
        let vel_before: Vec<[u32; 3]> = b
            .world()
            .particles()
            .iter()
            .map(|p| [p.velocity.x.to_bits(), p.velocity.y.to_bits(), p.velocity.z.to_bits()])
            .collect();
        let ke_before = b.kinetic_energy();
        // 只做碰撞解算（不经求解循环）：速度/动能必须逐位不变。
        col.resolve_strand_contacts(b.world_mut().particles_mut());
        col.resolve_sphere_contacts(b.world_mut().particles_mut());
        let vel_after: Vec<[u32; 3]> = b
            .world()
            .particles()
            .iter()
            .map(|p| [p.velocity.x.to_bits(), p.velocity.y.to_bits(), p.velocity.z.to_bits()])
            .collect();
        assert_eq!(vel_before, vel_after, "碰撞位置修正不应改写任何速度（逐位）");
        assert_eq!(
            ke_before.to_bits(),
            b.kinetic_energy().to_bits(),
            "碰撞不改速度 ⇒ 动能应逐位不变"
        );
        // (2) 循环有界性：长时程不发散（初态深度重叠 ⇒ 校正期能量瞬态上冲，
        //     但随后收敛——不是发散）。断言：全程有限 + 峰值有界 + 末态回落。
        let (_b2, col2, _, _) = two_parallel_strands(0.03, 0.1);
        let mut e_peak = 0.0f32;
        let mut e_last = 0.0f32;
        for _ in 0..160 {
            b.step_with_collision(1.0 / 60.0, 2, 4, 1, &col2).unwrap();
            assert!(b.stabilize(), "求解应保持有限");
            let e = b.kinetic_energy();
            assert!(e.is_finite(), "动能应有限：{e}");
            if e > e_peak {
                e_peak = e;
            }
            e_last = e;
        }
        // 峰值有界（抓发散：无外力系统若自激发散，峰值会呈指数增长；本场景
        // 含「发根 pin（外部锚，会做功）+ 深度重叠校正」的固有瞬态上冲，
        // 但**有界且不指数增长**）。
        assert!(
            e_peak < 100.0,
            "长时程动能峰值应有界（不发散）：peak={e_peak} ke0={ke_before}"
        );
        // 末态回落到峰值以下（瞬态衰减，非持续泵浦）。
        assert!(
            e_last < e_peak,
            "碰撞循环动能应瞬态衰减（末态 < 峰值）：peak={e_peak} last={e_last}"
        );
        let _ = (ia, ib);
    }

    #[test]
    fn velocity_contact_removes_approaching_component_only() {
        // 速度级接触：接近的相对法向速度被消除为 0（非接近），且**只减不增**
        // （切向分量不动；分离中的相对速度不被改写）。
        // 自证非恒真：若符号写反（把 vn 当正号），接近分量会被**放大**成 2×。
        let (mut w, idx) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.05), // 距离 0.05 < min_distance 0.1（接触）
            Vec3::new(0.0, 1.0, 0.05),
        ]);
        // 初始相对速度：b 朝 a 运动（接近，沿 −Z）；切向给 +X（应保留）。
        {
            let ps = w.particles_mut();
            ps[idx[0]].velocity = Vec3::new(1.0, 0.0, 0.0);
            ps[idx[1]].velocity = Vec3::new(1.0, 0.0, 0.0);
            ps[idx[2]].velocity = Vec3::new(1.0, 0.0, -2.0);
            ps[idx[3]].velocity = Vec3::new(1.0, 0.0, -2.0);
        }
        let mut col = StrandCollider::new(0.1, 1.0).unwrap();
        col.register_strand(&[idx[0], idx[1]]).unwrap();
        col.register_strand(&[idx[2], idx[3]]).unwrap();
        let handled = col.resolve_strand_velocity_contacts(w.particles_mut());
        assert!(handled > 0, "应有接触对被速度修正");
        // 分离方向 n ≈ +Z；相对法向速度 vn = (v_b − v_a)·n。
        let ps = w.particles();
        let va = ps[idx[0]].velocity;
        let vb = ps[idx[2]].velocity;
        let vn = vb.z - va.z; // n ≈ +Z
        assert!(vn.abs() < 1e-4, "接近法向速度应被消除为 ~0：vn={vn}");
        // 切向（+X）分量应保留（无摩擦，切向不动）。
        assert!(
            (ps[idx[0]].velocity.x - 1.0).abs() < 1e-5,
            "切向速度不应被改写：{}",
            ps[idx[0]].velocity.x
        );
        // 动能不增（速度级仅消除接近分量）。
        // 分离中的情形：相对速度已远离 → 不改写（逐位）。
        let (mut w2, idx2) = world_with_points(&[
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.05),
            Vec3::new(0.0, 1.0, 0.05),
        ]);
        {
            let ps = w2.particles_mut();
            ps[idx2[0]].velocity = Vec3::new(0.0, 0.0, -1.0);
            ps[idx2[2]].velocity = Vec3::new(0.0, 0.0, 1.0); // 远离
        }
        let before = [
            w2.particles()[idx2[0]].velocity.z.to_bits(),
            w2.particles()[idx2[2]].velocity.z.to_bits(),
        ];
        col.resolve_strand_velocity_contacts(w2.particles_mut());
        let after = [
            w2.particles()[idx2[0]].velocity.z.to_bits(),
            w2.particles()[idx2[2]].velocity.z.to_bits(),
        ];
        assert_eq!(before, after, "分离中的相对速度不应被改写（逐位）");
    }
}

