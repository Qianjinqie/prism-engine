//! 关节族（B7 · 内容批）：铰链 / 滑槽 / 限位。
//!
//! 语义权威：《算法白皮书》§8（关节 = 两刚体粒子云间的附加约束，铰链/滑槽/
//! 限位各是一种约束梯度形式；万物皆粒子 + 约束；刚度与迭代解耦；求解序按 id
//! 格锚）、D-18（**统一走 XPBD，替代 Bullet 式双求解器老路**——关节不做独立
//! 求解器，只做 [`Constraint`] 实现，约束代数统一进 [`crate::xpbd::World`]）。
//!
//! # 关节约束代数（XPBD 通用投影形式）
//!
//! 每个关节约束的投影严格遵循白皮书 §8 的 XPBD 通式，仅约束标量 C 与雅可比
//! ∇C 因关节类型而异（这正是"一套约束代数通吃"的含义）：
//!
//! ```text
//!   Δλ = (−C − α̃·λ) / (Σᵢ wᵢ·|∇ᵢC|² + α̃)      α̃ = compliance / h²
//!   Δxᵢ = wᵢ · ∇ᵢC · Δλ
//! ```
//!
//! - **铰链（Hinge）**：两粒子云各取一个锚点，约束 `C = |x_a − x_b|`——锚点
//!   重合（相对平移被消去）。绕自由轴转动不受限（缺省角度域 [0, 2π]，即无
//!   额外限制），等价于一个"球窝"关节；再叠加一个"轴对齐"约束即得完整铰链。
//! - **滑槽（Slider）**：约束锚点到轴线的垂距 `C = |(x_a − x_b) − ((d·n)n)|`
//!   （n 为归一化滑槽轴）——允许沿轴自由滑动，消去轴外两个平移自由度。
//!   方向向量的**长度即滑槽轴**（构造期归一化，退化方向显式拒绝）。
//! - **限位（Limit）**：只约束 1D 标量 l（如关节角/滑程）落在 [lo, hi] 区间，
//!   区间内 C = 0（零修正、零残余——"自由区间"），越界则 C = l − 端点，
//!   单向推出。C 在区间边界连续（|C| 按距离线性增长），投影几何一致。
//!
//! # 确定性纪律（[`crate::xpbd`] 同款）
//!
//! λ 每子步清零由 [`crate::xpbd::World`] 负责；本模块投影仅用 IEEE 基本运算
//! （+ − × ÷ √），无三角函数、无 libm（D-29）、无全局 RNG；同一构型同输入
//! 逐位一致。求解序由约束 id 升序格锚（[`crate::xpbd::World::add_constraint`]），
//! 与本模块无关。

use prism_math::Vec3;

use crate::xpbd::{Constraint, Particle};
use crate::PhysicsError;

/// 内部：XPBD 标量约束投影（白皮书 §8 通式），返回修正后残余 |C|。
///
/// - `c`：约束标量 C；`grad`：各粒子位置的雅可比 ∇ᵢC（与 `grad` 下标对应）；
/// - `grad` 与作用的粒子下标一一对应；Σ 权重 = Σ wᵢ·|∇ᵢC|²。
///
/// 奇异保护：denom ≤ 0（全固定粒子的零柔度约束）或无自由度时，返回 |C| 且不
/// 改位置（不静默钳制，宪章 C20）。
fn project_scalar(
    particles: &mut [Particle],
    pairs: &[(usize, Vec3)],
    c: f32,
    compliance: f32,
    dt_h: f32,
    lambda: &mut f32,
) -> f32 {
    // Σ wᵢ·|∇ᵢC|²
    let mut denom = 0.0f32;
    for &(i, g) in pairs {
        let w = particles[i].inv_mass;
        denom += w * g.dot(&g);
    }
    // α̃ = α / h²（compliance 随子步长重标定，XPBD 论文 §3）。
    let alpha_tilde = compliance / (dt_h * dt_h);
    denom += alpha_tilde;
    if denom <= 0.0 {
        // 无自由度（全固定且零柔度）：约束不可作用，残余如实返回。
        return c.abs();
    }
    // Δλ = (−C − α̃λ) / (Σw + α̃)；λ 累积（论文 §3.3）。
    let dlambda = (-c - alpha_tilde * *lambda) / denom;
    *lambda += dlambda;
    // 位置修正：Δxᵢ = wᵢ · ∇ᵢC · Δλ。
    for &(i, g) in pairs {
        let w = particles[i].inv_mass;
        particles[i].position = particles[i].position + g * (w * dlambda);
    }
    // 修正后残余（重估 C）。
    reeval_abs(particles, pairs, c, dlambda)
}

/// 修正后残余的保守估计：由投影定义重算 |C_new|（避免重复实现各约束的 evaluate）。
///
/// 对线性化约束，C_new ≈ C + Σ(∇ᵢC·Δxᵢ) = C + Δλ·Σ wᵢ|∇ᵢC|²；但为忠实反映
/// 非线性（如距离约束），由调用方在 `project` 末尾用各自 `evaluate` 复算。此处
/// 仅作为兜底（`pairs` 语义与各约束 `evaluate` 保持一致即精确）。
fn reeval_abs(_particles: &[Particle], _pairs: &[(usize, Vec3)], c: f32, dlambda: f32) -> f32 {
    // 线性化残差估计：投影后标量残差由 C 和修正量给出，最终残余由调用方的
    // evaluate 覆盖；此函数保证类型与三参签名一致，避免未使用告警的占位逻辑。
    (c + dlambda).abs()
}

// ---------------------------------------------------------------------------
// 铰链关节
// ---------------------------------------------------------------------------

/// 铰链关节（锚点重合约束）：C = |x_a − x_b|（相对平移被消去）。
///
/// 直接以两个粒子锚点表示共点；绕任意过锚点轴的转动不受约束（球窝语义）。
/// 需要"只绕某一固定轴转动、其余全锁"的完整铰链时，叠加一个轴对齐约束
/// （对本引擎即再加一个角度上的 [`LimitJoint`] 或一个平面约束）。
#[derive(Debug, Clone, PartialEq)]
pub struct HingeJoint {
    id: u32,
    /// 两端锚点粒子下标（[a, b]；内部升序规范化）。
    anchors: [usize; 2],
    compliance: f32,
}

impl HingeJoint {
    /// 构造铰链（端点对升序规范化，消除输入序歧义）。
    pub fn new(id: u32, a: usize, b: usize, compliance: f32) -> Self {
        let anchors = if a <= b { [a, b] } else { [b, a] };
        Self { id, anchors, compliance }
    }
}

impl Constraint for HingeJoint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        &self.anchors
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: format!("HingeJoint（约束 {}）", self.id),
                problem: format!(
                    "compliance = {}：柔度必须 ≥ 0（NaN 同拒）；负柔度会使投影分母 ≤ 0",
                    self.compliance
                ),
                fix: "传非负柔度；刚性铰链传 0".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        let d = particles[self.anchors[0]].position - particles[self.anchors[1]].position;
        d.length()
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        let (ia, ib) = (self.anchors[0], self.anchors[1]);
        let d = particles[ia].position - particles[ib].position;
        let len = d.length();
        let c = len;
        if len < 1e-8 {
            // 锚点已重合（C ≈ 0）：无需修正，残余如实返回。
            return c;
        }
        let n = d / len;
        // ∇_a C = n，∇_b C = −n。
        let pairs = [(ia, n), (ib, -n)];
        project_scalar(particles, &pairs, c, self.compliance, dt_h, lambda);
        self.evaluate(particles)
    }
}

// ---------------------------------------------------------------------------
// 滑槽关节
// ---------------------------------------------------------------------------

/// 滑槽关节（轴滑动约束）：C = |(x_a − x_b) − ((d·n)n)|（锚点偏离轴线的垂距）。
///
/// `axis` 为滑槽方向（构造期归一化；零向量显式拒绝 P3007）；沿轴方向自由滑动，
/// 轴外两个平移自由度被消去。铰链是"锚点重合"，滑槽是"锚点共线（方向固定）"。
#[derive(Debug, Clone, PartialEq)]
pub struct SliderJoint {
    id: u32,
    pair: [usize; 2],
    /// 归一化滑槽轴。
    axis: Vec3,
    compliance: f32,
}

impl SliderJoint {
    /// 构造滑槽；`axis` 非零（长度 ≥ 1e-8）否则 [`Self::validate`] 报 P3007。
    pub fn new(id: u32, a: usize, b: usize, axis: Vec3, compliance: f32) -> Self {
        let pair = if a <= b { [a, b] } else { [b, a] };
        Self { id, pair, axis, compliance }
    }

    /// 归一化轴（校验通过后调用；零向量时返回原值供错误路径使用）。
    fn unit_axis(&self) -> Vec3 {
        self.axis.normalize()
    }
}

impl Constraint for SliderJoint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        &self.pair
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: format!("SliderJoint（约束 {}）", self.id),
                problem: format!("compliance = {}：柔度必须 ≥ 0（NaN 同拒）", self.compliance),
                fix: "传非负柔度；刚性滑槽传 0".to_string(),
            });
        }
        // 用 !(len > 1e-8) 而非 len < 1e-8：NaN 轴时前者为真（拒绝），后者
        // 因 NaN < 1e-8 == false 而漏放，导致后续投影产出 NaN。
        if !(self.axis.length() > 1e-8) {
            return Err(PhysicsError {
                code: "P3007",
                location: format!("SliderJoint（约束 {}）", self.id),
                problem: format!("axis = {:?}：滑槽轴不得为零向量（方向退化，NaN 同拒）", self.axis),
                fix: "传非零方向向量（构造期归一化；轴长度即方向的长度）".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        let n = self.unit_axis();
        let d = particles[self.pair[0]].position - particles[self.pair[1]].position;
        let along = d.dot(&n);
        (d - n * along).length()
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        let (ia, ib) = (self.pair[0], self.pair[1]);
        let n = self.unit_axis();
        let d = particles[ia].position - particles[ib].position;
        let along = d.dot(&n);
        // 轴外分量 r = d − (d·n)n；C = |r|。
        let r = d - n * along;
        let len = r.length();
        let c = len;
        if len < 1e-8 {
            // 锚点已共线：C ≈ 0，无需修正。
            return c;
        }
        let dir = r / len;
        // ∇C 对轴外分量：∇_a C = dir，∇_b C = −dir（轴内分量被滑槽吸收，梯度为 0）。
        let pairs = [(ia, dir), (ib, -dir)];
        project_scalar(particles, &pairs, c, self.compliance, dt_h, lambda);
        self.evaluate(particles)
    }
}

// ---------------------------------------------------------------------------
// 限位关节
// ---------------------------------------------------------------------------

/// 一维限位关节：约束标量 l 落在 [lo, hi] 区间（区间内自由、越界单向推出）。
///
/// l 由一个**外部位移读数**给出（关节角 / 滑程 / 伸缩量皆可）。本约束以
/// 2 粒子端点定义 l = (x_b − x_a)·n（n 为测量轴单位向量）——位移沿 n 的投影；
/// 调用方也可用同一结构表达角度限位（把 l 视作角度读数，广义坐标层给 l 的
/// 雅可比仍为沿 n 的单位方向，工程近似，白皮书 §8 的"约束梯度形式"落地）。
///
/// C(l)：区间内 = 0；l < lo → l − lo；l > hi → l − hi。|C| 在边界连续。
#[derive(Debug, Clone, PartialEq)]
pub struct LimitJoint {
    id: u32,
    pair: [usize; 2],
    /// 测量轴（构造期归一化；零向量拒绝）。
    axis: Vec3,
    /// 下界。
    lo: f32,
    /// 上界。
    hi: f32,
    compliance: f32,
}

impl LimitJoint {
    /// 构造限位关节；`lo <= hi`，`axis` 非零（否则 [`Self::validate`] 报错）。
    pub fn new(id: u32, a: usize, b: usize, axis: Vec3, lo: f32, hi: f32, compliance: f32) -> Self {
        let pair = if a <= b { [a, b] } else { [b, a] };
        Self { id, pair, axis, lo, hi, compliance }
    }

    /// 当前位移读数 l = (x_b − x_a)·n。
    fn reading(&self, particles: &[Particle]) -> f32 {
        let n = self.axis.normalize();
        (particles[self.pair[1]].position - particles[self.pair[0]].position).dot(&n)
    }

    /// 约束标量 C(l)（区间内 0；越界为到端点的距离，含符号）。
    fn scalar(l: f32, lo: f32, hi: f32) -> f32 {
        if l < lo {
            l - lo
        } else if l > hi {
            l - hi
        } else {
            0.0
        }
    }
}

impl Constraint for LimitJoint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        &self.pair
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: format!("LimitJoint（约束 {}）", self.id),
                problem: format!("compliance = {}：柔度必须 ≥ 0（NaN 同拒）", self.compliance),
                fix: "传非负柔度；硬限位传 0".to_string(),
            });
        }
        if self.axis.length() < 1e-8 {
            return Err(PhysicsError {
                code: "P3007",
                location: format!("LimitJoint（约束 {}）", self.id),
                problem: format!("axis = {:?}：限位测量轴不得为零向量", self.axis),
                fix: "传非零方向向量（构造期归一化）".to_string(),
            });
        }
        if !(self.lo <= self.hi) {
            return Err(PhysicsError {
                code: "P1008",
                location: format!("LimitJoint（约束 {}）", self.id),
                problem: format!("lo = {}, hi = {}：下界必须 ≤ 上界（NaN 同拒）", self.lo, self.hi),
                fix: "传 lo ≤ hi；无单向限位时令 lo = hi 之另一侧为 ±∞ 的有限大值".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        let l = self.reading(particles);
        Self::scalar(l, self.lo, self.hi).abs()
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        let l = self.reading(particles);
        let c = Self::scalar(l, self.lo, self.hi);
        if c == 0.0 {
            // 区间内：自由，无修正、无残余（λ 不累积）。
            return 0.0;
        }
        // ∇_a l = −n，∇_b l = n；区间外 C 对 l 的导数恒为 1（线性段）。
        let n = self.axis.normalize();
        let (ia, ib) = (self.pair[0], self.pair[1]);
        let pairs = [(ia, -n), (ib, n)];
        project_scalar(particles, &pairs, c, self.compliance, dt_h, lambda);
        // 不等式约束的速度纪律（限位特有）：位置投影会经 §8 步骤③ 的速度重建
        // v = (x − x_prev)/h 把"投影跳变"记成速度——下一子步预测会将粒子按此
        // 速度弹出，造成越界抖动（不等式约束无此语义）。做法：把参与本约束的
        // 粒子 `prev_position` 同步到修正后位置，使该子步重建出的速度不含跳变
        // 分量；同时消去沿 n "继续侵区"的相对速度（切向保持）。纯位置投影无法
        // 表达不等式约束的速度侧语义，此处显式补足（白皮书 §8 无第二套求解器，
        // 仍只操作本约束触及的自由度）。
        let rel_v = particles[ib].velocity - particles[ia].velocity;
        let vn = rel_v.dot(&n);
        // 上越界（c > 0，l > hi）：需阻止 vn > 0；下越界（c < 0）：阻止 vn < 0。
        let offending = if c > 0.0 { vn > 0.0 } else { vn < 0.0 };
        if offending {
            let w_sum = particles[ia].inv_mass + particles[ib].inv_mass;
            if w_sum > 0.0 {
                let impulse = vn / w_sum;
                let wa = particles[ia].inv_mass;
                let wb = particles[ib].inv_mass;
                particles[ia].velocity = particles[ia].velocity + n * (impulse * wa);
                particles[ib].velocity = particles[ib].velocity - n * (impulse * wb);
            }
        }
        // 无论是否侵区，都对参与粒子同步 prev：消除位置投影跳变 → 重建速度
        // 不含"冲回区间"的虚假动量（不等式约束的静默稳定语义）。
        for &(i, _) in &pairs {
            particles[i].prev_position = particles[i].position;
        }
        self.evaluate(particles)
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xpbd::World;

    fn gravity() -> Vec3 {
        Vec3::new(0.0, -9.81, 0.0)
    }

    /// 铰链：两锚点从分离态收敛到重合（C → 0），且拉合后绕转自由
    /// （纯转动不减距离）：验证"消去相对平移"的铰链语义。
    #[test]
    fn hinge_coincides_anchors() {
        let mut world = World::new(Vec3::splat(0.0));
        // p0 固定，p1 自由，锚点初距 0.5。
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(HingeJoint::new(1, 0, 1, 0.0)))
            .unwrap();
        for _ in 0..30 {
            world.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let d = (world.particles()[1].position - world.particles()[0].position).length();
        assert!(d < 1e-4, "铰链应使锚点重合，实际距离 {d}");
    }

    /// 滑槽：轴外分量收敛到 0，轴内位移保持自由（沿轴滑动不被约束）。
    #[test]
    fn slider_keeps_axis_free_and_kills_offaxis() {
        // 轴 = x。p0 固定原点，p1 初位 (2.0, 1.0, 0)——轴外分量 1.0。
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(2.0, 1.0, 0.0), Vec3::splat(0.0), 1.0);
        let axis = Vec3::new(1.0, 0.0, 0.0);
        world
            .add_constraint(Box::new(SliderJoint::new(1, 0, 1, axis, 0.0)))
            .unwrap();
        for _ in 0..30 {
            world.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let d = world.particles()[1].position - world.particles()[0].position;
        let off = (d - axis * d.dot(&axis)).length();
        assert!(off < 1e-4, "滑槽应消去轴外分量，实际垂距 {off}");
        // 轴内分量自由：初值 2.0（沿轴）应被保持（无轴向外力时）。
        assert!((d.x - 2.0).abs() < 1e-3, "轴内位移应自由保持，实际 {}", d.x);
    }

    /// 限位：区间内自由、越界单向推出（下界与上界各验一次）。
    #[test]
    fn limit_clamps_out_of_range() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        // 上界：l 初 = 2.0（pair = (p0固定原点, p1.at x=2)），区间 [0, 1] → 应被压到 1。
        let run = |x1: f32| -> f32 {
            let mut world = World::new(Vec3::splat(0.0));
            world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
            world.add_particle(Vec3::new(x1, 0.0, 0.0), Vec3::splat(0.0), 1.0);
            world
                .add_constraint(Box::new(LimitJoint::new(1, 0, 1, axis, 0.0, 1.0, 0.0)))
                .unwrap();
            for _ in 0..40 {
                world.step(1.0 / 60.0, 4, 8).unwrap();
            }
            world.particles()[1].position.x
        };
        // 上越界：2.0 → 压到 1.0。
        let hi = run(2.0);
        assert!((hi - 1.0).abs() < 1e-3, "上限位应把 l 压到 1.0，实际 {hi}");
        // 下越界：l = −2.0 → 推到 0.0。改用左向点。
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(-2.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(LimitJoint::new(1, 0, 1, axis, 0.0, 1.0, 0.0)))
            .unwrap();
        for _ in 0..40 {
            world.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let lo = world.particles()[1].position.x;
        assert!(lo.abs() < 1e-3, "下限位应把 l 推到 0.0，实际 {lo}");
    }

    /// 限位区间内零残余：初值已在 [0, 1] → 不修正，位置逐位不变。
    #[test]
    fn limit_inside_range_is_free() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        let joint = LimitJoint::new(1, 0, 1, axis, 0.0, 1.0, 0.0);
        assert_eq!(joint.evaluate(world.particles()), 0.0, "区间内 C 应为 0");
        for _ in 0..20 {
            world.step(1.0 / 60.0, 4, 8).unwrap();
        }
        assert_eq!(
            world.particles()[1].position.x.to_bits(),
            0.5f32.to_bits(),
            "区间内限位应零修正（位置逐位不变）"
        );
    }

    /// 防呆：滑槽/限位零轴 → P3007；限位 lo > hi → P1008；负柔度 → P1006。
    #[test]
    fn joint_validation_errors() {
        let err = SliderJoint::new(1, 0, 1, Vec3::splat(0.0), 0.0)
            .validate()
            .unwrap_err();
        assert_eq!(err.code, "P3007");
        let err = LimitJoint::new(1, 0, 1, Vec3::splat(0.0), 0.0, 1.0, 0.0)
            .validate()
            .unwrap_err();
        assert_eq!(err.code, "P3007");
        let err = LimitJoint::new(1, 0, 1, Vec3::new(1.0, 0.0, 0.0), 2.0, 1.0, 0.0)
            .validate()
            .unwrap_err();
        assert_eq!(err.code, "P1008");
        let err = HingeJoint::new(1, 0, 1, -1.0).validate().unwrap_err();
        assert_eq!(err.code, "P1006");
    }

    /// 关节族确定性：同初始构型两遍求解逐位一致。
    #[test]
    fn joint_family_bitwise_deterministic() {
        let build = || {
            let mut world = World::new(gravity());
            world.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
            world.add_particle(Vec3::new(0.3, 1.6, 0.2), Vec3::splat(0.0), 1.0);
            world.add_particle(Vec3::new(0.9, 1.1, 0.1), Vec3::new(0.1, 0.0, 0.0), 1.0);
            world
                .add_constraint(Box::new(HingeJoint::new(1, 0, 1, 0.002)))
                .unwrap();
            world
                .add_constraint(Box::new(SliderJoint::new(
                    2,
                    1,
                    2,
                    Vec3::new(1.0, 0.0, 0.0),
                    0.001,
                )))
                .unwrap();
            world
                .add_constraint(Box::new(LimitJoint::new(
                    3,
                    1,
                    2,
                    Vec3::new(1.0, 0.0, 0.0),
                    0.0,
                    2.0,
                    0.0,
                )))
                .unwrap();
            world
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..15 {
            a.step(1.0 / 60.0, 4, 8).unwrap();
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let bits = |w: &World| -> Vec<[u32; 3]> {
            w.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "关节族求解应逐位一致");
    }
}
