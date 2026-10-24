//! XPBD 统一求解器核心（骨架批）。
//!
//! 语义权威：（万物皆粒子+约束；compliance 刚度与迭代解耦；
//! substepping 保稳）、（压缩映射 → 收敛保证：投影求解 = 不动点迭代，
//! 子步数/迭代数显式标价）、 纪律 3（λ = 对偶变量，免费输出约束残余作
//! 质量指标）、 S3 行（子步数固定、求解序按 id 格锚）。
//!
//! # 确定性纪律（第 4 条）
//!
//! - 约束求解顺序按约束 id 升序（[`World::add_constraint`] 内部按 id 有序
//!   插入——输入序无关，同约束集必得同求解序）；
//! - 子步数 / 迭代数固定（格锚 Zᵏ），无并行歧义，无浮点三角函数；全程仅
//!   IEEE 基本运算（+ − × ÷ √），跨平台逐位一致；
//! - 同输入同输出逐位一致（测试把关）。
//!
//! # 收敛依据（的工程化）
//!
//! 每子步的 Gauss–Seidel 投影是不动点迭代。数学定位（求值层《能力对齐的
//! 数学分层》§3.1 命题 E2/E3）：约束投影 = 到凸集的投影 = **非扩张**
//! （Lip = 1）——迭代不发散，但 Banach 定理**不给出衰减率**（容差停机的
//! 误差界 `ε/(1−L)` 除零），即**无先验误差界**；实测的「残差按 O(q^iters)
//! 衰减，q < 1」来自约束雅可比的**结构性衰减率**（逐类约束可证），不是
//! 泛泛的压缩性假设——**禁止伪造 Banach 界**。故本求解器注册为 **M3
//! 模式**（[`SolveMode`]）：固定步数显式标价 + 残差记录（[`StepReport`]，
//! 事后质量指标），不含 M2 容差停机。子步长 h 越小，α̃ = α/h² 越大、单步
//! 投影越保守，即"小步比多迭代稳"。默认档（4 子步 × 8 迭代）为骨架参考
//! 标价，调用方按场景在 [`World::step`] 参数处显式给定（危险默认值禁止，
//! 宪章 C21）。含 Lipschitz > 1 分量（非凸/折叠约束）时的发散风险由
//! [`DivergenceDetector`] 事后检测（E3：连续若干步残差上升超 ε 余量才判
//! 发散，不是单步）。
//!
//! # 积分格式（长时程物理漂移）
//!
//! 求解器的**预测/速度更新**格式可切换（[`Integrator`]）：
//!
//! - [`Integrator::SemiImplicitEuler`]（**默认**）：`v += a·h; x += v·h`，
//!   位置投影后 `v = (x − x_prev)/h`。既有逐位基线，语义不变。
//! - [`Integrator::VelocityVerlet`]（辛格式族，实装）：kick-drift-kick
//!   变体，`x* = x + v·h + ½a·h²`（drift 含半加速度），投影后速度
//!   `v += a·h + Δx/h`（Δt 修正 = 位置投影对速度的反馈）。**无约束纯自由系统**
//!   下 Δx = 0，退化为教科书速度 Verlet，二阶精度、辛（symplectic）——长时程
//!   能量误差**有界不漂移**（对比半隐式欧拉一阶累积漂移；4800 步实测位置漂移
//!   小 600 余倍，见测试 `symplectic_verlet_position_drift_far_below_euler` /
//!   `symplectic_verlet_energy_drift_bounded_over_long_horizon`）。投影仍在同一套
//!   XPBD 循环内，积分格式
//!   只是预测/速度更新的重排（"辛积分器 = S3 求解器的积分格式升级"）。
//!
//! 二者共用同一套约束投影（Gauss–Seidel，按 id 序）与确定性纪律；调用方经
//! [`World::set_integrator`] 显式选择，默认恒为半隐式欧拉（零回归，其他求解
//! 器族不因本项接通而改行为）。

use prism_math::Vec3;

use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 粒子与约束
// ---------------------------------------------------------------------------

/// 粒子（XPBD"万物皆粒子"， 第 1 条；刚体/软体/布料/流体在后续阶段
/// 全部化为粒子云 + 约束族）。
///
/// 数组下标即粒子 id（格锚 Zᵏ，构建顺序即身份）；`inv_mass = 0` 表示固定
/// 粒子（w = 0，投影不动）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    /// 当前位置。
    pub position: Vec3,
    /// 子步起点位置（投影基准；每个子步的预测阶段重写，不属跨帧状态）。
    pub prev_position: Vec3,
    /// 速度。
    pub velocity: Vec3,
    /// 逆质量（0 = 固定）。
    pub inv_mass: f32,
}

impl Particle {
    /// 构造粒子（`prev_position` 由首个子步的预测阶段重写，取初值位置即可）。
    pub fn new(position: Vec3, velocity: Vec3, inv_mass: f32) -> Self {
        Self { position, prev_position: position, velocity, inv_mass }
    }
}

/// XPBD 约束接口。两个方法对应 XPBD 论文（Macklin et al. 2016, §3）的两个
/// 阶段：`evaluate` = 约束估计（estimate），`project` = 约束求解（dsolve）。
///
/// 约束族（铰链/滑槽/限位关节、PBF 密度、形状匹配、SDF 碰撞）
/// 经本 trait 接入求解循环——骨架阶段仅落 [`DistanceConstraint`]
/// 作框架验证载体（这不是布料实装，是求解器的最小可测形态）。
pub trait Constraint {
    /// 约束 id：求解序格锚键（求解按 id 升序， 第 4 条）。
    fn id(&self) -> u32;

    /// 作用粒子下标（升序；端点必须小于粒子总数，[`World::add_constraint`]
    /// 校验）。
    fn indices(&self) -> &[usize];

    /// 约束参数结构校验（C18 防呆：参数域非法显式拒绝；C20 禁静默容错）。
    ///
    /// 两处把关：[`World::add_constraint`]（约束入世界时）与 [`World::step`]
    /// （每步防御性再查——防字段被直接改写、防自定义约束族绕过构造路径）。
    /// 缺省实现 = 约束无标量参数域（恒合法）；有参数域的约束族必须覆写
    /// （如 [`DistanceConstraint`] 的 rest_length / compliance ≥ 0）。
    fn validate(&self) -> Result<(), PhysicsError> {
        Ok(())
    }

    /// 约束估计（estimate）：当前粒子构型下的约束标量 C(x)。
    fn evaluate(&self, particles: &[Particle]) -> f32;

    /// 约束求解（dsolve）：XPBD 投影一步——拉格朗日乘子 λ 更新 + 位置修正。
    ///
    /// `dt_h` 为子步长；`lambda` 为本约束的对偶变量（每子步清零重置，XPBD
    /// 论文语义）。返回修正后的 |C| 作约束残余（纪律 3：免费
    /// 质量指标）。
    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32;
}

/// 距离约束：C(p_a, p_b) = |p_a − p_b| − rest_length。
///
/// `compliance` = 1/stiffness（柔度，单位 m/N）：0 = 完全刚性；XPBD 刚度与
/// 迭代次数解耦的关键参数（第 2 条）。
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceConstraint {
    id: u32,
    /// 作用粒子对（升序规范化）。
    pair: [usize; 2],
    rest_length: f32,
    compliance: f32,
}

impl DistanceConstraint {
    /// 构造距离约束（端点对内部升序规范化，消除输入序歧义）。
    ///
    /// 参数域（rest_length / compliance ≥ 0，NaN 同拒）不在本构造器内静默
    /// 钳制或吞掉（C20）：由 [`Constraint::validate`] 在 [`World::add_constraint`]
    /// 入世界时显式校验（P1005/P1006 三段式报错），[`World::step`] 每步
    /// 防御性再查——任一环节都不许静默放行非法参数。
    pub fn new(id: u32, a: usize, b: usize, rest_length: f32, compliance: f32) -> Self {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        Self { id, pair: [a, b], rest_length, compliance }
    }
}

impl Constraint for DistanceConstraint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        &self.pair
    }

    /// 参数域校验（对齐 `SimSpec::validate` 的 P1005/P1006 语义）：负值或
    /// NaN 显式拒绝。负 compliance 会使 α̃ < 0、denom = Σw + α̃ ≤ 0，投影
    /// 走"无自由度"分支静默吞掉约束（C20 禁止）；负 rest_length 使
    /// C = |d| − rest 恒不满足，求解行为无物理意义。
    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.rest_length >= 0.0) {
            return Err(PhysicsError {
                code: "P1005",
                location: format!("DistanceConstraint（约束 {}）", self.id),
                problem: format!(
                    "rest_length = {}：静止长度必须 ≥ 0（NaN 同拒）",
                    self.rest_length
                ),
                fix: "传非负静止长度".to_string(),
            });
        }
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: format!("DistanceConstraint（约束 {}）", self.id),
                problem: format!(
                    "compliance = {}：柔度必须 ≥ 0（NaN 同拒）；负柔度会使投影分母 ≤ 0，约束被静默吞掉",
                    self.compliance
                ),
                fix: "传非负柔度；完全刚性传 0".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        let d = particles[self.pair[0]].position - particles[self.pair[1]].position;
        d.length() - self.rest_length
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        let (ia, ib) = (self.pair[0], self.pair[1]);
        let d = particles[ia].position - particles[ib].position;
        let len = d.length();
        if len < 1e-8 {
            // 奇异构型（重合粒子）：投影方向 n = d/len 无定义，跳过修正。
            // 残余如实返回（不静默钳制，宪章 C20；与 prism-math normalize
            // 的退化保护同款纪律）。
            return self.evaluate(particles).abs();
        }
        let c = len - self.rest_length;
        // ∇_a C = n，∇_b C = −n（n 为 a−b 方向单位向量）。
        let n = d / len;
        let w_sum = particles[ia].inv_mass + particles[ib].inv_mass;
        // α̃ = α/h²（XPBD 论文 §3：compliance 随子步长重标定）。
        let alpha_tilde = self.compliance / (dt_h * dt_h);
        let denom = w_sum + alpha_tilde;
        if denom <= 0.0 {
            // 双固定粒子（w_sum = 0）且零柔度：无自由度，约束不可作用。
            return c.abs();
        }
        // Δλ = (−C − α̃λ) / (Σw + α̃)；λ 累积（论文 §3.3）。
        let dlambda = (-c - alpha_tilde * *lambda) / denom;
        *lambda += dlambda;
        // 位置修正：Δp_i = w_i ∇_i C Δλ。
        let wa = particles[ia].inv_mass;
        let wb = particles[ib].inv_mass;
        particles[ia].position = particles[ia].position + n * (wa * dlambda);
        particles[ib].position = particles[ib].position - n * (wb * dlambda);
        // 修正后残余。
        self.evaluate(particles).abs()
    }
}

// ---------------------------------------------------------------------------
// 世界与求解循环
// ---------------------------------------------------------------------------

/// 积分格式（求解器的预测/速度更新重排）。
///
/// 两者共用同一套约束投影（Gauss–Seidel，按 id 序），仅**预测**与**速度
/// 更新**两步的数学形式不同。默认 [`Integrator::SemiImplicitEuler`] 保持
/// 既有逐位基线；[`Integrator::VelocityVerlet`] 为辛格式族（速度 Verlet），
/// 用于长时程模拟抑制能量漂移。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Integrator {
    /// 半隐式（辛）欧拉：`v += a·h; x += v·h; 投影; v = (x − x_prev)/h`。
    ///
    /// 一阶精度。默认值——既有全部求解器族（布料/SPH/关节/破碎/耦合）的
    /// 逐位基线由此格式定义，切换默认值会动渲染行为（宪章 C15/C21）。
    #[default]
    SemiImplicitEuler,
    /// 速度 Verlet（kick-drift-kick，辛、二阶）：`x* = x + v·h + ½a·h²`；
    /// 投影后 `v += a·h + Δx/h`（Δx = x − x*，位置投影修正）。
    ///
    /// 无约束纯自由系统下 Δx = 0，退化为教科书速度 Verlet，位置/速度精确、
    /// 长时程能量误差有界（不随步数单调漂移）。本实装格式。
    VelocityVerlet,
}

/// 一步求解报告：每子步每轮投影完成后的最大约束残余 |C|（收敛曲线；
/// 纪律 3 的免费质量指标）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepReport {
    /// 长度 = substeps × iterations；顺序 = 子步外层 × 迭代内层（固定）。
    pub residuals: Vec<f32>,
}

// ---------------------------------------------------------------------------
// 推进-停机模式声明（M3）与发散检测（E3）
// ---------------------------------------------------------------------------
//
// W3：类型本体已上提为内核一等属性（`prism_core::fixpoint`，与环推进实例
// 共用单一实现——模式元数据收拢进签名、发散检测单一判据）。此处按原路径
// 重导出，physics 侧 API / 语义 / 错误码零变化。

pub use prism_core::fixpoint::{
    DivergenceDetector, SolveMode, SolveModeMeta, DIVERGENCE_WINDOW_DEFAULT,
    RESIDUAL_EPS_DEFAULT,
};

/// XPBD 世界：粒子集 + 约束集 + 常量重力 + 求解循环。
///
/// 确定性不变式：`constraints` 恒按约束 id 升序存放（[`World::add_constraint`]
/// 二分有序插入）——构建输入序不影响求解序与结果（格锚纪律）。
pub struct World {
    particles: Vec<Particle>,
    constraints: Vec<Box<dyn Constraint>>,
    /// 与 `constraints` 平行的 λ 数组（每子步清零，XPBD 论文语义）。
    lambdas: Vec<f32>,
    gravity: Vec3,
    /// 积分格式（预测/速度更新重排；默认半隐式欧拉）。
    integrator: Integrator,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // dyn Constraint 不实现 Debug：只报告规模概要（确定性状态经由
        // 粒子切片与 StepReport 观察，不依赖诊断输出）。
        f.debug_struct("World")
            .field("particles", &self.particles.len())
            .field("constraints", &self.constraints.len())
            .field("gravity", &self.gravity)
            .field("integrator", &self.integrator)
            .finish()
    }
}

impl World {
    /// 构造世界（重力常量外力；后续阶段换 ForceField 族——）。
    ///
    /// 积分格式默认 [`Integrator::SemiImplicitEuler`]（既有逐位基线）；
    /// 长时程场景经 [`World::set_integrator`] 显式切换为辛格式。
    pub fn new(gravity: Vec3) -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
            lambdas: Vec::new(),
            gravity,
            integrator: Integrator::default(),
        }
    }

    /// 设置积分格式（默认半隐式欧拉，切换即改渲染行为——宪章 C15）。
    ///
    /// 非跨帧隐藏状态：格式是求解器配置，不参与快照（状态 = 位置 + 速度，
    /// [`World::set_state`] 语义不变）。
    pub fn set_integrator(&mut self, integrator: Integrator) {
        self.integrator = integrator;
    }

    /// 当前积分格式。
    pub fn integrator(&self) -> Integrator {
        self.integrator
    }

    /// 加入粒子，返回其下标（= 粒子 id）。
    pub fn add_particle(&mut self, position: Vec3, velocity: Vec3, inv_mass: f32) -> usize {
        self.particles.push(Particle::new(position, velocity, inv_mass));
        self.particles.len() - 1
    }

    /// 加入约束。内部按约束 id 升序二分插入（求解序格锚，输入序无关）；
    /// 端点越界为结构错误、参数域非法（rest_length / compliance < 0 或 NaN）
    /// 为防呆错误，均三段式报错（宪章 C18），非法约束不入世界（无部分状态）。
    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) -> Result<(), PhysicsError> {
        let n = self.particles.len();
        let indices = constraint.indices().to_vec();
        if indices.is_empty() || indices.iter().any(|&i| i >= n) {
            return Err(PhysicsError {
                code: "P3006",
                location: "World::add_constraint".to_string(),
                problem: format!(
                    "约束 {} 引用越界粒子下标（约束端点 {:?}，当前粒子数 {n}）",
                    constraint.id(),
                    indices
                ),
                fix: "先 add_particle 建齐粒子，再加入引用其下标的约束".to_string(),
            });
        }
        // 参数域校验（World 直接构造路径的防呆：与 SimSpec::validate 的
        // P1005/P1006 同标准，C18/C20）。
        constraint.validate()?;
        let pos = self
            .constraints
            .partition_point(|existing| existing.id() <= constraint.id());
        self.constraints.insert(pos, constraint);
        self.lambdas.insert(pos, 0.0);
        Ok(())
    }

    /// 粒子切片（下标即 id）。
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// 粒子切片（可变；供上层约束族组织代码做构型初始化——求解循环仍只在
    /// [`World::step`] 内投影，不暴露约束/λ 内部状态）。
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    /// 约束条数。
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// 用外部状态覆写粒子 pos/vel（SimSession 快照装配路径）。
    ///
    /// 状态完备性：`prev_position` 由下一子步预测阶段重写，跨帧状态 = 位置
    /// + 速度，故二者即完备。长度不符返回 `false`（调用方按快照未命中处理，
    /// 绝不部分覆写——C20）。
    pub fn set_state(&mut self, positions: &[Vec3], velocities: &[Vec3]) -> bool {
        if positions.len() != self.particles.len() || velocities.len() != self.particles.len() {
            return false;
        }
        for (p, (&pos, &vel)) in self.particles.iter_mut().zip(positions.iter().zip(velocities)) {
            p.position = pos;
            p.velocity = vel;
            p.prev_position = pos;
        }
        true
    }

    /// 推进一步 `dt`（固定步长）：`substeps` 个子步 × `iterations` 轮投影。
    ///
    /// 每子步（XPBD 论文 §3）：①半隐式欧拉预测；②按 id 序 Gauss–Seidel
    /// 投影（λ 每子步清零）；③速度更新 v = (x − x_prev)/h。
    pub fn step(
        &mut self,
        dt: f32,
        substeps: u32,
        iterations: u32,
    ) -> Result<StepReport, PhysicsError> {
        // 防呆校验（C18/C21：显式报错，不给静默兜底）。
        if !(dt > 0.0) {
            return Err(PhysicsError {
                code: "P1002",
                location: "World::step".to_string(),
                problem: format!("dt = {dt}：步长必须为正（NaN 同样被拒）"),
                fix: "传正步长，如 dt = 1.0 / fps".to_string(),
            });
        }
        if substeps == 0 {
            return Err(PhysicsError {
                code: "P1003",
                location: "World::step".to_string(),
                problem: "substeps = 0：子步数必须 ≥1（子步数固定是确定性纪律）".to_string(),
                fix: "传 substeps ≥ 1，参考标价 4（小步比多迭代稳）".to_string(),
            });
        }
        if iterations == 0 {
            return Err(PhysicsError {
                code: "P1004",
                location: "World::step".to_string(),
                problem: "iterations = 0：迭代数必须 ≥1（迭代数固定是确定性纪律）".to_string(),
                fix: "传 iterations ≥ 1，参考标价 8（刚度与迭代解耦，多迭代只降残差不改刚度）"
                    .to_string(),
            });
        }
        // 防御性再查（C20：构造期验过 ≠ 永远合法——字段可能被直接改写，
        // 自定义约束族也可能携带非法参数域；每步显式校验，绝不静默假设）。
        for constraint in &self.constraints {
            constraint.validate()?;
        }

        let h = dt / substeps as f32;
        let gravity_h = self.gravity * h;
        let half_gravity_h = self.gravity * (0.5 * h);
        let mut report = StepReport::default();
        for _ in 0..substeps {
            // ① 预测（固定粒子 inv_mass = 0：prev 同步、位置不动，两格式一致）。
            //
            // - 半隐式欧拉：`v += a·h; x += v·h`；`prev_position` 存子步起点 x_n。
            // - 速度 Verlet（辛）：`x* = x + v·h + ½a·h²`（drift 含半加速度）；
            //   `prev_position` 存**投影前预测位置 x***，供速度更新取投影修正
            //   Δx = x_{n+1} − x*（见 ③）。
            match self.integrator {
                Integrator::SemiImplicitEuler => {
                    for p in &mut self.particles {
                        if p.inv_mass > 0.0 {
                            p.velocity = p.velocity + gravity_h;
                            p.prev_position = p.position;
                            p.position = p.position + p.velocity * h;
                        } else {
                            p.prev_position = p.position;
                        }
                    }
                }
                Integrator::VelocityVerlet => {
                    for p in &mut self.particles {
                        if p.inv_mass > 0.0 {
                            p.prev_position = p.position + p.velocity * h + half_gravity_h * h;
                            p.position = p.prev_position;
                        } else {
                            p.prev_position = p.position;
                        }
                    }
                }
            }
            // ② 投影：λ 每子步重置；每轮按 id 升序遍历（格锚求解序）。
            for l in &mut self.lambdas {
                *l = 0.0;
            }
            for _ in 0..iterations {
                for (ci, constraint) in self.constraints.iter().enumerate() {
                    let lambda = &mut self.lambdas[ci];
                    constraint.project(&mut self.particles, h, lambda);
                }
                // 迭代后最大约束残余（收敛曲线采样点）。
                let mut max_residual = 0.0f32;
                for constraint in &self.constraints {
                    let r = constraint.evaluate(&self.particles).abs();
                    if r > max_residual {
                        max_residual = r;
                    }
                }
                report.residuals.push(max_residual);
            }
            // ③ 速度更新。
            //
            // - 半隐式欧拉：`v = (x − x_prev)/h`（XPBD 论文式；x_prev 为子步起点）。
            // - 速度 Verlet（辛）：`v += a·h + Δx/h`，Δx = x_{n+1} − x*（投影
            //   修正）。写成**增量形式**而非全量 `(x − x_prev)/h`——关键数值理由：
            //   无约束时 Δx = 0，速度仅作 `v += a·h` 累加，避开 `(x − x_prev)/h`
            //   中"大数 x 减小量 (v·h)"的灾难性抵消（f32 下长时间积分的位置误差
            //   主源）。既保留投影修正对速度的反馈（XPBD 语义），又使自由飞行
            //   精确。固定粒子速度恒 0。
            let inv_h = 1.0 / h;
            match self.integrator {
                Integrator::SemiImplicitEuler => {
                    for p in &mut self.particles {
                        p.velocity = (p.position - p.prev_position) * inv_h;
                    }
                }
                Integrator::VelocityVerlet => {
                    for p in &mut self.particles {
                        if p.inv_mass > 0.0 {
                            p.velocity =
                                p.velocity + gravity_h + (p.position - p.prev_position) * inv_h;
                        } else {
                            p.velocity = Vec3::splat(0.0);
                        }
                    }
                }
            }
        }
        Ok(report)
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 常量签名风格的重力（测试场景统一向下）。
    fn gravity() -> Vec3 {
        Vec3::new(0.0, -9.81, 0.0)
    }

    /// 关卡 1：粒子对从拉伸态恢复到静止长（误差 < 1e-4），迭代数收敛曲线
    /// 可复现（残余序列逐位一致 + 显著下降）。
    #[test]
    fn distance_constraint_recovers_rest_length_with_convergence_trace() {
        // 三粒子链：p0 固定于原点，p1/p2 自由，初始拉伸 20%（间距 1.2），
        // rest = 1.0。可满足稳态：p1 → 1.0、p2 → 2.0（零柔度全恢复）。
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world.add_particle(Vec3::new(2.4, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        for (id, (a, b)) in [(1u32, (0usize, 1usize)), (2, (1, 2))] {
            world
                .add_constraint(Box::new(DistanceConstraint::new(id, a, b, 1.0, 0.0)))
                .unwrap();
        }

        let dt = 1.0 / 60.0;
        let (substeps, iterations) = (4u32, 8u32);
        let report = world.step(dt, substeps, iterations).unwrap();

        // 收敛判定：最终约束残余 < 1e-4（任务关卡 1 阈值）。
        let final_residual = report.residuals[report.residuals.len() - 1];
        assert!(
            final_residual < 1e-4,
            "最终残余 {final_residual} 应 < 1e-4"
        );
        // 收敛曲线形状：首值显著大于末值（迭代确实在收敛，非平台巧合）。
        let first_residual = report.residuals[0];
        assert!(
            first_residual > final_residual * 10.0,
            "首残余 {first_residual} 应显著大于末残余 {final_residual}（收敛曲线应有下降）"
        );
        // 末轮残余稳定在阈值内（不动点邻域：速度传播使残差在 1e-6 级微振，
        // 不作逐位相同的过强断言，只断言收敛判定成立）。
        let last = report.residuals.len();
        assert!(report.residuals[last - 2] < 1e-4 && report.residuals[last - 1] < 1e-4);

        // 恢复到静止长：三粒子间距。
        let ps = world.particles();
        let gap1 = (ps[1].position - ps[0].position).length();
        let gap2 = (ps[2].position - ps[1].position).length();
        assert!((gap1 - 1.0).abs() < 1e-4, "gap1 = {gap1} 应恢复到 rest 1.0");
        assert!((gap2 - 1.0).abs() < 1e-4, "gap2 = {gap2} 应恢复到 rest 1.0");

        // 收敛曲线可复现：同构型重建再解，残余序列逐位一致。
        let mut world2 = World::new(Vec3::splat(0.0));
        world2.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world2.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world2.add_particle(Vec3::new(2.4, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world2
            .add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, 0.0)))
            .unwrap();
        world2
            .add_constraint(Box::new(DistanceConstraint::new(2, 1, 2, 1.0, 0.0)))
            .unwrap();
        let report2 = world2.step(dt, substeps, iterations).unwrap();
        assert_eq!(
            report.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
            report2.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
            "收敛曲线应逐位可复现"
        );

        println!(
            "收敛曲线（首 8 点 + 末点）：{:?} … {}（substeps={substeps}, iterations={iterations}）",
            &report.residuals[..8.min(report.residuals.len())],
            final_residual
        );
    }

    /// 关卡 2：刚度解耦——同迭代数下不同 compliance 行为单调合理；且刚度
    /// 不随迭代数改变（XPBD 特性验证， 第 2 条）。
    #[test]
    fn stiffness_decoupled_from_iterations_monotonic() {
        // 单摆：p0 固定，p1 挂在正下方（rest 1.0），重力把约束拉长。
        // 稳态拉伸随 compliance 单调增（柔度越大越软、垂得越长）。
        let stretch_at = |compliance: f32, iterations: u32| -> f32 {
            let mut world = World::new(gravity());
            world.add_particle(Vec3::new(0.0, 1.0, 0.0), Vec3::splat(0.0), 0.0);
            world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
            world
                .add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, compliance)))
                .unwrap();
            let dt = 1.0 / 60.0;
            for _ in 0..90 {
                world.step(dt, 8, iterations).unwrap();
            }
            // 稳态拉伸 = |p1 − p0| − rest。
            (world.particles()[1].position - world.particles()[0].position).length() - 1.0
        };

        let rigid = stretch_at(0.0, 8);
        let soft = stretch_at(0.1, 8);
        let softer = stretch_at(2.0, 8);

        // 刚性（α=0）：近零拉伸。
        assert!(rigid.abs() < 1e-3, "α=0 应近刚性，拉伸 {rigid}");
        // 柔度单调：0 < stretch(0.1) < stretch(2.0)。
        assert!(soft > 0.0, "α=0.1 应有可见拉伸（重力把软约束拉长）");
        assert!(soft < softer, "柔度单调：stretch(0.1)={soft} < stretch(2.0)={softer}");

        // 解耦验证：同 compliance、不同迭代数 → 稳态拉伸几乎相同
        // （多迭代只降低残差，不改变刚度——PBD 顽疾的 XPBD 解法）。
        let iters4 = stretch_at(0.1, 4);
        let iters16 = stretch_at(0.1, 16);
        assert!(
            (iters4 - iters16).abs() < 1e-3,
            "稳态拉伸不应依赖迭代数：iters4={iters4} vs iters16={iters16}"
        );

        println!("刚度解耦：rigid={rigid}, soft(0.1)={soft}, softer(2.0)={softer}, iters4={iters4}, iters16={iters16}");
    }

    /// 关卡 3a：同初始状态求解两遍逐位一致（位置位模式 + 残余序列位模式）。
    #[test]
    fn bitwise_determinism_repeat() {
        let build = || {
            let mut world = World::new(gravity());
            world.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
            world.add_particle(Vec3::new(0.5, 1.5, 0.0), Vec3::splat(0.0), 1.0);
            world.add_particle(Vec3::new(1.0, 1.0, 0.1), Vec3::splat(0.0), 1.0);
            world.add_particle(Vec3::new(1.5, 0.5, 0.0), Vec3::new(0.1, 0.0, 0.0), 1.0);
            // 约束按 id 0..4（链式）。
            for id in 0..3u32 {
                world
                    .add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        id as usize,
                        id as usize + 1,
                        0.707_1,
                        0.01,
                    )))
                    .unwrap();
            }
            world
        };

        let mut a = build();
        let mut b = build();
        let dt = 1.0 / 60.0;
        let mut reports_a = Vec::new();
        let mut reports_b = Vec::new();
        for _ in 0..10 {
            reports_a.push(a.step(dt, 4, 8).unwrap());
            reports_b.push(b.step(dt, 4, 8).unwrap());
        }

        let bits_a: Vec<[u32; 3]> = a
            .particles()
            .iter()
            .map(|p| {
                [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()]
            })
            .collect();
        let bits_b: Vec<[u32; 3]> = b
            .particles()
            .iter()
            .map(|p| {
                [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()]
            })
            .collect();
        assert_eq!(bits_a, bits_b, "两遍求解粒子位置应逐位一致");
        for (ra, rb) in reports_a.iter().zip(&reports_b) {
            assert_eq!(
                ra.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                rb.residuals.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
                "残余序列应逐位一致"
            );
        }
    }

    /// 关卡 3b：约束按 id 排序——打乱构建输入序，结果逐位不变。
    #[test]
    fn constraint_order_shuffled_input_identical() {
        let dt = 1.0 / 60.0;

        // 顺序构建：约束 id 0,1,2,3。
        let mut ordered = World::new(gravity());
        ordered.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
        for i in 1..5 {
            ordered.add_particle(Vec3::new(0.3 * i as f32, 2.0 - 0.3 * i as f32, 0.0),
                Vec3::splat(0.0), 1.0);
        }
        for id in 0..4u32 {
            ordered
                .add_constraint(Box::new(DistanceConstraint::new(
                    id,
                    id as usize,
                    id as usize + 1,
                    0.424_3,
                    0.005,
                )))
                .unwrap();
        }

        // 乱序构建：同约束集以 id 3,1,0,2 插入（World 内部按 id 有序化）。
        let mut shuffled = World::new(gravity());
        shuffled.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
        for i in 1..5 {
            shuffled.add_particle(Vec3::new(0.3 * i as f32, 2.0 - 0.3 * i as f32, 0.0),
                Vec3::splat(0.0), 1.0);
        }
        for id in [3u32, 1, 0, 2] {
            shuffled
                .add_constraint(Box::new(DistanceConstraint::new(
                    id,
                    id as usize,
                    id as usize + 1,
                    0.424_3,
                    0.005,
                )))
                .unwrap();
        }

        for _ in 0..12 {
            ordered.step(dt, 4, 8).unwrap();
            shuffled.step(dt, 4, 8).unwrap();
        }

        let bits = |w: &World| -> Vec<[u32; 3]> {
            w.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&ordered), bits(&shuffled), "打乱约束输入序结果应逐位不变");
    }

    /// 固定粒子（inv_mass = 0）在重力与约束作用下位置逐位不动。
    #[test]
    fn fixed_particle_stays_put() {
        let mut world = World::new(gravity());
        let anchor = world.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
        let free = world.add_particle(Vec3::new(0.6, 1.6, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(DistanceConstraint::new(1, anchor, free, 0.8, 0.0)))
            .unwrap();

        let dt = 1.0 / 60.0;
        for _ in 0..20 {
            world.step(dt, 4, 8).unwrap();
        }
        let p = &world.particles()[anchor];
        assert_eq!(p.position.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(p.position.y.to_bits(), 2.0f32.to_bits());
        assert_eq!(p.position.z.to_bits(), 0.0f32.to_bits());
    }

    /// 退化构型：重合粒子对跳过投影（不 panic、不产生 NaN）。
    #[test]
    fn degenerate_coincident_particles_skip_projection() {
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, 0.0)))
            .unwrap();
        world.step(1.0 / 60.0, 2, 4).unwrap();
        for p in world.particles() {
            assert!(p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite());
        }
    }

    /// step 防呆：dt/substeps/iterations 非法 → 三段式报错（含建议段，C19）。
    #[test]
    fn step_rejects_bad_params_with_advice() {
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);

        for (params, code) in [
            ((0.0, 4, 8), "P1002"),
            ((-1.0 / 60.0, 4, 8), "P1002"),
            ((1.0 / 60.0, 0, 8), "P1003"),
            ((1.0 / 60.0, 4, 0), "P1004"),
        ] {
            let (dt, substeps, iterations) = params;
            let err = world.step(dt, substeps, iterations).unwrap_err();
            assert_eq!(err.code, code, "{params:?}");
            assert!(err.to_string().contains("建议："), "必须含建议段：{err}");
        }
        // NaN 步长同样被拒（!(dt > 0) 涵盖 NaN）。
        let err = world.step(f32::NAN, 4, 8).unwrap_err();
        assert_eq!(err.code, "P1002");
    }

    /// add_constraint 防呆：端点越界 → P3006 三段式。
    #[test]
    fn add_constraint_out_of_bounds_is_structured_error() {
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
        let err = world
            .add_constraint(Box::new(DistanceConstraint::new(1, 0, 5, 1.0, 0.0)))
            .unwrap_err();
        assert_eq!(err.code, "P3006");
        assert!(err.problem.contains("越界"), "{}", err.problem);
        assert!(err.fix.contains("add_particle"), "{}", err.fix);
    }

    // -----------------------------------------------------------------------
    // 长时程物理漂移（辛积分器）
    // -----------------------------------------------------------------------

    /// 无约束系统的机械能 E = Σ[½m|v|² − m·g·x]（m = 1/inv_mass；势能
    /// U = −m·g·x，因为力 F = m·g = −∇U）。自由粒子（常量重力、无耗散）
    /// 物理上严格守恒——是检验积分器保能量的干净标尺。
    fn total_energy(world: &World) -> f32 {
        let g = world.gravity;
        let mut e = 0.0f32;
        for p in world.particles() {
            if p.inv_mass > 0.0 {
                let m = 1.0 / p.inv_mass;
                e += 0.5 * m * p.velocity.dot(&p.velocity) - m * g.dot(&p.position);
            }
        }
        e
    }

    /// 自由抛体世界：单粒子，初位置/初速度给定，无约束，重力向下。
    fn projectile(gravity: Vec3, pos: Vec3, vel: Vec3) -> World {
        let mut w = World::new(gravity);
        w.add_particle(pos, vel, 1.0);
        w
    }

    /// 对比关卡（对比断言 + 长时程）：固定物理时段（有界位移，避 f32 大数
    /// 抵消）下用**大量积分步**（长时程 4800 步）对比位置漂移——辛格式（速度
    /// Verlet）相对解析解的位置误差**远小于**半隐式欧拉。
    ///
    /// 数学：常量重力解析解 x(t) = x₀ + v₀t + ½g·t²。半隐式欧拉每步位置多算
    /// ½a·h²，累积 O(h·t) 系统漂移；速度 Verlet 的 drift 含半加速度 + 速度
    /// 增量补另半，常加速度下位置**精确**（误差仅 f32 增量累加舍入）。
    #[test]
    fn symplectic_verlet_position_drift_far_below_euler() {
        let gravity = Vec3::new(0.0, -1.0, 0.0);
        let pos0 = Vec3::new(0.0, 0.0, 0.0);
        let vel0 = Vec3::new(0.0, 1.0, 0.0);
        let dt = 1.0 / 2400.0;
        let steps = 4_800u32; // t = 2.0 s（一个完整上抛时段，位移有界）

        // 解析解位置（t = steps·dt）。
        let t = dt * steps as f32;
        let exact = pos0 + vel0 * t + gravity * (0.5 * t * t);

        let drift = |integrator: Integrator| -> f32 {
            let mut w = projectile(gravity, pos0, vel0);
            w.set_integrator(integrator);
            for _ in 0..steps {
                w.step(dt, 1, 1).unwrap();
            }
            (w.particles()[0].position - exact).length()
        };

        let drift_euler = drift(Integrator::SemiImplicitEuler);
        let drift_verlet = drift(Integrator::VelocityVerlet);

        // 辛格式位置漂移应比欧拉小一个数量级以上。
        assert!(
            drift_verlet * 10.0 < drift_euler,
            "速度 Verlet 位置漂移 {drift_verlet:e} 应远小于半隐式欧拉 {drift_euler:e}"
        );
        // 辛格式漂移本身应接近 f32 舍入量级（守恒质量高）。
        assert!(
            drift_verlet < 1e-4,
            "速度 Verlet 位置漂移应 < 1e-4（近解析精确），实际 {drift_verlet:e}"
        );
        println!(
            "长时程位置漂移（{steps} 步 / t={t}s）：欧拉 {drift_euler:e} vs Verlet {drift_verlet:e}（比 {:.0}×）",
            drift_euler / drift_verlet.max(f32::MIN_POSITIVE)
        );
    }

    /// 守恒界关卡：辛格式能量漂移**有界不增长**——把长时程切成前后
    /// 两半，后半段漂移与前半段同量级（无累积趋势）。
    ///
    /// 诚实边界：常量重力下加速度恒定，半隐式欧拉同属辛族、能量误差亦有界
    /// （故此处不对欧拉断言"单调发散"，只断言辛格式的守恒界——这是积分器
    /// 本身的性质，与对照解耦）。参数取良态量级（能量项 O(10)、位移有限，
    /// 避免大数相减掩盖积分器差异）。
    #[test]
    fn symplectic_verlet_energy_drift_bounded_over_long_horizon() {
        let gravity = Vec3::new(0.0, -0.5, 0.0);
        let pos0 = Vec3::new(0.0, 0.0, 0.0);
        let vel0 = Vec3::new(0.2, 0.5, 0.0);
        let dt = 1.0 / 2400.0;
        let half = 2_400u32;

        // 返回（前半能量漂移幅度，后半能量漂移幅度），相对初能量。
        let split = |integrator: Integrator| -> (f32, f32) {
            let mut w = projectile(gravity, pos0, vel0);
            w.set_integrator(integrator);
            let e0 = total_energy(&w);
            for _ in 0..half {
                w.step(dt, 1, 1).unwrap();
            }
            let e_mid = total_energy(&w);
            for _ in 0..half {
                w.step(dt, 1, 1).unwrap();
            }
            let e_end = total_energy(&w);
            (((e_mid - e0) / e0).abs(), ((e_end - e0) / e0).abs())
        };

        let (v_first, v_second) = split(Integrator::VelocityVerlet);

        // 辛格式：后段漂移不显著大于前段（有界，无累积趋势）。
        assert!(
            v_second <= v_first * 1.5 + 1e-5,
            "辛格式能量漂移应有界：前段 {v_first:e} → 后段 {v_second:e}"
        );
        // 辛格式漂移本身处于小量级。
        assert!(v_second < 1e-4, "辛格式能量相对漂移应 < 1e-4，实际 {v_second:e}");
        // 对比：辛格式能量漂移应不大于欧拉（实测小 2~3 个数量级）。
        let (e_first, e_second) = split(Integrator::SemiImplicitEuler);
        assert!(
            v_second < e_second,
            "辛格式能量漂移 {v_second:e} 应小于半隐式欧拉 {e_second:e}"
        );
        println!(
            "漂移有界（各 {half} 步）：Verlet {v_first:e}→{v_second:e}；欧拉 {e_first:e}→{e_second:e}"
        );
    }

    /// 确定性关卡：辛格式同输入两遍推进逐位一致（位置 + 速度
    /// 位模式），延续求解器的全链路确定性纪律。
    #[test]
    fn velocity_verlet_bitwise_deterministic() {
        let build = || {
            let mut w = World::new(gravity());
            w.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
            w.add_particle(Vec3::new(0.3, 1.6, 0.1), Vec3::new(0.2, 0.0, 0.0), 1.0);
            w.add_particle(Vec3::new(0.6, 1.2, 0.0), Vec3::new(-0.1, 0.05, 0.0), 1.0);
            for id in 0..2u32 {
                w.add_constraint(Box::new(DistanceConstraint::new(
                    id,
                    id as usize,
                    id as usize + 1,
                    0.5,
                    0.01,
                )))
                .unwrap();
            }
            w.set_integrator(Integrator::VelocityVerlet);
            w
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..30 {
            a.step(1.0 / 60.0, 4, 8).unwrap();
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let bits = |w: &World| -> Vec<[u32; 6]> {
            w.particles()
                .iter()
                .map(|p| {
                    [
                        p.position.x.to_bits(),
                        p.position.y.to_bits(),
                        p.position.z.to_bits(),
                        p.velocity.x.to_bits(),
                        p.velocity.y.to_bits(),
                        p.velocity.z.to_bits(),
                    ]
                })
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "辛格式两遍求解应逐位一致");
    }

    /// 有约束仍收敛关卡：辛格式接入同一套 XPBD 投影，约束残余仍
    /// 收敛到阈值内——积分格式升级不动投影语义。
    #[test]
    fn velocity_verlet_with_constraints_still_converges() {
        let mut world = World::new(Vec3::splat(0.0));
        world.set_integrator(Integrator::VelocityVerlet);
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world.add_particle(Vec3::new(2.4, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, 0.0)))
            .unwrap();
        world
            .add_constraint(Box::new(DistanceConstraint::new(2, 1, 2, 1.0, 0.0)))
            .unwrap();

        let report = world.step(1.0 / 60.0, 4, 8).unwrap();
        let final_residual = report.residuals[report.residuals.len() - 1];
        assert!(
            final_residual < 1e-4,
            "辛格式下约束残余仍应收敛到 < 1e-4，实际 {final_residual}"
        );
        let ps = world.particles();
        let gap1 = (ps[1].position - ps[0].position).length();
        assert!((gap1 - 1.0).abs() < 1e-3, "辛格式投影后间距应恢复 rest，实际 {gap1}");
    }

    /// 默认格式不变：`World::new` 后 integrator = SemiImplicitEuler，且
    /// 欧拉格式下同输入逐位复现（零回归护栏——辛格式为可选升级）。
    #[test]
    fn default_integrator_is_semi_implicit_euler() {
        let w = World::new(gravity());
        assert_eq!(w.integrator(), Integrator::SemiImplicitEuler, "默认必须保持欧拉（零回归）");

        // 显式设回欧拉与默认逐位一致。
        let run = |set: bool| -> Vec<[u32; 3]> {
            let mut w = World::new(gravity());
            if set {
                w.set_integrator(Integrator::SemiImplicitEuler);
            }
            w.add_particle(Vec3::new(0.5, 1.5, 0.0), Vec3::new(0.1, 0.0, 0.0), 1.0);
            w.add_particle(Vec3::new(0.5, 0.5, 0.0), Vec3::splat(0.0), 1.0);
            w.add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, 0.0)))
                .unwrap();
            for _ in 0..20 {
                w.step(1.0 / 60.0, 4, 8).unwrap();
            }
            w.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(run(false), run(true), "显式设欧拉应与默认逐位一致");
    }

    // -----------------------------------------------------------------------
    // M3 模式声明与发散检测（求值层 E2/E3 的物理侧落地）
    // -----------------------------------------------------------------------

    /// M3 显式化：模式元数据声明固定步数 + (ε, n) 缓存键分量；无 M2 变体
    /// （非扩张框架无先验误差界，容差停机即伪造 Banach 界）。
    #[test]
    fn solve_mode_metadata_declares_m3_with_cache_key_params() {
        let meta = SolveModeMeta::new(4, 8, RESIDUAL_EPS_DEFAULT);
        assert_eq!(meta.mode, SolveMode::FixedIterations);
        assert_eq!(meta.mode, SolveMode::default(), "M3 是唯一（默认）模式");
        assert_eq!((meta.substeps, meta.iterations), (4, 8));
        assert_eq!(meta.residual_eps, 1e-4);
        // 模式参数即缓存键分量（§3.2 S3）：改迭代数 ⇒ 元数据变 ⇒ 键变。
        assert_ne!(meta, SolveModeMeta::new(4, 16, RESIDUAL_EPS_DEFAULT));
        assert_ne!(meta, SolveModeMeta::new(4, 8, 1e-6));
    }

    /// E3：单步上升 + ε 余量内抖动不判发散（连续窗口未达 / 超不过余量）。
    #[test]
    fn divergence_detector_ignores_single_steps_and_epsilon_jitter() {
        let mut d = DivergenceDetector::new(3, 1e-3);
        assert!(!d.observe(1.0));
        assert!(!d.observe(1.5), "单步尖峰不是发散（E3：非单步）");
        assert!(!d.observe(0.9), "回落 → streak 归零");
        assert!(!d.observe(1.1));
        assert!(!d.diverged(), "尖峰 + 回落不应判发散");

        // ε 余量内的缓慢抖动：每步 +1e-4 < margin 1e-3 → 平台抖动，不累计。
        let mut j = DivergenceDetector::new(3, 1e-3);
        for k in 0..10 {
            let r = 1.0 + (k as f32) * 1e-4;
            assert!(!j.observe(r), "ε 余量内抖动不应累计：step {k}");
        }
        assert_eq!(j.streak(), 0);
    }

    /// E3：连续 window 步上升超 ε 余量才判发散；含「Lip = 1 + O(ε)」
    /// 缓慢上升的边界情形（每步 +2·margin 被抓，每步 +0.5·margin 不抓）。
    #[test]
    fn divergence_detector_flags_consecutive_rise_above_margin() {
        let mut d = DivergenceDetector::new(3, 1e-3);
        assert!(!d.observe(1.0));
        assert!(!d.observe(1.01), "streak 1");
        assert!(!d.observe(1.02), "streak 2");
        assert!(d.observe(1.03), "连续 3 步上升超余量应判发散");
        assert!(d.diverged());
        assert_eq!(d.streak(), 3);
        // 趋势判定非锁存：回落后解除。
        assert!(!d.observe(0.5));
        assert!(!d.diverged());

        // 缓慢上升（E3 边界情形）：每步 +2·margin。
        let slow: Vec<f32> = (0..6).map(|k| 1.0 + 2e-3 * k as f32).collect();
        assert!(DivergenceDetector::scan(&slow, 3, 1e-3), "缓慢上升应判发散");
        // 亚余量爬行：每步 +0.5·margin → 永不判。
        let crawl: Vec<f32> = (0..6).map(|k| 1.0 + 5e-4 * k as f32).collect();
        assert!(!DivergenceDetector::scan(&crawl, 3, 1e-3), "亚余量爬行不应判发散");
    }

    /// 真实收敛曲线（StepReport 残差序列）不得被判发散——检测器不误伤
    /// 正常求解（子步边界的单步回升 < 连续窗口）。
    #[test]
    fn divergence_detector_clears_real_convergence_trace() {
        // 关卡 1 同构型：三粒子链、零柔度、4 子步 × 8 迭代。
        let mut world = World::new(Vec3::splat(0.0));
        world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
        world.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world.add_particle(Vec3::new(2.4, 0.0, 0.0), Vec3::splat(0.0), 1.0);
        world
            .add_constraint(Box::new(DistanceConstraint::new(1, 0, 1, 1.0, 0.0)))
            .unwrap();
        world
            .add_constraint(Box::new(DistanceConstraint::new(2, 1, 2, 1.0, 0.0)))
            .unwrap();
        let report = world.step(1.0 / 60.0, 4, 8).unwrap();
        assert_eq!(report.residuals.len(), 32, "4 子步 × 8 迭代 = 32 个采样点");
        assert!(
            !DivergenceDetector::scan(
                &report.residuals,
                DIVERGENCE_WINDOW_DEFAULT,
                RESIDUAL_EPS_DEFAULT
            ),
            "真实收敛曲线不应被判发散（防误伤）"
        );
    }
}
