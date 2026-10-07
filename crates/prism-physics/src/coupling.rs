//! 力场与耦合器（内容批）。
//!
//! 语义权威：`ForceField` trait：噪声场/吸引子/风/重力，布料的
//! 风、粒子的力场、SPH 的外力**共用同一族**；D-28（有状态模拟经 EvalCtx
//! 快照库解决——快照键 = (子图签名, 帧)，签名含通道定义哈希；LRU 逐出只影响
//! 重算不影响结果）、D-16（Mini Code：力场是纯函数，耦合器是纯装配）。
//!
//! # 设计（纯函数化外力 + 快照复用 session.rs 机制）
//!
//! - [`ForceField`]：`accel(pos, t) -> Vec3` 的纯函数——风/重力/吸引子/湍流
//!   皆此一形。**无隐藏状态**（确定性红线：无全局 RNG；湍流用 (pos, t) 的
//!   哈希派生，不用随机流）。
//! - [`Coupler`]：把多个力场装配成合外力，逐粒子求值。耦合（粒子推布料、
//!  流体冲刚体）在 XPBD 里天然免费——本模块提供"跨域共享
//!   力场"的节点族落地。
//! - [`SimSession`] 复用：[`CoupledSession`] 把 [`crate::joints`]/
//!   [`crate::cloth`] 的初态与力场装配进 [`crate::session`] 的快照机制
//!   （键 = (子图签名, 帧)，D-28）——seek 等价与 LRU 无关性由 session 层保证。

use prism_math::Vec3;

use crate::session::{SimSession, SimSpec};
use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 物理量上限（单位约定）
// ---------------------------------------------------------------------------

/// 力场加速度幅值的物理上限（|a| ≤ MAX_ACCEL，单位 = 世界单位/秒²；引擎世界
/// 单位即米，标准重力 g₀ = 9.80665 m/s²，见下面取值依据）。
///
/// # 取值依据（非凭感觉）
///
/// 引擎物理域以 SI 米-秒为世界单位（全部既有测试与规格用 g = 9.81 量级重力、
/// 米级尺度几何）。耦合器/积分器（XPBD 显式积分）的数值稳定要求加速度 × 时间步
/// 的位移增量不炸穿一个世界单位尺度。取 g₀ 的 **100 倍 ≈ 981**，向上取整到
/// **1000.0**：
///
/// - 正常内容量级（重力 9.81、风/湍流个位数、吸引子在米级构型下 ~1e0–1e2）
///   全部 ≪ 1000 → 限幅**逐位不生效**（Mini Code，正常量级结果不变）；
/// - 极端构型（q 饱和到 f32::MAX、|d| 非零）→ 加速度不再达到 ~1e8–3.4e38 的
///   天文数字，而是被夹到 1000，使下游显式积分器（a·dt²，dt ~ 1/60）单步位移
///   增量 ≤ ~0.28 世界单位，量级可控，杜绝数值崩坏；
/// - 1000 = 约 102·g₀，物理上仍属"强风/爆炸冲击"可接受量级，不产生非物理假象。
///
/// 不用舍入/超越函数（D-29 无 libm），仅常量比较与 f32 夹取。
pub const MAX_ACCEL: f32 = 1000.0;

// ---------------------------------------------------------------------------
// 力场
// ---------------------------------------------------------------------------

/// 力场：位置的纯函数加速度（风 / 重力 / 吸引子 / 湍流统一形态）。
///
/// 求值是纯函数（同 (pos, t) 必同结果）——这是"无全局 RNG、模拟状态纯函数化"
/// 红线的力场侧落地。实现者不得引入内部可变状态。
pub trait ForceField {
    /// 在某位置 `pos`、时刻 `t` 处的加速度。
    fn accel(&self, pos: Vec3, t: f32) -> Vec3;
}

/// 常量重力场。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gravity {
    /// 加速度向量。
    pub g: Vec3,
}

impl ForceField for Gravity {
    fn accel(&self, _pos: Vec3, _t: f32) -> Vec3 {
        self.g
    }
}

/// 均匀风场（沿方向 dir 的常量风，强度 = dir 的长度）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wind {
    /// 风向与强度。
    pub dir: Vec3,
}

impl ForceField for Wind {
    fn accel(&self, _pos: Vec3, _t: f32) -> Vec3 {
        self.dir
    }
}

/// 点吸引子（朝 center 的平方反比吸引力；含软化半径 eps 防奇点）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attractor {
    /// 引力中心。
    pub center: Vec3,
    /// 强度（> 0 吸引，< 0 排斥）。
    pub strength: f32,
    /// 软化半径（避免 r → 0 的奇点）。
    pub eps: f32,
}

impl Attractor {
    /// 校验参数合法性（C18 防呆；P3xxx 结构错误码段）。
    ///
    /// 拒绝：`strength`/`eps` 为 NaN 或 ±Inf（非法规格主动拒绝，不留 NaN/Inf
    /// 透传下游——clamp 不拦 NaN）；`eps` 为负数（软化半径半径语义非负）。
    /// 校验通过后 [`ForceField::accel`] 输出**恒有限**（有限 strength/eps + 饱和
    /// 钳制 + 加速度上限三重保证）。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        // 用 `is_finite()` 一次拦 NaN 与 ±Inf（clamp 对 NaN 是恒等映射，无法兜底）。
        if !self.strength.is_finite() {
            return Err(PhysicsError {
                code: "P3008",
                location: "Attractor::validate（strength）".to_string(),
                problem: format!(
                    "strength = {}：强度必须为有限实数（NaN/±Inf 会透传为非有限加速度）",
                    self.strength
                ),
                fix: "传有限强度（> 0 吸引，< 0 排斥）；非法量请在构造期拒绝".to_string(),
            });
        }
        if !self.eps.is_finite() {
            return Err(PhysicsError {
                code: "P3008",
                location: "Attractor::validate（eps）".to_string(),
                problem: format!(
                    "eps = {}：软化半径必须为有限实数（NaN/±Inf 会使 r² 退化/溢出）",
                    self.eps
                ),
                fix: "传有限非负软化半径（eps = 0 合法，由 MIN_POSITIVE 兜底防奇点）".to_string(),
            });
        }
        if !(self.eps >= 0.0) {
            return Err(PhysicsError {
                code: "P3008",
                location: "Attractor::validate（eps）".to_string(),
                problem: format!("eps = {}：软化半径必须 ≥ 0（半径语义非负）", self.eps),
                fix: "传非负软化半径；不需软化传 0".to_string(),
            });
        }
        Ok(())
    }
}

impl ForceField for Attractor {
    fn accel(&self, pos: Vec3, _t: f32) -> Vec3 {
        let d = self.center - pos;
        // 软化：r² = |d|² + eps²，并夹到极小下限——当 eps = 0 且粒子恰在中心
        // 时 r² = 0 → strength/0 = Inf → 0·Inf = NaN。夹下限保证分母恒 > 0。
        let r2 = (d.dot(&d) + self.eps * self.eps).max(f32::MIN_POSITIVE);
        // 标量系数 q = strength / r²。r² 夹到 MIN_POSITIVE 后，强量级 strength
        // （如 1e30）除以 MIN_POSITIVE（≈1.18e-38）仍可溢出为 ±Inf；再乘 d 于
        // d=0 处得 0·Inf = NaN、于 d≠0 处得 ±Inf，均违约"输出永不 NaN/Inf"。
        // 故对 q 做饱和钳制到 ±f32::MAX（有限），再乘 d——三通道恒有限。
        // 物理语义保持：正常量级下 strength/r² ≪ f32::MAX，钳制不生效，结果不变
        // （Mini Code，最小改动）。不用 libm（D-29），仅 core 浮点运算。
        let q = (self.strength / r2).clamp(-f32::MAX, f32::MAX);
        // 加速度上限（风险 1 清除）：q 饱和到 f32::MAX 后 |a| 理论可达
        // f32::MAX·|d|；|d| 小但非零（如 1e-30）→ ~3.4e8 天文数字。虽有限，
        // 下游显式积分器（a·dt²）可致数值崩坏。故对输出三分量逐项夹到
        // ±MAX_ACCEL。正常量级（|a| ≪ 1000）下逐位不生效，结果不变。
        let a = d * q;
        Vec3::new(
            a.x.clamp(-MAX_ACCEL, MAX_ACCEL),
            a.y.clamp(-MAX_ACCEL, MAX_ACCEL),
            a.z.clamp(-MAX_ACCEL, MAX_ACCEL),
        )
    }
}

/// 确定性湍流场：(pos, t) 的哈希派生流——**无随机状态**，纯函数。
///
/// 用整数混淆从 (量化 pos, 量化 t) 派生 3D 方向（splitmix64 风格），再乘强度。
/// 时间量化使 t 连续时场也连续（相邻帧近似）。完全确定：同 (pos,t) 同结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turbulence {
    /// 强度（加速度幅度）。
    pub strength: f32,
    /// 空间频率（量化尺度，越大越细碎）。
    pub freq: f32,
    /// 时间频率。
    pub time_freq: f32,
}

impl Turbulence {
    #[inline]
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn rand_signed(&self, a: i64, b: i64, c: i64) -> f32 {
        let key = (a as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).rotate_left(17)
            ^ ((c as u64).wrapping_mul(0x1656_67B1_9E37_79F9)).rotate_left(31);
        let r = Self::mix(key);
        // 高 24 位 → [−1, 1]。
        ((r >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
    }
}

impl ForceField for Turbulence {
    fn accel(&self, pos: Vec3, t: f32) -> Vec3 {
        // 空间量化（freq 越大网格越密）+ 时间量化。
        let qx = (pos.x * self.freq).floor() as i64;
        let qy = (pos.y * self.freq).floor() as i64;
        let qz = (pos.z * self.freq).floor() as i64;
        let qt = (t * self.time_freq).floor() as i64;
        Vec3::new(
            self.rand_signed(qx, qy, qz ^ qt),
            self.rand_signed(qy, qz ^ qt, qx),
            self.rand_signed(qz ^ qt, qx, qy),
        ) * self.strength
    }
}

// ---------------------------------------------------------------------------
// 耦合器
// ---------------------------------------------------------------------------

/// 耦合器：多力场叠合成合外力，逐粒子求值（跨域共享力场）。
pub struct Coupler {
    fields: Vec<Box<dyn ForceField>>,
}

impl std::fmt::Debug for Coupler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coupler").field("fields", &self.fields.len()).finish()
    }
}

impl Coupler {
    /// 空耦合器。
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// 加入力场。
    pub fn add(&mut self, field: Box<dyn ForceField>) {
        self.fields.push(field);
    }

    /// 合外力加速度（Σ 各力场；纯函数）。
    pub fn accel(&self, pos: Vec3, t: f32) -> Vec3 {
        let mut a = Vec3::splat(0.0);
        for f in &self.fields {
            a = a + f.accel(pos, t);
        }
        a
    }

    /// 力场数。
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl Default for Coupler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 耦合会话（复用 SimSession 快照机制，D-28）
// ---------------------------------------------------------------------------

/// 耦合模拟会话：在 [`SimSession`] 之上以"合外力 = 常量重力 + 力场"驱动。
///
/// 落地方式（D-16 Mini Code）：力场把"每步的等效重力"折算为常量重力项——均匀
/// 力场（重力/风）本就等价于重力；非均匀力场（吸引子/湍流）在**会话构建期**按
/// 初态构型求值一次作为常量近似（本批是"耦合框架 + 快照接线"，逐点可变力场
/// 的每子步重求值留 GPU/后续轮次；此处明确标注近似边界，不静默）。
///
/// 快照键 = (子图签名, 帧) 由 [`SimSession`] 承载；seek 等价与 LRU 无关性沿用
/// session 层保证（本类型不新增缓存语义）。
pub struct CoupledSession {
    inner: SimSession,
    /// 采样力场的参考位置（会话初态质心；用于非均匀力场的常量折算）。
    sample_pos: Vec3,
    /// 已折算的等效重力（重力 + 力场在 sample_pos、t=0 的取值）。
    effective_gravity: Vec3,
}

impl std::fmt::Debug for CoupledSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoupledSession")
            .field("frame", &self.inner.frame())
            .field("effective_gravity", &self.effective_gravity)
            .finish()
    }
}

impl CoupledSession {
    /// 构造：把力场在初态质心处的取值并入重力，构建底层 [`SimSession`]。
    pub fn new(
        spec: SimSpec,
        coupler: &Coupler,
        dt: f32,
        substeps: u32,
        iterations: u32,
    ) -> Result<Self, PhysicsError> {
        // 初态质心（无粒子时取原点）。
        let n = spec.particles.len().max(1) as f32;
        let mut c = Vec3::splat(0.0);
        for p in &spec.particles {
            c = c + p.position;
        }
        let sample_pos = c / n;
        // 等效重力 = 规格重力 + 力场在 (sample_pos, t=0) 的取值。
        let effective_gravity = spec.gravity + coupler.accel(sample_pos, 0.0);
        // 集成处防御（风险 2 清除）：等效重力必须有限——任何力场产出 NaN/Inf
        // （含非法规格透传）在此结构化拒绝，杜绝非有限加速度进入积分器。
        if !effective_gravity.x.is_finite()
            || !effective_gravity.y.is_finite()
            || !effective_gravity.z.is_finite()
        {
            return Err(PhysicsError {
                code: "P3008",
                location: "CoupledSession::new（等效重力）".to_string(),
                problem: format!(
                    "等效重力 = {:?}：合外力含 NaN/±Inf（力场参数非法或规格重力非有限）",
                    effective_gravity
                ),
                fix: "检查各方参 strength/eps/gravity 为有限实数；Attractor 先调 validate()".to_string(),
            });
        }
        let mut eff = spec.clone();
        eff.gravity = effective_gravity;
        let inner = SimSession::new(eff, dt, substeps, iterations)?;
        Ok(Self { inner, sample_pos, effective_gravity })
    }

    /// 底层会话（只读）。
    pub fn session(&self) -> &SimSession {
        &self.inner
    }

    /// 参考采样位置。
    pub fn sample_pos(&self) -> Vec3 {
        self.sample_pos
    }

    /// 等效重力。
    pub fn effective_gravity(&self) -> Vec3 {
        self.effective_gravity
    }

    /// 推进/求值到 (签名, 帧)（转发 [`SimSession::advance_to`]）。
    pub fn advance_to(
        &mut self,
        sig: [u8; 32],
        frame: u64,
        store: &mut prism_core::node::SnapshotStore,
    ) -> Result<(), PhysicsError> {
        self.inner.advance_to(sig, frame, store)
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{DistanceConstraintSpec, ParticleInit};
    use prism_core::node::SnapshotStore;

    #[test]
    fn gravity_and_wind_sum_in_coupler() {
        let mut c = Coupler::new();
        c.add(Box::new(Gravity { g: Vec3::new(0.0, -9.81, 0.0) }));
        c.add(Box::new(Wind { dir: Vec3::new(1.0, 0.0, 0.0) }));
        let a = c.accel(Vec3::splat(0.0), 0.0);
        assert!((a.x - 1.0).abs() < 1e-6);
        assert!((a.y + 9.81).abs() < 1e-6);
    }

    #[test]
    fn attractor_pulls_toward_center() {
        let at = Attractor { center: Vec3::new(1.0, 0.0, 0.0), strength: 1.0, eps: 0.1 };
        let a = at.accel(Vec3::splat(0.0), 0.0);
        assert!(a.x > 0.0, "应被拉向中心（+x），实际 {a:?}");
    }

    #[test]
    fn turbulence_is_pure_function() {
        let t = Turbulence { strength: 1.0, freq: 2.0, time_freq: 3.0 };
        let p = Vec3::new(0.3, 0.7, 0.1);
        assert_eq!(t.accel(p, 0.5), t.accel(p, 0.5), "同 (pos,t) 应同结果");
    }

    #[test]
    fn coupled_session_matches_effective_gravity() {
        // 单摆链 + 重力 + 风（x 向）→ 等效重力含 x 分量。
        let spec = SimSpec {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            particles: vec![
                ParticleInit { position: Vec3::new(0.0, 2.0, 0.0), velocity: Vec3::splat(0.0), inv_mass: 0.0 },
                ParticleInit { position: Vec3::new(0.0, 1.0, 0.0), velocity: Vec3::splat(0.0), inv_mass: 1.0 },
            ],
            constraints: vec![DistanceConstraintSpec { id: 1, a: 0, b: 1, rest_length: 1.0, compliance: 0.0 }],
        };
        let mut c = Coupler::new();
        c.add(Box::new(Wind { dir: Vec3::new(2.0, 0.0, 0.0) }));
        let mut s = CoupledSession::new(spec, &c, 1.0 / 60.0, 4, 8).unwrap();
        assert!((s.effective_gravity().x - 2.0).abs() < 1e-6);
        let mut store = SnapshotStore::new();
        s.advance_to([0x11; 32], 10, &mut store).unwrap();
        // 风力把自由粒子吹向 +x。
        assert!(s.session().world().particles()[1].position.x > 0.0);
    }

    /// 耦合会话 seek 等价（D-28）：seek 第 N 帧 == 顺序推进第 N 帧（逐位）。
    #[test]
    fn coupled_session_seek_equivalence() {
        let spec = SimSpec {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            particles: vec![
                ParticleInit { position: Vec3::new(0.0, 2.0, 0.0), velocity: Vec3::splat(0.0), inv_mass: 0.0 },
                ParticleInit { position: Vec3::new(0.2, 1.5, 0.0), velocity: Vec3::splat(0.0), inv_mass: 1.0 },
                ParticleInit { position: Vec3::new(0.4, 1.0, 0.0), velocity: Vec3::new(0.3, 0.0, 0.0), inv_mass: 1.0 },
            ],
            constraints: vec![
                DistanceConstraintSpec { id: 1, a: 0, b: 1, rest_length: 0.5, compliance: 0.0 },
                DistanceConstraintSpec { id: 2, a: 1, b: 2, rest_length: 0.5, compliance: 0.0 },
            ],
        };
        let mut c = Coupler::new();
        c.add(Box::new(Gravity { g: Vec3::new(0.0, -9.81, 0.0) }));
        c.add(Box::new(Turbulence { strength: 0.5, freq: 1.0, time_freq: 1.0 }));

        let sig = [0x22; 32];
        let mut a = CoupledSession::new(spec.clone(), &c, 1.0 / 60.0, 4, 8).unwrap();
        let mut store_a = SnapshotStore::new();
        for f in 0..=12u64 {
            a.advance_to(sig, f, &mut store_a).unwrap();
        }
        let mut b = CoupledSession::new(spec, &c, 1.0 / 60.0, 4, 8).unwrap();
        let mut store_b = SnapshotStore::new();
        b.advance_to(sig, 8, &mut store_b).unwrap();
        b.advance_to(sig, 3, &mut store_b).unwrap();
        b.advance_to(sig, 12, &mut store_b).unwrap();

        let bits = |s: &CoupledSession| -> Vec<[u32; 3]> {
            s.session()
                .world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "耦合会话 seek 应等价于顺序推进（D-28）");
    }
}
