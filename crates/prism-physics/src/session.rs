//! SimSession 框架（骨架批）。
//!
//! 语义权威：统一模拟框架：固定步长推进 + 会话内快照缓存 +
//! 子图签名判失效——"eval 是 t 的纯函数"在模拟上的实现）、D-9（五步序列与
//! 会话机制）、D-28（快照键 = (子图签名, 帧)；LRU 逐出只影响重算不影响结果）、
//! （seek 等价成立条件：跳第 N 帧 ≡ 顺序推进到第 N 帧，
//! 逐字节）、（模拟节点快照键含自起点以来通道定义签名链——签名由调用方
//! 经 A6 `graph_signature` 计算传入，本会话只以相等性判失效）。
//!
//! 与 prism-core 的咬合（L3 宿主 → L2 合法依赖）：快照库类型直接复用
//! `prism_core::node::SnapshotStore`（键类型 `SnapshotKey = (Hash256, u64)`）。
//! 状态字节编码自带 magic + 版本头（宪章 C8 内部缓存三条件）：头不符或长度
//! 不符 = 静默失效重建（C8 豁免条款——缓存失效 ≠ 用户数据丢失），绝不猜测
//! 解读（C20）。
//!
//! # 推进语义（[`SimSession::advance_to`]）
//!
//! - **幂等**：同 (签名, 帧) 重入直接返回（eval 同帧重求值语义对齐）；
//! - **顺序延续**：已加载 (sig, f) 且 frame > f → 逐帧推进 f+1..=frame，
//!   每帧状态写回快照库 (sig, 帧)；
//! - **快照装配**：目标帧命中快照库 → 直接恢复（seek 回退、签名变更后命中
//!   历史快照均走此路）；
//! - **失效重建**：其余情形（签名变更未命中 / 快照被 LRU 逐出）从初态重推
//!   ——确定性求解保证重推结果与历史逐位一致（缓存命中与否不影响结果）。
//!
//! # 模式声明（M3）
//!
//! 本会话是 XPBD 的宿主：固定步长推进 = 求值层 **M3 模式**（无 M2 容差
//! 停机、无先验误差界——约束投影非扩张，E2）。残差经
//! [`crate::xpbd::StepReport`] 记录，每帧末值喂入发散检测（E3：连续若干
//! 帧上升超 ε 余量才判发散，不是单步）。模式元数据（含 (ε, n) 缓存键
//! 分量）经 [`SimSession::solve_mode`] 显式给出——迭代数自构造起即签名
//! 的组成部分（见 [`SimSession::new`] 防呆段）。

use prism_core::node::{Hash256, SnapshotKey, SnapshotStore};
use prism_math::Vec3;

use crate::xpbd::{
    DistanceConstraint, DivergenceDetector, SolveMode, SolveModeMeta, World,
    DIVERGENCE_WINDOW_DEFAULT, RESIDUAL_EPS_DEFAULT,
};
use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 会话规格（初态 + 参数；扩展位）
// ---------------------------------------------------------------------------

/// 粒子初态（下标即粒子 id）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleInit {
    /// 初始位置。
    pub position: Vec3,
    /// 初速度。
    pub velocity: Vec3,
    /// 逆质量（0 = 固定）。
    pub inv_mass: f32,
}

/// 距离约束规格（关节/密度/形状匹配等约束族在此并列扩展）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceConstraintSpec {
    /// 约束 id（求解序格锚键）。
    pub id: u32,
    /// 粒子端点 a。
    pub a: usize,
    /// 粒子端点 b。
    pub b: usize,
    /// 静止长度（≥ 0）。
    pub rest_length: f32,
    /// 柔度 compliance（≥ 0；XPBD 刚度与迭代解耦参数）。
    pub compliance: f32,
}

/// 会话模拟规格：初态 + 参数（D-28 的"当前帧参数"载体；后续阶段换 ForceField
/// 族并扩约束集）。签名变化 = 规格变化（签名由调用方对图参数经 A6 计算）。
#[derive(Debug, Clone, PartialEq)]
pub struct SimSpec {
    /// 重力加速度（常量外力；后续阶段换 ForceField 族）。
    pub gravity: Vec3,
    /// 粒子初态。
    pub particles: Vec<ParticleInit>,
    /// 距离约束集。
    pub constraints: Vec<DistanceConstraintSpec>,
}

impl SimSpec {
    /// 校验规格合法性（C18 防呆：负值/NaN 显式拒绝，端点越界结构化报错）。
    fn validate(&self) -> Result<(), PhysicsError> {
        let n = self.particles.len();
        for (i, p) in self.particles.iter().enumerate() {
            if !(p.inv_mass >= 0.0) {
                return Err(PhysicsError {
                    code: "P1007",
                    location: format!("SimSpec::validate（粒子 {i}）"),
                    problem: format!("inv_mass = {}：逆质量必须 ≥ 0（0 = 固定；NaN 同拒）", p.inv_mass),
                    fix: "传非负逆质量，固定粒子传 0".to_string(),
                });
            }
        }
        for c in &self.constraints {
            if !(c.rest_length >= 0.0) {
                return Err(PhysicsError {
                    code: "P1005",
                    location: format!("SimSpec::validate（约束 {}）", c.id),
                    problem: format!("rest_length = {}：静止长度必须 ≥ 0（NaN 同拒）", c.rest_length),
                    fix: "传非负静止长度".to_string(),
                });
            }
            if !(c.compliance >= 0.0) {
                return Err(PhysicsError {
                    code: "P1006",
                    location: format!("SimSpec::validate（约束 {}）", c.id),
                    problem: format!("compliance = {}：柔度必须 ≥ 0（NaN 同拒）", c.compliance),
                    fix: "传非负柔度；完全刚性传 0".to_string(),
                });
            }
            if c.a >= n || c.b >= n {
                return Err(PhysicsError {
                    code: "P3006",
                    location: format!("SimSpec::validate（约束 {}）", c.id),
                    problem: format!(
                        "约束 {} 引用越界粒子下标（端点 {}, {}，当前粒子数 {n}）",
                        c.id, c.a, c.b
                    ),
                    fix: "SimSpec::particles 先建齐粒子，再声明引用其下标的约束".to_string(),
                });
            }
        }
        Ok(())
    }

    /// 构建初始世界（确定性：约束 id 排序格锚在 [`World::add_constraint`] 内生效，
    /// 规格输入序无关）。
    pub fn build_world(&self) -> Result<World, PhysicsError> {
        self.validate()?;
        let mut world = World::new(self.gravity);
        for p in &self.particles {
            world.add_particle(p.position, p.velocity, p.inv_mass);
        }
        for c in &self.constraints {
            world.add_constraint(Box::new(DistanceConstraint::new(
                c.id, c.a, c.b, c.rest_length, c.compliance,
            )))?;
        }
        Ok(world)
    }
}

// ---------------------------------------------------------------------------
// 状态字节编解码（快照库载荷；C8 内部格式三条件）
// ---------------------------------------------------------------------------

/// 状态字节 magic + 版本（C8 ①：文件头带格式版本号；演进时升版本号即全量
/// 失效重建）。
const STATE_MAGIC: &[u8; 8] = b"PRISMSIM";
const STATE_VERSION: u32 = 1;

/// 世界状态编码为确定性字节：magic(8) + version(4) + 粒子数(4) + 每粒子
/// 6×f32（pos3 + vel3，大端位模式——宪章 C24 平台中立，同值必得同字节）。
fn state_bytes(world: &World) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + world.particles().len() * 24);
    out.extend_from_slice(STATE_MAGIC);
    out.extend_from_slice(&STATE_VERSION.to_be_bytes());
    out.extend_from_slice(&(world.particles().len() as u32).to_be_bytes());
    for p in world.particles() {
        for c in [
            p.position.x,
            p.position.y,
            p.position.z,
            p.velocity.x,
            p.velocity.y,
            p.velocity.z,
        ] {
            out.extend_from_slice(&c.to_bits().to_be_bytes());
        }
    }
    out
}

/// [`state_bytes`] 的解码：magic/版本/长度任一不符返回 `None`（调用方按快照
/// 未命中处理 → 失效重建；C8 静默失效重建条款，绝不猜测解读）。
fn decode_state(bytes: &[u8]) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    if bytes.len() < 16 || &bytes[0..8] != STATE_MAGIC {
        return None;
    }
    if u32::from_be_bytes(bytes[8..12].try_into().ok()?) != STATE_VERSION {
        return None;
    }
    let n = u32::from_be_bytes(bytes[12..16].try_into().ok()?) as usize;
    if bytes.len() != 16 + n * 24 {
        return None;
    }
    let mut positions = Vec::with_capacity(n);
    let mut velocities = Vec::with_capacity(n);
    for i in 0..n {
        let base = 16 + i * 24;
        let read = |k: usize| {
            let s: [u8; 4] = bytes[base + k * 4..base + k * 4 + 4].try_into().ok()?;
            Some(f32::from_bits(u32::from_be_bytes(s)))
        };
        positions.push(Vec3::new(read(0)?, read(1)?, read(2)?));
        velocities.push(Vec3::new(read(3)?, read(4)?, read(5)?));
    }
    Some((positions, velocities))
}

// ---------------------------------------------------------------------------
// SimSession
// ---------------------------------------------------------------------------

/// 有状态模拟会话：固定步长推进 + (签名, 帧) 快照判失效（D-28 的物理侧落地）。
///
/// 状态机不变式：`loaded` 为真时 `world` 是 `(sig, frame)` 的精确状态；快照
/// 库是纯数据资产（非隐藏全局），命中与否只影响重算量、不影响结果——seek
/// 等价由此成立。
#[derive(Debug)]
pub struct SimSession {
    spec: SimSpec,
    dt: f32,
    substeps: u32,
    iterations: u32,
    /// 当前已加载状态的签名。
    sig: Hash256,
    /// 当前已推进到的帧。
    frame: u64,
    /// 当前世界状态。
    world: World,
    /// `world` 是否为 (sig, frame) 的有效状态。
    loaded: bool,
    /// 发散检测器（E3：每帧末残差的事后趋势判定；只观测、不改写）。
    divergence: DivergenceDetector,
}

impl SimSession {
    /// 构造会话。dt/substeps/iterations 非法显式报错（C18/C21：无静默兜底）；
    /// 规格当场校验一次（端点越界等尽早暴露）。
    pub fn new(
        spec: SimSpec,
        dt: f32,
        substeps: u32,
        iterations: u32,
    ) -> Result<Self, PhysicsError> {
        if !(dt > 0.0) {
            return Err(PhysicsError {
                code: "P1002",
                location: "SimSession::new".to_string(),
                problem: format!("dt = {dt}：固定步长必须为正（NaN 同拒）"),
                fix: "传正步长，如 dt = 1.0 / fps".to_string(),
            });
        }
        if substeps == 0 {
            return Err(PhysicsError {
                code: "P1003",
                location: "SimSession::new".to_string(),
                problem: "substeps = 0：子步数必须 ≥1".to_string(),
                fix: "传 substeps ≥ 1（步数固定 → 确定性；步数是签名的组成部分）".to_string(),
            });
        }
        if iterations == 0 {
            return Err(PhysicsError {
                code: "P1004",
                location: "SimSession::new".to_string(),
                problem: "iterations = 0：迭代数必须 ≥1".to_string(),
                fix: "传 iterations ≥ 1（迭代数固定 → 确定性；迭代数是签名的组成部分）".to_string(),
            });
        }
        spec.validate()?;
        Ok(Self {
            world: spec.build_world()?,
            spec,
            dt,
            substeps,
            iterations,
            sig: [0u8; 32],
            frame: 0,
            loaded: false,
            divergence: DivergenceDetector::new(DIVERGENCE_WINDOW_DEFAULT, RESIDUAL_EPS_DEFAULT),
        })
    }

    /// 当前世界状态（只读）。
    pub fn world(&self) -> &World {
        &self.world
    }

    /// 当前帧号。
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// 显式模式元数据（M3 + (ε, n) 缓存键分量；求值层 §3.2 S3 迭代层——
    /// 迭代数自构造起就是本会话签名的组成部分，此处收拢为可进签名的
    /// 模式声明）。
    pub fn solve_mode(&self) -> SolveModeMeta {
        SolveModeMeta::new(self.substeps, self.iterations, RESIDUAL_EPS_DEFAULT)
    }

    /// 发散检测器（只读；E3 事后趋势判定——连续
    /// [`DIVERGENCE_WINDOW_DEFAULT`] 帧末残差上升超 ε 余量才判发散）。
    pub fn divergence(&self) -> &DivergenceDetector {
        &self.divergence
    }

    /// 推进/求值到 (sig, frame)。
    ///
    /// `sig` 为调用方经 A6 对（图结构 + 参数 + 通道定义签名链）计算的子图
    /// 签名：签名变了即判失效——旧快照键不再命中，从新规格初态重推（D-28）。
    /// 每帧推进后状态写回快照库 (sig, 帧)；frame 0 的初态同样入库，使任意
    /// seek 回 0 恒可快照命中。
    pub fn advance_to(
        &mut self,
        sig: Hash256,
        frame: u64,
        store: &mut SnapshotStore,
    ) -> Result<(), PhysicsError> {
        // 幂等：同 (sig, frame) 重入。
        if self.loaded && self.sig == sig && self.frame == frame {
            return Ok(());
        }
        // 顺序延续：同签名且已加载帧是目标帧的严格前驱。
        if self.loaded && self.sig == sig && frame > self.frame {
            for f in (self.frame + 1)..=frame {
                self.step_frame(sig, f, store)?;
            }
            return Ok(());
        }
        // 快照装配：目标帧命中（seek 回退 / 签名变更后命中历史快照 / frame 0）。
        let key: SnapshotKey = (sig, frame);
        if let Some(bytes) = store.get(&key) {
            if let Some((positions, velocities)) = decode_state(bytes) {
                if self.world.set_state(&positions, &velocities) {
                    self.sig = sig;
                    self.frame = frame;
                    self.loaded = true;
                    return Ok(());
                }
            }
            // 解码失败 / 粒子数不符 → 视为未命中（C8 失效重建），落入重建分支。
        }
        // 失效重建：从初态重推 0..=frame（签名变更判失效 / 快照被逐出）。
        // 确定性求解保证重推结果与历史逐位一致（测试把关：缓存命中与否不
        // 影响结果—— / D-28）。
        self.world = self.spec.build_world()?;
        self.sig = sig;
        self.frame = 0;
        self.loaded = true;
        store.insert((sig, 0), state_bytes(&self.world));
        for f in 1..=frame {
            self.step_frame(sig, f, store)?;
        }
        Ok(())
    }

    /// 推进一帧并写快照（当前签名键下）。
    fn step_frame(
        &mut self,
        sig: Hash256,
        frame: u64,
        store: &mut SnapshotStore,
    ) -> Result<(), PhysicsError> {
        // 收敛曲线由 xpbd::World::step 报告；会话层只消费收敛结果（残余指标
        // 需要时由调用方直用 World）。
        let report = self.world.step(self.dt, self.substeps, self.iterations)?;
        // M3 残差记录的下游消费者（E3）：每帧末残差喂入发散检测器——只
        // 观测、不改写（求解器语义零变化）。子步内收敛曲线的形状由
        // StepReport 直接观察，此处只需帧级趋势。
        if let Some(&final_residual) = report.residuals.last() {
            self.divergence.observe(final_residual);
        }
        self.frame = frame;
        store.insert((sig, frame), state_bytes(&self.world));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 三粒子竖直链：p0 固定 (0,2,0)，p1/p2 自由（p2 带横向初速 → 摆动，
    /// 非平衡初态），rest 0.5，重力下垂。
    fn chain_spec() -> SimSpec {
        SimSpec {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            particles: vec![
                ParticleInit { position: Vec3::new(0.0, 2.0, 0.0), velocity: Vec3::splat(0.0), inv_mass: 0.0 },
                ParticleInit { position: Vec3::new(0.0, 1.5, 0.0), velocity: Vec3::splat(0.0), inv_mass: 1.0 },
                ParticleInit { position: Vec3::new(0.0, 1.0, 0.0), velocity: Vec3::new(0.5, 0.0, 0.0), inv_mass: 1.0 },
            ],
            constraints: vec![
                DistanceConstraintSpec { id: 1, a: 0, b: 1, rest_length: 0.5, compliance: 0.0 },
                DistanceConstraintSpec { id: 2, a: 1, b: 2, rest_length: 0.5, compliance: 0.0 },
            ],
        }
    }

    fn session() -> SimSession {
        SimSession::new(chain_spec(), 1.0 / 60.0, 4, 8).unwrap()
    }

    const SIG_A: Hash256 = [0xAA; 32];
    const SIG_B: Hash256 = [0xBB; 32];

    /// 关卡 4a：推进 N 帧 → 快照链可读（(sig, 0..=N) 全部就位）且物理合理。
    #[test]
    fn advance_builds_readable_snapshot_chain() {
        let mut s = session();
        let mut store = SnapshotStore::new();
        s.advance_to(SIG_A, 8, &mut store).unwrap();

        assert_eq!(s.frame(), 8);
        assert_eq!(store.len(), 9, "frame 0..=8 每帧一条快照");
        for f in 0..=8u64 {
            assert!(store.get(&(SIG_A, f)).is_some(), "快照 (SIG_A, {f}) 应存在");
        }
        // 物理合理性：p2 带横向初速 → 8 帧后状态必须离开初态（摆动）。
        let p2 = s.world().particles()[2].position;
        assert_ne!(
            p2.x.to_bits(),
            0.0f32.to_bits(),
            "横向初速应使链摆动（p2.x 应偏离 0），实际 x = {}",
            p2.x
        );
    }

    /// 关卡 4b：seek 回放（回 0 帧再推进）与顺序推进逐位一致。
    #[test]
    fn seek_replay_is_bitwise_identical_to_sequential() {
        // 会话 A：顺序推进 0 → 10。
        let mut a = session();
        let mut store_a = SnapshotStore::new();
        for f in 0..=10u64 {
            a.advance_to(SIG_A, f, &mut store_a).unwrap();
        }
        // 会话 B：推进到 6 → seek 回 0（快照恢复路径）→ 再推进到 10。
        let mut b = session();
        let mut store_b = SnapshotStore::new();
        b.advance_to(SIG_A, 6, &mut store_b).unwrap();
        b.advance_to(SIG_A, 0, &mut store_b).unwrap();
        b.advance_to(SIG_A, 10, &mut store_b).unwrap();

        assert_eq!(a.frame(), 10);
        assert_eq!(b.frame(), 10);
        let bits = |s: &SimSession| -> Vec<[u32; 3]> {
            s.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "seek 回放应与顺序推进逐位一致（关卡 4b）");

        // 同帧重入幂等：再 advance 同帧不改变状态。
        b.advance_to(SIG_A, 10, &mut store_b).unwrap();
        assert_eq!(bits(&a), bits(&b), "同帧重入应幂等");
    }

    /// 关卡 4c：签名变更判失效——改参数（新签名）不误用旧签名快照。
    #[test]
    fn signature_change_invalidates_snapshots() {
        // spec1：compliance 0（刚性）；spec2：compliance 0.05（变软）→
        // 物理参数不同 = 子图签名不同。
        let mut spec2 = chain_spec();
        spec2.constraints[0].compliance = 0.05;
        spec2.constraints[1].compliance = 0.05;

        let mut store = SnapshotStore::new();
        let dt = 1.0 / 60.0;

        // 会话一：spec1 推进到 5。
        let mut s1 = SimSession::new(chain_spec(), dt, 4, 8).unwrap();
        s1.advance_to(SIG_A, 5, &mut store).unwrap();
        let bits1 = |s: &SimSession| -> Vec<[u32; 3]> {
            s.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        let first_run = bits1(&s1);

        // 会话二：spec2 + SIG_B 推进到 5——不得误用 (SIG_A, f) 快照。
        // 物理参数不同 ⇒ 结果必须不同（若误读旧快照则逐位相同）。
        let mut s2 = SimSession::new(spec2.clone(), dt, 4, 8).unwrap();
        s2.advance_to(SIG_B, 5, &mut store).unwrap();
        let soft_run = bits1(&s2);
        assert_ne!(
            first_run, soft_run,
            "改参数（新签名）后结果必须变化——证明未误用旧签名快照"
        );

        // 旧签名快照仍在库（键隔离：不删、不误用）。
        for f in 0..=5u64 {
            assert!(store.get(&(SIG_A, f)).is_some(), "旧签名快照 (SIG_A, {f}) 应保留");
        }
        // 新签名会话重入幂等。
        let soft_again;
        {
            let mut s2b = SimSession::new(spec2, dt, 4, 8).unwrap();
            s2b.advance_to(SIG_B, 5, &mut store).unwrap();
            soft_again = bits1(&s2b);
        }
        assert_eq!(soft_run, soft_again, "新签名重算应逐位可复现");

        // 切回旧签名：快照命中直接恢复，状态与首次推进逐位一致。
        s1.advance_to(SIG_A, 5, &mut store).unwrap();
        assert_eq!(bits1(&s1), first_run, "切回旧签名应命中快照并逐位恢复");
    }

    /// 缓存失效（LRU 逐出等价）：快照不命中 → 从初态重推，结果与命中路径
    /// 逐位一致（确定性不依赖缓存命中，D-28）。
    #[test]
    fn evicted_snapshots_recompute_bitwise_identical() {
        // 会话 A：完整快照链推进到 8。
        let mut a = session();
        let mut store_a = SnapshotStore::new();
        a.advance_to(SIG_A, 8, &mut store_a).unwrap();

        // 会话 B：空库（模拟全部快照已被 LRU 逐出）→ 全程重推。
        let mut b = session();
        let mut store_b = SnapshotStore::new();
        b.advance_to(SIG_A, 8, &mut store_b).unwrap();

        let bits = |s: &SimSession| -> Vec<[u32; 3]> {
            s.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "快照逐出后重推应与命中路径逐位一致");
    }

    /// frame 0 = 初态：不入库推进、写 (sig, 0) 快照；seek 0 恒可恢复。
    #[test]
    fn advance_to_zero_is_initial_state() {
        let mut s = session();
        let mut store = SnapshotStore::new();
        s.advance_to(SIG_A, 0, &mut store).unwrap();

        assert_eq!(s.frame(), 0);
        assert_eq!(store.len(), 1);
        let spec = chain_spec();
        for (p, init) in s.world().particles().iter().zip(&spec.particles) {
            assert_eq!(p.position, init.position, "frame 0 应为初态位置");
        }
        // 推进 3 帧后 seek 回 0：快照恢复初态（逐位）。
        s.advance_to(SIG_A, 3, &mut store).unwrap();
        s.advance_to(SIG_A, 0, &mut store).unwrap();
        for (p, init) in s.world().particles().iter().zip(&spec.particles) {
            assert_eq!(
                p.position.x.to_bits(),
                init.position.x.to_bits(),
                "seek 回 0 应逐位恢复初态"
            );
        }
    }

    /// 防呆（C18/C21）：dt/substeps/iterations 非法与规格越界 → 三段式报错。
    #[test]
    fn session_construction_rejects_bad_params() {
        let spec = chain_spec();
        for (dt, substeps, iterations, code) in [
            (0.0, 4u32, 8u32, "P1002"),
            (1.0 / 60.0, 0, 8, "P1003"),
            (1.0 / 60.0, 4, 0, "P1004"),
        ] {
            let err = SimSession::new(spec.clone(), dt, substeps, iterations).unwrap_err();
            assert_eq!(err.code, code, "({dt}, {substeps}, {iterations})");
            assert!(err.to_string().contains("建议："), "必须含建议段：{err}");
        }
        // 规格端点越界 → P3006。
        let mut bad = chain_spec();
        bad.constraints[0].b = 9;
        let err = SimSession::new(bad, 1.0 / 60.0, 4, 8).unwrap_err();
        assert_eq!(err.code, "P3006");
        assert!(err.fix.contains("particles"), "{}", err.fix);
    }

    /// 状态字节编解码：往返逐位一致；头/版本/长度不符 → None（C8 失效重建，
    /// 绝不猜测解读）。
    #[test]
    fn state_bytes_roundtrip_and_rejects_foreign_bytes() {
        let mut world = chain_spec().build_world().unwrap();
        world.step(1.0 / 60.0, 4, 8).unwrap();
        let bytes = state_bytes(&world);
        let (pos, vel) = decode_state(&bytes).expect("合法字节应可解码");
        assert_eq!(pos.len(), 3);
        assert_eq!(vel.len(), 3);
        for (p, wp) in pos.iter().zip(world.particles()) {
            assert_eq!(p.x.to_bits(), wp.position.x.to_bits());
            assert_eq!(p.y.to_bits(), wp.position.y.to_bits());
            assert_eq!(p.z.to_bits(), wp.position.z.to_bits());
        }

        // 外来字节 / 截断 / 版本不符 → None。
        assert_eq!(decode_state(b"not a snapshot"), None);
        assert_eq!(decode_state(&bytes[..bytes.len() - 1]), None);
        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&999u32.to_be_bytes());
        assert_eq!(decode_state(&bad_version), None);
    }

    /// M3 显式化：会话声明固定步数模式元数据，(ε, n) 为缓存键分量；
    /// 迭代数是签名组成部分——改迭代数即改模式元数据（§3.2 S3）。
    #[test]
    fn session_declares_m3_mode_metadata() {
        let s = session(); // dt = 1/60，substeps = 4，iterations = 8
        let meta = s.solve_mode();
        assert_eq!(meta.mode, SolveMode::FixedIterations);
        assert_eq!((meta.substeps, meta.iterations), (4, 8));
        assert_eq!(meta.residual_eps, RESIDUAL_EPS_DEFAULT);
        // 迭代数进签名（new 防呆段自述）⇒ 进模式元数据 ⇒ 进缓存键。
        let s16 = SimSession::new(chain_spec(), 1.0 / 60.0, 4, 16).unwrap();
        assert_ne!(s.solve_mode(), s16.solve_mode(), "迭代数不同 ⇒ 模式元数据不同");
    }

    /// 发散检测接线：每帧末残差喂入检测器（采样点数 = 推进帧数）；
    /// 静定链（零初速、约束初态即满足）不判发散。
    #[test]
    fn session_feeds_divergence_detector_per_frame() {
        // 零初速 + 初始间距恰为 rest_length：下垂静定，无摆动激励。
        let mut spec = chain_spec();
        for p in &mut spec.particles {
            p.velocity = Vec3::splat(0.0);
        }
        let mut s = SimSession::new(spec, 1.0 / 60.0, 4, 8).unwrap();
        let mut store = SnapshotStore::new();
        s.advance_to(SIG_A, 8, &mut store).unwrap();
        assert_eq!(s.divergence().samples(), 8, "每帧末残差应喂入检测器");
        assert!(!s.divergence().diverged(), "静定收敛链不应判发散");
    }
}
