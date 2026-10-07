//! Verlet 布料（内容批）：迁移进 XPBD 统一求解器。
//!
//! 语义权威：（布料 = 距离/弯曲约束网格；**不引入独立 Verlet
//! 求解器，而是把 Verlet 语义作为 XPBD 的退化/特长形态**——XPBD 的距离约束在
//! compliance = 0 时即完全投影（PBD/Verlet 的位置型刚性语义），阻尼项由速度
//! 侧承担。Verlet 布料三要素：3 pin 模式 / 风 / 阻尼）、D-18（走
//! B7 XPBD 统一求解器）。
//!
//! # 迁移要点（Verlet → XPBD）
//!
//! 经典 Verlet 布料以"位置 = 2·x − x_prev + a·h²"隐式积分 + 逐约束位置投影，
//! 与本引擎 XPBD 的"半隐式欧拉预测 + XPBD 投影"在无外加约束时等价（XPBD 的
//! 速度侧显式存储 v = (x − x_prev)/h 即 Verlet 的隐式速度）。二者的唯一差异
//! 是刚性投影的分母：Verlet 一次投影无 compliance；XPBD 的 compliance = 0
//! 给出同样的"完全投影"。故迁移 = **以 XPBD 距离约束（含弯曲约束）承载布料，
//! 速度阻尼作为显式外力**——求解器仍是 [`crate::xpbd::World`]，零新求解器。
//!
//! # 三 pin 模式
//!
//! [`PinMode::Corners`]（对角两角固定，自由垂摆——布料经典预置）、
//! [`PinMode::TopEdge`]（上边全固定，帘幕/旗帜）、[`PinMode::SingleTop`]
//! （单点悬挂，锥形垂落）。角点固定 = 对应粒子 `inv_mass = 0`，能量单调降的
//! 自然边界（固定点不注入能量）。
//!
//! # 能量单调降（验收项）
//!
//! 无外力（重力 = 0）且无初速的布料，从拉伸初态出发，总能量 E = Σ½m|v|² + Σ½kC²
//! 应在每个时间步上**单调非增**（XPBD 投影是压缩映射，；速度阻尼
//! 进一步耗散）。本模块暴露 [`Cloth::energy`] 供验收断言，并用
//! [`Cloth::step_with_energy`] 返回每步能量序列。

use prism_math::Vec3;

use crate::xpbd::{DistanceConstraint, Particle, World};
use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 布料规格
// ---------------------------------------------------------------------------

/// 角点固定模式（Verlet 布料三 pin 预置）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    /// 四角固定（自由垂摆的方形布）。
    Corners,
    /// 上边整行固定（帘幕 / 旗帜）。
    TopEdge,
    /// 单个上中角点固定（锥形垂落）。
    SingleTop,
}

/// 布料网格规格：`rows × cols` 粒子网格 + pin 模式 + 材料参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClothSpec {
    /// 行数（≥ 2）。
    pub rows: usize,
    /// 列数（≥ 2）。
    pub cols: usize,
    /// 网格边长（相邻粒子静止距离）。
    pub spacing: f32,
    /// 重力。
    pub gravity: Vec3,
    /// 固定模式。
    pub pin_mode: PinMode,
    /// 结构约束（相邻）compliance（0 = 完全刚性）。
    pub structural_compliance: f32,
    /// 弯曲约束（隔一个）compliance（抗折刚度；越大越软）。
    pub bend_compliance: f32,
    /// 速度阻尼系数 ∈ [0, 1]：每子步末 v ← v·(1 − damping)（0 = 无阻尼）。
    pub damping: f32,
}

impl ClothSpec {
    /// 校验规格：网格 ≥ 2×2、spacing > 0、参数域非负、阻尼 ∈ [0, 1]。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        if self.rows < 2 || self.cols < 2 {
            return Err(PhysicsError {
                code: "P1009",
                location: "ClothSpec::validate".to_string(),
                problem: format!("网格 {}×{}：行/列均须 ≥ 2（布料网格）", self.rows, self.cols),
                fix: "传 rows ≥ 2 且 cols ≥ 2".to_string(),
            });
        }
        if !(self.spacing > 0.0) {
            return Err(PhysicsError {
                code: "P1010",
                location: "ClothSpec::validate".to_string(),
                problem: format!("spacing = {}：边长必须 > 0（NaN 同拒）", self.spacing),
                fix: "传正边长".to_string(),
            });
        }
        for (name, v) in [
            ("structural_compliance", self.structural_compliance),
            ("bend_compliance", self.bend_compliance),
        ] {
            if !(v >= 0.0) {
                return Err(PhysicsError {
                    code: "P1006",
                    location: "ClothSpec::validate".to_string(),
                    problem: format!("{name} = {v}：柔度必须 ≥ 0（NaN 同拒）"),
                    fix: "传非负柔度".to_string(),
                });
            }
        }
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(PhysicsError {
                code: "P1011",
                location: "ClothSpec::validate".to_string(),
                problem: format!("damping = {}：阻尼系数必须 ∈ [0, 1]（NaN 同拒）", self.damping),
                fix: "传 [0, 1] 区间内的阻尼系数（0 = 无阻尼）".to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 布料
// ---------------------------------------------------------------------------

/// 布料：网格粒子云 + 结构/弯曲距离约束 + pin 固定。
///
/// 内部持有一个 [`World`]（XPBD 统一求解器）——布料不是独立求解器，是约束集
/// 的组织方式。粒子下标 = `row * cols + col`。
pub struct Cloth {
    spec: ClothSpec,
    world: World,
    /// 网格粒子数（= rows × cols）。
    count: usize,
    /// 每粒子质量的倒数（用于能量估计；固定粒子记 0）。
    inv_mass: Vec<f32>,
}

impl std::fmt::Debug for Cloth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cloth")
            .field("rows", &self.spec.rows)
            .field("cols", &self.spec.cols)
            .field("particles", &self.count)
            .finish()
    }
}

impl Cloth {
    /// 构造布料：铺开网格、施加结构 + 弯曲约束、按 pin 模式固定角点。
    ///
    /// 网格在第 r 行第 c 列的位置 = (c·spacing, 0, r·spacing)（XZ 平面平铺，
    /// 重力沿 −Y 垂下）。约束 id 连续编号（结构先、弯曲后），求解序按 id 格锚。
    pub fn new(spec: ClothSpec) -> Result<Self, PhysicsError> {
        spec.validate()?;
        let (rows, cols) = (spec.rows, spec.cols);
        let count = rows * cols;
        let mut world = World::new(spec.gravity);
        // 全部粒子质量 1（逆质量 1）；角点在下方置 0。
        for r in 0..rows {
            for c in 0..cols {
                let p = Vec3::new(c as f32 * spec.spacing, 0.0, r as f32 * spec.spacing);
                world.add_particle(p, Vec3::splat(0.0), 1.0);
            }
        }
        let idx = |r: usize, c: usize| r * cols + c;
        let mut id = 0u32;
        // 结构约束：水平 + 垂直相邻（静止长 = spacing）。
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 1),
                        spec.spacing,
                        spec.structural_compliance,
                    )))?;
                    id += 1;
                }
                if r + 1 < rows {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 1, c),
                        spec.spacing,
                        spec.structural_compliance,
                    )))?;
                    id += 1;
                }
            }
        }
        // 弯曲约束：隔一个格子（抵抗折痕；静止长 = 2·spacing）。
        for r in 0..rows {
            for c in 0..cols {
                if c + 2 < cols {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 2),
                        2.0 * spec.spacing,
                        spec.bend_compliance,
                    )))?;
                    id += 1;
                }
                if r + 2 < rows {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 2, c),
                        2.0 * spec.spacing,
                        spec.bend_compliance,
                    )))?;
                    id += 1;
                }
            }
        }
        // pin 固定：把选中粒子逆质量置 0。World 无直接改逆质量 API，故重建
        // 世界粒子（保持下标与约束端点一致）。
        let pins = Self::pin_indices(&spec);
        let mut inv_mass = vec![1.0f32; count];
        for &i in &pins {
            inv_mass[i] = 0.0;
        }
        // 用 pin 后的逆质量重建同一世界（粒子位置/速度初值不变）。
        // 说明：World::add_particle 的 inv_mass 即身份，重建保证 pin 生效且
        // 约束端点不变（约束已按同一 idx 建立，此处整体重建，避免半状态）。
        world = Self::rebuild_with_pins(&spec, &inv_mass, &pins)?;
        Ok(Self { spec, world, count, inv_mass })
    }

    /// pin 模式对应的固定粒子下标集。
    fn pin_indices(spec: &ClothSpec) -> Vec<usize> {
        let (rows, cols) = (spec.rows, spec.cols);
        let idx = |r: usize, c: usize| r * cols + c;
        match spec.pin_mode {
            PinMode::Corners => vec![
                idx(0, 0),
                idx(0, cols - 1),
                idx(rows - 1, 0),
                idx(rows - 1, cols - 1),
            ],
            PinMode::TopEdge => (0..cols).map(|c| idx(0, c)).collect(),
            PinMode::SingleTop => vec![idx(0, cols / 2)],
        }
    }

    /// 以给定逆质量重建世界（pin 生效）；约束与 [`Self::new`] 一致。
    fn rebuild_with_pins(
        spec: &ClothSpec,
        inv_mass: &[f32],
        _pins: &[usize],
    ) -> Result<World, PhysicsError> {
        let (rows, cols) = (spec.rows, spec.cols);
        let mut world = World::new(spec.gravity);
        for r in 0..rows {
            for c in 0..cols {
                let p = Vec3::new(c as f32 * spec.spacing, 0.0, r as f32 * spec.spacing);
                world.add_particle(p, Vec3::splat(0.0), inv_mass[r * cols + c]);
            }
        }
        let idx = |r: usize, c: usize| r * cols + c;
        let mut id = 0u32;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 1),
                        spec.spacing,
                        spec.structural_compliance,
                    )))?;
                    id += 1;
                }
                if r + 1 < rows {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 1, c),
                        spec.spacing,
                        spec.structural_compliance,
                    )))?;
                    id += 1;
                }
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                if c + 2 < cols {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 2),
                        2.0 * spec.spacing,
                        spec.bend_compliance,
                    )))?;
                    id += 1;
                }
                if r + 2 < rows {
                    world.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 2, c),
                        2.0 * spec.spacing,
                        spec.bend_compliance,
                    )))?;
                    id += 1;
                }
            }
        }
        Ok(world)
    }

    /// 只读世界。
    pub fn world(&self) -> &World {
        &self.world
    }

    /// 粒子数。
    pub fn particle_count(&self) -> usize {
        self.count
    }

    /// 可写粒子切片（供初始化构型，如拉伸/初始扰动——确定性由调用方输入决定）。
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        self.world.particles_mut()
    }

    /// 当前总能量 E = Σ½m|v|² + Σ½k C²（m = 1/inv_mass；k = 1/compliance 为弹性
    /// 刚度；compliance = 0 的刚性约束以有限大刚度代理 k = 1/ε 计算——趋于无穷
    /// 但 C 同时趋于 0，产物有界）。
    ///
    /// 势能 = ½·k·C²：Σ 结构 + 弯曲约束（布料的物理弹性势能， 的
    /// compliance 即 1/k）。这是可比的物理量：无阻尼时近守恒、有阻尼时单调耗散
    /// （验收项"能量单调降"）。
    pub fn energy(&self) -> f32 {
        let ps = self.world.particles();
        let mut e = 0.0f32;
        for (p, &w) in ps.iter().zip(&self.inv_mass) {
            if w > 0.0 {
                let m = 1.0 / w;
                e += 0.5 * m * p.velocity.dot(&p.velocity);
            }
        }
        e += self.potential_energy();
        e
    }

    /// 弹性势能 Σ½·k·C²（k = 1/compliance；compliance = 0 用 ε = 1e-6 代理 ——
    /// 刚性极限，C 亦趋 0，产物有界）。
    fn potential_energy(&self) -> f32 {
        let (rows, cols) = (self.spec.rows, self.spec.cols);
        let ps = self.world.particles();
        let k_struct = 1.0 / self.spec.structural_compliance.max(1e-6);
        let k_bend = 1.0 / self.spec.bend_compliance.max(1e-6);
        let idx = |r: usize, c: usize| r * cols + c;
        let c_of = |a: usize, b: usize, rest: f32| -> f32 {
            (ps[a].position - ps[b].position).length() - rest
        };
        let mut e = 0.0f32;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    let cc = c_of(idx(r, c), idx(r, c + 1), self.spec.spacing);
                    e += 0.5 * k_struct * cc * cc;
                }
                if r + 1 < rows {
                    let cc = c_of(idx(r, c), idx(r + 1, c), self.spec.spacing);
                    e += 0.5 * k_struct * cc * cc;
                }
                if c + 2 < cols {
                    let cc = c_of(idx(r, c), idx(r, c + 2), 2.0 * self.spec.spacing);
                    e += 0.5 * k_bend * cc * cc;
                }
                if r + 2 < rows {
                    let cc = c_of(idx(r, c), idx(r + 2, c), 2.0 * self.spec.spacing);
                    e += 0.5 * k_bend * cc * cc;
                }
            }
        }
        e
    }

    /// 推进一帧（XPBD 统一求解器；含速度阻尼）。
    pub fn step(&mut self, dt: f32, substeps: u32, iterations: u32) -> Result<(), PhysicsError> {
        self.world.step(dt, substeps, iterations)?;
        self.apply_damping();
        Ok(())
    }

    /// 推进一帧并返回本帧结束时能量（供能量单调降断言）。
    pub fn step_with_energy(
        &mut self,
        dt: f32,
        substeps: u32,
        iterations: u32,
    ) -> Result<f32, PhysicsError> {
        self.step(dt, substeps, iterations)?;
        Ok(self.energy())
    }

    /// 速度阻尼：v ← v·(1 − damping)（Verlet 布料"阻尼"要素的 XPBD 承载）。
    /// 固定粒子（inv_mass = 0）速度恒 0，无需处理。
    fn apply_damping(&mut self) {
        if self.spec.damping <= 0.0 {
            return;
        }
        let factor = 1.0 - self.spec.damping;
        for p in self.world.particles_mut() {
            if p.inv_mass > 0.0 {
                p.velocity = p.velocity * factor;
            }
        }
    }

    /// 角点是否固定于其初位（验收：角点固定）。
    pub fn pin_is_fixed(&self, dt: f32, substeps: u32, iterations: u32) -> bool {
        let pins = Self::pin_indices(&self.spec);
        let before: Vec<Vec3> = pins.iter().map(|&i| self.world.particles()[i].position).collect();
        // 只读检查——用一临时世界步进（不污染本布料）。
        let mut tmp = Cloth { spec: self.spec, world: self.clone_world(), count: self.count, inv_mass: self.inv_mass.clone() };
        if tmp.step(dt, substeps, iterations).is_err() {
            return false;
        }
        for (k, &i) in pins.iter().enumerate() {
            let p = tmp.world.particles()[i].position;
            if p.x.to_bits() != before[k].x.to_bits()
                || p.y.to_bits() != before[k].y.to_bits()
                || p.z.to_bits() != before[k].z.to_bits()
            {
                return false;
            }
        }
        true
    }

    /// 复制当前世界（用于只读步进检查）。
    fn clone_world(&self) -> World {
        let mut w = World::new(self.spec.gravity);
        for p in self.world.particles() {
            w.add_particle(p.position, p.velocity, p.inv_mass);
        }
        // 约束按同一构造路径重建（id 连续）。
        let (rows, cols) = (self.spec.rows, self.spec.cols);
        let idx = |r: usize, c: usize| r * cols + c;
        let mut id = 0u32;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    w.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 1),
                        self.spec.spacing,
                        self.spec.structural_compliance,
                    )))
                    .unwrap();
                    id += 1;
                }
                if r + 1 < rows {
                    w.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 1, c),
                        self.spec.spacing,
                        self.spec.structural_compliance,
                    )))
                    .unwrap();
                    id += 1;
                }
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                if c + 2 < cols {
                    w.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r, c + 2),
                        2.0 * self.spec.spacing,
                        self.spec.bend_compliance,
                    )))
                    .unwrap();
                    id += 1;
                }
                if r + 2 < rows {
                    w.add_constraint(Box::new(DistanceConstraint::new(
                        id,
                        idx(r, c),
                        idx(r + 2, c),
                        2.0 * self.spec.spacing,
                        self.spec.bend_compliance,
                    )))
                    .unwrap();
                    id += 1;
                }
            }
        }
        w
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_cloth(mode: PinMode) -> ClothSpec {
        ClothSpec {
            rows: 4,
            cols: 4,
            spacing: 0.2,
            gravity: Vec3::splat(0.0),
            pin_mode: mode,
            structural_compliance: 0.0,
            bend_compliance: 0.001,
            damping: 0.0,
        }
    }

    /// 角点固定：Corners 模式四角在重力作用下位置逐位不动。
    #[test]
    fn pins_stay_fixed_under_gravity() {
        let mut spec = flat_cloth(PinMode::Corners);
        spec.gravity = Vec3::new(0.0, -9.81, 0.0);
        let mut cloth = Cloth::new(spec).unwrap();
        let pins = Cloth::pin_indices(&spec);
        let before: Vec<Vec3> = pins.iter().map(|&i| cloth.world().particles()[i].position).collect();
        for _ in 0..50 {
            cloth.step(1.0 / 60.0, 4, 8).unwrap();
        }
        for (k, &i) in pins.iter().enumerate() {
            let p = cloth.world().particles()[i].position;
            assert_eq!(p.x.to_bits(), before[k].x.to_bits(), "角点 x 应固定");
            assert_eq!(p.y.to_bits(), before[k].y.to_bits(), "角点 y 应固定");
            assert_eq!(p.z.to_bits(), before[k].z.to_bits(), "角点 z 应固定");
        }
    }

    /// 能量单调降（验收项）：无重力、无初速、拉伸初态 → 每帧能量非增。
    #[test]
    fn energy_monotonically_decreases() {
        // 拉伸初态：把内部粒子沿 +x 拉出（制造约束残余）。用有限柔度（软布）
        // 使能量为可比物理量（刚性 compliance=0 的刚度无穷、势能不可比）。
        let mut spec = flat_cloth(PinMode::Corners);
        spec.structural_compliance = 0.001;
        spec.bend_compliance = 0.001;
        spec.damping = 0.01;
        let mut cloth = Cloth::new(spec).unwrap();
        for (i, p) in cloth.particles_mut().iter_mut().enumerate() {
            if i % 4 != 0 {
                p.position = p.position + Vec3::new(0.05, 0.0, 0.0);
            }
        }
        let mut prev = cloth.energy();
        let mut seq = vec![prev];
        for _ in 0..40 {
            let e = cloth.step_with_energy(1.0 / 60.0, 4, 8).unwrap();
            seq.push(e);
            assert!(
                e <= prev + 1e-6,
                "能量应单调非增：prev={prev} → e={e}（序列尾部 {:?}）",
                &seq[seq.len().saturating_sub(6)..]
            );
            prev = e;
        }
        // 且确实下降（非平台：拉伸初态应显著耗散）。
        assert!(seq[0] > seq[seq.len() - 1], "拉伸初态能量应下降");
    }

    /// TopEdge / SingleTop 两种 pin 模式下布料在重力中下垂（自由粒子离开平面）。
    #[test]
    fn free_particles_sag_under_gravity() {
        for mode in [PinMode::TopEdge, PinMode::SingleTop] {
            let mut spec = flat_cloth(mode);
            spec.gravity = Vec3::new(0.0, -9.81, 0.0);
            let mut cloth = Cloth::new(spec).unwrap();
            for _ in 0..60 {
                cloth.step(1.0 / 60.0, 4, 8).unwrap();
            }
            // 至少一个自由粒子 y < 0（下垂）。
            let sag = cloth.world().particles().iter().any(|p| p.position.y < -1e-4);
            assert!(sag, "{mode:?} 模式下布料应下垂（自由粒子 y < 0）");
        }
    }

    /// 规则校验：网格尺寸/边长/阻尼域非法 → 三段式报错。
    #[test]
    fn cloth_spec_validation() {
        let mut bad = flat_cloth(PinMode::Corners);
        bad.rows = 1;
        assert_eq!(Cloth::new(bad).unwrap_err().code, "P1009");
        let mut bad = flat_cloth(PinMode::Corners);
        bad.spacing = 0.0;
        assert_eq!(Cloth::new(bad).unwrap_err().code, "P1010");
        let mut bad = flat_cloth(PinMode::Corners);
        bad.damping = 1.5;
        assert_eq!(Cloth::new(bad).unwrap_err().code, "P1011");
        let mut bad = flat_cloth(PinMode::Corners);
        bad.structural_compliance = -0.1;
        assert_eq!(Cloth::new(bad).unwrap_err().code, "P1006");
    }

    /// 布料确定性：同规格两遍推进逐位一致。
    #[test]
    fn cloth_bitwise_deterministic() {
        let build = || {
            let mut spec = flat_cloth(PinMode::TopEdge);
            spec.gravity = Vec3::new(0.0, -9.81, 0.0);
            spec.damping = 0.02;
            Cloth::new(spec).unwrap()
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..30 {
            a.step(1.0 / 60.0, 4, 8).unwrap();
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let bits = |c: &Cloth| -> Vec<[u32; 3]> {
            c.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "布料求解应逐位一致");
    }
}
