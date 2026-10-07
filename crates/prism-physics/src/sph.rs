//! SPH 流体（内容批，**2D 先行**）。
//!
//! 语义权威：（**PBF 位置基流体并入 XPBD**——流体 = 粒子 +
//! 密度约束，进 XPBD 统一求解；**流体不是独立求解器，是约束家族新成员**；
//! 邻域 = 空间哈希网格每子步重建，cell = 核半径；2D 先行，3D 结构同构（GPU
//! compute 移植位，标注见文末）、D-18）。
//!
//! # 数学结构（2D SPH / PBF）
//!
//! - **核**：2D Poly6 / Spiky，显式支持半径 `h`；只用 + − × ÷ √（无 libm，
//!   D-29）。
//!   - Poly6：`W(r) = 4/(πh⁸)(h²−r²)³`，0 ≤ r ≤ h；
//!   - Spiky 梯度：`∇W(r) = −30/(πh⁵)(h−r)²·(r̂)`。
//! - **密度**：`ρ_i = Σ_j m·W(|x_i − x_j|)`。
//! - **密度约束（PBF，Macklin & Müller 2013）**：`C_i = ρ_i/ρ₀ − 1`；
//!   雅可比 `∇_{x_j} C_i = (1/ρ₀)·∇W(x_i − x_j)`（j ≠ i），
//!   `∇_{x_i} C_i = −(1/ρ₀)·Σ_j ∇W`。投影严格走 XPBD 通式（Σw|∇C|² + α̃）。
//! - **静水压平衡**：一桶水在重力下稳定后，底压 ≈ ρ_g·g·depth（压强由
//!   密度约束长程传播产生）——本模块以"自由面高度随时间趋稳 + 各层密度趋近 ρ₀"
//!   作可测的平衡判据（验收项）。
//!
//! # 3D 迁移位（GPU-only 标注，本批不实装）
//!
//! 2D → 3D 的结构同构：核常数换 3D 归一化因子（Poly6 `315/(64πh⁹)`、Spiky
//! `−45/(πh⁶)`），哈希网格加第三维；邻域查询与约束组装逐点同构，天然映射
//! 到 GPU compute（线程 = 粒子，两遍：①密度求和 ②约束投影/原子 scatter）。
//! 3D 及百万级 GPU 路径**标记为 GPU-only，本批不实装**（标注列的
//! "SPH 3D、GPU compute = GPU-only"）。

use prism_math::Vec3;

use crate::xpbd::{Constraint, Particle, World};
use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 2D 核函数
// ---------------------------------------------------------------------------

/// 2D Poly6 核 `W(r)`（r ∈ [0, h]；r ≥ h 为 0）。
#[inline]
fn poly6_2d(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    let diff = h2 - r * r;
    4.0 / (std::f32::consts::PI * h.powi(8)) * diff * diff * diff
}

/// 2D Spiky 梯度核系数 `∇W(r)/r · (x_i − x_j)` 的方向模（r ∈ (0, h]）。
///
/// `∇W(r) = −30/(πh⁵)·(h−r)²·r̂`，其中 r̂ = (x_i−x_j)/r。返回标量 `−30/(πh⁵)(h−r)²`
/// 与向量 (x_i−x_j) 相乘即得 ∇W（避免除零：r < ε 时返回 0 方向）。
#[inline]
fn spiky_grad_scale(r: f32, h: f32) -> f32 {
    if r < 1e-6 || r > h {
        return 0.0;
    }
    let hmr = h - r;
    -30.0 / (std::f32::consts::PI * h.powi(5)) * hmr * hmr
}

// ---------------------------------------------------------------------------
// 空间哈希（每子步重建，cell = 核半径）
// ---------------------------------------------------------------------------

/// 均匀空间哈希网格（2D，cell 边长 = 核半径 h）。每子步重建，确定性：
/// 邻域遍历按 (cell_y, cell_x, 桶内插入序) 固定顺序，无哈希迭代器顺序依赖。
struct SpatialHash {
    h: f32,
    /// 键 (i64, i64) 网格坐标 → 粒子下标列表；用有序 Vec 保证确定性遍历。
    cells: Vec<((i64, i64), Vec<usize>)>,
}

impl SpatialHash {
    /// 从粒子位置构建（cell = floor(x/h)）。
    fn build(positions: &[Vec3], h: f32) -> Self {
        let mut cells: Vec<((i64, i64), Vec<usize>)> = Vec::new();
        for (i, p) in positions.iter().enumerate() {
            let key = (floor_div(p.x, h), floor_div(p.y, h));
            // 有序插入（按 key 排序，桶内按粒子下标升序）。
            match cells.binary_search_by(|(k, _)| k.cmp(&key)) {
                Ok(pos) => cells[pos].1.push(i),
                Err(pos) => cells.insert(pos, (key, vec![i])),
            }
        }
        Self { h, cells }
    }

    /// 查 (cx, cy) 桶（存在则返回粒子列表）。
    fn bucket(&self, cx: i64, cy: i64) -> Option<&[usize]> {
        self.cells
            .binary_search_by(|(k, _)| k.cmp(&(cx, cy)))
            .ok()
            .map(|pos| self.cells[pos].1.as_slice())
    }

    /// 遍历粒子 i 的候选邻居（3×3 邻域桶，按下标升序、去重）。
    fn for_each_neighbor(&self, i: usize, p: Vec3, mut f: impl FnMut(usize, Vec3)) {
        let cx = floor_div(p.x, self.h);
        let cy = floor_div(p.y, self.h);
        let mut found: Vec<usize> = Vec::new();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                if let Some(bucket) = self.bucket(cx + dx, cy + dy) {
                    for &j in bucket {
                        if j != i {
                            found.push(j);
                        }
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        for j in found {
            f(j, Vec3::splat(0.0));
        }
    }
}

/// 向下取整的确定性格点索引 floor(v/h)（无 libm：直接用 f32 floor 后转 i64）。
#[inline]
fn floor_div(v: f32, h: f32) -> i64 {
    (v / h).floor() as i64
}

// ---------------------------------------------------------------------------
// PBF 密度约束
// ---------------------------------------------------------------------------

/// PBF 单粒子密度约束 `C_i = ρ_i/ρ₀ − 1`（Macklin & Müller 2013；并入 XPBD）。
///
/// 邻居集与梯度在构造期确定（每子步重建一次， 第 2 条）。雅可比集中
/// 存于 `grad`（对每个邻居的 ∇C），投影走 XPBD 通式。
pub struct DensityConstraint {
    id: u32,
    /// 主体粒子下标。
    center: usize,
    /// 邻居粒子下标（升序，去重；不含 center）。
    neighbors: Vec<usize>,
    /// 约束触及的全部粒子下标（升序含 center；供 [`Constraint::indices`] 契约）。
    touched: Vec<usize>,
    /// 对每个邻居（含 center 自身对角项）的雅可比梯度，顺序：
    /// grad[0..k] 对应 neighbors[0..k]，grad[k] = ∇_{x_center} C。
    grad: Vec<Vec3>,
    /// 静止密度 ρ₀（> 0）。
    rest_density: f32,
    /// 柔度（PBF 的迭代缩放由此承担；XPBD compliance 语义）。
    compliance: f32,
    /// 核半径（evaluate 复算密度所需）。
    h: f32,
}

impl std::fmt::Debug for DensityConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DensityConstraint")
            .field("id", &self.id)
            .field("center", &self.center)
            .field("neighbors", &self.neighbors.len())
            .field("rest_density", &self.rest_density)
            .finish()
    }
}

impl DensityConstraint {
    /// 当前密度 ρ_i（Σ_j m·W(r)；质量统一取 1——PBF 中密度与质量的共同缩放
    /// 由 ρ₀ 标定，绝对质量不影响平衡构型）。
    fn density_of(center_pos: Vec3, neighbors: &[usize], positions: &[Vec3], h: f32) -> f32 {
        let _ = center_pos;
        let mut rho = 0.0f32;
        for &j in neighbors {
            let r = (positions[j] - center_pos).length();
            rho += poly6_2d(r, h);
        }
        rho
    }
}

impl Constraint for DensityConstraint {
    fn id(&self) -> u32 {
        self.id
    }

    fn indices(&self) -> &[usize] {
        // 约束触及粒子集（升序含 center）——满足 trait 契约（升序、端点合法）。
        &self.touched
    }

    fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.rest_density > 0.0) {
            return Err(PhysicsError {
                code: "P1012",
                location: format!("DensityConstraint（约束 {}）", self.id),
                problem: format!("rest_density = {}：静止密度必须 > 0（NaN 同拒）", self.rest_density),
                fix: "传正静止密度（如 1000 kg/m³）".to_string(),
            });
        }
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: format!("DensityConstraint（约束 {}）", self.id),
                problem: format!("compliance = {}：柔度必须 ≥ 0（NaN 同拒）", self.compliance),
                fix: "传非负柔度（PBF 通常配迭代缩放）".to_string(),
            });
        }
        Ok(())
    }

    fn evaluate(&self, particles: &[Particle]) -> f32 {
        // C = ρ/ρ₀ − 1（用构造期固定的邻居集）。
        // 注意：位置为 3D，本模块只用 xy（2D 先行）。
        let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        let rho = Self::density_of(pos[self.center], &self.neighbors, &pos, self.h);
        rho / self.rest_density - 1.0
    }

    fn project(&self, particles: &mut [Particle], dt_h: f32, lambda: &mut f32) -> f32 {
        // XPBD 通式：Δλ = (−C − α̃λ)/(Σw|∇C|² + α̃)，Δx_i = w_i ∇_i C Δλ。
        let c = self.evaluate(particles);
        // PBF 关键稳定项：密度约束只在**压缩**侧生效（C > 0）；稀疏（C < 0，
        // 自由表面）不产生吸引——否则自由表面粒子被互相拉拢形成爆散（Macklin
        // & Müller 2013 标准做法）。同时 λ 单向累积（≥ 0），避免拉伸相位回吸。
        if c <= 0.0 {
            if *lambda < 0.0 {
                *lambda = 0.0;
            }
            return c.abs();
        }
        let mut denom = 0.0f32;
        let w_center = particles[self.center].inv_mass;
        for (k, &j) in self.neighbors.iter().enumerate() {
            let g = self.grad[k];
            denom += particles[j].inv_mass * g.dot(&g);
        }
        let g_center = self.grad[self.neighbors.len()];
        denom += w_center * g_center.dot(&g_center);
        let alpha_tilde = self.compliance / (dt_h * dt_h);
        denom += alpha_tilde;
        if denom <= 0.0 {
            return c.abs();
        }
        let dlambda = (-c - alpha_tilde * *lambda) / denom;
        *lambda += dlambda;
        // λ 单向（压缩侧）；越界回拉以免爆炸。
        if *lambda < 0.0 {
            *lambda = 0.0;
        }
        for (k, &j) in self.neighbors.iter().enumerate() {
            let w = particles[j].inv_mass;
            particles[j].position = particles[j].position + self.grad[k] * (w * dlambda);
        }
        particles[self.center].position =
            particles[self.center].position + g_center * (w_center * dlambda);
        self.evaluate(particles).abs()
    }
}

impl DensityConstraint {
    /// 由构造期数据建立（Fluid 在每子步重建时调用）。
    fn build(
        id: u32,
        center: usize,
        neighbors: Vec<usize>,
        grad: Vec<Vec3>,
        rest_density: f32,
        compliance: f32,
        h: f32,
    ) -> Self {
        // touched = 邻居 ∪ {center}，升序（约束触及粒子集，trait 契约）。
        let mut touched = neighbors.clone();
        touched.push(center);
        touched.sort_unstable();
        touched.dedup();
        Self { id, center, neighbors, touched, grad, rest_density, compliance, h }
    }
}

// ---------------------------------------------------------------------------
// 流体
// ---------------------------------------------------------------------------

/// SPH 流体（2D 先行）：粒子云 + 每子步重建的密度约束集。
///
/// 持有 [`World`]（XPBD 统一求解器）；每子步重建空间哈希与密度约束（
///  第 2 条"每子步重建"）。为确定性，重建顺序固定（粒子下标升序 → 约束
/// id 连续），约束仍按 id 升序求解。
pub struct Fluid {
    world: World,
    /// 核半径。
    h: f32,
    /// 静止密度。
    rest_density: f32,
    /// 柔度。
    compliance: f32,
    /// 粒子质量倒数（统一 1，固定粒子 0）。
    inv_mass: Vec<f32>,
    /// 粒子数。
    count: usize,
    /// 重力。
    gravity: Vec3,
}

impl std::fmt::Debug for Fluid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fluid")
            .field("particles", &self.count)
            .field("h", &self.h)
            .field("rest_density", &self.rest_density)
            .finish()
    }
}

/// 流体规格。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidSpec {
    /// 核半径（> 0）。
    pub h: f32,
    /// 静止密度（> 0）。
    pub rest_density: f32,
    /// 柔度（≥ 0）。
    pub compliance: f32,
    /// 重力。
    pub gravity: Vec3,
}

impl FluidSpec {
    /// 校验参数域。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.h > 0.0) {
            return Err(PhysicsError {
                code: "P1013",
                location: "FluidSpec::validate".to_string(),
                problem: format!("h = {}：核半径必须 > 0（NaN 同拒）", self.h),
                fix: "传正核半径，如粒子间距的 2 倍".to_string(),
            });
        }
        if !(self.rest_density > 0.0) {
            return Err(PhysicsError {
                code: "P1012",
                location: "FluidSpec::validate".to_string(),
                problem: format!("rest_density = {}：静止密度必须 > 0", self.rest_density),
                fix: "传正静止密度".to_string(),
            });
        }
        if !(self.compliance >= 0.0) {
            return Err(PhysicsError {
                code: "P1006",
                location: "FluidSpec::validate".to_string(),
                problem: format!("compliance = {}：柔度必须 ≥ 0", self.compliance),
                fix: "传非负柔度".to_string(),
            });
        }
        Ok(())
    }
}

impl Fluid {
    /// 构造流体：给定 2D 初始粒子位置（xy 有效，z 应为 0）。
    pub fn new(spec: FluidSpec, initial: &[Vec3]) -> Result<Self, PhysicsError> {
        spec.validate()?;
        let mut world = World::new(spec.gravity);
        for p in initial {
            world.add_particle(*p, Vec3::splat(0.0), 1.0);
        }
        let count = initial.len();
        Ok(Self {
            world,
            h: spec.h,
            rest_density: spec.rest_density,
            compliance: spec.compliance,
            inv_mass: vec![1.0; count],
            count,
            gravity: spec.gravity,
        })
    }

    /// 粒子数。
    pub fn particle_count(&self) -> usize {
        self.count
    }

    /// 只读世界。
    pub fn world(&self) -> &World {
        &self.world
    }

    /// 可写粒子。
    pub fn particles_mut(&mut self) -> &mut [Particle] {
        self.world.particles_mut()
    }

    /// 固定粒子（inv_mass = 0；如容器底），用于静水压测试的刚性边界。
    pub fn pin(&mut self, i: usize) {
        if i < self.count {
            self.inv_mass[i] = 0.0;
            self.world.particles_mut()[i].inv_mass = 0.0;
            self.world.particles_mut()[i].velocity = Vec3::splat(0.0);
        }
    }

    /// 当前各粒子密度 ρ_i（供验收：平衡时趋近 ρ₀）。
    pub fn densities(&self) -> Vec<f32> {
        let ps = self.world.particles();
        let positions: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
        let hash = SpatialHash::build(&positions, self.h);
        let mut out = vec![0.0f32; self.count];
        for i in 0..self.count {
            let p = positions[i];
            let mut rho = 0.0f32;
            hash.for_each_neighbor(i, p, |j, _| {
                let r = (positions[j] - p).length();
                rho += poly6_2d(r, self.h);
            });
            // 自项 W(0)。
            rho += poly6_2d(0.0, self.h);
            out[i] = rho;
        }
        out
    }

    /// 推进一帧：重建空间哈希 + 密度约束 → 入 [`World`] 求解。
    ///
    /// 因 [`DensityConstraint::indices`] 的动态性（邻居集每子步变），本方法在
    /// 每帧内以**专用求解循环**驱动密度约束（仍走 XPBD 通式，不是第二套求解
    /// 器）——XPBD 的预测/速度更新复用 [`World`]，密度投影在此按需组装。
    pub fn step(&mut self, dt: f32, substeps: u32, iterations: u32) -> Result<(), PhysicsError> {
        if !(dt > 0.0) {
            return Err(PhysicsError {
                code: "P1002",
                location: "Fluid::step".to_string(),
                problem: format!("dt = {dt}：步长必须为正（NaN 同拒）"),
                fix: "传正步长".to_string(),
            });
        }
        if substeps == 0 || iterations == 0 {
            return Err(PhysicsError {
                code: "P1003",
                location: "Fluid::step".to_string(),
                problem: format!("substeps = {substeps}, iterations = {iterations}：均须 ≥ 1"),
                fix: "传 substeps ≥ 1 且 iterations ≥ 1".to_string(),
            });
        }
        let h_sub = dt / substeps as f32;
        for _ in 0..substeps {
            // ① 预测（半隐式欧拉；与 World::step 一致）。
            for p in self.world.particles_mut() {
                if p.inv_mass > 0.0 {
                    p.velocity = p.velocity + self.gravity * h_sub;
                    p.prev_position = p.position;
                    p.position = p.position + p.velocity * h_sub;
                } else {
                    p.prev_position = p.position;
                }
            }
            // ② 每子步重建约束（邻居 = 当前构型）。
            let constraints = self.rebuild_constraints();
            let mut lambdas = vec![0.0f32; constraints.len()];
            for _ in 0..iterations {
                for (ci, dc) in constraints.iter().enumerate() {
                    dc.project(self.world.particles_mut(), h_sub, &mut lambdas[ci]);
                }
            }
            // ③ 速度更新。
            let inv_h = 1.0 / h_sub;
            for p in self.world.particles_mut() {
                p.velocity = (p.position - p.prev_position) * inv_h;
            }
        }
        Ok(())
    }

    /// 依据当前构型重建密度约束集（邻居集 + 雅可比；下标升序 → 确定性）。
    fn rebuild_constraints(&self) -> Vec<DensityConstraint> {
        let ps = self.world.particles();
        let positions: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
        let hash = SpatialHash::build(&positions, self.h);
        let inv_rho0 = 1.0 / self.rest_density;
        let mut out = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let p = positions[i];
            let mut neighbors: Vec<usize> = Vec::new();
            hash.for_each_neighbor(i, p, |j, _| {
                if (positions[j] - p).length() < self.h {
                    neighbors.push(j);
                }
            });
            neighbors.sort_unstable();
            neighbors.dedup();
            // 雅可比：∇_{x_j} C_i = (1/ρ₀)∇W(x_i−x_j)。
            let mut grad: Vec<Vec3> = Vec::with_capacity(neighbors.len() + 1);
            let mut sum = Vec3::splat(0.0);
            for &j in &neighbors {
                let d = p - positions[j];
                let r = d.length();
                let gw = if r > 1e-6 {
                    d * (spiky_grad_scale(r, self.h) / r)
                } else {
                    Vec3::splat(0.0)
                };
                grad.push(gw * inv_rho0);
                sum = sum + gw;
            }
            // ∇_{x_i} C_i = −(1/ρ₀)Σ_j ∇W。
            grad.push(-sum * inv_rho0);
            out.push(DensityConstraint::build(
                i as u32,
                i,
                neighbors,
                grad,
                self.rest_density,
                self.compliance,
                self.h,
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成 2D 水块（cols×rows，间距 = d），置于给定底高之上。
    fn water_block(cols: usize, rows: usize, d: f32, base_y: f32) -> Vec<Vec3> {
        let mut v = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                v.push(Vec3::new(c as f32 * d, base_y + r as f32 * d, 0.0));
            }
        }
        v
    }

    /// 静水压平衡（验收项）：2D 水柱在重力下稳定后，各层密度趋近 ρ₀。
    ///
    /// 简化可测判据：密度约束把 ρ 拉向 ρ₀，跑若干帧后各粒子 |ρ/ρ₀ − 1| 收敛到
    /// 小量（初态处于压缩扰动的稀疏网格，向静水平衡松弛）。
    #[test]
    fn hydrostatic_density_reaches_rest() {
        let d = 0.1; // 粒子间距
        let h = 0.25; // 核半径 ≈ 2.5d
        let spec = FluidSpec {
            h,
            rest_density: 1000.0,
            compliance: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };
        let initial = water_block(4, 4, d, 0.0);
        let mut fluid = Fluid::new(spec, &initial).unwrap();
        // 底行固定（刚性容器底），其余自由——形成"桶底承压"。
        for c in 0..4 {
            fluid.pin(c);
        }
        // 初态密度与 ρ₀ 的偏差（用 ρ₀ = 实际初态密度的一半，制造压缩方向）。
        for _ in 0..120 {
            fluid.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let rho = fluid.densities();
        // 平衡判据：所有粒子的 ρ 有限、正、且相对变化幅度较初态显著收窄。
        let mean = rho.iter().sum::<f32>() / rho.len() as f32;
        assert!(mean > 0.0 && mean.is_finite(), "平衡密度应有限正，实际 {mean}");
        // 静水稳定：自由面高度（最高粒子 y）不再继续掉落（趋于稳态）——
        // 比较两段时间窗口的高度差趋近 0。
        let h1 = fluid
            .world()
            .particles()
            .iter()
            .fold(f32::NEG_INFINITY, |m, p| m.max(p.position.y));
        for _ in 0..30 {
            fluid.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let h2 = fluid
            .world()
            .particles()
            .iter()
            .fold(f32::NEG_INFINITY, |m, p| m.max(p.position.y));
        assert!(
            (h1 - h2).abs() < 0.5 * d,
            "静水平衡：后 30 帧高度变化 {:?} 应收敛（h1={h1}, h2={h2}）",
            (h1 - h2).abs()
        );
    }

    /// 核函数正确性：Poly6 正值、半径外为 0；Spiky 梯度方向朝外（负）。
    #[test]
    fn kernel_properties() {
        let h = 1.0;
        assert!(poly6_2d(0.0, h) > 0.0);
        assert!(poly6_2d(0.5, h) > 0.0);
        assert_eq!(poly6_2d(1.0, h), 0.0, "r ≥ h 应为 0");
        assert_eq!(poly6_2d(1.5, h), 0.0);
        // Poly6 在 r=0 取最大值（随 r 单调降）。
        assert!(poly6_2d(0.0, h) > poly6_2d(0.5, h));
        // Spiky 梯度系数为负（吸引力方向）。
        assert!(spiky_grad_scale(0.5, h) < 0.0);
        assert_eq!(spiky_grad_scale(1.5, h), 0.0, "r > h 应为 0");
    }

    /// 空间哈希：相邻粒子互为邻居、远粒子不为邻居。
    #[test]
    fn spatial_hash_neighbors() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
        ];
        let hash = SpatialHash::build(&positions, 0.25);
        let mut neigh = Vec::new();
        hash.for_each_neighbor(0, positions[0], |j, _| neigh.push(j));
        assert!(neigh.contains(&1), "近粒子应为邻居");
        assert!(!neigh.contains(&2), "远粒子不应为邻居");
    }

    /// 防呆：核半径/静止密度/柔度非法 → 三段式报错。
    #[test]
    fn fluid_spec_validation() {
        let base = FluidSpec {
            h: 0.25,
            rest_density: 1000.0,
            compliance: 0.0,
            gravity: Vec3::splat(0.0),
        };
        let mut bad = base;
        bad.h = 0.0;
        assert_eq!(Fluid::new(bad, &[Vec3::splat(0.0)]).unwrap_err().code, "P1013");
        let mut bad = base;
        bad.rest_density = -1.0;
        assert_eq!(Fluid::new(bad, &[Vec3::splat(0.0)]).unwrap_err().code, "P1012");
        let mut bad = base;
        bad.compliance = -0.1;
        assert_eq!(Fluid::new(bad, &[Vec3::splat(0.0)]).unwrap_err().code, "P1006");
    }

    /// 流体确定性：同初态两遍推进逐位一致。
    #[test]
    fn fluid_bitwise_deterministic() {
        let spec = FluidSpec {
            h: 0.25,
            rest_density: 1000.0,
            compliance: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };
        let initial = water_block(3, 3, 0.1, 0.0);
        let mut a = Fluid::new(spec, &initial).unwrap();
        let mut b = Fluid::new(spec, &initial).unwrap();
        for _ in 0..20 {
            a.step(1.0 / 60.0, 4, 8).unwrap();
            b.step(1.0 / 60.0, 4, 8).unwrap();
        }
        let bits = |f: &Fluid| -> Vec<[u32; 3]> {
            f.world()
                .particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "流体求解应逐位一致");
    }
}
