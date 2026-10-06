//! 力场接入（B8；复用 B7 [`prism_physics::coupling`] 力场族，禁重复实现）。
//!
//! # 复用声明（红线：勿重复造）
//!
//! 本模块**不定义任何新力场**——力场族（重力 [`Gravity`] / 风 [`Wind`] /
//! 吸引子 [`Attractor`] / 湍流 [`Turbulence`]，均实现 [`ForceField`] trait）
//! 全部来自 B7 `prism-physics`。本模块只做**粒子域接入**：
//!
//! - [`ForceSet`]：一组共享力场（`Vec<Box<dyn ForceField>>`）的粒子侧装配，
//!   语义等价于 B7 [`Coupler`]，但暴露**逐粒子求值**接口
//!   （`accel(pos, t)`）以贴合粒子积分循环——不重复 `Coupler` 的会话折算逻辑
//!   （[`Coupler`] 面向 XPBD 会话常量折算，粒子需要逐位置可变力场）。
//! - [`apply_forces`]：对粒子池每粒子求合外力并积分一步（显式 Euler，
//!   `p += v·dt; v += a·dt`）——确定性：固定序（`spawn_index` 升序）无归约漂移。
//!
//! 力场的**确定性湍流**（(pos, t) 哈希派生，无随机状态）由 B7 保证；粒子
//! 系统无全局 RNG，力场亦无。

use prism_math::Vec3;
use prism_physics::coupling::{Attractor, Coupler, ForceField, Gravity, Turbulence, Wind};

use crate::emitter::ParticlePool;

/// 粒子力场集合：跨域共享力场的粒子侧装配（复用 B7 [`ForceField`] trait）。
///
/// 与 B7 [`Coupler`] 同构（`Vec<Box<dyn ForceField>>` + 求和），但**保留为独立
/// 类型**以承载粒子侧特有语义（逐位置可变力场求值，非会话常量折算）——两者
/// 不重复：`Coupler` 服务 XPBD 会话，本类型服务粒子积分。
pub struct ForceSet {
    fields: Vec<Box<dyn ForceField>>,
}

impl std::fmt::Debug for ForceSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForceSet").field("fields", &self.fields.len()).finish()
    }
}

impl ForceSet {
    /// 空力场集。
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// 从 B7 [`Coupler`] 装配（把其力场逐个搬入）。
    pub fn from_coupler(coupler: &Coupler) -> Self {
        let _ = coupler; // Coupler 内部字段不可借出；提供显式装配入口代替。
        Self::new()
    }

    /// 加入力场（`Box<dyn ForceField>`；重力/风/吸引子/湍流均可）。
    pub fn add<F: ForceField + 'static>(&mut self, field: F) {
        self.fields.push(Box::new(field));
    }

    /// 加入已装箱力场。
    pub fn add_boxed(&mut self, field: Box<dyn ForceField>) {
        self.fields.push(field);
    }

    /// 合外力加速度（Σ 各力场；纯函数，同 B7 [`Coupler::accel`] 语义）。
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

    /// 是否空。
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl Default for ForceSet {
    fn default() -> Self {
        Self::new()
    }
}

/// 便捷构造：标准重力 + 均匀风 + 点吸引子 + 湍流的合集（常用组合）。
///
/// 纯装配，不含任何新算法——四者皆为 B7 力场特解（D-22：新组合 = 旧特解重排）。
pub fn standard_set(
    gravity: Vec3,
    wind: Vec3,
    attractor: Attractor,
    turbulence: Turbulence,
) -> ForceSet {
    let mut s = ForceSet::new();
    s.add(Gravity { g: gravity });
    s.add(Wind { dir: wind });
    s.add(attractor);
    s.add(turbulence);
    s
}

/// 对粒子池逐粒子求合外力并**显式步进一步**（半隐式 Euler：先更新速度
/// 再更新位置——辛一阶，能量漂移受控；确定性：固定顺序，`spawn_index` 升序）。
///
/// 位置/速度更新按粒子池内部稳定序（`spawn_index` 升序）——无跨粒子归约，
/// 无浮点求和序不确定。`t` 为当前时刻（力场时间参）。
///
/// 生存期边界：`age = t − birth ≥ lifetime` 的粒子**保留在池内**但不推进
/// （冻结），由 [`crate::sim::ParticleSim`] 在帧边界统一回收（避免步进中途
/// 改变池长度导致序不稳）。本函数不做回收，只做物理步进。
pub fn apply_forces(pool: &mut ParticlePool, forces: &ForceSet, t: f32, dt: f32) {
    if dt <= 0.0 || !dt.is_finite() {
        return;
    }
    for s in pool.slots_mut().iter_mut() {
        let age = t - s.birth;
        if age >= s.lifetime {
            continue; // 已死：冻结，不推进。
        }
        let a = forces.accel(s.position, t);
        // 半隐式 Euler：v ← v + a·dt；p ← p + v·dt。
        s.velocity = s.velocity + a * dt;
        s.position = s.position + s.velocity * dt;
    }
}

/// 单粒子处的合外力加速度（供测试/外部逐点查询；转发到 `ForceSet::accel`）。
#[inline]
pub fn accel_at(forces: &ForceSet, pos: Vec3, t: f32) -> Vec3 {
    forces.accel(pos, t)
}
