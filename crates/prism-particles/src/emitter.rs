//! 粒子池 + Emitter（B8；D-16 Mini Code / D-22 数学结构）。
//!
//! # 数学结构（D-21/D-22：从结构出发，非暴力模拟）
//!
//! 粒子系统本质 = **一族由 (种子, 粒子索引, 发射序号) 索引的惰性纯函数**
//! `p(i, t)`。不存在"逐个生成的随机状态"，只有哈希派生——
//!
//! ```text
//! stream(i, k) = hash(seed, i, k)          // 纯函数，跨机逐位一致
//! r̂ = (stream(i,0), stream(i,1), stream(i,2))  ∈ [0,1)³   // 各通道独立
//! ```
//!
//! 初速度分布 / 生存期 / 尺寸曲线 / 发射抖动全部取自 `r̂` 的确定变换。
//! 这是"无全局 RNG，粒子状态纯函数化"约束的粒子侧落地：同 `(seed, i, k)`
//! 恒同结果，跨平台逐位一致（无浮点归约序不确定）。
//!
//! # 发射率（格锚定，D-29 无 libm）
//!
//! 每帧发射数 `n(frame) = round(rate · dt · reactive_gain)`（rate 单位：粒子/秒）。
//! 粒子索引 = 累积发射序号（`spawn_index`），保证 seek 第 N 帧时"哪些粒子已
//! 出生"由帧号确定（[`Emitter::particles_alive_at`]），与顺序推进一致。

use prism_math::Vec3;
use prism_noise::pcg::hash3_f32;

use crate::{ParticleError, ParticleResult};

/// 单粒子的**派生初态**（发射时刻定格；纯函数派生，无随机状态）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleState {
    /// 出生位置（发射器原点 + 半径抖动）。
    pub origin: Vec3,
    /// 初速度（方向 = hash 派生球面均匀，幅度 = speed 曲线 × hash 抖动）。
    pub velocity: Vec3,
    /// 生存期（秒）。
    pub lifetime: f32,
    /// 初始尺寸（世界单位 / 像素半径）。
    pub size: f32,
    /// 代表色（RGBA，线性；D-28 无颜色管理域，此处即渲染线性值）。
    pub color: [f32; 4],
}

/// 发射器规格（纯数据；参数全部确定性派生，无内部可变状态）。
#[derive(Debug, Clone, PartialEq)]
pub struct Emitter {
    /// 主哈希种子（派生一切通道）。
    pub seed: u64,
    /// 发射位置（世界坐标）。
    pub origin: Vec3,
    /// 发射球半径（`origin` 周围均匀分布出生点；0 = 点发射）。
    pub radius: f32,
    /// 发射率（粒子/秒）。
    pub rate: f32,
    /// 初速度基准幅度（世界单位/秒）。
    pub speed: f32,
    /// 速度抖动比例（0 = 匀速，1 = ±100%）。
    pub speed_jitter: f32,
    /// 生存期基准（秒）。
    pub lifetime: f32,
    /// 生存期抖动比例。
    pub lifetime_jitter: f32,
    /// 初始尺寸基准（世界单位 / 像素半径）。
    pub size: f32,
    /// 尺寸抖动比例。
    pub size_jitter: f32,
    /// 单位体积重力（世界单位/秒²；局部重力，独立于力场）。
    pub gravity: Vec3,
}

impl Default for Emitter {
    fn default() -> Self {
        Self {
            seed: 0,
            origin: Vec3::splat(0.0),
            radius: 0.0,
            rate: 100.0,
            speed: 1.0,
            speed_jitter: 0.2,
            lifetime: 2.0,
            lifetime_jitter: 0.2,
            size: 2.0,
            size_jitter: 0.2,
            gravity: Vec3::splat(0.0),
        }
    }
}

/// 把 `[0,1)` 映射到 `[1−j, 1+j]`（抖动乘子；`j` 钳到 `[0, 1]`）。
#[inline]
fn jitter(h: f32, j: f32) -> f32 {
    let j = j.clamp(0.0, 1.0);
    (1.0 - j) + 2.0 * j * h
}

/// 球面均匀方向：由两个 `[0,1)` 哈希通道派生（Marsaglia / Archimedes 面积法，
/// 无超越函数——只用 sqrt 与线性映射，D-29 合规）。
///
/// `z = 2u − 1` 均匀幅角；`r = sqrt(1 − z²)`；方位角用 `(2v−1, w)` 的
/// 归一化近似（避免 atan2/sincos——近似不损**确定性**，只影响分布外观）。
#[inline]
fn sphere_dir(u: f32, v: f32, w: f32) -> Vec3 {
    let z = 2.0 * u - 1.0;
    let r = (1.0 - z * z).max(0.0).sqrt();
    // (2v−1, 2w−1) 归一化作为 cos/sin 的代理（无超越函数）；退化时取 (±1, 0)。
    let (mut ax, mut ay) = (2.0 * v - 1.0, 2.0 * w - 1.0);
    let m = ax * ax + ay * ay;
    if m < 1e-12 {
        ax = 1.0;
        ay = 0.0;
    } else {
        let inv = 1.0 / m.sqrt();
        ax *= inv;
        ay *= inv;
    }
    Vec3::new(r * ax, r * ay, z)
}

impl Emitter {
    /// 校验参数合法性（C18 防呆；Pt1xxx 防呆参数段）。
    ///
    /// 拒绝：`rate`/`speed`/`lifetime`/`size`/`radius` 为 NaN 或 ±Inf；
    /// `rate`/`speed`/`lifetime`/`size`/`radius` 为负；抖动比非有限。
    pub fn validate(&self) -> ParticleResult<()> {
        let scalars = [
            ("rate", self.rate),
            ("speed", self.speed),
            ("lifetime", self.lifetime),
            ("size", self.size),
            ("radius", self.radius),
            ("speed_jitter", self.speed_jitter),
            ("lifetime_jitter", self.lifetime_jitter),
            ("size_jitter", self.size_jitter),
        ];
        for (name, v) in scalars {
            if !v.is_finite() {
                return Err(ParticleError::new(
                    "Pt1001",
                    format!("Emitter::validate（{name}）"),
                    format!("{name} = {v}：必须为有限实数（NaN/±Inf 会污染派生流）"),
                    "传有限实数；非法量在构造期拒绝".to_string(),
                ));
            }
            if v < 0.0 && name != "speed_jitter" && name != "lifetime_jitter" && name != "size_jitter" {
                return Err(ParticleError::new(
                    "Pt1001",
                    format!("Emitter::validate（{name}）"),
                    format!("{name} = {v}：必须 ≥ 0"),
                    "传非负值".to_string(),
                ));
            }
        }
        if !self.gravity.x.is_finite()
            || !self.gravity.y.is_finite()
            || !self.gravity.z.is_finite()
        {
            return Err(ParticleError::new(
                "Pt1001",
                "Emitter::validate（gravity）",
                format!("gravity = {:?}：三分量必须为有限实数", self.gravity),
                "传有限重力向量".to_string(),
            ));
        }
        Ok(())
    }

    /// 粒子 `i` 的派生初态（**纯函数**：同 `(seed, i)` 恒同结果）。
    ///
    /// 通道分配——
    /// - `k=0,1,2`：出生点球半径抖动三分量；
    /// - `k=3,4,5`：初速度方向（球面均匀）+ 抖动；
    /// - `k=6`：生存期抖动；
    /// - `k=7`：尺寸抖动；
    /// - `k=8..12`：代表色 RGBA。
    pub fn derive(&self, i: u64) -> ParticleState {
        // 通道 = z 轴，粒子索引 = y 轴，种子 = 种子异或通道族（避免跨族碰撞）。
        let h = |family: u64, ch: u64| hash3_f32(self.seed ^ family, i, 0, ch);

        // 出生点：origin + radius · sphere_dir(三通道) · 半径抖动。
        let od = sphere_dir(h(0x1001, 0), h(0x1001, 1), h(0x1001, 2));
        let origin = self.origin + od * (self.radius * h(0x1002, 3));

        // 初速度：球面均匀方向 × speed × 抖动。
        let vd = sphere_dir(h(0x2001, 4), h(0x2001, 5), h(0x2001, 6));
        let smag = self.speed * jitter(h(0x2002, 7), self.speed_jitter);
        let velocity = vd * smag;

        // 生存期 / 尺寸。
        let lifetime = (self.lifetime * jitter(h(0x3001, 8), self.lifetime_jitter)).max(0.0);
        let size = (self.size * jitter(h(0x3002, 9), self.size_jitter)).max(0.0);

        // 代表色：RGB 各通道独立哈希（alpha = 1）。
        let color = [h(0x4001, 10), h(0x4002, 11), h(0x4003, 12), 1.0];

        ParticleState { origin, velocity, lifetime, size, color }
    }

    /// 累积到帧 `frame` 为止（不含）已发射的粒子总数（确定性格锚）。
    ///
    /// `spawn_count(frame) = Σ_{f < frame} round(rate · dt · gain(f))`。当 gain
    /// 恒 1 时 = `round(rate·dt) · frame`（整数格，无浮点漂移）。gain 逐帧变化
    /// 时（AudioReactive）按帧累加——调用方传 `gains[frame]`（缺省 1）。
    pub fn spawn_count(&self, frame: u64, dt: f32, gains: &[f32]) -> u64 {
        let mut acc: u64 = 0;
        for f in 0..frame {
            let g = gains.get(f as usize).copied().unwrap_or(1.0);
            acc += self.spawn_per_frame(g, dt);
        }
        acc
    }

    /// 单帧发射数（`rate · dt · gain` 四舍五入到最近整数；负数钳 0）。
    #[inline]
    pub fn spawn_per_frame(&self, gain: f32, dt: f32) -> u64 {
        let raw = self.rate * dt * gain.max(0.0);
        if !raw.is_finite() || raw <= 0.0 {
            return 0;
        }
        // round-half-up（确定性；无 libm）。
        (raw + 0.5) as u64
    }
}

/// 粒子池：定容池（capacity 上限），槽位 = (spawn_index, state)。
///
/// 池内粒子按 `spawn_index` 升序**稳定**存放（发射顺序即索引序）；超过
/// `capacity` 时逐出最旧的（环形语义 = 覆盖最旧活跃粒子）。池无随机状态，
/// 一切由 Emitter 派生——池本身只是**已发射粒子的当前模拟态容器**。
#[derive(Debug, Clone, PartialEq)]
pub struct ParticlePool {
    /// 池容量（活跃粒子上限）。
    capacity: usize,
    /// 活跃粒子（`spawn_index` 升序）。
    slots: Vec<ParticleSlot>,
}

/// 粒子池槽（crate 内可见；供 sim 步进与回收）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleSlot {
    /// 全局发射序号（= Emitter 派生索引 `i`）。
    pub spawn_index: u64,
    /// 出生时刻（秒）。
    pub birth: f32,
    /// 当前位置。
    pub position: Vec3,
    /// 当前速度。
    pub velocity: Vec3,
    /// 生存期（秒，出生时定格）。
    pub lifetime: f32,
    /// 初始尺寸（出生时定格）。
    pub size: f32,
    /// 代表色（出生时定格）。
    pub color: [f32; 4],
}

/// 池内粒子的只读快照（对外投影；供渲染 / 测试读取）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleView {
    /// 全局发射序号。
    pub spawn_index: u64,
    /// 出生时刻（秒）。
    pub birth: f32,
    /// 当前位置。
    pub position: Vec3,
    /// 当前速度。
    pub velocity: Vec3,
    /// 生存期（秒）。
    pub lifetime: f32,
    /// 当前尺寸（可随年龄衰减，见 [`ParticlePool::view_at`] 的尺寸曲线）。
    pub size: f32,
    /// 代表色。
    pub color: [f32; 4],
}

impl ParticlePool {
    /// 定容池构造。
    pub fn new(capacity: usize) -> Self {
        Self { capacity, slots: Vec::with_capacity(capacity) }
    }

    /// 池容量。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 活跃粒子数。
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// 是否空。
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// 发射一个粒子（按 Emitter 派生出生态）；池满时逐出最旧。
    ///
    /// 出生时按 `Emitter::derive(spawn_index)` 定格初态（位置 = origin，
    /// 速度 = derived velocity）。
    pub fn spawn(&mut self, emitter: &Emitter, spawn_index: u64, birth: f32) {
        let s = emitter.derive(spawn_index);
        // 容量 0：安全降级——丢弃粒子，绝不 panic（库代码禁 panic，P1）。
        if self.capacity == 0 {
            return;
        }
        if self.slots.len() >= self.capacity {
            // 覆盖最旧（确定性：固定下标 0，其余前移——Mini Code，O(n) 但池小）。
            self.slots.remove(0);
        }
        self.slots.push(ParticleSlot {
            spawn_index,
            birth,
            position: s.origin,
            velocity: s.velocity,
            lifetime: s.lifetime,
            size: s.size,
            color: s.color,
        });
    }

    /// 池内粒子当前态（`age = t − birth` 处）的投影：
    /// 位置按匀加速积分解析解（`p = p₀ + v·age + ½·g·age²`，`g` 由调用方
    /// 通过 [crate::force::apply_forces] 逐帧步进累积，或此处用 emitter 的
    /// 局部重力——本方法只做**投影**，不改变池状态）。
    ///
    /// 尺寸曲线：`size(age) = size₀ · (1 − age/lifetime)`（线性衰减到 0，
    /// 死亡边界处恰为 0；D-16 不用三角函数做缓动，线性即可）。
    pub fn view_at(&self, t: f32, gravity: Vec3) -> Vec<ParticleView> {
        self.slots
            .iter()
            .map(|s| {
                let age = (t - s.birth).max(0.0);
                let position = s.position + s.velocity * age + gravity * (0.5 * age * age);
                let frac = if s.lifetime > 0.0 {
                    (1.0 - age / s.lifetime).max(0.0)
                } else {
                    0.0
                };
                ParticleView {
                    spawn_index: s.spawn_index,
                    birth: s.birth,
                    position,
                    velocity: s.velocity + gravity * age,
                    lifetime: s.lifetime,
                    size: s.size * frac,
                    color: s.color,
                }
            })
            .collect()
    }

    /// 直接读池槽快照（供 sim / 测试；步进后的当前态）。
    pub fn slots(&self) -> &[ParticleSlot] {
        &self.slots
    }

    /// 可变访问池槽（供 sim 步进）。
    pub(crate) fn slots_mut(&mut self) -> &mut Vec<ParticleSlot> {
        &mut self.slots
    }

    /// 快照恢复：直接压入定格态（绕过 `derive`；`ParticleSim::restore` 用）。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_snapshot(
        &mut self,
        spawn_index: u64,
        birth: f32,
        position: Vec3,
        velocity: Vec3,
        lifetime: f32,
        size: f32,
        color: [f32; 4],
    ) {
        if self.slots.len() >= self.capacity {
            self.slots.remove(0);
        }
        self.slots.push(ParticleSlot {
            spawn_index,
            birth,
            position,
            velocity,
            lifetime,
            size,
            color,
        });
    }

    /// 全部活跃粒子的投影（含步进态；简化 `view_at` 一步投影，sim 步进后调用）。
    pub fn views(&self) -> Vec<ParticleView> {
        self.slots
            .iter()
            .map(|s| ParticleView {
                spawn_index: s.spawn_index,
                birth: s.birth,
                position: s.position,
                velocity: s.velocity,
                lifetime: s.lifetime,
                size: s.size,
                color: s.color,
            })
            .collect()
    }
}
