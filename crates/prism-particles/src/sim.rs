//! 确定性推进 + seek 快照（D-28 等价；B8 验收）。
//!
//! # 数学结构（D-22：状态 = 帧号的纯函数）
//!
//! 粒子模拟是**有状态系统**（第 N 帧依赖第 N−1 帧），D-28 以 EvalCtx 快照库
//! 解决 seek。但对粒子而言有更强的结构：**发射与初态是帧号的纯函数**
//! （[`crate::emitter::Emitter::derive`] + [`Emitter::spawn_count`]），
//! 故"N 帧状态"可由"重放 0..N 帧"确定性重建。本模块落地两种等价路径：
//!
//! 1. [`ParticleSim::advance_to`]：顺序推进到帧 `N`（重放）。
//! 2. [`ParticleSim::seek`]：**丢弃当前态**，从空池重放至帧 `N` ——
//!    与顺序推进**逐位一致**（D-28 seek 等价在本域的证明）。
//!
//! 两路径共享同一 `step`（帧级纯函数），故等价性是结构保证而非巧合。
//! 快照语义（D-28）：每步产物可写入 `EvalCtx.snapshots`（键 = (子图签名, 帧)）；
//! 本模块提供 [`ParticleSim::snapshot_bytes`] / [`ParticleSim::restore`] 以纯字节
//! 承载状态——LRU 逐出只影响重算不影响结果（快照缺失时 `seek` 重放即得同值）。

use prism_math::Vec3;

use crate::audio::{modulation_for_frame, ReactiveModulation};
use crate::emitter::{Emitter, ParticlePool, ParticleView};
use crate::force::{apply_forces, ForceSet};
use crate::ParticleError;

/// 烘焙表类型别名（feature `audio` 时 = B6 [`prism_audio::reactive::ReactiveBake`]；
/// 关闭时 = `()` 占位，配合 [`modulation_for_frame`] 的降级分支）。
#[cfg(feature = "audio")]
pub type Bake = prism_audio::reactive::ReactiveBake;
/// 无音频后端的占位类型。
#[cfg(not(feature = "audio"))]
pub type Bake = ();

/// 粒子模拟：发射器 + 力场 + 池的确定性推进器。
///
/// `frame` 记录已推进到的帧号（下一 `step` 将推进到 `frame+1`）。`dt` 为每帧
/// 时长（秒）。可选 AudioReactive 驱动（[`ReactiveBake`] + [`ReactiveModulation`]）。
#[derive(Debug)]
pub struct ParticleSim {
    /// 发射器（参数规格）。
    pub emitter: Emitter,
    /// 共享力场集（复用 B7）。
    pub forces: ForceSet,
    /// 每帧时长（秒）。
    pub dt: f32,
    /// 粒子池容量。
    pub capacity: usize,
    /// 当前帧号（已推进到）。
    frame: u64,
    /// 当前时刻（秒）= frame · dt。
    time: f32,
    /// 已发射粒子总数（累积发射序号，供 Emitter::derive 索引）。
    spawned: u64,
    /// 粒子池（当前活跃态）。
    pool: ParticlePool,
    /// AudioReactive 烘焙（可选）。
    pub bake: Option<crate::sim::Bake>,
    /// AudioReactive 映射规格。
    pub modulation: ReactiveModulation,
}

impl ParticleSim {
    /// 构造（帧 0 空池空时刻）。
    pub fn new(emitter: Emitter, forces: ForceSet, dt: f32, capacity: usize) -> Self {
        Self {
            emitter,
            forces,
            dt,
            capacity,
            frame: 0,
            time: 0.0,
            spawned: 0,
            pool: ParticlePool::new(capacity),
            bake: None,
            modulation: ReactiveModulation::default(),
        }
    }

    /// 当前帧号。
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// 当前时刻（秒）。
    pub fn time(&self) -> f32 {
        self.time
    }

    /// 已发射粒子总数。
    pub fn spawned(&self) -> u64 {
        self.spawned
    }

    /// 粒子池（只读）。
    pub fn pool(&self) -> &ParticlePool {
        &self.pool
    }

    /// 池内当前态投影（供渲染 / 测试）。
    pub fn views(&self) -> Vec<ParticleView> {
        self.pool.views()
    }

    /// 推进一帧（帧级纯函数步：发射 → 力场积分 → 生存期回收）。
    ///
    /// 步序（固定序，确定性）：
    /// 1. 按 AudioReactive 增益决定本帧发射数，逐粒子 `spawn`（出生态定格）；
    /// 2. 逐粒子 `apply_forces`（半隐式 Euler，`spawn_index` 升序）；
    /// 3. 回收 `age ≥ lifetime` 的粒子（帧边界统一回收——不破坏步内序）。
    pub fn step(&mut self) {
        // 1. 发射。
        let gains = modulation_for_frame(self.bake.as_ref(), &self.modulation, self.frame);
        let n = self.emitter.spawn_per_frame(gains.rate, self.dt);
        for _ in 0..n {
            self.pool.spawn(&self.emitter, self.spawned, self.time);
            self.spawned += 1;
        }
        // 2. 力场步进（半隐式 Euler）。
        apply_forces(&mut self.pool, &self.forces, self.time, self.dt);
        // 3. 帧边界回收（age ≥ lifetime）。
        let t = self.time;
        self.pool.slots_mut().retain(|s| (t - s.birth) < s.lifetime);
        // 推进帧号 / 时刻。
        self.frame += 1;
        self.time += self.dt;
    }

    /// 顺序推进到帧 `target`（`target <= frame` 时无操作）。
    pub fn advance_to(&mut self, target: u64) {
        while self.frame < target {
            self.step();
        }
    }

    /// **seek 到帧 `target`**：从空池重放 0..target —— 结果与顺序推进逐位一致
    /// （D-28 seek 等价）。丢弃当前态。力场集由 `forces` 注入（`ForceSet` 为 trait
    /// object 不可克隆，调用方按同规格重建；力场本身纯函数、无隐藏状态，
    /// 同规格重建即等价——D-23「定义跟着代码走」）。
    pub fn seek_with(&self, target: u64, forces: ForceSet) -> Self {
        let mut fresh = Self {
            emitter: self.emitter.clone(),
            forces,
            dt: self.dt,
            capacity: self.capacity,
            frame: 0,
            time: 0.0,
            spawned: 0,
            pool: ParticlePool::new(self.capacity),
            bake: self.bake.clone(),
            modulation: self.modulation.clone(),
        };
        fresh.advance_to(target);
        fresh
    }

    /// **seek 到帧 `target`**（沿用当前力场**无法**拷贝，故重建为空集——需保留
    /// 力场者用 [`Self::seek_with`]）。语义同 [`Self::seek_with`]。
    pub fn seek(&self, target: u64) -> Self {
        let mut fresh = Self {
            emitter: self.emitter.clone(),
            forces: ForceSet::new(),
            dt: self.dt,
            capacity: self.capacity,
            frame: 0,
            time: 0.0,
            spawned: 0,
            pool: ParticlePool::new(self.capacity),
            bake: self.bake.clone(),
            modulation: self.modulation.clone(),
        };
        fresh.advance_to(target);
        fresh
    }

    /// 注入力场（供 seek 后重建；力场纯函数，同规格重建即等价）。
    pub fn set_forces(&mut self, forces: ForceSet) {
        self.forces = forces;
    }

    /// 状态序列化为确定性字节（D-28 快照承载；版本化格式）。
    ///
    /// 格式：`PPPS` + 版本 + (frame, time, spawned, slot_count) + 逐槽定长记录。
    /// 逐位可复现（`to_bits` 存浮点）。
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PPPS"); // Prism Particle Pool Snapshot
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&self.frame.to_le_bytes());
        out.extend_from_slice(&self.time.to_bits().to_le_bytes());
        out.extend_from_slice(&self.spawned.to_le_bytes());
        let slots = self.pool.views();
        out.extend_from_slice(&(slots.len() as u32).to_le_bytes());
        for v in &slots {
            out.extend_from_slice(&v.spawn_index.to_le_bytes());
            out.extend_from_slice(&v.birth.to_bits().to_le_bytes());
            push_vec3(&mut out, v.position);
            push_vec3(&mut out, v.velocity);
            out.extend_from_slice(&v.lifetime.to_bits().to_le_bytes());
            out.extend_from_slice(&v.size.to_bits().to_le_bytes());
            for c in v.color {
                out.extend_from_slice(&c.to_bits().to_le_bytes());
            }
        }
        out
    }

    /// 从 [`Self::snapshot_bytes`] 恢复（帧号/时刻/已发射数/池槽）。
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), ParticleError> {
        let mut r = Reader { buf: bytes, pos: 0 };
        if r.take(4)? != b"PPPS" {
            return Err(err("ParticleSim::restore（magic）", "非 PPPS 快照", "传 snapshot_bytes 输出"));
        }
        let ver = r.u32()?;
        if ver != 1 {
            return Err(err("ParticleSim::restore（version）", format!("版本 {ver} 不支持"), "用版本 1"));
        }
        self.frame = r.u64()?;
        self.time = r.f32()?;
        self.spawned = r.u64()?;
        let count = r.u32()? as usize;
        self.pool = ParticlePool::new(self.capacity);
        for _ in 0..count {
            let spawn_index = r.u64()?;
            let birth = r.f32()?;
            let position = r.vec3()?;
            let velocity = r.vec3()?;
            let lifetime = r.f32()?;
            let size = r.f32()?;
            let mut color = [0.0_f32; 4];
            for c in &mut color {
                *c = r.f32()?;
            }
            // 直接压入池（绕过 derive——快照态已定格）。
            self.pool.push_snapshot(spawn_index, birth, position, velocity, lifetime, size, color);
        }
        Ok(())
    }
}

/// 追加 Vec3（逐分量 to_bits；确定性字节序）。
fn push_vec3(out: &mut Vec<u8>, v: Vec3) {
    out.extend_from_slice(&v.x.to_bits().to_le_bytes());
    out.extend_from_slice(&v.y.to_bits().to_le_bytes());
    out.extend_from_slice(&v.z.to_bits().to_le_bytes());
}

fn err(loc: impl Into<String>, problem: impl Into<String>, fix: impl Into<String>) -> ParticleError {
    ParticleError::new("Pt3001", loc, problem, fix)
}

/// 最小字节读取器（快照反序列化；越界结构化报错）。
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ParticleError> {
        if self.pos + n > self.buf.len() {
            return Err(err("ParticleSim::restore（越界）", "快照字节不足", "检查快照完整性"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ParticleError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, ParticleError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn f32(&mut self) -> Result<f32, ParticleError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn vec3(&mut self) -> Result<Vec3, ParticleError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
}
