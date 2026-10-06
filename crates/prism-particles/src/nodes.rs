//! 粒子节点（B8 → prism-core 注册表；D-27 静态汇入制）。
//!
//! 节点 = 三元组的实例形态：本文件实现 [`MathStructure`] 的粒子特解，经
//! `registry!`（宏展开为 `__prism_collect(reg)` 注册函数）静态汇入
//! `NodeRegistry`——可执行 crate（prism-app/用户 bin）启动时汇入一次，
//! 无动态加载、无脚本层（D-27）。
//!
//! # 编号口径
//!
//! 走用户段 `StructureId::custom`（S15/S16 @ prism-particles）：官方段 S1–S12
//! 为全局单例，B8 的粒子力场装配与点精灵渲染以自定义段挂接，跨 crate 编号
//! 天然隔离（D-23）。三问：核 = 复合映射（力场装配 / splat 核），对偶 = 域对偶
//! （粒子域 / 像素域），格锚 = 帧号 × 哈希种子整数格（帧级纯函数）。
//!
//! # 数据接线（架构 §4.4）
//!
//! 求值纯函数契约：无隐藏状态、无全局 RNG。粒子采样输出 `Payload::Scalar`
//! （粒子索引处的尺寸/寿命环），供下游 DSL 组合；力场加速度输出 `Payload::Vec3`。

use prism_core::node::{
    DualityPair, EvalCtx, KernelKind, LatticePlan, MathStructure, OpCapabilities, Payload,
    StructureId, StructureMeta,
};

use crate::audio::modulation_for_frame;
use crate::emitter::Emitter;
use crate::force::ForceSet;
use crate::sim::Bake;

/// 粒子三问元数据（转移核 × 样本-测度对偶 × 格锚）。
fn particle_meta(n: u16) -> StructureMeta {
    StructureMeta::new(
        StructureId::custom(n, env!("CARGO_CRATE_NAME")),
        KernelKind::Transfer,
        DualityPair::SampleMeasure,
        LatticePlan::LatticeAnchored,
        OpCapabilities { invert: false, sample: false },
    )
}

/// ParticleEmitter 节点（S15 @ prism-particles）：按帧与粒子索引派生粒子
/// 寿命环 / 尺寸（纯函数）。输入 `Payload::Scalar(i)` = 粒子索引（取整）；
/// 输出 `Payload::Scalar` = 该粒子在当前帧的尺寸（来自 Emitter 派生）。
#[derive(Debug)]
pub struct ParticleEmitterNode {
    meta: StructureMeta,
}

impl Default for ParticleEmitterNode {
    fn default() -> Self {
        Self { meta: particle_meta(15) }
    }
}

impl MathStructure for ParticleEmitterNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload {
        let i = match x {
            Payload::Scalar(v) => {
                if !v.is_finite() || *v < 0.0 {
                    return Payload::Empty;
                }
                *v as u64
            }
            _ => return Payload::Empty,
        };
        // Emitter 由 ctx.global_seed 作种子（节点无隐藏参数；参数经 DSL 组合
        // 注入，本层以种子确定性派生——D-28：一切随机源 = hash 派生）。
        let emitter = Emitter { seed: ctx.global_seed, ..Emitter::default() };
        Payload::Scalar(emitter.derive(i).size)
    }
}

/// ParticleForce 节点（S16 @ prism-particles）：力场在 `Payload::Vec3` 输入
/// 位置处的合外力加速度（复用 B7 [`ForceSet`]，纯函数）。输入 `Payload::Vec3`
/// = 位置；输出 `Payload::Vec3` = 加速度。力场集为空 → 零向量。
#[derive(Debug)]
pub struct ParticleForceNode {
    meta: StructureMeta,
}

impl Default for ParticleForceNode {
    fn default() -> Self {
        Self { meta: particle_meta(16) }
    }
}

impl MathStructure for ParticleForceNode {
    fn meta(&self) -> &StructureMeta {
        &self.meta
    }

    fn apply(&self, x: &Payload, ctx: &EvalCtx) -> Payload {
        let pos = match x {
            Payload::Vec3(p) => *p,
            _ => return Payload::Empty,
        };
        // 力场集为空（默认节点无外部装配）——加速度为零；外部装配走
        // ParticleSim（有状态推进器）。本节点仅暴露纯函数接口面。
        let forces = ForceSet::new();
        let a = forces.accel(pos, ctx.frame as f32);
        Payload::Vec3(a)
    }
}

/// AudioReactive 驱动增益查询辅助（供测试/接线；不注册为节点——
/// 频段查询节点已由 B6 `AudioReactiveNode` 提供，禁重复）。
pub fn reactive_rate_gain(bake: Option<&Bake>, modu: &crate::audio::ReactiveModulation, frame: u64) -> f32 {
    modulation_for_frame(bake, modu, frame).rate
}

prism_core::registry! {
    structure ParticleEmitterNode => StructureId::custom(15, env!("CARGO_CRATE_NAME"));
    structure ParticleForceNode => StructureId::custom(16, env!("CARGO_CRATE_NAME"));
}
