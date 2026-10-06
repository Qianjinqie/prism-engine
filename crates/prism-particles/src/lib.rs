//! `prism-particles` — B8 粒子系统（Emitter / 力场 / 点精灵渲染 / AudioReactive）。
//!
//! 语义权威：《强大版路线图》§399（B8 行：Emitter/力场/点精灵渲染/AudioReactive）
//! 与 Phase 2 验收（Cpu 可跑）、D-16（Mini Code 原则）、D-22（数学结构第一性：
//! 粒子 = 状态纯函数 + 力场特解装配）、D-28（快照等价 / `SampleIndex` 像素域）、
//! D-29（禁 libm：超越函数自研入 prism-math）。
//!
//! # 红线（贯穿全 crate）
//!
//! - **零第三方依赖**（铁律一）：仅 std + workspace 内 crate。
//! - **无全局 RNG**（架构 §2.6.3）：粒子状态纯函数化——粒子 `i` 的初态由
//!   `hash(seed, i)` 派生（[`emitter`]）；湍流场用 (pos, t) 哈希派生
//!   （复用 [`prism_physics::coupling::Turbulence`]）。同种子两遍逐位一致。
//! - **复用 B6/B7 既有实现勿重复造**：力场族复用
//!   [`prism_physics::coupling`]（`ForceField`/`Gravity`/`Wind`/`Attractor`/
//!   `Turbulence`）；AudioReactive 参数通道复用 [`prism_audio::reactive`]
//!   （`ReactiveBake` 查表）。本 crate 只做**粒子域的特解装配**。
//!
//! # 模块
//!
//! - [`emitter`]：粒子池 + Emitter（发射率/初速度分布/生存期/尺寸曲线，
//!   全部由 hash(种子, 粒子索引) 派生）。
//! - [`force`]：力场接入（复用 B7 力场族；[`force::apply_forces`] 逐粒子求值）。
//! - [`render`]：点精灵渲染（对接 [`prism_render::tile`] tile 协议；
//!   粒子→像素覆盖，RGBA16F）。
//! - [`audio`]：AudioReactive 通道（复用 B6 烘焙表驱动发射率/速度/尺寸）。
//! - [`nodes`]：节点注册（D-27 静态汇入制）。
//! - [`sim`]：确定性推进 + seek 快照（D-28 等价）。

pub mod audio;
pub mod emitter;
pub mod force;
pub mod gpu_sim_ref;
pub mod nodes;
pub mod render;
pub mod sim;

use std::fmt;

/// 粒子域结构化错误（对齐宪章 C18 三段式；错误码分段 Pt1xxx 防呆 /
/// Pt3xxx 结构错误，与 prism-physics 的 P3xxx 段并列隔离）。
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：发生位置。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl ParticleError {
    /// 构造三段式错误。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { code, location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl fmt::Display for ParticleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for ParticleError {}

/// 本 crate 结果别名。
pub type ParticleResult<T> = Result<T, ParticleError>;

pub use emitter::{Emitter, ParticlePool, ParticleState};
pub use nodes::__prism_collect;
pub use sim::ParticleSim;
