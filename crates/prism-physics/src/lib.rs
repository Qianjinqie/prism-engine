//! `prism-physics` — B7 物理域（XPBD 统一求解器）。
//!
//! **当前是骨架（与 A4 DSL 并行）**：SimSession 框架（[`session`]）+
//! XPBD 求解器核心（[`xpbd`]）+ 单测。刚体/软体/布料/SPH/破碎实装与关节族
//! 是后续阶段装内容（D-18：统一走 XPBD，替代 Bullet 式
//! 双求解器老路）——经两个扩展位接入，当前不越界实装：
//!
//! - [`xpbd::Constraint`] trait：关节/密度（PBF）/形状匹配/碰撞约束族实现
//!   本 trait 即入求解循环；
//! - [`session::SimSpec`]：初态与参数规格，约束族与 ForceField 在此扩展。
//!
//! 语义权威：XPBD 统一求解、压缩映射 → 收敛保证、
//! D-9 / D-28（SimSession + 快照判失效）、
//! （seek 等价）。零第三方依赖（铁律一）；确定性纪律见 [`xpbd`] 模块文档。

use std::fmt;

pub mod cloth;
pub mod coupling;
pub mod fracture;
pub mod hair;
pub mod joints;
pub mod session;
pub mod smoke;
pub mod smoke_sparse;
pub mod sph;
pub mod xpbd;

/// 物理域结构化错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19：必含
/// 建议段）。错误码分段对齐 `prism-core::eval::EvalError`：P1xxx 防呆参数 /
/// P3xxx 结构错误。
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：发生位置。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for PhysicsError {}
