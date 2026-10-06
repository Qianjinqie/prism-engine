//! `prism-aesthetic` — 二级调色与曲线（B14：HSL 选区 / RGB 曲线 /
//! lift-gamma-gain / Power Window + 二级调色曲线库）。
//!
//! 语义权威：《强大版路线图》B14 行（二级调色：HSL 选区 / RGB 曲线 /
//! lift-gamma-gain / Power Window）；《算法白皮书》§26（像素内固定序归约）；
//! 架构 v2.4 §4.3/§4.4（tile 无关性、唯一随机性入口）。
//!
//! # 组成
//!
//! - [`secondary`] — 二级调色：HSL 亮度/色相/饱和选区 + lift-gamma-gain +
//!   Power Window（矩形/椭圆窗口羽化蒙版），选区数学有断言；
//! - [`curves`] — 曲线库：单调分段线性 RGB 曲线 + 主/红/绿/蓝四通道；
//! - [`critic`] — critic 闭环（C5）：四维可计算指标（曝光/对比/色彩/清晰度 +
//!   构图）+ 评分 + 诊断 + 增量重渲闭环 + 质量回升判定。
//!
//! # 确定性纪律（D-29）
//!
//! 零超越函数：只用 `+ - × ÷`、IEEE 精确的 `floor`/`sqrt`/`abs` 与整数
//! 运算；幂（gamma）用整数指数连乘的实现（`pow_int`），不使用 `powf`。
//! 全部算子为求值纯函数，无隐藏状态、无私开 RNG。

pub mod cmd;
pub mod critic;
pub mod critic_loop;
pub mod curves;
pub mod nodes;
pub mod secondary;

pub use cmd::{CriticCommand, CritiqueReport, Phase};
pub use critic::{
    analyze, diagnose, score, Diagnosis, Dim, Direction, Metrics, Rubric, Score, Target,
};
pub use critic_loop::{
    apply_suggestion, convergence, critique_frame, suggest, Convergence, CriticLoop,
    MetricsAccumulator, RoundReport, Suggestion, DEFAULT_EPS,
};
pub use curves::{Curve, CurveSet, CurvePoint};
pub use nodes::{__prism_collect, QualityScoreNode};
pub use secondary::{
    HslQualifier, LiftGammaGain, PowerWindow, Secondary, WindowShape,
};

/// Rec.709 亮度权重（二级调色选区与调色共用）。
pub(crate) const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// 整数幂（连乘，零超越函数；负指数取倒数）。`exp = 0` 恒返回 1。
///
/// D-29：整数幂一律连乘不用 `powf`——跨平台逐位一致。
#[inline]
pub(crate) fn pow_int(base: f32, exp: i32) -> f32 {
    if exp == 0 {
        return 1.0;
    }
    let neg = exp < 0;
    let n = exp.unsigned_abs();
    let mut out = 1.0f32;
    for _ in 0..n {
        out *= base;
    }
    if neg {
        1.0 / out
    } else {
        out
    }
}

/// 逐像素亮度（Rec.709）。
#[inline]
pub(crate) fn luma(c: [f32; 4]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}
