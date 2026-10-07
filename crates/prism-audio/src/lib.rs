//! `prism-audio` — 音频全家（工作流 B6；；
//! S6 谱分解 + S7 闭式线性响应 + S9 分段多项式的特解宿主）。
//!
//! 内容：
//!
//! - [`buffer`] — PCM 缓冲（交错 f32；混合/缩放/峰值 = 固定序纯函数）；
//! - [`wav`] — WAV 读写（16/24/32 bit PCM，逐字节往返一致）；
//! - [`fft`] — radix-2 Cooley-Tukey FFT（位反转置换 + 蝶形，自研禁第三方）；
//! - [`synth`] — 振荡器族（PolyBLEP 防混叠）+ ADSR 闭式包络 + 4 情绪预设；
//! - [`theory`] — 音阶库（大小/五声/中古调式）+ 音级↔频率 12-TET 查表；
//! - [`harmony`] — 和弦进行引擎（功能语法状态机）+ 声部引导规则；
//! - [`arp`] — 琶音器（模式/速率/八度范围，确定性派生）；
//! - [`drums`] — step 鼓机（步进网格 + 音色映射，接 synth）；
//! - [`midi`] — MIDI 文件导入（手写 SMF 解析：MThd/MTrk/事件/VLQ）；
//! - [`score`] — 配乐编排（小节/节拍网格/声部叠加，确定性配乐段）；
//! - [`beats`] — 节拍/onset 检测（频谱通量正部 + 自适应阈值）；
//! - [`reactive`] — AudioReactive 烘焙（第 N 帧读 `[N·hop, N·hop+win)` 窗，
//!   帧级纯函数，seek 等价）；
//! - [`mixer`] — BGM 混音（多轨线性混合 + 淡入淡出 + 峰值防削波）；
//! - [`nodes`] — audio/reactive 节点（`registry!` D-27 静态汇入 prism-core）。
//!
//! # 确定性契约
//!
//! 一切公开函数都是其参数的纯函数：无内部可变状态、无全局 RNG（随机性一律
//! 经 `prism-noise` PCG 整数哈希派生）、无 libm。sin/cos 由 [`det`]
//! 以固定系数多项式实现（f64 内部运算，IEEE 逐运算确定，D-29——与
//! `prism-anim`/`prism-geometry` 的 crate 内自含过渡实现同式，待 prism-math
//! 确定性数学件落地后统一收编）。因此**同输入跨平台逐位一致**。
//!
//! 边界：EBU R128 响度归一属 C4，不在本包。

pub mod arp;
pub mod beats;
pub mod buffer;
pub mod det;
pub mod drums;
pub mod fft;
pub mod fx;
pub mod granular;
pub mod harmony;
pub mod midi;
pub mod mixer;
pub mod nodes;
pub mod physical;
pub mod reactive;
pub mod sampler;
pub mod score;
pub mod sfz;
pub mod spatial;
pub mod synth;
pub mod theory;
pub mod wav;

pub use buffer::PcmBuffer;
pub use harmony::{progression, voice_lead, Chord, ChordQuality, Function, Progression, ProgressionSpec};
pub use nodes::__prism_collect;
pub use reactive::{ReactiveBake, ReactiveConfig};
pub use synth::presets::Mood;
pub use theory::{scale, Key, Scale, ScaleKind};
pub use wav::{decode_wav, encode_wav, BitDepth};

use std::fmt;

/// 音频域错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
///
/// 错误码分段：`AU1xxx` 缓冲/参数、`WAV1xxx` WAV 结构与格式、`MIX1xxx`
/// 混音轨道、`SCN1xxx` 配乐编排。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl AudioError {
    /// 构造一个三段式错误。
    pub fn new(
        code: &'static str,
        location: impl Into<String>,
        problem: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self { code, location: location.into(), problem: problem.into(), fix: fix.into() }
    }
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}] {}\n  什么事：{}\n  建议：{}", self.code, self.location, self.problem, self.fix)
    }
}

impl std::error::Error for AudioError {}

/// 音频域 `Result` 别名。
pub type AudioResult<T> = Result<T, AudioError>;
