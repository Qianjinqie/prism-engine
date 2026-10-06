//! `prism-media` — 多媒体嵌入（工作流 B11；路线图 B11 行 / 白皮书 §19；D-17 读入
//! 策略 / D-16 Mini Code / D-27 静态汇入制）。
//!
//! 素材本已存在，引擎只负责"读入 → 进图 → 被一切能力作用"：不生成、不转码。
//!
//! # 内容
//!
//! - [`source`] — 三类源抽象（image/video/audio_source）+ 读入策略 [`source::ReadKind`]
//!   （D-17：ffmpeg 探测协作 + 原生降级），素材"能不能读"如实标注，永静默不失败；
//! - [`timing`] — 时间映射器（direct/loop/pingpong/变速/Expr 五模式，白皮书 §19：
//!   纯函数映射 + 帧间混合，seek 零成本且音画共享同一 f(t)）；
//! - [`gif`] — GIF 原生解码（自研 LZW 解压 + 帧合成）；
//! - [`apng`] — APNG 原生解码（复用 zlib inflate + PNG 滤波重建）；
//! - [`video`] — ffmpeg 探测读入（自带最小探测 stub：PATH 存在性检测，不实现解码器）；
//! - [`audio`] — 音频源接入（复用 prism-audio WAV 解码）；
//! - [`inflate`] — 自研 DEFLATE/zlib inflate（GIF/APNG 共用，禁第三方）；
//! - [`nodes`] — embed 预置子图节点（`registry!` D-27 静态汇入 prism-core）；
//! - [`nle`] — NLE 非线性编辑（U14：多轨时间线 + 裁剪/变速/倒放时间映射 +
//!   转场混合 + §26 固定形状音频归约）。
//!
//! # 确定性契约
//!
//! 一切公开函数都是其参数的纯函数：无内部可变状态、无全局 RNG、无 libm、无网络。
//! 解码按固定字节序遍历、时间映射为闭式纯函数——同输入跨平台逐位一致。
//!
//! # 边界（D-17）
//!
//! mp4/mov/webm 等解码不自研：经运行时探测 ffmpeg 协作（有则增强、无则降级为
//! 原生可解格式 GIF/APNG/序列帧/WAV）；编码维持 D-10 不自研。

pub mod apng;
pub mod audio;
pub mod gif;
pub mod inflate;
pub mod nle;
pub mod nodes;
pub mod source;
pub mod timing;
pub mod video;

pub use audio::load_audio;
pub use gif::{decode_gif, GifFrame, GifImage};
pub use apng::{decode_apng, ApngFrame, ApngImage};
pub use nodes::__prism_collect;
pub use nle::{
    composite, mixdown, AudioTrackInput, BlendMode, Clip, ClipId, Frame, MediaClipSource, NleError,
    NleResult, RateCurve, Timeline, Track, TrackIndex, TrackKind, Transform, Transition,
    TransitionEasing, TransitionId, TransitionKind,
};
pub use source::{MediaSource, PixelImage, ReadKind, SourceError, SourceKind};
pub use timing::{TimeMap, TimingMode};

use std::fmt;

/// 媒体域错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
///
/// 错误码分段：`MD1xxx` 源与读入、`GIF1xxx` GIF 结构、`APNG1xxx` APNG 结构、
/// `INF1xxx` DEFLATE/zlib 结构、`VID1xxx` ffmpeg 探测。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl MediaError {
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

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for MediaError {}

/// 媒体域 `Result` 别名。
pub type MediaResult<T> = Result<T, MediaError>;
