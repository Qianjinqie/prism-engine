//! `prism-mux` — 容器交付（工作流 C4； 容器路线；D-10 容器/编码
//! 决策 / D-16 Mini Code / D-17 ffmpeg 运行时协作 / D-29 禁 libm）。
//!
//! 引擎内置终态容器链（D-10：**永远可用、零外部依赖**）：
//!
//! `Framebuffer(RGBA8)` → `JPEG`（灰度最小基线）→ `AVI(MJPEG)` 或 `MP4(MJPEG)+PCM`。
//!
//! H.264/HEVC **不自研编码器**（设计约束）；由 [`extprobe`] 运行时探测外部 ffmpeg 或
//! [`hwenc`] 硬件编码器 API 协作，可用则交付 h264、不可用自动降级 MJPEG。
//!
//! # 内容
//!
//! - [`jpeg`] — 基线 JPEG 编码（灰度最小基线：量化 + Huffman + 位流装配）；
//! - [`avi`] — AVI(rawvideo) 写入（"双击可播"底线证明）；
//! - [`boxes`] — ISO-BMFF 盒原语（ftyp/moov/trak/stbl/mdat，手写）；
//! - [`mp4`] — MP4 容器写入（MJPEG 视频轨 + PCM 音频轨）；
//! - [`loudnorm`] — EBU R128 响度归一（BS.1770 K 加权 + 双门控）；
//! - [`extprobe`] — ffmpeg 探测器（PATH 检测 + 编码器能力探测 + 自动降级链，C7）；
//! - [`hwenc`] — 硬件编码器适配层（VideoToolbox/MediaFoundation/NVENC/VAAPI，骨架）。
//! - [`rate_control`] — 帧级率控（water-filling 位分配 + 等失真证书 + B 帧闭式解）。
//!
//! # 确定性契约
//!
//! 一切公开函数都是其参数的纯函数（写入器接收 `&[u8]` 返回 `Vec<u8>`）：无内部
//! 可变状态、无全局 RNG、无 libm（D-29）、无网络。JPEG/MJPEG 逐 tile 编码、AVI/MP4
//! 按固定盒序遍历——同输入跨平台逐字节一致。
//!
//! # 边界
//!
//! H.264/HEVC 编码维持 D-10 不自研（设计约束）；[`extprobe`]/[`hwenc`] 只做探测与
//! 适配骨架，真机编码由外部可执行文件或厂商 API 执行，超本 crate 范围。

pub mod avi;
pub mod boxes;
pub mod extprobe;
pub mod hwenc;
pub mod jpeg;
pub mod loudnorm;
pub mod mp4;
pub mod rate_control;
pub mod video;
/// V3.1 编码 IO：零拷贝 RGBA8 帧视图 + 视图版 MP4(MJPEG) 出口。
pub mod view;

pub use avi::{
    write_avi, write_avi_jpeg, write_avi_mjpeg, write_avi_mjpeg_audio, write_avi_mjpeg_audio_frames,
    AviConfig,
};
pub use mp4::{write_mp4, write_mp4_jpeg, write_mp4_jpeg_color, TrackSpec};
pub use jpeg::{encode_jpeg_gray, JpegConfig};
pub use video::{decode_jpeg_color, decode_mp4_video, write_mp4_video, VideoConfig};

use std::fmt;

/// 交付域错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改；C19 必含建议段）。
///
/// 错误码分段：`MUX1xxx` 容器结构/参数、`JPG1xxx` JPEG 编码、`LOUD1xxx` 响度。
#[derive(Debug, Clone, PartialEq)]
pub struct MuxError {
    /// 错误码。
    pub code: &'static str,
    /// 三段式一【在哪】：模块 / 对象。
    pub location: String,
    /// 三段式二【什么事】：检测到的事实。
    pub problem: String,
    /// 三段式三【怎么改】：可行动建议。
    pub fix: String,
}

impl MuxError {
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

impl fmt::Display for MuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for MuxError {}

/// 交付域 `Result` 别名。
pub type MuxResult<T> = Result<T, MuxError>;
