//! 音频源接入（B11；复用 prism-audio `wav.rs`/`buffer.rs`）。
//!
//! 原生路径：WAV 字节流 → [`prism_audio::decode_wav`] → [`PcmBuffer`]，并给出
//! 时长（供 [`crate::timing::TimeMap`] 的 `dur` 使用，保证音画共享同一 `f(t)`）。
//! ffmpeg 协作路径：mp3/aac 等经探测读入（本层只标注，不启动解码进程）。

use prism_audio::{decode_wav, PcmBuffer};

use crate::source::{classify_read_kind, extension_of, ReadKind, SourceKind};
use crate::{MediaError, MediaResult};

/// 已读入的音频源：PCM 缓冲 + 时长。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSource {
    /// 解码后的 PCM 缓冲（交错 f32）。
    pub buffer: PcmBuffer,
}

impl AudioSource {
    /// 时长（秒）——时间映射器的 `dur`。
    pub fn duration_secs(&self) -> f32 {
        self.buffer.duration_secs()
    }

    /// 采样率（Hz）。
    pub fn sample_rate(&self) -> u32 {
        self.buffer.sample_rate()
    }

    /// 声道数。
    pub fn channels(&self) -> u16 {
        self.buffer.channels()
    }

    /// 第 `frame` 帧第 `channel` 声道样本（越界 `None`）。
    pub fn sample(&self, frame: usize, channel: u16) -> Option<f32> {
        self.buffer.sample(frame, channel)
    }
}

/// 从字节流读入音频源（按扩展名选定原生/外部路径）。
///
/// - `.wav` → 原生 [`decode_wav`]；
/// - `.mp3`/`.aac`/... → 返回三段式错误，提示经 ffmpeg 协作（本层不启动进程，
///   解码器不自研，D-17）；
/// - 其他 → 不可读。
pub fn load_audio(path: &str, bytes: &[u8]) -> MediaResult<AudioSource> {
    match classify_read_kind(path, SourceKind::AudioSource) {
        ReadKind::Native => {
            let ext = extension_of(path);
            if ext != "wav" {
                return Err(MediaError::new(
                    "MD1002",
                    "audio::load_audio",
                    format!("原生路径仅支持 wav，收到 .{ext}"),
                    "改用 WAV，或经 ffmpeg 协作读入",
                ));
            }
            let buffer = decode_wav(bytes).map_err(|e| {
                MediaError::new(
                    "MD1003",
                    "audio::load_audio",
                    format!("WAV 解码失败（{}）：{}", e.code, e.problem),
                    e.fix,
                )
            })?;
            Ok(AudioSource { buffer })
        }
        ReadKind::ExternalFfmpeg => Err(MediaError::new(
            "MD1004",
            "audio::load_audio",
            format!("格式 `.{ext}` 需外部 ffmpeg 协作（本层不启动解码进程）", ext = extension_of(path)),
            "探测到 ffmpeg 后经外部进程读入为 WAV，再交本函数解码",
        )),
        ReadKind::Unreadable => Err(MediaError::new(
            "MD1005",
            "audio::load_audio",
            format!("扩展名 `.{ext}` 不可识别", ext = extension_of(path)),
            "使用 WAV，或受支持的音频扩展名（mp3/aac/m4a/ogg/flac）",
        )),
    }
}

/// 便捷：从纯样本构造音频源（测试/程序化素材）。
pub fn from_mono(sample_rate: u32, samples: Vec<f32>) -> MediaResult<AudioSource> {
    let buffer = PcmBuffer::from_mono(sample_rate, samples).map_err(|e| {
        MediaError::new("MD1003", "audio::from_mono", format!("缓冲构造失败：{}", e.problem), e.fix)
    })?;
    Ok(AudioSource { buffer })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio::{encode_wav, BitDepth};

    #[test]
    fn wav_roundtrip_via_source() {
        let buf = PcmBuffer::from_mono(8_000, vec![0.0, 0.5, -0.5, 1.0]).unwrap();
        let bytes = encode_wav(&buf, BitDepth::B16);
        let src = load_audio("clip.wav", &bytes).expect("WAV 必可原生读入");
        assert_eq!(src.sample_rate(), 8_000);
        assert_eq!(src.channels(), 1);
        assert!((src.duration_secs() - 4.0 / 8_000.0).abs() < 1e-9);
        assert_eq!(src.sample(1, 0).unwrap(), 0.5);
    }

    #[test]
    fn mp3_requires_ffmpeg() {
        let err = load_audio("song.mp3", &[]).unwrap_err();
        assert_eq!(err.code, "MD1004");
        assert!(err.problem.contains("ffmpeg"));
    }

    #[test]
    fn unknown_extension_unreadable() {
        let err = load_audio("data.xyz", &[]).unwrap_err();
        assert_eq!(err.code, "MD1005");
    }

    #[test]
    fn bad_wav_bytes_structured_error() {
        let err = load_audio("bad.wav", b"not wav").unwrap_err();
        assert_eq!(err.code, "MD1003");
    }
}
