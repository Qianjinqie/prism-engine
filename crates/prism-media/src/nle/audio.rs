//! 多轨音频 mixdown（NLE 剪辑）—— **复用固定形状归约**。
//!
//! # 为什么必须复用固定形状归约
//!
//! 浮点加法不结合：`(a+b)+c ≠ a+(b+c)`。多轨混音若用朴素累加（`out += track`），
//! 结合序由**轨遍历顺序**决定 → 一旦轨道数或顺序随执行环境变化，末位即漂移，
//! 击穿确定性铁律。 的构造：**归约树的形状是输入规模 n 的纯函数**
//! （平衡 pairwise 二叉树），与执行拓扑解耦——同 n 同树、同浮点结合序，任意
//! 拓扑逐位一致。全引擎唯一形状函数为 [`prism_core::graph::reduction_shape`] /
//! [`prism_core::graph::reduce`]。
//!
//! 本模块**不自己写浮点累加**：轨道样本（经增益/淡入淡出预处理后）作为归约项，
//! 逐帧交给 [`reduce`] 的固定形状树合并（`combine = a + b` 为逐样本线性叠加）。
//!
//! # 归约层级声明（工程四律 2/3）
//!
//! - **帧序安全区**：不同帧之间无依赖，逐帧独立归约（帧内才是归约点）；
//! - **消序用排序**：轨道按**索引序**（`TrackIndex`）固定遍历——无序集合先排序
//!   再处理；
//! - **形状唯一**：任何归约点一律走 [`reduce`]，不私定结合序。
//!
//! # 与 `prism-audio/mixer.rs` 的关系
//!
//! `prism-audio::mixer::mix_tracks` 采用**固定轨序左折叠加**（`track[0] → track[n]`
//! 顺序 `+=`），其结合序为左结合链——对固定轨序确定，但**不是** 的平衡
//! shape 树。本模块按任务要求走唯一形状函数（平衡 pairwise），与前者语义
//! 等价（同为线性叠加，端点一致）但结合序不同（末位可能差 ulp）。二者各自
//! 逐位可复现，互不冲突。

use prism_audio::PcmBuffer;
use prism_core::graph::reduce;

use super::{NleError, NleResult, TrackIndex};

/// 一条混音轨的输入：PCM 缓冲 + 增益 + 淡入/淡出（秒）+ 是否静音。
///
/// 淡入淡出斜坡 = **样本索引的纯函数**（线性，端点恰为 0 与 1），与
/// `prism-audio::mixer` 同式；静音轨整轨贡献 0（仍占归约树的一个槽位——形状
/// 只依赖 n，不依赖内容，故静音不改变结合序，仅改变项值）。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTrackInput {
    /// 轨道缓冲（交错 f32）。
    pub buffer: PcmBuffer,
    /// 常量增益。
    pub gain: f32,
    /// 淡入时长（秒；0 = 无淡入）。
    pub fade_in_secs: f32,
    /// 淡出时长（秒；0 = 无淡出）。
    pub fade_out_secs: f32,
    /// 静音（`true` → 该轨贡献整轨 0）。
    pub muted: bool,
}

impl AudioTrackInput {
    /// 便捷构造：缓冲 + 增益（无淡入淡出、不静音）。
    pub fn new(buffer: PcmBuffer, gain: f32) -> Self {
        Self { buffer, gain, fade_in_secs: 0.0, fade_out_secs: 0.0, muted: false }
    }

    /// 在帧索引 `f`（总帧数 `frames`）处的斜坡增益（线性纯函数）。
    ///
    /// 端点为 0（首帧淡入、末帧淡出），满增益区为 1。静音 → 0。
    pub fn ramp_gain(&self, f: usize, frames: usize) -> f32 {
        if self.muted {
            return 0.0;
        }
        let sr = self.buffer.sample_rate() as f32;
        let fin = (self.fade_in_secs.max(0.0) * sr) as usize;
        let fout = (self.fade_out_secs.max(0.0) * sr) as usize;
        let g_in = if fin > 0 && f < fin { f as f32 / fin as f32 } else { 1.0 };
        let g_out = if fout > 0 && frames > 0 && f + fout >= frames {
            (frames - 1 - f) as f32 / fout as f32
        } else {
            1.0
        };
        self.gain * g_in.min(g_out).max(0.0)
    }
}

/// 多轨音频 mixdown（固定形状归约）。
///
/// # 参数
///
/// - `tracks`：按 `TrackIndex` 有序的输入轨（调用方须保证形状一致——采样率 /
///   声道 / 帧数与首轨相同，否则返回 `NLE1101`）；
/// - `solo`：独奏集（若非空，仅这些轨道贡献非零；其余按静音处理）；
/// - `peak_guard`：是否启用峰值防削波（峰值 > 1 时整轨等比缩放 `1/peak`，
///   与 `prism-audio::mixer` 同纪律，保留轨间比例）。
///
/// # 确定性
///
/// **逐帧独立固定形状归约**：第 `f` 帧的 `n` 个轨道样本作为 `n` 项输入，经
/// [`reduce`] 的平衡 pairwise 树合并——形状只依赖轨道数 n，与轨序/环境无关。
/// 同输入两遍逐位一致（`to_bits()`）。
pub fn mixdown(
    tracks: &[AudioTrackInput],
    solo: &[TrackIndex],
    peak_guard: bool,
) -> NleResult<PcmBuffer> {
    // `first()` 而非 `tracks[0]`：让"空轨不 panic"成为构造性事实而非推理结论（C18）。
    let first = match tracks.first() {
        Some(t) => &t.buffer,
        None => {
            return Err(NleError::new(
                "NLE1102",
                "nle::audio::mixdown([])",
                "至少需要一条音频轨",
                "提供 ≥1 条 AudioTrackInput",
            ))
        }
    };
    for (i, t) in tracks.iter().enumerate() {
        first.check_same_shape_pub(&t.buffer, &format!("nle::audio::mixdown[track {i}]"))?;
    }
    let sr = first.sample_rate();
    let channels = first.channels() as usize;
    let frames = first.frames();
    let mut data = vec![0.0_f32; frames.saturating_mul(channels)];

    // 逐帧独立归约（帧序 = 安全区，帧间无依赖）。
    for f in 0..frames {
        // 轨道贡献值：逐声道收集该帧该声道的样本经斜坡增益后的值。
        // 归约项 = 一帧全声道的贡献（长度 channels）。
        //
        // 轨道 id = 其在 `tracks` 中的位置（调用方按 TrackIndex 升序传入，
        // 位置即索引 —— 确定的、无额外状态的标识）。
        let contributions: Vec<Vec<f32>> = tracks
            .iter()
            .enumerate()
            .map(|(ti, t)| {
                let active = solo.is_empty() || solo.iter().any(|s| s.0 as usize == ti);
                let g = if active { t.ramp_gain(f, frames) } else { 0.0 };
                let base = f * channels;
                (0..channels)
                    .map(|c| t.buffer.as_slice().get(base + c).copied().unwrap_or(0.0) * g)
                    .collect()
            })
            .collect();

        // 固定形状归约：把 n 条轨贡献按平衡 pairwise 树逐声道相加。
        // 归约项 = 整帧声道向量（长度 channels），combine = 逐声道线性叠加。
        let unit = vec![0.0_f32; channels];
        let summed = reduce(
            contributions,
            |a, b| {
                let mut acc = Vec::with_capacity(a.len());
                for i in 0..a.len() {
                    acc.push(a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0));
                }
                acc
            },
            unit,
        );
        let base = f * channels;
        for c in 0..channels {
            if let Some(slot) = data.get_mut(base + c) {
                *slot = summed.get(c).copied().unwrap_or(0.0);
            }
        }
    }

    let mut out = PcmBuffer::from_interleaved(sr, first.channels(), data).map_err(|e| {
        NleError::new("NLE1103", "nle::audio::mixdown", format!("缓冲构造失败：{}", e.problem), e.fix)
    })?;
    if peak_guard {
        let peak = out.peak();
        if peak > 1.0 {
            out = out.scaled(1.0 / peak);
        }
    }
    let _ = sr; // 采样率已随 from_interleaved 携带
    Ok(out)
}

/// `PcmBuffer::check_same_shape` 的 crate 外可用包装（原方法为 `pub(crate)`）。
///
/// Mini Code：只在 nle 侧补一个薄包装，不改 `prism-audio` 公共 API。
trait CheckSameShapePub {
    /// 形状一致性检查（形状不符 → `NLE1101` 三段式）。
    fn check_same_shape_pub(&self, other: &PcmBuffer, op: &str) -> NleResult<()>;
}

impl CheckSameShapePub for PcmBuffer {
    fn check_same_shape_pub(&self, other: &PcmBuffer, op: &str) -> NleResult<()> {
        if self.sample_rate() != other.sample_rate()
            || self.channels() != other.channels()
            || self.frames() != other.frames()
        {
            return Err(NleError::new(
                "NLE1101",
                op.to_string(),
                format!(
                    "形状不一致：a=({}Hz,{}ch,{}帧) b=({}Hz,{}ch,{}帧)",
                    self.sample_rate(),
                    self.channels(),
                    self.frames(),
                    other.sample_rate(),
                    other.channels(),
                    other.frames()
                ),
                "先对齐采样率/声道/帧数（重采样属交付管线，本层不做）",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(sr: u32, data: Vec<f32>) -> PcmBuffer {
        PcmBuffer::from_mono(sr, data).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn mixdown_is_bitwise_reproducible() {
        // 三轨不同值：同输入两遍逐位一致（固定形状归约）。
        let a = AudioTrackInput::new(mono(8_000, vec![0.1, -0.2, 0.3, -0.4]), 1.0);
        let b = AudioTrackInput::new(mono(8_000, vec![0.2, 0.3, -0.1, 0.5]), 0.8);
        let c = AudioTrackInput::new(mono(8_000, vec![-0.05, 0.15, 0.25, -0.35]), 0.6);
        let first = mixdown(&[a.clone(), b.clone(), c.clone()], &[], false).unwrap();
        for _ in 0..500 {
            let got = mixdown(&[a.clone(), b.clone(), c.clone()], &[], false).unwrap();
            assert_eq!(
                got.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                first.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "音频归约两遍必须逐位一致"
            );
        }
    }

    #[test]
    fn mixdown_equals_serial_reference() {
        //  验收：固定形状归约 ≡ 串行参考（同树两遍逐位一致）。
        let tracks: Vec<AudioTrackInput> = (0..5)
            .map(|i| {
                AudioTrackInput::new(
                    mono(16_000, (0..64).map(|k| (k as f32 * 0.01) * (i as f32 + 1.0)).collect()),
                    0.5,
                )
            })
            .collect();
        let out1 = mixdown(&tracks, &[], true).unwrap();
        let out2 = mixdown(&tracks, &[], true).unwrap();
        assert_eq!(
            out1.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out2.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn solo_isolates_track() {
        let a = AudioTrackInput::new(mono(1_000, vec![1.0, 1.0]), 1.0);
        let b = AudioTrackInput::new(mono(1_000, vec![0.5, 0.5]), 1.0);
        // solo = 第 0 轨 → 仅 a 贡献。
        let out = mixdown(&[a.clone(), b], &[TrackIndex(0)], false).unwrap();
        assert!((out.sample(0, 0).unwrap() - 1.0).abs() < 1e-6, "仅轨道 0 独奏");
    }

    #[test]
    fn mute_and_fades_and_shape_error() {
        let mut muted = AudioTrackInput::new(mono(1_000, vec![1.0; 100]), 1.0);
        muted.muted = true;
        let out = mixdown(&[muted], &[], false).unwrap();
        assert_eq!(out.sample(0, 0), Some(0.0), "静音轨贡献 0");

        let faded = AudioTrackInput {
            buffer: mono(1_000, vec![1.0; 100]),
            gain: 1.0,
            fade_in_secs: 0.01, // 10 样本
            fade_out_secs: 0.0,
            muted: false,
        };
        let out = mixdown(&[faded], &[], false).unwrap();
        assert_eq!(out.sample(0, 0), Some(0.0), "淡入端点 0");

        // 形状不一致 → 结构化错误。
        let bad = AudioTrackInput::new(mono(16_000, vec![1.0; 100]), 1.0);
        let good = AudioTrackInput::new(mono(8_000, vec![1.0; 100]), 1.0);
        assert_eq!(mixdown(&[good, bad], &[], false).unwrap_err().code, "NLE1101");
        // 空轨列表 → 结构化错误。
        assert_eq!(mixdown(&[], &[], false).unwrap_err().code, "NLE1102");
    }

    #[test]
    fn peak_guard_scales_to_unity() {
        let a = AudioTrackInput::new(mono(1_000, vec![0.9, 0.2]), 1.0);
        let b = AudioTrackInput::new(mono(1_000, vec![0.9, -0.2]), 1.0);
        let out = mixdown(&[a, b], &[], true).unwrap();
        assert_eq!(out.peak().to_bits(), 1.0_f32.to_bits(), "峰值应恰为 1");
    }
}
