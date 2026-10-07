//! BGM 混音：多轨线性混合 + 淡入淡出 + 峰值防削波。
//!
//! 固定轨序累加（`track[0] → track[n]`），淡入淡出为样本索引的纯函数
//! （线性斜坡，格点端点 0→1）；输出峰值 > 1 时整轨等比缩放 `1/peak`
//! （防削波，不逐样本限幅——保留轨间比例，确定性）。
//!
//! 能量守恒（无削波路径）：`E(Σ gᵢ·xᵢ) = Σ gᵢ²·E(xᵢ)`——线性叠加，测试
//! 以 Parseval 口径逐位验证（f32 容差内）。

use crate::buffer::PcmBuffer;
use crate::{AudioError, AudioResult};

/// 一条混音轨：缓冲 + 增益 + 淡入/淡出（秒）。
#[derive(Debug, Clone, PartialEq)]
pub struct MixTrack {
    /// 轨道音频。
    pub buffer: PcmBuffer,
    /// 常量增益。
    pub gain: f32,
    /// 淡入时长（秒；0 = 无淡入）。
    pub fade_in_secs: f32,
    /// 淡出时长（秒；0 = 无淡出）。
    pub fade_out_secs: f32,
}

impl MixTrack {
    /// 构造：缓冲 + 增益（无淡入淡出）。
    pub fn new(buffer: PcmBuffer, gain: f32) -> Self {
        Self { buffer, gain, fade_in_secs: 0.0, fade_out_secs: 0.0 }
    }
}

/// 混合多轨（≥1 轨；形状一致，否则 `MIX1001`；空轨列表 `AU1002`）。
pub fn mix_tracks(tracks: &[MixTrack]) -> AudioResult<PcmBuffer> {
    if tracks.is_empty() {
        return Err(AudioError::new(
            "AU1002",
            "mixer::mix_tracks([])",
            "至少需要一条混音轨",
            "提供 ≥1 条 MixTrack",
        ));
    }
    let first = &tracks[0].buffer;
    for (i, t) in tracks.iter().enumerate() {
        first.check_same_shape(&t.buffer, &format!("mix_tracks[track {i}]")).map_err(|mut e| {
            e.code = "MIX1001";
            e
        })?;
    }
    let sr = first.sample_rate() as f32;
    let frames = first.frames();
    let mut data = vec![0.0_f32; frames * first.channels() as usize];

    for t in tracks {
        let fin = (t.fade_in_secs.max(0.0) * sr) as usize;
        let fout = (t.fade_out_secs.max(0.0) * sr) as usize;
        let ch = t.buffer.channels() as usize;
        for f in 0..frames {
            // 线性斜坡淡入淡出：索引的纯函数，端点皆格点（0 与 1）。
            let g_in = if fin > 0 && f < fin { f as f32 / fin as f32 } else { 1.0 };
            let g_out = if fout > 0 && f + fout >= frames {
                (frames - 1 - f) as f32 / fout as f32
            } else {
                1.0
            };
            let g = t.gain * g_in.min(g_out).max(0.0);
            for c in 0..ch {
                data[f * ch + c] += t.buffer.as_slice()[f * ch + c] * g;
            }
        }
    }

    let mut out = PcmBuffer::from_interleaved(first.sample_rate(), first.channels(), data)?;
    let peak = out.peak();
    if peak > 1.0 {
        out = out.scaled(1.0 / peak); // 峰值防削波：整轨等比，保留轨间比例
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 能量守恒：两轨半幅正弦相加 = 全幅正弦（Parseval 口径，f32 容差）。
    #[test]
    fn linear_sum_energy_conservation() {
        let sr = 8_000_u32;
        let x: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.05).sin()).collect();
        let t1 = MixTrack::new(PcmBuffer::from_mono(sr, x.clone()).unwrap(), 0.5);
        let t2 = MixTrack::new(PcmBuffer::from_mono(sr, x).unwrap(), 0.5);
        let mixed = mix_tracks(&[t1, t2]).unwrap();
        let energy: f64 = mixed.as_slice().iter().map(|v| (v * v) as f64).sum();
        let source: f64 = (0..4_000).map(|i| (i as f32 * 0.05).sin() as f64).map(|v| v * v).sum();
        let rel = ((energy - source) / source).abs();
        assert!(rel < 1e-5, "能量守恒破坏：mixed={energy} source={source} rel={rel}");
        assert!(mixed.peak() <= 1.0);
    }

    /// 峰值防削波：0.8+0.8 → 等比缩到峰值 1，轨间比例保留。
    #[test]
    fn peak_guard_scales_uniformly() {
        let sr = 1_000_u32;
        let a = PcmBuffer::from_mono(sr, vec![0.8, 0.4]).unwrap();
        let b = PcmBuffer::from_mono(sr, vec![0.8, -0.4]).unwrap();
        let mixed = mix_tracks(&[MixTrack::new(a, 1.0), MixTrack::new(b, 1.0)]).unwrap();
        assert_eq!(mixed.peak().to_bits(), 1.0_f32.to_bits(), "峰值应恰为 1");
        assert_eq!(mixed.sample(1, 0), Some(0.0), "0.4−0.4 保持抵消（等比缩放）");
    }

    /// 淡入淡出：端点为 0、进入满增益后与无淡入轨逐位一致。
    #[test]
    fn fades_are_linear_ramps() {
        let sr = 1_000_u32;
        let x = vec![1.0_f32; 1_000];
        let faded = mix_tracks(&[MixTrack {
            buffer: PcmBuffer::from_mono(sr, x.clone()).unwrap(),
            gain: 1.0,
            fade_in_secs: 0.1, // 100 样本
            fade_out_secs: 0.1,
        }])
        .unwrap();
        assert_eq!(faded.sample(0, 0), Some(0.0), "淡入端点 0");
        assert!((faded.sample(99, 0).unwrap() - 0.99).abs() < 1e-6);
        assert_eq!(faded.sample(500, 0), Some(1.0), "满增益区不受影响");
        assert_eq!(faded.sample(999, 0), Some(0.0), "淡出端点 0");
        let plain = mix_tracks(&[MixTrack::new(PcmBuffer::from_mono(sr, x).unwrap(), 1.0)]).unwrap();
        assert_eq!(plain.sample(500, 0), Some(1.0));
    }

    /// 形状不一致/空轨列表三段式报错；混音固定序确定性。
    #[test]
    fn validation_and_determinism() {
        assert_eq!(mix_tracks(&[]).unwrap_err().code, "AU1002");
        let a = MixTrack::new(PcmBuffer::from_mono(8_000, vec![0.1; 100]).unwrap(), 1.0);
        let b = MixTrack::new(PcmBuffer::from_mono(16_000, vec![0.1; 100]).unwrap(), 1.0);
        assert_eq!(mix_tracks(&[a.clone(), b]).unwrap_err().code, "MIX1001");
        let first = mix_tracks(&[a.clone(), a.clone()]).unwrap();
        for _ in 0..100 {
            assert_eq!(mix_tracks(&[a.clone(), a.clone()]).unwrap().as_slice(), first.as_slice());
        }
    }
}
