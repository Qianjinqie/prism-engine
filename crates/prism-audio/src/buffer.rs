//! PCM 缓冲（B6 基件）：采样率 / 声道 / 帧数 + 交错 f32 样本。
//!
//! 混合（[`PcmBuffer::mixed`]）、缩放（[`PcmBuffer::scaled`]）、峰值
//! （[`PcmBuffer::peak`]）均为**固定序纯函数**：单遍按下标递增遍历、无隐藏
//! 状态，同输入逐位一致（确定性契约，lib.rs）。

use crate::{AudioError, AudioResult};

/// 交错 PCM 缓冲：`data[frame * channels + ch]` 为第 `frame` 帧第 `ch` 声道样本。
///
/// 不变量：`data.len() == frames() * channels()`（构造函数维护，`frames` 由
/// 长度派生，不单独存储——Mini Code：单一事实源）。
#[derive(Debug, Clone, PartialEq)]
pub struct PcmBuffer {
    sample_rate: u32,
    channels: u16,
    data: Vec<f32>,
}

impl PcmBuffer {
    /// 构造空缓冲（`sample_rate ≥ 1`、`channels ≥ 1`；违约即编程错误 panic）。
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        assert!(sample_rate >= 1, "采样率必须 ≥ 1，收到 {sample_rate}");
        assert!(channels >= 1, "声道数必须 ≥ 1，收到 {channels}");
        Self { sample_rate, channels, data: Vec::new() }
    }

    /// 由交错样本构造（`data.len()` 必须是 `channels` 的整数倍）。
    pub fn from_interleaved(sample_rate: u32, channels: u16, data: Vec<f32>) -> AudioResult<Self> {
        let buf = Self::new(sample_rate, channels);
        if data.len() % channels as usize != 0 {
            return Err(AudioError::new(
                "AU1001",
                format!("PcmBuffer::from_interleaved(sr={sample_rate}, ch={channels})"),
                format!("交错样本长度 {} 不是声道数 {channels} 的整数倍", data.len()),
                format!("补齐或截断 data 至 {} 的整数倍", channels as usize),
            ));
        }
        Ok(Self { data, ..buf })
    }

    /// 由单声道样本构造。
    pub fn from_mono(sample_rate: u32, data: Vec<f32>) -> AudioResult<Self> {
        Self::from_interleaved(sample_rate, 1, data)
    }

    /// 采样率（Hz）。
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 声道数。
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// 帧数（一帧 = 全声道各一个样本）。
    pub fn frames(&self) -> usize {
        self.data.len() / self.channels as usize
    }

    /// 交错样本切片（帧主序）。
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// 读取第 `frame` 帧第 `channel` 声道样本；越界返回 `None`。
    pub fn sample(&self, frame: usize, channel: u16) -> Option<f32> {
        if channel >= self.channels {
            return None;
        }
        self.data.get(frame * self.channels as usize + channel as usize).copied()
    }

    /// 时长（秒）。
    pub fn duration_secs(&self) -> f32 {
        self.frames() as f32 / self.sample_rate as f32
    }

    /// 下混为单声道：逐帧按声道序求算术平均（固定序，确定性）。
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.clone();
        }
        let (ch, frames) = (self.channels as usize, self.frames());
        let mut out = vec![0.0_f32; frames];
        for f in 0..frames {
            // 固定声道序累加后一次除法：除法次数固定，跨平台逐位一致。
            let mut acc = 0.0_f32;
            for c in 0..ch {
                acc += self.data[f * ch + c];
            }
            out[f] = acc / ch as f32;
        }
        out
    }

    /// 与 `other` 逐帧线性叠加（混合原语；形状不一致报 `AU1001`）。
    pub fn mixed(&self, other: &PcmBuffer) -> AudioResult<PcmBuffer> {
        self.check_same_shape(other, "mixed")?;
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Self { sample_rate: self.sample_rate, channels: self.channels, data })
    }

    /// 全体样本乘常数增益。
    pub fn scaled(&self, gain: f32) -> PcmBuffer {
        let data = self.data.iter().map(|x| x * gain).collect();
        Self { sample_rate: self.sample_rate, channels: self.channels, data }
    }

    /// 峰值 = max |x|（空缓冲为 0；固定下标序取 max，确定性）。
    pub fn peak(&self) -> f32 {
        self.data.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    /// 形状（采样率/声道/帧数）一致性检查，供混合类操作复用。
    pub(crate) fn check_same_shape(&self, other: &PcmBuffer, op: &str) -> AudioResult<()> {
        if self.sample_rate != other.sample_rate
            || self.channels != other.channels
            || self.frames() != other.frames()
        {
            return Err(AudioError::new(
                "AU1001",
                format!("PcmBuffer::{op}"),
                format!(
                    "形状不一致：self=({}Hz, {}ch, {}帧) other=({}Hz, {}ch, {}帧)",
                    self.sample_rate, self.channels, self.frames(),
                    other.sample_rate, other.channels, other.frames()
                ),
                "先对齐采样率/声道/帧数（重采样属 C 系列交付管线，本层不做）",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_accessors() {
        let buf = PcmBuffer::from_interleaved(44_100, 2, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
            .expect("6 样本 / 2 声道应合法");
        assert_eq!((buf.sample_rate(), buf.channels(), buf.frames()), (44_100, 2, 3));
        assert_eq!(buf.sample(1, 0), Some(0.3));
        assert_eq!(buf.sample(1, 1), Some(-0.4));
        assert_eq!(buf.sample(3, 0), None, "越界帧返回 None");
        assert_eq!(buf.sample(0, 2), None, "越界声道返回 None");
        assert!((buf.duration_secs() - 3.0 / 44_100.0).abs() < 1e-12);
    }

    #[test]
    fn interleaved_length_must_match_channels() {
        let err = PcmBuffer::from_interleaved(44_100, 2, vec![0.1, 0.2, 0.3]).unwrap_err();
        assert_eq!(err.code, "AU1001");
        assert!(err.to_string().contains("整数倍"), "三段式应说明事实：{err}");
        assert!(!err.fix.is_empty(), "建议段必填（宪章 C19）");
    }

    #[test]
    fn to_mono_averages_channels_in_fixed_order() {
        let buf = PcmBuffer::from_interleaved(8_000, 2, vec![0.0, 1.0, -1.0, 1.0]).unwrap();
        assert_eq!(buf.to_mono(), vec![0.5, 0.0]);
        let mono = PcmBuffer::from_mono(8_000, vec![0.25, -0.75]).unwrap();
        assert_eq!(mono.to_mono(), vec![0.25, -0.75], "单声道应原样返回");
    }

    /// 混合 = 线性叠加；两个相同正弦半幅轨相加 = 全幅正弦（能量守恒的缓冲层锚）。
    #[test]
    fn mixed_is_linear_sum_and_scaled_scales() {
        let a = PcmBuffer::from_mono(8, vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        let b = PcmBuffer::from_mono(8, vec![0.5, -0.5, 0.5, -0.5]).unwrap();
        assert_eq!(a.mixed(&b).unwrap().as_slice(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(a.scaled(0.25).as_slice(), &[0.125; 4]);
        let wrong = PcmBuffer::from_mono(16, vec![0.0; 4]).unwrap();
        assert_eq!(a.mixed(&wrong).unwrap_err().code, "AU1001");
    }

    /// 固定序确定性：混合/缩放/峰值重复调用逐位一致。
    #[test]
    fn pure_ops_are_bitwise_reproducible() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let a = PcmBuffer::from_mono(48_000, data.clone()).unwrap();
        let b = PcmBuffer::from_mono(48_000, data).unwrap();
        let expect = (a.mixed(&b).unwrap(), a.scaled(0.7), a.peak());
        for _ in 0..1_000 {
            let got = (a.mixed(&b).unwrap(), a.scaled(0.7), a.peak());
            assert_eq!(
                (got.0.as_slice(), got.1.as_slice(), got.2.to_bits()),
                (expect.0.as_slice(), expect.1.as_slice(), expect.2.to_bits())
            );
        }
        assert!(a.peak() > 0.9 && a.peak() <= 1.0);
        assert_eq!(PcmBuffer::new(8, 1).peak(), 0.0, "空缓冲峰值为 0");
    }
}
