//! 节拍 / onset 检测。
//!
//! 流程（全部固定顺序，无隐藏累加器——窗内容完全由位置与窗长决定，任意
//! t 的节拍表可独立重算， "状态 = 纯函数(t)"）：
//!
//! 1. 下混单声道 → `(win, hop)` 分帧 → Hann 窗 → FFT 半谱幅值；
//! 2. **频谱通量正部**：`flux[f] = Σ_k max(0, |X_f[k]| − |X_{f−1}[k]|)`
//!    （跳过 DC bin）；
//! 3. **自适应阈值**：`thr[f] = k · mean(flux[f±ctx])`（滑动窗口均值 × 系数，
//!  阈值系数暴露为参数—— 风险项的对策）；
//! 4. **局部极大值拾取** → onset 表（分析帧号 + 样本位置 + 强度）；
//! 5. **BPM**：拍间隔（onset 间隔）**中值滤波** → `bpm = 60 / 中值秒`
//!    （中值序 = 升序排序后取中位；偶数取中间两位均值）。

use crate::buffer::PcmBuffer;
use crate::fft::{hann_window, Fft};
use crate::{AudioError, AudioResult};

/// 检测参数（win/hop 为分析格锚——整数样本，格 Zᵏ）。
#[derive(Debug, Clone, PartialEq)]
pub struct BeatsConfig {
    /// 分析窗长（样本，2 的幂）。
    pub win: usize,
    /// 帧移（样本；onset 时间分辨率 = hop/SR）。
    pub hop: usize,
    /// 自适应阈值系数（flux 相对局部均值的倍数，> 1）。
    pub threshold: f32,
    /// 局部均值窗口的半宽（分析帧数）。
    pub context_frames: usize,
}

impl Default for BeatsConfig {
    fn default() -> Self {
        Self { win: 2048, hop: 512, threshold: 1.5, context_frames: 16 }
    }
}

/// 单个 onset（攻击点）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Onset {
    /// 分析帧号（hop 格）。
    pub window: usize,
    /// 样本位置（= window × hop）。
    pub sample: usize,
    /// 强度（该帧谱通量值）。
    pub strength: f32,
}

/// 节拍表（可缓存进签名缓存的确定性纯数据资产， 超越点）。
#[derive(Debug, Clone, PartialEq)]
pub struct BeatTable {
    /// onset 列表（按时间升序）。
    pub onsets: Vec<Onset>,
    /// BPM 估计（onset < 2 时为 0）。
    pub bpm: f32,
}

/// 检测节拍与 onset（纯函数；同输入逐位一致）。
pub fn detect_beats(buf: &PcmBuffer, cfg: &BeatsConfig) -> AudioResult<BeatTable> {
    if cfg.win < 2 || !cfg.win.is_power_of_two() || cfg.hop < 1 {
        return Err(AudioError::new(
            "AU1002",
            format!("beats::detect_beats(win={}, hop={})", cfg.win, cfg.hop),
            "win 必须为 ≥2 的 2 的幂，hop ≥ 1",
            "win 取 1024/2048，hop 取 win/4 量级",
        ));
    }
    if cfg.threshold <= 1.0 || cfg.context_frames < 1 {
        return Err(AudioError::new(
            "AU1002",
            "beats::detect_beats(threshold, context_frames)",
            "阈值系数必须 > 1（均值倍数），局部窗口半宽 ≥ 1",
            "threshold 取 1.3–2.0，context_frames 取 8–32",
        ));
    }
    let mono = buf.to_mono();
    if mono.len() < cfg.win {
        return Ok(BeatTable { onsets: Vec::new(), bpm: 0.0 });
    }
    let frame_count = (mono.len() - cfg.win) / cfg.hop + 1;
    let fft = Fft::new(cfg.win)?;
    let hann = hann_window(cfg.win);
    let bins = cfg.win / 2;

    // 谱通量正部（帧 0 无前帧记 0——能量不是通量）；固定顺序：帧 → bin。
    let mut flux = vec![0.0_f32; frame_count];
    let mut prev = vec![0.0_f32; bins];
    for f in 0..frame_count {
        let start = f * cfg.hop;
        let windowed: Vec<f32> = (0..cfg.win)
            .map(|i| mono[start + i] * hann[i])
            .collect();
        let mag = fft.magnitudes(&windowed)?;
        if f > 0 {
            let mut acc = 0.0_f32;
            for k in 1..bins {
                let d = mag[k] - prev[k];
                if d > 0.0 {
                    acc += d;
                }
            }
            flux[f] = acc;
        }
        prev.copy_from_slice(&mag[1..bins + 1]);
    }

    // 自适应阈值 + 局部极大值 → onset。
    let mut onsets = Vec::new();
    for f in 0..frame_count {
        let lo = f.saturating_sub(cfg.context_frames);
        let hi = (f + cfg.context_frames + 1).min(frame_count);
        let mean = flux[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        let thr = mean * cfg.threshold;
        if flux[f] > thr
            && flux[f] >= if f > 0 { flux[f - 1] } else { 0.0 }
            && flux[f] >= if f + 1 < frame_count { flux[f + 1] } else { 0.0 }
            && flux[f] > 0.0
        {
            onsets.push(Onset { window: f, sample: f * cfg.hop, strength: flux[f] });
        }
    }

    // 拍间隔中值滤波 → BPM。
    let bpm = if onsets.len() >= 2 {
        let mut iois: Vec<f32> =
            onsets.windows(2).map(|w| (w[1].window - w[0].window) as f32).collect();
        iois.sort_by(|a, b| a.partial_cmp(b).expect("IOI 无 NaN"));
        let n = iois.len();
        let median = if n % 2 == 1 { iois[n / 2] } else { (iois[n / 2 - 1] + iois[n / 2]) / 2.0 };
        let median_secs = median * cfg.hop as f32 / buf.sample_rate() as f32;
        if median_secs > 0.0 {
            60.0 / median_secs
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(BeatTable { onsets, bpm })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 120 BPM 节拍机（单样本 click）：onset 落点 ±1 hop 帧、BPM≈120。
    #[test]
    fn click_train_onsets_within_one_frame_and_bpm() {
        let (sr, hop, period) = (8_000_u32, 250_usize, 4_000_usize); // 0.5s = 16 帧
        let mut mono = vec![0.0_f32; 64_000];
        for i in (0..mono.len()).step_by(period) {
            mono[i] = 1.0;
        }
        let clicks: Vec<usize> = (0..mono.len()).step_by(period).collect();
        let buf = PcmBuffer::from_mono(sr, mono).unwrap();
        let table = detect_beats(&buf, &BeatsConfig { win: 512, hop, ..Default::default() })
            .unwrap();
        // 窗 0 的 click 被 Hann 压零，其余 15 个 click 应全部检出。
        assert!((14..=16).contains(&table.onsets.len()), "检出 {}", table.onsets.len());
        // 每个 onset 到最近 click 的距离不超过 ±1 hop（帧定位验收）。
        for o in &table.onsets {
            let nearest = clicks
                .iter()
                .copied()
                .min_by_key(|&c| (c as isize - o.sample as isize).abs())
                .unwrap();
            let dist = (nearest as isize - o.sample as isize).abs() as usize;
            assert!(dist <= hop, "onset 样本 {} 偏离 click {nearest} 达 {dist} > ±1 hop", o.sample);
        }
        assert!((table.bpm - 120.0).abs() < 2.0, "BPM 估计 {} 应≈120", table.bpm);
    }

    /// 静音与缓变信号（无攻击点）不产生 onset。
    #[test]
    fn silence_and_slow_signal_have_no_onsets() {
        let silence = PcmBuffer::from_mono(8_000, vec![0.0_f32; 16_000]).unwrap();
        let t = detect_beats(&silence, &BeatsConfig::default()).unwrap();
        assert!(t.onsets.is_empty() && t.bpm == 0.0);
        // 幅度缓变的正弦无攻击：flux 恒低，不超过均值×1.5。
        let slow: Vec<f32> = (0..32_000)
            .map(|i| (i as f32 * 0.01).sin() * (i as f32 / 32_000.0))
            .collect();
        let t2 = detect_beats(&PcmBuffer::from_mono(8_000, slow).unwrap(), &BeatsConfig::default())
            .unwrap();
        assert!(t2.onsets.is_empty(), "缓变信号误检 {} 个 onset", t2.onsets.len());
    }

    /// 短于窗长的输入返回空表；非法参数三段式报错；检测逐位确定。
    #[test]
    fn degenerate_inputs_and_determinism() {
        let tiny = PcmBuffer::from_mono(8_000, vec![0.5; 100]).unwrap();
        let t = detect_beats(&tiny, &BeatsConfig::default()).unwrap();
        assert!(t.onsets.is_empty());
        let buf = PcmBuffer::from_mono(8_000, vec![0.0; 8_000]).unwrap();
        assert_eq!(
            detect_beats(&buf, &BeatsConfig { win: 100, ..Default::default() }).unwrap_err().code,
            "AU1002"
        );
        assert_eq!(
            detect_beats(&buf, &BeatsConfig { threshold: 1.0, ..Default::default() })
                .unwrap_err()
                .code,
            "AU1002"
        );
        let noisy: Vec<f32> =
            (0..16_000).map(|i| prism_noise::pcg::hash1_f32(7, i) * 2.0 - 1.0).collect();
        let b = PcmBuffer::from_mono(8_000, noisy).unwrap();
        let first = detect_beats(&b, &BeatsConfig::default()).unwrap();
        for _ in 0..10 {
            assert_eq!(detect_beats(&b, &BeatsConfig::default()).unwrap(), first);
        }
    }
}
