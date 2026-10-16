//! 物理建模合成：Karplus-Strong 弦 + 模态打击（modal synthesis）。
//!
//! # Karplus-Strong 弦
//!
//! 经典算法（Karplus & Strong 1983）：激励噪声填入长度 `L` 的延迟线，循环时
//! 用**二点平均 + 衰减**低通滤波，产生拨弦音色。音高由延迟长度决定：
//!
//! ```text
//! f0 = SR / (L + 0.5)      （+0.5 补偿平均滤波引入的半样本相位延迟）
//! ```
//!
//! 为满足确定性（无全局 RNG），激励取**确定性伪随机**：`prism-noise` PCG 整数
//! 哈希按 `hash(seed, i)` 派生。默认激励用理想冲激（起播即全谐波）。
//!
//! # 模态打击
//!
//! 把物体建模为叠加的共振模态：每个模态是阻尼正弦 `Aₖ·e^(−t/τₖ)·sin(2πfₖt)`，
//! 总响应 = Σ 模态。频率比（fₖ/f₀）由振型几何决定（此处取理想化棒/板模态比）。

use crate::{AudioError, AudioResult};

/// Karplus-Strong 弦参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KarplusStrong {
    /// 基频（Hz）。
    pub freq: f32,
    /// 衰减系数 ∈ (0, 1)：每次循环幅度乘以该系数（越小衰减越快）。
    pub decay: f32,
    /// 激励种子（确定性伪随机，PCG 哈希；`None` 用理想冲激）。
    pub seed: Option<u64>,
    /// 亮度 ∈ [0,1]：0 = 只做二点平均（最暗），1 = 不做平均（最亮）。
    pub brightness: f32,
}

impl Default for KarplusStrong {
    fn default() -> Self {
        Self { freq: 220.0, decay: 0.996, seed: None, brightness: 0.5 }
    }
}

impl KarplusStrong {
    /// 渲染 `frames` 个样本（纯函数：同参数逐位一致）。
    pub fn render(&self, frames: usize, sample_rate: u32) -> AudioResult<Vec<f32>> {
        if self.freq <= 0.0 || self.freq >= sample_rate as f32 * 0.5 {
            return Err(AudioError::new(
                "AU1003",
                format!("KarplusStrong::render(freq={}, SR={sample_rate})", self.freq),
                "基频必须 ∈ (0, SR/2)，否则延迟线长度无意义",
                "把 freq 设在听域内（20 Hz … SR/2）",
            ));
        }
        // 延迟线长度：L = SR/f0 − 0.5（补偿二点平均的半样本延迟），至少 2。
        let l = ((sample_rate as f32 / self.freq) - 0.5).round().max(2.0) as usize;
        // 激励：确定性伪随机或理想冲激。
        let mut buf: Vec<f32> = (0..l)
            .map(|i| match self.seed {
                Some(s) => prism_noise::pcg::hash2_f32(s, i as u64, 0) * 2.0 - 1.0,
                None => if i == 0 { 1.0 } else { 0.0 },
            })
            .collect();
        let bright = self.brightness.clamp(0.0, 1.0);
        let decay = self.decay.clamp(0.0, 1.0);
        let mut out = Vec::with_capacity(frames);
        let mut idx = 0_usize;
        for _ in 0..frames {
            let cur = buf[idx];
            out.push(cur);
            // 二点平均低通，按亮度修正（bright=1 时不衰减高频）。
            let next_idx = (idx + 1) % l;
            let avg = 0.5 * (cur + buf[next_idx]);
            let filtered = avg * (1.0 - bright) + cur * bright;
            buf[idx] = filtered * decay;
            idx = next_idx;
        }
        Ok(out)
    }
}

/// 模态打击参数。
#[derive(Debug, Clone, PartialEq)]
pub struct ModalStrike {
    /// 基频（Hz）。
    pub freq: f32,
    /// 各模态相对幅度（≥1 个）。
    pub amplitudes: Vec<f32>,
    /// 各模态相对频率比（fₖ = ratioₖ·f0；长度须与 amplitudes 一致）。
    pub ratios: Vec<f32>,
    /// 各模态衰减时间常数（秒；长度须与 amplitudes 一致）。
    pub decays: Vec<f32>,
}

impl ModalStrike {
    /// 各模态幅度/频率比/衰减长度须一致（否则 `AU1003`）。
    pub fn validate(&self) -> AudioResult<()> {
        let n = self.amplitudes.len();
        if n == 0 || self.ratios.len() != n || self.decays.len() != n {
            return Err(AudioError::new(
                "AU1003",
                "ModalStrike::validate",
                format!(
                    "模态数组长度不一致：amplitudes={n} ratios={} decays={}",
                    self.ratios.len(),
                    self.decays.len()
                ),
                "令 amplitudes/ratios/decays 等长且非空",
            ));
        }
        Ok(())
    }

    /// 渲染 `frames` 个样本：`Σ Aₖ·e^(−t/τₖ)·sin(2πfₖt)`（纯函数）。
    pub fn render(&self, frames: usize, sample_rate: u32) -> AudioResult<Vec<f32>> {
        self.validate()?;
        let mut out = vec![0.0_f32; frames];
        for ((&amp, &ratio), &tau) in self.amplitudes.iter().zip(&self.ratios).zip(&self.decays) {
            let f = self.freq * ratio;
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f64 / sample_rate as f64;
                let env = crate::det::exp_f64(-(t / (tau.max(1e-4) as f64)));
                let phase = 2.0 * std::f64::consts::PI * f as f64 * t;
                *o += (amp as f64 * env * crate::det::sin_f64(phase)) as f32;
            }
        }
        Ok(out)
    }
}

/// 理想化棒/板模态：频率比与相对幅度（谐波式近似——真实金属棒含非谐模态，
/// 此处用等价比便于音高断言；外部可自定义 ratios）。
pub fn bar_modes(count: usize) -> ModalStrike {
    let amplitudes: Vec<f32> = (0..count).map(|k| 1.0 / (k as f32 + 1.0)).collect();
    let ratios: Vec<f32> = (0..count).map(|k| k as f32 + 1.0).collect();
    let decays: Vec<f32> = (0..count).map(|k| 1.0 / (k as f32 + 1.0)).collect();
    ModalStrike { freq: 440.0, amplitudes, ratios, decays }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 过零率反推基频：Karplus-Strong 音高应准确（±2%）。
    #[test]
    fn karplus_strong_pitch_accuracy() {
        let sr = 44_100_u32;
        // 理想冲激激励（noise=None）给出干净周期；过零率测后半段稳态。
        let ks = KarplusStrong { freq: 220.0, decay: 0.999, seed: None, brightness: 0.0 };
        let out = ks.render(44_100, sr).unwrap();
        let f_est = estimate_freq_zerocross(&out[10_000..], sr);
        assert!((f_est - 220.0).abs() / 220.0 < 0.02, "估计基频 {f_est} 应 ≈ 220");
    }

    /// 多音高：不同 freq 过零率随之改变（线性关系）。
    #[test]
    fn karplus_strong_tracks_frequency() {
        let sr = 44_100_u32;
        for &f in &[110.0_f32, 330.0, 550.0] {
            let out = KarplusStrong { freq: f, decay: 0.999, seed: None, brightness: 0.0 }
                .render(44_100, sr)
                .unwrap();
            let est = estimate_freq_zerocross(&out[10_000..], sr);
            assert!((est - f).abs() / f < 0.03, "f={f} 估计 {est}");
        }
    }

    /// 确定性：同种子逐位一致；不同种子不同（激励差异）。
    #[test]
    fn karplus_strong_deterministic_by_seed() {
        let sr = 22_050_u32;
        let a = KarplusStrong { freq: 300.0, seed: Some(1), ..Default::default() }.render(5_000, sr).unwrap();
        let b = KarplusStrong { freq: 300.0, seed: Some(1), ..Default::default() }.render(5_000, sr).unwrap();
        assert_eq!(a, b, "同种子逐位一致");
        let c = KarplusStrong { freq: 300.0, seed: Some(2), ..Default::default() }.render(5_000, sr).unwrap();
        assert_ne!(a, c, "不同种子激励不同");
    }

    /// 衰减：decay < 1 时尾段能量远小于头段。
    #[test]
    fn karplus_strong_decays() {
        let sr = 44_100_u32;
        let out = KarplusStrong { freq: 220.0, decay: 0.99, seed: None, brightness: 0.0 }
            .render(44_100, sr)
            .unwrap();
        let head: f64 = out[..1000].iter().map(|&x| (x * x) as f64).sum();
        let tail: f64 = out[40_000..].iter().map(|&x| (x * x) as f64).sum();
        assert!(tail < head, "应衰减：head={head} tail={tail}");
    }

    /// 模态打击：主模态频率处 FFT 能量峰（音高准确）。
    #[test]
    fn modal_strike_fundamental_peak() {
        let sr = 8_000_u32;
        let n = 4_096_usize;
        let strike = ModalStrike {
            freq: 500.0,
            amplitudes: vec![1.0, 0.3],
            ratios: vec![1.0, 2.0],
            decays: vec![1.0, 0.5],
        };
        let out = strike.render(n, sr).unwrap();
        let fft = crate::fft::Fft::new(n).unwrap();
        let mag = fft.magnitudes(&out).unwrap();
        let bin = (500.0 * n as f32 / sr as f32).round() as usize;
        let peak_bin = mag[..n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((peak_bin as i32 - bin as i32).abs() <= 1, "峰值 bin {peak_bin} 应 ≈ 基频 bin {bin}");
    }

    /// 非法参数三段式报错。
    #[test]
    fn validation_errors() {
        let err = KarplusStrong { freq: 30_000.0, ..Default::default() }
            .render(100, 44_100)
            .unwrap_err();
        assert_eq!(err.code, "AU1003");
        let bad = ModalStrike {
            freq: 440.0,
            amplitudes: vec![1.0, 1.0],
            ratios: vec![1.0],
            decays: vec![1.0],
        };
        assert_eq!(bad.validate().unwrap_err().code, "AU1003");
    }

    /// 过零率基频估计（测试工具；去直流后数正向过零/时长）。
    fn estimate_freq_zerocross(x: &[f32], sr: u32) -> f32 {
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let mut zc = 0_usize;
        let mut prev = x[0] - mean;
        for &v in &x[1..] {
            let cur = v - mean;
            if prev <= 0.0 && cur > 0.0 {
                zc += 1;
            }
            prev = cur;
        }
        // 每个周期 1 次正向过零。
        zc as f32 * sr as f32 / x.len() as f32
    }
}
