//! 颗粒合成：颗粒窗 / 密度 / 音高散布（确定性派生）。
//!
//! **颗粒合成**把源音频切成重叠的小片段（grains），每个颗粒有独立的起播时刻、
//! 窗形、音高与声像，在输出时间轴上叠加。本实现全部参数**确定性派生**：
//! 颗粒的起播位置/音高比/窗相位经 `prism-noise` PCG 哈希按 `hash(seed, 颗粒号, 槽)`
//! 生成（无全局 RNG、无状态泄漏）——同种子逐位一致。
//!
//! 每个颗粒是源的一段读区间，按 `pitch_ratio` 重采样（线性插值，格点对齐），
//! 乘**窗函数**（Hann）后叠加到输出。颗粒密度 = 每秒颗粒数；`position` 为源读
//! 位置（帧，可为小数，线性插值）。

use crate::{AudioError, AudioResult};

/// 颗粒窗形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainWindow {
    /// Hann（升余弦）。
    Hann,
    /// 三角。
    Triangle,
    /// 矩形（无窗）。
    Rect,
}

impl GrainWindow {
    /// `phase ∈ [0, 1]` 处的窗值（端点 0，Hann/Triangle；Rect 恒 1）。
    fn at(&self, phase: f32) -> f32 {
        let p = phase.clamp(0.0, 1.0);
        match self {
            GrainWindow::Hann => 0.5 * (1.0 - crate::det::cos_f64(2.0 * std::f64::consts::PI * p as f64) as f32),
            GrainWindow::Triangle => 1.0 - (2.0 * p - 1.0).abs(),
            GrainWindow::Rect => 1.0,
        }
    }
}

/// 颗粒合成参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrainSynth {
    /// 颗粒时长（秒）。
    pub grain_secs: f32,
    /// 密度（每秒颗粒数）。
    pub density: f32,
    /// 颗粒窗形。
    pub window: GrainWindow,
    /// 源读位置（帧；线性插值小数位置）。
    pub position: f32,
    /// 音高比中心（1.0 = 原速）。
    pub pitch: f32,
    /// 音高散布（比例；0 = 无散布）——经 PCG 哈希确定性派生。
    pub pitch_spread: f32,
    /// 位置散布（秒；源读位置抖动）。
    pub position_spread: f32,
    /// 派生种子（确定性）。
    pub seed: u64,
}

impl Default for GrainSynth {
    fn default() -> Self {
        Self {
            grain_secs: 0.05,
            density: 40.0,
            window: GrainWindow::Hann,
            position: 0.0,
            pitch: 1.0,
            pitch_spread: 0.0,
            position_spread: 0.0,
            seed: 0,
        }
    }
}

impl GrainSynth {
    /// 把 `source` 颗粒化渲染为 `frames` 个输出样本（纯函数，同种子逐位一致）。
    pub fn render(&self, source: &[f32], frames: usize, sample_rate: u32) -> AudioResult<Vec<f32>> {
        if source.is_empty() {
            return Err(AudioError::new(
                "AU1004",
                "GrainSynth::render",
                "源音频为空，无法切颗粒",
                "提供 ≥1 样本的源缓冲",
            ));
        }
        if self.grain_secs <= 0.0 || self.density <= 0.0 {
            return Err(AudioError::new(
                "AU1004",
                format!("GrainSynth::render(grain={}, density={})", self.grain_secs, self.density),
                "颗粒时长与密度必须 > 0",
                "设 grain_secs 与 density 为正值",
            ));
        }
        let sr = sample_rate as f32;
        let grain_len = (self.grain_secs * sr).round().max(1.0) as usize;
        let step = (sr / self.density).round().max(1.0) as usize; // 颗粒间隔（样本）
        let mut out = vec![0.0_f32; frames];
        // 颗粒号递增遍历（固定序）。
        let mut g: u64 = 0;
        let mut start = 0_usize;
        while start < frames {
            // 确定性派生：音高比与位置抖动。
            let r0 = prism_noise::pcg::hash2_f32(self.seed, g, 0); // ∈ [0,1)
            let r1 = prism_noise::pcg::hash2_f32(self.seed ^ 0x9E37_79B9, g, 1);
            let pitch = self.pitch * (1.0 + self.pitch_spread * (2.0 * r0 - 1.0));
            let pos_jitter = self.position_spread * sr * (2.0 * r1 - 1.0);
            let read_start = (self.position + pos_jitter).max(0.0);
            for k in 0..grain_len {
                let oi = start + k;
                if oi >= frames {
                    break;
                }
                let phase = k as f32 / grain_len as f32;
                let win = self.window.at(phase);
                // 源读位置 = read_start + k·pitch（线性插值）。
                let src_pos = read_start + k as f32 * pitch;
                out[oi] += win * lerp_sample(source, src_pos);
            }
            g += 1;
            start += step;
        }
        Ok(out)
    }
}

/// 在（可能为小数的）帧位置 `pos` 处线性插值采样源；越界钳到端样本。
#[inline]
fn lerp_sample(src: &[f32], pos: f32) -> f32 {
    let p = pos.max(0.0);
    let i = p.floor() as usize;
    if i + 1 >= src.len() {
        return src[src.len() - 1];
    }
    let frac = p - i as f32;
    src[i] * (1.0 - frac) + src[i + 1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 空源 / 非法参数三段式报错。
    #[test]
    fn validation() {
        assert_eq!(GrainSynth::default().render(&[], 100, 8_000).unwrap_err().code, "AU1004");
        let bad = GrainSynth { density: 0.0, ..Default::default() };
        assert_eq!(bad.render(&[0.1, 0.2], 100, 8_000).unwrap_err().code, "AU1004");
    }

    /// 确定性：同种子逐位一致；不同种子（有散布时）不同。
    #[test]
    fn grain_render_deterministic() {
        let src: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.05).sin()).collect();
        let gs = GrainSynth { pitch_spread: 0.25, position_spread: 0.01, seed: 3, ..Default::default() };
        let a = gs.render(&src, 8_000, 8_000).unwrap();
        let b = gs.render(&src, 8_000, 8_000).unwrap();
        assert_eq!(a, b, "同种子逐位一致");
        let c = GrainSynth { seed: 4, ..gs }.render(&src, 8_000, 8_000).unwrap();
        assert_ne!(a, c, "不同种子应不同");
    }

    /// 输出长度正确、有限、幅度受窗控制。
    #[test]
    fn grain_output_shape_bounded() {
        let src: Vec<f32> = vec![0.5; 2_000];
        let out = GrainSynth { density: 50.0, ..Default::default() }.render(&src, 4_000, 8_000).unwrap();
        assert_eq!(out.len(), 4_000);
        assert!(out.iter().all(|v| v.is_finite() && v.abs() < 5.0));
    }

    /// 窗形端点：Hann/Triangle 端点 0、中点 1；Rect 恒 1。
    #[test]
    fn window_endpoints() {
        assert!(GrainWindow::Hann.at(0.0).abs() < 1e-6);
        assert!((GrainWindow::Hann.at(0.5) - 1.0).abs() < 1e-6);
        assert!(GrainWindow::Triangle.at(0.0).abs() < 1e-6);
        assert!((GrainWindow::Triangle.at(0.5) - 1.0).abs() < 1e-6);
        assert_eq!(GrainWindow::Rect.at(0.3), 1.0);
    }

    /// 密度影响颗粒数：高密度起点更多（总能量更大）。
    #[test]
    fn density_increases_overlap() {
        let src: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.02).sin()).collect();
        let low = GrainSynth { density: 10.0, grain_secs: 0.05, ..Default::default() };
        let high = GrainSynth { density: 100.0, grain_secs: 0.05, ..Default::default() };
        let el: f64 = low.render(&src, 8_000, 8_000).unwrap().iter().map(|&x| (x * x) as f64).sum();
        let eh: f64 = high.render(&src, 8_000, 8_000).unwrap().iter().map(|&x| (x * x) as f64).sum();
        assert!(eh > el, "高密度应更多叠加能量：low={el} high={eh}");
    }
}
