//! 混响：Schroeder 梳状+全通级联 与 FDN 反馈延迟网。
//!
//! # Schroeder 结构
//!
//! 4 梳状滤波器（并联）→ 3 全通（串联）。梳状延迟长度取互相质素的样本数，
//! 避免梳齿共振叠加成单频；反馈系数由 RT60 反推：
//!
//! ```text
//! g = 10^(−3·d / (RT60·SR))   3 倍因子：衰减 60 dB 对应幅度 ×10⁻³
//! ```
//!
//! 全通 `y[i] = −g·x[i] + x[i−d] + g·y[i−d]`（平坦幅度、色散相位）。
//!
//! # FDN
//!
//! `N` 条延迟线经正交混合矩阵耦合（此处用确定性 Hadamard 矩阵——元素 ±1/√N，
//! 无乘法近似、无 RNG），反馈增益同 RT60 反推。结构比 Schroeder 更稠密、无
//! 早期梳齿感。
//!
//! 一切延迟长度量化为整数样本（格点对齐），处理单遍索引递增（固定序），
//! 无跨调用状态 → seek/重算等价。

use super::MonoFx;
use crate::det;

/// Schroeder 混响参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchroederReverb {
    /// RT60 混响时长（秒）。
    pub rt60_secs: f32,
    /// 干湿比 ∈ [0, 1]（1 = 全湿）。
    pub mix: f32,
    /// 阻尼 ∈ [0, 1)：每反射高频衰减系数（一阶低通）。
    pub damping: f32,
}

impl Default for SchroederReverb {
    fn default() -> Self {
        Self { rt60_secs: 1.8, mix: 0.3, damping: 0.2 }
    }
}

/// 梳状滤波器延迟长度基数（质素样本，@44.1k 口径；按 SR 缩放）。
const COMB_BASE: [usize; 4] = [1_557, 1_617, 1_491, 1_422];
/// 全通延迟长度基数。
const ALLPASS_BASE: [usize; 3] = [225, 556, 441];

/// 由 RT60 反推反馈系数：`g = 10^(−3d/(RT60·SR))`。
fn rt60_gain(delay_samples: usize, rt60_secs: f32, sr: u32) -> f64 {
    let d = delay_samples as f64;
    let t = (rt60_secs.max(1e-3) as f64) * sr as f64;
    det::exp_f64(-3.0 * d / t * std::f64::consts::LN_10)
}

/// 把基数延迟长度按 `sr/44100` 缩放为整数样本（≥1）。
fn scale_delay(base: usize, sr: u32) -> usize {
    ((base as f64 * sr as f64 / 44_100.0).round() as usize).max(1)
}

impl MonoFx for SchroederReverb {
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let n = input.len();
        let comb_delays: Vec<usize> = COMB_BASE.iter().map(|&b| scale_delay(b, sample_rate)).collect();
        let ap_delays: Vec<usize> = ALLPASS_BASE.iter().map(|&b| scale_delay(b, sample_rate)).collect();
        let damp = self.damping.clamp(0.0, 0.999) as f32;

        // 并联梳状：各自独立缓冲，湿信号求和后取平均（固定序）。
        let mut wet = vec![0.0_f32; n];
        for &d in &comb_delays {
            let g = rt60_gain(d, self.rt60_secs, sample_rate) as f32;
            let mut buf = vec![0.0_f32; d];
            let mut lp = 0.0_f32;
            for i in 0..n {
                let delayed = buf[i % d];
                lp = delayed * (1.0 - damp) + lp * damp; // 一阶阻尼低通（反馈路径）
                let y = input[i] + g * lp;
                buf[i % d] = y;
                wet[i] += y;
            }
        }
        let inv = 1.0 / comb_delays.len() as f32;
        for w in wet.iter_mut() {
            *w *= inv;
        }

        // 串联全通（平坦幅度，色散相位）。
        for &d in &ap_delays {
            let g = 0.5_f32;
            let mut buf = vec![0.0_f32; d];
            for i in 0..n {
                let buffered = buf[i % d];
                let y = -g * wet[i] + buffered;
                buf[i % d] = wet[i] + g * y;
                wet[i] = y;
            }
        }

        let mix = self.mix.clamp(0.0, 1.0);
        (0..n).map(|i| input[i] * (1.0 - mix) + wet[i] * mix).collect()
    }
}

/// FDN 混响参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdnReverb {
    /// RT60（秒）。
    pub rt60_secs: f32,
    /// 干湿比。
    pub mix: f32,
    /// 延迟线数量（4 / 8；用 ≤8 的 Hadamard 阶）。
    pub lines: usize,
}

impl Default for FdnReverb {
    fn default() -> Self {
        Self { rt60_secs: 2.0, mix: 0.3, lines: 4 }
    }
}

/// FDN 延迟线基数（质素风格，@44.1k）。
const FDN_BASE: [usize; 8] = [1_681, 1_601, 2_059, 1_927, 1_379, 1_257, 1_349, 1_543];

impl MonoFx for FdnReverb {
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let n_lines = self.lines.clamp(1, 8);
        // Hadamard 阶：≥ n_lines 的 2 的幂。
        let order = n_lines.next_power_of_two();
        let delays: Vec<usize> = (0..n_lines).map(|k| scale_delay(FDN_BASE[k], sample_rate)).collect();
        let gains: Vec<f32> =
            delays.iter().map(|&d| rt60_gain(d, self.rt60_secs, sample_rate) as f32).collect();
        let norm = 1.0 / (order as f32).sqrt();

        let mut bufs: Vec<Vec<f32>> = delays.iter().map(|&d| vec![0.0_f32; d]).collect();
        let mut tails: Vec<f32> = vec![0.0_f32; n_lines];
        let mut wet = vec![0.0_f32; input.len()];

        for i in 0..input.len() {
            // 读出各线尾部。
            for l in 0..n_lines {
                tails[l] = bufs[l][i % delays[l]];
            }
            // Hadamard 混合（±1/√order）：以 n_lines≤order，缺省线输入 0。
            let mut mixed = vec![0.0_f32; n_lines];
            for l in 0..n_lines {
                let mut acc = 0.0_f32;
                for m in 0..order {
                    let xm = if m < n_lines { tails[m] } else { 0.0 };
                    // Hadamard H[l][m] = (−1)^popcount(l & m)。
                    let sign = if ((l & m).count_ones() & 1) == 0 { 1.0 } else { -1.0 };
                    acc += sign * xm;
                }
                mixed[l] = acc * norm;
            }
            // 注入 + 反馈写入。
            for l in 0..n_lines {
                let y = input[i] + gains[l] * mixed[l];
                bufs[l][i % delays[l]] = y;
            }
            // 湿信号 = 各线尾部和（取均）。
            wet[i] = tails.iter().sum::<f32>() / n_lines as f32;
        }

        let mix = self.mix.clamp(0.0, 1.0);
        (0..input.len()).map(|i| input[i] * (1.0 - mix) + wet[i] * mix).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 冲激响应衰减：RT60 后能量应大幅下降（混响尾巴应单调衰减不爆炸）。
    #[test]
    fn schroeder_impulse_decays() {
        let sr = 44_100_u32;
        let reverb = SchroederReverb { rt60_secs: 0.5, mix: 1.0, damping: 0.3 };
        let mut input = vec![0.0_f32; sr as usize]; // 1 秒
        input[0] = 1.0;
        let out = reverb.process(&input, sr);
        // 早期能量 vs 尾部能量。
        let e_head: f64 = out[..sr as usize / 10].iter().map(|&x| (x * x) as f64).sum();
        let e_tail: f64 = out[sr as usize * 9 / 10..].iter().map(|&x| (x * x) as f64).sum();
        assert!(e_tail < e_head * 0.05, "尾部能量应远小于头部：head={e_head} tail={e_tail}");
        assert!(out.iter().all(|v| v.is_finite() && v.abs() < 10.0), "混响应稳定不发散");
    }

    /// RT60 越长，尾部能量越大（时长单调性）。
    #[test]
    fn longer_rt60_retains_more_tail() {
        let sr = 22_050_u32;
        let mk = |rt| SchroederReverb { rt60_secs: rt, mix: 1.0, damping: 0.1 };
        let mut input = vec![0.0_f32; sr as usize];
        input[0] = 1.0;
        let short: f64 = mk(0.2).process(&input, sr)[sr as usize / 2..].iter().map(|&x| (x * x) as f64).sum();
        let long: f64 = mk(2.0).process(&input, sr)[sr as usize / 2..].iter().map(|&x| (x * x) as f64).sum();
        assert!(long > short, "RT60 长应保留更多尾能：short={short} long={long}");
    }

    /// FDN：同样衰减稳定、确定性。
    #[test]
    fn fdn_impulse_stable_and_deterministic() {
        let sr = 44_100_u32;
        let reverb = FdnReverb { rt60_secs: 1.0, mix: 1.0, lines: 4 };
        let mut input = vec![0.0_f32; sr as usize];
        input[0] = 1.0;
        let a = reverb.process(&input, sr);
        let b = reverb.process(&input, sr);
        assert_eq!(a, b, "逐位一致");
        assert!(a.iter().all(|v| v.is_finite() && v.abs() < 10.0));
        // 早期回声窗（延迟线首次返回后）能量应远大于末段（RT60=1s，尾部衰减）。
        let peak: f64 = a.iter().map(|&x| (x * x) as f64).fold(0.0, f64::max);
        let e_tail: f64 = a[sr as usize - 1_000..].iter().map(|&x| (x * x) as f64).sum();
        assert!(peak > 0.0, "应有回声能量");
        assert!(e_tail < peak * 0.5, "FDN 尾部应大幅衰减：peak={peak} tail_sum={e_tail}");
    }

    /// 干湿比：mix=0 完全直通。
    #[test]
    fn zero_mix_is_dry_passthrough() {
        let reverb = SchroederReverb { rt60_secs: 1.0, mix: 0.0, damping: 0.0 };
        let input: Vec<f32> = (0..500).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = reverb.process(&input, 44_100);
        assert_eq!(out, input);
    }
}
