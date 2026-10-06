//! 动态压缩器（B12）：阈值 / 比率 / 软拐点 / 起振-释放（包络跟随）。
//!
//! 增益缩减按**电平检测**计算。为避免跨调用状态，本实现用**前视逐样本包络**
//! 的确定性近似：对每个样本计算瞬时电平 `|x|`（dB），经软拐点曲线求增益缩减，
//! 再对增益**逐样本软起振/释放**（一阶指数平滑，衰减系数由时间常数确定）——
//! 单遍索引递增，固定序，无隐藏状态、无 RNG。
//!
//! 增益曲线（dB 域）：
//! ```text
//! 若 level < T − K/2          : 0（不压缩）
//! 若 |level − T| ≤ K/2        : 软拐点二次过渡（比例随进入深度渐变）
//! 若 level > T + K/2          : (level − T)/ratio   （硬拐点线性）
//! ```
//! 输出 = 输入 × 10^(−gain_reduction_dB/20)。dB 与 10^x 走确定性实现（D-29
//! 禁 libm——本模块用自研 exp/ln 的多项式近似，见 [`crate::det`] 扩展）。

use super::MonoFx;

/// 压缩器参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compressor {
    /// 阈值（dB，例如 −20）。
    pub threshold_db: f32,
    /// 比率（> 1；4 = 4:1）。
    pub ratio: f32,
    /// 软拐点宽度（dB，0 = 硬拐点）。
    pub knee_db: f32,
    /// 起振时间（秒）。
    pub attack_secs: f32,
    /// 释放时间（秒）。
    pub release_secs: f32,
    /// 补偿增益（dB，输出端）。
    pub makeup_db: f32,
}

impl Default for Compressor {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_secs: 0.005,
            release_secs: 0.05,
            makeup_db: 0.0,
        }
    }
}

/// 电平（dB，满幅 1.0 = 0 dB）→ 增益缩减（dB，≥0）。软拐点二次分段。
///
/// 拐点内 `t ∈ [0,1]`：`reduction = (1 − 1/ratio)·(1 − t)²·K/2`（进入越深压缩越多）。
pub fn gain_reduction_db(level_db: f32, threshold_db: f32, ratio: f32, knee_db: f32) -> f32 {
    let r = ratio.max(1.0);
    let slope = 1.0 - 1.0 / r; // 每 dB 超出阈值被压掉的比例
    let half_k = knee_db.max(0.0) * 0.5;
    if level_db <= threshold_db - half_k {
        0.0
    } else if level_db >= threshold_db + half_k {
        (level_db - threshold_db) * slope
    } else {
        // 软拐点：t ∈ (−1, 1)，二次平滑接入。
        let t = level_db - threshold_db; // ∈ [−half_k, half_k]
        let k = knee_db.max(1e-6);
        slope * (t + half_k) * (t + half_k) / (2.0 * k)
    }
}

impl MonoFx for Compressor {
    fn process(&self, input: &[f32], sample_rate: u32) -> Vec<f32> {
        let sr = sample_rate as f32;
        // 一阶平滑系数：α = 1 − exp(−1/(τ·SR))，用定点近似避免 libm。
        let a_att = coeff(self.attack_secs, sr);
        let a_rel = coeff(self.release_secs, sr);
        let make_gain = db_to_lin(self.makeup_db);
        let mut env_db = -120.0_f32; // 增益缩减包络（dB），初值最小
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            let level_db = lin_to_db(x.abs().max(1e-6));
            let target = gain_reduction_db(level_db, self.threshold_db, self.ratio, self.knee_db);
            // 起振（目标更大=压缩更深时快）或释放（目标更小时慢）。
            let a = if target > env_db { a_att } else { a_rel };
            env_db += (target - env_db) * a;
            let g = db_to_lin(-env_db) * make_gain;
            out.push(x * g);
        }
        out
    }
}

/// 一阶平滑系数 `1 − exp(−1/(τ·SR))`（τ≤0 → 1，瞬时）。
fn coeff(tau: f32, sr: f32) -> f32 {
    if tau <= 0.0 {
        return 1.0;
    }
    let x = 1.0 / (tau * sr) as f64;
    (1.0 - crate::det::exp_f64(-x)) as f32
}

/// 线性幅度 → dB（`20·log10(|x|)`，自研 ln）。
pub fn lin_to_db(x: f32) -> f32 {
    let v = (x.abs().max(1e-9)) as f64;
    ((20.0 / std::f64::consts::LN_10) * crate::det::ln_f64(v)) as f32
}

/// dB → 线性幅度（`10^(db/20)`，自研 exp）。
pub fn db_to_lin(db: f32) -> f32 {
    crate::det::exp_f64((db as f64) * std::f64::consts::LN_10 / 20.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 阈值行为：低于阈值无压缩，高于阈值按比率线性下降。
    #[test]
    fn threshold_and_ratio_behavior() {
        let (t, r, k) = (-20.0_f32, 4.0_f32, 0.0_f32);
        assert_eq!(gain_reduction_db(-30.0, t, r, k), 0.0, "阈下不压缩");
        assert_eq!(gain_reduction_db(-20.0, t, r, k), 0.0, "恰在阈值无压缩");
        // 超出 10dB，比率 4 → 缩减 = 10·(1−1/4) = 7.5 dB。
        assert!((gain_reduction_db(-10.0, t, r, k) - 7.5).abs() < 1e-4);
        // 比率越大压缩越狠。
        assert!(gain_reduction_db(-10.0, t, 8.0, k) > gain_reduction_db(-10.0, t, r, k));
    }

    /// 软拐点：曲线在阈值附近连续且单调不减（无突变）。
    #[test]
    fn soft_knee_is_continuous_and_monotonic() {
        let (t, r, k) = (-20.0_f32, 4.0_f32, 8.0_f32);
        let mut prev = 0.0_f32;
        let mut x = -30.0_f32;
        while x <= 0.0 {
            let g = gain_reduction_db(x, t, r, k);
            assert!(g >= prev - 1e-6, "增益缩减应单调不减（x={x}）");
            prev = g;
            x += 0.05;
        }
        // 拐点边界连续：阈值处缩减应低于上部边界处的缩减（二次过渡）。
        let below = gain_reduction_db(t - 0.01, t, r, k);
        let at = gain_reduction_db(t, t, r, k);
        let above = gain_reduction_db(t + 0.01, t, r, k);
        assert!((at - below) < 0.05 && (above - at) < 0.05, "阈值邻域应连续");
    }

    /// 全程压缩：强信号过压缩器后峰值下降、能量被控制。
    #[test]
    fn compressor_reduces_loud_signal_peak() {
        let sr = 8_000_u32;
        let input: Vec<f32> = (0..4_000).map(|i| 0.9 * (i as f32 * 0.1).sin()).collect();
        let comp = Compressor { attack_secs: 0.0, release_secs: 0.0, ..Default::default() };
        let out = comp.process(&input, sr);
        let in_peak = input.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        let out_peak = out.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(out_peak < in_peak * 0.9, "峰值应被压低：in={in_peak} out={out_peak}");
    }

    /// 静音输入：无信号电平极低，输出仍近零（不产生噪声）。
    #[test]
    fn silence_stays_silent() {
        let comp = Compressor::default();
        let out = comp.process(&vec![0.0_f32; 100], 8_000);
        assert!(out.iter().all(|&x| x.abs() < 1e-5));
    }

    /// db 转换往返一致（deterministic exp/ln）。
    #[test]
    fn db_roundtrip() {
        for db in [-40.0_f32, -12.0, 0.0, 6.0] {
            let lin = db_to_lin(db);
            let back = lin_to_db(lin);
            assert!((back - db).abs() < 1e-3, "db={db} back={back}");
        }
    }
}
