//! radix-2 Cooley-Tukey FFT（B6；白皮书 §20：蝶形运算固定顺序，确定性）。
//!
//! 实现：迭代式 DIT（时域抽取）——输入做位反转置换，再按 `len = 2, 4, …, n`
//! 逐级蝶形；每级内部按下标升序三重循环，**运算顺序唯一固定**（白皮书 §20
//! "蝴蝶运算固定顺序"），同输入跨平台逐位一致。旋转因子在构造期由 [`crate::det`]
//! 确定性三角件算出（f64 内部 → f32 一次舍入），运行期零超越函数调用。
//!
//! 复杂度：构造 O(n)，正向变换 O(n log n)。输出不做归一化（原始 DFT 和），
//! 能量口径由调用方约定（Parseval：Σ|X|² = n·Σ|x|²）。

use crate::det;
use crate::{AudioError, AudioResult};

/// radix-2 FFT 实例：位反转表 + 旋转因子表（`n` 固定，可对任意多组输入复用）。
#[derive(Debug, Clone)]
pub struct Fft {
    n: usize,
    /// 位反转置换表：`rev[i]` = i 的 log₂n 位逆序。
    rev: Vec<u32>,
    /// 旋转因子 cos 表：`cos_tab[k] = cos(2πk/n)`，k ∈ [0, n/2)。
    cos_tab: Vec<f32>,
    /// 旋转因子 sin 表（负号已并入）：`sin_tab[k] = -sin(2πk/n)`。
    sin_tab: Vec<f32>,
}

impl Fft {
    /// 构造 `n` 点 FFT（`n` 必须为 ≥2 的 2 的幂，否则 `AU1002`）。
    pub fn new(n: usize) -> AudioResult<Self> {
        if n < 2 || !n.is_power_of_two() {
            return Err(AudioError::new(
                "AU1002",
                format!("Fft::new(n={n})"),
                "FFT 点数必须为 ≥2 的 2 的幂（radix-2 Cooley-Tukey 的结构前提）",
                "窗长取 1024/2048 等常规值，或对输入补零到 2 的幂",
            ));
        }
        let bits = n.trailing_zeros();
        let rev = (0..n as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        let half = n / 2;
        let mut cos_tab = Vec::with_capacity(half);
        let mut sin_tab = Vec::with_capacity(half);
        for k in 0..half {
            let ang = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            cos_tab.push(det::cos_f64(ang) as f32);
            sin_tab.push(det::sin_f64(ang) as f32);
        }
        Ok(Self { n, rev, cos_tab, sin_tab })
    }

    /// 变换点数。
    pub fn n(&self) -> usize {
        self.n
    }

    /// 正向 DFT：`X[m] = Σ x[i]·e^(-2πi·mi/n)`。输入长度必须为 `n`（否则
    /// `AU1001`）。返回 (实部, 虚部)，未归一化。
    pub fn forward(&self, input: &[f32]) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        if input.len() != self.n {
            return Err(AudioError::new(
                "AU1001",
                format!("Fft(n={}).forward", self.n),
                format!("输入长度 {} 与变换点数 {} 不一致", input.len(), self.n),
                format!("补零或截断输入到 {} 点", self.n),
            ));
        }
        let n = self.n;
        let mut re = vec![0.0_f32; n];
        let mut im = vec![0.0_f32; n];
        for (i, &x) in input.iter().enumerate() {
            re[self.rev[i] as usize] = x;
        }
        let mut len = 2_usize;
        while len <= n {
            let half = len / 2;
            let tstep = n / len; // 旋转因子表步进：w_len^j = w_n^(j·n/len)
            let mut start = 0;
            while start < n {
                let mut j = 0;
                while j < half {
                    let k = j * tstep;
                    let (wr, wi) = (self.cos_tab[k], self.sin_tab[k]);
                    let vr = re[start + j + half];
                    let vi = im[start + j + half];
                    // 蝶形：t = v·w（w = cosθ − i·sinθ），(u+t, u−t) 原地写回。
                    let tr = vr * wr - vi * wi;
                    let ti = vr * wi + vi * wr;
                    re[start + j + half] = re[start + j] - tr;
                    im[start + j + half] = im[start + j] - ti;
                    re[start + j] += tr;
                    im[start + j] += ti;
                    j += 1;
                }
                start += len;
            }
            len <<= 1;
        }
        Ok((re, im))
    }

    /// 半谱幅值 `|X[k]|`，k ∈ [0, n/2]（实输入谱对称，上半谱冗余）。
    pub fn magnitudes(&self, input: &[f32]) -> AudioResult<Vec<f32>> {
        let (re, im) = self.forward(input)?;
        Ok(re.iter().zip(&im).map(|(&r, &i)| (r * r + i * i).sqrt()).collect())
    }
}

/// 周期 Hann 窗：`w[i] = 0.5·(1 − cos(2πi/n))`（STFT 惯例用周期形式；
/// [`crate::det`] 确定性 cos）。相干增益 0.5——幅度归一口径见 reactive。
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let ang = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            0.5 * (1.0 - det::cos_f64(ang)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 解析解对拍：n=1024、恰落 bin 17 的正弦，能量应集中于 |X[17]| = n/2
    /// （及共轭 bin），其余 bin 泄漏可忽略（验收正条）。
    #[test]
    fn single_tone_energy_concentrates_in_correct_bin() {
        let (n, bin) = (1024_usize, 17_usize);
        let fft = Fft::new(n).unwrap();
        let x: Vec<f32> = (0..n)
            .map(|i| det::sin_f64(2.0 * std::f64::consts::PI * (bin * i) as f64 / n as f64) as f32)
            .collect();
        let mag = fft.magnitudes(&x).unwrap();
        let peak = mag[bin];
        assert!((peak - n as f32 / 2.0).abs() < 1e-2, "|X[{bin}] = {peak}，解析值 {}", n / 2);
        for (k, &m) in mag.iter().enumerate() {
            if k != bin && k != n - bin {
                assert!(m < 1e-2, "bin {k} 泄漏 {m} 过大");
            }
        }
        // 相位：正弦（初相 0）在 bin 17 的谱应为 -i·n/2，即实部≈0、虚部≈-n/2。
        let (re, im) = fft.forward(&x).unwrap();
        assert!(re[bin].abs() < 1e-2 && (im[bin] + n as f32 / 2.0).abs() < 1e-2);
    }

    /// Parseval 能量守恒：Σ|X[k]|² = n·Σ|x[i]|²（f32 累积容差内）。
    #[test]
    fn parseval_energy_conservation() {
        let n = 512_usize;
        let fft = Fft::new(n).unwrap();
        let x: Vec<f32> = (0..n).map(|i| ((i * 7) as f32 * 0.031).sin() * 0.7).collect();
        let (re, im) = fft.forward(&x).unwrap();
        let spec: f64 = re.iter().zip(&im).map(|(&r, &i)| (r * r + i * i) as f64).sum();
        let time: f64 = x.iter().map(|&v| (v * v) as f64).sum();
        let rel = ((spec - n as f64 * time) / (n as f64 * time)).abs();
        assert!(rel < 1e-4, "Parseval 相对偏差 {rel}");
    }

    /// 冲激输入 → 全谱平坦 |X[k]| = 1；恒零输入 → 全零谱。
    #[test]
    fn impulse_and_silence_degenerate_cases() {
        let n = 256_usize;
        let fft = Fft::new(n).unwrap();
        let mut impulse = vec![0.0_f32; n];
        impulse[0] = 1.0;
        let mag = fft.magnitudes(&impulse).unwrap();
        for (k, &m) in mag.iter().enumerate() {
            assert!((m - 1.0).abs() < 1e-6, "冲激谱 bin {k} = {m}");
        }
        let mag0 = fft.magnitudes(&vec![0.0_f32; n]).unwrap();
        assert!(mag0.iter().all(|&m| m == 0.0));
    }

    /// 长度/点数非法 → 三段式报错；同输入重复变换逐位一致。
    #[test]
    fn validation_and_bitwise_determinism() {
        assert_eq!(Fft::new(0).unwrap_err().code, "AU1002");
        assert_eq!(Fft::new(1000).unwrap_err().code, "AU1002");
        let fft = Fft::new(64).unwrap();
        let x: Vec<f32> = (0..63).map(|i| i as f32 * 0.1).collect();
        assert_eq!(fft.forward(&x).unwrap_err().code, "AU1001");
        let full: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let first = fft.forward(&full).unwrap();
        for _ in 0..1_000 {
            let again = fft.forward(&full).unwrap();
            assert_eq!(again.0, first.0);
            assert_eq!(again.1, first.1);
        }
    }

    /// Hann 窗：端点为 0、中点为 1（周期形式 cos(π)=−1）、确定性复算一致。
    #[test]
    fn hann_window_shape() {
        let n = 8;
        let w = hann_window(n);
        assert_eq!(w[0], 0.0);
        assert!((w[n / 2] - 1.0).abs() < 1e-6);
        assert_eq!(w, hann_window(n), "同输入逐位一致");
    }
}
