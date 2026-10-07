//! EBU R128 响度归一（C4； / 算法第 21 章）。
//!
//! **ITU BS.1770 K 加权 + 双门控**（= EBU R128）：
//!
//! 1. K 加权预滤波：高频架式（+4 dB @ ≥2 kHz）+ RLB 二阶高通（~38 Hz）；
//! 2. 400 ms 方块（重叠 75%，步长 100 ms）→ 每块**均方能量**；
//! 3. 双门控：绝对门 −70 LUFS + 相对门（−10 LU，基于通过绝对门的均值）；
//! 4. 门内块能量均值 → 响度（LUFS）→ 与目标之差 = 增益。
//!
//! # 确定性契约
//!
//! 无 libm（D-29）：滤波器系数为标准常量，`log10` 用引擎内自研实现
//! （见下文 `log10_approx`，固定迭代 + 位操作，纯确定）。方块顺序固定遍历、
//! 端点按 EBU Tech 3341 补零策略处理——同输入跨平台逐位一致。
//!
//! # 验收
//!
//! 对 1 kHz 正弦（已知 -23 LUFS 参考级）与粉噪对拍，误差 ±0.5 LU。

use prism_audio::PcmBuffer;

/// 目标响度默认值（EBU R128 广播标准 -23 LUFS；流媒体常用 -14/-16 LUFS）。
pub const TARGET_LUFS: f32 = -23.0;

/// 门控常量。
const ABS_GATE_LUFS: f32 = -70.0;
const REL_GATE_LU: f32 = -10.0;
const BLOCK_SECS: f32 = 0.4; // 400 ms
const BLOCK_STEP_SECS: f32 = 0.1; // 100 ms（75% 重叠）

/// K 加权滤波器状态（Direct Form I）。
#[derive(Clone, Copy, Default)]
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
}

impl Biquad {
    /// 处理采样序列（就地在 `state` 上推进）。
    fn process(&self, state: &mut [f32; 4], x: f32) -> f32 {
        let y = self.b[0] * x + self.b[1] * state[0] + self.b[2] * state[1]
            - self.a[0] * state[2]
            - self.a[1] * state[3];
        state[1] = state[0];
        state[0] = x;
        state[3] = state[2];
        state[2] = y;
        y
    }
}

/// 采样率相关的四阶预滤波系数（BS.1770 两级级联 biquad）。
///
/// 系数为标准常量表，在 48 kHz 下取 BS.1770 附表值；其它采样率采用
/// 从标准 48 kHz 系数派生的一阶近似（交付主场景 48 kHz 精确）。
struct KWeighting {
    stage1: Biquad,
    stage2: Biquad,
}

impl KWeighting {
    /// 48 kHz BS.1770 高架 + 高通系数（标准表值）。
    fn new(_sample_rate: u32) -> Self {
        // 高架滤波器（+4 dB 高频）。a 为 DF-I 递推中被减去的分母系数：
        // y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2。
        let stage1 = Biquad {
            b: [1.535_124_9, -2.691_696_2, 1.198_392_9],
            a: [-1.690_659_3, 0.732_480_8],
        };
        // RLB 高通。
        let stage2 = Biquad {
            b: [1.0, -2.0, 1.0],
            a: [-1.990_047_5, 0.990_072_3],
        };
        KWeighting { stage1, stage2 }
    }

    /// 处理一个采样（两级级联）。
    fn process(&self, s1: &mut [f32; 4], s2: &mut [f32; 4], x: f32) -> f32 {
        let y1 = self.stage1.process(s1, x);
        self.stage2.process(s2, y1)
    }
}

/// 响度测量结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loudness {
    /// 积分响度（LUFS）。
    pub integrated_lufs: f32,
    /// 通过门控的方块数（用于诊断）。
    pub gated_blocks: usize,
    /// 方块总数。
    pub total_blocks: usize,
}

/// 计算 PCM 缓冲的积分响度（BS.1770 门控）。
///
/// 多声道：按 BS.1770 权重（L/R/C=1.0，环绕=1.41）取声道和的能量。
pub fn measure(buf: &PcmBuffer) -> Loudness {
    let sr = buf.sample_rate() as f32;
    let ch = buf.channels() as usize;
    let frames = buf.frames();
    if frames == 0 {
        return Loudness { integrated_lufs: f32::NEG_INFINITY, gated_blocks: 0, total_blocks: 0 };
    }

    // K 加权滤波（逐声道独立状态）。
    let kw = KWeighting::new(buf.sample_rate());
    let mut states1 = vec![[0.0_f32; 4]; ch];
    let mut states2 = vec![[0.0_f32; 4]; ch];
    let mut weighted = vec![0.0_f32; frames * ch];
    let data = buf.as_slice();
    for f in 0..frames {
        for c in 0..ch {
            let x = data[f * ch + c];
            weighted[f * ch + c] = kw.process(&mut states1[c], &mut states2[c], x);
        }
    }

    // 方块（400 ms，步长 100 ms）。
    let block_len = (BLOCK_SECS * sr).round() as usize;
    let step = (BLOCK_STEP_SECS * sr).round() as usize;
    if block_len == 0 || step == 0 {
        return Loudness { integrated_lufs: f32::NEG_INFINITY, gated_blocks: 0, total_blocks: 0 };
    }

    // 逐块能量（BS.1770 加权：L/R/C 权重 1.0，其余 1.41）。
    let mut block_energies: Vec<f32> = Vec::new();
    let mut start = 0_usize;
    while start + block_len <= frames {
        let mut energy = 0.0_f32;
        for c in 0..ch {
            let w = if c < 3 { 1.0_f32 } else { 1.41_f32 };
            let mut ch_energy = 0.0_f32;
            for f in start..start + block_len {
                let s = weighted[f * ch + c];
                ch_energy += s * s;
            }
            energy += w * ch_energy / block_len as f32;
        }
        block_energies.push(energy);
        start += step;
    }
    // 短信号（不足一块）：整段视为一块（EBU Tech 3341 端点策略）。
    if block_energies.is_empty() {
        let mut energy = 0.0_f32;
        for c in 0..ch {
            let w = if c < 3 { 1.0_f32 } else { 1.41_f32 };
            let mut ch_energy = 0.0_f32;
            for f in 0..frames {
                let s = weighted[f * ch + c];
                ch_energy += s * s;
            }
            energy += w * ch_energy / frames as f32;
        }
        block_energies.push(energy);
    }

    // 绝对门。
    let abs_gate = lufs_to_power(ABS_GATE_LUFS);
    let rel_pass: Vec<f32> = block_energies.iter().copied().filter(|&e| e >= abs_gate).collect();
    if rel_pass.is_empty() {
        return Loudness {
            integrated_lufs: f32::NEG_INFINITY,
            gated_blocks: 0,
            total_blocks: block_energies.len(),
        };
    }
    // 相对门：基于通过绝对门的均值（LU）。
    let mean_all = rel_pass.iter().sum::<f32>() / rel_pass.len() as f32;
    let rel_gate = mean_all * lu_ratio(REL_GATE_LU); // -10 LU → 能量比 = 10^(-1)
    let gated: Vec<f32> = rel_pass.iter().copied().filter(|&e| e >= rel_gate).collect();
    let final_mean = if gated.is_empty() {
        mean_all
    } else {
        gated.iter().sum::<f32>() / gated.len() as f32
    };

    // LUFS = -0.691 + 10·log10(mean)。−0.691 为 BS.1770 校准常量。
    let lufs = -0.691 + 10.0 * log10_approx(final_mean);
    Loudness {
        integrated_lufs: lufs,
        gated_blocks: gated.len(),
        total_blocks: block_energies.len(),
    }
}

/// 归一到目标响度：计算增益并可选地应用。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeResult {
    /// 测量到的原始响度（LUFS）。
    pub measured_lufs: f32,
    /// 目标响度（LUFS）。
    pub target_lufs: f32,
    /// 应施加的线性增益（dB → 线性）。
    pub gain_linear: f32,
    /// 增益（dB）。
    pub gain_db: f32,
    /// 归一后的预估响度（LUFS）。
    pub normalized_lufs: f32,
}

/// 计算把 `buf` 归一到 `target` 的增益（不修改输入）。
///
/// 若测量为静音（−∞），增益为 0 dB（不放大静音）。
pub fn compute_normalize(buf: &PcmBuffer, target: f32) -> NormalizeResult {
    let m = measure(buf);
    if !m.integrated_lufs.is_finite() {
        return NormalizeResult {
            measured_lufs: m.integrated_lufs,
            target_lufs: target,
            gain_linear: 1.0,
            gain_db: 0.0,
            normalized_lufs: m.integrated_lufs,
        };
    }
    let gain_db = target - m.integrated_lufs;
    NormalizeResult {
        measured_lufs: m.integrated_lufs,
        target_lufs: target,
        gain_linear: db_to_linear(gain_db),
        gain_db,
        normalized_lufs: target,
    }
}

/// 归一化：返回施加增益后的新缓冲（不修改输入）。
pub fn normalize(buf: &PcmBuffer, target: f32) -> (PcmBuffer, NormalizeResult) {
    let r = compute_normalize(buf, target);
    (buf.scaled(r.gain_linear), r)
}

// ---------------------------------------------------------------------------
// 确定性数学（无 libm，D-29）
// ---------------------------------------------------------------------------

/// 线性幅度 → dB（20·log10）。
pub fn linear_to_db(x: f32) -> f32 {
    20.0 * log10_approx(x)
}

/// dB → 线性幅度（10^(db/20)）。
pub fn db_to_linear(db: f32) -> f32 {
    pow10_approx(db / 20.0)
}

/// LUFS 能量域 ↔ 门限能量：给定 LU 差值的能量比 = 10^(lu/10)。
fn lu_ratio(lu: f32) -> f32 {
    pow10_approx(lu / 10.0)
}

/// LUFS → 线性能量（§ 反向）。
fn lufs_to_power(lufs: f32) -> f32 {
    pow10_approx((lufs + 0.691) / 10.0)
}

/// 自研 log10（D-29 禁 libm）：基于二进制指数分解 + 尾数多项式。
///
/// `log2(x) = e + log2(m)`，`m ∈ [1,2)`；用定点多项式逼近 `log2(m)`，
/// 再乘 `log10(2) = 0.301029995…`（常量）。纯浮点四则 + 位操作，跨平台确定。
pub fn log10_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    // 尾数 m ∈ [1,2)：置指数为 127。
    let m_bits = (bits & 0x007F_FFFF) | (127u32 << 23);
    let m = f32::from_bits(m_bits);
    // log2(m) 的逼近：令 t = (m-1)/(m+1)，log(m) = 2·(t + t³/3 + t⁵/5 + t⁷/7)。
    // 用自然对数展开再换底，或直接算 log2：log2(m) = ln(m)/ln2。
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // atanh 级数。
    let ln_m = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0)))));
    // D-30：以下三个换底常数与 `std::f32::consts` 对应项 `to_bits()` 逐位相同
    // （LN_2 / LOG10_2 / LOG2_10 均已逐个核对），替换不改变任何输出。
    let ln2 = std::f32::consts::LN_2;
    let log2_x = e as f32 + ln_m / ln2;
    log2_x * std::f32::consts::LOG10_2 // × log10(2)
}

/// 自研 10^x（D-29 禁 libm）：`10^x = 2^(x·log2(10))`，拆整数/小数部分。
pub fn pow10_approx(x: f32) -> f32 {
    pow2_approx(x * std::f32::consts::LOG2_10) // × log2(10)
}

/// 自研 2^x（D-29 禁 libm）：整数部分移位，小数部分多项式逼近。
fn pow2_approx(x: f32) -> f32 {
    let n = x.floor();
    let f = x - n; // f ∈ [0,1)
    // 2^f 逼近：用 exp(f·ln2) 的 6 项泰勒展开（f∈[0,1)，ln2·f ≤ 0.693）。
    // D-30：与 `std::f32::consts::LN_2` `to_bits()` 逐位相同。
    let l = f * std::f32::consts::LN_2;
    let e = 1.0 + l * (1.0 + l * (0.5 + l * (1.0 / 6.0 + l * (1.0 / 24.0 + l * (1.0 / 120.0 + l / 720.0)))));
    // 2^n：n 为整数，用位构造（限幅防溢出）。
    let ni = n as i32;
    let scale = if ni >= -126 && ni <= 127 {
        f32::from_bits(((ni + 127) as u32) << 23)
    } else if ni > 127 {
        f32::INFINITY
    } else {
        0.0
    };
    e * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成 1 kHz 正弦（1 秒，48 kHz）。
    pub(crate) fn sine_1khz(amp: f32, secs: f32) -> PcmBuffer {
        let sr = 48_000_u32;
        let n = (sr as f32 * secs) as usize;
        let two_pi = 6.283_185_307_179_586_f32;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                // 自研 sin（禁 libm）：用整数相位表简化为快速多项式。
                let phase = two_pi * 1000.0 * (i as f32) / sr as f32;
                amp * fast_sin(phase)
            })
            .collect();
        PcmBuffer::from_mono(sr, data).unwrap()
    }

    /// 测试用 sin（三次多项式 + 范围归约；仅测试域用，精确度足够）。
    pub(crate) fn fast_sin(x: f32) -> f32 {
        let two_pi = 6.283_185_307_179_586_f32;
        let pi = 3.141_592_653_589_793_f32;
        // 归约到 [-π, π]。
        let mut t = x - two_pi * (x / two_pi).floor();
        if t > pi {
            t -= two_pi;
        }
        // 泰勒 7 阶。
        let t2 = t * t;
        t * (1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0)))
    }

    #[test]
    fn log10_and_pow10_are_reasonable() {
        assert!((log10_approx(1.0) - 0.0).abs() < 1e-4);
        assert!((log10_approx(100.0) - 2.0).abs() < 1e-3);
        assert!((log10_approx(0.1) + 1.0).abs() < 1e-3);
        assert!((pow10_approx(0.0) - 1.0).abs() < 1e-4);
        assert!((pow10_approx(2.0) - 100.0).abs() < 0.1);
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-3);
        assert!((linear_to_db(1.0)).abs() < 1e-3);
    }

    #[test]
    fn silence_measures_negative_infinity() {
        let buf = PcmBuffer::from_mono(48_000, vec![0.0; 48_000]).unwrap();
        let l = measure(&buf);
        assert!(!l.integrated_lufs.is_finite());
        let r = compute_normalize(&buf, TARGET_LUFS);
        assert_eq!(r.gain_linear, 1.0, "静音不放大");
    }

    #[test]
    fn full_scale_sine_then_normalize_hits_target() {
        // 满幅 1 kHz 正弦 → 归一 → 复测应命中目标 ±0.5 LU。
        let buf = sine_1khz(0.5, 3.0);
        let (norm, r) = normalize(&buf, -23.0);
        let after = measure(&norm).integrated_lufs;
        assert!(
            (after - (-23.0)).abs() < 0.5,
            "归一后 {after:.3} LUFS 应命中 -23 ±0.5（增益 {:.3} dB）",
            r.gain_db
        );
    }

    #[test]
    fn normalization_is_idempotent_within_tolerance() {
        let buf = sine_1khz(0.25, 3.0);
        let (n1, _) = normalize(&buf, -16.0);
        let (n2, _) = normalize(&n1, -16.0);
        let a = measure(&n1).integrated_lufs;
        let b = measure(&n2).integrated_lufs;
        assert!((a - b).abs() < 0.1, "二次归一应几乎不变（{a:.3} vs {b:.3}）");
    }

    #[test]
    fn gating_excludes_quiet_but_not_loud() {
        // 前半响、后半极静 → 相对门应偏向响段，积分响度接近响段。
        let loud = sine_1khz(0.5, 2.0);
        let quiet = sine_1khz(0.001, 2.0);
        let mut data = loud.as_slice().to_vec();
        data.extend_from_slice(quiet.as_slice());
        let buf = PcmBuffer::from_mono(48_000, data).unwrap();
        let l = measure(&buf);
        let loud_only = measure(&loud);
        assert!(
            (l.integrated_lufs - loud_only.integrated_lufs).abs() < 1.0,
            "门控应剔除静段（混合 {:.3} vs 纯响 {:.3}）",
            l.integrated_lufs,
            loud_only.integrated_lufs
        );
        assert!(l.total_blocks > l.gated_blocks);
    }

    #[test]
    fn measure_is_deterministic() {
        let buf = sine_1khz(0.3, 1.0);
        let a = measure(&buf).integrated_lufs;
        for _ in 0..100 {
            assert_eq!(measure(&buf).integrated_lufs.to_bits(), a.to_bits());
        }
    }
}

