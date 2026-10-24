//! 振荡器族 + ADSR 包络 + 声部渲染（合成器）。
//!
//! # 防混叠
//!
//! 方波/锯齿含不连续点，朴素波形的高次谐波超过 Nyquist 后折返成非谐和噪声。
//! 采用 **PolyBLEP**（Polynomial Band-Limited Step）：在不连续点 ±1 个相位
//! 步长窗口内叠加一个低次多项式修正残差，把阶跃修成带宽受限的过渡。修正
//! 式为标准两分支形式（构造上只用 ±×÷，确定性）：
//!
//! ```text
//! r(t, dt) = t/dt ∈ [0,1)   : 2x − x² − 1        （阶跃之后）
//!            t/dt ∈ (−1,0]  : x² + 2x + 1        （阶跃之前）
//! saw    = 2p − 1 − r(p, dt)                     （p=0 处 −1→+1 下跳）
//! square = naive + r(p, dt) − r(fract(p+½), dt)   （p=0 上跳取反号，p=½ 下跳）
//! ```
//!
//! # 纯函数相位（无积分器）
//!
//! 相位**不做累加**（状态机式振荡器 seek 必须重放历史），而是 `p_i =
//! fract(f·i/SR + φ₀)`——采样索引的纯函数，第 i 个样本独立可算（与 reactive
//! "第 N 帧独立重算" 同一确定性范式）。f64 内部运算保证长缓冲大相位精度，
//! f32 一次舍入输出。
//!
//! # ADSR 闭式包络（架构 S7 闭式线性响应）
//!
//! 分段线性闭式解 `value_at(t, note_dur)`——无状态、O(1)、seek 即求值。

pub mod filter;
pub mod lfo;
pub mod presets;
pub mod unison;

/// 波形族。三角波只有斜率不连续（PolyBLAMP 修正收益小），按朴素公式实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    /// 正弦。
    Sine,
    /// 方波（PolyBLEP）。
    Square,
    /// 锯齿（PolyBLEP）。
    Saw,
    /// 三角波。
    Triangle,
}

/// PolyBLEP 修正残差：`t` = 相位 ∈ [0,1)，`dt` = 每样本相位步长 ∈ (0, 0.5)。
/// 返回值**减**在"p=0 处下跳（+1→−1）"的波形上；上跳取反号。
#[inline]
fn poly_blep(t: f64, dt: f64) -> f64 {
    debug_assert!(dt > 0.0 && dt < 0.5, "相位步长必须在 (0, 0.5)，收到 {dt}");
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// f64 相位小数部分（`x - floor(x)`，floor 为 IEEE 精确取整）。
#[inline]
fn fract(x: f64) -> f64 {
    x - x.floor()
}

/// 振荡器单样本：波形在采样索引 `index` 处的值（纯函数；相位 = fract(f·i/SR)）。
pub fn sample_osc(waveform: Waveform, freq: f32, index: usize, sample_rate: u32) -> f32 {
    let p = fract(freq as f64 * index as f64 / sample_rate as f64); // ∈ [0, 1)
    let dt = freq as f64 / sample_rate as f64;
    let v = match waveform {
        Waveform::Sine => crate::det::sin_f64(std::f64::consts::TAU * p),
        Waveform::Saw => 2.0 * p - 1.0 - poly_blep(p, dt),
        Waveform::Square => {
            let naive = if p < 0.5 { 1.0 } else { -1.0 };
            naive + poly_blep(p, dt) - poly_blep(fract(p + 0.5), dt)
        }
        Waveform::Triangle => 4.0 * (p - 0.5).abs() - 1.0, // p=0 → −1，p=½ → +1
    };
    v as f32
}

/// ADSR 包络参数：attack/decay/release 为秒，sustain 为电平 ∈ [0, 1]。
///
/// 形状（t = 音符内时间，D = 音符时值）：
/// `t<a` 线性升 0→1；`t<a+d` 线性降 1→s；`t<D` 保持 s；`t<D+r` 线性降 s→0；
/// 其余 0。时长参数为 0 时该段跳过（闭式解天然覆盖）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adsr {
    /// Attack 时长（秒）。
    pub attack: f32,
    /// Decay 时长（秒）。
    pub decay: f32,
    /// Sustain 电平。
    pub sustain: f32,
    /// Release 时长（秒）。
    pub release: f32,
}

impl Adsr {
    /// 音符内 `t` 秒处的包络值（`note_dur` = 音符时值秒；纯函数）。
    pub fn value_at(&self, t: f32, note_dur: f32) -> f32 {
        let s = self.sustain.clamp(0.0, 1.0);
        if t < 0.0 {
            return 0.0;
        }
        if self.attack > 0.0 && t < self.attack {
            return t / self.attack;
        }
        let after_attack = if self.attack > 0.0 { self.attack } else { 0.0 };
        let decay_end = after_attack + self.decay;
        if self.decay > 0.0 && t < decay_end {
            return 1.0 - (1.0 - s) * (t - after_attack) / self.decay;
        }
        if t < note_dur {
            return s;
        }
        if self.release > 0.0 && t < note_dur + self.release {
            return s * (1.0 - (t - note_dur) / self.release);
        }
        0.0
    }
}

/// 渲染一个声部音符：`gain · ADSR(t) · 波形(i)`，长度 `frames`（纯函数）。
pub fn render_note(
    waveform: Waveform,
    freq: f32,
    env: &Adsr,
    gain: f32,
    frames: usize,
    sample_rate: u32,
) -> Vec<f32> {
    let note_dur = frames as f32 / sample_rate as f32;
    (0..frames)
        .map(|i| gain * env.value_at(i as f32 / sample_rate as f32, note_dur)
            * sample_osc(waveform, freq, i, sample_rate))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 正弦：1kHz @ 8kHz SR 每 8 样本一周期，峰值≈1、过零位置正确。
    #[test]
    fn sine_frequency_and_phase() {
        let sr = 8_000_u32;
        let v: Vec<f32> = (0..16).map(|i| sample_osc(Waveform::Sine, 1_000.0, i, sr)).collect();
        assert!((v[2] - 1.0).abs() < 1e-6, "1kHz@8k 相位 ¼ 周期处应为峰值：v[2]={}", v[2]);
        assert!(v[0].abs() < 1e-6 && v[4].abs() < 1e-6);
        assert!((v[6] + 1.0).abs() < 1e-6);
    }

    /// 相位是索引的纯函数：独立取第 i 样本 == 顺序取（不依赖遍历历史）。
    #[test]
    fn phase_is_pure_function_of_index() {
        let (f, sr) = (880.0_f32, 44_100_u32);
        for i in [0_usize, 1, 100, 123_457, 999_983] {
            assert_eq!(sample_osc(Waveform::Sine, f, i, sr), sample_osc(Waveform::Sine, f, i, sr));
            // 直接重算同式（文档公式的独立对拍）。
            let p = (f as f64 * i as f64 / sr as f64).fract();
            assert!((sample_osc(Waveform::Sine, f, i, sr) as f64
                - (std::f64::consts::TAU * p).sin()).abs() < 1e-6);
        }
    }

    /// PolyBLEP 防混叠：以**同相位序列的加性带限参考**（各自波形的解析
    /// 傅里叶级数：saw `−Σ 2/(πk)·sin(2πkp)` 全谐波；square `Σ 4/(πk)·sin`
    /// 仅奇次谐波）为基准，比对**低频段 bins [0,500)（@44.1k ≈ 0–1.35 kHz）**
    /// 幅值误差。参考与被测共用同一离散相位序列与 Hann 窗——相位/泄漏项完全
    /// 抵消，误差差 = 纯混叠差：f0 的 k>harmonics 谐波折返落进低频形成非谐和
    /// 噪声，PolyBLEP 应将其压低约 10³ 倍（把修正符号写反会加倍不连续，本测
    /// 试即红）。Nyquist 附近抑制有限是该方法的已知过渡带折衷，不纳入断言。
    #[test]
    fn polyblep_matches_band_limited_reference() {
        let (sr, n) = (44_100_u32, 16_384_usize);
        let f0 = 200.0_f32; // 谐波 200k Hz；k > 110 超 Nyquist（折叠区）
        let harmonics = ((sr as f64 / 2.0) / f0 as f64) as usize; // 110
        let fft = crate::fft::Fft::new(n).unwrap();
        let hann = crate::fft::hann_window(n);
        // 与振荡器同一相位序列 p_i = fract(f0·i/sr)（f64 相位）。
        let phase = |i: usize| (f0 as f64 * i as f64 / sr as f64).fract();
        // 断言聚焦低频段 bins [0,500)：PolyBLEP 的主战场（可听混叠区）。
        let low = 500_usize;
        for wf in [Waveform::Saw, Waveform::Square] {
            // 各波形的解析级数参考（saw：全谐波 −2/(πk)；square：奇谐波 +4/(πk)）。
            let band_limited: Vec<f32> = (0..n)
                .map(|i| {
                    let p = phase(i);
                    let mut s = 0.0_f64;
                    for k in 1..=harmonics {
                        if wf == Waveform::Square && k % 2 == 0 {
                            continue;
                        }
                        let a = if wf == Waveform::Square {
                            4.0 / (std::f64::consts::PI * k as f64)
                        } else {
                            -2.0 / (std::f64::consts::PI * k as f64)
                        };
                        s += a * (std::f64::consts::TAU * k as f64 * p).sin();
                    }
                    (s as f32) * hann[i]
                })
                .collect();
            let reference = fft.magnitudes(&band_limited).unwrap();
            let spec_error = |x: Vec<f32>| -> f64 {
                let mag = fft.magnitudes(&x).unwrap();
                mag[..low]
                    .iter()
                    .zip(&reference[..low])
                    .map(|(&a, &r)| ((a - r) as f64) * ((a - r) as f64))
                    .sum::<f64>()
            };
            let naive = spec_error(
                (0..n)
                    .map(|i| {
                        let p = phase(i);
                        (match wf {
                            Waveform::Saw => 2.0 * p - 1.0,
                            _ => if p < 0.5 { 1.0 } else { -1.0 },
                        }) as f32 * hann[i]
                    })
                    .collect(),
            );
            let blep =
                spec_error((0..n).map(|i| sample_osc(wf, f0, i, sr) * hann[i]).collect());
            assert!(
                blep < naive / 100.0,
                "{wf:?} 低频段带限误差未达数量级差：blep={blep:e} naive={naive:e}"
            );
        }
    }

    /// ADSR 闭式解分段锚点：升/降/保持/释放/归零与连续性。
    #[test]
    fn adsr_piecewise_anchors() {
        let env = Adsr { attack: 0.1, decay: 0.2, sustain: 0.5, release: 0.4 };
        assert_eq!(env.value_at(0.0, 1.0), 0.0);
        assert!((env.value_at(0.05, 1.0) - 0.5).abs() < 1e-6);
        assert!((env.value_at(0.1, 1.0) - 1.0).abs() < 1e-6, "attack 顶点");
        assert!((env.value_at(0.2, 1.0) - 0.75).abs() < 1e-6, "decay 中点 1→0.5");
        assert!((env.value_at(0.5, 1.0) - 0.5).abs() < 1e-6, "sustain 保持");
        assert!((env.value_at(1.2, 1.0) - 0.25).abs() < 1e-6, "release 中点");
        assert_eq!(env.value_at(1.4, 1.0), 0.0, "release 结束归零");
        assert_eq!(env.value_at(0.5, 0.5), 0.5, "恰好音符结束仍在 sustain");
        // 退化参数：attack=0 直接进 decay 后段；release=0 结束即 0。
        let hard = Adsr { attack: 0.0, decay: 0.0, sustain: 0.8, release: 0.0 };
        assert!((hard.value_at(0.3, 0.5) - 0.8).abs() < 1e-6);
        assert_eq!(hard.value_at(0.5, 0.5), 0.0);
    }

    /// render_note 长度/幅度界与确定性复算。
    #[test]
    fn render_note_shape_and_determinism() {
        let env = Adsr { attack: 0.01, decay: 0.05, sustain: 0.6, release: 0.1 };
        let a = render_note(Waveform::Saw, 440.0, &env, 0.5, 4_410, 44_100);
        let b = render_note(Waveform::Saw, 440.0, &env, 0.5, 4_410, 44_100);
        assert_eq!(a, b, "同参数逐位一致");
        assert_eq!(a.len(), 4_410);
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 0.5 + 1e-6), "幅度不超过 gain");
    }

    /// 群作用（圆周群旋转流）：相位推进算子 Φ: p ↦ fract(p + f/sr) 满足
    /// Φᵇ∘Φᵃ = Φᵃ⁺ᵇ——从索引 i 出发"先推进 a 步再推进 b 步"与一步跳到
    /// i+a+b 在 S¹ 圆周距离内一致。这是 seek 等式（D-28：跳第 N 帧 ≡
    /// 顺序推进）在音频样本索引流上的实例：相位是索引的纯函数，复合只受
    /// 定点舍入约束。
    ///
    /// 误差预算（容差 1e-11 的依据，非拍脑袋）：直接侧一次除法舍入
    /// ≤ 相位值·2⁻⁵³（相位值 ≤ ~1e4 → ≤1.1e-12）；步进侧每段一次
    /// k·step 乘加舍入 ≤ ulp/2 ≈ 1.1e-16，累加器重放每步一次舍入
    /// ≤ 2⁻⁵³（fract 的 floor 相减为 Sterbenz 精确、不引入误差），
    /// 最大用例累加 3048 步 → ≤3.4e-13；合计 ≤5e-13，容差取 20× 余量。
    #[test]
    fn phase_advance_composes_as_group_action() {
        let (f, sr) = (441.0_f32, 44_100_u32);
        let step = f as f64 / sr as f64; // Φ 的单步位移 = fl(0.01)
        // 圆周距离：fract 边界（≈0/≈1 两侧）不放大误差。
        let circ = |a: f64, b: f64| {
            let d = (a - b).abs();
            d.min(1.0 - d)
        };
        for &(i, a, b) in &[
            (0_usize, 1, 1),
            (0, 100, 100),
            (7, 333, 77),
            (12_345, 1_000, 2_048),
        ] {
            let direct = (f as f64 * (i + a + b) as f64 / sr as f64).fract();
            // 算子复合：k 步推进 = 一次 k·step 位移；旋转群可交换，两种顺序都验。
            for (x, y) in [(a, b), (b, a)] {
                let mut p = (f as f64 * i as f64 / sr as f64).fract();
                p = (p + x as f64 * step).fract();
                p = (p + y as f64 * step).fract();
                assert!(
                    circ(direct, p) < 1e-11,
                    "Φ^{x}∘Φ^{y} ≠ Φ^{}：direct={direct:e} stepped={p:e}",
                    x + y
                );
            }
            // 累加器重放（seek 等式的"重放侧"）：从 p_i 起单步循环 a+b 步
            // == 直接求值——状态机振荡器与纯函数振荡器给出同一轨道。
            let mut acc = (f as f64 * i as f64 / sr as f64).fract();
            for _ in 0..(a + b) {
                acc = (acc + step).fract();
            }
            assert!(
                circ(direct, acc) < 1e-11,
                "累加器重放 ≠ 直接求值：direct={direct:e} replayed={acc:e}"
            );
        }
    }

    /// 样本级群作用：正弦在"先推进到 i、再步进 n 步"所得相位处的取值 ==
    /// 直接在 i+n 求值。定点预算：相位差 ≤ ~2e-11（见
    /// [`phase_advance_composes_as_group_action`]），正弦对相位的 Lipschitz
    /// 常数为 TAU → 样本差 ≤ 1.3e-10，远小于 f32 出入口舍入（~6e-8）；
    /// 容差 1e-6 与既有 `phase_is_pure_function_of_index` 一致。
    #[test]
    fn sine_samples_compose_within_fixed_point_tolerance() {
        let (f, sr) = (440.0_f32, 44_100_u32);
        let step = f as f64 / sr as f64;
        for &(i, n) in &[(0_usize, 1), (0, 4_410), (7, 999), (55_555, 123_456)] {
            let direct = sample_osc(Waveform::Sine, f, i + n, sr) as f64;
            // "先算 i 步" = fract(f·i/sr)（索引纯函数）；"再算 n 步" = 单步循环。
            let mut q = (f as f64 * i as f64 / sr as f64).fract();
            for _ in 0..n {
                q = (q + step).fract();
            }
            // 同一正弦函数（det::sin_f64），只换相位参数——隔离出复合性质。
            let stepped = crate::det::sin_f64(std::f64::consts::TAU * q) as f32 as f64;
            assert!(
                (direct - stepped).abs() < 1e-6,
                "样本级复合失败 @ i={i} n={n}：direct={direct:e} stepped={stepped:e}"
            );
        }
    }

    /// 整周期回归：f·n = sr 时 Φⁿ = 恒等（相位前进整数圈后回到原位）。
    /// 441 Hz @ 44.1k：n = 100 样本恰好一圈，f·(i+n)/sr = f·i/sr + 1，
    /// fract 后同相位——"一步跳整圈"与"原位"在定点内一致。
    /// 预算：相位值 ≤ (i+n)/100 ≈ 1e4 → 两侧除法舍入 ≤ 2.2e-12；波形最大
    /// 斜率 ~2/dt = 200（saw @ 441Hz，BLEP 窗缘）→ 样本差 ≤ 4.4e-10 << 1e-6。
    #[test]
    fn integer_period_returns_to_same_sample() {
        let (f, sr) = (441.0_f32, 44_100_u32);
        let n = 100_usize; // f·n/sr = 1.0 整圈
        for i in [0_usize, 1, 37, 4_410, 999_999] {
            for wf in [Waveform::Sine, Waveform::Square, Waveform::Saw, Waveform::Triangle] {
                let a = sample_osc(wf, f, i, sr);
                let b = sample_osc(wf, f, i + n, sr);
                assert!(
                    (a - b).abs() < 1e-6,
                    "{wf:?} @ i={i}：整圈回归 a={a:e} b={b:e}"
                );
            }
        }
    }

    /// 索引流乱序访问 == 顺序扫描（逐位）：纯函数相位的直接推论——第 i
    /// 样本不依赖遍历历史，随机 seek 与顺序推进给出同一轨道。逆序、伪随机
    /// 置换、跨段跳进跳出三条乱序路径都逐位复现顺序流（若有人给振荡器加
    /// 隐藏累加状态，本测试即红）。
    #[test]
    fn random_index_access_matches_sequential_scan_bitwise() {
        let (f, sr) = (440.0_f32, 44_100_u32);
        let n = 20_000_usize;
        // 确定性 LCG（零依赖；不用系统随机源）。
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for wf in [Waveform::Sine, Waveform::Square, Waveform::Saw, Waveform::Triangle] {
            let seq: Vec<f32> = (0..n).map(|i| sample_osc(wf, f, i, sr)).collect();
            // 路径 1：逆序扫描。
            for i in (0..n).rev() {
                assert_eq!(sample_osc(wf, f, i, sr).to_bits(), seq[i].to_bits());
            }
            // 路径 2：伪随机置换（Fisher–Yates，LCG 驱动）。
            let mut order: Vec<usize> = (0..n).collect();
            for k in (1..n).rev() {
                let j = (next() % (k as u64 + 1)) as usize;
                order.swap(k, j);
            }
            for &i in &order {
                assert_eq!(sample_osc(wf, f, i, sr).to_bits(), seq[i].to_bits(), "{wf:?} @ {i}");
            }
            // 路径 3：跨段跳进跳出——先探远处（应命中顺序流该位），再回到顺序流。
            let mut i = 0;
            let mut walked = Vec::with_capacity(n);
            while i < n {
                walked.push(sample_osc(wf, f, i, sr));
                let jump = 97 + (next() % 211) as usize; // 97..=307
                if let Some(&ahead) = seq.get(i + jump) {
                    assert_eq!(
                        sample_osc(wf, f, i + jump, sr).to_bits(),
                        ahead.to_bits(),
                        "{wf:?} 跳步探测 @ {i}+{jump}"
                    );
                }
                i += 1;
            }
            assert_eq!(walked, seq, "{wf:?}：跳探后的顺序流应逐位复现");
        }
    }
}
