//! 采样回放核：定点相位累加器 + Catmull-Rom 四点插值 +
//! 内嵌程序化音色库。
//!
//! # 定点相位累加器（超越点）
//!
//! 读取位置用 **Q16.16 整数定点**累加音高比增量——浮点相位长序列累加会漂移
//! （业界采样器的隐性不确定源），定点累加保证**逐字节复现**。音高比 ⇒ 定点增量
//! 由 [`sfz::semitone_ratio`]（禁 pow，D-29）算出后一次转定点：
//!
//! ```text
//! step = round(ratio · 65536)      （每样本相位增量，整数）
//! pos  += step                     （整数累加，零漂移）
//! ```
//!
//! # Catmull-Rom 四点三次插值
//!
//! 在整数格点 `i` 处取 `p[i−1..i+2]` 四点，`t` = 小数相位：
//!
//! ```text
//! h(t) = 0.5·[ 2p₁ + (−p₀+p₂)t + (2p₀−5p₁+4p₂−p₃)t² + (−p₀+3p₁−3p₂+p₃)t³ ]
//! ```
//!
//! 等价三次 Hermite，比线性少可闻混叠。
//!
//! # 内嵌音色库（程序化生成，见数据资产登记）
//!
//! **如实标注**：本包**未**内嵌任何外部 CC0 音色样本文件——`prism-audio` 无
//! 文件系统 I/O，且当前版本未内嵌可入仓的 CC0 采样资产。为不失功能，内嵌库
//! 由**程序化合成**生成（正弦谐波叠加模拟钢琴/弦乐/鼓），**非**真实采样录音。
//! 登记于数据资产清单（来源=引擎自研合成、许可=随引擎许可、合规动作=代码注释
//! 标注为合成音色）。外部 SFZ（`sample=` 指向真实样本）为后续正式路径。

use crate::buffer::PcmBuffer;
use crate::sfz::Region;
use crate::{AudioError, AudioResult};

/// 定点相位小数位数（Q16.16 的 16 位小数）。
const FRAC_BITS: u32 = 16;
/// 定点一（1.0）的整数表示。
const ONE: i64 = 1 << FRAC_BITS;

/// 采样器内核：内嵌音色 + 定点相位 + Catmull-Rom。
#[derive(Debug, Clone)]
pub struct Sampler {
    /// 音色库（键名 → 单周期/整段单声道样本）。
    library: TimbreLibrary,
}

impl Default for Sampler {
    fn default() -> Self {
        Self { library: TimbreLibrary::procedural() }
    }
}

impl Sampler {
    /// 用默认程序化音色库构造。
    pub fn new() -> Self {
        Self::default()
    }

    /// 取音色库中某键样本（不存在返回 `None`）。
    pub fn sample_asset(&self, name: &str) -> Option<&PcmBuffer> {
        self.library.get(name)
    }

    /// 按 SFZ region 渲染一个音符：`frames` 输出样本。
    ///
    /// 音高由 [`Region::playback_ratio`] 决定（定点累加），循环区间若设则回绕，
    /// 包络用 [`Adsr`]（复用合成器代码）。返回幅度 = 库样本幅度 × 包络 ×
    /// `10^(volume_db/20)`。
    pub fn render_region(
        &self,
        region: &Region,
        key: u8,
        frames: usize,
        sample_rate: u32,
    ) -> AudioResult<Vec<f32>> {
        let asset = self.library.get(&region.sample).ok_or_else(|| {
            AudioError::new(
                "AU1006",
                format!("Sampler::render_region(sample={})", region.sample),
                "音色库中无此键名（内嵌库为程序化合成音色）",
                "改用内嵌键名（piano/violin/drum）或先加载外部 SFZ 样本",
            )
        })?;
        let src = asset.as_slice();
        if src.len() < 4 {
            return Err(AudioError::new(
                "AU1006",
                format!("Sampler::render_region(sample={})", region.sample),
                "样本短于 4 点，无法做 Catmull-Rom 四点插值",
                "样本需 ≥4 个样本点",
            ));
        }
        let ratio = region.playback_ratio(key);
        // 定点增量（每输出样本的源相位推进）。
        let step = (ratio as f64 * ONE as f64).round() as i64;
        let loop_region = match (region.loop_start, region.loop_end) {
            (Some(s), Some(e)) if e > s && (e as usize) < src.len() => Some((s as i64 * ONE, e as i64 * ONE)),
            _ => None,
        };
        let total_secs = frames as f32 / sample_rate as f32;
        let gain = crate::det::exp_f64(region.volume_db as f64 * std::f64::consts::LN_10 / 20.0) as f32;

        let mut pos: i64 = 0;
        let mut out = Vec::with_capacity(frames);
        let max_pos = (src.len() as i64 - 2) * ONE; // 保证 i+2 可取
        for i in 0..frames {
            // 循环回绕（格点对齐）。
            if let Some((ls, le)) = loop_region {
                if pos >= le {
                    pos = ls + (pos - le) % (le - ls).max(ONE);
                }
            } else if pos >= max_pos {
                break; // 一遍到底，尾静音
            }
            let tap = catmull_rom(src, pos);
            let env = region.amp_envelope.value_at(i as f32 / sample_rate as f32, total_secs);
            out.push(tap * env * gain);
            pos += step;
        }
        out.resize(frames, 0.0);
        Ok(out)
    }
}

/// Catmull-Rom 四点插值：`pos` 为 Q16.16 定点源位置。
pub fn catmull_rom(src: &[f32], pos: i64) -> f32 {
    let i = (pos >> FRAC_BITS) as usize; // 整数格点
    let t = (pos & (ONE - 1)) as f32 / ONE as f32; // 小数相位 ∈ [0,1)
    // 端点钳制（取可用邻居）。
    let p0 = src[i.saturating_sub(1)];
    let p1 = src[i];
    let p2 = src[(i + 1).min(src.len() - 1)];
    let p3 = src[(i + 2).min(src.len() - 1)];
    // Catmull-Rom 基函数（t 多项式）。
    let a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let a2 = -0.5 * p0 + 0.5 * p2;
    let a3 = p1;
    ((a0 * t + a1) * t + a2) * t + a3
}

/// 内嵌程序化音色库（非真实采样；见模块头如实标注）。
#[derive(Debug, Clone)]
pub struct TimbreLibrary {
    entries: Vec<(String, PcmBuffer)>,
}

impl TimbreLibrary {
    /// 生成程序化音色库：piano / violin / drum（各 1 秒 @44.1k 单声道）。
    pub fn procedural() -> Self {
        let sr = 44_100_u32;
        let n = sr as usize;
        let piano = synth_tone(n, sr, 261.63, &[1.0, 0.5, 0.25, 0.12, 0.06], 1.2);
        let violin = synth_tone(n, sr, 440.0, &[1.0, 0.7, 0.45, 0.3, 0.2, 0.12], 2.5);
        let drum = synth_drum(n, sr);
        let entries = vec![
            ("piano".to_string(), PcmBuffer::from_mono(sr, piano).expect("单声道合法")),
            ("violin".to_string(), PcmBuffer::from_mono(sr, violin).expect("单声道合法")),
            ("drum".to_string(), PcmBuffer::from_mono(sr, drum).expect("单声道合法")),
        ];
        Self { entries }
    }

    /// 按键名取样本。
    pub fn get(&self, name: &str) -> Option<&PcmBuffer> {
        self.entries.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// 库中所有键名（固定序）。
    pub fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }
}

/// 合成谐波叠加音（`amps[k]` = 第 k 次谐波幅度），带指数衰减整体包络。
fn synth_tone(n: usize, sr: u32, f0: f32, amps: &[f32], decay_secs: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let env = crate::det::exp_f64(-t / decay_secs as f64 * 3.0);
            let mut s = 0.0_f64;
            for (k, &a) in amps.iter().enumerate() {
                let h = (k + 1) as f64;
                s += a as f64 * crate::det::sin_f64(2.0 * std::f64::consts::PI * f0 as f64 * h * t);
            }
            let norm = amps.iter().sum::<f32>().max(1e-6) as f64;
            (s / norm * env) as f32
        })
        .collect()
}

/// 合成鼓音（正弦基频快速衰减 + 少量噪声感谐波）。
fn synth_drum(n: usize, sr: u32) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        let t = i as f64 / sr as f64;
        // 基频随时间下滑（80→40 Hz）的确定性线性扫频（无 pow）。
        let f = 80.0 - 40.0 * (t / 0.3).min(1.0);
        let body = crate::det::exp_f64(-t * 18.0) * crate::det::sin_f64(2.0 * std::f64::consts::PI * f * t);
        // 确定性「噪声」：PCG 哈希整数派生，快速衰减。
        let noise = (prism_noise::pcg::hash2_f32(1234, i as u64, 7) * 2.0 - 1.0) as f64
            * crate::det::exp_f64(-t * 60.0);
        *o = (body * 0.8 + noise * 0.2) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::Adsr;

    /// Catmull-Rom：恰在格点上应等于样本原值（插值性质）。
    #[test]
    fn catmull_rom_interpolates_grid_points() {
        let src = [0.0_f32, 1.0, 0.0, -1.0, 0.5, 0.2];
        for i in 0..src.len() {
            let v = catmull_rom(&src, i as i64 * ONE);
            assert!((v - src[i]).abs() < 1e-5, "格点 {i} 应精确复现：{v} vs {}", src[i]);
        }
    }

    /// 线性斜坡上 Catmull-Rom 应精确复现线性（多项式精度）。
    #[test]
    fn catmull_rom_reproduces_linearity() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        for k in 1..6 {
            for &t in &[0.0_f32, 0.25, 0.5, 0.75] {
                let pos = (k as i64 * ONE) + (t * ONE as f32) as i64;
                let v = catmull_rom(&src, pos);
                let expect = k as f32 + t;
                assert!((v - expect).abs() < 1e-3, "线性应复现：{v} vs {expect}");
            }
        }
    }

    /// 音高准确：region 根音播放时输出基频 = 样本基频（定点相位 + 插值）。
    #[test]
    fn sampler_pitch_accuracy_at_center() {
        let sampler = Sampler::new();
        let region = Region { sample: "piano".into(), pitch_keycenter: 60, ..Default::default() };
        let out = sampler.render_region(&region, 60, 44_100, 44_100).unwrap();
        let f = estimate_freq_zerocross(&out, 44_100);
        assert!((f - 261.63).abs() / 261.63 < 0.03, "根音播放应复现 261.63 Hz，实际 {f}");
    }

    /// 上八度：key=72 相对 center=60 播放速率 ×2 → 输出频率翻倍。
    #[test]
    fn sampler_octave_up_doubles_pitch() {
        let sampler = Sampler::new();
        let base = Region { sample: "violin".into(), pitch_keycenter: 60, ..Default::default() };
        // 取稳态窗（避开尾零）：ratio=1 时前 15000 样本均非静音；ratio=2 时该窗亦非空。
        let f_base = estimate_freq_zerocross(
            &sampler.render_region(&base, 60, 44_100, 44_100).unwrap()[..15_000],
            44_100,
        );
        let f_up = estimate_freq_zerocross(
            &sampler.render_region(&base, 72, 44_100, 44_100).unwrap()[..15_000],
            44_100,
        );
        assert!((f_up / f_base - 2.0).abs() < 0.05, "上八度应 ×2：{f_base} → {f_up}");
    }

    /// 包络：attack 期间输出从 0 渐升（首样本近零）。
    #[test]
    fn sampler_envelope_attack_ramps() {
        let sampler = Sampler::new();
        let region = Region {
            sample: "piano".into(),
            pitch_keycenter: 60,
            amp_envelope: Adsr { attack: 0.1, decay: 0.0, sustain: 1.0, release: 0.0 },
            ..Default::default()
        };
        let out = sampler.render_region(&region, 60, 4_410, 44_100).unwrap();
        assert!(out[0].abs() < out[2_205].abs(), "attack 期间应递增");
        assert!(out[0].abs() < 0.05, "首样本近零");
    }

    /// 未知音色键 → 三段式报错。
    #[test]
    fn sampler_unknown_sample_errors() {
        let sampler = Sampler::new();
        let region = Region { sample: "nope".into(), ..Default::default() };
        assert_eq!(sampler.render_region(&region, 60, 100, 44_100).unwrap_err().code, "AU1006");
    }

    /// 定点相位零漂移：同 region 重复渲染逐位一致。
    #[test]
    fn sampler_bitwise_deterministic() {
        let sampler = Sampler::new();
        let region = Region { sample: "piano".into(), pitch_keycenter: 60, ..Default::default() };
        let a = sampler.render_region(&region, 64, 8_000, 44_100).unwrap();
        for _ in 0..50 {
            assert_eq!(sampler.render_region(&region, 64, 8_000, 44_100).unwrap(), a);
        }
    }

    /// 程序化库含三键。
    #[test]
    fn procedural_library_has_three_timbres() {
        let lib = TimbreLibrary::procedural();
        let keys = lib.keys();
        assert!(keys.contains(&"piano") && keys.contains(&"violin") && keys.contains(&"drum"));
    }

    /// 过零率基频估计（测试工具）。
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
        zc as f32 * sr as f32 / x.len() as f32
    }
}
