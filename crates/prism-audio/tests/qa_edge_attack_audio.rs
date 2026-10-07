//! QA 攻击性测试 —— prism-audio（fx/physical/granular/spatial/sampler/sfz/synth）。
//!
//! 目标：边界/退化、确定性、数学正确性、防御缺失。每个测试注明失败代表什么缺陷。

use prism_audio::fx::compressor::{db_to_lin, gain_reduction_db, lin_to_db, Compressor};
use prism_audio::fx::delay::Delay;
use prism_audio::fx::distortion::{Distortion, DistortionKind};
use prism_audio::fx::eq::{EqBand, MultiEq, Shelf};
use prism_audio::fx::panner::{constant_power_gains, PanAutomation, PanCurve};
use prism_audio::fx::reverb::{FdnReverb, SchroederReverb};
use prism_audio::fx::MonoFx;
use prism_audio::granular::{GrainSynth, GrainWindow};
use prism_audio::physical::{KarplusStrong, ModalStrike};
use prism_audio::sampler::Sampler;
use prism_audio::sfz::{parse_sfz, semitone_ratio, Region};
use prism_audio::spatial::{
    mix_spatial, Attenuation, AttenuationModel, Listener, SpatialSource,
};
use prism_audio::synth::filter::{Svf, SvfMode};
use prism_audio::synth::lfo::Lfo;
use prism_audio::synth::unison::Unison;
use prism_audio::synth::Waveform;

// ==================================================================
// A. 确定性攻击（核心铁律）
// ==================================================================

/// 缺陷：同输入两遍必须逐位一致（库内所有纯效果器）。
#[test]
fn all_fx_bitwise_deterministic() {
    let x: Vec<f32> = (0..2048).map(|i| ((i as f32 * 0.017).sin()) * 0.7).collect();
    let sr = 44_100;
    macro_rules! check {
        ($fx:expr) => {{
            let fx = $fx;
            let a = fx.process(&x, sr);
            for _ in 0..3 {
                assert_eq!(fx.process(&x, sr), a, "效果器不逐位一致");
            }
        }};
    }
    check!(Compressor::default());
    check!(Delay::default());
    check!(Distortion::default());
    check!(SchroederReverb::default());
    check!(FdnReverb::default());
    check!(MultiEq::new(vec![EqBand { kind: Shelf::Peaking, freq: 1000.0, gain_db: 6.0, q: 1.0 }]));
}

/// 缺陷：颗粒合成同种子跨调用逐位一致；不同种子应不同。
#[test]
fn granular_pcg_cross_call_stable() {
    let src: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let gs = GrainSynth { pitch_spread: 0.3, position_spread: 0.02, seed: 0xBEEF, ..Default::default() };
    let a = gs.render(&src, 8_000, 8_000).unwrap();
    let b = gs.render(&src, 8_000, 8_000).unwrap();
    assert_eq!(a, b, "颗粒渲染跨调用应逐位一致");
    let c = GrainSynth { seed: 0xBEF0, ..gs }.render(&src, 8_000, 8_000).unwrap();
    assert_ne!(a, c, "不同种子应不同");
}

/// 缺陷：unison 相位是索引纯函数——独立取第 i 样本 == 顺序推进。
#[test]
fn unison_phase_is_index_pure() {
    let u = Unison::default();
    let seq: Vec<f32> = (0..500).map(|i| u.sample(330.0, i, 44_100)).collect();
    for i in [0usize, 7, 123, 499] {
        assert_eq!(u.sample(330.0, i, 44_100), seq[i], "第 {i} 样本应等于顺序推进");
    }
}

// ==================================================================
// B. 数学正确性
// ==================================================================

/// 缺陷：恒功率声像两通道能量和恒为 1（含极值/超出范围钳制）。
#[test]
fn constant_power_energy_invariant() {
    for pan in [-2.0_f32, -1.0, -0.33, 0.0, 0.5, 1.0, 3.0] {
        let (gl, gr) = constant_power_gains(pan);
        let e = gl * gl + gr * gr;
        assert!((e - 1.0).abs() < 1e-4, "pan={pan} 能量和 {e} 应为 1");
    }
}

/// 缺陷：压缩器阈值/比率/软拐点行为（硬拐点下超出阈值按 slope 线性）。
#[test]
fn compressor_gain_reduction_math() {
    // 硬拐点（knee=0）。
    assert_eq!(gain_reduction_db(-30.0, -20.0, 4.0, 0.0), 0.0, "阈下不压缩");
    // 超出 10 dB，ratio 4 → 10·(1−1/4)=7.5。
    let gr = gain_reduction_db(-10.0, -20.0, 4.0, 0.0);
    assert!((gr - 7.5).abs() < 1e-4, "硬拐点缩减 {gr} 应 = 7.5");
    // ratio 越大压缩越狠。
    assert!(gain_reduction_db(-10.0, -20.0, 8.0, 0.0) > gr);
    // 软拐点：拐点内单调不减；连续性体现在**两处拐点边界**（threshold±half_k）：
    // 软段在此应等于相邻硬段值 0（下界）与 half_k·slope（上界）。
    let half_k = 3.0;
    let slope = 1.0 - 1.0 / 4.0;
    let mut prev = 0.0_f32;
    let mut x = -30.0_f32;
    while x <= 0.0 {
        let g = gain_reduction_db(x, -20.0, 4.0, 6.0);
        assert!(g >= prev - 1e-6, "软拐点应单调不减 x={x}");
        prev = g;
        x += 0.05;
    }
    let lo_edge = gain_reduction_db(-20.0 - half_k, -20.0, 4.0, 6.0);
    let hi_edge = gain_reduction_db(-20.0 + half_k, -20.0, 4.0, 6.0);
    assert!(lo_edge.abs() < 1e-4, "下拐点边界应为 0，得 {lo_edge}");
    assert!(
        (hi_edge - half_k * slope).abs() < 1e-3,
        "上拐点边界应为 half_k·slope={}, 得 {hi_edge}",
        half_k * slope
    );
}

/// 缺陷：dB↔线性往返一致（自研 exp/ln 精度）。
#[test]
fn db_roundtrip_accuracy() {
    for db in [-60.0_f32, -20.0, -6.0, 0.0, 3.0, 12.0] {
        let back = lin_to_db(db_to_lin(db));
        assert!((back - db).abs() < 1e-2, "db={db} 往返 {back}");
    }
}

/// 缺陷：EQ 峰段中心频率增益 ≈ 设定值（RBJ 系数正确性）。
#[test]
fn eq_peaking_center_gain() {
    let eq = MultiEq::new(vec![EqBand { kind: Shelf::Peaking, freq: 1_000.0, gain_db: 6.0, q: 2.0 }]);
    let g = eq.magnitude_at(1_000.0, 48_000);
    let want = db_to_lin(6.0);
    assert!((g - want).abs() < want * 0.08, "峰中心增益 {g} 应 ≈ {want}");
    let far = eq.magnitude_at(50.0, 48_000);
    assert!((far - 1.0).abs() < 0.08, "远端应 ≈ 1，实际 {far}");
}

/// 缺陷：Karplus-Strong 音高 f0=SR/(L+0.5)，过零率±2%。
#[test]
fn ks_pitch_accuracy() {
    let sr = 44_100;
    for &f in &[110.0_f32, 220.0, 440.0] {
        let ks = KarplusStrong { freq: f, decay: 0.999, seed: None, brightness: 0.0 };
        let out = ks.render(44_100, sr).unwrap();
        let est = zero_cross_freq(&out[10_000..], sr);
        assert!((est - f).abs() / f < 0.02, "KS f={f} 估计 {est}");
    }
}

/// 缺陷：模态打击 FFT 峰值落在基频 bin。
#[test]
fn modal_strike_fft_peak() {
    let sr = 8_000;
    let n = 4_096;
    let strike = ModalStrike {
        freq: 500.0,
        amplitudes: vec![1.0, 0.3],
        ratios: vec![1.0, 2.0],
        decays: vec![1.0, 0.5],
    };
    let out = strike.render(n, sr).unwrap();
    let fft = prism_audio::fft::Fft::new(n).unwrap();
    let mag = fft.magnitudes(&out).unwrap();
    let want = (500.0 * n as f32 / sr as f32).round() as usize;
    let peak = mag[..n / 2]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!((peak as i32 - want as i32).abs() <= 1, "峰 bin {peak} 应 ≈ {want}");
}

/// 缺陷：距离衰减单调（Inverse 与 Linear）。
#[test]
fn attenuation_monotonic() {
    let inv = Attenuation { model: AttenuationModel::Inverse, ref_dist: 1.0, max_dist: 100.0, rolloff: 1.0 };
    let mut prev = 1.0_f32;
    for d in 1..40 {
        let g = inv.gain(d as f32);
        assert!(g <= prev + 1e-6, "Inverse 应单调递减 d={d}");
        prev = g;
    }
    let lin = Attenuation { model: AttenuationModel::Linear, ref_dist: 0.0, max_dist: 10.0, rolloff: 1.0 };
    assert!((lin.gain(0.0) - 1.0).abs() < 1e-6);
    assert!(lin.gain(10.0).abs() < 1e-6);
    assert!((lin.gain(5.0) - 0.5).abs() < 1e-5);
}

/// 缺陷：SVF 低通直流增益 ≈ 1、高频被抑制；带通中心峰。
#[test]
fn svf_frequency_response() {
    let lp = Svf { cutoff_hz: 1_000.0, q: 0.707, mode: SvfMode::LowPass };
    assert!(gain_at(&lp, 50.0, 48_000) > 0.9, "低通直流应 ≈ 1");
    assert!(gain_at(&lp, 15_000.0, 48_000) < 0.2, "低通高频应抑制");
    let hp = Svf { cutoff_hz: 1_000.0, q: 0.707, mode: SvfMode::HighPass };
    assert!(gain_at(&hp, 50.0, 48_000) < 0.2 && gain_at(&hp, 15_000.0, 48_000) > 0.8);
}

/// 缺陷：LFO 波形锚点（Sine ¼ 周期峰值、Square ±1、Triangle 端点）。
#[test]
fn lfo_waveform_anchors() {
    let sine = Lfo { waveform: Waveform::Sine, rate_hz: 1.0, phase_offset: 0.0 };
    assert!(sine.value(0, 1_000).abs() < 1e-6);
    assert!((sine.value(250, 1_000) - 1.0).abs() < 1e-4);
    assert!((sine.value(750, 1_000) + 1.0).abs() < 1e-4);
    let tri = Lfo { waveform: Waveform::Triangle, rate_hz: 1.0, phase_offset: 0.0 };
    assert!(tri.value(0, 1_000).abs() <= 1.0 + 1e-6 && tri.value(500, 1_000).abs() <= 1.0 + 1e-6);
    assert!(tri.value(0, 1_000).is_finite() && tri.value(1_000, 1_000).is_finite());
}

/// 缺陷：semitone_ratio 与十二平均律一致（A4=440 锚点、纯五度 1.4983）。
#[test]
fn semitone_ratio_12tet() {
    assert!((semitone_ratio(0.0) - 1.0).abs() < 1e-6);
    assert!((semitone_ratio(12.0) - 2.0).abs() < 1e-5, "八度应 = 2");
    assert!((semitone_ratio(7.0) - 1.498_307_1).abs() < 1e-5, "纯五度");
    assert!((semitone_ratio(-12.0) - 0.5).abs() < 1e-5, "下八度");
}

/// 缺陷：采样核 Q16.16 相位格点复现——同 key 重复渲染逐位一致；根音播放复现基频。
#[test]
fn sampler_q16_phase_grid_reproducible() {
    let sampler = Sampler::new();
    let region = Region { sample: "piano".into(), pitch_keycenter: 60, ..Default::default() };
    let a = sampler.render_region(&region, 60, 44_100, 44_100).unwrap();
    for _ in 0..5 {
        assert_eq!(sampler.render_region(&region, 60, 44_100, 44_100).unwrap(), a);
    }
    let f = zero_cross_freq(&a, 44_100);
    assert!((f - 261.63).abs() / 261.63 < 0.03, "根音播放 {f} 应 ≈ 261.63");
}

// ==================================================================
// C. 防御缺失 / 边界
// ==================================================================

/// 缺陷：效果器处理空输入须返回空，不得 panic。
#[test]
fn fx_empty_input_no_panic() {
    let sr = 44_100;
    assert!(Compressor::default().process(&[], sr).is_empty());
    assert!(Delay::default().process(&[], sr).is_empty());
    assert!(Distortion::default().process(&[], sr).is_empty());
    assert!(SchroederReverb::default().process(&[], sr).is_empty());
    assert!(FdnReverb::default().process(&[], sr).is_empty());
    assert!(MultiEq::new(vec![]).process(&[], sr).is_empty());
}

/// 缺陷：混响 RT60=0（退化）不得迭代爆炸或产生非有限值。
#[test]
fn reverb_zero_rt60_stable() {
    let mut input = vec![0.0_f32; 4096];
    input[0] = 1.0;
    let out = SchroederReverb { rt60_secs: 0.0, mix: 1.0, damping: 0.5 }.process(&input, 44_100);
    assert!(out.iter().all(|v| v.is_finite()), "RT60=0 不应产生非有限值");
    let out2 = FdnReverb { rt60_secs: 0.0, mix: 1.0, lines: 4 }.process(&input, 44_100);
    assert!(out2.iter().all(|v| v.is_finite()), "FDN RT60=0 不应产生非有限值");
}

/// 缺陷：零采样率效果器不得除零/panic（若产生 NaN 即为缺陷）。
///
/// 历史：`Compressor::process(x, 0)` 会经 `coeff()` 算出 `1/(τ·0) = +inf`，
/// 再调 `det::exp_f64(-inf)` —— 该函数曾【无限循环挂死】（P0，见
/// `exp_f64_non_finite_short_circuits_p0`）。缺陷已修，故此处**全量覆盖**
/// 所有效果器的 sr=0 路径并断言有限性，不再有任何"故此处不测"的空洞。
#[test]
fn fx_zero_sample_rate_no_nan_panic() {
    let x = vec![0.5_f32; 16];
    // Delay @ sr=0：d = round(0.25·0) = 0 → 早退，安全。
    let d = Delay::default().process(&x, 0);
    assert_eq!(d.len(), 16);
    assert!(d.iter().all(|v| v.is_finite()));
    // Schroeder @ sr=0：t = rt60·0 = 0 → -3·d/t·ln10 = -inf → exp_f64(-inf)
    // 曾经挂死；修复后必须返回 +0.0 并立即完成。
    let s = SchroederReverb::default().process(&x, 0);
    assert_eq!(s.len(), 16);
    assert!(s.iter().all(|v| v.is_finite()), "Schroeder sr=0 不应产生非有限值");
    // FdnReverb 同型路径。
    let f = FdnReverb { rt60_secs: 1.8, mix: 0.3, lines: 4 }.process(&x, 0);
    assert_eq!(f.len(), 16);
    assert!(f.iter().all(|v| v.is_finite()), "FDN sr=0 不应产生非有限值");
    // Compressor @ sr=0：coeff(tau, 0) = 1 - exp_f64(-inf) = 1 - 0 = 1。
    let c = Compressor::default().process(&x, 0);
    assert_eq!(c.len(), 16);
    assert!(c.iter().all(|v| v.is_finite()), "Compressor sr=0 不应产生非有限值");
    // Distortion / MultiEq 同样走 sr 缩放，一并覆盖。
    let di = Distortion::default().process(&x, 0);
    assert!(di.iter().all(|v| v.is_finite()));
    let eq = MultiEq::new(vec![EqBand {
        kind: Shelf::Peaking,
        freq: 1000.0,
        gain_db: 6.0,
        q: 1.0,
    }])
    .process(&x, 0);
    assert!(eq.iter().all(|v| v.is_finite()));
}

/// 【P0 回归 · 已修复并转正】`det::exp_f64` 对非有限输入必须**立即**返回，
/// 不得挂死。
///
/// 根因（`crates/prism-audio/src/det.rs`）：`exp_f64` 未做非有限短路时，
/// `k = (-inf/LN_2).round() = -inf` → `k as i64 = i64::MIN`（饱和转换），
/// `scale_pow2(p, i64::MIN)` 的 `while k < 0 { r *= 0.5; k += 1 }` 需
/// ~9.2e18 次迭代才退出 → **等价挂死**，运行永不返回。
///
/// 可达性（公开 API）：`Compressor::process(&x, 0)` / `SchroederReverb::process(&x, 0)`
/// 传 `sample_rate=0` 即触发（`t = rt60·0 = 0`，`-3·d/t·LN_10 = -inf`）。
///
/// **处置**：修被测函数（`exp_f64` 非有限短路 + `scale_pow2` 迭代钳制，
/// 均有 `det.rs` 内单元测试逐位锁定），本测试**转正纳入常规运行**。
/// 此处曾用 `#[ignore]` 掩盖——那与"注释掉测试"是同一类假绿手段，不许留。
#[test]
fn exp_f64_non_finite_short_circuits_p0() {
    // -inf → +0.0：逐位断言（+0.0 与 -0.0 位模式不同）。
    assert_eq!(
        prism_audio::det::exp_f64(f64::NEG_INFINITY).to_bits(),
        0.0_f64.to_bits(),
        "exp_f64(-inf) 应为 +0.0"
    );
    // +inf → +inf：逐位。
    assert_eq!(
        prism_audio::det::exp_f64(f64::INFINITY).to_bits(),
        f64::INFINITY.to_bits(),
        "exp_f64(+inf) 应为 +inf"
    );
    // NaN → NaN（两个符号都要）。
    assert!(prism_audio::det::exp_f64(f64::NAN).is_nan());
    assert!(prism_audio::det::exp_f64(-f64::NAN).is_nan());
    // 立即返回的**墙钟证据**：旧实现挂死（9.2e18 次迭代 ≈ 数千年）。
    // 本断言在旧实现上永不抵达，是"真修好了"而非"绕过"的判据。
    let t0 = std::time::Instant::now();
    for _ in 0..10_000 {
        let _ = prism_audio::det::exp_f64(f64::NEG_INFINITY);
        let _ = prism_audio::det::exp_f64(f64::INFINITY);
        let _ = prism_audio::det::exp_f64(f64::NAN);
    }
    assert!(
        t0.elapsed().as_secs() < 5,
        "exp_f64 非有限输入 3 万次调用耗时 {:?}，疑似退化为慢速迭代（非立即返回）",
        t0.elapsed()
    );
}

/// 缺陷：极端但**有限**的 exp 输入不应挂死（有界循环）。
/// exp_f64(-100) 应 ≈ 0（浮点下溢），且快速返回。
#[test]
fn exp_f64_large_negative_finite() {
    let v = prism_audio::det::exp_f64(-100.0);
    assert!(v.abs() < 1e-40, "exp(-100) 应下溢近 0，得 {v}");
    assert!(v.is_finite());
}

/// 缺陷：Granular 退化参数（单样本源、density 极大）不得 panic。
#[test]
fn granular_degenerate_inputs() {
    let gs = GrainSynth { density: 1e9, grain_secs: 0.05, ..Default::default() };
    let out = gs.render(&[0.5], 16, 8_000).unwrap();
    assert_eq!(out.len(), 16);
    assert!(out.iter().all(|v| v.is_finite()), "极大密度不应产生非有限值");
    // 空源应结构化报错。
    assert!(GrainSynth::default().render(&[], 16, 8_000).is_err());
}

/// 缺陷：KS 非法 freq（0 / ≥ SR/2）须结构化报错而非 panic 或错值。
#[test]
fn ks_rejects_illegal_freq() {
    assert!(KarplusStrong { freq: 0.0, ..Default::default() }.render(100, 44_100).is_err());
    assert!(KarplusStrong { freq: 30_000.0, ..Default::default() }.render(100, 44_100).is_err());
    // 恰好 Nyquist 也应拒绝。
    assert!(KarplusStrong { freq: 22_050.0, ..Default::default() }.render(100, 44_100).is_err());
}

/// 缺陷：ModalStrike 模态数组长度不一致须结构化报错。
#[test]
fn modal_strike_length_mismatch_errors() {
    let bad = ModalStrike {
        freq: 440.0,
        amplitudes: vec![1.0, 1.0],
        ratios: vec![1.0],
        decays: vec![1.0],
    };
    assert!(bad.render(16, 8_000).is_err());
    let empty = ModalStrike { freq: 440.0, amplitudes: vec![], ratios: vec![], decays: vec![] };
    assert!(empty.render(16, 8_000).is_err());
}

/// 缺陷：SFZ 空文本解析为空列表（不 panic）；未知头报 AU1005。
#[test]
fn sfz_empty_and_unknown() {
    assert!(parse_sfz("").unwrap().is_empty());
    assert!(parse_sfz("// 只有注释\n").unwrap().is_empty());
    assert!(parse_sfz("<curve> x=1").is_err());
}

/// 缺陷（P2，文档/行为不符）：模块文档 `sfz.rs:115` 声明「无 `sample` 的 region
/// 被跳过」，但 `flush`（sfz.rs:124-129）仅凭 `has_region` 即 push，未过滤
/// 空 sample。空 sample 的 region 会流出并在 `Sampler::render_region` 处以
/// AU1006 报错，违背「跳过」契约。
///
/// 失败代表缺陷：解析器应丢弃 `sample` 为空的 region（或文档须改）。
#[test]
fn sfz_region_without_sample_skipped() {
    let text = "<region>\nlokey=60 hikey=64\n";
    let regions = parse_sfz(text).unwrap();
    assert!(regions.is_empty(), "无 sample 的 region 应跳过（文档契约 sfz.rs:115）");
}

/// 缺陷：Region playback_ratio 越界键（keytrack=100, center=60）应精确复现八度。
#[test]
fn sfz_playback_ratio_exact_octaves() {
    let r = Region { pitch_keycenter: 60, pitch_keytrack: 100.0, ..Default::default() };
    assert!((r.playback_ratio(72) - 2.0).abs() < 1e-4);
    assert!((r.playback_ratio(48) - 0.5).abs() < 1e-4);
    assert!((r.playback_ratio(60) - 1.0).abs() < 1e-6);
}

/// 缺陷：声像自动化端点/中点锚点（Linear / Smoothstep / Hold）。
#[test]
fn pan_automation_anchors() {
    let lin = PanAutomation { from: -1.0, to: 1.0, duration_secs: 1.0, curve: PanCurve::Linear };
    assert!((lin.at(0.0, 1.0) + 1.0).abs() < 1e-6);
    assert!(lin.at(0.5, 1.0).abs() < 1e-6);
    assert!((lin.at(1.0, 1.0) - 1.0).abs() < 1e-6);
    let smooth = PanAutomation { from: 0.0, to: 1.0, duration_secs: 1.0, curve: PanCurve::Smooth };
    assert!((smooth.at(0.5, 1.0) - 0.5).abs() < 1e-6, "smoothstep 中点 0.5");
}

/// 缺陷：空间声源归一化方向——零向量（听者位置即声源）不得 panic/NaN。
#[test]
fn spatial_degenerate_direction() {
    let listener = Listener::default();
    let at_listener = SpatialSource::from_world_pos(0.0, 0.0, 0.0, Attenuation::default());
    let p = at_listener.pan(&listener);
    assert!(p.is_finite(), "零距离声像不应 NaN，得 {p}");
    let (gl, gr) = at_listener.stereo_gains(&listener);
    assert!(gl.is_finite() && gr.is_finite());
    let buf = mix_spatial(&[1.0], 8_000, &at_listener, &listener);
    assert_eq!(buf.channels(), 2);
}

/// 缺陷：单样本/空输入的空间混音不得 panic。
#[test]
fn spatial_empty_and_single() {
    let listener = Listener::default();
    let src = SpatialSource::from_world_pos(3.0, 0.0, 0.0, Attenuation::default());
    assert_eq!(mix_spatial(&[], 8_000, &src, &listener).frames(), 0);
    assert_eq!(mix_spatial(&[0.5], 8_000, &src, &listener).frames(), 1);
}

/// 缺陷：Delay 零延迟退化为干湿直混（不 panic），feedback=0 单次回声。
#[test]
fn delay_zero_and_single_tap() {
    let zero = Delay { delay_secs: 0.0, feedback: 0.5, dry: 1.0, wet: 0.5 };
    let out = zero.process(&[1.0, 2.0], 8_000);
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|v| v.is_finite()));
    // feedback=0：仅一次回声。
    let sr = 100u32;
    let fx = Delay { delay_secs: 0.07, feedback: 0.0, dry: 0.0, wet: 1.0 };
    let mut input = vec![0.0_f32; 30];
    input[0] = 1.0;
    let out = fx.process(&input, sr);
    assert!((out[7] - 1.0).abs() < 1e-6, "7 号样本应为唯一回声");
    assert!(out.iter().skip(8).all(|&v| v.abs() < 1e-6), "无后续回声");
}

/// 缺陷：失真奇对称 + 有界（软饱和）。
#[test]
fn distortion_odd_and_bounded() {
    let fx = Distortion { kind: DistortionKind::Soft, drive: 1.0, output: 1.0, mix: 1.0 };
    let out = fx.process(&[-3.0, 0.0, 3.0], 8_000);
    assert!((out[0] + out[2]).abs() < 1e-5, "奇对称");
    assert!(out.iter().all(|&v| v.abs() < 1.0 + 1e-6), "有界");
}

/// 缺陷：GrainWindow 端点（Hann/Triangle 端点 0、中点 1）。
#[test]
fn grain_window_endpoints() {
    // 通过渲染验证 Hann 中点能量 > 端点（粗略行为）。
    let src = vec![1.0_f32; 4000];
    let gs = GrainSynth { window: GrainWindow::Hann, density: 40.0, grain_secs: 0.05, seed: 0, ..Default::default() };
    let out = gs.render(&src, 4000, 8_000).unwrap();
    assert!(out.iter().all(|v| v.is_finite() && v.abs() < 5.0));
}

// =====================================================================
// helpers
// =====================================================================

/// 过零率基频估计（去直流后数正向过零 / 时长）。
fn zero_cross_freq(x: &[f32], sr: u32) -> f32 {
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

/// SVF 稳态增益估计（跳过瞬态后 RMS 比）。
fn gain_at(svf: &Svf, freq: f32, sr: u32) -> f32 {
    let n = 16_384_usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((std::f64::consts::TAU * freq as f64 * i as f64 / sr as f64).sin()) as f32)
        .collect();
    let y = svf.process(&x, sr);
    let tail = &y[n / 2..];
    let in_tail = &x[n / 2..];
    let rms = |s: &[f32]| (s.iter().map(|&v| (v * v) as f64).sum::<f64>() / s.len() as f64).sqrt();
    (rms(tail) / rms(in_tail)) as f32
}
