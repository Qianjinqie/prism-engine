//! prism-audio 边缘攻击测试（QA）：FFT / WAV / mixer / buffer / det 的
//! 边界、退化、确定性与能量守恒攻击。
//!
//! 每个断言都在"若失败表示何种缺陷"的口径下撰写（不是复述既有测试正条）。

use prism_audio::buffer::PcmBuffer;
use prism_audio::det;
use prism_audio::fft::{hann_window, Fft};
use prism_audio::mixer::{mix_tracks, MixTrack};
use prism_audio::wav::{decode_wav, encode_wav, BitDepth};

/// 简易索引可复现伪随机（splitmix64）产生任意 f32 位模式样本。
fn lcg_samples(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u32 << 24) as f32 - 0.5 // [-0.5,0.5)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

/// 与朴素 DFT 解析对拍：n=64 随机输入，逐 bin 实/虚部偏差应 < 1e-3。
/// 失败 => radix-2 蝶形/位反转/旋转因子表步进存在数学错误。
#[test]
fn fft_forward_matches_naive_dft() {
    let n = 64_usize;
    let fft = Fft::new(n).unwrap();
    let x = lcg_samples(0x1234_5678, n);
    let (re, im) = fft.forward(&x).unwrap();
    for m in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for (i, &xi) in x.iter().enumerate() {
            // 用 det 三角件构造 e^{-2πi·mi/n}（与实现同源，专测结构不做幅度归约）。
            let ang = -2.0 * std::f64::consts::PI * (m * i) as f64 / n as f64;
            sr += xi as f64 * det::cos_f64(ang);
            si += xi as f64 * det::sin_f64(ang);
        }
        let dr = (re[m] as f64 - sr).abs();
        let di = (im[m] as f64 - si).abs();
        assert!(dr < 1e-3 && di < 1e-3, "bin {m}: 实差 {dr} 虚差 {di}");
    }
}

/// DC 输入（全 1）→ 仅 X[0]=n，其余 bin≈0；恒值即单一 DC 分量。
/// 失败 => FFT 泄漏或归一化结构错误。
#[test]
fn fft_dc_input_concentrates_at_bin_zero() {
    let n = 128_usize;
    let fft = Fft::new(n).unwrap();
    let (re, im) = fft.forward(&vec![1.0_f32; n]).unwrap();
    assert!((re[0] - n as f32).abs() < 1e-2, "X[0] 应为 n，实得 {}", re[0]);
    for k in 1..n {
        assert!(
            re[k].abs() < 1e-2 && im[k].abs() < 1e-2,
            "bin {k} 非零泄漏 re={} im={}",
            re[k],
            im[k]
        );
    }
}

/// 线性性：FFT(a·x + b·y) = a·FFT(x) + b·FFT(y)。
/// 失败 => 蝶形运算实现非线性的结构缺陷。
#[test]
fn fft_is_linear() {
    let n = 32_usize;
    let fft = Fft::new(n).unwrap();
    let x = lcg_samples(1, n);
    let y = lcg_samples(2, n);
    let (a, b) = (0.75_f32, -1.25_f32);
    let comb: Vec<f32> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + b * yi).collect();
    let (cr, ci) = fft.forward(&comb).unwrap();
    let (xr, xi_) = fft.forward(&x).unwrap();
    let (yr, yi_) = fft.forward(&y).unwrap();
    for m in 0..n {
        let wr = a * xr[m] + b * yr[m];
        let wi = a * xi_[m] + b * yi_[m];
        assert!(
            (cr[m] - wr).abs() < 1e-3 && (ci[m] - wi).abs() < 1e-3,
            "bin {m} 线性性破坏"
        );
    }
}

/// Parseval：Σ|X|² = n·Σ|x|²（f32 累积容差）。失败 => 蝶形能量泄漏。
#[test]
fn fft_parseval_holds_on_random_signal() {
    let n = 256_usize;
    let fft = Fft::new(n).unwrap();
    let x = lcg_samples(0xDEAD_BEEF, n);
    let (re, im) = fft.forward(&x).unwrap();
    let spec: f64 = re.iter().zip(&im).map(|(&r, &i)| (r * r + i * i) as f64).sum();
    let time: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let rel = ((spec - n as f64 * time) / (n as f64 * time)).abs();
    assert!(rel < 1e-3, "Parseval 相对偏差 {rel}");
}

/// 微小点数 n=2 与非法 n 的边界：2 点应等于直接加减；1/非幂拒绝。
/// 失败 => 边界点数下位反转/表构造越界或漏拒。
#[test]
fn fft_two_point_and_illegal_sizes() {
    let fft = Fft::new(2).unwrap();
    let (re, im) = fft.forward(&[1.0_f32, 2.0]).unwrap();
    assert_eq!((re[0], re[1]), (3.0, -1.0), "2 点 DFT 应为 (和, 差)");
    assert_eq!((im[0], im[1]), (0.0, 0.0));
    assert_eq!(Fft::new(1).unwrap_err().code, "AU1002");
    assert_eq!(Fft::new(3).unwrap_err().code, "AU1002");
    assert_eq!(Fft::new(0).unwrap_err().code, "AU1002");
}

/// Hann 窗：长度 0/1 不 panic；n=1 无零除；对称性 w[i]==w[n-i]（周期形式）。
/// 失败 => 空/单元素窗发生除零或越界。
#[test]
fn hann_window_degenerate_lengths() {
    assert!(hann_window(0).is_empty());
    let w1 = hann_window(1);
    assert_eq!(w1.len(), 1);
    assert!(w1[0].is_finite(), "n=1 不应产生 NaN（0/0）");
    let n = 16;
    let w = hann_window(n);
    // 周期形式：w[0]=0，且 w[i] == w[n-i] 对 i∈[1,n) 成立（对称）。
    assert_eq!(w[0], 0.0);
    for i in 1..n {
        assert!(
            (w[i] - w[n - i]).abs() < 1e-6,
            "Hann 窗非对称 i={i}: {} vs {}",
            w[i],
            w[n - i]
        );
    }
}

/// 变换确定性 + 反复复用实例不串扰：fwd 多次逐位一致，且顺序无关。
/// 失败 => FFT 实例隐含可变状态导致帧间串扰（非"纯函数"）。
#[test]
fn fft_reuse_no_state_leak_bitwise() {
    let n = 64_usize;
    let fft = Fft::new(n).unwrap();
    let x = lcg_samples(7, n);
    let y = lcg_samples(9, n);
    let xf = fft.forward(&x).unwrap();
    let _interleaved = fft.forward(&y).unwrap(); // 夹一次其它输入
    let xf_again = fft.forward(&x).unwrap();
    assert_eq!(xf.0, xf_again.0);
    assert_eq!(xf.1, xf_again.1);
}

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

/// 三种位深对任意位模式样本编码：编码→解码→重编码逐字节一致。
/// 失败 => 舍入/符号扩展/钳位在某区间破坏往返（量化栅格外的解码值）。
#[test]
fn wav_roundtrip_bit_exact_random_bits() {
    for depth in [BitDepth::B16, BitDepth::B24, BitDepth::B32] {
        // 覆盖负零、极大、极小、普通值、满幅边界。
        let mut data: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 0.5, -0.5, f32::MIN_POSITIVE];
        data.extend(lcg_samples(0xABCD, 61));
        let buf = PcmBuffer::from_mono(44_100, data).unwrap();
        let bytes = encode_wav(&buf, depth);
        let dec = decode_wav(&bytes).expect("自产 WAV 必可解码");
        assert_eq!(dec.frames(), buf.frames(), "{depth:?} 帧数守恒");
        let again = encode_wav(&dec, depth);
        assert_eq!(again, bytes, "{depth:?} 往返逐字节应一致");
    }
}

/// 空缓冲（0 帧）编码 → data 长度 0，解码得 0 帧；不 panic。
/// 失败 => 0 帧 WAV 的块尺寸/对齐处理漏拒或崩溃。
#[test]
fn wav_zero_frames_roundtrip() {
    let buf = PcmBuffer::new(44_100, 1);
    let bytes = encode_wav(&buf, BitDepth::B16);
    assert_eq!(bytes.len(), 44, "0 帧 WAV 应仅头 44 字节");
    let dec = decode_wav(&bytes).unwrap();
    assert_eq!(dec.frames(), 0);
}

/// 截断/损坏字节流不得 panic，只返回结构化错误（WAV1001/1002）。
/// 失败 => 解码器在信任的尺寸字段上越界 panic（库代码 panic = 缺陷）。
#[test]
fn wav_truncated_and_corrupt_never_panics() {
    let good = encode_wav(
        &PcmBuffer::from_mono(8_000, vec![0.1, 0.2, 0.3]).unwrap(),
        BitDepth::B24,
    );
    // 在任意长度处截断，必须 Err 不 panic。
    for cut in 0..good.len() {
        let r = std::panic::catch_unwind(|| decode_wav(&good[..cut]));
        assert!(r.is_ok(), "截断到 {cut} 字节时 panic");
        assert!(r.unwrap().is_err(), "截断到 {cut} 字节应报错");
    }
    // 篡改 data 尺寸字段为超大值 → 应报"截断"类错误，不 panic。
    let mut over = good.clone();
    // data 块头位置：44 - ... 由编码器固定序，找到 b"data" 后 4 字节。
    let dpos = over.windows(4).position(|w| w == b"data").unwrap();
    over[dpos + 4..dpos + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    let r = std::panic::catch_unwind(|| decode_wav(&over));
    assert!(r.is_ok(), "data 尺寸溢出时 panic");
    assert_eq!(r.unwrap().unwrap_err().code, "WAV1001");
}

/// 单声道/立体声通道布局还原正确，且 24bit 符号扩展对负值正确。
/// 失败 => 交错拆解或 24bit 补码符号扩展错位。
#[test]
fn wav_24bit_sign_extension_and_channels() {
    let inter = vec![-1.0_f32, 1.0, -0.5, 0.5]; // 2 帧 × 2 ch
    let buf = PcmBuffer::from_interleaved(48_000, 2, inter).unwrap();
    let dec = decode_wav(&encode_wav(&buf, BitDepth::B24)).unwrap();
    assert_eq!((dec.channels(), dec.frames()), (2, 2));
    assert_eq!(dec.sample(0, 0), Some(-1.0));
    assert_eq!(dec.sample(0, 1), Some(8_388_607.0 / 8_388_608.0));
    // 负半幅应精确还原到最近栅格（非符号错误造成的正数）。
    assert!(dec.sample(1, 0).unwrap() < 0.0, "负值被错误符号扩展");
}

// ---------------------------------------------------------------------------
// mixer / buffer
// ---------------------------------------------------------------------------

/// NaN 输入传播检查：NaN 样本经混音不得静默变 0（应保持 NaN 可见）或 panic。
/// 失败 => 混音对非法样本静默吞掉（掩盖上游缺陷），与"结构化错误"契约不符。
#[test]
fn mixer_nan_input_not_silently_zeroed() {
    let a = PcmBuffer::from_mono(1_000, vec![f32::NAN, 0.5, 0.0, 1.0]).unwrap();
    let mixed = mix_tracks(&[MixTrack::new(a, 1.0)]).unwrap();
    assert!(mixed.sample(0, 0).unwrap().is_nan(), "NaN 应在混音后可见");
    // peak 含 NaN：实现用 fold max，NaN 语义需可预期（不 panic 即可）。
    let _ = mixed.peak();
}

/// 增益 0 的轨道应产生全零输出（peak=0），且不触发除零（peak>1 分支不进入）。
/// 失败 => 0 增益下误触 1/peak 产生 Inf/NaN。
#[test]
fn mixer_zero_gain_yields_silence_no_div_zero() {
    let a = PcmBuffer::from_mono(1_000, vec![1.0, -1.0, 0.5]).unwrap();
    let mut t = MixTrack::new(a, 0.0);
    t.fade_in_secs = 0.0;
    let out = mix_tracks(&[t]).unwrap();
    assert!(out.as_slice().iter().all(|&x| x == 0.0), "0 增益应静音");
    assert_eq!(out.peak(), 0.0);
}

/// 淡入淡出时长超过缓冲长度时不越界、端点为 0，且不产生负增益。
/// 失败 => 斜坡索引越界或 g_out 变负（负增益翻转相位）。
#[test]
fn mixer_fade_longer_than_buffer_is_safe() {
    let a = PcmBuffer::from_mono(1_000, vec![1.0; 10]).unwrap();
    let t = MixTrack {
        buffer: a,
        gain: 1.0,
        fade_in_secs: 5.0,  // 5000 样本 > 10 帧
        fade_out_secs: 5.0,
    };
    let out = mix_tracks(&[t]).unwrap();
    assert_eq!(out.frames(), 10);
    assert!(out.as_slice().iter().all(|&x| x >= 0.0), "不应出现负样本");
    assert_eq!(out.sample(0, 0), Some(0.0), "淡入端点必 0");
}

/// 混音顺序无关性（交换律）：两轨相加与顺序无关，逐位一致。
/// 失败 => 累加顺序影响结果（浮点结合性泄漏进契约）。
#[test]
fn mixer_two_tracks_commutative_bitwise() {
    let a = PcmBuffer::from_mono(8_000, lcg_samples(11, 256)).unwrap();
    let b = PcmBuffer::from_mono(8_000, lcg_samples(22, 256)).unwrap();
    // 两轨都不削波（幅度小），保证 1/peak 缩放不发生，纯加法交换律。
    let ab = mix_tracks(&[MixTrack::new(a.clone(), 1.0), MixTrack::new(b.clone(), 1.0)]).unwrap();
    let ba = mix_tracks(&[MixTrack::new(b, 1.0), MixTrack::new(a, 1.0)]).unwrap();
    assert_eq!(ab.as_slice(), ba.as_slice(), "浮点加法应对称");
}

/// 空 PcmBuffer 的派生量：frames=0、duration=0、peak=0、to_mono 空。
/// 失败 => 空缓冲除零或返回非法时长。
#[test]
fn pcm_buffer_empty_derived_quantities() {
    let b = PcmBuffer::new(48_000, 2);
    assert_eq!(b.frames(), 0);
    assert_eq!(b.duration_secs(), 0.0);
    assert_eq!(b.peak(), 0.0);
    assert!(b.to_mono().is_empty());
    assert_eq!(b.sample(0, 0), None);
}

/// 超多声道下混：固定序单次除法的确定性（重复逐位一致）。
/// 失败 => to_mono 的累加/除法非确定（跨帧状态或不同序）。
#[test]
fn pcm_buffer_to_mono_deterministic_multichannel() {
    let ch = 8u16;
    let data: Vec<f32> = lcg_samples(33, ch as usize * 32);
    let b = PcmBuffer::from_interleaved(48_000, ch, data).unwrap();
    let m1 = b.to_mono();
    for _ in 0..500 {
        assert_eq!(b.to_mono(), m1, "to_mono 非确定");
    }
}

// ---------------------------------------------------------------------------
// det（确定性三角件）
// ---------------------------------------------------------------------------

/// det 三角件值域：对宽范围输入 |sin|≤1、|cos|≤1，无 NaN/Inf。
/// 失败 => 区间折叠在某象限越界给出 >1 或非有限值（相位归约缺陷）。
#[test]
fn det_trig_bounded_over_wide_range() {
    let mut x = -1e6_f64;
    while x <= 1e6 {
        let s = det::sin_f64(x);
        let c = det::cos_f64(x);
        assert!(s.is_finite() && s.abs() <= 1.0 + 1e-9, "sin({x})={s}");
        assert!(c.is_finite() && c.abs() <= 1.0 + 1e-9, "cos({x})={c}");
        // 勾股恒等：sin²+cos²=1。
        assert!((s * s + c * c - 1.0).abs() < 1e-6, "勾股恒等破坏 @{x}");
        x += 7919.0; // 素数步长覆盖大量相位模态
    }
}

/// det 奇偶性：sin(-x) = -sin(x)，cos(-x)=cos(x)。
/// 失败 => 象限查表对符号处理错误。
#[test]
fn det_trig_parity() {
    for &x in &[0.1_f64, 1.234, 3.0, 100.5, 12345.678] {
        assert!((det::sin_f64(-x) + det::sin_f64(x)).abs() < 1e-9, "sin 非奇");
        assert!((det::cos_f64(-x) - det::cos_f64(x)).abs() < 1e-9, "cos 非偶");
    }
}
