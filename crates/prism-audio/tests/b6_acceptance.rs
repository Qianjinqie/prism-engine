//! W5-B6 验收集成测试（黑盒：仅经 `prism-audio` 公共 API 与 `prism-core`
//! 公共类型，不触碰 crate 内部）。
//!
//! 覆盖任务书验收项：
//!
//! 1. 端到端：乐谱编排 → PCM 渲染 → WAV 编码 → 解码（逐字节往返）；
//! 2. 响应式寻址等效：第 N 帧独立重算（抽窗 + 单帧烘焙）== 顺序烘焙行（逐位）；
//! 3. 起始检测：脉冲列 onset 定位 ±1 帧（±1 hop）、BPM ≈ 120；
//! 4. 混音器：线性叠加能量守恒、峰值防削波、淡入淡出斜坡；
//! 5. 节点注册表（D-27 静态汇入）+ EvalCtx 快照接线（架构 §4.4）；
//! 6. 乐谱黄金文件（固定规格 → 事件表字面量冻结）。

use std::f64::consts::TAU;

use prism_audio::beats::{detect_beats, BeatsConfig};
use prism_audio::buffer::PcmBuffer;
use prism_audio::mixer::{mix_tracks, MixTrack};
use prism_audio::nodes::{__prism_collect, AudioReactiveNode};
use prism_audio::reactive::{bake_reactive, install_bake, ReactiveConfig};
use prism_audio::score::{compose, midi_to_freq, render, ScoreSpec, Voice};
use prism_audio::synth::presets::Mood;
use prism_audio::wav::{decode_wav, encode_wav, BitDepth};
use prism_core::node::MathStructure;
use prism_core::{EvalCtx, NodeRegistry, Payload};

/// 端到端：compose → render → encode_wav → decode_wav。
///
/// 16/24 bit：解码值落在量化半格内，且**重编码逐字节一致**（编码是解码的
/// 逆映射——网格值 round(clamp(x)·2^(bits−1)) 幂等）；32 bit 为原始位，
/// 逐样本位一致。
#[test]
fn end_to_end_score_to_wav_roundtrip() {
    let score = compose(&ScoreSpec { mood: Mood::Bright, bars: 2, seed: 5 }, 8_000).unwrap();
    let buf = render(&score);
    assert_eq!(buf.sample_rate(), 8_000);
    assert_eq!(buf.frames(), score.total_frames, "渲染帧数 = 编排总帧数");
    assert!(buf.peak() > 0.0, "配乐段非平凡（有声音）");

    // 16 bit：半格 = 0.5/2^15。
    let bytes16 = encode_wav(&buf, BitDepth::B16);
    let dec16 = decode_wav(&bytes16).unwrap();
    assert_eq!(dec16.frames(), buf.frames());
    assert_eq!(encode_wav(&dec16, BitDepth::B16), bytes16, "16bit 重编码逐字节一致");
    for (i, (a, b)) in buf.as_slice().iter().zip(dec16.as_slice()).enumerate() {
        assert!(
            (a - b).abs() <= 0.5 / 32_768.0 + 1e-9,
            "样本 {i} 超 16bit 量化半格：{a} vs {b}"
        );
    }

    // 24 bit：半格 = 0.5/2^23。
    let bytes24 = encode_wav(&buf, BitDepth::B24);
    let dec24 = decode_wav(&bytes24).unwrap();
    assert_eq!(encode_wav(&dec24, BitDepth::B24), bytes24, "24bit 重编码逐字节一致");
    for (i, (a, b)) in buf.as_slice().iter().zip(dec24.as_slice()).enumerate() {
        assert!((a - b).abs() <= 0.5 / 8_388_608.0 + 1e-12, "样本 {i} 超 24bit 半格");
    }

    // 32 bit：原始 f32 位，无损。
    let bytes32 = encode_wav(&buf, BitDepth::B32);
    let dec32 = decode_wav(&bytes32).unwrap();
    for (i, (a, b)) in buf.as_slice().iter().zip(dec32.as_slice()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "32bit 样本 {i} 位不一致");
    }
}

/// 响应式寻址等效（公共 API）：顺序烘焙的第 N 帧行 == 抽取窗口
/// `[N·hop, N·hop+win)` 后以 hop=win 单帧烘焙的第 0 帧行（逐位）。
#[test]
fn reactive_addressing_equivalence() {
    let (sr, len) = (8_000_u32, 8_000_usize);
    let data: Vec<f32> =
        (0..len).map(|i| (TAU * 1_000.0 * i as f64 / sr as f64).sin() as f32).collect();
    let buf = PcmBuffer::from_mono(sr, data.clone()).unwrap();
    let cfg = ReactiveConfig { win: 512, hop: 250, band_edges: vec![0.0, 200.0, 2_000.0, 4_000.0] };
    let bake = bake_reactive(&buf, &cfg).unwrap();
    assert_eq!(bake.frames, len.div_ceil(250), "帧数 = ⌈len/hop⌉");
    assert_eq!(bake.band_count(), 3);

    // 首帧 / 中间 / 尾部（尾部窗口需要零填充）逐帧对拍。
    for n in [0_usize, 1, 15, 30, bake.frames - 1] {
        let start = n * 250;
        let mut window = vec![0.0_f32; 512];
        for (i, w) in window.iter_mut().enumerate() {
            let idx = start + i;
            if idx < len {
                *w = data[idx];
            }
        }
        let single = bake_reactive(
            &PcmBuffer::from_mono(sr, window).unwrap(),
            &ReactiveConfig { win: 512, hop: 512, band_edges: cfg.band_edges.clone() },
        )
        .unwrap();
        for b in 0..single.band_count() {
            assert_eq!(
                single.band(0, b).to_bits(),
                bake.band(n as u64, b).to_bits(),
                "帧 {n} 频段 {b}：独立重算 ≠ 顺序行"
            );
        }
    }
}

/// 起始检测（公共 API）：8 个单位脉冲（120 BPM @ 8kHz，间隔 4000 样本），
/// 窗 0 的脉冲被 Hann 压零，其余应检出；每个 onset 距最近 click ≤ ±1 hop；
/// IOI 中值 → BPM ≈ 120。
#[test]
fn onset_detection_within_one_frame() {
    let (sr, period) = (8_000_u32, 4_000_usize);
    let mono: Vec<f32> =
        (0..32_000).map(|i| if i % period == 0 { 1.0 } else { 0.0 }).collect();
    let clicks: Vec<usize> = (0..32_000).step_by(period).collect();
    let table = detect_beats(
        &PcmBuffer::from_mono(sr, mono).unwrap(),
        &BeatsConfig { win: 512, hop: 250, ..Default::default() },
    )
    .unwrap();
    assert!((6..=8).contains(&table.onsets.len()), "检出 {} 个 onset", table.onsets.len());
    for o in &table.onsets {
        let nearest = clicks
            .iter()
            .copied()
            .min_by_key(|&c| (c as isize - o.sample as isize).abs())
            .unwrap();
        let dist = (nearest as isize - o.sample as isize).abs() as usize;
        assert!(dist <= 250, "onset 样本 {} 偏离 click {nearest} 达 {dist} > ±1 hop", o.sample);
    }
    assert!((table.bpm - 120.0).abs() < 2.0, "BPM 估计 {} 应≈120", table.bpm);
}

/// 混音器（公共 API）：① 0.5+0.5 线性叠加逐位还原源（能量守恒特例）；
/// ② 不同增益的能量守恒（f64 相对误差 < 1e-5）+ 峰值 ≤ 1；③ 淡入淡出斜坡
/// 中点与终点。
#[test]
fn mixer_energy_conservation_and_fades() {
    let (sr, n) = (8_000_u32, 4_000_usize);
    let sig: Vec<f32> =
        (0..n).map(|i| (TAU * 440.0 * i as f64 / sr as f64).sin() as f32).collect();
    let sig2: Vec<f32> =
        (0..n).map(|i| (TAU * 660.0 * i as f64 / sr as f64).sin() as f32).collect();
    let energy = |x: &[f32]| -> f64 { x.iter().map(|v| (*v as f64) * (*v as f64)).sum() };

    // ① 0.5·x + 0.5·x = x（f32 半值精确，逐位还原）。
    let src = PcmBuffer::from_mono(sr, sig.clone()).unwrap();
    let out = mix_tracks(&[
        MixTrack::new(PcmBuffer::from_mono(sr, sig.clone()).unwrap(), 0.5),
        MixTrack::new(PcmBuffer::from_mono(sr, sig.clone()).unwrap(), 0.5),
    ])
    .unwrap();
    assert_eq!(out.as_slice(), src.as_slice(), "0.5+0.5 应逐位还原源");

    // ② 能量守恒：E(0.6a+0.3b) = 0.36·E(a) + 0.09·E(b)。
    let out = mix_tracks(&[
        MixTrack::new(PcmBuffer::from_mono(sr, sig.clone()).unwrap(), 0.6),
        MixTrack::new(PcmBuffer::from_mono(sr, sig2.clone()).unwrap(), 0.3),
    ])
    .unwrap();
    let e_exp = 0.36 * energy(&sig) + 0.09 * energy(&sig2);
    let e_out = energy(out.as_slice());
    assert!(
        (e_out - e_exp).abs() / e_exp < 1e-5,
        "能量不守恒：out={e_out:e} exp={e_exp:e}"
    );
    assert!(out.peak() <= 1.0 + 1e-6, "峰值防削波失效：{}", out.peak());

    // ③ 淡入淡出：单轨 fin=fout=0.1s（400 样本）；中点按斜坡取值，首尾为 0。
    let out = mix_tracks(&[MixTrack {
        buffer: PcmBuffer::from_mono(sr, sig.clone()).unwrap(),
        gain: 1.0,
        fade_in_secs: 0.1,
        fade_out_secs: 0.1,
    }])
    .unwrap();
    assert_eq!(out.sample(0, 0), Some(0.0), "淡入起点为 0");
    assert_eq!(out.sample(n - 1, 0), Some(0.0), "淡出终点为 0");
    let mid_in = out.sample(100, 0).unwrap();
    assert!(
        (mid_in - sig[100] * (100.0 / 400.0)).abs() < 1e-6,
        "淡入中点 {mid_in} 偏离线性斜坡"
    );
}

/// 节点注册表 + EvalCtx 接线（公共 API 端到端）：
/// D-27 静态汇入两节点 → 乐谱渲染 → 烘焙 → install_bake 装入快照库 →
/// `apply(Scalar(band))` 与直查烘焙表逐位一致；缺烘焙显式降级 Empty。
#[test]
fn registry_and_evalctx_wiring() {
    let mut reg = NodeRegistry::new();
    __prism_collect(&mut reg);
    assert_eq!(reg.len(), 2, "prism-audio 恰好汇入 2 个节点");
    let reactive = reg.resolve_by_name("AudioReactiveNode").expect("按名解析 reactive");
    let level = reg.resolve_by_name("AudioLevelNode").expect("按名解析 level");

    // 端到端数据链：配乐 → PCM → 烘焙表 → 快照库。
    let score = compose(&ScoreSpec { mood: Mood::Calm, bars: 2, seed: 9 }, 8_000).unwrap();
    let bake = bake_reactive(
        &render(&score),
        &ReactiveConfig { win: 512, hop: 250, band_edges: vec![0.0, 200.0, 2_000.0, 4_000.0] },
    )
    .unwrap();

    for frame in [0_u64, 10, bake.frames as u64 - 1] {
        let mut ctx = EvalCtx::new(1, 1, frame);
        install_bake(&bake, &mut ctx);
        for band in 0..bake.band_count() {
            assert_eq!(
                reactive.apply(&Payload::Scalar(band as f32), &ctx),
                Payload::Scalar(bake.band(frame, band)),
                "帧 {frame} 频段 {band}：节点输出应逐位等于直查"
            );
        }
        let mut sum = 0.0_f32;
        for b in 0..bake.band_count() {
            sum += bake.band(frame, b);
        }
        assert_eq!(
            level.apply(&Payload::Empty, &ctx),
            Payload::Scalar(sum / bake.band_count() as f32),
            "帧 {frame}：level = 固定序频段均值"
        );
    }

    // 缺烘焙（未装配快照）→ Empty，不静默编造数值。
    let bare = EvalCtx::new(1, 1, 0);
    assert_eq!(reactive.apply(&Payload::Scalar(0.0), &bare), Payload::Empty);
    assert_eq!(level.apply(&Payload::Empty, &bare), Payload::Empty);

    // 直接构造的节点与注册表解析实例行为一致（同一实现）。
    let direct = AudioReactiveNode::default();
    let mut ctx = EvalCtx::new(1, 1, 5);
    install_bake(&bake, &mut ctx);
    assert_eq!(
        direct.apply(&Payload::Scalar(1.0), &ctx),
        reactive.apply(&Payload::Scalar(1.0), &ctx)
    );
}

/// 乐谱黄金文件：`Tense / 2 小节 / 种子 42 / 8000 Hz` 的事件表字面量冻结。
///
/// 网格算术：138 BPM @ 8kHz → step = round(60/138/2·8000) = **1739** 帧，
/// bar = step·8 = 13912 帧，total = 27824 帧。构造序 = 小节 → 声部
/// （Pad 三和弦 → Bass 两拍根音 → Lead 逐 step 门控）→ step。任何破坏
/// 网格算术、12-TET 音高、PCG 槽位布局或声部顺序的改动都会使本表失配。
#[test]
fn score_golden_events() {
    let score = compose(&ScoreSpec { mood: Mood::Tense, bars: 2, seed: 42 }, 8_000).unwrap();
    assert!((score.bpm - 138.0).abs() < 1e-6, "bpm = {}", score.bpm);
    assert_eq!(score.total_frames, 27_824, "total = 2 bar × 13912");
    const PAD: f32 = 0.22;
    const BASS: f32 = 0.5;
    const LEAD: f32 = 0.4;
    let golden: &[(usize, usize, i32, Voice, f32)] = &[
        // 小节 0（degree 0）：Pad 三和弦 45/48/52，Bass 根音低八度 33。
        (0, 13912, 45, Voice::Pad, PAD),
        (0, 13912, 48, Voice::Pad, PAD),
        (0, 13912, 52, Voice::Pad, PAD),
        (0, 3478, 33, Voice::Bass, BASS),
        (6956, 3478, 33, Voice::Bass, BASS),
        // 小节 0 Lead：step 0–5 门控入，step 6–7 休止（密度 0.75）。
        (0, 1739, 84, Voice::Lead, LEAD),
        (1739, 1739, 77, Voice::Lead, LEAD),
        (3478, 1739, 81, Voice::Lead, LEAD),
        (5217, 1739, 88, Voice::Lead, LEAD),
        (6956, 1739, 76, Voice::Lead, LEAD),
        (8695, 1739, 81, Voice::Lead, LEAD),
        // 小节 1（degree 1）：Pad 46/50/53，Bass 34。
        (13912, 13912, 46, Voice::Pad, PAD),
        (13912, 13912, 50, Voice::Pad, PAD),
        (13912, 13912, 53, Voice::Pad, PAD),
        (13912, 3478, 34, Voice::Bass, BASS),
        (20868, 3478, 34, Voice::Bass, BASS),
        // 小节 1 Lead：step 0–7 全部门控入。
        (13912, 1739, 74, Voice::Lead, LEAD),
        (15651, 1739, 70, Voice::Lead, LEAD),
        (17390, 1739, 74, Voice::Lead, LEAD),
        (19129, 1739, 79, Voice::Lead, LEAD),
        (20868, 1739, 91, Voice::Lead, LEAD),
        (22607, 1739, 77, Voice::Lead, LEAD),
        (24346, 1739, 74, Voice::Lead, LEAD),
        (26085, 1739, 82, Voice::Lead, LEAD),
    ];
    assert_eq!(score.events.len(), golden.len(), "事件数应与黄金表一致");
    for (i, (e, g)) in score.events.iter().zip(golden.iter()).enumerate() {
        assert_eq!(e.start_frame, g.0, "事件 {i} start_frame");
        assert_eq!(e.frames, g.1, "事件 {i} frames");
        assert_eq!(e.midi, g.2, "事件 {i} midi");
        assert_eq!(e.voice, g.3, "事件 {i} voice");
        assert_eq!(e.gain.to_bits(), g.4.to_bits(), "事件 {i} gain");
        assert_eq!(e.freq, midi_to_freq(e.midi), "事件 {i} freq = midi_to_freq(midi)");
    }
}
