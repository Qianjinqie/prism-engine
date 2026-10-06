//! W7 QA 攻击性测试 —— prism-audio W7-B13（theory/harmony/arp/drums/midi）。
//!
//! 目标：乐理正确性、MIDI VLQ/running status、确定性、边界。每个测试注明失败含义。

use prism_audio::arp::{arpeggiate, total_steps, ArpMode, ArpSpec};
use prism_audio::drums::{sequence, Drum, DrumSpec};
use prism_audio::harmony::{build_chord, ChordQuality, ProgressionSpec};
use prism_audio::harmony::{degree_function, progression, voice_lead, Function};
use prism_audio::midi::parse;
use prism_audio::theory::{
    midi_to_freq, pitch_to_midi, scale, validate_root, Key, ScaleKind, SEMITONE_RATIO,
};

// ==================================================================
// A. 乐理正确性
// ==================================================================

/// 缺陷：A4 = MIDI 69 = 440 Hz，全八度锚点精确（2 的整数幂连乘）。
#[test]
fn a4_and_octave_anchors_exact() {
    assert_eq!(midi_to_freq(69).to_bits(), 440.0_f32.to_bits(), "A4 应精确 440");
    assert_eq!(midi_to_freq(57).to_bits(), 220.0_f32.to_bits());
    assert_eq!(midi_to_freq(81).to_bits(), 880.0_f32.to_bits());
    assert_eq!(midi_to_freq(60).to_bits(), 261.625_55_f32.to_bits(), "C4 ≈ 261.626");
}

/// 缺陷：音阶度数↔频率正确（C 大调 degree 4 = G4 = 392 Hz）。
#[test]
fn scale_degree_to_freq() {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    assert_eq!(key.degree_to_midi(0), 60, "I");
    assert_eq!(key.degree_to_midi(4), 67, "V → G");
    let g = key.degree_to_freq(4);
    assert!((g - 391.995).abs() < 0.5, "G4 ≈ 392，得 {g}");
    assert_eq!(key.degree_to_midi(7), 72, "上八度 C");
    assert_eq!(key.degree_to_midi(-1), 59, "导音 B3");
}

/// 缺陷：大三和弦构成 = 4+3 半音（根-大三-纯五）。
#[test]
fn major_triad_intervals() {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    let i = build_chord(&key, 0, ChordQuality::Triad);
    assert_eq!(i.midis, vec![60, 64, 67], "C 大三 = 60/64/67");
    assert_eq!(i.midis[1] - i.midis[0], 4, "根→三度 = 4 半音");
    assert_eq!(i.midis[2] - i.midis[1], 3, "三度→五度 = 3 半音");
}

/// 缺陷：小调三级 = 3+4 半音（A 小三 = A-C-E）。
#[test]
fn minor_triad_intervals() {
    let key = Key { kind: ScaleKind::NaturalMinor, root_midi: 57 };
    let i = build_chord(&key, 0, ChordQuality::Triad);
    assert_eq!(i.midis, vec![57, 60, 64], "Am = A-C-E");
    assert_eq!(i.midis[1] - i.midis[0], 3, "小三度 = 3 半音");
    assert_eq!(i.midis[2] - i.midis[1], 4, "大三度 = 4 半音");
}

/// 缺陷：七和弦 = 三和弦 + 三度（C 大调 Imaj7 = C-E-G-B）。
#[test]
fn seventh_chord_construction() {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    let i7 = build_chord(&key, 0, ChordQuality::Seventh);
    assert_eq!(i7.midis, vec![60, 64, 67, 71], "Imaj7 = C-E-G-B");
    assert_eq!(i7.midis.len(), 4);
}

/// 缺陷：SEMITONE_RATIO 表与 2^(k/12) 一致（无 pow 的常数表正确性）。
#[test]
fn semitone_table_matches_12tet() {
    // 纯五度（k=7）= 1.4983071；参考标准值。
    assert!((SEMITONE_RATIO[7] - 1.498_307_1).abs() < 1e-6);
    assert!((SEMITONE_RATIO[1] - 1.059_463_1).abs() < 1e-6);
    // C5/C4 比值 = 表[0]=1（八度由连乘处理）。
    assert!((midi_to_freq(72) / midi_to_freq(60) - 2.0).abs() < 1e-5);
}

/// 缺陷：pitch_to_midi 科学音高记号（C4=60、A4=69、越界折八度）。
#[test]
fn pitch_class_notation() {
    assert_eq!(pitch_to_midi(4, 0), 60);
    assert_eq!(pitch_to_midi(4, 9), 69);
    assert_eq!(pitch_to_midi(4, 12), 72, "音级 12 → C5");
    assert_eq!(pitch_to_midi(4, -1), 59, "音级 −1 → B3");
}

/// 缺陷：根音越界校验报 THE1001。
#[test]
fn root_validation_bounds() {
    assert!(validate_root(0).is_ok() && validate_root(127).is_ok());
    assert_eq!(validate_root(128).unwrap_err().code, "THE1001");
    assert_eq!(validate_root(-1).unwrap_err().code, "THE1001");
}

/// 缺陷：库内所有音阶规范（含根音 0、严格递增、< 12）。
#[test]
fn all_scales_wellformed() {
    for kind in prism_audio::theory::ALL_SCALES {
        let s = scale(kind);
        assert_eq!(s.degrees[0], 0, "{kind:?} 必含根音");
        assert!(s.degrees.windows(2).all(|w| w[0] < w[1]), "{kind:?} 须严格递增");
        assert!(s.degrees.iter().all(|&d| d < 12), "{kind:?} 增量须 < 12");
    }
}

// ==================================================================
// B. 和声 / 声部引导
// ==================================================================

/// 缺陷：进行必从 T 起、每步合法后继、同种子逐位一致。
#[test]
fn progression_functional_deterministic() {
    let spec = ProgressionSpec {
        key: Key { kind: ScaleKind::Major, root_midi: 60 },
        chords: 8,
        seventh: false,
        seed: 0x1234,
    };
    let p = progression(&spec).unwrap();
    assert_eq!(p.functions[0], Function::Tonic, "须从主功能起");
    for w in p.functions.windows(2) {
        let ok = match w[0] {
            Function::Tonic => true,
            Function::Subdominant => w[1] != Function::Subdominant,
            Function::Dominant => w[1] == Function::Tonic,
        };
        assert!(ok, "非法功能转移 {:?}→{:?}", w[0], w[1]);
    }
    for _ in 0..20 {
        assert_eq!(progression(&spec).unwrap(), p, "同种子应逐位一致");
    }
}

/// 缺陷：chords=0 报 HRM1001。
#[test]
fn progression_zero_chords_errors() {
    let spec = ProgressionSpec {
        key: Key { kind: ScaleKind::Major, root_midi: 60 },
        chords: 0,
        seventh: false,
        seed: 1,
    };
    assert_eq!(progression(&spec).unwrap_err().code, "HRM1001");
}

/// 缺陷：和弦成员永远落在调内（度数折叠正确）。
#[test]
fn chord_members_in_key() {
    let key = Key { kind: ScaleKind::NaturalMinor, root_midi: 55 };
    for deg in -5..12 {
        let c = build_chord(&key, deg, ChordQuality::Seventh);
        for &m in &c.midis {
            assert!(key.contains_midi(m), "和弦音 {m} 应在调内（度 {deg}）");
        }
        assert!(c.midis.windows(2).all(|w| w[0] < w[1]), "成分严格递增 deg={deg}");
    }
}

/// 缺陷：度数→功能映射符合文档（大调 I/iii/vi = T、ii/IV = S、V/vii = D）。
#[test]
fn degree_function_mapping() {
    assert_eq!(degree_function(ScaleKind::Major, 0), Function::Tonic);
    assert_eq!(degree_function(ScaleKind::Major, 2), Function::Tonic);
    assert_eq!(degree_function(ScaleKind::Major, 5), Function::Tonic);
    assert_eq!(degree_function(ScaleKind::Major, 1), Function::Subdominant);
    assert_eq!(degree_function(ScaleKind::Major, 3), Function::Subdominant);
    assert_eq!(degree_function(ScaleKind::Major, 4), Function::Dominant);
    assert_eq!(degree_function(ScaleKind::Major, 6), Function::Dominant);
    // 度数负/越界应折 e（rem_euclid 7）。
    assert_eq!(degree_function(ScaleKind::Major, 7), Function::Tonic);
    assert_eq!(degree_function(ScaleKind::Major, -1), Function::Dominant);
}

/// 缺陷：声部引导四声部等长、落音区、无小二度冲突、确定性。
#[test]
fn voice_leading_invariants() {
    for seed in [1u64, 42, 7777] {
        let prog = progression(&ProgressionSpec {
            key: Key { kind: ScaleKind::Major, root_midi: 60 },
            chords: 8,
            seventh: false,
            seed,
        })
        .unwrap();
        let vl = voice_lead(&prog);
        assert_eq!(vl.parts.len(), 4);
        for part in &vl.parts {
            assert_eq!(part.len(), prog.len());
        }
        for ci in 0..prog.len() {
            for vi in 1..4 {
                assert_ne!(
                    (vl.parts[vi][ci] - vl.parts[vi - 1][ci]).abs(),
                    1,
                    "相邻小二度冲突 seed={seed} ci={ci}"
                );
            }
        }
        assert_eq!(voice_lead(&prog), vl, "确定性");
    }
}

// ==================================================================
// C. 琶音器
// ==================================================================

fn c_major_triad() -> prism_audio::harmony::Chord {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    build_chord(&key, 0, ChordQuality::Triad)
}

/// 缺陷：上行模式布局与 step 时序。
#[test]
fn arp_up_layout() {
    let chord = c_major_triad();
    let notes = arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 1, octaves: 2, repeats: 1, seed: 0 }).unwrap();
    assert_eq!(notes.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![60, 64, 67, 72, 76, 79]);
    for (i, n) in notes.iter().enumerate() {
        assert_eq!(n.start_step, i);
        assert_eq!(n.freq, midi_to_freq(n.midi));
    }
    assert_eq!(total_steps(&notes), 6);
}

/// 缺陷：上下往复含端点（60,64,67,64）。
#[test]
fn arp_updown_endpoints() {
    let chord = c_major_triad();
    let ud = arpeggiate(&chord, &ArpSpec { mode: ArpMode::UpDown, step: 1, octaves: 1, repeats: 1, seed: 0 }).unwrap();
    assert_eq!(ud.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![60, 64, 67, 64]);
}

/// 缺陷：随机模式确定性 + 是成分的排列（PCG Fisher-Yates 正确性）。
#[test]
fn arp_random_permutation_deterministic() {
    let chord = c_major_triad();
    let spec = ArpSpec { mode: ArpMode::Random, step: 1, octaves: 2, repeats: 1, seed: 0x5EED };
    let a = arpeggiate(&chord, &spec).unwrap();
    for _ in 0..50 {
        assert_eq!(arpeggiate(&chord, &spec).unwrap(), a, "随机模式跨调用应逐位一致");
    }
    let mut got: Vec<i32> = a.iter().map(|n| n.midi).collect();
    let mut want = vec![60, 64, 67, 72, 76, 79];
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want, "随机模式应为上行成分的排列");
}

/// 缺陷：非法参数 step<1 / octaves>4 / repeats<1 报 ARP1001。
#[test]
fn arp_rejects_bad_params() {
    let chord = c_major_triad();
    assert_eq!(arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 0, octaves: 2, repeats: 1, seed: 0 }).unwrap_err().code, "ARP1001");
    assert_eq!(arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 1, octaves: 5, repeats: 1, seed: 0 }).unwrap_err().code, "ARP1001");
    assert_eq!(arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 1, octaves: 2, repeats: 0, seed: 0 }).unwrap_err().code, "ARP1001");
}

// ==================================================================
// D. 鼓机
// ==================================================================

/// 缺陷：4/4 默认型网格正确、越界静音。
#[test]
fn drum_four_on_floor_grid() {
    let spec = DrumSpec::four_on_floor(16).unwrap();
    for s in 0..16 {
        assert_eq!(spec.hits(Drum::Kick, s), s % 4 == 0);
        assert_eq!(spec.hits(Drum::Snare, s), s % 4 == 2);
        assert_eq!(spec.hits(Drum::Hat, s), s % 2 == 0);
    }
    assert!(!spec.hits(Drum::Kick, 16), "越界应静音");
}

/// 缺陷：编排步对齐、总帧数精确、事件序固定升序。
#[test]
fn drum_sequence_alignment() {
    let spec = DrumSpec::four_on_floor(16).unwrap();
    let track = sequence(&spec, 120.0, 8_000).unwrap();
    assert_eq!(track.hits.len(), 16, "kick4+snare4+hat8");
    assert_eq!(track.total_frames, 16 * 1000, "16 分音符 @120bpm 8k = 1000 帧/步");
    let keys: Vec<(usize, usize)> = track.hits.iter().map(|h| (h.step, h.drum as usize)).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(keys, sorted, "事件序应为 (step,drum) 升序");
}

/// 缺陷：bpm≤0 报 DRM1002；steps=0 报 DRM1001。
#[test]
fn drum_validation() {
    assert_eq!(sequence(&DrumSpec::four_on_floor(4).unwrap(), 0.0, 8_000).unwrap_err().code, "DRM1002");
    assert!(DrumSpec::four_on_floor(0).is_err());
}

// ==================================================================
// E. MIDI 解析（VLQ / running status / delta 累积）
// ==================================================================

fn minimal_smf() -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(b"MThd");
    f.extend_from_slice(&6u32.to_be_bytes());
    f.extend_from_slice(&0u16.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&96u16.to_be_bytes());
    let mut ev = Vec::new();
    ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
    ev.extend_from_slice(&[0x60, 0x80, 60, 0]);
    ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    f.extend_from_slice(b"MTrk");
    f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
    f.extend_from_slice(&ev);
    f
}

/// 缺陷：基本音符配对与时值（tick 96 @120bpm 44.1k = 22050 帧）。
#[test]
fn midi_basic_note() {
    let mf = parse(&minimal_smf(), 44_100).unwrap();
    assert_eq!(mf.ppq, 96);
    assert_eq!(mf.notes.len(), 1);
    let n = &mf.notes[0];
    assert_eq!(n.midi, 60);
    assert_eq!(n.velocity, 100);
    assert_eq!(n.frames, 22_050);
    assert_eq!(n.freq, midi_to_freq(60));
}

/// 缺陷：VLQ 解码符合规范（含 4 字节上限）。
#[test]
fn midi_vlq_spec_conformance() {
    assert_eq!(parse_vlq(&[0x7F]), Some(127));
    assert_eq!(parse_vlq(&[0x81, 0x7F]), Some(255));
    assert_eq!(parse_vlq(&[0x82, 0x80, 0x00]), Some(32768));
    assert_eq!(parse_vlq(&[0xC0, 0x00]), Some(8192));
    // 5 字节 VLQ 应报错（超 4 字节）。
    assert!(build_bad_vlq_and_parse().is_err());
}

/// 缺陷：running status + note-on(vel=0) 结束音符 + delta 累积正确。
#[test]
fn midi_running_status_and_delta_accumulation() {
    let mut f = Vec::new();
    f.extend_from_slice(b"MThd");
    f.extend_from_slice(&6u32.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&96u16.to_be_bytes());
    let mut ev = Vec::new();
    ev.extend_from_slice(&[0x00, 0x90, 60, 100]); // t=0 note on
    ev.extend_from_slice(&[0x30, 64, 80]); // delta 48, running status, note on 64
    ev.extend_from_slice(&[0x30, 60, 0]); // delta 48, running status, note-on vel0 = off 60
    ev.extend_from_slice(&[0x00, 0x80, 64, 0]); // note off 64
    ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    f.extend_from_slice(b"MTrk");
    f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
    f.extend_from_slice(&ev);

    let mf = parse(&f, 8_000).unwrap();
    assert_eq!(mf.notes.len(), 2);
    let n60 = mf.notes.iter().find(|n| n.midi == 60).unwrap();
    let n64 = mf.notes.iter().find(|n| n.midi == 64).unwrap();
    assert_eq!(n60.start_frame, 0, "60 起于 tick0");
    assert_eq!(n60.frames, 4_000, "60 时长 tick0–96 = 4000 帧");
    assert_eq!(n64.start_frame, 2_000, "64 起于 tick48 = 2000 帧");
    assert_eq!(n64.frames, 2_000);
    assert_eq!(n64.velocity, 80);
}

/// 缺陷：错误路径——非头块 / sr=0 / SMPTE。
#[test]
fn midi_error_paths() {
    assert_eq!(parse(b"XXXXnope", 8_000).unwrap_err().code, "MID1001");
    assert_eq!(parse(&minimal_smf(), 0).unwrap_err().code, "MID1002");
    let mut smpte = minimal_smf();
    smpte[12] = 0xE8;
    smpte[13] = 0x50;
    assert_eq!(parse(&smpte, 8_000).unwrap_err().code, "MID1003");
}

/// 缺陷：截断/畸形字节流不得 panic（结构化报错或安全中断）。
#[test]
fn midi_truncated_no_panic() {
    let full = minimal_smf();
    for cut in 0..full.len() {
        let _ = parse(&full[..cut], 44_100);
    }
}

// ==================================================================
// helper：通过公开 parse 间接验证 VLQ（无直接导出 read_vlq）
// ==================================================================

/// 用单事件 SMF 探出 VLQ 值：delta = 该值，note-off 后 frames 反映 tick。
fn parse_vlq(bytes: &[u8]) -> Option<u32> {
    // 构造 note-on @0, note-off @delta=bytes, 用 1000000 SR → frames 微小。
    let mut f = Vec::new();
    f.extend_from_slice(b"MThd");
    f.extend_from_slice(&6u32.to_be_bytes());
    f.extend_from_slice(&0u16.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&96u16.to_be_bytes()); // ppq 96
    let mut ev = Vec::new();
    ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
    ev.push(0x00); // placeholder delta byte replaced below
    ev.pop();
    ev.extend_from_slice(bytes);
    ev.extend_from_slice(&[0x80, 60, 0]);
    ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    f.extend_from_slice(b"MTrk");
    f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
    f.extend_from_slice(&ev);
    // ppq 96, sr = 96*2 = 192 → frames_per_qn = 192*0.5 = 96 → 1 tick = 1 frame.
    let mf = parse(&f, 192).ok()?;
    mf.notes.first().map(|n| n.frames as u32)
}

/// 构造含 5 字节 VLQ 的 SMF，应解析失败（超 4 字节）。
fn build_bad_vlq_and_parse() -> Result<prism_audio::midi::MidiFile, prism_audio::AudioError> {
    let mut f = Vec::new();
    f.extend_from_slice(b"MThd");
    f.extend_from_slice(&6u32.to_be_bytes());
    f.extend_from_slice(&0u16.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&96u16.to_be_bytes());
    let mut ev = Vec::new();
    ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
    ev.extend_from_slice(&[0x81, 0x81, 0x81, 0x81, 0x00]); // 5 字节 VLQ
    ev.extend_from_slice(&[0x80, 60, 0]);
    ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    f.extend_from_slice(b"MTrk");
    f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
    f.extend_from_slice(&ev);
    parse(&f, 192)
}
