//! W7-B13 作曲系统验收集成测试（黑盒：仅经 `prism-audio` 公共 API）。
//!
//! 覆盖任务书验收项：
//!
//! 1. 音阶库与音级↔频率 12-TET 查表（A4 锚点全八度逐位精确）；
//! 2. 和弦构造 + 功能语法状态机进行（合法性/确定性）+ 声部引导规则；
//! 3. 琶音器确定性（同种子逐位一致、模式语义）；
//! 4. step 鼓机步进断言（网格落点/时值/事件序）；
//! 5. MIDI 导入对拍（手写 SMF → 事件表）；
//! 6. 作曲系统驱动的 4 情绪配乐（网格合法/确定性/渲染非平凡）。

use prism_audio::arp::{arpeggiate, total_steps, ArpMode, ArpSpec};
use prism_audio::drums::{render as drum_render, sequence as drum_seq, Drum, DrumSpec};
use prism_audio::harmony::{
    build_chord, degree_function, progression, voice_lead, ChordQuality, Function, ProgressionSpec,
};
use prism_audio::midi::parse as midi_parse;
use prism_audio::score::{compose_pieces, midi_to_freq, render, ComposedSpec, Voice};
use prism_audio::synth::presets::Mood;
use prism_audio::theory::{scale, Key, ScaleKind};

/// 1. 音阶库规范 + 音高换算锚点（A4=440 全八度逐位精确）。
#[test]
fn scales_and_pitch_anchors() {
    for kind in [
        ScaleKind::Major,
        ScaleKind::NaturalMinor,
        ScaleKind::MajorPentatonic,
        ScaleKind::MinorPentatonic,
        ScaleKind::Dorian,
        ScaleKind::Phrygian,
        ScaleKind::Lydian,
        ScaleKind::Mixolydian,
    ] {
        let s = scale(kind);
        assert_eq!(s.degrees[0], 0);
        assert!(s.degrees.windows(2).all(|w| w[0] < w[1] && w[1] < 12));
    }
    assert_eq!(midi_to_freq(69).to_bits(), 440.0_f32.to_bits());
    assert_eq!(midi_to_freq(57).to_bits(), 220.0_f32.to_bits());
    assert_eq!(midi_to_freq(81).to_bits(), 880.0_f32.to_bits());
    assert_eq!(midi_to_freq(45).to_bits(), 110.0_f32.to_bits());
}

/// 2. 和弦构造 + 功能语法进行（合法后继 + 确定性）+ 声部引导规则。
#[test]
fn harmony_chords_progression_and_voice_leading() {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    assert_eq!(build_chord(&key, 0, ChordQuality::Triad).midis, vec![60, 64, 67]);
    assert_eq!(degree_function(ScaleKind::Major, 4), Function::Dominant);
    assert_eq!(degree_function(ScaleKind::Major, 3), Function::Subdominant);

    let spec = ProgressionSpec { key, chords: 8, seventh: false, seed: 0xB13 };
    let p = progression(&spec).unwrap();
    assert_eq!(p.len(), 8);
    assert_eq!(p.functions[0], Function::Tonic);
    // 重复调用逐位一致（确定性核心）。
    for _ in 0..50 {
        assert_eq!(progression(&spec).unwrap(), p);
    }
    // 声部引导：四声部等长、相邻声部无小二度冲突、确定性。
    let vl = voice_lead(&p);
    assert_eq!(vl.parts.len(), 4);
    for par in &vl.parts {
        assert_eq!(par.len(), 8);
    }
    for ci in 0..8 {
        for vi in 1..4 {
            assert_ne!((vl.parts[vi][ci] - vl.parts[vi - 1][ci]).abs(), 1);
        }
    }
    assert_eq!(voice_lead(&p), vl);
}

/// 3. 琶音器：模式语义 + 确定性 + 成分排列性质。
#[test]
fn arpeggiator_modes_and_determinism() {
    let key = Key { kind: ScaleKind::Major, root_midi: 60 };
    let chord = build_chord(&key, 0, ChordQuality::Triad);
    let up = arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 1, octaves: 2, repeats: 1, seed: 0 }).unwrap();
    assert_eq!(up.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![60, 64, 67, 72, 76, 79]);
    assert_eq!(total_steps(&up), 6);
    let down = arpeggiate(&chord, &ArpSpec { mode: ArpMode::Down, step: 1, octaves: 1, repeats: 1, seed: 0 }).unwrap();
    assert_eq!(down.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![67, 64, 60]);
    // 随机模式确定性 + 排列。
    let rspec = ArpSpec { mode: ArpMode::Random, step: 1, octaves: 2, repeats: 1, seed: 0x5EED };
    let a = arpeggiate(&chord, &rspec).unwrap();
    for _ in 0..50 {
        assert_eq!(arpeggiate(&chord, &rspec).unwrap(), a);
    }
    let mut got: Vec<i32> = a.iter().map(|n| n.midi).collect();
    let mut exp = vec![60, 64, 67, 72, 76, 79];
    got.sort_unstable();
    exp.sort_unstable();
    assert_eq!(got, exp);
    // 每个音的频率与音高一致。
    for n in &a {
        assert_eq!(n.freq, midi_to_freq(n.midi));
    }
}

/// 4. step 鼓机：网格落点/时值/事件序/渲染确定性。
#[test]
fn drum_machine_step_assertions() {
    let spec = DrumSpec::four_on_floor(16).unwrap();
    let track = drum_seq(&spec, 120.0, 8_000).unwrap();
    // 16 步 @120bpm 16 分音符 8kHz：step = 1000 帧。
    assert_eq!(track.total_frames, 16_000);
    for h in &track.hits {
        assert_eq!(h.start_frame, h.step * 1_000);
        assert_eq!(h.frames, 1_000);
        match h.drum {
            Drum::Kick => assert_eq!(h.step % 4, 0),
            Drum::Snare => assert_eq!(h.step % 4, 2),
            Drum::Hat => assert_eq!(h.step % 2, 0),
        }
    }
    let buf = drum_render(&track);
    assert_eq!(buf.as_slice(), drum_render(&track).as_slice(), "鼓机渲染逐位一致");
    assert!(buf.peak() > 0.05);
}

/// 5. MIDI 导入对拍：手写 SMF（format 0、PPQ 96、单音 + running status）。
#[test]
fn midi_import_matches_reference() {
    let mut f: Vec<u8> = Vec::new();
    f.extend_from_slice(b"MThd");
    f.extend_from_slice(&6u32.to_be_bytes());
    f.extend_from_slice(&0u16.to_be_bytes());
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&96u16.to_be_bytes());
    let mut ev: Vec<u8> = Vec::new();
    ev.extend_from_slice(&[0x00, 0x90, 60, 100]); // note-on C4
    ev.extend_from_slice(&[0x60, 0x80, 60, 0]); // note-off @ tick 96
    ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
    f.extend_from_slice(b"MTrk");
    f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
    f.extend_from_slice(&ev);

    let mf = midi_parse(&f, 44_100).unwrap();
    assert_eq!(mf.format, 0);
    assert_eq!(mf.ppq, 96);
    assert_eq!(mf.notes.len(), 1);
    let n = &mf.notes[0];
    assert_eq!(n.midi, 60);
    assert_eq!(n.velocity, 100);
    assert_eq!(n.start_frame, 0);
    assert_eq!(n.frames, 22_050, "tick96 @120bpm 44.1k = 四分音符 = 22050 帧");
    assert_eq!(mf.total_frames, 22_050);
    // 幂等：同字节重复解析逐位一致。
    assert_eq!(midi_parse(&f, 44_100).unwrap(), mf);
}

/// 6. 作曲系统驱动的 4 情绪配乐：网格合法、确定性、渲染非平凡。
#[test]
fn composed_scores_all_moods() {
    for mood in [Mood::Tense, Mood::Calm, Mood::Bright, Mood::Epic] {
        let spec = ComposedSpec { mood, bars: 4, seed: 0x42, arpeggiate: false };
        let score = compose_pieces(&spec, 8_000).unwrap();
        for ev in &score.events {
            assert!(ev.start_frame + ev.frames <= score.total_frames, "{mood:?} 事件越界");
            assert_eq!(ev.freq, midi_to_freq(ev.midi));
        }
        for _ in 0..20 {
            assert_eq!(compose_pieces(&spec, 8_000).unwrap(), score);
        }
        let buf = render(&score);
        assert!(buf.peak() > 0.05, "{mood:?} 配乐不应静音");
        assert_eq!(buf.frames(), score.total_frames);
        // 存在三个声部。
        for v in [Voice::Bass, Voice::Pad, Voice::Lead] {
            assert!(score.events.iter().any(|e| e.voice == v), "{mood:?} 缺 {v:?} 声部");
        }
    }
}
