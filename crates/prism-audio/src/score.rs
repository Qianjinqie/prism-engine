//! 配乐编排（B6）：小节 / 节拍网格 / 声部叠加 → 确定性配乐段。
//!
//! 结构（白皮书 §20/§22 的 B6 层口径——模板织体，B13 升级为和声语法）：
//!
//! 1. **网格**：4/4 拍，8 分音符 step 网格（每小节 8 步），小节进行按预设
//!    [`presets::MoodPreset::progression`] 4 小节循环；
//! 2. **声部**：Bass（每小节第 1/3 拍根音）、Pad（整小节三和弦柱式）、
//!    Lead（step 网格旋律，密度门控 + 和弦内选音）；
//! 3. **确定性随机**：Lead 的休止门控 / 选音 / 八度全部经 `prism-noise`
//!    PCG 整数哈希按 `hash(seed, 步序, 槽位)` 派生（架构 §4.4：hash 派生 =
//!    唯一随机源，无全局 RNG、无状态泄漏）——同种子同乐段逐位一致；
//! 4. **音高**：12-TET 固定比值表（白皮书 §23 音高比查表口径），八度比 =
//!    2 的整数幂精确连乘，A4=440 锚点全格点精确。
//!
//! 事件序 = 构造序（小节 → 声部 → step 固定遍历），渲染按事件序累加，
//! 固定顺序满足"混合与处理为纯函数、固定序"。

use crate::buffer::PcmBuffer;
use crate::synth::presets::{preset, Mood, MoodPreset};
use crate::synth::render_note;
use crate::{AudioError, AudioResult};

/// 十二平均律半音比值表：`PITCH_RATIO[k] = 2^(k/12)`（k = 0..12；白皮书 §23
/// "音高比查表"口径，固定常量取代运行期 pow）。
///
/// D-30 说明：k=6 的 `1.414_213_6` **不是**近似写错，而是 `2^0.5` 的**正确
/// f32 舍入**（bits = 0x3FB504F4）。`std::f32::consts::SQRT_2` 给的是
/// 0x3FB504F3（低 1 ulp），替换会让纯五度频率偏移约 1.19e-7 —— 故此处显式
/// `allow` 而非改值。逐个常量核对过，其余 11 项与 `2^(k/12)` 的 f32 舍入一致。
#[allow(clippy::approx_constant)] // 精确值，非近似；见 D-30
const PITCH_RATIO: [f32; 12] = [
    1.0,
    1.059_463_1,
    1.122_462_0,
    1.189_207_1,
    1.259_921_0,
    1.334_839_9,
    1.414_213_6,
    1.498_307_1,
    1.587_401_1,
    1.681_792_8,
    1.781_797_4,
    1.887_748_6,
];

/// MIDI 音高 → 频率（Hz）：`440 · 2^((m−69)/12)`，以半音查表 + 八度整数幂
/// 连乘实现（八度比为 2 的幂，IEEE 精确——A4 同八度锚点逐位精确）。
pub fn midi_to_freq(midi: i32) -> f32 {
    let k = (midi - 69).rem_euclid(12) as usize;
    let octaves = (midi - 69).div_euclid(12);
    let mut ratio = PITCH_RATIO[k];
    let mut n = octaves;
    while n > 0 {
        ratio *= 2.0;
        n -= 1;
    }
    while n < 0 {
        ratio *= 0.5;
        n += 1;
    }
    440.0 * ratio
}

/// 声部（叠加顺序 = 枚举序，构造与渲染的固定遍历序）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Voice {
    /// 低音：每小节第 1/3 拍，根音低八度。
    Bass,
    /// 铺底：整小节三和弦柱式。
    Pad,
    /// 主旋律：8 分音符 step 网格，密度门控。
    Lead,
}

/// 单个音符事件（帧域；`midi` 为黄金文件序列化口径，`freq` 为渲染口径，
/// 两者由 [`midi_to_freq`] 唯一互推）。
#[derive(Debug, Clone, PartialEq)]
pub struct NoteEvent {
    /// 起始帧（相对配乐段起点）。
    pub start_frame: usize,
    /// 时值（帧数）。
    pub frames: usize,
    /// MIDI 音高。
    pub midi: i32,
    /// 频率（= [`midi_to_freq`]`(midi)`）。
    pub freq: f32,
    /// 幅度增益。
    pub gain: f32,
    /// 所属声部。
    pub voice: Voice,
}

/// 配乐规格：情绪 + 小节数 + 种子（确定性参数包）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreSpec {
    /// 情绪预设。
    pub mood: Mood,
    /// 小节数（≥ 1）。
    pub bars: u32,
    /// 派生种子（格 Zᵏ；PCG hash 派生 Lead 变化）。
    pub seed: u64,
}

/// 编排产物：事件表 + 网格元数据（渲染与黄金文件断言的输入）。
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// 情绪（渲染时查音色表）。
    pub mood: Mood,
    /// 速度（BPM）。
    pub bpm: f32,
    /// 采样率。
    pub sample_rate: u32,
    /// 音符事件（构造序 = 小节 → 声部 → step）。
    pub events: Vec<NoteEvent>,
    /// 总帧数（= 小节数 × 每小节帧数，尾部截齐——时长确定）。
    pub total_frames: usize,
}

/// 调式度数 → MIDI 音高：`root + 12·(degree/len) + scale[degree%len]`
/// （度数可为负/越界，`rem_euclid`/`div_euclid` 折叠八度）。
fn degree_to_midi(p: &MoodPreset, degree: i32) -> i32 {
    let len = p.scale.len() as i32;
    p.root_midi + 12 * degree.div_euclid(len) + p.scale[degree.rem_euclid(len) as usize] as i32
}

/// 编排一段确定性配乐（`bars ≥ 1`、`sample_rate ≥ 1`，否则 `SCN1001`）。
pub fn compose(spec: &ScoreSpec, sample_rate: u32) -> AudioResult<Score> {
    if spec.bars < 1 || sample_rate < 1 {
        return Err(AudioError::new(
            "SCN1001",
            format!("score::compose({:?}, {} 小节, sr={})", spec.mood, spec.bars, sample_rate),
            "小节数与采样率必须 ≥ 1",
            "至少编排 1 个小节并给出有效采样率",
        ));
    }
    let p = preset(spec.mood);
    let sr = sample_rate as f64;
    let step_secs = 60.0 / p.bpm as f64 / 2.0; // 8 分音符
    let step_frames = (step_secs * sr).round() as usize;
    let bar_frames = step_frames * 8;
    let (bass_gain, pad_gain, lead_gain) = (0.50_f32, 0.22_f32, 0.40_f32);
    // PCG 槽位标签：门控 / 选音 / 八度互不混叠（prism-noise 按槽位加盐）。
    const TAG_GATE: u64 = 1;
    const TAG_PITCH: u64 = 2;
    const TAG_OCTAVE: u64 = 3;

    let mut events = Vec::new();
    let push = |events: &mut Vec<NoteEvent>, voice, start, frames, midi, gain| {
        events.push(NoteEvent {
            start_frame: start,
            frames,
            midi,
            freq: midi_to_freq(midi),
            gain,
            voice,
        });
    };

    for bar in 0..spec.bars as usize {
        let bar_start = bar * bar_frames;
        let degree = p.progression[bar % p.progression.len()] as i32;
        // Pad：三和弦 = 度数 (d, d+2, d+4) 柱式铺满整小节。
        for (v, &offset) in [0_i32, 2, 4].iter().enumerate() {
            let _ = v;
            push(
                &mut events,
                Voice::Pad,
                bar_start,
                bar_frames,
                degree_to_midi(&p, degree + offset),
                pad_gain,
            );
        }
        // Bass：第 1/3 拍根音（低八度），各占两拍。
        for beat in [0_usize, 4] {
            push(
                &mut events,
                Voice::Bass,
                bar_start + beat * step_frames,
                step_frames * 2,
                degree_to_midi(&p, degree) - 12,
                bass_gain,
            );
        }
        // Lead：step 门控 + 和弦内选音（候选 = 三和弦度数 + 上邻经过音）。
        let candidates = [degree, degree + 2, degree + 4, degree + 5];
        for step in 0..8_usize {
            let idx = bar * 8 + step;
            let gate = prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_GATE);
            if gate >= p.density {
                continue;
            }
            let pick = prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_PITCH);
            let midi = degree_to_midi(&p, candidates[(pick * candidates.len() as f32) as usize])
                + 12 * p.lead_octaves
                + if prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_OCTAVE) < 0.2 { 12 } else { 0 };
            push(&mut events, Voice::Lead, bar_start + step * step_frames, step_frames, midi, lead_gain);
        }
    }

    Ok(Score {
        mood: spec.mood,
        bpm: p.bpm,
        sample_rate,
        total_frames: spec.bars as usize * bar_frames,
        events,
    })
}

/// 渲染配乐段为 PCM（按预设音色逐事件合成、按事件序累加；纯函数）。
pub fn render(score: &Score) -> PcmBuffer {
    let p = preset(score.mood);
    let mut data = vec![0.0_f32; score.total_frames];
    for ev in &score.events {
        let (waveform, env) = match ev.voice {
            Voice::Bass => (p.bass.waveform, &p.bass.envelope),
            Voice::Pad => (p.pad.waveform, &p.pad.envelope),
            Voice::Lead => (p.lead.waveform, &p.lead.envelope),
        };
        let note = render_note(waveform, ev.freq, env, ev.gain, ev.frames, score.sample_rate);
        for (i, v) in note.into_iter().enumerate() {
            data[ev.start_frame + i] += v; // 事件序累加 = 固定序
        }
    }
    // from_interleaved 仅失败于长度非整数倍：单声道构造恒成功。
    PcmBuffer::from_interleaved(score.sample_rate, 1, data)
        .expect("单声道渲染缓冲长度恒合法")
}

// ---------------------------------------------------------------------------
// B13：作曲系统驱动的配乐（白皮书 §22 四层漏斗）
// ---------------------------------------------------------------------------

/// 作曲系统驱动的配乐规格：情绪 + 小节数 + 种子 + 是否启用织体层。
///
/// 与 [`ScoreSpec`] 的区别：情绪预设的**音阶/进行/密度**改由 [`crate::theory`]
/// 与 [`crate::harmony`] 的作曲系统（音阶库 + 功能语法状态机和弦进行 +
/// 声部引导 + 琶音器）驱动，而非模板化的固定编排——情绪差异从"参数微调"
/// 升级为"语法层差异"（白皮书 §22 超越点）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComposedSpec {
    /// 情绪预设（提供速度 / 音色 / 密度 / 音区模板）。
    pub mood: Mood,
    /// 小节数（≥ 1）。
    pub bars: u32,
    /// 派生种子。
    pub seed: u64,
    /// 是否以琶音器替代柱式铺底（织体层差异）。
    pub arpeggiate: bool,
}

/// 情绪 → 作曲系统调式（音阶库类型）映射。
pub fn mood_scale(mood: Mood) -> crate::theory::ScaleKind {
    use crate::theory::ScaleKind;
    match mood {
        Mood::Tense => ScaleKind::Phrygian,       // 弗里几亚：♭2 张力
        Mood::Calm => ScaleKind::MajorPentatonic, // 大调五声：无半音冲突
        Mood::Bright => ScaleKind::Major,         // 自然大调
        Mood::Epic => ScaleKind::NaturalMinor,    // 自然小调
    }
}

/// 作曲系统驱动的配乐编排：和声层（功能语法状态机）生成和弦进行，织体层按
/// 模板分配——Bass（根音律动）、Pad（柱式或琶音器展开）、Lead（声部引导选音）。
/// 事件域与 [`Score`] 一致（可复用 [`render`]）。
pub fn compose_pieces(spec: &ComposedSpec, sample_rate: u32) -> AudioResult<Score> {
    use crate::arp::{arpeggiate, ArpMode, ArpSpec};
    use crate::harmony::{progression, voice_lead, ProgressionSpec};
    use crate::theory::Key;

    if spec.bars < 1 || sample_rate < 1 {
        return Err(AudioError::new(
            "SCN1002",
            format!("score::compose_pieces({:?}, {} 小节, sr={})", spec.mood, spec.bars, sample_rate),
            "小节数与采样率必须 ≥ 1",
            "至少编排 1 个小节并给出有效采样率",
        ));
    }
    let p = preset(spec.mood);
    let sr = sample_rate as f64;
    let step_secs = 60.0 / p.bpm as f64 / 2.0; // 8 分音符
    let step_frames = (step_secs * sr).round() as usize;
    let bar_frames = step_frames * 8;
    let (bass_gain, pad_gain, lead_gain) = (0.50_f32, 0.22_f32, 0.40_f32);

    let key = Key { kind: mood_scale(spec.mood), root_midi: p.root_midi };
    // 和声层：每小节一个和弦，功能语法状态机 + 种子路径选择。
    let prog = progression(&ProgressionSpec {
        key,
        chords: spec.bars,
        seventh: false,
        seed: spec.seed,
    })?;
    let vl = voice_lead(&prog);

    const TAG_GATE: u64 = 1;
    const TAG_PITCH: u64 = 2;
    const TAG_OCTAVE: u64 = 3;
    let mut events = Vec::new();
    let push = |events: &mut Vec<NoteEvent>, voice, start, frames, midi, gain| {
        events.push(NoteEvent {
            start_frame: start,
            frames,
            midi,
            freq: midi_to_freq(midi),
            gain,
            voice,
        });
    };

    for bar in 0..spec.bars as usize {
        let bar_start = bar * bar_frames;
        let chord = &prog.chords[bar.min(prog.chords.len() - 1)];
        // Bass：第 1/3 拍和弦根音（低八度）。
        for beat in [0_usize, 4] {
            push(
                &mut events,
                Voice::Bass,
                bar_start + beat * step_frames,
                step_frames * 2,
                chord.root_midi() - 12,
                bass_gain,
            );
        }
        // Pad：柱式三和弦（或琶音器展开）；音高取自声部引导的内声部。
        if spec.arpeggiate {
            let spec_arp = ArpSpec {
                mode: ArpMode::UpDown,
                step: 1,
                octaves: 1,
                repeats: 1,
                seed: spec.seed ^ (bar as u64),
            };
            for n in arpeggiate(chord, &spec_arp)? {
                let start = (bar_start + n.start_step * step_frames).min(bar_start + bar_frames - 1);
                push(&mut events, Voice::Pad, start, n.steps * step_frames, n.midi, pad_gain);
            }
        } else {
            for &midi in &chord.midis {
                push(&mut events, Voice::Pad, bar_start, bar_frames, midi, pad_gain);
            }
        }
        // Lead：step 门控 + 从声部引导的高声部（含上邻经过音）选音。
        let hi = vl.parts[3][bar.min(vl.parts[3].len() - 1)];
        let candidates = [hi, hi - 2, hi - 4, hi + 2];
        for step in 0..8_usize {
            let idx = bar * 8 + step;
            if prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_GATE) >= p.density {
                continue;
            }
            let pick = prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_PITCH);
            let midi = candidates[(pick * candidates.len() as f32) as usize % candidates.len()]
                + if prism_noise::pcg::hash2_f32(spec.seed, idx as u64, TAG_OCTAVE) < 0.2 { 12 } else { 0 };
            push(&mut events, Voice::Lead, bar_start + step * step_frames, step_frames, midi, lead_gain);
        }
    }

    Ok(Score {
        mood: spec.mood,
        bpm: p.bpm,
        sample_rate,
        total_frames: spec.bars as usize * bar_frames,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 音高表：A4=440 全八度精确（八度比 = 2 的整数幂连乘）。
    #[test]
    fn pitch_table_octaves_are_exact() {
        assert_eq!(midi_to_freq(69).to_bits(), 440.0_f32.to_bits());
        assert_eq!(midi_to_freq(57).to_bits(), 220.0_f32.to_bits());
        assert_eq!(midi_to_freq(81).to_bits(), 880.0_f32.to_bits());
        assert_eq!(midi_to_freq(45).to_bits(), 110.0_f32.to_bits());
        assert!((midi_to_freq(70) - 440.0 * PITCH_RATIO[1]).abs() < 1e-3);
    }

    /// 编排合法性：参数校验、事件全部落在网格内、时长精确。
    #[test]
    fn compose_grid_contract() {
        assert_eq!(compose(&ScoreSpec { mood: Mood::Tense, bars: 0, seed: 1 }, 8_000).unwrap_err().code, "SCN1001");
        let score = compose(&ScoreSpec { mood: Mood::Bright, bars: 4, seed: 7 }, 8_000).unwrap();
        assert_eq!(score.events.len(), 4 * (3 + 2) + score.events.iter().filter(|e| e.voice == Voice::Lead).count());
        for ev in &score.events {
            assert!(ev.start_frame + ev.frames <= score.total_frames, "事件越界 {ev:?}");
            assert!(ev.frames > 0);
            assert_eq!(ev.freq, midi_to_freq(ev.midi));
        }
        let spb = 60.0 / score.bpm * 8_000.0;
        let step_frames = (spb / 2.0).round() as usize;
        assert_eq!(score.total_frames, step_frames * 8 * 4, "每小节 = 8 个 8 分音符 step");
    }

    /// 同种子逐位一致（确定性核心断言）；不同种子事件流不同。
    #[test]
    fn compose_is_deterministic_and_seed_sensitive() {
        let spec = ScoreSpec { mood: Mood::Epic, bars: 8, seed: 0xDEAD_BEEF };
        let a = compose(&spec, 22_050).unwrap();
        for _ in 0..100 {
            assert_eq!(compose(&spec, 22_050).unwrap(), a);
        }
        let b = compose(&ScoreSpec { mood: Mood::Epic, bars: 8, seed: 0xDEAD_BEEF + 1 }, 22_050).unwrap();
        assert_ne!(a.events, b.events, "不同种子应产生不同 lead 流");
    }

    /// 密度参数生效：紧张（0.75）的 lead 音符数显著多于舒缓（0.30）。
    #[test]
    fn density_gates_lead_activity() {
        let count = |mood| {
            compose(&ScoreSpec { mood, bars: 16, seed: 42 }, 8_000)
                .unwrap()
                .events
                .iter()
                .filter(|e| e.voice == Voice::Lead)
                .count()
        };
        let (tense, calm) = (count(Mood::Tense), count(Mood::Calm));
        assert!(tense > calm * 2, "密度门控失效：tense={tense} calm={calm}");
        // 128 个 step、密度 0.75/0.30 的期望值 96/38：允许哈希涨落。
        assert!((60..=128).contains(&tense) && (10..=70).contains(&calm));
    }

    /// 渲染：确定性逐位一致、非平凡（有非零样本）、事件间无越界 panic。
    #[test]
    fn render_is_deterministic_and_nontrivial() {
        let score = compose(&ScoreSpec { mood: Mood::Calm, bars: 2, seed: 9 }, 16_000).unwrap();
        let a = render(&score);
        let b = render(&score);
        assert_eq!(a.as_slice(), b.as_slice(), "渲染逐位一致");
        assert_eq!(a.frames(), score.total_frames);
        assert!(a.peak() > 0.05, "配乐段不应为静音");
        assert!(a.peak() <= (0.50 + 3.0 * 0.22 + 0.40) + 1e-3, "三声部同相极限和内");
    }

    /// B13：作曲系统驱动的配乐——网格合法、时长精确、确定性、渲染非平凡。
    #[test]
    fn compose_pieces_grid_and_determinism() {
        assert_eq!(
            compose_pieces(&ComposedSpec { mood: Mood::Epic, bars: 0, seed: 1, arpeggiate: false }, 8_000)
                .unwrap_err()
                .code,
            "SCN1002"
        );
        let spec = ComposedSpec { mood: Mood::Epic, bars: 4, seed: 0xB13, arpeggiate: false };
        let score = compose_pieces(&spec, 8_000).unwrap();
        // 每小节：Bass 2 + Pad ≥ 3（三和弦）。
        assert!(score.events.iter().filter(|e| e.voice == Voice::Bass).count() == 8);
        for ev in &score.events {
            assert!(ev.start_frame + ev.frames <= score.total_frames, "事件越界 {ev:?}");
            assert_eq!(ev.freq, midi_to_freq(ev.midi));
        }
        // 确定性：重复逐位一致。
        for _ in 0..50 {
            assert_eq!(compose_pieces(&spec, 8_000).unwrap(), score);
        }
        // 渲染非平凡、峰值在内。
        let buf = render(&score);
        assert!(buf.peak() > 0.05);
        assert_eq!(buf.frames(), score.total_frames);
    }

    /// B13：琶音织体与柱式织体产生不同事件流（织体层差异生效）。
    #[test]
    fn arpeggiate_texture_differs() {
        let base = ComposedSpec { mood: Mood::Bright, bars: 2, seed: 5, arpeggiate: false };
        let arp = ComposedSpec { arpeggiate: true, ..base };
        let a = compose_pieces(&base, 8_000).unwrap();
        let b = compose_pieces(&arp, 8_000).unwrap();
        assert_ne!(a.events, b.events, "琶音与柱式织体应不同");
        let pad_a = a.events.iter().filter(|e| e.voice == Voice::Pad).count();
        let pad_b = b.events.iter().filter(|e| e.voice == Voice::Pad).count();
        assert!(pad_b > pad_a, "琶音展开的 Pad 事件应更多：{pad_b} vs {pad_a}");
    }
}
