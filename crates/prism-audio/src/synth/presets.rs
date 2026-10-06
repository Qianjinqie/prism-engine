//! 4 情绪配乐预设（B6）：紧张 / 舒缓 / 明快 / 恢弘。
//!
//! 情绪 = 参数包（路线图 B6 行："参数化：速度/调式/密度/音色"），不是随机
//! 种子的副产品——模板差异而非随机差异（白皮书 §22 情绪参数口径，B13 作曲
//! 系统在此基础上升级为语法层差异）。
//!
//! 调式为音级（semitone）音阶表：`scale[degree]` = 相对根音的半音数。
//! 进行（progression）为小节级调式度数序列（4 小节循环）。

use crate::synth::{Adsr, Waveform};

/// 配乐情绪。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    /// 紧张：弗里几亚、高密度、锯齿。
    Tense,
    /// 舒缓：大调五声、低密度、正弦长音。
    Calm,
    /// 明快：自然大调、中高密度、三角波琶音。
    Bright,
    /// 恢弘：自然小调、中密度、锯齿宽铺底。
    Epic,
}

/// 音色 = 波形 + ADSR。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timbre {
    /// 波形。
    pub waveform: Waveform,
    /// ADSR 包络（秒 / 电平）。
    pub envelope: Adsr,
}

/// 情绪预设参数包：速度 / 调式 / 密度 / 音色 / 和弦进行 / 音区。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoodPreset {
    /// 速度（BPM）。
    pub bpm: f32,
    /// 调式音阶（半音表，度数 0 = 根音）。
    pub scale: &'static [u8],
    /// 密度 ∈ [0, 1]：lead 声部 step 触发概率（休止概率 = 1 − density）。
    pub density: f32,
    /// 低音声部音色。
    pub bass: Timbre,
    /// 铺底声部音色。
    pub pad: Timbre,
    /// 主旋律声部音色。
    pub lead: Timbre,
    /// 和弦进行（小节级调式度数，4 小节循环）。
    pub progression: &'static [u8],
    /// 根音 MIDI 音高（低音在其下 12 半音，主旋律在其上 `lead_octaves`×12）。
    pub root_midi: i32,
    /// 主旋律相对根音的八度上移（半音 = 12×该值）。
    pub lead_octaves: i32,
}

/// 紧张：BPM 138，弗里几亚（♭2 张力），密度 0.75，短促锯齿 + 低音律动。
const TENSE: MoodPreset = MoodPreset {
    bpm: 138.0,
    scale: &[0, 1, 3, 5, 7, 8, 10], // Phrygian
    density: 0.75,
    bass: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.005, decay: 0.10, sustain: 0.4, release: 0.08 },
    },
    pad: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.25, decay: 0.30, sustain: 0.55, release: 0.40 },
    },
    lead: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.004, decay: 0.08, sustain: 0.35, release: 0.10 },
    },
    progression: &[0, 1, 0, 4], // i – ♭II – i – v（弗里几亚标志性和声张力）
    root_midi: 45,              // A2
    lead_octaves: 2,
};

/// 舒缓：BPM 72，大调五声（无半音冲突），密度 0.30，正弦长音慢包络。
const CALM: MoodPreset = MoodPreset {
    bpm: 72.0,
    scale: &[0, 2, 4, 7, 9], // Major pentatonic
    density: 0.30,
    bass: Timbre {
        waveform: Waveform::Sine,
        envelope: Adsr { attack: 0.05, decay: 0.30, sustain: 0.5, release: 0.60 },
    },
    pad: Timbre {
        waveform: Waveform::Sine,
        envelope: Adsr { attack: 0.80, decay: 0.40, sustain: 0.7, release: 1.20 },
    },
    lead: Timbre {
        waveform: Waveform::Sine,
        envelope: Adsr { attack: 0.06, decay: 0.25, sustain: 0.55, release: 0.50 },
    },
    progression: &[0, 3, 1, 4],
    root_midi: 48, // C3
    lead_octaves: 2,
};

/// 明快：BPM 126，自然大调，密度 0.60，三角波琶音 + I–IV–V–I。
const BRIGHT: MoodPreset = MoodPreset {
    bpm: 126.0,
    scale: &[0, 2, 4, 5, 7, 9, 11], // Major
    density: 0.60,
    bass: Timbre {
        waveform: Waveform::Triangle,
        envelope: Adsr { attack: 0.004, decay: 0.15, sustain: 0.45, release: 0.10 },
    },
    pad: Timbre {
        waveform: Waveform::Triangle,
        envelope: Adsr { attack: 0.15, decay: 0.25, sustain: 0.6, release: 0.30 },
    },
    lead: Timbre {
        waveform: Waveform::Triangle,
        envelope: Adsr { attack: 0.003, decay: 0.06, sustain: 0.4, release: 0.08 },
    },
    progression: &[0, 3, 4, 0], // I – IV – V – I
    root_midi: 48,              // C3
    lead_octaves: 2,
};

/// 恢弘：BPM 92，自然小调，密度 0.50，锯齿宽铺底 + i–VI–iv–v。
const EPIC: MoodPreset = MoodPreset {
    bpm: 92.0,
    scale: &[0, 2, 3, 5, 7, 8, 10], // Natural minor
    density: 0.50,
    bass: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.01, decay: 0.20, sustain: 0.55, release: 0.20 },
    },
    pad: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.40, decay: 0.30, sustain: 0.65, release: 0.80 },
    },
    lead: Timbre {
        waveform: Waveform::Saw,
        envelope: Adsr { attack: 0.02, decay: 0.12, sustain: 0.5, release: 0.25 },
    },
    progression: &[0, 5, 3, 4], // i – VI – iv – v
    root_midi: 43,              // G2
    lead_octaves: 2,
};

/// 取情绪预设（纯函数，编译期常量分派）。
pub fn preset(mood: Mood) -> MoodPreset {
    match mood {
        Mood::Tense => TENSE,
        Mood::Calm => CALM,
        Mood::Bright => BRIGHT,
        Mood::Epic => EPIC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 四情绪参数完整性：BPM/密度在界内、音阶严格递增且含根音、进行非空。
    #[test]
    fn presets_are_well_formed() {
        for mood in [Mood::Tense, Mood::Calm, Mood::Bright, Mood::Epic] {
            let p = preset(mood);
            assert!((60.0..=180.0).contains(&p.bpm), "{mood:?} BPM 越界");
            assert!((0.0..1.0).contains(&p.density), "{mood:?} 密度越界");
            assert_eq!(p.scale[0], 0, "{mood:?} 音阶必含根音");
            assert!(p.scale.windows(2).all(|w| w[0] < w[1] && w[1] < 12), "{mood:?} 音阶需严格递增");
            assert!(!p.progression.is_empty());
            assert!(p.progression.iter().all(|&d| (d as usize) < p.scale.len() + 4),
                "{mood:?} 进行度数应可折叠到音阶内");
            assert!((0..=2).contains(&p.lead_octaves));
        }
    }

    /// 四情绪参数互异（情绪 = 模板差异，不允许两个情绪退化成同参数）。
    #[test]
    fn moods_differ_in_parameters() {
        let moods = [Mood::Tense, Mood::Calm, Mood::Bright, Mood::Epic];
        for i in 0..moods.len() {
            for j in (i + 1)..moods.len() {
                let (a, b) = (preset(moods[i]), preset(moods[j]));
                assert!(
                    a.bpm != b.bpm || a.scale != b.scale || a.density != b.density
                        || a.lead != b.lead,
                    "{:?} 与 {:?} 参数包不应全同", moods[i], moods[j]
                );
            }
        }
    }
}
