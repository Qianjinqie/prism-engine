//! 琶音器（作曲系统·织体层； 第 3 层）。
//!
//! 把一个和弦按**模式 / 速率 / 八度范围**展开为确定性的音符事件序列。
//!
//! - **模式**：上行 / 下行 / 上下（往复）/ 随机（种子上行序哈希扰动）；
//! - **速率**：每音的 step 时值（以 step 为单位，1 = 八分音符粒度）；
//! - **八度范围**：`octaves` ∈ [1, 4]，第 k 遍叠 +12·k 半音；
//! - **确定性**：随机模式经 `prism-noise` PCG 哈希按 `hash(seed, step, slot)`
//!   派生（无全局 RNG，同种子同序列逐位一致）。

use crate::harmony::Chord;
use crate::theory::midi_to_freq;
use crate::AudioError;
use crate::AudioResult;

/// 琶音模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArpMode {
    /// 上行。
    Up,
    /// 下行。
    Down,
    /// 上下往复（含端点）。
    UpDown,
    /// 随机（种子扰动上行序）。
    Random,
}

/// 琶音器规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpSpec {
    /// 模式。
    pub mode: ArpMode,
    /// 每音 step 时值（≥ 1）。
    pub step: u32,
    /// 八度范围（1..=4）。
    pub octaves: u32,
    /// 每个八度内的重复音数（≥ 1；`note_count = 和弦成分数 × 该值`）。
    pub repeats: u32,
    /// 派生种子。
    pub seed: u64,
}

/// 琶音音符事件（step 域；`freq` 由 `midi_to_freq` 唯一派生）。
#[derive(Debug, Clone, PartialEq)]
pub struct ArpNote {
    /// 起始 step（相对琶音起点）。
    pub start_step: usize,
    /// 时值（step 数）。
    pub steps: usize,
    /// MIDI 音高。
    pub midi: i32,
    /// 频率。
    pub freq: f32,
}

/// 展开一个和弦为琶音序列（`step ≥ 1`、`octaves ∈ 1..=4`、`repeats ≥ 1`，
/// 否则 `ARP1001`）。
pub fn arpeggiate(chord: &Chord, spec: &ArpSpec) -> AudioResult<Vec<ArpNote>> {
    if spec.step < 1 || !(1..=4).contains(&spec.octaves) || spec.repeats < 1 {
        return Err(AudioError::new(
            "ARP1001",
            format!("arp::arpeggiate(step={}, octaves={}, repeats={})", spec.step, spec.octaves, spec.repeats),
            "琶音参数越界：step 需 ≥ 1、octaves 需 1..=4、repeats 需 ≥ 1",
            "修正 step / octaves / repeats（如 step 1 octaves 2 repeats 1）",
        ));
    }
    // 基础上行序：八度内循环展开和弦成分。
    let mut base: Vec<i32> = Vec::new();
    for k in 0..spec.octaves as i32 {
        for _ in 0..spec.repeats {
            for &m in &chord.midis {
                base.push(m + 12 * k);
            }
        }
    }
    // 按模式生成索引序。
    let order: Vec<i32> = match spec.mode {
        ArpMode::Up => base.clone(),
        ArpMode::Down => {
            let mut v = base.clone();
            v.reverse();
            v
        }
        ArpMode::UpDown => {
            // 上行后接去端点下行（往复），偶数成分时首尾各一次。
            let mut v = base.clone();
            let mut down = base.clone();
            down.reverse();
            if down.len() > 2 {
                v.extend_from_slice(&down[1..down.len() - 1]);
            }
            v
        }
        ArpMode::Random => {
            // 随机：对上行序做一次确定性 Fisher–Yates（PCG 哈希取模）。
            let mut v = base.clone();
            let mut n = v.len();
            while n > 1 {
                let r = prism_noise::pcg::hash2_f32(spec.seed, (n - 1) as u64, 77);
                let j = (r * n as f32) as usize % n;
                v.swap(n - 1, j);
                n -= 1;
            }
            v
        }
    };

    let notes: Vec<ArpNote> = order
        .into_iter()
        .enumerate()
        .map(|(i, midi)| ArpNote {
            start_step: i * spec.step as usize,
            steps: spec.step as usize,
            midi,
            freq: midi_to_freq(midi),
        })
        .collect();
    Ok(notes)
}

/// 琶音序列总 step 数（尾端 = 末音起点 + 时值；空序列为 0）。
pub fn total_steps(notes: &[ArpNote]) -> usize {
    notes.iter().map(|n| n.start_step + n.steps).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::{build_chord, ChordQuality};
    use crate::theory::{Key, ScaleKind};

    fn c_major_triad() -> Chord {
        let key = Key { kind: ScaleKind::Major, root_midi: 60 };
        build_chord(&key, 0, ChordQuality::Triad)
    }

    /// 上行：八度内循环展开，起始 step 等距递增。
    #[test]
    fn up_mode_layout() {
        let chord = c_major_triad(); // 60 64 67
        let spec = ArpSpec { mode: ArpMode::Up, step: 1, octaves: 2, repeats: 1, seed: 0 };
        let notes = arpeggiate(&chord, &spec).unwrap();
        assert_eq!(notes.len(), 6, "3 成分 × 2 八度");
        assert_eq!(
            notes.iter().map(|n| n.midi).collect::<Vec<_>>(),
            vec![60, 64, 67, 72, 76, 79]
        );
        for (i, n) in notes.iter().enumerate() {
            assert_eq!(n.start_step, i);
            assert_eq!(n.steps, 1);
            assert_eq!(n.freq, midi_to_freq(n.midi));
        }
        assert_eq!(total_steps(&notes), 6);
    }

    /// 下行 = 上行反序；上下往复含端点。
    #[test]
    fn down_and_updown_modes() {
        let chord = c_major_triad();
        let up = arpeggiate(&chord, &ArpSpec { mode: ArpMode::Up, step: 1, octaves: 1, repeats: 1, seed: 0 }).unwrap();
        let down = arpeggiate(&chord, &ArpSpec { mode: ArpMode::Down, step: 1, octaves: 1, repeats: 1, seed: 0 }).unwrap();
        assert_eq!(down.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![67, 64, 60]);
        let ud = arpeggiate(&chord, &ArpSpec { mode: ArpMode::UpDown, step: 1, octaves: 1, repeats: 1, seed: 0 }).unwrap();
        assert_eq!(ud.iter().map(|n| n.midi).collect::<Vec<_>>(), vec![60, 64, 67, 64]);
        assert_eq!(up.len(), 3);
    }

    /// 速率与重复：step 倍率与音符数。
    #[test]
    fn step_timing_and_repeats() {
        let chord = c_major_triad();
        let notes = arpeggiate(
            &chord,
            &ArpSpec { mode: ArpMode::Up, step: 2, octaves: 1, repeats: 2, seed: 0 },
        )
        .unwrap();
        assert_eq!(notes.len(), 6, "3 成分 × 2 重复");
        assert_eq!(notes[1].start_step, 2, "step 倍率生效");
        assert_eq!(notes[1].steps, 2);
        assert_eq!(total_steps(&notes), 12);
    }

    /// 随机模式确定性：同种子逐位一致、不同种子（大概率）不同，且为原成分排列。
    #[test]
    fn random_mode_is_deterministic_and_a_permutation() {
        let chord = c_major_triad();
        let spec = ArpSpec { mode: ArpMode::Random, step: 1, octaves: 2, repeats: 1, seed: 0x5EED };
        let a = arpeggiate(&chord, &spec).unwrap();
        for _ in 0..100 {
            assert_eq!(arpeggiate(&chord, &spec).unwrap(), a);
        }
        // 是上行序的排列（音高集合一致）。
        let mut got: Vec<i32> = a.iter().map(|n| n.midi).collect();
        let mut expect = vec![60, 64, 67, 72, 76, 79];
        got.sort_unstable();
        expect.sort_unstable();
        assert_eq!(got, expect, "随机模式应为成分排列");
        let b = arpeggiate(&chord, &ArpSpec { seed: 0x5EEE, ..spec }).unwrap();
        assert_ne!(a.iter().map(|n| n.midi).collect::<Vec<_>>(), b.iter().map(|n| n.midi).collect::<Vec<_>>());
    }

    /// 非法参数：step 0 / octaves 0 或 5 / repeats 0 → ARP1001。
    #[test]
    fn arp_rejects_bad_params() {
        let chord = c_major_triad();
        let bad = |step, octaves, repeats| ArpSpec { mode: ArpMode::Up, step, octaves, repeats, seed: 0 };
        assert_eq!(arpeggiate(&chord, &bad(0, 2, 1)).unwrap_err().code, "ARP1001");
        assert_eq!(arpeggiate(&chord, &bad(1, 0, 1)).unwrap_err().code, "ARP1001");
        assert_eq!(arpeggiate(&chord, &bad(1, 5, 1)).unwrap_err().code, "ARP1001");
        assert_eq!(arpeggiate(&chord, &bad(1, 2, 0)).unwrap_err().code, "ARP1001");
    }
}
