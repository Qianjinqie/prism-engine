//! Step 鼓机（作曲系统·节奏层； 第 3 层节奏网格）。
//!
//! 结构：**步进网格**（`steps` 步 × `voices` 鼓件）+ **音色映射**（鼓件 →
//! 合成器音色：波形 / ADSR / 基频 / 增益）。每格一位（0 = 静音，1 = 触发），
//! 命中即产出一个 `step` 时值的打击事件。渲染经 [`crate::synth::render_note`]
//! 合成，接 synth 振荡器族（"节奏 step 网格（密度参数控制休止）"）。
//!
//! 打击乐基频映射（相对 [D-29] 禁 libm 的确定性口径，固定常量）：
//! kick 低频扫落（近似单音三角 55 Hz）、snare 中频噪声近似（方波 200 Hz）、
//! hat 高频（方波 3200 Hz——低于 8 kHz 采样率的 Nyquist，保证 PolyBLEP 相位
//! 步长 < 0.5）。

use crate::buffer::PcmBuffer;
use crate::synth::presets::Timbre;
use crate::synth::{render_note, Adsr, Waveform};
use crate::AudioError;
use crate::AudioResult;

/// 鼓件（固定枚举序 = 渲染遍历序 = 确定性）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drum {
    /// 底鼓。
    Kick,
    /// 军鼓。
    Snare,
    /// 踩镲。
    Hat,
}

/// 鼓件数（网格行数）。
pub const DRUM_COUNT: usize = 3;

/// 鼓件音色：波形 + ADSR + 基频 + 增益。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrumTimbre {
    /// 合成波形。
    pub waveform: Waveform,
    /// ADSR 包络。
    pub envelope: Adsr,
    /// 基频（Hz）。
    pub freq: f32,
    /// 增益。
    pub gain: f32,
}

/// 取鼓件默认音色（纯函数，编译期常量表）。
pub fn drum_timbre(drum: Drum) -> DrumTimbre {
    match drum {
        Drum::Kick => DrumTimbre {
            waveform: Waveform::Triangle,
            envelope: Adsr { attack: 0.001, decay: 0.09, sustain: 0.0, release: 0.02 },
            freq: 55.0,
            gain: 0.9,
        },
        Drum::Snare => DrumTimbre {
            waveform: Waveform::Square,
            envelope: Adsr { attack: 0.001, decay: 0.06, sustain: 0.0, release: 0.03 },
            freq: 200.0,
            gain: 0.5,
        },
        Drum::Hat => DrumTimbre {
            waveform: Waveform::Square,
            envelope: Adsr { attack: 0.0005, decay: 0.02, sustain: 0.0, release: 0.01 },
            freq: 3_200.0,
            gain: 0.28,
        },
    }
}

/// 鼓机规格：步数 + 各鼓件步进网格（位集）。
#[derive(Debug, Clone, PartialEq)]
pub struct DrumSpec {
    /// 步数（≥ 1）。
    pub steps: usize,
    /// 各鼓件网格：`grid[drum][step]` ∈ {0, 1}，`drum` 按 [`Drum`] 枚举序。
    pub grid: Vec<Vec<u8>>,
}

impl DrumSpec {
    /// 构造 4/4 默认节奏型（kick on 1&3、snare on 2&4、hat on 每 8 分）。
    pub fn four_on_floor(steps: usize) -> AudioResult<Self> {
        if steps < 1 {
            return Err(AudioError::new(
                "DRM1001",
                format!("drums::DrumSpec::four_on_floor({steps})"),
                "步数必须 ≥ 1",
                "给出 ≥ 1 的步数（如 16）",
            ));
        }
        let mut grid: Vec<Vec<u8>> = vec![vec![0; steps]; DRUM_COUNT];
        for s in 0..steps {
            if s % 4 == 0 {
                grid[Drum::Kick as usize][s] = 1; // 1 / 3 拍
            }
            if s % 4 == 2 {
                grid[Drum::Snare as usize][s] = 1; // 2 / 4 拍
            }
            if s % 2 == 0 {
                grid[Drum::Hat as usize][s] = 1; // 每 8 分
            }
        }
        Ok(Self { steps, grid })
    }

    /// 单格是否触发（越界视为静音）。
    pub fn hits(&self, drum: Drum, step: usize) -> bool {
        self.grid
            .get(drum as usize)
            .and_then(|row| row.get(step))
            .map_or(false, |&v| v != 0)
    }
}

/// 校验网格规格（步数 ≥ 1、`grid` 行数 = [`DRUM_COUNT`]、每行长度 = steps，
/// 否则 `DRM1001`）；合法后回填缺行/补足列。
fn normalize(spec: &mut DrumSpec) -> AudioResult<()> {
    if spec.steps < 1 {
        return Err(AudioError::new(
            "DRM1001",
            format!("drums::normalize(steps={})", spec.steps),
            "步数必须 ≥ 1",
            "给出 ≥ 1 的步数（如 16）",
        ));
    }
    if spec.grid.len() > DRUM_COUNT {
        return Err(AudioError::new(
            "DRM1001",
            format!("drums::normalize(grid 行数 = {})", spec.grid.len()),
            format!("鼓件网格行数不得超过 {DRUM_COUNT}（kick/snare/hat）"),
            "按 Drum 枚举序提供至多 3 行网格",
        ));
    }
    // 行数不足补零行；每行长度按 steps 归一（短补 0、长截断）。
    while spec.grid.len() < DRUM_COUNT {
        spec.grid.push(vec![0; spec.steps]);
    }
    for row in spec.grid.iter_mut() {
        row.resize(spec.steps, 0);
        for v in row.iter_mut() {
            *v = if *v != 0 { 1 } else { 0 };
        }
    }
    Ok(())
}

/// 单个打击事件（帧域）。
#[derive(Debug, Clone, PartialEq)]
pub struct DrumHit {
    /// 起始帧。
    pub start_frame: usize,
    /// 时值（帧数）。
    pub frames: usize,
    /// 鼓件。
    pub drum: Drum,
    /// 触发步（网格列）。
    pub step: usize,
}

/// 鼓机产物：打击事件表 + 总帧数。
#[derive(Debug, Clone, PartialEq)]
pub struct DrumTrack {
    /// 打击事件（构造序 = 步 × 鼓件固定遍历）。
    pub hits: Vec<DrumHit>,
    /// 采样率。
    pub sample_rate: u32,
    /// 总帧数（= steps × step_frames）。
    pub total_frames: usize,
    /// 网格规格（渲染/断言输入）。
    pub spec: DrumSpec,
}

/// 编排鼓机：`bpm` 的 16 分音符为 1 步（`step_frames = round(60·sr/(bpm·4))`）。
pub fn sequence(spec: &DrumSpec, bpm: f32, sample_rate: u32) -> AudioResult<DrumTrack> {
    if bpm <= 0.0 || sample_rate < 1 {
        return Err(AudioError::new(
            "DRM1002",
            format!("drums::sequence(bpm={bpm}, sr={sample_rate})"),
            "BPM 与采样率必须为正",
            "给出正 BPM（如 120）与正采样率（如 44100）",
        ));
    }
    let mut spec = spec.clone();
    normalize(&mut spec)?;
    let step_frames = ((60.0_f64 / bpm as f64 / 4.0) * sample_rate as f64).round() as usize;
    let step_frames = step_frames.max(1);
    let mut hits: Vec<DrumHit> = Vec::new();
    // 固定遍历序：步 → 鼓件枚举序（与网格存储序一致）。
    for step in 0..spec.steps {
        for drum_idx in 0..DRUM_COUNT {
            let drum = match drum_idx {
                0 => Drum::Kick,
                1 => Drum::Snare,
                _ => Drum::Hat,
            };
            if spec.hits(drum, step) {
                hits.push(DrumHit {
                    start_frame: step * step_frames,
                    frames: step_frames,
                    drum,
                    step,
                });
            }
        }
    }
    Ok(DrumTrack {
        hits,
        sample_rate,
        total_frames: spec.steps * step_frames,
        spec,
    })
}

/// 渲染鼓机为 PCM（逐事件经 synth 合成、按事件序累加；纯函数）。
pub fn render(track: &DrumTrack) -> PcmBuffer {
    let mut data = vec![0.0_f32; track.total_frames];
    for hit in &track.hits {
        let t = drum_timbre(hit.drum);
        let note = render_note(t.waveform, t.freq, &t.envelope, t.gain, hit.frames, track.sample_rate);
        for (i, v) in note.into_iter().enumerate() {
            data[hit.start_frame + i] += v; // 事件序累加 = 固定序
        }
    }
    PcmBuffer::from_interleaved(track.sample_rate, 1, data)
        .expect("单声道鼓机渲染缓冲长度恒合法")
}

/// 鼓件 → 音色摘要（供 DSL codegen 白名单与渲染共用同一映射）。
pub fn timbre_of(drum: Drum) -> Timbre {
    let t = drum_timbre(drum);
    Timbre { waveform: t.waveform, envelope: t.envelope }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4/4 默认型：kick 落 0/4/8/12、snare 落 2/6/10/14、hat 落偶数步。
    #[test]
    fn four_on_floor_grid_pattern() {
        let spec = DrumSpec::four_on_floor(16).unwrap();
        for s in 0..16 {
            assert_eq!(spec.hits(Drum::Kick, s), s % 4 == 0, "kick step {s}");
            assert_eq!(spec.hits(Drum::Snare, s), s % 4 == 2, "snare step {s}");
            assert_eq!(spec.hits(Drum::Hat, s), s % 2 == 0, "hat step {s}");
        }
        assert!(!spec.hits(Drum::Kick, 16), "越界静音");
    }

    /// 步进编排：事件数 = 命中格数、步对齐、总帧数精确、固定遍历序。
    #[test]
    fn sequence_step_alignment_and_order() {
        let spec = DrumSpec::four_on_floor(16).unwrap();
        let track = sequence(&spec, 120.0, 8_000).unwrap();
        // 命中数：kick 4 + snare 4 + hat 8 = 16。
        assert_eq!(track.hits.len(), 16);
        // step_frames @120bpm 16 分音符 8kHz = round(0.125 × 8000) = 1000。
        assert_eq!(track.total_frames, 16 * 1000);
        for h in &track.hits {
            assert_eq!(h.start_frame, h.step * 1000);
            assert_eq!(h.frames, 1000);
        }
        // 固定序：先按 step 再按 drum 枚举序（kick<snare<hat）。
        let keys: Vec<(usize, usize)> = track.hits.iter().map(|h| (h.step, h.drum as usize)).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "事件序应为 (step, drum) 升序");
    }

    /// 渲染：非平凡、确定性逐位一致、不削波。
    #[test]
    fn render_is_deterministic_and_nontrivial() {
        let spec = DrumSpec::four_on_floor(8).unwrap();
        let track = sequence(&spec, 140.0, 16_000).unwrap();
        let a = render(&track);
        assert_eq!(a.as_slice(), render(&track).as_slice(), "逐位一致");
        assert_eq!(a.frames(), track.total_frames);
        assert!(a.peak() > 0.05, "鼓组不应静音");
        // 三鼓件增益和上限（kick 0.9 + snare 0.5 + hat 0.28）；同相极限不削波。
        assert!(a.peak() <= 0.9 + 0.5 + 0.28 + 1e-3, "三鼓件同相极限和内");
    }

    /// 网格归一：行数补齐、列数补足、非零归一为 1；非法步数报 DRM1001。
    #[test]
    fn grid_normalization() {
        let mut spec = DrumSpec { steps: 4, grid: vec![vec![1, 0]] };
        normalize(&mut spec).unwrap();
        assert_eq!(spec.grid.len(), DRUM_COUNT);
        assert_eq!(spec.grid[0], vec![1, 0, 0, 0], "列补齐");
        assert!(spec.grid[1].iter().all(|&v| v == 0), "缺行补零");
        let mut bad = DrumSpec { steps: 0, grid: vec![] };
        assert_eq!(normalize(&mut bad).unwrap_err().code, "DRM1001");
        assert_eq!(
            sequence(&DrumSpec::four_on_floor(4).unwrap(), 0.0, 8_000).unwrap_err().code,
            "DRM1002"
        );
    }
}
