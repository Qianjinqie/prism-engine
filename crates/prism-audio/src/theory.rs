//! 音阶库与音高换算（B13 作曲系统·调式层；白皮书 §22 第 1 层）。
//!
//! # 音高换算（D-29 禁 libm）
//!
//! 十二平均律音高比 = `2^(k/12)`（k ∈ 0..12）以固定常量表给出；八度比 =
//! 2 的整数幂**连乘**（禁 `pow`）。任何 MIDI 音高 `m` 的频率
//! `440 · 2^((m−69)/12)` 由"半音查表 × 八度连乘"唯一确定，A4 及全八度锚点
//! 逐位精确（与 [`crate::score::midi_to_freq`] 同式同源）。
//!
//! # 音阶库
//!
//! 音阶 = 相对根音的半音增量表（度数 0 = 根音，严格递增、皆 < 12）。库含
//! 大调 / 自然小调 / 大调五声 / 小调五声 / 四中古调式（多利亚 / 弗里几亚 /
//! 利底亚 / 混合利底亚）。度数可为任意整数，越界经 `div_euclid`/`rem_euclid`
//! 折叠八度——正向越界升八度、负度降八度（与 `score::degree_to_midi` 同规）。

use crate::AudioError;
use crate::AudioResult;

/// 十二平均律半音比值表：`SEMITONE_RATIO[k] = 2^(k/12)`（k = 0..12，固定常量，
/// 白皮书 §23 音高比查表口径，取代运行期 `pow`）。
///
/// D-30 说明：k=6 的 `1.414_213_6` 是 `2^0.5` 的**正确 f32 舍入**
/// （0x3FB504F4），比 `std::f32::consts::SQRT_2` 高 1 ulp。替换会让纯五度
/// 偏移约 1.19e-7，故显式 `allow` 而非改值。详见 `score::PITCH_RATIO` 同注。
#[allow(clippy::approx_constant)] // 精确值，非近似；见 D-30
pub const SEMITONE_RATIO: [f32; 12] = [
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

/// 音阶类型（调式库）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleKind {
    /// 自然大调（Ionian）：0 2 4 5 7 9 11。
    Major,
    /// 自然小调（Aeolian）：0 2 3 5 7 8 10。
    NaturalMinor,
    /// 大调五声：0 2 4 7 9。
    MajorPentatonic,
    /// 小调五声：0 3 5 7 10。
    MinorPentatonic,
    /// 多利亚：0 2 3 5 7 9 10。
    Dorian,
    /// 弗里几亚：0 1 3 5 7 8 10。
    Phrygian,
    /// 利底亚：0 2 4 6 7 9 11。
    Lydian,
    /// 混合利底亚：0 2 4 5 7 9 10。
    Mixolydian,
}

/// 音阶 = 半音增量表。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    /// 音阶类型。
    pub kind: ScaleKind,
    /// 相对根音的半音增量（度数 0 = 根音，严格递增、皆 < 12）。
    pub degrees: &'static [u8],
}

impl Scale {
    /// 度数个数（音阶规模，如大调 = 7、五声 = 5）。
    pub fn len(&self) -> usize {
        self.degrees.len()
    }

    /// 是否为空音阶（恒 false——库内音阶非空；`clippy::len_without_is_empty` 兼容）。
    pub fn is_empty(&self) -> bool {
        self.degrees.is_empty()
    }

    /// 度数 → 相对根音的半音增量（任意整数度数，折叠八度）。
    pub fn semi(&self, degree: i32) -> i32 {
        let n = self.len() as i32;
        12 * degree.div_euclid(n) + self.degrees[degree.rem_euclid(n) as usize] as i32
    }

    /// 度数 → MIDI 音高：`root_midi + semi(degree)`（折叠八度）。
    pub fn degree_to_midi(&self, root_midi: i32, degree: i32) -> i32 {
        root_midi + self.semi(degree)
    }
}

/// 自然大调半音表。
const MAJOR: &[u8] = &[0, 2, 4, 5, 7, 9, 11];
/// 自然小调半音表。
const NATURAL_MINOR: &[u8] = &[0, 2, 3, 5, 7, 8, 10];
/// 大调五声半音表。
const MAJOR_PENTATONIC: &[u8] = &[0, 2, 4, 7, 9];
/// 小调五声半音表。
const MINOR_PENTATONIC: &[u8] = &[0, 3, 5, 7, 10];
/// 多利亚半音表。
const DORIAN: &[u8] = &[0, 2, 3, 5, 7, 9, 10];
/// 弗里几亚半音表。
const PHRYGIAN: &[u8] = &[0, 1, 3, 5, 7, 8, 10];
/// 利底亚半音表。
const LYDIAN: &[u8] = &[0, 2, 4, 6, 7, 9, 11];
/// 混合利底亚半音表。
const MIXOLYDIAN: &[u8] = &[0, 2, 4, 5, 7, 9, 10];

/// 取音阶（纯函数，编译期常量分派）。
pub fn scale(kind: ScaleKind) -> Scale {
    let degrees: &'static [u8] = match kind {
        ScaleKind::Major => MAJOR,
        ScaleKind::NaturalMinor => NATURAL_MINOR,
        ScaleKind::MajorPentatonic => MAJOR_PENTATONIC,
        ScaleKind::MinorPentatonic => MINOR_PENTATONIC,
        ScaleKind::Dorian => DORIAN,
        ScaleKind::Phrygian => PHRYGIAN,
        ScaleKind::Lydian => LYDIAN,
        ScaleKind::Mixolydian => MIXOLYDIAN,
    };
    Scale { kind, degrees }
}

/// 调式库全表（测试与遍历用）。
pub const ALL_SCALES: [ScaleKind; 8] = [
    ScaleKind::Major,
    ScaleKind::NaturalMinor,
    ScaleKind::MajorPentatonic,
    ScaleKind::MinorPentatonic,
    ScaleKind::Dorian,
    ScaleKind::Phrygian,
    ScaleKind::Lydian,
    ScaleKind::Mixolydian,
];

/// MIDI 音高 → 频率（Hz）：`440 · 2^((m−69)/12)`，半音查表 + 八度 2 的整数幂
/// 连乘（禁 `pow`；A4 锚点全八度逐位精确）。
pub fn midi_to_freq(midi: i32) -> f32 {
    let offset = midi - 69;
    let k = offset.rem_euclid(12) as usize;
    let octaves = offset.div_euclid(12);
    let mut ratio = SEMITONE_RATIO[k];
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

/// 音级 + 八度 → MIDI 音高：`(octave + 1) · 12 + pitch_class`（科学音高记号，
/// C4 = MIDI 60、A4 = 69）。`pitch_class` 与 `octave` 任意整数经 `rem_euclid`
/// 归一（如 pitch_class = 12 → 上移一个八度）。
pub fn pitch_to_midi(octave: i32, pitch_class: i32) -> i32 {
    let pc = pitch_class.rem_euclid(12);
    let extra = pitch_class.div_euclid(12);
    (octave + 1 + extra) * 12 + pc
}

/// 音阶度数规格：调式 + 根音。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    /// 调式音阶。
    pub kind: ScaleKind,
    /// 根音 MIDI 音高。
    pub root_midi: i32,
}

impl Key {
    /// 度数 → MIDI 音高（折叠八度）。
    pub fn degree_to_midi(&self, degree: i32) -> i32 {
        scale(self.kind).degree_to_midi(self.root_midi, degree)
    }

    /// 度数 → 频率（= [`midi_to_freq`]`(degree_to_midi(degree))`）。
    pub fn degree_to_freq(&self, degree: i32) -> f32 {
        midi_to_freq(self.degree_to_midi(degree))
    }

    /// MIDI 音高是否落在本调音阶内（`pitch_class` 相对根音 → 查表）。
    pub fn contains_midi(&self, midi: i32) -> bool {
        let rel = (midi - self.root_midi).rem_euclid(12);
        scale(self.kind).degrees.iter().any(|&d| d as i32 == rel)
    }
}

/// 校验：根音 MIDI 落在合法区间 `[0, 127]`（否则 `THE1001`）。
pub fn validate_root(root_midi: i32) -> AudioResult<()> {
    if !(0..=127).contains(&root_midi) {
        return Err(AudioError::new(
            "THE1001",
            format!("theory::validate_root({root_midi})"),
            "根音 MIDI 音高超出 [0, 127]",
            "使用合法 MIDI 音高（如 60 = C4）；过低/过高时先折回八度",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 库内音阶皆规范：非空、含根音 0、严格递增、皆 < 12。
    #[test]
    fn scale_library_is_well_formed() {
        for kind in ALL_SCALES {
            let s = scale(kind);
            assert!(!s.is_empty(), "{kind:?} 音阶不应为空");
            assert_eq!(s.degrees[0], 0, "{kind:?} 必含根音");
            assert!(s.degrees.iter().all(|&d| d < 12), "{kind:?} 半音增量须 < 12");
            assert!(
                s.degrees.windows(2).all(|w| w[0] < w[1]),
                "{kind:?} 音阶须严格递增"
            );
        }
        assert_eq!(scale(ScaleKind::Major).len(), 7);
        assert_eq!(scale(ScaleKind::MajorPentatonic).len(), 5);
        assert_eq!(scale(ScaleKind::MinorPentatonic).len(), 5);
    }

    /// 度数折叠：正负越界与同度数同八度等价；semi(0) = 0、semi(len) = 12。
    #[test]
    fn degree_folding_octaves() {
        let s = scale(ScaleKind::Major);
        assert_eq!(s.semi(0), 0);
        assert_eq!(s.semi(7), 12, "度数 7 = 上八度根音");
        assert_eq!(s.semi(8), 14, "度数 8 = 上八度第二级");
        assert_eq!(s.semi(-7), -12);
        assert_eq!(s.semi(-1), -1, "度数 −1 = 下方导音（大调 11 − 12 = −1）");
        assert_eq!(s.semi(14), s.semi(7) + s.semi(7));
    }

    /// 音高换算：A4 及全八度锚点逐位精确（八度比 = 2 的整数幂连乘）。
    #[test]
    fn midi_to_freq_octaves_are_exact() {
        assert_eq!(midi_to_freq(69).to_bits(), 440.0_f32.to_bits());
        assert_eq!(midi_to_freq(57).to_bits(), 220.0_f32.to_bits());
        assert_eq!(midi_to_freq(81).to_bits(), 880.0_f32.to_bits());
        assert_eq!(midi_to_freq(45).to_bits(), 110.0_f32.to_bits());
        assert_eq!(midi_to_freq(117).to_bits(), 7040.0_f32.to_bits());
        // 半音比表锚点：C5/ C#5 比值 = SEMITONE_RATIO[1]。
        let c = midi_to_freq(72);
        let cs = midi_to_freq(73);
        assert!((cs / c - SEMITONE_RATIO[1]).abs() < 1e-6);
    }

    /// pitch_to_midi：科学音高记号锚点 + 音级越界折八度。
    #[test]
    fn pitch_class_notation_anchors() {
        assert_eq!(pitch_to_midi(4, 0), 60, "C4 = 60");
        assert_eq!(pitch_to_midi(4, 9), 69, "A4 = 69");
        assert_eq!(pitch_to_midi(3, 0), 48, "C3 = 48");
        assert_eq!(pitch_to_midi(4, 12), 72, "音级 12 → 上八度 C5");
        assert_eq!(pitch_to_midi(4, -1), 59, "音级 −1 → 下方 B3");
    }

    /// Key：度数→MIDI 与 contains_midi 音阶内判定。
    #[test]
    fn key_degree_and_membership() {
        let key = Key { kind: ScaleKind::Major, root_midi: 60 }; // C 大调
        assert_eq!(key.degree_to_midi(0), 60);
        assert_eq!(key.degree_to_midi(4), 67, "G");
        assert_eq!(key.degree_to_midi(7), 72, "上八度 C");
        assert!(key.contains_midi(60) && key.contains_midi(64) && key.contains_midi(71));
        assert!(!key.contains_midi(61), "C# 不在 C 大调");
        assert_eq!(key.degree_to_midi(9), 76, "度数 9 = 上八度 E");
        assert_eq!(key.degree_to_freq(9).to_bits(), midi_to_freq(76).to_bits());
    }

    /// 根音校验：合法区间通过、越界报 THE1001。
    #[test]
    fn root_validation() {
        assert!(validate_root(60).is_ok());
        assert!(validate_root(0).is_ok());
        assert!(validate_root(127).is_ok());
        assert_eq!(validate_root(128).unwrap_err().code, "THE1001");
        assert_eq!(validate_root(-1).unwrap_err().code, "THE1001");
    }
}
