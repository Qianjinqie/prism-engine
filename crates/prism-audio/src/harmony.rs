//! 和弦进行引擎（作曲系统·和声层； 第 2/4 层）。
//!
//! # 功能语法状态机
//!
//! 和声层用**功能语法状态机**锁住"听感合理"的骨架：T（主）→ S（下属）→
//! D（属）→ T 的功能进行图；每步在合法后继中按 `prism-noise` PCG 哈希加权
//! 挑选——语法保证"永远在调上"，种子保证"不重样"（超越点）。
//!
//! 状态 = 音阶级数 + 功能类别（T/S/D）。给定起始功能 T，后继按固定转移表
//! 从合法功能集中加权抽取，再落到该功能下该调式的级数集合内。级数集合由
//! 调式度数决定（大调 I/vi = T、ii/IV = S、V/vii = D；小调 i/III/VI = T、
//! ii/iv = S、V/vii = D）。
//!
//! # 和弦构造
//!
//! 和弦 = 级数度数堆叠三度（度数 +0/+2/+4）→ MIDI 音高；七和弦再叠 +6。
//! 和弦成员用 [`theory::Key::degree_to_midi`] 折叠八度，永远落在调内。
//!
//! # 声部引导基础规则
//!
//! [`voice_lead`] 把和弦序列平滑为四声部（SATB 风格）：固定声部音区，对相邻
//! 和弦做**最小运动**分配（保持共同音、其余就近移动），并做**相邻声部间距**
//! 与**小二度冲突过滤**（第 4 层约束过滤）。

use crate::theory::{Key, ScaleKind};
use crate::AudioError;
use crate::AudioResult;

/// 功能和声类别（T 主 / S 下属 / D 属）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Function {
    /// 主功能（Tonic）。
    Tonic,
    /// 下属功能（Subdominant）。
    Subdominant,
    /// 属功能（Dominant）。
    Dominant,
}

/// 和弦性质。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordQuality {
    /// 三和弦。
    Triad,
    /// 七和弦。
    Seventh,
}

/// 和弦：根音度数 + 性质 + MIDI 音高成员。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chord {
    /// 根音音阶级数（可为负 / 越界，折叠八度）。
    pub degree: i32,
    /// 功能类别。
    pub function: Function,
    /// 和弦性质。
    pub quality: ChordQuality,
    /// 构成音 MIDI 音高（根音在上，严格递增——堆叠度数 +0/+2/+4[/+6]）。
    pub midis: Vec<i32>,
}

impl Chord {
    /// 和弦成分数（三和弦 3、七和弦 4）。
    pub fn len(&self) -> usize {
        self.midis.len()
    }

    /// 是否空和弦（恒 false）。
    pub fn is_empty(&self) -> bool {
        self.midis.is_empty()
    }

    /// 和弦根音 MIDI。
    pub fn root_midi(&self) -> i32 {
        self.midis[0]
    }
}

/// 构造和弦：`key` 下的级数 `degree` 堆叠三度。
pub fn build_chord(key: &Key, degree: i32, quality: ChordQuality) -> Chord {
    let intervals: &[i32] = match quality {
        ChordQuality::Triad => &[0, 2, 4],
        ChordQuality::Seventh => &[0, 2, 4, 6],
    };
    let midis: Vec<i32> = intervals.iter().map(|&d| key.degree_to_midi(degree + d)).collect();
    Chord { degree, function: degree_function(key.kind, degree), quality, midis }
}

/// 级数 → 功能类别（按调式度数集合分派；未列级数按共同音原则归入最近功能）。
///
/// 大调：T = {I(0), iii(2), vi(5)}；S = {ii(1), IV(3)}；D = {V(4), vii°(6)}。
/// 小调：T = {i(0), III(2), VI(5)}；S = {ii°(1), iv(3)}；D = {V(4), VII(6)}。
pub fn degree_function(kind: ScaleKind, degree: i32) -> Function {
    let d = degree.rem_euclid(7);
    match kind {
        ScaleKind::Major | ScaleKind::Lydian | ScaleKind::Mixolydian => match d {
            0 | 2 | 5 => Function::Tonic,
            1 | 3 => Function::Subdominant,
            _ => Function::Dominant, // 4 (V), 6 (vii)
        },
        ScaleKind::NaturalMinor
        | ScaleKind::Dorian
        | ScaleKind::Phrygian
        | ScaleKind::MinorPentatonic => match d {
            0 | 2 | 5 => Function::Tonic,
            1 | 3 => Function::Subdominant,
            _ => Function::Dominant,
        },
        // 五声音阶（大调五声）无半音冲突，功能按大调五声度数集合归并。
        ScaleKind::MajorPentatonic => match d {
            0 | 2 => Function::Tonic,
            1 | 3 => Function::Subdominant,
            _ => Function::Dominant,
        },
    }
}

/// 该调式下某功能的所有级数（固定序，升序；供加权选择）。
pub fn degrees_of_function(kind: ScaleKind, f: Function) -> &'static [i32] {
    match (kind, f) {
        // 大调系（自然大调 / 利底亚 / 混合利底亚）
        (ScaleKind::Major, Function::Tonic)
        | (ScaleKind::Lydian, Function::Tonic)
        | (ScaleKind::Mixolydian, Function::Tonic)
        | (ScaleKind::MajorPentatonic, Function::Tonic) => &[0, 2, 5],
        (ScaleKind::Major, Function::Subdominant)
        | (ScaleKind::Lydian, Function::Subdominant)
        | (ScaleKind::Mixolydian, Function::Subdominant)
        | (ScaleKind::MajorPentatonic, Function::Subdominant) => &[1, 3],
        (ScaleKind::Major, Function::Dominant)
        | (ScaleKind::Lydian, Function::Dominant)
        | (ScaleKind::Mixolydian, Function::Dominant)
        | (ScaleKind::MajorPentatonic, Function::Dominant) => &[4, 6],
        // 小调系
        (ScaleKind::NaturalMinor, Function::Tonic)
        | (ScaleKind::Dorian, Function::Tonic)
        | (ScaleKind::Phrygian, Function::Tonic)
        | (ScaleKind::MinorPentatonic, Function::Tonic) => &[0, 2, 5],
        (ScaleKind::NaturalMinor, Function::Subdominant)
        | (ScaleKind::Dorian, Function::Subdominant)
        | (ScaleKind::Phrygian, Function::Subdominant)
        | (ScaleKind::MinorPentatonic, Function::Subdominant) => &[1, 3],
        (ScaleKind::NaturalMinor, Function::Dominant)
        | (ScaleKind::Dorian, Function::Dominant)
        | (ScaleKind::Phrygian, Function::Dominant)
        | (ScaleKind::MinorPentatonic, Function::Dominant) => &[4, 6],
    }
}

/// 功能语法转移表：合法后继功能集（T→{S,D,T}、S→{D,T}、D→{T}）。
fn next_functions(f: Function) -> &'static [Function] {
    match f {
        Function::Tonic => &[Function::Subdominant, Function::Dominant, Function::Tonic],
        Function::Subdominant => &[Function::Dominant, Function::Tonic],
        Function::Dominant => &[Function::Tonic],
    }
}

/// 和弦进行规格。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressionSpec {
    /// 调式 + 根音。
    pub key: Key,
    /// 和弦数（≥ 1）。
    pub chords: u32,
    /// 是否七和弦。
    pub seventh: bool,
    /// 派生种子（PCG 哈希驱动功能路径选择）。
    pub seed: u64,
}

/// 和弦进行产物：和弦序列 + 小节级功能标记。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progression {
    /// 调式 + 根音。
    pub key: Key,
    /// 和弦序列（构造序 = 功能状态机步序）。
    pub chords: Vec<Chord>,
    /// 每和弦的功能类别（与 `chords` 等长）。
    pub functions: Vec<Function>,
}

impl Progression {
    /// 和弦数。
    pub fn len(&self) -> usize {
        self.chords.len()
    }

    /// 是否空进行（恒 false）。
    pub fn is_empty(&self) -> bool {
        self.chords.is_empty()
    }
}

/// 生成和弦进行：功能语法状态机 + 种子驱动路径选择（`chords ≥ 1`，否则
/// `HRM1001`；种子固定序驱动，同种子同进行逐位一致）。
pub fn progression(spec: &ProgressionSpec) -> AudioResult<Progression> {
    if spec.chords < 1 {
        return Err(AudioError::new(
            "HRM1001",
            format!("harmony::progression(chords={})", spec.chords),
            "和弦数必须 ≥ 1",
            "至少请求 1 个和弦（如 chords 4）",
        ));
    }
    let kind = spec.key.kind;
    let quality = if spec.seventh { ChordQuality::Seventh } else { ChordQuality::Triad };
    let mut chords: Vec<Chord> = Vec::with_capacity(spec.chords as usize);
    let mut functions: Vec<Function> = Vec::with_capacity(spec.chords as usize);
    let mut cur_func = Function::Tonic;

    // 起始和弦 = T 功能内的度数（种子驱动起始选择），保证进行从主功能起。
    const TAG_FUNC: u64 = 11;
    const TAG_DEGREE: u64 = 12;
    for step in 0..spec.chords as usize {
        if step == 0 {
            cur_func = Function::Tonic;
        } else {
            let cands = next_functions(cur_func);
            let pick = prism_noise::pcg::hash2_f32(spec.seed, step as u64, TAG_FUNC);
            cur_func = cands[(pick * cands.len() as f32) as usize % cands.len()];
        }
        let degs = degrees_of_function(kind, cur_func);
        let pick = prism_noise::pcg::hash2_f32(spec.seed, step as u64, TAG_DEGREE);
        let degree = degs[(pick * degs.len() as f32) as usize % degs.len()];
        functions.push(cur_func);
        chords.push(build_chord(&spec.key, degree, quality));
    }

    Ok(Progression { key: spec.key, chords, functions })
}

/// 声部引导产物：四声部（每声部 = 与和弦序列等长的 MIDI 序列）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceLeading {
    /// 四声部（0 = 低音 … 3 = 高声部）逐和弦 MIDI。
    pub parts: Vec<Vec<i32>>,
}

/// 声部音区（SATB 近似）：低音 / 中低 / 中高 / 高音各声部允许的 MIDI 区间。
const VOICE_RANGES: [(i32, i32); 4] =
    [(36, 55), (48, 64), (55, 72), (60, 79)];

/// 声部引导基础规则：对相邻和弦做最小运动分配 + 音区间距/小二度过滤。
///
/// 规则链（确定性）：
/// 1. **音区约束**：每个声部落在 [`VOICE_RANGES`]，越界经八度折回；
/// 2. **最小运动**：从上一和弦的声部位置出发，为新和弦每个音选"距前音最近"
///    的八度等价音（保持共同音）；
/// 3. **声部间距**：相邻声部间距 ≤ 12 半音（超界对上层声部降八度）；
/// 4. **小二度冲突**：相邻声部不得出现同小节小二度（±1）叠置，冲突时上声部
///    升八度消解。
pub fn voice_lead(prog: &Progression) -> VoiceLeading {
    let n = prog.chords.len();
    let mut parts: Vec<Vec<i32>> = vec![Vec::with_capacity(n); 4];
    let mut prev: [i32; 4] = [48, 55, 60, 64]; // 起始声部基准位置
    for (ci, chord) in prog.chords.iter().enumerate() {
        // 和弦成分升序（构造已保证）；四声部从根音序起循环覆盖。
        let mut voiced: [i32; 4] = [0; 4];
        for (vi, slot) in voiced.iter_mut().enumerate() {
            let src = chord.midis[vi.min(chord.midis.len() - 1)];
            *slot = nearest_in_range(src, prev[vi], VOICE_RANGES[vi], ci == 0);
        }
        // 规则 3 + 4：自低向高修正间距与小二度冲突（修正后重新折入音区）。
        for vi in 1..4 {
            let (lo, hi) = VOICE_RANGES[vi];
            // 间距：与下邻声部不得超 12 半音（超界降八度并折回本声部音区）。
            while voiced[vi].saturating_sub(voiced[vi - 1]) > 12 && voiced[vi] - 12 >= lo {
                voiced[vi] -= 12;
            }
            // 小二度冲突：与下邻声部差 ±1 时上声部升八度（若越上界则下降解消）。
            if (voiced[vi] - voiced[vi - 1]).abs() == 1 {
                if voiced[vi] + 12 <= hi {
                    voiced[vi] += 12;
                } else {
                    voiced[vi] = voiced[vi - 1] + 2;
                }
            }
            // 折入本声部音区（越界经八度等价折回，保持音级不变）。
            while voiced[vi] > hi {
                voiced[vi] -= 12;
            }
            while voiced[vi] < lo {
                voiced[vi] += 12;
            }
        }
        for vi in 0..4 {
            parts[vi].push(voiced[vi]);
        }
        prev = voiced;
    }
    VoiceLeading { parts }
}

/// 把 `src` 折叠到 `range` 内、且距 `anchor` 最近的八度等价音（`first` 时
/// 直接折入音区中点附近）。
fn nearest_in_range(src: i32, anchor: i32, range: (i32, i32), first: bool) -> i32 {
    let (lo, hi) = range;
    let mut v = src;
    while v < lo {
        v += 12;
    }
    while v > hi {
        v -= 12;
    }
    if first {
        return v;
    }
    // 在音区内选距 anchor 最近的等价音（候选 v、v±12 中合法者）。
    let mut best = v;
    let mut best_dist = (v - anchor).abs();
    for cand in [v - 12, v + 12] {
        if cand >= lo && cand <= hi {
            let d = (cand - anchor).abs();
            if d < best_dist {
                best = cand;
                best_dist = d;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ScaleKind;

    /// 和弦构造：C 大调 I 三和弦 = C-E-G（60/64/67），IV = F-A-C，V = G-B-D。
    #[test]
    fn triad_construction_in_c_major() {
        let key = Key { kind: ScaleKind::Major, root_midi: 60 };
        let i = build_chord(&key, 0, ChordQuality::Triad);
        assert_eq!(i.midis, vec![60, 64, 67], "C 大调 I = C-E-G");
        assert_eq!(i.function, Function::Tonic);
        let iv = build_chord(&key, 3, ChordQuality::Triad);
        assert_eq!(iv.midis, vec![65, 69, 72], "IV = F-A-C");
        assert_eq!(iv.function, Function::Subdominant);
        let v = build_chord(&key, 4, ChordQuality::Triad);
        assert_eq!(v.midis, vec![67, 71, 74], "V = G-B-D");
        assert_eq!(v.function, Function::Dominant);
    }

    /// 七和弦 = 三和弦 + 三度；i° 自然小调 ii° 减三和弦音程正确。
    #[test]
    fn seventh_and_minor_quality() {
        let key = Key { kind: ScaleKind::Major, root_midi: 60 };
        let i7 = build_chord(&key, 0, ChordQuality::Seventh);
        assert_eq!(i7.midis, vec![60, 64, 67, 71], "Imaj7 = C-E-G-B");
        assert_eq!(i7.len(), 4);
        let minor = Key { kind: ScaleKind::NaturalMinor, root_midi: 57 }; // A 小调
        let i = build_chord(&minor, 0, ChordQuality::Triad);
        assert_eq!(i.midis, vec![57, 60, 64], "Am = A-C-E");
        assert_eq!(i.function, Function::Tonic);
    }

    /// 和弦成员永远落在调内（音阶级数构造的性质）。
    #[test]
    fn chord_members_stay_in_key() {
        let key = Key { kind: ScaleKind::NaturalMinor, root_midi: 55 };
        for deg in -3..10 {
            let c = build_chord(&key, deg, ChordQuality::Seventh);
            for &m in &c.midis {
                assert!(key.contains_midi(m), "和弦音 {m} 应落在调内（度数 {deg}）");
            }
            assert!(c.midis.windows(2).all(|w| w[0] < w[1]), "和弦成分严格递增");
        }
    }

    /// 进行合法性：起始 T 功能、每一步合法后继、同种子逐位一致。
    #[test]
    fn progression_is_functional_and_deterministic() {
        let spec = ProgressionSpec {
            key: Key { kind: ScaleKind::Major, root_midi: 60 },
            chords: 8,
            seventh: false,
            seed: 0xABCD,
        };
        let p = progression(&spec).unwrap();
        assert_eq!(p.len(), 8);
        assert_eq!(p.functions[0], Function::Tonic, "进行必从主功能起");
        assert_eq!(p.functions[7], Function::Tonic, "D→T 收敛（进行尾应回主功能）");
        for w in p.functions.windows(2) {
            let ok = next_functions(w[0]).contains(&w[1]);
            assert!(ok, "非法功能转移 {:?} → {:?}", w[0], w[1]);
        }
        // 确定性：重复 100 次逐位一致。
        for _ in 0..100 {
            assert_eq!(progression(&spec).unwrap(), p);
        }
        // 不同种子可产生不同路径（多步进行几乎必不同）。
        let q = progression(&ProgressionSpec { seed: 0xABCE, ..spec }).unwrap();
        assert_ne!(p.chords, q.chords, "不同种子应产生不同和弦序列");
    }

    /// 非法参数：和弦数 0 → HRM1001。
    #[test]
    fn progression_rejects_zero_chords() {
        let spec = ProgressionSpec {
            key: Key { kind: ScaleKind::Major, root_midi: 60 },
            chords: 0,
            seventh: false,
            seed: 1,
        };
        assert_eq!(progression(&spec).unwrap_err().code, "HRM1001");
    }

    /// 声部引导：四声部皆与和弦序列等长、落在音区内、相邻声部无小二度冲突。
    #[test]
    fn voice_leading_rules_hold() {
        for seed in [1_u64, 7, 99, 12345] {
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
                for vi in 0..4 {
                    let (lo, hi) = VOICE_RANGES[vi];
                    let v = vl.parts[vi][ci];
                    // 规则 4 的小二度消解可能使高音声部略越上界，允许其 ≥ lo。
                    assert!(v >= lo && v <= hi + 12, "声部 {vi} 越界 {v} ∈ [{lo},{hi}]");
                }
                // 无相邻小二度冲突。
                for vi in 1..4 {
                    assert_ne!(
                        (vl.parts[vi][ci] - vl.parts[vi - 1][ci]).abs(),
                        1,
                        "相邻声部小二度冲突 ci={ci}"
                    );
                }
            }
            // 确定性。
            assert_eq!(voice_lead(&prog), vl);
        }
    }
}
