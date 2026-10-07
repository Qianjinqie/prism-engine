//! SFZ 文本格式解析：region/opcode 子集 → [`Region`] 列表。
//!
//! SFZ 是采样器的文本映射格式（David Viens / rgc:audio 规范）。本实现覆盖采样
//! 回放所需的**子集**：`<region>` / `<global>` / `<group>` 头 + opcode 键值对。
//!
//! 支持的 opcode：
//! - `sample=` 采样（此处为内嵌库键名或外部路径）
//! - `lokey=`/`hikey=`/`key=`/`pitch_keycenter=` 键区与根音
//! - `pitch_keytrack=` 音高跟踪（%）
//! - `loop_start=`/`loop_end=` 循环区间（可选）
//! - `ampeg_attack=`/`ampeg_decay=`/`ampeg_sustain=`/`ampeg_release=` 包络
//! - `volume=` 增益（dB）
//!
//! 解析为纯函数（无 I/O），错误三段式（`AU1005`）。

use crate::synth::Adsr;
use crate::{AudioError, AudioResult};

/// 一个 SFZ 采样区（region）：键区 + 根音 + 循环 + 包络。
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// 采样键名/路径（`sample=` 原值）。
    pub sample: String,
    /// 键区下界（MIDI，含）。
    pub lokey: u8,
    /// 键区上界（MIDI，含）。
    pub hikey: u8,
    /// 根音采样点（MIDI；此键按下播放原始速率）。
    pub pitch_keycenter: i32,
    /// 音高跟踪百分比（100 = 完全跟踪；0 = 不随键变）。
    pub pitch_keytrack: f32,
    /// 循环起点（样本帧；None = 无循环）。
    pub loop_start: Option<u32>,
    /// 循环终点（样本帧；None = 无循环 / 到末尾）。
    pub loop_end: Option<u32>,
    /// 包络（秒 / 电平）。
    pub amp_envelope: Adsr,
    /// 增益（dB）。
    pub volume_db: f32,
}

impl Default for Region {
    fn default() -> Self {
        Self {
            sample: String::new(),
            lokey: 0,
            hikey: 127,
            pitch_keycenter: 60,
            pitch_keytrack: 100.0,
            loop_start: None,
            loop_end: None,
            amp_envelope: Adsr { attack: 0.0, decay: 0.0, sustain: 1.0, release: 0.0 },
            volume_db: 0.0,
        }
    }
}

impl Region {
    /// 该 region 是否覆盖 MIDI 键 `key`。
    pub fn covers(&self, key: u8) -> bool {
        key >= self.lokey && key <= self.hikey
    }

    /// 按下键 `key` 时的播放速率比（`2^((key−center)·track/1200)`）。
    ///
    /// 用 12-TET 半音查表 + 八度整数幂连乘（D-29 禁 pow；帧格点精确）。
    pub fn playback_ratio(&self, key: u8) -> f32 {
        let semis = (key as i32 - self.pitch_keycenter) as f32 * (self.pitch_keytrack / 100.0);
        semitone_ratio(semis)
    }
}

/// 半音（可为小数）→ 频率比 `2^(semis/12)`。
///
/// 整数半音走 [`crate::score`] 同款查表（八度 = 2 的整数幂，IEEE 精确）；小数
/// 部分线性插值于相邻格点（音高比查表口径）。
///
/// D-30 说明：表内 k=6 的 `1.414_213_6` 是 `2^0.5` 的**正确 f32 舍入**
/// （0x3FB504F4），比 `std::f32::consts::SQRT_2` 高 1 ulp。替换会让纯五度
/// 偏移约 1.19e-7，故显式 `allow` 而非改值。详见 `theory::SEMITONE_RATIO` 同注。
#[allow(clippy::approx_constant)] // 精确值，非近似；见 D-30
pub fn semitone_ratio(semis: f32) -> f32 {
    let whole = semis.round();
    let frac = semis - whole;
    let mut n = whole as i32;
    let octaves = n.div_euclid(12);
    n = n.rem_euclid(12);
    let table = [
        1.0_f32,
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
        2.0,
    ];
    let base = table[n as usize];
    let next = table[(n as usize + 1).min(12)];
    let grid = base + (next - base) * frac; // 相邻格点线性插值
    // 八度整数幂连乘（2 的整数次幂 IEEE 精确）。
    let mut r = grid;
    let mut o = octaves;
    while o > 0 {
        r *= 2.0;
        o -= 1;
    }
    while o < 0 {
        r *= 0.5;
        o += 1;
    }
    r
}

/// 解析 SFZ 文本为 region 列表（纯函数；无 `sample` 的 region 被跳过）。
pub fn parse_sfz(text: &str) -> AudioResult<Vec<Region>> {
    let mut regions: Vec<Region> = Vec::new();
    let mut current = Region::default();
    let mut has_region = false;
    // global/group 作为默认值继承（本子集简化为「当前 region 初值来自 global」）。
    let mut global = Region::default();
    let mut in_global = false;

    let flush = |cur: &mut Region, has: &mut bool, out: &mut Vec<Region>| {
        // 仅凭 has_region 不足以判定有效：无 `sample` 的 region 按文档跳过
        // （缺陷 4，文档 115 行「无 sample 的 region 被跳过」）。
        if *has && !cur.sample.is_empty() {
            out.push(std::mem::take(cur));
        }
        *has = false;
    };

    for raw_line in text.lines() {
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        for token in line.split_whitespace() {
            if token.starts_with('<') {
                let head = token.trim_matches(|c| c == '<' || c == '>').to_ascii_lowercase();
                match head.as_str() {
                    "region" => {
                        flush(&mut current, &mut has_region, &mut regions);
                        current = global.clone();
                        has_region = true;
                        in_global = false;
                    }
                    "global" => {
                        flush(&mut current, &mut has_region, &mut regions);
                        global = Region::default();
                        in_global = true;
                    }
                    "group" => {
                        flush(&mut current, &mut has_region, &mut regions);
                        current = global.clone();
                        in_global = false;
                    }
                    other => {
                        return Err(AudioError::new(
                            "AU1005",
                            format!("parse_sfz <{other}>"),
                            format!("不支持的 SFZ 头：<{other}>"),
                            "仅支持 <region> / <global> / <group>",
                        ));
                    }
                }
                continue;
            }
            let Some((key, val)) = token.split_once('=') else { continue };
            let target = if in_global { &mut global } else { &mut current };
            apply_opcode(target, key.trim(), val.trim());
        }
    }
    flush(&mut current, &mut has_region, &mut regions);
    Ok(regions)
}

/// 把单个 opcode 应用到 region（未知 opcode 静默忽略——SFZ 规范允许扩展）。
fn apply_opcode(r: &mut Region, key: &str, val: &str) {
    match key {
        "sample" => r.sample = val.to_string(),
        "lokey" => r.lokey = val.parse().unwrap_or(0),
        "hikey" => r.hikey = val.parse().unwrap_or(127),
        "key" => {
            let k: u8 = val.parse().unwrap_or(60);
            r.lokey = k;
            r.hikey = k;
            r.pitch_keycenter = k as i32;
        }
        "pitch_keycenter" => r.pitch_keycenter = val.parse().unwrap_or(60),
        "pitch_keytrack" => r.pitch_keytrack = val.parse().unwrap_or(100.0),
        "loop_start" => r.loop_start = val.parse().ok(),
        "loop_end" => r.loop_end = val.parse().ok(),
        "ampeg_attack" => r.amp_envelope.attack = val.parse().unwrap_or(0.0),
        "ampeg_decay" => r.amp_envelope.decay = val.parse().unwrap_or(0.0),
        "ampeg_sustain" => r.amp_envelope.sustain = val.parse().unwrap_or(1.0),
        "ampeg_release" => r.amp_envelope.release = val.parse().unwrap_or(0.0),
        "volume" => r.volume_db = val.parse().unwrap_or(0.0),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
// 测试音色
<global>
ampeg_attack=0.01
ampeg_release=0.2

<region>
sample=piano
lokey=36 hikey=48 pitch_keycenter=40
volume=-3

<region>
sample=violin
key=57
"#;

    /// 解析两 region、global 继承、键区与根音正确。
    #[test]
    fn parse_regions_and_globals() {
        let regions = parse_sfz(SAMPLE).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].sample, "piano");
        assert_eq!((regions[0].lokey, regions[0].hikey), (36, 48));
        assert_eq!(regions[0].pitch_keycenter, 40);
        assert!((regions[0].volume_db + 3.0).abs() < 1e-6);
        // global 包络继承。
        assert!((regions[0].amp_envelope.attack - 0.01).abs() < 1e-6);
        assert!((regions[1].amp_envelope.release - 0.2).abs() < 1e-6);
        // key= 单键。
        assert_eq!((regions[1].lokey, regions[1].hikey), (57, 57));
        assert_eq!(regions[1].pitch_keycenter, 57);
    }

    /// 键区覆盖判断。
    #[test]
    fn region_covers_key_range() {
        let regions = parse_sfz(SAMPLE).unwrap();
        assert!(regions[0].covers(40));
        assert!(!regions[0].covers(60));
        assert!(regions[1].covers(57) && !regions[1].covers(56));
    }

    /// 播放速率比：根音 = 1.0，上八度 = 2.0（精确），下八度 = 0.5。
    #[test]
    fn playback_ratio_keytrack() {
        let r = Region { pitch_keycenter: 60, pitch_keytrack: 100.0, ..Default::default() };
        assert!((r.playback_ratio(60) - 1.0).abs() < 1e-5);
        assert!((r.playback_ratio(72) - 2.0).abs() < 1e-5, "上八度应为 2");
        assert!((r.playback_ratio(48) - 0.5).abs() < 1e-5, "下八度应为 0.5");
        // keytrack=0 → 恒定 1.0。
        let flat = Region { pitch_keycenter: 60, pitch_keytrack: 0.0, ..Default::default() };
        assert!((flat.playback_ratio(84) - 1.0).abs() < 1e-6);
    }

    /// 未知头报错三段式。
    #[test]
    fn unknown_header_errors() {
        let err = parse_sfz("<curve> sample=x").unwrap_err();
        assert_eq!(err.code, "AU1005");
        assert!(!err.fix.is_empty());
    }

    /// semitone_ratio 与十二平均律一致（对照 score 的 midi_to_freq 口径）。
    #[test]
    fn semitone_ratio_matches_12tet() {
        assert!((semitone_ratio(0.0) - 1.0).abs() < 1e-6);
        assert!((semitone_ratio(12.0) - 2.0).abs() < 1e-5);
        assert!((semitone_ratio(7.0) - 1.498_307_1).abs() < 1e-5, "纯五度");
    }
}
