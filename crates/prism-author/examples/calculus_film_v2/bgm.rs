//! F4：全片 BGM 实时合成——`prism-audio` 旧 API 四声部，120BPM × 45 小节。
//!
//! 情绪曲线按 PRD §7（神秘开场 → 冷静建立 → 紧张推进 → 明亮高潮 →
//! 累积溶解 → 亲密沉静 → 辉煌收尾），以**平滑段落增益包络**落地：
//! 每个声部的增益由 [`voice_gain_at`] 纯函数按时刻给出（相邻小节
//! smoothstep 过渡、音符内逐样本连续，无阶梯 click），全链无时钟 /
//! 随机 / 内部可变状态（同参数逐位一致）。
//!
//! 声部（轨序 pad → arp → bass → drums，混音增益沿用 v1 已验证值）：
//! - **pad**：每小节根 / 三 / 五长音（Calm 预设音色），收尾大绽放两小节
//!   加高八度根音；
//! - **arp**：八分 / 十六分音符轮转和弦音（高两个八度）；开场稀疏单音
//!   标记 S1 三个符号入场时刻；推进段末小节走下行琶音；沉静段叠平行五度
//!   双旋律（73.0s 起汇合同音）；收尾段走高空十六分「流星」音；
//! - **bass**：推进段起的行走低音（根音低八度）；
//! - **drums**：高潮段进入的四四拍 kick + 反拍 hat，60.0s 骤停；
//!   47.5s（S4 重音时刻）加 snare + hat 重击。
//!
//! 场景音乐提示（PRD §7「场景切换音乐提示」）以**一次性事件**落位：
//! 1.0 / 2.0 / 3.0s 稀疏 arp 单音、61.0s F 大调和弦、84.0 / 84.75s
//! 清脆泛音——全部由 (小节, 步) 纯函数推导。
//!
//! v3 音乐系统重制（对成片 BGM「听起来像死人」尸检结论的工程回应；
//! 全部丰富度在**本文件内基于旧 API 手工实现**，不随引擎演进而破）：
//! - **立体声**：四声部仍按现状混成单声道母带（`mix_tracks` 不动），
//!   母带经 [`widen_to_stereo`] 展宽——L / R 不同微偏移（Haas 9ms
//!   量级差）+ 左右不同混响抽头，交织样本经
//!   `PcmBuffer::from_interleaved` 输出；`frames()` 保持 3_969_000
//!   （采样帧口径不变，AVI 复用器按采样帧切片，block_align =
//!   channels×2，立体声不破坏该约束）；
//! - **手工混响**：Schroeder 结构（4 并联梳状 + 2 串联全通）在
//!   [`StereoReverb`] 内手工实现，湿声 ≈0.22，尾音自然截断于 90s；
//! - **音色丰富化**（仅 Sine / Triangle）：微失谐多副本 + 2/3 次谐波
//!   低增益叠合（[`render_rich`]），pad 另叠 ±0.5% 慢速拍音绒毛；
//! - **段落动态**：小节级阶梯增益改平滑插值；混响后接分段软膝限幅
//!   （[`soft_knee`]：|x|≤0.7 恒等——中频段动态保留，峰值渐近 ≤1）；
//! - **人性化**：对 (bar, step, voice) 做整数哈希（splitmix64 终结化，
//!   禁 rand crate）——音符 onset ±12ms、增益 ±8%；鼓组反拍八分
//!   （16 分网格 s%4==2）后移 6% 个十六分（经典 shuffle；kicks 与
//!   47.5s 重音保持网格）。
//!
//! 母带淡入 1.5s / 淡出 1.2s 施加于**展宽后的立体声母带**（逐帧包络，
//! 两声道端点样本恒 0）。若按 v2 次序先淡变再展宽，微偏移通道（R 后移
//! 12ms）的末样本将是淡变前母带副本（≈6e-4），违反「淡变端点样本恒 0」
//! 不变量；淡变时长 / 线性形状 / 首尾 0.05s 窗幅度均不变。长度不变量：
//! 帧数 = `round(duration × sample_rate)`，淡变端点样本恒 0。
//!
//! 本文件与 T02 / T03 引擎改动**零依赖**（仅依赖 `prism-audio` 旧
//! API：`drums` / `mixer` / `synth` / `presets` / `theory` /
//! `PcmBuffer`）——架构 §6.4-2 为此设有机械证明门禁：对本文件做引擎
//! 原语名的 grep 计数必须恒为 0（即本文件不得出现任何引擎新增原语
//! 字样）；所有丰富度均在文件内基于旧 API 手工实现。
//!
//! [`voice_gain_at`]: section_gains_at
//! [`widen_to_stereo`]: widen_to_stereo
//! [`StereoReverb`]: StereoReverb
//! [`render_rich`]: render_rich
//! [`soft_knee`]: soft_knee

use prism_audio::drums::{self, DrumSpec};
use prism_audio::mixer::{mix_tracks, MixTrack};
use prism_audio::synth::presets::{preset, Mood, Timbre};
use prism_audio::synth::{render_note, Adsr, Waveform};
use prism_audio::theory::midi_to_freq;
use prism_audio::PcmBuffer;

/// BGM 速度（BPM）：120 ⇒ 每小节 2.0s，45 小节精确闭合 90.000s。
pub const BGM_BPM: f32 = 120.0;
/// BGM 采样率（44100——44100 % 30 = 0，满足 AVI 按帧切音频 chunk
/// 的整除要求：1470 采样/帧；立体声下 block_align = 2×2 字节，
/// 复用器按采样帧切片，帧约束不变）。
pub const BGM_SR: u32 = 44_100;
/// 全片总长（秒）。与 `common::TOTAL_DURATION_SEC` 的一致性由单测钉住
/// （本文件自持常量以保持独立可编译）。
pub const BGM_DURATION_SEC: f32 = 90.0;
/// 母带淡入时长（秒）。
pub const MASTER_FADE_IN_SEC: f32 = 1.5;
/// 母带淡出时长（秒）。
pub const MASTER_FADE_OUT_SEC: f32 = 1.2;
/// 输出声道数：立体声（母带展宽后交织输出；采样帧数不变）。
pub const BGM_CHANNELS: u16 = 2;

/// 混响湿声增益（Schroeder 尾音在湿声中的占比）。
const REVERB_WET: f32 = 0.22;
/// 立体声微偏移（Haas 展宽）：L 3ms / R 12ms（相差 9ms，8–15ms 量级）。
const HAAS_OFFSET_L_SEC: f32 = 0.003;
/// 立体声微偏移（Haas 展宽）：见 [`HAAS_OFFSET_L_SEC`]。
const HAAS_OFFSET_R_SEC: f32 = 0.012;
/// 软膝限幅的线性区上界（|x| ≤ 该值恒等，中频段动态保留）。
const SOFT_KNEE: f32 = 0.7;
/// 人性化 onset 抖动幅度（秒；±12ms）。
const HUMAN_ONSET_SEC: f32 = 0.012;
/// 人性化增益抖动幅度（比例；±8%）。
const HUMAN_GAIN: f32 = 0.08;
/// 鼓组 swing：反拍八分（16 分网格 s%4==2）后移比例（6% 个十六分）。
const SWING_SHIFT_RATIO: f32 = 0.06;
/// swing 移动副本的槽头余弦淡入时长（秒；消除重落位 click，保打击峰值）。
const SWING_EDGE_SEC: f32 = 0.0005;

/// 人性化哈希的声部维度（voice；(bar, step, voice) 三元组定键）。
const VOICE_PAD: u64 = 0;
/// 人性化哈希的声部维度：arp 主旋律。
const VOICE_ARP: u64 = 1;
/// 人性化哈希的声部维度：arp 平行五度副旋律。
const VOICE_ARP2: u64 = 2;
/// 人性化哈希的声部维度：bass。
const VOICE_BASS: u64 = 3;
/// 人性化哈希的声部维度：F 大调和弦提示。
const VOICE_CHORD: u64 = 4;
/// 人性化哈希的声部维度：清脆泛音提示。
const VOICE_BELL: u64 = 5;
/// 人性化哈希的声部维度：S1 稀疏单音（+ 事件序号）。
const VOICE_SINGLE: u64 = 6;

/// 一个和弦的声部音（MIDI）：[根音, 三音, 五音, 高八度根音]。
#[derive(Clone, Copy)]
struct Chord {
    /// MIDI 音高表（根 / 三 / 五 / 高八度根）。
    notes: [i32; 4],
}

/// Am–F–C–G（A 小调 vi–IV–I–V）每小节一和弦循环（v1 已验证编排）。
/// 音高按十二平均律 MIDI 推算（与 v1 同值）：
/// - Am（A+C+E）：A2=45 / C4=60 / E4=64 / A4=69；
/// - F （F+A+C）：F2=53 / A3=57 / C4=60 / F4=65；
/// - C （C+E+G）：C3=48 / E4=64 / G4=67 / C5=72；
/// - G （G+B+D）：G2=43 / B3=59 / D4=62 / G4=67。
const CHORDS: [Chord; 4] = [
    Chord { notes: [45, 60, 64, 69] },
    Chord { notes: [53, 57, 60, 65] },
    Chord { notes: [48, 64, 67, 72] },
    Chord { notes: [43, 59, 62, 67] },
];

/// 小节级四声部增益（PRD §7 情绪曲线的工程落地）。
struct SectionGains {
    /// pad 声部增益。
    pad: f32,
    /// arp 声部增益。
    arp: f32,
    /// bass 声部增益。
    bass: f32,
    /// drums 声部增益（>0 = 该小节鼓组激活；鼓组包络经网格激活实现）。
    drums: f32,
}

/// 段落增益包络（纯函数；小节索引 0 起，bar 0..44 ↔ 全局 0..90s）。
///
/// | 小节（0 起） | 全局 | 场景 | 编排 |
/// |---|---|---|---|
/// | 0–3 | 0–8s | S1 | pad drone + 稀疏单音，无鼓无低音 |
/// | 4–8 | 8–18s | S2 | arp 八分进入，pad 渐强 |
/// | 9–16 | 18–34s | S3 | bass 行走 + arp 加密（16 分），末小节下行琶音 |
/// | 17–24 | 34–50s | S4 | 鼓进入，全声部 |
/// | 25–29 | 50–60s | S5 前半 | 维持高潮密度 |
/// | 30–32 | 60–66s | S5 后半 | 鼓骤停、arp 转八分（「溶解」） |
/// | 33–39 | 66–80s | S6 | 鼓 / 低音撤，pad + 柔 arp 双旋律 |
/// | 40–41 | 80–84s | S7 前半 | pad 大绽放 + 高空流星音 |
/// | 42–43 | 84–88s | S7 中段 | 衰减，仅 pad |
/// | 44 | 88–90s | S7 落版 | 仅氛围 pad（母带淡出兜底） |
fn section_gains(bar: usize) -> SectionGains {
    match bar {
        0..=3 => SectionGains { pad: 1.00, arp: 0.00, bass: 0.00, drums: 0.00 },
        4..=8 => SectionGains {
            pad: 0.75 + 0.25 * (bar - 4) as f32 / 4.0, // Pad 渐强
            arp: 0.85,
            bass: 0.00,
            drums: 0.00,
        },
        9..=16 => SectionGains { pad: 0.90, arp: 1.00, bass: 0.90, drums: 0.00 },
        17..=24 => SectionGains { pad: 0.95, arp: 0.90, bass: 1.00, drums: 1.00 },
        25..=29 => SectionGains { pad: 0.90, arp: 0.90, bass: 0.95, drums: 1.00 },
        30..=32 => SectionGains { pad: 0.85, arp: 0.80, bass: 0.85, drums: 0.00 },
        33..=39 => SectionGains { pad: 0.95, arp: 0.55, bass: 0.00, drums: 0.00 },
        40..=41 => SectionGains { pad: 1.00, arp: 0.70, bass: 0.00, drums: 0.00 },
        42..=43 => SectionGains { pad: 0.60, arp: 0.00, bass: 0.00, drums: 0.00 },
        _ => SectionGains { pad: 0.30, arp: 0.00, bass: 0.00, drums: 0.00 },
    }
}

/// 声部选择器（段落增益查询的 voice 维度）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Voice {
    /// 铺底声部。
    Pad,
    /// 琶音声部。
    Arp,
    /// 低音声部。
    Bass,
}

/// smoothstep：u∈[0,1] → u²(3−2u)（C1 连续；小节间增益无 click）。
fn smoothstep(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// 平滑段落增益：相邻小节间 smoothstep 插值（纯函数；替代 v1 的小节级
/// 阶梯，消除小节边界的增益跳变 click）。鼓激活保持硬切换（60.0s 骤停，
/// 与网格激活逻辑一致，不做插值）。
fn section_gains_at(t_sec: f32) -> SectionGains {
    let bar_sec = 4.0 * 60.0 / BGM_BPM; // 每小节秒数（2.0s）
    let t = t_sec.max(0.0);
    let bar = (t / bar_sec).floor();
    let u = t / bar_sec - bar;
    // 大 t / 越界小节由 section_gains 的 `_` 兜底分支承接（不 panic）。
    let g0 = section_gains(bar as usize);
    let g1 = section_gains(bar as usize + 1);
    let s = smoothstep(u);
    SectionGains {
        pad: g0.pad + (g1.pad - g0.pad) * s,
        arp: g0.arp + (g1.arp - g0.arp) * s,
        bass: g0.bass + (g1.bass - g0.bass) * s,
        drums: g0.drums, // 鼓激活为硬切换（骤停），不插值
    }
}

/// 指定声部在 `t_sec` 时刻的平滑段落增益（纯函数）。
fn voice_gain_at(t_sec: f32, voice: Voice) -> f32 {
    let g = section_gains_at(t_sec);
    match voice {
        Voice::Pad => g.pad,
        Voice::Arp => g.arp,
        Voice::Bass => g.bass,
    }
}

/// arp 声部的小节节奏型（PRD §7 段落编排）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArpMode {
    /// 开场：无常规 arp（稀疏单音另作一次性事件）。
    Sparse,
    /// 八分音符轮转。
    Eighth,
    /// 十六分音符轮转。
    Sixteenth,
    /// 推进段末小节：下行琶音（16 分）。
    Descending,
    /// 沉静段：八分 + 平行五度双旋律。
    DualEighth,
    /// 收尾段：高空十六分「流星」音。
    Meteor,
    /// 静默。
    Silent,
}

/// arp 节奏型表（小节索引 0 起）。
fn arp_mode(bar: usize) -> ArpMode {
    match bar {
        0..=3 => ArpMode::Sparse,
        4..=8 => ArpMode::Eighth,
        9..=15 => ArpMode::Sixteenth,
        16 => ArpMode::Descending, // 第 17 小节（1 起）末下行琶音
        17..=29 => ArpMode::Sixteenth,
        30..=32 => ArpMode::Eighth, // 60.0s 鼓骤停后 arp 转八分
        33..=39 => ArpMode::DualEighth,
        40..=41 => ArpMode::Meteor,
        _ => ArpMode::Silent,
    }
}

/// 推进段末小节（bar 16）的下行琶音音高表（MIDI，16 分音符 16 步）。
const DESCEND_LINE: [i32; 16] =
    [72, 69, 64, 60, 69, 64, 60, 57, 64, 60, 57, 53, 60, 57, 53, 48];

/// S1 三个符号入场时刻的稀疏 arp 单音（全局秒, MIDI）：A4 / C5 / E5。
const ARP_SINGLES: [(f32, i32); 3] = [(1.0, 69), (2.0, 72), (3.0, 76)];

/// S5 答案时刻（全局 61.0s）的 F 大调和弦（F4 / A4 / C5 / F5）。
const F_MAJOR_CHORD: [i32; 4] = [65, 69, 72, 77];
/// F 大调和弦落位时刻（秒）。
const F_CHORD_AT_SEC: f32 = 61.0;

/// S7 切平面时刻（全局 84.0s / 84.75s）的清脆泛音（A5 / E6）。
const BELLS: [(f32, i32); 2] = [(84.0, 81), (84.75, 88)];

/// S6 双旋律汇合同音时刻（全局 73.0s；onset >= 该时刻的双声部同音，
/// 此前平行五度）。
const UNISON_ONSET_SEC: f32 = 73.0;

/// S4 重音时刻（全局 47.5s）在 16 分网格中的步号：
/// 47.5s → 第 24 小节（1 起，0 起 23）内偏移 1.5s → 步 12。
const ACCENT_STEP: usize = 23 * 16 + 12;

// ---------------------------------------------------------------------------
// 人性化：确定性整数哈希（禁 rand crate）
// ---------------------------------------------------------------------------

/// 确定性整数哈希（splitmix64 终结化；禁 rand crate）：
/// 对 (bar, step, voice) 三元组映射到 u64——同键恒同值、异键高离散，
/// 全链无时钟 / 随机 / 内部可变状态。
fn hash3(bar: u64, step: u64, voice: u64) -> u64 {
    // 三元组各乘一个奇数常量错位相消（对称性破除），再走 splitmix64
    // 终结化（两次 xor-shift-multiply），低位亦均匀。
    let mut z = bar
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(step.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(voice.wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 取哈希的 24 位精度归一化小数 ∈ [0,1)（f32 尾数 24 位，无精度浪费）。
fn hash_unit(h: u64) -> f32 {
    (h >> 40) as f32 / ((1u64 << 24) as f32)
}

/// 音符 onset 人性化偏移：±12ms（确定性；对 (bar, step, voice) 哈希）。
fn onset_offset_sec(bar: u64, step: u64, voice: u64) -> f32 {
    (hash_unit(hash3(bar, step, voice)) * 2.0 - 1.0) * HUMAN_ONSET_SEC
}

/// 音符增益人性化：×[0.92, 1.08]（确定性；与 onset 抖动用不同盐，
/// 两者独立）。
fn gain_jitter(bar: u64, step: u64, voice: u64) -> f32 {
    let salted = hash3(bar, step, voice.wrapping_add(0x5DEE_CE6D));
    1.0 + (hash_unit(salted) * 2.0 - 1.0) * HUMAN_GAIN
}

/// 由标称时刻推导 (bar, step) 人性化哈希键（16 分网格量化；一次性
/// 事件无自然步号时使用）。
fn bar_step_of(t_sec: f32) -> (u64, u64) {
    let bar_sec = 4.0 * 60.0 / BGM_BPM;
    let step_sec = bar_sec / 16.0;
    let t = t_sec.max(0.0);
    let bar = (t / bar_sec).floor();
    let step = ((t - bar * bar_sec) / step_sec).round().clamp(0.0, 15.0);
    (bar as u64, step as u64)
}

// ---------------------------------------------------------------------------
// 音色丰富化：微失谐多副本 + 谐波层叠（仅 Sine / Triangle，经 render_note）
// ---------------------------------------------------------------------------

/// 一个音色副本：频率倍率（微失谐 / 谐波）+ 权重。
#[derive(Debug, Clone, Copy)]
struct RichCopy {
    /// 相对基频的倍率（1.0 = 基频；2.0 / 3.0 = 谐波层叠）。
    ratio: f32,
    /// 叠合权重（副本栈权重和 ≈ 1，保持声部电平与 v1 可比）。
    weight: f32,
}

/// pad 副本栈（长音）：基频 + 两对微失谐（±0.25% / ±0.5% ⇒ 慢速拍音
/// 绒毛，近似颤音动感）+ 2/3 次谐波低增益（加法合成的类锯齿温暖度）。
const PAD_COPIES: [RichCopy; 7] = [
    RichCopy { ratio: 1.0000, weight: 0.46 },
    RichCopy { ratio: 1.0025, weight: 0.15 },
    RichCopy { ratio: 0.9975, weight: 0.15 },
    RichCopy { ratio: 1.0050, weight: 0.05 },
    RichCopy { ratio: 0.9950, weight: 0.05 },
    RichCopy { ratio: 2.0000, weight: 0.09 },
    RichCopy { ratio: 3.0000, weight: 0.05 },
];

/// arp / 提示音副本栈（短音）：基频 + 一对微失谐（±0.3%）+ 2 次谐波。
const LEAD_COPIES: [RichCopy; 4] = [
    RichCopy { ratio: 1.000, weight: 0.66 },
    RichCopy { ratio: 1.003, weight: 0.12 },
    RichCopy { ratio: 0.997, weight: 0.12 },
    RichCopy { ratio: 2.000, weight: 0.10 },
];

/// bass 副本栈：基频 + 一对微失谐（±0.4%）+ 2 次谐波（低音温暖度）。
const BASS_COPIES: [RichCopy; 4] = [
    RichCopy { ratio: 1.000, weight: 0.70 },
    RichCopy { ratio: 1.004, weight: 0.10 },
    RichCopy { ratio: 0.996, weight: 0.10 },
    RichCopy { ratio: 2.000, weight: 0.10 },
];

/// 加法富化渲染：副本栈逐副本经 `render_note`（旧 API）渲染后加权叠合
/// ——手工微失谐齐奏 + 谐波层叠，无任何引擎新原语；纯函数、逐位确定。
fn render_rich(
    waveform: Waveform,
    freq: f32,
    env: &Adsr,
    gain: f32,
    frames: usize,
    sample_rate: u32,
    copies: &[RichCopy],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; frames];
    for c in copies {
        let note =
            render_note(waveform, freq * c.ratio, env, gain * c.weight, frames, sample_rate);
        for (o, v) in out.iter_mut().zip(note.iter()) {
            *o += *v;
        }
    }
    out
}

/// 音符内平滑段落包络：`note` 已按 onset 增益渲染，此处按样本乘
/// `voice_gain_at(t)/onset_gain`——音符内增益连续变化（长音跨小节边界
/// 时也无阶梯 click）；纯函数。
fn apply_section_envelope(
    note: &mut [f32],
    at_sec: f32,
    onset_gain: f32,
    voice: Voice,
    sample_rate: u32,
) {
    if onset_gain <= 0.0 || note.is_empty() {
        return;
    }
    for (i, v) in note.iter_mut().enumerate() {
        let t = at_sec + i as f32 / sample_rate as f32;
        *v *= voice_gain_at(t, voice) / onset_gain;
    }
}

// ---------------------------------------------------------------------------
// 手工混响：Schroeder 结构（4 并联梳状 + 2 串联全通），L/R 不同抽头
// ---------------------------------------------------------------------------

/// 单侧混响链：4 并联梳状滤波器 + 2 串联全通滤波器（Schroeder 经典
/// 结构；延迟线为环形缓冲，状态仅延迟线本身，无时钟 / 随机）。
struct ReverbSide {
    /// 梳状延迟线长度（样本）。
    comb_len: [usize; 4],
    /// 梳状反馈增益（<1 保证稳定；各异 ⇒ 模态密度）。
    comb_gain: [f32; 4],
    /// 梳状环形缓冲（存历史输出 y）。
    comb_buf: [Vec<f32>; 4],
    /// 梳状写指针。
    comb_idx: [usize; 4],
    /// 全通延迟线长度（样本）。
    ap_len: [usize; 2],
    /// 全通增益（<1 保证稳定）。
    ap_gain: [f32; 2],
    /// 全通环形缓冲（存历史输出 y）。
    ap_buf: [Vec<f32>; 2],
    /// 全通写指针。
    ap_idx: [usize; 2],
}

impl ReverbSide {
    /// 构造一侧混响链（延迟线清零）。
    fn new(
        comb_len: [usize; 4],
        comb_gain: [f32; 4],
        ap_len: [usize; 2],
        ap_gain: [f32; 2],
    ) -> Self {
        Self {
            comb_buf: [
                vec![0.0_f32; comb_len[0]],
                vec![0.0_f32; comb_len[1]],
                vec![0.0_f32; comb_len[2]],
                vec![0.0_f32; comb_len[3]],
            ],
            comb_idx: [0; 4],
            ap_buf: [vec![0.0_f32; ap_len[0]], vec![0.0_f32; ap_len[1]]],
            ap_idx: [0; 2],
            comb_len,
            comb_gain,
            ap_len,
            ap_gain,
        }
    }

    /// 处理一个输入样本，返回该侧湿声（梳状并联 ⇒ 全通串联）。
    fn process(&mut self, x: f32) -> f32 {
        // 反馈梳状：y = x + g·y[t−d]（并联 4 路，均值化防能量过冲）。
        let mut bank = 0.0_f32;
        for i in 0..4 {
            let d = self.comb_len[i];
            let j = self.comb_idx[i];
            let y = self.comb_buf[i][j];
            let v = x + self.comb_gain[i] * y;
            self.comb_buf[i][j] = v;
            self.comb_idx[i] = if j + 1 == d { 0 } else { j + 1 };
            bank += v;
        }
        let mut s = bank * 0.25;
        // 全通：y = −g·x + x[t−d] + g·y[t−d]（单缓冲经典形式；|H|=1，
        // 只改相位不改幅——抹平梳状的梳齿状频响）。
        for i in 0..2 {
            let d = self.ap_len[i];
            let j = self.ap_idx[i];
            let delayed = self.ap_buf[i][j];
            let y = -self.ap_gain[i] * s + delayed;
            self.ap_buf[i][j] = s + self.ap_gain[i] * y;
            self.ap_idx[i] = if j + 1 == d { 0 } else { j + 1 };
            s = y;
        }
        s
    }
}

/// 立体声混响：左右两侧梳状 / 全通参数错开（R 侧延迟线整体 +37 样本
/// ≈0.84ms@44100）⇒ 左右抽头去相关，形成立体声像；尾音长度约 1.2s
/// （反馈 ≤0.86），自然截断于母带长度内。
struct StereoReverb {
    /// 左声道混响链。
    left: ReverbSide,
    /// 右声道混响链。
    right: ReverbSide,
}

impl StereoReverb {
    /// 按采样率构造（延迟线按 44100 设计，其它采样率等比缩放取整）。
    fn new(sample_rate: u32) -> Self {
        let scale = sample_rate.max(1) as f32 / BGM_SR as f32;
        let sc = |d: u32| -> usize { ((d as f32 * scale).round() as usize).max(1) };
        Self {
            // L：基准 Schroeder 延迟线（1116/1188/1277/1356 + 全通 556/441）。
            left: ReverbSide::new(
                [sc(1116), sc(1188), sc(1277), sc(1356)],
                [0.80, 0.82, 0.84, 0.86],
                [sc(556), sc(441)],
                [0.5, 0.7],
            ),
            // R：整体 +37 样本微差 + 反馈序错开 ⇒ 去相关。
            right: ReverbSide::new(
                [sc(1153), sc(1225), sc(1314), sc(1393)],
                [0.82, 0.84, 0.86, 0.80],
                [sc(593), sc(478)],
                [0.7, 0.5],
            ),
        }
    }

    /// 整块处理：`dry` → 左右湿声（同输入、不同抽头）。
    fn process_block(&mut self, dry: &[f32], wet_l: &mut [f32], wet_r: &mut [f32]) {
        for i in 0..dry.len() {
            wet_l[i] = self.left.process(dry[i]);
            wet_r[i] = self.right.process(dry[i]);
        }
    }
}

/// 分段软膝限幅（tanh 近似）：|x| ≤ [`SOFT_KNEE`] 恒等（中频段动态
/// 保留），超出按 `1 − (1−T)²/((a−T)+(1−T))` 平滑压缩——a=T 处取值 T、
/// 斜率 1（与线性区 C1 拼接，无导数断裂），严格单调、渐近上限 1（峰值
/// 恒 ≤1，无硬 clip）；纯算术（无三角函数调用），逐位确定。NaN 归零、
/// ±inf 饱和到 ±1。
fn soft_knee(x: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let a = x.abs();
    if a <= SOFT_KNEE {
        return x;
    }
    // 软膝曲线：a=T 处取值 T、斜率 1（与线性区 C1 拼接），a→∞ 时 →1。
    let c = 1.0
        - (1.0 - SOFT_KNEE) * (1.0 - SOFT_KNEE)
            / ((a - SOFT_KNEE) + (1.0 - SOFT_KNEE));
    if x < 0.0 {
        -c
    } else {
        c
    }
}

/// 立体声展宽：单声道母带 → 左右不同混响抽头 + L/R 微偏移（Haas）
/// → 软限幅 → 交织立体声样本（`LRLR…`，长度 = 2×母带帧数）。
fn widen_to_stereo(mono: &[f32], sample_rate: u32) -> Vec<f32> {
    let n = mono.len();
    let mut wet_l = vec![0.0_f32; n];
    let mut wet_r = vec![0.0_f32; n];
    let mut reverb = StereoReverb::new(sample_rate);
    reverb.process_block(mono, &mut wet_l, &mut wet_r);
    let off_l = (HAAS_OFFSET_L_SEC * sample_rate as f32).round() as usize;
    let off_r = (HAAS_OFFSET_R_SEC * sample_rate as f32).round() as usize;
    let mut out = vec![0.0_f32; n * 2];
    for i in 0..n {
        let dry_l = if i >= off_l { mono[i - off_l] } else { 0.0 };
        let dry_r = if i >= off_r { mono[i - off_r] } else { 0.0 };
        out[2 * i] = soft_knee(dry_l + REVERB_WET * wet_l[i]);
        out[2 * i + 1] = soft_knee(dry_r + REVERB_WET * wet_r[i]);
    }
    out
}

// ---------------------------------------------------------------------------
// 鼓组 swing：反拍八分后移（16 分网格切片重落位）
// ---------------------------------------------------------------------------

/// 16 分步的 swing 后移量（样本）：反拍八分（s%4==2，即 1 基计数下
/// 第 2/4 个八分位）后移 6% 个十六分；kicks（s%4==0）与 47.5s 重音
/// （s=380，s%4==0）保持网格。
/// 注：网格鼓点全部落在偶数 16 分步上，字面「偶数 16 分步 +6%」会整体
/// 后移全部鼓点（无 swing 听感）——故按经典 shuffle 落地为反拍八分
/// 后移；如需坚持字面语义可改为 `s % 2 == 0`。
fn swing_shift_frames(step: usize, step_frames: usize) -> usize {
    if step % 4 == 2 {
        (SWING_SHIFT_RATIO * step_frames as f32).round() as usize
    } else {
        0
    }
}

/// 对鼓组单声道母带施加 swing：按 16 分槽切片后移（移动副本槽头加
/// 余弦淡入消 click；溢出的衰减尾加性叠入下一槽——打击尾自然搭接）。
/// 槽长不变（母带长度是全片唯一事实源）；纯函数。
fn apply_drum_swing(buf: &[f32], bpm: f32, sample_rate: u32) -> Vec<f32> {
    let n = buf.len();
    let mut out = vec![0.0_f32; n];
    if n == 0 || bpm <= 0.0 || sample_rate < 1 {
        return out;
    }
    let step_frames = ((60.0 / bpm / 4.0) * sample_rate as f32).round() as usize;
    if step_frames == 0 {
        // 极低采样率：网格无法切片，恒等返回。
        out.copy_from_slice(buf);
        return out;
    }
    let edge =
        ((SWING_EDGE_SEC * sample_rate as f32).round() as usize).clamp(1, step_frames / 4);
    let slots = n.div_ceil(step_frames);
    for s in 0..slots {
        let shift = swing_shift_frames(s, step_frames);
        let start = s * step_frames;
        let end = (start + step_frames).min(n);
        for j in 0..(end - start) {
            let dst = start + j + shift;
            if dst >= n {
                break; // 尾部溢出的衰减尾：截断（母带末尾之外无声）
            }
            // 槽头余弦淡入仅施加于移动副本（shift=0 时恒等，不动网格鼓点）。
            let g = if shift > 0 && j < edge {
                0.5 - 0.5 * (std::f32::consts::PI * j as f32 / edge as f32).cos()
            } else {
                1.0
            };
            out[dst] += buf[start + j] * g;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 母带工具
// ---------------------------------------------------------------------------

/// 把一段音符样本叠加进 `buf` 的 `at_sec` 秒偏移处（越界部分截断）。
fn place(buf: &mut [f32], at_sec: f32, samples: &[f32], sample_rate: u32) {
    let start = (at_sec * sample_rate as f32).round() as usize;
    for (i, v) in samples.iter().enumerate() {
        if start + i >= buf.len() {
            break; // 音符伸到母带末尾之外：截断
        }
        buf[start + i] += v;
    }
}

/// 长度规整：不足补 0、超出截断（母带长度是全片唯一事实源）。
fn fit_len(buf: PcmBuffer, total: usize) -> PcmBuffer {
    let mut data = buf.as_slice().to_vec();
    data.resize(total, 0.0);
    PcmBuffer::from_mono(buf.sample_rate(), data).expect("单声道缓冲长度恒合法")
}

/// Pad 音色：取 `Mood::Calm` 预设的铺底音色（正弦波 + 慢 ADSR）。
///
/// 预设为编译期常量表、恒可取；仍防御性兜底——参数退化时手写慢 ADSR。
fn pad_timbre() -> Timbre {
    let t = preset(Mood::Calm).pad;
    if t.envelope.attack + t.envelope.decay + t.envelope.release <= 0.0 {
        Timbre {
            waveform: t.waveform,
            envelope: Adsr { attack: 0.80, decay: 0.40, sustain: 0.70, release: 1.20 },
        }
    } else {
        t
    }
}

/// 母带淡变包络：淡入 [`MASTER_FADE_IN_SEC`] 线性 0→1、淡出
/// [`MASTER_FADE_OUT_SEC`] 线性 1→0（末帧恒 0）；按**帧**计（交错
/// 多声道下同帧各声道同步）；纯函数、逐样本。
fn apply_master_fade(data: &mut [f32], sample_rate: u32, channels: u16) {
    let ch = channels.max(1) as usize;
    let frames = data.len() / ch;
    if frames == 0 {
        return;
    }
    let fin = (MASTER_FADE_IN_SEC * sample_rate as f32).round() as usize;
    let fout = (MASTER_FADE_OUT_SEC * sample_rate as f32).round() as usize;
    for f in 0..frames {
        let g_in = if fin > 0 && f < fin { f as f32 / fin as f32 } else { 1.0 };
        let g_out = if fout > 0 && f + fout >= frames {
            (frames - 1 - f) as f32 / fout as f32
        } else {
            1.0
        };
        let g = g_in.min(g_out).max(0.0);
        for c in 0..ch {
            data[f * ch + c] *= g;
        }
    }
}

/// 合成整片 BGM（立体声，`duration_sec` 秒）。
///
/// - 小节数 = `ceil(duration / (4×60/BPM))`（120BPM ⇒ 每小节 2.0s，
///   90s ⇒ 恰 45 小节精确闭合）；
/// - 四声部各渲染为等长单声道缓冲后经 [`mix_tracks`] 固定序线性混合
///   （内含峰值防削波：混合峰值 >1 时整轨等比缩放）；
/// - 母带经 [`widen_to_stereo`] 展宽：左右不同混响抽头 + L/R 微偏移
///   （Haas）+ 分段软膝限幅（峰值渐近 ≤1）；
/// - **母带淡入 1.5s / 淡出 1.2s** 施加于展宽后的立体声母带（端点
///   样本恒 0；先淡变再展宽会使微偏移通道末样本为非零副本，违反该
///   不变量）；
/// - 长度不变量：帧数 = `round(duration×sample_rate)`（交织样本数 =
///   2×帧数），端点样本恒 0；
/// - 非法输入不 panic：时长 / 采样率钳到最小合法值（正时长 / 正采样率
///   是调用方契约）。
pub fn synth_bgm(duration_sec: f32, sample_rate: u32) -> PcmBuffer {
    let sr = sample_rate.max(1);
    let total = ((duration_sec.max(0.0) * sr as f32).round() as usize).max(1);
    let bar_sec = 4.0 * 60.0 / BGM_BPM; // 每小节秒数（2.0s）
    let bars = ((duration_sec.max(0.0) / bar_sec).ceil() as usize).max(1);

    // ---- 声部 1：Pad（每小节根 / 三 / 五长音；大绽放两小节加高八度根）----
    let pad = pad_timbre();
    let pad_frames = (1.9 * sr as f32).round() as usize;
    let mut pad_buf = vec![0.0_f32; total];
    for bar in 0..bars {
        let chord = &CHORDS[bar % CHORDS.len()];
        let at0 = bar as f32 * bar_sec;
        let bloom = (40..=41).contains(&bar); // 大绽放：加高八度根音（全和弦）
        for (idx, &m) in chord.notes.iter().enumerate() {
            if idx == 3 && !bloom {
                continue;
            }
            let g0 = voice_gain_at(at0, Voice::Pad);
            if g0 <= 0.0 {
                continue;
            }
            let at = (at0 + onset_offset_sec(bar as u64, idx as u64, VOICE_PAD)).max(0.0);
            let mut note = render_rich(
                pad.waveform,
                midi_to_freq(m),
                &pad.envelope,
                g0,
                pad_frames,
                sr,
                &PAD_COPIES,
            );
            apply_section_envelope(&mut note, at, g0, Voice::Pad, sr);
            let gj = gain_jitter(bar as u64, idx as u64, VOICE_PAD);
            for v in note.iter_mut() {
                *v *= gj;
            }
            place(&mut pad_buf, at, &note, sr);
        }
    }

    // ---- 声部 2：Arp（节奏型 / 增益随段落平滑变化）----
    let arp_env = Adsr { attack: 0.005, decay: 0.20, sustain: 0.16, release: 0.05 };
    let arp_frames = (0.18 * sr as f32).round() as usize;
    let meteor_env = Adsr { attack: 0.003, decay: 0.10, sustain: 0.04, release: 0.05 };
    let meteor_frames = (0.14 * sr as f32).round() as usize;
    let mut arp_buf = vec![0.0_f32; total];
    // 开场稀疏单音（S1 符号入场标记；一次性事件，不经段落增益）。
    for (k, &(at0, m)) in ARP_SINGLES.iter().enumerate() {
        let (b, st) = bar_step_of(at0);
        let at = (at0 + onset_offset_sec(b, st, VOICE_SINGLE + k as u64)).max(0.0);
        let gj = gain_jitter(b, st, VOICE_SINGLE + k as u64);
        let mut note = render_rich(
            Waveform::Triangle,
            midi_to_freq(m),
            &arp_env,
            0.22,
            arp_frames,
            sr,
            &LEAD_COPIES,
        );
        for v in note.iter_mut() {
            *v *= gj;
        }
        place(&mut arp_buf, at, &note, sr);
    }
    for bar in 0..bars {
        let mode = arp_mode(bar);
        if mode == ArpMode::Sparse || mode == ArpMode::Silent {
            continue;
        }
        let chord = &CHORDS[bar % CHORDS.len()];
        let bar_start = bar as f32 * bar_sec;
        let steps = match mode {
            ArpMode::Eighth | ArpMode::DualEighth => 8,
            _ => 16,
        };
        let step_sec = bar_sec / steps as f32;
        for s in 0..steps {
            let at0 = bar_start + s as f32 * step_sec;
            let g0 = voice_gain_at(at0, Voice::Arp);
            if g0 <= 0.0 {
                continue;
            }
            let pitch = match mode {
                ArpMode::Descending => DESCEND_LINE[s],
                ArpMode::DualEighth => chord.notes[s % 4] + 12,
                ArpMode::Meteor => chord.notes[s % 4] + 24,
                _ => chord.notes[s % 4] + 24, // Eighth / Sixteenth
            };
            let (env, frames) = match mode {
                ArpMode::Meteor => (&meteor_env, meteor_frames),
                _ => (&arp_env, arp_frames),
            };
            let at = (at0 + onset_offset_sec(bar as u64, s as u64, VOICE_ARP)).max(0.0);
            let mut note = render_rich(
                Waveform::Triangle,
                midi_to_freq(pitch),
                env,
                g0,
                frames,
                sr,
                &LEAD_COPIES,
            );
            apply_section_envelope(&mut note, at, g0, Voice::Arp, sr);
            let gj = gain_jitter(bar as u64, s as u64, VOICE_ARP);
            for v in note.iter_mut() {
                *v *= gj;
            }
            place(&mut arp_buf, at, &note, sr);
            if mode == ArpMode::DualEighth {
                // 平行五度双旋律；onset >= UNISON_ONSET_SEC 起汇合同音。
                let interval = if at >= UNISON_ONSET_SEC { 0 } else { 7 };
                let at2 =
                    (at0 + onset_offset_sec(bar as u64, s as u64 + 16, VOICE_ARP2)).max(0.0);
                let mut note2 = render_rich(
                    Waveform::Triangle,
                    midi_to_freq(pitch + interval),
                    &arp_env,
                    g0 * 0.8,
                    frames,
                    sr,
                    &LEAD_COPIES,
                );
                apply_section_envelope(&mut note2, at2, g0 * 0.8, Voice::Arp, sr);
                let gj2 = gain_jitter(bar as u64, s as u64 + 16, VOICE_ARP2);
                for v in note2.iter_mut() {
                    *v *= gj2;
                }
                place(&mut arp_buf, at2, &note2, sr);
            }
        }
    }

    // ---- 声部 3：Bass（推进段起的行走低音，根音低八度）----
    let bass_env = Adsr { attack: 0.01, decay: 0.18, sustain: 0.55, release: 0.25 };
    let bass_frames = (1.2 * sr as f32).round() as usize;
    let mut bass_buf = vec![0.0_f32; total];
    for bar in 0..bars {
        let at0 = bar as f32 * bar_sec;
        let g0 = voice_gain_at(at0, Voice::Bass);
        if g0 <= 0.0 {
            continue;
        }
        let root = CHORDS[bar % CHORDS.len()].notes[0] - 12; // 根音低八度
        let at = (at0 + onset_offset_sec(bar as u64, 0, VOICE_BASS)).max(0.0);
        let mut note = render_rich(
            Waveform::Sine,
            midi_to_freq(root),
            &bass_env,
            g0,
            bass_frames,
            sr,
            &BASS_COPIES,
        );
        apply_section_envelope(&mut note, at, g0, Voice::Bass, sr);
        let gj = gain_jitter(bar as u64, 0, VOICE_BASS);
        for v in note.iter_mut() {
            *v *= gj;
        }
        place(&mut bass_buf, at, &note, sr);
    }

    // ---- 一次性音乐提示（PRD §7 场景切换提示；落在 pad 层）----
    // 61.0s F 大调和弦点亮（S5 答案时刻）。
    let chord_env = Adsr { attack: 0.01, decay: 0.35, sustain: 0.55, release: 0.45 };
    let chord_frames = (1.5 * sr as f32).round() as usize;
    let (cb, cs) = bar_step_of(F_CHORD_AT_SEC);
    for (i, &m) in F_MAJOR_CHORD.iter().enumerate() {
        let at = (F_CHORD_AT_SEC + onset_offset_sec(cb, cs + i as u64, VOICE_CHORD)).max(0.0);
        let gj = gain_jitter(cb, cs + i as u64, VOICE_CHORD);
        let mut note = render_rich(
            Waveform::Triangle,
            midi_to_freq(m),
            &chord_env,
            0.20,
            chord_frames,
            sr,
            &LEAD_COPIES,
        );
        for v in note.iter_mut() {
            *v *= gj;
        }
        place(&mut pad_buf, at, &note, sr);
    }
    // 84.0s / 84.75s 清脆泛音（S7 切平面时刻）。
    let bell_env = Adsr { attack: 0.004, decay: 0.85, sustain: 0.05, release: 0.30 };
    let bell_frames = (1.2 * sr as f32).round() as usize;
    for (k, &(at0, m)) in BELLS.iter().enumerate() {
        let (b, st) = bar_step_of(at0);
        let at = (at0 + onset_offset_sec(b, st, VOICE_BELL + k as u64)).max(0.0);
        let gj = gain_jitter(b, st, VOICE_BELL + k as u64);
        let mut note = render_rich(
            Waveform::Sine,
            midi_to_freq(m),
            &bell_env,
            0.18,
            bell_frames,
            sr,
            &LEAD_COPIES,
        );
        for v in note.iter_mut() {
            *v *= gj;
        }
        place(&mut pad_buf, at, &note, sr);
    }

    // ---- 声部 4：鼓组（高潮段四四拍 kick + 反拍 hat；60.0s 骤停）----
    // 以 drums.rs 既有 API 编排：16 分音符网格，逐事件经 synth 合成累加。
    let steps = 16 * bars; // 16 步/小节（一步 = 16 分音符）
    let mut grid = vec![vec![0_u8; steps]; 3]; // 行序 = Kick / Snare / Hat
    for s in 0..steps {
        let bar = s / 16;
        if section_gains(bar).drums <= 0.0 {
            continue; // 段落外不激活（60.0s 骤停 = bar 30 起全零）
        }
        if s % 4 == 0 {
            grid[0][s] = 1; // 四分音符（四四拍）一 kick
        }
        if s % 4 == 2 {
            grid[2][s] = 1; // 反拍 hat
        }
    }
    if ACCENT_STEP < steps {
        // 47.5s 重音（S4 crash 位）：snare + hat 重击。
        grid[1][ACCENT_STEP] = 1;
        grid[2][ACCENT_STEP] = 1;
    }
    let drum_spec = DrumSpec { steps, grid };
    let drum_track =
        drums::sequence(&drum_spec, BGM_BPM, sr).expect("鼓机编排参数合法（BPM/SR 为正、步数 ≥1）");
    let drums_buf = fit_len(drums::render(&drum_track), total);
    // swing：反拍八分后移 6% 个十六分（人性化节奏）。
    let swung = apply_drum_swing(drums_buf.as_slice(), BGM_BPM, sr);

    // ---- 混音（轨序 pad→arp→bass→drums；增益见分镜规格）----
    let mix = mix_tracks(&[
        MixTrack::new(
            PcmBuffer::from_mono(sr, pad_buf).expect("pad 缓冲构造"),
            0.22,
        ),
        MixTrack::new(
            PcmBuffer::from_mono(sr, arp_buf).expect("arp 缓冲构造"),
            0.16,
        ),
        MixTrack::new(
            PcmBuffer::from_mono(sr, bass_buf).expect("bass 缓冲构造"),
            0.26,
        ),
        MixTrack::new(
            PcmBuffer::from_mono(sr, swung).expect("鼓组缓冲构造"),
            0.45, // soft kick
        ),
    ])
    .expect("四声部等长同型，混音不应失败");

    // ---- 立体声展宽（混响抽头 + L/R 微偏移 + 软限幅）→ 母带淡变 ----
    let mut stereo = widen_to_stereo(mix.as_slice(), sr);
    apply_master_fade(&mut stereo, sr, BGM_CHANNELS);
    PcmBuffer::from_interleaved(sr, BGM_CHANNELS, stereo).expect("立体声母带构造")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::TOTAL_DURATION_SEC;
    use std::collections::HashSet;

    #[test]
    fn bgm_duration_pins_film_total() {
        // BGM 时长与全片总长单一事实源一致；120BPM × 45 小节精确闭合。
        assert_eq!(BGM_DURATION_SEC, TOTAL_DURATION_SEC, "BGM 时长必须与全片总长一致");
        let bar_sec = 4.0 * 60.0 / BGM_BPM;
        assert!((bar_sec - 2.0).abs() < 1e-6, "120BPM 每小节必须 = 2.0s");
        assert_eq!(
            (BGM_DURATION_SEC / bar_sec).ceil() as usize,
            45,
            "90.000s 必须恰为 45 小节"
        );
    }

    #[test]
    fn section_gains_follow_mood_curve() {
        // 鼓：开场 / 推进无；高潮进；60.0s（bar 30）骤停；沉静 / 收尾无。
        assert!(section_gains(0).drums <= 0.0, "开场不应有鼓");
        assert!(section_gains(10).drums <= 0.0, "推进段（S3）不应有鼓");
        assert!(section_gains(20).drums > 0.0, "高潮段（S4）必须有鼓");
        assert!(section_gains(29).drums > 0.0, "持续段鼓应延续到 60.0s 前");
        assert!(section_gains(30).drums <= 0.0, "bar 30（60.0s）鼓必须骤停");
        assert!(section_gains(37).drums <= 0.0, "沉静段（S6）无鼓");
        assert!(section_gains(44).drums <= 0.0, "收尾段（S7）无鼓");
        // 低音：仅推进 / 高潮 / 持续（18–66s）。
        assert!(section_gains(5).bass <= 0.0);
        assert!(section_gains(12).bass > 0.0 && section_gains(20).bass > 0.0);
        assert!(section_gains(35).bass <= 0.0, "沉静段无低音");
        // 收尾仅氛围 pad（其余声部全静默），且比沉静段更轻。
        let tail = section_gains(44);
        assert!(tail.arp <= 0.0 && tail.bass <= 0.0 && tail.drums <= 0.0 && tail.pad > 0.0);
        assert!(tail.pad < section_gains(33).pad, "落版氛围音应低于沉静段");
        // 大绽放两小节（80–84s）pad 增益达到全片顶点之一。
        assert!(section_gains(40).pad >= 1.0);
    }

    #[test]
    fn master_fade_ramp_shape() {
        // 1.5s 淡入 / 1.2s 淡出（线性；端点恒 0）。sr=1000 ⇒ fin=1500、fout=1200；
        // 缓冲 5000 帧 > fin+fout=2700，中段 [1500, 3800) 增益恒 1（两坡不重叠）。
        let mut x = vec![1.0_f32; 5000];
        apply_master_fade(&mut x, 1000, 1);
        // 淡入段：g = i/1500。
        assert_eq!(x[0], 0.0, "淡入端点必须为 0");
        assert!((x[750] - 0.5).abs() < 1e-6, "1.5s 淡入中点（750 样本）增益应 = 0.5");
        assert!((x[1499] - 1499.0 / 1500.0).abs() < 1e-6, "淡入末样本增益应 = 1499/1500");
        assert_eq!(x[1500], 1.0, "淡入结束即刻增益应为 1");
        // 平稳段：i ≥ fin 且 i + fout < n ⇒ 增益恒 1。
        assert_eq!(x[3000], 1.0, "中段平稳区增益必须为 1");
        // 淡出段：i + 1200 ≥ 5000 ⟺ i ≥ 3800；g = (4999−i)/1200。
        assert!((x[3800] - 1199.0 / 1200.0).abs() < 1e-6, "淡出首样本增益应 = 1199/1200");
        assert!((x[4400] - 599.0 / 1200.0).abs() < 1e-6, "淡出中段增益应 = 599/1200");
        assert_eq!(x[4999], 0.0, "淡出端点必须为 0");
        // 全程增益不超过 1（包络只衰减不放大）。
        for (i, v) in x.iter().enumerate() {
            assert!(*v <= 1.0 + 1e-6, "样本 {i} 增益越界：{v}");
        }
    }

    #[test]
    fn master_fade_multichannel_is_per_frame() {
        // 多声道下淡变必须按帧同步（交错样本同帧同增益），而非按样本索引
        // （否则立体声淡变速度翻倍）。
        let mut x = vec![1.0_f32; 4000]; // 2000 帧 × 2 声道
        apply_master_fade(&mut x, 1000, 2);
        // fin=1500 帧 ⇒ 第 750 帧增益 0.5（两声道一致）。
        assert!((x[1500] - 0.5).abs() < 1e-6, "L 声道第 750 帧增益应 = 0.5");
        assert!((x[1501] - 0.5).abs() < 1e-6, "R 声道第 750 帧增益应 = 0.5");
        assert_eq!(x[3998], 0.0, "淡出端点（L）必须为 0");
        assert_eq!(x[3999], 0.0, "淡出端点（R）必须为 0");
    }

    #[test]
    fn synth_bgm_full_length_peak_and_fades() {
        let bgm = synth_bgm(BGM_DURATION_SEC, BGM_SR);
        assert_eq!(bgm.frames(), 3_969_000, "90.000s × 44100 必须 = 3,969,000 采样帧");
        assert_eq!(
            bgm.channels(),
            2,
            "必须立体声（母带展宽：L/R 微偏移 + 混响抽头；采样帧口径不变）"
        );
        assert_eq!(
            bgm.as_slice().len(),
            3_969_000 * 2,
            "交织样本数 = 帧数 × 声道数 = 7,938,000"
        );
        assert!(bgm.peak() <= 1.0, "峰值必须 ≤1（软限幅兜底），实测 {}", bgm.peak());
        assert_eq!(bgm.sample(0, 0), Some(0.0), "淡入端点（L）样本必须为 0");
        assert_eq!(bgm.sample(0, 1), Some(0.0), "淡入端点（R）样本必须为 0");
        assert_eq!(bgm.sample(bgm.frames() - 1, 0), Some(0.0), "淡出端点（L）样本必须为 0");
        assert_eq!(bgm.sample(bgm.frames() - 1, 1), Some(0.0), "淡出端点（R）样本必须为 0");
        let win = (0.05 * BGM_SR as f32) as usize;
        let head = bgm.as_slice()[..win].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let tail = bgm.as_slice()[bgm.frames() * 2 - win..].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        assert!(head < 0.01, "首 0.05s 幅度必须 <0.01（母带淡入 1.5s），实测 {head}");
        assert!(tail < 0.01, "尾 0.05s 幅度必须 <0.01（母带淡出 1.2s），实测 {tail}");
    }

    #[test]
    fn synth_bgm_deterministic_and_illegal_inputs_safe() {
        // 确定性：同参数两次合成逐位一致（纯函数契约；覆盖立体声两声道、
        // 人性化哈希与混响状态——无时钟 / 随机 / 内部可变状态泄漏）。
        let a = synth_bgm(6.0, 16_000);
        let b = synth_bgm(6.0, 16_000);
        assert_eq!(a.frames(), b.frames());
        assert_eq!(a.channels(), b.channels());
        for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "BGM 必须是逐位确定的纯函数");
        }
        // 非法输入不 panic：零时长 / 零采样率钳到最小合法缓冲。
        let zero = synth_bgm(0.0, BGM_SR);
        assert!(zero.frames() >= 1, "零时长应钳成 ≥1 样本而不 panic");
        let no_sr = synth_bgm(1.0, 0);
        assert!(no_sr.frames() >= 1, "零采样率应钳成 ≥1 样本而不 panic");
    }

    #[test]
    fn rich_note_has_harmonics_and_beating() {
        // 音色丰富化：谐波层叠（2f/3f 能量可闻）+ 微失谐拍音（包络波动）。
        let sr = 16_000u32;
        let frames = (2.0 * sr as f32) as usize;
        let env = Adsr { attack: 0.01, decay: 0.10, sustain: 1.00, release: 0.10 };
        let freq = 250.0; // 整数 bin（250×16000/16000 = 250 周期），投影无泄漏
        let plain = render_note(Waveform::Sine, freq, &env, 1.0, frames, sr);
        let rich = render_rich(Waveform::Sine, freq, &env, 1.0, frames, sr, &PAD_COPIES);
        assert_eq!(rich.len(), frames);
        // 单频投影（整数 bin 下正交）：测 2f/3f 分量幅度。
        let band = |buf: &[f32], f: f32| -> f32 {
            let (mut re, mut im) = (0.0_f32, 0.0_f32);
            for (i, v) in buf.iter().enumerate() {
                let ph = 2.0 * std::f32::consts::PI * f * i as f32 / sr as f32;
                re += v * ph.cos();
                im += v * ph.sin();
            }
            (re * re + im * im).sqrt() / (buf.len() as f32 / 2.0)
        };
        let h2_plain = band(&plain, 2.0 * freq);
        let h2_rich = band(&rich, 2.0 * freq);
        assert!(
            h2_rich > h2_plain * 5.0 + 0.01,
            "2 次谐波必须可闻：plain={h2_plain} rich={h2_rich}"
        );
        assert!(band(&rich, 3.0 * freq) > 0.01, "3 次谐波必须存在");
        // 微失谐拍音：平推包络下短窗 RMS 必须有波动（plain 近恒定）。
        let rms_fluct = |buf: &[f32]| -> f32 {
            let w = (0.1 * sr as f32) as usize;
            let mut rms = Vec::new();
            let mut i = 0;
            while i + w <= buf.len() {
                rms.push((buf[i..i + w].iter().map(|v| v * v).sum::<f32>() / w as f32).sqrt());
                i += w / 2;
            }
            let m = rms.iter().sum::<f32>() / rms.len() as f32;
            (rms.iter().map(|v| (v - m).powi(2)).sum::<f32>() / rms.len() as f32).sqrt()
        };
        let fp = rms_fluct(&plain);
        let fr = rms_fluct(&rich);
        assert!(fr > 0.01, "微失谐必须产生包络拍音，实测 RMS 波动 {fr}");
        assert!(fr > fp * 3.0, "富化音符包络波动必须远大于平面对照：plain={fp} rich={fr}");
        // 确定性：两次渲染逐位一致。
        let rich2 = render_rich(Waveform::Sine, freq, &env, 1.0, frames, sr, &PAD_COPIES);
        for (x, y) in rich.iter().zip(rich2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "富化渲染必须逐位确定");
        }
    }

    #[test]
    fn reverb_tail_energy_decays_monotonically() {
        // 混响尾能量衰减单调性：脉冲串后接静音，逐窗能量必须单调不增。
        let sr = 16_000u32;
        let mut rev = StereoReverb::new(sr);
        let n = (1.6 * sr as f32) as usize;
        let mut input = vec![0.0_f32; n];
        for x in input.iter_mut().take((0.05 * sr as f32) as usize) {
            *x = 0.8;
        }
        let (mut wl, mut wr) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        rev.process_block(&input, &mut wl, &mut wr);
        let win = (0.25 * sr as f32) as usize;
        let start = (0.05 * sr as f32) as usize; // 输入结束处起测尾音
        let mut prev = f32::INFINITY;
        let mut first = 0.0_f32;
        let mut w = start;
        let mut k = 0;
        while w + win <= n {
            let e: f32 = wl[w..w + win].iter().map(|v| v * v).sum();
            if k == 0 {
                first = e;
            }
            assert!(
                e <= prev * 1.08 + 1e-9,
                "混响尾能量必须单调衰减（窗 {k}：{e} > 前窗 {prev}）"
            );
            prev = e;
            w += win;
            k += 1;
        }
        assert!(first > 0.0, "混响必须有可听尾音");
        assert!(prev < first * 0.02, "1.2s 内尾音应衰减到 2% 以下，实测 {}", prev / first);
    }

    #[test]
    fn reverb_wet_differs_between_channels() {
        // 立体声像：左右混响抽头必须去相关（L 湿区 ≠ R 湿区）。
        let sr = 16_000u32;
        let mut rev = StereoReverb::new(sr);
        let n = (0.5 * sr as f32) as usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let (mut wl, mut wr) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        rev.process_block(&input, &mut wl, &mut wr);
        let mut diff = 0_usize;
        for k in 0..wl.len() {
            if (wl[k] - wr[k]).abs() > 1e-4 {
                diff += 1;
            }
        }
        assert!(diff > n / 10, "左右混响抽头必须去相关（立体声像），实测差异样本 {diff}/{n}");
        let dot: f32 = wl.iter().zip(wr.iter()).map(|(a, b)| a * b).sum();
        let el: f32 = wl.iter().map(|v| v * v).sum();
        let er: f32 = wr.iter().map(|v| v * v).sum();
        let corr = dot / (el * er).sqrt().max(1e-12);
        assert!(corr.abs() < 0.99, "左右湿声必须去相关，实测相关 {corr}");
    }

    #[test]
    fn soft_knee_bounds_peak_and_preserves_mids() {
        // 线性区恒等（中频段动态保留）。
        for &x in &[0.0_f32, 0.1, 0.35, 0.7, -0.5, -0.7] {
            assert_eq!(soft_knee(x).to_bits(), x.to_bits(), "|x|≤0.7 必须恒等");
        }
        // 峰值渐近 ≤1 且全程严格单调（无硬 clip）。
        let mut prev = -f32::INFINITY;
        for &x in &[-8.0_f32, -2.0, -1.0, -0.7, 0.0, 0.7, 1.0, 2.0, 8.0, f32::MAX] {
            let y = soft_knee(x);
            assert!(y.abs() <= 1.0, "软限幅输出必须 ≤1，实测 {y}");
            assert!(y > prev, "软限幅必须严格单调：{x} → {y} ≤ {prev}");
            prev = y;
        }
        // 中频段相对动态保留：0.5 / 0.2 的比例不变（未被压死）。
        assert_eq!(soft_knee(0.5) / soft_knee(0.2), 0.5 / 0.2);
        // 膝点 C1 平滑（无导数断裂）：膝点两侧单侧斜率接近（线性区 = 1）。
        let eps = 0.001_f32;
        let slope_below =
            (soft_knee(SOFT_KNEE) - soft_knee(SOFT_KNEE - eps)) / eps;
        let slope_above =
            (soft_knee(SOFT_KNEE + eps) - soft_knee(SOFT_KNEE)) / eps;
        assert!(
            (slope_above - slope_below).abs() < 0.05,
            "膝点必须 C1 平滑（无导数断裂），实测斜率 {slope_below} → {slope_above}"
        );
        // 非法值不 panic：NaN 归零、±inf 饱和 ±1。
        assert_eq!(soft_knee(f32::NAN), 0.0);
        assert_eq!(soft_knee(f32::INFINITY), 1.0);
        assert_eq!(soft_knee(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn humanization_bounded_and_deterministic() {
        // 有界：onset ±12ms、增益 ±8%（全曲域采样）。
        for bar in 0..45_u64 {
            for step in 0..16_u64 {
                for voice in [VOICE_PAD, VOICE_ARP, VOICE_BASS, VOICE_CHORD] {
                    let o = onset_offset_sec(bar, step, voice);
                    assert!(o.abs() <= HUMAN_ONSET_SEC + 1e-9, "onset 偏移越界：{o}");
                    let g = gain_jitter(bar, step, voice);
                    assert!((g - 1.0).abs() <= HUMAN_GAIN + 1e-6, "增益抖动越界：{g}");
                }
            }
        }
        // 确定：同键恒同值。
        assert_eq!(
            onset_offset_sec(3, 7, VOICE_ARP).to_bits(),
            onset_offset_sec(3, 7, VOICE_ARP).to_bits()
        );
        // 真实变化：不同键必须产生不同值（非恒定函数）。
        let onsets: HashSet<u32> = (0..200_u64)
            .map(|k| onset_offset_sec(k / 16, k % 16, VOICE_ARP).to_bits())
            .collect();
        assert!(onsets.len() > 150, "onset 抖动必须真实变化，实测 {} 种", onsets.len());
        let gains: HashSet<u32> = (0..200_u64)
            .map(|k| gain_jitter(k / 16, k % 16, VOICE_ARP).to_bits())
            .collect();
        assert!(gains.len() > 150, "增益抖动必须真实变化，实测 {} 种", gains.len());
        // onset 与增益抖动相互独立（不同哈希盐）。
        let pairs: HashSet<(u32, u32)> = (0..200_u64)
            .map(|k| {
                (
                    onset_offset_sec(k, 0, 0).to_bits(),
                    gain_jitter(k, 0, 0).to_bits(),
                )
            })
            .collect();
        assert!(pairs.len() > 150, "onset 与增益抖动必须独立，实测 {} 种", pairs.len());
    }

    #[test]
    fn drum_swing_moves_offbeat_eighths_only() {
        // swing：反拍八分（s%4==2）后移 6% 个十六分；kicks / 重音不动。
        let sr = 16_000u32;
        let step_frames = ((60.0 / BGM_BPM / 4.0) * sr as f32).round() as usize; // 2000
        let slots = 8;
        let mut buf = vec![0.0_f32; slots * step_frames];
        // 模拟真实网格：鼓点全部落在偶数 16 分步（kick s%4==0 / hat s%4==2）。
        for s in 0..slots {
            if s % 2 == 0 {
                for j in 0..32 {
                    buf[s * step_frames + j] = 0.5;
                }
            }
        }
        let out = apply_drum_swing(&buf, BGM_BPM, sr);
        let shift = (SWING_SHIFT_RATIO * step_frames as f32).round() as usize; // 120
        assert_eq!(out.len(), buf.len(), "swing 不得改变母带长度");
        for s in 0..slots {
            let base = s * step_frames;
            if s % 4 == 0 {
                // kick 位保持网格（无移动、无槽头淡入）。
                assert!((out[base] - 0.5).abs() < 1e-6, "kick 位（s%4==0）不得移动：s={s}");
            } else if s % 4 == 2 {
                // 反拍 hat：原位清空、内容后移 6%。
                assert!(out[base].abs() < 1e-6, "反拍 hat 原位必须清空：s={s}");
                let moved = &out[base + shift..base + shift + 32];
                assert!(
                    moved.iter().any(|v| (v - 0.5).abs() < 1e-3),
                    "反拍 hat 必须后移 6% 个十六分：s={s}"
                );
            } else {
                // 奇数 16 分步无鼓点：整槽必须静音。
                assert!(
                    out[base..base + step_frames].iter().all(|v| v.abs() < 1e-6),
                    "奇数 16 分步不得有内容：s={s}"
                );
            }
        }
    }

    #[test]
    fn section_gains_smooth_has_no_steps() {
        // 段落动态：小节边界无阶梯跳变（smoothstep C1）——逐 1ms 扫描，
        // 增益一阶差分有界；对照 v1 阶梯版在边界处 = 全摆幅。
        let bar_sec = 4.0 * 60.0 / BGM_BPM;
        let dt = 0.001_f32;
        let mut prev = section_gains_at(0.0);
        let mut t = dt;
        while t < 12.0 * bar_sec {
            let g = section_gains_at(t);
            assert!(
                (g.pad - prev.pad).abs() < 0.05,
                "pad 增益在 t={t:.3}s 跳变（阶梯 click）"
            );
            assert!(
                (g.arp - prev.arp).abs() < 0.05,
                "arp 增益在 t={t:.3}s 跳变（阶梯 click）"
            );
            assert!(
                (g.bass - prev.bass).abs() < 0.05,
                "bass 增益在 t={t:.3}s 跳变（阶梯 click）"
            );
            prev = g;
            t += dt;
        }
        // 小节边界两侧增益必须接近（连续）。
        for bar in 1..12_usize {
            let t0 = bar as f32 * bar_sec;
            let before = section_gains_at(t0 - 0.001);
            let after = section_gains_at(t0 + 0.001);
            assert!((after.pad - before.pad).abs() < 0.01, "小节 {bar} 边界 pad 必须平滑");
            assert!((after.arp - before.arp).abs() < 0.01, "小节 {bar} 边界 arp 必须平滑");
            assert!((after.bass - before.bass).abs() < 0.01, "小节 {bar} 边界 bass 必须平滑");
        }
    }

    #[test]
    fn synth_bgm_stereo_image_present() {
        // 立体声像：L/R 微偏移 + 不同混响抽头 ⇒ 绝大多数帧 L≠R，且两声道
        // 同源相关（非反相 / 非噪声）。
        let bgm = synth_bgm(6.0, 16_000);
        let s = bgm.as_slice();
        let mut diff = 0_usize;
        for f in 0..bgm.frames() {
            if (s[2 * f] - s[2 * f + 1]).abs() > 1e-4 {
                diff += 1;
            }
        }
        assert!(
            diff > bgm.frames() / 2,
            "立体声像必须存在（L/R 大范围不同），实测 {diff}/{}",
            bgm.frames()
        );
        let dot: f32 = (0..bgm.frames()).map(|f| s[2 * f] * s[2 * f + 1]).sum();
        let el: f32 = s.iter().step_by(2).map(|v| v * v).sum();
        let er: f32 = s.iter().skip(1).step_by(2).map(|v| v * v).sum();
        let corr = dot / (el * er).sqrt().max(1e-12);
        assert!(
            corr > 0.0 && corr < 0.95,
            "L/R 必须同源但被微偏移去相关（0<corr<0.95），实测 {corr}"
        );
    }
}
