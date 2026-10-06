//! MIDI 文件导入（B13 作曲系统·外部输入；手写 SMF 解析，零第三方依赖）。
//!
//! # 格式（Standard MIDI File 1.0）
//!
//! SMF = `MThd` 头块 + 一个或多个 `MTrk` 轨块：
//!
//! - **头块** `"MThd" + u32(len=6) + u16(format) + u16(ntrks) + u16(division)`；
//!   `division` 最高位 = 0 时低 15 位 = 每四分音符 tick 数（PPQ），= 1 时为
//!   SMPTE 定时（本解析器仅支持 PPQ，SMPTE 返回 `MID1003`）。
//! - **轨块** `"MTrk" + u32(len) + 事件流`；每个事件 = `delta-time`（VLQ）+
//!   事件体。事件体分三类：MIDI 通道事件（8n/9n/An/Bn/Cn/Dn/En，含
//!   **running status** 复用）、Meta 事件（`FF type VLQ-len data`，含
//!   `FF 51 03` 设速度、`FF 2F 00` 轨道结束）、SysEx 事件（`F0`/`F7`）。
//! - **VLQ**：7 位/字节、最高位置 1 表示续接、末字节最高位 0、大端序。
//!
//! # 确定性
//!
//! 解析为纯函数（输入字节 → 输出事件表），delta-time 按轨内累积成绝对 tick，
//! 再按 `division` 换算为帧（`frame = round(tick / ppq · 每四分音符帧数)`）。
//! 音符配对采用**同音高 + 同通道**的 note-on/note-off 栈（后进先出，确定性）。

use crate::theory::midi_to_freq;
use crate::AudioError;
use crate::AudioResult;

/// 一个已解析的音符事件（帧域）。
#[derive(Debug, Clone, PartialEq)]
pub struct MidiNote {
    /// 起始帧。
    pub start_frame: usize,
    /// 时值（帧数）。
    pub frames: usize,
    /// MIDI 通道（0..16）。
    pub channel: u8,
    /// MIDI 音高。
    pub midi: i32,
    /// 力度（0..128）。
    pub velocity: u8,
    /// 频率（= [`midi_to_freq`]`(midi)`）。
    pub freq: f32,
    /// 所属轨序（0-based）。
    pub track: usize,
}

/// 解析产物：音符表 + 头块元数据。
#[derive(Debug, Clone, PartialEq)]
pub struct MidiFile {
    /// SMF 格式（0 / 1 / 2）。
    pub format: u16,
    /// 每四分音符 tick 数（PPQ）。
    pub ppq: u16,
    /// 首见 `FF 51` 设速度（微秒/四分音符；缺省 500000 = 120 BPM）。
    pub tempo_us: u32,
    /// 采样率（换算帧用）。
    pub sample_rate: u32,
    /// 音符事件（构造序 = 轨序 → 事件序）。
    pub notes: Vec<MidiNote>,
    /// 总帧数（= 最大音符结束帧；空文件为 0）。
    pub total_frames: usize,
}

/// 读取大端 u16。
fn be_u16(d: &[u8], at: usize) -> AudioResult<u16> {
    if at + 2 > d.len() {
        return Err(parse_err(at, "读取 u16 越界"));
    }
    Ok(((d[at] as u16) << 8) | d[at + 1] as u16)
}

/// 读取大端 u32。
fn be_u32(d: &[u8], at: usize) -> AudioResult<u32> {
    if at + 4 > d.len() {
        return Err(parse_err(at, "读取 u32 越界"));
    }
    Ok(((d[at] as u32) << 24) | ((d[at + 1] as u32) << 16) | ((d[at + 2] as u32) << 8) | d[at + 3] as u32)
}

/// 读取 VLQ（变长量）：7 位/字节、最高位续接、大端、最多 4 字节。
/// 返回 (值, 消耗字节数)。
fn read_vlq(d: &[u8], at: usize) -> AudioResult<(u32, usize)> {
    let mut value: u32 = 0;
    let mut i = at;
    let mut count = 0;
    loop {
        if i >= d.len() {
            return Err(parse_err(at, "VLQ 在文件末尾截断"));
        }
        let b = d[i];
        value = (value << 7) | (b & 0x7F) as u32;
        i += 1;
        count += 1;
        if b & 0x80 == 0 {
            break;
        }
        if count >= 4 {
            return Err(parse_err(at, "VLQ 超过 4 字节（最大值 0x0FFFFFFF）"));
        }
    }
    Ok((value, count))
}

/// 构造三段式解析错误（MIDI 段 `MIDxxxx`）。
fn parse_err(at: usize, problem: &str) -> AudioError {
    AudioError::new(
        "MID1001",
        format!("midi 解析（字节偏移 {at}）"),
        problem,
        "确认输入为合法 Standard MIDI File（SMF 1.0：MThd + MTrk）",
    )
}

/// 单轨解析中间态：某音高的未配对 note-on 累计（同通道）。
#[derive(Clone, Copy)]
struct Pending {
    /// note-on 的绝对 tick。
    tick: u64,
    /// 力度。
    velocity: u8,
    /// 是否已占用（避免用 Option 数组的冗余）。
    active: bool,
}

/// 解析 SMF 字节流为音符事件（`sample_rate ≥ 1`，否则 `MID1002`）。
///
/// 支持 format 0/1/2 的公共子集：逐轨解析、`note-on (vel>0)` / `note-off` 与
/// `note-on (vel=0)` 配对、running status、`FF 51` 速度、`FF 2F` 轨结束。
/// 与速度无关的音符时值以 PPQ 换算为帧（tempo 仅记录）。
pub fn parse(bytes: &[u8], sample_rate: u32) -> AudioResult<MidiFile> {
    if sample_rate < 1 {
        return Err(AudioError::new(
            "MID1002",
            format!("midi::parse(sample_rate={sample_rate})"),
            "采样率必须 ≥ 1",
            "给出有效采样率（如 44100）",
        ));
    }
    // 头块。
    if bytes.len() < 14 || &bytes[0..4] != b"MThd" {
        return Err(parse_err(0, "缺少 'MThd' 头块标识"));
    }
    let header_len = be_u32(bytes, 4)? as usize;
    let format = be_u16(bytes, 8)?;
    let ntrks = be_u16(bytes, 10)?;
    let division = be_u16(bytes, 12)?;
    if division & 0x8000 != 0 {
        return Err(AudioError::new(
            "MID1003",
            format!("midi::parse(division=0x{division:04X})"),
            "SMPTE 定时（division 最高位 = 1）暂不支持",
            "使用每四分音符 tick 数（PPQ）定时的 MIDI 文件，或以 24 fps 重新导出",
        ));
    }
    let ppq = division & 0x7FFF;
    if ppq == 0 {
        return Err(parse_err(12, "division（PPQ）为 0，无法换算时值"));
    }
    let (mut tempo_us, mut pos) = (500_000_u32, 8 + header_len);
    let mut notes: Vec<MidiNote> = Vec::new();

    for track in 0..ntrks as usize {
        if pos + 8 > bytes.len() || &bytes[pos..pos + 4] != b"MTrk" {
            // 缺轨块视为文件提前结束（宽容：已解析部分保留）。
            break;
        }
        let trk_len = be_u32(bytes, pos + 4)? as usize;
        let trk_start = pos + 8;
        let trk_end = (trk_start + trk_len).min(bytes.len());
        let mut i = trk_start;
        let mut abs_tick: u64 = 0;
        let mut running_status: Option<u8> = None;
        // 未配对 note-on：按 (通道 × 音高) 索引，LIFO 配对。
        let mut pending: Vec<Pending> = vec![Pending { tick: 0, velocity: 0, active: false }; 16 * 128];

        while i < trk_end {
            let (delta, consumed) = read_vlq(bytes, i)?;
            i += consumed;
            abs_tick += delta as u64;
            if i >= trk_end {
                break;
            }
            let mut status = bytes[i];
            let mut data_at = i + 1;
            // running status：字节 < 0x80 表示复用上一 status，当前字节是数据。
            if status < 0x80 {
                let Some(rs) = running_status else {
                    return Err(parse_err(i, "事件缺 status 且无 running status 可复用"));
                };
                status = rs;
                data_at = i;
            } else {
                running_status = Some(status);
            }

            if status == 0xFF {
                // Meta 事件：FF type VLQ-len data。
                let meta_type = bytes.get(data_at).copied().unwrap_or(0xFF);
                let (mlen, mc) = read_vlq(bytes, data_at + 1)?;
                let mstart = data_at + 1 + mc;
                let mend = mstart + mlen as usize;
                if mend > bytes.len() {
                    return Err(parse_err(mstart, "meta 事件数据越界"));
                }
                if meta_type == 0x51 && mlen >= 3 {
                    tempo_us = ((bytes[mstart] as u32) << 16)
                        | ((bytes[mstart + 1] as u32) << 8)
                        | bytes[mstart + 2] as u32;
                }
                i = mend;
                if meta_type == 0x2F {
                    break; // End of Track
                }
                continue;
            }
            if status == 0xF0 || status == 0xF7 {
                // SysEx：F0/F7 VLQ-len data。
                let (slen, sc) = read_vlq(bytes, data_at)?;
                i = data_at + sc + slen as usize;
                continue;
            }

            // MIDI 通道事件。
            let kind = status & 0xF0;
            let channel = status & 0x0F;
            let data_len = if kind == 0xC0 || kind == 0xD0 { 1 } else { 2 };
            if data_at + data_len > bytes.len() {
                return Err(parse_err(data_at, "通道事件数据越界"));
            }
            let d1 = bytes[data_at];
            let d2 = if data_len == 2 { bytes[data_at + 1] } else { 0 };
            i = data_at + data_len;

            match kind {
                0x90 if d2 > 0 => {
                    // Note-on：登记未配对 note。
                    let slot = channel as usize * 128 + (d1 & 0x7F) as usize;
                    pending[slot] = Pending { tick: abs_tick, velocity: d2, active: true };
                }
                0x80 | 0x90 => {
                    // Note-off / note-on(vel=0)：与同槽 note-on 配对。
                    let slot = channel as usize * 128 + (d1 & 0x7F) as usize;
                    if pending[slot].active {
                        let p = pending[slot];
                        pending[slot].active = false;
                        let start_frame = tick_to_frame(p.tick, ppq, sample_rate);
                        let end_frame = tick_to_frame(abs_tick, ppq, sample_rate);
                        notes.push(MidiNote {
                            start_frame,
                            frames: end_frame.saturating_sub(start_frame).max(1),
                            channel,
                            midi: (d1 & 0x7F) as i32,
                            velocity: p.velocity,
                            freq: midi_to_freq((d1 & 0x7F) as i32),
                            track,
                        });
                    }
                }
                _ => {} // 其他通道事件（aftertouch/cc/program/pitch）解析跳过。
            }
        }
        // 轨内未收尾的 note-on：按时值 0（至少 1 帧）补一个短音。
        for (slot_idx, p) in pending.iter().enumerate() {
            if p.active {
                let midi = (slot_idx % 128) as i32;
                let channel = (slot_idx / 128) as u8;
                let start_frame = tick_to_frame(p.tick, ppq, sample_rate);
                notes.push(MidiNote {
                    start_frame,
                    frames: 1,
                    channel,
                    midi,
                    velocity: p.velocity,
                    freq: midi_to_freq(midi),
                    track,
                });
            }
        }
        pos = trk_end;
    }

    let total_frames = notes.iter().map(|n| n.start_frame + n.frames).max().unwrap_or(0);
    Ok(MidiFile { format, ppq, tempo_us, sample_rate, notes, total_frames })
}

/// tick → 帧：`round(tick / ppq · 每四分音符帧数)`（`每四分音符帧数 = 60/bpm·sr`
/// 与 tempo 无关的 PPQ 换算，保持纯函数；tempo 仅作元数据）。
///
/// 采用固定参考速度 120 BPM（`60/120·sr`）以维持确定性——导入仅做速率无关的
/// 结构还原，具体演奏速度由导出方（score/drums）另给 `bpm` 决定。
fn tick_to_frame(tick: u64, ppq: u16, sample_rate: u32) -> usize {
    let frames_per_qn = sample_rate as f64 * 60.0 / 120.0; // 120 BPM 参考
    let f = tick as f64 / ppq as f64 * frames_per_qn;
    f.round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小合法 SMF：format 0、PPQ 96、单轨，
    /// `note-on 60 @ tick0, note-off @ tick96`（1000 → C5 音，这里用 60）。 
    fn minimal_smf() -> Vec<u8> {
        let mut f: Vec<u8> = Vec::new();
        // MThd
        f.extend_from_slice(b"MThd");
        f.extend_from_slice(&6u32.to_be_bytes());
        f.extend_from_slice(&0u16.to_be_bytes()); // format 0
        f.extend_from_slice(&1u16.to_be_bytes()); // ntrks
        f.extend_from_slice(&96u16.to_be_bytes()); // ppq
        // MTrk
        let mut ev: Vec<u8> = Vec::new();
        // delta 0, note-on ch0 60 vel100
        ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
        // delta 96 (VLQ 0x60), note-off ch0 60 vel0
        ev.extend_from_slice(&[0x60, 0x80, 60, 0]);
        // delta 0, end of track
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        f.extend_from_slice(b"MTrk");
        f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
        f.extend_from_slice(&ev);
        f
    }

    /// 基本解析：头块元数据、音符配对、时值（tick 96 = 1 四分音符）。
    #[test]
    fn parse_minimal_note() {
        let f = parse(&minimal_smf(), 44_100).unwrap();
        assert_eq!(f.format, 0);
        assert_eq!(f.ppq, 96);
        assert_eq!(f.notes.len(), 1);
        let n = &f.notes[0];
        assert_eq!(n.midi, 60);
        assert_eq!(n.velocity, 100);
        assert_eq!(n.channel, 0);
        assert_eq!(n.track, 0);
        assert_eq!(n.start_frame, 0);
        // tick 96 @120bpm 44.1k = 0.5s = 22050 帧。
        assert_eq!(n.frames, 22_050, "四分音符 @120bpm 44.1k = 22050 帧");
        assert_eq!(n.freq, midi_to_freq(60));
        assert_eq!(f.total_frames, 22_050);
    }

    /// VLQ 解码：规范例证（0x7F→127、0x81 0x7F→255、0x82 0x80 0x00→32768）。
    #[test]
    fn vlq_decoding_matches_spec() {
        assert_eq!(read_vlq(&[0x7F], 0).unwrap(), (127, 1));
        assert_eq!(read_vlq(&[0x81, 0x7F], 0).unwrap(), (255, 2));
        assert_eq!(read_vlq(&[0x82, 0x80, 0x00], 0).unwrap(), (32768, 3));
        assert_eq!(read_vlq(&[0xC0, 0x00], 0).unwrap(), (8192, 2));
        assert_eq!(read_vlq(&[0xFF, 0xFF, 0x7F], 0).unwrap(), (0x1F_FFFF, 3));
        assert_eq!(read_vlq(&[0x83], 0).unwrap_err().code, "MID1001");
    }

    /// running status + note-on(vel=0) 结束音符 + delta-time 累积。
    #[test]
    fn running_status_and_vel0_noteoff() {
        let mut f: Vec<u8> = Vec::new();
        f.extend_from_slice(b"MThd");
        f.extend_from_slice(&6u32.to_be_bytes());
        f.extend_from_slice(&1u16.to_be_bytes());
        f.extend_from_slice(&1u16.to_be_bytes());
        f.extend_from_slice(&96u16.to_be_bytes());
        let mut ev: Vec<u8> = Vec::new();
        // delta 0 note-on ch0 60 vel100（status 0x90）
        ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
        // delta 48 note-on 64 vel80（**复用 running status**：仅数据字节）
        ev.extend_from_slice(&[0x30, 64, 80]);
        // delta 48 note-on 60 vel0（= note-off，且复用 status）
        ev.extend_from_slice(&[0x30, 60, 0]);
        // delta 0 note-off ch0 64 vel0
        ev.extend_from_slice(&[0x00, 0x80, 64, 0]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        f.extend_from_slice(b"MTrk");
        f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
        f.extend_from_slice(&ev);

        let mf = parse(&f, 8_000).unwrap();
        assert_eq!(mf.notes.len(), 2, "两个音符应各配对一次");
        let n60 = mf.notes.iter().find(|n| n.midi == 60).unwrap();
        let n64 = mf.notes.iter().find(|n| n.midi == 64).unwrap();
        // tick 96 -> 帧：8k·0.5 = 4000。
        assert_eq!(n60.frames, 4_000, "60 音 @tick0–96");
        // 64 音 @tick48–96 = 48 tick = 2000 帧。
        assert_eq!(n64.start_frame, 2_000);
        assert_eq!(n64.frames, 2_000);
        assert_eq!(n64.velocity, 80);
    }

    /// tempo meta 记录；非法头块 / SMPTE / samplerate 报错。
    #[test]
    fn tempo_and_error_paths() {
        // 插一个 FF 51 03 tempo（500000 = 120bpm）。
        let mut f = minimal_smf();
        // 在轨首插入 tempo meta（需要在 MTrk 事件流里）——重建。
        let mut mthd = f[0..14].to_vec();
        let mut ev: Vec<u8> = Vec::new();
        ev.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500000
        ev.extend_from_slice(&[0x00, 0x90, 60, 100]);
        ev.extend_from_slice(&[0x60, 0x80, 60, 0]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        f.clear();
        f.append(&mut mthd);
        f.extend_from_slice(b"MTrk");
        f.extend_from_slice(&(ev.len() as u32).to_be_bytes());
        f.extend_from_slice(&ev);
        let mf = parse(&f, 8_000).unwrap();
        assert_eq!(mf.tempo_us, 500_000);
        assert_eq!(mf.notes.len(), 1);

        // 非 MThd 头。
        assert_eq!(parse(b"XXXXnope", 8_000).unwrap_err().code, "MID1001");
        // 采样率 0。
        assert_eq!(parse(&minimal_smf(), 0).unwrap_err().code, "MID1002");
        // SMPTE division（最高位 1）。
        let mut smpte = minimal_smf();
        smpte[12] = 0xE8; // 0xE8xx：SMPTE
        smpte[13] = 0x50;
        assert_eq!(parse(&smpte, 8_000).unwrap_err().code, "MID1003");
    }
}
