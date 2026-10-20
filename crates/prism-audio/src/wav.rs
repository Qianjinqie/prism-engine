//! WAV 读写：16/24/32 bit PCM，逐字节往返一致。
//!
//! 规范：RIFF 容器，`fmt ` 块 16 字节 PCM 头 + `data` 块；未知块按 RIFF
//! 词对齐跳过。编码格式：位深 16/24 用整数 PCM（format tag 1），位深 32 用
//! IEEE float（format tag 3，WAV 规范对 32bit 的标准载体，也是引擎原生格式）。
//!
//! # 往返一致性（判据：逐字节）
//!
//! - **16/24 bit**：编码 `i = round(clamp(x) · 2^(bits−1))`（IEEE 定义整数
//!   舍入，确定性）；解码 `x = i / 2^(bits−1)`——除以 2 的幂在 IEEE 中**精确**，
//!   f32 有效位（24 bit）恰好容纳全部 i24/i16 值，故 `encode(decode(b)) == b`
//!   逐字节成立。
//! - **32 bit**：f32 位模式原样拷贝，天然逐字节一致。
//!
//! 全程无 libm、无环境依赖：同输入跨平台逐字节一致。

use crate::buffer::PcmBuffer;
use crate::{AudioError, AudioResult};

/// WAV 编码位深。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    /// 16 bit 整数 PCM（format tag 1）。
    B16,
    /// 24 bit 整数 PCM（format tag 1）。
    B24,
    /// 32 bit IEEE float（format tag 3，引擎原生）。
    B32,
}

impl BitDepth {
    /// 每样本字节数。
    pub fn bytes_per_sample(self) -> usize {
        match self {
            BitDepth::B16 => 2,
            BitDepth::B24 => 3,
            BitDepth::B32 => 4,
        }
    }

    /// WAV format tag：1 = 整数 PCM，3 = IEEE float。
    fn format_tag(self) -> u16 {
        match self {
            BitDepth::B32 => 3,
            _ => 1,
        }
    }
}

/// 编码为 WAV 字节流（RIFF 小端；固定块序 fmt → data，无附加块）。
pub fn encode_wav(buf: &PcmBuffer, depth: BitDepth) -> Vec<u8> {
    let bps = depth.bytes_per_sample();
    let channels = buf.channels() as usize;
    let data_len = buf.frames() * channels * bps;
    let block_align = channels * bps;
    let byte_rate = buf.sample_rate() as usize * block_align;

    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36_u32 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&depth.format_tag().to_le_bytes());
    out.extend_from_slice(&(buf.channels()).to_le_bytes());
    out.extend_from_slice(&buf.sample_rate().to_le_bytes());
    out.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&((bps * 8) as u16).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());

    for &x in buf.as_slice() {
        match depth {
            // round 为 IEEE 定义的远离零取整：跨平台逐位一致。
            BitDepth::B16 => {
                let v = (x.clamp(-1.0, 1.0) * 32_768.0).round() as i32;
                let v = v.clamp(-32_768, 32_767) as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
            BitDepth::B24 => {
                let v = (x.clamp(-1.0, 1.0) * 8_388_608.0).round() as i32;
                let v = v.clamp(-8_388_608, 8_388_607);
                let bytes = v.to_le_bytes();
                out.extend_from_slice(&bytes[..3]); // 二进制补码低 3 字节
            }
            BitDepth::B32 => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        }
    }
    out
}

/// 解码 WAV 字节流（只接受 PCM tag 1 × 16/24 bit 与 IEEE float tag 3 ×
/// 32 bit；结构非法报 `WAV1001`，格式不支持报 `WAV1002`）。
pub fn decode_wav(bytes: &[u8]) -> AudioResult<PcmBuffer> {
    let invalid = |problem: String, fix: String| {
        AudioError::new("WAV1001", "wav::decode_wav", problem, fix)
    };
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid(
            "缺少 RIFF/WAVE 魔数".into(),
            "输入必须是标准 RIFF WAVE 文件（RIFF….WAVE…）".into(),
        ));
    }

    // 块遍历：fmt 必须先于 data；未知块按 RIFF 词对齐跳过。
    let (mut tag, mut channels, mut sample_rate, mut bits) = (0_u16, 0_u16, 0_u32, 0_u16);
    let (mut has_fmt, mut data) = (false, None::<(&[u8],)>);
    let mut pos = 12_usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = bytes.get(pos + 8..pos + 8 + size).ok_or_else(|| {
            invalid(format!("块 {id:?} 声明 {size} 字节但文件只剩 {}", bytes.len() - pos - 8),
                "文件被截断：补全或重新导出 WAV".into())
        })?;
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(invalid(
                        format!("fmt 块 {} 字节，不足 16", body.len()),
                        "fmt 块必须 ≥16 字节（PCM 头）".into(),
                    ));
                }
                tag = u16::from_le_bytes([body[0], body[1]]);
                channels = u16::from_le_bytes([body[2], body[3]]);
                sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                bits = u16::from_le_bytes([body[14], body[15]]);
                has_fmt = true;
            }
            b"data" => {
                if !has_fmt {
                    return Err(invalid(
                        "data 块出现在 fmt 块之前".into(),
                        "WAV 规范要求 fmt 先于 data；请用标准编码器导出".into(),
                    ));
                }
                data = Some((body,));
                break; // data 之后不再关心（写出器也不产生附加块）
            }
            _ => {}
        }
        pos += 8 + size + (size & 1); // RIFF 块按 2 字节对齐
    }

    let (body,) = data.ok_or_else(|| invalid("缺少 data 块".into(), "输入必须含 data 块".into()))?;
    let (tag_ok, bps) = match (tag, bits) {
        (1, 16) => (true, 2_usize),
        (1, 24) => (true, 3),
        (3, 32) => (true, 4),
        _ => (false, 0),
    };
    if !tag_ok {
        return Err(AudioError::new(
            "WAV1002",
            "wav::decode_wav",
            format!("不支持的格式：format tag {tag} × 位深 {bits}"),
            "支持 16/24 bit 整数 PCM（tag 1）与 32 bit IEEE float（tag 3）",
        ));
    }
    if channels < 1 || sample_rate < 1 {
        return Err(invalid(
            format!("声道数 {channels} 或采样率 {sample_rate} 非法"),
            "声道数与采样率必须 ≥ 1".into(),
        ));
    }
    let block_align = channels as usize * bps;
    if body.len() % block_align != 0 {
        return Err(invalid(
            format!("data 长度 {} 不是块对齐 {} 的整数倍", body.len(), block_align),
            "data 块长度必须为 block_align 整数倍；文件可能损坏".into(),
        ));
    }

    let mut samples = Vec::with_capacity(body.len() / bps);
    match bits {
        16 => body.chunks_exact(2).for_each(|c| {
            let v = i16::from_le_bytes([c[0], c[1]]);
            samples.push(v as f32 / 32_768.0); // 除以 2 的幂：IEEE 精确
        }),
        24 => body.chunks_exact(3).for_each(|c| {
            // 符号扩展：低 3 字节 → i32 → 除以 2²³（f32 有效位恰好容纳，精确）。
            let mut v = c[0] as i32 | (c[1] as i32) << 8 | (c[2] as i32) << 16;
            if v & 0x0080_0000 != 0 {
                v |= !0x00FF_FFFF;
            }
            samples.push(v as f32 / 8_388_608.0);
        }),
        _ => body.chunks_exact(4).for_each(|c| {
            samples.push(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }),
    }
    PcmBuffer::from_interleaved(sample_rate, channels, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 三种位深：写 → 读 → 写逐字节一致（判据）。
    #[test]
    fn roundtrip_is_byte_exact_for_16_24_32() {
        for depth in [BitDepth::B16, BitDepth::B24, BitDepth::B32] {
            let data: Vec<f32> = (0..97) // 质数帧数防步长巧合
                .map(|i| (i as f32 * 0.05).sin() * 0.9 - 0.3)
                .chain([1.0, -1.0, 0.0]) // 满幅钳位边界
                .collect();
            let buf = PcmBuffer::from_interleaved(48_000, 2, data).unwrap();
            let bytes = encode_wav(&buf, depth);
            let decoded = decode_wav(&bytes).expect("自产 WAV 必可解码");
            assert_eq!(decoded.sample_rate(), 48_000);
            assert_eq!(decoded.channels(), 2);
            assert_eq!(decoded.frames(), buf.frames());
            let again = encode_wav(&decoded, depth);
            assert_eq!(again, bytes, "{depth:?} 往返必须逐字节一致");
        }
    }

    /// 16/24 bit 解码值精确落在量化栅格上（i / 2^(bits-1)）。
    #[test]
    fn decode_values_lie_on_quantization_grid() {
        let buf = PcmBuffer::from_mono(8_000, vec![0.25, -1.0, 1.0, 0.0]).unwrap();
        let b16 = decode_wav(&encode_wav(&buf, BitDepth::B16)).unwrap();
        let want = [8_192_i32 as f32 / 32_768.0, -1.0, 32_767.0 / 32_768.0, 0.0];
        for (i, w) in want.iter().enumerate() {
            let got = b16.sample(i, 0).unwrap();
            assert_eq!(got.to_bits(), (*w as f32).to_bits(), "B16 栅格点 {i}");
        }
        let b24 = decode_wav(&encode_wav(&buf, BitDepth::B24)).unwrap();
        let want24 = [0.25, -1.0, 8_388_607.0 / 8_388_608.0, 0.0];
        for (i, w) in want24.iter().enumerate() {
            let got = b24.sample(i, 0).unwrap();
            assert_eq!(got.to_bits(), (*w as f32).to_bits(), "B24 栅格点 {i}");
        }
        let b32 = decode_wav(&encode_wav(&buf, BitDepth::B32)).unwrap();
        assert_eq!(b32.as_slice(), buf.as_slice(), "B32 位模式原样");
    }

    /// 头字段与规范一致（RIFF 尺寸 / fmt / data 声明）。
    #[test]
    fn header_matches_spec() {
        let buf = PcmBuffer::from_interleaved(44_100, 2, vec![0.0; 6]).unwrap();
        let bytes = encode_wav(&buf, BitDepth::B16);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1, "整数 PCM tag");
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 44_100);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 44_100 * 2 * 2);
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 12);
        assert_eq!(bytes.len(), 44 + 12);
        // 32bit 用 IEEE float tag 3。
        let bytes32 = encode_wav(&buf, BitDepth::B32);
        assert_eq!(u16::from_le_bytes(bytes32[20..22].try_into().unwrap()), 3);
    }

    /// 非法输入三段式拒绝：坏魔数 / 截断 / 未知位深 / data 先于 fmt。
    #[test]
    fn malformed_inputs_are_rejected() {
        let good = encode_wav(&PcmBuffer::from_mono(8_000, vec![0.5]).unwrap(), BitDepth::B16);
        assert_eq!(decode_wav(b"not a wav").unwrap_err().code, "WAV1001");
        let mut truncated = good.clone();
        truncated.truncate(40);
        assert_eq!(decode_wav(&truncated).unwrap_err().code, "WAV1001");
        let mut bad_depth = good.clone();
        bad_depth[34] = 12; // bits = 12
        assert_eq!(decode_wav(&bad_depth).unwrap_err().code, "WAV1002");
        let mut no_fmt = good.clone();
        no_fmt[12..16].copy_from_slice(b"LIST"); // 唯一 fmt 被改名
        assert_eq!(decode_wav(&no_fmt).unwrap_err().code, "WAV1001");
    }

    /// 未知块跳过（词对齐）后仍可解码。
    #[test]
    fn unknown_chunks_are_skipped() {
        let mut bytes = encode_wav(&PcmBuffer::from_mono(8_000, vec![0.5, -0.5]).unwrap(), BitDepth::B16);
        // 在 WAVE 后插入 3 字节奇数长 LIST 块（+ 对齐垫 1 字节）。
        let extra: Vec<u8> = b"LIST".iter().copied()
            .chain(3_u32.to_le_bytes())
            .chain([0xAA, 0xBB, 0xCC, 0x00])
            .collect();
        bytes.splice(12..12, extra);
        let decoded = decode_wav(&bytes).unwrap();
        assert_eq!(decoded.frames(), 2);
        assert_eq!(decoded.sample(0, 0), Some(0.5));
    }
}
