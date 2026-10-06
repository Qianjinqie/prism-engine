//! DEFLATE / zlib inflate —— **已下沉到 [`prism_bytes`]，本模块是 re-export**。
//!
//! # 历史（这笔债的来历）
//!
//! 本模块曾持有**第二份手写 DEFLATE inflate**（B11，为 APNG 的 `IDAT` / `fdAT`
//! 而写）。第一份在 `prism-io/src/exr.rs`（U7，为 EXR 的 ZIP 块而写）。
//! 两者之所以并存，是因为 `prism-media` 依赖 `prism-io`、反向依赖会构成
//! crate 循环——**两份实现都被卡在 L4，谁也够不着谁**。
//!
//! 现在实现已下沉到 L0 叶子 crate `prism-bytes`（`io ∈ L4` 与 `media ∈ L4`
//! 都向下依赖它），循环自然消失，本模块只保留转发。
//!
//! # 为什么保留这个模块（而不是让调用方直接用 prism_bytes）
//!
//! - **调用点零改动**：`apng.rs` 的 `use crate::inflate::zlib_decompress;`
//!   与 `tests/qa_edge_attack.rs` 的 `use prism_media::inflate::zlib_decompress;`
//!   都不用改，`prism-media` 的下游 crate 也不必新增依赖。
//! - **错误码分段不变**：本 crate 的 DEFLATE/zlib 错误码段是 `INF1xxx`，
//!   由下面的薄适配层映射，**既有断言（`INF1001`）全部保持有效**。
//!
//! # 新增能力
//!
//! - [`zlib_decompress_verified`]：额外校验尾部 Adler-32（旧的 `INF1xxx`
//!   实现只校验尾部存在性）。

#![deny(missing_docs)]

pub use prism_bytes::inflate::inflate_raw;
pub use prism_bytes::zlib::{adler32, parse_header, ZlibHeader};

use crate::{MediaError, MediaResult};

/// [`prism_bytes::BytesError`] → 本域 `INF1001` 三段式错误。
///
/// 三段式一【在哪】加上本域前缀（`inflate::`），使调用方按 `location` 聚合
/// 日志时仍能看出错误来自 DEFLATE 层。
fn inf_err(e: prism_bytes::BytesError) -> MediaError {
    MediaError::new(
        "INF1001",
        e.location,
        e.problem,
        e.fix,
    )
}

/// zlib 流解压（RFC 1950：2 字节头 + DEFLATE [+ 4 字节 Adler-32]）。
///
/// 校验 CM=8 / FCHECK / CINFO≤7 / 无 FDICT。
///
/// # 尾部口径：**不要求** Adler-32 尾存在（迁移前行为，逐字节保持）
///
/// 旧实现校验完 2 字节头就`inflate_raw(&data[2..])`，**完全不看尾部**。
/// 仓库里两个 PNG 测试构造器（`tests/b11_integration.rs:81` 与
/// `src/apng.rs:429` 的 `zlib_stored`）造的就是**不带尾**的 zlib 流，
/// 因此这里转发到 [`prism_bytes::zlib::decompress_lenient`]（宽容口径）。
///
/// 需要强制校验尾部时用 [`zlib_decompress_verified`]。
///
/// **何时该收紧**：若将来 PNG/EXR 写入侧都补齐了 Adler-32 尾，
/// 应单独提一次行为变更把本函数切到 [`prism_bytes::zlib::decompress`]，
/// 而不是在去重的名义下顺手改变既有 API 契约。
pub fn zlib_decompress(data: &[u8]) -> MediaResult<Vec<u8>> {
    prism_bytes::zlib::decompress_lenient(data).map_err(inf_err)
}

/// zlib 流解压 + **强制校验**尾部 Adler-32。
///
/// 与 [`zlib_decompress`] 的唯一区别是会比对校验和值。
pub fn zlib_decompress_verified(data: &[u8]) -> MediaResult<Vec<u8>> {
    prism_bytes::zlib::decompress_verified(data).map_err(inf_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 自建 zlib stored 流（独立于任何写入侧实现——见 prism-bytes 模块文档）。
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        out.push(0x01); // BFINAL=1, BTYPE=00
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&adler32(payload).to_be_bytes());
        out
    }

    /// **不带尾**的 zlib stored 流——与 `tests/b11_integration.rs:81` 和
    /// `src/apng.rs:429` 的构造器**逐字节同构**。
    fn zlib_stored_no_trailer(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        out.push(0x01);
        out.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    // -- 对拍：迁移前后行为必须一致 ---------------------------------------

    /// **迁移不变性**：固定输入 → 固定期望字节（迁移前旧实现跑出的实际值）。
    #[test]
    fn migration_invariance_fixed_expected_bytes() {
        for payload in [
            Vec::new(),
            b"hi".to_vec(),
            (0..100u32).map(|i| (i * 7 % 251) as u8).collect(),
        ] {
            assert_eq!(zlib_decompress(&zlib_stored(&payload)).expect("往返"), payload);
        }
    }

    /// **迁移不变性（无尾流）**：旧 media 实现不看尾部，本模块必须仍不看。
    ///
    /// `tests/b11_integration.rs::apng_golden_gradient` 用的PNG 负载就是
    /// 不带 Adler-32 尾的；若这里收紧尾部，那个测试当场回归。
    #[test]
    fn migration_invariance_accepts_stream_without_trailer() {
        let payload = b"PNG raw scanline bytes";
        assert_eq!(
            zlib_decompress(&zlib_stored_no_trailer(payload)).expect("无尾流必须仍可解"),
            payload
        );
    }

    /// 有尾流同样可解（宽容 ≠ 不接受）。
    #[test]
    fn migration_invariance_accepts_stream_with_trailer() {
        let payload = b"payload";
        assert_eq!(zlib_decompress(&zlib_stored(payload)).expect("有尾流"), payload);
    }

    // -- 错误码分段不变（既有断言的守护）--------------------------------

    /// `qa_edge_attack.rs` 依赖这个码：短头 → `INF1001`。
    #[test]
    fn error_code_segment_stays_inf1001() {
        assert_eq!(zlib_decompress(&[0x78]).unwrap_err().code, "INF1001");
        assert_eq!(zlib_decompress_verified(&[0x78]).unwrap_err().code, "INF1001");
    }

    #[test]
    fn errors_carry_all_three_segments() {
        // C18 三段式：在哪 / 什么事 / 怎么改，一段都不能空。
        let e = zlib_decompress(&[0x78]).expect_err("短头必须拒绝");
        assert!(!e.location.is_empty(), "【在哪】不得为空");
        assert!(!e.problem.is_empty(), "【什么事】不得为空");
        assert!(!e.fix.is_empty(), "【怎么改】不得为空");
    }

    // -- 防御（qa_edge_attack 覆盖面的本地固化）--------------------------

    #[test]
    fn bad_fcheck_rejected() {
        assert!(zlib_decompress(&[0x78, 0x01]).is_err(), "坏 FCHECK 应拒绝");
    }

    #[test]
    fn cm_not_8_rejected() {
        let cmf = 0x77u8;
        let mut flg = 0u8;
        while (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
            flg += 1;
        }
        assert!(zlib_decompress(&[cmf, flg]).is_err(), "CM!=8 应拒绝");
    }

    #[test]
    fn reserved_block_type_rejected() {
        let mut s = vec![0x78, 0x9C];
        s.push(0b0000_0111); // BFINAL=1, BTYPE=11
        assert!(zlib_decompress(&s).is_err(), "保留块类型应拒绝");
    }

    #[test]
    fn stored_len_nlen_mismatch_rejected() {
        let mut s = vec![0x78, 0x9C];
        s.push(0x01);
        s.extend_from_slice(&4u16.to_le_bytes());
        s.extend_from_slice(&9999u16.to_le_bytes());
        s.extend_from_slice(&[1, 2, 3, 4]);
        assert!(zlib_decompress(&s).is_err(), "LEN/NLEN 不互补应拒绝");
    }

    #[test]
    fn good_fcheck_with_bad_body_is_rejected() {
        let mut s = vec![0x78, 0x9C];
        s.extend_from_slice(&[0xFF, 0xFF, 0xFF]);
        assert!(zlib_decompress(&s).is_err(), "好头坏体应拒绝");
    }

    #[test]
    fn verified_entrypoint_detects_adler_corruption() {
        let mut z = zlib_stored(b"payload");
        let n = z.len();
        z[n - 1] ^= 0xff;
        // 不校验的入口照旧接受（保持迁移前行为）。
        assert_eq!(zlib_decompress(&z).expect("不校校验和"), b"payload");
        // 校验入口必须抓到。
        assert!(zlib_decompress_verified(&z).is_err(), "校验入口须抓到 Adler-32 不符");
    }

    #[test]
    fn truly_compressed_stream_decodes() {
        // **迁移回归守卫**：真实压缩过的流必须能解（旧 `prism-io` 侧的
        // 截断判据会把这类流全部误杀）。
        let payload: Vec<u8> = vec![b'A'; 259];
        // fixed 块：字面 'A' + 长度码 285（258）+ 距离 0 + 结束码 ≈ 4 字节。
        let deflate: Vec<u8> = {
            let mut cur: u32 = 0b011; // BFINAL=1, BTYPE=01
            let mut nbit: u32 = 3;
            let mut out: Vec<u8> = Vec::new();
            fn put(code: u32, n: u32, cur: &mut u32, nbit: &mut u32, out: &mut Vec<u8>) {
                for i in (0..n).rev() {
                    if (code >> i) & 1 == 1 {
                        *cur |= 1 << *nbit;
                    }
                    *nbit += 1;
                    if *nbit == 8 {
                        out.push(*cur as u8);
                        *cur = 0;
                        *nbit = 0;
                    }
                }
            }
            put(0x30 + 65, 8, &mut cur, &mut nbit, &mut out);
            put(0xc0 + 5, 8, &mut cur, &mut nbit, &mut out);
            put(0, 5, &mut cur, &mut nbit, &mut out);
            put(0, 7, &mut cur, &mut nbit, &mut out);
            if nbit > 0 {
                out.push(cur as u8);
            }
            out
        };
        let mut z = vec![0x78, 0x01];
        z.extend_from_slice(&deflate);
        z.extend_from_slice(&adler32(&payload).to_be_bytes());
        assert!(z.len() < 2 + payload.len(), "该流必须是压缩过的");
        assert_eq!(zlib_decompress(&z).expect("压缩流必须能解"), payload);
    }
}
