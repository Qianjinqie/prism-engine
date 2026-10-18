//! DEFLATE / zlib 解压门面 —— 为 `prism-io` 的图像解码路径提供统一入口。
//!
//! # 实现归属（为什么这里是门面而非第二份实现）
//!
//! 手写 DEFLATE inflate 的**唯一实现**位于 L0 叶子 crate [`prism_bytes`]
//! （`prism_bytes::inflate` / `prism_bytes::zlib`）。`prism-io` 与
//! `prism-media` 同处 L4，彼此不能互相依赖，但都可以向下依赖 `prism-bytes`；
//! 因此本模块只做两件事：
//!
//! 1. **re-export** 底层原语（[`inflate_raw`]、[`adler32`]、[`parse_header`]…）；
//! 2. 把 `BytesError` 的三段式信息**压平成 `String`**（`prism-io` 的 PNG/EXR
//!    解码 API 统一以 `Result<_, String>` 报错，不引入跨 crate 错误类型耦合）。
//!
//! 镜像关系：`crates/prism-media/src/inflate.rs` 是同一份实现的历史门面
//! （带 `INF1xxx` 错误码适配），行为口径一致，**无第二份 inflate 代码**。

pub use prism_bytes::inflate::inflate_raw_prefix;
pub use prism_bytes::zlib::{ZlibHeader, parse_header};

use prism_bytes::BytesError;

/// 把字节层三段式错误压平为单行说明（保留「在哪 / 什么事 / 怎么改」三要素）。
fn flat(e: BytesError) -> String {
    format!("{} 失败：{}（建议：{}）", e.location, e.problem, e.fix)
}

/// 原始 DEFLATE 流解压（RFC 1951，无 zlib 头尾）。
///
/// # 参数
/// * `data` — 从 `BFINAL/BTYPE` 位开始的裸 deflate 字节流。
///
/// # 返回
/// 解压后的字节；格式非法或数据截断时返回带位置的错误说明。
pub fn inflate_raw(data: &[u8]) -> Result<Vec<u8>, String> {
    prism_bytes::inflate::inflate_raw(data).map_err(flat)
}

/// Adler-32 校验和（模 65521），zlib 流尾部校验用。
pub use prism_bytes::zlib::adler32;

/// zlib 流解压（RFC 1950：2 字节头 + DEFLATE，**不要求** Adler-32 尾存在）。
///
/// 宽容口径与 `prism-media::inflate::zlib_decompress` 逐字节等价：仓库内多处
/// 测试构造器造的是不带尾的 zlib 流，故默认不校验尾部。
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    prism_bytes::zlib::decompress_lenient(data).map_err(flat)
}

/// zlib 流解压 + **强制**校验尾部 Adler-32（PNG 的 IDAT 应走这条）。
pub fn zlib_decompress_verified(data: &[u8]) -> Result<Vec<u8>, String> {
    prism_bytes::zlib::decompress_verified(data).map_err(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小的 zlib 存储型（BTYPE=00）流：`78 01 | 01 len nlen payload | adler`。
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let mut out: Vec<u8> = vec![0x78, 0x01];
        out.push(0x01); // BFINAL=1, BTYPE=00
        let len: u16 = payload.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&adler32(payload).to_be_bytes());
        out
    }

    #[test]
    fn zlib_stored_roundtrip() {
        let payload: Vec<u8> = b"prism-io inflate facade".to_vec();
        let stream: Vec<u8> = zlib_stored(&payload);
        assert_eq!(zlib_decompress(&stream).unwrap(), payload);
        assert_eq!(zlib_decompress_verified(&stream).unwrap(), payload);
    }

    #[test]
    fn zlib_bad_header_reports_error() {
        // CM != 8（非 deflate）：必须如实报错，不得静默返回空数据。
        let err: String = zlib_decompress(&[0x77, 0x01, 0x00]).unwrap_err();
        assert!(err.contains("失败"), "错误文案应含「失败」，实际：{}", err);
    }

    #[test]
    fn inflate_truncated_reports_error() {
        let err: String = inflate_raw(&[0x01, 0x05, 0x00, 0xFA]).unwrap_err();
        assert!(err.contains("失败"), "截断流应如实报错，实际：{}", err);
    }
}
