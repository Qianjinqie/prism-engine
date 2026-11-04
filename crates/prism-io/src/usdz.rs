//! V2.5-B2：USDZ（zip 包）解析（零第三方依赖）。
//!
//! USDZ = 一个 zip 包，首条目是根层（`.usdc` 或 `.usda`，规范为 **stored /
//! method 0** 且 64 B 对齐）。本模块手写 zip 目录解析（EOCD → 中央目录 →
//! 本地头 → 数据），**不引 `zip` crate**；deflate 条目复用仓库已有的
//! [`prism_bytes::inflate_raw`]（全引擎唯一的 DEFLATE 解码真源），
//! 不从零造 inflate。
//!
//! 根层的 `.usdc` 交给 [`crate::usdc::parse_crate`]（真二进制 Crate 解析），
//! 根层的 `.usda` 交给 [`crate::usd::parse_str`]；解包后的根层字节与裸
//! `.usdc` 逐位相同（判据 V2）。

use crate::usd::{ParsedUsda, UsdError};
use crate::usdc;

const EOCD: &[u8; 4] = b"PK\x05\x06";
const CENTRAL: &[u8; 4] = b"PK\x01\x02";
const LOCAL: &[u8; 4] = b"PK\x03\x04";

fn err_zip(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1016",
        "usdz::zip",
        problem,
        "确认是合法 USDZ/zip：EOCD 签名存在、中央目录偏移/条目数自洽、条目压缩方式只取 stored(0) 或 deflate(8)",
    )
}

fn err_root(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1017",
        "usdz::root_layer",
        problem,
        "USDZ 首条目必须是根层 `.usdc` / `.usda` / `.usd`；用 usdzip 重新打包",
    )
}

fn u16le(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*d.get(o)?, *d.get(o + 1)?]))
}

fn u32le(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*d.get(o)?, *d.get(o + 1)?, *d.get(o + 2)?, *d.get(o + 3)?]))
}

#[derive(Debug, Clone)]
struct Entry {
    name: String,
    method: u16,
    comp_size: u64,
    uncomp_size: u64,
    local_offset: u64,
}

/// 解出 USDZ 包内全部条目名与根层字节（诊断 / V2 逐位对拍用）。
pub fn extract_root(bytes: &[u8]) -> Result<(String, Vec<u8>, Vec<String>), UsdError> {
    let entries: Vec<Entry> = central_directory(bytes)?;
    let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    // USDZ 规范：首条目即根层；兼容时也接受第一个 `.usdc/.usda/.usd` 条目。
    let root: &Entry = entries
        .iter()
        .find(|e| is_layer(&e.name))
        .ok_or_else(|| err_root(format!("包内没有根层（条目：{names:?}）")))?;
    let payload: Vec<u8> = extract(bytes, root)?;
    Ok((root.name.clone(), payload, names))
}

/// 解析 USDZ（zip）字节 → 与 `.usda` / `.usdc` 同构的 [`ParsedUsda`]。
pub fn parse_usdz(bytes: &[u8]) -> Result<ParsedUsda, UsdError> {
    let (name, payload, _) = extract_root(bytes)?;
    if name.ends_with(".usdc") || name.ends_with(".usd") {
        usdc::parse_crate(&payload)
    } else {
        let text: &str = std::str::from_utf8(&payload).map_err(|e| {
            err_zip(format!("根层 `{name}` 不是合法 UTF-8：{e}"))
        })?;
        crate::usd::parse_str(text)
    }
}

fn is_layer(name: &str) -> bool {
    let lower: String = name.to_ascii_lowercase();
    lower.ends_with(".usdc") || lower.ends_with(".usda") || lower.ends_with(".usd")
}

fn central_directory(bytes: &[u8]) -> Result<Vec<Entry>, UsdError> {
    // EOCD 在文件尾（最多 65535 B 注释 + 22 B 头）。
    let scan_start: usize = bytes.len().saturating_sub(65_557);
    let mut eocd: Option<usize> = None;
    for i in (scan_start..bytes.len().saturating_sub(3)).rev() {
        if bytes.get(i..i + 4) == Some(EOCD.as_slice()) {
            eocd = Some(i);
            break;
        }
    }
    let e: usize = eocd.ok_or_else(|| err_zip("找不到 EOCD（不是 zip/USDZ）"))?;
    let total: u16 = u16le(bytes, e + 10).ok_or_else(|| err_zip("EOCD 截断"))?;
    let cd_size: u32 = u32le(bytes, e + 12).ok_or_else(|| err_zip("EOCD 截断"))?;
    let cd_off: u32 = u32le(bytes, e + 16).ok_or_else(|| err_zip("EOCD 截断"))?;
    if total == 0 {
        return Err(err_zip("中央目录条目数为 0"));
    }
    let mut pos: u64 = cd_off as u64;
    let cd_end: u64 = pos + cd_size as u64;
    if cd_end > bytes.len() as u64 {
        return Err(err_zip(format!("中央目录 [{pos}, {cd_end}) 越过文件末尾")));
    }
    let mut out: Vec<Entry> = Vec::with_capacity(total as usize);
    for i in 0..total {
        let o: usize = usize::try_from(pos).map_err(|_| err_zip("中央目录偏移溢出"))?;
        if bytes.get(o..o + 4) != Some(CENTRAL.as_slice()) {
            return Err(err_zip(format!("第 {i} 条中央目录签名不对（偏移 {o}）")));
        }
        let method: u16 = u16le(bytes, o + 10).ok_or_else(|| err_zip("中央目录截断"))?;
        let comp: u32 = u32le(bytes, o + 20).ok_or_else(|| err_zip("中央目录截断"))?;
        let uncomp: u32 = u32le(bytes, o + 24).ok_or_else(|| err_zip("中央目录截断"))?;
        let nlen: u16 = u16le(bytes, o + 28).ok_or_else(|| err_zip("中央目录截断"))?;
        let xlen: u16 = u16le(bytes, o + 30).ok_or_else(|| err_zip("中央目录截断"))?;
        let clen: u16 = u16le(bytes, o + 32).ok_or_else(|| err_zip("中央目录截断"))?;
        let local: u32 = u32le(bytes, o + 42).ok_or_else(|| err_zip("中央目录截断"))?;
        let ns: usize = o + 46;
        let ne: usize = ns.checked_add(nlen as usize).ok_or_else(|| err_zip("文件名长度溢出"))?;
        let name_bytes: &[u8] = bytes.get(ns..ne).ok_or_else(|| err_zip("中央目录文件名截断"))?;
        let name: String = String::from_utf8_lossy(name_bytes).to_string();
        // ZIP64：0xFFFFFFFF 哨兵本版本不认（明确报错，不猜）。
        if comp == u32::MAX || uncomp == u32::MAX || local == u32::MAX {
            return Err(err_zip(format!("条目 `{name}` 使用 ZIP64 哨兵，本版本未实现")));
        }
        out.push(Entry {
            name,
            method,
            comp_size: comp as u64,
            uncomp_size: uncomp as u64,
            local_offset: local as u64,
        });
        pos += 46 + nlen as u64 + xlen as u64 + clen as u64;
    }
    Ok(out)
}

fn extract(bytes: &[u8], e: &Entry) -> Result<Vec<u8>, UsdError> {
    let o: usize = usize::try_from(e.local_offset).map_err(|_| err_zip("本地头偏移溢出"))?;
    if bytes.get(o..o + 4) != Some(LOCAL.as_slice()) {
        return Err(err_zip(format!("条目 `{}` 的本地头签名不对", e.name)));
    }
    let nlen: u16 = u16le(bytes, o + 26).ok_or_else(|| err_zip("本地头截断"))?;
    let xlen: u16 = u16le(bytes, o + 28).ok_or_else(|| err_zip("本地头截断"))?;
    let ds: u64 = e.local_offset + 30 + nlen as u64 + xlen as u64;
    let de: u64 = ds.checked_add(e.comp_size).ok_or_else(|| err_zip("条目长度溢出"))?;
    if de > bytes.len() as u64 {
        return Err(err_zip(format!("条目 `{}` 的数据越过文件末尾", e.name)));
    }
    let payload: &[u8] = &bytes[ds as usize..de as usize];
    let out: Vec<u8> = match e.method {
        0 => payload.to_vec(),
        8 => prism_bytes::inflate_raw(payload).map_err(|err| {
            err_zip(format!("条目 `{}` 的 deflate 解压失败：{err}", e.name))
        })?,
        other => {
            return Err(err_zip(format!(
                "条目 `{}` 的压缩方式 {other} 不支持（只允许 0/8）",
                e.name
            )))
        }
    };
    if out.len() as u64 != e.uncomp_size {
        return Err(err_zip(format!(
            "条目 `{}` 解压后 {} B，目录声明 {} B",
            e.name,
            out.len(),
            e.uncomp_size
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_zip_input_is_structurally_rejected() {
        let e = parse_usdz(b"PK\x03\x04not a zip").unwrap_err();
        assert_eq!(e.code, "PU1016");
        assert!(e.fix.contains("EOCD"));
    }
}
