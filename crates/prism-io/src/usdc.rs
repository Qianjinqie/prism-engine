//! V2.5-B2：USD `.usdc` **Crate 二进制格式**真解析（零第三方依赖）。
//!
//! # 格式权威
//! 布局逐字段对齐 Pixar OpenUSD `pxr/usd/sdf/crateFile.{h,cpp}` /
//! `crateDataTypes.h` / `crateValueInliners.h` / `integerCoding.cpp` /
//! `tf/fastCompression.cpp`：
//!
//! - **Bootstrap（88 B）**：`ident[8]="PXR-USDC"`、`version[8]`（major/minor/patch
//!   + 5 保留）、`int64 tocOffset`、`int64 _reserved[8]`；
//! - **TOC**（在 `tocOffset`）：`uint64 numSections` + 每段
//!   `char name[16] / int64 start / int64 size`（32 B）；
//! - **六段**：`TOKENS` / `STRINGS` / `FIELDS` / `FIELDSETS` / `PATHS` / `SPECS`；
//! - **0.4.0+ 压缩段**：整数段走 `Sdf_IntegerCompression`（差分 + 最常见值 +
//!   2-bit 码 + 变长整数），通用段走 `TfFastCompression`（1 B chunk 数 + 裸 LZ4
//!   block）。本模块自研 LZ4 block 解压与整数解码，**不引 lz4 crate**；
//!   USDZ 的 zip deflate 复用 `prism-bytes::inflate_raw`（见 `usdz.rs`）；
//! - **ValueRep（8 B）**：`[isArray:1][isInlined:1][isCompressed:1][isArrayEdit:1]
//!   [type:8][payload:48]`，payload 内联值或**文件绝对偏移**；
//! - **Spec**：`pathIndex / fieldSetIndex / specType`（`SdfSpecType`：
//!   Attribute=1 / Prim=6 / PseudoRoot=7 / Relationship=8 …）。
//!
//! # 版本支持
//! - `0.4.0 ≤ v ≤ 0.15.0`：压缩结构段（本轮 golden 是真实 USD 写出的 0.8.0）；
//! - `v < 0.4.0`：非压缩结构段（`_PathItemHeader` 树 / 原始 `Field` / `Spec`）。
//!
//! # 错误纪律
//! 截断、错 TOC、错压缩段、错 LZ4 偏移、越界索引、未实现的 value rep →
//! `PU1011`–`PU1015` 三段式结构化报错（在哪 / 什么事 / 怎么改），
//! **不 panic、不静默吞、不半解析出看似合理的场景**。
//! 真正的「magic 都不成 crate」输入由 `usd::parse_bytes` 的预检保持 `PU1001`。

use crate::usd::{
    AttrValue, Attribute, Layer, LayerReport, ParsedUsda, Prim, PrimKind, Relationship, UsdError,
};

/// Bootstrap 结构字节数（`ident 8 + version 8 + tocOffset 8 + reserved 64`）。
pub const BOOTSTRAP_SIZE: usize = 88;
/// USDZ zip 本地文件头签名（由 `usdz.rs` 使用，此处集中定义）。
pub const ZIP_LOCAL_MAGIC: &[u8; 4] = b"PK\x03\x04";

/// Crate 文件级信息（诊断 / golden 对拍用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateInfo {
    /// 版本字符串（`major.minor.patch`）。
    pub version: String,
    /// TOC 段名 / 起始偏移 / 字节数。
    pub sections: Vec<(String, u64, u64)>,
    /// token 表长度。
    pub token_count: usize,
    /// field 数。
    pub field_count: usize,
    /// spec 数。
    pub spec_count: usize,
}

/// 解析 `.usdc` 字节 → 与 `.usda` 同构的 [`ParsedUsda`]。
pub fn parse_crate(bytes: &[u8]) -> Result<ParsedUsda, UsdError> {
    let c: Crate = Crate::read(bytes)?;
    let layer: Layer = c.build_layer()?;
    Ok(ParsedUsda { layer, report: LayerReport::default() })
}

/// 只读结构信息（不改场景，供诊断 / 负控断言）。
pub fn inspect(bytes: &[u8]) -> Result<CrateInfo, UsdError> {
    let c: Crate = Crate::read(bytes)?;
    Ok(CrateInfo {
        version: c.version_string(),
        sections: c.sections_raw(),
        token_count: c.tokens.len(),
        field_count: c.fields.len(),
        spec_count: c.specs.len(),
    })
}

// ===========================================================================
//  字节读取器（全边界检查）
// ===========================================================================

struct Reader<'a> {
    d: &'a [u8],
}

impl<'a> Reader<'a> {
    fn slice(&self, off: u64, len: u64, what: &str) -> Result<&'a [u8], UsdError> {
        let o: usize = usize::try_from(off).map_err(|_| err_truncated(what, off, len))?;
        let l: usize = usize::try_from(len).map_err(|_| err_truncated(what, off, len))?;
        let end: usize = o.checked_add(l).ok_or_else(|| err_truncated(what, off, len))?;
        self.d.get(o..end).ok_or_else(|| err_truncated(what, off, len))
    }

    fn u8(&self, off: u64, what: &str) -> Result<u8, UsdError> {
        Ok(self.slice(off, 1, what)?[0])
    }
    fn u16(&self, off: u64, what: &str) -> Result<u16, UsdError> {
        let s = self.slice(off, 2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&self, off: u64, what: &str) -> Result<u32, UsdError> {
        let s = self.slice(off, 4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn i32(&self, off: u64, what: &str) -> Result<i32, UsdError> {
        Ok(self.u32(off, what)? as i32)
    }
    fn u64(&self, off: u64, what: &str) -> Result<u64, UsdError> {
        let s = self.slice(off, 8, what)?;
        Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    fn i64(&self, off: u64, what: &str) -> Result<i64, UsdError> {
        Ok(self.u64(off, what)? as i64)
    }
    fn f32(&self, off: u64, what: &str) -> Result<f64, UsdError> {
        Ok(f32::from_bits(self.u32(off, what)?) as f64)
    }
    fn f64(&self, off: u64, what: &str) -> Result<f64, UsdError> {
        Ok(f64::from_bits(self.u64(off, what)?))
    }
}

fn err_truncated(what: &str, off: u64, len: u64) -> UsdError {
    UsdError::new(
        "PU1011",
        "usdc::Reader",
        format!("{what} 在偏移 {off} 处需要 {len} 字节但文件已截断/越界"),
        "确认 .usdc 文件完整（未被截断）；USDZ 请先解包出根层；损坏文件不做半解析",
    )
}

fn err_toc(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1012",
        "usdc::TOC",
        problem,
        "检查 bootstrap 的 tocOffset 是否指向文件内、段 start/size 是否越界、段名是否为六个已知段之一",
    )
}

fn err_compressed(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1013",
        "usdc::compressed_section",
        problem,
        "确认压缩段未被破坏：LZ4 block 的 match offset 必须非零且不越过已产出字节，整数段的 2-bit 码与变长段须自洽",
    )
}

fn err_value(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1014",
        "usdc::value_rep",
        problem,
        "该 value rep 的类型/编码本版本未实现；用 usdcat 导出为 .usda，或补上对应类型的解码分支",
    )
}

fn err_version(problem: impl Into<String>) -> UsdError {
    UsdError::new(
        "PU1015",
        "usdc::bootstrap",
        problem,
        "本解析器支持 crate 版本 0.0.1–0.15.0；用 usdcat --usda 导出文本层",
    )
}

// ===========================================================================
//  LZ4 block + TfFastCompression + Sdf_IntegerCompression
// ===========================================================================

/// 裸 LZ4 block 解压（token / 扩展字面长 / match offset / 扩展匹配长）。
/// 全边界检查，非法 offset 立即结构化报错；`cap` 是输出上限（防畸形输入撑爆内存），
/// **不**要求产出恰好 cap 字节——整数段的变长部分可以比编码缓冲上限短。
pub fn lz4_block(src: &[u8], cap: usize) -> Result<Vec<u8>, UsdError> {
    let mut out: Vec<u8> = Vec::with_capacity(cap.min(1 << 16));
    let mut i: usize = 0;
    while i < src.len() {
        let token: u8 = src[i];
        i += 1;
        let mut lit: usize = (token >> 4) as usize;
        if lit == 15 {
            loop {
                let b: u8 = *src.get(i).ok_or_else(|| err_compressed("LZ4 扩展字面长截断"))?;
                i += 1;
                lit = lit.checked_add(b as usize).ok_or_else(|| err_compressed("LZ4 字面长溢出"))?;
                if b != 255 {
                    break;
                }
            }
        }
        let end: usize = i.checked_add(lit).ok_or_else(|| err_compressed("LZ4 字面长溢出"))?;
        if end > src.len() {
            return Err(err_compressed("LZ4 字面段越过压缩块末尾"));
        }
        if out.len() + lit > cap {
            return Err(err_compressed(format!("LZ4 产出超过上限 {cap} B")));
        }
        out.extend_from_slice(&src[i..end]);
        i = end;
        if i >= src.len() {
            break; // 末块只有字面段
        }
        if i + 2 > src.len() {
            return Err(err_compressed("LZ4 match offset 截断"));
        }
        let off: usize = (src[i] as usize) | ((src[i + 1] as usize) << 8);
        i += 2;
        if off == 0 || off > out.len() {
            return Err(err_compressed(format!("LZ4 match offset={off} 非法（已产出 {} B）", out.len())));
        }
        let mut ml: usize = (token & 15) as usize + 4;
        if token & 15 == 15 {
            loop {
                let b: u8 = *src.get(i).ok_or_else(|| err_compressed("LZ4 扩展匹配长截断"))?;
                i += 1;
                ml = ml.checked_add(b as usize).ok_or_else(|| err_compressed("LZ4 匹配长溢出"))?;
                if b != 255 {
                    break;
                }
            }
        }
        if out.len() + ml > cap {
            return Err(err_compressed(format!("LZ4 产出超过上限 {cap} B")));
        }
        for _ in 0..ml {
            let b: u8 = out[out.len() - off];
            out.push(b);
        }
    }
    Ok(out)
}

/// `TfFastCompression::DecompressFromBuffer`：首字节 chunk 数（0 = 单块），
/// 多块时每块前置 `int32` 长度；每块是裸 LZ4 block。返回**完整**解压结果。
fn tf_decompress(src: &[u8], cap: usize) -> Result<Vec<u8>, UsdError> {
    if src.is_empty() {
        return Err(err_compressed("TfFastCompression 段为空（缺 chunk 数字节）"));
    }
    let n: u8 = src[0];
    let mut pos: usize = 1;
    let mut out: Vec<u8> = Vec::new();
    if n == 0 {
        out = lz4_block(&src[1..], cap)?;
    } else {
        for k in 0..n {
            if pos + 4 > src.len() {
                return Err(err_compressed(format!("第 {k} 个 LZ4 chunk 的长度头截断")));
            }
            let sz: u32 = u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]]);
            pos += 4;
            let end: usize = pos.checked_add(sz as usize).ok_or_else(|| err_compressed("chunk 长度溢出"))?;
            if end > src.len() {
                return Err(err_compressed(format!("第 {k} 个 LZ4 chunk 越过段末尾")));
            }
            let room: usize = cap.saturating_sub(out.len());
            let part: Vec<u8> = lz4_block(&src[pos..end], room)?;
            out.extend_from_slice(&part);
            pos = end;
        }
    }
    if out.len() > cap {
        return Err(err_compressed(format!("解压产出 {} B 超过上限 {cap} B", out.len())));
    }
    Ok(out)
}

/// `Sdf_IntegerCompression::DecompressFromBuffer`：TfFastCompression →
/// `[commonValue:i32][2-bit codes][varints]`，还原差分序列的前缀和。
/// 变长段可短于 `4 + codes + 4*count` 的编码上限，故只按实际可读长度解码，
/// 尾部剩余字节忽略（USD 原实现同样按 2-bit 码读取，不要求填满缓冲）。
fn decode_ints(compressed: &[u8], count: usize) -> Result<Vec<i64>, UsdError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let ncb: usize = (count * 2 + 7) / 8;
    let cap: usize = 4 + ncb + count * 4 + 8;
    let raw: Vec<u8> = tf_decompress(compressed, cap)?;
    if raw.len() < 4 + ncb {
        return Err(err_compressed(format!(
            "整数段解码缓冲只有 {} B，至少需要 {} B（{} 个整数）",
            raw.len(),
            4 + ncb,
            count
        )));
    }
    let common: i64 = i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64;
    let mut vpos: usize = 4 + ncb;
    let mut prev: i64 = 0;
    let mut out: Vec<i64> = Vec::with_capacity(count);
    for i in 0..count {
        let code: u8 = (raw[4 + i / 4] >> (2 * (i % 4))) & 3;
        let delta: i64 = match code {
            0 => common,
            1 => {
                let b: u8 = *raw.get(vpos).ok_or_else(|| err_compressed("整数段 int8 变长项截断"))?;
                vpos += 1;
                (b as i8) as i64
            }
            2 => {
                let b0: u8 = *raw.get(vpos).ok_or_else(|| err_compressed("整数段 int16 变长项截断"))?;
                let b1: u8 = *raw.get(vpos + 1).ok_or_else(|| err_compressed("整数段 int16 变长项截断"))?;
                vpos += 2;
                i16::from_le_bytes([b0, b1]) as i64
            }
            _ => {
                let b0: u8 = *raw.get(vpos).ok_or_else(|| err_compressed("整数段 int32 变长项截断"))?;
                let b1: u8 = *raw.get(vpos + 1).ok_or_else(|| err_compressed("整数段 int32 变长项截断"))?;
                let b2: u8 = *raw.get(vpos + 2).ok_or_else(|| err_compressed("整数段 int32 变长项截断"))?;
                let b3: u8 = *raw.get(vpos + 3).ok_or_else(|| err_compressed("整数段 int32 变长项截断"))?;
                vpos += 4;
                i32::from_le_bytes([b0, b1, b2, b3]) as i64
            }
        };
        prev = prev.wrapping_add(delta);
        out.push(prev);
    }
    Ok(out)
}

/// 读一个 `[u64 compressedSize][bytes]` 的整数数组，返回 `(值, 新偏移)`。
fn read_int_array(r: &Reader<'_>, pos: u64, count: usize, what: &str) -> Result<(Vec<i64>, u64), UsdError> {
    let size: u64 = r.u64(pos, &format!("{what} 长度"))?;
    let payload: u64 = pos + 8;
    let src: &[u8] = r.slice(payload, size, &format!("{what} 压缩数据"))?;
    let vals: Vec<i64> = decode_ints(src, count)?;
    Ok((vals, payload + size))
}

// ===========================================================================
//  Crate 结构
// ===========================================================================

#[derive(Debug, Clone)]
struct Section {
    name: String,
    start: u64,
    size: u64,
}

#[derive(Debug, Clone)]
struct Field {
    name: String,
    rep: u64,
}

#[derive(Debug, Clone)]
struct Spec {
    path: usize,
    fieldset: usize,
    kind: u32,
}

#[derive(Debug, Clone)]
enum CrateValue {
    None,
    Bool(bool),
    Number(f64),
    Str(String),
    Numbers(Vec<f64>),
    Rows(Vec<Vec<f64>>),
    Matrix(Vec<Vec<f64>>),
    Strings(Vec<String>),
    Paths(Vec<String>),
    Specifier(u32),
    Variability(u32),
    TimeSamples,
}

struct Crate<'a> {
    r: Reader<'a>,
    version: [u8; 3],
    sections: Vec<Section>,
    tokens: Vec<String>,
    strings: Vec<String>,
    fields: Vec<Field>,
    fieldsets: Vec<i64>,
    specs: Vec<Spec>,
    paths: Vec<String>,
    /// 路径是否为 prim property path（`.attr` 形态）。
    property: Vec<bool>,
    /// 结构解码诊断（path/token 表装配用；不参与场景语义）。
    diag: String,
}

/// 结构解码诊断（V2 判据调试 / 负控定位用）。
pub fn crate_diag(bytes: &[u8]) -> Result<String, UsdError> {
    let c: Crate = Crate::read(bytes)?;
    Ok(c.diag)
}

/// 重建出的 SdfPath 表（诊断 / 逐字段对拍用）。
pub fn path_table(bytes: &[u8]) -> Result<Vec<String>, UsdError> {
    let c: Crate = Crate::read(bytes)?;
    Ok(c.paths)
}

impl<'a> Crate<'a> {
    fn version_string(&self) -> String {
        format!("{}.{}.{}", self.version[0], self.version[1], self.version[2])
    }

    fn sections_raw(&self) -> Vec<(String, u64, u64)> {
        self.sections.iter().map(|s| (s.name.clone(), s.start, s.size)).collect()
    }

    fn section(&self, name: &str) -> Result<&Section, UsdError> {
        self.sections.iter().find(|s| s.name == name).ok_or_else(|| {
            err_toc(format!("缺少必需段 `{name}`（TOC 里有：{:?}）", self.sections.iter().map(|s| &s.name).collect::<Vec<_>>()))
        })
    }

    fn compressed(&self) -> bool {
        // 0.4.0 起结构段改为压缩；0.0.1–0.3.0 为原始布局。
        (self.version[0], self.version[1]) >= (0, 4)
    }

    fn array_count_u64(&self) -> bool {
        // 0.7.0 起数组长度写成 u64，之前为 u32。
        (self.version[0], self.version[1]) >= (0, 7)
    }

    fn read(data: &'a [u8]) -> Result<Self, UsdError> {
        let r = Reader { d: data };
        if data.len() < BOOTSTRAP_SIZE {
            return Err(err_truncated("bootstrap", 0, BOOTSTRAP_SIZE as u64));
        }
        if &data[..8] != b"PXR-USDC" {
            return Err(err_version("magic 不是 `PXR-USDC`"));
        }
        let version: [u8; 3] = [data[8], data[9], data[10]];
        if version[0] != 0 || version[1] > 15 {
            return Err(err_version(format!("不支持的 crate 版本 {}", fmt_ver(version))));
        }
        let toc_offset: u64 = r.u64(16, "bootstrap.tocOffset")?;
        if toc_offset >= data.len() as u64 {
            return Err(err_toc(format!(
                "tocOffset={toc_offset} 越界（文件 {} B）",
                data.len()
            )));
        }
        let n: u64 = r.u64(toc_offset, "TOC.numSections")?;
        if n == 0 || n > 64 {
            return Err(err_toc(format!("TOC 段数 {n} 不合法（期望 1..=64）")));
        }
        let mut sections: Vec<Section> = Vec::with_capacity(n as usize);
        for i in 0..n {
            let o: u64 = toc_offset + 8 + i * 32;
            let raw: &[u8] = r.slice(o, 32, &format!("TOC 第 {i} 段"))?;
            let end: usize = raw[..16].iter().position(|&b| b == 0).unwrap_or(16);
            let name: String = String::from_utf8_lossy(&raw[..end]).to_string();
            let start: u64 = u64::from_le_bytes(raw[16..24].try_into().unwrap_or([0; 8]));
            let size: u64 = u64::from_le_bytes(raw[24..32].try_into().unwrap_or([0; 8]));
            if name.is_empty() || !name.is_ascii() {
                return Err(err_toc(format!("TOC 第 {i} 段的段名非法：{name:?}")));
            }
            if start > data.len() as u64 || size > data.len() as u64 - start {
                return Err(err_toc(format!(
                    "段 `{name}` 的 start={start} size={size} 越过文件末尾 {}",
                    data.len()
                )));
            }
            sections.push(Section { name, start, size });
        }
        let mut c = Crate {
            r,
            version,
            sections,
            tokens: Vec::new(),
            strings: Vec::new(),
            fields: Vec::new(),
            fieldsets: Vec::new(),
            specs: Vec::new(),
            paths: Vec::new(),
            property: Vec::new(),
            diag: String::new(),
        };
        c.read_tokens()?;
        c.read_strings()?;
        c.read_fields()?;
        c.read_fieldsets()?;
        c.read_paths()?;
        c.read_specs()?;
        Ok(c)
    }

    fn token(&self, idx: i64) -> Result<String, UsdError> {
        if idx < 0 || idx as usize >= self.tokens.len() {
            return Err(err_value(format!(
                "token index {idx} 越界（表长 {}）",
                self.tokens.len()
            )));
        }
        Ok(self.tokens[idx as usize].clone())
    }

    fn string(&self, idx: i64) -> Result<String, UsdError> {
        if idx < 0 || idx as usize >= self.strings.len() {
            return Err(err_value(format!(
                "string index {idx} 越界（表长 {}）",
                self.strings.len()
            )));
        }
        Ok(self.strings[idx as usize].clone())
    }

    fn path(&self, idx: i64) -> Result<String, UsdError> {
        if idx < 0 || idx as usize >= self.paths.len() {
            return Err(err_value(format!(
                "path index {idx} 越界（表长 {}）",
                self.paths.len()
            )));
        }
        Ok(self.paths[idx as usize].clone())
    }

    // ---- TOKENS ----------------------------------------------------------
    fn read_tokens(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("TOKENS")?.clone();
        let mut pos: u64 = s.start;
        let count: u64 = self.r.u64(pos, "TOKENS.count")?;
        pos += 8;
        let data: Vec<u8>;
        if self.compressed() {
            let unc: u64 = self.r.u64(pos, "TOKENS.uncompressedSize")?;
            pos += 8;
            let comp: u64 = self.r.u64(pos, "TOKENS.compressedSize")?;
            pos += 8;
            let src: &[u8] = self.r.slice(pos, comp, "TOKENS.compressed")?;
            data = tf_decompress(src, unc as usize)?;
            if data.len() != unc as usize {
                return Err(err_compressed(format!(
                    "TOKENS 解压产出 {} B，声明 {unc} B",
                    data.len()
                )));
            }
        } else {
            let total: u64 = self.r.u64(pos, "TOKENS.totalBytes")?;
            pos += 8;
            data = self.r.slice(pos, total, "TOKENS.data")?.to_vec();
        }
        let mut out: Vec<String> = Vec::new();
        let mut cur: Vec<u8> = Vec::new();
        for b in data {
            if b == 0 {
                out.push(String::from_utf8_lossy(&cur).to_string());
                cur.clear();
            } else {
                cur.push(b);
            }
        }
        if out.len() != count as usize {
            // 末尾无 `\0` 时仍收最后一个（容错但不静默：数量不符即报错）。
            if out.len() + 1 == count as usize && !cur.is_empty() {
                out.push(String::from_utf8_lossy(&cur).to_string());
            } else {
                return Err(err_compressed(format!(
                    "TOKENS 声明 {count} 个，实际解出 {} 个（cur {} B）",
                    out.len(),
                    cur.len()
                )));
            }
        }
        self.tokens = out;
        Ok(())
    }

    // ---- STRINGS ---------------------------------------------------------
    fn read_strings(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("STRINGS")?.clone();
        let count: u64 = self.r.u64(s.start, "STRINGS.count")?;
        let mut out: Vec<String> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let idx: u32 = self.r.u32(s.start + 8 + i * 4, "STRINGS.tokenIndex")?;
            out.push(self.token(idx as i64)?);
        }
        self.strings = out;
        Ok(())
    }

    // ---- FIELDS ----------------------------------------------------------
    fn read_fields(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("FIELDS")?.clone();
        let mut pos: u64 = s.start;
        let count: u64 = self.r.u64(pos, "FIELDS.count")?;
        pos += 8;
        let (names, reps): (Vec<i64>, Vec<u64>) = if self.compressed() {
            let (idx, next): (Vec<i64>, u64) =
                read_int_array(&self.r, pos, count as usize, "FIELDS.tokenIndexes")?;
            pos = next;
            let rsz: u64 = self.r.u64(pos, "FIELDS.repsSize")?;
            pos += 8;
            let raw: Vec<u8> = tf_decompress(
                self.r.slice(pos, rsz, "FIELDS.reps")?,
                count as usize * 8,
            )?;
            if raw.len() != count as usize * 8 {
                return Err(err_compressed(format!(
                    "FIELDS.reps 解压产出 {} B，期望 {} B",
                    raw.len(),
                    count as usize * 8
                )));
            }
            let mut reps: Vec<u64> = Vec::with_capacity(count as usize);
            for i in 0..count as usize {
                reps.push(u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8])));
            }
            (idx, reps)
        } else {
            let mut idx: Vec<i64> = Vec::with_capacity(count as usize);
            let mut reps: Vec<u64> = Vec::with_capacity(count as usize);
            for i in 0..count {
                let o: u64 = pos + i as u64 * 16;
                idx.push(self.r.u32(o + 4, "Field.tokenIndex")? as i64);
                reps.push(self.r.u64(o + 8, "Field.valueRep")?);
            }
            (idx, reps)
        };
        let mut out: Vec<Field> = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            out.push(Field { name: self.token(names[i])?, rep: reps[i] });
        }
        self.fields = out;
        Ok(())
    }

    // ---- FIELDSETS -------------------------------------------------------
    fn read_fieldsets(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("FIELDSETS")?.clone();
        let mut pos: u64 = s.start;
        if self.compressed() {
            let count: u64 = self.r.u64(pos, "FIELDSETS.count")?;
            pos += 8;
            let (vals, _): (Vec<i64>, u64) =
                read_int_array(&self.r, pos, count as usize, "FIELDSETS")?;
            self.fieldsets = vals;
        } else {
            let count: u64 = self.r.u64(pos, "FIELDSETS.count")?;
            pos += 8;
            let mut vals: Vec<i64> = Vec::with_capacity(count as usize);
            for i in 0..count {
                vals.push(self.r.u32(pos + i as u64 * 4, "FieldIndex")? as i64);
            }
            self.fieldsets = vals;
        }
        Ok(())
    }

    // ---- PATHS -----------------------------------------------------------
    fn read_paths(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("PATHS")?.clone();
        let total: u64 = self.r.u64(s.start, "PATHS.count")?;
        let mut pos: u64 = s.start + 8;
        self.paths = vec![String::new(); total as usize];
        self.property = vec![false; total as usize];
        if self.compressed() {
            let n: u64 = self.r.u64(pos, "PATHS.numEncoded")?;
            pos += 8;
            let (pidx, next): (Vec<i64>, u64) =
                read_int_array(&self.r, pos, n as usize, "PATHS.pathIndexes")?;
            let (etok, next): (Vec<i64>, u64) =
                read_int_array(&self.r, next, n as usize, "PATHS.elementTokenIndexes")?;
            let (jumps, _): (Vec<i64>, u64) =
                read_int_array(&self.r, next, n as usize, "PATHS.jumps")?;
            self.diag.push_str(&format!(
                "pidx={pidx:?} etok={etok:?} jumps={jumps:?}\n"
            ));
            self.build_paths_compressed(&pidx, &etok, &jumps)?;
            self.diag.push_str(&format!("paths={:?}\n", self.paths));
        } else {
            self.build_paths_tree(pos, String::new())?;
        }
        Ok(())
    }

    /// 0.4.0+ 压缩路径树：`jumps` 语义 `0=仅兄弟`、`-1=仅子`、`-2=叶`、
    /// 正数=既有子也有兄弟且为兄弟子树起始下标偏移。
    fn build_paths_compressed(
        &mut self,
        pidx: &[i64],
        etok: &[i64],
        jumps: &[i64],
    ) -> Result<(), UsdError> {
        if !(pidx.len() == etok.len() && etok.len() == jumps.len()) {
            return Err(err_compressed("PATHS 三个数组长度不一致"));
        }
        fn walk(
            c: &mut Crate<'_>,
            pidx: &[i64],
            etok: &[i64],
            jumps: &[i64],
            start: usize,
            parent: String,
        ) -> Result<(), UsdError> {
            let mut i: usize = start;
            let mut parent_path: String = parent;
            loop {
                if i >= pidx.len() {
                    return Err(err_compressed("PATHS jumps 越过数组末尾"));
                }
                let idx: usize = pidx[i] as usize;
                if idx >= c.paths.len() {
                    return Err(err_value(format!("pathIndex {} 越界", pidx[i])));
                }
                let path: String = if parent_path.is_empty() {
                    "/".to_string()
                } else {
                    let e: i64 = etok[i];
                    let tok: String = c.token(e.abs())?;
                    if e < 0 {
                        // prim property path：`<parent>.<name>`
                        c.property[idx] = true;
                        format!("{parent_path}.{tok}")
                    } else if parent_path == "/" {
                        format!("/{tok}")
                    } else {
                        format!("{parent_path}/{tok}")
                    }
                };
                c.paths[idx] = path.clone();
                let j: i64 = jumps[i];
                let has_child: bool = j > 0 || j == -1;
                let has_sibling: bool = j >= 0;
                if has_child && has_sibling {
                    let sibling: usize = i.checked_add(j as usize).ok_or_else(|| {
                        err_compressed("PATHS sibling 下标溢出")
                    })?;
                    walk(c, pidx, etok, jumps, sibling, parent_path.clone())?;
                }
                if has_child {
                    parent_path = path;
                }
                // C++ 口径：`do { ... } while (hasChild || hasSibling)`。
                // 只有 `-2`（叶）才停；`-1`（只有子）继续沿数组下一项深入子树。
                if !has_child && !has_sibling {
                    return Ok(());
                }
                i += 1;
            }
        }
        walk(self, pidx, etok, jumps, 0, String::new())
    }

    /// v<0.4.0 的 `_PathItemHeader` 树（12 B：u32 pathIndex / u32 tokenIndex /
    /// u8 bits / 3 B 保留；既有子又有兄弟时紧跟一个 i64 绝对偏移）。
    fn build_paths_tree(&mut self, pos: u64, parent: String) -> Result<u64, UsdError> {
        let mut cur: u64 = pos;
        let parent_path: String = parent;
        loop {
            let idx: u32 = self.r.u32(cur, "PathItemHeader.index")?;
            let tok: u32 = self.r.u32(cur + 4, "PathItemHeader.elementToken")?;
            let bits: u8 = self.r.u8(cur + 8, "PathItemHeader.bits")?;
            let has_child: bool = bits & 1 != 0;
            let has_sibling: bool = bits & 2 != 0;
            let is_prop: bool = bits & 4 != 0;
            if idx as usize >= self.paths.len() {
                return Err(err_value(format!("pathIndex {idx} 越界")));
            }
            let path: String = if parent_path.is_empty() {
                "/".to_string()
            } else {
                let e: String = self.token(tok as i64)?;
                if is_prop {
                    self.property[idx as usize] = true;
                    format!("{parent_path}.{e}")
                } else if parent_path == "/" {
                    format!("/{e}")
                } else {
                    format!("{parent_path}/{e}")
                }
            };
            self.paths[idx as usize] = path.clone();
            let mut next: u64 = cur + 12;
            let sibling_abs: Option<u64> = if has_child && has_sibling {
                let off: u64 = self.r.u64(next, "PathItemHeader.siblingOffset")?;
                next += 8;
                Some(off)
            } else {
                None
            };
            if has_child {
                next = self.build_paths_tree(next, path.clone())?;
            }
            if let Some(off) = sibling_abs {
                cur = off;
            } else {
                cur = next;
            }
            if !has_sibling {
                return Ok(cur);
            }
        }
    }

    // ---- SPECS -----------------------------------------------------------
    fn read_specs(&mut self) -> Result<(), UsdError> {
        let s: Section = self.section("SPECS")?.clone();
        let mut pos: u64 = s.start;
        let count: u64 = self.r.u64(pos, "SPECS.count")?;
        pos += 8;
        let (p, f, k): (Vec<i64>, Vec<i64>, Vec<i64>) = if self.compressed() {
            let (p, next): (Vec<i64>, u64) =
                read_int_array(&self.r, pos, count as usize, "SPECS.pathIndexes")?;
            let (f, next): (Vec<i64>, u64) =
                read_int_array(&self.r, next, count as usize, "SPECS.fieldSetIndexes")?;
            let (k, _): (Vec<i64>, u64) =
                read_int_array(&self.r, next, count as usize, "SPECS.specTypes")?;
            (p, f, k)
        } else {
            let mut p: Vec<i64> = Vec::new();
            let mut f: Vec<i64> = Vec::new();
            let mut k: Vec<i64> = Vec::new();
            let stride: u64 = if self.version == [0, 0, 1] { 16 } else { 12 };
            let skip: u64 = if stride == 16 { 4 } else { 0 };
            for i in 0..count {
                let o: u64 = pos + i as u64 * stride + skip;
                p.push(self.r.u32(o, "Spec.pathIndex")? as i64);
                f.push(self.r.u32(o + 4, "Spec.fieldSetIndex")? as i64);
                k.push(self.r.u32(o + 8, "Spec.specType")? as i64);
            }
            (p, f, k)
        };
        let mut out: Vec<Spec> = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            if p[i] < 0 || f[i] < 0 || k[i] < 0 {
                return Err(err_value("SPECS 出现负索引"));
            }
            out.push(Spec { path: p[i] as usize, fieldset: f[i] as usize, kind: k[i] as u32 });
        }
        self.specs = out;
        Ok(())
    }

    /// 取一个 spec 的字段（fieldset 以 `0xFFFFFFFF` 分隔）。
    fn spec_fields(&self, spec: &Spec) -> Result<Vec<(String, CrateValue)>, UsdError> {
        if spec.fieldset >= self.fieldsets.len() {
            return Err(err_value(format!(
                "fieldSetIndex {} 越界（表长 {}）",
                spec.fieldset,
                self.fieldsets.len()
            )));
        }
        let mut out: Vec<(String, CrateValue)> = Vec::new();
        let mut i: usize = spec.fieldset;
        while i < self.fieldsets.len() {
            let v: i64 = self.fieldsets[i];
            if v == -1 || v == 0xFFFF_FFFF {
                break;
            }
            if v < 0 || v as usize >= self.fields.len() {
                return Err(err_value(format!("fieldIndex {v} 越界")));
            }
            let f: &Field = &self.fields[v as usize];
            let value: CrateValue = self.unpack_value(f.rep)?;
            out.push((f.name.clone(), value));
            i += 1;
        }
        Ok(out)
    }

    // ---- ValueRep --------------------------------------------------------
    fn unpack_value(&self, rep: u64) -> Result<CrateValue, UsdError> {
        let type_id: u32 = ((rep >> 48) & 0xff) as u32;
        let is_array: bool = rep >> 63 & 1 != 0;
        let is_inlined: bool = rep >> 62 & 1 != 0;
        let is_compressed: bool = rep >> 61 & 1 != 0;
        let payload: u64 = rep & ((1u64 << 48) - 1);
        if is_array {
            return self.unpack_array(type_id, payload, is_compressed);
        }
        if is_inlined {
            return self.unpack_inline(type_id, payload);
        }
        // 非内联：payload 是**文件绝对偏移**。
        let off: u64 = payload;
        match type_id {
            1 => Ok(CrateValue::Bool(self.r.u8(off, "bool")? != 0)),
            2 => Ok(CrateValue::Number(self.r.u8(off, "uchar")? as f64)),
            3 => Ok(CrateValue::Number(self.r.i32(off, "int")? as f64)),
            4 => Ok(CrateValue::Number(self.r.u32(off, "uint")? as f64)),
            5 => Ok(CrateValue::Number(self.r.i64(off, "int64")? as f64)),
            6 => Ok(CrateValue::Number(self.r.u64(off, "uint64")? as f64)),
            7 => Ok(CrateValue::Number(half_to_f64(self.r.u16(off, "half")?))),
            8 => Ok(CrateValue::Number(self.r.f32(off, "float")?)),
            9 => Ok(CrateValue::Number(self.r.f64(off, "double")?)),
            10 | 12 => {
                let idx: u32 = self.r.u32(off, "StringIndex")?;
                Ok(CrateValue::Str(self.string(idx as i64)?))
            }
            11 => {
                let idx: u32 = self.r.u32(off, "TokenIndex")?;
                Ok(CrateValue::Str(self.token(idx as i64)?))
            }
            19 | 20 | 21 | 22 => self.read_vec(off, 2, type_id), // Vec2*
            23 | 24 | 25 | 26 => self.read_vec(off, 3, type_id), // Vec3*
            27 | 28 | 29 | 30 => self.read_vec(off, 4, type_id), // Vec4*
            16 | 17 | 18 => self.read_vec(off, 4, type_id),       // Quat*
            13 => self.read_matrix(off, 2, 13),
            14 => self.read_matrix(off, 3, 14),
            15 => self.read_matrix(off, 4, 15),
            31 => Ok(CrateValue::None), // 空 VtDictionary（内联/非内联都无数据）
            40 => self.read_index_vector(off, true),  // PathVector
            41 => self.read_index_vector(off, false), // TokenVector
            50 => self.read_index_vector(off, false), // StringVector
            32 | 33 | 34 => self.read_list_op(off, type_id),
            35 | 36 | 37 | 38 | 39 | 55 => Err(err_value(format!(
                "list op type={type_id} 的元素不是 u32 索引（本版本未实现）"
            ))),
            42 => Ok(CrateValue::Specifier(self.r.u32(off, "Specifier")?)),
            43 => Ok(CrateValue::Number(self.r.u32(off, "Permission")? as f64)),
            44 => Ok(CrateValue::Variability(self.r.u32(off, "Variability")?)),
            46 => Ok(CrateValue::TimeSamples),
            other => Err(err_value(format!(
                "未实现的非内联 value rep type={other}（payload 偏移 {off}）"
            ))),
        }
    }

    fn unpack_inline(&self, type_id: u32, payload: u64) -> Result<CrateValue, UsdError> {
        let bits: u32 = payload as u32;
        match type_id {
            1 => Ok(CrateValue::Bool(bits != 0)),
            2 => Ok(CrateValue::Number((bits & 0xff) as f64)),
            3 => Ok(CrateValue::Number((bits as i32) as f64)),
            4 => Ok(CrateValue::Number(bits as f64)),
            8 | 9 => Ok(CrateValue::Number(f32::from_bits(bits) as f64)),
            10 | 12 => Ok(CrateValue::Str(self.string(bits as i64)?)),
            11 => Ok(CrateValue::Str(self.token(bits as i64)?)),
            42 => Ok(CrateValue::Specifier(bits)),
            43 => Ok(CrateValue::Number(bits as f64)),
            44 => Ok(CrateValue::Variability(bits)),
            31 => Ok(CrateValue::None),
            // 向量/四元数/矩阵：payload 的低位字节是各分量的 int8。
            19 | 20 | 21 | 22 => Ok(CrateValue::Numbers(inline_ints(bits, 2))),
            23 | 24 | 25 | 26 => Ok(CrateValue::Numbers(inline_ints(bits, 3))),
            27 | 28 | 29 | 30 | 16 | 17 | 18 => Ok(CrateValue::Numbers(inline_ints(bits, 4))),
            13 => Ok(CrateValue::Numbers(inline_ints(bits, 2))),
            14 => Ok(CrateValue::Numbers(inline_ints(bits, 3))),
            15 => Ok(CrateValue::Numbers(inline_ints(bits, 4))),
            other => Err(err_value(format!("未实现的内联 value rep type={other}"))),
        }
    }

    fn read_vec(&self, off: u64, n: usize, type_id: u32) -> Result<CrateValue, UsdError> {
        let mut v: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let o: u64 = off + (i as u64) * elem_size(type_id);
            let x: f64 = match type_id {
                19 | 23 | 27 | 16 => self.r.f64(o, "vec.d")?,
                20 | 24 | 28 | 17 => self.r.f32(o, "vec.f")?,
                21 | 25 | 29 | 18 => half_to_f64(self.r.u16(o, "vec.h")?),
                22 | 26 | 30 => self.r.i32(o, "vec.i")? as f64,
                _ => return Err(err_value(format!("vec type {type_id} 不支持"))),
            };
            v.push(x);
        }
        Ok(CrateValue::Numbers(v))
    }

    fn read_matrix(&self, off: u64, n: usize, type_id: u32) -> Result<CrateValue, UsdError> {
        let sz: u64 = elem_size(type_id);
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for r in 0..n {
            let mut row: Vec<f64> = Vec::with_capacity(n);
            for c in 0..n {
                let o: u64 = off + ((r * n + c) as u64) * sz;
                let x: f64 = match type_id {
                    13 | 14 | 15 => self.r.f64(o, "matrix.d")?,
                    _ => return Err(err_value("matrix 只实现 double")),
                };
                row.push(x);
            }
            rows.push(row);
        }
        Ok(CrateValue::Matrix(rows))
    }

    /// `vector<Path>` / `vector<Token>` / `vector<string>`：`u64 count` + `u32` 索引。
    fn read_index_vector(&self, off: u64, is_path: bool) -> Result<CrateValue, UsdError> {
        let count: u64 = self.r.u64(off, "vector.count")?;
        let mut out: Vec<String> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let idx: u32 = self.r.u32(off + 8 + i * 4, "vector.index")?;
            out.push(if is_path { self.path(idx as i64)? } else { self.token(idx as i64)? });
        }
        Ok(if is_path { CrateValue::Paths(out) } else { CrateValue::Strings(out) })
    }

    /// `Sdf*ListOp`：`u8 header` + 若干 `vector<T>`（按 bit 出现顺序）。
    fn read_list_op(&self, off: u64, type_id: u32) -> Result<CrateValue, UsdError> {
        const IS_EXPLICIT: u8 = 1;
        const HAS_EXPLICIT: u8 = 2;
        const HAS_ADDED: u8 = 4;
        const HAS_DELETED: u8 = 8;
        const HAS_ORDERED: u8 = 16;
        const HAS_PREPENDED: u8 = 32;
        const HAS_APPENDED: u8 = 64;
        let bits: u8 = self.r.u8(off, "ListOp.header")?;
        let mut pos: u64 = off + 1;
        let is_path: bool = type_id == 34;
        let mut all: Vec<String> = Vec::new();
        let masks: [u8; 6] = [HAS_EXPLICIT, HAS_ADDED, HAS_PREPENDED, HAS_APPENDED, HAS_DELETED, HAS_ORDERED];
        for m in masks {
            if bits & m == 0 {
                continue;
            }
            let count: u64 = self.r.u64(pos, "ListOp.vector.count")?;
            pos += 8;
            for i in 0..count {
                let idx: u32 = self.r.u32(pos + i as u64 * 4, "ListOp.vector.index")?;
                all.push(if is_path { self.path(idx as i64)? } else { self.token(idx as i64)? });
            }
            pos += count * 4;
        }
        let _ = IS_EXPLICIT;
        Ok(if is_path { CrateValue::Paths(all) } else { CrateValue::Strings(all) })
    }

    fn unpack_array(&self, type_id: u32, payload: u64, is_compressed: bool) -> Result<CrateValue, UsdError> {
        let off: u64 = payload;
        if off == 0 {
            return Ok(CrateValue::Numbers(Vec::new()));
        }
        let count: u64 = if self.array_count_u64() {
            self.r.u64(off, "array.count")?
        } else {
            self.r.u32(off, "array.count")? as u64
        };
        let mut pos: u64 = off + if self.array_count_u64() { 8 } else { 4 };
        // 压缩整数数组（0.5.0+）：`[u64 compressedSize][TfFastCompression ints]`
        if is_compressed && matches!(type_id, 3 | 4 | 5 | 6) {
            let size: u64 = self.r.u64(pos, "compressedArray.size")?;
            pos += 8;
            let src: &[u8] = self.r.slice(pos, size, "compressedArray.data")?;
            let vals: Vec<i64> = decode_ints(src, count as usize)?;
            return Ok(CrateValue::Numbers(vals.iter().map(|&v| v as f64).collect()));
        }
        // 压缩浮点数组（0.6.0+）：1 B code（`i` 整数 / `t` LUT）。
        if is_compressed && matches!(type_id, 7 | 8 | 9) {
            let code: u8 = self.r.u8(pos, "compressedFloatArray.code")?;
            pos += 1;
            if code == b'i' {
                let size: u64 = self.r.u64(pos, "compressedArray.size")?;
                pos += 8;
                let src: &[u8] = self.r.slice(pos, size, "compressedArray.data")?;
                let vals: Vec<i64> = decode_ints(src, count as usize)?;
                return Ok(CrateValue::Numbers(vals.iter().map(|&v| v as f64).collect()));
            }
            if code == b't' {
                let lut_n: u32 = self.r.u32(pos, "lut.size")?;
                pos += 4;
                let mut lut: Vec<f64> = Vec::with_capacity(lut_n as usize);
                for i in 0..lut_n as u64 {
                    let o: u64 = pos + i * elem_size(type_id);
                    lut.push(match type_id {
                        8 => self.r.f32(o, "lut.f")?,
                        9 => self.r.f64(o, "lut.d")?,
                        _ => half_to_f64(self.r.u16(o, "lut.h")?),
                    });
                }
                pos += lut_n as u64 * elem_size(type_id);
                let size: u64 = self.r.u64(pos, "lut.indexes.size")?;
                pos += 8;
                let src: &[u8] = self.r.slice(pos, size, "lut.indexes")?;
                let idx: Vec<i64> = decode_ints(src, count as usize)?;
                let mut out: Vec<f64> = Vec::with_capacity(count as usize);
                for v in idx {
                    let i: usize = (v as u32) as usize;
                    out.push(*lut.get(i).ok_or_else(|| err_value("LUT index 越界"))?);
                }
                return Ok(CrateValue::Numbers(out));
            }
            return Err(err_value(format!("未知压缩浮点数组 code `{}`", code as char)));
        }
        if is_compressed {
            return Err(err_value(format!("type={type_id} 的压缩数组未实现")));
        }
        // 未压缩：按元素类型直读。
        let es: u64 = elem_size(type_id);
        match type_id {
            1 => {
                let mut v: Vec<f64> = Vec::with_capacity(count as usize);
                for i in 0..count {
                    v.push(self.r.u8(pos + i, "bool[]")? as f64);
                }
                Ok(CrateValue::Numbers(v))
            }
            2 => {
                let mut v: Vec<f64> = Vec::with_capacity(count as usize);
                for i in 0..count {
                    v.push(self.r.u8(pos + i, "uchar[]")? as f64);
                }
                Ok(CrateValue::Numbers(v))
            }
            3 | 4 | 5 | 6 | 7 | 8 | 9 => {
                let mut v: Vec<f64> = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let o: u64 = pos + i * es;
                    v.push(match type_id {
                        3 => self.r.i32(o, "int[]")? as f64,
                        4 => self.r.u32(o, "uint[]")? as f64,
                        5 => self.r.i64(o, "int64[]")? as f64,
                        6 => self.r.u64(o, "uint64[]")? as f64,
                        7 => half_to_f64(self.r.u16(o, "half[]")?),
                        8 => self.r.f32(o, "float[]")?,
                        _ => self.r.f64(o, "double[]")?,
                    });
                }
                Ok(CrateValue::Numbers(v))
            }
            10 | 11 | 12 => {
                let mut v: Vec<String> = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let idx: u32 = self.r.u32(pos + i as u64 * 4, "string/token[]")?;
                    v.push(if type_id == 10 {
                        self.string(idx as i64)?
                    } else {
                        self.token(idx as i64)?
                    });
                }
                Ok(CrateValue::Strings(v))
            }
            19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 16 | 17 | 18 => {
                let n: usize = vec_dim(type_id);
                let mut rows: Vec<Vec<f64>> = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let mut row: Vec<f64> = Vec::with_capacity(n);
                    for c in 0..n {
                        let o: u64 = pos + (i as u64 * n as u64 + c as u64) * es;
                        row.push(match type_id {
                            19 | 23 | 27 | 16 => self.r.f64(o, "vec[].d")?,
                            20 | 24 | 28 | 17 => self.r.f32(o, "vec[].f")?,
                            21 | 25 | 29 | 18 => half_to_f64(self.r.u16(o, "vec[].h")?),
                            _ => self.r.i32(o, "vec[].i")? as f64,
                        });
                    }
                    rows.push(row);
                }
                Ok(CrateValue::Rows(rows))
            }
            15 => {
                let mut rows: Vec<Vec<f64>> = Vec::with_capacity(count as usize * 4);
                for i in 0..count {
                    for r in 0..4 {
                        let mut row: Vec<f64> = Vec::with_capacity(4);
                        for c in 0..4 {
                            let o: u64 = pos + (i * 16 + r as u64 * 4 + c as u64) * 8;
                            row.push(self.r.f64(o, "matrix4d[]")?);
                        }
                        rows.push(row);
                    }
                }
                Ok(CrateValue::Matrix(rows))
            }
            other => Err(err_value(format!("未实现的数组 value rep type={other}"))),
        }
    }

    // ---- Layer 装配 ------------------------------------------------------
    fn build_layer(&self) -> Result<Layer, UsdError> {
        // 1) 建 prim 表（path → Prim），字段逐条解出。
        let mut prims: Vec<Prim> = Vec::new();
        let mut prim_index: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        let mut children_names: Vec<Vec<String>> = Vec::new();
        let mut prop_names: Vec<Vec<String>> = Vec::new();
        let mut metadata: Vec<(String, String)> = Vec::new();

        for spec in &self.specs {
            let path: String = self.path(spec.path as i64)?;
            let fields: Vec<(String, CrateValue)> = self.spec_fields(spec)?;
            if spec.kind == 7 || path == "/" {
                // PseudoRoot：层元数据（跳过结构字段 primChildren / properties）。
                for (k, v) in &fields {
                    if k == "primChildren" || k == "properties" || k == "specifier" {
                        continue;
                    }
                    if let Some(s) = crate_value_to_string(v) {
                        metadata.push((k.clone(), s));
                    }
                }
                continue;
            }
            if spec.kind == 6 {
                let mut p = Prim {
                    kind: PrimKind::Def,
                    type_name: String::new(),
                    name: path.rsplit('/').next().unwrap_or("").to_string(),
                    path: path.clone(),
                    attributes: Vec::new(),
                    relationships: Vec::new(),
                    children: Vec::new(),
                    xform_op_order: Vec::new(),
                    references: Vec::new(),
                    payloads: Vec::new(),
                    variant_sets: Vec::new(),
                };
                let mut kids: Vec<String> = Vec::new();
                let mut props: Vec<String> = Vec::new();
                for (k, v) in &fields {
                    match k.as_str() {
                        "specifier" => {
                            if let CrateValue::Specifier(s) = v {
                                p.kind = match *s {
                                    0 => PrimKind::Def,
                                    1 => PrimKind::Over,
                                    2 => PrimKind::Class,
                                    _ => PrimKind::Def,
                                };
                            }
                        }
                        "typeName" => {
                            if let Some(s) = crate_value_to_string(v) {
                                p.type_name = s;
                            }
                        }
                        "primChildren" => {
                            if let CrateValue::Strings(v) = v {
                                kids = v.clone();
                            }
                        }
                        "properties" => {
                            if let CrateValue::Strings(v) = v {
                                props = v.clone();
                            }
                        }
                        "references" => {
                            if let CrateValue::Strings(v) = v {
                                p.references = v.clone();
                                p.references.sort();
                            }
                        }
                        "payload" => {
                            if let CrateValue::Strings(v) = v {
                                p.payloads = v.clone();
                                p.payloads.sort();
                            }
                        }
                        "variantSet" | "variantSets" => {
                            if let CrateValue::Strings(v) = v {
                                p.variant_sets = v.clone();
                                p.variant_sets.sort();
                            }
                        }
                        _ => {}
                    }
                }
                prim_index.insert(path.clone(), prims.len());
                children_names.push(kids);
                prop_names.push(props);
                prims.push(p);
            }
        }

        // 2) 属性 / 关系 spec 挂到父 prim（按 properties 声明顺序）。
        for spec in &self.specs {
            if spec.kind != 1 && spec.kind != 8 {
                continue;
            }
            let path: String = self.path(spec.path as i64)?;
            let (parent, name) = split_property_path(&path).ok_or_else(|| {
                err_value(format!("属性/关系 spec 路径 `{path}` 不是 prim property path"))
            })?;
            let pi: usize = *prim_index.get(&parent).ok_or_else(|| {
                err_value(format!("属性 `{path}` 的父 prim `{parent}` 不在 prim 表中"))
            })?;
            let fields: Vec<(String, CrateValue)> = self.spec_fields(spec)?;
            if spec.kind == 8 {
                let mut targets: Vec<String> = Vec::new();
                for (k, v) in &fields {
                    if k == "targetPaths" {
                        match v {
                            CrateValue::Paths(p) => targets = p.clone(),
                            CrateValue::Strings(s) => targets = s.clone(),
                            _ => {}
                        }
                    }
                }
                prims[pi].relationships.push(Relationship { name: name.clone(), targets });
            } else {
                // 两遍：先取 typeName（crate 不保证字段顺序），再按类型解 default。
                let type_name: String = fields
                    .iter()
                    .find(|(k, _)| k == "typeName")
                    .and_then(|(_, v)| crate_value_to_string(v))
                    .unwrap_or_default();
                let mut value: AttrValue = AttrValue::None;
                let mut uniform: bool = false;
                let mut custom: bool = false;
                let mut interpolation: Option<String> = None;
                let mut has_time_samples: bool = false;
                for (k, v) in &fields {
                    match k.as_str() {
                        "typeName" => {}
                        "default" => value = to_attr_value(v, &type_name),
                        "variability" => {
                            if let CrateValue::Variability(x) = v {
                                uniform = *x == 1;
                            }
                        }
                        "custom" => {
                            if let CrateValue::Bool(b) = v {
                                custom = *b;
                            }
                        }
                        "interpolation" => interpolation = crate_value_to_string(v),
                        "timeSamples" => has_time_samples = true,
                        _ => {}
                    }
                }
                if type_name.is_empty() {
                    // 有些 crate 把类型名放在 default 的 rep 里；缺类型名仍要显式登记，
                    // 不能静默当 None（C20）。
                    return Err(err_value(format!("属性 `{path}` 缺 typeName 字段")));
                }
                if name == "xformOpOrder" {
                    prims[pi].xform_op_order = value.as_strings();
                }
                prims[pi].attributes.push(Attribute {
                    name: name.clone(),
                    type_name,
                    value,
                    uniform,
                    custom,
                    interpolation,
                    has_time_samples,
                });
            }
        }

        // 3) 按 `properties` / `primChildren` 的声明顺序重排（与 `.usda` 源顺序一致）。
        for i in 0..prims.len() {
            let order: Vec<String> = prop_names.get(i).cloned().unwrap_or_default();
            prims[i].attributes.sort_by(|a, b| {
                let ia: usize = order.iter().position(|n| n == &a.name).unwrap_or(usize::MAX);
                let ib: usize = order.iter().position(|n| n == &b.name).unwrap_or(usize::MAX);
                ia.cmp(&ib)
            });
            prims[i].relationships.sort_by(|a, b| {
                let ia: usize = order.iter().position(|n| n == &a.name).unwrap_or(usize::MAX);
                let ib: usize = order.iter().position(|n| n == &b.name).unwrap_or(usize::MAX);
                ia.cmp(&ib)
            });
        }

        // 4) 组装 prim 树（children 按 primChildren 顺序；无声明时按路径升序兜底）。
        let mut roots: Vec<Prim> = Vec::new();
        let mut all_paths: Vec<String> = Vec::new();
        // 先按路径深度升序建索引，保证父节点先于子节点处理。
        let mut order: Vec<usize> = (0..prims.len()).collect();
        order.sort_by(|&a, &b| {
            let da: usize = prims[a].path.matches('/').count();
            let db: usize = prims[b].path.matches('/').count();
            da.cmp(&db).then_with(|| prims[a].path.cmp(&prims[b].path))
        });
        let mut built: std::collections::BTreeMap<String, Prim> = std::collections::BTreeMap::new();
        for &i in order.iter().rev() {
            let mut p: Prim = prims[i].clone();
            p.children = Vec::new();
            for child in prims.iter() {
                if parent_path(&child.path).as_deref() == Some(p.path.as_str()) {
                    if let Some(c) = built.remove(&child.path) {
                        p.children.push(c);
                    }
                }
            }
            let kids: Vec<String> = children_names.get(i).cloned().unwrap_or_default();
            p.children.sort_by(|a, b| {
                let ia: usize = kids.iter().position(|n| n == &a.name).unwrap_or(usize::MAX);
                let ib: usize = kids.iter().position(|n| n == &b.name).unwrap_or(usize::MAX);
                ia.cmp(&ib).then_with(|| a.path.cmp(&b.path))
            });
            built.insert(p.path.clone(), p);
        }
        for p in prims.iter() {
            all_paths.push(p.path.clone());
            if parent_path(&p.path).as_deref() == Some("/") || !p.path[1..].contains('/') {
                if let Some(top) = built.remove(&p.path) {
                    roots.push(top);
                }
            }
        }
        roots.sort_by(|a, b| a.path.cmp(&b.path));
        all_paths.sort();
        all_paths.dedup();
        Ok(Layer {
            // `.usdc` 没有 `#usda 1.0` 头；`Layer.version` 记的是 **USD 层语法版本**，
            // 统一规范为 "1"（与 `#usda 1.0` 的 `parse_header` 结果一致），
            // crate 自身的二进制版本由 `usdc::inspect` 单独给出。
            version: "1".to_string(),
            metadata,
            prims: roots,
            all_paths,
        })
    }
}

fn fmt_ver(v: [u8; 3]) -> String {
    format!("{}.{}.{}", v[0], v[1], v[2])
}

fn elem_size(type_id: u32) -> u64 {
    match type_id {
        7 | 21 | 25 | 29 | 18 => 2, // half
        3 | 4 | 8 | 20 | 22 | 24 | 26 | 28 | 30 | 17 => 4,
        5 | 6 | 9 | 19 | 23 | 27 | 16 | 13 | 14 | 15 => 8,
        _ => 4,
    }
}

fn vec_dim(type_id: u32) -> usize {
    match type_id {
        19 | 20 | 21 | 22 => 2,
        23 | 24 | 25 | 26 => 3,
        27 | 28 | 29 | 30 | 16 | 17 | 18 => 4,
        _ => 3,
    }
}

/// 内联向量的 int8 分量（payload 低位字节，LE）。
fn inline_ints(bits: u32, n: usize) -> Vec<f64> {
    let b: [u8; 4] = bits.to_le_bytes();
    (0..n).map(|i| (b[i] as i8) as f64).collect()
}

/// 2 的整数次幂（逐次连乘，零 libm、确定性）。
fn pow2_int(n: i32) -> f64 {
    let mut v: f64 = 1.0;
    if n >= 0 {
        for _ in 0..n { v *= 2.0; }
    } else {
        for _ in 0..(-n) { v *= 0.5; }
    }
    v
}

/// IEEE 754 binary16 → f64（自写，零 libm）。
fn half_to_f64(h: u16) -> f64 {
    let sign: f64 = if h >> 15 != 0 { -1.0 } else { 1.0 };
    let exp: u32 = ((h >> 10) & 0x1f) as u32;
    let frac: u32 = (h & 0x3ff) as u32;
    if exp == 0 {
        sign * (frac as f64) / 1024.0 * pow2_int(-14)
    } else if exp == 31 {
        if frac == 0 { sign * f64::INFINITY } else { f64::NAN }
    } else {
        sign * (1.0 + frac as f64 / 1024.0) * pow2_int(exp as i32 - 15)
    }
}

fn crate_value_to_string(v: &CrateValue) -> Option<String> {
    match v {
        CrateValue::Str(s) => Some(s.clone()),
        CrateValue::Number(n) => Some(format!("{n}")),
        CrateValue::Bool(b) => Some(if *b { "true" } else { "false" }.to_string()),
        CrateValue::Strings(v) => Some(v.join(",")),
        CrateValue::Paths(v) => Some(v.join(",")),
        _ => None,
    }
}

/// 把 CrateValue 映射成 [`AttrValue`]（按属性声明类型 `type_name` 分流）。
fn to_attr_value(v: &CrateValue, type_name: &str) -> AttrValue {
    let base: &str = type_name.trim_end_matches("[]");
    let is_array: bool = type_name.ends_with("[]");
    match v {
        CrateValue::None | CrateValue::TimeSamples => AttrValue::None,
        CrateValue::Bool(b) => AttrValue::Bool(*b),
        CrateValue::Number(n) => AttrValue::Number(*n),
        CrateValue::Str(s) => {
            if base == "asset" {
                AttrValue::Asset(s.clone())
            } else {
                AttrValue::Str(s.clone())
            }
        }
        CrateValue::Numbers(ns) => {
            if is_array {
                AttrValue::Numbers(ns.clone())
            } else if matches!(base, "matrix4d" | "matrix4f") {
                // 内联矩阵的对角/行主序展开按一维数值保留（渲染侧按 type_name 审计）。
                AttrValue::Numbers(ns.clone())
            } else {
                AttrValue::Numbers(ns.clone())
            }
        }
        CrateValue::Rows(rows) => AttrValue::Rows(rows.clone()),
        CrateValue::Matrix(rows) => AttrValue::Matrix(rows.clone()),
        CrateValue::Strings(v) => AttrValue::Strings(v.clone()),
        CrateValue::Paths(v) => AttrValue::Paths(v.clone()),
        CrateValue::Specifier(s) => AttrValue::Number(*s as f64),
        CrateValue::Variability(s) => AttrValue::Number(*s as f64),
    }
}

/// `/root/cube.points` → (`/root/cube`, `points`)；属性名里允许 `:`。
fn split_property_path(path: &str) -> Option<(String, String)> {
    let at: usize = path.rfind('.')?;
    if at == 0 || at + 1 >= path.len() {
        return None;
    }
    Some((path[..at].to_string(), path[at + 1..].to_string()))
}

/// `/root/cube` → `/root`；顶层 `/root` → `/`。
fn parent_path(path: &str) -> Option<String> {
    let at: usize = path.rfind('/')?;
    if at == 0 {
        Some("/".to_string())
    } else {
        Some(path[..at].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_block_handles_matches_and_literals() {
        // "abc" + 6 个 'a' 的 match（offset=3）+ 尾字面 "z"
        let src: [u8; 6] = [0x31, b'a', b'b', b'c', 0x03, 0x00];
        let out: Vec<u8> = lz4_block(&src, 16).unwrap();
        assert_eq!(out, b"abcabcab");
        // match offset=0 → 必须结构化报错
        let src2: [u8; 5] = [0x10, b'x', 0x00, 0x00, b'z'];
        assert!(lz4_block(&src2, 16).is_err());
    }

    #[test]
    fn decode_ints_roundtrip_known_vector() {
        // TfFastCompression 单块：token 0x50 = 5 字面，载荷 [common=1, codes=0]
        let comp: [u8; 7] = [0, 0x50, 1, 0, 0, 0, 0];
        let out: Vec<i64> = decode_ints(&comp, 4).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4]);
    }
}
