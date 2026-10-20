//! OpenType / SFNT 容器与布局表解析（补完； 文字管线）。
//!
//! # 与 [`crate::ttf`] 的分工（不重复造轮子）
//!
//! | 关注点 | 归属 |
//! | --- | --- |
//! | SFNT 容器 / 表目录 / **逐表校验和** | 本模块 |
//! | `head` `maxp` `hhea` `hmtx` `cmap` `loca` `glyf` **取值** | [`crate::ttf::Font`]（已验证的唯一实现） |
//! | `GSUB` / `GPOS` 布局表结构与施加 | [`crate::gsub`] |
//! | `name` 字符串表 | 本模块 |
//!
//! 字形级读取（`cmap` format 4/12 查找、`loca` 区间、`glyf` 二次贝塞尔与
//! 复合字形展开）**一律委派**给 [`crate::ttf::Font`] —— 那里的
//! `maxp.numGlyphs` 是 u16、`glyf_slice` 双向钳制等防御已被44 个既有测试
//! 钉住，重写第二份只会引入第二个真相源。本模块提供的是 `ttf.rs` **没有**的
//! 能力：表目录与校验和的**结构化验证**、`name` 解析、轮廓格式判定
//! （glyf / CFF / CFF2）与布局表的原始字节出口。
//!
//! # 轮廓格式
//!
//! 内嵌资产 `PrismSansSC-*.ttf` 是** glyf 轮廓**（`sfntVersion == 0x00010000`，
//! 目录含 `glyf`+`loca`，6888 简单字形 / 0 复合字形）。**CFF2 与 CFF 均不在
//! 支持范围**：本引擎的自研解析器只解 TrueType 二次轮廓，遇到 `OTTO` /
//! `CFF `/`CFF2` 表一律返回结构化错误 [`FontError`]（码 `T016`）而**不是**
//! 静默错图或 panic。
//!
//! # 确定性
//!
//! 表目录按**文件内出现顺序**存 `Vec`（不排序、不用 HashMap），校验和遍历
//! 序即目录序。`name` 记录同样按目录序返回。全部读取为大端整数（OpenType
//! 规范），越界一律经 [`Reader`] 归一为 [`FontError`]。同字节流 ⇒ 同结果。
//!
//! # 越界纪律（宪章 C18）
//!
//! 本模块**零** `unwrap()` / `expect()` / `panic!()` / 裸下标 / 整数溢出。
//! 表目录的 `offset`/`length` 是外部数据，`offset + length` 一律走
//! `checked_add`；[`Reader`] 的每次读取都经 `slice::get`。

use crate::ttf::{Font, FontError, Outline};

/// 结构化错误快捷构造（宪章 C18 三段式：在哪 / 什么事 / 怎么改）。
fn err(code: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> FontError {
    FontError { code, problem: problem.into(), fix: fix.into() }
}

// ===========================================================================
//  边界检查的大端游标
// ===========================================================================

/// 边界检查的大端读取游标（OpenType 是大端）。
///
/// 所有布局表解析（`GSUB` / `GPOS` / `name` / `cmap` 目录）共用本游标，
/// 是全模块**唯一**的字节读取入口：越界即 [`FontError`]，绝不 panic。
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// 从字节流起点构造。
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    /// 从指定偏移构造（偏移越界在首次读取时报错，不 panic）。
    pub fn at(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos }
    }

    /// 当前游标偏移。
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 剩余字节数（饱和减，下溢归0）。
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// 定位到绝对偏移（越界 → 结构化错误）。
    pub fn seek(&mut self, pos: usize) -> Result<(), FontError> {
        if pos > self.data.len() {
            return Err(err("T011", format!("游标定位越界：偏移 {pos} > 流长 {}", self.data.len()),
                "表偏移/length 字段损坏；确认字体文件完整未截断"));
        }
        self.pos = pos;
        Ok(())
    }

    /// 前进n 字节（越界 → 结构化错误）。
    pub fn skip(&mut self, n: usize) -> Result<(), FontError> {
        let next = self.pos.checked_add(n).ok_or_else(|| {
            err("T011", format!("游标前进溢出：{}+{}", self.pos, n), "表结构字段损坏")
        })?;
        self.seek(next)
    }

    /// 读 1 字节。
    pub fn u8(&mut self) -> Result<u8, FontError> {
        let b = *self.data.get(self.pos).ok_or_else(|| {
            err("T011", format!("读u8 越界（偏移 {}，流长 {}）", self.pos, self.data.len()),
                "表结构字段损坏或表被截断")
        })?;
        self.pos = self.pos.saturating_add(1);
        Ok(b)
    }

    /// 读大端 u16。
    pub fn u16(&mut self) -> Result<u16, FontError> {
        let hi = self.u8()? as u16;
        let lo = self.u8()? as u16;
        Ok((hi << 8) | lo)
    }

    /// 读大端 i16。
    pub fn i16(&mut self) -> Result<i16, FontError> {
        Ok(self.u16()? as i16)
    }

    /// 读大端 u32。
    pub fn u32(&mut self) -> Result<u32, FontError> {
        let hi = self.u16()? as u32;
        let lo = self.u16()? as u32;
        Ok((hi << 16) | lo)
    }

    /// 读 4 字节 tag。
    pub fn tag(&mut self) -> Result<[u8; 4], FontError> {
        let a = self.u8()?;
        let b = self.u8()?;
        let c = self.u8()?;
        let d = self.u8()?;
        Ok([a, b, c, d])
    }

    /// 读 `n` 字节切片（越界 → 结构化错误；返回的切片借自底层流）。
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], FontError> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            err("T011", format!("读 {n} B 长度溢出（偏移 {}）", self.pos), "表结构字段损坏")
        })?;
        let s = self.data.get(self.pos..end).ok_or_else(|| {
            err("T011", format!("读 {n} B 越界（偏移 {}..{end}，流长 {}）", self.pos, self.data.len()),
                "表结构字段损坏或表被截断")
        })?;
        self.pos = end;
        Ok(s)
    }
}

// ===========================================================================
//  表tag 常量
// ===========================================================================

/// `head` 表tag。
pub const TAG_HEAD: [u8; 4] = *b"head";
/// `maxp` 表 tag。
pub const TAG_MAXP: [u8; 4] = *b"maxp";
/// `hhea` 表 tag。
pub const TAG_HHEA: [u8; 4] = *b"hhea";
/// `hmtx` 表 tag。
pub const TAG_HMTX: [u8; 4] = *b"hmtx";
/// `cmap` 表 tag。
pub const TAG_CMAP: [u8; 4] = *b"cmap";
/// `loca` 表 tag。
pub const TAG_LOCA: [u8; 4] = *b"loca";
/// `glyf` 表 tag。
pub const TAG_GLYF: [u8; 4] = *b"glyf";
/// `GSUB` 表 tag（字形替换）。
pub const TAG_GSUB: [u8; 4] = *b"GSUB";
/// `GPOS` 表 tag（字形定位）。
pub const TAG_GPOS: [u8; 4] = *b"GPOS";
/// `name` 表 tag。
pub const TAG_NAME: [u8; 4] = *b"name";
/// `CFF ` 表 tag（Compact Font Format，Type 1B 轮廓）。
pub const TAG_CFF: [u8; 4] = *b"CFF ";
/// `CFF2` 表 tag（Compact Font Format 2）。
pub const TAG_CFF2: [u8; 4] = *b"CFF2";

/// TrueType 轮廓的 `sfntVersion`。
pub const SFNT_TRUETYPE: u32 = 0x0001_0000;
/// CFF 轮廓 OTF 的 `sfntVersion`（`'OTTO'`）。
pub const SFNT_OTTO: u32 = 0x4F54_544F;

/// `head.magicNumber` 的正确值。
pub const HEAD_MAGIC: u32 = 0x5F0F_3CF5;

// ===========================================================================
//  表目录
// ===========================================================================

/// 一条表目录记录。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableRecord {
    /// 4 字节 tag（如 `b"head"`）。
    pub tag: [u8; 4],
    /// 目录中记录的校验和。
    pub checksum: u32,
    /// 表数据在文件中的偏移。
    pub offset: u32,
    /// 表数据长度（字节）。
    pub length: u32,
}

impl TableRecord {
    /// tag 是否为可打印 ASCII（用于错误信息与调试输出）。
    pub fn tag_str(&self) -> String {
        self.tag.iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect()
    }
}

/// SFNT 表目录（按**文件内出现顺序**，确定性遍历）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDirectory {
    /// `sfntVersion`。
    pub sfnt_version: u32,
    /// 全部表记录，序即文件内序。
    pub records: Vec<TableRecord>,
}

impl TableDirectory {
    /// 解析表目录（校验 12 B 头与每条记录的 `offset + length <=文件长`）。
    ///
    /// **不做**校验和验证（[`OtFont::verify_checksum`] 按需单独调用）——
    /// 目录解析只管结构，这样"校验和错但结构好"的字体仍可解析并被诊断。
    pub fn parse(data: &[u8]) -> Result<Self, FontError> {
        if data.len() < 12 {
            return Err(err("T010", format!("字节流仅 {} B，不足 SFNT 头（12 B）", data.len()),
                "确认传入的是完整 TTF/OTF 文件而非截断片段"));
        }
        let mut r = Reader::new(data);
        let sfnt_version = r.u32()?;
        let num_tables = r.u16()? as usize;
        // searchRange(2) + entrySelector(2) + rangeShift(2)：对定长表目录是
        // 二分搜索的提示值，本解析器线性扫描，跳过即可（表记录从偏移 12 开始）。
        r.skip(6)?;
        // 目录区必须完整落在流内：12 + num_tables*16 <= len。
        let dir_len = num_tables.checked_mul(16).and_then(|n| n.checked_add(12)).ok_or_else(|| {
            err("T010", format!("numTables = {num_tables} 导致目录长度溢出"), "字体文件损坏")
        })?;
        if dir_len > data.len() {
            return Err(err("T010",
                format!("表目录需 {dir_len} B（numTables = {num_tables}），文件仅 {} B", data.len()),
                "字体被截断；确认内嵌资产是完整文件"));
        }
        let mut records = Vec::with_capacity(num_tables.min(4096));
        for i in 0..num_tables {
            let tag = r.tag()?;
            let checksum = r.u32()?;
            let offset = r.u32()?;
            let length = r.u32()?;
            // offset + length 越界 ⇒ 结构化错误（不 panic、不静默跳表）。
            let end = (offset as usize).checked_add(length as usize).ok_or_else(|| {
                err("T010", format!("表 {} (#{i}) 的 offset+length 溢出 u32", tag_str(tag)),
                    "字体文件损坏")
            })?;
            if end > data.len() {
                return Err(err("T010",
                    format!("表 {} (#{i}) 区间 {offset}..{end} 越出文件（{} B）", tag_str(tag), data.len()),
                    "字体文件损坏或被截断；检查构建期资产"));
            }
            records.push(TableRecord { tag, checksum, offset, length });
        }
        Ok(TableDirectory { sfnt_version, records })
    }

    /// 按 tag 线性查找（目录序，**首个**匹配胜出）。
    pub fn find(&self, tag: [u8; 4]) -> Option<&TableRecord> {
        self.records.iter().find(|r| r.tag == tag)
    }

    /// 目录中是否含某表。
    pub fn contains(&self, tag: [u8; 4]) -> bool {
        self.find(tag).is_some()
    }

    /// 全部 tag（目录序）。
    pub fn tags(&self) -> Vec<[u8; 4]> {
        self.records.iter().map(|r| r.tag).collect()
    }
}

/// 4 字节 tag → 可打印字符串（错误信息用）。
fn tag_str(tag: [u8; 4]) -> String {
    tag.iter().map(|&b| if b.is_ascii_graphic() { b as char } else { '.' }).collect()
}

// ===========================================================================
//  轮廓格式
// ===========================================================================

/// 字形轮廓格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlineFlavor {
    /// TrueType 二次贝塞尔（`glyf` + `loca`）—— 本引擎唯一支持。
    Glyf,
    /// CFF / CFF2 三次贝塞尔 —— **不支持**（结构化拒绝）。
    Cff,
}

/// 按表目录判定轮廓格式。
///
/// 判定序：显式 `CFF2` / `CFF ` 表优先，其次 `sfntVersion == 'OTTO'`，
/// 最后回落 `glyf`+`loca`。
pub fn detect_flavor(dir: &TableDirectory) -> OutlineFlavor {
    if dir.contains(TAG_CFF2) || dir.contains(TAG_CFF) || dir.sfnt_version == SFNT_OTTO {
        return OutlineFlavor::Cff;
    }
    OutlineFlavor::Glyf
}

// ===========================================================================
//  head / maxp / hhea
// ===========================================================================

/// `head` 表关键字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    /// em 方框单位数（16..=16384）。
    pub units_per_em: u16,
    /// `loca` 条目格式：0 = 短（offset/2），1 = 长。
    pub index_to_loc_format: i16,
    /// 字体全局包围盒 `[xMin, yMin, xMax, yMax]`。
    pub bbox: [i16; 4],
    /// `magicNumber`（应为 [`HEAD_MAGIC`]）。
    pub magic: u32,
}

/// `maxp` 表关键字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maxp {
    /// 字形总数。
    pub num_glyphs: u16,
    /// `maxp` 表版本（0x00005000 = 0.5，0x00010000 = 1.0）。
    pub version: u32,
}

/// `hhea` 表关键字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hhea {
    /// 上伸部。
    pub ascender: i16,
    /// 下伸部（负）。
    pub descender: i16,
    /// 行间隙。
    pub line_gap: i16,
    /// `hmtx` 中的长度量条数。
    pub num_h_metrics: u16,
}

// ===========================================================================
//  name 表
// ===========================================================================

/// 一条 `name` 记录（字符串已按平台编码解码）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameRecord {
    /// 平台 ID（3 = Windows，1 = Macintosh，0 = Unicode）。
    pub platform_id: u16,
    /// 编码 ID（3/1 与 0/4 视作 UTF-16BE；1/0 视作 Latin-1）。
    pub encoding_id: u16,
    /// 语言 ID。
    pub language_id: u16,
    /// 名称 ID（1 = 家族名，2 = 子家族名，4 = 全名，6 = PostScript 名，13/14 = 许可）。
    pub name_id: u16,
    /// 解码后的字符串（解码失败 → 空串）。
    pub value: String,
}

/// UTF-16BE → `String`（手写，零 `unicode-*` 依赖）。
///
/// 处理代理对（U+D800–U+DFFF）：高代理后须跟低代理；孤立代理 → `None`。
fn utf16be_to_string(bytes: &[u8]) -> Option<String> {
    if bytes.len() % 2 != 0 {
        return None; // 奇数长度：截断的 UTF-16 单元
    }
    let units: Vec<u16> =
        bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
    let mut out = String::with_capacity(units.len());
    let mut i = 0usize;
    while i < units.len() {
        let unit = units[i];
        if (0xD800..0xDC00).contains(&unit) {
            let lo = *units.get(i + 1)?;
            if !(0xDC00..0xE000).contains(&lo) {
                return None; // 高代理后无低代理
            }
            let cp = 0x1_0000u32
                + ((u32::from(unit) - 0xD800) << 10)
                + (u32::from(lo) - 0xDC00);
            out.push(char::from_u32(cp)?);
            i = i.saturating_add(2);
        } else if (0xDC00..0xE000).contains(&unit) {
            return None; // 孤立低代理
        } else {
            out.push(char::from_u32(u32::from(unit))?);
            i = i.saturating_add(1);
        }
    }
    Some(out)
}

// ===========================================================================
//  cmap 子表目录
// ===========================================================================

/// 一条 `cmap` 编码记录。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CmapSubtable {
    /// 平台 ID。
    pub platform_id: u16,
    /// 编码 ID。
    pub encoding_id: u16,
    /// 子表在 `cmap` 表内的偏移。
    pub offset: u32,
    /// 子表格式（4 = 段映射，12 = 分组映射，6 = 截断表，0 = 字节编码）。
    pub format: u16,
}

// ===========================================================================
//  包围盒
// ===========================================================================

/// 字形轮廓包围盒（字体单位，y 向上）。
///
/// 由 [`Outline`] 的**全部控制点与锚点**取极值——与 `head`/`glyf` 头里
/// 预存的 `xMin..yMax` 是两条独立路径，可交叉验证解析正确性。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// 最小 x。
    pub x_min: f32,
    /// 最小 y。
    pub y_min: f32,
    /// 最大 x。
    pub x_max: f32,
    /// 最大 y。
    pub y_max: f32,
}

impl BBox {
    /// 宽（`x_max - x_min`，恒 ≥ 0）。
    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    /// 高（`y_max - y_min`，恒 ≥ 0）。
    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }
}

/// 轮廓包围盒（空轮廓或全空环 → `None`）。
pub fn outline_bbox(outline: &Outline) -> Option<BBox> {
    let mut bb: Option<BBox> = None;
    for ring in &outline.contours {
        for p in ring {
            bb = Some(match bb {
                None => BBox { x_min: p.x, y_min: p.y, x_max: p.x, y_max: p.y },
                Some(b) => BBox {
                    x_min: b.x_min.min(p.x),
                    y_min: b.y_min.min(p.y),
                    x_max: b.x_max.max(p.x),
                    y_max: b.y_max.max(p.y),
                },
            });
        }
    }
    bb
}

// ===========================================================================
//  主入口
// ===========================================================================

/// 已解析的 OpenType 字体容器（持有字节流引用 + 表目录）。
#[derive(Debug, Clone)]
pub struct OtFont<'a> {
    data: &'a [u8],
    dir: TableDirectory,
    flavor: OutlineFlavor,
}

impl<'a> OtFont<'a> {
    /// 解析 SFNT 容器与表目录。
    ///
    /// **不**要求 `head`/`glyf` 等必需表存在——那是 [`crate::ttf::Font`] 的职责。
    /// 本函数只做容器层验证（魔数范围、目录完整性、每条记录区间在界内）。
    pub fn parse(data: &'a [u8]) -> Result<Self, FontError> {
        let dir = TableDirectory::parse(data)?;
        // 容器版本白名单：TrueType 轮廓或 CFF 轮廓。'true'/ 集合( ttcf ) /
        // wOFF 等一律结构化拒绝——它们不是本引擎的输入形态。
        if dir.sfnt_version != SFNT_TRUETYPE && dir.sfnt_version != SFNT_OTTO {
            return Err(err("T010",
                format!("sfntVersion = {:#010x}（既非 TrueType 0x00010000 也非 'OTTO'）", dir.sfnt_version),
                "本解析器只吃 SFNT 容器；ttcf 集合 / wOFF 需先拆包转换"));
        }
        let flavor = detect_flavor(&dir);
        Ok(OtFont { data, dir, flavor })
    }

    /// 底层字节流。
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }

    /// 表目录。
    pub fn directory(&self) -> &TableDirectory {
        &self.dir
    }

    /// `sfntVersion`。
    pub fn sfnt_version(&self) -> u32 {
        self.dir.sfnt_version
    }

    /// 轮廓格式。
    pub fn flavor(&self) -> OutlineFlavor {
        self.flavor
    }

    /// 取表数据切片（不存在 → 结构化错误）。
    ///
    /// 区间在 [`TableDirectory::parse`] 已验证在界内，此处 `get` 是纯防御。
    pub fn table(&self, tag: [u8; 4]) -> Result<&'a [u8], FontError> {
        let rec = self.dir.find(tag).ok_or_else(|| {
            err("T010", format!("字体缺少 {} 表", tag_str(tag)), "确认内嵌资产是完整未过度裁剪的字体")
        })?;
        let start = rec.offset as usize;
        let end = start.saturating_add(rec.length as usize);
        self.data.get(start..end).ok_or_else(|| {
            err("T011", format!("{} 表区间 {start}..{end} 越出文件", tag_str(tag)),
                "字体文件损坏或被截断")
        })
    }

    /// 取表数据切片（不存在 → `None`；用于可选表如 `GSUB`/`GPOS`）。
    pub fn optional_table(&self, tag: [u8; 4]) -> Option<&'a [u8]> {
        let rec = self.dir.find(tag)?;
        let start = rec.offset as usize;
        let end = start.saturating_add(rec.length as usize);
        self.data.get(start..end)
    }

    /// `GSUB` 表原始字节（无 → `None`）。
    ///
    /// 交给 [`crate::gsub`] 解析布局表。**注意**：本项目的内嵌资产 GSUB 是
    /// 一个 66 B 的**空壳**（`featureCount = 0`、`lookupCount = 0`，且 `grek`
    /// 脚本偏移 104 越出 66 B 表长）—— 空壳是合法输入，`gsub` 必须给出
    /// "0特征 0查找" 而非报错。
    pub fn gsub(&self) -> Option<&'a [u8]> {
        self.optional_table(TAG_GSUB)
    }

    /// `GPOS` 表原始字节（无 → `None`）。
    pub fn gpos(&self) -> Option<&'a [u8]> {
        self.optional_table(TAG_GPOS)
    }

    /// 按 OpenType 规范计算某表的校验和。
    ///
    /// 算法：表数据按 4 B 分组求大端 u32 之和（尾部不足 4 B 补零），
    /// 全程 u32 回绕（`wrapping_add`）。`head` 表的 `checkSumAdjustment`
    /// （表内偏移 8..12）在计算目录校验和时**须视为 0**（规范 §head）。
    pub fn compute_checksum(&self, tag: [u8; 4]) -> Result<u32, FontError> {
        let rec = *self.dir.find(tag).ok_or_else(|| {
            err("T010", format!("字体缺少 {} 表，无法计算校验和", tag_str(tag)), "检查字体完整性")
        })?;
        let start = rec.offset as usize;
        let end = start.saturating_add(rec.length as usize);
        let body = self.data.get(start..end).ok_or_else(|| {
            err("T011", format!("{} 表区间越出文件", tag_str(tag)), "字体文件损坏")
        })?;
        let zero_adj = tag == TAG_HEAD;
        let mut sum: u32 = 0;
        for (i, chunk) in body.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            let n = chunk.len().min(4);
            word[..n].copy_from_slice(&chunk[..n]);
            if zero_adj && i == 2 {
                // 跳过 checkSumAdjustment 所在的第 3 个 u32（字节 8..12）。
                word = [0, 0, 0, 0];
            }
            sum = sum.wrapping_add(u32::from_be_bytes(word));
        }
        Ok(sum)
    }

    /// 校验某表校验和与目录记录是否一致。
    pub fn verify_checksum(&self, tag: [u8; 4]) -> Result<bool, FontError> {
        let rec = *self.dir.find(tag).ok_or_else(|| {
            err("T010", format!("字体缺少 {} 表", tag_str(tag)), "检查字体完整性")
        })?;
        Ok(self.compute_checksum(tag)? == rec.checksum)
    }

    /// 校验全部表的校验和，返回**首个**不匹配者（`None` = 全部通过）。
    pub fn first_checksum_mismatch(&self) -> Option<[u8; 4]> {
        for rec in &self.dir.records {
            match self.verify_checksum(rec.tag) {
                Ok(true) => {}
                _ => return Some(rec.tag),
            }
        }
        None
    }

    /// 解析 `head` 表关键字段（校验 `magicNumber`）。
    pub fn head(&self) -> Result<Head, FontError> {
        let t = self.table(TAG_HEAD)?;
        let mut r = Reader::new(t);
        r.skip(12)?; // version(4) + fontRevision(4) + checkSumAdjustment(4)
        let magic = r.u32()?;
        if magic != HEAD_MAGIC {
            return Err(err("T013", format!("head.magicNumber = {magic:#010x}（应为 {HEAD_MAGIC:#010x}）"),
                "该字节流不是 TrueType 字体；确认资产来源"));
        }
        r.skip(2)?; // flags(2) ⇒ unitsPerEm 在偏移 18
        let units_per_em = r.u16()?;
        if !(16..=16384).contains(&units_per_em) {
            return Err(err("T013", format!("head.unitsPerEm = {units_per_em}（须16..=16384）"),
                "字体文件损坏或子集化时写错 head"));
        }
        r.skip(16)?; // created(8) + modified(8) ⇒ xMin 在偏移 36
        let x_min = r.i16()?;
        let y_min = r.i16()?;
        let x_max = r.i16()?;
        let y_max = r.i16()?;
        r.skip(6)?; // macStyle(2) + lowestRecPPEM(2) + fontDirectionHint(2)
        let index_to_loc_format = r.i16()?;
        if index_to_loc_format != 0 && index_to_loc_format != 1 {
            return Err(err("T013",
                format!("head.indexToLocFormat = {index_to_loc_format}（须 0 或 1）"),
                "字体文件损坏或子集化重写 head 时写错"));
        }
        Ok(Head { units_per_em, index_to_loc_format, bbox: [x_min, y_min, x_max, y_max], magic })
    }

    /// 解析 `maxp` 表关键字段。
    ///
    /// `numGlyphs` 是 **u16**（TrueType §maxp：0.5 与 1.0 版均为 u16，
    /// 其后`maxPoints` 等亦全部 u16）—— 历史上曾误用 u32 读取把它与
    /// `maxPoints` 拼成放大 6.5 万倍的假值，见 `ttf.rs` 的回归测试。
    pub fn maxp(&self) -> Result<Maxp, FontError> {
        let t = self.table(TAG_MAXP)?;
        let mut r = Reader::new(t);
        let version = r.u32()?;
        let num_glyphs = r.u16()?;
        Ok(Maxp { num_glyphs, version })
    }

    /// 解析 `hhea` 表关键字段。
    pub fn hhea(&self) -> Result<Hhea, FontError> {
        let t = self.table(TAG_HHEA)?;
        let mut r = Reader::new(t);
        r.skip(4)?; // version(4)
        let ascender = r.i16()?;
        let descender = r.i16()?;
        let line_gap = r.i16()?;
        r.skip(24)?; // advanceWidthMax..metricDataFormat 共 24 B
        let num_h_metrics = r.u16()?;
        Ok(Hhea { ascender, descender, line_gap, num_h_metrics })
    }

    /// 枚举 `cmap` 的全部编码记录（含各自格式）。
    pub fn cmap_subtables(&self) -> Result<Vec<CmapSubtable>, FontError> {
        let t = self.table(TAG_CMAP)?;
        let mut r = Reader::new(t);
        r.skip(2)?; // version(2)
        let n = r.u16()? as usize;
        // 每条记录 8 B，必须在表内。
        let need = n.checked_mul(8).and_then(|v| v.checked_add(4)).ok_or_else(|| {
            err("T014", "cmap 编码记录区长度溢出", "字体文件损坏")
        })?;
        if need > t.len() {
            return Err(err("T014",
                format!("cmap 需 {need} B 容纳 {n} 条编码记录，表仅 {} B", t.len()),
                "cmap 表被截断"));
        }
        let mut out = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let platform_id = r.u16()?;
            let encoding_id = r.u16()?;
            let offset = r.u32()?;
            let sub_off = offset as usize;
            let format = Reader::at(t, sub_off).u16().unwrap_or(0);
            out.push(CmapSubtable { platform_id, encoding_id, offset, format });
        }
        Ok(out)
    }

    /// 解析 `name` 表全部记录（字符串已解码，序即目录序）。
    pub fn name_records(&self) -> Result<Vec<NameRecord>, FontError> {
        let t = self.table(TAG_NAME)?;
        let mut r = Reader::new(t);
        let format = r.u16()?;
        if format != 0 {
            return Err(err("T015", format!("name.format = {format}（仅支持 0）"),
                "name 表版本超出支持范围"));
        }
        let count = r.u16()? as usize;
        let string_offset = r.u16()? as usize;
        let mut out = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let platform_id = r.u16()?;
            let encoding_id = r.u16()?;
            let language_id = r.u16()?;
            let name_id = r.u16()?;
            let length = r.u16()? as usize;
            let offset = r.u16()? as usize;
            let abs = string_offset.checked_add(offset).ok_or_else(|| {
                err("T015", "name 字符串偏移溢出", "name 表损坏")
            })?;
            let raw = t.get(abs..abs.saturating_add(length)).unwrap_or(&[]);
            // Windows(3) 与 Unicode(0) 平台按 UTF-16BE 解；Macintosh(1) 按 Latin-1。
            let value = match platform_id {
                0 | 3 => utf16be_to_string(raw).unwrap_or_default(),
                _ => raw.iter().map(|&b| b as char).collect(),
            };
            out.push(NameRecord { platform_id, encoding_id, language_id, name_id, value });
        }
        Ok(out)
    }

    /// 按 `nameID` 取首个非空字符串。
    pub fn name(&self, name_id: u16) -> Result<Option<String>, FontError> {
        Ok(self.name_records()?.into_iter().find(|r| r.name_id == name_id && !r.value.is_empty()).map(|r| r.value))
    }

    /// 断言轮廓格式是 `glyf`（TrueType 二次贝塞尔）。
    ///
    /// CFF / CFF2 → 结构化错误 [`FontError`]（码 `T016`）：本引擎的自研
    /// 解析器不解三次贝塞尔，**明确拒绝**而非静默错图。
    pub fn require_glyf(&self) -> Result<(), FontError> {
        match self.flavor {
            OutlineFlavor::Glyf => Ok(()),
            OutlineFlavor::Cff => Err(err("T016",
                format!(
                    "字体是 CFF/CFF2 轮廓（sfntVersion = {:#010x}，含 CFF/CFF2 表）",
                    self.dir.sfnt_version
                ),
                "本引擎只解 glyf 二次轮廓；请在构建期用 fontTools 的 otf2ttf/cu2qu 转为 TTF")),
        }
    }

    /// 构造 [`crate::ttf::Font`]（字形级读取的唯一入口）。
    ///
    /// 先 [`Self::require_glyf`] 拒绝 CFF/CFF2，再委派 `ttf::Font::parse`
    /// —— 后者已钉住 `maxp` u16 字段、`loca` 容量守卫与 `glyf` 切片钳制。
    pub fn glyphs(&self) -> Result<Font<'a>, FontError> {
        self.require_glyf()?;
        Font::parse(self.data)
    }
}

// ===========================================================================
//  最小字体字节流构造器（测试用 + 生产生成器复用）
// ===========================================================================

// 该模块**不**再 `#[cfg(test)]` 门控：`crate::multiscript_font`（生产模块）
// 复用其 `Blob` 写入器与 `assemble_sfnt` / `coverage_*` / `build_lookup` /
// `assemble_layout_table` 助手来生成项目正式字体资产，避免重复第二份 SFNT
// 组装逻辑。对外仍是 `pub(crate)`——不进入公开 API。
//
// 模块内的 `build_font` / `gpos_table` 等仅测试使用（生产走 multiscript_font
// 的同名逻辑），故显式允许 dead_code，避免把「测试专用」误报成缺陷。
#[allow(dead_code)]
pub(crate) mod synth_font {
    //! 在测试代码内**自造**最小 SFNT 字节流（零外部文件、零网络）。
    //!
    //! 内嵌的 Prism Sans SC 子集**不含阿拉伯/天城文字形**（`cmap` 在
    //! U+0600–U+06FF / U+0900–U+097F 覆盖为 **0**），其 `GSUB` 更是 66 B 的
    //! 空壳（0 特征 0 查找）。要验证 GSUB 四态替换与 lam-alef 连字的
    //! **正确性**，唯一诚实路径是在测试内构造一个**结构完整、轮廓真实**的
    //! 最小字体——本模块就是那个构造器。
    //!
    //! 产出的字形是**真实二次贝塞尔矩形**（4 个 on-curve 锚点），因此能一路
    //! 走到 [`crate::raster`] 产出**非空像素**，而不只是 gid 非 0。

    /// 大端字节写入器（带 u16 回填，用于 Offset16 补丁）。
    #[derive(Debug, Default)]
    pub struct Blob {
        b: Vec<u8>,
    }

    impl Blob {
        /// 新建空写入器。
        pub fn new() -> Self {
            Blob { b: Vec::new() }
        }

        /// 写 1 字节。
        pub fn u8(&mut self, v: u8) {
            self.b.push(v);
        }

        /// 写大端 u16。
        pub fn u16(&mut self, v: u16) {
            self.b.extend_from_slice(&v.to_be_bytes());
        }

        /// 写大端 i16。
        pub fn i16(&mut self, v: i16) {
            self.b.extend_from_slice(&v.to_be_bytes());
        }

        /// 写大端 u32。
        pub fn u32(&mut self, v: u32) {
            self.b.extend_from_slice(&v.to_be_bytes());
        }

        /// 写 4 字节 tag。
        pub fn tag(&mut self, t: &[u8; 4]) {
            self.b.extend_from_slice(t);
        }

        /// 写原始字节。
        pub fn raw(&mut self, s: &[u8]) {
            self.b.extend_from_slice(s);
        }

        /// 当前长度（用作相对偏移基准）。
        pub fn len(&self) -> usize {
            self.b.len()
        }

        /// 是否为空。
        pub fn is_empty(&self) -> bool {
            self.b.is_empty()
        }

        /// 在绝对偏移 `at` 处回填大端 u16（用于 Offset16 补丁）。
        pub fn patch_u16(&mut self, at: usize, v: u16) {
            if let Some(slot) = self.b.get_mut(at..at.saturating_add(2)) {
                slot.copy_from_slice(&v.to_be_bytes());
            }
        }

        /// 追加 4 B 对齐填充（`glyf` 表要求字形 4 B 对齐最佳实践）。
        pub fn pad4(&mut self) {
            while self.b.len() % 4 != 0 {
                self.b.push(0);
            }
        }

        /// 取字节。
        pub fn as_slice(&self) -> &[u8] {
            &self.b
        }

        /// 取字节（消耗自身）。
        pub fn into_vec(self) -> Vec<u8> {
            self.b
        }
    }

    // ---- 自造字体的字形编号（固定，测试按此断言）----
    /// `.notdef`（空字形）。
    pub const G_NOTDEF: u16 = 0;
    /// 空格（空字形）。
    pub const G_SPACE: u16 = 1;
    /// ALEF 基础形。
    pub const G_ALEF: u16 = 2;
    /// BEH 基础形（GSUB 的四态输入）。
    pub const G_BEH: u16 = 3;
    /// BEH 孤立形。
    pub const G_BEH_ISOL: u16 = 4;
    /// BEH 词尾形。
    pub const G_BEH_FINAL: u16 = 5;
    /// BEH 词首形。
    pub const G_BEH_INITIAL: u16 = 6;
    /// BEH 词中形。
    pub const G_BEH_MEDIAL: u16 = 7;
    /// LAM 基础形（连字输入之一）。
    pub const G_LAM: u16 = 8;
    /// LAM 孤立形。
    pub const G_LAM_ISOL: u16 = 9;
    /// LAM 词尾形。
    pub const G_LAM_FINAL: u16 = 10;
    /// LAM 词首形。
    pub const G_LAM_INITIAL: u16 = 11;
    /// lam-alef 连字（`liga` 产出）。
    pub const G_LAM_ALEF_LIG: u16 = 12;
    /// DEVANAGARI KA 辅音（GPOS mark 附着的基字符）。
    pub const G_KA: u16 = 13;
    /// DEVANAGARI I-MATRA 前置元音（GPOS mark）。
    pub const G_IMATRA: u16 = 14;
    /// 合成字体的字形总数。
    pub const NUM_GLYPHS: u16 = 15;

    /// 矩形字形数据（4 个 on-curve 锚点的**真实**二次轮廓）。
    ///
    /// 环序：`(x0,y0) (x1,y0) (x1,y1) (x0,y1)`；x/y 增量均为 int16。
    pub fn rect_glyph(x0: i16, y0: i16, x1: i16, y1: i16) -> Vec<u8> {
        let mut b = Blob::new();
        b.i16(1); // numberOfContours
        b.i16(x0.min(x1));
        b.i16(y0.min(y1));
        b.i16(x0.max(x1));
        b.i16(y0.max(y1));
        b.u16(3); // endPtsOfContours[0] = 3（末点下标）
        b.u16(0); // instructionLength = 0
        for _ in 0..4 {
            b.u8(0x01); // flags: ON_CURVE，无 short/same 位 ⇒ 坐标走 int16 增量
        }
        b.i16(x0); // p0.x
        b.i16(x1.saturating_sub(x0)); // p1.x
        b.i16(0); // p2.x
        b.i16(x0.saturating_sub(x1)); // p3.x
        b.i16(y0); // p0.y
        b.i16(0); // p1.y
        b.i16(y1.saturating_sub(y0)); // p2.y
        b.i16(y0.saturating_sub(y1)); // p3.y
        b.into_vec()
    }

    /// 15 个字形的 `glyf` 数据（空字形为**零长度**，loca 首尾相等）。
    ///
    /// 每个非空字形是**尺寸各异**的矩形——四态的包围盒因此两两不同，
    /// 端到端测试可以直接断言"四态像素互不相同"，而非只断言 gid 不同。
    pub fn glyph_data(index: u16) -> Vec<u8> {
        match index {
            G_NOTDEF | G_SPACE => Vec::new(),
            G_ALEF => rect_glyph(140, 0, 260, 760),
            G_BEH => rect_glyph(100, 0, 400, 700),
            G_BEH_ISOL => rect_glyph(120, 0, 380, 700),
            G_BEH_FINAL => rect_glyph(120, 0, 430, 480),
            G_BEH_INITIAL => rect_glyph(70, 0, 430, 480),
            G_BEH_MEDIAL => rect_glyph(70, 0, 380, 700),
            G_LAM => rect_glyph(100, 0, 400, 750),
            G_LAM_ISOL => rect_glyph(120, 0, 380, 750),
            G_LAM_FINAL => rect_glyph(120, 0, 440, 500),
            G_LAM_INITIAL => rect_glyph(60, 0, 440, 500),
            G_LAM_ALEF_LIG => rect_glyph(60, 0, 900, 760),
            G_KA => rect_glyph(90, 0, 510, 620),
            G_IMATRA => rect_glyph(230, 640, 350, 760),
            _ => Vec::new(),
        }
    }

    /// 每字形的步进宽（字体单位）。
    pub fn glyph_advance(index: u16) -> u16 {
        match index {
            G_NOTDEF => 500,
            G_SPACE => 250,
            G_ALEF => 260,
            G_BEH => 500,
            G_BEH_ISOL => 500,
            G_BEH_FINAL | G_BEH_INITIAL => 500,
            G_BEH_MEDIAL => 450,
            G_LAM => 500,
            G_LAM_ISOL => 500,
            G_LAM_FINAL | G_LAM_INITIAL => 520,
            G_LAM_ALEF_LIG => 900,
            G_KA => 600,
            G_IMATRA => 0, // 零宽记号
            _ => 0,
        }
    }

    /// 每字形的左bearing。
    pub fn glyph_lsb(index: u16) -> i16 {
        match index {
            G_ALEF => 140,
            G_BEH => 100,
            G_BEH_ISOL => 120,
            G_BEH_FINAL | G_BEH_INITIAL => 120,
            G_BEH_MEDIAL => 70,
            G_LAM => 100,
            G_LAM_ISOL => 120,
            G_LAM_FINAL | G_LAM_INITIAL => 120,
            G_LAM_ALEF_LIG => 60,
            G_KA => 90,
            G_IMATRA => 230,
            _ => 0,
        }
    }

    /// 要写入 `cmap` 的码位 → 字形映射（按码位升序，format 4/12 共用）。
    pub fn charmap_pairs() -> Vec<(u32, u16)> {
        vec![
            (0x0020, G_SPACE),             // U+0020 SPACE
            (0x0627, G_ALEF),              // U+0627 ALEF
            (0x0628, G_BEH),               // U+0628 BEH
            (0x0644, G_LAM),               // U+0644 LAM
            (0x0915, G_KA),                // U+0915 DEVANAGARI KA
            (0x093F, G_IMATRA),            // U+093F DEVANAGARI I-MATRA
            (0xFEFB, G_LAM_ALEF_LIG),      // U+FEFB lam-alef 孤立呈现形
            (0xFEFC, G_LAM_ALEF_LIG),      // U+FEFC lam-alef 词尾呈现形
        ]
    }

    /// 构造 `cmap` format 4 子表（段映射，仅用 `idDelta`）。
    pub fn cmap_format4() -> Vec<u8> {
        let mut segs: Vec<(u32, u32, u16)> =
            charmap_pairs().iter().map(|&(cp, g)| (cp, cp, g)).collect();
        segs.push((0xFFFF, 0xFFFF, 0)); // 强制终止段（规范要求）
        let seg_count = segs.len() as u16;
        let seg_x2 = seg_count.saturating_mul(2);
        // searchRange = 2 * 2^floor(log2(segCount))；entrySelector = log2(searchRange/2)。
        let mut entry_selector = 0u16;
        while (1u32 << (entry_selector.saturating_add(1))) <= u32::from(seg_count) {
            entry_selector = entry_selector.saturating_add(1);
        }
        let search_range = 2u16.saturating_mul(1u16 << entry_selector);
        let range_shift = seg_x2.saturating_sub(search_range);

        let mut b = Blob::new();
        b.u16(4); // format
        b.u16(0); // length（回填）
        b.u16(0); // language
        b.u16(seg_x2);
        b.u16(search_range);
        b.u16(entry_selector);
        b.u16(range_shift);
        for &(_s, e, _) in &segs {
            b.u16(e as u16);
        }
        b.u16(0); // reservedPad
        for &(s, _, _) in &segs {
            b.u16(s as u16);
        }
        for &(s, _, g) in &segs {
            b.u16((g as u16).wrapping_sub(s as u16)); // idDelta
        }
        for _ in &segs {
            b.u16(0); // idRangeOffset：全0 ⇒ 走 idDelta
        }
        let len = b.len() as u16;
        b.patch_u16(2, len);
        b.into_vec()
    }

    /// 构造 `cmap` format 12 子表（分组映射，覆盖全部 BMP + 一个增补平面码位）。
    pub fn cmap_format12() -> Vec<u8> {
        let mut groups: Vec<(u32, u32, u32)> =
            charmap_pairs().iter().map(|&(cp, g)| (cp, cp, u32::from(g))).collect();
        // 增补平面码位：验证 format 12 的 32 位码位路径（BMP 之外）。
        groups.push((0x1_0A00, 0x1_0A00, u32::from(G_ALEF)));
        groups.sort_by_key(|g| g.0);

        // format 12 头：format(2) + reserved(2) + length(4) + language(4) + numGroups(4)。
        // 注意 `length` 是 **u32**（format 4 才是 u16）——写错会让 numGroups 读成垃圾。
        let sub_len: u32 = 16u32.saturating_add(groups.len().saturating_mul(12) as u32);
        let mut b = Blob::new();
        b.u16(12); // format
        b.u16(0); // reserved
        b.u32(sub_len); // length
        b.u32(0); // language
        b.u32(groups.len() as u32);
        for &(s, e, g) in &groups {
            b.u32(s);
            b.u32(e);
            b.u32(g);
        }
        b.into_vec()
    }

    /// 构造 `cmap` 表（含 format 4 与 format 12 两条编码记录）。
    pub fn cmap_table() -> Vec<u8> {
        let sub4 = cmap_format4();
        let sub12 = cmap_format12();
        let mut b = Blob::new();
        b.u16(0); // version
        b.u16(2); // numTables
        // 记录序即目录序（确定性）：先 (3,1) format 4，再 (3,10) format 12。
        b.u16(3);
        b.u16(1);
        b.u32(4 + 2 * 8); // sub4 偏移
        b.u16(3);
        b.u16(10);
        b.u32((4 + 2 * 8 + sub4.len()) as u32); // sub12 偏移
        b.raw(&sub4);
        b.raw(&sub12);
        b.into_vec()
    }

    /// 构造 `GSUB` 表：4 态 SingleSubst（isol/init/medi/fina）+ lam-alef 连字。
    ///
    /// 结构（全部偏移相对各自子结构起点，符合 OpenType 规范）：
    /// ```text
    /// GSUBHeader: version(1.0) scriptList featureList lookupList
    /// ScriptList 'arab': DefaultLangSys → featureIndices[0..5]
    /// FeatureList（序 = 施加序）: isol init medi fina liga
    /// LookupList: 5 个查找
    ///   0 SingleSubst fmt2  beh_base → beh_isol
    ///   1 SingleSubst fmt1  beh_base → beh_initial   （delta 路径）
    ///   2 SingleSubst fmt2  beh_base → beh_medial
    ///   3 SingleSubst fmt2  beh_base → beh_final
    ///   4 LigatureSubst    lam_base + alef_base → lam_alef_lig
    /// ```
    pub fn gsub_table() -> Vec<u8> {
        // ---- Lookup 0..3：SingleSubst ----
        // fmt2 = 显式 substitute 数组；fmt1 = 单一 delta（覆盖另一条代码路径）。
        let mk_single_fmt1 = |gid: u16, delta: i16| -> Vec<u8> {
            let cov = coverage_fmt1(&[gid]);
            let mut b = Blob::new();
            b.u16(1); // substFormat = 1
            b.u16(0); // coverageOffset（回填）
            b.i16(delta);
            let cov_off = b.len() as u16;
            b.patch_u16(2, cov_off);
            b.raw(&cov);
            b.into_vec()
        };
        let mk_single_fmt2 = |gid: u16, sub: u16| -> Vec<u8> {
            let cov = coverage_fmt1(&[gid]);
            let mut b = Blob::new();
            b.u16(2); // substFormat = 2
            b.u16(0); // coverageOffset（回填）
            b.u16(1); // glyphCount
            b.u16(sub);
            let cov_off = b.len() as u16;
            b.patch_u16(2, cov_off);
            b.raw(&cov);
            b.into_vec()
        };
        // lookup 1 用 fmt1（delta 路径：BEH(3) + 3 = BEH_INITIAL(6)），
        // 其余用 fmt2（显式 substitute 数组）—— 两条代码路径都被覆盖。

        // ---- Lookup 4：LigatureSubst（lam + alef → 连字）----
        // Coverage format 1 规范要求**字形 ID 升序**：ALEF(2) < LAM(8)。
        // ⇒ 覆盖索引 0 = ALEF，索引 1 = LAM。
        let lig_cov = coverage_fmt1(&[G_ALEF, G_LAM]);

        // LigatureSet[1]（LAM，索引 1）：1 条连字，组件 = [隐含的 LAM, ALEF]。
        let mut ligature = Blob::new();
        ligature.u16(G_LAM_ALEF_LIG); // ligatureGlyph
        ligature.u16(2); // componentCount（含首组件）
        ligature.u16(G_ALEF); // componentGlyphIDs[0]（首组件隐含为 coverage 命中的 LAM）
        let ligature_bytes = ligature.into_vec();

        // LigatureSet[0]（ALEF，索引 0）：无连字。
        let mut set0 = Blob::new();
        set0.u16(0); // ligatureCount = 0
        let set0_bytes = set0.into_vec();

        // LigatureSet[1]（LAM，索引 1）：1 条连字。
        let mut set1 = Blob::new();
        set1.u16(1); // ligatureCount
        set1.u16(0); // ligatureOffsets[0]（回填；相对 LigatureSet 起点）
        let lig_off_in_set = set1.len() as u16; // = 4
        set1.patch_u16(2, lig_off_in_set);
        set1.raw(&ligature_bytes);
        let set1_bytes = set1.into_vec();

        let mut lig_sub = Blob::new();
        lig_sub.u16(1); // substFormat = 1
        lig_sub.u16(0); // coverageOffset（回填）
        lig_sub.u16(2); // ligatureSetCount
        lig_sub.u16(0); // ligatureSetOffsets[0]（回填）
        lig_sub.u16(0); // ligatureSetOffsets[1]（回填）
        let set0_off = lig_sub.len() as u16;
        let set1_off = set0_off.saturating_add(set0_bytes.len() as u16);
        lig_sub.patch_u16(6, set0_off);
        lig_sub.patch_u16(8, set1_off);
        lig_sub.raw(&set0_bytes);
        lig_sub.raw(&set1_bytes);
        let cov_off = lig_sub.len() as u16;
        lig_sub.patch_u16(2, cov_off);
        lig_sub.raw(&lig_cov);
        let lig_sub = lig_sub.into_vec();

        // ---- 组装 LookupList（4 个 SingleSubst + 1 个 Ligature）----
        let subtables: Vec<(u16, Vec<u8>)> = vec![
            (1, mk_single_fmt2(G_BEH, G_BEH_ISOL)),    // 0: isol（数组路径）
            (1, mk_single_fmt1(G_BEH, 3)),            // 1: init（delta 路径：BEH(3)+3=INITIAL(6)）
            (1, mk_single_fmt2(G_BEH, G_BEH_MEDIAL)),  // 2: medi
            (1, mk_single_fmt2(G_BEH, G_BEH_FINAL)),   // 3: fina
            (4, lig_sub),                             // 4: liga（连字）
        ];
        let lookups: Vec<Vec<u8>> =
            subtables.iter().map(|(kind, st)| build_lookup(*kind, st)).collect();

        assemble_layout_table(b"GSUB", &lookups, &feature_tags(), &LANG_FEATURE_INDICES)
    }

    /// FeatureList 的 tag 序 = **施加序**（确定性纪律的核心）。
    fn feature_tags() -> Vec<[u8; 4]> {
        vec![*b"isol", *b"init", *b"medi", *b"fina", *b"liga"]
    }

    /// DefaultLangSys 引用的 feature 索引（与 [`feature_tags`] 同序）。
    const LANG_FEATURE_INDICES: [u16; 5] = [0, 1, 2, 3, 4];

    /// Coverage format 1（显式字形数组）。
    pub fn coverage_fmt1(glyphs: &[u16]) -> Vec<u8> {
        let mut b = Blob::new();
        b.u16(1); // coverageFormat = 1
        b.u16(glyphs.len() as u16);
        for &g in glyphs {
            b.u16(g);
        }
        b.into_vec()
    }

    /// Coverage format 2（区间 + 起始覆盖索引）。
    pub fn coverage_fmt2(ranges: &[(u16, u16, u16)]) -> Vec<u8> {
        let mut b = Blob::new();
        b.u16(2); // coverageFormat = 2
        b.u16(ranges.len() as u16);
        for &(s, e, i) in ranges {
            b.u16(s);
            b.u16(e);
            b.u16(i);
        }
        b.into_vec()
    }

    /// 构造单个 Lookup（`lookupType` + `lookupFlag=0` + 1 个子表）。
    pub fn build_lookup(kind: u16, subtable: &[u8]) -> Vec<u8> {
        let mut lk = Blob::new();
        lk.u16(kind);
        lk.u16(0); // lookupFlag = 0（不过滤记号）
        lk.u16(1); // subTableCount
        lk.u16(0); // subtableOffsets[0]（回填）
        // 头长度 = 2+2+2+2 = 8
        lk.patch_u16(6, 8);
        lk.raw(subtable);
        lk.into_vec()
    }

    /// 组装 GSUB/GPOS 布局表（ScriptList/FeatureList/LookupList 三段）。
    ///
    /// `feature_tags` 的**下标即施加序**；`lang_feature_indices` 引用它们。
    pub fn assemble_layout_table(
        tag: &[u8; 4],
        lookups: &[Vec<u8>],
        feature_tags: &[[u8; 4]],
        lang_feature_indices: &[u16],
    ) -> Vec<u8> {
        // ---- LookupList ----
        // 规范：LookupList 里的每个偏移相对 **LookupList 表起点**（本Blob 的0），
        // 不是相对第一项偏移所在的位置。
        let mut lookup_list = Blob::new();
        lookup_list.u16(lookups.len() as u16);
        let lookup_index_base = lookup_list.len();
        for _ in lookups {
            lookup_list.u16(0); // 占位
        }
        for (i, lk) in lookups.iter().enumerate() {
            let off = lookup_list.len() as u16;
            lookup_list.patch_u16(lookup_index_base + i * 2, off);
            lookup_list.raw(lk);
        }

        // ---- FeatureList ----
        let mut feature_list = Blob::new();
        feature_list.u16(feature_tags.len() as u16);
        let feature_index_base = feature_list.len();
        for ft in feature_tags {
            feature_list.tag(ft);
            feature_list.u16(0); // 占位
        }
        for (i, ft) in feature_tags.iter().enumerate() {
            let _ = ft;
            // 特征 i 引用查找 i（合成字体里一特征一查找，序即施加序）。
            let mut f = Blob::new();
            f.u16(0); // featureParams = NULL
            f.u16(1); // lookupIndexCount
            f.u16(i as u16); // lookupListIndices[0] = i
            let off = feature_list.len() as u16;
            feature_list.patch_u16(feature_index_base + i * 6 + 4, off);
            feature_list.raw(f.as_slice());
        }

        // ---- ScriptList（单脚本 'arab'，仅 DefaultLangSys）----
        let mut lang_sys = Blob::new();
        lang_sys.u16(0); // lookupOrderOffset = NULL
        lang_sys.u16(0xFFFF); // requiredFeatureIndex = 无
        lang_sys.u16(lang_feature_indices.len() as u16);
        for &fi in lang_feature_indices {
            lang_sys.u16(fi);
        }
        let mut script = Blob::new();
        script.u16(0); // defaultLangSysOffset（回填）
        script.u16(0); // langSysCount = 0
        let lang_off = script.len() as u16;
        script.patch_u16(0, lang_off);
        script.raw(lang_sys.as_slice());

        let mut script_list = Blob::new();
        script_list.u16(1); // scriptCount
        script_list.tag(b"arab");
        script_list.u16(0); // scriptOffset（回填）
        let script_off = script_list.len() as u16;
        script_list.patch_u16(6, script_off);
        script_list.raw(script.as_slice());

        // ---- 头部 ----
        // 头部三段偏移都是**回填**到头部字段（偏移 4/6/8），不是追加到流里。
        let mut out = Blob::new();
        out.u16(1); // majorVersion
        out.u16(0); // minorVersion
        out.u16(0); // scriptListOffset（回填）
        out.u16(0); // featureListOffset（回填）
        out.u16(0); // lookupListOffset（回填）
        out.patch_u16(4, 10); // scriptListOffset = 10（头部固定长）
        out.raw(script_list.as_slice());
        out.patch_u16(6, out.len() as u16); // featureList 紧随 ScriptList
        out.raw(feature_list.as_slice());
        out.patch_u16(8, out.len() as u16); // lookupList 紧随 FeatureList
        out.raw(lookup_list.as_slice());
        let _ = tag;
        out.into_vec()
    }

    /// 构造 `GPOS` 表：`abvm` 特征 → MarkToBasePos（i-matra 附着到 KA）。
    pub fn gpos_table() -> Vec<u8> {
        // ---- MarkArray：1 条记录（i-matra，class 0），锚点 (0, 640) ----
        let mut mark_anchor = Blob::new();
        mark_anchor.u16(1); // anchorFormat = 1
        mark_anchor.i16(0); // xCoordinate
        mark_anchor.i16(640); // yCoordinate
        let mark_anchor_bytes = mark_anchor.into_vec();

        // MarkArray：markCount(2) + MarkRecord{markClass(2), markAnchorOffset(2)} = 6 B 头。
        // markAnchorOffset 相对 **MarkArray 起点**，故 = 6。
        let mut mark_array = Blob::new();
        mark_array.u16(1); // markCount
        mark_array.u16(0); // markClass
        mark_array.u16(6); // markAnchorOffset（相对 MarkArray 起点）
        let mark_array_bytes = {
            let mut v = mark_array.into_vec();
            v.extend_from_slice(&mark_anchor_bytes);
            v
        };

        // ---- BaseArray：1 个基字符（KA），class 0 锚点 (230, 0) ----
        let mut base_anchor = Blob::new();
        base_anchor.u16(1); // anchorFormat = 1
        base_anchor.i16(230); // xCoordinate
        base_anchor.i16(0); // yCoordinate
        let base_anchor_bytes = base_anchor.into_vec();

        // BaseArray：baseCount(2) + baseAnchorOffsets[1][1] = 4 B 头。
        // baseAnchorOffset 相对 **BaseArray 起点**，故 = 4。
        let mut base_array = Blob::new();
        base_array.u16(1); // baseCount
        base_array.u16(4); // baseAnchorOffsets[0][0]
        let base_array_bytes = {
            let mut v = base_array.into_vec();
            v.extend_from_slice(&base_anchor_bytes);
            v
        };

        // ---- MarkToBasePos format 1 ----
        // 全部偏移相对 **本子表起点**（规范 §GPOS MarkBasePos）。
        let mark_cov = coverage_fmt1(&[G_IMATRA]);
        let base_cov = coverage_fmt1(&[G_KA]);
        let mut sub = Blob::new();
        sub.u16(1); // posFormat = 1
        sub.u16(0); // markCoverageOffset（回填）
        sub.u16(0); // baseCoverageOffset（回填）
        sub.u16(1); // markClassCount
        sub.u16(0); // markArrayOffset（回填）
        sub.u16(0); // baseArrayOffset（回填）
        let mark_cov_off = sub.len() as u16;
        sub.patch_u16(2, mark_cov_off);
        let base_cov_off = mark_cov_off.saturating_add(mark_cov.len() as u16);
        sub.patch_u16(4, base_cov_off);
        let mark_arr_off = base_cov_off.saturating_add(base_cov.len() as u16);
        sub.patch_u16(8, mark_arr_off);
        let base_arr_off = mark_arr_off.saturating_add(mark_array_bytes.len() as u16);
        sub.patch_u16(10, base_arr_off);
        sub.raw(&mark_cov);
        sub.raw(&base_cov);
        sub.raw(&mark_array_bytes);
        sub.raw(&base_array_bytes);

        let lookup = build_lookup(4, sub.as_slice()); // GPOS lookupType 4 = MarkToBasePos
        assemble_layout_table(b"GPOS", &[lookup], &[*b"abvm"], &[0])
    }

    /// 组装完整 SFNT 文件（表目录按 tag 升序，符合规范）。
    ///
    /// `cmap_formats`：要写入哪些子表格式（`4` / `12` / 两者）。分开构造是
    /// 为了让format 4 与 format 12 的查找路径各自被独立覆盖。
    pub fn build_font(cmap_formats: &[u16]) -> Vec<u8> {
        let mut tables: Vec<([u8; 4], Vec<u8>)> = Vec::new();

        // glyf + loca
        let mut glyf = Blob::new();
        let mut loca = Blob::new();
        for gid in 0..NUM_GLYPHS {
            loca.u32(glyf.len() as u32);
            let mut g = glyph_data(gid);
            glyf.raw(&g);
            while glyf.len() % 4 != 0 {
                glyf.u8(0); // 4 B 对齐
            }
            g.clear();
        }
        loca.u32(glyf.len() as u32); // 哨兵项
        tables.push((*b"glyf", glyf.into_vec()));
        tables.push((*b"loca", loca.into_vec()));

        // head
        let mut head = Blob::new();
        head.u32(0x0001_0000); // version 1.0
        head.u32(0x0001_0000); // fontRevision 1.0
        head.u32(0); // checkSumAdjustment（构建期不回填）
        head.u32(crate::opentype::HEAD_MAGIC);
        head.u16(0); // flags
        head.u16(1000); // unitsPerEm
        head.u32(0); // created(8B：两u32)
        head.u32(0);
        head.u32(0); // modified(8B)
        head.u32(0);
        head.i16(0); // xMin
        head.i16(-200); // yMin
        head.i16(1000); // xMax
        head.i16(800); // yMax
        head.u16(0); // macStyle
        head.u16(8); // lowestRecPPEM
        head.i16(2); // fontDirectionHint
        head.i16(1); // indexToLocFormat = 1（长 loca）
        head.i16(0); // glyphDataFormat
        tables.push((*b"head", head.into_vec()));

        // maxp
        let mut maxp = Blob::new();
        maxp.u32(0x0001_0000); // version 1.0
        maxp.u16(NUM_GLYPHS); // numGlyphs（u16！）
        maxp.u16(4); // maxPoints
        maxp.u16(1); // maxContours
        maxp.u16(0); // maxCompositePoints
        maxp.u16(0); // maxCompositeContours
        maxp.u16(2); // maxZones
        maxp.u16(0); // maxTwilightPoints
        maxp.u16(0); // maxStorage
        maxp.u16(0); // maxFunctionDefs
        maxp.u16(0); // maxInstructionDefs
        maxp.u16(0); // maxStackElements
        maxp.u16(0); // maxSizeOfInstructions
        maxp.u16(0); // maxComponentElements
        maxp.u16(0); // maxComponentDepth
        tables.push((*b"maxp", maxp.into_vec()));

        // hhea
        let mut hhea = Blob::new();
        hhea.u32(0x0001_0000);
        hhea.i16(800); // ascender
        hhea.i16(-200); // descender
        hhea.i16(0); // lineGap
        hhea.u16(1000); // advanceWidthMax
        hhea.i16(0); // minLeftSideBearing
        hhea.i16(0); // minRightSideBearing
        hhea.i16(1000); // xMaxExtent
        hhea.i16(1); // caretSlopeRise
        hhea.i16(0); // caretSlopeRun
        hhea.i16(0); // caretOffset
        hhea.i16(0); // reserved × 4
        hhea.i16(0);
        hhea.i16(0);
        hhea.i16(0);
        hhea.i16(0); // metricDataFormat
        hhea.u16(NUM_GLYPHS); // numberOfHMetrics
        tables.push((*b"hhea", hhea.into_vec()));

        // hmtx
        let mut hmtx = Blob::new();
        for gid in 0..NUM_GLYPHS {
            hmtx.u16(glyph_advance(gid));
            hmtx.i16(glyph_lsb(gid));
        }
        tables.push((*b"hmtx", hmtx.into_vec()));

        // cmap
        let cmap = if cmap_formats.contains(&12) && cmap_formats.contains(&4) {
            cmap_table()
        } else if cmap_formats.contains(&12) {
            wrap_single_cmap(&cmap_format12(), 3, 10)
        } else {
            wrap_single_cmap(&cmap_format4(), 3, 1)
        };
        tables.push((*b"cmap", cmap));

        // GSUB + GPOS
        tables.push((*b"GSUB", gsub_table()));
        tables.push((*b"GPOS", gpos_table()));

        // name
        tables.push((*b"name", name_table()));

        // 目录按 tag 升序（规范要求；也便于人工检视）。
        tables.sort_by(|a, b| a.0.cmp(&b.0));
        assemble_sfnt(0x0001_0000, &tables)
    }

    /// 只含一条编码记录的 `cmap` 表。
    fn wrap_single_cmap(sub: &[u8], platform: u16, encoding: u16) -> Vec<u8> {
        let mut b = Blob::new();
        b.u16(0);
        b.u16(1); // numTables
        b.u16(platform);
        b.u16(encoding);
        b.u32(4 + 8);
        b.raw(sub);
        b.into_vec()
    }

    /// 最小 `name` 表（family=1 / subfamily=2 / full=4 / ps=6）。
    fn name_table() -> Vec<u8> {
        let entries: [(u16, &str); 4] = [
            (1, "Prism Test Arabic"),
            (2, "Regular"),
            (4, "Prism Test Arabic Regular"),
            (6, "PrismTestArabic-Regular"),
        ];
        let mut b = Blob::new();
        b.u16(0); // format
        b.u16(entries.len() as u16);
        b.u16((6 + entries.len() * 12) as u16); // stringOffset
        let mut strings = Blob::new();
        for (id, text) in entries {
            let off = strings.len() as u16;
            for unit in text.encode_utf16() {
                strings.u16(unit);
            }
            b.u16(3); // platformID = Windows
            b.u16(1); // encodingID = UCS-2
            b.u16(0x0409); // languageID = en-US
            b.u16(id);
            b.u16(text.len() as u16 * 2);
            b.u16(off);
        }
        b.raw(strings.as_slice());
        b.into_vec()
    }

    /// 组装 SFNT 容器（offset 4 B 对齐，逐表填校验和）。
    pub fn assemble_sfnt(sfnt_version: u32, tables: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let num_tables = tables.len();
        let header_len = 12 + num_tables * 16;
        let mut offset = header_len;
        let mut placed: Vec<(u32, u32)> = Vec::with_capacity(num_tables); // (offset, length)
        for (_, body) in tables {
            placed.push((offset as u32, body.len() as u32));
            offset += body.len();
            // 表数据 4 B 对齐（规范建议；loca/glyf 已内部对齐）。
            while offset % 4 != 0 {
                offset += 1;
            }
        }

        let mut out = Blob::new();
        out.u32(sfnt_version);
        out.u16(num_tables as u16);
        // searchRange = 16 * 2^floor(log2(numTables))，entrySelector = log2(除数)
        let mut entry_selector = 0u32;
        while (1u32 << (entry_selector + 1)) <= num_tables as u32 {
            entry_selector += 1;
        }
        let search_range = 16u32.saturating_mul(1u32 << entry_selector);
        out.u16(search_range as u16);
        out.u16(entry_selector as u16);
        out.u16((num_tables.saturating_mul(16).saturating_sub(search_range as usize)) as u16);

        for (i, (tag, body)) in tables.iter().enumerate() {
            let (off, len) = placed.get(i).copied().unwrap_or((0, 0));
            out.tag(tag);
            out.u32(table_checksum(body, *tag == *b"head"));
            out.u32(off);
            out.u32(len);
        }
        for (i, (_, body)) in tables.iter().enumerate() {
            let _ = i;
            out.raw(body);
            while out.len() % 4 != 0 {
                out.u8(0);
            }
        }
        out.into_vec()
    }

    /// 表校验和（4 B 分组求和；`head` 的 `checkSumAdjustment` 视为 0）。
    pub fn table_checksum(body: &[u8], zero_adj: bool) -> u32 {
        let mut sum: u32 = 0;
        for (i, chunk) in body.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            let n = chunk.len().min(4);
            word[..n].copy_from_slice(&chunk[..n]);
            if zero_adj && i == 2 {
                word = [0, 0, 0, 0];
            }
            sum = sum.wrapping_add(u32::from_be_bytes(word));
        }
        sum
    }
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{font_regular, FONT_REGULAR};
    use synth_font::{self as sf, Blob};

    // ---- O1：内嵌字体的表目录 ----

    /// O1：内嵌 Prism Sans SC 的表目录 = 12 张表，tag 集合正确。
    #[test]
    fn o1_embedded_directory_lists_expected_tables() {
        let f = font_regular();
        let data: &[u8] = FONT_REGULAR;
        let ot = OtFont::parse(data).expect("内嵌字体应可解析容器");
        assert_eq!(ot.sfnt_version(), SFNT_TRUETYPE, "内嵌资产是 TrueType 容器");
        let tags = ot.directory().tags();
        assert_eq!(tags.len(), 12, "内嵌子集应有 12 张表，实际 {}", tags.len());
        for expect in [
            TAG_HEAD, TAG_MAXP, TAG_HHEA, TAG_HMTX, TAG_CMAP, TAG_LOCA, TAG_GLYF, TAG_NAME,
            TAG_GSUB, TAG_GPOS,
        ] {
            assert!(ot.directory().contains(expect), "目录应含 {}", String::from_utf8_lossy(&expect));
        }
        // 与 ttf.rs 的独立解析交叉验证：unitsPerEm / numGlyphs 必须一致。
        assert_eq!(ot.head().expect("head").units_per_em, f.units_per_em());
        assert_eq!(u32::from(ot.maxp().expect("maxp").num_glyphs), f.num_glyphs());
    }

    /// O2：逐表校验和全部通过（含 `head` 的 `checkSumAdjustment` 归零特例）。
    #[test]
    fn o2_all_table_checksums_verify() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        assert_eq!(ot.first_checksum_mismatch(), None, "全部 12 张表校验和应一致");
        // 逐表再确认一遍（给出定位信息）。
        for rec in &ot.directory().records {
            assert!(ot.verify_checksum(rec.tag).expect("计算校验和"), "表 {} 校验和应匹配", rec.tag_str());
        }
    }

    /// O3：`head` 字段（unitsPerEm / indexToLocFormat / magic / bbox）。
    #[test]
    fn o3_head_fields_match_known_values() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let h = ot.head().expect("head 表");
        assert_eq!(h.units_per_em, 1000);
        assert_eq!(h.index_to_loc_format, 1, "内嵌资产用长 loca");
        assert_eq!(h.magic, HEAD_MAGIC);
        assert_eq!(h.bbox, [-44, -279, 994, 880], "内嵌子集的全局包围盒");
    }

    /// O4：`maxp.numGlyphs` 是 u16 字段（回归：不得与 `maxPoints` 拼接）。
    #[test]
    fn o4_maxp_num_glyphs_is_u16() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let m = ot.maxp().expect("maxp 表");
        assert_eq!(m.version, 0x0001_0000);
        assert_eq!(m.num_glyphs, 6889, "numGlyphs 是 u16，不得吃进 maxPoints");
        let inflated = 6889u32 | (247u32 << 16);
        assert_ne!(inflated, u32::from(m.num_glyphs));
    }

    /// O5：`hhea` 度量（内嵌子集被裁剪过行高，与上游 Noto 不同）。
    #[test]
    fn o5_hhea_metrics() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let h = ot.hhea().expect("hhea 表");
        assert_eq!((h.ascender, h.descender, h.line_gap), (1160, -288, 0));
        assert_eq!(h.num_h_metrics, 99, "hmtx 只有 99 条长度量");
        // 与 ttf.rs 独立读出的值一致。
        let f = font_regular();
        assert_eq!(h.ascender, f.ascender());
        assert_eq!(h.descender, f.descender());
    }

    /// O6：`cmap` 目录含 format 4（内嵌资产无 format 12——诚实记录）。
    #[test]
    fn o6_embedded_cmap_has_format4() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let subs = ot.cmap_subtables().expect("cmap 目录");
        assert_eq!(subs.len(), 2, "(0,3) 与 (3,1) 两条记录");
        assert!(subs.iter().all(|s| s.format == 4), "内嵌资产只有 format 4");
        // 交叉验证：两条记录同指 format 4 子表。
        assert_eq!(subs[0].offset, subs[1].offset);
    }

    /// O7：`hmtx` advance 与 `loca`/`glyf` 轮廓包围盒（**真实轮廓**断言）。
    #[test]
    fn o7_hmtx_advance_and_glyf_bbox() {
        let f = font_regular();
        // 已知字形（由构建期子集产物交叉验证）：U+4E00 '一' gid=114。
        let gid = f.glyph_index('一').expect("一 在子集内");
        assert_eq!(gid, 114);
        assert_eq!(f.advance(gid), 1000, "'一' 是全角，advance = 1000 = em");

        // 由**轮廓点**独立算出的包围盒，必须等于 glyf 头预存的 xMin..yMax。
        let outline = f.outline(gid).expect("轮廓可解析");
        let bb = outline_bbox(&outline).expect("'一' 有轮廓");
        assert_eq!((bb.x_min, bb.y_min, bb.x_max, bb.y_max), (44.0, 349.0, 960.0, 431.0),
            "'一' 是单轮廓横画，包围盒应与构建期统计一致");
        assert!(bb.width() > 900.0 && bb.height() < 100.0, "横画：宽而扁");

        // 再验一个多轮廓字母（A：2 轮廓）。
        let ga = f.glyph_index('A').expect("A 在子集内");
        let bb_a = outline_bbox(&f.outline(ga).expect("A 轮廓")).expect("A 有轮廓");
        assert_eq!((bb_a.x_min, bb_a.y_min, bb_a.x_max, bb_a.y_max), (4.0, 0.0, 604.0, 733.0));
    }

    /// O8：`name` 表解析（UTF-16BE 解码 + OFL 改名链路）。
    #[test]
    fn o8_name_table_decodes_family_and_license() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let recs = ot.name_records().expect("name 表");
        assert_eq!(recs.len(), 7, "内嵌资产记录 7 条 name");
        assert_eq!(ot.name(1).expect("name 表").as_deref(), Some("Prism Sans SC"),
            "family 名已按 OFL 的 RFN 条款改名");
        assert_eq!(ot.name(2).expect("name 表").as_deref(), Some("Regular"));
        assert_eq!(ot.name(6).expect("name 表").as_deref(), Some("PrismSansSC-Regular"));
        // nameID 14 = 许可 URL（OFL 合规链的一环）。
        let url = ot.name(14).expect("name 表").unwrap_or_default();
        assert!(url.starts_with("http"), "nameID 14 应是许可 URL，实际 {url:?}");
    }

    /// O9：轮廓格式判定 = glyf（内嵌资产），且 `require_glyf` 放行。
    #[test]
    fn o9_embedded_flavor_is_glyf() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        assert_eq!(ot.flavor(), OutlineFlavor::Glyf);
        ot.require_glyf().expect("glyf 轮廓应放行");
        assert!(ot.glyphs().is_ok(), "应能委派给 ttf::Font");
    }

    // ---- O10..O14：畸形字体攻击（全部结构化错误，零 panic）----

    /// O10：截断到 sfnt 头之前 / 目录不完整 ⇒ T010。
    #[test]
    fn o10_truncated_font_rejected() {
        for n in [0usize, 4, 11] {
            let e = OtFont::parse(&FONT_REGULAR[..n.min(FONT_REGULAR.len())])
                .expect_err("截断字体应被拒");
            assert_eq!(e.code, "T010", "长度 {n} 应得 T010，实际 {}", e.code);
            assert!(!e.fix.is_empty(), "错误须带修复建议");
        }
    }

    /// O11：错误 sfnt 版本（ttcf 集合 / wOFF）⇒ T010。
    #[test]
    fn o11_bad_sfnt_version_rejected() {
        let mut d = FONT_REGULAR.to_vec();
        d[0..4].copy_from_slice(b"wOFF"); // WOFF 压缩包装，不是裸 SFNT
        let e = OtFont::parse(&d).expect_err("wOFF 应被拒");
        assert_eq!(e.code, "T010");
        assert!(e.problem.contains("wOFF") || e.problem.contains("sfntVersion"));
    }

    /// O12：表记录 offset/length 越界 ⇒ T010（不 panic、不静默跳表）。
    #[test]
    fn o12_table_offset_out_of_range_rejected() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        // 找 head 记录，把 offset 改到文件尾之外。
        let head_idx = ot
            .directory()
            .records
            .iter()
            .position(|r| r.tag == TAG_HEAD)
            .expect("目录含 head");
        let rec_base = 12 + head_idx * 16;
        let mut d = FONT_REGULAR.to_vec();
        let bad = (d.len() as u32) + 4096;
        d[rec_base + 8..rec_base + 12].copy_from_slice(&bad.to_be_bytes());
        let e = OtFont::parse(&d).expect_err("越界 offset 应被拒");
        assert_eq!(e.code, "T010");
        assert!(e.problem.contains("越出文件"), "实际：{}", e.problem);
    }

    /// O13：length 超表（offset+length 溢出/越界）⇒ T010。
    #[test]
    fn o13_table_length_overflow_rejected() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let head_idx = ot
            .directory()
            .records
            .iter()
            .position(|r| r.tag == TAG_HEAD)
            .expect("目录含 head");
        let rec_base = 12 + head_idx * 16;
        let mut d = FONT_REGULAR.to_vec();
        d[rec_base + 12..rec_base + 16].copy_from_slice(&u32::MAX.to_be_bytes());
        let e = OtFont::parse(&d).expect_err("length 溢出应被拒");
        assert_eq!(e.code, "T010");
        // offset(204) + u32::MAX 不触发 usize 溢出，但区间远超文件长 ⇒ 同样结构化拒绝。
        assert!(
            e.problem.contains("越出文件") || e.problem.contains("溢出"),
            "实际：{}",
            e.problem
        );
    }

    /// O14：错误的 `head.magicNumber` ⇒ T013（结构化拒绝，不是 TTF）。
    #[test]
    fn o14_bad_head_magic_rejected() {
        let ot = OtFont::parse(FONT_REGULAR).expect("容器可解析");
        let head = ot.table(TAG_HEAD).expect("head 表");
        let head_file_off = head.as_ptr() as usize - FONT_REGULAR.as_ptr() as usize;
        let mut d = FONT_REGULAR.to_vec();
        d[head_file_off + 12..head_file_off + 16].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        let ot2 = OtFont::parse(&d).expect("容器层仍可解析（只坏了 magic）");
        let e = ot2.head().expect_err("坏 magic 应被拒");
        assert_eq!(e.code, "T013");
        assert!(e.problem.contains("magicNumber"));
    }

    /// O15：CFF / CFF2 字体 ⇒ T016 结构化拒绝（本引擎只解 glyf 二次轮廓）。
    #[test]
    fn o15_cff_font_structurally_rejected() {
        // 用最小表目录伪造一个 'OTTO' + CFF2 的容器。
        let cff2 = vec![0u8; 16];
        let mut tables: Vec<([u8; 4], Vec<u8>)> = Vec::new();
        let mut head = Blob::new();
        head.u32(0x0001_0000);
        head.u32(0);
        head.u32(0);
        head.u32(HEAD_MAGIC);
        head.u16(0);
        head.u16(1000);
        for _ in 0..4 {
            head.u32(0);
        }
        head.i16(0);
        head.i16(0);
        head.i16(1000);
        head.i16(1000);
        head.u16(0);
        head.u16(8);
        head.i16(2);
        head.i16(1);
        head.i16(0);
        tables.push((*b"head", head.into_vec()));
        tables.push((*b"CFF2", cff2));
        let font = sf::assemble_sfnt(SFNT_OTTO, &tables);

        let ot = OtFont::parse(&font).expect("容器层可解析");
        assert_eq!(ot.flavor(), OutlineFlavor::Cff, "CFF2 表应被判为 CFF 轮廓");
        let e = ot.require_glyf().expect_err("CFF 轮廓必须结构化拒绝");
        assert_eq!(e.code, "T016");
        assert!(e.fix.contains("glyf"), "修复建议应指向 glyf 转换");
        assert!(ot.glyphs().is_err(), "委派 ttf::Font 也必须失败而非错图");
    }

    // ---- O16..O20：自造最小字体的容器层 ----

    /// O16：自造字体（format 4 only）容器 + 目录 + 必需表齐备。
    #[test]
    fn o16_synth_font_container_and_directory() {
        let data = sf::build_font(&[4]);
        let ot = OtFont::parse(&data).expect("自造字体应可解析容器");
        assert_eq!(ot.sfnt_version(), SFNT_TRUETYPE);
        assert_eq!(ot.directory().records.len(), 10, "9 表 + 哨兵");
        for t in [TAG_HEAD, TAG_MAXP, TAG_HHEA, TAG_HMTX, TAG_CMAP, TAG_LOCA, TAG_GLYF, TAG_GSUB, TAG_GPOS, TAG_NAME] {
            assert!(ot.directory().contains(t), "自造字体应含 {}", String::from_utf8_lossy(&t));
        }
        // 目录按 tag 升序（规范）。
        let tags = ot.directory().tags();
        let mut sorted = tags.clone();
        sorted.sort();
        assert_eq!(tags, sorted, "表目录须按 tag 升序");
        // 校验和全通过。
        assert_eq!(ot.first_checksum_mismatch(), None);
        // 表数据切片可取。
        assert!(!ot.table(TAG_GSUB).expect("GSUB").is_empty());
        assert!(ot.gsub().is_some() && ot.gpos().is_some());
    }

    /// O17：自造字体的 `head`/`maxp`/`hhea` 可解析，且与 ttf.rs 一致。
    #[test]
    fn o17_synth_font_head_maxp_hhea() {
        let data = sf::build_font(&[4]);
        let ot = OtFont::parse(&data).expect("容器可解析");
        let h = ot.head().expect("head");
        assert_eq!(h.units_per_em, 1000);
        assert_eq!(h.index_to_loc_format, 1);
        assert_eq!(h.bbox, [0, -200, 1000, 800]);
        let m = ot.maxp().expect("maxp");
        assert_eq!(m.num_glyphs, sf::NUM_GLYPHS);
        let hh = ot.hhea().expect("hhea");
        assert_eq!((hh.ascender, hh.descender, hh.line_gap), (800, -200, 0));
        assert_eq!(hh.num_h_metrics, sf::NUM_GLYPHS);
        // 委派 ttf::Font 读出的度量必须一致。
        let f = ot.glyphs().expect("ttf::Font 委派");
        assert_eq!(f.units_per_em(), h.units_per_em);
        assert_eq!(f.num_glyphs(), u32::from(m.num_glyphs));
        assert_eq!(f.ascender(), hh.ascender);
    }

    /// O18：自造字体的字形轮廓是**真实**二次贝塞尔（包围盒 + 四态各异）。
    #[test]
    fn o18_synth_glyphs_have_real_outlines() {
        let data = sf::build_font(&[4]);
        let ot = OtFont::parse(&data).expect("容器可解析");
        let f = ot.glyphs().expect("ttf::Font 委派");
        // 空格无墨。
        assert!(f.outline(sf::G_SPACE).expect("空格轮廓").contours.is_empty());
        // BEH 是矩形 (100,0)-(400,700)。
        let bb = outline_bbox(&f.outline(sf::G_BEH).expect("BEH 轮廓")).expect("BEH 有轮廓");
        assert_eq!((bb.x_min, bb.y_min, bb.x_max, bb.y_max), (100.0, 0.0, 400.0, 700.0));
        // 四态的包围盒两两不同（端到端"像素可辨"的前提）。
        let boxes: Vec<(f32, f32, f32, f32)> = [
            sf::G_BEH_ISOL,
            sf::G_BEH_INITIAL,
            sf::G_BEH_MEDIAL,
            sf::G_BEH_FINAL,
        ]
        .iter()
        .map(|&g| {
            let b = outline_bbox(&f.outline(g).expect("四态轮廓")).expect("四态有轮廓");
            (b.x_min, b.y_min, b.x_max, b.y_max)
        })
        .collect();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                assert_ne!(boxes[i], boxes[j], "四态包围盒必须互异（{i} vs {j}）");
            }
        }
    }

    /// O19：cmap format 4 路径（自造字体只含 format 4 子表）。
    #[test]
    fn o19_cmap_format4_lookup() {
        let data = sf::build_font(&[4]);
        let ot = OtFont::parse(&data).expect("容器可解析");
        let subs = ot.cmap_subtables().expect("cmap 目录");
        assert_eq!(subs.len(), 1, "只含一条编码记录");
        assert_eq!(subs[0].format, 4);
        let f = ot.glyphs().expect("ttf::Font");
        // format 4 只覆盖 BMP。
        assert_eq!(f.glyph_index('\u{0628}'), Some(sf::G_BEH), "BEH");
        assert_eq!(f.glyph_index('\u{0644}'), Some(sf::G_LAM), "LAM");
        assert_eq!(f.glyph_index('\u{093F}'), Some(sf::G_IMATRA), "i-matra");
        // 增补平面码位在 format 4 下查不到（BMP 之外）。
        assert_eq!(f.glyph_index('\u{1_0A00}'), None, "format 4 不含增补平面");
    }

    /// O20：cmap format 12 路径（自造字体只含 format 12 子表）+ 增补平面码位。
    #[test]
    fn o20_cmap_format12_lookup() {
        let data = sf::build_font(&[12]);
        let ot = OtFont::parse(&data).expect("容器可解析");
        let subs = ot.cmap_subtables().expect("cmap 目录");
        assert_eq!(subs[0].format, 12, "应选中 format 12 子表");
        let f = ot.glyphs().expect("ttf::Font");
        assert_eq!(f.glyph_index('\u{0628}'), Some(sf::G_BEH));
        // format 12 覆盖 32 位码位：增补平面可查到。
        assert_eq!(f.glyph_index('\u{1_0A00}'), Some(sf::G_ALEF), "format 12 含增补平面");
    }

    /// O21：`Reader` 越界纪律（每次读取都结构化报错，零 panic）。
    #[test]
    fn o21_reader_bounds_discipline() {
        let d = [0u8, 1, 2, 3];
        let mut r = Reader::new(&d);
        assert_eq!(r.u16().expect("前 2 字节"), 0x0001);
        // seek 越界。
        assert_eq!(r.seek(99).expect_err("越界 seek").code, "T011");
        // 读到流尾后再读越界。
        let mut r2 = Reader::new(&d);
        r2.skip(4).expect("恰好到尾");
        assert_eq!(r2.u8().expect_err("尾后读越界").code, "T011");
        // bytes() 越界。
        let mut r3 = Reader::new(&d);
        assert_eq!(r3.bytes(5).expect_err("读 5B 越界").code, "T011");
        // 长度溢出。
        let mut r4 = Reader::new(&d);
        assert_eq!(r4.bytes(usize::MAX).expect_err("长度溢出").code, "T011");
    }
}
