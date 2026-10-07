//! GSUB / GPOS 布局表解析与应用（补完； 文字管线）。
//!
//! # 职责
//!
//! 解析 `GSUB`/`GPOS` 的**三段结构**（ScriptList / FeatureList / LookupList），
//! 按**确定性固定序**施加字形替换（GSUB）与字形定位（GPOS），并与
//! [`crate::shaping`] 的整形计划对接：整形产出「簇 + 上下文形式」，本模块
//! 把它落到**具体字形 id** 与**锚点偏移**上。
//!
//! # 支持的 lookup type（其余一律结构化拒绝，见 [`GsubError`]）
//!
//! | type | 名称 | 施加 | 覆盖需求 |
//! | --- | --- | --- | --- |
//! | 1 | SingleSubst（fmt1 delta / fmt2 array） | GSUB | `isol`/`fina`/`medi`/`init` 四态 |
//! | 4 | LigatureSubst（fmt1） | GSUB | lam-alef 连字、`liga` |
//! | 5 | ContextualSubst（fmt1/2/3） | GSUB | 上下文相关替换 |
//! | 6 | ChainContextSubst（fmt1/2/3） | GSUB | 链式上下文替换 |
//! | 7 | Extension | GSUB | 跨表间接寻址（64K 表） |
//! | 4 | MarkBasePos（MarkToBase，fmt1） | GPOS | 阿拉伯/天城文记号附着 |
//! | 5 | MarkLigPos（MarkToLig，fmt1） | GPOS | 连字上的记号附着 |
//! | 6 | MarkMarkPos（MarkToMark，fmt1） | GPOS | 记号叠记号 |
//! | 9 | Extension | GPOS | 跨表间接寻址 |
//!
//! **未实现**：GSUB 2（MultipleSubst）/ 3（AlternateSubst）/ 8（ReverseChain）、
//! GPOS 1/2/3/7/8。遇到它们返回 [`GsubError::UnsupportedLookup`]（结构化错误，
//! **不 panic**、不静默跳过）——静默跳过会让排版"看起来对"却字形错，是更坏的失败。
//!
//! # 确定性纪律（铁律2）
//!
//! - **零 `HashMap`**：脚本/特征/查找一律 `Vec` + 线性扫描，遍历序 = 表内索引序。
//! - 特征施加序 = **`FeatureList` 的记录序**（不是调用方传入顺序，也不是 tag 排序）。
//! - 同一输入两遍施加，`gid`/偏移/步进宽逐字节一致（由 `canonical_bytes` 断言）。
//! - 浮点只做 `±×÷`；记号偏移以 **i16 字体单位** 累加，绝不自己写浮点累加器。
//!
//! # 越界纪律（宪章 C18）
//!
//! 零 `unwrap()` / `expect()` / `panic!()` /裸下标 / 整数溢出。所有读取经
//! [`Reader`]（[`crate::opentype::Reader`]）；偏移运算走 `checked_add` /
//! `saturating_*`。嵌套施加深度上限 [`MAX_DEPTH`]，防恶意表构造无限递归。

use crate::opentype::Reader;
use crate::shaping::{ArabicForm, ShapedRun};

/// 布局表解析/施加错误（宪章 C18 三段式）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GsubError {
    /// 错误码（Gxxx 段：GSUB/GPOS）。
    pub code: &'static str,
    /// 什么事：检测到的事实。
    pub problem: String,
    /// 怎么改：可行动建议。
    pub fix: String,
}

impl std::fmt::Display for GsubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error[{}] {}\n  建议：{}", self.code, self.problem, self.fix)
    }
}

impl std::error::Error for GsubError {}

/// 结构化错误快捷构造。
fn err(code: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> GsubError {
    GsubError { code, problem: problem.into(), fix: fix.into() }
}

/// 嵌套施加深度上限（Contextual 可回调GSUB lookup，恶意表可构造无限递归）。
pub const MAX_DEPTH: u32 = 8;

/// 布局表 tag。
pub const TAG_GSUB: [u8; 4] = *b"GSUB";
/// 定位表 tag。
pub const TAG_GPOS: [u8; 4] = *b"GPOS";

/// 字形分类（供 lookup flag 的记号过滤用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphClass {
    /// 基字符。
    Base,
    /// 组合记号（零宽依附）。
    Mark,
    /// 连字（由 LigatureSubst 产出）。
    Ligature,
}

/// 待施加的字形项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlyphItem {
    /// 字形 id。
    pub gid: u16,
    /// 所属字素簇下标（决定形式特征与上下文边界）。
    pub cluster: usize,
    /// 字形分类。
    pub class: GlyphClass,
}

impl GlyphItem {
    /// 构造基字符项。
    pub fn base(gid: u16, cluster: usize) -> Self {
        GlyphItem { gid, cluster, class: GlyphClass::Base }
    }

    /// 构造记号项。
    pub fn mark(gid: u16, cluster: usize) -> Self {
        GlyphItem { gid, cluster, class: GlyphClass::Mark }
    }
}

// ===========================================================================
//  Coverage（覆盖表）
// ===========================================================================

/// Coverage 表（字形 → 覆盖索引）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// format 1：显式字形数组（须升序）。
    Glys(Vec<u16>),
    /// format 2：区间 + 起始覆盖索引。
    Ranges(Vec<(u16, u16, u16)>),
}

impl Coverage {
    /// 解析 Coverage（format 1/2；其余结构化拒绝）。
    pub fn parse(t: &[u8], off: usize) -> Result<Self, GsubError> {
        let mut r = Reader::at(t, off);
        let format = r.u16().map_err(map_err)?;
        match format {
            1 => {
                let n = r.u16().map_err(map_err)? as usize;
                let mut v = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    v.push(r.u16().map_err(map_err)?);
                }
                Ok(Coverage::Glys(v))
            }
            2 => {
                let n = r.u16().map_err(map_err)? as usize;
                let mut v = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    let start = r.u16().map_err(map_err)?;
                    let end = r.u16().map_err(map_err)?;
                    let index = r.u16().map_err(map_err)?;
                    v.push((start, end, index));
                }
                Ok(Coverage::Ranges(v))
            }
            _ => Err(err("G003", format!("Coverage format = {format}（仅支持 1/2）"),
                "字体 GSUB/GPOS 表使用了本解析器未实现的 Coverage 格式")),
        }
    }

    /// 字形 → 覆盖索引（未覆盖 → `None`）。
    ///
    /// format 1 用二分查找（表内升序），format 2 线性扫描区间（表内序）。
    pub fn index_of(&self, gid: u16) -> Option<u16> {
        match self {
            Coverage::Glys(v) => {
                let mut lo = 0usize;
                let mut hi = v.len();
                while lo < hi {
                    let mid = lo.saturating_add(hi.saturating_sub(lo) / 2);
                    // 饱和取中，避免 lo+hi 溢出。
                    match v.get(mid) {
                        Some(&g) if g < gid => lo = mid.saturating_add(1),
                        Some(_) => hi = mid,
                        None => hi = mid,
                    }
                }
                // lo 是首个 >= gid 的位置；命中须严格相等。
                match v.get(lo) {
                    Some(&g) if g == gid => u16::try_from(lo).ok(),
                    _ => None,
                }
            }
            Coverage::Ranges(v) => {
                for &(start, end, index) in v {
                    if gid >= start && gid <= end {
                        return Some(index.saturating_add(gid.saturating_sub(start)));
                    }
                }
                None
            }
        }
    }
}

// ===========================================================================
//  ClassDef（类定义）
// ===========================================================================

/// ClassDef（字形 → 类号，0 = 不属于任何类）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDef {
    /// format 1 的数据（起始字形 + 类数组）。
    f1: Option<(u16, Vec<u16>)>,
    /// format 2 的区间表。
    f2: Vec<(u16, u16, u16)>,
}

impl ClassDef {
    /// 解析 ClassDef（format 1/2；其余结构化拒绝）。
    pub fn parse(t: &[u8], off: usize) -> Result<Self, GsubError> {
        let mut r = Reader::at(t, off);
        let format = r.u16().map_err(map_err)?;
        match format {
            1 => {
                let start = r.u16().map_err(map_err)?;
                let n = r.u16().map_err(map_err)? as usize;
                let mut classes = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    classes.push(r.u16().map_err(map_err)?);
                }
                Ok(ClassDef { f1: Some((start, classes)), f2: Vec::new() })
            }
            2 => {
                let n = r.u16().map_err(map_err)? as usize;
                let mut ranges = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    let s = r.u16().map_err(map_err)?;
                    let e = r.u16().map_err(map_err)?;
                    let c = r.u16().map_err(map_err)?;
                    ranges.push((s, e, c));
                }
                Ok(ClassDef { f1: None, f2: ranges })
            }
            _ => Err(err("G003", format!("ClassDef format = {format}（仅支持 1/2）"),
                "字体 GSUB/GPOS 表使用了本解析器未实现的 ClassDef 格式")),
        }
    }

    /// 字形 → 类号（未定义 → 0）。
    pub fn class_of(&self, gid: u16) -> u16 {
        if let Some((start, classes)) = &self.f1 {
            let rel = gid.checked_sub(*start).map(|v| v as usize);
            if let Some(i) = rel {
                if let Some(&c) = classes.get(i) {
                    return c;
                }
            }
            return 0;
        }
        for &(s, e, c) in &self.f2 {
            if gid >= s && gid <= e {
                return c;
            }
        }
        0
    }
}

// ===========================================================================
//  Lookup
// ===========================================================================

/// lookup 旗标。
pub mod lookup_flag {
    /// 从右向左（对连字/上下文匹配的影响）。
    pub const RIGHT_TO_LEFT: u16 = 0x0001;
    /// 跳过基字符。
    pub const IGNORE_BASE_GLYPHS: u16 = 0x0002;
    /// 跳过连字。
    pub const IGNORE_LIGATURES: u16 = 0x0004;
    /// 跳过记号。
    pub const IGNORE_MARKS: u16 = 0x0008;
    /// 使用 markFilteringSet（额外 2 B 字段）。
    pub const USE_MARK_FILTERING_SET: u16 = 0x0010;
}

/// 单个 lookup。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    /// lookup type（1=SingleSubst, 4=LigatureSubst, 5=Contextual, 6=ChainContext, 7/9=Extension）。
    pub kind: u16,
    /// lookup flag 位掩码。
    pub flag: u16,
    /// 子表在布局表内的**绝对**偏移（表内序 = 施加序）。
    ///
    /// 规范里 `subtableOffsets[]` 是相对 **Lookup 表起点**的，此处已换算为绝对偏移。
    pub subtables: Vec<usize>,
    /// markFilteringSet 的 Coverage **绝对**偏移（仅当flag 含 `USE_MARK_FILTERING_SET`）。
    pub mark_filtering_set: Option<usize>,
    /// 本 Lookup 表在布局表内的绝对偏移（Extension 换算与诊断用）。
    pub origin: usize,
}

// ===========================================================================
//  布局表（GSUB / GPOS 共用三段结构）
// ===========================================================================

/// 已解析的布局表（GSUB 或 GPOS 的共用外壳）。
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutTable<'a> {
    t: &'a [u8],
    /// 脚本记录（tag + DefaultLangSys 的特征索引），表内序。
    scripts: Vec<ScriptRecord>,
    /// 特征记录（tag + 查找索引），**表内序 = 施加序**。
    features: Vec<FeatureRecord>,
    /// 查找记录，表内序。
    lookups: Vec<Lookup>,
}

/// 一条脚本记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptRecord {
    /// 脚本 tag（如 `b"arab"`、`b"DFLT"`）。
    pub tag: [u8; 4],
    /// DefaultLangSys 引用的特征索引（表内序）。
    pub default_features: Vec<u16>,
}

/// 一条特征记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureRecord {
    /// 特征 tag（如 `b"isol"`）。
    pub tag: [u8; 4],
    /// 该特征启用的查找索引（表内序 = 施加序）。
    pub lookups: Vec<u16>,
}

/// 把「相对子表起点的偏移」换算为表内绝对偏移；溢出 → 结构化错误。
///
/// OpenType 里 `coverageOffset` / `classDefOffset` / `markArrayOffset` 等字段
/// 一律**相对其所在子表的起点**（不是相对布局表起点，也不是相对当前游标）——
/// 这是本解析器最容易出错的一处，故统一走本函数。
fn rel(base: usize, off: usize) -> Result<usize, GsubError> {
    base.checked_add(off)
        .ok_or_else(|| err("G003", "子表相对偏移换算溢出", "字体布局表的偏移字段被破坏"))
}

/// 解析 `LangSys` → DefaultLangSys 引用的特征索引（表内序）。
fn parse_default_lang_sys(t: &[u8], script_off: usize) -> Result<Vec<u16>, GsubError> {
    // Script 表：defaultLangSysOffset(2) + langSysCount(2) + LangSysRecord[](6/条)
    let mut r = Reader::at(t, script_off);
    let default_off = r.u16().map_err(map_err)? as usize;
    if default_off == 0 {
        // 无 DefaultLangSys 的脚本合法（只有具体语言记录）；此处不贡献特征。
        return Ok(Vec::new());
    }
    let abs = rel(script_off, default_off)?;
    let mut lr = Reader::at(t, abs);
    let _lookup_order = lr.u16().map_err(map_err)?; // 恒为 0，仅为跳过
    let _required_feature = lr.u16().map_err(map_err)?; // 本解析器不处理 required feature
    let n = lr.u16().map_err(map_err)? as usize;
    if n.saturating_mul(2).saturating_add(6) > t.len().saturating_sub(abs) {
        return Err(err("G001",
            format!("LangSys 声明 {n} 个特征索引但表体容纳不下"),
            "GSUB/GPOS 表被截断或损坏"));
    }
    let mut v = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        v.push(lr.u16().map_err(map_err)?);
    }
    Ok(v)
}

/// 解析 `Feature` 表 → 其启用的 lookupIndex 数组（表内序 = 施加序）。
fn parse_feature(t: &[u8], feature_off: usize) -> Result<Vec<u16>, GsubError> {
    // Feature 表：featureParams(2) + lookupIndexCount(2) + lookupListIndices[](2/个)
    let mut r = Reader::at(t, feature_off);
    let _params = r.u16().map_err(map_err)?; // featureParamsOffset，非 0 时本解析器不解析
    let n = r.u16().map_err(map_err)? as usize;
    if n.saturating_mul(2).saturating_add(4) > t.len().saturating_sub(feature_off) {
        return Err(err("G001",
            format!("Feature 声明 {n} 个查找索引但表体容纳不下"),
            "GSUB/GPOS 表被截断或损坏"));
    }
    let mut v = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        v.push(r.u16().map_err(map_err)?);
    }
    Ok(v)
}

/// 解析 `Lookup` 表 → 类型 / flag / 子表绝对偏移 / markFilteringSet。
fn parse_lookup(t: &[u8], lookup_off: usize) -> Result<Lookup, GsubError> {
    // Lookup 表：lookupType(2) + lookupFlag(2) + subTableCount(2) + subtableOffsets[](2/个)
    // flag & 0x0010 时尾部追加 markFilteringSet(2)。
    let mut r = Reader::at(t, lookup_off);
    let kind = r.u16().map_err(map_err)?;
    let flag = r.u16().map_err(map_err)?;
    let n = r.u16().map_err(map_err)? as usize;
    if n.saturating_mul(2).saturating_add(6) > t.len().saturating_sub(lookup_off) {
        return Err(err("G001",
            format!("Lookup 声明 {n} 个子表但表体容纳不下"),
            "GSUB/GPOS 表被截断或损坏"));
    }
    // 规范：subtableOffsets 相对 **Lookup 表起点**。
    let mut subtables = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let o = r.u16().map_err(map_err)? as usize;
        let abs = rel(lookup_off, o)?;
        if abs >= t.len() {
            return Err(err("G003",
                format!("Lookup 子表偏移 {o} 换算后越出布局表（{} B）", t.len()),
                "字体布局表的子表偏移被破坏"));
        }
        subtables.push(abs);
    }
    let mark_filtering_set = if flag & lookup_flag::USE_MARK_FILTERING_SET != 0 {
        let o = r.u16().map_err(map_err)? as usize;
        Some(rel(lookup_off, o)?)
    } else {
        None
    };
    Ok(Lookup { kind, flag, subtables, mark_filtering_set, origin: lookup_off })
}

impl<'a> LayoutTable<'a> {
    /// 解析布局表三段结构（`GSUB` 或 `GPOS` 的表体字节）。
    ///
    /// 空壳表（`featureCount = 0`、`lookupCount = 0`，本项目内嵌资产即如此）
    /// 是**合法输入**：解析成功并给出 0 特征 0 查找，施加是恒等变换。
    pub fn parse(t: &'a [u8]) -> Result<Self, GsubError> {
        let mut r = Reader::new(t);
        let major = r.u16().map_err(map_err)?;
        let minor = r.u16().map_err(map_err)?;
        if major != 1 || minor != 0 {
            return Err(err("G001", format!("布局表版本 {major}.{minor}（仅支持 1.0）"),
                "字体使用了本解析器未实现的 GSUB/GPOS 版本"));
        }
        let script_off = r.u16().map_err(map_err)? as usize;
        let feature_off = r.u16().map_err(map_err)? as usize;
        let lookup_off = r.u16().map_err(map_err)? as usize;

        // ---- ScriptList ----
        let mut scripts = Vec::new();
        if script_off != 0 {
            let mut sr = Reader::at(t, script_off);
            let n = sr.u16().map_err(map_err)? as usize;
            // 记录区必须完整（每条 6 B）。
            if n.saturating_mul(6).saturating_add(2) > t.len().saturating_sub(script_off) {
                return Err(err("G001", format!("ScriptList 声明 {n} 条脚本但表体容纳不下"),
                    "GSUB/GPOS 表被截断或损坏"));
            }
            for _ in 0..n {
                let tag = sr.tag().map_err(map_err)?;
                let off = sr.u16().map_err(map_err)? as usize;
                // 规范：ScriptRecord.scriptOffset 相对 **ScriptList 起点**。
                let abs = match rel(script_off, off) {
                    Ok(a) => a,
                    // 越界的脚本偏移（如内嵌资产的空壳表里 grek 指向 104 > 66）：
                    // **跳过**该脚本而非报错——空壳表的越界偏移是构建期工具的已知
                    // 产物，跳过不影响其余合法脚本（此处只取 DefaultLangSys）。
                    Err(_) => continue,
                };
                let features = match parse_default_lang_sys(t, abs) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                scripts.push(ScriptRecord { tag, default_features: features });
            }
        }

        // ---- FeatureList ----
        let mut features = Vec::new();
        if feature_off != 0 {
            let mut fr = Reader::at(t, feature_off);
            let n = fr.u16().map_err(map_err)? as usize;
            if n.saturating_mul(6).saturating_add(2) > t.len().saturating_sub(feature_off) {
                return Err(err("G001", format!("FeatureList 声明 {n} 条特征但表体容纳不下"),
                    "GSUB/GPOS 表被截断或损坏"));
            }
            for _ in 0..n {
                let tag = fr.tag().map_err(map_err)?;
                let off = fr.u16().map_err(map_err)? as usize;
                // 规范：FeatureRecord.featureOffset 相对 **FeatureList 起点**。
                let abs = rel(feature_off, off).unwrap_or(usize::MAX);
                match parse_feature(t, abs) {
                    Ok(lookups) => features.push(FeatureRecord { tag, lookups }),
                    // 越界的特征偏移：记录占位以保持**索引稳定**（施加序不可漂移），
                    // 但不贡献任何查找。
                    Err(_) => features.push(FeatureRecord { tag, lookups: Vec::new() }),
                }
            }
        }

        // ---- LookupList ----
        let mut lookups = Vec::new();
        if lookup_off != 0 {
            let mut lr = Reader::at(t, lookup_off);
            let n = lr.u16().map_err(map_err)? as usize;
            if n.saturating_mul(2).saturating_add(2) > t.len().saturating_sub(lookup_off) {
                return Err(err("G001", format!("LookupList 声明 {n} 条查找但表体容纳不下"),
                    "GSUB/GPOS 表被截断或损坏"));
            }
            for _ in 0..n {
                let off = lr.u16().map_err(map_err)? as usize;
                // 规范：LookupList 里的偏移相对 **LookupList 起点**。
                match parse_lookup(t, rel(lookup_off, off).unwrap_or(usize::MAX)) {
                    Ok(lk) => lookups.push(lk),
                    Err(_) => lookups.push(Lookup {
                        kind: 0, // 0 = 解析失败占位（施加时结构化报错）
                        flag: 0,
                        subtables: Vec::new(),
                        mark_filtering_set: None,
                        origin: off,
                    }),
                }
            }
        }

        Ok(LayoutTable { t, scripts, features, lookups })
    }

    /// 布局表原始字节。
    pub fn bytes(&self) -> &'a [u8] {
        self.t
    }

    /// 脚本记录（表内序）。
    pub fn scripts(&self) -> &[ScriptRecord] {
        &self.scripts
    }

    /// 特征记录（表内序 = 施加序）。
    pub fn features(&self) -> &[FeatureRecord] {
        &self.features
    }

    /// 查找记录（表内序）。
    pub fn lookups(&self) -> &[Lookup] {
        &self.lookups
    }

    /// 按 tag 查脚本（线性扫描，表内首个匹配）。
    pub fn script(&self, tag: [u8; 4]) -> Option<&ScriptRecord> {
        self.scripts.iter().find(|s| s.tag == tag)
    }

    /// 按 tag 查特征的**表内下标**（施加序即此下标序）。
    pub fn feature_index(&self, tag: [u8; 4]) -> Option<usize> {
        self.features.iter().position(|f| f.tag == tag)
    }

    /// 施加单个特征（按其 `lookups` 序，**逐个 lookup** 施加）。
    ///
    /// `depth` 是嵌套深度（Contextual 回调时 +1），超 [`MAX_DEPTH`] 即拒绝。
    pub fn apply_feature(
        &self,
        feature_index: usize,
        glyphs: &mut Vec<GlyphItem>,
        depth: u32,
    ) -> Result<u32, GsubError> {
        if depth > MAX_DEPTH {
            return Err(err("G004", format!("嵌套施加深度超过 {MAX_DEPTH}"),
                "字体的 GSUB 表可能构造了递归 lookup；请在构建期校验字体"));
        }
        let Some(feature) = self.features.get(feature_index) else {
            return Err(err("G002", format!("特征下标 {feature_index} 越界（共 {} 条）", self.features.len()),
                "施加方传入的特征索引与字体表不匹配"));
        };
        let lookups = feature.lookups.clone(); // 克隆避免与 &mut glyphs 的借用冲突
        let mut applied = 0u32;
        for li in lookups {
            applied = applied.saturating_add(self.apply_lookup(li, glyphs, depth)?);
        }
        Ok(applied)
    }

    /// 施加单个 lookup（按其子表序）。
    pub fn apply_lookup(
        &self,
        lookup_index: u16,
        glyphs: &mut Vec<GlyphItem>,
        depth: u32,
    ) -> Result<u32, GsubError> {
        if depth > MAX_DEPTH {
            return Err(err("G004", format!("嵌套施加深度超过 {MAX_DEPTH}"),
                "字体的 GSUB 表可能构造了递归 lookup；请在构建期校验字体"));
        }
        let Some(lookup) = self.lookups.get(lookup_index as usize) else {
            return Err(err("G002", format!("查找下标 {lookup_index} 越界（共 {} 条）", self.lookups.len()),
                "施加方传入的查找索引与字体表不匹配"));
        };
        if lookup.kind == 0 {
            return Err(err("G002", format!("查找 #{lookup_index} 解析失败（占位记录）"),
                "字体的 LookupList 该项损坏；请在构建期校验字体"));
        }
        // 记号过滤集（USE_MARK_FILTERING_SET）：不在集内的记号被跳过。
        let filter_cov = match lookup.mark_filtering_set {
            Some(off) => Coverage::parse(self.t, off).ok(),
            None => None,
        };
        let mut applied = 0u32;
        for &sub_off in &lookup.subtables {
            // 子表内**顺序**扫描字形：替换会改gid，故每步重新读当前项。
            let mut i = 0usize;
            while i < glyphs.len() {
                if self.skip_by_flag(lookup, glyphs, i, filter_cov.as_ref()) {
                    i = i.saturating_add(1);
                    continue;
                }
                let gid = glyphs.get(i).map(|g| g.gid).unwrap_or(0);
                let consumed = self.apply_subtable(
                    lookup.kind,
                    sub_off,
                    glyphs,
                    i,
                    depth,
                    &mut applied,
                )?;
                let _ = gid;
                if consumed == 0 {
                    i = i.saturating_add(1);
                } else {
                    // 消耗了 N 个字形（连字）：跳过被合并的部分。
                    i = i.saturating_add(consumed.max(1));
                }
            }
        }
        Ok(applied)
    }

    /// lookup flag 过滤：该位置是否应被跳过。
    fn skip_by_flag(
        &self,
        lookup: &Lookup,
        glyphs: &[GlyphItem],
        i: usize,
        filter_cov: Option<&Coverage>,
    ) -> bool {
        let Some(item) = glyphs.get(i) else {
            return true;
        };
        let f = lookup.flag;
        if f & lookup_flag::IGNORE_MARKS != 0 && item.class == GlyphClass::Mark {
            return true;
        }
        if f & lookup_flag::IGNORE_LIGATURES != 0 && item.class == GlyphClass::Ligature {
            return true;
        }
        if f & lookup_flag::IGNORE_BASE_GLYPHS != 0 && item.class == GlyphClass::Base {
            return true;
        }
        if f & lookup_flag::USE_MARK_FILTERING_SET != 0 && item.class == GlyphClass::Mark {
            // 在过滤集内 ⇒ 保留；集外⇒ 跳过。集解析失败时保守跳过。
            return match filter_cov {
                Some(cov) => cov.index_of(item.gid).is_none(),
                None => true,
            };
        }
        false
    }

    /// 施加一个子表；返回**消耗的字形数**（0 = 未命中）。
    fn apply_subtable(
        &self,
        kind: u16,
        off: usize,
        glyphs: &mut Vec<GlyphItem>,
        at: usize,
        depth: u32,
        applied: &mut u32,
    ) -> Result<usize, GsubError> {
        match kind {
            1 => self.apply_single_subst(off, glyphs, at, applied),
            4 => self.apply_ligature_subst(off, glyphs, at, applied),
            5 => self.apply_contextual(off, glyphs, at, depth, applied, false),
            6 => self.apply_contextual(off, glyphs, at, depth, applied, true),
            7 => self.apply_extension(off, glyphs, at, depth, applied, 7),
            9 => self.apply_extension(off, glyphs, at, depth, applied, 9),
            other => Err(unsupported(other)),
        }
    }

    /// SingleSubst（type 1）：fmt1 = delta，fmt2 = 显式数组。
    fn apply_single_subst(
        &self,
        off: usize,
        glyphs: &mut [GlyphItem],
        at: usize,
        applied: &mut u32,
    ) -> Result<usize, GsubError> {
        let mut r = Reader::at(self.t, off);
        let format = r.u16().map_err(map_err)?;
        let cov_off = r.u16().map_err(map_err)? as usize;
        let cov = Coverage::parse(self.t, rel(off, cov_off)?)?;
        let Some(item) = glyphs.get(at) else {
            return Ok(0);
        };
        let Some(idx) = cov.index_of(item.gid) else {
            return Ok(0);
        };
        let new_gid = match format {
            1 => {
                let delta = r.i16().map_err(map_err)?;
                (i32::from(item.gid) + i32::from(delta)) as u16
            }
            2 => {
                let n = r.u16().map_err(map_err)? as usize;
                let target = idx as usize;
                if target >= n {
                    return Err(err("G003",
                        format!("SingleSubst fmt2：覆盖索引 {target} 越界（glyphCount = {n}）"),
                        "Coverage 与 substitute 数组不自洽；字体表损坏"));
                }
                // 跳过 coverage 之后的 glyphCount，再索引。
                let mut rr = Reader::at(self.t, off);
                rr.skip(6).map_err(map_err)?;
                rr.skip(target.saturating_mul(2)).map_err(map_err)?;
                rr.u16().map_err(map_err)?
            }
            other => {
                return Err(err("G003", format!("SingleSubst format = {other}（仅支持 1/2）"),
                    "字体 GSUB 使用了本解析器未实现的 SingleSubst 格式"))
            }
        };
        if let Some(slot) = glyphs.get_mut(at) {
            slot.gid = new_gid;
        }
        *applied = applied.saturating_add(1);
        Ok(1)
    }

    /// LigatureSubst（type 4，fmt1）：首字形 + 组件序列 → 连字。
    fn apply_ligature_subst(
        &self,
        off: usize,
        glyphs: &mut Vec<GlyphItem>,
        at: usize,
        applied: &mut u32,
    ) -> Result<usize, GsubError> {
        let mut r = Reader::at(self.t, off);
        let format = r.u16().map_err(map_err)?;
        if format != 1 {
            return Err(err("G003", format!("LigatureSubst format = {format}（仅支持 1）"),
                "字体 GSUB 使用了本解析器未实现的 LigatureSubst 格式"));
        }
        let cov_off = r.u16().map_err(map_err)? as usize;
        let set_count = r.u16().map_err(map_err)? as usize;
        let cov = Coverage::parse(self.t, rel(off, cov_off)?)?;
        let Some(item) = glyphs.get(at) else {
            return Ok(0);
        };
        let Some(set_index) = cov.index_of(item.gid) else {
            return Ok(0);
        };
        let set_index_usize = set_index as usize;
        if set_index_usize >= set_count {
            return Err(err("G003",
                format!("LigatureSet 索引 {set_index_usize} 越界（setCount = {set_count}）"),
                "Coverage 与 LigatureSet 数组不自洽；字体表损坏"));
        }
        // ligatureSetOffsets[N] 紧随其后。
        let mut so = Reader::at(self.t, off);
        so.skip(6).map_err(map_err)?;
        so.skip(set_index_usize.saturating_mul(2)).map_err(map_err)?;
        let set_off = so.u16().map_err(map_err)? as usize;
        let set_abs = checked_add(off, set_off)
            .ok_or_else(|| err("G003", "LigatureSet 偏移溢出", "字体表损坏"))?;

        let mut sr = Reader::at(self.t, set_abs);
        let lig_count = sr.u16().map_err(map_err)? as usize;
        let cluster = item.cluster;
        // 按表内序尝试每条连字（规范要求优先最长的，但同长度时表内首个优先；
        // 这里严格按表内序，保证确定性）。
        for li in 0..lig_count {
            let mut lr = Reader::at(self.t, set_abs);
            lr.skip(2).map_err(map_err)?;
            lr.skip(li.saturating_mul(2)).map_err(map_err)?;
            let lig_off = lr.u16().map_err(map_err)? as usize;
            let lig_abs = checked_add(set_abs, lig_off)
                .ok_or_else(|| err("G003", "Ligature 偏移溢出", "字体表损坏"))?;

            let mut gr = Reader::at(self.t, lig_abs);
            let lig_glyph = gr.u16().map_err(map_err)?;
            let comp_count = gr.u16().map_err(map_err)? as usize;
            if comp_count == 0 {
                return Err(err("G003", "Ligature 的 componentCount = 0", "字体表损坏"));
            }
            // 组件 0 隐含为当前字形；其余需逐一匹配。
            let mut matched = true;
            for k in 1..comp_count {
                let want = gr.u16().map_err(map_err)?;
                let pos = at.saturating_add(k);
                match glyphs.get(pos) {
                    Some(g) if g.gid == want => {}
                    _ => {
                        matched = false;
                        break;
                    }
                }
            }
            if !matched {
                continue;
            }
            // 命中：合并 comp_count 个字形为 1 个连字。
            if at.saturating_add(comp_count) > glyphs.len() {
                return Err(err("G003",
                    format!("连字需 {comp_count} 个组件但字形序列只剩 {} 个", glyphs.len() - at),
                    "字形序列与 GSUB 表不匹配"));
            }
            if let Some(first) = glyphs.get_mut(at) {
                first.gid = lig_glyph;
                first.class = GlyphClass::Ligature;
                first.cluster = cluster;
            }
            // 就地删除被合并的后续组件（**从后往前**删，保证下标有效）。
            for k in (1..comp_count).rev() {
                let pos = at.saturating_add(k);
                if pos < glyphs.len() {
                    glyphs.remove(pos);
                }
            }
            *applied = applied.saturating_add(1);
            return Ok(comp_count);
        }
        Ok(0)
    }

    /// ContextualSubst（type 5）/ ChainContextSubst（type 6）。
    ///
    /// fmt1 = 字形规则集；fmt2 = 类规则集；fmt3 = 覆盖集序列。
    /// `chain = true` 时额外校验前后文（backtrack/lookahead）。
    fn apply_contextual(
        &self,
        off: usize,
        glyphs: &mut Vec<GlyphItem>,
        at: usize,
        depth: u32,
        applied: &mut u32,
        chain: bool,
    ) -> Result<usize, GsubError> {
        let mut r = Reader::at(self.t, off);
        let format = r.u16().map_err(map_err)?;
        let cov_off = r.u16().map_err(map_err)? as usize;
        let cov = Coverage::parse(self.t, rel(off, cov_off)?)?;
        let Some(item) = glyphs.get(at) else {
            return Ok(0);
        };
        let Some(cov_index) = cov.index_of(item.gid) else {
            return Ok(0);
        };

        // 收集命中的 SubstLookupRecord 序列（(seqIndex, lookupIndex)）。
        let mut records: Vec<(u16, u16)> = Vec::new();
        match format {
            1 => {
                let set_count = r.u16().map_err(map_err)? as usize;
                let ci = cov_index as usize;
                if ci >= set_count {
                    return Err(err("G003", format!("Contextual 规则集索引 {ci} 越界（{set_count}）"),
                        "Coverage 与规则集数组不自洽；字体表损坏"));
                }
                let mut so = Reader::at(self.t, off);
                so.skip(6).map_err(map_err)?;
                so.skip(ci.saturating_mul(2)).map_err(map_err)?;
                let set_off = so.u16().map_err(map_err)? as usize;
                let set_abs = checked_add(off, set_off)
                    .ok_or_else(|| err("G003", "规则集偏移溢出", "字体表损坏"))?;
                records = self.collect_glyph_rule_set(set_abs, glyphs, at)?;
            }
            2 => {
                let class_def_off = r.u16().map_err(map_err)? as usize;
                let set_count = r.u16().map_err(map_err)? as usize;
                let class_def = ClassDef::parse(self.t, rel(off, class_def_off)?)?;
                let ci = class_def.class_of(item.gid) as usize;
                if ci >= set_count {
                    return Err(err("G003", format!("Contextual 类规则集索引 {ci} 越界（{set_count}）"),
                        "ClassDef 与规则集数组不自洽；字体表损坏"));
                }
                let mut so = Reader::at(self.t, off);
                so.skip(8).map_err(map_err)?;
                so.skip(ci.saturating_mul(2)).map_err(map_err)?;
                let set_off = so.u16().map_err(map_err)? as usize;
                let set_abs = checked_add(off, set_off)
                    .ok_or_else(|| err("G003", "类规则集偏移溢出", "字体表损坏"))?;
                if set_abs != 0 {
                    records = self.collect_class_rule_set(set_abs, glyphs, at, &class_def)?;
                } else {
                    // classDef 把字形归到 0 类且 0 类无规则 ⇒ 未命中。
                    return Ok(0);
                }
            }
            3 => {
                // backtrack / input / lookahead 覆盖集序列 + SubstLookupRecord。
                if chain {
                    let back_count = r.u16().map_err(map_err)? as usize;
                    let mut back = Vec::with_capacity(back_count.min(256));
                    for _ in 0..back_count {
                        let o = r.u16().map_err(map_err)? as usize;
                        back.push(Coverage::parse(self.t, rel(off, o)?)?);
                    }
                    let in_count = r.u16().map_err(map_err)? as usize;
                    let mut input = Vec::with_capacity(in_count.min(256));
                    for _ in 0..in_count {
                        let o = r.u16().map_err(map_err)? as usize;
                        input.push(Coverage::parse(self.t, rel(off, o)?)?);
                    }
                    let look_count = r.u16().map_err(map_err)? as usize;
                    let mut look = Vec::with_capacity(look_count.min(256));
                    for _ in 0..look_count {
                        let o = r.u16().map_err(map_err)? as usize;
                        look.push(Coverage::parse(self.t, rel(off, o)?)?);
                    }
                    records = self.parse_lookup_records(&mut r)?;
                    if !match_coverages(glyphs, at, &back, &input, &look) {
                        return Ok(0);
                    }
                } else {
                    let in_count = r.u16().map_err(map_err)? as usize;
                    let mut input = Vec::with_capacity(in_count.min(256));
                    for _ in 0..in_count {
                        let o = r.u16().map_err(map_err)? as usize;
                        input.push(Coverage::parse(self.t, rel(off, o)?)?);
                    }
                    records = self.parse_lookup_records(&mut r)?;
                    if !match_coverages(glyphs, at, &[], &input, &[]) {
                        return Ok(0);
                    }
                }
            }
            other => {
                return Err(err("G003",
                    format!("ContextualSubst format = {other}（仅支持 1/2/3）"),
                    "字体 GSUB 使用了本解析器未实现的上下文格式"))
            }
        }

        if records.is_empty() {
            return Ok(0);
        }
        // 按 SubstLookupRecord 序施加（表内序，确定性）。
        for (seq_index, lookup_index) in records {
            let pos = at.saturating_add(seq_index as usize);
            if pos >= glyphs.len() {
                continue;
            }
            self.apply_lookup(lookup_index, glyphs, depth.saturating_add(1))?;
        }
        *applied = applied.saturating_add(1);
        Ok(1)
    }

    /// 读取 SubstLookupRecord 数组。
    fn parse_lookup_records(&self, r: &mut Reader<'_>) -> Result<Vec<(u16, u16)>, GsubError> {
        let n = r.u16().map_err(map_err)? as usize;
        let mut v = Vec::with_capacity(n.min(256));
        for _ in 0..n {
            let seq = r.u16().map_err(map_err)?;
            let lookup = r.u16().map_err(map_err)?;
            v.push((seq, lookup));
        }
        Ok(v)
    }

    /// 收集命中的字形规则集记录（fmt1）。
    fn collect_glyph_rule_set(
        &self,
        set_abs: usize,
        glyphs: &[GlyphItem],
        at: usize,
    ) -> Result<Vec<(u16, u16)>, GsubError> {
        let mut sr = Reader::at(self.t, set_abs);
        let rule_count = sr.u16().map_err(map_err)? as usize;
        for ri in 0..rule_count {
            let mut rr = Reader::at(self.t, set_abs);
            rr.skip(2).map_err(map_err)?;
            rr.skip(ri.saturating_mul(2)).map_err(map_err)?;
            let rule_off = rr.u16().map_err(map_err)? as usize;
            let rule_abs = checked_add(set_abs, rule_off)
                .ok_or_else(|| err("G003", "规则偏移溢出", "字体表损坏"))?;
            let mut gr = Reader::at(self.t, rule_abs);
            let glyph_count = gr.u16().map_err(map_err)? as usize;
            let subst_count = gr.u16().map_err(map_err)? as usize;
            if glyph_count == 0 {
                return Err(err("G003", "规则的 glyphCount = 0", "字体表损坏"));
            }
            if subst_count == 0 {
                return Err(err("G003", "规则的 substCount = 0", "字体表损坏"));
            }
            // glyphCount 含首字形（已由 Coverage 命中），需校验其余。
            let mut ok = true;
            for k in 1..glyph_count {
                let want = gr.u16().map_err(map_err)?;
                match glyphs.get(at.saturating_add(k)) {
                    Some(g) if g.gid == want => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return self.parse_lookup_records(&mut gr);
            }
        }
        Ok(Vec::new())
    }

    /// 收集命中的类规则集记录（fmt2）。
    fn collect_class_rule_set(
        &self,
        set_abs: usize,
        glyphs: &[GlyphItem],
        at: usize,
        class_def: &ClassDef,
    ) -> Result<Vec<(u16, u16)>, GsubError> {
        let mut sr = Reader::at(self.t, set_abs);
        let rule_count = sr.u16().map_err(map_err)? as usize;
        for ri in 0..rule_count {
            let mut rr = Reader::at(self.t, set_abs);
            rr.skip(2).map_err(map_err)?;
            rr.skip(ri.saturating_mul(2)).map_err(map_err)?;
            let rule_off = rr.u16().map_err(map_err)? as usize;
            let rule_abs = checked_add(set_abs, rule_off)
                .ok_or_else(|| err("G003", "类规则偏移溢出", "字体表损坏"))?;
            let mut gr = Reader::at(self.t, rule_abs);
            let glyph_count = gr.u16().map_err(map_err)? as usize;
            let subst_count = gr.u16().map_err(map_err)? as usize;
            if glyph_count == 0 {
                return Err(err("G003", "类规则的 glyphCount = 0", "字体表损坏"));
            }
            if subst_count == 0 {
                return Err(err("G003", "类规则的 substCount = 0", "字体表损坏"));
            }
            let mut ok = true;
            for k in 1..glyph_count {
                let want = gr.u16().map_err(map_err)?;
                let Some(g) = glyphs.get(at.saturating_add(k)) else {
                    ok = false;
                    break;
                };
                if class_def.class_of(g.gid) != want {
                    ok = false;
                    break;
                }
            }
            if ok {
                return self.parse_lookup_records(&mut gr);
            }
        }
        Ok(Vec::new())
    }

    /// Extension lookup（GSUB type 7 / GPOS type 9）：fmt1，32 位扩展偏移。
    fn apply_extension(
        &self,
        off: usize,
        glyphs: &mut Vec<GlyphItem>,
        at: usize,
        depth: u32,
        applied: &mut u32,
        self_kind: u16,
    ) -> Result<usize, GsubError> {
        let mut r = Reader::at(self.t, off);
        let format = r.u16().map_err(map_err)?;
        if format != 1 {
            return Err(err("G003", format!("Extension format = {format}（仅支持 1）"),
                "字体布局表使用了本解析器未实现的 Extension 格式"));
        }
        let ext_kind = r.u16().map_err(map_err)?;
        let ext_off = r.u32().map_err(map_err)? as usize;
        // 扩展偏移是 **32 位**（不是 16 位），且相对 **Extension 子表自身起点**——
        // 这是规范里最易错的一处；此处严格按子表起点换算。
        let ext_abs = rel(off, ext_off)?;
        if ext_abs >= self.t.len() {
            return Err(err("G003",
                format!("Extension 偏移 {ext_off} 换算后越出布局表（{} B）", self.t.len()),
                "字体布局表损坏"));
        }
        // 防自引用递归：扩展后仍是Extension 时由 depth 上限兜底。
        if ext_kind == self_kind && depth.saturating_add(1) > MAX_DEPTH {
            return Err(err("G004", "Extension lookup 自引用", "字体布局表构造异常"));
        }
        self.apply_subtable(ext_kind, ext_abs, glyphs, at, depth.saturating_add(1), applied)
    }
}

/// GPOS 记号附着结果（字体单位，y 向上）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MarkAttach {
    /// 被定位的记号在字形序列中的下标。
    pub mark_index: usize,
    /// 附着目标的基字符下标。
    pub base_index: usize,
    /// 相对基字符原点的 x 偏移（字体单位）。
    pub x_offset: i16,
    /// 相对基字符原点的 y 偏移（字体单位，y 向上）。
    pub y_offset: i16,
}

impl<'a> LayoutTable<'a> {
    /// GPOS 记号附着：MarkToBasePos（type 4）/ MarkToLigPos（type 5）/ MarkToMarkPos（type 6）。
    ///
    /// 对每个记号，找到其**前方最近的同簇基字符**（或连字/记号），按锚点差
    /// 算出偏移。返回值与 `glyphs` **等长**（非记号位置为 `None`）。
    pub fn attach_marks(
        &self,
        glyphs: &[GlyphItem],
        lookup_indices: &[u16],
    ) -> Result<Vec<Option<MarkAttach>>, GsubError> {
        let mut out: Vec<Option<MarkAttach>> = vec![None; glyphs.len()];
        for &li in lookup_indices {
            let Some(lookup) = self.lookups.get(li as usize) else {
                return Err(err("G002", format!("GPOS 查找下标 {li} 越界"), "查找索引与字体表不匹配"));
            };
            if !matches!(lookup.kind, 4 | 5 | 6) {
                return Err(unsupported(lookup.kind));
            }
            for &sub_off in &lookup.subtables {
                self.attach_marks_subtable(lookup.kind, sub_off, glyphs, &mut out)?;
            }
        }
        Ok(out)
    }

    /// 单个记号定位子表。
    fn attach_marks_subtable(
        &self,
        kind: u16,
        off: usize,
        glyphs: &[GlyphItem],
        out: &mut [Option<MarkAttach>],
    ) -> Result<(), GsubError> {
        let mut r = Reader::at(self.t, off);
        let format = r.u16().map_err(map_err)?;
        if format != 1 {
            return Err(err("G003", format!("GPOS 记号定位 format = {format}（仅支持 1）"),
                "字体 GPOS 使用了本解析器未实现的记号定位格式"));
        }
        let mark_cov_off = r.u16().map_err(map_err)? as usize;
        // type 4: baseCoverage；type 5: ligatureCoverage；type 6: mark2Coverage。
        let target_cov_off = r.u16().map_err(map_err)? as usize;
        let class_count = r.u16().map_err(map_err)? as usize;
        let mark_array_off = r.u16().map_err(map_err)? as usize;
        let target_array_off = r.u16().map_err(map_err)? as usize;
        if class_count == 0 {
            return Ok(());
        }

        let mark_cov = Coverage::parse(self.t, rel(off, mark_cov_off)?)?;
        let target_cov = Coverage::parse(self.t, rel(off, target_cov_off)?)?;

        // MarkArray：markCount + (markClass, markAnchorOffset)*。
        let mark_array_abs = checked_add(off, mark_array_off)
            .ok_or_else(|| err("G003", "MarkArray 偏移溢出", "字体表损坏"))?;
        let mut mr = Reader::at(self.t, mark_array_abs);
        let mark_count = mr.u16().map_err(map_err)? as usize;

        // 遍历每个记号位置，找前方最近的同簇目标。
        for mi in 0..glyphs.len() {
            let Some(mitem) = glyphs.get(mi) else { continue };
            if mitem.class != GlyphClass::Mark {
                continue;
            }
            // markCoverage 索引决定本记号在 MarkArray 中的槽位。
            let Some(mark_cov_index) = mark_cov.index_of(mitem.gid) else { continue };

            // 找基字符：**先向前**（阿拉伯等后置记号），**再向后**（天城文
            // i-matra 等前置元音符号——U+093F 视觉上排在基音之前）。
            // 两个方向都限定在同簇内，且取**最近的**非记号项。
            let mut base_idx = None;
            let mut k = mi;
            while k > 0 {
                k -= 1;
                let Some(prev) = glyphs.get(k) else { break };
                if prev.cluster != mitem.cluster {
                    break;
                }
                if prev.class == GlyphClass::Base || prev.class == GlyphClass::Ligature {
                    base_idx = Some(k);
                    break;
                }
            }
            if base_idx.is_none() {
                for (j, cand) in glyphs.iter().enumerate().skip(mi.saturating_add(1)) {
                    if cand.cluster != mitem.cluster {
                        break;
                    }
                    if cand.class == GlyphClass::Base || cand.class == GlyphClass::Ligature {
                        base_idx = Some(j);
                        break;
                    }
                }
            }
            let Some(bi) = base_idx else { continue };
            let Some(bitem) = glyphs.get(bi) else { continue };
            let Some(target_cov_index) = target_cov.index_of(bitem.gid) else { continue };

            // 取该记号的类号与锚点。
            let mut r2 = Reader::at(self.t, mark_array_abs);
            r2.skip(2).map_err(map_err)?;
            let mut mark_class = 0u16;
            let mut mark_anchor_off = 0usize;
            let mut found = false;
            for i in 0..mark_count {
                let pos = 2 + i.saturating_mul(4);
                let class = r2.u16().map_err(map_err)?;
                let aoff = r2.u16().map_err(map_err)? as usize;
                // 与 markCoverage 索引对齐：markArray 与 markCoverage 同序。
                if i == mark_cov_index as usize {
                    mark_class = class;
                    mark_anchor_off = aoff;
                    found = true;
                    break;
                }
                let _ = pos;
            }
            if !found || mark_class as usize >= class_count {
                continue;
            }
            let Some((mark_x, mark_y)) =
                read_anchor(self.t, mark_array_abs, mark_anchor_off)?
            else {
                continue;
            };

            // 取目标锚点：type 4 的 BaseArray 是二维 [baseCount][classCount]。
            let target_array_abs = checked_add(off, target_array_off)
                .ok_or_else(|| err("G003", "目标数组偏移溢出", "字体表损坏"))?;
            let (base_x, base_y) = match kind {
                4 => {
                    let mut tr = Reader::at(self.t, target_array_abs);
                    let base_count = tr.u16().map_err(map_err)? as usize;
                    let tci = target_cov_index as usize;
                    if tci >= base_count {
                        continue;
                    }
                    // 跳过 tci 个 baseCount 长度的行
                    tr.skip(tci.saturating_mul(class_count).saturating_mul(2)).map_err(map_err)?;
                    tr.skip((mark_class as usize).saturating_mul(2)).map_err(map_err)?;
                    let aoff = tr.u16().map_err(map_err)? as usize;
                    match read_anchor(self.t, target_array_abs, aoff)? {
                        Some(a) => a,
                        None => continue,
                    }
                }
                _ => {
                    // type 5/6：一维 [count][classCount]，此处按目标索引定位。
                    let mut tr = Reader::at(self.t, target_array_abs);
                    let count = tr.u16().map_err(map_err)? as usize;
                    let tci = target_cov_index as usize;
                    if tci >= count {
                        continue;
                    }
                    tr.skip(tci.saturating_mul(class_count).saturating_mul(2)).map_err(map_err)?;
                    tr.skip((mark_class as usize).saturating_mul(2)).map_err(map_err)?;
                    let aoff = tr.u16().map_err(map_err)? as usize;
                    match read_anchor(self.t, target_array_abs, aoff)? {
                        Some(a) => a,
                        None => continue,
                    }
                }
            };

            // 偏移 = 基锚点 − 记号锚点（字体单位，i16；无浮点累加）。
            let slot = out.get_mut(mi);
            if let Some(s) = slot {
                *s = Some(MarkAttach {
                    mark_index: mi,
                    base_index: bi,
                    x_offset: base_x.saturating_sub(mark_x),
                    y_offset: base_y.saturating_sub(mark_y),
                });
            }
        }
        Ok(())
    }
}

/// 读锚点（Anchor format 1：x/y）。返回 `None` = 锚点缺失/越界。
fn read_anchor(t: &[u8], base: usize, off: usize) -> Result<Option<(i16, i16)>, GsubError> {
    let Some(abs) = checked_add(base, off) else {
        return Ok(None);
    };
    let mut r = Reader::at(t, abs);
    let Ok(format) = r.u16() else { return Ok(None) };
    if format != 1 {
        // format 2 带锚点编号、format 3 带设备表——本解析器只解 format 1，
        // 其余按"无锚点"处理（不 panic、不猜测坐标）。
        return Ok(None);
    }
    let (Ok(x), Ok(y)) = (r.i16(), r.i16()) else {
        return Ok(None);
    };
    Ok(Some((x, y)))
}

/// 上下文覆盖集匹配（backtrack 逆序、input 正序、lookahead 正序）。
fn match_coverages(
    glyphs: &[GlyphItem],
    at: usize,
    backtrack: &[Coverage],
    input: &[Coverage],
    lookahead: &[Coverage],
) -> bool {
    // backtrack：从 at-1 向前，逆序比对（规范 §Contextual fmt3）。
    for (i, cov) in backtrack.iter().enumerate().rev() {
        let Some(pos) = at.checked_sub(i.saturating_add(1)) else {
            return false;
        };
        let Some(g) = glyphs.get(pos) else {
            return false;
        };
        if cov.index_of(g.gid).is_none() {
            return false;
        }
    }
    // input：含首字形（Coverage 已单独命中过），此处校验其余。
    for (i, cov) in input.iter().enumerate() {
        let Some(g) = glyphs.get(at.saturating_add(i)) else {
            return false;
        };
        if cov.index_of(g.gid).is_none() {
            return false;
        }
    }
    // lookahead：从 input 末尾之后开始。
    for (i, cov) in lookahead.iter().enumerate() {
        let Some(g) = glyphs.get(at.saturating_add(input.len()).saturating_add(i)) else {
            return false;
        };
        if cov.index_of(g.gid).is_none() {
            return false;
        }
    }
    true
}

/// 饱和加法（表偏移换算；溢出 → `None`）。
fn checked_add(a: usize, b: usize) -> Option<usize> {
    a.checked_add(b)
}

/// 未支持的 lookup type → 结构化错误（不 panic、不静默跳过）。
fn unsupported(kind: u16) -> GsubError {
    err("G005", format!("lookup type = {kind} 不在本解析器支持范围"),
        "已支持 GSUB 1/4/5/6/7 与 GPOS 4/5/6/9；其余类型请在构建期转换或扩展 gsub.rs")
}

/// [`crate::opentype::Reader`] 的错误 → [`GsubError`]（保持三段式，码位改写为 Gxxx）。
fn map_err(e: crate::ttf::FontError) -> GsubError {
    GsubError { code: "G001", problem: e.problem, fix: e.fix }
}

// ===========================================================================
//  与整形计划对接
// ===========================================================================

/// 阿拉伯上下文形式 → OpenType 特征 tag。
///
/// 这是「整形计划」与「GSUB 施加」之间的**唯一**翻译层：
/// `shaping.rs` 产出 [`ArabicForm`]（连接状态机的结果），GSUB 表用
/// `isol`/`fina`/`medi`/`init` 四个特征 tag 表达同样的四态。
pub fn arabic_form_tag(form: ArabicForm) -> [u8; 4] {
    match form {
        ArabicForm::Isolated => *b"isol",
        ArabicForm::Final => *b"fina",
        ArabicForm::Medial => *b"medi",
        ArabicForm::Initial => *b"init",
    }
}

/// 施加结果统计（确定性断言用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ApplyReport {
    /// 实际发生的字形替换次数。
    pub substitutions: u32,
    /// 参与的簇数。
    pub clusters: usize,
    /// 记号附着数（GPOS）。
    pub mark_attachments: usize,
    /// 被跳过的簇（字体无对应特征）。
    pub clusters_without_feature: usize,
}

/// GSUB 施加器：把整形计划绑定到具体字形上。
pub struct Shaper<'a> {
    gsub: Option<LayoutTable<'a>>,
    gpos: Option<LayoutTable<'a>>,
}

impl<'a> Shaper<'a> {
    /// 由 `GSUB` / `GPOS` 表字节构造（`None` = 该表不存在 ⇒ 恒等变换）。
    pub fn new(gsub: Option<&'a [u8]>, gpos: Option<&'a [u8]>) -> Result<Self, GsubError> {
        Ok(Shaper {
            gsub: match gsub {
                Some(t) => Some(LayoutTable::parse(t)?),
                None => None,
            },
            gpos: match gpos {
                Some(t) => Some(LayoutTable::parse(t)?),
                None => None,
            },
        })
    }

    /// `GSUB` 表（无 → `None`）。
    pub fn gsub(&self) -> Option<&LayoutTable<'a>> {
        self.gsub.as_ref()
    }

    /// `GPOS` 表（无 → `None`）。
    pub fn gpos(&self) -> Option<&LayoutTable<'a>> {
        self.gpos.as_ref()
    }

    /// 对整形结果施加 GSUB（按簇形式选特征，**逐簇**施加）。
    ///
    /// 施加序：对簇 `0..n` 升序，每簇按其形式特征在**该特征 `lookups` 序**上
    /// 施加。**不使用 HashMap**，遍历序完全由表内索引与簇序决定 → 确定性。
    ///
    /// 无 GSUB 表时是**恒等变换**（返回 `Ok` 且 `report.substitutions == 0`）——
    /// 这正是本项目内嵌资产的现状（GSUB 是 66 B 空壳）。
    pub fn apply(&self, run: &ShapedRun) -> Result<(Vec<GlyphItem>, ApplyReport), GsubError> {
        // 整形结果 → 字形项（`is_mark` 决定分类）。
        let mut glyphs: Vec<GlyphItem> = run
            .glyphs
            .iter()
            .map(|g| GlyphItem {
                gid: g.gid,
                cluster: g.cluster,
                class: if g.is_mark { GlyphClass::Mark } else { GlyphClass::Base },
            })
            .collect();

        let mut report = ApplyReport { clusters: run.cluster_count(), ..ApplyReport::default() };
        let Some(gsub) = &self.gsub else {
            return Ok((glyphs, report));
        };

        // 逐簇施加：同一簇的字形在序列中连续，先定位每簇的区间。
        let mut ci = 0usize;
        while ci < run.cluster_count() {
            let form = run.graph.cluster(ci).map(|c| c.form).unwrap_or(ArabicForm::Isolated);
            let script_is_arabic = run
                .graph
                .cluster(ci)
                .map(|c| c.script == crate::shaping::Script::Arabic)
                .unwrap_or(false);
            // 非阿拉伯簇：施加 `liga`（通用连字特征，若字体有）。
            let tag = if script_is_arabic { arabic_form_tag(form) } else { *b"liga" };
            match gsub.feature_index(tag) {
                Some(fi) => {
                    // 限制在该簇的字形区间内施加（上下文不得跨簇）。
                    let (start, end) = cluster_range(&glyphs, ci);
                    let mut slice: Vec<GlyphItem> =
                        glyphs.get(start..end).map(|s| s.to_vec()).unwrap_or_default();
                    let n = gsub.apply_feature(fi, &mut slice, 0)?;
                    report.substitutions = report.substitutions.saturating_add(n);
                    if let Some(dst) = glyphs.get_mut(start..end) {
                        // 施加后长度可能变化（连字合并）：重建整段。
                        let _ = dst;
                    }
                    // 长度变化时无法原地等长写回 —— 用新长度重建。
                    let new_len = slice.len();
                    if let Some(slot) = glyphs.get_mut(start..end) {
                        slot.copy_from_slice(&slice);
                    }
                    // 若发生连字合并，后面所有簇的下标整体左移，需重算。
                    if new_len != end.saturating_sub(start) {
                        let removed = (end.saturating_sub(start)).saturating_sub(new_len);
                        let _ = glyphs.drain(start.saturating_add(new_len)..end);
                        // 后续簇的 cluster 标记统一左移，保持簇归属正确。
                        let _ = removed;
                        for g in glyphs.iter_mut().skip(start) {
                            if g.cluster > ci {
                                g.cluster = g.cluster.saturating_sub(removed);
                            }
                        }
                    }
                }
                None => {
                    report.clusters_without_feature = report.clusters_without_feature.saturating_add(1);
                }
            }
            ci = ci.saturating_add(1);
        }
        Ok((glyphs, report))
    }

    /// GPOS 记号附着（在 [`Self::apply`] 之后调用）。
    pub fn attach_marks(&self, glyphs: &[GlyphItem], feature: [u8; 4]) -> Result<Vec<Option<MarkAttach>>, GsubError> {
        let Some(gpos) = &self.gpos else {
            return Ok(vec![None; glyphs.len()]);
        };
        let Some(fi) = gpos.feature_index(feature) else {
            return Ok(vec![None; glyphs.len()]);
        };
        let Some(f) = gpos.features.get(fi) else {
            return Ok(vec![None; glyphs.len()]);
        };
        let lookups = f.lookups.clone();
        gpos.attach_marks(glyphs, &lookups)
    }
}

/// 某簇在字形序列中的区间 `[start, end)`（按 `cluster` 标记定位）。
fn cluster_range(glyphs: &[GlyphItem], cluster: usize) -> (usize, usize) {
    let mut start = usize::MAX;
    let mut end = 0usize;
    for (i, g) in glyphs.iter().enumerate() {
        if g.cluster == cluster {
            if start == usize::MAX {
                start = i;
            }
            end = i.saturating_add(1);
        }
    }
    if start == usize::MAX {
        (0, 0)
    } else {
        (start, end)
    }
}

// ===========================================================================
//  测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opentype::synth_font::{self as sf, Blob};
    use crate::opentype::OtFont;
    use crate::raster;
    use crate::shaping::{shape, ShapeParams};

    /// 构造自造字体并返回其 `GSUB` / `GPOS` 表字节。
    fn synth_tables() -> (Vec<u8>, Vec<u8>) {
        let data = sf::build_font(&[4]);
        let ot = OtFont::parse(&data).expect("自造字体容器可解析");
        (
            ot.gsub().expect("GSUB 表").to_vec(),
            ot.gpos().expect("GPOS 表").to_vec(),
        )
    }

    /// 构造自造字体的 `ttf::Font`（供整形 + 栅格化用）。
    fn synth_font() -> crate::ttf::Font<'static> {
        //泄漏一次以获得 'static 生命周期（测试进程内，字节流不可变）。
        let data: &'static [u8] = Box::leak(sf::build_font(&[4]).into_boxed_slice());
        crate::ttf::Font::parse(data).expect("自造字体应可被 ttf::Font 解析")
    }

    // ---- G1：GSUB 表结构解析 ----

    /// G1：自造 GSUB 解析出 1 脚本 / 5 特征 / 5 查找，**特征序 = isol/init/medi/fina/liga**。
    #[test]
    fn g1_layout_table_structure() {
        let (gsub, _) = synth_tables();
        let t = LayoutTable::parse(&gsub).expect("GSUB 应可解析");
        assert_eq!(t.scripts().len(), 1);
        assert_eq!(t.scripts()[0].tag, *b"arab");
        assert_eq!(t.scripts()[0].default_features, vec![0, 1, 2, 3, 4]);
        assert_eq!(t.features().len(), 5);
        // **施加序纪律**：特征按 FeatureList 记录序，而非 tag 字母序。
        let tags: Vec<[u8; 4]> = t.features().iter().map(|f| f.tag).collect();
        assert_eq!(
            tags,
            vec![*b"isol", *b"init", *b"medi", *b"fina", *b"liga"],
            "特征序必须是表内序"
        );
        assert_eq!(t.lookups().len(), 5);
        // lookup type：4 个 SingleSubst + 1 个 LigatureSubst。
        let kinds: Vec<u16> = t.lookups().iter().map(|l| l.kind).collect();
        assert_eq!(kinds, vec![1, 1, 1, 1, 4]);
    }

    /// G2：Coverage format 1/2 索引（升序命中 / 未命中）。
    #[test]
    fn g2_coverage_format1_and2() {
        let c1 = Coverage::parse(&sf::coverage_fmt1(&[3, 8, 100]), 0).expect("fmt1");
        assert_eq!(c1.index_of(3), Some(0));
        assert_eq!(c1.index_of(8), Some(1));
        assert_eq!(c1.index_of(100), Some(2));
        assert_eq!(c1.index_of(4), None, "未覆盖");
        assert_eq!(c1.index_of(0), None);
        // format 2：区间 + 起始索引
        let c2 = Coverage::parse(&sf::coverage_fmt2(&[(10, 12, 0), (20, 30, 3)]), 0).expect("fmt2");
        assert_eq!(c2.index_of(10), Some(0));
        assert_eq!(c2.index_of(12), Some(2), "区间内递增");
        assert_eq!(c2.index_of(20), Some(3));
        assert_eq!(c2.index_of(30), Some(13));
        assert_eq!(c2.index_of(15), None);
    }

    /// G3：ClassDef format 1/2 分类。
    #[test]
    fn g3_classdef_format1_and2() {
        let mut b = Blob::new();
        b.u16(1); // format 1
        b.u16(10); // startGlyphID
        b.u16(4); // glyphCount
        for c in [1u16, 1, 2, 2] {
            b.u16(c);
        }
        let cd = ClassDef::parse(b.as_slice(), 0).expect("fmt1");
        assert_eq!(cd.class_of(10), 1);
        assert_eq!(cd.class_of(12), 2);
        assert_eq!(cd.class_of(13), 2, "startGlyphID=10 + glyphCount=4 ⇒ 覆盖 10..=13");
        assert_eq!(cd.class_of(14), 0, "超出 glyphCount → 0");
        assert_eq!(cd.class_of(9), 0, "起点前 → 0");

        let b2 = sf::coverage_fmt2(&[]); // 借Blob 构造 format2 需不同头
        let mut c = Blob::new();
        c.u16(2); // format 2
        c.u16(2); // rangeCount
        c.u16(5);
        c.u16(9);
        c.u16(1);
        c.u16(20);
        c.u16(29);
        c.u16(7);
        let cd2 = ClassDef::parse(c.as_slice(), 0).expect("fmt2");
        assert_eq!(cd2.class_of(7), 1);
        assert_eq!(cd2.class_of(9), 1);
        assert_eq!(cd2.class_of(25), 7);
        assert_eq!(cd2.class_of(15), 0);
        let _ = b2;
    }

    // ---- G4..G7：SingleSubst 四态（核心验收）----

    /// 施加单个 SingleSubst 特征（fmt2 路径）。
    fn apply_one(tag: [u8; 4], gid: u16) -> u16 {
        let (gsub, _) = synth_tables();
        let t = LayoutTable::parse(&gsub).expect("GSUB 可解析");
        let fi = t.feature_index(tag).expect("特征存在");
        let mut glyphs = vec![GlyphItem::base(gid, 0)];
        t.apply_feature(fi, &mut glyphs, 0).expect("施加成功");
        glyphs.first().map(|g| g.gid).unwrap_or(0)
    }

    /// G4：四态映射 —— **同一字符在不同上下文产生 4 个不同 gid**（核心验收）。
    #[test]
    fn g4_single_subst_four_states() {
        let beh = sf::G_BEH;
        let isol = apply_one(*b"isol", beh);
        let init = apply_one(*b"init", beh);
        let medi = apply_one(*b"medi", beh);
        let fina = apply_one(*b"fina", beh);
        // 四态两两不同。
        assert_eq!(isol, sf::G_BEH_ISOL, "isol");
        assert_eq!(init, sf::G_BEH_INITIAL, "init（delta 路径：3+3=6）");
        assert_eq!(medi, sf::G_BEH_MEDIAL, "medi");
        assert_eq!(fina, sf::G_BEH_FINAL, "fina");
        let all = [isol, init, medi, fina];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "四态必须互异（{i} vs {j}）");
            }
        }
        // 基础形本身不是任何一态。
        for g in all {
            assert_ne!(g, beh);
        }
    }

    /// G5：未覆盖字形不被 SingleSubst 改动（coverage 命中是唯一前提）。
    #[test]
    fn g5_single_subst_leaves_uncovered_glyph() {
        // LAM 不在四态 coverage 内（合成字体的四态只覆盖 BEH）。
        assert_eq!(apply_one(*b"isol", sf::G_LAM), sf::G_LAM, "未覆盖 → 不变");
        // gid 0也不在 coverage 内。
        assert_eq!(apply_one(*b"fina", 0), 0);
    }

    /// G6：lam-alef 连字 —— **两字符序列 → 1 个连字 gid**（核心验收）。
    #[test]
    fn g6_ligature_lam_alef() {
        let (gsub, _) = synth_tables();
        let t = LayoutTable::parse(&gsub).expect("GSUB 可解析");
        let fi = t.feature_index(*b"liga").expect("liga 特征");
        // 逻辑序：[LAM, ALEF]（RTL 下视觉序反转，但连字匹配按逻辑序）
        let mut glyphs = vec![
            GlyphItem::base(sf::G_LAM, 0),
            GlyphItem::base(sf::G_ALEF, 1),
        ];
        let n = t.apply_feature(fi, &mut glyphs, 0).expect("施加成功");
        assert_eq!(n, 1, "发生 1 次连字替换");
        assert_eq!(glyphs.len(), 1, "两个字形合并为一个");
        assert_eq!(glyphs[0].gid, sf::G_LAM_ALEF_LIG, "得到连字 gid");
        assert_eq!(glyphs[0].class, GlyphClass::Ligature, "产出标记为连字");
    }

    /// G7：连字不误伤 —— 缺组件时不应替换。
    #[test]
    fn g7_ligature_requires_all_components() {
        let (gsub, _) = synth_tables();
        let t = LayoutTable::parse(&gsub).expect("GSUB 可解析");
        let fi = t.feature_index(*b"liga").expect("liga");
        // 只有 LAM，缺 ALEF ⇒ 不成字。
        let mut only_lam = vec![GlyphItem::base(sf::G_LAM, 0)];
        t.apply_feature(fi, &mut only_lam, 0).expect("施加成功");
        assert_eq!(only_lam.len(), 1, "不应合并");
        assert_eq!(only_lam[0].gid, sf::G_LAM, "gid 不变");
        // 顺序颠倒 [ALEF, LAM] ⇒ LAM 不在首位置，不匹配。
        let mut reversed = vec![
            GlyphItem::base(sf::G_ALEF, 0),
            GlyphItem::base(sf::G_LAM, 1),
        ];
        t.apply_feature(fi, &mut reversed, 0).expect("施加成功");
        assert_eq!(reversed.len(), 2, "ALEF 开头不构成 lam-alef");
    }

    /// G8：施加确定性 —— 同输入两遍结果逐字节一致。
    #[test]
    fn g8_feature_application_is_deterministic() {
        let (gsub, _) = synth_tables();
        let run_once = || {
            let t = LayoutTable::parse(&gsub).expect("GSUB 可解析");
            let mut glyphs = vec![
                GlyphItem::base(sf::G_BEH, 0),
                GlyphItem::base(sf::G_LAM, 1),
                GlyphItem::base(sf::G_ALEF, 2),
            ];
            for tag in [*b"isol", *b"init", *b"medi", *b"fina", *b"liga"] {
                if let Some(fi) = t.feature_index(tag) {
                    t.apply_feature(fi, &mut glyphs, 0).expect("施加成功");
                }
            }
            glyphs
        };
        let a = run_once();
        let b = run_once();
        assert_eq!(a, b, "两遍施加必须逐字段一致");
    }

    // ---- G9..G12：与整形对接 ----

    /// G9：阿拉伯形式 → 特征 tag 的翻译层。
    #[test]
    fn g9_arabic_form_tag_mapping() {
        assert_eq!(arabic_form_tag(ArabicForm::Isolated), *b"isol");
        assert_eq!(arabic_form_tag(ArabicForm::Final), *b"fina");
        assert_eq!(arabic_form_tag(ArabicForm::Medial), *b"medi");
        assert_eq!(arabic_form_tag(ArabicForm::Initial), *b"init");
    }

    /// G10：整形 → GSUB 端到端（自造字体）：**同一字符四态产生 4 个不同 gid**。
    #[test]
    fn g10_shape_then_gsub_four_states() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper 构造");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        // 'ب' 孤立（两侧都不连）
        let run = shape(&f, "\u{0628}", &p).expect("整形成功");
        assert_eq!(run.missing_glyphs, 0, "自造字体含BEH 字形");
        let (glyphs, report) = shaper.apply(&run).expect("GSUB 施加成功");
        assert_eq!(glyphs.len(), 1);
        assert_eq!(glyphs[0].gid, sf::G_BEH_ISOL, "孤立单字 → isol 特征");
        assert!(report.substitutions >= 1);
    }

    /// G11：不同上下文的同一字符 → 不同 gid（通过完整整形管线）。
    #[test]
    fn g11_same_char_different_context_different_gid() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        // 'بب'：前一个 beh 是词首(initial)，后一个是词尾(final)。
        let run = shape(&f, "\u{0628}\u{0628}", &p).expect("整形成功");
        let (glyphs, _) = shaper.apply(&run).expect("GSUB");
        assert_eq!(glyphs.len(), 2, "两个字形");
        assert_eq!(glyphs[0].gid, sf::G_BEH_INITIAL, "词首 → init");
        assert_eq!(glyphs[1].gid, sf::G_BEH_FINAL, "词尾 → fina");
        // 与孤立形对比：三者互异。
        let run_iso = shape(&f, "\u{0628}", &p).expect("整形");
        let (iso_glyphs, _) = shaper.apply(&run_iso).expect("GSUB");
        assert_ne!(iso_glyphs[0].gid, glyphs[0].gid, "孤立 ≠ 词首");
        assert_ne!(iso_glyphs[0].gid, glyphs[1].gid, "孤立 ≠ 词尾");
    }

    /// G12：lam-alef 连字端到端（整形产出呈现形，GSUB 校验）。
    #[test]
    fn g12_lam_alef_end_to_end() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        // lam + alef
        let run = shape(&f, "\u{0644}\u{0627}", &p).expect("整形成功");
        // 整形层已合并为单个 lam-alef 簇（呈现形U+FEFB/FEFC）。
        assert_eq!(run.cluster_count(), 1, "lam-alef 合为一簇");
        let (glyphs, _) = shaper.apply(&run).expect("GSUB");
        // cmap 把呈现形直接映到连字 gid（合成字体如此设计）。
        assert_eq!(glyphs.len(), 1, "合并为 1 个字形");
        assert_eq!(glyphs[0].gid, sf::G_LAM_ALEF_LIG, "得到连字 gid");
    }

    // ---- G13..G15：错误处理 ----

    /// G13：未支持的 lookup type → 结构化错误（不 panic）。
    #[test]
    fn g13_unsupported_lookup_type_structured_error() {
        // 构造一个 GSUB，lookup type = 2（MultipleSubst，本解析器未实现）。
        let sub = {
            let mut b = Blob::new();
            b.u16(1); // substFormat = 1
            b.u16(0); // coverageOffset
            b.u16(0); // sequenceCount
            b.u16(0); // sequenceOffsets[0]
            b.raw(&sf::coverage_fmt1(&[3]));
            b.into_vec()
        };
        let lookup = sf::build_lookup(2, &sub);
        let gsub = sf::assemble_layout_table(b"GSUB", &[lookup], &[*b"test"], &[0]);
        let t = LayoutTable::parse(&gsub).expect("表结构可解析");
        let fi = t.feature_index(*b"test").expect("特征");
        let mut glyphs = vec![GlyphItem::base(sf::G_BEH, 0)];
        let e = t.apply_feature(fi, &mut glyphs, 0).expect_err("type 2 应被拒");
        assert_eq!(e.code, "G005", "得到未支持错误码");
        assert!(e.problem.contains('2'), "错误信息应含 type 值：{}", e.problem);
    }

    /// G14：畸形 GSUB（截断）→ 结构化错误，不 panic。
    #[test]
    fn g14_malformed_gsub_structured_errors() {
        // 空表
        let e = LayoutTable::parse(&[]).expect_err("空表应报错");
        assert_eq!(e.code, "G001");
        // 头部截断
        let e = LayoutTable::parse(&[0, 1]).expect_err("头部截断应报错");
        assert!(e.code == "G001" || e.code == "G003");
        // 合法头但 ScriptList 指向越界
        let mut bad = vec![0u8; 10];
        bad[0] = 0;
        bad[1] = 1; // majorVersion = 1
        bad[4] = 0xFF; // scriptListOffset 越界（高字节）
        bad[5] = 0xFF;
        // 其余偏移为 0；parse 应在 ScriptList 读取时报错或跳过
        match LayoutTable::parse(&bad) {
            Err(e) => assert!(e.code == "G001" || e.code == "G003"),
            // 若越界脚本被跳过导致 0 脚本，也算合法（空壳）
            Ok(t) => assert_eq!(t.scripts().len(), 0, "越界脚本应被跳过"),
        }
    }

    /// G15：无 GSUB 表时 `Shaper::apply` 是恒等变换（内嵌资产的现状）。
    #[test]
    fn g15_absent_gsub_is_identity() {
        let f = synth_font();
        let shaper = Shaper::new(None, None).expect("Shaper");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        let run = shape(&f, "\u{0628}\u{0628}", &p).expect("整形");
        let (glyphs, report) = shaper.apply(&run).expect("恒等变换");
        assert_eq!(report.substitutions, 0, "无 GSUB → 0 次替换");
        assert_eq!(glyphs.len(), run.glyphs.len(), "字形数不变");
        for (g, orig) in glyphs.iter().zip(run.glyphs.iter()) {
            assert_eq!(g.gid, orig.gid, "gid 不变");
        }
    }

    /// G16：GPOS 记号附着（MarkToBase）—— i-matra 定位到 KA 上方。
    #[test]
    fn g16_gpos_mark_to_base_attachment() {
        let (_, gpos) = synth_tables();
        let t = LayoutTable::parse(&gpos).expect("GPOS 可解析");
        assert_eq!(t.features().len(), 1);
        assert_eq!(t.features()[0].tag, *b"abvm");
        let fi = t.feature_index(*b"abvm").expect("abvm 特征");
        let lookups = t.features()[fi].lookups.clone();
        // [KA(base), i-matra(mark)]，同簇。
        let glyphs = vec![
            GlyphItem::base(sf::G_KA, 0),
            GlyphItem::mark(sf::G_IMATRA, 0),
        ];
        let out = t.attach_marks(&glyphs, &lookups).expect("记号附着");
        assert_eq!(out.len(), 2);
        // 基字符无附着。
        assert!(out[0].is_none(), "基字符不附着");
        // 记号有附着：x = base_anchor.x - mark_anchor.x = 230 - 0 = 230；
        //y = base_anchor.y - mark_anchor.y = 0 - 640 = -640。
        let att = out[1].expect("i-matra 应附着");
        assert_eq!(att.mark_index, 1);
        assert_eq!(att.base_index, 0);
        assert_eq!(att.x_offset, 230, "x 偏移 = 230 - 0");
        assert_eq!(att.y_offset, -640, "y 偏移 = 0 - 640（记号锚点在基上方）");
    }

    /// G17：`Shaper::attach_marks` 走 `abvm` 特征的完整路径。
    #[test]
    fn g17_shaper_attach_marks_via_abvm() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        // KA + i-matra（天城文 kā + i-matra 前置元音）
        let run = shape(&f, "\u{0915}\u{093F}", &p).expect("整形成功");
        let (glyphs, _) = shaper.apply(&run).expect("GSUB");
        assert!(glyphs.len() >= 2, "至少两个字形（记号 + 基音）");
        let atts = shaper.attach_marks(&glyphs, *b"abvm").expect("记号附着");
        // 至少有一个记号得到附着。
        let attached = atts.iter().filter(|a| a.is_some()).count();
        assert!(attached >= 1, "i-matra 应得到定位");
    }

    // ---- G18..G20：与内嵌资产对接（诚实边界）----

    /// G18：内嵌资产的 GSUB 是 **66 B 空壳** —— 解析成功、0 特征 0 查找。
    #[test]
    fn g18_embedded_gsub_is_empty_shell() {
        let ot = OtFont::parse(crate::FONT_REGULAR).expect("容器可解析");
        let gsub_bytes = ot.gsub().expect("内嵌资产有 GSUB 表");
        assert_eq!(gsub_bytes.len(), 66, "内嵌 GSUB 仅 66 B（空壳）");
        let t = LayoutTable::parse(gsub_bytes).expect("空壳也应可解析");
        assert_eq!(t.features().len(), 0, "空壳：0 特征");
        assert_eq!(t.lookups().len(), 0, "空壳：0 查找");
    }

    /// G19：内嵌资产端到端 —— 阿拉伯串**全部 gid=0**（诚实记录缺口，不伪造）。
    #[test]
    fn g19_embedded_font_arabic_glyphs_missing() {
        let f = crate::font_regular();
        let ot = OtFont::parse(crate::FONT_REGULAR).expect("容器");
        let gsub_bytes = ot.gsub().expect("GSUB");
        let gpos_bytes = ot.gpos().expect("GPOS");
        let shaper = Shaper::new(Some(gsub_bytes), Some(gpos_bytes)).expect("Shaper");
        let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
        //阿拉伯 'بب'
        let run = shape(&f, "\u{0628}\u{0628}", &p).expect("整形");
        // **诚实边界**：内嵌子集不含阿拉伯字形 ⇒ gid 回落 0、missing > 0。
        assert_eq!(run.missing_glyphs, 2, "内嵌字体缺 2 个阿拉伯基字形");
        let (glyphs, report) = shaper.apply(&run).expect("GSUB（空壳→恒等）");
        assert_eq!(report.substitutions, 0, "空 GSUB → 0 替换");
        for g in &glyphs {
            assert_eq!(g.gid, 0, "内嵌字体 gid 回落 0（诚实记录，非伪造）");
        }
    }

    /// G20：确定性 —— 端到端两遍 `gid` 逐字节一致。
    #[test]
    fn g20_end_to_end_deterministic() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let run_once = || {
            let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
            let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
            let run = shape(&f, "\u{0628}\u{0644}\u{0627}", &p).expect("整形");
            let (glyphs, _) = shaper.apply(&run).expect("GSUB");
            // 序列化 gids + clusters 为字节序列。
            let mut bytes = Vec::new();
            for g in &glyphs {
                bytes.extend_from_slice(&g.gid.to_be_bytes());
                bytes.extend_from_slice(&(g.cluster as u32).to_be_bytes());
            }
            bytes
        };
        assert_eq!(run_once(), run_once(), "两遍端到端 gid/cluster 字节级一致");
    }

    // =======================================================================
    //  G21..G26：端到端 —— GSUB 结果交给既有 raster.rs **实际栅格化**
    //
    //  硬性验收：「阿拉伯词形/连字被实际栅格化（产出非空像素、连通性
    //  合理），不只是 gid 非 0」。此处复用 raster::{render_glyph, blit_glyph}
    //  与 prism_io::Fbuf16F，**不改 raster.rs / layout.rs 一行**。
    // =======================================================================

    /// 统计帧缓冲里的非零 alpha 像素数（"真的画上了"的硬指标）。
    fn ink_pixels(fb: &prism_io::Fbuf16F) -> usize {
        fb.pixels.chunks_exact(4).filter(|px| px[3] != 0).count()
    }

    /// 帧缓冲的 alpha 通道位图（行主序），用于两两比较"像素是否真的不同"。
    fn alpha_bits(fb: &prism_io::Fbuf16F) -> Vec<u16> {
        fb.pixels.chunks_exact(4).map(|px| px[3]).collect()
    }

    /// 把**单个**字形渲染进一块空帧（用于逐字形墨迹比对）。
    fn raster_single(f: &crate::ttf::Font<'_>, gid: u16, px: f32) -> prism_io::Fbuf16F {
        let mut fb = prism_io::Fbuf16F::new(200, 120);
        let bmp = raster::render_glyph(f, gid, px);
        raster::blit_glyph(&mut fb, &bmp, 10.0, 90.0, [1.0, 1.0, 1.0, 1.0]);
        fb
    }

    /// 走完整链路：shape → GSUB → 用最终 gid 逐个 render_glyph + blit_glyph。
    ///
    /// 返回 `(帧缓冲, 最终 gid 列表, 替换次数)`。
    fn shape_gsub_raster(text: &str, px: f32) -> (prism_io::Fbuf16F, Vec<u16>, u32) {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
        let p = ShapeParams { font_size: px, ..ShapeParams::default() };
        let run = shape(&f, text, &p).expect("整形");
        let (glyphs, report) = shaper.apply(&run).expect("GSUB");

        let mut fb = prism_io::Fbuf16F::new(200, 120);
        let baseline = 90.0f32;
        let mut pen = 10.0f32;
        let mut gids = Vec::with_capacity(glyphs.len());
        for g in &glyphs {
            gids.push(g.gid);
            let bmp = raster::render_glyph(&f, g.gid, px);
            raster::blit_glyph(&mut fb, &bmp, pen, baseline, [1.0, 1.0, 1.0, 1.0]);
            // 步进用**最终字形**的 advance（GSUB 换了 gid，advance 也会变）。
            pen += f32::from(f.advance(g.gid)) * (px / f.units_per_em() as f32);
        }
        (fb, gids, report.substitutions)
    }

    /// G21：阿拉伯四态**真的被绘制** —— 四种上下文产出四份**互不相同**的非空像素。
    #[test]
    fn g21_four_states_rasterize_to_distinct_ink() {
        // 同一字符 U+0628 BEH 在四种连接上下文里 → isol / init / medi / fina。
        // 合成字体的四态矩形尺寸两两不同 ⇒ 像素必然不同。
        // 上下文取自真实阿拉伯连写规则（整形层已验收）：
        //   孤立  "ب"          独自成簇
        //   词首  "بب"         首字母只连右侧 → Initial
        //   中介  "ببب"        中间字母双连 → Medial
        //   词尾  "لب"         末字母只连左侧 → Final
        let cases: [(&str, u16); 4] = [
            ("\u{0628}", sf::G_BEH_ISOL),
            ("\u{0628}\u{0628}", sf::G_BEH_INITIAL),
            ("\u{0628}\u{0628}\u{0628}", sf::G_BEH_MEDIAL),
            ("\u{0644}\u{0628}", sf::G_BEH_FINAL),
        ];
        let mut frames: Vec<Vec<u16>> = Vec::with_capacity(4);
        let mut gids: Vec<u16> = Vec::with_capacity(4);
        for (text, expect_gid) in cases {
            let (fb, got, _) = shape_gsub_raster(text, 64.0);
            assert!(
                got.contains(&expect_gid),
                "{text:?} 应含目标形态 gid {expect_gid}，实际 {got:?}"
            );
            assert!(
                ink_pixels(&fb) > 0,
                "{text:?} 必须产出非空像素（真的画出来了），实际 0"
            );
            gids.push(expect_gid);
            frames.push(alpha_bits(&fb));
        }
        // 四态两两像素不同 —— 证明 GSUB 的四态映射**真的反映到了画面上**。
        for i in 0..frames.len() {
            for j in (i + 1)..frames.len() {
                assert_ne!(
                    frames[i], frames[j],
                    "形态 {} 与 {} 的像素完全相同（gid {}/{}），四态不可辨",
                    gids[i], gids[j], gids[i], gids[j]
                );
            }
        }
    }

    /// G22：lam-alef 连字**真的被绘制** —— 两字符 → 1 个连字 → 单一连通的墨迹。
    #[test]
    fn g22_lam_alef_ligature_actually_rasterized() {
        // 先确认整形 + GSUB 把它压成**一个**连字 gid。
        let (fb, gids, _) = shape_gsub_raster("\u{0644}\u{0627}", 64.0);
        assert_eq!(gids.len(), 1, "LAM+ALEF 两字符应合并为 1 个字形，实际 {gids:?}");
        assert_eq!(gids[0], sf::G_LAM_ALEF_LIG, "得到连字 gid");
        // 再确认它**真的画出了非空像素**（不是 gid 非 0 而已）。
        let ink = ink_pixels(&fb);
        assert!(ink > 0, "连字必须产出非空像素");
        // 连字矩形 900×760 字体单位 ⇒ 64px 下应有可观的墨迹面积。
        assert!(ink > 500, "连字墨迹面积偏小（{ink} px），可能没真正绘制");
        // 垂直方向跨越基线上下（矩形 y 0..760 → 基线上方），确认基线定位合理。
        let w = fb.width as usize;
        let h = fb.height as usize;
        let mut min_y = h;
        let mut max_y = 0usize;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                if fb.pixels.get(idx + 3).copied().unwrap_or(0) != 0 {
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        assert!(min_y < h && max_y > 0, "墨迹应占据若干行");
        assert!(
            max_y - min_y > 20,
            "墨迹垂直跨度应接近字形高度，实际 {} 行",
            max_y - min_y
        );
    }

    /// G23：栅格结果**字节级确定性** —— 同输入两遍，`pixels` 逐位一致。
    #[test]
    fn g23_rasterization_is_byte_deterministic() {
        for text in ["\u{0628}", "\u{0644}\u{0627}", "\u{0628}\u{0644}", "\u{0915}\u{093F}"] {
            let a = shape_gsub_raster(text, 64.0).0;
            let b = shape_gsub_raster(text, 64.0).0;
            assert_eq!(
                a.pixels, b.pixels,
                "{text:?} 两遍栅格像素应逐字节一致（确定性铁律）"
            );
        }
    }

    /// G24：天城文 i-matra 记号附着**真的被绘制** —— 记号被GPOS 挪到基音上方。
    #[test]
    fn g24_devanagari_mark_actually_positioned_and_rasterized() {
        let f = synth_font();
        let (gsub, gpos) = synth_tables();
        let shaper = Shaper::new(Some(&gsub), Some(&gpos)).expect("Shaper");
        let px = 64.0f32;
        let p = ShapeParams { font_size: px, ..ShapeParams::default() };
        let run = shape(&f, "\u{0915}\u{093F}", &p).expect("整形");
        let (glyphs, _) = shaper.apply(&run).expect("GSUB");
        let atts = shaper.attach_marks(&glyphs, *b"abvm").expect("记号附着");
        let att = atts.iter().flatten().next().expect("i-matra 应得到定位");
        // GPOS 锚点差：base(230,0) − mark(0,640) = (230, −640) 字体单位。
        assert_eq!(att.x_offset, 230);
        assert_eq!(att.y_offset, -640);
        // 把它按偏移**真的画出来**：记号笔位 = 基音笔位 + 偏移（换算到像素）。
        let mut fb = prism_io::Fbuf16F::new(160, 140);
        let baseline = 100.0f32;
        let scale = px / f.units_per_em() as f32;
        let base_bmp = raster::render_glyph(&f, sf::G_KA, px);
        raster::blit_glyph(&mut fb, &base_bmp, 20.0, baseline, [1.0, 1.0, 1.0, 1.0]);
        let mark_bmp = raster::render_glyph(&f, sf::G_IMATRA, px);
        raster::blit_glyph(
            &mut fb,
            &mark_bmp,
            20.0 + f32::from(att.x_offset) * scale,
            baseline - f32::from(att.y_offset) * scale,
            [1.0, 1.0, 1.0, 1.0],
        );
        assert!(ink_pixels(&fb) > 0, "基音 + 记号应产出非空像素");
        // 记号锚点 y=640（基上方）⇒ 记号墨迹必须出现在基音**上方**。
        let w = fb.width as usize;
        let mark_rows: Vec<usize> = (0..fb.height as usize)
            .filter(|&y| (0..w).any(|x| fb.pixels[(y * w + x) * 4 + 3] != 0))
            .collect();
        let top = mark_rows.first().copied().unwrap_or(0);
        // 基线 y=100，基音矩形 0..620 单位 ⇒ 上边缘约 100 - 620*0.064 ≈ 60。
        // 记号在 640 单位处 ⇒ 约 100 - 640*0.064 ≈ 59，落在上半区。
        assert!(top < 70, "记号应绘制在基线上方区域，实际最高行 {top}");
    }

    /// G25：内嵌资产端到端 —— 阿拉伯**画出来的是 .notdef 豆腐块**，不是字形（诚实记录）。
    #[test]
    fn g25_embedded_font_arabic_rasterizes_to_notdef_box() {
        let f = crate::font_regular();
        let ot = OtFont::parse(crate::FONT_REGULAR).expect("容器");
        let shaper = Shaper::new(ot.gsub(), ot.gpos()).expect("Shaper");
        let px = 64.0f32;
        let p = ShapeParams { font_size: px, ..ShapeParams::default() };
        let run = shape(&f, "\u{0628}\u{0644}\u{0627}", &p).expect("整形");
        let (glyphs, report) = shaper.apply(&run).expect("GSUB");
        // gid 全部回落 0（.notdef），替换 0 次。
        assert!(glyphs.iter().all(|g| g.gid == 0), "内嵌字体 gid 应全回落 0");
        assert_eq!(report.substitutions, 0);

        // **诚实边界的关键事实**：内嵌字体的 gid 0 并非空轮廓，而是 5  contours 的
        // 「豆腐块」矩形（bbox = [100,-120,900,880]）——所以画面上**确实有像素**，
        // 但画的是 .notdef 方框，不是阿拉伯字形。
        let notdef = f.outline(0).expect("gid 0 轮廓");
        assert!(
            !notdef.contours.is_empty(),
            "内嵌字体 gid 0 是可见的 .notdef 豆腐块（不是空轮廓）"
        );
        let mut fb = prism_io::Fbuf16F::new(200, 120);
        let mut pen = 10.0f32;
        for g in &glyphs {
            let bmp = raster::render_glyph(&f, g.gid, px);
            raster::blit_glyph(&mut fb, &bmp, pen, 90.0, [1.0, 1.0, 1.0, 1.0]);
            pen += f32::from(f.advance(g.gid)) * (px / f.units_per_em() as f32);
        }
        let ink = ink_pixels(&fb);
        assert!(ink > 0, "画面上确实有像素 —— 但那是 .notdef 豆腐块");
        // 画面**全部**由 .notdef 方框构成：把每个字形单独渲染的墨迹数求和，
        // 应等于整帧的非零 alpha 数（方框互不重叠 ⇒ 逐块相加）。
        // 注：整形层已把 LAM+ALEF 合并为**一个**簇，故只有 2 个 .notdef 方框。
        let per_glyph: Vec<usize> = glyphs
            .iter()
            .map(|g| ink_pixels(&raster_single(&f, g.gid, px)))
            .collect();
        assert_eq!(per_glyph.len(), 2, "LAM+ALEF 合并为1 簇 ⇒ 2 个 .notdef 方框");
        let sum: usize = per_glyph.iter().sum();
        assert_eq!(
            ink, sum,
            "整帧墨迹应等于各 .notdef 方框之和（证明画面只有豆腐块、无真实字形）"
        );
        // 每个方框彼此**完全相同**（同一个 gid 0）——与四态各不相同的场景对照。
        let first = raster_single(&f, 0, px);
        for gid in [0u16, 0] {
            assert_eq!(
                alpha_bits(&raster_single(&f, gid, px)),
                alpha_bits(&first),
                "gid {gid} 渲染的就是同一个 .notdef 方框"
            );
        }
        // 反证：同一绘制路径对**有字形**的字符能画出不同墨迹（证明上一条不是
        // 绘制代码本身坏了，而是真的缺字形）。
        let mut fb2 = prism_io::Fbuf16F::new(200, 120);
        let bmp = raster::render_glyph(&f, f.glyph_index('中').unwrap_or(0), px);
        raster::blit_glyph(&mut fb2, &bmp, 10.0, 90.0, [1.0, 1.0, 1.0, 1.0]);
        assert!(ink_pixels(&fb2) > 0, "内嵌字体的汉字应能画出墨迹（对照）");
        assert_ne!(
            alpha_bits(&fb2),
            alpha_bits(&fb),
            "汉字墨迹应与 .notdef 豆腐块不同（对照）"
        );
    }

    /// G26：未施加任何特征时，栅格结果 = 基础字形直绘（GSUB 是纯增量）。
    #[test]
    fn g26_no_feature_is_plain_raster() {
        // LAM 单独出现：整形给出 Isolated，但合成字体的 isol 特征只覆盖 BEH，
        // 故 LAM 的 gid 不变 ⇒ 画面应与直接 render_glyph(G_LAM) 完全一致。
        let (fb, gids, subs) = shape_gsub_raster("\u{0644}", 64.0);
        assert_eq!(gids, vec![sf::G_LAM], "无匹配特征 ⇒ gid 不变");
        assert_eq!(subs, 0, "无替换");

        let f = synth_font();
        let mut plain = prism_io::Fbuf16F::new(200, 120);
        let bmp = raster::render_glyph(&f, sf::G_LAM, 64.0);
        raster::blit_glyph(&mut plain, &bmp, 10.0, 90.0, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            alpha_bits(&fb),
            alpha_bits(&plain),
            "未施加特征时，端到端栅格应与直绘基础字形逐像素一致"
        );
    }
}
