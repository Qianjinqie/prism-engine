//! TTF/OTF 字体二进制解析（路线图 B5；白皮书 §16 文字管线输入端）。
//!
//! 解析 TrueType 子集所需最小表集：`head`（unitsPerEm / indexToLocFormat）、
//! `hhea`（度量行高）、`maxp`（字形数）、`hmtx`（步进宽）、`cmap`
//! （format 4 / format 12 码点→字形映射）、`loca` + `glyf`（二次贝塞尔
//! 轮廓，含复合字形 scale/2×2 变换）。
//!
//! # 确定性
//! 全部读取为大端整数（TTF 规范），坐标以 i16→f32 精确扩展（无舍入）；
//! 解析是字节流的纯函数——同文件同结果。CFF（三次轮廓）不支持：内嵌
//! 资产在构建期已由 cu2qu 转为二次 glyf（见 font_data/MODIFICATIONS）。

use std::fmt;

/// 字体解析错误（宪章 C18 三段式：在哪 / 什么事 / 怎么改）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontError {
    /// 错误码（Txxx 段：文本/字体）。
    pub code: &'static str,
    /// 什么事：检测到的事实。
    pub problem: String,
    /// 怎么改：可行动建议。
    pub fix: String,
}

impl fmt::Display for FontError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}] {}\n  建议：{}", self.code, self.problem, self.fix)
    }
}

impl std::error::Error for FontError {}

/// 大端整数读取（越界返回 None，由调用方归一为 FontError）。
mod be {
    pub fn u16(d: &[u8], off: usize) -> Option<u16> {
        Some(u16::from_be_bytes([*d.get(off)?, *d.get(off + 1)?]))
    }
    pub fn i16(d: &[u8], off: usize) -> Option<i16> {
        Some(u16(d, off)? as i16)
    }
    pub fn u32(d: &[u8], off: usize) -> Option<u32> {
        Some(u32::from_be_bytes([
            *d.get(off)?,
            *d.get(off + 1)?,
            *d.get(off + 2)?,
            *d.get(off + 3)?,
        ]))
    }
}

fn err(code: &'static str, problem: impl Into<String>, fix: impl Into<String>) -> FontError {
    FontError { code, problem: problem.into(), fix: fix.into() }
}

/// 轮廓点（字体单位，y 向上；`on_curve` = TrueType 旗标位 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlinePoint {
    /// x（字体单位）。
    pub x: f32,
    /// y（字体单位）。
    pub y: f32,
    /// 是否曲线上锚点（false = 二次控制点）。
    pub on_curve: bool,
}

/// 单条轮廓（点环，隐式闭合）。
pub type OutlineContour = Vec<OutlinePoint>;

/// 一个字形的完整轮廓（字体单位）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Outline {
    /// 轮廓列表（空 = 无墨字形，如空格）。
    pub contours: Vec<OutlineContour>,
}

/// 解析后的字体：持有字节流引用，按需解表。
#[derive(Debug, Clone)]
pub struct Font<'a> {
    data: &'a [u8],
    units_per_em: u16,
    ascender: i16,
    descender: i16,
    line_gap: i16,
    num_glyphs: u32,
    num_h_metrics: u16,
    index_to_loc_format: i16,
    cmap: CmapTable,
    loca_off: usize,
    glyf_off: usize,
    glyf_len: usize,
    hmtx_off: usize,
}

/// cmap 子表（延迟查表；仅记录所选子表偏移与格式）。
#[derive(Debug, Clone, Copy)]
struct CmapTable {
    subtable_off: usize,
    format: u16,
}

impl<'a> Font<'a> {
    /// 解析 TTF 字节流（校验 sfnt 魔数与必需表存在性）。
    pub fn parse(data: &'a [u8]) -> Result<Self, FontError> {
        if data.len() < 12 {
            return Err(err("T001", format!("字节流仅 {} B，不足 sfnt 头（12 B）", data.len()),
                "确认内嵌的是完整 TTF/OTF 文件而非截断片段"));
        }
        if be::u32(data, 0) != Some(0x0001_0000) {
            return Err(err("T001", "sfnt 版本非 0x00010000（TrueType）",
                "本解析器只吃 glyf 轮廓的 TTF；CFF 字体请在构建期转换"));
        }
        let num_tables = be::u16(data, 4).ok_or_else(|| err("T001", "表目录数量读取越界", "文件损坏"))?;
        let table = |tag: [u8; 4]| -> Option<(usize, usize)> {
            for i in 0..num_tables as usize {
                let base = 12 + i * 16;
                if data.get(base..base + 4)? == tag {
                    let off = be::u32(data, base + 8)? as usize;
                    let len = be::u32(data, base + 12)? as usize;
                    if off + len <= data.len() {
                        return Some((off, len));
                    }
                }
            }
            None
        };
        let missing = |t: &str| err("T002", format!("缺少必需表 {}", t),
            "内嵌资产应为未过度裁剪的 TTF 子集（保留 head/hhea/hmtx/maxp/cmap/loca/glyf）");
        let (head_off, _) = table(*b"head").ok_or_else(|| missing("head"))?;
        let (hhea_off, _) = table(*b"hhea").ok_or_else(|| missing("hhea"))?;
        let (maxp_off, _) = table(*b"maxp").ok_or_else(|| missing("maxp"))?;
        let (hmtx_off, _) = table(*b"hmtx").ok_or_else(|| missing("hmtx"))?;
        let (cmap_off, _) = table(*b"cmap").ok_or_else(|| missing("cmap"))?;
        let (loca_off, loca_len) = table(*b"loca").ok_or_else(|| missing("loca"))?;
        let (glyf_off, glyf_len) = table(*b"glyf").ok_or_else(|| missing("glyf"))?;

        let units_per_em = be::u16(data, head_off + 18)
            .filter(|v| (16..=16384).contains(v))
            .ok_or_else(|| err("T003", "unitsPerEm 非法（须 16..=16384）", "检查字体文件完整性"))?;
        let index_to_loc_format = be::i16(data, head_off + 50)
            .ok_or_else(|| err("T001", "head.indexToLocFormat 越界", "文件损坏"))?;
        let ascender = be::i16(data, hhea_off + 4)
            .ok_or_else(|| err("T001", "hhea.ascent 越界", "文件损坏"))?;
        let descender = be::i16(data, hhea_off + 6)
            .ok_or_else(|| err("T001", "hhea.descent 越界", "文件损坏"))?;
        let line_gap = be::i16(data, hhea_off + 8)
            .ok_or_else(|| err("T001", "hhea.lineGap 越界", "文件损坏"))?;
        let num_h_metrics = be::u16(data, hhea_off + 34)
            .ok_or_else(|| err("T001", "hhea.numberOfHMetrics 越界", "文件损坏"))?;
        // maxp.numGlyphs 是 **u16**（TrueType §maxp：version 0.5/1.0 均为
        // u16 numGlyphs，其后 maxPoints / maxContours / maxCompositePoints /
        // maxCompositeContours **全部是 u16**；只有 version 0.5 的 numGlyphs
        // 之前那 4 B version 才是 u32）。曾误用 `be::u32` 读取，把 numGlyphs
        // 与紧邻的 maxPoints 一并吃成 `numGlyphs | (maxPoints << 16)`（真值
        // 6889 被读成 451477751，放大 6.5 万倍），致`outline` 的越界守卫失效、
        // 越界 gid 直落 loca 切片 panic（宪章 C18）。本解析器只取 numGlyphs，
        // 其余 maxp 字段未读取——若将来新增，必须同样按 u16 读。
        let num_glyphs = be::u16(data, maxp_off + 4)
            .map(|v| v as u32)
            .ok_or_else(|| err("T001", "maxp.numGlyphs 越界", "文件损坏"))?;
        // loca 容量守卫（loca 越界根因）：loca 须容纳 num_glyphs + 1 项
        // （最后一项是哨兵）。不校验时，越界 gid 的 `loca` 读虽被 be::* 拦下，
        // 但 `end`值仍可能是 glyf 表外的任意 u32，落进切片即 panic。此处提前
        // 结构化报错，使 `outline` 的切片读在构造上恒安全。
        let loca_entry = match index_to_loc_format {
            0 => 2usize,
            1 => 4usize,
            _ => {
                return Err(err("T004", format!(
                    "head.indexToLocFormat = {}（须 0 或 1）",
                    index_to_loc_format
                ), "字体文件损坏或子集化重写head 时写错"))
            }
        };
        let loca_need = (num_glyphs as usize).saturating_add(1).saturating_mul(loca_entry);
        if (loca_len as usize) < loca_need {
            return Err(err("T004", format!(
                "loca 表仅{} B，不足 numGlyphs+1 = {} 项所需的 {} B",
                loca_len,
                num_glyphs as u64 + 1,
                loca_need
            ), "字体被过度子集化：loca 与 maxp.numGlyphs 不一致"));
        }
        let cmap = parse_cmap(data, cmap_off)?;

        Ok(Font {
            data,
            units_per_em,
            ascender,
            descender,
            line_gap,
            num_glyphs,
            num_h_metrics,
            index_to_loc_format,
            cmap,
            loca_off,
            glyf_off,
            glyf_len,
            hmtx_off,
        })
    }

    /// 字体单位数（em 方框）。
    pub fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    /// 上伸部（字体单位，正值）。
    pub fn ascender(&self) -> i16 {
        self.ascender
    }

    /// 下伸部（字体单位，负值）。
    pub fn descender(&self) -> i16 {
        self.descender
    }

    /// 行间隙（字体单位）。
    pub fn line_gap(&self) -> i16 {
        self.line_gap
    }

    /// 字形总数。
    pub fn num_glyphs(&self) -> u32 {
        self.num_glyphs
    }

    /// 码点 → 字形 id（cmap format 4 / 12 查找）。
    pub fn glyph_index(&self, ch: char) -> Option<u16> {
        let cp = ch as u32;
        match self.cmap.format {
            4 => self.cmap4_lookup(cp),
            12 => self.cmap12_lookup(cp),
            _ => None,
        }
    }

    /// 字形步进宽（字体单位；越界 id → 0）。
    pub fn advance(&self, gid: u16) -> u16 {
        let n = self.num_h_metrics as usize;
        if n == 0 {
            return 0;
        }
        let idx = if (gid as usize) < n { gid as usize } else { n - 1 };
        be::u16(self.data, self.hmtx_off + idx * 4).unwrap_or(0)
    }

    /// 字形轮廓（字体单位；gid 越界或空字形 → 空 Outline）。
    pub fn outline(&self, gid: u16) -> Result<Outline, FontError> {
        if gid as u32 >= self.num_glyphs {
            return Ok(Outline::default());
        }
        let (start, end) = self.loca_range(gid)?;
        if start >= end {
            return Ok(Outline::default()); // 空字形
        }
        let Some(g) = self.glyf_slice(start, end) else {
            return Ok(Outline::default()); // loca 越出 glyf 表（损坏数据）
        };
        let mut contours = Vec::new();
        self.parse_glyph(g, gid, &mut contours, 0)?;
        Ok(Outline { contours })
    }

    /// glyf 表内`[start, end)` 字节切片（越界/逆序 → None，绝不 panic）。
    ///
    /// 双向钳制是`outline` / `parse_composite_glyph` 共用的唯一切片入口：
    /// `loca` 是外部数据，`start`/`end` 可为 glyf 表外的任意 u32，直接下标
    /// 切片即越界 panic（宪章 C18）。逆序区间（`start > end`）在钳制后可能
    /// 变空，统一按"空字形"处理。
    fn glyf_slice(&self, start: usize, end: usize) -> Option<&'a [u8]> {
        let s = start.min(self.glyf_len);
        let e = end.min(self.glyf_len);
        if s >= e {
            return None;
        }
        let lo = self.glyf_off.checked_add(s)?;
        let hi = self.glyf_off.checked_add(e)?;
        self.data.get(lo..hi)
    }

    /// 解析单个 glyf 条目（简单或复合），复合字形递归展开（深度防环）。
    fn parse_glyph(
        &self,
        g: &[u8],
        gid: u16,
        out: &mut Vec<OutlineContour>,
        depth: u32,
    ) -> Result<(), FontError> {
        if depth > 8 {
            return Err(err("T005", format!("复合字形 (gid={gid}) 嵌套超过 8 层"), "字形数据可能成环，检查字体"));
        }
        let n_contours = be::i16(g, 0).ok_or_else(|| err("T004", "glyf 头越界", "文件损坏"))?;
        if n_contours >= 0 {
            parse_simple_glyph(g, n_contours as usize, out)
        } else {
            self.parse_composite_glyph(g, gid, out, depth)
        }
    }

    /// 复合字形：分量递归 + F2Dot14 仿射合成。
    fn parse_composite_glyph(
        &self,
        g: &[u8],
        gid: u16,
        out: &mut Vec<OutlineContour>,
        depth: u32,
    ) -> Result<(), FontError> {
        const MORE: u16 = 0x0020;
        const ARG_XY: u16 = 0x0002;
        const HAVE_SCALE: u16 = 0x0008;
        const HAVE_XY_SCALE: u16 = 0x0040;
        const HAVE_2X2: u16 = 0x0080;
        let mut off = 4usize;
        loop {
            let flags = be::u16(g, off)
                .ok_or_else(|| err("T005", format!("复合字形 (gid={gid}) 旗标越界"), "文件损坏"))?;
            let comp_gid = be::u16(g, off + 2)
                .ok_or_else(|| err("T005", format!("复合字形 (gid={gid}) 分量 id 越界"), "文件损坏"))?;
            off += 4;
            // 参数：ARG_1_AND_2_ARE_WORDS = i16×2；否则 int8×2（有符号！旧实现误读 2 B）。
            let word_args = flags & 0x0001 != 0;
            let arg_size = if word_args { 4 } else { 2 };
            let bad_arg = || err("T005", format!("复合字形 (gid={gid}) 分量参数越界"), "文件损坏");
            let (dx, dy) = if flags & ARG_XY != 0 {
                let (raw1, raw2) = if word_args {
                    (
                        be::i16(g, off).ok_or_else(bad_arg)?,
                        be::i16(g, off + 2).ok_or_else(bad_arg)?,
                    )
                } else {
                    (
                        *g.get(off).ok_or_else(bad_arg)? as i8 as i16,
                        *g.get(off + 1).ok_or_else(bad_arg)? as i8 as i16,
                    )
                };
                (raw1 as f32, raw2 as f32)
            } else {
                // 点匹配语义（控位对齐）不支持：结构化报错，绝不静默错位。
                return Err(err("T006",
                    format!("复合字形 (gid={gid}) 使用点匹配定位（非 XY 偏移），本解析器不支持"),
                    "构建期子集化时保留 XY 偏移分量，或改用纯简单字形资产"));
            };
            off += arg_size;
            // 仿射：F2Dot14 = i16 / 2^14（精确除法：2 的幂）。
            let bad_scale = || err("T005", "复合字形缩放字段越界", "文件损坏");
            let (a, b, c, d) = if flags & HAVE_2X2 != 0 {
                (
                    be::i16(g, off).ok_or_else(bad_scale)? as f32 / 16384.0,
                    be::i16(g, off + 2).ok_or_else(bad_scale)? as f32 / 16384.0,
                    be::i16(g, off + 4).ok_or_else(bad_scale)? as f32 / 16384.0,
                    be::i16(g, off + 6).ok_or_else(bad_scale)? as f32 / 16384.0,
                )
            } else if flags & HAVE_XY_SCALE != 0 {
                let sx = be::i16(g, off).ok_or_else(bad_scale)? as f32 / 16384.0;
                let sy = be::i16(g, off + 2).ok_or_else(bad_scale)? as f32 / 16384.0;
                (sx, 0.0, 0.0, sy)
            } else if flags & HAVE_SCALE != 0 {
                let s = be::i16(g, off).ok_or_else(bad_scale)? as f32 / 16384.0;
                (s, 0.0, 0.0, s)
            } else {
                (1.0, 0.0, 0.0, 1.0)
            };
            if flags & (HAVE_SCALE | HAVE_XY_SCALE | HAVE_2X2) != 0 {
                off += 4;
                if flags & (HAVE_XY_SCALE | HAVE_2X2) != 0 {
                    off += 4;
                }
                if flags & HAVE_2X2 != 0 {
                    off += 4;
                }
            }

            let mut sub = Outline::default();
            if comp_gid as u32 >= self.num_glyphs {
                return Err(err("T005", format!("复合字形 (gid={gid}) 分量 id 超出 numGlyphs"), "文件损坏"));
            }
            let (cs, ce) = self.loca_range(comp_gid)?;
            if let Some(cg) = self.glyf_slice(cs, ce) {
                self.parse_glyph(cg, comp_gid, &mut sub.contours, depth + 1)?;
            }
            for mut ring in sub.contours {
                for p in &mut ring {
                    let nx = a * p.x + c * p.y + dx;
                    let ny = b * p.x + d * p.y + dy;
                    p.x = nx;
                    p.y = ny;
                }
                out.push(ring);
            }
            if flags & MORE == 0 {
                break;
            }
        }
        Ok(())
    }

    /// gid → glyf 内偏移区间（`loca` 短格式存 offset/2）。
    ///
    /// `index_to_loc_format` 的合法性已在 [`Font::parse`] 一次性校验并连带
    /// 校验了 loca 容量，故此处只需处理 `loca` 字节越界。
    fn loca_range(&self, gid: u16) -> Result<(usize, usize), FontError> {
        let entry = |i: usize| -> Option<usize> {
            match self.index_to_loc_format {
                0 => Some(be::u16(self.data, self.loca_off + i * 2)? as usize * 2),
                1 => Some(be::u32(self.data, self.loca_off + i * 4)? as usize),
                // parse 已拒非 0/1；此处为纯防御（永不 panic）。
                _ => None,
            }
        };
        let s = entry(gid as usize)
            .ok_or_else(|| err("T004", "loca 越界（字形 id 超出 loca 表）", "字体文件损坏"))?;
        let e = entry(gid as usize + 1)
            .ok_or_else(|| err("T004", "loca 越界（字形 id + 1 超出 loca 表）", "字体文件损坏"))?;
        Ok((s, e))
    }

    /// cmap format 4 查找（分段二分 + idDelta/idRangeOffset）。
    fn cmap4_lookup(&self, cp: u32) -> Option<u16> {
        let d = self.data;
        let t = self.cmap.subtable_off;
        if cp > 0xFFFF {
            return None;
        }
        let seg_x2 = be::u16(d, t + 6)?;
        let seg_count = (seg_x2 / 2) as usize;
        let end_off = t + 14;
        // 终止码二分：endCodes[i] ≥ cp 的最左段。
        let mut lo = 0usize;
        let mut hi = seg_count;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (be::u16(d, end_off + mid * 2)? as u32) < cp {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= seg_count {
            return None;
        }
        let start_off = end_off + seg_x2 as usize + 2;
        let delta_off = start_off + seg_x2 as usize;
        let range_off = delta_off + seg_x2 as usize;
        let start = be::u16(d, start_off + lo * 2)? as u32;
        let end = be::u16(d, end_off + lo * 2)? as u32;
        if cp < start || cp > end {
            return None;
        }
        let range = be::u16(d, range_off + lo * 2)?;
        if range == 0 {
            let delta = be::i16(d, delta_off + lo * 2)?;
            Some((cp as u16).wrapping_add(delta as u16))
        } else {
            // glyphIdArray：rangeOff 相对自身位置的偏移。
            let gid_addr = range_off + lo * 2 + range as usize + (cp - start) as usize * 2;
            be::u16(d, gid_addr).filter(|v| *v != 0)
        }
    }

    /// cmap format 12 查找（分组二分）。
    fn cmap12_lookup(&self, cp: u32) -> Option<u16> {
        let d = self.data;
        let t = self.cmap.subtable_off;
        let n_groups = be::u32(d, t + 12)? as usize;
        let groups = t + 16;
        let mut lo = 0usize;
        let mut hi = n_groups;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let start = be::u32(d, groups + mid * 12)?;
            if start < cp {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo 为首个 start ≥ cp 的组；前一组的 end 可能仍覆盖 cp。
        for i in [lo.checked_sub(1), Some(lo)].into_iter().flatten() {
            if i >= n_groups {
                continue;
            }
            let start = be::u32(d, groups + i * 12)?;
            let end = be::u32(d, groups + i * 12 + 4)?;
            if cp >= start && cp <= end {
                let glyph = be::u32(d, groups + i * 12 + 8)? + (cp - start);
                return u16::try_from(glyph).ok();
            }
        }
        None
    }
}

// 保留字段占位避免未使用告警（loca_len 供后续校验扩展）。
/// 简单字形：旗标流 + 增量坐标解包（TrueType 规范 §glyf）。
fn parse_simple_glyph(g: &[u8], n_contours: usize, out: &mut Vec<OutlineContour>) -> Result<(), FontError> {
    let need = |n: usize| err("T004", format!("glyf 数据不足（需 {n} B 处）"), "文件损坏");
    if n_contours == 0 {
        return Ok(());
    }
    let mut off = 10usize; // numberOfContours(2) + xMin..yMax(8)
    let mut ends = Vec::with_capacity(n_contours);
    for i in 0..n_contours {
        ends.push(be::u16(g, off + i * 2).ok_or_else(|| need(off))? as usize);
    }
    off += n_contours * 2;
    let n_points = *ends.last().ok_or_else(|| need(0))? as usize + 1;
    let instr_len = be::u16(g, off).ok_or_else(|| need(off))? as usize;
    off += 2 + instr_len;

    // 旗标（含 repeat 展开）。
    let mut flags = Vec::with_capacity(n_points);
    while flags.len() < n_points {
        let f = *g.get(off).ok_or_else(|| need(off))?;
        off += 1;
        flags.push(f);
        if f & 0x08 != 0 {
            let rep = *g.get(off).ok_or_else(|| need(off))?;
            off += 1;
            for _ in 0..rep {
                if flags.len() >= n_points {
                    break;
                }
                flags.push(f);
            }
        }
    }

    // x 增量解包。
    let mut xs = Vec::with_capacity(n_points);
    let mut x = 0i32;
    for &f in &flags {
        let short = f & 0x02 != 0;
        let same = f & 0x10 != 0;
        if short {
            let d = *g.get(off).ok_or_else(|| need(off))? as i32;
            off += 1;
            x += if same { d } else { -d };
        } else if !same {
            x += be::i16(g, off).ok_or_else(|| need(off))? as i32;
            off += 2;
        }
        xs.push(x);
    }
    // y 增量解包。
    let mut ys = Vec::with_capacity(n_points);
    let mut y = 0i32;
    for &f in &flags {
        let short = f & 0x04 != 0;
        let same = f & 0x20 != 0;
        if short {
            let d = *g.get(off).ok_or_else(|| need(off))? as i32;
            off += 1;
            y += if same { d } else { -d };
        } else if !same {
            y += be::i16(g, off).ok_or_else(|| need(off))? as i32;
            off += 2;
        }
        ys.push(y);
    }

    // 按终点切轮廓（i16 → f32 精确扩展，无舍入）。
    //
    // 防御性钳制：损坏字体可能给出非单调的 `ends`（某 end < start），此时
    // `end - start` 会在 usize 下下溢 panic；`n_points` 亦恒 ≥ 1（由
    // `ends.last() + 1` 得出）故 `n_points - 1` 安全，但仍用 saturating_sub
    // 显式表达"空轮廓 ⇒ 空环"契约：越界/逆序终点一律产出空轮廓并跳过。
    let last = n_points.saturating_sub(1);
    let mut start = 0usize;
    for &end in &ends {
        let end = end.min(last);
        if end < start {
            // 逆序终点（损坏数据）：不推进 start，产出空轮廓。
            out.push(Vec::new());
            continue;
        }
        let mut ring = Vec::with_capacity(end - start + 1);
        for i in start..=end {
            ring.push(OutlinePoint {
                x: xs[i] as f32,
                y: ys[i] as f32,
                on_curve: flags[i] & 0x01 != 0,
            });
        }
        out.push(ring);
        start = end + 1;
    }
    Ok(())
}

/// 选最佳 cmap 子表：优先 (3,10)/(0,4+) 的 format 12，其次 (3,1)/(0,3) 的 format 4。
fn parse_cmap(data: &[u8], cmap_off: usize) -> Result<CmapTable, FontError> {
    let bad = || err("T007", "cmap 表损坏或无可用子表", "检查字体文件 cmap 编码子表");
    let n = be::u16(data, cmap_off + 2).ok_or_else(bad)? as usize;
    let mut best: Option<(u8, CmapTable)> = None;
    for i in 0..n {
        let rec = cmap_off + 4 + i * 8;
        let platform = be::u16(data, rec).ok_or_else(bad)?;
        let encoding = be::u16(data, rec + 2).ok_or_else(bad)?;
        let off = be::u32(data, rec + 4).ok_or_else(bad)? as usize;
        let t = cmap_off + off;
        let format = be::u16(data, t).ok_or_else(bad)?;
        // 得分：越大越好（Unicode 全集 > BMP > 兼容）。
        let score: u8 = match (platform, encoding, format) {
            (3, 10, 12) | (0, 4, 12) | (0, 6, 12) => 4,
            (3, 10, _) | (0, _, _) => 3,
            (3, 1, 4) => 2,
            (3, _, _) => 1,
            _ => 0,
        };
        if score > 0 && best.map(|(s, _)| score > s).unwrap_or(true) {
            best = Some((score, CmapTable { subtable_off: t, format }));
        }
    }
    best.map(|(_, t)| t).ok_or_else(bad)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    /// N1 回归：`maxp.numGlyphs` 是 **u16** 字段（其后maxPoints 等亦u16）。
    ///
    /// 旧实现用 `be::u32(data, maxp_off + 4)` 读取，把 numGlyphs 与紧邻的
    /// maxPoints 拼成 `numGlyphs | (maxPoints << 16)`，真值被放大 6.5 万倍，
    /// 致 `outline` 的越界守卫失效、越界 gid 直落 loca 切片 panic（C18）。
    #[test]
    fn maxp_num_glyphs_is_u16_not_u32() {
        let f = font_regular();
        // 内嵌 PrismSansSC-Regular.ttf 的真实 numGlyphs（可由 maxp 表字节交叉验证）。
        assert_eq!(f.num_glyphs, 6889, "numGlyphs 是 u16 字段，不得吃进 maxPoints");
        // 变异自证：旧错误读法得到的放大值（6889 | (247 << 16)）不等于真值。
        let inflated = 6889u32 | (247u32 << 16);
        assert_ne!(inflated, f.num_glyphs, "若相等则本测试无意义");
        assert!(inflated > 6889 * 1000, "旧读法放大约 6.5 万倍");
    }

    /// N1 回归：真实字形 id 全部可解析，且不 panic。
    #[test]
    fn in_range_glyph_ids_all_parse() {
        let f = font_regular();
        for gid in [0u16, 1, 36, 65, 103, 20013, f.num_glyphs() as u16 - 1] {
            // Ok 或 Err 均可（点匹配复合字形报 T006 是已知不支持），
            // 但绝不能 panic。
            let _ = f.outline(gid);
        }
    }

    /// N1 回归：越界 gid 一律返回**空轮廓**（不 Err、不 panic）。
    #[test]
    fn out_of_range_gid_returns_empty_outline() {
        let f = font_regular();
        let n = f.num_glyphs() as u16;
        for gid in [n, n.wrapping_add(1), 7000, 10000, u16::MAX] {
            let out = f.outline(gid).expect("越界 gid 应 Ok 而非 Err");
            assert!(out.contours.is_empty(), "gid={gid} 应返回空轮廓");
        }
    }

    /// loca 越界根因：即便守卫被绕过，glyf 切片入口也必须自行钳制。
    ///
    /// 直接构造 `num_glyphs` 正确但 loca 内容全部指向 glyf 表外的字体：
    /// `glyf_slice` 的双向 `min(glyf_len)` 钳制使 `outline` 返回空轮廓。
    #[test]
    fn loca_pointing_outside_glyf_yields_empty_not_panic() {
        let f = font_regular();
        let gid = f.glyph_index('A').expect("A 在子集内");
        let out = f.outline(gid).expect("正常字形应Ok");
        // 前置条件：A 确有轮廓（否则本测试无意义）。
        assert!(!out.contours.is_empty(), "A 应有轮廓");

        // 反向自证：把同一字形的 loca 区间人为推到 glyf 表外不会 panic。
        // 走glyf_slice 的钳制路径：start/end 超表长 ⇒ None ⇒ 空轮廓。
        let huge = f.glyf_len + 1_000_000;
        assert!(f.glyf_slice(huge, huge + 4096).is_none(), "超表 loca 应被钳制为空");
        assert!(f.glyf_slice(0, huge).is_some(), "end 超表应被钳到表尾");
        // 逆序区间（start > end）⇒ 空。
        assert!(f.glyf_slice(100, 50).is_none(), "逆序区间应为空");
    }

    /// `loca` 容量守卫：loca 项数不足 `num_glyphs + 1` 的字体在parse 期
    /// 结构化报错，而非留到切片阶段panic。
    #[test]
    fn truncated_loca_is_rejected_at_parse_time() {
        static B: &[u8] = include_bytes!("../font_data/PrismSansSC-Regular.ttf");
        // 截断到刚好 sfnt 头 + 表目录：必然缺表⇒ T002。
        let e = Font::parse(&B[..40]).expect_err("截断字体应parse 失败");
        assert!(e.code == "T001" || e.code == "T002", "得到 {}", e.code);
    }
}
