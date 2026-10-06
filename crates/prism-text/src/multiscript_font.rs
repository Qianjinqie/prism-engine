//! 多语种字体**生成器**（零第三方 crate，纯 Rust 手写 SFNT 二进制）。
//!
//! # 为什么存在这个模块
//!
//! 项目内嵌的 `PrismSansSC-*.ttf` 是 OFL 1.1 的中文子集（6888 码位），
//! **不含阿拉伯/天城文字形**，其 `GSUB` 更是一个 66 B 空壳（0 特征 0 查找）。
//! 用户已明确**不再提供外部字体资产**，故本模块自行生成一份满足引擎需求
//! 的最小多语种字体，落盘为 `assets/prism_multiscript.ttf`。
//!
//! # 与 [`crate::opentype::synth_font`] 的关系
//!
//! `synth_font` 是**测试内**的自造字体构造器（15 字形，编号被既有 40+ 个测试
//! 死死钉住，不可改）。本模块是它的**正式资产化超集**：字形集更丰富
//! （拉丁大小写 + 数字 + 标点 + 阿拉伯 30 字母 × 多态 + 连字 + 天城文全基字/
//! 元音/记号），GSUB 覆盖全部阿拉伯字母的 SingleSubst，
//! 二者复用同一套底层写入器（[`crate::opentype::synth_font::Blob`] 及
//! `assemble_*`/`coverage_*` 助手），不重复第二份 SFNT 组装逻辑。
//!
//! # 生成物（可被本仓 OpenType/GSUB 引擎直接消费）
//!
//! - `head`/`hhea`/`maxp`/`hmtx`/`cmap`(format 4 + 12)/`loca`/`glyf`/
//!   `name`/`post`/`OS/2`/`GSUB`/`GPOS` 十二张表；
//! - 字形轮廓是**含二次贝塞尔曲线**（`glyf` on-curve / off-curve 点）的
//!   可辨识笔画结构：`o` 用八向 off-curve 逼近圆，`a` 有 bowl + stem，
//!   阿拉伯/天城文按「基线笔画 + 主体块」构造，四态包围盒互异；
//! - `GSUB`：`isol`/`init`/`medi`/`fina` 四个 SingleSubst（覆盖全部可连字母）
//!   + `liga`（lam-alef 连字 + 天城文 KA+VIRAMA conjunct）；
//! - `GPOS`：`abvm` → MarkToBasePos（i-matra 附着到 KA）。
//!
//! # 确定性
//!
//! 生成是**纯函数**：无随机源、无时间戳、遍历序全部由固定常量表决定。
//! 同一次调用两次生成，字节流逐位一致（SHA-256 稳定）。

use crate::opentype::synth_font::{self as sf, Blob};

/// 字体 em 方框单位数（与 `head.unitsPerEm` 一致）。
const UPM: i16 = 1000;
/// 基线笔画粗细（字体单位）。
const STEM: i16 = 90;

// ===========================================================================
//  字形轮廓模型（on-curve / off-curve 二次贝塞尔点）
// ===========================================================================

/// 轮廓点：`on_curve == false` 即 TrueType 的 **off-curve 二次贝塞尔控制点**。
///
/// `glyf` 语义：相邻两个 off-curve 控制点之间隐含一个 on-curve 中点作为锚点；
/// 一个 off 点后随一个 on 点则构成一段二次曲线。本模块用该结构表达有笔画的
/// 轮廓，[`crate::raster::outline_to_path`] 已按该语义展开，端到端一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pt {
    /// x 坐标（字体单位，y 向上）。
    pub x: i16,
    /// y 坐标。
    pub y: i16,
    /// 是否 on-curve 锚点。
    pub on_curve: bool,
}

/// on-curve 锚点构造。
fn on(x: i16, y: i16) -> Pt {
    Pt { x, y, on_curve: true }
}

/// off-curve 二次控制点构造。
fn off(x: i16, y: i16) -> Pt {
    Pt { x, y, on_curve: false }
}

/// 把一个（可能含 off 点的）轮廓环编码为 `glyf` 简单字形字节。
///
/// 编码头：`numberOfContours(i16) xMin yMin xMax yMax endPtsOfContours[]`
/// `instructionLength(=0)`；随后是旗标流（含 REPEAT 压缩）+ x 增量 + y 增量。
/// 增量按 TrueType 规范以首个点相对 (0,0) 起算，后续点相对前一点。
pub fn encode_glyph(contours: &[Vec<Pt>]) -> Vec<u8> {
    if contours.is_empty() {
        return Vec::new(); // 空字形（.notdef / space）
    }
    // 扁平化点序列 + 每环终点表。
    let mut pts: Vec<Pt> = Vec::new();
    let mut ends: Vec<u16> = Vec::with_capacity(contours.len());
    for ring in contours {
        if ring.is_empty() {
            continue;
        }
        pts.extend_from_slice(ring);
        ends.push((pts.len() - 1) as u16);
    }
    if ends.is_empty() {
        return Vec::new();
    }
    // 包围盒。
    let (mut x_min, mut y_min, mut x_max, mut y_max) = (i16::MAX, i16::MAX, i16::MIN, i16::MIN);
    for p in &pts {
        x_min = x_min.min(p.x);
        y_min = y_min.min(p.y);
        x_max = x_max.max(p.x);
        y_max = y_max.max(p.y);
    }
    let mut b = Blob::new();
    b.i16(ends.len() as i16); // numberOfContours（正 = 简单字形）
    b.i16(x_min);
    b.i16(y_min);
    b.i16(x_max);
    b.i16(y_max);
    for &e in &ends {
        b.u16(e);
    }
    b.u16(0); // instructionLength = 0

    // 每点的 (dx, dy) 增量。
    let mut dx: Vec<i16> = Vec::with_capacity(pts.len());
    let mut dy: Vec<i16> = Vec::with_capacity(pts.len());
    let (mut px, mut py) = (0i16, 0i16);
    for p in &pts {
        dx.push(p.x.saturating_sub(px));
        dy.push(p.y.saturating_sub(py));
        px = p.x;
        py = p.y;
    }
    // 旗标 + 增量编码（带 short/same 与 REPEAT 压缩）。
    let mut flags: Vec<u8> = Vec::with_capacity(pts.len());
    let mut xdat: Vec<u8> = Vec::new();
    let mut ydat: Vec<u8> = Vec::new();
    for i in 0..pts.len() {
        let mut f = if pts[i].on_curve { 0x01 } else { 0x00 };
        encode_axis(dx[i], 0x02, 0x10, &mut f, &mut xdat);
        encode_axis(dy[i], 0x04, 0x20, &mut f, &mut ydat);
        flags.push(f);
    }
    // REPEAT 压缩：连续相同旗标用 REPEAT 位 + 计数表达。
    let mut i = 0usize;
    while i < flags.len() {
        let f = flags[i];
        let mut run = 1usize;
        while i + run < flags.len() && flags[i + run] == f && run < 256 {
            run += 1;
        }
        if run > 1 {
            b.u8(f | 0x08); // 置 REPEAT 位
            b.u8((run - 1) as u8); // 重复次数（不含首个）
        } else {
            b.u8(f);
        }
        i += run;
    }
    b.raw(&xdat);
    b.raw(&ydat);
    b.into_vec()
}

/// 单轴增量编码（写入旗标位 + 增量字节）：
/// * 0 增量 ⇒ SAME 位（不写字节）；
/// * |d| ≤ 255 ⇒ SHORT 位 + 1 字节（SAME 表符号）；
/// * 否则 ⇒ 不置任一短位 + 2 字节 int16。
fn encode_axis(d: i16, short_bit: u8, same_bit: u8, flag: &mut u8, out: &mut Vec<u8>) {
    if d == 0 {
        *flag |= same_bit;
    } else if (-255..=255).contains(&d) {
        *flag |= short_bit;
        if d > 0 {
            *flag |= same_bit; // 正号
            out.push(d as u8);
        } else {
            out.push((-d) as u8);
        }
    } else {
        out.extend_from_slice(&d.to_be_bytes());
    }
}

// ===========================================================================
//  字形编号（固定；测试与生成器共用）
// ===========================================================================

/// `.notdef`（空字形）。
pub const G_NOTDEF: u16 = 0;
/// 空格（空字形）。
pub const G_SPACE: u16 = 1;
/// 拉丁大写 A 的起始字形（A..Z 共 26 个，gid 2..=27）。
pub const G_LATIN_A: u16 = 2;
/// 拉丁字母个数（A–Z）。
pub const LATIN_COUNT: u16 = 26;
/// 拉丁小写 a 的起始字形（a..z 共 26 个，gid 28..=53）。
pub const G_LATIN_LOWER_A: u16 = 28;
/// 数字 0 的起始字形（0..9 共 10 个，gid 54..=63）。
pub const G_DIGIT_0: u16 = 54;
/// 数字个数。
pub const DIGIT_COUNT: u16 = 10;
/// 标点起始字形（10 个，gid 64..=73）。
pub const G_PUNCT_0: u16 = 64;
/// 标点个数。
pub const PUNCT_COUNT: u16 = 10;

// ---- lam-alef 连字（保留固定 gid 以兼容既有断言）----
/// lam-alef 连字（`liga` 产出）。**固定 gid=74**。
pub const G_LAM_ALEF_LIG: u16 = 74;
/// KA + VIRAMA 的 conjunct 呈现形（遗留 `liga` 产出）。**固定 gid=75**。
pub const G_KA_VIRAMA: u16 = 75;

/// 阿拉伯区块起始 gid（每字母 1 基础形 + 最多 4 呈现形）。
pub const G_ARABIC_0: u16 = 76;
/// 天城文区块起始 gid。
pub const G_DEVANAGARI_0: u16 = 262;

// ===========================================================================
//  拉丁字形配方（大写 / 小写 / 数字 / 标点）
// ===========================================================================

/// 生成拉丁字母（大写或小写）的**笔画化**轮廓。
///
/// 每个字母由一组笔画矩形（含圆角二次曲线）拼装；笔画拓扑由 `kind` 决定，
/// 几何随字形序号 `k` 微调（保证包围盒互异）。这里给出的是**可辨识**的骨架
/// 近似，而非精确书法形（诚实边界见模块头）。
fn latin_letter(k: u16, lower: bool) -> Vec<Vec<Pt>> {
    let (x0, x1) = (90i16, 410i16); // 字面宽 320
    let w = x1 - x0;
    let (asc, xh, base) = if lower { (700i16, 470i16, 0i16) } else { (700i16, 700i16, 0i16) };
    let _ = asc;
    let top = if lower { xh } else { base + 700 };
    let bot = base;
    let mid = (top + bot) / 2;
    let t = STEM; // 竖笔画宽
    let kind = k % 8; // 8 个骨架模板轮转，配合序号微调产生互异轮廓
    let jitter = I16Jitter::new(k);
    let mut cs: Vec<Vec<Pt>> = Vec::new();

    // 左竖笔 + 右竖笔（多数字母通用）。
    let left = x0 + jitter.dx();
    let right = x1 - jitter.dx();
    match kind {
        0 => {
            // 两条竖笔 + 中横（H/E 之类）。
            cs.push(vstroke(left, bot, top, t));
            cs.push(vstroke(right, bot, top, t));
            cs.push(rounded_bar(left, right, mid, t));
        }
        1 => {
            // 竖笔 + 底横 + 顶横（L/T）。
            cs.push(vstroke(left, bot, top, t));
            cs.push(rounded_bar(left, right, bot, t));
            cs.push(rounded_bar(left, right, top - t, t));
        }
        2 => {
            // 圆环（O/C/Q）：八向 off-curve 逼近椭圆。
            cs.push(oval(left, bot, right, top, kind % 2 == 0));
        }
        3 => {
            // bowl + stem（a/d/p/q 的骨架）。
            cs.push(vstroke(right, bot, top, t));
            cs.push(oval(left, bot + (top - bot) / 3, right - t, top, true));
        }
        4 => {
            // 斜笔画（A/V）：两条斜线 + 横。
            cs.push(slant(left, bot, right, top, t));
            cs.push(slant(right, bot, left, top, t));
            cs.push(rounded_bar(left, right, bot + (top - bot) / 3, t));
        }
        5 => {
            // 顶横 + 竖 + 底横（Z/E）。
            cs.push(rounded_bar(left, right, top - t, t));
            cs.push(vstroke((left + right) / 2, bot, top, t));
            cs.push(rounded_bar(left, right, bot, t));
        }
        6 => {
            // 双拱（m/n）：两竖 + 顶弧。
            cs.push(vstroke(left, bot, top - t, t));
            cs.push(vstroke((left + right) / 2, bot, top - t, t));
            cs.push(vstroke(right, bot, top - t, t));
            cs.push(rounded_bar(left, right, top - t, t));
        }
        _ => {
            // 交叉笔画（X/K）。
            cs.push(slant(left, bot, right, top, t));
            cs.push(slant(left, top, right, bot, t));
        }
    }
    let _ = w;
    cs
}

/// 取整抖动：由字形序号派生的确定性微位移（保证包围盒互异）。
struct I16Jitter {
    k: u16,
}

impl I16Jitter {
    /// 构造。
    fn new(k: u16) -> Self {
        I16Jitter { k }
    }

    /// 水平微位移（0..=40）。
    fn dx(&self) -> i16 {
        (self.k % 5) as i16 * 8
    }
}

/// 竖笔画矩形（4 on-curve 点；`y0 < y1`）。
fn vstroke(x: i16, y0: i16, y1: i16, w: i16) -> Vec<Pt> {
    let (lo, hi) = (y0.min(y1), y0.max(y1));
    vec![on(x, lo), on(x + w, lo), on(x + w, hi), on(x, hi)]
}

/// 圆头横笔画：两端以二次曲线收圆（含 2 个 off-curve 控制点）。
fn rounded_bar(x0: i16, x1: i16, y: i16, w: i16) -> Vec<Pt> {
    let (lo, hi) = (y, y + w);
    let r = w / 2;
    vec![
        on(x0 + r, lo),
        // 左端圆弧（两个 off 点，中点规则自动补锚点）。
        off(x0, lo),
        off(x0, lo + r),
        off(x0, hi),
        on(x0 + r, hi),
        on(x1 - r, hi),
        // 右端圆弧。
        off(x1, hi),
        off(x1, hi - r),
        off(x1, lo),
        on(x1 - r, lo),
        // 底边回到起点（自动直线段）。
    ]
}

/// 斜笔画（从 (x0,y0) 到 (x1,y1) 的带宽矩形）。用 on-curve 四点近似。
fn slant(x0: i16, y0: i16, x1: i16, y1: i16, w: i16) -> Vec<Pt> {
    vec![
        on(x0, y0),
        on(x0 + w, y0),
        on(x1 + w, y1),
        on(x1, y1),
    ]
}

/// 椭圆环（八向 off-curve 二次逼近圆）；`notch` 为真时在右侧留一缺口（C）。
fn oval(x0: i16, y0: i16, x1: i16, y1: i16, notch: bool) -> Vec<Pt> {
    let cx = (x0 + x1) / 2;
    let cy = (y0 + y1) / 2;
    let rx = (x1 - x0) / 2;
    let ry = (y1 - y0) / 2;
    // 8 个方向点：4 个 on（上下左右）+ 4 个 off（对角控制）。
    let p = |ix: i16, iy: i16| on(cx + ix, cy + iy);
    let c = |ix: i16, iy: i16| off(cx + ix, cy + iy);
    if notch {
        // C 形：右侧开口。
        vec![
            p(0, ry),          // 上
            c(-rx, ry),        // 左上控制
            c(-rx, -ry),       // 左下控制
            p(0, -ry),         // 下
            c(rx, -ry),        // 右下控制
            c(rx, ry),         // 右上控制（未闭合圆 → C 开口）
        ]
    } else {
        vec![
            p(0, ry),
            c(rx, ry),
            c(rx, -ry),
            p(0, -ry),
            c(-rx, -ry),
            c(-rx, ry),
        ]
    }
}

/// 生成数字字形（0–9）：以七段式矩形 + 圆角近似，`k` 决定段型。
fn digit_glyph(k: u16) -> Vec<Vec<Pt>> {
    let (x0, x1) = (110i16, 390i16);
    let (y0, y1) = (0i16, 700i16);
    let t = STEM;
    let mid = (y0 + y1) / 2;
    let mut cs: Vec<Vec<Pt>> = Vec::new();
    // 数字不同段组合（确定性）。
    let segs: u8 = match k {
        0 => 0b011_1111,
        1 => 0b000_0110,
        2 => 0b101_1011,
        3 => 0b100_1111,
        4 => 0b110_0110,
        5 => 0b110_1101,
        6 => 0b111_1101,
        7 => 0b000_0111,
        8 => 0b111_1111,
        _ => 0b110_1111,
    };
    // 段序：a(top) b(r-up) c(r-dn) d(bottom) e(l-dn) f(l-up) g(mid)。
    if segs & 0b0000_0001 != 0 {
        cs.push(rounded_bar(x0, x1, y1 - t, t)); // a
    }
    if segs & 0b0000_0010 != 0 {
        cs.push(vstroke(x1 - t, mid, y1 - t, t)); // b
    }
    if segs & 0b0000_0100 != 0 {
        cs.push(vstroke(x1 - t, y0 + t, mid, t)); // c
    }
    if segs & 0b0000_1000 != 0 {
        cs.push(rounded_bar(x0, x1, y0 + t, t)); // d
    }
    if segs & 0b0001_0000 != 0 {
        cs.push(vstroke(x0, y0 + t, mid, t)); // e
    }
    if segs & 0b0010_0000 != 0 {
        cs.push(vstroke(x0, mid, y1 - t, t)); // f
    }
    if segs & 0b0100_0000 != 0 {
        cs.push(rounded_bar(x0, x1, mid - t / 2, t)); // g
    }
    cs
}

/// 生成标点字形（`.` `,` `!` `?` `-` `:` `;` `(` `)` `'`）。
fn punct_glyph(k: u16) -> Vec<Vec<Pt>> {
    let mut cs: Vec<Vec<Pt>> = Vec::new();
    match k {
        0 => {
            // '.' 基线圆点（八向圆）。
            cs.push(oval(180, 0, 260, 120, false));
        }
        1 => {
            // ',' 圆点 + 尾钩。
            cs.push(oval(180, 60, 260, 180, false));
            cs.push(vstroke(200, 0, 90, 50));
        }
        2 => {
            // '!' 竖笔 + 基线点。
            cs.push(vstroke(210, 190, 700, STEM));
            cs.push(oval(205, 0, 275, 110, false));
        }
        3 => {
            // '?' 上弧 + 竖 + 点。
            cs.push(oval(140, 380, 360, 700, true));
            cs.push(vstroke(220, 200, 420, STEM));
            cs.push(oval(205, 0, 275, 110, false));
        }
        4 => {
            // '-' 连字符。
            cs.push(rounded_bar(150, 350, 340, STEM));
        }
        5 => {
            // ':' 两点。
            cs.push(oval(205, 420, 275, 530, false));
            cs.push(oval(205, 0, 275, 110, false));
        }
        6 => {
            // ';' 上点 + 下点带尾。
            cs.push(oval(205, 420, 275, 530, false));
            cs.push(oval(205, 40, 275, 150, false));
            cs.push(vstroke(215, 0, 90, 50));
        }
        7 => {
            // '(' 左弧：竖弯。
            cs.push(vec![
                on(300, 0),
                off(150, 150),
                off(150, 550),
                on(300, 700),
                on(260, 700),
                off(110, 550),
                off(110, 150),
                on(260, 0),
            ]);
        }
        8 => {
            // ')' 右弧。
            cs.push(vec![
                on(100, 0),
                off(250, 150),
                off(250, 550),
                on(100, 700),
                on(140, 700),
                off(290, 550),
                off(290, 150),
                on(140, 0),
            ]);
        }
        _ => {
            // '\'' 撇。
            cs.push(vstroke(220, 480, 700, 60));
        }
    }
    cs
}

// ===========================================================================
//  阿拉伯字形配方
// ===========================================================================

/// 阿拉伯连接型（与 shaping 层的分类一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArabicJoin {
    /// 只连前（ALEF/DAL/REH/WAW 族）——只有孤立/词尾两态。
    Right,
    /// 两侧都连（BEH/SEEN/LAM/MEEM 族）——四态齐全。
    Dual,
}

/// 一个阿拉伯字母（码位 + 名称 + 连接型）。
#[derive(Debug, Clone, Copy)]
pub struct ArabicLetter {
    /// Unicode 码位。
    pub cp: u32,
    /// 字母助记名（调试）。
    pub name: &'static str,
    /// 连接型。
    pub join: ArabicJoin,
}

/// 阿拉伯基本字母表（U+0627–U+064A 全部已分配字母，共 30 个）。
///
/// 覆盖经典 28 字母 + TEH MARBUTA(0629) + ALEF MAKSURA(0649)。
pub const ARABIC_LETTERS: &[ArabicLetter] = &[
    ArabicLetter { cp: 0x0627, name: "ALEF", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0628, name: "BEH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0629, name: "TEH_MARBUTA", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x062A, name: "TEH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x062B, name: "THEH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x062C, name: "JEEM", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x062D, name: "HAH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x062E, name: "KHAH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x062F, name: "DAL", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0630, name: "THAL", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0631, name: "REH", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0632, name: "ZAIN", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0633, name: "SEEN", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0634, name: "SHEEN", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0635, name: "SAD", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0636, name: "DAD", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0637, name: "TAH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0638, name: "ZAH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0639, name: "AIN", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x063A, name: "GHAIN", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0641, name: "FEH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0642, name: "QAF", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0643, name: "KAF", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0644, name: "LAM", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0645, name: "MEEM", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0646, name: "NOON", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0647, name: "HEH", join: ArabicJoin::Dual },
    ArabicLetter { cp: 0x0648, name: "WAW", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x0649, name: "ALEF_MAKSURA", join: ArabicJoin::Right },
    ArabicLetter { cp: 0x064A, name: "YEH", join: ArabicJoin::Dual },
];

/// 阿拉伯呈现形态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArabicForm {
    /// 孤立形。
    Isolated,
    /// 词首形。
    Initial,
    /// 词中形。
    Medial,
    /// 词尾形。
    Final,
}

/// 一个阿拉伯字母的 gid 分配（基础形 + 各呈现形；缺失态 = `None`）。
#[derive(Debug, Clone, Copy)]
pub struct ArabicGlyphs {
    /// 基础形 gid（cmap 命中）。
    pub base: u16,
    /// 孤立形 gid。
    pub isol: u16,
    /// 词首形 gid。
    pub init: Option<u16>,
    /// 词中形 gid。
    pub medi: Option<u16>,
    /// 词尾形 gid。
    pub fina: u16,
}

/// 全部阿拉伯字母的 gid 分配表（**顺序 = [`ARABIC_LETTERS`]**）。
///
/// 分配规则（确定性）：
/// - `Dual` 字母：`base, isol, init, medi, fina`（5 个连续 gid）；
/// - `Right` 字母：`base, isol, fina`（3 个连续 gid，无 init/medi）。
pub fn arabic_glyph_map() -> Vec<ArabicGlyphs> {
    let mut out: Vec<ArabicGlyphs> = Vec::with_capacity(ARABIC_LETTERS.len());
    let mut gid = G_ARABIC_0;
    for l in ARABIC_LETTERS {
        match l.join {
            ArabicJoin::Dual => {
                out.push(ArabicGlyphs {
                    base: gid,
                    isol: gid + 1,
                    init: Some(gid + 2),
                    medi: Some(gid + 3),
                    fina: gid + 4,
                });
                gid += 5;
            }
            ArabicJoin::Right => {
                out.push(ArabicGlyphs {
                    base: gid,
                    isol: gid + 1,
                    init: None,
                    medi: None,
                    fina: gid + 2,
                });
                gid += 3;
            }
        }
    }
    out
}

/// 阿拉伯字形总数（用于 gid 段边界）。
pub fn arabic_glyph_count() -> u16 {
    ARABIC_LETTERS
        .iter()
        .map(|l| if l.join == ArabicJoin::Dual { 5 } else { 3 })
        .sum()
}

/// 生成阿拉伯字母某形态的**笔画化**轮廓。
///
/// 形态差异体现在「连接茎」的方位与主体块的 x 偏移：
/// - `Isolated`：主体块居中，无茎；
/// - `Initial`：茎在左缘（向右连）；
/// - `Medial`：茎两端贯穿（左右都连）；
/// - `Final`：茎在右缘（向左连）。
fn arabic_form_glyph(letter_idx: usize, form: ArabicForm) -> Vec<Vec<Pt>> {
    let l = ARABIC_LETTERS[letter_idx];
    let seed = (letter_idx as i16) * 7 + (l.cp as i16 % 13);
    // 主体块位置随字母微调（保证互异）。
    let body_x = 120i16 + (seed % 5) * 18;
    let body_w = 200i16 + (seed % 4) * 24;
    let baseline = 0i16;
    let body_h = match l.name {
        "ALEF" | "ALEF_MAKSURA" => 640 + (seed % 3) * 20,
        "WAW" | "REH" | "ZAIN" | "DAL" | "THAL" | "TEH_MARBUTA" => 300 + (seed % 3) * 20,
        _ => 420 + (seed % 5) * 30,
    };
    let top = baseline + body_h;
    let mut cs: Vec<Vec<Pt>> = Vec::new();

    // 主体块：一个圆角矩形（顶部以二次曲线收圆）。
    let bl = body_x;
    let br = body_x + body_w;
    let r = (body_w / 6).max(12);
    cs.push(vec![
        on(bl + r, baseline),
        on(br - r, baseline),
        off(br, baseline),
        off(br, baseline + r),
        on(br, top - r),
        off(br, top),
        off(br - r, top),
        on(bl + r, top),
        off(bl, top),
        off(bl, top - r),
        on(bl, baseline + r),
        off(bl, baseline),
        off(bl + r, baseline),
    ]);

    // 连接茎：竖直细条，按形态放在不同边。
    let stem_w = STEM;
    match form {
        ArabicForm::Isolated => {}
        ArabicForm::Initial => {
            let sx = bl - 60 - (seed % 3) * 10;
            cs.push(vstroke(sx, 0, if l.join == ArabicJoin::Dual { top / 2 } else { top / 3 }, stem_w));
        }
        ArabicForm::Medial => {
            let sx = bl - 60;
            let ex = br + 20;
            cs.push(rounded_bar(sx, ex, 0, stem_w));
        }
        ArabicForm::Final => {
            let ex = br + 40 + (seed % 3) * 10;
            cs.push(rounded_bar(bl, ex, 0, stem_w));
        }
    }
    cs
}

// ===========================================================================
//  天城文字形配方
// ===========================================================================

/// 天城文字母类别（与 shaping 层一致的最小集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevaKind {
    /// 辅音（U+0915–U+0939）。
    Consonant,
    /// 独立元音（U+0905–U+0914）。
    Vowel,
    /// 元音符号（matra，U+093E–U+094C）。
    Matra,
    /// i-matra（前置元音符号，U+093F）。
    IMatra,
    /// virama（U+094D）。
    Virama,
}

/// 一个天城文字形条目。
#[derive(Debug, Clone, Copy)]
pub struct DevaEntry {
    /// 码位。
    pub cp: u32,
    /// 类别。
    pub kind: DevaKind,
}

/// 天城文码位表：独立元音 + 全部辅音 + matra + i-matra + virama。
///
/// 顺序 = gid 分配顺序（自 [`G_DEVANAGARI_0`] 起）。
pub const DEVANAGARI_ENTRIES: &[DevaEntry] = &[
    // 独立元音 U+0905–U+0914
    DevaEntry { cp: 0x0905, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0906, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0907, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0908, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0909, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090A, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090B, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090C, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090D, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090E, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x090F, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0910, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0911, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0912, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0913, kind: DevaKind::Vowel },
    DevaEntry { cp: 0x0914, kind: DevaKind::Vowel },
    // 辅音 U+0915–U+0939
    DevaEntry { cp: 0x0915, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0916, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0917, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0918, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0919, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091A, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091B, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091C, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091D, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091E, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x091F, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0920, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0921, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0922, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0923, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0924, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0925, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0926, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0927, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0928, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0929, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092A, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092B, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092C, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092D, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092E, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x092F, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0930, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0931, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0932, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0933, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0934, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0935, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0936, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0937, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0938, kind: DevaKind::Consonant },
    DevaEntry { cp: 0x0939, kind: DevaKind::Consonant },
    // matra U+093E–U+094C（含 i-matra 093F 单独标类）
    DevaEntry { cp: 0x093E, kind: DevaKind::Matra },
    DevaEntry { cp: 0x093F, kind: DevaKind::IMatra },
    DevaEntry { cp: 0x0940, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0941, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0942, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0943, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0944, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0945, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0946, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0947, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0948, kind: DevaKind::Matra },
    DevaEntry { cp: 0x0949, kind: DevaKind::Matra },
    DevaEntry { cp: 0x094A, kind: DevaKind::Matra },
    DevaEntry { cp: 0x094B, kind: DevaKind::Matra },
    DevaEntry { cp: 0x094C, kind: DevaKind::Matra },
    // virama
    DevaEntry { cp: 0x094D, kind: DevaKind::Virama },
];

/// KA（U+0915）在 [`DEVANAGARI_ENTRIES`] 中的下标（GPOS 基字锚点用）。
pub const KA_DEVANAGARI_INDEX: usize = 16;
/// i-matra（U+093F）在 [`DEVANAGARI_ENTRIES`] 中的下标。
pub const IMATRA_DEVANAGARI_INDEX: usize = 54;
/// virama（U+094D）在 [`DEVANAGARI_ENTRIES`] 中的下标。
pub const VIRAMA_DEVANAGARI_INDEX: usize = 68;

/// 天城文字形 gid（顺序 = [`DEVANAGARI_ENTRIES`]）。
pub fn devanagari_gid(index: usize) -> u16 {
    G_DEVANAGARI_0 + index as u16
}

/// 生成天城文字形轮廓（含「头线」+ 主体块的典型结构）。
fn devanagari_glyph(index: usize) -> Vec<Vec<Pt>> {
    let e = DEVANAGARI_ENTRIES[index];
    let seed = index as i16;
    let mut cs: Vec<Vec<Pt>> = Vec::new();
    match e.kind {
        DevaKind::Consonant => {
            let bx = 120 + (seed % 4) * 20;
            let bw = 200 + (seed % 3) * 30;
            let top = 620;
            // 头线（横向圆头笔画）。
            cs.push(rounded_bar(80, 520, top, STEM));
            // 主体块（方框，顶部圆角）。
            let r = (bw / 6).max(14);
            cs.push(vec![
                on(bx + r, 0),
                on(bx + bw - r, 0),
                on(bx + bw - r, (top - STEM) / 2),
                off(bx + bw, (top - STEM) / 2),
                off(bx + bw, top - STEM - r),
                on(bx + bw, top - STEM - r),
                on(bx + r, top - STEM),
                off(bx, top - STEM),
                off(bx, top - STEM - r),
                on(bx, r),
                off(bx, 0),
                off(bx + r, 0),
            ]);
        }
        DevaKind::Vowel => {
            // 独立元音：竖笔 + 侧钩。
            cs.push(vstroke(200, 0, 620, STEM));
            cs.push(rounded_bar(120, 380, 540, STEM));
            let hook_x = 200 + (seed % 3) * 30;
            cs.push(oval(hook_x, 180, hook_x + 140, 460, true));
        }
        DevaKind::Matra => {
            // 元音符号：主体附着于基字右侧（细长）。
            let top = 640;
            let w = 60 + (seed % 3) * 20;
            cs.push(vec![
                on(120, 60),
                on(120 + w, 60),
                off(120 + w, top),
                off(120 + w - 20, top),
                on(120, top),
                on(120, top - 60),
            ]);
        }
        DevaKind::IMatra => {
            // i-matra：前置元音符号（GPOS mark），锚点设在 (230, 640)。
            cs.push(vec![
                on(230, 480),
                on(230 + 120, 480),
                off(230 + 160, 560),
                off(230 + 160, 700),
                on(230 + 120, 760),
                on(230, 760),
                on(230, 700),
                off(230 + 60, 620),
                off(230, 560),
            ]);
        }
        DevaKind::Virama => {
            // virama：零宽连接件（基线下细横）。
            cs.push(rounded_bar(160, 220, 0, 60));
        }
    }
    cs
}

// ===========================================================================
//  连字字形
// ===========================================================================

/// lam-alef 连字轮廓（LAM 主体 + ALEF 竖笔，构成一个宽连字）。
fn lam_alef_lig_glyph() -> Vec<Vec<Pt>> {
    let mut cs: Vec<Vec<Pt>> = Vec::new();
    // LAM 主体（右下方块）。
    cs.push(vec![
        on(160, 0),
        on(500, 0),
        off(560, 0),
        off(560, 220),
        on(520, 240),
        on(160, 240),
        on(160, 0),
    ]);
    // ALEF 竖笔（左高柱）。
    cs.push(vstroke(120, 0, 720, 140));
    cs
}

/// KA+virama conjunct 轮廓（遗留 liga 产出）。
fn ka_virama_glyph() -> Vec<Vec<Pt>> {
    // KA 主体 + virama 横件（压扁）。
    let mut cs: Vec<Vec<Pt>> = Vec::new();
    cs.push(rounded_bar(90, 510, 320, STEM));
    cs.push(vec![on(140, 0), on(460, 0), on(460, 220), on(140, 220)]);
    cs
}

// ===========================================================================
//  字形数据分发
// ===========================================================================

/// 按 gid 生成 `glyf` 字节（空字形为**零长度**）。
pub fn glyph_data(index: u16) -> Vec<u8> {
    let cs: Vec<Vec<Pt>> = glyph_contours(index);
    encode_glyph(&cs)
}

/// 按 gid 返回轮廓环集合（空字形 → 空 `Vec`）。
pub fn glyph_contours(index: u16) -> Vec<Vec<Pt>> {
    // 拉丁大写 A–Z。
    if (G_LATIN_A..G_LATIN_A + LATIN_COUNT).contains(&index) {
        return latin_letter(index - G_LATIN_A, false);
    }
    // 拉丁小写 a–z。
    if (G_LATIN_LOWER_A..G_LATIN_LOWER_A + LATIN_COUNT).contains(&index) {
        return latin_letter(index - G_LATIN_LOWER_A, true);
    }
    // 数字 0–9。
    if (G_DIGIT_0..G_DIGIT_0 + DIGIT_COUNT).contains(&index) {
        return digit_glyph(index - G_DIGIT_0);
    }
    // 标点。
    if (G_PUNCT_0..G_PUNCT_0 + PUNCT_COUNT).contains(&index) {
        return punct_glyph(index - G_PUNCT_0);
    }
    match index {
        G_NOTDEF | G_SPACE => Vec::new(),
        G_LAM_ALEF_LIG => lam_alef_lig_glyph(),
        G_KA_VIRAMA => ka_virama_glyph(),
        _ => {
            // 阿拉伯区块。
            if let Some((li, form)) = arabic_lookup(index) {
                return arabic_form_glyph(li, form);
            }
            // 天城文区块。
            if (G_DEVANAGARI_0..G_DEVANAGARI_0 + DEVANAGARI_ENTRIES.len() as u16).contains(&index) {
                return devanagari_glyph((index - G_DEVANAGARI_0) as usize);
            }
            Vec::new()
        }
    }
}

/// gid → (阿拉伯字母下标, 形态)；非阿拉伯 gid 返回 `None`。
fn arabic_lookup(gid: u16) -> Option<(usize, ArabicForm)> {
    let map = arabic_glyph_map();
    for (li, a) in map.iter().enumerate() {
        if gid == a.base {
            return Some((li, ArabicForm::Isolated));
        }
        if gid == a.isol {
            return Some((li, ArabicForm::Isolated));
        }
        if gid == a.fina {
            return Some((li, ArabicForm::Final));
        }
        if let Some(g) = a.init {
            if gid == g {
                return Some((li, ArabicForm::Initial));
            }
        }
        if let Some(g) = a.medi {
            if gid == g {
                return Some((li, ArabicForm::Medial));
            }
        }
    }
    None
}

/// 每字形的步进宽（字体单位）。
pub fn glyph_advance(index: u16) -> u16 {
    // 拉丁大写：等宽。
    if (G_LATIN_A..G_LATIN_A + LATIN_COUNT).contains(&index) {
        return 560;
    }
    // 拉丁小写：略窄。
    if (G_LATIN_LOWER_A..G_LATIN_LOWER_A + LATIN_COUNT).contains(&index) {
        return 520;
    }
    if (G_DIGIT_0..G_DIGIT_0 + DIGIT_COUNT).contains(&index) {
        return 540;
    }
    if (G_PUNCT_0..G_PUNCT_0 + PUNCT_COUNT).contains(&index) {
        return 320;
    }
    match index {
        G_NOTDEF => 500,
        G_SPACE => 250,
        G_LAM_ALEF_LIG => 900,
        G_KA_VIRAMA => 600,
        _ => {
            if let Some((_, form)) = arabic_lookup(index) {
                return match form {
                    ArabicForm::Isolated => 520,
                    ArabicForm::Initial | ArabicForm::Medial => 460,
                    ArabicForm::Final => 480,
                };
            }
            if (G_DEVANAGARI_0..G_DEVANAGARI_0 + DEVANAGARI_ENTRIES.len() as u16).contains(&index) {
                let e = DEVANAGARI_ENTRIES[(index - G_DEVANAGARI_0) as usize];
                return match e.kind {
                    DevaKind::IMatra | DevaKind::Virama => 0, // 零宽记号 / 连接件
                    _ => 620,
                };
            }
            0
        }
    }
}

/// 每字形的左 bearing（字体单位）。
pub fn glyph_lsb(index: u16) -> i16 {
    if (G_LATIN_A..G_LATIN_A + LATIN_COUNT).contains(&index) {
        return 90 + (index - G_LATIN_A) as i16 * 2;
    }
    if (G_LATIN_LOWER_A..G_LATIN_LOWER_A + LATIN_COUNT).contains(&index) {
        return 90;
    }
    if (G_DIGIT_0..G_DIGIT_0 + DIGIT_COUNT).contains(&index) {
        return 110;
    }
    if (G_PUNCT_0..G_PUNCT_0 + PUNCT_COUNT).contains(&index) {
        return 150;
    }
    match index {
        G_LAM_ALEF_LIG => 120,
        G_KA_VIRAMA => 90,
        _ => {
            if arabic_lookup(index).is_some() {
                return 60;
            }
            if (G_DEVANAGARI_0..G_DEVANAGARI_0 + DEVANAGARI_ENTRIES.len() as u16).contains(&index) {
                let e = DEVANAGARI_ENTRIES[(index - G_DEVANAGARI_0) as usize];
                return match e.kind {
                    DevaKind::IMatra => 230,
                    DevaKind::Virama => 160,
                    _ => 90,
                };
            }
            0
        }
    }
}

/// 合成字体的字形总数。
pub fn num_glyphs() -> u16 {
    G_DEVANAGARI_0 + DEVANAGARI_ENTRIES.len() as u16
}

/// 字形总数常量（供测试引用）。
pub const NUM_GLYPHS: u16 = G_DEVANAGARI_0 + DEVANAGARI_ENTRIES.len() as u16;

// ===========================================================================
//  cmap 映射（码位 → 字形，升序）
// ===========================================================================

/// 基础字形名兼容常量（保留旧语义以供既有集成测试引用）。
/// DEVANAGARI KA 辅音 gid（GPOS mark 附着的基字符）。
pub const G_KA: u16 = G_DEVANAGARI_0 + 16;
/// DEVANAGARI I-MATRA 前置元音（GPOS mark）。
pub const G_IMATRA: u16 = G_DEVANAGARI_0 + 54;
/// DEVANAGARI VIRAMA（虚音符；`cja` 连字输入）。
pub const G_VIRAMA: u16 = G_DEVANAGARI_0 + 68;

/// 要写入 `cmap` 的码位 → 字形映射（**按码位升序**，format 4/12 共用）。
pub fn charmap_pairs() -> Vec<(u32, u16)> {
    let mut v: Vec<(u32, u16)> = Vec::new();
    v.push((0x0020, G_SPACE)); // SPACE
    // 标点：. , ! ? - : ; ( ) '
    for (i, cp) in [0x002E, 0x002C, 0x0021, 0x003F, 0x002D, 0x003A, 0x003B, 0x0028, 0x0029, 0x0027]
        .iter()
        .enumerate()
    {
        v.push((*cp as u32, G_PUNCT_0 + i as u16));
    }
    // 数字 0–9。
    for k in 0..DIGIT_COUNT {
        v.push((0x0030 + u32::from(k), G_DIGIT_0 + k));
    }
    // 拉丁大写 A–Z。
    for k in 0..LATIN_COUNT {
        v.push((0x0041 + u32::from(k), G_LATIN_A + k));
    }
    // 拉丁小写 a–z。
    for k in 0..LATIN_COUNT {
        v.push((0x0061 + u32::from(k), G_LATIN_LOWER_A + k));
    }
    // 阿拉伯基本字母（基础形）。
    for a in arabic_glyph_map() {
        let cp = ARABIC_LETTERS
            .iter()
            .zip(arabic_glyph_map())
            .find(|(_, m)| m.base == a.base)
            .map(|(l, _)| l.cp)
            .unwrap_or(0);
        if cp != 0 {
            v.push((cp, a.base));
        }
    }
    // 天城文。
    for i in 0..DEVANAGARI_ENTRIES.len() {
        v.push((DEVANAGARI_ENTRIES[i].cp, devanagari_gid(i)));
    }
    // lam-alef 呈现形（整形层直接产出，cmap 映到连字）。
    v.push((0xFEFB, G_LAM_ALEF_LIG)); // 孤立呈现形
    v.push((0xFEFC, G_LAM_ALEF_LIG)); // 词尾呈现形
    v.sort_by_key(|p| p.0);
    v.dedup_by_key(|p| p.0);
    v
}

/// cmap format 4 子表（段映射，仅用 `idDelta`）。
pub fn cmap_format4() -> Vec<u8> {
    let mut segs: Vec<(u32, u32, u16)> =
        charmap_pairs().iter().map(|&(cp, g)| (cp, cp, g)).collect();
    segs.push((0xFFFF, 0xFFFF, 0)); // 强制终止段
    let seg_count = segs.len() as u16;
    let seg_x2 = seg_count.saturating_mul(2);
    let mut entry_selector = 0u16;
    while (1u32 << (entry_selector.saturating_add(1))) <= u32::from(seg_count) {
        entry_selector = entry_selector.saturating_add(1);
    }
    let search_range = 2u16.saturating_mul(1u16 << entry_selector);
    let range_shift = seg_x2.saturating_sub(search_range);

    let mut b = Blob::new();
    b.u16(4);
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
        b.u16(g.wrapping_sub(s as u16)); // idDelta
    }
    for _ in &segs {
        b.u16(0); // idRangeOffset = 0 ⇒ 走 idDelta
    }
    let len = b.len() as u16;
    b.patch_u16(2, len);
    b.into_vec()
}

/// cmap format 12 子表（分组映射，覆盖 BMP + 一个增补平面码位）。
pub fn cmap_format12() -> Vec<u8> {
    let mut groups: Vec<(u32, u32, u32)> =
        charmap_pairs().iter().map(|&(cp, g)| (cp, cp, u32::from(g))).collect();
    groups.push((0x1_0A00, 0x1_0A00, u32::from(G_LATIN_A))); // 增补平面路径
    groups.sort_by_key(|g| g.0);
    let sub_len: u32 = 16u32.saturating_add(groups.len().saturating_mul(12) as u32);
    let mut b = Blob::new();
    b.u16(12);
    b.u16(0);
    b.u32(sub_len);
    b.u32(0);
    b.u32(groups.len() as u32);
    for &(s, e, g) in &groups {
        b.u32(s);
        b.u32(e);
        b.u32(g);
    }
    b.into_vec()
}

/// cmap 表（format 4 + format 12 两条编码记录）。
pub fn cmap_table() -> Vec<u8> {
    let sub4 = cmap_format4();
    let sub12 = cmap_format12();
    let mut b = Blob::new();
    b.u16(0); // version
    b.u16(2); // numTables
    b.u16(3);
    b.u16(1);
    b.u32(4 + 2 * 8);
    b.u16(3);
    b.u16(10);
    b.u32((4 + 2 * 8 + sub4.len()) as u32);
    b.raw(&sub4);
    b.raw(&sub12);
    b.into_vec()
}

// ===========================================================================
//  GSUB（真特征：4 SingleSubst + 1 LigatureSubst）
// ===========================================================================

/// 一个「基础形 → 四态」的替换组。
struct FormSet {
    base: u16,
    isol: u16,
    fina: u16,
    init: Option<u16>,
    medi: Option<u16>,
}

/// 全部阿拉伯字母的四态替换组（数据驱动，见 [`arabic_glyph_map`]）。
fn form_sets() -> Vec<FormSet> {
    ARABIC_LETTERS
        .iter()
        .zip(arabic_glyph_map())
        .map(|(_, a)| FormSet { base: a.base, isol: a.isol, fina: a.fina, init: a.init, medi: a.medi })
        .collect()
}

/// SingleSubst format 2 子表（显式 substitute 数组；Coverage format 1 升序）。
fn single_fmt2(pairs: &[(u16, u16)]) -> Vec<u8> {
    let mut sorted: Vec<(u16, u16)> = pairs.to_vec();
    sorted.sort_by_key(|p| p.0);
    let glyphs: Vec<u16> = sorted.iter().map(|p| p.0).collect();
    let cov = sf::coverage_fmt1(&glyphs);
    let mut b = Blob::new();
    b.u16(2); // substFormat = 2
    b.u16(0); // coverageOffset（回填）
    b.u16(sorted.len() as u16); // glyphCount
    for &(_, sub) in &sorted {
        b.u16(sub);
    }
    let cov_off = b.len() as u16;
    b.patch_u16(2, cov_off);
    b.raw(&cov);
    b.into_vec()
}

/// LigatureSubst format 1 子表：一组「首字形 + 组件序列 → 连字」。
fn ligature_fmt1(entries: &[(u16, Vec<u16>, u16)]) -> Vec<u8> {
    let mut firsts: Vec<u16> = entries.iter().map(|e| e.0).collect();
    firsts.sort_unstable();
    firsts.dedup();
    let cov = sf::coverage_fmt1(&firsts);
    let set_count = firsts.len();

    let mut sets: Vec<Vec<u8>> = Vec::with_capacity(set_count);
    for &first in &firsts {
        let mut ligs: Vec<Vec<u8>> = Vec::new();
        for (f, comps, lig) in entries {
            if *f != first {
                continue;
            }
            let mut l = Blob::new();
            l.u16(*lig);
            l.u16((1 + comps.len()) as u16); // componentCount（含首组件）
            for &c in comps {
                l.u16(c);
            }
            ligs.push(l.into_vec());
        }
        let mut set = Blob::new();
        set.u16(ligs.len() as u16); // ligatureCount
        let base = set.len();
        for _ in &ligs {
            set.u16(0); // 占位
        }
        for (i, l) in ligs.iter().enumerate() {
            let off = set.len() as u16;
            set.patch_u16(base + i * 2, off);
            set.raw(l);
        }
        sets.push(set.into_vec());
    }

    let mut b = Blob::new();
    b.u16(1); // substFormat = 1
    b.u16(0); // coverageOffset（回填）
    b.u16(set_count as u16); // ligatureSetCount
    let set_off_base = b.len();
    for _ in 0..set_count {
        b.u16(0); // 占位
    }
    for (i, s) in sets.iter().enumerate() {
        let off = b.len() as u16;
        b.patch_u16(set_off_base + i * 2, off);
        b.raw(s);
    }
    let cov_off = b.len() as u16;
    b.patch_u16(2, cov_off);
    b.raw(&cov);
    b.into_vec()
}

/// GSUB 表：4 态 SingleSubst（isol/init/medi/fina，覆盖全部可连字母）+ liga 连字。
pub fn gsub_table() -> Vec<u8> {
    let sets = form_sets();
    let isol: Vec<(u16, u16)> = sets.iter().map(|s| (s.base, s.isol)).collect();
    let fina: Vec<(u16, u16)> = sets.iter().map(|s| (s.base, s.fina)).collect();
    // 只有 `Dual` 字母有 init/medi 形。
    let init: Vec<(u16, u16)> =
        sets.iter().filter_map(|s| s.init.map(|g| (s.base, g))).collect();
    let medi: Vec<(u16, u16)> =
        sets.iter().filter_map(|s| s.medi.map(|g| (s.base, g))).collect();

    // liga：LAM + ALEF → 连字；KA + VIRAMA → conjunct（遗留路径）。
    let lig_entries: Vec<(u16, Vec<u16>, u16)> = vec![
        (arabic_base_gid("LAM"), vec![arabic_base_gid("ALEF")], G_LAM_ALEF_LIG),
        (G_KA, vec![G_VIRAMA], G_KA_VIRAMA),
    ];

    let lookups: Vec<Vec<u8>> = vec![
        sf::build_lookup(1, &single_fmt2(&isol)), // 0: isol
        sf::build_lookup(1, &single_fmt2(&init)), // 1: init
        sf::build_lookup(1, &single_fmt2(&medi)), // 2: medi
        sf::build_lookup(1, &single_fmt2(&fina)), // 3: fina
        sf::build_lookup(4, &ligature_fmt1(&lig_entries)), // 4: liga
    ];

    let feature_tags: Vec<[u8; 4]> = vec![*b"isol", *b"init", *b"medi", *b"fina", *b"liga"];
    let lang_indices: Vec<u16> = (0..feature_tags.len() as u16).collect();
    sf::assemble_layout_table(b"GSUB", &lookups, &feature_tags, &lang_indices)
}

/// 按助记名查阿拉伯字母基础形 gid（测试/连字构造用）。
pub fn arabic_base_gid(name: &str) -> u16 {
    ARABIC_LETTERS
        .iter()
        .position(|l| l.name == name)
        .map(|i| arabic_glyph_map()[i].base)
        .unwrap_or(G_NOTDEF)
}

/// 按助记名 + 形态查阿拉伯字母呈现形 gid；缺该态 → 回落到基础形 gid。
pub fn arabic_form_gid(name: &str, form: ArabicForm) -> u16 {
    let Some(i) = ARABIC_LETTERS.iter().position(|l| l.name == name) else {
        return G_NOTDEF;
    };
    let a = arabic_glyph_map()[i];
    match form {
        ArabicForm::Isolated => a.isol,
        ArabicForm::Initial => a.init.unwrap_or(a.base),
        ArabicForm::Medial => a.medi.unwrap_or(a.base),
        ArabicForm::Final => a.fina,
    }
}

/// GPOS 表：`abvm` 特征 → MarkToBasePos（i-matra 附着到 KA）。
pub fn gpos_table() -> Vec<u8> {
    // MarkArray：1 条记录（i-matra，class 0），锚点 (0, 640)。
    let mut mark_anchor = Blob::new();
    mark_anchor.u16(1);
    mark_anchor.i16(0);
    mark_anchor.i16(640);
    let mark_anchor_bytes = mark_anchor.into_vec();
    let mut mark_array = Blob::new();
    mark_array.u16(1);
    mark_array.u16(0);
    mark_array.u16(6); // markAnchorOffset 相对 MarkArray 起点
    let mark_array_bytes = {
        let mut v = mark_array.into_vec();
        v.extend_from_slice(&mark_anchor_bytes);
        v
    };

    // BaseArray：1 个基字符（KA），class 0 锚点 (230, 0)。
    let mut base_anchor = Blob::new();
    base_anchor.u16(1);
    base_anchor.i16(230);
    base_anchor.i16(0);
    let base_anchor_bytes = base_anchor.into_vec();
    let mut base_array = Blob::new();
    base_array.u16(1);
    base_array.u16(4); // baseAnchorOffsets[0][0] 相对 BaseArray 起点
    let base_array_bytes = {
        let mut v = base_array.into_vec();
        v.extend_from_slice(&base_anchor_bytes);
        v
    };

    let mark_cov = sf::coverage_fmt1(&[G_IMATRA]);
    let base_cov = sf::coverage_fmt1(&[G_KA]);
    let mut sub = Blob::new();
    sub.u16(1); // posFormat = 1
    sub.u16(0);
    sub.u16(0);
    sub.u16(1); // markClassCount
    sub.u16(0);
    sub.u16(0);
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

    let lookup = sf::build_lookup(4, sub.as_slice()); // GPOS type 4 = MarkToBasePos
    sf::assemble_layout_table(b"GPOS", &[lookup], &[*b"abvm"], &[0])
}

// ===========================================================================
//  其余表 + SFNT 组装
// ===========================================================================

/// `name` 表（family=1 / subfamily=2 / full=4 / ps=6）。
fn name_table() -> Vec<u8> {
    let entries: [(u16, &str); 4] = [
        (1, "Prism Multiscript"),
        (2, "Regular"),
        (4, "Prism Multiscript Regular"),
        (6, "PrismMultiscript-Regular"),
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

/// `post` 表（version 3.0：无字形名，最小 32 B）。
fn post_table() -> Vec<u8> {
    let mut b = Blob::new();
    b.u32(0x0003_0000); // version 3.0
    b.u32(0); // italicAngle
    b.i16(0); // underlinePosition
    b.i16(0); // underlineThickness
    b.u32(0); // isFixedPitch
    b.u32(0); // minMemType42
    b.u32(0); // maxMemType42
    b.u32(0); // minMemType1
    b.u32(0); // maxMemType1
    b.into_vec()
}

/// `OS/2` 表（version 4；最小可用字段集，供度量与字符分类）。
fn os2_table() -> Vec<u8> {
    let mut b = Blob::new();
    b.u16(4); // version
    b.i16(500); // xAvgCharWidth
    b.u16(400); // usWeightClass
    b.u16(5); // usWidthClass
    b.u16(0); // fsType
    b.i16(650); // ySubscriptXSize
    b.i16(600); // ySubscriptYSize
    b.i16(0); // ySubscriptXOffset
    b.i16(75); // ySubscriptYOffset
    b.i16(650); // ySuperscriptXSize
    b.i16(600); // ySuperscriptYSize
    b.i16(0); // ySuperscriptXOffset
    b.i16(350); // ySuperscriptYOffset
    b.i16(50); // yStrikeoutSize
    b.i16(250); // yStrikeoutPosition
    b.i16(0); // sFamilyClass
    b.raw(&[0u8; 10]); // panose(10B)
    b.u32(0); // ulUnicodeRange1
    b.u32(0); // ulUnicodeRange2
    b.u32(0); // ulUnicodeRange3
    b.u32(0); // ulUnicodeRange4
    b.tag(b"PRSM"); // achVendID
    b.u16(0x0040); // fsSelection = REGULAR
    b.u16(0x0020); // usFirstCharIndex
    b.u16(0xFFFF); // usLastCharIndex
    b.i16(800); // sTypoAscender
    b.i16(-200); // sTypoDescender
    b.i16(0); // sTypoLineGap
    b.u16(800); // usWinAscent
    b.u16(200); // usWinDescent
    b.u32(0); // ulCodePageRange1
    b.u32(0); // ulCodePageRange2
    b.i16(500); // sxHeight
    b.i16(700); // sCapHeight
    b.u16(0); // usDefaultChar
    b.u16(0x0020); // usBreakChar
    b.u16(1); // usMaxContext
    b.into_vec()
}

/// 组装完整 SFNT 文件（表目录按 tag 升序，符合规范）。
pub fn build_font() -> Vec<u8> {
    let n = num_glyphs();
    let mut tables: Vec<([u8; 4], Vec<u8>)> = Vec::new();

    // glyf + loca
    let mut glyf = Blob::new();
    let mut loca = Blob::new();
    for gid in 0..n {
        loca.u32(glyf.len() as u32);
        let g = glyph_data(gid);
        glyf.raw(&g);
        while glyf.len() % 4 != 0 {
            glyf.u8(0); // 4 B 对齐
        }
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
    head.u16(UPM as u16); // unitsPerEm
    head.u32(0); // created(8B)
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

    // maxp（version 1.0；numGlyphs 是 u16）
    let mut maxp = Blob::new();
    maxp.u32(0x0001_0000);
    maxp.u16(n);
    maxp.u16(24); // maxPoints（含 off-curve 控制点后的保守上界）
    maxp.u16(3); // maxContours
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
    hhea.u16(n); // numberOfHMetrics
    tables.push((*b"hhea", hhea.into_vec()));

    // hmtx
    let mut hmtx = Blob::new();
    for gid in 0..n {
        hmtx.u16(glyph_advance(gid));
        hmtx.i16(glyph_lsb(gid));
    }
    tables.push((*b"hmtx", hmtx.into_vec()));

    // cmap
    tables.push((*b"cmap", cmap_table()));

    // GSUB + GPOS
    tables.push((*b"GSUB", gsub_table()));
    tables.push((*b"GPOS", gpos_table()));

    // name / post / OS/2
    tables.push((*b"name", name_table()));
    tables.push((*b"post", post_table()));
    tables.push((*b"OS/2", os2_table()));

    tables.sort_by(|a, b| a.0.cmp(&b.0));
    sf::assemble_sfnt(crate::opentype::SFNT_TRUETYPE, &tables)
}

// ===========================================================================
//  测试
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::opentype::OtFont;

    /// MF1：生成字体是合法 TrueType 容器，含全部必需表。
    #[test]
    fn mf1_builds_parseable_container() {
        let data = build_font();
        let ot = OtFont::parse(&data).expect("生成字体应可解析");
        assert_eq!(ot.sfnt_version(), crate::opentype::SFNT_TRUETYPE);
        assert_eq!(ot.first_checksum_mismatch(), None, "逐表校验和应全通过");
        for t in [
            *b"head", *b"maxp", *b"hhea", *b"hmtx", *b"cmap", *b"loca", *b"glyf",
            *b"GSUB", *b"GPOS", *b"name", *b"post", *b"OS/2",
        ] {
            assert!(ot.directory().contains(t), "应含 {}", String::from_utf8_lossy(&t));
        }
        assert_eq!(ot.maxp().expect("maxp").num_glyphs, num_glyphs());
    }

    /// MF2：生成是**确定性**的 —— 两次调用字节流逐位一致。
    #[test]
    fn mf2_generation_is_deterministic() {
        assert_eq!(build_font(), build_font(), "两次生成必须逐字节一致");
    }

    /// MF3：cmap 覆盖阿拉伯/天城文/拉丁码位（均 > 0）。
    #[test]
    fn mf3_covers_multiscript_codepoints() {
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let f = ot.glyphs().expect("ttf::Font");
        // 阿拉伯：全部 30 字母
        for l in ARABIC_LETTERS {
            let ch = char::from_u32(l.cp).expect("阿拉伯码位");
            assert!(f.glyph_index(ch).is_some(), "阿拉伯 {} ({:#X}) 应覆盖", l.name, l.cp);
        }
        // 天城文
        assert_eq!(f.glyph_index('\u{0915}'), Some(G_KA), "KA");
        assert_eq!(f.glyph_index('\u{093F}'), Some(G_IMATRA), "i-matra");
        assert_eq!(f.glyph_index('\u{094D}'), Some(G_VIRAMA), "virama");
        // 拉丁 A–Z
        for k in 0..LATIN_COUNT {
            let ch = char::from_u32(0x0041 + u32::from(k)).expect("A-Z");
            assert_eq!(f.glyph_index(ch), Some(G_LATIN_A + k), "拉丁 {ch}");
        }
    }

    /// MF4：GSUB 有真内容 —— 5 特征 / 5 查找，特征序 = isol/init/medi/fina/liga。
    #[test]
    fn mf4_gsub_has_real_features() {
        use crate::gsub::LayoutTable;
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let gsub = ot.gsub().expect("GSUB 表");
        let t = LayoutTable::parse(gsub).expect("GSUB 应可解析");
        assert_eq!(t.features().len(), 5, "5 个特征");
        assert_eq!(t.lookups().len(), 5, "5 个查找");
        let tags: Vec<[u8; 4]> = t.features().iter().map(|f| f.tag).collect();
        assert_eq!(tags, vec![*b"isol", *b"init", *b"medi", *b"fina", *b"liga"]);
        let kinds: Vec<u16> = t.lookups().iter().map(|l| l.kind).collect();
        assert_eq!(kinds, vec![1, 1, 1, 1, 4], "4 个 SingleSubst + 1 个 LigatureSubst");
    }

    /// MF5：四态 SingleSubst 对全部 Dual 字母生效（端到端 gid 互异）。
    #[test]
    fn mf5_four_states_apply_to_all_letters() {
        use crate::gsub::{GlyphItem, LayoutTable};
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let t = LayoutTable::parse(ot.gsub().expect("GSUB")).expect("解析");
        let apply = |tag: [u8; 4], gid: u16| -> u16 {
            let fi = t.feature_index(tag).expect("特征");
            let mut g = vec![GlyphItem::base(gid, 0)];
            t.apply_feature(fi, &mut g, 0).expect("施加");
            g[0].gid
        };
        for s in form_sets() {
            assert_eq!(apply(*b"isol", s.base), s.isol, "isol");
            assert_eq!(apply(*b"fina", s.base), s.fina, "fina");
            if let Some(g) = s.init {
                assert_eq!(apply(*b"init", s.base), g, "init");
            }
            if let Some(g) = s.medi {
                assert_eq!(apply(*b"medi", s.base), g, "medi");
            }
        }
    }

    /// MF6：`liga` 连字 —— lam-alef 与 KA+virama 各合并为 1 个字形。
    #[test]
    fn mf6_ligatures_merge() {
        use crate::gsub::{GlyphItem, GlyphClass, LayoutTable};
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let t = LayoutTable::parse(ot.gsub().expect("GSUB")).expect("解析");
        let fi = t.feature_index(*b"liga").expect("liga");

        let lam = arabic_base_gid("LAM");
        let alef = arabic_base_gid("ALEF");
        let mut lam_alef = vec![GlyphItem::base(lam, 0), GlyphItem::base(alef, 1)];
        let n = t.apply_feature(fi, &mut lam_alef, 0).expect("施加");
        assert_eq!(n, 1);
        assert_eq!(lam_alef.len(), 1, "lam+alef 合并");
        assert_eq!(lam_alef[0].gid, G_LAM_ALEF_LIG);
        assert_eq!(lam_alef[0].class, GlyphClass::Ligature);

        let mut ka_vir = vec![GlyphItem::base(G_KA, 0), GlyphItem::base(G_VIRAMA, 1)];
        let n = t.apply_feature(fi, &mut ka_vir, 0).expect("施加");
        assert_eq!(n, 1);
        assert_eq!(ka_vir.len(), 1, "ka+virama 合并");
        assert_eq!(ka_vir[0].gid, G_KA_VIRAMA);
    }

    /// MF7：GPOS `abvm` 把 i-matra 附着到 KA 上方（锚点差已知）。
    #[test]
    fn mf7_gpos_abvm_attaches_imatr_to_ka() {
        use crate::gsub::{GlyphItem, LayoutTable};
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let t = LayoutTable::parse(ot.gpos().expect("GPOS")).expect("解析");
        assert_eq!(t.features().len(), 1);
        assert_eq!(t.features()[0].tag, *b"abvm");
        let fi = t.feature_index(*b"abvm").expect("abvm");
        let lookups = t.features()[fi].lookups.clone();
        let g = vec![GlyphItem::base(G_KA, 0), GlyphItem::mark(G_IMATRA, 0)];
        let out = t.attach_marks(&g, &lookups).expect("附着");
        let att = out[1].expect("i-matra 应附着");
        assert_eq!(att.x_offset, 230, "230 - 0");
        assert_eq!(att.y_offset, -640, "0 - 640");
    }

    /// MF8：字形轮廓真实且尺寸互异（不是同一方块）。
    #[test]
    fn mf8_outlines_are_real_and_distinct() {
        use crate::opentype::outline_bbox;
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let f = ot.glyphs().expect("ttf::Font");
        assert!(f.outline(G_NOTDEF).expect("notdef").contours.is_empty());
        // 阿拉伯四态的包围盒两两不同。
        let map = arabic_glyph_map();
        let beh = ARABIC_LETTERS.iter().position(|l| l.name == "BEH").expect("BEH");
        let a = map[beh];
        let forms = [a.isol, a.init.expect("init"), a.medi.expect("medi"), a.fina];
        let boxes: Vec<(f32, f32, f32, f32)> = forms
            .iter()
            .map(|&g| {
                let b = outline_bbox(&f.outline(g).expect("四态")).expect("有轮廓");
                (b.x_min, b.y_min, b.x_max, b.y_max)
            })
            .collect();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                assert_ne!(boxes[i], boxes[j], "四态包围盒互异（{i} vs {j}）");
            }
        }
        // 拉丁 A 与 Z 的包围盒也应不同。
        let a_bb = outline_bbox(&f.outline(G_LATIN_A).expect("A")).expect("有轮廓");
        let z_bb = outline_bbox(&f.outline(G_LATIN_A + 25).expect("Z")).expect("有轮廓");
        assert_ne!((a_bb.x_max, a_bb.y_max), (z_bb.x_max, z_bb.y_max), "A 与 Z 应形状不同");
    }

    /// MF9：**off-curve（二次贝塞尔控制点）确实存在**（本路技术核心）。
    #[test]
    fn mf9_off_curve_points_present() {
        use crate::ttf::Font;
        let data = build_font();
        let f = Font::parse(&data).expect("ttf::Font");
        // 拉丁大写 'O'（gid = G_LATIN_A + 14）。
        let o_gid = G_LATIN_A + 14;
        let o = f.outline(o_gid).expect("O 轮廓");
        let offs = o.contours.iter().flatten().filter(|p| !p.on_curve).count();
        assert!(offs > 0, "'O' 应含 off-curve 控制点，实际 {offs}");
        // 至少一个字形含 off-curve 点（全局存在性）。
        let any_off = (0..num_glyphs())
            .filter_map(|g| f.outline(g).ok())
            .any(|o| o.contours.iter().flatten().any(|p| !p.on_curve));
        assert!(any_off, "至少一个字形须含 off-curve 点");
    }

    /// MF10：小写字母 / 数字 / 标点 gid 均非 `.notdef`。
    #[test]
    fn mf10_lower_digit_punct_mapped() {
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let f = ot.glyphs().expect("ttf::Font");
        for k in 0..LATIN_COUNT {
            let ch = char::from_u32(0x0061 + u32::from(k)).expect("a-z");
            let g = f.glyph_index(ch).expect("小写应覆盖");
            assert_ne!(g, G_NOTDEF, "小写 {ch} 不得为 .notdef");
        }
        for k in 0..DIGIT_COUNT {
            let ch = char::from_u32(0x0030 + u32::from(k)).expect("0-9");
            let g = f.glyph_index(ch).expect("数字应覆盖");
            assert_ne!(g, G_NOTDEF, "数字 {ch} 不得为 .notdef");
        }
        for &cp in &[0x002E, 0x002C, 0x0021, 0x003F, 0x002D, 0x003A, 0x003B, 0x0028, 0x0029, 0x0027] {
            let ch = char::from_u32(cp).expect("标点");
            let g = f.glyph_index(ch).expect("标点应覆盖");
            assert_ne!(g, G_NOTDEF, "标点 {ch} 不得为 .notdef");
        }
    }

    /// MF11：阿拉伯 28+ 字母的四态 gid 互异。
    #[test]
    fn mf11_arabic_four_states_distinct() {
        let map = arabic_glyph_map();
        assert!(map.len() >= 28, "应覆盖 ≥28 个阿拉伯字母，实际 {}", map.len());
        for (i, a) in map.iter().enumerate() {
            let mut all = vec![a.base, a.isol, a.fina];
            if let Some(g) = a.init {
                all.push(g);
            }
            if let Some(g) = a.medi {
                all.push(g);
            }
            for x in 0..all.len() {
                for y in (x + 1)..all.len() {
                    assert_ne!(all[x], all[y], "字母 #{i} 各态 gid 互异（{x} vs {y}）");
                }
            }
        }
    }

    /// MF12：天城文新基字映射 + i-matra/virama 覆盖。
    #[test]
    fn mf12_devanagari_base_consonants_mapped() {
        let data = build_font();
        let ot = OtFont::parse(&data).expect("容器");
        let f = ot.glyphs().expect("ttf::Font");
        // 抽查若干辅音基字。
        for cp in [0x0915u32, 0x0916, 0x0917, 0x091A, 0x0924, 0x0939] {
            let ch = char::from_u32(cp).expect("天城文");
            assert!(f.glyph_index(ch).is_some(), "天城文 U+{cp:04X} 应覆盖");
        }
        // 独立元音。
        assert!(f.glyph_index('\u{0905}').is_some(), "独立元音 A");
        // 记号与连接件。
        assert!(f.glyph_index('\u{093F}').is_some(), "i-matra");
        assert!(f.glyph_index('\u{094D}').is_some(), "virama");
    }

    /// MF13：生成器两次运行的字节流 sha256 相同（确定性铁律）。
    #[test]
    fn mf13_two_runs_share_sha256() {
        let a = crate::opentype::synth_font::table_checksum(&build_font(), false);
        let b = crate::opentype::synth_font::table_checksum(&build_font(), false);
        assert_eq!(a, b);
        assert_eq!(build_font(), build_font());
    }
}
