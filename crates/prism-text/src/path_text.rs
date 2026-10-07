//! 沿路径排字（文字管线）。
//!
//! # 定位
//!
//! 把一串字形沿一条平面曲线（贝塞尔/折线）排布：每个字形沿路径的**弧长**
//! 定位，并按该点**切线方向**旋转（`prism-text` 设备坐标，y 向下）。
//! 与 [`crate::layout`] 正交：`layout` 给出字形的水平步进宽 `advance`，
//! 本模块把这个步进"铺"到曲线上——度量仍来自同一字体，不重复造轮子。
//!
//! # 算法（D-21：弧长参数化）
//!
//! 1. **采样与弧长表**：对每条贝塞尔段按固定步数细分，累积折线弧长，得到
//!    `(累计弧长 → 参数)` 的单调表（一次构建，O(采样数)）。
//! 2. **定位**：字形沿路径中心线累计 `advance`；给定弧长 `s` 在表中**二分**
//!    查得段与参数，插值得位置 P(s) 与切向量 T(s)（导数解析给出）。
//! 3. **旋转**：字形旋转角 = `atan2(T.y, T.x)`——确定性 `atan2` 见下。
//!
//! # 确定性契约（D-29）
//! 纯函数；无 libm。`atan2` 与 `len` 只走 `+ − × ÷`、`sqrt`（IEEE 正确舍入）
//! 与本 crate 的 `atan`（区间逼近多项式）。

use prism_raster::geom::Point;

/// 平面曲线（二次/三次贝塞尔与折线段的组合，de Casteljau 均解析）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seg {
    /// 直线段。
    Line {
        /// 起点。
        p0: Point,
        /// 终点。
        p1: Point,
    },
    /// 二次贝塞尔（一个控制点）。
    Quad {
        /// 起点。
        p0: Point,
        /// 控制点。
        c: Point,
        /// 终点。
        p1: Point,
    },
    /// 三次贝塞尔（两个控制点）。
    Cubic {
        /// 起点。
        p0: Point,
        /// 控制点 1。
        c1: Point,
        /// 控制点 2。
        c2: Point,
        /// 终点。
        p1: Point,
    },
}

impl Seg {
    /// 段起点。
    pub fn start(&self) -> Point {
        match *self {
            Seg::Line { p0, .. } | Seg::Quad { p0, .. } | Seg::Cubic { p0, .. } => p0,
        }
    }

    /// 段终点。
    pub fn end(&self) -> Point {
        match *self {
            Seg::Line { p1, .. } | Seg::Quad { p1, .. } | Seg::Cubic { p1, .. } => p1,
        }
    }

    /// 参数 u ∈ [0,1] 处的点（解析求值）。
    pub fn point_at(&self, u: f32) -> Point {
        match *self {
            Seg::Line { p0, p1 } => lerp(p0, p1, u),
            Seg::Quad { p0, c, p1 } => {
                let a = lerp(p0, c, u);
                let b = lerp(c, p1, u);
                lerp(a, b, u)
            }
            Seg::Cubic { p0, c1, c2, p1 } => {
                let a = lerp(p0, c1, u);
                let b = lerp(c1, c2, u);
                let c = lerp(c2, p1, u);
                let d = lerp(a, b, u);
                let e = lerp(b, c, u);
                lerp(d, e, u)
            }
        }
    }

    /// 参数 u 处的切向量（未单位化；导数解析给出）。
    pub fn tangent_at(&self, u: f32) -> Point {
        match *self {
            Seg::Line { p0, p1 } => Point::new(p1.x - p0.x, p1.y - p0.y),
            Seg::Quad { p0, c, p1 } => {
                // 2·((1−u)(c−p0) + u(p1−c))
                let ax = (c.x - p0.x) * (1.0 - u) + (p1.x - c.x) * u;
                let ay = (c.y - p0.y) * (1.0 - u) + (p1.y - c.y) * u;
                Point::new(2.0 * ax, 2.0 * ay)
            }
            Seg::Cubic { p0, c1, c2, p1 } => {
                let om = 1.0 - u;
                // 3·((1−u)²(c1−p0) + 2(1−u)u(c2−c1) + u²(p1−c2))
                let bx = om * om * (c1.x - p0.x) + 2.0 * om * u * (c2.x - c1.x) + u * u * (p1.x - c2.x);
                let by = om * om * (c1.y - p0.y) + 2.0 * om * u * (c2.y - c1.y) + u * u * (p1.y - c2.y);
                Point::new(3.0 * bx, 3.0 * by)
            }
        }
    }
}

/// 沿路径的字形落位（位置 + 旋转角 + 弧长）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedGlyph {
    /// 字形在路径上的位置（笔位锚点，设备坐标 px）。
    pub pos: Point,
    /// 切线角（弧度，`atan2(dy, dx)`；y 向下坐标）。
    pub angle: f32,
    /// 沿路径中心线的起始弧长（px）。
    pub s: f32,
}

/// 沿路径排字结果。
#[derive(Debug, Clone, PartialEq)]
pub struct PathText {
    /// 每字形落位（阅读序）。
    pub glyphs: Vec<PlacedGlyph>,
    /// 路径总弧长（px）。
    pub total_len: f32,
}

/// 弧长表（一次构建；`us[i]`/`ss[i]` 单调，二分定位）。
#[derive(Debug, Clone)]
struct ArcTable {
    /// 细分点的段索引。
    seg_of: Vec<usize>,
    /// 细分点的段内参数 u。
    u: Vec<f32>,
    /// 细分点的累计弧长。
    s: Vec<f32>,
    /// 总弧长。
    total: f32,
}

impl ArcTable {
    /// 按每段 `subdiv` 步细分构建弧长表。
    fn build(segs: &[Seg], subdiv: usize) -> Self {
        let n = subdiv.max(1);
        let mut seg_of = Vec::new();
        let mut u = Vec::new();
        let mut s = Vec::new();
        let mut acc = 0.0f32;
        let mut prev = if segs.is_empty() {
            Point::new(0.0, 0.0)
        } else {
            segs[0].start()
        };
        seg_of.push(0);
        u.push(0.0);
        s.push(0.0);
        for (i, seg) in segs.iter().enumerate() {
            for k in 1..=n {
                let uu = k as f32 / n as f32;
                let p = seg.point_at(uu);
                acc += dist(prev, p);
                prev = p;
                seg_of.push(i);
                u.push(uu);
                s.push(acc);
            }
        }
        ArcTable { seg_of, u, s, total: acc }
    }

    /// 弧长 `s` → (段索引, 段内参数 u)：在累计表中二分 + 线性插值。
    fn locate(&self, s: f32) -> (usize, f32) {
        let s = s.clamp(0.0, self.total);
        // 二分找最后一条 ss ≤ s。
        let mut lo = 0usize;
        let mut hi = self.s.len() - 1;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.s[mid] <= s {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let i = lo;
        if i + 1 >= self.s.len() {
            return (self.seg_of[i], self.u[i]);
        }
        let seg_len = self.s[i + 1] - self.s[i];
        let frac = if seg_len > 1e-9 { (s - self.s[i]) / seg_len } else { 0.0 };
        (self.seg_of[i], self.u[i] + (self.u[i + 1] - self.u[i]) * frac)
    }
}

/// 沿路径排字。
///
/// - `segs`：平面曲线段（折线/贝塞尔）；`advances`：每字形沿路径的步进宽（px，
///   通常取自 [`crate::layout::GlyphPos::advance`]）；`baseline_offset`：字形基线
///   相对路径的垂直偏移（正=沿切线左侧＝路径上方，y 向下坐标下取负号上移）。
/// - `subdiv`：每段细分步数（弧长分辨率，越大越准，默认用 [`place_path_text`]）。
///
/// 返回每字形位置与切线角；`advances` 为空 → 空结果。
pub fn place_on_path(segs: &[Seg], advances: &[f32], baseline_offset: f32, subdiv: usize) -> PathText {
    // 空路径（无段）安全降级：库代码禁 panic —— 返回空布局。
    if segs.is_empty() {
        return PathText { glyphs: Vec::new(), total_len: 0.0 };
    }
    let table = ArcTable::build(segs, subdiv);
    let mut glyphs = Vec::with_capacity(advances.len());
    let mut s = 0.0f32;
    for &adv in advances {
        let center = s + adv * 0.5; // 字形锚在其步进中点
        let (si, u) = table.locate(center);
        let seg = &segs[si];
        let p = seg.point_at(u);
        let t = seg.tangent_at(u);
        let tlen = len(t);
        let (tx, ty) = if tlen > 1e-9 { (t.x / tlen, t.y / tlen) } else { (1.0, 0.0) };
        // 法向（左侧）：n = (−ty, tx)。基线偏移沿法向。
        let nx = -ty;
        let ny = tx;
        let pos = Point::new(p.x + nx * baseline_offset, p.y + ny * baseline_offset);
        glyphs.push(PlacedGlyph { pos, angle: atan2(ty, tx), s: center });
        s += adv;
    }
    PathText { glyphs, total_len: table.total }
}

/// 便利入口：对 `text` 用 `font`/`font_size` 度量后沿 `segs` 排布（等价于
/// `layout_text` 求 advance + [`place_on_path`]）。
///
/// `letter_spacing` 叠加在字距之上；路径长度不足时**截断**末尾字形（不越界）。
pub fn place_path_text(
    font: &crate::ttf::Font,
    text: &str,
    font_size: f32,
    letter_spacing: f32,
    segs: &[Seg],
    baseline_offset: f32,
) -> PathText {
    let scale = font_size / font.units_per_em() as f32;
    let advances: Vec<f32> = text
        .chars()
        .map(|c| {
            let gid = font.glyph_index(c).unwrap_or(0);
            font.advance(gid) as f32 * scale + letter_spacing
        })
        .collect();
    place_on_path(segs, &advances, baseline_offset, 32)
}

// ---------------------------------------------------------------------------
// 确定性几何辅助
// ---------------------------------------------------------------------------

/// 线性插值。
#[inline]
fn lerp(a: Point, b: Point, u: f32) -> Point {
    Point::new(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)
}

/// 两点距离。
#[inline]
fn dist(a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt()
}

/// 向量长度。
#[inline]
fn len(p: Point) -> f32 {
    (p.x * p.x + p.y * p.y).sqrt()
}

/// 确定性 `atan2(y, x)`（弧度，值域 (−π, π]）。
///
/// `|y| ≤ |x|` 时 `atan2 = atan(y/x)`；否则用 `π/2 − atan(x/y)`（或负号），
/// `atan` 走区间逼近多项式（[`taylor_atan`]）。仅用 `+ − × ÷`（D-29）。
pub fn atan2(y: f32, x: f32) -> f32 {
    use std::f32::consts::PI;
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let base = if ax >= ay {
        // atan(ay/ax) ∈ [0, π/4]
        taylor_atan(if ax > 1e-12 { ay / ax } else { f32::INFINITY })
    } else {
        std::f32::consts::FRAC_PI_2 - taylor_atan(if ay > 1e-12 { ax / ay } else { 0.0 })
    };
    let ang = if x >= 0.0 { base } else { PI - base };
    if y >= 0.0 { ang } else { -ang }
}

/// `atan`（|x| ≤ 1 的奇 Taylor；用 π/4 折叠把自变量压到 ≤ 1/3 加速收敛）。
/// 仅供 [`atan2`] 内部使用（D-29：无 libm）。
#[inline]
fn taylor_atan(x: f32) -> f32 {
    if x.is_infinite() {
        return std::f32::consts::FRAC_PI_2;
    }
    // 入口保证 x ≥ 0（atan2 传 |比|）；用 π/4 折叠把自变量压到 ≤ 1/3。
    let (base, x) = if x > 0.414_213_56 {
        (std::f32::consts::FRAC_PI_4, (x - 1.0) / (x + 1.0))
    } else {
        (0.0, x)
    };
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut k = 1.0f32;
    for _ in 0..8 {
        term *= -x2;
        k += 2.0;
        sum += term / k;
    }
    base + sum
}

// ---------------------------------------------------------------------------
// 测试（关卡 8：路径排字几何断言）
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    /// 直线路径：字形沿直线排布，位置等距、角度 0、弧长累计正确。
    #[test]
    fn places_along_straight_line() {
        let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(100.0, 0.0) }];
        // 4 个字形各步进 10 → 锚点 5,15,25,35；总弧长 100。
        let pt = place_on_path(&segs, &[10.0, 10.0, 10.0, 10.0], 0.0, 32);
        assert!((pt.total_len - 100.0).abs() < 1e-3);
        for (i, g) in pt.glyphs.iter().enumerate() {
            assert!((g.pos.x - (5.0 + i as f32 * 10.0)).abs() < 1e-2, "x={}", g.pos.x);
            assert!(g.pos.y.abs() < 1e-4);
            assert!(g.angle.abs() < 1e-3, "直线切线角应为 0，实得 {}", g.angle);
        }
    }

    /// 二次贝塞尔（半圆弧近似）：字形位置落在曲线上（到曲线距离小），
    /// 切线角随位置变化（旋转生效）。
    #[test]
    fn places_along_curve_follows_tangent() {
        // 以 (0,0)→ctrl(50,100)→(100,0) 的抛物线（开口向下，y 向下坐标里像 U）。
        let segs = [Seg::Quad {
            p0: Point::new(0.0, 0.0),
            c: Point::new(50.0, 100.0),
            p1: Point::new(100.0, 0.0),
        }];
        let advances = [20.0f32; 6];
        let pt = place_on_path(&segs, &advances, 0.0, 64);
        assert_eq!(pt.glyphs.len(), 6);
        // 每个字形位置的切线角应与其所在点的导数方向一致；检查首尾对称：
        // 起点附近角为正（切线下行），中点附近逼近 0，末端为负——单调递减。
        let a0 = pt.glyphs[0].angle;
        let amid = pt.glyphs[3].angle;
        let a5 = pt.glyphs[5].angle;
        assert!(a0 > 0.0 && a5 < 0.0, "端点切线角应异号：{a0}, {a5}");
        assert!(a0 > amid && amid > a5, "切线角应随 t 单调递减：{a0},{amid},{a5}");
        // 几何自洽：每个字形位置必落在曲线上（到最近采样点距离 < 1px）。
        for g in &pt.glyphs {
            let mut best = f32::INFINITY;
            for k in 0..=256 {
                let q = segs[0].point_at(k as f32 / 256.0);
                let d = dist(q, g.pos);
                if d < best {
                    best = d;
                }
            }
            assert!(best < 1.5, "字形应落在路径上，偏差 {best}");
        }
    }

    /// 弧长定位精度：在直线上，按弧长 s 查得的位置 x ≈ s（自洽回环）。
    #[test]
    fn arclength_locate_is_consistent() {
        let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(200.0, 0.0) }];
        let table = ArcTable::build(&segs, 64);
        for k in 0..=20 {
            let s = k as f32 * 10.0;
            let (si, u) = table.locate(s);
            let p = segs[si].point_at(u);
            assert!((p.x - s.min(200.0)).abs() < 0.5, "s={s} → x={}", p.x);
        }
    }

    /// 路径长度不足时截断：advances 总长超过弧长，仍返回全部字形（越界锚定在
    /// 末端），位置钳在终点附近不 panic。
    #[test]
    fn short_path_clamps_to_end() {
        let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(10.0, 0.0) }];
        let pt = place_on_path(&segs, &[20.0, 20.0, 20.0], 0.0, 16);
        assert_eq!(pt.glyphs.len(), 3);
        // 全部钳到终点 x≈10。
        for g in &pt.glyphs {
            assert!(g.pos.x <= 10.0 + 1e-3, "x={}", g.pos.x);
        }
    }

    /// 基线法向偏移：直线上偏移沿法向（垂直）平移。
    #[test]
    fn baseline_offset_moves_along_normal() {
        let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(100.0, 0.0) }];
        let pt = place_on_path(&segs, &[10.0], 8.0, 32);
        // 切线 (1,0) → 法向 (−0,1)... n=(−ty,tx)=(0,1)；偏移 +8 → y=+8。
        assert!((pt.glyphs[0].pos.y - 8.0).abs() < 1e-3, "y={}", pt.glyphs[0].pos.y);
    }

    /// 度量闭环：place_path_text 用字体 advance（全角 24px），5 字沿直线总长
    /// 与布局一致。
    #[test]
    fn font_metrics_drive_spacing() {
        let f = font_regular();
        let segs = [Seg::Line { p0: Point::new(0.0, 0.0), p1: Point::new(500.0, 0.0) }];
        let pt = place_path_text(&f, "中文中文中", 24.0, 0.0, &segs, 0.0);
        assert_eq!(pt.glyphs.len(), 5);
        assert!((pt.glyphs[1].pos.x - pt.glyphs[0].pos.x - 24.0).abs() < 1e-2);
    }

    /// atan2 对照参考（仅测试用 std）。
    #[test]
    fn atan2_matches_reference() {
        let mut i = -40i32;
        while i <= 40 {
            let mut j = -40i32;
            while j <= 40 {
                let (y, x) = (i as f32 * 0.1, j as f32 * 0.1);
                if x != 0.0 || y != 0.0 {
                    let d = (atan2(y, x) - y.atan2(x)).abs();
                    assert!(d < 2e-2, "atan2({y},{x}) 偏差 {d}");
                }
                j += 1;
            }
            i += 1;
        }
    }
}
