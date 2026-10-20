//! 字形光栅化：TrueType 轮廓 → 覆盖率位图 → 合成进帧缓冲。
//!
//! 管线（文字管线末段）：
//! 1. `glyf` 二次轮廓 → [`outline_to_path`]（隐式中点规则补全 on-curve 锚点）；
//! 2. 字体单位 → 设备像素（scale = px/upem，y 轴翻转由仿射 d = −scale 完成）；
//! 3. 路径平铺 + 扫描线光栅化（`prism-raster`，几何精确覆盖率）；
//! 4. 覆盖率量化 u8（RNE：`floor(c·255 + 0.5)`，IEEE 精确）；
//! 5. [`blit_glyph`] 经直通 alpha src-over 合成进 `Fbuf16F`。
//!
//! # 确定性契约
//! 同 gid + 同 px_size 必得逐字节同位图；pen 位置向下取整对齐像素格
//! （亚像素笔位不做重采样——输出是 pen 的阶梯函数，确定且无抖动）。

use prism_io::Fbuf16F;
use prism_raster::geom::{Point, Transform};
use prism_raster::path::PathBuilder;
use prism_raster::scanline::{rasterize, FillRule};

use crate::cache::GlyphCache;
use crate::layout::{layout_text, GlyphPos, TextParams};
use crate::math_symbols::outline_for_gid;
use crate::ttf::{Font, Outline, OutlinePoint};

/// 字形位图（设备像素空间；`alpha` 行主序 ∈ [0,255]）。
///
/// `left`/`top` 为位图左上角相对**笔位/基线**的偏移：
/// 绘制锚点 = (pen_x + left, baseline_y + top)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlyphBitmap {
    /// 位图左上角相对笔位的 x 偏移（像素，可为负）。
    pub left: i32,
    /// 位图左上角相对基线的 y 偏移（像素，通常为负——字形在基线上方）。
    pub top: i32,
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// alpha 覆盖率（行主序，RNE 量化）。
    pub alpha: Vec<u8>,
}

impl GlyphBitmap {
    /// 构造空位图（无墨字形，如空格）。
    pub fn empty() -> Self {
        GlyphBitmap { left: 0, top: 0, width: 0, height: 0, alpha: Vec::new() }
    }

    /// 是否无墨（无像素）。
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// 视觉顺时针旋转 90°（竖排标点转向用）。
    ///
    /// 位图映射 `dst(x', y') = src(h−1−y, x)`（宽高互换）；盒偏移由连续
    /// 坐标变换 `(px, py) → (−py, px)` 推得：`left' = −(top+height)`、
    /// `top' = left`。纯重排，无插值——逐字节确定。
    ///
    /// 字段是 `pub`，调用方可构造任意值；`alpha.len()` 与 `width·height`
    /// 不自洽时按"无法重排"处理，原样返回（不越界读，C18）。
    pub fn rotate_cw(&self) -> GlyphBitmap {
        if self.is_empty() {
            return self.clone();
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let Some(total) = w.checked_mul(h) else {
            return self.clone();
        };
        if total != self.alpha.len() {
            return self.clone(); // 尺寸与数据不自洽（外部构造的畸形位图）
        }
        let mut alpha = vec![0u8; total];
        for y in 0..h {
            for x in 0..w {
                alpha[x * h + (h - 1 - y)] = self.alpha[y * w + x];
            }
        }
        GlyphBitmap {
            // `top + height` 走饱和加/饱和取负：pub字段可被外部置为
            // i32::MIN/MAX，直接相加会溢出 panic。
            left: self.top.saturating_add(self.height as i32).saturating_neg(),
            top: self.left,
            width: self.height,
            height: self.width,
            alpha,
        }
    }
}

/// 覆盖率量化（RNE）：`floor(c·255 + 0.5)` 钳制到 [0,255]。
/// floor 与乘加均为 IEEE 精确操作（非 libm）。
fn quantize_alpha(c: f32) -> u8 {
    let v = (c * 255.0 + 0.5).floor();
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v as u8
    }
}

/// TrueType 二次轮廓 → 路径（隐式中点规则）。
///
/// 规范（TrueType §glyf）：相邻两个 off-curve 点之间插入其中点作为隐式
/// on-curve 锚点；环若全为 off-curve 点，插入的中点即天然提供起点。
/// 展开后序列中 off 点互不相邻——每个 off 点必后随 on 点（成 quad），
/// 环尾悬置的控制点以 quad 闭合回起点。
pub fn outline_to_path(outline: &Outline) -> PathBuilder {
    let mut pb = PathBuilder::new();
    for ring in &outline.contours {
        if ring.is_empty() {
            continue;
        }
        // 展开：off–off 相邻处插中点（on）。
        let n = ring.len();
        let mut pts: Vec<OutlinePoint> = Vec::with_capacity(n + 2);
        for i in 0..n {
            let cur = ring[i];
            let next = ring[(i + 1) % n];
            pts.push(cur);
            if !cur.on_curve && !next.on_curve {
                pts.push(OutlinePoint {
                    x: (cur.x + next.x) * 0.5,
                    y: (cur.y + next.y) * 0.5,
                    on_curve: true,
                });
            }
        }
        // 展开后必存在 on 点（全 off 环的中点即 on）；防御性跳过异常环。
        let start_idx = match pts.iter().position(|p| p.on_curve) {
            Some(i) => i,
            None => continue,
        };
        pb.move_to(Point::new(pts[start_idx].x, pts[start_idx].y));
        let m = pts.len();
        let mut i = (start_idx + 1) % m;
        let mut ctrl: Option<Point> = None;
        while i != start_idx {
            let p = pts[i];
            if p.on_curve {
                match ctrl.take() {
                    Some(c) => pb.quad_to(c, Point::new(p.x, p.y)),
                    None => pb.line_to(Point::new(p.x, p.y)),
                }
            } else {
                // 展开保证 off 不相邻：此处 ctrl 必为 None。
                ctrl = Some(Point::new(p.x, p.y));
            }
            i = (i + 1) % m;
        }
        // 环尾悬置控制点：quad 闭合回起点。
        if let Some(c) = ctrl.take() {
            let sp = pts[start_idx];
            pb.quad_to(c, Point::new(sp.x, sp.y));
        }
        pb.close();
    }
    pb
}

/// 设备窗口**坐标**幅度上界（像素，单侧）。
///
/// `x0/x1/y0/y1` 各自必须落在 `[−cap, +cap]`——超出则 `left/top` 偏移
/// 超出可表示/可用范围，且 `Transform.tx/ty` 失去意义（局部坐标不再从 0 起）。
const MAX_COORD_EDGE: i64 = 4096;

/// 设备窗口**单轴边长**上界（像素）。
///
/// 覆盖率缓冲是 `Vec<f32>`，边长 `e × e` 意味 `4·e²` 字节；4096 即
/// **64 MiB/字形**，已远超任何实际排版需要（正文级字号 ≤ 数百 px）。
/// 此值与 `sdf::MAX_ATLAS_CELL` 同阶，保持两层栅格化的内存上界一致。
///
/// **与 [`MAX_COORD_EDGE`] 分离的原因（缺陷 N4）**：坐标幅度与区间长度是
/// 两个独立量。只约束坐标幅度时边长可达 `2 · MAX_COORD_EDGE = 8192`，
/// 覆盖缓冲膨胀到 256 MiB/字形，且 `width * height ≤ 4096²` 的文档断言
/// 不成立（契约谎言）。二者分离后，面积上界 `MAX_WINDOW_EDGE²` 由构造保证。
const MAX_WINDOW_EDGE: i64 = 4096;

/// 渲染一个字形为覆盖率位图。
///
/// `px_size` 为 em 框像素高（字号）。窗口 = 字形设备 bbox 外扩到整像素格；
/// 仿射 `(a=scale, d=−scale)` 同时完成缩放与 y 轴翻转（字体空间 y 向上）。
///
/// # 边界语义（C18：永不 panic）
///
/// | `px_size` | 行为 |
/// |---|---|
/// | `NaN` / `≤ 0`（含 `−0.0`） | 空位图 |
/// | `+Inf` / 超 [`MAX_COORD_EDGE`] 或 [`MAX_WINDOW_EDGE`] 的窗口 | 空位图（**拒绝**而非钳制，见下） |
/// | 其余 | 正常渲染 |
///
/// 极大有限 `px_size`（1e10 级，远低于 f32 上限）会让浮点→整数**饱和**为
/// `i64::MIN/MAX`，此处在 i64 域做差并与 [`MAX_COORD_EDGE`]（坐标幅度）
/// 及 [`MAX_WINDOW_EDGE`]（边长）比较：超界返回**空位图**。语义选择说明：
/// 窗口尺寸钳到 4096 会让一个"4096 像素宽的'中'字"被截成左上角一块——
/// 静默错图比不画更糟（下游无法区分"太大"与"空"）；而拒绝是可观测、可预期的
/// （`is_empty()` 为真，调用方走"无墨字形"既有分支）。
/// 真正需要超大字号时，正确做法是分段放大 + SDF 双线性重建（`sdf` 模块），
/// 而非让覆盖率栅格化去分配天文数字。
pub fn render_glyph(font: &Font, gid: u16, px_size: f32) -> GlyphBitmap {
    if !(px_size > 0.0) || !px_size.is_finite() {
        return GlyphBitmap::empty();
    }
    // 字体轮廓优先（真实字形逐位不变）；空/越界 gid 时回退到
    // `math_symbols` 程序化符号表（cmap 缺失的希腊/数学符号）。
    let outline = match font.outline(gid) {
        Ok(o) if !o.contours.is_empty() => o,
        _ => match outline_for_gid(gid) {
            Some(o) => o,
            None => return GlyphBitmap::empty(),
        },
    };
    if outline.contours.is_empty() {
        return GlyphBitmap::empty();
    }
    let scale = px_size / font.units_per_em() as f32;

    // 字体单位 bbox（y 向上）。
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for ring in &outline.contours {
        for p in ring {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
    }
    // 设备窗口（基线 y=0，向下为正）：x ∈ [x_min·s, x_max·s]，
    // y ∈ [−y_max·s, −y_min·s]，外扩到整像素格。
    //
    // i64 域计算：浮点→整数是**饱和**转换，`x1 - x0` 在 i32 下会溢出
    // panic（`x1`→`i32::MAX`、`x0`→`i32::MIN`）。经 `f64` 中间量升到 i64
    // 装得下二者之差（≤ 2^53），随后分别与坐标幅度/边长上界比较。
    // 注：`x0/x1/y0/y1` 之间无算术，故此处不需饱和加减；`x1 - x0` 只在
    // 坐标守卫**通过后**才计算（此时二者 ∈ [−cap, cap]，差值 ≤ 2·cap，
    // i64 远 enough，不可能溢出）。
    let x0 = ((x_min * scale).floor() as f64) as i64;
    let x1 = ((x_max * scale).ceil() as f64) as i64;
    let y0 = ((-y_max * scale).floor() as f64) as i64;
    let y1 = ((-y_min * scale).ceil() as f64) as i64;
    let cap = MAX_COORD_EDGE;
    // 窗口两端各自须在 ±cap 内：否则 `left/top` 偏移会超出可表示/可用范围，
    // 且 `Transform.tx/ty` 失去意义（局部坐标不再从 0 起）。
    if x0 < -cap || x1 > cap || y0 < -cap || y1 > cap {
        return GlyphBitmap::empty();
    }
    // 边长另受 `MAX_WINDOW_EDGE` 约束（N4）：坐标幅度守卫只给出
    // `x1 - x0 ≤ 2 · cap` 的推论，边长可达 8192 ⇒ 覆盖缓冲 256 MiB/字形。
    // 拒绝（返回空位图）而非钳制，语义与坐标守卫统一。
    let width_i = (x1 - x0).max(0);
    let height_i = (y1 - y0).max(0);
    if width_i > MAX_WINDOW_EDGE || height_i > MAX_WINDOW_EDGE {
        return GlyphBitmap::empty();
    }
    if width_i == 0 || height_i == 0 {
        return GlyphBitmap::empty();
    }
    // 边长 ≤ MAX_WINDOW_EDGE = 4096 ⇒ `width * height ≤ 4096²`，
    // `Vec` 分配与 `Coverage::new` 的 `w*h` 均在 usize/u32 预算内
    // （不再二次校验）。
    let width = width_i as u32;
    let height = height_i as u32;
    // 字体单位 → 窗口局部像素坐标。`x0/y0 ∈ [−cap, cap]`，`as i32`/`as f32`
    // 皆为精确转换（无饱和、无舍入）。
    let tf = Transform {
        a: scale,
        b: 0.0,
        c: 0.0,
        d: -scale,
        tx: -(x0 as f32),
        ty: -(y0 as f32),
    };
    let path = outline_to_path(&outline);
    let contours = path.flatten(&tf);
    let cov = rasterize(&contours, FillRule::Winding, width, height);
    let alpha = cov.data.iter().map(|&c| quantize_alpha(c)).collect();
    GlyphBitmap { left: x0 as i32, top: y0 as i32, width, height, alpha }
}

/// 把字形位图合成进帧缓冲（直通 alpha src-over；pen 向下取整对齐像素格）。
///
/// 位图字段是 `pub`，外部可构造 `alpha.len() < width·height` 的畸形位图；
/// 此处用 `checked_mul` 预算一次并对每像素下标做 `get`，越界像素跳过而非panic。
pub fn blit_glyph(
    fbuf: &mut Fbuf16F,
    bmp: &GlyphBitmap,
    pen_x: f32,
    baseline_y: f32,
    color: [f32; 4],
) {
    if bmp.is_empty() {
        return;
    }
    let Some(total) = (bmp.width as usize).checked_mul(bmp.height as usize) else {
        return;
    };
    if bmp.alpha.len() < total {
        return;
    }
    // 偏移相加走**饱和加**（C18 / 确定性铁律）：
    // `f32 as i64` 是饱和转换（`±1e30 → i64::MIN/MAX`），`left`/`top` 是
    // `pub` 字段（外部可置 `i32::MIN/MAX`）。裸 `+` 在 debug 下溢 panic，
    // 在 release 静默回绕成"贴边错图"——同输入在不同优化级别下输出不同，
    // 直接违反铁律 2。饱和后 `ox/oy` 必落在视口外，被下方
    // `fx < 0 || fx >= w { continue }` 自然丢弃（笔位在视口外本就不该画）。
    // NaN/Inf 笔位必须显式拒绝。Rust 的 `NaN as i64 == 0`，故饱和转换
    // 会把 NaN **静默归一化为 0** 并照常在原点绘制 —— 不 panic，但会画出一幅
    // 位置完全错误的图。语义上与 `render_glyph` 的 `!px_size.is_finite()`
    // 拒绝口径对齐：非有限笔位视为无效输入，直接早退（等价于"这一笔没画"）。
    if !pen_x.is_finite() || !baseline_y.is_finite() {
        return;
    }
    let ox = (pen_x.floor() as i64).saturating_add(bmp.left as i64);
    let oy = (baseline_y.floor() as i64).saturating_add(bmp.top as i64);
    let bw = bmp.width as i64;
    let bh = bmp.height as i64;
    for row in 0..bh {
        // `ox/oy` 可能是 `i64::MIN/MAX`（饱和结果），`+ row/col` 同样须饱和，
        // 否则此处二次溢出（debug panic / release 错图）。
        let fy = oy.saturating_add(row);
        if fy < 0 || fy >= fbuf.height as i64 {
            continue;
        }
        for col in 0..bw {
            let fx = ox.saturating_add(col);
            if fx < 0 || fx >= fbuf.width as i64 {
                continue;
            }
            let idx = (row as usize) * bmp.width as usize + col as usize;
            let Some(&a) = bmp.alpha.get(idx) else {
                continue;
            };
            if a == 0 {
                continue;
            }
            let sa = a as f32 / 255.0;
            let src = [color[0], color[1], color[2], color[3] * sa];
            let dst = fbuf.get(fx as u32, fy as u32);
            fbuf.set(fx as u32, fy as u32, prism_raster::blend_pixel_over(src, dst));
        }
    }
}

/// 渲染并合成一串已定位字形（经缓存；pen 取整对齐像素格）。
pub fn draw_glyphs(
    fbuf: &mut Fbuf16F,
    font: &Font,
    glyphs: &[GlyphPos],
    size: f32,
    color: [f32; 4],
    dx: f32,
    dy: f32,
    cache: &mut GlyphCache,
) {
    for g in glyphs {
        let bmp = cache.get_or_render(font, g.gid, size, 0);
        blit_glyph(fbuf, bmp, g.x + dx, g.y + dy, color);
    }
}

/// 高层入口：排版 + 渲染一步到位（align/换行由 `params` 决定）。
pub fn draw_text(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    params: &TextParams,
    color: [f32; 4],
    cache: &mut GlyphCache,
) {
    let layout = layout_text(font, text, params);
    draw_glyphs(fbuf, font, &layout.glyphs, params.font_size, color, 0.0, 0.0, cache);
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    #[test]
    fn space_yields_empty_bitmap() {
        let f = font_regular();
        let gid = f.glyph_index(' ').expect("空格在 cmap 中");
        let bmp = render_glyph(&f, gid, 24.0);
        assert!(bmp.is_empty(), "空格无墨");
    }

    #[test]
    fn cjk_bitmap_bbox_and_coverage() {
        let f = font_regular();
        let gid = f.glyph_index('中').expect("中 在子集内");
        let bmp = render_glyph(&f, gid, 32.0);
        assert!(!bmp.is_empty());
        // 全角 32px：bbox 顶在基线上方（top ≤ 0），底不超 em 框（高 ≤ 40）。
        assert!(bmp.top <= 0);
        assert!(bmp.height <= 40 && bmp.width <= 40);
        // 中心区域必有墨（中 字中心有竖笔）。
        let cx = (bmp.width / 2) as usize;
        let cy = (bmp.height / 2) as usize;
        assert!(bmp.alpha[cy * bmp.width as usize + cx] > 200, "中心竖笔应近全黑");
        // 部分边界像素透明（字形不满框）。
        assert_eq!(bmp.alpha[0], 0, "左上角应透明");
    }

    #[test]
    fn render_is_deterministic_byte_wise() {
        let f = font_regular();
        let gid = f.glyph_index('永').expect("永 字法齐全");
        let a = render_glyph(&f, gid, 48.0);
        let b = render_glyph(&f, gid, 48.0);
        assert_eq!(a, b, "同 gid 同字号逐字节一致");
    }

    #[test]
    fn rotate_cw_swaps_dims_and_keeps_mass() {
        let bmp = GlyphBitmap {
            left: -2,
            top: -20,
            width: 4,
            height: 6,
            alpha: (0..24u8).collect(),
        };
        let r = bmp.rotate_cw();
        assert_eq!((r.width, r.height), (6, 4), "宽高互换");
        // 偏移：left' = −(top+height) = −(−20+6) = 14；top' = left = −2。
        assert_eq!(r.left, 14);
        assert_eq!(r.top, -2);
        // 像素质量守恒（纯重排无插值）。
        let sum_src: u32 = bmp.alpha.iter().map(|&v| v as u32).sum();
        let sum_dst: u32 = r.alpha.iter().map(|&v| v as u32).sum();
        assert_eq!(sum_src, sum_dst);
    }

    #[test]
    fn rotate_cw_offsets_match_continuous_transform() {
        // 连续坐标验证：left' = −(top+height)，top' = left。
        let bmp = GlyphBitmap { left: -3, top: -25, width: 10, height: 20, alpha: vec![1; 200] };
        let r = bmp.rotate_cw();
        assert_eq!(r.left, -(-25 + 20));
        assert_eq!(r.top, -3);
        // 像素质量守恒（纯重排）。
        let sum_src: u32 = bmp.alpha.iter().map(|&v| v as u32).sum();
        let sum_dst: u32 = r.alpha.iter().map(|&v| v as u32).sum();
        assert_eq!(sum_src, sum_dst);
        // 角点映射：src 左下 (0, h−1) → dst 左上 (0, 0)。
        assert_eq!(r.alpha[0], bmp.alpha[(20 - 1) * 10 + 0]);
    }

    #[test]
    fn blit_respects_viewport_and_alpha() {
        let mut fbuf = Fbuf16F::new(16, 16);
        let f = font_regular();
        let gid = f.glyph_index('一').expect("一 在子集内");
        let bmp = render_glyph(&f, gid, 16.0);
        blit_glyph(&mut fbuf, &bmp, 4.0, 12.0, [1.0, 0.0, 0.0, 1.0]);
        // 至少一个像素被涂上（横笔必落墨）。
        let painted = (0..16u32).any(|y| (0..16u32).any(|x| fbuf.get(x, y)[3] > 0.0));
        assert!(painted, "横笔字形应产出像素");
        // 视口裁剪：大偏移不 panic 且不产出。
        let mut small = Fbuf16F::new(2, 2);
        blit_glyph(&mut small, &bmp, 100.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
        assert!(small.pixels.iter().all(|&p| p == 0));
    }

    // -----------------------------------------------------------------------
    // N2 回归：`f32 as i32` 饱和转换后的 i32 减法溢出（C18 永不 panic）
    // -----------------------------------------------------------------------

    /// 极大但有限的 `px_size` 全部不 panic，且返回**空位图**（拒绝语义）。
    ///
    /// 旧实现 `let width = (x1 - x0).max(0) as u32`（x0/x1 是 `f32 as i32` 的
    /// 饱和结果）在 `px_size ≳ 1e10` 时 `x1 - x0` 溢出 panic。现改为 i64 域
    /// 计算 + `MAX_WINDOW_EDGE` 上界，超界返回空位图。
    #[test]
    fn huge_finite_px_size_never_panics() {
        let f = font_regular();
        let gid = f.glyph_index('中').expect("中 在子集内");
        for &px in &[1.0e3f32, 1.0e5, 1.0e10, 1.0e20, 1.0e30, 3.0e38, f32::MAX] {
            let bmp = render_glyph(&f, gid, px);
            if px > MAX_WINDOW_EDGE as f32 {
                assert!(bmp.is_empty(), "px_size={px:e} 超上限应返回空位图（拒绝语义）");
            } else {
                assert!(!bmp.is_empty(), "px_size={px:e} 在上限内应正常渲染");
            }
        }
    }

    /// 非正/非有限 `px_size` 一律空位图（`NaN` / `±Inf` / `0` / 负数）。
    #[test]
    fn non_positive_and_non_finite_px_size_yield_empty() {
        let f = font_regular();
        let gid = f.glyph_index('中').expect("中 在子集内");
        for &px in &[0.0f32, -0.0, -1.0, -1e30, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bmp = render_glyph(&f, gid, px);
            assert!(bmp.is_empty(), "px_size={px} 应返回空位图");
            assert_eq!(bmp.alpha.len(), 0, "px_size={px} 不应分配alpha");
        }
    }

    /// 上限边界：恰好等于 `MAX_WINDOW_EDGE` 的整倍字号产出**自洽**位图
    /// （`alpha.len() == width*height`，无 OOM）。
    #[test]
    fn window_edge_cap_yields_self_consistent_bitmap() {
        let f = font_regular();
        let gid = f.glyph_index('中').expect("中 在子集内");
        for &px in &[1024.0f32, 2048.0, 4096.0] {
            let bmp = render_glyph(&f, gid, px);
            if bmp.is_empty() {
                continue;
            }
            assert_eq!(
                bmp.alpha.len(),
                bmp.width as usize * bmp.height as usize,
                "px={px} 位图尺寸不自洽：{}x{} alpha={}",
                bmp.width,
                bmp.height,
                bmp.alpha.len()
            );
            assert!(bmp.width as i64 <= MAX_WINDOW_EDGE);
            assert!(bmp.height as i64 <= MAX_WINDOW_EDGE);
        }
    }

    /// 越界 gid 在 `render_glyph` 层同样返回空位图（N1 影响面）。
    #[test]
    fn out_of_range_gid_yields_empty_bitmap() {
        let f = font_regular();
        for gid in [6889u16, 7000, u16::MAX] {
            let bmp = render_glyph(&f, gid, 32.0);
            assert!(bmp.is_empty(), "gid={gid} 应返回空位图");
        }
    }

    /// 畸形位图（`alpha.len() < width*height`）不得让 `rotate_cw` /
    /// `blit_glyph` 越界读panic（C18：pub 字段可被外部任意构造）。
    #[test]
    fn malformed_bitmap_does_not_panic() {
        let bad = GlyphBitmap { left: 0, top: 0, width: 1000, height: 1000, alpha: vec![7u8; 10] };
        let r = bad.rotate_cw();
        assert_eq!(r.alpha, bad.alpha, "不自洽位图应原样返回");
        let mut fbuf = Fbuf16F::new(8, 8);
        blit_glyph(&mut fbuf, &bad, 0.0, 0.0, [1.0, 1.0, 1.0, 1.0]);
        assert!(fbuf.pixels.iter().all(|&p| p == 0), "畸形位图不应涂任何像素");
    }

    /// `rotate_cw` 的偏移计算走饱和加减（`top = i32::MIN` 不溢出）。
    #[test]
    fn rotate_cw_offset_saturates() {
        // 常规值：i32::MIN + 4 可表示，取负不溢出。
        let bmp = GlyphBitmap {
            left: 0,
            top: i32::MIN,
            width: 4,
            height: 4,
            alpha: vec![3u8; 16],
        };
        assert_eq!(bmp.rotate_cw().left, 2147483644);
        // 真正会溢出的组合：top + height 贴近 i32::MIN，取负须饱和而非 panic。
        let bmp2 = GlyphBitmap { left: 0, top: i32::MIN, width: 1, height: 1, alpha: vec![1u8] };
        assert_eq!(bmp2.rotate_cw().left, i32::MAX, "−(i32::MIN+1) 应饱和为 i32::MAX");
    }
}
