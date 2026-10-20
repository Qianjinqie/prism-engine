//! 坐标系与曲线采样（T04 axes.rs，F2/F3）：
//!
//! - [`axes_tracks`]：视口内的网格 / 主轴 / 箭头 / 刻度 / 轴标签生成器
//!   （静态 `Track` 列表，时间域全 0，供 T05 分镜按需复制改期）；
//! - [`sample_curve`]：函数 `y = f(x)` 在数学区间上均匀采样为设备折线
//!   （视口 x 域裁剪）；
//! - [`riemann_rects`]：Riemann 矩形集（积分域 = 视口 x 域，右端点，
//!   基线贴 `y=0` 或视口边缘，顶 clamp 视口）。
//!
//! 全部像素落位经 [`crate::common::Viewport::to_px`]（唯一
//! 映射处），本文件不出现任何内联坐标映射公式。非法输入（空域 / 点数
//! 不足 / 非有限函数值 / 非法视口）一律返回空而不 panic。

use crate::common::{
    clamp_domain, AXIS_RGBA, GRID_RGBA, HEIGHT, TEXT_RGBA, TICK_RGBA, Viewport, WIDTH,
};
use prism_author::{FontKind, PathTrack, StaticText, Track};

/// 网格步长（数学单位）：0.5（分镜网格密度）。
pub const GRID_STEP: f32 = 0.5;
/// 刻度步长（数学单位）：1.0（整数刻度）。
pub const TICK_STEP: f32 = 1.0;
/// 网格线宽（px）。
pub const GRID_LINE_W: f32 = 2.0;
/// 主轴 / 箭头线宽（px）。
pub const AXIS_LINE_W: f32 = 3.0;
/// 刻度半长（px）。
pub const TICK_HALF_LEN: f32 = 4.0;
/// 箭头 Chevron 半宽（px）。
pub const ARROW_HALF_W: f32 = 5.0;
/// 箭头 Chevron 回退深度（px，翼尖到尖顶的轴向距离）。
pub const ARROW_BACK: f32 = 9.0;
/// 轴标签字号（px）。
pub const AXIS_LABEL_SIZE: f32 = 26.0;
/// 网格 / 刻度线族数量上限（病理视口保护：超出则整族跳过，不 panic）。
pub const MAX_GRID_LINES: u64 = 512;
/// 轴标签距轴端的水平偏移（px）。
const LABEL_DX: f32 = 20.0;
/// 轴标签距轴端的纵向偏移（px）。
const LABEL_DY: f32 = 12.0;
/// 轴标签锚点距画幅边缘的安全边际（px）。
const LABEL_MARGIN: f32 = 8.0;

/// 生成坐标轴静态轨道：网格 → 主轴 → 4 箭头 → 整数刻度 → x/y 轴标签。
///
/// 语义：
/// - 网格：`GRID_STEP` 倍数处满幅直线（白 alpha 0.10）；
/// - 主轴：`y=0` / `x=0` 处直线；原点不在域内时**贴边**
///   （[`clamp_domain`]，轴画在最近域边上，白 alpha 0.90）；
/// - 箭头：两轴各端一枚 Chevron（指向数学正方向）；
/// - 刻度：整数处短刻度（白 alpha 0.75，`TICK_HALF_LEN`）；
/// - 标签：`"x"` / `"y"`（`StaticText`，E3 `center_x` 锚点）。
///
/// 所有轨道 `at / in_sec / out_sec / draw_sec = 0`（静态常显）。视口非法
/// 时返回空列表。
pub fn axes_tracks(vp: &Viewport) -> Vec<Track> {
    let mut tracks: Vec<Track> = Vec::new();
    if !vp.is_valid() {
        return tracks;
    }
    // 轴位置：原点不在域内时贴最近域边（§3.3 贴边语义）。
    let y_axis = clamp_domain(0.0, vp.y0, vp.y1);
    let x_axis = clamp_domain(0.0, vp.x0, vp.x1);

    // 1) 网格（先画，垫在最底层）。
    let mut gx: Vec<f32> = Vec::new();
    multiples(GRID_STEP, vp.x0, vp.x1, &mut gx);
    for x in gx {
        push_math_line(&mut tracks, vp, x, vp.y0, x, vp.y1, GRID_RGBA, GRID_LINE_W);
    }
    let mut gy: Vec<f32> = Vec::new();
    multiples(GRID_STEP, vp.y0, vp.y1, &mut gy);
    for y in gy {
        push_math_line(&mut tracks, vp, vp.x0, y, vp.x1, y, GRID_RGBA, GRID_LINE_W);
    }

    // 2) 主轴（覆盖网格线上的轴线段）。
    push_math_line(&mut tracks, vp, vp.x0, y_axis, vp.x1, y_axis, AXIS_RGBA, AXIS_LINE_W);
    push_math_line(&mut tracks, vp, x_axis, vp.y0, x_axis, vp.y1, AXIS_RGBA, AXIS_LINE_W);

    // 3) 箭头：x 轴左右端、y 轴上下端各一枚 Chevron（数学正方向）。
    let x_right = vp.to_px(vp.x1, y_axis);
    let x_left = vp.to_px(vp.x0, y_axis);
    let y_top = vp.to_px(x_axis, vp.y1); // 数学 y1 → 设备上方
    let y_bottom = vp.to_px(x_axis, vp.y0);
    push_chevron(&mut tracks, x_right, -1.0, 0.0, AXIS_RGBA); // →
    push_chevron(&mut tracks, x_left, 1.0, 0.0, AXIS_RGBA); // ←
    push_chevron(&mut tracks, y_top, 0.0, 1.0, AXIS_RGBA); // ↑（设备 y 向下，反方向为 +y）
    push_chevron(&mut tracks, y_bottom, 0.0, -1.0, AXIS_RGBA); // ↓

    // 4) 刻度：整数处短刻度（贴在主轴上）。
    let mut tx: Vec<f32> = Vec::new();
    multiples(TICK_STEP, vp.x0, vp.x1, &mut tx);
    for x in tx {
        let a = vp.to_px(x, y_axis);
        push_line(
            &mut tracks,
            [a[0], a[1] - TICK_HALF_LEN],
            [a[0], a[1] + TICK_HALF_LEN],
            TICK_RGBA,
            GRID_LINE_W,
        );
    }
    let mut ty: Vec<f32> = Vec::new();
    multiples(TICK_STEP, vp.y0, vp.y1, &mut ty);
    for y in ty {
        let a = vp.to_px(x_axis, y);
        push_line(
            &mut tracks,
            [a[0] - TICK_HALF_LEN, a[1]],
            [a[0] + TICK_HALF_LEN, a[1]],
            TICK_RGBA,
            GRID_LINE_W,
        );
    }

    // 5) 轴标签（E3 center_x 锚点；锚点 clamp 进画幅安全区）。
    let lx = clamp_anchor(x_right[0] + LABEL_DX, x_right[1] - LABEL_DY);
    tracks.push(Track::Static(StaticText {
        content: String::from("x"),
        font: FontKind::Bold,
        size: AXIS_LABEL_SIZE,
        color: TEXT_RGBA,
        at: 0.0,
        in_sec: 0.0,
        letter_spacing: 0.0,
        y: lx[1],
        out_sec: 0.0,
        out_at: None,
        center_x: Some(lx[0]),
    }));
    let ly = clamp_anchor(y_top[0] + LABEL_DX, y_top[1] + LABEL_DY);
    tracks.push(Track::Static(StaticText {
        content: String::from("y"),
        font: FontKind::Bold,
        size: AXIS_LABEL_SIZE,
        color: TEXT_RGBA,
        at: 0.0,
        in_sec: 0.0,
        letter_spacing: 0.0,
        y: ly[1],
        out_sec: 0.0,
        out_at: None,
        center_x: Some(ly[0]),
    }));

    tracks
}

/// 函数 `y = f(x)` 在数学区间 `[x0, x1]` 上均匀采样 `n` 点，经视口映射
/// 为设备折线（`Vec<[f32; 2]>`，可直接喂 `PathTrack` / `DynPath` 的几何）。
///
/// 纪律：
/// - 采样点数学 x 落在视口 x 域之外 → **丢弃**（视口裁剪；折线自然
///   断在域边界，T05 动画几何按需取交集区间即可）；
/// - `f` 返回非有限值 / 视口非法 / `n < 2` / 区间非有限 → 返回空；
/// - y 方向不裁剪（曲线允许视觉上冲出画幅，由光栅层裁掉）。
pub fn sample_curve(
    f: impl Fn(f32) -> f32,
    x0: f32,
    x1: f32,
    n: usize,
    vp: &Viewport,
) -> Vec<[f32; 2]> {
    let mut out: Vec<[f32; 2]> = Vec::new();
    if n < 2 || !vp.is_valid() || !x0.is_finite() || !x1.is_finite() {
        return out;
    }
    let lo = vp.x0.min(vp.x1);
    let hi = vp.x0.max(vp.x1);
    let span = x1 - x0;
    for i in 0..n {
        let mx = x0 + span * (i as f32 / (n - 1) as f32);
        if !mx.is_finite() {
            continue;
        }
        // 视口 x 域裁剪。
        if mx < lo || mx > hi {
            continue;
        }
        let my = f(mx);
        if !my.is_finite() {
            continue;
        }
        let p = vp.to_px(mx, my);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        out.push(p);
    }
    out
}

/// Riemann 矩形集：积分域 = 视口 x 域 `[x0, x1]`，`n` 等分，取**右端点**
/// 函数值为高；基线 = `y=0`（不在域内则贴边，见 [`clamp_domain`]）；
/// 矩形顶 clamp 进视口 y 域（防冲出画幅）。
///
/// 返回每个矩形的四角（设备像素，顺时针）。`n == 0` / 视口非法 / 函数
/// 在右端点返回非有限值（该矩形跳过）/ 映射非有限 → 不计入，不 panic。
///
/// 与 §4.2 签名 `riemann_rects(n, f, vp)` 一致：积分域取视口 x 域——
/// S5 分镜视口 x 域即 `[0, 2]`，无需另传区间。
pub fn riemann_rects(n: u32, f: impl Fn(f32) -> f32, vp: &Viewport) -> Vec<Vec<[f32; 2]>> {
    let mut out: Vec<Vec<[f32; 2]>> = Vec::new();
    if n == 0 || !vp.is_valid() {
        return out;
    }
    let lo = vp.x0.min(vp.x1);
    let hi = vp.x0.max(vp.x1);
    if !(hi > lo) {
        return out;
    }
    let dx = (hi - lo) / n as f32;
    let baseline = clamp_domain(0.0, vp.y0, vp.y1);
    for i in 1..=n {
        let xr = lo + dx * i as f32; // 右端点
        let xl = xr - dx;
        let yv = f(xr);
        if !xr.is_finite() || !yv.is_finite() {
            continue; // NaN / Inf 高度：跳过该矩形
        }
        let top = clamp_domain(yv, vp.y0, vp.y1);
        let a = vp.to_px(xl, baseline);
        let b = vp.to_px(xr, baseline);
        let c = vp.to_px(xr, top);
        let d = vp.to_px(xl, top);
        let quad = [a, b, c, d];
        if quad.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
            continue;
        }
        out.push(quad.to_vec());
    }
    out
}

// ---- 内部辅助 ----

/// 收集 `[lo, hi]` 内 `step` 的全部倍数（含端点；`lo > hi` / 非有限 /
/// 数量超 [`MAX_GRID_LINES`] 时不产出）。f32→i64 为饱和转换，无 panic。
fn multiples(step: f32, lo: f32, hi: f32, out: &mut Vec<f32>) {
    if !(step > 0.0) || !step.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
        return;
    }
    let first = (lo / step).ceil();
    let last = (hi / step).floor();
    if !first.is_finite() || !last.is_finite() || !(last >= first) {
        return;
    }
    let k_first = first as i64;
    let k_last = last as i64;
    if k_last < k_first {
        return;
    }
    let count = (k_last - k_first + 1) as u64;
    if count > MAX_GRID_LINES {
        return; // 病理视口：整族跳过
    }
    for k in k_first..=k_last {
        out.push(k as f32 * step);
    }
}

/// 压一段两点直线轨（非有限坐标丢弃）。
fn push_line(tracks: &mut Vec<Track>, a: [f32; 2], b: [f32; 2], color: [f32; 4], width: f32) {
    if !a[0].is_finite() || !a[1].is_finite() || !b[0].is_finite() || !b[1].is_finite() {
        return;
    }
    tracks.push(Track::Path(PathTrack {
        points: vec![a, b],
        color,
        width,
        fill: false,
        at: 0.0,
        in_sec: 0.0,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));
}

/// 数学域两点 → 设备像素直线轨（唯一映射经 `Viewport::to_px`）。
fn push_math_line(
    tracks: &mut Vec<Track>,
    vp: &Viewport,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
    width: f32,
) {
    push_line(tracks, vp.to_px(x0, y0), vp.to_px(x1, y1), color, width);
}

/// 压一枚三点 Chevron 箭头轨：`(bx, by)` 为单位「箭杆反方向」（设备
/// 像素），两翼沿其垂直方向展开 `ARROW_HALF_W`、回退 `ARROW_BACK`。
fn push_chevron(tracks: &mut Vec<Track>, tip: [f32; 2], bx: f32, by: f32, color: [f32; 4]) {
    let (px, py) = (-by, bx); // 垂直方向
    let w1 = [
        tip[0] + bx * ARROW_BACK + px * ARROW_HALF_W,
        tip[1] + by * ARROW_BACK + py * ARROW_HALF_W,
    ];
    let w2 = [
        tip[0] + bx * ARROW_BACK - px * ARROW_HALF_W,
        tip[1] + by * ARROW_BACK - py * ARROW_HALF_W,
    ];
    if tip[0].is_finite() && tip[1].is_finite() && w1[0].is_finite() && w1[1].is_finite() {
        tracks.push(Track::Path(PathTrack {
            points: vec![w1, tip, w2],
            color,
            width: AXIS_LINE_W,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
    }
}

/// 轴标签锚点 clamp 进画幅安全区（`[MARGIN, W-MARGIN] × [MARGIN+size, H-MARGIN]`）。
fn clamp_anchor(x: f32, y: f32) -> [f32; 2] {
    let lo_x = LABEL_MARGIN;
    let hi_x = (WIDTH as f32 - LABEL_MARGIN).max(lo_x);
    let lo_y = LABEL_MARGIN + AXIS_LABEL_SIZE;
    let hi_y = (HEIGHT as f32 - LABEL_MARGIN).max(lo_y);
    let x = if x.is_finite() { x.clamp(lo_x, hi_x) } else { WIDTH as f32 * 0.5 };
    let y = if y.is_finite() { y.clamp(lo_y, hi_y) } else { HEIGHT as f32 * 0.5 };
    [x, y]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 鞋带公式（多边形有向面积；设备系 y 向下，取绝对值）。
    fn shoelace(pts: &[[f32; 2]]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..pts.len() {
            let a = pts[i];
            let b = pts[(i + 1) % pts.len()];
            s += a[0] * b[1] - b[0] * a[1];
        }
        (s * 0.5).abs()
    }

    #[test]
    fn sample_curve_sin_within_viewport() {
        // 视口：x∈[0, 2π]、y∈[-1.5, 1.5]，绘图区中心 (480,270)、半区 400×200。
        let two_pi = 2.0 * std::f32::consts::PI;
        let vp = Viewport::fit_plot(0.0, two_pi, -1.5, 1.5, 480.0, 270.0, 400.0, 200.0);
        assert!(vp.is_valid());
        let pts = sample_curve(f32::sin, 0.0, two_pi, 64, &vp);
        assert_eq!(pts.len(), 64, "64 采样点全部落在视口 x 域内");
        // 逐点还原数学坐标：x∈[0, 2π]、|y| ≤ 1。
        let (mut max_y, mut min_y) = (f32::MIN, f32::MAX);
        for p in &pts {
            let m = vp.to_math(p[0], p[1]);
            assert!((-1e-3..=two_pi + 1e-3).contains(&m[0]), "x 越界 {}", m[0]);
            assert!(m[1] >= -1.0 - 1e-6 && m[1] <= 1.0 + 1e-6, "y 越界 {}", m[1]);
            assert!((0.0..=960.0).contains(&p[0]) && (0.0..=540.0).contains(&p[1]));
            max_y = max_y.max(m[1]);
            min_y = min_y.min(m[1]);
        }
        assert!(max_y > 0.999 && min_y < -0.999, "必须采到峰谷邻域（{max_y}, {min_y}）");
    }

    #[test]
    fn sample_curve_clips_outside_viewport_domain() {
        // 视口 x 域 [0, 2]；采样 [0, 4] 的 5 点 → 只留 x=0,1,2 三点。
        let vp = Viewport::fit_plot(0.0, 2.0, -1.0, 5.0, 480.0, 270.0, 300.0, 200.0);
        let pts = sample_curve(|x| x * x, 0.0, 4.0, 5, &vp);
        assert_eq!(pts.len(), 3, "界外采样被裁");
        let left = vp.to_px(0.0, 0.0);
        let right = vp.to_px(2.0, 4.0);
        assert!((pts[0][0] - left[0]).abs() < 1e-3 && (pts[0][1] - left[1]).abs() < 1e-3);
        assert!((pts[2][0] - right[0]).abs() < 1e-3 && (pts[2][1] - right[1]).abs() < 1e-3);
        // 非法输入：n<2 / 非有限区间 / 非有限函数值 / 非法视口 → 空，不 panic。
        assert!(sample_curve(|x| x, 0.0, 1.0, 1, &vp).is_empty());
        assert!(sample_curve(|x| x, f32::NAN, 1.0, 8, &vp).is_empty());
        assert!(sample_curve(|_| f32::INFINITY, 0.0, 1.0, 8, &vp).is_empty());
        let bad = Viewport::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        assert!(sample_curve(|x| x, 0.0, 1.0, 8, &bad).is_empty());
        // 裁剪后的点仍可含函数值冲出 y 域的样本（x 域内即保留）。
        let pts2 = sample_curve(|x| if x > 1.0 { 100.0 } else { x }, 0.0, 2.0, 3, &vp);
        assert_eq!(pts2.len(), 3, "y 向不裁剪");
    }

    #[test]
    fn riemann_rects_x_squared_matches_closed_form() {
        // 积分域 = 视口 x 域 [0, 2]（单测内自证域）；闭式 ∫x²dx = (x1³-x0³)/3。
        let vp = Viewport::fit_plot(0.0, 2.0, -0.5, 4.5, 480.0, 270.0, 300.0, 200.0);
        let n = 2000u32;
        let rects = riemann_rects(n, |x| x * x, &vp);
        assert_eq!(rects.len() as u32, n, "每分一块一矩形");
        // 像素面积（鞋带）累加 ÷ px_per_unit² = 数学面积。
        let px_area: f32 = rects.iter().map(|r| shoelace(r)).sum();
        let math_area = px_area / (vp.px_per_unit * vp.px_per_unit);
        let closed = (vp.x1 * vp.x1 * vp.x1 - vp.x0 * vp.x0 * vp.x0) / 3.0;
        let rel = (math_area - closed).abs() / closed;
        assert!(rel < 0.005, "右端和 vs 闭式 8/3 相对误差 {rel}（{math_area} vs {closed}）");
        // 全部矩形四角有限且在画幅内。
        for r in &rects {
            assert_eq!(r.len(), 4);
            for p in r {
                assert!(p[0].is_finite() && p[1].is_finite());
                assert!((0.0..=960.0).contains(&p[0]) && (0.0..=540.0).contains(&p[1]));
            }
        }
    }

    #[test]
    fn riemann_rects_wellformed_and_invalid_safe() {
        // 结构断言：四角必须构成轴对齐矩形（角点由同一 `to_px` 产生，共享
        // 基线行 / 顶行 / 左右列可逐位判定，不依赖浮点面积）；
        // 尺寸断言：宽 = dx·s、高 = (top−baseline)·s（容差吸收 `to_px` 舍入）。
        let vp = Viewport::fit_plot(0.0, 2.0, -0.5, 4.5, 480.0, 270.0, 300.0, 200.0);
        let s = vp.px_per_unit;
        let n = 200u32;
        let rects = riemann_rects(n, |x| x * x, &vp);
        assert_eq!(rects.len(), n as usize, "x² 在右端点处处有限：200 块全产出");
        let dx = 2.0f32 / n as f32;
        let baseline = 0.0f32; // clamp(0, −0.5, 4.5) = 0，且 0 为域内值
        for (i, r) in rects.iter().enumerate() {
            assert_eq!(r.len(), 4, "块 {i} 必须是四角");
            for p in r {
                assert!(p[0].is_finite() && p[1].is_finite(), "块 {i} 坐标必须有限");
                assert!(
                    (0.0..=WIDTH as f32).contains(&p[0]) && (0.0..=HEIGHT as f32).contains(&p[1]),
                    "块 {i} 坐标必须在画幅内：{p:?}"
                );
            }
            // 角序：a=左下 b=右下 c=右上 d=左上（顺时针）。同行 / 同列逐位相等。
            assert_eq!(r[0][1], r[1][1], "块 {i} 底边两点必须同基线行（逐位）");
            assert_eq!(r[2][1], r[3][1], "块 {i} 顶边两点必须同顶行（逐位）");
            assert_eq!(r[0][0], r[3][0], "块 {i} 左边两点必须同列（逐位）");
            assert_eq!(r[1][0], r[2][0], "块 {i} 右边两点必须同列（逐位）");
            assert!(r[0][0] < r[1][0], "块 {i} 左角必须在右角左侧（x 向右）");
            assert!(r[2][1] < r[1][1], "块 {i} 顶行必须在基线上方（y 轴翻转）");
            // 尺寸：宽 = dx·s；top = xr²（xr ≤ 2 ⇒ ≤ 4 < y1 = 4.5，未触钳位）。
            let xr = dx * (i as f32 + 1.0);
            let top = xr * xr;
            let w_expect = dx * s;
            let h_expect = (top - baseline) * s;
            let w_got = r[1][0] - r[0][0];
            let h_got = r[1][1] - r[2][1];
            assert!(
                (w_got - w_expect).abs() < 1e-3 * w_expect + 1e-3,
                "块 {i} 宽 {w_got} vs 期望 {w_expect}"
            );
            assert!(
                (h_got - h_expect).abs() < 1e-3 * h_expect + 1e-3,
                "块 {i} 高 {h_got} vs 期望 {h_expect}"
            );
        }
        // NaN 函数值 → 跳过该矩形（右端点 0.5 < 1 → NaN；其余 3 块保留）。
        let rects_nan = riemann_rects(4, |x| if x < 1.0 { f32::NAN } else { x * x }, &vp);
        assert_eq!(rects_nan.len(), 3, "NaN 矩形被跳过");
        // n=0 / 非法视口 → 空，不 panic。
        assert!(riemann_rects(0, |x| x, &vp).is_empty());
        let bad = Viewport::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0);
        assert!(riemann_rects(8, |x| x, &bad).is_empty());
    }

    #[test]
    fn axes_tracks_structure_and_bounds() {
        // 视口 x∈[-3,3]、y∈[-2,2]；半区 400×230 → 等比 min(400/3, 230/2) ≈ 115。
        let vp = Viewport::fit_plot(-3.0, 3.0, -2.0, 2.0, 480.0, 270.0, 400.0, 230.0);
        let tracks = axes_tracks(&vp);
        // 结构计数：网格 13+9、主轴 2、箭头 4、刻度 7+5、标签 2。
        let (mut grid, mut axis2, mut chev, mut tick, mut labels) = (0, 0, 0, 0, 0);
        for t in &tracks {
            match t {
                Track::Path(p) => {
                    for q in &p.points {
                        assert!(q[0].is_finite() && q[1].is_finite(), "坐标必须有限");
                        assert!(
                            (0.0..=WIDTH as f32).contains(&q[0]) && (0.0..=HEIGHT as f32).contains(&q[1]),
                            "坐标必须在画幅内：{q:?}"
                        );
                    }
                    if p.color == GRID_RGBA {
                        grid += 1;
                    } else if p.color == AXIS_RGBA && p.points.len() == 2 {
                        axis2 += 1;
                    } else if p.color == AXIS_RGBA && p.points.len() == 3 {
                        chev += 1;
                    } else if p.color == TICK_RGBA {
                        tick += 1;
                    } else {
                        panic!("未登记的颜色族 {:?}", p.color);
                    }
                }
                Track::Static(s) => {
                    labels += 1;
                    assert!(s.content == "x" || s.content == "y", "轴标签 {:?}", s.content);
                    assert_eq!(s.size, AXIS_LABEL_SIZE);
                    assert_eq!(s.color, TEXT_RGBA);
                    let cx = s.center_x.expect("轴标签必须带 E3 center_x 锚点");
                    assert!(cx.is_finite() && (0.0..=WIDTH as f32).contains(&cx));
                    assert!(s.y.is_finite() && (0.0..=HEIGHT as f32).contains(&s.y));
                }
                _ => panic!("axes_tracks 只应产出 Path/Static 轨"),
            }
        }
        assert_eq!(grid, 22, "网格 13 竖 + 9 横");
        assert_eq!(axis2, 2, "x / y 主轴");
        assert_eq!(chev, 4, "四枚箭头");
        assert_eq!(tick, 12, "刻度 7 + 5");
        assert_eq!(labels, 2, "x / y 标签");
        assert_eq!(tracks.len(), 42);
    }

    #[test]
    fn axes_tracks_clamps_axis_to_domain_edge() {
        // y 域 [0.5, 2.5]：原点不在域内 → x 轴贴下边 y0=0.5。
        let vp = Viewport::fit_plot(-1.0, 3.0, 0.5, 2.5, 480.0, 270.0, 300.0, 180.0);
        let tracks = axes_tracks(&vp);
        let mut found = false;
        for t in &tracks {
            if let Track::Path(p) = t {
                if p.color == AXIS_RGBA && p.points.len() == 2 && p.width == AXIS_LINE_W {
                    // 水平主轴：两点同 py。
                    if (p.points[0][1] - p.points[1][1]).abs() < 1e-6 {
                        let my = vp.to_math(480.0, p.points[0][1])[1];
                        assert!((my - vp.y0).abs() < 1e-3, "x 轴应贴 y0，实际数学 y={my}");
                        found = true;
                    }
                }
            }
        }
        assert!(found, "必须存在贴边的 x 主轴");
        // x 域 [0.5, 3]：原点不在域内 → y 轴贴左边 x0=0.5。
        let vp2 = Viewport::fit_plot(0.5, 3.0, -2.0, 2.0, 480.0, 270.0, 300.0, 180.0);
        let tracks2 = axes_tracks(&vp2);
        let mut found2 = false;
        for t in &tracks2 {
            if let Track::Path(p) = t {
                if p.color == AXIS_RGBA && p.points.len() == 2 && p.width == AXIS_LINE_W {
                    if (p.points[0][0] - p.points[1][0]).abs() < 1e-6 {
                        let mx = vp2.to_math(p.points[0][0], 270.0)[0];
                        assert!((mx - vp2.x0).abs() < 1e-3, "y 轴应贴 x0，实际数学 x={mx}");
                        found2 = true;
                    }
                }
            }
        }
        assert!(found2, "必须存在贴边的 y 主轴");
        // 非法视口 → 空（不 panic）。
        let bad = Viewport::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        assert!(axes_tracks(&bad).is_empty());
    }
}
