//! **优化前实现的逐字副本**（仅 `#[cfg(test)]` 编译）：差分对拍的黄金参考。
//!
//! 语义与优化前的 `scanline::rasterize` 完全一致：逐行带 → 以「段端点 / 段间
//! 交点 / 段穿整像素列」为事件切子带 → 子带内全量 filter + sort → 逐列梯形
//! 积分。它慢得离谱（star2048@4K 约 23 s），但每一步都是可手工验算的解析式，
//! 因此作为对拍基准而不是「另一份优化代码」。
//!
//! 另附一份**同算法、f64 数值**的 `rasterize_reference_f64`：算法结构同上，只把
//! 数值精度提到 f64。它的结果与「精确有理数积分」一致到 1e-12 量级（见
//! `scanline_diff.rs` 的说明），因此充当**精度标尺**：
//! 旧 f32 实现在 1080p 坐标下的自身量化误差约 5e-5，任何独立舍入的实现都不可
//! 能与它逐位或 2e-6 内一致；能验证的是「新实现贴近真值」。
//!
//! 本模块只被 `src/scanline_diff.rs` 的测试引用，不进任何 release 产物。

use crate::geom::Point;
use crate::scanline::FillRule;
use crate::Coverage;

/// 行带内的一段斜线：`x(y) = xa + (y − lo)·k`，`lo < hi`，附 winding 符号。
struct BandSeg {
    xa: f32,
    xb: f32,
    lo: f32,
    hi: f32,
    k: f32,
    sign: f32,
}

impl BandSeg {
    #[inline]
    fn x_at(&self, y: f32) -> f32 {
        self.xa + (y - self.lo) * self.k
    }
}

/// 参考实现：与优化前 `scanline::rasterize` 逐字等价。
pub fn rasterize_reference(
    contours: &[Vec<Point>],
    rule: FillRule,
    width: u32,
    height: u32,
) -> Coverage {
    let mut cov = Coverage::new(width, height);
    if width == 0 || height == 0 || contours.is_empty() {
        return cov;
    }

    let mut edges: Vec<(Point, Point)> = Vec::new();
    for c in contours {
        let n = c.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = c[i];
            let b = c[(i + 1) % n];
            if a.y != b.y {
                edges.push((a, b));
            }
        }
    }
    if edges.is_empty() {
        return cov;
    }
    edges.sort_by(|x, y| {
        let kx = x.0.y.min(x.1.y);
        let ky = y.0.y.min(y.1.y);
        kx.total_cmp(&ky)
    });
    let mut next_edge = 0usize;
    let mut active: Vec<(Point, Point)> = Vec::new();
    let mut segs: Vec<BandSeg> = Vec::new();
    let mut events: Vec<f32> = Vec::new();
    let mut row_acc: Vec<f32> = vec![0.0; width as usize];

    for row in 0..height as i64 {
        let y0 = row as f32;
        let y1 = y0 + 1.0;

        while next_edge < edges.len() && edges[next_edge].0.y.min(edges[next_edge].1.y) < y1 {
            active.push(edges[next_edge]);
            next_edge += 1;
        }
        active.retain(|(a, b)| a.y.max(b.y) > y0);
        if active.is_empty() && next_edge >= edges.len() {
            break;
        }

        segs.clear();
        for &(a, b) in &active {
            let lo = a.y.min(b.y).max(y0);
            let hi = a.y.max(b.y).min(y1);
            if lo >= hi {
                continue;
            }
            let dy = b.y - a.y;
            let t_lo = (lo - a.y) / dy;
            let t_hi = (hi - a.y) / dy;
            let xa = a.x + t_lo * (b.x - a.x);
            let xb = a.x + t_hi * (b.x - a.x);
            let sign = if dy > 0.0 { 1.0f32 } else { -1.0f32 };
            segs.push(BandSeg { xa, xb, lo, hi, k: (xb - xa) / (hi - lo), sign });
        }
        if segs.is_empty() {
            continue;
        }

        events.clear();
        for s in &segs {
            events.push(s.lo);
            events.push(s.hi);
        }
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                let (si, sj) = (&segs[i], &segs[j]);
                let den = si.k - sj.k;
                if den == 0.0 {
                    continue;
                }
                let ystar = (sj.xa - sj.lo * sj.k - (si.xa - si.lo * si.k)) / den;
                let lo = si.lo.max(sj.lo);
                let hi = si.hi.min(sj.hi);
                if ystar > lo && ystar < hi {
                    events.push(ystar);
                }
            }
        }
        for s in &segs {
            let (mn, mx) = if s.xa < s.xb { (s.xa, s.xb) } else { (s.xb, s.xa) };
            let n0 = mn.floor() as i64;
            let n1 = mx.floor() as i64;
            for n in n0..=n1 {
                let nf = n as f32;
                if nf > mn && nf < mx {
                    events.push(s.lo + (nf - s.xa) / s.k);
                }
            }
        }
        events.sort_by(f32::total_cmp);

        for w in events.windows(2) {
            let p = w[0];
            let q = w[1];
            if q <= p {
                continue;
            }
            let mut act: Vec<&BandSeg> = segs.iter().filter(|s| s.lo <= p && s.hi >= q).collect();
            if act.len() < 2 {
                continue;
            }
            let pm = (p + q) * 0.5;
            act.sort_by(|a, b| a.x_at(pm).total_cmp(&b.x_at(pm)));

            let total = act.len();
            let mut prefix_w = 0.0f32;
            for j in 0..total - 1 {
                let s = act[j];
                let t = act[j + 1];
                prefix_w += s.sign;
                let winding = -prefix_w;
                let inside = match rule {
                    FillRule::Winding => winding != 0.0,
                    FillRule::EvenOdd => (total - 1 - j) % 2 == 1,
                };
                if !inside {
                    continue;
                }
                let xlp = s.x_at(p);
                let xlq = s.x_at(q);
                let xrp = t.x_at(p);
                let xrq = t.x_at(q);
                let c0 = xlp.min(xlq).floor() as i64;
                let c1 = xrp.max(xrq).floor() as i64;
                for col in c0..=c1 {
                    if col < 0 || col >= width as i64 {
                        continue;
                    }
                    let xf = col as f32;
                    let wp = (xrp.min(xf + 1.0) - xlp.max(xf)).max(0.0);
                    let wq = (xrq.min(xf + 1.0) - xlq.max(xf)).max(0.0);
                    row_acc[col as usize] += (wp + wq) * 0.5 * (q - p);
                }
            }
        }

        for (x, v) in row_acc.iter().enumerate() {
            if *v != 0.0 {
                cov.data[row as usize * width as usize + x] = *v;
            }
        }
        for v in row_acc.iter_mut() {
            *v = 0.0;
        }
    }
    cov
}

/// 行带内的一段斜线（f64 版）。
struct BandSegF64 {
    xa: f64,
    xb: f64,
    lo: f64,
    hi: f64,
    k: f64,
    sign: f64,
}

impl BandSegF64 {
    #[inline]
    fn x_at(&self, y: f64) -> f64 {
        self.xa + (y - self.lo) * self.k
    }
}

/// 与 [`rasterize_reference`] **同算法**、f64 数值的精度标尺。
///
/// 结构完全一致（行带 → 端点/交点/列折点切子带 → 子带内 filter+sort → 逐列梯形），
/// 只是全部中间量走 f64。子带内 w(y) 线性，梯形公式是精确积分，因此它给出的
/// 覆盖率是对「给定 f32 顶点」的精确面积，误差仅来自最后一次 f64 累加（~1e-13）。
pub fn rasterize_reference_f64(
    contours: &[Vec<Point>],
    rule: FillRule,
    width: u32,
    height: u32,
) -> Coverage {
    let mut cov = Coverage::new(width, height);
    if width == 0 || height == 0 || contours.is_empty() {
        return cov;
    }

    let mut edges: Vec<(Point, Point)> = Vec::new();
    for c in contours {
        let n = c.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = c[i];
            let b = c[(i + 1) % n];
            if a.y != b.y {
                edges.push((a, b));
            }
        }
    }
    if edges.is_empty() {
        return cov;
    }
    edges.sort_by(|x, y| x.0.y.min(x.1.y).total_cmp(&y.0.y.min(y.1.y)));
    let mut next_edge = 0usize;
    let mut active: Vec<(Point, Point)> = Vec::new();
    let mut segs: Vec<BandSegF64> = Vec::new();
    let mut events: Vec<f64> = Vec::new();
    let mut row_acc: Vec<f64> = vec![0.0; width as usize];

    for row in 0..height as i64 {
        let y0 = row as f64;
        let y1 = y0 + 1.0;
        while next_edge < edges.len() && (edges[next_edge].0.y.min(edges[next_edge].1.y) as f64) < y1 {
            active.push(edges[next_edge]);
            next_edge += 1;
        }
        active.retain(|(a, b)| (a.y.max(b.y) as f64) > y0);
        if active.is_empty() && next_edge >= edges.len() {
            break;
        }

        segs.clear();
        for &(a, b) in &active {
            let (ax, ay) = (a.x as f64, a.y as f64);
            let (bx, by) = (b.x as f64, b.y as f64);
            let lo = ay.min(by).max(y0);
            let hi = ay.max(by).min(y1);
            if lo >= hi {
                continue;
            }
            let dy = by - ay;
            let dx = bx - ax;
            let xa = ax + ((lo - ay) / dy) * dx;
            let xb = ax + ((hi - ay) / dy) * dx;
            let sign = if dy > 0.0 { 1.0f64 } else { -1.0f64 };
            segs.push(BandSegF64 { xa, xb, lo, hi, k: (xb - xa) / (hi - lo), sign });
        }
        if segs.is_empty() {
            continue;
        }

        events.clear();
        for s in &segs {
            events.push(s.lo);
            events.push(s.hi);
        }
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                let (si, sj) = (&segs[i], &segs[j]);
                let den = si.k - sj.k;
                if den == 0.0 {
                    continue;
                }
                let ystar = (sj.xa - sj.lo * sj.k - (si.xa - si.lo * si.k)) / den;
                let lo = si.lo.max(sj.lo);
                let hi = si.hi.min(sj.hi);
                if ystar > lo && ystar < hi {
                    events.push(ystar);
                }
            }
        }
        for s in &segs {
            let (mn, mx) = if s.xa < s.xb { (s.xa, s.xb) } else { (s.xb, s.xa) };
            let n0 = mn.floor() as i64;
            let n1 = mx.floor() as i64;
            for n in n0..=n1 {
                let nf = n as f64;
                if nf > mn && nf < mx {
                    events.push(s.lo + (nf - s.xa) / s.k);
                }
            }
        }
        events.sort_by(f64::total_cmp);

        for w in events.windows(2) {
            let p = w[0];
            let q = w[1];
            if q <= p {
                continue;
            }
            let mut act: Vec<&BandSegF64> = segs.iter().filter(|s| s.lo <= p && s.hi >= q).collect();
            if act.len() < 2 {
                continue;
            }
            let pm = (p + q) * 0.5;
            act.sort_by(|a, b| a.x_at(pm).total_cmp(&b.x_at(pm)));
            let total = act.len();
            let mut prefix_w = 0.0f64;
            for j in 0..total - 1 {
                let s = act[j];
                let t = act[j + 1];
                prefix_w += s.sign;
                let inside = match rule {
                    FillRule::Winding => (-prefix_w) != 0.0,
                    FillRule::EvenOdd => (total - 1 - j) % 2 == 1,
                };
                if !inside {
                    continue;
                }
                let xlp = s.x_at(p);
                let xlq = s.x_at(q);
                let xrp = t.x_at(p);
                let xrq = t.x_at(q);
                let c0 = xlp.min(xlq).floor() as i64;
                let c1 = xrp.max(xrq).floor() as i64;
                for col in c0..=c1 {
                    if col < 0 || col >= width as i64 {
                        continue;
                    }
                    let xf = col as f64;
                    let wp = (xrp.min(xf + 1.0) - xlp.max(xf)).max(0.0);
                    let wq = (xrq.min(xf + 1.0) - xlq.max(xf)).max(0.0);
                    row_acc[col as usize] += (wp + wq) * 0.5 * (q - p);
                }
            }
        }

        for (x, v) in row_acc.iter().enumerate() {
            if *v != 0.0 {
                cov.data[row as usize * width as usize + x] = *v as f32;
            }
        }
        for v in row_acc.iter_mut() {
            *v = 0.0;
        }
    }
    cov
}
