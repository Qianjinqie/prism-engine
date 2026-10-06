//! 扫描线光栅核心：行带解析覆盖率（白皮书 §16 选定算法）。
//!
//! # 算法（每行带 [y, y+1) 一次）
//!
//! 1. 轮廓边裁剪到行带（半开区间，共享顶点不重不计）→ 行带段
//!    （段是行带平面内的一条斜线 `x(y)`，附 winding 符号）；
//! 2. 收集 y 事件：段端点、段间交点、段穿整像素边界（x(y) = 整数）；
//! 3. 相邻事件切出子带：子带内每段完整覆盖或不覆盖，且跨越序固定；
//! 4. 子带内按 x 排序游走：相邻跨越之间的区域 winding 恒定（后缀和），
//!    区域端点是线性函数 → 与像素列 [X, X+1) 求交后端点仍线性 →
//!    梯形公式精确积分。
//!
//! 覆盖率是区域 {winding ≠ 0} 的**集合测度**：区域在任意 y 处 x 区间
//! 互不相交，重叠轮廓（winding=2）不重复计数——解析抗锯齿对重叠天然
//! 正确。
//!
//! # 确定性
//! 只用 ±×÷ 与比较；浮点排序 `total_cmp`；无状态。同输入同字节。

use crate::geom::Point;
use crate::Coverage;

/// 填充规则。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    /// 非零 winding（TrueType 字形标准规则）。
    Winding,
    /// 奇偶（交替）。
    EvenOdd,
}

/// 行带内的一段斜线：`x(y) = xa + (y − lo)·k`，`lo < hi`，附 winding 符号。
struct BandSeg {
    /// 段顶（lo）处 x。
    xa: f32,
    /// 段底（hi）处 x。
    xb: f32,
    /// 行带内 y 范围。
    lo: f32,
    hi: f32,
    /// 斜率 dx/dy。
    k: f32,
    /// winding 符号（+1 向下边，−1 向上边）。
    sign: f32,
}

impl BandSeg {
    #[inline]
    fn x_at(&self, y: f32) -> f32 {
        self.xa + (y - self.lo) * self.k
    }
}

/// 光栅化闭合轮廓集合 → 每像素覆盖率。
///
/// 轮廓视为闭合环（首尾自动相连，无需重复末点）；坐标为设备像素，
/// 输出按 `width × height` 裁剪。
pub fn rasterize(contours: &[Vec<Point>], rule: FillRule, width: u32, height: u32) -> Coverage {
    let mut cov = Coverage::new(width, height);
    if width == 0 || height == 0 || contours.is_empty() {
        return cov;
    }

    // 全部边 → (p0, p1)，跳过水平边（横射线模型中不改变 winding）。
    let mut edges: Vec<(Point, Point)> = Vec::new();
    for c in contours {
        let n = c.len();
        if n < 3 {
            continue; // 少于 3 点不构成有面积轮廓
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
    // 按 min_y 排序做行扫描的进入队列（total_cmp 全序）。
    edges.sort_by(|x, y| {
        let kx = x.0.y.min(x.1.y);
        let ky = y.0.y.min(y.1.y);
        kx.total_cmp(&ky)
    });
    let mut next_edge = 0usize;
    let mut active: Vec<(Point, Point)> = Vec::new(); // 当前行带仍跨越的边
    let mut segs: Vec<BandSeg> = Vec::new();
    let mut events: Vec<f32> = Vec::new();
    let mut row_acc: Vec<f32> = vec![0.0; width as usize];

    for row in 0..height as i64 {
        let y0 = row as f32;
        let y1 = y0 + 1.0;

        // 进入新跨越的边（行带是 [y0,y1) 区间积分：min_y < y1 即可能
        // 跨越本行带，顶点落在带内的边必须入队，否则同侧链断配对）；
        // 淘汰已结束的边。
        while next_edge < edges.len() && edges[next_edge].0.y.min(edges[next_edge].1.y) < y1 {
            active.push(edges[next_edge]);
            next_edge += 1;
        }
        active.retain(|(a, b)| a.y.max(b.y) > y0);
        if active.is_empty() && next_edge >= edges.len() {
            // 后续不会再有边进入（按 min_y 排序）——可提前收工。
            if next_edge >= edges.len() {
                break;
            }
        }

        // 裁剪到行带。
        segs.clear();
        for &(a, b) in &active {
            let lo = a.y.min(b.y).max(y0);
            let hi = a.y.max(b.y).min(y1);
            if lo >= hi {
                continue;
            }
            // 统一为"顶 → 底"几何段；winding 符号保留原方向。
            // 参数 t 已含方向信息：x(y=lo) 与 x(y=hi) 对上下坡一致。
            let dy = b.y - a.y; // ≠ 0（水平边已剔除）
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

        // y 事件：段端点。
        events.clear();
        for s in &segs {
            events.push(s.lo);
            events.push(s.hi);
        }
        // 段间交点（行带平面内两斜线相交）。
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                let (si, sj) = (&segs[i], &segs[j]);
                let den = si.k - sj.k;
                if den == 0.0 {
                    continue; // 平行
                }
                let ystar = (sj.xa - sj.lo * sj.k - (si.xa - si.lo * si.k)) / den;
                let lo = si.lo.max(sj.lo);
                let hi = si.hi.min(sj.hi);
                if ystar > lo && ystar < hi {
                    events.push(ystar);
                }
            }
        }
        // 段穿整像素边界（保证子带内区域端点不跨列）。
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

        // 逐子带精确积分。
        for w in events.windows(2) {
            let p = w[0];
            let q = w[1];
            if q <= p {
                continue;
            }
            // 活跃段 = 完整覆盖子带的段。
            let mut act: Vec<&BandSeg> =
                segs.iter().filter(|s| s.lo <= p && s.hi >= q).collect();
            if act.len() < 2 {
                continue;
            }
            // 按子带中点的 x 排序：交点事件保证开子带内序恒定；恰在子带
            // 边界的交点不切分本子带，其序翻转只发生在边界之外——中点处
            // 严格有序（完全重合的退化段并列，区域零宽，无害）。
            let pm = (p + q) * 0.5;
            act.sort_by(|a, b| a.x_at(pm).total_cmp(&b.x_at(pm)));

            // 左→右游走：跨越 j 与 j+1 之间的区域 winding = −前缀和
            // （闭合轮廓使总和为 0）；odd = 右侧跨越数（奇偶规则）。
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
                // 区域 [xl(y), xr(y))（端点线性、子带内不跨整列）。
                let xlp = s.x_at(p);
                let xlq = s.x_at(q);
                let xrp = t.x_at(p);
                let xrq = t.x_at(q);
                // 区域触及的列范围（两端各自整块落在同一列内）。
                let c0 = xlp.min(xlq).floor() as i64;
                let c1 = xrp.max(xrq).floor() as i64;
                for col in c0..=c1 {
                    if col < 0 || col >= width as i64 {
                        continue;
                    }
                    let xf = col as f32;
                    // 梯形积分：子带两端各算一次夹在列内的宽度。
                    let wp = (xrp.min(xf + 1.0) - xlp.max(xf)).max(0.0);
                    let wq = (xrq.min(xf + 1.0) - xlq.max(xf)).max(0.0);
                    row_acc[col as usize] += (wp + wq) * 0.5 * (q - p);
                }
            }
        }

        // 行带积分完成 → 写覆盖率缓冲。
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

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
        vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ]
    }

    #[test]
    fn unit_square_exact_coverage() {
        // 方块 [1.5,3)×[1.5,3)：内部 1.0，边界列/行 0.5（半开语义）。
        let cov = rasterize(&[square(1.5, 1.5, 3.0, 3.0)], FillRule::Winding, 5, 5);
        assert_eq!(cov.get(2, 2), 1.0, "内部像素全覆盖");
        assert_eq!(cov.get(1, 2), 0.5, "左边界列恰半覆盖");
        assert_eq!(cov.get(2, 1), 0.5, "上边界行恰半覆盖");
        assert_eq!(cov.get(1, 1), 0.25, "角落 1/4 覆盖");
        assert_eq!(cov.get(0, 0), 0.0);
        assert_eq!(cov.get(3, 2), 0.0, "x∈[3,4) 属半开区间外");
    }

    #[test]
    fn diagonal_triangle_half_plane() {
        // 直角三角形 (0,0)(2,0)(0,2)：斜边 x+y=2。
        // 像素 (0,0)：{x+y<2} ⊇ [0,1)²（仅触角点）→ 全覆盖 1。
        // 像素 (1,0)/(0,1)：对角线切半 → 0.5；像素 (1,1) 在斜边外侧 → 0。
        let tri = vec![Point::new(0.0, 0.0), Point::new(2.0, 0.0), Point::new(0.0, 2.0)];
        let cov = rasterize(&[tri], FillRule::Winding, 4, 4);
        assert!((cov.get(0, 0) - 1.0).abs() < 1e-6, "实际 {}", cov.get(0, 0));
        assert!((cov.get(1, 0) - 0.5).abs() < 1e-6, "实际 {}", cov.get(1, 0));
        assert!((cov.get(0, 1) - 0.5).abs() < 1e-6, "实际 {}", cov.get(0, 1));
        assert_eq!(cov.get(1, 1), 0.0, "对角线上方为空");
        // 总面积 = 2（三角形面积），覆盖率总和守恒。
        let sum: f32 = cov.data.iter().sum();
        assert!((sum - 2.0).abs() < 1e-5, "面积守恒：{sum}");
    }

    #[test]
    fn winding_vs_evenodd_on_overlap() {
        // 两个同向重叠方块 [0,2)² 与 [1,3)²：
        // winding 规则 = 并集（7）；even-odd 规则重叠区为偶数 = 洞（6）。
        let contours = [square(0.0, 0.0, 2.0, 2.0), square(1.0, 1.0, 3.0, 3.0)];
        let w = rasterize(&contours, FillRule::Winding, 5, 5);
        let e = rasterize(&contours, FillRule::EvenOdd, 5, 5);
        let sum_w: f32 = w.data.iter().sum();
        let sum_e: f32 = e.data.iter().sum();
        assert!((sum_w - 7.0).abs() < 1e-4, "winding 并集面积 7，实际 {sum_w}");
        assert!((sum_e - 6.0).abs() < 1e-4, "even-odd 重叠区成洞：6，实际 {sum_e}");
        // 重叠像素：winding=2 → 覆盖 1（不重复计数）；even-odd → 洞。
        assert!((w.get(1, 1) - 1.0).abs() < 1e-6, "重叠区 = 1.0，实际 {}", w.get(1, 1));
        assert_eq!(e.get(1, 1), 0.0, "even-odd 重叠区为洞");
    }

    #[test]
    fn hole_ring_winding_excludes_counter() {
        // 外方 [0,4)² + 反向内方 [1,3)²：环形面积 16−4=12。
        let mut outer = square(0.0, 0.0, 4.0, 4.0);
        outer.reverse();
        let inner = square(1.0, 1.0, 3.0, 3.0);
        let w = rasterize(&[outer, inner], FillRule::Winding, 6, 6);
        let sum: f32 = w.data.iter().sum();
        assert!((sum - 12.0).abs() < 1e-4, "环形面积 12，实际 {sum}");
        assert_eq!(w.get(2, 2), 0.0, "孔内为空");
        assert!((w.get(0, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_bitwise_across_runs() {
        let tri = vec![
            Point::new(0.3, 0.7),
            Point::new(4.9, 1.3),
            Point::new(2.2, 5.8),
        ];
        let a = rasterize(&[tri.clone()], FillRule::Winding, 8, 8);
        let b = rasterize(&[tri], FillRule::Winding, 8, 8);
        assert_eq!(a.data.len(), b.data.len());
        for (x, y) in a.data.iter().zip(b.data.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "两次光栅化必须逐位一致");
        }
    }

    #[test]
    fn out_of_bounds_geometry_clips_safely() {
        // 超出缓冲的大方块：覆盖率不越界、被视口裁剪。
        let cov = rasterize(&[square(-5.0, -5.0, 10.0, 10.0)], FillRule::Winding, 3, 3);
        for v in &cov.data {
            assert!((v - 1.0).abs() < 1e-6, "视口内应全覆盖，实际 {v}");
        }
    }
}
