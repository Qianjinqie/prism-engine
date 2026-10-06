//! 形状插值（morph，B9）。
//!
//! 两矢量形状间的渐变需要**公共参数域**：把每条路径按固定弧长参数
//! 重采样为 N 段折线（[`resample`]），再逐顶点点对点线性插值
//! （[`morph`]）。端点吻合由构造保证：u=0 得源形状、u=1 得目标形状
//! （见测试）。
//!
//! 只做顶点匹配的线性 morph（D-16 mini：够用即止）；更复杂的对应关系
//! （旋转对齐/最优传输）不在本波范围，属 YAGNI。

use crate::path::{Path, SubPath, Vertex};
use prism_raster::geom::Point;

/// 把一条折线按弧长重采样为 `segments` 段（返回 `segments+1` 个锚点）。
///
/// 弧长参数是几何纯函数：同输入必得同输出，跨机逐位一致（只用 ±×÷ 与
/// 比较）。闭合路径把首点再连回，弧长含闭合边。退化为零长度时全部返回
/// 首点（确定性）。
pub fn resample(points: &[Point], closed: bool, segments: usize) -> Vec<Point> {
    if segments == 0 {
        return if points.is_empty() { Vec::new() } else { vec![points[0]] };
    }
    // 顶点环（闭合时把首点追加到尾部形成整体弧长为周长）。
    let mut ring: Vec<Point> = points.to_vec();
    if closed {
        if let Some(first) = points.first() {
            ring.push(*first);
        }
    }
    if ring.len() < 2 {
        let p = ring.first().copied().unwrap_or(Point::new(0.0, 0.0));
        return vec![p; segments + 1];
    }

    // 累计弧长表。
    let mut cum = Vec::with_capacity(ring.len());
    cum.push(0.0f32);
    let mut total = 0.0f32;
    for i in 1..ring.len() {
        let dx = ring[i].x - ring[i - 1].x;
        let dy = ring[i].y - ring[i - 1].y;
        total += (dx * dx + dy * dy).sqrt();
        cum.push(total);
    }

    let mut out = Vec::with_capacity(segments + 1);
    // 均匀弧长目标：target_j = total · j / segments。
    for j in 0..=segments {
        let target = total * (j as f32) / (segments as f32);
        out.push(point_at_arc(&ring, &cum, target));
    }
    out
}

/// 在累计弧长表上取弧长 `s` 处的点（线性插值段内）。
fn point_at_arc(ring: &[Point], cum: &[f32], s: f32) -> Point {
    // 二分：找到 cum[i] <= s <= cum[i+1]。
    let n = cum.len();
    if n < 2 {
        return ring[0];
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    // NaN 或越界：钳制到端点（确定性）。
    if !(s >= cum[0]) {
        return ring[0];
    }
    if s >= cum[n - 1] {
        return ring[n - 1];
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cum[mid] <= s {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let seg_len = cum[lo + 1] - cum[lo];
    let t = if seg_len > 0.0 { (s - cum[lo]) / seg_len } else { 0.0 };
    let a = ring[lo];
    let b = ring[lo + 1];
    Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

/// 两形状间渐变插值：`u∈[0,1]`，`u=0` 得 `a`、`u=1` 得 `b`。
///
/// `a` 与 `b` 的子路径数须一致（不一致时取二者较小者，多余子路径丢弃，
/// 确定性）。每条子路径重采样到 `segments` 段后逐点线性插值。
pub fn morph(a: &Path, b: &Path, u: f32, segments: usize) -> Path {
    let sub_count = a.subpaths.len().min(b.subpaths.len());
    let mut out = Path::new();
    for i in 0..sub_count {
        let sa = &a.subpaths[i];
        let sb = &b.subpaths[i];
        let pa = anchors_of(sa);
        let pb = anchors_of(sb);
        let ra = resample(&pa, sa.closed, segments);
        let rb = resample(&pb, sb.closed, segments);
        let n = ra.len().min(rb.len());
        let mut verts = Vec::with_capacity(n);
        for k in 0..n {
            let x = ra[k].x + (rb[k].x - ra[k].x) * u;
            let y = ra[k].y + (rb[k].y - ra[k].y) * u;
            verts.push(Vertex::anchor(x, y));
        }
        out.push(SubPath { vertices: verts, closed: sa.closed || sb.closed });
    }
    out
}

/// 取子路径的锚点（含控制点前的锚点语义：控制点前的锚点已序列化；
/// morph 只用锚点做重采样，控制点被其后的锚点吸收）。
fn anchors_of(sub: &SubPath) -> Vec<Point> {
    sub.vertices.iter().filter(|v| !v.control).map(|v| v.p).collect()
}

/// [`morph`] 的别名，语义更贴近"两形状间渐变"（公共 API 两种叫法一致）。
pub fn morph_paths(a: &Path, b: &Path, u: f32, segments: usize) -> Path {
    morph(a, b, u, segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::rasterize_path;
    use prism_raster::{FillRule, Transform};

    fn square(s: f32) -> Path {
        Path::from_polygon(&[
            Point::new(0.0, 0.0),
            Point::new(s, 0.0),
            Point::new(s, s),
            Point::new(0.0, s),
        ])
    }

    #[test]
    fn resample_uniform_arc_length_on_segment() {
        // 长 10 的水平线段重采样 5 段：等距顶点。
        let pts = [Point::new(0.0, 0.0), Point::new(10.0, 0.0)];
        let r = resample(&pts, false, 5);
        assert_eq!(r.len(), 6);
        for (i, p) in r.iter().enumerate() {
            assert!((p.x - (i as f32) * 2.0).abs() < 1e-5, "点 {i} x={}", p.x);
            assert!(p.y.abs() < 1e-6);
        }
    }

    #[test]
    fn morph_endpoints_match_source_and_target() {
        let a = square(10.0);
        let b = square(20.0);
        let m0 = morph(&a, &b, 0.0, 16);
        let m1 = morph(&a, &b, 1.0, 16);
        // 端点吻合：u=0 覆盖面积 ≈ 10²、u=1 ≈ 20²。
        let cov0 = rasterize_path(&m0, &Transform::IDENTITY, FillRule::Winding, 32, 32);
        let cov1 = rasterize_path(&m1, &Transform::IDENTITY, FillRule::Winding, 32, 32);
        let s0: f32 = cov0.data.iter().sum();
        let s1: f32 = cov1.data.iter().sum();
        assert!((s0 - 100.0).abs() < 1.0, "u=0 面积应≈100，实际 {s0}");
        assert!((s1 - 400.0).abs() < 1.0, "u=1 面积应≈400，实际 {s1}");
    }

    #[test]
    fn morph_is_deterministic_bitwise() {
        let a = square(8.0);
        let b = square(24.0);
        let m1 = morph(&a, &b, 0.37, 20);
        let m2 = morph(&a, &b, 0.37, 20);
        for (s1, s2) in m1.subpaths.iter().zip(&m2.subpaths) {
            for (v1, v2) in s1.vertices.iter().zip(&s2.vertices) {
                assert_eq!(v1.p.x.to_bits(), v2.p.x.to_bits());
                assert_eq!(v1.p.y.to_bits(), v2.p.y.to_bits());
            }
        }
    }
}
