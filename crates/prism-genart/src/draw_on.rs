//! Draw-on 逐步绘制动画（B9）。
//!
//! "路径逐步绘制"= 给定进度 `s ∈ [0,1]`，只显示路径前 `s` 比例的部分
//! （按**弧长**切分）。进度来源接 `prism-anim` 的 `Track`（关键帧轨道采样），
//! 保证 seek 纯函数、逐帧可复现。
//!
//! 复现 W5-B5 光栅：截断后的折线经 `prism-vector` → `prism-raster` 扫描线
//! 光栅（**不重复造轮子**）。
//!
//! 确定性：截断是弧长纯函数；同 s 必得同轮廓，进度对 s 单调（见测试）。

use prism_anim::Track;
use prism_raster::{Contour, Transform};
use prism_vector::Path;

use crate::generators::rasterize_lines;

/// 按进度 `s∈[0,1]` 截断路径的**平铺轮廓**（弧长参数）：
///
/// - `s<=0` → 空；`s>=1` → 全路径轮廓；
/// - 否则每条折线保留累计弧长 ≤ `s·总弧长` 的顶点，末段插值截到精确点。
///
/// 返回设备空间（经 `transform`）的折线集合，供光栅与显示。
pub fn progress_contours(path: &Path, s: f32, transform: &Transform) -> Vec<Contour> {
    let all = path.to_builder().flatten(transform);
    if s <= 0.0 {
        return Vec::new();
    }
    if s >= 1.0 {
        return all;
    }
    // 全路径总弧长 → 目标弧长。
    let total: f32 = all.iter().map(|c| polyline_len(c)).sum();
    let target = total * s;
    let mut out: Vec<Contour> = Vec::new();
    let mut used = 0.0f32;
    for c in &all {
        if used >= target {
            break;
        }
        let seg = truncate_by_arc(c, target - used);
        if seg.len() >= 1 {
            used += polyline_len(&seg);
            out.push(seg);
        }
    }
    out
}

/// 折线弧长。
fn polyline_len(pts: &[prism_raster::geom::Point]) -> f32 {
    let mut l = 0.0f32;
    for w in pts.windows(2) {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        l += (dx * dx + dy * dy).sqrt();
    }
    l
}

/// 把一条折线截到总弧长 `budget`（末段线性插值），至少保留首点。
fn truncate_by_arc(pts: &[prism_raster::geom::Point], budget: f32) -> Contour {
    let mut out: Contour = Vec::new();
    if pts.is_empty() {
        return out;
    }
    out.push(pts[0]);
    let mut acc = 0.0f32;
    for w in pts.windows(2) {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        let seg = (dx * dx + dy * dy).sqrt();
        if acc + seg <= budget {
            out.push(w[1]);
            acc += seg;
        } else {
            let remain = (budget - acc).max(0.0);
            if seg > 1e-6 {
                let t = remain / seg;
                out.push(prism_raster::geom::Point::new(w[0].x + dx * t, w[0].y + dy * t));
            }
            break;
        }
    }
    out
}

/// 路径 → 进度 s 的覆盖率缓冲（draw-on 画面；复用 genart 线段光栅）。
pub fn draw_on_path(
    path: &Path,
    s: f32,
    transform: &Transform,
    width: u32,
    height: u32,
) -> prism_raster::Coverage {
    let contours = progress_contours(path, s, transform);
    rasterize_lines(&contours, width, height)
}

/// 绑定 `prism-anim` 轨道：按时间 `t` 采样进度并绘制（seek 纯函数）。
pub fn draw_on(
    path: &Path,
    track: &Track,
    t: f32,
    transform: &Transform,
    width: u32,
    height: u32,
) -> prism_raster::Coverage {
    let s = track.sample(t).clamp(0.0, 1.0);
    draw_on_path(path, s, transform, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_anim::Keyframe;
    use prism_raster::geom::Point;
    use prism_raster::FillRule;

    fn square() -> Path {
        Path::from_polygon(&[
            Point::new(0.0, 0.0),
            Point::new(40.0, 0.0),
            Point::new(40.0, 40.0),
            Point::new(0.0, 40.0),
        ])
    }

    #[test]
    fn progress_monotone_non_decreasing() {
        let p = square();
        let mut last = 0.0f32;
        let mut last_area = -1.0f32;
        for k in 0..=10 {
            let s = k as f32 / 10.0;
            let mut c = progress_contours(&p, s, &Transform::IDENTITY);
            // 每步覆盖率（面积）不应减少（单调推进）。
            let cov = rasterize_lines(&c, 64, 64);
            let area: f32 = cov.data.iter().sum();
            assert!(area >= last_area - 1e-3, "进度 {s} 面积回退：{area} < {last_area}");
            last_area = area;
            last = s;
            c.clear();
        }
        assert_eq!(last, 1.0);
    }

    #[test]
    fn endpoints_empty_and_full() {
        let p = square();
        assert!(progress_contours(&p, 0.0, &Transform::IDENTITY).is_empty());
        let full = progress_contours(&p, 1.0, &Transform::IDENTITY);
        let cov = prism_raster::rasterize(&full, FillRule::Winding, 64, 64);
        // s=1 应是完整方框轮廓（含光栅面积）。
        assert!(cov.data.iter().sum::<f32>() > 0.0);
    }

    #[test]
    fn track_driven_drawon_seek_is_pure() {
        // 进度轨道：0→1 线性。
        let track = Track::new(vec![
            Keyframe::new(0.0, 0.0, prism_anim::EasingKind::Linear),
            Keyframe::new(1.0, 1.0, prism_anim::EasingKind::Linear),
        ]);
        let p = square();
        let a = draw_on(&p, &track, 0.5, &Transform::IDENTITY, 64, 64);
        let b = draw_on(&p, &track, 0.5, &Transform::IDENTITY, 64, 64);
        assert_eq!(a.data, b.data, "seek 纯函数：同 t 同帧");
        // 进度越大覆盖越多。
        let lo = draw_on_path(&p, 0.25, &Transform::IDENTITY, 64, 64);
        let hi = draw_on_path(&p, 0.75, &Transform::IDENTITY, 64, 64);
        assert!(hi.data.iter().sum::<f32>() >= lo.data.iter().sum::<f32>());
    }
}
