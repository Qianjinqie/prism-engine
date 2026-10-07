//! 路径构建与曲线平铺（flatten）。
//!
//! 二次贝塞尔（字体轮廓的原生曲线）按风险项要求做**自适应
//! 平铺**：控制点对弦的偏差 ≤ 0.5px（设备空间）即落直线，否则中点
//! de Casteljau 细分。平铺是几何的纯函数——同路径必得同折线。

use crate::geom::{Point, Transform};

/// 一条闭合折线轮廓（平铺产物，设备像素坐标）。
pub type Contour = Vec<Point>;

/// 路径元素：直线或二次贝塞尔（自当前点出发）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathElement {
    /// 直线到目标点。
    Line(Point),
    /// 二次贝塞尔：控制点 + 终点。
    Quad { ctrl: Point, to: Point },
}

/// 单条子路径（move_to 起头，close 收尾）。
#[derive(Debug, Clone, PartialEq)]
pub struct SubPath {
    /// 起点。
    pub start: Point,
    /// 后续元素。
    pub elements: Vec<PathElement>,
    /// 是否闭合（close 调用置位；未闭合时平铺按起点直连闭合——
    /// 填充语义要求轮廓封闭）。
    pub closed: bool,
}

/// 路径构建器：move/line/quad/close 语义，输出子路径列表。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathBuilder {
    /// 子路径列表。
    pub subpaths: Vec<SubPath>,
    current: Option<Point>,
}

/// 平铺误差阈值（设备像素，：自适应平铺误差 0.5px）。
const FLATTEN_TOLERANCE: f32 = 0.5;

/// 细分深度上限（2^16 段足够任何实际曲线；同时防病态输入无限递归）。
const MAX_DEPTH: u32 = 16;

impl PathBuilder {
    /// 构造空路径。
    pub fn new() -> Self {
        Self::default()
    }

    /// 起一条新子路径。
    pub fn move_to(&mut self, p: Point) {
        self.subpaths.push(SubPath { start: p, elements: Vec::new(), closed: false });
        self.current = Some(p);
    }

    /// 直线段。
    pub fn line_to(&mut self, p: Point) {
        let from = self.current.unwrap_or(p);
        if self.subpaths.is_empty() {
            self.subpaths.push(SubPath { start: from, elements: Vec::new(), closed: false });
        }
        self.subpaths.last_mut().unwrap().elements.push(PathElement::Line(p));
        self.current = Some(p);
    }

    /// 二次贝塞尔段。
    pub fn quad_to(&mut self, ctrl: Point, to: Point) {
        let from = self.current.unwrap_or(to);
        if self.subpaths.is_empty() {
            self.subpaths.push(SubPath { start: from, elements: Vec::new(), closed: false });
        }
        self.subpaths
            .last_mut()
            .unwrap()
            .elements
            .push(PathElement::Quad { ctrl, to });
        self.current = Some(to);
    }

    /// 闭合当前子路径（终点直连回起点）。
    pub fn close(&mut self) {
        if let Some(last) = self.subpaths.last_mut() {
            last.closed = true;
        }
        self.current = self.subpaths.last().map(|s| s.start);
    }

    /// 平铺全部子路径：应用变换后输出闭合折线轮廓列表。
    ///
    /// 曲线自适应细分阈值以**变换后**的设备空间弦偏差度量。
    pub fn flatten(&self, transform: &Transform) -> Vec<Contour> {
        let mut out = Vec::with_capacity(self.subpaths.len());
        for sub in &self.subpaths {
            let mut contour: Contour = Vec::new();
            let start = transform.apply(sub.start);
            let mut prev = start;
            contour.push(start);
            for el in &sub.elements {
                match *el {
                    PathElement::Line(to) => {
                        let p = transform.apply(to);
                        contour.push(p);
                        prev = p;
                    }
                    PathElement::Quad { ctrl, to } => {
                        let c = transform.apply(ctrl);
                        let t = transform.apply(to);
                        flatten_quad(prev, c, t, FLATTEN_TOLERANCE, 0, &mut contour);
                        prev = t;
                    }
                }
            }
            // 填充要求封闭：终点未回到起点则补一条闭合边。
            if let Some(first) = contour.first().copied() {
                if let Some(last) = contour.last().copied() {
                    if first.x != last.x || first.y != last.y {
                        contour.push(first);
                    }
                }
            }
            out.push(contour);
        }
        out
    }
}

/// 二次贝塞尔自适应平铺（递归中点细分）。
///
/// 平坦判据：控制点 `c` 到弦 `p0–p2` 的垂直偏差 ≤ `tol`（设备空间）。
/// 偏差用点到直线距离公式（±×÷ 与乘法，无 libm）。
fn flatten_quad(p0: Point, c: Point, p2: Point, tol: f32, depth: u32, out: &mut Contour) {
    // 弦向量与法向距离：dist = |cross(chord, c−p0)| / |chord|。
    let dx = p2.x - p0.x;
    let dy = p2.y - p0.y;
    let wx = c.x - p0.x;
    let wy = c.y - p0.y;
    let cross = (dx * wy - dy * wx).abs();
    let chord_sq = dx * dx + dy * dy;
    // 偏差² ≤ tol² ⇔ cross² ≤ tol²·chord_sq（避免开方；chord_sq=0 时
    // 三点共点，落直线）。
    let flat = cross * cross <= tol * tol * chord_sq;
    if flat || depth >= MAX_DEPTH {
        out.push(p2);
        return;
    }
    // de Casteljau 中点细分。
    let ab = Point::new((p0.x + c.x) * 0.5, (p0.y + c.y) * 0.5);
    let bc = Point::new((c.x + p2.x) * 0.5, (c.y + p2.y) * 0.5);
    let mid = Point::new((ab.x + bc.x) * 0.5, (ab.y + bc.y) * 0.5);
    flatten_quad(p0, ab, mid, tol, depth + 1, out);
    flatten_quad(mid, bc, p2, tol, depth + 1, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_lines_flatten_verbatim() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0));
        pb.line_to(Point::new(10.0, 0.0));
        pb.line_to(Point::new(10.0, 10.0));
        pb.close();
        let contours = pb.flatten(&Transform::IDENTITY);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].len(), 4, "三顶点 + 闭合边");
        assert_eq!(contours[0][3], Point::new(0.0, 0.0), "闭合边回到起点");
    }

    #[test]
    fn quad_near_straight_reduces_to_single_segment() {
        // 控制点恰在弦上 → 一次平铺即直线（两顶点 + 闭合）。
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0));
        pb.quad_to(Point::new(5.0, 0.0), Point::new(10.0, 0.0));
        pb.close();
        let contours = pb.flatten(&Transform::IDENTITY);
        assert_eq!(contours[0].len(), 3, "起点 + 曲线终点 + 闭合点");
    }

    #[test]
    fn quad_curved_subdivides_to_tolerance() {
        // 大弓形：细分后段数充足，且平铺多边形面积收敛到精确值。
        // 二次贝塞尔与弦围成的面积 = 2/3 × 控制三角形面积（经典结果）：
        // 控制三角形 (0,0)(50,100)(100,0) 面积 5000 → 弓形 10000/3。
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(0.0, 0.0));
        pb.quad_to(Point::new(50.0, 100.0), Point::new(100.0, 0.0));
        pb.close();
        let contours = pb.flatten(&Transform::IDENTITY);
        let poly = &contours[0];
        assert!(poly.len() > 8, "弓形必须细分，实际段数 {}", poly.len() - 1);
        let cov = crate::rasterize(std::slice::from_ref(poly), crate::FillRule::Winding, 110, 110);
        let sum: f32 = cov.data.iter().sum();
        // 期望界 = 精确值 − 平铺固有误差（弦贴曲线内侧，0.5px 控制偏差阈值
        // 下实测 ~13/3333 ≈ 0.4%）；若光栅器本身丢面积（>20）即失败。
        assert!(
            (sum - 10000.0 / 3.0).abs() < 20.0,
            "平铺+光栅面积应收敛到 2/3 控制三角形，实际 {sum}"
        );
    }

    #[test]
    fn flatten_applies_transform_and_closes() {
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(1.0, 1.0));
        pb.line_to(Point::new(2.0, 1.0));
        pb.line_to(Point::new(2.0, 2.0));
        pb.close();
        let tf = Transform::scale(10.0).then(&Transform::translation(100.0, 0.0));
        let c = &pb.flatten(&tf)[0];
        assert_eq!(c[0], Point::new(110.0, 10.0));
        assert_eq!(c[2], Point::new(120.0, 20.0));
    }
}
