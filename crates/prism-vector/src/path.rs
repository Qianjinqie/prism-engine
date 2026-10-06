//! 矢量路径表达（B9 公共数据面）。
//!
//! [`Path`] 是多重子路径的矢量形状：每个子路径由 TTF 惯例的
//! points/tags 表达——`tag=true` 的点是二次贝塞尔控制点、`false` 是锚点
//! （on-curve）。这与 `prism-text`/`prism-raster` 的字形轮廓同构，故
//! 直接复用 `prism-raster` 的扫描线光栅对接像素覆盖（白皮书 §16）。
//!
//! 平铺（flatten）委托 `prism-raster::path::PathBuilder`——不重复造曲线
//! 细分轮子；本文件只做"矢量化数据 → PathBuilder → 覆盖率"的适配。

use prism_raster::geom::{Point, Transform};
use prism_raster::path::PathBuilder;
use prism_raster::{rasterize, Coverage, Contour, FillRule};

/// 一个路径点：坐标 + 是否控制点（on-curve）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    /// 坐标（设备像素或在调用方坐标系）。
    pub p: Point,
    /// `true` = 二次贝塞尔控制点，`false` = 锚点（on-curve）。
    pub control: bool,
}

impl Vertex {
    /// 构造锚点（on-curve）。
    pub fn anchor(x: f32, y: f32) -> Self {
        Self { p: Point::new(x, y), control: false }
    }

    /// 构造控制点（off-curve）。
    pub fn control(x: f32, y: f32) -> Self {
        Self { p: Point::new(x, y), control: true }
    }
}

/// 一条子路径：闭合与否 + 顶点序列（TTF 惯例）。
#[derive(Debug, Clone, PartialEq)]
pub struct SubPath {
    /// 顶点序列（首点必须为锚点；控制点后必跟一个锚点）。
    pub vertices: Vec<Vertex>,
    /// 是否闭合。
    pub closed: bool,
}

/// 矢量化路径：一组子路径。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    /// 子路径列表。
    pub subpaths: Vec<SubPath>,
}

impl Path {
    /// 构造空路径。
    pub fn new() -> Self {
        Self::default()
    }

    /// 从单条折线（锚点序列）构造闭合路径。
    pub fn from_polygon(points: &[Point]) -> Self {
        Self {
            subpaths: vec![SubPath {
                vertices: points.iter().map(|p| Vertex { p: *p, control: false }).collect(),
                closed: true,
            }],
        }
    }

    /// 追加一条子路径。
    pub fn push(&mut self, sub: SubPath) {
        self.subpaths.push(sub);
    }

    /// 所有子路径的总顶点数。
    pub fn vertex_count(&self) -> usize {
        self.subpaths.iter().map(|s| s.vertices.len()).sum()
    }

    /// 适配为 `prism-raster` 的 [`PathBuilder`]（坐标以锚点/控制点展开）。
    /// 首点必须为锚点：控制点起头的子路径由其首个锚点做 move_to 之前忽略
    /// 起手控制点（TTF 语义里起手控制点无前锚，退化处理）。
    pub fn to_builder(&self) -> PathBuilder {
        let mut pb = PathBuilder::new();
        for sub in &self.subpaths {
            // 找到第一个锚点做起手（控制点起手无前锚，数学上退化忽略）。
            let mut i = 0usize;
            while i < sub.vertices.len() && sub.vertices[i].control {
                i += 1;
            }
            if i >= sub.vertices.len() {
                continue; // 全控制点：无锚点，空子路径
            }
            pb.move_to(sub.vertices[i].p);
            i += 1;
            while i < sub.vertices.len() {
                let v = sub.vertices[i];
                if v.control {
                    // 控制点后的下一个锚点构成二次贝塞尔；缺失则用控制点自身退化为直线锚点。
                    let to = if i + 1 < sub.vertices.len() {
                        let a = sub.vertices[i + 1].p;
                        i += 1;
                        a
                    } else {
                        v.p
                    };
                    pb.quad_to(v.p, to);
                } else {
                    pb.line_to(v.p);
                }
                i += 1;
            }
            if sub.closed {
                pb.close();
            }
        }
        pb
    }
}

/// 自适应平铺：路径 → 设备空间闭合折线轮廓（委托 `prism-raster`）。
pub fn flatten(path: &Path, transform: &Transform) -> Vec<Contour> {
    path.to_builder().flatten(transform)
}

/// 路径 → 覆盖率缓冲（对接 `prism-raster` 扫描线光栅；白皮书 §16）。
pub fn rasterize_path(
    path: &Path,
    transform: &Transform,
    rule: FillRule,
    width: u32,
    height: u32,
) -> Coverage {
    let contours = flatten(path, transform);
    rasterize(&contours, rule, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_builder_roundtrips_through_raster() {
        let sq = Path::from_polygon(&[
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ]);
        let cov = rasterize_path(&sq, &Transform::IDENTITY, FillRule::Winding, 4, 4);
        // 全覆盖单位方：每像素覆盖率 1。
        for v in &cov.data {
            let d = (v - 1.0f32).abs();
            assert!(d < 1e-4, "像素覆盖率应近 1，实际 {v}");
        }
    }

    #[test]
    fn quad_curve_flattens_with_control_point() {
        let mut p = Path::new();
        p.push(SubPath {
            vertices: vec![
                Vertex::anchor(0.0, 0.0),
                Vertex::control(50.0, 100.0),
                Vertex::anchor(100.0, 0.0),
            ],
            closed: true,
        });
        let contours = flatten(&p, &Transform::IDENTITY);
        assert_eq!(contours.len(), 1);
        assert!(contours[0].len() > 8, "弓形应细分，实际 {} 点", contours[0].len());
    }

    #[test]
    fn vertex_count_and_leading_control_is_skipped() {
        let mut p = Path::new();
        p.push(SubPath {
            vertices: vec![Vertex::control(1.0, 1.0), Vertex::anchor(0.0, 0.0), Vertex::anchor(2.0, 2.0)],
            closed: false,
        });
        assert_eq!(p.vertex_count(), 3);
        // 起手控制点被忽略，首锚点做起手。
        let b = p.to_builder();
        assert_eq!(b.subpaths[0].start, Point::new(0.0, 0.0));
    }
}
