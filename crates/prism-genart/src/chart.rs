//! A6 图表能力（并入 B9）：简单图表节点。
//!
//! 折线图 / 柱状图的最小实现——复用生成艺术的路径面与线段光栅，不做
//! 坐标轴文字排版（那属 B5 文本），只出几何形状（D-16 mini：够用即止）。
//! 数据坐标 → 像素坐标按图表矩形的线性映射（确定性纯函数）。

use prism_raster::{Coverage, Point};

use crate::generators::rasterize_lines;

/// 图表类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartKind {
    /// 折线图。
    Line,
    /// 柱状图。
    Bar,
}

/// 图表规格。
#[derive(Debug, Clone, PartialEq)]
pub struct ChartSpec {
    /// 类型。
    pub kind: ChartKind,
    /// 数据点（y 值按序）。
    pub data: Vec<f32>,
    /// 基线 y（数据最小值映射到的像素 y；柱状从基线上长）。
    pub baseline: f32,
    /// 上边距（像素）。
    pub margin: u32,
}

impl Default for ChartSpec {
    fn default() -> Self {
        Self { kind: ChartKind::Line, data: Vec::new(), baseline: 0.0, margin: 8 }
    }
}

impl ChartSpec {
    /// 生成图表几何为折线集合（像素坐标）。
    ///
    /// y 映射：`py = margin + (1 − norm(d)) · (h − 2·margin)`，
    /// 其中 `norm(d) = (d − min)/(max − min)`（max==min 时全部置中线）。
    pub fn polylines(&self, width: u32, height: u32) -> Vec<Vec<Point>> {
        let n = self.data.len();
        if n == 0 || width == 0 || height == 0 {
            return Vec::new();
        }
        let m = self.margin as f32;
        let h = height as f32;
        let w = width as f32;
        let (lo, hi) = self.data.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let span = hi - lo;
        let norm = |d: f32| -> f32 {
            if span.abs() < 1e-9 {
                0.5
            } else {
                (d - lo) / span
            }
        };
        let py = |d: f32| m + (1.0 - norm(d)) * (h - 2.0 * m);
        let dx = if n > 1 { (w - 2.0 * m) / (n as f32 - 1.0) } else { 0.0 };

        let mut out: Vec<Vec<Point>> = Vec::new();
        match self.kind {
            ChartKind::Line => {
                let line: Vec<Point> = (0..n)
                    .map(|i| Point::new(m + dx * i as f32, py(self.data[i])))
                    .collect();
                out.push(line);
            }
            ChartKind::Bar => {
                let bw = if n > 1 { (w - 2.0 * m) / n as f32 } else { w - 2.0 * m };
                let base_y = m + (1.0 - norm(self.baseline)) * (h - 2.0 * m);
                for (i, &d) in self.data.iter().enumerate() {
                    let x0 = m + bw * i as f32 + bw * 0.15;
                    let x1 = m + bw * i as f32 + bw * 0.85;
                    let y = py(d);
                    out.push(vec![
                        Point::new(x0, y),
                        Point::new(x1, y),
                        Point::new(x1, base_y),
                        Point::new(x0, base_y),
                        Point::new(x0, y),
                    ]);
                }
            }
        }
        out
    }

    /// 生成图表覆盖率缓冲。
    pub fn rasterize(&self, width: u32, height: u32) -> Coverage {
        let polys = self.polylines(width, height);
        rasterize_lines(&polys, width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_line_maps_endpoints_to_margins() {
        let spec = ChartSpec {
            kind: ChartKind::Line,
            data: vec![0.0, 5.0, 10.0],
            baseline: 0.0,
            margin: 10,
        };
        let polys = spec.polylines(120, 100);
        assert_eq!(polys.len(), 1);
        let line = &polys[0];
        assert_eq!(line.len(), 3);
        assert!((line[0].x - 10.0).abs() < 1e-4);
        assert!((line[2].x - 110.0).abs() < 1e-4);
        // 最低数据在底边、最高在顶边（y 向下）。
        assert!((line[0].y - 90.0).abs() < 1e-4);
        assert!((line[2].y - 10.0).abs() < 1e-4);
    }

    #[test]
    fn chart_deterministic() {
        let spec = ChartSpec {
            kind: ChartKind::Bar,
            data: vec![3.0, 1.0, 4.0, 1.0, 5.0],
            baseline: 0.0,
            margin: 6,
        };
        let a = spec.rasterize(100, 80);
        let b = spec.rasterize(100, 80);
        assert_eq!(a.data, b.data);
        assert!(a.data.iter().sum::<f32>() > 0.0);
    }

    #[test]
    fn empty_chart_is_empty() {
        let spec = ChartSpec::default();
        assert!(spec.polylines(50, 50).is_empty());
        assert_eq!(spec.rasterize(50, 50).data.iter().sum::<f32>(), 0.0);
    }
}
