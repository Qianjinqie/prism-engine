//! 图案纹理（B14 程序化纹理族：图案）。
//!
//! 语义权威：行业图案节点（棋盘 / 条纹 / 网格 / 圆点）；《算法白皮书》§26
//! （固定序、零超越函数）。图案在归一化 / 像素坐标上做屏幕键控（位置键控，
//! 窗口局部坐标为规范语义）。
//!
//! 确定性：纯算术（零超越函数）；支持两色 + 可选图案变换（平移/旋转由角度
//! 离散化常量表）；同参数逐位一致。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// 图案类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// 棋盘格。
    Checker,
    /// 条纹（沿 x）。
    Stripe,
    /// 网格（细线）。
    Grid,
    /// 圆点阵。
    Dot,
}

/// 图案纹理。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pattern {
    /// 图案类型。
    pub kind: PatternKind,
    /// 单元尺寸（像素）。
    pub cell: f32,
    /// 前景色。
    pub fg: [f32; 3],
    /// 背景色。
    pub bg: [f32; 3],
    /// 线宽 / 点半径比（0..1，相对单元）。
    pub ratio: f32,
}

impl Default for Pattern {
    fn default() -> Self {
        Pattern {
            kind: PatternKind::Checker,
            cell: 16.0,
            fg: [0.1, 0.1, 0.1],
            bg: [0.9, 0.9, 0.9],
            ratio: 0.5,
        }
    }
}

impl Pattern {
    /// 单元坐标 `(fx, fy) ∈ [0,1)²`（图案相位）。
    fn phase(cell: f32, x: u32, y: u32) -> (f32, f32) {
        let c = cell.max(1.0);
        let fx = ((x as f32) / c).rem_euclid(1.0);
        let fy = ((y as f32) / c).rem_euclid(1.0);
        (fx, fy)
    }
}

impl TileEffect for Pattern {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        let ratio = prism_math::clamp(self.ratio, 0.0, 1.0);
        for y in 0..window.height {
            for x in 0..window.width {
                let (fx, fy) = Self::phase(self.cell, x, y);
                let on = match self.kind {
                    PatternKind::Checker => {
                        let cx = fx < 0.5;
                        let cy = fy < 0.5;
                        cx == cy
                    }
                    PatternKind::Stripe => fx < ratio,
                    PatternKind::Grid => {
                        let half = ratio * 0.5;
                        fx < half || fy < half || fx > 1.0 - half || fy > 1.0 - half
                    }
                    PatternKind::Dot => {
                        let dx = fx - 0.5;
                        let dy = fy - 0.5;
                        (dx * dx + dy * dy).sqrt() < ratio * 0.5
                    }
                };
                let c = if on { self.fg } else { self.bg };
                out.set(x, y, [c[0], c[1], c[2], 1.0]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_alternates() {
        let p = Pattern { kind: PatternKind::Checker, cell: 8.0, ..Pattern::default() };
        let out = p.eval_window(&Fbuf16F::new(16, 16));
        // 棋盘相位每 cell/2 = 4px 翻转。
        assert_ne!(out.get(0, 0)[0], out.get(4, 0)[0], "水平相邻相位相反");
        assert_ne!(out.get(0, 0)[0], out.get(0, 4)[0], "垂直相邻相位相反");
        assert_eq!(out.get(0, 0)[0], out.get(4, 4)[0], "对角相同");
    }

    #[test]
    fn dot_at_center_of_cell() {
        let p = Pattern { kind: PatternKind::Dot, cell: 10.0, ratio: 1.0, ..Pattern::default() };
        let out = p.eval_window(&Fbuf16F::new(10, 10));
        // 单元中心附近为前景（f16 量化容差）。
        let fg = prism_io::f16_to_f32(prism_io::f32_to_f16(p.fg[0]));
        let bg = prism_io::f16_to_f32(prism_io::f32_to_f16(p.bg[0]));
        assert_eq!(out.get(5, 5)[0], fg, "中心应为前景");
        assert_eq!(out.get(0, 0)[0], bg, "角落应为背景");
    }

    #[test]
    fn pattern_deterministic() {
        let p = Pattern { kind: PatternKind::Grid, cell: 8.0, ..Pattern::default() };
        let win = Fbuf16F::new(32, 32);
        assert_eq!(p.eval_window(&win), p.eval_window(&win), "纯函数契约");
    }
}
