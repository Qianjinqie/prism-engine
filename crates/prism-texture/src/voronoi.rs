//! Voronoi 纹理（程序化纹理族：Voronoi）。
//!
//! 语义权威：行业 Voronoi / Worley 噪声节点；（种子派生：
//! `hash(seed, 胞元格点)`）。在胞元格点上派生随机特征点，对每个像素取最近
//! 特征点的距离 F1（及第二近 F2）——输出距离场、F2−F1 边缘、或胞元随机色。
//!
//! # 确定性
//!
//! 特征点位置 = 胞元格点 `(cx, cy)` 经 PCG hash 派生（纯函数）⇒ 同 seed
//! 逐位一致；胞元色同源派生。零超越函数（只用 `sqrt`/`floor`/整数运算）。
//! 邻域搜索固定 3×3 胞元（覆盖最近点），固定遍历序。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// Voronoi 输出模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoronoiMode {
    /// F1 距离场（灰度）。
    F1,
    /// F2 − F1（胞元边界处趋 0 → 边缘线）。
    Edges,
    /// 胞元随机色（每胞元一色）。
    Cells,
}

/// Voronoi 纹理。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voronoi {
    /// 胞元尺寸（像素）。
    pub cell: f32,
    /// 输出模式。
    pub mode: VoronoiMode,
    /// 种子。
    pub seed: u64,
    /// 距离场调色下限色。
    pub color_a: [f32; 3],
    /// 距离场调色上限色。
    pub color_b: [f32; 3],
}

impl Default for Voronoi {
    fn default() -> Self {
        Voronoi {
            cell: 24.0,
            mode: VoronoiMode::F1,
            seed: 0,
            color_a: [0.0, 0.0, 0.0],
            color_b: [1.0, 1.0, 1.0],
        }
    }
}

impl Voronoi {
    /// 胞元 `(cx, cy)` 的特征点偏移（∈ [0,1)²，PCG hash 派生）。
    fn feature(seed: u64, cx: i64, cy: i64) -> (f32, f32) {
        let hx = prism_noise::pcg::hash2_f32(seed, cx as u64, cy as u64);
        let hy = prism_noise::pcg::hash2_f32(seed ^ 0x51D3, cx as u64, cy as u64);
        (hx, hy)
    }

    /// 胞元随机色（PCG hash 派生）。
    fn cell_color(seed: u64, cx: i64, cy: i64) -> [f32; 3] {
        [
            prism_noise::pcg::hash2_f32(seed ^ 0xC01A, cx as u64, cy as u64),
            prism_noise::pcg::hash2_f32(seed ^ 0xC01B, cx as u64, cy as u64),
            prism_noise::pcg::hash2_f32(seed ^ 0xC01C, cx as u64, cy as u64),
        ]
    }

    /// 计算 (x, y) 的最近 (F1)、次近 (F2) 距离与最近胞元索引。
    fn eval_point(&self, x: u32, y: u32) -> (f32, f32, (i64, i64)) {
        let cell = self.cell.max(1.0);
        let gx = x as f32 / cell;
        let gy = y as f32 / cell;
        let base_cx = gx.floor() as i64;
        let base_cy = gy.floor() as i64;
        let mut f1 = f32::INFINITY;
        let mut f2 = f32::INFINITY;
        let mut nearest = (base_cx, base_cy);
        // 固定 3×3 胞元搜索（不足时最近的仍在其中）。
        for oy in -1i64..=1 {
            for ox in -1i64..=1 {
                let cx = base_cx + ox;
                let cy = base_cy + oy;
                let (fx, fy) = Self::feature(self.seed, cx, cy);
                let px = (cx as f32 + fx) * cell;
                let py = (cy as f32 + fy) * cell;
                let dx = px - x as f32;
                let dy = py - y as f32;
                let d = (dx * dx + dy * dy).sqrt();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    nearest = (cx, cy);
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
        (f1, f2, nearest)
    }
}

impl TileEffect for Voronoi {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        let norm = self.cell.max(1.0);
        for y in 0..window.height {
            for x in 0..window.width {
                let (f1, f2, idx) = self.eval_point(x, y);
                let rgb = match self.mode {
                    VoronoiMode::F1 => {
                        let t = prism_math::clamp(f1 / norm, 0.0, 1.0);
                        [
                            self.color_a[0] + (self.color_b[0] - self.color_a[0]) * t,
                            self.color_a[1] + (self.color_b[1] - self.color_a[1]) * t,
                            self.color_a[2] + (self.color_b[2] - self.color_a[2]) * t,
                        ]
                    }
                    VoronoiMode::Edges => {
                        let t = prism_math::clamp((f2 - f1) / norm, 0.0, 1.0);
                        [
                            self.color_a[0] + (self.color_b[0] - self.color_a[0]) * t,
                            self.color_a[1] + (self.color_b[1] - self.color_a[1]) * t,
                            self.color_a[2] + (self.color_b[2] - self.color_a[2]) * t,
                        ]
                    }
                    VoronoiMode::Cells => Self::cell_color(self.seed, idx.0, idx.1),
                };
                out.set(x, y, [rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let win = Fbuf16F::new(40, 40);
        for mode in [VoronoiMode::F1, VoronoiMode::Edges, VoronoiMode::Cells] {
            let v = Voronoi { mode, ..Voronoi::default() };
            assert_eq!(v.eval_window(&win), v.eval_window(&win), "{mode:?} 纯函数");
            let s = Voronoi { seed: 9, ..v }.eval_window(&win);
            assert_ne!(v.eval_window(&win).pixels, s.pixels, "{mode:?} seed 去相关");
        }
    }

    #[test]
    fn f1_distance_zero_at_feature_points_and_bounded() {
        // F1 距离场处处有限、非负、且存在接近 0 的点（特征点处）。
        let v = Voronoi::default();
        let out = v.eval_window(&Fbuf16F::new(48, 48));
        let mut min = f32::INFINITY;
        for y in 0..48u32 {
            for x in 0..48u32 {
                let c = out.get(x, y);
                for k in 0..3 {
                    assert!(c[k] >= 0.0 && c[k] <= 1.0, "越界 {c:?}");
                }
                min = min.min(c[0]);
            }
        }
        assert!(min < 0.1, "应有接近 0 的 F1（特征点处）：min = {min}");
    }

    #[test]
    fn edges_mode_has_dark_boundaries() {
        // Edges 模式：胞元边界（F2−F1 小）暗、胞元内部亮。
        let v = Voronoi { mode: VoronoiMode::Edges, ..Voronoi::default() };
        let out = v.eval_window(&Fbuf16F::new(48, 48));
        let mut min = 1.0f32;
        let mut max = 0.0f32;
        for y in 0..48u32 {
            for x in 0..48u32 {
                let c = out.get(x, y)[0];
                min = min.min(c);
                max = max.max(c);
            }
        }
        assert!(max > min + 0.1, "边界与内部必须有对比：{min}..{max}");
    }

    #[test]
    fn cells_mode_uses_few_distinct_colors() {
        let v = Voronoi { mode: VoronoiMode::Cells, cell: 16.0, ..Voronoi::default() };
        let out = v.eval_window(&Fbuf16F::new(48, 48));
        let mut seen = std::collections::BTreeSet::new();
        for y in 0..48u32 {
            for x in 0..48u32 {
                let c = out.get(x, y);
                seen.insert(((c[0] * 255.0) as u32, (c[1] * 255.0) as u32, (c[2] * 255.0) as u32));
            }
        }
        // 胞元数有限（48/16 = 3×3 邻域若干），颜色数应有界且 > 1。
        assert!(seen.len() > 1 && seen.len() < 50, "胞元色数应有限：{}", seen.len());
    }
}
