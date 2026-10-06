//! Kuwahara（B14 NPR 族：油画笔触保边滤波）。
//!
//! 语义权威：《算法白皮书》§14 第 2 条——"广义 Kuwahara——像素邻域分区（8
//! 象限圆盘）各自算均值/方差，取方差最小的均值——笔画感保边缘"。
//!
//! # 实现
//!
//! 以 `(2r+1)²` 方窗为邻域，按**四象限**（左上 / 右上 / 左下 / 右下）分区，
//! 各象限统计均值与方差（方差 = E[x²] − E[x]²，零超越），取方差最小的象限
//! 均值作为输出。方差比较用 f64 消除累积舍入（跨平台一致）；像素内固定序
//! 归约（象限内行主序升序）。
//!
//! halo = r。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// Kuwahara 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kuwahara {
    /// 邻域半径（象限为 r×r 方块）。
    pub radius: u32,
}

impl Default for Kuwahara {
    fn default() -> Self {
        Kuwahara { radius: 2 }
    }
}

impl TileEffect for Kuwahara {
    fn halo_radius(&self) -> u32 {
        self.radius
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        let r = self.radius;
        for y in 0..window.height {
            for x in 0..window.width {
                let mut best_mean = [0.0f32; 4];
                let mut best_var = f64::INFINITY;
                // 四象限：dx ∈ {−r..0, 0..r}，dy 同（含中心像素归属左上）。
                for qy in 0..2i64 {
                    for qx in 0..2i64 {
                        let (x_lo, x_hi) = if qx == 0 {
                            (-(r as i64), 0i64)
                        } else {
                            (0i64, r as i64)
                        };
                        let (y_lo, y_hi) = if qy == 0 {
                            (-(r as i64), 0i64)
                        } else {
                            (0i64, r as i64)
                        };
                        let mut sum = [0.0f64; 4];
                        let mut sum_sq = [0.0f64; 4];
                        let mut n = 0.0f64;
                        for dy in y_lo..=y_hi {
                            for dx in x_lo..=x_hi {
                                let sx = (x as i64 + dx).clamp(0, window.width as i64 - 1) as u32;
                                let sy = (y as i64 + dy).clamp(0, window.height as i64 - 1) as u32;
                                let c = window.get(sx, sy);
                                for k in 0..4 {
                                    let v = c[k] as f64;
                                    sum[k] += v;
                                    sum_sq[k] += v * v;
                                }
                                n += 1.0;
                            }
                        }
                        // 亮度通道方差做主判据（RGB 方差同序，白皮书取亮度结构）。
                        let mean_l = (sum[0] + sum[1] + sum[2]) / (3.0 * n);
                        let mean_sq_l = (sum_sq[0] + sum_sq[1] + sum_sq[2]) / (3.0 * n);
                        let var = mean_sq_l - mean_l * mean_l;
                        if var < best_var {
                            best_var = var;
                            for k in 0..4 {
                                best_mean[k] = (sum[k] / n) as f32;
                            }
                        }
                    }
                }
                out.set(x, y, best_mean);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_hard_edge_in_flat_regions() {
        // 硬阶跃边缘：两侧平坦区内，Kuwahara 应保持原值（选到零方差象限）。
        let mut win = Fbuf16F::new(16, 8);
        for y in 0..8u32 {
            for x in 0..16u32 {
                let v = if x < 8 { 0.2 } else { 0.8 };
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let out = Kuwahara { radius: 2 }.eval_window(&win);
        // 远离边缘处（x=2, x=13）应保边不模糊。
        assert!((out.get(2, 4)[0] - 0.2).abs() < 0.01, "左侧平坦区应保 0.2");
        assert!((out.get(13, 4)[0] - 0.8).abs() < 0.01, "右侧平坦区应保 0.8");
    }

    #[test]
    fn flat_input_is_fixed_point() {
        let mut win = Fbuf16F::new(9, 9);
        for y in 0..9u32 {
            for x in 0..9u32 {
                win.set(x, y, [0.4, 0.5, 0.6, 1.0]);
            }
        }
        let out = Kuwahara { radius: 3 }.eval_window(&win);
        // f16 量化容差：期望值须过同一写回域。
        let want = [
            prism_io::f16_to_f32(prism_io::f32_to_f16(0.4)),
            prism_io::f16_to_f32(prism_io::f32_to_f16(0.5)),
            prism_io::f16_to_f32(prism_io::f32_to_f16(0.6)),
            1.0,
        ];
        for y in 0..9u32 {
            for x in 0..9u32 {
                assert_eq!(out.get(x, y), want, "常量场不动点");
            }
        }
    }

    #[test]
    fn halo_and_purity() {
        let mut win = Fbuf16F::new(12, 10);
        let mut state: u64 = 0x4B77;
        for y in 0..10u32 {
            for x in 0..12u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let k = Kuwahara { radius: 2 };
        assert_eq!(k.halo_radius(), 2);
        assert_eq!(k.eval_window(&win), k.eval_window(&win), "纯函数契约");
    }
}
