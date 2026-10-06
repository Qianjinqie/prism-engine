//! 素描（B14 NPR 族：铅笔 / 炭笔素描）。
//!
//! 语义权威：《强大版路线图》B14 行（素描）；《算法白皮书》§14 框架下
//! "边缘 = 梯度算子"的散射态——素描 = 反相边缘图叠加于纸色。
//!
//! # 实现
//!
//! 1. **边缘**：中心差分梯度幅值 `g`（[`super::gradient_mag`]）；
//! 2. **纸色混合**：`out = paper · (1 − shade) + ink · shade`，其中
//!    `shade = clamp(gain · g, 0, 1)` 为笔墨浓度——边缘越强笔画越深；
//!    另可选亮度倒相项 `invert_luma`（暗部先铺一层浅灰，模拟炭笔底）。
//!
//! halo = 1（3×3 梯度）；零超越函数。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::{gradient_mag, luma};

/// 素描效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sketch {
    /// 梯度增益（笔墨浓度 = gain·grad 钳到 1）。
    pub gain: f32,
    /// 纸色。
    pub paper: [f32; 3],
    /// 墨色（笔画色）。
    pub ink: [f32; 3],
    /// 暗部底灰权重（0 = 纯边缘素描，1 = 明暗全用）。
    pub invert_luma: f32,
}

impl Default for Sketch {
    fn default() -> Self {
        Sketch {
            gain: 4.0,
            paper: [0.96, 0.95, 0.92],
            ink: [0.10, 0.10, 0.12],
            invert_luma: 0.0,
        }
    }
}

impl TileEffect for Sketch {
    fn halo_radius(&self) -> u32 {
        1
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let g = gradient_mag(window, x, y);
                let mut shade = prism_math::clamp(self.gain * g, 0.0, 1.0);
                // 暗部底灰：亮度越低底灰越重。
                let inv = prism_math::clamp(self.invert_luma, 0.0, 1.0);
                if inv > 0.0 {
                    let l = prism_math::clamp(luma(c), 0.0, 1.0);
                    shade = prism_math::clamp(shade + (1.0 - l) * inv, 0.0, 1.0);
                }
                let rgb = [
                    self.paper[0] * (1.0 - shade) + self.ink[0] * shade,
                    self.paper[1] * (1.0 - shade) + self.ink[1] * shade,
                    self.paper[2] * (1.0 - shade) + self.ink[2] * shade,
                ];
                out.set(x, y, [rgb[0], rgb[1], rgb[2], c[3]]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_region_is_paper_exact() {
        // 平坦区无梯度 → shade 0 → 输出 = 纸色（逐位）。
        let mut win = Fbuf16F::new(8, 8);
        for y in 0..8u32 {
            for x in 0..8u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let s = Sketch::default();
        let out = s.eval_window(&win);
        let want = [
            prism_io::f16_to_f32(prism_io::f32_to_f16(s.paper[0])),
            prism_io::f16_to_f32(prism_io::f32_to_f16(s.paper[1])),
            prism_io::f16_to_f32(prism_io::f32_to_f16(s.paper[2])),
            1.0,
        ];
        for y in 1..7u32 {
            for x in 1..7u32 {
                assert_eq!(out.get(x, y), want, "平坦区应为纸色");
            }
        }
    }

    #[test]
    fn edges_become_ink() {
        // 强边缘处 shade → 1 → 输出 ≈ 墨色。
        let mut win = Fbuf16F::new(10, 10);
        for y in 0..10u32 {
            for x in 0..10u32 {
                let v = if x < 5 { 0.0 } else { 1.0 };
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let s = Sketch::default();
        let out = s.eval_window(&win);
        let edge = out.get(4, 5);
        assert!(luma(edge) < 0.3, "边缘应为墨色，得到 {edge:?}");
    }

    #[test]
    fn sketch_deterministic_and_halo_one() {
        let mut win = Fbuf16F::new(12, 12);
        let mut state: u64 = 0x5E7C;
        for y in 0..12u32 {
            for x in 0..12u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.6, v * 0.2, 1.0]);
            }
        }
        let s = Sketch::default();
        assert_eq!(s.halo_radius(), 1);
        assert_eq!(s.eval_window(&win), s.eval_window(&win), "纯函数契约");
    }
}
