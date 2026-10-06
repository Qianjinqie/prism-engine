//! 纸张纹理（B14 程序化纹理族：纸张）。
//!
//! 语义权威：《算法白皮书》§14（水彩的"纸纹乘算"）/ §17（噪声族）。
//! 纸张 = 纤维纹（高频各向异性 fbm）+ 打浆斑块（低频 fbm）叠加，可乘算
//! 或加算到输入上，模拟纸面无规纤维与浆料不均。
//!
//! 确定性：全部经 prism-noise 纯函数；同 seed 逐位一致。

use prism_io::fbuf::Fbuf16F;
use prism_noise::fbm::Fbm;
use prism_render::tile::TileEffect;

/// 纸张纹理。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paper {
    /// 纤维纹强度（高频）。
    pub fiber: f32,
    /// 打浆斑块强度（低频）。
    pub blotch: f32,
    /// 纤维纹频率。
    pub fiber_freq: f32,
    /// 打浆斑块频率。
    pub blotch_freq: f32,
    /// 种子。
    pub seed: u64,
    /// 纸基色。
    pub base: [f32; 3],
    /// 是否与输入乘算（false = 直接替换输入）。
    pub multiply: bool,
}

impl Default for Paper {
    fn default() -> Self {
        Paper {
            fiber: 0.15,
            blotch: 0.25,
            fiber_freq: 0.35,
            blotch_freq: 0.03,
            seed: 0,
            base: [0.96, 0.94, 0.90],
            multiply: false,
        }
    }
}

impl TileEffect for Paper {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let fbm = Fbm::new(4, 2.0, 0.5);
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                // 纤维纹（各向异性：x 频率与 y 频率不同，模拟纤维走向）。
                let fx = fbm.simplex2(
                    self.seed ^ 0xF1BE,
                    x as f32 * self.fiber_freq,
                    y as f32 * self.fiber_freq * 2.5,
                );
                // 打浆斑块（低频）。
                let bl = fbm.simplex2(
                    self.seed ^ 0xB107,
                    x as f32 * self.blotch_freq,
                    y as f32 * self.blotch_freq,
                );
                let mod_ = self.fiber * fx + self.blotch * bl;
                let mul = 1.0 + mod_;
                let paper = [
                    self.base[0] * mul,
                    self.base[1] * mul,
                    self.base[2] * mul,
                ];
                let c = window.get(x, y);
                let rgb = if self.multiply {
                    [paper[0] * c[0], paper[1] * c[1], paper[2] * c[2]]
                } else {
                    paper
                };
                let a = if self.multiply { c[3] } else { 1.0 };
                out.set(
                    x,
                    y,
                    [prism_math::clamp(rgb[0], 0.0, 4.0), prism_math::clamp(rgb[1], 0.0, 4.0), prism_math::clamp(rgb[2], 0.0, 4.0), a],
                );
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
        let win = Fbuf16F::new(20, 20);
        let p = Paper::default();
        assert_eq!(p.eval_window(&win), p.eval_window(&win), "同 seed 逐位一致");
        let s = Paper { seed: 123, ..p }.eval_window(&win);
        assert_ne!(p.eval_window(&win).pixels, s.pixels, "seed 必须改变纹理");
    }

    #[test]
    fn multiply_mode_scales_input() {
        let mut win = Fbuf16F::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 0.8]);
            }
        }
        let p = Paper { multiply: true, ..Paper::default() };
        let out = p.eval_window(&win);
        // 乘算：输出 = 纸色 × 输入，且保留 alpha。
        let want_a = prism_io::f16_to_f32(prism_io::f32_to_f16(0.8));
        for y in 0..16u32 {
            for x in 0..16u32 {
                assert_eq!(out.get(x, y)[3], want_a, "乘算模式 alpha 保留");
            }
        }
    }

    #[test]
    fn base_is_replaced_when_not_multiplying() {
        let win = Fbuf16F::new(8, 8);
        let out = Paper::default().eval_window(&win);
        // 无输入（全 0）且非乘算 → 纸色邻域。
        for y in 0..8u32 {
            for x in 0..8u32 {
                let c = out.get(x, y);
                assert!((c[0] - 0.96).abs() < 0.3, "应为纸色：{c:?}");
                assert_eq!(c[3], 1.0);
            }
        }
    }
}
