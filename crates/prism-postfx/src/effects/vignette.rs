//! vignette（B4 清单：vignette——径向压暗）。
//!
//! 语义权威：《强大版路线图》B4 行（vignette）；宪章 C4（行业概念：AE /
//! 达芬奇 Lens Vignette 同名概念，词汇直用）。
//!
//! # 实现
//!
//! 以**窗口局部中心**为圆心的径向衰减（位置键控，整幅求值为规范语义——
//! 见 `effects::mod` 说明）：归一化半径 `r ∈ [0, 1]`（0 = 中心，1 = 半对角
//! 端点），`inner` 为衰减起点占比，过渡带内 smoothstep
//! `s = t²(3 − 2t)`（纯多项式），衰减系数 `factor = 1 − strength·s`，
//! RGB 乘算、A 通道不动（晕影是光度效应，不改透明度）。
//!
//! halo = 0（逐像素，无邻域读）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// vignette 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vignette {
    /// 最大压暗强度（0 = 无效果，1 = 过渡带端点压到全黑）。
    pub strength: f32,
    /// 衰减起点（归一化半径占比；0 = 从圆心开始，典型 0.4–0.7）。
    pub inner: f32,
}

impl Default for Vignette {
    fn default() -> Self {
        Vignette { strength: 0.5, inner: 0.55 }
    }
}

impl TileEffect for Vignette {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        // 窗口局部中心与半对角（r 的归一化基准；sqrt 为 IEEE 精确运算）。
        let cx = (window.width as f32 - 1.0) * 0.5;
        let cy = (window.height as f32 - 1.0) * 0.5;
        let half_diag = (cx * cx + cy * cy).sqrt();
        let band = 1.0 - self.inner;

        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt() / half_diag;
                let t = prism_math::clamp((r - self.inner) / band, 0.0, 1.0);
                let s = t * t * (3.0 - 2.0 * t);
                let factor = 1.0 - self.strength * s;
                let c = window.get(x, y);
                out.set(x, y, [c[0] * factor, c[1] * factor, c[2] * factor, c[3]]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint2_center_stays_bright_corners_darken_radially() {
        // 关卡 2：中心亮四角暗，且沿半径单调变暗（径向断言）。
        let w = 33u32;
        let h = 33u32;
        let mut win = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                win.set(x, y, [0.8, 0.8, 0.8, 1.0]);
            }
        }
        let vig = Vignette { strength: 0.8, inner: 0.3 };
        let out = vig.eval_window(&win);

        let center = out.get(16, 16)[0];
        let mid = out.get(26, 16)[0]; // 沿 +x 离心
        let corner = out.get(0, 0)[0];
        assert_eq!(
            center,
            win.get(16, 16)[0],
            "中心（r=0 < inner）必须保持原亮度（与输入 f16 位模式一致）"
        );
        assert!(mid < center, "离心点必须更暗：{mid} < {center}");
        assert!(corner < mid, "角点必须最暗：{corner} < {mid}");
        // 对角对称：四个角一致（径向对称性）。
        assert_eq!(corner, out.get(32, 0)[0]);
        assert_eq!(corner, out.get(0, 32)[0]);
        assert_eq!(corner, out.get(32, 32)[0]);
    }

    #[test]
    fn strength_zero_is_bit_exact_identity() {
        // strength 0 → factor 恒 1.0 → 乘 1.0 位精确恒等。
        let mut win = Fbuf16F::new(9, 5);
        for y in 0..5u32 {
            for x in 0..9u32 {
                let v = (x + y) as f32 / 16.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let out = Vignette { strength: 0.0, inner: 0.3 }.eval_window(&win);
        assert_eq!(out, win, "strength 0 必须逐位恒等");
        // alpha 通道不受晕影影响。
        let out2 = Vignette { strength: 1.0, inner: 0.0 }.eval_window(&win);
        for y in 0..5u32 {
            for x in 0..9u32 {
                assert_eq!(out2.get(x, y)[3], win.get(x, y)[3], "alpha 必须原样");
            }
        }
    }

    #[test]
    fn vignette_is_pure() {
        let mut win = Fbuf16F::new(12, 8);
        let mut state: u64 = 0x7E51;
        for y in 0..8u32 {
            for x in 0..12u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let vig = Vignette::default();
        assert_eq!(vig.eval_window(&win), vig.eval_window(&win), "纯函数契约");
    }
}
