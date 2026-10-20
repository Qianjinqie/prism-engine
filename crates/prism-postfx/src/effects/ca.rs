//! CA —— chromatic aberration（清单：CA —— 横向色散）。
//!
//! 语义权威：CA与（PostFX
//! 卷积族效果特解）；宪章 C5（行业通用词汇）。
//!
//! # 实现
//!
//! 横向色散（lateral CA）：R 通道采样自 `p + shift`、B 采样自 `p − shift`、
//! G/A 原位——色散沿固定向量平移采样，**可平移不变**（tile 一致性验证集
//! 成员，关卡 1）。亚像素位移经共享双线性原语（格点恒等、固定权重式）；
//! 整数位移时退化为位模式搬运。径向（镜头离心式）CA 需全局中心上下文，
//! 当前 trait 不携带——缺口如实上报。
//!
//! halo = `max(|shift.x|, |shift.y|)` 上取整。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::bilinear_sample_offset;

/// CA 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaticAberration {
    /// 色散向量（像素；R 沿 +shift、B 沿 −shift 采样）。
    pub shift: (f32, f32),
}

impl Default for ChromaticAberration {
    fn default() -> Self {
        ChromaticAberration { shift: (1.0, 0.0) }
    }
}

impl ChromaticAberration {
    /// 亚像素位移读取半径（halo 申报依据）。
    fn read_radius(&self) -> u32 {
        self.shift.0.abs().max(self.shift.1.abs()).ceil() as u32
    }
}

impl TileEffect for ChromaticAberration {
    fn halo_radius(&self) -> u32 {
        self.read_radius()
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let (sx, sy) = self.shift;
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let r = bilinear_sample_offset(window, x, y, sx, sy)[0];
                let b = bilinear_sample_offset(window, x, y, -sx, -sy)[2];
                let c = window.get(x, y);
                out.set(x, y, [r, c[1], b, c[3]]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_shift_is_bit_exact_identity() {
        let mut win = Fbuf16F::new(8, 6);
        for y in 0..6u32 {
            for x in 0..8u32 {
                let v = (x * 3 + y * 7) as f32 / 40.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let ca = ChromaticAberration { shift: (0.0, 0.0) };
        assert_eq!(ca.eval_window(&win), win, "零位移必须逐位恒等");
        assert_eq!(ca.halo_radius(), 0);
    }

    #[test]
    fn integral_shift_moves_channels_exactly() {
        // 整数位移 = 通道位模式搬运：R 取 x+3 处、B 取 x−3 处、G 原位。
        let mut win = Fbuf16F::new(12, 4);
        for y in 0..4u32 {
            for x in 0..12u32 {
                let v = (x * 13 + y) as f32 / 64.0;
                win.set(x, y, [v, 1.0 - v, v * 0.25, 1.0]);
            }
        }
        let ca = ChromaticAberration { shift: (3.0, 0.0) };
        let out = ca.eval_window(&win);
        for y in 0..4u32 {
            for x in 0..12u32 {
                let rx = (x as i64 + 3).clamp(0, 11) as u32;
                let bx = (x as i64 - 3).clamp(0, 11) as u32;
                assert_eq!(out.get(x, y)[0], win.get(rx, y)[0], "R 通道位移不符 ({x},{y})");
                assert_eq!(out.get(x, y)[2], win.get(bx, y)[2], "B 通道位移不符 ({x},{y})");
                assert_eq!(out.get(x, y)[1], win.get(x, y)[1], "G 通道必须原位");
            }
        }
        assert_eq!(ca.halo_radius(), 3, "halo = 位移半径");
    }

    /// 测试期望的 f16 量化（实现写回量化 f16，期望须过同一域——格点采样
    /// 无损，亚像素采样存在量化差）。
    fn q(v: f32) -> f32 {
        prism_io::f16_to_f32(prism_io::f32_to_f16(v))
    }

    #[test]
    fn subpixel_shift_uses_shared_bilinear() {
        // 半像素位移 = 相邻像素均值（共享双线性原语，固定权重式）。
        let mut win = Fbuf16F::new(6, 4);
        for y in 0..4u32 {
            for x in 0..6u32 {
                let v = x as f32 * 0.1;
                win.set(x, y, [v, 0.0, 0.0, 1.0]);
            }
        }
        let ca = ChromaticAberration { shift: (0.5, 0.0) };
        let out = ca.eval_window(&win);
        for y in 0..4u32 {
            for x in 0..6u32 {
                // 半像素位移的解析期望：权重 (0.5, 0.5)、y 向权重 0/1 精确
                // ⇒ out = c(x)·0.5 + c(x+1)·0.5（与实现同一表达式，逐位同）。
                let expected =
                    win.get(x, y)[0] * 0.5 + win.get((x + 1).min(5), y)[0] * 0.5;
                assert_eq!(
                    out.get(x, y)[0],
                    q(expected),
                    "亚像素 R 必须等于共享双线性值（f16 量化域）"
                );
            }
        }
        // 纯函数契约。
        assert_eq!(ca.eval_window(&win), ca.eval_window(&win));
    }
}
