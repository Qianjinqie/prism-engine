//! Mandelbrot_bg（清单：Mandelbrot_bg —— Mandelbrot 集程序化背景）。
//!
//! 语义权威：Mandelbrot_bg；域 M（生成艺术输出
//! 可再加工——背景板语义同 plasma）。
//!
//! # 实现
//!
//! 逃逸时间迭代：窗口局部坐标归一（中心 = 窗口几何中心）后 `c = p·scale +
//! center_offset`，`z₀ = 0`，迭代 `z ← z² + c` 至 `|z|² > 4` 或 `max_iter`
//! 上限（固定迭代上限 = 确定性；纯算术，无超越函数）。逃逸点按
//! `iter / max_iter` 进三段调色板（`effects::mod` 共享原语），集合内点为黑。
//! 生成器语义：替换输入窗口，A 通道写 1。
//!
//! 位置键控（窗口局部坐标）；halo = 0。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::palette3;

/// 调色板三锚点（深空 → 靛蓝 → 琥珀；固定常量）。
const STOP_A: [f32; 3] = [0.0, 0.0, 0.05];
const STOP_B: [f32; 3] = [0.10, 0.20, 0.60];
const STOP_C: [f32; 3] = [1.00, 0.80, 0.30];

/// Mandelbrot_bg 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MandelbrotBackground {
    /// 坐标缩放（复平面单位/像素）。
    pub scale: f32,
    /// 复平面中心偏移 `(re, im)`（窗口中心对应的复平面点）。
    pub center: (f32, f32),
    /// 最大迭代数（固定上限 = 确定性与成本标价）。
    pub max_iter: u32,
}

impl Default for MandelbrotBackground {
    fn default() -> Self {
        MandelbrotBackground { scale: 0.02, center: (0.0, 0.0), max_iter: 64 }
    }
}

impl TileEffect for MandelbrotBackground {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let cx = (window.width as f32 - 1.0) * 0.5;
        let cy = (window.height as f32 - 1.0) * 0.5;
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let cre = (x as f32 - cx) * self.scale + self.center.0;
                let cim = (y as f32 - cy) * self.scale + self.center.1;
                let (mut zx, mut zy) = (0.0f32, 0.0f32);
                let mut escaped_at = 0u32;
                for i in 0..self.max_iter {
                    let (zx2, zy2) = (zx * zx, zy * zy);
                    if zx2 + zy2 > 4.0 {
                        escaped_at = i;
                        break;
                    }
                    zy = 2.0 * zx * zy + cim;
                    zx = zx2 - zy2 + cre;
                }
                let rgb = if escaped_at == 0 {
                    [0.0, 0.0, 0.0]
                } else {
                    palette3(escaped_at as f32 / self.max_iter as f32, STOP_A, STOP_B, STOP_C)
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
    fn set_interior_is_black_escape_is_graded() {
        // 集合内（c=0，z 恒 0）为黑；集合外沿迭代数进调色板（非黑、有界）。
        let m = MandelbrotBackground { scale: 0.02, center: (0.0, 0.0), max_iter: 64 };
        let win = Fbuf16F::new(33, 33);
        let out = m.eval_window(&win);

        let center = out.get(16, 16);
        assert_eq!(center, [0.0, 0.0, 0.0, 1.0], "c=0 在集合内，必须为黑");

        // 宽窗口右端：c.re = (128−64)·0.02 = 1.28，显然逃逸进调色板。
        let win2 = Fbuf16F::new(129, 5);
        let out2 = m.eval_window(&win2);
        let far = out2.get(128, 2); // c.re = (128−64)·0.02 = 1.28，快速逃逸
        assert!(far[0] > 0.0 || far[1] > 0.0 || far[2] > 0.0, "逃逸点必须进调色板");
        for ch in far[..3].iter() {
            assert!((0.0..=1.0).contains(ch), "调色板输出有界：{ch}");
        }
        assert_eq!(far[3], 1.0);
    }

    #[test]
    fn deterministic_and_parameter_sensitive() {
        let a = MandelbrotBackground::default().eval_window(&Fbuf16F::new(24, 24));
        let b = MandelbrotBackground::default().eval_window(&Fbuf16F::new(24, 24));
        assert_eq!(a, b, "纯函数契约（同参逐位一致）");
        let c = MandelbrotBackground { center: (0.1, 0.0), ..MandelbrotBackground::default() }
            .eval_window(&Fbuf16F::new(24, 24));
        assert_ne!(a.pixels, c.pixels, "center 必须改变图案");
    }
}
