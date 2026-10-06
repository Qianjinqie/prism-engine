//! sharpen（B4 清单：sharpen——非锐化掩模）。
//!
//! 语义权威：《强大版路线图》B4 行（sharpen）；宪章 C4（行业概念：AE/PS
//! Unsharp Mask 同名概念）。S2 卷积（模糊核）与线性组合的复合特解。
//!
//! # 实现
//!
//! `out = in + amount · (in − blur(in))`，blur 为 radius 可分离 box blur
//! （复用 `blur::box_blur_buf`）。常量场是零差不动点（blur 常量场 = 常量，
//! 差为零 ⇒ 输出逐位不变——关卡 2 "全中值恒等图不变"）。
//!
//! halo = `radius`（唯一邻域读来自 blur 核）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::blur::box_blur_buf;

/// sharpen 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sharpen {
    /// 锐化强度（0 = 恒等；典型 0.3–1.0，过冲不钳制——HDR 线性域）。
    pub amount: f32,
    /// 非锐化掩模的模糊半径（像素）。
    pub radius: u32,
}

impl Default for Sharpen {
    fn default() -> Self {
        Sharpen { amount: 0.5, radius: 1 }
    }
}

impl TileEffect for Sharpen {
    fn halo_radius(&self) -> u32 {
        self.radius
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let blurred = box_blur_buf(window, self.radius);
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let b = blurred.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.amount * (c[0] - b[0]),
                        c[1] + self.amount * (c[1] - b[1]),
                        c[2] + self.amount * (c[2] - b[2]),
                        c[3] + self.amount * (c[3] - b[3]),
                    ],
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
    fn checkpoint2_uniform_midfield_is_bit_exact_unchanged() {
        // 关卡 2：全中值（0.5）恒等图不变——常量场零差不动点，位精确。
        let mut win = Fbuf16F::new(10, 8);
        for y in 0..8u32 {
            for x in 0..10u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let sh = Sharpen { amount: 1.5, radius: 2 };
        let out = sh.eval_window(&win);
        assert_eq!(out, win, "全中值恒等图必须逐位不变");
        assert_eq!(sh.halo_radius(), 2);
    }

    #[test]
    fn amount_zero_is_bit_exact_identity() {
        // amount 0：差被乘零吸收 → in + 0 逐位恒等（正输入域）。
        let mut win = Fbuf16F::new(8, 6);
        let mut state: u64 = 0x54A2;
        for y in 0..6u32 {
            for x in 0..8u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let out = Sharpen { amount: 0.0, radius: 2 }.eval_window(&win);
        assert_eq!(out, win, "amount 0 必须逐位恒等");
    }

    #[test]
    fn step_edge_gets_local_contrast_overshoot() {
        // 陡边：亮侧过冲（> 原值）、暗侧下冲（< 原值）——锐化的行为定义。
        let mut win = Fbuf16F::new(9, 5);
        for y in 0..5u32 {
            for x in 0..9u32 {
                let v = if x < 4 { 0.25 } else { 0.75 };
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let out = Sharpen { amount: 1.0, radius: 1 }.eval_window(&win);
        let bright_side = out.get(4, 2)[0]; // 亮侧边缘（blur 拉低 → 差为正）
        let dark_side = out.get(3, 2)[0]; // 暗侧边缘
        assert!(bright_side > 0.75, "亮侧必须过冲：{bright_side} > 0.75");
        assert!(dark_side < 0.25, "暗侧必须下冲：{dark_side} < 0.25");
        // 远离边缘处回到原值（blur = 原值的区域差为零）。
        assert_eq!(out.get(0, 2)[0], 0.25, "均匀区差为零");
        assert_eq!(out.get(8, 2)[0], 0.75, "均匀区差为零");
    }

    #[test]
    fn sharpen_is_pure() {
        let mut win = Fbuf16F::new(7, 5);
        let mut state: u64 = 0x55E;
        for y in 0..5u32 {
            for x in 0..7u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let sh = Sharpen::default();
        assert_eq!(sh.eval_window(&win), sh.eval_window(&win), "纯函数契约");
    }
}
