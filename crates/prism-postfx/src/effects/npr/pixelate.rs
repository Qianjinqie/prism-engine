//! 像素化（B14 NPR 族：马赛克 / 块均化）。
//!
//! 语义权威：《强大版路线图》B14 行（像素化）与 §3.5（像素级块操作）。
//!
//! # 实现
//!
//! 将画面划分为 `block × block` 的块，每块内取块均值赋给块内所有像素
//! （马赛克）。块锚在窗口局部坐标（位置键控——整幅求值为规范语义，与 B4
//! 位置键控效果同约定）。
//!
//! halo = 0：块均值只读块内像素（不跨块读）——但块划分与窗口原点耦合，
//! 故为位置键控而非平移不变。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// 像素化效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixelate {
    /// 块边长（像素；1 = 恒等）。
    pub block: u32,
}

impl Default for Pixelate {
    fn default() -> Self {
        Pixelate { block: 8 }
    }
}

impl TileEffect for Pixelate {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        let b = self.block.max(1);
        // 块锚在窗口局部坐标（位置键控）。
        let mut by = 0u32;
        while by < window.height {
            let mut bx = 0u32;
            while bx < window.width {
                let x1 = (bx + b).min(window.width);
                let y1 = (by + b).min(window.height);
                let mut acc = [0.0f64; 4];
                let mut n = 0.0f64;
                for y in by..y1 {
                    for x in bx..x1 {
                        let c = window.get(x, y);
                        for k in 0..4 {
                            acc[k] += c[k] as f64;
                        }
                        n += 1.0;
                    }
                }
                let mean = [
                    (acc[0] / n) as f32,
                    (acc[1] / n) as f32,
                    (acc[2] / n) as f32,
                    (acc[3] / n) as f32,
                ];
                for y in by..y1 {
                    for x in bx..x1 {
                        out.set(x, y, mean);
                    }
                }
                bx += b;
            }
            by += b;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_one_is_bit_exact_identity() {
        let mut win = Fbuf16F::new(9, 7);
        for y in 0..7u32 {
            for x in 0..9u32 {
                let v = (x * 13 + y * 7) as f32 / 100.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let out = Pixelate { block: 1 }.eval_window(&win);
        assert_eq!(out, win, "块大小 1 必须逐位恒等");
    }

    #[test]
    fn block_is_constant_inside() {
        let mut win = Fbuf16F::new(16, 16);
        let mut state: u64 = 0x9E37;
        for y in 0..16u32 {
            for x in 0..16u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v, 1.0 - v, 1.0]);
            }
        }
        let out = Pixelate { block: 4 }.eval_window(&win);
        // 每个 4×4 块内像素相同。
        for by in 0..4u32 {
            for bx in 0..4u32 {
                let v0 = out.get(bx * 4, by * 4);
                for y in by * 4..by * 4 + 4 {
                    for x in bx * 4..bx * 4 + 4 {
                        assert_eq!(out.get(x, y), v0, "块内必须常量 @ ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn pixelate_is_deterministic() {
        let mut win = Fbuf16F::new(20, 20);
        for y in 0..20u32 {
            for x in 0..20u32 {
                win.set(x, y, [(x as f32) / 20.0, (y as f32) / 20.0, 0.5, 1.0]);
            }
        }
        let p = Pixelate { block: 5 };
        assert_eq!(p.eval_window(&win), p.eval_window(&win), "纯函数契约");
    }
}
