//! glitch（清单：glitch —— 条带位移故障）。
//!
//! 语义权威：glitch；（glitch 的
//! 噪声派生纪律：一切位移量由确定性哈希派生，无私开 RNG、无全局状态）。
//!
//! # 实现
//!
//! 水平条带位移：把图按 `block` 像素高切成行带（带索引锚窗口局部 y），
//! 每带掷两粒确定性骰子——门控 `hash3(seed, band, frame, 1) < intensity`
//! 决定该带是否故障，位移 `dxi = round((hash3(seed, band, frame, 2) − ½)
//! · 2 · max_offset)` 为整像素水平位移；带内整行按 `x + dxi`（边缘钳制）
//! 原样取位（整像素位移 = 位模式搬运，无重采样）。
//!
//! 帧号进哈希 ⇒ 故障图案随帧跳动；同帧同参 ⇒ 逐位复现。位置键控（带网格
//! 锚窗口局部坐标）；halo = `max_offset` 上取整（位移读带的真实宽度）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// glitch 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glitch {
    /// 故障带触发概率门（0 = 无故障，1 = 全部带位移）。
    pub intensity: f32,
    /// 最大水平位移（像素；带位移 ∈ [−max_offset, +max_offset] 取整）。
    pub max_offset: f32,
    /// 条带高度（像素）。
    pub block: u32,
    /// 种子。
    pub seed: u64,
    /// 帧号（故障图案随帧去相关）。
    pub frame: u64,
}

impl Default for Glitch {
    fn default() -> Self {
        Glitch { intensity: 0.3, max_offset: 4.0, block: 8, seed: 0, frame: 0 }
    }
}

impl TileEffect for Glitch {
    fn halo_radius(&self) -> u32 {
        self.max_offset.ceil() as u32
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            let band = (y / self.block.max(1)) as u64;
            let gate = prism_noise::pcg::hash3_f32(self.seed, band, self.frame, 1);
            let dxi: i64 = if gate < self.intensity {
                let dx =
                    (prism_noise::pcg::hash3_f32(self.seed, band, self.frame, 2) - 0.5)
                        * 2.0
                        * self.max_offset;
                dx.round() as i64
            } else {
                0
            };
            for x in 0..window.width {
                let sx = (x as i64 + dxi).clamp(0, window.width as i64 - 1) as u32;
                out.set(x, y, window.get(sx, y));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_zero_is_bit_exact_identity() {
        // 门控 hash < 0 永假 → 全部带原样 → 逐位恒等。
        let mut win = Fbuf16F::new(12, 16);
        for y in 0..16u32 {
            for x in 0..12u32 {
                let v = (x * 5 + y * 3) as f32 / 64.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let g = Glitch { intensity: 0.0, max_offset: 4.0, block: 4, seed: 7, frame: 0 };
        assert_eq!(g.eval_window(&win), win, "intensity 0 必须逐位恒等");
        assert_eq!(g.halo_radius(), 4, "halo = max_offset 上取整");
    }

    #[test]
    fn full_intensity_displacement_matches_hash_derivation() {
        // intensity 1 → 全带位移；期望位移可由同一公开哈希精确重算——
        // 输出行 = 输入行平移 dxi（钳制），逐位原样。
        let mut win = Fbuf16F::new(20, 16);
        for y in 0..16u32 {
            for x in 0..20u32 {
                let v = (x * 31 + y * 7) as f32 / 128.0;
                win.set(x, y, [v, v * 0.25, v * 0.5, 1.0]);
            }
        }
        let g = Glitch { intensity: 1.0, max_offset: 5.0, block: 4, seed: 11, frame: 2 };
        let out = g.eval_window(&win);
        for y in 0..16u32 {
            let band = (y / 4) as u64;
            let dx = (prism_noise::pcg::hash3_f32(11, band, 2, 2) - 0.5) * 2.0 * 5.0;
            let dxi = dx.round() as i64;
            for x in 0..20u32 {
                let sx = (x as i64 + dxi).clamp(0, 19) as u32;
                assert_eq!(out.get(x, y), win.get(sx, y), "带 {band} 行 {y} 位移 {dxi} 不符");
            }
        }
    }

    #[test]
    fn glitch_is_pure_and_frame_decorrelated() {
        let mut win = Fbuf16F::new(16, 24);
        let mut state: u64 = 0x6C1C7;
        for y in 0..24u32 {
            for x in 0..16u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let g = Glitch { intensity: 0.5, max_offset: 3.0, block: 4, seed: 3, frame: 1 };
        assert_eq!(g.eval_window(&win), g.eval_window(&win), "纯函数契约");
        let other = Glitch { intensity: 0.5, max_offset: 3.0, block: 4, seed: 3, frame: 2 };
        assert_ne!(g.eval_window(&win), other.eval_window(&win), "帧间故障图案必须去相关");
    }
}
