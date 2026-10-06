//! plasma（B4 清单：plasma——程序化等离子背景）。
//!
//! 语义权威：《强大版路线图》B4 行（plasma）与 §3.0 域 M（生成艺术输出
//! "可再加工"——本效果即像素域生成器，输出接任意后效）；《算法白皮书》
//! §17（噪声族 = S1 标量场特解；fbm = 核的谱合成）。
//!
//! # 实现
//!
//! fbm simplex 场 → 三段渐变调色板：`u = (fbm(seed, x·scale, y·scale)+1)/2
//! ∈ [0,1]`，`color = palette3(u)`（`effects::mod` 共享原语，纯算术）。
//! 生成器语义：**替换**输入窗口（背景板），A 通道写 1。位置键控（窗口
//! 局部坐标）；halo = 0。
//!
//! 全部随机性来自种子化 fbm（prism-noise 纯函数）——同 seed 同参逐位一致。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::palette3;

/// 调色板三锚点（深紫 → 洋红 → 青；固定常量）。
const STOP_A: [f32; 3] = [0.05, 0.02, 0.15];
const STOP_B: [f32; 3] = [0.90, 0.20, 0.55];
const STOP_C: [f32; 3] = [0.20, 0.85, 0.95];

/// plasma 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plasma {
    /// 场频率（坐标缩放；越大纹越细）。
    pub scale: f32,
    /// 种子。
    pub seed: u64,
    /// fbm 八度数（典型 4–6，白皮书 §17）。
    pub octaves: u32,
}

impl Default for Plasma {
    fn default() -> Self {
        Plasma { scale: 0.02, seed: 0, octaves: 4 }
    }
}

impl TileEffect for Plasma {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let fbm = prism_noise::fbm::Fbm::new(self.octaves, 2.0, 0.5);
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let n = fbm.simplex2(self.seed, x as f32 * self.scale, y as f32 * self.scale);
                let rgb = palette3((n + 1.0) * 0.5, STOP_A, STOP_B, STOP_C);
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
        let a = Plasma { scale: 0.03, seed: 1, octaves: 4 }.eval_window(&Fbuf16F::new(16, 12));
        let b = Plasma { scale: 0.03, seed: 1, octaves: 4 }.eval_window(&Fbuf16F::new(16, 12));
        assert_eq!(a, b, "同 seed 同参必须逐位一致");
        let c = Plasma { scale: 0.03, seed: 2, octaves: 4 }.eval_window(&Fbuf16F::new(16, 12));
        assert_ne!(a.pixels, c.pixels, "seed 必须改变图案");
    }

    #[test]
    fn output_is_bounded_palette_field_not_constant() {
        // 输出落在调色板凸包内（通道 ∈ [min 锚, max 锚]），且非常量场。
        let out = Plasma { scale: 0.05, seed: 3, octaves: 4 }.eval_window(&Fbuf16F::new(24, 24));
        let mut min = [1.0f32; 3];
        let mut max = [0.0f32; 3];
        for y in 0..24u32 {
            for x in 0..24u32 {
                let c = out.get(x, y);
                assert_eq!(c[3], 1.0, "生成器 alpha = 1");
                for i in 0..3 {
                    min[i] = min[i].min(c[i]);
                    max[i] = max[i].max(c[i]);
                }
            }
        }
        for i in 0..3 {
            assert!(min[i] >= STOP_A[i].min(STOP_B[i]).min(STOP_C[i]) - 1e-6);
            assert!(max[i] <= STOP_A[i].max(STOP_B[i]).max(STOP_C[i]) + 1e-6);
            assert!(max[i] > min[i], "通道 {i} 必须非常量场");
        }
    }

    #[test]
    fn plasma_is_pure() {
        let win = Fbuf16F::new(10, 10); // 生成器不读输入——零输入也成立
        let p = Plasma::default();
        assert_eq!(p.eval_window(&win), p.eval_window(&win), "纯函数契约");
    }
}
