//! grain（B4 清单：grain——胶片颗粒）。
//!
//! 语义权威：《强大版路线图》B4 行（grain）与 §3.5（"grain 在感知域"——本
//! 效果为域无关加性颗粒，感知域权重随 B14 科学化颗粒细化）；《算法白皮书》
//! §15/§17（噪声派生纪律：随机性 = 确定性哈希，无私开 RNG）；架构 v2.4
//! §4.4（SampleIndex 像素域：逐像素派生必不同流，防网格伪影）。
//!
//! # 实现
//!
//! 逐像素加性颗粒：`out = in + amount · n(x, y, frame)`，其中
//! `n = hash3(seed, x, y, frame)·2 − 1 ∈ [−1, 1)`（PCG 整数 hash → f32，
//! prism-noise 纯函数）。帧号进哈希 ⇒ 帧间去相关（颗粒"活"），同帧同参数
//! ⇒ 逐位可复现、任意 seek 可重算。RGB 三通道同噪（亮度颗粒），A 通道不动。
//!
//! 位置键控（窗口局部坐标）；halo = 0。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// grain 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grain {
    /// 颗粒幅度（加性系数；0 = 无效果）。
    pub amount: f32,
    /// 颗粒种子（与节点 id / 全局种子汇流的调用方约定位）。
    pub seed: u64,
    /// 帧号（格锚：帧间去相关；同帧逐位复现）。
    pub frame: u64,
}

impl Default for Grain {
    fn default() -> Self {
        Grain { amount: 0.05, seed: 0, frame: 0 }
    }
}

impl TileEffect for Grain {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let n = prism_noise::pcg::hash3_f32(self.seed, x as u64, y as u64, self.frame)
                    * 2.0
                    - 1.0;
                let c = window.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.amount * n,
                        c[1] + self.amount * n,
                        c[2] + self.amount * n,
                        c[3],
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
    fn deterministic_per_pixel_and_seek_safe() {
        // 同 (seed, frame) 同输出；帧号不同 → 噪声不同（去相关）；同帧重算
        // 逐位一致（任意 seek = 纯函数重放）。
        let mut win = Fbuf16F::new(8, 6);
        for y in 0..6u32 {
            for x in 0..8u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let a = Grain { amount: 0.1, seed: 7, frame: 3 }.eval_window(&win);
        let b = Grain { amount: 0.1, seed: 7, frame: 3 }.eval_window(&win);
        assert_eq!(a, b, "同参数必须逐位一致");

        let c = Grain { amount: 0.1, seed: 7, frame: 4 }.eval_window(&win);
        assert_ne!(a.pixels, c.pixels, "帧号必须去相关（颗粒随帧变化）");

        let d = Grain { amount: 0.1, seed: 8, frame: 3 }.eval_window(&win);
        assert_ne!(a.pixels, d.pixels, "种子必须去相关");
    }

    #[test]
    fn amplitude_bounded_and_pixel_decorrelated() {
        // |out − in| < amount（噪声 ∈ [−1, 1) 开区间）；相邻像素噪声不同流
        // （架构 §4.4 像素域——防网格伪影的抽样检查）。
        let mut win = Fbuf16F::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let amount = 0.1f32;
        let out = Grain { amount, seed: 1, frame: 0 }.eval_window(&win);
        let mut distinct = 0usize;
        for y in 0..16u32 {
            for x in 0..16u32 {
                let delta = out.get(x, y)[0] - 0.5;
                assert!(delta.abs() < amount, "颗粒幅度越界：{delta}");
                if x + 1 < 16 && out.get(x, y)[0] != out.get(x + 1, y)[0] {
                    distinct += 1;
                }
            }
        }
        assert!(distinct > 16 * 12, "相邻像素颗粒必须大部分不同流（数得 {distinct}）");
    }

    #[test]
    fn grain_is_pure_and_alpha_untouched() {
        let mut win = Fbuf16F::new(6, 4);
        let mut state: u64 = 0x63A1;
        for y in 0..4u32 {
            for x in 0..6u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 0.25]);
            }
        }
        let g = Grain::default();
        assert_eq!(g.eval_window(&win), g.eval_window(&win), "纯函数契约");
        let out = g.eval_window(&win);
        for y in 0..4u32 {
            for x in 0..6u32 {
                assert_eq!(out.get(x, y)[3], win.get(x, y)[3], "alpha 必须原样");
            }
        }
    }
}
