//! dissolve（B4 清单：dissolve——溶解转场）。
//!
//! 语义权威：《强大版路线图》B4 行（dissolve）；宪章 C4 对照 #4（行业概念：
//! 剪映 / iMovie Dissolve，Premiere"溶解"= 随机像素显影的同名转场）。
//!
//! # 实现
//!
//! 噪声阈值显影：`out = in  if hash2(seed, x, y) < t  else  0`。哈希 ∈
//! [0, 1) ⇒ **端点精确**：`t = 0` 时无像素满足（全黑）、`t = 1` 时全部满足
//! （输出与输入逐位一致）。像素级显影（非逐帧交叉淡化——双输入交叉淡化属
//! 合成域 Merge 算子，架构 §3.5；本效果是单输入的溶解出画原语）。
//!
//! 位置键控（窗口局部坐标）；halo = 0。`t` 走属性通道（A2）由时间驱动。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// dissolve 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dissolve {
    /// 显影阈值 t ∈ [0, 1]（0 = 全黑，1 = 完整显示）。
    pub t: f32,
    /// 显影种子（决定哪些像素先显影的确定性图案）。
    pub seed: u64,
}

impl Default for Dissolve {
    fn default() -> Self {
        Dissolve { t: 0.0, seed: 0 }
    }
}

impl TileEffect for Dissolve {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let h = prism_noise::pcg::hash2_f32(self.seed, x as u64, y as u64);
                if h < self.t {
                    out.set(x, y, window.get(x, y));
                } // else 保持全零（Fbuf16F::new 初始化）
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 24×16 全非零输入。
    fn populated() -> Fbuf16F {
        let mut win = Fbuf16F::new(24, 16);
        for y in 0..16u32 {
            for x in 0..24u32 {
                let v = (x * 11 + y * 7) as f32 / 300.0 + 0.1;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 0.75]);
            }
        }
        win
    }

    #[test]
    fn checkpoint2_endpoints_t0_black_and_t1_bit_exact() {
        // 关卡 2：dissolve t=0 / t=1 端点精确。
        let win = populated();
        let t0 = Dissolve { t: 0.0, seed: 5 }.eval_window(&win);
        assert!(
            t0.pixels.iter().all(|&p| p == 0),
            "t=0 必须精确全黑（全零位模式）"
        );
        let t1 = Dissolve { t: 1.0, seed: 5 }.eval_window(&win);
        assert_eq!(t1, win, "t=1 必须与输入逐位一致");
    }

    #[test]
    fn coverage_is_monotone_in_t() {
        // 显影覆盖随 t 单调不减（同为像素计数）。
        let win = populated();
        let coverage = |t: f32| {
            Dissolve { t, seed: 9 }
                .eval_window(&win)
                .pixels
                .chunks_exact(4)
                .filter(|px| px[0] != 0)
                .count()
        };
        let (c25, c50, c75) = (coverage(0.25), coverage(0.5), coverage(0.75));
        assert!(c25 <= c50 && c50 <= c75, "覆盖必须单调：{c25} ≤ {c50} ≤ {c75}");
        assert!(c25 > 0 && c75 < 24 * 16, "中间阈值必须部分显影：{c25} / {c75}");
        // 哈希均匀性：t=0.5 显影约半数（±15%）。
        let half = c50 as f64 / (24.0 * 16.0);
        assert!((0.35..=0.65).contains(&half), "t=0.5 覆盖率应近半：{half}");
    }

    #[test]
    fn dissolve_is_pure_and_seed_dependent() {
        let win = populated();
        let a = Dissolve { t: 0.5, seed: 3 }.eval_window(&win);
        assert_eq!(
            Dissolve { t: 0.5, seed: 3 }.eval_window(&win),
            a,
            "纯函数契约"
        );
        let b = Dissolve { t: 0.5, seed: 4 }.eval_window(&win);
        assert_ne!(a.pixels, b.pixels, "种子必须改变显影图案");
        // 显影像素原样保留位模式（非重算）。
        for y in 0..16u32 {
            for x in 0..24u32 {
                let av = a.get(x, y);
                if av[0] != 0.0 {
                    assert_eq!(av, win.get(x, y), "显影像素必须逐位原样");
                }
            }
        }
    }
}
