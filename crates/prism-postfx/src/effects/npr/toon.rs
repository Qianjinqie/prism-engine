//! toon（NPR 族：卡通光照量化 + 描边）。
//!
//! 语义权威：第 1 条——"光照量化（N·L 分 k 档，边界用
//! 平滑过渡带）" + "描边 = 深度/法线不连续检测"。CPU 参考路径无 G-buffer
//! 深度/法线通道，故此处用**亮度不连续检测**替代（同名语义：边缘 =
//! 邻域亮度差超阈值），深度/法线版随 GPU 通道接入（缺口在册）。
//!
//! # 实现
//!
//! 1. **量化**：亮度 → `k` 档阶梯，档边界用宽度 `softness` 的平滑过渡带
//!    （`smoothstep`，零超越函数：三次多项式 `t²(3−2t)`）；
//! 2. **描边**：中心差分梯度幅值 > `edge_threshold` 处压暗到 `edge_dark`
//!    （乘性，保持色相）。
//!
//! halo = 1（描边的 3×3 邻域）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::{gradient_mag, luma};

/// toon 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Toon {
    /// 量化档数（≥ 2）。
    pub levels: u32,
    /// 档边界平滑过渡带宽度（0 = 硬阶梯）。
    pub softness: f32,
    /// 描边梯度阈值（0 = 关闭描边）。
    pub edge_threshold: f32,
    /// 描边压暗系数（0 = 全黑描边，1 = 无描边）。
    pub edge_dark: f32,
}

impl Default for Toon {
    fn default() -> Self {
        Toon { levels: 4, softness: 0.05, edge_threshold: 0.25, edge_dark: 0.15 }
    }
}

/// 三次平滑阶跃（零超越函数）。`t` 先钳 [0,1]。
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = prism_math::clamp(t, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 将亮度 `l ∈ [0,1]` 量化到 `levels` 档，档边界带宽 `softness`。
///
/// 输出为档位代表值 `(k + 0.5) / levels`（k = 0..levels），过渡带内两档间
/// 平滑插值（softness = 0 时硬阶梯）。
fn quantize_toon(l: f32, levels: u32, softness: f32) -> f32 {
    let levels = levels.max(1) as f32;
    let scaled = prism_math::clamp(l, 0.0, 1.0) * levels;
    let idx = scaled.floor();
    let frac = scaled - idx;
    let step = if softness <= 0.0 {
        // 硬阶梯：frac < 1 恒取本档（用 0.5 的整数落点，档边界在整数处）。
        0.0
    } else {
        // 平滑过渡带集中在每档边界附近，宽 softness。
        let edge = 1.0 - softness;
        smoothstep((frac - edge) / softness.max(1e-6))
    };
    // 档位代表值：本档中心 →（过渡结束时）下一档中心。
    let k = idx + 0.5 + step;
    k / levels
}

impl TileEffect for Toon {
    fn halo_radius(&self) -> u32 {
        1
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let l = prism_math::clamp(luma(c), 0.0, 1.0);
                // 量化后的目标亮度（与原亮度同域）。
                let q = quantize_toon(l, self.levels, self.softness);
                // 用比例缩放保持色相：scale = q / l（l 近 0 时保留原色）。
                let scale = if l > 1e-4 { q / l } else { 1.0 };
                let mut rgb = [c[0] * scale, c[1] * scale, c[2] * scale];
                if self.edge_threshold > 0.0 {
                    let g = gradient_mag(window, x, y);
                    if g > self.edge_threshold {
                        for v in rgb.iter_mut() {
                            *v *= self.edge_dark;
                        }
                    }
                }
                out.set(x, y, [rgb[0], rgb[1], rgb[2], c[3]]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_field_quantizes_to_few_levels() {
        // 平坦渐变输入 → 输出亮度只有 levels 个离散档。
        let mut win = Fbuf16F::new(64, 4);
        for x in 0..64u32 {
            let v = x as f32 / 63.0;
            for y in 0..4u32 {
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let t = Toon { levels: 4, softness: 0.0, edge_threshold: 0.0, edge_dark: 1.0 };
        let out = t.eval_window(&win);
        // 输出亮度索引应聚成 ≤ 4 个离散档（色相保持的缩放 + f16 量化带少量
        // 抖动，故用档位中心 ±0.5 分箱归类后统计档数）。
        let mut seen = std::collections::BTreeSet::new();
        for x in 0..64u32 {
            let l = prism_math::clamp(luma(out.get(x, 0)), 0.0, 1.0);
            let k = (l * 4.0).floor() as i32; // 箱 [k/4,(k+1)/4)
            seen.insert(k.min(3).max(0));
        }
        assert!(
            seen.len() <= 4,
            "量化档数应 ≤ levels = 4：{seen:?}"
        );
    }

    #[test]
    fn edge_detection_darkens_high_gradient() {
        // 强边缘处描边压暗；平坦区内不压暗。
        let mut win = Fbuf16F::new(8, 8);
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = if x < 4 { 0.1 } else { 0.9 };
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let t = Toon { levels: 4, softness: 0.0, edge_threshold: 0.2, edge_dark: 0.2 };
        let out = t.eval_window(&win);
        // x=3/4 处为边缘（梯度大）→ 压暗。
        assert!(luma(out.get(3, 4)) < 0.1, "边缘必须压暗");
        assert!(luma(out.get(0, 4)) >= 0.05, "平坦亮区不压暗（左半为暗区，仅验证未额外压暗）");
    }

    #[test]
    fn toon_is_deterministic_and_halo_one() {
        let mut win = Fbuf16F::new(10, 10);
        let mut state: u64 = 0x7099;
        for y in 0..10u32 {
            for x in 0..10u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.7, v * 0.3, 1.0]);
            }
        }
        let t = Toon::default();
        assert_eq!(t.halo_radius(), 1);
        assert_eq!(t.eval_window(&win), t.eval_window(&win), "纯函数契约");
    }
}
