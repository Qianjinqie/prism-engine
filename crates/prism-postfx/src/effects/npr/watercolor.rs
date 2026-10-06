//! 水彩（B14 NPR 族：水彩画仿真）。
//!
//! 语义权威：《算法白皮书》§14 第 4 条——"亮度分域平坦化 + 边缘沉积暗化
//! （梯度幅值加权）+ 域扭曲（低频噪声轻推采样坐标——与 A3 噪声族咬合）+
//! 纸纹乘算（B14 纹理族）"。
//!
//! # 实现（四步，固定序）
//!
//! 1. **域扭曲**：用 fbm simplex 低频场轻推采样坐标（`± warp_amp`），
//!    采样钳制窗口边缘；
//! 2. **分域平坦化**：对扭曲采样后的颜色按亮度分 `bands` 档（smoothstep
//!    过渡），档内取代表色 → 大面积平涂感；
//! 3. **边缘沉积**：原图梯度幅值加权暗化（色素在边缘沉积）；
//! 4. **纸纹乘算**：低频 value 噪声（`paper` 参数）乘算亮度。
//!
//! halo = radius（扭曲 + 梯度邻域）；随机性经 prism-noise PCG（seed 决定）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::{gradient_mag, luma};

/// 水彩效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Watercolor {
    /// 域扭曲幅度（采样坐标像素偏移上限）。
    pub warp_amp: f32,
    /// 域扭曲频率。
    pub warp_freq: f32,
    /// 亮度分域档数。
    pub bands: u32,
    /// 边缘沉积强度（梯度加权暗化系数）。
    pub edge_deposit: f32,
    /// 纸纹强度（0 = 无纸纹）。
    pub paper: f32,
    /// 种子（域扭曲 + 纸纹场）。
    pub seed: u64,
    /// 采样邻域半径（gradient 用）。
    pub radius: u32,
}

impl Default for Watercolor {
    fn default() -> Self {
        Watercolor {
            warp_amp: 1.5,
            warp_freq: 0.03,
            bands: 5,
            edge_deposit: 0.4,
            paper: 0.12,
            seed: 0,
            radius: 1,
        }
    }
}

/// 三次平滑阶跃（零超越函数）。
fn smoothstep(t: f32) -> f32 {
    let t = prism_math::clamp(t, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl TileEffect for Watercolor {
    fn halo_radius(&self) -> u32 {
        self.radius
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let w = window.width;
        let h = window.height;
        let fbm = prism_noise::fbm::Fbm::new(4, 2.0, 0.5);
        let bands = self.bands.max(1) as f32;
        let mut out = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // 步 1：域扭曲（低频 fbm 轻推采样坐标）。
                let wx = fbm.simplex2(self.seed, x as f32 * self.warp_freq, y as f32 * self.warp_freq);
                let wy = fbm.simplex2(
                    self.seed ^ 0x9E37_79B9,
                    x as f32 * self.warp_freq + 31.0,
                    y as f32 * self.warp_freq + 17.0,
                );
                let sx = prism_math::clamp(
                    x as f32 + wx * self.warp_amp,
                    0.0,
                    (w - 1) as f32,
                );
                let sy = prism_math::clamp(
                    y as f32 + wy * self.warp_amp,
                    0.0,
                    (h - 1) as f32,
                );
                let sample = bilinear(window, sx, sy);

                // 步 2：亮度分域平坦化（档内取代表色）。
                let l = prism_math::clamp(luma(sample), 0.0, 1.0);
                let scaled = l * bands;
                let idx = scaled.floor();
                let frac = scaled - idx;
                let edge = 0.5 - 0.15;
                let t = smoothstep((frac - edge) / 0.3);
                let rep = ((idx + 0.5) / bands) * (1.0 - t) + ((idx + 1.5) / bands) * t;
                let rep = prism_math::clamp(rep, 0.0, 1.0);
                let scale = if l > 1e-4 { rep / l } else { 1.0 };
                let mut rgb = [sample[0] * scale, sample[1] * scale, sample[2] * scale];

                // 步 3：边缘沉积暗化（梯度加权）。
                let g = gradient_mag(window, x, y);
                let deposit = prism_math::clamp(self.edge_deposit * g, 0.0, 1.0);
                for v in rgb.iter_mut() {
                    *v *= 1.0 - deposit;
                }

                // 步 4：纸纹乘算（低频 value 噪声）。
                if self.paper > 0.0 {
                    let grain = prism_noise::value::value2(
                        self.seed ^ 0xA5A5,
                        x as f32 * 0.15,
                        y as f32 * 0.15,
                    ) * 0.5
                        + 0.5; // [0,1]
                    let f = 1.0 - self.paper * (1.0 - grain);
                    for v in rgb.iter_mut() {
                        *v *= f;
                    }
                }

                out.set(
                    x,
                    y,
                    [rgb[0], rgb[1], rgb[2], window.get(x, y)[3]],
                );
            }
        }
        out
    }
}

/// 双线性采样（窗口边缘钳制复制；整数位移分解形式，权重与绝对坐标无关）。
fn bilinear(buf: &Fbuf16F, x: f32, y: f32) -> [f32; 4] {
    let x0f = x.floor();
    let y0f = y.floor();
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = (x0f as i64).clamp(0, buf.width as i64 - 1) as u32;
    let x1 = (x0f as i64 + 1).clamp(0, buf.width as i64 - 1) as u32;
    let y0 = (y0f as i64).clamp(0, buf.height as i64 - 1) as u32;
    let y1 = (y0f as i64 + 1).clamp(0, buf.height as i64 - 1) as u32;
    let c00 = buf.get(x0, y0);
    let c10 = buf.get(x1, y0);
    let c01 = buf.get(x0, y1);
    let c11 = buf.get(x1, y1);
    let (wx0, wx1) = (1.0 - tx, tx);
    let (wy0, wy1) = (1.0 - ty, ty);
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        out[i] =
            c00[i] * wx0 * wy0 + c10[i] * wx1 * wy0 + c01[i] * wx0 * wy1 + c11[i] * wx1 * wy1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watercolor_deterministic_and_seed_sensitive() {
        let mut win = Fbuf16F::new(24, 20);
        let mut state: u64 = 0xA1C0;
        for y in 0..20u32 {
            for x in 0..24u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.8, v * 0.4, 1.0]);
            }
        }
        let wc = Watercolor::default();
        let a = wc.eval_window(&win);
        assert_eq!(a, wc.eval_window(&win), "同 seed 逐位一致");
        let b = Watercolor { seed: 42, ..wc }.eval_window(&win);
        assert_ne!(a.pixels, b.pixels, "seed 必须改变纹理");
    }

    #[test]
    fn flat_field_reduces_to_banded_paper() {
        // 常量输入：分域平坦化后输出近似常量（档内平涂），且未越界。
        let mut win = Fbuf16F::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let wc = Watercolor::default();
        let out = wc.eval_window(&win);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let c = out.get(x, y);
                for k in 0..3 {
                    assert!(c[k] >= 0.0 && c[k] <= 1.0, "输出越界：{c:?}");
                }
            }
        }
    }

    #[test]
    fn halo_and_purity() {
        let win = Fbuf16F::new(10, 10);
        let wc = Watercolor::default();
        assert_eq!(wc.halo_radius(), 1);
        assert_eq!(wc.eval_window(&win), wc.eval_window(&win), "纯函数契约");
    }
}
