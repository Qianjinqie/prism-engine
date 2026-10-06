//! bloom（B4 清单：bloom——镜头光学泛光）。
//!
//! 语义权威：《强大版路线图》B4 行（bloom）与 §3.5（"bloom/blur 在线性域"
//! ——合成用线性域加算）；《算法白皮书》§15（bloom 是镜头层光学泛光，与
//! 胶片层 halation 分立——语义核心 = 亮部提取 + 扩散 + 线性加算）。
//!
//! # 管线（单 `TileEffect::eval_window` 内三步）
//!
//! 1. **亮部提取**：逐通道 `max(0, c − threshold)`（HDR 线性域阈值）；
//! 2. **扩散**：全分辨率可分离 box blur（复用 `blur::box_blur_buf` 共享
//!    原语，两趟固定序、窗口边缘钳制复制）；
//! 3. **合成**：`out = in + intensity · bloom`（线性域加算，能量注入有界）。
//!
//! # 设计决策：为什么不是 ×2 降采样金字塔
//!
//! 金字塔实现（降采样×2 → 半分辨率模糊 → 上采样）与 tile 协议存在**相位
//! 冲突**：半分辨率网格的奇偶相位取决于窗口原点 `tile.x − halo` 的奇偶——
//! 窗口原点为奇数时窗口内的半分辨率块与整幅错位半格，核心区输出随之
//! 偏移（关卡 1 tile 一致性实测检出：bloom 单效果 tile ≠ 整幅）。半分辨率
//! 金字塔**不是平移不变算子**，halo 申报再足额也无法修复。改用全分辨率
//! 可分离 box blur：严格平移不变 ⇒ tile 化与整幅逐字节一致，泛光语义
//! 不变（扩散范围由 `radius` 直接控制）。
//!
//! # halo
//! 输出 p 只读输入 `p ± radius`（box blur 支撑半径）⇒ 申报 `radius`，
//! 足额 ⇒ 关卡 1 tile 一致性成立（本 crate 集成测试复验）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::blur::box_blur_buf;

/// bloom 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bloom {
    /// HDR 线性域亮部阈值（逐通道减除）。
    pub threshold: f32,
    /// 扩散半径（全分辨率 box blur 半径，像素）。
    pub radius: u32,
    /// 泛光注入强度（线性域加算系数）。
    pub intensity: f32,
}

impl Default for Bloom {
    fn default() -> Self {
        Bloom { threshold: 1.0, radius: 4, intensity: 0.25 }
    }
}

impl TileEffect for Bloom {
    fn halo_radius(&self) -> u32 {
        self.radius
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        // 步 1：亮部提取（逐像素，零邻域）。
        let mut bright = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                bright.set(
                    x,
                    y,
                    [
                        (c[0] - self.threshold).max(0.0),
                        (c[1] - self.threshold).max(0.0),
                        (c[2] - self.threshold).max(0.0),
                        (c[3] - self.threshold).max(0.0),
                    ],
                );
            }
        }

        // 步 2：可分离 box blur 扩散（平移不变 ⇒ tile 一致性）。
        let blurred = box_blur_buf(&bright, self.radius);

        // 步 3：线性域加算合成。
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let b = blurred.get(x, y);
                let c = window.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.intensity * b[0],
                        c[1] + self.intensity * b[1],
                        c[2] + self.intensity * b[2],
                        c[3] + self.intensity * b[3],
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
    fn below_threshold_is_bit_exact_identity() {
        // 全输入 ≤ 阈值 → 亮部全零 → bloom 项为零 → out = in + 0 逐位恒等。
        let mut win = Fbuf16F::new(9, 7);
        for y in 0..7u32 {
            for x in 0..9u32 {
                let v = (x * 13 + y * 5) as f32 / 400.0; // ≤ 0.335 < 阈值 1.0
                win.set(x, y, [v, v * 0.5, 0.25, 1.0]);
            }
        }
        let bloom = Bloom { threshold: 1.0, radius: 4, intensity: 0.5 };
        let out = bloom.eval_window(&win);
        assert_eq!(out, win, "低于阈值的输入必须逐位不变");
        assert_eq!(bloom.halo_radius(), 4);
    }

    #[test]
    fn bright_impulse_bleeds_to_neighbors_symmetrically() {
        // 单个超阈像素 → 泛光对称扩散；box blur 冲激响应是平台（半径内
        // 等值），衰减发生在支撑边界外。
        let mut win = Fbuf16F::new(13, 13);
        win.set(6, 6, [4.0, 4.0, 4.0, 1.0]);
        let bloom = Bloom { threshold: 1.0, radius: 2, intensity: 1.0 };
        let out = bloom.eval_window(&win);

        let center_gain = out.get(6, 6)[0] - 4.0;
        let near_gain = out.get(8, 6)[0]; // 距中心 2 ≤ radius 2：平台内
        let far_gain = out.get(12, 6)[0]; // 距中心 6 > 支撑 [−2,+2]：平台外
        assert!(
            center_gain >= near_gain,
            "平台内等值（box blur 冲激响应）：{center_gain} vs {near_gain}"
        );
        assert!(
            near_gain > far_gain,
            "泛光随距离衰减（blur 支撑边界外）：{near_gain} vs {far_gain}"
        );
        assert_eq!(out.get(8, 6)[0], out.get(4, 6)[0], "左右对称");
        assert_eq!(out.get(6, 8)[0], out.get(8, 6)[0], "各向同性（box blur 对称）");
        assert!(center_gain > 0.0, "亮部像素本身获得泛光增益");
        assert!(far_gain == 0.0, "支撑边界外必须无泄漏（{far_gain}）");
    }

    #[test]
    fn intensity_zero_is_identity_and_alpha_channel_blooms() {
        // intensity 0：加算项归零 → 恒等；A 通道同参参与管线（与 blur 一致）。
        let mut win = Fbuf16F::new(8, 8);
        for y in 0..8u32 {
            for x in 0..8u32 {
                win.set(x, y, [0.0, 0.0, 0.0, if (x + y) % 2 == 0 { 2.0 } else { 0.0 }]);
            }
        }
        let bloom = Bloom { threshold: 1.0, radius: 4, intensity: 0.0 };
        let out = bloom.eval_window(&win);
        assert_eq!(out, win, "intensity 0 必须恒等");
        let bloom_a = Bloom { threshold: 1.0, radius: 1, intensity: 1.0 };
        let out_a = bloom_a.eval_window(&win);
        assert!(
            out_a.get(4, 4)[3] > 0.0,
            "A 通道泛光应注入（全通道同参管线）"
        );
    }

    #[test]
    fn bloom_is_pure() {
        let mut win = Fbuf16F::new(11, 9);
        let mut state: u64 = 0xB1003;
        for y in 0..9u32 {
            for x in 0..11u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v * 2.0, v, v * 0.25, 1.0]);
            }
        }
        let bloom = Bloom { threshold: 0.5, radius: 2, intensity: 0.3 };
        assert_eq!(bloom.eval_window(&win), bloom.eval_window(&win), "纯函数契约");
    }
}
