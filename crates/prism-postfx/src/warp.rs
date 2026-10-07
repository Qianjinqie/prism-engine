//! UV 重映射 warp：单原语 `uv = f(uv)`，重映射场取输入缓冲通道。
//!
//! 语义权威：UV warp：单原语 uv=f(uv)，重映射
//! 场=Fbuf 通道、双线性确定性采样；halo 协议。
//!
//! # 位移约定
//! 重映射场 = 输入的 R/G 通道（`c = 0.5` 为无位移中心）：
//! `du = (R·2 − 1)·scale`，`dv = (G·2 − 1)·scale`；`uv' = uv + (du, dv)`。
//! 位移上界 `|d| ≤ |scale|` ⇒ halo = `⌈|scale|⌉`（双线性触点不越出该半径
//! ——足额申报 ⇒ tile 化与整幅逐位一致，关卡 1/5）。
//!
//! # 确定性纪律（关卡 5）
//! - 采样用共享原语
//!   [`bilinear_sample_offset`](crate::effects::bilinear_sample_offset)
//!   （位移分解：整数步进精确 + 小数权重与绝对坐标无关）：
//!   四项固定权重式、IEEE floor ⇒ 同输入逐位一致；
//! - **整数格点采样逐位恒等**（权重 (1,0,·,0) 精确）⇒ 整数位移 = 位搬运；
//! - 零超越函数（D-29）；alpha 随几何整场重采样（位移是几何操作，四通道
//!   一致对待）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use crate::effects::bilinear_sample_offset;

/// UV 重映射 warp：输出 (x,y) = 输入 (x + du, y + dv) 的双线性采样，
/// 位移由输入 R/G 通道驱动。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Warp {
    /// 重映射强度：`d = (c·2 − 1)·scale`（`scale = 0` 恒等直通，默认值）。
    pub scale: f32,
}

impl Default for Warp {
    fn default() -> Self {
        Warp { scale: 0.0 }
    }
}

impl TileEffect for Warp {
    fn halo_radius(&self) -> u32 {
        // |du|,|dv| ≤ |scale|；双线性触点 floor(p+d)..floor(p+d)+1 不越出
        // p ± ⌈|scale|⌉。NaN 防退化：as u32 = 0（安全直通）。
        self.scale.abs().ceil() as u32
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let s = self.scale;
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let du = (c[0] * 2.0 - 1.0) * s;
                let dv = (c[1] * 2.0 - 1.0) * s;
                let rgba = bilinear_sample_offset(window, x, y, du, dv);
                out.set(x, y, rgba);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试：关卡 5（恒等逐位 + 半像素）/ 整数位移位搬运 / halo 申报
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性非均匀测试图（prism-noise hash 派生，f16 精确域内）。
    fn noisy_input(w: u32, h: u32) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = prism_noise::pcg::hash2_f32(5, x as u64, y as u64);
                let b = prism_noise::pcg::hash2_f32(9, x as u64, y as u64);
                fb.set(x, y, [a, b, (a + b) * 0.5, 1.0]);
            }
        }
        fb
    }

    #[test]
    fn checkpoint5_warp_identity_scale_zero_bit_exact() {
        // scale = 0 ⇒ du=dv=0 ⇒ 采样整数格点 ⇒ 双线性格点逐位恒等。
        let input = noisy_input(12, 9);
        let out = Warp::default().eval_window(&input);
        assert_eq!(Warp::default().halo_radius(), 0);
        assert_eq!(
            out.pixels, input.pixels,
            "恒等 warp 必须与输入位模式逐位一致"
        );
    }

    #[test]
    fn checkpoint5_warp_half_pixel_shift_matches_expected_bilinear() {
        // R=0.75 全场 ⇒ du = (0.75·2−1)·1 = 0.5（半像素）；G=0.5 ⇒ dv = 0。
        // 数据放 G 通道（x≥3 为 1 的竖直阶跃），半像素位移 ⇒
        // out(x) = (in(x) + in(x+1)) / 2（精确 2 的幂链）。
        let mut input = Fbuf16F::new(6, 3);
        for y in 0..3u32 {
            for x in 0..6u32 {
                let g = if x >= 3 { 1.0 } else { 0.0 };
                input.set(x, y, [0.75, g, 0.5, 0.25]);
            }
        }
        let warp = Warp { scale: 1.0 };
        assert_eq!(warp.halo_radius(), 1);
        let out = warp.eval_window(&input);

        for y in 0..3u32 {
            // G 通道：x+0.5 中点采样（x+1 越界按钳制复制）。
            let expect_g: [f32; 6] = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
            for x in 0..6u32 {
                let px = out.get(x, y);
                assert_eq!(px[1], expect_g[x as usize], "G@({x},{y}) 半像素平移错位");
                // R 输出：0.75 常量场的双线性 = 0.75（2 的幂精确）。
                assert_eq!(px[0], 0.75, "R@({x},{y}) 常量场双线性必须精确");
                // B：0.5 常量场；A：0.25 常量场（alpha 随几何重采样）。
                assert_eq!(px[2], 0.5, "B@({x},{y}) 常量场双线性必须精确");
                assert_eq!(px[3], 0.25, "A@({x},{y}) alpha 必须随场重采样");
            }
        }
    }

    #[test]
    fn warp_integer_shift_is_bit_exact_source_fetch() {
        // R=1.0 ⇒ du = 1·1 = 1（整数位移）；G=0.5 ⇒ dv = 0；数据放 B 通道。
        // 整数位移 = 格点采样逐位恒等 ⇒ 输出 B = 源 B 右移 1 列（位搬运），
        // R/G 常量场平移不变（2 的幂精确）。
        let mut input = Fbuf16F::new(6, 4);
        for y in 0..4u32 {
            for x in 0..6u32 {
                let b = prism_noise::pcg::hash2_f32(3, x as u64, y as u64);
                input.set(x, y, [1.0, 0.5, b, 1.0]);
            }
        }
        let out = Warp { scale: 1.0 }.eval_window(&input);
        for y in 0..4u32 {
            for x in 0..6u32 {
                let sx = (x + 1).min(5); // 右边缘钳制
                assert_eq!(
                    out.get(x, y)[2],
                    input.get(sx, y)[2],
                    "整数位移必须等于源像素位搬运（{x},{y}）"
                );
                assert_eq!(out.get(x, y)[0], 1.0, "R 常量场平移不变（{x},{y}）");
                assert_eq!(out.get(x, y)[1], 0.5, "G 常量场平移不变（{x},{y}）");
            }
        }
    }

    #[test]
    fn warp_halo_covers_max_displacement() {
        assert_eq!(Warp { scale: 0.0 }.halo_radius(), 0, "恒等零半径");
        assert_eq!(Warp { scale: 2.0 }.halo_radius(), 2, "整数 scale 精确覆盖");
        assert_eq!(Warp { scale: 2.5 }.halo_radius(), 3, "亚像素 scale 上取整");
        assert_eq!(Warp { scale: -1.5 }.halo_radius(), 2, "负 scale 取绝对值");
        assert_eq!(Warp { scale: 0.25 }.halo_radius(), 1, "亚像素至少 1（双线性触点）");
    }
}
