//! box blur（清单：blur）。
//!
//! 语义权威：blur；（S2
//! 卷积特解，halo = 核半径）；A5 `BoxBlur3` 的可参数化推广。
//!
//! # 实现
//!
//! 可分离两趟 box blur：先水平后垂直，各趟对 `(2r+1)` 个 tap 按 `-r..=r`
//! **升序固定顺序**求和后 ÷ `(2r+1)`（像素内固定序归约）；
//! 窗口边缘按钳制复制采样（与 `BoxBlur3` 同约定）。半径 0 为恒等
//! （单 tap、÷1，位模式不变）。
//!
//! halo = `radius`：可分离支撑恰为每轴 ±r，输出像素 p 只读输入 `p ± r`。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// 可分离 box blur 核心（本 crate 内共享：bloom 的半分辨率模糊、sharpen 的
/// 非锐化掩模均复用；`radius = 0` 返回逐位相同的副本）。
pub(crate) fn box_blur_buf(input: &Fbuf16F, radius: u32) -> Fbuf16F {
    let w = input.width;
    let h = input.height;
    let n = (2 * radius + 1) as f32;
    let mut tmp = Fbuf16F::new(w, h);
    // 趟一：水平（行主序遍历，tap 升序固定顺序求和）。
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for dx in -(radius as i64)..=(radius as i64) {
                let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as u32;
                let px = input.get(sx, y);
                for (a, p) in acc.iter_mut().zip(px.iter()) {
                    *a += *p;
                }
            }
            tmp.set(
                x,
                y,
                [acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n],
            );
        }
    }
    // 趟二：垂直（对趟一结果）。
    let mut out = Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for dy in -(radius as i64)..=(radius as i64) {
                let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as u32;
                let px = tmp.get(x, sy);
                for (a, p) in acc.iter_mut().zip(px.iter()) {
                    *a += *p;
                }
            }
            out.set(
                x,
                y,
                [acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n],
            );
        }
    }
    out
}

/// box blur 效果：半径 `radius` 的可分离均值模糊（RGBA 四通道同参）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blur {
    /// 模糊半径（像素；0 = 恒等）。
    pub radius: u32,
}

impl Default for Blur {
    fn default() -> Self {
        Blur { radius: 2 }
    }
}

impl TileEffect for Blur {
    fn halo_radius(&self) -> u32 {
        self.radius
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        box_blur_buf(window, self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 期望值的 f16 量化（测试期望须过同一写回域，A5 tile.rs 同款）。
    fn q(v: f32) -> f32 {
        prism_io::f16_to_f32(prism_io::f32_to_f16(v))
    }

    #[test]
    fn radius_zero_is_bit_exact_identity() {
        let mut win = Fbuf16F::new(5, 4);
        for y in 0..4u32 {
            for x in 0..5u32 {
                let v = (x * 7 + y * 3) as f32 / 11.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let out = Blur { radius: 0 }.eval_window(&win);
        assert_eq!(out, win, "radius 0 必须逐位恒等");
    }

    #[test]
    fn delta_impulse_spreads_by_kernel_width() {
        // δ 冲激：中心单点亮 → 半径 1 时水平/垂直各 ±1 邻域共享能量；
        // 可分离均值 = 3×3 均值（与 A5 BoxBlur3 同值，交叉验证）。
        let mut win = Fbuf16F::new(5, 5);
        win.set(2, 2, [1.0, 0.0, 0.0, 1.0]);
        let out = Blur { radius: 1 }.eval_window(&win);
        for (x, y, expected) in [
            (2u32, 2u32, 1.0 / 9.0),
            (3u32, 2u32, 1.0 / 9.0),
            (1u32, 1u32, 1.0 / 9.0),
            (0u32, 0u32, 0.0),
        ] {
            let got = out.get(x, y)[0];
            assert_eq!(got, q(expected), "({x},{y}) 冲激扩散值不符");
        }
    }

    #[test]
    fn uniform_field_is_fixed_point_and_energy_conserved() {
        // 常量场不动点 + 总能量守恒（模糊不增减能量）。
        let mut win = Fbuf16F::new(8, 6);
        for y in 0..6u32 {
            for x in 0..8u32 {
                win.set(x, y, [0.375, 0.25, 0.5, 1.0]);
            }
        }
        let out = Blur { radius: 3 }.eval_window(&win);
        for y in 0..6u32 {
            for x in 0..8u32 {
                assert_eq!(out.get(x, y), [0.375, 0.25, 0.5, 1.0], "({x},{y}) 不动点破坏");
            }
        }
    }

    #[test]
    fn halo_declaration_matches_kernel_radius() {
        // halo 申报 = 核半径（关卡 1 的静态面；动态面由 tile 一致性测试覆盖）。
        assert_eq!(Blur { radius: 0 }.halo_radius(), 0);
        assert_eq!(Blur { radius: 5 }.halo_radius(), 5);
    }

    #[test]
    fn blur_is_pure_and_shared_core_matches_effect() {
        // 纯函数契约 + 共享核心与效果入口一致（bloom/sharpen 复用面）。
        let mut win = Fbuf16F::new(7, 5);
        let mut state: u64 = 0xC0FFEE;
        for y in 0..5u32 {
            for x in 0..7u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let effect = Blur { radius: 2 }.eval_window(&win);
        let core = box_blur_buf(&win, 2);
        assert_eq!(effect, core);
        assert_eq!(
            Blur { radius: 2 }.eval_window(&win),
            effect,
            "同输入必同输出"
        );
    }
}
