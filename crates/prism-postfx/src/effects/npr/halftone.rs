//! halftone（B14 NPR 族：网点印刷仿真）。
//!
//! 语义权威：《算法白皮书》§14 第 3 条——"亮度 → 点阵密度映射 + 点形函数
//! （圆/线可选）+ 屏幕角度族（四角度模拟印刷分版）；彩色版按通道不同角度"。
//!
//! # 实现
//!
//! 对每个通道用**旋转屏幕**（旋转角 `angle`）坐标上取格点中心，计算到格点
//! 中心的距离（圆点）或到中心线距离（线点），与由亮度决定的半径比较：
//! 阈值内 → 前景（墨色），否则 → 背景。旋转用固定序 2×2 旋转矩阵（cos/sin
//! 为编译期常量或多项式近似——D-29 禁 libm，角度取经典 15°/45°/75°，三角
//! 值以常量表给出）。
//!
//! halo = 0（逐像素屏幕键控）；屏幕锚在窗口局部坐标（位置键控，与 B4
//! plasma/vignette 同约定）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::luma;

/// 网点形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalftoneShape {
    /// 圆点（到格点中心距离）。
    Dot,
    /// 线点（到中心线距离）。
    Line,
}

/// halftone 效果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halftone {
    /// 网点格距（像素）。
    pub cell: f32,
    /// 屏幕角度（度；经典 15°/45°/75° 之一）。
    pub angle: f32,
    /// 点形。
    pub shape: HalftoneShape,
    /// 墨色（前景色）。
    pub ink: [f32; 3],
    /// 纸色（背景色）。
    pub paper: [f32; 3],
    /// 每通道独立角度（彩色分版；false = 单角度共用）。
    pub per_channel_angle: bool,
}

impl Default for Halftone {
    fn default() -> Self {
        Halftone {
            cell: 6.0,
            angle: 45.0,
            shape: HalftoneShape::Dot,
            ink: [0.0, 0.0, 0.0],
            paper: [1.0, 1.0, 1.0],
            per_channel_angle: false,
        }
    }
}

/// 经典印刷四角度（青/品红/黄/黑分版），度数 → 弧度以常量表给出三角值。
/// 14 位常数精度足够屏幕仿真；角度是离散选择的编译期常量 ⇒ 跨平台一致。
///
/// D-30：45° 项的 `0.707_106_8`（bits 0x3F3504F4）与
/// `std::f32::consts::FRAC_1_SQRT_2`（0x3F3504F3）**差 1 ulp**。本表是
/// 印刷四角度的**历史固化值**，替换会改变网屏角生成的每一像素 ⇒ 保留原值。
/// 注：0.707_106_8 恰是 1/√2 的正确 f32 向上舍入，写作无误。
#[allow(clippy::approx_constant)] // 精确值 + 跨层固化契约；见 D-30
const ANGLES: [(f32, f32); 4] = [
    // (cos, sin) for 15°, 45°, 75°, 0°
    (0.965_925_8, 0.258_819_0),
    (0.707_106_8, 0.707_106_8),
    (0.258_819_0, 0.965_925_8),
    (1.0, 0.0),
];

/// 角度 → (cos, sin)：取最接近的经典角（离散化，确定性）。
fn angle_cs(deg: f32) -> (f32, f32) {
    if deg < 7.5 {
        ANGLES[3]
    } else if deg < 30.0 {
        ANGLES[0]
    } else if deg < 60.0 {
        ANGLES[1]
    } else {
        ANGLES[2]
    }
}

impl TileEffect for Halftone {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let cell = self.cell.max(1.0);
        let half = 0.5;
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let l = prism_math::clamp(luma(c), 0.0, 1.0);
                // 网点半径：亮度越暗点越大（0.05..0.7 屏幕单位）。
                let radius = (1.0 - l) * 0.7 + 0.05;
                let mut on = [false; 3];
                for ch in 0..3 {
                    let idx = if self.per_channel_angle { ch } else { 0 };
                    let (cs, sn) = angle_cs(if self.per_channel_angle {
                        [15.0, 45.0, 75.0][idx]
                    } else {
                        self.angle
                    });
                    // 旋转屏幕坐标（每像素确定性；窗口局部坐标）。
                    let px = (x as f32 + half) / cell;
                    let py = (y as f32 + half) / cell;
                    let rx = px * cs - py * sn;
                    let ry = px * sn + py * cs;
                    // 到最近格点中心的距离（格点间距 1）。
                    let fx = rx - (rx + 0.5).floor();
                    let fy = ry - (ry + 0.5).floor();
                    let d = match self.shape {
                        HalftoneShape::Dot => (fx * fx + fy * fy).sqrt(),
                        HalftoneShape::Line => fx.abs(),
                    };
                    on[ch] = d < radius;
                }
                // 前景/背景选色（逐通道）。
                let mut rgb = [0.0f32; 3];
                for ch in 0..3 {
                    rgb[ch] = if on[ch] { self.ink[ch] } else { self.paper[ch] };
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
    fn output_is_two_tone_and_darker_input_more_ink() {
        let mut win = Fbuf16F::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                // 左半暗（多墨）、右半亮（少墨）。
                let v = if x < 16 { 0.2 } else { 0.9 };
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let h = Halftone::default();
        let out = h.eval_window(&win);
        let mut ink_left = 0usize;
        let mut ink_right = 0usize;
        for y in 0..32u32 {
            for x in 0..32u32 {
                let c = out.get(x, y);
                // 只在墨/纸两色（f16 量化容差）。
                assert!(
                    (c[0] < 0.05) || (c[0] > 0.95),
                    "必须双色输出，得到 {c:?}"
                );
                if luma(c) < 0.5 {
                    if x < 16 {
                        ink_left += 1;
                    } else {
                        ink_right += 1;
                    }
                }
            }
        }
        assert!(ink_left > ink_right, "暗区网点应更密：{ink_left} vs {ink_right}");
    }

    #[test]
    fn halftone_is_deterministic_and_halo_zero() {
        let mut win = Fbuf16F::new(20, 20);
        let mut state: u64 = 0x4A17;
        for y in 0..20u32 {
            for x in 0..20u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let h = Halftone { per_channel_angle: true, ..Halftone::default() };
        assert_eq!(h.halo_radius(), 0);
        assert_eq!(h.eval_window(&win), h.eval_window(&win), "纯函数契约");
    }

    #[test]
    fn line_shape_differs_from_dot() {
        let mut win = Fbuf16F::new(24, 24);
        for y in 0..24u32 {
            for x in 0..24u32 {
                win.set(x, y, [0.5, 0.5, 0.5, 1.0]);
            }
        }
        let dot = Halftone { shape: HalftoneShape::Dot, ..Halftone::default() }.eval_window(&win);
        let line = Halftone { shape: HalftoneShape::Line, ..Halftone::default() }.eval_window(&win);
        assert_ne!(dot.pixels, line.pixels, "点形/线形图案必须不同");
    }
}
