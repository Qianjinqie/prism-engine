//! grade（清单：grade —— 调色）。
//!
//! 语义权威：grade与 （"线性域白平衡/曝光/
//! 饱和——色彩空间是节点的显式参数"；grade/lut3d 是色彩节点链）；宪章 C4
//! 对照 #7（Color Grade / 调色（达芬奇、Lumetri）→ `grade` 节点，词汇直用）。
//!
//! # 实现（固定级联序，逐级中性跳过）
//!
//! `exposure（线性倍率） → slope（通道增益） → offset（通道偏移） →
//! contrast（对比，绕 0.5 枢轴） → saturation（Rec.709 亮度轴）`。
//!
//! 每级在参数中性时**整级跳过**（参数是效果实例的常量 → 分支确定）：
//! 保证中性级联逐位恒等（关卡 2）——`(c − 0.5)·1 + 0.5` 在 f32 下并不
//! 恒等（两次舍入），跳过是正确性要求而非性能优化。power（gamma）段需
//! 确定性 log2 数学件（D-29 落位 prism-math），当前缺席——缺口如实上报。
//!
//! halo = 0；可平移不变（tile 一致性验证集成员）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// 调色效果（参数全部走属性通道；默认全中性 = 恒等）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grade {
    /// 线性曝光倍率（1 = 中性；stops→linear 换算属上层）。
    pub exposure: f32,
    /// 对比度（绕 0.5 枢轴的线性斜率；1 = 中性）。
    pub contrast: f32,
    /// 饱和度（0 = 灰度，1 = 中性）。
    pub saturation: f32,
    /// 通道增益 slope（RGB；[1,1,1] = 中性）。
    pub slope: [f32; 3],
    /// 通道偏移 offset（RGB；[0,0,0] = 中性）。
    pub offset: [f32; 3],
}

impl Default for Grade {
    fn default() -> Self {
        Grade {
            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
        }
    }
}

/// Rec.709 亮度系数（BT.709 luma，固定常量）。
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

impl Grade {
    /// 对单个像素颜色应用级联（供 ApplyLut 导出链等复用的纯函数）。
    pub fn apply_color(&self, c: [f32; 4]) -> [f32; 4] {
        let mut rgb = [c[0], c[1], c[2]];

        if self.exposure != 1.0 {
            for v in rgb.iter_mut() {
                *v *= self.exposure;
            }
        }
        if self.slope != [1.0, 1.0, 1.0] {
            for (v, s) in rgb.iter_mut().zip(self.slope.iter()) {
                *v *= s;
            }
        }
        if self.offset != [0.0, 0.0, 0.0] {
            for (v, o) in rgb.iter_mut().zip(self.offset.iter()) {
                *v += o;
            }
        }
        if self.contrast != 1.0 {
            for v in rgb.iter_mut() {
                *v = (*v - 0.5) * self.contrast + 0.5;
            }
        }
        if self.saturation != 1.0 {
            let lum = rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2];
            for v in rgb.iter_mut() {
                *v = lum + (*v - lum) * self.saturation;
            }
        }
        [rgb[0], rgb[1], rgb[2], c[3]]
    }
}

impl TileEffect for Grade {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                out.set(x, y, self.apply_color(window.get(x, y)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint2_neutral_grade_is_bit_exact_identity() {
        // 关卡 2：全中性 grade 输入输出逐位一致。
        let mut win = Fbuf16F::new(10, 6);
        for y in 0..6u32 {
            for x in 0..10u32 {
                let v = (x * 17 + y * 5) as f32 / 90.0;
                win.set(x, y, [v, 1.0 - v, v * 0.3, 0.8]);
            }
        }
        let out = Grade::default().eval_window(&win);
        assert_eq!(out, win, "中性 grade 必须逐位恒等");
    }

    #[test]
    fn stages_behave_as_named() {
        let gray_in = [0.5f32, 0.25, 0.75, 1.0];

        // exposure：线性倍率。
        let g = Grade { exposure: 2.0, ..Grade::default() };
        let out = g.apply_color(gray_in);
        assert_eq!((out[0], out[1], out[2]), (1.0, 0.5, 1.5));

        // offset：通道偏移。
        let g = Grade { offset: [0.1, 0.0, -0.1], ..Grade::default() };
        let out = g.apply_color(gray_in);
        assert_eq!((out[0], out[1], out[2]), (0.6, 0.25, 0.65));

        // contrast：0.5 枢轴不动，斜率放大间距。
        let g = Grade { contrast: 2.0, ..Grade::default() };
        assert_eq!(g.apply_color([0.5, 0.0, 1.0, 1.0])[0], 0.5);
        let out = g.apply_color(gray_in);
        assert_eq!(out[0], (0.5 - 0.5) * 2.0 + 0.5);
        assert_eq!(out[1], (0.25 - 0.5) * 2.0 + 0.5);

        // saturation 0：投影到 Rec.709 亮度轴（三通道相等）。
        let g = Grade { saturation: 0.0, ..Grade::default() };
        let out = g.apply_color(gray_in);
        let lum = 0.5 * LUMA[0] + 0.25 * LUMA[1] + 0.75 * LUMA[2];
        assert_eq!((out[0], out[1], out[2]), (lum, lum, lum));

        // alpha 全程不动。
        assert_eq!(g.apply_color(gray_in)[3], 1.0);
    }

    #[test]
    fn partial_neutral_stages_do_not_disturb() {
        // 单级启用时其余级必须零扰动（位精确）——跳过式级联的核心性质。
        let mut win = Fbuf16F::new(8, 4);
        for y in 0..4u32 {
            for x in 0..8u32 {
                let v = (x + y) as f32 / 12.0;
                win.set(x, y, [v, 1.0 - v, v * 0.5, 1.0]);
            }
        }
        let exposure_only = Grade { exposure: 0.5, ..Grade::default() };
        let out = exposure_only.eval_window(&win);
        for y in 0..4u32 {
            for x in 0..8u32 {
                let c = win.get(x, y);
                assert_eq!(out.get(x, y), [c[0] * 0.5, c[1] * 0.5, c[2] * 0.5, c[3]]);
            }
        }
    }

    #[test]
    fn grade_is_pure_and_translation_invariant_input() {
        let mut win = Fbuf16F::new(6, 5);
        let mut state: u64 = 0x6ADE;
        for y in 0..5u32 {
            for x in 0..6u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let g = Grade { exposure: 1.2, contrast: 0.9, saturation: 1.4, slope: [1.0, 0.95, 1.05], offset: [0.01, 0.0, -0.01] };
        assert_eq!(g.eval_window(&win), g.eval_window(&win), "纯函数契约");
    }
}
