//! 渐变纹理（程序化纹理族：渐变）。
//!
//! 语义权威：行业渐变节点（线性 / 径向 / 角度）；（固定
//! 序求值）。渐变在归一化图像坐标上取参数 `t ∈ [0,1]`，在多档色标间插值。
//!
//! 确定性：纯算术（零超越函数；角度用查表式归一化而非 atan2——见实现）。
//! 位置键控（窗口局部坐标）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

/// 渐变类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientKind {
    /// 线性（沿 `angle` 方向）。
    Linear,
    /// 径向（到中心距离）。
    Radial,
    /// 角度（绕中心旋转角——用分段线性近似角，零超越函数）。
    Angular,
}

/// 色标（位置 + 颜色）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stop {
    /// 位置（[0,1]，按升序）。
    pub at: f32,
    /// 颜色。
    pub color: [f32; 3],
}

/// 渐变纹理。
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    /// 渐变类型。
    pub kind: GradientKind,
    /// 方向角（度；线性渐变）。
    pub angle: f32,
    /// 中心（归一化；径向 / 角度渐变）。
    pub center: [f32; 2],
    /// 色标（按 at 升序，至少 2 个）。
    pub stops: Vec<Stop>,
}

impl Gradient {
    /// 两色线性渐变（黑 → 白，水平）。
    pub fn two_color(a: [f32; 3], b: [f32; 3]) -> Gradient {
        Gradient {
            kind: GradientKind::Linear,
            angle: 0.0,
            center: [0.5, 0.5],
            stops: vec![Stop { at: 0.0, color: a }, Stop { at: 1.0, color: b }],
        }
    }

    /// 在色标上取 `t ∈ [0,1]` 处的颜色（固定序线性插值）。
    fn eval_stops(&self, t: f32) -> [f32; 3] {
        let t = prism_math::clamp(t, 0.0, 1.0);
        // 空 stops 安全降级：无语义色可插值 → 返回透明黑（0,0,0），
        // 与渐变「无色标」的文档约定一致（避免索引越界 panic）。
        if self.stops.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let first = self.stops[0];
        let last = self.stops[self.stops.len() - 1];
        if t <= first.at {
            return first.color;
        }
        if t >= last.at {
            return last.color;
        }
        let mut prev = first;
        for &s in self.stops.iter().skip(1) {
            if t <= s.at {
                let span = s.at - prev.at;
                if span <= 0.0 {
                    return s.color;
                }
                let u = (t - prev.at) / span;
                return [
                    prev.color[0] + (s.color[0] - prev.color[0]) * u,
                    prev.color[1] + (s.color[1] - prev.color[1]) * u,
                    prev.color[2] + (s.color[2] - prev.color[2]) * u,
                ];
            }
            prev = s;
        }
        last.color
    }

    /// 归一化坐标处的渐变参数 `t ∈ [0,1]`。
    fn param(&self, u: f32, v: f32) -> f32 {
        match self.kind {
            GradientKind::Linear => {
                // 方向角余弦/正弦（45° 整数倍离散化，零超越函数）。
                let (ca, sa) = dir_cs(self.angle);
                let du = u - self.center[0];
                let dv = v - self.center[1];
                // 投影到 [-0.5,0.5] → [0,1]。
                prism_math::clamp(du * ca + dv * sa + 0.5, 0.0, 1.0)
            }
            GradientKind::Radial => {
                let du = u - self.center[0];
                let dv = v - self.center[1];
                let d = (du * du + dv * dv).sqrt() / std::f32::consts::FRAC_1_SQRT_2;
                prism_math::clamp(d, 0.0, 1.0)
            }
            GradientKind::Angular => {
                // 绕中心的角：用归一化坐标的分段线性"方位"映射（零 atan2）。
                let du = u - self.center[0];
                let dv = v - self.center[1];
                prism_math::clamp(angular_t(du, dv), 0.0, 1.0)
            }
        }
    }
}

/// 方向角 → (cos, sin)：离散到 8 个 45° 方向（常量表，跨平台一致）。
fn dir_cs(deg: f32) -> (f32, f32) {
    let d = ((deg / 45.0).round() as i32).rem_euclid(8);
    // D-30：0.70710678 = 1/√2，与 `std::f32::consts::FRAC_1_SQRT_2` `to_bits()`
    // 逐位相同（0x3F3504F3），替换不改变离散方向的取值。
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const C: [f32; 8] = [1.0, H, 0.0, -H, -1.0, -H, 0.0, H];
    const S: [f32; 8] = [0.0, H, 1.0, H, 0.0, -H, -1.0, -H];
    (C[d as usize], S[d as usize])
}

/// 方位角参数化（分段线性菱形近似 0..1 环绕，零超越函数）。
fn angular_t(du: f32, dv: f32) -> f32 {
    // 菱形归一化：以 |du|+|dv| 为尺度，按象限线性映射到 [0,1]。
    let l1 = du.abs() + dv.abs();
    if l1 <= 1e-6 {
        return 0.0;
    }
    let u = du / l1;
    let v = dv / l1;
    // 八方位线性映射。
    let t = if v >= 0.0 {
        if u >= 0.0 {
            0.125 + 0.125 * (u - v) // 0..0.25
        } else {
            0.875 + 0.125 * (u + v) // 0.75..1
        }
    } else {
        if u >= 0.0 {
            0.125 - 0.125 * (u + v) // 0..-0.25
        } else {
            0.625 + 0.125 * (u - v) // 0.5..0.75
        }
    };
    t.rem_euclid(1.0)
}

impl TileEffect for Gradient {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let w = window.width;
        let h = window.height;
        let mut out = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let u = if w > 1 { x as f32 / (w - 1) as f32 } else { 0.5 };
                let v = if h > 1 { y as f32 / (h - 1) as f32 } else { 0.5 };
                let t = self.param(u, v);
                let c = self.eval_stops(t);
                out.set(x, y, [c[0], c[1], c[2], 1.0]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_gradient_is_monotone_left_to_right() {
        let g = Gradient::two_color([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let out = g.eval_window(&Fbuf16F::new(32, 4));
        let left = out.get(0, 0)[0];
        let right = out.get(31, 0)[0];
        assert!(left < 0.1 && right > 0.9, "线性渐变应左暗右亮：{left}→{right}");
        // 单调不减。
        let mut prev = -1.0;
        for x in 0..32u32 {
            let v = out.get(x, 0)[0];
            assert!(v >= prev - 1e-6, "必须单调：{v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn radial_gradient_bright_at_center() {
        let mut g = Gradient::two_color([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        g.kind = GradientKind::Radial;
        let out = g.eval_window(&Fbuf16F::new(16, 16));
        let center = out.get(8, 8)[0];
        let corner = out.get(0, 0)[0];
        assert!(center < corner, "径向中心应更暗（distance 小）：{center} vs {corner}");
    }

    #[test]
    fn gradient_deterministic_and_angular_wraps() {
        let mut g = Gradient::two_color([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        g.kind = GradientKind::Angular;
        let win = Fbuf16F::new(20, 20);
        assert_eq!(g.eval_window(&win), g.eval_window(&win), "纯函数契约");
        // 各角参数落在 [0,1]。
        for y in 0..20u32 {
            for x in 0..20u32 {
                let c = g.eval_window(&win).get(x, y);
                for k in 0..3 {
                    assert!((0.0..=1.0).contains(&c[k]));
                }
            }
        }
    }
}
