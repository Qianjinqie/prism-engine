//! 曲线库（B14：二级调色曲线）。
//!
//! 语义权威：RGB 曲线；行业对照（达芬奇曲线 /
//! Photoshop 曲线）。曲线 = 单位域 [0,1] 上的**单调分段线性**映射，由若干
//! 控制点定义；求值零超越函数（D-29），固定序线性插值 ⇒ 逐位可复现。
//!
//! # 单调性保证
//!
//! 控制点按 x 升序存储；求值只做相邻点线性插值。调用方按序给点即单调；
//! 本库不强制单调（与行业工具一致），但提供 `is_monotone` 供断言。

/// 曲线控制点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvePoint {
    /// 输入（x，建议 [0,1]）。
    pub x: f32,
    /// 输出（y）。
    pub y: f32,
}

/// 单条单调分段线性曲线。
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    /// 控制点（按 x 升序；至少 2 点）。
    pub points: Vec<CurvePoint>,
}

impl Curve {
    /// 构造曲线；空或单点时补默认端点（[0,0] / [1,1] 恒等）。
    pub fn new(points: Vec<CurvePoint>) -> Curve {
        let mut c = Curve { points };
        if c.points.is_empty() {
            c.points = vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 1.0 }];
        } else if c.points.len() == 1 {
            let p = c.points[0];
            c.points = vec![CurvePoint { x: 0.0, y: p.y }, CurvePoint { x: 1.0, y: p.y }];
        }
        c.points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        c
    }

    /// 恒等曲线（[0,0]→[1,1]，逐位恒等）。
    pub fn identity() -> Curve {
        Curve { points: vec![CurvePoint { x: 0.0, y: 0.0 }, CurvePoint { x: 1.0, y: 1.0 }] }
    }

    /// 是否是恒等曲线（两端点 = 对角线；求值逐位恒等）。
    pub fn is_identity(&self) -> bool {
        self.points.len() == 2
            && self.points[0] == CurvePoint { x: 0.0, y: 0.0 }
            && self.points[1] == CurvePoint { x: 1.0, y: 1.0 }
    }

    /// 单点求值：x 先钳到控制点范围，再做相邻点线性插值（固定序）。
    pub fn eval(&self, x: f32) -> f32 {
        let first = self.points[0];
        let last = self.points[self.points.len() - 1];
        if x <= first.x {
            return first.y;
        }
        if x >= last.x {
            return last.y;
        }
        let mut prev = first;
        for &p in self.points.iter().skip(1) {
            if x <= p.x {
                let span = p.x - prev.x;
                if span <= 0.0 {
                    return p.y;
                }
                let t = (x - prev.x) / span;
                return prev.y + (p.y - prev.y) * t;
            }
            prev = p;
        }
        last.y
    }

    /// 控制点 x 单调不减（非严格）。
    pub fn is_monotone_x(&self) -> bool {
        self.points.windows(2).all(|w| w[1].x >= w[0].x)
    }

    /// 曲线整体单调不减（相邻段斜率 >= 0）。
    pub fn is_monotone(&self) -> bool {
        self.points.windows(2).all(|w| w[1].y >= w[0].y)
    }
}

/// 四通道曲线集（主 / 红 / 绿 / 蓝；达芬奇式）。
///
/// 求值固定序：先各通道曲线，再主曲线（luma 曲线）。中性（全恒等）时
/// 逐级跳过 ⇒ 逐位恒等（关卡 2）。
#[derive(Debug, Clone, PartialEq)]
pub struct CurveSet {
    /// 主曲线（按亮度施加于三通道）。
    pub master: Curve,
    /// R 通道曲线。
    pub red: Curve,
    /// G 通道曲线。
    pub green: Curve,
    /// B 通道曲线。
    pub blue: Curve,
}

impl Default for CurveSet {
    fn default() -> Self {
        CurveSet {
            master: Curve::identity(),
            red: Curve::identity(),
            green: Curve::identity(),
            blue: Curve::identity(),
        }
    }
}

impl CurveSet {
    /// 是否全恒等。
    pub fn is_identity(&self) -> bool {
        self.master.is_identity()
            && self.red.is_identity()
            && self.green.is_identity()
            && self.blue.is_identity()
    }

    /// 对单像素应用曲线集（固定序：通道曲线 → 主曲线）。
    pub fn apply(&self, c: [f32; 4]) -> [f32; 4] {
        // 通道曲线（中性跳过）。
        let mut rgb = [c[0], c[1], c[2]];
        if !self.red.is_identity() {
            rgb[0] = self.red.eval(rgb[0]);
        }
        if !self.green.is_identity() {
            rgb[1] = self.green.eval(rgb[1]);
        }
        if !self.blue.is_identity() {
            rgb[2] = self.blue.eval(rgb[2]);
        }
        // 主曲线（按各通道自身值映射——行业"主曲线"语义按通道值）。
        if !self.master.is_identity() {
            for v in rgb.iter_mut() {
                *v = self.master.eval(*v);
            }
        }
        [rgb[0], rgb[1], rgb[2], c[3]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve_is_bit_exact_on_grid() {
        let c = Curve::identity();
        for &x in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(c.eval(x), x, "恒等曲线网格点必须逐位一致");
        }
    }

    #[test]
    fn monotone_curve_eval_is_clamped_and_linear_between_points() {
        let c = Curve::new(vec![
            CurvePoint { x: 0.0, y: 0.0 },
            CurvePoint { x: 0.5, y: 0.25 },
            CurvePoint { x: 1.0, y: 1.0 },
        ]);
        assert!(c.is_monotone() && c.is_monotone_x());
        assert_eq!(c.eval(0.25), 0.125, "段内线性插值");
        assert_eq!(c.eval(0.5), 0.25);
        // 越界钳制到端点。
        assert_eq!(c.eval(-1.0), 0.0);
        assert_eq!(c.eval(2.0), 1.0);
    }

    #[test]
    fn curve_set_neutral_is_identity_and_per_channel_applies() {
        let cs = CurveSet::default();
        assert!(cs.is_identity());
        let c = [0.3f32, 0.6, 0.9, 0.5];
        assert_eq!(cs.apply(c), c, "中性曲线集逐位恒等");

        // 仅红通道反相。
        let mut red = CurveSet::default();
        red.red = Curve::new(vec![
            CurvePoint { x: 0.0, y: 1.0 },
            CurvePoint { x: 1.0, y: 0.0 },
        ]);
        let out = red.apply(c);
        assert_eq!(out[0], 0.7, "红通道反相 0.3 → 0.7");
        assert_eq!((out[1], out[2]), (0.6, 0.9), "其余通道不动");
        assert_eq!(out[3], 0.5, "alpha 不动");
    }

    #[test]
    fn curve_set_is_deterministic() {
        let mut cs = CurveSet::default();
        cs.master = Curve::new(vec![
            CurvePoint { x: 0.0, y: 0.0 },
            CurvePoint { x: 0.5, y: 0.6 },
            CurvePoint { x: 1.0, y: 1.0 },
        ]);
        for i in 0..100 {
            let x = i as f32 / 99.0;
            assert_eq!(cs.apply([x, x, x, 1.0]), cs.apply([x, x, x, 1.0]));
        }
    }
}
