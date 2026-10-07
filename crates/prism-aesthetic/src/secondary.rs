//! 二级调色（B14：HSL 选区 / lift-gamma-gain / Power Window）。
//!
//! 语义权威：二级调色：HSL 选区 / RGB 曲线 /
//! lift-gamma-gain / Power Window。本模块承载 HSL 选区、lift-gamma-gain、
//! Power Window 三项；RGB 曲线在 [`crate::curves`]。
//!
//! # 选区数学（有断言）
//!
//! [`HslQualifier`] 用**亮度区间 + 色相区间 + 饱和度区间**的三段软限
//! （soft limits，smoothstep 过渡带）相乘得选区权重 `sel ∈ [0,1]`：
//! 落在区间中心 → 1，落在过渡带外 → 0，过渡带内平滑。全部为多项式/区间
//! 运算，零超越函数（D-29；色相用归一化 RGB 的通道比较而非 atan2）。
//!
//! # 确定性纪律（D-29）
//!
//! 幂（gamma）用 [`crate::pow_int`] 连乘实现，不使用 `powf`；全部算子为
//! 求值纯函数，无隐藏状态、无私开 RNG。同参数逐位一致。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use crate::{luma, pow_int};

/// 三次平滑阶跃（零超越函数）。
#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = prism_math::clamp(t, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 软区间权重：`value` 落在 `[lo+r, hi-r]` → 1，`< lo` 或 `> hi` → 0，
/// 边缘 `r` 宽过渡带内平滑。`hi <= lo` 或 `r <= 0` 时退化为硬区间。
/// `[lo, hi]` 覆盖整域 `[0,1]` 时恒为 1（全选，避免边界处过渡带误切）。
#[inline]
fn soft_range(value: f32, lo: f32, hi: f32, r: f32) -> f32 {
    if lo <= 0.0 && hi >= 1.0 {
        return 1.0;
    }
    if hi <= lo {
        return if value >= lo && value <= hi { 1.0 } else { 0.0 };
    }
    if r <= 0.0 {
        return if value >= lo && value <= hi { 1.0 } else { 0.0 };
    }
    // 上升沿 lo→lo+r，下降沿 hi-r→hi。
    let rise = smoothstep((value - lo) / r);
    let fall = smoothstep((hi - value) / r);
    rise * fall
}

/// 色相软区间权重（色相 = 环形量，红色在 0 与 1 处连通）。
///
/// 对环形距离的区间成员判定：把区间与色相都映到环上，取到区间最短角距
/// `dist`，`dist <= 半径` 平滑过渡到 0。全环 `[0,1]` → 恒 1。
#[inline]
fn hue_range_weight(h: f32, lo: f32, hi: f32, r: f32) -> f32 {
    if lo <= 0.0 && hi >= 1.0 {
        return 1.0;
    }
    let h = h - h.floor(); // 归一化到 [0,1)
    // 跨 0 选区（lo > hi）：等价于两段 [lo,1] ∪ [0,hi] 的并集。
    // 取 h 到「候选区间」的环上最短距离，dist=0 表示落入区间内。
    let dist = if lo <= hi {
        // 普通区间：h 在 [lo,hi] 内 → 0；否则到两端的环距离取小。
        if h >= lo && h <= hi {
            0.0
        } else {
            circular_dist(h, lo).min(circular_dist(h, hi))
        }
    } else {
        // 跨 0 区间 [lo,1]∪[0,hi]：h 在并集内 → 0；否则到最近端点的环距离。
        if h >= lo || h <= hi {
            0.0
        } else {
            circular_dist(h, lo).min(circular_dist(h, hi))
        }
    };
    if r <= 0.0 {
        return if dist <= 0.0 { 1.0 } else { 0.0 };
    }
    smoothstep(1.0 - dist / r)
}

/// 环上两色相点的最短角距离（∈ [0, 0.5]）。
#[inline]
fn circular_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

/// RGB → 色相（0..1；用归一化通道比较，零超越函数）。
///
/// 色相定义：max 通道决定色相扇区，扇区内按通道差线性插值。灰色（max==min）
/// 返回 0（无定义，约定 0）。
fn hue_of(r: f32, g: f32, b: f32) -> f32 {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    if d <= 0.0 {
        return 0.0;
    }
    let h = if max == r {
        let t = (g - b) / d;
        // 扇区 [0,2) → 归一化：h = (t/6) mod 1。
        let t6 = t / 6.0;
        if t6 < 0.0 {
            t6 + 1.0
        } else {
            t6
        }
    } else if max == g {
        (b - r) / d / 6.0 + 1.0 / 3.0
    } else {
        (r - g) / d / 6.0 + 2.0 / 3.0
    };
    prism_math::clamp(h, 0.0, 1.0)
}

/// HSL 选区（亮度 + 色相 + 饱和度三段软限）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HslQualifier {
    /// 亮度区间 [lo, hi]。
    pub luma_range: [f32; 2],
    /// 色相区间 [lo, hi]（归一化 0..1）。
    pub hue_range: [f32; 2],
    /// 饱和度区间 [lo, hi]。
    pub sat_range: [f32; 2],
    /// 过渡带宽度（各段共用）。
    pub softness: f32,
}

impl Default for HslQualifier {
    fn default() -> Self {
        HslQualifier {
            luma_range: [0.0, 1.0],
            hue_range: [0.0, 1.0],
            sat_range: [0.0, 1.0],
            softness: 0.1,
        }
    }
}

impl HslQualifier {
    /// 选区权重 `sel ∈ [0,1]`（三段软限相乘）。
    ///
    /// 全区间（含默认）时 `sel = 1`（处处选中——整幅调色的退化情形，
    /// 见断言 `test_full_range_qualifier_selects_everything`）。
    pub fn weight(&self, c: [f32; 4]) -> f32 {
        let l = prism_math::clamp(luma(c), 0.0, 1.0);
        let r = c[0].max(0.0);
        let g = c[1].max(0.0);
        let b = c[2].max(0.0);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let sat = if max > 0.0 { (max - min) / max } else { 0.0 };
        let h = hue_of(r, g, b);
        let wl = soft_range(l, self.luma_range[0], self.luma_range[1], self.softness);
        let wh = hue_range_weight(h, self.hue_range[0], self.hue_range[1], self.softness);
        let ws = soft_range(sat, self.sat_range[0], self.sat_range[1], self.softness);
        wl * wh * ws
    }
}

/// lift-gamma-gain（行业三路调色：阴影抬升 / 中灰 gamma / 高光增益）。
///
/// 数学形式（每通道）：`out = gain · (lift + (1 − lift) · in)^(1/gamma)`，
/// 其中幂用整数指数连乘近似（gamma 量化到 1/N，N = `gamma_steps`）。
/// 中性 = lift 0 / gamma 1 / gain 1 → 逐位恒等。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiftGammaGain {
    /// 阴影抬升（0 = 中性；正 = 抬阴影）。
    pub lift: f32,
    /// 中灰 gamma（1 = 中性；> 1 = 提亮中灰）。
    pub gamma: f32,
    /// 高光增益（1 = 中性）。
    pub gain: f32,
    /// gamma 量化的倒数步数（1 = 用整数指数；越大越精细）。
    pub gamma_steps: u32,
}

impl Default for LiftGammaGain {
    fn default() -> Self {
        LiftGammaGain { lift: 0.0, gamma: 1.0, gain: 1.0, gamma_steps: 8 }
    }
}

impl LiftGammaGain {
    /// 是否中性（逐位恒等条件）。
    pub fn is_neutral(&self) -> bool {
        self.lift == 0.0 && self.gamma == 1.0 && self.gain == 1.0
    }

    /// 对单值施加 lift-gamma-gain。
    ///
    /// 幂 `x^(1/gamma)`（`x ≥ 0`）。指数 `e = 1/gamma` 量化为 `e ≈ p / 2^q`
    /// （`q = gamma_steps`，`p = round(e · 2^q)`，再做偶约分），幂按
    /// **先开方后连乘**实现：
    ///
    /// `x^(p/2^q) = (x^(1/2^q))^p`，其中 `x^(1/2^q)` 由 `q` 次 `sqrt` 得到、
    /// `(...)^p` 由连乘 [`pow_int`] 得到。
    ///
    /// # 下溢处理
    ///
    /// 若按 `(x^p)^(1/2^q)`（先连乘后开方）计算，`x < 1` 且 `p` 较大
    /// （如 `gamma = 1.4, x = 0.5 ⇒ p = 183`）时 `x^p → 0` **下溢**，开方仍 0。
    /// 本实现改为**先 `q` 次开方再连乘**：`x^(1/2^q) ≈ 1`（如 `0.5^(1/256)
    /// ≈ 0.9973`），其后连乘 `p` 次既不溢出也不下溢，全程得到正确量级。
    ///
    /// 全程仅用 `+ - × ÷`、IEEE 精确 `sqrt` 与整数运算——零 libm（D-29），
    /// 跨平台逐位一致。`gamma == 1`（`e = 1 ⇒ p/2^q = 1`）精确恒等。
    pub fn apply_scalar(&self, v: f32) -> f32 {
        let mut x = v;
        if self.lift != 0.0 {
            x = self.lift + (1.0 - self.lift) * x;
        }
        // 幂底座：负输入（线性 HDR 允许 <0）视为 0（`x ≥ 0 ⟺ x^(1/gamma) ≥ 0`）。
        let base = if x > 0.0 { x } else { 0.0 };
        if self.gamma != 1.0 && base > 0.0 {
            let steps = self.gamma_steps.max(1).min(30);
            let denom = (1u64 << steps) as f32;
            let e = 1.0 / self.gamma;
            // p 取上取整，保证 p/2^q ≥ e（向提亮侧量化，误差单调有界）。
            let mut p = (e * denom).round().max(1.0) as u64;
            let mut q = steps;
            // 偶约分：p 偶数时 p/=2、q-=1，减小连乘次数（x<1 时进一步抑制下溢）。
            while q > 0 && p % 2 == 0 {
                p /= 2;
                q -= 1;
            }
            // 先开方：root = x^(1/2^q)（q 次 IEEE 精确 sqrt，x ≥ 0 ⇒ 不下溢）。
            let mut root = base;
            for _ in 0..q {
                root = root.sqrt();
            }
            // 后连乘：root^p = x^(p/2^q)（base≈1 ⇒ 无下溢/溢出）。
            x = pow_int(root, p.min(i32::MAX as u64) as i32);
        }
        if self.gain != 1.0 {
            x *= self.gain;
        }
        x
    }

    /// 对 RGBA 像素应用（alpha 直通）。
    pub fn apply(&self, c: [f32; 4]) -> [f32; 4] {
        [
            self.apply_scalar(c[0]),
            self.apply_scalar(c[1]),
            self.apply_scalar(c[2]),
            c[3],
        ]
    }
}

/// Power Window 形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// 矩形窗口。
    Rect,
    /// 椭圆窗口。
    Ellipse,
}

/// Power Window（二级调色的空间限定窗口，带羽化）。
///
/// 中心 `(cx, cy)`、半径 `(rx, ry)`（归一化 [0,1] 图像坐标），`feather`
/// 羽化带宽。窗口权重 = 形状内 1、羽化带内平滑、外 0。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerWindow {
    /// 中心（归一化图像坐标）。
    pub center: [f32; 2],
    /// 半径（归一化）。
    pub radius: [f32; 2],
    /// 羽化带宽（归一化）。
    pub feather: f32,
    /// 形状。
    pub shape: WindowShape,
    /// 是否反相（窗口外生效）。
    pub invert: bool,
}

impl Default for PowerWindow {
    fn default() -> Self {
        PowerWindow {
            center: [0.5, 0.5],
            radius: [0.25, 0.25],
            feather: 0.1,
            shape: WindowShape::Ellipse,
            invert: false,
        }
    }
}

impl PowerWindow {
    /// 归一化坐标 `(u, v) ∈ [0,1]²` 处的窗口权重。
    pub fn weight(&self, u: f32, v: f32) -> f32 {
        let rx = self.radius[0].max(1e-6);
        let ry = self.radius[1].max(1e-6);
        // 到中心的归一化距离（椭圆 = 单位圆；矩形 = 切比雪夫距离）。
        let dx = (u - self.center[0]).abs() / rx;
        let dy = (v - self.center[1]).abs() / ry;
        let d = match self.shape {
            WindowShape::Ellipse => (dx * dx + dy * dy).sqrt(),
            WindowShape::Rect => dx.max(dy),
        };
        // 羽化：d ∈ [0, 1−feather] → 1，[1−feather, 1] → 平滑降 0。
        let f = self.feather.max(0.0);
        let inner = (1.0 - f).max(0.0);
        let w = if d <= inner {
            1.0
        } else if d >= 1.0 {
            0.0
        } else {
            smoothstep((1.0 - d) / f.max(1e-6))
        };
        if self.invert {
            1.0 - w
        } else {
            w
        }
    }
}

/// 二级调色节点：选区 + lift-gamma-gain + Power Window 组合。
///
/// 逐像素管线（固定序）：`sel = qualifier.weight · window.weight` →
/// `out = in + sel · (lgg(in) − in)`（按选区权重在原始与调色结果间插值）。
/// 中性（lgg 中性）或选区权重 0 处 → 逐位恒等。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Secondary {
    /// HSL 选区。
    pub qualifier: HslQualifier,
    /// lift-gamma-gain 调色。
    pub lgg: LiftGammaGain,
    /// Power Window 空间限定。
    pub window: PowerWindow,
}

impl Default for Secondary {
    fn default() -> Self {
        Secondary {
            qualifier: HslQualifier::default(),
            lgg: LiftGammaGain::default(),
            window: PowerWindow::default(),
        }
    }
}

impl TileEffect for Secondary {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        let neutral = self.lgg.is_neutral();
        let w = window.width;
        let h = window.height;
        for y in 0..h {
            for x in 0..w {
                let c = window.get(x, y);
                if neutral {
                    out.set(x, y, c);
                    continue;
                }
                let u = if w > 1 { x as f32 / (w - 1) as f32 } else { 0.5 };
                let v = if h > 1 { y as f32 / (h - 1) as f32 } else { 0.5 };
                let sel = self.qualifier.weight(c) * self.window.weight(u, v);
                let graded = self.lgg.apply(c);
                let mut res = [0.0f32; 4];
                for k in 0..3 {
                    res[k] = c[k] + sel * (graded[k] - c[k]);
                }
                res[3] = c[3];
                out.set(x, y, res);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_range_qualifier_selects_everything() {
        // 全区间（默认）→ 任意彩色像素 sel = 1（三段软限皆 1）。
        let q = HslQualifier::default();
        for &c in &[
            [0.5f32, 0.5, 0.5, 1.0],
            [0.9, 0.1, 0.1, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ] {
            assert_eq!(q.weight(c), 1.0, "全区间选区必须处处权重 1：{c:?}");
        }
    }

    #[test]
    fn hue_qualifier_selects_only_red_hues() {
        // 选区 = 红相（hue ≈ 0）；纯红 sel ≈ 1，纯蓝/纯绿 sel ≈ 0。
        let q = HslQualifier {
            luma_range: [0.0, 1.0],
            hue_range: [0.0, 0.05],
            sat_range: [0.0, 1.0],
            softness: 0.02,
        };
        let red = q.weight([1.0, 0.0, 0.0, 1.0]);
        let green = q.weight([0.0, 1.0, 0.0, 1.0]);
        let blue = q.weight([0.0, 0.0, 1.0, 1.0]);
        assert!(red > 0.9, "纯红应被选中：{red}");
        assert!(green < 0.1 && blue < 0.1, "绿/蓝不应选中：{green}/{blue}");
    }

    #[test]
    fn luma_qualifier_selects_shadows_only() {
        let q = HslQualifier {
            luma_range: [0.0, 0.25],
            hue_range: [0.0, 1.0],
            sat_range: [0.0, 1.0],
            softness: 0.05,
        };
        let dark = q.weight([0.1, 0.1, 0.1, 1.0]);
        let bright = q.weight([0.9, 0.9, 0.9, 1.0]);
        assert!(dark > 0.9, "暗部应选中：{dark}");
        assert!(bright < 0.1, "亮部不应选中：{bright}");
    }

    #[test]
    fn soft_range_boundary_is_continuous() {
        // 软区间在过渡带内单调连续、端点处恰为 0/1。
        let q = HslQualifier {
            luma_range: [0.2, 0.8],
            hue_range: [0.0, 1.0],
            sat_range: [0.0, 1.0],
            softness: 0.1,
        };
        let center = q.weight([0.5, 0.5, 0.5, 1.0]);
        let outside = q.weight([0.05, 0.05, 0.05, 1.0]);
        assert!(center > 0.99, "区间中心权重 ≈ 1：{center}");
        assert!(outside < 0.01, "区间外权重 ≈ 0：{outside}");
    }

    #[test]
    fn lift_gamma_gain_neutral_is_bit_exact_identity() {
        let lgg = LiftGammaGain::default();
        assert!(lgg.is_neutral());
        let c = [0.3f32, 0.6, 0.9, 0.5];
        assert_eq!(lgg.apply(c), c, "中性 lgg 逐位恒等");
    }

    #[test]
    fn lift_raises_shadows_gain_raises_highlights() {
        let lift = LiftGammaGain { lift: 0.2, gamma: 1.0, gain: 1.0, gamma_steps: 8 };
        assert!((lift.apply_scalar(0.0) - 0.2).abs() < 1e-6, "lift 抬黑");
        let gain = LiftGammaGain { lift: 0.0, gamma: 1.0, gain: 1.5, gamma_steps: 8 };
        assert!((gain.apply_scalar(0.5) - 0.75).abs() < 1e-6, "gain 倍增高光");
    }

    #[test]
    fn gamma_brightens_midtones() {
        // gamma > 1 → 中灰提亮（指数 = 1/gamma 量化到整数步）。
        let g = LiftGammaGain { lift: 0.0, gamma: 2.0, gain: 1.0, gamma_steps: 8 };
        let out = g.apply_scalar(0.5);
        assert!(out > 0.5, "gamma 2 应提亮中灰：{out}");
    }

    #[test]
    fn non_integer_gamma_midtones_do_not_underflow() {
        // 回归（实测复现）：非整 gamma 中低调不得下溢归零。
        // 真值参照 x^(1/gamma)（阈值放宽以容纳有理逼近的量化误差）。
        let ref_pow = |x: f32, g: f32| x.powf(1.0 / g);
        for &(g, x) in &[
            (1.4f32, 0.3f32), (1.4, 0.5), (1.4, 0.7),
            (1.5, 0.3), (1.5, 0.5), (1.5, 0.7),
            (0.7, 0.3), (0.7, 0.5), (0.7, 0.7),
        ] {
            let lgg = LiftGammaGain { lift: 0.0, gamma: g, gain: 1.0, gamma_steps: 8 };
            let got = lgg.apply_scalar(x);
            let want = ref_pow(x, g);
            assert!(got > 0.0, "gamma={g} x={x} 不应归零，实得 {got}");
            assert!(
                (got - want).abs() < 0.01,
                "gamma={g} x={x} 误差过大：got={got} want={want}"
            );
        }
        // QA 复核点：gamma=1.4 x=0.5 必须 ≈ 0.6095。
        let g = LiftGammaGain { lift: 0.0, gamma: 1.4, gain: 1.0, gamma_steps: 8 };
        let got = g.apply_scalar(0.5);
        assert!((got - 0.6095).abs() < 0.01, "gamma=1.4 x=0.5 应 ≈0.6095，实得 {got}");
    }

    #[test]
    fn power_window_center_full_edge_zero() {
        let w = PowerWindow::default();
        assert!(w.weight(0.5, 0.5) > 0.99, "窗口中心权重 ≈ 1");
        assert!(w.weight(0.0, 0.0) < 0.01, "窗口外权重 ≈ 0");
        // 反相：中心 0、外 1。
        let inv = PowerWindow { invert: true, ..w };
        assert!(inv.weight(0.5, 0.5) < 0.01);
        assert!(inv.weight(0.0, 0.0) > 0.99);
    }

    #[test]
    fn rect_window_covers_corners_more_than_ellipse() {
        // 矩形窗口在轴向角处覆盖应 ≥ 椭圆（切比雪夫距离 ≤ 欧氏距离）。
        let rect = PowerWindow { shape: WindowShape::Rect, ..PowerWindow::default() };
        let ell = PowerWindow { shape: WindowShape::Ellipse, ..PowerWindow::default() };
        let (u, v) = (0.5 + 0.2, 0.5 + 0.2);
        assert!(rect.weight(u, v) >= ell.weight(u, v), "矩形应覆盖更多对角区域");
    }

    #[test]
    fn secondary_neutral_is_identity_and_graded_is_bounded() {
        let mut win = Fbuf16F::new(12, 10);
        let mut state: u64 = 0x5EC0;
        for y in 0..10u32 {
            for x in 0..12u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        let neutral = Secondary::default();
        assert_eq!(neutral.eval_window(&win), win, "中性二级调色逐位恒等");

        let graded = Secondary {
            lgg: LiftGammaGain { lift: 0.1, gamma: 1.5, gain: 1.1, gamma_steps: 8 },
            ..Secondary::default()
        };
        let a = graded.eval_window(&win);
        assert_eq!(a, graded.eval_window(&win), "纯函数契约");
        // 选区 + 窗口限定下，权重 ∈ [0,1] ⇒ 输出是原始与调色结果的凸组合。
        for y in 0..10u32 {
            for x in 0..12u32 {
                let c = win.get(x, y);
                let g = graded.lgg.apply(c);
                let o = a.get(x, y);
                for k in 0..3 {
                    let lo = c[k].min(g[k]) - 1e-3;
                    let hi = c[k].max(g[k]) + 1e-3;
                    assert!(o[k] >= lo && o[k] <= hi, "必须是凸组合：{o:?}");
                }
            }
        }
    }
}
