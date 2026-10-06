//! 轨道合成（U14 NLE 剪辑）—— 多轨叠加的混合模式与不透明度（S2 特解）。
//!
//! # 数学路径
//!
//! 轨道叠加 = **逐分量线性合成**（S2 混合特解，零相位）：下层 `b` 与上层 `t` 在
//! 不透明度 `a ∈ [0,1]` 下按混合模式 `f(b, t)` 合成
//!
//! ```text
//! out = (1 − a)·b + a·f(b, t)
//! ```
//!
//! `a = 0` 恒等于下层、`a = 1` 恒等于上层（**合成族的单位元纪律**，与转场
//! 凸组合同源）。四种模式全部只走 `+ − ×`（无 libm、无分支 transcendent）：
//!
//! - [`BlendMode::Normal`]：`f = t`（普通覆盖，α 混合即上式 `(1−a)b + at`）；
//! - [`BlendMode::Add`]：`f = b + t`（加光，辉光/叠加发光层）；
//! - [`BlendMode::Multiply`]：`f = b·t`（正片叠底，压暗下层）；
//! - [`BlendMode::Screen`]：`f = b + t − b·t`（滤色，抬亮下层）。
//!
//! # 尺寸与越界
//!
//! 输出尺寸取 `max(下层, 上层)`；任一方尺寸不足时按**边缘钳位**采样（不缩放、
//! 不补零）——尺寸不一致是编排常态而非错误，静默编造尺寸会破坏确定性。
//! 空帧（尺寸 0）作合成单位元：上层空 → 原样返回下层；下层空 → 以全零为下层
//! 按上式合成（`Normal` 得 `a·t`，即透明度直接调制）。
//!
//! # 确定性
//!
//! 逐像素单遍、固定下标序（先 y 后 x、先 R 后 G 后 B 后 A）；合成式括号固定，
//! 故同输入逐位一致。

use super::frame::Frame;

/// 混合模式（四种，只走 `+ − ×`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// 普通覆盖：`f(b, t) = t`。
    Normal,
    /// 加光：`f(b, t) = b + t`。
    Add,
    /// 正片叠底：`f(b, t) = b · t`。
    Multiply,
    /// 滤色：`f(b, t) = b + t − b · t`。
    Screen,
}

impl BlendMode {
    /// 混合模式的分量作用（`b` = 下层分量，`t` = 上层分量）。
    #[inline]
    pub fn apply(self, b: f32, t: f32) -> f32 {
        match self {
            Self::Normal => t,
            Self::Add => b + t,
            Self::Multiply => b * t,
            Self::Screen => b + t - b * t,
        }
    }
}

/// Clip / 轨道的变换参数（不透明度 + 混合模式）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// 不透明度 `a ∈ [0,1]`（构造时钳位；`0` = 完全透明、`1` = 完全不透明）。
    pub opacity: f32,
    /// 混合模式。
    pub blend: BlendMode,
}

impl Default for Transform {
    fn default() -> Self {
        Self { opacity: 1.0, blend: BlendMode::Normal }
    }
}

impl Transform {
    /// 构造变换（`opacity` 钳到 `[0,1]`，非有限取 `1.0`）。
    pub fn new(opacity: f32, blend: BlendMode) -> Self {
        let a = if opacity.is_finite() { opacity.clamp(0.0, 1.0) } else { 1.0 };
        Self { opacity: a, blend }
    }

    /// 恒等变换（不透明 + 普通覆盖）。
    pub fn identity() -> Self {
        Self { opacity: 1.0, blend: BlendMode::Normal }
    }
}

/// 边缘钳位采样：把 `(x, y)` 钳进帧尺寸后取像素（尺寸 0 → 全零）。
#[inline]
fn sample_clamped(f: &Frame, x: u32, y: u32, w: u32, h: u32) -> [f32; 4] {
    if f.is_empty() || w == 0 || h == 0 {
        return [0.0; 4];
    }
    let cx = x.min(w - 1).min(f.width.saturating_sub(1));
    let cy = y.min(h - 1).min(f.height.saturating_sub(1));
    f.pixel(cx, cy)
}

/// 多轨合成：`out = (1 − a)·b + a·f(b, t)`。
///
/// 尺寸取 `max`；越界按边缘钳位采样；空帧作单位元（见模块文档）。
pub fn composite(base: &Frame, top: &Frame, transform: Transform) -> Frame {
    if top.is_empty() {
        return base.clone();
    }
    let (w, h) = (base.width.max(top.width), base.height.max(top.height));
    if w == 0 || h == 0 {
        return Frame::new(0, 0);
    }
    let a = transform.opacity;
    let ia = 1.0 - a;
    let mode = transform.blend;
    let mut out = Frame::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let cb = sample_clamped(base, x, y, w, h);
            let ct = sample_clamped(top, x, y, w, h);
            let mut c = [0.0_f32; 4];
            // 固定分量序 R,G,B,A —— 括号固定保证结合序确定。
            for i in 0..4 {
                c[i] = ia * cb[i] + a * mode.apply(cb[i], ct[i]);
            }
            out.set(x, y, c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Frame {
        let mut f = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                f.set(x, y, c);
            }
        }
        f
    }

    #[test]
    fn opacity_zero_is_base_and_one_is_top() {
        let b = solid(2, 2, [0.2, 0.4, 0.6, 1.0]);
        let t = solid(2, 2, [0.8, 0.1, 0.3, 0.5]);
        let lo = composite(&b, &t, Transform::new(0.0, BlendMode::Normal));
        assert_eq!(lo.pixels, b.pixels, "透明度 0 恒等于下层（逐位）");
        let hi = composite(&b, &t, Transform::new(1.0, BlendMode::Normal));
        assert_eq!(hi.pixels, t.pixels, "透明度 1 恒等于上层（逐位）");
    }

    #[test]
    fn opacity_half_is_convex_combination() {
        let b = solid(1, 1, [0.0, 0.0, 0.0, 1.0]);
        let t = solid(1, 1, [1.0, 1.0, 1.0, 1.0]);
        let out = composite(&b, &t, Transform::new(0.5, BlendMode::Normal));
        let c = out.pixel(0, 0);
        assert!((c[0] - 0.5).abs() < 1e-6, "半透明各半：{c:?}");
    }

    #[test]
    fn blend_modes_operate_componentwise() {
        let b = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let t = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let full = |m| composite(&b, &t, Transform::new(1.0, m)).pixel(0, 0)[0];
        assert!((full(BlendMode::Normal) - 0.5).abs() < 1e-6);
        assert!((full(BlendMode::Add) - 1.0).abs() < 1e-6, "0.5+0.5");
        assert!((full(BlendMode::Multiply) - 0.25).abs() < 1e-6, "0.5·0.5");
        assert!((full(BlendMode::Screen) - 0.75).abs() < 1e-6, "0.5+0.5−0.25");
    }

    #[test]
    fn empty_frames_are_composite_identity() {
        let t = solid(2, 2, [1.0, 1.0, 1.0, 1.0]);
        let empty = Frame::new(0, 0);
        // 上层空 → 原样返回下层。
        let b = solid(2, 2, [0.3; 4]);
        assert_eq!(composite(&b, &empty, Transform::identity()).pixels, b.pixels);
        // 下层空 → 以零为下层，Normal 下透明度直接调制上层。
        let out = composite(&empty, &t, Transform::new(0.25, BlendMode::Normal));
        assert!((out.pixel(0, 0)[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn size_mismatch_takes_max_and_clamps() {
        let small = solid(1, 1, [1.0, 0.0, 0.0, 1.0]);
        let big = solid(3, 2, [0.0, 0.0, 1.0, 1.0]);
        // 半透明才能看出下层透出（opacity=1 时 Normal 恒等于上层）。
        let out = composite(&small, &big, Transform::new(0.5, BlendMode::Normal));
        assert_eq!((out.width, out.height), (3, 2), "尺寸取 max");
        // (2,1) 处下层只有 (0,0) 一个像素 ⟹ 边缘钳位采样得红，非补零。
        let c = out.pixel(2, 1);
        assert!((c[0] - 0.5).abs() < 1e-6, "边缘钳位而非补零：{c:?}");
        assert!((c[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn composite_is_bitwise_deterministic() {
        let b = solid(3, 2, [0.1, 0.2, 0.3, 0.4]);
        let t = solid(3, 2, [0.7, 0.6, 0.5, 0.9]);
        let tr = Transform::new(0.3, BlendMode::Screen);
        let first = composite(&b, &t, tr);
        for _ in 0..200 {
            let got = composite(&b, &t, tr);
            assert_eq!(
                got.pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                first.pixels.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "合成必须逐位可复现"
            );
        }
    }
}
