//! 转场（U14 NLE 剪辑）—— **S2 卷积 / 混合特解**。
//!
//! 语义权威：架构总设计 v2.4 §2.2（`transition` = S9 过渡曲线 + S5 时间映射）、
//! 兼容性宪章对照表 #4（Transition / 转场，词汇沿用剪映/AE/iMovie）、白皮书
//! §18（合成/遮罩族，`merge` 混合）。
//!
//! # 数学路径
//!
//! 转场 = 两 Clip 输出在**重叠区间**内按进度 `p ∈ [0,1]` 做**混合**（S2）：
//!
//! - [`TransitionKind::Dissolve`]（交叉溶解）：`out = (1−p)·A + p·B`——凸组合，
//!   即两图在重叠区间的线性混合（S2 加权核的零相位特解）；
//! - [`TransitionKind::Fade`]（淡入淡出）：`A` 经 `1−p` 淡出、`B` 经 `p` 淡入，
//!   逐分量乘性斜坡后线性相加（与 [`crate::nle::audio`] 的淡入淡出同式）；
//! - [`TransitionKind::Wipe`]（擦除）：按归一化坐标 `u < p` 选 `B`、否则选 `A`
//!   ——硬边遮罩（S2 覆盖率硬阈特解；方向由 `direction` 指定）。
//!
//! 进度 `p` 可为线性，或经 [`prism_anim::EasingKind`] 缓动（确定：纯函数，
//! 无 libm 超越函数——见 `prism-anim/easing.rs`）。
//!
//! # 确定性
//!
//! 逐像素单遍、固定下标序遍历；混合仅走 `+ − × ÷`；同输入逐位一致。

use prism_anim::EasingKind;

use super::frame::Frame;

/// 擦除方向（硬边遮罩推进方向）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// 从左到右（`u = x / width`）。
    LeftToRight,
    /// 从右到左（`u = 1 − x / width`）。
    RightToLeft,
    /// 从上到下（`u = y / height`）。
    TopToBottom,
    /// 从下到上（`u = 1 − y / height`）。
    BottomToTop,
}

impl WipeDirection {
    /// 归一化推进坐标 `u ∈ [0,1]`（像素 `(x,y)` 对帧尺寸 `(w,h)`）。
    ///
    /// 边界：`w/h` 为 0 时返回 0（确定性兜底，不除零 panic）。
    pub fn progress_of(self, x: u32, y: u32, w: u32, h: u32) -> f32 {
        match self {
            Self::LeftToRight => {
                if w == 0 {
                    0.0
                } else {
                    x as f32 / w as f32
                }
            }
            Self::RightToLeft => {
                if w == 0 {
                    0.0
                } else {
                    1.0 - x as f32 / w as f32
                }
            }
            Self::TopToBottom => {
                if h == 0 {
                    0.0
                } else {
                    y as f32 / h as f32
                }
            }
            Self::BottomToTop => {
                if h == 0 {
                    0.0
                } else {
                    1.0 - y as f32 / h as f32
                }
            }
        }
    }
}

/// 转场种类（三种最小集 + 擦除方向）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionKind {
    /// 交叉溶解：`out = (1−p)·A + p·B`（凸组合）。
    Dissolve,
    /// 淡入淡出：`out = (1−p)·A + p·B`（逐分量乘性斜坡，与 Dissolve 在
    /// 线性进度下同式；语义区分供 DSL 命名与缓动默认值，RGB 与 alpha 同规则）。
    Fade,
    /// 擦除：硬边遮罩按方向推进（`u < p` 取 B、否则取 A）。
    Wipe(WipeDirection),
}

/// 缓动选择：线性或 [`EasingKind`]（确定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionEasing {
    /// 线性。
    Linear,
    /// 缓动曲线（`prism-anim` S9 分段多项式纯函数）。
    Ease(EasingKind),
}

impl TransitionEasing {
    /// 把原始进度 `raw ∈ [0,1]` 映射为缓动后进度（末值仍钳 [0,1]）。
    pub fn apply(self, raw: f32) -> f32 {
        let clamped = raw.clamp(0.0, 1.0);
        match self {
            Self::Linear => clamped,
            Self::Ease(kind) => kind.apply(clamped).clamp(0.0, 1.0),
        }
    }
}

/// 一个转场描述：种类 + 缓动 + 重叠区间（时间线秒）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transition {
    /// 转场种类。
    pub kind: TransitionKind,
    /// 缓动。
    pub easing: TransitionEasing,
    /// 重叠区间起点（时间线秒；含）。
    pub start: f32,
    /// 重叠区间时长（秒；`> 0`）。
    pub duration: f32,
}

impl Transition {
    /// 构造转场（`duration ≤ 0` 视为无效，调用方经 [`Self::is_valid`] 检查）。
    pub fn new(kind: TransitionKind, easing: TransitionEasing, start: f32, duration: f32) -> Self {
        Self { kind, easing, start, duration }
    }

    /// 是否有效（有限参数 + 正时长）。
    pub fn is_valid(&self) -> bool {
        self.start.is_finite() && self.duration.is_finite() && self.duration > 0.0
    }

    /// 重叠区间终点（时间线秒）。
    pub fn end(&self) -> f32 {
        self.start + self.duration
    }

    /// 时刻 `t` 是否落在重叠区间 `[start, end)` 内。
    pub fn covers(&self, t: f32) -> bool {
        t >= self.start && t < self.end()
    }

    /// 时刻 `t` 的（缓动后）进度 `p ∈ [0,1]`。
    pub fn progress_at(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        self.easing.apply((t - self.start) / self.duration)
    }

    /// 在重叠时刻 `t` 混合两帧输出（S2）。
    ///
    /// `a` 为下层（前段 Clip）、`b` 为上层（后段 Clip）。尺寸不一致时钳到
    /// 二者最小公共尺寸逐像素混合（不 panic）。空帧（尺寸 0）→ 直接返回另一方。
    pub fn blend(&self, t: f32, a: &Frame, b: &Frame) -> Frame {
        let p = self.progress_at(t);
        match self.kind {
            TransitionKind::Dissolve | TransitionKind::Fade => blend_lerp(a, b, p),
            TransitionKind::Wipe(dir) => blend_wipe(a, b, p, dir),
        }
    }
}

/// 凸组合混合：`out = (1−p)·a + p·b`（尺寸取最小公共尺寸）。
fn blend_lerp(a: &Frame, b: &Frame, p: f32) -> Frame {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let w = a.width.min(b.width);
    let h = a.height.min(b.height);
    let mut out = Frame::new(w, h);
    let ip = 1.0 - p;
    for y in 0..h {
        for x in 0..w {
            let ca = a.pixel(x.min(a.width - 1), y.min(a.height - 1));
            let cb = b.pixel(x.min(b.width - 1), y.min(b.height - 1));
            // 逐分量线性混合（固定序：R,G,B,A）。
            let mut c = [0.0_f32; 4];
            for i in 0..4 {
                c[i] = ip * ca[i] + p * cb[i];
            }
            out.set(x, y, c);
        }
    }
    out
}

/// 硬边擦除：`dir.progress_of(x,y) < p` 取 `b`，否则取 `a`。
fn blend_wipe(a: &Frame, b: &Frame, p: f32, dir: WipeDirection) -> Frame {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    let w = a.width.min(b.width);
    let h = a.height.min(b.height);
    let mut out = Frame::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let u = dir.progress_of(x, y, w, h);
            let c = if u < p {
                b.pixel(x.min(b.width - 1), y.min(b.height - 1))
            } else {
                a.pixel(x.min(a.width - 1), y.min(a.height - 1))
            };
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
    fn dissolve_midpoint_is_fifty_fifty() {
        let a = solid(2, 2, [1.0, 0.0, 0.0, 1.0]);
        let b = solid(2, 2, [0.0, 0.0, 1.0, 1.0]);
        let tr = Transition::new(
            TransitionKind::Dissolve,
            TransitionEasing::Linear,
            0.0,
            1.0,
        );
        let out = tr.blend(0.5, &a, &b);
        let c = out.pixel(0, 0);
        assert!((c[0] - 0.5).abs() < 1e-6 && (c[2] - 0.5).abs() < 1e-6, "中点各 50%: {c:?}");
    }

    #[test]
    fn dissolve_endpoints_select_layers() {
        let a = solid(1, 1, [1.0, 0.0, 0.0, 1.0]);
        let b = solid(1, 1, [0.0, 1.0, 0.0, 1.0]);
        let tr = Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.0, 1.0);
        assert_eq!(tr.blend(0.0, &a, &b).pixel(0, 0), [1.0, 0.0, 0.0, 1.0], "起点全 A");
        assert_eq!(tr.blend(1.0, &a, &b).pixel(0, 0), [0.0, 1.0, 0.0, 1.0], "终点全 B");
    }

    #[test]
    fn wipe_left_to_right_advances() {
        let a = solid(4, 1, [1.0, 0.0, 0.0, 1.0]);
        let b = solid(4, 1, [0.0, 1.0, 0.0, 1.0]);
        let tr = Transition::new(
            TransitionKind::Wipe(WipeDirection::LeftToRight),
            TransitionEasing::Linear,
            0.0,
            1.0,
        );
        // p=0.5：x=0,1 (u=0,0.25) 取 B；x=2,3 (u=0.5,0.75) 取 A。
        let out = tr.blend(0.5, &a, &b);
        assert_eq!(out.pixel(0, 0), [0.0, 1.0, 0.0, 1.0], "左半 B");
        assert_eq!(out.pixel(3, 0), [1.0, 0.0, 0.0, 1.0], "右半 A");
    }

    #[test]
    fn transition_blend_is_deterministic() {
        let a = solid(3, 2, [0.2, 0.4, 0.6, 1.0]);
        let b = solid(3, 2, [0.8, 0.1, 0.3, 0.5]);
        let tr = Transition::new(TransitionKind::Dissolve, TransitionEasing::Linear, 0.0, 1.0);
        let first = tr.blend(0.37, &a, &b);
        for _ in 0..100 {
            assert_eq!(tr.blend(0.37, &a, &b).pixels, first.pixels);
        }
    }
}
