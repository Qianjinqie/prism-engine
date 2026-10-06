//! stroke reveal 文字（路线图 B15；白皮书 §16 文字管线 + §11 动画）。
//!
//! # 定位
//!
//! "手写/描边逐步显现"效果：把整段文本视作一条**书写笔迹**（沿字形基线
//! 逐个推进），随时间 `t` 由 [`prism_anim::Track`] 给出**揭示进度** `p ∈ [0,1]`，
//! 再按 `p` 决定每个字形描到哪一段——已揭示部分完整绘制，临界处描边被
//! **裁剪**到部分笔画，未揭示部分不绘制。
//!
//! 与 [`crate::layout`]/[`crate::path_text`] 正交：字形的**落位**由布局/路径
//! 排字给出（本模块消费 `advance` 序列），本模块只负责"揭示到哪"。
//!
//! # 模型（D-21：解析进度）
//!
//! 设字形 `i` 的起始/终止归一化进度为 `a_i`、`b_i`（由累计 advance 除以总长
//! 得到，`a_i < b_i`），揭示进度 `p`：
//!
//! ```text
//!   ink_i(p) = clamp((p − a_i) / (b_i − a_i), 0, 1)
//! ```
//!
//! `ink_i = 1` 完整显现、`0 < ink_i < 1` 部分、`0` 未显现。**进度单调**
//! （`p` 增 → `ink_i` 单调不减），这是本模块的核心不变量（含测试断言）。
//!
//! # 确定性契约
//! 纯函数；进度只经 `Track::sample`（t 的纯函数）；浮点仅 `+ − × ÷`。

use prism_anim::Track;

/// 单字形的揭示状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevealGlyph {
    /// 字形序号（阅读序）。
    pub index: usize,
    /// 归一化起始进度 `a_i`（[0,1]）。
    pub start: f32,
    /// 归一化终止进度 `b_i`（[0,1]）。
    pub end: f32,
    /// 揭示墨量 `ink ∈ [0,1]`（1 = 完整显现）。
    pub ink: f32,
}

impl RevealGlyph {
    /// 是否此刻已开始显现（`ink > 0`）。
    pub fn is_visible(&self) -> bool {
        self.ink > 0.0
    }

    /// 是否完整显现（`ink == 1`）。
    pub fn is_complete(&self) -> bool {
        self.ink >= 1.0
    }
}

/// 揭示计划：由字形 advance 序列构建的进度分段（不含时间）。
#[derive(Debug, Clone, PartialEq)]
pub struct RevealPlan {
    /// 每字形的归一化区间（阅读序）。
    pub spans: Vec<(f32, f32)>,
}

impl RevealPlan {
    /// 由每个字形沿书写路径的步进宽 `advances` 构建计划。
    ///
    /// 字形 `i` 占据累计总长的 `[a_i, b_i)`，归一化到 `[0,1]`。
    /// 空输入 → 空计划；总长为 0（全零步进）→ 各段退化在 0，`ink` 恒 0。
    pub fn from_advances(advances: &[f32]) -> Self {
        let total: f32 = advances.iter().sum();
        let mut spans = Vec::with_capacity(advances.len());
        if total <= 1e-12 {
            spans.resize(advances.len(), (0.0, 0.0));
            return RevealPlan { spans };
        }
        let mut acc = 0.0f32;
        for &a in advances {
            let start = acc / total;
            acc += a;
            let end = acc / total;
            spans.push((start, end));
        }
        RevealPlan { spans }
    }

    /// 字形数量。
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// 是否为空计划。
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// 给定归一化揭示进度 `p`，求全部字形墨量（纯函数）。
    pub fn ink_at(&self, p: f32) -> Vec<RevealGlyph> {
        self.spans
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                let span = b - a;
                let ink = if span <= 1e-12 {
                    if p >= b {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    ((p - a) / span).clamp(0.0, 1.0)
                };
                RevealGlyph { index: i, start: a, end: b, ink }
            })
            .collect()
    }
}

/// stroke reveal 通道：揭示计划 + 时间 → 墨量的确定性求值器。
///
/// `track` 给出时间 `t` 上的**揭示进度** `p ∈ [0,1]`（通常是一条从 0 到 1 的
/// 关键帧轨道，可带缓动/回退——本模块不强制单调于 `t`，只保证 `ink` 对 `p`
/// 单调；若要"只显不回"，请让 `track` 单调）。
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeReveal {
    /// 揭示计划。
    pub plan: RevealPlan,
    /// 进度轨道（`sample(t)` → p）。
    pub track: Track,
}

impl StrokeReveal {
    /// 构造 stroke reveal（计划 + 进度轨道）。
    pub fn new(plan: RevealPlan, track: Track) -> Self {
        Self { plan, track }
    }

    /// 便利构造：线性从 0 揭示到 1，时长 `duration` 秒。
    pub fn linear(advances: &[f32], duration: f32) -> Self {
        use prism_anim::{EasingKind, Keyframe};
        let track = Track::new(vec![
            Keyframe::new(0.0, 0.0, EasingKind::Linear),
            Keyframe::new(duration.max(1e-6), 1.0, EasingKind::Linear),
        ]);
        Self { plan: RevealPlan::from_advances(advances), track }
    }

    /// 时刻 `t` 的全部字形墨量（`p = track.sample(t)`）。
    pub fn glyphs_at(&self, t: f32) -> Vec<RevealGlyph> {
        let p = self.track.sample(t).clamp(0.0, 1.0);
        self.plan.ink_at(p)
    }

    /// 时刻 `t` 的整体揭示进度 `p`（钳到 [0,1]）。
    pub fn progress_at(&self, t: f32) -> f32 {
        self.track.sample(t).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 8：reveal 进度单调）
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn walks() -> [f32; 5] {
        [10.0, 10.0, 10.0, 10.0, 10.0]
    }

    /// 计划分段：5 字等步进 → 每段宽 0.2，累计到 1。
    #[test]
    fn plan_spans_partition_unit() {
        let plan = RevealPlan::from_advances(&walks());
        assert_eq!(plan.len(), 5);
        for (i, &(a, b)) in plan.spans.iter().enumerate() {
            assert!((a - i as f32 * 0.2).abs() < 1e-4);
            assert!((b - (i + 1) as f32 * 0.2).abs() < 1e-4);
        }
        // 末端合到 1。
        assert!((plan.spans[4].1 - 1.0).abs() < 1e-6);
    }

    /// ink 对 p 单调不减（关卡 8 核心）：p 从 0 增到 1，逐字形墨量单调。
    #[test]
    fn ink_is_monotone_in_progress() {
        let plan = RevealPlan::from_advances(&walks());
        let mut prev = plan.ink_at(0.0);
        let steps = 100;
        for k in 1..=steps {
            let p = k as f32 / steps as f32;
            let cur = plan.ink_at(p);
            for (a, b) in prev.iter().zip(cur.iter()) {
                assert!(
                    b.ink >= a.ink - 1e-6,
                    "墨量应随 p 单调不减：p={p} glyph {} {} → {}",
                    b.index,
                    a.ink,
                    b.ink
                );
            }
            prev = cur;
        }
        // p=1 → 全部完整。
        assert!(prev.iter().all(|g| g.is_complete()));
    }

    /// 边界语义：p=0 全未见；p=0.5 前两字半透/前字形完整、后续未见。
    #[test]
    fn ink_boundaries() {
        let plan = RevealPlan::from_advances(&walks());
        assert!(plan.ink_at(0.0).iter().all(|g| g.ink == 0.0));
        let half = plan.ink_at(0.5);
        // 第 0、1 字完整（span [0,0.2],[0.2,0.4]）；第 2 字（[0.4,0.6]）半透；3、4 未见。
        assert!(half[0].is_complete());
        assert!(half[1].is_complete());
        assert!((half[2].ink - 0.5).abs() < 1e-4, "第 2 字应半透，实得 {}", half[2].ink);
        assert_eq!(half[3].ink, 0.0);
        assert_eq!(half[4].ink, 0.0);
    }

    /// track 驱动的确定性：同 t 同结果逐位一致；progress 钳到 [0,1]。
    #[test]
    fn track_driven_deterministic() {
        let sr = StrokeReveal::linear(&walks(), 2.0);
        for t in [0.0_f32, 0.5, 1.0, 2.0, 5.0] {
            let a = sr.glyphs_at(t);
            for _ in 0..500 {
                let b = sr.glyphs_at(t);
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(x.ink.to_bits(), y.ink.to_bits());
                }
            }
        }
        // t=0 → p=0；t=2 → p=1；越界钳制。
        assert_eq!(sr.progress_at(0.0), 0.0);
        assert_eq!(sr.progress_at(2.0), 1.0);
        assert_eq!(sr.progress_at(10.0), 1.0);
    }

    /// 空输入退化：无字形、无 panic。
    #[test]
    fn empty_plan() {
        let plan = RevealPlan::from_advances(&[]);
        assert!(plan.is_empty());
        assert!(plan.ink_at(0.7).is_empty());
    }

    /// 全零步进退化：总长 0，ink 在 p≥0 时对退化为 [0,0] 的字形立即完整。
    #[test]
    fn zero_advances_degenerate() {
        let plan = RevealPlan::from_advances(&[0.0, 0.0]);
        // spans 均为 (0,0)；ink_at(0) → 因 p≥b 全 1。
        let ink = plan.ink_at(0.0);
        assert_eq!(ink.len(), 2);
    }
}
