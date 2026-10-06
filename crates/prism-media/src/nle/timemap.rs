//! 时间映射（U14 NLE 剪辑）—— **S5 重参数化特解**。
//!
//! 语义权威：白皮书 §19（视频时间映射：纯函数映射 + 帧间混合）、架构总设计
//! v2.4 S5 行（重参数化：变量替换 + 方程求根；`prism-media` 宿主）。本文件把
//! 既有 [`crate::timing::TimeMap`]（素材内归一化映射）**提升**为 NLE 编辑语义：
//!
//! ```text
//! 时间线时间 t_tl  --(本模块 Trim/Speed/Reverse 关键帧速率)-->  源内时间 t_src
//! ```
//!
//! # 与 S5 既有结构对齐
//!
//! [`crate::timing::TimeMap`] 解决的是"素材内时间 → 归一化 `u'`（+ 帧间混合
//! 权重）"，是 S5 的一支；NLE 裁剪需要的是"**时间线区间 ↔ 素材区间**"的
//! 仿射重参数化（trim in/out + 恒定变速 + 倒放）与"**关键帧速率**"的重参数化
//! （τ(t) = ∫k ds，S9 可积，与既有 `TimingMode::SpeedCurve` 同构）。本模块因此：
//!
//! 1. **裁掉**用仿射映射表达：`t_src = src_in + (t_tl − tl_in) · speed`（倒放
//!    时再加镜像 `t_src = src_out − (t_tl − tl_in) · speed`）——映射为**严格
//!    单调仿射**，天然可逆（S5 局部逆 = 解析逆，无需牛顿迭代，因为次数 1）；
//! 2. **变速关键帧**用 [`RateCurve`] 表达 k(t)，映射 `τ(t_tl) = ∫ k(s) ds` 走
//!    **分段线性闭式积分**（梯形法精确，S9 可积）——与既有
//!    `TimingMode::SpeedCurve` 同式，保证映射单调、seek 纯函数；
//! 3. 输出源内时间后，交由 S5 的素材级 [`TimeMap`] 落帧（帧索引 + 混合权重），
//!    音画共享同一 `f(t)`（白皮书 §19 超越点）。
//!
//! # 退化点结构化错误（C18 永不 panic）
//!
//! 零长度区间、负速率、非有限参数 → 返回 [`NleError`]（三段式），不 panic。
//!
//! # 确定性
//!
//! 纯函数、无状态、无 libm、无第三方；只走 `+ − × ÷` 与 `clamp`，同输入跨平台
//! 逐位一致。

use crate::timing::{TimeMap, TimingMode};

use super::{NleError, NleResult};

/// 变速曲线：时间线时间 → 恒定或关键帧速率（倍率）。
///
/// - [`RateCurve::Constant`]：恒定倍率（`k > 0`）；
/// - [`RateCurve::Keyframed`]：关键帧速率 `(时间线秒, 倍率)` 采样点，按时间升序，
///   线性插值；区间外取最近端点（阶跃保持）。倍率经 `clamp(·, ε, ∞)` 保证映射
///   单调（同 [`crate::timing::TimingMode::SpeedCurve`] 的整数积分纪律）。
#[derive(Debug, Clone, PartialEq)]
pub enum RateCurve {
    /// 恒定倍率（`> 0` 为正常速度，`= 1.0` 为原速）。
    Constant(f32),
    /// 关键帧速率：`(时间线秒, 倍率)` 采样点，按时间升序，线性插值。
    Keyframed(Vec<(f32, f32)>),
}

impl RateCurve {
    /// 关键帧速率在时间线时刻 `t`（相对该 clip 的时间线起点）的倍率。
    ///
    /// 恒定 → 直接返回；关键帧 → 分段线性插值；越界延伸端点值。全路径
    /// `clamp(·, 0, ∞)`，映射单调不减（速率非负）。
    pub fn rate_at(&self, t: f32) -> f32 {
        match self {
            Self::Constant(k) => k.max(0.0),
            Self::Keyframed(samples) => {
                if samples.is_empty() {
                    return 1.0;
                }
                if t <= samples[0].0 {
                    return samples[0].1.max(0.0);
                }
                for w in samples.windows(2) {
                    let (t0, k0) = w[0];
                    let (t1, k1) = w[1];
                    if t <= t1 {
                        let span = t1 - t0;
                        if span <= 0.0 {
                            return k1.max(0.0);
                        }
                        let a = (t - t0) / span;
                        return (k0 + (k1 - k0) * a).max(0.0);
                    }
                }
                samples[samples.len() - 1].1.max(0.0)
            }
        }
    }

    /// `∫₀ᵗ rate(s) ds` 的分段线性闭式积分（梯形法精确）。
    ///
    /// 与 [`crate::timing::TimingMode::SpeedCurve`] 的整数积分同式——倍率非负
    /// 保证积分单调不减。
    pub fn integral(&self, t: f32) -> f32 {
        match self {
            Self::Constant(k) => k.max(0.0) * t.max(0.0),
            Self::Keyframed(samples) => integral_curve(samples, t),
        }
    }
}

/// 分段线性闭式积分 `∫₀ᵗ clamp(k(s), 0, ∞) ds`（梯形法精确）。
///
/// 采样点按时间升序；区间外取端点倍率。与 `timing.rs` 的 `integral_curve`
/// 同式（此处独立实现以避免跨模块私有耦合——Mini Code：两处皆 ~30 行）。
fn integral_curve(samples: &[(f32, f32)], t: f32) -> f32 {
    if samples.is_empty() {
        return t.max(0.0); // 无采样 = 速率 1（恒等映射）
    }
    if t <= samples[0].0 {
        return samples[0].1.max(0.0) * t.max(0.0);
    }
    let mut acc = 0.0_f32;
    let first_t = samples[0].0;
    if first_t > 0.0 {
        acc += samples[0].1.max(0.0) * first_t;
    }
    for w in samples.windows(2) {
        let (t0, k0) = w[0];
        let (t1, k1) = w[1];
        let (k0n, k1n) = (k0.max(0.0), k1.max(0.0));
        if t >= t1 {
            acc += (k0n + k1n) * 0.5 * (t1 - t0);
        } else {
            let span = t1 - t0;
            if span <= 0.0 {
                return acc;
            }
            let a = (t - t0) / span;
            let kt = k0n + (k1n - k0n) * a;
            acc += (k0n + kt) * 0.5 * (t - t0);
            return acc;
        }
    }
    let last = samples[samples.len() - 1];
    acc += last.1.max(0.0) * (t - last.0).max(0.0);
    acc
}

/// 单调时间映射：时间线时间 ↔ 素材源时间（S5 重参数化特解）。
///
/// # 映射公式
///
/// 设时间线区间 `[tl_in, tl_in + tl_dur)`、源区间 `[src_in, src_out)`、
/// 速率曲线 `k(·)`（关键帧情况下为时间线相对时刻的函数）。
///
/// - **正放**（`speed > 0`）：
///   `t_src = src_in + τ(t_tl − tl_in)`，其中 `τ = ∫ k ds`；恒定速率下
///   `τ(t) = k·t`，退化为仿射 `t_src = src_in + k·(t_tl − tl_in)`。
/// - **倒放**（`reverse == true`）：镜像
///   `t_src = src_out − τ(t_tl − tl_in)`，首尾互换。
///
/// 速率非负 ⟹ `τ` 单调不减 ⟹ 映射单调（严格，除非速率为 0）；仿射特解在
/// 速率恒定时为**严格单调仿射**，逆映射即解析解（S5 局部逆退化为恒等，因
/// 次数 = 1）。
#[derive(Debug, Clone, PartialEq)]
pub struct TimeMap2 {
    /// 素材源时长（秒），`> 0`。
    src_dur: f32,
    /// 源区间入点（秒），`0 ≤ src_in < src_out ≤ src_dur`。
    src_in: f32,
    /// 源区间出点（秒）。
    src_out: f32,
    /// 时间线区间入点（秒）。
    tl_in: f32,
    /// 时间线区间时长（秒），`> 0`。
    tl_dur: f32,
    /// 速率曲线（恒定或关键帧）。
    rate: RateCurve,
    /// 是否倒放。
    reverse: bool,
}

impl TimeMap2 {
    /// 构造一个时间映射（trim in/out + 变速 + 倒放）。
    ///
    /// `src_in`/`src_out` 为源素材内裁剪区间（秒），`tl_in`/`tl_dur` 为时间线
    /// 落点。任意退化点（非有限、`src_dur ≤ 0`、`src_out ≤ src_in`、`tl_dur ≤ 0`、
    /// 恒定速率 ≤ 0）返回 [`NleError`]（三段式，C18 永不 panic）。
    pub fn new(
        src_dur: f32,
        src_in: f32,
        src_out: f32,
        tl_in: f32,
        tl_dur: f32,
        rate: RateCurve,
        reverse: bool,
    ) -> NleResult<Self> {
        if !src_dur.is_finite() || src_dur <= 0.0 {
            return Err(NleError::new(
                "NLE1001",
                "nle::TimeMap2::new",
                format!("源时长非法：src_dur={src_dur}"),
                "提供 src_dur > 0 的有限值",
            ));
        }
        if !src_in.is_finite() || !src_out.is_finite() || src_out <= src_in {
            return Err(NleError::new(
                "NLE1002",
                "nle::TimeMap2::new",
                format!("源裁剪区间非法：src_in={src_in} src_out={src_out}（要求 src_in < src_out）"),
                "确认 src_in < src_out 且均为有限值",
            ));
        }
        if src_in < 0.0 || src_out > src_dur {
            return Err(NleError::new(
                "NLE1002",
                "nle::TimeMap2::new",
                format!("源裁剪区间越出素材 [0, {src_dur}]：src_in={src_in} src_out={src_out}"),
                "把 src_in/src_out 钳进 [0, src_dur]",
            ));
        }
        if !tl_in.is_finite() || !tl_dur.is_finite() || tl_dur <= 0.0 {
            return Err(NleError::new(
                "NLE1003",
                "nle::TimeMap2::new",
                format!("时间线区间非法：tl_in={tl_in} tl_dur={tl_dur}（要求 tl_dur > 0）"),
                "提供 tl_dur > 0 的有限值",
            ));
        }
        if let RateCurve::Constant(k) = rate {
            if !k.is_finite() || k <= 0.0 {
                return Err(NleError::new(
                    "NLE1004",
                    "nle::TimeMap2::new",
                    format!("恒定速率非法：k={k}（要求 0 < k < ∞）"),
                    "提供正的有限速率（零速率不可逆，退化为退化映射）",
                ));
            }
        }
        Ok(Self { src_dur, src_in, src_out, tl_in, tl_dur, rate, reverse })
    }

    /// 便捷构造：全片直通（源全区间、时间线起点 0、原速、正放）。
    pub fn direct(src_dur: f32, fps: f32) -> NleResult<Self> {
        let _ = fps; // fps 由素材级 TimeMap 持有，此处仅表意
        Self::new(src_dur, 0.0, src_dur, 0.0, src_dur, RateCurve::Constant(1.0), false)
    }

    /// 源素材时长（秒）。
    pub fn source_duration(&self) -> f32 {
        self.src_dur
    }

    /// 源裁剪入点（秒）。
    pub fn src_in(&self) -> f32 {
        self.src_in
    }

    /// 源裁剪出点（秒）。
    pub fn src_out(&self) -> f32 {
        self.src_out
    }

    /// 时间线落点（秒）。
    pub fn timeline_in(&self) -> f32 {
        self.tl_in
    }

    /// 该映射占据的时间线时长（秒）。
    pub fn timeline_duration(&self) -> f32 {
        self.tl_dur
    }

    /// 速率曲线引用。
    pub fn rate(&self) -> &RateCurve {
        &self.rate
    }

    /// 是否倒放。
    pub fn is_reversed(&self) -> bool {
        self.reverse
    }

    /// 时间线时刻 `t_tl` → 素材源时间（秒），钳在 `[src_in, src_out]`。
    ///
    /// 时间线区间外按端点钳制（不越出源区间）；速率非负 ⟹ 单调。
    pub fn source_at(&self, t_tl: f32) -> f32 {
        if !t_tl.is_finite() {
            // 非有限时间：钳到区间入点（确定性兜底，不传播 NaN）。
            return if self.reverse { self.src_out } else { self.src_in };
        }
        let local = (t_tl - self.tl_in).clamp(0.0, self.tl_dur);
        let travelled = self.rate.integral(local);
        let span = self.src_out - self.src_in;
        // 积分到达时长端点的行进量应恰为 span（恒定速率下恒等；关键帧下由
        // 调用方保证 ∫₀^tl_dur k = span 的语义宽松——此处再钳一次保证边界）。
        let travelled = travelled.clamp(0.0, span.max(0.0));
        if self.reverse {
            self.src_out - travelled
        } else {
            self.src_in + travelled
        }
    }

    /// 素材源时间 → 时间线时刻（`source_at` 的解析逆；严格单调映射下唯一）。
    ///
    /// 源时间在裁剪区间外时，返回值钳到 `[tl_in, tl_in + tl_dur]`。
    /// 速率曲线在区间内单调不减 ⟹ 逆存在且唯一。
    pub fn timeline_at(&self, t_src: f32) -> f32 {
        let span = self.src_out - self.src_in;
        if span <= 0.0 {
            return self.tl_in;
        }
        // 源内行进量（先按方向换算为"从入点出发的正向行进"）。
        let travelled = if self.reverse {
            (self.src_out - t_src).clamp(0.0, span)
        } else {
            (t_src - self.src_in).clamp(0.0, span)
        };
        // 反解 τ(t) = travelled。恒定速率 → 解析；关键帧 → 分段线性反查。
        let local = self.invert_integral(travelled);
        self.tl_in + local.clamp(0.0, self.tl_dur)
    }

    /// 反解 `τ(t) = target` 的时间线相对时刻（分段线性、单调）。
    fn invert_integral(&self, target: f32) -> f32 {
        match &self.rate {
            RateCurve::Constant(k) => {
                let k = k.max(f32::MIN_POSITIVE);
                target / k
            }
            RateCurve::Keyframed(samples) => invert_curve(samples, target),
        }
    }

    /// 落帧：把时间线时刻映射为**素材级 S5 [`TimeMap`]** 的归一化输出。
    ///
    /// 输出 `(归一化 u, 帧索引, 帧间混合权重)` 三元组——`u = t_src / src_dur`
    /// 交给素材级 [`TimeMap`]（`TimingMode::Direct`）落帧，音画共享同一 `f(t)`。
    pub fn frame_at(&self, t_tl: f32, fps: f32) -> (f32, u32, f32) {
        let t_src = self.source_at(t_tl);
        let u = (t_src / self.src_dur.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0);
        // 用素材级 S5 TimeMap 落帧（与 timing.rs 同一落帧契约，避免重复实现）。
        let map = TimeMap::new(self.src_dur, fps, TimingMode::Direct);
        let idx = map.frame_at(t_src);
        let w = map.blend_at(t_src);
        (u, idx, w)
    }
}

/// 分段线性单调函数 `τ(t)` 的反查：给定积分值 `target`，求时间线时刻。
///
/// 与 [`integral_curve`] 互逆（梯形法端点一致）。样本按时间升序；越界延伸端点
/// 速率外推。
fn invert_curve(samples: &[(f32, f32)], target: f32) -> f32 {
    if samples.is_empty() {
        return target.max(0.0); // 速率 1
    }
    if target <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    let first_t = samples[0].0;
    if first_t > 0.0 {
        let k0 = samples[0].1.max(0.0);
        let seg = k0 * first_t;
        if target <= seg {
            return if k0 > 0.0 { target / k0 } else { 0.0 };
        }
        acc += seg;
    }
    for w in samples.windows(2) {
        let (t0, k0) = w[0];
        let (t1, k1) = w[1];
        let (k0n, k1n) = (k0.max(0.0), k1.max(0.0));
        let seg = (k0n + k1n) * 0.5 * (t1 - t0);
        if target <= acc + seg {
            // 段内反解：τ(u) = k0n·u + (k1n−k0n)/(2·span)·u² —— 对 u 求根。
            let span = t1 - t0;
            if span <= 0.0 {
                return t0;
            }
            let rem = target - acc;
            let dk = k1n - k0n;
            let u = if dk.abs() < f32::MIN_POSITIVE {
                // 常速段：线性。
                if k0n > 0.0 {
                    rem / k0n
                } else {
                    0.0
                }
            } else {
                // 解 (dk/(2·span))·u² + k0n·u − rem = 0 的正根。
                let a = dk / (2.0 * span);
                // 判别式 ≥ 0（因目标在段内）；用稳定求根。
                let disc = (k0n * k0n + 4.0 * a * rem).max(0.0);
                let sq = disc.sqrt();
                // 取使 u ∈ [0, span] 的根。
                if a > 0.0 {
                    (-k0n + sq) / (2.0 * a)
                } else {
                    // a < 0 时 τ 上凸但仍在 [0, span] 单调；取较大根（正）。
                    let denom = 2.0 * a;
                    let r1 = (-k0n - sq) / denom;
                    let r2 = (-k0n + sq) / denom;
                    if r1 > 0.0 {
                        r1
                    } else {
                        r2
                    }
                }
            };
            return t0 + u.clamp(0.0, span);
        }
        acc += seg;
    }
    // 超出末段：以末点速率外推。
    let last = samples[samples.len() - 1];
    let k = last.1.max(f32::MIN_POSITIVE);
    last.0 + (target - acc).max(0.0) / k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_map_is_identity() {
        let m = TimeMap2::direct(4.0, 30.0).unwrap();
        assert!((m.source_at(0.0) - 0.0).abs() < 1e-6);
        assert!((m.source_at(2.0) - 2.0).abs() < 1e-6);
        assert!((m.source_at(4.0) - 4.0).abs() < 1e-6, "端点");
        // 逆映射往返。
        for i in 0..=40 {
            let t = i as f32 * 0.1;
            let back = m.timeline_at(m.source_at(t));
            assert!((back - t).abs() < 1e-4, "往返不一致 t={t} back={back}");
        }
    }

    #[test]
    fn trim_in_out_offsets_mapping() {
        // 源 10s，裁 [2, 6)、落时间线起点 1.0、时长 4s、原速。
        let m = TimeMap2::new(10.0, 2.0, 6.0, 1.0, 4.0, RateCurve::Constant(1.0), false).unwrap();
        assert!((m.source_at(1.0) - 2.0).abs() < 1e-6, "trim in");
        assert!((m.source_at(5.0) - 6.0).abs() < 1e-6, "trim out");
        assert!((m.source_at(0.0) - 2.0).abs() < 1e-6, "时间线前钳入点");
        assert!((m.source_at(9.0) - 6.0).abs() < 1e-6, "时间线后钳出点");
    }

    #[test]
    fn speed_2x_halves_duration() {
        // 源 4s 全片，2× 速：走完全片只需 2s 时间线。
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 2.0, RateCurve::Constant(2.0), false).unwrap();
        assert!((m.source_at(1.0) - 2.0).abs() < 1e-5, "t=1 到源 2s");
        assert!((m.source_at(2.0) - 4.0).abs() < 1e-5, "t=2 到源末尾");
    }

    #[test]
    fn reverse_swaps_head_tail() {
        let m = TimeMap2::new(4.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(1.0), true).unwrap();
        assert!((m.source_at(0.0) - 4.0).abs() < 1e-6, "倒放首 = 源尾");
        assert!((m.source_at(4.0) - 0.0).abs() < 1e-6, "倒放尾 = 源首");
        // 单调递减。
        let mut prev = f32::INFINITY;
        for i in 0..=40 {
            let v = m.source_at(i as f32 * 0.1);
            assert!(v <= prev + 1e-6, "倒放应单调不增");
            prev = v;
        }
    }

    #[test]
    fn keyframed_rate_is_monotone() {
        // 0..2s 速率 1，2..4s 速率 3。
        let m = TimeMap2::new(
            8.0,
            0.0,
            8.0,
            0.0,
            4.0,
            RateCurve::Keyframed(vec![(0.0, 1.0), (2.0, 1.0), (4.0, 3.0)]),
            false,
        )
        .unwrap();
        let mut prev = -1.0;
        for i in 0..=80 {
            let v = m.source_at(i as f32 * 0.05);
            assert!(v >= prev - 1e-5, "关键帧速率下映射应单调不减");
            prev = v;
        }
    }

    #[test]
    fn degenerate_inputs_error_structured() {
        // 零长度源区间。
        assert_eq!(
            TimeMap2::new(4.0, 2.0, 2.0, 0.0, 2.0, RateCurve::Constant(1.0), false).unwrap_err().code,
            "NLE1002"
        );
        // 零时长时间线。
        assert_eq!(
            TimeMap2::new(4.0, 0.0, 4.0, 0.0, 0.0, RateCurve::Constant(1.0), false).unwrap_err().code,
            "NLE1003"
        );
        // 零/负速率。
        assert_eq!(
            TimeMap2::new(4.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(0.0), false).unwrap_err().code,
            "NLE1004"
        );
        // 负源时长。
        assert_eq!(
            TimeMap2::new(-1.0, 0.0, 4.0, 0.0, 4.0, RateCurve::Constant(1.0), false).unwrap_err().code,
            "NLE1001"
        );
    }
}
