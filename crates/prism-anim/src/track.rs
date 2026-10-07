//! 关键帧轨道与循环模式（工作流 A2；S9 分段多项式特解）。
//!
//! [`Track`] 是关键帧的有序集合：相邻两帧之间按左帧的 [`EasingKind`] 缓动
//! 插值（AE 语义：缓动归属段起点，宪章对照示例 `0s: -80 easeOutCubic, 1.2s: 0`）。
//!
//! 采样规则（确定性，文档化）：
//! - `t < 首帧时间` → 首帧值；`t ≥ 末帧时间` → 末帧值（钳制）；`t` 恰等于
//!   某帧时间时走主逻辑，段左端取**最后一条 time ≤ t 的帧**（同刻帧"后者
//!   生效"）；
//! - 帧间：`u = (t − tᵢ)/(tᵢ₊₁ − tᵢ)`，值 = `lerp(vᵢ, vᵢ₊₁, easingᵢ(u))`；
//! - 非有限时间：`t` 为 NaN 时返回 NaN（值语义传播——NaN 不满足任何 IEEE
//!   比较、不落入任何段，不得进入段查找）；`t` 为 ±∞ 时走首/末钳制分支
//!   （`t < 首帧` / `t ≥ 末帧`）早退；
//! - 同时刻多帧（构造时稳定排序）：后者生效（最后一条 time ≤ t 的帧为段
//!   左端，段长为 0 时直接取右帧值）——确定性规则，DSL 层将以
//!   "关键帧时间倒序/重复"防呆规则预警（宪章 C18）。
//!
//! [`LoopMode`] 把 `[0, duration]` 上的表现按 direct/loop/pingpong 铺满整条
//! 时间轴（时间映射公式的 A2 通道形态：`u mod 1` 与三角折返）。

use crate::easing::EasingKind;

/// 关键帧：时刻（秒）、值、以及**以本帧为段起点**时的缓动函数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    /// 时刻（秒，格间浮点插值的端点皆格锚语义见）。
    pub time: f32,
    /// 值。
    pub value: f32,
    /// 从本帧到下一帧的缓动（本帧为段起点）。
    pub easing: EasingKind,
}

impl Keyframe {
    /// 以时刻 / 值 / 缓动构造关键帧。
    pub fn new(time: f32, value: f32, easing: EasingKind) -> Self {
        Self { time, value, easing }
    }
}

/// 关键帧轨道：构造时按时间**稳定排序**（同输入同序，确定性）。
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    keyframes: Vec<Keyframe>,
}

impl Track {
    /// 以关键帧集合构造轨道（按时间稳定排序；空轨道采样恒 0，DSL 层拒绝空 anim 块）。
    pub fn new(mut keyframes: Vec<Keyframe>) -> Self {
        // 稳定排序：相等时间保持输入序（确定性 tie-break）。
        keyframes.sort_by(|a, b| {
            a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { keyframes }
    }

    /// 排序后的关键帧切片。
    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// 采样：t 的纯函数（同 t 同值，逐位）。
    ///
    /// 非有限时间语义（文档化）：NaN 时间 → NaN 值（值语义传播；NaN 不
    /// 满足任何 IEEE 比较、不落入任何段，若放行到段查找会因 `partition_point`
    /// 谓词恒 false 得到 `0 − 1` 的 usize 下溢，QA-02）；−∞ → 首帧值、
    /// +∞ → 末帧值（走钳制分支早退）。
    pub fn sample(&self, t: f32) -> f32 {
        let k = &self.keyframes;
        if k.is_empty() {
            return 0.0;
        }
        // NaN 早退（QA-02 修复）：NaN 时间按值语义传播 NaN，不进入段查找
        // （NaN 谓词恒 false 会导致索引下溢 panic，报错不 panic 精神）。
        if t.is_nan() {
            return f32::NAN;
        }
        if t < k[0].time {
            return k[0].value;
        }
        let last = k.len() - 1;
        if t >= k[last].time {
            return k[last].value;
        }
        // t 严格落在 (首帧, 末帧) 内：最后一条 time ≤ t 的帧为段左端。
        let i = k.partition_point(|kf| kf.time <= t) - 1;
        let dt = k[i + 1].time - k[i].time;
        if dt <= 0.0 {
            // 同时刻帧：后者生效（确定性规则，见模块文档）。
            return k[i + 1].value;
        }
        let u = (t - k[i].time) / dt;
        let e = k[i].easing.apply(u);
        k[i].value + (k[i + 1].value - k[i].value) * e
    }
}

/// 循环模式（direct / loop / pingpong； 时间映射三段式的通道形态）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// 直通：`[0, duration]` 之外钳制（首前取首、末后取末）。
    Direct,
    /// 循环：`t mod duration` 回绕（t = duration 整回绕到起点，`u mod 1`）。
    Loop,
    /// 往复：以 `2·duration` 为周期三角折返（`1 − |1 − 2(u mod 1)|` 同构）。
    PingPong,
}

impl LoopMode {
    /// 把引擎时间 t 折算进内层时间轴 `[0, duration]`（纯函数）。
    ///
    /// 文档化退化行为：负 t 钳 0；`duration ≤ 0` 退化为恒等映射（DSL 层
    /// 以防呆规则拦截非法 duration，宪章 C18 防呆规则库）。
    pub fn map(&self, t: f32, duration: f32) -> f32 {
        if duration <= 0.0 {
            return t;
        }
        if t < 0.0 {
            return 0.0;
        }
        match self {
            Self::Direct => {
                if t > duration {
                    duration
                } else {
                    t
                }
            }
            Self::Loop => t % duration,
            Self::PingPong => {
                let period = duration * 2.0;
                let phase = t % period;
                if phase <= duration {
                    phase
                } else {
                    period - phase
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 2：关键帧插值边界逐段断言；关卡 3：Loop 三模式边界值断言）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 双帧线性轨道：0s:0 → 1s:10。
    fn linear_pair() -> Track {
        Track::new(vec![
            Keyframe::new(0.0, 0.0, EasingKind::Linear),
            Keyframe::new(1.0, 10.0, EasingKind::Linear),
        ])
    }

    /// 关卡 2：首帧前 / 末帧后 / 两帧间逐段断言。
    #[test]
    fn track_boundaries_before_between_after() {
        let tr = linear_pair();
        // 首帧前：钳首帧值。
        assert_eq!(tr.sample(-5.0), 0.0);
        assert_eq!(tr.sample(0.0), 0.0);
        // 两帧间：线性插值 u=0.25/0.5/0.75。
        assert!((tr.sample(0.25) - 2.5).abs() < 1e-6);
        assert!((tr.sample(0.5) - 5.0).abs() < 1e-6);
        assert!((tr.sample(0.75) - 7.5).abs() < 1e-6);
        // 末帧后：钳末帧值（t=1 恰在末帧，t>1 超出）。
        assert!((tr.sample(1.0) - 10.0).abs() < 1e-6);
        assert!((tr.sample(2.0) - 10.0).abs() < 1e-6);
    }

    /// 关卡 2：段起点缓动生效——CubicOut(0.5) = 0.875。
    #[test]
    fn track_applies_left_keyframe_easing() {
        let tr = Track::new(vec![
            Keyframe::new(0.0, 0.0, EasingKind::CubicOut),
            Keyframe::new(1.0, 10.0, EasingKind::Linear),
        ]);
        // u = 0.5 → ease = 1 − (1−0.5)³ = 0.875 → 值 = 8.75。
        assert!((tr.sample(0.5) - 8.75).abs() < 1e-6);
    }

    /// 关卡 2：关键帧点精确命中帧值（构造时乱序，采样仍正确）。
    #[test]
    fn track_hits_keyframe_values_after_unsorted_construction() {
        let tr = Track::new(vec![
            Keyframe::new(2.0, 20.0, EasingKind::Linear),
            Keyframe::new(0.0, 0.0, EasingKind::Linear),
            Keyframe::new(1.0, 10.0, EasingKind::Linear),
        ]);
        assert_eq!(tr.keyframes()[0].time, 0.0, "构造时应按时间排序");
        assert_eq!(tr.sample(0.0), 0.0);
        assert!((tr.sample(1.0) - 10.0).abs() < 1e-6);
        assert!((tr.sample(2.0) - 20.0).abs() < 1e-6);
        assert!((tr.sample(3.0) - 20.0).abs() < 1e-6, "末帧后钳制");
    }

    /// 关卡 2：同时刻多帧按"后者生效"确定性解析；单帧轨道为常量。
    #[test]
    fn track_equal_times_and_single_keyframe() {
        let tr = Track::new(vec![
            Keyframe::new(0.5, 1.0, EasingKind::Linear),
            Keyframe::new(0.5, 2.0, EasingKind::Linear),
            Keyframe::new(1.0, 3.0, EasingKind::Linear),
        ]);
        // t=0.5 恰在重复帧时刻：段左端取最后一条 time≤t 的帧（值 2.0），
        // 段长 0 → 取右帧值 2.0；首前/末后钳制两侧仍成立。
        assert_eq!(tr.sample(0.5), 2.0);
        assert_eq!(tr.sample(0.4), 1.0, "首帧前（0.4 < 0.5）钳首帧值");

        let single = Track::new(vec![Keyframe::new(1.0, 7.0, EasingKind::Linear)]);
        for t in [-1.0_f32, 0.0, 1.0, 9.0] {
            assert_eq!(single.sample(t), 7.0, "单帧轨道为常量");
        }

        let empty = Track::new(vec![]);
        assert_eq!(empty.sample(1.0), 0.0, "空轨道恒 0（文档化退化）");
    }

    /// 关卡 3：Loop 三模式边界值（0s:0 → 1s:10，duration = 1s）。
    #[test]
    fn loop_mode_boundary_values() {
        let tr = linear_pair();
        // Direct：首前取首、末后取末。
        let cases_direct = [
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 5.0),
            (1.0, 10.0),
            (2.5, 10.0),
        ];
        for (t, want) in cases_direct {
            let mapped = LoopMode::Direct.map(t, 1.0);
            assert!((tr.sample(mapped) - want).abs() < 1e-6, "Direct t={t}");
        }
        // Loop：周期回绕，t=duration 整回绕到 0（u mod 1）。
        let cases_loop = [(0.0, 0.0), (0.5, 5.0), (1.0, 0.0), (1.5, 5.0), (2.0, 0.0)];
        for (t, want) in cases_loop {
            let mapped = LoopMode::Loop.map(t, 1.0);
            assert!((tr.sample(mapped) - want).abs() < 1e-6, "Loop t={t}");
        }
        // PingPong：t=duration 为折返点（末帧），一个整周期（2s）后回首帧。
        let cases_pp = [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0), (1.5, 5.0), (2.0, 0.0), (2.5, 5.0)];
        for (t, want) in cases_pp {
            let mapped = LoopMode::PingPong.map(t, 1.0);
            assert!((tr.sample(mapped) - want).abs() < 1e-6, "PingPong t={t}");
        }
    }

    /// Loop 退化输入：负 t 钳 0；duration ≤ 0 恒等映射（文档化行为）。
    #[test]
    fn loop_mode_degenerate_inputs() {
        assert_eq!(LoopMode::Loop.map(-0.5, 1.0), 0.0);
        assert_eq!(LoopMode::PingPong.map(-3.0, 1.0), 0.0);
        assert_eq!(LoopMode::Direct.map(-3.0, 1.0), 0.0);
        for mode in [LoopMode::Direct, LoopMode::Loop, LoopMode::PingPong] {
            assert_eq!(mode.map(1.7, 0.0), 1.7, "duration=0 退化为恒等");
            assert_eq!(mode.map(1.7, -1.0), 1.7, "duration<0 退化为恒等");
        }
    }

    /// QA-02：NaN 时间返回 NaN（值语义传播）而非 panic；空轨道恒 0 的
    /// 文档化退化不受影响（空轨道无值可传播，恒 0 优先）。
    #[test]
    fn track_nan_time_returns_nan_without_panic() {
        let tr = linear_pair();
        assert!(tr.sample(f32::NAN).is_nan(), "NaN 时间应传播 NaN，不 panic");
        let empty = Track::new(vec![]);
        assert_eq!(empty.sample(f32::NAN), 0.0, "空轨道恒 0（文档化退化，优先于 NaN 传播）");
    }
}
