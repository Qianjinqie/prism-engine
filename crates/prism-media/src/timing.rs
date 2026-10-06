//! 时间映射器（B11；白皮书 §19 视频时间映射）。
//!
//! **问题**：引擎时间轴 t → 素材内时间/帧索引（+ 音画同步）。
//!
//! **选定**：纯函数映射 + 帧间混合（三段式）——
//! 标准化 `u = t / dur` → 模式映射（direct/loop/pingpong/变速/Expr）→
//! 帧索引 = `floor(u'·fps)`、小数部分 = 帧间混合权重。
//!
//! **超越点**：播放头状态机 seek 要重置状态；**纯函数映射 seek 零成本且音画
//! 共享同一 `f(t)`**——确定性架构把"同步问题"消灭在表示层（±1 帧由共享 f
//! 保证，不是事后校准）。
//!
//! 变速支持常量 k（直接重映射 `u' = u·k`）与**通道 k(t)**（`τ(t) = ∫₀ᵗ k(s)ds`
//! 的分段线性闭式积分，S9 可积）——映射单调、seek 纯函数。
//!
//! 本模块零第三方、零 libm、纯函数。

/// 时间映射模式（五模式）。
#[derive(Debug, Clone, PartialEq)]
pub enum TimingMode {
    /// 直通：`u' = clamp(u, 0, 1)`（超出素材时长即钳在末帧）。
    Direct,
    /// 循环：`u' = u mod 1`（取小数部分）。
    Loop,
    /// 乒乓：三角折返 `u' = 1 − |1 − 2·(u mod 1)|`。
    PingPong,
    /// 变速：常量倍率 `k`（`u' = clamp(u·k, 0, 1)`）。
    Speed(f32),
    /// 变速（通道）：`τ(t) = ∫₀ᵗ clamp(k(s),0,∞) ds / dur` 的归一化（分段线性
    /// 闭式积分）。`Ts` 为 `(秒, 倍率)` 采样点，按时间升序；越界/未覆盖区间
    /// 取最近端点倍率（阶跃保持），保证映射为时间的连续分段线性纯函数。
    SpeedCurve { samples: Vec<(f32, f32)>, dur: f32 },
    /// 用户表达式：采样点 `(秒, 归一化映射值 u')`，线性插值；越界延伸端点值。
    Expr { samples: Vec<(f32, f32)> },
}

/// 视频/音频素材的时间映射器。
///
/// `dur` 为素材时长（秒）、`fps` 为素材帧率。`map(t)` 返回素材内归一化时间
/// `u' ∈ [0, 1]`（闭式纯函数，seek 即调用，无状态）。
#[derive(Debug, Clone, PartialEq)]
pub struct TimeMap {
    dur: f32,
    fps: f32,
    mode: TimingMode,
}

/// 小数位数（帧索引定点量化），10 位足以分辨任何实用 fps 的小数帧。
const FRAC_BITS: u32 = 10;

impl TimeMap {
    /// 构造映射器（`dur > 0`、`fps > 0`；违约钳到 `f32::EPSILON` 级别以避免除零，
    /// 越界属调用方参数错误——不静默产生 NaN）。
    pub fn new(dur: f32, fps: f32, mode: TimingMode) -> Self {
        Self {
            dur: if dur > 0.0 { dur } else { f32::MIN_POSITIVE },
            fps: if fps > 0.0 { fps } else { f32::MIN_POSITIVE },
            mode,
        }
    }

    /// 素材时长（秒）。
    pub fn duration(&self) -> f32 {
        self.dur
    }

    /// 素材帧率。
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// 映射模式引用。
    pub fn mode(&self) -> &TimingMode {
        &self.mode
    }

    /// 时间映射的纯函数核心：`t`（秒）→ 归一化素材内时间 `u' ∈ [0, 1]`。
    ///
    /// `u = t / dur` 为标准化引擎时间；各模式把 `u` 映射到 `u'`（白皮书 §19）。
    pub fn map(&self, t: f32) -> f32 {
        let u = t / self.dur;
        let v = match &self.mode {
            TimingMode::Direct => u.clamp(0.0, 1.0),
            TimingMode::Loop => fract(u),
            TimingMode::PingPong => 1.0 - (1.0 - 2.0 * fract(u)).abs(),
            TimingMode::Speed(k) => (u * *k).clamp(0.0, 1.0),
            TimingMode::SpeedCurve { samples, dur } => {
                (integral_curve(samples, t) / dur.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0)
            }
            TimingMode::Expr { samples } => sample_curve(samples, t).clamp(0.0, 1.0),
        };
        v.clamp(0.0, 1.0)
    }

    /// 当前时间的素材帧索引（`floor(u'·fps)`，钳在总帧数内）。
    pub fn frame_at(&self, t: f32) -> u32 {
        let u = self.map(t);
        let total = self.total_frames();
        let idx = (u * self.fps) as i64;
        idx.clamp(0, total as i64 - 1) as u32
    }

    /// 帧间混合权重 = `fract(u'·fps)`（前帧/后帧 lerp 系数）。
    pub fn blend_at(&self, t: f32) -> f32 {
        let u = self.map(t);
        // 末帧处权重归零，避免越界取 next 帧。
        let total = self.total_frames();
        let raw = u * self.fps;
        if raw >= total as f32 - 1.0 {
            0.0
        } else {
            fract(raw)
        }
    }

    /// 帧间混合的目标对：`(前帧索引, 后帧索引, 权重)`——权重 0 时前后帧相同。
    pub fn frame_pair_at(&self, t: f32) -> (u32, u32, f32) {
        let f = self.frame_at(t);
        let w = self.blend_at(t);
        let next = if w > 0.0 { (f + 1).min(self.total_frames() - 1) } else { f };
        (f, next, w)
    }

    /// 素材总帧数（由 `dur × fps` 派生，至少 1）。
    pub fn total_frames(&self) -> u32 {
        let n = (self.dur * self.fps).round() as i64;
        n.max(1) as u32
    }
}

/// 小数部分（`u - floor(u)`）。
#[inline]
fn fract(u: f32) -> f32 {
    u - u.floor()
}

/// `∫₀ᵗ clamp(k(s), 0, ∞) ds` 的分段线性闭式积分（梯形法精确）。
///
/// 倍率非负钳制保证映射单调；采样点按时间升序，区间外取端点倍率。
fn integral_curve(samples: &[(f32, f32)], t: f32) -> f32 {
    if samples.is_empty() || t <= samples[0].0 {
        // t 在首点之前：以首点倍率外推当前区间（不为负）。
        let k0 = samples.first().map(|s| s.1).unwrap_or(1.0).max(0.0);
        return k0 * t.max(0.0);
    }
    let mut acc = 0.0_f32;
    let first_t = samples[0].0;
    if first_t > 0.0 {
        // 0..first_t：以首点倍率恒定外推。
        acc += samples[0].1.max(0.0) * first_t;
    }
    for w in samples.windows(2) {
        let (t0, k0) = w[0];
        let (t1, k1) = w[1];
        let (k0n, k1n) = (k0.max(0.0), k1.max(0.0));
        if t >= t1 {
            // 整段梯形积分。
            acc += (k0n + k1n) * 0.5 * (t1 - t0);
        } else {
            // 当前段部分积分（线性倍率 → 梯形精确）。
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
    // t 超出末点：以末点倍率外推。
    let last = samples[samples.len() - 1];
    acc += last.1.max(0.0) * (t - last.0).max(0.0);
    acc
}

/// 采样曲线线性插值（Expr 模式；点集按时间升序；越界延伸端点值）。
fn sample_curve(samples: &[(f32, f32)], t: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    if t <= samples[0].0 {
        return samples[0].1;
    }
    for w in samples.windows(2) {
        let (t0, v0) = w[0];
        let (t1, v1) = w[1];
        if t <= t1 {
            let span = t1 - t0;
            if span <= 0.0 {
                return v1;
            }
            let a = (t - t0) / span;
            return v0 + (v1 - v0) * a;
        }
    }
    samples[samples.len() - 1].1
}

/// 帧计数（供 seek 等价断言使用）。
///
/// 直接按 [`TimeMap::frame_at`] 计算——与顺序播放使用**同一个纯函数**，
/// 故"跳第 N 帧 ≡ 顺序播放到第 N 帧"是同一计算的两种调用方式（白皮书 §19）。
pub fn frame_index(map: &TimeMap, t: f32) -> (u32, u32) {
    let f = map.frame_at(t);
    let quant = quantize(map.map(t));
    (f, quant)
}

/// 把归一化时间量化到 10 位定点（用于逐帧断言的可比较整数键）。
fn quantize(u: f32) -> u32 {
    (u.clamp(0.0, 1.0) * ((1u32 << FRAC_BITS) as f32)).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_clamps_out_of_range() {
        let m = TimeMap::new(2.0, 10.0, TimingMode::Direct);
        assert_eq!(m.map(0.0), 0.0);
        assert_eq!(m.map(1.0), 0.5);
        assert_eq!(m.map(2.0), 1.0);
        assert_eq!(m.map(5.0), 1.0, "超出时长钳在末帧");
        assert_eq!(m.map(-1.0), 0.0);
    }

    #[test]
    fn loop_wraps_and_pingpong_folds() {
        let lp = TimeMap::new(2.0, 10.0, TimingMode::Loop);
        assert!((lp.map(2.5) - 0.25).abs() < 1e-6);
        assert!((lp.map(4.0) - 0.0).abs() < 1e-6);
        let pp = TimeMap::new(2.0, 10.0, TimingMode::PingPong);
        assert!((pp.map(0.5) - 0.5).abs() < 1e-6, "u=0.25 → 折返半程前半");
        assert!((pp.map(1.0) - 1.0).abs() < 1e-6, "u=0.5 → 端点");
        assert!((pp.map(2.0) - 0.0).abs() < 1e-6, "u=1 → 回到起点");
        assert!((pp.map(2.5) - 0.5).abs() < 1e-6, "折返半程");
    }

    #[test]
    fn speed_constant_remaps() {
        let m = TimeMap::new(4.0, 10.0, TimingMode::Speed(2.0));
        assert!((m.map(1.0) - 0.5).abs() < 1e-6, "2× 在 t=1 到半程");
        assert_eq!(m.map(2.0), 1.0, "2× 在 t=2(dur 一半)即时长耗尽");
    }

    #[test]
    fn speed_curve_integral_is_monotone_and_bounded() {
        // k(t) = 1（0..1s），k(t) = 3（1..2s）；dur = 2s。
        let m = TimeMap::new(
            2.0,
            30.0,
            TimingMode::SpeedCurve { samples: vec![(0.0, 1.0), (1.0, 1.0), (2.0, 3.0)], dur: 2.0 },
        );
        let mut prev = -1.0;
        for i in 0..=40 {
            let t = i as f32 * 0.05;
            let u = m.map(t);
            assert!(u >= prev - 1e-6, "映射必须单调不减 (t={t})");
            assert!((0.0..=1.0).contains(&u));
            prev = u;
        }
    }

    #[test]
    fn expr_linear_sample_curve() {
        let m = TimeMap::new(2.0, 10.0, TimingMode::Expr { samples: vec![(0.0, 0.0), (2.0, 1.0)] });
        assert!((m.map(1.0) - 0.5).abs() < 1e-6);
        assert_eq!(m.map(9.0), 1.0, "越界延伸端点");
    }

    #[test]
    fn seek_equivalent_to_sequential_playback() {
        // 跳第 N 帧 ≡ 顺序播放（白皮书 §19：同一纯函数 f(t) 的两种调用）。
        let modes = [
            TimingMode::Direct,
            TimingMode::Loop,
            TimingMode::PingPong,
            TimingMode::Speed(1.5),
            TimingMode::SpeedCurve { samples: vec![(0.0, 1.0), (3.0, 2.0)], dur: 3.0 },
            TimingMode::Expr { samples: vec![(0.0, 0.0), (1.0, 0.3), (3.0, 1.0)] },
        ];
        for mode in modes {
            let m = TimeMap::new(3.0, 24.0, mode.clone());
            // 顺序播放 0..3s，按 1/240s 步进采样。
            let steps = 720;
            let mut seq: Vec<(u32, u32)> = Vec::with_capacity(steps);
            for i in 0..steps {
                let t = i as f32 * (3.0 / steps as f32);
                seq.push(frame_index(&m, t));
            }
            // 随机顺序 seek：任意步的结果必须与顺序播放逐帧一致。
            for i in (0..steps).rev() {
                let t = i as f32 * (3.0 / steps as f32);
                assert_eq!(frame_index(&m, t), seq[i], "seek 与顺序播放不一致（模式 {mode:?}）");
            }
        }
    }

    #[test]
    fn frame_pair_and_total_frames() {
        let m = TimeMap::new(1.0, 10.0, TimingMode::Direct);
        assert_eq!(m.total_frames(), 10);
        // t=0.05 → u'=0.05 → idx=0, w=0.5 → (0,1,0.5)。
        let (a, b, w) = m.frame_pair_at(0.05);
        assert_eq!((a, b), (0, 1));
        assert!((w - 0.5).abs() < 1e-6);
        // 末帧权重归零。
        let (a2, b2, w2) = m.frame_pair_at(1.0);
        assert_eq!((a2, b2), (9, 9));
        assert_eq!(w2, 0.0);
    }
}
