//! AudioReactive 通道（B8；复用 B6 [`prism_audio::reactive`] 烘焙表，禁重复实现）。
//!
//! # 复用声明（约束：勿重复造）
//!
//! 频段能量表由 B6 [`prism_audio::reactive::bake_reactive`] 烘焙（FFT 加窗谱
//! 分析 → 频段能量），本模块**只做粒子侧映射**：
//!
//! - [`ReactiveModulation`]：把频段映射到粒子参数增益
//!   （发射率 / 速度 / 尺寸），线性混合（`param · (1 + Σ w_b · band_b · amount)`）。
//! - [`modulation_for_frame`]：从 [`ReactiveBake`] 查第 `frame` 帧频段，
//!   返回该帧的三增益——**帧级纯函数**（帧级纯函数保证，seek 等价免获得）。
//!
//! 无新增频段分析代码；无隐藏状态；无全局 RNG（驱动量来自烘焙表，纯查表）。
//!
//! # feature 门
//!
//! [`ReactiveModulation`] / [`FrameGains`] 为纯映射（无外部依赖），恒可用；
//! 依赖 B6 烘焙表类型的 [`modulation_for_frame`] / [`gains_series`] 在
//! feature `audio`（默认开）下提供。`--no-default-features` 时 AudioReactive
//! 降级为单位增益（下游按无音频驱动处理，不静默编造数值）。

/// 频段 → 粒子参数的映射规格。
///
/// 每参数一个频段权重向量（`w_b`，长度 = 频段数）；增益 =
/// `1 + amount · Σ_b w_b · band_b`（`band_b ∈ [0,1]`）。`amount` 为总强度
/// （0 = 无驱动，1 = 满驱动）。
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveModulation {
    /// 频段 → 发射率增益的权重。
    pub rate_weights: Vec<f32>,
    /// 频段 → 速度增益的权重。
    pub speed_weights: Vec<f32>,
    /// 频段 → 尺寸增益的权重。
    pub size_weights: Vec<f32>,
    /// 发射率驱动强度。
    pub rate_amount: f32,
    /// 速度驱动强度。
    pub speed_amount: f32,
    /// 尺寸驱动强度。
    pub size_amount: f32,
}

impl Default for ReactiveModulation {
    /// 默认：低频驱动尺寸、中频驱动发射率、高频驱动速度（三分段常用映射）。
    fn default() -> Self {
        Self {
            rate_weights: vec![0.0, 1.0, 0.0],
            speed_weights: vec![0.0, 0.0, 1.0],
            size_weights: vec![1.0, 0.0, 0.0],
            rate_amount: 1.0,
            speed_amount: 1.0,
            size_amount: 1.0,
        }
    }
}

/// 一帧的驱动增益（发射率 / 速度 / 尺寸；≥ 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameGains {
    /// 发射率增益（1 = 无变化）。
    pub rate: f32,
    /// 速度增益。
    pub speed: f32,
    /// 尺寸增益。
    pub size: f32,
}

impl Default for FrameGains {
    fn default() -> Self {
        Self { rate: 1.0, speed: 1.0, size: 1.0 }
    }
}

/// 权重 · 频段的确定性内积（固定序求和；越界权重视为 0）。
#[inline]
fn weighted(weights: &[f32], bands: &[f32]) -> f32 {
    let n = weights.len().min(bands.len());
    let mut acc = 0.0_f32;
    for b in 0..n {
        acc += weights[b] * bands[b];
    }
    acc
}

impl ReactiveModulation {
    /// 从某帧频段行求三增益（纯函数）。
    pub fn gains(&self, bands: &[f32]) -> FrameGains {
        let r = 1.0 + self.rate_amount * weighted(&self.rate_weights, bands);
        let s = 1.0 + self.speed_amount * weighted(&self.speed_weights, bands);
        let z = 1.0 + self.size_amount * weighted(&self.size_weights, bands);
        FrameGains { rate: r.max(0.0), speed: s.max(0.0), size: z.max(0.0) }
    }
}

// ---------------------------------------------------------------------------
// 依赖 B6 烘焙表的桥接（feature `audio`）
// ---------------------------------------------------------------------------

/// 从烘焙表查第 `frame` 帧的驱动增益（**帧级纯函数**；seek 第 N 帧 ≡ 顺序推进）。
///
/// `bake` 为 `None` 时返回单位增益（无音频驱动；不静默编造数值）。
#[cfg(feature = "audio")]
pub fn modulation_for_frame(
    bake: Option<&prism_audio::reactive::ReactiveBake>,
    modu: &ReactiveModulation,
    frame: u64,
) -> FrameGains {
    match bake {
        Some(b) => modu.gains(b.frame_bands(frame)),
        None => FrameGains::default(),
    }
}

/// 烘焙表全帧发射率增益序列（供 [`crate::emitter::Emitter::spawn_count`] 累积）。
#[cfg(feature = "audio")]
pub fn gains_series(
    bake: &prism_audio::reactive::ReactiveBake,
    modu: &ReactiveModulation,
) -> Vec<f32> {
    (0..bake.frames)
        .map(|f| modu.gains(bake.frame_bands(f as u64)).rate)
        .collect()
}

/// 无音频后端时的降级：[`modulation_for_frame`] 的单位增益形态
/// （`--no-default-features`；签名不含 B6 类型，调用方按 `None` 分支处理）。
#[cfg(not(feature = "audio"))]
pub fn modulation_for_frame(
    _bake: Option<&()>,
    _modu: &ReactiveModulation,
    _frame: u64,
) -> FrameGains {
    FrameGains::default()
}
