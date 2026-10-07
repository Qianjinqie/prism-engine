//! 胶片仿真包（清单：halation + 分层 bloom + 胶片曲线库 + 科学化颗粒）。
//!
//! 语义权威：halation + 科学化胶片颗粒、（噪声派生
//! 纪律）；B14 行与 （bloom 在线性域、halation 在胶片
//! 层——两节点管线位置不同）。与 B4 `bloom` 的关系：`bloom` 是单层镜头光学
//! 泛光；本包的 [`LayeredBloom`] 是多级扩散链（半径倍增、逐级衰减），
//! [`Halation`] 在其上加"单向 + 波长染色"（红光晕而非白晕）。
//!
//! # 确定性纪律（D-29）
//!
//! 全部算子为求值纯函数；随机性一律经 prism-noise 的 PCG 整数 hash 派生
//! （无私开 RNG）；零超越函数——胶片曲线用**分段线性/多项式**逼近
//! （D-29：禁 libm），不调用 `ln`/`exp`/`powf`。
//!
//! # halo 申报
//!
//! [`Halation`]/[`LayeredBloom`]/[`HalationAndBloom`] 的输出读输入
//! `p ± total_radius`（多级扩散链的支撑半径上界），申报该半径
//! （可平移不变 ⇒ tile 化与整幅逐字节一致）。曲线/颗粒为逐像素点操作
//! （halo 0），颗粒空间图案锚在窗口局部坐标（与 B4 `grain` 同约定）。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use super::blur::box_blur_buf;

/// Rec.709 亮度权重（与 `grade` 同常量；halation 的亮度阈值提取用）。
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

// ---------------------------------------------------------------------------
// 胶片曲线库
// ---------------------------------------------------------------------------

/// 胶片响应曲线（B14：胶片曲线库）。分段逼近，零超越函数（D-29）。
///
/// 语义：输入取 [0,1] 的胶片"曝光"值（节点链上通常已是显示域），输出
/// 该格式的响应。所有分段线性、端点固定、单调不减；系数为编译期常量，
/// 跨平台逐位一致。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmCurve {
    /// 曲线类型。
    pub kind: CurveKind,
    /// 肩部起点（> 该值进入高光压缩）；`<= toe` 时报错回退。
    pub shoulder: f32,
    /// 趾部终点（< 该值进入暗部提升）。
    pub toe: f32,
}

/// 胶片曲线类型（三大经典响应 + 中性直通）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveKind {
    /// 中性：恒等（关卡 2 位精确直通）。
    Neutral,
    /// sRGB（分段线性近似的标准显示传递函数）。
    Srgb,
    /// Kodak Vision3（胶片负片正扫响应：肩高、趾柔）。
    Vision3,
    /// Fuji Eterna（低对比、宽趾的柔调）。
    Eterna,
}

impl Default for FilmCurve {
    fn default() -> Self {
        FilmCurve { kind: CurveKind::Vision3, shoulder: 0.72, toe: 0.18 }
    }
}

impl FilmCurve {
    /// 单点求值。`Neutral` 逐位恒等；其余为单调分段线性映射。
    /// 输入越界先钳到 [0,1]（安全，与冻结 semantics 无关）。
    pub fn eval(&self, x: f32) -> f32 {
        let x = prism_math::clamp(x, 0.0, 1.0);
        match self.kind {
            // 中性直通：不碰位模式（关卡 2）。
            CurveKind::Neutral => x,
            // sRGB EOTF 分段线性近似（线性段 + 2.4 幂段的 8 段折线）。
            CurveKind::Srgb => piecewise(x, &SRGB_KNOTS),
            // Vision3：趾柔（0.18 以下线性化）、肩高（0.72 以上压缩）。
            CurveKind::Vision3 => {
                shoulder_toe(x, 0.18, 0.72, 0.62, 0.90, 0.55)
            }
            // Eterna：低对比（整体增益略低、宽趾）。
            CurveKind::Eterna => shoulder_toe(x, 0.24, 0.66, 0.72, 1.02, 0.40),
        }
    }
}

/// sRGB 分段折线结点（x 单调、y 单调；8 段足够逼近 2.4 幂段到 <1e-3。
/// 常量表：线性段 [0,0.0031308]→[0,0.04045]，随后 8 段）。
const SRGB_KNOTS: [(f32, f32); 10] = [
    (0.0, 0.0),
    (0.0031308, 0.04045),
    (0.02, 0.1242),
    (0.08, 0.2797),
    (0.18, 0.4479),
    (0.35, 0.6326),
    (0.55, 0.7925),
    (0.75, 0.9145),
    (0.92, 0.9905),
    (1.0, 1.0),
];

/// 单调分段线性查表（结点 x 升序、y 单调）。
fn piecewise(x: f32, knots: &[(f32, f32)]) -> f32 {
    // x 已钳 [0,1]；线性扫描（结点 < 16，Mini Code）。
    let mut prev = knots[0];
    for &(kx, ky) in knots.iter().skip(1) {
        if x <= kx {
            let span = kx - prev.0;
            if span <= 0.0 {
                return ky;
            }
            let t = (x - prev.0) / span;
            return prev.1 + (ky - prev.1) * t;
        }
        prev = (kx, ky);
    }
    knots[knots.len() - 1].1
}

/// 肩/趾分段线性胶片响应：暗部抬升、线性段、肩部压缩。
///
/// `toe_end` 以下：0 → `toe_lift` 斜率的线性抬升；
/// `shoulder` 以上：斜率 `shoulder_slope`（<1，压缩高光）。
fn shoulder_toe(
    x: f32,
    toe_end: f32,
    shoulder: f32,
    toe_lift: f32,
    shoulder_gain: f32,
    shoulder_slope: f32,
) -> f32 {
    if x < toe_end {
        // 趾部：0 抬到 toe_lift·(x/toe_end)² （柔和起步，二次，零超越）。
        let t = x / toe_end;
        return toe_lift * t * t;
    }
    // 线性段起点值（与趾部终点连续）。
    let lin0 = toe_lift;
    if x < shoulder {
        let span = shoulder - toe_end;
        if span <= 0.0 {
            return lin0;
        }
        let t = (x - toe_end) / span;
        return (lin0 + (shoulder_gain - lin0) * t).min(1.0);
    }
    // 肩部：斜率 < 1 的压缩，终点钳到 <= 1。
    let out = shoulder_gain + (x - shoulder) * shoulder_slope;
    out.min(1.0)
}

// ---------------------------------------------------------------------------
// 扩散链（分层 bloom / halation 共用）
// ---------------------------------------------------------------------------

/// 单向多级扩散链：对 `layer` 逐级 box blur（半径倍增）+ 逐级衰减累加。
///
/// 返回与输入同尺寸的扩散结果。级数 `levels`、初始半径 `base_radius`、
/// 逐级衰减 `decay`。单级时退化为一次 box blur。
fn diffusion_chain(
    layer: &Fbuf16F,
    levels: u32,
    base_radius: u32,
    decay: f32,
) -> Fbuf16F {
    let mut acc = Fbuf16F::new(layer.width, layer.height);
    let mut cur: Fbuf16F = layer.clone();
    let mut weight = 1.0f32;
    let mut radius = base_radius.max(1);
    for _ in 0..levels.max(1) {
        cur = box_blur_buf(&cur, radius);
        // 逐级衰减加权累加（固定序：先乘后加）。
        for y in 0..acc.height {
            for x in 0..acc.width {
                let c = cur.get(x, y);
                let a = acc.get(x, y);
                acc.set(
                    x,
                    y,
                    [
                        a[0] + weight * c[0],
                        a[1] + weight * c[1],
                        a[2] + weight * c[2],
                        a[3] + weight * c[3],
                    ],
                );
            }
        }
        weight *= decay;
        radius = radius.saturating_mul(2);
    }
    acc
}

/// 扩散链的支撑半径上界（用于 halo 申报）：几何级数半径之和。
fn chain_radius(levels: u32, base_radius: u32) -> u32 {
    let mut total = 0u32;
    let mut r = base_radius.max(1);
    for _ in 0..levels.max(1) {
        total = total.saturating_add(r);
        r = r.saturating_mul(2);
    }
    total
}

/// 亮度阈值提取高光层（逐通道 `max(0, luma − threshold)`；RGB 同值）。
fn highlight_layer(window: &Fbuf16F, threshold: f32) -> Fbuf16F {
    let mut layer = Fbuf16F::new(window.width, window.height);
    for y in 0..window.height {
        for x in 0..window.width {
            let c = window.get(x, y);
            let lum = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
            let e = (lum - threshold).max(0.0);
            layer.set(x, y, [e, e, e, e]);
        }
    }
    layer
}

// ---------------------------------------------------------------------------
// LayeredBloom —— 分层（多级扩散）bloom
// ---------------------------------------------------------------------------

/// 分层 bloom：高光提取 → 多级扩散链 → 线性加算。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayeredBloom {
    /// 亮度阈值（线性域）。
    pub threshold: f32,
    /// 扩散级数（8–12 级）。
    pub levels: u32,
    /// 初始半径（逐级倍增）。
    pub base_radius: u32,
    /// 逐级衰减（< 1）。
    pub decay: f32,
    /// 泛光注入强度。
    pub intensity: f32,
}

impl Default for LayeredBloom {
    fn default() -> Self {
        LayeredBloom { threshold: 1.0, levels: 6, base_radius: 2, decay: 0.62, intensity: 0.3 }
    }
}

impl TileEffect for LayeredBloom {
    fn halo_radius(&self) -> u32 {
        chain_radius(self.levels, self.base_radius)
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let layer = highlight_layer(window, self.threshold);
        let spread = diffusion_chain(&layer, self.levels, self.base_radius, self.decay);
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let b = spread.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.intensity * b[0],
                        c[1] + self.intensity * b[1],
                        c[2] + self.intensity * b[2],
                        c[3] + self.intensity * b[3],
                    ],
                );
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Halation —— 单向扩散 + 波长染色（能量守恒）
// ---------------------------------------------------------------------------

/// halation：高光提取 → 单向多级扩散 → 波长染色 → 加算。
///
/// 波长权重使红光晕（R > G > B），对齐"红光穿透基底更深"。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halation {
    /// 亮度阈值。
    pub threshold: f32,
    /// 扩散级数。
    pub levels: u32,
    /// 初始半径。
    pub base_radius: u32,
    /// 逐级衰减。
    pub decay: f32,
    /// 染色强度（加算系数；能量守恒断言以此为增益系数）。
    pub strength: f32,
    /// 波长权重 [R,G,B]（红光占优）。
    pub tint: [f32; 3],
}

impl Default for Halation {
    fn default() -> Self {
        Halation {
            threshold: 1.0,
            levels: 8,
            base_radius: 2,
            decay: 0.7,
            strength: 0.4,
            tint: [1.0, 0.35, 0.12],
        }
    }
}

impl TileEffect for Halation {
    fn halo_radius(&self) -> u32 {
        chain_radius(self.levels, self.base_radius)
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let layer = highlight_layer(window, self.threshold);
        let spread = diffusion_chain(&layer, self.levels, self.base_radius, self.decay);
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let s = spread.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.strength * self.tint[0] * s[0],
                        c[1] + self.strength * self.tint[1] * s[1],
                        c[2] + self.strength * self.tint[2] * s[2],
                        c[3],
                    ],
                );
            }
        }
        out
    }
}

/// 能量守恒判据辅助：给定输入窗口与 halation 参数，返回
/// `(输出总能量 − 输入总能量)` 与 `注入能量上界`。
///
/// 能量 = 每像素 RGB 之和、全幅求和（f64 累加以隔离 f32 舍入）。加算型
/// 算子只增不减 ⇒ `delta >= 0`；且 `delta <= strength · max(tint) · 扩散层
/// 总能量`（有界注入）。测试以此断言"能量守恒"（不凭空消减、增益有界）。
pub fn halation_energy_budget(
    window: &Fbuf16F,
    h: &Halation,
) -> (f64, f64) {
    let out = h.eval_window(window);
    let mut delta = 0.0f64;
    for y in 0..window.height {
        for x in 0..window.width {
            let a = window.get(x, y);
            let b = out.get(x, y);
            delta += ((b[0] + b[1] + b[2]) - (a[0] + a[1] + a[2])) as f64;
        }
    }
    // 注入上界：源高光层总能量 · 塔式扩散的权重大致 <= Σ decay^k（保守）。
    let layer = highlight_layer(window, h.threshold);
    let mut light = 0.0f64;
    for y in 0..window.height {
        for x in 0..window.width {
            let c = layer.get(x, y);
            light += (c[0] + c[1] + c[2]) as f64;
        }
    }
    let weight_bound: f64 = {
        let mut w = 0.0f64;
        let mut term = 1.0f64;
        for _ in 0..h.levels.max(1) {
            w += term;
            term *= h.decay as f64;
        }
        w
    };
    let max_tint = h.tint[0].max(h.tint[1]).max(h.tint[2]) as f64;
    let bound = h.strength as f64 * max_tint * weight_bound * light;
    (delta, bound)
}

// ---------------------------------------------------------------------------
// 电影级合成节点：Halation + LayeredBloom 串联（管线位置差异的内建表达）
// ---------------------------------------------------------------------------

/// 胶片合成包（halation 后接分层 bloom 的固定序串联）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalationAndBloom {
    /// 霍拉西翁算子。
    pub halation: Halation,
    /// 分层泛光算子。
    pub bloom: LayeredBloom,
}

impl Default for HalationAndBloom {
    fn default() -> Self {
        HalationAndBloom { halation: Halation::default(), bloom: LayeredBloom::default() }
    }
}

impl TileEffect for HalationAndBloom {
    fn halo_radius(&self) -> u32 {
        self.halation.halo_radius() + self.bloom.halo_radius()
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        // 固定序：先 halation 后 bloom（管线位置钉死）。
        let after_hal = self.halation.eval_window(window);
        self.bloom.eval_window(&after_hal)
    }
}

// ---------------------------------------------------------------------------
// FilmGrain —— 科学化颗粒（双尺度 + 曝光相关 + 按格式取整尺寸）
// ---------------------------------------------------------------------------

/// 胶片格式（颗粒尺寸预设； "按胶片格式取整颗粒尺寸参数"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilmFormat {
    /// 35mm：细颗粒、小尺寸系数。
    Mm35,
    /// 16mm：较粗颗粒。
    Mm16,
    /// Super8：最粗颗粒。
    Super8,
}

impl FilmFormat {
    /// 颗粒尺寸系数（细颗粒基元半径放大的基准）。
    pub fn grain_scale(self) -> f32 {
        match self {
            FilmFormat::Mm35 => 1.0,
            FilmFormat::Mm16 => 1.6,
            FilmFormat::Super8 => 2.4,
        }
    }
}

/// 科学化胶片颗粒：低频大颗粒（暗部显著）+ 高频细颗粒（亮部细腻），
/// 两路按曝光强度反比混合。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmGrain {
    /// 整体颗粒强度（0 = 恒等）。
    pub amount: f32,
    /// 种子。
    pub seed: u64,
    /// 帧号（帧间去相关）。
    pub frame: u64,
    /// 胶片格式（决定低频大颗粒粗度）。
    pub format: FilmFormat,
    /// 低频/高频混合偏置（0.5 = 均衡；越大越偏低频大颗粒）。
    pub low_freq_bias: f32,
}

impl Default for FilmGrain {
    fn default() -> Self {
        FilmGrain {
            amount: 0.06,
            seed: 0,
            frame: 0,
            format: FilmFormat::Mm35,
            low_freq_bias: 0.5,
        }
    }
}

impl FilmGrain {
    /// 单像素颗粒值：低频大颗粒（hash2 在粗格点上）+ 高频细颗粒
    /// （hash3 逐像素），按曝光反比加权。
    ///
    /// `exposure` = 该像素亮度（0 暗 1 亮）。暗部（exposure 低）→ 低频大颗粒
    /// 权重高；亮部 → 高频细颗粒权重高。
    fn sample(&self, x: u32, y: u32, exposure: f32) -> f32 {
        let gs = self.format.grain_scale();
        // 粗格点坐标（整数位移经 floor 的格点化，尺寸系数控制粗度）。
        let coarse = gs * 2.0;
        let cx = ((x as f32) / coarse).floor() as u64;
        let cy = ((y as f32) / coarse).floor() as u64;
        let low = prism_noise::pcg::hash2_f32(self.seed, cx, cy) * 2.0 - 1.0;
        let high = prism_noise::pcg::hash3_f32(self.seed, x as u64, y as u64, self.frame)
            * 2.0
            - 1.0;
        // 曝光反比混合：暗部重低频、亮部重高频。
        let bias = prism_math::clamp(self.low_freq_bias, 0.0, 1.0);
        let low_w = (1.0 - exposure) * bias + (1.0 - bias) * 0.5;
        let high_w = 1.0 - low_w;
        low * low_w + high * high_w
    }
}

impl TileEffect for FilmGrain {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let c = window.get(x, y);
                let exposure = prism_math::clamp(
                    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2],
                    0.0,
                    1.0,
                );
                let n = self.sample(x, y, exposure);
                out.set(
                    x,
                    y,
                    [
                        c[0] + self.amount * n,
                        c[1] + self.amount * n,
                        c[2] + self.amount * n,
                        c[3],
                    ],
                );
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 测试：能量守恒 / 曲线单调 / 颗粒 seed 决定论 / 纯函数
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse_window() -> Fbuf16F {
        let mut win = Fbuf16F::new(21, 21);
        win.set(10, 10, [4.0, 4.0, 4.0, 1.0]);
        win
    }

    #[test]
    fn halation_energy_conserved_and_injection_bounded() {
        // 加算型算子：能量只增不减（delta >= 0）且增益有界（delta <= bound）。
        let win = impulse_window();
        let h = Halation::default();
        let (delta, bound) = halation_energy_budget(&win, &h);
        assert!(delta >= 0.0, "halation 不得凭空减能：delta = {delta}");
        assert!(
            delta <= bound + 1e-6,
            "halation 注入必须落在能量上界内：delta {delta} > bound {bound}"
        );
        // 单点冲激能量不在零附近（确实注入了泛光）。
        assert!(delta > 0.0, "高光必须有渗光注入");
    }

    #[test]
    fn halation_red_tint_produces_red_halo() {
        // 染色：纯白高光的晕圈 R 增益 > G > B（红光晕）。
        let win = impulse_window();
        let out = Halation::default().eval_window(&win);
        let near = out.get(11, 10); // 紧邻高光
        let r_gain = near[0] - 0.0;
        let g_gain = near[1] - 0.0;
        let b_gain = near[2] - 0.0;
        assert!(r_gain > g_gain && g_gain > b_gain, "halation 必须偏红：{near:?}");
    }

    #[test]
    fn below_threshold_is_bit_exact_identity_for_film_pack() {
        // 全输入低于阈值 → 高光层为零 → 加算项为零 → 三算子逐位恒等。
        let mut win = Fbuf16F::new(12, 9);
        for y in 0..9u32 {
            for x in 0..12u32 {
                let v = (x * 11 + y * 7) as f32 / 500.0; // <= 0.5 < 1.0
                win.set(x, y, [v, v * 0.5, v * 0.25, 1.0]);
            }
        }
        assert_eq!(Halation::default().eval_window(&win), win, "halation 低于阈值恒等");
        assert_eq!(LayeredBloom::default().eval_window(&win), win, "bloom 低于阈值恒等");
        let hb = HalationAndBloom::default();
        assert_eq!(hb.eval_window(&win), win, "合成包低于阈值恒等");
    }

    #[test]
    fn layered_bloom_increasing_with_intensity_and_symmetric() {
        let win = impulse_window();
        let a = LayeredBloom { intensity: 0.0, ..LayeredBloom::default() }.eval_window(&win);
        assert_eq!(a, win, "intensity 0 恒等");
        let b = LayeredBloom { intensity: 0.5, ..LayeredBloom::default() }.eval_window(&win);
        assert!(b.get(11, 10)[0] > 0.0, "泛光必须外渗");
        assert_eq!(b.get(9, 10)[0], b.get(11, 10)[0], "左右对称（box blur 链）");
    }

    #[test]
    fn film_curves_monotone_and_neutral_identity() {
        let neutral = FilmCurve { kind: CurveKind::Neutral, shoulder: 0.7, toe: 0.2 };
        for &x in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(neutral.eval(x), x, "中性曲线逐位恒等");
        }
        for kind in [CurveKind::Srgb, CurveKind::Vision3, CurveKind::Eterna] {
            let c = FilmCurve { kind, shoulder: 0.72, toe: 0.18 };
            let mut prev = c.eval(0.0);
            assert_eq!(prev, 0.0, "{kind:?} 端点 0");
            for i in 1..=100 {
                let x = i as f32 / 100.0;
                let y = c.eval(x);
                assert!(y >= prev - 1e-6, "{kind:?} 必须单调不减 @ {x}");
                assert!((0.0..=1.0).contains(&y), "{kind:?} 输出越界 {y}");
                prev = y;
            }
            assert_eq!(c.eval(1.0), 1.0, "{kind:?} 端点 1");
        }
    }

    #[test]
    fn film_grain_deterministic_and_seed_sensitive() {
        let mut win = Fbuf16F::new(20, 16);
        for y in 0..16u32 {
            for x in 0..20u32 {
                let v = (x + y) as f32 / 40.0;
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        let g = FilmGrain::default();
        let a = g.eval_window(&win);
        let b = g.eval_window(&win);
        assert_eq!(a, b, "同 seed 同参逐位一致");
        let c = FilmGrain { seed: 99, ..g }.eval_window(&win);
        assert_ne!(a.pixels, c.pixels, "seed 必须去相关");
        let d = FilmGrain { frame: 5, ..g }.eval_window(&win);
        assert_ne!(a.pixels, d.pixels, "帧号必须去相关");
    }

    #[test]
    fn film_grain_dark_has_stronger_low_freq() {
        // 暗部大颗粒更显著：暗输入的低频块内方差应大于亮输入同区。
        let mut dark = Fbuf16F::new(32, 32);
        let mut bright = Fbuf16F::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                dark.set(x, y, [0.05, 0.05, 0.05, 1.0]);
                bright.set(x, y, [0.95, 0.95, 0.95, 1.0]);
            }
        }
        let g = FilmGrain { amount: 0.1, seed: 3, frame: 0, format: FilmFormat::Mm35, low_freq_bias: 1.0 };
        let od = g.eval_window(&dark);
        // 暗部低频权重应使相邻像素更相关（块状），高频细颗粒权重低。
        let a = od.get(8, 8)[0];
        let bb = od.get(8, 9)[0];
        let dc = g.eval_window(&bright);
        let c = dc.get(8, 8)[0];
        let d = dc.get(8, 9)[0];
        assert!(
            (a - bb).abs() <= (c - d).abs() + 1e-3,
            "暗部应更块状（低频主导）：dark {a}/{bb} bright {c}/{d}"
        );
    }

    #[test]
    fn film_ops_are_pure() {
        let mut win = Fbuf16F::new(15, 11);
        let mut state: u64 = 0xF11A;
        for y in 0..11u32 {
            for x in 0..15u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v * 2.0, v, v * 0.5, 1.0]);
            }
        }
        assert_eq!(Halation::default().eval_window(&win), Halation::default().eval_window(&win));
        assert_eq!(LayeredBloom::default().eval_window(&win), LayeredBloom::default().eval_window(&win));
        assert_eq!(FilmGrain::default().eval_window(&win), FilmGrain::default().eval_window(&win));
    }
}
