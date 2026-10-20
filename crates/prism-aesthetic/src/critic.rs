//! critic 闭环（C5）：对渲染结果做**可计算**质量评估。
//!
//! 语义权威：C5 行（analyze / critique / suggest /
//! apply + `--autofix`（走增量渲染）；"critic 内建为引擎命令且接增量渲染"）、
//! 合成测试帧诊断命中 100%；`--autofix` 2 轮评分回升；
//! 像素内固定序归约——逐像素求和顺序确定；D-16（Mini
//! Code）、D-29（禁 libm：只用 `+ - × ÷` 与 IEEE 精确的 `floor` / `sqrt` /
//! `abs`）；D-30 无关。
//!
//! # 四维可计算指标（全部纯函数、逐帧确定性）
//!
//! | 指标 | 计算本质 | 诊断阈值 |
//! |------|----------|----------|
//! | 曝光 exposure | 平均对数亮度（几何均值）偏离中灰 0.18 | 过暗 / 过亮 |
//! | 对比 contrast | 亮度标准差（线性域） | 平板 / 过冲 |
//! | 色彩 color | 平均饱和度 + 色偏（通道均值差） | 灰淡 / 偏色 |
//! | 清晰度 sharpness | 拉普拉斯梯度能量（高频） | 过软 / 过锐 |
//! | 构图 composition | 三分线能量占比 + 主体居中（重心） | 主体偏边 / 空 |
//!
//! # 评分
//!
//! 每维由"目标值 + 容差"映射到 `[0,1]`（[`score_dim`]：容差内 1，容差外随
//! 距离线性衰减）。总分 = 各维**等权算术平均**（固定顺序求和 ⇒ 逐位可复现）。
//!
//! # 诊断命中（判据：诊断命中 100%）
//!
//! 每个指标越界即产出**具名诊断**（[`Diagnosis`]），携带维度、方向与置信
//! （越界幅度归一）。合成测试帧（纯黑 / 纯白 / 灰卡 / 过锐棋盘）的诊断集合
//! 与期望**逐一对应**——这是"命中 100%"的可测口径。

use prism_io::fbuf::Fbuf16F;

use crate::luma;

/// 中灰参考亮度（线性域，0.18 参考白）。
pub const MID_GRAY: f32 = 0.18;

/// 质量维度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// 曝光（平均亮度是否偏离中灰）。
    Exposure,
    /// 对比（亮度动态范围）。
    Contrast,
    /// 色彩（饱和度 + 色偏）。
    Color,
    /// 清晰度（高频细节）。
    Sharpness,
    /// 构图（主体分布）。
    Composition,
}

impl Dim {
    /// 全部维度（固定顺序；评分求和顺序确定性来源）。
    pub const ALL: [Dim; 5] = [
        Dim::Exposure,
        Dim::Contrast,
        Dim::Color,
        Dim::Sharpness,
        Dim::Composition,
    ];

    /// 维度的短名（诊断 / 报告用；确定性）。
    pub fn name(self) -> &'static str {
        match self {
            Dim::Exposure => "exposure",
            Dim::Contrast => "contrast",
            Dim::Color => "color",
            Dim::Sharpness => "sharpness",
            Dim::Composition => "composition",
        }
    }
}

/// 越界方向（诊断的可行动方向）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// 偏低（如过暗 / 过软 / 欠饱和）。
    Low,
    /// 偏高（如过亮 / 过锐 / 过饱和）。
    High,
}

impl Direction {
    /// 方向短名。
    pub fn name(self) -> &'static str {
        match self {
            Direction::Low => "low",
            Direction::High => "high",
        }
    }
}

/// 一条诊断（维度 + 方向 + 置信），可行动。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnosis {
    /// 越界维度。
    pub dim: Dim,
    /// 越界方向。
    pub direction: Direction,
    /// 越界幅度归一化到 `[0,1]`（0 = 恰在容差边界，1 = 远越界）。
    pub severity: f32,
    /// 实测值（诊断上下文）。
    pub measured: f32,
    /// 目标值（诊断上下文）。
    pub target: f32,
}

/// 四维 + 构图的原值快照（分析与评分共用）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// 几何均值亮度（曝光代理）。
    pub log_mean_luma: f32,
    /// 线性域亮度标准差（对比代理）。
    pub luma_std: f32,
    /// 平均饱和度 `[0,1]`。
    pub mean_saturation: f32,
    /// 通道均值最大差（色偏代理，`[0,1]`；0 = 中性）。
    pub color_cast: f32,
    /// 拉普拉斯能量（清晰度代理；已按像素数归一）。
    pub sharpness: f32,
    /// 三分线能量占比 `[0,1]`（构图代理；越大越贴三分线）。
    pub thirds_energy: f32,
    /// 亮度重心到画面中心的归一化距离 `[0,1]`（0 = 居中）。
    pub centroid_offset: f32,
    /// 有效像素数（诊断 / 统计）。
    pub pixels: u64,
}

/// 目标与容差（评分映射的规格；每维一组）。
///
/// 评分 `score_dim`：`|measured − target| <= tol` → 1；超出后线性衰减到 0
/// （衰减到 `hard` 处为 0，之后钳 0）。容差外仍给部分分，鼓励"接近"。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Target {
    /// 目标值。
    pub target: f32,
    /// 零损失容差（此范围内满分为 1）。
    pub tol: f32,
    /// 归零距离（超过 `tol` 后线性衰减，到此距离得 0）。
    pub hard: f32,
}

impl Target {
    /// 构造目标规格。
    pub fn new(target: f32, tol: f32, hard: f32) -> Self {
        Self { target, tol, hard }
    }
}

/// 评分规格：五个维度的目标（顺序与 [`Dim::ALL`] 对齐）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rubric {
    /// 曝光目标（log 亮度偏离中灰）。
    pub exposure: Target,
    /// 对比目标（亮度标准差）。
    pub contrast: Target,
    /// 色彩目标（饱和度；色偏用独立硬阈，见 [`Rubric::color_cast_max`]）。
    pub color: Target,
    /// 清晰度目标（拉普拉斯能量）。
    pub sharpness: Target,
    /// 构图目标（三分线能量占比）。
    pub composition: Target,
    /// 色偏硬上限（超过即判偏色，扣除色彩分）。
    pub color_cast_max: f32,
}

impl Default for Rubric {
    /// 默认评分规格（中性、宽容，供通用场景起步；调用方可覆写）。
    fn default() -> Self {
        Self {
            // 曝光：log 亮度目标 = ln(0.18) 附近；容差 0.6（≈±0.8EV）。
            exposure: Target::new(0.18f32.ln_approx(), 0.6, 2.4),
            // 对比：线性亮度 std 目标 0.18，容差 0.08。
            contrast: Target::new(0.18, 0.08, 0.32),
            // 色彩：平均饱和度目标 0.35，容差 0.12。
            color: Target::new(0.35, 0.12, 0.45),
            // 清晰度：拉普拉斯能量目标 0.02，容差 0.01。
            sharpness: Target::new(0.02, 0.01, 0.05),
            // 构图：三分线能量占比目标 0.5，容差 0.2。
            composition: Target::new(0.5, 0.2, 0.5),
            color_cast_max: 0.12,
        }
    }
}

/// 一维评分映射：容差内 1，容差外线性衰减到 `hard` 处 0。
pub fn score_dim(measured: f32, t: &Target) -> f32 {
    let d = (measured - t.target).abs();
    if d <= t.tol {
        return 1.0;
    }
    let span = t.hard - t.tol;
    if span <= 0.0 {
        return 0.0;
    }
    let s = 1.0 - (d - t.tol) / span;
    prism_math::clamp(s, 0.0, 1.0)
}

/// `ln` 的确定性近似（D-29 禁 libm）——本 crate 局部实现。
///
/// 目标用途仅是把曝光目标从中灰线性值换到 log 域，数值精度要求低（容差 0.6）。
/// 采用 `ln(x) = 2·atanh((x-1)/(x+1))` 的多项式截断：对 `x ∈ (0, 2]` 用
/// 二次项截断，超出范围按 2 的幂缩放（`x = m·2^k`，`ln x = ln m + k·ln2`）。
trait LnApprox {
    /// 对数近似（自然对数）。
    fn ln_approx(self) -> f32;
}

impl LnApprox for f32 {
    fn ln_approx(self) -> f32 {
        if self <= 0.0 {
            return f32::NEG_INFINITY;
        }
        // 归一到 m ∈ [1, 2)：k 为 2 的幂次。
        let mut m = self;
        let mut k = 0i32;
        while m >= 2.0 {
            m *= 0.5;
            k += 1;
        }
        while m < 1.0 {
            m *= 2.0;
            k -= 1;
        }
        // t = (m-1)/(m+1) ∈ [0, 1/3)，ln m = 2·(t + t³/3 + t⁵/5)。
        let t = (m - 1.0) / (m + 1.0);
        let t2 = t * t;
        let ln_m = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0)));
        // D-30：与 `std::f32::consts::LN_2` `to_bits()` 逐位相同，替换零行为变化。
        const LN2: f32 = std::f32::consts::LN_2;
        ln_m + (k as f32) * LN2
    }
}

/// `sqrt`（IEEE 754 精确，非 libm 超越函数，D-29 允许）。
#[inline]
fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

/// 分析一帧，计算全部指标（纯函数；逐像素固定序归约）。
///
/// 空帧（`width==0 || height==0`）→ 返回全零指标（不 panic）。
pub fn analyze(fb: &Fbuf16F) -> Metrics {
    let w = fb.width as usize;
    let h = fb.height as usize;
    let n = w * h;
    if n == 0 {
        return Metrics {
            log_mean_luma: f32::NEG_INFINITY,
            luma_std: 0.0,
            mean_saturation: 0.0,
            color_cast: 0.0,
            sharpness: 0.0,
            thirds_energy: 0.0,
            centroid_offset: 1.0,
            pixels: 0,
        };
    }

    // 固定序累加（行主序）：亮度、饱和度、通道均值、重心、对数亮度。
    let mut sum_luma = 0.0f64;
    let mut sum_luma_sq = 0.0f64;
    let mut sum_log = 0.0f64;
    let mut sum_sat = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut wsum = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;

    // 三分线带（|x - 1/3| 或 |x - 2/3| 落在带宽内）能量。
    let third = 1.0f32 / 3.0;
    let band = 0.08f32;
    let mut third_luma = 0.0f64;
    let mut total_luma = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let c = fb.get(x as u32, y as u32);
            let l = luma(c);
            let lg = (l + 1e-6).ln_approx();
            sum_luma += l as f64;
            sum_luma_sq += (l * l) as f64;
            sum_log += lg as f64;
            let mx = c[0].max(c[1]).max(c[2]);
            let mn = c[0].min(c[1]).min(c[2]);
            let sat = if mx > 1e-6 { (mx - mn) / mx } else { 0.0 };
            sum_sat += sat as f64;
            sum_r += c[0] as f64;
            sum_g += c[1] as f64;
            sum_b += c[2] as f64;
            wsum += l as f64;
            cx += (x as f64) * (l as f64);
            cy += (y as f64) * (l as f64);
            // 三分线：像素 x 或 y 落在三分带内则计入能量。
            let fx = (x as f32 + 0.5) / (w as f32);
            let fy = (y as f32 + 0.5) / (h as f32);
            let on_third = (fx - third).abs() <= band
                || (fx - 2.0 * third).abs() <= band
                || (fy - third).abs() <= band
                || (fy - 2.0 * third).abs() <= band;
            total_luma += l as f64;
            if on_third {
                third_luma += l as f64;
            }
        }
    }

    let inv = 1.0 / (n as f64);
    let mean = sum_luma * inv;
    let var = (sum_luma_sq * inv - mean * mean).max(0.0);
    let std = sqrtf(var as f32);
    let log_mean = (sum_log * inv) as f32;
    let mean_sat = (sum_sat * inv) as f32;
    let mr = (sum_r * inv) as f32;
    let mg = (sum_g * inv) as f32;
    let mb = (sum_b * inv) as f32;
    let color_cast = (mr - mg).abs().max((mg - mb).abs()).max((mr - mb).abs());

    // 重心偏移（归一化到 [0,1]）。
    let (cx_n, cy_n) = if wsum > 1e-9 {
        (cx / wsum / (w as f64), cy / wsum / (h as f64))
    } else {
        (0.5, 0.5)
    };
    let dx = (cx_n as f32 - 0.5).abs();
    let dy = (cy_n as f32 - 0.5).abs();
    // 到中心的归一化距离（对角一半为 1）。
    let centroid_offset = prism_math::clamp(sqrtf(dx * dx + dy * dy) / 0.5, 0.0, 1.0);

    // 拉普拉斯能量：3×3 核 [0 1 0;1 -4 1;0 1 0]，边缘钳制。
    let mut lap_energy = 0.0f64;
    let mut lap_count = 0u64;
    if w >= 3 && h >= 3 {
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let c = luma(fb.get(x as u32, y as u32));
                let up = luma(fb.get(x as u32, (y - 1) as u32));
                let dn = luma(fb.get(x as u32, (y + 1) as u32));
                let lf = luma(fb.get((x - 1) as u32, y as u32));
                let rt = luma(fb.get((x + 1) as u32, y as u32));
                let lap = up + dn + lf + rt - 4.0 * c;
                lap_energy += (lap * lap) as f64;
                lap_count += 1;
            }
        }
    }
    let sharpness = if lap_count > 0 {
        (lap_energy / lap_count as f64) as f32
    } else {
        0.0
    };

    let thirds_energy = if total_luma > 1e-9 {
        prism_math::clamp((third_luma / total_luma) as f32, 0.0, 1.0)
    } else {
        0.0
    };

    Metrics {
        log_mean_luma: log_mean,
        luma_std: std,
        mean_saturation: mean_sat,
        color_cast,
        sharpness,
        thirds_energy,
        centroid_offset,
        pixels: n as u64,
    }
}

/// 依据指标与评分规格产出诊断集合（固定顺序：按 [`Dim::ALL`]）。
///
/// 越界判据：`|measured − target| > tol`（与 [`score_dim`] 口径一致）。
/// 每维至多一条诊断（取越界主方向）；色偏超 `color_cast_max` 追加一条色彩
/// 诊断。返回集合**顺序确定**（维度序），供"命中 100%"比对。
pub fn diagnose(m: &Metrics, r: &Rubric) -> Vec<Diagnosis> {
    let mut out: Vec<Diagnosis> = Vec::new();
    // 各维在指标里的实际"测量量"。
    let measured = [
        m.log_mean_luma,
        m.luma_std,
        m.mean_saturation,
        m.sharpness,
        m.thirds_energy,
    ];
    let targets = [
        r.exposure, r.contrast, r.color, r.sharpness, r.composition,
    ];

    for (i, dim) in Dim::ALL.iter().enumerate() {
        let t = targets[i];
        let mv = measured[i];
        let d = (mv - t.target).abs();
        if d > t.tol {
            let direction = if mv < t.target { Direction::Low } else { Direction::High };
            let span = (t.hard - t.tol).max(1e-6);
            let severity = prism_math::clamp((d - t.tol) / span, 0.0, 1.0);
            out.push(Diagnosis {
                dim: *dim,
                direction,
                severity,
                measured: mv,
                target: t.target,
            });
        }
    }

    // 色偏独立硬阈：超限追加色彩诊断（色偏是"偏色"，非饱和度问题）。
    if m.color_cast > r.color_cast_max {
        let severity =
            prism_math::clamp((m.color_cast - r.color_cast_max) / r.color_cast_max, 0.0, 1.0);
        out.push(Diagnosis {
            dim: Dim::Color,
            direction: Direction::High,
            severity,
            measured: m.color_cast,
            target: r.color_cast_max,
        });
    }

    out
}

/// 评分结果：总分 + 各维分 + 诊断集合。
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// 总分 `[0,1]`（各维等权算术平均，固定顺序求和）。
    pub total: f32,
    /// 各维分（顺序与 [`Dim::ALL`] 一致）。
    pub dims: [f32; 5],
    /// 诊断集合（固定顺序）。
    pub diagnoses: Vec<Diagnosis>,
}

impl Score {
    /// 取某维分。
    pub fn dim(&self, d: Dim) -> f32 {
        let i = match d {
            Dim::Exposure => 0,
            Dim::Contrast => 1,
            Dim::Color => 2,
            Dim::Sharpness => 3,
            Dim::Composition => 4,
        };
        self.dims[i]
    }
}

/// 评分：指标 × 规格 → [`Score`]。
///
/// 总分 = 五维等权平均（**固定顺序** `Dim::ALL` 求和 ⇒ 逐位可复现）；色偏
/// 超限时按超出比例线性折减色彩分（并入色彩维，不引入新维）。
pub fn score(m: &Metrics, r: &Rubric) -> Score {
    // 色彩维：饱和度分 × 色偏惩罚（色偏越大，色彩分越低）。
    let sat_score = score_dim(m.mean_saturation, &r.color);
    let cast_penalty = if m.color_cast > r.color_cast_max {
        prism_math::clamp(m.color_cast / r.color_cast_max, 0.0, 2.0)
    } else {
        0.0
    };
    let color_score = prism_math::clamp(sat_score - 0.5 * cast_penalty, 0.0, 1.0);

    let dims = [
        score_dim(m.log_mean_luma, &r.exposure),
        score_dim(m.luma_std, &r.contrast),
        color_score,
        score_dim(m.sharpness, &r.sharpness),
        score_dim(m.thirds_energy, &r.composition),
    ];

    let mut total = 0.0f32;
    for s in dims.iter() {
        total += *s;
    }
    total /= dims.len() as f32;

    Score { total, dims, diagnoses: diagnose(m, r) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造纯色帧。
    fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, c);
            }
        }
        fb
    }

    #[test]
    fn analyze_solid_gray_has_zero_contrast_and_sharpness() {
        let fb = solid(16, 16, [0.18, 0.18, 0.18, 1.0]);
        let m = analyze(&fb);
        assert!((m.luma_std).abs() < 1e-6, "纯色 → 对比 0");
        assert!(m.sharpness.abs() < 1e-9, "纯色 → 清晰度 0");
        assert!(m.color_cast.abs() < 1e-6, "中性灰 → 色偏 0");
        assert!(m.mean_saturation < 1e-6, "中性灰 → 饱和度 0");
        assert_eq!(m.pixels, 256);
    }

    #[test]
    fn analyze_exposure_is_monotone_in_brightness() {
        let dark = analyze(&solid(8, 8, [0.01, 0.01, 0.01, 1.0]));
        let mid = analyze(&solid(8, 8, [0.18, 0.18, 0.18, 1.0]));
        let bright = analyze(&solid(8, 8, [0.9, 0.9, 0.9, 1.0]));
        assert!(dark.log_mean_luma < mid.log_mean_luma);
        assert!(mid.log_mean_luma < bright.log_mean_luma);
        // ln(0.18) ≈ -1.715。
        assert!((mid.log_mean_luma - (0.18f32.ln_approx())).abs() < 0.05);
    }

    #[test]
    fn analyze_detects_color_cast() {
        let fb = solid(8, 8, [0.5, 0.2, 0.2, 1.0]);
        let m = analyze(&fb);
        assert!(m.color_cast > 0.25, "红主导 → 明显色偏：{}", m.color_cast);
        assert!(m.mean_saturation > 0.5, "饱和红 → 高饱和：{}", m.mean_saturation);
    }

    #[test]
    fn sharpness_high_on_checkerboard_low_on_flat() {
        let mut cb = Fbuf16F::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let on = ((x / 1) + (y / 1)) % 2 == 0;
                let v = if on { 1.0 } else { 0.0 };
                cb.set(x, y, [v, v, v, 1.0]);
            }
        }
        let sharp = analyze(&cb);
        let flat = analyze(&solid(16, 16, [0.5, 0.5, 0.5, 1.0]));
        assert!(sharp.sharpness > flat.sharpness);
        assert!(sharp.sharpness > 0.1, "棋盘高频 → 大拉普拉斯能量：{}", sharp.sharpness);
    }

    #[test]
    fn empty_frame_is_safe() {
        let fb = Fbuf16F::new(0, 0);
        let m = analyze(&fb);
        assert_eq!(m.pixels, 0);
        let s = score(&m, &Rubric::default());
        assert!(s.total >= 0.0 && s.total <= 1.0);
    }

    #[test]
    fn score_dim_is_one_in_tolerance_and_zero_beyond_hard() {
        let t = Target::new(10.0, 1.0, 3.0);
        assert_eq!(score_dim(10.0, &t), 1.0);
        assert_eq!(score_dim(11.0, &t), 1.0);
        assert_eq!(score_dim(13.0, &t), 0.0);
        assert_eq!(score_dim(20.0, &t), 0.0);
        // 中点：12.0 → 0.5。
        assert!((score_dim(12.0, &t) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn diagnosis_hits_for_underexposed_gray_card() {
        // 极暗灰卡：曝光偏低 + 对比偏低 + 清晰度偏低。
        let fb = solid(32, 32, [0.005, 0.005, 0.005, 1.0]);
        let m = analyze(&fb);
        let d = diagnose(&m, &Rubric::default());
        let has = |dim: Dim, dir: Direction| d.iter().any(|x| x.dim == dim && x.direction == dir);
        assert!(has(Dim::Exposure, Direction::Low), "应诊断过暗：{d:?}");
        assert!(has(Dim::Contrast, Direction::Low), "应诊断平板：{d:?}");
        // 纯中性灰卡无**色偏**（不产生色偏类色彩诊断）；饱和度为 0 属"欠饱和"
        // （色彩维 Low），这是如实诊断而非偏色。
        let has_cast = d.iter().any(|x| x.dim == Dim::Color && x.direction == Direction::High);
        assert!(!has_cast, "中性灰不应有色偏（High）诊断：{d:?}");
    }

    #[test]
    fn score_is_deterministic_same_input_same_score() {
        let fb = solid(16, 16, [0.4, 0.2, 0.1, 1.0]);
        let r = Rubric::default();
        let a = score(&analyze(&fb), &r);
        let b = score(&analyze(&fb), &r);
        assert_eq!(a, b, "同输入同评分（逐位）");
    }

    #[test]
    fn well_exposed_midgray_scores_higher_than_black() {
        let r = Rubric::default();
        let good = score(&analyze(&solid(32, 32, [0.18, 0.18, 0.18, 1.0])), &r);
        let black = score(&analyze(&solid(32, 32, [0.0, 0.0, 0.0, 1.0])), &r);
        assert!(good.total > black.total, "中灰 > 纯黑：{} vs {}", good.total, black.total);
    }
}
