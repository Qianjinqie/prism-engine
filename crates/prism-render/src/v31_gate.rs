//! V3.1 五条误差判据（**序列级聚合口径**）+ golden 哈希锚口径。
//!
//! # 口径来源（任务书 8.1 / 第零节，2026-11-13 第二次放宽）
//! 五条判据从「逐帧一票否决 × 全序列 AND」改为 **整段序列（36 帧）全部像素
//! 汇总后，对每条判据只算一次聚合量**再与上限比。六个数值上限
//! （8 / 32 / 38 dB / 0.15 / 0.90 / 5%）一个不动，只改统计层级。
//!
//! 判据② PSNR **peak 固定为满量程 255**（旧 `parity_report` 取帧实际峰值，
//! 暗场景同 RMSE 下 dB 被压低），且改单一「序列平均 PSNR ≥ 38 dB」，去掉
//! 旧「逐帧 ≥38 且序列均值 ≥42」双 AND。`parity_of_u8` / `judge_parity`
//! 仍原样复用 `prism_rt::wavefront` 基座（逐位同源另有负例测试钉住），
//! 但判据② 的 dB 由本模块按固定 peak 255 + 确定性 `log10_det`（无 libm）
//! 单独计算。
//!
//! # 五条判据（序列级聚合）
//! 1. ① 全序列所有像素 max-channel delta 汇总 → 99 分位 ≤8、全局 max ≤32；
//! 2. ② 全序列逐帧 dB 的算术均值 ≥38 dB，peak 固定 255；
//! 3. ③ 全序列逐帧亮度分布 L1 总变差 `D = Σ|o_i−g_i|/(2n)` 的均值 ≤0.15；
//! 4. ④ 全序列所有强边缘像素汇总后 `sum_got/sum_gold ≥ 0.90`；
//! 5. ⑤ 全序列所有相邻帧对的异常像素汇总占比 ≤5%。
//!
//! [`compare_frame_2d`] / [`compare_frame_2d_expected`] 仍返回**单帧**报告
//! （供单帧门禁与负例测试），其中 `hist_bin_max_rel` 字段口径已改为该帧的 D；
//! [`V31SequenceAggregate`] 逐帧喂入原始量、[`V31SequenceAggregate::finalize`]
//! 一次性产出整段聚合报告 + [`judge_v31`] 判定。`v31_verify::finalize_verify`
//! 用聚合器，不再逐帧累积 reasons。
//!
//! # 尺寸裁定
//! 设计 §3 的 [`compare_frame`] 没有 width/height 参数，无法做真正的二维
//! 3×3 邻域。为严格保留签名，本模块让 [`compare_frame`] 把帧视为
//! `width = n, height = 1` 的行主序二维视图委托给 [`compare_frame_2d`]
//! （同一套 3×3 Sobel 核，垂直项恒 0）；逐帧 / 序列门禁一律调用带真实尺寸
//! 的入口，这是判据 ④ 的权威口径。

use crate::v31_config::{
    DELTA_HARD, HIST_BINS, STRONG_EDGE_MIN,
};
use prism_rt::wavefront::{
    judge as judge_parity_base, parity_report, ParityGate, ParityReport, ParityVerdict,
};

/// 一帧对一帧的五判据统计量（全部为诊断/判定量，不参与渲染输出）。
///
/// 序列聚合时，同一结构承载整段序列的聚合量：字段语义不变，只是统计层级
/// 从「单帧」升到「整段」。`judge_v31` 对两种来源一视同仁。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V31ErrorReport {
    /// 判据 ①：全帧（或整段）8-bit max-channel delta 的最大值（全局硬上限）。
    pub max_delta: u8,
    /// 判据 ①：每像素 max-channel delta 的 99 分位数。
    pub p99_delta: u8,
    /// 判据 ②：PSNR（dB；peak 固定 255）。序列聚合时为逐帧 dB 算术均值。
    pub psnr_db: f64,
    /// 判据 ③：亮度分布归一化 L1 总变差 `D = Σ|o_i−g_i|/(2n)`。
    /// 序列聚合时为逐帧 D 的算术均值（口径 ≤ `HIST_BIN_REL` = 0.15）。
    /// 字段名为历史兼容保留，度量已由「逐 bin 最大相对偏差」改为 D。
    pub hist_bin_max_rel: f64,
    /// 判据 ④：3×3 Sobel 强边缘幅值保留率（聚合 = Σgot/Σgold）。
    pub edge_retention: f64,
    /// 判据 ⑤：相邻帧差分异常像素占比（无 prev ⇒ 0；聚合 = Σ异常/Σ对数像素）。
    pub temporal_outlier_ratio: f64,
}

/// 判定结果（沿用 `wavefront::ParityVerdict` 的 `pass + reasons + report` 风格）。
#[derive(Debug, Clone, PartialEq)]
pub struct V31Verdict {
    /// 五条判据全过才为 true。
    pub pass: bool,
    /// 判红原因（空 = 全过）。
    pub reasons: Vec<String>,
    /// 统计量原文（报告用）。
    pub report: V31ErrorReport,
}

/// 8-bit RGBA 帧 → f32 线性帧，交给 `wavefront::parity_report` 求 PSNR。
///
/// alpha 恒填 0 且 `channels = 3`（只比较 RGB）。**本函数逐位复用基座，
/// 不是判据② 的 dB 来源**：判据② 用固定 peak 255 的 [`FrameCoreRaw::psnr_db`]。
/// 保留它是为 `judge_parity` 与「逐位同源」负例测试提供基座适配。
pub fn parity_of_u8(golden: &[[u8; 4]], got: &[[u8; 4]]) -> ParityReport {
    // 离线诊断（`PRISM_V31_PARITY_DIAG=1`）：打印最大通道差所在的像素序号与
    // 双方 RGBA。定位"插值把哪个像素算错"时必须知道坐标，靠聚合数值猜是浪费。
    if std::env::var("PRISM_V31_PARITY_DIAG").ok().as_deref() == Some("1") {
        let n: usize = golden.len().min(got.len());
        let mut best: u32 = 0;
        let mut at: usize = 0;
        for i in 0..n {
            for c in 0..3 {
                let d: u32 = (golden[i][c] as i32 - got[i][c] as i32).unsigned_abs();
                if d > best {
                    best = d;
                    at = i;
                }
            }
        }
        let w: usize = std::env::var("PRISM_V31_PARITY_W")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(640);
        eprintln!(
            "[parity-diag] 最大差 {best} 位于像素 #{at} (x={}, y={}) golden={:?} got={:?}",
            at % w,
            at / w,
            &golden[at],
            &got[at]
        );
    }
    let to_f32 = |frame: &[[u8; 4]]| -> Vec<[f32; 4]> {
        frame
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0f32])
            .collect()
    };
    parity_report(&to_f32(golden), &to_f32(got), 3)
}

/// 原样复用 `prism_rt::wavefront::judge`（语义级两项 + 输出级 PSNR 三门）。
///
/// 供需要「f32 线性帧 parity」时直接调用，避免在编排层另写判据。
pub fn judge_parity(report: &ParityReport, gate: &ParityGate) -> ParityVerdict {
    judge_parity_base(report, gate)
}

/// 序列平均 PSNR（固定帧序累加，空序列返回 −∞ ⇒ 必判红，防假绿）。
///
/// 完全一致帧的逐帧 PSNR 是 `+∞`，算术均值同为 `+∞`，满足 ≥ 38 dB。
pub fn sequence_mean_psnr(per_frame_psnr: &[f64]) -> f64 {
    if per_frame_psnr.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut sum: f64 = 0.0;
    for &p in per_frame_psnr {
        sum += p;
    }
    sum / per_frame_psnr.len() as f64
}

/// 确定性 `log10`（与 `wavefront::log10_det` 同源：二进制指数分解 + atanh
/// 级数，**无 libm**，任意平台同输入同输出）。判据② 固定 peak 的 dB 用它算。
fn log10_det(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut m = x;
    let mut e: i32 = 0;
    while m >= 1.0 {
        m *= 0.5;
        e += 1;
    }
    while m < 0.5 {
        m *= 2.0;
        e -= 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = 1.0f64;
    let mut acc = 0.0f64;
    for k in 0..11u32 {
        acc += term / ((2 * k + 1) as f64);
        term *= z2;
    }
    let ln_m = 2.0 * z * acc;
    const LN2: f64 = 0.693_147_180_559_945_3;
    const LN10: f64 = 2.302_585_092_994_046;
    (ln_m + e as f64 * LN2) / LN10
}

/// 满量程峰值（255²），判据② 固定 peak 的归一化基准。
const PEAK255_SQ: f64 = 255.0 * 255.0;

/// 每像素 8-bit max-channel delta（RGB 三通道，忽略 alpha）。
#[inline]
fn pixel_max_delta(a: &[u8; 4], b: &[u8; 4]) -> u8 {
    let d0: i32 = a[0] as i32 - b[0] as i32;
    let d1: i32 = a[1] as i32 - b[1] as i32;
    let d2: i32 = a[2] as i32 - b[2] as i32;
    d0.abs().max(d1.abs()).max(d2.abs()) as u8
}

/// Rec.601 整数亮度（0..255）：`(77R + 150G + 29B) >> 8`，无浮点、跨平台确定。
#[inline]
fn luma8(p: &[u8; 4]) -> u32 {
    (77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8
}

/// 亮度 → 64 bin（每 bin 4 级；上界钳到 63）。
#[inline]
fn luma_bin(l: u32) -> usize {
    ((l >> 2) as usize).min(HIST_BINS - 1)
}

/// 3×3 Sobel 的 L1 梯度幅值（8-bit 亮度域，边界按 clamp 复制）。
///
/// 幅值 `= |Gx| + |Gy|`，范围 0..2040；这是判据 ④ 的「强边缘」度量。
#[inline]
fn sobel_l1(buf: &[u32], w: usize, h: usize, x: usize, y: usize) -> u32 {
    let wi: i32 = w as i32;
    let hi: i32 = h as i32;
    let at = |xi: i32, yi: i32| -> u32 {
        let cx: usize = xi.clamp(0, wi - 1) as usize;
        let cy: usize = yi.clamp(0, hi - 1) as usize;
        buf[cy * w + cx]
    };
    let xi: i32 = x as i32;
    let yi: i32 = y as i32;
    let right: i32 =
        (at(xi + 1, yi - 1) + 2 * at(xi + 1, yi) + at(xi + 1, yi + 1)) as i32;
    let left: i32 =
        (at(xi - 1, yi - 1) + 2 * at(xi - 1, yi) + at(xi - 1, yi + 1)) as i32;
    let bottom: i32 =
        (at(xi - 1, yi + 1) + 2 * at(xi, yi + 1) + at(xi + 1, yi + 1)) as i32;
    let top: i32 =
        (at(xi - 1, yi - 1) + 2 * at(xi, yi - 1) + at(xi + 1, yi - 1)) as i32;
    ((right - left).abs() + (bottom - top).abs()) as u32
}

/// 从 delta 直方图取 99 分位（整数运算避免浮点边界歧义）。
fn p99_from_hist(hist: &[u64; 256], n: u64) -> u8 {
    if n == 0 {
        return 0;
    }
    let target: u64 = (n * 99 + 99) / 100;
    let mut cum: u64 = 0;
    for d in 0..=255usize {
        cum += hist[d];
        if cum >= target {
            return d as u8;
        }
    }
    255
}

/// 一帧的「原始聚合量」：判据 ①②③④ 的可累加分量 + 判据⑤ 的分子分母。
///
/// 单帧报告 [`compare_frame_2d`] 与整段聚合 [`V31SequenceAggregate`] 共用
/// 同一套原始量构造，杜绝「单帧口径」与「序列口径」度量漂移。
struct FrameCoreRaw {
    /// 判据①：0..255 的 max-channel delta 直方图（全帧像素计数）。
    delta_hist: [u64; 256],
    /// 判据①：本帧像素总数。
    n: u64,
    /// 判据①：本帧 delta 全局最大值。
    max_delta: u8,
    /// 判据②：RGB 三通道 Σ(a−b)²（与 `parity_report` 的 `sum_sq_err` 同款）。
    sum_sq_err: f64,
    /// 判据②：Σ(a−b)² 的通道计数（= 3n）。
    sq_count: u64,
    /// 判据③：Σ_i |o_i − g_i|（64 个亮度 bin 的计数绝对差之和）。
    l1_abs: u64,
    /// 判据④：全帧强边缘（golden）Sobel 幅值之和。
    edge_gold: u64,
    /// 判据④：同一批强边缘像素在 got 上的 Sobel 幅值之和。
    edge_got: u64,
}

impl FrameCoreRaw {
    /// 判据②：固定 peak 255 的逐帧 dB = `10·log10(255² / mse)`。
    fn psnr_db(&self) -> f64 {
        if self.sq_count == 0 {
            return f64::INFINITY;
        }
        let mse: f64 = self.sum_sq_err / self.sq_count as f64;
        if mse <= 0.0 {
            f64::INFINITY
        } else {
            10.0 * log10_det(PEAK255_SQ / mse)
        }
    }

    /// 判据③：归一化 L1 总变差 `D = Σ|o_i−g_i|/(2n)`，天然 0..1。
    fn d(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.l1_abs as f64 / (2.0 * self.n as f64)
    }

    /// 判据④：强边缘幅值保留率（全平帧 ⇒ 无边缘可丢 ⇒ 1.0）。
    fn edge_retention(&self) -> f64 {
        if self.edge_gold == 0 {
            1.0
        } else {
            self.edge_got as f64 / self.edge_gold as f64
        }
    }

    /// 组装一份 [`V31ErrorReport`]（单帧或整段聚合共用）。
    fn to_report(&self, temporal_outliers: u64, temporal_total: u64) -> V31ErrorReport {
        let temporal_outlier_ratio: f64 = if temporal_total == 0 {
            0.0
        } else {
            temporal_outliers as f64 / temporal_total as f64
        };
        V31ErrorReport {
            max_delta: self.max_delta,
            p99_delta: p99_from_hist(&self.delta_hist, self.n),
            psnr_db: self.psnr_db(),
            hist_bin_max_rel: self.d(),
            edge_retention: self.edge_retention(),
            temporal_outlier_ratio,
        }
    }
}

/// 计算一帧的判据 ①②③④ 原始量（判据⑤ 由调用方按口径另算）。
///
/// # Panics（结构化报错，不可静默关）
/// - `golden` / `got` 长度不等；
/// - `width × height` 与帧长不符；
/// - 空帧（无可比较像素 ⇒ P0-6 硬失败，防止「空集全过」假绿）。
fn frame_core_raw(
    golden: &[[u8; 4]],
    got: &[[u8; 4]],
    width: u32,
    height: u32,
) -> FrameCoreRaw {
    assert_eq!(
        golden.len(),
        got.len(),
        "V31 判据：golden / got 帧长不等（golden={}, got={}）",
        golden.len(),
        got.len()
    );
    let n: usize = golden.len();
    assert!(n > 0, "V31 判据：空帧不可比较（P0-6 golden 资产缺失，禁止静默通过）");
    assert_eq!(
        (width as usize).saturating_mul(height as usize),
        n,
        "V31 判据：width×height={}×{} 与帧长 {} 不符",
        width,
        height,
        n
    );
    let w: usize = width as usize;
    let h: usize = height as usize;

    let mut delta_hist: [u64; 256] = [0; 256];
    let mut max_delta: u8 = 0;
    let mut sum_sq_err: f64 = 0.0;
    let mut sq_count: u64 = 0;
    let mut hg: [u64; HIST_BINS] = [0; HIST_BINS];
    let mut ho: [u64; HIST_BINS] = [0; HIST_BINS];
    let mut gold_luma: Vec<u32> = Vec::with_capacity(n);
    let mut got_luma: Vec<u32> = Vec::with_capacity(n);

    for (g, o) in golden.iter().zip(got.iter()) {
        let d: u8 = pixel_max_delta(g, o);
        delta_hist[d as usize] += 1;
        if d > max_delta {
            max_delta = d;
        }
        for c in 0..3 {
            let diff: f64 = g[c] as f64 - o[c] as f64;
            sum_sq_err += diff * diff;
            sq_count += 1;
        }
        let lg: u32 = luma8(g);
        let lo: u32 = luma8(o);
        hg[luma_bin(lg)] += 1;
        ho[luma_bin(lo)] += 1;
        gold_luma.push(lg);
        got_luma.push(lo);
    }

    // 判据③：Σ_i |o_i − g_i|（64 bin 计数绝对差）。
    let mut l1_abs: u64 = 0;
    for i in 0..HIST_BINS {
        l1_abs += ho[i].abs_diff(hg[i]);
    }

    // 判据④：3×3 Sobel 强边缘幅值汇总。
    let mut edge_gold: u64 = 0;
    let mut edge_got: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let gg: u32 = sobel_l1(&gold_luma, w, h, x, y);
            if gg >= STRONG_EDGE_MIN {
                edge_gold += gg as u64;
                edge_got += sobel_l1(&got_luma, w, h, x, y) as u64;
            }
        }
    }

    FrameCoreRaw {
        delta_hist,
        n: n as u64,
        max_delta,
        sum_sq_err,
        sq_count,
        l1_abs,
        edge_gold,
        edge_got,
    }
}

/// 判据⑤（原始口径）：`got` 相对前一帧 `got` 的 max-channel delta > 32 的
/// 像素数与比较总数。无 prev ⇒ (0, 0)。
fn temporal_raw(got: &[[u8; 4]], prev: Option<&[[u8; 4]]>) -> (u64, u64) {
    match prev {
        None => (0, 0),
        Some(p) if p.len() != got.len() => (0, 0),
        Some(p) => {
            let mut outliers: u64 = 0;
            for (o, q) in got.iter().zip(p.iter()) {
                if pixel_max_delta(o, q) as u32 > DELTA_HARD as u32 {
                    outliers += 1;
                }
            }
            (outliers, got.len() as u64)
        }
    }
}

/// 判据⑤（时域重投影验收口径）：「`got` 的帧间变化是否偏离 ground truth
/// 的帧间变化」。`|(got_cur−got_prev) − (golden_cur−golden_prev)| > 32` 的
/// 像素数与比较总数。任一凭证不齐 ⇒ (0, 0)。
fn temporal_expected(
    golden: &[[u8; 4]],
    got: &[[u8; 4]],
    prev_golden: Option<&[[u8; 4]]>,
    prev_got: Option<&[[u8; 4]]>,
) -> (u64, u64) {
    let n: usize = golden.len();
    match (prev_golden, prev_got) {
        (Some(pg), Some(po)) if pg.len() == n && po.len() == n => {
            let mut outliers: u64 = 0;
            for i in 0..n {
                let g = golden[i];
                let o = got[i];
                let pgp = pg[i];
                let pop = po[i];
                let mut d: i32 = 0;
                for c in 0..3 {
                    let dg: i32 = o[c] as i32 - pop[c] as i32;
                    let dx: i32 = g[c] as i32 - pgp[c] as i32;
                    d = d.max((dg - dx).abs());
                }
                if d as u32 > DELTA_HARD as u32 {
                    outliers += 1;
                }
            }
            (outliers, n as u64)
        }
        _ => (0, 0),
    }
}

/// 设计 §3 签名：无尺寸参数时的五判据统计。
///
/// 委托 [`compare_frame_2d`] 并把帧视为 `n × 1`（行主序一维视图）；判据 ④
/// 在这一视图下退化为水平 3×3 Sobel。逐帧门禁请用带尺寸的 [`compare_frame_2d`]。
pub fn compare_frame(
    golden: &[[u8; 4]],
    got: &[[u8; 4]],
    prev: Option<&[[u8; 4]]>,
) -> V31ErrorReport {
    let n: usize = golden.len();
    compare_frame_2d(golden, got, prev, n as u32, 1)
}

/// 带真实尺寸的**单帧**五判据统计（判据 ④ 的权威 3×3 口径）。
///
/// 判据② dB 用固定 peak 255，判据③ 用 L1 总变差 D，判据⑤ 用原始 prev 口径。
pub fn compare_frame_2d(
    golden: &[[u8; 4]],
    got: &[[u8; 4]],
    prev: Option<&[[u8; 4]]>,
    width: u32,
    height: u32,
) -> V31ErrorReport {
    let core: FrameCoreRaw = frame_core_raw(golden, got, width, height);
    let (o, t): (u64, u64) = temporal_raw(got, prev);
    core.to_report(o, t)
}

/// 带「期望时序变化」的**单帧**五判据统计（判据 ⑤ 权威口径）。
///
/// ①②③④ 与 [`compare_frame_2d`] 同源；判据⑤ 改为
/// `|(got_cur−got_prev) − (golden_cur−golden_prev)| > 32` 的像素占比，
/// 真运动不误判、重投影闪烁才判红。
pub fn compare_frame_2d_expected(
    golden: &[[u8; 4]],
    got: &[[u8; 4]],
    prev_golden: Option<&[[u8; 4]]>,
    prev_got: Option<&[[u8; 4]]>,
    width: u32,
    height: u32,
) -> V31ErrorReport {
    let core: FrameCoreRaw = frame_core_raw(golden, got, width, height);
    let (o, t): (u64, u64) = temporal_expected(golden, got, prev_golden, prev_got);
    core.to_report(o, t)
}

/// 整段序列聚合器：把 36 帧全部像素汇总后，对五条判据各算一次聚合量。
///
/// - ①：合并全序列 delta 直方图 → 一次 99 分位 + 一次全局 max；
/// - ②：逐帧 dB（peak 255）算术均值；
/// - ③：逐帧 D 算术均值；
/// - ④：Σedge_got / Σedge_gold（全序列强边缘像素汇总）；
/// - ⑤：Σ相邻帧异常像素 / Σ相邻帧比较像素。
///
/// `push_expected` 用判据⑤ 的「期望时序变化」口径（真运动片源权威口径）。
/// `finalize` 调一次 [`judge_v31`]，reasons 只来自这一次判定，不逐帧累积。
pub struct V31SequenceAggregate {
    delta_hist: [u64; 256],
    delta_total: u64,
    max_delta: u8,
    /// 逐帧 dB（peak 255），供序列均值。
    frame_dbs: Vec<f64>,
    /// 逐帧 D，供序列均值。
    frame_ds: Vec<f64>,
    edge_gold: u64,
    edge_got: u64,
    temporal_outliers: u64,
    temporal_total: u64,
    frames: u32,
}

impl V31SequenceAggregate {
    /// 构造空聚合器。
    pub fn new() -> Self {
        V31SequenceAggregate {
            delta_hist: [0; 256],
            delta_total: 0,
            max_delta: 0,
            frame_dbs: Vec::new(),
            frame_ds: Vec::new(),
            edge_gold: 0,
            edge_got: 0,
            temporal_outliers: 0,
            temporal_total: 0,
            frames: 0,
        }
    }

    /// 已喂入的帧数。
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// 逐帧最差 dB（展示用；空序列 ⇒ −∞，全 inf ⇒ +∞）。
    pub fn worst_psnr_db(&self) -> f64 {
        if self.frame_dbs.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut worst: f64 = f64::INFINITY;
        for &d in &self.frame_dbs {
            worst = worst.min(d);
        }
        worst
    }

    /// 喂入一帧，返回该帧的**单帧**报告（仅供诊断/逐帧明细，不参与判定）。
    ///
    /// 判据⑤ 走「相对 golden 期望时序变化」口径；`prev_golden` / `prev_got`
    /// 由调用方按帧序维护（首帧传 `None`）。
    pub fn push_expected(
        &mut self,
        golden: &[[u8; 4]],
        got: &[[u8; 4]],
        prev_golden: Option<&[[u8; 4]]>,
        prev_got: Option<&[[u8; 4]]>,
        width: u32,
        height: u32,
    ) -> V31ErrorReport {
        let core: FrameCoreRaw = frame_core_raw(golden, got, width, height);
        let (o, t): (u64, u64) = temporal_expected(golden, got, prev_golden, prev_got);
        let frame_report: V31ErrorReport = core.to_report(o, t);

        // ① delta 直方图与全局 max 跨帧合并。
        for d in 0..256usize {
            self.delta_hist[d] += core.delta_hist[d];
        }
        self.delta_total += core.n;
        self.max_delta = self.max_delta.max(core.max_delta);
        // ② 逐帧 dB（peak 255）。
        self.frame_dbs.push(core.psnr_db());
        // ③ 逐帧 D。
        self.frame_ds.push(core.d());
        // ④ 强边缘幅值跨帧汇总。
        self.edge_gold += core.edge_gold;
        self.edge_got += core.edge_got;
        // ⑤ 相邻帧异常跨帧汇总。
        self.temporal_outliers += o;
        self.temporal_total += t;
        self.frames += 1;

        frame_report
    }

    /// 一次性产出整段序列的聚合报告与判定（五条判据各算一次）。
    ///
    /// 上限按档取：`一致` / 诊断用 [`JudgeLimits::STRICT`]，出片链路的
    /// `均衡` / `快速` 由 [`V31SequenceAggregate::finalize_with`] 传入 §8.1
    /// 对应那一组。旧「一套尺子量到底」的口径已按任务书 §8.1 作废。
    pub fn finalize(&self) -> (V31ErrorReport, V31Verdict) {
        self.finalize_with(&crate::v31_config::JudgeLimits::STRICT)
    }

    /// 按指定一档上限产出聚合报告与判定（§8.1 分档口径的唯一出口）。
    pub fn finalize_with(
        &self,
        limits: &crate::v31_config::JudgeLimits,
    ) -> (V31ErrorReport, V31Verdict) {
        // ① 全序列所有像素汇总后的 99 分位与全局 max。
        let p99_delta: u8 = p99_from_hist(&self.delta_hist, self.delta_total);
        // ② 逐帧 dB 的算术均值（固定 peak 255）。
        let seq_mean: f64 = sequence_mean_psnr(&self.frame_dbs);
        // ③ 逐帧 D 的算术均值。
        let d_mean: f64 = if self.frame_ds.is_empty() {
            0.0
        } else {
            self.frame_ds.iter().sum::<f64>() / self.frame_ds.len() as f64
        };
        // ④ 全序列强边缘像素汇总后的保留率。
        let edge_retention: f64 = if self.edge_gold == 0 {
            1.0
        } else {
            self.edge_got as f64 / self.edge_gold as f64
        };
        // ⑤ 全序列相邻帧对异常像素汇总占比。
        let temporal_outlier_ratio: f64 = if self.temporal_total == 0 {
            0.0
        } else {
            self.temporal_outliers as f64 / self.temporal_total as f64
        };
        let report: V31ErrorReport = V31ErrorReport {
            max_delta: self.max_delta,
            p99_delta,
            psnr_db: seq_mean,
            hist_bin_max_rel: d_mean,
            edge_retention,
            temporal_outlier_ratio,
        };
        let verdict: V31Verdict = judge_v31_with(&report, seq_mean, limits);
        (report, verdict)
    }
}

impl Default for V31SequenceAggregate {
    fn default() -> Self {
        Self::new()
    }
}

/// 依据五条判据判定一份 [`V31ErrorReport`]（全部 `and` 语义，任一不过即判红）。
///
/// 判据② 为单一「序列平均 PSNR ≥ 38 dB」（peak 固定 255；报告里的
/// `psnr_db` 已按此口径）。单帧调用时 `seq_mean_psnr` 传该帧 dB 即可，
/// 序列聚合调用时传整段逐帧 dB 均值。旧「逐帧 ≥38 且序列均值 ≥42」双
/// AND 已按任务书 8.1 作废。
pub fn judge_v31(r: &V31ErrorReport, seq_mean_psnr: f64) -> V31Verdict {
    judge_v31_with(r, seq_mean_psnr, &crate::v31_config::JudgeLimits::STRICT)
}

/// 按指定一档上限判定（§8.1 分档口径）：六项上限全部来自 `limits`，
/// 判定语义仍是「全部 and、任一不过即判红」的确定性布尔。
pub fn judge_v31_with(
    r: &V31ErrorReport,
    seq_mean_psnr: f64,
    limits: &crate::v31_config::JudgeLimits,
) -> V31Verdict {
    let mut reasons: Vec<String> = Vec::new();
    let p99_lim: u8 = limits.p99_delta;
    let max_lim: u8 = limits.max_delta;
    let psnr_lim: f64 = limits.psnr_db;
    let l1_lim: f64 = limits.hist_l1;
    let edge_lim: f64 = limits.edge_keep;
    let temp_lim: f64 = limits.temporal;
    if !(r.p99_delta <= p99_lim) {
        reasons.push(format!(
            "判据①：99% 像素 delta 分位 {} > 上限 {}（8-bit）",
            r.p99_delta, p99_lim
        ));
    }
    if !(r.max_delta <= max_lim) {
        reasons.push(format!(
            "判据①：全局最大 delta {} > 硬上限 {}（8-bit）",
            r.max_delta, max_lim
        ));
    }
    if !(seq_mean_psnr >= psnr_lim) {
        reasons.push(format!(
            "判据②：序列平均 PSNR {:.2} dB < 下限 {:.2} dB（peak 固定 255）",
            seq_mean_psnr, psnr_lim
        ));
    }
    if !(r.hist_bin_max_rel <= l1_lim) {
        reasons.push(format!(
            "判据③：亮度分布 L1 总变差 D {:.4} > 上限 {:.4}",
            r.hist_bin_max_rel, l1_lim
        ));
    }
    if !(r.edge_retention >= edge_lim) {
        reasons.push(format!(
            "判据④：3×3 强边缘幅值保留率 {:.4} < 下限 {:.4}",
            r.edge_retention, edge_lim
        ));
    }
    if !(r.temporal_outlier_ratio <= temp_lim) {
        reasons.push(format!(
            "判据⑤：相邻帧差分异常像素占比 {:.4} > 上限 {:.4}",
            r.temporal_outlier_ratio, temp_lim
        ));
    }
    V31Verdict {
        pass: reasons.is_empty(),
        reasons,
        report: *r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_frame_passes_all_five_criteria() {
        let f = vec![[10u8, 20, 30, 255]; 64];
        let r = compare_frame_2d(&f, &f, None, 8, 8);
        assert_eq!(r.max_delta, 0);
        assert_eq!(r.p99_delta, 0);
        assert!(r.psnr_db.is_infinite() && r.psnr_db > 0.0);
        assert_eq!(r.hist_bin_max_rel, 0.0);
        assert_eq!(r.edge_retention, 1.0);
        assert_eq!(r.temporal_outlier_ratio, 0.0);
        let v = judge_v31(&r, sequence_mean_psnr(&[r.psnr_db]));
        assert!(v.pass, "{:?}", v.reasons);
    }

    #[test]
    fn compare_frame_delegates_to_row_view() {
        let f = vec![[1u8, 2, 3, 255]; 32];
        let a = compare_frame(&f, &f, None);
        let b = compare_frame_2d(&f, &f, None, 32, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn parity_base_is_reused_verbatim() {
        let g = vec![[100u8, 120, 140, 255]; 16];
        let o = vec![[104u8, 120, 140, 255]; 16];
        let got = parity_of_u8(&g, &o);
        let want = parity_report(
            &vec![[100.0f32, 120.0, 140.0, 0.0]; 16],
            &vec![[104.0f32, 120.0, 140.0, 0.0]; 16],
            3,
        );
        assert_eq!(got.psnr_db.to_bits(), want.psnr_db.to_bits());
        assert_eq!(got.mse.to_bits(), want.mse.to_bits());
        let gate = ParityGate::default();
        assert_eq!(judge_parity(&got, &gate), judge_parity_base(&want, &gate));
    }

    #[test]
    fn empty_sequence_psnr_is_negative_infinity() {
        assert_eq!(sequence_mean_psnr(&[]), f64::NEG_INFINITY);
        assert!(!judge_v31(
            &V31ErrorReport {
                max_delta: 0,
                p99_delta: 0,
                psnr_db: 60.0,
                hist_bin_max_rel: 0.0,
                edge_retention: 1.0,
                temporal_outlier_ratio: 0.0,
            },
            f64::NEG_INFINITY,
        )
        .pass);
    }

    #[test]
    fn expected_temporal_passes_real_motion_when_got_tracks_golden() {
        let (w, h) = (8u32, 8u32);
        let golden_prev = vec![[10u8, 20, 30, 255]; 64];
        let mut golden_cur = golden_prev.clone();
        for i in 0..64usize {
            golden_cur[i][0] = 10 + (i as u8 % 7) * 20;
        }
        let got_prev = golden_prev.clone();
        let got_cur = golden_cur.clone();
        let r = compare_frame_2d_expected(
            &golden_cur, &got_cur, Some(&golden_prev), Some(&got_prev), w, h,
        );
        assert_eq!(r.max_delta, 0);
        assert_eq!(r.temporal_outlier_ratio, 0.0, "忠实跟随 golden 的运动不得判异常");
        let v = judge_v31(&r, sequence_mean_psnr(&[r.psnr_db]));
        assert!(v.pass, "{:?}", v.reasons);
    }

    #[test]
    fn expected_temporal_flags_unexpected_flicker() {
        let (w, h) = (8u32, 8u32);
        let golden = vec![[100u8, 100, 100, 255]; 64];
        let mut got = golden.clone();
        got[0] = [200, 200, 200, 255];
        let r = compare_frame_2d_expected(
            &golden, &got, Some(&golden), Some(&golden), w, h,
        );
        assert!(r.temporal_outlier_ratio > 0.0, "got 突变但 golden 未变必须被判时序异常");
        let v = judge_v31(&r, sequence_mean_psnr(&[r.psnr_db]));
        assert!(!v.pass);
    }

    #[test]
    fn expected_temporal_reduces_to_raw_on_static_and_first_frame() {
        let (w, h) = (8u32, 8u32);
        let golden = vec![[50u8, 60, 70, 255]; 64];
        let got = golden.clone();
        let raw = compare_frame_2d(&golden, &got, Some(&got), w, h);
        let exp = compare_frame_2d_expected(
            &golden, &got, Some(&golden), Some(&got), w, h,
        );
        assert_eq!(raw.temporal_outlier_ratio, exp.temporal_outlier_ratio);
        let first = compare_frame_2d_expected(&golden, &got, None, None, w, h);
        assert_eq!(first.temporal_outlier_ratio, 0.0);
    }

    #[test]
    fn fixed_peak_psnr_is_deterministic_and_infinite_on_identical() {
        let g = vec![[10u8, 20, 30, 255]; 16];
        let core_same = frame_core_raw(&g, &g, 4, 4);
        assert!(core_same.psnr_db().is_infinite());
        // 固定 peak 255：整体偏移 8 的 mse = 64/3 ⇒ dB = 10·log10(255²/(64/3))。
        let o: Vec<[u8; 4]> = g.iter().map(|p| [p[0] + 8, p[1], p[2], p[3]]).collect();
        let core = frame_core_raw(&g, &o, 4, 4);
        let mse: f64 = core.sum_sq_err / core.sq_count as f64;
        assert!((mse - 64.0 / 3.0).abs() < 1e-9, "mse={mse}");
        let want: f64 = 10.0 * log10_det(PEAK255_SQ / (64.0 / 3.0));
        assert!((core.psnr_db() - want).abs() < 1e-9, "{} vs {}", core.psnr_db(), want);
    }

    #[test]
    fn l1_total_variation_is_bounded_by_one() {
        let g = vec![[200u8, 200, 200, 255]; 64];
        let o = vec![[0u8, 0, 0, 255]; 64];
        let core = frame_core_raw(&g, &o, 8, 8);
        // 全部像素从高亮 bin 搬到暗 bin：Σ|o_i−g_i| = 2n ⇒ D = 1。
        assert!((core.d() - 1.0).abs() < 1e-12, "D={}", core.d());
    }

    // ---- 序列级聚合：正向 + 负向（负例仍判红，不许靠聚合放水） ----

    /// 含强边缘与宽亮度分布的确定性测试帧。
    fn checker_frame(w: u32, h: u32) -> Vec<[u8; 4]> {
        let mut out: Vec<[u8; 4]> = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let checker: u8 = if (x / 4 + y / 4) % 2 == 0 { 220 } else { 20 };
                let ramp: u8 = ((x * 255) / w.max(1)) as u8;
                let band: u8 = ((y * 255) / h.max(1)) as u8;
                out.push([checker, ramp, 255 - band, 255]);
            }
        }
        out
    }

    fn contrast_compress(f: &[[u8; 4]]) -> Vec<[u8; 4]> {
        f.iter()
            .map(|p| {
                let mut q = *p;
                for k in 0..3 {
                    let v: i32 = p[k] as i32;
                    q[k] = (128 + (v - 128) / 2).clamp(0, 255) as u8;
                }
                q
            })
            .collect()
    }

    fn blur_frame(f: &[[u8; 4]], w: u32, h: u32) -> Vec<[u8; 4]> {
        let mut out: Vec<[u8; 4]> = Vec::with_capacity(f.len());
        for y in 0..h {
            for x in 0..w {
                let mut sums: [u32; 3] = [0; 3];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let cx: usize = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let cy: usize = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let p: [u8; 4] = f[cy * w as usize + cx];
                        for k in 0..3 {
                            sums[k] += p[k] as u32;
                        }
                    }
                }
                let mut q: [u8; 4] = [255; 4];
                for k in 0..3 {
                    q[k] = ((sums[k] + 4) / 9) as u8;
                }
                out.push(q);
            }
        }
        out
    }

    /// 逐帧明显变化的真运动 golden（R 通道随帧号摆动）。
    fn motion_golden(w: u32, h: u32, frame: u32) -> Vec<[u8; 4]> {
        let base = checker_frame(w, h);
        base.iter()
            .enumerate()
            .map(|(i, p)| {
                let shift: u8 = ((i as u32 + frame * 37) % 200) as u8;
                [p[0].wrapping_add(shift), p[1], p[2], p[3]]
            })
            .collect()
    }

    #[test]
    fn sequence_aggregate_identical_sequence_passes() {
        let (w, h) = (16u32, 12u32);
        let frame = checker_frame(w, h);
        let mut agg = V31SequenceAggregate::new();
        let mut prev: Option<Vec<[u8; 4]>> = None;
        for _ in 0..6 {
            agg.push_expected(&frame, &frame, prev.as_deref(), prev.as_deref(), w, h);
            prev = Some(frame.clone());
        }
        assert_eq!(agg.frames(), 6);
        let (r, v) = agg.finalize();
        assert!(v.pass, "{:?}", v.reasons);
        assert_eq!(r.max_delta, 0);
        assert!(r.psnr_db.is_infinite());
        assert_eq!(r.hist_bin_max_rel, 0.0);
    }

    #[test]
    fn sequence_aggregate_contrast_compression_fails_l1_tv() {
        let (w, h) = (16u32, 12u32);
        let g = checker_frame(w, h);
        let o = contrast_compress(&g);
        let mut agg = V31SequenceAggregate::new();
        let mut prev_g: Option<Vec<[u8; 4]>> = None;
        let mut prev_o: Option<Vec<[u8; 4]>> = None;
        for _ in 0..6 {
            agg.push_expected(&g, &o, prev_g.as_deref(), prev_o.as_deref(), w, h);
            prev_g = Some(g.clone());
            prev_o = Some(o.clone());
        }
        let (_, v) = agg.finalize();
        assert!(!v.pass, "对比度压缩（宏观色调漂移）必须判红");
        assert!(v.reasons.iter().any(|s| s.contains("判据③")), "{:?}", v.reasons);
    }

    #[test]
    fn sequence_aggregate_blur_fails_edge_retention() {
        let (w, h) = (16u32, 12u32);
        let g = checker_frame(w, h);
        let o = blur_frame(&g, w, h);
        let mut agg = V31SequenceAggregate::new();
        let mut prev_g: Option<Vec<[u8; 4]>> = None;
        let mut prev_o: Option<Vec<[u8; 4]>> = None;
        for _ in 0..6 {
            agg.push_expected(&g, &o, prev_g.as_deref(), prev_o.as_deref(), w, h);
            prev_g = Some(g.clone());
            prev_o = Some(o.clone());
        }
        let (_, v) = agg.finalize();
        assert!(!v.pass, "大面积模糊抹平强边缘必须判红");
        assert!(v.reasons.iter().any(|s| s.contains("判据④")), "{:?}", v.reasons);
    }

    #[test]
    fn sequence_aggregate_frozen_motion_fails_temporal() {
        let (w, h) = (16u32, 12u32);
        let frozen = motion_golden(w, h, 0);
        let mut agg = V31SequenceAggregate::new();
        let mut prev_g: Option<Vec<[u8; 4]>> = None;
        let mut prev_o: Option<Vec<[u8; 4]>> = None;
        for f in 0..6u32 {
            let g = motion_golden(w, h, f);
            agg.push_expected(&g, &frozen, prev_g.as_deref(), prev_o.as_deref(), w, h);
            prev_g = Some(g);
            prev_o = Some(frozen.clone());
        }
        let (_, v) = agg.finalize();
        assert!(!v.pass, "冻结场景（真运动却恒定输出）必须判红");
        assert!(v.reasons.iter().any(|s| s.contains("判据⑤")), "{:?}", v.reasons);
    }

    #[test]
    fn sequence_aggregate_noise_fails_hard_delta() {
        let (w, h) = (16u32, 12u32);
        let g = checker_frame(w, h);
        let mut agg = V31SequenceAggregate::new();
        let mut prev_g: Option<Vec<[u8; 4]>> = None;
        let mut prev_o: Option<Vec<[u8; 4]>> = None;
        for f in 0..6u32 {
            let mut o = g.clone();
            for i in 0..o.len() {
                if (i as u32 + f * 11) % 53 == 0 {
                    o[i][0] = o[i][0].saturating_add(70);
                }
            }
            agg.push_expected(&g, &o, prev_g.as_deref(), prev_o.as_deref(), w, h);
            prev_g = Some(g.clone());
            prev_o = Some(o);
        }
        let (_, v) = agg.finalize();
        assert!(!v.pass, "稀疏噪点（欠采样伪影）必须判红");
        assert!(v.reasons.iter().any(|s| s.contains("判据①")), "{:?}", v.reasons);
    }
}
