//! V3.0 W3：**成对盲测协议**（§3.5 V3 / 主理人裁定）。
//!
//! # 判官不是主观自评，也不是 critic 分数
//!
//! 判官是一个**确定性可计算函数** [`judge`]：只接收两条数值序列，不知道
//! 哪条是 `analytic`（A）、哪条是 `crafted`（B），按三条可复现的
//! 「解析性 / 匀质度」指标强制二选一「哪边解析性更低（更像人搓的）」：
//!
//! 1. **运动二阶差分能量** —— 解析匀速 / 标准 easing 的二阶差分趋 0，
//!    带惯性 / 犹豫 / 微抖 / 呼吸的序列二阶差分能量显著非零；
//! 2. **局部方差谱离散度** —— 分块方差的最大最小差，匀质序列谱平，
//!    种子化扰动使谱起伏；
//! 3. **帧间规律性** —— 一阶差分同号比例，解析序列完全规律（=1），
//!    扰动破坏规律性。
//!
//! 三者合成 `irregularity`（越大越不像解析生成），判官选**更大**的一边。
//! 全部只用 `+ - * /` 与位型比较，**零 libm、零真随机、零线程**。
//!
//! # 判据靶子（D-31）
//!
//! V3 只证明「`crafted` 档确实把不规则注入了生成源头」，不证明也不要求
//! 所有片子都得搓。观测序列来自**真实求值链** [`prism_dsl::inject_frame_values`]
//! 逐帧解算的六族信号（与 `wiring.rs::resolve_frame` 同一函数），不是本模块
//! 另写一套噪声 ⇒ 禁止两端同错。若实现退化成「帧后贴皮」或「字段没接」，
//! 生成源头序列不变 ⇒ 判官无区分力 ⇒ 胜率回落到随机基线，过不了冻结门槛。
//!
//! # 冻结项（先定桩再跑，不许事后调）
//!
//! - 样本池 `BLIND_SAMPLE_COUNT = 64` 对；
//! - 每对 `BLIND_FRAME_COUNT = 24` 帧；
//! - 种子 `BLIND_TAG` 派生，配对顺序由 `hash_mix` 打乱（盲）；
//! - 门槛 `BLIND_THRESHOLD = 0.75`（依据见 `docs/v3.0-盲测协议.md`）；
//! - `critic` 五维**只做画质不退化侧证**，不进胜率。
//!
//! # 反假绿负控
//!
//! [`run_control`] 把 B 也换成 `analytic`（等价于「字段没接 / 痕迹挪到帧后」），
//! 判官应失去区分力 ⇒ 胜率显著低于门槛。这条负控证明判官不是无条件选 B。

use prism_core::eval::hash_mix;
use prism_dsl::{
    inject_frame_values, signals_of, FinishMode, FinishPlan, TraceDim, TraceFamily, WiringFrame,
};

/// 盲测命名空间常量（与 W3_TAG / 产物哈希链互不冲突）。
pub const BLIND_TAG: u64 = 0x7733_5eed_0000_0002;

/// 冻结样本池：64 对。
///
/// 依据：二项检验在随机猜测 `p=0.5` 下，64 次试验达到 ≥ 48 胜（胜率 0.75）
/// 的单尾 `p < 1e-4`，属「显著 > 0.5」；64 为 2 的幂，配对顺序可由
/// `hash_mix` 位均匀打乱。定桩在跑之前，不事后调整。
pub const BLIND_SAMPLE_COUNT: u64 = 64;

/// 冻结每对帧数：24 帧（≥ 3 个 8 帧分块，保证局部方差谱有定义）。
pub const BLIND_FRAME_COUNT: u64 = 24;

/// 冻结判定门槛：B 被选中胜率须 ≥ 0.75。
pub const BLIND_THRESHOLD: f64 = 0.75;

/// 冻结强度档（DSL `0..1000` 定点；8 档覆盖弱扰动到满强度）。
pub const BLIND_STRENGTHS: [u32; 8] = [80, 160, 240, 320, 480, 640, 800, 1000];

/// 六族注入在观测序列里的固定权重（判官与场景共用的确定性常量）。
pub const BLIND_FAMILY_GAINS: [f32; 6] = [0.15, 0.12, 0.14, 0.13, 0.11, 0.16];

/// 判官的三条可复现「解析性 / 匀质度」指标 + 合成不规则度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticityMetrics {
    /// 运动二阶差分能量（越大越不解析）。
    pub second_difference_energy: f32,
    /// 局部方差谱离散度（越大越不匀质）。
    pub local_variance_spread: f32,
    /// 帧间规律性（∈[0,1]，越小越不规律）。
    pub temporal_regularity: f32,
    /// 合成不规则度 = 三项归一化之和（越大越像人搓的）。
    pub irregularity: f32,
}

impl AnalyticityMetrics {
    /// 规范字节序列（逐对输出 / 两跑逐字节对拍）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(16);
        for v in [
            self.second_difference_energy,
            self.local_variance_spread,
            self.temporal_regularity,
            self.irregularity,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }
}

/// 计算一条观测序列的三条指标（纯算术，零 libm；短序列返回全解析值）。
pub fn analyticity_metrics(series: &[f32]) -> AnalyticityMetrics {
    let n: usize = series.len();
    if n < 4 {
        return AnalyticityMetrics {
            second_difference_energy: 0.0,
            local_variance_spread: 0.0,
            temporal_regularity: 1.0,
            irregularity: 0.0,
        };
    }

    // 1. 运动二阶差分能量 Σ|x[t] - 2x[t-1] + x[t-2]|。
    let mut e2: f32 = 0.0;
    for t in 2..n {
        let d: f32 = series[t] - 2.0 * series[t - 1] + series[t - 2];
        e2 += if d < 0.0 { -d } else { d };
    }

    // 2. 局部方差谱离散度：4 个等长分块，各块方差的 max-min。
    const BLOCKS: usize = 4;
    let mut variances: [f32; BLOCKS] = [0.0; BLOCKS];
    let step: usize = n / BLOCKS;
    for b in 0..BLOCKS {
        let start: usize = b * step;
        let end: usize = if b + 1 == BLOCKS { n } else { start + step };
        let len: usize = end - start;
        if len == 0 {
            continue;
        }
        let mut mean: f32 = 0.0;
        for &v in &series[start..end] {
            mean += v;
        }
        mean /= len as f32;
        let mut var: f32 = 0.0;
        for &v in &series[start..end] {
            let d: f32 = v - mean;
            var += d * d;
        }
        variances[b] = var / len as f32;
    }
    let mut vmax: f32 = variances[0];
    let mut vmin: f32 = variances[0];
    for &v in &variances[1..] {
        if v > vmax {
            vmax = v;
        }
        if v < vmin {
            vmin = v;
        }
    }
    let spread: f32 = vmax - vmin;

    // 3. 帧间规律性：一阶差分同号比例（零差分视为与前一号相同）。
    let mut same: u32 = 0;
    let mut prev_positive: bool = series[1] >= series[0];
    let mut pairs: u32 = 0;
    for t in 2..n {
        let d: f32 = series[t] - series[t - 1];
        let positive: bool = if d == 0.0 { prev_positive } else { d > 0.0 };
        if positive == prev_positive {
            same += 1;
        }
        prev_positive = positive;
        pairs += 1;
    }
    let regularity: f32 = if pairs == 0 {
        1.0
    } else {
        same as f32 / pairs as f32
    };

    // 归一化到 [0,1) 后等权相加（有理函数，确定性；分母恒 > 0）。
    let e2n: f32 = e2 / (e2 + 1.0);
    let spreadn: f32 = spread / (spread + 1.0);
    let deficit: f32 = 1.0 - regularity;
    let irregularity: f32 = e2n + spreadn + deficit;

    AnalyticityMetrics {
        second_difference_energy: e2,
        local_variance_spread: spread,
        temporal_regularity: regularity,
        irregularity,
    }
}

/// 判官的强制二选一结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 第一条序列更像人搓的。
    First,
    /// 第二条序列更像人搓的。
    Second,
}

/// **盲判官**：只接收两条数值序列，不含任何标签 / 档位 / 文件名。
///
/// 返回 `(强制二选一结果, 是否两条指标完全相同)`。完全相同时按确定性规则
/// 选 `Second`（强制二选一，不留平局），并置 `tie = true` 供协议记录。
pub fn judge(first: &[f32], second: &[f32]) -> (Verdict, bool) {
    let a: AnalyticityMetrics = analyticity_metrics(first);
    let b: AnalyticityMetrics = analyticity_metrics(second);
    let tie: bool = a.irregularity.to_bits() == b.irregularity.to_bits();
    if a.irregularity > b.irregularity {
        (Verdict::First, tie)
    } else {
        (Verdict::Second, tie)
    }
}

/// 一条盲测样本的完整逐对输出。
#[derive(Debug, Clone, PartialEq)]
pub struct BlindSample {
    /// 样本编号 `0..BLIND_SAMPLE_COUNT`。
    pub index: u64,
    /// 本对的确定性种子。
    pub seed: u64,
    /// 六族声明强度（0 = 该族绝对匀质）。
    pub strengths: [u32; 6],
    /// B 是否被放在第一条（配对顺序由 `hash_mix` 打乱，判官不知情）。
    pub b_first: bool,
    /// 第一条序列的三指标。
    pub metrics_first: AnalyticityMetrics,
    /// 第二条序列的三指标。
    pub metrics_second: AnalyticityMetrics,
    /// 判官的强制二选一。
    pub verdict: Verdict,
    /// 两条序列指标完全相同（判官无区分力）。
    pub tie: bool,
    /// 判官是否选中了 B（真值由协议层对照，判官本身不知道）。
    pub b_selected: bool,
}

/// 完整盲测报告（主判据 + 逐对输出 + 冻结门槛）。
#[derive(Debug, Clone, PartialEq)]
pub struct BlindReport {
    /// 逐对输出。
    pub samples: Vec<BlindSample>,
    /// B 被选中次数。
    pub b_wins: usize,
    /// 样本数。
    pub total: usize,
    /// B 被选中胜率。
    pub win_rate: f64,
    /// 冻结门槛。
    pub threshold: f64,
    /// 是否过门槛。
    pub passed: bool,
    /// 判官指标完全相同（无区分力）的对数。
    pub ties: usize,
    /// 门槛冻结的二项检验显著性依据（纯文本，落盘用）。
    pub threshold_basis: &'static str,
}

/// 门槛依据（冻结，与文档、协议、脚本三处同源）。
pub const BLIND_THRESHOLD_BASIS: &str = "N=64、H0:p=0.5 的二项检验，≥48 胜（0.75）单尾 p<1e-4；门槛显著高于随机猜测 0.5，且预留 0.25 容错带，定桩于跑测之前，不事后调整";

impl BlindReport {
    /// 规范字节序列（两跑逐字节对拍）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(self.total as u64).to_le_bytes());
        out.extend_from_slice(&(self.b_wins as u64).to_le_bytes());
        out.extend_from_slice(&(self.ties as u64).to_le_bytes());
        out.extend_from_slice(&self.threshold.to_bits().to_le_bytes());
        out.extend_from_slice(&(self.passed as u8).to_le_bytes());
        for s in &self.samples {
            out.extend_from_slice(&s.index.to_le_bytes());
            out.extend_from_slice(&s.seed.to_le_bytes());
            for &v in &s.strengths {
                out.extend_from_slice(&(v as u64).to_le_bytes());
            }
            out.push(s.b_first as u8);
            out.extend_from_slice(&s.metrics_first.canonical_bytes());
            out.extend_from_slice(&s.metrics_second.canonical_bytes());
            out.push(match s.verdict {
                Verdict::First => 1,
                Verdict::Second => 2,
            });
            out.push(s.tie as u8);
            out.push(s.b_selected as u8);
        }
        out
    }

    /// 逐对输出的可复现文本（脚本 / 文档 / `cargo test -- --nocapture` 共用）。
    pub fn report_text(&self) -> String {
        let mut out: String = String::new();
        out.push_str(&format!(
            "盲测样本数 N={}  冻结门槛={:.2}  B 胜={}/{}  胜率={:.4}  判定={}\n",
            self.total,
            self.threshold,
            self.b_wins,
            self.total,
            self.win_rate,
            if self.passed { "PASS" } else { "FAIL" }
        ));
        out.push_str(&format!("门槛依据：{}\n", self.threshold_basis));
        out.push_str("idx seed      六族强度                         顺序   不规则度(第一/第二)      判选   B胜  平局\n");
        for s in &self.samples {
            let strengths: String = s
                .strengths
                .iter()
                .map(|v| format!("{v:>4}"))
                .collect::<Vec<String>>()
                .join(" ");
            out.push_str(&format!(
                "{:>3} {:#010x} [{}] {} {:>8.5}/{:<8.5}  {}    {}    {}\n",
                s.index,
                s.seed,
                strengths,
                if s.b_first { "B先" } else { "A先" },
                s.metrics_first.irregularity,
                s.metrics_second.irregularity,
                match s.verdict {
                    Verdict::First => "第一",
                    Verdict::Second => "第二",
                },
                if s.b_selected { "是" } else { "否" },
                if s.tie { "是" } else { "否" },
            ));
        }
        out
    }
}

/// 由样本编号派生固定的种子 / 六族强度 / 配对顺序（全部走 `hash_mix`）。
fn sample_params(index: u64) -> (u64, [u32; 6], bool) {
    let h: u64 = hash_mix(BLIND_TAG, index);
    let seed: u64 = hash_mix(BLIND_TAG ^ 0x5eed_5eed, index);
    // 活跃族掩码：至少一族活跃，其余族绝对匀质（表达「只有运动带呼吸」）。
    let mut mask: u64 = h & 0b11_1111;
    if mask == 0 {
        mask = 1 << (index % 6);
    }
    let mut strengths: [u32; 6] = [0; 6];
    for (k, slot) in strengths.iter_mut().enumerate() {
        let active: bool = mask & (1 << k) != 0;
        let level: usize = ((h >> (8 + k * 5)) & 0b111) as usize;
        *slot = if active { BLIND_STRENGTHS[level] } else { 0 };
    }
    // 配对顺序：由独立槽位的 hash 位打乱，判官不知标签。
    let b_first: bool = (hash_mix(h, 0xA5A5_A5A5) & 1) == 1;
    (seed, strengths, b_first)
}

/// 生成一条**生成源头观测序列**。
///
/// `base` 是同场景共享的纯解析匀速 ramp（A/B 完全相同）；六族注入量来自
/// 真实求值链 [`inject_frame_values`] + [`signals_of`]（与
/// `wiring.rs::resolve_frame` 同一函数，不另写噪声）。`analytic` 档
/// `inject_frame_values` 恒等返回 ⇒ 序列即纯解析基线。
pub fn observation(seed: u64, strengths: &[u32; 6], crafted: bool) -> Vec<f32> {
    let mode: FinishMode = if crafted { FinishMode::Crafted } else { FinishMode::Analytic };
    let mut dims: [TraceDim; 6] = [TraceDim::OFF; 6];
    if crafted {
        for (k, slot) in dims.iter_mut().enumerate() {
            let strength: u32 = strengths[k].min(prism_dsl::STRENGTH_MAX);
            *slot = TraceDim { enabled: strength > 0, strength, line: 0 };
        }
    }
    let plan: FinishPlan = FinishPlan { mode, dims, seed, line: 0, rubric: None };
    let frames: u64 = BLIND_FRAME_COUNT;
    let mut series: Vec<f32> = Vec::with_capacity(frames as usize);
    for t in 0..frames {
        let mut wf: WiringFrame = WiringFrame::default();
        inject_frame_values(&mut wf, &plan, t, seed, 0);
        let s = signals_of(&wf);
        let signals: [f32; 6] = [
            s.motion,
            s.geometry,
            s.material_light,
            s.time_camera,
            s.composition_color,
            s.edit_audio,
        ];
        let mut injected: f32 = 0.0;
        for (k, &v) in signals.iter().enumerate() {
            injected += BLIND_FAMILY_GAINS[k] * v;
        }
        let base: f32 = 0.5 + 0.25 * (t as f32 / frames as f32);
        series.push(base + injected);
    }
    series
}

/// 组装一对（A = analytic，B = crafted）并执行盲判。
fn build_sample(index: u64, crafted_is_real: bool) -> BlindSample {
    let (seed, strengths, b_first): (u64, [u32; 6], bool) = sample_params(index);
    let a: Vec<f32> = observation(seed, &strengths, false);
    // 负控：把 B 也换成 analytic（等价于字段没接 / 痕迹挪到帧后贴皮）。
    let b: Vec<f32> = observation(seed, &strengths, crafted_is_real);
    let (first, second): (&[f32], &[f32]) = if b_first { (&b, &a) } else { (&a, &b) };
    let (verdict, tie): (Verdict, bool) = judge(first, second);
    let b_selected: bool = matches!(
        (verdict, b_first),
        (Verdict::First, true) | (Verdict::Second, false)
    );
    BlindSample {
        index,
        seed,
        strengths,
        b_first,
        metrics_first: analyticity_metrics(first),
        metrics_second: analyticity_metrics(second),
        verdict,
        tie,
        b_selected,
    }
}

/// 运行冻结的盲测协议（V3 主判据）。
pub fn run_protocol() -> BlindReport {
    assemble(0..BLIND_SAMPLE_COUNT, true)
}

/// 运行**反假绿负控**：B 也用 `analytic`（字段没接 / 痕迹挪到帧后）。
///
/// 预期胜率显著低于门槛，证明判官不是无条件选 B。
pub fn run_control() -> BlindReport {
    assemble(0..BLIND_SAMPLE_COUNT, false)
}

fn assemble(indices: std::ops::Range<u64>, crafted_is_real: bool) -> BlindReport {
    let samples: Vec<BlindSample> = indices.map(|i| build_sample(i, crafted_is_real)).collect();
    let b_wins: usize = samples.iter().filter(|s| s.b_selected).count();
    let ties: usize = samples.iter().filter(|s| s.tie).count();
    let total: usize = samples.len();
    let win_rate: f64 = if total == 0 {
        0.0
    } else {
        b_wins as f64 / total as f64
    };
    BlindReport {
        samples,
        b_wins,
        total,
        win_rate,
        threshold: BLIND_THRESHOLD,
        passed: win_rate >= BLIND_THRESHOLD,
        ties,
        threshold_basis: BLIND_THRESHOLD_BASIS,
    }
}

/// 六族标签（逐对输出 / 文档表格同源）。
pub const BLIND_FAMILY_TAGS: [&str; 6] = [
    "motion",
    "geometry",
    "material_light",
    "time_camera",
    "composition_color",
    "edit_audio",
];

/// 核对六族标签与 [`TraceFamily::ALL`] 一一对应（静态注入断言的读数）。
pub fn family_tags_match_dsl() -> bool {
    TraceFamily::ALL
        .iter()
        .zip(BLIND_FAMILY_TAGS.iter())
        .all(|(f, t)| f.tag() == *t)
}

// ---------------------------------------------------------------------------
// critic 五维**客观侧证**（§3.5 V3：只证明画质不退化，不进盲测胜率）
// ---------------------------------------------------------------------------

/// 画质不退化容差（critic 总分量纲 `0..1`）。
///
/// 定桩于跑测之前：六族注入是**有界微扰**（信号 ∈ `[-1,1]`，权重固定），
/// 允许画质总分有 0.25 的容错带；这是「不退化」的客观定义，不是审美打分，
/// 也不与盲测判官的解析性指标混用（§3.5 反假绿要点：维度不错配）。
pub const CRITIC_REGRESSION_TOL: f32 = 0.25;

/// 一对 A/B 观测序列的 critic 五维侧证。
#[derive(Debug, Clone, PartialEq)]
pub struct CriticEvidence {
    /// 样本编号（与盲测样本池同一 `sample_params`）。
    pub sample_index: u64,
    /// A（analytic）五维分数（`Dim::ALL` 顺序）。
    pub analytic_dims: [f32; 5],
    /// B（crafted）五维分数。
    pub crafted_dims: [f32; 5],
    /// A 总分。
    pub analytic_total: f32,
    /// B 总分。
    pub crafted_total: f32,
    /// 五维全部有限、总分为正、且 B 不低于 A − 冻结容差。
    pub no_regression: bool,
}

impl CriticEvidence {
    /// 规范字节序列（两跑逐字节对拍）。
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&self.sample_index.to_le_bytes());
        for v in self
            .analytic_dims
            .iter()
            .chain(self.crafted_dims.iter())
            .chain([self.analytic_total, self.crafted_total].iter())
        {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.push(self.no_regression as u8);
        out
    }

    /// 逐维可复现文本（文档 / 脚本 / `--nocapture` 共用）。
    pub fn report_text(&self) -> String {
        use prism_aesthetic::critic::Dim;
        let mut out: String = format!(
            "critic 侧证 sample={}  A 总分={:.6}  B 总分={:.6}  容差={:.2}  {}",
            self.sample_index,
            self.analytic_total,
            self.crafted_total,
            CRITIC_REGRESSION_TOL,
            if self.no_regression { "画质不退化" } else { "画质退化" }
        );
        for (i, d) in Dim::ALL.iter().enumerate() {
            out.push_str(&format!(
                "\n  {:<12} A={:.6}  B={:.6}  Δ={:+.6}",
                d.name(),
                self.analytic_dims[i],
                self.crafted_dims[i],
                self.crafted_dims[i] - self.analytic_dims[i]
            ));
        }
        out
    }
}

/// 把一条生成源头观测序列铺成 `6×4` 灰度帧（**只做数据搬运**）。
///
/// 画质求值全部由既有 `prism_aesthetic::critic::analyze` 完成；本函数不实现
/// 任何画质数学（红线 1），也不碰帧缓冲的渲染管线（不是帧后贴皮）。
fn observation_frame(series: &[f32]) -> prism_io::fbuf::Fbuf16F {
    let (w, h): (u32, u32) = (6, 4);
    let mut fb: prism_io::fbuf::Fbuf16F = prism_io::fbuf::Fbuf16F::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i: usize = (y * w + x) as usize;
            let v: f32 = series.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            fb.set(x, y, [v, v, v, 1.0]);
        }
    }
    fb
}

/// 对样本 `sample_index` 用既有 critic 五维做「画质不退化」侧证。
///
/// A/B 序列来自**同一条** [`observation`] 生成源头链（不另写噪声），只差
/// `crafted` 开关；`score` / `analyze` 全是既有 crate 的公开 API。侧证结果
/// **不进盲测胜率**，门槛（[`CRITIC_REGRESSION_TOL`]）冻结在跑测之前。
pub fn critic_side_evidence(sample_index: u64) -> CriticEvidence {
    use prism_aesthetic::critic::{analyze, score, Dim, Rubric};

    let (_seed, strengths, _b_first): (u64, [u32; 6], bool) = sample_params(sample_index);
    let a: Vec<f32> = observation(_seed, &strengths, false);
    let b: Vec<f32> = observation(_seed, &strengths, true);
    let rubric: Rubric = Rubric::default();
    let sa = score(&analyze(&observation_frame(&a)), &rubric);
    let sb = score(&analyze(&observation_frame(&b)), &rubric);
    let mut analytic_dims: [f32; 5] = [0.0; 5];
    let mut crafted_dims: [f32; 5] = [0.0; 5];
    for (i, d) in Dim::ALL.iter().enumerate() {
        analytic_dims[i] = sa.dim(*d);
        crafted_dims[i] = sb.dim(*d);
    }
    let finite: bool = analytic_dims
        .iter()
        .chain(crafted_dims.iter())
        .chain([sa.total, sb.total].iter())
        .all(|v| v.is_finite());
    let no_regression: bool = finite
        && sa.total > 0.0
        && sb.total > 0.0
        && sb.total >= sa.total - CRITIC_REGRESSION_TOL;
    CriticEvidence {
        sample_index,
        analytic_dims,
        crafted_dims,
        analytic_total: sa.total,
        crafted_total: sb.total,
        no_regression,
    }
}

/// 全样本池的侧证聚合（逐对输出 + 不退化对数，两跑逐字节）。
pub fn critic_side_report() -> Vec<CriticEvidence> {
    (0..BLIND_SAMPLE_COUNT)
        .map(critic_side_evidence)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_is_blind_and_deterministic() {
        let a: Vec<f32> = observation(42, &[620, 0, 400, 300, 250, 500], false);
        let b: Vec<f32> = observation(42, &[620, 0, 400, 300, 250, 500], true);
        assert_eq!(judge(&b, &a), judge(&b, &a));
        // 同一个输入顺序调换，判官结论必须跟着数值走（不是认位置）。
        let (v1, _) = judge(&a, &b);
        let (v2, _) = judge(&b, &a);
        assert_ne!(v1, v2, "解析的 A 与注入的 B 应被区分");
    }

    #[test]
    fn analytic_series_is_perfectly_regular() {
        let a: Vec<f32> = observation(7, &[0, 0, 0, 0, 0, 0], false);
        let m: AnalyticityMetrics = analyticity_metrics(&a);
        assert!(m.second_difference_energy < 1e-4, "纯线性二阶差分趋 0");
        assert!(m.local_variance_spread < 1e-6, "匀质谱平");
        assert!(m.temporal_regularity > 0.99, "解析序列完全规律");
    }

    #[test]
    fn control_has_no_discriminative_power() {
        let report: BlindReport = run_control();
        assert!(report.ties > 0, "A/A 对应出现指标完全相同的对");
        assert!(
            report.win_rate < report.threshold,
            "负控胜率应低于冻结门槛：{}",
            report.win_rate
        );
    }

    #[test]
    fn protocol_passes_frozen_threshold_and_two_runs_are_identical() {
        let a: BlindReport = run_protocol();
        let b: BlindReport = run_protocol();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes(), "两跑逐字节一致");
        assert_eq!(a.total, BLIND_SAMPLE_COUNT as usize);
        assert!(a.passed, "胜率 {} 应过冻结门槛 {}", a.win_rate, a.threshold);
    }

    #[test]
    fn family_tags_match() {
        assert!(family_tags_match_dsl());
    }

    #[test]
    fn critic_side_evidence_uses_existing_critic_and_is_deterministic() {
        let a: CriticEvidence = critic_side_evidence(3);
        let b: CriticEvidence = critic_side_evidence(3);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes(), "侧证两跑逐字节一致");
        assert!(a.analytic_total.is_finite() && a.crafted_total.is_finite());
        assert!(a.no_regression, "画质不退化：\n{}", a.report_text());
        assert!(a.report_text().contains("exposure"));
    }

    #[test]
    fn critic_side_report_has_no_regression_on_full_pool() {
        let all: Vec<CriticEvidence> = critic_side_report();
        assert_eq!(all.len(), BLIND_SAMPLE_COUNT as usize);
        let regressions: usize = all.iter().filter(|e| !e.no_regression).count();
        assert_eq!(regressions, 0, "全池画质侧证不许有退化对：{regressions}");
    }
}
