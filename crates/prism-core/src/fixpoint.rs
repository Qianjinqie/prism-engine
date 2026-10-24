//! 环推进（不动点迭代）：推进-停机四元组的环实例。
//!
//! 语义权威：`docs/设计/研究/环推进-不动点实例.md`。四元组在本实例的取值：
//! `X = X ⊕ Λ`（位置/速度 ⊕ 约束平行 λ 数组；λ **跨轮存活**，仅子步边界
//! 清零——这是 XPBD 对框架 `Φ: X → X` 设定的一处细化）、`⊑` = 平凡序、
//! `P` = 一轮 Gauss–Seidel 扫描、`H` = M3 定数停机（`n = substeps ×
//! iterations`）+ 残差记录 + 发散检测（连续 [`DIVERGENCE_WINDOW_DEFAULT`]
//! 个采样点残差上升超 ε 余量）。
//!
//! # 两层入口（与 `eval.rs` 的时间实例同风格）
//!
//! - [`run_fixed_point`]：**帧内** M3 定数迭代本体——给定 `Φ` 与 `(ε, n)`
//!   跑满 n 轮（不早停），逐轮记录残差并喂入 [`DivergenceDetector`]。
//!   只观测、不改写：发散检测不接入迭代控制流（E3 的工程化纪律）。
//! - [`LoopDriver`]：**帧间** seek 驱动——自 `prism-render::movie` 的
//!   `SimTimeline` 上提泛化：顺序延续 / 快照装配 / 首帧初态 / 冷 seek
//!   `P3012`，快照键 = `(子图签名, 帧)`，命中与否只影响重算量、不影响结果。
//!
//! # 模式元数据进签名
//!
//! [`SolveModeMeta`]（M3 + `(ε, n)`）经 [`solve_mode_def`] 编码为
//! `ChannelDef`（tag = [`SOLVE_MODE_CHANNEL_TAG`] = `0x5333`），再经
//! [`crate::signature::channel_definition_hash`] 收拢进
//! [`loop_frame_signature`]——改 `iterations` 或 `residual_eps` 任一 ⇒ 帧签名
//! 变 ⇒ 快照 / 缓存键不误复用（命题 S3 迭代层缓存键 `(ε, n)` 纪律）。
//!
//! # 非扩张性（如实记）
//!
//! XPBD 修正步的 `Lip ≤ 1` **未证**（文档 §3-2 / §7-1）：凸投影 `Lip = 1`
//! 是已证命题，但「引擎实现步（柔度正则 + λ 累积 + 部分修正）复合后
//! `Lip ≤ 1`」仍是工作假设。故本模块不提供 M2 容差停机（那会伪造 Banach
//! 界），安全兜底是 [`DivergenceDetector`] 的事后判定，不是分析结论。
//!
//! # 未决问题（文档 §7；本实装按现状最保守路线）
//!
//! - **§7-1** 修正步 `Lip ≤ 1` 未证 ⟹ 只走 M3 + 检测器，不开 M2；
//! - **§7-2** `DistanceConstraint` 的结构性衰减率 q 未形式化 ⟹ 不做 q 先验，
//!   残差下降只作为事后观察；
//! - **§7-3** `Integrator` 未进 0x5333 ⟹ 签名只收 `(substeps, iterations, ε)`，
//!   积分格式仍由宿主规格自述（未进签名，缺口如实保留）；
//! - **§7-4** 「步」的单位未跨实例归一化 ⟹ 本实装钉死：一帧 = 一次 `step`
//!   闭包（XPBD 的 `substeps × iterations` 轮），`n` 按乘积计；
//! - **§7-5** 帧内子步级 seek 未工程化 ⟹ 快照粒度 = 帧末，λ 不入快照
//!   （子步边界清零，不属跨帧状态）；
//! - **§7-6** `Lip > 1` 先验识别未形式化 ⟹ 只有事后 [`DivergenceDetector`]
//!   一路，无构造期判据。

use crate::eval::EvalError;
use crate::graph::Graph;
use crate::node::{Hash256, SnapshotStore};
use crate::signature::{
    channel_definition_hash, node_signature, snapshot_signature, ChannelDef, ChannelDefs,
    SignatureError,
};

// ---------------------------------------------------------------------------
// 模式声明（M3）与模式元数据（(ε, n) 缓存键分量）
// ---------------------------------------------------------------------------

/// 推进-停机模式（求值层 §3.1 三定理查表）。
///
/// 环实例的唯一诚实模式是 **M3**：XPBD 约束投影非扩张（`Lip = 1`，E2）⟹
/// 容差停机的先验误差界 `ε/(1−L)` 除零，M2 不适用；f32 点值域有无穷升链
/// ⟹ Kleene（M1）无有限步保证（命题 3.1 / 3.9 推论 9a）。固定步数显式标价
/// 是本求解器唯一可谈的停止判据。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveMode {
    /// M3：固定步数 + 残差记录（无容差停机、无先验误差界）。
    #[default]
    FixedIterations,
}

/// 默认残差判定阈值 ε：两个用途都是**事后**（非停机判据）——① 残差质量
/// 指标判定；② 发散检测的 ε 余量。
pub const RESIDUAL_EPS_DEFAULT: f32 = 1e-4;

/// 发散检测默认窗口：连续 3 个采样点残差上升超 ε 余量判发散。
pub const DIVERGENCE_WINDOW_DEFAULT: usize = 3;

/// 模式元数据通道定义的 tag = `0x5333`（两个字节 `0x53`/`0x33` 即 ASCII 的
/// `'S'`/`'3'`；W1 已落盘，此处收拢为内核一等属性，字节编码保持逐位不变）。
pub const SOLVE_MODE_CHANNEL_TAG: u16 = 0x5333;

/// 模式元数据：M3 声明 + 缓存键模式参数。
///
/// `substeps / iterations / residual_eps` 任一变化都必须改变帧签名（S3
/// 迭代层缓存键 `(ε, n)`）；`(ε, n)` 在此的成员身份是**缓存键分量 + 事后
/// 判定**，不是停机判据（M3 的停机只看步数）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveModeMeta {
    /// 模式声明（M3）。
    pub mode: SolveMode,
    /// 子步数（固定标价；缓存键分量）。
    pub substeps: u32,
    /// 迭代数 n（固定标价；缓存键分量）。
    pub iterations: u32,
    /// 残差判定阈值 ε（缓存键分量；事后质量指标，非停机判据）。
    pub residual_eps: f32,
}

impl SolveModeMeta {
    /// 构造 M3 模式元数据（`(ε, n)` 显式标价，供签名 / 缓存键收拢）。
    pub fn new(substeps: u32, iterations: u32, residual_eps: f32) -> Self {
        Self { mode: SolveMode::FixedIterations, substeps, iterations, residual_eps }
    }

    /// 一个 AH 步的投影轮数 `n = substeps × iterations`（u64 防乘积溢出）。
    pub fn rounds(&self) -> u64 {
        self.substeps as u64 * self.iterations as u64
    }

    /// 模式防呆（C18/C21）：子步数 / 迭代数 ≥ 1，ε 正有限。
    ///
    /// 固定步数是确定性纪律：0 不能被解读为「不推进」或「无限迭代」，
    /// 一律结构化拒绝（危险默认值禁止）。
    pub fn validate(&self, location: &str) -> Result<(), EvalError> {
        if self.substeps == 0 {
            return Err(EvalError {
                code: "P1003",
                location: location.to_string(),
                problem: "substeps = 0：子步数必须 ≥ 1（步数固定是确定性纪律）".to_string(),
                fix: "传 substeps ≥ 1（参考标价：小步比多迭代稳）".to_string(),
            });
        }
        if self.iterations == 0 {
            return Err(EvalError {
                code: "P1004",
                location: location.to_string(),
                problem: "iterations = 0：迭代数必须 ≥ 1（M3 定数停机不含零步）".to_string(),
                fix: "传 iterations ≥ 1（迭代数与刚度解耦：多迭代只降残差）".to_string(),
            });
        }
        if !(self.residual_eps.is_finite() && self.residual_eps > 0.0) {
            return Err(EvalError {
                code: "P1006",
                location: location.to_string(),
                problem: format!(
                    "residual_eps = {}：残差阈值必须为正有限数（它只作事后判定与发散余量）",
                    self.residual_eps
                ),
                fix: format!("传正数，参考标价 {RESIDUAL_EPS_DEFAULT}"),
            });
        }
        Ok(())
    }
}

/// 模式元数据的通道定义编码（判别 0 = M3 固定步数 ++ substeps ++
/// iterations ++ residual_eps 位模式，全大端）。
///
/// 签名只消费相等性、不解读内容（与 [`ChannelDefs`] 契约同款）；编码与 W1
/// 的 `movie.rs` 私有实现逐字节一致，故历史签名 / 快照键零迁移。
pub fn solve_mode_def(mode: &SolveModeMeta) -> ChannelDef {
    let mut def_bytes: Vec<u8> = Vec::with_capacity(13);
    def_bytes.push(match mode.mode {
        SolveMode::FixedIterations => 0u8,
    });
    def_bytes.extend_from_slice(&mode.substeps.to_be_bytes());
    def_bytes.extend_from_slice(&mode.iterations.to_be_bytes());
    def_bytes.extend_from_slice(&mode.residual_eps.to_bits().to_be_bytes());
    ChannelDef { tag: SOLVE_MODE_CHANNEL_TAG, def_bytes }
}

/// 环实例的帧签名：节点子图签名再经模式元数据通道定义哈希链收拢。
///
/// `(ε, n)` 自此既在缓存键里、也在快照键里：模式参数变化 ⇒ 签名变化 ⇒
/// 旧快照 / 缓存绝不误复用（命题 4.2 推论 4.2a + 命题 S3）。
pub fn loop_frame_signature(
    graph: &Graph,
    id: u64,
    defs: &ChannelDefs,
    mode: &SolveModeMeta,
) -> Result<Hash256, SignatureError> {
    let subgraph: Hash256 = node_signature(graph, id, defs)?;
    Ok(snapshot_signature(
        &subgraph,
        &[channel_definition_hash(&solve_mode_def(mode))],
    ))
}

// ---------------------------------------------------------------------------
// 发散检测（E3 推论 (c)）
// ---------------------------------------------------------------------------

/// 发散检测器（E3 推论 (c) 的工程化）：对残差采样序列做事后趋势判定。
///
/// 推理链：复合投影 `Φ = P_n ∘ … ∘ P_1` 的 `Lip(Φ) ≤ Π_i Lip(P_i)`；凸
/// 投影 `Lip = 1`（E2 安全），非凸 / 折叠约束可 `> 1`（有发散风险）。有限
/// 精度下 `Lip = 1` 的边界情形（`1 + O(ε_machine)`）在长迭代中显现为残差
/// **缓慢上升**——判据必须是「连续 `window` 个采样点的上升量超过 ε 余量」：
/// 单步不降在收敛平台期正常出现（微振），不构成发散证据；ε 余量把平台
/// 抖动与真实上升区分开。
///
/// 只观测、不改写：不接入求解循环（求解语义零变化），采样粒度由调用方
/// 选定（帧内每轮残差，或帧末残差）。
#[derive(Debug, Clone)]
pub struct DivergenceDetector {
    /// 连续上升步数阈值（≥ 1）。
    window: usize,
    /// 上升判定余量 ε（上升量 ≤ ε 视为平台抖动，不计）。
    margin: f32,
    /// 上一采样点残差。
    prev: Option<f32>,
    /// 当前连续上升步数。
    streak: usize,
    /// 已喂入的采样点数。
    samples: usize,
    /// 最近一次观测后的发散状态（连续上升达窗口 = true）。
    diverged: bool,
}

impl DivergenceDetector {
    /// 构造（`window` ≥ 1；`margin` 负值钳为 0）。
    pub fn new(window: usize, margin: f32) -> Self {
        Self {
            window: window.max(1),
            margin: if margin > 0.0 { margin } else { 0.0 },
            prev: None,
            streak: 0,
            samples: 0,
            diverged: false,
        }
    }

    /// 喂入一个残差采样点。返回 `true` = 当前处于「连续 `window` 步上升
    /// 超 ε 余量」的发散状态（E3 判据）。
    ///
    /// NaN 输入：全部比较为 false ⟹ 不计上升（streak 归零）。非有限残差的
    /// 健康判定归调用方的有限值守卫——本检测器只做趋势判定，不越权吞掉
    /// 也不改写（C20）。
    pub fn observe(&mut self, residual: f32) -> bool {
        self.samples += 1;
        self.diverged = match self.prev {
            Some(prev) if residual > prev + self.margin => {
                self.streak += 1;
                self.streak >= self.window
            }
            _ => {
                self.streak = 0;
                false
            }
        };
        self.prev = Some(residual);
        self.diverged
    }

    /// 当前是否处于发散状态（连续上升达窗口）。
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    /// 当前连续上升步数。
    pub fn streak(&self) -> usize {
        self.streak
    }

    /// 已喂入的采样点数。
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// 清零趋势状态（保留窗口 / 余量配置；切换场景时用）。
    pub fn reset(&mut self) {
        self.prev = None;
        self.streak = 0;
        self.samples = 0;
        self.diverged = false;
    }

    /// 批量扫描残差序列（事后审计 / 测试用；语义等价于逐点
    /// [`Self::observe`]，任一点触发即返回 `true`）。
    pub fn scan(samples: &[f32], window: usize, margin: f32) -> bool {
        let mut detector = Self::new(window, margin);
        samples.iter().any(|&r| detector.observe(r))
    }
}

// ---------------------------------------------------------------------------
// 帧内 M3 定数迭代（Φ 与 (ε, n)）
// ---------------------------------------------------------------------------

/// 一次定数迭代的报告：逐轮残差 + 发散判定。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FixpointReport {
    /// 本轮迭代的残差记录，长度 = `iterations`，顺序 = 轮次升序（固定，
    /// 命题 3.11 迭代输出为输入的确定函数的前提）。
    pub residuals: Vec<f32>,
    /// 逐点喂入 [`DivergenceDetector`] 后的趋势判定（只观测、不改写）。
    pub diverged: bool,
}

impl FixpointReport {
    /// 末轮残差（M3 下即本步质量读数；`iterations ≥ 1` 时恒为 `Some`）。
    pub fn final_residual(&self) -> Option<f32> {
        self.residuals.last().copied()
    }

    /// 记录条数（= iterations）。
    pub fn len(&self) -> usize {
        self.residuals.len()
    }

    /// 记录是否为空。
    pub fn is_empty(&self) -> bool {
        self.residuals.is_empty()
    }
}

/// 通用环驱动：给定 `Φ: X⊕Λ → X⊕Λ` 与 `(ε, n)`，跑 **M3 定数迭代**。
///
/// - 恰好调用 `scan` n = `mode.iterations` 次（**不早停**——发散检测只观测，
///   不改写迭代控制流；「M3 仍跑完定数步，但报告 diverged」）；
/// - 每轮末残差进 [`FixpointReport::residuals`]，并逐点喂入发散检测器
///   （窗口 [`DIVERGENCE_WINDOW_DEFAULT`]，余量 `mode.residual_eps`）；
/// - `state` 是 `X ⊕ Λ`：**λ 的跨轮存活由调用方持有**（本函数只把它当普通
///   状态字段，不重置）；子步边界的 λ 清零是调用方的帧内语义（XPBD §3）。
///
/// # Errors
/// 模式参数非法（0 迭代 / 非正 ε；0 子步同拒）→ `P1003/P1004/P1006`。
/// 校验先于任何 `scan` 调用：报错路径不改写 `state`。
pub fn run_fixed_point<S, F>(
    state: &mut S,
    mut scan: F,
    mode: &SolveModeMeta,
) -> Result<FixpointReport, EvalError>
where
    F: FnMut(&mut S) -> f32,
{
    mode.validate("run_fixed_point（环推进参数）")?;
    let mut detector: DivergenceDetector =
        DivergenceDetector::new(DIVERGENCE_WINDOW_DEFAULT, mode.residual_eps);
    let mut residuals: Vec<f32> = Vec::with_capacity(mode.iterations as usize);
    for _ in 0..mode.iterations {
        // 一轮 GS 扫描 = Φ 的一次求值（λ 随 state 存活，见文档 §1）。
        let residual: f32 = scan(state);
        detector.observe(residual);
        residuals.push(residual);
    }
    Ok(FixpointReport { residuals, diverged: detector.diverged() })
}

// ---------------------------------------------------------------------------
// 帧间 seek 驱动（自 movie.rs SimTimeline 上提泛化）
// ---------------------------------------------------------------------------

/// 快照载荷编解码（`X` 一帧状态的确定性字节形态）。
///
/// 契约：同状态必得同字节（C24 平台中立）；`decode` 对不识别字节返回
/// `None`（绝不猜测——C20），调用方按快照未命中处理。
pub trait LoopCodec<S> {
    /// 状态 → 确定性字节。
    fn encode(&self, state: &S) -> Vec<u8>;
    /// 字节 → 状态；格式 / 长度 / 版本不符返回 `None`。
    fn decode(&self, bytes: &[u8]) -> Option<S>;
}

/// 通用环实例的帧驱动：跨帧快照库 + 顺序延续 + seek 装配。
///
/// 语义对齐 `prism-core::eval` 的 ConstraintFlow 帧循环接线：快照库是纯数据
/// 资产（非隐藏全局），命中与否只影响重算量、不影响结果——seek 等价由此
/// 成立。状态机不变式：`loaded` 为真时 `state` 是 `(sig, frame)` 的精确状态。
///
/// 驱动只负责**帧边界**（seek / 快照 / 发散采样）；帧内 M3 定数迭代由
/// `step` 闭包执行（通常内部调用 [`run_fixed_point`]），二者分层但不重复。
#[derive(Debug)]
pub struct LoopDriver<S, C> {
    /// 诊断标签（三段式错误的【在哪】段；如 "sim 节点 9"）。
    label: String,
    /// 首帧初态（frame = 0 的语义来源；`(sig, 0)` 恒可恢复）。
    initial: S,
    /// 快照编解码器。
    codec: C,
    /// 模式元数据（(ε, n) 进签名；ε 兼作发散余量）。
    mode: SolveModeMeta,
    /// 当前活状态（`X ⊕ Λ` 的帧末值）。
    state: S,
    /// 当前已加载状态的签名。
    sig: Hash256,
    /// 当前已推进到的帧。
    frame: u64,
    /// `state` 是否为 `(sig, frame)` 的有效状态。
    loaded: bool,
    /// 逐帧末残差的事后趋势判定（E3；只观测、不改写）。
    divergence: DivergenceDetector,
    /// 实际步进过的帧末残差序列（seek 装配不产生步进，故不追加）。
    frame_residuals: Vec<f32>,
}

impl<S: Clone, C: LoopCodec<S>> LoopDriver<S, C> {
    /// 构造驱动（模式防呆：非法 `(ε, n)` 结构化报错，无静默兜底）。
    pub fn new(
        label: impl Into<String>,
        initial: S,
        codec: C,
        mode: SolveModeMeta,
    ) -> Result<Self, EvalError> {
        mode.validate("环推进（LoopDriver 构造）")?;
        Ok(Self {
            label: label.into(),
            state: initial.clone(),
            initial,
            codec,
            mode,
            sig: [0u8; 32],
            frame: 0,
            loaded: false,
            divergence: DivergenceDetector::new(
                DIVERGENCE_WINDOW_DEFAULT,
                mode.residual_eps,
            ),
            frame_residuals: Vec::new(),
        })
    }

    /// 推进 / 求值到 `(sig, frame)`。
    ///
    /// 分支（与 `eval.rs` / `SimTimeline` 逐条同构）：
    /// ① 幂等（同签名同帧重入直接返回）；
    /// ② 顺序延续（已加载同签名且目标在前方）——逐帧步进并写快照；
    /// ③ 快照装配（目标帧命中：seek 回退 / 预热后跳帧）；
    /// ④ 首帧初态（frame == 0：初态入库，`(sig, 0)` 恒可恢复）；
    /// ⑤ 前驱命中（恢复 `(sig, frame−1)` 再步进一帧）；
    /// ⑥ 冷 seek / 断档 → `P3012`（绝不静默 fallback 到初态）。
    ///
    /// `step` 执行一帧：原地推进状态并返回帧末残差（供发散检测）。返回
    /// `Err` 时状态必须未被推进（校验先于改写），否则跨帧状态会被污染。
    pub fn advance_to<F>(
        &mut self,
        sig: Hash256,
        frame: u64,
        store: &mut SnapshotStore,
        mut step: F,
    ) -> Result<(), EvalError>
    where
        F: FnMut(&mut S) -> Result<f32, EvalError>,
    {
        // ① 幂等。
        if self.loaded && self.sig == sig && self.frame == frame {
            return Ok(());
        }
        // ② 顺序延续（同签名且已加载帧在目标前方）。
        if self.loaded && self.sig == sig && frame > self.frame {
            for f in (self.frame + 1)..=frame {
                self.step_frame(sig, f, store, &mut step)?;
            }
            return Ok(());
        }
        // ③ 快照装配（目标帧命中）。
        if let Some(bytes) = store.get(&(sig, frame)) {
            if let Some(state) = self.codec.decode(bytes) {
                self.state = state;
                self.sig = sig;
                self.frame = frame;
                self.loaded = true;
                return Ok(());
            }
            // 字节不识别 / 状态基数不符 → 断档，落入后续分支。
        }
        // ④ 首帧初态。
        if frame == 0 {
            self.state = self.initial.clone();
            self.sig = sig;
            self.frame = 0;
            self.loaded = true;
            let bytes: Vec<u8> = self.codec.encode(&self.state);
            store.insert((sig, 0), bytes);
            return Ok(());
        }
        // ⑤ 前驱命中：恢复 (sig, frame−1) 再步进一帧。
        if let Some(bytes) = store.get(&(sig, frame - 1)) {
            if let Some(state) = self.codec.decode(bytes) {
                self.state = state;
                self.sig = sig;
                self.frame = frame - 1;
                self.loaded = true;
                self.step_frame(sig, frame, store, &mut step)?;
                return Ok(());
            }
        }
        // ⑥ 冷 seek / 断档：结构化报错（C20；修复责任交还调用方）。
        Err(EvalError {
            code: "P3012",
            location: format!("环推进（{}，帧 {frame}）", self.label),
            problem: format!(
                "帧间状态缺失：帧 {frame} 求值，但快照 (sig, {}) 不存在——seek 断档或缓存未顺序预热",
                frame - 1
            ),
            fix: format!(
                "从 0 帧起顺序推进（快照库随帧累积），或为该实例显式提供帧 {} 的初态快照",
                frame - 1
            ),
        })
    }

    /// 步进一帧并写快照（当前签名键下）。
    fn step_frame<F>(
        &mut self,
        sig: Hash256,
        frame: u64,
        store: &mut SnapshotStore,
        step: &mut F,
    ) -> Result<(), EvalError>
    where
        F: FnMut(&mut S) -> Result<f32, EvalError>,
    {
        let residual: f32 = step(&mut self.state)?;
        // M3 残差记录的下游消费者（E3）：帧末残差喂入检测器——只观测。
        self.divergence.observe(residual);
        self.frame = frame;
        self.frame_residuals.push(residual);
        let bytes: Vec<u8> = self.codec.encode(&self.state);
        store.insert((sig, frame), bytes);
        Ok(())
    }

    /// 当前活状态（`X ⊕ Λ` 的帧末值）。
    pub fn state(&self) -> &S {
        &self.state
    }

    /// 当前活状态的确定性字节（快照同款编码；对拍 / 指纹用）。
    pub fn state_bytes(&self) -> Vec<u8> {
        self.codec.encode(&self.state)
    }

    /// 模式元数据（(ε, n) 缓存键分量）。
    pub fn mode(&self) -> SolveModeMeta {
        self.mode
    }

    /// 当前已推进到的帧。
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// 任一帧步进后是否处于发散趋势（E3 事后指标；只观测、不改写）。
    pub fn diverged(&self) -> bool {
        self.divergence.diverged()
    }

    /// 发散检测器（只读；窗口 / 余量 / 采样数可查）。
    pub fn divergence(&self) -> &DivergenceDetector {
        &self.divergence
    }

    /// 实际步进过的帧末残差序列（顺序 = 帧号升序；快照装配不追加）。
    pub fn frame_residuals(&self) -> &[f32] {
        &self.frame_residuals
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Composition, Node, StructureId};

    const SIG: Hash256 = [0x5A; 32];

    // —— 测试特解：一维线性化距离约束的阻尼 GS 扫描（λ 跨轮存活） ——

    /// 状态 = `X ⊕ Λ`：位置 x 与对偶变量 λ（λ 跨轮累积，仅子步边界清零）。
    #[derive(Debug, Clone, PartialEq)]
    struct Spring {
        x: f32,
        lambda: f32,
    }

    /// 目标位置（约束 C = x − TARGET）。
    const TARGET: f32 = 1.0;

    /// 一轮 GS 扫描：Δλ = −C / (Σw + α̃)（Σw = 1、α̃ = 0），λ 累积，
    /// x 只走半步（阻尼 0.5，保证多轮可见收敛曲线）。返回本轮末 |C|。
    fn spring_scan(s: &mut Spring) -> f32 {
        let c: f32 = s.x - TARGET;
        let dlambda: f32 = -c;
        s.lambda += dlambda;
        s.x += 0.5 * dlambda;
        (s.x - TARGET).abs()
    }

    /// 病态 Φ（Lip > 1）：每轮把偏离放大 2 倍，残差单调上升。
    fn expanding_scan(s: &mut Spring) -> f32 {
        s.x *= 2.0;
        s.x.abs()
    }

    /// 快照编解码：magic 2 字节 + x/λ 位模式（大端），长度严格校验。
    #[derive(Debug, Clone, Copy)]
    struct SpringCodec;

    impl LoopCodec<Spring> for SpringCodec {
        fn encode(&self, s: &Spring) -> Vec<u8> {
            let mut out: Vec<u8> = Vec::with_capacity(10);
            out.extend_from_slice(b"SP");
            out.extend_from_slice(&s.x.to_bits().to_be_bytes());
            out.extend_from_slice(&s.lambda.to_bits().to_be_bytes());
            out
        }

        fn decode(&self, bytes: &[u8]) -> Option<Spring> {
            if bytes.len() != 10 || &bytes[0..2] != b"SP" {
                return None;
            }
            let x: f32 = f32::from_bits(u32::from_be_bytes(bytes[2..6].try_into().ok()?));
            let lambda: f32 =
                f32::from_bits(u32::from_be_bytes(bytes[6..10].try_into().ok()?));
            Some(Spring { x, lambda })
        }
    }

    fn initial_spring() -> Spring {
        Spring { x: 0.0, lambda: 0.0 }
    }

    /// 一帧 = 一轮 M3 定数迭代（帧末残差供驱动层喂发散检测）。
    fn spring_frame(state: &mut Spring, mode: &SolveModeMeta) -> Result<f32, EvalError> {
        let report: FixpointReport = run_fixed_point(state, spring_scan, mode)?;
        Ok(report.final_residual().unwrap_or(f32::NAN))
    }

    fn driver(iterations: u32) -> LoopDriver<Spring, SpringCodec> {
        LoopDriver::new(
            "测试环",
            initial_spring(),
            SpringCodec,
            SolveModeMeta::new(1, iterations, RESIDUAL_EPS_DEFAULT),
        )
        .expect("模式合法")
    }

    // —— 验收 1：通用环驱动（M3 定数迭代 + 残差记录 + λ 跨轮存活） ——

    #[test]
    fn run_fixed_point_runs_exact_iterations_and_records_residuals() {
        let mode = SolveModeMeta::new(1, 4, RESIDUAL_EPS_DEFAULT);
        let mut s: Spring = initial_spring();
        let report: FixpointReport = run_fixed_point(&mut s, spring_scan, &mode).unwrap();

        assert_eq!(report.len(), 4, "M3 恰好跑 n 轮（不早停）");
        // 0.5 阻尼线性扫描：残差 1/2, 1/4, 1/8, 1/16（2^-k 精确）。
        assert_eq!(report.residuals, vec![0.5, 0.25, 0.125, 0.0625]);
        assert_eq!(report.final_residual(), Some(0.0625));
        assert!(!report.diverged, "单调下降不判发散");
        assert!(!report.is_empty());

        // λ 跨轮存活：1 + 0.5 + 0.25 + 0.125 = 1.875（若每轮清零，λ 恒为 1）。
        assert_eq!(s.lambda, 1.875, "λ 必须跨轮累积（X ⊕ Λ 的状态语义）");
        assert_eq!(s.x, 0.9375);
    }

    #[test]
    fn fixed_point_mode_rejects_zero_or_nonfinite() {
        for (mode, code) in [
            (SolveModeMeta::new(0, 8, RESIDUAL_EPS_DEFAULT), "P1003"),
            (SolveModeMeta::new(4, 0, RESIDUAL_EPS_DEFAULT), "P1004"),
            (SolveModeMeta::new(4, 8, 0.0), "P1006"),
            (SolveModeMeta::new(4, 8, f32::NAN), "P1006"),
        ] {
            let mut s: Spring = initial_spring();
            let err = run_fixed_point(&mut s, spring_scan, &mode).unwrap_err();
            assert_eq!(err.code, code, "{mode:?}");
            assert!(err.to_string().contains("建议："), "必含建议段（C19）");
            // 报错先于任何 scan：状态不被改写。
            assert_eq!(s, initial_spring(), "非法模式不得推进状态");
        }
    }

    #[test]
    fn divergence_detector_needs_full_window_and_ignores_plateau() {
        // 缓慢上升（每点 +1e-3 > 1e-4）：第 3 个上升步判发散。
        let mut d: DivergenceDetector = DivergenceDetector::new(3, 1e-3);
        assert!(!d.observe(1.0));
        assert!(!d.observe(1.002)); // streak 1
        assert!(!d.observe(1.004)); // streak 2
        assert!(d.observe(1.006), "连续 3 步上升超 ε 余量判发散");
        assert!(d.diverged());
        assert_eq!(d.samples(), 4);

        // 平台微振（上升量 ≤ ε 余量）不计发散。
        assert!(!DivergenceDetector::scan(&[1.0, 1.00005, 1.0, 1.00005], 3, 1e-4));
        assert!(!DivergenceDetector::scan(&[1.0, 0.5, 0.4, 0.3], 3, 1e-4));
    }

    // —— 验收 4：Lip > 1 病态 Φ ⇒ 定数步跑完但报告 diverged ——

    #[test]
    fn expansive_phi_diverges_but_m3_still_runs_full_steps() {
        let mode = SolveModeMeta::new(1, 8, RESIDUAL_EPS_DEFAULT);
        let mut s = Spring { x: 1e-3, lambda: 0.0 };
        let report = run_fixed_point(&mut s, expanding_scan, &mode).unwrap();

        assert!(report.diverged, "残差连续上升必须判发散（E3）");
        assert_eq!(report.len(), 8, "发散检测只观测：M3 仍跑完定数 8 步");
        // 1e-3 × 2^8 = 0.256（二进制精确）。
        assert_eq!(s.x, 0.256, "状态被完整推进到第 n 步（无早停）");
        assert_eq!(report.final_residual(), Some(0.256));
    }

    // —— 验收 2：seek 三路径字节级对拍 ——

    #[test]
    fn seek_three_paths_are_bitwise_identical() {
        const N: u64 = 10;
        const MID: u64 = 5;
        let mode = SolveModeMeta::new(1, 2, RESIDUAL_EPS_DEFAULT);
        let step = |s: &mut Spring| spring_frame(s, &mode);

        // 路径 A：顺序推进 0 → N（快照库随帧累积）。
        let mut a = driver(2);
        let mut store_a: SnapshotStore = SnapshotStore::new();
        let mut sequential: Vec<Vec<u8>> = Vec::new();
        for f in 0..=N {
            a.advance_to(SIG, f, &mut store_a, step).expect("顺序推进");
            sequential.push(a.state_bytes());
        }

        // 路径 B：显式重放（不碰快照库，直接从 x₀ 逐帧重算到 N）。
        let mut replayed: Spring = initial_spring();
        for _ in 0..N {
            spring_frame(&mut replayed, &mode).expect("重放帧");
        }
        let replayed_bytes: Vec<u8> = SpringCodec.encode(&replayed);

        // 路径 C：帧 MID 快照 → 恢复 → 续推到 N（新驱动 + 只含中间快照的库）。
        let mut store_c: SnapshotStore = SnapshotStore::new();
        store_c.insert(
            (SIG, MID),
            store_a.get(&(SIG, MID)).expect("中间快照存在").clone(),
        );
        let mut c = driver(2);
        c.advance_to(SIG, MID, &mut store_c, step).expect("快照装配");
        assert_eq!(c.frame(), MID);
        assert!(c.frame_residuals().is_empty(), "快照装配不产生步进");
        c.advance_to(SIG, N, &mut store_c, step).expect("续推");

        // 三路径逐位一致（位置 + λ 的位模式；不只是数值近似）。
        assert_eq!(replayed_bytes, sequential[N as usize], "显式重放 ≡ 顺序推进");
        assert_eq!(c.state_bytes(), sequential[N as usize], "快照续算 ≡ 顺序推进");
        assert_eq!(
            store_c.get(&(SIG, N)),
            store_a.get(&(SIG, N)),
            "终态快照字节逐位一致"
        );
        assert_eq!(c.frame_residuals().len(), (N - MID) as usize, "续算残差条数 = 步进帧数");

        // 全部帧两两不同（防「收敛到常值使对拍空转」，帧 0 = 初态）。
        for f in 1..=N as usize {
            assert_ne!(
                sequential[f], sequential[f - 1],
                "第 {f} 帧状态必须与上一帧不同（轨道真在推进）"
            );
        }
    }

    #[test]
    fn cold_seek_is_structured_p3012_and_writes_nothing() {
        let mode = SolveModeMeta::new(1, 2, RESIDUAL_EPS_DEFAULT);
        let mut d = driver(2);
        let mut store: SnapshotStore = SnapshotStore::new();
        let err = d
            .advance_to(SIG, 3, &mut store, |s| spring_frame(s, &mode))
            .unwrap_err();
        assert_eq!(err.code, "P3012");
        assert!(err.location.contains("测试环"), "三段式一【在哪】：{}", err.location);
        assert!(err.problem.contains("快照"), "三段式二【什么事】：{}", err.problem);
        assert!(err.fix.contains("顺序推进"), "三段式三【怎么改】：{}", err.fix);
        assert!(err.to_string().contains("建议："), "必含建议段（C19）");
        assert!(store.is_empty(), "P3012 报错路径不得写入任何快照");

        // frame = 0 合法初值：(sig, 0) 恒可恢复。
        d.advance_to(SIG, 0, &mut store, |s| spring_frame(s, &mode))
            .expect("首帧");
        assert_eq!(store.len(), 1);
        assert_eq!(d.state_bytes(), SpringCodec.encode(&initial_spring()));
    }

    #[test]
    fn driver_feeds_frame_residuals_to_divergence_detector() {
        // 帧级病态 Φ：每帧 x *= 2（残差逐帧上升超 ε）。
        let mut d = driver(8);
        let mut store: SnapshotStore = SnapshotStore::new();
        // 初态 1e-3：先用快照把 frame 0 换成放大起点。
        let seed: Vec<u8> = SpringCodec.encode(&Spring { x: 1e-3, lambda: 0.0 });
        store.insert((SIG, 0), seed);

        for f in 0..=5u64 {
            d.advance_to(SIG, f, &mut store, |s| {
                s.x *= 2.0;
                Ok(s.x.abs())
            })
            .expect("帧推进");
        }
        assert_eq!(d.frame(), 5);
        assert_eq!(d.frame_residuals().len(), 5, "5 次步进 = 5 个帧末残差");
        assert!(d.diverged(), "帧末残差连续上升必须被驱动层判发散");
        assert_eq!(d.divergence().samples(), 5);
        // 状态确实推进了 5 帧（1e-3 × 2^5 = 0.032）。
        assert_eq!(d.state().x, 0.032);
    }

    // —— 验收 3：(ε, n) 进签名（0x5333 通道定义） ——

    #[test]
    fn mode_metadata_changes_frame_signature() {
        let mut g = Graph::new();
        g.add_node(Node::new(7, StructureId::ConstraintFlow, Composition::Feed))
            .unwrap();
        let defs: ChannelDefs = ChannelDefs::new();
        let base: Hash256 =
            loop_frame_signature(&g, 7, &defs, &SolveModeMeta::new(4, 8, 1e-4)).unwrap();

        // 同参数同签名（确定性）。
        assert_eq!(
            base,
            loop_frame_signature(&g, 7, &defs, &SolveModeMeta::new(4, 8, 1e-4)).unwrap()
        );
        // 改 iterations / residual_eps / substeps ⇒ 签名变。
        for mode in [
            SolveModeMeta::new(4, 9, 1e-4),
            SolveModeMeta::new(4, 8, 1e-3),
            SolveModeMeta::new(5, 8, 1e-4),
        ] {
            assert_ne!(
                base,
                loop_frame_signature(&g, 7, &defs, &mode).unwrap(),
                "模式 {mode:?} 必须改变帧签名（S3 迭代层缓存键 (ε, n)）"
            );
        }
    }

    #[test]
    fn solve_mode_channel_def_is_stable_0x5333() {
        let mode = SolveModeMeta::new(4, 8, 1e-4);
        let def = solve_mode_def(&mode);
        assert_eq!(def.tag, 0x5333, "W1 已落盘的 tag，内核收拢后不得变");
        assert_eq!(def.def_bytes.len(), 13);
        assert_eq!(def.def_bytes[0], 0, "判别 0 = M3 固定步数");
        assert_eq!(&def.def_bytes[1..5], &4u32.to_be_bytes());
        assert_eq!(&def.def_bytes[5..9], &8u32.to_be_bytes());
        assert_eq!(&def.def_bytes[9..13], &1e-4f32.to_bits().to_be_bytes());
        // 判别 + 参数顺序固定：同定义同字节（签名只消费相等性）。
        assert_eq!(def.def_bytes, solve_mode_def(&SolveModeMeta::new(4, 8, 1e-4)).def_bytes);
        assert_eq!(mode.rounds(), 32);
    }

    #[test]
    fn codec_rejects_foreign_bytes() {
        assert!(SpringCodec.decode(&SpringCodec.encode(&initial_spring())).is_some());
        assert_eq!(SpringCodec.decode(b"XX"), None);
        assert_eq!(SpringCodec.decode(&[b'S', b'P', 0, 0]), None, "长度不符");
        assert_eq!(
            SpringCodec.decode(&[b'X', b'P', 0, 0, 0, 0, 0, 0, 0, 0]),
            None,
            "magic 不符"
        );
    }
}
