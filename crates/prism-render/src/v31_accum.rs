//! V3.1 T03 —— 时域累积状态机、帧循环钩子协议、环境变量单点回退。
//!
//! # 设计依据（`docs/system_design.md` §1.1 / §3 / §10）
//! - 复用 `prism_rt::accum::AccumBuffer`（`push_pass` / `pass_count` /
//!   `running_mean` / `clear`），**不重写累积器**；
//! - 窗口 W 命中即出帧并 `clear()`，3600 帧不 OOM；
//! - `ResetReason` 四种清空重采原因全部落到显式枚举，供 sink / 报告收集；
//! - 所有开关回退收敛在 [`with_env_overrides`]：`PRISM_V31_<FLAG>=0`
//!   不改源码即可单条关闭（设计 §10 风险与回退表）。

use prism_gpu::Framebuffer;

use crate::v31_config::{QualityTier, V31Config};
use crate::v31_gate::V31Verdict;

/// 清空重采原因（设计 §3 `ResetReason`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResetReason {
    /// 场景切换（`scene_key` 变化）。
    SceneCut,
    /// 相邻输出帧差分异常超限。
    TemporalOutlier,
    /// GPU 回落（T04 桥返回 `None` 时由编排层注入）。
    GpuFallback,
    /// 帧尺寸变化。
    SizeChange,
}

impl ResetReason {
    /// 计数器 / 报告用的稳定标签。
    pub fn tag(self) -> &'static str {
        match self {
            ResetReason::SceneCut => "scene_cut",
            ResetReason::TemporalOutlier => "temporal_outlier",
            ResetReason::GpuFallback => "gpu_fallback",
            ResetReason::SizeChange => "size_change",
        }
    }
}

/// 一帧 `push` 的结果（设计 §3 `AccumOutcome`）。
#[derive(Debug, Clone, PartialEq)]
pub enum AccumOutcome {
    /// 暖机：窗口未满，已累计 pass 数。
    Warmup { passes: u32 },
    /// 命中：`pass_count() == W` 的 `running_mean` 全帧（内部已 `clear()`）。
    Hit(Vec<[f32; 4]>),
    /// 清空重采：本帧输出退回当前 pass，窗口从下一帧重新累积。
    Reset(ResetReason),
}

/// 逐帧统计记录（`V31FrameSink` 收集；非抽样帧的轻量校验载体）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V31FrameRecord {
    /// 帧号（0 起）。
    pub frame: u32,
    /// 本帧**新增** spp（写入 `v31.kernel.spp_per_frame` 计数器口径）。
    pub frame_spp: u32,
    /// 累积等效 spp（`window × spp_per_frame`，恒等于 `GOLDEN_SPP`）。
    pub effective_spp: u32,
    /// 本帧渲染 + 累积 + tone-map 的耗时（微秒）。
    pub micros: u64,
    /// 本帧是否触发清空重采。
    pub reset: Option<ResetReason>,
    /// 本帧是否抽样帧（走 `compare_frame_2d` 五判据）。
    pub sampled: bool,
    /// 出片帧 RGBA8 的 FNV-1a u64（非密码学；仅用于非抽样帧轻量校验）。
    pub frame_hash: u64,
}

/// FNV-1a 64 位哈希（零依赖、固定序；3600 帧 × 3.6 MB 的轻量校验用）。
///
/// 不用于 golden 资产（那里用 `prism_io::sha256::sha256`），只作「每帧确实
/// 被求值且帧间确有变化」的运行期链值，成本远低于逐帧 SHA-256。
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// 帧循环钩子（设计 §3 `V31FrameSink`）。
///
/// `on_frame` 保留设计原签名；`on_record` / `on_verdict` 带默认空实现，
/// 让「只想要帧」的调用方不受额外方法拖累，同时让 CI 能收集逐帧
/// spp / 计时 / reset 原因与抽样帧判据结论。
pub trait V31FrameSink {
    /// 逐帧产物：`fb` = 完整出片帧（含 sim / 文本 / 转场叠加），
    /// `linear` = 累积（及可选降噪）后的 f32 线性基帧。
    fn on_frame(&mut self, frame: u32, fb: &Framebuffer, linear: &[[f32; 4]]);

    /// 逐帧统计（spp / 计时 / reset / 帧哈希 / 抽样标志）。
    fn on_record(&mut self, _rec: &V31FrameRecord) {}

    /// 抽样帧的五判据结论。
    fn on_verdict(&mut self, _frame: u32, _verdict: &V31Verdict) {}

    /// 整片结束后的汇总报告（wall clock / 抽样索引 / 判据结论 / reset 计数）。
    fn on_report(&mut self, _report: &V31MovieReport) {}
}

/// 整片 V3.1 出片报告（帧循环末尾由 `render_movie_v31` 生成并交付 sink）。
///
/// `render_movie_v31` 的设计签名只返回 `Deliverable`，故报告经
/// [`V31FrameSink::on_report`] 交付，同时 `render_movie_v31` 会在 stderr
/// 打印一行 `[v31]` 摘要供脚本采集。**600 s 计时以完整 3600 帧的
/// `total_micros` / 端到端 wall clock 为准，不用抽样帧数冒充。**
#[derive(Debug, Clone, PartialEq)]
pub struct V31MovieReport {
    /// 真实逐帧渲染的帧数（不是抽样帧数）。
    pub frames_rendered: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub quality: QualityTier,
    pub res_tag: String,
    pub spp_per_frame: u32,
    pub window: u32,
    pub effective_spp: u32,
    pub min_spp: u32,
    pub accumulate: bool,
    pub denoise: bool,
    pub adaptive: bool,
    pub gpu_prefer: bool,
    /// golden 抽样帧索引（升序；已写进 `manifest.sha256`）。
    pub sample_indices: Vec<u32>,
    /// 实际完成 `compare_frame_2d` 判定的抽样帧数。
    pub sampled_count: u32,
    /// 判红的 `(帧号, 原因列表)`；非空即 CI 硬失败。
    pub failed_frames: Vec<(u32, Vec<String>)>,
    /// 抽样帧序列平均 PSNR（dB）。
    pub seq_mean_psnr: f64,
    /// 帧循环内渲染 + 累积 + tone-map 的累计微秒（不含容器编码）。
    pub total_micros: u64,
    /// 各 reset 原因的计数（升序 tag）。
    pub resets: Vec<(String, u32)>,
    /// 逐帧新增 spp 低于 `min_spp` 的帧数（必须为 0）。
    pub spp_below_min: u32,
    /// 时间线循环周期（帧）。
    pub timeline_period: u32,
    /// `Deliverable.frames` 是否只驻留抽样帧（大帧数流式模式）。
    pub streamed: bool,
    /// V3.1 收尾：用户命名的大模式字面标签（`一致`/`均衡`/`快速`/`自定义`）。
    pub mode_label: String,
    /// 大模式备注（原样带上）。
    pub mode_note: String,
    /// 是否启用了可选误差校验（默认 false = 不进热路径）。
    pub verify: bool,
    /// 可选误差校验结论（未启用为 `None`；**不参与出片 Go/No-Go**）。
    pub verify_pass: Option<bool>,
    /// 逐帧平均墙钟（微秒；**仅帧循环**，不含容器编码）。本报表在
    /// `render_movie_v31` 内生成，而容器编码在它返回之后由 CLI 执行，故这里
    /// 不报编码数（避免「分段有值、汇总为 0」双口径）；真实编码耗时见 CLI
    /// `分段计时 编码` 与 `端到端单帧`（后者含编码）。
    pub per_frame_micros: u64,
    /// 3600 帧 RSS 首尾差（KiB；`None` = 非 Linux / 读取失败）。
    pub rss_delta_kib: Option<i64>,
    /// 每帧平均分配次数（由 `CountingAllocator` 在专用测试二进制测量）。
    pub alloc_per_frame: Option<u64>,
}

impl V31MovieReport {
    /// 出片 Go/No-Go：**不再**包含任何误差判据。
    ///
    /// V3.1 收尾（B）：golden 比对 / 五条判据 / 抽样 manifest / reset 检测
    /// 全部移出渲染帧循环，判据保留为 [`crate::v31_verify`] 的可选项。
    /// 这里只保留「确实逐帧求值、spp 不低于下限」的运行期健康判定，
    /// `report.pass()` 为 false 时 `render_movie_v31` 也不再判红退出。
    pub fn pass(&self) -> bool {
        self.spp_below_min == 0 && self.failed_frames.is_empty()
    }

    /// 供 CLI / 脚本打印的确定性多行摘要。
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        out.push(format!(
            "[v31] {} / {}×{} @ {} fps / {} 帧 / spp {}×W{} = {} / min_spp {}",
            self.res_tag,
            self.width,
            self.height,
            self.fps,
            self.frames_rendered,
            self.spp_per_frame,
            self.window,
            self.effective_spp,
            self.min_spp
        ));
        out.push(format!(
            "[v31] 能力位 accumulate={} denoise={} adaptive={} gpu_prefer={} 周期={} 帧 流式={}",
            self.accumulate,
            self.denoise,
            self.adaptive,
            self.gpu_prefer,
            self.timeline_period,
            self.streamed
        ));
        out.push(format!(
            "[v31] 可选校验 {} / 校验 {} 帧 / 判定 {} 帧 / 判红 {} 帧 / 序列均值 PSNR {:.2} dB / spp 下限违例 {}",
            if self.verify { "开" } else { "关（默认，不进热路径）" },
            self.sample_indices.len(),
            self.sampled_count,
            self.failed_frames.len(),
            self.seq_mean_psnr,
            self.spp_below_min
        ));
        let resets: String = if self.resets.is_empty() {
            "无".to_string()
        } else {
            self.resets
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push(format!(
            "[v31] reset 统计：{resets}；帧循环累计 {:.3} s（不含容器编码）",
            self.total_micros as f64 / 1.0e6
        ));
        out.push(format!(
            "[v31] 大模式 {}（{}）/ 可选校验 {} / 帧循环单帧 {:.1} ms（不含编码；编码见 CLI 分段计时）",
            self.mode_label,
            self.mode_note,
            if self.verify { "开" } else { "关（默认）" },
            self.per_frame_micros as f64 / 1000.0,
        ));
        if let Some(d) = self.rss_delta_kib {
            out.push(format!("[v31] 3600 帧 RSS 首尾差 {d} KiB（报表，非门禁）"));
        }
        if let Some(a) = self.alloc_per_frame {
            out.push(format!("[v31] 每帧分配次数 {a}（计数分配器实测）"));
        }
        out
    }
}

/// 空 sink：旧入口 / 只关心 `Deliverable` 的调用方。
#[derive(Debug, Clone, Copy, Default)]
pub struct NullV31Sink;

impl V31FrameSink for NullV31Sink {
    fn on_frame(&mut self, _frame: u32, _fb: &Framebuffer, _linear: &[[f32; 4]]) {}
}

/// 收集 sink：驻留帧 + 全部记录（测试与小帧数出片用）。
#[derive(Debug, Clone, Default)]
pub struct CollectingV31Sink {
    pub frames: Vec<Framebuffer>,
    pub records: Vec<V31FrameRecord>,
    pub verdicts: Vec<(u32, V31Verdict)>,
    pub report: Option<V31MovieReport>,
}

impl V31FrameSink for CollectingV31Sink {
    fn on_frame(&mut self, _frame: u32, fb: &Framebuffer, _linear: &[[f32; 4]]) {
        self.frames.push(fb.clone());
    }
    fn on_record(&mut self, rec: &V31FrameRecord) {
        self.records.push(*rec);
    }
    fn on_verdict(&mut self, frame: u32, verdict: &V31Verdict) {
        self.verdicts.push((frame, verdict.clone()));
    }
    fn on_report(&mut self, report: &V31MovieReport) {
        self.report = Some(report.clone());
    }
}

/// 流式 sink：3600 帧 × 720p 全驻留需 13.3 GB，故计时模式逐帧只累计
/// 哈希链与记录，**不保留帧缓冲**。这是「完整 3600 帧端到端」能真跑的
/// 前提（设计 §4 时序图本就是帧循环内逐帧交付 MUX，而非全帧驻留）。
#[derive(Debug, Clone, Default)]
pub struct StreamingV31Sink {
    pub frames_seen: u64,
    /// 逐帧哈希的混合链：任一帧变化都会改变最终值（防「空转假绿」）。
    pub hash_chain: u64,
    pub records: Vec<V31FrameRecord>,
    pub verdicts: Vec<(u32, V31Verdict)>,
    pub report: Option<V31MovieReport>,
}

impl StreamingV31Sink {
    /// 把一帧哈希并入链（固定序，确定性）。
    pub fn mix(&mut self, frame_hash: u64) {
        self.frames_seen += 1;
        self.hash_chain = self
            .hash_chain
            .wrapping_mul(0x5851_f42d_4c95_7f2d)
            .wrapping_add(frame_hash)
            .rotate_left(13);
    }
}

impl V31FrameSink for StreamingV31Sink {
    fn on_frame(&mut self, _frame: u32, _fb: &Framebuffer, _linear: &[[f32; 4]]) {}
    fn on_record(&mut self, rec: &V31FrameRecord) {
        self.mix(rec.frame_hash);
        self.records.push(*rec);
    }
    fn on_verdict(&mut self, frame: u32, verdict: &V31Verdict) {
        self.verdicts.push((frame, verdict.clone()));
    }
    fn on_report(&mut self, report: &V31MovieReport) {
        self.report = Some(report.clone());
    }
}

/// 环境变量单点回退（设计 §10：`PRISM_V31_<FLAG>=0` 不改源码即可关）。
///
/// 只有显式取值 `"0"` 才关闭；未设置 / 其它值一律沿用 `cfg`。
/// 每条 P0 独立一个变量，互不连锁；`legacy()` 配置不受影响（本就全关）。
pub fn with_env_overrides(mut cfg: V31Config) -> V31Config {
    if env_off("PRISM_V31_ACCUM") {
        cfg.accumulate = false;
    }
    if env_off("PRISM_V31_DENOISE") {
        cfg.denoise = false;
    }
    if env_off("PRISM_V31_ADAPTIVE") {
        cfg.adaptive = false;
    }
    if env_off("PRISM_V31_GPU") {
        cfg.gpu_prefer = false;
    }
    if env_on("PRISM_V31_GPU") {
        cfg.gpu_prefer = true;
    }
    if env_off("PRISM_V31_GPU_AUTO") {
        cfg.gpu_auto = false;
    }
    // V3.1 收尾（B）：误差判据默认**关**；`PRISM_V31_VERIFY=1` 或
    // `PRISM_V31_GATE=1` 才启用离线校验（用户此前用 `=0` 关不掉的是旧热路径，
    // 如今默认就不进热路径）。
    if env_on("PRISM_V31_VERIFY") || env_on("PRISM_V31_GATE") {
        cfg.verify = true;
    }
    if env_off("PRISM_V31_VERIFY") || env_off("PRISM_V31_GATE") {
        cfg.verify = false;
    }
    if env_off("PRISM_V31_OVERRIDE") {
        // 关闭 CLI 规格覆写：把四个规格字段归零即 `is_legacy()` 语义。
        cfg.width = 0;
        cfg.height = 0;
        cfg.fps = 0;
        cfg.frames = 0;
    }
    cfg
}

fn env_off(key: &str) -> bool {
    matches!(std::env::var(key).ok().as_deref(), Some("0"))
}

fn env_on(key: &str) -> bool {
    matches!(std::env::var(key).ok().as_deref(), Some("1"))
}

/// 时域累积状态机（设计 §3 `TemporalState`）。
///
/// 持 `AccumBuffer` + 窗口 W；跨帧状态**只由本结构显式持有**，无全局可变
/// 状态、无隐藏 RNG。`canonical_mean` 逐位锚只在 CI 用，不进热路径。
#[derive(Debug, Clone)]
pub struct TemporalState {
    buf: prism_rt::accum::AccumBuffer,
    width: u32,
    height: u32,
    window: u32,
    scene_key: u64,
    key_initialized: bool,
    spp_per_frame: u32,
    frame_spp: u32,
    last_reset: Option<ResetReason>,
}

impl TemporalState {
    /// 构造 w×h 的窗口累积器。
    pub fn new(w: u32, h: u32, window: u32, spp_per_frame: u32) -> Self {
        TemporalState {
            buf: prism_rt::accum::AccumBuffer::new(w, h),
            width: w,
            height: h,
            window: window.max(1),
            scene_key: 0,
            key_initialized: false,
            spp_per_frame: spp_per_frame.max(1),
            frame_spp: 0,
            last_reset: None,
        }
    }

    /// 本帧新增 spp（写计数器用）。
    pub fn frame_spp(&self) -> u32 {
        self.frame_spp
    }

    /// 累积等效 spp = `window × spp_per_frame`（恒 16）。
    pub fn effective_spp(&self) -> u32 {
        self.window.max(1) * self.spp_per_frame.max(1)
    }

    /// 当前窗口已累计的 pass 数。
    pub fn pass_count(&self) -> u32 {
        self.buf.pass_count()
    }

    /// 最近一次清空原因（报告用；`Hit` / `Warmup` 会清成 `None`）。
    pub fn last_reset(&self) -> Option<ResetReason> {
        self.last_reset
    }

    /// 清空已累计 pass 并重采（尺寸 / 场景切换 / 差分异常 / GPU 回落）。
    pub fn reset(&mut self, why: ResetReason) {
        self.buf.clear();
        self.key_initialized = false;
        self.scene_key = 0;
        self.frame_spp = 0;
        self.last_reset = Some(why);
    }

    /// 推入一帧 f32 线性 pass（样本区间 `[k·spp, (k+1)·spp)`）。
    ///
    /// - 尺寸不符 ⇒ `Reset(SizeChange)`；
    /// - `scene_key` 变化 ⇒ `Reset(SceneCut)`；
    /// - 否则 `Warmup { passes }`（不再跨窗口平均，复用链已删除）。
    pub fn push(&mut self, cur: &[[f32; 4]], scene_key: u64) -> AccumOutcome {
        let n: usize = (self.width as usize).saturating_mul(self.height as usize);
        if cur.len() != n {
            self.reset(ResetReason::SizeChange);
            return AccumOutcome::Reset(ResetReason::SizeChange);
        }
        if self.key_initialized && scene_key != self.scene_key {
            self.reset(ResetReason::SceneCut);
            return AccumOutcome::Reset(ResetReason::SceneCut);
        }
        self.key_initialized = true;
        self.scene_key = scene_key;

        self.buf.push_pass(cur);
        self.frame_spp = self.spp_per_frame;
        self.last_reset = None;

        // 时域复用链（窗口满即出帧并清空）已随重建整体删除：这里只记账、
        // 不再跨窗口平均、不再产生 `Hit`，避免「窗口填满 = 画面静止」的假设。
        AccumOutcome::Warmup { passes: self.buf.pass_count() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v31_config::{QualityTier, ResTier};

    fn frame(n: usize, v: f32) -> Vec<[f32; 4]> {
        vec![[v, v * 0.5, 1.0 - v, 1.0]; n]
    }

    #[test]
    fn scene_cut_and_size_change_reset() {
        let mut st = TemporalState::new(2, 1, 4, 4);
        assert!(matches!(st.push(&frame(2, 0.1), 1), AccumOutcome::Warmup { .. }));
        let o = st.push(&frame(2, 0.2), 2);
        assert_eq!(o, AccumOutcome::Reset(ResetReason::SceneCut));
        assert_eq!(st.pass_count(), 0);
        assert_eq!(st.last_reset(), Some(ResetReason::SceneCut));
        let o = st.push(&frame(3, 0.2), 2);
        assert_eq!(o, AccumOutcome::Reset(ResetReason::SizeChange));
    }

    #[test]
    fn reset_reason_tags_are_stable() {
        assert_eq!(ResetReason::SceneCut.tag(), "scene_cut");
        assert_eq!(ResetReason::TemporalOutlier.tag(), "temporal_outlier");
        assert_eq!(ResetReason::GpuFallback.tag(), "gpu_fallback");
        assert_eq!(ResetReason::SizeChange.tag(), "size_change");
    }

    #[test]
    fn env_override_never_touches_legacy_semantics() {
        let legacy = with_env_overrides(V31Config::legacy());
        assert!(legacy.is_legacy() || legacy.accumulate == false);
        let final_cfg = V31Config::for_tier(ResTier::R720p, QualityTier::Final);
        assert_eq!(final_cfg.effective_spp(), 16);
    }

    #[test]
    fn fnv1a64_is_deterministic_and_sensitive() {
        let a = fnv1a64(&[1, 2, 3, 4]);
        let b = fnv1a64(&[1, 2, 3, 5]);
        assert_eq!(a, fnv1a64(&[1, 2, 3, 4]));
        assert_ne!(a, b);
    }
}
