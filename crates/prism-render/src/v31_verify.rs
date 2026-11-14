//! V3.1 收尾 B —— **可选**误差校验：五条判据 + 16 spp golden 全部移出热路径。
//!
//! # 为什么要这样切
//! 旧 `render_movie_v31` 在每个抽样帧内做：16 spp golden 重渲 → 逐帧五判据
//! → 序列 PSNR → `manifest.sha256` 写盘 → reset 统计 → 任一判红即 `R5031` 退出。
//! 这些把省下的时间吃光，还出不了片。
//!
//! 本模块把最贵的 golden 重渲与全部判据搬到帧循环结束之后（或完全不跑）：
//! - 主链路 `render_movie_v31` 默认 `verify=false`：帧循环内零 golden、零判据、
//!   零 manifest 写入、零判红退出，只做渲染 + 重建 + 出片 + 计时；
//! - `verify=true` 时，帧循环只在抽样帧收集 `got` 基帧的 RGBA8 副本与该帧
//!   请求（每帧一次小拷贝，不重渲、不比较）；
//! - 循环结束后 [`finalize_verify`] 才逐帧重渲 16 spp 权威 golden，把整段序列
//!   喂入 [`V31SequenceAggregate`]，五条判据各算一次聚合量并判定一次；
//! - 判据本身一条没删：阈值来自 [`crate::v31_config`]，负向用例仍判红。

use std::collections::BTreeMap;
use std::path::PathBuf;

use prism_gpu::SceneRenderRequest;

use crate::v31_config::V31Config;
use crate::v31_gate::{V31ErrorReport, V31SequenceAggregate};
use crate::v31_golden::GoldenStore;
use crate::v31_sample::SamplePlan;

/// 抽样帧在热路径里收集的**最小**凭证（不含 golden、不含判据结果）。
#[derive(Debug, Clone)]
pub struct V31VerifySample {
    pub frame: u32,
    /// 重建链路输出的 3D 基帧（tone-map 后 RGBA8；不含 sim/文本/转场叠加）。
    pub got: Vec<[u8; 4]>,
    /// 前一个抽样帧的 `got`（保留字段，供上层诊断）。
    pub prev: Option<Vec<[u8; 4]>>,
    /// 该帧自身请求；`None` = 该帧无 3D 场景（黑帧 / 透传帧），跳过判定。
    pub req: Option<SceneRenderRequest>,
}

/// 帧循环侧的收集器（只有 `verify=true` 才被填充）。
#[derive(Debug, Clone, Default)]
pub struct V31VerifyCollector {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<V31VerifySample>,
    /// reset 事件（仅 `verify=true` 时登记的默认离线统计口径）：(帧号, 原因标签)。
    pub reset_events: Vec<(u32, String)>,
}

impl V31VerifyCollector {
    /// 构造空收集器。
    pub fn new(width: u32, height: u32) -> Self {
        V31VerifyCollector { width, height, samples: Vec::new(), reset_events: Vec::new() }
    }

    /// 收集一帧（热路径：仅 `got` 的克隆 + 可选 `req` 克隆，无重渲/无比较）。
    pub fn push(&mut self, sample: V31VerifySample) {
        self.samples.push(sample);
    }

    /// 登记一帧的 reset 原因标签（默认不调用；离线汇总用）。
    pub fn record_reset(&mut self, frame: u32, tag: &str) {
        self.reset_events.push((frame, tag.to_string()));
    }
}

/// 离线汇总 reset 统计（循环后调用，不在帧循环里做 BTreeMap 聚合）。
pub fn summarize_resets(collector: &V31VerifyCollector) -> Vec<(String, u32)> {
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for (_, tag) in &collector.reset_events {
        *counts.entry(tag.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

/// 可选校验：把抽样计划写进 `manifest.sha256`（**帧循环之外**，默认不调用）。
///
/// 落点可由 `PRISM_V31_VERIFY_ROOT` 指定（默认进程临时目录），绝不触碰
/// 仓库内 `docs/v3.1-golden` 受保护资产。返回 manifest 路径供报告列明。
pub fn write_verify_manifest(plan: &SamplePlan) -> Result<PathBuf, String> {
    let root: PathBuf = match std::env::var("PRISM_V31_VERIFY_ROOT") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => std::env::temp_dir().join(format!("prism-v31-verify-{}", std::process::id())),
    };
    std::fs::create_dir_all(&root)
        .map_err(|e| format!("创建校验目录 {} 失败：{e}", root.display()))?;
    let store: GoldenStore = GoldenStore::with_root(&root);
    let path: PathBuf = store.manifest_path();
    if !path.exists() {
        std::fs::write(&path, b"")
            .map_err(|e| format!("初始化 {}: {e}", path.display()))?;
    }
    plan.write_to_manifest(&store)
        .map_err(|e| format!("写抽样 manifest 失败：{e}"))?;
    Ok(path)
}

/// 可选校验报告（**不进 Go/No-Go**：出片链路永远成功）。
#[derive(Debug, Clone, PartialEq)]
pub struct V31VerifyReport {
    /// 实际喂入序列聚合器的帧数。
    pub checked: u32,
    /// 收集到但无可比 golden 的帧数（黑帧 / 无请求）。
    pub skipped: u32,
    /// 判据①：整段序列汇总后的全局最大 max-channel delta。
    pub max_delta: u8,
    /// 判据①：整段序列汇总后的 99 分位 delta。
    pub p99_delta: u8,
    /// 逐帧最差 PSNR（dB，诊断用；判定只看序列均值）。
    pub worst_psnr_db: f64,
    /// 判据②：序列平均 PSNR（dB，peak 固定 255）。
    pub seq_mean_psnr: f64,
    /// 判据③：序列均值归一化 L1 总变差 D。
    pub hist_bin_max_rel: f64,
    /// 判据④：全序列强边缘汇总保留率。
    pub edge_retention_min: f64,
    /// 判据⑤：全序列相邻帧对异常像素汇总占比。
    pub temporal_outlier_ratio: f64,
    /// 五条序列级聚合判据是否全过（含「至少判定过一帧」）。
    pub pass: bool,
    /// 判红原因（整段序列只判定一次，不逐帧累积）。
    pub reasons: Vec<String>,
    /// 逐帧误差报告（诊断）。
    pub per_frame: Vec<(u32, V31ErrorReport)>,
    /// golden 重渲 + 判据总耗时（微秒）。
    pub micros: u64,
}

/// PSNR 的确定性文本：`+inf` = 逐位一致（不是误差），`-inf` = 无有效帧。
fn fmt_psnr(v: f64) -> String {
    if v == f64::INFINITY {
        "inf（逐位一致）".to_string()
    } else if v == f64::NEG_INFINITY {
        "n/a（未判定）".to_string()
    } else {
        format!("{v:.2}")
    }
}

impl V31VerifyReport {
    /// 供命令行 / 脚本打印的确定性摘要。
    pub fn summary_lines(&self) -> Vec<String> {
        vec![
            format!(
                "[v31-verify] 判定 {} 帧 / 跳过 {} 帧 / max_delta {} / p99 {} / 最差逐帧 PSNR {} dB / 序列均值 {} dB",
                self.checked,
                self.skipped,
                self.max_delta,
                self.p99_delta,
                fmt_psnr(self.worst_psnr_db),
                fmt_psnr(self.seq_mean_psnr)
            ),
            format!(
                "[v31-verify] 亮度 L1 总变差 D {:.4} / 边缘保留 {:.4} / 时序异常 {:.4} / 结论 {} / 校验耗时 {:.3} 秒（不进出片热路径）",
                self.hist_bin_max_rel,
                self.edge_retention_min,
                self.temporal_outlier_ratio,
                if self.pass { "通过" } else { "未通过（仅报告，不阻断出片）" },
                self.micros as f64 / 1.0e6
            ),
        ]
    }
}

/// 帧循环结束后执行：重渲 16 spp 权威 golden + 整段序列聚合判定一次。
///
/// 这是唯一还跑 golden 的地方，且默认根本不被调用。纯函数（除计时与原子
/// 计数器外无副作用），不写 manifest、不抛 `R5031`。
pub fn finalize_verify(
    samples: &[V31VerifySample],
    cfg: &V31Config,
    width: u32,
    height: u32,
) -> V31VerifyReport {
    let started = std::time::Instant::now();
    let mut agg: V31SequenceAggregate = V31SequenceAggregate::new();
    let mut checked: u32 = 0;
    let mut skipped: u32 = 0;
    let mut per_frame: Vec<(u32, V31ErrorReport)> = Vec::new();
    let mut prev_golden: Option<Vec<[u8; 4]>> = None;
    let mut prev_got: Option<Vec<[u8; 4]>> = None;

    for s in samples {
        let Some(req) = s.req.as_ref() else {
            skipped += 1;
            continue;
        };
        // 独立 golden：用该帧自身请求逐帧独立重渲 16 spp（同源 frame-0
        // 确定性种子，不锁存、不自比、不消费 got 缓存）。
        let golden: Vec<[u8; 4]> = crate::v31_temporal::independent_golden_u8(req);
        let want: usize = (width as usize).saturating_mul(height as usize);
        if golden.len() != want || s.got.len() != want {
            skipped += 1;
            continue;
        }
        let r: V31ErrorReport = agg.push_expected(
            &golden,
            &s.got,
            prev_golden.as_deref(),
            prev_got.as_deref(),
            width,
            height,
        );
        prev_golden = Some(golden);
        prev_got = Some(s.got.clone());
        per_frame.push((s.frame, r));
        checked += 1;
    }

    // §8.1 分档上限：`均衡` / `快速` 各一组，判定仍是确定性布尔。
    let limits = crate::v31_modes::judge_limits(cfg.mode);
    let (report, verdict) = agg.finalize_with(&limits);
    let mut reasons: Vec<String> = verdict.reasons;
    let pass: bool = checked > 0 && verdict.pass;
    if checked == 0 {
        reasons.push("无任何可比 golden 帧（抽样帧全为黑帧 / 无 3D 请求）".to_string());
    }
    crate::perf_domains::record(
        crate::perf_domains::Domain::Verify,
        started.elapsed().as_micros() as u64,
    );
    let worst_psnr_db: f64 = if checked == 0 {
        f64::NEG_INFINITY
    } else {
        agg.worst_psnr_db()
    };
    V31VerifyReport {
        checked,
        skipped,
        max_delta: report.max_delta,
        p99_delta: report.p99_delta,
        worst_psnr_db,
        seq_mean_psnr: report.psnr_db,
        hist_bin_max_rel: report.hist_bin_max_rel,
        edge_retention_min: report.edge_retention,
        temporal_outlier_ratio: report.temporal_outlier_ratio,
        pass,
        reasons,
        per_frame,
        micros: started.elapsed().as_micros() as u64,
    }
}

/// 空报告（verify 关闭时用，便于命令行统一打印）。
pub fn disabled_report() -> V31VerifyReport {
    V31VerifyReport {
        checked: 0,
        skipped: 0,
        max_delta: 0,
        p99_delta: 0,
        worst_psnr_db: f64::NEG_INFINITY,
        seq_mean_psnr: f64::NEG_INFINITY,
        hist_bin_max_rel: 0.0,
        edge_retention_min: 1.0,
        temporal_outlier_ratio: 0.0,
        pass: true,
        reasons: Vec::new(),
        per_frame: Vec::new(),
        micros: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v31_config::ResTier;
    use crate::v31_modes::V31Mode;

    #[test]
    fn disabled_report_never_blocks_output() {
        let r = disabled_report();
        assert!(r.pass);
        assert_eq!(r.checked, 0);
        assert!(r.summary_lines()[0].contains("判定 0 帧"));
    }

    #[test]
    fn empty_samples_report_is_explicitly_red_but_non_blocking() {
        let cfg = V31Mode::Consistent.preset(ResTier::R320x180).unwrap();
        let r = finalize_verify(&[], &cfg, 320, 180);
        assert!(!r.pass);
        assert!(!r.reasons.is_empty());
        assert_eq!(r.checked, 0);
    }

    #[test]
    fn collector_pushes_without_rendering() {
        let mut c = V31VerifyCollector::new(8, 8);
        c.push(V31VerifySample {
            frame: 3,
            got: vec![[1u8, 2, 3, 255]; 64],
            prev: None,
            req: None,
        });
        assert_eq!(c.samples.len(), 1);
        let cfg = V31Mode::Consistent.preset(ResTier::R320x180).unwrap();
        let r = finalize_verify(&c.samples, &cfg, 8, 8);
        assert_eq!(r.skipped, 1);
        assert_eq!(r.checked, 0);
    }

    #[test]
    fn reset_summary_is_offline_and_deterministic() {
        let mut c = V31VerifyCollector::new(8, 8);
        c.record_reset(1, "scene_cut");
        c.record_reset(7, "scene_cut");
        c.record_reset(9, "temporal_outlier");
        let rows = summarize_resets(&c);
        assert_eq!(rows, vec![
            ("scene_cut".to_string(), 2),
            ("temporal_outlier".to_string(), 1),
        ]);
    }

    #[test]
    fn sample_manifest_is_written_outside_hot_path() {
        let root = std::env::temp_dir().join(format!("v31-verify-test-{}", std::process::id()));
        std::env::set_var("PRISM_V31_VERIFY_ROOT", &root);
        let plan = crate::v31_sample::SamplePlan {
            indices: vec![3, 7, 11],
            requested: vec![0, 4, 8],
            window: 4,
            frames: 16,
            period: 16,
            scene_count: 1,
            transition_count: 0,
        };
        let path = write_verify_manifest(&plan).expect("抽样 manifest 应写出");
        assert!(path.exists());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("indices=3,7,11"), "抽样索引必须落盘：{text}");
        write_verify_manifest(&plan).unwrap();
        let text2 = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text2.lines().filter(|l| l.starts_with("# v31.sample_frames")).count(), 1);
        std::env::remove_var("PRISM_V31_VERIFY_ROOT");
        let _ = std::fs::remove_dir_all(&root);
    }
}
