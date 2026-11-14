//! V3.1 T03 —— golden 抽样计划与 `manifest.sha256` 抽样索引。
//!
//! # 主理人裁定 1（本批必须照做）
//! 3600 帧 × 720p × 16 spp 的逐帧 golden 等价于 1.2 小时，本沙箱不可承受。
//! golden **只在抽样帧上生成**：每个场景段至少覆盖首帧 / 中段 / 末帧 +
//! 场景切换点前后各 1 帧，抽样帧数 ≥ 12 且覆盖全部场景段；
//! `compare_frame_2d` 只在抽样帧上逐帧判定，其余帧走轻量校验
//! （帧哈希 + 计数器 + spp 下限）。抽样索引写进 `manifest.sha256` 并在
//! 报告里列明。**不得用「只测抽样帧」掩盖全链路计时**。
//!
//! # 时间线循环（`timeline_period_frames`）
//! 示例 DSL 的内容时长只有 3 s，`--frames 3600 --fps 60` 覆写 `duration`
//! 为 60 s 后，帧 36..3599 若不重映射就全是黑帧 —— 那样「600 s 计时」
//! 会变成 36 帧真渲 + 3564 帧零成本黑帧，是假绿。故 v31 链路把帧号
//! 对**内容周期**取模（`t_frame = frame % period`），3600 帧全部落在
//! 真实场景区间内、每帧都付完整 4 spp 线性渲染成本。DSL 场景 IR 不改，
//! 只在 v31 编排层做时间重映射（纯读、不写回 compile）。
//!
//! # 抽样帧必须落在「窗口命中帧」
//! `TemporalState` 只在 `pass_count() == W` 那一帧产出 16 样本的完整
//! `running_mean`；暖机帧只有 1..W−1 个 pass，拿它和 16 spp golden 比
//! 会因采样噪声判红（那不是累积链路的误差，是暖机语义）。故抽样候选
//! 一律对齐到 `frame ≡ W−1 (mod W)` 的窗口命中帧。

use std::collections::BTreeSet;
use std::fs;

use prism_dsl::CompileV2Output;

use crate::v31_golden::{GoldenError, GoldenStore};

/// 抽样帧数下限（主理人裁定 1：≥ 12）。
pub const MIN_SAMPLE_FRAMES: usize = 12;
/// manifest 注释行前缀（`sha256sum -c` 与 `verify_manifest` 均忽略 `#` 行）。
pub const MANIFEST_TAG: &str = "# v31.sample_frames";

/// golden 抽样计划。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplePlan {
    /// 实际抽样帧（升序、去重、全部为窗口命中帧）。
    pub indices: Vec<u32>,
    /// 抽样请求帧（场景首/中/末 + 切换点 ±1，对齐前的原始候选）。
    pub requested: Vec<u32>,
    /// 累积窗口 W。
    pub window: u32,
    /// 影片总帧数。
    pub frames: u32,
    /// 内容周期帧数（时间线循环周期）。
    pub period: u32,
    /// 覆盖的场景段数。
    pub scene_count: usize,
    /// 覆盖的过渡点数。
    pub transition_count: usize,
}

impl SamplePlan {
    /// 该帧是否走 `compare_frame_2d` 五判据。
    pub fn is_sampled(&self, frame: u32) -> bool {
        self.indices.binary_search(&frame).is_ok()
    }

    /// 与 `manifest.sha256` 兼容的单行注释（确定性、可解析）。
    pub fn manifest_line(&self) -> String {
        let list: String = self
            .indices
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{MANIFEST_TAG} window={} frames={} period={} scenes={} transitions={} count={} indices={list}",
            self.window,
            self.frames,
            self.period,
            self.scene_count,
            self.transition_count,
            self.indices.len()
        )
    }

    /// 把抽样索引写进 `manifest.sha256`（一行注释，幂等覆盖旧行）。
    ///
    /// `regenerate_manifest()` 只扫 `frame_*.png` 会重写整份文件，故本方法
    /// **必须在 regenerate 之后调用**。写入是「保留原有全部条目 + 替换
    /// 旧抽样行 + 追加新抽样行」，不触碰任何哈希条目。
    pub fn write_to_manifest(&self, store: &GoldenStore) -> Result<(), GoldenError> {
        let path = store.manifest_path();
        let old: String = fs::read_to_string(&path).unwrap_or_default();
        let mut text: String = String::with_capacity(old.len() + 256);
        for line in old.lines() {
            if line.starts_with(MANIFEST_TAG) {
                continue;
            }
            text.push_str(line);
            text.push('\n');
        }
        text.push_str(&self.manifest_line());
        text.push('\n');
        fs::write(&path, text).map_err(|e| GoldenError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })
    }

    /// 从 manifest 读回抽样索引（报告 / T04 复验用）。
    pub fn read_from_manifest(store: &GoldenStore) -> Option<Vec<u32>> {
        let text: String = fs::read_to_string(store.manifest_path()).ok()?;
        let line: &str = text.lines().find(|l| l.starts_with(MANIFEST_TAG))?;
        let field: &str = line.split_whitespace().find(|p| p.starts_with("indices="))?;
        let list: &str = &field["indices=".len()..];
        let indices: Vec<u32> = list.split(',').filter_map(|s| s.parse().ok()).collect();
        if indices.is_empty() {
            None
        } else {
            Some(indices)
        }
    }
}

/// 内容周期帧数：DSL 场景时间线覆盖的帧数（不受 `apply_to` 的 duration
/// 覆写影响），最小 1。时间线循环用它把 3600 帧铺满真实场景。
pub fn timeline_period_frames(compile: &CompileV2Output) -> u32 {
    let fps: u32 = compile.fps.max(1);
    let mut content: f32 = 0.0;
    for s in &compile.scenes {
        content = content.max(s.start + s.duration);
    }
    if !content.is_finite() || content <= 0.0 {
        content = compile.duration.max(0.0);
    }
    if !content.is_finite() || content <= 0.0 {
        return 1;
    }
    (content * fps as f32).ceil().max(1.0) as u32
}

/// 把候选帧对齐到 ≥ 它的最近窗口命中帧。
fn align_to_window_hit(frame: u32, window: u32, frames: u32) -> Option<u32> {
    if frames == 0 || window == 0 {
        return None;
    }
    let w: u32 = window.max(1);
    let hit: u32 = (frame / w) * w + (w - 1);
    (hit < frames).then_some(hit)
}

/// 生成抽样计划。
///
/// 候选 = 每场景首/中/末 + 每过渡起点前后各 1 帧 + 周期首尾；全部对齐到
/// 窗口命中帧；不足 [`MIN_SAMPLE_FRAMES`] 时按全片均匀补足命中帧，且额外
/// 在每个循环周期重复覆盖一遍场景候选（保证「覆盖全部场景段」在 3600 帧
/// 的多个周期里都成立）。
pub fn build_sample_plan(
    compile: &CompileV2Output,
    frames: u32,
    window: u32,
) -> SamplePlan {
    let fps: u32 = compile.fps.max(1);
    let period: u32 = timeline_period_frames(compile).min(frames.max(1));
    let w: u32 = window.max(1);
    let mut requested: BTreeSet<u32> = BTreeSet::new();

    // 场景段首 / 中 / 末（先落在循环周期内的帧号）。
    for s in &compile.scenes {
        let first: u32 = (s.start.max(0.0) * fps as f32).floor() as u32;
        let last: u32 = ((s.start + s.duration).max(0.0) * fps as f32).ceil() as u32;
        let last: u32 = last.saturating_sub(1).min(period.saturating_sub(1));
        let mid: u32 = (first + last) / 2;
        for f in [first, mid, last] {
            requested.insert(f.min(period.saturating_sub(1)));
        }
    }
    // 过渡起点前后各 1 帧。
    for tr in &compile.transitions {
        let at: u32 = (tr.start.max(0.0) * fps as f32).floor() as u32;
        for d in [-1i64, 0, 1] {
            let f: i64 = at as i64 + d;
            if f >= 0 {
                requested.insert((f as u32).min(period.saturating_sub(1)));
            }
        }
    }
    requested.insert(0);
    requested.insert(period.saturating_sub(1));

    let mut hits: BTreeSet<u32> = BTreeSet::new();
    // 每个循环周期都覆盖一遍场景候选，保证全部场景段在长片里被反复校到。
    let mut cycle: u32 = 0;
    while cycle < frames {
        for &f in &requested {
            if let Some(h) = align_to_window_hit(cycle + f, w, frames) {
                hits.insert(h);
            }
            if hits.len() >= 16 {
                break;
            }
        }
        if hits.len() >= 16 || period == 0 {
            break;
        }
        cycle = cycle.saturating_add(period);
    }
    // 均匀补足到下限。
    if hits.len() < MIN_SAMPLE_FRAMES && frames > 0 {
        let need: usize = MIN_SAMPLE_FRAMES.saturating_sub(hits.len());
        for k in 0..need {
            let f: u32 = ((k as u64 + 1) * frames as u64 / (need as u64 + 1)) as u32;
            if let Some(h) = align_to_window_hit(f, w, frames) {
                hits.insert(h);
            }
        }
    }

    let mut indices: Vec<u32> = hits.into_iter().collect();
    indices.sort_unstable();
    let requested_vec: Vec<u32> = requested.into_iter().collect();
    SamplePlan {
        indices,
        requested: requested_vec,
        window: w,
        frames,
        period,
        scene_count: compile.scenes.len(),
        transition_count: compile.transitions.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_compile() -> CompileV2Output {
        const SRC: &str = concat!(
            "// prism-dsl 1\n",
            "movie \"s\" { duration 3s fps 12 resolution 64x36 }\n",
            "scene \"a\" { duration 1.5s stage3d { camera m { orbit { center 0,0,0 radius 7 } fov 45deg } } }\n",
            "scene \"b\" { duration 1.5s stage3d { camera n { orbit { center 0,0,0 radius 5 } fov 40deg } } }\n",
            "transition \"t\" { from \"a\" to \"b\" duration 0.5s kind crossfade }\n",
        );
        prism_dsl::compile_v2(SRC).expect("抽样计划测试 DSL 应编译成功")
    }

    #[test]
    fn plan_covers_at_least_twelve_window_hits() {
        let c = sample_compile();
        let plan = build_sample_plan(&c, 3600, 4);
        assert!(plan.indices.len() >= MIN_SAMPLE_FRAMES, "{:?}", plan.indices);
        assert!(plan.indices.iter().all(|&f| f < 3600));
        assert!(plan.indices.iter().all(|&f| f % 4 == 3), "必须全是窗口命中帧");
        assert!(plan.indices.windows(2).all(|p| p[0] < p[1]), "升序去重");
        assert_eq!(plan.scene_count, 2);
        assert_eq!(plan.transition_count, 1);
    }

    #[test]
    fn timeline_period_ignores_overridden_duration() {
        let mut c = sample_compile();
        c.fps = 60;
        c.duration = 60.0;
        // 场景经 0.5 s crossfade 重叠折算后内容时长 2.5 s（第二幕 start=1.0s），
        // 故周期 = ceil(2.5 × 60) = 150 帧，而不是被覆写的 3600 帧 / 3 s×60。
        assert_eq!(timeline_period_frames(&c), 150);
        assert!(timeline_period_frames(&c) < 3600);
    }

    #[test]
    fn manifest_roundtrip_is_idempotent() {
        let root = std::env::temp_dir().join(format!("v31-sample-{}", std::process::id()));
        let _ = fs::create_dir_all(&root);
        let store = GoldenStore::with_root(&root);
        let plan = SamplePlan {
            indices: vec![3, 7, 11, 15],
            requested: vec![0, 1, 2, 3],
            window: 4,
            frames: 16,
            period: 16,
            scene_count: 1,
            transition_count: 0,
        };
        // 伪造一份最小 manifest（含一个占位条目），再写两次抽样行。
        fs::write(store.manifest_path(), "aaaa  x.png\n").unwrap();
        plan.write_to_manifest(&store).unwrap();
        plan.write_to_manifest(&store).unwrap();
        assert_eq!(SamplePlan::read_from_manifest(&store), Some(vec![3, 7, 11, 15]));
        let text = fs::read_to_string(store.manifest_path()).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with(MANIFEST_TAG)).count(), 1);
        assert!(text.contains("aaaa  x.png"), "原有哈希条目不得被改写");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn manifest_line_is_parseable_and_deterministic() {
        let c = sample_compile();
        let a = build_sample_plan(&c, 360, 4);
        let b = build_sample_plan(&c, 360, 4);
        assert_eq!(a, b, "抽样计划必须确定性");
        assert!(a.manifest_line().contains("indices="));
        assert!(a.manifest_line().contains("count="));
    }
}
