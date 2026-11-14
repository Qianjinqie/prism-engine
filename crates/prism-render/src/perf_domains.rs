//! V3.1 收尾 C —— 全域分段计时计数器（零分配、进程级、可快照）。
//!
//! 覆盖 PRD「DSL / 场景翻译计时、文本 / 音频 / 纹理 / 粒子计时计数器、
//! 编码 IO 分段计时」：每个域一个 `AtomicU64`，`record` 只做一次 relaxed
//! fetch_add；`snapshot` 取全量，`delta_since` 求区间增量。
//!
//! 计数器位于 `prism-render`（依赖图里几乎所有出片 crate 都能单向下达它），
//! `prism-dsl` 因依赖方向不能反向引用本模块，故其编译 / 场景翻译计时在
//! `prism-dsl/src/wiring.rs` 内自带一份同口径计数器，由 `prism-app` 汇总。

use std::sync::atomic::{AtomicU64, Ordering};

/// 计时域（报告 / 计数器键 `v31.<domain>.<metric>` 的 `<domain>` 段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    /// DSL 编译（prism-dsl 侧另有权威计时，此处为 app 层端到端口径）。
    DslCompile,
    /// 场景翻译 / wiring 逐帧解算。
    Wiring,
    /// 文本图层。
    Text,
    /// 音频。
    Audio,
    /// 纹理。
    Texture,
    /// 粒子。
    Particles,
    /// 光栅 5 项基准。
    Raster,
    /// 3D 渲染（逐帧采样）。
    Render,
    /// 时空重建。
    Reconstruct,
    /// 误差校验（可选，默认不进热路径）。
    Verify,
    /// 容器编码 / IO。
    Encode,
    /// 帧循环总墙钟。
    FrameLoop,
}

impl Domain {
    /// 域总数（数组长度）。
    pub const COUNT: usize = 12;

    /// 稳定数组下标。
    pub const fn index(self) -> usize {
        match self {
            Domain::DslCompile => 0,
            Domain::Wiring => 1,
            Domain::Text => 2,
            Domain::Audio => 3,
            Domain::Texture => 4,
            Domain::Particles => 5,
            Domain::Raster => 6,
            Domain::Render => 7,
            Domain::Reconstruct => 8,
            Domain::Verify => 9,
            Domain::Encode => 10,
            Domain::FrameLoop => 11,
        }
    }

    /// 计数器键后缀（稳定）。
    pub const fn key(self) -> &'static str {
        match self {
            Domain::DslCompile => "dsl.compile",
            Domain::Wiring => "wiring.resolve",
            Domain::Text => "text",
            Domain::Audio => "audio",
            Domain::Texture => "texture",
            Domain::Particles => "particles",
            Domain::Raster => "raster",
            Domain::Render => "kernel.render",
            Domain::Reconstruct => "kernel.reconstruct",
            Domain::Verify => "verify",
            Domain::Encode => "io.encode",
            Domain::FrameLoop => "frame_loop",
        }
    }

    /// 全部域（固定序，报告用）。
    pub const ALL: [Domain; Domain::COUNT] = [
        Domain::DslCompile,
        Domain::Wiring,
        Domain::Text,
        Domain::Audio,
        Domain::Texture,
        Domain::Particles,
        Domain::Raster,
        Domain::Render,
        Domain::Reconstruct,
        Domain::Verify,
        Domain::Encode,
        Domain::FrameLoop,
    ];
}

static MICROS: [AtomicU64; Domain::COUNT] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
static CALLS: [AtomicU64; Domain::COUNT] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

const RELAXED: Ordering = Ordering::Relaxed;

/// 全量快照（微秒 / 调用次数，按 [`Domain::ALL`] 顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainSnapshot {
    pub micros: [u64; Domain::COUNT],
    pub calls: [u64; Domain::COUNT],
}

impl DomainSnapshot {
    /// 区间增量（右减左，`wrapping_sub` 防下溢）。
    pub fn delta_since(&self, base: &DomainSnapshot) -> DomainSnapshot {
        let mut micros = [0u64; Domain::COUNT];
        let mut calls = [0u64; Domain::COUNT];
        for i in 0..Domain::COUNT {
            micros[i] = self.micros[i].wrapping_sub(base.micros[i]);
            calls[i] = self.calls[i].wrapping_sub(base.calls[i]);
        }
        DomainSnapshot { micros, calls }
    }

    /// 逐域可读行（`v31.<domain>.micros=<n> calls=<m>`）。
    pub fn lines(&self) -> Vec<String> {
        Domain::ALL
            .iter()
            .map(|d| {
                let i = d.index();
                format!(
                    "v31.{}.micros={} calls={}",
                    d.key(), self.micros[i], self.calls[i]
                )
            })
            .collect()
    }
}

/// 取全量快照。
pub fn snapshot() -> DomainSnapshot {
    let mut micros = [0u64; Domain::COUNT];
    let mut calls = [0u64; Domain::COUNT];
    for i in 0..Domain::COUNT {
        micros[i] = MICROS[i].load(RELAXED);
        calls[i] = CALLS[i].load(RELAXED);
    }
    DomainSnapshot { micros, calls }
}

/// 记一次耗时（微秒）。
pub fn record(domain: Domain, micros: u64) {
    let i = domain.index();
    MICROS[i].fetch_add(micros, RELAXED);
    CALLS[i].fetch_add(1, RELAXED);
}

/// 清零全部计数（不影响任何数值路径）。
pub fn reset() {
    for i in 0..Domain::COUNT {
        MICROS[i].store(0, RELAXED);
        CALLS[i].store(0, RELAXED);
    }
}

/// RAII 分段计时器：显式 [`SegmentTimer::finish`] 或离开作用域自动 `record`，
/// 二者只记一次（`started: Option` 作「已结算」哨兵）。
pub struct SegmentTimer {
    domain: Domain,
    started: Option<std::time::Instant>,
}

impl SegmentTimer {
    /// 开始一个分段。
    pub fn start(domain: Domain) -> Self {
        SegmentTimer { domain, started: Some(std::time::Instant::now()) }
    }

    /// 不结算地读取当前耗时（微秒），供分段进度打印。
    pub fn elapsed_micros(&self) -> u64 {
        self.started
            .map(|s| s.elapsed().as_micros() as u64)
            .unwrap_or(0)
    }

    /// 立即结算并返回耗时（微秒）；随后 Drop 不再重复计数。
    pub fn finish(mut self) -> u64 {
        let micros = self.elapsed_micros();
        self.started = None;
        record(self.domain, micros);
        micros
    }
}

impl Drop for SegmentTimer {
    fn drop(&mut self) {
        if let Some(started) = self.started.take() {
            record(self.domain, started.elapsed().as_micros() as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_have_unique_indices_and_keys() {
        let mut seen = std::collections::BTreeSet::new();
        for d in Domain::ALL {
            assert!(d.index() < Domain::COUNT);
            assert!(seen.insert(d.key()), "重复键 {}", d.key());
        }
    }

    #[test]
    fn record_and_delta_are_consistent() {
        reset();
        let base = snapshot();
        record(Domain::Text, 12);
        record(Domain::Text, 8);
        record(Domain::Encode, 5);
        let cur = snapshot();
        let d = cur.delta_since(&base);
        assert_eq!(d.micros[Domain::Text.index()], 20);
        assert_eq!(d.calls[Domain::Text.index()], 2);
        assert_eq!(d.micros[Domain::Encode.index()], 5);
        assert_eq!(d.calls[Domain::Encode.index()], 1);
        assert_eq!(d.calls[Domain::Audio.index()], 0);
    }

    #[test]
    fn snapshot_lines_are_parseable() {
        let s = snapshot();
        assert_eq!(s.lines().len(), Domain::COUNT);
        assert!(s.lines()[Domain::Raster.index()].contains("v31.raster.micros="));
    }
}
