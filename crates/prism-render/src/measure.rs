//! 确定性可数分配/拷贝计数器（T09 / V2.2-D-2）。
//!
//! 自研 [`CountingAllocator`]（`GlobalAlloc` 包装），**零 crate**，只统计
//! 分配/释放的次数与字节，不改写任何分配语义。拷贝计数由 [`copy_bytes`]
//! 显式统计——因为 `GlobalAlloc` 看不见 `memcpy`，凡是真正发生拷贝的路径
//! 必须走 [`copy_bytes`] 才能被度量捕获；反之声明为零拷贝的路径**不得**走
//! 它（见 [`crate::zero_copy`]）。
//!
//! 安装方式（设计 §6.2 / Line 516）：`#[global_allocator]` **只在专用测试二进制**
//! 安装，生产零改动。本模块只提供类型与计数原语；注册在
//! `tests/qa_perf_counters.rs` 完成。

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// 进程级计数（仅本模块写入，通过 pub 函数读出）
// ---------------------------------------------------------------------------

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static COPY_COUNT: AtomicU64 = AtomicU64::new(0);
static COPY_BYTES: AtomicU64 = AtomicU64::new(0);

const RELAXED: Ordering = Ordering::Relaxed;

/// 计数快照：某时刻的累计值。
///
/// 所有字段为累计计数；用 [`Counters::delta_since`] 求相邻两次快照之差（区间增量）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// `alloc` 调用次数（含 `alloc_zeroed` / `realloc` 的新分配）。
    pub alloc_count: u64,
    /// `alloc` 累计字节数。
    pub alloc_bytes: u64,
    /// `dealloc` 调用次数。
    pub dealloc_count: u64,
    /// `dealloc` 累计字节数。
    pub dealloc_bytes: u64,
    /// 显式 [`copy_bytes`] 调用次数（memcpy）。
    pub copy_count: u64,
    /// 显式 [`copy_bytes`] 累计字节数。
    pub copy_bytes: u64,
}

impl Counters {
    /// 右减左的增量（区间计数；用 `wrapping_sub` 防下溢）。
    pub fn delta_since(&self, base: &Counters) -> Counters {
        Counters {
            alloc_count: self.alloc_count.wrapping_sub(base.alloc_count),
            alloc_bytes: self.alloc_bytes.wrapping_sub(base.alloc_bytes),
            dealloc_count: self.dealloc_count.wrapping_sub(base.dealloc_count),
            dealloc_bytes: self.dealloc_bytes.wrapping_sub(base.dealloc_bytes),
            copy_count: self.copy_count.wrapping_sub(base.copy_count),
            copy_bytes: self.copy_bytes.wrapping_sub(base.copy_bytes),
        }
    }
}

/// 取当前累计计数快照。
pub fn snapshot() -> Counters {
    Counters {
        alloc_count: ALLOC_COUNT.load(RELAXED),
        alloc_bytes: ALLOC_BYTES.load(RELAXED),
        dealloc_count: DEALLOC_COUNT.load(RELAXED),
        dealloc_bytes: DEALLOC_BYTES.load(RELAXED),
        copy_count: COPY_COUNT.load(RELAXED),
        copy_bytes: COPY_BYTES.load(RELAXED),
    }
}

/// 清零所有累计计数（不影响分配器语义，只清计数）。
///
/// 用于「两次运行」测量前重置基线；更常见用法是「基线快照 + 区间 `delta_since`」。
pub fn reset_counters() {
    ALLOC_COUNT.store(0, RELAXED);
    ALLOC_BYTES.store(0, RELAXED);
    DEALLOC_COUNT.store(0, RELAXED);
    DEALLOC_BYTES.store(0, RELAXED);
    COPY_COUNT.store(0, RELAXED);
    COPY_BYTES.store(0, RELAXED);
}

/// 显式统计的字节拷贝：把 `src` 拷到 `dst`，并累加拷贝次数与字节。
///
/// **度量核心**：`GlobalAlloc` 看不见 `memcpy`，故真正发生拷贝的路径必须走本函数，
/// 才能被 [`Counters::copy_count`] 捕获；声明为零拷贝的路径（[`crate::zero_copy`]）
/// 不得调用本函数。
///
/// 长度取两者较小者，越界部分不拷贝、不 panic（与 `ptr::copy` 语义一致）。
pub fn copy_bytes(dst: &mut [u8], src: &[u8]) {
    let n: usize = src.len().min(dst.len());
    if n == 0 {
        return;
    }
    // SAFETY: `n <= src.len()` 且 `n <= dst.len()`；源/目的不重叠由调用方保证
    // （本函数用于「渲染结果迁入池化缓冲」等明确不重叠场景）。
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), n);
    }
    COPY_COUNT.fetch_add(1, RELAXED);
    COPY_BYTES.fetch_add(n as u64, RELAXED);
}

// ---------------------------------------------------------------------------
// 自研 GlobalAlloc 包装
// ---------------------------------------------------------------------------

/// 计数分配器：薄包装任意 [`GlobalAlloc`]，只计数、不改写分配语义。
///
/// 设计为「零 crate」自研实现，不引入任何第三方分配器。计数用 `Relaxed` 序——
/// 本模块只关心总数，不依赖计数间的跨线程顺序；单线程固定场景两次运行必然
/// 得到相同分配序列（确定性由场景本身保证）。
///
/// 安装为 `#[global_allocator]` 只在专用测试二进制进行（见 `tests/qa_perf_counters.rs`），
/// 生产构建零改动。
pub struct CountingAllocator<A: GlobalAlloc> {
    inner: A,
}

impl<A: GlobalAlloc> CountingAllocator<A> {
    /// 包装一个底层分配器。
    pub const fn new(inner: A) -> Self {
        CountingAllocator { inner }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for CountingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr: *mut u8 = self.inner.alloc(layout);
        if !ptr.is_null() {
            ALLOC_COUNT.fetch_add(1, RELAXED);
            ALLOC_BYTES.fetch_add(layout.size() as u64, RELAXED);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !ptr.is_null() {
            DEALLOC_COUNT.fetch_add(1, RELAXED);
            DEALLOC_BYTES.fetch_add(layout.size() as u64, RELAXED);
        }
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr: *mut u8 = self.inner.alloc_zeroed(layout);
        if !ptr.is_null() {
            ALLOC_COUNT.fetch_add(1, RELAXED);
            ALLOC_BYTES.fetch_add(layout.size() as u64, RELAXED);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr: *mut u8 = self.inner.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // 近似记一次新分配；若旧指针非空，记一次释放（容量变化）。
            ALLOC_COUNT.fetch_add(1, RELAXED);
            ALLOC_BYTES.fetch_add(new_size as u64, RELAXED);
            if !ptr.is_null() {
                DEALLOC_COUNT.fetch_add(1, RELAXED);
                DEALLOC_BYTES.fetch_add(layout.size() as u64, RELAXED);
            }
        }
        new_ptr
    }
}

// ---------------------------------------------------------------------------
// 峰值 RSS 报表（**非门禁**：环境相关，绝对值只作报表）
// ---------------------------------------------------------------------------

/// 峰值 RSS（进程高水位）报表。
///
/// **非门禁**：绝对值随环境变化（沙箱无稳定基线），只产出报表，不作 Go/No-Go。
/// 判据 ④ 只要求「报表可产出」。
#[derive(Debug, Clone)]
pub struct RssReport {
    /// 场景标识（调用方声明，用于报表可读性）。
    pub scenario: String,
    /// 场景帧宽（像素）。
    pub width: u32,
    /// 场景帧高（像素）。
    pub height: u32,
    /// 峰值常驻内存（KiB）；非 Linux 平台为 `None`。
    pub peak_rss_kib: Option<u64>,
}

/// 读进程峰值常驻内存（VmHWM，KiB）。
///
/// Linux 解析 `/proc/self/status` 的 `VmHWM:` 行；其他平台返回 `None`。
/// 该值由内核维护为「进程生命周期峰值」，故无需采样即可得峰值。
pub fn peak_rss_kib() -> Option<u64> {
    let status: String = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

/// 产出固定场景的峰值 RSS 报表（判据 ④）。
///
/// 仅构造报表结构，不判绝对值；`peak_rss_kib` 可能为 `None`（非 Linux）。
pub fn rss_report(scenario: &str, width: u32, height: u32) -> RssReport {
    RssReport {
        scenario: scenario.to_string(),
        width,
        height,
        peak_rss_kib: peak_rss_kib(),
    }
}

// ---------------------------------------------------------------------------
// V3.1 收尾 C：RSS 首尾差 + 每帧分配次数（3600 帧内存报表）
// ---------------------------------------------------------------------------

/// 读进程**当前**常驻内存（`VmRSS`，KiB）。
///
/// 与 [`peak_rss_kib`]（生命周期高水位 `VmHWM`）不同：首尾差必须用当前值，
/// 否则「跑完 3600 帧后 RSS 是否回到起点」这个问题根本问不出来。
/// Linux 解析 `/proc/self/status`；其他平台返回 `None`。
pub fn current_rss_kib() -> Option<u64> {
    let status: String = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

/// 3600 帧 RSS 首尾差报表（**非门禁**：绝对值随环境变化，只作报表）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RssDeltaReport {
    pub scenario: String,
    pub frames: u64,
    pub first_kib: Option<u64>,
    pub last_kib: Option<u64>,
    /// `last − first`（KiB）；任一读取失败为 `None`。
    pub delta_kib: Option<i64>,
    /// 每帧平均分配次数（由 [`Counters::delta_since`] 给出）。
    pub alloc_per_frame: u64,
    /// 每帧平均分配字节。
    pub alloc_bytes_per_frame: u64,
}

impl RssDeltaReport {
    /// 供脚本采集的确定性行。
    pub fn summary_line(&self) -> String {
        let (f, l, d) = match (self.first_kib, self.last_kib, self.delta_kib) {
            (Some(f), Some(l), Some(d)) => (f.to_string(), l.to_string(), format!("{d}")),
            _ => ("n/a".to_string(), "n/a".to_string(), "n/a".to_string()),
        };
        format!(
            "[v31-mem] 场景={} 帧数={} RSS 首={} KiB 尾={} KiB 差={} KiB / 每帧分配 {} 次 {} B",
            self.scenario,
            self.frames,
            f,
            l,
            d,
            self.alloc_per_frame,
            self.alloc_bytes_per_frame
        )
    }
}

/// 组装 RSS 首尾差报表（纯函数，输入是两个已采集的快照）。
pub fn rss_delta_report(
    scenario: &str,
    frames: u64,
    first_kib: Option<u64>,
    last_kib: Option<u64>,
    before: Counters,
    after: Counters,
) -> RssDeltaReport {
    let d: Counters = after.delta_since(&before);
    let denom: u64 = frames.max(1);
    RssDeltaReport {
        scenario: scenario.to_string(),
        frames,
        first_kib,
        last_kib,
        delta_kib: match (first_kib, last_kib) {
            (Some(a), Some(b)) => Some(b as i64 - a as i64),
            _ => None,
        },
        alloc_per_frame: d.alloc_count / denom,
        alloc_bytes_per_frame: d.alloc_bytes / denom,
    }
}

/// 每帧分配次数（帧池验证用：同一场景池化路径 vs 非池化路径）。
pub fn alloc_per_frame(before: Counters, after: Counters, frames: u64) -> u64 {
    after.delta_since(&before).alloc_count / frames.max(1)
}
