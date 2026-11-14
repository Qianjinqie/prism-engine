//! 帧池：复用 [`Framebuffer`]，命中率可数（T09 / V2.2-D-2）。
//!
//! 池按 (宽, 高) 维度缓存已分配的 [`Framebuffer`]；`acquire` 时命中缓存则复用
//! （不新分配），未命中则新建。`release` 把缓冲归还池。命中计数用于判据 ① 的
//! 确定性比较与判据 ② 的「池中性」（复用 vs 非复用输出逐字节一致）。
//!
//! 设计 §6.2 / Line 516：本模块**不**注册 `#[global_allocator]`，生产零改动；
//! 计数分配器只在 `tests/qa_perf_counters.rs` 安装。

use prism_core::Scene;
use prism_gpu::{Framebuffer, GpuDevice};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static POOL_HITS: AtomicU64 = AtomicU64::new(0);
static POOL_MISSES: AtomicU64 = AtomicU64::new(0);

const RELAXED: Ordering = Ordering::Relaxed;

#[derive(Debug, Default)]
struct PoolInner {
    /// 每个 (w, h) 一个 LIFO 空闲栈（复用缓冲）。
    free: HashMap<(u32, u32), Vec<Framebuffer>>,
}

thread_local! {
    static POOL: RefCell<PoolInner> = RefCell::new(PoolInner::default());
}

/// 帧池命中/未命中统计快照。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// 命中次数（复用既有缓冲，零新分配）。
    pub hits: u64,
    /// 未命中次数（新建缓冲）。
    pub misses: u64,
}

impl PoolStats {
    /// 缓冲复用率（命中 / 总取用；无取用时为 0）。
    pub fn reuse_ratio(&self) -> f64 {
        let total: u64 = self.hits.saturating_add(self.misses);
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// 供脚本采集的确定性行（帧池贯穿出片链路的报表口径）。
    pub fn summary_line(&self) -> String {
        format!(
            "[v31-pool] 命中 {} / 未命中 {} / 复用率 {:.4}（每帧大分配随命中下降）",
            self.hits,
            self.misses,
            self.reuse_ratio()
        )
    }
}

/// 取当前池统计快照。
pub fn pool_stats() -> PoolStats {
    PoolStats {
        hits: POOL_HITS.load(RELAXED),
        misses: POOL_MISSES.load(RELAXED),
    }
}

/// 清零池统计（不影响池中缓冲）。
pub fn reset_pool_stats() {
    POOL_HITS.store(0, RELAXED);
    POOL_MISSES.store(0, RELAXED);
}

/// 帧池：复用 [`Framebuffer`]。
///
/// 进程级单例（线程局部）；当前场景单线程，无需跨线程锁。
pub struct FramePool;

impl FramePool {
    /// 取一个 (w, h) 帧缓冲：命中复用，未命中新建。
    ///
    /// 命中时累加 [`POOL_HITS`]，未命中累加 [`POOL_MISSES`]。
    pub fn acquire(w: u32, h: u32) -> Framebuffer {
        let reused: Option<Framebuffer> = POOL.with(|p| {
            let mut g = p.borrow_mut();
            g.free.get_mut(&(w, h)).and_then(|stack| stack.pop())
        });
        match reused {
            Some(fb) => {
                POOL_HITS.fetch_add(1, RELAXED);
                fb
            }
            None => {
                POOL_MISSES.fetch_add(1, RELAXED);
                Framebuffer::new(w, h)
            }
        }
    }

    /// 归还帧缓冲到池（按维度入栈）。无效缓冲（被拒尺寸）不入池。
    pub fn release(fb: Framebuffer) {
        if !fb.is_valid() {
            return;
        }
        POOL.with(|p| {
            p.borrow_mut()
                .free
                .entry((fb.width, fb.height))
                .or_default()
                .push(fb);
        });
    }

    /// 清空池（释放全部缓存缓冲）。测试隔离用。
    pub fn clear() {
        POOL.with(|p| p.borrow_mut().free.clear());
    }
}

/// 从池中取一帧并把 RGBA8 像素**就地写入**（出片主链组装点）。
///
/// 与 `Framebuffer { pixels: Vec::with_capacity(..) + extend }` 逐字节一致，
/// 但命中池时复用既有 `Vec<u8>` 的容量 ⇒ 该帧零次大分配。调用方在帧交付后
/// 用 [`FramePool::release`] 归还，即可让下一帧命中。
pub fn pack_rgba8(rgba: &[[u8; 4]], w: u32, h: u32) -> Framebuffer {
    let mut fb: Framebuffer = FramePool::acquire(w, h);
    let need: usize = rgba.len() * 4;
    fb.pixels.clear();
    if fb.pixels.capacity() < need {
        fb.pixels.reserve(need - fb.pixels.capacity());
    }
    for p in rgba {
        fb.pixels.extend_from_slice(p);
    }
    fb.width = w;
    fb.height = h;
    fb
}

/// 池化单帧渲染（判据 ② 中性路径）：与非池化 [`crate::render_frame`] 输出
/// **逐字节一致**。池只负责复用缓冲，不修改任何像素。
///
/// 实现：先用既有 [`crate::render_frame`] 渲出 `rendered`，再经显式
/// [`crate::measure::copy_bytes`] 把像素逐字节迁入池化缓冲。本路径是「拷贝」路径
/// （明确计拷贝），**非**零拷贝——零拷贝判据见 [`crate::zero_copy`]。
pub fn render_frame_pooled(
    device: &mut dyn GpuDevice,
    scene: &Scene,
    t: f32,
    duration: f32,
    w: u32,
    h: u32,
) -> Framebuffer {
    let rendered: Framebuffer = crate::render_frame(device, scene, t, duration, w, h);
    let mut pooled: Framebuffer = FramePool::acquire(w, h);
    crate::measure::copy_bytes(&mut pooled.pixels, &rendered.pixels);
    pooled
}
