//! 编辑器预览的**异步刷新**机制。
//!
//! # 定位
//! Real（Final）档单帧耗时（几十毫秒量级）远超 Draft 几何图案（亚毫秒）。
//! 若在宿主交互线程里同步渲染 Final 档，编辑操作会被卡住。本模块把渲染
//! 搬到**后台工作线程**：
//!
//! 1. 宿主 [`AsyncPreview::submit`] 投递一次刷新请求后**立即返回**；
//! 2. 工作线程持有一份跨帧的 [`PreviewRenderer`]（"子图签名增量"的缓存
//!    记忆在此累积，与同步路径逐字节同结果）；
//! 3. 渲染完成后经通道回传，宿主 [`AsyncPreview::poll_latest`] 非阻塞取回
//!    **最新**一帧并**替换预览缓冲**。
//!
//! # latest-wins
//! 连续 submit 多次（如拖动滑条）时，中间结果可能乱序完成——本机制只认
//! **最大 seq**，过期结果直接丢弃，绝不回退到旧帧。
//!
//! # 为什么不动 ABI
//! 编辑器 C-ABI（`abi.rs`）已冻结（对外兼容性约束），且 C-ABI 本身是同步调用
//! 模型、无法表达"稍后回调"。故异步机制是**库级新增**：供宿主壳（编辑器
//! 页面 / CLI 预览循环）在 ABI 之外使用；ABI 的同步渲染路径保持不变
//! （Draft 档几何图案足够快，无需异步）。

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use prism_core::signature::ChannelDefs;
use prism_core::Graph;

use super::preview::{PreviewError, PreviewRenderer, Rendered};
use super::state::PreviewQuality;

/// 一次异步刷新请求（与 [`PreviewRenderer::render`] 参数同构；可跨线程发送）。
#[derive(Debug, Clone)]
pub struct PreviewRequest {
    /// 当前图（由编辑器模型现建）。
    pub graph: Graph,
    /// 通道定义表（进签名；空表即无通道绑定）。
    pub defs: ChannelDefs,
    /// 质量档位（Draft 同步路径已够快；Real 档是异步的目标场景）。
    pub quality: PreviewQuality,
    /// 输出宽（1..=[`super::preview::MAX_DIM`]）。
    pub width: u32,
    /// 输出高。
    pub height: u32,
    /// 引擎版本（进缓存键，C13）。
    pub engine_version: u64,
    /// 帧号（进缓存键，D-28）。
    pub frame: u64,
}

/// 一次异步刷新产物：提交序号 + 渲染结果（或结构化失败）。
#[derive(Debug, Clone)]
pub struct AsyncFrame {
    /// 提交序号（latest-wins 判据；越大越新）。
    pub seq: u64,
    /// 渲染结果（错误为 [`PreviewError`] 三段式，绝不 panic）。
    pub result: Result<Rendered, PreviewError>,
}

/// 后台工作线程消息（Job / Stop）。
enum WorkerMsg {
    /// 渲染任务。
    Job { seq: u64, req: PreviewRequest },
    /// 退出主循环。
    Stop,
}

/// 异步预览刷新器。
///
/// 典型用法（宿主交互线程）：
/// ```text
/// let mut preview = AsyncPreview::new();
/// preview.submit(req);                 // 立即返回，交互不卡
/// if let Some(frame) = preview.poll_latest() {
///     swap_preview_buffer(frame.result?); // 替换预览缓冲
/// }
/// ```
#[derive(Debug)]
pub struct AsyncPreview {
    /// 任务通道（宿主 → 工作线程）。
    tx: Sender<WorkerMsg>,
    /// 结果通道（工作线程 → 宿主）。
    rx: Receiver<AsyncFrame>,
    /// 下一个提交序号（从 1 起；0 保留作"未提交"哨兵）。
    next_seq: u64,
    /// 工作线程句柄（shutdown 时 join）。
    worker: Option<JoinHandle<()>>,
}

impl AsyncPreview {
    /// 新建异步刷新器并启动后台工作线程。
    pub fn new() -> Self {
        let (tx, rx): (Sender<WorkerMsg>, Receiver<WorkerMsg>) = mpsc::channel();
        let (res_tx, res_rx): (Sender<AsyncFrame>, Receiver<AsyncFrame>) = mpsc::channel();
        let worker: JoinHandle<()> = std::thread::spawn(move || worker_loop(rx, res_tx));
        Self { tx, rx: res_rx, next_seq: 1, worker: Some(worker) }
    }

    /// 提交一次刷新；返回提交序号（后续 [`poll_latest`](Self::poll_latest)
    /// 的 latest-wins 判据）。工作线程已退出 → `None`（绝不 panic）。
    ///
    /// 序号仅在**发送成功**后消费——失败的提交不得留下"已提交"假象。
    pub fn submit(&mut self, req: PreviewRequest) -> Option<u64> {
        let seq: u64 = self.next_seq;
        match self.tx.send(WorkerMsg::Job { seq, req }) {
            Ok(()) => {
                self.next_seq = self.next_seq.wrapping_add(1).max(1);
                Some(seq)
            }
            Err(_) => None,
        }
    }

    /// 非阻塞取回**最新**完成的一帧：抽干结果通道，只保留最大 seq
    /// （过期结果直接丢弃——回退到旧帧比不返回更糟）。
    pub fn poll_latest(&mut self) -> Option<AsyncFrame> {
        let mut newest: Option<AsyncFrame> = None;
        while let Ok(frame) = self.rx.try_recv() {
            if newest.as_ref().is_none_or(|n| frame.seq >= n.seq) {
                newest = Some(frame);
            }
        }
        newest
    }

    /// 已提交的最大序号（0 = 尚未提交任何任务）。
    pub fn latest_submitted_seq(&self) -> u64 {
        self.next_seq - 1
    }

    /// 关闭工作线程并 join（幂等；[`Drop`] 自动调用）。
    pub fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerMsg::Stop);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Default for AsyncPreview {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncPreview {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// 工作线程主循环：持有跨帧的 [`PreviewRenderer`]（增量缓存记忆在此累积），
/// 逐任务渲染并回传。宿主消失（结果通道断开）即退出。
fn worker_loop(rx: Receiver<WorkerMsg>, tx: Sender<AsyncFrame>) {
    let mut renderer: PreviewRenderer = PreviewRenderer::new();
    while let Ok(msg) = rx.recv() {
        match msg {
            WorkerMsg::Job { seq, req } => {
                let result = renderer.render(
                    &req.graph,
                    &req.defs,
                    req.quality,
                    req.width,
                    req.height,
                    req.engine_version,
                    req.frame,
                );
                if tx.send(AsyncFrame { seq, result }).is_err() {
                    break; // 宿主已消失，无需再渲
                }
            }
            WorkerMsg::Stop => break,
        }
    }
}

// ---------------------------------------------------------------------------
// 单元测试（后台线程 + latest-wins + 确定性穿线程）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::state::build_reference_graph;
    use prism_core::node::ParamValue;
    use std::time::{Duration, Instant};

    /// 参考图的刷新请求（档位 / 帧号可调）。
    fn req(quality: PreviewQuality, frame: u64) -> PreviewRequest {
        PreviewRequest {
            graph: build_reference_graph(),
            defs: ChannelDefs::new(),
            quality,
            width: 64,
            height: 48,
            engine_version: 1,
            frame,
        }
    }

    /// 有界等待最新一帧（后台完成通知；超时返回 None）。
    fn wait_latest(preview: &mut AsyncPreview, timeout: Duration) -> Option<AsyncFrame> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(frame) = preview.poll_latest() {
                return Some(frame);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn async_render_matches_sync_pixels() {
        // 穿线程结果与同步渲染逐字节一致（同渲染器逻辑、同缓存语义）。
        let mut sync = PreviewRenderer::new();
        let want = sync
            .render(
                &build_reference_graph(),
                &ChannelDefs::new(),
                PreviewQuality::Draft,
                64,
                48,
                1,
                0,
            )
            .expect("同步渲染应成功");

        let mut async_preview = AsyncPreview::new();
        assert_eq!(async_preview.submit(req(PreviewQuality::Draft, 0)), Some(1));
        let frame = wait_latest(&mut async_preview, Duration::from_secs(10))
            .expect("后台结果应到达");
        assert_eq!(frame.seq, 1);
        let got = frame.result.expect("后台渲染应成功");
        assert_eq!(got.rgba, want.rgba, "异步结果必须与同步逐字节一致");
        assert_eq!(got.stats, want.stats, "增量统计也必须一致");
    }

    #[test]
    fn real_quality_renders_through_async_worker() {
        // Real（Final）档经后台线程出帧：采样数 = `PreviewQuality::Final.spp()`。
        let mut async_preview = AsyncPreview::new();
        assert_eq!(async_preview.submit(req(PreviewQuality::Final, 0)), Some(1));
        let frame = wait_latest(&mut async_preview, Duration::from_secs(30))
            .expect("Real 档后台结果应到达");
        let got = frame.result.expect("Real 档后台渲染应成功");
        assert_eq!(got.stats.spp, crate::editor::PreviewQuality::Final.spp());
        assert_eq!(got.rgba.len(), 64 * 48 * 4);
    }

    #[test]
    fn submit_does_not_block_on_slow_job() {
        // 非阻塞证据：投递 512x512 Real 档（几十毫秒级）后立即 poll 应无结果；
        // 随后有界等待应能收到（证明任务确实在后台执行而非被丢弃）。
        let mut async_preview = AsyncPreview::new();
        let big = PreviewRequest {
            graph: build_reference_graph(),
            defs: ChannelDefs::new(),
            quality: PreviewQuality::Final,
            width: 512,
            height: 512,
            engine_version: 1,
            frame: 0,
        };
        assert_eq!(async_preview.submit(big), Some(1));
        assert!(
            async_preview.poll_latest().is_none(),
            "submit 后立即 poll 不得已有结果（否则说明渲染发生在调用线程里）"
        );
        let frame = wait_latest(&mut async_preview, Duration::from_secs(30))
            .expect("512x512 Real 档应在超时内完成");
        let got = frame.result.expect("后台渲染应成功");
        assert_eq!(got.rgba.len(), 512 * 512 * 4);
    }

    #[test]
    fn latest_wins_discards_stale_results() {
        // 连投两帧（帧 0 / 帧 1）：poll_latest 只认最新 seq，旧帧结果被丢弃。
        let mut async_preview = AsyncPreview::new();
        assert_eq!(async_preview.submit(req(PreviewQuality::Draft, 0)), Some(1));
        assert_eq!(async_preview.submit(req(PreviewQuality::Draft, 1)), Some(2));

        // 两帧都完成后：最新一帧的 seq 必须是 2，且像素等于帧 1 的同步渲染。
        let mut want1 = PreviewRenderer::new();
        let _ = want1
            .render(&build_reference_graph(), &ChannelDefs::new(), PreviewQuality::Draft, 64, 48, 1, 0)
            .expect("帧 0 同步渲染应成功");
        let frame1 = want1
            .render(&build_reference_graph(), &ChannelDefs::new(), PreviewQuality::Draft, 64, 48, 1, 1)
            .expect("帧 1 同步渲染应成功");

        // 等到 seq 2 出现为止（期间可能先收到 seq 1，应被覆盖）。
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut newest: Option<AsyncFrame> = None;
        while Instant::now() < deadline {
            if let Some(f) = async_preview.poll_latest() {
                newest = Some(f);
                if newest.as_ref().map_or(false, |n| n.seq == 2) {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let newest = newest.expect("应收到后台结果");
        assert_eq!(newest.seq, 2, "latest-wins：只保留最大 seq");
        assert_eq!(
            newest.result.expect("后台渲染应成功").rgba,
            frame1.rgba,
            "最新一帧必须是最后一次提交（帧 1）的渲染"
        );
    }

    #[test]
    fn worker_cache_persists_across_submits() {
        // 缓存活在工作线程：同参两次提交，第二次应零重算（增量穿线程保持）。
        let mut async_preview = AsyncPreview::new();
        assert_eq!(async_preview.submit(req(PreviewQuality::Final, 0)), Some(1));
        let first = wait_latest(&mut async_preview, Duration::from_secs(30))
            .expect("首次结果应到达")
            .result
            .expect("首次渲染应成功");
        assert!(!first.stats.recomputed.is_empty(), "首帧必须重算");

        assert_eq!(async_preview.submit(req(PreviewQuality::Final, 0)), Some(2));
        let second = wait_latest(&mut async_preview, Duration::from_secs(30))
            .expect("二次结果应到达")
            .result
            .expect("二次渲染应成功");
        assert!(
            second.stats.recomputed.is_empty(),
            "工作线程缓存应命中：零重算；实际 {:?}",
            second.stats.recomputed
        );
        assert_eq!(second.stats.reused.len(), 4, "参考图 4 节点全部复用");
    }

    #[test]
    fn error_surfaces_as_structured_preview_error() {
        // 病态请求（0 宽）不得 panic：错误经通道结构化回传。
        let mut async_preview = AsyncPreview::new();
        let bad = PreviewRequest {
            graph: build_reference_graph(),
            defs: ChannelDefs::new(),
            quality: PreviewQuality::Draft,
            width: 0,
            height: 48,
            engine_version: 1,
            frame: 0,
        };
        assert_eq!(async_preview.submit(bad), Some(1));
        let frame = wait_latest(&mut async_preview, Duration::from_secs(10))
            .expect("错误结果也应回传");
        assert!(frame.result.is_err(), "0 宽应结构化报错");
        assert!(frame.result.unwrap_err().to_string().contains("建议："));
    }

    #[test]
    fn shutdown_is_idempotent_and_rejects_further_submits() {
        let mut async_preview = AsyncPreview::new();
        assert!(async_preview.submit(req(PreviewQuality::Draft, 0)).is_some());
        async_preview.shutdown();
        async_preview.shutdown(); // 幂等：不得 panic / 二次 join
        assert_eq!(
            async_preview.submit(req(PreviewQuality::Draft, 1)),
            None,
            "关闭后提交应失败而非 panic"
        );
        assert_eq!(async_preview.latest_submitted_seq(), 1);
    }

    #[test]
    fn param_edit_flows_through_async_boundary() {
        // 编辑 → 提交 → 后台重算增量集合仍精确（{1,3,4}，兄弟叶 2 复用）。
        use crate::editor::state::EditorState;
        let mut state = EditorState::with_reference_graph();
        let mut async_preview = AsyncPreview::new();
        let g0 = state.model().to_graph().unwrap();
        assert_eq!(
            async_preview.submit(PreviewRequest {
                graph: g0,
                defs: ChannelDefs::new(),
                quality: PreviewQuality::Final,
                width: 32,
                height: 32,
                engine_version: 1,
                frame: 0,
            }),
            Some(1)
        );
        let _ = wait_latest(&mut async_preview, Duration::from_secs(30)).expect("首帧应到达");

        let dirty = state
            .apply_edit(1, "scale", ParamValue::F32(0.9))
            .expect("编辑应成功");
        assert_eq!(dirty.to_vec(), vec![1, 3, 4]);
        let g1 = state.model().to_graph().unwrap();
        assert_eq!(
            async_preview.submit(PreviewRequest {
                graph: g1,
                defs: ChannelDefs::new(),
                quality: PreviewQuality::Final,
                width: 32,
                height: 32,
                engine_version: 1,
                frame: 0,
            }),
            Some(2)
        );
        let second = wait_latest(&mut async_preview, Duration::from_secs(30))
            .expect("二次结果应到达")
            .result
            .expect("二次渲染应成功");
        assert_eq!(second.stats.recomputed, vec![1, 3, 4], "穿线程增量集合必须精确");
        assert_eq!(second.stats.reused, vec![2], "兄弟叶 2 必须命中缓存");
    }
}
