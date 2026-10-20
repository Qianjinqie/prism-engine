//! 影片级农场分发（PE-1 分布式渲染）：**真实渲染帧**走帧级任务分发 +
//! N 个子进程真跑 + 确定性合并。
//!
//! # 与 `worker.rs` 32 字节替身的分工（诚实边界）
//! - `worker.rs` 的 [`render_frame_payload`](crate::farm::worker::render_frame_payload)
//!   是 splitmix64 合成的**成本替身**：验的是调度 / 传输 / 断点续算的确定性
//!   （载荷廉价，可跑大帧数）；
//! - 本模块验的是**真实渲染产物**的并行一致性：载荷 = 帧像素（RGBA 字节）。
//!   "N worker 合并 MD5 == 单机逐字节一致"只有在真实像素上成立，才配叫
//!   "分布式渲染"——替身验不出渲染器在跨进程路径下的字节漂移。
//!
//! # 帧级并行为什么免费（复述农场的核心纪律）
//! 切分（[`split_frames`]）是 `(start, end, workers)` 的纯函数；渲染是
//! `(源码, 帧号)` 的纯函数（DSL 编译无全局态、`CpuRefBackend` 无 RNG）；
//! 合并按**帧号**归位 + 唯一 [`manifest_digest`](crate::farm::evidence::manifest_digest)
//! 形状 ⇒ worker 数、完成顺序、哪个进程算哪块都不进结果 ⇒ "并行 == 串行"
//! 是定理而非巧合。
//!
//! # 源码怎么到 worker 进程（环境变量契约）
//! 线格式（[`FrameTask`]）只载"帧区间 + 配置哈希"，不载源码——协议保持与
//! 替身路径同一份。源码走环境变量 [`MOVIE_SRC_ENV`]（主节点分发时由调度器在
//! **进程级分发互斥锁内**注入，worker 启动时读一次编译一次，范式同既有的
//! `PRISM_FARM_COST`）。worker 渲染前校验任务 `config_hash` 与自己编译产物的
//! 配置哈希一致，不一致 ⇒ 结构化错误（绝不渲染"手头恰好有的"另一部影片——C20）。
//!
//! # 环境受限声明（方法论降级，如实）
//! 单机无多机环境：N 进程 = 单机多进程真跑（协议 / 编解码 / 合并与跨机
//! 同一份代码，仅介质从 socket 换管道——[`TcpTransport`] 已留接口、当前版本未实装，
//! 见 `FarmError::TransportUnavailable`）。加速比实测反映单机多进程视角，
//! **不代表**跨机集群数字。

use crate::farm::evidence::manifest_digest;
use crate::farm::scheduler::{
    dispatch_to_local_workers_movie, dispatch_to_local_workers_movie_cached, DispatchReport,
};
use crate::farm::split::split_frames;
use crate::farm::task::{ConfigHash, FrameTask};
use crate::farm::FarmError;
use prism_core::Scene;
use prism_gpu::{CpuRefBackend, Framebuffer};
use std::path::Path;

/// 影片源码环境变量（主节点 → worker 子进程的源码通道）。
///
/// 与 `PRISM_FARM_COST` 同范式：主节点 spawn 时不便拼参数（子进程参数表固定），
/// 走环境继承。**只在分发临界区内置放、用完即删**——由
/// [`dispatch_to_local_workers_movie`](crate::farm::scheduler::dispatch_to_local_workers_movie)
/// 在进程级分发互斥锁内完成，并发分发之间不会互相置错对方的源码。
pub const MOVIE_SRC_ENV: &str = "PRISM_FARM_MOVIE_SRC";

/// 影片作业 / 渲染器错误（三段式：在哪 / 什么事 / 建议；P5011）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MovieError {
    /// 面向人类的完整消息（含"建议："）。
    pub message: String,
}

impl std::fmt::Display for MovieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for MovieError {}

fn movie_err(message: impl Into<String>) -> MovieError {
    MovieError { message: message.into() }
}

/// 影片渲染器（每进程一份）：编译一次 DSL，逐帧渲染真实像素。
///
/// 主节点侧与 worker 子进程侧共用同一类型——"渲染一帧"的代码只有一份，
/// 并行与串行不可能因此漂移。
#[derive(Debug, Clone)]
pub struct MovieRenderer {
    /// DSL 源码（保留一份：node_id 之外的排查线索 / 错误上下文）。
    source: String,
    /// 配置哈希 = `ConfigHash::of_bytes(source)`（同源码 ⇒ 同作业）。
    config_hash: ConfigHash,
    /// 编译产物场景（一次编译，逐帧复用）。
    scene: Scene,
    /// 分辨率宽（DSL 声明；本农场不提供覆盖参数——见模块文档"口径"节）。
    width: u32,
    /// 分辨率高。
    height: u32,
    /// 帧率（DSL 声明）。
    fps: u32,
    /// 时长（秒，DSL 声明）。
    duration: f32,
}

impl MovieRenderer {
    /// 从 DSL 源码构建（编译一次）。
    ///
    /// # Errors
    /// 编译失败 ⇒ P5011（原样携带 DSL 三段式消息）；`fps == 0` 或
    /// `duration <= 0` ⇒ P5011（无帧可渲，提前显式拒绝而非产出空作业）。
    pub fn from_source(source: &str) -> Result<MovieRenderer, MovieError> {
        let compiled: prism_dsl::CompileOutput =
            prism_dsl::compile(source).map_err(|e| {
                movie_err(format!(
                    "error[P5011] farm::movie::MovieRenderer::from_source\n  \
                     什么事：影片 DSL 编译失败（第 {} 行）——{}\n  \
                     建议：修正源码后重试；渲染是（源码, 帧号）的纯函数，修好即可整段重跑，\
                     断点续算缓存按配置哈希分键，坏源码不会污染好作业",
                    e.line, e.message
                ))
            })?;
        if compiled.fps == 0 || !(compiled.duration > 0.0) {
            return Err(movie_err(format!(
                "error[P5011] farm::movie::MovieRenderer::from_source\n  \
                 什么事：DSL 声明 fps={} duration={}——按公式 fps×duration 产出 0 帧，无帧可渲\n  \
                 建议：把 timeline 的 fps / duration 设为正值；空作业没有分发意义",
                compiled.fps, compiled.duration
            )));
        }
        if compiled.width == 0 || compiled.height == 0 {
            return Err(movie_err(format!(
                "error[P5011] farm::movie::MovieRenderer::from_source\n  \
                 什么事：DSL 声明分辨率 {}x{}——含 0 维，渲染无处落笔\n  \
                 建议：output 块写正分辨率（如 resolution 96x64）",
                compiled.width, compiled.height
            )));
        }
        Ok(MovieRenderer {
            source: source.to_string(),
            config_hash: ConfigHash::of_bytes(source.as_bytes()),
            scene: compiled.scene,
            width: compiled.width,
            height: compiled.height,
            fps: compiled.fps,
            duration: compiled.duration,
        })
    }

    /// 从环境变量 [`MOVIE_SRC_ENV`] 构建（worker 启动路径）。
    ///
    /// - 环境变量未设置 ⇒ `Ok(None)`（替身模式，既有行为不变）；
    /// - 设置了但编译失败 ⇒ `Err`（**不静默退回替身**——那会让主节点拿到
    ///   一份"看起来成功"的错产物，是 C20 明令禁止的）。
    ///
    /// # Errors
    /// 同 [`MovieRenderer::from_source`]（映射为 `FarmError::TransportUnavailable`：
    /// worker 起不来 = 传输不可用，细节保留影片错误原文）。
    pub fn from_env() -> Result<Option<MovieRenderer>, FarmError> {
        let Ok(source): Result<String, std::env::VarError> = std::env::var(MOVIE_SRC_ENV) else {
            return Ok(None);
        };
        if source.is_empty() {
            return Err(FarmError::TransportUnavailable {
                location: "farm::movie::MovieRenderer::from_env".to_string(),
                detail: format!(
                    "环境变量 {MOVIE_SRC_ENV} 为空串——要么不设（替身模式），要么设完整 DSL 源码"
                ),
            });
        }
        MovieRenderer::from_source(&source).map(Some).map_err(|e| FarmError::TransportUnavailable {
            location: "farm::movie::MovieRenderer::from_env".to_string(),
            detail: e.message,
        })
    }

    /// 本渲染器的配置哈希（worker 用它校验任务同源）。
    pub fn config_hash(&self) -> ConfigHash {
        self.config_hash
    }

    /// 帧数：`ceil(fps × duration)`——与 [`crate::render_video`] 的采样数
    /// **同一公式**（测试 `frame_sampling_matches_render_video_exactly` 钉死）。
    pub fn frame_count(&self) -> usize {
        (self.fps as f32 * self.duration).ceil() as usize
    }

    /// 帧号 → 时刻 `t`：`frame / fps`——同 [`crate::render_video`] 的采样时刻。
    pub fn frame_time(&self, frame: u64) -> f32 {
        frame as f32 / self.fps as f32
    }

    /// **渲染一帧**：返回 RGBA 像素字节（载荷 = 帧像素，PE-1 的真实产物）。
    ///
    /// 纯函数：同 `(源码, 帧号)` 在任何进程、任何次数下逐字节相同。
    pub fn render_pixels(&self, frame: u64) -> Vec<u8> {
        let fb: Framebuffer = crate::render_frame(
            &mut CpuRefBackend,
            &self.scene,
            self.frame_time(frame),
            self.duration,
            self.width,
            self.height,
        );
        fb.pixels
    }

    /// 单帧载荷字节数（`w * h * 4`；供文档 / 测试口径）。
    pub fn payload_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 4
    }

    /// 源码（只读）。
    pub fn source(&self) -> &str {
        &self.source
    }

    /// 帧率（DSL 声明）。
    pub fn fps(&self) -> u32 {
        self.fps
    }

    /// 时长（秒，DSL 声明）。
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// 分辨率宽 / 高（DSL 声明）。
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// 影片作业（主节点侧）：一份 DSL 源 + 派生参数 + 分发 / 参考渲染入口。
#[derive(Debug, Clone)]
pub struct MovieJob {
    /// 渲染器（兼作序列参考实现）。
    renderer: MovieRenderer,
}

impl MovieJob {
    /// 从 DSL 源码构建作业。
    ///
    /// # Errors
    /// 同 [`MovieRenderer::from_source`]。
    pub fn new(source: &str) -> Result<MovieJob, MovieError> {
        Ok(MovieJob { renderer: MovieRenderer::from_source(source)? })
    }

    /// 借用渲染器（参数查询 / 单帧渲染）。
    pub fn renderer(&self) -> &MovieRenderer {
        &self.renderer
    }

    /// 帧数（DSL 声明派生）。
    pub fn frame_count(&self) -> usize {
        self.renderer.frame_count()
    }

    /// 配置哈希（内容寻址：同源码同哈希）。
    pub fn config_hash(&self) -> ConfigHash {
        self.renderer.config_hash()
    }

    /// 帧任务切分（帧级：余数前置的纯函数切分，见 [`split_frames`]）。
    ///
    /// # Errors
    /// 帧区间非法 / worker 数为 0 → [`FarmError`]（P5001 / P5002）。
    pub fn tasks(&self, workers: usize) -> Result<Vec<FrameTask>, FarmError> {
        let total: u64 = self.frame_count() as u64;
        if total == 0 {
            return Ok(Vec::new());
        }
        split_frames(0, total - 1, workers.max(1))?
            .into_iter()
            .map(|range| Ok(FrameTask::new(range, self.config_hash())))
            .collect()
    }

    /// **串行参考**：本进程顺序渲染全部帧（对照基准，同时也是合并判据的左侧）。
    pub fn render_reference(&self) -> Vec<Vec<u8>> {
        (0..self.frame_count() as u64).map(|f| self.renderer.render_pixels(f)).collect()
    }

    /// 参考摘要（串行全部帧的 MD5）。
    pub fn reference_digest(&self) -> String {
        manifest_digest(&self.render_reference())
    }

    /// **农场分发**：N 个 `farm-worker` 子进程本机真跑，按帧号归位合并。
    ///
    /// 源码注入由调度器完成：`dispatch_to_local_workers_movie` 在进程级分发
    /// 互斥锁内置放 [`MOVIE_SRC_ENV`]（子进程 spawn 时继承），用完即删——
    /// 并发分发之间不会串味，dispatch 之后也不泄漏到别的代码路径。
    ///
    /// # Errors
    /// 找不到 / 起不动 worker（P5009）；worker 报错 / 帧被篡改（P5007 /
    /// P5008）；分片未恰好覆盖（P5005）。
    pub fn dispatch(&self, workers: usize) -> Result<DispatchReport, FarmError> {
        let tasks: Vec<FrameTask> = self.tasks(workers)?;
        dispatch_to_local_workers_movie(&tasks, workers, self.renderer.source())
    }

    /// 同 [`MovieJob::dispatch`]，但每个 worker 一个独立断点续算缓存文件
    /// （目录内 `farm-worker-<i>.cache`；中断后重启跳过已完成帧）。
    ///
    /// # Errors
    /// 同 [`MovieJob::dispatch`]；缓存 IO 失败 ⇒ P5010。
    pub fn dispatch_with_cache(
        &self,
        workers: usize,
        cache_dir: &Path,
    ) -> Result<DispatchReport, FarmError> {
        let tasks: Vec<FrameTask> = self.tasks(workers)?;
        dispatch_to_local_workers_movie_cached(&tasks, workers, cache_dir, self.renderer.source())
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::transport::discover_worker_exe;
    use crate::farm::transport::WORKER_BIN_NAME;

    /// 影片 DSL（小分辨率：帧成本真实但体量可控；球体 + 轨道相机 + 一盏光）。
    const MOVIE_SRC: &str = "scene \"farm_movie\" {\n  sphere ball {\n    radius 1.0\n    material {\n      albedo #c0392b\n      roughness 0.35\n      metalness 0.05\n    }\n  }\n  camera main {\n    orbit {\n      center 0,0,0\n      radius 8.0\n    }\n    fov 45\n  }\n  light sun {\n    direction (1.0, -1.0, -0.5)\n    color #fff4e6\n    intensity 3.0\n  }\n}\ntimeline {\n  duration 0.5s\n  fps 8\n}\noutput {\n  resolution 64x48\n  format frames\n}\n";

    /// 吞吐基准 DSL·轻载（体量刻意放大：96x64@8fps 太小，进程启动 + 编译的固定
    /// 成本会淹没渲染成本，量不出并行的真实收益；128x96 × 20 帧让渲染成本
    /// 占主导，加速比才有统计意义）。
    const BENCH_SRC: &str = "scene \"bench\" {\n  sphere ball {\n    radius 1.1\n    material {\n      albedo #2a6f9e\n      roughness 0.3\n      metalness 0.1\n    }\n  }\n  camera main {\n    orbit {\n      center 0,0,0\n      radius 7.5\n    }\n    fov 42\n  }\n  light sun {\n    direction (1.0, -1.0, -0.5)\n    color #fff4e6\n    intensity 3.2\n  }\n}\ntimeline {\n  duration 2.0s\n  fps 10\n}\noutput {\n  resolution 128x96\n  format frames\n}\n";

    /// 吞吐基准 DSL·重载（同帧数、像素 ×4）：固定成本（spawn + 每 worker 一次
    /// 编译 + 协议往返）不变、渲染成本变 ⇒ 加速比随帧成本上升——Amdahl 的
    /// 诚实形态：分布式农场的收益来自"帧成本 >> 传输/启动成本"。
    const HEAVY_SRC: &str = "scene \"heavy\" {\n  sphere ball {\n    radius 1.1\n    material {\n      albedo #2a6f9e\n      roughness 0.3\n      metalness 0.1\n    }\n  }\n  camera main {\n    orbit {\n      center 0,0,0\n      radius 7.5\n    }\n    fov 42\n  }\n  light sun {\n    direction (1.0, -1.0, -0.5)\n    color #fff4e6\n    intensity 3.2\n  }\n}\ntimeline {\n  duration 2.0s\n  fps 10\n}\noutput {\n  resolution 256x192\n  format frames\n}\n";

    fn job() -> MovieJob {
        MovieJob::new(MOVIE_SRC).expect("影片 DSL 应可编译")
    }

    /// worker 可执行文件不在位时 loud skip（不伪装成功；既有范式的约定）。
    fn require_worker_exe() -> bool {
        if discover_worker_exe().is_none() {
            eprintln!(
                "[跳过] 未找到 {WORKER_BIN_NAME}：请先 `cargo build -p prism-render --bin {WORKER_BIN_NAME}`"
            );
            return false;
        }
        true
    }

    #[test]
    fn frame_sampling_matches_render_video_exactly() {
        // 帧数公式与时刻公式必须与 render_video 逐位一致——否则"并行第 k 帧"
        // 与"串行第 k 帧"渲染的根本不是同一时刻，MD5 一致也无意义。
        let j: MovieJob = job();
        assert_eq!(j.frame_count(), 4, "8fps × 0.5s = 4 帧");
        let compiled: prism_dsl::CompileOutput =
            prism_dsl::compile(MOVIE_SRC).expect("应可编译");
        let video: Vec<Framebuffer> = crate::render_video(
            &mut CpuRefBackend,
            &compiled.scene,
            compiled.width,
            compiled.height,
            compiled.fps,
            compiled.duration,
        );
        assert_eq!(video.len(), j.frame_count());
        let reference: Vec<Vec<u8>> = j.render_reference();
        for (k, fb) in video.iter().enumerate() {
            assert_eq!(reference[k], fb.pixels, "第 {k} 帧必须与 render_video 逐字节相同");
        }
        // 相邻帧必须不同（防止"时间参数没接上"的假确定）。
        assert_ne!(reference[0], reference[1], "相邻帧像素必须不同");
        // 载荷长度 = w·h·4（真实像素，不是 32 字节替身）。
        assert_eq!(reference[0].len(), 64 * 48 * 4);
    }

    #[test]
    fn render_pixels_is_pure_and_config_hash_identifies_source() {
        let j: MovieJob = job();
        let a: Vec<u8> = j.renderer().render_pixels(2);
        let b: Vec<u8> = j.renderer().render_pixels(2);
        assert_eq!(a, b, "同 (源码, 帧号) 两遍逐字节相同");
        assert_eq!(j.config_hash(), ConfigHash::of_bytes(MOVIE_SRC.as_bytes()));
        // 换源码 ⇒ 换配置哈希（内容寻址）。
        let other: MovieJob = MovieJob::new(BENCH_SRC).expect("应可编译");
        assert_ne!(j.config_hash(), other.config_hash());
        // 坏源码明确报错（P5011 + 建议段），不产出半截作业。
        let err: MovieError = MovieJob::new("not a movie\n").expect_err("坏源码必须拒绝");
        assert!(err.message.contains("P5011"), "须带错误码：{}", err.message);
        assert!(err.message.contains("建议"), "必含建议段（C19）：{}", err.message);
        // fps=0 的无帧作业明确拒绝。
        let zero: &str = "scene \"z\" {\n  sphere b { radius 1.0 material { albedo #fff roughness 0.5 metalness 0.0 } }\n  camera c { orbit { center 0,0,0 radius 5.0 } fov 45 }\n  light l { direction (1.0,-1.0,-0.5) color #fff intensity 1.0 }\n}\ntimeline { duration 1s fps 0 }\noutput { resolution 32x24 format frames }\n";
        assert!(MovieJob::new(zero).is_err(), "fps=0 必须拒绝");
    }

    #[test]
    fn tasks_cover_all_frames_without_gaps() {
        let j: MovieJob = job();
        for workers in [1usize, 2, 3, 4, 8] {
            let tasks: Vec<FrameTask> = j.tasks(workers).expect("可切分");
            let covered: u64 = tasks.iter().map(|t| t.frames.len()).sum();
            assert_eq!(covered, j.frame_count() as u64, "workers={workers}：不重不漏");
            for t in &tasks {
                assert_eq!(t.config_hash, j.config_hash(), "任务必须携带本作业配置");
            }
        }
    }

    #[test]
    fn parallel_movie_digest_matches_serial_on_real_pixels() {
        if !require_worker_exe() {
            return;
        }
        let j: MovieJob = job();
        let reference: Vec<Vec<u8>> = j.render_reference();
        let reference_digest: String = manifest_digest(&reference);

        // 4 worker 子进程真跑：合并 MD5 与单机串行逐字节一致（PE-1 主判据）。
        let report: DispatchReport = j.dispatch(4).expect("四 worker 分发应跑通");
        assert_eq!(report.frames, j.frame_count() as u64, "4 块必须覆盖全部帧");
        assert_eq!(
            report.digest(),
            reference_digest,
            "4 worker 合并 MD5 必须与单机串行一致（真实像素）"
        );
        // 逐字节级证据（不止 MD5）。
        assert_eq!(
            report.payloads_in_frame_order(),
            reference,
            "逐帧像素必须与串行参考逐字节相同"
        );
        // 多进程证据：至少 2 个不同节点 id（真起了多个进程）。
        assert!(report.node_ids().len() >= 2, "多进程应出现多个节点 id");
        // 载荷是真实像素而非替身。
        assert_eq!(report.shards[0].payloads[0].len(), j.renderer().payload_bytes());

        // worker 数不得改变结果（1 / 2 / 4 三档全对拍）。
        for workers in [1usize, 2, 4] {
            let r: DispatchReport = j.dispatch(workers).expect("应跑通");
            assert_eq!(r.digest(), reference_digest, "workers={workers}：MD5 必须一致");
        }
    }

    #[test]
    fn movie_resume_from_disk_cache_skips_done_frames() {
        if !require_worker_exe() {
            return;
        }
        let j: MovieJob = job();
        let reference_digest: String = manifest_digest(&j.render_reference());
        let total: usize = j.frame_count();
        assert!(total >= 4, "本 DSL 应有 ≥4 帧才能演示中断续算");

        let cache_dir: std::path::PathBuf =
            std::env::temp_dir().join(format!("prism-farm-movie-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&cache_dir).expect("建缓存目录");

        // 第一轮（模拟中断）：只下发前半程任务（4 切分取前 2 块）；
        // worker 0 的缓存落盘。
        let all_tasks: Vec<FrameTask> = j.tasks(4).expect("可切分");
        assert_eq!(all_tasks.len(), 4, "4 帧 4 切分，每块 1 帧");
        let half: Vec<FrameTask> = all_tasks.into_iter().take(2).collect();
        // 第一轮也走影片分发变体（源码在调度器锁内注入，不在测试里手置环境变量）。
        let first: Result<DispatchReport, FarmError> =
            dispatch_to_local_workers_movie_cached(&half, 1, &cache_dir, j.renderer().source());
        let first: DispatchReport = first.expect("前半程应跑通");
        let cache_file: std::path::PathBuf = cache_dir.join("farm-worker-0.cache");
        let cached_done: usize =
            crate::farm::worker::FrameCache::open(&cache_file).expect("缓存应可读").len();
        assert_eq!(
            cached_done,
            first.frames as usize,
            "中断前完成的帧必须已落盘（续算的前提）"
        );

        // 第二轮（重启后续算）：完整任务集 + 同一缓存目录。
        let second: DispatchReport =
            j.dispatch_with_cache(1, &cache_dir).expect("完整任务应跑通");
        assert_eq!(second.frames, total as u64, "完整任务覆盖全部帧");
        assert_eq!(
            second.digest(),
            reference_digest,
            "续算结果必须与一次算完逐字节一致"
        );
        let cached_all: usize =
            crate::farm::worker::FrameCache::open(&cache_file).expect("缓存应可读").len();
        assert_eq!(cached_all, total, "第二轮后缓存应收敛到全部帧");

        // 缓存里帧的载荷 == 参考渲染（续算复用不得改变内容）。
        // 当前版本两轮分发都是 1 个 worker，全部帧在 farm-worker-0.cache 里。
        let cache: crate::farm::worker::FrameCache =
            crate::farm::worker::FrameCache::open(&cache_file).expect("缓存应可读");
        for f in 0..total as u64 {
            let key: [u8; 32] = crate::farm::worker::frame_cache_key(j.config_hash(), f);
            let cached: Vec<u8> = cache
                .manifest()
                .get(&key)
                .cloned()
                .unwrap_or_else(|| panic!("第 {f} 帧必须在缓存里"));
            assert_eq!(cached, j.renderer().render_pixels(f), "缓存载荷必须等于参考渲染");
        }
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn throughput_speedup_measured_across_worker_counts() {
        if !require_worker_exe() {
            return;
        }
        // 实数报告：轻 / 重两档负载 × 1 / 2 / 4 worker 三档，真实渲染成本
        // （无空转注入——本路径的"成本"就是渲染本身）。硬门槛只有"MD5 不随
        // worker 数改变"；加速比打印入档（数值随机器波动，不做脆断言——
        // 那是把测试变成跑分）。
        //
        // 两档负载的意义：固定成本（spawn + 每 worker 一次 DSL 编译 + 协议
        // 往返）不随负载变，渲染成本随负载变 ⇒ 加速比随帧成本上升。这正是
        // "农场收益来自帧成本 >> 传输成本"的实证形态（Amdahl）；在 CPU 配额
        // 受限的平台上，加速比还受该配额上限约束。
        for (label, src) in [("轻载", BENCH_SRC), ("重载", HEAVY_SRC)] {
            let j: MovieJob = MovieJob::new(src).expect("基准 DSL 应可编译");
            let total: usize = j.frame_count();
            assert!(total >= 8, "{label}：基准作业应有足够帧数（实际 {total}）");
            let reference_digest: String = manifest_digest(&j.render_reference());

            let mut timings: Vec<(usize, std::time::Duration)> = Vec::new();
            for workers in [1usize, 2, 4] {
                let t0: std::time::Instant = std::time::Instant::now();
                let report: DispatchReport =
                    j.dispatch(workers).unwrap_or_else(|e| panic!("{workers} worker 应跑通：{e}"));
                let elapsed: std::time::Duration = t0.elapsed();
                assert_eq!(
                    report.digest(),
                    reference_digest,
                    "{label} workers={workers}：MD5 不得改变"
                );
                timings.push((workers, elapsed));
            }
            let t1: std::time::Duration = timings[0].1;
            let line: String = timings
                .iter()
                .map(|(w, t)| {
                    let speedup: f64 =
                        if t.as_secs_f64() > 0.0 { t1.as_secs_f64() / t.as_secs_f64() } else { 0.0 };
                    format!("{w}w={t:?}({speedup:.2}x)")
                })
                .collect::<Vec<_>>()
                .join(" ");
            let (w, h): (u32, u32) = j.renderer().dimensions();
            println!(
                "[PE-1 真实渲染加速比·{label}] {total} 帧 @ {w}x{h} {}fps：{}；MD5 {}",
                j.renderer().fps(),
                line,
                reference_digest
            );
        }
    }
}
