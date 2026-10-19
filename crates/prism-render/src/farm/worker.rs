//! 农场 worker：**读任务 → 渲染帧 → 回传**，附**断点续算**（cache_manifest 落地）。
//!
//! # 三件事
//!
//! 1. **渲染一帧**（[`render_frame_payload`]）：`(配置哈希, 帧号)` 的**纯函数**，
//!    产出定长 32 字节载荷。这是真实农场的"渲染一帧像素"替身——本轮验的是
//!    **调度/传输/续算的确定性**，不是成像正确性（见汇报"诚实边界"）。
//! 2. **算一个任务**（[`run_task`]）：逐帧产出载荷，长度恒等于帧区间长度
//!    （不静默补帧，C20）。命中缓存（[`FrameCache`]）的帧**直接复用、不重算**。
//! 3. **服务一条 stdio 帧流**（[`run_stdio_worker`]）：`farm-worker` 子进程的全部
//!    逻辑，走 [`crate::farm::transport`] 的长度前缀 + SHA-256 帧协议。
//!
//! # 断点续算（接 `prism_io::cache_manifest`）
//!
//! [`FrameCache`] 是 `prism_io::CacheManifest` 的一层薄封装（**直接复用**既有
//! C3 件，不新写第二份清单格式）：
//!
//! - **键** = `sha256("PRISM-FARM-CACHE" ++ cfg_be ++ frame_be)`——内容派生、
//!   定长 32 字节，与 `CacheManifest` 的键域同一；
//! - **值** = 该帧的 32 字节载荷；
//! - **落盘** = `CacheManifest::to_bytes()`（键升序、大端定长、版本头 ⇒ 同内容
//!   两次导出逐字节一致），写临时文件后 `rename`（原子替换，半截文件不会顶替
//!   好文件）。
//!
//! 于是"中断后重启跳过已完成帧"就是一句 `if cache.has(cfg, f) { 复用 }`：缓存里
//! 有的帧绝不重算，没有的帧必算——**不重不漏**，且**结果与算过几遍无关**（渲染
//! 是纯函数，缓存里存的就是这个纯函数的像）。
//!
//! # 渲染成本是"可配置空转"，且**不泄漏进结果**
//!
//! [`render_frame_payload_cost`] 的 `work_iters` 只控制空转循环次数（供吞吐
//! 实测放大 CPU 时间），累加结果一律 `black_box` 丢弃——所以**同 (配置, 帧)
//! 在任何 `work_iters` 下逐字节同载荷**。这不是巧合而是设计：吞吐实验不允许
//! 改变正确性判据。

use crate::farm::task::{ConfigHash, FrameTask};
use crate::farm::transport::{
    decode_task, encode_shard, read_frame, write_frame, FrameKind, FramePayloadShard,
};
use crate::farm::FarmError;
use prism_io::CacheManifest;
use std::path::{Path, PathBuf};

/// 单帧载荷字节数（32：非 2 的幂帧数会让 MD5 走到 padding 分支，压测更狠）。
pub const FRAME_PAYLOAD_BYTES: usize = 32;

/// 缓存键的域分隔标签（防与 tile / 快照键跨域拼接歧义）。
const CACHE_TAG: &[u8; 16] = b"PRISM-FARM-CACHE";

/// splitmix64 增量（纯整数、零依赖、跨平台逐位一致）。
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// 一帧的**渲染成本**（空转迭代数）；`0` = 不空转（生产默认）。
pub const DEFAULT_WORK_ITERS: u32 = 0;

// ---------------------------------------------------------------------------
// 渲染（纯函数）
// ---------------------------------------------------------------------------

/// splitmix64 单步：状态推进 + 输出混合（纯整数）。
fn splitmix_next(state: u64) -> (u64, u64) {
    let next: u64 = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z: u64 = next;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (next, z)
}

/// **渲染一帧**：`(配置哈希, 帧号)` → 定长 32 字节载荷的**纯函数**。
///
/// 同输入两遍逐字节相同；与机器、线程数、进程数、`work_iters` 无关。
pub fn render_frame_payload(config_hash: ConfigHash, frame: u64) -> Vec<u8> {
    render_frame_payload_cost(config_hash, frame, DEFAULT_WORK_ITERS)
}

/// 同上，但可注入**渲染成本**（空转迭代数，供吞吐实测）。
///
/// `work_iters` **不影响输出**：空转累加值一律 `black_box` 丢弃，载荷只由
/// `(config_hash, frame)` 决定。测试 `cost_does_not_leak_into_payload` 钉住这一点。
pub fn render_frame_payload_cost(
    config_hash: ConfigHash,
    frame: u64,
    work_iters: u32,
) -> Vec<u8> {
    // ---- 可配置成本：与输出无关的确定性空转 ----
    let mut acc: u64 = config_hash
        .value()
        .wrapping_mul(SPLITMIX_GAMMA)
        .wrapping_add(frame)
        .wrapping_add(0x5eed_1234_abcd_0001);
    for i in 0..work_iters {
        acc = acc.rotate_left(7) ^ acc.wrapping_mul(0x2545_f491_4f6c_dd1d);
        acc = acc.wrapping_add(SPLITMIX_GAMMA ^ u64::from(i & 0xff));
        acc ^= acc >> 29;
    }
    // 丢弃副作用：编译器不得把上面整段优化掉，但它也不进载荷。
    std::hint::black_box(acc);

    // ---- 载荷：只由 (config_hash, frame) 决定 ----
    let mut state: u64 = config_hash.value() ^ frame.wrapping_mul(SPLITMIX_GAMMA);
    let mut out: Vec<u8> = Vec::with_capacity(FRAME_PAYLOAD_BYTES);
    while out.len() < FRAME_PAYLOAD_BYTES {
        let (next, word) = splitmix_next(state);
        state = next;
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.truncate(FRAME_PAYLOAD_BYTES);
    out
}

// ---------------------------------------------------------------------------
// 帧缓存（断点续算；prism_io::CacheManifest 薄封装）
// ---------------------------------------------------------------------------

/// 某帧的缓存键：`sha256(域标签 ++ cfg_be ++ frame_be)`（定长 32 字节）。
pub fn frame_cache_key(config_hash: ConfigHash, frame: u64) -> [u8; 32] {
    let mut buf: Vec<u8> = Vec::with_capacity(CACHE_TAG.len() + 16);
    buf.extend_from_slice(CACHE_TAG);
    buf.extend_from_slice(&config_hash.value().to_be_bytes());
    buf.extend_from_slice(&frame.to_be_bytes());
    prism_io::sha256(&buf)
}

/// 逐帧缓存：键（`sha256(cfg, frame)`）→ 32 字节载荷，可按路径落盘续用。
///
/// 直接持有 [`CacheManifest`]：清单的确定性序列化（键升序 + 版本头）由
/// `prism-io` 既有实现负责，本结构只管"帧语义"与原子落盘。
#[derive(Debug, Clone, Default)]
pub struct FrameCache {
    manifest: CacheManifest,
    path: Option<PathBuf>,
    dirty: bool,
}

impl FrameCache {
    /// 空内存缓存（不落盘）。
    pub fn new() -> FrameCache {
        FrameCache { manifest: CacheManifest::new(), path: None, dirty: false }
    }

    /// 打开/新建磁盘缓存：文件不存在 ⇒ 空缓存（首次运行），存在 ⇒ 解析。
    ///
    /// 解析失败（损坏 / 版本不符）**不静默重建**：返回结构化错误，由调用方决定
    /// 是否删文件重建（C13：旧格式只失效、不误复用）。
    pub fn open(path: impl Into<PathBuf>) -> Result<FrameCache, FarmError> {
        let path: PathBuf = path.into();
        let location: &str = "farm::worker::FrameCache::open";
        if !path.exists() {
            return Ok(FrameCache { manifest: CacheManifest::new(), path: Some(path), dirty: false });
        }
        let bytes: Vec<u8> = std::fs::read(&path).map_err(|e| FarmError::CacheIo {
            location: location.to_string(),
            detail: format!("读取 {} 失败：{e}", path.display()),
        })?;
        let manifest: CacheManifest = CacheManifest::from_bytes(&bytes).map_err(|e| {
            FarmError::CacheIo {
                location: location.to_string(),
                detail: format!("解析 {} 失败：{e}", path.display()),
            }
        })?;
        Ok(FrameCache { manifest, path: Some(path), dirty: false })
    }

    /// 缓存文件路径（内存缓存为 `None`）。
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// 是否命中某帧。
    pub fn has(&self, config_hash: ConfigHash, frame: u64) -> bool {
        self.manifest.contains(&frame_cache_key(config_hash, frame))
    }

    /// 取某帧载荷（未命中 ⇒ `None`，不返回默认值）。
    pub fn get(&self, config_hash: ConfigHash, frame: u64) -> Option<Vec<u8>> {
        self.manifest.get(&frame_cache_key(config_hash, frame)).cloned()
    }

    /// 写入某帧载荷（标记脏，等待 [`FrameCache::flush`]）。
    pub fn put(&mut self, config_hash: ConfigHash, frame: u64, payload: Vec<u8>) {
        self.manifest.insert(frame_cache_key(config_hash, frame), payload);
        self.dirty = true;
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.manifest.len()
    }

    /// 是否空。
    pub fn is_empty(&self) -> bool {
        self.manifest.is_empty()
    }

    /// 是否有未落盘的改动。
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// 借用底层清单（只读）。
    pub fn manifest(&self) -> &CacheManifest {
        &self.manifest
    }

    /// 落盘（原子：写 `.tmp` 后 `rename`；无路径 / 无改动 ⇒ 空操作）。
    pub fn flush(&mut self) -> Result<(), FarmError> {
        let Some(path) = self.path.clone() else {
            self.dirty = false;
            return Ok(());
        };
        if !self.dirty {
            return Ok(());
        }
        let location: &str = "farm::worker::FrameCache::flush";
        let bytes: Vec<u8> = self.manifest.to_bytes();
        let tmp: PathBuf = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| FarmError::CacheIo {
            location: location.to_string(),
            detail: format!("写临时缓存 {} 失败：{e}", tmp.display()),
        })?;
        std::fs::rename(&tmp, &path).map_err(|e| FarmError::CacheIo {
            location: location.to_string(),
            detail: format!("原子替换 {} 失败：{e}", path.display()),
        })?;
        self.dirty = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 算一个任务
// ---------------------------------------------------------------------------

/// 任务执行统计（续算证据：复用多少帧、真算多少帧）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkerStats {
    /// 任务帧数。
    pub frames: u64,
    /// 实际渲染的帧数。
    pub rendered: u64,
    /// 命中缓存复用（跳过渲染）的帧数。
    pub reused: u64,
}

/// **算一个任务**：逐帧产出载荷（命中缓存即复用）。
///
/// 返回的分片**逐帧齐备**（`payloads.len() == frames.len()`），节点 id 由调用方
/// 在回传时填入（worker 内部不知道自己的对外名）。
///
/// 确定性：帧 `f` 的载荷只由 `(config_hash, f)` 决定，与遍历顺序、缓存状态、
/// 曾算过几遍都无关 ⇒ 续算与全算逐字节一致。
pub fn run_task_with_cost(
    task: &FrameTask,
    cache: &mut FrameCache,
    work_iters: u32,
) -> (FramePayloadShard, WorkerStats) {
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(usize_min(task.frames.len()));
    let mut stats: WorkerStats = WorkerStats::default();
    let mut frame: u64 = task.frames.start;
    loop {
        stats.frames = stats.frames.saturating_add(1);
        match cache.get(task.config_hash, frame) {
            Some(payload) => {
                stats.reused = stats.reused.saturating_add(1);
                payloads.push(payload);
            }
            None => {
                let payload: Vec<u8> =
                    render_frame_payload_cost(task.config_hash, frame, work_iters);
                cache.put(task.config_hash, frame, payload.clone());
                stats.rendered = stats.rendered.saturating_add(1);
                payloads.push(payload);
            }
        }
        if frame == task.frames.end {
            break;
        }
        frame = frame.saturating_add(1);
    }
    (
        FramePayloadShard::new("", task.frames, task.config_hash, payloads),
        stats,
    )
}

/// [`run_task_with_cost`] 的零成本版（生产默认）。
pub fn run_task(task: &FrameTask, cache: &mut FrameCache) -> (FramePayloadShard, WorkerStats) {
    run_task_with_cost(task, cache, DEFAULT_WORK_ITERS)
}

/// `u64` 帧数 → `usize` 容量提示（饱和，不用于正确性）。
fn usize_min(frames: u64) -> usize {
    usize::try_from(frames).unwrap_or(usize::MAX)
}

// ---------------------------------------------------------------------------
// stdio worker（`farm-worker` 子进程的全部逻辑）
// ---------------------------------------------------------------------------

/// worker 启动选项。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerOptions {
    /// 断点续算缓存文件（`None` = 纯内存，进程退出即失）。
    pub cache_path: Option<PathBuf>,
    /// 渲染成本（空转迭代数；不影响输出）。
    pub work_iters: u32,
    /// 对外节点 id（`None` = `worker-<pid>`）。
    pub node_id: Option<String>,
}

impl WorkerOptions {
    /// 默认选项（零成本、内存缓存）。
    pub fn new() -> WorkerOptions {
        WorkerOptions { cache_path: None, work_iters: DEFAULT_WORK_ITERS, node_id: None }
    }

    /// 设定缓存文件。
    pub fn with_cache(mut self, path: impl Into<PathBuf>) -> WorkerOptions {
        self.cache_path = Some(path.into());
        self
    }

    /// 设定渲染成本。
    pub fn with_cost(mut self, work_iters: u32) -> WorkerOptions {
        self.work_iters = work_iters;
        self
    }
}

/// **worker 主循环**：从 `input` 读任务帧，向 `output` 回分片帧，见 `Done` / EOF 退出。
///
/// 协议（lock-step）：
///
/// ```text
/// worker → master : Hello(node_id)
/// master → worker : Task(encode_task)
/// worker → master : Shard(encode_shard)     ← 每收一个 Task 回一个 Shard
/// master → worker : Done                    ← 或直接关管道（干净 EOF）
/// ```
///
/// 返回已处理的任务数（供 bin 打印）。
///
/// # Errors
/// 线格式畸形 / 校验和不符 / 缓存 IO 失败 → [`FarmError`]（三段式，已 flush 缓存）。
pub fn run_worker_stream<R: std::io::Read, W: std::io::Write>(
    input: &mut R,
    output: &mut W,
    options: &WorkerOptions,
) -> Result<u32, FarmError> {
    let node_id: String = options
        .node_id
        .clone()
        .unwrap_or_else(|| format!("worker-{}", std::process::id()));
    let mut cache: FrameCache = match &options.cache_path {
        Some(p) => FrameCache::open(p)?,
        None => FrameCache::new(),
    };
    write_frame(output, FrameKind::Hello, node_id.as_bytes())?;
    let mut served: u32 = 0;
    loop {
        let next = read_frame(input)?;
        match next {
            None => break,                       // 干净 EOF：master 关了管道
            Some((FrameKind::Done, _)) => break, // 显式收工
            Some((FrameKind::Hello, _)) => continue,
            Some((FrameKind::Task, payload)) => {
                let task: FrameTask = decode_task(&payload)?;
                let (mut shard, stats) =
                    run_task_with_cost(&task, &mut cache, options.work_iters);
                shard.node_id = node_id.clone();
                if !shard.is_complete() {
                    // 绝不回半截分片：宁可报错让主节点重派（C20）。
                    let err: FarmError = FarmError::ShardLengthMismatch {
                        location: "farm::worker::run_worker_stream".to_string(),
                        start: shard.frames.start,
                        end: shard.frames.end,
                        declared: shard.frames.len(),
                        values: shard.payloads.len(),
                    };
                    let _ = write_frame(output, FrameKind::Error, err.to_string().as_bytes());
                    return Err(err);
                }
                let _ = stats; // 统计本轮不落盘（manifest 由主节点持有）
                write_frame(output, FrameKind::Shard, &encode_shard(&shard))?;
                served = served.saturating_add(1);
            }
            Some((kind, _)) => {
                let err: FarmError = FarmError::MalformedEnvelope {
                    location: "farm::worker::run_worker_stream".to_string(),
                    detail: format!("worker 不接受 {kind:?} 帧"),
                };
                let _ = write_frame(output, FrameKind::Error, err.to_string().as_bytes());
                return Err(err);
            }
        }
    }
    // 无论正常退出还是中断，落盘已完成帧（断点续算的全部意义）。
    if let Err(e) = cache.flush() {
        let _ = write_frame(output, FrameKind::Error, e.to_string().as_bytes());
        return Err(e);
    }
    Ok(served)
}

/// 在进程的标准输入/输出上跑 worker（`farm-worker --stdio` 的实现）。
pub fn run_stdio_worker(options: &WorkerOptions) -> Result<u32, FarmError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    run_worker_stream(&mut input, &mut output, options)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::transport::{decode_shard, encode_task, read_frame};
    use crate::incremental::FrameRange;
    use crate::farm::task::ResourceNeed;

    fn task(start: u64, end: u64, cfg: &[u8]) -> FrameTask {
        FrameTask {
            frames: FrameRange { start, end },
            config_hash: ConfigHash::of_bytes(cfg),
            priority: 0,
            resources: ResourceNeed::unbounded(),
        }
    }

    /// 唯一临时目录（避免测试间互相踩）。
    fn temp_path(tag: &str) -> PathBuf {
        let mut p: PathBuf = std::env::temp_dir();
        p.push(format!("prism-farm-worker-{}-{tag}.cache", std::process::id()));
        p
    }

    #[test]
    fn render_is_pure_and_cost_does_not_leak_into_payload() {
        let cfg: ConfigHash = ConfigHash::of_bytes(b"render");
        // 纯函数：两遍一致。
        assert_eq!(render_frame_payload(cfg, 7), render_frame_payload(cfg, 7));
        // 定长 32 字节。
        assert_eq!(render_frame_payload(cfg, 0).len(), FRAME_PAYLOAD_BYTES);
        // 不同帧 / 不同配置 ⇒ 不同载荷。
        assert_ne!(render_frame_payload(cfg, 1), render_frame_payload(cfg, 2));
        assert_ne!(
            render_frame_payload(cfg, 1),
            render_frame_payload(ConfigHash::of_bytes(b"other"), 1)
        );
        // 成本不泄漏：任意 work_iters 下载荷逐字节相同。
        for iters in [0u32, 1, 17, 1000, 50_000] {
            assert_eq!(
                render_frame_payload_cost(cfg, 42, iters),
                render_frame_payload(cfg, 42),
                "work_iters={iters} 不得改变载荷（否则吞吐实验会污染正确性判据）"
            );
        }
    }

    #[test]
    fn frame_cache_key_is_content_derived_and_domain_separated() {
        let cfg: ConfigHash = ConfigHash::of_bytes(b"k");
        assert_eq!(frame_cache_key(cfg, 3), frame_cache_key(cfg, 3), "纯函数");
        assert_ne!(frame_cache_key(cfg, 3), frame_cache_key(cfg, 4), "帧不同键不同");
        assert_ne!(
            frame_cache_key(cfg, 3),
            frame_cache_key(ConfigHash::of_bytes(b"k2"), 3),
            "配置不同键不同"
        );
        assert_eq!(frame_cache_key(cfg, 3).len(), 32, "SHA-256 定长 32 字节");
    }

    #[test]
    fn cache_roundtrips_through_disk_deterministically() {
        let path: PathBuf = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);
        let cfg: ConfigHash = ConfigHash::of_bytes(b"disk");
        {
            let mut cache: FrameCache = FrameCache::open(&path).expect("首次打开应为空缓存");
            assert!(cache.is_empty());
            for f in 0..5u64 {
                cache.put(cfg, f, render_frame_payload(cfg, f));
            }
            assert_eq!(cache.len(), 5);
            assert!(cache.is_dirty());
            cache.flush().expect("落盘应成功");
            assert!(!cache.is_dirty(), "落盘后不得仍标脏");
            assert!(path.exists(), "缓存文件必须存在");
        }
        // 重新打开：内容逐帧一致。
        let reopened: FrameCache = FrameCache::open(&path).expect("重新打开应成功");
        assert_eq!(reopened.len(), 5);
        for f in 0..5u64 {
            assert_eq!(
                reopened.get(cfg, f),
                Some(render_frame_payload(cfg, f)),
                "第 {f} 帧必须原样读回"
            );
            assert!(reopened.has(cfg, f));
        }
        // 未写入的帧不命中（不返回默认值）。
        assert_eq!(reopened.get(cfg, 99), None);
        // 两次导出逐字节一致（清单确定性由 prism-io 保证，这里钉住整链）。
        let first: Vec<u8> = reopened.manifest().to_bytes();
        let second: Vec<u8> = FrameCache::open(&path).expect("再打开").manifest().to_bytes();
        assert_eq!(first, second, "同内容清单两次导出必须逐字节一致");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_task_is_complete_and_reuses_cached_frames() {
        let cfg: ConfigHash = ConfigHash::of_bytes(b"task");
        let t: FrameTask = task(100, 109, b"task");
        let mut cache: FrameCache = FrameCache::new();

        let (shard, stats) = run_task(&t, &mut cache);
        assert!(shard.is_complete(), "10 帧必须 10 个载荷");
        assert_eq!(stats, WorkerStats { frames: 10, rendered: 10, reused: 0 });
        assert_eq!(cache.len(), 10, "逐帧入缓存");
        // 逐帧载荷 == 纯函数值（顺序与帧号严格对应）。
        for (i, p) in shard.payloads.iter().enumerate() {
            assert_eq!(*p, render_frame_payload(cfg, 100 + i as u64));
        }
        // 再算一遍：全部命中缓存，零渲染，载荷逐字节不变。
        let (again, stats2) = run_task(&t, &mut cache);
        assert_eq!(stats2, WorkerStats { frames: 10, rendered: 0, reused: 10 });
        assert_eq!(again.payloads, shard.payloads, "复用不得改变结果");
        assert_eq!(again.digest(), shard.digest());
    }

    #[test]
    fn worker_stream_serves_single_and_multi_frame_tasks() {
        let cfg: ConfigHash = ConfigHash::of_bytes(b"stream");
        let tasks: Vec<FrameTask> = vec![task(0, 0, b"stream"), task(1, 8, b"stream")];

        // 构造 master → worker 的输入流：Task, Task, Done。
        let mut input: Vec<u8> = Vec::new();
        for t in &tasks {
            crate::farm::transport::write_frame(
                &mut input,
                FrameKind::Task,
                &encode_task(t),
            )
            .expect("写任务应成功");
        }
        crate::farm::transport::write_frame(&mut input, FrameKind::Done, b"").expect("写收工应成功");

        let mut output: Vec<u8> = Vec::new();
        let served: u32 = run_worker_stream(
            &mut std::io::Cursor::new(input),
            &mut output,
            &WorkerOptions::new(),
        )
        .expect("worker 应正常收工");
        assert_eq!(served, 2, "两个任务都必须被服务");

        // 解析 worker 的输出：Hello + 2 个 Shard。
        let mut reader = std::io::Cursor::new(output);
        let (k0, p0) = read_frame(&mut reader).expect("读 Hello").expect("Hello 必须存在");
        assert_eq!(k0, FrameKind::Hello);
        assert!(!p0.is_empty(), "Hello 载荷必须带节点 id");
        let mut shards: Vec<FramePayloadShard> = Vec::new();
        while let Some((k, p)) = read_frame(&mut reader).expect("读分片") {
            assert_eq!(k, FrameKind::Shard, "worker 只应回 Shard");
            shards.push(decode_shard(&p).expect("分片应可解码"));
        }
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].frames, FrameRange { start: 0, end: 0 }, "单帧任务");
        assert_eq!(shards[0].len(), 1);
        assert_eq!(shards[1].frames, FrameRange { start: 1, end: 8 }, "多帧任务");
        assert_eq!(shards[1].len(), 8);
        for s in &shards {
            assert!(s.is_complete());
            assert!(!s.node_id.is_empty(), "回传必须带节点 id");
            assert_eq!(s.config_hash, cfg);
        }
    }

    #[test]
    fn worker_stream_resumes_from_disk_cache_and_skips_done_frames() {
        let path: PathBuf = temp_path("resume");
        let _ = std::fs::remove_file(&path);
        let t: FrameTask = task(0, 5, b"resume");

        // 第一轮：只服务前 3 帧（模拟中断——任务只下发 0..2）。
        let mut input: Vec<u8> = Vec::new();
        crate::farm::transport::write_frame(
            &mut input,
            FrameKind::Task,
            &encode_task(&task(0, 2, b"resume")),
        )
        .expect("写任务");
        crate::farm::transport::write_frame(&mut input, FrameKind::Done, b"").expect("写收工");
        let mut out: Vec<u8> = Vec::new();
        run_worker_stream(
            &mut std::io::Cursor::new(input),
            &mut out,
            &WorkerOptions::new().with_cache(&path),
        )
        .expect("第一轮应成功");
        let cache: FrameCache = FrameCache::open(&path).expect("缓存应已落盘");
        assert_eq!(cache.len(), 3, "中断前完成的 3 帧必须落盘");

        // 第二轮：重启进程、重下**整个**任务；前 3 帧直接复用。
        let mut cache: FrameCache = FrameCache::open(&path).expect("重新打开");
        let (shard, stats) = run_task(&t, &mut cache);
        assert_eq!(stats.reused, 3, "已完成的 3 帧必须被跳过");
        assert_eq!(stats.rendered, 3, "只剩 3-5 帧需要真算");
        assert!(shard.is_complete());
        // 续算结果与"一次算完"逐字节一致。
        let mut fresh: FrameCache = FrameCache::new();
        let (reference, _) = run_task(&t, &mut fresh);
        assert_eq!(shard.payloads, reference.payloads, "续算不得改变结果");
        assert_eq!(shard.digest(), reference.digest());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn worker_stream_reports_malformed_envelope_as_error_frame() {
        // 畸形帧（改了长度头）⇒ worker 必须报结构化错误并回 Error 帧。
        let mut bad: Vec<u8> = crate::farm::transport::encode_frame(FrameKind::Task, &[0u8; 40]);
        bad[10] ^= 0xFF; // 篡改 payload_len
        let mut out: Vec<u8> = Vec::new();
        let err = run_worker_stream(
            &mut std::io::Cursor::new(bad),
            &mut out,
            &WorkerOptions::new(),
        )
        .expect_err("畸形帧必须报错");
        assert!(err.to_string().contains("error[P50"), "必须是三段式错误码：{err}");
        // 报错前已经握手（Hello 在前），这就是协议秩序。
        let mut reader = std::io::Cursor::new(out);
        let (k, _) = read_frame(&mut reader).expect("读 Hello").expect("Hello 存在");
        assert_eq!(k, FrameKind::Hello);
    }
}
