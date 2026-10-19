//! 分布式渲染农场调度器。
//!
//! 语义权威：固定形状归约代数 + 工程四律、（帧维
//! 复用）；帧级天然并行；分布式 P2 设计：任务 = 帧号
//! 区间 + 配置哈希，产物 = 帧文件 + manifest，主节点校验 md5 合帧、D-16
//! （Mini Code）、宪章 C18（三段式错误，永不 panic）、C19（必含建议段）。
//!
//! # 一句话
//! **切分是输入规模的纯函数，归约形状是输入规模的纯函数，二者都与执行环境
//! 无关——于是"并行 md5 == 串行 md5"是定理而非巧合。**
//!
//! # 为什么确定性"免费"
//!
//!  工程四律之二：**帧级是天然并行安全区**（帧间无依赖），四律之一：任何
//! 归约点不得私定结合序，一律走唯一形状函数 [`prism_core::graph::reduce`]。
//! 把两条并起来读，得到本模块的核心纪律：
//!
//! > **worker 只产出"逐帧部分结果"，绝不本地预归约；归约只在主节点发生一次，
//! > 形状是 [`prism_core::graph::reduction_shape`]。**
//!
//! 这条纪律直接给出三条可证的性质：
//!
//! 1. **切分是纯函数**：[`split_frames`] / [`split_frames_shape_aligned`] 只读
//!    `(frame_start, frame_end, num_workers)`，不读时钟、不读线程数、不读
//!    `HashMap` 迭代序。同输入两遍逐字段相同。
//! 2. **合并不重不漏**：[`merge_shards`] 按**帧号**（数据内容）而非**完成
//!  顺序**归位（律三"消序用排序"），并在合并点校验覆盖完整性。
//! 3. **分组不变性**：把 n 帧切成 1 份或 k 份，切完再按帧号拼回同一个序列，
//!    [`prism_core::graph::reduce`] 看到的是**同一棵树** ⇒ 任意并行度、任意
//!    机器数、任意完成顺序，逐位一致。
//!
//! 推论（负面对照，也被测试钉住）：**若 worker 本地先预归约再交给主节点，
//! 结合序就变了**——即使切分完全确定，浮点结果也会漂移。所以本模块的
//! [`FrameShard`] 刻意只装逐帧值（[`FrameShard::per_frame`]），没有
//! "worker 汇总值"这个字段可填。
//!
//! # 错误码
//!
//! 沿用宪章 C18 三段式（在哪 / 什么事 / 建议）。P5xxx 段 = 农场调度（本轮新
//! 段，见 [`FarmError`]）。

use std::fmt;

mod digest;
mod evidence;
mod merge;
mod queue;
mod scheduler;
mod split;
mod task;
mod transport;
mod worker;

pub use digest::md5_hex;
pub use evidence::{
    frame_payload, manifest_digest, parallel_manifest_digest, serial_manifest_digest,
};
pub use merge::{
    merge_shards, merge_shards_full, reduce_layered_parallel, FrameShard, FrameValue, MergeAudit,
};
pub use queue::{FarmQueue, QueueState, RetryPolicy, SubmitOutcome, ThroughputLedger};
pub use scheduler::{
    dispatch_to_local_workers, dispatch_to_local_workers_with_cache, flatten_in_frame_order,
    plan_dispatch, CachedShardOutcome, DispatchPlan, DispatchReport, FarmNode, LocalScheduler,
    ShardOutcome,
};
pub use split::{
    frame_count, slot_counts, slot_ranges, split_frames, split_frames_shape_aligned,
};
pub use task::{ConfigHash, FrameTask, ResourceNeed, TaskId, TaskOrderKey};
pub use transport::{
    checksum_hex, checksum_of, decode_frame, decode_shard, decode_task, discover_worker_exe,
    encode_frame, encode_shard, encode_task, read_frame, spawn_local_workers, write_frame,
    FrameKind, FramePayloadShard, InProcessTransport, LocalProcessTransport, ShardTransport,
    TcpTransport, TransportKind, CHECKSUM_BYTES, FRAME_MAGIC, HEADER_BYTES, MAX_FRAMES_PER_SHARD,
    MAX_PAYLOAD_BYTES, TRANSPORT_VERSION, WORKER_BIN_NAME,
};
pub use worker::{
    frame_cache_key, render_frame_payload, render_frame_payload_cost, run_stdio_worker, run_task,
    run_task_with_cost, run_worker_stream, FrameCache, WorkerOptions, WorkerStats,
    DEFAULT_WORK_ITERS, FRAME_PAYLOAD_BYTES,
};

// ---------------------------------------------------------------------------
// 错误（C18 三段式；C19 必含建议段）
// ---------------------------------------------------------------------------

/// 农场调度错误。
///
/// 三段式：**在哪**（`location`）/ **什么事**（检测到的事实）/ **建议**
/// （可行动修复）。错误码取 P5xxx 段（农场调度，本轮新段——C18 已定义的
/// P1xxx 防呆参数 / P2xxx 类型语法 / P3xxx 图结构 / P4xxx 资产 IO 都不覆盖
/// 调度域，故新开 P5xxx；待裁决）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FarmError {
    /// 帧区间非法：`end < start`（空区间 / 负区间）。
    InvalidFrameRange {
        /// 三段式一【在哪】。
        location: String,
        /// 区间起点。
        start: u64,
        /// 区间终点。
        end: u64,
    },
    /// 工作单元数为 0：无法切分（律四的"分布式同树"要求至少 1 份）。
    ZeroWorkers {
        /// 三段式一【在哪】。
        location: String,
    },
    /// 帧数溢出：`end - start + 1` 超出 `u64`（仅当 `start = 0`、
    /// `end = u64::MAX` 时可达）。
    FrameCountOverflow {
        /// 三段式一【在哪】。
        location: String,
        /// 区间起点。
        start: u64,
        /// 区间终点。
        end: u64,
    },
    /// 归约槽位不可推导：内部 `reduction_shape` 的层结构与帧数不自洽。
    SlotDerivationFailed {
        /// 三段式一【在哪】。
        location: String,
        /// 帧数。
        frames: u64,
        /// 请求的槽深度。
        depth: usize,
    },
    /// 合并期覆盖不完整：分片未恰好覆盖全部帧（漏帧或重帧）。
    FrameCoverageBroken {
        /// 三段式一【在哪】。
        location: String,
        /// 首个出问题的帧号。
        frame: u64,
        /// 期望帧数。
        expected: u64,
        /// 实际归位帧数。
        got: usize,
    },
    /// 分片内部契约被破坏：逐帧值条数与帧区间的帧数不等。
    ShardLengthMismatch {
        /// 三段式一【在哪】。
        location: String,
        /// 分片帧区间。
        start: u64,
        /// 分片帧区间终点。
        end: u64,
        /// 声明的帧数。
        declared: u64,
        /// 逐帧值条数。
        values: usize,
    },
    /// 线格式畸形：无法解析为农场任务 / 分片。
    MalformedEnvelope {
        /// 三段式一【在哪】。
        location: String,
        /// 畸形细节。
        detail: String,
    },
    /// 校验和不符：帧在传输/存储中被篡改或损坏（长度前缀帧协议的 SHA-256 兜底）。
    ///
    /// 与 [`FarmError::MalformedEnvelope`] 的分工：结构层面能解析、但**内容**
    /// 与声明的摘要不符，归本变体——这样"改了一个字节"与"根本不是本格式"在报错
    /// 上可区分（排查方向完全不同）。
    ChecksumMismatch {
        /// 三段式一【在哪】。
        location: String,
        /// 按收到的头部 + 载荷重算出的摘要（小写十六进制）。
        expected: String,
        /// 帧里携带的摘要（小写十六进制）。
        actual: String,
    },
    /// 传输不可用：进程起不来 / 管道断裂 / 该变体本轮未实装。
    TransportUnavailable {
        /// 三段式一【在哪】。
        location: String,
        /// 具体原因。
        detail: String,
    },
    /// 缓存 IO 失败：断点续算的 `cache_manifest` 读写不成功。
    CacheIo {
        /// 三段式一【在哪】。
        location: String,
        /// 具体原因。
        detail: String,
    },
}

impl fmt::Display for FarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FarmError::InvalidFrameRange { location, start, end } => write!(
                f,
                "error[P5001] {location}\n  \
                 什么事：帧区间非法——起点 {start} 大于终点 {end}（空区间 / 负区间）\n  \
                 建议：给出 start <= end 的闭区间；若意图是空作业，请直接提交空任务列表而不是空区间"
            ),
            FarmError::ZeroWorkers { location } => write!(
                f,
                "error[P5002] {location}\n  \
                 什么事：工作单元数为 0——无法切分（分布式同树要求至少 1 份）\n  \
                 建议：传入 num_workers >= 1；单机顺序执行请显式传 1"
            ),
            FarmError::FrameCountOverflow { location, start, end } => write!(
                f,
                "error[P5003] {location}\n  \
                 什么事：帧数溢出——区间 [{start}, {end}] 的长度 end - start + 1 超出 u64\n  \
                 建议：把作业拆成多段分别下发；单段帧号跨度不要覆盖 0..=u64::MAX 全域"
            ),
            FarmError::SlotDerivationFailed { location, frames, depth } => write!(
                f,
                "error[P5004] {location}\n  \
                 什么事：无法从 reduction_shape({frames}) 的层结构推导深度 {depth} 的槽划分\n  \
                 建议：这是内部不自洽，请附上帧数与深度报给引擎维护者；\
                 临时绕行可改用 split_frames（均分切分，同样是纯函数）"
            ),
            FarmError::FrameCoverageBroken { location, frame, expected, got } => write!(
                f,
                "error[P5005] {location}\n  \
                 什么事：分片未恰好覆盖全部帧——首个问题在帧 {frame}，期望 {expected} 帧，实际归位 {got} 帧（漏帧或重帧）\n  \
                 建议：核对各分片的帧区间首尾相接（前一 end + 1 == 后一 start）且并集等于 [0, {expected})；\
                 切分请走 split_frames / split_frames_shape_aligned 以保证不重不漏"
            ),
            FarmError::ShardLengthMismatch { location, start, end, declared, values } => write!(
                f,
                "error[P5006] {location}\n  \
                 什么事：分片契约被破坏——帧区间 [{start}, {end}] 声明 {declared} 帧，却带了 {values} 个逐帧值\n  \
                 建议：worker 必须为区间内**每一帧**产出一个值（含尾帧）；\
                 FrameShard::per_frame 的长度必须等于 FrameRange::len()"
            ),
            FarmError::MalformedEnvelope { location, detail } => write!(
                f,
                "error[P5007] {location}\n  \
                 什么事：线格式畸形——{detail}\n  \
                 建议：核对字段是否齐全（FrameTask 需 frames/config_hash/priority/resources；\
                 FrameShard 需 frames/values）；本轮线格式由 prism-infra 的 farm_wire 编解码"
            ),
            FarmError::ChecksumMismatch { location, expected, actual } => write!(
                f,
                "error[P5008] {location}\n  \
                 什么事：SHA-256 校验和不符——按收到的头部+载荷重算得 {expected}，帧内声明 {actual}\n  \
                 建议：帧在传输途中被改写或落盘时损坏；丢弃该帧并在主节点对该分片重派一次\
                 （渲染是纯函数，重算结果与首次一致）；若反复不符，检查是否有进程在同时写同一个管道"
            ),
            FarmError::TransportUnavailable { location, detail } => write!(
                f,
                "error[P5009] {location}\n  \
                 什么事：传输不可用——{detail}\n  \
                 建议：本地子进程路径请先 `cargo build -p prism-render --bin farm-worker`，\
                 或设环境变量 PRISM_FARM_WORKER 指向该可执行文件；跨机 TCP 变体本轮未实装，\
                 请改用本地子进程（同一条帧协议，仅介质不同）"
            ),
            FarmError::CacheIo { location, detail } => write!(
                f,
                "error[P5010] {location}\n  \
                 什么事：断点续算缓存 IO 失败——{detail}\n  \
                 建议：确认缓存目录可写且磁盘未满；缓存文件损坏时删除后重建\
                 （已完成的帧会重算，结果不变，只是慢一轮）；C13 决定旧版本缓存只失效不误复用"
            ),
        }
    }
}

impl std::error::Error for FarmError {}
