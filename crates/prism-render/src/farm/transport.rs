//! 农场传输层：**长度前缀帧协议 + SHA-256 校验 + 本地子进程优先**。
//!
//! # 线格式（本模块唯一权威）
//!
//! ```text
//! +--------+---------+-------+-------+-------------+----------+----------+
//! | magic  | version | kind  | flags | payload_len | payload  | checksum |
//! | 4 B    | 2 B BE  | 1 B   | 1 B   | 4 B BE      | 变长     | 32 B     |
//! +--------+---------+-------+-------+-------------+----------+----------+
//! ```
//!
//! - `magic` = `PFRM`（宪章 C8：内部格式带魔数头，非本格式立刻拒收）；
//! - `version` = [`TRANSPORT_VERSION`]（C13：版本变更 ⇒ 旧对端只拒收、永不误解释）；
//! - `kind` = [`FrameKind`]（`Hello` / `Task` / `Shard` / `Done` / `Error`）；
//! - `payload_len` = 大端 `u32`，上限 [`MAX_PAYLOAD_BYTES`]（防呆：超长先拒，不预分配）；
//! - `checksum` = `sha256(域标签 ++ 上面全部头部字节 ++ payload)`，**直接复用**
//!   `prism_io::sha256`（同仓库既有 SHA-256，不自写第二份）。
//!
//! **长度前缀 + 校验和** 一起给出两条硬性质：
//!
//! 1. **粘包/拆包无关**：读端先读定长头拿 `payload_len`，再精确读 `payload_len`
//!    字节与 32 字节校验和 ⇒ 与底层 `Read` 的分块边界无关（管道 / socket 都一样）；
//! 2. **篡改可检**：改任意一个字节（含头部的 `kind` / `payload_len`）都会让
//!    校验和不符或结构校验失败 ⇒ 必报错，绝不"猜一个值继续跑"（C20 禁静默容错）。
//!
//! 第 2 条不是"测出来的"，是**覆盖性**的：校验和覆盖 `magic..payload` 的每一个
//! 字节，所以任何单字节改动都落在校验和的作用域内；测试
//! `tamper_is_detected_at_every_byte_position` 逐位置翻转一位穷举钉住这一点。
//!
//! # 传输实现（诚实边界）
//!
//! | 变体 | 状态 | 说明 |
//! |---|---|---|
//! | [`InProcessTransport`] | **已实装** | 同进程内跑 [`crate::farm::worker`]，零 OS 进程开销，供单测与回退路径 |
//! | [`LocalProcessTransport`] | **已实装** | **本地子进程优先**：`spawn` 自身 `farm-worker` 子命令，走 stdin/stdout 管道跑同一套帧协议 |
//! | [`TcpTransport`] | **仅接口** | 跨机 TCP **当前版本未实装**，所有方法返回 [`FarmError::TransportUnavailable`]（如实声明，不做假成功） |
//!
//! 为什么"本地子进程"是优先路径：农场要验的是**跨进程**语义（序列化、校验、
//! 断点、进程边界），而本地子进程用**同一条**线格式、**同一套** worker 代码、
//! 真实 OS 进程边界覆盖它，无需网络即可回归。TCP 只是把同一条帧流塞进 socket，
//! 属于传输介质替换而非协议变更——所以当前版本先钉协议，再留接口。
//!
//! # 为什么 worker 只回"逐帧载荷"
//!
//! [`FramePayloadShard`] 只装**逐帧**载荷（每帧一段字节），**没有**"块内汇总"
//! 字段——与 [`crate::farm::merge::FrameShard`] 同一条纪律：块内预归约会改变
//! 结合序，让浮点结果随切分漂移。主节点按**帧号**（数据内容）归位，再走唯一
//! 一次归约（或直接拼字节流算 MD5）。

use crate::farm::evidence::manifest_digest;
use crate::farm::task::{ConfigHash, FrameTask, ResourceNeed};
use crate::farm::worker::{run_task_with_cost, FrameCache};
use crate::farm::FarmError;
use crate::incremental::FrameRange;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/// 帧魔数（宪章 C8：非本格式立刻拒收）。
pub const FRAME_MAGIC: [u8; 4] = *b"PFRM";

/// 线格式版本（C13：不兼容变更必须递增，旧对端拒收而非误解释）。
pub const TRANSPORT_VERSION: u16 = 1;

/// 定长头字节数：`magic(4) + version(2) + kind(1) + flags(1) + payload_len(4)`。
pub const HEADER_BYTES: usize = 12;

/// 校验和字节数（SHA-256）。
pub const CHECKSUM_BYTES: usize = 32;

/// 单帧载荷上限（64 MiB）——超长先拒，不按声明值预分配（防呆）。
pub const MAX_PAYLOAD_BYTES: u32 = 64 * 1024 * 1024;

/// 单分片帧数上限（防畸形 `count` 触发巨量分配）。
pub const MAX_FRAMES_PER_SHARD: u64 = 1 << 24;

/// 校验和域分隔标签（与 `signature.rs` / `distribution.rs` 同思路：防跨域拼接歧义）。
const CHECKSUM_TAG: &[u8; 16] = b"PRISM-FARM-FRAME";

/// worker 可执行文件名（本地子进程优先路径）。
#[cfg(windows)]
pub const WORKER_BIN_NAME: &str = "farm-worker.exe";
/// worker 可执行文件名（本地子进程优先路径）。
#[cfg(not(windows))]
pub const WORKER_BIN_NAME: &str = "farm-worker";

/// `decide_task` 定长载荷字节数：`start(8)+end(8)+cfg(8)+prio(4)+cpu(4)+mem(4)+vram(4)`。
const TASK_BYTES: usize = 40;

/// 传输实现种类（日志与状态输出用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// 同进程内直接跑 worker（无 OS 进程边界）。
    InProcess,
    /// 本地子进程（`spawn` 自身 `farm-worker`）。
    LocalProcess,
    /// 跨机 TCP（**当前版本未实装**，仅接口）。
    Tcp,
}

// ---------------------------------------------------------------------------
// 帧类型
// ---------------------------------------------------------------------------

/// 帧类型（线格式 `kind` 字节）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// 握手：worker → 主节点，载荷 = 节点 id 的 UTF-8 字节。
    Hello = 1,
    /// 任务：主节点 → worker（[`encode_task`] 定长 40 字节）。
    Task = 2,
    /// 分片产物：worker → 主节点（[`encode_shard`]）。
    Shard = 3,
    /// 收工：主节点 → worker（载荷可为空），worker 见之即退。
    Done = 4,
    /// 错误：worker → 主节点，载荷 = 三段式文本（UTF-8）。
    Error = 5,
}

impl FrameKind {
    /// 线格式字节值。
    pub fn code(self) -> u8 {
        self as u8
    }

    /// 由线格式字节还原（未知字节 → `None`，由调用方报畸形）。
    pub fn from_code(code: u8) -> Option<FrameKind> {
        match code {
            1 => Some(FrameKind::Hello),
            2 => Some(FrameKind::Task),
            3 => Some(FrameKind::Shard),
            4 => Some(FrameKind::Done),
            5 => Some(FrameKind::Error),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// 校验和
// ---------------------------------------------------------------------------

/// 一帧的 SHA-256 校验和：`sha256(域标签 ++ 头部全字节 ++ payload)`。
///
/// `header` 必须是**已编码的 12 字节定长头**（`magic..payload_len`）；把头部也算
/// 进去，是为了让 `kind` / `payload_len` 的篡改同样暴露（否则改 `payload_len`
/// 就能诱导读端错位解析）。
pub fn checksum_of(header: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut buf: Vec<u8> = Vec::with_capacity(CHECKSUM_TAG.len() + header.len() + payload.len());
    buf.extend_from_slice(CHECKSUM_TAG);
    buf.extend_from_slice(header);
    buf.extend_from_slice(payload);
    prism_io::sha256(&buf)
}

/// 校验和的十六进制文本（32 字节 → 64 字符，日志/报错用）。
pub fn checksum_hex(sum: &[u8; 32]) -> String {
    let mut out: String = String::with_capacity(64);
    for b in sum {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ---------------------------------------------------------------------------
// 编解码：单帧
// ---------------------------------------------------------------------------

/// 编码一帧：`header ++ payload ++ checksum`。
///
/// # Panics
/// `payload.len() > MAX_PAYLOAD_BYTES as usize` 时 panic——这是**编造**超大帧的
/// 内部编程错误（合法路径上 payload 由本模块自己产出，必远小于上限）；读端对
/// 外部输入一律走 [`decode_frame`] 的结构化错误，不 panic。
pub fn encode_frame(kind: FrameKind, payload: &[u8]) -> Vec<u8> {
    let len: u32 = u32::try_from(payload.len())
        .expect("farm::transport::encode_frame：单帧载荷超过 u32（内部错误，不应发生）");
    assert!(
        len <= MAX_PAYLOAD_BYTES,
        "farm::transport::encode_frame：单帧载荷 {len} 字节超过上限 {MAX_PAYLOAD_BYTES}（内部错误）"
    );
    let mut out: Vec<u8> = Vec::with_capacity(HEADER_BYTES + payload.len() + CHECKSUM_BYTES);
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&TRANSPORT_VERSION.to_be_bytes());
    out.push(kind.code());
    out.push(0); // flags：当前版本保留，必须为 0
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    let sum: [u8; 32] = checksum_of(&out[..HEADER_BYTES], payload);
    out.extend_from_slice(&sum);
    out
}

/// 解码**恰好一帧**（不接受尾随字节，便于"逐位置篡改"穷举测试）。
///
/// # Errors
/// - 长度不足 / 魔数不符 / 版本不符 / `kind` 未知 / `payload_len` 超上限 /
///   总长与声明不符 → [`FarmError::MalformedEnvelope`]；
/// - 校验和不符 → [`FarmError::ChecksumMismatch`]。
pub fn decode_frame(bytes: &[u8]) -> Result<(FrameKind, Vec<u8>), FarmError> {
    let location: &str = "farm::transport::decode_frame";
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!(
                "帧过短：{} 字节，不足最小帧长 {} 字节（头 {HEADER_BYTES} + 校验和 {CHECKSUM_BYTES}）",
                bytes.len(),
                HEADER_BYTES + CHECKSUM_BYTES
            ),
        });
    }
    if bytes[0..4] != FRAME_MAGIC {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("魔数不符：期望 {:?}，实际 {:?}", FRAME_MAGIC, &bytes[0..4]),
        });
    }
    let version: u16 = u16::from_be_bytes([bytes[4], bytes[5]]);
    if version != TRANSPORT_VERSION {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!(
                "线格式版本不符：对端 {version}，本端 {TRANSPORT_VERSION}（C13：不误解释旧格式）"
            ),
        });
    }
    let Some(kind) = FrameKind::from_code(bytes[6]) else {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("未知帧类型字节 {}", bytes[6]),
        });
    };
    if bytes[7] != 0 {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("保留 flags 必须为 0，实际 {}", bytes[7]),
        });
    }
    let len: u32 = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if len > MAX_PAYLOAD_BYTES {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("载荷长度 {len} 超过上限 {MAX_PAYLOAD_BYTES}"),
        });
    }
    let total: usize = HEADER_BYTES + len as usize + CHECKSUM_BYTES;
    if bytes.len() != total {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!(
                "帧长与声明不符：声明载荷 {len} 字节（总长应为 {total}），实际 {} 字节",
                bytes.len()
            ),
        });
    }
    let payload: &[u8] = &bytes[HEADER_BYTES..HEADER_BYTES + len as usize];
    let expected: [u8; 32] = checksum_of(&bytes[..HEADER_BYTES], payload);
    let actual: [u8; 32] = {
        let mut sum = [0u8; CHECKSUM_BYTES];
        sum.copy_from_slice(&bytes[HEADER_BYTES + len as usize..total]);
        sum
    };
    if expected != actual {
        return Err(FarmError::ChecksumMismatch {
            location: location.to_string(),
            expected: checksum_hex(&expected),
            actual: checksum_hex(&actual),
        });
    }
    Ok((kind, payload.to_vec()))
}

// ---------------------------------------------------------------------------
// 编解码：流（长度前缀的读端）
// ---------------------------------------------------------------------------

/// 从流里读**恰好** `buf.len()` 字节；`clean_eof` 允许"一个字节都没读到"时返回
/// `Ok(false)`（流正常结束），读到一半则报截断畸形。
fn read_exact_or_eof<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
    location: &str,
) -> Result<bool, FarmError> {
    let mut filled: usize = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => {
                if filled == 0 {
                    return Ok(false);
                }
                return Err(FarmError::MalformedEnvelope {
                    location: location.to_string(),
                    detail: format!("流被截断：需要 {} 字节，只读到 {filled} 字节", buf.len()),
                });
            }
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(FarmError::MalformedEnvelope {
                    location: location.to_string(),
                    detail: format!("读流失败：{e}"),
                });
            }
        }
    }
    Ok(true)
}

/// 读满 `buf.len()` 字节，`EOF` 即报截断（用于头之后的 payload / 校验和）。
fn read_full<R: Read>(reader: &mut R, buf: &mut [u8], location: &str) -> Result<(), FarmError> {
    if read_exact_or_eof(reader, buf, location)? {
        Ok(())
    } else {
        Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("流提前结束：需要 {} 字节，读到 0 字节", buf.len()),
        })
    }
}

/// 从流里读一帧（长度前缀 ⇒ 与底层分块边界无关）。
///
/// 返回 `Ok(None)` = 流在**帧边界**处干净结束（对端正常关闭，不是错误）。
///
/// # Errors
/// 同 [`decode_frame`]，另加"流被截断"（读到半帧）。
pub fn read_frame<R: Read>(reader: &mut R) -> Result<Option<(FrameKind, Vec<u8>)>, FarmError> {
    let location: &str = "farm::transport::read_frame";
    let mut header = [0u8; HEADER_BYTES];
    if !read_exact_or_eof(reader, &mut header, location)? {
        return Ok(None);
    }
    // 头完整 ⇒ 后续字段任何缺失都是截断（不是干净结束）。
    let len: u32 = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    if len > MAX_PAYLOAD_BYTES {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("载荷长度 {len} 超过上限 {MAX_PAYLOAD_BYTES}（拒绝按声明值分配）"),
        });
    }
    let mut payload: Vec<u8> = vec![0u8; len as usize];
    read_full(reader, &mut payload, location)?;
    let mut sum = [0u8; CHECKSUM_BYTES];
    read_full(reader, &mut sum, location)?;
    // 复用 decode_frame 的**同一套**结构校验 + 校验和校验（单点实现）。
    let mut whole: Vec<u8> = Vec::with_capacity(HEADER_BYTES + payload.len() + CHECKSUM_BYTES);
    whole.extend_from_slice(&header);
    whole.extend_from_slice(&payload);
    whole.extend_from_slice(&sum);
    decode_frame(&whole).map(Some)
}

/// 写一帧并 `flush`（管道对端必须能立刻看到，否则 lock-step 会互等）。
pub fn write_frame<W: Write>(writer: &mut W, kind: FrameKind, payload: &[u8]) -> Result<(), FarmError> {
    let bytes: Vec<u8> = encode_frame(kind, payload);
    writer.write_all(&bytes).map_err(|e| FarmError::TransportUnavailable {
        location: "farm::transport::write_frame".to_string(),
        detail: format!("写流失败（对端可能已退出）：{e}"),
    })?;
    writer.flush().map_err(|e| FarmError::TransportUnavailable {
        location: "farm::transport::write_frame".to_string(),
        detail: format!("flush 失败：{e}"),
    })
}

// ---------------------------------------------------------------------------
// 编解码：任务 / 分片载荷
// ---------------------------------------------------------------------------

/// 带边界检查的只读游标（解码用；越界一律结构化报错）。
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    location: &'static str,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], location: &'static str) -> Cursor<'a> {
        Cursor { data, pos: 0, location }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FarmError> {
        let end: usize = self.pos.saturating_add(n);
        if end > self.data.len() {
            return Err(FarmError::MalformedEnvelope {
                location: self.location.to_string(),
                detail: format!(
                    "载荷越界：偏移 {} 起需 {n} 字节，总长仅 {} 字节",
                    self.pos,
                    self.data.len()
                ),
            });
        }
        let out: &'a [u8] = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16, FarmError> {
        let b: &[u8] = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, FarmError> {
        let b: &[u8] = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, FarmError> {
        let b: &[u8] = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

/// 编码任务为定长 40 字节（大端；字段序与 [`FrameTask`] 一致）。
pub fn encode_task(task: &FrameTask) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(TASK_BYTES);
    out.extend_from_slice(&task.frames.start.to_be_bytes());
    out.extend_from_slice(&task.frames.end.to_be_bytes());
    out.extend_from_slice(&task.config_hash.value().to_be_bytes());
    out.extend_from_slice(&task.priority.to_be_bytes());
    out.extend_from_slice(&task.resources.cpu_cores.to_be_bytes());
    out.extend_from_slice(&task.resources.memory_mb.to_be_bytes());
    out.extend_from_slice(&task.resources.vram_mb.to_be_bytes());
    debug_assert_eq!(out.len(), TASK_BYTES);
    out
}

/// 解码任务（定长 40 字节；长度不符即畸形）。
pub fn decode_task(payload: &[u8]) -> Result<FrameTask, FarmError> {
    let location: &str = "farm::transport::decode_task";
    if payload.len() != TASK_BYTES {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("任务载荷应为 {TASK_BYTES} 字节，实际 {} 字节", payload.len()),
        });
    }
    let mut cur: Cursor<'_> = Cursor::new(payload, location);
    let start: u64 = cur.u64()?;
    let end: u64 = cur.u64()?;
    let cfg: u64 = cur.u64()?;
    let priority: u32 = cur.u32()?;
    let cpu: u32 = cur.u32()?;
    let mem: u32 = cur.u32()?;
    let vram: u32 = cur.u32()?;
    if end < start {
        return Err(FarmError::InvalidFrameRange { location: location.to_string(), start, end });
    }
    Ok(FrameTask {
        frames: FrameRange { start, end },
        config_hash: ConfigHash::from_raw(cfg),
        priority,
        resources: ResourceNeed { cpu_cores: cpu, memory_mb: mem, vram_mb: vram },
    })
}

/// 一个 worker 交回的分片：**逐帧载荷**（无块内汇总字段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePayloadShard {
    /// 产出节点 id（内容派生，日志/追踪用）。
    pub node_id: String,
    /// 覆盖的帧区间。
    pub frames: FrameRange,
    /// 配置哈希（manifest / 缓存键用）。
    pub config_hash: ConfigHash,
    /// 逐帧载荷，长度必须等于 `frames.len()`。
    pub payloads: Vec<Vec<u8>>,
}

impl FramePayloadShard {
    /// 构造（不校验长度——校验在解码与合并点做）。
    pub fn new(
        node_id: impl Into<String>,
        frames: FrameRange,
        config_hash: ConfigHash,
        payloads: Vec<Vec<u8>>,
    ) -> FramePayloadShard {
        FramePayloadShard { node_id: node_id.into(), frames, config_hash, payloads }
    }

    /// 逐帧载荷条数。
    pub fn len(&self) -> usize {
        self.payloads.len()
    }

    /// 是否无帧。
    pub fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    /// 逐帧齐备（条数 == 帧区间长度）。
    pub fn is_complete(&self) -> bool {
        self.payloads.len() as u64 == self.frames.len()
    }

    /// 该分片的 MD5（按帧号顺序拼字节流；与单机顺序渲染的判据同一算法）。
    pub fn digest(&self) -> String {
        manifest_digest(&self.payloads)
    }
}

/// 编码分片：`start(8) end(8) cfg(8) count(4) node_len(2) node count×(len(4)+bytes)`。
pub fn encode_shard(shard: &FramePayloadShard) -> Vec<u8> {
    let node: &[u8] = shard.node_id.as_bytes();
    let count: u32 = u32::try_from(shard.payloads.len())
        .expect("farm::transport::encode_shard：单分片帧数超过 u32（内部错误，不应发生）");
    let mut out: Vec<u8> = Vec::with_capacity(30 + node.len() + shard.payloads.len() * 36);
    out.extend_from_slice(&shard.frames.start.to_be_bytes());
    out.extend_from_slice(&shard.frames.end.to_be_bytes());
    out.extend_from_slice(&shard.config_hash.value().to_be_bytes());
    out.extend_from_slice(&count.to_be_bytes());
    out.extend_from_slice(&u16::try_from(node.len()).unwrap_or(u16::MAX).to_be_bytes());
    out.extend_from_slice(node);
    for p in &shard.payloads {
        out.extend_from_slice(&u32::try_from(p.len()).unwrap_or(u32::MAX).to_be_bytes());
        out.extend_from_slice(p);
    }
    out
}

/// 解码分片（逐帧齐备性在这里就校验：条数 ≠ 帧区间长度即畸形）。
pub fn decode_shard(payload: &[u8]) -> Result<FramePayloadShard, FarmError> {
    let location: &str = "farm::transport::decode_shard";
    let mut cur: Cursor<'_> = Cursor::new(payload, location);
    let start: u64 = cur.u64()?;
    let end: u64 = cur.u64()?;
    let cfg: u64 = cur.u64()?;
    let count: u32 = cur.u32()?;
    let node_len: u16 = cur.u16()?;
    let node: String = String::from_utf8_lossy(cur.take(node_len as usize)?).into_owned();
    if end < start {
        return Err(FarmError::InvalidFrameRange { location: location.to_string(), start, end });
    }
    let frames: FrameRange = FrameRange { start, end };
    let declared: u64 = frames.len();
    if declared > MAX_FRAMES_PER_SHARD {
        return Err(FarmError::MalformedEnvelope {
            location: location.to_string(),
            detail: format!("分片声明 {declared} 帧，超过上限 {MAX_FRAMES_PER_SHARD}"),
        });
    }
    if count as u64 != declared {
        return Err(FarmError::ShardLengthMismatch {
            location: location.to_string(),
            start,
            end,
            declared,
            values: count as usize,
        });
    }
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(count as usize);
    for i in 0..count {
        let len: u32 = cur.u32()?;
        // 剩余字节不够 ⇒ 结构化报错（不是 panic）。
        if cur.remaining() < len as usize {
            return Err(FarmError::MalformedEnvelope {
                location: location.to_string(),
                detail: format!(
                    "第 {i} 帧载荷越界：需 {len} 字节，剩余 {} 字节",
                    cur.remaining()
                ),
            });
        }
        payloads.push(cur.take(len as usize)?.to_vec());
    }
    Ok(FramePayloadShard {
        node_id: node,
        frames,
        config_hash: ConfigHash::from_raw(cfg),
        payloads,
    })
}

// ---------------------------------------------------------------------------
// 传输抽象
// ---------------------------------------------------------------------------

/// 传输抽象：把任务送下去、把分片收回来。
///
/// 刻意是**lock-step**（发一个任务收一个分片）：既有背压、又不会因为一方写满
/// 管道而对死（两侧都是小载荷，但 lock-step 让"多大载荷都安全"这条成立）。
pub trait ShardTransport {
    /// 节点 id（内容派生）。
    fn node_id(&self) -> &str;

    /// 传输种类。
    fn kind(&self) -> TransportKind;

    /// 下发一个任务。
    fn send_task(&mut self, task: &FrameTask) -> Result<(), FarmError>;

    /// 收回一个分片（对应上一次 [`ShardTransport::send_task`]）。
    fn recv_shard(&mut self) -> Result<FramePayloadShard, FarmError>;

    /// 收工（释放对端资源）。
    fn shutdown(&mut self) -> Result<(), FarmError>;
}

/// 同进程内传输：直接把 [`crate::farm::worker`] 跑在调用线程上。
///
/// 用途：单测（无 OS 进程、毫秒级）与"子进程不可用"时的回退路径。它跑的是
/// **同一套** worker 逻辑（[`run_task_with_cost`]），所以结果与子进程路径逐位相同。
#[derive(Debug)]
pub struct InProcessTransport {
    node_id: String,
    cache: FrameCache,
    work_iters: u32,
    pending: Option<FrameTask>,
    finished: bool,
}

impl InProcessTransport {
    /// 以节点 id 构造（空缓存、零渲染成本）。
    pub fn new(node_id: impl Into<String>) -> InProcessTransport {
        InProcessTransport {
            node_id: node_id.into(),
            cache: FrameCache::new(),
            work_iters: 0,
            pending: None,
            finished: false,
        }
    }

    /// 设定"渲染成本"（纯空转迭代数，**不影响输出**，见 [`crate::farm::worker`]）。
    pub fn with_cost(mut self, work_iters: u32) -> InProcessTransport {
        self.work_iters = work_iters;
        self
    }

    /// 复用给定缓存（断点续算测试用）。
    pub fn with_cache(mut self, cache: FrameCache) -> InProcessTransport {
        self.cache = cache;
        self
    }

    /// 借用当前缓存。
    pub fn cache(&self) -> &FrameCache {
        &self.cache
    }
}

impl ShardTransport for InProcessTransport {
    fn node_id(&self) -> &str {
        &self.node_id
    }

    fn kind(&self) -> TransportKind {
        TransportKind::InProcess
    }

    fn send_task(&mut self, task: &FrameTask) -> Result<(), FarmError> {
        self.pending = Some(*task);
        self.finished = false;
        Ok(())
    }

    fn recv_shard(&mut self) -> Result<FramePayloadShard, FarmError> {
        let Some(task) = self.pending.take() else {
            return Err(FarmError::MalformedEnvelope {
                location: "farm::transport::InProcessTransport::recv_shard".to_string(),
                detail: "没有待收的分片：必须先 send_task 再 recv_shard（lock-step 契约）"
                    .to_string(),
            });
        };
        let (shard, _stats) = run_task_with_cost(&task, &mut self.cache, self.work_iters);
        self.finished = true;
        Ok(FramePayloadShard::new(&self.node_id, shard.frames, shard.config_hash, shard.payloads))
    }

    fn shutdown(&mut self) -> Result<(), FarmError> {
        self.finished = true;
        Ok(())
    }
}

/// 本地子进程传输：`spawn` 自身 `farm-worker` 子命令，走 stdin/stdout 管道。
///
/// 这是"**本地子进程优先**"的落地：协议、编解码、校验和、worker 逻辑都与跨机
/// 路径同一份代码，只把 socket 换成管道。
#[derive(Debug)]
pub struct LocalProcessTransport {
    node_id: String,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<std::io::BufReader<ChildStdout>>,
    cache_path: Option<PathBuf>,
    finished: bool,
}

impl LocalProcessTransport {
    /// 启动一个 worker 子进程（`exe` 为 `farm-worker` 可执行文件路径）。
    ///
    /// 首个动作是收 `Hello` 帧——拿不到就说明起不来，立刻报错（不让半死进程
    /// 在后续重试中制造"神秘的超时"）。
    pub fn spawn(exe: &Path, cache_path: Option<PathBuf>) -> Result<LocalProcessTransport, FarmError> {
        let location: &str = "farm::transport::LocalProcessTransport::spawn";
        let mut command: Command = Command::new(exe);
        command.arg("--stdio");
        if let Some(p) = &cache_path {
            command.arg("--cache").arg(p);
        }
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        let mut child: Child = command.spawn().map_err(|e| FarmError::TransportUnavailable {
            location: location.to_string(),
            detail: format!("启动 worker 子进程失败（{}）：{e}", exe.display()),
        })?;
        let stdin: ChildStdin = child.stdin.take().ok_or_else(|| FarmError::TransportUnavailable {
            location: location.to_string(),
            detail: "worker 子进程未提供 stdin 管道".to_string(),
        })?;
        let stdout: ChildStdout = child.stdout.take().ok_or_else(|| {
            FarmError::TransportUnavailable {
                location: location.to_string(),
                detail: "worker 子进程未提供 stdout 管道".to_string(),
            }
        })?;
        let mut reader = std::io::BufReader::new(stdout);
        let node_id: String = match read_frame(&mut reader)? {
            Some((FrameKind::Hello, payload)) => String::from_utf8_lossy(&payload).into_owned(),
            Some((kind, _)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(FarmError::MalformedEnvelope {
                    location: location.to_string(),
                    detail: format!("worker 首帧必须是 Hello，实际是 {kind:?}"),
                });
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(FarmError::TransportUnavailable {
                    location: location.to_string(),
                    detail: "worker 子进程未握手就关闭了 stdout".to_string(),
                });
            }
        };
        Ok(LocalProcessTransport {
            node_id,
            child,
            stdin: Some(stdin),
            stdout: Some(reader),
            cache_path,
            finished: false,
        })
    }

    /// 子进程 pid（日志用）。
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// 子进程使用的缓存文件路径（若启用断点续算）。
    pub fn cache_path(&self) -> Option<&Path> {
        self.cache_path.as_deref()
    }
}

impl ShardTransport for LocalProcessTransport {
    fn node_id(&self) -> &str {
        &self.node_id
    }

    fn kind(&self) -> TransportKind {
        TransportKind::LocalProcess
    }

    fn send_task(&mut self, task: &FrameTask) -> Result<(), FarmError> {
        let payload: Vec<u8> = encode_task(task);
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(FarmError::TransportUnavailable {
                location: "farm::transport::LocalProcessTransport::send_task".to_string(),
                detail: "子进程 stdin 已关闭（shutdown 之后不得再下发任务）".to_string(),
            });
        };
        write_frame(stdin, FrameKind::Task, &payload)
    }

    fn recv_shard(&mut self) -> Result<FramePayloadShard, FarmError> {
        let location: &str = "farm::transport::LocalProcessTransport::recv_shard";
        let Some(reader) = self.stdout.as_mut() else {
            return Err(FarmError::TransportUnavailable {
                location: location.to_string(),
                detail: "子进程 stdout 已关闭".to_string(),
            });
        };
        loop {
            match read_frame(reader)? {
                Some((FrameKind::Shard, payload)) => return decode_shard(&payload),
                Some((FrameKind::Error, payload)) => {
                    return Err(FarmError::MalformedEnvelope {
                        location: location.to_string(),
                        detail: format!(
                            "worker 报错：{}",
                            String::from_utf8_lossy(&payload)
                        ),
                    });
                }
                Some((FrameKind::Hello, _)) => continue,
                Some((kind, _)) => {
                    return Err(FarmError::MalformedEnvelope {
                        location: location.to_string(),
                        detail: format!("期望 Shard，实际收到 {kind:?}"),
                    });
                }
                None => {
                    return Err(FarmError::TransportUnavailable {
                        location: location.to_string(),
                        detail: "worker 子进程提前关闭 stdout（未交回分片）".to_string(),
                    });
                }
            }
        }
    }

    fn shutdown(&mut self) -> Result<(), FarmError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let location: &str = "farm::transport::LocalProcessTransport::shutdown";
        if let Some(stdin) = self.stdin.as_mut() {
            // 送 Done 帧；对端退出后管道写失败是可接受的（可能已经自己退了）。
            let _ = write_frame(stdin, FrameKind::Done, b"");
        }
        self.stdin = None; // 关管道 ⇒ 对端 read 到干净 EOF 也会退
        self.stdout = None;
        let status = self.child.wait().map_err(|e| FarmError::TransportUnavailable {
            location: location.to_string(),
            detail: format!("等待 worker 子进程退出失败：{e}"),
        })?;
        if !status.success() {
            return Err(FarmError::TransportUnavailable {
                location: location.to_string(),
                detail: format!("worker 子进程异常退出：{status}"),
            });
        }
        Ok(())
    }
}

impl Drop for LocalProcessTransport {
    fn drop(&mut self) {
        // best-effort：未 shutdown 就被丢弃（测试失败 / 提前返回）也不能留孤儿进程。
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// 跨机 TCP 传输——**当前版本仅接口，未实装**。
///
/// 刻意**不假装成功**：所有方法返回 [`FarmError::TransportUnavailable`]，并在
/// 建议段如实写明"当前版本未实装、本地子进程已覆盖同一帧协议"。这样调用方在
/// 误用时看到的是明确的三段式错误，而不是一个静默挂住的 socket。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpTransport {
    /// 对端地址（`host:port`），仅用于报错信息。
    pub endpoint: String,
}

impl TcpTransport {
    /// 构造（不连接；当前版本不实装连接）。
    pub fn new(endpoint: impl Into<String>) -> TcpTransport {
        TcpTransport { endpoint: endpoint.into() }
    }
}

impl ShardTransport for TcpTransport {
    fn node_id(&self) -> &str {
        &self.endpoint
    }

    fn kind(&self) -> TransportKind {
        TransportKind::Tcp
    }

    fn send_task(&mut self, _task: &FrameTask) -> Result<(), FarmError> {
        Err(FarmError::TransportUnavailable {
            location: format!("farm::transport::TcpTransport::send_task({})", self.endpoint),
            detail: "跨机 TCP 传输当前版本未实装（仅保留接口）".to_string(),
        })
    }

    fn recv_shard(&mut self) -> Result<FramePayloadShard, FarmError> {
        Err(FarmError::TransportUnavailable {
            location: format!("farm::transport::TcpTransport::recv_shard({})", self.endpoint),
            detail: "跨机 TCP 传输当前版本未实装（仅保留接口）".to_string(),
        })
    }

    fn shutdown(&mut self) -> Result<(), FarmError> {
        // 收工对未连接的传输是幂等空操作（不报错，避免掩盖真正的首个错误）。
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// worker 可执行文件发现 + 批量 spawn
// ---------------------------------------------------------------------------

/// 发现 `farm-worker` 可执行文件。
///
/// 顺序：
/// 1. 环境变量 `PRISM_FARM_WORKER`（显式覆盖，测试与部署用）；
/// 2. 从 `current_exe()` 往上找 `debug` / `release` 目录，再拼 `farm-worker`
///    （`cargo test` 会把 bin 编到 `target/{debug,release}/`，而测试进程在
///    `target/{debug,release}/deps/` 下）。
///
/// 找不到返回 `None`——由调用方决定是回退进程内传输还是报错，**本函数不 panic**。
pub fn discover_worker_exe() -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("PRISM_FARM_WORKER") {
        let candidate: PathBuf = PathBuf::from(raw);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let exe: PathBuf = std::env::current_exe().ok()?;
    let mut dir: &Path = exe.parent()?;
    loop {
        if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
            if name == "debug" || name == "release" {
                let candidate: PathBuf = dir.join(WORKER_BIN_NAME);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        dir = dir.parent()?;
    }
}

/// 启动 `count` 个本地 worker 子进程（`count` 归一为 ≥ 1）。
///
/// 任何一个起不来 ⇒ 返回错误并（由 `Drop`）回收已起的进程，不留孤儿。
pub fn spawn_local_workers(
    count: usize,
    cache_dir: Option<&Path>,
) -> Result<Vec<LocalProcessTransport>, FarmError> {
    let n: usize = count.max(1);
    let exe: PathBuf = discover_worker_exe().ok_or_else(|| FarmError::TransportUnavailable {
        location: "farm::transport::spawn_local_workers".to_string(),
        detail: format!(
            "找不到 {WORKER_BIN_NAME} 可执行文件（先 `cargo build -p prism-render --bin {WORKER_BIN_NAME}`，\
             或设 PRISM_FARM_WORKER 指向它）"
        ),
    })?;
    let mut out: Vec<LocalProcessTransport> = Vec::with_capacity(n);
    for i in 0..n {
        let cache_path: Option<PathBuf> = cache_dir.map(|d| d.join(format!("farm-worker-{i}.cache")));
        out.push(LocalProcessTransport::spawn(&exe, cache_path)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task() -> FrameTask {
        FrameTask {
            frames: FrameRange { start: 10, end: 19 },
            config_hash: ConfigHash::of_bytes(b"wire"),
            priority: 3,
            resources: ResourceNeed { cpu_cores: 2, memory_mb: 512, vram_mb: 256 },
        }
    }

    fn sample_shard() -> FramePayloadShard {
        FramePayloadShard::new(
            "node-7",
            FrameRange { start: 10, end: 12 },
            ConfigHash::of_bytes(b"wire"),
            vec![b"a".to_vec(), b"bb".to_vec(), Vec::new()],
        )
    }

    #[test]
    fn frame_roundtrip_preserves_kind_and_payload() {
        for (kind, payload) in [
            (FrameKind::Hello, b"worker-1".to_vec()),
            (FrameKind::Task, vec![0u8; 40]),
            (FrameKind::Shard, vec![7u8; 1000]),
            (FrameKind::Done, Vec::new()),
            (FrameKind::Error, "三段式错误".as_bytes().to_vec()),
        ] {
            let bytes: Vec<u8> = encode_frame(kind, &payload);
            let (got_kind, got_payload) = decode_frame(&bytes).expect("必须可解码");
            assert_eq!(got_kind, kind);
            assert_eq!(got_payload, payload, "载荷必须逐字节往返");
            // 头部字段自洽。
            assert_eq!(&bytes[0..4], &FRAME_MAGIC);
            assert_eq!(u16::from_be_bytes([bytes[4], bytes[5]]), TRANSPORT_VERSION);
            assert_eq!(bytes[6], kind.code());
            assert_eq!(bytes[7], 0);
            assert_eq!(
                u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                payload.len(),
                "长度前缀必须等于载荷长度"
            );
            // 编码是纯函数：两遍逐字节一致。
            assert_eq!(encode_frame(kind, &payload), bytes);
        }
    }

    #[test]
    fn tamper_is_detected_at_every_byte_position() {
        // 穷举：帧里**每一个字节**翻一位都必须报错（校验和覆盖全帧 + 结构校验）。
        let bytes: Vec<u8> = encode_frame(FrameKind::Shard, b"payload-for-tamper-check");
        assert!(decode_frame(&bytes).is_ok(), "原始帧必须合法");
        for i in 0..bytes.len() {
            let mut broken: Vec<u8> = bytes.clone();
            broken[i] ^= 0x01;
            let res = decode_frame(&broken);
            assert!(
                res.is_err(),
                "第 {i} 字节翻转一位后仍被接受——校验和/结构覆盖有漏洞"
            );
        }
        // 单字节载荷内容被改（语义最直接的篡改）必须落到 ChecksumMismatch。
        let mut tampered: Vec<u8> = encode_frame(FrameKind::Task, &[1u8; 40]);
        let payload_at: usize = HEADER_BYTES + 5;
        tampered[payload_at] ^= 0xFF;
        match decode_frame(&tampered) {
            Err(FarmError::ChecksumMismatch { expected, actual, .. }) => {
                assert_ne!(expected, actual, "报错必须带上期望/实际两个摘要");
                assert_eq!(expected.len(), 64, "SHA-256 十六进制应为 64 字符");
            }
            other => panic!("改一字节必须报 ChecksumMismatch，实际 {other:?}"),
        }
        // 截断（去掉尾部一字节）也必须报错。
        let mut short: Vec<u8> = bytes.clone();
        short.pop();
        assert!(decode_frame(&short).is_err(), "截断必须报错");
        // 追加尾随字节 ⇒ 长度不符（不静默忽略）。
        let mut longer: Vec<u8> = bytes.clone();
        longer.push(0);
        assert!(decode_frame(&longer).is_err(), "尾随字节必须报错（长度不符）");
    }

    #[test]
    fn stream_read_write_is_chunking_agnostic_and_distinguishes_clean_eof() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame(&mut buf, FrameKind::Hello, b"n0").expect("写必须成功");
        write_frame(&mut buf, FrameKind::Task, &encode_task(&sample_task())).expect("写必须成功");
        write_frame(&mut buf, FrameKind::Done, b"").expect("写必须成功");

        // 一次性读完三帧，顺序与内容一致。
        let mut cursor = std::io::Cursor::new(buf.clone());
        let (k1, p1) = read_frame(&mut cursor).expect("读必须成功").expect("首帧存在");
        assert_eq!((k1, p1.as_slice()), (FrameKind::Hello, &b"n0"[..]));
        let (k2, p2) = read_frame(&mut cursor).expect("读必须成功").expect("次帧存在");
        assert_eq!(k2, FrameKind::Task);
        assert_eq!(decode_task(&p2).expect("任务应可解码").frames.start, 10);
        let (k3, p3) = read_frame(&mut cursor).expect("读必须成功").expect("三帧存在");
        assert_eq!((k3, p3.len()), (FrameKind::Done, 0));
        // 干净 EOF：Ok(None)，不是错误。
        assert!(read_frame(&mut cursor).expect("干净 EOF 不得报错").is_none());

        // 分块喂（每 3 字节一块）也必须读出同样的三帧——长度前缀的意义所在。
        struct Chunky {
            data: Vec<u8>,
            pos: usize,
            step: usize,
        }
        impl Read for Chunky {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if self.pos >= self.data.len() {
                    return Ok(0);
                }
                let n = self.step.min(self.data.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
                self.pos += n;
                Ok(n)
            }
        }
        let mut chunky = Chunky { data: buf.clone(), pos: 0, step: 3 };
        let mut kinds: Vec<FrameKind> = Vec::new();
        while let Some((k, _)) = read_frame(&mut chunky).expect("分块读不得报错") {
            kinds.push(k);
        }
        assert_eq!(kinds, vec![FrameKind::Hello, FrameKind::Task, FrameKind::Done]);

        // 半帧（截断流）必须报错，不能当成干净 EOF。
        let mut half = std::io::Cursor::new(buf[..buf.len() - 1].to_vec());
        let mut last: Option<FarmError> = None;
        loop {
            match read_frame(&mut half) {
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(e) => {
                    last = Some(e);
                    break;
                }
            }
        }
        assert!(last.is_some(), "截断流必须报错");
    }

    #[test]
    fn task_and_shard_wire_codecs_roundtrip() {
        let task: FrameTask = sample_task();
        let bytes: Vec<u8> = encode_task(&task);
        assert_eq!(bytes.len(), TASK_BYTES, "任务载荷必须是定长 40 字节");
        assert_eq!(decode_task(&bytes).expect("往返一致"), task);
        // 长度不符即畸形（不做宽容解析）。
        assert!(decode_task(&bytes[..39]).is_err());
        assert!(decode_task(&[0u8; 41]).is_err());
        // 反序区间必须被拒。
        let mut bad: Vec<u8> = bytes.clone();
        bad[0..8].copy_from_slice(&20u64.to_be_bytes());
        bad[8..16].copy_from_slice(&10u64.to_be_bytes());
        assert!(decode_task(&bad).is_err(), "end < start 必须被拒");

        let shard: FramePayloadShard = sample_shard();
        let sb: Vec<u8> = encode_shard(&shard);
        assert_eq!(decode_shard(&sb).expect("往返一致"), shard);
        assert!(shard.is_complete(), "3 帧 3 载荷应齐备");
        // 条数与声明帧数不符必须被拒（P5006 语义在线格式层也成立）。
        let wrong: FramePayloadShard = FramePayloadShard::new(
            "n",
            FrameRange { start: 10, end: 12 },
            ConfigHash::of_bytes(b"wire"),
            vec![b"a".to_vec()],
        );
        match decode_shard(&encode_shard(&wrong)) {
            Err(FarmError::ShardLengthMismatch { declared, values, .. }) => {
                assert_eq!((declared, values), (3, 1));
            }
            other => panic!("条数不符必须报 P5006，实际 {other:?}"),
        }
        // 载荷区被截断必须被拒（游标越界 ⇒ 结构化错误）。
        assert!(decode_shard(&sb[..sb.len() - 1]).is_err(), "截断分片必须报错");
    }

    #[test]
    fn in_process_transport_is_lock_step_and_matches_worker_output() {
        let mut transport: InProcessTransport = InProcessTransport::new("in-proc");
        assert_eq!(transport.kind(), TransportKind::InProcess);
        assert_eq!(transport.node_id(), "in-proc");
        // 未下发就收 ⇒ 结构化错误（不是 panic）。
        assert!(transport.recv_shard().is_err());

        let task: FrameTask = sample_task();
        transport.send_task(&task).expect("下发应成功");
        let shard: FramePayloadShard = transport.recv_shard().expect("收回应成功");
        assert_eq!(shard.node_id, "in-proc");
        assert_eq!(shard.frames, task.frames);
        assert!(shard.is_complete(), "子进程/进程内路径都必须逐帧齐备");
        // 与直接调 worker 逐字节相同（两条路径同一份渲染代码）。
        let mut cache: FrameCache = FrameCache::new();
        let (direct, _stats) = run_task_with_cost(&task, &mut cache, 0);
        assert_eq!(shard.payloads, direct.payloads);
        transport.shutdown().expect("收工应成功");
    }

    #[test]
    fn tcp_transport_declares_unimplemented_instead_of_faking_success() {
        let mut tcp: TcpTransport = TcpTransport::new("127.0.0.1:9000");
        assert_eq!(tcp.kind(), TransportKind::Tcp);
        assert_eq!(tcp.node_id(), "127.0.0.1:9000");
        match tcp.send_task(&sample_task()) {
            Err(FarmError::TransportUnavailable { location, detail }) => {
                assert!(location.contains("TcpTransport"), "位置必须可定位：{location}");
                assert!(detail.contains("未实装"), "必须如实声明未实装：{detail}");
            }
            other => panic!("TCP 变体必须明确报未实装，实际 {other:?}"),
        }
        assert!(tcp.recv_shard().is_err());
        // 未连接的 shutdown 是幂等空操作。
        assert!(tcp.shutdown().is_ok());
    }

    #[test]
    fn io_errors_surface_as_structured_transport_errors() {
        // 写端固定失败 ⇒ TransportUnavailable（不是 panic，不带病继续）。
        struct BadWriter;
        impl Write for BadWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "注入的断管"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut w = BadWriter;
        let err = write_frame(&mut w, FrameKind::Task, b"x").expect_err("断管必须报错");
        assert!(matches!(err, FarmError::TransportUnavailable { .. }), "实际 {err:?}");
        assert!(err.to_string().contains("建议"), "三段式必须含建议段（C19）");
    }
}
