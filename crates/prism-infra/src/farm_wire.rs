//! 农场线格式：主节点 ⇄ worker 的任务 / 分片编解码（路线图 §2.6 P2）。
//!
//! # 为什么放在 `prism-infra` 而不是 `prism-render`
//!
//! **分层**：`prism-render::farm` 是**语义层**（切分 / 调度 / 归约的确定性与
//! 数学），不该知道任何传输格式；`prism-infra` 是**传输层**（JSON-RPC /
//! HTTP / batch 都在这里），已经自写了 [`crate::jsonrpc::Json`] 值模型。依赖
//! 方向是 `prism-infra → prism-render`（`prism-render` 不依赖 infra），所以
//! 格式适配层放 infra 侧**不会引入循环依赖**。
//!
//! # 复用什么（任务书"若有可复用的任务表达 / 传输优先复用"）
//!
//! - **复用** [`crate::jsonrpc::Json`]：值模型 + 递归下降解析器 + 确定性序列化
//!   （对象保留插入序 ⇒ 同输入逐字节可复现）。不引 `serde`（零第三方依赖铁律）。
//! - **复用** `prism_render::farm` 的类型：[`FrameTask`] / [`FrameShard`] /
//!   [`ConfigHash`] / [`ResourceNeed`] / [`FrameRange`] 全部**按值编解码**，
//!   **不重复定义** farm 侧的领域类型。
//! - **不复用** `crate::batch::SubTask`：它的字段是"配置变体"
//!   （source/out/width/height/fps），与农场"帧号区间 + 配置哈希"不同源；
//!   硬套会把两个语义层搅在一起（D-16：能写清楚比能复用更要紧）。
//!
//! # 线格式（版本化）
//!
//! ```text
//! 任务下行（主节点 → worker）：
//! {"v":1,"kind":"task","id":"cfg=.../frames=0-99","start":0,"end":99,
//!  "config_hash":12345678901234567890,"priority":0,
//!  "cpu_cores":0,"memory_mb":0,"vram_mb":0}
//!
//! 分片上行（worker → 主节点）：
//! {"v":1,"kind":"shard","node":"node-a","start":0,"end":24,
//!  "config_hash":12345678901234567890,"attempts":1,"values":[...]}
//! ```
//!
//! - `v` 是版本号，便于将来演进而不破坏兼容；未知版本 → P5007。
//! - **键序固定**（写入顺序即上表顺序）⇒ 序列化文本逐字节确定，可直接做
//!   manifest 校验与 diff。
//! - 畸形输入一律返回 [`FarmError::MalformedEnvelope`]（P5007，C18 三段式 +
//!   C19 建议段），**不 panic**、不静默补默认值（C20）。
//!
//! # 整数精度（重要边界）
//!
//! [`Json::Num`] 是 `f64`，只有 53 位尾数精度。`ConfigHash` 是 **u64**，超过
//! `2^53` 的哈希值**无法无损往返**。本模块的处理是：编码时把 u64 拆成
//! **两个 32 位十六进制字符串**（`hi` / `lo`），解码时拼回——**无损**，且
//! 仍是纯文本、可读、可 diff。
//! （若直接写 `Num(u64 as f64)`，高位会被静默截断，那是真正的错误。）

use crate::jsonrpc::Json;
use prism_render::farm::{ConfigHash, FarmError, FrameShard, FrameTask, ResourceNeed};
use prism_render::incremental::FrameRange;

/// 线格式版本号（当前唯一版本）。
pub const WIRE_VERSION: i64 = 1;

/// 64 位哈希的十六进制位数（补零到定长，保证可 diff）。
const HEX64: usize = 16;

// ---------------------------------------------------------------------------
// 编码
// ---------------------------------------------------------------------------

/// 任务 → JSON（下行）。
pub fn encode_task(task: &FrameTask) -> Json {
    let (hi, lo) = split_u64(task.config_hash.value());
    Json::object(vec![
        ("v", Json::int(WIRE_VERSION)),
        ("kind", Json::Str("task".to_string())),
        ("id", Json::Str(task.id().to_string())),
        ("start", json_u64(task.frames.start)),
        ("end", json_u64(task.frames.end)),
        ("config_hash_hi", Json::Str(hi)),
        ("config_hash_lo", Json::Str(lo)),
        ("priority", Json::int(i64::from(task.priority))),
        ("cpu_cores", Json::int(i64::from(task.resources.cpu_cores))),
        ("memory_mb", Json::int(i64::from(task.resources.memory_mb))),
        ("vram_mb", Json::int(i64::from(task.resources.vram_mb))),
    ])
}

/// 带帧区间的分片 → JSON（上行，完整线格式）。
pub fn encode_shard<T, F>(
    node_id: &str,
    frames: FrameRange,
    config_hash: ConfigHash,
    attempts: u32,
    values: &[T],
    to_json: F,
) -> Json
where
    F: Fn(&T) -> Json,
{
    let (hi, lo) = split_u64(config_hash.value());
    Json::object(vec![
        ("v", Json::int(WIRE_VERSION)),
        ("kind", Json::Str("shard".to_string())),
        ("node", Json::Str(node_id.to_string())),
        ("start", json_u64(frames.start)),
        ("end", json_u64(frames.end)),
        ("config_hash_hi", Json::Str(hi)),
        ("config_hash_lo", Json::Str(lo)),
        ("attempts", Json::int(i64::from(attempts))),
        ("values", Json::Array(values.iter().map(|v| to_json(v)).collect())),
    ])
}

// ---------------------------------------------------------------------------
// 解码
// ---------------------------------------------------------------------------

/// JSON → 任务（下行）。
///
/// # Errors
/// 字段缺失 / 类型不符 / 版本未知 / 区间非法（`end < start`）→
/// [`FarmError::MalformedEnvelope`]（P5007）或 [`FarmError::InvalidFrameRange`]。
pub fn decode_task(value: &Json) -> Result<FrameTask, FarmError> {
    let loc: &str = "farm_wire::decode_task";
    check_version(value, loc, "task")?;
    let start: u64 = need_u64(value, "start", loc)?;
    let end: u64 = need_u64(value, "end", loc)?;
    let frames: FrameRange = FrameRange { start, end };
    if end < start {
        return Err(FarmError::InvalidFrameRange {
            location: loc.to_string(),
            start,
            end,
        });
    }
    let config_hash: ConfigHash = ConfigHash::from_raw(need_hash(value, loc)?);
    let priority: u32 = need_u32(value, "priority", loc)?;
    let resources: ResourceNeed = ResourceNeed {
        cpu_cores: need_u32(value, "cpu_cores", loc)?,
        memory_mb: need_u32(value, "memory_mb", loc)?,
        vram_mb: need_u32(value, "vram_mb", loc)?,
    };
    Ok(FrameTask { frames, config_hash, priority, resources })
}

/// JSON → 分片（上行）。`from_json` 负责把每个 `Json` 还原成 `T`。
///
/// # Errors
/// 同 [`decode_task`]，另加 `values` 数组元素类型不符（由 `from_json` 的错误
/// 上卷为 P5007）。
pub fn decode_shard<T, F>(value: &Json, from_json: F) -> Result<FrameShard<T>, FarmError>
where
    T: Clone + Send,
    F: Fn(&Json) -> Result<T, String>,
{
    let loc: &str = "farm_wire::decode_shard";
    check_version(value, loc, "shard")?;
    let start: u64 = need_u64(value, "start", loc)?;
    let end: u64 = need_u64(value, "end", loc)?;
    if end < start {
        return Err(FarmError::InvalidFrameRange {
            location: loc.to_string(),
            start,
            end,
        });
    }
    let config_hash: ConfigHash = ConfigHash::from_raw(need_hash(value, loc)?);
    let Some(Json::Array(items)) = value.get("values") else {
        return Err(malformed(loc, "缺 values 数组"));
    };
    let mut values: Vec<T> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let v: T = from_json(item)
            .map_err(|e| malformed(loc, &format!("values[{i}] 无法解析：{e}")))?;
        values.push(v);
    }
    Ok(FrameShard::new(FrameRange { start, end }, values, config_hash))
}

/// 从分片 JSON 取节点 id（用于日志与 manifest；缺失时给 `""`）。
pub fn shard_node_id(value: &Json) -> String {
    value.get("node").and_then(|v| v.as_str()).unwrap_or("").to_string()
}

/// 从分片 JSON 取实际尝试次数（缺失时给 1）。
pub fn shard_attempts(value: &Json) -> u32 {
    value
        .get("attempts")
        .and_then(|v| v.as_f64())
        .and_then(|n| u32::try_from(n as i64).ok())
        .unwrap_or(1)
        .max(1)
}

// ---------------------------------------------------------------------------
// 内部工具
// ---------------------------------------------------------------------------

fn malformed(loc: &str, detail: &str) -> FarmError {
    FarmError::MalformedEnvelope { location: loc.to_string(), detail: detail.to_string() }
}

/// 校验版本号与 `kind`。
fn check_version(value: &Json, loc: &str, expect_kind: &str) -> Result<(), FarmError> {
    match value.get("v") {
        None => return Err(malformed(loc, "缺版本字段 v")),
        Some(Json::Num(n)) if *n == WIRE_VERSION as f64 => {}
        Some(Json::Num(n)) => {
            return Err(malformed(loc, &format!("版本不支持：{n}（本端只支持 {WIRE_VERSION}）")))
        }
        Some(_) => return Err(malformed(loc, "版本字段 v 必须是数值")),
    }
    match value.get("kind").and_then(|v| v.as_str()) {
        Some(k) if k == expect_kind => Ok(()),
        Some(k) => Err(malformed(loc, &format!("kind 期望 {expect_kind}，实得 {k}"))),
        None => Err(malformed(loc, "缺 kind 字段")),
    }
}

/// 取 u64 字段。`Json::Num` 是 f64，故先校验"是整数且在 u64 范围内"。
fn need_u64(value: &Json, key: &str, loc: &str) -> Result<u64, FarmError> {
    let Some(field) = value.get(key) else {
        return Err(malformed(loc, &format!("缺字段 {key}")));
    };
    let Some(n) = field.as_f64() else {
        return Err(malformed(loc, &format!("字段 {key} 必须是数值")));
    };
    if n.fract() != 0.0 || n < 0.0 {
        return Err(malformed(loc, &format!("字段 {key} 必须是 u64（实得 {n}）")));
    }
    // f64 → u64：先转 i64 校验范围（> i64::MAX 的 u64 在 f64 里已失真，
    // 线格式约定帧号用 u64 但实际不超过 i64::MAX；超界显式报错而非静默截断）。
    if n > 9_223_372_036_854_775_808.0_f64 {
        return Err(malformed(loc, &format!("字段 {key} 超出可精确表示范围（{n}）")));
    }
    Ok(n as u64)
}

/// 取 u32 字段。
fn need_u32(value: &Json, key: &str, loc: &str) -> Result<u32, FarmError> {
    let v: u64 = need_u64(value, key, loc)?;
    u32::try_from(v).map_err(|_| malformed(loc, &format!("字段 {key} 超出 u32 范围（{v}）")))
}

/// 取配置哈希：两个 16 位十六进制串拼回 u64（**无损**，见模块文档）。
fn need_hash(value: &Json, loc: &str) -> Result<u64, FarmError> {
    let hi: u64 = need_hex64(value, "config_hash_hi", loc)?;
    let lo: u64 = need_hex64(value, "config_hash_lo", loc)?;
    Ok((hi << 32) | lo)
}

fn need_hex64(value: &Json, key: &str, loc: &str) -> Result<u64, FarmError> {
    let Some(field) = value.get(key) else {
        return Err(malformed(loc, &format!("缺字段 {key}")));
    };
    let Some(s) = field.as_str() else {
        return Err(malformed(loc, &format!("字段 {key} 必须是十六进制字符串")));
    };
    if s.len() != HEX64 {
        return Err(malformed(loc, &format!("字段 {key} 必须恰好 {HEX64} 位十六进制（实得 {}）", s.len())));
    }
    u64::from_str_radix(s, 16)
        .map_err(|e| malformed(loc, &format!("字段 {key} 十六进制解析失败：{e}")))
}

/// u64 → 固定 16 位小写十六进制（不用 `format!`，避免临时 String）。
fn hex64(v: u64) -> String {
    const DIGITS: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', //
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];
    let mut out: String = String::with_capacity(HEX64);
    for i in 0..HEX64 {
        let shift: u32 = u32::try_from(HEX64 - 1 - i).unwrap_or(0) * 4;
        let nibble: usize = usize::try_from((v >> shift) & 0xf).unwrap_or(0);
        out.push(DIGITS[nibble]);
    }
    out
}

/// u64 → (高 32 位十六进制, 低 32 位十六进制)，各 16 位定长。
fn split_u64(v: u64) -> (String, String) {
    (hex64(v >> 32), hex64(v & 0xffff_ffff))
}

/// u64 → `Json::Num`（帧号在 i64 范围内，可无损表示）。
fn json_u64(v: u64) -> Json {
    // u64 → i64 用 try_from：越界给 0（帧号越界在 split 层已报 P5003，
    // 这里只是防御性编码，不会静默改变合法值）。
    Json::int(i64::try_from(v).unwrap_or(i64::MAX))
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render::farm::split_frames;

    fn sample_task() -> FrameTask {
        FrameTask {
            frames: FrameRange { start: 0, end: 99 },
            // 故意用超过 2^53 的哈希（0xdead_beef_cafe_babe ≈ 1.6e19），验证无损往返（f64 精度陷阱）。
            config_hash: ConfigHash::from_raw(0xdead_beef_cafe_babe),
            priority: 7,
            resources: ResourceNeed { cpu_cores: 4, memory_mb: 2048, vram_mb: 8192 },
        }
    }

    #[test]
    fn task_round_trips_without_precision_loss() {
        let task: FrameTask = sample_task();
        let json: Json = encode_task(&task);
        let back: FrameTask = decode_task(&json).expect("应可解码");
        assert_eq!(back, task, "任务编解码必须逐字段相同（含超 2^53 的哈希）");
        // 显式确认哈希没被 f64 截断。
        assert_eq!(back.config_hash.value(), 0xdead_beef_cafe_babe);
    }

    #[test]
    fn task_serialization_is_byte_stable() {
        // 确定性序列化：同输入两次逐字节相同（键序固定）。
        let task: FrameTask = sample_task();
        let a: String = encode_task(&task).to_string().expect("应可序列化");
        let b: String = encode_task(&task).to_string().expect("应可序列化");
        assert_eq!(a, b, "同输入必须逐字节可复现");
        assert!(a.contains("\"kind\":\"task\""), "缺 kind：{a}");
        assert!(a.contains("\"v\":1"), "缺版本：{a}");
    }

    #[test]
    fn shard_round_trips_per_frame_values_verbatim() {
        let cfg: ConfigHash = ConfigHash::from_raw(0x0123_4567_89ab_cdef);
        let values: Vec<u64> = (0..25u64).collect();
        let json: Json = encode_shard(
            "node-a",
            FrameRange { start: 0, end: 24 },
            cfg,
            2,
            &values,
            |v| Json::int(i64::try_from(*v).unwrap_or(0)),
        );
        assert_eq!(shard_node_id(&json), "node-a");
        assert_eq!(shard_attempts(&json), 2);
        let back: FrameShard<u64> =
            decode_shard(&json, |j| j.as_f64().map(|n| n as u64).ok_or_else(|| "非数值".to_string()))
                .expect("应可解码");
        assert_eq!(back.frames, FrameRange { start: 0, end: 24 });
        assert_eq!(back.values, values, "逐帧值必须原样往返");
        assert_eq!(back.config_hash, cfg);
    }

    #[test]
    fn split_ranges_survive_the_wire_round_trip() {
        // 端到端：切分 → 每个分片编码 → 解码 → 合并（帧区间与逐帧值都要对）。
        let n: u64 = 100;
        let parts: Vec<FrameRange> = split_frames(0, n - 1, 7).expect("应可切分");
        let cfg: ConfigHash = ConfigHash::from_raw(0x0f0f_0f0f_0f0f_0f0f);
        let mut shards: Vec<FrameShard<u64>> = Vec::new();
        for p in &parts {
            let values: Vec<u64> = (p.start..=p.end).collect();
            let json: Json =
                encode_shard("n1", *p, cfg, 1, &values, |v| Json::int(i64::from(*v as i32)));
            shards.push(
                decode_shard(&json, |j| {
                    j.as_f64().map(|n| n as u64).ok_or_else(|| "非数值".to_string())
                })
                .expect("应可解码"),
            );
        }
        let merged: u64 = shards
            .iter()
            .flat_map(|s| s.values.iter().copied())
            .fold(0u64, |a, b| a.wrapping_add(b));
        assert_eq!(merged, (0..n).sum::<u64>(), "线格式往返后合帧结果必须不变");
    }

    #[test]
    fn malformed_envelopes_return_structured_errors_without_panic() {
        let loc = "farm_wire::decode_task";
        // 非对象。
        let err: FarmError = decode_task(&Json::Null).expect_err("应报错");
        assert!(err.to_string().contains("P5007"), "非对象应报 P5007：{err}");
        // 缺版本。
        let err: FarmError = decode_task(&Json::object(vec![("kind", Json::Str("task".into()))]))
            .expect_err("应报错");
        assert!(err.to_string().contains("P5007"));
        // 版本不支持。
        let err: FarmError = decode_task(&Json::object(vec![
            ("v", Json::int(99)),
            ("kind", Json::Str("task".into())),
        ]))
        .expect_err("应报错");
        assert!(err.to_string().contains("P5007"), "未知版本应报 P5007：{err}");
        // kind 不符。
        let err: FarmError = decode_task(&Json::object(vec![
            ("v", Json::int(1)),
            ("kind", Json::Str("shard".into())),
        ]))
        .expect_err("应报错");
        assert!(err.to_string().contains("P5007"), "kind 不符应报 P5007：{err}");
        // 缺 start。
        let err: FarmError = decode_task(&Json::object(vec![
            ("v", Json::int(1)),
            ("kind", Json::Str("task".into())),
        ]))
        .expect_err("应报错");
        assert!(err.to_string().contains("P5007"));
        // end < start（负区间）⇒ P5001。
        let err: FarmError = decode_task(&Json::object(vec![
            ("v", Json::int(1)),
            ("kind", Json::Str("task".into())),
            ("start", Json::int(10)),
            ("end", Json::int(5)),
            ("config_hash_hi", Json::Str("0000000000000000".into())),
            ("config_hash_lo", Json::Str("0000000000000000".into())),
            ("priority", Json::int(0)),
            ("cpu_cores", Json::int(0)),
            ("memory_mb", Json::int(0)),
            ("vram_mb", Json::int(0)),
        ]))
        .expect_err("应报错");
        assert!(err.to_string().contains("P5001"), "负区间应报 P5001：{err}");
        // 哈希位数不对。
        let err: FarmError = decode_task(&Json::object(vec![
            ("v", Json::int(1)),
            ("kind", Json::Str("task".into())),
            ("start", Json::int(0)),
            ("end", Json::int(1)),
            ("config_hash_hi", Json::Str("abc".into())),
            ("config_hash_lo", Json::Str("0000000000000000".into())),
            ("priority", Json::int(0)),
            ("cpu_cores", Json::int(0)),
            ("memory_mb", Json::int(0)),
            ("vram_mb", Json::int(0)),
        ]))
        .expect_err("应报错");
        assert!(err.to_string().contains("P5007"), "哈希位数错应报 P5007：{err}");
        let _ = loc;
    }

    #[test]
    fn shard_without_values_array_is_rejected() {
        let json: Json = Json::object(vec![
            ("v", Json::int(1)),
            ("kind", Json::Str("shard".into())),
            ("start", Json::int(0)),
            ("end", Json::int(0)),
            ("config_hash_hi", Json::Str("0000000000000000".into())),
            ("config_hash_lo", Json::Str("0000000000000000".into())),
        ]);
        let err: FarmError = decode_shard::<u64, _>(&json, |_| Ok(0)).expect_err("应报错");
        assert!(err.to_string().contains("P5007"), "缺 values 应报 P5007：{err}");
    }

    #[test]
    fn hex64_is_fixed_width_and_zero_padded() {
        assert_eq!(hex64(0), "0000000000000000");
        assert_eq!(hex64(1), "0000000000000001");
        assert_eq!(hex64(u64::MAX), "ffffffffffffffff");
        // 拆合无损。
        for v in [0u64, 1, 0x1_0000_0000, 0xffff_ffff_0000_0000, u64::MAX] {
            let (hi, lo) = split_u64(v);
            assert_eq!(hi.len(), 16);
            assert_eq!(lo.len(), 16);
            let back: u64 = (u64::from_str_radix(&hi, 16).unwrap_or(0) << 32)
                | u64::from_str_radix(&lo, 16).unwrap_or(0);
            assert_eq!(back, v, "拆合必须无损：v={v}");
        }
    }
}
