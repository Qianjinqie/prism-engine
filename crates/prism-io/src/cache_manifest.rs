//! 增量渲染缓存清单（cache.json）——C3（C3 行）。
//!
//! 语义权威：（缓存键 = 子图签名 × 分辨率 × spp 档 × tile ×
//! engine 版本 × schema 版本；[T] 红利：签名不变即累积续用）；
//! 缓存键五元组 + tile，与 `signature.rs` 咬合、D-28
//! （快照键 = (子图签名, 帧)）；D-16（Mini Code）、D-28
//! （快照库 LRU 逐出只影响重算不影响结果）。
//!
//! # 职责
//! - **manifest 序列化**：逐字节确定性（条目按缓存键升序、字段定长/长度前缀、
//!   大端位模式），同名清单两次导出逐字节一致——缓存本身也守确定性铁律；
//! - **快照库读写**：把 `prism-core::SnapshotStore`（内存态）与磁盘 manifest
//!   咬合，供跨会话复用（D-28：状态是纯数据资产，读档等价 seek）；
//! - 本模块**不 import prism-core**（依赖方向：io∈L4 不得依赖 core∈L2，
//!   与 `signature.rs` 的 SHA-256 双实现说明同一约束），故清单的键/条目以
//!   纯字节（`[u8; 32]`）表达，core 侧 [`Hash256`] 与之位模式同一。

use std::collections::BTreeMap;
use std::fmt;

use crate::sha256::sha256;

// ---------------------------------------------------------------------------
// manifest 版本与域分隔（宪章 C13/R3：版本变更 → 旧缓存只失效重建、永不误复用）
// ---------------------------------------------------------------------------

/// cache.json schema 版本。任何字节布局变更都必须递增（C13）。
pub const CACHE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// 清单魔数头（宪章 C8：内部格式带版本头；静默失效重建 ≠ 用户数据丢失）。
const MANIFEST_MAGIC: &[u8; 12] = b"PRISMCACHEMJ";

// ---------------------------------------------------------------------------
// 缓存条目与清单
// ---------------------------------------------------------------------------

/// 单条缓存条目：键 + 产物字节（产物 = 已量化的 tile RGBA16F 位模式，或
/// 模拟快照字节，视场景而定）。
///
/// `payload` 的语义由使用方定：tile 渲染产物走 [`CachedTile`]，模拟快照走
/// 快照库路径——本清单只保证**逐字节存取往返**。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// 缓存键（= `signature::cache_key` 输出；运到这里为纯 32 字节）。
    pub key: [u8; 32],
    /// 产物字节（长度前缀序列化）。
    pub payload: Vec<u8>,
}

/// 缓存清单：缓存键 → 产物，序列化序固定（键升序）。
///
/// 内存态条目用 [`BTreeMap`]，保证遍历与导出序确定；LRU 的"最近使用"策略
/// 由高层 [`crate`] 调用方维护（本模块只做确定性序列化与存取）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheManifest {
    entries: BTreeMap<[u8; 32], Vec<u8>>,
}

impl CacheManifest {
    /// 构造空清单。
    pub fn new() -> Self {
        Self::default()
    }

    /// 写入/覆盖一条条目。
    pub fn insert(&mut self, key: [u8; 32], payload: Vec<u8>) {
        self.entries.insert(key, payload);
    }

    /// 读取一条条目。
    pub fn get(&self, key: &[u8; 32]) -> Option<&Vec<u8>> {
        self.entries.get(key)
    }

    /// 是否含某键（cache 命中判定）。
    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.entries.contains_key(key)
    }

    /// 删除一条（LRU 逐出 / 失效重建）。
    pub fn remove(&mut self, key: &[u8; 32]) -> Option<Vec<u8>> {
        self.entries.remove(key)
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 全部键（升序，确定性遍历序）。
    pub fn keys(&self) -> impl Iterator<Item = &[u8; 32]> + '_ {
        self.entries.keys()
    }

    /// 导出为确定性字节（宪章 C8 版本头；键升序、长度前缀、大端）。
    ///
    /// 同内容清单两次导出逐字节一致（缓存确定性判据）。
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&CACHE_MANIFEST_SCHEMA_VERSION.to_be_bytes());
        out.extend_from_slice(&(self.entries.len() as u64).to_be_bytes());
        for (key, payload) in &self.entries {
            out.extend_from_slice(key);
            out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    /// 从字节解析（版本不符 / 长度不符 → 结构化错误，永不静默）。
    pub fn from_bytes(data: &[u8]) -> Result<Self, ManifestError> {
        let location = "cache_manifest.解析";
        if data.len() < 12 + 4 + 8 {
            return Err(ManifestError {
                code: "PF2001",
                location: location.to_string(),
                problem: format!("清单过短（{} 字节），不足固定头长度", data.len()),
                fix: "确认文件是 cache.json；损坏文件直接删除后重建".to_string(),
            });
        }
        if &data[0..12] != MANIFEST_MAGIC {
            return Err(ManifestError {
                code: "PF2001",
                location: location.to_string(),
                problem: "清单魔数不匹配，非本引擎 cache.json".to_string(),
                fix: "确认文件是 cache.json；非本格式直接删除后重建".to_string(),
            });
        }
        let mut schema_bytes = [0u8; 4];
        schema_bytes.copy_from_slice(&data[12..16]);
        let schema = u32::from_be_bytes(schema_bytes);
        if schema != CACHE_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError {
                code: "PF2002",
                location: location.to_string(),
                problem: format!(
                    "schema 版本不符：文件 {schema}，当前 {}",
                    CACHE_MANIFEST_SCHEMA_VERSION
                ),
                fix: "旧版本缓存不误复用，删除 cache.json 后按当前版本重建（C13）".to_string(),
            });
        }
        let mut count_bytes = [0u8; 8];
        count_bytes.copy_from_slice(&data[16..24]);
        let count = u64::from_be_bytes(count_bytes) as usize;

        let mut entries = BTreeMap::new();
        let mut pos = 24usize;
        for i in 0..count {
            if pos + 32 + 4 > data.len() {
                return Err(ManifestError {
                    code: "PF2001",
                    location: location.to_string(),
                    problem: format!("第 {i} 条条目键区越界（偏移 {pos}）"),
                    fix: "清单被截断，删除后重建".to_string(),
                });
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&data[pos..pos + 32]);
            pos += 32;
            let mut len_bytes = [0u8; 4];
            len_bytes.copy_from_slice(&data[pos..pos + 4]);
            pos += 4;
            let plen = u32::from_be_bytes(len_bytes) as usize;
            if pos + plen > data.len() {
                return Err(ManifestError {
                    code: "PF2001",
                    location: location.to_string(),
                    problem: format!("第 {i} 条条目产物区越界（需 {plen} 字节）"),
                    fix: "清单被截断，删除后重建".to_string(),
                });
            }
            let payload = data[pos..pos + plen].to_vec();
            pos += plen;
            entries.insert(key, payload);
        }
        Ok(CacheManifest { entries })
    }

    /// 清单内容指纹（用于判据：两次导出 md5/哈希一致）。
    pub fn content_hash(&self) -> [u8; 32] {
        sha256(&self.to_bytes())
    }
}

// ---------------------------------------------------------------------------
// 快照库磁盘镜像（[T] 红利：跨会话续用累积/模拟状态）
// ---------------------------------------------------------------------------

/// 快照清单项：模拟快照键 = (签名位, 帧) → 状态字节。
///
/// 与 `prism-core::node::SnapshotKey` 位模式一致（`([u8; 32], u64)`）；本模块
/// 不 import core（依赖方向），故以本结构镜像表达。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRecord {
    /// 快照签名位（`signature::snapshot_signature` 输出）。
    pub sig: [u8; 32],
    /// 帧号。
    pub frame: u64,
    /// 状态字节（纯数据资产）。
    pub data: Vec<u8>,
}

/// 快照库磁盘镜像：键 (签名, 帧) → 状态字节，序列化序固定（元组升序）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SnapshotManifest {
    entries: BTreeMap<([u8; 32], u64), Vec<u8>>,
}

impl SnapshotManifest {
    /// 构造空快照清单。
    pub fn new() -> Self {
        Self::default()
    }

    /// 写入一条快照。
    pub fn insert(&mut self, sig: [u8; 32], frame: u64, data: Vec<u8>) {
        self.entries.insert((sig, frame), data);
    }

    /// 读取一条快照。
    pub fn get(&self, sig: &[u8; 32], frame: u64) -> Option<&Vec<u8>> {
        self.entries.get(&(*sig, frame))
    }

    /// 删除一条（LRU 逐出）。
    pub fn remove(&mut self, sig: &[u8; 32], frame: u64) -> Option<Vec<u8>> {
        self.entries.remove(&(*sig, frame))
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 导出为确定性字节（元组升序、长度前缀、大端）。
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.entries.len() as u64).to_be_bytes());
        for ((sig, frame), data) in &self.entries {
            out.extend_from_slice(sig);
            out.extend_from_slice(&frame.to_be_bytes());
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(data);
        }
        out
    }

    /// 从字节解析（截断 → 结构化错误）。
    pub fn from_bytes(data: &[u8]) -> Result<Self, ManifestError> {
        let location = "cache_manifest.snapshot.解析";
        if data.len() < 8 {
            return Err(ManifestError {
                code: "PF2001",
                location: location.to_string(),
                problem: format!("快照清单过短（{} 字节）", data.len()),
                fix: "删除损坏的快照清单后重建".to_string(),
            });
        }
        let mut count_bytes = [0u8; 8];
        count_bytes.copy_from_slice(&data[0..8]);
        let count = u64::from_be_bytes(count_bytes) as usize;

        let mut entries = BTreeMap::new();
        let mut pos = 8usize;
        for i in 0..count {
            if pos + 32 + 8 + 4 > data.len() {
                return Err(ManifestError {
                    code: "PF2001",
                    location: location.to_string(),
                    problem: format!("第 {i} 条快照键区越界（偏移 {pos}）"),
                    fix: "快照清单被截断，删除后重建".to_string(),
                });
            }
            let mut sig = [0u8; 32];
            sig.copy_from_slice(&data[pos..pos + 32]);
            pos += 32;
            let mut fbytes = [0u8; 8];
            fbytes.copy_from_slice(&data[pos..pos + 8]);
            pos += 8;
            let frame = u64::from_be_bytes(fbytes);
            let mut lbytes = [0u8; 4];
            lbytes.copy_from_slice(&data[pos..pos + 4]);
            pos += 4;
            let plen = u32::from_be_bytes(lbytes) as usize;
            if pos + plen > data.len() {
                return Err(ManifestError {
                    code: "PF2001",
                    location: location.to_string(),
                    problem: format!("第 {i} 条快照数据区越界（需 {plen} 字节）"),
                    fix: "快照清单被截断，删除后重建".to_string(),
                });
            }
            entries.insert((sig, frame), data[pos..pos + plen].to_vec());
            pos += plen;
        }
        Ok(SnapshotManifest { entries })
    }

    /// 全部记录（升序，绕过 BTreeMap 私有键，供遍历）。
    pub fn records(&self) -> Vec<SnapshotRecord> {
        self.entries
            .iter()
            .map(|((sig, frame), data)| SnapshotRecord {
                sig: *sig,
                frame: *frame,
                data: data.clone(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 清单错误（宪章 C18 三段式；C19：必含建议段）
// ---------------------------------------------------------------------------

/// cache.json 解析错误。错误码取 PF2 段（cache 清单族，独立前缀避免与
/// postfx PF1xxx 混淆）。
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestError {
    /// 错误码（PF2xxx 段）。
    pub code: &'static str,
    /// 三段式一【在哪】。
    pub location: String,
    /// 三段式二【什么事】。
    pub problem: String,
    /// 三段式三【怎么改】。
    pub fix: String,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}] {}\n  什么事：{}\n  建议：{}",
            self.code, self.location, self.problem, self.fix
        )
    }
}

impl std::error::Error for ManifestError {}

// ---------------------------------------------------------------------------
// 测试：确定性序列化 + 字节往返 + 版本/损坏拒收
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> [u8; 32] {
        [n; 32]
    }

    #[test]
    fn manifest_roundtrip_and_deterministic_bytes() {
        let mut m = CacheManifest::new();
        m.insert(key(3), vec![1, 2, 3]);
        m.insert(key(1), vec![]);
        m.insert(key(2), vec![9, 9, 9, 9, 9]);

        // 逐字节确定性：两次导出一致，且带魔数头（C8）。
        let b1 = m.to_bytes();
        let b2 = m.to_bytes();
        assert_eq!(b1, b2, "同清单两次导出必须逐字节一致");
        assert_eq!(&b1[0..12], MANIFEST_MAGIC, "清单带魔数头");

        // 往返：解析回同内容，键序不敏感（BTreeMap）。
        let parsed = CacheManifest::from_bytes(&b1).unwrap();
        assert_eq!(parsed, m);
        assert_eq!(parsed.get(&key(3)), Some(&vec![1, 2, 3]));
        assert_eq!(parsed.get(&key(1)), Some(&vec![]));
        assert!(parsed.contains(&key(2)));
        assert_eq!(parsed.len(), 3);

        // 插入序无关：不同插入序同字节。
        let mut m2 = CacheManifest::new();
        m2.insert(key(2), vec![9, 9, 9, 9, 9]);
        m2.insert(key(1), vec![]);
        m2.insert(key(3), vec![1, 2, 3]);
        assert_eq!(m2.to_bytes(), b1, "插入序不影响导出字节");

        // 内容指纹稳定。
        assert_eq!(m.content_hash(), m2.content_hash());
    }

    #[test]
    fn manifest_rejects_bad_magic_and_schema() {
        // 魔数不符 → PF2001。
        let mut bad = CacheManifest::new().to_bytes();
        bad[0] = b'X';
        let e = CacheManifest::from_bytes(&bad).unwrap_err();
        assert_eq!(e.code, "PF2001");
        assert!(e.to_string().contains("建议："));

        // schema 版本不符 → PF2002（旧版本不误复用，C13）。
        let mut old = CacheManifest::new().to_bytes();
        old[12..16].copy_from_slice(&999u32.to_be_bytes());
        let e = CacheManifest::from_bytes(&old).unwrap_err();
        assert_eq!(e.code, "PF2002");
        assert!(e.problem.contains("schema"));

        // 截断 → 结构化错误。
        let e = CacheManifest::from_bytes(&[0u8; 4]).unwrap_err();
        assert_eq!(e.code, "PF2001");
    }

    #[test]
    fn snapshot_manifest_roundtrip_preserves_frames() {
        let mut s = SnapshotManifest::new();
        s.insert([1u8; 32], 42, vec![7, 7]);
        s.insert([1u8; 32], 43, vec![8]);
        s.insert([2u8; 32], 0, vec![]);

        let bytes = s.to_bytes();
        assert_eq!(bytes, s.to_bytes(), "快照清单导出确定");
        let parsed = SnapshotManifest::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, s);

        // 键位不同互不误取（帧位 / 签名位，D-28 seek 等价前提）。
        assert_eq!(parsed.get(&[1u8; 32], 42), Some(&vec![7, 7]));
        assert!(parsed.get(&[1u8; 32], 44).is_none());
        assert!(parsed.get(&[9u8; 32], 42).is_none());

        // LRU 逐出不影响其余条目（只影响重算，不影响结果）。
        let mut p2 = parsed.clone();
        assert_eq!(p2.remove(&[1u8; 32], 42), Some(vec![7, 7]));
        assert!(p2.get(&[1u8; 32], 42).is_none());
        assert_eq!(p2.get(&[1u8; 32], 43), Some(&vec![8]));
        assert_eq!(p2.len(), 2);

        assert_eq!(parsed.records().len(), 3);
    }
}
