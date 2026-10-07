//! prism-io cache_manifest 攻击测试。
//!
//! 面：crates/prism-io/src/cache_manifest.rs。
//! 重点：序列化逐字节确定、往返、版本/损坏拒收、空清单、快照帧位区分。

use prism_io::cache_manifest::{CacheManifest, SnapshotManifest, CACHE_MANIFEST_SCHEMA_VERSION};

const MAGIC: &[u8; 12] = b"PRISMCACHEMJ";

fn key(n: u8) -> [u8; 32] {
    [n; 32]
}

/// 缺陷面：同内容两次导出逐字节一致（缓存确定性铁律）。
#[test]
fn manifest_bytes_deterministic() {
    let mut m = CacheManifest::new();
    m.insert(key(3), vec![1, 2, 3]);
    m.insert(key(1), vec![]);
    m.insert(key(2), vec![9; 5]);
    assert_eq!(m.to_bytes(), m.to_bytes(), "两次导出必须逐字节一致");
    // 插入序无关。
    let mut m2 = CacheManifest::new();
    m2.insert(key(2), vec![9; 5]);
    m2.insert(key(1), vec![]);
    m2.insert(key(3), vec![1, 2, 3]);
    assert_eq!(m.to_bytes(), m2.to_bytes(), "插入序不得影响导出");
}

/// 缺陷面：空清单往返 + 魔数头 + 版本。
#[test]
fn manifest_empty_roundtrip() {
    let m = CacheManifest::new();
    assert!(m.is_empty());
    let bytes = m.to_bytes();
    assert_eq!(&bytes[0..12], MAGIC, "魔数头");
    assert_eq!(bytes.len(), 12 + 4 + 8, "空清单 = 头 + 版本 + 计数 0");
    let parsed = CacheManifest::from_bytes(&bytes).unwrap();
    assert_eq!(parsed, m, "空清单往返");
    assert_eq!(parsed.len(), 0);
}

/// 缺陷面：往返保真（含空 payload 与长 payload）。
#[test]
fn manifest_roundtrip_fidelity() {
    let mut m = CacheManifest::new();
    m.insert(key(0), vec![]);
    m.insert(key(255), (0..=255u8).collect());
    let parsed = CacheManifest::from_bytes(&m.to_bytes()).unwrap();
    assert_eq!(parsed, m);
    assert_eq!(parsed.get(&key(0)), Some(&vec![]));
    assert_eq!(parsed.get(&key(255)).unwrap().len(), 256);
}

/// 缺陷面：损坏拒收——魔数、schema、截断，均结构化错误（不改既有文件）。
#[test]
fn manifest_rejects_corruption() {
    // 魔数坏。
    let mut bad = CacheManifest::new().to_bytes();
    bad[0] = b'X';
    assert_eq!(CacheManifest::from_bytes(&bad).unwrap_err().code, "PF2001");

    // schema 版本不符 → PF2002。
    let mut old = CacheManifest::new().to_bytes();
    old[12..16].copy_from_slice(&(CACHE_MANIFEST_SCHEMA_VERSION + 1).to_be_bytes());
    let e = CacheManifest::from_bytes(&old).unwrap_err();
    assert_eq!(e.code, "PF2002");
    assert!(e.to_string().contains("建议："), "必含建议段");

    // 过短。
    assert_eq!(
        CacheManifest::from_bytes(&[0u8; 4]).unwrap_err().code,
        "PF2001"
    );

    // 头部 OK 但条目区截断（伪造 1 条但无数据）。
    let mut trunc = CacheManifest::new().to_bytes();
    trunc[16..24].copy_from_slice(&1u64.to_be_bytes()); // 声称 1 条
    assert_eq!(
        CacheManifest::from_bytes(&trunc).unwrap_err().code,
        "PF2001"
    );
}

/// 缺陷面：content_hash 稳定（同内容同指纹，不同内容不同指纹）。
#[test]
fn manifest_content_hash_stable() {
    let mut a = CacheManifest::new();
    a.insert(key(1), vec![1]);
    let mut b = CacheManifest::new();
    b.insert(key(1), vec![1]);
    assert_eq!(a.content_hash(), b.content_hash(), "同内容同指纹");
    let mut c = CacheManifest::new();
    c.insert(key(1), vec![2]);
    assert_ne!(a.content_hash(), c.content_hash(), "不同内容不同指纹");
}

/// 缺陷面：快照清单帧位区分（D-28：键 = (签名, 帧)）。
#[test]
fn snapshot_frame_bit_distinguishes() {
    let mut s = SnapshotManifest::new();
    s.insert([1u8; 32], 42, vec![7, 7]);
    s.insert([1u8; 32], 43, vec![8]);
    s.insert([2u8; 32], 0, vec![]);
    assert_eq!(s.get(&[1u8; 32], 42), Some(&vec![7, 7]));
    assert!(s.get(&[1u8; 32], 44).is_none(), "帧号不同不得误取");
    assert!(s.get(&[9u8; 32], 42).is_none(), "签名不同不得误取");
    assert_eq!(s.len(), 3);
}

/// 缺陷面：快照清单导出确定 + 往返保真。
#[test]
fn snapshot_roundtrip_deterministic() {
    let mut s = SnapshotManifest::new();
    s.insert([1u8; 32], 42, vec![7, 7]);
    s.insert([1u8; 32], 43, vec![8]);
    let bytes = s.to_bytes();
    assert_eq!(bytes, s.to_bytes(), "导出确定");
    assert_eq!(SnapshotManifest::from_bytes(&bytes).unwrap(), s, "往返保真");
    assert_eq!(s.records().len(), 2);
}

/// 缺陷面：快照清单损坏拒收。
#[test]
fn snapshot_rejects_corruption() {
    assert_eq!(
        SnapshotManifest::from_bytes(&[0u8; 4]).unwrap_err().code,
        "PF2001"
    );
    let mut trunc = SnapshotManifest::new().to_bytes();
    trunc[0..8].copy_from_slice(&1u64.to_be_bytes()); // 声称 1 条但无数据
    assert_eq!(
        SnapshotManifest::from_bytes(&trunc).unwrap_err().code,
        "PF2001"
    );
}

/// 缺陷面：空快照清单往返。
#[test]
fn snapshot_empty_roundtrip() {
    let s = SnapshotManifest::new();
    assert!(s.is_empty());
    let parsed = SnapshotManifest::from_bytes(&s.to_bytes()).unwrap();
    assert_eq!(parsed, s);
}

/// 缺陷面：remove/keys 契约（逐出后不影响其余条目）。
#[test]
fn manifest_remove_isolates() {
    let mut m = CacheManifest::new();
    m.insert(key(1), vec![1]);
    m.insert(key(2), vec![2]);
    assert_eq!(m.remove(&key(1)), Some(vec![1]));
    assert!(!m.contains(&key(1)));
    assert!(m.contains(&key(2)));
    assert_eq!(m.len(), 1);
    // keys 升序确定。
    let ks: Vec<u8> = m.keys().map(|k| k[0]).collect();
    assert_eq!(ks, vec![2u8]);
}
