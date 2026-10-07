//! 确定性哈希原语（B10； 唯一随机源纪律）。
//!
//! mock 层一切"随机样貌"的派生都经本模块的整数哈希——无全局 RNG、无状态、
//! 无网络：同输入跨平台逐位一致。FNV-1a 64 位（简单、无查表、逐运算 IEEE
//! 确定），配 splitmix 式混洗常量提升雪崩性。

/// FNV-1a 64 位哈希偏移基。
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a 64 位质数。
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// 对字节切片做 FNV-1a 64 位哈希（纯函数，确定性）。
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// 对字符串做 FNV-1a 64 位哈希。
pub fn fnv1a64_str(s: &str) -> u64 {
    fnv1a64(s.as_bytes())
}

/// 混洗（splitmix64 finalizer）：把弱雪崩的哈希扩散到全部 64 位。
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// 派生一个确定性 u64：`seed` 与 `salt` 组合后混洗。
pub fn derive(seed: u64, salt: &str) -> u64 {
    mix64(seed ^ fnv1a64_str(salt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_distinguishes() {
        assert_eq!(fnv1a64_str("hello"), fnv1a64_str("hello"));
        assert_ne!(fnv1a64_str("hello"), fnv1a64_str("hellp"));
        assert_eq!(derive(7, "a"), derive(7, "a"));
        assert_ne!(derive(7, "a"), derive(8, "a"));
        assert_ne!(derive(7, "a"), derive(7, "b"));
    }
}
