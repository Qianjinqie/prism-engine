//! 基数排序（GPU 基础设施，D0）。
//!
//! 语义权威：radix sort，SPH 空间哈希 / 粒子排序共用、
//! D-15、D-21、D-26。采用 **LSD（最低有效位优先）多趟计数排序**：每趟对 8 位
//! 数位做稳定计数排序，`bits/8` 趟后全局有序。
//!
//! 结构映射（D-21）：计数排序 = 直方图 + **前缀和（ExScan）** + 稳定散布。
//! 每趟位桶的 ExScan 复用 [`crate::scan`] 的 Blelloch 树形扫描——与 D-26 同
//! 形状语义。稳定性来自"按输入序从左到右散布"（同位数者保序）。
//!
//! 提供：
//! * [`radix_sort_u32`]：GPU 风格 LSD 基数排序（CPU 仿真，真机路径待接入）；
//! * [`sort_cpu_ref`]：串行参考（稳定插入口径），定义语义真值；
//! * 键可携带载荷（`(key, payload)`）——SPH 粒子排序需同步重排粒子索引。

use crate::scan::scan_blelloch_cpu;

/// 每趟处理的位数（8 位 ⇒ 256 桶；`u32` 共 4 趟）。GPU 侧常用 4/8 位，
/// 8 位在桶数与趟数间取平衡（D-16 Mini Code：常数即参数）。
const RADIX_BITS: u32 = 8;
/// 桶数 = 2^RADIX_BITS。
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// 单趟稳定计数排序：按 `shift` 处的 8 位数字重排 `(key, payload)`。
///
/// 直方图 → ExScan（经 Blelloch）→ 稳定散布。返回新数组，不改动入参。
fn counting_sort_pass(items: &[(u32, u32)], shift: u32) -> Vec<(u32, u32)> {
    // 1) 直方图（各桶计数）。
    let mut hist = [0u32; RADIX_BUCKETS];
    for &(k, _) in items {
        let digit = ((k >> shift) & (RADIX_BUCKETS as u32 - 1)) as usize;
        hist[digit] = hist[digit].wrapping_add(1);
    }
    // 2) ExScan：bucket_off[b] = Σ_{d<b} hist[d]（Blelloch 树形扫描，同 D-26 形状）。
    //    scan_blelloch_cpu 返回 inclusive 前缀和：incl[b] = Σ_{d≤b} hist[d]。
    //    桶 b 的起始偏移 = incl[b] - hist[b] = Σ_{d<b} hist[d]。
    let incl = scan_blelloch_cpu(&hist);
    let mut offset = [0u32; RADIX_BUCKETS];
    for b in 0..RADIX_BUCKETS {
        offset[b] = incl[b].wrapping_sub(hist[b]);
    }
    // 3) 稳定散布：按输入序从左到右写入（同数位保持相对序 ⇒ 稳定）。
    let mut out: Vec<(u32, u32)> = vec![(0, 0); items.len()];
    let mut cursor = offset;
    for &(k, p) in items {
        let digit = ((k >> shift) & (RADIX_BUCKETS as u32 - 1)) as usize;
        out[cursor[digit] as usize] = (k, p);
        cursor[digit] = cursor[digit].wrapping_add(1);
    }
    out
}

/// GPU 风格 LSD 基数排序（CPU 仿真）：稳定排序 `(key, payload)`。
///
/// 键升序；同键者按输入序稳定保序。`u32` 键需 ⌈32/8⌉ = 4 趟。空/单元素直接返回。
pub fn radix_sort_u32(items: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut cur: Vec<(u32, u32)> = items.to_vec();
    let mut shift: u32 = 0;
    while shift < 32 {
        cur = counting_sort_pass(&cur, shift);
        shift += RADIX_BITS;
    }
    cur
}

/// 串行参考排序（稳定）：插入排序口径直白定义语义真值。
///
/// 稳定（同键保输入序），供 radix 结果比对。规模小，参考实现重在正确性。
pub fn sort_cpu_ref(items: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = items.to_vec();
    let n = out.len();
    for i in 1..n {
        let cur = out[i];
        let mut j = i;
        // 严格 `<` 保证稳定（相等不后移）。
        while j > 0 && out[j - 1].0 > cur.0 {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = cur;
    }
    out
}

/// 仅排键（丢弃载荷）的便捷入口——SPH 空间哈希键排序等场景。
pub fn radix_sort_keys(keys: &[u32]) -> Vec<u32> {
    let items: Vec<(u32, u32)> = keys.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect();
    radix_sort_u32(&items).into_iter().map(|(k, _)| k).collect()
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// 确定性伪随机源（无全局 RNG）：LCG，种子显式给定。
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 32) as u32
    }

    /// radix 与参考排序逐元素一致（多尺寸含非 2 幂）。
    #[test]
    fn radix_matches_reference_sort() {
        for n in [0usize, 1, 2, 3, 5, 17, 100, 257] {
            let mut seed: u64 = 0x1234_5678_9abc_def0;
            let items: Vec<(u32, u32)> = (0..n)
                .map(|i| (lcg(&mut seed), i as u32))
                .collect();
            let radix = radix_sort_u32(&items);
            let reference = sort_cpu_ref(&items);
            assert_eq!(radix, reference, "n={n}：radix 应与参考排序逐元素一致");
            // 升序性校验。
            for w in radix.windows(2) {
                assert!(w[0].0 <= w[1].0, "n={n}：结果应升序");
            }
        }
    }

    /// 稳定性：大量重复键下，同键载荷顺序 = 输入序。
    #[test]
    fn radix_sort_is_stable() {
        let mut seed: u64 = 42;
        // 键限制在小集合内，制造大量重复键。
        let items: Vec<(u32, u32)> = (0..500)
            .map(|i| (lcg(&mut seed) % 4, i as u32))
            .collect();
        let sorted = radix_sort_u32(&items);

        // 对每个键，其载荷序列应与输入中出现序一致。
        let mut expected: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &(k, p) in &items {
            expected.entry(k).or_default().push(p);
        }
        let mut idx = 0usize;
        for (k, payloads) in &expected {
            for &p in payloads {
                assert_eq!(sorted[idx].0, *k, "键 {k} 分组应连续且有序");
                assert_eq!(sorted[idx].1, p, "键 {k} 的载荷应保持输入序（稳定）");
                idx += 1;
            }
        }
        assert_eq!(idx, items.len());
    }

    /// 确定性：同输入两遍结果一致。
    #[test]
    fn radix_sort_is_deterministic() {
        let mut seed: u64 = 7;
        let items: Vec<(u32, u32)> = (0..1000)
            .map(|i| (lcg(&mut seed), i as u32))
            .collect();
        assert_eq!(radix_sort_u32(&items), radix_sort_u32(&items));
    }

    /// 边界：空与单元素。
    #[test]
    fn radix_sort_edge_cases() {
        assert!(radix_sort_u32(&[]).is_empty());
        assert_eq!(radix_sort_u32(&[(9, 5)]), vec![(9, 5)]);
        // u32 边界值（含最高位）排序正确——覆盖第 4 趟（shift=24）。
        let edge = vec![(u32::MAX, 0), (0, 1), (0x8000_0000, 2), (1, 3)];
        let sorted = radix_sort_u32(&edge);
        assert_eq!(
            sorted.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            vec![0, 1, 0x8000_0000, u32::MAX]
        );
    }
}
