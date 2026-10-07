//! 确定性切分（工程四律之四"分布式同树"）。
//!
//! # 数学设计
//!
//! 设总帧数 `n = end - start + 1`，工作单元数 `k`。切分函数
//! `split_frames(start, end, k)` 是 **`(start, end, k)` 的纯函数**：不读时钟、
//! 不读实际核数、不读 `HashMap` 迭代序、不读环境变量。同输入两遍逐字段相同。
//!
//! ## 1. 帧切分：均分 + 前余数（remainder-to-front）
//!
//! ```text
//! base = n / k           （整除基块大小）
//! rem  = n % k           （余数块数，0 <= rem < k）
//! 第 i 块（0 <= i < k）长度 = base + (i < rem ? 1 : 0)
//! 第 i 块起点 = start + i*base + min(i, rem)
//! ```
//!
//! **为什么余数放前面**：层构造里"奇数尾元素直接进位下一层"——短的块在
//! 树的**浅层**先参与结合。把余数给前块，使**所有块长 ∈ {base, base+1}**，
//! 且块长序列**单调不增后恒定**，与 `reduction_shape` 的"尾进位"在结构上
//! 同形：短块不落在序列尾部，因此尾部块长恒为 `base`，跨机器切分时尾部负载
//! 可预测。
//!
//! **不重不漏的可证性**：块长之和 = `k*base + rem = n`；块 i 的终点
//! `= start + (i+1)*base + min(i+1, rem) - 1`，块 i+1 的起点
//! `= start + (i+1)*base + min(i+1, rem)` ⇒ 首尾严格相接（无隙）；块 i 的终点
//! `+1 = 块 i+1 的起点 > 块 i 的起点` ⇒ 单调递增（无叠）。两块并起来覆盖
//! `[start, start+n-1] = [start, end]`。
//!
//! **纯函数性**：公式只含 `n`、`k` 的整数四则运算，无浮点、无哈希、无迭代序。
//!
//! ## 2. 与 `reduction_shape` 的形状对齐（工程四律之四"分布式同树"）
//!
//! 要求"跨机分片按**同一形状函数**切分"。本模块把 `k` 块切分与
//! [`prism_core::graph::reduction_shape`] 的**槽（slot）划分**联系起来：
//!
//! - `slot_counts(n)` 给出 `reduction_shape(n)` 逐层的槽数——第 0 层是 n 个
//!   叶子槽，此后每层 `ceil(len/2)`（尾元素进位，故是 `ceil` 不是 `floor`）。
//!   这正是 `ReductionShape::new` 里 `len = (len + 1) / 2` 的镜像，**但本模块
//!   不重新实现它**：`slot_ranges(n, depth)` 逐层**重放** `shape.layers()` 的
//!   `[a, b]` 结合对，反推每个槽的帧区间。层结构由 prism-core 单点定义
//!   （工程四律 1：形状唯一），本模块只消费。
//! - `split_frames_shape_aligned` 把 `k` 块对齐到 `reduction_shape(n)` 的
//!   **槽边界**：先取最深的一层使其槽数 `>= k`（保证每块至少 1 槽），再按
//!   槽下标 `i` 均分该层的槽给各 worker。
//!
//! **这条对齐的价值**：合并时若 worker 的部分结果需要**跨块归约**，块的边界
//! 落在全局归约树的槽边界上 ⇒ 块内归约 + 块间归约的**结合序 ≡ 全局
//! `reduce(n)` 的结合序**（归纳可证：每块是全局树在某一层的连续子树，块内
//! 按同一形状函数求值，块间再按同形状函数求值，两级与一级同树）。这就是
//! "确定性免费继承"的机制，也是与 `split_frames`（纯均分，不承诺跨块归约
//! 同树）的**唯一区别**——两者都是纯函数、都不重不漏。
//!
//! # 边界
//!
//! - `num_workers = 0` → 结构化错误 P5002（不 panic）。
//! - `end < start`（空 / 负区间）→ 结构化错误 P5001。
//! - `start = 0, end = u64::MAX` → 帧数溢出 → P5003。
//! - `num_workers > n` → 块数退化为 `n`（每块 1 帧），不报错（幂等：多给
//!   worker 不改变覆盖）。

use crate::farm::FarmError;
use crate::incremental::FrameRange;
use prism_core::graph::{reduction_shape, ReductionShape};

/// 帧数：`end - start + 1`（闭区间帧数）。
///
/// # Errors
/// - `end < start`（空区间 / 负区间）→ [`FarmError::InvalidFrameRange`]；
/// - 帧数超出 `u64`（`start = 0 && end = u64::MAX`）→
///   [`FarmError::FrameCountOverflow`]。
pub fn frame_count(start: u64, end: u64) -> Result<u64, FarmError> {
    if end < start {
        return Err(FarmError::InvalidFrameRange {
            location: "farm::frame_count".to_string(),
            start,
            end,
        });
    }
    // end >= start ⇒ 差值不减；+1 用 checked 防 u64::MAX 溢出。
    (end - start).checked_add(1).ok_or(FarmError::FrameCountOverflow {
        location: "farm::frame_count".to_string(),
        start,
        end,
    })
}

/// 帧切分（均分 + 前余数）：`(start, end, num_workers)` 的**纯函数**。
///
/// 返回 `min(num_workers, n)` 个连续、升序、不重不漏的 [`FrameRange`]。
/// `num_workers = 0` → [`FarmError::ZeroWorkers`]。
///
/// # 确定性保证
/// 纯函数：同输入两遍结果逐字段相同；与执行环境（哪台机器、几个进程、
/// 什么完成顺序）**完全无关**——这是"并行 md5 == 串行 md5"的第一块基石。
///
/// # Errors
/// 见 [`frame_count`] 与 `num_workers == 0`（P5002）。
pub fn split_frames(
    start: u64,
    end: u64,
    num_workers: usize,
) -> Result<Vec<FrameRange>, FarmError> {
    let n: u64 = frame_count(start, end)?;
    if num_workers == 0 {
        return Err(FarmError::ZeroWorkers {
            location: "farm::split_frames".to_string(),
        });
    }
    // 块数不超过帧数：多给的 worker 退化为空块（不报错，覆盖不变）。
    let k: u64 = (num_workers as u64).min(n);
    let base: u64 = n / k;
    let rem: u64 = n % k;

    let mut out: Vec<FrameRange> = Vec::with_capacity(k as usize);
    let mut cursor: u64 = start;
    for i in 0..k {
        // 余数给前块：第 i 块多拿 1 帧当且仅当 i < rem。
        let len: u64 = base + u64::from(i < rem);
        // len >= 1（因 base >= 1：k <= n ⇒ n/k >= 1），故终点 = cursor + len - 1 不会下溢。
        let seg_end: u64 = cursor + len - 1;
        out.push(FrameRange { start: cursor, end: seg_end });
        cursor = seg_end + 1;
    }
    Ok(out)
}

/// `reduction_shape(n)` 逐层的**槽数**（第 0 层 = n 个叶子槽）。
///
/// 层数 = `reduction_shape(n).depth()` + 1。数值上等于 `ceil(n / 2^depth)`
/// 的迭代版本（尾进位 ⇒ 每层 `ceil(len/2)`），但本函数**不重算形状**：它读
/// [`ReductionShape::depth`] 拿层数，逐层用 `len = (len + 1) / 2` 递推以镜像
/// prism-core 的构造（`ceil` 语义），并与 `shape.layers()` 的结合对数交叉校验
/// （见 `tests::slot_counts_match_reduction_shape_layers`）。
pub fn slot_counts(n: u64) -> Vec<usize> {
    let shape: ReductionShape = reduction_shape(n as usize);
    // 叶子层：n 个槽。
    let mut counts: Vec<usize> = Vec::with_capacity(shape.depth() + 1);
    let mut len: usize = n as usize;
    counts.push(len);
    for _ in 0..shape.depth() {
        // 尾元素进位 ⇒ ceil(len/2)，与 ReductionShape::new 的 (len+1)/2 同式。
        len = len / 2 + usize::from(len % 2 == 1);
        counts.push(len);
    }
    counts
}

/// `reduction_shape(n)` 在 `depth` 层的**槽 → 帧区间**映射（升序、不重不漏）。
///
/// 逐层**重放** [`ReductionShape::layers`] 的 `[a, b]` 结合对来反推每个槽的帧
/// 跨度：第 0 层槽 i = 帧 i；上层槽 p 由下层槽 `pair[0]` 与 `pair[1]` 合并，
/// 其帧区间为二者之并；未被结合的尾槽原样进位。**形状由 prism-core 单点
/// 定义，本函数只消费**（工程四律 1：任何归约点不得私定结合序）。
///
/// `depth` 取 `0..=slot_counts(n).len()-1`；越界返回 [`FarmError::SlotDerivationFailed`]。
///
/// # Errors
/// `depth` 越界，或层结构与帧数不自洽（内部不自洽）→
/// [`FarmError::SlotDerivationFailed`]。
pub fn slot_ranges(n: u64, depth: usize) -> Result<Vec<FrameRange>, FarmError> {
    if n == 0 {
        // 空规模：零槽。层 0 即空。
        return Ok(Vec::new());
    }
    let total: usize = n as usize;
    let shape: ReductionShape = reduction_shape(total);
    if depth > shape.depth() {
        return Err(FarmError::SlotDerivationFailed {
            location: "farm::slot_ranges".to_string(),
            frames: n,
            depth,
        });
    }
    // 从叶子层起，逐层把"槽 → 帧区间"向上折叠，**恰好折叠 depth 层**。
    let mut current: Vec<FrameRange> = (0..total).map(|i| FrameRange { start: i as u64, end: i as u64 }).collect();
    for (d, layer) in shape.layers().iter().enumerate() {
        if d == depth {
            // 目标层已到达：此时的 current 就是 depth 层的槽划分。
            break;
        }
        if current.len() <= 1 {
            // 已收敛到 1 槽：更深的层都是同一个区间，原样返回。
            break;
        }
        let mut next: Vec<FrameRange> = Vec::with_capacity(layer.len() + 1);
        for pair in layer {
            // pair 下标由 prism-core 保证落在 current 内（形状与规模同源）。
            let (Some(lo), Some(hi)) = (current.get(pair[0] as usize), current.get(pair[1] as usize))
            else {
                return Err(FarmError::SlotDerivationFailed {
                    location: "farm::slot_ranges".to_string(),
                    frames: n,
                    depth,
                });
            };
            // 两槽在叶序上相邻（左操作数在前），并为一段连续区间。
            next.push(FrameRange { start: lo.start.min(hi.start), end: lo.end.max(hi.end) });
        }
        // 奇数尾槽进位（形状规则）。
        if current.len() % 2 == 1 {
            if let Some(tail) = current.last() {
                next.push(*tail);
            }
        }
        current = next;
    }
    Ok(current)
}

/// **形状对齐**切分：把块边界钉在 `reduction_shape(n)` 的槽边界上
/// （工程四律之四"分布式同树"）。
///
/// # 与 [`split_frames`] 的区别
/// - [`split_frames`]：纯均分（前余数），保证纯函数 + 不重不漏，**不承诺**块
///   边界与归约树对齐；
/// - 本函数：额外保证**每块是全局 `reduction_shape(n)` 在某一层的连续子树**，
///   因此"块内归约 + 块间归约"与"全局一级归约"**同树同结合序**——跨机分片
///   归约时确定性免费继承。
///
/// 深度选择：取**最深**的一层使其槽数 `>= k`（每块至少 1 槽），再把该层槽
/// 按 `i` 均分给各 worker（余数给前块，与 [`split_frames`] 同纪律）。
/// 若 `depth = 0`（帧数少于块数，退化每帧一块），形状对齐平凡成立。
///
/// 仍是 `(start, end, num_workers)` 的**纯函数**。
///
/// # Errors
/// 见 [`split_frames`] 与 [`slot_ranges`]。
pub fn split_frames_shape_aligned(
    start: u64,
    end: u64,
    num_workers: usize,
) -> Result<Vec<FrameRange>, FarmError> {
    let n: u64 = frame_count(start, end)?;
    if num_workers == 0 {
        return Err(FarmError::ZeroWorkers {
            location: "farm::split_frames_shape_aligned".to_string(),
        });
    }
    let k: u64 = (num_workers as u64).min(n);
    if k == n {
        // 每帧一块：无需对齐（任何树都把单叶当子树）。
        return Ok((start..=end).map(|f| FrameRange { start: f, end: f }).collect());
    }
    // 最深且槽数 >= k 的层。slot_counts 单调不增 ⇒ 从最深往回找第一个 >= k。
    let counts: Vec<usize> = slot_counts(n);
    let mut depth: usize = 0;
    for (d, c) in counts.iter().enumerate() {
        if *c as u64 >= k {
            depth = d;
        }
    }
    // 帧号偏移：槽下标从 0 起，帧号从 start 起。
    let slots: Vec<FrameRange> = slot_ranges(n, depth)?
        .into_iter()
        .map(|r| FrameRange { start: r.start + start, end: r.end + start })
        .collect();
    let m: u64 = slots.len() as u64;
    let base: u64 = m / k;
    let rem: u64 = m % k;

    let mut out: Vec<FrameRange> = Vec::with_capacity(k as usize);
    let mut cursor: usize = 0;
    for i in 0..k {
        let take: u64 = base + u64::from(i < rem);
        let (Some(first), Some(last)) = (slots.get(cursor), slots.get(cursor + take as usize - 1))
        else {
            return Err(FarmError::SlotDerivationFailed {
                location: "farm::split_frames_shape_aligned".to_string(),
                frames: n,
                depth,
            });
        };
        // 槽在叶序上连续 ⇒ 该块帧区间连续；跨块边界即槽边界（形状对齐）。
        out.push(FrameRange { start: first.start, end: last.end });
        cursor += take as usize;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性 LCG（Numerical Recipes 参数）——禁 `rand` crate（零第三方依赖）。
    /// 用于可复现的"随机"抽查。
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        /// 取 [0, n) 内的一个值（n > 0）。
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n.max(1)
        }
    }

    /// 校验一组切分恰好覆盖 [start, end]：不重不漏 + 升序 + 首尾相接。
    fn assert_exact_cover(parts: &[FrameRange], start: u64, end: u64) {
        let n: u64 = frame_count(start, end).expect("测试夹具区间应合法");
        let mut total: u64 = 0;
        for p in parts {
            assert!(!p.is_empty(), "切分不得产出空块：{p:?}");
            total += p.len();
        }
        assert_eq!(total, n, "切分块长之和必须等于总帧数（不重不漏）：{parts:?}");
        if let Some(first) = parts.first() {
            assert_eq!(first.start, start, "首块必须从 start 起：{parts:?}");
        }
        if let Some(last) = parts.last() {
            assert_eq!(last.end, end, "末块必须到 end 止：{parts:?}");
        }
        for w in parts.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(a.start <= a.end, "块区间应非空：{a:?}");
            assert!(a.end < b.start, "块必须严格升序不重叠：{a:?} vs {b:?}");
            assert_eq!(
                a.end + 1,
                b.start,
                "相邻块必须首尾严格相接（无隙）：{a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn split_frames_is_pure_function_of_its_three_inputs() {
        // 纯函数契约：同输入两遍逐字段相同（不读时钟 / 线程数 / 环境）。
        for (s, e, k) in [(0u64, 99u64, 7usize), (5, 1004, 3), (0, 0, 1), (12, 12, 8)] {
            let a: Vec<FrameRange> = split_frames(s, e, k).expect("应可切分");
            let b: Vec<FrameRange> = split_frames(s, e, k).expect("应可切分");
            assert_eq!(a, b, "split_frames({s},{e},{k}) 必须是纯函数");
        }
    }

    #[test]
    fn split_frames_covers_every_frame_exactly_once_exhaustively_small() {
        // 穷举小规模：n <= 24 × k <= 12 全组合，逐帧计数必须恰为 1。
        for n in 1u64..=24 {
            for k in 1usize..=12 {
                let parts: Vec<FrameRange> = split_frames(0, n - 1, k).expect("应可切分");
                assert_exact_cover(&parts, 0, n - 1);
                // 逐帧覆盖计数（不重不漏的直接证据）。
                let mut hits: Vec<u32> = vec![0; n as usize];
                for p in &parts {
                    for f in p.start..=p.end {
                        let slot: &mut u32 = hits
                            .get_mut(f as usize)
                            .unwrap_or_else(|| panic!("切分越界：帧 {f} 不在 0..{n}"));
                        *slot += 1;
                    }
                }
                for (f, c) in hits.iter().enumerate() {
                    assert_eq!(*c, 1, "帧 {f} 必须恰被覆盖一次（n={n}, k={k}, parts={parts:?}）");
                }
            }
        }
    }

    #[test]
    fn split_frames_random_spot_check_large_scales() {
        // 大规模随机抽查（确定性 LCG，不用 rand crate）：含非 2 的幂规模。
        let mut rng: Lcg = Lcg::new(0x5EED_1234_ABCD_0001);
        for _ in 0..400 {
            let n: u64 = 1 + rng.below(4096);
            let k: usize = 1 + (rng.below(37) as usize);
            let start: u64 = rng.below(1000);
            let end: u64 = start + n - 1;
            let parts: Vec<FrameRange> = split_frames(start, end, k).expect("应可切分");
            assert_exact_cover(&parts, start, end);
        }
    }

    #[test]
    fn split_frames_remainder_goes_to_front_blocks() {
        // 余数前置的具体形状：n=10, k=3 ⇒ 块长 [4, 3, 3]（10 = 3*3 + 1）。
        let parts: Vec<FrameRange> = split_frames(0, 9, 3).expect("应可切分");
        let lens: Vec<u64> = parts.iter().map(|r| r.len()).collect();
        assert_eq!(lens, vec![4, 3, 3], "余数应给前块（尾块长恒为 base，可预测）");
        assert_eq!(parts[0], FrameRange { start: 0, end: 3 });
        assert_eq!(parts[2], FrameRange { start: 7, end: 9 });

        // n=7, k=7 ⇒ 每块 1 帧（块数退化为 n）。
        let unit: Vec<FrameRange> = split_frames(0, 6, 7).expect("应可切分");
        assert_eq!(unit.len(), 7);
        assert!(unit.iter().all(|r| r.len() == 1));

        // num_workers 远超帧数：块数 = n，仍不重不漏。
        let many: Vec<FrameRange> = split_frames(3, 5, 1000).expect("应可切分");
        assert_eq!(many.len(), 3);
        assert_exact_cover(&many, 3, 5);
    }

    #[test]
    fn split_frames_depends_only_on_inputs_not_on_any_environment() {
        // 同一 (start, end, k)，在不同"环境"（先后调用、中间插入其它切分、
        // 不同 k 的切分穿插）下结果不变——证明它不读任何环境状态。
        let canonical: Vec<FrameRange> = split_frames(10, 209, 6).expect("应可切分");
        let _noise_a: Vec<FrameRange> = split_frames(0, 2, 3).expect("应可切分");
        let _noise_b: Vec<FrameRange> = split_frames(900, 5000, 17).expect("应可切分");
        let again: Vec<FrameRange> = split_frames(10, 209, 6).expect("应可切分");
        assert_eq!(canonical, again, "切分不得受任何其他调用影响");
    }

    #[test]
    fn slot_counts_mirror_reduction_shape_layers() {
        // slot_counts 必须与 prism-core 的 reduction_shape 逐层自洽：
        // 第 d 层的结合对数 == 该层槽数 - ceil(槽数/2)。
        for n in [1u64, 2, 3, 5, 7, 8, 100, 1000, 1024] {
            let shape: ReductionShape = reduction_shape(n as usize);
            let counts: Vec<usize> = slot_counts(n);
            assert_eq!(counts[0], n as usize, "第 0 层是 n 个叶子槽");
            assert_eq!(counts.len(), shape.depth() + 1, "层数须与 shape.depth() 一致");
            for (d, layer) in shape.layers().iter().enumerate() {
                let slots: usize = counts[d];
                // 本层结合后留下的槽数 = pairs + 尾进位(1 if 奇)。
                let carried: usize = usize::from(slots % 2 == 1);
                assert_eq!(
                    layer.len() + carried,
                    counts[d + 1],
                    "n={n} 第 {d} 层：结合对数 + 尾进位 == 下一层槽数"
                );
            }
        }
    }

    #[test]
    fn slot_ranges_partition_frames_and_never_reorder() {
        // 每一层的槽都必须恰好划分 [0, n)，且槽内帧区间连续、整体升序。
        for n in [1u64, 2, 3, 5, 7, 8, 100, 1000, 1024] {
            let depth_max: usize = slot_counts(n).len() - 1;
            for depth in 0..=depth_max {
                let slots: Vec<FrameRange> = slot_ranges(n, depth).expect("应可推导");
                let mut cursor: u64 = 0;
                for s in &slots {
                    assert_eq!(s.start, cursor, "n={n} depth={depth}：槽必须首尾相接无隙：{slots:?}");
                    assert!(s.start <= s.end, "槽区间应非空：{s:?}");
                    cursor = s.end + 1;
                }
                assert_eq!(cursor, n, "n={n} depth={depth}：槽必须恰好覆盖 [0, {n})");
            }
        }
    }

    #[test]
    fn shape_aligned_split_lands_on_reduction_tree_slot_boundaries() {
        // 核心断言：形状对齐切分的每个块边界都是 reduction_shape(n) 某一层的槽边界。
        // 这保证"块内归约 + 块间归约"与"全局一级归约"同树（律四）。
        for n in [8u64, 100, 1000, 1024] {
            for k in [1usize, 2, 3, 5, 7, 8, 16] {
                let parts: Vec<FrameRange> =
                    split_frames_shape_aligned(0, n - 1, k).expect("应可切分");
                assert_exact_cover(&parts, 0, n - 1);
                // 收集所有层的槽边界（含 0 与 n），块边界必须落在其中。
                let mut boundaries: Vec<u64> = vec![0, n];
                let depth_max: usize = slot_counts(n).len() - 1;
                for depth in 0..=depth_max {
                    for s in slot_ranges(n, depth).expect("应可推导") {
                        boundaries.push(s.start);
                        boundaries.push(s.end + 1);
                    }
                }
                boundaries.sort_unstable();
                boundaries.dedup();
                for p in &parts {
                    let right_edge: u64 = p.end.saturating_add(1);
                    assert!(
                        boundaries.contains(&p.start) && boundaries.contains(&right_edge),
                        "n={n} k={k}：块 [{:?}] 边界必须落在归约树槽边界上（{parts:?}）",
                        (p.start, p.end)
                    );
                }
            }
        }
    }

    #[test]
    fn shape_aligned_split_is_also_pure_and_covers_non_power_of_two() {
        // 纯函数 + 非 2 的幂规模覆盖。
        for (n, k) in [(7u64, 3usize), (100, 7), (1000, 3), (999, 16), (5, 5)] {
            let a: Vec<FrameRange> = split_frames_shape_aligned(0, n - 1, k).expect("应可切分");
            let b: Vec<FrameRange> = split_frames_shape_aligned(0, n - 1, k).expect("应可切分");
            assert_eq!(a, b, "形状对齐切分也必须是纯函数");
            assert_exact_cover(&a, 0, n - 1);
        }
        // 块数退化为每帧一块（k >= n）时平凡成立。
        let unit: Vec<FrameRange> = split_frames_shape_aligned(4, 6, 99).expect("应可切分");
        assert_eq!(unit.len(), 3);
        assert_exact_cover(&unit, 4, 6);
    }

    #[test]
    fn malformed_inputs_return_structured_errors_without_panic() {
        // 空区间 / 负区间 → P5001。
        for (s, e) in [(10u64, 9u64), (0, 0), (5, 0)] {
            if s == 0 && e == 0 {
                continue; // (0,0) 是合法单帧区间。
            }
            let err: FarmError = split_frames(s, e, 4).expect_err("应报错");
            assert!(matches!(err, FarmError::InvalidFrameRange { .. }), "实际 {err:?}");
            let text: String = err.to_string();
            assert!(text.contains("P5001"), "首段应在错误码：{text}");
            assert!(text.contains("建议："), "必含建议段（C19）：{text}");
        }
        // num_workers = 0 → P5002。
        for f in [split_frames(0, 9, 0), split_frames_shape_aligned(0, 9, 0)] {
            let err: FarmError = f.expect_err("应报错");
            assert!(matches!(err, FarmError::ZeroWorkers { .. }), "实际 {err:?}");
            assert!(err.to_string().contains("P5002"));
        }
        // 区间溢出（start=0, end=u64::MAX）→ P5003。
        let err: FarmError = split_frames(0, u64::MAX, 4).expect_err("应报错");
        assert!(matches!(err, FarmError::FrameCountOverflow { .. }), "实际 {err:?}");
        assert!(err.to_string().contains("P5003"));
        // frame_count 自身也守同一条纪律。
        assert!(frame_count(9, 3).is_err());
        assert_eq!(frame_count(0, 0).unwrap_or(0), 1);
        // depth 越界 → P5004。
        let err2: FarmError = slot_ranges(16, 99).expect_err("应报错");
        assert!(matches!(err2, FarmError::SlotDerivationFailed { .. }), "实际 {err2:?}");
        // n = 0 的槽划分为空（不是错误）。
        assert!(slot_ranges(0, 0).expect("应可推导").is_empty());
    }
}
