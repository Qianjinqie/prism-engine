//! 结果合并：形状对齐的确定性归约（固定形状归约代数）。
//!
//! # 铁律：块内不归约，归约只在主节点发生一次
//!
//! 本模块的合并入口 [`merge_shards`] 收 [`FrameShard`]——**每个分片带的是逐帧
//! 部分值，不是块内汇总值**。合并步骤：
//!
//! 1. **按帧号归位**（律三"消序用排序"）：分片可以任意顺序到达、按任意
//!    顺序被调度器分发；归位键是**帧号**（数据内容），不是完成顺序；
//! 2. **不重不漏校验**：归位后逐帧值必须恰好覆盖全部帧，缺一帧即 P5005；
//! 3. **一次归约**：[`prism_core::graph::reduce`] 按
//!    [`prism_core::graph::reduction_shape`] 求值——形状是帧数 n 的纯函数。
//!
//! # 为什么"并行 ≡ 串行"是定理
//!
//! 串行参考 = `reduce(全部 n 个逐帧值)`。并行 = 各 worker 算自己那几帧的逐帧值
//! + 主节点把它们**按帧号拼回同一个 n 元序列** + 同一个 `reduce`。两条路径喂给
//! `reduce` 的是**逐位相同的序列**（同样的值、同样的顺序），`reduction_shape(n)`
//! 又与执行拓扑解耦 ⇒ 结果逐位一致。**worker 数、完成顺序、机器数都不出现在
//! 任何一个因子里。**
//!
//! # 反面纪律（为什么不能块内预归约）
//!
//! 若 worker 先把块内浮点预归约成一个值再交给主节点，主节点看到的是 k 个值
//! 而不是 n 个值，`reduce` 求的是 `reduction_shape(k)` 那棵树——**与
//! `reduction_shape(n)` 是不同的树，结合序不同，浮点结果会漂移**。所以
//! [`FrameShard`] 没有"块内汇总"字段；这不是 API 疏漏，是纪律的固化。
//!
//! [`reduce_layered_parallel`] 是给"归约本身也想并行"准备的第二入口：它按
//! `reduction_shape(n)` 的**层**并行——层内各对互不依赖，可任意顺序、任意
//! 并行度求值；跨层推进严格按形状给的结合序。它与串行 `reduce` 逐位一致
//! （同树同结合序），可作对照证据。

use crate::farm::FarmError;
use crate::farm::task::ConfigHash;
use crate::incremental::FrameRange;
use prism_core::graph::{reduce, reduction_shape, ReductionShape};

/// 逐帧部分值：可克隆 + 可跨线程发送（并行归约的最小能力要求）。
pub trait FrameValue: Clone + Send {}

impl<T: Clone + Send> FrameValue for T {}

/// 一个分片的产物：帧区间 + **逐帧**部分值。
///
/// **刻意不提供"块内汇总值"字段**——见模块文档的反面纪律：块内预归约会改变
/// 结合序，使结果随切分漂移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameShard<T> {
    /// 本分片覆盖的帧区间。
    pub frames: FrameRange,
    /// 逐帧部分值，长度必须等于 `frames.len()`（否则 P5006）。
    pub values: Vec<T>,
    /// 产出本分片的配置哈希（manifest 用）。
    pub config_hash: ConfigHash,
}

impl<T: Clone + Send> FrameShard<T> {
    /// 构造分片（不校验长度——校验在合并点做，见 [`merge_shards`]）。
    pub fn new(frames: FrameRange, values: Vec<T>, config_hash: ConfigHash) -> FrameShard<T> {
        FrameShard { frames, values, config_hash }
    }
}

/// 合并审计：覆盖范围 + 归约树形状（形状 = 帧数的纯函数，可被测试直接观测）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeAudit {
    /// 归位后的帧数。
    pub frames: u64,
    /// 归约树深度（`reduction_shape(n).depth()` = ⌈log₂ n⌉）。
    pub tree_depth: usize,
    /// 归约树叶槽数（`reduction_shape(n)` 的第 0 层槽数 = n）。
    pub leaf_slots: usize,
}

/// 合并各分片的逐帧部分值：**按帧号归位 → 校验覆盖 → 一次归约**。
///
/// `shards` 可以是**任意顺序**（完成顺序、分发顺序、跨机到达顺序都无妨）——
/// 归位键是帧号。结果与 [`reduce`] 的串行参考逐位一致（见模块文档证明）。
///
/// # 参数
/// - `shards`：各分片产物（逐帧值，**不是**块内汇总值）。
/// - `expected`：期望被**恰好覆盖**的帧区间（如影片的 `[start, end]`）。
///   区间外的帧一律报错——这是"合并不重不漏"的可执行定义。
/// - `combine` / `unit`：直接透传给 [`reduce`]（`unit` 用于空区间）。
///
/// # Errors
/// - 分片 `values` 长度与 `frames.len()` 不符 → [`FarmError::ShardLengthMismatch`]；
/// - 归位后覆盖不完整（漏帧 / 重帧）或越出 `expected` →
///   [`FarmError::FrameCoverageBroken`]。
pub fn merge_shards<T, F>(
    shards: &[FrameShard<T>],
    expected: FrameRange,
    combine: F,
    unit: T,
) -> Result<T, FarmError>
where
    T: FrameValue,
    F: Fn(T, T) -> T,
{
    merge_shards_full(shards, expected, combine, unit).map(|(value, _audit)| value)
}

/// [`merge_shards`] 的完整版：额外返回覆盖与树形审计。
pub fn merge_shards_full<T, F>(
    shards: &[FrameShard<T>],
    expected: FrameRange,
    combine: F,
    unit: T,
) -> Result<(T, MergeAudit), FarmError>
where
    T: FrameValue,
    F: Fn(T, T) -> T,
{
    let empty_audit: MergeAudit =
        MergeAudit { frames: 0, tree_depth: reduction_shape(0).depth(), leaf_slots: 0 };
    // 空作业：空集归约 = 单位元（与 reduce 的空集语义一致）。
    if expected.is_empty() {
        return Ok((unit, empty_audit));
    }
    // 帧数 → 槽下标的上界。u64 → usize 收窄用 try_from 而非 `as`（禁截断）。
    let declared: u64 = expected.len();
    let total: usize = usize::try_from(declared).map_err(|_| FarmError::FrameCountOverflow {
        location: "farm::merge_shards".to_string(),
        start: expected.start,
        end: expected.end,
    })?;
    // 槽下标 = 帧号 - base_frame（支持非零起点，如影片第 1000..1099 帧）。
    let base_frame: u64 = expected.start;

    // ---- 步骤 1：分片契约校验（逐帧齐备） ----
    for s in shards {
        let shard_frames: u64 = s.frames.len();
        if s.values.len() as u64 != shard_frames {
            return Err(FarmError::ShardLengthMismatch {
                location: "farm::merge_shards".to_string(),
                start: s.frames.start,
                end: s.frames.end,
                declared: shard_frames,
                values: s.values.len(),
            });
        }
    }

    // ---- 步骤 2：按帧号归位（消序用排序；不看完成顺序） ----
    // 每帧一个槽，槽下标 = 帧号 - base_frame。任一帧被写两次 ⇒ 重帧。
    let mut slots: Vec<Option<T>> = vec![None; total];
    for s in shards {
        // 分片必须整体落在 expected 内（先做无溢出的区间包含判断）。
        if s.frames.start < base_frame || s.frames.end > expected.end {
            return Err(coverage_error(s.frames.start, declared, 0));
        }
        // 帧号 → 槽下标（C22 铁律：禁越界索引，故用 checked_sub + get_mut 判空）。
        let Some(offset) = s.frames.start.checked_sub(base_frame) else {
            return Err(coverage_error(s.frames.start, declared, 0));
        };
        let Ok(base) = usize::try_from(offset) else {
            return Err(coverage_error(s.frames.start, declared, 0));
        };
        if base >= total {
            return Err(coverage_error(s.frames.start, declared, 0));
        }
        // 区间末尾也不得越界（frames.len() 与 values.len() 已在步骤 1 对齐）。
        if base.checked_add(s.values.len()).is_none_or(|end| end > total) {
            return Err(coverage_error(s.frames.start, declared, 0));
        }
        for (i, v) in s.values.iter().enumerate() {
            let Some(slot) = slots.get_mut(base + i) else {
                return Err(coverage_error(s.frames.start, declared, 0));
            };
            // 重帧：同一帧被两个分片产出 ⇒ 报错，不静默覆盖（C20）。
            if slot.is_some() {
                let dup: u64 = base_frame.saturating_add(u64::try_from(base + i).unwrap_or(u64::MAX));
                return Err(coverage_error(dup, declared, 0));
            }
            *slot = Some(v.clone());
        }
    }

    // ---- 步骤 3：不重不漏校验（漏帧即报错，绝不静默补单位元） ----
    let mut ordered: Vec<T> = Vec::with_capacity(total);
    for (i, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(v) => ordered.push(v),
            None => {
                let frame: u64 = base_frame.saturating_add(u64::try_from(i).unwrap_or(u64::MAX));
                return Err(coverage_error(frame, declared, i));
            }
        }
    }

    // ---- 步骤 4：一次归约（形状 = reduction_shape(n)，与拓扑解耦） ----
    let shape: ReductionShape = reduction_shape(ordered.len());
    let audit: MergeAudit = MergeAudit {
        frames: declared,
        tree_depth: shape.depth(),
        leaf_slots: shape.n(),
    };
    let value: T = reduce(ordered, combine, unit);
    Ok((value, audit))
}

fn coverage_error(frame: u64, expected: u64, got: usize) -> FarmError {
    FarmError::FrameCoverageBroken {
        location: "farm::merge_shards".to_string(),
        frame,
        expected,
        got,
    }
}

/// 分层并行归约：按 [`reduction_shape`] 的**层**并行，跨层严格按形状推进。
///
/// 与串行 [`reduce`] **逐位一致**（同树同结合序）：层内各对互不依赖，任意
/// 并行度、任意完成顺序；层间由形状函数给定的结合序强制。线程数只影响吞吐，
/// 不影响结果。
///
/// `unit` 语义与 [`reduce`] 一致：仅在 `items` 为空时返回（空集归约 = 单位元）。
///
/// 这是"归约本身也可并行"的证据：即使把浮点归约摊到多线程，`reduction_shape`
/// 仍是唯一的形状来源（工程四律 1：任何归约点不得私定结合序）。
pub fn reduce_layered_parallel<T, F>(
    items: Vec<T>,
    unit: T,
    workers: usize,
    combine: F,
) -> T
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    let n: usize = items.len();
    // 空集 / 单元素 / 单线程：走串行参考（同树，结果必然一致）。
    if n <= 1 || workers <= 1 {
        return reduce(items, combine, unit);
    }
    let shape: ReductionShape = reduction_shape(n);
    let mut layer: Vec<T> = items;
    for pairs in shape.layers() {
        if layer.len() <= 1 {
            break;
        }
        // ---- 层内并行：各对互不依赖，线程只读 layer、结果按下标回填 ----
        // 回填按下标（而非完成序）⇒ 任意完成顺序给出同一结果。
        // `layer` 在闭包里按引用借用（`&Vec<T>` 是 Sync 因 T: Sync），
        // 循环末尾才整体替换，故借用不跨迭代存活。
        let pairs_slice: &[[u32; 2]] = pairs.as_slice();
        let computed: Vec<Option<T>> = parallel_map(pairs_slice.len(), workers, |i| {
            let pair: [u32; 2] = pairs_slice[i];
            // u32 槽下标 → usize 收窄用 try_from 而非 `as`（禁截断）。
            // 收窄失败与越界同义：都意味着形状与规模脱节，一律标 None，
            // 由下面的"槽数校验"兜住（宁可退回串行参考也不静默改变结果）。
            let (Ok(a_idx), Ok(b_idx)) =
                (usize::try_from(pair[0]), usize::try_from(pair[1]))
            else {
                return None;
            };
            let (Some(a), Some(b)) = (layer.get(a_idx), layer.get(b_idx)) else {
                // 形状与规模同源，理论上不可达；不可达时以 None 标记，由下面的
                // "槽数校验"兜住（宁可退回串行参考也不静默改变结果）。
                return None;
            };
            Some(combine(a.clone(), b.clone()))
        });

        // ---- 跨层推进：按对下标升序取槽（确定性），奇数尾原样进位 ----
        let mut next: Vec<T> = Vec::with_capacity((layer.len() + 1) / 2);
        let mut ok: bool = true;
        for v in computed {
            match v {
                Some(v) => next.push(v),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || next.is_empty() {
            // 内部不自洽：不猜测、不补默认值——退回串行参考（结果仍正确）。
            return reduce(layer, combine, unit);
        }
        // 奇数尾元素直接进位下一层（形状规则）。
        let carry_from: usize = pairs.len() * 2;
        if let Some(tail) = layer.get(carry_from) {
            next.push(tail.clone());
        }
        layer = next;
    }
    // n >= 2 时逐层收敛到恰好 1 个元素；取首个即结果（不用 unwrap）。
    match layer.into_iter().next() {
        Some(v) => v,
        None => unit,
    }
}

/// 层内并行映射：把 `0..n` 的下标按 round-robin 分给 `workers` 个线程，
/// **按下标升序**返回结果（结果序与完成序无关 ⇒ 确定性）。
///
/// 返回 `Vec<Option<T>>`：槽位写入成功为 `Some`，锁中毒 / 映射返回 `None` 则
/// 该槽留 `None`——**上层据此退回串行参考，绝不猜测、绝不补默认值**。
fn parallel_map<T, F>(n: usize, workers: usize, f: F) -> Vec<Option<T>>
where
    T: Send + Sync,
    F: Fn(usize) -> Option<T> + Send + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    if workers <= 1 || n == 1 {
        return (0..n).map(f).collect();
    }
    // 结果槽：Mutex 保护按下标的写入（与 batch.rs 的槽位回填同风格）。
    let slots: Vec<std::sync::Mutex<Option<T>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let slot_refs: Vec<&std::sync::Mutex<Option<T>>> = slots.iter().collect();
    // `&Vec<...>` / `&F`（F 借）为 Copy：循环内 move 的是引用本身，
    // 因此不会 E0382。
    let slots_ref: &Vec<&std::sync::Mutex<Option<T>>> = &slot_refs;
    let f_ref: &F = &f;
    std::thread::scope(|scope| {
        for w in 0..workers {
            let taken: Vec<usize> = (0..n).filter(|i| i % workers == w).collect();
            scope.spawn(move || {
                for i in taken {
                    let v: Option<T> = f_ref(i);
                    if let Some(slot) = slots_ref.get(i) {
                        // 锁中毒（前一线程 panic）时跳过该槽：上层会检出槽数不符
                        // 并退回串行参考，不会产出错误结果。
                        if let Ok(mut guard) = slot.lock() {
                            *guard = v;
                        }
                    }
                }
            });
        }
    });
    // 按下标升序收集（确定性）；锁中毒或未写入的槽以 None 标记，交给上层兜底。
    slots.into_iter().map(|s| s.into_inner().unwrap_or(None)).collect()
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::farm::split::{split_frames, split_frames_shape_aligned};

    /// 确定性 LCG（禁 rand crate）。
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
    }

    /// 确定性"帧载荷"：由帧号 + 种子经 splitmix64 派生（纯整数，无浮点）。
    fn payload(frame: u64, seed: u64) -> u64 {
        let mut z: u64 = seed.wrapping_add(frame.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 浮点载荷：结合序敏感的小数（的反例来源：浮点加法不结合）。
    fn fval(frame: u64) -> f32 {
        0.1 * (frame as f32) + 0.05
    }

    fn cfg() -> ConfigHash {
        ConfigHash::of_bytes(b"farm-u8")
    }

    /// 按给定切分造分片（每个分片带逐帧值）。
    fn shards_for(parts: &[FrameRange], seed: u64) -> Vec<FrameShard<u64>> {
        parts
            .iter()
            .map(|r| {
                let values: Vec<u64> = (r.start..=r.end).map(|f| payload(f, seed)).collect();
                FrameShard::new(*r, values, cfg())
            })
            .collect()
    }

    fn shards_f32(parts: &[FrameRange]) -> Vec<FrameShard<f32>> {
        parts
            .iter()
            .map(|r| {
                let values: Vec<f32> = (r.start..=r.end).map(fval).collect();
                FrameShard::new(*r, values, cfg())
            })
            .collect()
    }

    /// 帧区间 `[0, n-1]`（`n = 0` 时给**空区间** `start > end`，不是 `[0,0]`）。
    fn span(n: u64) -> FrameRange {
        match n {
            0 => FrameRange { start: 1, end: 0 },
            other => FrameRange { start: 0, end: other - 1 },
        }
    }

    /// 串行参考：单分片（覆盖全部帧）走一遍 `reduce`。
    fn serial_u64(n: u64, seed: u64) -> u64 {
        let values: Vec<u64> = (0..n).map(|f| payload(f, seed)).collect();
        merge_shards(&[FrameShard::new(span(n), values, cfg())], span(n), |a, b| {
            a.wrapping_add(b)
        }, 0)
        .expect("串行参考应可归约")
    }

    fn serial_f32(n: u64) -> f32 {
        let values: Vec<f32> = (0..n).map(fval).collect();
        merge_shards(&[FrameShard::new(span(n), values, cfg())], span(n), |a, b| {
            a + b
        }, 0.0_f32)
        .expect("串行参考应可归约")
    }

    #[test]
    fn parallel_merge_matches_serial_reference_across_scales() {
        // 核心一致性判据（"并行 md5 == 串行 md5"）：多规模逐位一致。
        // 规模含非 2 的幂（3 / 7 / 100 / 1000）与 2 的幂（1 / 2 / 8 / 1024）。
        for n in [1u64, 2, 3, 7, 8, 100, 1000, 1024] {
            let parts: Vec<FrameRange> = split_frames(0, n - 1, 8).expect("应可切分");
            let shards: Vec<FrameShard<u64>> = shards_for(&parts, 42);
            let parallel: u64 = merge_shards(&shards, span(n), |a, b| a.wrapping_add(b), 0).expect("应可归约");
            assert_eq!(parallel, serial_u64(n, 42), "n={n}：并行归约必须逐位等于串行参考");
        }
    }

    #[test]
    fn merge_is_independent_of_shard_arrival_order() {
        // 顺序无关：打乱分片到达顺序，结果不变（归位键是帧号不是到达序）。
        let n: u64 = 1024;
        let parts: Vec<FrameRange> = split_frames(0, n - 1, 7).expect("应可切分");
        let shards: Vec<FrameShard<u64>> = shards_for(&parts, 7);
        let canonical: u64 = merge_shards(&shards, span(n), |a, b| a.wrapping_add(b), 0).expect("应可归约");
        let mut rng: Lcg = Lcg::new(0xC0FFEE_1234_5678);
        for _ in 0..50 {
            let mut shuffled: Vec<FrameShard<u64>> = shards.clone();
            // Fisher–Yates（确定性 LCG，不用 rand crate）。
            for i in (1..shuffled.len()).rev() {
                let j: usize = (rng.next_u64() % (i as u64 + 1)) as usize;
                shuffled.swap(i, j);
            }
            let got: u64 = merge_shards(&shuffled, span(n), |a, b| a.wrapping_add(b), 0).expect("应可归约");
            assert_eq!(got, canonical, "分片到达顺序不得影响归约结果");
        }
    }

    #[test]
    fn float_merge_is_bitwise_identical_to_serial_reference() {
        // 浮点结合序敏感场景：逐位（to_bits）一致，不是"近似相等"。
        for n in [3u64, 5, 7, 8, 37, 100, 1000, 1024] {
            let parts: Vec<FrameRange> = split_frames(0, n - 1, 5).expect("应可切分");
            let parallel: f32 = merge_shards(&shards_f32(&parts), span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
            assert_eq!(
                parallel.to_bits(),
                serial_f32(n).to_bits(),
                "n={n}：浮点并行归约必须与串行参考逐位一致（f32 位模式）"
            );
        }
    }

    #[test]
    fn shape_aligned_split_also_merges_bitwise_identically() {
        // 形状对齐切分（块边界钉在归约树槽边界上）同样逐位一致。
        for n in [8u64, 100, 1000, 1024] {
            let parts: Vec<FrameRange> = split_frames_shape_aligned(0, n - 1, 4).expect("应可切分");
            let parallel: f32 = merge_shards(&shards_f32(&parts), span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
            assert_eq!(
                parallel.to_bits(),
                serial_f32(n).to_bits(),
                "n={n}：形状对齐切分 + 归约必须逐位一致"
            );
        }
    }

    #[test]
    fn merge_audit_tree_shape_is_pure_function_of_frame_count() {
        // 归约树形状 = 帧数的纯函数（核心构造）：审计字段可观测。
        for n in [1u64, 2, 3, 7, 8, 100, 1000, 1024] {
            let parts: Vec<FrameRange> = split_frames(0, n - 1, 3).expect("应可切分");
            let (_value, audit): (u64, MergeAudit) =
                merge_shards_full(&shards_for(&parts, 1), span(n), |a, b| a.wrapping_add(b), 0).expect("应可归约");
            assert_eq!(audit.frames, n, "审计帧数须等于总帧数");
            assert_eq!(audit.leaf_slots, n as usize, "叶槽数 == n（形状是 n 的纯函数）");
            assert_eq!(audit.tree_depth, reduction_shape(n as usize).depth(), "树深须等于 ⌈log₂ n⌉");
            // 形状只依赖 n：换切分方式审计不变。
            let aligned: Vec<FrameRange> = split_frames_shape_aligned(0, n - 1, 5).expect("应可切分");
            let (_v2, audit2): (u64, MergeAudit) =
                merge_shards_full(&shards_for(&aligned, 1), span(n), |a, b| a.wrapping_add(b), 0).expect("应可归约");
            assert_eq!(audit, audit2, "审计（树形）只依赖帧数，与切分方式无关");
        }
    }

    #[test]
    fn merge_rejects_missing_and_duplicate_coverage() {
        // 漏帧 → P5005（不静默补单位元）。
        let n: u64 = 10;
        let parts: Vec<FrameRange> = split_frames(0, n - 1, 2).expect("应可切分");
        let mut shards: Vec<FrameShard<u64>> = shards_for(&parts, 1);
        shards.truncate(1); // 丢掉第二块 ⇒ 帧 5..9 缺失
        let err: FarmError = merge_shards(&shards, span(n), |a, b| a.wrapping_add(b), 0).expect_err("应报错");
        assert!(matches!(err, FarmError::FrameCoverageBroken { .. }), "实际 {err:?}");
        assert!(err.to_string().contains("P5005"));

        // 重帧 → P5005（不静默覆盖）。
        let dup: Vec<FrameShard<u64>> = vec![
            FrameShard::new(FrameRange { start: 0, end: 4 }, (0..=4).map(|f| payload(f, 1)).collect(), cfg()),
            FrameShard::new(FrameRange { start: 3, end: 9 }, (3..=9).map(|f| payload(f, 1)).collect(), cfg()),
        ];
        let err2: FarmError = merge_shards(&dup, span(n), |a, b| a.wrapping_add(b), 0).expect_err("应报错");
        assert!(matches!(err2, FarmError::FrameCoverageBroken { .. }), "实际 {err2:?}");
    }

    #[test]
    fn merge_rejects_shard_length_mismatch() {
        // 分片契约破坏 → P5006。
        let bad: Vec<FrameShard<u64>> = vec![FrameShard::new(
            FrameRange { start: 0, end: 9 },
            vec![1, 2, 3], // 只给了 3 个值，声明 10 帧
            cfg(),
        )];
        let err: FarmError = merge_shards(&bad, span(10), |a, b| a.wrapping_add(b), 0).expect_err("应报错");
        assert!(matches!(err, FarmError::ShardLengthMismatch { .. }), "实际 {err:?}");
        assert!(err.to_string().contains("P5006"));
    }

    #[test]
    fn empty_job_reduces_to_unit_without_panic() {
        // 空作业：空集归约 = 单位元（与 reduce 的空集语义一致）。
        let got: u64 = merge_shards::<u64, _>(&[], span(0), |a, b| a.wrapping_add(b), 12345).expect("空作业应可归约");
        assert_eq!(got, 12345, "空集归约必须返回单位元");
        // 空作业的审计：零帧、零叶槽、零深度。
        let (_v, audit): (u64, MergeAudit) =
            merge_shards_full::<u64, _>(&[], span(0), |a, b| a.wrapping_add(b), 7).expect("应可归约");
        assert_eq!(audit, MergeAudit { frames: 0, tree_depth: 0, leaf_slots: 0 });
    }

    #[test]
    fn layered_parallel_reduce_matches_serial_bitwise() {
        // 归约本身并行化：层内各对并行、跨层按形状推进 ⇒ 与串行逐位一致。
        for n in [1usize, 2, 3, 7, 8, 100, 1000, 1024] {
            let vals: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32) + 0.05).collect();
            let serial: f32 = reduce(vals.clone(), |a, b| a + b, 0.0_f32);
            for workers in [1usize, 2, 3, 8, 16] {
                let par: f32 = reduce_layered_parallel(vals.clone(), 0.0_f32, workers, |a, b| a + b);
                assert_eq!(
                    par.to_bits(),
                    serial.to_bits(),
                    "n={n} workers={workers}：分层并行归约必须与串行逐位一致"
                );
            }
        }
        // 空集返回单位元（与 reduce 同约定）。
        let empty: f32 = reduce_layered_parallel(Vec::<f32>::new(), 2.5_f32, 4, |a, b| a + b);
        assert_eq!(empty.to_bits(), 2.5_f32.to_bits());
    }

    #[test]
    fn merge_handles_non_zero_frame_start_offset() {
        // 帧号偏移（如影片第 1000..1099 帧）也必须正确归位。
        let start: u64 = 1000;
        let n: u64 = 100;
        let parts: Vec<FrameRange> = split_frames(start, start + n - 1, 4).expect("应可切分");
        let merged: u64 = merge_shards(&shards_for(&parts, 9), FrameRange { start, end: start + n - 1 }, |a, b| a.wrapping_add(b), 0)
            .expect("应可归约");
        let expected: u64 = (0..n).fold(0u64, |acc, f| acc.wrapping_add(payload(start + f, 9)));
        assert_eq!(merged, expected, "带偏移的帧号必须按序归位");
    }
}
