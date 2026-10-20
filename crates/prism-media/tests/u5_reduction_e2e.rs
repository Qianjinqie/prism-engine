//! 端到端逐位一致性证据（固定形状归约代数）。
//!
//! 本文件是归约代数作为**独立子项**的端到端判据：证明"同输入逐字节可复现"在本
//! 仓的多条归约路径上**真的成立**，且证明"禁止块内预归约"这条纪律**真的
//! 有效**（反向测试）。
//!
//! # 三条归约路径（同一个数学对象：`reduction_shape(n)` 平衡 pairwise 树）
//!
//! 1. **直接路径**：`prism_core::graph::reduce`（串行参考 = 对同一形状树的
//!  固定顺序求值）；
//! 2. **音频路径**：`prism_media::nle::audio::mixdown`（多轨混音复用同一
//!    `reduce`，逐帧归约）；
//! 3. **农场路径**：`prism_render::farm::{split_frames, merge_shards}`（分布式
//!    分布式调度；worker 只交逐帧载荷，主节点按帧号归位后**一次** `reduce`）。
//!
//! 判据：三条路径对**等价输入**给出**逐位相同**的结果（`to_bits()`）。
//!
//! # 反向测试（本路核心证据）
//!
//! 故意让"worker"在块内预归约成一个值再交给主节点 → 主节点看到的是 k 个块值
//! ⇒ 走 `reduction_shape(k)` 而非 `reduction_shape(n)` ⇒ **结果必须不同**。
//! 若反向测试反而"相同"，则这条纪律形同虚设——本文件把它变成可执行断言。

use prism_core::graph::{ceil_log2, error_bound_coeff, reduce, reduction_shape};
use prism_media::nle::audio::{mixdown, AudioTrackInput};
use prism_media::nle::TrackIndex;
use prism_audio::PcmBuffer;
use prism_render::farm::{merge_shards, split_frames, FrameShard, ConfigHash};
use prism_render::incremental::FrameRange;

/// 确定性 LCG（禁第三方 rand crate，铁律一）。
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

/// 结合序敏感的浮点载荷：`0.1 * frame + 0.05`（与 farm 单测同式——末位漂移
/// 敏感，是"浮点加法不结合"的放大镜）。
fn fval(frame: u64) -> f32 {
    0.1 * (frame as f32) + 0.05
}

/// 帧区间 `[0, n-1]`（n=0 时给空区间 `start > end`，与 farm 约定一致）。
fn span(n: u64) -> FrameRange {
    match n {
        0 => FrameRange { start: 1, end: 0 },
        other => FrameRange { start: 0, end: other - 1 },
    }
}

fn cfg() -> ConfigHash {
    ConfigHash::of_bytes(b"u5-e2e")
}

// ---------------------------------------------------------------------------
// 1. 三条路径对同一序列逐位一致（决定性验证）
// ---------------------------------------------------------------------------

#[test]
fn three_reduction_paths_agree_bitwise_for_same_sequence() {
    // 农场路径与直接 reduce 对同一组逐帧值逐位一致（跨规模，含非 2 的幂）。
    for n in [1u64, 2, 3, 5, 7, 8, 37, 100, 1000, 1024] {
        // 直接路径：串行参考。
        let vals: Vec<f32> = (0..n).map(fval).collect();
        let direct: f32 = reduce(vals.clone(), |a, b| a + b, 0.0_f32);

        // 农场路径：切分（任意块数）→ worker 交逐帧载荷 → 主节点归位一次 reduce。
        for workers in [1usize, 2, 3, 5, 8] {
            let parts: Vec<FrameRange> = split_frames(0, n - 1, workers).expect("应可切分");
            let shards: Vec<FrameShard<f32>> = parts
                .iter()
                .map(|r| {
                    let values: Vec<f32> = (r.start..=r.end).map(fval).collect();
                    FrameShard::new(*r, values, cfg())
                })
                .collect();
            let via_farm: f32 =
                merge_shards(&shards, span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
            assert_eq!(
                direct.to_bits(),
                via_farm.to_bits(),
                "n={n} workers={workers}：农场归约须与直接 reduce 逐位一致"
            );
        }
    }
}

#[test]
fn farm_arrival_order_does_not_change_bits() {
    // 到达顺序无关（律三"消序用排序"）：打乱分片顺序，逐位不变。
    let n: u64 = 512;
    let parts: Vec<FrameRange> = split_frames(0, n - 1, 8).expect("应可切分");
    let shards: Vec<FrameShard<f32>> = parts
        .iter()
        .map(|r| {
            let values: Vec<f32> = (r.start..=r.end).map(fval).collect();
            FrameShard::new(*r, values, cfg())
        })
        .collect();
    let canonical: f32 =
        merge_shards(&shards, span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
    let mut rng = Lcg::new(0xDEAD_BEEF_1234_5678);
    for _ in 0..30 {
        let mut shuffled = shards.clone();
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let got: f32 =
            merge_shards(&shuffled, span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
        assert_eq!(got.to_bits(), canonical.to_bits(), "分片到达顺序不得影响结果");
    }
}

// ---------------------------------------------------------------------------
// 2. 音频路径：mixdown 复用同一 reduce，逐位可复现
// ---------------------------------------------------------------------------

#[test]
fn audio_mixdown_uses_the_same_bitwise_discipline() {
    // 三轨不同值：两遍逐位一致（单轨自身归约 = reduce(n=1) 退化情形）。
    let mono = |data: Vec<f32>| PcmBuffer::from_mono(8_000, data).expect("缓冲应可构造");
    let a = AudioTrackInput::new(mono((0..64).map(|k| 0.01 * k as f32).collect()), 1.0);
    let b = AudioTrackInput::new(mono((0..64).map(|k| -0.005 * k as f32).collect()), 0.8);
    let c = AudioTrackInput::new(mono((0..64).map(|k| 0.02 * k as f32 - 0.3).collect()), 0.6);
    let first = mixdown(&[a.clone(), b.clone(), c.clone()], &[], false).expect("应可混音");
    for _ in 0..200 {
        let got = mixdown(&[a.clone(), b.clone(), c.clone()], &[], false).expect("应可混音");
        assert_eq!(
            got.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            first.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "音频 mixdown 两遍必须逐位一致"
        );
    }
}

#[test]
fn audio_mixdown_matches_direct_reduce_per_frame() {
    // 端到端交叉验证：mixdown 的第 f 帧第 0 声道结果，必须等于"把该帧各轨贡献
    // 收集为序列后走直接 reduce"的结果——证明 mixdown 确实是同一归约。
    let mono = |data: Vec<f32>| PcmBuffer::from_mono(16_000, data).expect("缓冲应可构造");
    let frames = 32usize;
    let tracks: Vec<AudioTrackInput> = (0..5)
        .map(|i| {
            let data: Vec<f32> = (0..frames).map(|k| (k as f32 + 1.0) * (i as f32 + 1.0) * 1e-3).collect();
            AudioTrackInput::new(mono(data), 0.5)
        })
        .collect();
    let out = mixdown(&tracks, &[], false).expect("应可混音");

    // 手工重建每帧贡献（增益 0.5、无淡入淡出/静音/独奏）。
    for f in 0..frames {
        let contributions: Vec<f32> = tracks
            .iter()
            .map(|t| {
                let base = f; // 单声道 ⇒ base = f * 1
                t.buffer.as_slice().get(base).copied().unwrap_or(0.0) * 0.5
            })
            .collect();
        let expect: f32 = reduce(contributions, |x, y| x + y, 0.0_f32);
        let got = out.sample(f, 0).expect("样本应存在");
        assert_eq!(
            got.to_bits(),
            expect.to_bits(),
            "帧 {f}：mixdown 须逐位等于对该帧贡献的直接 reduce"
        );
    }
}

#[test]
fn audio_mixdown_solo_selection_is_order_independent_bits() {
    // 独奏集顺序不影响结果（solo 判定是集合成员判断，无顺序依赖）。
    let mono = |data: Vec<f32>| PcmBuffer::from_mono(8_000, data).expect("缓冲应可构造");
    let tracks: Vec<AudioTrackInput> = (0..4)
        .map(|i| AudioTrackInput::new(mono(vec![0.1 * (i as f32 + 1.0); 16]), 1.0))
        .collect();
    let out1 = mixdown(&tracks, &[TrackIndex(1), TrackIndex(3)], false).expect("应可混音");
    let out2 = mixdown(&tracks, &[TrackIndex(3), TrackIndex(1)], false).expect("应可混音");
    assert_eq!(
        out1.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out2.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "独奏集顺序不得影响混音结果"
    );
}

// ---------------------------------------------------------------------------
// 3. 反向测试（本路核心证据）：块内预归约 → 结果必须不同
// ---------------------------------------------------------------------------

#[test]
fn block_prereduction_in_farm_path_changes_result_and_discipline_is_real() {
    // 反向测试（农场语境）：正确路径让 worker 交**逐帧值**，主节点一次 reduce；
    // 违规路径让 worker 在块内先把本块**预归约成 1 个值**，主节点只 reduce 块值。
    // 判据：两者**逐位不同**（证明纪律真的有效）。
    let n: u64 = 5;
    // 实测反例值（对 `split_frames(0,4,2)` = `[0..2],[3..4]` 两分块实测反例）：
    // 逐帧归约得 1.8531088e7，块内预归约得 1.8531090e7（相差 2，远超 ulp）。
    let vals: Vec<f32> = vec![162.67969, -0.088718414, -2449.0625, 1.8533376e7, -0.040593147];

    // 正确路径（纪律）：分片携带逐帧值。
    let parts: Vec<FrameRange> = split_frames(0, n - 1, 2).expect("应可切分");
    assert_eq!(parts.len(), 2, "两分块");
    let shards: Vec<FrameShard<f32>> = parts
        .iter()
        .map(|r| {
            let values: Vec<f32> = (r.start..=r.end).map(|f| vals[f as usize]).collect();
            FrameShard::new(*r, values, cfg())
        })
        .collect();
    let disciplined: f32 =
        merge_shards(&shards, span(n), |a, b| a + b, 0.0_f32).expect("应可归约");

    // 违规路径：worker 块内预归约成 1 值，主节点 reduce 2 个块值。
    let block_values: Vec<f32> = parts
        .iter()
        .map(|r| {
            let seg: Vec<f32> = (r.start..=r.end).map(|f| vals[f as usize]).collect();
            reduce(seg, |a, b| a + b, 0.0_f32)
        })
        .collect();
    let violating: f32 = reduce(block_values, |a, b| a + b, 0.0_f32);

    // 断言：结果必须不同（否则纪律无效）。
    assert_ne!(
        disciplined.to_bits(),
        violating.to_bits(),
        "反向测试失败：块内预归约竟与逐帧归约同结果——纪律形同虚设"
    );
    assert_eq!(disciplined.to_bits(), 1.8531088e7_f32.to_bits(), "纪律路径保序");
    assert_eq!(violating.to_bits(), (1.8531088e7_f32 + 2.0).to_bits(), "违规路径丢末位");

    // 形状根据：n=5 与 n=2 是**不同的树**（键不同）——原因钉在形状层。
    assert_ne!(
        reduction_shape(5).shape_key(),
        reduction_shape(2).shape_key(),
        "n=5 与 n=2 必为不同树"
    );
}

#[test]
fn block_prereduction_differs_for_many_partitions() {
    // 反向测试的加强版：对多种分块数，块内预归约都**不等**于逐帧归约
    // （只要树形不同）。给出至少一个逐位不同的具体见证。
    let n: u64 = 5;
    let vals: Vec<f32> = vec![162.67969, -0.088718414, -2449.0625, 1.8533376e7, -0.040593147];
    let disciplined: f32 = reduce(vals.clone(), |a, b| a + b, 0.0_f32);
    let mut any_diff = false;
    for workers in [2usize, 3, 4] {
        let parts: Vec<FrameRange> = split_frames(0, n - 1, workers).expect("应可切分");
        let block_values: Vec<f32> = parts
            .iter()
            .map(|r| {
                let seg: Vec<f32> = (r.start..=r.end).map(|f| vals[f as usize]).collect();
                reduce(seg, |a, b| a + b, 0.0_f32)
            })
            .collect();
        let violating: f32 = reduce(block_values, |a, b| a + b, 0.0_f32);
        if violating.to_bits() != disciplined.to_bits() {
            any_diff = true;
        }
    }
    assert!(any_diff, "至少一种块内预归约分块必须使结果不同（否则纪律无效）");
}

// ---------------------------------------------------------------------------
// 4. 形状契约的跨规模一致性（核心构造的可观测面）
// ---------------------------------------------------------------------------

#[test]
fn shape_depth_is_ceil_log2_and_key_is_pure_function_of_n() {
    for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 100, 1000, 1024, 4096] {
        let s = reduction_shape(n);
        assert_eq!(s.depth(), ceil_log2(n), "n={n}：树深 = ⌈log₂n⌉");
        assert_eq!(s.leaf_slots(), n, "n={n}：叶槽 = n");
        assert_eq!(s.combine_count(), n.saturating_sub(1), "n={n}：结合次数 = n−1");
        // 键是 n 的纯函数：同 n 两遍构造必同键。
        assert_eq!(s.shape_key(), reduction_shape(n).shape_key(), "n={n}：同 n 必同形状键");
    }
    // 不同 n（含 ⌈log₂n⌉ 相同者）形状键必须不同。
    assert_ne!(reduction_shape(4).shape_key(), reduction_shape(5).shape_key());
    assert_ne!(reduction_shape(7).shape_key(), reduction_shape(8).shape_key());
}

#[test]
fn pairwise_error_bound_holds_for_farm_merge() {
    //  树深的数值推论：pairwise 求和误差 ≤ ⌈log₂n⌉·u·Σ|aᵢ|/(1−⌈log₂n⌉·u)。
    // 用 f64 参考值对拍农场归约的 f32 结果（浮点给证）。
    let u_f32 = (f32::EPSILON as f64) / 2.0;
    for n in [3u64, 5, 8, 37, 100, 1000] {
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let s = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                s * (1.0 + i as f32 * 0.001) - 1.0 + i as f32 * 1e-4
            })
            .collect();
        let parts: Vec<FrameRange> = split_frames(0, n - 1, 4).expect("应可切分");
        let shards: Vec<FrameShard<f32>> = parts
            .iter()
            .map(|r| {
                let values: Vec<f32> = (r.start..=r.end).map(|f| vals[f as usize]).collect();
                FrameShard::new(*r, values, cfg())
            })
            .collect();
        let got: f32 =
            merge_shards(&shards, span(n), |a, b| a + b, 0.0_f32).expect("应可归约");
        let exact: f64 = vals.iter().map(|&v| v as f64).sum();
        let abs_sum: f64 = vals.iter().map(|&v| (v as f64).abs()).sum();
        let err: f64 = (got as f64 - exact).abs();
        let coeff = error_bound_coeff(n as usize) as f64;
        let bound: f64 = coeff * u_f32 * abs_sum / (1.0 - coeff * u_f32);
        assert!(err <= bound, "n={n}：误差 {err:e} 超过上界 {bound:e}（coeff={coeff}）");
    }
}

#[test]
fn same_n_different_data_walks_same_tree() {
    // 形状与数据解耦：同 n 的两组完全不同数据，reduce 与"按 shape 树手工求值"
    // 逐位一致（证明结合序完全由 n 决定，与数据无关）。
    let set_a: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let set_b: Vec<f32> = vec![1e7, -3.5, 2.25, 1e-7, 8.0];
    let manual = |v: &[f32]| -> f32 {
        let s = reduction_shape(v.len());
        let mut layer: Vec<f32> = v.to_vec();
        for pairs in s.layers() {
            let mut next: Vec<f32> = Vec::new();
            for p in pairs {
                next.push(layer[p[0] as usize] + layer[p[1] as usize]);
            }
            let carry_from = pairs.len() * 2;
            next.extend_from_slice(&layer[carry_from..]);
            layer = next;
        }
        layer[0]
    };
    assert_eq!(reduce(set_a.clone(), |a, b| a + b, 0.0).to_bits(), manual(&set_a).to_bits());
    assert_eq!(reduce(set_b.clone(), |a, b| a + b, 0.0).to_bits(), manual(&set_b).to_bits());
}

#[test]
fn empty_and_single_item_reductions_are_unit_and_identity() {
    // 边界：空集 → 单位元；单元素 → 原样（退化情形）。
    assert_eq!(reduce(Vec::<f32>::new(), |a, b| a + b, 2.5_f32).to_bits(), 2.5_f32.to_bits());
    assert_eq!(reduce(vec![7.5_f32], |a, b| a + b, 0.0).to_bits(), 7.5_f32.to_bits());
    // 农场空作业 → 单位元（不静默补值）。
    let got: u64 = merge_shards::<u64, _>(&[], span(0), |a, b| a.wrapping_add(b), 999)
        .expect("空作业应可归约");
    assert_eq!(got, 999);
}
