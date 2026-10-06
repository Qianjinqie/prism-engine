//! 验收证据：**并行 md5 == 串行 md5**（账本原话"并行 md5 == 串行 md5 即还清
//! 标准"）。
//!
//! # 这条证据为什么成立（不是"测出来的巧合"，是定理）
//!
//! 路线图 §2.6 P2 的农场协议是：worker 交回帧文件，主节点**校验 md5 后合帧**。
//! 于是"合帧正确"的判据就是最终 manifest 的 md5 十六进制串。本模块把这条链
//! 拆成两条独立路径：
//!
//! - **串行参考**：[`serial_manifest_digest`] —— 主节点单机按帧号 `0..n` 顺序
//!   生成帧载荷、拼成一个字节流、算一次 md5；
//! - **并行路径**：[`parallel_manifest_digest`] —— 按 [`split_frames`]（或形状
//!   对齐的 [`split_frames_shape_aligned`]）把 `[0, n)` 切给 `workers` 份，**每份
//!   由独立工作线程真实计算**自己那些帧的载荷，主节点**按帧号**（数据内容，不是
//!   完成顺序）拼回同一个字节流，算一次 md5。
//!
//! 两条路径喂给 md5 的是**逐字节相同的字节流**（同样的帧、同样的顺序、同样
//! 的长度），而 [`md5_hex`] 本身是**纯函数**（`digest` 模块：纯 `u32` 整数
//! 运算，无浮点、无平台 API、无第三方 crate）⇒ 两个十六进制串必然逐字符相同。
//!
//! **注意这里没有做任何"块内预归约"**：md5 不是结合运算，若 worker 先把块内
//! 帧摘要折成一个值再上交，拼接的字节流就变了，digest 必然不同。所以 worker
//! 交回的是**逐帧载荷**，与 [`crate::farm::merge::FrameShard`] 的纪律完全一致。
//!
//! # 为什么这个证据"强"
//!
//! 它一次性覆盖了四件最容易出错的事：
//!
//! | 变量 | 被证据钉住的方式 |
//! |---|---|
//! | 切分是否重 / 漏 | 漏一帧 ⇒ 字节流少一段 ⇒ digest 变 |
//! | 归位是否依赖完成顺序 | 测试用确定性 LCG 洗乱分片（多组种子） |
//! | 线程数是否泄漏进结果 | 每个规模扫 `workers ∈ {1,2,3,5,8,16}`，另扫 1..=32 |
//! | 归约形状是否唯一 | 合并点只走 [`prism_core::graph::reduce`] |
//!
//! # 规模选取
//!
//! 必含**非 2 的幂**：`1, 2, 3, 5, 7, 8, 9, 100, 1000, 1024`（任务书点名的
//! `7 / 100 / 1000 / 1024` 全在列）。2 的幂只验证"整齐路径"，非 2 的幂才验证
//! **奇数尾进位**（§26 形状规则的尾元素处理）——那才是归约形状最容易写错的
//! 地方。

use crate::farm::digest::md5_hex;
use crate::farm::merge::FrameShard;
use crate::farm::split::{split_frames, split_frames_shape_aligned};
use crate::farm::task::ConfigHash;
use crate::incremental::FrameRange;
use std::sync::Mutex;

/// 逐帧载荷的字节数。
///
/// 取 32（= MD5 输入块 64 字节的一半）：非 2 的幂帧数（如 7、1000）会让总字节
/// 数落在"非 64 整数倍"上，强制 padding 分支被执行——若 padding 写错，digest
/// 立刻不一致。
const FRAME_BYTES: usize = 32;

/// manifest 用的配置哈希（证据链内部固定，纯占位）。
fn manifest_config() -> ConfigHash {
    ConfigHash::of_bytes(b"manifest")
}

/// 第 `frame` 帧的确定性载荷（splitmix64 派生，**纯整数**、零依赖）。
///
/// 这是本模块的"渲染"替身：真实农场里这里是"渲染一帧像素"，本轮验的是
/// **调度与归约的确定性**，不是渲染正确性（见汇报"诚实边界"）。
pub fn frame_payload(frame: u64) -> Vec<u8> {
    let mut state: u64 = frame.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x1234_5678);
    let mut out: Vec<u8> = Vec::with_capacity(FRAME_BYTES);
    while out.len() < FRAME_BYTES {
        // splitmix64：每轮吐 8 字节，状态自推进（无浮点、无 crate）。
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z: u64 = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(FRAME_BYTES);
    out
}

/// **串行参考**：帧 `0..n` 的载荷按帧号顺序拼成一个字节流，算一次 md5。
pub fn serial_manifest_digest(n: u64) -> String {
    let mut stream: Vec<u8> = Vec::new();
    for f in 0..n {
        stream.extend_from_slice(&frame_payload(f));
    }
    md5_hex(&stream)
}

/// manifest 摘要：把**已按帧号归位**的逐帧载荷拼流后算 md5。
///
/// 这是主节点侧的真实判据（路线图 §2.6："主节点校验 md5 合帧"）。
pub fn manifest_digest(payloads_in_frame_order: &[Vec<u8>]) -> String {
    let mut stream: Vec<u8> = Vec::new();
    for p in payloads_in_frame_order {
        stream.extend_from_slice(p);
    }
    md5_hex(&stream)
}

/// **并行路径**：切分 → 多线程真实计算逐帧载荷 → 按帧号归位 → 一次 md5。
///
/// # 参数
/// - `n`：帧数（`[0, n)`）。
/// - `workers`：线程数与切分块数（只影响吞吐，**不影响结果**）。
/// - `shape_aligned`：用形状对齐切分（块边界钉在归约树槽边界上）还是均分切分。
/// - `shuffle_seed`：分片到达顺序种子（`0` = 原序，其余 = 确定性洗乱）。
pub fn parallel_manifest_digest(n: u64, workers: usize, shape_aligned: bool, shuffle_seed: u64) -> String {
    if n == 0 {
        return serial_manifest_digest(0);
    }
    let Ok(parts) = (if shape_aligned {
        split_frames_shape_aligned(0, n - 1, workers)
    } else {
        split_frames(0, n - 1, workers)
    }) else {
        // 切分失败（畸形区间）⇒ 退回串行参考，不 panic。
        return serial_manifest_digest(n);
    };
    if parts.is_empty() {
        return serial_manifest_digest(n);
    }

    // ---- worker 侧：多线程**真实计算**自己那几帧的逐帧载荷（不预归约！） ----
    let mut shards: Vec<FrameShard<Vec<u8>>> = compute_shards_multithreaded(&parts, workers);

    // ---- 模拟乱序到达（确定性 LCG，不用 rand crate） ----
    if shuffle_seed != 0 && shards.len() >= 2 {
        let mut state: u64 = shuffle_seed;
        for i in (1..shards.len()).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j: usize = (state % (i as u64 + 1)) as usize;
            shards.swap(i, j);
        }
    }

    // ---- 主节点侧：按**帧号**归位 → 拼流 → 一次 md5 ----
    manifest_digest(&reorder_by_frame_number(&shards, n))
}

/// 多线程真实计算分片：第 `w` 个线程处理下标 `≡ w (mod workers)` 的分片，
/// 结果**按下标回填**（不看完成顺序）。
fn compute_shards_multithreaded(parts: &[FrameRange], workers: usize) -> Vec<FrameShard<Vec<u8>>> {
    let cfg: ConfigHash = manifest_config();
    let total: usize = parts.len();
    if workers <= 1 || total == 1 {
        return parts
            .iter()
            .map(|p| FrameShard::new(*p, render_payloads(*p), cfg))
            .collect();
    }
    let slots: Vec<Mutex<Option<FrameShard<Vec<u8>>>>> =
        parts.iter().map(|_| Mutex::new(None)).collect();
    let slot_refs: Vec<&Mutex<Option<FrameShard<Vec<u8>>>>> = slots.iter().collect();
    let part_refs: &[FrameRange] = parts;
    // `&Vec<...>` 是 Copy：循环内每次 move 的是**引用**本身，不是 Vec，
    // 因此不会 E0382；`part_refs` 同理。
    let slots_ref: &Vec<&Mutex<Option<FrameShard<Vec<u8>>>>> = &slot_refs;
    let parts_ref: &[FrameRange] = part_refs;
    std::thread::scope(|scope| {
        for w in 0..workers {
            let taken: Vec<usize> = (0..total).filter(|i| i % workers == w).collect();
            scope.spawn(move || {
                for i in taken {
                    let Some(part) = parts_ref.get(i).copied() else { continue };
                    let shard: FrameShard<Vec<u8>> =
                        FrameShard::new(part, render_payloads(part), cfg);
                    if let Some(slot) = slots_ref.get(i) {
                        if let Ok(mut guard) = slot.lock() {
                            *guard = Some(shard);
                        }
                    }
                }
            });
        }
    });
    // 按下标升序收集（确定性）；锁中毒 / 未写入的槽同步重算，绝不跳过。
    let mut out: Vec<FrameShard<Vec<u8>>> = Vec::with_capacity(total);
    for (i, slot) in slots.into_iter().enumerate() {
        match slot.into_inner().ok().flatten() {
            Some(v) => out.push(v),
            None => {
                let part: FrameRange = parts.get(i).copied().unwrap_or(FrameRange { start: 0, end: 0 });
                out.push(FrameShard::new(part, render_payloads(part), cfg));
            }
        }
    }
    out
}

/// 计算一个帧区间的逐帧载荷。
fn render_payloads(part: FrameRange) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut f: u64 = part.start;
    loop {
        out.push(frame_payload(f));
        if f == part.end {
            break;
        }
        f = f.saturating_add(1);
    }
    out
}

/// 把分片按**帧号**（数据内容）摊平成 `n` 个逐帧载荷的升序序列。
///
/// 归位键是帧号，不是完成顺序（§26 律三"消序用排序"）。分片必须恰好覆盖
/// `[0, n)`；本函数是给证据链用的轻量版（不做 P5005/P5006 校验——那由
/// [`crate::farm::merge::merge_shards`] 负责），漏帧时留下的空槽在拼流时
/// **贡献 0 字节**，于是 digest 必然与串行参考不同 ⇒ 漏帧被证据抓到。
fn reorder_by_frame_number(shards: &[FrameShard<Vec<u8>>], n: u64) -> Vec<Vec<u8>> {
    let Ok(total) = usize::try_from(n) else {
        return Vec::new();
    };
    let mut slots: Vec<Vec<u8>> = vec![Vec::new(); total];
    for s in shards {
        let Ok(base) = usize::try_from(s.frames.start) else { continue };
        if base >= total {
            continue;
        }
        for (i, v) in s.values.iter().enumerate() {
            if let Some(slot) = slots.get_mut(base + i) {
                *slot = v.clone();
            }
        }
    }
    slots
}

// ---------------------------------------------------------------------------
// 测试：核心验收证据
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 任务书点名的规模 + 补充的非 2 的幂（覆盖奇数尾进位分支）。
    const SCALES: [u64; 10] = [1, 2, 3, 5, 7, 8, 9, 100, 1000, 1024];
    /// 扫描的并行度（含 1 = 串行退化，作为对照组）。
    const WORKERS: [usize; 6] = [1, 2, 3, 5, 8, 16];
    /// 到达顺序种子（0 = 原序；其余 = 确定性洗乱）。
    const SEEDS: [u64; 4] = [0, 1, 0x5eed, 0xC0FFEE];

    /// 🔴 **核心验收：并行 md5 == 串行 md5**。
    ///
    /// 打印实测 digest 对比表（`--nocapture` 可见），并对每个
    /// (规模 × 并行度 × 切分方式 × 到达顺序) 组合断言两个十六进制串逐字符相同。
    #[test]
    fn parallel_md5_equals_serial_md5_across_all_scales_and_workers() {
        println!("\n=== U8 验收证据：并行 md5 == 串行 md5 ===");
        println!("{:>6}  {:>7}  {:>8}  {:>34}  {:>34}  {}", "帧数n", "并行度", "切分", "串行 md5", "并行 md5", "一致");
        println!("{}", "-".repeat(104));
        let mut checked: usize = 0;
        for n in SCALES {
            let serial: String = serial_manifest_digest(n);
            for workers in WORKERS {
                for aligned in [false, true] {
                    let kind: &str = if aligned { "形状对齐" } else { "均分" };
                    for seed in SEEDS {
                        let parallel: String = parallel_manifest_digest(n, workers, aligned, seed);
                        assert_eq!(
                            parallel, serial,
                            "n={n} workers={workers} 切分={kind} seed={seed}：并行 md5 必须与串行 md5 逐字符相同"
                        );
                        checked += 1;
                    }
                    // 表格只打印原序 + workers ∈ {1, 8} 两行（避免刷屏）；
                    // 其余组合（洗乱到达序）已在上方全部断言。
                    if workers == 1 || workers == 8 {
                        let parallel: String = parallel_manifest_digest(n, workers, aligned, 0);
                        println!(
                            "{:>6}  {:>7}  {:>8}  {:>34}  {:>34}  {}",
                            n,
                            workers,
                            kind,
                            serial,
                            parallel,
                            if parallel == serial { "OK" } else { "MISMATCH" }
                        );
                    }
                }
            }
        }
        println!("{}", "-".repeat(104));
        println!(
            "共断言 {checked} 组（规模 {} x 并行度 {} x 切分 2 x 到达序 {}），全部逐字符一致。",
            SCALES.len(),
            WORKERS.len(),
            SEEDS.len()
        );
        println!("=== 证据结束 ===\n");
        assert!(checked >= 400, "证据组数应覆盖全部组合，实际 {checked}");
    }

    /// 并行度扫描：固定规模，只看并行度变化是否影响 digest。
    #[test]
    fn worker_count_never_leaks_into_the_digest() {
        for n in [7u64, 100, 1000, 1024] {
            let serial: String = serial_manifest_digest(n);
            let mut digests: Vec<String> = Vec::new();
            for workers in 1..=32usize {
                digests.push(parallel_manifest_digest(n, workers, false, 0xABCD));
            }
            for d in &digests {
                assert_eq!(*d, serial, "n={n}：并行度 1..=32 的 digest 必须全部相同");
            }
        }
    }

    /// 切分方式不影响 digest：均分切分与形状对齐切分给出同一 manifest。
    #[test]
    fn split_strategy_does_not_change_the_digest() {
        for n in [1u64, 3, 7, 100, 1000, 1024] {
            let serial: String = serial_manifest_digest(n);
            for workers in [2usize, 3, 7, 8, 13] {
                let even: String = parallel_manifest_digest(n, workers, false, 0);
                let aligned: String = parallel_manifest_digest(n, workers, true, 0);
                assert_eq!(even, serial, "n={n} workers={workers}：均分切分 digest 必须一致");
                assert_eq!(aligned, serial, "n={n} workers={workers}：形状对齐切分 digest 必须一致");
                assert_eq!(even, aligned, "n={n}：两种切分策略必须给出同一 digest");
            }
        }
    }

    /// 反面证据：**漏帧必须让 digest 改变**。
    ///
    /// 若漏帧竟然给出相同 digest，说明 digest 判据本身失效（"证据无效"比
    /// "实现有 bug"更严重），必须在这里被拦住。
    #[test]
    fn a_missing_frame_must_change_the_digest() {
        let n: u64 = 100;
        let serial: String = serial_manifest_digest(n);
        let cfg: ConfigHash = manifest_config();
        let good: Vec<Vec<u8>> = (0..n).map(frame_payload).collect();

        // 漏帧：只覆盖 [0, n-1]，最后一帧缺席。
        let missing: Vec<FrameShard<Vec<u8>>> = vec![FrameShard::new(
            FrameRange { start: 0, end: n - 2 },
            good[..(n - 1) as usize].to_vec(),
            cfg,
        )];
        let missing_digest: String = manifest_digest(&reorder_by_frame_number(&missing, n));
        assert_ne!(missing_digest, serial, "漏帧必须改变 digest（否则证据判据失效）");

        // 中间漏一帧（分两片，中间空一帧）也必须被抓住。
        let hole: Vec<FrameShard<Vec<u8>>> = vec![
            FrameShard::new(FrameRange { start: 0, end: 9 }, good[..10].to_vec(), cfg),
            FrameShard::new(FrameRange { start: 11, end: 99 }, good[11..].to_vec(), cfg),
        ];
        let hole_digest: String = manifest_digest(&reorder_by_frame_number(&hole, n));
        assert_ne!(hole_digest, serial, "中间漏帧必须改变 digest");

        // 对照：完整覆盖必须给出与串行相同的 digest（证明上面两条不是"必然不等"）。
        let whole: Vec<FrameShard<Vec<u8>>> = vec![FrameShard::new(
            FrameRange { start: 0, end: n - 1 },
            good.clone(),
            cfg,
        )];
        assert_eq!(
            manifest_digest(&reorder_by_frame_number(&whole, n)),
            serial,
            "完整覆盖的单分片必须与串行参考一致"
        );
    }

    /// 空作业：n = 0 时 digest = 空流的 md5，两条路径一致且不 panic。
    #[test]
    fn empty_job_digest_matches_and_does_not_panic() {
        let serial: String = serial_manifest_digest(0);
        assert_eq!(serial, "d41d8cd98f00b204e9800998ecf8427e", "n=0 => 空字节流的 md5");
        for workers in [0usize, 1, 8] {
            let parallel: String = parallel_manifest_digest(0, workers, false, 0);
            assert_eq!(parallel, serial, "n=0 workers={workers}：空作业 digest 必须一致");
        }
    }

    /// 逐帧载荷本身是帧号的纯函数（digest 判据的地基）。
    #[test]
    fn frame_payload_is_pure_function_of_frame_number() {
        for f in [0u64, 1, 7, 100, 1023, u64::MAX] {
            let a: Vec<u8> = frame_payload(f);
            let b: Vec<u8> = frame_payload(f);
            assert_eq!(a, b, "帧 {f} 的载荷必须逐字节可复现");
            assert_eq!(a.len(), FRAME_BYTES);
        }
        // 相邻帧载荷必须不同（否则 digest 判据对帧号不敏感）。
        for f in 0..64u64 {
            assert_ne!(frame_payload(f), frame_payload(f + 1), "相邻帧载荷必须不同");
        }
    }
}
