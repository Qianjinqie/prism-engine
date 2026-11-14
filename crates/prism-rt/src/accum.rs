//! 累计缓冲：progressive 累加 + 固定形状归约均值。
//!
//! # 两条均值通道（语义分工）
//! - [`AccumBuffer::running_mean`]：f64 顺序累加（固定像素序、固定 pass 序
//!   ——逐位可复现），供 progressive 预览。
//! - [`AccumBuffer::canonical_mean`] / [`mean_samples`]：prism-core `reduce`
//!  固定形状树（串行参考语义），**md5 锚定的规范输出**。
//! 同一数据两条通道结合序不同、结果可差舍入，但各自逐位可复现；关卡
//! "累计一致性"：push 8 个单样本 pass 的 canonical 均值 ≡ 一次 spp=8 渲染
//! （逐位一致）。
//!
//! # 内存说明
//! B1 CpuRef 保存逐 pass 样本（小分辨率验证用）；规模化流式归约属后续分块
//! 流水线（内存策略另行评审），此处不做。

/// 单帧累计缓冲：保存逐 pass 的 rgba 样本（pass 序 = push 序）。
#[derive(Debug, Clone)]
pub struct AccumBuffer {
    width: u32,
    height: u32,
    passes: Vec<Vec<[f32; 4]>>,
}

impl AccumBuffer {
    /// 构造 w×h 空缓冲。
    pub fn new(width: u32, height: u32) -> Self {
        AccumBuffer { width, height, passes: Vec::new() }
    }

    /// 帧尺寸。
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// 已累计 pass 数（每 pass = 一帧渲染）。
    pub fn pass_count(&self) -> u32 {
        self.passes.len() as u32
    }

    /// 推入一个 pass（整帧 rgba，行主序 w×h）。
    ///
    /// # Panics
    /// 帧长与缓冲尺寸不符时 panic。
    pub fn push_pass(&mut self, frame: &[[f32; 4]]) {
        assert_eq!(
            frame.len(),
            (self.width as usize) * (self.height as usize),
            "pass 帧长与缓冲尺寸不符"
        );
        self.passes.push(frame.to_vec());
    }

    /// 顺序均值（progressive 预览）：f64 顺序累加 / pass 数，固定顺序可复现。
    ///
    /// # Panics
    /// 无 pass 时 panic（先 push 再读）。
    pub fn running_mean(&self, x: u32, y: u32) -> [f32; 4] {
        assert!(!self.passes.is_empty(), "running_mean 前必须先 push_pass");
        let idx = (y as usize) * self.width as usize + x as usize;
        let n = self.passes.len() as f64;
        let mut acc = [0.0f64; 4];
        for p in &self.passes {
            let s = p[idx];
            acc[0] += s[0] as f64;
            acc[1] += s[1] as f64;
            acc[2] += s[2] as f64;
            acc[3] += s[3] as f64;
        }
        [
            (acc[0] / n) as f32,
            (acc[1] / n) as f32,
            (acc[2] / n) as f32,
            (acc[3] / n) as f32,
        ]
    }

    /// 规范均值（md5 锚定输出）：该像素全部 pass 样本经 prism-core reduce
    /// 固定形状树求和再除以样本数。像素级局部归约，tile 无关（禁跨 tile）。
    pub fn canonical_mean(&self, x: u32, y: u32) -> [f32; 4] {
        assert!(!self.passes.is_empty(), "canonical_mean 前必须先 push_pass");
        let idx = (y as usize) * self.width as usize + x as usize;
        let mut col = Vec::with_capacity(self.passes.len());
        for p in &self.passes {
            col.push(p[idx]);
        }
        mean_samples(&col)
    }

    /// 清空已累计的 pass（V3.1 T03 `TemporalState` 窗口复用）。
    ///
    /// 只清 pass 列表，`width` / `height` 保持不变；`running_mean` /
    /// `canonical_mean` / [`mean_samples`] / [`mean_samples_fill`] 的逐位语义
    /// **一律不动**。窗口命中流程：逐帧 `push_pass(cur)` → `pass_count()==W`
    /// 用 `running_mean` 出帧 → `clear()` → 下一窗口重新采（3600 帧不 OOM）。
    pub fn clear(&mut self) {
        self.passes.clear();
    }
}

/// 栈上定长缓冲容量：覆盖常规 spp（16/32/64）；更大 spp 走堆回退。
/// `[f32; 4]` 是纯 POD，栈缓冲按位零初始化，不引入任何隐藏状态。
const MEAN_STACK_CAP: usize = 64;

/// 样本集均值（spp 均值的唯一出口）：固定形状 pairwise 树求和 / 样本数。
/// 空集返回 [0,0,0,0]（单位元）。
///
/// # 性能（B.2 热点，逐位不变）
///
/// 旧实现 `reduce(samples.to_vec(), ..)` 对 spp=16 触发 **5 次堆分配**
/// （1 次 `to_vec` + 逐层 4 次 `Vec::with_capacity`）。本实现复现**完全相同**
/// 的固定形状 pairwise 结合序（相邻成对 `(0,1),(2,3),…`、奇数尾直接进位、
/// 逐层减半），在**单个缓冲上就地压实**：小 spp 用栈上定长数组，零堆分配；
/// 大 spp 用一块堆缓冲，同样零中间分配。
///
/// 结合序与 `prism_core::reduce`（其 `reduction_shape` 是同一规则的唯一权威）
/// **逐句一致**，故输出与旧实现**逐字节一致**——`mean_samples_matches_reduce_bitwise`
/// 对 n=0..80 逐位断言这一点。这不是「测出来一致」，而是同一加法树的两种
/// 存储写法：浮点加法按相同顺序结合 ⇒ 相同舍入 ⇒ 相同位型。
pub fn mean_samples(samples: &[[f32; 4]]) -> [f32; 4] {
    mean_samples_fill(samples.len(), |i| samples[i])
}

/// [`mean_samples`] 的「生产者」变体（B.2 热点）：不要求调用方先把样本物化成
/// 一个切片，而是用 `emit(i)` 现算第 `i` 个样本直接填进归约缓冲，再走**完全相同**
/// 的固定形状 pairwise 树归约。
///
/// 与 `mean_samples(&v)` 逐字节一致：当 `emit(i) = v[i]` 时，缓冲内容按同一
/// 顺序逐项相同，`[f32;4]` 是 `Copy`（逐项赋值 ≡ `copy_from_slice`），归约
/// 结合序不变 ⇒ 同一舍入 ⇒ 同一位型。判据：`mean_samples_fill_matches_slice_bitwise`。
///
/// # 为什么快
/// 逐像素渲染（[`crate::integrator::render_pixel_range`]）原本每像素
/// `Vec::with_capacity(spp)` + 逐样本 `push`，再把整块切片交给
/// [`mean_samples`] **二次拷贝**进它的就地缓冲——每像素一次堆分配、一次
/// 释放、spp 次额外拷贝。本变体让 `emit` 直接写入同一块缓冲：小 spp 全栈、
/// 零堆分配，大 spp 仅一块堆缓冲，且**没有中间 Vec 与二次拷贝**。
/// 归约树、除法、分支全部不变，故不改任何数值与确定性。
#[inline]
pub fn mean_samples_fill<F>(n: usize, mut emit: F) -> [f32; 4]
where
    F: FnMut(usize) -> [f32; 4],
{
    if n == 0 {
        return [0.0; 4];
    }

    // 取一块长度恰为 n 的可变缓冲：n ≤ 容量走栈，否则走一块堆缓冲。
    let mut stack = [[0.0f32; 4]; MEAN_STACK_CAP];
    let mut heap: Vec<[f32; 4]> = Vec::new();
    let buf: &mut [[f32; 4]] = if n <= MEAN_STACK_CAP {
        &mut stack[..n]
    } else {
        heap.resize(n, [0.0; 4]);
        &mut heap[..]
    };
    for i in 0..n {
        buf[i] = emit(i);
    }

    // 固定形状 pairwise 树，就地压实：与 reduce 的层序/层内顺序一致。
    let mut len = n;
    while len > 1 {
        let half = len / 2;
        let mut j = 0usize;
        while j < half {
            let a = buf[2 * j];
            let b = buf[2 * j + 1];
            buf[j] = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
            j += 1;
        }
        if len & 1 == 1 {
            buf[half] = buf[len - 1]; // 奇数尾进位（reduce 的 None 分支）
        }
        len = (len + 1) / 2;
    }
    let s = buf[0];
    let nf = n as f32;
    [s[0] / nf, s[1] / nf, s[2] / nf, s[3] / nf]
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// V3.1 T02：`clear()` 供 T03 窗口复用，清空后重新累计必须与全新缓冲
    /// **逐位一致**（running / canonical 两条通道都查），且尺寸不变。
    #[test]
    fn clear_reuses_window_bitwise() {
        let frames: Vec<Vec<[f32; 4]>> = (0..5u32)
            .map(|k| {
                vec![
                    [k as f32 * 0.125, 1.0 - k as f32 * 0.1, 0.3, 1.0],
                    [0.7, 0.2, k as f32 * 0.005, 1.0],
                ]
            })
            .collect();

        let mut buf: AccumBuffer = AccumBuffer::new(2, 1);
        for f in &frames {
            buf.push_pass(f);
        }
        assert_eq!(buf.pass_count(), 5);
        let r1: [f32; 4] = buf.running_mean(0, 0);
        let c1: [f32; 4] = buf.canonical_mean(1, 0);

        buf.clear();
        assert_eq!(buf.pass_count(), 0, "clear 后窗口必须归零");
        assert_eq!(buf.size(), (2, 1), "clear 不得改尺寸");

        for f in &frames {
            buf.push_pass(f);
        }
        assert_eq!(buf.pass_count(), 5);
        let r2: [f32; 4] = buf.running_mean(0, 0);
        let c2: [f32; 4] = buf.canonical_mean(1, 0);

        let mut fresh: AccumBuffer = AccumBuffer::new(2, 1);
        for f in &frames {
            fresh.push_pass(f);
        }
        let rf: [f32; 4] = fresh.running_mean(0, 0);
        let cf: [f32; 4] = fresh.canonical_mean(1, 0);
        for ch in 0..4 {
            assert_eq!(r1[ch].to_bits(), r2[ch].to_bits(), "running ch={ch}");
            assert_eq!(c1[ch].to_bits(), c2[ch].to_bits(), "canonical ch={ch}");
            assert_eq!(r2[ch].to_bits(), rf[ch].to_bits(), "running vs fresh ch={ch}");
            assert_eq!(c2[ch].to_bits(), cf[ch].to_bits(), "canonical vs fresh ch={ch}");
        }
    }

    /// B.2 硬判据：新的就地归约与 `prism_core::reduce` 固定形状树对 n=0..96
    /// **逐字节一致**（含栈快路径 n≤64 与堆回退 n>64）。这是「优化不改判据」
    /// 的机械保证，不是容差内接近。
    #[test]
    fn mean_samples_matches_reduce_bitwise() {
        // 造一批非平凡样本（含负、大小量级混合，放大结合序差异的可见性）。
        let mk = |i: usize| -> [f32; 4] {
            let s = i as f32;
            [
                (s * 0.37) - 3.1,
                (s * s * 0.001) + 0.5,
                if i % 2 == 0 { 1.0e3 + s } else { -0.125 * s },
                1.0 + s * 0.03125,
            ]
        };
        for n in 0..=96usize {
            let samples: Vec<[f32; 4]> = (0..n).map(mk).collect();
            // 旧口径（唯一权威）：to_vec + reduce 固定形状树。
            let reference = if n == 0 {
                [0.0f32; 4]
            } else {
                let sum = prism_core::reduce(
                    samples.clone(),
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
                    [0.0; 4],
                );
                let nf = n as f32;
                [sum[0] / nf, sum[1] / nf, sum[2] / nf, sum[3] / nf]
            };
            let got = mean_samples(&samples);
            for c in 0..4 {
                assert_eq!(
                    got[c].to_bits(),
                    reference[c].to_bits(),
                    "n={n} ch={c}: 就地归约与 reduce 逐位一致，got={} ref={}",
                    got[c],
                    reference[c]
                );
            }
        }
    }

    /// B.2 硬判据：生产者变体 `mean_samples_fill(n, |i| v[i])` 与切片变体
    /// `mean_samples(&v)` 对 n=0..=128 **逐字节一致**（含栈快路径 n≤64 与堆
    /// 回退 n>64）。覆盖非平凡值（负、大小量级混合）放大结合序差异。
    /// 这是「优化不改判据」的机械保证：同一加法树、同一结合序 ⇒ 同一位型。
    #[test]
    fn mean_samples_fill_matches_slice_bitwise() {
        let mk = |i: usize| -> [f32; 4] {
            let s = i as f32;
            [
                (s * 0.61) - 7.3,
                (s * s * 0.0007) - 0.25,
                if i % 3 == 0 { -5.0e2 + s } else { 0.375 * s },
                0.5 + s * 0.0078125,
            ]
        };
        for n in 0..=128usize {
            let v: Vec<[f32; 4]> = (0..n).map(mk).collect();
            let by_slice = mean_samples(&v);
            let by_fill = mean_samples_fill(n, |i| v[i]);
            for c in 0..4 {
                assert_eq!(
                    by_fill[c].to_bits(),
                    by_slice[c].to_bits(),
                    "n={n} ch={c}: fill 与 slice 逐位一致, got={} want={}",
                    by_fill[c],
                    by_slice[c]
                );
            }
        }
        // 生产者惰性：n=0 时绝不应调用 emit（返回单位元）。
        let mut called = false;
        let empty = mean_samples_fill(0, |_| {
            called = true;
            [1.0; 4]
        });
        assert!(!called, "空集不得调用 emit");
        assert_eq!(empty, [0.0; 4]);
    }

    /// B.2 同进程前后对比：模拟逐像素「分配 Vec + push + 二次拷贝」旧路径 vs
    /// 生产者变体，在同一进程交替计时。只打印，不作阈值判据（避免 flaky）。
    ///
    /// 关键：生产者用 `black_box` 参与运算，防止 LLVM 把纯算术生产者常量折叠
    /// 掉（否则 fill 分支会被优化成 0.3 ns 的假绿 200x）。两条路径执行**完全相同**
    /// 的逐样本工作，唯一差别是「物化到中间 Vec 再拷贝」还是「直接写入归约缓冲」，
    /// 因此测到的差就是每像素那次堆分配 + spp 次额外拷贝的真实开销。
    #[test]
    fn mean_samples_fill_bench_same_process() {
        use std::hint::black_box;
        use std::time::Instant;
        let spp = 16usize;
        // 每个样本都做一次不可折叠的工作（black_box 打断常量传播），模拟真实
        // trace_sample 的「每样本至少有点成本」，但把成本压到能测出分配差异。
        #[inline]
        fn sample(seed: u64, i: usize) -> [f32; 4] {
            let s = black_box(seed ^ (i as u64)) as f32 * 1e-6;
            [s * 0.37 - 3.1, s * s * 0.001 + 0.5, 1.0e3 + s, 1.0 + s * 0.03125]
        }
        let iters = 100_000u32;
        let t_old = Instant::now();
        for k in 0..iters {
            // V3.1：显式 wrapping —— release 下 `*` 本就 wrapping（overflow-checks
            // 关闭），故逐位相同；debug 下避免该计时 bench 因整数溢出 panic。
            let seed = black_box(k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut v: Vec<[f32; 4]> = Vec::with_capacity(spp);
            for i in 0..spp {
                v.push(sample(seed, i));
            }
            black_box(mean_samples(black_box(&v)));
        }
        let old_ns = t_old.elapsed().as_secs_f64() * 1e9 / iters as f64;
        let t_new = Instant::now();
        for k in 0..iters {
            let seed = black_box(k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            black_box(mean_samples_fill(spp, |i| sample(seed, i)));
        }
        let new_ns = t_new.elapsed().as_secs_f64() * 1e9 / iters as f64;
        println!("[fill-bench] spp={spp} 同进程 before(Vec+copy)={old_ns:.2} ns/像素 after(fill)={new_ns:.2} ns/像素 speedup={:.3}x ({:.1}%)",
            old_ns / new_ns, (1.0 - new_ns / old_ns) * 100.0);
    }

    #[test]
    fn mean_samples_accurate_and_deterministic() {
        let v = [0.25f32, 0.5, 0.75, 1.0];
        for n in [1usize, 3, 8, 16, 100] {
            let samples: Vec<[f32; 4]> = vec![v; n];
            let m = mean_samples(&samples);
            for c in 0..4 {
                assert!(
                    (m[c] - v[c]).abs() < 1e-6,
                    "n={n} ch={c}: {} vs {}",
                    m[c],
                    v[c]
                );
            }
        }
        // 确定性：两遍逐位一致。
        let samples: Vec<[f32; 4]> =
            (0..37).map(|i| [i as f32, 1.0, 2.0, 1.0]).collect();
        let a = mean_samples(&samples);
        let b = mean_samples(&samples);
        for c in 0..4 {
            assert_eq!(a[c].to_bits(), b[c].to_bits());
        }
        // 空集 = 单位元。
        assert_eq!(mean_samples(&[]), [0.0; 4]);
    }

    #[test]
    fn accumulation_consistency_push8_equals_single_pass() {
        // 8 个单样本 pass 的 canonical 均值 ≡ 一次 spp=8 渲染的均值（逐位）。
        let values: Vec<[f32; 4]> = (0..8)
            .map(|i| {
                [
                    i as f32 * 0.125,
                    1.0 - i as f32 * 0.1,
                    0.3,
                    1.0,
                ]
            })
            .collect();
        let mut buf = AccumBuffer::new(1, 1);
        for v in &values {
            buf.push_pass(std::slice::from_ref(v));
        }
        assert_eq!(buf.pass_count(), 8);
        let a = buf.canonical_mean(0, 0);
        let b = mean_samples(&values);
        for c in 0..4 {
            assert_eq!(
                a[c].to_bits(),
                b[c].to_bits(),
                "累计一致性破坏 ch={c}"
            );
        }
        // running 通道与同序 f64 参考逐位一致。
        let r = buf.running_mean(0, 0);
        let mut acc = [0.0f64; 4];
        for v in &values {
            for c in 0..4 {
                acc[c] += v[c] as f64;
            }
        }
        let expected: [f32; 4] = [
            (acc[0] / 8.0) as f32,
            (acc[1] / 8.0) as f32,
            (acc[2] / 8.0) as f32,
            (acc[3] / 8.0) as f32,
        ];
        for c in 0..4 {
            assert_eq!(r[c].to_bits(), expected[c].to_bits());
        }
    }

    #[test]
    fn running_and_canonical_agree_within_tolerance() {
        // 两条通道结合序不同：结果只差舍入（1e-6 内），不要求逐位相等。
        let mut buf = AccumBuffer::new(2, 1);
        for k in 0..13u32 {
            let f = vec![
                [k as f32 * 0.01, 0.1, 0.9, 1.0],
                [0.7, 0.2, k as f32 * 0.005, 1.0],
            ];
            buf.push_pass(&f);
        }
        assert_eq!(buf.size(), (2, 1));
        for x in 0..2u32 {
            let r = buf.running_mean(x, 0);
            let c = buf.canonical_mean(x, 0);
            for ch in 0..4 {
                assert!(
                    (r[ch] - c[ch]).abs() < 1e-6,
                    "x={x} ch={ch}: {r:?} vs {c:?}"
                );
            }
        }
    }
}
