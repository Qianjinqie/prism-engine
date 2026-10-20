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

use prism_core::reduce;

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
}

/// 样本集均值（spp 均值的唯一出口）：prism-core `reduce` 固定形状树
/// 逐位可复现；空集返回 [0,0,0,0]（单位元）。
pub fn mean_samples(samples: &[[f32; 4]]) -> [f32; 4] {
    if samples.is_empty() {
        return [0.0; 4];
    }
    let n = samples.len() as f32;
    let sum = reduce(
        samples.to_vec(),
        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
        [0.0; 4],
    );
    [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
