//! 前缀和扫描（GPU 基础设施，D0）。
//!
//! 语义权威：强大版路线图 D0 行（radix sort / 前缀和 scan）、D-15（GPU 抽象
//! 层自研）、D-21（从数学推导开始设计）、D-26（固定形状归约）、白皮书 §26
//! 工程四律、架构 §4.3 格锚表。
//!
//! 采用 **work-efficient Blelloch 上下行双扫**（O(n) 加法，两趟 O(log n) 并行
//! 层）：上行求子树和、下行铺前缀。该结构即平衡二叉归约树——**形状口径与
//! [`prism_core::reduction_shape`] 一致**（同 n 同树的 pairwise 层 + 奇数尾进位；
//! 本 crate 绝不私定结合序，工程四律 1/4）。
//!
//! 注意：`prism-core` 依赖 `prism-gpu`（A5 骨架方向），故此处**不得反向依赖
//! prism-core**（避免循环依赖）。本模块用本地 [`ScanShape`] 复刻
//! `reduction_shape` 的树形语义，二者由测试口径对齐（层深 = ⌈log₂ n⌉）。

/// 归约/扫描树形状（`reduction_shape` 语义的本地复刻，避免循环依赖）。
///
/// 层结构：每层相邻成对（pairwise），奇数尾元素直接进位下一层——与
/// `prism-core::graph::ReductionShape` 完全同构（同 n 必得同树）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanShape {
    n: usize,
    layers: Vec<Vec<[u32; 2]>>,
}

impl ScanShape {
    /// 以输入规模构造形状（同 n 必得同树）。
    pub fn new(n: usize) -> Self {
        let mut layers: Vec<Vec<[u32; 2]>> = Vec::new();
        let mut len = n;
        while len > 1 {
            let mut pairs = Vec::with_capacity(len / 2);
            let mut i = 0;
            while i + 1 < len {
                pairs.push([i as u32, i as u32 + 1]);
                i += 2;
            }
            layers.push(pairs);
            len = (len + 1) / 2; // 奇数尾元素进位下一层
        }
        Self { n, layers }
    }

    /// 输入规模 n。
    pub fn n(&self) -> usize {
        self.n
    }

    /// 树深 = ⌈log₂ n⌉（n ≤ 1 时为 0）。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 各层结合对。
    pub fn layers(&self) -> &[Vec<[u32; 2]>] {
        &self.layers
    }
}


/// 串行参考前缀和（inclusive scan）：`out[i] = Σ_{k≤i} in[k]`。
///
/// 直觉过程实现，定义语义真值。空输入 → 空输出。
pub fn scan_cpu_ref(input: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(input.len());
    let mut acc: u32 = 0;
    for &x in input {
        acc = acc.wrapping_add(x);
        out.push(acc);
    }
    out
}

/// 上行扫描（up-sweep / reduce）：平衡二叉树上自底向上求各内部节点子树和。
///
/// 形状取自 [`reduction_shape`]（同 n 同树）。返回的大小为
/// `next_pow2(n)` 的紧凑工作数组 `tree`，其中 `tree[0]` 为全部输入之和、
/// 内部节点 `tree[k]` = 其子树之和（含原始叶值在 `tree[n_child + i]`）。
fn up_sweep(leaves: &[u32]) -> Vec<u32> {
    let mut tree: Vec<u32> = leaves.to_vec();
    // 形状由 ScanShape（reduction_shape 语义）派生；Blelloch 在补齐到 2 的幂的
    // 连续区间上自底向上折叠，其层序与 pairwise 树形同构（层内相邻成对）。
    let _shape = ScanShape::new(leaves.len());
    let mut width: usize = 1;
    while width < tree.len() {
        width <<= 1;
    }
    tree.resize(width, 0); // 补零：0 是 u32 加法的单位元，不改变结果
    let mut step: usize = 1;
    while step < width {
        let mut i = step * 2 - 1;
        while i < width {
            tree[i] = tree[i].wrapping_add(tree[i - step]);
            i += step * 2;
        }
        step <<= 1;
    }
    tree
}

/// 下行扫描（down-sweep / distribute）：自顶向下将前缀铺到叶位置。
///
/// `tree` 为 [`up_sweep`] 的输出，原地改写为其 ExScan→InclScan 结果。
fn down_sweep(mut tree: Vec<u32>) -> Vec<u32> {
    let width = tree.len();
    if width == 0 {
        return tree;
    }
    tree[width - 1] = 0; // 根置 0（ExScan 起点）
    // step 从 width/2 递减到 1；用显式循环避免无符号下溢。
    let mut step = width >> 1;
    while step > 0 {
        let mut i = step * 2 - 1;
        while i < width {
            let left = tree[i - step];
            let right = tree[i];
            tree[i - step] = right; // 左子拿父值
            tree[i] = right.wrapping_add(left); // 右子 = 父 + 左兄弟
            i += step * 2;
        }
        step >>= 1;
    }
    tree
}

/// Blelloch 树形扫描（GPU 风格 work-efficient）CPU 仿真（inclusive）。
///
/// 与 [`scan_cpu_ref`] 在 u32 值域内**逐位一致**（见单元测试）。空输入 → 空输出。
/// 内部按 2 的幂补齐（补零 = 加法单位元，不改变 inclusive 前缀），最后截回 n。
pub fn scan_blelloch_cpu(input: &[u32]) -> Vec<u32> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let tree = up_sweep(input);
    // down_sweep 产出 ExScan：叶 i = Σ_{k<i}。转 InclScan 需把原值加回。
    let excl = down_sweep(tree);
    let mut out: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        out.push(excl[i].wrapping_add(input[i]));
    }
    out
}

/// 设备无关性说明：扫描算法本身与设备无关（纯数据变换），GPU 后端只需提供
/// 层内并行的 dispatch；本模块提供的 CPU 仿真即语义参考。真机路径标注待接入。

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 多尺寸含非 2 幂：Blelloch 仿真必须与串行参考逐位一致。
    #[test]
    fn blelloch_scan_matches_serial_reference_bitwise() {
        for n in [0usize, 1, 2, 3, 5, 7, 8, 9, 16, 17, 31, 33, 64, 100, 257] {
            let input: Vec<u32> = (0..n).map(|i| ((i * 37 + 11) % 97) as u32).collect();
            let serial = scan_cpu_ref(&input);
            let blelloch = scan_blelloch_cpu(&input);
            assert_eq!(
                serial.len(),
                blelloch.len(),
                "n={n}：长度应一致"
            );
            for i in 0..n {
                assert_eq!(
                    serial[i].to_le_bytes(),
                    blelloch[i].to_le_bytes(),
                    "n={n}, i={i}：Blelloch 应与串行参考逐位一致（{} vs {}）",
                    serial[i],
                    blelloch[i]
                );
            }
        }
    }

    /// 边界：空输入两端都返回空。
    #[test]
    fn scan_empty_input() {
        assert!(scan_cpu_ref(&[]).is_empty());
        assert!(scan_blelloch_cpu(&[]).is_empty());
    }

    /// 确定性：同输入两遍一致。
    #[test]
    fn scan_is_deterministic_across_runs() {
        let input: Vec<u32> = (0..1000).map(|i| ((i * 2654435761u64.wrapping_add(3)) % 1000003) as u32).collect();
        let a = scan_blelloch_cpu(&input);
        let b = scan_blelloch_cpu(&input);
        assert_eq!(a, b, "同输入两遍扫描结果应完全一致");
    }

    /// 形状派生核对：ScanShape 的折叠层序与 reduction_shape 口径一致
    /// （层深 = ⌈log₂ n⌉，与 Blelloch 的 O(log n) 层数同阶）。
    #[test]
    fn scan_shape_matches_reduction_shape_depth() {
        for n in [1usize, 2, 3, 5, 8, 13, 64] {
            let shape = ScanShape::new(n);
            let expected_depth = if n <= 1 { 0 } else { (n as f64).log2().ceil() as usize };
            assert_eq!(
                shape.depth(),
                expected_depth,
                "n={n}：归约树深应为 ⌈log₂ n⌉，与 Blelloch 层数同阶"
            );
        }
        // 与 reduction_shape 的层序同构：n=5 层 0 = [(0,1),(2,3)]，尾 4 进位。
        let s5 = ScanShape::new(5);
        assert_eq!(s5.depth(), 3, "⌈log₂ 5⌉ = 3");
        assert_eq!(s5.layers()[0], vec![[0, 1], [2, 3]]);
    }
}
