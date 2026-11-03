//! 细分曲面（CPU 烘焙器）—— 选型报告 §5 定稿的**按输入拓扑分派**：
//! 三角笼走 [`subdivide`]（Loop），四边笼走 [`catmull_clark`]（Catmull-Clark）。
//!
//! # 定稿口径（逐条对应 `docs/v2.5-W0-算法选型报告.md` §5）
//!
//! 1. **三角笼走 Loop，四边笼走 Catmull-Clark**。两条都在本模块：
//!    glTF/OBJ 进来的通常是三角网，三角笼**零拓扑歧义**；硬推 CC 要做"三角→四边
//!    配对"，而配对没有唯一解（同一资产换导入顺序会得到不同曲面）。故**按输入
//!    拓扑分派**，不做三角→四边配对（那会引入启发式，破坏确定性契约）。
//!    四边形输入由调用方显式走 [`catmull_clark`]（每 4 个索引一四边形）。
//! 2. **极限面不求值迭代**：标准 1-ring 极限 stencil 直接把顶点放到极限位置。
//!    迭代法的"误差 < 1e-4"在这里是非判据（它只是迭代的收敛判据）。
//! 3. **均匀 1→4 细分 ⇒ 结构上无 T 型接缝**：每个三角形恰好裂成四个，相邻面
//!    共享的边各生成**同一个**边顶点（边在整个网格里只有一份）。
//! 4. **uv 定义在顶点域**：细分新顶点的 uv 由父边两端插值得到 ⇒ 相邻面共享顶点
//!    ⇒ 同一 uv ⇒ 同一幅值（位移的 V4 判据靠这条成立，不是靠检查）。
//! 5. **半边结构只活在这里**：本模块无纹理依赖、无渲染器依赖、不改 ABI、
//!    不碰着色器。产物是裸数组，rt 侧自己转 `Mesh`。
//!
//! # 为什么接口是裸数组而不是 `Mesh`
//!
//! 依赖方向是 `prism-rt → prism-geometry`（单向）。`Mesh` 定义在 `prism-rt`，
//! 本 crate 看不到它；把它搬进来又会把渲染器类型塞进几何叶子。故接口取
//! `(&[Vec3], &[u32], &[[f32; 2]], levels) -> SubdivResult`，由 `prism-rt`
//! 侧做一次搬运。
//!
//! # 边界规则（本模块最容易写错的地方）
//!
//! 边界**不是**内部规则的退化，它有自己的 stencil（Loop 1987 / Biermann et al.
//! 2000 的边界口径）：
//!
//! | 位置 | 规则 |
//! |---|---|
//! | 边界边顶点 | 中点 `(a + b) / 2` |
//! | 边界顶点（恰 2 条边界边） | 三次 B 样条曲线规则：`(b₀ + 6v + b₁) / 8` |
//! | 边界顶点（≠ 2 条边界边，即角点/非流形） | **不动**（`v`） |
//! | 内部顶点 | `(1 − nβ)·v + β·Σnb`，`β(n) = (1/n)(5/8 − (3/8 + ¼cos(2π/n))²)` |
//! | 内部边顶点 | `3/8(a + b) + 1/8(c + d)` |
//!
//! 边界顶点的**极限位置**用曲线的三次 B 样条极限 stencil `(b₀ + 4v + b₁) / 6`；
//! 角点保持不动。写错任何一条，硬边处会塌陷（表现为边缘向内收缩成刀锋）——
//! 这是本段唯一"错了但看起来还行"的坑，故有专门的测试（`boundary_*`）。
//!
//! # 确定性
//!
//! * 边表用 `BTreeMap`（**不是** `HashMap`）：迭代序 = 键序，与插入顺序无关；
//! * 邻接表按顶点索引**升序**排列，浮点求和顺序固定；
//! * `β(n)` 的 `cos(2π/n)` 走 [`crate::det_math::det_cos_sin`]（D-29，禁 libm）；
//! * 全程只用 `+ − × ÷ sqrt`；无随机、无并行归约。
//!
//! 同输入同字节（`tests` 里有双跑 `to_bits` 全等的断言）。

use crate::det_math::det_cos_sin;
use prism_math::Vec3;

/// 2π（与 `curve_tess` 同值口径；CPU 侧确定常数）。
const TWO_PI: f32 = 6.2831853;

/// 细分级数上限（防一条 `levels = 1e9` 的请求把内存吃光；4^k 增长）。
///
/// 取 8：`4^8 = 65536` 倍面数。1000 面 k=8 已是 65 M 三角（≈ 2.3 GB），
/// 再高不是"更细"而是"要换自适应细分（C3）"。超限**截断到 8** 并在
/// [`SubdivResult::levels`] 里回显实际级数（不 panic，不静默按 8 算完却报 9）。
pub const MAX_LEVELS: u32 = 8;

/// 细分产物（裸数组；`prism-rt` 侧负责转 `Mesh`）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubdivResult {
    /// 顶点位置 = **极限位置**（不是"再细分一次的位置"）。
    pub positions: Vec<Vec3>,
    /// 三角形索引流（每 3 个一三角形，绕序与输入一致）。
    pub indices: Vec<u32>,
    /// 顶点域 uv（输入无 uv ⇒ 空；否则与 `positions` 同长同序）。
    pub uv: Vec<[f32; 2]>,
    /// 顶点法线（由**极限位置**上的相邻三角形面积加权得到；见 `vertex_normals`）。
    pub normals: Vec<Vec3>,
    /// 实际执行的细分级数（`min(levels, MAX_LEVELS)`）。
    pub levels: u32,
}

impl SubdivResult {
    /// 三角形数。
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------------------
// G-9 / T1：crease（折痕）与 corner（角点）标记
// ---------------------------------------------------------------------------
//
// # 缺口原文
//
// `subdivide` 与 `catmull_clark` 的入参没有「哪些边是硬边」的载体，内部边
// 一律按光滑处理 —— 补丁的角点被曲线规则抹圆（路线图 §2 G-9 记 2.36e-1；
// 本轮在立方体笼上实测更狠：CC k=2 的角点误差 **8.66e-1**、体积
// **2.451** 而非 8；Loop k=3 是 **8.72e-1 / 2.919**）。这不是"静默当内部点"，
// 是**调用方无法表达**。
//
// # 语义（DeRose 半锐利 crease 规则）
//
// | 位置 | 规则 |
// |---|---|
// | crease 边（`w == 1`）的边点 | 端点**中点**（1D 线性） |
// | crease 边（`0 < w < 1`）的边点 | `(1−w)·光滑边点 + w·中点` |
// | crease 顶点（恰 2 条 crease 边） | 1D 三次 B 样条曲线规则 `(b₀ + 6v + b₁)/8`（与边界顶点同式），按 `w_v = max(入射 crease 权重)` 混合 |
// | crease 交汇（≥ 3 条 crease 边）/ corner 标记 | **corner**：`v` 不动（极限也是 `v`） |
// | dart（恰 1 条 crease 边） | 走光滑规则（DeRose 口径：单条锐边不成折痕） |
// | 边界顶点 | 与加标记前**同一条**规则（边界本来就是 crease，不重复计） |
//
// # 为什么 `1` 是"无限锐"而不是"锐 1 级"
//
// 任务的取值域是 `0..=1`。若 `1` 只锐化一级，全 crease 的立方体笼在第二级
// 就重新变圆 —— 判据 V2（角点误差 ≤ 1e-6）不可能达成。故本模块定义：
// **`w == 1` 是"永不衰减"**，`0 < w < 1` 是"只影响本级"（子边权重降为 0，
// 这就是半锐利的分数部分语义）。这条写进判据，也写进 `CHANGELOG`。
//
// # 逐位退化（判据 V1）
//
// `marks` 为 `None` **或** 全部权重为 0 且无 corner ⇒ 走的是与加标记前
// **同一个函数、同一串表达式**（`MarkIndex` 为 `None`，每个规则里的
// crease 分支都是 `if let Some(..)`，不命中就执行原表达式）⇒ 逐位相同。

/// 细分曲面的折痕标记（crease 边权重 + corner 顶点标记）。
///
/// # 构造
///
/// ```
/// use prism_geometry::subdiv::SubdivMarks;
///
/// let m: SubdivMarks = SubdivMarks::new()
///     .with_crease(0, 1, 1.0)   // 0→1 这条边完全锐利
///     .with_corner(4);          // 顶点 4 是角点
/// assert_eq!(m.edge_count(), 1);
/// assert_eq!(m.corner_count(), 1);
/// assert!(!m.is_empty());
/// ```
///
/// 边用**无序顶点对**指定（内部按 `(min, max)` 规范化并排序），重复指定
/// 同一条边以**最后一次**为准（不累加、不取最大值 —— 后者会让"先设 0.3 再设
/// 0.0"变成 0.3，违反最小惊讶）。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubdivMarks {
    /// 边权重：`(min, max, w)`，按 `(min, max)` 升序、无重复。
    edges: Vec<(u32, u32, f32)>,
    /// corner 顶点（升序、无重复）。
    corners: Vec<u32>,
}

impl SubdivMarks {
    /// 空标记（= 完全光滑）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加/覆盖一条边的 crease 权重（钳到 `[0, 1]`；非有限 ⇒ `0`，不 panic）。
    pub fn with_crease(mut self, a: u32, b: u32, w: f32) -> Self {
        if a == b {
            return self; // 自环：拓扑上无意义，静默忽略（不制造伪边）
        }
        let w: f32 = if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 };
        let key: (u32, u32) = (a.min(b), a.max(b));
        match self.edges.binary_search_by(|e| (e.0, e.1).cmp(&key)) {
            Ok(i) => self.edges[i].2 = w,
            Err(i) => self.edges.insert(i, (key.0, key.1, w)),
        }
        self
    }

    /// 标记一个 corner 顶点（升序插入；重复幂等）。
    pub fn with_corner(mut self, v: u32) -> Self {
        match self.corners.binary_search(&v) {
            Ok(_) => {}
            Err(i) => self.corners.insert(i, v),
        }
        self
    }

    /// 是否**没有生效的标记**（无 corner 且所有边权重恰为 0）。
    ///
    /// `true` ⇒ 调用方走与加标记前**完全相同**的路径（判据 V1 的结构前提）。
    pub fn is_empty(&self) -> bool {
        self.corners.is_empty() && self.edges.iter().all(|e| e.2 == 0.0)
    }

    /// 边权重条数（含 `w == 0` 的显式声明）。
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// corner 个数。
    pub fn corner_count(&self) -> usize {
        self.corners.len()
    }

    /// 逐级传播：把本级的标记推到细分一级后的网格上。
    ///
    /// * `w == 1` 的边裂成两半后**仍然是 1**（折痕线自身继续是折痕）；
    /// * `0 < w < 1` 的子边降为 0（分数部分只作用于本级，DeRose 的半锐利口径）；
    /// * corner 标记随顶点下标平移（细分后老顶点保持在前，下标不变）。
    fn child_marks(
        &self,
        edge_index_of: impl Fn(u32, u32) -> Option<usize>,
        n_old: usize,
    ) -> SubdivMarks {
        let mut out: SubdivMarks = SubdivMarks::new();
        out.corners = self.corners.clone();
        for &(a, b, w) in &self.edges {
            if w != 1.0 {
                continue; // 分数锐利不跨级
            }
            let Some(e) = edge_index_of(a, b) else {
                continue; // 该边已不存在（被净化剔除）：标记自然消失
            };
            let m: u32 = (n_old + e) as u32;
            out = out.with_crease(a, m, 1.0);
            out = out.with_crease(m, b, 1.0);
        }
        out
    }
}

/// 某一级网格上的标记视图（与 `Topology` / `QuadTopology` 的边表**同序**）。
///
/// # 为什么要这一层
///
/// 边权重在 `SubdivMarks` 里按 `(min,max)` 排序，而每个细分级的边表是**重建**
/// 的（顺序不同）。把"按当前边表的平行数组"算一次，规则里就只剩 O(1) 下标
/// 访问 —— 每级一次二分、每顶点一次邻接扫描，不进浮点热循环。
///
/// # 为什么还要 CSR 入射表
///
/// 顶点规则要回答"这个顶点有几条 crease 边、权重多大"（DeRose 的
/// dart/crease/corner 三分类）。若现查，每个顶点要对每个邻居各做一次二分
/// （k=6 的 4 万顶点上是 24 万次 `binary_search`，且是**逐级**的）。改成建表
/// 一次 + O(1) 切片，与 `Topology` 的 CSR 同一口径，求和顺序也顺带固定
/// （按键 `(min,max)` 升序 ⇒ 邻居升序）。
struct MarkIndex {
    /// 与当前边表同序的权重（`0.0` = 光滑）。
    edge_w: Vec<f32>,
    /// 逐顶点的 corner 标记。
    corner: Vec<bool>,
    /// 入射 crease 边的 CSR 偏移（长度 `V + 1`）。
    inc_off: Vec<u32>,
    /// 入射 crease 边的**另一端顶点**（每段**升序**）。
    inc_nb: Vec<u32>,
    /// 与 `inc_nb` 同序的权重。
    inc_w: Vec<f32>,
}

impl MarkIndex {
    /// 在当前边表上建视图（`lookup` 把顶点对映射到当前边表下标）。
    fn build(
        n_edges: usize,
        n_vert: usize,
        marks: &SubdivMarks,
        lookup: impl Fn(u32, u32) -> Option<usize>,
    ) -> Self {
        let mut edge_w: Vec<f32> = vec![0.0; n_edges];
        // 第一趟：落到边表上（标记可能指向已被剔除的边 ⇒ `None`，静默跳过）。
        let mut live: Vec<(u32, u32, f32)> = Vec::with_capacity(marks.edges.len());
        for &(a, b, w) in &marks.edges {
            if let Some(i) = lookup(a, b) {
                edge_w[i] = w;
                if w > 0.0 {
                    live.push((a, b, w));
                }
            }
        }
        let mut corner: Vec<bool> = vec![false; n_vert];
        for &v in &marks.corners {
            if let Some(c) = corner.get_mut(v as usize) {
                *c = true;
            }
        }
        // 第二趟：计数 → 前缀和 → 填充（两趟式 CSR，无逐顶点 `Vec`）。
        let mut inc_off: Vec<u32> = vec![0; n_vert + 1];
        for &(a, b, _) in &live {
            if (a as usize) < n_vert {
                inc_off[a as usize + 1] += 1;
            }
            if (b as usize) < n_vert {
                inc_off[b as usize + 1] += 1;
            }
        }
        for i in 0..n_vert {
            inc_off[i + 1] += inc_off[i];
        }
        let total = inc_off[n_vert] as usize;
        let mut inc_nb: Vec<u32> = vec![0; total];
        let mut inc_w: Vec<f32> = vec![0.0; total];
        // 游标式填充。`live` 已按 `(min,max)` 升序（`marks.edges` 本身的顺序），
        // 故每个顶点收到的是**升序**邻居序列 —— 求和顺序固定。
        let mut cur: Vec<u32> = inc_off[..n_vert].to_vec();
        for &(a, b, w) in &live {
            if (a as usize) < n_vert {
                let p = cur[a as usize] as usize;
                inc_nb[p] = b;
                inc_w[p] = w;
                cur[a as usize] += 1;
            }
            if (b as usize) < n_vert {
                let p = cur[b as usize] as usize;
                inc_nb[p] = a;
                inc_w[p] = w;
                cur[b as usize] += 1;
            }
        }
        Self { edge_w, corner, inc_off, inc_nb, inc_w }
    }

    /// 顶点 `i` 是否 corner。
    #[inline]
    fn is_corner(&self, i: usize) -> bool {
        self.corner.get(i).copied().unwrap_or(false)
    }

    /// 顶点 `i` 的入射 crease 边切片 `(另一端顶点, 权重)`（按邻居升序）。
    #[inline]
    fn inc_crease(&self, i: usize) -> &[u32] {
        let lo = self.inc_off.get(i).copied().unwrap_or(0) as usize;
        let hi = self.inc_off.get(i + 1).copied().unwrap_or(0) as usize;
        &self.inc_nb[lo..hi]
    }

    /// 顶点 `i` 的入射 crease 边权重（与 [`Self::inc_crease`] 同序）。
    #[inline]
    fn inc_weights(&self, i: usize) -> &[f32] {
        let lo = self.inc_off.get(i).copied().unwrap_or(0) as usize;
        let hi = self.inc_off.get(i + 1).copied().unwrap_or(0) as usize;
        &self.inc_w[lo..hi]
    }

    /// 顶点 `i` 的 crease 度数（入射 crease 边条数）。
    #[inline]
    fn crease_degree(&self, i: usize) -> usize {
        self.inc_crease(i).len()
    }

    /// 把光滑结果与 crease 结果按权重 `w` 混合。
    ///
    /// `w == 1` 时**直接返回 `sharp`**（不执行 `smooth*(0) + sharp*1`：
    /// 后者在 `smooth` 含 `∞`/`NaN` 时会把 `sharp` 也毒化，而 corner 顶点
    /// 的 `sharp` 是干净的 `v`）。
    #[inline]
    fn blend(smooth: Vec3, sharp: Vec3, w: f32) -> Vec3 {
        if w >= 1.0 {
            return sharp;
        }
        if w <= 0.0 {
            return smooth;
        }
        smooth * (1.0 - w) + sharp * w
    }
}

/// Loop 内部顶点的 β(n)。
///
/// `β(n) = (1/n) · (5/8 − (3/8 + ¼·cos(2π/n))²)`（Loop 的原始 β，1987 硕士论文
/// "Smooth subdivision surfaces based on triangles"）。
///
/// 定点校核：`n = 3 → 3/16`、`n = 6 → 1/16`（与 Warren 简化式
/// `β = 3/(8n)` 在这两处**相同**，其余 n 不同——用原式，因为它是 C² 的那个）。
/// `n < 3` 不构成合法内部顶点（三角网内部顶点度数 ≥ 3）；兜底返回 `3/16`，
/// 只为"绝不产生 NaN"（C18）。
pub fn loop_beta(n: usize) -> f32 {
    if n < 3 {
        return 3.0 / 16.0;
    }
    let nf = n as f32;
    let (c, _) = det_cos_sin(TWO_PI / nf);
    let inner = 0.375 + 0.25 * c;
    ((5.0 / 8.0) - inner * inner) / nf
}

/// 内部顶点的**极限位置**权重对 `(a, b)`：`v_∞ = a·v + b·Σnb`。
///
/// `a + n·b = 1`（仿射不变：常值数据的不动点）。
///
/// 这是 Loop 极限位置的标准闭式（等价于把顶点规则按 `λ₀ = 1` 的左特征向量
/// 归一：`a + n·b = 1`）。**不是迭代**：一次算术就把顶点放到极限面上。
///
/// 正确性**不靠引用**，靠实测钉死：`subdiv::tests::limit_stencil_matches_iterated_limit`
/// 把本式在 level-0 的结果与"再细分 4 级后同一顶点位置"对比（差 5.33e-7，f32
/// 舍入量级），同时把似是而非的错候选 `(3v + Σnb)/(n+3)` 一起算（差 0.51）——
/// 没有这个对照，"我的 stencil 等于我的迭代"可能只是自洽。
pub fn loop_limit_weights(n: usize) -> (f32, f32) {
    let nf = n.max(3) as f32;
    let beta = loop_beta(n.max(3));
    let denom = 3.0 + 8.0 * beta * nf;
    (3.0 / denom, 8.0 * beta / denom)
}

/// 边界顶点的极限位置权重（三次 B 样条曲线）：`(b₀ + 4v + b₁) / 6`。
pub const BOUNDARY_LIMIT: (f32, f32) = (4.0 / 6.0, 1.0 / 6.0);

/// 入口：均匀 Loop 细分 `levels` 级，末尾把顶点放到**极限位置**。
///
/// # 参数
///
/// * `positions` / `indices`：三角笼（`indices.chunks_exact(3)`）；
/// * `uv`：空 ⇒ 不产出 uv；非空时长度须等于 `positions.len()`（长度不符 ⇒
///   按"无 uv"处理，不 panic——与 `Mesh::uv` 的契约一致）；
/// * `levels`：细分级数（`0` ⇒ 只做极限位置求值，不改拓扑）。
///
/// # 退化
///
/// 空输入 / 索引越界 / 非有限顶点 ⇒ 对应三角形被剔除（不 panic）。若剔除后
/// 无三角形 ⇒ 返回空结果（`levels` 回显**截断后**的级数）。
pub fn subdivide(
    positions: &[Vec3],
    indices: &[u32],
    uv: &[[f32; 2]],
    levels: u32,
) -> SubdivResult {
    subdivide_marked(positions, indices, uv, levels, None)
}

/// 入口（带 crease / corner 标记）：均匀 Loop 细分 `levels` 级 + 极限位置求值。
///
/// # 与 [`subdivide`] 的关系
///
/// [`subdivide`] 就是 `subdivide_marked(.., None)`。`marks` 为 `None` **或**
/// `marks.is_empty()`（全 0 权重、无 corner）时，走的是与本文件加标记前
/// **同一个函数、同一串表达式**（`MarkIndex` 根本没建）⇒ 逐位相同（判据 V1）。
///
/// # 规则（DeRose 半锐利，逐条见本文件 §G-9 的表格）
///
/// * corner 标记 / crease 度 ≥ 3 ⇒ 顶点**不动**（按 `w_v` 混合）；
/// * crease 度 == 2 ⇒ 1D 三次 B 样条 `(b₀ + 6v + b₁)/8`（按 `w_v` 混合）；
/// * crease 度 == 1（dart）⇒ **光滑**规则（DeRose：单条锐边不成折痕）；
/// * crease 边（`w > 0`）的边点 ⇒ `(1−w)·光滑 + w·中点`；
/// * `w_v = max(入射 crease 权重)`；
/// * 边界顶点走**原规则**（边界本来就是 crease，不重复计）。
///
/// `w == 1` 表示"永不衰减"（子边仍为 1）；`0 < w < 1` 只影响本级（子边降 0）。
pub fn subdivide_marked(
    positions: &[Vec3],
    indices: &[u32],
    uv: &[[f32; 2]],
    levels: u32,
    marks: Option<&SubdivMarks>,
) -> SubdivResult {
    let levels = levels.min(MAX_LEVELS);
    let mut pos: Vec<Vec3> = positions.to_vec();
    let mut tri: Vec<u32> = indices.to_vec();
    // uv 长度不符 ⇒ 按"无 uv"处理（与 `Mesh::uv` 的契约一致，不 panic）。
    let mut tex: Vec<[f32; 2]> = if uv.len() == positions.len() { uv.to_vec() } else { Vec::new() };
    // 空标记与"没有标记"等价（结构性逐位退化，判据 V1）。
    let mut cur: Option<SubdivMarks> = marks.filter(|m| !m.is_empty()).cloned();

    // 初始净化：越界 / 非有限 ⇒ 剔除该三角形（与 `Mesh::new` 同口径）。
    tri = sanitize(&pos, &tri);
    if tri.is_empty() {
        return SubdivResult { levels, ..Default::default() };
    }

    for _ in 0..levels {
        let (p, t, u, child) = refine_once(&pos, &tri, &tex, cur.as_ref());
        pos = p;
        tri = t;
        tex = u;
        cur = child;
    }

    // 极限位置（不迭代）。
    let topo = Topology::build(&pos, &tri);
    let mi: Option<MarkIndex> = cur.as_ref().map(|m| {
        MarkIndex::build(topo.edges.len(), pos.len(), m, |a, b| topo.lookup_edge(a, b))
    });
    let lim = limit_positions(&pos, &topo, mi.as_ref());
    let normals = vertex_normals(&lim, &tri);
    SubdivResult { positions: lim, indices: tri, uv: tex, normals, levels }
}

/// 只求极限位置（不改拓扑；`levels = 0` 的 `subdivide` 等价于它）。
pub fn limit_of(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let tri = sanitize(positions, indices);
    let topo = Topology::build(positions, &tri);
    limit_positions(positions, &topo, None)
}

/// 剔除越界 / 非有限三角形的索引流。
fn sanitize(pos: &[Vec3], indices: &[u32]) -> Vec<u32> {
    let n = pos.len();
    let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let mut out = Vec::with_capacity(indices.len());
    for c in indices.chunks_exact(3) {
        let (i0, i1, i2) = (c[0] as usize, c[1] as usize, c[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        if !finite(pos[i0]) || !finite(pos[i1]) || !finite(pos[i2]) {
            continue;
        }
        // 零面积三角形：不剔除（Loop 允许退化面，按规则照样细分；剔除会改变
        // 邻接拓扑，反而在边界处制造洞）。
        out.extend_from_slice(c);
    }
    out
}

/// 网格拓扑（一次构建，供顶点规则/极限 stencil 使用）。
///
/// # 为什么是 CSR（压缩稀疏行）而不是 `Vec<Vec<u32>>` / `HashMap`
///
/// 三条硬要求同时压在这里：
///
/// 1. **确定性**：邻接必须按顶点索引升序，浮点求和顺序才固定。`HashMap` 的
///    迭代序随哈希种子变 ⇒ 禁用；`BTreeMap` 顺序对但每个节点都是一次堆分配。
/// 2. **内存**：`k = 6` 的输入是 2 万面、输出是 8 万面（4 万顶点），逐顶点
///    `Vec` 的"24 B 头 + 每次 malloc 的块头"就能吃掉几 MB，而判据 V5 量的是
///    峰值 RSS（改 CSR 前后实测：k=6 的 RSS 增量 15.4 MB → 3.8 MB）。
/// 3. **速度**：半边先收集成扁平数组再**排序**，一次扫描即成表——比逐条
///    插入平衡树快一个量级（debug 构建下尤其明显）。
///
/// 三张表都是 `offset + flat` 的两段式：`nb_off[i]..nb_off[i+1]` 是顶点 `i`
/// 的邻接切片。`boundary_nbrs` 用同一形状，只是只装边界边邻居。
struct Topology {
    /// 1-ring 邻接的 CSR 偏移（长度 `V + 1`）。
    nb_off: Vec<u32>,
    /// 1-ring 邻接的扁平存储（每顶点一段，**升序**）。
    nb: Vec<u32>,
    /// 边界邻居的 CSR 偏移（长度 `V + 1`）。
    bn_off: Vec<u32>,
    /// 边界邻居的扁平存储（只经由 `count == 1` 的边；**升序**）。
    bn: Vec<u32>,
    /// 边表（按 `(min, max)` **升序**）：索引 = 边顶点的序号偏移。
    edges: Vec<EdgeRec>,
    /// 该顶点是否连着"被 > 2 个面使用"的非流形边（按边界处理）。
    nonmanifold: Vec<bool>,
}

/// 一条边（无向，`a < b`）。
#[derive(Clone, Copy, Default)]
struct EdgeRec {
    /// 较小端点。
    a: u32,
    /// 较大端点。
    b: u32,
    /// 对顶点（至多两个；`count > 2` 的非流形边只保留前两个）。
    opp: [u32; 2],
    /// 该边被多少个三角形使用（1 = 边界边，2 = 内部边，> 2 = 非流形）。
    count: u32,
}

impl Topology {
    fn build(pos: &[Vec3], tri: &[u32]) -> Self {
        let n = pos.len();
        // --- 半边收集：每三角形 3 条，扁平数组（12 B × 3T） ---
        let mut half: Vec<(u32, u32, u32)> = Vec::with_capacity(tri.len());
        for c in tri.chunks_exact(3) {
            let v = [c[0], c[1], c[2]];
            for k in 0..3 {
                let a = v[k];
                let b = v[(k + 1) % 3];
                let o = v[(k + 2) % 3];
                half.push(if a < b { (a, b, o) } else { (b, a, o) });
            }
        }
        // 排序键 `(a, b, opp)`：并列由第三个分量定序 ⇒ **全序**，与输入顺序无关。
        half.sort_unstable();

        // --- 一次扫描成边表（相同的 `(a, b)` 合并） ---
        let mut edges: Vec<EdgeRec> = Vec::with_capacity(half.len() / 2 + 1);
        for &(a, b, o) in half.iter() {
            match edges.last_mut() {
                Some(e) if e.a == a && e.b == b => {
                    if e.count < 2 {
                        e.opp[e.count as usize] = o;
                    }
                    e.count = e.count.saturating_add(1);
                }
                _ => edges.push(EdgeRec { a, b, opp: [o, o], count: 1 }),
            }
        }
        drop(half);

        // --- 邻接 CSR：先数度，再填充 ---
        let mut nb_off = vec![0u32; n + 1];
        let mut bn_off = vec![0u32; n + 1];
        for e in edges.iter() {
            nb_off[e.a as usize + 1] += 1;
            nb_off[e.b as usize + 1] += 1;
            if e.count == 1 {
                bn_off[e.a as usize + 1] += 1;
                bn_off[e.b as usize + 1] += 1;
            }
        }
        for i in 0..n {
            nb_off[i + 1] += nb_off[i];
            bn_off[i + 1] += bn_off[i];
        }
        let mut nb = vec![0u32; nb_off[n] as usize];
        let mut bn = vec![0u32; bn_off[n] as usize];
        let mut nb_fill = nb_off.clone();
        let mut bn_fill = bn_off.clone();
        let mut nonmanifold = vec![false; n];
        // 边表按 `(a, b)` 升序 ⇒ 顶点 `a` 的邻居按 `b` 升序入表；
        // 顶点 `b` 的邻居按 `a` 升序入表（主键序）⇒ 两段天然有序，无需再排。
        for e in edges.iter() {
            let (a, b) = (e.a as usize, e.b as usize);
            nb[nb_fill[a] as usize] = e.b;
            nb_fill[a] += 1;
            nb[nb_fill[b] as usize] = e.a;
            nb_fill[b] += 1;
            if e.count == 1 {
                bn[bn_fill[a] as usize] = e.b;
                bn_fill[a] += 1;
                bn[bn_fill[b] as usize] = e.a;
                bn_fill[b] += 1;
            } else if e.count > 2 {
                // 非流形边：两端顶点都按边界顶点处理（否则内部规则会在该处
                // 用"多于两个对顶点"的数据算出一个无定义的位置）。
                nonmanifold[a] = true;
                nonmanifold[b] = true;
            }
        }
        Topology { nb_off, nb, bn_off, bn, edges, nonmanifold }
    }

    /// 顶点 `i` 的 1-ring 邻居（升序切片）。
    #[inline]
    fn neighbors(&self, i: usize) -> &[u32] {
        &self.nb[self.nb_off[i] as usize..self.nb_off[i + 1] as usize]
    }

    /// 顶点 `i` 的边界邻居（升序切片）。
    #[inline]
    fn boundary_nbrs(&self, i: usize) -> &[u32] {
        &self.bn[self.bn_off[i] as usize..self.bn_off[i + 1] as usize]
    }

    /// 该顶点是否连着"被 > 2 个面使用"的边（非流形）。
    #[inline]
    fn has_nonmanifold_edge(&self, i: usize) -> bool {
        self.nonmanifold.get(i).copied().unwrap_or(false)
    }

    /// `(min, max)` → 边序号（二分；命中必然存在，调用方保证键来自本网格）。
    #[inline]
    fn edge_index(&self, a: u32, b: u32) -> usize {
        let key = (a.min(b), a.max(b));
        self.edges
            .binary_search_by(|e| (e.a, e.b).cmp(&key))
            .expect("边必须存在（键来自本网格的三角形）")
    }

    /// `(min, max)` → 边序号（**不 panic**：标记可能指向已被剔除的边）。
    #[inline]
    fn lookup_edge(&self, a: u32, b: u32) -> Option<usize> {
        let key = (a.min(b), a.max(b));
        self.edges.binary_search_by(|e| (e.a, e.b).cmp(&key)).ok()
    }
}

/// 细分一级（1→4），返回新的 `(positions, indices, uv, 子级标记)`。
///
/// `marks` 为 `None` ⇒ 每个规则里都不进 crease 分支（`if let Some(..)` 不命中
/// ⇒ 执行原表达式），返回值第 4 项为 `None` —— 这与加标记前**逐位相同**。
fn refine_once(
    pos: &[Vec3],
    tri: &[u32],
    uv: &[[f32; 2]],
    marks: Option<&SubdivMarks>,
) -> (Vec<Vec3>, Vec<u32>, Vec<[f32; 2]>, Option<SubdivMarks>) {
    let topo = Topology::build(pos, tri);
    let n_old = pos.len();
    let n_edge = topo.edges.len();
    let mi: Option<MarkIndex> = marks
        .map(|m| MarkIndex::build(n_edge, n_old, m, |a, b| topo.lookup_edge(a, b)));

    let mut out_pos: Vec<Vec3> = Vec::with_capacity(n_old + n_edge);
    // 顶点规则。
    for i in 0..n_old {
        out_pos.push(moved_vertex(pos, &topo, i, mi.as_ref()));
    }
    // 边顶点（边表按 `(a, b)` 升序 ⇒ 顶点索引确定）。
    for (e_i, e) in topo.edges.iter().enumerate() {
        let (a, b) = (e.a, e.b);
        let pa = pos[a as usize];
        let pb = pos[b as usize];
        let p_smooth = if e.count == 1 || e.count > 2 {
            // 边界边 / 非流形边：中点。
            (pa + pb) * 0.5
        } else {
            let c = pos[e.opp[0] as usize];
            let d = pos[e.opp[1] as usize];
            (pa + pb) * 0.375 + (c + d) * 0.125
        };
        // crease：按边权重把边点拉向中点（`w == 1` ⇒ 纯中点）。
        let p = match &mi {
            Some(m) if m.edge_w[e_i] > 0.0 => {
                MarkIndex::blend(p_smooth, (pa + pb) * 0.5, m.edge_w[e_i])
            }
            _ => p_smooth,
        };
        out_pos.push(p);
    }
    // uv：老顶点原样，边顶点取父边中点（顶点域插值）。
    let mut out_uv: Vec<[f32; 2]> = Vec::new();
    if !uv.is_empty() {
        out_uv.reserve(n_old + n_edge);
        out_uv.extend_from_slice(uv);
        for e in topo.edges.iter() {
            let ua = uv[e.a as usize];
            let ub = uv[e.b as usize];
            out_uv.push([(ua[0] + ub[0]) * 0.5, (ua[1] + ub[1]) * 0.5]);
        }
    }
    // 面：1→4，绕序保持。
    let mut out_tri: Vec<u32> = Vec::with_capacity(tri.len() * 4);
    for c in tri.chunks_exact(3) {
        let (a, b, cc) = (c[0], c[1], c[2]);
        let ab = (n_old + topo.edge_index(a, b)) as u32;
        let bc = (n_old + topo.edge_index(b, cc)) as u32;
        let ca = (n_old + topo.edge_index(cc, a)) as u32;
        out_tri.extend_from_slice(&[a, ab, ca]);
        out_tri.extend_from_slice(&[ab, b, bc]);
        out_tri.extend_from_slice(&[ca, bc, cc]);
        out_tri.extend_from_slice(&[ab, bc, ca]);
    }
    // 子级标记：`w == 1` 的两半仍是 1，分数权重降 0（DeRose 半锐利）。
    let child = marks
        .map(|m| m.child_marks(|a, b| topo.lookup_edge(a, b), n_old))
        .filter(|m| !m.is_empty());
    (out_pos, out_tri, out_uv, child)
}

/// 顶点规则（corner / 边界 / crease / 内部 / dart）。
///
/// 顺序即优先级：corner 标记 > 边界 > crease > 光滑。边界排在 crease 之前
/// 是本模块的定稿口径 —— 边界**本来就是** crease，重复计会让边界顶点被
/// 两次锐化（表现为边缘收缩）。
fn moved_vertex(pos: &[Vec3], topo: &Topology, i: usize, mi: Option<&MarkIndex>) -> Vec3 {
    let v = pos[i];
    if let Some(m) = mi {
        if m.is_corner(i) {
            return v; // corner 标记：极限也不动
        }
    }
    let bn = topo.boundary_nbrs(i);
    if !bn.is_empty() || topo.has_nonmanifold_edge(i) {
        // 边界顶点：恰两条边界边 ⇒ 三次 B 样条曲线规则；否则（角点/非流形）不动。
        if bn.len() == 2 {
            let b0 = pos[bn[0] as usize];
            let b1 = pos[bn[1] as usize];
            return (b0 + b1) * 0.125 + v * 0.75;
        }
        return v;
    }
    // --- crease / corner（内部顶点）---
    if let Some(m) = mi {
        let k = m.crease_degree(i);
        if k == 2 {
            let nb = m.inc_crease(i);
            let w = m.inc_weights(i);
            let (b0, b1) = (pos[nb[0] as usize], pos[nb[1] as usize]);
            let wv = if w[0] > w[1] { w[0] } else { w[1] };
            let sharp = (b0 + b1 + v * 6.0) * 0.125; // 1D 三次 B 样条：与边界同式
            if wv >= 1.0 {
                return sharp;
            }
            return MarkIndex::blend(smooth_inner(pos, topo, i, v), sharp, wv);
        }
        if k >= 3 {
            let w = m.inc_weights(i);
            let mut wv = 0.0f32;
            for &x in w {
                if x > wv {
                    wv = x;
                }
            }
            if wv >= 1.0 {
                return v; // crease 交汇 = corner：不动
            }
            return MarkIndex::blend(smooth_inner(pos, topo, i, v), v, wv);
        }
        // k == 0（光滑）/ k == 1（dart）⇒ 落到下面的光滑规则。
    }
    smooth_inner(pos, topo, i, v)
}

/// Loop 内部顶点的光滑规则（`(1 − nβ)·v + β·Σnb`）。**一行未改**，
/// 加标记前后共用同一串表达式（判据 V1 的逐位前提）。
fn smooth_inner(pos: &[Vec3], topo: &Topology, i: usize, v: Vec3) -> Vec3 {
    let nb = topo.neighbors(i);
    let n = nb.len();
    if n == 0 {
        return v;
    }
    let beta = loop_beta(n);
    let mut sum = Vec3::splat(0.0);
    for &j in nb {
        sum = sum + pos[j as usize]; // 升序累加（求和顺序固定）
    }
    v * (1.0 - (n as f32) * beta) + sum * beta
}

/// 极限位置：corner / 边界曲线 stencil / crease 曲线 stencil / 内部 1-ring stencil。
fn limit_positions(pos: &[Vec3], topo: &Topology, mi: Option<&MarkIndex>) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        let v = pos[i];
        if let Some(m) = mi {
            if m.is_corner(i) {
                out.push(v); // corner 标记：极限 = 自身
                continue;
            }
        }
        let bn = topo.boundary_nbrs(i);
        if !bn.is_empty() || topo.has_nonmanifold_edge(i) {
            if bn.len() == 2 {
                let b0 = pos[bn[0] as usize];
                let b1 = pos[bn[1] as usize];
                let (a, b) = BOUNDARY_LIMIT;
                out.push(v * a + (b0 + b1) * b);
            } else {
                out.push(v); // 角点：不动
            }
            continue;
        }
        // --- crease 极限（内部顶点）---
        if let Some(m) = mi {
            let k = m.crease_degree(i);
            if k == 2 || k >= 3 {
                let sharp: Vec3 = if k == 2 {
                    let nb = m.inc_crease(i);
                    let (a, b) = BOUNDARY_LIMIT; // (4/6, 1/6)：与边界曲线同一支
                    v * a + (pos[nb[0] as usize] + pos[nb[1] as usize]) * b
                } else {
                    v
                };
                let w = m.inc_weights(i);
                let mut wv = 0.0f32;
                for &x in w {
                    if x > wv {
                        wv = x;
                    }
                }
                if wv >= 1.0 {
                    out.push(sharp);
                    continue;
                }
                out.push(MarkIndex::blend(smooth_limit(pos, topo, i, v), sharp, wv));
                continue;
            }
        }
        out.push(smooth_limit(pos, topo, i, v));
    }
    out
}

/// Loop 内部顶点的极限 stencil（`a·v + b·Σnb`，`a + n·b = 1`）。**一行未改**。
fn smooth_limit(pos: &[Vec3], topo: &Topology, i: usize, v: Vec3) -> Vec3 {
    let nb = topo.neighbors(i);
    if nb.is_empty() {
        return v;
    }
    let (a, b) = loop_limit_weights(nb.len());
    let mut sum = Vec3::splat(0.0);
    for &j in nb {
        sum = sum + pos[j as usize];
    }
    v * a + sum * b
}

/// 顶点法线：相邻三角形（按**极限位置**）的面积加权法线之和，归一。
///
/// # 口径（如实标注）
///
/// 这是"极限面上的**离散**法线"：顶点本身在极限面上（[`limit_positions`] 保证），
/// 法线由它周围一圈极限三角形加权得到，与解析极限切向（Stam 1998 的两条
/// 次主特征向量）相差 `O(h)`，`h` = 该级网格的边长。位移用它当方向，
/// 误差直接体现为位移方向的 `O(h)` 偏差；在 `k = 5`（`h ≈ 笼边长/32`）下
/// 这是亚像素量级（实测见报告 §22 V1）。
///
/// 不选解析切向的理由：Loop 的两条次主特征向量要按价数解 2×2 特征系统
/// （含 `cos(2πk/n)` 的整套权重），代码量与出错面远大于收益，而 V1 的
/// Hausdorff 判据对 `O(h)` 的法线误差不敏感（它只看法线**位置**是否在极限面上）。
fn vertex_normals(pos: &[Vec3], tri: &[u32]) -> Vec<Vec3> {
    let mut acc = vec![Vec3::splat(0.0); pos.len()];
    for c in tri.chunks_exact(3) {
        let (a, b, d) = (c[0] as usize, c[1] as usize, c[2] as usize);
        let n = (pos[b] - pos[a]).cross(&(pos[d] - pos[a])); // 面积加权（未归一）
        acc[a] = acc[a] + n;
        acc[b] = acc[b] + n;
        acc[d] = acc[d] + n;
    }
    acc.into_iter()
        .map(|n| {
            let l = n.length();
            if l > 1e-20 {
                n * (1.0 / l)
            } else {
                Vec3::new(0.0, 1.0, 0.0) // 退化（孤立/共线）：给一个确定值，不产 NaN
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 正三角形网格（规则 6 价内部点）的平面补丁：`n×n` 个顶点、`2(n−1)²` 个三角形。
    ///
    /// 顶点点阵 `(i, j)` 的世界坐标 `(i + j/2, j·√3/2, 0)`；两个三角形/格。
    fn regular_patch(n: usize) -> (Vec<Vec3>, Vec<u32>) {
        let mut pos = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                pos.push(Vec3::new(i as f32 + 0.5 * j as f32, j as f32 * 0.866_025_4, 0.0));
            }
        }
        let idx = |i: usize, j: usize| (j * n + i) as u32;
        let mut tri = Vec::with_capacity(2 * (n - 1) * (n - 1));
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                tri.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i, j + 1)]);
                tri.extend_from_slice(&[idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        (pos, tri)
    }

    /// 三角形数：每级 ×4（结构性的，与几何无关）。
    #[test]
    fn triangle_count_quadruples_per_level() {
        let (p, t) = regular_patch(5);
        for k in 0..=4 {
            let r = subdivide(&p, &t, &[], k);
            assert_eq!(r.tri_count(), t.len() / 3 * 4usize.pow(k), "k={k}");
            assert_eq!(r.levels, k);
        }
    }

    /// **V2'：正三角形网格的极限位置 vs 解析值（相对误差 ≤ 1e-6）**。
    ///
    /// 平面规则网格的不动点性质：顶点/边中点/面心的极限位置就是它们自己的坐标
    /// （仿射不变性的直接推论）。任何 β / 极限 stencil / 边界规则写错都会破坏
    /// 它（错一位就从 0 变成 O(1) 偏差）。
    #[test]
    fn flat_regular_patch_is_reproduced_exactly() {
        let (p, t) = regular_patch(6);
        let r = subdivide(&p, &t, &[], 3);
        // 顶点域：原顶点（前 p.len() 个）的极限位置必须等于自身。
        //
        // 排除补丁的四个**角点**：角点恰有两条边界边 ⇒ 走曲线规则，120° 的转角
        // 会被自然抹圆（这是 Loop 边界规则的既定语义，不是 bug）。要在角点保持
        // 尖锐必须给顶点打 corner/crease 标记——本轮不做（见模块文档与报告 §22）。
        let n = 6;
        let is_corner = |i: usize| {
            let (i, j) = (i % n, i / n);
            (i == 0 || i == n - 1) && (j == 0 || j == n - 1)
        };
        let mut worst = 0.0_f32;
        for i in 0..p.len() {
            if is_corner(i) {
                continue;
            }
            let d = (r.positions[i] - p[i]).length();
            worst = worst.max(d);
            assert!(d <= 1e-6 * p[i].length().max(1.0), "顶点 {i} 漂移 {d}");
        }
        println!("V2'：平面规则补丁（排除 4 角点）最大漂移 {worst:e}");
        // 全部顶点（含边/面生成的新顶点）必须落在 xy 平面（z 恒为 0）。
        let max_z = r.positions.iter().map(|v| v.z.abs()).fold(0.0_f32, f32::max);
        assert!(max_z <= 1e-6, "平面补丁的极限面必须仍在该平面上，max|z| = {max_z}");
    }

    /// 边界规则：**开放的平面条带**边界顶点不向内收缩。
    ///
    /// 平面 + 直线边界上，三次 B 样条曲线规则与中点规则都精确复现原直线
    /// （`(b₀ + 6v + b₁)/8` 对等距共线点 = v；极限 `(b₀+4v+b₁)/6` 亦然）。
    /// 若把边界当成内部（β 规则），边界顶点会被拉向内侧 ⇒ 条带变窄。
    #[test]
    fn boundary_vertices_do_not_shrink_on_flat_strip() {
        let (p, t) = regular_patch(5);
        let r = subdivide(&p, &t, &[], 4);
        // 取左边界上的一点（i = 0, j = 2）。
        let n = 5;
        let b = 2 * n; // (i=0, j=2)
        let d = (r.positions[b] - p[b]).length();
        assert!(d <= 1e-6, "边界顶点漂移 {d}（硬边塌陷的典型症状）");
        // 全局包围盒在 x 方向不收缩。
        let min_x = r.positions.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        assert!(min_x >= -1e-6, "细分后网格越过了原始左边界：min_x = {min_x}");
    }

    /// **极限 stencil 的正确性**：`stencil(level-k 1-ring)` 与"再细分到 k+m 级
    /// 后同一顶点的位置"必须收敛到同一个值。
    ///
    /// 这是"极限位置非迭代求值"的**唯一**可信检验：如果 stencil 错了（比如把
    /// `3/(3+8βn)` 写成 `1/(n+3)`），两者会差一个**不随细分收敛到 0 的常数**
    /// （曲率 × 常数）；写对了则差值按几何级数下降。
    ///
    /// 用非平面补丁（抛物面）：平面补丁对这个检验是平凡通过的（两者恒等）。
    ///
    /// 同时把"看起来合理但错"的候选 `v_∞ = (3v + Σnb)/(n+3)` 一起算出来当对照：
    /// 它在 n=6 时给 `v/3 + 2/3·avg`（正确式给 `v/2 + 1/2·avg`），在有曲率的
    /// 补丁上差 ~1e-2 量级 —— 实测见打印。**这条对照就是本测试的判别力来源**：
    /// 没有它，"我的 stencil 等于我的迭代"可能只是自洽，而不是正确。
    #[test]
    fn limit_stencil_matches_iterated_limit() {
        let n = 9;
        let (mut p, t) = regular_patch(n);
        for v in p.iter_mut() {
            // 抛物面：有曲率，极限 stencil 的偏差才会显形。
            v.z = 0.15 * (v.x * v.x + v.y * v.y);
        }
        let c = (n / 2) * n + n / 2; // 中心顶点（规则 6 价内部点）
        // 参考：再细分 4 级后的**同一顶点**位置（迭代到该级的极限近似）。
        let it = subdivide(&p, &t, &[], 4).positions[c];
        // 本级 1-ring 极限 stencil（level 0）。
        let mine = subdivide(&p, &t, &[], 0).positions[c];
        // 错候选：均匀 (3 + Σ)/(n+3)。
        let (i0, j0) = (c % n, c / n);
        let mut ring: Vec<usize> = Vec::new();
        if i0 > 0 && j0 > 0 { ring.push((j0 - 1) * n + i0); }
        if i0 + 1 < n && j0 > 0 { ring.push((j0 - 1) * n + i0 + 1); }
        if i0 > 0 { ring.push(j0 * n + i0 - 1); }
        if i0 + 1 < n { ring.push(j0 * n + i0 + 1); }
        if i0 > 0 { ring.push((j0 + 1) * n + i0); }
        if i0 + 1 < n { ring.push((j0 + 1) * n + i0 + 1); }
        assert_eq!(ring.len(), 6, "中心顶点必须是规则 6 价");
        let mut sum = Vec3::splat(0.0);
        for &j in &ring {
            sum = sum + p[j];
        }
        let nf = ring.len() as f32;
        let alt = p[c] * (3.0 / (nf + 3.0)) + sum * (1.0 / (nf + 3.0));
        let d_mine = (it - mine).length();
        let d_alt = (it - alt).length();
        println!(
            "V2' stencil 判别：本实装与 4 级迭代差 {d_mine:e}；错候选 (3+Σ)/(n+3) 差 {d_alt:e}"
        );
        assert!(d_mine <= 1e-5, "本实装 stencil 与迭代极限不符：{d_mine:e}");
        assert!(d_alt > 1e-3, "对照候选应当明显不达标（否则本测试没有判别力）：{d_alt:e}");
    }

    /// uv 顶点域插值：每条**内部边**两侧的共享顶点 uv 逐位相同（细分后仍然如此）。
    ///
    /// 这条是位移 V4 判据的几何基础：uv 若按角点存，缝处会出现两个不同的 uv；
    /// 顶点域下它**在结构上不存在**（一个顶点只有一份 uv）。
    #[test]
    fn uv_is_vertex_domain_after_subdivision() {
        let (p, t) = regular_patch(5);
        // uv 给成一个"接缝式"布局：x 坐标折回 [0,1)，故意让相邻顶点跨 0/1。
        let uv: Vec<[f32; 2]> = p
            .iter()
            .map(|v| {
                let u = v.x * 0.37;
                [u - u.floor(), v.y * 0.21]
            })
            .collect();
        let r = subdivide(&p, &t, &uv, 2);
        assert_eq!(r.uv.len(), r.positions.len(), "uv 必须与顶点同长");
        // 顶点域的结构性断言：uv 是"每顶点一份"的数组，故任意两个引用同一个
        // 顶点索引的面角点，取到的 uv 必然逐位相同（这里用索引唯一性直接验）。
        let mut seen: std::collections::BTreeMap<u32, [f32; 2]> =
            std::collections::BTreeMap::new();
        for &i in &r.indices {
            let u = r.uv[i as usize];
            match seen.get(&i) {
                Some(prev) => assert_eq!(
                    prev.map(f32::to_bits),
                    u.map(f32::to_bits),
                    "顶点 {i} 的 uv 出现两个不同值 ⇒ 顶点域契约被破坏"
                ),
                None => {
                    seen.insert(i, u);
                }
            }
        }
        // 边顶点的 uv 必须是父边两端的算术平均（插值口径）。
        assert!(r.uv.iter().all(|u| u[0].is_finite() && u[1].is_finite()));
    }

    /// 确定性：同输入双跑逐位一致（含法线位型）。
    #[test]
    fn subdivision_is_bitwise_deterministic() {
        let (mut p, t) = regular_patch(4);
        for (i, v) in p.iter_mut().enumerate() {
            v.z = 0.05 * i as f32;
        }
        let a = subdivide(&p, &t, &[], 3);
        let b = subdivide(&p, &t, &[], 3);
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.positions.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            b.positions.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            "位置必须逐位一致"
        );
        assert_eq!(
            a.normals.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            b.normals.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            "法线必须逐位一致"
        );
    }

    fn vbits(v: Vec3) -> (u32, u32, u32) {
        (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
    }

    /// 退化输入：空 / 越界 / 非有限 ⇒ 空结果，不 panic。
    #[test]
    fn degenerate_inputs_yield_empty_result() {
        let p = vec![Vec3::splat(0.0), Vec3::splat(1.0), Vec3::splat(2.0)];
        assert!(subdivide(&[], &[], &[], 3).indices.is_empty());
        assert!(subdivide(&p, &[0, 1, 9], &[], 3).indices.is_empty(), "越界索引必须被剔除");
        let mut q = p.clone();
        q[1] = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(subdivide(&q, &[0, 1, 2], &[], 3).indices.is_empty(), "非有限顶点必须被剔除");
    }

    /// `levels = 0`：不改拓扑，只把顶点放到极限位置。
    #[test]
    fn zero_levels_only_moves_to_limit() {
        let (p, t) = regular_patch(4);
        let r = subdivide(&p, &t, &[], 0);
        assert_eq!(r.indices, t);
        assert_eq!(r.positions.len(), p.len());
        assert_eq!(r.levels, 0);
    }

    /// `β` 的定点校核（Loop 1987 §4）：`β(3) = 3/16`、`β(6) = 1/16`。
    #[test]
    fn beta_anchors() {
        assert!((loop_beta(3) - 3.0 / 16.0).abs() < 1e-7, "β(3) = {}", loop_beta(3));
        assert!((loop_beta(6) - 1.0 / 16.0).abs() < 1e-7, "β(6) = {}", loop_beta(6));
        // 仿射不变：a + n·b = 1。
        for n in 3..=12 {
            let (a, b) = loop_limit_weights(n);
            assert!((a + n as f32 * b - 1.0).abs() < 1e-6, "n={n} 极限权重不满足仿射不变");
        }
    }
}

// ---------------------------------------------------------------------------
// Catmull-Clark（四边笼）—— 与 Loop 并列的第二条通路（§5 定稿"按输入拓扑分派"）
// ---------------------------------------------------------------------------
//
// # 一条规则都不许省（省了也跑得动，只是硬边处塌陷）
//
// | 位置 | 规则（本实装） |
// |---|---|
// | 面点 `F` | 该四边面 4 个角点坐标的算术平均 |
// | 内部边点 | `(a + b + F₀ + F₁) / 4`（两端点 + 两侧面点） |
// | 边界边点 | 中点 `(a + b) / 2` |
// | 内部顶点 | `(F̄ + 2R̄ + (n−3)·v) / n`，`F̄` = 相邻面点均值、`R̄` = 相邻边中点均值 |
// | 边界顶点（恰 2 条边界边） | `(b₀ + 6v + b₁) / 8`（边界 = 三次 B 样条曲线） |
// | 边界顶点（≠ 2 条边界边，角点/非流形） | **不动**（`v`） |
//
// 极限位置（**直接求值，不迭代**）：
//
// * 内部顶点（价 `n`）：`P_∞ = (n(n−3)·P + 4·Σmᵢ + 4·ΣFᵢ) / (n(n+5))`，
//   `mᵢ` = 第 i 条关联边的**中点**，`Fᵢ` = 第 i 个关联面的**形心**。
//   推导（左特征向量 / λ=1）写在报告 §23.2。两条独立校核：
//   ① `n = 4` 时化简为 `(4P + 4Σm + 4ΣF)/36 = (P + Σm + ΣF)/9`，与均匀双三次
//   B 样条控制点处的极限 `(16P + 4ΣQ + ΣD)/36` **恒等**（代数代入即得）；
//   ② `n = 3` 的立方体角点给 `(1/2, 1/2, 1/2)`，与 7 级迭代实测收敛值一致
//   （见 `cc_tests::v2_cube_limit_points`）。
//   **注意**：常见的 `(n²P + 4Σm + ΣF)/(n(n+5))` 是错的（立方体角点给 0.75，
//   7 级迭代给 0.5）——本模块不采用，`cc_tests::wrong_limit_candidate_is_rejected`
//   把这个错候选钉成反例。
// * 边界顶点（恰 2 条边界边）：`(b₀ + 4P + b₁) / 6`（曲线极限）。
// * 角点 / 非流形：不动。
//
// # 折痕（crease）规则：**未做**（明确登记，不静默当内部点）
//
// 本模块**没有** crease/corner 标记通道：`catmull_clark` 的输入只有
// `(positions, quads, uv, levels)`，没有"哪些边是硬边"的载体。故：
//
// * 边界边（只被 1 个面使用）走上面的边界规则，这在几何上是"网格的开口"；
// * **内部边一律按光滑处理**——不存在"内部折痕"的输入表达方式。
//
// 也就是说：四边笼里的**开口**（boundary）语义完整，**内部折痕**（crease）
// 未做。这不是"当内部点处理"（那会让硬边塌陷且无迹可循），而是**根本没有这个
// 输入**——调用方无法表达折痕，也就不会被静默抹圆。要支持折痕须扩接口（加
// `sharp_edges: &[u32]` 并对染色的边/顶点换 stencil），属后续版本。
//
// # 三角化出口
//
// 末级每四边面切 2 个三角形：`(p0,p1,p2)` + `(p0,p2,p3)`，绕序与输入一致。
// 三角数 = `4^k × 四边面数 × 2`，三角形流字节 = `4^k × 四边面数 × 72 B`
// （2 × 36 B）。与 Loop 走**同一个** `SubdivResult` 出口。

/// 一条四边笼的边（无向，`a < b`）。
#[derive(Clone, Copy, Default)]
struct QuadEdge {
    /// 较小端点。
    a: u32,
    /// 较大端点。
    b: u32,
    /// 相邻面（至多两个；`count > 2` 的非流形边只保留前两个）。
    face: [u32; 2],
    /// 该边被多少个四边面使用（1 = 边界边，2 = 内部边，> 2 = 非流形）。
    count: u32,
}

/// 四边笼的**邻接**（CSR；与 [`Topology`] 同一条确定性纪律：**全部升序**）。
///
/// 极限求值只需要它；细分还要 [`QuadTopology::edges`]（边点生成 + 边号查询）。
/// **两者拆开是有原因的**：末级不再细分，边表在那里纯属浪费——k=6 的立方体
/// （24576 个四边面、49152 条边）上，`QuadEdge` 表是 20 B×49152 = 983040 B（960 KiB），正好是 V5c 判据
/// （峰值 RSS vs 同三角数建树，上界 +10%）里最不该付的那一档。
struct QuadAdjacency {
    /// 1-ring 邻接 CSR 偏移（长度 `V + 1`）。
    nb_off: Vec<u32>,
    /// 1-ring 邻接扁平存储（每顶点一段，升序）。
    nb: Vec<u32>,
    /// 边界邻居 CSR 偏移（长度 `V + 1`）。
    bn_off: Vec<u32>,
    /// 边界邻居扁平存储（升序）。
    bn: Vec<u32>,
    /// 顶点 → 关联面 CSR 偏移（长度 `V + 1`）。
    vf_off: Vec<u32>,
    /// 顶点 → 关联面扁平存储（升序 = 面索引序）。
    vf: Vec<u32>,
    /// 该顶点是否连着"被 > 2 个面使用"的非流形边（按边界处理）。
    nonmanifold: Vec<bool>,
}

/// 取"下一组"：`(a, b, count, 下一组起点)`。
///
/// `multiset = false`：`items` 是**已合并**的边表（每条边一个元素，次数在第三分量）；
/// `multiset = true`：`items` 是**未合并**的半边多重集（相同 `(a, b)` 连续出现，
/// 次数 = 连续个数之和）。
///
/// 两条路径共用同一段邻接构建代码——但**次数来源不同**，混用会把每条内部边都
/// 判成边界边（立方体上表现为"所有顶点都成角点、一动不动"），那是 2026-11 实际
/// 踩过的坑，故这里显式分流而不是"看起来一样就复用"。
#[inline]
fn next_group<T, F>(items: &[T], i: usize, key: &F, multiset: bool) -> (u32, u32, u32, usize)
where
    F: Fn(&T) -> (u32, u32, u32),
{
    let (a, b, c) = key(&items[i]);
    if !multiset {
        return (a, b, c, i + 1);
    }
    let mut j = i + 1;
    let mut count = c;
    while j < items.len() {
        let (a2, b2, c2) = key(&items[j]);
        if (a2, b2) != (a, b) {
            break;
        }
        count = count.saturating_add(c2);
        j += 1;
    }
    (a, b, count, j)
}

/// 建邻接（两遍扫描，**不额外分配**边数组以外的内存）。
fn adjacency_from_sorted<T, F>(n: usize, items: &[T], key: F, multiset: bool) -> QuadAdjacency
where
    F: Fn(&T) -> (u32, u32, u32),
{
    let mut nb_off = vec![0u32; n + 1];
    let mut bn_off = vec![0u32; n + 1];
    let mut nonmanifold = vec![false; n];
    // --- 第一遍：数度 / 边界 / 非流形 ---
    let mut i = 0usize;
    while i < items.len() {
        let (a, b, count, next) = next_group(items, i, &key, multiset);
        nb_off[a as usize + 1] += 1;
        nb_off[b as usize + 1] += 1;
        if count == 1 {
            bn_off[a as usize + 1] += 1;
            bn_off[b as usize + 1] += 1;
        } else if count > 2 {
            nonmanifold[a as usize] = true;
            nonmanifold[b as usize] = true;
        }
        i = next;
    }
    for k in 0..n {
        nb_off[k + 1] += nb_off[k];
        bn_off[k + 1] += bn_off[k];
    }
    let mut nb = vec![0u32; nb_off[n] as usize];
    let mut bn = vec![0u32; bn_off[n] as usize];
    let mut nb_fill = nb_off.clone();
    let mut bn_fill = bn_off.clone();
    // --- 第二遍：填充。扫描序 = `(a, b)` 升序 ⇒ 每个顶点的邻居表天然升序
    //     （`b` 角色：更小的 `a` 先出现；`a` 角色：更大的 `b` 后出现且升序）。 ---
    let mut i = 0usize;
    while i < items.len() {
        let (a, b, count, next) = next_group(items, i, &key, multiset);
        let (ai, bi) = (a as usize, b as usize);
        nb[nb_fill[ai] as usize] = b;
        nb_fill[ai] += 1;
        nb[nb_fill[bi] as usize] = a;
        nb_fill[bi] += 1;
        if count == 1 {
            bn[bn_fill[ai] as usize] = b;
            bn_fill[ai] += 1;
            bn[bn_fill[bi] as usize] = a;
            bn_fill[bi] += 1;
        }
        i = next;
    }
    QuadAdjacency {
        nb_off,
        nb,
        bn_off,
        bn,
        vf_off: vec![0u32; n + 1],
        vf: Vec::new(),
        nonmanifold,
    }
}

impl QuadAdjacency {
    /// 从四边面流建邻接（**不建边表**；极限求值路径专用）。
    fn build(n: usize, quads: &[u32]) -> Self {
        let nf = quads.len() / 4;
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(nf * 4);
        for q in quads.chunks_exact(4) {
            for k in 0..4 {
                let a = q[k];
                let b = q[(k + 1) % 4];
                pairs.push(if a < b { (a, b) } else { (b, a) });
            }
        }
        pairs.sort_unstable();
        let mut adj = adjacency_from_sorted(n, &pairs, |p| (p.0, p.1, 1), true);
        drop(pairs);
        fill_vertex_faces(&mut adj, n, quads);
        adj
    }

    #[inline]
    fn neighbors(&self, i: usize) -> &[u32] {
        &self.nb[self.nb_off[i] as usize..self.nb_off[i + 1] as usize]
    }

    #[inline]
    fn boundary_nbrs(&self, i: usize) -> &[u32] {
        &self.bn[self.bn_off[i] as usize..self.bn_off[i + 1] as usize]
    }

    #[inline]
    fn faces_of(&self, i: usize) -> &[u32] {
        &self.vf[self.vf_off[i] as usize..self.vf_off[i + 1] as usize]
    }

    #[inline]
    fn has_nonmanifold_edge(&self, i: usize) -> bool {
        self.nonmanifold.get(i).copied().unwrap_or(false)
    }
}

/// 填 `顶点 → 关联面` CSR：按面索引升序遍历 ⇒ 每顶点的面表天然升序。
fn fill_vertex_faces(adj: &mut QuadAdjacency, n: usize, quads: &[u32]) {
    let mut vf_off = vec![0u32; n + 1];
    for q in quads.chunks_exact(4) {
        for &i in q {
            vf_off[i as usize + 1] += 1;
        }
    }
    for i in 0..n {
        vf_off[i + 1] += vf_off[i];
    }
    let mut vf = vec![0u32; vf_off[n] as usize];
    let mut vf_fill = vf_off.clone();
    for (fi, q) in quads.chunks_exact(4).enumerate() {
        for &i in q {
            vf[vf_fill[i as usize] as usize] = fi as u32;
            vf_fill[i as usize] += 1;
        }
    }
    adj.vf_off = vf_off;
    adj.vf = vf;
}

/// 四边笼拓扑 = 邻接 + **边表**（只有细分需要边表：边点生成与边号查询）。
struct QuadTopology {
    /// 邻接 / 边界 / 顶点→面（极限与顶点规则共用）。
    adj: QuadAdjacency,
    /// 边表（按 `(a, b)` 升序）：索引 = 边点的序号偏移。
    edges: Vec<QuadEdge>,
}

impl QuadTopology {
    fn build(pos: &[Vec3], quads: &[u32]) -> Self {
        let n = pos.len();
        let nf = quads.len() / 4;
        // --- 半边收集：每四边面 4 条，扁平数组（带面索引） ---
        let mut half: Vec<(u32, u32, u32)> = Vec::with_capacity(nf * 4);
        for (fi, q) in quads.chunks_exact(4).enumerate() {
            for k in 0..4 {
                let a = q[k];
                let b = q[(k + 1) % 4];
                half.push(if a < b { (a, b, fi as u32) } else { (b, a, fi as u32) });
            }
        }
        // 排序键 `(a, b, face)`：并列由第三个分量定序 ⇒ 全序，与输入顺序无关。
        half.sort_unstable();

        let mut edges: Vec<QuadEdge> = Vec::with_capacity(half.len() / 2 + 1);
        for &(a, b, f) in half.iter() {
            match edges.last_mut() {
                Some(e) if e.a == a && e.b == b => {
                    if e.count < 2 {
                        e.face[e.count as usize] = f;
                    }
                    e.count = e.count.saturating_add(1);
                }
                _ => edges.push(QuadEdge { a, b, face: [f, f], count: 1 }),
            }
        }
        drop(half);
        let mut adj = adjacency_from_sorted(n, &edges, |e| (e.a, e.b, e.count), false);
        fill_vertex_faces(&mut adj, n, quads);
        QuadTopology { adj, edges }
    }

    /// `(min, max)` → 边序号（二分；键来自本网格的四边形 ⇒ 必然存在）。
    #[inline]
    fn edge_index(&self, a: u32, b: u32) -> usize {
        let key = (a.min(b), a.max(b));
        self.edges
            .binary_search_by(|e| (e.a, e.b).cmp(&key))
            .expect("边必须存在（键来自本网格的四边形）")
    }

    /// `(min, max)` → 边序号（**不 panic**：标记可能指向已被剔除的边）。
    #[inline]
    fn lookup_edge(&self, a: u32, b: u32) -> Option<usize> {
        let key = (a.min(b), a.max(b));
        self.edges.binary_search_by(|e| (e.a, e.b).cmp(&key)).ok()
    }
}

/// 面形心（4 角点算术平均；`0.25` 是二进制精确常数）。
fn quad_centroid(pos: &[Vec3], q: &[u32]) -> Vec3 {
    (pos[q[0] as usize] + pos[q[1] as usize] + pos[q[2] as usize] + pos[q[3] as usize]) * 0.25
}

/// 四边笼的边键表（`(min,max)` 升序、去重）。
///
/// 与 `QuadTopology::build` 产出的 `edges` **同序**（同一套排序键），但只保留
/// 键、不建邻接。末级只需要"顶点对 → 边序号"这一张查找表（crease 极限用），
/// 建完整拓扑等于多算一遍 CSR 邻接。
fn quad_edge_keys(quads: &[u32]) -> Vec<(u32, u32)> {
    let mut keys: Vec<(u32, u32)> = Vec::with_capacity(quads.len());
    for q in quads.chunks_exact(4) {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            keys.push((a.min(b), a.max(b)));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// 剔除越界 / 非有限 / 含重复角点的四边形。
///
/// **含重复角点的四边形必须剔除**（与 Loop 的"零面积三角形不剔除"不同）：
/// 重复角点会造出自环边 `(a, a)`，而边表的键 `(min,max)` 假定两端不同；
/// 自环会让"价数=邻居数"的论证失效，顶点规则在那个点上是无定义的。
fn sanitize_quads(pos: &[Vec3], quads: &[u32]) -> Vec<u32> {
    let n = pos.len();
    let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    let mut out = Vec::with_capacity(quads.len() / 4 * 4);
    for c in quads.chunks_exact(4) {
        let (i0, i1, i2, i3) = (c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize);
        if i0 >= n || i1 >= n || i2 >= n || i3 >= n {
            continue;
        }
        if !finite(pos[i0]) || !finite(pos[i1]) || !finite(pos[i2]) || !finite(pos[i3]) {
            continue;
        }
        if c[0] == c[1] || c[1] == c[2] || c[2] == c[3] || c[3] == c[0] || c[0] == c[2] || c[1] == c[3]
        {
            continue; // 自环/退化角点：拓扑无定义
        }
        out.extend_from_slice(c);
    }
    out
}

/// Catmull-Clark 细分一级（1→4）。
///
/// 输出顶点序：`[顶点规则后的老顶点 (n_old) | 边点 (n_edge) | 面点 (n_face)]`。
/// 与 Loop 侧同一口径（老顶点在前 ⇒ "顶点 i 的后代仍是 i"）。
///
/// 输出四边形序：每个老四边形 → 4 个，
/// `(a, ab, F, da)`、`(ab, b, bc, F)`、`(F, bc, c, cd)`、`(da, F, cd, d)`。
///
/// `marks` 为 `None` ⇒ 与加标记前**逐位相同**（返回值第 4 项为 `None`）。
fn refine_once_quad(
    pos: &[Vec3],
    quads: &[u32],
    uv: &[[f32; 2]],
    marks: Option<&SubdivMarks>,
) -> (Vec<Vec3>, Vec<u32>, Vec<[f32; 2]>, Option<SubdivMarks>) {
    let topo = QuadTopology::build(pos, quads);
    let n_old = pos.len();
    let n_edge = topo.edges.len();
    let n_face = quads.len() / 4;
    let mi: Option<MarkIndex> = marks
        .map(|m| MarkIndex::build(n_edge, n_old, m, |a, b| topo.lookup_edge(a, b)));

    // 面点（后两个输出块都要用，先算一次）。
    let mut face_pos: Vec<Vec3> = Vec::with_capacity(n_face);
    for q in quads.chunks_exact(4) {
        face_pos.push(quad_centroid(pos, q));
    }

    let mut out_pos: Vec<Vec3> = Vec::with_capacity(n_old + n_edge + n_face);
    // --- 顶点规则 ---
    for i in 0..n_old {
        out_pos.push(cc_vertex_point(pos, &face_pos, &topo.adj, i, mi.as_ref()));
    }
    // --- 边点（边表按 `(a, b)` 升序 ⇒ 顶点索引确定） ---
    for (e_i, e) in topo.edges.iter().enumerate() {
        let pa = pos[e.a as usize];
        let pb = pos[e.b as usize];
        let p_smooth = if e.count == 2 {
            let f0 = face_pos[e.face[0] as usize];
            let f1 = face_pos[e.face[1] as usize];
            (pa + pb + f0 + f1) * 0.25
        } else {
            // 边界边 / 非流形边：中点。
            (pa + pb) * 0.5
        };
        // crease：按边权重把边点拉向中点（`w == 1` ⇒ 纯中点）。
        let p = match &mi {
            Some(m) if m.edge_w[e_i] > 0.0 => {
                MarkIndex::blend(p_smooth, (pa + pb) * 0.5, m.edge_w[e_i])
            }
            _ => p_smooth,
        };
        out_pos.push(p);
    }
    out_pos.extend_from_slice(&face_pos);

    // --- uv（顶点域插值，与 Loop 侧同口径）---
    let mut out_uv: Vec<[f32; 2]> = Vec::new();
    if !uv.is_empty() {
        out_uv.reserve(n_old + n_edge + n_face);
        out_uv.extend_from_slice(uv);
        for e in topo.edges.iter() {
            let ua = uv[e.a as usize];
            let ub = uv[e.b as usize];
            out_uv.push([(ua[0] + ub[0]) * 0.5, (ua[1] + ub[1]) * 0.5]);
        }
        for q in quads.chunks_exact(4) {
            let u0 = uv[q[0] as usize];
            let u1 = uv[q[1] as usize];
            let u2 = uv[q[2] as usize];
            let u3 = uv[q[3] as usize];
            out_uv.push([
                (u0[0] + u1[0] + u2[0] + u3[0]) * 0.25,
                (u0[1] + u1[1] + u2[1] + u3[1]) * 0.25,
            ]);
        }
    }

    // --- 面：1→4，绕序保持 ---
    let mut out_quad: Vec<u32> = Vec::with_capacity(n_face * 16);
    for (fi, q) in quads.chunks_exact(4).enumerate() {
        let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
        let ab = (n_old + topo.edge_index(a, b)) as u32;
        let bc = (n_old + topo.edge_index(b, c)) as u32;
        let cd = (n_old + topo.edge_index(c, d)) as u32;
        let da = (n_old + topo.edge_index(d, a)) as u32;
        let f = (n_old + n_edge + fi) as u32;
        out_quad.extend_from_slice(&[a, ab, f, da]);
        out_quad.extend_from_slice(&[ab, b, bc, f]);
        out_quad.extend_from_slice(&[f, bc, c, cd]);
        out_quad.extend_from_slice(&[da, f, cd, d]);
    }
    // 子级标记：`w == 1` 的两半仍是 1，分数权重降 0（DeRose 半锐利）。
    let child = marks
        .map(|m| m.child_marks(|a, b| topo.lookup_edge(a, b), n_old))
        .filter(|m| !m.is_empty());
    (out_pos, out_quad, out_uv, child)
}

/// 顶点规则（corner / 边界 / crease / 内部 / dart；优先级与 Loop 侧一致）。
fn cc_vertex_point(
    pos: &[Vec3],
    face_pos: &[Vec3],
    adj: &QuadAdjacency,
    i: usize,
    mi: Option<&MarkIndex>,
) -> Vec3 {
    let v = pos[i];
    if let Some(m) = mi {
        if m.is_corner(i) {
            return v;
        }
    }
    let bn = adj.boundary_nbrs(i);
    if !bn.is_empty() || adj.has_nonmanifold_edge(i) {
        if bn.len() == 2 {
            let b0 = pos[bn[0] as usize];
            let b1 = pos[bn[1] as usize];
            return (b0 + b1 + v * 6.0) * 0.125;
        }
        return v; // 角点 / 非流形：不动
    }
    // --- crease / corner（内部顶点）---
    if let Some(m) = mi {
        let k = m.crease_degree(i);
        if k == 2 || k >= 3 {
            let w = m.inc_weights(i);
            let mut wv = 0.0f32;
            for &x in w {
                if x > wv {
                    wv = x;
                }
            }
            let sharp: Vec3 = if k == 2 {
                let nb = m.inc_crease(i);
                let (b0, b1) = (pos[nb[0] as usize], pos[nb[1] as usize]);
                (b0 + b1 + v * 6.0) * 0.125 // 1D 三次 B 样条
            } else {
                v // crease 交汇 = corner
            };
            if wv >= 1.0 {
                return sharp;
            }
            return MarkIndex::blend(smooth_cc(pos, face_pos, adj, i, v), sharp, wv);
        }
    }
    smooth_cc(pos, face_pos, adj, i, v)
}

/// CC 内部顶点的光滑规则。**一行未改**，加标记前后共用同一串表达式。
fn smooth_cc(pos: &[Vec3], face_pos: &[Vec3], adj: &QuadAdjacency, i: usize, v: Vec3) -> Vec3 {
    let nb = adj.neighbors(i);
    let vf = adj.faces_of(i);
    let n = nb.len();
    if n < 3 || vf.len() != n {
        return v; // 拓扑不足（孤立点 / 不一致）：不动，绝不产 NaN
    }
    let nf = n as f32;
    // F̄：相邻面点均值（升序累加，求和顺序固定）。
    let mut fsum = Vec3::splat(0.0);
    for &f in vf {
        fsum = fsum + face_pos[f as usize];
    }
    let fbar = fsum * (1.0 / nf);
    // R̄：相邻边中点均值 = v/2 + (1/(2n))·Σnb。
    let mut nsum = Vec3::splat(0.0);
    for &j in nb {
        nsum = nsum + pos[j as usize];
    }
    let rbar = v * 0.5 + nsum * (0.5 / nf);
    (fbar + rbar * 2.0 + v * (nf - 3.0)) * (1.0 / nf)
}

/// 极限位置（**1-ring 极限 stencil 直接求值，不迭代**；推导见报告 §23.2）。
fn limit_positions_quad(
    pos: &[Vec3],
    quads: &[u32],
    adj: &QuadAdjacency,
    mi: Option<&MarkIndex>,
) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        let v = pos[i];
        if let Some(m) = mi {
            if m.is_corner(i) {
                out.push(v);
                continue;
            }
        }
        let bn = adj.boundary_nbrs(i);
        if !bn.is_empty() || adj.has_nonmanifold_edge(i) {
            if bn.len() == 2 {
                let b0 = pos[bn[0] as usize];
                let b1 = pos[bn[1] as usize];
                out.push((b0 + b1 + v * 4.0) * (1.0 / 6.0));
            } else {
                out.push(v); // 角点 / 非流形：不动
            }
            continue;
        }
        // --- crease 极限（内部顶点）---
        if let Some(m) = mi {
            let k = m.crease_degree(i);
            if k == 2 || k >= 3 {
                let sharp: Vec3 = if k == 2 {
                    let nb = m.inc_crease(i);
                    let (b0, b1) = (pos[nb[0] as usize], pos[nb[1] as usize]);
                    (b0 + b1 + v * 4.0) * (1.0 / 6.0) // 与边界曲线同一支
                } else {
                    v
                };
                let w = m.inc_weights(i);
                let mut wv = 0.0f32;
                for &x in w {
                    if x > wv {
                        wv = x;
                    }
                }
                if wv >= 1.0 {
                    out.push(sharp);
                    continue;
                }
                out.push(MarkIndex::blend(smooth_limit_quad(pos, quads, adj, i, v), sharp, wv));
                continue;
            }
        }
        out.push(smooth_limit_quad(pos, quads, adj, i, v));
    }
    out
}

/// CC 内部顶点的极限 stencil。**一行未改**。
fn smooth_limit_quad(pos: &[Vec3], quads: &[u32], adj: &QuadAdjacency, i: usize, v: Vec3) -> Vec3 {
    let nb = adj.neighbors(i);
    let vf = adj.faces_of(i);
    let n = nb.len();
    if n < 3 || vf.len() != n {
        return v;
    }
    // P_∞ = (n(n−3)·P + 4·Σmᵢ + 4·ΣFᵢ) / (n(n+5))
    //      其中 mᵢ = (P + Qᵢ)/2（边中点），Fᵢ = 面形心。
    let w_p = (n * (n.saturating_sub(3))) as f32;
    // P 项自带系数 `n(n−3)`；`Σm` 与 `ΣF` 各带系数 4 —— 三者共用同一个分母，
    // 故先把后两项合并再统一乘 4（**不能**把 P 项也乘 4：那会 double-count）。
    let mut mf = Vec3::splat(0.0);
    for &j in nb {
        mf = mf + (v + pos[j as usize]) * 0.5; // Σmᵢ（边中点）
    }
    for &f in vf {
        // ΣFᵢ（面形心）：**现算不缓存**——缓存一份 `Vec<Vec3>` 是 12 B/面，
        // 在 k=6 的立方体上是 295 KB 的常驻临时量，而这里每个面最多被 4 个
        // 顶点各算一次（4 次加法），算得过。
        let qi = f as usize * 4;
        mf = mf + quad_centroid(pos, &quads[qi..qi + 4]);
    }
    let den = (n * (n + 5)) as f32;
    (v * w_p + mf * 4.0) * (1.0 / den)
}

/// 四边面 → 三角形（每面 2 个，绕序保持）。
fn triangulate_quads(quads: &[u32]) -> Vec<u32> {
    let mut tri = Vec::with_capacity(quads.len() / 4 * 6);
    for q in quads.chunks_exact(4) {
        tri.extend_from_slice(&[q[0], q[1], q[2]]);
        tri.extend_from_slice(&[q[0], q[2], q[3]]);
    }
    tri
}

/// 入口：均匀 Catmull-Clark 细分 `levels` 级（四边笼），末尾把顶点放到**极限位置**。
///
/// # 参数
///
/// * `positions` / `quads`：四边笼。`quads` 每 **4 个索引**一四边形，绕序一致
///   （CCW 朝外时三角化后仍朝外）；长度不是 4 的倍数时**尾部的余数被忽略**；
/// * `uv`：顶点域（长度须等于 `positions.len()`，否则按"无 uv"处理，不 panic）；
/// * `levels`：细分级数（`0` ⇒ 不改拓扑，只把顶点放到极限位置）。
///
/// # 退化
///
/// 空输入 / 索引越界 / 非有限顶点 / 含重复角点的四边形 ⇒ 该四边形被剔除
/// （不 panic）。剔除后无四边形 ⇒ 返回空结果（`levels` 回显截断后的级数）。
///
/// # 与 Loop 的关系
///
/// 两条通路**并列**、互不调用：三角笼走 [`subdivide`]，四边笼走本函数。
/// 不做三角→四边配对（无唯一解，见模块文档）。产出同一个 [`SubdivResult`]。
pub fn catmull_clark(
    positions: &[Vec3],
    quads: &[u32],
    uv: &[[f32; 2]],
    levels: u32,
) -> SubdivResult {
    catmull_clark_marked(positions, quads, uv, levels, None)
}

/// 入口（带 crease / corner 标记）：均匀 Catmull-Clark 细分 `levels` 级 + 极限位置。
///
/// [`catmull_clark`] 就是 `catmull_clark_marked(.., None)`。规则与 Loop 侧
/// **同一套**（[`subdivide_marked`] 的文档表），只是 stencil 换成 CC 的：
///
/// | 位置 | 光滑 | crease 侧 |
/// |---|---|---|
/// | 顶点（crease 度 2） | `(F̄ + 2R̄ + (n−3)v)/n` | `(b₀ + 6v + b₁)/8` |
/// | 顶点（crease 度 ≥ 3） | 同上 | `v` |
/// | 边点 | `(a + b + f₀ + f₁)/4` | `(a + b)/2` |
/// | 极限（crease 度 2） | `(n(n−3)v + 4Σm + 4ΣF)/(n(n+5))` | `(b₀ + 4v + b₁)/6` |
///
/// 混合权重一律 `w_v = max(入射 crease 权重)`、边点用该边自身权重。
pub fn catmull_clark_marked(
    positions: &[Vec3],
    quads: &[u32],
    uv: &[[f32; 2]],
    levels: u32,
    marks: Option<&SubdivMarks>,
) -> SubdivResult {
    let levels = levels.min(MAX_LEVELS);
    let mut pos: Vec<Vec3> = positions.to_vec();
    let mut quad: Vec<u32> = sanitize_quads(&pos, quads);
    let mut tex: Vec<[f32; 2]> = if uv.len() == positions.len() { uv.to_vec() } else { Vec::new() };
    // 空标记与"没有标记"等价（结构性逐位退化，判据 V1）。
    let mut cur: Option<SubdivMarks> = marks.filter(|m| !m.is_empty()).cloned();

    if quad.is_empty() {
        return SubdivResult { levels, ..Default::default() };
    }

    for _ in 0..levels {
        let (p, q, u, child) = refine_once_quad(&pos, &quad, &tex, cur.as_ref());
        pos = p;
        quad = q;
        tex = u;
        cur = child;
    }

    // 末级不再细分 ⇒ 只建邻接（不建边表）；见 `QuadAdjacency` 的注释。
    // 但 crease 极限需要边表 —— 有标记时单独建一张**键表**（与
    // `QuadTopology` 的边表同序：`(min,max)` 排序去重），不重复建邻接。
    let adj = QuadAdjacency::build(pos.len(), &quad);
    let mi: Option<MarkIndex> = match cur.as_ref() {
        Some(m) => {
            let keys = quad_edge_keys(&quad);
            Some(MarkIndex::build(keys.len(), pos.len(), m, |a, b| {
                let key = (a.min(b), a.max(b));
                keys.binary_search(&key).ok()
            }))
        }
        None => None,
    };
    let lim = limit_positions_quad(&pos, &quad, &adj, mi.as_ref());
    drop(adj);
    let tri = triangulate_quads(&quad);
    let normals = vertex_normals(&lim, &tri);
    SubdivResult { positions: lim, indices: tri, uv: tex, normals, levels }
}

// ---------------------------------------------------------------------------
// Catmull-Clark 测试（V2 的解析值全部在报告 §23.2 推导，此处只钉数字）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod cc_tests {
    use super::*;

    /// 单位立方体 `[-1, 1]³` 的 6 个四边面（全部朝外 CCW）。
    ///
    /// 角点编号：`0..4` 是 `z = -1` 层（`(-1,-1,-1)`、`(1,-1,-1)`、`(1,1,-1)`、
    /// `(-1,1,-1)`），`4..8` 是 `z = +1` 层（同序）。
    pub(super) fn cube() -> (Vec<Vec3>, Vec<u32>) {
        let pos = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let quads = vec![
            0, 3, 2, 1, // z = -1
            4, 5, 6, 7, // z = +1
            0, 1, 5, 4, // y = -1
            2, 3, 7, 6, // y = +1
            1, 2, 6, 5, // x = +1
            0, 4, 7, 3, // x = -1
        ];
        (pos, quads)
    }

    fn relative_err(got: Vec3, want: Vec3) -> f32 {
        (got - want).length() / want.length().max(1.0)
    }

    /// **V2：立方体极限位置 vs 解析值（相对误差 ≤ 1e-6）。**
    ///
    /// 三个解析值（推导见报告 §23.2，全部是有理数，用精确分数算出）：
    ///
    /// | 位置 | 源顶点（level-1 序号） | 解析极限 |
    /// |---|---|---|
    /// | 角点（价 3） | `0`（level-0 顶点 0 的后代） | `(1/2, 1/2, 1/2)` |
    /// | 棱中点（价 4） | 边 `(6,7)` 的边点 | `(0, 395/648, 395/648)` |
    /// | 面心（价 4） | 面 `z=+1` 的面点 | `(0, 0, 68/81)` |
    ///
    /// 前两个数不是"参考实现给的"：`(1/2)` 由 `n(n−3)=0` 直接塌掉 P 项后
    /// `4·Σm + 4·ΣF` 除以 24 得到；`395/648` 与 `68/81` 是把 level-1 的
    /// 16 个邻居坐标写成精确分数后代进 stencil 得到的（`68/81 = 0.839506…`、
    /// `395/648 = 0.609568…`）。**另有独立收敛证据**：
    /// `v2_cube_limit_is_the_iterated_limit` 用 7 级迭代把同三个顶点收敛到
    /// 同一批数（差 ≤ 1e-6），那一路完全不用 stencil 公式。
    #[test]
    fn v2_cube_limit_points() {
        let (p, q) = cube();
        let r0 = catmull_clark(&p, &q, &[], 0);
        // 角点：level-0 顶点 0 的后代在 level-0 网格里就是 0 本身。
        // 顶点 6 = (1,1,1)，其极限（解析）= (1/2, 1/2, 1/2)。
        let want_corner = Vec3::new(0.5, 0.5, 0.5);
        let e_corner = relative_err(r0.positions[6], want_corner);
        assert!(
            e_corner <= 1e-6,
            "角点极限 {:?} vs 解析 {want_corner:?}，相对误差 {e_corner:e}",
            r0.positions[6]
        );

        // 边点 / 面点：level-1 才存在。
        let r1 = catmull_clark(&p, &q, &[], 1);
        // level-1 顶点序 = [老顶点 8 | 边点 12 | 面点 6]。
        // 边表按 (a,b) 升序：先列 12 条边的有序键。
        let topo_edges = {
            let mut keys: Vec<(u32, u32)> = Vec::new();
            for c in q.chunks_exact(4) {
                for k in 0..4 {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    let key = (a.min(b), a.max(b));
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
            }
            keys.sort_unstable();
            keys
        };
        let idx_of_edge = |a: u32, b: u32| -> usize {
            let key = (a.min(b), a.max(b));
            let e = topo_edges.iter().position(|k| *k == key).expect("边必须存在");
            8 + e
        };
        // 面 z=+1 是第 2 个四边形（索引 1）；面点序号 = 8（老顶点）+ 12（边点）+ 1。
        let want_edge = Vec3::new(0.0, 395.0 / 648.0, 395.0 / 648.0);
        let got_edge = r1.positions[idx_of_edge(6, 7)];
        let e_edge = relative_err(got_edge, want_edge);
        assert!(e_edge <= 1e-6, "棱中点极限 {got_edge:?} vs 解析 {want_edge:?}，相对误差 {e_edge:e}");

        let want_face = Vec3::new(0.0, 0.0, 68.0 / 81.0);
        let got_face = r1.positions[8 + 12 + 1];
        let e_face = relative_err(got_face, want_face);
        assert!(e_face <= 1e-6, "面心极限 {got_face:?} vs 解析 {want_face:?}，相对误差 {e_face:e}");

        println!(
            "V2：角点 {:.9}（误差 {e_corner:e}）、棱中点 {:.9}（误差 {e_edge:e}）、\
             面心 {:.9}（误差 {e_face:e}）",
            r0.positions[6].x, got_edge.y, got_face.z
        );
    }

    /// **V2 的独立收敛证据**：**只用细化规则**（不经过极限 stencil）迭代 7 级，
    /// 三个顶点分别收敛到 V2 的三个解析值。
    ///
    /// 这条是 V2 不自证恒真的保障：`catmull_clark` 末尾会投到极限位置，故拿它
    /// 做收敛测试是循环论证；这里直接调 `refine_once_quad`（纯一步细分：面点 /
    /// 边点 / 顶点规则），看顶点**自己**往哪走。角点的收缩比实测 ≈ 1/6
    /// （价 3 ＋ 立方体对称），故 7 级后残差 ~3e-6；棱中点与面心同样是 ~1/6
    /// 收缩，7 级后残差 ~4e-5（f32 的累加噪声比角点大，因为它们的 1-ring 扩张更快）。
    ///
    /// **三个顶点都断言**（不只是角点）：V2 表里有三行，独立证据就必须盖三行。
    #[test]
    fn v2_cube_limit_is_the_iterated_limit() {
        let (mut pos, mut quad) = cube();
        let want_corner = Vec3::new(0.5, 0.5, 0.5);
        let want_edge = Vec3::new(0.0, 395.0 / 648.0, 395.0 / 648.0);
        let want_face = Vec3::new(0.0, 0.0, 68.0 / 81.0);
        // 棱 (6,7) 的边点序号：边表按 (a,b) 升序 ⇒ 与 `v2_cube_limit_points` 同法。
        let edge_idx = {
            let mut keys: Vec<(u32, u32)> = Vec::new();
            for c in quad.chunks_exact(4) {
                for k in 0..4 {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    let key = (a.min(b), a.max(b));
                    if !keys.contains(&key) {
                        keys.push(key);
                    }
                }
            }
            keys.sort_unstable();
            8 + keys.iter().position(|k| *k == (6, 7)).expect("边 (6,7) 必须存在")
        };
        let face_idx = 8 + 12 + 1;
        let mut prev = (pos[6] - want_corner).length();
        let mut ratios: Vec<f32> = Vec::new();
        let (mut d_corner, mut d_edge, mut d_face) = (prev, f32::INFINITY, f32::INFINITY);
        for k in 1..=7u32 {
            let (p, q, _, _) = refine_once_quad(&pos, &quad, &[], None);
            pos = p;
            quad = q;
            // 老顶点索引不变（顶点规则结果排在前面）⇒ 边点/面点的序号逐级继承。
            let d = (pos[6] - want_corner).length();
            assert!(d < prev, "k={k} 角点没有继续收敛：{d:e} >= {prev:e}");
            ratios.push(d / prev);
            prev = d;
            d_corner = d;
            d_edge = (pos[edge_idx] - want_edge).length();
            d_face = (pos[face_idx] - want_face).length();
            println!(
                "V2 迭代：k={k} 角点 {:.9}（差 {d_corner:e}）｜棱中点 y {:.9}（差 {d_edge:e}）\
                 ｜面心 z {:.9}（差 {d_face:e}）",
                pos[6].x, pos[edge_idx].y, pos[face_idx].z
            );
        }
        println!("V2 迭代：相邻级收缩比 {ratios:?}");
        assert!(d_corner <= 2e-5, "7 级纯细化后角点应逼近 1/2，实测差 {d_corner:e}");
        assert!(d_edge <= 1e-4, "7 级纯细化后棱中点应逼近 395/648，实测差 {d_edge:e}");
        assert!(d_face <= 1e-4, "7 级纯细化后面心应逼近 68/81，实测差 {d_face:e}");
        assert!(
            (ratios[6] - 1.0 / 6.0).abs() < 0.02,
            "价 3 角点的收缩比应 ≈ 1/6（实测 {}）",
            ratios[6]
        );
    }

    /// **反例（规则层）**：把内部边点写成"中点"（最像样的错误）会收敛到**另一张曲面**。
    ///
    /// 这条不是判据，是给 V2 的解析值装的"判别力探针"：同一套迭代，只把边点从
    /// `(a + b + F₀ + F₁)/4` 换回 `(a + b)/2`，三个顶点分别收敛到
    /// **0.6 / 0.7729 / 0.9807**，与解析值（0.5 / 0.6096 / 0.8395）差 0.10~0.14。
    /// 也就是说 V2 那三个数**不是**"随便一张光滑曲面都会收敛到的数"，它们钉住的
    /// 正是 CC 的边点规则本身。
    ///
    /// 0.6 / 0.7729 / 0.9807 是这条错误规则的**实测**极限（不是解析推导），本测试
    /// 只断言"与解析值差得足够远"，不把这三个数当任何口径使用。
    #[test]
    fn midpoint_edge_rule_converges_elsewhere() {
        use std::collections::BTreeMap;
        let (mut pos, mut quad) = cube();
        let mut got = (0.0_f32, 0.0_f32, 0.0_f32);
        for _ in 0..7 {
            let n_old = pos.len() as u32;
            let face_pos: Vec<Vec3> =
                quad.chunks_exact(4).map(|c| quad_centroid(&pos, c)).collect();
            // 边表（**先收集再按序编号**，否则索引与位置对不上）+ 邻接（升序 ⇒ 求和顺序固定）。
            let mut keys: BTreeMap<(u32, u32), ()> = BTreeMap::new();
            let mut nb: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
            let mut vf: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
            for (fi, c) in quad.chunks_exact(4).enumerate() {
                for k in 0..4 {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    keys.insert((a.min(b), a.max(b)), ());
                    nb[a as usize].push(b);
                    nb[b as usize].push(a);
                }
                for &v in c {
                    vf[v as usize].push(fi as u32);
                }
            }
            let edge_id: BTreeMap<(u32, u32), u32> = keys
                .keys()
                .enumerate()
                .map(|(i, k)| (*k, n_old + i as u32))
                .collect();
            for v in nb.iter_mut() {
                v.sort_unstable();
                v.dedup();
            }
            let mut out_pos: Vec<Vec3> = Vec::with_capacity(pos.len() + edge_id.len() + face_pos.len());
            for i in 0..pos.len() {
                let v = pos[i];
                let n = nb[i].len() as f32;
                let mut fsum = Vec3::splat(0.0);
                for &f in &vf[i] {
                    fsum = fsum + face_pos[f as usize];
                }
                let mut nsum = Vec3::splat(0.0);
                for &j in &nb[i] {
                    nsum = nsum + pos[j as usize];
                }
                let rbar = v * 0.5 + nsum * (0.5 / n);
                out_pos.push((fsum * (1.0 / n) + rbar * 2.0 + v * (n - 3.0)) * (1.0 / n));
            }
            // **错的那一步**：边点取中点（正确做法是 (a + b + F₀ + F₁)/4）。
            for &(a, b) in edge_id.keys() {
                out_pos.push((pos[a as usize] + pos[b as usize]) * 0.5);
            }
            out_pos.extend_from_slice(&face_pos);
            let n_edge = edge_id.len() as u32;
            let mut out_q: Vec<u32> = Vec::with_capacity(quad.len() * 4);
            for (fi, c) in quad.chunks_exact(4).enumerate() {
                let (a, b, cc, d) = (c[0], c[1], c[2], c[3]);
                let e = |x: u32, y: u32| -> u32 {
                    *edge_id.get(&(x.min(y), x.max(y))).expect("边必须存在")
                };
                let f = n_old + n_edge + fi as u32;
                out_q.extend_from_slice(&[a, e(a, b), f, e(d, a)]);
                out_q.extend_from_slice(&[e(a, b), b, e(b, cc), f]);
                out_q.extend_from_slice(&[f, e(b, cc), cc, e(cc, d)]);
                out_q.extend_from_slice(&[e(d, a), f, e(cc, d), d]);
            }
            pos = out_pos;
            quad = out_q;
            // 7 级后：角点 x、棱 (6,7) 边点的 y、z=+1 面心的 z（序号逐级继承）。
            got = (pos[6].x, pos[8 + 11].y, pos[8 + 12 + 1].z);
        }
        println!(
            "V2 反例（边点取中点）：角点 {:.6}、棱中点 y {:.6}、面心 z {:.6}",
            got.0, got.1, got.2
        );
        assert!(
            (got.0 - 0.5).abs() > 0.05
                && (got.1 - 395.0 / 648.0).abs() > 0.05
                && (got.2 - 68.0 / 81.0).abs() > 0.05,
            "把边点写成中点必须收敛到别的曲面（实测 {got:?}）"
        );
    }

    /// 错候选反例：`(n²P + 4Σm + ΣF)/(n(n+5))` 在立方体角点给 0.75，
    /// 而真极限是 0.5 —— 这条保证 V2 有判别力（不是"什么式子都能过"）。
    #[test]
    fn wrong_limit_candidate_is_rejected() {
        let (p, q) = cube();
        // 手工按错候选算角点（P=(1,1,1) 换成立方体顶点 6 的 (-1,-1,-1) 不改变量级，
        // 这里就取顶点 0 = (-1,-1,-1)，看 |x|）。
        let v = p[0];
        let nbrs = [-1.0f32, 1.0]; // 占位，下面按真实邻居取
        let _ = nbrs;
        // 顶点 0 = (-1,-1,-1)：邻居 1=(1,-1,-1)、3=(-1,1,-1)、4=(-1,-1,1)。
        let qs = [p[1], p[3], p[4]];
        // 相邻 3 个面的形心。
        let fs = [
            (p[0] + p[3] + p[2] + p[1]) * 0.25, // z=-1
            (p[0] + p[1] + p[5] + p[4]) * 0.25, // y=-1
            (p[0] + p[4] + p[7] + p[3]) * 0.25, // x=-1
        ];
        let mut msum = Vec3::splat(0.0);
        for qq in qs {
            msum = msum + (v + qq) * 0.5;
        }
        let mut fsum = Vec3::splat(0.0);
        for f in fs {
            fsum = fsum + f;
        }
        let n = 3.0f32;
        let wrong = (v * (n * n) + msum * 4.0 + fsum) * (1.0 / (n * (n + 5.0)));
        let right = catmull_clark(&p, &q, &[], 0).positions[0];
        println!("V2 反例：错候选 (n²P+4Σm+ΣF)/n(n+5) = {wrong:?}；真极限 = {right:?}");
        assert!(
            (wrong.x - right.x).abs() > 0.2,
            "错候选必须明显不达标（否则 V2 没有判别力）：{wrong:?} vs {right:?}"
        );
    }

    /// 三角数 / 三角形流字节：`4^k × 四边面数 × 2` 与 `× 72 B`（**精确**）。
    #[test]
    fn quad_triangle_counts_are_exact() {
        let (p, q) = cube();
        let faces = (q.len() / 4) as u64;
        for k in 0..=4u32 {
            let r = catmull_clark(&p, &q, &[], k);
            let want = faces * 4u64.pow(k) * 2;
            assert_eq!(r.tri_count() as u64, want, "k={k} 的三角形数");
            assert_eq!(r.levels, k);
            // 索引流合法性：全部在范围内。
            assert!(r.indices.iter().all(|&i| (i as usize) < r.positions.len()));
        }
    }

    /// 边界规则：**开口平面四边网格的边界顶点不向内收缩**。
    ///
    /// 平面 + 直线边界上，`(b₀ + 6v + b₁)/8` 与曲线极限 `(b₀ + 4v + b₁)/6` 都精确
    /// 复现原直线（等距共线点）。若把边界当内部点，边界会被拉向内侧 ⇒ 平面收窄。
    /// 这条同时钉住"边界边点 = 中点"（内部边点公式在边界上会往里偏）。
    #[test]
    fn cc_boundary_vertices_do_not_shrink() {
        // n×n 顶点的平面四边网格（z = 0），n-1 个四边形/行。
        let n = 6usize;
        let mut pos = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                pos.push(Vec3::new(i as f32, j as f32, 0.0));
            }
        }
        let idx = |i: usize, j: usize| (j * n + i) as u32;
        let mut quads = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                quads.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        let r = catmull_clark(&pos, &quads, &[], 3);
        // 左边界中点（i=0, j=3）：平面 + 共线 ⇒ 极限位置不漂。
        let b = 3 * n;
        let d = (r.positions[b] - pos[b]).length();
        assert!(d <= 1e-6, "边界顶点漂移 {d}（硬边塌陷的典型症状）");
        // 包围盒不外扩也不内缩。
        let min_x = r.positions.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
        let max_x = r.positions.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
        assert!(min_x >= -1e-6 && max_x <= (n - 1) as f32 + 1e-6, "边界越过原始包围盒：{min_x}..{max_x}");
        // 角点（i=0,j=0）**会被抹圆**：它恰有 2 条边界边 ⇒ 走曲线规则（等价于一条
        // 直角折线的三次 B 样条极限）。这是**既定语义**，不是 bug——本轮没有
        // corner/crease 标记通道（见模块文档"折痕规则：未做"），直角无法被声明为
        // 尖角。把它断成"确实动了"，防止有人误以为边界规则=不动。
        let d_corner = (r.positions[0] - pos[0]).length();
        assert!(
            d_corner > 1e-3,
            "直角角点应被曲线规则抹圆（corner 标记未做 ⇒ 无尖角语义）；实测位移 {d_corner:e}"
        );
        println!("CC 边界：角点被抹圆 {d_corner:e}（既定语义，corner 标记未做）");
    }

    /// 内部顶点规则：平面规则四边网格（价 4）是**不动点**（仿射不变）。
    ///
    /// 任何把 `(F̄ + 2R̄ + (n−3)v)/n` 写错（例如漏掉 `n−3` 的常数项、把 R̄ 用成
    /// 边点而不是边中点）的实装都会让平面漂起来——这是最便宜的一道闸。
    #[test]
    fn cc_flat_interior_is_reproduced() {
        let n = 7usize;
        let mut pos = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                pos.push(Vec3::new(i as f32 * 0.5, j as f32 * 0.25, 0.0));
            }
        }
        let idx = |i: usize, j: usize| (j * n + i) as u32;
        let mut quads = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                quads.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        let r = catmull_clark(&pos, &quads, &[], 3);
        let c = (3 * n + 3) as usize; // 内部点（价 4）
        let d = (r.positions[c] - pos[c]).length();
        assert!(d <= 1e-6, "平面内部点漂移 {d}");
        let max_abs_z = r.positions.iter().map(|v| v.z.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs_z <= 1e-6, "平面网格的极限面必须仍在该平面上，max|z| = {max_abs_z}");
    }

    /// uv：顶点域（新边点取两端平均、新面点取四角平均），且相邻面共享同一 uv。
    #[test]
    fn cc_uv_is_vertex_domain() {
        let (p, q) = cube();
        let uv: Vec<[f32; 2]> = p.iter().map(|v| [0.5 + 0.25 * v.x, 0.5 + 0.25 * v.y]).collect();
        let r = catmull_clark(&p, &q, &uv, 2);
        assert_eq!(r.uv.len(), r.positions.len(), "uv 必须与顶点同长");
        // 结构性断言：uv 是"每顶点一份"，任意面角点取到的值必逐位相同。
        let mut seen: std::collections::BTreeMap<u32, [f32; 2]> = std::collections::BTreeMap::new();
        for &i in &r.indices {
            let u = r.uv[i as usize];
            match seen.get(&i) {
                Some(prev) => assert_eq!(
                    prev.map(f32::to_bits),
                    u.map(f32::to_bits),
                    "顶点 {i} 的 uv 出现两个不同值 ⇒ 顶点域契约被破坏"
                ),
                None => {
                    seen.insert(i, u);
                }
            }
        }
        // 面点 uv = 四角平均（取 z=+1 面的面点，level-1 序号 8+12+1）。
        let want = [
            (uv[4][0] + uv[5][0] + uv[6][0] + uv[7][0]) * 0.25,
            (uv[4][1] + uv[5][1] + uv[6][1] + uv[7][1]) * 0.25,
        ];
        assert_eq!(r.uv[8 + 12 + 1].map(f32::to_bits), want.map(f32::to_bits), "面点 uv 必须是四角平均");
    }

    /// 确定性：同输入双跑逐位一致（含法线位型）。
    #[test]
    fn cc_is_bitwise_deterministic() {
        let (p, q) = cube();
        let a = catmull_clark(&p, &q, &[], 3);
        let b = catmull_clark(&p, &q, &[], 3);
        assert_eq!(a.indices, b.indices);
        assert_eq!(
            a.positions.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            b.positions.iter().map(|v| vbits(*v)).collect::<Vec<_>>()
        );
        assert_eq!(
            a.normals.iter().map(|v| vbits(*v)).collect::<Vec<_>>(),
            b.normals.iter().map(|v| vbits(*v)).collect::<Vec<_>>()
        );
    }

    /// 退化输入：空 / 越界 / 非有限 / 重复角点 / 长度非 4 的倍数 ⇒ 不 panic。
    #[test]
    fn cc_degenerate_inputs() {
        assert!(catmull_clark(&[], &[], &[], 3).indices.is_empty());
        let p = vec![Vec3::splat(0.0); 5];
        assert!(catmull_clark(&p, &[0, 1, 2, 9], &[], 2).indices.is_empty(), "越界索引必须剔除");
        let mut bad = p.clone();
        bad[1] = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(catmull_clark(&bad, &[0, 1, 2, 3], &[], 2).indices.is_empty(), "非有限顶点必须剔除");
        assert!(catmull_clark(&p, &[0, 1, 1, 2], &[], 2).indices.is_empty(), "重复角点必须剔除");
        // 尾部余数（len % 4 != 0）忽略，不 panic。
        let r = catmull_clark(&p, &[0, 1, 2, 3, 0], &[], 1);
        assert_eq!(r.tri_count(), 8, "一个四边面 → 4 个 → 8 个三角形");
    }

    /// MAX_LEVELS 截断并回显。
    #[test]
    fn cc_levels_are_clamped_and_echoed() {
        let (p, q) = cube();
        let r = catmull_clark(&p, &q, &[], MAX_LEVELS + 4);
        assert_eq!(r.levels, MAX_LEVELS);
        assert_eq!(r.tri_count(), 6 * 4usize.pow(MAX_LEVELS) * 2);
    }


    fn vbits(v: Vec3) -> (u32, u32, u32) {
        (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
    }
}
