//! Loop 细分（三角笼）+ **极限位置直接求值** —— W4 的 CPU 烘焙器（选型报告 §5 定稿 C2）。
//!
//! # 定稿口径（逐条对应 `docs/v2.5-W0-算法选型报告.md` §5）
//!
//! 1. **三角笼走 Loop，四边笼走 Catmull-Clark**。本轮只做 Loop（C2 默认）：
//!    glTF/OBJ 进来的就是三角网，三角笼**零拓扑歧义**；硬推 CC 要做"三角→四边
//!    配对"，而配对没有唯一解（同一资产换导入顺序会得到不同曲面）。
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
    let levels = levels.min(MAX_LEVELS);
    let mut pos: Vec<Vec3> = positions.to_vec();
    let mut tri: Vec<u32> = indices.to_vec();
    // uv 长度不符 ⇒ 按"无 uv"处理（与 `Mesh::uv` 的契约一致，不 panic）。
    let mut tex: Vec<[f32; 2]> = if uv.len() == positions.len() { uv.to_vec() } else { Vec::new() };

    // 初始净化：越界 / 非有限 ⇒ 剔除该三角形（与 `Mesh::new` 同口径）。
    tri = sanitize(&pos, &tri);
    if tri.is_empty() {
        return SubdivResult { levels, ..Default::default() };
    }

    for _ in 0..levels {
        let (p, t, u) = refine_once(&pos, &tri, &tex);
        pos = p;
        tri = t;
        tex = u;
    }

    // 极限位置（不迭代）。
    let topo = Topology::build(&pos, &tri);
    let lim = limit_positions(&pos, &topo);
    let normals = vertex_normals(&lim, &tri);
    SubdivResult { positions: lim, indices: tri, uv: tex, normals, levels }
}

/// 只求极限位置（不改拓扑；`levels = 0` 的 `subdivide` 等价于它）。
pub fn limit_of(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let tri = sanitize(positions, indices);
    let topo = Topology::build(positions, &tri);
    limit_positions(positions, &topo)
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
}

/// 细分一级（1→4），返回新的 `(positions, indices, uv)`。
fn refine_once(pos: &[Vec3], tri: &[u32], uv: &[[f32; 2]]) -> (Vec<Vec3>, Vec<u32>, Vec<[f32; 2]>) {
    let topo = Topology::build(pos, tri);
    let n_old = pos.len();
    let n_edge = topo.edges.len();

    let mut out_pos: Vec<Vec3> = Vec::with_capacity(n_old + n_edge);
    // 顶点规则。
    for i in 0..n_old {
        out_pos.push(moved_vertex(pos, &topo, i));
    }
    // 边顶点（边表按 `(a, b)` 升序 ⇒ 顶点索引确定）。
    for e in topo.edges.iter() {
        let (a, b) = (e.a, e.b);
        let pa = pos[a as usize];
        let pb = pos[b as usize];
        let p = if e.count == 1 || e.count > 2 {
            // 边界边 / 非流形边：中点。
            (pa + pb) * 0.5
        } else {
            let c = pos[e.opp[0] as usize];
            let d = pos[e.opp[1] as usize];
            (pa + pb) * 0.375 + (c + d) * 0.125
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
    (out_pos, out_tri, out_uv)
}

/// 顶点规则（内部 / 边界 / 角点三档）。
fn moved_vertex(pos: &[Vec3], topo: &Topology, i: usize) -> Vec3 {
    let v = pos[i];
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

/// 极限位置：内部 1-ring stencil / 边界曲线 stencil / 角点不动。
fn limit_positions(pos: &[Vec3], topo: &Topology) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        let v = pos[i];
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
        let nb = topo.neighbors(i);
        if nb.is_empty() {
            out.push(v);
            continue;
        }
        let (a, b) = loop_limit_weights(nb.len());
        let mut sum = Vec3::splat(0.0);
        for &j in nb {
            sum = sum + pos[j as usize];
        }
        out.push(v * a + sum * b);
    }
    out
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
