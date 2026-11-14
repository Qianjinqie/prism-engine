//! 2D 矢量路径光栅化：**GPU 优先、CPU 兜底**（与 [`crate::background`] 同形态）。
//!
//! # 为什么这份实现落在 `prism-gpu`（依赖方向不成就环）
//!
//! * `prism-gpu` 的 `[dependencies]` 只有 `prism-math` / `prism-geometry`；
//! * `prism-raster` 的 `[dependencies]` 只有 `prism-io` / `prism-math` / `prism-core`。
//!
//! 两者**互不依赖**：本模块自带一份 POD 路径输入（[`RasterPoint`]）、一份 f64
//! 预处理（[`prepare`]）、一份 f32 参考（[`rasterize_cpu`]）与 GPU kernel
//! （[`crate::vulkan::spirv_path`] / `shaders/path_raster.wgsl`）。真实消费方
//! `prism-postfx` 同时依赖 `prism-gpu` 与 `prism-raster`，是**单向聚合层**：
//! 它把 `prism_raster::geom::Point` 转成 [`RasterPoint`] 后调用本入口，并在测试
//! 里拿 `prism_raster::scanline::rasterize`（f64 几何权威）当精度标尺。
//! 依赖图是 DAG：`postfx → {gpu, raster}`，`gpu ↛ raster`、`raster ↛ gpu`，
//! 因此既无环，也没有把整条 `prism-raster` 塞进 wasm 产物。
//!
//! # 算子（三端同式的唯一权威）
//!
//! 覆盖率是区域 `{winding ≠ 0}`（或 even-odd 奇区）与像素矩形 `P` 的**精确
//! 交面积**。CPU 侧 [`prepare`] 用 f64 把路径切成「带符号跳变权重的边片段」；
//! 对每个 y 子区间 `[p,q]`，把活跃跨越按 x 升序排成 `X_0…X_{m-1}`，令
//! `inside_k` 为 `X_k` 与 `X_{k+1}` 之间区域的填充布尔（`inside_{-1}=
//! inside_{m-1}=0`），则沿 x 的指示函数满足**恒等式**
//!
//! ```text
//! 1_R(x,y) = Σ_k delta_k · 1[x < X_k(y)],   delta_k = inside_{k-1} − inside_k
//! ```
//!
//! （ telescopes：前缀和 `Σ_{k≤j} delta_k = −inside_j`）。对像素列
//! `[px,px+1)` 积分 `∫1[x<X]dx = clamp(X−px, 0, 1)`，而
//! `clamp(u,0,1) = u⁺ − (u−1)⁺`，`∫(c+k·t)⁺dt` 的解析原函数就是
//! `pos_area`（三角/梯形，纯 + − × ÷ 与比较）。于是
//!
//! ```text
//! cov(P) = Σ_片段 delta · [pos_area(u0,u1,h,fac) − pos_area(u0−1,u1−1,h,fac)]
//! ```
//!
//! 与 `prism-raster` 扫描线的「区域 `[xl,xr)` 边界列解析积分」是**同一个面积
//! 积分的两种写法**（Green/射线法恒等式），不是近似、不是超采样。
//!
//! # f32/f64 口径（判据 P1 / P2 的设计依据，动手前写死）
//!
//! SPIR-V 1.0 与 WGSL 都只有 f32（WGSL 无 f64）。若直接把全局 x 坐标以 f32
//! 传入，1080p/4K 坐标的 ulp 是 1e-4 量级，覆盖率会偏离 f64 标尺到 1e-4，
//! **逐字节对 f64 标尺不可能**。为不靠"放宽阈值"过关，本实现改数据编码：
//! 片段的 x 拆成 **整数基准 + 小量小数**（`X = col_base + f`，
//! `col_base = floor(min(xp,xq))` 是精确整数，走 `base = col_base + 2²⁰` 的
//! f32 精确整数；`fp/fq` 是 `[0, 1+|Δx|)` 的小量）。kernel 里
//! `u = (base − 2²⁰ − px) + f`：括号内是**整数差**（≤ 2²¹，f32 精确），
//! 只有 `u ∈ (−1, 2)` 的敏感区才影响 `pos_area`，此时 `u` 量级 ≤ 2，
//! f32 ulp ≤ 2.4e-7。每像素最多 4 条穿越边贡献 ⇒ 覆盖率误差
//! `≤ 4 × 2.4e-7 < 1e-6 < 2e-6`。这就是 P2 阈值的**推导来源**，不是实测调出来的。
//!
//! * **P1（主判据，逐字节）**：GPU 输出与 [`rasterize_cpu`] 在同一份 f32 片段
//!   缓冲上必须**逐字节一致**（同式 + 同运算顺序 + `NoContraction`）。
//! * **P2（数学等价）**：与 `prism-raster` f64 几何权威的每像素最大绝对差
//!   `≤ 2e-6`，u8 量化视图 `differing_pixels = 0` / `max_channel_delta = 0`，
//!   面积总量相对误差 `≤ 1e-6`。
//!
//! 两条判据与负控写在 `tests/vulkan_path_raster_gpu.rs` 与
//! `crates/prism-postfx/tests/qa_path_raster_gpu_parity.rs` 的文件头，**不许事后
//! 改尺子**。
//!
//! # 确定性 / 零 libm
//!
//! 只用 `+ − × ÷` 与比较；排序键 `f64::total_cmp` 全序；无隐藏状态、无随机源。
//! f64 只出现在 CPU 预处理（几何），GPU kernel 与 f32 参考**全程 f32**，两端
//! 求值树逐句对应。同输入两跑逐字节一致。

use std::cell::RefCell;

/// 片段的 f32 字段数：`(h, 0, base, fp, fq, delta)`。
///
/// 字段 0 是 y 子区间高度 `h = q−p`（由 f64 预处理一次舍入写入，kernel 不需要
/// y 的绝对坐标，只需区间高度）；字段 1 保留占位恒 0（历史布局对齐，使片段
/// 仍是 6 个 f32）。`base = col_base + COORD_BIAS`，`fp/fq` 是 x 的小量小数
/// （切分保证 `∈ [0,3)`），`delta` 是带符号跳变权重。
pub const SEGMENT_FLOATS: usize = 6;
/// x 整数基准的偏置（2²⁰）；`base = col_base + COORD_BIAS` 在 f32 里是精确整数。
pub const COORD_BIAS: f32 = 1_048_576.0;
/// GPU 侧片段缓冲上限（字节）：64 MiB。超过时 `rasterize_gpu` 如实返回 `None`
/// 让 [`rasterize`] 回退 CPU（资源守卫，与 `MAX_FRAMEBUFFER_BYTES` 同类，不是能力缺口）。
pub const MAX_GPU_SEGMENT_BYTES: usize = 1 << 26;

/// POD 路径顶点（设备像素坐标，y 向下）。刻意不复用 `prism_raster::geom::Point`：
/// 那会让 `prism-gpu` 反向依赖 `prism-raster`（见文件头依赖论证）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterPoint {
    /// x（像素）。
    pub x: f32,
    /// y（像素）。
    pub y: f32,
}

impl RasterPoint {
    /// 构造顶点。
    pub const fn new(x: f32, y: f32) -> Self {
        RasterPoint { x, y }
    }
}

/// 填充规则（与 `prism-raster` 的两个规则一一对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFillRule {
    /// 非零 winding（TrueType 字形标准规则）。
    Winding,
    /// 奇偶（交替）。
    EvenOdd,
}

/// 覆盖率缓冲（`prism_raster::Coverage` 的等价物：行主序 f32，`∈[0,1]`）。
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageMask {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// 覆盖率（行主序）。
    pub data: Vec<f32>,
}

impl CoverageMask {
    /// 构造全零缓冲（尺寸守卫：超限返回空 data）。
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize)
            .checked_mul(height as usize)
            .filter(|&n| n <= (crate::MAX_FRAMEBUFFER_BYTES / 4));
        CoverageMask { width, height, data: vec![0.0; n.unwrap_or(0)] }
    }

    /// 读一个像素（越界返回 0）。
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data[(y as usize) * self.width as usize + x as usize]
    }

    /// u8 量化视图（覆盖率 × 255 四舍五入，钳到 [0,255]）。P2 的量化判据用。
    pub fn to_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| {
                let q = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                q.min(255) as u8
            })
            .collect()
    }
}

/// [`prepare`] 的产物：GPU kernel 与 [`rasterize_cpu`] 消费的**同一份**缓冲。
#[derive(Debug, Clone, PartialEq)]
pub struct RasterInput {
    /// 画布宽。
    pub width: u32,
    /// 画布高。
    pub height: u32,
    /// 扁平片段数组，每 [`SEGMENT_FLOATS`] 个 f32 一个片段。
    pub segments: Vec<f32>,
    /// 每行 `[start, count]`（以片段为单位），长度 `2*height`。
    pub rows: Vec<u32>,
}

impl RasterInput {
    /// 空输入（无片段）：仍带一个全零哑片段，保证 binding 元素数 ≥ 1。
    fn empty(width: u32, height: u32) -> Self {
        RasterInput {
            width,
            height,
            segments: vec![0.0; SEGMENT_FLOATS],
            rows: vec![0; (height as usize).saturating_mul(2).max(2)],
        }
    }

    /// 参数块（4 × u32 little-endian）：`[w, h, segments_f32_len, rows_u32_len]`。
    pub fn params_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.width.to_le_bytes());
        out[4..8].copy_from_slice(&self.height.to_le_bytes());
        out[8..12].copy_from_slice(&(self.segments.len() as u32).to_le_bytes());
        out[12..16].copy_from_slice(&(self.rows.len() as u32).to_le_bytes());
        out
    }

    /// 片段缓冲（f32 位型 → little-endian u32，GPU 侧 `OpBitcast` 还原）。
    pub fn segments_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.segments.len() * 4);
        for &v in &self.segments {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }

    /// 行范围缓冲（u32 little-endian）。
    pub fn rows_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rows.len() * 4);
        for &v in &self.rows {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// 片段总数。
    pub fn segment_count(&self) -> usize {
        self.segments.len() / SEGMENT_FLOATS
    }
}

// ---------------------------------------------------------------------------
// f64 预处理：路径 → 带符号跳变权重的边片段
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Edge {
    xlo: f64,
    lo: f64,
    k: f64,
    hi: f64,
    xmin: f64,
    xmax: f64,
    sign: i32,
}

impl Edge {
    #[inline]
    fn x_at(&self, y: f64) -> f64 {
        self.xlo + (y - self.lo) * self.k
    }
}

#[inline]
fn lt(a: f64, b: f64) -> bool {
    a.partial_cmp(&b) == Some(core::cmp::Ordering::Less)
}

#[inline]
fn less_edge(a: &Edge, b: &Edge, y: f64, ia: u32, ib: u32) -> bool {
    match a.x_at(y).total_cmp(&b.x_at(y)) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => match a.k.total_cmp(&b.k) {
            core::cmp::Ordering::Less => true,
            core::cmp::Ordering::Greater => false,
            core::cmp::Ordering::Equal => ia < ib,
        },
    }
}

/// 近乎有序数组的插入排序（事件之间次序只可能局部变化）。
fn sort_active(order: &mut [usize], edges: &[Edge], y: f64) {
    let mut i = 1usize;
    while i < order.len() {
        let v = order[i];
        let mut j = i;
        while j > 0 && less_edge(&edges[v], &edges[order[j - 1]], y, v as u32, order[j - 1] as u32) {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = v;
        i += 1;
    }
}

/// 追加片段：x 拆成「精确整数基准 + f32 小量小数」（见文件头精度论证）。
///
/// **按 x 跨度切分**：同一条直线段若在一个 y 子区间内 `|Δx| > 2`，就等分成
/// 若干子片段，使每个子片段的 `fp/fq ∈ [0,3)`。这纯粹是把线性函数分段积分
/// （数学恒等），却把 f32 里 `a0 − a1` 的**大数消去**限制在 `u ∈ [−3,5]`
/// 量级（ulp ≤ 4.8e-7），是 P2 的 `2e-6` 判据可达的关键。
///
/// 字段 0 直接存 `h = q−p`（f64 差只舍入一次，误差 ≤ 6e-8），不再存 `p,q`
/// 的绝对值——kernel 的面积计算根本不需要 y 的绝对坐标，只需区间高度。
/// 字段 1 保留（恒 0），使片段布局仍是 6 个 f32。
fn push_segment(
    p: f64,
    q: f64,
    e: &Edge,
    delta: i32,
    segments: &mut Vec<f32>,
    row_counts: &mut [u32],
    height: u32,
) {
    let row = p.floor() as i64;
    if row < 0 || row >= height as i64 || !(q > p) {
        return;
    }
    let xp = e.x_at(p);
    let xq = e.x_at(q);
    if !xp.is_finite() || !xq.is_finite() {
        return;
    }
    let span = (xq - xp).abs();
    let parts = (span * 0.5).ceil().max(1.0) as usize;
    if parts > (1 << 20) {
        return; // 病态坐标（单像素行内横跨 > 2²¹ px）：不生成片段，交回安全空区
    }
    let dp = (q - p) / parts as f64;
    let dpf = delta as f32;
    for j in 0..parts {
        let p0 = p + dp * j as f64;
        let q0 = if j + 1 == parts { q } else { p + dp * (j + 1) as f64 };
        let x0 = e.x_at(p0);
        let x1 = e.x_at(q0);
        let col = x0.min(x1).floor() as i64;
        // 病态坐标钳到偏置可精确表示的范围（f32 整数精度 2²⁴）。
        let col = col.clamp(-1_048_576, 1_048_576);
        let cf = col as f64;
        let base = (col as f64 + f64::from(COORD_BIAS)) as f32;
        segments.push((q0 - p0) as f32);
        segments.push(0.0);
        segments.push(base);
        segments.push((x0 - cf) as f32);
        segments.push((x1 - cf) as f32);
        segments.push(dpf);
        row_counts[row as usize] += 1;
    }
}

/// 把路径预处理成 GPU/f32 参考共用的片段缓冲。
///
/// 几何量全程 f64；只有片段字段写回时舍入一次到 f32（见文件头精度论证）。
/// 事件 = 段端点 ∪ 段间交点 ∪ **每个整数行边界**（保证片段不跨像素行）。
pub fn prepare(
    contours: &[Vec<RasterPoint>],
    rule: PathFillRule,
    width: u32,
    height: u32,
) -> RasterInput {
    if width == 0 || height == 0 || contours.is_empty() {
        return RasterInput::empty(width, height);
    }
    if width > crate::MAX_FRAMEBUFFER_DIM || height > crate::MAX_FRAMEBUFFER_DIM {
        return RasterInput::empty(width, height);
    }
    let hf = height as f64;

    // ---- 1. 边规范化（裁到画布 y 范围，水平边丢弃） ----
    let mut edges: Vec<Edge> = Vec::new();
    for c in contours {
        let n = c.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = c[i];
            let b = c[(i + 1) % n];
            if a.y == b.y || !a.x.is_finite() || !a.y.is_finite()
                || !b.x.is_finite() || !b.y.is_finite()
            {
                continue;
            }
            let (ax, ay) = (a.x as f64, a.y as f64);
            let (bx, by) = (b.x as f64, b.y as f64);
            let dy = by - ay;
            let dx = bx - ax;
            let (mut lo, mut hi) = if ay < by { (ay, by) } else { (by, ay) };
            if lo < 0.0 {
                lo = 0.0;
            }
            if hi > hf {
                hi = hf;
            }
            if !lt(lo, hi) {
                continue;
            }
            let k = dx / dy;
            let xlo = ax + ((lo - ay) / dy) * dx;
            let xhi = ax + ((hi - ay) / dy) * dx;
            let (xmin, xmax) = if xlo < xhi { (xlo, xhi) } else { (xhi, xlo) };
            edges.push(Edge {
                xlo,
                lo,
                k,
                hi,
                xmin,
                xmax,
                sign: if dy > 0.0 { 1 } else { -1 },
            });
        }
    }
    if edges.len() < 2 {
        return RasterInput::empty(width, height);
    }
    let ne = edges.len();

    // ---- 2. 事件：端点 + 交点（x 区间剪枝）+ 整数行边界 ----
    let mut events: Vec<f64> = Vec::with_capacity(ne * 2 + height as usize + 64);
    for e in &edges {
        events.push(e.lo);
        events.push(e.hi);
    }
    for r in 0..=height {
        events.push(r as f64);
    }
    {
        let mut by_x: Vec<usize> = (0..ne).collect();
        by_x.sort_unstable_by(|&i, &j| {
            edges[i].xmin.total_cmp(&edges[j].xmin).then(i.cmp(&j))
        });
        for ii in 0..ne {
            let ea = edges[by_x[ii]];
            for jj in (ii + 1)..ne {
                let eb = edges[by_x[jj]];
                if eb.xmin > ea.xmax {
                    break;
                }
                let lo = if ea.lo > eb.lo { ea.lo } else { eb.lo };
                let hi = if ea.hi < eb.hi { ea.hi } else { eb.hi };
                if !lt(lo, hi) {
                    continue;
                }
                let den = ea.k - eb.k;
                if den == 0.0 {
                    continue;
                }
                let ystar = lo + (eb.x_at(lo) - ea.x_at(lo)) / den;
                if ystar >= lo && ystar <= hi && ystar.is_finite() {
                    events.push(ystar);
                }
            }
        }
    }
    events.sort_unstable_by(f64::total_cmp);
    events.dedup();

    // ---- 3. y 扫描：活跃跨越增量维护 + 每个事件区间生成片段 ----
    let mut by_lo: Vec<usize> = (0..ne).collect();
    by_lo.sort_unstable_by(|&i, &j| edges[i].lo.total_cmp(&edges[j].lo).then(i.cmp(&j)));
    let mut order: Vec<usize> = Vec::with_capacity(ne);
    let mut cur_lo = 0usize;
    let mut segments: Vec<f32> = Vec::new();
    // 每行片段数（事件按 y 升序 ⇒ 行号单调不减，最后统一做前缀和）。
    let mut row_counts: Vec<u32> = vec![0; height as usize];

    let mut open: f64 = 0.0;
    let mut ev = 0usize;
    while ev < events.len() {
        let ye = events[ev];
        let p = if open > 0.0 { open } else { 0.0 };
        if lt(p, ye) && ye > 0.0 && p < hf {
            let q = if ye < hf { ye } else { hf };
            // 活跃集：删除已结束、插入新入场。
            order.retain(|&id| edges[id].hi > p);
            while cur_lo < ne && edges[by_lo[cur_lo]].lo <= p {
                order.push(by_lo[cur_lo]);
                cur_lo += 1;
            }
            let mid = (p + q) * 0.5;
            sort_active(&mut order, &edges, mid);
            let m = order.len();
            if m >= 2 {
                let mut winding = 0i32;
                let mut inside: Vec<i32> = Vec::with_capacity(m - 1);
                for k in 0..(m - 1) {
                    winding += edges[order[k]].sign;
                    let ins = if rule == PathFillRule::EvenOdd {
                        (((m - 1 - k) & 1) == 1) as i32
                    } else {
                        (winding != 0) as i32
                    };
                    inside.push(ins);
                }
                for k in 0..m {
                    let left = if k == 0 { 0 } else { inside[k - 1] };
                    let right = if k == m - 1 { 0 } else { inside[k] };
                    let delta = left - right;
                    if delta != 0 {
                        push_segment(p, q, &edges[order[k]], delta, &mut segments, &mut row_counts, height);
                    }
                }
            }
        }
        open = ye;
        if open >= hf {
            break;
        }
        ev += 1;
    }

    // 行范围表 = 每行片段数的前缀和。
    let mut rows: Vec<u32> = vec![0; (height as usize) * 2];
    let mut start = 0u32;
    for r in 0..height as usize {
        rows[r * 2] = start;
        rows[r * 2 + 1] = row_counts[r];
        start = start.saturating_add(row_counts[r]);
    }
    if segments.is_empty() {
        // 至少一个哑片段，保证 storage binding 元素数 ≥ 1。
        segments.extend_from_slice(&[0.0; SEGMENT_FLOATS]);
    }
    RasterInput { width, height, segments, rows }
}

// ---------------------------------------------------------------------------
// f32 参考（与 SPIR-V / WGSL kernel 逐句同式）
// ---------------------------------------------------------------------------

/// `∫ (u(y))⁺ dy` 的解析原函数：`u` 由 `u0` 线性变到 `u1`，`h=q−p`。
/// 两端非负 → 梯形；否则正部是三角形 `fac·max(u0,u1,0)²`。
#[inline]
fn pos_area(u0: f32, u1: f32, h: f32, fac: f32) -> f32 {
    let trap = (u0 + u1) * 0.5 * h;
    let p = if u0 > u1 { u0 } else { u1 };
    let pp = if p > 0.0 { p } else { 0.0 };
    let tri = fac * pp * pp;
    if (if u0 < u1 { u0 } else { u1 }) >= 0.0 {
        trap
    } else {
        tri
    }
}

/// 三角形模式因子 `0.5·h/|u0−u1|`；`Δ=0`（竖直边）取 0。
#[inline]
fn tri_factor(u0: f32, u1: f32, h: f32) -> f32 {
    let den = u0 - u1;
    if den == 0.0 {
        0.0
    } else {
        0.5 * h / (if den < 0.0 { -den } else { den })
    }
}

/// 单片段对像素 `(px, py)` 的覆盖率贡献（f32，与 kernel 逐句同式）。
///
/// 饱和短路：`X(y) − px` 全程 > 1 ⇒ 整列满覆盖 `h`；全程 < 0 ⇒ 0。两条都
/// 是精确结论（切分保证 `fp/fq ∈ [0,3)`），且把 f32 大数消去挡在一般路径
/// 之外。一般路径里 `u0/u1 ∈ [−3,5]`，`a0 − a1` 的舍入误差 ≤ 5e-7。
#[inline]
fn segment_area(seg: &[f32], px: f32) -> f32 {
    let h = seg[0];
    let base = seg[2];
    let fp = seg[3];
    let fq = seg[4];
    let delta = seg[5];
    let bias_px = (base - COORD_BIAS) - px;
    let lo = if fp < fq { fp } else { fq };
    let hi = if fp < fq { fq } else { fp };
    let u0 = bias_px + fp;
    let u1 = bias_px + fq;
    let fac = tri_factor(u0, u1, h);
    let a0 = pos_area(u0, u1, h, fac);
    let a1 = pos_area(u0 - 1.0, u1 - 1.0, h, fac);
    let area = a0 - a1;
    let normal = delta * area;
    let saturated = delta * h;
    if bias_px + lo > 1.0 {
        saturated
    } else if bias_px + hi < 0.0 {
        0.0
    } else {
        normal
    }
}

/// CPU 参考实现：**消费与 GPU 完全相同的 f32 片段缓冲**，运算顺序逐句对应
/// `spirv_path` 与 `path_raster.wgsl`。这是判据 P1 的权威定义。
pub fn rasterize_cpu_prepared(input: &RasterInput) -> CoverageMask {
    let w = input.width;
    let h = input.height;
    let mut cov = CoverageMask::new(w, h);
    if cov.data.len() != (w as usize) * (h as usize) {
        return cov;
    }
    let segs = &input.segments;
    for y in 0..h {
        let r = y as usize * 2;
        if r + 1 >= input.rows.len() {
            break;
        }
        let start = input.rows[r] as usize;
        let count = input.rows[r + 1] as usize;
        let base = y as usize * w as usize;
        for x in 0..w {
            let px = x as f32;
            let mut acc = 0.0f32;
            for s in 0..count {
                let i = (start + s) * SEGMENT_FLOATS;
                if i + SEGMENT_FLOATS > segs.len() {
                    break;
                }
                acc = acc + segment_area(&segs[i..i + SEGMENT_FLOATS], px);
            }
            cov.data[base + x as usize] = acc;
        }
    }
    cov
}

/// CPU 参考（路径 → 覆盖率，自包含；GPU 不可用时的兜底）。
pub fn rasterize_cpu(
    contours: &[Vec<RasterPoint>],
    rule: PathFillRule,
    width: u32,
    height: u32,
) -> CoverageMask {
    rasterize_cpu_prepared(&prepare(contours, rule, width, height))
}

// ---------------------------------------------------------------------------
// GPU 路径（thread_local 设备槽 + 双门自检，与 background.rs 同形态）
// ---------------------------------------------------------------------------

/// GPU 出帧（真机 dispatch → fence → readback），失败返回 `None`。
///
/// 降级分两级：初始化级（无设备/自检失败永久标记 Unavailable）与单帧级
/// （某帧派发失败只让这一帧回退）。任何条件下不 panic。
pub fn rasterize_gpu(
    contours: &[Vec<RasterPoint>],
    rule: PathFillRule,
    width: u32,
    height: u32,
) -> Option<CoverageMask> {
    let input = prepare(contours, rule, width, height);
    rasterize_gpu_prepared(&input)
}

/// 已预处理输入的 GPU 出帧（避免 [`rasterize`] 重复预处理）。
pub fn rasterize_gpu_prepared(input: &RasterInput) -> Option<CoverageMask> {
    if input.width == 0 || input.height == 0 {
        return None;
    }
    if input.segments.len() * 4 > MAX_GPU_SEGMENT_BYTES {
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        GPU_SLOT.with(|slot| {
            let Ok(mut slot) = slot.try_borrow_mut() else {
                return None; // 重入：保守回退
            };
            if matches!(*slot, GpuSlot::Uninit) {
                let backend = crate::vulkan::VulkanBackend::new();
                *slot = if backend.probed_device()
                    && backend.compute_available()
                    && backend.path_frame_available()
                {
                    GpuSlot::Ready(Box::new(backend))
                } else {
                    GpuSlot::Unavailable
                };
            }
            match &mut *slot {
                GpuSlot::Ready(backend) => {
                    let report = backend.rasterize_path_frame(input);
                    if report.is_ok() {
                        Some(report.coverage)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = input;
        None // wasm32 无原生 Vulkan；WebGPU 侧消费 `shaders/path_raster.wgsl`
    }
}

enum GpuSlot {
    Uninit,
    Unavailable,
    #[cfg(not(target_arch = "wasm32"))]
    Ready(Box<crate::vulkan::VulkanBackend>),
}

thread_local! {
    static GPU_SLOT: RefCell<GpuSlot> = const { RefCell::new(GpuSlot::Uninit) };
}

/// 引擎入口：**GPU 优先，CPU 兜底**。与 `render_background` 同形态：能力提升
/// 对上层透明，结果不变（P1 逐字节 + P2 数学等价）。
pub fn rasterize(
    contours: &[Vec<RasterPoint>],
    rule: PathFillRule,
    width: u32,
    height: u32,
) -> CoverageMask {
    let input = prepare(contours, rule, width, height);
    if let Some(cov) = rasterize_gpu_prepared(&input) {
        return cov;
    }
    rasterize_cpu_prepared(&input)
}

// ---------------------------------------------------------------------------
// 测试（无设备也执行）：算子恒等式 / 确定性 / 片段编码
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<RasterPoint> {
        vec![
            RasterPoint::new(x0, y0),
            RasterPoint::new(x1, y0),
            RasterPoint::new(x1, y1),
            RasterPoint::new(x0, y1),
        ]
    }

    #[test]
    fn unit_square_exact_coverage_and_area() {
        let c = rasterize_cpu(&[square(1.5, 1.5, 3.0, 3.0)], PathFillRule::Winding, 5, 5);
        assert_eq!(c.get(2, 2), 1.0, "内部全覆盖");
        assert_eq!(c.get(1, 2), 0.5, "左边界列半覆盖");
        assert_eq!(c.get(1, 1), 0.25, "角落 1/4");
        assert_eq!(c.get(0, 0), 0.0);
        let area: f32 = c.data.iter().sum();
        assert!((area - 2.25).abs() < 1e-5, "面积 {area}");
    }

    #[test]
    fn triangle_half_plane_and_area_conservation() {
        let tri = vec![
            RasterPoint::new(0.0, 0.0),
            RasterPoint::new(2.0, 0.0),
            RasterPoint::new(0.0, 2.0),
        ];
        let c = rasterize_cpu(&[tri], PathFillRule::Winding, 4, 4);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-6, "{}", c.get(0, 0));
        assert!((c.get(1, 0) - 0.5).abs() < 1e-6, "{}", c.get(1, 0));
        assert!((c.get(0, 1) - 0.5).abs() < 1e-6, "{}", c.get(0, 1));
        let area: f32 = c.data.iter().sum();
        assert!((area - 2.0).abs() < 1e-5, "面积守恒 {area}");
    }

    #[test]
    fn winding_vs_evenodd_overlap_and_hole() {
        let contours = [square(0.0, 0.0, 2.0, 2.0), square(1.0, 1.0, 3.0, 3.0)];
        let w = rasterize_cpu(&contours, PathFillRule::Winding, 5, 5);
        let e = rasterize_cpu(&contours, PathFillRule::EvenOdd, 5, 5);
        let sw: f32 = w.data.iter().sum();
        let se: f32 = e.data.iter().sum();
        assert!((sw - 7.0).abs() < 1e-4, "winding 并集 7，实际 {sw}");
        assert!((se - 6.0).abs() < 1e-4, "even-odd 洞 6，实际 {se}");
        assert_eq!(e.get(1, 1), 0.0, "even-odd 重叠区为洞");
    }

    #[test]
    fn counter_wound_ring_excludes_hole() {
        let mut outer = square(0.0, 0.0, 4.0, 4.0);
        outer.reverse();
        let inner = square(1.0, 1.0, 3.0, 3.0);
        let c = rasterize_cpu(&[outer, inner], PathFillRule::Winding, 6, 6);
        let area: f32 = c.data.iter().sum();
        assert!((area - 12.0).abs() < 1e-4, "环形 12，实际 {area}");
        assert_eq!(c.get(2, 2), 0.0);
    }

    #[test]
    fn cpu_reference_is_bitwise_deterministic() {
        let tri = vec![
            RasterPoint::new(0.3, 0.7),
            RasterPoint::new(4.9, 1.3),
            RasterPoint::new(2.2, 5.8),
        ];
        let a = rasterize_cpu(core::slice::from_ref(&tri), PathFillRule::Winding, 8, 8);
        let b = rasterize_cpu(&[tri], PathFillRule::Winding, 8, 8);
        assert_eq!(a.data.len(), b.data.len());
        for (x, y) in a.data.iter().zip(b.data.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "两跑逐位一致");
        }
    }

    #[test]
    fn prepared_input_is_deterministic_and_row_counts_sum() {
        let star: Vec<RasterPoint> = (0..32)
            .map(|i| {
                let t = (i as f32) * (core::f32::consts::TAU / 32.0);
                let r = if i % 2 == 0 { 22.0 } else { 9.0 };
                RasterPoint::new(32.0 + r * t.cos(), 32.0 + r * t.sin())
            })
            .collect();
        let a = prepare(&[star.clone()], PathFillRule::Winding, 64, 48);
        let b = prepare(&[star], PathFillRule::Winding, 64, 48);
        assert_eq!(a, b, "预处理必须确定性（同输入逐位同片段）");
        let total: u32 = (0..48).map(|r| a.rows[r as usize * 2 + 1]).sum();
        assert_eq!(total as usize, a.segment_count(), "行计数之和 = 片段总数");
    }

    #[test]
    fn coord_bias_keeps_large_integer_differences_exact() {
        // bias − bias − px 对 |col| ≤ 2²⁰、px ≤ 2²⁰ 必须逐位精确（整数差）。
        for col in [-1_000_000i64, -3, 0, 1, 4095, 1_000_000] {
            for px in [0u32, 1, 4095] {
                let base = (col as f64 + f64::from(COORD_BIAS)) as f32;
                let got = (base - COORD_BIAS) - px as f32;
                assert_eq!(got, (col - px as i64) as f32, "col={col} px={px}");
            }
        }
    }

    #[test]
    fn empty_and_oversize_inputs_are_safe() {
        for (w, h) in [(0u32, 0u32), (0, 8), (8, 0), (u32::MAX, u32::MAX)] {
            let r = std::panic::catch_unwind(|| rasterize_cpu(&[], PathFillRule::Winding, w, h));
            assert!(r.is_ok(), "({w},{h}) 不得 panic");
        }
        let c = rasterize_cpu(&[], PathFillRule::Winding, 4, 4);
        assert_eq!(c.data.len(), 16);
        assert!(c.data.iter().all(|&v| v == 0.0));
    }
}
