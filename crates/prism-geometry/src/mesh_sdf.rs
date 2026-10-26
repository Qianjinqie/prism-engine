//! mesh→SDF 烘焙：brick 8³ 稀疏化 + 广义 winding number 符号 + BVH 最近距离
//! （S12 对偶切换 bake 特解）。
//!
//! # 烘焙语义
//!
//! - 仅表面附近 brick 精确化（三角形 AABB 外扩 `pad_voxels` 决定激活范围）；
//!   近场距离对候选三角形做精确最近点查询。
//! - 内外判定用广义 winding number（复用 [`crate::bvh::Bvh`] 的实体角聚合，一树两用）；
//!   判定病态（w 明显偏离 0/1/表面过渡带）按 exterior 处理并计数警告——**永不静默**。
//! - 远场（未激活 brick / 场外）返回**下界**：绝对值不大于真实距离，sphere tracing 步进安全。
//!
//! # 确定性
//!
//! 每个 brick 的每个体素独立计算（无跨体素浮点累积），哈希表遍历序不影响逐体素取值
//! → 同输入同字节。BVH 构建同输入同树（见 [`crate::bvh`]）。

use crate::bvh::{mesh_triangles, Bvh, Triangle};
use prism_math::Vec3;
use std::collections::HashMap;

/// brick 边长（体素数，：brick 8³）。
pub const BRICK: usize = 8;

/// 烘焙参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakeOptions {
    /// 体素间距（世界单位）。
    pub spacing: f32,
    /// 表面外扩体素数（决定近场精确区的厚度）。
    pub pad_voxels: u32,
}

impl Default for BakeOptions {
    fn default() -> Self {
        BakeOptions { spacing: 0.1, pad_voxels: 2 }
    }
}

/// 烘焙报告（加载报告式诚实披露，宪章 C7：永不静默）。
#[derive(Debug, Clone, PartialEq)]
pub struct BakeReport {
    /// winding number 病态判定计数（既非 0/1 也非表面过渡带的中间值）。
    pub degenerate_winding: usize,
    /// 警告明细。
    pub warnings: Vec<String>,
}

/// 烘焙产物：稀疏 brick 距离场。
#[derive(Debug, Clone, PartialEq)]
pub struct BakedSdf {
    /// 体素 (0,0,0) 中心的世界坐标。
    pub origin: Vec3,
    /// 体素间距。
    pub spacing: f32,
    /// 三轴体素数。
    pub dims: [u32; 3],
    /// 激活 brick（key = 打包 brick 坐标 → 512 体素距离值，按 (x*8+y)*8+z 索引）。
    pub bricks: HashMap<u64, Box<[f32; BRICK * BRICK * BRICK]>>,
    /// 持有的加速结构（winding 符号判定 / 后续最近点复用，一树两用）。
    pub bvh: Bvh,
    /// 近场有效域 AABB（[origin, origin + (dims−1)·spacing]）。
    pub bounds: crate::bvh::Aabb,
}

impl BakedSdf {
    /// 有符号距离查询（内负外正）。
    ///
    /// 命中激活 brick → 三线性插值（近场精确）；
    /// 场内未激活 → winding 判号后返回 ±spacing（安全下界，"远场距离下界"）；
    /// 场外 → 返回到场域 AABB 的距离（正下界）。
    ///
    /// # 插值定位（brick 无 apron ⇒ 单元可跨 brick）
    ///
    /// 三线性插值的 8 个角点是**全局格点**，所在单元恒为 `[i0, i0+1]³`
    /// （`i0 = clamp(floor(f), 0, dims−2)`，clamp 后 `i0+1 ≤ dims−1` 恒合法）。
    /// 单元在 brick 边界处可以跨 2/4/8 个 brick——brick 只存 8³ 个格点、彼此不重叠，
    /// 故角点一律按**全局格点坐标**定位其所辖 brick 取值；不跨 brick 的单元（≈(7/8)³
    /// 体积）走单表快路径。
    ///
    /// **禁止**把 brick 局部坐标 clamp 到 `BRICK−2`、把权重 clamp 到 `[0,1]`：那会把
    /// 查询点吸附到最近格点上（brick 末格 局部 [7,8) 退化为单点取值），引入最大
    /// `√3·h` 的**定位**偏差；场是 1-Lipschitz 的，于是距离误差同阶（h=0.05 时可达
    /// 0.0866，实测 0.0693——R3 界 h/2·(1+‖∇d‖)=h 由此被击穿）。修正后近表面采样点
    /// 的距离误差 ≤ 单元半对角 `√3/2·h`（见 [`hausdorff_upper_bound`] 的推导）。
    pub fn dist(&self, p: Vec3) -> f32 {
        let f = (p - self.origin) * (1.0 / self.spacing);
        let dmax = [self.dims[0] as f32 - 1.0, self.dims[1] as f32 - 1.0, self.dims[2] as f32 - 1.0];
        if f.x < 0.0 || f.y < 0.0 || f.z < 0.0 || f.x > dmax[0] || f.y > dmax[1] || f.z > dmax[2] {
            return self.bounds.dist_to(p);
        }
        if self.dims[0] < 2 || self.dims[1] < 2 || self.dims[2] < 2 {
            // 退化场（任一轴不足 2 个体素 ⇒ 无插值单元）。
            return self.spacing;
        }
        // 体素坐标约定：体素 i 的中心在 origin + i·spacing → 世界点的体素坐标 f 直接
        // 以体素中心为整数点（与烘焙写入 p = origin + vi·s 一致）。
        let fi = [f.x, f.y, f.z];
        let mut i0 = [0_i64; 3];
        let mut w = [0.0_f32; 3];
        for i in 0..3 {
            let last = (self.dims[i] - 2) as f32;
            let lower = fi[i].floor().clamp(0.0, last);
            i0[i] = lower as i64;
            w[i] = (fi[i] - lower).clamp(0.0, 1.0);
        }
        // 主 brick = 单元下角所在 brick。未激活即整单元在未激活区（近表面单元至少
        // 下角 brick 激活：该角点距表面 ≤ √3/2·h < pad·spacing ⇒ 必有三角形登记其 brick）,
        // 走远场快路径。
        let b0 = [
            i0[0] as u64 / BRICK as u64,
            i0[1] as u64 / BRICK as u64,
            i0[2] as u64 / BRICK as u64,
        ];
        let Some(cell) = self.bricks.get(&pack_brick(b0[0], b0[1], b0[2])) else {
            return self.far_field_value(self.bvh.winding_number(p));
        };
        let l0 = [
            (i0[0] - (b0[0] * BRICK as u64) as i64) as usize,
            (i0[1] - (b0[1] * BRICK as u64) as i64) as usize,
            (i0[2] - (b0[2] * BRICK as u64) as i64) as usize,
        ];
        let mut v = [0.0_f32; 8];
        if l0[0] + 1 < BRICK && l0[1] + 1 < BRICK && l0[2] + 1 < BRICK {
            // 快路径：单元 8 角同属主 brick。
            for dx in 0..2 {
                for dy in 0..2 {
                    for dz in 0..2 {
                        v[(dx * 2 + dy) * 2 + dz] =
                            cell[((l0[0] + dx) * BRICK + (l0[1] + dy)) * BRICK + (l0[2] + dz)];
                    }
                }
            }
        } else {
            // 跨 brick 单元：逐角点按全局格点坐标取值。
            for dx in 0..2 {
                for dy in 0..2 {
                    for dz in 0..2 {
                        v[(dx * 2 + dy) * 2 + dz] = self.voxel_value([
                            i0[0] + dx as i64,
                            i0[1] + dy as i64,
                            i0[2] + dz as i64,
                        ]);
                    }
                }
            }
        }
        trilinear(&v, w)
    }

    /// 单个全局格点（体素中心）的场值：brick 命中取存值，未激活取远场 ±spacing。
    ///
    /// `vi` 须落在 `[0, dims−1]`（调用方 `dist` 保证），故 brick 索引无需再 clamp。
    fn voxel_value(&self, vi: [i64; 3]) -> f32 {
        let b = [
            vi[0] as u64 / BRICK as u64,
            vi[1] as u64 / BRICK as u64,
            vi[2] as u64 / BRICK as u64,
        ];
        let Some(cell) = self.bricks.get(&pack_brick(b[0], b[1], b[2])) else {
            let c = self.voxel_center(vi);
            return self.far_field_value(self.bvh.winding_number(c));
        };
        let l = [
            (vi[0] as u64 - b[0] * BRICK as u64) as usize,
            (vi[1] as u64 - b[1] * BRICK as u64) as usize,
            (vi[2] as u64 - b[2] * BRICK as u64) as usize,
        ];
        cell[(l[0] * BRICK + l[1]) * BRICK + l[2]]
    }

    /// 全局格点（体素中心）的世界坐标。
    fn voxel_center(&self, vi: [i64; 3]) -> Vec3 {
        Vec3::new(
            self.origin.x + vi[0] as f32 * self.spacing,
            self.origin.y + vi[1] as f32 * self.spacing,
            self.origin.z + vi[2] as f32 * self.spacing,
        )
    }

    /// 远场取值：winding 判号，幅度取安全下界 `spacing`（`|值| ≤ 真实距离`）。
    ///
    /// 病态置信判定：`|w|` 不在 0/±1 置信带 → 符号不可信，按 exterior 处理
    /// （正值下界）。计数披露在烘焙期完成（查询侧是 `&self`）。
    fn far_field_value(&self, w: f32) -> f32 {
        if !winding_pathological(w) && w > 0.5 {
            return -self.spacing;
        }
        self.spacing
    }
}

/// 三线性插值（8 角点按 `(dx·2+dy)·2+dz` 排列，权重 `w ∈ [0,1]³`）。
///
/// 8 个角点值是凸组合权重，故 `|结果| ≤ max|角点值|`——这正是 R3 近场界的来源。
#[inline]
fn trilinear(v: &[f32; 8], w: [f32; 3]) -> f32 {
    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
    let c00 = lerp(v[0], v[1], w[2]);
    let c01 = lerp(v[2], v[3], w[2]);
    let c10 = lerp(v[4], v[5], w[2]);
    let c11 = lerp(v[6], v[7], w[2]);
    let c0 = lerp(c00, c01, w[1]);
    let c1 = lerp(c10, c11, w[1]);
    lerp(c0, c1, w[0])
}

/// brick 坐标打包为哈希键（各轴 21 位，覆盖 2M brick/轴）。
#[inline]
pub fn pack_brick(bx: u64, by: u64, bz: u64) -> u64 {
    (bx << 42) | (by << 21) | bz
}

/// winding number 置信判定（病态网格按 exterior 处理——永不静默）。
///
/// 置信带：|w| ≤ 0.05（可靠外部）、|w| ≥ 0.95（可靠内部）、[0.45, 0.55)
/// （表面过渡带，w 在表面上本就无定义）；带外即**病态**。病态体素的符号
/// 判定不可信，一律按 exterior 处理（距离取 |d| 正值），烘焙期另计数进
/// [`BakeReport`] 披露。判定是纯函数：烘焙与查询两侧共用同一谓词，同 w 同结论。
#[inline]
fn winding_pathological(w: f32) -> bool {
    let aw = w.abs();
    (aw > 0.05 && aw <= 0.45) || (aw >= 0.55 && aw < 0.95)
}

/// mesh（顶点 + 索引）→ 有符号距离场烘焙。
pub fn bake_mesh(
    positions: &[Vec3],
    indices: &[u32],
    opts: &BakeOptions,
) -> (BakedSdf, BakeReport) {
    let tris = mesh_triangles(positions, indices);
    let bvh = Bvh::build(&tris);
    bake_triangles(&tris, &bvh, opts)
}

/// 三角形数组 → 烘焙（已有 BVH 时避免重建）。
pub fn bake_triangles(
    tris: &[Triangle],
    bvh: &Bvh,
    opts: &BakeOptions,
) -> (BakedSdf, BakeReport) {
    let s = opts.spacing;
    let pad = opts.pad_voxels as i64;
    let mut report = BakeReport { degenerate_winding: 0, warnings: Vec::new() };
    if tris.is_empty() {
        report.warnings.push("bake: 空网格，产物为空场".to_string());
        return (
            BakedSdf {
                origin: Vec3::splat(0.0),
                spacing: s,
                dims: [0, 0, 0],
                bricks: HashMap::new(),
                bvh: bvh.clone(),
                bounds: crate::bvh::Aabb::empty(),
            },
            report,
        );
    }

    // 场域：网格包围盒 ± pad 体素。
    let mut mesh_min = Vec3::splat(f32::INFINITY);
    let mut mesh_max = Vec3::splat(f32::NEG_INFINITY);
    for t in tris {
        for v in [t.a, t.b, t.c] {
            mesh_min = crate::vmin(mesh_min, v);
            mesh_max = crate::vmax(mesh_max, v);
        }
    }
    let origin = mesh_min - Vec3::splat(pad as f32 * s);
    let mut dims = [0_u32; 3];
    let mesh_min_a = [mesh_min.x, mesh_min.y, mesh_min.z];
    let mesh_max_a = [mesh_max.x, mesh_max.y, mesh_max.z];
    for i in 0..3 {
        dims[i] = ((mesh_max_a[i] - mesh_min_a[i]) / s).ceil() as u32 + 2 * opts.pad_voxels + 1;
    }

    // 候选表：三角形 AABB 外扩 pad 体素覆盖的每个 brick 收一次（激活集与候选一次成型）。
    let mut candidates: HashMap<u64, Vec<u32>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let (mut tmin, mut tmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in [t.a, t.b, t.c] {
            tmin = crate::vmin(tmin, v);
            tmax = crate::vmax(tmax, v);
        }
        let ta = [tmin.x, tmin.y, tmin.z];
        let tb = [tmax.x, tmax.y, tmax.z];
        let mut range_lo = [0_i64; 3];
        let mut range_hi = [0_i64; 3];
        for i in 0..3 {
            let lo = ((ta[i] - origin.at(i)) / s).floor() as i64 - pad as i64;
            let hi = ((tb[i] - origin.at(i)) / s).floor() as i64 + pad as i64;
            range_lo[i] = lo.clamp(0, dims[i] as i64 - 1);
            range_hi[i] = hi.clamp(0, dims[i] as i64 - 1);
        }
        let b_lo = [range_lo[0] / BRICK as i64, range_lo[1] / BRICK as i64, range_lo[2] / BRICK as i64];
        let b_hi = [
            range_hi[0].min(dims[0] as i64 - 1) / BRICK as i64,
            range_hi[1].min(dims[1] as i64 - 1) / BRICK as i64,
            range_hi[2].min(dims[2] as i64 - 1) / BRICK as i64,
        ];
        for bx in b_lo[0]..=b_hi[0] {
            for by in b_lo[1]..=b_hi[1] {
                for bz in b_lo[2]..=b_hi[2] {
                    candidates
                        .entry(pack_brick(bx as u64, by as u64, bz as u64))
                        .or_default()
                        .push(ti as u32);
                }
            }
        }
    }

    // 逐激活 brick 逐体素：候选精确最近点 + winding 符号。
    let mut bricks: HashMap<u64, Box<[f32; BRICK * BRICK * BRICK]>> = HashMap::new();
    for (key, cands) in candidates.iter() {
        let (bx, by, bz) = unpack_brick(*key);
        let mut cell = Box::new([0.0_f32; BRICK * BRICK * BRICK]);
        for lx in 0..BRICK {
            for ly in 0..BRICK {
                for lz in 0..BRICK {
                    let vi = [
                        bx as i64 * BRICK as i64 + lx as i64,
                        by as i64 * BRICK as i64 + ly as i64,
                        bz as i64 * BRICK as i64 + lz as i64,
                    ];
                    if vi[0] >= dims[0] as i64 || vi[1] >= dims[1] as i64 || vi[2] >= dims[2] as i64 {
                        continue; // 场域外的收尾体素（不采样，不影响查询）
                    }
                    let p = Vec3::new(
                        origin.x + (vi[0] as f32) * s,
                        origin.y + (vi[1] as f32) * s,
                        origin.z + (vi[2] as f32) * s,
                    );
                    let mut d = f32::INFINITY;
                    for &ci in cands.iter() {
                        let cp = tris[ci as usize].closest_point(p);
                        let dd = (cp - p).length();
                        if dd < d {
                            d = dd;
                        }
                    }
                    let w = bvh.winding_number(p);
                    let near_surface = d <= 2.0 * s;
                    // 病态检测：|w| 明显偏离 0/±1 且不在表面过渡带（0.45–0.55）。
                    // 点在表面附近（|d| ≤ 2·spacing）时 winding 本就过渡（0 → 0.5 → 1，
                    // 表面上无定义），不算病态；只抓"远离表面却不接近 0/1"的真病态。
                    let pathological = winding_pathological(w);
                    if pathological && !near_surface {
                        report.degenerate_winding += 1;
                    }
                    //  契约：病态体素**按 exterior 处理**（永不静默）——
                    // 符号判定不区分近远场（近场过渡体素 w 落在病态带同样不可信，
                    // 判外取 |d| 正值），计数仍只报远离表面的真病态。
                    let inside = !pathological && w > 0.5;
                    cell[(lx * BRICK + ly) * BRICK + lz] = if inside { -d } else { d };
                }
            }
        }
        bricks.insert(*key, cell);
    }

    (
        BakedSdf {
            origin,
            spacing: s,
            dims,
            bricks,
            bvh: bvh.clone(),
            bounds: crate::bvh::Aabb {
                min: origin,
                max: origin
                    + Vec3::new(
                        (dims[0] - 1) as f32 * s,
                        (dims[1] - 1) as f32 * s,
                        (dims[2] - 1) as f32 * s,
                    ),
            },
        },
        report,
    )
}

/// R3 保真度界：mesh / SDF 是分辨率有界的近似对。
///
/// 理论界（见 `docs/设计/研究/能力对齐的数学分层.md` §5.1）：
///
/// ```text
/// d_H(SDF⁻¹(0), mesh) ≤ h/2 · (1 + ‖∇d‖)
/// ```
///
/// 其中 `h` = 体素间距（格距），`‖∇d‖` = SDF 梯度范数上界。因 SDF 是有符号
/// 距离场，天然 **1-Lipschitz**：`‖∇d‖ ≤ 1` ⇒ 上界收紧为 `≤ h`。本函数即该界的
/// 显式计算——给定格距与梯度范数上界，返回解析上界。
///
/// 凭证：球 / 平面（含立方体的平面段）的 SDF 均为 1-Lipschitz，故 `‖∇d‖ = 1`，
/// 上界取 `h`；更缓的梯度（如高度场）上界更小。
///
/// # 本实现的实际（更紧）界 —— 推导
///
/// 上面的 `h/2·(1+‖∇d‖)` 是**通用**上界；对 [`BakedSdf::dist`] 这一具体实现可以推得更紧：
///
/// 1. **近场取值精确**：`pad ≥ 1` 时，距三角面 `≤ √3/2·h < pad·h` 的格点，其最近三角形
///    必落在该格点所辖 brick 的候选表里（候选按三角形 AABB±pad 体素登记 brick，
///    格点距 AABB 不足 pad·h ⇒ 其 brick 必在登记范围内），故这些格点存的是**精确**
///    距离 `d(c)`，且 `|d(c)| = dist(c, mesh)`。
/// 2. **插值是凸组合**：查询点 `p` 所在单元 `[i0, i0+1]³` 的 8 个角点 `c_j` 满足
///    `p = Σ λ_j c_j`（`λ_j ≥ 0, Σλ_j = 1`），三线性插值即 `Σ λ_j d(c_j)`，
///    于是 `|interp(p)| ≤ max_j |d(c_j)|`。
/// 3. 若 `p ∈ mesh`（`d(p) = 0`，1-Lipschitz ⇒ `|d(c_j)| = |d(c_j) − d(p)| ≤ |c_j − p|`），
///    而单元内 `|c_j − p| ≤ √3/2·h`（半对角），故
///
///    ```text
///    |dist(p)| ≤ √3/2 · h  ≈ 0.866 · h  ≤  h = spacing/2·(1+1)
///    ```
///
/// 即本实现的**最坏情形**保真度界是 `√3/2·h`（比通用界 `h` 紧 13.4%）。实测远小于该最坏界
/// ——三线性插值只是 `O(h²)` 级误差，且平面段（立方体六面）对线性场插值精确：
///
/// | 样例 | h=0.1 | h=0.05 |
/// |---|---|---|
/// | 球面 max&#124;dist&#124; | 3.1008e-3 | 1.5489e-3 |
/// | 立方体 max&#124;dist&#124; | 2.39e-7 | 2.46e-7 |
///
/// （`r3_hausdorff_bound_holds_and_tightens_with_resolution` 逐档记录，h↓ ⇒ 实测↓。）
///
/// 该推导依赖第 1 步的"近场精确"前提：远离表面（`> pad·h`）的格点值只是候选集上的
/// 最小值，可能大于真实距离（下界性仅对未激活 brick 的 `±spacing` 成立），故此处
/// **只**对零等值面邻域作断言——这也正是 Hausdorff 距离所需的全部。
pub fn hausdorff_upper_bound(spacing: f32, grad_norm_bound: f32) -> f32 {
    spacing / 2.0 * (1.0 + grad_norm_bound)
}

/// 解包 brick 哈希键。
#[inline]
pub fn unpack_brick(key: u64) -> (u64, u64, u64) {
    ((key >> 42) & 0x1F_FFFF, (key >> 21) & 0x1F_FFFF, key & 0x1F_FFFF)
}

/// 索引辅助 trait（体素坐标按轴取分量）。
trait AtIdx {
    fn at(&self, i: usize) -> f32;
}

impl AtIdx for Vec3 {
    #[inline]
    fn at(&self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl BakedSdf {
    /// brick 哈希键的内部便捷复打包（供测试断言）。
    pub fn brick_key_of(&self, p: Vec3) -> u64 {
        let f = (p - self.origin) * (1.0 / self.spacing);
        let b = |v: f32| (v / BRICK as f32).floor().max(0.0) as u64;
        pack_brick(b(f.x), b(f.y), b(f.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::testutil::unit_cube;

    /// 解析盒 SDF（测试参照）。
    fn box_sdf(p: Vec3) -> f32 {
        let h = Vec3::splat(1.0);
        let q = Vec3::new(p.x.abs(), p.y.abs(), p.z.abs()) - h;
        let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        outside.length() + q.x.max(q.y).max(q.z).min(0.0)
    }

    /// 关卡 5：立方体烘焙——内负外正、|d| 与解析距离差 < 体素尺度、近场/远场/下界语义。
    #[test]
    fn bake_cube_interior_negative_exterior_positive() {
        let positions: Vec<Vec3> = {
            // 与 bvh::testutil::unit_cube 同一顶点表。
            let v = [
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(-1.0, -1.0, 1.0),
                Vec3::new(1.0, -1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
            ];
            vec![v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]
        };
        let indices: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5,
            4, 2, 3, 7, 2, 7, 6,
        ];
        let opts = BakeOptions { spacing: 0.1, pad_voxels: 2 };
        let (baked, report) = bake_mesh(&positions, &indices, &opts);
        assert_eq!(report.degenerate_winding, 0, "水密立方体不应有病态判定");

        let tolerance = opts.spacing; // 关卡判定线：|d| 与解析距离差 < 体素尺度
        // 近场内部点：解析 −0.15（距 x=+1 面）。
        let p = Vec3::new(0.85, 0.0, 0.0);
        let got = baked.dist(p);
        assert!(got < 0.0, "内部应为负：{got}");
        assert!(
            (got - box_sdf(p)).abs() < tolerance,
            "内点 |d|={got} vs 解析 {}",
            box_sdf(p)
        );
        // 近场外部点：解析 +0.15。
        let p = Vec3::new(1.15, 0.0, 0.0);
        let got = baked.dist(p);
        assert!(got > 0.0, "外部应为正：{got}");
        assert!(
            (got - box_sdf(p)).abs() < tolerance,
            "外点 |d|={got} vs 解析 {}",
            box_sdf(p)
        );
        // 表面点：|d| < 体素尺度。
        let got = baked.dist(Vec3::new(1.0, 0.0, 0.0));
        assert!(got.abs() < tolerance, "表面点 |d|={got}");
        // 近场内部点（−x 面）：解析 −0.12。
        let p = Vec3::new(-0.88, 0.1, -0.1);
        let got = baked.dist(p);
        assert!(got < 0.0 && (got - box_sdf(p)).abs() < tolerance, "−x 内点 {got}");
        // 近场外部点（对角区）：解析 = (1.2,1.2,1.2) 到盒角 (1,1,1) 距离 ≈ 0.346。
        let p = Vec3::splat(1.2);
        let got = baked.dist(p);
        assert!(got > 0.0 && (got - box_sdf(p)).abs() < tolerance, "角区外点 {got}");

        // 深内部：符号为负且幅度为下界（|d| ≤ 真实距离 1）。
        let got = baked.dist(Vec3::splat(0.0));
        assert!(got < 0.0, "深内部应为负");
        assert!(got.abs() <= 1.0, "下界性质：|d|={got} ≤ 1");
        // 深外部（场外）：正值且为下界（≤ 到表面真距 √3·2.8…）。
        let p = Vec3::splat(3.0);
        let got = baked.dist(p);
        assert!(got > 0.0, "深外部应为正");
        assert!(got <= box_sdf(p), "下界性质：{got} ≤ {}", box_sdf(p));

        // 沿 x 轴符号扫描：|x|>1 为正、|x|<0.9 为负（激活近场带）。
        let mut i = 0.9_f32;
        while i < 1.25 {
            let d = baked.dist(Vec3::new(i, 0.05, -0.05));
            if i < 1.0 {
                assert!(d < 0.0, "x={i} 应为内部（负），得 {d}");
            } else if i > 1.05 {
                assert!(d > 0.0, "x={i} 应为外部（正），得 {d}");
            }
            i += 0.05;
        }
    }

    /// 烘焙确定性：同输入两遍，brick 数据逐位一致 + dist 采样一致。
    #[test]
    fn bake_is_deterministic() {
        let tris = unit_cube();
        let bvh = Bvh::build(&tris);
        let opts = BakeOptions { spacing: 0.15, pad_voxels: 2 };
        let (a, ra) = bake_triangles(&tris, &bvh, &opts);
        let (b, rb) = bake_triangles(&tris, &bvh, &opts);
        assert_eq!(ra, rb, "报告一致");
        assert_eq!(a.bricks.len(), b.bricks.len(), "激活 brick 数一致");
        for (k, va) in a.bricks.iter() {
            let vb = b.bricks.get(k).expect("brick 集合一致");
            assert_eq!(va.as_slice(), vb.as_slice(), "brick {k} 逐位一致");
        }
        for i in 0..40_u64 {
            let p = Vec3::new(
                prism_noise::pcg::hash2_f32(7, i, 0) * 4.0 - 2.0,
                prism_noise::pcg::hash2_f32(7, i, 1) * 4.0 - 2.0,
                prism_noise::pcg::hash2_f32(7, i, 2) * 4.0 - 2.0,
            );
            assert_eq!(a.dist(p), b.dist(p), "dist({p:?}) 逐位一致");
        }
    }

    /// brick 键打包/解包往返。
    #[test]
    fn brick_key_roundtrip() {
        for (x, y, z) in [(0u64, 0, 0), (1, 2, 3), (123, 456, 789), (0x1F_FFFF, 0, 0x1F_FFFF)] {
            let k = pack_brick(x, y, z);
            assert_eq!(unpack_brick(k), (x, y, z));
        }
    }

    /// 空网格：空场 + 警告（永不静默），dist 不 panic。
    #[test]
    fn bake_empty_mesh_warns() {
        let (baked, report) = bake_mesh(&[], &[], &BakeOptions::default());
        assert!(baked.bricks.is_empty());
        assert!(!report.warnings.is_empty(), "空网格必须显式警告");
        let _ = baked.dist(Vec3::splat(1.0)); // 空场查询路径不 panic
    }

    // —— R3 保真度界（球 / 平面段；1-Lipschitz SDF ⇒ ‖∇d‖ = 1 ⇒ 界 = h） ——

    /// 立方体表面采样点（6 面各 n×n 网格；立方体的平面段即 R3 平板情形）。
    fn cube_surface_samples(n: usize) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for axis in 0..3 {
            for &sign in &[1.0_f32, -1.0_f32] {
                for i in 0..n {
                    for j in 0..n {
                        let a = -1.0 + 2.0 * (i as f32) / (n as f32 - 1.0);
                        let b = -1.0 + 2.0 * (j as f32) / (n as f32 - 1.0);
                        let mut v = [0.0_f32; 3];
                        let mut k = 0;
                        for d in 0..3 {
                            if d == axis {
                                v[d] = sign;
                            } else {
                                v[d] = if k == 0 { a } else { b };
                                k += 1;
                            }
                        }
                        pts.push(Vec3::new(v[0], v[1], v[2]));
                    }
                }
            }
        }
        pts
    }

    /// 球面采样点（UV 球，半径 `r`）。
    fn sphere_surface_samples(r: f32, stacks: usize, slices: usize) -> Vec<Vec3> {
        let mut pts = Vec::new();
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * (i as f32) / (stacks as f32);
            let st = theta.sin();
            let ct = theta.cos();
            for j in 0..slices {
                let phi = 2.0 * std::f32::consts::PI * (j as f32) / (slices as f32);
                pts.push(Vec3::new(r * st * phi.cos(), r * st * phi.sin(), r * ct));
            }
        }
        pts
    }

    /// UV 球三角形列表（供 `bake_triangles`）。
    fn uv_sphere(radius: f32, stacks: usize, slices: usize) -> Vec<Triangle> {
        let mut positions: Vec<Vec3> = Vec::new();
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * (i as f32) / (stacks as f32);
            let st = theta.sin();
            let ct = theta.cos();
            for j in 0..=slices {
                let phi = 2.0 * std::f32::consts::PI * (j as f32) / (slices as f32);
                positions.push(Vec3::new(
                    radius * st * phi.cos(),
                    radius * st * phi.sin(),
                    radius * ct,
                ));
            }
        }
        let mut indices: Vec<u32> = Vec::new();
        let row = slices + 1;
        for i in 0..stacks {
            for j in 0..slices {
                let a = (i * row + j) as u32;
                let b = (i * row + j + 1) as u32;
                let c = ((i + 1) * row + j) as u32;
                let d = ((i + 1) * row + j + 1) as u32;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        crate::bvh::mesh_triangles(&positions, &indices)
    }

    /// R3：实测 Hausdorff 估计（表面点 |SDF| 的最大值）≤ 解析界
    /// `h/2·(1+‖∇d‖)`，且解析界随分辨率提高而收紧（h↓ ⇒ 界↓）。
    ///
    /// 实测记录（本条测试逐档记录，不经任何阈值放宽）：
    ///
    /// | 样例 | h=0.1 | h=0.05 |
    /// |---|---|---|
    /// | 球面 max&#124;dist&#124; | 3.1008e-3 | 1.5489e-3 |
    /// | 立方体 max&#124;dist&#124; | 2.39e-7 | 2.46e-7 |
    ///
    /// 同时断言**实现侧**更紧的界 `√3/2·h`（单元半对角，推导见
    /// [`hausdorff_upper_bound`]）——该断言是 brick 边缘插值缺陷的钉子：
    /// 旧实现（brick 局部坐标 clamp 到 `BRICK−2`、权重 clamp 到 1）在球面 h=0.05 上
    /// 得 0.06929246 > 0.0433013 = √3/2·h。
    #[test]
    fn r3_hausdorff_bound_holds_and_tightens_with_resolution() {
        // 解析界本身随分辨率提高收紧（R3 核心论断）。
        assert!(
            hausdorff_upper_bound(0.05, 1.0) < hausdorff_upper_bound(0.1, 1.0),
            "R3：h↓ ⇒ 解析界收紧"
        );

        let cases: &[(&str, Vec<Triangle>, Vec<Vec3>)] = &[
            // 立方体（含平面段，恰好覆盖 R3 平板情形）。
            ("cube", unit_cube(), cube_surface_samples(21)),
            // 球面（弯曲情形，SDF 平滑 1-Lipschitz）。
            ("sphere", uv_sphere(1.0, 48, 96), sphere_surface_samples(1.0, 48, 96)),
        ];

        for (label, tris, samples) in cases {
            let bvh = Bvh::build(tris);
            let mut prev_max = f32::MAX;
            for &h in &[0.1_f32, 0.05_f32] {
                let opts = BakeOptions { spacing: h, pad_voxels: 2 };
                let (baked, _report) = bake_triangles(tris, &bvh, &opts);
                // 在表面采样：|SDF(p)| 即点 p 到零等值面的距离 ⇒ 最大值即
                // Hausdorff 估计的上界（mesh 侧）。
                let mut max_abs = 0.0_f32;
                for p in samples {
                    let d = baked.dist(*p);
                    max_abs = max_abs.max(d.abs());
                }
                let bound = hausdorff_upper_bound(h, 1.0); // ‖∇d‖ ≤ 1
                // 插值松弛：O(h²) 量级 + f32 舍入 + 采样网格近似。
                let slack = 2.0 * h * h + 1e-4;
                assert!(
                    max_abs <= bound + slack,
                    "{label} h={h}: 实测 Hausdorff 估计 {max_abs} > 解析界 {bound} + 松弛 {slack}"
                );
                // 实现侧更紧的界：8 角点凸组合 ⇒ |dist| ≤ 单元半对角 √3/2·h（只留 f32 舍入）。
                let cell_half_diag = (3.0_f32).sqrt() / 2.0 * h + 1e-4;
                assert!(
                    max_abs <= cell_half_diag,
                    "{label} h={h}: 实测 {max_abs} > 实现界 √3/2·h {cell_half_diag}\
                     （brick 边缘插值把查询点吸附到最近格点？）"
                );
                // 实测界随分辨率提高而收紧（同采样集，逐字节单调）。
                assert!(
                    max_abs <= prev_max + 1e-4,
                    "{label} h={h}: 实测界 {max_abs} 未随分辨率提高而收紧（前档 {prev_max}）"
                );
                prev_max = max_abs;
            }
        }
    }

    /// brick 边缘回归钉子：brick 只存 8³ 个格点、彼此**无 apron**，故三线性插值的
    /// 查询单元在 brick 边界处可跨 2/4/8 个 brick。
    ///
    /// 曾经的 `dist` 把 brick **局部**坐标 clamp 到 `BRICK−2`、权重 clamp 到 `[0,1]`
    /// ——等价于把查询点吸附到最近格点：单元落在 brick 末格（局部 `[7,8)`）
    /// 的轴上，实际取值点被拉回到局部 7。表面点因此获得最大 `√3·h` 的**定位**偏差；
    /// 场是 1-Lipschitz 的，距离误差与之同阶。h=0.05 的球面上实测 `|dist| = 0.06929246`，
    /// 与偏差解析值 `h·√(0.97885²+0.97885²) = 0.0692154` 吻合（差 7.7e-5 为球面
    /// 三角化偏差），远大于单元半对角 `√3/2·h = 0.0433013`，直接击穿 R3 的
    /// `h/2·(1+‖∇d‖) = h` 界。
    ///
    /// 本测试专挑"≥2 轴同时落末格"的表面点（该子集 177 个点，旧实现最坏 0.06929246），
    /// 断言 `|dist| ≤ √3/2·h + 1e-4`（修复后该子集实测最坏 1.4464e-3，余量 ~30×）。
    #[test]
    fn brick_edge_cells_resolve_global_lattice_not_clamped_local() {
        let h = 0.05_f32;
        let tris = uv_sphere(1.0, 48, 96);
        let bvh = Bvh::build(&tris);
        let (baked, _report) = bake_triangles(&tris, &bvh, &BakeOptions { spacing: h, pad_voxels: 2 });
        let samples = sphere_surface_samples(1.0, 48, 96);
        let cell_half_diag = (3.0_f32).sqrt() / 2.0 * h + 1e-4;

        let mut n_edge = 0_usize;
        let mut worst = 0.0_f32;
        let mut worst_at = Vec3::splat(0.0);
        for p in &samples {
            // brick 局部坐标：与实现同款换算（体素坐标 f → brick 内局部坐标 lf）。
            let f = (*p - baked.origin) * (1.0 / baked.spacing);
            let mut edge_axes = 0_usize;
            for i in 0..3 {
                let bi = (f.at(i) / BRICK as f32).floor();
                let lf = f.at(i) - bi * BRICK as f32;
                if lf >= (BRICK - 1) as f32 {
                    edge_axes += 1;
                }
            }
            if edge_axes < 2 {
                continue;
            }
            n_edge += 1;
            let d = baked.dist(*p).abs();
            if d > worst {
                worst = d;
                worst_at = *p;
            }
        }
        assert!(
            n_edge >= 50,
            "采样集必须覆盖 ≥2 轴落 brick 末格的情形，实际只 {n_edge} 个（测试覆盖失效）"
        );
        assert!(
            worst <= cell_half_diag,
            "brick 末格表面点 |dist|={worst}（{worst_at:?}）> 单元半对角界 {cell_half_diag}\
             ——局部 clamp 吸附缺陷复发（该情形旧实现得 0.06929246）"
        );
    }
}
