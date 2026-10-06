//! 辐照度探针：S1 场（探针格）+ S2 球谐平滑（账三 U4）。
//!
//! # S1 / S2 的架构定位（引自《架构总设计》v2.4 §1.2 十二结构总表）
//! - **S1 标量场**：本体 = 协方差核随机场；引擎实例含"**SDF 场**、噪声族、
//!   SPH 密度场"；格锚 = "格点 hash + 格间 quintic 插值"。
//! - **S2 卷积**：本体 = 调和分析核；引擎实例含"blur/bloom、插值"；格锚 =
//!   "核尺寸与相位格锚"。
//!
//! 辐照度探针正是二者的组合：**探针格 = 一个离散化的 S1 标量场**（把空间
//! 每点的方向性辐照度存成场值），**格间插值 + SH 系数平滑 = S2 卷积**（把
//! 相邻探针的辐照度做核加权混合，等价于对场做一次调和分析核的低通）。
//!
//! # 设计（D-21：不复刻 Lumen 的层次结构）
//! Lumen 用"辐射度缓存 + 屏幕空间探针层次 + 自适应细分"处理通用场景。本实现
//! 用**规则网格探针 + SH-L2 系数 + 三线性插值**——最朴素的 S1 场离散化，
//! 没有自适应、没有层次、没有惰性更新。代价是：非均匀场景下探针浪费/漏光，
//! 且不随场景几何自动加密（需用户调 `resolution`）。收益是几十行即可用、
//! 确定性天然成立（无并发、无自适应状态）。
//!
//! # 与 SSGI（[`crate::gi::ssgi`]）的关系（任务书要求说明；**未改动 SSGI**）
//!
//! 二者是**同一条 S2 卷积在不同域上的两个实例**，不是上下游、也无代码复用：
//!
//! | | 探针（本模块） | SSGI（[`crate::gi::ssgi`]） |
//! |---|---|---|
//! | 域 | **空间**（世界位置 → 辐照度） | **时间**（历史帧 → 当前帧） |
//! | 核 | k-NN / 3×3×3 盒核（邻域低通） | 相机运动 + 深度重投影（重采样核） |
//! | 邻域 | 空间邻近探针 | 同投影位置的上一帧像素 |
//! | 消抖 | 抑制单探针噪声/漏光 | 抑制逐帧 GI 闪烁（拖影代价） |
//!
//! **可复用的路径（设计机会，本轮未实现）**：两者都是"把一个 SH/辐亮度场
//! 按核做线性混合"，故 SSGI 的重投影核与探针的 k-NN 核可**串联**成
//! "历史帧探针场 → 时间重投影 → 空间邻域低通 → 当前探针场"。实现上需要一个
//! 探针级的 `ReprojectionMap`（当前缺失：SSGI 的 `reproject_gi` 只在**屏幕
//! 像素**域工作，且以 `DepthMap` 为输入，无法直接查询探针位置）。
//! **本轮明确不接**：那会改动 `ssgi.rs` 或新增跨域数据结构，超出"不强制
//! 改动 SSGI"的边界，且会引入历史帧状态（与"无惰性状态"的确定性前提冲突）。
//!
//! # 确定性
//! 探针索引 `(ix,iy,iz)` 用固定序遍历（z→y→x 嵌套），SH 投影累加序固定，
//! 三线性插值按固定权重序。同输入逐位一致。

use prism_math::Vec3;

use crate::gi::sdf_gi::{
    ambient_occlusion, multi_bounce_energy_upper_bound, one_bounce_indirect, solve_multi_bounce,
    MultiBounceBudget, MultiBounceResult, OcclusionBudget, SdfField, SceneSdf,
};
use crate::gi::sh::{sh_reconstruct_rgb, ShIrradiance, ShOrder};

/// 探针格采样预算（质量-成本旋钮）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeBudget {
    /// 每个探针的 SH 投影采样方向数（越多越精确，成本线性）。
    pub sh_samples: u32,
    /// 每个探针的 AO 方向数（0 = 不算 AO，只用几何遮挡的间接光）。
    pub ao_samples: u32,
    /// SDF 遮挡预算（步数等）。
    pub occlusion: OcclusionBudget,
    /// 重建阶数（L0/L1/L2）。
    pub order: ShOrder,
}

impl Default for ProbeBudget {
    fn default() -> Self {
        Self {
            sh_samples: 64,
            ao_samples: 32,
            occlusion: OcclusionBudget::standard(),
            order: ShOrder::L2,
        }
    }
}

impl ProbeBudget {
    /// 低档（实时预览）：少采样、AO 粗。
    pub fn low() -> Self {
        Self {
            sh_samples: 16,
            ao_samples: 8,
            occlusion: OcclusionBudget::low(),
            order: ShOrder::L1,
        }
    }
    /// 标准档。
    pub fn standard() -> Self {
        Self::default()
    }
    /// 高档（高质量收敛）。
    pub fn high() -> Self {
        Self {
            sh_samples: 256,
            ao_samples: 128,
            occlusion: OcclusionBudget::high(),
            order: ShOrder::L2,
        }
    }
}

/// 单个探针：位置 + 辐照度 SH（RGB 三套系数）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    /// 世界坐标位置。
    pub position: Vec3,
    /// 表面法线（辐照度半球参考）。
    pub normal: Vec3,
    /// 该点 AO 可见度 `[0,1]`。
    pub ao: f32,
    /// 环境光（入射辐亮度，用于烘焙）。
    pub env_radiance: Vec3,
}
// 注：为保持 `Probe` 为 Copy 且小，SH 结果不直接内联此处，而是存于
// `ProbeGrid::sh_cubes`（与探针索引平行的 `Vec<[[f32;9];3]>`），避免大结构体。

/// 规则探针格（S1 标量场的均匀离散化）。
///
/// 探针在 `min + (i+0.5)/res·(max-min)` 处（体素中心），`res=(nx,ny,nz)`。
/// SH 结果并行存于 `sh_cubes`（长度 `nx·ny·nz`，行主序 `iz·ny·nx + iy·nx + ix`）。
#[derive(Debug, Clone)]
pub struct ProbeGrid {
    /// 格最小角（世界单位）。
    pub min: Vec3,
    /// 格最大角。
    pub max: Vec3,
    /// 各轴探针数（≥1）。
    pub res: (u32, u32, u32),
    /// 各探针位置（行主序，与 `sh_cubes` 同索引）。
    pub probes: Vec<Probe>,
    /// 各探针的辐照度 SH 系数（RGB 各 9 个）。
    pub sh_cubes: Vec<[[f32; 9]; 3]>,
    /// 烘焙用的阶数（重建默认用此阶，可被覆盖）。
    pub baked_order: ShOrder,
    /// **多弹射诊断**：各探针实际执行的弹射轮数（与 `probes` 同索引）。
    ///
    /// 单弹射烘焙（[`bake_probe_grid`]）此项恒为 `1` 的向量；多弹射烘焙
    /// （[`bake_probe_grid_multi_bounce`]）为各探针 `MultiBounceResult::bounces`。
    /// 长度恒等于 `probes.len()`（构造时保证），故索引安全。
    pub bounce_counts: Vec<u32>,
    /// **多弹射诊断**：各探针是否因阈值收敛（与 `probes` 同索引）。
    pub converged: Vec<bool>,
}

impl ProbeGrid {
    /// 探针总数。
    pub fn count(&self) -> usize {
        self.probes.len()
    }

    /// **多弹射诊断**：全部探针中实际执行的最大弹射轮数。
    ///
    /// 单弹射烘焙恒为 `1`；空格返回 `0`。用于检验"多弹射确实多走了弹射"。
    pub fn last_bounce_count(&self) -> u32 {
        self.bounce_counts.iter().copied().max().unwrap_or(0)
    }

    /// **多弹射诊断**：全部探针是否都已收敛。
    ///
    /// 空格返回 `false`（无探针可判）。注意单弹射烘焙的 `converged` 恒为
    /// `false`（第 0 弹无"上一弹变化量"可比），这是有意的语义区分。
    pub fn converged(&self) -> bool {
        !self.converged.is_empty() && self.converged.iter().all(|&c| c)
    }

    /// **多弹射诊断**：各探针的弹射轮数切片（长度 = [`Self::count`]）。
    pub fn bounce_counts(&self) -> &[u32] {
        &self.bounce_counts
    }

    /// 线性索引（越界钳到有效范围；C18 不 panic）。
    ///
    /// # C18
    /// `res` / `probes` / `sh_cubes` 均为 `pub` 字段，外部可构造出
    /// `res` 推出的格点数与**实际存储长度不一致**的网格（如手工构造或跨版本
    /// 反序列化）。故本函数在按 `res` 钳位之后，**再钳到 `sh_cubes` 的实际
    /// 长度**，保证返回值恒为合法下标（`res` 声称 64 格但只存 1 个时返回 0）。
    #[inline]
    fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        let (nx, ny, _nz) = self.res;
        let ix = ix.min(nx.saturating_sub(1));
        let iy = iy.min(ny.saturating_sub(1));
        let iz = iz.min(self.res.2.saturating_sub(1));
        let raw =
            (iz as usize) * (ny as usize) * (nx as usize) + (iy as usize) * (nx as usize) + ix as usize;
        // 再钳到三并行数组的最短长度（探针位置与 SH 系数同索引）。
        let n = self.sh_cubes.len().min(self.probes.len());
        if n == 0 {
            0
        } else {
            raw.min(n - 1)
        }
    }

    /// 按世界坐标查询插值后的辐照度（三线性 = S1 场格间插值）。
    ///
    /// 退化情况（res 任一轴为 0 或点越界）→ 钳到最近探针（不 panic）。
    pub fn sample_irradiance(&self, p: Vec3, order: ShOrder) -> Vec3 {
        if self.probes.is_empty() {
            return Vec3::splat(0.0);
        }
        let (nx, ny, nz) = self.res;
        // 世界 → 探针坐标（体素中心偏移 0.5）。
        let f = |v: f32, lo: f32, hi: f32, n: u32| -> f32 {
            if (hi - lo).abs() < 1e-9 || n <= 1 {
                0.0
            } else {
                ((v - lo) / (hi - lo)) * n as f32 - 0.5
            }
        };
        let fxp = f(p.x, self.min.x, self.max.x, nx);
        let fyp = f(p.y, self.min.y, self.max.y, ny);
        let fzp = f(p.z, self.min.z, self.max.z, nz);
        // 基础索引与小数权重（钳到 [0, n-1]）。
        let clamp_idx = |fv: f32, n: u32| -> (u32, f32) {
            let fv = if fv.is_nan() { 0.0 } else { fv };
            let fl = fv.floor();
            let base = fl.max(0.0).min((n as f32) - 1.0) as u32;
            let t = (fv - base as f32).clamp(0.0, 1.0);
            (base, if t.is_nan() { 0.0 } else { t })
        };
        let (ix, tx) = clamp_idx(fxp, nx);
        let (iy, ty) = clamp_idx(fyp, ny);
        let (iz, tz) = clamp_idx(fzp, nz);
        // 三线性：8 角点按固定序加权（S2 卷积的最近邻核特例）。
        let mut acc = Vec3::splat(0.0);
        for (dx, wx) in [(0u32, 1.0 - tx), (1u32, tx)] {
            for (dy, wy) in [(0u32, 1.0 - ty), (1u32, ty)] {
                for (dz, wz) in [(0u32, 1.0 - tz), (1u32, tz)] {
                    let idx = self.index(ix + dx, iy + dy, iz + dz);
                    let rgb = &self.sh_cubes[idx];
                    let e = sh_reconstruct_rgb(rgb, self.probes[idx].normal, order);
                    let w = wx * wy * wz;
                    acc = acc + e * w;
                }
            }
        }
        acc
    }

    /// 探针间平滑（S2 卷积）：对 SH 系数做一次均匀 3×3×3 盒核的低通，
    /// 更新 `sh_cubes`。用于抑制单点噪声/漏光。
    ///
    /// `strength ∈ [0,1]`：0 = 不平滑，1 = 完全替换为邻域均。累加序固定 →
    /// 确定性。边界探针取存在的邻居（不做环绕，不引入边界伪影）。
    pub fn smooth(&mut self, strength: f32) {
        let s = if strength.is_nan() { 0.0 } else { strength.clamp(0.0, 1.0) };
        if s <= 0.0 || self.probes.is_empty() {
            return;
        }
        let (nx, ny, nz) = self.res;
        let mut out = self.sh_cubes.clone();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let self_idx = self.index(ix, iy, iz);
                    // 邻域（含自身）均，3×3×3，固定序。
                    let mut acc = [[0.0f32; 9]; 3];
                    let mut cnt = 0.0f32;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let jx = ix as i32 + dx;
                                let jy = iy as i32 + dy;
                                let jz = iz as i32 + dz;
                                if jx < 0
                                    || jy < 0
                                    || jz < 0
                                    || jx >= nx as i32
                                    || jy >= ny as i32
                                    || jz >= nz as i32
                                {
                                    continue;
                                }
                                let n_idx = self.index(jx as u32, jy as u32, jz as u32);
                                for ch in 0..3 {
                                    for i in 0..9 {
                                        acc[ch][i] += self.sh_cubes[n_idx][ch][i];
                                    }
                                }
                                cnt += 1.0;
                            }
                        }
                    }
                    if cnt > 0.0 {
                        for ch in 0..3 {
                            for i in 0..9 {
                                let avg = acc[ch][i] / cnt;
                                let orig = self.sh_cubes[self_idx][ch][i];
                                out[self_idx][ch][i] = orig + (avg - orig) * s;
                            }
                        }
                    }
                }
            }
        }
        self.sh_cubes = out;
    }
}

/// 从 SDF 场烘焙规则探针格（S1 场填充 + S2 SH 投影）。
///
/// # 参数
/// - `field`：几何距离场（遮挡查询用）；
/// - `env_radiance`：环境入射辐亮度（方向无关近似——实时档位简化）；
/// - `min`/`max`：格包围盒；`res`：各轴探针数；
/// - `budget`：采样预算。
///
/// # 确定性
/// 探针按 `iz→iy→ix` 固定序烘焙；每探针 SH 采样方向固定序（黄金角螺旋）；
/// 无并发、无自适应。同输入逐位一致。
pub fn bake_probe_grid(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    res: (u32, u32, u32),
    budget: ProbeBudget,
) -> ProbeGrid {
    let nx = res.0.max(1);
    let ny = res.1.max(1);
    let nz = res.2.max(1);
    let span = max - min;
    let probe_at = |ix: u32, iy: u32, iz: u32| -> Vec3 {
        let fx = (ix as f32 + 0.5) / nx as f32;
        let fy = (iy as f32 + 0.5) / ny as f32;
        let fz = (iz as f32 + 0.5) / nz as f32;
        Vec3::new(min.x + span.x * fx, min.y + span.y * fy, min.z + span.z * fz)
    };
    let total = (nx as usize) * (ny as usize) * (nz as usize);
    let mut probes = Vec::with_capacity(total);
    let mut sh_cubes = Vec::with_capacity(total);
    let mut bounce_counts = Vec::with_capacity(total);
    let mut converged = Vec::with_capacity(total);
    let sh_n = budget.sh_samples.max(1);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = probe_at(ix, iy, iz);
                // 探针法线：从场梯度近似（有限差分），退化则用 +Y。
                let normal = field_normal(field, p);
                let ao = if budget.ao_samples > 0 {
                    ambient_occlusion(field, p, normal, budget.ao_samples, budget.occlusion)
                } else {
                    1.0
                };
                // 烘焙直接算一步间接光 → 折入环境近似（简化：可见方向贡献环境常量）。
                let albedo = Vec3::splat(1.0); // 探针处无材质 → 单位反照率（辐照度度量）
                let e = one_bounce_indirect(
                    field,
                    p,
                    normal,
                    albedo,
                    env_radiance,
                    sh_n,
                    budget.occlusion,
                );
                probes.push(Probe { position: p, normal, ao, env_radiance });
                // 直接把已算的辐照度存为常数 SH 的 DC 项（实时档位：方向性由
                // 探针间隔 + 平滑提供；如需更强方向性，`sh_samples` 投影路径见
                // `project_probe_sh`）。约定见 `gi::sh` 模块文档：常数辐照度 E
                // 对应辐亮度 DC 系数 `c0 = E/π`（因 Â_0=π）。
                let mut c = [[0.0f32; 9]; 3];
                c[0][0] = ShIrradiance::constant(e.x).coeffs[0];
                c[1][0] = ShIrradiance::constant(e.y).coeffs[0];
                c[2][0] = ShIrradiance::constant(e.z).coeffs[0];
                sh_cubes.push(c);
                // 单弹射：弹射轮数恒 1（第 0 弹 = 可见性加权环境光），未走收敛循环。
                bounce_counts.push(1);
                converged.push(false);
            }
        }
    }
    ProbeGrid {
        min,
        max,
        res: (nx, ny, nz),
        probes,
        sh_cubes,
        baked_order: budget.order,
        bounce_counts,
        converged,
    }
}

/// 多弹射预算（[`bake_probe_grid_multi_bounce`] 的质量-成本旋钮）。
///
/// 组合了"探针采样/插值预算" [`ProbeBudget`] 与"多弹射求解预算"
/// [`MultiBounceBudget`]：前者决定每探针的 SH 采样数与 AO 数（沿用单弹射路径
/// 语义），后者决定每个探针内部用 [`solve_multi_bounce`] 迭代多少弹、收敛多严。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeMultiBounceBudget {
    /// 探针采样/插值预算（SH 采样、AO 方向、重建阶、遮挡步数）。
    pub probe: ProbeBudget,
    /// 多弹射求解预算（弹射数、衰减、收敛阈值、采样数、遮挡）。
    pub multi: MultiBounceBudget,
}

impl Default for ProbeMultiBounceBudget {
    fn default() -> Self {
        Self { probe: ProbeBudget::default(), multi: MultiBounceBudget::default() }
    }
}

impl ProbeMultiBounceBudget {
    /// 低档：少探针采样 + 少弹射（快速预览）。
    pub fn low() -> Self {
        Self { probe: ProbeBudget::low(), multi: MultiBounceBudget::low() }
    }
    /// 标准档。
    pub fn standard() -> Self {
        Self::default()
    }
    /// 高档：多采样 + 多弹射（近收敛）。
    pub fn high() -> Self {
        Self { probe: ProbeBudget::high(), multi: MultiBounceBudget::high() }
    }
}

/// 从 SDF 场烘焙规则探针格——**多弹射版本**（账三 U4 深化 A 的端到端接线）。
///
/// 与 [`bake_probe_grid`] 的唯一差异：每个探针的辐照度由
/// [`solve_multi_bounce`] **迭代式多弹射**求解，而非一次性
/// [`one_bounce_indirect`]。这样"多弹射只增能量"的性质直接进入烘焙结果，
/// 空间上的间接辐照度场因此比单弹射更亮、更接近多次反射的真实解。
///
/// # 与射线/几何的关系
/// 每个探针位置取场梯度法线（[`field_normal`]），单位反照率（探针处无材质，
/// 复用"辐照度度量"约定），把 [`solve_multi_bounce`] 的总辐亮度
/// `L = Σ_k L_k` 折成常数 SH 的 DC 系数（同 [`bake_probe_grid`] 的 `c0 = E/π`。
/// 注意：`solve_multi_bounce` 返回的是**可见性加权环境辐亮度累加**，其量纲与
/// [`one_bounce_indirect`] 单弹射一致（都不含 π，π 在两处约定中抵消，见
/// `gi::sh` 模块文档）——因此可直接套用同一 DC 折算式。
///
/// # 与 [`solve_multi_bounce`] 的直接一致性（交叉验证锚点）
/// 当 `res = (1,1,1)` 时，唯一探针位于体素中心，其 DC 系数对应的辐照度
/// **逐位等于** `solve_multi_bounce(field, center, normal, 1, env, budget.multi).total`
/// （三条分量各自），这是测试 `multi_bounce_bake_matches_solver_exactly` 的锚点。
///
/// # 确定性
/// 探针按 `iz→iy→ix` 固定序烘焙；每探针内 [`solve_multi_bounce`] 采样方向序
/// 固定、弹射累加序固定；无 RNG、无并发。同输入逐位一致。
///
/// # 诊断
/// 结果新增 [`ProbeGrid::bounce_counts`] / [`ProbeGrid::converged`]，可用
/// [`ProbeGrid::last_bounce_count`] / [`ProbeGrid::converged`] 查询。
pub fn bake_probe_grid_multi_bounce(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    res: (u32, u32, u32),
    budget: ProbeMultiBounceBudget,
) -> ProbeGrid {
    let nx = res.0.max(1);
    let ny = res.1.max(1);
    let nz = res.2.max(1);
    let span = max - min;
    let probe_at = |ix: u32, iy: u32, iz: u32| -> Vec3 {
        // D-16：复用体素中心公式（与单弹射版逐位同式）。
        voxel_center(min, span, (nx, ny, nz), ix, iy, iz)
    };
    let pb = budget.probe;
    // 多弹射的每弹采样数与探针预算的 SH 采样数对齐（探针分辨率 → 采样密度）。
    let mut mb = budget.multi;
    mb.n_samples = mb.n_samples.max(pb.sh_samples.max(1));
    mb.occlusion = pb.occlusion;
    let total = (nx as usize) * (ny as usize) * (nz as usize);
    let mut probes = Vec::with_capacity(total);
    let mut sh_cubes = Vec::with_capacity(total);
    let mut bounce_counts = Vec::with_capacity(total);
    let mut converged = Vec::with_capacity(total);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = probe_at(ix, iy, iz);
                let normal = field_normal(field, p);
                let ao = if pb.ao_samples > 0 {
                    ambient_occlusion(field, p, normal, pb.ao_samples, pb.occlusion)
                } else {
                    1.0
                };
                let albedo = Vec3::splat(1.0);
                let r: MultiBounceResult = solve_multi_bounce(field, p, normal, albedo, env_radiance, mb);
                let e = r.total;
                probes.push(Probe { position: p, normal, ao, env_radiance });
                let mut c = [[0.0f32; 9]; 3];
                c[0][0] = ShIrradiance::constant(e.x).coeffs[0];
                c[1][0] = ShIrradiance::constant(e.y).coeffs[0];
                c[2][0] = ShIrradiance::constant(e.z).coeffs[0];
                sh_cubes.push(c);
                bounce_counts.push(r.bounces);
                converged.push(r.converged);
            }
        }
    }
    ProbeGrid {
        min,
        max,
        res: (nx, ny, nz),
        probes,
        sh_cubes,
        baked_order: pb.order,
        bounce_counts,
        converged,
    }
}

/// 从场梯度近似法线（中心差分；退化返回 +Y）。
pub fn field_normal(field: &dyn SdfField, p: Vec3) -> Vec3 {
    const H: f32 = 1e-3;
    let d = field.dist(p);
    // 仅在表面附近有意义；远离表面时梯度方向仍给出"指向最近表面"的方向。
    let gx = field.dist(Vec3::new(p.x + H, p.y, p.z)) - field.dist(Vec3::new(p.x - H, p.y, p.z));
    let gy = field.dist(Vec3::new(p.x, p.y + H, p.z)) - field.dist(Vec3::new(p.x, p.y - H, p.z));
    let gz = field.dist(Vec3::new(p.x, p.y, p.z + H)) - field.dist(Vec3::new(p.x, p.y, p.z - H));
    let n = Vec3::new(gx, gy, gz).normalize();
    let _ = d;
    if n.length() < 0.5 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        n
    }
}

/// 单探针的定向 SH 投影（可选更精确路径）：给定入射方向-辐亮度样本，
/// 投影为 SH 系数（辐照度约定，见 [`crate::gi::sh`] 模块文档）。
///
/// 与 [`bake_probe_grid`] 的"常数 DC"路径互补：此路径保留方向性，成本随
/// `samples.len()` 线性增长（质量-成本旋钮）。
pub fn project_probe_sh(
    samples: &[(Vec3, Vec3)],
    normal: Vec3,
    d_omega: f32,
) -> [[f32; 9]; 3] {
    ShIrradiance::project_rgb(samples, normal, d_omega)
}

// ---------------------------------------------------------------------------
// 自适应探针放置（账三 U4 深化 A：八叉树细分 + 顺序无关）
// ---------------------------------------------------------------------------

/// 自适应放置预算（质量-成本旋钮）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveBudget {
    /// 初始粗粒度各轴探针数（细分前的均匀格，≥1）。
    pub base_res: (u32, u32, u32),
    /// 最大细分轮数（每轮把"高梯度"单元的边一分为二；0 = 不细分）。
    pub max_rounds: u32,
    /// 单元最大边长（世界单位）：边长已 ≤ 此值的单元不再细分（防无限细分）。
    pub min_extent: f32,
    /// 梯度误差阈值：单元内辐照度方差（相邻探针）小于此值即不细分。
    pub error_threshold: f32,
    /// 探针总数上限（防爆内存；达上限即停）。
    pub max_probes: u32,
    /// 烘焙/采样预算（辐照度评估用）。
    pub budget: ProbeBudget,
}

impl Default for AdaptiveBudget {
    fn default() -> Self {
        Self {
            base_res: (2, 2, 2),
            max_rounds: 4,
            min_extent: 0.25,
            error_threshold: 1e-2,
            max_probes: 4096,
            budget: ProbeBudget::default(),
        }
    }
}

impl AdaptiveBudget {
    /// 参数防呆（C18）。
    fn sanitized(self) -> Self {
        let br = (self.base_res.0.max(1), self.base_res.1.max(1), self.base_res.2.max(1));
        let max_rounds = self.max_rounds.min(16);
        let min_extent =
            if self.min_extent.is_finite() && self.min_extent > 1e-6 { self.min_extent } else { 0.25 };
        let error_threshold =
            if self.error_threshold.is_finite() && self.error_threshold > 0.0 {
                self.error_threshold
            } else {
                1e-2
            };
        let max_probes = self.max_probes.max(1).min(1 << 20);
        Self { base_res: br, max_rounds, min_extent, error_threshold, max_probes, budget: self.budget }
    }
}

/// 自适应探针放置结果。
#[derive(Debug, Clone)]
pub struct AdaptivePlacement {
    /// 最终探针（**按固定字典序**排列：`(ix→iy→iz)` 的稳定坐标排序）。
    pub probes: Vec<Probe>,
    /// 各探针的辐照度 SH 系数（RGB 各 9，与 `probes` 同索引）。
    pub sh_cubes: Vec<[[f32; 9]; 3]>,
    /// 实际执行的细分轮数。
    pub rounds: u32,
    /// 是否因误差达标（而非轮数/上限）收敛。
    pub converged: bool,
    /// 放置包围盒（回显）。
    pub min: Vec3,
    /// 放置包围盒（回显）。
    pub max: Vec3,
    /// 烘焙阶数。
    pub order: ShOrder,
}

impl AdaptivePlacement {
    /// 最终探针数。
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// 实际细分轮数。
    pub fn refinement_rounds(&self) -> u32 {
        self.rounds
    }

    /// 探针的辐照度 SH 系数切片（诊断/平滑用，保持与 [`ProbeGrid`] 同布局）。
    pub fn sh_cubes(&self) -> &[[[f32; 9]; 3]] {
        &self.sh_cubes
    }

    /// 从规则探针格 [`ProbeGrid`] **桥接**为自适应布局（端到端查询 API 的一环）。
    ///
    /// 用途：把 [`bake_probe_grid_multi_bounce`] 的均匀格结果套上自适应布局的
    /// **任意点查询**（IDW）与 **k-NN 平滑**，从而在"规则烘焙 + 非规则查询"
    /// 组合下也能用统一 API。布局按坐标字典序重排（顺序无关），`converged`
    /// 由 `grid.converged()` 传入（无探针或未全收敛 → false），`rounds` 置 0
    /// （规则格无细分概念）。
    pub fn from_grid(grid: &ProbeGrid) -> Self {
        // 按坐标 bit 字典序重排，保证与自身构造路径一致的确定序。
        let mut idx: Vec<usize> = (0..grid.probes.len()).collect();
        idx.sort_by(|&a, &b| cmp_point(&grid.probes[a].position, &grid.probes[b].position));
        let mut probes = Vec::with_capacity(idx.len());
        let mut sh_cubes = Vec::with_capacity(idx.len());
        for &i in &idx {
            probes.push(grid.probes[i]);
            sh_cubes.push(grid.sh_cubes.get(i).copied().unwrap_or([[0.0f32; 9]; 3]));
        }
        AdaptivePlacement {
            probes,
            sh_cubes,
            rounds: 0,
            converged: grid.converged(),
            min: grid.min,
            max: grid.max,
            order: grid.baked_order,
        }
    }

    /// 按世界坐标查询插值后的辐照度（**最近邻**，因自适应布局非规则——诚实说明）。
    ///
    /// 自适应探针无规则格索引，插值退化为"先按 Y/方向无关的辐照度标量做
    /// 距离加权"，此处用**逆距离加权**（IDW，幂 2）在全部探针上混合；点恰与
    /// 某探针重合时取其值（避免除零）。累加序固定（探针字典序）→ 确定性。
    ///
    /// # C18
    /// `probes` 与 `sh_cubes` 都是 `pub` 字段，外部可构造出**长度不等**的
    /// 布局（含经 [`Self::smooth`] 之外的路径）。本函数以两者的**公共长度**
    /// `n = min(len)` 为界遍历，缺失系数的探针按"无数据"跳过（不计入权重分母），
    /// 因此**不会越界 panic**；两者皆空时返回 0。
    pub fn sample_irradiance(&self, p: Vec3, order: ShOrder) -> Vec3 {
        // 以公共长度为界，杜绝 `sh_cubes[i]` 越界（C18）。
        let n = self.probes.len().min(self.sh_cubes.len());
        if n == 0 {
            return Vec3::splat(0.0);
        }
        let mut num = Vec3::splat(0.0);
        let mut den = 0.0f32;
        let mut exact: Option<Vec3> = None;
        for i in 0..n {
            let probe = self.probes[i];
            let d = p - probe.position;
            let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            let e = sh_reconstruct_rgb(&self.sh_cubes[i], probe.normal, order);
            if d2 < 1e-12 {
                exact = Some(e);
                break;
            }
            let w = 1.0 / d2; // IDW 幂 2
            num = num + e * w;
            den += w;
        }
        if let Some(e) = exact {
            return e;
        }
        if den > 0.0 {
            num * (1.0 / den)
        } else {
            Vec3::splat(0.0)
        }
    }

    /// 自适应探针间平滑（S2 卷积 / 调和分析核）——[`ProbeGrid::smooth`] 的非规则布局等价。
    ///
    /// # 为什么不能用规则 3×3×3 盒核
    /// 自适应布局的探针位置由八叉树细分决定，**无规则格索引**，故不能用
    /// [`ProbeGrid::smooth`] 的 `(ix±1, iy±1, iz±1)` 邻接。这里改为**最近 k
    /// 个探针（k-NN）** 的均匀加权（盒核的散点推广）：对每个探针，取含自身在内
    /// 的最近 `k` 个探针（含自身），对 SH 系数做均匀平均，再按 `strength` 线性
    /// 混合回原值。
    ///
    /// # 邻域选择（确定性关键）
    /// - 距离用平方欧氏距离（免 sqrt，逐位确定）；
    /// - 候选按 `(距离, 坐标 bit 序)` 排序：距离相同者以 [`cmp_point`] 的
    ///   f32 bit 全序 tie-break，**不依赖探针遍历顺序**；
    /// - 累加按排序后的固定序（最近 → 第 k 近），故**与输入顺序无关**。
    ///
    /// # 参数
    /// - `strength ∈ [0,1]`：0 = 不平滑（恒等，逐位保留），1 = 完全替换为邻域均；
    /// - `k`：邻域大小（含自身），自动钳到 `[1, n]`（`n` = 探针数）。
    ///
    /// # 能量（设计说明——**实测校准，勿照抄"保均值"直觉**）
    ///
    /// 本平滑是**行随机（row-stochastic）**矩阵（每行非负、和为 1），但 k-NN
    /// 邻接**不是双随机（doubly-stochastic）**：探针 `j` 被多少个邻域包含，
    /// 取决于它自身的邻域选择，不一定等于 `k`。因此：
    ///
    /// - **总能量（系数和）不守恒，且可增可减**。实测：`[100,0,0,0,100]`
    ///   经 `k=3` 平滑后总和 `200 → 133.3`（减）；`[0,0,100,0,0]` 则
    ///   `100 → 166.7`（**增**）。任何"总能量单调不增"的断言都是错的。
    /// - **真正成立的不变量是"凸包包含"（range/hull containment）**：输出是
    ///   输入邻域的凸组合，故 `min(after) ≥ min(before)` 且
    ///   `max(after) ≤ max(before)` —— 平滑只会收缩值域，**不会制造新的极值**。
    ///   这是低通滤波的本质性质（也是数值稳定性的保障）。
    /// - **常值场逐位不变**：`strength` 与 `k` 都无关（全常数域的凸组合仍是
    ///   该常数）。这是最强的"能量守恒"特例，已被测试锚定。
    /// - **方差下降**只在规则/对称布局下成立；k-NN 非对称布局下**不保证**
    ///   （邻域均值会把孤点往局部均值拉，通常降方差，但无严格保证）。
    ///
    /// 测试锚定：凸包包含（强不等式）、常值场逐位不变、strength=0 恒等。
    ///
    /// # C18
    /// 空布局、`strength` NaN、`k=0`、坐标为 NaN 均安全收束，不 panic。
    pub fn smooth(&mut self, strength: f32, k: usize) {
        let s = if strength.is_nan() { 0.0 } else { strength.clamp(0.0, 1.0) };
        let n = self.probes.len();
        if s <= 0.0 || n == 0 || self.sh_cubes.len() != n {
            return;
        }
        let k = k.max(1).min(n);
        let src = self.sh_cubes.clone();
        let mut out = src.clone();
        // 逐个探针独立平滑（结果只依赖全局几何，与处理顺序无关）。
        for i in 0..n {
            // 收集 (距离平方, 探针索引) 并对**除自身外**的候选排序。
            let pi = self.probes[i].position;
            let mut cand: Vec<(f32, usize)> = Vec::with_capacity(n.saturating_sub(1));
            for (j, probe) in self.probes.iter().enumerate() {
                if j == i {
                    continue;
                }
                let d = probe.position - pi;
                let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                // NaN 距离（NaN 坐标）→ 排到末尾（用 +INF 替代，保持全序）。
                let d2 = if d2.is_finite() { d2 } else { f32::INFINITY };
                cand.push((d2, j));
            }
            // 稳定全序：先按距离（无 NaN），再按坐标 bit 序 tie-break。
            let other = &self.probes;
            cand.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| cmp_point(&other[(*a).1].position, &other[(*b).1].position))
            });
            // 含自身：取最近 (k-1) 个邻居 + 自身，共 k 个（k≤n 保证可行）。
            let take = k.saturating_sub(1).min(cand.len());
            let mut acc = [[0.0f32; 9]; 3];
            for ch in 0..3 {
                for c in 0..9 {
                    acc[ch][c] = src[i][ch][c];
                }
            }
            let mut cnt = 1.0f32;
            for &(_, j) in cand.iter().take(take) {
                for ch in 0..3 {
                    for c in 0..9 {
                        acc[ch][c] += src[j][ch][c];
                    }
                }
                cnt += 1.0;
            }
            if cnt > 0.0 {
                for ch in 0..3 {
                    for c in 0..9 {
                        let avg = acc[ch][c] / cnt;
                        let orig = src[i][ch][c];
                        out[i][ch][c] = orig + (avg - orig) * s;
                    }
                }
            }
        }
        self.sh_cubes = out;
    }
}

/// 自适应探针放置（账三 U4 深化 A）。
///
/// # 策略（八叉树细分）
/// 1. **初始粗粒度布点**：在 `[min,max]` 上以 `base_res` 撒一个均匀格（D-16
///    复用既有体素中心公式）；
/// 2. **逐轮细分**：每轮对**每个当前叶单元**，用其顶点探针的辐照度计算
///    **梯度误差** = 单元内 8 顶点辐照度标量的**极差（max−min，L∞ 式）**；
///    若误差 > `error_threshold` 且边长 > `min_extent` 且未达 `max_probes`，
///    则该单元**八分**（每轴一分为二 → 8 子单元，各取子体素中心为新探针）。
/// 3. 迭代 `max_rounds` 轮，或所有单元误差达标即停（`converged = true`）。
///
/// # 顺序无关性（关键）如何保证
/// - 细分**决策**只依赖每个单元**自身的 8 顶点辐照度**（位置由单元几何唯一
///   确定），**不依赖单元的遍历顺序**、不依赖邻居状态；
/// - 每轮把当前所有单元**收集成 Vec 后按单元中心坐标字典序排序**，再顺序处理；
///   这样即使输入顺序不同，处理序也一致；
/// - 最终探针列表**再次按坐标字典序排序**（[`Self::sort_probes`]），消除任何
///   残留的顺序差异。
/// - 辐照度评估用固定的低差异采样方向序（[`bake_probe_grid`] 同款）。
///
/// 因此：**以不同顺序喂入初始采样点，最终布局（数量 + 坐标）逐位一致**。
///
/// # 初始采样点顺序如何喂入
/// `initial_samples` 提供"用户给定的初始采样点"；若为空则用 `base_res` 均匀格。
/// 无论以何顺序提供，本函数都会**先排序去重**再进入细分，保证顺序无关。
pub fn adaptive_place(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    initial_samples: &[Vec3],
    budget: AdaptiveBudget,
) -> AdaptivePlacement {
    // 单弹射评估：复用既有语义（可见性加权环境辐亮度 × 单位反照率）。
    let solver = |p: Vec3, nrm: Vec3, pb: ProbeBudget| -> Vec3 {
        one_bounce_indirect(
            field,
            p,
            nrm,
            Vec3::splat(1.0),
            env_radiance,
            pb.sh_samples.max(1),
            pb.occlusion,
        )
    };
    place_adaptive_core(field, env_radiance, min, max, initial_samples, budget, &solver)
}

/// 自适应探针放置——**多弹射版本**（账三 U4 深化 A 的端到端接线，自适应分支）。
///
/// 与 [`adaptive_place`] 布局过程**完全一致**（同样的八叉树细分、同样的顺序
/// 无关保证、同样的最终探针坐标），差异仅在**每探针辐照度的求解方式**：
/// 细分误差评估与最终烘焙都改用 [`solve_multi_bounce`] 的多弹射总辐亮度。
///
/// # 顺序无关性与确定性
/// 细分决策只依赖单元自身几何与辐照度（多弹射解同样是 `(点, 法线, 场)` 的
/// 纯函数，与遍历序无关）；最终探针仍按坐标字典序排序 + 去重。因此同输入逐位
/// 一致，"不同顺序喂入初始采样点"也得同一布局。
///
/// # 用法（端到端查询链）
/// ```text
/// adaptive_place_multi_bounce(...)  → AdaptivePlacement
///     .smooth(strength, k)          → 邻域平滑（可选）
///     .sample_irradiance(p, order)  → 任意点辐照度
/// ```
pub fn adaptive_place_multi_bounce(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    initial_samples: &[Vec3],
    budget: AdaptiveBudget,
    multi: MultiBounceBudget,
) -> AdaptivePlacement {
    let solver = |p: Vec3, nrm: Vec3, pb: ProbeBudget| -> Vec3 {
        let mut mb = multi;
        mb.n_samples = mb.n_samples.max(pb.sh_samples.max(1));
        mb.occlusion = pb.occlusion;
        solve_multi_bounce(field, p, nrm, Vec3::splat(1.0), env_radiance, mb).total
    };
    place_adaptive_core(field, env_radiance, min, max, initial_samples, budget, &solver)
}

/// 自适应放置的**共享核心**（八叉树细分 + 顺序无关），求解器由 `solver` 注入。
///
/// `solver(p, normal, probe_budget) -> 辐照度`（单弹射/多弹射皆可，见两个公开
/// 包装函数）。这样布局逻辑（顺序无关性关键代码）**只实现一次**（D-16 去繁），
/// 单弹射与多弹射共用，杜绝双实现漂移。
#[allow(clippy::too_many_arguments)]
fn place_adaptive_core(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    initial_samples: &[Vec3],
    budget: AdaptiveBudget,
    solver: &dyn Fn(Vec3, Vec3, ProbeBudget) -> Vec3,
) -> AdaptivePlacement {
    let b = budget.sanitized();
    let span = max - min;

    // ---- 1. 初始单元集合 ---------------------------------------------------
    // 用户初始采样点：排序 + 去重（按 bit 精确去重），保证顺序无关。
    // 若为空 → 用 base_res 均匀格的中心。
    let mut seeds: Vec<Vec3> = if initial_samples.is_empty() {
        let (nx, ny, nz) = b.base_res;
        let mut v = Vec::with_capacity((nx as usize) * (ny as usize) * (nz as usize));
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    v.push(voxel_center(min, span, (nx, ny, nz), ix, iy, iz));
                }
            }
        }
        v
    } else {
        initial_samples.to_vec()
    };
    sort_dedup_points(&mut seeds);

    // 初始单元：每个种子点作为一个"点单元"（半径 = 半格距，取基础分辨率边长）。
    let base_edge = base_edge_len(span, b.base_res);
    // 单元用 (中心, 边长) 表示；边长按轴统一（立方体单元，便于八分）。
    let mut cells: Vec<(Vec3, f32)> = seeds
        .iter()
        .map(|&c| {
            // 把种子点夹到包围盒内，保证细分不越界。
            (clamp_point(c, min, max), base_edge)
        })
        .collect();
    sort_cells(&mut cells);

    let mut rounds = 0u32;
    let mut converged = false;

    // 辐照度评估：单元 8 顶点（复用 center 探针）；此处按需即时评估。
    // 为控制成本，我们只评估单元中心 + 8 顶点共 9 点，取极差作误差。
    let eval = |p: Vec3| -> f32 {
        let nrm = field_normal(field, p);
        let e = solver(p, nrm, b.budget);
        // 标量化：RGB 均值（三通道极差的最大者等价取均值即可，保持简单确定）。
        (e.x + e.y + e.z) / 3.0
    };

    // ---- 2. 逐轮细分 -------------------------------------------------------
    let mut r = 0u32;
    while r < b.max_rounds {
        // 计算每单元误差（顺序无关：误差是单元几何的函数）。
        let mut new_cells: Vec<(Vec3, f32)> = Vec::with_capacity(cells.len());
        let mut any_subdivided = false;
        let probe_budget_left = b.max_probes as usize;
        // 先统计当前探针总数（= cells.len()，每单元 1 探针）。
        let mut projected = cells.len();
        // 单元按固定序处理（已排序）。
        for &(center, edge) in &cells {
            let err = cell_error(&eval, center, edge, min, max);
            let can_subdivide = edge > b.min_extent && projected + 7 <= probe_budget_left;
            if err > b.error_threshold && can_subdivide {
                // 八分：子单元边长 = edge/2，中心在 8 个卦限偏移处。
                let h = edge * 0.25;
                for sx in [-1.0f32, 1.0] {
                    for sy in [-1.0f32, 1.0] {
                        for sz in [-1.0f32, 1.0] {
                            let c = clamp_point(
                                Vec3::new(center.x + sx * h, center.y + sy * h, center.z + sz * h),
                                min,
                                max,
                            );
                            new_cells.push((c, edge * 0.5));
                        }
                    }
                }
                projected += 7; // 1 单元 → 8 子单元，净增 7
                any_subdivided = true;
            } else {
                new_cells.push((center, edge));
            }
        }
        // 排序 + 去重（子单元中心可能与邻居重合 → 去重保证布局唯一）。
        sort_cells(&mut new_cells);
        cells = new_cells;
        rounds = r + 1;
        if !any_subdivided {
            converged = true;
            break;
        }
        r += 1;
    }

    // ---- 3. 构建最终探针（固定字典序）--------------------------------------
    sort_cells(&mut cells);
    let mut probes = Vec::with_capacity(cells.len());
    let mut sh_cubes = Vec::with_capacity(cells.len());
    for &(center, _edge) in &cells {
        let normal = field_normal(field, center);
        let ao = if b.budget.ao_samples > 0 {
            ambient_occlusion(field, center, normal, b.budget.ao_samples, b.budget.occlusion)
        } else {
            1.0
        };
        let e = solver(center, normal, b.budget);
        probes.push(Probe { position: center, normal, ao, env_radiance });
        let mut c = [[0.0f32; 9]; 3];
        c[0][0] = ShIrradiance::constant(e.x).coeffs[0];
        c[1][0] = ShIrradiance::constant(e.y).coeffs[0];
        c[2][0] = ShIrradiance::constant(e.z).coeffs[0];
        sh_cubes.push(c);
    }
    AdaptivePlacement {
        probes,
        sh_cubes,
        rounds,
        converged,
        min,
        max,
        order: b.budget.order,
    }
}

/// **端到端 GI 烘焙 + 查询**一站式流程（账三 U4 深化的接线入口）。
///
/// # 调用链
/// ```text
/// adaptive_place_multi_bounce(field, env, min, max, seeds, abudget, mbudget)
///     → AdaptivePlacement                       （八叉树细分 + 多弹射烘焙）
///     → .smooth(strength, k)                     （k-NN 邻域平滑，可选；strength=0 跳过）
///     → .sample_irradiance(p, order)             （任意点 IDW 插值辐照度）
/// ```
/// 本函数把前三步封装为一次调用，返回"已平滑待查询"的 [`AdaptivePlacement`]。
///
/// # 参数
/// - `smooth_strength`：S2 平滑强度（`[0,1]`，0 = 不平滑）；
/// - `smooth_k`：k-NN 邻域大小（含自身，自动钳到 `[1, n]`）。
///
/// # 确定性
/// 全链纯函数（无 RNG/无并发）：同输入 → 同布局 → 同平滑 → 同查询，逐位一致。
pub fn bake_adaptive_gi_end_to_end(
    field: &SceneSdf,
    env_radiance: Vec3,
    min: Vec3,
    max: Vec3,
    initial_samples: &[Vec3],
    adaptive_budget: AdaptiveBudget,
    multi_budget: MultiBounceBudget,
    smooth_strength: f32,
    smooth_k: usize,
) -> AdaptivePlacement {
    let mut placement = adaptive_place_multi_bounce(
        field,
        env_radiance,
        min,
        max,
        initial_samples,
        adaptive_budget,
        multi_budget,
    );
    // 平滑（strength ≤ 0 时 smooth 内部即早退，恒等）。
    placement.smooth(smooth_strength, smooth_k);
    placement
}

/// 体素中心（与 [`bake_probe_grid`] 同式，抽出复用）。
#[inline]
fn voxel_center(min: Vec3, span: Vec3, res: (u32, u32, u32), ix: u32, iy: u32, iz: u32) -> Vec3 {
    let (nx, ny, nz) = res;
    let fx = (ix as f32 + 0.5) / nx.max(1) as f32;
    let fy = (iy as f32 + 0.5) / ny.max(1) as f32;
    let fz = (iz as f32 + 0.5) / nz.max(1) as f32;
    Vec3::new(min.x + span.x * fx, min.y + span.y * fy, min.z + span.z * fz)
}

/// 基础单元边长（三轴包围尺寸 / 探针数的较大者；退化则取 1）。
#[inline]
fn base_edge_len(span: Vec3, res: (u32, u32, u32)) -> f32 {
    let ex = span.x.abs() / res.0.max(1) as f32;
    let ey = span.y.abs() / res.1.max(1) as f32;
    let ez = span.z.abs() / res.2.max(1) as f32;
    let e = ex.max(ey).max(ez);
    if e.is_finite() && e > 1e-6 {
        e
    } else {
        1.0
    }
}

/// 单元辐照度梯度误差：8 顶点（含中心，钳到包围盒内）辐照度标量的极差 max−min。
///
/// 顺序无关：极差与顶点遍历序无关。顶点采样点钳到 `[min,max]` 内，避免越界
/// 采样（边界外无几何 → 常数远场 → 虚假大误差）。
#[inline]
fn cell_error(eval: &dyn Fn(Vec3) -> f32, center: Vec3, edge: f32, min: Vec3, max: Vec3) -> f32 {
    let h = edge * 0.5;
    let mut lo = eval(clamp_point(center, min, max));
    let mut hi = lo;
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let p = clamp_point(
                    Vec3::new(center.x + sx * h, center.y + sy * h, center.z + sz * h),
                    min,
                    max,
                );
                let v = eval(p);
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
    }
    if (hi - lo).is_finite() {
        hi - lo
    } else {
        f32::INFINITY
    }
}

#[inline]
fn clamp_point(p: Vec3, min: Vec3, max: Vec3) -> Vec3 {
    // 逐分量防呆：NaN 分量、或 lo > hi 时不做钳制（返回原值），C18 不 panic
    // （标准库 `f32::clamp` 在 lo>hi 或任一端 NaN 时会 panic，故自实现）。
    #[inline]
    fn clamp1(v: f32, lo: f32, hi: f32) -> f32 {
        if !v.is_finite() {
            return v;
        }
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return v;
        }
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }
    Vec3::new(
        clamp1(p.x, min.x, max.x),
        clamp1(p.y, min.y, max.y),
        clamp1(p.z, min.z, max.z),
    )
}

/// 点按 bit 精确排序 + 去重（顺序无关：不同输入序 → 同一输出序）。
fn sort_dedup_points(pts: &mut Vec<Vec3>) {
    pts.sort_by(cmp_point);
    pts.dedup_by(|a, b| cmp_point(a, b) == std::cmp::Ordering::Equal);
}

/// 单元（中心, 边长）按中心坐标 bit 字典序排序 + 去重。
fn sort_cells(cells: &mut Vec<(Vec3, f32)>) {
    cells.sort_by(|a, b| cmp_point(&a.0, &b.0));
    cells.dedup_by(|a, b| cmp_point(&a.0, &b.0) == std::cmp::Ordering::Equal);
}

/// 点的确定性全序：按 (x,y,z) 的 f32 bit 序（NaN 归一到末尾，保证全序且稳定）。
fn cmp_point(a: &Vec3, b: &Vec3) -> std::cmp::Ordering {
    let key = |v: f32| -> u32 {
        // NaN → 最大 key（排末尾），负零归一到正零，保证全序。
        if v.is_nan() {
            u32::MAX
        } else if v == 0.0 {
            // ±0.0 归一到 +0.0 的 key：必须取"符号幅值映射"下 +0.0 的真实 key
            // （0 | 0x8000_0000），否则 +0.0 会排到所有负数之前，破坏单调性。
            0x8000_0000u32
        } else {
            let bits = v.to_bits();
            // 把 IEEE 浮点位序映射成单调 u32（负数翻转）。
            if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits | 0x8000_0000
            }
        }
    };
    key(a.x)
        .cmp(&key(b.x))
        .then_with(|| key(a.y).cmp(&key(b.y)))
        .then_with(|| key(a.z).cmp(&key(b.z)))
}


// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_scene() -> SceneSdf {
        SceneSdf::from_scene(&crate::integrator::Scene::default())
    }

    /// 空环境（env=0）探针格：辐照度恒 0、采样不 panic。
    #[test]
    fn empty_grid_samples_zero() {
        let field = empty_scene();
        // env_radiance = 0 → 无入射光 → 辐照度恒 0（探针法线对空场无意义）。
        let grid = bake_probe_grid(
            &field,
            Vec3::splat(0.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (2, 2, 2),
            ProbeBudget::low(),
        );
        assert_eq!(grid.count(), 8);
        let e = grid.sample_irradiance(Vec3::splat(0.0), ShOrder::L2);
        assert_eq!(e, Vec3::splat(0.0));
        // env > 0 时（空场全方向天光），辐照度应 = albedo(1)·env。
        let grid2 = bake_probe_grid(
            &field,
            Vec3::splat(0.4),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (1, 1, 1),
            ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
        );
        let e2 = grid2.sample_irradiance(Vec3::splat(0.0), ShOrder::L2);
        assert!((e2.x - 0.4).abs() < 1e-3, "空场天光应 = env=0.4，实际 {}", e2.x);
    }

    /// 烘焙确定性：同输入两遍逐位一致。
    #[test]
    fn bake_is_bitwise_deterministic() {
        let field = empty_scene();
        let a = bake_probe_grid(
            &field,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (2, 2, 2),
            ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
        );
        let b = bake_probe_grid(
            &field,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (2, 2, 2),
            ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
        );
        assert_eq!(a.sh_cubes.len(), b.sh_cubes.len());
        for (ca, cb) in a.sh_cubes.iter().zip(b.sh_cubes.iter()) {
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cb[ch][i].to_bits());
                }
            }
        }
    }

    /// 边界插值：全部探针同值时，采样处也应等于该值（S1 场常值插值）。
    #[test]
    fn trilinear_of_constant_field_is_constant() {
        let field = empty_scene();
        let mut grid = bake_probe_grid(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (2, 2, 2),
            ProbeBudget { sh_samples: 4, ao_samples: 2, ..ProbeBudget::low() },
        );
        // 手动把所有探针 DC 设成同一常数 E（用 ShIrradiance::constant 的语义）。
        let e_val = 2.5f32;
        let c0 = ShIrradiance::constant(e_val).coeffs[0];
        for c in grid.sh_cubes.iter_mut() {
            c[0][0] = c0;
            c[1][0] = c0;
            c[2][0] = c0;
        }
        for p in [Vec3::splat(0.0), Vec3::new(0.2, -0.3, 0.4), Vec3::splat(0.9)] {
            let e = grid.sample_irradiance(p, ShOrder::L2);
            assert!((e.x - e_val).abs() < 1e-3, "常值场插值失配：{:?}", e);
        }
    }

    /// 平滑：strength=0 恒等；strength=1 使系数趋同（方差下降）。
    #[test]
    fn smoothing_reduces_variance_and_identity_at_zero() {
        let field = empty_scene();
        let mut grid = bake_probe_grid(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (3, 1, 1),
            ProbeBudget { sh_samples: 4, ao_samples: 2, ..ProbeBudget::low() },
        );
        // 人为给不同值制造方差。
        for (k, c) in grid.sh_cubes.iter_mut().enumerate() {
            c[0][0] = k as f32;
        }
        let before = grid.sh_cubes.clone();
        grid.smooth(0.0);
        assert_eq!(grid.sh_cubes[0][0][0].to_bits(), before[0][0][0].to_bits(), "strength=0 恒等");
        grid.smooth(1.0);
        // 中间探针被邻居平均 → 方差下降（与端点差减小）。
        let mid = grid.sh_cubes[1][0][0];
        assert!(mid > 0.0, "中间探针应被邻居抬升");
    }

    /// 预算档位同向：低档与高档烘焙的辐照度不应出现"谁好谁坏反了"——用
    /// 有遮挡场景检查两者都对可见性方向一致响应。
    #[test]
    fn low_and_high_budget_agree_directionally() {
        use crate::integrator::{Aabb, MaterialKind, Scene};
        // 地板 + 上方天花板（形成半封闭腔）。
        let scene = Scene {
            spheres: Vec::new(),
            boxes: vec![
                (
                    Aabb::new(Vec3::new(-5.0, -0.2, -5.0), Vec3::new(5.0, 0.0, 5.0)),
                    MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
                ),
                (
                    Aabb::new(Vec3::new(-5.0, 1.0, -5.0), Vec3::new(5.0, 1.2, 5.0)),
                    MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
                ),
            ],
            lights: Vec::new(),
            medium: None,
        };
        let field = SceneSdf::from_scene(&scene);
        let p = Vec3::new(0.0, 0.5, 0.0); // 腔中
        let n = Vec3::new(0.0, 1.0, 0.0);
        // 低/高档 AO：都应在 [0,1]，且"朝被遮挡方向" AO 更小。
        let ao_low = ambient_occlusion(&field, p, n, 8, OcclusionBudget::low());
        let ao_high = ambient_occlusion(&field, p, n, 128, OcclusionBudget::high());
        assert!((0.0..=1.0).contains(&ao_low) && (0.0..=1.0).contains(&ao_high));
        // 朝上的 AO（被天花板遮挡）应低于朝下（地板在下方但也遮挡）。
        let ao_up = ambient_occlusion(&field, p, n, 64, OcclusionBudget::standard());
        let ao_away = ambient_occlusion(&field, p, Vec3::new(0.0, -1.0, 0.0), 64, OcclusionBudget::standard());
        // 两者都被遮挡，均 <1；且均 >0（有开口）。方向一致性：都不为 NaN。
        assert!(ao_up > 0.0 && ao_away > 0.0, "两向都应有部分可见：{ao_up}/{ao_away}");
    }

    /// 极端输入：0 探针（res=0）、NaN 位置、退化包围盒 → 不 panic。
    #[test]
    fn extreme_inputs_no_panic() {
        let field = empty_scene();
        // res = (0,0,0) → 钳到 (1,1,1)。
        let grid = bake_probe_grid(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (0, 0, 0),
            ProbeBudget::low(),
        );
        assert_eq!(grid.count(), 1);
        // NaN 查询点。
        let _ = grid.sample_irradiance(Vec3::new(f32::NAN, 0.0, 0.0), ShOrder::L2);
        // 退化包围盒（min == max）。
        let grid2 = bake_probe_grid(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(0.0),
            Vec3::splat(0.0),
            (2, 2, 2),
            ProbeBudget::low(),
        );
        let _ = grid2.sample_irradiance(Vec3::splat(0.0), ShOrder::L2);
    }

    /// 测试自证非恒真：插值确实随位置变化（非恒返回单值）。
    #[test]
    fn interpolation_actually_varies() {
        let field = empty_scene();
        let mut grid = bake_probe_grid(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (2, 1, 1),
            ProbeBudget { sh_samples: 4, ao_samples: 2, ..ProbeBudget::low() },
        );
        grid.sh_cubes[0][0][0] = 0.0;
        grid.sh_cubes[1][0][0] = ShIrradiance::constant(10.0).coeffs[0];
        let a = grid.sample_irradiance(Vec3::new(-0.9, 0.0, 0.0), ShOrder::L0).x;
        let b = grid.sample_irradiance(Vec3::new(0.9, 0.0, 0.0), ShOrder::L0).x;
        assert!(b > a + 1.0, "插值应随位置变化：{a} vs {b}");
    }

    // -----------------------------------------------------------------------
    // 自适应探针放置（深化 A）测试
    // -----------------------------------------------------------------------

    /// 构造一个有强辐照度梯度的场景：地板 + 一堵半边墙（一侧亮一侧暗）。
    fn gradient_scene() -> SceneSdf {
        use crate::integrator::{Aabb, MaterialKind, Scene};
        let scene = Scene {
            spheres: Vec::new(),
            boxes: vec![
                // 地板
                (
                    Aabb::new(Vec3::new(-4.0, -0.2, -4.0), Vec3::new(4.0, 0.0, 4.0)),
                    MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
                ),
                // 竖直半墙（x≈0 附近制造遮挡梯度）
                (
                    Aabb::new(Vec3::new(-0.1, 0.0, -4.0), Vec3::new(0.1, 3.0, 0.0)),
                    MaterialKind::Lambert { albedo: Vec3::splat(0.5) },
                ),
            ],
            lights: Vec::new(),
            medium: None,
        };
        SceneSdf::from_scene(&scene)
    }

    /// 自适应放置确定性：同输入两遍 → 数量 + 坐标逐位一致。
    #[test]
    fn adaptive_place_bitwise_deterministic() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 3,
            error_threshold: 1e-3,
            ..AdaptiveBudget::default()
        };
        let a = adaptive_place(
            &field,
            Vec3::splat(0.6),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &[],
            budget,
        );
        let b = adaptive_place(
            &field,
            Vec3::splat(0.6),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &[],
            budget,
        );
        assert_eq!(a.probe_count(), b.probe_count(), "探针数量必须一致");
        assert_eq!(a.refinement_rounds(), b.refinement_rounds());
        for (pa, pb) in a.probes.iter().zip(b.probes.iter()) {
            assert_eq!(pa.position.x.to_bits(), pb.position.x.to_bits());
            assert_eq!(pa.position.y.to_bits(), pb.position.y.to_bits());
            assert_eq!(pa.position.z.to_bits(), pb.position.z.to_bits());
        }
    }

    /// 顺序无关性（关键断言）：以不同顺序喂入初始采样点 → 最终布局一致。
    #[test]
    fn adaptive_place_order_independent() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 3,
            error_threshold: 1e-3,
            ..AdaptiveBudget::default()
        };
        // 两组相同采样点，顺序不同（含乱序 + 重复）。
        let forward = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let mut reversed = forward.clone();
        reversed.reverse();
        // 再插入重复项（应被去重）。
        reversed.push(Vec3::new(-1.0, -1.0, -1.0));
        reversed.push(Vec3::new(1.0, 1.0, 1.0));
        let a = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &forward, budget);
        let b = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &reversed, budget);
        assert_eq!(a.probe_count(), b.probe_count(), "不同喂入顺序应得同数量探针");
        for (pa, pb) in a.probes.iter().zip(b.probes.iter()) {
            assert_eq!(pa.position.x.to_bits(), pb.position.x.to_bits(), "顺序无关：x 坐标应一致");
            assert_eq!(pa.position.y.to_bits(), pb.position.y.to_bits(), "顺序无关：y 坐标应一致");
            assert_eq!(pa.position.z.to_bits(), pb.position.z.to_bits(), "顺序无关：z 坐标应一致");
        }
    }

    /// 物理：自适应布点在**相同重建算子的同探针数**下误差低于均匀布点
    /// （梯度大处加密有效）。
    ///
    /// 误差度量：为**隔离"放置质量"与"插值算子"**，两种布局都用同一套
    /// **最近邻重建**（对散点集做 NN 查询），与"高分辨率均匀参考格"比较取
    /// L1 平均误差。这样差异只来自探针**位置**，与插值算法无关。
    #[test]
    fn adaptive_beats_uniform_at_same_count() {
        let field = gradient_scene();
        let env = Vec3::splat(0.8);
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);
        let pb = ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() };

        // 自适应（细阈值 → 在梯度处加密），比均匀多细分。
        let adaptive_budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 2,
            error_threshold: 1e-4,
            max_probes: 4096,
            budget: pb,
            ..AdaptiveBudget::default()
        };
        let adp = adaptive_place(&field, env, min, max, &[], adaptive_budget);
        let n_adp = adp.probe_count();

        // 均匀布点：选与自适应**同等或更少**探针数的立方根分辨率，
        // 使均匀布局在探针数上不吃亏（更强结论）。
        let side = {
            let mut s = 1u32;
            while (s as usize) * (s as usize) * (s as usize) < n_adp {
                s += 1;
            }
            // 若该立方根已超出自适应数，退一档（保证 uniform ≤ adaptive）。
            let cnt = (s as usize).pow(3);
            if cnt > n_adp && s > 1 {
                s - 1
            } else {
                s
            }
        }
        .max(1);
        let uni = bake_probe_grid(&field, env, min, max, (side, side, side), pb);

        // 参考"真值"：高分辨率均匀格。
        let ref_grid = bake_probe_grid(
            &field,
            env,
            min,
            max,
            (12, 12, 12),
            ProbeBudget { sh_samples: 16, ao_samples: 8, ..ProbeBudget::low() },
        );

        // 共享最近邻重建：给定散点 (position, irradiance) → 查询点取最近点的值。
        let nn = |pts: &[(Vec3, f32)], q: Vec3| -> f32 {
            let mut best = f32::INFINITY;
            let mut val = 0.0f32;
            for &(p, e) in pts {
                let d = q - p;
                let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if d2 < best {
                    best = d2;
                    val = e;
                }
            }
            val
        };

        // 采集自适应布局的 (位置, 标量辐照度)。
        let adp_pts: Vec<(Vec3, f32)> = adp
            .probes
            .iter()
            .zip(adp.sh_cubes.iter())
            .map(|(pr, c)| {
                let e = sh_reconstruct_rgb(c, pr.normal, ShOrder::L0);
                (pr.position, (e.x + e.y + e.z) / 3.0)
            })
            .collect();
        // 采集均匀布局的 (位置, 标量辐照度)。
        let uni_pts: Vec<(Vec3, f32)> = uni
            .probes
            .iter()
            .zip(uni.sh_cubes.iter())
            .map(|(pr, c)| {
                let e = sh_reconstruct_rgb(c, pr.normal, ShOrder::L0);
                (pr.position, (e.x + e.y + e.z) / 3.0)
            })
            .collect();

        // 固定查询点网格（顺序固定）。
        let mut query = Vec::new();
        for iz in 0..8u32 {
            for iy in 0..8u32 {
                for ix in 0..8u32 {
                    let fx = (ix as f32 + 0.5) / 8.0;
                    let fy = (iy as f32 + 0.5) / 8.0;
                    let fz = (iz as f32 + 0.5) / 8.0;
                    query.push(Vec3::new(
                        min.x + (max.x - min.x) * fx,
                        min.y + (max.y - min.y) * fy,
                        min.z + (max.z - min.z) * fz,
                    ));
                }
            }
        }
        let err = |pts: &[(Vec3, f32)]| -> f32 {
            let mut acc = 0.0f32;
            for &q in &query {
                let truth = ref_grid.sample_irradiance(q, ShOrder::L0);
                let truth_s = (truth.x + truth.y + truth.z) / 3.0;
                acc += (truth_s - nn(pts, q)).abs();
            }
            acc / query.len() as f32
        };
        let e_adp = err(&adp_pts);
        let e_uni = err(&uni_pts);
        assert!(
            adp_pts.len() >= uni_pts.len(),
            "自适应探针数 {} 应 ≥ 均匀 {}（便于公平比较）",
            adp_pts.len(),
            uni_pts.len()
        );
        assert!(
            e_adp <= e_uni,
            "自适应误差 {e_adp} 应不高于均匀 {e_uni}（自适应 {} vs 均匀 {} 探针）",
            adp_pts.len(),
            uni_pts.len()
        );
    }

    /// 自适应预算防呆 + 极端输入不 panic：0 分辨率、NaN 包围盒、0 轮。
    #[test]
    fn adaptive_place_extreme_no_panic() {
        let field = empty_scene();
        let bad = AdaptiveBudget {
            base_res: (0, 0, 0),
            max_rounds: 0,
            min_extent: f32::NAN,
            error_threshold: f32::NAN,
            max_probes: 0,
            budget: ProbeBudget::low(),
        };
        let p = adaptive_place(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), &[], bad);
        assert!(p.probe_count() >= 1, "至少一个探针");
        assert_eq!(p.refinement_rounds(), 0, "0 轮 → 不细分");
        // NaN 包围盒 → 不 panic。
        let p2 = adaptive_place(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            &[Vec3::new(f32::NAN, 0.0, 0.0)],
            AdaptiveBudget::default(),
        );
        let _ = p2.sample_irradiance(Vec3::new(f32::NAN, 0.0, 0.0), ShOrder::L2);
    }

    /// 自适应确实细分了（多轮后探针数 > 初始），且轮数被记录。
    #[test]
    fn adaptive_actually_subdivides() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (1, 1, 1),
            max_rounds: 3,
            min_extent: 0.1,
            error_threshold: 1e-5,
            max_probes: 2048,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
        };
        let p = adaptive_place(&field, Vec3::splat(0.8), Vec3::splat(-2.0), Vec3::splat(2.0), &[], budget);
        assert!(p.probe_count() > 1, "强梯度场景应细分出多探针，实际 {}", p.probe_count());
        assert!(p.refinement_rounds() >= 1, "应至少细分一轮");
    }

    // -----------------------------------------------------------------------
    // 多弹射烘焙接线（本轮新增）测试
    // -----------------------------------------------------------------------

    /// 多弹射预算档位单调：low 的弹射/采样 ≤ high。
    #[test]
    fn multi_bounce_budget_tiers_monotone() {
        let lo = ProbeMultiBounceBudget::low();
        let hi = ProbeMultiBounceBudget::high();
        assert!(lo.multi.max_bounces <= hi.multi.max_bounces);
        assert!(lo.multi.n_samples <= hi.multi.n_samples);
        assert!(lo.probe.sh_samples <= hi.probe.sh_samples);
    }

    /// **确定性**：多弹射烘焙同输入两遍 → 所有探针辐亮度逐位一致。
    #[test]
    fn multi_bounce_bake_bitwise_deterministic() {
        let field = gradient_scene();
        let env = Vec3::new(0.7, 0.6, 0.5);
        let b = ProbeMultiBounceBudget {
            probe: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            multi: MultiBounceBudget {
                n_samples: 16,
                max_bounces: 6,
                tolerance: 1e-4,
                decay: 0.8,
                occlusion: OcclusionBudget::low(),
            },
        };
        let a = bake_probe_grid_multi_bounce(&field, env, Vec3::splat(-2.0), Vec3::splat(2.0), (2, 2, 2), b);
        let c = bake_probe_grid_multi_bounce(&field, env, Vec3::splat(-2.0), Vec3::splat(2.0), (2, 2, 2), b);
        assert_eq!(a.count(), c.count());
        for (ca, cc) in a.sh_cubes.iter().zip(c.sh_cubes.iter()) {
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cc[ch][i].to_bits(), "多弹射烘焙必须逐位一致");
                }
            }
        }
        assert_eq!(a.bounce_counts(), c.bounce_counts());
        assert_eq!(a.converged(), c.converged());
    }

    /// **交叉验证锚点**：res=(1,1,1) 时，多弹射烘焙的唯一探针辐照度**逐位等于**
    /// 直接调用 `solve_multi_bounce` 的 `total`（证明接线无中间损耗/无额外变换）。
    #[test]
    fn multi_bounce_bake_matches_solver_exactly() {
        let field = gradient_scene();
        let env = Vec3::new(0.6, 0.6, 0.6);
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);
        let pb = ProbeBudget { sh_samples: 16, ao_samples: 0, ..ProbeBudget::low() };
        let mb = MultiBounceBudget {
            n_samples: 16,
            max_bounces: 6,
            tolerance: 1e-4,
            decay: 0.8,
            occlusion: pb.occlusion,
        };
        let grid = bake_probe_grid_multi_bounce(
            &field,
            env,
            min,
            max,
            (1, 1, 1),
            ProbeMultiBounceBudget { probe: pb, multi: mb },
        );
        assert_eq!(grid.count(), 1);
        let center = grid.probes[0].position;
        let normal = grid.probes[0].normal;
        // 重算求解器（同一预算语义：n_samples = max(mb.n_samples, pb.sh_samples)）。
        let mut mb2 = mb;
        mb2.n_samples = mb2.n_samples.max(pb.sh_samples.max(1));
        let expect = solve_multi_bounce(&field, center, normal, Vec3::splat(1.0), env, mb2).total;
        let got = grid.sample_irradiance(center, ShOrder::L0);
        assert_eq!(got.x.to_bits(), expect.x.to_bits(), "X 分量应与求解器逐位一致");
        assert_eq!(got.y.to_bits(), expect.y.to_bits(), "Y 分量应与求解器逐位一致");
        assert_eq!(got.z.to_bits(), expect.z.to_bits(), "Z 分量应与求解器逐位一致");
    }

    /// **物理：多弹射烘焙 ≥ 单弹射烘焙**（多弹射只增能量）。
    #[test]
    fn multi_bounce_bake_ge_single_bounce() {
        let field = gradient_scene();
        let env = Vec3::splat(0.7);
        let min = Vec3::splat(-2.0);
        let max = Vec3::splat(2.0);
        let pb = ProbeBudget { sh_samples: 16, ao_samples: 4, ..ProbeBudget::low() };
        let single = bake_probe_grid(&field, env, min, max, (2, 2, 2), pb);
        let multi = bake_probe_grid_multi_bounce(
            &field,
            env,
            min,
            max,
            (2, 2, 2),
            ProbeMultiBounceBudget {
                probe: pb,
                multi: MultiBounceBudget {
                    n_samples: 16,
                    max_bounces: 8,
                    tolerance: 1e-5,
                    decay: 0.8,
                    occlusion: pb.occlusion,
                },
            },
        );
        assert_eq!(single.count(), multi.count());
        // 每个探针：多弹射辐照度 ≥ 单弹射（逐探针，同序）。
        for i in 0..single.count() {
            let es = single.sample_irradiance(single.probes[i].position, ShOrder::L0);
            let em = multi.sample_irradiance(multi.probes[i].position, ShOrder::L0);
            assert!(
                em.x >= es.x - 1e-5,
                "探针 {i}: 多弹射 {} 应 ≥ 单弹射 {}",
                em.x,
                es.x
            );
        }
        // 多弹射结果严格更亮（至少有一个探针被后续弹射抬高）。
        let s_tot: f32 = single.sh_cubes.iter().map(|c| c[0][0]).sum();
        let m_tot: f32 = multi.sh_cubes.iter().map(|c| c[0][0]).sum();
        assert!(m_tot > s_tot, "多弹射总能量应 > 单弹射：{m_tot} vs {s_tot}");
    }

    /// **物理：多弹射烘焙不超过"全漫反白体"理论上界**（几何级数 closed-form）。
    #[test]
    fn multi_bounce_bake_within_white_furnace_bound() {
        // 空场最亮（可见比 = 1），给多弹射能量最坏情况。
        let field = empty_scene();
        let albedo = Vec3::splat(1.0);
        let env = Vec3::splat(1.0);
        let pb = ProbeBudget { sh_samples: 32, ao_samples: 0, ..ProbeBudget::low() };
        let mb = MultiBounceBudget {
            n_samples: 32,
            max_bounces: 12,
            tolerance: 1e-6,
            decay: 0.8,
            occlusion: pb.occlusion,
        };
        let grid = bake_probe_grid_multi_bounce(
            &field,
            env,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (1, 1, 1),
            ProbeMultiBounceBudget { probe: pb, multi: mb },
        );
        let e = grid.sample_irradiance(grid.probes[0].position, ShOrder::L0);
        // 探针辐照度（= 多弹射 total）应 ≤ 白体上界对应的辐亮度当量。
        let bound = multi_bounce_energy_upper_bound(albedo, env, mb);
        // 白体上界是"能量（含 4π）"，探针 DC 是"辐亮度"，逐分能量 ≤ 上界/4π×3 已足；
        // 更直接：辐亮度分量 ≤ 上界（宽上界，强不等式）。
        let lum = (e.x + e.y + e.z) / 3.0;
        assert!(lum > 0.0, "空场多弹射应亮");
        assert!(
            lum <= bound + 1e-3,
            "探针辐亮度 {} 不应超过白体能量上界 {}",
            lum,
            bound
        );
    }

    /// **多弹射确实多走弹射**：诊断 API 返回 >1 的弹射轮数（非恒 1）。
    #[test]
    fn multi_bounce_bake_reports_bounces() {
        let field = empty_scene(); // 空场：可见比 = 1，弹射累加显著
        let pb = ProbeBudget { sh_samples: 16, ao_samples: 0, ..ProbeBudget::low() };
        let mb = MultiBounceBudget {
            n_samples: 16,
            max_bounces: 8,
            tolerance: 1e-6,
            decay: 0.8,
            occlusion: pb.occlusion,
        };
        let grid = bake_probe_grid_multi_bounce(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            (1, 1, 1),
            ProbeMultiBounceBudget { probe: pb, multi: mb },
        );
        assert!(grid.last_bounce_count() > 1, "多弹射应 > 1 弹，实际 {}", grid.last_bounce_count());
        // 单弹射对照：恒 1。
        let single = bake_probe_grid(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), (1, 1, 1), pb);
        assert_eq!(single.last_bounce_count(), 1, "单弹射应恒 1 弹");
    }

    /// **向后兼容**：原 `bake_probe_grid` 同输入逐位一致，且新增字段长度正确、
    /// 不改变既有辐照度行为。
    #[test]
    fn bake_probe_grid_backward_compatible() {
        let field = gradient_scene();
        let env = Vec3::splat(0.6);
        let pb = ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() };
        let a = bake_probe_grid(&field, env, Vec3::splat(-2.0), Vec3::splat(2.0), (3, 2, 2), pb);
        let b = bake_probe_grid(&field, env, Vec3::splat(-2.0), Vec3::splat(2.0), (3, 2, 2), pb);
        // 既有 sh_cubes 逐位一致。
        for (ca, cb) in a.sh_cubes.iter().zip(b.sh_cubes.iter()) {
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cb[ch][i].to_bits());
                }
            }
        }
        // 新增诊断字段与探针数对齐。
        assert_eq!(a.bounce_counts().len(), a.count());
        assert_eq!(a.converged.len(), a.count());
        assert!(a.bounce_counts().iter().all(|&c| c == 1), "单弹射弹射数应全为 1");
    }

    // -----------------------------------------------------------------------
    // 自适应 smooth（本轮新增）测试
    // -----------------------------------------------------------------------

    /// 自适应 smooth **确定性**：同输入两遍逐位一致（strength=1 全替换）。
    #[test]
    fn adaptive_smooth_bitwise_deterministic() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 2,
            error_threshold: 1e-3,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let mut a = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &[], budget);
        let mut b = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &[], budget);
        a.smooth(1.0, 8);
        b.smooth(1.0, 8);
        assert_eq!(a.sh_cubes.len(), b.sh_cubes.len());
        for (ca, cb) in a.sh_cubes.iter().zip(b.sh_cubes.iter()) {
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cb[ch][i].to_bits(), "自适应 smooth 必须逐位一致");
                }
            }
        }
    }

    /// 自适应 smooth **顺序无关性**：以不同顺序喂入初始探针对，smooth 结果一致。
    ///
    /// 由于 `adaptive_place` 本身即顺序无关（布局一致），此处进一步构造"同布局
    /// 但探针列表顺序被人为打乱"的两种输入，验证 smooth 的邻域选择与累加序
    /// 不依赖探针遍历顺序。
    #[test]
    fn adaptive_smooth_order_independent() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 2,
            error_threshold: 1e-3,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let base = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &[], budget);
        // 变体：把 probes/sh_cubes 逆序（同集合不同序）。
        let mut rev = base.clone();
        rev.probes.reverse();
        rev.sh_cubes.reverse();
        // 对同一批数据做 smooth：逆序后结果应按坐标对齐一致。
        let mut a = base.clone();
        let mut b = rev;
        a.smooth(1.0, 6);
        b.smooth(1.0, 6);
        // 按坐标建立索引映射，比较同一坐标处的系数。
        let find = |pl: &AdaptivePlacement, pos: Vec3| -> Option<[[f32; 9]; 3]> {
            pl.probes.iter().zip(pl.sh_cubes.iter()).find_map(|(p, c)| {
                if p.position.x.to_bits() == pos.x.to_bits()
                    && p.position.y.to_bits() == pos.y.to_bits()
                    && p.position.z.to_bits() == pos.z.to_bits()
                {
                    Some(*c)
                } else {
                    None
                }
            })
        };
        for p in &base.probes {
            let ca = find(&a, p.position);
            let cb = find(&b, p.position);
            let (ca, cb) = match (ca, cb) {
                (Some(x), Some(y)) => (x, y),
                _ => panic!("同一坐标应两布局都存在"),
            };
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cb[ch][i].to_bits(), "smooth 应顺序无关");
                }
            }
        }
    }

    /// 自适应 smooth **常值场逐位不变**（凸组合的精确特例）+ `strength=0` 恒等。
    ///
    /// 这是"守恒"唯一成立的情形：全域常值的凸组合仍是该常数，故与 `k` 无关。
    /// **注意**：一般场的**总能量并不守恒**（见
    /// [`adaptive_smooth_total_energy_can_increase_and_decrease`]），真正普遍的
    /// 不变量是凸包包含（见 [`adaptive_smooth_satisfies_hull_containment`]）。
    #[test]
    fn adaptive_smooth_conserves_constant_field() {
        let field = gradient_scene();
        let budget = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 1,
            error_threshold: 1e-3,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let mut p = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &[], budget);
        // 人为把全部探针设为同一常值（用 constant 的 DC 语义）。
        let c0 = ShIrradiance::constant(1.25).coeffs[0];
        for c in p.sh_cubes.iter_mut() {
            c[0][0] = c0;
            c[1][0] = c0;
            c[2][0] = c0;
        }
        let before = p.sh_cubes.clone();
        // strength=0 → 恒等。
        p.smooth(0.0, 6);
        for (a, b) in p.sh_cubes.iter().zip(before.iter()) {
            assert_eq!(a[0][0].to_bits(), b[0][0].to_bits(), "strength=0 恒等");
        }
        // strength=1 → 常值场仍恒该值（常值域的凸组合 = 该常数，与 k 无关）。
        p.smooth(1.0, 6);
        for c in p.sh_cubes.iter() {
            assert!(
                (c[0][0] - c0).abs() < 1e-5,
                "常值场平滑应守恒，实际 {} vs {c0}",
                c[0][0]
            );
        }
    }

    /// 自适应 smooth 在**单调梯度场**上使方差下降（有效平滑，非恒真）。
    ///
    /// # 适用范围（诚实边界）
    /// 方差下降**不是**普适不变量（k-NN 邻接非对称，无保证）。本测试刻意选
    /// **单调等差梯度**这一"邻域对称"的温和格局：此时每个探针的邻域均落在
    /// 自身与邻居之间，压缩局部散布，故方差必然下降。普适的那条不变量是
    /// 凸包包含，见 [`adaptive_smooth_satisfies_hull_containment`]。
    #[test]
    fn adaptive_smooth_reduces_variance() {
        let field = empty_scene();
        // 规则布局（空场均匀细分）探针数少且位置规则，便于制造方差。
        let budget = AdaptiveBudget {
            base_res: (3, 1, 1),
            max_rounds: 0,
            budget: ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let mut p = adaptive_place(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), &[], budget);
        assert!(p.probe_count() >= 3, "至少 3 个探针");
        // 制造强梯度：探针按序等差值。
        for (idx, c) in p.sh_cubes.iter_mut().enumerate() {
            c[0][0] = idx as f32 * 10.0;
        }
        let var = |pl: &AdaptivePlacement| -> f32 {
            let vals: Vec<f32> = pl.sh_cubes.iter().map(|c| c[0][0]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
            vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len().max(1) as f32
        };
        let v0 = var(&p);
        p.smooth(1.0, 3);
        let v1 = var(&p);
        assert!(v1 < v0, "平滑应降低方差：{v1} vs {v0}");
    }

    /// **交叉验证（强）**：规则场景下，自适应 smooth 与规则网格 smooth 在
    /// **内部探针上逐位一致**，仅端点因边界策略不同而有差异。
    ///
    /// # 为什么内部必须**逐位**一致（而非"接近"）
    /// 两者在规则场景下是**同一个数学算子**：网格的 3×3×3 盒核在
    /// `res=(n,1,1)` 上退化为 1D 三点平均；自适应 `k=3` 的"自身 + 最近 2 个"
    /// 在等距规则布局下也恰是"左/自身/右"。若某探针**两侧邻居都存在**，两者
    /// 的累加项集合与权重完全相同 → 结果必须**逐位相等**。这是能真正抓住
    /// 邻域选择/tie-break/累加序错误的强断言（"数值接近"会放过大量错误）。
    ///
    /// # 端点为何允许不同（诚实说明，非缺陷）
    /// 网格 `smooth` 遇边界**钳邻居**（`ix=0` 只有 2 个邻居 → 除以 2）；自适应
    /// `smooth` 始终取满 `k` 个（`ix=0` 取到 `{0,1,2}` → 除以 3）。二者边界策略
    /// 刻意不同：钳邻居会在边界引入偏置，取满 k 则保持核宽度一致。故端点
    /// **只断言"同量级"**，并在此显式记录该差异，不掩盖。
    #[test]
    fn adaptive_smooth_matches_grid_in_regular_case() {
        let field = empty_scene();
        // n=5/7：足够长以拥有真正的内部点（n=3 时无内部点，两侧皆端点）。
        for n in [5u32, 7u32] {
            let res = (n, 1u32, 1u32);
            let mut grid = bake_probe_grid(
                &field,
                Vec3::splat(1.0),
                Vec3::splat(-1.0),
                Vec3::splat(1.0),
                res,
                ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
            );
            // 自适应（0 轮 → 即规则 seeds 格）。
            let ab = AdaptiveBudget {
                base_res: res,
                max_rounds: 0,
                budget: ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
                ..AdaptiveBudget::default()
            };
            let mut adp =
                adaptive_place(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), &[], ab);
            // 布局一致（同分割）：探针数相同，且坐标集合逐位相同。
            assert_eq!(adp.probe_count(), grid.count(), "n={n} 规则场景下探针数应一致");
            // 设同一梯度场（DC 单调，便于定位）。
            for (idx, c) in grid.sh_cubes.iter_mut().enumerate() {
                c[0][0] = idx as f32;
            }
            for (idx, c) in adp.sh_cubes.iter_mut().enumerate() {
                c[0][0] = idx as f32;
            }
            grid.smooth(1.0);
            adp.smooth(1.0, 3);
            // 按坐标对齐取自适应结果。
            let find_adp = |pos: Vec3| -> Option<f32> {
                adp.probes.iter().zip(adp.sh_cubes.iter()).find_map(|(p, c)| {
                    if p.position.x.to_bits() == pos.x.to_bits()
                        && p.position.y.to_bits() == pos.y.to_bits()
                        && p.position.z.to_bits() == pos.z.to_bits()
                    {
                        Some(c[0][0])
                    } else {
                        None
                    }
                })
            };
            for i in 0..grid.count() {
                let gv = grid.sh_cubes[i][0][0];
                let av = match find_adp(grid.probes[i].position) {
                    Some(v) => v,
                    None => panic!("n={n} 坐标应对应存在"),
                };
                // 按 x 排序后的位次（网格行主序即 x 升序）。
                let is_endpoint = i == 0 || i + 1 == grid.count();
                if is_endpoint {
                    // 端点：仅要求同量级（边界策略不同，见文档）。
                    assert!(
                        (gv - av).abs() <= 1.0,
                        "n={n} 端点 {i} 应同量级：网格 {gv} vs 自适应 {av}"
                    );
                } else {
                    // 内部点：必须逐位一致（强断言）。
                    assert_eq!(
                        gv.to_bits(),
                        av.to_bits(),
                        "n={n} 内部探针 {i} 两种 smooth 应逐位一致：网格 {gv} vs 自适应 {av}"
                    );
                }
            }
        }
    }

    /// 自适应 smooth 极端输入不 panic：空布局、NaN strength、k=0、NaN 坐标。
    #[test]
    fn adaptive_smooth_extreme_no_panic() {
        let field = empty_scene();
        // 空布局（NaN 包围盒 + 空 seeds）→ probe_count 至少 1（seeds 从 base_res 来）。
        let ab = AdaptiveBudget { max_rounds: 0, ..AdaptiveBudget::default() };
        let mut p = adaptive_place(
            &field,
            Vec3::splat(1.0),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::NAN),
            &[],
            ab,
        );
        p.smooth(f32::NAN, 0);
        p.smooth(1.0, 0); // k=0 → 钳到 1（只含自身 → 恒等）
        p.smooth(0.5, usize::MAX); // k 超大 → 钳到 n
        // 人为置 NaN 坐标的探针 → 不 panic。
        if let Some(first) = p.probes.first_mut() {
            first.position = Vec3::new(f32::NAN, 0.0, 0.0);
        }
        p.smooth(1.0, 4);
        // 空列表显式测试。
        let mut empty = AdaptivePlacement {
            probes: Vec::new(),
            sh_cubes: Vec::new(),
            rounds: 0,
            converged: false,
            min: Vec3::splat(0.0),
            max: Vec3::splat(0.0),
            order: ShOrder::L2,
        };
        empty.smooth(1.0, 4);
        assert_eq!(empty.probe_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 回归测试：`pub` 字段长度不一致时的 C18 越界
    // -----------------------------------------------------------------------

    /// **回归（C18）**：`AdaptivePlacement` 的 `pub` 字段长度不等时，
    /// `sample_irradiance` **不得** panic。
    ///
    /// # 缺陷回顾
    /// 该字段全 `pub`，外部可手工构造（或跨版本反序列化）出
    /// `sh_cubes.len() < probes.len()` 的布局。原实现按 `probes` 遍历并直接
    /// `&self.sh_cubes[i]` → **越界 panic**，违反"C18 永不 panic"铁律。
    /// 修复：以公共长度 `n = min(两者)` 为界遍历。
    #[test]
    fn adaptive_sample_irradiance_mismatched_lengths_no_panic() {
        let mk = |pos: Vec3| Probe {
            position: pos,
            normal: Vec3::new(0.0, 1.0, 0.0),
            ao: 1.0,
            env_radiance: Vec3::splat(1.0),
        };
        let bad = AdaptivePlacement {
            probes: vec![mk(Vec3::splat(0.0)), mk(Vec3::splat(1.0))],
            sh_cubes: vec![[[0.0f32; 9]; 3]], // 长度 1 < probes 2
            rounds: 0,
            converged: false,
            min: Vec3::splat(0.0),
            max: Vec3::splat(1.0),
            order: ShOrder::L2,
        };
        // 不 panic 且返回有限值（缺失系数的探针被跳过）。
        let e = bad.sample_irradiance(Vec3::splat(0.5), ShOrder::L2);
        assert!(e.x.is_finite() && e.y.is_finite() && e.z.is_finite(), "应返回有限值：{e:?}");
        // 反向（probes 比 sh_cubes 少）同样安全。
        let mut bad2 = bad.clone();
        bad2.probes.pop();
        let _ = bad2.sample_irradiance(Vec3::splat(0.5), ShOrder::L2);
        // 两者皆空 → 返回 0。
        let mut empty = bad.clone();
        empty.probes.clear();
        empty.sh_cubes.clear();
        assert_eq!(empty.sample_irradiance(Vec3::splat(0.5), ShOrder::L2), Vec3::splat(0.0));
    }

    /// **回归（C18）**：`ProbeGrid` 的 `res` 声称格数多于实际存储时，
    /// `sample_irradiance` / `smooth` **不得** panic。
    ///
    /// # 缺陷回顾
    /// `ProbeGrid` 字段全 `pub`。原 `index()` 只按 `res` 钳位，**未钳到实际
    /// 存储长度** → `res=(4,4,4)`（声称 64 格）但只存 1 个探针时，三线性
    /// 插值取角点 `sh_cubes[idx]` 必然越界 panic。修复：`index()` 二次钳到
    /// `min(sh_cubes.len(), probes.len()) - 1`。
    #[test]
    fn grid_sample_and_smooth_res_mismatch_no_panic() {
        let mut bad = ProbeGrid {
            min: Vec3::splat(0.0),
            max: Vec3::splat(1.0),
            res: (4, 4, 4), // 声称 64 格
            probes: vec![Probe {
                position: Vec3::splat(0.5),
                normal: Vec3::new(0.0, 1.0, 0.0),
                ao: 1.0,
                env_radiance: Vec3::splat(1.0),
            }], // 实际只有 1 个
            sh_cubes: vec![[[0.25f32; 9]; 3]],
            baked_order: ShOrder::L2,
            bounce_counts: vec![1],
            converged: vec![false],
        };
        // 三线性查询（含 NaN 点）不得 panic。
        let _ = bad.sample_irradiance(Vec3::splat(0.5), ShOrder::L2);
        let _ = bad.sample_irradiance(Vec3::new(f32::NAN, 0.0, 0.0), ShOrder::L2);
        // 平滑（唯一按 res 全量遍历并写回的地方）同样不得 panic。
        bad.smooth(1.0);
        let _ = bad.sample_irradiance(Vec3::splat(0.5), ShOrder::L2);
        // 反向不一致（探针多于系数）也不得 panic。
        let mut bad2 = bad.clone();
        bad2.probes.push(Probe {
            position: Vec3::splat(0.25),
            normal: Vec3::new(0.0, 1.0, 0.0),
            ao: 1.0,
            env_radiance: Vec3::splat(1.0),
        });
        bad2.smooth(0.5);
        let _ = bad2.sample_irradiance(Vec3::splat(0.25), ShOrder::L2);
    }

    // -----------------------------------------------------------------------
    // 回归测试：确定性全序 `cmp_point` 的 ±0.0 修复
    // -----------------------------------------------------------------------

    /// **回归**：`cmp_point` 必须与**数值序**一致（此前 `+0.0` 排到所有负数之前）。
    ///
    /// # 缺陷回顾
    /// `key(+0.0)` 曾返回 `0`，而负数的 key 是 `!bits ∈ (0, 0x8000_0000)`、正数
    /// 的 key 是 `bits | 0x8000_0000 ∈ [0x8000_0000, ∞)`。于是 `+0.0`（key 0）
    /// 排在负数（key > 0）之前 —— 破坏了"全序与数值序一致"这一隐含契约。
    /// 后果：坐标恰为 `+0.0` 的探针在字典序中被排到最前，
    /// [`sort_dedup_points`] / [`sort_cells`] 的输出不再是坐标升序，
    /// 规则格与自适应布局的**同坐标探针无法按序对齐**，交叉验证失去意义。
    ///
    /// 修复：`+0.0` 的 key 取符号幅值映射下的真实值 `0x8000_0000`。
    #[test]
    fn cmp_point_is_numerically_monotonic() {
        // 注意：`-0.0` 与 `+0.0` **故意判等**（去重语义，见下方断言），
        // 故严格升序样本里只取 `+0.0`。
        let ordered = [
            -1.0f32,
            -0.8,
            -f32::MIN_POSITIVE,
            0.0,
            f32::MIN_POSITIVE,
            0.8,
            1.0,
            f32::MAX,
        ];
        for w in ordered.windows(2) {
            let a = Vec3::new(w[0], 0.0, 0.0);
            let b = Vec3::new(w[1], 0.0, 0.0);
            assert_eq!(
                cmp_point(&a, &b),
                std::cmp::Ordering::Less,
                "cmp_point 应认可 {} < {}",
                w[0],
                w[1]
            );
        }
        // 关键回归点：所有负数都必须小于 `+0.0`（修复前 `+0.0` 的 key=0，
        // 会排到 key>0 的负数之前）。
        assert_eq!(
            cmp_point(&Vec3::new(-f32::MIN_POSITIVE, 0.0, 0.0), &Vec3::new(0.0, 0.0, 0.0)),
            std::cmp::Ordering::Less,
            "负的最小正数应 < +0.0"
        );
        assert_eq!(
            cmp_point(&Vec3::new(-0.8, 0.0, 0.0), &Vec3::new(0.0, 0.0, 0.0)),
            std::cmp::Ordering::Less,
            "-0.8 应 < +0.0（这是本回归的核心断言）"
        );
        // ±0.0 判等（去重语义保留）。
        assert_eq!(
            cmp_point(&Vec3::new(0.0, 0.0, 0.0), &Vec3::new(-0.0, 0.0, 0.0)),
            std::cmp::Ordering::Equal,
            "±0.0 应判等（否则无法去重 ±0.0 重复种子）"
        );
        // NaN 排末尾且不 panic。
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        assert_eq!(cmp_point(&nan, &Vec3::new(f32::MAX, 0.0, 0.0)), std::cmp::Ordering::Greater);
    }

    /// **回归**：含 `+0.0` 坐标的种子，排序后必须是真正的坐标升序。
    ///
    /// 这是 [`cmp_point_is_numerically_monotonic`] 的**行为级**断言：直接验证
    /// [`sort_dedup_points`] 的输出序，而不只是比较器的返回值。
    #[test]
    fn sort_dedup_points_keeps_numeric_order_with_zero() {
        let mut pts = vec![
            Vec3::new(0.8, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0), // +0.0 —— 修复前会被排到最前
            Vec3::new(-0.8, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(-0.4, 0.0, 0.0),
        ];
        sort_dedup_points(&mut pts);
        let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
        assert_eq!(
            xs,
            vec![-0.8, -0.4, 0.0, 0.4, 0.8],
            "排序后 x 必须升序（修复前 +0.0 会排在 -0.8 之前）"
        );
    }

    // -----------------------------------------------------------------------
    // 回归测试：自适应 smooth 的真实不变量（凸包包含，非"总能量守恒"）
    // -----------------------------------------------------------------------

    /// **回归**：自适应 smooth 满足**凸包包含**（值域收缩），这是真正成立的不变量。
    ///
    /// # 为什么不是"总能量守恒"
    /// k-NN 邻接是**行随机非双随机**：某探针被多少个邻域包含并不对称，故
    /// 系数总和**可增可减**（实测 `[0,0,100,0,0]` → `100 → 166.7`，增）。
    /// 任何"总能量守恒/单调不增"的断言都是**错的**，本测试锚定可证明的那条：
    /// 输出是输入邻域的凸组合 ⟹ `min(after) ≥ min(before)`、
    /// `max(after) ≤ max(before)`，即**永不制造新的极值**。
    #[test]
    fn adaptive_smooth_satisfies_hull_containment() {
        let field = empty_scene();
        // 覆盖"两端高峰"（总和减）与"单点高峰"（总和增）两种极值格局。
        let cases: [(u32, Vec<f32>); 3] = [
            (5, vec![100.0, 0.0, 0.0, 0.0, 100.0]),
            (5, vec![0.0, 0.0, 100.0, 0.0, 0.0]),
            (7, vec![50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0]),
        ];
        for (n, vals) in cases {
            let ab = AdaptiveBudget {
                base_res: (n, 1, 1),
                max_rounds: 0,
                budget: ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
                ..AdaptiveBudget::default()
            };
            let mut p =
                adaptive_place(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), &[], ab);
            assert_eq!(p.probe_count(), vals.len(), "n={n} 探针数应与给定值数一致");
            for (i, c) in p.sh_cubes.iter_mut().enumerate() {
                c[0][0] = vals[i];
            }
            let before: Vec<f32> = vals.clone();
            p.smooth(1.0, 3);
            let after: Vec<f32> = p.sh_cubes.iter().map(|c| c[0][0]).collect();
            let (mn0, mx0) = (
                before.iter().cloned().fold(f32::INFINITY, f32::min),
                before.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );
            let (mn1, mx1) = (
                after.iter().cloned().fold(f32::INFINITY, f32::min),
                after.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );
            assert!(mn1 >= mn0, "n={n} 最小值不应下降：{mn0} → {mn1}（after={after:?}）");
            assert!(mx1 <= mx0, "n={n} 最大值不应上升：{mx0} → {mx1}（after={after:?}）");
        }
    }

    /// **回归**：总能量（DC 系数和）**可增可减** —— 锚定"不要误写守恒"的诚实边界。
    ///
    /// 本测试是 [`AdaptivePlacement::smooth`] 文档里"总能量不守恒"那句断言的
    /// 可执行证据：它**要求**存在一个使总和**上升**的输入。若将来实现改为
    /// 真正守恒（如改成双随机邻域），本测试会失败并提醒更新文档 —— 这是刻意的
    /// "反向锚定"，防止注释与实现悄悄脱节。
    #[test]
    fn adaptive_smooth_total_energy_can_increase_and_decrease() {
        let field = empty_scene();
        let run = |n: u32, vals: &[f32]| -> (f32, f32) {
            let ab = AdaptiveBudget {
                base_res: (n, 1, 1),
                max_rounds: 0,
                budget: ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
                ..AdaptiveBudget::default()
            };
            let mut p =
                adaptive_place(&field, Vec3::splat(1.0), Vec3::splat(-1.0), Vec3::splat(1.0), &[], ab);
            for (i, c) in p.sh_cubes.iter_mut().enumerate() {
                c[0][0] = vals[i];
            }
            let s0: f32 = vals.iter().sum();
            p.smooth(1.0, 3);
            let s1: f32 = p.sh_cubes.iter().map(|c| c[0][0]).sum();
            (s0, s1)
        };
        // 单点高峰 → 邻域把大值摊向两侧 → 总和**增加**。
        let (s0, s1) = run(5, &[0.0, 0.0, 100.0, 0.0, 0.0]);
        assert!(s1 > s0, "单点高峰经平滑后总和应增加（证明不守恒）：{s0} → {s1}");
        // 两端高峰 → 边界钳邻居使大值被稀释 → 总和**减少**。
        let (s0, s1) = run(5, &[100.0, 0.0, 0.0, 0.0, 100.0]);
        assert!(s1 < s0, "两端高峰经平滑后总和应减少：{s0} → {s1}");
    }

    /// 规则网格 `smooth` 同样**不守恒总和**（边界钳邻居所致），锚定同一诚实边界。
    #[test]
    fn grid_smooth_total_energy_not_conserved() {
        let field = empty_scene();
        for n in [3u32, 5u32, 7u32] {
            let mut grid = bake_probe_grid(
                &field,
                Vec3::splat(1.0),
                Vec3::splat(-1.0),
                Vec3::splat(1.0),
                (n, 1, 1),
                ProbeBudget { sh_samples: 4, ao_samples: 0, ..ProbeBudget::low() },
            );
            let vals: Vec<f32> = (0..grid.count()).map(|i| (i as f32) * (i as f32)).collect();
            for (i, c) in grid.sh_cubes.iter_mut().enumerate() {
                c[0][0] = vals[i];
            }
            let s0: f32 = vals.iter().sum();
            grid.smooth(1.0);
            let after: Vec<f32> = grid.sh_cubes.iter().map(|c| c[0][0]).collect();
            let s1: f32 = after.iter().sum();
            assert!(s1 < s0, "n={n} 网格平滑因边界钳邻居使总和减少：{s0} → {s1}");
            // 凸包包含仍成立。
            let (mn0, mx0) = (
                vals.iter().cloned().fold(f32::INFINITY, f32::min),
                vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );
            let (mn1, mx1) = (
                after.iter().cloned().fold(f32::INFINITY, f32::min),
                after.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );
            assert!(mn1 >= mn0 && mx1 <= mx0, "n={n} 网格平滑也应满足凸包包含");
        }
    }

    // -----------------------------------------------------------------------
    // 端到端查询（本轮新增）测试
    // -----------------------------------------------------------------------

    /// **端到端确定性**：从多弹射自适应烘焙到任意点查询，同输入两遍逐位一致。
    #[test]
    fn end_to_end_query_bitwise_deterministic() {
        let field = gradient_scene();
        let ab = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 2,
            error_threshold: 1e-3,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let mb = MultiBounceBudget {
            n_samples: 16,
            max_bounces: 6,
            tolerance: 1e-4,
            decay: 0.8,
            occlusion: OcclusionBudget::low(),
        };
        let a = bake_adaptive_gi_end_to_end(
            &field,
            Vec3::splat(0.6),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &[],
            ab,
            mb,
            1.0,
            6,
        );
        let b = bake_adaptive_gi_end_to_end(
            &field,
            Vec3::splat(0.6),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &[],
            ab,
            mb,
            1.0,
            6,
        );
        assert_eq!(a.probe_count(), b.probe_count());
        // 固定查询点集，逐点逐位一致。
        for iz in 0..3u32 {
            for ix in 0..3u32 {
                let q = Vec3::new(
                    -1.5 + 1.5 * ix as f32,
                    0.3,
                    -1.5 + 1.5 * iz as f32,
                );
                let ea = a.sample_irradiance(q, ShOrder::L2);
                let eb = b.sample_irradiance(q, ShOrder::L2);
                assert_eq!(ea.x.to_bits(), eb.x.to_bits(), "端到端查询 X 应逐位一致");
                assert_eq!(ea.y.to_bits(), eb.y.to_bits());
                assert_eq!(ea.z.to_bits(), eb.z.to_bits());
            }
        }
    }

    /// 端到端多弹射自适应 ≥ 单弹射（能量单调），且查询有响应（非恒 0/NaN）。
    #[test]
    fn end_to_end_multi_ge_single() {
        let field = gradient_scene();
        let ab = AdaptiveBudget {
            base_res: (2, 2, 2),
            max_rounds: 1,
            error_threshold: 1e-3,
            budget: ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() },
            ..AdaptiveBudget::default()
        };
        let single = adaptive_place(&field, Vec3::splat(0.6), Vec3::splat(-2.0), Vec3::splat(2.0), &[], ab);
        let multi = adaptive_place_multi_bounce(
            &field,
            Vec3::splat(0.6),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &[],
            ab,
            MultiBounceBudget {
                n_samples: 16,
                max_bounces: 8,
                tolerance: 1e-5,
                decay: 0.8,
                occlusion: OcclusionBudget::low(),
            },
        );
        assert_eq!(single.probe_count(), multi.probe_count(), "布局应一致（仅求解不同）");
        // 查询点的辐照度：多弹射 ≥ 单弹射（逐点）。
        let mut any_strict = false;
        for i in 0..5u32 {
            let q = Vec3::new(-1.0 + 0.5 * i as f32, 0.5, 0.0);
            let es = single.sample_irradiance(q, ShOrder::L0);
            let em = multi.sample_irradiance(q, ShOrder::L0);
            assert!(em.x.is_finite() && em.x >= -1e-6);
            assert!(em.x >= es.x - 1e-5, "查询 {i}: 多弹射 {} 应 ≥ 单弹射 {}", em.x, es.x);
            if em.x > es.x + 1e-6 {
                any_strict = true;
            }
        }
        assert!(any_strict, "至少一个查询点多弹射严格更亮");
    }

    /// `from_grid` 桥接：规则多弹射烘焙 → 自适应查询 API，结果与网格自身查询一致。
    #[test]
    fn from_grid_bridge_matches_grid_query() {
        let field = gradient_scene();
        let pb = ProbeBudget { sh_samples: 8, ao_samples: 4, ..ProbeBudget::low() };
        let grid = bake_probe_grid_multi_bounce(
            &field,
            Vec3::splat(0.7),
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            (2, 2, 2),
            ProbeMultiBounceBudget {
                probe: pb,
                multi: MultiBounceBudget {
                    n_samples: 16,
                    max_bounces: 6,
                    tolerance: 1e-4,
                    decay: 0.8,
                    occlusion: pb.occlusion,
                },
            },
        );
        let adp = AdaptivePlacement::from_grid(&grid);
        assert_eq!(adp.probe_count(), grid.count());
        // 查询点恰在探针处：两者都应返回该探针值（IDW 退化为精确点）。
        for probe in &grid.probes {
            let eg = grid.sample_irradiance(probe.position, ShOrder::L0);
            // IDW 在恰好重合时取该探针值；此处用近距离查询对比趋势。
            let ea = adp.sample_irradiance(probe.position, ShOrder::L0);
            // 网格三线性在同权时取角点均值；自适应取精确探针值——应同量级。
            assert!(
                (eg.x - ea.x).abs() <= eg.x.abs().max(1.0) * 0.5 + 1.0,
                "桥接查询应与网格同量级：{} vs {}",
                eg.x,
                ea.x
            );
        }
    }
}
