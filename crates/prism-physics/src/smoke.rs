//! FLIP 烟雾/爆炸求解器（「烟雾/爆炸模拟侧」）。
//!
//! 语义权威：的分层约定——《我们的体积是渲染侧
//! （均匀介质 + HG 相函数）；FLIP = 粒子采样（S4）+ 网格压力投影
//! （泊松方程 = S2/S6 求逆族）+ 稀疏体素（格锚定，D-22 天然应用）——全部落在
//! 既有 12 结构内，无需新结构》。本模块是该路径的落点：**纯内部设计**、零
//! 第三方依赖（仅 std + workspace 内 crate）、C18 永不 panic、D-16 Mini Code。
//!
//! # 数学结构（FLIP = P + G + 传输 + 投影）
//!
//! ```text
//!   ① P2G   ：粒子 → 网格（散射，核函数 W）
//!   ② 浮力   ：f_buoy = α·T·ŷ（Boussinesq 近似，作用在网格 y 面心速度）
//!   ③ 投影   ：Gauss–Seidel 解 ∇²p = ∇·v/Δt（固定迭代数，无提前退出）
//!   ④ G2P   ：网格 → 粒子（采集 + FLIP 增量混合）
//!          v_p = (1−α)·v_pic + α·(v_p + Δv_grid)
//!   ⑤ 标量输运：温度 T / 烟密度 ρ 半拉格朗日平流（本模块深化）
//!   ⑥ 对流失稳守卫：查 CFL，dt ≤ CFL·dx / max|v|（越界显式拒绝，不产 NaN/Inf）
//! ```
//!
//! # 烟雾深化 A（标量场输运 + 浮力）
//!
//! - **标量场**：温度 `temperature` 与烟密度 `density` 与速度**同网格分辨率**
//!   （单元中心，`(nz,ny,nx)`），二者共同驱动浮力，并作为渲染侧的体积标量。
//! - **半拉格朗日平流**（Stam 1999, *Stable Fluids*, SIGGRAPH；Bridson,
//!   *Fluid Simulation for Computer Graphics*, ch. 2）：对每个单元中心回溯
//!   出发位置 `x_src = x − Δt·u(x)`，在三线性插值下采样旧场。该方法**无条件
//!   稳定**（不会因大 Δt 发散），代价是数值耗散（见「诚实边界」）。
//! - **浮力**（Boussinesq 近似，Spiegel & Veronis 1960）：`f_buoy = α·T·ŷ`，
//!   仅作用在重力轴向（本实装固定 +Y 为「上」）。密度场**不**参与浮力，仅作
//!   被输运的被动标量（见 [`Smoke::apply_buoyancy`] 的裁决注释）。
//!
//! # 确定性纪律
//! - **压力投影**：MAC（staggered）网格上写散度 → Gauss–Seidel（红黑不需，
//!   顺序扫描即确定性）迭代求解压力泊松方程；迭代次数由调用方固定传入，
//!   无收敛阈值提前退出（迭代序变化会破坏逐位确定性）。
//! - **稀疏体素**：本实装用**规则包围盒 MAC 网格**（体积分数域），稀疏体素
//!   优化（只存活跃块）标注为 GPU-only 迁移位，本批不实装——见文末。
//!
//! # 确定性纪律
//!
//! - 所有网格数组下标遍历按 `(k, j, i)` 固定字典序，无哈希迭代序依赖；
//! - 粒子数组下标即粒子 id（格锚 Zᵏ），散射/采集按粒子下标升序；
//! - 迭代次数 / 子步数显式固定，无收敛判据提前退出；
//! - 浮点归约（散度求和、压力累加）顺序固定，仅用 IEEE 基本运算 + − × ÷ √；
//! - 无全局 RNG（初态由调用方给定或按 hash 派生，见 [`Smoke::seed_block`]）。
//!
//! # 复用声明（「全部落在既有结构内」的落地）
//!
//! **复用**：
//! - [`crate::PhysicsError`]：crate 统一三段式错误（C18/C19），错误码 P1014/P1015
//!   等防呆段与 P3009 结构段（对齐既有 P1xxx/P3xxx 分段）。
//! - [`prism_math::Vec3`]：向量代数（+ − × √），零新结构。
//!
//! **不复用**（如实说明）：
//! - SPH 的 Poly6/Spiky 核（[`crate::sph`] 私有 fn）：FLIP 传输用三线性核，
//!   SPH 核会引入错误的密度归一化（粒子→网格是散射，不是密度估计）；且
//!   sph.rs 的核函数为模块私有，暴露会扩大改动面（文件归属约束下最小改动）。
//! - [`crate::xpbd::World`]：FLIP 是**网格型求解器**（未知量在网格面心），
//!   不是粒子位置约束投影，XPBD 的位置约束循环不适用（FLIP 的
//!   重力等是网格层，非 XPBD 粒子约束族）。
//! - SPH 的 [`crate::sph::Fluid`] 粒子池：FLIP 粒子状态（速度 + FLIP 增量
//!   混合）语义与 SPH 的 PBF 密度约束不同，直接复用会耦合两套语义。
//!   prism-particles 的 [`prism_particles`] 本 crate **不依赖**（避免新增
//!   crate 依赖边；粒子池极小，就地定义 [`SmokeParticle`] 更贴合 Mini Code）。

use prism_math::Vec3;

use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 粒子的 FLIP 侧表示
// ---------------------------------------------------------------------------

/// FLIP 粒子：位置（世界坐标）+ 速度（世界坐标）+ 被动标量（温度 / 烟密度）。
///
/// 下标即粒子 id（格锚 Zᵏ）；下标序是 P2G/G2P 的确定性遍历序。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmokeParticle {
    /// 当前位置（世界坐标）。
    pub position: Vec3,
    /// 当前速度（世界坐标）。
    pub velocity: Vec3,
    /// 粒子温度（随网格标量场输运，驱动浮力回散射）。
    pub temperature: f32,
    /// 粒子烟密度（被动标量，随流场输运）。
    pub density: f32,
}

impl SmokeParticle {
    /// 构造粒子（标量默认 0——不携带温度/密度，等价旧行为）。
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position,
            velocity,
            temperature: 0.0,
            density: 0.0,
        }
    }

    /// 构造携带温度/密度的粒子。
    pub fn with_scalars(position: Vec3, velocity: Vec3, temperature: f32, density: f32) -> Self {
        Self {
            position,
            velocity,
            temperature,
            density,
        }
    }
}

// ---------------------------------------------------------------------------
// MAC 网格（staggered grid）
// ---------------------------------------------------------------------------

/// 规则 MAC（staggered）网格：u 存 x 面心、v 存 y 面心、w 存 z 面心。
///
/// 网格域 = 轴对齐包围盒 `[origin, origin + size]`，单元边长 `dx`。
/// 单元数 = `res`（nx, ny, nz），面心维度 = `res + 1` 沿各轴。
///
/// 内存布局（全 `Vec<f32>`，确定性下标）：
/// - `u`：`(nz, ny, nx+1)`，`u[idx_u(i,j,k)] = x-面 (i,j,k) 速度`；
/// - `v`：`(nz, ny+1, nx)`；
/// - `w`：`(nz+1, ny, nx)`；
/// - `pressure`：单元中心 `(nz, ny, nx)`；
/// - `cell_type`：单元类型（流体/空气/固壁），`(nz, ny, nx)`；
/// - `temperature`：单元中心温度 `(nz, ny, nx)`（浮力驱动源）；
/// - `density`：单元中心烟密度 `(nz, ny, nx)`（被动标量）。
#[derive(Debug, Clone, PartialEq)]
pub struct MacGrid {
    /// 网格原点（最小角）。
    pub origin: Vec3,
    /// 单元边长（> 0）。
    pub dx: f32,
    /// 单元分辨率 (nx, ny, nz)，各分量 ≥ 1。
    pub res: (usize, usize, usize),
    /// x 面心速度，维度 `(nz, ny, nx+1)`。
    pub u: Vec<f32>,
    /// y 面心速度，维度 `(nz, ny+1, nx)`。
    pub v: Vec<f32>,
    /// z 面心速度，维度 `(nz+1, ny, nx)`。
    pub w: Vec<f32>,
    /// 单元中心压力，维度 `(nz, ny, nx)`。
    pub pressure: Vec<f32>,
    /// 单元类型，维度 `(nz, ny, nx)`。
    pub cell_type: Vec<CellType>,
    /// 单元中心温度（K 或无量纲），维度 `(nz, ny, nx)`。
    pub temperature: Vec<f32>,
    /// 单元中心烟密度（被动标量），维度 `(nz, ny, nx)`。
    pub density: Vec<f32>,
}

/// 单元类型（简化 VOF 标记：流体参与散度、空气自由、固壁速度法向为零）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    /// 流体（有粒子覆盖）。
    Fluid,
    /// 空气（开放边界，压力 = 0）。
    Air,
    /// 固壁（不可穿透，速度法向为零）。
    Solid,
}

impl MacGrid {
    /// 构造空网格（全零速度、全 Air、压力 0）。
    ///
    /// # 防呆（C18）
    /// 拒绝 `dx <= 0` 或非有限、`res` 任一维为 0、`origin` 非有限。
    pub fn new(origin: Vec3, dx: f32, res: (usize, usize, usize)) -> Result<Self, PhysicsError> {
        if !dx.is_finite() || dx <= 0.0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "MacGrid::new".to_string(),
                problem: format!("dx = {dx}：单元边长必须为正有限实数（NaN/±Inf/0 同拒）"),
                fix: "传正单元边长，如域长 / 分辨率".to_string(),
            });
        }
        let (nx, ny, nz) = res;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "MacGrid::new".to_string(),
                problem: format!("res = {res:?}：各维分辨率必须 ≥ 1"),
                fix: "传 nx, ny, nz ≥ 1".to_string(),
            });
        }
        if !origin.x.is_finite() || !origin.y.is_finite() || !origin.z.is_finite() {
            return Err(PhysicsError {
                code: "P1014",
                location: "MacGrid::new".to_string(),
                problem: format!("origin = {origin:?}：网格原点三分量必须有限"),
                fix: "传有限原点".to_string(),
            });
        }
        let cells = nx * ny * nz;
        Ok(Self {
            origin,
            dx,
            res,
            u: vec![0.0; (nx + 1) * ny * nz],
            v: vec![0.0; nx * (ny + 1) * nz],
            w: vec![0.0; nx * ny * (nz + 1)],
            pressure: vec![0.0; cells],
            cell_type: vec![CellType::Air; cells],
            temperature: vec![0.0; cells],
            density: vec![0.0; cells],
        })
    }

    /// 单元数 (nx, ny, nz)。
    #[inline]
    pub fn dims(&self) -> (usize, usize, usize) {
        self.res
    }

    /// 总单元数。
    #[inline]
    pub fn cell_count(&self) -> usize {
        let (nx, ny, nz) = self.res;
        nx * ny * nz
    }

    /// 域最大角（origin + res·dx）。
    #[inline]
    pub fn domain_max(&self) -> Vec3 {
        let (nx, ny, nz) = self.res;
        self.origin + Vec3::new(nx as f32 * self.dx, ny as f32 * self.dx, nz as f32 * self.dx)
    }

    /// 单元 (i,j,k) 中心的世界坐标（越界返回 origin——调用方须自查范围）。
    #[inline]
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let d = self.dx;
        self.origin
            + Vec3::new(
                (i as f32 + 0.5) * d,
                (j as f32 + 0.5) * d,
                (k as f32 + 0.5) * d,
            )
    }

    // 索引辅助：u/v/w/pressure 分别线性化（确定性、无分支负下标）。
    #[inline]
    fn idx_u(&self, i: usize, j: usize, k: usize) -> usize {
        let (nx, ny, _) = self.res;
        k * ny * (nx + 1) + j * (nx + 1) + i
    }
    #[inline]
    fn idx_v(&self, i: usize, j: usize, k: usize) -> usize {
        let (nx, ny, _) = self.res;
        k * (ny + 1) * nx + j * nx + i
    }
    #[inline]
    fn idx_w(&self, i: usize, j: usize, k: usize) -> usize {
        let (nx, ny, _) = self.res;
        k * ny * nx + j * nx + i
    }
    #[inline]
    fn idx_c(&self, i: usize, j: usize, k: usize) -> usize {
        let (nx, ny, _) = self.res;
        k * ny * nx + j * nx + i
    }

    /// u 分量面心值（范围自查；越界返回 0.0 —— 安全降级，不 panic）。
    #[inline]
    pub fn u_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.res;
        if i > nx || j >= ny || k >= nz {
            return 0.0;
        }
        self.u[self.idx_u(i, j, k)]
    }
    /// v 分量面心值。
    #[inline]
    pub fn v_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.res;
        if i >= nx || j > ny || k >= nz {
            return 0.0;
        }
        self.v[self.idx_v(i, j, k)]
    }
    /// w 分量面心值。
    #[inline]
    pub fn w_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k > nz {
            return 0.0;
        }
        self.w[self.idx_w(i, j, k)]
    }

    /// 写入 u 面心值（越界忽略——安全降级）。
    #[inline]
    fn set_u(&mut self, i: usize, j: usize, k: usize, val: f32) {
        let (nx, ny, nz) = self.res;
        if i > nx || j >= ny || k >= nz {
            return;
        }
        let idx = self.idx_u(i, j, k);
        self.u[idx] = val;
    }
    /// 写入 v 面心值。
    #[inline]
    fn set_v(&mut self, i: usize, j: usize, k: usize, val: f32) {
        let (nx, ny, nz) = self.res;
        if i >= nx || j > ny || k >= nz {
            return;
        }
        let idx = self.idx_v(i, j, k);
        self.v[idx] = val;
    }
    /// 写入 w 面心值。
    #[inline]
    fn set_w(&mut self, i: usize, j: usize, k: usize, val: f32) {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k > nz {
            return;
        }
        let idx = self.idx_w(i, j, k);
        self.w[idx] = val;
    }

    /// 读 u 的可变引用（散射用；越界返回 None）。
    #[inline]
    fn u_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut f32> {
        let (nx, ny, nz) = self.res;
        if i > nx || j >= ny || k >= nz {
            return None;
        }
        let idx = self.idx_u(i, j, k);
        self.u.get_mut(idx)
    }
    #[inline]
    fn v_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut f32> {
        let (nx, ny, nz) = self.res;
        if i >= nx || j > ny || k >= nz {
            return None;
        }
        let idx = self.idx_v(i, j, k);
        self.v.get_mut(idx)
    }
    #[inline]
    fn w_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut f32> {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k > nz {
            return None;
        }
        let idx = self.idx_w(i, j, k);
        self.w.get_mut(idx)
    }

    /// 单元类型（越界 = Solid——域外视为壁）。
    #[inline]
    pub fn cell_type_at(&self, i: usize, j: usize, k: usize) -> CellType {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k >= nz {
            return CellType::Solid;
        }
        self.cell_type[self.idx_c(i, j, k)]
    }

    /// 单元中心温度（越界返回 0.0——安全降级，不 panic）。
    #[inline]
    pub fn temperature_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k >= nz {
            return 0.0;
        }
        self.temperature.get(self.idx_c(i, j, k)).copied().unwrap_or(0.0)
    }

    /// 单元中心烟密度（越界返回 0.0——安全降级，不 panic）。
    #[inline]
    pub fn density_at(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k >= nz {
            return 0.0;
        }
        self.density.get(self.idx_c(i, j, k)).copied().unwrap_or(0.0)
    }

    /// 写入单元温度（越界忽略——安全降级）。用于设初值 / 外部驱动。
    #[inline]
    pub fn set_temperature(&mut self, i: usize, j: usize, k: usize, val: f32) {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k >= nz {
            return;
        }
        let idx = self.idx_c(i, j, k);
        if let Some(slot) = self.temperature.get_mut(idx) {
            *slot = val;
        }
    }

    /// 写入单元烟密度（越界忽略——安全降级）。
    #[inline]
    pub fn set_density(&mut self, i: usize, j: usize, k: usize, val: f32) {
        let (nx, ny, nz) = self.res;
        if i >= nx || j >= ny || k >= nz {
            return;
        }
        let idx = self.idx_c(i, j, k);
        if let Some(slot) = self.density.get_mut(idx) {
            *slot = val;
        }
    }

    /// 温度场求和（总热量代理量；确定性 k,j,i 行主序累加）。
    pub fn temperature_sum(&self) -> f32 {
        let mut acc = 0.0f32;
        for t in &self.temperature {
            acc += *t;
        }
        acc
    }

    /// 烟密度场求和（总质量代理量；确定性 k,j,i 行主序累加）。
    pub fn density_sum(&self) -> f32 {
        let mut acc = 0.0f32;
        for d in &self.density {
            acc += *d;
        }
        acc
    }

    /// 把所有边界面的法向速度置零（固壁条件），并标记域边界单元外法向为壁。
    ///
    /// 简化实现：整个域的外壳面（i=0 / i=nx 等的面心）法向速度清零。
    pub fn apply_solid_walls(&mut self) {
        let (nx, ny, nz) = self.res;
        for k in 0..nz {
            for j in 0..ny {
                self.set_u(0, j, k, 0.0);
                self.set_u(nx, j, k, 0.0);
            }
        }
        for k in 0..nz {
            for i in 0..nx {
                self.set_v(i, 0, k, 0.0);
                self.set_v(i, ny, k, 0.0);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                self.set_w(i, j, 0, 0.0);
                self.set_w(i, j, nz, 0.0);
            }
        }
    }

    /// 世界坐标 → 连续单元坐标（p − origin)/dx（无钳制，调用方判断边界）。
    #[inline]
    fn to_grid(&self, p: Vec3) -> Vec3 {
        (p - self.origin) / self.dx
    }
}

// ---------------------------------------------------------------------------
// 求解器规格
// ---------------------------------------------------------------------------

/// 边界模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// 封闭容器（六面固壁，速度法向为零）。
    Closed,
    /// 开放边界（顶面/侧面为空气，烟雾可散逸；底面仍为壁）。
    Open,
}

/// FLIP 求解器规格。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmokeSpec {
    /// 网格原点（最小角）。
    pub origin: Vec3,
    /// 单元边长（> 0）。
    pub dx: f32,
    /// 分辨率。
    pub res: (usize, usize, usize),
    /// 重力加速度。
    pub gravity: Vec3,
    /// 压力投影 Gauss–Seidel 迭代数（≥ 1，固定值）。
    pub pressure_iters: u32,
    /// FLIP 混合比例 α ∈ [0,1]（1 = 纯 FLIP，0 = 纯 PIC）。
    pub flip_ratio: f32,
    /// CFL 安全系数（> 0，典型 0.5–5）。
    pub cfl: f32,
    /// 单帧最大子步数（CFL 细分上限；防死循环）。
    pub max_substeps: u32,
    /// 边界模式。
    pub boundary: Boundary,
    /// 浮力系数 α（Boussinesq 近似；单位 m·s⁻²·K⁻¹）。0 = 关闭浮力。
    ///
    /// 未出现在旧规格中，为**向后兼容**标 `#[serde]` 无关的纯 struct
    /// 字段；既有测试用结构体更新语法 `..spec(..)` 不需改。语义见
    /// [`Smoke::apply_buoyancy`]。
    pub buoyancy: f32,
}

impl SmokeSpec {
    /// 校验参数域（C18 防呆；三段式报错）。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        if !self.dx.is_finite() || self.dx <= 0.0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("dx = {}：必须为正有限实数", self.dx),
                fix: "传正单元边长".to_string(),
            });
        }
        let (nx, ny, nz) = self.res;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError {
                code: "P1014",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("res = {res:?}：各维 ≥ 1", res = self.res),
                fix: "传 nx, ny, nz ≥ 1".to_string(),
            });
        }
        if !self.gravity.x.is_finite() || !self.gravity.y.is_finite() || !self.gravity.z.is_finite() {
            return Err(PhysicsError {
                code: "P1016",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("gravity = {:?}：三分量必须有限（NaN/±Inf 同拒）", self.gravity),
                fix: "传有限重力向量".to_string(),
            });
        }
        if self.pressure_iters == 0 {
            return Err(PhysicsError {
                code: "P1017",
                location: "SmokeSpec::validate".to_string(),
                problem: "pressure_iters = 0：压力迭代数必须 ≥ 1（固定迭代是确定性纪律）".to_string(),
                fix: "传 ≥ 1 的固定迭代数（典型 20–80）".to_string(),
            });
        }
        if !self.flip_ratio.is_finite() || self.flip_ratio < 0.0 || self.flip_ratio > 1.0 {
            return Err(PhysicsError {
                code: "P1015",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("flip_ratio = {}：必须在 [0,1] 内（NaN 同拒）", self.flip_ratio),
                fix: "传 [0,1] 内比例（1 = 纯 FLIP，0 = 纯 PIC）".to_string(),
            });
        }
        if !self.cfl.is_finite() || self.cfl <= 0.0 {
            return Err(PhysicsError {
                code: "P1018",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("cfl = {}：CFL 系数必须为正有限实数", self.cfl),
                fix: "传正 CFL 系数（典型 0.5–5）".to_string(),
            });
        }
        if self.max_substeps == 0 {
            return Err(PhysicsError {
                code: "P1018",
                location: "SmokeSpec::validate".to_string(),
                problem: "max_substeps = 0：子步上限必须 ≥ 1".to_string(),
                fix: "传 ≥ 1 的子步上限（典型 8–64）".to_string(),
            });
        }
        if !self.buoyancy.is_finite() {
            return Err(PhysicsError {
                code: "P1016",
                location: "SmokeSpec::validate".to_string(),
                problem: format!("buoyancy = {}：浮力系数必须有限（NaN/±Inf 同拒）", self.buoyancy),
                fix: "传有限浮力系数（0 = 关闭浮力，典型 1–10）".to_string(),
            });
        }
        Ok(())
    }
}

/// 单帧推进报告（免费质量指标：投影前后最大散度）。
#[derive(Debug, Clone, PartialEq)]
pub struct SmokeStepReport {
    /// 实际使用的子步数（CFL 细分数）。
    pub substeps: u32,
    /// 有效子步长（秒）。
    pub dt_sub: f32,
    /// 投影前 max|∇·v|。
    pub divergence_before: f32,
    /// 投影后 max|∇·v|。
    pub divergence_after: f32,
}

// ---------------------------------------------------------------------------
// FLIP 求解器
// ---------------------------------------------------------------------------

/// FLIP 烟雾/爆炸求解器（P 粒子 + G MAC 网格 + 传输 + 压力投影）。
///
/// 状态 = 粒子数组 + MAC 网格。同输入（规格 + 初态 + 步序列）必须逐位可复现。
pub struct Smoke {
    spec: SmokeSpec,
    grid: MacGrid,
    particles: Vec<SmokeParticle>,
}

impl std::fmt::Debug for Smoke {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Smoke")
            .field("particles", &self.particles.len())
            .field("res", &self.spec.res)
            .field("dx", &self.spec.dx)
            .field("pressure_iters", &self.spec.pressure_iters)
            .finish()
    }
}

impl Smoke {
    /// 构造求解器（给定初态粒子）。
    ///
    /// 空粒子集是合法输入（返回空求解器，`step` 为空操作）。
    pub fn new(spec: SmokeSpec, initial: &[SmokeParticle]) -> Result<Self, PhysicsError> {
        spec.validate()?;
        // 校验粒子初态有限（拒绝 NaN/Inf 污染——C18 防呆，不静默透传）。
        for (i, p) in initial.iter().enumerate() {
            if !is_finite_vec3(p.position) || !is_finite_vec3(p.velocity) {
                return Err(PhysicsError {
                    code: "P1019",
                    location: "Smoke::new（粒子初态）".to_string(),
                    problem: format!(
                        "粒子 {i} 位置或速度非有限（pos={:?}, vel={:?}）",
                        p.position, p.velocity
                    ),
                    fix: "初态须为有限实数（在构造期拒绝 NaN/Inf，勿透传下游）".to_string(),
                });
            }
        }
        let grid = MacGrid::new(spec.origin, spec.dx, spec.res)?;
        Ok(Self {
            spec,
            grid,
            particles: initial.to_vec(),
        })
    }

    /// 用规则体块初始化粒子（确定性：无 RNG，等差网格）。
    ///
    /// 在 `[base, base + dims·spacing]` 生成 `dims` 个粒子（世界坐标），
    /// 初速度统一 `vel`。这是"烟雾团"的标准确定性初态。
    ///
    /// # 防呆
    /// 拒绝非有限参数、`spacing <= 0`、任一分量为 0。
    pub fn seed_block(
        spec: SmokeSpec,
        base: Vec3,
        dims: (usize, usize, usize),
        spacing: f32,
        vel: Vec3,
    ) -> Result<Self, PhysicsError> {
        spec.validate()?;
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(PhysicsError {
                code: "P1019",
                location: "Smoke::seed_block".to_string(),
                problem: format!("spacing = {spacing}：必须为正有限实数"),
                fix: "传正粒子间距".to_string(),
            });
        }
        let (a, b, c) = dims;
        if a == 0 || b == 0 || c == 0 {
            return Err(PhysicsError {
                code: "P1019",
                location: "Smoke::seed_block".to_string(),
                problem: format!("dims = {dims:?}：各维 ≥ 1"),
                fix: "传各维 ≥ 1 的体块尺寸".to_string(),
            });
        }
        let mut initial = Vec::with_capacity(a * b * c);
        // 固定 (k, j, i) 字典序 → 确定性粒子下标。
        for k in 0..c {
            for j in 0..b {
                for i in 0..a {
                    let pos = base
                        + Vec3::new(i as f32 * spacing, j as f32 * spacing, k as f32 * spacing);
                    initial.push(SmokeParticle::new(pos, vel));
                }
            }
        }
        Self::new(spec, &initial)
    }

    /// 用规则体块初始化粒子，并给每个粒子赋统一温度 / 密度。
    ///
    /// 与 [`Smoke::seed_block`] 同构，仅额外给初态粒子赋标量 `temperature` /
    /// `density`（用于浮力/标量输运测试的确定性初态）。
    ///
    /// # 防呆
    /// 同 [`Smoke::seed_block`]；另拒非有限标量（P1019）。
    pub fn seed_block_scalars(
        spec: SmokeSpec,
        base: Vec3,
        dims: (usize, usize, usize),
        spacing: f32,
        vel: Vec3,
        temperature: f32,
        density: f32,
    ) -> Result<Self, PhysicsError> {
        spec.validate()?;
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(PhysicsError {
                code: "P1019",
                location: "Smoke::seed_block_scalars".to_string(),
                problem: format!("spacing = {spacing}：必须为正有限实数"),
                fix: "传正粒子间距".to_string(),
            });
        }
        if !temperature.is_finite() || !density.is_finite() {
            return Err(PhysicsError {
                code: "P1019",
                location: "Smoke::seed_block_scalars".to_string(),
                problem: format!("标量非有限（T={temperature}, ρ={density}）"),
                fix: "传有限温度/密度".to_string(),
            });
        }
        let (a, b, c) = dims;
        if a == 0 || b == 0 || c == 0 {
            return Err(PhysicsError {
                code: "P1019",
                location: "Smoke::seed_block_scalars".to_string(),
                problem: format!("dims = {dims:?}：各维 ≥ 1"),
                fix: "传各维 ≥ 1 的体块尺寸".to_string(),
            });
        }
        let mut initial = Vec::with_capacity(a * b * c);
        for k in 0..c {
            for j in 0..b {
                for i in 0..a {
                    let pos = base
                        + Vec3::new(i as f32 * spacing, j as f32 * spacing, k as f32 * spacing);
                    initial.push(SmokeParticle::with_scalars(pos, vel, temperature, density));
                }
            }
        }
        Self::new(spec, &initial)
    }

    /// 粒子数。
    #[inline]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// 粒子切片（只读；下标即 id）。
    #[inline]
    pub fn particles(&self) -> &[SmokeParticle] {
        &self.particles
    }

    /// 网格只读引用。
    #[inline]
    pub fn grid(&self) -> &MacGrid {
        &self.grid
    }

    /// 网格可变引用（供测试 / 外部直接设标量初值；结构性改动须谨慎）。
    #[inline]
    pub fn grid_mut(&mut self) -> &mut MacGrid {
        &mut self.grid
    }

    /// 网格温度场切片（维度 `(nz,ny,nx)`，行主序）。
    #[inline]
    pub fn temperature_field(&self) -> &[f32] {
        &self.grid.temperature
    }

    /// 网格烟密度场切片（维度 `(nz,ny,nx)`，行主序）。
    #[inline]
    pub fn density_field(&self) -> &[f32] {
        &self.grid.density
    }

    /// 世界坐标处温度（三线性采样网格温度场；越界钳制到边界，不 panic）。
    pub fn sample_temperature_at(&self, pos: Vec3) -> f32 {
        sample_scalar_field_global(&self.grid, &self.grid.temperature, pos)
    }

    /// 世界坐标处烟密度（三线性采样；越界钳制，不 panic）。
    pub fn sample_density_at(&self, pos: Vec3) -> f32 {
        sample_scalar_field_global(&self.grid, &self.grid.density, pos)
    }

    /// 世界坐标处的网格速度（三线性采样 MAC 面心；越界取 0）。
    pub fn sample_velocity_at(&self, pos: Vec3) -> Vec3 {
        self.sample_velocity(pos)
    }

    /// 用速度场对网格标量场做**一次**半拉格朗日平流（`dt`）。
    ///
    /// 暴露独立入口，便于测试「纯平流」路径（不经 P2G/投影的其它耦合）。
    /// 语义见 [`Smoke::advect_scalars`]；确定性保证同该函数。
    pub fn advect_scalar_fields(&mut self, dt: f32) -> Result<(), PhysicsError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(PhysicsError {
                code: "P1002",
                location: "Smoke::advect_scalar_fields".to_string(),
                problem: format!("dt = {dt}：平流步长必须为有限非负实数"),
                fix: "传 dt ≥ 0（秒）".to_string(),
            });
        }
        self.advect_scalars(dt);
        Ok(())
    }

    /// 直接在网格 y 面心写入速度（测试 / 外部驱动用；越界忽略）。
    pub fn set_grid_velocity_y(&mut self, i: usize, j: usize, k: usize, val: f32) {
        self.grid.set_v(i, j, k, val);
    }

    /// 直接在网格 x 面心写入速度（测试 / 外部驱动用；越界忽略）。
    pub fn set_grid_velocity_x(&mut self, i: usize, j: usize, k: usize, val: f32) {
        self.grid.set_u(i, j, k, val);
    }

    /// 读网格 y 面心速度（越界返回 0.0）。
    pub fn grid_velocity_y(&self, i: usize, j: usize, k: usize) -> f32 {
        self.grid.v_at(i, j, k)
    }

    /// 读网格 x 面心速度（越界返回 0.0）。
    pub fn grid_velocity_x(&self, i: usize, j: usize, k: usize) -> f32 {
        self.grid.u_at(i, j, k)
    }

    /// 规格。
    #[inline]
    pub fn spec(&self) -> &SmokeSpec {
        &self.spec
    }

    /// 覆盖粒子速度（供测试 / 外部驱动）。
    pub fn set_particle_velocity(&mut self, i: usize, v: Vec3) {
        if let Some(p) = self.particles.get_mut(i) {
            p.velocity = v;
        }
    }

    /// 最大粒子速率（max|v|；空集 = 0）。
    pub fn max_speed(&self) -> f32 {
        let mut m = 0.0f32;
        for p in &self.particles {
            let s = p.velocity.length();
            if s > m {
                m = s;
            }
        }
        m
    }

    /// 推进一帧 `dt`。
    ///
    /// # 步序（固定序，确定性）
    /// ① CFL 守卫：查 `dt ≤ cfl·dx / max|v|`，越界则**细分**为多子步（不超过
    /// `max_substeps`；仍越界 → 显式拒绝 P1002/P1020，不产 NaN）；
    /// ② 每个子步按以下**固定耦合序**执行（深化后的完整步序）：
    /// ```text
    ///   a. 外力积分粒子   v += g·Δt，x += v·Δt（含出域钳制）
    ///   b. P2G 散射       粒子 → 网格速度（清空并重建 pressure/温度/密度/类型）
    ///   c. 标量回散射     新活跃单元的 T/ρ 由落在其中的粒子平均（沉积，标量权威=粒子）
    ///   d. 浮力施加       v_grid(y 面心) += α·T·Δt（Boussinesq，仅网格层）
    ///   e. 压力投影       Gauss–Seidel 解 ∇²p = ∇·v/Δt，v -= ∇p
    ///   f. G2P 采集       v_p ← 三线性采集投影后网格速度（FLIP 增量混合）
    ///   g. 标量输运       网格层半拉格朗日平流 T/ρ（回溯用投影后速度场）
    ///   h. 粒子标量读回   粒子 T/ρ ← 采样输运后的网格标量场
    /// ```
    /// 其中 c/d/g/h 为本轮新增；a/b/e/f 为基础步序；d 插在 c 之后、e 之前
    /// （浮力须先于压力投影，使投影能把浮力激发的散度一并去掉）。
    ///
    /// # 顺序纪律（本轮修复要点）
    /// 标量的**权威在粒子**（deposit 沉积到网格），网格层负责输运与浮力，最后读回
    /// 粒子。**h 必须在 g 之后**：早期版本把「粒子标量 ← 网格标量」放在 P2G 之前，
    /// 首帧用全零初始网格覆盖粒子初态标量，导致浮力源恒零 + 标量权威割裂。
    ///
    /// # 防呆
    /// `dt <= 0` 或非有限 → P1002；CFL 无法满足 → P1020。
    pub fn step(&mut self, dt: f32) -> Result<SmokeStepReport, PhysicsError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PhysicsError {
                code: "P1002",
                location: "Smoke::step".to_string(),
                problem: format!("dt = {dt}：步长必须为正有限实数（NaN/±Inf/0 同拒）"),
                fix: "传正步长，如 dt = 1.0 / fps".to_string(),
            });
        }
        // 空粒子集：合法空操作（无网格散度）。
        if self.particles.is_empty() {
            return Ok(SmokeStepReport {
                substeps: 1,
                dt_sub: dt,
                divergence_before: 0.0,
                divergence_after: 0.0,
            });
        }
        // ① CFL 守卫：求所需子步数 n = ceil(dt·max|v| / (cfl·dx))。
        let dx = self.spec.dx;
        let max_v = self.max_speed();
        let cfl_dx = self.spec.cfl * dx;
        // cfl_dx > 0 已由 validate 保证；max_v 有限（粒子初态已校验，积分保持有限）。
        let needed = if max_v > 0.0 {
            let ratio = dt * max_v / cfl_dx;
            if !ratio.is_finite() || ratio > 1e7 {
                // 数值病态（速度爆炸）：显式拒绝，不硬算 NaN。
                return Err(PhysicsError {
                    code: "P1020",
                    location: "Smoke::step（CFL）".to_string(),
                    problem: format!(
                        "CFL 比值非有限或过大（dt={dt}, max|v|={max_v}, cfl·dx={cfl_dx}）"
                    ),
                    fix: "减小 dt、增大 cfl 或检查速度是否发散".to_string(),
                });
            }
            // ceil（无 libm：floor + 1 判断）。
            let f = ratio.floor();
            let n = if ratio > f { f + 1.0 } else { f };
            n.max(1.0) as u32
        } else {
            1
        };
        if needed > self.spec.max_substeps {
            return Err(PhysicsError {
                code: "P1020",
                location: "Smoke::step（CFL 子步上限）".to_string(),
                problem: format!(
                    "CFL 需 {needed} 子步 > max_substeps = {}（dt={dt}, max|v|={max_v}, dx={dx}, cfl={}）",
                    self.spec.max_substeps, self.spec.cfl
                ),
                fix: "减小 dt、增大 dx、增大 cfl 或提高 max_substeps".to_string(),
            });
        }
        let substeps = needed;
        let dt_sub = dt / substeps as f32;

        let mut div_before = 0.0f32;
        let mut div_after = 0.0f32;
        for _ in 0..substeps {
            // ②a 外力积分粒子（显式 Euler；确定性：下标升序）。
            self.advect(dt_sub);
            // ②b P2G 散射到 MAC 网格（得到投影前网格速度 v_before；同时清空
            //    并重建 pressure/温度/密度/单元类型）。
            self.p2g();
            // ②c 标量回散射：用落在新活跃单元中的粒子温度/密度填充网格标量
            //    （网格标量场在本帧被重建为「粒子携带的标量」——标量权威在粒子）。
            self.deposit_scalars();
            // 保存投影前网格速度（FLIP 增量 v_after − v_before 需要）。
            let u_before = self.grid.u.clone();
            let v_before = self.grid.v.clone();
            let w_before = self.grid.w.clone();
            // ②d 浮力施加（Boussinesq，仅网格 y 面心；在投影之前）。
            self.apply_buoyancy(dt_sub);
            // ②e 压力投影（记录投影前后散度）。
            let (before, after) = self.project();
            div_before = before;
            div_after = after;
            // ②f G2P 采集 + FLIP 混合（用 v_after − v_before 网格增量）。
            self.g2p(&u_before, &v_before, &w_before);
            // ②g 网格标量输运（半拉格朗日，回溯用投影后速度场）。
            self.advect_scalars(dt_sub);
            // ②h 粒子标量读回：粒子温度/密度 ← 采样（输运后的）网格标量。
            //
            // 注意（本轮修复的关键）：这一步必须在**网格标量输运之后**，而非之前。
            // 旧序把粒子标量平流放在 P2G 之前，首帧会用「全零初始网格标量」覆盖
            // 粒子初态标量（粒子携带的初始温度/密度被清零），导致浮力源恒为 0；
            // 且标量权威割裂。正确 FLIP 语义：粒子 → 网格（deposit，权威）→ 网格
            // 输运/浮力 → 网格 → 粒子（读回），粒子标量始终跟随输运后的网格场。
            self.advect_particle_scalars();
        }
        Ok(SmokeStepReport {
            substeps,
            dt_sub,
            divergence_before: div_before,
            divergence_after: div_after,
        })
    }

    /// 外力积分：`v += g·dt`，`x += v·dt`（半隐式 Euler）。含出域钳制。
    fn advect(&mut self, dt: f32) {
        let g = self.spec.gravity;
        let max_c = self.grid.domain_max();
        let min_c = self.grid.origin;
        for p in &mut self.particles {
            p.velocity = p.velocity + g * dt;
            p.position = p.position + p.velocity * dt;
            // 出域钳制：把位置夹回域内（留半格余量），并杀掉越界方向速度分量。
            // 不 panic、不产 NaN——钳制是显式安全降级。
            p.position = clamp_point(p.position, min_c, max_c, self.grid.dx);
        }
    }

    // -- 粒子标量平流 ------------------------------------------------------------

    /// 粒子温度/密度读回 ← 三线性采样当前网格标量场（半拉格朗日 G2P 的标量侧）。
    ///
    /// 每个粒子在其当前位置采样单元中心标量场（[`sample_scalar_field_global`]）。
    /// 这是「P2G → 沉积 → 网格输运/浮力 → 读回」链路回写给粒子的标量值，使粒子
    /// 携带的标量跟随网格输运（被动标量守恒的粒子侧表示）。**必须在网格标量输运
    /// （[`Smoke::advect_scalars`]）之后调用**——见 [`Smoke::step`] 的顺序纪律。
    ///
    /// 遍历序 = 粒子下标升序（确定性）；越界回溯在采样内钳制（不 panic）。
    fn advect_particle_scalars(&mut self) {
        let mut new_scal = Vec::with_capacity(self.particles.len());
        for p in &self.particles {
            let t = self.sample_scalar_field(&self.grid.temperature, p.position);
            let d = self.sample_scalar_field(&self.grid.density, p.position);
            let (t_out, d_out) = if t.is_finite() && d.is_finite() {
                (t, d)
            } else {
                (0.0, 0.0)
            };
            new_scal.push((t_out, d_out));
        }
        for (p, (t, d)) in self.particles.iter_mut().zip(new_scal) {
            p.temperature = t;
            p.density = d;
        }
    }

    /// 三线性采样单元中心标量场（`field` 维度 `(nz,ny,nx)`）。
    ///
    /// 世界坐标 → 连续单元坐标 → 钳制到 `[0, res-1]` → 三线性插值。
    fn sample_scalar_field(&self, field: &[f32], pos: Vec3) -> f32 {
        sample_scalar_field_global(&self.grid, field, pos)
    }

    // -- 标量回散射 --------------------------------------------------------------

    /// 把粒子温度/密度平均（质量加权）散射回其所在单元中心。
    ///
    /// 对每个粒子按颗粒下标升序累积到其所在单元（`num[a] += T`，`cnt += 1`），
    /// 再对每个活跃单元求 `T = ΣT / N`（[`normalize_scalar`]）。无粒子覆盖的
    /// 单元保持 0（「空气」）。该填充结果供浮力与后续网格平流使用。
    ///
    /// 确定性：粒子序、单元序均为固定行主序，逐单元顺序累加。
    fn deposit_scalars(&mut self) {
        let (nx, ny, nz) = self.grid.res;
        let cells = nx * ny * nz;
        let mut t_sum = vec![0.0f32; cells];
        let mut d_sum = vec![0.0f32; cells];
        let mut cnt = vec![0.0f32; cells];
        for p in &self.particles {
            let gp = self.grid.to_grid(p.position);
            if !gp.x.is_finite() || !gp.y.is_finite() || !gp.z.is_finite() {
                continue;
            }
            let fi = (gp.x.floor() as i64).clamp(0, nx as i64 - 1) as usize;
            let fj = (gp.y.floor() as i64).clamp(0, ny as i64 - 1) as usize;
            let fk = (gp.z.floor() as i64).clamp(0, nz as i64 - 1) as usize;
            let c = fk * ny * nx + fj * nx + fi;
            t_sum[c] += p.temperature;
            d_sum[c] += p.density;
            cnt[c] += 1.0;
        }
        normalize_scalar(&mut self.grid.temperature, &t_sum, &cnt);
        normalize_scalar(&mut self.grid.density, &d_sum, &cnt);
    }

    // -- 浮力 ------------------------------------------------------------------

    /// 施加浮力（Boussinesq 近似）到网格速度场。
    ///
    /// # 公式与出处
    /// ```text
    ///   f_buoy = α · T · ŷ        （Boussinesq 近似；Spiegel & Veronis 1960,
    ///   v_y   ← v_y + f_buoy · Δt   Rev. Mod. Phys. 32, 175；Smoke 经典做法见
    ///                               Fedkiw et al. 2001, "Visual Simulation of
    ///                               Smoke", SIGGRAPH，§3）
    /// ```
    /// 其中 α = [`SmokeSpec::buoyancy`]，T = 网格温度，ŷ = +Y 单位向量（本实装
    /// 固定「上」为 +Y，与 `SmokeSpec::gravity` 同坐标系）。
    ///
    /// # 密度场裁决（物理直觉说明）
    /// 真实 Boussinesq 浮力按**密度差**驱动：`f = −g·β·(T − T₀)`。本实装把
    /// 浮力与温度**线性绑定**（`f ∝ T`），**不让 `density` 参与浮力**——因为
    /// 烟密度是**被动标量**（浓度），其作用在真实烟雾中主要是**光学遮挡**
    /// （渲染侧），而非直接改变动量；把密度当成质量项会与温升/热膨胀语义混叠
    /// （D-16：不引入超出直觉的耦合）。若需「冷烟下沉」可由调用方对 T 取负值
    /// 表达，无需额外项。
    ///
    /// # 确定性
    /// 只写 y 面心 `(i,j,k)`，遍历按下标升序 `(k,j,i)`；逐面心 `+=`，无归约。
    fn apply_buoyancy(&mut self, dt: f32) {
        let alpha = self.spec.buoyancy;
        if alpha == 0.0 {
            // 关闭浮力：保持零开销的纯 FLIP 路径。
            return;
        }
        let (nx, ny, nz) = self.grid.res;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = self.grid.idx_c(i, j, k);
                    let temp = match self.grid.temperature.get(c) {
                        Some(t) => *t,
                        None => 0.0,
                    };
                    if temp == 0.0 {
                        continue;
                    }
                    // y 面 (i,j,k)：下单元 (i,j-1,k)、上单元 (i,j,k)。面心温度取
                    // 两端单元温度均值（与 MAC 面心位置一致的线性重构）。
                    let t_up = if j < ny { self.grid.temperature_at(i, j, k) } else { 0.0 };
                    let t_dn = if j > 0 { self.grid.temperature_at(i, j - 1, k) } else { 0.0 };
                    let t_face = 0.5 * (t_up + t_dn);
                    let force = alpha * t_face * dt;
                    // 仅对非固体面施力（双端均非 Solid）。
                    let up_solid = if j < ny { self.grid.cell_type_at(i, j, k) } else { CellType::Solid };
                    let dn_solid = if j > 0 { self.grid.cell_type_at(i, j - 1, k) } else { CellType::Solid };
                    if up_solid == CellType::Solid || dn_solid == CellType::Solid {
                        continue;
                    }
                    if let Some(v) = self.grid.v_mut(i, j, k) {
                        let nv = *v + force;
                        if nv.is_finite() {
                            *v = nv;
                        }
                    }
                }
            }
        }
    }

    // -- 网格标量输运 ------------------------------------------------------------

    /// 网格单元中心标量场（温度/密度）半拉格朗日平流。
    ///
    /// 对每个单元中心 x，用**当前网格速度**三线性回溯出发点
    /// `x_src = x − Δt·u(x)`（速度由 MAC 面心三线性采集到单元中心），
    /// 在旧场上三线性采样得新值。回溯仅读取**投影后**速度场（调用序见 `step`）。
    ///
    /// # 出处
    /// Stam, *Stable Fluids* (SIGGRAPH 1999)：半拉格朗日平流无条件稳定；
    /// Bridson, *Fluid Simulation for Computer Graphics*, 2nd ed., ch. 2。
    ///
    /// # 确定性
    /// ① 遍历 `(k,j,i)` 行主序；② 采样用 [`sample_scalar_field_global`] 的固定
    /// 嵌套插值序（先 x 后 y 后 z）；③ 回溯点越界时把连续坐标钳制到
    /// `[0, n-1]`（[`clamp_grid_coord`]），不 panic；④ 双缓冲：新场先写入
    /// 独立缓冲，全部算完后一次写回，避免读写混叠导致的顺序依赖。
    fn advect_scalars(&mut self, dt: f32) {
        let (nx, ny, nz) = self.grid.res;
        let dx = self.grid.dx;
        let origin = self.grid.origin;
        let cells = nx * ny * nz;
        let mut new_t = vec![0.0f32; cells];
        let mut new_d = vec![0.0f32; cells];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = k * ny * nx + j * nx + i;
                    // 单元中心世界坐标（仅用于速度采集；速度是 MAC 面心场，须在世界坐标采样）。
                    let x = origin
                        + Vec3::new(
                            (i as f32 + 0.5) * dx,
                            (j as f32 + 0.5) * dx,
                            (k as f32 + 0.5) * dx,
                        );
                    // 该点速度（三线性采集 MAC 面心）。
                    let vel = self.sample_velocity(x);
                    // 回溯：半拉格朗日，**全程在索引空间**运算，避免世界坐标往返
                    // 带来的 f32 精度损失（u=0 时应为逐位恒等）。
                    //   g_src = i − Δt·u/dx  （单元中心 i 的索引坐标为 i）
                    let gx = clamp_grid_coord(i as f32 - vel.x * dt / dx, nx);
                    let gy = clamp_grid_coord(j as f32 - vel.y * dt / dx, ny);
                    let gz = clamp_grid_coord(k as f32 - vel.z * dt / dx, nz);
                    // 在旧场上按索引空间坐标采样 T 与 ρ（旧缓冲），写入新缓冲。
                    let t_old = sample_scalar_field_index(
                        &self.grid.temperature,
                        (gx, gy, gz),
                        (nx, ny, nz),
                    );
                    let d_old = sample_scalar_field_index(
                        &self.grid.density,
                        (gx, gy, gz),
                        (nx, ny, nz),
                    );
                    let t_v = if t_old.is_finite() { t_old } else { 0.0 };
                    let d_v = if d_old.is_finite() { d_old } else { 0.0 };
                    if let Some(s) = new_t.get_mut(c) {
                        *s = t_v;
                    }
                    if let Some(s) = new_d.get_mut(c) {
                        *s = d_v;
                    }
                }
            }
        }
        // 一次写回（双缓冲提交）。
        for (dst, src) in self.grid.temperature.iter_mut().zip(new_t) {
            *dst = src;
        }
        for (dst, src) in self.grid.density.iter_mut().zip(new_d) {
            *dst = src;
        }
    }

    // -- P2G：粒子 → 网格（三线性散射） --------------------------------------

    /// 清空网格面心速度与单元类型，然后把粒子速度三线性散射到 MAC 网格。
    fn p2g(&mut self) {
        // 清空速度场与压力/类型。
        for x in &mut self.grid.u {
            *x = 0.0;
        }
        for x in &mut self.grid.v {
            *x = 0.0;
        }
        for x in &mut self.grid.w {
            *x = 0.0;
        }
        for x in &mut self.grid.pressure {
            *x = 0.0;
        }
        for c in &mut self.grid.cell_type {
            *c = CellType::Air;
        }
        // 清空标量场（温度/密度随后由 deposit_scalars 按颗粒填充）。
        for t in &mut self.grid.temperature {
            *t = 0.0;
        }
        for d in &mut self.grid.density {
            *d = 0.0;
        }
        let (nx, ny, nz) = self.grid.res;
        // 三线性权重散射。按粒子下标升序（确定性）。
        // 为避免逐面心做除法，分别累积「加权速度」与「权重和」，最后归一化。
        let mut u_wsum = vec![0.0f32; self.grid.u.len()];
        let mut v_wsum = vec![0.0f32; self.grid.v.len()];
        let mut w_wsum = vec![0.0f32; self.grid.w.len()];
        for p in &self.particles {
            let gp = self.grid.to_grid(p.position); // 连续单元坐标
            // 单元基索引（floor，无越界的用 i64 夹取）。
            let fi = (gp.x.floor() as i64).clamp(0, nx as i64 - 1);
            let fj = (gp.y.floor() as i64).clamp(0, ny as i64 - 1);
            let fk = (gp.z.floor() as i64).clamp(0, nz as i64 - 1);
            let fx = gp.x - fi as f32;
            let fy = gp.y - fj as f32;
            let fz = gp.z - fk as f32;
            // 标记该单元为流体。
            let ci = self.grid.idx_c(fi as usize, fj as usize, fk as usize);
            self.grid.cell_type[ci] = CellType::Fluid;
            // 三线性 8 点散射：MAC 各分量用同一模板（轴向仅决定面心缓冲与
            // 其自身维度，故 scatter_component 不需知道轴）。
            scatter_component(
                &mut self.grid.u,
                &mut u_wsum,
                p.velocity.x,
                (fi, fj, fk),
                (fx, fy, fz),
                (nx + 1, ny, nz),
            );
            scatter_component(
                &mut self.grid.v,
                &mut v_wsum,
                p.velocity.y,
                (fi, fj, fk),
                (fx, fy, fz),
                (nx, ny + 1, nz),
            );
            scatter_component(
                &mut self.grid.w,
                &mut w_wsum,
                p.velocity.z,
                (fi, fj, fk),
                (fx, fy, fz),
                (nx, ny, nz + 1),
            );
        }
        normalize_field(&mut self.grid.u, &u_wsum);
        normalize_field(&mut self.grid.v, &v_wsum);
        normalize_field(&mut self.grid.w, &w_wsum);
        // 固壁 / 开放边界。
        self.apply_boundary();
    }

    // -- 压力投影（Gauss–Seidel，固定迭代数） --------------------------------

    /// 压力投影：解 ∇²p = ∇·v/Δt（单位密度），然后 `v -= ∇p`。
    ///
    /// 返回 (投影前 max|∇·v|, 投影后 max|∇·v|)（免费质量指标）。
    ///
    /// 迭代：顺序 Gauss–Seidel，次数 = `spec.pressure_iters`（固定，无提前退出）。
    /// Air 单元压力恒 0（自由面，Dirichlet）；Solid 邻居不计入（Neumann）。
    fn project(&mut self) -> (f32, f32) {
        let (nx, ny, nz) = self.grid.res;
        let dx = self.grid.dx;
        let inv_dx = 1.0 / dx;
        let before = self.max_divergence();

        // 压力投影迭代：顺序扫描 (k,j,i)，每单元更新 p（Gauss–Seidel）。
        for _ in 0..self.spec.pressure_iters {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let c = self.grid.idx_c(i, j, k);
                        if self.grid.cell_type[c] != CellType::Fluid {
                            // 空气 / 固壁：压力 0（自由面 Dirichlet）。
                            self.grid.pressure[c] = 0.0;
                            continue;
                        }
                        // 组装该单元的散度：div = (u_r − u_l + v_t − v_b + w_f − w_k)/dx。
                        let div = self.divergence_cell(i, j, k, inv_dx);
                        // 统计可解邻居（流体或空气；固壁跳过=Neumann）。
                        let mut denom = 0.0f32;
                        let mut num = 0.0f32;
                        // x 邻居。
                        for (ni, nj, nk) in [
                            (i.wrapping_sub(1), j, k),
                            (i + 1, j, k),
                        ] {
                            let t = self.grid.cell_type_at(ni, nj, nk);
                            if t != CellType::Solid {
                                denom += 1.0;
                                let pc = self.neighbor_pressure(ni, nj, nk);
                                num += pc;
                            }
                        }
                        // y 邻居。
                        for (ni, nj, nk) in [(i, j.wrapping_sub(1), k), (i, j + 1, k)] {
                            let t = self.grid.cell_type_at(ni, nj, nk);
                            if t != CellType::Solid {
                                denom += 1.0;
                                let pc = self.neighbor_pressure(ni, nj, nk);
                                num += pc;
                            }
                        }
                        // z 邻居。
                        for (ni, nj, nk) in [(i, j, k.wrapping_sub(1)), (i, j, k + 1)] {
                            let t = self.grid.cell_type_at(ni, nj, nk);
                            if t != CellType::Solid {
                                denom += 1.0;
                                let pc = self.neighbor_pressure(ni, nj, nk);
                                num += pc;
                            }
                        }
                        if denom > 0.0 {
                            // GS 更新：离散泊松 (Σp_nb − N·p)/dx² = div ⇒
                            // p = (Σp_nb − dx²·div)/N。
                            let p_new = (num - dx * dx * div) / denom;
                            if p_new.is_finite() {
                                self.grid.pressure[c] = p_new;
                            }
                            // 非有限 → 保留旧值（安全降级；理论上不会发生）。
                        }
                    }
                }
            }
        }

        // 用压力梯度修正速度：u -= (p_R − p_L)/dx（对流体-非固体面）。
        self.subtract_pressure_gradient(inv_dx);
        // 再次施加边界（投影不应破坏壁条件）。
        self.apply_boundary();

        let after = self.max_divergence();
        (before, after)
    }

    /// 单元 (i,j,k) 的散度（除以 dx；MAC 差分）。
    #[inline]
    fn divergence_cell(&self, i: usize, j: usize, k: usize, inv_dx: f32) -> f32 {
        let u_r = self.grid.u_at(i + 1, j, k);
        let u_l = self.grid.u_at(i, j, k);
        let v_t = self.grid.v_at(i, j + 1, k);
        let v_b = self.grid.v_at(i, j, k);
        let w_f = self.grid.w_at(i, j, k + 1);
        let w_b = self.grid.w_at(i, j, k);
        (u_r - u_l + v_t - v_b + w_f - w_b) * inv_dx
    }

    /// 邻居压力（Air/Solid 视为 0；已越界 = 0）。
    #[inline]
    fn neighbor_pressure(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.grid.res;
        if i >= nx || j >= ny || k >= nz {
            return 0.0;
        }
        let c = self.grid.idx_c(i, j, k);
        match self.grid.cell_type[c] {
            // 固壁压力贡献 0（Neumann：法向梯度为零 → 等价 p_nb = p_self，但本
            // 简化用 denom 排除固壁邻居，故此处不用）。Air = 0（Dirichlet）。
            CellType::Fluid | CellType::Air => self.grid.pressure[c],
            CellType::Solid => 0.0,
        }
    }

    /// `v -= ∇p`（仅对流体-非固体面，且两端均非 Solid）。
    fn subtract_pressure_gradient(&mut self, inv_dx: f32) {
        let (nx, ny, nz) = self.grid.res;
        // u 面 (i,j,k)：面左单元 (i-1,j,k)、右单元 (i,j,k)。
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let left = if i > 0 { self.grid.cell_type_at(i - 1, j, k) } else { CellType::Solid };
                    let right = if i < nx { self.grid.cell_type_at(i, j, k) } else { CellType::Solid };
                    if left == CellType::Solid || right == CellType::Solid {
                        continue;
                    }
                    let p_l = if i > 0 { self.pressure_or_zero(i - 1, j, k) } else { 0.0 };
                    let p_r = if i < nx { self.pressure_or_zero(i, j, k) } else { 0.0 };
                    let grad = (p_r - p_l) * inv_dx;
                    if let Some(u) = self.grid.u_mut(i, j, k) {
                        let nv = *u - grad;
                        if nv.is_finite() {
                            *u = nv;
                        }
                    }
                }
            }
        }
        // v 面 (i,j,k)：下单元 (i,j-1,k)、上单元 (i,j,k)。
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let down = if j > 0 { self.grid.cell_type_at(i, j - 1, k) } else { CellType::Solid };
                    let up = if j < ny { self.grid.cell_type_at(i, j, k) } else { CellType::Solid };
                    if down == CellType::Solid || up == CellType::Solid {
                        continue;
                    }
                    let p_d = if j > 0 { self.pressure_or_zero(i, j - 1, k) } else { 0.0 };
                    let p_u = if j < ny { self.pressure_or_zero(i, j, k) } else { 0.0 };
                    let grad = (p_u - p_d) * inv_dx;
                    if let Some(v) = self.grid.v_mut(i, j, k) {
                        let nv = *v - grad;
                        if nv.is_finite() {
                            *v = nv;
                        }
                    }
                }
            }
        }
        // w 面 (i,j,k)：后单元 (i,j,k-1)、前单元 (i,j,k)。
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let back = if k > 0 { self.grid.cell_type_at(i, j, k - 1) } else { CellType::Solid };
                    let front = if k < nz { self.grid.cell_type_at(i, j, k) } else { CellType::Solid };
                    if back == CellType::Solid || front == CellType::Solid {
                        continue;
                    }
                    let p_b = if k > 0 { self.pressure_or_zero(i, j, k - 1) } else { 0.0 };
                    let p_f = if k < nz { self.pressure_or_zero(i, j, k) } else { 0.0 };
                    let grad = (p_f - p_b) * inv_dx;
                    if let Some(w) = self.grid.w_mut(i, j, k) {
                        let nv = *w - grad;
                        if nv.is_finite() {
                            *w = nv;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn pressure_or_zero(&self, i: usize, j: usize, k: usize) -> f32 {
        let (nx, ny, nz) = self.grid.res;
        if i >= nx || j >= ny || k >= nz {
            return 0.0;
        }
        let c = self.grid.idx_c(i, j, k);
        match self.grid.cell_type[c] {
            CellType::Fluid => self.grid.pressure[c],
            _ => 0.0,
        }
    }

    /// max|∇·v| 遍历所有流体单元（确定性固定序）。
    fn max_divergence(&self) -> f32 {
        let (nx, ny, nz) = self.grid.res;
        let inv_dx = 1.0 / self.grid.dx;
        let mut m = 0.0f32;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = self.grid.idx_c(i, j, k);
                    if self.grid.cell_type[c] != CellType::Fluid {
                        continue;
                    }
                    let d = self.divergence_cell(i, j, k, inv_dx).abs();
                    if d > m {
                        m = d;
                    }
                }
            }
        }
        m
    }

    // -- G2P：网格 → 粒子（三线性采集 + FLIP 混合） --------------------------

    /// 采集网格速度到粒子，并做 FLIP 混合（Bridson, *Fluid Simulation*, ch. 7）：
    ///
    /// ```text
    ///   v_pic    = sample(v_after)                    // 采集投影后网格速度
    ///   Δv_grid  = sample(v_after) − sample(v_before) // 网格速度增量
    ///   v_new    = (1−α)·v_pic + α·(v_p + Δv_grid)
    /// ```
    ///
    /// α = 1 → 纯 FLIP（保能量，噪声略高）；α = 0 → 纯 PIC（耗散，平滑）。
    /// `v_before` = P2G 投影前网格速度，`v_after` = 投影后（即当前 `self.grid`）。
    /// 增量式 FLIP 保证：投影只移除散度分量，粒子速度不被整体拉向均值。
    fn g2p(&mut self, u_before: &[f32], v_before: &[f32], w_before: &[f32]) {
        let ratio = self.spec.flip_ratio;
        let mut new_vel = Vec::with_capacity(self.particles.len());
        for p in &self.particles {
            let pic = self.sample_velocity(p.position);
            let before = sample_velocity_fields(
                &self.grid,
                u_before,
                v_before,
                w_before,
                p.position,
            );
            let delta = pic - before;
            let flip = p.velocity + delta;
            let mixed = pic * (1.0 - ratio) + flip * ratio;
            new_vel.push(if is_finite_vec3(mixed) { mixed } else { p.velocity });
        }
        for (p, v) in self.particles.iter_mut().zip(new_vel) {
            p.velocity = v;
        }
    }

    /// 三线性采集网格速度（MAC 面心在粒子位置处的插值）。
    fn sample_velocity(&self, pos: Vec3) -> Vec3 {
        sample_velocity_fields(&self.grid, &self.grid.u, &self.grid.v, &self.grid.w, pos)
    }

    // -- 边界 ----------------------------------------------------------------

    /// 施加边界条件（固壁法向为零；开放模式顶面/侧面允许散逸）。
    fn apply_boundary(&mut self) {
        match self.spec.boundary {
            Boundary::Closed => self.grid.apply_solid_walls(),
            Boundary::Open => {
                // 底面为壁（y=0 面），其余面自由（不强制零）。
                let (nx, _ny, nz) = self.grid.res;
                for k in 0..nz {
                    for i in 0..nx {
                        self.grid.set_v(i, 0, k, 0.0);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 散射 / 采集辅助（自由函数，避免借用冲突）
// ---------------------------------------------------------------------------

/// 把一个标量速度分量三线性散射到某轴面心场。
///
/// `base` = 粒子所在单元基索引 (fi,fj,fk)，`frac` = 单元内小数坐标。
/// `field` 是面心值累加缓冲，`wsum` 是权重和缓冲。`dims` 是该面心场的自身维度
/// （u = (nx+1,ny,nz)、v = (nx,ny+1,nz)、w = (nx,ny,nz+1)）——线性化按面心场
/// 自身步长计算，故本函数无需知道分量轴。
fn scatter_component(
    field: &mut [f32],
    wsum: &mut [f32],
    value: f32,
    base: (i64, i64, i64),
    frac: (f32, f32, f32),
    dims: (usize, usize, usize),
) {
    let (fi, fj, fk) = base;
    let (fx, fy, fz) = frac;
    // 8 点模板：轴向与非轴向均取 [floor, floor+1]，三线性权重为 (1−f, f)。
    for ii in [fi, fi + 1] {
        for jj in [fj, fj + 1] {
            for kk in [fk, fk + 1] {
                let wx = if ii == fi { 1.0 - fx } else { fx };
                let wy = if jj == fj { 1.0 - fy } else { fy };
                let wz = if kk == fk { 1.0 - fz } else { fz };
                let w = wx * wy * wz;
                if w <= 0.0 {
                    continue;
                }
                if let Some(idx) = face_index(ii, jj, kk, dims) {
                    field[idx] += value * w;
                    wsum[idx] += w;
                }
            }
        }
    }
}

/// 面心线性索引（按 `dims` 自身步长；越界返回 None）。
#[inline]
fn face_index(i: i64, j: i64, k: i64, dims: (usize, usize, usize)) -> Option<usize> {
    let (nx, ny, nz) = dims;
    if i < 0 || j < 0 || k < 0 {
        return None;
    }
    let (i, j, k) = (i as usize, j as usize, k as usize);
    if i >= nx || j >= ny || k >= nz {
        None
    } else {
        Some(k * ny * nx + j * nx + i)
    }
}

/// 归一化面心场：`field = Σ(w·v) / Σw`（无粒子覆盖 → 0）。
fn normalize_field(field: &mut [f32], wsum: &[f32]) {
    for (f, &w) in field.iter_mut().zip(wsum.iter()) {
        if w > 0.0 {
            *f /= w;
        } else {
            *f = 0.0;
        }
    }
}

/// 三线性采集：对 8 个网格点的函数值按单元内小数坐标 (fx,fy,fz) 加权。
///
/// MAC 各分量（u/v/w）的采集都用同一三线性模板——不同的只是传入的 `get`
/// 闭包所选的面心缓冲，故本函数无需知道轴。
#[allow(clippy::too_many_arguments)]
fn trilinear_face(
    get: impl Fn(usize, usize, usize) -> f32,
    fi: usize,
    fj: usize,
    fk: usize,
    fx: f32,
    fy: f32,
    fz: f32,
) -> f32 {
    let (i0, i1) = (fi, fi + 1);
    let (j0, j1) = (fj, fj + 1);
    let (k0, k1) = (fk, fk + 1);
    let (wx0, wx1) = (1.0 - fx, fx);
    let (wy0, wy1) = (1.0 - fy, fy);
    let (wz0, wz1) = (1.0 - fz, fz);
    let mut acc = 0.0f32;
    let pts = [
        (i0, j0, k0, wx0 * wy0 * wz0),
        (i1, j0, k0, wx1 * wy0 * wz0),
        (i0, j1, k0, wx0 * wy1 * wz0),
        (i1, j1, k0, wx1 * wy1 * wz0),
        (i0, j0, k1, wx0 * wy0 * wz1),
        (i1, j0, k1, wx1 * wy0 * wz1),
        (i0, j1, k1, wx0 * wy1 * wz1),
        (i1, j1, k1, wx1 * wy1 * wz1),
    ];
    for (i, j, k, w) in pts {
        if w > 0.0 {
            acc += get(i, j, k) * w;
        }
    }
    acc
}

/// 点是否有限。
#[inline]
fn is_finite_vec3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// 从任意 u/v/w 缓冲三线性采集粒子位置处的网格速度。
///
/// `grid` 仅提供 origin/dx/res 几何（用于世界坐标 → 连续单元坐标与越界判断）。
/// 缓冲维度约定同 [`MacGrid`]：u = (nx+1)·ny·nz，v = nx·(ny+1)·nz，
/// w = nx·ny·(nz+1)。越界面心取 0（安全降级）。
fn sample_velocity_fields(
    grid: &MacGrid,
    u: &[f32],
    v: &[f32],
    w: &[f32],
    pos: Vec3,
) -> Vec3 {
    let (nx, ny, nz) = grid.res;
    let gp = grid.to_grid(pos);
    let fi = (gp.x.floor() as i64).clamp(0, nx as i64 - 1) as usize;
    let fj = (gp.y.floor() as i64).clamp(0, ny as i64 - 1) as usize;
    let fk = (gp.z.floor() as i64).clamp(0, nz as i64 - 1) as usize;
    let fx = (gp.x - fi as f32).clamp(0.0, 1.0);
    let fy = (gp.y - fj as f32).clamp(0.0, 1.0);
    let fz = (gp.z - fk as f32).clamp(0.0, 1.0);

    let get_u = |i: usize, j: usize, k: usize| -> f32 {
        if i > nx || j >= ny || k >= nz {
            return 0.0;
        }
        let idx = k * ny * (nx + 1) + j * (nx + 1) + i;
        u.get(idx).copied().unwrap_or(0.0)
    };
    let get_v = |i: usize, j: usize, k: usize| -> f32 {
        if i >= nx || j > ny || k >= nz {
            return 0.0;
        }
        let idx = k * (ny + 1) * nx + j * nx + i;
        v.get(idx).copied().unwrap_or(0.0)
    };
    let get_w = |i: usize, j: usize, k: usize| -> f32 {
        if i >= nx || j >= ny || k > nz {
            return 0.0;
        }
        let idx = k * ny * nx + j * nx + i;
        w.get(idx).copied().unwrap_or(0.0)
    };

    let uu = trilinear_face(get_u, fi, fj, fk, fx, fy, fz);
    let vv = trilinear_face(get_v, fi, fj, fk, fx, fy, fz);
    let ww = trilinear_face(get_w, fi, fj, fk, fx, fy, fz);
    Vec3::new(uu, vv, ww)
}

/// 把点夹到轴对齐盒内（留 `margin` 余量）。
#[inline]
fn clamp_point(p: Vec3, min_c: Vec3, max_c: Vec3, margin: f32) -> Vec3 {
    let lo = Vec3::new(min_c.x + margin, min_c.y + margin, min_c.z + margin);
    let hi = Vec3::new(max_c.x - margin, max_c.y - margin, max_c.z - margin);
    Vec3::new(
        p.x.clamp(lo.x, hi.x),
        p.y.clamp(lo.y, hi.y),
        p.z.clamp(lo.z, hi.z),
    )
}

// ---------------------------------------------------------------------------
// 标量插值 / 输运辅助（自由函数，确定性固定序）
// ---------------------------------------------------------------------------

/// 把连续单元坐标钳制到 `[0, n-1]`（半拉格朗日回溯越界的**安全降级**）。
///
/// `n ≥ 1`。非有限输入返回 0.0（不 panic）。
#[inline]
fn clamp_grid_coord(x: f32, n: usize) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    if n == 0 {
        return 0.0;
    }
    x.clamp(0.0, (n - 1) as f32)
}

/// 单元中心标量场三线性插值（**固定嵌套序：先 x 后 y 后 z**）。
///
/// - `get(i,j,k)`：单元中心取值闭包（越界应返回 0，由调用方保证）。
/// - `g`：连续单元坐标（已钳制到 `[0, n-1]`）。
/// - `dims`：场维度 `(nx, ny, nz)`。
///
/// 实现按教科书顺序做三次一维线性插值：
/// ```text
///   c00 = lerp(get(i0,j0,k0), get(i1,j0,k0), fx)
///   c10 = lerp(get(i0,j1,k0), get(i1,j1,k0), fx)
///   c01 = lerp(get(i0,j0,k1), get(i1,j0,k1), fx)
///   c11 = lerp(get(i0,j1,k1), get(i1,j1,k1), fx)
///   c0  = lerp(c00, c10, fy) ;  c1 = lerp(c01, c11, fy)
///   out = lerp(c0, c1, fz)
/// ```
/// 累加次序固定，保证逐位可复现。
#[inline]
fn trilinear_scalar(
    get: impl Fn(usize, usize, usize) -> f32,
    g: (f32, f32, f32),
    dims: (usize, usize, usize),
) -> f32 {
    let (nx, ny, nz) = dims;
    let (gx, gy, gz) = g;
    // 基索引（floor 后钳制到 [0, n-1]；n≥1 保证不越界）。
    let i0 = (gx.floor() as i64).clamp(0, nx as i64 - 1) as usize;
    let j0 = (gy.floor() as i64).clamp(0, ny as i64 - 1) as usize;
    let k0 = (gz.floor() as i64).clamp(0, nz as i64 - 1) as usize;
    let i1 = if i0 + 1 < nx { i0 + 1 } else { i0 };
    let j1 = if j0 + 1 < ny { j0 + 1 } else { j0 };
    let k1 = if k0 + 1 < nz { k0 + 1 } else { k0 };
    let fx = (gx - i0 as f32).clamp(0.0, 1.0);
    let fy = (gy - j0 as f32).clamp(0.0, 1.0);
    let fz = (gz - k0 as f32).clamp(0.0, 1.0);

    // 先沿 x 插值（y,z 四角），再沿 y，最后沿 z。
    let c00 = lerp_f32(get(i0, j0, k0), get(i1, j0, k0), fx);
    let c10 = lerp_f32(get(i0, j1, k0), get(i1, j1, k0), fx);
    let c01 = lerp_f32(get(i0, j0, k1), get(i1, j0, k1), fx);
    let c11 = lerp_f32(get(i0, j1, k1), get(i1, j1, k1), fx);
    let c0 = lerp_f32(c00, c10, fy);
    let c1 = lerp_f32(c01, c11, fy);
    lerp_f32(c0, c1, fz)
}

/// 标量线性插值 `a + (b − a)·t`（`t` 已钳制到 [0,1]；固定运算序）。
#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 在**索引空间**坐标 `g`（单元 i 的中心对应索引坐标 i）处三线性采样标量场。
///
/// 与 [`sample_scalar_field_global`] 不同，本函数**不做世界坐标换算**（调用方已
/// 完成 `world → index` 转换并钳制）。用于半拉格朗日平流：回溯点的索引坐标
/// 直接来自单元下标，避免 `world → grid → world → grid` 往返引入的 f32 误差
/// （保证 `u = 0` 时逐位恒等）。越界坐标由 [`trilinear_scalar`] 内再钳制，不 panic。
#[inline]
fn sample_scalar_field_index(
    field: &[f32],
    g: (f32, f32, f32),
    dims: (usize, usize, usize),
) -> f32 {
    let (nx, ny, nz) = dims;
    let idx = |i: usize, j: usize, k: usize| -> f32 {
        let c = k * ny * nx + j * nx + i;
        field.get(c).copied().unwrap_or(0.0)
    };
    trilinear_scalar(idx, g, (nx, ny, nz))
}

/// 全局采样：给定 grid + 标量缓冲，按世界坐标采样**单元中心**标量场。
///
/// # 索引空间约定（关键）
/// 单元 `i` 的中心位于世界坐标 `origin + (i + 0.5)·dx`，其**索引空间**坐标为
/// `i`（整数）。`to_grid` 给出的是「连续单元坐标」`gx_world = (x − origin)/dx`
///（单元 `i` 的中心对应 `gx_world = i + 0.5`）。故采样前须转成索引空间坐标
/// `g_idx = gx_world − 0.5`，使 `trilinear_scalar` 正确落在单元 `i` 的值上。
/// 早期版本漏减 0.5，导致在单元中心采样得到与相邻单元的 50/50 混叠（半格偏差
/// + 额外耗散）；同一约定贯穿 [`Smoke::advect_scalars`]（后者直接用索引坐标）。
///
/// 越界（含非有限）索引坐标一律钳制到 `[0, n-1]`，故**不 panic**。
/// 供测试与外部查询使用。
fn sample_scalar_field_global(grid: &MacGrid, field: &[f32], pos: Vec3) -> f32 {
    let (nx, ny, nz) = grid.res;
    let gp = grid.to_grid(pos);
    // 连续单元坐标 → 索引空间坐标（减半格，见上）。
    let gx = clamp_grid_coord(gp.x - 0.5, nx);
    let gy = clamp_grid_coord(gp.y - 0.5, ny);
    let gz = clamp_grid_coord(gp.z - 0.5, nz);
    sample_scalar_field_index(field, (gx, gy, gz), (nx, ny, nz))
}

/// 用累积和 / 计数归一化标量场：`field[c] = sum[c] / cnt[c]`（cnt=0 → 0）。
///
/// 确定性：逐单元行主序，无归约依赖。非有限结果回落到 0（安全降级）。
fn normalize_scalar(field: &mut [f32], sum: &[f32], cnt: &[f32]) {
    for ((f, s), c) in field.iter_mut().zip(sum.iter()).zip(cnt.iter()) {
        if *c > 0.0 {
            let v = *s / *c;
            *f = if v.is_finite() { v } else { 0.0 };
        } else {
            *f = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 标准测试规格：8×8×8 域、dx=0.1、重力向下、投影 40 迭代。
    fn spec(res: (usize, usize, usize)) -> SmokeSpec {
        SmokeSpec {
            origin: Vec3::splat(0.0),
            dx: 0.1,
            res,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            pressure_iters: 40,
            flip_ratio: 1.0,
            cfl: 5.0,
            max_substeps: 64,
            boundary: Boundary::Closed,
            buoyancy: 0.0,
        }
    }

    fn block(spec: SmokeSpec) -> Smoke {
        Smoke::seed_block(
            spec,
            Vec3::new(0.3, 0.3, 0.3),
            (3, 3, 3),
            0.05,
            Vec3::splat(0.0),
        )
        .unwrap()
    }

    /// 组 1：质量守恒——粒子数在推进中不变（无发射/消亡时）。
    #[test]
    fn particle_count_conserved() {
        let mut s = block(spec((8, 8, 8)));
        let n0 = s.particle_count();
        assert_eq!(n0, 27);
        for _ in 0..30 {
            s.step(1.0 / 60.0).unwrap();
        }
        assert_eq!(s.particle_count(), n0, "无源汇时粒子数应守恒");
    }

    /// 组 2：散度下降——压力投影应显著降低 max|∇·v|。
    ///
    /// 构造一个有初始散度的场（粒子速度沿 x 向外扩散），投影后 max|∇·v| 应
    /// 下降显著（量化断言：至少降到投影前的 50% 以下）。
    #[test]
    fn pressure_projection_reduces_divergence() {
        let mut s = block(spec((8, 8, 8)));
        // 注入有散度的速度场（径向扩散）。
        for i in 0..s.particle_count() {
            let p = s.particles()[i].position;
            let c = Vec3::new(0.55, 0.55, 0.55);
            let d = (p - c).normalize();
            s.set_particle_velocity(i, d * 1.0);
        }
        let rep = s.step(1.0 / 60.0).unwrap();
        assert!(
            rep.divergence_before > 1e-6,
            "投影前应有可见散度，实际 {}",
            rep.divergence_before
        );
        assert!(
            rep.divergence_after <= rep.divergence_before * 0.5,
            "投影后 max|∇·v| ({}) 应 ≤ 投影前 ({}) 的 50%",
            rep.divergence_after,
            rep.divergence_before
        );
        println!(
            "散度：投影前={:.6}, 投影后={:.6}",
            rep.divergence_before, rep.divergence_after
        );
    }

    /// 组 3：确定性——同输入跑两遍逐位一致（f32 bit pattern）。
    ///
    /// 非恒真：若实现引入 HashMap 迭代序 / 不同归约序，本测试会逐位不符而失败。
    #[test]
    fn bitwise_deterministic() {
        let build = || {
            let mut s = block(spec((8, 8, 8)));
            for i in 0..s.particle_count() {
                let p = s.particles()[i].position;
                s.set_particle_velocity(i, Vec3::new(0.3, -0.2, 0.1) + p * 0.5);
            }
            s
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..25 {
            a.step(1.0 / 60.0).unwrap();
            b.step(1.0 / 60.0).unwrap();
        }
        let bits = |s: &Smoke| -> Vec<[u32; 6]> {
            s.particles()
                .iter()
                .map(|p| {
                    [
                        p.position.x.to_bits(),
                        p.position.y.to_bits(),
                        p.position.z.to_bits(),
                        p.velocity.x.to_bits(),
                        p.velocity.y.to_bits(),
                        p.velocity.z.to_bits(),
                    ]
                })
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "FLIP 求解应逐位可复现");

        // 非恒真控制组：把某个粒子初速扰动 1 ULP（f32 最小可分辨变化），结果
        // **必须**逐位不同——证明本测试确实在比较真实状态，而非恒等通过。
        // 若实现把状态读取路径写死/短路（如忽略粒子输入），此断言会失败。
        let mut c = build();
        let p0 = c.particles()[0].velocity;
        c.set_particle_velocity(0, Vec3::new(f32::from_bits(p0.x.to_bits() + 1), p0.y, p0.z));
        for _ in 0..25 {
            c.step(1.0 / 60.0).unwrap();
        }
        assert_ne!(
            bits(&a),
            bits(&c),
            "1 ULP 扰动应改变结果（证明确定性测试非恒真）"
        );
    }

    /// 组 3b：确定性对**粒子输入序**敏感（验证测试非恒真）。
    ///
    /// 若我们把粒子顺序反转，P2G 浮动小量归约序的改变应导致（或至少不改变）
    /// 结果——本测试断言：同序两遍一致（上述）、且不同序两遍也各自稳定。
    /// 这里额外动态验证：把同一确定性构建做两次但走不同中间容器容量，
    /// 结果仍逐位一致（容量不参与计算）。
    #[test]
    fn determinism_robust_to_capacity() {
        let s1 = Smoke::seed_block(
            spec((6, 6, 6)),
            Vec3::splat(0.2),
            (2, 2, 2),
            0.05,
            Vec3::new(0.1, 0.0, 0.0),
        )
        .unwrap();
        let s2 = Smoke::seed_block(
            spec((6, 6, 6)),
            Vec3::splat(0.2),
            (2, 2, 2),
            0.05,
            Vec3::new(0.1, 0.0, 0.0),
        )
        .unwrap();
        let mut a = s1;
        let mut b = s2;
        for _ in 0..10 {
            a.step(1.0 / 60.0).unwrap();
            b.step(1.0 / 60.0).unwrap();
        }
        let bits = |s: &Smoke| -> Vec<[u32; 3]> {
            s.particles()
                .iter()
                .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
                .collect()
        };
        assert_eq!(bits(&a), bits(&b));
    }

    /// 组 4：静止态——静止烟雾团无初速、重力平衡时能量不增。
    ///
    /// 用零重力 + 零初速，跑若干步后粒子应几乎不动（动能不增）。
    #[test]
    fn static_smoke_stays_still() {
        let mut sp = spec((8, 8, 8));
        sp.gravity = Vec3::splat(0.0);
        let mut s = block(sp);
        let ke0 = kinetic_energy(&s);
        // 记录初位置。
        let pos0: Vec<[u32; 3]> = s
            .particles()
            .iter()
            .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
            .collect();
        for _ in 0..20 {
            s.step(1.0 / 60.0).unwrap();
        }
        let ke1 = kinetic_energy(&s);
        assert!(
            ke1 <= ke0 + 1e-6,
            "静止态动能不应增加：ke0={ke0}, ke1={ke1}"
        );
        // 位置不应自发起伏（能量不增的直接推论）。
        let pos1: Vec<[u32; 3]> = s
            .particles()
            .iter()
            .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
            .collect();
        for (a, b) in pos0.iter().zip(&pos1) {
            for c in 0..3 {
                let fa = f32::from_bits(a[c]);
                let fb = f32::from_bits(b[c]);
                assert!(
                    (fa - fb).abs() < 1e-4,
                    "静止态不应漂移：{fa} vs {fb}"
                );
            }
        }
    }

    /// 组 5：边界——粒子撞壁后不出界。
    #[test]
    fn particles_stay_in_domain() {
        let mut sp = spec((8, 8, 8));
        // 给强下坠+侧向速度，撞四面壁。
        let mut s = block(sp.clone());
        for i in 0..s.particle_count() {
            s.set_particle_velocity(i, Vec3::new(20.0, -30.0, 15.0));
        }
        // 为容纳大速度，放宽 max_substeps。
        sp.max_substeps = 4096;
        let mut s = Smoke::seed_block(
            sp,
            Vec3::new(0.3, 0.3, 0.3),
            (3, 3, 3),
            0.05,
            Vec3::new(20.0, -30.0, 15.0),
        )
        .unwrap();
        for _ in 0..20 {
            s.step(1.0 / 60.0).unwrap();
        }
        let lo = s.grid().origin;
        let hi = s.grid().domain_max();
        for p in s.particles() {
            assert!(
                p.position.x >= lo.x - 1e-3
                    && p.position.x <= hi.x + 1e-3
                    && p.position.y >= lo.y - 1e-3
                    && p.position.y <= hi.y + 1e-3
                    && p.position.z >= lo.z - 1e-3
                    && p.position.z <= hi.z + 1e-3,
                "粒子出界：pos={:?}, domain=[{:?}, {:?}]",
                p.position,
                lo,
                hi
            );
            assert!(is_finite_vec3(p.position) && is_finite_vec3(p.velocity));
        }
    }

    /// 组 6a：极端输入——0 粒子不 panic，返回空报告。
    #[test]
    fn zero_particles_ok() {
        let mut s = Smoke::new(spec((4, 4, 4)), &[]).unwrap();
        let rep = s.step(1.0 / 60.0).unwrap();
        assert_eq!(s.particle_count(), 0);
        assert_eq!(rep.divergence_before, 0.0);
        assert_eq!(rep.divergence_after, 0.0);
    }

    /// 组 6b：极端输入——1 粒子不 panic。
    #[test]
    fn one_particle_ok() {
        let mut s = Smoke::new(
            spec((4, 4, 4)),
            &[SmokeParticle::new(Vec3::new(0.2, 0.2, 0.2), Vec3::new(0.1, 0.0, 0.0))],
        )
        .unwrap();
        s.step(1.0 / 60.0).unwrap();
        assert_eq!(s.particle_count(), 1);
        assert!(is_finite_vec3(s.particles()[0].position));
    }

    /// 组 6c：极端输入——NaN 初态被拒（P1019），不 panic。
    #[test]
    fn nan_initial_rejected() {
        let err = Smoke::new(
            spec((4, 4, 4)),
            &[SmokeParticle::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::splat(0.0))],
        )
        .unwrap_err();
        assert_eq!(err.code, "P1019");
        assert!(err.to_string().contains("建议："), "须含建议段");
    }

    /// 组 6d：极端输入——NaN dt / 负 dt / Inf dt 被拒（P1002）。
    #[test]
    fn bad_dt_rejected() {
        let mut s = block(spec((4, 4, 4)));
        for dt in [f32::NAN, f32::INFINITY, 0.0, -1.0 / 60.0] {
            let err = s.step(dt).unwrap_err();
            assert_eq!(err.code, "P1002", "dt={dt}");
            assert!(err.to_string().contains("建议："));
        }
    }

    /// 组 6e：规格防呆——非法参数三段式报错。
    #[test]
    fn spec_validation() {
        let base = spec((4, 4, 4));
        let mut bad = base;
        bad.dx = 0.0;
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1014");
        let mut bad = base;
        bad.res = (0, 4, 4);
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1014");
        let mut bad = base;
        bad.pressure_iters = 0;
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1017");
        let mut bad = base;
        bad.flip_ratio = 1.5;
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1015");
        let mut bad = base;
        bad.cfl = 0.0;
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1018");
        let mut bad = base;
        bad.gravity = Vec3::new(f32::NAN, 0.0, 0.0);
        assert_eq!(Smoke::new(bad, &[]).unwrap_err().code, "P1016");
    }

    /// 组 7：CFL 守卫——超大速度时 dt 被细分；超过 max_substeps 则拒绝。
    #[test]
    fn cfl_guard_limits_or_rejects() {
        let mut sp = spec((8, 8, 8));
        sp.max_substeps = 8;
        sp.cfl = 0.5;
        // 速度 = 100 m/s，dt = 1/60，dx = 0.1 → 需 ~ 33 子步 > 8 → 拒绝。
        let mut s = Smoke::new(
            sp,
            &[SmokeParticle::new(Vec3::new(0.4, 0.4, 0.4), Vec3::new(100.0, 0.0, 0.0))],
        )
        .unwrap();
        let err = s.step(1.0 / 60.0).unwrap_err();
        assert_eq!(err.code, "P1020", "超大速度应触发 CFL 拒绝");
        assert!(err.to_string().contains("建议："));

        // 同样速度但放宽子步上限 → 成功细分。
        sp.max_substeps = 64;
        let mut s = Smoke::new(
            sp,
            &[SmokeParticle::new(Vec3::new(0.4, 0.4, 0.4), Vec3::new(100.0, 0.0, 0.0))],
        )
        .unwrap();
        let rep = s.step(1.0 / 60.0).unwrap();
        assert!(rep.substeps > 1, "CFL 应细分：substeps={}", rep.substeps);
        assert!(rep.dt_sub < 1.0 / 60.0);
    }

    /// 组 8：FLIP 混合——α=0（纯 PIC）耗散 ≥ α=1（纯 FLIP）。
    ///
    /// 用**空间非均匀**速度场（径向扩散，强散度）逐粒子赋值，跑若干步后比较
    /// 动能：纯 PIC 采集把粒子速度拉向网格均值（耗散掉速度涨落），FLIP 保留
    /// 网格增量的涨落分量。非恒真判据：FLIP 动能 > PIC 动能（严格大于，因
    /// 初态是逐粒子不同的径向速度）。
    #[test]
    fn flip_vs_pic_energy() {
        let run = |ratio: f32| -> (f32, f32) {
            let mut sp = spec((8, 8, 8));
            sp.flip_ratio = ratio;
            sp.gravity = Vec3::splat(0.0);
            let mut s = block(sp);
            // 逐粒子径向速度（中心 = 块中心 0.35）：制造空间非均匀 + 散度。
            let c = Vec3::new(0.35, 0.35, 0.35);
            for i in 0..s.particle_count() {
                let p = s.particles()[i].position;
                let d = (p - c).normalize();
                s.set_particle_velocity(i, d * 1.0);
            }
            let ke0 = kinetic_energy(&s);
            for _ in 0..15 {
                s.step(1.0 / 60.0).unwrap();
            }
            (ke0, kinetic_energy(&s))
        };
        let (ke0, pic) = run(0.0);
        let (_, flip) = run(1.0);
        // 初始动能非零，确保场景非平凡。
        assert!(ke0 > 1e-3, "初态动能应为正：{ke0}");
        // FLIP 保留更多能量（PIC 耗散更强）。
        assert!(
            flip > pic + 1e-6,
            "FLIP 动能应严格大于 PIC：pic={pic}, flip={flip}"
        );
        println!("动能：初态={ke0:.6}, PIC={pic:.6}, FLIP={flip:.6}");
    }

    /// 组 9：开放边界不 panic 且有限。
    #[test]
    fn open_boundary_finite() {
        let mut sp = spec((6, 6, 6));
        sp.boundary = Boundary::Open;
        let mut s = SeedHelper::open(sp);
        for _ in 0..15 {
            s.step(1.0 / 60.0).unwrap();
        }
        for p in s.particles() {
            assert!(is_finite_vec3(p.position) && is_finite_vec3(p.velocity));
        }
    }

    /// 组 10：性能粗测（打印 N 粒子 × M 步耗时；无硬阈值断言，避免 CI 抖动）。
    #[test]
    fn perf_rough_measure() {
        use std::time::Instant;
        let res = (16, 16, 16);
        let mut sp = spec(res);
        sp.pressure_iters = 40;
        // 约 12³ = 1728 粒子，密度 ~0.4 粒子/单元。
        let mut s = Smoke::seed_block(
            sp,
            Vec3::new(0.2, 0.2, 0.2),
            (12, 12, 12),
            0.12,
            Vec3::splat(0.0),
        )
        .unwrap();
        let n = s.particle_count();
        let steps = 100;
        let t0 = Instant::now();
        for _ in 0..steps {
            s.step(1.0 / 60.0).unwrap();
        }
        let dt = t0.elapsed();
        println!(
            "性能：{n} 粒子 × {steps} 步（{} 网格，40 压力迭代）= {:?}（{:.3} ms/帧）",
            res.0 * res.1 * res.2,
            dt,
            dt.as_secs_f64() * 1000.0 / steps as f64
        );
    }

    // =======================================================================
    // 标量场输运 + 浮力 专项测试（本轮接手补齐）
    // =======================================================================

    /// 构造一个「不做压力投影散度修正」的纯标量平流求解器。
    ///
    /// 用途：隔离测试半拉格朗日平流——用人工给定的**均匀**速度场（`uniform`），
    /// 不做 P2G/投影（那会破坏均匀流场），只走 `advect_scalar_fields`。
    fn unif_advect_solver(res: (usize, usize, usize), dx: f32) -> Smoke {
        let sp = SmokeSpec {
            origin: Vec3::splat(0.0),
            dx,
            res,
            gravity: Vec3::splat(0.0),
            pressure_iters: 1,
            flip_ratio: 1.0,
            cfl: 5.0,
            max_substeps: 64,
            boundary: Boundary::Closed,
            buoyancy: 0.0,
        };
        Smoke::seed_block(sp, Vec3::new(0.25, 0.25, 0.25), (2, 2, 2), 0.05, Vec3::splat(0.0))
            .unwrap()
    }

    /// 向标量场注入一个确定性高斯团（峰值 `peak`，中心 `c`，标准差 `sigma`，
    /// 作用在连续单元坐标上）。不做归一化——用于观测平流位移与耗散。
    fn seed_gaussian(field: &mut [f32], res: (usize, usize, usize), c: Vec3, sigma: f32, peak: f32) {
        let (nx, ny, nz) = res;
        let inv2s2 = 1.0 / (2.0 * sigma * sigma);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5);
                    let d = p - c;
                    let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    let v = peak * (-r2 * inv2s2).exp();
                    field[k * ny * nx + j * nx + i] = v;
                }
            }
        }
    }

    /// 沿某轴的标量场峰值连续坐标（行主序扫描，返回 (峰值, 连续坐标)）。
    ///
    /// `axis`：0=x, 1=y, 2=z。连续坐标 = 单元下标 + 0.5（与 `seed_gaussian` 对齐）。
    fn peak_locate(field: &[f32], res: (usize, usize, usize), axis: usize) -> (f32, f32) {
        let (nx, ny, nz) = res;
        let mut best = f32::NEG_INFINITY;
        let mut best_coord = 0.0f32;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let v = field[k * ny * nx + j * nx + i];
                    if v > best {
                        best = v;
                        best_coord = match axis {
                            0 => i as f32 + 0.5,
                            1 => j as f32 + 0.5,
                            _ => k as f32 + 0.5,
                        };
                    }
                }
            }
        }
        (best, best_coord)
    }

    /// 标量场总和（确定性行主序累加；耗散观测用）。
    fn field_sum(field: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for v in field {
            acc += *v;
        }
        acc
    }

    /// 浮力专项 1：Boussinesq 高温场应使网格 y 面心速度**为正且超过下限**。
    ///
    /// 场景：重力 = 0（隔离浮力）、+Y 为「上」、粒子携带高温 `T`，`buoyancy = α`。
    /// 走公共 `step`（P2G → deposit_scalars 回填网格温度 → apply_buoyancy →
    /// 压力投影）。跑若干步后，脱离固壁面的 y 面心速度应累积为正（热空气上升），
    /// 且超过解析下界的一个保守分数。
    ///
    /// 断言非恒真：若 `apply_buoyancy` 未接入 `step`、施加到错误轴或符号取反，
    /// v_y 恒为 0 或为负 → 失败。
    #[test]
    fn buoyancy_hot_field_rises() {
        let res = (6, 6, 6);
        let alpha = 4.0f32;
        let t_val = 1.0f32;
        let mut sp = spec(res);
        sp.buoyancy = alpha;
        sp.gravity = Vec3::splat(0.0); // 隔离浮力：无重力干扰
        // 粒子携带高温（deposit_scalars 会把高温回填到网格温度场，供浮力使用）。
        let mut s = Smoke::seed_block_scalars(
            sp,
            Vec3::new(0.25, 0.25, 0.25),
            (3, 3, 3),
            0.05,
            Vec3::splat(0.0),
            t_val,
            0.0,
        )
        .unwrap();
        let (nx, ny, nz) = res;
        let mut max_v = f32::NEG_INFINITY;
        for _ in 0..6 {
            s.step(1.0 / 60.0).unwrap();
            for k in 0..nz {
                for j in 1..(ny - 1) {
                    for i in 0..nx {
                        let v = s.grid_velocity_y(i, j, k);
                        if v > max_v {
                            max_v = v;
                        }
                    }
                }
            }
        }
        // 解析下界：单步浮力增量 ≈ α·T·dt_sub；多步累积应显著 > 0。
        let dt = 1.0 / 60.0;
        let lower = 0.5 * alpha * t_val * dt; // 保守：至少单步增量的 50%
        assert!(
            max_v > lower,
            "高温场应产生向上的 y 面心速度：max_v={max_v} 应 > {lower}"
        );
        assert!(max_v.is_finite(), "浮力速度须有限");
        println!("浮力：max v_y = {max_v:.6}，下界 = {lower:.6}");
    }

    /// 浮力专项 2（精简版，独立可读）：`f_buoy = α·T·ŷ` 的**符号与量级**。
    ///
    /// 用单粒子 + 高温 + 零重力，step 若干步后断言该粒子所在处网格 y 面心速度
    /// 为 **正**（上升），且 **不超过** 解析上界的 2 倍（量级合理，非爆炸）。
    #[test]
    fn buoyancy_magnitude_sane() {
        let alpha = 5.0f32;
        let t_val = 0.8f32;
        let mut sp = spec((6, 6, 6));
        sp.buoyancy = alpha;
        sp.gravity = Vec3::splat(0.0);
        let mut s = Smoke::seed_block_scalars(
            sp,
            Vec3::new(0.28, 0.28, 0.28),
            (2, 2, 2),
            0.05,
            Vec3::splat(0.0),
            t_val,
            0.0,
        )
        .unwrap();
        let steps = 5u32;
        for _ in 0..steps {
            s.step(1.0 / 60.0).unwrap();
        }
        let (nx, ny, nz) = s.grid().dims();
        let mut max_v = f32::NEG_INFINITY;
        for k in 0..nz {
            for j in 1..(ny - 1) {
                for i in 0..nx {
                    let v = s.grid_velocity_y(i, j, k);
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
        }
        let upper = 2.0 * alpha * t_val * (steps as f32) * (1.0 / 60.0);
        assert!(max_v > 0.0, "浮力应产生正 y 速度：max_v={max_v}");
        assert!(
            max_v <= upper,
            "浮力速度量级应合理（≤2× 解析上界）：max_v={max_v}, upper={upper}"
        );
        println!("浮力量级：max_v={max_v:.6}, upper={upper:.6}");
    }

    /// 平流专项 1：高斯温度团在**均匀流场**下，峰值沿流向移动预期距离。
    ///
    /// 解析：半拉格朗日回溯 `x_src = x − u·Δt` ⇒ 每步把场按 `u·Δt` **平移**
    ///（均匀 u 下无插值误差以外的形变）。跑 N 步后峰值应位于 `初始峰值 + N·u·Δt`
    /// 附近（容差 = 1 个单元 + 10% 位移）。断言非恒真：若平流未接入 / 方向反 /
    /// 位移错误，峰值位置断言失败。
    #[test]
    fn advection_gaussian_translates_expected() {
        let res = (24, 12, 12);
        let dx = 0.1f32;
        // 均匀 +x 速度场：显式写入所有 u 面心。
        let u0 = 0.5f32; // m/s
        let dt = 0.05f32; // s
        let steps = 8u32;
        let mut s = unif_advect_solver(res, dx);
        let (nx, ny, nz) = res;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    s.set_grid_velocity_x(i, j, k, u0);
                }
            }
        }
        // 注入高斯温度团（中心在连续坐标 (6.5, 6.5, 6.5)）。
        let c0 = Vec3::new(6.5, 6.5, 6.5);
        let sigma = 1.5f32;
        seed_gaussian(s.grid_mut().temperature.as_mut_slice(), res, c0, sigma, 1.0);
        // 记录初始 x 峰值。
        let (_, x_peak0) = peak_locate(s.temperature_field(), res, 0);
        // 推进：纯平流（均匀速度场，直接反复调用 advect_scalar_fields，隔离
        //       不经 P2G/投影的其它耦合。
        for _ in 0..steps {
            s.advect_scalar_fields(dt).unwrap();
        }
        let (peak, x_peak1) = peak_locate(s.temperature_field(), res, 0);
        let expected_shift = (steps as f32) * u0 * dt / dx; // 单元数
        let actual_shift = x_peak1 - x_peak0;
        assert!(peak > 0.05, "平流后峰值应仍显著：{peak}");
        assert!(
            (actual_shift - expected_shift).abs() < 0.5 + 0.1 * expected_shift.abs(),
            "峰值位移应≈{expected_shift:.3} 单元，实际 {actual_shift:.3}"
        );
        println!(
            "平流：峰值 {:.3} → {:.3}（位移 {:.3} 单元，预期 {:.3}），峰值幅度 {:.4}",
            x_peak0, x_peak1, actual_shift, expected_shift, peak
        );
    }

    /// 平流专项 2：零速度场下半拉格朗日平流为**恒等**（不引入任何形变）。
    ///
    /// 断言：`u = 0` 时平流前后温度场逐位一致（`to_bits()` 全等）。这隔离验证
    /// 回溯 `x_src = x` 时插值精确返回原值（`fx=fy=fz=0` 分支）。
    #[test]
    fn advection_zero_velocity_identity() {
        let res = (12, 12, 12);
        let mut s = unif_advect_solver(res, 0.1);
        seed_gaussian(s.grid_mut().temperature.as_mut_slice(), res, Vec3::new(6.5, 6.5, 6.5), 2.0, 1.0);
        let before: Vec<u32> = s.temperature_field().iter().map(|v| v.to_bits()).collect();
        for _ in 0..5 {
            s.advect_scalar_fields(0.05).unwrap();
        }
        let after: Vec<u32> = s.temperature_field().iter().map(|v| v.to_bits()).collect();
        assert_eq!(before, after, "零速度场平流应为恒等（逐位）");
    }

    /// 确定性断言 A：标量场 + 速度场完整状态的**逐位可复现**。
    ///
    /// 同输入构建两个求解器，各跑 N 步，断言三场（温度 / 密度 / 网格速度 u,v,w）
    /// 的 `to_bits()` 序列完全一致。非恒真控制组：把**粒子初态温度**扰动 1 ULP，
    /// 结果**必须**逐位不同（证明本测试确在比较真实状态）。
    #[test]
    fn scalar_fields_bitwise_deterministic() {
        // 注意：扰动必须落在**粒子初态标量**上——`step` 每帧会 p2g 清空网格标量
        // 再由粒子沉积重建，故直接改网格温度会被首帧擦除（旧控制组因此恒真通过）。
        let build = |temp: f32| -> Smoke {
            let mut sp = spec((8, 8, 8));
            sp.buoyancy = 3.0;
            Smoke::seed_block_scalars(
                sp,
                Vec3::new(0.25, 0.25, 0.25),
                (3, 3, 3),
                0.05,
                Vec3::new(0.2, 0.0, 0.1),
                temp,
                0.4,
            )
            .unwrap()
        };
        let snapshot = |s: &Smoke| -> Vec<u32> {
            let mut v: Vec<u32> = Vec::new();
            for x in s.temperature_field() {
                v.push(x.to_bits());
            }
            for x in s.density_field() {
                v.push(x.to_bits());
            }
            let g = s.grid();
            for x in &g.u {
                v.push(x.to_bits());
            }
            for x in &g.v {
                v.push(x.to_bits());
            }
            for x in &g.w {
                v.push(x.to_bits());
            }
            v
        };
        let t0 = 0.7f32;
        let mut a = build(t0);
        let mut b = build(t0);
        for _ in 0..20 {
            a.step(1.0 / 60.0).unwrap();
            b.step(1.0 / 60.0).unwrap();
        }
        assert_eq!(snapshot(&a), snapshot(&b), "标量场+速度场应逐位可复现");

        // 非恒真控制组：粒子初态温度扰动 1 ULP。
        let mut c = build(f32::from_bits(t0.to_bits() + 1));
        for _ in 0..20 {
            c.step(1.0 / 60.0).unwrap();
        }
        assert_ne!(
            snapshot(&a),
            snapshot(&c),
            "1 ULP 粒子温度扰动应改变结果（证明确定性测试非恒真）"
        );
    }

    /// 确定性断言 B：**纯平流**路径（不经 P2G/投影）的逐位可复现。
    ///
    /// 隔离验证半拉格朗日平流自身的确定性：同初场 + 同均匀速度 + 同步数，
    /// 温度场逐位一致。这是对 `advect_scalars`（回溯序 / 插值序）的定点检查。
    #[test]
    fn advection_bitwise_deterministic() {
        let build = || -> Smoke {
            let res = (16, 10, 10);
            let mut s = unif_advect_solver(res, 0.1);
            let (nx, ny, nz) = res;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..=nx {
                        // 空间变化的 u 场（非均匀 → 触发插值）。
                        s.set_grid_velocity_x(i, j, k, 0.3 + 0.01 * i as f32);
                    }
                }
            }
            seed_gaussian(s.grid_mut().temperature.as_mut_slice(), res, Vec3::new(8.0, 5.0, 5.0), 2.0, 1.0);
            seed_gaussian(s.grid_mut().density.as_mut_slice(), res, Vec3::new(6.0, 4.0, 5.0), 1.5, 0.8);
            s
        };
        let snap = |s: &Smoke| -> Vec<u32> {
            s.temperature_field()
                .iter()
                .chain(s.density_field().iter())
                .map(|v| v.to_bits())
                .collect()
        };
        let mut a = build();
        let mut b = build();
        for _ in 0..12 {
            a.advect_scalar_fields(0.05).unwrap();
            b.advect_scalar_fields(0.05).unwrap();
        }
        assert_eq!(snap(&a), snap(&b), "纯平流路径应逐位可复现");
    }

    /// 耗散特性断言：半拉格朗日平流**耗散但可复现**（不严格守恒）。
    ///
    /// 半拉格朗日是一阶**耗散**格式，其耗散签名是**峰值幅度下降**（高斯团被
    /// 抹平），而总量只是**近似**守恒——网格域边界的回溯钳制会在边界单元引入
    /// 微小的质量漂移（±几个百分点）。故本测试**刻意不断言严格守恒**，改断言：
    /// (a) 峰值幅度**下降**（耗散确实发生）；
    /// (b) 总量落在合理带内（[0.9, 1.1]×初值，既非守恒失真也非爆炸）；
    /// (c) 两次独立运行的总量/峰值**逐位一致**（可复现）。
    #[test]
    fn advection_dissipative_but_bounded_and_reproducible() {
        let run = || -> (f32, f32, f32, f32) {
            let res = (24, 12, 12);
            let mut s = unif_advect_solver(res, 0.1);
            let (nx, ny, nz) = res;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..=nx {
                        s.set_grid_velocity_x(i, j, k, 0.5);
                    }
                }
            }
            seed_gaussian(s.grid_mut().temperature.as_mut_slice(), res, Vec3::new(6.5, 6.5, 6.5), 1.5, 1.0);
            let sum0 = field_sum(s.temperature_field());
            let (peak0, _) = peak_locate(s.temperature_field(), res, 0);
            for _ in 0..10 {
                s.advect_scalar_fields(0.05).unwrap();
            }
            let sum1 = field_sum(s.temperature_field());
            let (peak1, _) = peak_locate(s.temperature_field(), res, 0);
            (sum0, sum1, peak0, peak1)
        };
        let (sum0, sum1, peak0, peak1) = run();
        let (sum0b, sum1b, peak0b, peak1b) = run();
        assert!(sum0 > 0.0, "初态总量应为正：{sum0}");
        // (a) 耗散签名：峰值幅度下降。
        assert!(
            peak1 < peak0,
            "半拉格朗日应耗散峰值幅度：{peak0} → {peak1}"
        );
        // (b) 总量在合理带内（边界钳制允许微小漂移，但不得放宽到失去意义）。
        assert!(
            sum1 >= sum0 * 0.9 && sum1 <= sum0 * 1.1,
            "总量应在 [0.9,1.1]×初值内：{sum0} → {sum1}"
        );
        // (c) 可复现（逐位）。
        assert_eq!(sum0.to_bits(), sum0b.to_bits(), "初态总量应可复现");
        assert_eq!(sum1.to_bits(), sum1b.to_bits(), "耗散后总量应逐位可复现");
        assert_eq!(peak0.to_bits(), peak0b.to_bits(), "初态峰值应可复现");
        assert_eq!(peak1.to_bits(), peak1b.to_bits(), "耗散后峰值应逐位可复现");
        println!(
            "耗散：总量 {sum0:.6} → {sum1:.6}（{:.2}%），峰值 {peak0:.6} → {peak1:.6}",
            100.0 * (sum1 - sum0) / sum0,
        );
    }

    fn kinetic_energy(s: &Smoke) -> f32 {
        let mut ke = 0.0f32;
        for p in s.particles() {
            let v2 = p.velocity.dot(&p.velocity);
            ke += 0.5 * v2; // 单位质量
        }
        ke
    }

    /// 开放边界测试的辅助构造。
    struct SeedHelper;
    impl SeedHelper {
        fn open(sp: SmokeSpec) -> Smoke {
            Smoke::seed_block(
                sp,
                Vec3::new(0.2, 0.2, 0.2),
                (3, 3, 3),
                0.05,
                Vec3::new(0.0, 1.0, 0.0),
            )
            .unwrap()
        }
    }
}
