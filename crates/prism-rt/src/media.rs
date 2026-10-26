//! 均匀 HG 体积介质：相函数抽样 + Delta tracking（异质网格待后续扩展）。
//!
//! # 无偏估计量（推导锚点）
//! - 自由程 t = −ln(u)/σt：碰撞点 pdf = σt·e^(−σt·t)；散射分支概率 = 反照率
//!   σs/σt → 散射事件联合 pdf = σs·e^(−σt·t) = 被积因子 T(t)·σs 本身 →
//!   **散射处 throughput 不变**（T(t) 与 pdf 相消）。
//! - 到面存活 = 隐式二值事件：未碰撞走到面的概率 E = e^(−σt·d) = T(d)，
//!   面贡献自动带权——运行期零 exp。
//! - 阴影透射率 = 单抽二值估计：自由程 ≥ 段长 → 1，否则 0；
//!   E[T̂] = e^(−σt·d) 精确等于 Beer 定律（测试断言之）。
//!
//! # 确定性数学（D-29 简化件，规范缺口在册项）
//! [`det_ln`] 为运行期必需（自由程）；位分解 + atanh 级数，纯 ±×÷ 与整数位
//! 运算，禁 libm，跨平台逐位一致。后续随 prism-math 的 D-29 确定性数学件上收。
//! （透射率测试参考值用 f64 exp——libm 禁令只约束引擎确定性路径，不约束测试。）

use std::f32::consts::PI;

use prism_math::{clamp, Vec3};

use crate::sampler::{cos_sin, onb};

/// u 的夹持下界（= 2^-24，Sobol 坐标量化格）：u = 0 时 −ln(u) = ∞ 会毒化
/// 路径；量化格上 u=0 的质量可忽略，注释声明（与 sampler 高 24 位换算同格）。
const U_EPS: f32 = 1.0 / 16_777_216.0;
/// ln 2（公开数学常量；D-30：与 `std::f32::consts::LN_2` `to_bits()` 逐位相同）。
const LN_2: f32 = std::f32::consts::LN_2;

/// 确定性自然对数（x > 0）。
///
/// 位分解 x = 2^e·m（m ∈ [1,2)）→ ln m = 2·atanh(s)，s = (m−1)/(m+1) ∈
/// [0, 1/3) → atanh 级数 7 项（s¹⁵ 余项 < 1e-8）。次正规数先乘 2^24（精确）
/// 挪入正规域再减 24·ln2。只含 ±×÷ 与整数位运算——D-29。
pub fn det_ln(x: f32) -> f32 {
    assert!(x > 0.0 && x.is_finite(), "det_ln 定义域为正有限数，收到 {x}");
    let bits = x.to_bits();
    if (bits >> 23) & 0xFF == 0 {
        // 次正规数：精确幂缩放进正规域（递归深度 1）。
        return det_ln(x * 16_777_216.0) - 24.0 * LN_2;
    }
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // m ∈ [1,2)
    let f = m - 1.0;
    let s = f / (2.0 + f); // ∈ [0, 1/3)
    let s2 = s * s;
    // atanh(s) = s·(1 + s²/3 + s⁴/5 + … + s¹²/13)，Horner 排布。
    let atanh = s
        * (1.0
            + s2 * (1.0 / 3.0
                + s2 * (1.0 / 5.0
                    + s2 * (1.0 / 7.0
                        + s2 * (1.0 / 9.0
                            + s2 * (1.0 / 11.0 + s2 * (1.0 / 13.0)))))));
    e as f32 * LN_2 + 2.0 * atanh
}

/// 均匀 HG 介质（σt 总消光、反照率 σs/σt、各向异性 g）。
#[derive(Debug, Clone, Copy)]
pub struct HomogeneousMedium {
    /// 总消光系数 σt = σa + σs（1/length）。
    pub sigma_t: f32,
    /// 单次散射反照率 σs/σt ∈ [0,1]。
    pub albedo: f32,
    /// HG 各向异性参数 g ∈ (−1,1)：g > 0 前向散射。
    pub g: f32,
}

impl HomogeneousMedium {
    /// 构造（断言系数域，宪章 C19）。
    pub fn new(sigma_t: f32, albedo: f32, g: f32) -> Self {
        assert!(
            sigma_t > 0.0,
            "σt 必须为正，收到 {sigma_t}；建议：σt = σa + σs 为总消光系数（单位 1/长度），检查介质密度与单位换算"
        );
        assert!(
            (0.0..=1.0).contains(&albedo),
            "反照率必须在 [0,1]，收到 {albedo}；建议：albedo = σs/σt，0 为纯吸收、1 为纯散射，常见材质 0.1–0.9"
        );
        assert!(
            g > -1.0 && g < 1.0,
            "g 必须在 (-1,1) 开区间，收到 {g}；建议：前向散射取正值（雾/皮肤常用 0.3–0.8），后向取负，各向同性填 0"
        );
        HomogeneousMedium { sigma_t, albedo, g }
    }

    /// 散射系数 σs。
    pub fn sigma_s(&self) -> f32 {
        self.sigma_t * self.albedo
    }

    /// 吸收系数 σa。
    pub fn sigma_a(&self) -> f32 {
        self.sigma_t * (1.0 - self.albedo)
    }

    /// 自由程抽样：t = −ln(u)/σt。
    pub fn free_path(&self, u: f32) -> f32 {
        -det_ln(clamp(u, U_EPS, 1.0 - U_EPS)) / self.sigma_t
    }

    /// 阴影透射率（delta tracking 单抽二值）：自由程 ≥ dist → 1，否则 0。
    /// E[T̂] = e^(−σt·dist)（Beer 定律），无偏。
    pub fn shadow_transmittance(&self, u: f32, dist: f32) -> f32 {
        if self.free_path(u) >= dist {
            1.0
        } else {
            0.0
        }
    }

    /// HG 相函数 pdf（按 cosθ）：(1−g²) / (4π·(1+g²−2g·cosθ)^{3/2})。
    pub fn phase(&self, cos_theta: f32) -> f32 {
        let g = self.g;
        let den = 1.0 + g * g - 2.0 * g * cos_theta;
        (1.0 - g * g) / (4.0 * PI * den * den.sqrt())
    }

    /// HG 方向抽样：绕入射（传播）方向 w 散射出的新方向。
    /// cosθ = [1+g² − ((1−g²)/(1−g+2g·u₁))²] / (2g)；g → 0 退化为均匀球。
    pub fn sample_phase(&self, w: Vec3, u1: f32, u2: f32) -> Vec3 {
        let g = self.g;
        let cos_theta = if g.abs() < 1e-6 {
            1.0 - 2.0 * u1 // 各向同性极限
        } else {
            let a = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            (1.0 + g * g - a * a) / (2.0 * g)
        };
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let (t, b) = onb(w);
        let (c, s) = cos_sin(2.0 * PI * u2);
        w * cos_theta + t * (sin_theta * c) + b * (sin_theta * s)
    }
}

// ---------------------------------------------------------------------------
// 异质体积介质（C2，#10 实施子集）：σt 空间变化，Delta tracking 扩展
// ---------------------------------------------------------------------------

/// 规则三网格（异质 σt 场载体）。
///
/// 行主序数据 `data[x + nx*(y + ny*nz)]`，**每格一个值、定位在格心**：格
/// `(i,j,k)` 覆盖 `[origin + (i,j,k)·spacing, origin + (i+1,j+1,k+1)·spacing]`，
/// 其 σt 代表值即 `data[i + nx*(j + ny*k)]`，取值于格心
/// `origin + (i+0.5, j+0.5, k+0.5)·spacing`。故网格物理跨度恒为
/// `(nx, ny, nz)·spacing`（8 格 × 0.25 = 2.0，而非 7 × 0.25 = 1.75）。
///
/// [`sample`](Grid3::sample) 做三线性插值（格心口径：格心处恰返回该格的
/// `data` 值），并把界外点夹到边界格心（常数外推——最外半格及更远处取边界
/// 格值），保证退化（单格）场处处取同一 σt。
#[derive(Debug, Clone)]
pub struct Grid3 {
    /// 各轴格数（> 0）。
    pub nx: usize,
    /// 各轴格数（> 0）。
    pub ny: usize,
    /// 各轴格数（> 0）。
    pub nz: usize,
    /// 网格原点（最小角），世界坐标。
    pub origin: Vec3,
    /// 每格物理尺寸（> 0）。
    pub spacing: Vec3,
    /// σt 数据（长度 nx*ny*nz，单位 1/length）。
    pub data: Vec<f32>,
}

impl Grid3 {
    /// 构造（数据长度须为 nx*ny*nz；断言域与数据一致，宪章 C19）。
    pub fn new(nx: usize, ny: usize, nz: usize, origin: Vec3, spacing: Vec3, data: Vec<f32>) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0, "网格各轴格数须 > 0");
        assert!(
            spacing.x > 0.0 && spacing.y > 0.0 && spacing.z > 0.0,
            "网格间距须 > 0"
        );
        assert_eq!(
            data.len(),
            nx * ny * nz,
            "σt 数据长度 {} 须等于 nx*ny*nz = {}",
            data.len(),
            nx * ny * nz
        );
        Grid3 { nx, ny, nz, origin, spacing, data }
    }

    /// σt 数据原始上界（`max(data)`，**不**含 U_EPS 地板；全零场返回 0）。
    ///
    /// 用于参数域校验：`majorant()` 的 U_EPS 地板使其恒 > 0（除零安全），
    /// 不能用作「场是否真空」的判据。
    pub fn raw_max(&self) -> f32 {
        let mut m = 0.0f32;
        for &v in &self.data {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// 主消光上限（Woodcock majorant）＝原始上界，地板到 U_EPS 防除零。
    ///
    /// 注意：地板使返回值恒 > 0，**不能**据此判真空场——真空判据用
    /// [`raw_max`](Grid3::raw_max)。
    pub fn majorant(&self) -> f32 {
        self.raw_max().max(U_EPS)
    }

    #[inline]
    fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[x + self.nx * (y + self.ny * z)]
    }

    /// 三线性采样 σt（格心口径；界外夹到边界格心 = 常数外推）。
    ///
    /// 归一化坐标带**半格偏移** `g = (x − origin)/spacing − 0.5`：格心
    /// `origin + (i+0.5)·spacing` 映射到整数 i ⇒ 恰返回 `data[i]`。整体夹到
    /// `[0, n−1]` ⇒ 首格左半格与末格右半格（及界外）取边界格值，网格覆盖
    /// `n·spacing` 全跨度（非 `(n−1)·spacing`）。
    pub fn sample(&self, pos: Vec3) -> f32 {
        let gx = ((pos.x - self.origin.x) / self.spacing.x - 0.5).clamp(0.0, (self.nx as f32) - 1.0);
        let gy = ((pos.y - self.origin.y) / self.spacing.y - 0.5).clamp(0.0, (self.ny as f32) - 1.0);
        let gz = ((pos.z - self.origin.z) / self.spacing.z - 0.5).clamp(0.0, (self.nz as f32) - 1.0);
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let z0 = gz.floor() as usize;
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let z1 = (z0 + 1).min(self.nz - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let fz = gz - z0 as f32;
        let c000 = self.at(x0, y0, z0);
        let c100 = self.at(x1, y0, z0);
        let c010 = self.at(x0, y1, z0);
        let c110 = self.at(x1, y1, z0);
        let c001 = self.at(x0, y0, z1);
        let c101 = self.at(x1, y0, z1);
        let c011 = self.at(x0, y1, z1);
        let c111 = self.at(x1, y1, z1);
        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        c0 + (c1 - c0) * fz
    }

    /// 构造单格恒定场（退化一致性测试用）：处处 σt = s。
    pub fn constant(s: f32) -> Self {
        Grid3::new(
            1,
            1,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            vec![s],
        )
    }
}

/// 空碰撞后派生下一对均匀（确定性、强去相关）。
///
/// # 为什么不用 `(u·π).fract()` 这类乘法链
/// QA 轮实测：乘法链派生的相邻步均匀**强相关**（同一初值的轨道点），
/// 使 `shadow_transmittance` 在变化 σt 场上产生 **+1.08% 正偏（4.7σ）**
/// （tracked 版喂独立均匀仅 +0.13%）。改为 splitmix64 雪崩混合：以
/// `(u_step, u_accept, step)` 三者为种子，输出两个独立 24-bit 均匀。
/// 确定性不变（同输入同输出，逐位可复现），仅去相关。
#[inline]
fn derive_uniforms(us: f32, ua: f32, step: u32) -> (f32, f32) {
    #[inline]
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let seed = ((us.to_bits() as u64) << 32)
        ^ (ua.to_bits() as u64)
        ^ ((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let h1 = mix(seed);
    let h2 = mix(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let to_unit = |h: u64| ((h >> 40) as f32) * (1.0 / 16_777_216.0); // 24-bit → [0,1)
    (to_unit(h1), to_unit(h2))
}

/// 异质 HG 介质（σt 空间变化，C2）。
///
/// 在 [`HomogeneousMedium`] 的均匀设定上扩展 σt 为空间变化场（`Grid3`），
/// 沿用同一套 HG 相函数与确定性 `det_ln`。自由程 / 阴影透射率改用
/// **Woodcock delta tracking**（σt 空间变化下的无偏估计量）：以场上界
/// `majorant` 做指数步进，逐段以 `σt(t)/majorant` 概率接受碰撞。
///
/// # 退化一致性（C2 核心判据）
/// σt 恒定场（单格 `Grid3`）下，Woodcock 首步即 `-ln(u)/σt`，接受概率
/// `σt/majorant = 1` 必中，故 [`free_path`]/[`shadow_transmittance`] 与
/// `HomogeneousMedium` 逐位一致（同 `det_ln` / 同 `clamp` / 同 `U_EPS`）。
#[derive(Debug, Clone)]
pub struct HeterogeneousMedium {
    /// 空间变化总消光场 σt = σa + σs（1/length）。
    pub sigma_t_field: Grid3,
    /// 单次散射反照率 σs/σt ∈ [0,1]（空间不变）。
    pub albedo: f32,
    /// HG 各向异性参数 g ∈ (-1,1)（空间不变）。
    pub g: f32,
}

impl HeterogeneousMedium {
    /// 构造（断言系数域，宪章 C19）。
    ///
    /// σt 场真空判据用**原始**上界 [`Grid3::raw_max`]：`majorant()` 带 U_EPS
    /// 地板（除零安全），恒 > 0，与真值无关，不能作为守卫。
    pub fn new(sigma_t_field: Grid3, albedo: f32, g: f32) -> Self {
        assert!(
            sigma_t_field.raw_max() > 0.0,
            "σt 场上界须为正（建议：密度与单位换算正确）；真空介质请用 σt>0 的薄层近似"
        );
        assert!(
            (0.0..=1.0).contains(&albedo),
            "反照率必须在 [0,1]，收到 {albedo}"
        );
        assert!(
            g > -1.0 && g < 1.0,
            "g 必须在 (-1,1) 开区间，收到 {g}"
        );
        HeterogeneousMedium { sigma_t_field, albedo, g }
    }

    /// 构造恒定 σt 场（退化一致性测试 / 均匀异质近似用）。
    pub fn constant_sigma_t(sigma_t: f32, albedo: f32, g: f32) -> Self {
        HeterogeneousMedium::new(Grid3::constant(sigma_t), albedo, g)
    }

    /// 散射系数 σs（在给定位置）。
    pub fn sigma_s_at(&self, pos: Vec3) -> f32 {
        self.sigma_t_field.sample(pos) * self.albedo
    }

    /// 吸收系数 σa（在给定位置）。
    pub fn sigma_a_at(&self, pos: Vec3) -> f32 {
        self.sigma_t_field.sample(pos) * (1.0 - self.albedo)
    }

    /// 自由程抽样（Woodcock delta tracking）。
    ///
    /// `u_step` 控制指数步进、`u_accept` 控制逐段碰撞接受；恒定场下二者仅
    /// `u_step` 决定结果（接受概率 = 1），与 `HomogeneousMedium::free_path`
    /// 逐位一致。`origin`/`dir` 在恒定场下不参与结果（恒接受首步）。
    pub fn free_path(&self, origin: Vec3, dir: Vec3, u_step: f32, u_accept: f32) -> f32 {
        let majorant = self.sigma_t_field.majorant();
        let mut t = 0.0f32;
        let mut us = u_step;
        let mut ua = u_accept;
        let mut step_ix = 0u32;
        loop {
            let step = -det_ln(clamp(us, U_EPS, 1.0 - U_EPS)) / majorant;
            t += step;
            let pos = origin + dir * t;
            let sigma = self.sigma_t_field.sample(pos);
            if ua <= sigma / majorant {
                return t;
            }
            // 空碰撞，继续步进；派生下一对均匀（splitmix64 去相关，见 [`derive_uniforms`]）。
            step_ix += 1;
            let (ns, na) = derive_uniforms(us, ua, step_ix);
            us = ns;
            ua = na;
        }
    }

    /// 阴影透射率（delta tracking 单抽二值）：从 origin 沿 dir 走到 dist 之前
    /// 未碰撞 → 1，否则 0。恒定场下首步即 `-ln(u)/σt`，与
    /// `HomogeneousMedium::shadow_transmittance` 逐位一致。
    pub fn shadow_transmittance(
        &self,
        origin: Vec3,
        dir: Vec3,
        dist: f32,
        u_step: f32,
        u_accept: f32,
    ) -> f32 {
        let majorant = self.sigma_t_field.majorant();
        let mut t = 0.0f32;
        let mut us = u_step;
        let mut ua = u_accept;
        let mut step_ix = 0u32;
        loop {
            let step = -det_ln(clamp(us, U_EPS, 1.0 - U_EPS)) / majorant;
            t += step;
            if t >= dist {
                return 1.0;
            }
            let pos = origin + dir * t;
            let sigma = self.sigma_t_field.sample(pos);
            if ua <= sigma / majorant {
                return 0.0;
            }
            // 空碰撞：splitmix64 去相关派生（QA-F1：乘法链 +1.08% 正偏的修复）。
            step_ix += 1;
            let (ns, na) = derive_uniforms(us, ua, step_ix);
            us = ns;
            ua = na;
        }
    }

    /// 多步 Woodcock delta tracking（供无偏性测试 / 生产多步进）：传入**独立**
    /// 均匀对序列 `(u_step, u_accept)`，耗尽仍未碰撞则返回 1.0 兜底。与
    /// [`shadow_transmittance`](HeterogeneousMedium::shadow_transmittance) 同
    /// 一估计量，区别仅在于逐段均匀由调用方提供（无哈希相关性）。
    pub fn shadow_transmittance_tracked(
        &self,
        origin: Vec3,
        dir: Vec3,
        dist: f32,
        uniforms: &[(f32, f32)],
    ) -> f32 {
        let majorant = self.sigma_t_field.majorant();
        let mut t = 0.0f32;
        for &(us, ua) in uniforms {
            let step = -det_ln(clamp(us, U_EPS, 1.0 - U_EPS)) / majorant;
            t += step;
            if t >= dist {
                return 1.0;
            }
            let pos = origin + dir * t;
            let sigma = self.sigma_t_field.sample(pos);
            if ua <= sigma / majorant {
                return 0.0;
            }
        }
        1.0
    }

    /// HG 相函数 pdf（按 cosθ）：(1−g²) / (4π·(1+g²−2g·cosθ)^{3/2})。
    pub fn phase(&self, cos_theta: f32) -> f32 {
        let g = self.g;
        let den = 1.0 + g * g - 2.0 * g * cos_theta;
        (1.0 - g * g) / (4.0 * PI * den * den.sqrt())
    }

    /// HG 方向抽样：绕入射（传播）方向 w 散射出的新方向（同 `HomogeneousMedium`）。
    pub fn sample_phase(&self, w: Vec3, u1: f32, u2: f32) -> Vec3 {
        let g = self.g;
        let cos_theta = if g.abs() < 1e-6 {
            1.0 - 2.0 * u1
        } else {
            let a = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
            (1.0 + g * g - a * a) / (2.0 * g)
        };
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let (t, b) = onb(w);
        let (c, s) = cos_sin(2.0 * PI * u2);
        w * cos_theta + t * (sin_theta * c) + b * (sin_theta * s)
    }
}

// ---------------------------------------------------------------------------
// 测试（关卡 4：HG 体积 vs Beer 定律解析值）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler;
    use prism_core::{EvalCtx, SampleIndex};

    #[test]
    fn det_ln_matches_known_values() {
        // 公开数学常量锚点。
        assert_eq!(det_ln(1.0).to_bits(), 0.0f32.to_bits());
        for (x, expect) in [
            (2.0, 0.693_147_2),
            (0.5, -0.693_147_2),
            (10.0, 2.302_585),
            (0.25, -1.386_294_4),
            (std::f32::consts::E, 1.0),
            (1.0e-6, -13.815_511),
            (16_777_216.0, 16.635_532),
        ] {
            let got = det_ln(x);
            assert!(
                (got - expect).abs() < 2e-5,
                "ln({x}) = {got}，期望 {expect}"
            );
        }
        // 确定性：两遍逐位一致。
        assert_eq!(det_ln(0.37).to_bits(), det_ln(0.37).to_bits());
    }

    #[test]
    fn grid3_sample_is_cell_centered() {
        // 8 格 × 0.25 = 跨度 2.0；data[i] 定位在格心 (i+0.5)·0.25。
        let data: Vec<f32> = (0..8).map(|i| 0.4 + 1.2 * (i as f32 + 0.5) / 8.0).collect();
        let g = Grid3::new(
            8,
            1,
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.25, 1.0, 1.0),
            data.clone(),
        );
        // 格心处精确返回该格值。
        for (i, &v) in data.iter().enumerate() {
            let x = (i as f32 + 0.5) * 0.25;
            assert_eq!(g.sample(Vec3::new(x, 0.5, 0.5)).to_bits(), v.to_bits(), "格心 x={x}");
        }
        // 最末半格 [1.75, 2.0] 属末格（data[7]），不得被角节点外推吞掉。
        assert_eq!(g.sample(Vec3::new(1.875, 0.5, 0.5)).to_bits(), data[7].to_bits());
        assert_eq!(g.sample(Vec3::new(1.999, 0.5, 0.5)).to_bits(), data[7].to_bits());
        // 网格跨度 = nx·spacing = 2.0（而非 (nx−1)·spacing = 1.75）。
        assert_eq!(g.sample(Vec3::new(2.0, 0.5, 0.5)).to_bits(), data[7].to_bits());
        // 密度积分 = Σ data_i·cell_w（格心口径下线性场精确 = 2.0）。
        let n = 100_000usize;
        let h = 2.0 / n as f32;
        let mut acc = 0.0f64;
        for k in 0..n {
            let x = (k as f32 + 0.5) * h;
            acc += g.sample(Vec3::new(x, 0.5, 0.5)) as f64 * h as f64;
        }
        assert!((acc - 2.0).abs() < 1e-3, "∫σt = {acc}，格心口径应为 2.0");
    }

    #[test]
    fn grid3_raw_max_excludes_eps_floor() {
        let zeros = Grid3::new(
            2,
            2,
            2,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            vec![0.0; 8],
        );
        assert_eq!(zeros.raw_max(), 0.0, "真空场原始上界必须为 0");
        assert!(zeros.majorant() > 0.0, "majorant 带 U_EPS 地板（除零安全）");
        let g = Grid3::constant(0.7);
        assert_eq!(g.raw_max(), 0.7);
        assert_eq!(g.majorant(), 0.7);
    }

    #[test]
    fn free_path_mean_is_inverse_sigma_t() {
        let m = HomogeneousMedium::new(0.8, 0.9, 0.2);
        let ctx = EvalCtx::new(31, 7, 3);
        let n = 50_000u32;
        let mut sum = 0.0f64;
        for k in 0..n {
            let si = SampleIndex {
                frame: 3,
                pixel: (5, 6),
                sample: k,
                dimension: 40,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            sum += m.free_path(u) as f64;
        }
        let mean = sum / n as f64;
        let expected = 1.0 / m.sigma_t as f64;
        assert!(
            (mean - expected).abs() / expected < 0.02,
            "E[自由程] = {mean}，期望 {expected}"
        );
    }

    #[test]
    fn shadow_transmittance_matches_beers_law() {
        let m = HomogeneousMedium::new(0.7, 1.0, 0.0);
        let dist = 1.3f32;
        let ctx = EvalCtx::new(97, 7, 3);
        let n = 100_000u32;
        let mut ones = 0u32;
        for k in 0..n {
            let si = SampleIndex {
                frame: 3,
                pixel: (1, 1),
                sample: k,
                dimension: 41,
                node_id: 7,
            };
            let u = sampler::sample_1d(&ctx, &si);
            let t = m.shadow_transmittance(u, dist);
            assert!(t == 0.0 || t == 1.0, "二值估计出现非二值 {t}");
            ones += t as u32;
        }
        let est = ones as f64 / n as f64;
        let beer = (-(m.sigma_t as f64) * dist as f64).exp(); // 测试参考值可用 f64
        assert!(
            (est - beer).abs() / beer < 0.02,
            "E[T̂] = {est}，Beer = {beer}"
        );
    }

    #[test]
    fn phase_integrates_to_one_and_moments_match() {
        for g in [0.3f32, -0.4] {
            let m = HomogeneousMedium::new(1.0, 1.0, g);
            let ctx = EvalCtx::new(11, 7, 3);
            let n = 40_000u32;
            // 归一化 ∫p dΩ = 1 必须用**均匀球面方向**做 MC（E[4π·p] = 1）；
            // 用 HG 抽样方向配 p 会变成 ∫p²——那是矩检验（下半段）的事。
            let (mut sum_p, mut sum_cos, mut sum_cos2) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..n {
                let si = SampleIndex {
                    frame: 3,
                    pixel: (2, 2),
                    sample: k,
                    dimension: 42,
                    node_id: 7,
                };
                let (u1, u2) = sampler::sample_2d(&ctx, &si);
                // 均匀球面方向：z = 1−2u ∈ [−1,1]，φ = 2πv。
                let z = 1.0 - 2.0 * u1;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let (c, s) = crate::sampler::cos_sin(2.0 * std::f32::consts::PI * u2);
                let d = Vec3::new(r * c, r * s, z); // 轴 = w = (0,0,1)
                let cos = d.z as f64;
                sum_p += 4.0 * std::f64::consts::PI * m.phase(cos as f32) as f64;
                // 矩检验用 HG 抽样方向。
                let d_hg = m.sample_phase(Vec3::new(0.0, 0.0, 1.0), u1, u2);
                let cos_hg = d_hg.dot(&Vec3::new(0.0, 0.0, 1.0)) as f64;
                assert!((d_hg.length() - 1.0).abs() < 1e-4, "g={g}: 方向未单位化");
                sum_cos += cos_hg;
                sum_cos2 += cos_hg * cos_hg;
            }
            let norm = sum_p / n as f64;
            assert!((norm - 1.0).abs() < 0.01, "g={g}: ∫p dΩ = {norm}");
            let mean_cos = sum_cos / n as f64;
            assert!(
                (mean_cos - g as f64).abs() < 0.01,
                "g={g}: E[cosθ] = {mean_cos}"
            );
            // HG 恒等式：E[cos²θ] = (1+2g²)/3。
            let mean_cos2 = sum_cos2 / n as f64;
            let expected = (1.0 + 2.0 * (g as f64) * (g as f64)) / 3.0;
            assert!(
                (mean_cos2 - expected).abs() < 0.02,
                "g={g}: E[cos²θ] = {mean_cos2}，期望 {expected}"
            );
        }
    }
}
