//! 实时 GI / 虚拟几何——CpuRef 可跑的实时全局光照档位。
//!
//! # 语义权威
//! 的分层约定：**SDF-based GI（sphere tracing 遮挡查询，\[S\] 红利）+
//! 辐照度探针（S1 场 + S2 球谐平滑）+ 屏幕空间 GI（S2 重投影）——质量-预算
//! 曲线的实时档位**。指导原则 D-21"好算法可以既好用且好实现"。
//!
//! # 三条路径（模块划分）
//! - [`sdf_gi`]：SDF sphere tracing 遮挡查询 → AO + 一步间接光（**架构红利**：
//!   距离场挡查询免 BVH 遍历栈）；
//! - [`probes`]：辐照度探针（S1 标量场的规则网格离散化）+ SH 系数平滑（S2
//!   卷积）→ 空间插值的间接辐照度场；
//! - [`ssgi`]：屏幕空间 GI 重投影（S2 时域重采样核）→ 历史帧复用。
//!
//! # S1 / S2 到底是什么（引自十二结构总表，原文）
//! - **S1 | 标量场 | 协方差核随机场 | 噪声族、SDF 场、SPH 密度场 | 作用为主 |
//!   格点 hash + 格间 quintic 插值**
//! - **S2 | 卷积 | 调和分析核 | blur/bloom、K 加权、插值、shutter 核 |
//!   谱逆 / 直接 | 核尺寸与相位格锚**
//!
//! 映射：**探针格 = S1 场**（空间离散的辐照度标量场，格点值 + 格间插值），
//! **SH 平滑与重投影 = S2 卷积**（调和分析核把相邻探针低通混合 / 时域重采样核）。
//!
//! # 与路径追踪（S8）的关系（诚实边界）
//! [`crate::integrator`] 是**离线高质**路径（弹射收敛的 S8 辐射传输积分）；
//! 本模块是**实时档位**：固定步数 SDF 遮挡 + 探针/屏幕缓存，牺牲精度换速度。
//! 两者是同一 GI 能力的"质量-预算曲线"两端，并非替代关系（不影响 integrator
//! 的既有确定性契约）。
//!
//! # 确定性总纲
//! 全模块是 `(场景, 参数, 相机, 帧)` 的纯函数：探针遍历序固定（`iz→iy→ix`）、
//! SH 系数累加序固定（`0..9`）、重投影逐像素固定序；无 RNG、无并发。同输入
//! 逐位复现（各子模块测试锚定）。
//!
//! # C18（永不 panic）
//! 所有公开入口对退化输入（0 探针、NaN、退化相机/包围盒）收束或返回安全值，
//! 不 panic；详见各子模块的 `extreme_inputs_no_panic` / `*_sanitization` 测试。

pub mod probes;
pub mod sdf_gi;
pub mod sh;
pub mod ssgi;

pub use probes::{
    adaptive_place, adaptive_place_multi_bounce, bake_adaptive_gi_end_to_end, bake_probe_grid,
    bake_probe_grid_multi_bounce, field_normal, project_probe_sh, AdaptiveBudget,
    AdaptivePlacement, Probe, ProbeBudget, ProbeGrid, ProbeMultiBounceBudget,
};
pub use sdf_gi::{
    ambient_occlusion, is_occluded, multi_bounce_energy_upper_bound, one_bounce_indirect,
    solve_multi_bounce, total_radiant_energy, MultiBounceBudget, MultiBounceResult,
    OcclusionBudget, SceneSdf, SdfField,
};
pub use sh::{
    sh_basis_l2, sh_irradiance_to_radiance, sh_reconstruct_rgb, ShIrradiance, ShOrder,
};
pub use ssgi::{reproject_gi, CameraMotion, ReprojectionMap};

use prism_math::Vec3;

/// 质量档位（"质量-预算曲线"的命名档）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiQuality {
    /// 实时预览：低采样、L1、少步数（毫秒级目标）。
    Preview,
    /// 标准实时档（默认）。
    RealTime,
    /// 高质量收敛（近离线）。
    Converged,
}

impl GiQuality {
    /// 该档的探针预算。
    pub fn probe_budget(self) -> ProbeBudget {
        match self {
            GiQuality::Preview => ProbeBudget::low(),
            GiQuality::RealTime => ProbeBudget::standard(),
            GiQuality::Converged => ProbeBudget::high(),
        }
    }

    /// 该档的遮挡预算。
    pub fn occlusion_budget(self) -> OcclusionBudget {
        match self {
            GiQuality::Preview => OcclusionBudget::low(),
            GiQuality::RealTime => OcclusionBudget::standard(),
            GiQuality::Converged => OcclusionBudget::high(),
        }
    }

    /// 该档的 SH 阶数。
    pub fn order(self) -> ShOrder {
        match self {
            GiQuality::Preview => ShOrder::L1,
            GiQuality::RealTime => ShOrder::L2,
            GiQuality::Converged => ShOrder::L2,
        }
    }

    /// 该档的探针平滑强度（S2 卷积核幅度）。
    pub fn smooth_strength(self) -> f32 {
        match self {
            GiQuality::Preview => 0.5,  // 低采样噪声大 → 强平滑
            GiQuality::RealTime => 0.25,
            GiQuality::Converged => 0.0, // 高采样无需平滑
        }
    }
}

/// 实时 GI 参数（一次性配置）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiParams {
    /// 质量档。
    pub quality: GiQuality,
    /// 探针格分辨率（各轴探针数；越大越细、越贵）。
    pub probe_res: (u32, u32, u32),
    /// 环境入射辐亮度（实时档位的方向无关环境近似）。
    pub env_radiance: Vec3,
    /// 探针格世界包围盒最小角。
    pub min: Vec3,
    /// 探针格世界包围盒最大角。
    pub max: Vec3,
}

impl Default for GiParams {
    fn default() -> Self {
        Self {
            quality: GiQuality::RealTime,
            probe_res: (4, 4, 4),
            env_radiance: Vec3::splat(0.5),
            min: Vec3::splat(-4.0),
            max: Vec3::splat(4.0),
        }
    }
}

/// 实时 GI 结果（探针格 + 可选重投影图）。
#[derive(Debug, Clone)]
pub struct GiFrame {
    /// 烘焙的探针格（空间间接辐照度场）。
    pub probes: ProbeGrid,
    /// 使用的档位。
    pub quality: GiQuality,
}

/// 烘焙一帧实时 GI（SDF 遮挡 + 探针场 + S2 平滑）。
///
/// # 参数
/// - `scene`：RT 场景（构建 SDF）；
/// - `params`：GI 参数（档位、探针格、环境）。
///
/// # 确定性
/// 探针遍历与 SH 累加序固定；`params.quality` 决定预算但同档同参必同结果。
pub fn bake_gi_frame(scene: &crate::integrator::Scene, params: &GiParams) -> GiFrame {
    let field = SceneSdf::from_scene(scene);
    let budget = params.quality.probe_budget();
    let mut grid = bake_probe_grid(
        &field,
        params.env_radiance,
        params.min,
        params.max,
        params.probe_res,
        budget,
    );
    grid.smooth(params.quality.smooth_strength());
    GiFrame { probes: grid, quality: params.quality }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::Scene;

    /// 档位预算单调：Preview 成本 ≤ RealTime ≤ Converged（采样/步数）。
    #[test]
    fn quality_budgets_are_monotone() {
        let p = GiQuality::Preview.probe_budget();
        let r = GiQuality::RealTime.probe_budget();
        let c = GiQuality::Converged.probe_budget();
        assert!(p.sh_samples <= r.sh_samples && r.sh_samples <= c.sh_samples);
        assert!(p.ao_samples <= r.ao_samples && r.ao_samples <= c.ao_samples);
        let po = GiQuality::Preview.occlusion_budget();
        let co = GiQuality::Converged.occlusion_budget();
        assert!(po.max_steps <= co.max_steps, "低档步数应 ≤ 高档");
        // 阶数单调：Preview(L1) ≤ RealTime(L2)。
        assert!(GiQuality::Preview.order().count() <= GiQuality::RealTime.order().count());
    }

    /// 端到端：烘焙一帧 GI 不 panic，且确定性。
    #[test]
    fn bake_gi_frame_deterministic() {
        let scene = Scene::default();
        let params = GiParams { probe_res: (2, 2, 2), ..GiParams::default() };
        let a = bake_gi_frame(&scene, &params);
        let b = bake_gi_frame(&scene, &params);
        assert_eq!(a.probes.count(), b.probes.count());
        for (ca, cb) in a.probes.sh_cubes.iter().zip(b.probes.sh_cubes.iter()) {
            for ch in 0..3 {
                for i in 0..9 {
                    assert_eq!(ca[ch][i].to_bits(), cb[ch][i].to_bits());
                }
            }
        }
    }

    /// 档位同向：低档与高档在"空环境"下都应给出 0 辐照度（不反向）。
    #[test]
    fn quality_levels_agree_directionally() {
        let scene = Scene::default();
        // 空环境（env=0）：两档都应为 0（同向）。
        for q in [GiQuality::Preview, GiQuality::Converged] {
            let params = GiParams {
                quality: q,
                probe_res: (2, 1, 1),
                env_radiance: Vec3::splat(0.0),
                ..GiParams::default()
            };
            let f = bake_gi_frame(&scene, &params);
            let e = f.probes.sample_irradiance(Vec3::splat(0.0), q.order());
            assert_eq!(e, Vec3::splat(0.0));
        }
        // 非空环境（env=0.3）：两档都应给出 ≈ env 的天光（同向，不反向）。
        for q in [GiQuality::Preview, GiQuality::RealTime] {
            let params = GiParams {
                quality: q,
                probe_res: (1, 1, 1),
                env_radiance: Vec3::splat(0.3),
                ..GiParams::default()
            };
            let f = bake_gi_frame(&scene, &params);
            let e = f.probes.sample_irradiance(Vec3::splat(0.0), q.order());
            assert!((e.x - 0.3).abs() < 0.05, "档 {q:?} 天光应 ≈0.3，实际 {}", e.x);
        }
    }

    /// 极端输入：0 探针分辨率、NaN 环境、退化包围盒 → 不 panic。
    #[test]
    fn extreme_params_no_panic() {
        let scene = Scene::default();
        let params = GiParams {
            probe_res: (0, 0, 0),
            env_radiance: Vec3::splat(f32::NAN),
            min: Vec3::splat(f32::NAN),
            max: Vec3::splat(f32::NAN),
            ..GiParams::default()
        };
        let f = bake_gi_frame(&scene, &params);
        assert!(f.probes.count() >= 1);
        // NaN 查询点仍不 panic。
        let _ = f.probes.sample_irradiance(Vec3::new(f32::NAN, 0.0, 0.0), ShOrder::L2);
    }
}
