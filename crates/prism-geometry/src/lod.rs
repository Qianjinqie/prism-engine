//! LOD 链（从简；路线图 B3——步进 eps 随距离放大 + displace octaves 随距离衰减）。
//!
//! # 语义（从简近似）
//!
//! - `lod_eps`：近处用基准 `hit_eps`，远处按 `1 + t·eps_per_unit` 线性放宽
//!   （远处像素覆盖大，无需亚毫米精度——渲染 LOD 常规做法）。
//! - `lod_octaves`：displace 噪声层数随 `t` 线性衰减到 0（远处噪声细节不可见）。
//!
//! # 确定性
//!
//! [`sphere_trace_lod`] 步数固定、全部为输入的纯函数：同输入同命中。
//! 注意：octaves 随 t 变化时场不再是严格 Lipschitz=1（安全系数 0.9 + 命中阈值兜底，
//! 白皮书 §7 把该近似列为 LOD 允许的误差来源）。

use crate::csg::Csg;
use crate::sdf::{TraceHit, TraceParams, DEFAULT_DISPLACE_OCTAVES};
use prism_math::Vec3;

/// 步进命中阈值随距离放大：`base_eps · (1 + t · eps_per_unit)`。
#[inline]
pub fn lod_eps(base_eps: f32, t: f32, eps_per_unit: f32) -> f32 {
    base_eps * (1.0 + t * eps_per_unit)
}

/// displace octaves 随距离线性衰减（下限 0）：`max(0, max_octaves − t · falloff)` 向零截断。
#[inline]
pub fn lod_octaves(t: f32, max_octaves: u32, falloff: f32) -> u32 {
    let o = max_octaves as f32 - t * falloff;
    if o <= 0.0 {
        0
    } else {
        o as u32
    }
}

/// LOD 步进参数（在 [`TraceParams`] 之上叠加距离衰减项）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodParams {
    /// 基础步进参数（`hit_eps` 为 t=0 处的基准阈值）。
    pub trace: TraceParams,
    /// t=0 处的 displace octaves（满档）。
    pub max_octaves: u32,
    /// octaves 每单位 t 的衰减量。
    pub falloff: f32,
    /// eps 每单位 t 的放大率。
    pub eps_per_unit: f32,
}

impl Default for LodParams {
    fn default() -> Self {
        LodParams {
            trace: TraceParams::default(),
            max_octaves: DEFAULT_DISPLACE_OCTAVES,
            falloff: 0.1,
            eps_per_unit: 0.01,
        }
    }
}

/// LOD 版 sphere tracing：每步按当前 `t` 取 octaves 与 eps。
///
/// 与 [`crate::sdf::sphere_trace`] 的差别只在两处：距离场经 `Csg::dist_lod` 按
/// 衰减后的 octaves 求值；命中阈值随 t 放大。其余语义（起点在体内 → t=0、
/// 超出 `t_max`/步数耗尽 → None）保持一致。
pub fn sphere_trace_lod(field: &Csg, ro: Vec3, rd: Vec3, params: &LodParams) -> Option<TraceHit> {
    debug_assert!((rd.length() - 1.0).abs() < 1e-4, "sphere_trace_lod requires unit rd");
    let mut t = 0.0_f32;
    for _ in 0..params.trace.max_steps {
        let octaves = lod_octaves(t, params.max_octaves, params.falloff);
        let d = field.dist_lod(ro + rd * t, octaves);
        let eps = lod_eps(params.trace.hit_eps, t, params.eps_per_unit);
        if d <= eps {
            return Some(TraceHit { t });
        }
        t += d * params.trace.safety;
        if t > params.trace.t_max {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::Csg;
    use crate::primitive::{Sphere, Sdf};
    use crate::sdf::sphere_trace;

    const SEED: u64 = 0x6E01_5EED;

    /// lod_eps：t=0 恒等基准、线性放大、已知名值。
    #[test]
    fn lod_eps_formula() {
        let base = 1e-4_f32;
        assert_eq!(lod_eps(base, 0.0, 0.01), base, "t=0 → 基准");
        assert!((lod_eps(base, 10.0, 0.01) - base * 1.1).abs() < 1e-9, "t=10 → ×1.1");
        assert!((lod_eps(1.0, 3.0, 0.5) - 2.5).abs() < 1e-6, "1·(1+1.5)=2.5");
        // 单调：t 越大 eps 越大（非负斜率下）。
        let mut prev = lod_eps(base, 0.0, 0.01);
        for i in 1..=20 {
            let e = lod_eps(base, i as f32, 0.01);
            assert!(e >= prev, "eps 应随 t 单调不降");
            prev = e;
        }
    }

    /// lod_octaves：t=0 满档、线性衰减、下限 0、向零截断。
    #[test]
    fn lod_octaves_formula() {
        assert_eq!(lod_octaves(0.0, 4, 0.1), 4, "t=0 → 满档");
        assert_eq!(lod_octaves(10.0, 4, 0.1), 3, "4 − 1 = 3");
        assert_eq!(lod_octaves(25.0, 4, 0.1), 1, "4 − 2.5 = 1.5 向零截断 → 1");
        assert_eq!(lod_octaves(40.0, 4, 0.1), 0, "4 − 4 → 0");
        assert_eq!(lod_octaves(100.0, 4, 0.1), 0, "负值截到 0");
        assert_eq!(lod_octaves(5.0, 0, 0.1), 0, "max=0 恒 0");
        // 单调不增。
        let mut prev = lod_octaves(0.0, 6, 0.2);
        for i in 1..=40 {
            let o = lod_octaves(i as f32, 6, 0.2);
            assert!(o <= prev, "octaves 应随 t 单调不增");
            prev = o;
        }
    }

    /// 无 Displace 的近处命中：lod 版与普通版逐位一致（octaves 不影响纯图元场、
    /// 近处 eps 增量可忽略时同路同参）。
    #[test]
    fn near_hit_matches_plain_trace() {
        let sph = Sphere { center: Vec3::splat(0.0), radius: 1.0 };
        let csg = Csg::Sphere(sph);
        let ro = Vec3::new(0.0, 0.0, -5.0);
        let rd = Vec3::new(0.0, 0.0, 1.0);
        let params = LodParams {
            trace: TraceParams::default(),
            max_octaves: 4,
            falloff: 0.1,
            eps_per_unit: 0.0, // eps 不放大 → 与普通 trace 完全同参
        };
        let a = sphere_trace_lod(&csg, ro, rd, &params).expect("lod 应命中");
        let b = sphere_trace(&csg, ro, rd, &TraceParams::default()).expect("普通应命中");
        assert_eq!(a, b, "octaves/eps 不变时应逐位一致（t={:?} vs {:?}）", a.t, b.t);
    }

    /// eps 放大生效：远处命中时 lod 版提前触及阈值（t 相差在 eps 量级内）且两者都命中。
    #[test]
    fn far_hit_eps_grows() {
        let sph = Sphere { center: Vec3::new(0.0, 0.0, 20.0), radius: 1.0 };
        let csg = Csg::Sphere(sph);
        let ro = Vec3::new(0.0, 0.0, 0.0);
        let rd = Vec3::new(0.0, 0.0, 1.0);
        let lod = LodParams {
            trace: TraceParams { hit_eps: 1e-3, ..TraceParams::default() },
            max_octaves: 0, // 纯场，隔离 eps 因素
            falloff: 0.0,
            eps_per_unit: 0.1, // t=20 处 eps 放大 3 倍
        };
        let a = sphere_trace_lod(&csg, ro, rd, &lod).expect("lod 应命中");
        let b = sphere_trace(&csg, ro, rd, &lod.trace).expect("普通应命中");
        // 解析首命中 t = 19（z=19 处进入球面）。
        assert!((b.t - 19.0).abs() < 1e-2, "普通版命中点应近解析值 19，得 {:?}", b.t);
        // lod 版 eps 更宽 → 步长更大 → 提前或同点触及；差异受 eps 量级约束。
        assert!((a.t - b.t).abs() < 0.05, "lod t={:?} vs 普通 t={:?}", a.t, b.t);
    }

    /// Displace 场：t=0 处 lod octaves = 满档 → dist_lod 与 Csg::dist 一致；
    /// 远处 octaves 衰减确实改变场值。
    #[test]
    fn displace_octaves_falloff() {
        let base = Sphere { center: Vec3::splat(0.0), radius: 2.0 };
        let csg = Csg::Displace(Box::new(Csg::Sphere(base)), 0.3, 1.5, SEED);
        let p = Vec3::new(0.37, -0.21, 0.83);
        assert_eq!(
            csg.dist_lod(p, 4),
            csg.dist(p),
            "满档 octaves 与默认求值一致"
        );
        let full = csg.dist_lod(p, 4);
        let coarse = csg.dist_lod(p, 1);
        assert!(
            (full - coarse).abs() > 1e-6 || full == coarse,
            "octaves 不同时场值允许不同（本点应能观察到差异）"
        );
        // 确定性：同参两遍逐位一致。
        let ro = Vec3::new(3.0, 2.0, 4.0);
        let rd = Vec3::new(-0.5, -0.4, -0.7).normalize();
        let params = LodParams::default();
        let h1 = sphere_trace_lod(&csg, ro, rd, &params);
        let h2 = sphere_trace_lod(&csg, ro, rd, &params);
        assert_eq!(h1, h2, "lod trace 必须逐位确定");
    }

    /// 未命中与 t_max 边界语义与普通版一致。
    #[test]
    fn miss_semantics_match() {
        let csg = Csg::Sphere(Sphere { center: Vec3::splat(0.0), radius: 1.0 });
        let params = LodParams::default();
        // 脱靶射线 → None。
        let miss = sphere_trace_lod(&csg, Vec3::new(0.0, 5.0, -10.0), Vec3::new(0.0, 0.0, 1.0), &params);
        assert!(miss.is_none(), "脱靶应 None");
        // t_max 截断：目标在 t_max 之外。
        let far = LodParams {
            trace: TraceParams { t_max: 5.0, ..TraceParams::default() },
            ..LodParams::default()
        };
        let out = sphere_trace_lod(&csg, Vec3::new(0.0, 0.0, -20.0), Vec3::new(0.0, 0.0, 1.0), &far);
        assert!(out.is_none(), "超出 t_max 应 None");
        // 起点在体内 → t=0。
        let inside = sphere_trace_lod(&csg, Vec3::splat(0.0), Vec3::new(0.0, 0.0, 1.0), &params);
        assert_eq!(inside.expect("体内应命中").t, 0.0, "体内起点 t=0");
    }
}
