//! 曲线图元三角化（V2.2-A-2，方案乙）—CPU 侧权威，落 `prism-geometry`（叶子 crate）。
//!
//! # 方案裁定（见 `docs/v2.2-增量设计.md` §1.2）
//!
//! GPU 侧解析求交（方案甲）在 D-29 零超越函数纪律下无合法实现（四次方程预解三次
//! 在 Δ≤0 分支走 `acos`/`cos`，GPU 发射器只许 `Sqrt/FMax/FClamp/Floor`），故取乙：
//! 曲线图元在 CPU 侧三角化为网格，交给我们已有的 SAH-BVH（`crate::bvh`）做射线求交，
//! 再由 GPU 走既有 Möller–Trumbore + 栈式 BVH 遍历（WGSL `rt_bvh.wgsl` 同式）。
//!
//! # 确定性 / 零超越（D-29）
//!
//! 三角化只用 `+ − × ÷ sqrt` 与 `floor`；圆周顶点需要 sin/cos，一律走 CPU 侧
//! `crate::det_math` 确定性件（`det_cos_sin`，纯 IEEE、跨平台逐位一致，禁 libm）。
//! 旋转（局部轴 Y → 世界轴）用 Rodrigues 公式，其中 `cosθ = Y·axis` 直接由点积得到，
//! `sinθ = √(1−cos²θ)`，全程不落任何超越函数。
//!
//! # 保真度界（PRD V2.2-A-2 验收 4；复用 V2.1 R3 `hausdorff_upper_bound` 方法学）
//!
//! 三角化误差上界声明为
//!
//! ```text
//! d_H(S(曲线), mesh) ≤ r_max · (1 − cos(π / seg))
//! ```
//!
//! `seg` = 一周切分段数（`seg = ⌈2π·r / tol⌉`），`r_max` = 该图元最大半径。
//! `seg↑ ⇒ 界↓`（cos 在 `[0,π]` 单调降）。
//! 测试对球面特例（圆环 `major = 0` 退化为球）实测 Hausdorff ≤ 该界，并断言 `seg↑ ⇒ 界↓`。
//!
//! # 退化形态（PRD 验收 3）
//!
//! `r=0 / h=0 / profile 单点 / 空集合 / 非有限` 一律 → 空三角形集（不 panic）。

use crate::bvh::Triangle;
use crate::det_math::det_cos_sin;
use prism_math::Vec3;

/// `2π` 常数（CPU 侧，确定值）。
const TWO_PI: f32 = 6.2831853; // ≈ 2·π，与 `det_math` 的 `TWO_PI` 同值口径
/// 退化切分下限（保证 mesh 非退化）。
const MIN_SEG: u32 = 3;
/// `axis` 归一化阈值（小于此视为未定义轴）。
const AXIS_EPS: f32 = 1e-9;

/// 曲线图元请求（四种：圆柱 / 圆锥 / 圆环 / 旋转曲面）。
///
/// 所有几何量均为世界坐标；`axis` 为图元的对称轴（单位向量，无需调用方归一，
/// 本模块内部 `orient` 只用其方向）。
#[derive(Debug, Clone, PartialEq)]
pub enum CurveReq {
    /// 圆柱：中心 `center`、半径 `radius`、高 `height`、对称轴 `axis`。
    Cylinder {
        center: Vec3,
        radius: f32,
        height: f32,
        axis: Vec3,
    },
    /// 圆锥：顶点在 `+height/2`、底半径 `radius`、高 `height`、对称轴 `axis`。
    Cone {
        center: Vec3,
        radius: f32,
        height: f32,
        axis: Vec3,
    },
    /// 圆环：主环半径 `major`、管半径 `minor`、对称轴 `axis`。
    /// `major = 0` 时退化为球（半径 `minor`），用于保真度界的球面特例。
    Torus {
        center: Vec3,
        major: f32,
        minor: f32,
        axis: Vec3,
    },
    /// 旋转曲面：`profile` 为局部平面内的一条折线（`x` = 到轴距离，`y` = 沿轴高度，
    /// 绕 `axis` 旋转 `segments` 段生成）。`profile` 长度 < 2（"单点"）或 `segments` < 3
    /// 一律退化。
    Revolution {
        center: Vec3,
        profile: Vec<Vec3>,
        segments: u32,
        axis: Vec3,
    },
}

impl CurveReq {
    /// 结构合法性（退化形态返回 `false`，`tessellate` 据此返回空集）。
    ///
    /// 覆盖 PRD 验收 3 的全部退化面：`r=0`/`h=0`/`profile 单点`/`非有限`/`axis` 未定义。
    /// 空集合（无曲线）由调用方在 `curves.is_empty()` 处短路，不进此判据。
    pub fn validate(&self) -> bool {
        let finite_vec = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        match self {
            CurveReq::Cylinder { radius, height, axis, .. } => {
                radius.is_finite()
                    && height.is_finite()
                    && *radius > 0.0
                    && *height > 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Cone { radius, height, axis, .. } => {
                radius.is_finite()
                    && height.is_finite()
                    && *radius > 0.0
                    && *height > 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Torus { major, minor, axis, .. } => {
                major.is_finite()
                    && minor.is_finite()
                    && *minor > 0.0
                    && *major >= 0.0
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
            }
            CurveReq::Revolution { profile, segments, axis, .. } => {
                profile.len() >= 2
                    && *segments >= MIN_SEG
                    && finite_vec(*axis)
                    && axis.length() > AXIS_EPS
                    && profile.iter().all(|p| p.x >= 0.0 && p.x.is_finite() && p.y.is_finite())
            }
        }
    }

    /// 该图元的最大半径 `r_max`（保真度界用）。
    ///
    /// 圆柱/圆锥 = 底半径；圆环 = `major + minor`；旋转曲面 = profile 最大 `x`。
    pub fn max_radius(&self) -> f32 {
        match self {
            CurveReq::Cylinder { radius, .. } => *radius,
            CurveReq::Cone { radius, .. } => *radius,
            CurveReq::Torus { major, minor, .. } => major + minor,
            CurveReq::Revolution { profile, .. } => {
                profile.iter().map(|p| p.x).fold(0.0_f32, f32::max)
            }
        }
    }
}

/// 由容忍误差 `tol` 推算某一半径对应的周向切分段数（`seg = ⌈2π·r / tol⌉`，下限 3）。
///
/// `tol` 为相邻采样点间的最大弧长（弦高预算）；确定性 ceil 实现（禁浮点 `ceil` 语义漂移）。
fn seg_for_radius(r: f32, tol: f32) -> u32 {
    let tol = tol.max(1e-6);
    let raw = TWO_PI * r / tol;
    let raw_floor = raw.floor();
    let seg = if (raw - raw_floor) > 1e-9 {
        raw_floor as u32 + 1
    } else {
        raw_floor as u32
    };
    seg.max(MIN_SEG)
}

/// 网格相对「界段数」的加密系数（A-2 保真度界能成立的关键）。
///
/// `hausdorff_bound` 给出的是**单方向**弦高 `r·(1 − cos(π/seg))`；而经纬网格的
/// 四边形最大偏离点在其**面心**，相对顶点的角距是 `√2·π/seg`（对角方向），故真实
/// 偏离约为 `r·(1 − cos(√2·π/S))`（`S` = 网格实际段数）。要令
/// `d_H ≤ r·(1 − cos(π/seg))` 成立，必须 `S ≥ √2·seg`；取 1.5 留 6% 数值裕度
/// （实测比值 0.89，见 `tests/qa_curve_tessellation.rs`）。
const TESS_DENSIFY: f32 = 1.5;

/// 网格实际使用的周向段数（= `⌈TESS_DENSIFY · seg_count⌉`）。
///
/// `seg_count` 仍按设计 §1.2.2 的 `⌈2π·r / tol⌉` 口径（界函数的入参），加密只
/// 发生在**网格生成**侧——界不因此放松，只会更保守。
fn tess_seg(r: f32, tol: f32) -> u32 {
    let s = seg_for_radius(r, tol) as f32 * TESS_DENSIFY;
    let f = s.floor();
    let up = if (s - f) > 1e-6 { f + 1.0 } else { f };
    (up as u32).max(MIN_SEG)
}

/// 该曲线在容忍误差 `tol` 下的**有效**周向切分段数（各方向取较粗者 = 较小 `seg`，
/// 保证界为上界）。
///
/// 圆柱/圆锥 = 侧壁段数；圆环 = `min(seg_u, seg_v)`；旋转曲面 = `min(seg_profile, segments)`。
pub fn seg_count(curve: &CurveReq, tol: f32) -> u32 {
    match curve {
        CurveReq::Cylinder { radius, .. } => seg_for_radius(*radius, tol),
        CurveReq::Cone { radius, .. } => seg_for_radius(*radius, tol),
        CurveReq::Torus { major, minor, .. } => {
            let su = seg_for_radius(major + minor, tol);
            let sv = seg_for_radius(*minor, tol);
            su.min(sv)
        }
        CurveReq::Revolution { profile, segments, .. } => {
            let max_r = profile.iter().map(|p| p.x).fold(0.0_f32, f32::max);
            seg_for_radius(max_r, tol).min(*segments)
        }
    }
}

/// 保真度界 `r_max · (1 − cos(π / seg))`。
///
/// `seg` 即 [`seg_count`] 的返回值；`seg↑ ⇒ 界↓`（cos 在 `[0, π]` 单调降）。
pub fn hausdorff_bound(curve: &CurveReq, seg: u32) -> f32 {
    // 退化/非法形态 → 空网格 → 保真度误差精确为 0（且必须是**有限值**：
    // 半径含 NaN/∞ 时 `max_radius()` 会把 NaN 传染到界上，破坏"界可比较"的契约）。
    if !curve.validate() {
        return 0.0;
    }
    let seg = seg.max(1) as f32;
    let (c, _) = det_cos_sin(std::f32::consts::PI / seg);
    curve.max_radius() * (1.0 - c)
}

/// 三角化入口：`tessellate(&CurveReq, tol) -> Vec<Triangle>`。
///
/// 非法（`!validate()`）或空集合 → 返回空 `Vec`（退化形态不 panic）。
pub fn tessellate(curve: &CurveReq, tol: f32) -> Vec<Triangle> {
    if !curve.validate() {
        return Vec::new();
    }
    match curve {
        CurveReq::Cylinder { center, radius, height, axis } => {
            tess_cylinder(*center, *radius, *height, *axis, tol)
        }
        CurveReq::Cone { center, radius, height, axis } => {
            tess_cone(*center, *radius, *height, *axis, tol)
        }
        CurveReq::Torus { center, major, minor, axis } => {
            tess_torus(*center, *major, *minor, *axis, tol)
        }
        CurveReq::Revolution { center, profile, segments, axis } => {
            tess_revolution(*center, profile, *segments, *axis, tol)
        }
    }
}

/// 三角化全部曲线并拼接（空曲线自动跳过，非法曲线贡献空集）。
pub fn tessellate_all(curves: &[CurveReq], tol: f32) -> Vec<Triangle> {
    let mut out = Vec::new();
    for c in curves {
        if c.validate() {
            out.extend(tessellate(c, tol));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 几何辅助
// ---------------------------------------------------------------------------

/// 把局部点（局部对称轴 = Y）经 Rodrigues 旋转到世界 `axis` 方向，再加 `center`。
///
/// 旋转把局部 Y 映射到 `axis`：`cosθ = Y·axis`（直接点积，无需 `acos`），
/// `sinθ = √(1 − cos²θ)`，`k = (Y × axis)` 归一为旋转轴。全程仅 `+ − × ÷ sqrt`。
fn orient(axis: Vec3, center: Vec3, local: Vec3) -> Vec3 {
    let y_axis = Vec3::new(0.0, 1.0, 0.0);
    let c = axis.dot(&y_axis).clamp(-1.0, 1.0); // cosθ
    let s = (1.0 - c * c).max(0.0).sqrt(); // sinθ
    if s < 1e-7 {
        // 轴 ≈ ±Y：c > 0 ⇒ 恒等；c < 0 ⇒ 绕 X 转 180°（Y → −Y）。
        if c > 0.0 {
            return local + center;
        }
        return Vec3::new(local.x, -local.y, -local.z) + center;
    }
    let k = y_axis.cross(&axis).normalize(); // 旋转轴（单位）
    let kp = k.dot(&local);
    let cross_kp = k.cross(&local);
    // Rodrigues: p' = p·c + (k × p)·s + k·(k·p)·(1 − c)
    let rotated = local * c + cross_kp * s + k * (kp * (1.0 - c));
    rotated + center
}

/// 推入一个三角形（顶点顺序仅影响法线朝向；着色时按"朝向射线原点"翻转，故此处随意）。
#[inline]
fn push_tri(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3) {
    out.push(Triangle { a, b, c });
}

/// 推入一个四边形（`a→b→c→d` 顺序）为两个三角形。
#[inline]
fn push_quad(out: &mut Vec<Triangle>, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
    push_tri(out, a, b, c);
    push_tri(out, a, c, d);
}

/// 周向环上一点（局部轴 Y，半径 `r`，高度 `y`，角度 `t`）。
#[inline]
fn ring_point(r: f32, y: f32, t: f32) -> Vec3 {
    let (ct, st) = det_cos_sin(t);
    Vec3::new(r * ct, y, r * st)
}

// ---------------------------------------------------------------------------
// 四类图元三角化
// ---------------------------------------------------------------------------

/// 圆柱：侧壁 `seg` 四边形（每个四边形 = 2 三角形）+ 上下盖扇形（`seg` 三角形各）。
fn tess_cylinder(
    center: Vec3,
    radius: f32,
    height: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let seg = tess_seg(radius, tol);
    let hy = height * 0.5;
    let mut tris = Vec::with_capacity((seg as usize) * 4);
    // 侧壁：seg 个四边形（每个 2 三角形）。
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let p0b = orient(axis, center, ring_point(radius, -hy, t0));
        let p1b = orient(axis, center, ring_point(radius, -hy, t1));
        let p0t = orient(axis, center, ring_point(radius, hy, t0));
        let p1t = orient(axis, center, ring_point(radius, hy, t1));
        push_quad(&mut tris, p0b, p1b, p1t, p0t);
    }
    // 上下盖扇形：各 seg 三角形。
    let ct = orient(axis, center, Vec3::new(0.0, hy, 0.0));
    let cb = orient(axis, center, Vec3::new(0.0, -hy, 0.0));
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let r0 = orient(axis, center, ring_point(radius, hy, t0));
        let r1 = orient(axis, center, ring_point(radius, hy, t1));
        push_tri(&mut tris, ct, r0, r1);
        let b0 = orient(axis, center, ring_point(radius, -hy, t0));
        let b1 = orient(axis, center, ring_point(radius, -hy, t1));
        push_tri(&mut tris, cb, b1, b0);
    }
    tris
}

/// 圆锥：侧壁 `seg` 三角形（顶点 → 底环）+ 底盖扇形 `seg` 三角形。
fn tess_cone(
    center: Vec3,
    radius: f32,
    height: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let seg = tess_seg(radius, tol);
    let hy = height * 0.5;
    let mut tris = Vec::with_capacity((seg as usize) * 2);
    let apex = orient(axis, center, Vec3::new(0.0, hy, 0.0));
    let cb = orient(axis, center, Vec3::new(0.0, -hy, 0.0));
    for i in 0..seg {
        let t0 = TWO_PI * (i as f32) / (seg as f32);
        let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
        let b0 = orient(axis, center, ring_point(radius, -hy, t0));
        let b1 = orient(axis, center, ring_point(radius, -hy, t1));
        push_tri(&mut tris, apex, b0, b1); // 侧壁三角形
        push_tri(&mut tris, cb, b1, b0); // 底盖扇形
    }
    tris
}

/// 圆环：主环 `seg_u` × 管 `seg_v` 四边形（每个 2 三角形）。
///
/// 球面特例（`major = 0`）退化为球：`seg_u = seg_v = seg`，界 = `minor·(1 − cos(π/seg))`。
fn tess_torus(
    center: Vec3,
    major: f32,
    minor: f32,
    axis: Vec3,
    tol: f32,
) -> Vec<Triangle> {
    let su = tess_seg(major + minor, tol);
    let sv = tess_seg(minor, tol);
    let mut tris = Vec::with_capacity((su as usize) * (sv as usize) * 2);
    let pt = |u: f32, v: f32| -> Vec3 {
        let (cu, su_) = det_cos_sin(u);
        let (cv, sv_) = det_cos_sin(v);
        let rr = major + minor * cv;
        orient(axis, center, Vec3::new(rr * cu, minor * sv_, rr * su_))
    };
    for iu in 0..su {
        let u0 = TWO_PI * (iu as f32) / (su as f32);
        let u1 = TWO_PI * ((iu + 1) as f32) / (su as f32);
        for iv in 0..sv {
            let v0 = TWO_PI * (iv as f32) / (sv as f32);
            let v1 = TWO_PI * ((iv + 1) as f32) / (sv as f32);
            let a = pt(u0, v0);
            let b = pt(u1, v0);
            let c = pt(u1, v1);
            let d = pt(u0, v1);
            push_quad(&mut tris, a, b, c, d);
        }
    }
    tris
}

/// 旋转曲面：`(profile.len() − 1)` 段 × `segments` 四边形。
///
/// `profile[j] = (x = 到轴距离, y = 沿轴高度, z 忽略)`；水平段（`x` 不变）特化为环带
/// 但仍三角化（保退化一致：与其它段走同一路径，不特殊处理）。
fn tess_revolution(
    center: Vec3,
    profile: &[Vec3],
    segments: u32,
    axis: Vec3,
    _tol: f32,
) -> Vec<Triangle> {
    if profile.len() < 2 {
        return Vec::new();
    }
    let seg = segments.max(MIN_SEG);
    let mut tris = Vec::with_capacity((profile.len() - 1) * (seg as usize) * 2);
    for j in 0..profile.len() - 1 {
        let pa = profile[j];
        let pb = profile[j + 1];
        for i in 0..seg {
            let t0 = TWO_PI * (i as f32) / (seg as f32);
            let t1 = TWO_PI * ((i + 1) as f32) / (seg as f32);
            let (c0, s0) = det_cos_sin(t0);
            let (c1, s1) = det_cos_sin(t1);
            let a = orient(axis, center, Vec3::new(pa.x * c0, pa.y, pa.x * s0));
            let b = orient(axis, center, Vec3::new(pb.x * c0, pb.y, pb.x * s0));
            let c = orient(axis, center, Vec3::new(pb.x * c1, pb.y, pb.x * s1));
            let d = orient(axis, center, Vec3::new(pa.x * c1, pa.y, pa.x * s1));
            push_quad(&mut tris, a, b, c, d);
        }
    }
    tris
}

// ---------------------------------------------------------------------------
// 单元测试（native：保真度界 / 退化形态 / 确定性 / 几何合理性）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::Vec3;

    /// 退化形态：r=0 / h=0 / profile 单点 / 非有限 → 一律空集（不 panic）。
    #[test]
    fn degenerate_forms_yield_empty_mesh() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let center = Vec3::new(0.0, 0.0, 0.0);
        let cases: Vec<CurveReq> = vec![
            // r = 0
            CurveReq::Cylinder { center, radius: 0.0, height: 1.0, axis },
            CurveReq::Cone { center, radius: 0.0, height: 1.0, axis },
            CurveReq::Torus { center, major: 1.0, minor: 0.0, axis },
            // h = 0
            CurveReq::Cylinder { center, radius: 1.0, height: 0.0, axis },
            CurveReq::Cone { center, radius: 1.0, height: 0.0, axis },
            // profile 单点（长度 < 2）
            CurveReq::Revolution {
                center,
                profile: vec![Vec3::new(1.0, 0.0, 0.0)],
                segments: 8,
                axis,
            },
            // 非有限
            CurveReq::Cylinder { center, radius: f32::NAN, height: 1.0, axis },
            CurveReq::Cone { center, radius: 1.0, height: f32::INFINITY, axis },
            CurveReq::Torus { center, major: f32::NAN, minor: 1.0, axis },
        ];
        for c in &cases {
            assert!(!c.validate(), "退化形态必须被判非法");
            assert_eq!(
                tessellate(c, 0.01),
                Vec::new(),
                "退化形态必须返回空 mesh（不 panic）"
            );
        }
    }

    /// 合法图元三角化：网格非空且为有限顶点。
    #[test]
    fn valid_curves_tessellate_to_finite_mesh() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let center = Vec3::new(0.5, -0.3, 0.2);
        let cyl = CurveReq::Cylinder { center, radius: 1.0, height: 2.0, axis };
        let cone = CurveReq::Cone { center, radius: 1.0, height: 2.0, axis };
        let torus = CurveReq::Torus { center, major: 1.0, minor: 0.4, axis };
        let rev = CurveReq::Revolution {
            center,
            profile: vec![
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            segments: 12,
            axis,
        };
        for c in [cyl, cone, torus, rev] {
            let tris = tessellate(&c, 0.05);
            assert!(!tris.is_empty(), "合法图元应有非空 mesh");
            for t in &tris {
                for v in [t.a, t.b, t.c] {
                    assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
                }
            }
        }
    }

    /// 确定性：同输入两次三角化逐位一致。
    #[test]
    fn tessellation_is_deterministic() {
        let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
        let curve = CurveReq::Torus {
            center: Vec3::new(0.0, 0.0, 0.0),
            major: 0.8,
            minor: 0.3,
            axis,
        };
        let a = tessellate(&curve, 0.03);
        let b = tessellate(&curve, 0.03);
        assert_eq!(a, b, "同输入三角化必须逐位一致");
    }
}
