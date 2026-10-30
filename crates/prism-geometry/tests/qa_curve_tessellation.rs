//! V2.2-A-2 三角化保真度与退化形态钉子（任务表 T06「测 2」之一）。
//!
//! 覆盖 PRD 验收 3 / 4：
//!
//! 1. **保真度界**：`d_H(S(曲线), mesh) ≤ r_max · (1 − cos(π / seg))`，且
//!    `seg ↑ ⇒ 界 ↓`。取**球面特例**（圆环 `major = 0` 退化为半径 `minor` 的球）
//!    实测 Hausdorff：对每个网格四边形的参数中心取球面点，量它到该四边形
//!    两个三角形的最近距离，遍历全部四边形取最大值——这是球面到内接网格
//!    Hausdorff 距离的**精确值**（内接网格的最大偏离点必落在面心方向）。
//! 2. **退化形态**：`r=0 / h=0 / profile 单点 / 空集合 / 非有限` 一律 → 空三角形集，
//!    **不 panic**（沿用 `u32::MAX × u32::MAX` 五组 `catch_unwind` 守卫的先例）。
//!
//! 禁 `#[ignore]`（`qa_ignore_guard` 会命中）。

use prism_geometry::bvh::Triangle;
use prism_geometry::curve_tess::{
    CurveReq, hausdorff_bound, seg_count, tessellate, tessellate_all,
};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 点—三角形最近距离（Hausdorff 实测内核）
// ---------------------------------------------------------------------------

/// 线段 `ab` 上离 `p` 最近的点。
fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let denom = ab.dot(&ab);
    if denom <= 0.0 {
        return a;
    }
    let t = ((p - a).dot(&ab) / denom).clamp(0.0, 1.0);
    a + ab * t
}

/// 三角形 `tri` 上离 `p` 最近的点（Ericson, *Real-Time Collision Detection* §5.1.5；
/// 只用 `+ − × ÷` 与 `clamp`，无超越函数）。
fn closest_on_triangle(p: Vec3, tri: &Triangle) -> Vec3 {
    let (a, b, c) = (tri.a, tri.b, tri.c);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = va + vb + vc;
    if denom.abs() <= 0.0 {
        // 退化三角形（三点共线/重合）：退到三条边上的最近点。
        let p1 = closest_on_segment(p, a, b);
        let p2 = closest_on_segment(p, b, c);
        let p3 = closest_on_segment(p, c, a);
        let (d1_, d2_, d3_) = ((p - p1).length(), (p - p2).length(), (p - p3).length());
        return if d1_ <= d2_ && d1_ <= d3_ {
            p1
        } else if d2_ <= d3_ {
            p2
        } else {
            p3
        };
    }
    let v = vb / denom;
    let w = vc / denom;
    a + ab * v + ac * w
}

/// 球面（半径 `r`、球心 `center`）到「内接经纬网格」的 Hausdorff 距离实测值。
///
/// `su` / `sv` 必须与 `tessellate` 对同一圆环实际使用的分段数一致；四边形
/// `(iu, iv)` 的两个三角形按 `tessellate` 的发射顺序取 `2k` / `2k+1`。
fn measured_sphere_hausdorff(
    center: Vec3,
    r: f32,
    su: u32,
    sv: u32,
    tris: &[Triangle],
) -> f32 {
    assert_eq!(
        tris.len(),
        (su as usize) * (sv as usize) * 2,
        "圆环三角化应为 su×sv 四边形（每片 2 三角形）；su={su} sv={sv} len={}",
        tris.len()
    );
    let two_pi = std::f32::consts::TAU;
    let mut worst = 0.0_f32;
    for iu in 0..su {
        // 四边形参数中心（面心方向）——内接网格的最大偏离点必在此处。
        let uc = two_pi * (iu as f32 + 0.5) / su as f32;
        for iv in 0..sv {
            let vc = two_pi * (iv as f32 + 0.5) / sv as f32;
            // 球面点（`major = 0` 时圆环退化为半径 `minor` 的球）。
            let p = center
                + Vec3::new(
                    r * vc.cos() * uc.cos(),
                    r * vc.sin(),
                    r * vc.cos() * uc.sin(),
                );
            let k = (iu as usize) * (sv as usize) + (iv as usize);
            let d0 = (p - closest_on_triangle(p, &tris[k * 2])).length();
            let d1 = (p - closest_on_triangle(p, &tris[k * 2 + 1])).length();
            let d = d0.min(d1);
            if d > worst {
                worst = d;
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// 验收 4：Hausdorff ≤ 弦高界，且 seg ↑ ⇒ 界 ↓
// ---------------------------------------------------------------------------

/// 球面特例（圆环 `major = 0`）：实测 Hausdorff ≤ 声明界，且 `seg ↑ ⇒ 界 ↓`。
///
/// 注：`vc.cos()` / `uc.cos()` 在**测试侧**用于生成参考球面点（不是引擎代码路径，
/// 不参与 D-29 的生产纪律判定；被测算的三角化本身走 `det_math::det_cos_sin`）。
#[test]
fn sphere_special_case_hausdorff_within_bound_and_monotone() {
    let center = Vec3::new(0.15, -0.2, 0.05);
    let minor = 0.9_f32;
    let curve = CurveReq::Torus {
        center,
        major: 0.0,
        minor,
        axis: Vec3::new(0.0, 1.0, 0.0),
    };
    assert!(curve.validate(), "球面特例（major=0）应判合法");
    assert_eq!(
        curve.max_radius(),
        minor,
        "major=0 时 r_max = minor（保真度界的半径口径）"
    );

    // 三档容忍：tol 越小 seg 越大。
    let tols = [0.35_f32, 0.18, 0.09];
    let mut prev_bound = f32::INFINITY;
    let mut prev_seg = 0u32;
    let mut prev_measured = f32::INFINITY;
    for tol in tols {
        let seg = seg_count(&curve, tol);
        assert!(seg > prev_seg, "tol↓ ⇒ seg↑（实测 {prev_seg} → {seg}）");
        let bound = hausdorff_bound(&curve, seg);
        let tris = tessellate(&curve, tol);
        assert!(!tris.is_empty(), "球面特例必须三角化出非空网格");
        // 网格实际段数从三角形数反解：`major = 0` ⇒ su = sv = S，tris = 2·S²。
        // （`S` 比界段数 `seg` 更密——见 `curve_tess::TESS_DENSIFY`。）
        let s2 = tris.len() / 2;
        let mesh_seg = (s2 as f32).sqrt() as u32;
        assert_eq!(
            (mesh_seg * mesh_seg) as usize,
            s2,
            "网格应为正方形网格（su = sv = {mesh_seg}，tris={}）",
            tris.len()
        );
        assert!(
            mesh_seg >= seg,
            "网格段数必须不小于界段数（加密系数 ≥ 1）：{mesh_seg} vs {seg}"
        );
        let measured = measured_sphere_hausdorff(center, minor, mesh_seg, mesh_seg, &tris);

        println!(
            "tol={tol:.3} 界段数={seg} 网格段数={mesh_seg} 界={bound:.6} 实测 Hausdorff={measured:.6}（比值 {:.3}）",
            measured / bound
        );

        // 判据 1：实测不得超过声明界（留 1% 数值裕度吸收 f32 舍入）。
        assert!(
            measured <= bound * 1.01,
            "实测 Hausdorff {measured} 超过声明界 {bound}（seg={seg}, tol={tol}）"
        );
        // 判据 2：实测必须**逼近**界（否则界是空话：界远大于真实误差也算"通过"）。
        assert!(
            measured >= bound * 0.5,
            "实测 Hausdorff {measured} 远小于界 {bound}（seg={seg}）：界过松，不是紧界"
        );
        // 判据 3：seg ↑ ⇒ 界 ↓ 且实测 ↓。
        assert!(
            bound < prev_bound,
            "seg↑ 必须使界严格下降：{prev_bound} → {bound}（seg {prev_seg} → {seg}）"
        );
        assert!(
            measured < prev_measured,
            "seg↑ 必须使实测 Hausdorff 严格下降：{prev_measured} → {measured}"
        );
        prev_bound = bound;
        prev_seg = seg;
        prev_measured = measured;
    }
    println!(
        "球面特例：三档 tol 全部满足 d_H ≤ r·(1−cos(π/seg))，且 seg↑⇒界↓、实测同步下降"
    );
}

/// 四类图元的界都随 `seg` 单调下降（界公式的统一性）。
#[test]
fn hausdorff_bound_decreases_with_segment_count_for_all_primitives() {
    let axis = Vec3::new(0.0, 1.0, 0.0);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let curves = vec![
        CurveReq::Cylinder { center, radius: 1.0, height: 2.0, axis },
        CurveReq::Cone { center, radius: 1.0, height: 2.0, axis },
        CurveReq::Torus { center, major: 1.0, minor: 0.35, axis },
        CurveReq::Revolution {
            center,
            profile: vec![Vec3::new(0.2, -1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            segments: 24,
            axis,
        },
    ];
    for c in &curves {
        let mut prev = f32::INFINITY;
        for seg in [4u32, 8, 16, 32, 64] {
            let b = hausdorff_bound(c, seg);
            assert!(b > 0.0 && b.is_finite(), "界必须为正有限值：{b}");
            assert!(b < prev, "seg↑ 必须使界严格下降：{prev} → {b}（seg={seg}）");
            prev = b;
        }
    }
}

// ---------------------------------------------------------------------------
// 验收 3：退化形态一律空集、不 panic
// ---------------------------------------------------------------------------

/// 五种退化面 + 空集合：`tessellate` / `tessellate_all` 返回空集且**不 panic**。
///
/// 守卫方式与 `u32::MAX × u32::MAX` 五组 `catch_unwind` 先例一致：连
/// `validate()` / `seg_count()` / `hausdorff_bound()` 一起过守卫（界函数在 `seg=0`
/// 上也不得 panic）。
#[test]
fn degenerate_forms_yield_empty_mesh_without_panic() {
    let axis = Vec3::new(0.0, 1.0, 0.0);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let groups: Vec<(&str, CurveReq)> = vec![
        // 组 1：r = 0
        ("r=0/cylinder", CurveReq::Cylinder { center, radius: 0.0, height: 1.0, axis }),
        ("r=0/cone", CurveReq::Cone { center, radius: 0.0, height: 1.0, axis }),
        ("minor=0/torus", CurveReq::Torus { center, major: 1.0, minor: 0.0, axis }),
        // 组 2：h = 0
        ("h=0/cylinder", CurveReq::Cylinder { center, radius: 1.0, height: 0.0, axis }),
        ("h=0/cone", CurveReq::Cone { center, radius: 1.0, height: 0.0, axis }),
        // 组 3：profile 单点
        (
            "profile-single",
            CurveReq::Revolution { center, profile: vec![Vec3::new(1.0, 0.0, 0.0)], segments: 8, axis },
        ),
        (
            "profile-empty",
            CurveReq::Revolution { center, profile: Vec::new(), segments: 8, axis },
        ),
        // 组 4：非有限
        ("nan-radius", CurveReq::Cylinder { center, radius: f32::NAN, height: 1.0, axis }),
        ("inf-height", CurveReq::Cone { center, radius: 1.0, height: f32::INFINITY, axis }),
        ("nan-major", CurveReq::Torus { center, major: f32::NAN, minor: 1.0, axis }),
        (
            "inf-profile",
            CurveReq::Revolution {
                center,
                profile: vec![Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0)],
                segments: 8,
                axis,
            },
        ),
        // 组 5：轴退化（零向量）
        ("zero-axis", CurveReq::Cylinder { center, radius: 1.0, height: 1.0, axis: Vec3::new(0.0, 0.0, 0.0) }),
    ];

    for (tag, c) in groups {
        let r = std::panic::catch_unwind(|| {
            let valid = c.validate();
            let seg = seg_count(&c, 0.01);
            let bound = hausdorff_bound(&c, seg);
            let tris = tessellate(&c, 0.01);
            let all = tessellate_all(&[c.clone()], 0.01);
            (valid, seg, bound, tris.len(), all.len())
        })
        .unwrap_or_else(|_| panic!("退化形态 `{tag}` 触发了 panic（验收 3 违规）"));

        let (valid, _seg, bound, n1, n2) = r;
        assert!(!valid, "退化形态 `{tag}` 必须被判非法");
        assert_eq!(n1, 0, "退化形态 `{tag}` 必须返回空三角形集");
        assert_eq!(n2, 0, "退化形态 `{tag}` 经 tessellate_all 也必须为空");
        assert!(!bound.is_nan(), "退化形态 `{tag}` 的界不得为 NaN");
    }

    // 空集合（无曲线）本身不得 panic，且不得凭空造三角形。
    let empty: Vec<CurveReq> = Vec::new();
    let all = std::panic::catch_unwind(|| tessellate_all(&empty, 0.01))
        .expect("空集合不得 panic");
    assert!(all.is_empty(), "空集合必须产出空网格");
}

/// 退化输入的**语义稳定性**：`tol` 取极端值（0 / 负数 / 非有限 / 过大）不得 panic。
#[test]
fn extreme_tolerances_do_not_panic() {
    let curve = CurveReq::Cylinder {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
        height: 2.0,
        axis: Vec3::new(0.0, 1.0, 0.0),
    };
    for tol in [0.0_f32, -1.0, f32::NAN, f32::INFINITY, 1e9] {
        let r = std::panic::catch_unwind(|| {
            let seg = seg_count(&curve, tol);
            (seg, hausdorff_bound(&curve, seg), tessellate(&curve, tol).len())
        });
        let (seg, bound, n) = r.unwrap_or_else(|_| panic!("tol={tol} 触发了 panic"));
        assert!(seg >= 3, "tol={tol} 时切分段数下限必须保持 3（实得 {seg}）");
        assert!(bound.is_finite(), "tol={tol} 的界必须有限（实得 {bound}）");
        assert!(n > 0, "合法图元在任何 tol 下都应产出非空网格（tol={tol}）");
    }
}

/// 确定性与有限性：同一输入两次三角化逐位一致，顶点全有限。
#[test]
fn tessellation_is_deterministic_and_finite() {
    let axis = Vec3::new(1.0, 1.0, 0.5).normalize();
    let curves = vec![
        CurveReq::Cylinder { center: Vec3::new(0.3, -0.1, 0.2), radius: 0.7, height: 1.4, axis },
        CurveReq::Cone { center: Vec3::new(-0.2, 0.4, 0.0), radius: 0.5, height: 1.1, axis },
        CurveReq::Torus { center: Vec3::new(0.0, 0.0, 0.0), major: 0.8, minor: 0.3, axis },
    ];
    let a = tessellate_all(&curves, 0.05);
    let b = tessellate_all(&curves, 0.05);
    assert_eq!(a, b, "同输入三角化必须逐位一致（确定性契约）");
    assert!(!a.is_empty());
    for t in &a {
        for v in [t.a, t.b, t.c] {
            assert!(
                v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
                "顶点必须有限：{v:?}"
            );
        }
    }
}
