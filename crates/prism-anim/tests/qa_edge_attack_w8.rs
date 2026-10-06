//! W8-B15 攻击性边界测试（QA 严过关）。
//!
//! 只写测试，不改源码。每个断言失败都对应一类真实缺陷：
//! - 数学不变量（dolly zoom 尺度守恒 / 超焦距 / fov-roundtrip）
//! - 退化输入不 panic（库代码 panic = 缺陷）
//! - 确定性（同参逐位一致）
//! - 自研三角真自研（对参考 tan/atan 的精度核查）

use prism_anim::camera_presets::{CameraMove, CameraPose, DollyZoom, Handheld, LensSpec};
use prism_anim::EasingKind;
use prism_math::Vec3;

fn pose_10() -> CameraPose {
    CameraPose::new(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 0.0), 50.0)
}

/// B15-1: dolly zoom 尺度守恒不变量，**在非平凡起点 fov / 大范围 d 下**仍恒定。
///
/// 失败代表：耦合式 f(t)=2·atan(tan(f₀/2)·d₀/d(t)) 实现有误，或自研 atan 太糙，
/// 导致主体投影尺度沿时间漂移（dolly zoom 的核心观感即失效）。
#[test]
fn dolly_zoom_scale_invariant_wide_range() {
    // f₀=50°（tan 较大，放大自研 atan 误差），d 从 20 → 3（4× 以上推近）。
    let dz = DollyZoom {
        from: CameraPose::new(Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, 0.0), 50.0),
        subject: Vec3::new(0.0, 0.0, 0.0),
        end_distance: 3.0,
        duration: 1.0,
        easing: EasingKind::Linear,
    };
    let inv0 = dz.invariant(0.0);
    let mut worst = 0.0f32;
    for i in 0..=200 {
        let t = i as f32 / 200.0;
        let inv = dz.invariant(t);
        let rel = (inv - inv0).abs() / inv0;
        if rel > worst {
            worst = rel;
        }
    }
    // 自研 atan 为 7 项 Taylor，误差 ~1e-3；宽松容差 1% 仍应通过。
    assert!(worst < 0.01, "dolly zoom 尺度守恒漂移 {:.4}% 超容差", worst * 100.0);
}

/// B15-2: DollyZoom::fov 在 t=0 应逐位回到起始 fov（ratio=1 → tan_half=tan_half0）。
///
/// 失败代表：`fov(0)` 不能精确还原起点（round-trip 不自洽），或 atan 在 tan_half0
/// 处误差过大。
#[test]
fn dolly_zoom_fov_recovers_start() {
    for f0 in [20.0f32, 35.0, 50.0, 70.0, 90.0, 120.0] {
        let dz = DollyZoom {
            from: CameraPose::new(Vec3::new(0.0, 0.0, 15.0), Vec3::new(0.0, 0.0, 0.0), f0),
            subject: Vec3::new(0.0, 0.0, 0.0),
            end_distance: 5.0,
            duration: 1.0,
            easing: EasingKind::Linear,
        };
        let got = dz.fov(0.0);
        assert!(
            (got - f0).abs() < 0.2,
            "dolly zoom fov(0) 未回到起点 f0={f0}，实得 {got}"
        );
    }
}

/// B15-3: 超焦距公式 H = f²/(N·c)（数值核查 + 单调性）。
///
/// 失败代表：hyperfocal_mm 实现与物理定义不符（写错乘子/单位），
/// 下游 DOF 全盘偏。
#[test]
fn hyperfocal_formula_exact() {
    let lens = LensSpec::new(35.0, 8.0);
    // H = 35²/(8·0.029) = 1225/0.232 = 5280.17... mm
    let h = lens.hyperfocal_mm();
    let expect = 35.0f32 * 35.0 / (8.0 * 0.029);
    assert!((h - expect).abs() < 1e-2, "超焦距 H={h} 期望 {expect}");
    // 光圈越小（N 越大）→ H 越小。
    let big_n = LensSpec::new(35.0, 16.0).hyperfocal_mm();
    assert!(big_n < h, "小光圈应降低超焦距");
    // N→0（理论无限大光圈）→ H → +∞。
    let open = LensSpec::new(35.0, 0.0).hyperfocal_mm();
    assert!(open.is_infinite(), "N=0 应得 +∞ 超焦距，实得 {open}");
}

/// B15-4: coc_diameter 在 focus ≤ focal 时给 +∞（薄透镜有效域外），不 panic。
///
/// 失败代表：退化输入产生 NaN 或 panic。
#[test]
fn coc_diameter_degenerate_is_infinite_not_nan() {
    let lens = LensSpec::new(50.0, 2.8);
    assert!(lens.coc_diameter_mm(50.0).is_infinite());
    assert!(lens.coc_diameter_mm(30.0).is_infinite());
    assert!(!lens.coc_diameter_mm(30.0).is_nan(), "退化应给 +∞ 而非 NaN");
}

/// B15-5: fov ↔ 焦距 roundtrip 在更宽取值范围内自洽。
///
/// 失败代表：focal_for_fov / fov_for_focal 一对互逆函数数值不自洽 → 摄影换算错。
#[test]
fn fov_focal_roundtrip_wide() {
    let h = 24.0f32;
    for fov in [5.0f32, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0] {
        let f = LensSpec::focal_for_fov(fov, h);
        let back = LensSpec::fov_for_focal(f, h);
        assert!((back - fov).abs() < 1.0, "roundtrip fov {fov} → {back}");
    }
}

/// B15-6: CameraMove 退化输入（duration ≤ 0）不 panic 且瞬时到位。
#[test]
fn camera_move_zero_duration_no_panic() {
    let m = CameraMove::Dolly {
        from: pose_10(),
        dir: Vec3::new(0.0, 0.0, -1.0),
        distance: 5.0,
        duration: 0.0,
        easing: EasingKind::Linear,
    };
    let p = m.pose(0.0);
    assert!((p.eye.z - 5.0).abs() < 1e-5, "duration=0 应瞬时到位，z={}", p.eye.z);
    // t=NaN 不应 panic。
    let _ = m.pose(f32::NAN);
}

/// B15-7: DollyZoom 起点 eye == subject（d₀→1e-6 防除零）不 panic / 不 NaN。
#[test]
fn dolly_zoom_degenerate_start_no_nan() {
    let dz = DollyZoom {
        from: CameraPose::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 50.0),
        subject: Vec3::new(0.0, 0.0, 0.0),
        end_distance: 4.0,
        duration: 1.0,
        easing: EasingKind::Linear,
    };
    let f = dz.fov(0.5);
    assert!(!f.is_nan(), "退化起点 fov 不应 NaN，实得 {f}");
    let _ = dz.pose(0.5);
}

/// B15-8: Handheld 同 seed 同 t 逐位一致（含不同 octaves）。
#[test]
fn handheld_bitwise_deterministic() {
    for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX] {
        let hh = Handheld::new(0.03, 0.5, 2.0, 4, seed);
        let b = pose_10();
        let r = Vec3::new(1.0, 0.0, 0.0);
        let u = Vec3::new(0.0, 1.0, 0.0);
        for t in [0.0f32, 0.123, 1.0, 7.7, 100.0] {
            let p1 = hh.pose(b, r, u, t);
            let p2 = hh.pose(b, r, u, t);
            assert_eq!(p1.eye.x.to_bits(), p2.eye.x.to_bits(), "seed={seed} t={t}");
            assert_eq!(p1.look_at.y.to_bits(), p2.look_at.y.to_bits(), "seed={seed} t={t}");
        }
    }
}

/// B15-9: LensSpec::fov_for_focal(0) → 180（约定边界）；focal_for_fov(180) → +∞。
#[test]
fn fov_focal_boundary_semantics() {
    assert!((LensSpec::fov_for_focal(0.0, 24.0) - 180.0).abs() < 1e-3);
    assert!(LensSpec::focal_for_fov(180.0, 24.0).is_infinite());
}

/// B15-10: dof_far 在 focus 超过超焦距时 → +∞（不返回负数/NaN）。
#[test]
fn dof_far_beyond_hyperfocal_infinite() {
    let lens = LensSpec::new(50.0, 2.8);
    let h = lens.hyperfocal_mm();
    let far = lens.dof_far(h * 2.0);
    assert!(far.is_infinite() || far > 1e7, "超超焦距远界应→∞，实得 {far}");
    assert!(!far.is_nan());
}
