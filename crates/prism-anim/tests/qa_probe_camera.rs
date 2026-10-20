//! 独立复核探针——prism-anim atan / dolly zoom。
use prism_anim::camera_presets::{CameraPose, DollyZoom, LensSpec};
use prism_math::Vec3;

/// atan：全域 [0,∞) f32 vs f64 参考；边界 x=0/1/0.414；极大 x。
#[test]
fn probe_p0_2_atan_full_domain() {
    let h = 24.288f32;
    let ref_fov = |focal: f32| -> f64 {
        (0.5 * h as f64 / focal as f64).atan() * 2.0 * 180.0 / std::f64::consts::PI
    };
    let focals = [1e-3f32, 0.01, 1.0, 4.0, 10.0, 24.288, 35.0, 50.0, 85.0, 200.0, 1000.0, 1e6];
    let mut max_err = 0.0f64;
    for &f in &focals {
        let got = LensSpec::fov_for_focal(f, h) as f64;
        let want = ref_fov(f);
        let e = (got - want).abs();
        max_err = max_err.max(e);
        assert!(e < 0.05, "atan fov 误差过大 focal={f}: got={got} want={want} err={e}");
    }
    // x=1（focal = h/2）-> atan=π/4 -> fov=90。
    let fov90 = LensSpec::fov_for_focal(h * 0.5, h);
    assert!((fov90 - 90.0).abs() < 0.1, "x=1 应 90deg，got {fov90}");
    // x=0.4142 (分支边界) -> focal = h/2/0.4142。
    let fb = (h * 0.5) / 0.41421356;
    let got_b = LensSpec::fov_for_focal(fb, h) as f64;
    let want_b = (0.41421356f64).atan() * 2.0 * 180.0 / std::f64::consts::PI;
    assert!((got_b - want_b).abs() < 0.05, "分支边界 x=0.414 err={}", (got_b - want_b).abs());
    eprintln!("[probe] atan max_err = {max_err}");

    // roundtrip fov -> focal -> fov。
    for &fov in &[20.0f32, 35.0, 50.0, 75.0, 90.0, 110.0, 140.0] {
        let focal = LensSpec::focal_for_fov(fov, h);
        let back = LensSpec::fov_for_focal(focal, h);
        assert!((back - fov).abs() < 0.1, "roundtrip fov={fov} back={back}");
    }
    // 极大 x 不 NaN。
    assert!(LensSpec::fov_for_focal(1e-6, h).is_finite());
}

/// 关键不变量：d·tan(f/2) 沿 t 恒定（独立计算，非复用 invariant 内部）。
#[test]
fn probe_dolly_zoom_scale_invariant() {
    for &(end_fov, end_d, start_d) in &[(20.0f32, 4.0f32, 20.0f32), (110.0, 20.0, 10.0), (140.0, 30.0, 8.0)] {
        let dz = DollyZoom {
            from: CameraPose::new(Vec3::new(0.0, 0.0, start_d), Vec3::new(0.0, 0.0, 0.0), 50.0),
            subject: Vec3::new(0.0, 0.0, 0.0),
            end_distance: end_d,
            duration: 1.0,
            easing: prism_anim::EasingKind::Linear,
        };
        let inv0 = dz.invariant(0.0);
        let mut max_dev = 0.0f32;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let inv = dz.invariant(t);
            let dev = (inv - inv0).abs() / inv0.abs().max(1e-9);
            max_dev = max_dev.max(dev);
        }
        assert!(max_dev < 0.02, "尺度守恒破坏 end_fov={end_fov} max_dev={max_dev}");
        // fov(0) 应逐位回到起始。
        assert!((dz.fov(0.0) - 50.0).abs() < 1e-4, "fov(0) 应 50，got {}", dz.fov(0.0));
    }
}
