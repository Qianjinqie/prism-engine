//! QA 边界反例攻击（验收清单 C-①②③；prism-anim 侧）。
//!
//! 与既有单元测试的分工：单元测试覆盖常规路径，本文件专攻**临界参数与
//! 恶意输入**——分支判定半径、退化参数、乱序/重复关键帧、深度嵌套、错误
//! 路径与 IEEE 语义边界。
//!
//! QA Round 1 修复记录：QA-01（Expr 深嵌套栈溢出 → 解析器深度上限 256，
//! 见 expr.rs 单元测试）与 QA-02（Track NaN 时间 panic → NaN 值语义传播）
//! 均已修复，无 `#[ignore]` 占位。

use prism_anim::easing::EasingKind;
use prism_anim::expr::{ExprError, ExprVar};
use prism_anim::{Expr, ExprEnv, Keyframe, LoopMode, Spring, Track};

// ---------------------------------------------------------------------------
// C-① Spring：三分支临界参数
// ---------------------------------------------------------------------------

/// QA 测试内的独立 RK4 参考（f64 步进；与 spring.rs 测试内实现**分别书写**，
/// 互为对照来源，状态方程同：x'=v, v'=ω²(T−x)−2ζωv）。
fn rk4_reference(spring: &Spring, t_end: f32, dt: f64) -> f64 {
    let (omega, zeta) = (spring.omega as f64, spring.zeta as f64);
    let target = spring.target as f64;
    let deriv = |x: f64, v: f64| (v, omega * omega * (target - x) - 2.0 * zeta * omega * v);
    let mut x = spring.x0 as f64;
    let mut v = spring.v0 as f64;
    let steps = (t_end as f64 / dt).round() as usize;
    let h = t_end as f64 / steps as f64;
    for _ in 0..steps {
        let (k1x, k1v) = deriv(x, v);
        let (k2x, k2v) = deriv(x + h / 2.0 * k1x, v + h / 2.0 * k1v);
        let (k3x, k3v) = deriv(x + h / 2.0 * k2x, v + h / 2.0 * k2v);
        let (k4x, k4v) = deriv(x + h * k3x, v + h * k3v);
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    x
}

/// C-①：ζ→1 邻域（分支判定半径 CRITICAL_EPS=1e-3 两侧各放三点）逐一对照
/// RK4，验收线 < 1e-3（Phase 1 关卡 1）。ζ=0.999/1.001 恰在
/// 闭区间内走临界分支，0.9989/1.0011 刚好跨出——覆盖切换边界两侧。
#[test]
fn qa_spring_zeta_near_one_matches_rk4() {
    let zetas: [(&str, f32); 7] = [
        ("欠阻尼侧 0.9989", 0.9989),
        ("临界闭边界 0.999", 0.999),
        ("临界内部 0.9995", 0.9995),
        ("临界精确 1.0", 1.0),
        ("临界内部 1.0005", 1.0005),
        ("临界闭边界 1.001", 1.001),
        ("过阻尼侧 1.0011", 1.0011),
    ];
    for (name, zeta) in zetas {
        let spring = Spring::new(6.0, zeta, 0.0, 0.0, 1.0);
        let mut max_err = 0.0_f64;
        let mut i = 1;
        while i <= 30 {
            let t = i as f32 * 0.05;
            let err = (spring.sample(t) as f64 - rk4_reference(&spring, t, 1.0 / 1000.0)).abs();
            max_err = max_err.max(err);
            i += 1;
        }
        assert!(
            max_err < 1e-3,
            "ζ={zeta}（{name}）：解析解 vs RK4 最大误差 {max_err:.3e}（验收线 1e-3）"
        );
    }
}

/// C-①：ω→0 的物理行为——ω=0 静态取 target（文档化退化）；ω 极小时弹簧
/// 近似静止（初速 0 的低频弹簧位移是一阶小量，短时内应接近 x0）。
#[test]
fn qa_spring_zero_and_tiny_omega_physical() {
    // ω=0：文档化退化 → target。
    assert_eq!(Spring::new(0.0, 0.35, 0.0, 0.0, 7.5).sample(1.0), 7.5);
    // ω=1e-4：1s 与 10s 内位移均应可忽略（ω²t²/2 量级 ≤ 5e-7）。
    let tiny = Spring::new(1e-4, 0.35, 0.0, 0.0, 1.0);
    assert!(tiny.sample(1.0).abs() < 1e-6, "ω=1e-4 时 1s 位移应近 0，实际 {}", tiny.sample(1.0));
    assert!(
        tiny.sample(10.0).abs() < 1e-5,
        "ω=1e-4 时 10s 位移应近 0，实际 {}",
        tiny.sample(10.0)
    );
    // 长时间后仍收敛（ωt 累积到 O(1) 需 t ≈ 1e4 s；此处只验证不发散/不 NaN）。
    let long = tiny.sample(2.0e4);
    assert!(long.is_finite(), "ω→0 长时采样应有限，实际 {long}");
    assert!((long - 1.0).abs() < 0.5, "长时后应显著向 target 靠近，实际 {long}");
}

/// NaN 传播冒烟：det::exp/sin/cos 有显式 NaN 守卫，Spring 采样 NaN 时间应
/// 返回 NaN 而非 panic/挂起（IEEE 语义；非 panic 即通过，不约束具体值）。
#[test]
fn qa_spring_nan_time_propagates_without_panic() {
    for zeta in [0.35_f32, 1.0, 1.8] {
        let v = Spring::new(6.0, zeta, 0.0, 0.0, 1.0).sample(f32::NAN);
        assert!(v.is_nan() || v.is_finite(), "ζ={zeta}：NaN 时间不得 panic/挂起，实际 {v}");
    }
}

// ---------------------------------------------------------------------------
// C-② Track：同刻多帧 / 时间乱序
// ---------------------------------------------------------------------------

/// C-②：乱序构造（5 帧乱序输入 + 2 对同刻）——构造稳定排序后，采样在
/// 关键帧点精确命中、同刻"后者生效"、段间按左帧缓动插值、首前末后钳制。
#[test]
fn qa_track_unsorted_with_equal_times() {
    let tr = Track::new(vec![
        Keyframe::new(3.0, 30.0, EasingKind::Linear),   // 乱序①
        Keyframe::new(1.0, 5.0, EasingKind::Linear),    // 同刻对 a1
        Keyframe::new(1.0, 6.0, EasingKind::Linear),    // 同刻对 a2（后者生效）
        Keyframe::new(0.0, 0.0, EasingKind::Linear),    // 乱序②
        Keyframe::new(2.0, 20.0, EasingKind::Linear),   // 乱序③
    ]);
    // 排序后：0→0, 1→5, 1→6, 2→20, 3→30。
    assert_eq!(tr.sample(-1.0), 0.0, "首前钳首帧");
    assert_eq!(tr.sample(0.0), 0.0);
    assert!((tr.sample(0.5) - 2.5).abs() < 1e-6, "[0,1] 段取首帧值 0");
    assert_eq!(tr.sample(1.0), 6.0, "同刻帧后者生效");
    assert!((tr.sample(1.5) - 13.0).abs() < 1e-6, "[1,2] 段从同刻后者 6 插值到 20");
    assert_eq!(tr.sample(2.0), 20.0);
    assert!((tr.sample(2.5) - 25.0).abs() < 1e-6);
    assert_eq!(tr.sample(3.0), 30.0);
    assert_eq!(tr.sample(9.9), 30.0, "末后钳末帧");
}

/// C-②：±inf 时间钳制冒烟（t=+inf 取末帧、t=-inf 取首帧，不 panic）。
#[test]
fn qa_track_infinite_time_clamps() {
    let tr = Track::new(vec![
        Keyframe::new(0.0, 1.0, EasingKind::Linear),
        Keyframe::new(1.0, 2.0, EasingKind::Linear),
    ]);
    assert_eq!(tr.sample(f32::NEG_INFINITY), 1.0, "-inf 钳首帧");
    assert_eq!(tr.sample(f32::INFINITY), 2.0, "+inf 钳末帧");
}

/// QA-02（已修复）：`Track::sample(f32::NAN)` 返回 NaN（值语义传播），
/// **不 panic**——NaN 不满足任何 IEEE 比较、不落入任何段（见 track.rs
/// `sample` 文档注释），此前在 track.rs:73 因 `partition_point` 谓词恒
/// false 导致 `0 − 1` usize 下溢。
#[test]
fn qa_track_nan_time_does_not_panic() {
    let tr = Track::new(vec![
        Keyframe::new(0.0, 1.0, EasingKind::Linear),
        Keyframe::new(1.0, 2.0, EasingKind::Linear),
    ]);
    let v = tr.sample(f32::NAN);
    assert!(v.is_nan(), "NaN 时间应返回 NaN（值语义传播），实际 {v}");
}

// ---------------------------------------------------------------------------
// C-③ Expr：深嵌套 / IEEE 除零 / 错误路径
// ---------------------------------------------------------------------------

/// C-③：200 层括号嵌套解析求值正确（深度健壮性下界；解析器深度上限
/// 256 层——QA-01 修复后 1000 层恶意嵌套返回 `ExprError::DepthLimit`
/// 而非栈溢出，见 expr.rs 单元测试 `deep_nesting_1000_levels_reports_depth_limit`）。
#[test]
fn qa_expr_deep_nesting_200_levels_ok() {
    let depth = 200;
    let src = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
    let e = Expr::parse(&src).expect("200 层括号嵌套应可解析");
    assert_eq!(e.sample(0.0), 1.0);
}

/// C-③：IEEE 除零语义——除零得 ±∞、0/0 得 NaN（模块文档"不做静默兜底"）。
#[test]
fn qa_expr_division_ieee_semantics() {
    let env = ExprEnv { t: 0.0, p: 0.0, w: 0.0, h: 0.0 };
    assert_eq!(Expr::parse("1/0").unwrap().eval(&env), f32::INFINITY, "1/0 = +inf");
    assert_eq!(Expr::parse("-1/0").unwrap().eval(&env), f32::NEG_INFINITY, "-1/0 = -inf");
    assert!(Expr::parse("0/0").unwrap().eval(&env).is_nan(), "0/0 = NaN");
    assert_eq!(
        Expr::parse("1/(0.5-0.5)").unwrap().eval(&env),
        f32::INFINITY,
        "表达式求值出的零分母同样 +inf"
    );
}

/// C-③：错误路径逐类命中（C19：错误 Display 必含"建议"段）。
#[test]
fn qa_expr_error_paths_with_advice() {
    // 未知变量。
    let err = Expr::parse("salary + 1").unwrap_err();
    assert_eq!(err, ExprError::UnknownVar { pos: 0, name: "salary".to_string() });
    assert!(err.to_string().contains("建议"), "C19：未知变量错误必含建议段");

    // 文法外字符。
    let err = Expr::parse("2 & 3").unwrap_err();
    assert!(matches!(err, ExprError::UnexpectedChar { pos: 2, ch: '&' }));
    assert!(err.to_string().contains("建议"));

    // 多余字符（文法内字符接不上文法）。
    let err = Expr::parse("2 3").unwrap_err();
    assert!(matches!(err, ExprError::TrailingChars { pos: 2 }));
    assert!(err.to_string().contains("建议"));

    // 空表达式 / 意外结束。
    assert_eq!(Expr::parse("   ").unwrap_err(), ExprError::Empty);
    let err = Expr::parse("1 +").unwrap_err();
    assert_eq!(err, ExprError::UnexpectedEnd);
    assert!(err.to_string().contains("建议"), "C19：意外结束错误必含建议段");

    // 缺右括号同样 UnexpectedEnd。
    assert_eq!(Expr::parse("(1 + 2").unwrap_err(), ExprError::UnexpectedEnd);
}

/// C-③：变量全集 t/p/w/h 解析与求值（ExprVar 名字面量单一出口）。
#[test]
fn qa_expr_variables_and_purity() {
    let e = Expr::parse("t * 2 + p + w + h").unwrap();
    let env = ExprEnv { t: 1.5, p: 0.25, w: 4.0, h: 8.0 };
    assert!((e.eval(&env) - 15.25).abs() < 1e-6);
    // 纯函数：同 env 重算逐位一致。
    let bits = e.eval(&env).to_bits();
    for _ in 0..1000 {
        assert_eq!(e.eval(&env).to_bits(), bits, "同 env 同值逐位");
    }
    assert_eq!(ExprVar::T.name(), "t");
    assert_eq!(ExprVar::H.name(), "h");
}

// ---------------------------------------------------------------------------
// C-② 补充：LoopMode 与 Track 组合的退化输入（duration ≤ 0 恒等映射）
// ---------------------------------------------------------------------------

#[test]
fn qa_loop_mode_zero_duration_identity_over_track() {
    let tr = Track::new(vec![
        Keyframe::new(0.0, 0.0, EasingKind::Linear),
        Keyframe::new(1.0, 10.0, EasingKind::Linear),
    ]);
    for mode in [LoopMode::Direct, LoopMode::Loop, LoopMode::PingPong] {
        // duration=0 → 恒等映射 → 采样按原始 t 直接进 Track（钳制生效）。
        for (t, want) in [(-1.0_f32, 0.0), (0.5, 5.0), (2.0, 10.0)] {
            let mapped = mode.map(t, 0.0);
            assert!((tr.sample(mapped) - want).abs() < 1e-6, "{mode:?} duration=0 t={t}");
        }
    }
}
