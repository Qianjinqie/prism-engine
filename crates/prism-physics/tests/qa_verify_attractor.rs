//! QA 独立复核——Attractor 边界修复终验。
//!
//! 不信"绿了"：本文件独立构造探针，直击声称修复的边界，
//! 每条断言注明"失败代表什么缺陷"。
//!
//! 不修改任何源码，仅新增测试。

use prism_math::Vec3;
use prism_physics::coupling::{
    Attractor, CoupledSession, Coupler, ForceField, Gravity, Wind, MAX_ACCEL,
};
use prism_physics::session::{ParticleInit, SimSpec};

// ---------------------------------------------------------------------------
// P1. MAX_ACCEL 是否真的在「输出」处生效且三分量独立 clamp
// ---------------------------------------------------------------------------

/// 探针 1a：三分量**独立**逐项 clamp——构造只在单轴超限的极端构型，
/// 若实现只 clamp 某一路（或按模长归一）会漏。失败 = 限幅实现不完整。
#[test]
fn probe_max_accel_clamps_each_component_independently() {
    // 构造：三分量各取不同 |d|，其中两轴（±y/±z 大偏移）必然触发 clamp，
    // 一轴（极近 x）保有真实小值——验证各分量**独立**逐项 clamp，
    // 而非按模长归一（后者会把小分量也按比例缩放）。
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    let a = at.accel(Vec3::new(1e-30, 5.0, -7.0), 0.0);
    assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite(), "三分量应全有限:{a:?}");
    // y/z：q = 1e30/74 未饱和，d.y=-5 → a.y = -5*q ≈ -6.7e28 → 必被 clamp 到 -1000。
    assert!(
        (a.y.abs() - MAX_ACCEL).abs() <= MAX_ACCEL * 1e-6,
        "y 分量应恰好被 clamp 到 ±MAX_ACCEL:{a:?}"
    );
    assert!(
        (a.z.abs() - MAX_ACCEL).abs() <= MAX_ACCEL * 1e-6,
        "z 分量应恰好被 clamp 到 ±MAX_ACCEL:{a:?}"
    );
    // x：d.x=-1e-30，解析 a.x = -1e-30 * q，q=1e30/74≈1.35e28 → a.x≈-0.0135。
    // 该值**等于未限幅解析值**（未被 y/z 超限牵连缩放）→ 证明逐项独立 clamp。
    // 若实现改为按模长归一，a.x 会被放大/改写，此处逐位对比即暴露。
    let d = Vec3::new(1e-30, 5.0, -7.0);
    let r2 = d.dot(&d);
    let q = 1e30f32 / r2;
    let expect_x = -1e-30f32 * q;
    assert_eq!(
        a.x.to_bits(),
        expect_x.to_bits(),
        "小分量应逐位等于其独立解析值（不被其它轴超限牵连）:{a:?}"
    );
}

/// 探针 1b：符号保持——clamp 不得翻转方向（吸引应仍指向中心）。
/// 失败 = 限幅破坏了力的方向语义（物理错误）。
#[test]
fn probe_max_accel_preserves_direction_sign() {
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    for off in [1e-30f32, 1e-8, 1e-3, 1.0] {
        let pos = Vec3::new(off, 0.0, 0.0);
        let a = at.accel(pos, 0.0);
        assert!(a.x <= 0.0, "远处正向偏移应被吸向 -x（方向不得翻转）:off={off} a={a:?}");
    }
    let pos_neg = Vec3::new(-1e-8, 0.0, 0.0);
    assert!(at.accel(pos_neg, 0.0).x >= 0.0, "负向偏移应被吸向 +x");
}

// ---------------------------------------------------------------------------
// P2. 正常量级「逐位不变」——直接对拍解析式
// ---------------------------------------------------------------------------

/// 探针 2：正常量级下，accel 输出必须**逐位等于**未限幅的解析式 d*(strength/r2)。
/// 失败 = 限幅在正常量级干扰了数值（Mini Code 违约，会改变既有内容结果）。
#[test]
fn probe_normal_magnitude_bitwise_equals_unclamped_analytic() {
    let cases = [
        (Vec3::new(1.0, 0.0, 0.0), 50.0f32, 0.1f32),
        (Vec3::new(0.0, 2.0, 0.0), 9.81, 0.0),
        (Vec3::new(0.3, -0.5, 0.2), 123.4, 0.05),
        (Vec3::new(-1.2, 0.8, 0.0), -30.0, 0.2),
    ];
    for (center, strength, eps) in cases {
        let at = Attractor { center, strength, eps };
        for k in 0..30 {
            let pos = Vec3::new(k as f32 * 0.21, k as f32 * -0.13, k as f32 * 0.07);
            let got = at.accel(pos, 0.0);
            // 重算未限幅解析式（与源码同路径，但跳过 clamp）。
            let d = center - pos;
            let r2 = (d.dot(&d) + eps * eps).max(f32::MIN_POSITIVE);
            let q = (strength / r2).clamp(-f32::MAX, f32::MAX);
            let expect = d * q;
            assert_eq!(
                [got.x.to_bits(), got.y.to_bits(), got.z.to_bits()],
                [expect.x.to_bits(), expect.y.to_bits(), expect.z.to_bits()],
                "正常量级应逐位等于未限幅解析式:center={center:?} pos={pos:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// P3. 「禁静默失败」攻击：限幅是否把本应报错的非法构型静默钳成合法值？
// ---------------------------------------------------------------------------

/// 探针 3【核心攻击】：极端 strength（如 1e30）在 eps=0、微小偏移处产生的
/// 真实物理加速度是 ~3.4e18（远离正常），这是"规格病态"而非"强风/爆炸"。
/// 限幅把它静默钳到 1000，**既不报错也不告警**——评估是否违反禁静默失败。
///
/// 本探针锁定事实：accel() 对病态 strength 不报错，只返回被钳值。
/// 记录的期望：若团队判定"病态 strength 应结构化拒绝"，则此测试失败代表
/// 「accel 静默掩盖病态输入」缺陷——路由增加 accel 级守卫。
#[test]
fn probe_clamp_silently_masks_pathological_strength() {
    // strength=1e30 远超任何"强风/爆炸"（1000 = 102·g₀）。此处 |d|=1e-8，
    // 真实 a ≈ f32::MAX*1e-8 ≈ 3.4e30 → 被静默钳到 1000。
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    let a = at.accel(Vec3::new(1e-8, 0.0, 0.0), 0.0);
    // 事实陈述：accel 不 panic 不报错，返回恰好 = -MAX_ACCEL（静默钳制）。
    assert!(
        (a.x.abs() - MAX_ACCEL).abs() <= MAX_ACCEL * 1e-6,
        "记录：病态 strength 被静默钳到 MAX_ACCEL（无任何告警）:{a:?}"
    );
    // 关键对比：validate() 对同一 strength=1e30 **放行**（有限即合法）。
    assert!(
        Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 }
            .validate()
            .is_ok(),
        "validate 对 1e30 放行——证明限幅承担了『掩盖病态量级』而非仅防崩坏"
    );
}

// ---------------------------------------------------------------------------
// P4. validate() 被绕过（直接构造不调用）时的行为
// ---------------------------------------------------------------------------

/// 探针 4a【绕过防线事实记录】：绕过 validate，NaN strength 直接调 accel。
///
/// 事实：accel 返回 NaN——因为 `NaN.clamp(-1000,1000) == NaN`（IEEE754，
/// clamp 对 NaN 恒等）。**这是设计预期**：accel 契约是"validate 通过后输出
/// 恒有限"，accel 自身非防御点。本探针锁定该边界，并验证下游集成守卫
/// （CoupledSession::new 的有限检查）**确实兜住** NaN 透传。
///
/// 判定：accel 不拦 NaN 属已知边界而非缺陷（设计文档明示 clamp 不拦 NaN，
/// 故设 validate + 集成守卫）。若探针 4c 显示集成守卫也漏，才构成真缺陷。
#[test]
fn probe_bypass_validate_nan_recorded_and_caught_downstream() {
    // accel 层：NaN 透传（记录事实，非缺陷——设计已知边界）。
    let a_nan = Attractor { center: Vec3::new(1.0, 0.0, 0.0), strength: f32::NAN, eps: 0.1 }
        .accel(Vec3::splat(0.0), 0.0);
    assert!(
        !a_nan.x.is_finite(),
        "记录：accel 对绕过 validate 的 NaN strength 返回非有限（设计边界，非缺陷）:{a_nan:?}"
    );
    // 下游集成守卫：NaN 力场进入 CoupledSession → 结构化 P3008 拒绝。
    let spec = SimSpec {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        particles: vec![ParticleInit {
            position: Vec3::new(1.0, 0.0, 0.0),
            velocity: Vec3::splat(0.0),
            inv_mass: 1.0,
        }],
        constraints: vec![],
    };
    let mut c = Coupler::new();
    c.add(Box::new(Attractor { center: Vec3::new(1.0, 0.0, 0.0), strength: f32::NAN, eps: 0.1 }));
    let r = CoupledSession::new(spec, &c, 1.0 / 60.0, 4, 8);
    assert!(r.is_err(), "下游集成守卫必须兜住 accel 透传的 NaN");
    assert_eq!(r.unwrap_err().code, "P3008");
}

/// 探针 4b：Inf strength 绕过 validate → accel 行为。
#[test]
fn probe_bypass_validate_inf_strength_behavior() {
    let a_inf = Attractor { center: Vec3::new(1.0, 0.0, 0.0), strength: f32::INFINITY, eps: 0.1 }
        .accel(Vec3::splat(0.0), 0.0);
    // 记录事实（无论有限与否），作为"绕过防线后的实际暴露面"证据。
    let _ = (a_inf.x.is_finite(), a_inf.y.is_finite(), a_inf.z.is_finite());
    assert!(
        a_inf.x.is_finite() && a_inf.y.is_finite() && a_inf.z.is_finite(),
        "绕过 validate 的 +Inf strength 应为有限（q 饱和 + clamp 应兜底）:{a_inf:?}"
    );
}

// ---------------------------------------------------------------------------
// P5. 限幅后连续大加速度下模拟稳定性（不下游崩坏）
// ---------------------------------------------------------------------------

/// 探针 5：极端吸引子驱动 CoupledSession 多帧推进，粒子位置必须恒有限。
/// 失败 = 限幅未阻断下游数值崩坏（声称的稳定收益不成立）。
#[test]
fn probe_simulation_stable_under_clamped_extreme_field() {
    let spec = SimSpec {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        particles: vec![ParticleInit {
            position: Vec3::new(1e-6, 0.0, 0.0),
            velocity: Vec3::splat(0.0),
            inv_mass: 1.0,
        }],
        constraints: vec![],
    };
    // 病态 strength，但有限——validate 放行；accel 被 clamp。
    let mut c = Coupler::new();
    c.add(Box::new(Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 }));
    let mut s = CoupledSession::new(spec, &c, 1.0 / 60.0, 4, 8).unwrap();
    let mut store = prism_core::node::SnapshotStore::new();
    for f in 0..=120u64 {
        s.advance_to([0x42; 32], f, &mut store).unwrap();
        let p = s.session().world().particles()[0].position;
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "帧 {f} 位置应恒有限（限幅应阻断崩坏）:{p:?}"
        );
    }
}

/// 探针 6：等效重力确实受 MAX_ACCEL 约束（集成路径端到端）。
/// 失败 = 集成层未继承子系统的限幅收益。
#[test]
fn probe_effective_gravity_bounded_via_session() {
    let spec = SimSpec {
        gravity: Vec3::splat(0.0),
        particles: vec![ParticleInit {
            position: Vec3::new(1e-6, 0.0, 0.0),
            velocity: Vec3::splat(0.0),
            inv_mass: 1.0,
        }],
        constraints: vec![],
    };
    let mut c = Coupler::new();
    c.add(Box::new(Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 }));
    let s = CoupledSession::new(spec, &c, 1.0 / 60.0, 4, 8).unwrap();
    let g = s.effective_gravity();
    assert!(
        g.x.abs() <= MAX_ACCEL * (1.0 + 1e-6)
            && g.y.abs() <= MAX_ACCEL * (1.0 + 1e-6)
            && g.z.abs() <= MAX_ACCEL * (1.0 + 1e-6),
        "等效重力分量应受 MAX_ACCEL 约束:{g:?}"
    );
}

// ---------------------------------------------------------------------------
// P6. 合法正常力场未被限幅误伤（回归保底）
// ---------------------------------------------------------------------------

/// 探针 7：重力/风等远低于上限的场，经过 Coupler 不被改动。
/// 失败 = 限幅误伤正常内容（回归破坏）。
#[test]
fn probe_normal_fields_untouched() {
    let mut c = Coupler::new();
    c.add(Box::new(Gravity { g: Vec3::new(0.0, -9.81, 0.0) }));
    c.add(Box::new(Wind { dir: Vec3::new(1.0, 0.0, 0.0) }));
    let a = c.accel(Vec3::splat(0.0), 0.0);
    assert_eq!(a.x.to_bits(), 1.0f32.to_bits(), "风力 x 应不变");
    assert_eq!(a.y.to_bits(), (-9.81f32).to_bits(), "重力 y 应不变");
}
