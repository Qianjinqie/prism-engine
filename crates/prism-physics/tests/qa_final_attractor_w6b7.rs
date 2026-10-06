//! QA 终验独立探针（W6-B7 Attractor 溢出收口复核）。
//!
//! 作者：严过关（QA）。本文件为**独立复核**，不复用既有断言：
//! 独立构造 Attractor 攻击向量，验证 coupling.rs `accel` 的
//! 修复（r² 夹下限 + q 饱和钳制 ±f32::MAX）在极端输入下的输出有界性。
//!
//! 攻击向量：
//! 1. strength=1e30, eps=0, 恰在中心 (d=0)。
//! 2. strength=1e30, eps=0, 偏离 1e-20（r² 极小）。
//! 3. strength=-1e30, eps=0 负向大值（排斥）。
//! 4. 新攻击点：从远处 d 很小 + q 钳到 f32::MAX → |a|=f32::MAX*|d|，
//!    记录该设计边界（是否天文数字加速度）。
//! 5. eps 本身为 NaN / +Inf 时的传播行为（记录为设计边界，不断言可通过）。

use prism_math::Vec3;
use prism_physics::coupling::{Attractor, ForceField};

fn all_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn max_abs(v: Vec3) -> f32 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

// ① strength=1e30, eps=0, 粒子恰在中心 → 三通道全有限。
#[test]
fn final_probe_huge_strength_zero_eps_at_center_finite() {
    let a = Attractor {
        center: Vec3::splat(0.0),
        strength: 1e30,
        eps: 0.0,
    };
    let r = a.accel(Vec3::splat(0.0), 0.0);
    assert!(all_finite(r), "中心处应全有限，实得 {r:?}");
    // d=0 → d*q=0（q 有限，即便巨大）。
    assert_eq!(r, Vec3::splat(0.0), "中心处 d=0 应得零向量，实得 {r:?}");
}

// ② strength=1e30, eps=0, 偏离 1e-20 → 三通道全有限。
#[test]
fn final_probe_huge_strength_zero_eps_tiny_offset_finite() {
    let a = Attractor {
        center: Vec3::splat(0.0),
        strength: 1e30,
        eps: 0.0,
    };
    let pos = Vec3::new(1e-20, 0.0, 0.0);
    let r = a.accel(pos, 0.0);
    assert!(all_finite(r), "微小偏移处应全有限，实得 {r:?}");
    // d 方向 = center - pos = -1e-20 x̂，故 x 分量应为负（吸引朝向中心）。
    assert!(r.x <= 0.0, "吸引方向应为负 x，实得 {r:?}");
}

// ③ strength=-1e30（负向大值/排斥），eps=0，偏移 → 全有限。
#[test]
fn final_probe_negative_huge_strength_finite() {
    let a = Attractor {
        center: Vec3::splat(0.0),
        strength: -1e30,
        eps: 0.0,
    };
    let r_center = a.accel(Vec3::splat(0.0), 0.0);
    let r_off = a.accel(Vec3::new(1e-20, 0.0, 0.0), 0.0);
    assert!(all_finite(r_center), "负向中心应有限，实得 {r_center:?}");
    assert!(all_finite(r_off), "负向偏移应有限，实得 {r_off:?}");
    // 排斥：远离中心 → 正 x 方向。
    assert!(r_off.x >= 0.0, "排斥方向应为正 x，实得 {r_off:?}");
}

// ④ 新攻击点（设计边界记录）：q 钳到 f32::MAX 后乘小 |d|。
//    验证"有限但巨大"仍是有限——即不违约 NaN/Inf；同时记录量级。
#[test]
fn final_probe_saturated_q_times_small_d_is_finite_but_huge() {
    let a = Attractor {
        center: Vec3::splat(0.0),
        strength: 1e30,
        eps: 0.0,
    };
    // d=1e-30 → r²=(1e-30)²=1e-60，夹到 MIN_POSITIVE(≈1.18e-38)
    // q=1e30/1.18e-38 ≈ 8.5e67 → 溢出 → 钳到 f32::MAX(≈3.4e38)
    // |a| = f32::MAX * 1e-30 ≈ 3.4e8。有限但巨大。
    let d_small = 1e-30_f32;
    let r = a.accel(Vec3::new(d_small, 0.0, 0.0), 0.0);
    assert!(all_finite(r), "钳制后应全有限，实得 {r:?}");
    let mag = max_abs(r);
    // 头号断言（契约）：必须有限——这是修复目标，硬断言。
    // 量级观测（非契约，仅记录设计边界）：远大于日常加速度 ~1e10。
    println!(
        "[设计边界] q 钳制后 |a| = {mag:e}（有限但巨大；普通物理加速度 ~1e1，\
         工程上限通常 ~1e6）。d={d_small:e}"
    );
    // 只断言有限性 + 小于 f32::MAX（不会溢出再翻 Inf/NaN）。
    assert!(mag <= f32::MAX, "不应超过 f32::MAX，实得 {mag:e}");
}

// ⑤ eps 为 NaN / +Inf 的传播（设计边界记录）。
//    目标：记录行为，明确是否会被下游静默接受。
#[test]
fn final_probe_eps_nan_inf_behavior_recorded() {
    let a_nan = Attractor {
        center: Vec3::splat(0.0),
        strength: 1.0,
        eps: f32::NAN,
    };
    let r_nan = a_nan.accel(Vec3::new(1.0, 0.0, 0.0), 0.0);
    println!("[设计边界] eps=NaN → accel={r_nan:?}，有限={}", all_finite(r_nan));

    let a_inf = Attractor {
        center: Vec3::splat(0.0),
        strength: 1.0,
        eps: f32::INFINITY,
    };
    let r_inf = a_inf.accel(Vec3::new(1.0, 0.0, 0.0), 0.0);
    println!(
        "[设计边界] eps=+Inf → accel={r_inf:?}，有限={}",
        all_finite(r_inf)
    );

    let a_nan_s = Attractor {
        center: Vec3::splat(0.0),
        strength: f32::NAN,
        eps: 1.0,
    };
    let r_nan_s = a_nan_s.accel(Vec3::new(1.0, 0.0, 0.0), 0.0);
    println!(
        "[设计边界] strength=NaN → accel={r_nan_s:?}，有限={}",
        all_finite(r_nan_s)
    );

    // 观测结论（修正后）：
    //  a) eps=NaN → d²+NaN=NaN → `NaN.max(MIN_POSITIVE)` 比较为 false 取另一支
    //     → r²=MIN_POSITIVE（有限）→ q=1/MIN_POSITIVE 溢出→钳到 f32::MAX → 有限。
    //     即 r² 夹下限**意外**把 eps=NaN 也收成了有限输出（即使量级巨大）。
    //  b) eps=+Inf → d²+Inf=Inf → r²=Inf → q=strength/Inf=0 → 输出零向量，有限。
    //  c) strength=NaN → q=(NaN).clamp(...)=NaN → d*NaN=NaN → 输出 NaN（**未设防**）。
    // 断言：合法有限 strength/eps 下溢出已收口（①②③④硬断言通过）。
    //        非法规格输入（缺省无防御）记录如下，不 asserting 为"正确"，仅固化观测。
    assert!(all_finite(r_nan), "eps=NaN 因 r² 夹下限被意外收为有限，实得 {r_nan:?}");
    assert!(all_finite(r_inf), "eps=+Inf 应有限（→0），实得 {r_inf:?}");
    // strength=NaN：clamp 不做 NaN 防御，NaN 透传——记录为遗留设计边界。
    assert!(
        r_nan_s.x.is_nan(),
        "strength=NaN 无防御应透传 NaN，实得 {r_nan_s:?}"
    );
}
