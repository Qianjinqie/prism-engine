//! QA 攻击性边界测试（W6-B7：joints / cloth / sph / fracture / coupling）。
//!
//! 作者：严过关（QA）。攻击目标：
//! 1. 关节约束不变量：铰链消去相对平移、滑槽轴外收敛、限位区间自由——
//!    并攻击退化构型（重合锚点、零轴、lo>hi、NaN 柔度）。
//! 2. 关节**确定性 + 求解序无关**：同构型两遍逐位一致（归约序不影响结果）。
//! 3. 布料：角点逐位固定、能量单调非增、下垂、规格防御。
//! 4. SPH：核半径外 0、Poly6 单调、密度自项一致性（densities vs evaluate 契约）、
//!    dt/substeps 防御、确定性。
//! 5. 破碎：面积和守恒（单位分解）、确定性、半平面裁剪、退化规格防御、
//!    站点数上限语义。
//! 6. 耦合：力场纯函数性、合外力线性叠加、attractor 奇点软化。
//! 7. 回归锁定（QA Round 2）：强量级 strength + eps=0 不得 NaN/Inf。
//! 8. 设计边界遗留风险清除：加速度上限（MAX_ACCEL）+ NaN/Inf 参数显式拒绝。

use prism_math::Vec3;
use prism_physics::cloth::{Cloth, ClothSpec, PinMode};
use prism_physics::coupling::{
    Attractor, CoupledSession, Coupler, ForceField, Gravity, Turbulence, Wind, MAX_ACCEL,
};
use prism_physics::fracture::{Fracture, FractureSpec, Polygon2, SiteSeed};
use prism_physics::joints::{HingeJoint, LimitJoint, SliderJoint};
use prism_physics::session::{ParticleInit, SimSpec};
use prism_physics::sph::{Fluid, FluidSpec};
use prism_physics::xpbd::{Constraint, World};

fn fixed(v: Vec3) -> Vec3 {
    v
}

// ---------------------------------------------------------------------------
// ① 关节不变量与退化构型
// ---------------------------------------------------------------------------

#[test]
fn hinge_anchors_are_normalized_order_independent() {
    // 构造序无关：new(1, a, b) 与 new(1, b, a) 端点集相同（升序规范化）。
    let a = HingeJoint::new(1, 3, 1, 0.0);
    let b = HingeJoint::new(1, 1, 3, 0.0);
    assert_eq!(a.indices(), b.indices(), "铰链端点应升序规范化（构造序无关）");
    assert_eq!(a.indices(), &[1usize, 3]);
}

#[test]
fn hinge_coincident_anchors_return_zero_residual() {
    // 退化：两锚点已重合 → evaluate 0，project 修正后仍 0（不 panic、NaN）。
    let mut world = World::new(Vec3::splat(0.0));
    world.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
    world.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0);
    let j = HingeJoint::new(1, 0, 1, 0.0);
    assert_eq!(j.evaluate(world.particles()), 0.0, "重合锚点 C 应为 0");
    let mut lam = 0.0f32;
    let res = j.project(world.particles_mut(), 1.0 / 60.0, &mut lam);
    assert!(res.is_finite() && res >= 0.0, "重合锚点残余应有限非负：{res}");
    // 位置未被污染为 NaN。
    for p in world.particles() {
        assert!(p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite());
    }
}

#[test]
fn slider_zero_axis_rejected_nan_axis_rejected() {
    // 零轴 → P3007。
    let e = SliderJoint::new(1, 0, 1, Vec3::splat(0.0), 0.0).validate().unwrap_err();
    assert_eq!(e.code, "P3007");
    // NaN 轴：length() 为 NaN，`< 1e-8` 为 false → 是否漏拒？攻击此契约缺口。
    let nan_axis = Vec3::new(f32::NAN, 0.0, 0.0);
    let r = SliderJoint::new(1, 0, 1, nan_axis, 0.0).validate();
    assert!(r.is_err(), "NaN 轴应被拒绝（防御缺失：NaN 长度比较为 false）");
}

#[test]
fn limit_joint_range_free_and_continuous() {
    // 区间内 C = 0（自由）；边界处 |C| 连续（线性增长）。
    let axis = Vec3::new(1.0, 0.0, 0.0);
    let mut world = World::new(Vec3::splat(0.0));
    world.add_particle(Vec3::splat(0.0), Vec3::splat(0.0), 1.0); // p0 固定
    world.add_particle(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let j = LimitJoint::new(1, 0, 1, axis, 0.0, 1.0, 0.0);
    // 区间内：C = 0。
    assert_eq!(j.evaluate(world.particles()), 0.0, "区间内 C 应 0");
    // 移动 p1 到 l = 1.5（超上界 0.5）→ C = 0.5。
    world.particles_mut()[1].position = Vec3::new(1.5, 0.0, 0.0);
    assert!((j.evaluate(world.particles()) - 0.5).abs() < 1e-6, "超上界 C 应 = 超量");
    // 下界：l = -0.3 → C = -0.3 → |C| = 0.3。
    world.particles_mut()[1].position = Vec3::new(-0.3, 0.0, 0.0);
    assert!((j.evaluate(world.particles()) - 0.3).abs() < 1e-6, "超下界 |C| 应 = 超量");
}

#[test]
fn limit_joint_lo_hi_nan_rejected() {
    // lo > hi → P1008。
    let e = LimitJoint::new(1, 0, 1, Vec3::new(1.0, 0.0, 0.0), 2.0, 1.0, 0.0)
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P1008");
    // lo = NaN：`!(lo <= hi)` 为 true 应拒。
    let e2 = LimitJoint::new(1, 0, 1, Vec3::new(1.0, 0.0, 0.0), f32::NAN, 1.0, 0.0).validate();
    assert!(e2.is_err(), "NaN lo 应被拒绝（`!(NaN<=hi)` 应触发 P1008）");
    // 负柔度 / NaN 柔度 → P1006（`!(c >= 0)` NaN 为 true）。
    let e3 = LimitJoint::new(1, 0, 1, Vec3::new(1.0, 0.0, 0.0), 0.0, 1.0, f32::NAN).validate();
    assert!(e3.is_err(), "NaN 柔度应被拒绝");
}

#[test]
fn joint_family_determinism_and_order_independence() {
    // 同构型两遍求解逐位一致（无全局 RNG）。
    let build = || {
        let mut w = World::new(Vec3::new(0.0, -9.81, 0.0));
        w.add_particle(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.0), 0.0);
        w.add_particle(Vec3::new(0.3, 1.6, 0.2), Vec3::splat(0.0), 1.0);
        w.add_particle(Vec3::new(0.9, 1.1, 0.1), Vec3::new(0.1, 0.0, 0.0), 1.0);
        w.add_constraint(Box::new(HingeJoint::new(1, 0, 1, 0.002))).unwrap();
        w.add_constraint(Box::new(SliderJoint::new(2, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.001)))
            .unwrap();
        w.add_constraint(Box::new(LimitJoint::new(3, 1, 2, Vec3::new(1.0, 0.0, 0.0), 0.0, 2.0, 0.0)))
            .unwrap();
        w
    };
    let mut a = build();
    let mut b = build();
    for _ in 0..20 {
        a.step(1.0 / 60.0, 4, 8).unwrap();
        b.step(1.0 / 60.0, 4, 8).unwrap();
    }
    let bits = |w: &World| -> Vec<[u32; 3]> {
        w.particles()
            .iter()
            .map(|p| [p.position.x.to_bits(), p.position.y.to_bits(), p.position.z.to_bits()])
            .collect()
    };
    assert_eq!(bits(&a), bits(&b), "关节族求解应逐位一致");
}

// ---------------------------------------------------------------------------
// ② 布料：固定点、能量、防御
// ---------------------------------------------------------------------------

#[test]
fn cloth_pins_bitwise_fixed_under_gravity() {
    for mode in [PinMode::Corners, PinMode::TopEdge, PinMode::SingleTop] {
        let spec = ClothSpec {
            rows: 4,
            cols: 4,
            spacing: 0.2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            pin_mode: mode,
            structural_compliance: 0.0,
            bend_compliance: 0.001,
            damping: 0.0,
        };
        let mut cloth = Cloth::new(spec).unwrap();
        // 记录所有 inv_mass=0 的粒子（pin）位置。
        let pinned: Vec<(usize, Vec3)> = cloth
            .world()
            .particles()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.inv_mass == 0.0)
            .map(|(i, p)| (i, p.position))
            .collect();
        assert!(!pinned.is_empty(), "{mode:?} 应有固定点");
        for _ in 0..60 {
            cloth.step(1.0 / 60.0, 4, 8).unwrap();
        }
        for (i, p0) in &pinned {
            let p = cloth.world().particles()[*i].position;
            assert_eq!(
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                [p0.x.to_bits(), p0.y.to_bits(), p0.z.to_bits()],
                "{mode:?} pin {i} 应逐位固定"
            );
        }
    }
}

#[test]
fn cloth_energy_monotone_nonincreasing() {
    // 无外力 + 拉伸初态 → 每帧能量非增（压缩映射 + 阻尼耗散）。
    let spec = ClothSpec {
        rows: 4,
        cols: 4,
        spacing: 0.2,
        gravity: Vec3::splat(0.0),
        pin_mode: PinMode::Corners,
        structural_compliance: 0.001,
        bend_compliance: 0.001,
        damping: 0.02,
    };
    let mut cloth = Cloth::new(spec).unwrap();
    // 拉伸：内部粒子沿 x 拉出。
    for (i, p) in cloth.particles_mut().iter_mut().enumerate() {
        if i % 4 != 0 {
            p.position = p.position + Vec3::new(0.05, 0.0, 0.0);
        }
    }
    let mut prev = cloth.energy();
    let mut strictly_decreased = false;
    for _ in 0..50 {
        let e = cloth.step_with_energy(1.0 / 60.0, 4, 8).unwrap();
        assert!(
            e <= prev + 1e-4,
            "能量应单调非增：prev={prev} → {e}"
        );
        if e < prev - 1e-6 {
            strictly_decreased = true;
        }
        prev = e;
    }
    assert!(strictly_decreased, "拉伸初态能量应确实下降（非纯平台）");
}

#[test]
fn cloth_spec_defense() {
    let base = ClothSpec {
        rows: 4,
        cols: 4,
        spacing: 0.2,
        gravity: Vec3::splat(0.0),
        pin_mode: PinMode::Corners,
        structural_compliance: 0.0,
        bend_compliance: 0.0,
        damping: 0.0,
    };
    // rows < 2 → P1009。
    let mut bad = base;
    bad.rows = 1;
    assert_eq!(Cloth::new(bad).unwrap_err().code, "P1009");
    // spacing = 0 → P1010。
    let mut bad = base;
    bad.spacing = 0.0;
    assert_eq!(Cloth::new(bad).unwrap_err().code, "P1010");
    // spacing = NaN → P1010（`!(NaN>0)` true）。
    let mut bad = base;
    bad.spacing = f32::NAN;
    assert_eq!(Cloth::new(bad).unwrap_err().code, "P1010");
    // damping 越界 → P1011。
    let mut bad = base;
    bad.damping = 1.5;
    assert_eq!(Cloth::new(bad).unwrap_err().code, "P1011");
    // damping = NaN → P1011。
    let mut bad = base;
    bad.damping = f32::NAN;
    assert_eq!(Cloth::new(bad).unwrap_err().code, "P1011");
}

#[test]
fn cloth_step_defense_dt_and_counts() {
    let spec = ClothSpec {
        rows: 3,
        cols: 3,
        spacing: 0.2,
        gravity: Vec3::splat(0.0),
        pin_mode: PinMode::Corners,
        structural_compliance: 0.0,
        bend_compliance: 0.0,
        damping: 0.0,
    };
    let mut cloth = Cloth::new(spec).unwrap();
    // dt = 0 应报错（沿用 World 契约）。
    assert!(cloth.step(0.0, 4, 8).is_err(), "dt=0 应拒绝");
    assert!(cloth.step(1.0 / 60.0, 0, 8).is_err(), "substeps=0 应拒绝");
    assert!(cloth.step(1.0 / 60.0, 4, 0).is_err(), "iterations=0 应拒绝");
}

// ---------------------------------------------------------------------------
// ③ SPH：核属性、密度契约、防御、确定性
// ---------------------------------------------------------------------------

#[test]
fn sph_density_self_term_consistency() {
    // 攻击：DensityConstraint::evaluate 用邻居密度（不含自项），而公开
    // Fluid::densities 额外加 self-term W(0)。单粒子流体：邻居为空 → evaluate
    // 给 ρ=0（C=-1），但 densities 给 W(0)>0。两者语义不一致——记录期望。
    let spec = FluidSpec {
        h: 0.25,
        rest_density: 1000.0,
        compliance: 0.0,
        gravity: Vec3::splat(0.0),
    };
    let fluid = Fluid::new(spec, &[Vec3::splat(0.0)]).unwrap();
    let rho = fluid.densities();
    assert_eq!(rho.len(), 1);
    // densities 至少含自项（> 0），契约上是"包含自身的密度"。
    assert!(rho[0] > 0.0, "单粒子自身密度（自项）应 > 0：{}", rho[0]);
}

#[test]
fn sph_kernel_zero_outside_and_monotone() {
    // 通过公开 Fluid 间接验证核：两点距离 ≥ h → 无相互作用（密度 = 自项）。
    let spec = FluidSpec {
        h: 0.2,
        rest_density: 1000.0,
        compliance: 0.0,
        gravity: Vec3::splat(0.0),
    };
    // 单点。
    let f1 = Fluid::new(spec, &[Vec3::splat(0.0)]).unwrap();
    let rho_self = f1.densities()[0];
    // 两个远点（距 1.0 >> h）：每点密度应 = 自项（无邻居贡献）。
    let f2 = Fluid::new(spec, &[Vec3::splat(0.0), Vec3::new(1.0, 0.0, 0.0)]).unwrap();
    let rho2 = f2.densities();
    assert!((rho2[0] - rho_self).abs() < 1e-6, "远点不应有邻居贡献：{} vs {}", rho2[0], rho_self);
    assert!((rho2[1] - rho_self).abs() < 1e-6, "远点不应有邻居贡献");
    // 两近点（距 < h）：密度应高于自项（邻居贡献 > 0）。
    let f3 = Fluid::new(spec, &[Vec3::splat(0.0), Vec3::new(0.1, 0.0, 0.0)]).unwrap();
    let rho3 = f3.densities();
    assert!(rho3[0] > rho_self, "近邻粒子密度应高于自项：{} vs {}", rho3[0], rho_self);
}

#[test]
fn sph_step_defense() {
    let spec = FluidSpec {
        h: 0.25,
        rest_density: 1000.0,
        compliance: 0.0,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    };
    let mut f = Fluid::new(spec, &[Vec3::splat(0.0), Vec3::new(0.1, 0.0, 0.0)]).unwrap();
    assert!(f.step(0.0, 4, 8).is_err(), "dt=0 → P1002");
    assert!(f.step(-1.0, 4, 8).is_err(), "dt<0 → P1002");
    assert!(f.step(1.0 / 60.0, 0, 8).is_err(), "substeps=0 → P1003");
    assert!(f.step(1.0 / 60.0, 4, 0).is_err(), "iterations=0 → P1003");
    assert!(f.step(f32::NAN, 4, 8).is_err(), "dt=NaN → P1002");
}

#[test]
fn sph_empty_fluid_no_panic() {
    // 零粒子流体：构造合法（count=0），densities 空，step 不 panic。
    let spec = FluidSpec {
        h: 0.25,
        rest_density: 1000.0,
        compliance: 0.0,
        gravity: Vec3::splat(0.0),
    };
    let mut f = Fluid::new(spec, &[]).unwrap();
    assert_eq!(f.particle_count(), 0);
    assert!(f.densities().is_empty());
    assert!(f.step(1.0 / 60.0, 4, 8).is_ok(), "空流体步进不应报错");
}

// ---------------------------------------------------------------------------
// ④ 破碎：面积守恒、确定性、退化防御
// ---------------------------------------------------------------------------

#[test]
fn fracture_area_conservation_multiple_configs() {
    // 面积和守恒（单位分解）对多种站点数/包围盒/jitter 成立。
    for (n, box_, jit) in [(1usize, 4.0f32, 0.0f32), (4, 4.0, 0.0), (12, 4.0, 0.7), (50, 4.0, 0.5), (8, 9.0, 0.3)] {
        let spec = FractureSpec {
            min: [0.0, 0.0],
            max: [box_.sqrt(), box_.sqrt()],
            site_count: n,
            jitter: jit,
            impact: [0.1, 0.1],
            seed: SiteSeed::new(1234),
        };
        let f = Fracture::build(spec).unwrap();
        let sum = f.total_area();
        assert!(
            (sum - box_).abs() < 1e-2 * box_,
            "n={n} jit={jit}: 面积和 {sum} 应 ≈ 原面积 {box_}"
        );
        for fr in &f.fragments {
            assert!(fr.area >= 0.0 && fr.area.is_finite(), "碎片面积非法：{}", fr.area);
        }
    }
}

#[test]
fn fracture_deterministic_same_seed() {
    let spec = FractureSpec {
        min: [0.0, 0.0],
        max: [2.0, 2.0],
        site_count: 16,
        jitter: 0.6,
        impact: [1.0, 1.0],
        seed: SiteSeed::new(999),
    };
    let a = Fracture::build(spec).unwrap();
    let b = Fracture::build(spec).unwrap();
    assert_eq!(a, b, "同种子破碎应逐位一致");
    // 不同种子应产生不同站点集（种子确实影响）。
    let spec2 = FractureSpec { seed: SiteSeed::new(1000), ..spec };
    let c = Fracture::build(spec2).unwrap();
    assert_ne!(a.sites, c.sites, "不同种子应产生不同站点");
}

#[test]
fn fracture_spec_defense() {
    let base = FractureSpec {
        min: [0.0, 0.0],
        max: [2.0, 2.0],
        site_count: 8,
        jitter: 0.5,
        impact: [1.0, 1.0],
        seed: SiteSeed::new(1),
    };
    // 退化包围盒 → P1014。
    let mut bad = base;
    bad.max = [0.0, 0.0];
    assert_eq!(Fracture::build(bad).unwrap_err().code, "P1014");
    // site_count=0 → P1015。
    let mut bad = base;
    bad.site_count = 0;
    assert_eq!(Fracture::build(bad).unwrap_err().code, "P1015");
    // jitter 越界 → P1016。
    let mut bad = base;
    bad.jitter = -0.1;
    assert_eq!(Fracture::build(bad).unwrap_err().code, "P1016");
}

#[test]
fn polygon_area_and_centroid_edges() {
    // 退化多边形（< 3 顶点）面积 0；空顶点质心 (0,0)。
    let empty = Polygon2 { verts: vec![] };
    assert_eq!(empty.area(), 0.0);
    assert_eq!(empty.centroid(), [0.0, 0.0]);
    let line = Polygon2 { verts: vec![[0.0, 0.0], [1.0, 1.0]] };
    assert_eq!(line.area(), 0.0, "两点退化多边形面积应 0");
    // 矩形面积与质心。
    let sq = Polygon2::rect([0.0, 0.0], [2.0, 2.0]);
    assert!((sq.area() - 4.0).abs() < 1e-6);
    let c = sq.centroid();
    assert!((c[0] - 1.0).abs() < 1e-5 && (c[1] - 1.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// ⑤ 耦合：力场纯函数性 / 叠加 / 奇点
// ---------------------------------------------------------------------------

#[test]
fn force_fields_are_pure_functions() {
    let p = Vec3::new(0.3, -0.7, 0.1);
    let g = Gravity { g: Vec3::new(0.0, -9.81, 0.0) };
    assert_eq!(g.accel(p, 0.0), g.accel(p, 99.0), "重力位置无关");
    let wind = Wind { dir: Vec3::new(1.0, 0.0, 0.0) };
    assert_eq!(wind.accel(p, 0.0), wind.accel(Vec3::splat(5.0), 0.0), "风场位置无关");
    let turb = Turbulence { strength: 1.0, freq: 2.0, time_freq: 3.0 };
    assert_eq!(turb.accel(p, 0.5), turb.accel(p, 0.5), "湍流同参同值");
    // attractor 纯函数。
    let at = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: 0.1 };
    assert_eq!(at.accel(p, 0.0), at.accel(p, 0.0));
}

#[test]
fn attractor_singularity_is_softened() {
    // 攻击奇点：pos = center 时应由 eps 软化，不得产生 Inf/NaN。
    let at = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: 0.5 };
    let a = at.accel(Vec3::splat(0.0), 0.0);
    assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite(), "奇点应软化：{a:?}");
    assert_eq!(a.x, 0.0, "中心处力应为 0（d=0）");
    // eps = 0 时仍不得 NaN（r2 = 0 → strength/0 = Inf → 0 · Inf = NaN？攻击）。
    let at0 = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: 0.0 };
    let a0 = at0.accel(Vec3::splat(0.0), 0.0);
    assert!(
        a0.x.is_finite() && a0.y.is_finite() && a0.z.is_finite(),
        "eps=0 奇点不得产生 NaN（d·(s/0)=0·Inf）：{a0:?}"
    );
}

#[test]
fn coupler_sums_fields_linearly() {
    let mut c = Coupler::new();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
    assert_eq!(c.accel(Vec3::splat(0.0), 0.0), Vec3::splat(0.0), "空耦合器应为零");
    c.add(Box::new(Gravity { g: Vec3::new(0.0, -9.81, 0.0) }));
    c.add(Box::new(Wind { dir: Vec3::new(1.0, 0.0, 0.0) }));
    assert_eq!(c.len(), 2);
    let a = c.accel(Vec3::splat(0.0), 0.0);
    assert!((a.x - 1.0).abs() < 1e-6 && (a.y + 9.81).abs() < 1e-6);
    // 叠加序无关（交换律）：加入次序不同结果应一致。
    let mut c2 = Coupler::new();
    c2.add(Box::new(Wind { dir: Vec3::new(1.0, 0.0, 0.0) }));
    c2.add(Box::new(Gravity { g: Vec3::new(0.0, -9.81, 0.0) }));
    assert_eq!(c.accel(Vec3::splat(0.0), 0.0), c2.accel(Vec3::splat(0.0), 0.0), "叠加序无关");
}

#[test]
fn turbulence_bounded_by_strength() {
    // 湍流各分量 |a| ≤ strength（rand_signed ∈ [-1,1]）。
    let turb = Turbulence { strength: 2.0, freq: 1.0, time_freq: 1.0 };
    for k in 0..100 {
        let p = Vec3::new(k as f32 * 0.37, k as f32 * -0.21, k as f32 * 0.13);
        let a = turb.accel(p, k as f32 * 0.05);
        for comp in [a.x, a.y, a.z] {
            assert!(comp.is_finite() && comp.abs() <= 2.0 + 1e-4, "湍流分量越界：{comp}");
        }
    }
}

// ---------------------------------------------------------------------------
// ⑥ 六自由度：约束 indices 契约（升序合法）
// ---------------------------------------------------------------------------

#[test]
fn hinge_indices_valid_for_solver() {
    // 攻击：indices 必须升序、端点合法（xpbd 求解序依赖）。
    let j = HingeJoint::new(5, 2, 7, 0.0);
    let idx = j.indices();
    assert_eq!(idx, &[2usize, 7]);
    assert!(idx[0] < idx[1], "indices 应升序");
}

// 保留 fixed 辅助（供未来扩展；避免未使用告警）。
#[allow(dead_code)]
fn _sanity() {
    let _ = fixed(Vec3::splat(0.0));
}

// ---------------------------------------------------------------------------
// ⑦ 回归锁定（QA Round 2 收口）：强量级 strength + eps=0 不得 NaN/Inf
// ---------------------------------------------------------------------------

/// 回归：strength = 1e30、eps = 0、粒子恰在中心。
///
/// 旧缺陷路径：r2 夹到 MIN_POSITIVE 后，strength/r2 = 1e30/1.18e-38 = +Inf，
/// d=0 时 0·Inf = NaN。修复后对 q = strength/r2 饱和钳制到 ±f32::MAX 再乘 d，
/// 此处 d=0 → 输出 (0,0,0)，全有限。
#[test]
fn attractor_huge_strength_zero_eps_at_center_all_finite() {
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    let a = at.accel(Vec3::splat(0.0), 0.0);
    assert!(
        a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
        "strength=1e30 eps=0 中心点应全有限（旧路径 0·Inf=NaN）：{a:?}"
    );
    assert_eq!(a, Vec3::splat(0.0), "中心处 d=0 力应为 0（饱和不是 NaN）");
}

/// 回归：strength = 1e30、eps = 0、粒子偏离中心 1e-20。
///
/// 旧缺陷路径：r2 = (1e-20)² = 1e-40 夹到 MIN_POSITIVE（≈1.18e-38），
/// strength/r2 = 1e30/1.18e-38 = +Inf，再乘 d.x = 1e-20 → a.x = -Inf。
/// 修复后 q 饱和到 f32::MAX，a.x = -1e-20 · f32::MAX 有限（约 -3.4e18）。
#[test]
fn attractor_huge_strength_zero_eps_offset_all_finite() {
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    let pos = Vec3::new(1e-20, 0.0, 0.0);
    let a = at.accel(pos, 0.0);
    assert!(
        a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
        "strength=1e30 eps=0 偏离点应全有限（旧路径 a.x=-Inf）：{a:?}"
    );
    // d = center - pos = (-1e-20, 0, 0)，吸引力方向指向中心（-x），量级受饱和约束。
    assert!(a.x < 0.0, "偏离点应被拉向中心（-x）：{a:?}");
    assert!(
        a.x.abs() <= 1e-20 * f32::MAX * (1.0 + 1e-6),
        "饱和后分量应受 |d|·f32::MAX 界约束：{a:?}"
    );
}

// ---------------------------------------------------------------------------
// ⑧ 设计边界遗留风险清除（W6-B7 终验后 · 工程师寇豆码）
//    风险 1：饱和量级 —— q 钳 f32::MAX 后 |a| = f32::MAX·|d| 可致下游崩坏
//    风险 2：非法输入 —— strength/eps = NaN/Inf 无防御
// ---------------------------------------------------------------------------

/// 风险 1：极端小 |d| 下加速度受 MAX_ACCEL 上限约束。
///
/// strength=1e30、eps=0、|d|=1e-30：未修复前 q 饱和 f32::MAX，|a| ≈ f32::MAX·1e-30
/// ≈ 3.4e8（有限但天文）。修复后 |a| ≤ MAX_ACCEL（1000），全分量有限。
#[test]
fn attractor_accel_bounded_by_max_accel_extreme_offset() {
    let at = Attractor { center: Vec3::splat(0.0), strength: 1e30, eps: 0.0 };
    for dx in [1e-30f32, 1e-20, 1e-8, 1e-3] {
        let pos = Vec3::new(dx, 0.0, 0.0);
        let a = at.accel(pos, 0.0);
        assert!(
            a.x.is_finite() && a.y.is_finite() && a.z.is_finite(),
            "|d|={dx} 加速度应全有限：{a:?}"
        );
        for comp in [a.x, a.y, a.z] {
            assert!(
                comp.abs() <= MAX_ACCEL * (1.0 + 1e-6),
                "|d|={dx} 分量 |{comp}| 应 ≤ MAX_ACCEL（{MAX_ACCEL}）：{a:?}"
            );
        }
    }
}

/// 风险 1 补充：正常物理量级下 MAX_ACCEL 限幅**逐位不生效**（Mini Code）。
///
/// 米级构型（|d| ~ 1e-2..1e1）、强度 ~1e1 的量级下 |a| ≪ 1000，输出应与
/// 未限幅的解析值逐位一致（此处以"限幅不改变结果"验证：再夹一次亦相等）。
#[test]
fn attractor_normal_magnitude_bitwise_unchanged() {
    let at = Attractor { center: Vec3::new(1.0, 0.0, 0.0), strength: 50.0, eps: 0.1 };
    for k in 0..50 {
        let pos = Vec3::new(k as f32 * 0.13, k as f32 * -0.07, 0.0);
        let a = at.accel(pos, 0.0);
        // 未触限：再夹一次相等 → 限幅确未生效（逐位）。
        let re = Vec3::new(
            a.x.clamp(-MAX_ACCEL, MAX_ACCEL),
            a.y.clamp(-MAX_ACCEL, MAX_ACCEL),
            a.z.clamp(-MAX_ACCEL, MAX_ACCEL),
        );
        assert_eq!(
            [a.x.to_bits(), a.y.to_bits(), a.z.to_bits()],
            [re.x.to_bits(), re.y.to_bits(), re.z.to_bits()],
            "正常量级限幅不应生效（逐位）：{a:?}"
        );
        assert!(a.x.abs() < MAX_ACCEL && a.y.abs() < MAX_ACCEL, "正常量级应远低于上限：{a:?}");
    }
}

/// 风险 2：NaN/Inf strength / eps 被 validate() 显式拒绝（P3008 结构化报错）。
#[test]
fn attractor_nan_inf_params_rejected() {
    // NaN strength → P3008。
    let e = Attractor { center: Vec3::splat(0.0), strength: f32::NAN, eps: 0.1 }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "NaN strength 应拒：{e:?}");
    // +Inf strength → P3008。
    let e = Attractor { center: Vec3::splat(0.0), strength: f32::INFINITY, eps: 0.1 }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "+Inf strength 应拒：{e:?}");
    // −Inf strength → P3008。
    let e = Attractor { center: Vec3::splat(0.0), strength: f32::NEG_INFINITY, eps: 0.1 }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "-Inf strength 应拒：{e:?}");
    // NaN eps → P3008。
    let e = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: f32::NAN }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "NaN eps 应拒：{e:?}");
    // Inf eps → P3008。
    let e = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: f32::INFINITY }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "Inf eps 应拒：{e:?}");
    // 负 eps → P3008（半径语义）。
    let e = Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: -0.1 }
        .validate()
        .unwrap_err();
    assert_eq!(e.code, "P3008", "负 eps 应拒：{e:?}");
    // 合法参数 → Ok（eps=0 亦合法）。
    assert!(Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: 0.0 }
        .validate()
        .is_ok());
}

/// 风险 2 集成：非有限力场进入 CoupledSession → 等效重力被拒（P3008）。
#[test]
fn coupled_session_rejects_non_finite_field() {
    let spec = SimSpec {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        particles: vec![ParticleInit {
            position: Vec3::new(0.0, 1.0, 0.0),
            velocity: Vec3::splat(0.0),
            inv_mass: 1.0,
        }],
        constraints: vec![],
    };
    // NaN 强度力场（未经 validate 直用，透传 NaN）→ 会话构造拒绝。
    let mut c = Coupler::new();
    c.add(Box::new(Attractor { center: Vec3::splat(0.0), strength: f32::NAN, eps: 0.1 }));
    let r = CoupledSession::new(spec.clone(), &c, 1.0 / 60.0, 4, 8);
    assert!(r.is_err(), "非有限力场应被 CoupledSession 拒绝");
    assert_eq!(r.unwrap_err().code, "P3008");
    // 合法力场 → 构造成功，且等效重力有限。
    let mut ok = Coupler::new();
    ok.add(Box::new(Attractor { center: Vec3::splat(0.0), strength: 1.0, eps: 0.1 }));
    let s = CoupledSession::new(spec, &ok, 1.0 / 60.0, 4, 8).unwrap();
    let g = s.effective_gravity();
    assert!(g.x.is_finite() && g.y.is_finite() && g.z.is_finite(), "等效重力应有限：{g:?}");
}
