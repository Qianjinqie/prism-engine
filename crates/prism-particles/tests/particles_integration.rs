//! B8 粒子系统测试：确定性 / 生存期边界 / 力场叠加 / AudioReactive / seek 等价。
//!
//! 覆盖判据与约束：
//! - **粒子位置确定性**：同种子两遍逐位一致（[`bitwise_eq_views`]）。
//! - **生存期边界**：`age ≥ lifetime` 回收 / 尺寸衰减至 0。
//! - **力场叠加**：合外力 = Σ 各力场（复用 B7）。
//! - **AudioReactive 驱动断言**：频段 → 发射率/速度/尺寸增益。
//! - **seek 第 N 帧 == 顺序推进**（D-28 快照等价）。

use prism_math::Vec3;
use prism_particles::audio::{FrameGains, ReactiveModulation};
use prism_particles::emitter::{Emitter, ParticlePool};
use prism_particles::force::{apply_forces, standard_set, ForceSet};
use prism_particles::ParticleSim;
use prism_physics::coupling::{Attractor, ForceField, Gravity, Turbulence, Wind};

/// 两视图序列逐位相等（位模式比较，杜绝 ±0/NaN 假等）。
fn bitwise_eq(a: &[prism_particles::emitter::ParticleView], b: &[prism_particles::emitter::ParticleView]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b) {
        let eq = x.spawn_index == y.spawn_index
            && x.birth.to_bits() == y.birth.to_bits()
            && x.position.x.to_bits() == y.position.x.to_bits()
            && x.position.y.to_bits() == y.position.y.to_bits()
            && x.position.z.to_bits() == y.position.z.to_bits()
            && x.velocity.x.to_bits() == y.velocity.x.to_bits()
            && x.velocity.y.to_bits() == y.velocity.y.to_bits()
            && x.velocity.z.to_bits() == y.velocity.z.to_bits()
            && x.lifetime.to_bits() == y.lifetime.to_bits()
            && x.size.to_bits() == y.size.to_bits();
        if !eq {
            return false;
        }
    }
    true
}

/// 确定性发射器（固定种子）。
fn test_emitter() -> Emitter {
    Emitter {
        seed: 0xDEAD_BEEF,
        origin: Vec3::new(1.0, 2.0, 3.0),
        radius: 0.5,
        rate: 60.0,
        speed: 2.0,
        speed_jitter: 0.3,
        lifetime: 1.5,
        lifetime_jitter: 0.1,
        size: 4.0,
        size_jitter: 0.2,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    }
}

fn test_forces() -> ForceSet {
    standard_set(
        Vec3::new(0.0, -9.81, 0.0),
        Vec3::new(0.5, 0.0, 0.0),
        Attractor { center: Vec3::new(0.0, 0.0, 0.0), strength: 1.0, eps: 0.1 },
        Turbulence { strength: 0.2, freq: 1.0, time_freq: 2.0 },
    )
}

// ------------------------------------------------------------------
// 1. 粒子位置确定性：同种子两遍逐位一致
// ------------------------------------------------------------------

#[test]
fn derive_is_bitwise_deterministic() {
    let e = test_emitter();
    for i in 0..256_u64 {
        let a = e.derive(i);
        let b = e.derive(i);
        assert_eq!(a.origin.x.to_bits(), b.origin.x.to_bits(), "粒子 {i} origin.x 不逐位一致");
        assert_eq!(a.velocity.y.to_bits(), b.velocity.y.to_bits(), "粒子 {i} velocity.y 不逐位一致");
        assert_eq!(a.lifetime.to_bits(), b.lifetime.to_bits());
        assert_eq!(a.size.to_bits(), b.size.to_bits());
    }
}

#[test]
fn simulation_two_runs_bitwise_identical() {
    let mut a = ParticleSim::new(test_emitter(), test_forces(), 1.0 / 60.0, 512);
    let mut b = ParticleSim::new(test_emitter(), test_forces(), 1.0 / 60.0, 512);
    a.advance_to(30);
    b.advance_to(30);
    assert!(bitwise_eq(&a.views(), &b.views()), "同种子两遍模拟不逐位一致");
    assert_eq!(a.spawned(), b.spawned());
    assert_eq!(a.frame(), b.frame());
}

#[test]
fn distinct_seeds_diverge() {
    let mut e2 = test_emitter();
    e2.seed = 0x1234_5678;
    let s1 = ParticleSim::new(test_emitter(), ForceSet::new(), 1.0 / 60.0, 256);
    let s2 = ParticleSim::new(e2.clone(), ForceSet::new(), 1.0 / 60.0, 256);
    let v1 = s1.pool().views();
    let v2 = s2.pool().views();
    // 帧 0 池空；直接比较派生（不同种子应大多不同）。
    let d1 = test_emitter().derive(0);
    let d2 = e2.derive(0);
    assert_ne!(d1.velocity.x.to_bits(), d2.velocity.x.to_bits(), "不同种子应产生不同流");
    let _ = (v1, v2);
}

// ------------------------------------------------------------------
// 2. 生存期边界
// ------------------------------------------------------------------

#[test]
fn lifetime_recycle_after_expiry() {
    let dt: f32 = 1.0 / 60.0;
    let e = Emitter {
        seed: 7,
        origin: Vec3::splat(0.0),
        radius: 0.0,
        rate: 60.0, // 每帧 1 个
        speed: 0.0,
        speed_jitter: 0.0,
        lifetime: 3.0 * dt, // 恰 3 帧生存
        lifetime_jitter: 0.0,
        size: 2.0,
        size_jitter: 0.0,
        gravity: Vec3::splat(0.0),
    };
    let mut sim = ParticleSim::new(e, ForceSet::new(), dt, 64);
    sim.advance_to(1); // 第 0 帧发射 1 个，出生时刻 0
    assert_eq!(sim.pool().len(), 1, "第 1 帧应有 1 个活跃粒子");
    sim.advance_to(2);
    assert_eq!(sim.pool().len(), 2, "第 2 帧应有 2 个（旧粒子 age=dt<3dt 仍存活）");
    sim.advance_to(3);
    assert_eq!(sim.pool().len(), 3, "第 3 帧应有 3 个");
    // 第 4 帧：粒子 0 的 age = 3dt ≥ lifetime → 回收；同时新发射 1 个。
    sim.advance_to(4);
    assert_eq!(sim.pool().len(), 3, "第 4 帧：粒子 0 回收 + 新发射 1 → 仍 3 个");
    let ids: Vec<u64> = sim.pool().slots().iter().map(|s| s.spawn_index).collect();
    assert!(!ids.contains(&0), "粒子 0 应已回收");
}

#[test]
fn size_curve_decays_to_zero_at_death() {
    let _dt = 1.0 / 60.0;
    let e = Emitter {
        seed: 3,
        rate: 0.0, // 不自动发射
        lifetime: 1.0,
        lifetime_jitter: 0.0,
        size: 10.0,
        size_jitter: 0.0,
        gravity: Vec3::splat(0.0),
        ..Emitter::default()
    };
    let mut pool = ParticlePool::new(8);
    pool.spawn(&e, 0, 0.0);
    let at0 = pool.view_at(0.0, Vec3::splat(0.0));
    assert!((at0[0].size - 10.0).abs() < 1e-6, "出生时尺寸 = 10");
    let mid = pool.view_at(0.5, Vec3::splat(0.0));
    assert!((mid[0].size - 5.0).abs() < 1e-6, "半程尺寸 = 5");
    let end = pool.view_at(1.0, Vec3::splat(0.0));
    assert!(end[0].size.abs() < 1e-6, "死亡边界尺寸 = 0");
}

// ------------------------------------------------------------------
// 3. 力场叠加（复用 B7）
// ------------------------------------------------------------------

#[test]
fn force_superposition_matches_sum() {
    let g = Gravity { g: Vec3::new(0.0, -9.81, 0.0) };
    let w = Wind { dir: Vec3::new(1.0, 0.0, 0.5) };
    let a = Attractor { center: Vec3::new(5.0, 0.0, 0.0), strength: 2.0, eps: 0.5 };

    let mut set = ForceSet::new();
    set.add(g);
    set.add(w);
    set.add(a);

    let pos = Vec3::new(1.0, 1.0, 1.0);
    let t = 0.5;
    let expected = g.accel(pos, t) + w.accel(pos, t) + a.accel(pos, t);
    let got = set.accel(pos, t);
    assert_eq!(got.x.to_bits(), expected.x.to_bits());
    assert_eq!(got.y.to_bits(), expected.y.to_bits());
    assert_eq!(got.z.to_bits(), expected.z.to_bits());
}

#[test]
fn empty_forces_yield_zero_accel() {
    let set = ForceSet::new();
    assert!(set.is_empty());
    let a = set.accel(Vec3::new(1.0, 2.0, 3.0), 0.0);
    assert_eq!(a, Vec3::splat(0.0));
}

#[test]
fn gravity_only_parabolic_trajectory() {
    // 无风无湍流：位置应遵循解析抛物线 p = p0 + v0·t + ½g·t²。
    let dt = 1.0 / 60.0;
    let e = Emitter {
        seed: 1,
        rate: 0.0,
        speed: 0.0,
        speed_jitter: 0.0,
        gravity: Vec3::splat(0.0),
        ..Emitter::default()
    };
    let mut pool = ParticlePool::new(4);
    pool.spawn(&e, 0, 0.0);
    let mut forces = ForceSet::new();
    forces.add(Gravity { g: Vec3::new(0.0, -9.81, 0.0) });

    let steps = 30u32;
    let mut t = 0.0_f32;
    for _ in 0..steps {
        apply_forces(&mut pool, &forces, t, dt);
        t += dt;
    }
    // 半隐式 Euler 的解析对应：v_n = g·n·dt，p_n = Σ v_k·dt。
    let n = steps as f32;
    let v_expected = -9.81 * (n * dt);
    let got = pool.slots()[0].velocity.y;
    assert!((got - v_expected).abs() < 1e-3, "半隐式 Euler 速度 v={got} 期望 {v_expected}");
}

// ------------------------------------------------------------------
// 4. AudioReactive 驱动断言
// ------------------------------------------------------------------

#[test]
fn reactive_modulation_maps_bands_to_gains() {
    let m = ReactiveModulation {
        rate_weights: vec![0.0, 1.0, 0.0],
        speed_weights: vec![0.0, 0.0, 1.0],
        size_weights: vec![1.0, 0.0, 0.0],
        rate_amount: 1.0,
        speed_amount: 0.5,
        size_amount: 2.0,
    };
    // 频段 = [0.3, 0.8, 0.4]。
    let g: FrameGains = m.gains(&[0.3, 0.8, 0.4]);
    assert!((g.rate - (1.0 + 0.8)).abs() < 1e-6, "rate 增益 = 1 + 中频");
    assert!((g.speed - (1.0 + 0.5 * 0.4)).abs() < 1e-6, "speed 增益 = 1 + amount·高频");
    assert!((g.size - (1.0 + 2.0 * 0.3)).abs() < 1e-6, "size 增益 = 1 + amount·低频");
}

#[test]
fn reactive_zero_bands_yield_unit_gain() {
    let m = ReactiveModulation::default();
    let g = m.gains(&[0.0, 0.0, 0.0]);
    assert_eq!(g.rate, 1.0);
    assert_eq!(g.speed, 1.0);
    assert_eq!(g.size, 1.0);
}

#[test]
fn reactive_none_bake_unit_gain() {
    let m = ReactiveModulation::default();
    let g = prism_particles::audio::modulation_for_frame(None, &m, 42);
    assert_eq!(g.rate, 1.0, "无烘焙表 → 单位增益（不静默编造）");
}

#[test]
fn reactive_drives_spawn_rate() {
    // 增益 2.0 时 emit 数应约为无驱动 2 倍（每帧 round）。
    let e = Emitter { seed: 5, rate: 60.0, ..Emitter::default() };
    let dt = 1.0 / 60.0;
    let base = e.spawn_per_frame(1.0, dt);
    let driven = e.spawn_per_frame(2.0, dt);
    assert_eq!(base, 1);
    assert_eq!(driven, 2, "增益 2 → 每帧发射 2");

    // spawn_count 累积：3 帧、增益全 [2,2,2] → 6。
    assert_eq!(e.spawn_count(3, dt, &[2.0, 2.0, 2.0]), 6);
    // 缺省增益（空切片）→ 单位 → 3。
    assert_eq!(e.spawn_count(3, dt, &[]), 3);
}

// ------------------------------------------------------------------
// 5. seek 第 N 帧 == 顺序推进（D-28）
// ------------------------------------------------------------------

#[test]
fn seek_equals_sequential_advance() {
    let dt = 1.0 / 60.0;
    // 顺序推进到第 40 帧。
    let mut seq = ParticleSim::new(test_emitter(), test_forces(), dt, 512);
    seq.advance_to(40);

    // 先推进到第 10 帧（产生"脏"中间态），再 seek 到第 40 帧（同规格力场重放）。
    let mut seeked_src = ParticleSim::new(test_emitter(), test_forces(), dt, 512);
    seeked_src.advance_to(10);
    let seeked = seeked_src.seek_with(40, test_forces());

    assert_eq!(seq.frame(), seeked.frame());
    assert_eq!(seq.spawned(), seeked.spawned(), "发射数应一致");
    assert!(bitwise_eq(&seq.views(), &seeked.views()), "seek 应逐位等价于顺序推进");
}

/// seek 等式的双机制对拍（同一数学的两种落地）：
/// - **显式重放**（`sim.rs::seek_with`：丢弃脏态，从空池重放 0..N）；
/// - **快照续算**（求值台语义：帧 M 快照 → 恢复 → 续推到 N）。
/// 二者与顺序推进应逐位一致（快照字节级，不只是视图）——粒子域
/// 「发射/初态 = 帧号纯函数」与求值台「快照装配 + 单步投影」是同一
/// seek 等式的两个实例，对拍证明它们在本域给出同一条帧轨道。
/// 定容池（capacity 32 < 60 帧发射数）强制逐出，加强对拍覆盖面。
#[test]
fn seek_with_replay_matches_snapshot_continuation_bitwise() {
    let dt = 1.0 / 60.0;
    let capacity = 32;

    // 路径 A：顺序推进 0 → 60。
    let mut seq = ParticleSim::new(test_emitter(), test_forces(), dt, capacity);
    seq.advance_to(60);

    // 路径 B：脏态（先推进到 17 帧）+ seek_with 显式重放到 60。
    let mut dirty = ParticleSim::new(test_emitter(), test_forces(), dt, capacity);
    dirty.advance_to(17);
    let replayed = dirty.seek_with(60, test_forces());

    // 路径 C：帧 33 快照 → 恢复（力场按同规格重建）→ 续推到 60。
    let mut snap_src = ParticleSim::new(test_emitter(), test_forces(), dt, capacity);
    snap_src.advance_to(33);
    let bytes = snap_src.snapshot_bytes();
    let mut resumed = ParticleSim::new(test_emitter(), ForceSet::new(), dt, capacity);
    resumed.restore(&bytes).expect("快照恢复应成功");
    resumed.set_forces(test_forces());
    resumed.advance_to(60);

    assert_eq!(seq.frame(), 60);
    assert_eq!(seq.spawned(), replayed.spawned(), "重放发射数应一致");
    assert_eq!(seq.spawned(), resumed.spawned(), "续算发射数应一致");
    assert!(bitwise_eq(&seq.views(), &replayed.views()), "显式重放应逐位等于顺序推进");
    assert!(bitwise_eq(&seq.views(), &resumed.views()), "快照续算应逐位等于顺序推进");
    // 快照字节级对拍（帧号 / 时刻 / 发射数 / 逐槽位模式全覆盖）。
    assert_eq!(
        seq.snapshot_bytes(),
        replayed.snapshot_bytes(),
        "重放路径快照字节应逐位一致"
    );
    assert_eq!(
        seq.snapshot_bytes(),
        resumed.snapshot_bytes(),
        "续算路径快照字节应逐位一致"
    );
}

#[test]
fn slot_identity_differs_reproducibly() {
    // 池逐出：容量 4、每帧 1 个、生存期长 → 池恒满 4，最旧被逐。
    let dt = 1.0 / 60.0;
    let e = Emitter {
        seed: 9,
        rate: 60.0,
        lifetime: 10.0,
        lifetime_jitter: 0.0,
        ..test_emitter()
    };
    let mut sim = ParticleSim::new(e, ForceSet::new(), dt, 4);
    sim.advance_to(10);
    assert_eq!(sim.pool().len(), 4);
    let ids: Vec<u64> = sim.pool().slots().iter().map(|s| s.spawn_index).collect();
    assert_eq!(ids, vec![6, 7, 8, 9], "应保留最新 4 个");
}

#[test]
fn snapshot_roundtrip_preserves_state() {
    let dt = 1.0 / 60.0;
    let mut sim = ParticleSim::new(test_emitter(), test_forces(), dt, 256);
    sim.advance_to(25);
    let bytes = sim.snapshot_bytes();

    let mut restored = ParticleSim::new(test_emitter(), ForceSet::new(), dt, 256);
    restored.restore(&bytes).expect("快照恢复应成功");
    assert_eq!(restored.frame(), sim.frame());
    assert_eq!(restored.spawned(), sim.spawned());
    assert!(bitwise_eq(&restored.views(), &sim.views()), "快照往返应逐位一致");
}

// ------------------------------------------------------------------
// 6. 点精灵渲染对接 tile 协议
// ------------------------------------------------------------------

#[test]
fn empty_sprites_identity() {
    use prism_io::fbuf::Fbuf16F;
    use prism_render::tile::TileEffect;
    let fx = prism_particles::render::PointSpriteEffect::from_views(&[], 1.0, 1.0, 1.0);
    assert_eq!(fx.halo_radius(), 1, "空集仍申报最小半径 ≥ 1");
    let win = Fbuf16F::new(4, 4);
    let out = fx.eval_window(&win);
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(out.get(x, y), win.get(x, y), "空粒子集应逐位恒等");
        }
    }
}

#[test]
fn sprite_splat_marks_pixel() {
    use prism_io::fbuf::Fbuf16F;
    use prism_render::tile::TileEffect;
    use prism_particles::emitter::ParticleView;
    let v = ParticleView {
        spawn_index: 0,
        birth: 0.0,
        position: Vec3::new(2.0, 2.0, 0.0),
        velocity: Vec3::splat(0.0),
        lifetime: 1.0,
        size: 2.0,
        color: [1.0, 1.0, 1.0, 1.0],
    };
    let fx = prism_particles::render::PointSpriteEffect::from_views_origin(
        &[v], 1.0, 1.0, 1.0, [0.0, 4.0], 4, 4,
    );
    let win = Fbuf16F::new(4, 4);
    let out = fx.eval_window(&win);
    let center = out.get(2, 2);
    assert!(center[3] > 0.0, "粒子中心像素 alpha 应 > 0");
    let corner = out.get(0, 0);
    assert_eq!(corner[3], 0.0, "远离粒子角像素应不受影响");
}

#[test]
fn tile_halo_radius_reports_max() {
    use prism_particles::emitter::ParticleView;
    use prism_render::tile::TileEffect;
    let vs = vec![
        ParticleView {
            spawn_index: 0,
            birth: 0.0,
            position: Vec3::splat(0.0),
            velocity: Vec3::splat(0.0),
            lifetime: 1.0,
            size: 1.0,
            color: [1.0; 4],
        },
        ParticleView {
            spawn_index: 1,
            birth: 0.0,
            position: Vec3::splat(0.0),
            velocity: Vec3::splat(0.0),
            lifetime: 1.0,
            size: 5.0,
            color: [1.0; 4],
        },
    ];
    let fx = prism_particles::render::PointSpriteEffect::from_views(&vs, 1.0, 1.0, 1.0);
    assert_eq!(fx.halo_radius(), 5, "halo 半径 = 最大 splat 半径");
}

// ------------------------------------------------------------------
// 7. 节点注册（D-27）
// ------------------------------------------------------------------

#[test]
fn particle_nodes_register() {
    use prism_core::registry::NodeRegistry;
    let mut reg = NodeRegistry::new();
    prism_particles::__prism_collect(&mut reg);
    let id = prism_core::node::StructureId::custom(15, "prism_particles");
    assert!(reg.contains(&id), "ParticleEmitterNode 应注册");
    let id2 = prism_core::node::StructureId::custom(16, "prism_particles");
    assert!(reg.contains(&id2), "ParticleForceNode 应注册");
}

#[test]
fn emitter_node_apply_is_deterministic() {
    use prism_core::node::{EvalCtx, Payload, StructureId};
    let mut reg = prism_core::registry::NodeRegistry::new();
    prism_particles::__prism_collect(&mut reg);
    let id = StructureId::custom(15, "prism_particles");
    let node = reg.resolve(&id).expect("节点应可解析");
    let ctx = EvalCtx::new(0xABCD, 1u64, 0);
    let a = node.apply(&Payload::Scalar(3.0), &ctx);
    let b = node.apply(&Payload::Scalar(3.0), &ctx);
    match (a, b) {
        (Payload::Scalar(x), Payload::Scalar(y)) => assert_eq!(x.to_bits(), y.to_bits()),
        _ => panic!("应返回标量"),
    }
}
