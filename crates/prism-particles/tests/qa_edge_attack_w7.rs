//! W7 QA 攻击性测试 —— prism-particles（W7-B8）。
//!
//! 目标：找真 Bug（边界/退化、确定性、防御缺失、API 契约），非复读已有断言。
//! 每个测试注明"失败代表什么缺陷"。

use prism_math::Vec3;
use prism_particles::audio::{FrameGains, ReactiveModulation};
use prism_particles::emitter::{Emitter, ParticlePool, ParticleView};
use prism_particles::force::{apply_forces, ForceSet};
use prism_particles::ParticleSim;
use prism_physics::coupling::Gravity;

// ==================================================================
// 1. 边界 / 退化输入
// ==================================================================

/// 缺陷：容量 0 的池被 spawn 时 `slots.len() >= 0` 恒真 → `remove(0)` 在空 vec 上 panic。
#[test]
fn capacity_zero_pool_spawn_must_not_panic() {
    let e = Emitter { seed: 1, rate: 0.0, ..Emitter::default() };
    let mut pool = ParticlePool::new(0);
    // 库代码不得 panic——容量 0 应安全降级（丢弃粒子），而非索引越界。
    pool.spawn(&e, 0, 0.0);
    assert_eq!(pool.len(), 0, "容量 0 池不应容纳任何粒子");
}

/// 缺陷：容量 0 的 sim 推进也应安全（每帧发射都触发 spawn）。
#[test]
fn capacity_zero_sim_step_must_not_panic() {
    let e = Emitter { seed: 1, rate: 1000.0, ..Emitter::default() };
    let mut sim = ParticleSim::new(e, ForceSet::new(), 1.0 / 60.0, 0);
    sim.advance_to(5);
    assert_eq!(sim.pool().len(), 0, "容量 0 池恒空");
}

/// 缺陷：零发射率应发射 0 粒子（空粒子集），池恒空。
#[test]
fn zero_rate_emits_nothing() {
    let e = Emitter { seed: 2, rate: 0.0, ..Emitter::default() };
    let mut sim = ParticleSim::new(e, ForceSet::new(), 1.0 / 60.0, 64);
    sim.advance_to(100);
    assert_eq!(sim.spawned(), 0, "零发射率不应发射粒子");
    assert!(sim.pool().is_empty());
    assert!(sim.views().is_empty());
}

/// 缺陷：零/负生存期粒子应即时死亡，不应残留在池中（或渲染出来）。
#[test]
fn zero_lifetime_particle_recycled_immediately() {
    let dt = 1.0 / 60.0;
    let e = Emitter {
        seed: 3,
        rate: 60.0,
        lifetime: 0.0,
        lifetime_jitter: 0.0,
        ..Emitter::default()
    };
    let mut sim = ParticleSim::new(e, ForceSet::new(), dt, 64);
    sim.advance_to(10);
    assert!(sim.pool().is_empty(), "寿命 0 的粒子应在帧边界立即回收");
}

/// 静息发射器（初速度 0），用于隔离重力积分（避免 derive 抖动初速干扰）。
fn rest_emitter(seed: u64) -> Emitter {
    Emitter {
        seed,
        rate: 0.0,
        speed: 0.0,
        speed_jitter: 0.0,
        lifetime: 1e9,
        lifetime_jitter: 0.0,
        ..Emitter::default()
    }
}

/// 缺陷：dt = 0 时 `spawn_per_frame` 应发 0；`apply_forces` 应 no-op 且不 NaN。
#[test]
fn zero_dt_degrades_safely() {
    let e = Emitter { seed: 4, rate: 60.0, ..Emitter::default() };
    assert_eq!(e.spawn_per_frame(1.0, 0.0), 0, "dt=0 不应发射粒子");
    let mut pool = ParticlePool::new(4);
    pool.spawn(&rest_emitter(4), 0, 0.0);
    let mut f = ForceSet::new();
    f.add(Gravity { g: Vec3::new(0.0, -9.81, 0.0) });
    apply_forces(&mut pool, &f, 0.0, 0.0);
    let v = pool.slots()[0].velocity;
    assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite(), "dt=0 不应产生 NaN/Inf");
    assert_eq!(v.y, 0.0, "dt=0 速度不应变化");
}

/// 缺陷：负 dt 应视为非法（apply_forces 负 dt 会倒推 → 应 no-op）。
#[test]
fn negative_dt_apply_forces_noop() {
    let mut pool = ParticlePool::new(4);
    pool.spawn(&rest_emitter(5), 0, 0.0);
    let mut f = ForceSet::new();
    f.add(Gravity { g: Vec3::new(0.0, -9.81, 0.0) });
    apply_forces(&mut pool, &f, 0.0, -0.1);
    assert_eq!(pool.slots()[0].velocity.y, 0.0, "负 dt 应 no-op");
}

/// 缺陷：极大发射率 × 大 dt 会溢出（u64 回绕 / float→u64 saturate）。
/// 期望：不 panic、结果有界（不应回绕成小值/0）。
#[test]
fn extreme_rate_does_not_wrap() {
    let e = Emitter { seed: 6, rate: 1.0e30, ..Emitter::default() };
    // 单帧巨大但有限：应饱和到 u64 上限附近而非回绕。
    let n = e.spawn_per_frame(1.0, 1.0);
    // 若回绕会得到荒谬小值；只断言"不为 0 且为极大值"。
    assert!(n > 1_000_000_000, "极端发射率应产生极大量（不回绕为小值），得 {n}");
}

// ==================================================================
// 2. 确定性攻击（核心铁律）
// ==================================================================

fn bitwise_eq(a: &[ParticleView], b: &[ParticleView]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).all(|(x, y)| {
        x.spawn_index == y.spawn_index
            && x.birth.to_bits() == y.birth.to_bits()
            && x.position.x.to_bits() == y.position.x.to_bits()
            && x.position.y.to_bits() == y.position.y.to_bits()
            && x.position.z.to_bits() == y.position.z.to_bits()
            && x.velocity.x.to_bits() == y.velocity.x.to_bits()
            && x.velocity.y.to_bits() == y.velocity.y.to_bits()
            && x.velocity.z.to_bits() == y.velocity.z.to_bits()
            && x.lifetime.to_bits() == y.lifetime.to_bits()
            && x.size.to_bits() == y.size.to_bits()
            && x.color[0].to_bits() == y.color[0].to_bits()
            && x.color[1].to_bits() == y.color[1].to_bits()
            && x.color[2].to_bits() == y.color[2].to_bits()
            && x.color[3].to_bits() == y.color[3].to_bits()
    })
}

fn emitter_windy() -> Emitter {
    Emitter {
        seed: 0xABCD_EF01,
        origin: Vec3::new(0.5, -1.0, 2.0),
        radius: 1.0,
        rate: 240.0,
        speed: 3.0,
        speed_jitter: 0.5,
        lifetime: 2.0,
        lifetime_jitter: 0.5,
        size: 3.0,
        size_jitter: 0.5,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    }
}

/// D-28 核心：seek 第 N 帧 ≡ 顺序推进到第 N 帧，逐位一致。
#[test]
fn seek_40_equals_sequential_40() {
    let mut seq = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 512);
    seq.advance_to(40);
    let seeked = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 512).seek(40);
    assert_eq!(seq.spawned(), seeked.spawned(), "spawned 计数应一致");
    assert!(bitwise_eq(&seq.views(), &seeked.views()), "seek(40) 应逐位等于顺序推进(40)");
}

/// 强确定性：同输入两遍完整模拟逐字节一致（含快照字节）。
#[test]
fn two_runs_snapshot_bytes_identical() {
    let mut a = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 256);
    let mut b = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 256);
    a.advance_to(77);
    b.advance_to(77);
    assert_eq!(a.snapshot_bytes(), b.snapshot_bytes(), "同输入两遍快照字节应逐位一致");
}

/// seek 到已过帧（target < frame）应无操作，不回退。
#[test]
fn seek_backward_is_noop() {
    let mut s = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 256);
    s.advance_to(20);
    // target <= frame → advance_to 无操作。
    let out = s.seek(10);
    assert_eq!(out.frame(), 10);
    // seek 返回新实例，原实例不变。
    assert_eq!(s.frame(), 20, "seek 不应修改原实例");
}

/// 快照-恢复-再推进 == 顺序推进（D-28 快照承载等价）。
#[test]
fn restore_then_continue_equals_sequential() {
    let dt = 1.0 / 60.0;
    let mut base = ParticleSim::new(emitter_windy(), ForceSet::new(), dt, 256);
    base.advance_to(30);
    let bytes = base.snapshot_bytes();

    let mut resumed = ParticleSim::new(emitter_windy(), ForceSet::new(), dt, 256);
    resumed.restore(&bytes).expect("恢复应成功");
    resumed.advance_to(60);

    let mut seq = ParticleSim::new(emitter_windy(), ForceSet::new(), dt, 256);
    seq.advance_to(60);

    assert_eq!(resumed.spawned(), seq.spawned());
    assert!(bitwise_eq(&resumed.views(), &seq.views()), "快照恢复续推应等于顺序推进");
}

/// 快照截断必须结构化报错，不得 panic。
#[test]
fn truncated_snapshot_errors_not_panic() {
    let mut s = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 64);
    s.advance_to(10);
    let bytes = s.snapshot_bytes();
    let mut target = ParticleSim::new(emitter_windy(), ForceSet::new(), 1.0 / 60.0, 64);
    // 各种截断长度都不应 panic。
    for cut in 0..bytes.len().min(64) {
        let _ = target.restore(&bytes[..cut]);
    }
    // 全长度成功。
    target.restore(&bytes).expect("完整快照应恢复成功");
}

// ==================================================================
// 3. 力场 / 积分正确性
// ==================================================================

/// 半隐式 Euler 恒定重力下的解析位置：p_n = ½·g·dt²·n(n+1)（v_n = g·n·dt 后 p += v·dt）。
#[test]
fn semi_implicit_euler_position_matches_closed_form() {
    let dt = 1.0 / 240.0;
    let g = -9.81_f32;
    let ee = rest_emitter(7);
    let mut pool = ParticlePool::new(4);
    pool.spawn(&ee, 0, 0.0);
    let mut f = ForceSet::new();
    f.add(Gravity { g: Vec3::new(0.0, g, 0.0) });

    let n: u32 = 60;
    let mut t = 0.0_f32;
    for _ in 0..n {
        apply_forces(&mut pool, &f, t, dt);
        t += dt;
    }
    let nf = n as f32;
    // v_n = g·n·dt; p_n = Σ_{k=1..n} v_k·dt = g·dt²·n(n+1)/2
    let v_exp = g * (nf * dt);
    let p_exp = g * dt * dt * (nf * (nf + 1.0) / 2.0);
    let v_got = pool.slots()[0].velocity.y;
    let p_got = pool.slots()[0].position.y;
    assert!((v_got - v_exp).abs() < 1e-3, "速度 v={v_got} 期望 {v_exp}");
    assert!((p_got - p_exp).abs() < 1e-3, "位置 p={p_got} 期望 {p_exp}");
}

// ==================================================================
// 4. AudioReactive 防御 / 契约
// ==================================================================

/// 缺陷：NaN 频段输入应被拒绝或钳制，不得产生 NaN 增益。
#[test]
fn reactive_nan_bands_yield_finite_gain() {
    let m = ReactiveModulation::default();
    let g = m.gains(&[f32::NAN, 0.0, 0.0]);
    assert!(g.rate.is_finite(), "NaN 频段不应产生 NaN 增益，得 {}", g.rate);
}

/// 权重与频段长度不匹配：越界应视为 0（契约：min 长度求和），不得 panic。
#[test]
fn reactive_weight_length_mismatch_no_panic() {
    let m = ReactiveModulation {
        rate_weights: vec![1.0],
        speed_weights: vec![],
        size_weights: vec![1.0, 1.0, 1.0],
        rate_amount: 1.0,
        speed_amount: 1.0,
        size_amount: 1.0,
    };
    let g: FrameGains = m.gains(&[0.5, 0.5, 0.5]);
    assert!(g.rate.is_finite());
    assert!(g.speed.is_finite());
    assert!(g.size.is_finite());
}

// ==================================================================
// 5. 渲染退化
// ==================================================================

mod render_attack {
    use super::*;
    use prism_io::fbuf::Fbuf16F;
    use prism_particles::render::PointSpriteEffect;
    use prism_render::tile::TileEffect;

    /// 缺陷：半径 0 或负的粒子不应绘制（跳过），输入窗口不被修改。
    #[test]
    fn zero_radius_sprite_not_drawn() {
        let v = ParticleView {
            spawn_index: 0,
            birth: 0.0,
            position: Vec3::new(1.0, 1.0, 0.0),
            velocity: Vec3::splat(0.0),
            lifetime: 1.0,
            size: 0.0, // → 半径 0
            color: [1.0; 4],
        };
        let fx = PointSpriteEffect::from_views_origin(&[v], 1.0, 1.0, 1.0, [0.0, 0.0], 4, 4);
        let win = Fbuf16F::new(4, 4);
        let out = fx.eval_window(&win);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(x, y), win.get(x, y), "半径 0 精灵不应改变像素");
            }
        }
    }

    /// 缺陷：NaN 位置/尺寸精灵不得 panic（可能 i64 转换 UB / 越界）。
    #[test]
    fn nan_sprite_no_panic() {
        let v = ParticleView {
            spawn_index: 0,
            birth: 0.0,
            position: Vec3::new(f32::NAN, f32::NAN, 0.0),
            velocity: Vec3::splat(0.0),
            lifetime: 1.0,
            size: f32::NAN,
            color: [1.0; 4],
        };
        let fx = PointSpriteEffect::from_views_origin(&[v], 1.0, 1.0, 1.0, [0.0, 0.0], 4, 4);
        let win = Fbuf16F::new(4, 4);
        let _ = fx.eval_window(&win);
    }

    /// 缺陷：像素中心处核权重应为最大（1−0²)² = intensity；边缘处权重趋 0。
    #[test]
    fn splat_kernel_peaks_at_center() {
        let v = ParticleView {
            spawn_index: 0,
            birth: 0.0,
            position: Vec3::new(2.0, 2.0, 0.0),
            velocity: Vec3::splat(0.0),
            lifetime: 1.0,
            size: 3.0,
            color: [0.0, 0.0, 0.0, 1.0], // 只测 alpha
        };
        let fx = PointSpriteEffect::from_views_origin(&[v], 1.0, 1.0, 1.0, [0.0, 4.0], 4, 4);
        let win = Fbuf16F::new(4, 4);
        let out = fx.eval_window(&win);
        // 世界(2,2) → 像素(2,4-2=2) 中心。
        let center = out.get(2, 2)[3];
        let near = out.get(3, 2)[3];
        assert!(center > near, "中心 alpha({center}) 应 > 邻近({near})");
        assert!(center > 0.0);
    }
}
