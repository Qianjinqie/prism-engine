//! 第 2 轮独立复核探针—— ParticlePool capacity==0。
//! 验证 P1 修复未遗漏路径：spawn 之外还有 view_at / views / slots 等只读入口。

use prism_math::Vec3;
use prism_particles::{Emitter, ParticlePool};

fn tiny_emitter() -> Emitter {
    Emitter { seed: 1, rate: 0.0, ..Emitter::default() }
}

/// capacity==0：反复 spawn 不 panic、len 恒 0。
#[test]
fn particle_zero_capacity_spawn_storm_no_panic() {
    let mut pool = ParticlePool::new(0);
    let e = tiny_emitter();
    for i in 0..1000 {
        pool.spawn(&e, i, i as f32);
    }
    assert_eq!(pool.len(), 0, "零容量池不应容纳粒子");
    assert_eq!(pool.capacity(), 0);
    assert!(pool.is_empty());
}

/// capacity==0：只读入口不得 panic。
#[test]
fn particle_zero_capacity_read_paths_no_panic() {
    let pool = ParticlePool::new(0);
    assert!(pool.slots().is_empty());
    assert!(pool.views().is_empty());
    let v = pool.view_at(1.0, Vec3::new(0.0, -9.8, 0.0));
    assert!(v.is_empty());
}

/// capacity=1：逐出最旧，len 恒 ≤ 1。
#[test]
fn particle_capacity_one_eviction() {
    let mut pool = ParticlePool::new(1);
    let e = tiny_emitter();
    for i in 0..10 {
        pool.spawn(&e, i, i as f32);
        assert!(pool.len() <= 1);
    }
    assert_eq!(pool.len(), 1);
}
