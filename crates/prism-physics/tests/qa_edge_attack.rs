//! QA 边界反例攻击（W3-B7 验收清单 C 项：物理骨架侧）。
//!
//! 与 src 内单测的分工：单测覆盖"实现者预期的行为"，本文件从**对抗视角**
//! 攻击公开 API 的边界——零长度/负值约束参数、奇异构型、极端刚度、快照
//! 字节注入与破坏。断言依据：算法白皮书 §8/§25（XPBD 语义）、架构总设计
//! §4.4（seek 等价）、兼容性宪章 C8/C18/C20（缓存三条件 / 防呆 / 禁静默）。

use prism_core::node::{Hash256, SnapshotStore};
use prism_math::Vec3;
use prism_physics::session::{
    DistanceConstraintSpec, ParticleInit, SimSession, SimSpec,
};
use prism_physics::xpbd::{DistanceConstraint, World};

const SIG_A: Hash256 = [0xAA; 32];

/// 三粒子竖直链（与 session.rs 单测同款基线：p0 固定，p1/p2 自由）。
fn chain_spec() -> SimSpec {
    SimSpec {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        particles: vec![
            ParticleInit { position: Vec3::new(0.0, 2.0, 0.0), velocity: Vec3::splat(0.0), inv_mass: 0.0 },
            ParticleInit { position: Vec3::new(0.0, 1.5, 0.0), velocity: Vec3::splat(0.0), inv_mass: 1.0 },
            ParticleInit { position: Vec3::new(0.0, 1.0, 0.0), velocity: Vec3::new(0.5, 0.0, 0.0), inv_mass: 1.0 },
        ],
        constraints: vec![
            DistanceConstraintSpec { id: 1, a: 0, b: 1, rest_length: 0.5, compliance: 0.0 },
            DistanceConstraintSpec { id: 2, a: 1, b: 2, rest_length: 0.5, compliance: 0.0 },
        ],
    }
}

fn session() -> SimSession {
    SimSession::new(chain_spec(), 1.0 / 60.0, 4, 8).unwrap()
}

/// 粒子位置/速度的位模式快照（逐位比较用，C24 平台中立）。
fn bits(s: &SimSession) -> Vec<[u32; 6]> {
    s.world()
        .particles()
        .iter()
        .map(|p| {
            [
                p.position.x.to_bits(),
                p.position.y.to_bits(),
                p.position.z.to_bits(),
                p.velocity.x.to_bits(),
                p.velocity.y.to_bits(),
                p.velocity.z.to_bits(),
            ]
        })
        .collect()
}

/// 按内部状态字节格式（magic "PRISMSIM" + version 1 + count + 每粒子
/// 6×f32 大端位模式）手工编码一份状态——用于向快照库注入。
fn encode_state(positions: &[Vec3], velocities: &[Vec3]) -> Vec<u8> {
    assert_eq!(positions.len(), velocities.len());
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"PRISMSIM");
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(&(positions.len() as u32).to_be_bytes());
    for (p, v) in positions.iter().zip(velocities) {
        for c in [p.x, p.y, p.z, v.x, v.y, v.z] {
            out.extend_from_slice(&c.to_bits().to_be_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 攻击 1：XPBD 约束参数边界（白皮书 §8：C = |d| − rest）
// ---------------------------------------------------------------------------

#[test]
fn zero_rest_length_converges_to_coincident_without_nan() {
    // rest = 0：目标 = 两自由粒子重合。求解应把间距收敛到奇异阈值附近，
    // 全程无 NaN/inf（重合后 evaluate 的 d.length() = 0 分支不得污染状态）。
    let mut world = World::new(Vec3::splat(0.0));
    let a = world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let b = world.add_particle(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    world
        .add_constraint(Box::new(DistanceConstraint::new(1, a, b, 0.0, 0.0)))
        .unwrap();

    for _ in 0..5 {
        let report = world.step(1.0 / 60.0, 4, 8).unwrap();
        assert!(
            report.residuals.iter().all(|r| r.is_finite()),
            "残余必须全程有限：{:?}",
            report.residuals
        );
    }
    let d = world.particles()[a].position - world.particles()[b].position;
    assert!(
        d.length() < 1e-3,
        "rest=0 应收敛到重合，实际间距 {}",
        d.length()
    );
    assert!(d.length().is_finite());
}

#[test]
fn coincident_particles_report_residual_truthfully() {
    // 重合粒子 + rest=1：奇异构型分支（len < 1e-8）必须**如实返回残余**
    // |C| = 1（白皮书 §25.5 纪律 3 / C20 不静默钳制），不得伪造收敛、
    // 不得 panic、不得产生 NaN。
    let mut world = World::new(Vec3::splat(0.0));
    let a = world.add_particle(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let b = world.add_particle(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    world
        .add_constraint(Box::new(DistanceConstraint::new(1, a, b, 1.0, 0.0)))
        .unwrap();

    let report = world.step(1.0 / 60.0, 4, 8).unwrap();
    let last = *report.residuals.last().unwrap();
    assert!(
        (last - 1.0).abs() < 1e-5,
        "重合粒子对 rest=1 的残余应如实 ≈ 1.0（不静默钳制），实际 {last}"
    );
    for p in world.particles() {
        assert!(p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite());
        assert!(p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite());
    }
}

#[test]
fn near_zero_compliance_converges_like_rigid() {
    // compliance = 1e-30（α̃ ≈ 0 的极限）：denom = Σw + α̃ 不得退化，
    // 收敛性应与完全刚性一致（XPBD 刚度谱上端边界）。
    let mut world = World::new(Vec3::splat(0.0));
    let a = world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
    let b = world.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    world
        .add_constraint(Box::new(DistanceConstraint::new(1, a, b, 1.0, 1e-30)))
        .unwrap();

    let report = world.step(1.0 / 60.0, 4, 8).unwrap();
    let final_residual = *report.residuals.last().unwrap();
    assert!(
        final_residual < 1e-4,
        "α→0 极限应如刚性收敛，最终残余 {final_residual}"
    );
    assert!(final_residual.is_finite());
}

#[test]
fn negative_compliance_in_world_path_is_rejected() {
    // 负 compliance（α < 0）：SimSpec 路径有 P1006 显式拒绝（session.rs
    // validate），但 World 直接构造路径必须同标准拒绝——否则 α̃ < 0 使
    // denom ≤ 0 分支静默吞掉约束（拉伸态不收敛且无任何报错），违反宪章
    // C20（禁止静默容错）与 C18 防呆一致性。
    let mut world = World::new(Vec3::splat(0.0));
    let a = world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 0.0);
    let b = world.add_particle(Vec3::new(1.2, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let add = world.add_constraint(Box::new(DistanceConstraint::new(1, a, b, 1.0, -0.5)));
    if add.is_ok() {
        // 若构造期未拦，step 必须拦（任一环节都不许静默放行）。
        let stepped = world.step(1.0 / 60.0, 4, 8);
        assert!(
            stepped.is_err(),
            "负 compliance 必须被显式拒绝（C20）：构造与 step 均静默放行，\
             约束将静默失效（denom≤0 分支吞掉投影）"
        );
    }
}

#[test]
fn negative_rest_length_in_world_path_is_rejected() {
    // 负 rest_length：SimSpec 路径有 P1005 拒绝；World 直接路径必须同标准
    // 拒绝（C = |d| + |rest| 恒不满足 → 求解器行为无物理意义）。
    let mut world = World::new(Vec3::splat(0.0));
    let a = world.add_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let b = world.add_particle(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(0.0), 1.0);
    let add = world.add_constraint(Box::new(DistanceConstraint::new(1, a, b, -1.0, 0.0)));
    if add.is_ok() {
        let stepped = world.step(1.0 / 60.0, 4, 8);
        assert!(
            stepped.is_err(),
            "负 rest_length 必须被显式拒绝（与 SimSpec P1005 对齐，C20）：\
             构造与 step 均静默放行"
        );
    }
}

// ---------------------------------------------------------------------------
// 攻击 2：SimSession 快照路径（D-28 / 架构 §4.4 / C8）
// ---------------------------------------------------------------------------

#[test]
fn snapshot_hit_restores_injected_bytes_proving_no_recompute() {
    // 【B③ 注入式证明】向 (SIG_A, 3) 槽注入一份"合法编码但与物理推进
    // 无关"的状态（全零）：seek 回 3 后若世界状态 == 注入值，即证明
    // advance_to 走的是快照恢复路径而非从初态重算（重算结果必然非零）。
    let mut s = session();
    let mut store = SnapshotStore::new();
    s.advance_to(SIG_A, 5, &mut store).unwrap();
    let physical = bits(&s);
    assert!(
        physical.iter().any(|p| p.iter().any(|&b| b != 0)),
        "前置校验：推进 5 帧后物理状态必须非零（否则本测试无区分力）"
    );

    let zero = vec![Vec3::splat(0.0); 3];
    let injected = encode_state(&zero, &zero);
    store.insert((SIG_A, 3), injected);

    s.advance_to(SIG_A, 3, &mut store).unwrap();
    for (i, p) in s.world().particles().iter().enumerate() {
        assert_eq!(
            p.position.x.to_bits(),
            0u32,
            "粒子 {i} 位置应等于注入值（快照命中即恢复，未经重算）"
        );
        assert_eq!(p.velocity.x.to_bits(), 0u32, "粒子 {i} 速度应等于注入值");
    }
}

#[test]
fn corrupt_snapshot_falls_back_to_rebuild_bitwise_identical() {
    // 快照槽内容破坏（magic 错 + 长度截断）：decode 必须失败（绝不猜测
    // 解读，C8/C20），advance_to 落入失效重建，结果与干净会话顺序推进
    // 逐位一致（架构 §4.4：缓存命中与否不影响结果）。
    let mut s = session();
    let mut store = SnapshotStore::new();
    s.advance_to(SIG_A, 5, &mut store).unwrap();

    // 注入两种破坏字节到 (SIG_A, 3)：magic 错误 / 长度截断。
    let mut bad_magic = encode_state(&[Vec3::splat(1.0); 3], &[Vec3::splat(1.0); 3]);
    bad_magic[0] = b'X';
    store.insert((SIG_A, 3), bad_magic);
    let truncated: Vec<u8> = {
        let real = encode_state(&[Vec3::splat(2.0); 3], &[Vec3::splat(0.0); 3]);
        real[..real.len() - 8].to_vec()
    };
    store.insert((SIG_A, 3), truncated);

    s.advance_to(SIG_A, 3, &mut store).unwrap();

    // 对照：干净会话顺序推进到 3。
    let mut clean = session();
    let mut clean_store = SnapshotStore::new();
    clean.advance_to(SIG_A, 3, &mut clean_store).unwrap();
    assert_eq!(
        bits(&s),
        bits(&clean),
        "破坏快照必须触发重建且与顺序推进逐位一致（绝不猜测解读坏字节）"
    );
}

#[test]
fn out_of_order_seek_is_bitwise_identical_to_sequential() {
    // 帧号乱序：推进 5 → 回 2（快照命中）→ 跳 7。三段混合路径的最终状态
    // 必须与干净会话顺序推进到 7 逐位一致（seek 等价的乱序强化版）。
    let mut s = session();
    let mut store = SnapshotStore::new();
    s.advance_to(SIG_A, 5, &mut store).unwrap();
    s.advance_to(SIG_A, 2, &mut store).unwrap();
    s.advance_to(SIG_A, 7, &mut store).unwrap();

    let mut clean = session();
    let mut clean_store = SnapshotStore::new();
    for f in 0..=7u64 {
        clean.advance_to(SIG_A, f, &mut clean_store).unwrap();
    }
    assert_eq!(
        bits(&s),
        bits(&clean),
        "乱序 seek 混合路径应与顺序推进逐位一致（架构 §4.4）"
    );
}
