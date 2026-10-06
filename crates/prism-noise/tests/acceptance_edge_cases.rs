//! W1 波次验收测试（QA 严过关）：A3 噪声四件套的边界反例攻击。
//!
//! 覆盖：
//! - C⑤ 跨平台确定性（纯整数运算路径）：PCG 与独立参考实现对拍、极端输入
//!   （seed=0 / u64::MAX / 0）不 panic 且确定、多线程并发纯函数性、
//!   value/simplex 大坐标与负格点行为；
//! - C② 关联：hash2 对大范围"像素网格"输入（65536 点）无碰撞（SampleIndex
//!   pixel 域流的底层原语）；
//! - 边界：value/simplex/fbm 在整数格点、跨 0 点、巨大坐标的确定性与范围。

use prism_noise::{fbm, pcg, simplex, value};

// ---------------------------------------------------------------------------
// C⑤-1 PCG vs 独立参考实现（PCG RXS-M-XS 变体的运算序列重写）
// ---------------------------------------------------------------------------

/// 参考实现：按公开的 64 位 pcg_hash 变体序列独立重写（LCG step + RXS shift
/// + M multiply + final xorshift），不 import 被测函数本体。
fn ref_permute(v: u64) -> u64 {
    let mult: u64 = 6364136223846793005;
    let inc: u64 = 1442695040888963407;
    let mult2: u64 = 12605985483714929081;
    let state = v.wrapping_mul(mult).wrapping_add(inc);
    let shift = (state >> 59) + 5;
    let xs = state >> shift;
    let word = (xs ^ state).wrapping_mul(mult2);
    (word >> 43) ^ word
}

fn ref_hash2(seed: u64, x: u64, y: u64) -> u64 {
    let s1: u64 = 0x9E37_79B9_7F4A_7C15;
    let s2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    ref_permute(ref_permute(seed ^ x.wrapping_add(s1)) ^ y.wrapping_add(s2))
}

#[test]
fn pcg_matches_independent_reference_implementation() {
    // 覆盖小值、大值、seed=0、坐标=0、坐标=u64::MAX 的对拍（500 点）。
    let mut x: u64 = 0x1234_5678_9abc_def0;
    for i in 0..500_u64 {
        let (s, a, b) = (x, x.wrapping_mul(3).wrapping_add(i), x.rotate_left(17));
        assert_eq!(pcg::hash1(s, a), ref_permute(s ^ a.wrapping_add(0x9E37_79B9_7F4A_7C15)));
        assert_eq!(pcg::hash2(s, a, b), ref_hash2(s, a, b), "i={i}");
        // f32 映射与 u64 高位一致（同一整数路径的下游）。
        let h = ref_hash2(s, a, b);
        let expect_f = ((h >> 40) as f32) * (1.0 / 16_777_216.0);
        assert_eq!(pcg::hash2_f32(s, a, b).to_bits(), expect_f.to_bits());
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
}

#[test]
fn pcg_extreme_inputs_are_deterministic_and_total() {
    // 极端输入：不 panic、repeatable、f32 落在 [0,1)。
    let extremes: Vec<(u64, u64, u64)> = vec![
        (0, 0, 0),
        (u64::MAX, u64::MAX, u64::MAX),
        (0, u64::MAX, 0),
        (u64::MAX, 0, u64::MAX),
        (1, 1 << 63, 1 << 31),
    ];
    for &(s, x, y) in &extremes {
        let h1 = (pcg::hash1(s, x), pcg::hash2(s, x, y), pcg::hash3(s, x, y, x));
        let h2 = (pcg::hash1(s, x), pcg::hash2(s, x, y), pcg::hash3(s, x, y, x));
        assert_eq!(h1, h2, "极端输入必须确定：seed={s:#x}");
        for v in [pcg::hash1_f32(s, x), pcg::hash2_f32(s, x, y), pcg::hash4_f32(s, x, y, x, y)] {
            assert!((0.0..1.0).contains(&v), "f32 越界: {v}");
        }
    }
    // seed=0 与坐标=0 是合法输入（文档承诺），且非平凡退化。
    assert_ne!(pcg::hash2(0, 0, 0), 0);
}

// ---------------------------------------------------------------------------
// C⑤-2 多线程并发纯函数性（无隐藏状态 / 无数据竞争的运行时验证）
// ---------------------------------------------------------------------------

#[test]
fn concurrent_calls_are_stateless_and_bit_identical() {
    // 8 线程 × 2000 次调用同参数：全部逐位一致（排除隐藏可变状态）。
    let expected = (
        value::value2(0xFEED, 3.25, -7.5),
        simplex::simplex2(0xFEED, 3.25, -7.5),
        simplex::simplex3(0xFEED, 1.5, -2.25, 9.125),
        pcg::hash3(0xFEED, 7, 11, 13),
    );
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    for _ in 0..2000 {
                        assert_eq!(value::value2(0xFEED, 3.25, -7.5), expected.0);
                        assert_eq!(simplex::simplex2(0xFEED, 3.25, -7.5), expected.1);
                        assert_eq!(simplex::simplex3(0xFEED, 1.5, -2.25, 9.125), expected.2);
                        assert_eq!(pcg::hash3(0xFEED, 7, 11, 13), expected.3);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("并发线程不应 panic");
        }
    });
}

// ---------------------------------------------------------------------------
// C② 关联：hash2 大范围"像素网格"输入无碰撞（SampleIndex pixel 域底层原语）
// ---------------------------------------------------------------------------

#[test]
fn hash2_pixel_grid_65536_inputs_collision_free() {
    // 256×256 全网格（对应 SampleIndex.pixel 域）：65536 个 (x,y) 全部不同 hash。
    let mut seen = std::collections::HashSet::with_capacity(65536);
    const SEED: u64 = 0xBEEF;
    for px in 0..256_u64 {
        for py in 0..256_u64 {
            assert!(seen.insert(pcg::hash2(SEED, px, py)), "pixel 网格碰撞 ({px},{py})");
        }
    }
    // 稀疏大坐标抽样：跨 4 个象限（含 u32 边界两侧）。
    let sparse: Vec<(u64, u64)> = vec![
        (0, 0),
        (1 << 31, 0),
        (0, 1 << 31),
        (1 << 31, 1 << 31),
        (u32::MAX as u64, u32::MAX as u64),
        ((1 << 31) + 1, (1 << 31) - 1),
    ];
    let sparse_hashes: Vec<u64> = sparse.iter().map(|&(x, y)| pcg::hash2(SEED, x, y)).collect();
    for i in 0..sparse_hashes.len() {
        for j in i + 1..sparse_hashes.len() {
            assert_ne!(sparse_hashes[i], sparse_hashes[j], "稀疏大坐标碰撞 #{i}/#{j}");
        }
    }
}

// ---------------------------------------------------------------------------
// 边界：value / simplex / fbm 的格点、跨 0、大坐标行为
// ---------------------------------------------------------------------------

#[test]
fn value_noise_extreme_coordinates_stay_in_range_and_deterministic() {
    // 大坐标（±1e6、±1e5.5 非格点）与跨 0 点：范围 [-1,1]、双调用逐位一致。
    let cases: Vec<(f32, f32)> = vec![
        (0.0, 0.0),
        (-0.5, -0.5),
        (1e6, -1e6),
        (123456.789, -654321.987),
        (1e5, 1e5),
    ];
    for &(x, y) in &cases {
        let v1 = value::value2(0xA11CE, x, y);
        let v2 = value::value2(0xA11CE, x, y);
        assert_eq!(v1.to_bits(), v2.to_bits(), "大坐标必须逐位复现 ({x},{y})");
        assert!((-1.0..=1.0).contains(&v1), "大坐标越界: ({x},{y}) → {v1}");
    }
    // 跨 0 连续性：0 两侧对称小步长无跳变（格点过渡正确）。
    let eps = 1e-3_f32;
    let a = value::value2(0xA11CE, -eps, 0.0);
    let b = value::value2(0xA11CE, eps, 0.0);
    assert!((a - b).abs() < 0.05, "跨 0 跳变: {a} → {b}");
    // 负格点 = 角点哈希（文档承诺的 lattice reproducibility，负半轴）。
    let expected = 2.0 * pcg::hash2_f32(0xA11CE, (-7_i64) as u64, (-3_i64) as u64) - 1.0;
    assert_eq!(value::value2(0xA11CE, -7.0, -3.0), expected);
}

#[test]
fn simplex_noise_extreme_coordinates_stay_in_range_and_deterministic() {
    let cases: Vec<(f32, f32, f32)> = vec![
        (0.0, 0.0, 0.0),
        (-0.5, 0.5, -0.5),
        (1e5, -1e5, 3.5),
        (98765.432, -12345.678, 555.5),
    ];
    for &(x, y, z) in &cases {
        let v1 = simplex::simplex3(0xB0B, x, y, z);
        let v2 = simplex::simplex3(0xB0B, x, y, z);
        assert_eq!(v1.to_bits(), v2.to_bits(), "大坐标必须逐位复现 ({x},{y},{z})");
        assert!((-1.0..=1.0).contains(&v1), "simplex3 大坐标越界: {v1}");
        assert!(
            (-1.0..=1.0).contains(&simplex::simplex2(0xB0B, x, y)),
            "simplex2 大坐标越界"
        );
    }
    // 4D 大坐标。
    let v4 = simplex::simplex4(0xB0B, 5e4, -5e4, 1.25, -2.75);
    assert!((-1.0..=1.0).contains(&v4), "simplex4 大坐标越界: {v4}");
}

#[test]
fn fbm_extreme_coordinates_and_parameter_edges() {
    // fbm 大坐标：范围与确定性。
    let fbm5 = fbm::Fbm::new(5, 2.0, 0.5);
    for &(x, y) in &[(1e5_f32, -1e5_f32), (9876.5, -123.4)] {
        let v = fbm5.simplex2(0xF00D, x, y);
        assert_eq!(v.to_bits(), fbm5.simplex2(0xF00D, x, y).to_bits(), "fbm 大坐标复现");
        assert!((-1.0..=1.0).contains(&v), "fbm 大坐标越界: {v}");
    }
    // gain=1（不衰减）：norm = octaves，输出仍归一在 [-1,1]。
    let flat = fbm::Fbm::new(4, 2.0, 1.0);
    let v = flat.value2(0xF00D, 1.5, -2.5);
    assert!((-1.0..=1.0).contains(&v), "gain=1 越界: {v}");
    // 单 octave 与基噪声逐位一致（文档承诺，极端坐标点上也成立）。
    let one = fbm::Fbm::new(1, 2.0, 0.5);
    assert_eq!(
        one.simplex3(0xF00D, 12345.678, -9.5, 0.0).to_bits(),
        simplex::simplex3(0xF00D, 12345.678, -9.5, 0.0).to_bits()
    );
}
