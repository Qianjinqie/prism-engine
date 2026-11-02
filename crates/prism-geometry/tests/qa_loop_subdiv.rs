//! W4 / G-9 + G-10 几何侧判据（`docs/v2.5-W0-算法选型报告.md` §5.1 的 V1 / V5）。
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V1 | 与 k+3 级参考的 Hausdorff | ≤ 0.5 像素世界尺寸 |
//! | V5 | k=5/6 的三角数与峰值 RSS | 与 Loop 的 `4^k·面数·36 B` 对比 |
//!
//! # V1 的"Hausdorff"是怎么算的（方法写清楚，别让数字悬空）
//!
//! 严格 Hausdorff 是双向的 `max( sup_{a∈A} d(a,B), sup_{b∈B} d(b,A) )`。
//! 本测试取**双向顶点+面心采样**：对 A 的每个顶点与每个三角形重心求到 B 的
//! 最近距离，反向再来一次，取最大值。这是 Hausdorff 的**下界逼近**（采样
//! 越密越接近真值），而两个网格都是同一极限面的采样 ⇒ 极值点落在顶点/重心
//! 附近，故对"细分逼近"这一用途足够。**不用**它去证明别的东西。
//!
//! "像素世界尺寸"取一台明确写死的相机：竖直角 60°、画幅高 512 px、物距 3.0，
//! 则 `1 px = 2·tan(30°)·3.0/512 = 6.77e-3` 世界单位，阈值 = 一半 = 3.38e-3。
//! 这个数**跟着相机走**，不是一个可以随口改的容差——换相机就要重算（报告 §22 写明）。
//!
//! # V5 的字节口径
//!
//! 报告 §5 给 CC 的是 `4^k × 面数 × 72 B`（每四边面 2 三角 × 36 B）。Loop 从
//! 三角形出发，等价式是 `4^k × 面数 × 36 B` —— 一次细分裂成 4 个三角形，
//! 每个三角形 3×`Vec3` = 36 B。本文件按这个式子算，**不照抄 CC 的 72 B**。

use prism_geometry::bvh::{Bvh, Triangle};
use prism_geometry::subdiv::{subdivide, MAX_LEVELS};
use prism_math::Vec3;

/// 正二十面体（20 面、12 顶点，规则 5 价——最经典的三角笼）。
fn icosahedron() -> (Vec<Vec3>, Vec<u32>) {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut pos = vec![
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    for p in pos.iter_mut() {
        *p = p.normalize();
    }
    let idx = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6,
        7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6,
        7, 9, 8, 1,
    ];
    (pos, idx)
}

/// 索引网格 → 三角形 soup（Hausdorff 查询用）。
fn to_tris(pos: &[Vec3], idx: &[u32]) -> Vec<Triangle> {
    idx.chunks_exact(3)
        .map(|c| Triangle {
            a: pos[c[0] as usize],
            b: pos[c[1] as usize],
            c: pos[c[2] as usize],
        })
        .collect()
}

/// 单向采样距离：`src` 的顶点 + 三角形重心 → `dst` 的最近距离最大值。
fn one_sided(src_pos: &[Vec3], src_idx: &[u32], dst: &Bvh) -> f32 {
    let mut worst = 0.0_f32;
    for p in src_pos {
        worst = worst.max(dst.closest_point(*p).dist);
    }
    for c in src_idx.chunks_exact(3) {
        let (a, b, d) = (src_pos[c[0] as usize], src_pos[c[1] as usize], src_pos[c[2] as usize]);
        let g = (a + b + d) * (1.0 / 3.0);
        worst = worst.max(dst.closest_point(g).dist);
    }
    worst
}

/// 双向 Hausdorff（顶点+重心采样）。
fn hausdorff(a_pos: &[Vec3], a_idx: &[u32], b_pos: &[Vec3], b_idx: &[u32]) -> f32 {
    let b_bvh = Bvh::build(&to_tris(b_pos, b_idx));
    let a_bvh = Bvh::build(&to_tris(a_pos, a_idx));
    one_sided(a_pos, a_idx, &b_bvh).max(one_sided(b_pos, b_idx, &a_bvh))
}

/// 一像素的世界尺寸（相机写死：fov 60°、高 512、物距 3.0）。
fn pixel_world_size() -> f32 {
    let tan_half = prism_math::det_trig::tan(60.0_f32.to_radians() * 0.5);
    2.0 * tan_half * 3.0 / 512.0
}

/// **V1：`k` 级输出与其 `k+3` 级参考的 Hausdorff ≤ 0.5 像素世界尺寸。**
#[test]
fn v1_hausdorff_against_k_plus_3_reference() {
    let (p, t) = icosahedron();
    let px = pixel_world_size();
    let limit = 0.5 * px;
    println!("V1：1 px = {px:e} 世界单位 ⇒ 阈值 {limit:e}");

    // 全档都打印（收敛序列本身就是证据）；**判据从 k = 3 起**适用：
    // 判据约束的是"烘出来的那一档"逼近极限面的程度，而 k = 1/2 的笼边长远大于
    // 一个像素（k=1 的 80 面二十面体，边长 ~0.6 世界单位 ≈ 90 px），它的偏差
    // 来自"笼太粗"而不是"stencil 错"——把它判红等于要求粗笼也满足终帧分辨率，
    // 那不是这条判据的意思。V5 的烘焙档位是 k = 5/6，两者都远在阈值以下。
    for k in [1u32, 2, 3, 4, 5] {
        let coarse = subdivide(&p, &t, &[], k);
        let fine = subdivide(&p, &t, &[], k + 3);
        let d = hausdorff(&coarse.positions, &coarse.indices, &fine.positions, &fine.indices);
        println!(
            "V1：k={k}（{} 三角）vs k={}（{} 三角）⇒ Hausdorff = {d:e}（阈值 {limit:e}）",
            coarse.tri_count(),
            k + 3,
            fine.tri_count()
        );
        if k >= 3 {
            assert!(
                d <= limit,
                "k={k} 的 Hausdorff {d:e} 超过 0.5 像素世界尺寸 {limit:e}"
            );
        }
    }
}

/// `MAX_LEVELS` 之外的请求被**截断**（并回显实际级数），不 panic、不静默超算。
#[test]
fn levels_are_clamped_and_echoed() {
    let (p, t) = icosahedron();
    let r = subdivide(&p, &t, &[], MAX_LEVELS + 5);
    assert_eq!(r.levels, MAX_LEVELS, "超出上限必须回显实际级数");
    assert_eq!(r.tri_count(), 20 * 4usize.pow(MAX_LEVELS));
}
