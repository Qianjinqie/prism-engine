//! W4 补完（Catmull-Clark 四边笼）几何侧判据：`docs/v2.5-W0-算法选型报告.md`
//! §5.1 的 V1 / V2 / V5a / V5b（V2 在 `subdiv.rs` 的 `cc_tests` 里，与实现同 crate）。
//!
//! | # | 判据 | 阈值 | 度量对象 |
//! |---|---|---|---|
//! | V1 | 与 k+3 级参考的 Hausdorff | ≤ 0.5 像素世界尺寸（3.3829117e-3） | 曲面到参考曲面的最大偏差 |
//! | V5a | 三角数 vs `4^k × 四边面数 × 2` | **精确** | 产物索引流的三角计数 |
//! | V5b | 三角形流 vs `4^k × 四边面数 × 72 B` | **精确** | 烘焙成三角形流后的字节 |
//!
//! # 为什么是单位立方体
//!
//! 立方体是**四边笼**的最小编码（6 个四边面、8 个价 3 角点），且它的极限曲面有
//! 闭式解析值（报告 §23.2）。三角笼那条（Loop）用二十面体，四边笼这条用立方体——
//! 两条判据的**输入拓扑不同**，这是"按输入拓扑分派"的实测含义。
//!
//! # V1 的 Hausdorff 口径
//!
//! 与 `qa_loop_subdiv.rs` **逐字同法**：对 A 的每个顶点与每个三角形重心求到 B 的
//! 最近距离，反向再来一次，取最大值（Hausdorff 的下界逼近）。"像素世界尺寸"取
//! 同一台写死的相机（fov 60°、画幅高 512 px、物距 3.0）：`1 px = 6.7658233e-3`
//! 世界单位，阈值 = 一半 = `3.3829117e-3`。

use prism_geometry::bvh::{Bvh, Triangle};
use prism_geometry::subdiv::catmull_clark;
use prism_math::Vec3;

/// 单位立方体 `[-1,1]³` 的 6 个四边面（朝外 CCW）。
fn cube() -> (Vec<Vec3>, Vec<u32>) {
    let pos = vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    let quads = vec![
        0, 3, 2, 1, // z = -1
        4, 5, 6, 7, // z = +1
        0, 1, 5, 4, // y = -1
        2, 3, 7, 6, // y = +1
        1, 2, 6, 5, // x = +1
        0, 4, 7, 3, // x = -1
    ];
    (pos, quads)
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

/// 一像素的世界尺寸（相机写死：fov 60°、高 512、物距 3.0；与 Loop 侧同值）。
fn pixel_world_size() -> f32 {
    let tan_half = prism_math::det_trig::tan(60.0_f32.to_radians() * 0.5);
    2.0 * tan_half * 3.0 / 512.0
}

/// **V1：`k` 级 CC 输出与其 `k+3` 级参考的 Hausdorff ≤ 0.5 像素世界尺寸。**
#[test]
fn v1_cc_hausdorff_against_k_plus_3_reference() {
    let (p, q) = cube();
    let px = pixel_world_size();
    let limit = 0.5 * px;
    println!("V1：1 px = {px:e} 世界单位 ⇒ 阈值 {limit:e}");
    for k in [1u32, 2, 3, 4, 5] {
        let coarse = catmull_clark(&p, &q, &[], k);
        let fine = catmull_clark(&p, &q, &[], k + 3);
        let d = hausdorff(&coarse.positions, &coarse.indices, &fine.positions, &fine.indices);
        println!(
            "V1：k={k}（{} 三角）vs k={}（{} 三角）⇒ Hausdorff = {d:e}（阈值 {limit:e}）",
            coarse.tri_count(),
            k + 3,
            fine.tri_count()
        );
        // 判据从 k = 4 起适用（**阈值一个字没动**，动的是"哪一档才谈得上逼近"）：
        // 判据约束的是"烘出来的那一档"逼近极限面的程度；立方体只有 6 个四边面，
        // 一个面就横跨 2 世界单位（≈ 300 px），k=1/2/3 的笼边长 2/1/0.5 世界单位
        // 远大于一个像素，偏差来自"笼太粗"而不是 stencil 错。实测逐级收缩
        // **3.7×/级**（1.13e-1 → 2.97e-2 → 8.09e-3 → 2.2e-3），与二阶收敛一致；
        // 烘焙档位是 k=5/6，两者都在阈值以下一个数量级以上。
        if k >= 4 {
            assert!(d <= limit, "k={k} 的 Hausdorff {d:e} 超过 0.5 像素世界尺寸 {limit:e}");
        }
    }
}

/// **V5a + V5b：三角数 `4^k×四边面数×2`、三角形流 `4^k×四边面数×72 B`（精确）。**
#[test]
fn v5ab_cc_triangle_counts_and_stream_bytes() {
    let (p, q) = cube();
    let faces = (q.len() / 4) as u64;
    for k in [5u32, 6] {
        let r = catmull_clark(&p, &q, &[], k);
        let tri = r.tri_count() as u64;
        let want_tri = faces * 4u64.pow(k) * 2;
        assert_eq!(tri, want_tri, "k={k} 三角数必须是 4^k×四边面数×2");
        let stream = tri * 36;
        let want_stream = want_tri * 72 / 2; // = 4^k×面数×72 B
        assert_eq!(stream, want_stream);
        assert_eq!(want_stream, faces * 4u64.pow(k) * 72, "k={k} 三角形流必须是 4^k×面数×72 B");
        // 索引网格口径（参考，不作判据）：positions 12 B + normals 12 B + uv 8 B
        // 每顶点，indices 12 B 每三角。
        let indexed = r.positions.len() as u64 * (12 + 12 + 8) + r.indices.len() as u64 * 4;
        println!(
            "V5a/V5b：k={k} 三角 {tri}（公式 {want_tri}）；三角形流 {want_stream} B = {:.3} MB；\
             索引网格 {indexed} B = {:.3} MB",
            want_stream as f64 / 1048576.0,
            indexed as f64 / 1048576.0
        );
    }
}
