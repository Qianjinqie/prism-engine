//! W4 补完（Catmull-Clark）：位移烘焙判据（V3 / V4）+ 位移后帧可出图。
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V3 | `displacement = 0` vs 未位移 | **逐位**一致（构造性） |
//! | V4 | CC 产物所有相邻细分面共享边上的位移值 | **逐位**相等（顶点域 uv） |
//!
//! 几何侧的 V1 / V2 / V5a / V5b 在 `prism-geometry`（`tests/qa_catmull_clark.rs`
//! 与 `subdiv.rs` 的 `cc_tests`）；V5c（峰值 RSS）在 `tests/qa_cc_memory.rs`。
//! 本文件只测**要用纹理采样器**的那两条（采样器在 `prism-rt`，几何叶子拿不到）。
//!
//! # 为什么四边笼用立方体
//!
//! 立方体是四边笼的最小编码（6 个四边面、8 个价 3 角点），且 CC 把它抹成一张
//! 光滑闭曲面——共享边覆盖 100%，V4 的"全部共享边"才有意义。

use std::collections::BTreeMap;

use prism_core::EvalCtx;
use prism_geometry::subdiv::catmull_clark;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::displace::{amplitude_at, bake_triangles, displace_positions, DisplaceParams};
use prism_rt::integrator::{render_pixel, Light, MaterialKind, RenderParams, Scene};
use prism_rt::texsample::TexTable;

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

/// 球面式 uv（**故意做成有缝**：`u` 在 ±π 处折回 [0,1)）。
///
/// 与 Loop 侧同口径：接缝存在，才谈得上"缝在结构上不存在"这条判据。
fn spherical_uv(p: Vec3) -> [f32; 2] {
    let u = 0.5 + 0.5 * (p.z / p.x.abs().max(1e-6)).atan() / std::f32::consts::PI;
    let v = 0.5 + 0.5 * p.y;
    [u, v]
}

/// 一张 16×16 的线性域纹理（含黑区，保证"幅值 0 的顶点"也出现在样本里）。
fn make_texture(scene: &mut Scene) -> u32 {
    let mut px = Vec::with_capacity(16 * 16);
    for y in 0..16u32 {
        for x in 0..16u32 {
            let on = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.0 };
            px.push([on, on, on, 1.0]);
        }
    }
    scene.add_texture_linear(16, 16, &px)
}

/// CC 细分到 3 级 + 球面 uv 的测试网格。
fn subdivided() -> prism_geometry::subdiv::SubdivResult {
    let (p, q) = cube();
    let uv: Vec<[f32; 2]> = p.iter().map(|v| spherical_uv(*v)).collect();
    catmull_clark(&p, &q, &uv, 3)
}

/// 顶点的位型三元组（`-0.0` 与 `+0.0` 不同 ⇒ 逐位比较必须用位型）。
fn bits(v: Vec3) -> (u32, u32, u32) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

/// **V3：`displacement = 0` 与未位移逐位一致**（并且非零幅值确实改变几何）。
#[test]
fn v3_cc_zero_displacement_is_bitwise_identity() {
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let sub = subdivided();
    assert!(sub.tri_count() > 100, "3 级 CC 应有足够三角形：{}", sub.tri_count());

    let disp0 = displace_positions(
        &sub.positions,
        &sub.normals,
        &sub.uv,
        table,
        handle,
        DisplaceParams::new(0.0),
    );
    assert_eq!(disp0.len(), sub.positions.len());
    let diff = disp0.iter().zip(sub.positions.iter()).filter(|(a, b)| bits(**a) != bits(**b)).count();
    assert_eq!(diff, 0, "幅值 0 必须逐位一致，实测 {diff} 个顶点有位差");

    // 非平凡性：同一张纹理 + 非零幅值必须真的改变几何。
    let amp = DisplaceParams::new(0.08);
    let disp = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);
    let moved = disp
        .iter()
        .zip(sub.positions.iter())
        .filter(|(a, b)| bits(**a) != bits(**b))
        .count();
    assert!(moved > sub.positions.len() / 4, "非零幅值应移动多数顶点，实测 {moved}");

    for i in 0..sub.positions.len() {
        let a = amplitude_at(table, handle, sub.uv[i], amp);
        if a == 0.0 {
            assert_eq!(bits(disp[i]), bits(sub.positions[i]), "零幅值顶点 {i} 不得移动");
        }
    }
    println!(
        "V3（CC）：{} 顶点，幅值 0 位差 0；幅值 0.08 移动 {moved} 个顶点",
        sub.positions.len()
    );
}

/// **V4：CC 产物所有相邻细分面共享边上的位移值逐位相等。**
///
/// 覆盖口径是**全部共享边**（枚举整张网格的边，取被 ≥ 2 个面使用的那些）：
/// 结构层（同一共享边两侧引用同一对顶点索引）→ 位值层（烘焙三角形里两端点位型
/// 相同）→ 反证层（角点域在接缝处会给同一顶点另一个 uv，幅值随之改变）。
#[test]
fn v4_cc_all_shared_edges_have_identical_displacement() {
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let sub = subdivided();
    let amp = DisplaceParams::new(0.08);
    let disp = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);

    let mut edge_faces: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (f, c) in sub.indices.chunks_exact(3).enumerate() {
        for k in 0..3 {
            let a = c[k];
            let b = c[(k + 1) % 3];
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let shared = edge_faces.values().filter(|f| f.len() >= 2).count();
    assert!(!edge_faces.is_empty(), "闭合网格必须有边");
    assert!(
        shared * 100 >= edge_faces.len() * 90,
        "共享边应覆盖绝大多数边（{shared} / {}）",
        edge_faces.len()
    );

    // 烘焙产物（三个后端读到的就是这份 `Vec<Triangle>`）。
    let tris = bake_triangles(&disp, &sub.indices);
    assert_eq!(tris.len(), sub.tri_count());
    let corner = |f: usize, v: u32| -> Vec3 {
        let base = f * 3;
        for k in 0..3 {
            if sub.indices[base + k] == v {
                let t = tris[f];
                return match k {
                    0 => t.a,
                    1 => t.b,
                    _ => t.c,
                };
            }
        }
        panic!("面 {f} 不含顶点 {v}");
    };

    let mut checked = 0usize;
    for (&(a, b), faces) in edge_faces.iter() {
        if faces.len() != 2 {
            continue;
        }
        let (f0, f1) = (faces[0], faces[1]);
        for v in [a, b] {
            assert_eq!(
                bits(corner(f0, v)),
                bits(corner(f1, v)),
                "共享边的顶点 {v}（面 {f0} vs {f1}）位移后位型不同 ⇒ 缝"
            );
            let amp_here = amplitude_at(table, handle, sub.uv[v as usize], amp);
            // 位移是**逐顶点**的：`disp[v]` 必须逐位等于 `p[v] + n[v]·amp(uv[v])`
            // （幅值 0 走短路 ⇒ 原样保留）。任何"逐角点各带一份 uv"的写法都会
            // 在这一步暴露出来——两个邻面会各算各的。
            let want = if amp_here == 0.0 {
                sub.positions[v as usize]
            } else {
                sub.positions[v as usize] + sub.normals[v as usize] * amp_here
            };
            assert_eq!(
                bits(disp[v as usize]),
                bits(want),
                "顶点 {v} 的位移必须只由它自己的 uv 决定（面 {f0} / {f1} 不得各算一份）"
            );
            checked += 1;
        }
    }

    // 反证层：角点域在接缝处会给同一顶点另一个 uv（u ± 1 是平铺的邻块）。
    let mut seam_demo: Option<(u32, f32, f32)> = None;
    for &(a, _b) in edge_faces.keys() {
        let u0 = sub.uv[a as usize];
        let here = amplitude_at(table, handle, u0, amp);
        for other in [[u0[0] + 1.0, u0[1]], [u0[0], u0[1] + 1.0]] {
            let there = amplitude_at(table, handle, other, amp);
            if there.to_bits() != here.to_bits() && seam_demo.is_none() {
                seam_demo = Some((a, here, there));
            }
        }
    }
    println!(
        "V4（CC）：{shared} 条共享边 × 2 端点 = {checked} 个角点，烘焙三角形逐位一致；反证 {}",
        match seam_demo {
            Some((v, x, y)) => format!("顶点 {v} 在 uv+1（接缝邻块）的幅值 {y} ≠ {x} ⇒ 角点域会在此裂"),
            None => "repeat 采样下 uv+1 落同一纹素；本组 uv 连 fx 的 1 ULP 都没差 ⇒ \
                     反向不可见（正向的「同顶点同位移」仍成立；Loop 侧 §22 触发过 1 ULP）"
                .to_string(),
        }
    );
}

/// CC + 位移 + 烘焙成三角形后，场景可出图（非平凡、全有限）。
///
/// 三后端逐位对拍在 `tests/w4_w5_gpu_parity.rs`（要走 GPU，本文件不碰设备）。
#[test]
fn cc_displaced_mesh_renders() {
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let sub = subdivided();
    let amp = DisplaceParams::new(0.12);
    let pos = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);
    let tris = bake_triangles(&pos, &sub.indices);
    assert_eq!(tris.len(), sub.tri_count());
    scene.add_bvh(tris, MaterialKind::Lambert { albedo: Vec3::splat(0.8) });
    scene.lights.push(Light::Rect {
        center: Vec3::new(0.0, 3.0, 1.5),
        edge_u: Vec3::new(2.0, 0.0, 0.0),
        edge_v: Vec3::new(0.0, 0.0, 2.0),
        radiance: Vec3::new(14.0, 12.0, 10.0),
    });
    scene.lights.push(Light::Ambient { radiance: Vec3::new(0.06, 0.08, 0.11) });
    let cam = Camera::new(
        Vec3::new(0.0, 0.6, 3.4),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        45.0,
        0.0,
        4.0,
        0.0,
        1.0 / 24.0,
    );
    let ctx = EvalCtx::new(20261103, 11, 5);
    let res = (16u32, 16u32);
    let params = RenderParams::new(8, 4);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for y in 0..res.1 {
        for x in 0..res.0 {
            let px = render_pixel(&scene, &cam, &ctx, (x, y), res, &params);
            for c in px {
                assert!(c.is_finite(), "像素 ({x},{y}) 非有限");
            }
            let l = px[0] + px[1] + px[2];
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    assert!(hi > 0.01, "场景应被照亮：max luma = {hi}");
    assert!(hi - lo > 1e-3, "图像应非平凡（应有明暗差异）");
    println!("W4（CC）帧：{} 三角，{res:?} spp{}，luma ∈ [{lo:.4}, {hi:.4}]", sub.tri_count(), params.spp);
}
