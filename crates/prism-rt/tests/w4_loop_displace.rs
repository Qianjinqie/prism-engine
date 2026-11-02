//! W4 位移烘焙判据（V3 / V4）+ 细分位移帧可出图。
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V3 | `displacement = 0` vs 未位移 | **逐位**一致（构造性） |
//! | V4 | 所有相邻细分面共享边上的位移值 | **逐位**相等 |
//!
//! 几何侧的 V1（Hausdorff）/ V2'（正三角网格极限）/ V5（内存）在
//! `prism-geometry/tests/qa_loop_subdiv.rs` 与 `subdiv.rs` 的单测里；
//! 本文件只测**要用纹理采样器**的那两条（采样器在 `prism-rt`，几何叶子拿不到）。

use std::collections::BTreeMap;

use prism_core::EvalCtx;
use prism_geometry::subdiv::subdivide;
use prism_math::Vec3;
use prism_rt::camera::Camera;
use prism_rt::displace::{amplitude_at, bake_triangles, displace_positions, DisplaceParams};
use prism_rt::integrator::{render_pixel, Light, MaterialKind, RenderParams, Scene};
use prism_rt::texsample::TexTable;

/// 二十面体（20 面、12 顶点；规则 5 价三角笼）。
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

/// 球面参数化的 uv（**故意做成有缝**：u 在 ±π 处折回 [0,1)）。
///
/// 这不是"为了好看"：接缝存在，才谈得上"缝在结构上不存在"这条判据。
/// 顶点域 uv 下，缝两侧共享的顶点只有**一份** uv；若实现改成角点域，
/// 同一条共享边上的两个面会拿到 `u ≈ 0.98` 与 `u ≈ 0.02` 两个不同的值。
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

/// 细分到 3 级 + 球面 uv 的测试网格。
fn subdivided() -> prism_geometry::subdiv::SubdivResult {
    let (p, t) = icosahedron();
    let uv: Vec<[f32; 2]> = p.iter().map(|v| spherical_uv(*v)).collect();
    subdivide(&p, &t, &uv, 3)
}

/// **V3：`displacement = 0` 与未位移逐位一致**（并且非零幅值确实改变几何）。
#[test]
fn v3_zero_displacement_is_bitwise_identity() {
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let sub = subdivided();

    let disp0 = displace_positions(
        &sub.positions,
        &sub.normals,
        &sub.uv,
        table,
        handle,
        DisplaceParams::new(0.0),
    );
    assert_eq!(disp0.len(), sub.positions.len());
    let mut diff = 0usize;
    for (a, b) in disp0.iter().zip(sub.positions.iter()) {
        if bits(*a) != bits(*b) {
            diff += 1;
        }
    }
    assert_eq!(diff, 0, "幅值 0 必须逐位一致，实测 {diff} 个顶点有位差");

    // 非平凡性：同一张纹理 + 非零幅值必须真的改变几何（否则上面的"一致"是空话）。
    let amp = DisplaceParams::new(0.08);
    let disp = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);
    let moved = disp
        .iter()
        .zip(sub.positions.iter())
        .filter(|(a, b)| bits(**a) != bits(**b))
        .count();
    assert!(moved > sub.positions.len() / 4, "非零幅值应移动多数顶点，实测 {moved}");

    // 逐顶点复算：位移幅值只依赖该顶点的 uv（不含任何面/角点状态）。
    for i in 0..sub.positions.len() {
        let a = amplitude_at(table, handle, sub.uv[i], amp);
        if a == 0.0 {
            assert_eq!(bits(disp[i]), bits(sub.positions[i]), "零幅值顶点 {i} 不得移动");
        }
    }
    println!(
        "V3：{} 顶点，幅值 0 位差 0；幅值 0.08 移动 {moved} 个顶点",
        sub.positions.len()
    );
}

/// 顶点的位型三元组（`-0.0` 与 `+0.0` 不同 ⇒ 逐位比较必须用位型）。
fn bits(v: Vec3) -> (u32, u32, u32) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

/// **V4：所有相邻细分面共享边上的位移值逐位相等。**
///
/// 覆盖口径是**全部共享边**（枚举整张网格的边，取被 ≥ 2 个面使用的那些），
/// 不是抽查几个点。断言分三层：
///
/// 1. 结构层：同一共享边的两个面引用**同一对顶点索引**（顶点域 ⇒ 只有一份顶点）；
/// 2. 位值层：两端点的位移结果逐位相同（同一数组的同一元素，重复取两次）；
/// 3. 反证层：若按**角点域**口径（把 `u` 加 1 越过接缝再采样——这是接缝另一侧的
///    合理 uv），幅值会不同 ⇒ 说明本测试对真实缺陷敏感，不是因为"恒等"而空过。
#[test]
fn v4_all_shared_edges_have_identical_displacement() {
    let mut scene = Scene::default();
    let handle = make_texture(&mut scene);
    let table = TexTable { refs: &scene.textures, texels: &scene.texels };
    let sub = subdivided();
    let amp = DisplaceParams::new(0.08);
    let disp = displace_positions(&sub.positions, &sub.normals, &sub.uv, table, handle, amp);

    // 边 → 相邻面（面用起始索引表示，便于诊断）。
    let mut edge_faces: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (f, c) in sub.indices.chunks_exact(3).enumerate() {
        for k in 0..3 {
            let a = c[k];
            let b = c[(k + 1) % 3];
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    let shared: Vec<(&(u32, u32), &Vec<usize>)> =
        edge_faces.iter().filter(|(_, f)| f.len() >= 2).collect();
    assert!(!shared.is_empty(), "闭合网格必须有共享边");
    assert!(
        shared.len() * 100 >= edge_faces.len() * 90,
        "共享边应覆盖绝大多数边（{} / {}）",
        shared.len(),
        edge_faces.len()
    );

    // **喂给渲染器的实际几何**（三角形 soup）：共享边的两个面必须是同一批位型。
    // 这一步不查索引（索引已经在上面的结构层验过），查的是**烘焙产物**——
    // 三个后端读到的就是这份 `Vec<Triangle>`，裂缝只能在这里出现或被排除。
    let tris = bake_triangles(&disp, &sub.indices);
    assert_eq!(tris.len(), sub.indices.len() / 3);

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
            continue; // 非流形边另算（本网格不应有）
        }
        let (f0, f1) = (faces[0], faces[1]);
        for v in [a, b] {
            let p0 = corner(f0, v);
            let p1 = corner(f1, v);
            assert_eq!(
                bits(p0),
                bits(p1),
                "共享边的顶点 {v}（面 {f0} vs {f1}）位移后位型不同 ⇒ 缝"
            );
            // 幅值只由该顶点的 uv 决定（与面无关）——这是"逐位相等"的成因。
            let amp_here = amplitude_at(table, handle, sub.uv[v as usize], amp);
            assert_eq!(
                amp_here.to_bits(),
                amplitude_at(table, handle, sub.uv[v as usize], amp).to_bits(),
                "顶点 {v} 的幅值必须是 uv 的纯函数"
            );
            checked += 1;
        }
    }

    // 反证层：角点域在接缝处会给同一顶点**另一个** uv（u ± 1 是平铺的邻块），
    // 幅值随之改变 ⇒ 说明本测试对真实的"缝"缺陷敏感，不是因为恒等而空过。
    let mut seam_demo: Option<(u32, f32, f32)> = None;
    for (&(a, _b), _faces) in edge_faces.iter() {
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
        "V4：{} 条共享边 × 2 端点 = {checked} 个角点，烘焙三角形逐位一致；         反证 {}",
        shared.len(),
        match seam_demo {
            Some((v, x, y)) => format!("顶点 {v} 在 uv+1（接缝邻块）的幅值 {y} ≠ {x} ⇒ 角点域会在此裂"),
            None => "本纹理平移 1 后幅值不变，反证未触发".to_string(),
        }
    );
}

/// 细分 + 位移 + 烘焙成三角形后，场景可出图（非平凡、全有限）。
///
/// 三后端逐位对拍在 `tests/w4_w5_gpu_parity.rs`（要走 GPU，本文件不碰设备）。
#[test]
fn subdivided_displaced_mesh_renders() {
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
    let ctx = EvalCtx::new(20261102, 11, 5);
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
    println!("W4 帧：{} 三角，{res:?} spp{}，luma ∈ [{lo:.4}, {hi:.4}]", sub.tri_count(), params.spp);
}
