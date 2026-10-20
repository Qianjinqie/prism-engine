//! QA 边界反例攻击（几何五项）。
//!
//! 断言以文档契约为准：
//! 1. CSG 20 层深嵌套：深树求值正确且与等价浅构造逐位一致（确定性契约）。
//! 2. BVH 退化输入：空 / 单图元 / 全重叠 / 零面积——doc 契约（bvh.rs
//!    "空输入 → 空树（遍历/查询均安全返回空语义）"）。
//! 3. glTF：JSON 深度边界（64 层内合法 / 超限拒绝）+ 超大 base64 完整解码。
//! 4. OBJ：仅顶点（无 f 行）与正/负索引越界 → 结构化错误不 panic。
//! 5. mesh_sdf 非封闭网格：不 panic + 病态计数 + 契约
//!    （"判定病态按 exterior 处理并计数警告——永不静默"）。

use prism_geometry::bvh::{mesh_triangles, Bvh, Triangle};
use prism_geometry::csg::Csg;
use prism_geometry::gltf::Json;
use prism_geometry::mesh_sdf::{bake_mesh, BakeOptions};
use prism_geometry::primitive::Sphere;
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// ① CSG 20 层深嵌套
// ---------------------------------------------------------------------------

#[test]
fn csg_20_level_deep_nesting_is_correct() {
    // 平移链 20 层 × 0.125（2^-3，精确浮点）：等价于单层平移 (2.5,0,0) 的球。
    let mut tree = Csg::Sphere(Sphere { center: Vec3::splat(0.0), radius: 1.0 });
    for _ in 0..20 {
        tree = Csg::Translate(Box::new(tree), Vec3::new(0.125, 0.0, 0.0));
    }
    let flat = Csg::Sphere(Sphere { center: Vec3::new(2.5, 0.0, 0.0), radius: 1.0 });
    for p in [
        Vec3::new(3.5, 0.0, 0.0),
        Vec3::new(2.5, 0.0, 0.0),
        Vec3::new(1.6, 0.3, -0.7),
        Vec3::new(-2.0, 4.0, 1.0),
    ] {
        let d_tree = tree.dist_lod(p, 4);
        let d_flat = flat.dist_lod(p, 4);
        assert_eq!(
            d_tree.to_bits(),
            d_flat.to_bits(),
            "20 层平移链与单层平移必须逐位一致 at {p:?}: {d_tree} vs {d_flat}"
        );
    }

    // 并集链 20 层（同心球 r = 1.0, 1.05, …, 1.95）：最外球主导。
    let mut chain = Csg::Sphere(Sphere { center: Vec3::splat(0.0), radius: 1.0 });
    for i in 1..20_u32 {
        chain = Csg::Union(
            Box::new(chain),
            Box::new(Csg::Sphere(Sphere { center: Vec3::splat(0.0), radius: 1.0 + 0.05 * i as f32 })),
        );
    }
    let outer = Csg::Sphere(Sphere { center: Vec3::splat(0.0), radius: 1.95 });
    for p in [Vec3::splat(0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.5, 0.5, 0.5)] {
        let a = chain.dist_lod(p, 4);
        let b = outer.dist_lod(p, 4);
        assert_eq!(a.to_bits(), b.to_bits(), "Union 链 min 单调性 at {p:?}: {a} vs {b}");
    }
    assert!(chain.dist_lod(Vec3::splat(0.0), 4) < 0.0, "原点在链内");

    // 深树 1e4 次求值不爆栈（递归深度 20 × 每层常数）。
    let mut acc = 0.0_f32;
    for k in 0..10_000_u32 {
        let p = Vec3::new(k as f32 * 0.001 - 5.0, 0.25, -0.5);
        acc += tree.dist_lod(p, 4);
    }
    assert!(acc.is_finite(), "深树批量求值应有限");
}

// ---------------------------------------------------------------------------
// ② BVH 退化输入
// ---------------------------------------------------------------------------

/// 空输入：doc 契约要求查询安全返回空语义（closest_point / winding_number
/// 已有防护；ray_intersect 亦应安全）。
#[test]
fn bvh_empty_input_safe_queries() {
    let bvh = Bvh::build(&[]);
    // bvh.rs:282 契约："空输入 → 空树（遍历/查询均安全返回空语义）"。
    let hit = bvh.ray_intersect(Vec3::splat(0.0), Vec3::new(0.0, 0.0, -1.0), 0.0, 10.0);
    assert_eq!(hit, None, "空 BVH ray_intersect 应安全返回 None（doc 契约）");
    let cp = bvh.closest_point(Vec3::splat(1.0));
    assert!(cp.dist.is_infinite(), "空 BVH closest_point 应返回无穷距离");
    let w = bvh.winding_number(Vec3::splat(0.0));
    assert_eq!(w, 0.0, "空 BVH winding_number 应为 0");
}

/// 单三角形：命中 / 未命中 / 最近点全部精确。
#[test]
fn bvh_single_triangle_queries() {
    let tri = Triangle {
        a: Vec3::new(0.0, 0.0, 0.0),
        b: Vec3::new(1.0, 0.0, 0.0),
        c: Vec3::new(0.0, 1.0, 0.0),
    };
    let bvh = Bvh::build(&[tri]);
    // 垂直命中 (0.25, 0.25, 1) → t = 1。
    let hit = bvh.ray_intersect(
        Vec3::new(0.25, 0.25, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        0.0,
        10.0,
    );
    let hit = hit.expect("单三角形必须命中");
    assert_eq!(hit.tri, 0);
    assert!((hit.t - 1.0).abs() < 1e-6, "命中 t={} 应为 1", hit.t);
    // 错开射线：未命中。
    let miss = bvh.ray_intersect(
        Vec3::new(0.9, 0.9, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        0.0,
        10.0,
    );
    assert_eq!(miss, None, "三角形外部射线不得命中");
    // 最近点：面内投影。
    let cp = bvh.closest_point(Vec3::new(0.25, 0.25, 5.0));
    assert!((cp.dist - 5.0).abs() < 1e-5, "最近点距离 {} 应为 5", cp.dist);
    assert!((cp.point.x - 0.25).abs() < 1e-5 && (cp.point.y - 0.25).abs() < 1e-5);
    // winding：单三角形不在任何点形成闭合 → |w| < 1/4（单三角形最大实体角比）。
    let w = bvh.winding_number(Vec3::new(0.0, 0.0, -1.0));
    assert!(w.abs() < 0.25, "单开三角形 winding |{w}| 应远小于 1");
}

/// 100 个全重叠三角形：构建确定 + t 并列按最小原图元索引 tie-break。
#[test]
fn bvh_fully_overlapping_primitives_deterministic() {
    let tri = Triangle {
        a: Vec3::new(0.0, 0.0, 0.0),
        b: Vec3::new(1.0, 0.0, 0.0),
        c: Vec3::new(0.0, 1.0, 0.0),
    };
    let tris: Vec<Triangle> = vec![tri; 100];
    let b1 = Bvh::build(&tris);
    let b2 = Bvh::build(&tris);
    assert_eq!(b1, b2, "同输入两次构建必须同树（确定性契约）");
    let h1 = b1.ray_intersect(Vec3::new(0.3, 0.3, 2.0), Vec3::new(0.0, 0.0, -1.0), 0.0, 10.0);
    let h2 = b2.ray_intersect(Vec3::new(0.3, 0.3, 2.0), Vec3::new(0.0, 0.0, -1.0), 0.0, 10.0);
    assert_eq!(h1, h2, "重叠图元命中必须确定");
    let hit = h1.expect("重叠三角形应命中");
    assert_eq!(hit.tri, 0, "t 并列必须 tie-break 到最小原图元索引");
    // 最近点同样确定。
    assert_eq!(b1.closest_point(Vec3::new(0.3, 0.3, 5.0)), b2.closest_point(Vec3::new(0.3, 0.3, 5.0)));
}

/// 零面积（共线）三角形：构建与查询不得 panic / NaN。
#[test]
fn bvh_degenerate_zero_area_triangle() {
    let tri = Triangle {
        a: Vec3::new(0.0, 0.0, 0.0),
        b: Vec3::new(1.0, 0.0, 0.0),
        c: Vec3::new(2.0, 0.0, 0.0),
    };
    let bvh = Bvh::build(&[tri]);
    let hit = bvh.ray_intersect(Vec3::new(0.5, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0), 0.0, 10.0);
    if let Some(h) = hit {
        assert!(h.t.is_finite(), "零面积三角形命中 t 必须有限");
    }
    let cp = bvh.closest_point(Vec3::new(0.0, 3.0, 0.0));
    assert!(cp.dist.is_finite() && cp.dist >= 0.0, "零面积最近点距离必须有限非负");
    let w = bvh.winding_number(Vec3::splat(0.0));
    assert!(w.is_finite(), "零面积 winding 不得 NaN");
}

// ---------------------------------------------------------------------------
// ③ glTF：JSON 深度边界 + 超大 base64
// ---------------------------------------------------------------------------

#[test]
fn gltf_json_depth_boundary() {
    // 64 是实现声明上限（"嵌套过深（>64 层）"）→ 65 层合法、66 层拒绝。
    let d65 = "[".repeat(65) + &"]".repeat(65);
    assert!(Json::parse(&d65).is_ok(), "65 层应在上限内合法");
    let d66 = "[".repeat(66) + &"]".repeat(66);
    let e = Json::parse(&d66).expect_err("66 层必须拒绝");
    assert!(e.msg.contains("嵌套"), "拒绝信息应说明嵌套过深：{}", e.msg);
}

#[test]
fn gltf_large_base64_buffer_decodes_completely() {
    // ≈1 MiB buffer（87381 顶点 × 12 字节 = 1048572 字节；87381 = 3 × 29127）。
    const COUNT: usize = 87_381;
    let mut bin = vec![0_u8; COUNT * 12];
    let put = |i: usize, v: [f32; 3], buf: &mut [u8]| {
        for (k, c) in v.iter().enumerate() {
            let b = (i * 3 + k) * 4;
            buf[b..b + 4].copy_from_slice(&c.to_le_bytes());
        }
    };
    put(0, [1.0, 2.0, 3.0], &mut bin);
    put(COUNT - 1, [4.0, 5.0, 6.0], &mut bin);
    let uri = format!("data:application/octet-stream;base64,{}", b64(&bin));
    let json = format!(
        r#"{{
            "asset": {{"version": "2.0"}},
            "scene": 0,
            "scenes": [{{"nodes": [0]}}],
            "nodes": [{{"mesh": 0}}],
            "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}}}]}}],
            "accessors": [{{"bufferView": 0, "componentType": 5126, "count": {COUNT}, "type": "VEC3"}}],
            "bufferViews": [{{"buffer": 0, "byteLength": {}}}],
            "buffers": [{{"byteLength": {}, "uri": "{uri}"}}]
        }}"#,
        COUNT * 12,
        COUNT * 12
    );
    let (meshes, warn) = prism_geometry::gltf::parse_gltf(&json, &[]).expect("大 buffer 应完整解码");
    assert!(warn.is_empty());
    assert_eq!(meshes.len(), 1);
    let m = &meshes[0];
    assert_eq!(m.positions.len(), COUNT, "顶点数必须完整");
    assert_eq!(m.positions[0], Vec3::new(1.0, 2.0, 3.0), "首顶点字节正确");
    assert_eq!(m.positions[COUNT - 1], Vec3::new(4.0, 5.0, 6.0), "末顶点字节正确");
    assert_eq!(m.indices, (0..COUNT as u32).collect::<Vec<_>>(), "非索引按顺序三角化");
}

/// 测试内 base64 编码器（标准表 + '=' padding）。
fn b64(data: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let w = (b0 << 16) | (b1 << 8) | b2;
        out.push(TBL[(w >> 18) as usize & 63] as char);
        out.push(TBL[(w >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TBL[(w >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TBL[w as usize & 63] as char } else { '=' });
    }
    out
}

// ---------------------------------------------------------------------------
// ④ OBJ 边界
// ---------------------------------------------------------------------------

#[test]
fn obj_vertices_only_rejected() {
    let e = prism_geometry::obj::parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n").expect_err("无 f 行必须拒绝");
    assert!(e.msg.contains("没有任何面"), "错误应说明缺面数据：{e}");
    assert!(e.to_string().contains("建议"), "三段式错误必须含建议段（宪章 C19）");
}

#[test]
fn obj_out_of_range_indices_rejected() {
    let verts = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
    // 正索引越界。
    let e = prism_geometry::obj::parse_obj(&format!("{verts}f 1 2 9\n")).expect_err("正越界");
    assert!(e.msg.contains("越界"), "{e}");
    // 负相对索引越界（-4 → 3-4 = -1）。
    let e = prism_geometry::obj::parse_obj(&format!("{verts}f -4 -2 -1\n")).expect_err("负越界");
    assert!(e.msg.contains("越界"), "{e}");
    // 索引 0 非法。
    let e = prism_geometry::obj::parse_obj(&format!("{verts}f 0 1 2\n")).expect_err("索引 0");
    assert!(e.msg.contains("0 非法"), "{e}");
    // vn 引用越界（v//vn 格式）。
    let src = format!("{verts}vn 0 0 1\nf 1//5 2//1 3//1\n");
    let e = prism_geometry::obj::parse_obj(&src).expect_err("vn 越界");
    assert!(e.msg.contains("越界"), "{e}");
    // 负索引恰好合法（-3 -2 -1 = 1 2 3）必须通过——反向验证拒绝逻辑不误伤。
    let (meshes, _) = prism_geometry::obj::parse_obj(&format!("{verts}f -3 -2 -1\n")).expect("合法负索引");
    assert_eq!(meshes[0].indices, vec![0, 1, 2]);
}

// ---------------------------------------------------------------------------
// ⑤ mesh_sdf 非封闭网格（病态契约）
// ---------------------------------------------------------------------------

/// 5 面开放盒：立方体 8 顶点、缺 +z 顶盖，5 面 × 2 三角形，全部外向。
fn open_box_mesh() -> (Vec<Vec3>, Vec<u32>) {
    let positions = vec![
        Vec3::new(-1.0, -1.0, -1.0), // 0
        Vec3::new(1.0, -1.0, -1.0),  // 1
        Vec3::new(1.0, 1.0, -1.0),   // 2
        Vec3::new(-1.0, 1.0, -1.0),  // 3
        Vec3::new(-1.0, -1.0, 1.0),  // 4
        Vec3::new(1.0, -1.0, 1.0),   // 5
        Vec3::new(1.0, 1.0, 1.0),    // 6
        Vec3::new(-1.0, 1.0, 1.0),   // 7
    ];
    // 外向三角形（(b−a)×(c−a) 指向盒外）：
    // 底 z=-1；前 y=-1；后 y=+1；右 x=+1；左 x=-1。顶盖缺失。
    let indices = vec![
        0, 2, 1, 0, 3, 2, // 底
        0, 1, 5, 0, 5, 4, // 前
        2, 7, 6, 2, 3, 7, // 后
        1, 6, 5, 1, 2, 6, // 右
        0, 4, 7, 0, 7, 3, // 左
    ];
    (positions, indices)
}

#[test]
fn mesh_sdf_open_mesh_pathology_contract() {
    let (positions, indices) = open_box_mesh();
    let opts = BakeOptions { spacing: 0.1, pad_voxels: 2 };
    let (baked, report) = bake_mesh(&positions, &indices, &opts);

    // ① 烘焙本身不 panic（已到此处）。
    // ② 开放面必然产生病态 winding（既非 0/1 也非过渡带）→ 必须被计数。
    assert!(
        report.degenerate_winding > 0,
        "5 面开放盒必须检出病态 winding 计数（永不静默契约），实际 {}",
        report.degenerate_winding
    );

    // ③ 契约："判定病态（w 明显偏离 0/1/表面过渡带）按 exterior 处理
    //    并计数警告——永不静默"。即：被判内部（负距离）的体素其 winding 必须
    //    可靠（> 0.95）；w 落在病态带 (0.55, 0.95) 的体素不得判为内部。
    let mut pathological_interior = 0_usize;
    for (bx, by, bz) in baked.bricks.keys().copied().map(prism_geometry::mesh_sdf::unpack_brick) {
        let cell = &baked.bricks[&prism_geometry::mesh_sdf::pack_brick(bx, by, bz)];
        for i in 0..prism_geometry::mesh_sdf::BRICK {
            for j in 0..prism_geometry::mesh_sdf::BRICK {
                for k in 0..prism_geometry::mesh_sdf::BRICK {
                    if cell[(i * prism_geometry::mesh_sdf::BRICK + j) * prism_geometry::mesh_sdf::BRICK + k] < 0.0 {
                        let p = Vec3::new(
                            baked.origin.x + (bx * 8 + i as u64) as f32 * baked.spacing,
                            baked.origin.y + (by * 8 + j as u64) as f32 * baked.spacing,
                            baked.origin.z + (bz * 8 + k as u64) as f32 * baked.spacing,
                        );
                        let w = baked.bvh.winding_number(p);
                        if w > 0.55 && w < 0.95 {
                            pathological_interior += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        pathological_interior, 0,
        "病态 winding 体素被判为内部（负距离）——违反'病态按 exterior 处理' 契约"
    );

    // ④ 内部远场点：若判内部（负距离），其 winding 必须可靠（>0.95）。
    let d = baked.dist(Vec3::splat(0.0));
    if d < 0.0 {
        let w = baked.bvh.winding_number(Vec3::splat(0.0));
        assert!(
            w > 0.95,
            "点 (0,0,0) 判内部（dist={d}）但 winding={w} 落在病态带——符号判定未按 exterior 契约兜底"
        );
    }
}

/// 封闭网格回归：病态计数为 0，内部为负（防止开放网格处理伤及正常路径）。
#[test]
fn mesh_sdf_closed_mesh_still_exact() {
    let (positions, indices) = open_box_mesh();
    // 补上顶盖 → 6 面封闭盒。
    let mut idx = indices;
    idx.extend_from_slice(&[6, 7, 4, 6, 4, 5]); // 顶 z=+1 外向
    let opts = BakeOptions { spacing: 0.1, pad_voxels: 2 };
    let (baked, report) = bake_mesh(&positions, &idx, &opts);
    assert_eq!(report.degenerate_winding, 0, "封闭盒不得有病态 winding");
    assert!(baked.dist(Vec3::splat(0.0)) < 0.0, "封闭盒中心必须判内部");
    assert!(baked.dist(Vec3::new(3.0, 0.0, 0.0)) > 0.0, "盒外必须判外部");
    // 解析对照：中心到面距离 = 1（±1 半宽）→ 体素尺度容差内。
    let d = baked.dist(Vec3::new(0.0, 0.0, 0.9));
    assert!((d.abs() - 0.1).abs() <= 0.1 + 1e-5, "近面点距离 |{d}|≈0.1（体素尺度容差）");
    let _ = mesh_triangles(&positions, &idx).len(); // 触碰公共 API 编译锚
}
