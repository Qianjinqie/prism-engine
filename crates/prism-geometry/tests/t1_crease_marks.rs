//! T1 / G-9 判据：细分曲面的 **crease（折痕）与 corner（角点）标记**。
//!
//! 对应任务原文的 V1–V4，逐条钉死：
//!
//! | # | 判据 | 阈值 |
//! |---|---|---|
//! | V1 | `None` / 全 0 权重 ⇒ 与 `93e59d0` **逐位相同** | 位型全等（不是 1e-6） |
//! | V2 | 立方体笼全 crease ⇒ 极限位置 = 原 8 角点，体积不缩水 | 误差 ≤ 1e-6 |
//! | V3 | 单条 crease 链 ⇒ 折痕顶点极限落在 1D 曲线极限上（解析值），邻域不受影响 | ≤ 1e-6 / 逐位 |
//! | V4 | 错规则必须 FAIL（≥ 2 个反例） | 反例误差 ≫ 1e-6 |
//!
//! # 为什么 V1 用位型而不是容差
//!
//! "加了标记但不用" 必须是**同一个**浮点表达式的结果，不能是"差得很少"。
//! 若实现偷偷把光滑规则换成 `(1-nβ)v + βΣnb` 的等价变形（例如先算 `β·Σnb`
//! 再累加），求和顺序一变末位就变——用容差判据会把这类回归放过去。故逐位。
//!
//! # V2 的"原 8 角点"是哪 8 个
//!
//! 立方体笼 `[-1,1]³`。细分后老顶点保持原下标（本模块的定稿口径：
//! "顶点 i 的后代仍是 i"），故 `positions[i]`（`i < 8`）= 原顶点 `i` 的极限。
//! 全 crease 时角点被 `k ≥ 3` 判成 corner ⇒ 不动 ⇒ 极限 = 自身。若规则写错
//! （例如 `w == 1` 只锐化一级），第二级就重新变圆，误差回弹到 1e-1 量级。
//!
//! # V3 的解析值怎么来的
//!
//! 正二十面体上把 `0–11`、`0–5` 两条边标成 `w = 1`。顶点 `0` 恰有 2 条 crease
//! 边 ⇒ 退化成 1D 三次 B 样条的控制点，其**极限**是 `(b₀ + 4v + b₁)/6`
//! （与边界曲线的极限 stencil 同一支）。这个值不依赖细分级数——B 样条细分
//! 是曲线不变的，故 levels = 0/1/2/3 必须给出**同一个点**（差 ≤ 1e-6）。
//! 相邻顶点（含两条 crease 的另一个端点 `11`，它是 **dart**）走光滑规则，
//! 在 levels = 0 时与无标记运行**逐位相同**。

use prism_geometry::subdiv::{
    catmull_clark, catmull_clark_marked, subdivide, subdivide_marked, SubdivMarks,
};
use prism_math::Vec3;

// ---------------------------------------------------------------------------
// 笼
// ---------------------------------------------------------------------------

/// 单位立方体 `[-1, 1]³` 的 6 个四边面（全部朝外 CCW）。
///
/// 角点编号：`0..4` 是 `z = -1` 层（`(-1,-1,-1)`、`(1,-1,-1)`、`(1,1,-1)`、
/// `(-1,1,-1)`），`4..8` 是 `z = +1` 层（同序）。
fn cube_quads() -> (Vec<Vec3>, Vec<u32>) {
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

/// 同一个立方体，每个四边面沿 `(k0, k2)` 对角线裂成 2 个三角形。
///
/// 对角线是**面内**的边（被该面的两个三角形共用），不会跨面 ⇒ 网格仍然闭合。
fn cube_tris() -> (Vec<Vec3>, Vec<u32>) {
    let (pos, quads) = cube_quads();
    let mut tri = Vec::with_capacity(quads.len() / 4 * 6);
    for q in quads.chunks_exact(4) {
        tri.extend_from_slice(&[q[0], q[1], q[2]]);
        tri.extend_from_slice(&[q[0], q[2], q[3]]);
    }
    (pos, tri)
}

/// 正二十面体（12 顶点、20 面，规则 5 价，闭合）。
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

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

/// 索引流里的**全部唯一边**（`(min, max)` 升序、去重）。
fn unique_edges(idx: &[u32], stride: usize) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for c in idx.chunks_exact(stride) {
        for k in 0..stride {
            let (a, b) = (c[k], c[(k + 1) % stride]);
            out.push((a.min(b), a.max(b)));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 全笼标 `w` 的 crease（corner 另加）。
fn all_crease(idx: &[u32], stride: usize, w: f32) -> SubdivMarks {
    let mut m = SubdivMarks::new();
    for (a, b) in unique_edges(idx, stride) {
        m = m.with_crease(a, b, w);
    }
    m
}

/// 位型三元组（`Vec3` 的逐位比较用；`==` 会把 `-0.0 == 0.0` 判真）。
fn vbits(v: Vec3) -> (u32, u32, u32) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

/// 位型序列（整个 `positions` 逐位）。
fn bits_of(pos: &[Vec3]) -> Vec<(u32, u32, u32)> {
    pos.iter().map(|p| vbits(*p)).collect()
}

/// 闭合网格的体积（散度定理，`f64` 累加）。
fn mesh_volume(pos: &[Vec3], idx: &[u32]) -> f64 {
    let mut sum = 0.0f64;
    for c in idx.chunks_exact(3) {
        let a = pos[c[0] as usize];
        let b = pos[c[1] as usize];
        let d = pos[c[2] as usize];
        let cr = b.cross(&d);
        sum += a.dot(&cr) as f64;
    }
    (sum / 6.0).abs()
}

/// 8 个角点的最大位置误差（`marked` 的极限 vs 原始立方体角点）。
fn corner_error(lim: &[Vec3], orig: &[Vec3]) -> f32 {
    let mut worst = 0.0f32;
    for i in 0..8 {
        let d = (lim[i] - orig[i]).length();
        if d > worst {
            worst = d;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// V1：无标记 ⇒ 逐位相同
// ---------------------------------------------------------------------------

/// **V1（Loop）：`None` / 空标记 / 全 0 权重 ⇒ 与 `subdivide` 逐位相同。**
///
/// 复用既有 V1 用例的规模：二十面体 k=3 ⇒ 642 顶点、1280 三角
/// （`12 + 30 + 120 + 480 = 642`，`F₃ = 20·4³ = 1280`）。顶点数本身就是
/// "拓扑没变"的证据——若标记通路偷偷多生成/少生成一条边，这里立刻红。
#[test]
fn v1_loop_none_and_zero_marks_are_bitwise_identical() {
    let (p, t) = icosahedron();
    let zero_mark = all_crease(&t, 3, 0.0);
    let base = subdivide(&p, &t, &[], 3);
    let none = subdivide_marked(&p, &t, &[], 3, None);
    let zero = subdivide_marked(&p, &t, &[], 3, Some(&zero_mark));
    let empty = subdivide_marked(&p, &t, &[], 3, Some(&SubdivMarks::new()));

    assert_eq!(base.positions.len(), 642, "二十面体 k=3 的顶点数应为 642");
    assert_eq!(base.tri_count(), 1280);
    for (name, r) in [("None", &none), ("全 0 权重", &zero), ("空标记", &empty)] {
        assert_eq!(r.positions.len(), base.positions.len(), "{name}: 顶点数变了");
        assert_eq!(r.indices, base.indices, "{name}: 索引流变了");
        assert_eq!(bits_of(&r.positions), bits_of(&base.positions), "{name}: 位置不是逐位相同");
        assert_eq!(bits_of(&r.normals), bits_of(&base.normals), "{name}: 法线不是逐位相同");
    }
    // 全 0 权重的标记**确实被登记过**（不是拿空对象蒙混过去）。
    assert_eq!(zero_mark.edge_count(), 30, "二十面体应有 30 条边");
    assert!(zero_mark.is_empty(), "全 0 权重必须判为空标记");
}

/// **V1（Catmull-Clark）：同一口径。** 立方体 k=3 ⇒ 386 顶点
/// （`8+12+6 = 26`、`26+48+24 = 98`、`98+192+96 = 386`）、768 四边面 → 1536 三角。
#[test]
fn v1_cc_none_and_zero_marks_are_bitwise_identical() {
    let (p, q) = cube_quads();
    let zero_mark = all_crease(&q, 4, 0.0);
    let base = catmull_clark(&p, &q, &[], 3);
    let none = catmull_clark_marked(&p, &q, &[], 3, None);
    let zero = catmull_clark_marked(&p, &q, &[], 3, Some(&zero_mark));

    assert_eq!(base.positions.len(), 386, "立方体 k=3 的顶点数应为 386");
    assert_eq!(base.tri_count(), 768, "6·4³ = 384 个四边面 → 每面 2 三角");
    for (name, r) in [("None", &none), ("全 0 权重", &zero)] {
        assert_eq!(r.indices, base.indices, "{name}: 索引流变了");
        assert_eq!(bits_of(&r.positions), bits_of(&base.positions), "{name}: 位置不是逐位相同");
        assert_eq!(bits_of(&r.normals), bits_of(&base.normals), "{name}: 法线不是逐位相同");
    }
    assert_eq!(zero_mark.edge_count(), 12, "立方体应有 12 条边");
}

/// **V1（uv 通路）：带 uv 时也逐位相同。**
///
/// uv 在顶点域插值，与 crease 无关，但它走的是**同一串**新顶点插入逻辑；
/// 若标记通路改了顶点顺序，uv 会先露馅（uv 是逐顶点对应的）。
#[test]
fn v1_uv_path_is_bitwise_identical() {
    let (p, q) = cube_quads();
    let uv: Vec<[f32; 2]> = (0..p.len()).map(|i| [(i as f32) * 0.125, 1.0 - (i as f32) * 0.125]).collect();
    let base = catmull_clark(&p, &q, &uv, 2);
    let marked = catmull_clark_marked(&p, &q, &uv, 2, Some(&SubdivMarks::new()));
    assert_eq!(base.uv.len(), base.positions.len());
    assert_eq!(
        base.uv.iter().map(|u| (u[0].to_bits(), u[1].to_bits())).collect::<Vec<_>>(),
        marked.uv.iter().map(|u| (u[0].to_bits(), u[1].to_bits())).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// V2：立方体全 crease ⇒ 极限 = 原角点，体积不缩水
// ---------------------------------------------------------------------------

/// **V2（CC）：全 crease 的立方体笼 ⇒ 8 个角点的极限 = 原角点，体积 = 8。**
///
/// 对照（无标记）在同一测试里现算：角点误差 ≈ 8.66e-1、体积明显缩水。
/// 这两个数**不是**引用的，是本测试当场跑出来的——判据的判别力靠它们。
#[test]
fn v2_cc_all_crease_cube_keeps_corners_and_volume() {
    let (p, q) = cube_quads();
    let marks = all_crease(&q, 4, 1.0);
    assert_eq!(marks.edge_count(), 12);
    assert!(!marks.is_empty());

    let smooth = catmull_clark(&p, &q, &[], 2);
    let sharp = catmull_clark_marked(&p, &q, &[], 2, Some(&marks));

    let e_smooth = corner_error(&smooth.positions, &p);
    let e_sharp = corner_error(&sharp.positions, &p);
    let v_smooth = mesh_volume(&smooth.positions, &smooth.indices);
    let v_sharp = mesh_volume(&sharp.positions, &sharp.indices);
    println!(
        "V2(CC k=2)：角点误差 光滑 {e_smooth:e} → crease {e_sharp:e}；\
         体积 光滑 {v_smooth:.9} → crease {v_sharp:.9}（解析 8）"
    );

    assert!(e_sharp <= 1e-6, "全 crease 的角点极限应等于原角点，实测误差 {e_sharp:e}");
    assert!((v_sharp - 8.0).abs() <= 1e-6, "全 crease 的体积应为 8，实测 {v_sharp:.9}");
    // 判别力：无标记必须**明显**不满足（否则这条判据什么也没测）。
    assert!(e_smooth > 1e-1, "对照（无标记）的角点误差应 ≫ 1e-6，实测 {e_smooth:e}");
    assert!((v_smooth - 8.0).abs() > 1e-1, "对照（无标记）的体积应缩水，实测 {v_smooth:.9}");
}

/// **V2（Loop）：三角化立方体全 crease ⇒ 同样保住 8 角点与体积。**
///
/// 三角笼的角点是价 6 顶点（3 条立方体棱 + 3 条面内对角线），全 crease 下
/// 命中 `k ≥ 3` 的 corner 分支。面内对角线也是 crease ⇒ 面保持平面 ⇒
/// 极限面就是立方体本身（不是"接近"）。
#[test]
fn v2_loop_all_crease_cube_keeps_corners_and_volume() {
    let (p, t) = cube_tris();
    let marks = all_crease(&t, 3, 1.0);
    assert_eq!(marks.edge_count(), 18, "12 条立方体棱 + 6 条面内对角线");

    let smooth = subdivide(&p, &t, &[], 3);
    let sharp = subdivide_marked(&p, &t, &[], 3, Some(&marks));

    let e_smooth = corner_error(&smooth.positions, &p);
    let e_sharp = corner_error(&sharp.positions, &p);
    let v_smooth = mesh_volume(&smooth.positions, &smooth.indices);
    let v_sharp = mesh_volume(&sharp.positions, &sharp.indices);
    println!(
        "V2(Loop k=3)：角点误差 光滑 {e_smooth:e} → crease {e_sharp:e}；\
         体积 光滑 {v_smooth:.9} → crease {v_sharp:.9}（解析 8）"
    );

    assert!(e_sharp <= 1e-6, "全 crease 的角点极限应等于原角点，实测误差 {e_sharp:e}");
    assert!((v_sharp - 8.0).abs() <= 1e-6, "全 crease 的体积应为 8，实测 {v_sharp:.9}");
    assert!(e_smooth > 1e-1, "对照（无标记）的角点误差应 ≫ 1e-6，实测 {e_smooth:e}");
    assert!((v_smooth - 8.0).abs() > 1e-1, "对照（无标记）的体积应缩水，实测 {v_smooth:.9}");
}

/// **V2 的"两级之后仍然锐"**：全 crease 时 k=1 与 k=4 的角点误差同量级。
///
/// 这条专打"`w == 1` 只锐化一级"的错实现——那种实现下 k=1 误差 ≈ 0
/// （第一级确实是中点/不动），k=4 就回弹到 1e-1。只看 k=2 一条会被它骗过。
#[test]
fn v2_sharpness_does_not_decay_with_levels() {
    let (p, q) = cube_quads();
    let marks = all_crease(&q, 4, 1.0);
    let mut worst = 0.0f32;
    for k in 1..=4u32 {
        let r = catmull_clark_marked(&p, &q, &[], k, Some(&marks));
        let e = corner_error(&r.positions, &p);
        println!("V2(CC k={k})：角点误差 {e:e}");
        assert!(e <= 1e-6, "k={k} 的角点误差 {e:e} 超过 1e-6 —— 锐度在衰减");
        worst = worst.max(e);
    }
    assert!(worst <= 1e-6);
}

// ---------------------------------------------------------------------------
// V3：单条 crease 链 ⇒ 折痕顶点极限落在 1D 曲线极限上
// ---------------------------------------------------------------------------

/// **V3：两条 crease 边交汇的顶点 ⇒ 极限 = `(b₀ + 4v + b₁)/6`（解析值）。**
///
/// 二十面体上标 `0–11`、`0–5`（`w = 1`）。顶点 0 恰 2 条 crease ⇒ 1D 曲线
/// 控制点；极限由 `BOUNDARY_LIMIT = (4/6, 1/6)` 给出，与细分级数**无关**
/// （B 样条细分是曲线不变的）⇒ levels = 0/1/2/3 必须给同一个点。
#[test]
fn v3_crease_vertex_limit_is_the_1d_curve_limit() {
    let (p, t) = icosahedron();
    let marks = SubdivMarks::new().with_crease(0, 11, 1.0).with_crease(0, 5, 1.0);
    let want = (p[11] + p[0] * 4.0 + p[5]) * (1.0 / 6.0);
    println!("V3：顶点 0 的 1D 曲线极限解析值 = {want:?}");

    let mut seen: Vec<Vec3> = Vec::new();
    for k in 0..=3u32 {
        let r = subdivide_marked(&p, &t, &[], k, Some(&marks));
        let got = r.positions[0];
        let e = (got - want).length();
        println!("V3：k={k} ⇒ {got:?}，误差 {e:e}");
        assert!(e <= 1e-6, "k={k} 的折痕顶点极限 {got:?} 偏离解析值 {want:?}，误差 {e:e}");
        seen.push(got);
    }
    // 级数无关：四个 k 给同一个点（逐位都行，这里用 1e-6 留舍入余量）。
    for w in seen.windows(2) {
        assert!((w[0] - w[1]).length() <= 1e-6, "不同级数给出了不同的极限点");
    }
}

/// **V3 的另一半：折痕**邻域**不受影响。**
///
/// levels = 0 时只有极限 stencil 生效，故：
/// * 顶点 5 / 11 是 **dart**（恰 1 条 crease）⇒ 走光滑规则 ⇒ 与无标记**逐位相同**；
/// * 其余顶点（`0–5`、`0–11` 之外）同样逐位相同。
///
/// 只有顶点 0 允许变。这条把"把 crease 误当成全局缩放"的实现直接判红。
#[test]
fn v3_crease_does_not_leak_into_the_smooth_neighbourhood() {
    let (p, t) = icosahedron();
    let marks = SubdivMarks::new().with_crease(0, 11, 1.0).with_crease(0, 5, 1.0);
    let base = subdivide(&p, &t, &[], 0);
    let marked = subdivide_marked(&p, &t, &[], 0, Some(&marks));

    assert_ne!(vbits(marked.positions[0]), vbits(base.positions[0]), "顶点 0 必须变（否则判据空转）");
    for i in 1..p.len() {
        assert_eq!(
            vbits(marked.positions[i]),
            vbits(base.positions[i]),
            "顶点 {i}（非 crease 控制点）被污染了"
        );
    }
}

/// **V3（CC 侧）：同一口径。** 立方体上标 `6–7`、`5–6` 两条棱。
///
/// 顶点 6 = `(1,1,1)`，`b₀ = p[7]`、`b₁ = p[5]`，极限 `(b₀ + 4v + b₁)/6`。
#[test]
fn v3_cc_crease_vertex_limit_is_the_1d_curve_limit() {
    let (p, q) = cube_quads();
    let marks = SubdivMarks::new().with_crease(6, 7, 1.0).with_crease(5, 6, 1.0);
    let want = (p[7] + p[6] * 4.0 + p[5]) * (1.0 / 6.0);
    for k in 0..=3u32 {
        let r = catmull_clark_marked(&p, &q, &[], k, Some(&marks));
        let e = (r.positions[6] - want).length();
        println!("V3(CC)：k={k} ⇒ {:?}，误差 {e:e}", r.positions[6]);
        assert!(e <= 1e-6, "k={k} 的 CC 折痕顶点极限偏离解析值，误差 {e:e}");
    }
}

// ---------------------------------------------------------------------------
// V4：判别力反例（错规则必须 FAIL）
// ---------------------------------------------------------------------------

/// **V4 反例 1：完全忽略 crease（= 现有 `93e59d0` 的行为）。**
///
/// 这条就是路线图里"部分"的那一态：笼的硬边被光滑规则抹圆。V2 的断言
/// （≤ 1e-6）在它上面**必须**红——若它意外通过，说明 V2 的阈值或测量点选错了。
#[test]
fn v4_counterexample_ignoring_crease_must_fail_v2() {
    let (p, q) = cube_quads();
    let unmarked = catmull_clark(&p, &q, &[], 2);
    let e = corner_error(&unmarked.positions, &p);
    println!("V4-1：忽略 crease ⇒ 角点误差 {e:e}（V2 的阈值 1e-6）");
    assert!(e > 0.5, "忽略 crease 必须显著违反 V2，实测 {e:e}");
}

/// **V4 反例 2：`w == 1` 只锐化一级（不跨级传播）。**
///
/// 用公开 API 拼出这个错规则：先带标记跑 **1 级**（把角点钉住 + 边点取中点），
/// 再把这一级的产物（位置 + 三角流）喂进**无标记**的 `subdivide` 继续细分。
/// 这与"子边权重降 0（连 `w == 1` 也降）"的实现在 k = 1 上完全一致，
/// 从 k = 2 起分道扬镳：角点是价 6 顶点，光滑规则会把它拉回笼内。
#[test]
fn v4_counterexample_sharp_only_one_level_must_fail_v2() {
    let (p, t) = cube_tris();
    let marks = all_crease(&t, 3, 1.0);
    let one = subdivide_marked(&p, &t, &[], 1, Some(&marks));
    // 一级之后"标记丢了"：同一份几何、不带标记继续细分 3 级。
    let rest = subdivide(&one.positions, &one.indices, &[], 3);
    let e = (rest.positions[6] - p[6]).length();
    println!("V4-2：只锐化一级 ⇒ 角点误差 {e:e}（V2 的阈值 1e-6）");
    assert!(e > 1e-1, "只锐化一级必须显著违反 V2，实测 {e:e}");
}

/// **V4 反例 3：把 `w == 0` 当成锐边。**
///
/// 全 0 权重的标记若被误判成 crease，角点会被钉住（V2 的形态），而正确
/// 实现给出的是光滑极限 `(1/2, 1/2, 1/2)`。这条同时锁住两个方向：
/// 0 权重**不锐化**，且 V1 的"逐位相同"不是靠"标记被忽略"蒙来的。
#[test]
fn v4_counterexample_zero_weight_treated_as_sharp_must_fail_v2() {
    let (p, q) = cube_quads();
    let zero = all_crease(&q, 4, 0.0);
    let r = catmull_clark_marked(&p, &q, &[], 2, Some(&zero));
    let e = corner_error(&r.positions, &p);
    // 正确实现：0 权重 ≡ 光滑 ⇒ 角点极限 (1/2,1/2,1/2)，误差 8.66e-1。
    let want_corner = p[6] * 0.5;
    let got = r.positions[6];
    println!("V4-3：w=0 ⇒ 角点极限 {got:?}（应为 {want_corner:?}），相对 V2 的误差 {e:e}");
    assert!((got - want_corner).length() <= 1e-6, "w=0 必须退化为光滑极限");
    assert!(e > 0.5, "w=0 若被当成锐边，V2 的断言会误通过（这里必须仍显著违反）");
}

// ---------------------------------------------------------------------------
// 半锐利（分数权重）：判据原文没给数字，这里给一条**可证伪**的下界
// ---------------------------------------------------------------------------

/// 分数权重 `w = 0.5` 必须落在"全光滑"与"全锐"之间，且**不跨级累积**。
///
/// 这条不是路线图的 V-判据，但它挡住三个最常见的错实现：
/// "分数权重被当成 0"（结果 = 光滑）、"被当成 1"（结果 = 锐）、
/// "分数权重也往子边传"（结果随级数一路漂向全锐）。
///
/// 第三条的判法：`w = 0.5` 只作用于**第一级**，之后是纯光滑 ——
/// 而光滑细分不改变极限面，故 `v(0.5, k)` 对 `k ≥ 1` **必须冻结**。
/// 若子边继承了分数权重，顶点会被逐级继续往 `v` 拉，`v(0.5,k)` 单调上漂。
#[test]
fn semi_sharp_fraction_does_not_propagate_to_children() {
    let (p, q) = cube_quads();
    let half = all_crease(&q, 4, 0.5);
    let full = all_crease(&q, 4, 1.0);
    let v = |k: u32, marks: Option<&SubdivMarks>| -> f32 {
        match marks {
            Some(m) => catmull_clark_marked(&p, &q, &[], k, Some(m)).positions[6].length(),
            None => catmull_clark(&p, &q, &[], k).positions[6].length(),
        }
    };
    let mut hs: Vec<f32> = Vec::new();
    for k in 1..=3u32 {
        let (a, m, b) = (v(k, None), v(k, Some(&half)), v(k, Some(&full)));
        println!("V-semi：k={k} 光滑 {a:.6} < 半锐利 {m:.6} < 全锐 {b:.6}");
        // 严格内点：既不是"当成 0"，也不是"当成 1"。
        assert!(a < m, "k={k}：w=0.5 必须比光滑更锐（{m} 应 > {a}）");
        assert!(m < b, "k={k}：w=0.5 必须比全锐更光滑（{m} 应 < {b}）");
        hs.push(m);
    }
    // 分数部分只吃一级 ⇒ 极限面在 k≥1 上冻结（不朝全锐漂）。
    for (i, m) in hs.iter().enumerate() {
        assert!(
            (m - hs[0]).abs() <= 1e-6,
            "k={} 的半锐利极限漂了（{} vs {}）—— 分数权重被子边继承了",
            i + 1,
            m,
            hs[0]
        );
    }
}

// ---------------------------------------------------------------------------
// 构造性契约
// ---------------------------------------------------------------------------

/// `SubdivMarks` 的构造契约：钳位、覆盖、自环、幂等、剔除。
#[test]
fn marks_construction_contract() {
    // 权重钳到 [0,1]；非有限 ⇒ 0。
    let m = SubdivMarks::new()
        .with_crease(3, 1, 5.0)
        .with_crease(7, 9, -2.0)
        .with_crease(2, 8, f32::NAN)
        .with_crease(4, 4, 1.0); // 自环：忽略
    assert_eq!(m.edge_count(), 3, "自环必须被忽略");
    for w in [5.0f32, -2.0, f32::NAN] {
        let one = SubdivMarks::new().with_crease(0, 1, w);
        let expect = if w.is_nan() { 0.0 } else { w.clamp(0.0, 1.0) };
        // 通过公开行为验证：空 ⇒ 与光滑逐位相同的那条路。
        assert_eq!(one.is_empty(), expect == 0.0, "权重 {w} 的钳位结果不对");
    }
    // 覆盖不累加：先 0.3 再 0.0 ⇒ 0.0（空）。
    let m = SubdivMarks::new().with_crease(0, 1, 0.3).with_crease(0, 1, 0.0);
    assert!(m.is_empty(), "同一条边以最后一次为准（覆盖而非累加/取最大）");
    assert_eq!(m.edge_count(), 1);
    // corner 幂等。
    let m = SubdivMarks::new().with_corner(5).with_corner(5).with_corner(2);
    assert_eq!(m.corner_count(), 2);
    assert!(!m.is_empty());
}

/// corner 标记：正二十面体标一个 corner ⇒ 该顶点极限**不动**，其余逐位不变。
///
/// corner 是最容易"写了但没接上"的一条（它只改一条 `if`）。这条用
/// 逐位对照证明它真的生效、且只影响一个顶点。
#[test]
fn corner_mark_pins_exactly_one_vertex() {
    let (p, t) = icosahedron();
    let base = subdivide(&p, &t, &[], 2);
    let marked = subdivide_marked(&p, &t, &[], 2, Some(&SubdivMarks::new().with_corner(3)));
    assert_eq!(
        vbits(marked.positions[3]),
        vbits(p[3]),
        "corner 顶点的极限必须等于自身"
    );
    assert_ne!(vbits(marked.positions[3]), vbits(base.positions[3]), "corner 必须真的改动该顶点");
    for i in 0..p.len() {
        if i == 3 {
            continue;
        }
        assert_eq!(vbits(marked.positions[i]), vbits(base.positions[i]), "顶点 {i} 被 corner 波及");
    }
}

/// 标记指向已被剔除的边 / 越界顶点 ⇒ 不 panic、不制造伪边。
#[test]
fn marks_on_dead_geometry_are_ignored() {
    let (p, q) = cube_quads();
    // (0,1) 是真实边，(0,99) 不存在，(200,201) 完全越界，corner 99 越界。
    let m = SubdivMarks::new()
        .with_crease(0, 1, 1.0)
        .with_crease(0, 99, 1.0)
        .with_crease(200, 201, 1.0)
        .with_corner(99);
    let r = catmull_clark_marked(&p, &q, &[], 2, Some(&m));
    assert_eq!(r.positions.len(), 98, "越界标记不得改变拓扑");
    // 有 (0,1) 这条真边 ⇒ 顶点 0 与 1 被钉住（3 条棱里 1 条 crease + 面内…CC
    // 里顶点 0 的入射边是 0-1、0-3、0-4；只有 0-1 是 crease ⇒ dart ⇒ 光滑）。
    // 故这里只断言"不 panic 且拓扑完整"，不臆断位置。
    assert!(r.positions.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));
}

/// 双跑逐字节一致（确定性契约，加标记通路不引入任何顺序依赖）。
#[test]
fn marked_subdivision_is_bitwise_deterministic() {
    let (p, t) = cube_tris();
    let marks = all_crease(&t, 3, 1.0);
    let a = subdivide_marked(&p, &t, &[], 3, Some(&marks));
    let b = subdivide_marked(&p, &t, &[], 3, Some(&marks));
    assert_eq!(bits_of(&a.positions), bits_of(&b.positions));
    assert_eq!(a.indices, b.indices);

    let (p, q) = cube_quads();
    let marks = all_crease(&q, 4, 1.0);
    let a = catmull_clark_marked(&p, &q, &[], 3, Some(&marks));
    let b = catmull_clark_marked(&p, &q, &[], 3, Some(&marks));
    assert_eq!(bits_of(&a.positions), bits_of(&b.positions));
    assert_eq!(a.indices, b.indices);
}
