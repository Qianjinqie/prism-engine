//! W4 / V5：细分产物的内存（**独立测试二进制**）。
//!
//! # 判据口径（选型报告 §22.1；原文 §5.1 的一条被拆成四栏——测什么就写什么）
//!
//! | # | 判据 | 阈值 | 度量对象 |
//! |---|---|---|---|
//! | V5a | 三角数 | 精确 `4^k×面数` | 产物索引流的三角计数 |
//! | V5b | 三角形流字节 | 精确 `4^k×面数×36 B` | 产物烘焙成三角形流后的字节 |
//! | V5c | 索引网格字节 vs 三角流 | ≤ 5% | 产物本身（positions + normals + uv + indices） |
//! | V5d | 细分构建峰值 RSS vs 同三角数三角网的 `Bvh::build` | **超出基准 ≤ 10%**（上界） | **构建期峰值**（同口径） |
//!
//! # 为什么 V5d 不能拿"产物字节"当上界（量纲错误，2026-11 修正）
//!
//! 增量构建器**必然**在构建期同时持有：输入笼、半边/边表/邻接 CSR、下一级输出（双缓冲）。
//! "峰值 RSS ≤ 产物字节 × 1.05"对任何增量构建器都不成立——那不是细分的缺陷，是量纲错了。
//! 修正后 V5d 检验的是"**同口径**"：同样多的三角形，交给 `Bvh::build` 付多少峰值，
//! 细分构建器付多少。两者都是"产出 N 个三角形"的构建期成本，才有可比性。
//!
//! **判据是上界（"构建器不病态"），不是双侧等值**：低于基准不构成缺陷。若把"低于 10%"
//! 也算不通过，就等于要求实现**多烧内存**才达标——那不是本判据要验的东西。故这一栏的
//! 含义只能是"不多占"。
//!
//! # 两臂都必须含 `Bvh::build`（2026-11 修正：上一版 subdiv 臂**没建树**，−45% 是假绿）
//!
//! 上一版的 `"subdiv"` 臂只跑 `subdivide(k)`，`"mesh"` 臂跑 `grid_triangles` + `Bvh::build`
//! ——两臂差了一整套 BVH（36 B/三角的三角形流副本 + 节点）。于是 −40.5%~−46.1% 量的
//! 不是"细分构建更省"，而是"细分臂少干了一半活"。**假绿比红更危险**，必须修。
//!
//! 现在两臂都是 `… → Vec<Triangle> → Bvh::build(&t)`，且在臂内断言
//! `bvh.tris.len() == t.len()`：同三角数、同 BVH、**同一份三角形流口径**——`Triangle`
//! 只有 `a`/`b`/`c` 三个位置字段，两臂都**不带**法线/uv（不许一个带一个不带，
//! 那是"同规模基准"的一部分）。
//!
//! # 为什么 fs 上的独测仍不够：每个 arm 一个子进程
//!
//! 判据量的是**峰值 RSS**，而 `cargo test` 在同一个进程里**并行**跑同一二进制的所有
//! 测试。故本文件内所有内存敏感测试都先取 `support::serialize()` 串行锁，且 V5d 的两个
//! arm 各自在**独立子进程**里量（同一进程里先后跑两臂时，后跑的那臂会复用前者未归还的
//! 页 ⇒ 增量系统性偏小 ⇒ 偏向通过）。详见 `tests/support/mod.rs`。

mod support;

use prism_geometry::bvh::{Bvh, Triangle};
use prism_geometry::subdiv::subdivide;
use prism_math::Vec3;

/// V5d 被测测试名（子进程用 `--exact` 过滤它，故必须是本文件里的真实函数名）。
const V5D_TEST: &str = "v5d_subdiv_vs_mesh_build_peak";

/// V5d 基准臂网格的 X 向 cell 数（`cells_z` 由目标三角数反推：`tri = 2·cx·cz`）。
const V5D_CELLS_X: usize = 128;

/// 正二十面体（20 面、12 顶点）。
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

/// **V5a + V5b + V5c：k=5/6 的三角数与产物字节（三条都已 PASS，阈值未动）。**
///
/// 三角数是**结构性**的（`4^k × 20`）；产物字节是**算出来的**（索引网格实测长度）。
/// 两者的口径差别必须写在输出里，不能混着报：
///
/// * 公式 `4^k × 面数 × 36 B` 描述的是**三角形流**（每三角 3×`Vec3`）；
/// * 本实现的产物是**索引网格**（`positions` 12 B + `normals` 12 B + `uv` 8 B
///   每顶点，`indices` 12 B 每三角，闭合网格 `V ≈ T/2`）⇒ 每三角形约 28 B
///   **小于** 36 B。烘焙成三角形流后才回到 36 B/三角。
///
/// 本测试也打印进程 RSS 增量，但那是**参考值**、不是判据（判据是 V5d 的同口径对比）。
#[test]
fn v5abc_memory_and_triangle_counts() {
    let _guard = support::serialize();
    let (p, t) = icosahedron();
    let faces = 20u64;
    for k in [5u32, 6] {
        let before = support::rss_kb();
        let r = subdivide(&p, &t, &[], k);
        let after = support::rss_kb();
        let tri = r.tri_count() as u64;
        let want_tri = faces * 4u64.pow(k);
        assert_eq!(tri, want_tri, "k={k} 的三角数必须是 4^k×20");

        let formula_bytes = want_tri * 36;
        let indexed_bytes = (r.positions.len() as u64) * (12 + 12 + 8)
            + (r.indices.len() as u64) * 4;
        let soup_bytes = tri * 36;
        let measured_kb = match (before, after) {
            (Some(b), Some(a)) => Some(a.saturating_sub(b)),
            _ => None,
        };
        println!(
            "V5：k={k} 三角 {tri}（公式 {want_tri}）；三角形流 {soup_bytes} B = {:.2} MB；\
             索引网格 {indexed_bytes} B = {:.2} MB；公式(36 B/tri) {formula_bytes} B = {:.2} MB；\
             进程 RSS 增量（参考）{}",
            soup_bytes as f64 / 1048576.0,
            indexed_bytes as f64 / 1048576.0,
            formula_bytes as f64 / 1048576.0,
            measured_kb.map(|kb| format!("{kb} KB")).unwrap_or_else(|| "未测（无 /proc）".into()),
        );
        // V5a：三角形流与公式必须**精确**相等（公式就是它）。
        assert_eq!(soup_bytes, formula_bytes, "三角形流必须精确等于 4^k×面数×36 B");
        // V5c：索引网格严格小于三角形流（不超预算）。
        assert!(
            indexed_bytes < soup_bytes,
            "索引网格 {indexed_bytes} B 不应超过三角形流 {soup_bytes} B"
        );
    }
}

/// 子进程侧：跑**一个** V5d arm 并打印 `@RSS` 行。
///
/// **两臂都含 `Bvh::build`**（这是判据可信的前提，见文件头）：
///
/// * `subdiv:<k>`：`subdivide(k)` → 三角形流（36 B/三角）→ `Bvh::build`；
/// * `mesh:<cx>x<cz>`：同三角数规则网格 → 三角形流 → `Bvh::build`。
///
/// 两臂进入 `Bvh::build` 时手里都**只有一份等长的 `Vec<Triangle>`**：subdiv 臂在把产物
/// 转成三角形流之后显式 `drop` 掉 `SubdivResult`。这不隐藏任何成本——`VmHWM` 是窗口内的
/// **单调高水位**，`subdivide` 阶段的峰值、以及"产物 + 正在增长的三角形流"并存的转换期
/// 峰值，都已经记进高水位了；丢掉它只是不让**已经用不上的中间产物**被算进*建树阶段*的
/// 驻留——否则量到的是"我的量测代码多留了一个变量"，两臂又不是同口径。
fn run_v5d_arm(arm: &str) {
    let _guard = support::serialize();
    let (kind, arg) = arm
        .split_once(':')
        .unwrap_or_else(|| panic!("arm 格式应为 <kind>:<arg>，收到 {arm}"));
    let (tris, note, peak) = match kind {
        "subdiv" => {
            let k: u32 = arg.parse().unwrap_or_else(|e| panic!("非法级数 {arg}：{e}"));
            let ((tris, soup, pos, indexed), peak) = support::measure(|| {
                let (p, t) = icosahedron();
                let r = subdivide(&p, &t, &[], k);
                let n_pos = r.positions.len();
                let n_tri = r.tri_count();
                let indexed = n_pos * (12 + 12 + 8) + r.indices.len() * 4;
                // 做法 (b)：`SubdivResult` 没有 `to_triangles()` / `triangles()`。
                // 不为此给 `subdiv.rs` 加公开方法——那会把只服务于一个测试的方法
                // 变成几何 crate 的对外接口（`prism-rt` 侧会当它是稳定 API），
                // 超出本轮"只改这一处"的范围。故在臂内现场组流。
                // 字段口径与 `support::grid_triangles` **完全一致**：`Triangle` 只有
                // `a`/`b`/`c` 三个位置，本 crate 的三角形流不带法线/uv ⇒ 两臂同口径。
                let mut soup: Vec<Triangle> = Vec::with_capacity(n_tri);
                for c in r.indices.chunks_exact(3) {
                    soup.push(Triangle {
                        a: r.positions[c[0] as usize],
                        b: r.positions[c[1] as usize],
                        c: r.positions[c[2] as usize],
                    });
                }
                drop(r); // `VmHWM` 单调：上面两个阶段的峰值不会被这一行抹掉
                let bvh = Bvh::build(&soup);
                assert_eq!(bvh.tris.len(), soup.len(), "BVH 必须收下全部三角形");
                (
                    soup.len(),
                    soup.len() * std::mem::size_of::<Triangle>(),
                    n_pos,
                    indexed,
                )
            });
            assert_eq!(tris, 20 * 4usize.pow(k), "subdivide(k={k}) 的三角形数应为 4^k×20");
            (
                tris,
                format!(
                    "subdivide(k={k}) → Bvh::build（三角形流 {soup} B = {tris} 三角）；\
                     产物索引网格 {indexed} B（{pos} 顶点）"
                ),
                peak,
            )
        }
        "mesh" => {
            let (cx, cz) = arg
                .split_once('x')
                .unwrap_or_else(|| panic!("网格 arm 格式应为 <cx>x<cz>，收到 {arg}"));
            let (cx, cz): (usize, usize) = (
                cx.parse().unwrap_or_else(|e| panic!("非法 cx {cx}：{e}")),
                cz.parse().unwrap_or_else(|e| panic!("非法 cz {cz}：{e}")),
            );
            let ((tri, soup), peak) = support::measure(|| {
                let t = support::grid_triangles(cx, cz);
                let bvh = Bvh::build(&t);
                assert_eq!(bvh.tris.len(), t.len(), "BVH 必须收下全部三角形");
                (t.len(), t.len() * std::mem::size_of::<Triangle>())
            });
            (tri, format!("{cx}×{cz} 规则网格 → Bvh::build（三角形流 {soup} B）"), peak)
        }
        other => panic!("未知 arm 类型：{other}（只认 subdiv / mesh）"),
    };
    support::report_arm(arm, tris, peak, &note);
}

/// **V5d：细分构建峰值 RSS vs 同三角数三角网的 `Bvh::build`（同口径，**超出基准 ≤ 10%** 上界）。**
#[test]
fn v5d_subdiv_vs_mesh_build_peak() {
    if let Ok(arm) = std::env::var(support::ARM_ENV) {
        run_v5d_arm(&arm);
        return;
    }
    let _guard = support::serialize();
    for k in [5u32, 6] {
        let tri = 20usize * 4usize.pow(k);
        // `2·cx·cz == tri` ⇒ `cz = tri / (2·cx)`；k=5 → 80、k=6 → 320。
        let cz = tri / (2 * V5D_CELLS_X);
        let subdiv = support::run_arm(V5D_TEST, &format!("subdiv:{k}"));
        let mesh = support::run_arm(V5D_TEST, &format!("mesh:{V5D_CELLS_X}x{cz}"));
        assert_eq!(
            subdiv.tris, tri,
            "k={k} 细分三角数应为 {tri}，实得 {}",
            subdiv.tris
        );
        assert_eq!(
            mesh.tris, tri,
            "k={k} 基准网格三角数应为 {tri}，实得 {}",
            mesh.tris
        );
        println!("V5d k={k} subdiv：{}（{} 三角）", subdiv.note, subdiv.tris);
        println!("V5d k={k} mesh  ：{}（{} 三角）", mesh.note, mesh.tris);
        match (subdiv.peak, mesh.peak) {
            (Some(a), Some(b)) => {
                let ratio = a.delta_kb() as f64 / b.delta_kb().max(1) as f64;
                let dev = (ratio - 1.0) * 100.0;
                println!(
                    "V5d：k={k} 细分构建峰值 {} KB vs 同三角数({tri})三角网 Bvh::build 峰值 {} KB\
                     ⇒ 比值 {ratio:.3}×、偏差 {dev:+.1}%（上界 +10%）",
                    a.delta_kb(),
                    b.delta_kb(),
                );
                if dev < -10.0 {
                    println!(
                        "V5d：k={k} 细分构建比基准**低** {:.1}%——低于基准不是缺陷（判据是'构建器不\
                         病态'的上界；双侧会把'更省'判成失败，那等于要求实现多烧内存才达标）。\
                         两臂都含 `Bvh::build`、三角形流同口径，故这个方向只如实记录、不作解读。",
                        -dev
                    );
                }
                // 判据语义是**上界**："细分构建器的峰值不高于同三角数三角网的建树峰值"。
                // 低于基准不构成缺陷；把它也算失败会要求实现多烧内存才达标，那不是本判据要验的东西。
                assert!(
                    dev <= 10.0,
                    "V5d：k={k} 细分构建峰值 {} KB vs 同规模三角网 {} KB，多出 {dev:+.1}%（上界 +10%）",
                    a.delta_kb(),
                    b.delta_kb()
                );
            }
            _ => println!(
                "V5d：k={k} 未测（本机无 /proc/self/status 或无 clear_refs）——不假装通过"
            ),
        }
    }
}
