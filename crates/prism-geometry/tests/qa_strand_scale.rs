//! W5 / V2：股数线性（1k / 10k / 100k 股的构建时间、三角形流、渲染驻留）。
//!
//! # 判据口径（选型报告 §21.1；原文 §8.1 的三条被拆成三栏——测什么就写什么）
//!
//! | # | 判据 | 阈值 | 度量对象 |
//! |---|---|---|---|
//! | V2a | 构建时间 | `O(n log n)` | 生成 + 三角化 + 建树的墙钟时间 |
//! | V2b | 三角形流内存 vs `1.1 KB/股` | 偏差 ≤ 10% | **三角形本体**（`36 B/三角`，3 万三角/万股） |
//! | V2c | 渲染驻留 vs 同规模三角网 | **超出基准 ≤ 10%**（上界） | **三角化 + `Bvh::build`** 的峰值 RSS 增量 |
//!
//! # 为什么 V2c 不能拿 `1.1 KB/股` 当上界（量纲错误，2026-11 修正）
//!
//! `1.1 KB/股` 的推导是 `30 三角/股 × 36 B = 1080 B`——它只含**三角形本体**。
//! 进程峰值 RSS 必然还包含 `Bvh::tris` 重排副本（又 1080 B/股）+ 节点数组 +
//! `tri_bounds`/`centroids`/`primids` 等构建期数组 + 分配器。
//! 拿"三角形本体的预算"去卡"含 BVH 的峰值 RSS"，**任何实现都不可能通过**；
//! 那不是毛发的问题，是量纲的问题。故 V2c 改成**同口径对比**：
//! 同样多的三角形，普通三角网付多少驻留，毛发通路付多少——差值才是"毛发通路
//! 相对行情有没有额外开销"。这才是 §8 想验的东西。
//!
//! **判据是上界（"无额外驻留"），不是双侧等值**：低于基准不构成缺陷（把它也算失败
//! 会要求实现多烧内存才达标）。实测方向见输出。
//!
//! # 1.1 KB/股 是怎么来的（报告 §8 的推导，本文件复算）
//!
//! 每股 3 个控制点（2 段）、半径 5 mm、`tol = 0.02`：
//! `seg = max(3, ⌈2π·0.005/0.02⌉) = 3`，网格加密系数 1.5 ⇒ 实际 5 段；
//! 管壁 2 段 × 5 × 2 = 20 个三角形 + 两端盖 2 × 5 = 10 ⇒ **30 三角/股**，
//! `30 × 36 B = 1080 B ≈ 1.1 KB`。

mod support;

use std::time::Instant;

use prism_geometry::bvh::{Bvh, Triangle};
use prism_geometry::curve_tess::{tessellate_all, CurveReq};
use prism_math::Vec3;

/// V2c 被测测试名（子进程用 `--exact` 过滤它，故必须是本文件里的真实函数名）。
const V2C_TEST: &str = "v2c_strand_vs_mesh_residency";

/// V2c 的规模：10k 股 ⇒ 300 000 个三角形。
const V2C_N: usize = 10_000;

/// 基准臂网格：`400 × 375` cell × 2 = **300 000** 三角（与 `30 × 10 000` 精确同规模）。
const V2C_CELLS: (usize, usize) = (400, 375);

/// `n` 股（每股 3 控制点、2 段、半径 5 mm）。
fn make_strands(n: usize) -> Vec<CurveReq> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // 确定性铺开：费马螺旋分布，避免所有股重叠（重叠不影响计数，但会让
        // BVH 退化成一棵链表，测出来的不是目标场景）。
        let k = i as f32 + 0.5;
        let r = (k / n as f32).sqrt() * 2.0;
        let a = k * 2.399_963_2; // 黄金角
        let (c, s) = prism_geometry::det_math::det_cos_sin(a);
        let base = Vec3::new(c * r, 0.0, s * r);
        let dir = Vec3::new(c * 0.05, 1.0, s * 0.05).normalize();
        out.push(CurveReq::Strand {
            points: vec![base, base + dir * 0.05, base + dir * 0.1],
            radii: vec![0.005],
        });
    }
    out
}

/// 跑一档：生成 + 三角化 + 建树，返回 `(耗时 ms, 三角数, RSS 增量 KB)`。
fn run(n: usize) -> (f64, usize, Option<u64>) {
    let before = support::rss_kb();
    let t0 = Instant::now();
    let strands = make_strands(n);
    let tris: Vec<Triangle> = tessellate_all(&strands, 0.02);
    let bvh = Bvh::build(&tris);
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    let after = support::rss_kb();
    // 防止优化器把结果整段消掉（也顺带证明树可用）。
    assert_eq!(bvh.tris.len(), tris.len(), "BVH 必须收下全部三角形");
    let rss = match (before, after) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    };
    (dt, tris.len(), rss)
}

/// **V2a + V2b：1k / 10k / 100k 股的构建时间与三角形流（两条都已 PASS，阈值未动）。**
///
/// 本测试同时打印进程 RSS 增量，但那是**参考值**、不是判据：进程 RSS 的判据是 V2c，
/// 它必须与同规模三角网对比才有意义（见文件头）。
#[test]
fn v2ab_strand_build_scales_and_triangle_stream() {
    let _guard = support::serialize();
    let mut rows: Vec<(usize, f64, usize, Option<u64>)> = Vec::new();
    for n in [1_000usize, 10_000, 100_000] {
        // 三档都**无条件真跑**（不做 debug/release 分流、不 `#[ignore]`）：
        // 100k 档在 debug 下约 27 s（300 万三角形的 SAH-BVH），release 约 4.8 s。
        let (ms, tri, rss) = run(n);
        println!(
            "V2ab：n={n:>7} 三角={tri:>9}（{:.1} 三角/股）耗时 {ms:>9.1} ms（{:.3} µs/股）\
             进程 RSS 增量（参考）{}",
            tri as f64 / n as f64,
            ms * 1000.0 / n as f64,
            rss.map(|kb| format!("{kb} KB（{:.0} B/股）", kb as f64 * 1024.0 / n as f64))
                .unwrap_or_else(|| "未测（无 /proc）".into()),
        );
        rows.push((n, ms, tri, rss));
    }
    assert!(rows.len() >= 2, "至少要有 1k / 10k 两档");

    // 每股三角形数必须一致（30/股），否则"股数线性"就成了空话。
    for (n, _, tri, _) in &rows {
        assert_eq!(
            *tri,
            30 * *n,
            "n={n} 每股三角形数应为 30（2 段×5 环×2 + 2 端盖×5）"
        );
    }

    // V2a 时间：n·log n 归一化后应近似常数（取末档/首档比较）。
    if rows.len() >= 2 {
        let (n0, t0, _, _) = rows[0];
        let (n1, t1, _, _) = rows[rows.len() - 1];
        let lhs = t1 / t0;
        let rhs = (n1 as f64 * (n1 as f64).ln()) / (n0 as f64 * (n0 as f64).ln());
        println!(
            "V2a：时间比 {lhs:.2}× vs n·log n 预期 {rhs:.2}×（偏差 {:.1}%）",
            (lhs / rhs - 1.0) * 100.0
        );
        // SAH 构建是 O(n log n)，但三角化是 O(n)、分配器与缓存效应会带来常数项；
        // 判据要求"O(n log n)"，这里给 2× 的宽带上界（超了说明出现了超线性的树构建）。
        assert!(
            lhs <= rhs * 2.0,
            "构建时间 {lhs:.2}× 远超 n·log n 的 {rhs:.2}×"
        );
    }

    // V2b 三角形流：与 1.1 KB/股 的偏差（**这一栏才是 1.1 KB/股 说得清的东西**）。
    for (n, _, tri, rss) in &rows {
        let budget_kb = 1.1 * *n as f64; // 1.1 KB × n，单位 KB
        let stream_bytes = (*tri as f64) * 36.0;
        let stream_kb = stream_bytes / 1024.0;
        let dev_stream = (stream_kb / budget_kb - 1.0) * 100.0;
        println!(
            "V2b：n={n} 三角形流 {stream_kb:.1} KB vs 预算 {budget_kb:.1} KB ⇒ 偏差 {dev_stream:+.1}%"
        );
        assert!(
            dev_stream.abs() <= 10.0,
            "n={n} 三角形流与 1.1 KB/股 偏差 {dev_stream:+.1}% 超过 10%"
        );
        if let Some(kb) = rss {
            println!(
                "V2c 参考（进程峰值增量，含 BVH 重排副本与节点；不构成判据，判据见 v2c_strand_vs_mesh_residency）：\
                 n={n} {kb} KB（{:.0} B/股）",
                *kb as f64 * 1024.0 / *n as f64
            );
        }
    }
}

/// 子进程侧：跑**一个** arm 并打印 `@RSS` 行。
///
/// * `strand`：10k 股 → `tessellate_all` → `Bvh::build`；
/// * `mesh`：300k 三角的规则网格 → `Bvh::build`。
///
/// 两臂的度量窗口完全一样（都是"产出 `Vec<Triangle>` + 建树"），三角形数也精确相同，
/// 故 `strand / mesh` 的比值就是"毛发通路相对同规模三角网的额外驻留"。
fn run_v2c_arm(arm: &str) {
    let _guard = support::serialize();
    let tri_size = std::mem::size_of::<Triangle>();
    let (tris, note, peak) = match arm {
        "strand" => {
            let ((tri, stream), peak) = support::measure(|| {
                let strands = make_strands(V2C_N);
                let t: Vec<Triangle> = tessellate_all(&strands, 0.02);
                let bvh = Bvh::build(&t);
                assert_eq!(bvh.tris.len(), t.len(), "BVH 必须收下全部三角形");
                (t.len(), t.len() * tri_size)
            });
            (
                tri,
                format!("{V2C_N} 股 ⇒ 三角化 {stream} B 三角形流（{tri_size} B/三角）"),
                peak,
            )
        }
        "mesh" => {
            let (cx, cz) = V2C_CELLS;
            let ((tri, stream), peak) = support::measure(|| {
                let t: Vec<Triangle> = support::grid_triangles(cx, cz);
                let bvh = Bvh::build(&t);
                assert_eq!(bvh.tris.len(), t.len(), "BVH 必须收下全部三角形");
                (t.len(), t.len() * tri_size)
            });
            (
                tri,
                format!("{cx}×{cz} 规则网格 ⇒ {stream} B 三角形流（{tri_size} B/三角）"),
                peak,
            )
        }
        other => panic!("未知 arm：{other}（只认 strand / mesh）"),
    };
    support::report_arm(arm, tris, peak, &note);
}

/// **V2c：毛发通路的渲染驻留 vs 同规模三角网（同口径，**超出基准 ≤ 10%** 上界）。**
///
/// 判据含义：**毛股通路（三角化 + 建树）没有超出行情的额外驻留**。
/// 两臂各在独立子进程里量峰值 RSS（见 `tests/support/mod.rs` 关于"释放不还页"的说明），
/// 故读数不受同一进程内先后顺序影响。
#[test]
fn v2c_strand_vs_mesh_residency() {
    if let Ok(arm) = std::env::var(support::ARM_ENV) {
        run_v2c_arm(&arm);
        return;
    }
    let _guard = support::serialize();
    let strand = support::run_arm(V2C_TEST, "strand");
    let mesh = support::run_arm(V2C_TEST, "mesh");
    assert_eq!(
        strand.tris, mesh.tris,
        "两臂三角形数必须相同（否则不是同规模）：strand={} mesh={}",
        strand.tris, mesh.tris
    );
    println!("V2c strand：{}（{} 三角）", strand.note, strand.tris);
    println!("V2c mesh  ：{}（{} 三角）", mesh.note, mesh.tris);
    match (strand.peak, mesh.peak) {
        (Some(a), Some(b)) => {
            let ratio = a.delta_kb() as f64 / b.delta_kb().max(1) as f64;
            let dev = (ratio - 1.0) * 100.0;
            println!(
                "V2c：毛发通路峰值 RSS 增量 {} KB（{} 三角，{:.2} KB/kTri）vs 同规模三角网 {} KB\
                 （{:.2} KB/kTri）⇒ 比值 {ratio:.3}×、偏差 {dev:+.1}%（上界 +10%）",
                a.delta_kb(),
                strand.tris,
                a.delta_kb() as f64 / (strand.tris as f64 / 1000.0),
                b.delta_kb(),
                b.delta_kb() as f64 / (mesh.tris as f64 / 1000.0),
            );
            if dev < -10.0 {
                println!(
                    "V2c：毛发通路比基准**低** {:.1}%——低于基准不是缺陷（判据是'无额外驻留'的上界）。",
                    -dev
                );
            }
            // 判据语义是**上界**："毛发通路相对同规模三角网无额外驻留"。
            // 低于基准不构成缺陷；把它也算失败会要求实现多烧内存才达标，那不是本判据要验的东西。
            assert!(
                dev <= 10.0,
                "V2c：毛发通路峰值 RSS {} KB vs 同规模三角网 {} KB，多出 {dev:+.1}%（上界 +10%）",
                a.delta_kb(),
                b.delta_kb()
            );
        }
        _ => println!("V2c：未测（本机无 /proc/self/status 或无 clear_refs）——不假装通过"),
    }
}
