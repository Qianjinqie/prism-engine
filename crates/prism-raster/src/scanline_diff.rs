//! 优化实现 vs 参考实现的差分对拍。
//!
//! # 判定口径（两层）
//!
//! 1. **精度（主判据）**：新实现 vs [`rasterize_reference_f64`]（同算法、f64 数值
//!    的精度标尺，结果与精确有理数积分一致到 1e-12 量级）——**最大绝对差 ≤ 2e-6**。
//!    这是「覆盖率是不是真的精确面积」的直接证据。
//! 2. **回归（副判据）**：新实现 vs [`rasterize_reference`]（优化前的 f32 实现，
//!    逐字保留）——界取「坐标量级下的 4 个 f32 ulp」。
//!
//! 为什么不要求「新 vs 旧 ≤ 2e-6」：旧实现全程 f32 插值，1080p 坐标（x ≈ 1300）
//! 的 f32 ulp 是 1.2e-4，它**自己**相对精确面积的误差就有 5e-5～7.6e-5；任何独立
//! 舍入的实现都不可能和它 2e-6 内一致。实测新实现相对精确面积 3e-8（1 ulp），
//! 旧实现 5.1e-5——两者之差就是旧实现的量化误差本身（见各类用例日志）。
//!
//! 输入覆盖：
//! - 基准四用例（circle64 / star512@1080p / star512@4K / star2048@4K）+ 孔洞环；
//! - 自交多边形、外环+内环、`Winding` / `EvenOdd` 两种规则；
//! - 退化：零面积、重合边、顶点落在行带边界、轮廓超画布、极细条、巨大坐标；
//! - ≥200 组固定种子伪随机多边形（`prism_noise::pcg`，无 `rand`）。

use prism_noise::pcg;

use crate::geom::Point;
use crate::scanline::{rasterize, FillRule};
use crate::scanline_ref::{rasterize_reference, rasterize_reference_f64};
use crate::Coverage;

/// 主判据：新实现 vs 精确面积（f64 标尺）。
const TOL_EXACT: f64 = 2e-6;

/// 副判据：新实现 vs 旧 f32 实现的界——坐标量级下 4 个 f32 ulp。
///
/// 旧实现在 `extent` 量级的坐标上做乘加，单步舍入 ~`extent·2⁻²⁴`；逐子带梯形
/// 累加把它放大到几倍 ulp。4 ulp 是实测值的 3～6 倍余量（1080p 用掉约 1/6，
/// 4K 约 1/4），既能抓住结构错（那是 1e-2 量级），又不会因舍入路径差异误报。
fn legacy_floor(extent: f32) -> f64 {
    4.0 * (extent as f64) * (1.0 / 16_777_216.0)
}

/// 两缓冲的最大绝对差；超过 `tol` 时打印首个超差位置。
fn max_abs_diff(a: &Coverage, b: &Coverage, name: &str, tol: f64) -> f64 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let mut worst = 0.0f64;
    for (i, (x, y)) in a.data.iter().zip(b.data.iter()).enumerate() {
        let d = (*x as f64 - *y as f64).abs();
        if d > worst {
            worst = d;
            if d > tol {
                let w = a.width as usize;
                println!("  [{name}] 超差 @({},{})：{x} vs {y} |Δ|={d:e}", i % w, i / w);
            }
        }
    }
    worst
}

/// 覆盖率值域自检（非负、≤1、有限）。
fn assert_range(cov: &Coverage, name: &str) {
    for v in &cov.data {
        assert!(v.is_finite(), "[{name}] 覆盖率非有限值 {v}");
        assert!(*v >= -1e-6 && *v <= 1.0 + 1e-6, "[{name}] 覆盖率越界 {v}");
    }
}

/// 主判据：对拍「精确面积」，返回最大绝对差。
fn check_exact(name: &str, contours: &[Vec<Point>], rule: FillRule, w: u32, h: u32) -> f64 {
    let new = rasterize(contours, rule, w, h);
    let oracle = rasterize_reference_f64(contours, rule, w, h);
    assert_range(&new, name);
    let d = max_abs_diff(&new, &oracle, name, TOL_EXACT);
    let same = new
        .data
        .iter()
        .zip(oracle.data.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "[{name}] {w}×{h} {rule:?} |Δexact|={d:.3e} 逐位一致 {same}/{}",
        new.data.len()
    );
    assert!(d <= TOL_EXACT, "[{name}] 相对精确面积的最大差 {d:e} > {TOL_EXACT:e}");
    d
}

/// 副判据：对拍「优化前的 f32 实现」，返回最大绝对差。
fn check_legacy(name: &str, contours: &[Vec<Point>], rule: FillRule, w: u32, h: u32) -> f64 {
    let new = rasterize(contours, rule, w, h);
    let old = rasterize_reference(contours, rule, w, h);
    assert_range(&new, name);
    let bound = legacy_floor(w.max(h) as f32);
    let d = max_abs_diff(&new, &old, name, bound);
    println!("[{name}] {w}×{h} {rule:?} |Δlegacy|={d:.3e}（界 {bound:.1e}）");
    assert!(d <= bound, "[{name}] 与旧实现差 {d:e} 超过 {bound:e}");
    d
}

// ---------------------------------------------------------------- 形状构造

/// 自研 sin/cos（测试内造几何；与光栅数学无关，只需确定性）。
fn sin_cos(t: f32) -> (f32, f32) {
    let tau = core::f32::consts::TAU;
    let mut x = t;
    while x > core::f32::consts::PI {
        x -= tau;
    }
    while x < -core::f32::consts::PI {
        x += tau;
    }
    let x2 = x * x;
    let s = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))));
    let c = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0 * (1.0 - x2 / 56.0)));
    (s, c)
}

fn circle(n: usize, cx: f32, cy: f32, r: f32) -> Vec<Vec<Point>> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32) * (core::f32::consts::TAU / n as f32);
        let (s, c) = sin_cos(t);
        pts.push(Point::new(cx + r * c, cy + r * s));
    }
    vec![pts]
}

fn star(n: usize, cx: f32, cy: f32, r: f32, inner_ratio: f32) -> Vec<Vec<Point>> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32) * (core::f32::consts::TAU / n as f32);
        let ri = if i % 2 == 0 { r } else { r * inner_ratio };
        let (s, c) = sin_cos(t);
        pts.push(Point::new(cx + ri * c, cy + ri * s));
    }
    vec![pts]
}

fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
    vec![
        Point::new(x0, y0),
        Point::new(x1, y0),
        Point::new(x1, y1),
        Point::new(x0, y1),
    ]
}

fn ring_hole(cx: f32, cy: f32, outer: f32, inner: f32) -> Vec<Vec<Point>> {
    let mut o = square(cx - outer, cy - outer, cx + outer, cy + outer);
    o.reverse(); // 外环顺时针
    let i = square(cx - inner, cy - inner, cx + inner, cy + inner); // 内环逆时针
    vec![o, i]
}

/// 伪随机多边形：顶点数 3..12，坐标可比画布大（含超界与贴边）。
fn random_contour(seed: u64, idx: u64, size: f32) -> Vec<Point> {
    let n = 3 + (pcg::hash2(seed, idx, 0) % 10) as usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let hx = pcg::hash2_f32(seed, idx, (i as u64) * 2 + 1);
        let hy = pcg::hash2_f32(seed, idx, (i as u64) * 2 + 2);
        // 约 1/5 的坐标落到画布外，覆盖裁剪路径。
        let ax = if hx < 0.1 { 1.25 } else { 1.0 };
        let ay = if hy < 0.1 { 1.25 } else { 1.0 };
        pts.push(Point::new((hx - 0.5) * size * ax, (hy - 0.5) * size * ay));
    }
    pts
}

/// 伪随机「星」（自交、winding>1）：半径交替。
fn random_star(seed: u64, idx: u64, size: f32) -> Vec<Point> {
    let n = (6 + (pcg::hash2(seed, idx, 0) % 20)) as usize;
    let (cx, cy) = (size * 0.5, size * 0.5);
    let r = size * (0.2 + 0.35 * pcg::hash2_f32(seed, idx, 1));
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32) * (core::f32::consts::TAU / n as f32);
        let jitter = 0.55 + 0.9 * pcg::hash2_f32(seed, idx, (i as u64) + 2);
        let ri = if i % 2 == 0 { r } else { r * jitter * 0.42 };
        let (s, c) = sin_cos(t);
        pts.push(Point::new(cx + ri * c, cy + ri * s));
    }
    pts
}

// ------------------------------------------------------------------- 用例

#[test]
fn bench_circle_and_ring() {
    let (w, h) = (1920u32, 1080u32);
    let c = circle(64, w as f32 * 0.5, h as f32 * 0.5, 0.37 * h as f32);
    check_exact("circle64_1080p", &c, FillRule::Winding, w, h);
    check_legacy("circle64_1080p", &c, FillRule::Winding, w, h);

    let (w, h) = (2048u32, 2048u32);
    let rh = ring_hole(w as f32 * 0.5, h as f32 * 0.5, w as f32 * 0.34, w as f32 * 0.13);
    check_exact("ring_hole_2k/w", &rh, FillRule::Winding, w, h);
    check_legacy("ring_hole_2k/w", &rh, FillRule::Winding, w, h);
    check_exact("ring_hole_2k/e", &rh, FillRule::EvenOdd, w, h);
}

#[test]
fn bench_star512() {
    let (w, h) = (1920u32, 1080u32);
    let s = star(512, w as f32 * 0.5, h as f32 * 0.5, 0.476 * w as f32, 0.42);
    check_exact("star512_1080p", &s, FillRule::Winding, w, h);
    check_legacy("star512_1080p", &s, FillRule::Winding, w, h);

    // 4K：debug 构建下缩到 1/4 边长（旧实现是 O(子带 × 活跃边²)，未优化构建太慢）。
    let scale = if cfg!(debug_assertions) { 4 } else { 1 };
    let (w, h) = (4096u32 / scale, 4096u32 / scale);
    let s = star(512, w as f32 * 0.5, h as f32 * 0.5, 0.476 * w as f32, 0.42);
    check_exact("star512_4k", &s, FillRule::Winding, w, h);
    check_legacy("star512_4k", &s, FillRule::Winding, w, h);
}

#[test]
fn bench_star2048() {
    // 密交点的重载用例：先在 1K 画布上做「精确面积」对拍（4K 的 f64 标尺要跑
    // 数十秒，精度结论由本用例 + star512 系列组合给出），再在 4K 上做旧实现回归。
    let scale = if cfg!(debug_assertions) { 4 } else { 1 };
    let (w, h) = (1024u32 / scale, 1024u32 / scale);
    let s = star(2048, w as f32 * 0.5, h as f32 * 0.5, 0.476 * w as f32, 0.42);
    check_exact("star2048_1k", &s, FillRule::Winding, w, h);
    check_legacy("star2048_1k", &s, FillRule::Winding, w, h);

    if !cfg!(debug_assertions) {
        let (w, h) = (4096u32, 4096u32);
        let s = star(2048, w as f32 * 0.5, h as f32 * 0.5, 0.476 * w as f32, 0.42);
        check_legacy("star2048_4k", &s, FillRule::Winding, w, h);
    }
}

#[test]
fn pathological_cases() {
    let (w, h) = (64u32, 64u32);

    // 自交 8 字形。
    let bowtie = vec![
        Point::new(2.0, 2.0),
        Point::new(40.0, 40.0),
        Point::new(40.0, 2.0),
        Point::new(2.0, 40.0),
    ];
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("bowtie", core::slice::from_ref(&bowtie), rule, w, h);
        check_legacy("bowtie", core::slice::from_ref(&bowtie), rule, w, h);
    }

    // 外环 + 内环孔洞。
    let rh = ring_hole(32.0, 32.0, 26.0, 11.0);
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("ring_hole", &rh, rule, w, h);
        check_legacy("ring_hole", &rh, rule, w, h);
    }

    // 两个同向重叠方块（winding=2 区不重复计数）+ 三重叠加（winding=3）。
    let overlap = [square(4.0, 4.0, 40.0, 40.0), square(20.0, 20.0, 58.0, 58.0)];
    let triple = [
        square(4.0, 4.0, 40.0, 40.0),
        square(10.0, 10.0, 46.0, 46.0),
        square(16.0, 16.0, 52.0, 52.0),
    ];
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("overlap2", &overlap, rule, w, h);
        check_legacy("overlap2", &overlap, rule, w, h);
    }
    check_exact("overlap3", &triple, FillRule::Winding, w, h);
    check_legacy("overlap3", &triple, FillRule::Winding, w, h);

    // 退化：零面积（单点 / 两点 / 全共线）。
    let degenerate = vec![
        vec![Point::new(5.0, 5.0)],
        vec![Point::new(0.0, 0.0), Point::new(30.0, 30.0)],
        vec![
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
            Point::new(4.0, 4.0),
        ],
    ];
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("degenerate", &degenerate, rule, w, h);
        check_legacy("degenerate", &degenerate, rule, w, h);
    }

    // 重合边：同一轮廓正反各一遍（相互抵消）、同向两遍（winding=2）。
    let a = square(8.0, 8.0, 40.0, 40.0);
    let mut b = a.clone();
    b.reverse();
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("coincident_both", &[a.clone(), b.clone()], rule, w, h);
        check_exact("coincident_same", &[a.clone(), a.clone()], rule, w, h);
        check_legacy("coincident_both", &[a.clone(), b.clone()], rule, w, h);
    }

    // 顶点落在行带边界（整数 y）上。
    let grid_aligned = vec![
        Point::new(8.0, 8.0),
        Point::new(56.0, 8.0),
        Point::new(56.0, 24.0),
        Point::new(20.0, 24.0),
        Point::new(20.0, 40.0),
        Point::new(56.0, 40.0),
        Point::new(56.0, 56.0),
        Point::new(8.0, 56.0),
    ];
    check_exact("grid_vertices", core::slice::from_ref(&grid_aligned), FillRule::Winding, w, h);
    check_legacy("grid_vertices", &[grid_aligned], FillRule::Winding, w, h);

    // 轮廓超画布（四方向）+ 半超界。
    let outside = vec![
        square(-40.0, -40.0, -2.0, -2.0),
        square(66.0, 66.0, 104.0, 104.0),
        square(-40.0, -40.0, 100.0, 100.0),
        square(-10.0, 20.0, 10.0, 44.0),
    ];
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("outside", &outside, rule, w, h);
        check_legacy("outside", &outside, rule, w, h);
    }

    // 巨大坐标（1e6 量级）：内部必须恰为满覆盖。
    let huge = vec![vec![
        Point::new(-1.0e6, -1.0e6),
        Point::new(2.0e6, -1.0e6),
        Point::new(2.0e6, 2.0e6),
        Point::new(-1.0e6, 2.0e6),
    ]];
    check_exact("huge", &huge, FillRule::Winding, w, h);

    // 极细长条（宽 0.01px，近水平）。
    let sliver = vec![vec![
        Point::new(0.0, 31.0),
        Point::new(64.0, 31.02),
        Point::new(64.0, 31.03),
        Point::new(0.0, 31.01),
    ]];
    check_exact("sliver", &sliver, FillRule::Winding, w, h);

    // 近水平锯齿：一条长边横跨整行并穿过上百条边（旧实现子带爆炸的极端形态）。
    let mut saw = Vec::with_capacity(200);
    for i in 0..100 {
        let x = i as f32 * 0.64;
        saw.push(Point::new(x, 20.0));
        saw.push(Point::new(x + 0.32, 21.0));
    }
    for rule in [FillRule::Winding, FillRule::EvenOdd] {
        check_exact("sawtooth", &[saw.clone()], rule, w, h);
        check_legacy("sawtooth", &[saw.clone()], rule, w, h);
    }

    // 重复顶点（零长度边）。
    let dup = vec![vec![
        Point::new(10.0, 10.0),
        Point::new(10.0, 10.0),
        Point::new(50.0, 10.0),
        Point::new(50.0, 50.0),
        Point::new(50.0, 50.0),
        Point::new(10.0, 50.0),
    ]];
    check_exact("dup_vertices", &dup, FillRule::Winding, w, h);
    check_legacy("dup_vertices", &dup, FillRule::Winding, w, h);
}

#[test]
fn random_polygons() {
    const SEED: u64 = 0x5EED_1234_ABCD_0001;
    let (w, h) = (96u32, 72u32);
    let mut cases = 0usize;
    let mut worst_exact = 0.0f64;
    let mut worst_legacy = 0.0f64;

    // 200 组随机多边形，两规则交替。
    for idx in 0..200u64 {
        let c = random_contour(SEED, idx, 96.0);
        let rule = if idx % 2 == 0 { FillRule::Winding } else { FillRule::EvenOdd };
        worst_exact = worst_exact.max(check_exact("rand_poly", &[c], rule, w, h));
        cases += 1;
    }
    // 40 组随机自交星。
    for idx in 0..40u64 {
        let c = random_star(SEED ^ 0x99, idx, 96.0);
        let rule = if idx % 2 == 0 { FillRule::EvenOdd } else { FillRule::Winding };
        worst_exact = worst_exact.max(check_exact("rand_star", &[c], rule, w, h));
        cases += 1;
    }
    // 10 组三轮廓随机组合 × 两规则。
    for idx in 0..10u64 {
        let cs: Vec<Vec<Point>> =
            (0..3).map(|j| random_contour(SEED ^ 0x1234, idx * 8 + j, 96.0)).collect();
        for rule in [FillRule::Winding, FillRule::EvenOdd] {
            worst_exact = worst_exact.max(check_exact("rand_multi", &cs, rule, w, h));
            cases += 1;
        }
    }
    // 旧实现回归对拍取子集（结构对拍已由上面 260 例覆盖）。
    for idx in (0..200u64).step_by(7) {
        let c = random_contour(SEED, idx, 96.0);
        let rule = if idx % 2 == 0 { FillRule::Winding } else { FillRule::EvenOdd };
        worst_legacy = worst_legacy.max(check_legacy("rand_poly", &[c], rule, w, h));
    }

    println!(
        "== 随机组 {cases} 例：|Δexact|max={worst_exact:.3e} |Δlegacy|max={worst_legacy:.3e} =="
    );
    assert!(cases >= 200, "随机用例数不足：{cases}");
}

/// 整数平移等价性：同一形状平移后同位置覆盖率必须同步平移。
///
/// 注意这里的误差主要来自**输入**：`p.x + 7` 在 f32 里会重新舍入
/// （66 px 处 ulp = 7.6e-6），所以平移后的顶点并不是原顶点的精确平移。
/// 实测 |Δ|max ≈ 1.9e-6，约为 ulp(66 px) 的 1/4，说明累加过程本身与平移无关
/// （内部列区间加、`S(X, t)` 原函数都是平移等变的）；阈值仍按 2e-6 判定。
#[test]
fn integer_translation_equivariance() {
    const SEED: u64 = 0x1234_5678_9ABC_DEF0;
    let (w, h) = (80u32, 64u32);
    let (dx, dy) = (7i32, 5i32);
    let mut worst = 0.0f64;

    for idx in 0..30u64 {
        let c = random_contour(SEED, idx, 60.0);
        let base = rasterize(core::slice::from_ref(&c), FillRule::Winding, w, h);
        let moved: Vec<Point> =
            c.iter().map(|p| Point::new(p.x + dx as f32, p.y + dy as f32)).collect();
        let shifted = rasterize(&[moved], FillRule::Winding, w, h);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (sx, sy) = (x + dx, y + dy);
                let expect = if sx >= 0 && sy >= 0 && (sx as u32) < w && (sy as u32) < h {
                    shifted.get(sx as u32, sy as u32)
                } else {
                    0.0
                };
                let d = (base.get(x as u32, y as u32) as f64 - expect as f64).abs();
                assert!(d <= 2e-6, "平移等价破坏 #{idx} @({x},{y})：Δ={d:e}");
                worst = worst.max(d);
            }
        }
    }
    println!("== 平移等价 |Δ|max={worst:.3e} ==");
}
