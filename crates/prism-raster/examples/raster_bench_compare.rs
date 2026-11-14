//! **B.1 性能对照基准**：`rasterize`（每帧新分配输出缓冲）vs
//! `rasterize_into`（复用同一缓冲）——5 个验收用例的前后数字。
//!
//! ```text
//! cargo run --release -p prism-raster --example raster_bench_compare
//! ```
//!
//! 环境变量：`RASTER_BENCH_ITERS`（默认 21，中位数）、`RASTER_BENCH_ONLY`（子串过滤）。
//!
//! 口径（动手前定死，绝不事后改尺子）：
//! - 两路径对**同一几何**必须产出**逐字节一致**的覆盖率（本基准逐位断言），
//!   否则优化无意义——性能数字只属于「语义不变」的优化；
//! - `before` = `rasterize`（`vec![0.0; w*h]` 每次新分配 + 首次触碰缺页）；
//! - `after` = `rasterize_into`（clear+resize 复用已驻留页，逐帧路径的真实形态）；
//! - 每个用例各跑 `iters` 次取中位数，预热 1 次不计；
//! - `speedup = before / after`。
//!
//! 与 `raster_bench.rs` 共用固定几何（同标定常数），造形的 sin/cos 是自研
//! 多项式（仅造输入，不参与光栅数学；crate 内禁 libm）。

use std::hint::black_box;
use std::time::Instant;

use prism_raster::{rasterize, rasterize_into, Coverage, FillRule, Point};

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

fn ring_hole(cx: f32, cy: f32, outer: f32, inner: f32) -> Vec<Vec<Point>> {
    let outer_ring = vec![
        Point::new(cx - outer, cy - outer),
        Point::new(cx - outer, cy + outer),
        Point::new(cx + outer, cy + outer),
        Point::new(cx + outer, cy - outer),
    ];
    let inner_ring = vec![
        Point::new(cx - inner, cy - inner),
        Point::new(cx + inner, cy - inner),
        Point::new(cx + inner, cy + inner),
        Point::new(cx - inner, cy + inner),
    ];
    vec![outer_ring, inner_ring]
}

struct Case {
    name: &'static str,
    contours: Vec<Vec<Point>>,
    rule: FillRule,
    width: u32,
    height: u32,
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn main() {
    let iters: usize = std::env::var("RASTER_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);
    let filter = std::env::var("RASTER_BENCH_ONLY").unwrap_or_default();

    let cases = vec![
        {
            let (w, h) = (1920u32, 1080u32);
            let r = 0.3711 * h as f32;
            Case { name: "circle64_1080p", contours: circle(64, w as f32 * 0.5, h as f32 * 0.5, r), rule: FillRule::Winding, width: w, height: h }
        },
        {
            let (w, h) = (1920u32, 1080u32);
            let r = 0.470 * w as f32;
            Case { name: "star512_1080p", contours: star(512, w as f32 * 0.5, h as f32 * 0.5, r, 0.42), rule: FillRule::Winding, width: w, height: h }
        },
        {
            let (w, h) = (4096u32, 4096u32);
            let r = 0.4647 * w as f32;
            Case { name: "star512_4k", contours: star(512, w as f32 * 0.5, h as f32 * 0.5, r, 0.42), rule: FillRule::Winding, width: w, height: h }
        },
        {
            let (w, h) = (4096u32, 4096u32);
            let r = 0.4647 * w as f32;
            Case { name: "star2048_4k", contours: star(2048, w as f32 * 0.5, h as f32 * 0.5, r, 0.42), rule: FillRule::Winding, width: w, height: h }
        },
        {
            let (w, h) = (2048u32, 2048u32);
            Case { name: "ring_hole_2k", contours: ring_hole(w as f32 * 0.5, h as f32 * 0.5, 700.0, 260.0), rule: FillRule::Winding, width: w, height: h }
        },
    ];

    println!(
        "{:<18} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "case", "before_ms", "after_ms", "speedup", "nonzero_px", "area_sum"
    );
    println!("{}", "-".repeat(80));
    for case in &cases {
        if !filter.is_empty() && !case.name.contains(&filter) {
            continue;
        }
        let w = case.width;
        let h = case.height;
        let rule = case.rule;

        // 预热 + 取规范输出（逐字节标尺）。
        let reference = rasterize(&case.contours, rule, w, h);
        let nonzero = reference.data.iter().filter(|v| **v != 0.0).count();
        let area: f64 = reference.data.iter().map(|v| *v as f64).sum();

        // 复用缓冲预热（首次触碰页，之后常驻）。
        let mut reused = Coverage::new(w, h);
        rasterize_into(&case.contours, rule, w, h, &mut reused);

        // before：rasterize 每次新分配。
        let mut before = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let cov = rasterize(&case.contours, rule, w, h);
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            black_box(&cov);
            before.push(dt);
        }
        // after：rasterize_into 复用缓冲；每次顺便核对逐字节一致。
        let mut after = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            rasterize_into(&case.contours, rule, w, h, &mut reused);
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            black_box(&reused);
            after.push(dt);
        }
        // 硬判据：两路径输出逐字节一致（性能数字仅对语义不变优化有效）。
        assert_eq!(reused.data.len(), reference.data.len(), "[{}] 复用路径长度", case.name);
        for (i, (a, b)) in reference.data.iter().zip(reused.data.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "[{}] 复用路径逐字节一致 @{i}", case.name);
        }

        let mb = median(&mut before);
        let ma = median(&mut after);
        println!(
            "{:<18} {:>12.3} {:>12.3} {:>9.2}x {:>12} {:>12.1}",
            case.name, mb, ma, mb / ma, nonzero, area
        );
    }
}
