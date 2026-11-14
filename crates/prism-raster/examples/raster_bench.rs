//! 零依赖扫描线光栅基准（`std::time`，无 criterion / 无第三方 crate）。
//!
//! # 跑法
//!
//! ```text
//! cargo run --release -p prism-raster --example raster_bench
//! ```
//!
//! 可选环境变量：
//! - `RASTER_BENCH_ITERS`：每个用例的计时迭代数（默认 5；预热 1 次不计）。
//! - `RASTER_BENCH_ONLY`：子串过滤，只跑名字含该子串的用例。
//!
//! 输出四列：用例名 / 中位耗时 / 非零像素 / 面积和（守恒自检用）。
//!
//! 用例（与性能验收表一一对应）：
//! - `circle64_1080p`：64 段圆 @1920×1080
//! - `star512_1080p`：512 段星 @1920×1080
//! - `star512_4k`：512 段星 @4096×4096
//! - `star2048_4k`：2048 段星 @4096×4096
//! - `ring_hole_2k`：外环 + 内环孔洞（反向绕行）@2048×2048

use std::time::Instant;

use prism_raster::{rasterize, Coverage, FillRule, Point};

/// 生成正 `n` 边形（圆的多边形逼近）。
fn circle(n: usize, cx: f32, cy: f32, r: f32) -> Vec<Vec<Point>> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f32) * (core::f32::consts::TAU / n as f32);
        let (s, c) = sin_cos(t);
        pts.push(Point::new(cx + r * c, cy + r * s));
    }
    vec![pts]
}

/// 星 = 半径交替 `n` 边形（外 `r`、内 `0.42r`），自交，用 `Winding` 规则填。
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

/// 环形：外方环 + 反向内方环（孔洞）。
fn ring_hole(cx: f32, cy: f32, outer: f32, inner: f32) -> Vec<Vec<Point>> {
    // 外环顺时针、内环逆时针：winding 在孔内相互抵消（真孔洞）。
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

/// 自研 sin/cos（crate 内禁用 libm；本文件仅造几何，用泰勒/表驱动足够）。
/// 这里用「半角递归 + 多项式」保证与平台无关的确定性——基准输入固定即可。
fn sin_cos(t: f32) -> (f32, f32) {
    // 归一化到 [-π, π]。
    let tau = core::f32::consts::TAU;
    let mut x = t;
    while x > core::f32::consts::PI {
        x -= tau;
    }
    while x < -core::f32::consts::PI {
        x += tau;
    }
    // 多项式（Abramowitz-Stegun 4.3.96 附近精度足够造形，不参与光栅数学）。
    let x2 = x * x;
    let s = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))));
    let c = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0 * (1.0 - x2 / 56.0)));
    (s, c)
}

struct Case {
    name: &'static str,
    contours: Vec<Vec<Point>>,
    rule: FillRule,
    width: u32,
    height: u32,
}

impl Case {
    fn run(&self) -> Coverage {
        rasterize(&self.contours, self.rule, self.width, self.height)
    }
}

fn main() {
    let iters: usize = std::env::var("RASTER_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let filter = std::env::var("RASTER_BENCH_ONLY").unwrap_or_default();

    let cases = vec![
        {
            let (w, h) = (1920u32, 1080u32);
            let r = 0.3711 * h as f32; // 标定到基线非零像素数 503 460
            Case {
                name: "circle64_1080p",
                contours: circle(64, w as f32 * 0.5, h as f32 * 0.5, r),
                rule: FillRule::Winding,
                width: w,
                height: h,
            }
        },
        {
            let (w, h) = (1920u32, 1080u32);
            let r = 0.470 * w as f32; // 标定到基线非零像素数 1 080 265
            Case {
                name: "star512_1080p",
                contours: star(512, w as f32 * 0.5, h as f32 * 0.5, r, 0.42),
                rule: FillRule::Winding,
                width: w,
                height: h,
            }
        },
        {
            let (w, h) = (4096u32, 4096u32);
            let r = 0.4647 * w as f32; // 标定到基线非零像素数 5 117 655
            Case {
                name: "star512_4k",
                contours: star(512, w as f32 * 0.5, h as f32 * 0.5, r, 0.42),
                rule: FillRule::Winding,
                width: w,
                height: h,
            }
        },
        {
            let (w, h) = (4096u32, 4096u32);
            let r = 0.4647 * w as f32; // 标定到基线非零像素数 6 121 379
            Case {
                name: "star2048_4k",
                contours: star(2048, w as f32 * 0.5, h as f32 * 0.5, r, 0.42),
                rule: FillRule::Winding,
                width: w,
                height: h,
            }
        },
        {
            let (w, h) = (2048u32, 2048u32);
            Case {
                name: "ring_hole_2k",
                contours: ring_hole(w as f32 * 0.5, h as f32 * 0.5, 700.0, 260.0),
                rule: FillRule::Winding,
                width: w,
                height: h,
            }
        },
    ];

    println!(
        "{:<18} {:>12} {:>12} {:>12} {:>16}",
        "case", "median_ms", "min_ms", "nonzero_px", "area_sum"
    );
    println!("{}", "-".repeat(74));
    for case in &cases {
        if !filter.is_empty() && !case.name.contains(&filter) {
            continue;
        }
        // 预热（分配 + 首次触碰页）。
        let warm = case.run();
        let nonzero = warm.data.iter().filter(|v| **v != 0.0).count();
        let area: f64 = warm.data.iter().map(|v| *v as f64).sum();

        let mut samples: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let cov = case.run();
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&cov);
            samples.push(dt);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = samples[samples.len() / 2];
        let min = samples[0];
        println!(
            "{:<18} {:>12.3} {:>12.3} {:>12} {:>16.1}",
            case.name, median, min, nonzero, area
        );
    }
}
