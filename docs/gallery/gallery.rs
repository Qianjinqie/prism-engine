//! gallery —— README 门面图 / 「三分钟看见引擎真实画质」的最小可复现示例。
//!
//! 只用 `prism-rt` 的公开 API（路径追踪 + Disney BSDF + 曲线图元）
//! 与 `prism-math` 的 ACES 输出变换，不碰 CLI / DSL，零第三方依赖。
//!
//! 本文件放在 `docs/gallery/`（不随分发包），复现时拷进示例目录再运行：
//! ```bash
//! cp docs/gallery/gallery.rs crates/prism-rt/examples/gallery.rs
//! cargo run -p prism-rt --release --example gallery -- 1200 675 256
//! ```
//! 参数依次为宽 / 高 / 每像素样本数（缺省 960 540 256）。
//! 输出 `gallery.png`（确定性：同参数两次运行逐字节一致）。

use std::sync::Arc;

use prism_core::node::EvalCtx;
use prism_gpu::Framebuffer;
use prism_math::aces::aces_fitted;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    render_pixel, Cylinder, Light, MaterialKind, RenderParams, Revolution, Scene, Sphere, Torus,
};
use prism_rt::bsdf::Bsdf;
use prism_math::Vec3;

/// 线性 → sRGB OETF（IEC 61966-2-1 分段式）。
fn srgb_oetf(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn build_scene() -> Scene {
    // ---- 材质 ----------------------------------------------------------
    // 暗色高光地台：粗糙度 0.18 的哑光电介质 —— 会映出物体与灯的倒影，
    // 是判断「这是真路径追踪」最直观的证据。
    let floor = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.105, 0.100, 0.098),
        0.0,
        0.18,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    // 背景墙：近黑，压低画面外围，把注意力逼回主体。
    let wall = MaterialKind::Lambert { albedo: Vec3::new(0.048, 0.052, 0.060) };

    // 磨砂金（metalness=1，粗糙度 0.30：把单点高光铺开成柔和金属渐变）。
    let gold = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.98, 0.68, 0.26),
        1.0,
        0.30,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    // 象牙瓷（亮、微透：给画面一个高价值点，同时不引入玻璃的强方差）。
    let porcelain = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.90, 0.88, 0.84),
        0.0,
        0.16,
        1.5,
        0.0,
        0.12,
        Vec3::splat(0.08),
    ));
    // 朱砂釉（半哑光陶瓷）。
    let vermilion = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.72, 0.11, 0.07),
        0.0,
        0.28,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));
    // 拉丝钢（圆环）。
    let steel = MaterialKind::Pbr(Bsdf::new(
        Vec3::new(0.62, 0.64, 0.68),
        1.0,
        0.32,
        1.5,
        0.0,
        0.0,
        Vec3::splat(0.3),
    ));

    // ---- 布光 ----------------------------------------------------------
    // 大半径球灯 = 软阴影 + 低方差；位置推在画框外，画面里不漏发光体。
    let key_pos = Vec3::new(-6.6, 6.8, 3.2);
    let key_rad = Vec3::new(150.0, 126.0, 96.0); // 暖主光
    let rim_pos = Vec3::new(4.0, 4.6, -6.4);
    let rim_rad = Vec3::new(84.0, 92.0, 104.0); // 略偏冷的中性轮廓光（过蓝会把金球染青）
    let kick_pos = Vec3::new(-1.1, 1.7, 6.1);
    let kick_rad = Vec3::new(26.0, 11.0, 5.0); // 越肩暖补光（勾正面）

    Scene {
        boxes: vec![
            // 地台
            (
                prism_rt::integrator::Aabb::new(
                    Vec3::new(-24.0, -0.4, -24.0),
                    Vec3::new(24.0, 0.0, 24.0),
                ),
                floor,
            ),
            // 背墙
            (
                prism_rt::integrator::Aabb::new(
                    Vec3::new(-24.0, 0.0, -8.0),
                    Vec3::new(24.0, 16.0, -7.8),
                ),
                wall,
            ),
        ],
        spheres: vec![
            // 金球（主体，略偏左，高度压出重量感）
            (Sphere { center: Vec3::new(-0.86, 0.78, 0.0), radius: 0.78 }, gold),
            // 瓷球（压在金球右前方，制造前后层次）
            (Sphere { center: Vec3::new(0.06, 0.50, 0.90), radius: 0.50 }, porcelain),
            // 朱砂球（右后，收住右侧）
            (Sphere { center: Vec3::new(1.30, 0.40, -0.22), radius: 0.40 }, vermilion),
            // 光源几何（画框外；Emissive 与 Light::Sphere 必须成对注册）
            (
                Sphere { center: key_pos, radius: 1.00 },
                MaterialKind::Emissive { radiance: key_rad },
            ),
            (
                Sphere { center: rim_pos, radius: 0.70 },
                MaterialKind::Emissive { radiance: rim_rad },
            ),
            (
                Sphere { center: kick_pos, radius: 0.55 },
                MaterialKind::Emissive { radiance: kick_rad },
            ),
        ],
        lights: vec![
            Light::Sphere { center: key_pos, radius: 1.00, radiance: key_rad },
            Light::Sphere { center: rim_pos, radius: 0.70, radiance: rim_rad },
            Light::Sphere { center: kick_pos, radius: 0.55, radiance: kick_rad },
        ],
        // 钢环：四次方程 Ferrari 闭式求交（曲线图元族），平放在前景左侧。
        toruses: vec![(
            Torus { center: Vec3::new(-1.62, 0.145, 0.72), major: 0.46, minor: 0.115 },
            steel,
        )],
        // 暗石圆柱：压住右后方，给画面一条竖直终止边。
        cylinders: vec![(
            Cylinder { center: Vec3::new(2.34, 0.92, -2.20), radius: 0.34, half_height: 0.92 },
            MaterialKind::Lambert { albedo: Vec3::new(0.26, 0.245, 0.23) },
        )],
        cones: vec![],
        revolutions: Vec::<(Revolution, MaterialKind)>::new(),
        medium: None,
        paraboloids: vec![],
        meshes: Vec::new(),
        planes: vec![],
        ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let w: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(960);
    let h: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(540);
    let spp: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    let scene = Arc::new(build_scene());
    let cam = Arc::new(Camera::new(
        Vec3::new(-0.08, 1.20, 4.75),
        Vec3::new(-0.04, 0.72, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        30.0,
        0.05, // 光圈：轻微景深
        4.85,
        0.0,
        1.0 / 48.0,
    ));
    // 间接光萤火钳制 6×中灰：出图档的常规做法（抑制小灯造成的孤立亮点）。
    // 需要「物理参考值」时把它调大或置 None —— 本示例只做展示，不做度量。
    let params = Arc::new(RenderParams {
        clamp_indirect: Some(8.0),
        ..RenderParams::new(spp, 10)
    });
    let ctx = EvalCtx::new(20261028, 0, 0);

    let res = (w, h);
    let mut linear = vec![[0.0f32; 3]; (w as usize) * (h as usize)];

    // 行带并行（std::thread，零第三方依赖）：把输出缓冲按行切成互斥块，
    // 每线程只写自己那块 ⇒ 无共享可变状态，与单线程结果逐位一致。
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(h as usize)
        .max(1);
    let w_us = w as usize;
    let band = (h as usize).div_ceil(threads);
    let ctx_ref = &ctx;

    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for (band_idx, chunk) in linear.chunks_mut(band * w_us).enumerate() {
            let scene = &scene;
            let cam = &cam;
            let params = &params;
            s.spawn(move || {
                for (local_y, row) in chunk.chunks_mut(w_us).enumerate() {
                    let y = (band_idx * band + local_y) as u32;
                    for (x, px) in row.iter_mut().enumerate() {
                        let c = render_pixel(
                            scene,
                            cam,
                            ctx_ref,
                            (x as u32, y),
                            res,
                            params,
                        );
                        *px = [c[0], c[1], c[2]];
                    }
                }
            });
        }
    });
    let elapsed = t0.elapsed();

    // 线性 HDR → ACES 输出变换 → sRGB OETF → 8bit。
    let mut fb = Framebuffer::new(w, h);
    for (i, c) in linear.iter().enumerate() {
        let toned = aces_fitted([c[0], c[1], c[2]]);
        for k in 0..3 {
            let v = srgb_oetf(toned[k].clamp(0.0, 1.0));
            fb.pixels[i * 4 + k] = (v * 255.0 + 0.5) as u8;
        }
        fb.pixels[i * 4 + 3] = 255;
    }

    let png = prism_io::encode_png(&fb);
    std::fs::write("gallery.png", &png).expect("写 gallery.png 失败");

    println!(
        "gallery.png  {w}x{h}  spp={spp}  线程={threads}  耗时={:.2}s  {} 字节",
        elapsed.as_secs_f64(),
        png.len()
    );
}
