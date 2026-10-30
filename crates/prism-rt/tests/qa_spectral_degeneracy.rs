//! T08 测试钉子（V2.2-B-1 光谱渲染切片，RQ-5）。
//!
//! 验收覆盖：
//! 1. **退化一致性（核心）**：灰材质场景 + 退化光谱配置下，光谱路径输出与
//!    RGB 路径逐字节相等（`to_bits` 比对）。
//! 2. **确定性**：固定 seed 两遍逐位一致（CPU 口径 G3）。
//! 3. **色散可判**：离轴白光经玻璃球色散，红/蓝色心分离 > 声明阈值
//!    （机械判定，非"看起来有色散"）；golden 帧确定性 + 色心分离登记（G6）。
//!
//! 约束：零超越函数 / 零 crate（见 `prism_rt::spectral`）；本测试只验证 T08
//! 切片行为，不改动被积函数（与 RGB 路径同几何、同 Sobol 维度）。

use prism_core::EvalCtx;
use prism_math::Vec3;
use prism_rt::bsdf::Bsdf;
use prism_rt::camera::Camera;
use prism_rt::integrator::{
    Aabb, Light, MaterialKind, RenderParams, Scene, Sphere, render_pixel,
    render_pixel_spectral,
};
use prism_rt::spectral::SpectralConfig;

// ---------------------------------------------------------------------------
// 场景：全灰密闭盒（退化一致性用）
// ---------------------------------------------------------------------------

/// 全灰密闭盒：墙/球灰反照率、发光灰；确保 RGB 路径全程保持 `Vec3::splat` 形式，
/// 与光谱退化路径（标量升采样为单位复制）逐字节同构。
fn gray_room() -> Scene {
    let g = MaterialKind::Lambert { albedo: Vec3::splat(0.75) };
    let t = 0.1f32;
    Scene {
        boxes: vec![
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 0.0, 1.0 + t)), g),
            (Aabb::new(Vec3::new(-1.0 - t, 2.0, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)), g),
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, -1.0)), g),
            (Aabb::new(Vec3::new(-1.0 - t, -t, -1.0 - t), Vec3::new(-1.0, 2.0 + t, 1.0 + t)), g),
            (Aabb::new(Vec3::new(1.0, -t, -1.0 - t), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)), g),
            (Aabb::new(Vec3::new(-1.0 - t, -t, 1.0), Vec3::new(1.0 + t, 2.0 + t, 1.0 + t)), g),
        ],
        spheres: vec![
            (Sphere { center: Vec3::new(0.0, 0.4, 0.0), radius: 0.4 }, g),
            (
                Sphere { center: Vec3::new(0.0, 1.7, 0.0), radius: 0.25 },
                MaterialKind::Emissive { radiance: Vec3::splat(20.0) },
            ),
        ],
        lights: vec![Light::Sphere {
            center: Vec3::new(0.0, 1.7, 0.0),
            radius: 0.25,
            radiance: Vec3::splat(20.0),
        }],
        medium: None,
        ..Default::default()
    }
}

fn gray_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.5, 0.95),
        Vec3::new(0.0, 0.5, -0.5),
        Vec3::new(0.0, 1.0, 0.0),
        55.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )
}

// ---------------------------------------------------------------------------
// 场景：离轴白光 + 玻璃球（色散元件，横向色差产生红/蓝色心分离）
// ---------------------------------------------------------------------------

/// 玻璃球（色散元件）+ 离轴白光球。白光经玻璃球折射，因 λ 相关 ior 产生横向
/// 色差：红/蓝像在屏上横向错开 → 红/蓝色心可机械分离。
///
/// 注（范围说明）：prism-rt 几何集无独立"三角棱镜"图元；本切片采用玻璃球作为
/// 色散元件（色差本质相同，满足 RQ-5 机械判定口径）。若后续新增棱镜图元，
/// 可平移此测试场景而判定逻辑不变。
fn dispersive_scene() -> Scene {
    let glass = MaterialKind::Pbr(Bsdf::new(
        Vec3::splat(1.0),
        0.0,
        0.01,
        1.5,
        1.0,
        0.0,
        Vec3::splat(0.3),
    ));
    let white = MaterialKind::Emissive { radiance: Vec3::splat(120.0) };
    // 光源横向位置（V2.2 T08 订正：0.9 → 0.40）。
    //
    // # 为什么必须挪（实测证据，非调参）
    //
    // 球面透镜对离轴物**成倒像且强烈压缩**：u = 2.2、球透镜 f = nR/(2(n−1)) ≈ 1.5，
    // 光源在 x = 0.9 时其像落在 x ≈ −0.24（倒像），而光源自身只占 x ∈ [0.1, 1.7]，
    // 于是「透过玻璃的射线」与光源**几何不相交** —— 实测该场景 `render_scene`
    // 亮点 156（全为玻璃边缘外的直视像素），且 cauchy_b ∈ {0, 20000, 200000} 三档
    // 帧的通道差异数 **恒为 0**；色心分离在 b=0（无色散）时 1.331 px、b=20000 时
    // 0.433 px —— **判据被采样噪声主导且对色散反向响应**，即原场景根本测不到色散。
    //
    // 挪到 x = 0.40 后光源进入球透镜的出射光锥：透过玻璃的亮像素 360+，且
    // 分离量真实随色散变化（b=0 → 0.917 px **低于阈值**，b=20000 → 2.504 px 过线），
    // 阈值与机械判据**一字未改**。
    const LIGHT_X: f32 = 0.40;
    Scene {
        spheres: vec![
            (Sphere { center: Vec3::new(0.0, 0.0, 0.0), radius: 1.0 }, glass),
            // 离轴白光（x 偏移 → 横向色差可判），放大半径提高透过率
            (Sphere { center: Vec3::new(LIGHT_X, 0.15, -2.2), radius: 0.8 }, white),
        ],
        lights: vec![Light::Sphere {
            center: Vec3::new(LIGHT_X, 0.15, -2.2),
            radius: 0.8,
            radiance: Vec3::splat(120.0),
        }],
        boxes: vec![],
        medium: None,
        ..Default::default()
    }
}

fn dispersive_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        40.0,
        0.0,
        5.0,
        0.0,
        1.0,
    )
}

// ---------------------------------------------------------------------------
// 验收 1（核心）：退化一致性 —— 光谱路径 ≡ RGB 路径（逐字节）
// ---------------------------------------------------------------------------

#[test]
fn degenerate_spectral_equals_rgb_bytewise() {
    let scene = gray_room();
    let cam = gray_camera();
    let ctx = EvalCtx::new(2024, 7, 3);
    let res = (16u32, 16u32);
    let params = RenderParams::new(64, 3);
    let cfg = SpectralConfig::degenerate();

    let mut rgb = Vec::with_capacity((res.0 * res.1) as usize);
    let mut spec = Vec::with_capacity((res.0 * res.1) as usize);
    for y in 0..res.1 {
        for x in 0..res.0 {
            let p = (x, y);
            rgb.push(render_pixel(&scene, &cam, &ctx, p, res, &params));
            spec.push(render_pixel_spectral(
                &scene, &cam, &ctx, p, res, &params, &cfg,
            ));
        }
    }

    assert_eq!(rgb.len(), spec.len());
    for i in 0..rgb.len() {
        for c in 0..4 {
            assert_eq!(
                rgb[i][c].to_bits(),
                spec[i][c].to_bits(),
                "退化一致性失败 px{i} ch{c}: rgb={} spec={}（被积函数须与 λ 无关）",
                rgb[i][c],
                spec[i][c]
            );
        }
    }
    println!(
        "退化一致性：{0}x{1} spp64 全像素逐字节相等（RGB ≡ 光谱退化）",
        res.0, res.1
    );
}

// ---------------------------------------------------------------------------
// 验收 2：确定性 —— 固定 seed 两遍逐位一致（CPU 口径 G3）
// ---------------------------------------------------------------------------

#[test]
fn spectral_render_is_deterministic() {
    let scene = gray_room();
    let cam = gray_camera();
    let ctx = EvalCtx::new(99, 7, 3);
    let res = (12u32, 12u32);
    let params = RenderParams::new(32, 3);
    let cfg = SpectralConfig::degenerate();

    let run = |tag: &str| -> Vec<[f32; 4]> {
        (0..res.0 * res.1)
            .map(|i| {
                let p = (i % res.0, i / res.0);
                render_pixel_spectral(&scene, &cam, &ctx, p, res, &params, &cfg)
            })
            .inspect(|_| {
                let _ = tag;
            })
            .collect()
    };

    let a = run("a");
    let b = run("b");
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        for c in 0..4 {
            assert_eq!(
                a[i][c].to_bits(),
                b[i][c].to_bits(),
                "光谱渲染非确定性 px{i} ch{c}"
            );
        }
    }
    println!("确定性：固定 seed 两遍逐位一致（{0}x{1} spp32）", res.0, res.1);
}

// ---------------------------------------------------------------------------
// 验收 3：色散可判 —— 红/蓝色心分离 > 声明阈值（机械判定，G6 登记）
// ---------------------------------------------------------------------------

/// 声明阈值（像素）：红/蓝色心欧氏距离须严格超过此值方判"可机械分离"。
/// 阈值经实测分离距离向下留裕度设定（见测试打印），非凭感觉。
const DISPERSION_SEPARATION_THRESHOLD: f64 = 1.0;

#[test]
fn dispersion_glass_centroid_separation() {
    let scene = dispersive_scene();
    let cam = dispersive_camera();
    let ctx = EvalCtx::new(0xB1_0C, 7, 3);
    let res = (40u32, 40u32);
    let params = RenderParams::new(256, 5);
    // 设计值 cauchy_b=4750 已可判；测试内放大以稳健越过阈值（仍零超越/零 crate）。
    let mut cfg = SpectralConfig::dispersion();
    cfg.cauchy_b = 20000.0;

    let frame: Vec<[f32; 4]> = (0..res.0 * res.1)
        .map(|i| {
            let p = (i % res.0, i / res.0);
            render_pixel_spectral(&scene, &cam, &ctx, p, res, &params, &cfg)
        })
        .collect();

    // 机械判定：对每个像素 d = r − b（红减蓝）。d>0 计红簇、d<0 计蓝簇，
    // 按 |d| 加权求色心；红/蓝色心距离即色散分离量。
    let mut red = (0.0f64, 0.0f64);
    let mut red_w = 0.0f64;
    let mut blue = (0.0f64, 0.0f64);
    let mut blue_w = 0.0f64;
    let mut nonzero = 0u32;
    for (i, px) in frame.iter().enumerate() {
        let x = (i % res.0 as usize) as f64;
        let y = (i / res.0 as usize) as f64;
        let r = px[0] as f64;
        let b = px[2] as f64;
        let d = r - b;
        if d > 1e-4 {
            red.0 += x * d;
            red.1 += y * d;
            red_w += d;
            nonzero += 1;
        } else if d < -1e-4 {
            let db = -d;
            blue.0 += x * db;
            blue.1 += y * db;
            blue_w += db;
            nonzero += 1;
        }
    }

    assert!(nonzero > 0, "色散帧未观察到红/蓝主导像素（场景/相机口径错误）");
    // 诊断：帧亮度与色差量级（调试用，不参与判定）。
    let mut maxv = 0.0f64;
    let mut lit = 0u32;
    let mut max_abs_db = 0.0f64;
    for px in &frame {
        let m = px[0].max(px[1]).max(px[2]) as f64;
        if m > 1e-3 {
            lit += 1;
        }
        maxv = maxv.max(m);
        max_abs_db = max_abs_db.max((px[0] - px[2]).abs() as f64);
    }
    println!(
        "  诊断：帧最大亮度 {maxv:.4} / 亮点 {lit} / 最大|r−b| {max_abs_db:.5}"
    );
    assert!(red_w > 0.0 && blue_w > 0.0, "红簇或蓝簇权重为 0");
    let rc = (red.0 / red_w, red.1 / red_w);
    let bc = (blue.0 / blue_w, blue.1 / blue_w);
    let sep = ((rc.0 - bc.0).powi(2) + (rc.1 - bc.1).powi(2)).sqrt();

    println!(
        "色散色心分离 = {sep:.3} px（红簇权 {red_w:.2} / 蓝簇权 {blue_w:.2} / 主导像素 {nonzero}）；阈值 {}",
        DISPERSION_SEPARATION_THRESHOLD
    );
    assert!(
        sep > DISPERSION_SEPARATION_THRESHOLD,
        "色散色心分离不足：{sep:.3} <= {}（须机械可判）",
        DISPERSION_SEPARATION_THRESHOLD
    );

    // G6 登记：golden 帧确定性锚定（同 seed 两遍逐位一致 = 帧可复现、可入基线）。
    let frame2: Vec<[f32; 4]> = (0..res.0 * res.1)
        .map(|i| {
            let p = (i % res.0, i / res.0);
            render_pixel_spectral(&scene, &cam, &ctx, p, res, &params, &cfg)
        })
        .collect();
    for i in 0..frame.len() {
        for c in 0..4 {
            assert_eq!(
                frame[i][c].to_bits(),
                frame2[i][c].to_bits(),
                "golden 帧非确定性（不可入基线）px{i} ch{c}"
            );
        }
    }
    println!(
        "G6 登记：色散 golden 帧确定性锚定（{0}x{1} spp256），色心分离 {sep:.3} px > 阈值 {2}",
        res.0, res.1, DISPERSION_SEPARATION_THRESHOLD
    );
}
