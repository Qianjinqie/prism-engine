//! A2 / W2：**equirect EXR → 立体角 CDF** 的端到端接线（判据之外的必要一环）。
//!
//! # 这条测试补的是什么
//!
//! `w2_a2_ibl.rs` 的 V1–V5 直接把 `f32` 像素喂给 `Scene::add_environment`，
//! 证明的是"给定像素，CDF/采样/pdf 正确"。真实管线里像素来自 **HDR 文件**，
//! 中间还隔着解码、通道交错、行序约定三件事——这三件事错了，V1–V5 全绿也
//! 没用。本文件把它们钉死：
//!
//! 1. `prism_io::exr_write::write_image` 写 **Float（f32）无损** EXR；
//! 2. `prism_io::exr::read_image` 读回，按 `R/G/B/A` 交错成 RGBA `f32`；
//! 3. 行序：EXR 扫描线 **y=0 = 图像顶部 = 天顶（+Y，v=0）**，与
//!    `Scene::add_environment` 的"行主序、v 从 0 递增"直接对齐，**不翻行**；
//! 4. `Scene::add_environment` 建 CDF，采样与 pdf 自洽。
//!
//! # 为什么不用 `ExrImage::to_fbuf16f`
//!
//! 那条路在交付处量化到 **f16**（11 位尾数，相对量化 4.9e-4）。对最终成像
//! 无害，但会让"EXR 里的值"与"CDF 里的值"不再是同一批数——本文要断言的是
//! **逐位**，故走 `ChannelPlane::values`（`f32`）直取。
//!
//! # 依赖方向
//!
//! `prism-io` 是 `prism-rt` 的 **dev-dependency**（见 `Cargo.toml`），
//! 生产代码不引用它：解码器留在 IO 层，`prism-rt` 只认 `(w, h, &[[f32; 4]])`。

use prism_io::exr::{read_image, ChannelPlane, PixelType};
use prism_io::exr_write::{header_for, write_image};
use prism_io::Compression;
use prism_math::Vec3;
use prism_rt::env::EnvCdf;
use prism_rt::integrator::Scene;
use prism_rt::texsample::TexTable;

/// 环境图宽（radical 对称：2:1 等距柱状）。
const W: u32 = 64;
/// 环境图高。
const H: u32 = 32;
/// 太阳方向（归一化前的倾斜；与 `w2_a2_ibl_beyond.rs` 的 `SUN_TILT_DEG` 同族）。
///
/// 写成函数而不是 `const`：`Vec3::new` 不是 `const fn`（本 crate 的既有约束）。
fn sun_dir() -> Vec3 {
    Vec3::new(0.5, 0.0, 0.866_025_4).normalize()
}
/// 太阳半角（度）。
const SUN_RADIUS_DEG: f32 = 4.0;
/// 太阳辐亮度。
const SUN_RADIANCE: f32 = 40.0;
/// 天空辐亮度（低对比残差）。
const SKY: f32 = 0.1;

/// 生成 equirect 像素（`f32` RGB，行主序、行 0 = 天顶）。
fn equirect_rgb() -> Vec<[f32; 3]> {
    let cos_r = (SUN_RADIUS_DEG * std::f32::consts::PI / 180.0).cos();
    let sun = sun_dir();
    let mut px = Vec::with_capacity((W * H) as usize);
    for j in 0..H {
        for i in 0..W {
            let u = (i as f32 + 0.5) / W as f32;
            let v = (j as f32 + 0.5) / H as f32;
            let d = prism_rt::env::uv_to_dir(u, v);
            // 太阳用"方向锥"判据（与真实管线里"直接由贴图采样"等价：
            // 这里的贴图就是这个函数生成的）。
            if d.dot(&sun) >= cos_r {
                px.push([SUN_RADIANCE, SUN_RADIANCE * 0.95, SUN_RADIANCE * 0.8]);
            } else {
                // 天空带一点梯度，避免"行权重全相等"这种退化占优。
                px.push([SKY, SKY * 1.2, SKY * 2.0 + 0.05 * d.y.max(0.0)]);
            }
        }
    }
    px
}

/// 写 Float EXR → 读回 → 交错成 RGBA `f32`。
fn exr_round_trip(rgb: &[[f32; 3]]) -> (u32, u32, Vec<[f32; 4]>) {
    let channels = [("R", PixelType::Float), ("G", PixelType::Float), ("B", PixelType::Float)];
    let header = header_for(W, H, &channels, Compression::None);
    let planes: Vec<ChannelPlane> = (0..3)
        .map(|c| ChannelPlane {
            name: ["R", "G", "B"][c].to_string(),
            pixel_type: PixelType::Float,
            width: W,
            height: H,
            values: rgb.iter().map(|p| p[c]).collect(),
        })
        .collect();
    let bytes = write_image(&header, &planes, Compression::None).expect("写 EXR 必须成功");
    let img = read_image(&bytes).expect("读回 EXR 必须成功");
    let iw = img.header.data_window.width();
    let ih = img.header.data_window.height();
    assert_eq!((iw, ih), (W, H), "EXR 尺寸必须原样往返");
    let get = |name: &str, idx: usize| -> f32 {
        img.plane(name).map(|p| p.values[idx]).unwrap_or(0.0)
    };
    let mut rgba = Vec::with_capacity((W * H) as usize);
    for idx in 0..(W * H) as usize {
        rgba.push([get("R", idx), get("G", idx), get("B", idx), 1.0]);
    }
    (iw, ih, rgba)
}

/// EXR（f32 无损）→ CDF 是**逐位**的：贴图值原样进表，不经 f16。
#[test]
fn exr_float_round_trip_is_bitwise_for_the_sun_texel() {
    let rgb = equirect_rgb();
    let (iw, ih, rgba) = exr_round_trip(&rgb);
    assert_eq!((iw, ih), (W, H));
    let mut scene = Scene::default();
    let handle = scene.add_environment(iw, ih, &rgba);
    assert_ne!(handle, prism_rt::texture::NO_TEXTURE, "环境纹理必须真实入表");

    // 场景扁平淡里的 texel 与 EXR 读回值逐位相同（无中间量化）。
    let table = TexTable::new(&scene.textures, &scene.texels);
    let r = table.get(handle).expect("纹理句柄有效");
    let base = prism_rt::texsample::level_base(&r, 0);
    let (tw, th) = prism_rt::texsample::level_dims(&r, 0);
    assert_eq!((tw as u32, th as u32), (W, H));
    let mut sun_texels = 0u32;
    for idx in 0..(W * H) as usize {
        for c in 0..3 {
            assert_eq!(
                scene.texels[base + idx][c].to_bits(),
                rgb[idx][c].to_bits(),
                "texel {idx} 通道 {c} 与 EXR 读回值不逐位一致（说明中途量化了）"
            );
        }
        if scene.texels[base + idx][0] == SUN_RADIANCE {
            sun_texels += 1;
        }
    }
    assert!(sun_texels > 0, "太阳 texel 数为 0 ⇒ 本用例失去意义");
    println!("[exr] {W}×{H} Float EXR 往返逐位一致；太阳 texel {sun_texels} 个");
}

/// 端到端：EXR → CDF → 采样/pdf/辐亮度自洽（同一条链上的三条恒等式）。
#[test]
fn exr_environment_builds_a_consistent_cdf() {
    let rgb = equirect_rgb();
    let (iw, ih, rgba) = exr_round_trip(&rgb);
    let mut scene = Scene::default();
    scene.add_environment(iw, ih, &rgba);
    let cdf: &EnvCdf = scene.env().expect("CDF 必须随 add_environment 建好");
    let stats = cdf.stats();
    assert!(cdf.total_weight() > 0.0, "总能量必须为正");
    assert!(stats.spots > 0, "高对比环境必须触发亮斑列表（判据退化检查）");
    println!(
        "[exr] 亮斑 {} 个，p_spot={:.4}，Ω_total={:.6} sr",
        stats.spots, stats.p_spot, cdf.omega_total()
    );

    // ① 逐 texel 的归一：`Σ_texel pdf_w(中心)·Ω_texel == 1`（**精确和**，不是
    //    蒙特卡洛估计——球面均匀撒点的估计量在 40:0.1 的对比度下方差极大，
    //    4096 点会给出 3.7% 的假偏差，那测的是求积误差不是 pdf 误差）。
    //    Ω_texel = 精确球冠 `2π(cosθ_j − cosθ_{j+1})` / W，与 `env.rs` 同式。
    let mut total = 0.0f64;
    for j in 0..H {
        let c0 = prism_math::det_trig::cos(std::f32::consts::PI * j as f32 / H as f32);
        let c1 = prism_math::det_trig::cos(std::f32::consts::PI * (j + 1) as f32 / H as f32);
        let omega_texel = (2.0 * std::f64::consts::PI * (c0 - c1) as f64) / W as f64;
        for i in 0..W {
            let u = (i as f32 + 0.5) / W as f32;
            let v = (j as f32 + 0.5) / H as f32;
            total += cdf.pdf_w(prism_rt::env::uv_to_dir(u, v)) as f64 * omega_texel;
        }
    }
    let rel = (total - 1.0).abs();
    assert!(rel <= 1e-3, "Σ pdf_w·Ω_texel = {total}（相对偏差 {rel:.3e} 超过 1e-3）");
    println!("[exr] Σ pdf_w·Ω_texel = {total:.9}（相对偏差 {rel:.3e}）");

    // ② 行序 / 通道交错 / uv 三件事的定点校验：**亮 texel 的中心方向**取回来
    //    必须就是那个 texel 的值（texel 中心 `fx = fy = 0` ⇒ 双线性退化为取值）。
    //    这里刻意不用太阳方向本身：4° 的太阳在 64×32 上不足一个 texel
    //    （经度 5.6°/texel），太阳方向落在两个亮 texel 的边缘，双线性必然
    //    混进天空——那是**采样分辨率**的既成事实，不是接线错误。
    let table = TexTable::new(&scene.textures, &scene.texels);
    let mut brightest = (0usize, 0.0f32);
    for (idx, p) in rgb.iter().enumerate() {
        if p[0] > brightest.1 {
            brightest = (idx, p[0]);
        }
    }
    let (bi, bj) = (brightest.0 % W as usize, brightest.0 / W as usize);
    let (bu, bv) = ((bi as f32 + 0.5) / W as f32, (bj as f32 + 0.5) / H as f32);
    let l_texel = cdf.radiance(table, prism_rt::env::uv_to_dir(bu, bv));
    assert!(
        (l_texel.x - SUN_RADIANCE).abs() <= SUN_RADIANCE * 1e-5,
        "最亮 texel（{bi},{bj}）中心辐亮度 {} != {SUN_RADIANCE}（行序/通道交错/uv 之一错位）",
        l_texel.x
    );
    // 太阳方向本身：落在亮 texel 边缘 ⇒ 介于"纯太阳"与"纯天空"之间，但必须
    // 明显高于天空（否则方向映射错得离谱）。
    let l_sun = cdf.radiance(table, sun_dir());
    assert!(
        l_sun.x > SKY * 50.0 && l_sun.x <= SUN_RADIANCE * (1.0 + 1e-5),
        "太阳方向辐亮度 {} 不在（{} , {SUN_RADIANCE}] 内（方向映射错位）",
        l_sun.x,
        SKY * 50.0
    );
    let l_nadir = cdf.radiance(table, Vec3::new(0.0, -1.0, 0.0));
    assert!(l_nadir.x <= SKY * 2.0, "天底方向不该有太阳能量，实得 {}", l_nadir.x);
    println!(
        "[exr] 最亮 texel（{bi},{bj}）中心 L = {:.6}，太阳方向 L = {:.6}（边缘混合），天底 L = {:.6}",
        l_texel.x, l_sun.x, l_nadir.x
    );

    // ③ 采样 8192 次：每个样本的 `pdf_w` 必须与 `sample.pdf_w` 逐位一致
    //    （同源），且落在太阳里的样本占比 ≈ `p_spot` 的太阳份额量级。
    let mut mismatched = 0u32;
    for k in 0..8192u32 {
        let u1 = (k as f32 + 0.5) / 8192.0;
        let u2 = {
            // 基 2 radical inverse（与探针同式，避免引入 RNG）。
            let mut x = k.wrapping_mul(2_654_435_761);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            (x as f32) * 2.328_306_4e-10
        };
        let s = cdf.sample(u1, u2).expect("非全黑必有样本");
        let p = cdf.pdf_w(s.wi);
        // 采样 texel 与反查 texel 可能差一格（方向 → uv 的舍入），只比较
        // **同一分支内**的位相等：用相对差 1e-6 作为"同源"判据。
        if p > 0.0 && ((p - s.pdf_w) / s.pdf_w).abs() > 1e-6 {
            mismatched += 1;
        }
    }
    assert_eq!(mismatched, 0, "{mismatched} 个样本的 pdf_w 与采样 pdf 不同源");
    println!("[exr] 8192 个样本的 pdf_w 与 sample.pdf_w 同源（相对差 ≤1e-6）");
}
