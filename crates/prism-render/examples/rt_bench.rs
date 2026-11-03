//! `rt_bench` —— 默认档（`RtBackend`）单帧计时 / 指纹工具。
//!
//! 零第三方依赖：只读 `std::time::Instant`，指纹用自带 FNV-1a 64。
//!
//! # 用法
//!
//! ```text
//! cargo run --release -p prism-render --example rt_bench -- \
//!     [w] [h] [spp] [bounces] [threads] [dump.raw]
//! ```
//!
//! 缺省 `1920 1080 16 6 0`（`threads=0` = 自动：`available_parallelism`）。给了 `dump.raw` 时把 RGBA8 原始字节写出，
//! 供 `cmp` 做逐字节对拍。
//!
//! # 场景
//!
//! 与 `rt_backend::tests::red_sphere_request` **同构**（单球 + 单方向光 +
//! 关后期），也就是选型报告 §13.5 的 1080p 基线场景。改这里就等于换基线，
//! 故刻意逐字段复制而非"顺手美化"。
//!
//! # 为什么不用 `--fast` 对照
//!
//! `--fast`（`CpuRefBackend`）是单弹射射线投射，量级差 200×，放在同一个
//! 计时里只会稀释读数；它的回归由 `rt_backend` 单测与 CLI 关卡各自守着。

use std::time::Instant;

use prism_gpu::{AestheticReq, CameraReq, GpuDevice, LightReq, SceneRenderRequest, SphereReq};
use prism_math::Vec3;
use prism_render::RtBackend;

/// 基线场景：单球 + 单方向光，关后期（与 `red_sphere_request` 同构）。
fn baseline_request(w: u32, h: u32) -> SceneRenderRequest {
    SceneRenderRequest {
        width: w,
        height: h,
        camera: CameraReq {
            eye: Vec3::new(0.0, 0.0, -4.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: w as f32 / h as f32,
        },
        lights: vec![LightReq {
            direction: Vec3::new(0.0, 0.0, -1.0),
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 3.0,
        }],
        spheres: vec![SphereReq {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
            albedo: Vec3::new(0.8, 0.2, 0.2),
            roughness: 0.35,
            metalness: 0.05,
            maps: prism_gpu::NO_MAPS,
        }],
        aesthetic: AestheticReq {
            teal_orange: false,
            contrast: 1.0,
            filmic: false,
        },
        curves: Vec::new(),
    ..Default::default()
}
}

/// FNV-1a 64（仅指纹用：跨进程比同一串字节是否相同）。
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, d: u32| -> u32 {
        args.get(i).and_then(|s| s.parse::<u32>().ok()).unwrap_or(d)
    };
    let w: u32 = get(0, 1920);
    let h: u32 = get(1, 1080);
    let spp: u32 = get(2, 16);
    let bounces: u32 = get(3, 6);
    let threads: usize = get(4, 0) as usize;
    let dump: Option<&String> = args.get(5);

    let req: SceneRenderRequest = baseline_request(w, h);
    let dev: RtBackend = RtBackend::new(spp, bounces).with_threads(threads);

    // 一帧热身（页错误 / 首次调用不落在计时里），再计时一帧。
    let warm: prism_gpu::Framebuffer = dev.clone().render_scene(&req);
    assert!(warm.is_valid(), "基准请求被拒（尺寸越界？）");

    let t0: Instant = Instant::now();
    let fb: prism_gpu::Framebuffer = dev.clone().render_scene(&req);
    let secs: f64 = t0.elapsed().as_secs_f64();

    assert!(fb.is_valid(), "基准请求被拒（尺寸越界？）");
    assert_eq!(fb.pixels, warm.pixels, "两跑必须逐字节一致（确定性铁律）");

    let samples: f64 = (w as f64) * (h as f64) * (spp as f64);
    println!("frame      : {w}x{h} / {spp} spp / {bounces} bounces / {} threads", dev.effective_threads());
    println!("wall       : {secs:.4} s  ({:.2} min, {:.2} h)", secs / 60.0, secs / 3600.0);
    println!("us/px/samp : {:.4}", secs * 1e6 / samples);
    println!("fnv64      : {:016x}", fnv1a64(&fb.pixels));

    if let Some(path) = dump {
        match std::fs::write(path, &fb.pixels) {
            Ok(()) => println!("dump       : {path} ({} bytes)", fb.pixels.len()),
            Err(e) => {
                eprintln!("dump 失败：{e}");
                std::process::exit(2);
            }
        }
    }
}
