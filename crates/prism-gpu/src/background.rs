//! 背景渲染：CPU 参考实现（GPU 管线兜底）。
//!
//! 本模块把 demo 的逐像素 `paint_background` 热点迁移为无 GPU 平台完整可运行的
//! CPU 后端，并作为 GPU shader（`shaders/background.wgsl`）的数值判对基准。
//! **两端公式严格一致**（hash value noise + 手写 smoothstep + 手写正弦逼近），
//! 保证「同输入同输出」；GPU 实际执行待 WebGPU 运行时（无 GPU 环境下仅校验
//! shader 编译/解析，不实际 GPU 渲染）。
//!
//! 接口：
//! - [`render_background_cpu`]：纯 CPU 实现，返回 RGBA8（`Vec<u8>`，行主序，len = w*h*4）。
//! - [`render_background`]：当前直接调用 CPU 参考；GPU 路径随 shader 构建就绪后接入。

use std::f32::consts::PI;

/// 背景色顶端（垂直渐变）。与 demo `BG_TOP` 一致。
const BG_TOP: [f32; 3] = [0.035, 0.055, 0.130];
/// 背景色底端（垂直渐变）。与 demo `BG_BOTTOM` 一致。
const BG_BOTTOM: [f32; 3] = [0.058, 0.090, 0.205];
/// 辉光冷青调系数。与 demo 一致。
const GLOW: [f32; 3] = [0.030, 0.070, 0.120];
/// 2π。
const TAU: f32 = 2.0 * PI;
/// `1 / 2^32`，与 WGSL `hash2i` 的归一化系数一致（均按 f32 计算）。
const INV_U32: f32 = 1.0 / 4_294_967_296.0;

/// 确定性 u32 哈希（与 WGSL `hash_u32` 位级一致）。
#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// 整数格点 + seed → [0,1) 伪随机（与 WGSL `hash2i` 同公式）。
#[inline]
fn hash2i(ix: u32, iy: u32, seed: u32) -> f32 {
    let mut h: u32 = ix
        .wrapping_mul(0x9e37_79b1)
        .wrapping_add(iy.wrapping_mul(0x85eb_ca77))
        .wrapping_add(seed.wrapping_mul(0xc2b2_ae3d));
    h = hash_u32(h);
    (h as f32) * INV_U32
}

/// 2D value noise（与 WGSL `value_noise` 同公式），返回 [0,1)。
#[inline]
fn value_noise(px: f32, py: f32, seed: u32) -> f32 {
    let ix = px.floor() as u32;
    let iy = py.floor() as u32;
    let fx = px - px.floor();
    let fy = py - py.floor();
    // smoothstep 权重，手写（与 WGSL 一致）。
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let x0 = ix;
    let y0 = iy;
    let x1 = ix + 1;
    let y1 = iy + 1;
    let v00 = hash2i(x0, y0, seed);
    let v10 = hash2i(x1, y0, seed);
    let v01 = hash2i(x0, y1, seed);
    let v11 = hash2i(x1, y1, seed);
    let a = v00 + (v10 - v00) * ux;
    let b = v01 + (v11 - v01) * ux;
    a + (b - a) * uy
}

/// demo `smooth`：clamp 到 [0,1] 再 t*t*(3-2t)。
#[inline]
fn smooth01(e: f32) -> f32 {
    let t = e.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 手写正弦逼近（Maclaurin 5 阶，与 WGSL `sin_approx` 同公式）。
#[inline]
fn sin_approx(x: f32) -> f32 {
    let mut a = x - TAU * (x / TAU).floor();
    if a > PI {
        a -= TAU;
    }
    if a < -PI {
        a += TAU;
    }
    let xx = a * a;
    let a3 = a * xx;
    let a5 = a3 * xx;
    let a7 = a5 * xx;
    let a9 = a7 * xx;
    a - a3 / 6.0 + a5 / 120.0 - a7 / 5040.0 + a9 / 362_880.0
}

/// 按像素坐标计算背景色（RGB，[0,1]）。与 WGSL `background_at` 同公式。
#[inline]
fn background_at(px: u32, py: u32, w: u32, h: u32, t: f32, seed: u32) -> [f32; 3] {
    let vx = px as f32 / (w as f32 - 1.0).max(1.0);
    let vy = py as f32 / (h as f32 - 1.0).max(1.0);

    // 1) 垂直渐变（lerp top/bottom，smoothstep）。
    let g = smooth01(vy);
    let mut col = [
        BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * g,
        BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * g,
        BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * g,
    ];

    // 2) 冷青辉光（二次衰减）。
    let gx = vx - 0.5;
    let gy = vy - 0.40;
    let d2 = gx * gx + gy * gy;
    let mut glow = 1.0 - d2 * 3.0;
    if glow < 0.0 {
        glow = 0.0;
    }
    let glow = glow * glow;
    col[0] += GLOW[0] * glow;
    col[1] += GLOW[1] * glow;
    col[2] += GLOW[2] * glow;

    // 3) 极低频噪声（hash value noise，振幅 ±3%，随时间极慢漂移）。
    let nx = vx * 2.2 + t * 0.05;
    let ny = vy * 1.3 - t * 0.035;
    let n = value_noise(nx, ny, seed) * 2.0 - 1.0; // [0,1) -> [-1,1)
    let modulation = 1.0 + 0.03 * n;

    // 4) 暗角。
    let cx = vx - 0.5;
    let cy = vy - 0.5;
    let rad = (cx * cx + cy * cy).sqrt();
    let vig = 1.0 - smooth01((rad - 0.45) / 0.55) * 0.55;

    // 5) 呼吸（正弦逼近，±1.5%）。
    let breath = 1.0 + 0.015 * sin_approx(t * (TAU / 5.0));

    let k = modulation * breath * vig;
    [col[0] * k, col[1] * k, col[2] * k]
}

/// CPU 参考实现：逐像素渲染背景为 RGBA8。
///
/// 返回行主序 `Vec<u8>`，每像素 4 字节 `[r, g, b, a]`，`len == w * h * 4`，
/// alpha 恒为 255。无 GPU 环境下这是当前可用后端（兜底）。
///
/// 公式与 `shaders/background.wgsl` 一致，供数值一致性对拍。
pub fn render_background_cpu(w: u32, h: u32, t: f32, seed: u64) -> Vec<u8> {
    let seed_u = seed as u32; // 高位折入低位，确定性；与 GPU uniform(u32) 同口径。
    let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = background_at(x, y, w, h, t, seed_u);
            let idx = ((y as usize) * (w as usize) + (x as usize)) * 4;
            out[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

/// 引擎背景渲染入口。
///
/// 当前直接调用 [`render_background_cpu`]（无 GPU 平台上的真实可运行后端）。
/// GPU 路径随 `shaders/background.wgsl` 构建就绪后接入——届时按 seed/时间上传
/// uniform、dispatch compute，再把 storage buffer 回读为 RGBA8。**注意**：无 GPU
/// 环境下 shader 仅做编译/解析校验，不实际 GPU 渲染；此处返回的是 CPU
/// 参考结果，可放心用于生产兜底。
pub fn render_background(w: u32, h: u32, t: f32, seed: u64) -> Vec<u8> {
    render_background_cpu(w, h, t, seed)
}
