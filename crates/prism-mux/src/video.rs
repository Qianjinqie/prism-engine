//! 时序视频编码器（I 帧 + P 帧整像素运动补偿）。
//!
//! 在 `prism-mux` 现有 JPEG 块编码之上实现**真正的帧间预测**，替代
//! `mp4.rs` 的"每帧一张 JPEG"MJPEG，显著提升压缩效率。
//!
//! # 设计要点
//!
//! - **I 帧**：完整彩色 JPEG（复用 [`crate::jpeg::encode_jpeg_color`]）。
//! - **P 帧**：对前一重建帧做整像素 16×16 块运动搜索（朴素 3-step 对数搜索，
//!   搜索窗 ±`search_radius`），得到每个宏块的运动矢量（MV）；残差 = 当前平面 −
//!   参考平面@MV（`Y/Cb/Cr` 三通道各自独立），各 +128 映射到 [0,255]，打包为三个
//!   独立灰度平面，分别用 `encode_jpeg_gray` 编码（**不做 RGB→YCbCr 颜色变换，避免
//!   残差被交叉混合 / 色度下采样**），三段灰度 JPEG 带长度前缀拼接为一个帧 blob。
//! - **零漂移**：编码器每编码一帧后，用**同一套自写 JPEG 解码器**把刚写出的
//!   字节解回作为下一 P 帧的参考帧，从而保证编码器参考 = 解码器重建，链路
//!   无累积漂移（无论 JPEG 量化损失多大）。
//! - **容器**：自定义轻量容器（magic `PRMV` + 头部 + 帧表[I/P、数据大小、每宏块
//!   MV] + 帧数据）。保留 `write_mp4_video`/`decode_mp4_video` 命名以兼容 API，
//!   但字节格式为本 crate 私有时序容器（非标准 MP4）。
//!
//! # 复用与约束
//!
//! - 仅复用 `jpeg.rs` 的 `encode_jpeg_color` 与 `encode_jpeg_gray`（DCT/量化/Huffman
//!   块编码），**不另写 JPEG 编码器**，也不改动 `write_mp4_jpeg_color`。
//! - 仅依赖标准库（零外部 crate）：JPEG 解码器为纯 Rust 实现（Huffman + 反量化 +
//!   定点 IDCT（纯整数，`IDCT_C_TABLE`/`IDCT_W_TABLE` 编译期常量，见 [`idct_8x8_fixed`]）。

use crate::jpeg::{encode_jpeg_color, encode_jpeg_gray};
use crate::JpegConfig;

/// 视频编码配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoConfig {
    /// 图像组长度：每 `gop` 帧一个 I 帧（帧 0, gop, 2·gop, … 为 I 帧）。
    pub gop: u32,
    /// JPEG 质量（1..=100；越高码率越大、失真越小）。I 帧与 P 帧残差共用。
    pub quality: u8,
    /// 整像素运动搜索半径（±`search_radius`，如 16）。MV 以 `i8` 存储，故上限 127。
    pub search_radius: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        VideoConfig { gop: 12, quality: 85, search_radius: 16 }
    }
}

/// P 帧率控后验（每 GOP 更新一次；`len == gop − 1`，索引即 GOP 内 P 帧序号）。
///
/// - `theta[i]`：失真敏感度 `θ_i = mse_i · 2^{2·R_i}`（实测反推的模型系数）；
/// - `r_base[i]`：上一 GOP 该帧实测 `R_i = bits / 符号数`（bit/符号）；
/// - `bits_total`：上一 GOP 各 P 帧 `R_i` 之和（**与 water_filling 的 `R` 同量纲**，
///   故可直接当预算传入）。
#[derive(Debug, Clone)]
struct PFrameRatePrior {
    theta: Vec<f64>,
    r_base: Vec<f64>,
    bits_total: f64,
}

impl PFrameRatePrior {
    /// 初值：`theta = 1`、`r_base = 0`、`bits_total = 0`（= 无后验）。
    fn new(gop: usize) -> Self {
        let len = gop.saturating_sub(1);
        PFrameRatePrior { theta: vec![1.0; len], r_base: vec![0.0; len], bits_total: 0.0 }
    }
}

/// ZigZag 次序表（自然序下标 → ZigZag 序位置）；与 `jpeg.rs` 同源标准表。
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// 容器魔数 `PRMV`（Prism Video）。
const MAGIC: &[u8; 4] = b"PRMV";
/// 容器版本（v2：宽/高/fps 升为 `u32` 大端；v1 仍可向后读）。
const VERSION: u8 = 2;

// ===========================================================================
// 公开 API
// ===========================================================================
/// 由后验算本 GOP 的逐 P 帧质量计划（纯函数；编码器与测试共用）。
///
/// - 无后验（`bits_total == 0`，即首 GOP）→ 全 `p_base_q`，**逐位等于历史启发式**；
/// - 有后验 → 依赖加权水注得 `R_i`，再 `q_i = round(base + (R_i − r_base_i)·170)`，
///   夹到 `[30,95]`。该帧三平面（Y/Cb/Cr）共用同一质量。
fn p_frame_quality_plan(gop: usize, p_base_q: u8, prior: &PFrameRatePrior) -> Vec<u8> {
    let n_p = gop.saturating_sub(1);
    let mut q_plan = vec![p_base_q; n_p];
    if prior.bits_total > 0.0 && n_p > 0 {
        // 依赖加权：第 i 个 P 帧影响其后 gop−i 帧。
        let theta_w: Vec<f64> = prior
            .theta
            .iter()
            .take(n_p)
            .enumerate()
            .map(|(i, &t)| t * (gop - i) as f64)
            .collect();
        let r = crate::rate_control::water_filling(&theta_w, prior.bits_total);
        for i in 0..n_p {
            let q = ((p_base_q as f64)
                + (r[i] - prior.r_base[i]) * crate::rate_control::QUALITY_PER_BIT)
                .round()
                .clamp(30.0, 95.0);
            q_plan[i] = q as u8;
        }
    }
    q_plan
}


/// 把 RGBA8 帧序列编码为带 I/P 帧间预测的视频字节流。
///
/// `frames` 为逐帧 RGBA8（`width*height*4`）。返回自定义 `PRMV` 容器字节；可用
/// [`decode_mp4_video`] 往返解码。
///
/// # 错误
/// 参数非法（空序列 / 尺寸不符 / `gop==0` / 超 u16 分辨率）或内部编码失败时返回
/// `String` 描述。
pub fn write_mp4_video(
    frames: &[Vec<u8>],
    spec: crate::TrackSpec,
    config: VideoConfig,
) -> Result<Vec<u8>, String> {
    let w = spec.width;
    let h = spec.height;
    if w == 0 || h == 0 || spec.fps == 0 {
        return Err(format!("非法视频参数 {w}x{h} @{fps}fps", fps = spec.fps));
    }
    // 分辨率上限：v2 头部用 u32 存宽/高/fps（见 `pack_container`），故只需防
    // `w·h·4` 在 usize 上溢出（u64 中间量，32 位目标也安全）。
    let bytes_needed = (w as u64) * (h as u64) * 4u64;
    if bytes_needed > usize::MAX as u64 {
        return Err(format!("分辨率 {w}x{h} 超出平台可寻址范围（需 {bytes_needed} 字节）"));
    }
    if config.gop == 0 {
        return Err("gop 必须为正整数（每 gop 帧一个 I 帧）".to_string());
    }
    if frames.is_empty() {
        return Err("帧序列为空".to_string());
    }
    let n = (w as usize) * (h as usize);
    for (i, f) in frames.iter().enumerate() {
        if f.len() != n * 4 {
            return Err(format!("第 {i} 帧长度 {} 应为 {}x{}x4={}", f.len(), w, h, n * 4));
        }
    }

    let jpeg_cfg = JpegConfig { quality: config.quality.clamp(1, 100), grayscale: false };
    // P 帧残差**基准**质量：历史启发式（`q−17` 夹到 `[50,80]`）。
    //
    // 首 GOP（以及任何尚无后验实测的 GOP）逐帧用它——与 V1.4 逐位一致；自第二 GOP
    // 起，由 [`crate::rate_control::water_filling`] 在上一 GOP 实测的失真敏感度上重新
    // 分配位预算，再经一阶控制器 `Δq = ΔR·QUALITY_PER_BIT` 折算成逐帧质量
    // （见下方「率控」段）。I 帧质量恒为 `config.quality`，不受率控影响。
    let p_base_q = (config.quality as i32 - 17).clamp(50, 80) as u8;
    let mb_w = (w as usize + 15) / 16;
    let mb_h = (h as usize + 15) / 16;
    let gop = config.gop as usize;
    let sym = (w as f64) * (h as f64) * 3.0; // 每帧符号数（RGB 三通道）
    let mut prior = PFrameRatePrior::new(gop);

    let mut reference: Option<Vec<u8>> = None; // 重建参考帧（RGBA）
    let mut frame_data: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let mut frame_types: Vec<u8> = Vec::with_capacity(frames.len()); // 0=I, 1=P
    let mut all_mvs: Vec<Vec<(i16, i16)>> = Vec::with_capacity(frames.len());

    // 逐 GOP 编码：每个 GOP 起始先定 P 帧质量计划，编码中实测 (bits, mse)，GOP 末尾
    // 刷新后验（θ / r_base / bits_total）供下一 GOP 使用。
    let mut g0 = 0usize;
    while g0 < frames.len() {
        let g1 = (g0 + gop).min(frames.len());
        let p_count = g1 - g0 - 1; // 本 GOP 内的 P 帧数（首帧为 I 帧）

        // ---- 率控：本 GOP 的逐 P 帧质量计划 ----
        let q_plan = p_frame_quality_plan(gop, p_base_q, &prior);

        let mut gop_bits = vec![0.0f64; p_count]; // 每 P 帧实测位数
        let mut gop_mse = vec![0.0f64; p_count]; // 每 P 帧实测 mse（RGB 三通道）
        let mut gop_r = vec![0.0f64; p_count]; // 每 P 帧实测 bit/符号

        for i in g0..g1 {
            let frame = &frames[i];
            if i == g0 {
                // ---- I 帧：完整彩色 JPEG ----
                let jpeg = encode_jpeg_color(frame, w, h, jpeg_cfg).map_err(|e| e.to_string())?;
                // 编码器参考帧 = 解码器将重建的帧（消除漂移）。
                let recon = decode_jpeg(&jpeg)?;
                frame_data.push(jpeg);
                frame_types.push(0);
                all_mvs.push(Vec::new());
                reference = Some(recon);
            } else {
                // ---- P 帧：运动补偿 + 三通道残差彩色 JPEG ----
                let pi = i - g0 - 1;
                // 率控给出的本帧残差质量（首 GOP 恒为 p_base_q，逐位不变）。
                let resid_cfg = JpegConfig { quality: q_plan[pi], grayscale: false };
                let ref_frame =
                    reference.as_ref().ok_or("P 帧缺少参考帧（首帧须为 I 帧）")?;
                // 当前帧与参考帧都转到 YCbCr，分别用于亮度运动估计与三通道残差构造。
                let mut cur_y = vec![0u8; n];
                let mut cur_cb = vec![0u8; n];
                let mut cur_cr = vec![0u8; n];
                rgb_to_ycbcr(frame, n, &mut cur_y, &mut cur_cb, &mut cur_cr);
                let mut ref_y = vec![0u8; n];
                let mut ref_cb = vec![0u8; n];
                let mut ref_cr = vec![0u8; n];
                rgb_to_ycbcr(ref_frame, n, &mut ref_y, &mut ref_cb, &mut ref_cr);
                let mut mvs: Vec<(i16, i16)> = Vec::with_capacity(mb_w * mb_h);
                for by in 0..mb_h {
                    for bx in 0..mb_w {
                        let mv = estimate_motion_vector(
                            &cur_y,
                            &ref_y,
                            w as usize,
                            h as usize,
                            bx,
                            by,
                            config.search_radius as i32,
                        );
                        mvs.push(mv);
                    }
                }
                // 构造三通道独立灰度残差平面（Y/Cb/Cr 各自 = 当前 − 参考@MV，+128 映射）；
                // 每平面单独以灰度 JPEG 编码（不做颜色变换），三段带长度前缀拼接成一个 blob。
                let (resid_y, resid_cb, resid_cr) = build_color_residual(
                    &cur_y, &cur_cb, &cur_cr, &ref_y, &ref_cb, &ref_cr, &mvs, w as usize,
                    h as usize,
                );
                let jpeg_y =
                    encode_jpeg_gray(&resid_y, w, h, resid_cfg).map_err(|e| e.to_string())?;
                let jpeg_cb =
                    encode_jpeg_gray(&resid_cb, w, h, resid_cfg).map_err(|e| e.to_string())?;
                let jpeg_cr =
                    encode_jpeg_gray(&resid_cr, w, h, resid_cfg).map_err(|e| e.to_string())?;
                let mut blob: Vec<u8> = Vec::new();
                blob.extend_from_slice(&(jpeg_y.len() as u32).to_be_bytes());
                blob.extend_from_slice(&jpeg_y);
                blob.extend_from_slice(&(jpeg_cb.len() as u32).to_be_bytes());
                blob.extend_from_slice(&jpeg_cb);
                blob.extend_from_slice(&(jpeg_cr.len() as u32).to_be_bytes());
                blob.extend_from_slice(&jpeg_cr);
                // 解码三段灰度残差 → 重建当前帧（与解码器完全相同路径，保证零漂移）。
                let (dec_y, dec_cb, dec_cr) = decode_pframe_residual(&blob)?;
                let recon = reconstruct_pframe(
                    ref_frame, &dec_y, &dec_cb, &dec_cr, &mvs, w as usize, h as usize,
                );
                // ---- 率控后验测量：位数 + 重建 vs 原始 RGB 的均方误差 ----
                let mut sse = 0i64;
                for px in 0..n {
                    for c in 0..3 {
                        let d = frame[px * 4 + c] as i64 - recon[px * 4 + c] as i64;
                        sse += d * d;
                    }
                }
                gop_bits[pi] = blob.len() as f64 * 8.0;
                gop_r[pi] = gop_bits[pi] / sym;
                gop_mse[pi] = sse as f64 / sym;
                frame_data.push(blob);
                frame_types.push(1);
                all_mvs.push(mvs);
                reference = Some(recon);
            }
        }

        // ---- GOP 末尾：刷新后验（θ_i = mse_i·2^{2R_i}，r_base_i = R_i）----
        if p_count > 0 {
            let mut total = 0.0f64;
            for pi in 0..p_count {
                // mse 下限 1e-9：避免全静止 GOP 把 θ 压成 0 后永久失活（不静默）。
                prior.theta[pi] = (gop_mse[pi] * prism_math::det_pow::det_exp2(2.0 * gop_r[pi]))
                    .max(1e-9);
                prior.r_base[pi] = gop_r[pi];
                total += gop_r[pi];
            }
            prior.bits_total = total;
        }
        g0 = g1;
    }

    Ok(pack_container(
        &frame_data,
        &frame_types,
        &all_mvs,
        w,
        h,
        spec.fps,
        config.gop,
        config.quality,
    ))
}

/// 解码 [`write_mp4_video`] 产生的字节流，返回逐帧 RGBA8。
///
/// 解码器复用以 `video.rs` 私有的基线 JPEG 解码器还原 I 帧与 P 帧残差，并按帧表
/// 中的 MV 做运动补偿重建 P 帧。
pub fn decode_mp4_video(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let container = unpack_container(data)?;
    let w = container.width as usize;
    let h = container.height as usize;
    let mut reference: Option<Vec<u8>> = None;
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(container.frames.len());

    for f in &container.frames {
        if f.kind == 0 {
            let rgba = decode_jpeg(&f.data)?;
            reference = Some(rgba.clone());
            out.push(rgba);
        } else {
            let ref_frame = reference.as_ref().ok_or("P 帧缺少参考帧")?;
            let (resid_y, resid_cb, resid_cr) = decode_pframe_residual(&f.data)?;
            let recon =
                reconstruct_pframe(ref_frame, &resid_y, &resid_cb, &resid_cr, &f.mvs, w, h);
            reference = Some(recon.clone());
            out.push(recon);
        }
    }
    Ok(out)
}

// ===========================================================================
// 运动估计（朴素 3-step 对数搜索，整像素）
// ===========================================================================

/// 把 RGBA8 转为亮度平面（BT.601 整数权重，与 `jpeg.rs` 一致）。
#[allow(dead_code)] // 仅在测试（debug_pframe_luma_vs_chroma）中使用。
fn rgb_to_luma(rgba: &[u8], n: usize) -> Vec<u8> {
    let mut y = vec![0u8; n];
    for i in 0..n {
        let r = rgba[i * 4] as u32;
        let g = rgba[i * 4 + 1] as u32;
        let b = rgba[i * 4 + 2] as u32;
        y[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
    }
    y
}

/// 计算一个 16×16 宏块在参考帧（按 MV 偏移后）的 SAD（绝对差和）。
fn sad_block(
    cur_y: &[u8],
    ref_y: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    mvx: i32,
    mvy: i32,
) -> u32 {
    let mut sad = 0u32;
    for yy in 0..16 {
        for xx in 0..16 {
            let px = bx * 16 + xx;
            let py = by * 16 + yy;
            if px >= w || py >= h {
                continue;
            }
            let rx = (px as i32 + mvx).clamp(0, w as i32 - 1) as usize;
            let ry = (py as i32 + mvy).clamp(0, h as i32 - 1) as usize;
            sad += (cur_y[py * w + px] as i32 - ref_y[ry * w + rx] as i32).unsigned_abs();
        }
    }
    sad
}

/// 3-step 对数搜索：从 (0,0) 起，逐步以 `radius`→`radius/2`→…→1 的步长在 8 邻域
/// 求精，得到使 SAD 最小的整像素 MV（限定在 ±`radius` 内）。
fn estimate_motion_vector(
    cur_y: &[u8],
    ref_y: &[u8],
    w: usize,
    h: usize,
    bx: usize,
    by: usize,
    radius: i32,
) -> (i16, i16) {
    let radius = radius.max(0);
    let mut best_mv = (0i32, 0i32);
    let mut best_sad = sad_block(cur_y, ref_y, w, h, bx, by, 0, 0);
    // 从不超过 radius 的最大 2 的幂起步（至少 1）。
    let mut step = 1;
    while step * 2 <= radius {
        step *= 2;
    }
    while step >= 1 {
        let neighbors = [
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (step, -step),
            (-step, step),
            (step, step),
        ];
        for (dx, dy) in neighbors {
            let mvx = (best_mv.0 + dx).clamp(-radius, radius);
            let mvy = (best_mv.1 + dy).clamp(-radius, radius);
            let sad = sad_block(cur_y, ref_y, w, h, bx, by, mvx, mvy);
            if sad < best_sad {
                best_sad = sad;
                best_mv = (mvx, mvy);
            }
        }
        step /= 2;
    }
    (best_mv.0 as i16, best_mv.1 as i16)
}

/// 构造三通道独立灰度残差平面。
///
/// 对每个像素，`Y_res = cur_Y - ref_Y@MV`、`Cb_res = cur_Cb - ref_Cb@MV`、
/// `Cr_res = cur_Cr - ref_Cr@MV`，各加 128 映射到 [0,255]（128 ≡ 零残差），返回三个
/// 独立的 `w*h` 灰度平面（而非打包成 RGBA）。各自以灰度 JPEG 编码（不做 RGB→YCbCr
/// 颜色变换），从而避免残差三通道被交叉混合、色度被半分辨率下采样——这正是之前 PSNR
/// 跌到 34 dB 的根因。编码器与解码器共用同一套打包/重建，链路零漂移。
fn build_color_residual(
    cur_y: &[u8],
    cur_cb: &[u8],
    cur_cr: &[u8],
    ref_y: &[u8],
    ref_cb: &[u8],
    ref_cr: &[u8],
    mvs: &[(i16, i16)],
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = w * h;
    let mut resid_y = vec![128u8; n]; // 128 ≡ 零残差
    let mut resid_cb = vec![128u8; n];
    let mut resid_cr = vec![128u8; n];
    let mb_w = (w + 15) / 16;
    for (i, &(mvx, mvy)) in mvs.iter().enumerate() {
        let bx = i % mb_w;
        let by = i / mb_w;
        for yy in 0..16 {
            for xx in 0..16 {
                let px = bx * 16 + xx;
                let py = by * 16 + yy;
                if px >= w || py >= h {
                    continue;
                }
                let rx = (px as i32 + mvx as i32).clamp(0, w as i32 - 1) as usize;
                let ry2 = (py as i32 + mvy as i32).clamp(0, h as i32 - 1) as usize;
                let o = py * w + px;
                let ro = ry2 * w + rx;
                resid_y[o] = (cur_y[o] as i32 - ref_y[ro] as i32 + 128).clamp(0, 255) as u8;
                resid_cb[o] = (cur_cb[o] as i32 - ref_cb[ro] as i32 + 128).clamp(0, 255) as u8;
                resid_cr[o] = (cur_cr[o] as i32 - ref_cr[ro] as i32 + 128).clamp(0, 255) as u8;
            }
        }
    }
    (resid_y, resid_cb, resid_cr)
}

/// 由参考帧（RGBA）+ 解码后的三平面灰度残差（各以 128 为中心）+ MV 重建 P 帧。
///
/// 重建规则（编码器与解码器共用，保证零漂移）：对每个像素按 MV 取参考帧的
/// Y/Cb/Cr（参考帧先 `rgb_to_ycbcr` 得到三平面），加上各自解码残差（减 128）得到
/// 当前帧 YCbCr，再经 YCbCr→RGBA 输出。亮度与色度均独立编码/重建，不再被颜色变换混
/// 合，运动边缘的色度不再从背景串色。
fn reconstruct_pframe(
    reference: &[u8],
    resid_y: &[u8],
    resid_cb: &[u8],
    resid_cr: &[u8],
    mvs: &[(i16, i16)],
    w: usize,
    h: usize,
) -> Vec<u8> {
    let n = w * h;
    let mut ref_y = vec![0u8; n];
    let mut ref_cb = vec![0u8; n];
    let mut ref_cr = vec![0u8; n];
    rgb_to_ycbcr(reference, n, &mut ref_y, &mut ref_cb, &mut ref_cr);
    let mut recon = vec![0u8; n * 4];
    let mb_w = (w + 15) / 16;
    for (i, &(mvx, mvy)) in mvs.iter().enumerate() {
        let bx = i % mb_w;
        let by = i / mb_w;
        for yy in 0..16 {
            for xx in 0..16 {
                let px = bx * 16 + xx;
                let py = by * 16 + yy;
                if px >= w || py >= h {
                    continue;
                }
                let rx = (px as i32 + mvx as i32).clamp(0, w as i32 - 1) as usize;
                let ry2 = (py as i32 + mvy as i32).clamp(0, h as i32 - 1) as usize;
                let o = py * w + px;
                let ro = ry2 * w + rx;
                let y_res = resid_y[o] as i32 - 128;
                let cb_res = resid_cb[o] as i32 - 128;
                let cr_res = resid_cr[o] as i32 - 128;
                let yv = (ref_y[ro] as i32 + y_res).clamp(0, 255) as u8;
                let cb_val = ref_cb[ro] as i32 + cb_res;
                let cr_val = ref_cr[ro] as i32 + cr_res;
                let cb_off = cb_val as f64 - 128.0;
                let cr_off = cr_val as f64 - 128.0;
                let r = (yv as f64 + 1.402 * cr_off).clamp(0.0, 255.0) as u8;
                let g =
                    (yv as f64 - 0.344136 * cb_off - 0.714136 * cr_off).clamp(0.0, 255.0) as u8;
                let b = (yv as f64 + 1.772 * cb_off).clamp(0.0, 255.0) as u8;
                recon[o * 4] = r;
                recon[o * 4 + 1] = g;
                recon[o * 4 + 2] = b;
                recon[o * 4 + 3] = 255;
            }
        }
    }
    recon
}

/// 从灰度 JPEG 字节流解码为单平面（取解码 RGBA 的 R 通道即灰度样本）。
fn decode_gray_plane(data: &[u8]) -> Result<Vec<u8>, String> {
    let rgba = decode_jpeg(data)?;
    let npix = rgba.len() / 4;
    let mut plane = vec![0u8; npix];
    for i in 0..npix {
        plane[i] = rgba[i * 4];
    }
    Ok(plane)
}

/// 以大端 `u32` 从 `b` 的 `pos` 处读出一个长度；越界报错。
fn read_u32_be(b: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > b.len() {
        return Err("P 帧残差 blob 长度前缀越界".to_string());
    }
    let v = u32::from_be_bytes([b[*pos], b[*pos + 1], b[*pos + 2], b[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

/// 从 P 帧 blob 拆出三段灰度 JPEG 并各自解码为灰度平面。
///
/// blob 布局（不含外层 `pack_container` 的长度前缀）：
/// `[u32 lenY][jpegY][u32 lenCb][jpegCb][u32 lenCr][jpegCr]`。
/// 编码器与解码器共用此函数，保证参考帧重建与解码路径完全一致（零漂移）。
fn decode_pframe_residual(
    blob: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let mut pos = 0usize;
    let len_y = read_u32_be(blob, &mut pos)? as usize;
    if pos + len_y > blob.len() {
        return Err("P 帧 Y 残差 JPEG 越界".to_string());
    }
    let jpeg_y = &blob[pos..pos + len_y];
    pos += len_y;
    let len_cb = read_u32_be(blob, &mut pos)? as usize;
    if pos + len_cb > blob.len() {
        return Err("P 帧 Cb 残差 JPEG 越界".to_string());
    }
    let jpeg_cb = &blob[pos..pos + len_cb];
    pos += len_cb;
    let len_cr = read_u32_be(blob, &mut pos)? as usize;
    if pos + len_cr > blob.len() {
        return Err("P 帧 Cr 残差 JPEG 越界".to_string());
    }
    let jpeg_cr = &blob[pos..pos + len_cr];
    let py = decode_gray_plane(jpeg_y)?;
    let pcb = decode_gray_plane(jpeg_cb)?;
    let pcr = decode_gray_plane(jpeg_cr)?;
    Ok((py, pcb, pcr))
}

/// RGBA8 → YCbCr 三平面（与 `jpeg::rgb_to_ycbcr` 同源的 JFIF 整数矩阵，无 libm）。
///
/// 用于 P 帧重建时从参考帧取得运动补偿所需的亮度/色度平面（避免每帧重复依赖
/// 编码器私有实现）。
fn rgb_to_ycbcr(rgba: &[u8], n: usize, y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    for i in 0..n {
        let r = rgba[i * 4] as i32;
        let g = rgba[i * 4 + 1] as i32;
        let b = rgba[i * 4 + 2] as i32;
        y[i] = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
        cb[i] = (((-43 * r - 85 * g + 128 * b) >> 8) + 128).clamp(0, 255) as u8;
        cr[i] = (((128 * r - 107 * g - 21 * b) >> 8) + 128).clamp(0, 255) as u8;
    }
}

// ===========================================================================
// 容器打包 / 解包
// ===========================================================================

/// 单帧记录（解包后）。
struct FrameRec {
    kind: u8, // 0=I, 1=P
    data: Vec<u8>,
    mvs: Vec<(i16, i16)>,
}

/// 解包后的容器。
#[allow(dead_code)]
struct Container {
    width: u32,
    height: u32,
    fps: u32,
    gop: u32,
    quality: u8,
    frames: Vec<FrameRec>,
}

/// 打包 PRMV 容器。
fn pack_container(
    frame_data: &[Vec<u8>],
    frame_types: &[u8],
    all_mvs: &[Vec<(i16, i16)>],
    w: u32,
    h: u32,
    fps: u32,
    gop: u32,
    quality: u8,
) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    // v2 头部：宽/高/fps 为 `u32` 大端（v1 为 `u16`，`unpack_container` 仍可向后读）；
    // gop 保持 `u16`（帧组长度，>65535 无意义）。
    out.extend_from_slice(&w.to_be_bytes());
    out.extend_from_slice(&h.to_be_bytes());
    out.extend_from_slice(&fps.to_be_bytes());
    out.extend_from_slice(&(gop.min(u16::MAX as u32) as u16).to_be_bytes());
    out.push(quality);
    out.extend_from_slice(&(frame_data.len() as u32).to_be_bytes());
    // 帧表。
    for i in 0..frame_data.len() {
        out.push(frame_types[i]);
        out.extend_from_slice(&(frame_data[i].len() as u32).to_be_bytes());
        if frame_types[i] == 1 {
            let mvs = &all_mvs[i];
            out.extend_from_slice(&(mvs.len() as u32).to_be_bytes());
            for &(mx, my) in mvs {
                // MV 以 i8 存储（每宏块 2 字节），覆盖 ±127，足够 search_radius≤127。
                out.push((mx.clamp(i8::MIN as i16, i8::MAX as i16)) as i8 as u8);
                out.push((my.clamp(i8::MIN as i16, i8::MAX as i16)) as i8 as u8);
            }
        }
    }
    // 帧数据。
    for f in frame_data {
        out.extend_from_slice(f);
    }
    out
}

/// 解包 PRMV 容器。
fn unpack_container(data: &[u8]) -> Result<Container, String> {
    if data.len() < 4 || &data[0..4] != MAGIC {
        return Err("不是 PRMV 容器（魔数不匹配）".to_string());
    }
    let mut pos = 4;
    if pos >= data.len() {
        return Err("容器头部截断（缺版本字节）".to_string());
    }
    let version = data[pos];
    pos += 1;
    // v1：宽/高/fps/gop 各 `u16`；v2：宽/高/fps 升 `u32`、gop 仍 `u16`。
    // 其他版本一律拒绝（不猜测布局，不静默读错位）。
    let (w, h, fps, gop) = match version {
        1 => {
            let need = pos + 8;
            if need > data.len() {
                return Err("容器 v1 头部截断（宽/高/fps/gop）".to_string());
            }
            let w = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
            pos += 2;
            let h = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
            pos += 2;
            let fps = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
            pos += 2;
            let gop = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
            pos += 2;
            (w, h, fps, gop)
        }
        2 => {
            let need = pos + 14;
            if need > data.len() {
                return Err("容器 v2 头部截断（宽/高/fps/gop）".to_string());
            }
            let w = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let h = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let fps = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let gop = u16::from_be_bytes([data[pos], data[pos + 1]]) as u32;
            pos += 2;
            (w, h, fps, gop)
        }
        other => {
            return Err(format!(
                "不支持的容器版本 {other}（本解码器支持 v1/v2；请用对应版本的 prism 解码）"
            ))
        }
    };
    if w == 0 || h == 0 {
        return Err(format!("容器头部宽/高为 0（{w}×{h}），结构损坏"));
    }
    if pos >= data.len() {
        return Err("容器头部截断（缺质量字节）".to_string());
    }
    let quality = data[pos];
    pos += 1;
    let num_frames =
        u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    // 先解析完整帧表（kind / size / MVs），记录每帧数据偏移。
    let mut entries: Vec<(u8, usize, Vec<(i16, i16)>)> = Vec::with_capacity(num_frames);
    for _ in 0..num_frames {
        if pos >= data.len() {
            return Err("帧表越界（容器损坏）".to_string());
        }
        let kind = data[pos];
        pos += 1;
        // 帧型 2 = B 帧：容器 v2 保留该编码，但编解码器未实装（V2.0-K 之后再落）。
        // 明确报错而非静默按 P 帧解（那会产出错帧却"成功"）。
        if kind == 2 {
            return Err(
                "B 帧编码未实装（容器 v2 保留帧型）：本解码器只支持帧型 0=I / 1=P".to_string(),
            );
        }
        if kind > 2 {
            return Err(format!("未知帧型 {kind}（合法值：0=I, 1=P）"));
        }
        let size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut mvs = Vec::new();
        if kind == 1 {
            let mv_count =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            for _ in 0..mv_count {
                let mx = i16::from(data[pos] as i8);
                let my = i16::from(data[pos + 1] as i8);
                mvs.push((mx, my));
                pos += 2;
            }
        }
        entries.push((kind, size, mvs));
    }
    // 帧表之后为连续帧数据区。
    let data_start = pos;
    let mut cursor = data_start;
    let mut frames: Vec<FrameRec> = Vec::with_capacity(num_frames);
    for (kind, size, mvs) in entries {
        if cursor + size > data.len() {
            return Err("帧数据越界（容器损坏）".to_string());
        }
        let fdata = data[cursor..cursor + size].to_vec();
        cursor += size;
        frames.push(FrameRec { kind, data: fdata, mvs });
    }
    Ok(Container { width: w, height: h, fps, gop, quality, frames })
}

// ===========================================================================
// 基线 JPEG 解码器（私有；仅服务于本时序编码器 / 解码器的往返）
//
// 支持本 crate 编码器产出的格式：1 分量灰度（4:4:4 退化）或 3 分量 4:4:4 彩色，
// 单量化表、标准 DC/AC Huffman 表（Annex K）、无重启动标记。
// ===========================================================================

/// Huffman 解码表（标准 JPEG 规范 §F.2.2.3 的 mincode/maxcode/valptr 法）。
struct HuffDec {
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    huffval: Vec<u8>,
}

// ---------------------------------------------------------------------------
// 定点 IDCT（D-29：零 libm；纯整数、逐位确定）
// ---------------------------------------------------------------------------

/// 定点 IDCT 余弦基表：`round(2^14 · K(u,x))`，`K(a,b) = cos((2b+1)·a·π/16)`。
///
/// 与 `jpeg.rs` 的前向基 `DCT_COS` **同值同布局**（`u*8+x`）——前向/逆向共用同一
/// 张编译期常量基，不出现任何运行时超越函数（D-29 清除：`f64::cos`/`PI` 零残留）。
pub const IDCT_C_TABLE: [i32; 64] = [
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16069, 13623, 9102, 3196, -3196, -9102,
    -13623, -16069, 15137, 6270, -6270, -15137, -15137, -6270, 6270, 15137, 13623, -3196, -16069,
    -9102, 9102, 16069, 3196, -13623, 11585, -11585, -11585, 11585, 11585, -11585, -11585, 11585,
    9102, -16069, 3196, 13623, -13623, -3196, 16069, -9102, 6270, -15137, 15137, -6270, -6270, 15137,
    -15137, 6270, 3196, -9102, 13623, -16069, 16069, -13623, 9102, -3196,
];

/// 定点 IDCT 归一化权重 `round(2^16 · c(u)·c(v))`，`c(0)=1/√2`、`c(k>0)=1`。
///
/// 取值只可能是三个常量：`c(0)c(0)=½ → 32768`、`c(0)c(k)=1/√2 → 46341`、
/// `c(k)c(l)=1 → 65536`。布局同样是 `u*8+v`。
pub const IDCT_W_TABLE: [i32; 64] = [
    32768, 46341, 46341, 46341, 46341, 46341, 46341, 46341, 46341, 65536, 65536, 65536, 65536,
    65536, 65536, 65536, 46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, 46341, 65536,
    65536, 65536, 65536, 65536, 65536, 65536, 46341, 65536, 65536, 65536, 65536, 65536, 65536,
    65536, 46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536, 46341, 65536, 65536, 65536,
    65536, 65536, 65536, 65536, 46341, 65536, 65536, 65536, 65536, 65536, 65536, 65536,
];

/// 走 `i64` 两级累加的安全上界：逐块 `max|D·W| ≤ 阈值` 时两级累加与末位舍入
/// （`+2^45`）都不会溢出 `i64`；超过则升级 `i128`（结果逐位相同，纯防御）。
pub const IDCT_STAGE2_I64_THRESHOLD: i64 = (1 << 29) - 1;

const _: () = assert!(IDCT_STAGE2_I64_THRESHOLD <= (1 << 29) - 1);

/// 定点 2D IDCT：`out = clamp(round((1/4)·ΣΣ c(u)c(v)·D·K(u,x)·K(v,y)))`。
///
/// # 输入
/// - `coef`：自然序**量化后**系数（`coef[v*8+u]` 对应 `F[u][v]`，未反量化）；
/// - `q`：自然序量化步长（来自 DQT，已按 ZigZag 反排成自然序）。
///
/// # 定点推导
/// 反量化 `D = coef·q`（整数）。两级分离：
/// `S1[u][y] = Σ_v D·W[u][v]·C[v][y]`、`S2[y][x] = Σ_u S1[u][y]·C[u][x]`。
/// 标度：`W ≈ 2^16·c(u)c(v)`、`C ≈ 2^14·K`，故 `S2 ≈ s·(1/4)^{-1}·2^{16+14+14}`
/// 的两级展开即 `S2 ≈ s·2^46`；`>>46` 即回真值，`+2^45` 为四舍五入。
///
/// # 输出范围
/// 输出为**去电平偏移**的样本（`[−128,127]`，调用方 `+128` 得 `[0,255]`），即
/// 与 `clamp(s+128, 0, 255) − 128` 等价——保持既有调用点的 `+128` 语义不变。
pub fn idct_8x8_fixed(out: &mut [i32; 64], coef: &[i32; 64], q: &[i32; 64]) {
    // 反量化 + 逐块幅度上界（决定 i64 / i128 路径）。
    let mut d = [0i64; 64];
    let mut max_dw = 0i64;
    for i in 0..64 {
        d[i] = (coef[i] as i64) * (q[i] as i64);
        let dw = d[i].saturating_mul(IDCT_W_TABLE[i] as i64).unsigned_abs();
        let dw = dw.min(i64::MAX as u64) as i64;
        if dw > max_dw {
            max_dw = dw;
        }
    }
    if max_dw <= IDCT_STAGE2_I64_THRESHOLD {
        idct_core::<i64>(out, &d);
    } else {
        idct_core::<i128>(out, &d);
    }
}

/// 两级整数 IDCT 内核（`T = i64` 快路径 / `i128` 防御路径，结果逐位相同）。
fn idct_core<T>(out: &mut [i32; 64], d: &[i64; 64])
where
    T: Copy
        + Default
        + From<i64>
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Into<i128>,
{
    // 反量化系数 × 归一化权重。
    let mut dw = [T::default(); 64];
    for i in 0..64 {
        dw[i] = T::from(d[i]) * T::from(IDCT_W_TABLE[i] as i64);
    }
    // Stage 1（列）：S1[u][y] = Σ_v dw[v*8+u] · C[v*8+y]。
    let mut s1 = [T::default(); 64]; // s1[u*8+y]
    for u in 0..8 {
        for y in 0..8 {
            let mut acc = T::default();
            for v in 0..8 {
                acc = acc + dw[v * 8 + u] * T::from(IDCT_C_TABLE[v * 8 + y] as i64);
            }
            s1[u * 8 + y] = acc;
        }
    }
    // Stage 2（行）：S2[y][x] = Σ_u S1[u][y] · C[u*8+x]，末位 round + 右移 46。
    for y in 0..8 {
        for x in 0..8 {
            let mut acc = T::default();
            for u in 0..8 {
                acc = acc + s1[u * 8 + y] * T::from(IDCT_C_TABLE[u * 8 + x] as i64);
            }
            let s: i128 = acc.into();
            let v = (s + (1i128 << 45)) >> 46;
            out[y * 8 + x] = v.clamp(-128, 127) as i32;
        }
    }
}

/// 由 BITS/VALS 构建 Huffman 解码表。
fn build_huff(bits: &[u8; 16], vals: &[u8]) -> HuffDec {
    let mut mincode = [0i32; 17];
    let mut maxcode = [-1i32; 17];
    let mut valptr = [0i32; 17];
    let mut huffval = vals.to_vec();
    let mut code = 0i32;
    let mut k = 0usize;
    for l in 1..=16 {
        if bits[l - 1] == 0 {
            maxcode[l] = -1;
        } else {
            valptr[l] = k as i32;
            mincode[l] = code;
            code += bits[l - 1] as i32;
            maxcode[l] = code - 1;
            k += bits[l - 1] as usize;
        }
        code <<= 1;
    }
    if huffval.len() < k {
        huffval.resize(k, 0);
    }
    HuffDec { mincode, maxcode, valptr, huffval }
}

/// JPEG 熵流位读取器（MSB 优先；处理 0xFF 0x00 填充与 0xFF 标记 → 1 填充）。
struct JpegBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u32,
    cnt: i32,
    eoi: bool,
}

impl<'a> JpegBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        JpegBitReader { data, pos: 0, buf: 0, cnt: 0, eoi: false }
    }

    fn fill(&mut self) {
        while self.cnt <= 24 {
            if self.eoi || self.pos >= self.data.len() {
                // 数据结束 / 遇标记：以全 1 填充（等价于熵段尾部 1 填充）。
                self.buf = (self.buf << 8) | 0xFF;
                self.cnt += 8;
                self.eoi = true;
                continue;
            }
            let b = self.data[self.pos];
            self.pos += 1;
            if b == 0xFF {
                if self.pos < self.data.len() && self.data[self.pos] == 0x00 {
                    // 字节填充：0xFF 是真实数据字节，其后 0x00 为填充物。
                    self.pos += 1;
                } else {
                    // 0xFF 后非 0x00 → 标记（如 EOI）：填充 1 位并停止。
                    self.eoi = true;
                    self.buf = (self.buf << 8) | 0xFF;
                    self.cnt += 8;
                    continue;
                }
            }
            self.buf = (self.buf << 8) | b as u32;
            self.cnt += 8;
        }
    }

    fn get_bit(&mut self) -> u32 {
        self.fill();
        self.cnt -= 1;
        (self.buf >> self.cnt) & 1
    }

    fn get_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.get_bit();
        }
        v
    }
}

/// Huffman 解码一个符号。
fn huff_decode(r: &mut JpegBitReader, h: &HuffDec) -> Result<u8, String> {
    let mut code = 0i32;
    for l in 1..=16 {
        code = (code << 1) | r.get_bit() as i32;
        if h.maxcode[l] >= 0 && code >= h.mincode[l] && code <= h.maxcode[l] {
            let idx = h.valptr[l] + (code - h.mincode[l]);
            return Ok(h.huffval[idx as usize]);
        }
    }
    Err("Huffman 解码：未匹配到码字（流损坏）".to_string())
}

/// 读取 `s` 位幅值并按 JPEG 规范还原带符号值。
fn receive_extend(r: &mut JpegBitReader, s: u32) -> i32 {
    if s == 0 {
        return 0;
    }
    let v = r.get_bits(s) as i32;
    if v < (1i32 << (s - 1)) {
        v - ((1i32 << s) - 1)
    } else {
        v
    }
}

/// 解码单个 8×8 块（DC 差分 + AC 游程），写入 `coef_zigzag`（ZigZag 序）。
fn decode_block(
    r: &mut JpegBitReader,
    hdc: &HuffDec,
    hac: &HuffDec,
    pred_dc: &mut i32,
    coef_zigzag: &mut [i32; 64],
) -> Result<(), String> {
    // DC。
    let t = huff_decode(r, hdc)? as u32;
    let diff = receive_extend(r, t);
    let dc = *pred_dc + diff;
    *pred_dc = dc;
    coef_zigzag[0] = dc;
    // AC。
    let mut k = 1usize;
    while k < 64 {
        let rs = huff_decode(r, hac)? as usize;
        let rrun = rs >> 4;
        let s = rs & 0x0F;
        if s == 0 {
            if rrun == 0 {
                break; // EOB
            } else {
                k += 16; // ZRL：跳过 16 个零
                continue;
            }
        } else {
            k += rrun;
            if k >= 64 {
                break;
            }
            coef_zigzag[k] = receive_extend(r, s as u32);
            k += 1;
        }
    }
    // 反量化不在此处做：D = coef·q 移入 `idct_8x8_fixed`（定点，一次乘完）。
    Ok(())
}

/// 解析并解码一个基线 JPEG 字节流，返回 RGBA8（灰度自动扩展为灰阶 RGB）。
fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 2 || &data[0..2] != [0xFF, 0xD8] {
        return Err("不是 JPEG（缺少 SOI）".to_string());
    }
    let mut pos = 2usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut ncomp = 0u8;
    let mut comp_ids: Vec<u8> = Vec::new();
    let mut comp_qid: Vec<usize> = Vec::new();
    let mut quant: [[u16; 64]; 4] = [[0u16; 64]; 4];
    let mut huff_dc: [Option<HuffDec>; 4] = [None, None, None, None];
    let mut huff_ac: [Option<HuffDec>; 4] = [None, None, None, None];
    let mut sos_map: Vec<(usize, usize, usize)> = Vec::new(); // (comp_index, dc_table, ac_table)
    let mut entropy_start = 0usize;

    'scan: while pos + 1 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        pos += 2;
        match marker {
            0xD9 => break 'scan,                  // EOI
            0xD0..=0xD7 | 0x01 => continue,      // RSTn / TEM：无长度段
            0xDA => {
                let len =
                    u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                let ns = data[pos + 2];
                let mut p = pos + 3;
                for _ in 0..ns {
                    let cid = data[p];
                    let ta = data[p + 1];
                    let cidx = comp_ids
                        .iter()
                        .position(|&id| id == cid)
                        .ok_or("SOS 引用了未知分量")?;
                    sos_map.push((cidx, (ta >> 4) as usize, (ta & 0x0F) as usize));
                    p += 2;
                }
                entropy_start = pos + len;
                break 'scan;
            }
            _ => {
                let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + len > data.len() {
                    return Err("标记段越界（JPEG 损坏）".to_string());
                }
                let seg = &data[pos + 2..pos + len];
                match marker {
                    0xDB => {
                        // DQT。
                        let mut p = 0usize;
                        while p < seg.len() {
                            let pq_tq = seg[p];
                            p += 1;
                            let id = (pq_tq & 0x0F) as usize;
                            if pq_tq & 0xF0 != 0 {
                                // 16-bit 精度（本编码器不产出，跳过）。
                                for _ in 0..64 {
                                    p += 2;
                                }
                            } else {
                                for k in 0..64 {
                                    quant[id][k] = seg[p] as u16;
                                    p += 1;
                                }
                            }
                        }
                    }
                    0xC0 => {
                        // SOF0（基线）。
                        height =
                            u16::from_be_bytes([seg[1], seg[2]]) as u32;
                        width = u16::from_be_bytes([seg[3], seg[4]]) as u32;
                        ncomp = seg[5];
                        let mut p = 6usize;
                        for _ in 0..ncomp {
                            let cid = seg[p];
                            let _samp = seg[p + 1];
                            let qid = seg[p + 2];
                            comp_ids.push(cid);
                            comp_qid.push(qid as usize);
                            p += 3;
                        }
                    }
                    0xC4 => {
                        // DHT。
                        let mut p = 0usize;
                        while p < seg.len() {
                            let tc_th = seg[p];
                            p += 1;
                            let tc = (tc_th >> 4) as usize;
                            let th = (tc_th & 0x0F) as usize;
                            let mut bits = [0u8; 16];
                            let mut total = 0usize;
                            for i in 0..16 {
                                bits[i] = seg[p];
                                total += bits[i] as usize;
                                p += 1;
                            }
                            let vals: Vec<u8> = seg[p..p + total].to_vec();
                            p += total;
                            let hd = build_huff(&bits, &vals);
                            if tc == 0 {
                                huff_dc[th] = Some(hd);
                            } else {
                                huff_ac[th] = Some(hd);
                            }
                        }
                    }
                    _ => {}
                }
                pos += len;
            }
        }
    }

    if width == 0 || height == 0 {
        return Err("JPEG 缺少有效 SOF0 尺寸".to_string());
    }
    let ncomp = ncomp as usize;
    let bw = ((width as usize) + 7) / 8;
    let bh = ((height as usize) + 7) / 8;
    let npix = (width as usize) * (height as usize);
    let mut planes: Vec<Vec<u8>> =
        (0..ncomp).map(|_| vec![0u8; npix]).collect();
    let mut pred_dc = vec![0i32; ncomp];
    let mut reader = JpegBitReader::new(&data[entropy_start..]);

    for by in 0..bh {
        for bx in 0..bw {
            for (ci, &(_, dc_t, ac_t)) in sos_map.iter().enumerate() {
                let hdc = huff_dc[dc_t].as_ref().ok_or("缺少 DC Huffman 表")?;
                let hac = huff_ac[ac_t].as_ref().ok_or("缺少 AC Huffman 表")?;
                let qid = comp_qid[ci];
                let mut coef_zigzag = [0i32; 64];
                decode_block(&mut reader, hdc, hac, &mut pred_dc[ci], &mut coef_zigzag)?;
                // ZigZag → 自然序（系数与量化步长都反排；反量化在定点 IDCT 内）。
                let mut coef_nat = [0i32; 64];
                let mut q_nat = [0i32; 64];
                for i in 0..64 {
                    coef_nat[ZIGZAG[i]] = coef_zigzag[i];
                    q_nat[ZIGZAG[i]] = quant[qid][i] as i32;
                }
                let mut samples = [0i32; 64];
                idct_8x8_fixed(&mut samples, &coef_nat, &q_nat);
                let base_x = bx * 8;
                let base_y = by * 8;
                for yy in 0..8 {
                    for xx in 0..8 {
                        let px = (base_x + xx).min(width as usize - 1);
                        let py = (base_y + yy).min(height as usize - 1);
                        let val = (samples[yy * 8 + xx] + 128).clamp(0, 255) as u8;
                        planes[ci][py * width as usize + px] = val;
                    }
                }
            }
        }
    }

    // YCbCr(4:4:4) → RGBA，或灰度 → 灰阶 RGBA。
    let mut rgba = vec![0u8; npix * 4];
    if ncomp == 1 {
        for i in 0..npix {
            let v = planes[0][i];
            rgba[i * 4] = v;
            rgba[i * 4 + 1] = v;
            rgba[i * 4 + 2] = v;
            rgba[i * 4 + 3] = 255;
        }
    } else {
        for i in 0..npix {
            let y = planes[0][i] as f64;
            let cb = planes[1][i] as f64 - 128.0;
            let cr = planes[2][i] as f64 - 128.0;
            let r = y + 1.402 * cr;
            let g = y - 0.344136 * cb - 0.714136 * cr;
            let b = y + 1.772 * cb;
            rgba[i * 4] = r.clamp(0.0, 255.0) as u8;
            rgba[i * 4 + 1] = g.clamp(0.0, 255.0) as u8;
            rgba[i * 4 + 2] = b.clamp(0.0, 255.0) as u8;
            rgba[i * 4 + 3] = 255;
        }
    }
    Ok(rgba)
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4::write_mp4_jpeg_color;
    use crate::JpegConfig;
    use crate::TrackSpec;

    /// 生成平滑运动测试序列：背景渐变 + 缓慢向右平移的色块。
    fn gen_test_frames(w: u32, h: u32, n: u32) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(n as usize);
        for f in 0..n {
            let t = f as f64;
            let pan = (t * 4.0) as i32; // 每帧 4 像素向右平移
            let mut buf = vec![0u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    let gx = (x as f64 / w as f64 * 255.0) as i32;
                    let gy = (y as f64 / h as f64 * 255.0) as i32;
                    let mut r = ((gx + gy) / 2).clamp(0, 255);
                    let mut g = gx.clamp(0, 255);
                    let mut b = gy.clamp(0, 255);
                    // 缓慢平移的色块（200×150）。
                    let bx = x as i32 - pan;
                    let by = y as i32 - (h as i32 / 3);
                    if bx >= 0 && bx < 200 && by >= 0 && by < 150 {
                        r = 220;
                        g = 40;
                        b = 90;
                    }
                    let idx = ((y * w + x) * 4) as usize;
                    buf[idx] = r as u8;
                    buf[idx + 1] = g as u8;
                    buf[idx + 2] = b as u8;
                    buf[idx + 3] = 255;
                }
            }
            out.push(buf);
        }
        out
    }

    /// 计算 RGBA 序列之间的 PSNR（dB），对比三通道。
    fn compute_psnr(orig: &[Vec<u8>], recon: &[Vec<u8>]) -> f64 {
        let mut sum_sq = 0.0;
        let mut count = 0.0;
        for (o, r) in orig.iter().zip(recon.iter()) {
            for i in (0..o.len()).step_by(4) {
                for c in 0..3 {
                    let d = o[i + c] as f64 - r[i + c] as f64;
                    sum_sq += d * d;
                    count += 1.0;
                }
            }
        }
        if sum_sq == 0.0 {
            return 99.0;
        }
        let mse = sum_sq / count;
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    #[test]
    fn single_iframe_roundtrip_is_high_quality() {
        // 单帧（仅 I 帧）往返，JPEG 解码质量应高。
        let frames = gen_test_frames(320, 180, 1);
        let spec = TrackSpec { width: 320, height: 180, fps: 24 };
        let cfg = VideoConfig { gop: 1, quality: 85, search_radius: 16 };
        let data = write_mp4_video(&frames, spec, cfg).unwrap();
        let decoded = decode_mp4_video(&data).unwrap();
        assert_eq!(decoded.len(), 1);
        let psnr = compute_psnr(&frames, &decoded);
        assert!(psnr >= 40.0, "单帧 I 帧 PSNR {psnr:.2} < 40dB");
    }

    #[test]
    fn video_roundtrip_psnr_above_40_and_smaller_than_mjpeg() {
        // 回归测试：5 秒类序列（缩小分辨率以保证 CI 速度），验证硬指标。
        let w = 320u32;
        let h = 180u32;
        let fps = 24u32;
        let nframes = 48u32;
        let frames = gen_test_frames(w, h, nframes);
        let spec = TrackSpec { width: w, height: h, fps };
        let cfg = VideoConfig { gop: 12, quality: 85, search_radius: 16 };

        let data = write_mp4_video(&frames, spec, cfg).unwrap();
        let decoded = decode_mp4_video(&data).unwrap();

        // 1) 解码帧数 == 原始帧数。
        assert_eq!(decoded.len(), frames.len(), "帧数不一致");

        // 2) 往返 PSNR ≥ 40 dB（对比原始 RGBA）。
        let psnr = compute_psnr(&frames, &decoded);
        assert!(psnr >= 40.0, "往返 PSNR {psnr:.2} dB < 40 dB");

        // 3) 体积显著小于同参数 MJPEG。
        let mjpeg = write_mp4_jpeg_color(
            &frames,
            spec,
            None,
            0,
            0,
            JpegConfig { quality: 85, grayscale: false },
        )
        .unwrap();
        assert!(
            data.len() < mjpeg.len(),
            "时序编码体积 {} 应小于 MJPEG {}", data.len(), mjpeg.len()
        );
    }

    #[test]
    fn debug_decoder_flat() {
        // 编码 8x8 纯 128 图像，解码应得 ≈128。
        let mut frame = vec![128u8; 8 * 8 * 4];
        for i in 0..8 * 8 {
            frame[i * 4 + 3] = 255;
        }
        let jpeg = crate::jpeg::encode_jpeg_color(&frame, 8, 8, JpegConfig { quality: 90, grayscale: false }).unwrap();
        let decoded = decode_jpeg(&jpeg).unwrap();
        eprintln!("flat decoded first 8: {:?}", &decoded[0..32]);
        let mut maxerr = 0i32;
        for i in 0..8 * 8 {
            for c in 0..3 {
                maxerr = maxerr.max((decoded[i * 4 + c] as i32 - 128).abs());
            }
        }
        eprintln!("flat maxerr = {maxerr}");
    }

    #[test]
    fn debug_decoder_gradient() {
        let w = 16u32;
        let h = 16u32;
        let mut frame = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) * 4) as u8;
                let idx = ((y * w + x) * 4) as usize;
                frame[idx] = v; frame[idx + 1] = v; frame[idx + 2] = v; frame[idx + 3] = 255;
            }
        }
        let jpeg = crate::jpeg::encode_jpeg_color(&frame, w, h, JpegConfig { quality: 95, grayscale: false }).unwrap();
        let decoded = decode_jpeg(&jpeg).unwrap();
        let mut sse = 0.0;
        let mut cnt = 0.0;
        for i in 0..(w * h) as usize {
            for c in 0..3 {
                let d = frame[i * 4 + c] as f64 - decoded[i * 4 + c] as f64;
                sse += d * d; cnt += 1.0;
            }
        }
        let mse = sse / cnt;
        let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
        eprintln!("gradient psnr = {psnr:.2}, mse = {mse:.2}");
    }

    #[test]
    fn idct_fixed_roundtrip_through_jpeg_codec() {
        // D-29 清除后：IDCT 为纯整数定点，不再有 f64 前向 DCT 可对拍。改为**走真实
        // 编解码链路**验证互逆性：`encode_jpeg_gray`（整数前向 DCT + 量化）→
        // `decode_jpeg`（Huffman + 定点 IDCT）。误差来源只剩量化与末位舍入，
        // 故 q=95 时最大绝对误差应很小（远小于缩放类 bug 的 ~35）。
        let w = 32u32;
        let h = 32u32;
        let mut plane = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                // 平滑斜面 + 低频起伏，覆盖 DC 与中低频系数。
                let v = 96.0 + (x as i32 * 5 - 40) as f64 * 0.7
                    + (y as i32 * 3 - 30) as f64 * 0.4
                    + ((x as i32 * y as i32) % 11) as f64;
                plane[(y * w + x) as usize] = v.clamp(0.0, 255.0) as u8;
            }
        }
        let jpeg = encode_jpeg_gray(&plane, w, h, JpegConfig { quality: 95, grayscale: true })
            .unwrap();
        let rgba = decode_jpeg(&jpeg).unwrap();
        let mut max_diff = 0i32;
        let mut sse = 0i64;
        for i in 0..(w * h) as usize {
            let d = rgba[i * 4] as i32 - plane[i] as i32;
            max_diff = max_diff.max(d.abs());
            sse += (d * d) as i64;
        }
        let mse = sse as f64 / (w * h) as f64;
        let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
        eprintln!("fixed IDCT roundtrip: max_diff={max_diff}, psnr={psnr:.2}");
        assert!(max_diff <= 6, "定点 IDCT 往返最大误差 {max_diff} 超 6");
        assert!(psnr >= 38.0, "定点 IDCT 往返 PSNR {psnr:.2} dB < 38 dB");
    }

    #[test]
    fn debug_frame_sizes() {
        // 诊断：打印 I/P 帧与 MV 表各自字节数，定位 PRMV 体积为何未小于 MJPEG。
        let w = 320u32;
        let h = 180u32;
        let frames = gen_test_frames(w, h, 48);
        let spec = TrackSpec { width: w, height: h, fps: 24 };
        let cfg = VideoConfig { gop: 12, quality: 85, search_radius: 16 };
        let jpeg_cfg = JpegConfig { quality: 85, grayscale: false };
        let mb_w = ((w as usize) + 15) / 16;
        let mb_h = ((h as usize) + 15) / 16;

        let mut i_total = 0usize;
        let mut p_resid = 0usize;
        let mut mv_total = 0usize;
        let mut reference: Option<Vec<u8>> = None;
        for (i, frame) in frames.iter().enumerate() {
            if (i as u32) % cfg.gop == 0 {
                let jpeg = crate::jpeg::encode_jpeg_color(frame, w, h, jpeg_cfg).unwrap();
                i_total += jpeg.len();
                reference = Some(decode_jpeg(&jpeg).unwrap());
                eprintln!("frame {i}: I  jpeg={} B", jpeg.len());
            } else {
                let ref_frame = reference.as_ref().unwrap();
                let mut cy = vec![0u8; (w * h) as usize];
                let mut ccb = vec![0u8; (w * h) as usize];
                let mut ccr = vec![0u8; (w * h) as usize];
                rgb_to_ycbcr(frame, (w * h) as usize, &mut cy, &mut ccb, &mut ccr);
                let mut ry = vec![0u8; (w * h) as usize];
                let mut rcb = vec![0u8; (w * h) as usize];
                let mut rcr = vec![0u8; (w * h) as usize];
                rgb_to_ycbcr(ref_frame, (w * h) as usize, &mut ry, &mut rcb, &mut rcr);
                let mut mvs = Vec::new();
                for by in 0..mb_h {
                    for bx in 0..mb_w {
                        mvs.push(estimate_motion_vector(
                            &cy, &ry, w as usize, h as usize, bx, by,
                            cfg.search_radius as i32,
                        ));
                    }
                }
                let (ry_p, rcb_p, rcr_p) = build_color_residual(
                    &cy, &ccb, &ccr, &ry, &rcb, &rcr, &mvs, w as usize, h as usize);
                let jy = crate::jpeg::encode_jpeg_gray(&ry_p, w, h, jpeg_cfg).unwrap();
                let jcb = crate::jpeg::encode_jpeg_gray(&rcb_p, w, h, jpeg_cfg).unwrap();
                let jcr = crate::jpeg::encode_jpeg_gray(&rcr_p, w, h, jpeg_cfg).unwrap();
                let mut blob = Vec::new();
                for j in [&jy, &jcb, &jcr] {
                    blob.extend_from_slice(&(j.len() as u32).to_be_bytes());
                    blob.extend_from_slice(j);
                }
                p_resid += blob.len();
                mv_total += mvs.len() * 2;
                let (dy, dcb, dcr) = decode_pframe_residual(&blob).unwrap();
                let recon =
                    reconstruct_pframe(ref_frame, &dy, &dcb, &dcr, &mvs, w as usize, h as usize);
                reference = Some(recon);
                eprintln!(
                    "frame {i}: P  resid_blob={} B | mv_table={} B",
                    blob.len(),
                    mvs.len() * 2
                );
            }
        }
        let mjpeg = write_mp4_jpeg_color(
            &frames, spec, None, 0, 0, JpegConfig { quality: 85, grayscale: false },
        )
        .unwrap();
        eprintln!(
            "I_total={i_total} P_resid_total={p_resid} MV_total={mv_total} | MJPEG={}",
            mjpeg.len()
        );
        eprintln!(
            "PRMV_estimate={} B",
            i_total + p_resid + mv_total + 18 + 5 * 48
        );
    }

    #[test]
    fn debug_pframe_luma_vs_chroma() {
        // 诊断 P 帧重建误差来源：逐通道比较 frame1（首个 P 帧）。
        let w = 64u32;
        let h = 64u32;
        let frames = gen_test_frames(w, h, 5);
        let spec = TrackSpec { width: w, height: h, fps: 24 };
        let cfg = VideoConfig { gop: 4, quality: 85, search_radius: 16 };
        let data = write_mp4_video(&frames, spec, cfg).unwrap();
        let decoded = decode_mp4_video(&data).unwrap();
        let n = (w * h) as usize;
        // 逐通道 PSNR。
        for c in 0..3 {
            let mut ss = 0.0;
            let mut cnt = 0.0;
            for i in 0..n {
                let d = frames[1][i * 4 + c] as f64 - decoded[1][i * 4 + c] as f64;
                ss += d * d;
                cnt += 1.0;
            }
            let mse = ss / cnt;
            let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
            eprintln!("frame1 channel {c}: psnr={psnr:.2} mse={mse:.2}");
        }
        // 直接重放 frame1 的编码链路，打印 MV 与参考色度。
        let ref0 = decode_jpeg(
            &encode_jpeg_color(&frames[0], w, h, JpegConfig { quality: 85, grayscale: false })
                .unwrap(),
        )
        .unwrap();
        let cur_y = rgb_to_luma(&frames[1], n);
        let ref_y = rgb_to_luma(&ref0, n);
        let mb_w = ((w as usize) + 15) / 16;
        let mut mvs = Vec::new();
        for by in 0..((h as usize + 15) / 16) {
            for bx in 0..mb_w {
                mvs.push(estimate_motion_vector(
                    &cur_y, &ref_y, w as usize, h as usize, bx, by, 16,
                ));
            }
        }
        let px = 40usize;
        let py = 40usize;
        let mbi = (py / 16) * mb_w + (px / 16);
        let (mvx, mvy) = mvs[mbi];
        eprintln!("pixel({px},{py}) MV=({mvx},{mvy})");
        let rx = ((px as i32 + mvx as i32).clamp(0, w as i32 - 1)) as usize;
        let ry = ((py as i32 + mvy as i32).clamp(0, h as i32 - 1)) as usize;
        let ro = ry * w as usize + rx;
        eprintln!(
            "ref RGB @ (rx={rx},ry={ry}) = ({},{},{})",
            ref0[ro * 4], ref0[ro * 4 + 1], ref0[ro * 4 + 2]
        );
        let o = (py * w as usize + px) * 4;
        eprintln!("orig RGB @ ({px},{py}) = ({},{},{})", frames[1][o], frames[1][o + 1], frames[1][o + 2]);
        eprintln!("dec  RGB @ ({px},{py}) = ({},{},{})", decoded[1][o], decoded[1][o + 1], decoded[1][o + 2]);
        let mut sr = 0u32; let mut sg = 0u32; let mut sb = 0u32; let mut cc = 0u32;
        for y in 30..50 {
            for x in 30..50 {
                let i = (y * w as usize + x) * 4;
                sr += ref0[i] as u32; sg += ref0[i + 1] as u32; sb += ref0[i + 2] as u32; cc += 1;
            }
        }
        eprintln!("ref0 block-region avg RGB = ({},{},{})", sr / cc, sg / cc, sb / cc);
    }

    #[test]
    fn debug_per_frame_psnr() {
        let w = 320u32;
        let h = 180u32;
        let fps = 24u32;
        let nframes = 48u32;
        let frames = gen_test_frames(w, h, nframes);
        let spec = TrackSpec { width: w, height: h, fps };
        let cfg = VideoConfig { gop: 12, quality: 85, search_radius: 16 };
        let data = write_mp4_video(&frames, spec, cfg).unwrap();
        let decoded = decode_mp4_video(&data).unwrap();
        for f in 0..frames.len() {
            let mut ss = 0.0; let mut cnt = 0.0;
            let n = (w*h) as usize;
            for i in 0..n {
                for c in 0..3 {
                    let d = frames[f][i*4+c] as f64 - decoded[f][i*4+c] as f64;
                    ss += d*d; cnt += 1.0;
                }
            }
            let psnr = 10.0*(255.0*255.0/(ss/cnt)).log10();
            eprintln!("frame {f:2}: psnr={psnr:.2} {}", if f%12==0 {"[I]"} else {"[P]"});
        }
    }

    #[test]
    fn debug_residual_fidelity() {
        let w = 320u32;
        let h = 180u32;
        let frames = gen_test_frames(w, h, 5);
        let cfg = VideoConfig { gop: 12, quality: 85, search_radius: 16 };
        let jpeg_cfg = JpegConfig { quality: 85, grayscale: false };
        let resid_cfg = JpegConfig { quality: 97, grayscale: false };
        let n = (w*h) as usize;
        // 重放 frame1。
        let ref0 = decode_jpeg(&encode_jpeg_color(&frames[0], w, h, jpeg_cfg).unwrap()).unwrap();
        let mut cy=vec![0u8;n]; let mut ccb=vec![0u8;n]; let mut ccr=vec![0u8;n];
        rgb_to_ycbcr(&frames[1], n, &mut cy, &mut ccb, &mut ccr);
        let mut ry=vec![0u8;n]; let mut rcb=vec![0u8;n]; let mut rcr=vec![0u8;n];
        rgb_to_ycbcr(&ref0, n, &mut ry, &mut rcb, &mut rcr);
        let mb_w = ((w as usize)+15)/16;
        let mb_h = ((h as usize)+15)/16;
        let mut mvs = Vec::new();
        for by in 0..mb_h { for bx in 0..mb_w {
            mvs.push(estimate_motion_vector(&cy,&ry,w as usize,h as usize,bx,by,cfg.search_radius as i32));
        }}
        let (ry_p, rcb_p, rcr_p) = build_color_residual(&cy,&ccb,&ccr,&ry,&rcb,&rcr,&mvs,w as usize,h as usize);
        // 残差编码前：统计各通道范围。
        let enc_planes = [&ry_p, &rcb_p, &rcr_p];
        for c in 0..3 {
            let mut mn=255u8; let mut mx=0u8;
            for i in 0..n { let v=enc_planes[c][i]; mn=mn.min(v); mx=mx.max(v); }
            eprintln!("residual ch{c} range [{mn},{mx}]");
        }
        let jy = encode_jpeg_gray(&ry_p, w, h, resid_cfg).unwrap();
        let jcb = encode_jpeg_gray(&rcb_p, w, h, resid_cfg).unwrap();
        let jcr = encode_jpeg_gray(&rcr_p, w, h, resid_cfg).unwrap();
        let mut blob = Vec::new();
        for j in [&jy, &jcb, &jcr] {
            blob.extend_from_slice(&(j.len() as u32).to_be_bytes());
            blob.extend_from_slice(j);
        }
        let (dy, dcb, dcr) = decode_pframe_residual(&blob).unwrap();
        let dec_planes = [&dy, &dcb, &dcr];
        for c in 0..3 {
            let mut ss=0.0; let mut cnt=0.0;
            for i in 0..n {
                let d = enc_planes[c][i] as f64 - dec_planes[c][i] as f64;
                ss += d*d; cnt += 1.0;
            }
            let mse = ss/cnt;
            let psnr = 10.0*(255.0*255.0/mse).log10();
            eprintln!("residual ch{c} roundtrip psnr={psnr:.2} mse={mse:.2}");
        }
        // 用解码残差重建，对比 orig。
        let recon = reconstruct_pframe(&ref0, &dy, &dcb, &dcr, &mvs, w as usize, h as usize);
        let mut ss=0.0; let mut cnt=0.0;
        for i in 0..n { for c in 0..3 {
            let d = frames[1][i*4+c] as f64 - recon[i*4+c] as f64; ss+=d*d; cnt+=1.0;
        }}
        let psnr = 10.0*(255.0*255.0/(ss/cnt)).log10();
        eprintln!("frame1 recon from decoded residual psnr={psnr:.2}");
    }

    #[test]
    fn rate_control_second_gop_redistributes_bits() {
        // 24 帧、gop=12：第二 GOP（帧 12..23）走率控，第一 GOP 逐位等于历史启发式。
        let w = 320u32;
        let h = 180u32;
        let frames = gen_test_frames(w, h, 24);
        let spec = TrackSpec { width: w, height: h, fps: 24 };
        let two_gop = write_mp4_video(
            &frames,
            spec,
            VideoConfig { gop: 12, quality: 85, search_radius: 16 },
        )
        .unwrap();
        // 对照：gop=24 ⇒ 全序列只有首 GOP，帧 13..23 用的是**纯启发式**质量。
        let one_gop = write_mp4_video(
            &frames,
            spec,
            VideoConfig { gop: 24, quality: 85, search_radius: 16 },
        )
        .unwrap();

        // (a) 计划函数直接证据：有后验时质量计划 ≠ 启发式。
        let gop = 12usize;
        let base_q = (85i32 - 17).clamp(50, 80) as u8; // = 68
        let prior = PFrameRatePrior {
            theta: (0..gop - 1).map(|i| 1.0 + (i as f64) * 0.35).collect(),
            r_base: vec![0.14; gop - 1],
            bits_total: 0.14 * (gop - 1) as f64,
        };
        let plan = p_frame_quality_plan(gop, base_q, &prior);
        assert_eq!(plan.len(), gop - 1);
        assert!(
            plan.iter().any(|&q| q != base_q),
            "有后验时质量计划应偏离启发式 {base_q}，实际全等"
        );
        // 无后验 → 必须逐位等于启发式。
        assert_eq!(p_frame_quality_plan(gop, base_q, &PFrameRatePrior::new(gop)), vec![base_q; gop - 1]);

        // (b) 端到端：第二 GOP 的 P 帧体积分布改变，但总位数仍相近（±20%）。
        let a = unpack_container(&two_gop).unwrap();
        let b = unpack_container(&one_gop).unwrap();
        assert_eq!(a.frames.len(), 24);
        assert_eq!(b.frames.len(), 24);
        let mut total_a = 0usize;
        let mut total_b = 0usize;
        let mut differs = false;
        for i in 13..24usize {
            let sa = a.frames[i].data.len();
            let sb = b.frames[i].data.len();
            total_a += sa;
            total_b += sb;
            if sa != sb {
                differs = true;
            }
        }
        assert!(differs, "第二 GOP 的 P 帧体积与纯启发式完全一致：率控未生效");
        let ratio = (total_a as f64 - total_b as f64).abs() / total_b as f64;
        assert!(
            ratio <= 0.20,
            "第二 GOP 总位数偏离启发式 {:.1}% 超 ±20%（{total_a} vs {total_b}）",
            ratio * 100.0
        );
        assert!(total_a > 0);
    }

    #[test]
    fn container_v2_roundtrip_and_v1_backread() {
        let frames = gen_test_frames(64, 64, 6);
        let spec = TrackSpec { width: 64, height: 64, fps: 30 };
        let data =
            write_mp4_video(&frames, spec, VideoConfig { gop: 3, quality: 85, search_radius: 16 })
                .unwrap();
        // 新写出的容器必须是 v2。
        assert_eq!(data[4], 2, "容器版本字节应为 2");
        let dec = decode_mp4_video(&data).unwrap();
        assert_eq!(dec.len(), 6);
        let psnr = compute_psnr(&frames, &dec);
        assert!(psnr >= 40.0, "v2 往返 PSNR {psnr:.2} < 40 dB");

        // 把 v2 头部改写成 v1 布局（宽/高/fps u16），其余字节原样 → 必须仍可解码，
        // 且重建帧逐位相同（v1 向后读的兼容性证明）。
        let mut v1: Vec<u8> = Vec::new();
        v1.extend_from_slice(&data[0..4]); // magic
        v1.push(1); // version 1
        v1.extend_from_slice(&(64u16).to_be_bytes());
        v1.extend_from_slice(&(64u16).to_be_bytes());
        v1.extend_from_slice(&(30u16).to_be_bytes());
        v1.extend_from_slice(&(3u16).to_be_bytes());
        let v2_header_len = 4 + 1 + 4 + 4 + 4 + 2; // magic+ver+w+h+fps+gop
        v1.extend_from_slice(&data[v2_header_len..]); // quality 起的全部剩余字节
        let c1 = unpack_container(&v1).unwrap();
        assert_eq!((c1.width, c1.height, c1.fps, c1.gop), (64, 64, 30, 3));
        let dec_v1 = decode_mp4_video(&v1).unwrap();
        assert_eq!(dec_v1.len(), dec.len());
        for (i, (x, y)) in dec.iter().zip(dec_v1.iter()).enumerate() {
            assert_eq!(x, y, "v1 向后读第 {i} 帧与 v2 不一致");
        }

        // 其他版本 → 明确报错。
        let mut bad = data.clone();
        bad[4] = 9;
        let err = decode_mp4_video(&bad).unwrap_err();
        assert!(err.contains("不支持的容器版本"), "错误信息应指明版本：{err}");
    }

    #[test]
    fn decode_rejects_b_frame_type() {
        // 手工构造 v2 容器：单帧、帧型 2（B 帧，容器保留但未实装）→ 必须明确报错。
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"PRMV");
        buf.push(2); // v2
        buf.extend_from_slice(&16u32.to_be_bytes());
        buf.extend_from_slice(&16u32.to_be_bytes());
        buf.extend_from_slice(&24u32.to_be_bytes());
        buf.extend_from_slice(&12u16.to_be_bytes());
        buf.push(85);
        buf.extend_from_slice(&1u32.to_be_bytes()); // 帧数 = 1
        buf.push(2); // 帧型 2 = B
        buf.extend_from_slice(&0u32.to_be_bytes()); // 数据长度 0
        let err = decode_mp4_video(&buf).unwrap_err();
        assert!(
            err.contains("B 帧编码未实装"),
            "B 帧帧型应报'未实装'，实际：{err}"
        );
        // 未知帧型 3 也应报错（不静默当 P 帧）。
        let mut buf3 = buf.clone();
        let kind_pos = 4 + 1 + 4 + 4 + 4 + 2 + 1 + 4;
        buf3[kind_pos] = 3;
        assert!(decode_mp4_video(&buf3).is_err());
    }

    #[test]
    fn rejects_bad_params() {
        let frames = gen_test_frames(16, 16, 2);
        let spec = TrackSpec { width: 16, height: 16, fps: 24 };
        assert!(write_mp4_video(&frames, spec, VideoConfig { gop: 0, ..Default::default() })
            .is_err());
        assert!(write_mp4_video(&[], spec, VideoConfig::default()).is_err());
        let bad = vec![vec![0u8; 10]];
        assert!(write_mp4_video(&bad, spec, VideoConfig::default()).is_err());
    }

    #[test]
    fn container_roundtrip_frame_table() {
        // 验证帧表标记正确（首帧 I，其余 P）。
        let frames = gen_test_frames(64, 64, 13);
        let spec = TrackSpec { width: 64, height: 64, fps: 30 };
        let data = write_mp4_video(&frames, spec, VideoConfig { gop: 4, ..Default::default() })
            .unwrap();
        let decoded = decode_mp4_video(&data).unwrap();
        assert_eq!(decoded.len(), 13);
        let psnr = compute_psnr(&frames, &decoded);
        assert!(psnr >= 40.0, "PSNR {psnr:.2} < 40");
    }

    // -----------------------------------------------------------------------
    // 硬指标报告（720p @ 24fps，5 秒 = 120 帧：压缩率 / PSNR / 耗时）原先是
    // 本模块一条 `#[ignore]` 测试。它不是待修缺陷，而是跑得慢的测量程序——
    // 挂 `#[ignore]` 与"注释掉测试"同属假绿手段，撞 `qa_ignore_guard` 红线。
    // 已按门禁文档推荐处置移出测试：改为可执行 example，逻辑照搬，数值可比。
    //
    //   cargo run --release -p prism-mux --example report_720p
    //
    // 见 `examples/report_720p.rs` 头部注释（与 prism-text 的
    // `regen_glyph_golden` 先例同一形态）。
    // -----------------------------------------------------------------------
}
