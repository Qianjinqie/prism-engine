//! PRMV 时序视频 720p 全序列硬指标报告（**工具形态，非常规测试**）。
//!
//! # 为什么是 `examples/` 而不是 `#[ignore]` 测试
//!
//! 本文件原先是 `src/video.rs` 测试模块里一条 `#[ignore]` 的
//! `report_720p_full_sequence`。它不是待修缺陷，而是一条**跑得慢的测量
//! 程序**（720p × 120 帧全序列编码 + 解码 + MJPEG 对照）。把它挂
//! `#[ignore]` 换"常规 `cargo test` 不拖慢"，与"注释掉测试"是同一类
//! 假绿手段——`prism-core` 的 `qa_ignore_guard` 门禁因此判红。
//! 门禁文档给出的合法处置即：**把生成器/报告器移出测试，移到
//! `examples/`**（先例：`prism-text/examples/regen_glyph_golden.rs`）。
//! 故改造为独立可执行入口：
//! - **不进 `cargo test`**，不拖慢常规测试，也不再产生 ignored 计数；
//! - 形态与职责自明（报告器就该是可执行程序，不是测试）；
//! - 生成/编码逻辑照搬原用例，报告数值与原 `--ignored` 运行一致。
//!
//! # 用法
//!
//! ```bash
//! # release 下运行（720p × 120 帧，debug 会非常慢）
//! cargo run --release -p prism-mux --example report_720p
//! ```
//!
//! # 报告内容
//!
//! 720p @ 24fps、5 秒 = 120 帧：PRMV（I/P 帧间预测）编码体积、同内容
//! MJPEG 体积、压缩比、往返 PSNR、编码/解码耗时。末段按原测试的三条
//! 硬指标逐项核验（帧数一致 / PSNR ≥ 40 dB / 体积 < MJPEG），打印
//! `[通过]`/`[未过]` 与实际值；任一未过以退出码 1 收尾。
//!
//! # 为什么原 assert 改为核验输出
//!
//! 原测试用 `assert!` 是因为它是**门禁**；本文件是**报告器**——panic
//! 会把"当前编码器未达标"摔成栈回溯，而报告器的职责是把测量结果与
//! 指标预期之间的差距**明码呈现**。这与本文件存在的理由同源：不许用
//! 任何手段（`#[ignore]`、注释、panic 掩盖）把未达标事实从检查视野里
//! 抹掉。硬指标门禁本身仍由 `video.rs` 内的小尺寸快速测试把持。
//!
//! # 纪律
//!
//! 修改 `video.rs` 的编码参数（gop/quality/残差质量窗）后重跑本
//! example，确认硬指标仍满足再合入；本文件不写盘、不依赖外部 crate。

use prism_mux::{
    decode_mp4_video, write_mp4_jpeg_color, write_mp4_video, JpegConfig, TrackSpec, VideoConfig,
};

/// 生成平滑运动测试序列：背景渐变 + 缓慢向右平移的色块。
///
/// 与 `src/video.rs` 测试模块的 `gen_test_frames` 逐行一致——报告数值
/// 必须与原 `#[ignore]` 用例可比，故不接受任何参数化改动。
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
///
/// 与 `src/video.rs` 测试模块的 `compute_psnr` 逐行一致。
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

fn main() {
    let w = 1280u32;
    let h = 720u32;
    let fps = 24u32;
    let nframes = 120u32;
    let frames = gen_test_frames(w, h, nframes);
    let spec = TrackSpec { width: w, height: h, fps };
    let cfg = VideoConfig { gop: 24, quality: 85, search_radius: 16, subpel: true };

    let t0 = std::time::Instant::now();
    let data = write_mp4_video(&frames, spec, cfg).unwrap();
    let encode_ms = t0.elapsed().as_millis();

    let t1 = std::time::Instant::now();
    let decoded = decode_mp4_video(&data).unwrap();
    let decode_ms = t1.elapsed().as_millis();

    let psnr = compute_psnr(&frames, &decoded);
    let mjpeg = write_mp4_jpeg_color(
        &frames,
        spec,
        None,
        0,
        0,
        JpegConfig { quality: 85, grayscale: false },
    )
    .unwrap();

    println!(
        "[PRMV 720p 报告] 帧={nframes} 分辨率={w}x{h} gop={gop} quality={q}\n\
         \x20 编码体积={} B ({:.2} MB) | MJPEG体积={} B ({:.2} MB) | 压缩比={:.2}x\n\
         \x20 往返PSNR={:.2} dB | 编码耗时={} ms | 解码耗时={} ms | 解码帧数={}",
        data.len(),
        data.len() as f64 / 1e6,
        mjpeg.len(),
        mjpeg.len() as f64 / 1e6,
        mjpeg.len() as f64 / data.len() as f64,
        psnr,
        encode_ms,
        decode_ms,
        decoded.len(),
        gop = cfg.gop,
        q = cfg.quality,
    );
    // 硬指标核验：原测试的三条 assert。本文件是报告器而非门禁，故逐项
    // 打印 [通过]/[未过] 与实际值；任一未过以退出码 1 收尾，把差距显式
    // 呈现在检查视野里，而不是 panic 摔栈或视而不见。
    let frames_ok = decoded.len() == frames.len();
    let psnr_ok = psnr >= 40.0;
    let size_ok = data.len() < mjpeg.len();
    println!(
        "[硬指标核验] 帧数一致(120)[{}] | PSNR>=40dB[{}] | 体积<MJPEG[{}]",
        mark(frames_ok),
        mark(psnr_ok),
        mark(size_ok)
    );
    if !frames_ok {
        eprintln!(
            "解码帧数 {} != 原始 {}：容器往返损坏，属实现缺陷。",
            decoded.len(),
            frames.len()
        );
        std::process::exit(1);
    }
    if !psnr_ok || !size_ok {
        eprintln!(
            "质量/体积硬指标未达标（PSNR {psnr:.2} dB，体积比 MJPEG 小 {:.1}%）。\
             这是对当前编码器参数的如实测量，非 example 自身缺陷；\
             调参（gop/quality/残差质量窗）后重跑，或与维护者确认指标预期。",
            (1.0 - data.len() as f64 / mjpeg.len() as f64) * 100.0
        );
        std::process::exit(1);
    }
}

/// 核验标记：通过为 `[通过]`，未过为 `[未过]`。
fn mark(ok: bool) -> &'static str {
    if ok {
        "[通过]"
    } else {
        "[未过]"
    }
}
