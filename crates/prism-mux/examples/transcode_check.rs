//! 转码交付自检（**工具形态，非常规测试**）：探测 → 决策 → 交付 → 打印一行结论。
//!
//! # 职责
//!
//! 端到端走一遍 V1.5-E 的交付链，把"能不能交 H.264"这件事**明码打印**，而不是让
//! 调用方在日志里猜：
//!
//! 1. 合成 8 帧 96×54@10fps 的确定性图案（LCG + 平移方块，无 `rand` 依赖）；
//! 2. [`prism_mux::extprobe::FfmpegProbe::detect`] 打印外部 ffmpeg 探测结果；
//! 3. [`prism_mux::extprobe::decide`] 打印降级链决策（目标 h264 → 实际编码族）；
//! 4. 有 ffmpeg：经 stdin 管道喂裸 RGBA 交 `libx264`，打印产物字节数；
//!    无 ffmpeg：打印 MJPEG 降级**三段式**说明（在哪 / 什么事 / 怎么改），并用 crate
//!    内置 MJPEG 出口（AVI-MJPEG）真正交付一份可播文件——降级不等于失败；
//! 5. 末行固定 `TRANSCODE_CHECK: mode=... frames=8`。
//!
//! # 两条路径都以退出码 0 收尾
//!
//! 本程序是**自检报告器**，不是门禁：无 ffmpeg 是合法环境（D-17 三层策略的第 ③ 层
//! 就是内置 MJPEG），不该判红。真出问题（合成/编码失败）才返回非 0。
//!
//! # 用法
//! ```bash
//! cargo run -p prism-mux --example transcode_check
//! ```

use prism_mux::extprobe::{decide, delivery_chain, ffmpeg_pipe_args, pipe_frames_to_ffmpeg,
                          Codec, FfmpegProbe};
use prism_mux::hwenc::{delivery_plan, HwBackend};

/// 线性同余发生器（确定性图案源；零外部 crate）。
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
}

/// 合成 8 帧 RGBA8：LCG 噪声底纹 + 每帧右移 7 像素的亮块（制造可预测的运动）。
fn synth_frames(w: u32, h: u32, n: u32) -> Vec<Vec<u8>> {
    let mut lcg = Lcg::new(0x2024_0E01);
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    let box_side = 16u32;
    for i in 0..n {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        // 底纹：低频纹理 + LCG 抖动（保证 P 帧有残差，不是纯色）。
        for y in 0..h {
            for x in 0..w {
                let r = lcg.next_u32();
                let base = ((x * 3 + y * 5) % 61) as i32 + ((r >> 28) as i32 % 9) - 4 + 90;
                let idx = ((y * w + x) * 4) as usize;
                buf[idx] = base.clamp(0, 255) as u8;
                buf[idx + 1] = (base / 2 + 40).clamp(0, 255) as u8;
                buf[idx + 2] = (255 - base).clamp(0, 255) as u8;
                buf[idx + 3] = 255;
            }
        }
        // 移动亮块：x = (i·7) % (w − 16)，y 固定居中。
        let bx0 = (i * 7) % (w.saturating_sub(box_side).max(1));
        let by0 = h.saturating_sub(box_side) / 2;
        for yy in 0..box_side {
            for xx in 0..box_side {
                let x = bx0 + xx;
                let y = by0 + yy;
                if x >= w || y >= h {
                    continue;
                }
                let idx = ((y * w + x) * 4) as usize;
                buf[idx] = 245;
                buf[idx + 1] = 230;
                buf[idx + 2] = 40;
                buf[idx + 3] = 255;
            }
        }
        frames.push(buf);
    }
    frames
}

fn main() {
    let w = 96u32;
    let h = 54u32;
    let fps = 10u32;
    let n = 8u32;
    let frames = synth_frames(w, h, n);
    println!("== 合成输入：{w}×{h}@{fps}fps × {n} 帧 RGBA8 ==");

    // ---- 1) 外部 ffmpeg 探测 ----
    let probe = FfmpegProbe::detect();
    println!(
        "detect(): available={} path={:?} encoders(h264={}, hevc={}, hw={})",
        probe.available,
        probe.path,
        probe.encoders.has_h264(),
        probe.encoders.has_hevc(),
        probe.encoders.hardware
    );

    // ---- 2) 降级链决策 ----
    let d = decide(Codec::H264, &probe);
    let chain: Vec<&str> = delivery_chain(Codec::H264).iter().map(|c| c.name()).collect();
    println!(
        "decide(): requested={} → codec={} degraded={} | 降级链={:?}",
        d.requested.name(),
        d.codec.name(),
        d.degraded,
        chain
    );

    // ---- 2b) 硬件后端探测 + 交付计划（只读环境检查，无 FFI）----
    let plan = delivery_plan(HwBackend::Vaapi);
    println!(
        "hwenc::delivery_plan(Vaapi): chosen={:?} reasons={:?}",
        plan.chosen.map(|b| b.name()),
        plan.reasons
    );

    // ---- 3) 交付 ----
    let out_path = std::env::temp_dir().join("prism_transcode_check.mp4");
    let out_str = out_path.to_string_lossy().to_string();
    // 默认终态 MJPEG（D-10 内置链永远可用）；ffmpeg 可用时升级为 h264。
    let mode: &str;
    if probe.is_available() {
        println!("ffmpeg 参数：{:?}", ffmpeg_pipe_args(w, h, fps, &out_str));
        match pipe_frames_to_ffmpeg(&frames, w, h, fps, &out_str) {
            Ok(()) => {
                let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
                println!("H.264 产物：{out_str}（{size} 字节）");
                mode = "h264";
            }
            Err(e) => {
                // 有 ffmpeg 但管道失败：不静默——打印三段式并降级内置 MJPEG。
                println!("[降级] ffmpeg 管道失败：{e}");
                mode = "mjpeg";
            }
        }
    } else {
        // 无 ffmpeg：三段式说明（在哪 / 什么事 / 怎么改），再真正交 MJPEG。
        println!(
            "未检测到 ffmpeg（在哪：examples/transcode_check.rs 交付链第 3 步；什么事：PATH 中无 ffmpeg，无法调用外部 H.264）\n  已降级 MJPEG：本程序将用 crate 内置 AVI-MJPEG 出口交付\n  建议：安装 ffmpeg 可获 H.264（如 `apt install ffmpeg` / `brew install ffmpeg`）"
        );
        mode = "mjpeg";
    }

    if mode == "mjpeg" {
        match prism_mux::write_avi_mjpeg(&frames, w, h, fps, 80) {
            Ok(bytes) => {
                let path = std::env::temp_dir().join("prism_transcode_check.avi");
                match std::fs::write(&path, &bytes) {
                    Ok(()) => println!(
                        "MJPEG 降级产物：{}（{} 字节）",
                        path.to_string_lossy(),
                        bytes.len()
                    ),
                    Err(e) => println!("[警告] 写 MJPEG 产物失败：{e}"),
                }
            }
            Err(e) => {
                println!("[错误] 内置 MJPEG 出口失败：{e}");
                std::process::exit(1);
            }
        }
    }

    println!("TRANSCODE_CHECK: mode={mode} frames={n}");
}
