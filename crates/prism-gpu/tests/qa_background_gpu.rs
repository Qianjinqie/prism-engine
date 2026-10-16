//! 背景渲染（GPU 管线兜底）数值一致性测试。
//!
//! 覆盖交付要求的三类断言：
//! (a) 输出非全黑、存在垂直渐变（顶/底像素亮度可测差异）；
//! (b) 输出 RGBA 各通道 ∈ [0,255] 且 alpha == 255；
//! (c) 同一 (w,h,t,seed) 两次调用结果逐字节一致（确定性）。
//!
//! 另含一项集成断言：确认 `shaders/background.wgsl` 已嵌入并通过 D-29 结构自检
//! （与既有 `cargo test` 的 WGSL 自检形成双重保险）。
//!
//! 零第三方依赖，仅 std + #[test]。

use prism_gpu::shaders::{check_shader, source};
use prism_gpu::{render_background, render_background_cpu};

/// ITU-R BT.601 luma（仅用于「垂直渐变」可测性断言）。
fn luma(rgb: &[u8]) -> f32 {
    0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32
}

#[test]
fn output_is_not_all_black_and_has_vertical_gradient() {
    let w: u32 = 64;
    let h: u32 = 32;
    let buf = render_background_cpu(w, h, 1.25, 0x9E37_79B1);
    assert_eq!(buf.len(), (w * h * 4) as usize, "RGBA8 长度应为 w*h*4");

    // 非全黑：至少存在一个非零像素（背景恒有底色）。
    assert!(buf.iter().any(|&b| b != 0), "背景输出不应全黑");

    // 垂直渐变：顶行平均亮度与底行平均亮度应有可测差异（BG_BOTTOM 比 BG_TOP 亮，
    // 即便辉光偏上，渐变两端差异仍清晰可见）。
    let row_luma = |y: u32| -> f32 {
        let mut sum = 0.0f32;
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            sum += luma(&buf[i..i + 3]);
        }
        sum / w as f32
    };
    let top = row_luma(0);
    let bottom = row_luma(h - 1);
    let diff = (top - bottom).abs();
    assert!(
        diff > 1.0,
        "顶/底行亮度差异应可测（实际 diff={diff:.3}），否则垂直渐变未生效"
    );
}

#[test]
fn channels_in_range_and_alpha_is_255() {
    let w: u32 = 48;
    let h: u32 = 24;
    let t_cases: [f32; 3] = [0.0, 1.25, 3.7];
    for &t in &t_cases {
        let buf = render_background_cpu(w, h, t, 0x1234_5678);
        assert_eq!(buf.len(), (w * h * 4) as usize);
        // 所有通道 ∈ [0,255]。
        for (i, &b) in buf.iter().enumerate() {
            assert!((0..=255).contains(&b), "通道越界 @{}: {}", i, b);
        }
        // alpha 必须恒为 255。
        for y in 0..h {
            for x in 0..w {
                let a = buf[((y * w + x) * 4 + 3) as usize];
                assert_eq!(a, 255, "alpha 必须恒为 255");
            }
        }
    }
}

#[test]
fn deterministic_same_input_same_bytes() {
    let w: u32 = 40;
    let h: u32 = 20;
    let t = 2.5f32;
    let seed: u64 = 0xDEAD_BEEF;
    let a = render_background_cpu(w, h, t, seed);
    let b = render_background_cpu(w, h, t, seed);
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b, "同一 (w,h,t,seed) 两次调用必须逐字节一致");

    // `render_background` 当前即 CPU 参考，亦应一致。
    let c = render_background(w, h, t, seed);
    assert_eq!(c, a, "render_background 应与 CPU 参考逐字节一致");
}

#[test]
fn embedded_background_shader_passes_structure_check() {
    // 确认新 shader 已嵌入 SHADERS 表，并通过 D-29 结构自检（无 libm / 有入口 /
    // 括号配平 / 无第三方 include）。
    let src = source("background.wgsl").expect("background.wgsl 应已通过 SHADERS 表嵌入");
    let chk = check_shader("background.wgsl", src);
    assert!(chk.has_entry, "background.wgsl 应含 @compute 入口");
    assert!(chk.brackets_balanced, "background.wgsl 括号应配平");
    assert!(chk.fn_count > 0, "background.wgsl 应含函数声明");
    assert!(!chk.has_libm, "background.wgsl 不得含被禁 libm 超越函数（D-29）");
    assert!(!chk.has_include, "background.wgsl 不得含第三方 include");
    assert!(chk.is_pass(), "background.wgsl 结构自检应整体通过：{:?}", chk);
}
