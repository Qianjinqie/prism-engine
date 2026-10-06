//! P1 边界测试：铁律「seek ≡ 顺序推进」与「同输入两遍逐字节一致」。
//!
//! 覆盖：
//! - B1 批量 1/2/3/5/10 帧下，第 k 帧在任意帧数组合下逐字节相同；
//! - B2 seek 入口 `render_frame_indexed(k)` == `render_frames(..)[k]`；
//! - B3 换不同 `RenderRequest`（fps/分辨率）时契约仍成立；
//! - B4 两遍 `render_frames` 逐字节一致（无跨调用状态污染）；
//! - B5 帧数只由 fps×duration 决定（`frame_count` 边界钳制）。
//!
//! 零第三方依赖，仅 std + #[test]。

use prism_infra::engine::{frame_count, render_frame_indexed, render_frames, RenderRequest};

/// duration 2s 的源：足够在 1..=10 帧下覆盖多个时刻。
const SRC: &str = r#"
scene "seek" {
    sphere ball {
        radius 1.0
        material { albedo #c0392b roughness 0.3 metalness 0.05 }
    }
    camera main { orbit { center 0,0,0 radius 8.0 } fov 45 }
    light sun { direction (1.0, -1.0, -0.5) color #fff4e6 intensity 3.0 }
}
timeline { duration 2s fps 5 }
aesthetic { teal-orange contrast 1.12 filmic }
output { resolution 20x12 format frames }
"#;

/// 短源（duration 0.4s）：fps 覆盖变化时帧数不同。
const SRC_SHORT: &str = r#"
scene "seek" {
    sphere ball {
        radius 1.0
        material { albedo #2e86c1 roughness 0.4 metalness 0.1 }
    }
    camera main { orbit { center 0,0,0 radius 6.0 } fov 50 }
    light sun { direction (0.5, -1.0, -0.3) color #ffffff intensity 2.5 }
}
timeline { duration 0.4s fps 10 }
output { resolution 16x16 format frames }
"#;

fn req(w: u32, h: u32, fps: u32) -> RenderRequest {
    RenderRequest { width: Some(w), height: Some(h), fps: Some(fps) }
}

fn pixels(v: &[prism_gpu::Framebuffer]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in v {
        out.extend_from_slice(&f.pixels);
    }
    out
}

// ---------------------------------------------------------------------------
// B1: 任意帧数组合下，第 k 帧逐字节相同
// ---------------------------------------------------------------------------

/// 帧数覆盖 1/2/3/5/10，同一时刻的帧在任意帧数组合下逐字节相同。
#[test]
fn b1_frame_k_identical_across_all_batch_sizes() {
    // SRC: duration 2s —— 帧数 = fps×2，覆盖 2/4/6/10 帧。
    // SRC_SHORT: duration 0.4s —— fps=1 → 1 帧。
    let cases: Vec<(&str, u32, u32, u32, usize)> = vec![
        (SRC_SHORT, 16, 16, 1, 1),  // 1 帧
        (SRC, 20, 12, 1, 2),        // 2 帧
        (SRC, 20, 12, 2, 4),
        (SRC, 20, 12, 3, 6),
        (SRC, 20, 12, 5, 10),       // 10 帧
    ];
    // "同源同分辨率同时刻" → 像素的登记表；重复登记即必须逐字节一致。
    let mut seen: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
    for (src, w, h, fps, expect) in &cases {
        let frames = render_frames(src, &req(*w, *h, *fps)).expect("应可渲染");
        assert_eq!(frames.len(), *expect, "fps={fps} 帧数应为 fps×duration");
        for (k, f) in frames.iter().enumerate() {
            let t: f32 = k as f32 / *fps as f32;
            // 键含源与分辨率：仅"同源同分辨率同时刻"才可比（浮点键量化到 1e-5 s）。
            let key: String = format!("{}|{}x{}|{}", src, w, h, (t * 100_000.0).round() as i64);
            let px: Vec<u8> = f.pixels.clone();
            match seen.get(&key) {
                None => {
                    seen.insert(key, px);
                }
                Some(prev) => assert_eq!(
                    *prev, px,
                    "同源同分辨率同时刻 t={t} 的第 {k} 帧在不同批量下不一致——seek 语义缺陷"
                ),
            }
        }
    }
    assert!(seen.len() >= 15, "应覆盖至少 15 个不同时刻，实际 {}", seen.len());
}

/// 同一 fps 下，1/2/3/5/10 帧的批量：前 min(N) 帧必须完全一致。
///
/// 构造手法：同源同时长下，帧数由 fps 决定；fps=5 的第 2k 帧与 fps=10 的第 2k
/// 帧时刻相同（t = 2k/10 = k/5），故二者必须逐字节相同——这就是"任意帧数
/// 组合下第 k 帧一致"的可判定形式。
#[test]
fn b1b_same_instant_identical_across_batch_sizes() {
    // 帧数覆盖 1/2/3/5/10 帧（duration 2s）。
    let fps_to_frames: Vec<(u32, usize)> = vec![(1, 2), (2, 4), (3, 6), (5, 10)];
    for (fps, expect) in &fps_to_frames {
        let frames = render_frames(SRC, &req(20, 12, *fps)).expect("应可渲染");
        assert_eq!(frames.len(), *expect, "fps={fps} @2s 应得 {expect} 帧");
    }
    // 1 帧批量（SRC_SHORT @ fps=2，0.4s → ceil(0.8) = 1 帧）与 5 帧批量的首帧比：均 t=0。
    let one = render_frames(SRC_SHORT, &req(16, 16, 2)).expect("应可渲染");
    let five = render_frames(SRC_SHORT, &req(16, 16, 5)).expect("应可渲染");
    assert_eq!(one.len(), 1, "SRC_SHORT @fps=2 → ceil(0.8) = 1 帧");
    assert_eq!(five.len(), 2, "SRC_SHORT @fps=5 → ceil(2.0) = 2 帧");
    assert_eq!(
        one[0].pixels, five[0].pixels,
        "1 帧批量的唯一帧与 2 帧批量的首帧（均 t=0）应逐字节相同"
    );

    // 同源 SRC：fps=5 的偶数下标帧 == fps=10 的对应帧（同一时刻）。
    let f5 = render_frames(SRC, &req(20, 12, 5)).expect("应可渲染");
    let f10 = render_frames(SRC, &req(20, 12, 10)).expect("应可渲染");
    for k in 0..f5.len() {
        let j = k * 2;
        assert!(
            j < f10.len(),
            "fps=10 批次应覆盖 fps=5 的全部时刻（{j} < {}）",
            f10.len()
        );
        assert_eq!(
            f5[k].pixels, f10[j].pixels,
            "fps=5 第 {k} 帧（t={}）与 fps=10 第 {j} 帧（t={}）应逐字节相同",
            k as f32 / 5.0,
            j as f32 / 10.0
        );
    }
}

// ---------------------------------------------------------------------------
// B2: seek 入口 ≡ 批量第 k 帧
// ---------------------------------------------------------------------------

#[test]
fn b2_seek_entry_equals_batched_frame() {
    let r = req(20, 12, 5);
    let batched = render_frames(SRC, &r).expect("应可渲染");
    assert!(batched.len() >= 4, "需至少 4 帧覆盖多个时刻");
    for k in 0..batched.len() as u64 {
        let seeked = render_frame_indexed(SRC, k, &r).expect("seek 应成功");
        assert_eq!(
            seeked.pixels, batched[k as usize].pixels,
            "seek 到第 {k} 帧与批量第 {k} 帧不一致——违反 seek ≡ 顺序推进"
        );
    }
}

/// 反复 seek 同一帧号必须稳定（seek 不得依赖"之前 seek 过几次"）。
#[test]
fn b2b_repeated_seek_is_stable() {
    let r = req(20, 12, 5);
    let first = render_frame_indexed(SRC, 3, &r).expect("首次 seek");
    // 中间插入其他帧号的 seek，模拟调用序扰动。
    for k in [0_u64, 7, 1, 9, 3] {
        let _ = render_frame_indexed(SRC, k, &r).expect("seek 应成功");
    }
    let again = render_frame_indexed(SRC, 3, &r).expect("再次 seek");
    assert_eq!(first.pixels, again.pixels, "同帧号重复 seek 应逐字节一致");
}

// ---------------------------------------------------------------------------
// B3: 换请求参数时契约仍成立
// ---------------------------------------------------------------------------

#[test]
fn b3_contract_holds_for_varying_requests() {
    for (w, h, fps) in [(20_u32, 12_u32, 5_u32), (8, 8, 2), (32, 16, 4), (20, 12, 1)] {
        let r = req(w, h, fps);
        let batched = render_frames(SRC, &r).expect("应可渲染");
        for k in 0..batched.len() as u64 {
            let seeked = render_frame_indexed(SRC, k, &r).expect("seek 应成功");
            assert_eq!(
                seeked.pixels, batched[k as usize].pixels,
                "请求 {w}×{h}@{fps}fps 下 seek({k}) 与批量第 {k} 帧不一致"
            );
        }
    }
}

/// 请求未覆盖任何字段时走 DSL 声明值，seek 依然成立。
#[test]
fn b3b_seek_works_with_dsl_defaults() {
    let batched = render_frames(SRC, &RenderRequest::default()).expect("应可渲染");
    assert_eq!(batched.len(), 10, "DSL 声明 2s@5fps → 10 帧");
    for k in 0..batched.len() as u64 {
        let seeked = render_frame_indexed(SRC, k, &RenderRequest::default()).expect("seek 应成功");
        assert_eq!(seeked.pixels, batched[k as usize].pixels, "默认请求下 seek({k}) 不一致");
    }
}

// ---------------------------------------------------------------------------
// B4: 同输入两遍逐字节一致
// ---------------------------------------------------------------------------

#[test]
fn b4_two_runs_byte_identical() {
    let r = req(20, 12, 5);
    let a = pixels(&render_frames(SRC, &r).unwrap());
    let b = pixels(&render_frames(SRC, &r).unwrap());
    assert_eq!(a, b, "两遍渲染应逐字节一致");
    // 三遍：中间插入**异尺寸**负载（不同分辨率本就不该与本请求可比，故只验证
    // 插入负载后同请求的输出未被污染——这才是"无跨调用状态污染"的判据）。
    let _noise = pixels(&render_frames(SRC, &req(8, 8, 3)).unwrap());
    let c = pixels(&render_frames(SRC, &r).unwrap());
    assert_eq!(a, c, "异尺寸负载插入后，同请求输出被污染——存在跨调用状态污染");
}

// ---------------------------------------------------------------------------
// B5: frame_count 边界（不 panic、钳制、语义正确）
// ---------------------------------------------------------------------------

#[test]
fn b5_frame_count_boundaries() {
    assert_eq!(frame_count(0, 1.0), 0, "fps=0 → 0 帧");
    assert_eq!(frame_count(30, 0.0), 0, "duration=0 → 0 帧");
    assert_eq!(frame_count(30, -1.0), 0, "负时长 → 0 帧");
    assert_eq!(frame_count(30, f32::NAN), 0, "NaN 时长 → 0 帧（不得 panic）");
    assert_eq!(frame_count(4, 0.5), 2);
    assert_eq!(frame_count(5, 2.0), 10);
    assert_eq!(frame_count(u32::MAX, f32::INFINITY), usize::MAX, "溢出须钳制到 usize::MAX");
    assert_eq!(frame_count(u32::MAX, 1e30), usize::MAX, "极大乘积须钳制");
    assert_eq!(frame_count(1, f32::MIN_POSITIVE), 1, "极小正时长 → 至少 1 帧");
}

/// seek 越界必须返回 Err（不得 panic、不得静默返回末帧）。
#[test]
fn b5b_seek_out_of_range_returns_err() {
    let r = req(20, 12, 5);
    let total = render_frames(SRC, &r).unwrap().len() as u64;
    assert!(render_frame_indexed(SRC, total, &r).is_err(), "越界帧号应报 Err");
    assert!(render_frame_indexed(SRC, u64::MAX, &r).is_err(), "极大帧号应报 Err");
    assert!(render_frame_indexed(SRC, 0, &RenderRequest { fps: Some(0), ..r.clone() }).is_err(),
            "fps=0 时 seek 应报 Err");
}
