//! QA 全仓缺陷普查 · 审计 2（确定性）攻击测试。
//!
//! 范围：`prism-infra::engine`（DSL → 渲染全链路主入口）+ `prism-io` 编码出口。
//! 契约：同输入两遍**逐字节一致**（铁律 3）。
//! 全部断言为字节级比对；失败即确定性缺陷。
//!
//! 零第三方依赖，仅 std + #[test]。

use prism_infra::engine::{
    frame_count, frame_time, render_frame_indexed, render_frames, RenderRequest,
};

const SRC: &str = r#"
scene "qa" {
    sphere ball {
        radius 1.0
        material {
            albedo #c0392b
            roughness 0.3
            metalness 0.05
        }
    }
    camera main {
        orbit {
            center 0,0,0
            radius 8.0
        }
        fov 45
    }
    light sun {
        direction (1.0, -1.0, -0.5)
        color #fff4e6
        intensity 3.0
    }
}
timeline {
    duration 0.5s
    fps 4
}
aesthetic {
    teal-orange
    contrast 1.12
    filmic
}
output {
    resolution 24x16
    format frames
}
"#;

fn req() -> RenderRequest {
    RenderRequest { width: Some(24), height: Some(16), fps: Some(4) }
}

fn pixels(v: &[prism_gpu::Framebuffer]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in v {
        out.extend_from_slice(&f.pixels);
    }
    out
}

// ---------------------------------------------------------------------------
// D-1: 同输入两遍逐字节一致
// ---------------------------------------------------------------------------

#[test]
fn d01_render_frames_two_runs_byte_identical() {
    let a = render_frames(SRC, &req()).expect("首跑应成功");
    let b = render_frames(SRC, &req()).expect("次跑应成功");
    assert_eq!(a.len(), b.len(), "两遍帧数应一致");
    assert_eq!(pixels(&a), pixels(&b), "两遍渲染输出非逐字节一致——确定性缺陷");
}

/// 三遍稳定性（含中间插入其他负载，排除缓存/池化状态污染）。
#[test]
fn d02_render_frames_three_runs_stable_under_interleaved_load() {
    let a = pixels(&render_frames(SRC, &req()).unwrap());
    // 插入一次不同尺寸的渲染，模拟全局资源池/缓存被扰动
    let other = RenderRequest { width: Some(8), height: Some(8), fps: Some(2) };
    let _ = render_frames(SRC, &other).unwrap();
    let b = pixels(&render_frames(SRC, &req()).unwrap());
    let _ = render_frames(SRC, &other).unwrap();
    let c = pixels(&render_frames(SRC, &req()).unwrap());
    assert_eq!(a, b, "第 2 遍与第 1 遍不一致——存在跨调用状态污染");
    assert_eq!(b, c, "第 3 遍与第 2 遍不一致——存在跨调用状态污染");
}

// ---------------------------------------------------------------------------
// D-2: 畸形 DSL 输入不得 panic（错误须走 Result，不得 panic）
// ---------------------------------------------------------------------------

/// 各类畸形/极端 DSL 源码：只允许 Err，不允许 panic。
#[test]
fn d03_malformed_dsl_sources_do_not_panic() {
    let sources: Vec<String> = vec![
        String::new(),
        "scene".to_string(),
        "scene {".to_string(),
        "scene { }".to_string(),
        "{{{".to_string(),
        "}}}".to_string(),
        "output { resolution 0x0 }".to_string(),
        "output { resolution 99999999999x99999999999 }".to_string(),
        "timeline { duration 0s fps 0 }".to_string(),
        "camera { fov NaN }".to_string(),
        "sphere { radius 1e400 }".to_string(),
        "a".repeat(100_000),
        "scene {".to_string() + &"sphere {".repeat(5_000),
    ];
    for s in sources {
        let r = std::panic::catch_unwind(|| render_frames(&s, &req()));
        assert!(r.is_ok(), "DSL 源码（len={}）导致 panic：{:?}", s.len(), &s[..s.len().min(60)]);
        // 若 Ok，帧缓冲长度须与请求尺寸自洽
        if let Ok(Ok(frames)) = r {
            for f in &frames {
                assert_eq!(
                    f.pixels.len(),
                    24 * 16 * 4,
                    "成功路径下帧缓冲尺寸须 == w*h*4（尺寸不变量）"
                );
            }
        }
    }
}

/// 深嵌套括号（栈溢出探测）。逐级加压到 10 万层。
#[test]
fn d04_deep_nesting_no_stack_overflow() {
    for depth in [1_000usize, 10_000, 100_000] {
        let src = format!(
            "{}",
            "a { ".repeat(depth)
        );
        let r = std::panic::catch_unwind(|| render_frames(&src, &req()));
        assert!(r.is_ok(), "深嵌套 depth={depth} 导致 panic（栈溢出）");
    }
}

// ---------------------------------------------------------------------------
// D-3: 极端尺寸参数
// ---------------------------------------------------------------------------

/// 各种 width/height/fps 组合：只允许 Ok 或 Err，不得 panic / abort。
#[test]
fn d05_extreme_dimension_requests_do_not_panic() {
    let dims: Vec<(u32, u32, u32)> = vec![
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 0),
        (65535, 65535, 1),
        (1, 1, 100_000),
        (u32::MAX, 1, 1),
        (1, u32::MAX, 1),
    ];
    for (w, h, f) in dims {
        let rq = RenderRequest { width: Some(w), height: Some(h), fps: Some(f) };
        let r = std::panic::catch_unwind(|| render_frames(SRC, &rq));
        assert!(r.is_ok(), "尺寸 ({w},{h},{f}) 导致 panic");
    }
}

// ---------------------------------------------------------------------------
// D-4: seek ≡ 顺序推进（铁律 3）
// ---------------------------------------------------------------------------

/// **正向契约**：`render_frame_indexed(SRC, k, req)` 必须逐字节等于批量渲染的第 k 帧。
///
/// # 历史（QA 裁决记录，勿回退）
///
/// 本测试此前写作「顺序推进 vs 批量」，实现方式是循环调用
/// `render_frames(SRC, &req())` 并取 `first()`。**该写法恒不成立，且与
/// 铁律 3 的本意相反**——因为 `single_src` 与 `req()` 全程不变，
/// `render_frames` 内部的帧号从 0 起算，所以每一次拿到的 `first()`
/// 都是 `t = 0` 的**同一张**第 0 帧；而批量路径的第 k 帧时刻是
/// `t = k / fps`。于是该断言实际退化为「`frame0 == frame1`」，即要求
/// 引擎把所有帧渲成同一张——数学上不可能转绿，除非把 `render_frames`
/// 退化成 `t ≡ 0` 的常量函数（那才会真的破坏铁律 3）。
///
/// 真正的 seek 入口是 `render_frame_indexed`（不渲染前序帧，直接按帧号
/// 定位）。本测试改为直接断言该契约，并在多个 fps 下交叉验证。
#[test]
fn d06_sequential_vs_batched_frames_identical() {
    let batched = render_frames(SRC, &req()).expect("批量渲染应成功");
    assert!(batched.len() >= 2, "样本须 ≥2 帧才能验证逐帧对拍，实得 {} 帧", batched.len());

    // 主契约：seek 第 k 帧 ≡ 批量第 k 帧，k 全覆盖。
    for k in 0..batched.len() as u64 {
        let seeked = render_frame_indexed(SRC, k, &req())
            .unwrap_or_else(|e| panic!("seek 第 {k} 帧应成功：{e:?}"));
        assert_eq!(
            seeked.pixels,
            batched[k as usize].pixels,
            "seek 第 {k} 帧（t={}）与批量第 {k} 帧不逐字节相同——seek 语义缺陷",
            frame_time(k, 4)
        );
    }

    // 越界必须 Err，不得 panic、不得返回垃圾帧。
    assert!(
        render_frame_indexed(SRC, batched.len() as u64, &req()).is_err(),
        "帧号越界必须返回 Err"
    );
}

/// **防退化守卫**：相邻帧必须真的不同。
///
/// 没有这条断言，`d06` 无法区分「seek 正确」与「所有帧被渲成同一张常量
/// 图」——后者的 `render_frame_indexed` 同样会逐字节等于批量结果。
/// 本场景相机绕行 0.5s 走满 360°，`t=0` 与 `t=0.25` 视点相差 180°，
/// 故像素必然不同。
#[test]
fn d06b_adjacent_frames_are_distinct_not_constant() {
    let batched = render_frames(SRC, &req()).unwrap();
    for k in 1..batched.len() {
        assert_ne!(
            batched[k - 1].pixels, batched[k].pixels,
            "第 {k} 帧与第 {} 帧逐字节相同——渲染已退化为常量帧（相机绕行应产生视差）",
            k - 1
        );
    }
}

/// 多帧率交叉验证：`fps × duration` 的帧数公式与 seek 定位必须同源。
#[test]
fn d06c_seek_equals_batched_across_fps() {
    for fps in [1u32, 2, 4, 5, 10] {
        let rq = RenderRequest { width: Some(24), height: Some(16), fps: Some(fps) };
        let batched = render_frames(SRC, &rq)
            .unwrap_or_else(|e| panic!("fps={fps} 批量渲染应成功：{e:?}"));
        assert_eq!(
            batched.len(),
            frame_count(fps, 0.5),
            "fps={fps} 帧数须与 frame_count 公式一致"
        );
        for k in 0..batched.len() as u64 {
            let seeked = render_frame_indexed(SRC, k, &rq)
                .unwrap_or_else(|e| panic!("fps={fps} seek 第 {k} 帧应成功：{e:?}"));
            assert_eq!(
                seeked.pixels, batched[k as usize].pixels,
                "fps={fps}：seek 第 {k} 帧（t={}）≠ 批量第 {k} 帧",
                frame_time(k, fps)
            );
        }
    }
}
