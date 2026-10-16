//! 例子（v4）：用**内容增强包**组一支 5 秒片头，封装为标准
//! MJPEG-in-AVI，输出到 `/workspace/prism_aurora_v4.avi`。
//!
//! 相对 `short_avi.rs`（v3）新增三件套能力：
//! 1. 新背景 [`Background::Aurora`]（青绿 / 青蓝流动极光带）；
//! 2. [`Track::Typewriter`] 打字机副标题（逐字 + 闪烁光标）；
//! 3. Clip 级片头 / 片尾淡入淡出（`render_short_avi_faded`）。
//!
//! 既有 kinetic 主标题 / 静态标签 / 装饰线 / 字幕条全部沿用——证明新能力
//! 与旧轨道可在同一支片里混用。

use prism_author::{
    render_short_avi_faded, AccentLine, Background, Caption, CaptionStyle, ClipSpec, EasingKind,
    FontKind, KineticText, StaticText, Track, TypewriterText,
};

fn main() {
    // 一份声明式规格：5 秒 1280×720 @24fps 极光片头。
    let spec = ClipSpec {
        width: 1280,
        height: 720,
        fps: 24,
        duration: 5.0,
        background: Background::Aurora,
        jpeg_quality: 88,
        tracks: vec![
            // 静态小标签（既有能力）。
            Track::Static(StaticText {
                content: "PRISM ENGINE".into(),
                font: FontKind::Regular,
                size: 22.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.3,
                in_sec: 0.5,
                letter_spacing: 7.0,
                y: 220.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            // 逐字 kinetic 主标题（既有能力）。
            Track::Kinetic(KineticText {
                content: "棱镜引擎".into(),
                font: FontKind::Bold,
                size: 96.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.4,
                stagger: 0.09,
                per_char: 0.45,
                scale_from: 0.72,
                rise_px: 22.0,
                easing: EasingKind::CubicOut,
                y: 350.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            // 装饰线（既有能力）。
            Track::Accent(AccentLine {
                center_x: 640.0,
                y: 398.0,
                w: 320.0,
                h: 3.0,
                color: [0.30, 0.75, 0.95, 0.9],
                at: 1.3,
                in_sec: 0.6,
                out_sec: 0.0,
                out_at: None,
            }),
            // 打字机副标题（新能力）：逐字打出 + 闪烁光标。
            Track::Typewriter(TypewriterText {
                content: "程序化视频 · 零外部依赖".into(),
                font: FontKind::Regular,
                size: 30.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 1.8,
                y: 470.0,
                cps: 14.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.2,
                out_sec: 0.0,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
            // 收尾字幕条（既有能力）。
            Track::Caption(Caption {
                content: "视频即代码 · 单文件源码分发".into(),
                at: 3.4,
                in_sec: 0.6,
                out_sec: 0.0,
                out_at: None,
                style: CaptionStyle {
                    font_size: 28.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                    bg: [0.0, 0.0, 0.0, 0.55],
                    pad_x: 22.0,
                    pad_y: 12.0,
                    margin_bottom: 40.0,
                    line_spacing: 0.0,
                },
            }),
        ],
    };

    // 新能力：Clip 级片头 0.8s 淡入 + 片尾 0.8s 淡出（纯后处理，
    // 不影响任何既有入口）。
    let avi = render_short_avi_faded(&spec, 0.8, 0.8).expect("AVI 封装失败");
    std::fs::write("/workspace/prism_aurora_v4.avi", &avi).expect("写入 AVI 失败");
    eprintln!(
        "prism_author::render_short_avi_faded 完成：{} 字节，{} 帧 @{}×{} {}fps（Aurora + 打字机 + 片头/片尾 0.8s 淡入淡出，标准 MJPEG-in-AVI）",
        avi.len(),
        spec.frame_count(),
        spec.width,
        spec.height,
        spec.fps
    );
}
