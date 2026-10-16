//! 例子：用新 API 渲染同款 5 秒文字片头，但封装为**标准 MJPEG-in-AVI**
//! （`render_short_avi`），输出到 `/workspace/prism_aurora_v3.avi`。
//!
//! 与 `short.rs`（MP4）同款 `ClipSpec`——证明作者 API 同时打通 MP4 与零依赖
//! 标准 AVI 两条封装链路；AVI 可被 VLC / 浏览器 / QuickTime 直接双击播放。

use prism_author::{
    render_short_avi, AccentLine, Background, Caption, CaptionStyle, ClipSpec, EasingKind, FontKind,
    KineticText, StaticText, Track,
};

fn main() {
    // 一份声明式规格，描述与 `short.rs` / `author_api_demo` 同款的 5 秒文字片头。
    let spec = ClipSpec {
        width: 1280,
        height: 720,
        fps: 24,
        duration: 5.0,
        background: Background::DarkGlow,
        jpeg_quality: 88,
        tracks: vec![
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
            Track::Kinetic(KineticText {
                content: "程序化视频 · 零外部依赖".into(),
                font: FontKind::Regular,
                size: 32.0,
                color: [0.82, 0.87, 0.96, 1.0],
                at: 1.9,
                stagger: 0.045,
                per_char: 0.40,
                scale_from: 0.80,
                rise_px: 10.0,
                easing: EasingKind::QuadOut,
                y: 462.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            Track::Caption(Caption {
                content: "视频即代码 · 单文件源码分发".into(),
                at: 3.2,
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

    // 5 秒短片 = 一次函数调用，产出标准 MJPEG-in-AVI。
    let avi = render_short_avi(&spec).expect("AVI 封装失败");
    std::fs::write("/workspace/prism_aurora_v3.avi", &avi).expect("写入 AVI 失败");
    eprintln!(
        "prism_author::render_short_avi 完成：{} 字节，{} 帧 @{}×{} {}fps（标准 MJPEG-in-AVI）",
        avi.len(),
        spec.frame_count(),
        spec.width,
        spec.height,
        spec.fps
    );
}
