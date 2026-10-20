//! 例子（v5）：**多场景 + 转场拼接**——把三支 2 秒子片按
//! `Transition::Wipe{Right}` 与 `Transition::Fade` 拼接为一支约 4.8 秒的
//! 短片，封装为标准 MJPEG-in-AVI，输出到 `prism_aurora_v5.avi`。
//!
//! 实现方式为**纯后处理**：复用已验证的 `render_rgba_frames` 逐支渲染，
//! 再在帧序列层做转场混合（交叉溶解 / 线性擦除）——不重构渲染核心、
//! 不引入任何新第三方 crate、不破坏既有 API（`render_short` 系列不变）。
//!
//! 三支子片各自携带：独立背景（Aurora / Scanline / DarkGlow）、静态小标签、
//! kinetic 主标题、打字机副标题、装饰线与收尾字幕条。每支子片的片头 /
//! 片尾淡入淡出由**轨道级** `in_sec` / `out_sec` 承担（拼接入口只接收
//! `ClipSpec` 列表，不提供 Clip 级淡变参数）。

use prism_author::{
    render_short_concat_avi, AccentLine, Background, Caption, CaptionStyle, ClipSpec, EasingKind,
    FontKind, KineticText, StaticText, Track, Transition, TypewriterText, WipeDir,
};

/// 构造第 `idx` 支子片（2.0 秒 1280×720 @24fps ⇒ 48 帧）。
///
/// 三支子片仅背景 / 文案不同；每支都含静态标签、kinetic 主标题、装饰线、
/// 打字机副标题与字幕条，且全部轨道带片头淡入（`in_sec`）与片尾淡出
/// （`out_sec`），使单支子片自身也有完整的起承转合。
fn sub_spec(
    bg: Background,
    tag: &'static str,
    title: &'static str,
    typed: &'static str,
    caption: &'static str,
) -> ClipSpec {
    ClipSpec {
        width: 1280,
        height: 720,
        fps: 24,
        duration: 2.0,
        background: bg,
        jpeg_quality: 88,
        tracks: vec![
            // 静态小标签（片头 0.5s 淡入、片尾 0.5s 淡出）。
            Track::Static(StaticText {
                content: tag.into(),
                font: FontKind::Regular,
                size: 22.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.2,
                in_sec: 0.5,
                letter_spacing: 7.0,
                y: 220.0,
                out_sec: 0.5,
                out_at: None,
                center_x: None,
            }),
            // 逐字 kinetic 主标题（片头 0.4s 淡入、片尾 0.6s 淡出）。
            Track::Kinetic(KineticText {
                content: title.into(),
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
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            // 装饰线（片头 0.4s 淡入、片尾 0.4s 淡出）。
            Track::Accent(AccentLine {
                center_x: 640.0,
                y: 398.0,
                w: 320.0,
                h: 3.0,
                color: [0.30, 0.75, 0.95, 0.9],
                at: 0.9,
                in_sec: 0.4,
                out_sec: 0.4,
                out_at: None,
            }),
            // 打字机副标题（逐字 + 闪烁光标；片头 0.25s 淡入、片尾 0.5s 淡出）。
            Track::Typewriter(TypewriterText {
                content: typed.into(),
                font: FontKind::Regular,
                size: 30.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 1.0,
                y: 470.0,
                cps: 16.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.25,
                out_sec: 0.5,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
            // 收尾字幕条（片头 0.5s 淡入、片尾 0.5s 淡出）。
            Track::Caption(Caption {
                content: caption.into(),
                at: 1.3,
                in_sec: 0.5,
                out_sec: 0.5,
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
    }
}

fn main() {
    // 三支子片：极光 → 扫描线 → 深空。
    let specs = vec![
        sub_spec(
            Background::Aurora,
            "SCENE 01 · 极光",
            "棱镜引擎",
            "程序化视频 · 零外部依赖",
            "多场景拼接 · 转场 v5",
        ),
        sub_spec(
            Background::Scanline,
            "SCENE 02 · 扫描线",
            "帧序列层",
            "交叉溶解 · 线性擦除",
            "纯后处理 · 渲染核心零改动",
        ),
        sub_spec(
            Background::DarkGlow,
            "SCENE 03 · 深空",
            "视频即代码",
            "三支子片 · 两次转场",
            "prism-author · short_v5",
        ),
    ];

    // 相邻第 i 对用 transitions[i]：01→02 用 Wipe{Right}，02→03 用 Fade。
    let transitions = vec![Transition::Wipe { dir: WipeDir::Right }, Transition::Fade];
    let trans_sec = 0.6;

    let avi = render_short_concat_avi(&specs, &transitions, trans_sec).expect("AVI 封装失败");
    std::fs::write("prism_aurora_v5.avi", &avi).expect("写入 AVI 失败");

    // 预期帧数：48 + 48 + 48 - round(0.6*24)=14 - 14 = 116（约 4.83s）。
    let mut total = specs[0].frame_count();
    for i in 1..specs.len() {
        let f = specs[i].frame_count();
        let tf = ((trans_sec * specs[i].fps as f32).round() as usize).clamp(1, total.min(f));
        total += f - tf;
    }

    eprintln!(
        "prism_author::render_short_concat_avi 完成：{} 字节，{} 帧 @{}×{} {}fps（3 支子片 × 2.0s + Wipe{{Right}}/Fade 各 {:.2}s 转场，约 {:.2}s，标准 MJPEG-in-AVI）",
        avi.len(),
        total,
        specs[0].width,
        specs[0].height,
        specs[0].fps,
        trans_sec,
        total as f32 / specs[0].fps as f32,
    );
}
