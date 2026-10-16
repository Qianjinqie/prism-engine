//! `prism-author` —— Prism 引擎的高层"视频即代码"作者 API。
//!
//! 把"强大版"路线图里「视频即代码」的愿景落地为正式的内部 crate：作者只写
//! 一份声明式 [`ClipSpec`]（画幅 / 背景预设 / 若干轨道），再调用一次
//! [`render_short`] 即可产出一支 MP4。
//!
//! 所有"脏活"都被收进 [`render`] 模块内部：
//! - 背景绘制（极低频 simplex 微波动 + 辉光 + 暗角 + 呼吸）复用 `prism-noise`；
//! - 逐字 kinetic 入场 / 静态文字 / 字幕条复用 `prism-text`；
//! - 装饰线 / 圆角矩形 / 椭圆等图元复用 `prism-raster::fill_path`；
//! - 帧循环 + RGBA16F→RGBA8 交接 + 官方 muxer 封装全部隐藏。
//!
//! 调用方（[`examples::short`]）不碰 `Fbuf16F`、帧循环与编码器。
//!
//! # 新增 / 扩展能力（相对 PoC）
//! - 自研 [`easing::EasingKind`]：**不**依赖 `prism-anim`，避免与其他工程师
//!   改动耦合；
//! - [`Track::Image`]：RGBA 缓冲 / 纯色 / 渐变面板叠加层，支持淡入淡出与位移；
//! - 各轨道通用 [`crate::spec::KineticText::in_sec`] / `out_sec` 淡入淡出字段；
//! - [`Track::Shape`]：用 `prism-raster` 画圆角矩形 / 椭圆等图元。
//!
//! # 内容增强包（零新第三方 crate，不破坏既有 API）
//! - [`Track::Typewriter`]：打字机文字（逐字显示 + 可选闪烁光标）；
//! - [`Background::Aurora`] / [`Background::Scanline`]：两个新背景预设
//!   （均复用 `prism-noise::Fbm` 与既有常量，低频网格双线性上采样）；
//! - [`render_short_faded`] / [`render_short_avi_faded`]：Clip 级片头 / 片尾
//!   淡入淡出入口（纯后处理，[`render_short`] / [`render_short_avi`] 行为不变）。
//!
//! # 多场景 + 转场拼接（纯后处理，渲染核心零改动）
//! - [`Transition`] / [`WipeDir`]：场景间转场声明（交叉溶解 / 四向线性擦除）；
//! - [`render_short_concat`] / [`render_short_concat_avi`]：把多支子
//!   [`ClipSpec`] 从左到右折叠拼接为一支 MP4 / AVI。内部复用已验证的
//!   `render_rgba_frames`，只在帧序列层按转场混合重叠区，
//!   **不重构渲染核心、零新依赖、零破坏既有 API**。
//!
//! # 微积分短片 v2 原语（E1–E4，加法性扩展）
//! - [`Track::DynPath`]（E1）：时间参数化折线 `GeomFn = Arc<dyn Fn(f32) ->
//!   Vec<[f32; 2]> + Send + Sync>`，每帧求值后与 [`Track::Path`] 进入
//!   **完全相同的光栅核心**（`render::paint_polyline`）；
//! - [`Track::Counter`]（E2）：实时数值读数（ASCII 定点 HUD），复用
//!   `draw_static` 文本管线；`(from, to, dur_sec, easing)` 与对应几何
//!   派生自同一纯函数 ⇒「读数恒等」；
//! - [`Track::Static`] / [`Track::Kinetic`] / [`Track::Typewriter`] 与
//!   [`CounterText`] 的 `center_x: Option<f32>`（E3）：`None` = 画幅水平
//!   中心（与 v1 逐位一致），`Some(x)` = 文本布局盒水平中心；
//! - [`Track::Sequence`]（E4）：RGBA8 帧序列 + 源帧率，按场景时间取帧，
//!   与 [`ImageTrack`] 的 Buffer 分支共用 blit 核心（`render::blit_rgba8`）。

pub mod easing;
pub mod render;
pub mod spec;

pub use easing::EasingKind;
pub use prism_text::presets::CaptionStyle;
pub use render::{
    render_short, render_short_avi, render_short_avi_faded, render_short_concat,
    render_short_concat_avi, render_short_concat_faded_frames, render_short_concat_faded_frames_each,
    render_short_faded,
};
pub use spec::{
    AccentLine, Background, Caption, ClipSpec, CounterText, DynPathTrack, FontKind, GeomFn,
    ImageSource, ImageTrack, KineticText, PathTrack, SequenceTrack, ShapeKind, ShapeTrack,
    StaticText, Track, Transition, TypewriterText, WipeDir,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::easing::EasingKind;
    use crate::render::{
        apply_clip_fade, clip_fade_factor, concat_with_transition, counter_text, counter_value,
        fade_factor, out_start_of, render_rgba_frames,
    };
    use std::sync::Arc;

    /// 构造一个小型确定性 spec（用于快速单元级渲染测试）。
    fn tiny_spec() -> ClipSpec {
        ClipSpec {
            width: 64,
            height: 48,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![
                Track::Static(StaticText {
                    content: "PRISM".into(),
                    font: FontKind::Regular,
                    size: 14.0,
                    color: [0.46, 0.78, 0.92, 0.85],
                    at: 0.2,
                    in_sec: 0.3,
                    letter_spacing: 2.0,
                    y: 20.0,
                    out_sec: 0.0,
                    out_at: None,
                    center_x: None,
                }),
                Track::Kinetic(KineticText {
                    content: "棱镜".into(),
                    font: FontKind::Bold,
                    size: 28.0,
                    color: [0.98, 0.98, 1.0, 1.0],
                    at: 0.3,
                    stagger: 0.08,
                    per_char: 0.4,
                    scale_from: 0.7,
                    rise_px: 8.0,
                    easing: EasingKind::CubicOut,
                    y: 36.0,
                    in_sec: 0.0,
                    out_sec: 0.0,
                    out_at: None,
                    center_x: None,
                }),
                Track::Accent(AccentLine {
                    center_x: 32.0,
                    y: 42.0,
                    w: 24.0,
                    h: 2.0,
                    color: [0.30, 0.75, 0.95, 0.9],
                    at: 0.6,
                    in_sec: 0.3,
                    out_sec: 0.0,
                    out_at: None,
                }),
                Track::Shape(ShapeTrack {
                    kind: ShapeKind::RoundRect { radius: 4.0 },
                    x: 8.0,
                    y: 8.0,
                    w: 48.0,
                    h: 32.0,
                    color: [0.1, 0.2, 0.4, 0.25],
                    at: 0.0,
                    in_sec: 0.5,
                    out_sec: 0.0,
                    out_at: None,
                }),
                Track::Image(ImageTrack {
                    source: ImageSource::Gradient {
                        top: [0.2, 0.4, 0.8, 0.4],
                        bottom: [0.1, 0.1, 0.3, 0.0],
                    },
                    x: 2.0,
                    y: 2.0,
                    w: 12.0,
                    h: 12.0,
                    at: 0.1,
                    in_sec: 0.3,
                    out_sec: 0.0,
                    out_at: None,
                    move_from: [10.0, 0.0],
                }),
            ],
        }
    }

    /// 合法 spec 返回非空 `Vec<u8>`。
    #[test]
    fn render_short_returns_nonempty_for_valid_spec() {
        let spec = tiny_spec();
        let mp4 = render_short(&spec);
        assert!(!mp4.is_empty(), "合法 spec 必须产出非空 MP4");
        // 顶层应为 ftyp 盒（MP4 魔数）。
        assert_eq!(&mp4[4..8], b"ftyp", "产物应是合法 MP4（顶层 ftyp 盒）");
    }

    /// 边界：空 `tracks` 仍产出合法（至少 1 帧）MP4，不 panic。
    #[test]
    fn empty_tracks_still_produces_valid_mp4() {
        let spec = ClipSpec {
            width: 32,
            height: 32,
            fps: 1,
            duration: 0.5,
            background: Background::DarkGlow,
            jpeg_quality: 70,
            tracks: vec![],
        };
        assert_eq!(spec.frame_count(), 1, "0.5s @1fps 应得 1 帧");
        let mp4 = render_short(&spec);
        assert!(!mp4.is_empty(), "空 tracks 不应产出空 MP4");
        assert_eq!(&mp4[4..8], b"ftyp");
    }

    /// 边界：极小 duration（0.1s）也能产出，不 panic。
    #[test]
    fn tiny_duration_still_produces_valid_mp4() {
        let spec = ClipSpec {
            width: 48,
            height: 36,
            fps: 24,
            duration: 0.1,
            background: Background::Gradient {
                top: [0.1, 0.1, 0.2],
                bottom: [0.0, 0.0, 0.1],
            },
            jpeg_quality: 75,
            tracks: vec![Track::Static(StaticText {
                content: "X".into(),
                font: FontKind::Regular,
                size: 12.0,
                color: [1.0, 1.0, 1.0, 1.0],
                at: 0.0,
                in_sec: 0.05,
                letter_spacing: 0.0,
                y: 18.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            })],
        };
        let mp4 = render_short(&spec);
        assert!(!mp4.is_empty(), "0.1s 极小 duration 应产出 MP4");
    }

    /// "黄金"确定性测试：固定 seed 渲染同一 spec 两次，字节流逐字节一致。
    #[test]
    fn golden_render_is_byte_identical() {
        let spec = tiny_spec();
        let a = render_short(&spec);
        let b = render_short(&spec);
        assert_eq!(a.len(), b.len(), "两次渲染字节数应一致");
        assert_eq!(a, b, "固定 spec 渲染必须逐字节一致（确定性）");
    }

    /// 形状 / 图像 / 文字轨道确实会改变输出（健全性，防止空实现）。比全黑背景多墨。
    #[test]
    fn tracks_actually_paint_pixels() {
        // 无轨道（仅背景）的产物字节。
        let bare = ClipSpec {
            width: 64,
            height: 48,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![],
        };
        let with_tracks = tiny_spec();
        assert_ne!(
            render_short(&bare),
            render_short(&with_tracks),
            "叠加轨道必须改变最终画面"
        );
    }

    /// `render_short_avi` 产出非空且为标准 MJPEG-in-AVI（不破坏既有 MP4 测试）。
    #[test]
    fn render_short_avi_returns_valid_avi() {
        let spec = tiny_spec();
        let avi = render_short_avi(&spec).expect("AVI 封装失败");
        assert!(!avi.is_empty(), "render_short_avi 必须产出非空 AVI");

        // 顶层 RIFF/AVI 魔数。
        assert_eq!(&avi[0..4], b"RIFF", "AVI 必须以 RIFF 起头");
        assert_eq!(&avi[8..12], b"AVI ", "AVI 必须为 'AVI ' 类型");
        // RIFF size = 文件长 - 8。
        let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size, avi.len() - 8);

        // 关键块与 MJPG 流标记齐全。
        let has = |needle: &[u8]| avi.windows(needle.len()).position(|w| w == needle).is_some();
        assert!(has(b"hdrl"), "应含 hdrl");
        assert!(has(b"movi"), "应含 movi");
        assert!(has(b"idx1"), "应含 idx1");
        assert!(has(b"vids"), "视频流类型应为 vids");
        assert!(has(b"MJPG"), "流处理器/压缩应为 MJPG");
        assert!(has(b"00dc"), "帧块四CC 应为 00dc");

        // 既有 MP4 API 不受影响。
        let mp4 = render_short(&spec);
        assert_eq!(&mp4[4..8], b"ftyp", "既有 MP4 行为必须保留");
    }

    // ---- 内容增强包：Clip 级淡入淡出 ----

    /// `clip_fade_factor`：片头从 0 升、中段 1、片尾回 0；fade<=0 全程 1。
    #[test]
    fn clip_fade_factor_ramps_endpoints() {
        // fps 仅防御性校验用；t 为秒。
        assert!(
            clip_fade_factor(0.0, 24.0, 2.0, 0.5, 0.5) <= f32::EPSILON,
            "片头起点（t=0）应为全黑"
        );
        assert!(
            (clip_fade_factor(1.0, 24.0, 2.0, 0.5, 0.5) - 1.0).abs() < 1e-6,
            "中段应为 1.0"
        );
        assert!(
            clip_fade_factor(2.0 - 1.0 / 24.0, 24.0, 2.0, 0.5, 0.5) < 0.2,
            "片尾末帧应接近全黑"
        );
        // fade <= 0 → 对应方向不淡变（全程 1.0）。
        assert!((clip_fade_factor(0.0, 24.0, 2.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((clip_fade_factor(1.999, 24.0, 2.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        // 防御：fps <= 0 视为无淡变。
        assert!((clip_fade_factor(0.0, 0.0, 2.0, 0.5, 0.5) - 1.0).abs() < 1e-6);
    }

    /// `apply_clip_fade`：fade_in>0 时首帧全黑（显著暗于中段），中段逐字节
    /// 不变，alpha 通道恒 255。
    #[test]
    fn clip_fade_first_frame_black_mid_untouched_alpha_preserved() {
        let spec = ClipSpec {
            width: 64,
            height: 48,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Static(StaticText {
                content: "FADE".into(),
                font: FontKind::Regular,
                size: 16.0,
                color: [0.9, 0.95, 1.0, 1.0],
                at: 0.0,
                in_sec: 0.2,
                letter_spacing: 2.0,
                y: 24.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            })],
        };
        let mut frames = render_rgba_frames(&spec);
        let mid = frames.len() / 2;
        let mid_before = frames[mid].clone();

        apply_clip_fade(&mut frames, spec.fps as f32, spec.duration, 0.5, 0.5);

        // 首帧全黑（片头从黑淡入），且 alpha 仍为 255。
        assert!(
            frames[0]
                .chunks_exact(4)
                .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 255),
            "fade_in>0 时首帧必须全黑（RGB=0）且 alpha=255"
        );
        // 中段帧（t=0.5 恰为淡入终点）不受影响——逐字节一致。
        assert_eq!(frames[mid], mid_before, "中段稳态帧必须逐字节不变");
        // 片尾末帧被显著压暗（相对中段）。
        let mean_rgb = |f: &Vec<u8>| -> f32 {
            let s: u32 = f
                .chunks_exact(4)
                .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
                .sum();
            s as f32 / (f.len() / 4) as f32 / 3.0
        };
        assert!(
            mean_rgb(frames.last().unwrap()) < mean_rgb(&frames[mid]) * 0.5,
            "片尾淡出末帧平均亮度必须显著低于中段"
        );
        // 所有帧 alpha 通道保持 255。
        for f in &frames {
            assert!(f.chunks_exact(4).all(|p| p[3] == 255), "alpha 必须恒 255");
        }
    }

    // ---- out_at 绝对淡出锚点（分拍阶跃 staging）----

    /// `out_start_of` 解析语义：绝对锚点优先、钳 `at`、非有限回落缺省、
    /// 缺省 + `out_sec<=0` = 不淡出。
    #[test]
    fn out_start_of_resolves_anchor_semantics() {
        // 绝对锚点原样返回。
        assert_eq!(out_start_of(Some(2.5), 0.3, 1.0, 10.0), Some(2.5));
        // 早于 at 钳到 at。
        assert_eq!(out_start_of(Some(0.2), 0.3, 1.0, 10.0), Some(1.0));
        // NaN / Inf 按 None 处理（数据违约不 panic）⇒ 回落缺省锚定片尾。
        assert_eq!(out_start_of(Some(f32::NAN), 0.3, 1.0, 10.0), Some(9.7));
        assert_eq!(out_start_of(Some(f32::INFINITY), 0.3, 1.0, 10.0), Some(9.7));
        // NaN + 零淡出同样按 None ⇒ 不淡出。
        assert_eq!(out_start_of(Some(f32::NAN), 0.0, 1.0, 10.0), None);
        // 缺省锚定片尾（duration - out_sec）。
        assert_eq!(out_start_of(None, 0.3, 1.0, 10.0), Some(9.7));
        // 缺省 + 零/负淡出 = 不淡出。
        assert_eq!(out_start_of(None, 0.0, 1.0, 10.0), None);
        assert_eq!(out_start_of(None, -1.0, 1.0, 10.0), None);
    }

    /// `fade_factor` + `out_at`：绝对锚点淡出（窗口内平滑、窗外归 0）、
    /// 零淡出时长硬切、缺省保持片尾锚定（回归）、淡入独立生效。
    #[test]
    fn fade_factor_out_at_absolute_anchor() {
        let d = 10.0_f32;
        // 绝对锚点 5.0 + 淡出 1.0：5.0 前满值，[5,6] 衰减，6.0 起归 0。
        assert_eq!(fade_factor(4.9, 1.0, 0.0, 1.0, Some(5.0), d), 1.0);
        let mid = fade_factor(5.5, 1.0, 0.0, 1.0, Some(5.0), d);
        assert!(mid > 0.0 && mid < 1.0, "淡出窗口中段应部分透明: {mid}");
        assert_eq!(fade_factor(6.0, 1.0, 0.0, 1.0, Some(5.0), d), 0.0);
        assert_eq!(fade_factor(9.9, 1.0, 0.0, 1.0, Some(5.0), d), 0.0);
        // 硬切：out_sec=0 ⇒ out_at 前满值、当刻起归 0。
        assert_eq!(fade_factor(4.999, 1.0, 0.0, 0.0, Some(5.0), d), 1.0);
        assert_eq!(fade_factor(5.0, 1.0, 0.0, 0.0, Some(5.0), d), 0.0);
        // 缺省行为回归：out_sec>0 仍锚定片尾（[9,10] 衰减）。
        assert_eq!(fade_factor(8.9, 1.0, 0.0, 1.0, None, d), 1.0);
        assert!(fade_factor(9.5, 1.0, 0.0, 1.0, None, d) < 1.0);
        // 缺省 + out_sec=0：可见至片尾（原语义）。
        assert_eq!(fade_factor(9.999, 1.0, 0.0, 0.0, None, d), 1.0);
        // 淡入独立生效（at=2, in_sec=0.5 ⇒ t=2.25 为半值）。
        assert!((fade_factor(2.25, 2.0, 0.5, 1.0, Some(5.0), d) - 0.5).abs() < 1e-6);
    }

    /// 渲染级：`out_at` 令文字轨在片中途消失（分拍阶跃的地基）；
    /// 缺省 `None` 行为不变（回归护栏）。
    #[test]
    fn out_at_removes_static_text_mid_clip() {
        let mk = |out_at: Option<f32>, out_sec: f32| ClipSpec {
            width: 96,
            height: 64,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Static(StaticText {
                content: "BEAT".into(),
                font: FontKind::Regular,
                size: 20.0,
                color: [0.9, 0.95, 1.0, 1.0],
                at: 0.1,
                in_sec: 0.0,
                letter_spacing: 2.0,
                y: 32.0,
                out_sec,
                out_at,
                center_x: None,
            })],
        };
        // 绝对锚点 0.5 + 淡出 0.2 ⇒ [0.5,0.7] 衰减，0.7 起消失。
        let frames = render_rgba_frames(&mk(Some(0.5), 0.2));
        let bg = render_rgba_frames(&ClipSpec {
            tracks: vec![],
            ..mk(Some(0.5), 0.2)
        });
        assert_ne!(frames[4], bg[4], "out_at 之前文字必须在场");
        assert_ne!(frames[7], bg[7], "淡出窗口中段文字仍在场（部分透明）");
        assert_eq!(frames[9], bg[9], "out_at+out_sec 之后文字必须完全消失");
        // 回归：out_at=None + out_sec=0 ⇒ 末帧仍在场。
        let legacy = render_rgba_frames(&mk(None, 0.0));
        let bg2 = render_rgba_frames(&ClipSpec {
            tracks: vec![],
            ..mk(None, 0.0)
        });
        assert_ne!(legacy[11], bg2[11], "缺省锚点 + 零淡出必须可见至片尾");
        // 回归：out_at=None + out_sec=0.2 ⇒ 中段逐字节不受影响、片尾
        // 淡出进行中（片尾锚定淡出在设计上延伸到片边界，片内不要求全消）。
        let legacy2 = render_rgba_frames(&mk(None, 0.2));
        let no_fade = render_rgba_frames(&mk(None, 0.0));
        assert_eq!(legacy2[6], no_fade[6], "中段帧必须不受片尾淡出影响");
        assert_ne!(legacy2[11], no_fade[11], "片尾淡出必须已降低末帧透明度");
    }

    /// 渲染级：分拍阶跃 HUD——同位置两条 Counter 经 `out_at` 硬切交接，
    /// 任一帧恰有一条在场（无叠字），读数各自正确。
    #[test]
    fn out_at_stages_sequential_counter_readouts() {
        let ca = CounterText {
            prefix: String::from("n = "),
            from: 2.0,
            to: 2.0,
            at: 0.1,
            dur_sec: 0.1,
            easing: EasingKind::Linear,
            decimals: 0,
            font: FontKind::Bold,
            size: 24.0,
            color: [0.9, 0.95, 1.0, 1.0],
            center_x: Some(80.0),
            y: 32.0,
            letter_spacing: 1.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: Some(0.5),
            value_at: None,
        };
        let cb = CounterText {
            from: 32.0,
            to: 32.0,
            at: 0.5,
            out_at: None,
            ..ca.clone()
        };
        // 读数恒等（静态值 ⇒ 全程恒定）。
        assert_eq!(counter_text(&ca, 0.3), "n = 2");
        assert_eq!(counter_text(&cb, 0.6), "n = 32");
        let mk = |tracks: Vec<Track>| ClipSpec {
            width: 160,
            height: 64,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks,
        };
        let fa = render_rgba_frames(&mk(vec![Track::Counter(ca.clone())]));
        let fb = render_rgba_frames(&mk(vec![Track::Counter(cb.clone())]));
        let fx = render_rgba_frames(&mk(vec![Track::Counter(ca), Track::Counter(cb)]));
        let bg = render_rgba_frames(&mk(vec![]));
        // 交接前（t=0.417，帧 5）：只有 A 在场。
        assert_eq!(fx[5], fa[5], "交接前混合帧必须恰为 A 单渲染");
        assert_ne!(fx[5], bg[5], "A 必须在场");
        // 交接后（t=0.5，帧 6）：只有 B 在场（A 硬切消失）。
        assert_eq!(fx[6], fb[6], "交接后混合帧必须恰为 B 单渲染");
        assert_ne!(fx[6], bg[6], "B 必须在场");
        // 全程无叠字：每帧混合结果必等于 bg / A / B 之一。
        for (i, f) in fx.iter().enumerate() {
            assert!(
                *f == bg[i] || *f == fa[i] || *f == fb[i],
                "帧 {i} 出现叠字或意外绘制"
            );
        }
    }

    // ---- 内容增强包：打字机轨道 ----

    /// 打字机轨道：渲染非空 + 黄金确定；不同 `cps` 产出不同帧序列
    /// （健全性，证明逐字生效）。
    #[test]
    fn typewriter_track_deterministic_and_cps_matters() {
        let base = |cps: f32| ClipSpec {
            width: 96,
            height: 64,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Typewriter(TypewriterText {
                content: "TYPEWRITER".into(),
                font: FontKind::Regular,
                size: 16.0,
                color: [0.9, 0.95, 1.0, 1.0],
                at: 0.1,
                y: 32.0,
                cps,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.2,
                out_sec: 0.0,
                out_at: None,
                letter_spacing: 1.0,
                center_x: None,
            })],
        };

        // 黄金确定性：两次渲染逐字节一致。
        let a = render_rgba_frames(&base(12.0));
        let b = render_rgba_frames(&base(12.0));
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa, fb, "打字机渲染必须黄金确定");
        }

        // 文字确实画上了（与纯背景产物不同）。
        let bg_only = render_rgba_frames(&ClipSpec {
            tracks: vec![],
            ..base(12.0)
        });
        assert!(
            a.iter().zip(bg_only.iter()).any(|(x, y)| x != y),
            "打字机文字必须改变画面"
        );

        // 不同 cps → 不同帧序列（逐字进度确实生效）。
        let fast = render_rgba_frames(&base(60.0));
        assert_ne!(a, fast, "不同 cps 必须产出不同帧序列");
    }

    // ---- 内容增强包：新背景预设 ----

    /// 新背景预设（Aurora / Scanline）：渲染非空 + 黄金确定 + 与 DarkGlow
    /// 产物不同。
    #[test]
    fn new_backgrounds_deterministic_and_differ_from_dark_glow() {
        let base = |bg: Background| ClipSpec {
            width: 64,
            height: 48,
            fps: 12,
            duration: 1.0,
            background: bg,
            jpeg_quality: 80,
            tracks: vec![],
        };
        let glow = render_rgba_frames(&base(Background::DarkGlow));
        for bg in [Background::Aurora, Background::Scanline] {
            let a = render_rgba_frames(&base(bg.clone()));
            let b = render_rgba_frames(&base(bg.clone()));
            assert_eq!(a.len(), b.len());
            for (fa, fb) in a.iter().zip(b.iter()) {
                assert_eq!(fa, fb, "新背景渲染必须黄金确定");
            }
            assert!(
                a.iter().zip(glow.iter()).any(|(x, y)| x != y),
                "新背景必须与 DarkGlow 画面不同"
            );
            assert!(
                a.iter().any(|f| f.iter().any(|&p| p > 0)),
                "新背景不应为全黑"
            );
        }
    }

    // ---- 多场景拼接：转场（纯后处理）----

    /// 构造 `n` 帧全同常量色（RGBA8）序列（尺寸 `w×h`）。
    fn solid_frames(n: usize, w: u32, h: u32, rgba: [u8; 4]) -> Vec<Vec<u8>> {
        let mut frame = vec![0u8; (w as usize) * (h as usize) * 4];
        for px in frame.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        (0..n).map(|_| frame.clone()).collect()
    }

    /// `concat_with_transition`（Fade）：总帧数 == `a.len()+b.len()-trans_frames`，
    /// 且头部 / 尾部分别逐字节等于 a 的前缀 / b 的后缀。
    #[test]
    fn concat_fade_total_frames_and_structure() {
        let (w, h, fps) = (8u32, 8u32, 10u32);
        let a = solid_frames(6, w, h, [200, 0, 0, 255]);
        let b = solid_frames(6, w, h, [0, 0, 200, 255]);
        let out = concat_with_transition(&a, &b, w, h, fps, Transition::Fade, 0.3);

        // trans_frames = round(0.3*10) = 3 ⇒ 总帧数 = 6+6-3 = 9。
        assert_eq!(out.len(), 6 + 6 - 3, "总帧数必须为 a.len()+b.len()-trans_frames");
        // 头部 3 帧 == a 的前 3 帧（逐字节）。
        for i in 0..3 {
            assert_eq!(out[i], a[i], "拼接头部必须逐字节等于 a 的前缀");
        }
        // 尾部 3 帧 == b 的后 3 帧（逐字节）。
        for i in 0..3 {
            assert_eq!(out[6 + i], b[3 + i], "拼接尾部必须逐字节等于 b 的后缀");
        }
    }

    /// `concat_with_transition`（Fade）数值断言：两段已知常量色序列，
    /// 重叠区中点帧（k=0.5）RGB ≈ [100,0,100]，alpha 恒 255；
    /// 首 / 末重叠帧分别 ≈ [150,0,50] / [50,0,150]。
    #[test]
    fn concat_fade_midpoint_matches_lerp() {
        let (w, h, fps) = (4u32, 4u32, 10u32);
        let a = solid_frames(5, w, h, [200, 0, 0, 255]);
        let b = solid_frames(5, w, h, [0, 0, 200, 255]);
        let out = concat_with_transition(&a, &b, w, h, fps, Transition::Fade, 0.3);

        // trans_frames = 3 ⇒ 重叠区为 out[2..5]，k = (i+1)/4 = 0.25 / 0.5 / 0.75。
        assert_eq!(out.len(), 5 + 5 - 3);
        let check = |frame: &Vec<u8>, expect: [i32; 3], what: &str| {
            assert_eq!(frame.len(), (w * h * 4) as usize);
            for px in frame.chunks_exact(4) {
                assert!(
                    (px[0] as i32 - expect[0]).abs() <= 1
                        && (px[1] as i32 - expect[1]).abs() <= 1
                        && (px[2] as i32 - expect[2]).abs() <= 1,
                    "{what}: 期望 RGB≈{:?}，实测 [{},{},{}]",
                    expect,
                    px[0],
                    px[1],
                    px[2]
                );
                assert_eq!(px[3], 255, "{what}: alpha 必须恒 255");
            }
        };
        check(&out[2], [150, 0, 50], "首个重叠帧（k=0.25）");
        check(&out[3], [100, 0, 100], "中点重叠帧（k=0.5）");
        check(&out[4], [50, 0, 150], "末个重叠帧（k=0.75）");
    }

    /// `trans_sec <= 0` 直接拼接（无重叠）：输出 == a ++ b。
    #[test]
    fn concat_zero_trans_sec_is_plain_concat() {
        let (w, h, fps) = (4u32, 4u32, 10u32);
        let a = solid_frames(3, w, h, [200, 0, 0, 255]);
        let b = solid_frames(2, w, h, [0, 0, 200, 255]);
        for ts in [0.0f32, -1.0] {
            let out = concat_with_transition(&a, &b, w, h, fps, Transition::Fade, ts);
            assert_eq!(out.len(), a.len() + b.len(), "trans_sec<=0 必须直接拼接");
            assert_eq!(&out[..a.len()], &a[..], "前半必须逐字节等于 a");
            assert_eq!(&out[a.len()..], &b[..], "后半必须逐字节等于 b");
        }
    }

    /// 确定性：同参数两次调用逐字节一致（Fade 与 Wipe 均验证）。
    #[test]
    fn concat_is_deterministic() {
        let (w, h, fps) = (8u32, 6u32, 12u32);
        let a = solid_frames(4, w, h, [200, 0, 0, 255]);
        let b = solid_frames(4, w, h, [0, 0, 200, 255]);
        for tr in [
            Transition::Fade,
            Transition::Wipe { dir: WipeDir::Right },
            Transition::Wipe { dir: WipeDir::Left },
            Transition::Wipe { dir: WipeDir::Up },
            Transition::Wipe { dir: WipeDir::Down },
        ] {
            let x = concat_with_transition(&a, &b, w, h, fps, tr, 0.5);
            let y = concat_with_transition(&a, &b, w, h, fps, tr, 0.5);
            assert_eq!(x.len(), y.len());
            for (fx, fy) in x.iter().zip(y.iter()) {
                assert_eq!(fx, fy, "转场拼接必须黄金确定（{:?}）", tr);
            }
        }
    }

    /// Wipe 逐像素位置阈值：中点帧（k=0.5）远侧纯 a / 远侧纯 b，
    /// 过渡带内为中间值；四个方向各自符合"入场侧"语义。
    #[test]
    fn concat_wipe_splits_by_position() {
        let (w, h, fps) = (16u32, 16u32, 10u32);
        let a = solid_frames(4, w, h, [200, 0, 0, 255]);
        let b = solid_frames(4, w, h, [0, 0, 200, 255]);
        // trans_frames = round(0.3*10) = 3 ⇒ 中点帧 out[2]（k = 2/4 = 0.5）。
        let px_at = |f: &Vec<u8>, x: u32, y: u32| -> [u8; 4] {
            let i = ((y * w + x) * 4) as usize;
            [f[i], f[i + 1], f[i + 2], f[i + 3]]
        };

        // Right：b 自右侧入场 ⇒ 左远侧 a、右远侧 b。
        let out = concat_with_transition(
            &a,
            &b,
            w,
            h,
            fps,
            Transition::Wipe { dir: WipeDir::Right },
            0.3,
        );
        let mid = &out[2];
        assert_eq!(px_at(mid, 2, 8), [200, 0, 0, 255], "Wipe Right 左远侧应为 a");
        assert_eq!(px_at(mid, 13, 8), [0, 0, 200, 255], "Wipe Right 右远侧应为 b");
        let band = px_at(mid, 8, 8); // 过渡带内（edge=8, band=2）
        assert!(
            band[0] > 1 && band[0] < 199,
            "过渡带内应为中间值，实测 {:?}",
            band
        );

        // Left：b 自左侧入场 ⇒ 左远侧 b、右远侧 a。
        let out = concat_with_transition(
            &a,
            &b,
            w,
            h,
            fps,
            Transition::Wipe { dir: WipeDir::Left },
            0.3,
        );
        let mid = &out[2];
        assert_eq!(px_at(mid, 2, 8), [0, 0, 200, 255], "Wipe Left 左远侧应为 b");
        assert_eq!(px_at(mid, 13, 8), [200, 0, 0, 255], "Wipe Left 右远侧应为 a");

        // Up：b 自底部入场 ⇒ 上远侧 a、下远侧 b。
        let out = concat_with_transition(
            &a,
            &b,
            w,
            h,
            fps,
            Transition::Wipe { dir: WipeDir::Up },
            0.3,
        );
        let mid = &out[2];
        assert_eq!(px_at(mid, 8, 2), [200, 0, 0, 255], "Wipe Up 上远侧应为 a");
        assert_eq!(px_at(mid, 8, 13), [0, 0, 200, 255], "Wipe Up 下远侧应为 b");

        // Down：b 自顶部入场 ⇒ 上远侧 b、下远侧 a。
        let out = concat_with_transition(
            &a,
            &b,
            w,
            h,
            fps,
            Transition::Wipe { dir: WipeDir::Down },
            0.3,
        );
        let mid = &out[2];
        assert_eq!(px_at(mid, 8, 2), [0, 0, 200, 255], "Wipe Down 上远侧应为 b");
        assert_eq!(px_at(mid, 8, 13), [200, 0, 0, 255], "Wipe Down 下远侧应为 a");
    }

    /// 边界：空序列 / 尺寸不匹配 → 合理结果、不 panic、确定性。
    #[test]
    fn concat_edge_cases_no_panic() {
        let (w, h, fps) = (4u32, 4u32, 10u32);
        let a = solid_frames(3, w, h, [10, 20, 30, 255]);
        let b = solid_frames(3, w, h, [40, 50, 60, 255]);

        // a 为空 ⇒ 直接得到 b（无法重叠）。
        let out = concat_with_transition(&[], &b, w, h, fps, Transition::Fade, 0.3);
        assert_eq!(out, b, "a 为空时应直接得到 b");
        // b 为空 ⇒ 直接得到 a。
        let out = concat_with_transition(&a, &[], w, h, fps, Transition::Fade, 0.3);
        assert_eq!(out, a, "b 为空时应直接得到 a");
        // 两者皆空 ⇒ 空序列。
        let out = concat_with_transition(&[], &[], w, h, fps, Transition::Fade, 0.3);
        assert!(out.is_empty(), "两者皆空时应得空序列");

        // 尺寸不匹配（帧字节数与画幅不符）⇒ 不 panic；总帧数公式仍成立，
        // 重叠帧退化为 a 的帧；结果确定。
        let bad = vec![vec![0u8; 5]; 3];
        let out = concat_with_transition(&a, &bad, w, h, fps, Transition::Fade, 0.3);
        assert_eq!(out.len(), 3 + 3 - 3, "尺寸不符时总帧数公式仍成立");
        let out2 = concat_with_transition(&a, &bad, w, h, fps, Transition::Fade, 0.3);
        assert_eq!(out, out2, "尺寸不符时也必须确定");
    }

    // ---- 折线路径轨道（Track::Path）----

    /// 统计一帧相对纯背景帧的"墨量"（不同的像素数）。
    fn ink_vs_bg(frame: &[u8], bg: &[u8]) -> usize {
        frame
            .chunks_exact(4)
            .zip(bg.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count()
    }

    /// 构造带 Path 轨道的小型 spec（96×64 @12fps / 1.2s ⇒ 15 帧）。
    fn path_spec(track: Track) -> ClipSpec {
        ClipSpec {
            width: 96,
            height: 64,
            fps: 12,
            duration: 1.2,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![track],
        }
    }

    /// 同尺寸纯背景 spec（Path 测试的对照基准）。
    fn bare_path_spec() -> ClipSpec {
        ClipSpec { tracks: vec![], ..path_spec(Track::Path(PathTrack {
            points: vec![],
            color: [0.0; 4],
            width: 0.0,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        })) }
    }

    /// 描边 / 填充帧非空：Path 轨道确实改变画面（相对纯背景有墨）。
    #[test]
    fn path_track_stroke_and_fill_paint_pixels() {
        let bg_only = render_rgba_frames(&bare_path_spec());

        // 描边：水平折线两段。
        let stroke = render_rgba_frames(&path_spec(Track::Path(PathTrack {
            points: vec![[12.0, 32.0], [48.0, 32.0], [84.0, 40.0]],
            color: [1.0, 1.0, 1.0, 1.0],
            width: 3.0,
            fill: false,
            at: 0.1,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        })));
        assert!(
            stroke.iter().zip(bg_only.iter()).any(|(f, b)| ink_vs_bg(f, b) > 0),
            "描边路径必须改变画面"
        );

        // 填充：闭合三角形面积。
        let fill = render_rgba_frames(&path_spec(Track::Path(PathTrack {
            points: vec![[20.0, 44.0], [48.0, 16.0], [76.0, 44.0]],
            color: [0.3, 0.9, 0.8, 0.6],
            width: 0.0,
            fill: true,
            at: 0.1,
            in_sec: 0.2,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: true,
        })));
        assert!(
            fill.iter().zip(bg_only.iter()).any(|(f, b)| ink_vs_bg(f, b) > 0),
            "填充路径必须改变画面"
        );
    }

    /// 绘制进度动画：draw_sec 内墨量随帧递增（t < at 时无墨）。
    #[test]
    fn path_track_draw_progress_animates() {
        let bg_only = render_rgba_frames(&bare_path_spec());
        let frames = render_rgba_frames(&path_spec(Track::Path(PathTrack {
            points: vec![[12.0, 32.0], [48.0, 32.0], [84.0, 40.0]],
            color: [1.0, 1.0, 1.0, 1.0],
            width: 3.0,
            fill: false,
            at: 0.1,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 1.0, // at 之后 1.0s 画完
            closed: false,
        })));
        // 15 帧 @12fps：t = 0 / 0.5 / 1.0 ⇒ progress = 0 / 0.4 / 0.9。
        let ink = |i: usize| ink_vs_bg(&frames[i], &bg_only[i]);
        assert_eq!(ink(0), 0, "t < at（progress=0）必须无墨");
        assert!(ink(6) > 0, "进度中段必须有墨");
        assert!(
            ink(12) > ink(6),
            "进度推进墨量必须递增（{} -> {}）",
            ink(6),
            ink(12)
        );
    }

    /// 非法输入（点数 < 2 / 坐标非有限）跳过该轨道，不 panic 且画面同纯背景。
    #[test]
    fn path_track_rejects_degenerate_points() {
        let bg_only = render_rgba_frames(&bare_path_spec());
        for points in [
            vec![],
            vec![[10.0, 10.0]],
            vec![[f32::NAN, 1.0], [2.0, 2.0]],
            vec![[0.0, 0.0], [f32::INFINITY, 1.0]],
        ] {
            let frames = render_rgba_frames(&path_spec(Track::Path(PathTrack {
                points,
                color: [1.0, 1.0, 1.0, 1.0],
                width: 3.0,
                fill: false,
                at: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                draw_sec: 0.0,
                closed: false,
            })));
            for (f, b) in frames.iter().zip(bg_only.iter()) {
                assert_eq!(ink_vs_bg(f, b), 0, "非法 Path 输入必须被跳过（不 panic）");
            }
        }
    }

    /// 黄金确定性：同 spec（描边 + 进度动画 + closed）渲染两次逐字节一致。
    #[test]
    fn path_track_is_deterministic() {
        let spec = path_spec(Track::Path(PathTrack {
            points: vec![[12.0, 32.0], [48.0, 32.0], [84.0, 40.0]],
            color: [0.9, 0.95, 1.0, 0.95],
            width: 3.5,
            fill: false,
            at: 0.1,
            in_sec: 0.2,
            out_sec: 0.2,
            out_at: None,
            draw_sec: 0.8,
            closed: true,
        }));
        let a = render_rgba_frames(&spec);
        let b = render_rgba_frames(&spec);
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa, fb, "Path 轨道渲染必须黄金确定");
        }
    }

    /// `render_short_concat_faded_frames`：帧数公式（Σ − 转场重叠）、片头全黑、
    /// 片尾压暗、与既有拼接入口行为兼容（不破坏）。
    #[test]
    fn concat_faded_frames_entry_works() {
        // 两支 1.0s @10fps 子片（各 10 帧）⇒ 转场 round(0.5×10)=5 ⇒ 15 帧。
        let sub = |rgb: [f32; 3]| ClipSpec {
            width: 32,
            height: 32,
            fps: 10,
            duration: 1.0,
            background: Background::Gradient {
                top: rgb,
                bottom: [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5],
            },
            jpeg_quality: 80,
            tracks: vec![Track::Static(StaticText {
                content: "F".into(),
                font: FontKind::Regular,
                size: 12.0,
                color: [1.0, 1.0, 1.0, 1.0],
                at: 0.0,
                in_sec: 0.1,
                letter_spacing: 1.0,
                y: 16.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            })],
        };
        let specs = vec![sub([0.8, 0.2, 0.2]), sub([0.2, 0.2, 0.8])];
        let frames = render_short_concat_faded_frames(&specs, 0.3, 0.3);
        assert_eq!(frames.len(), 10 + 10 - 5, "帧数 = Σ − 转场重叠");
        for f in &frames {
            assert_eq!(f.len(), 32 * 32 * 4, "每帧 RGBA8 尺寸正确");
        }
        // 片头淡入：首帧全黑（fade_in=0.3 ⇒ t=0 因子 0）。
        assert!(
            frames[0].chunks_exact(4).all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 255),
            "片头淡入首帧必须全黑且 alpha=255"
        );
        // 片尾淡出：末帧显著暗于中段。
        let mean_rgb = |f: &Vec<u8>| -> f32 {
            let s: u32 = f
                .chunks_exact(4)
                .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
                .sum();
            s as f32 / (f.len() / 4) as f32 / 3.0
        };
        assert!(
            mean_rgb(frames.last().unwrap()) < mean_rgb(&frames[7]) * 0.5,
            "片尾淡出末帧必须显著暗于中段"
        );
        // 确定性：两次调用逐字节一致。
        let again = render_short_concat_faded_frames(&specs, 0.3, 0.3);
        for (x, y) in frames.iter().zip(again.iter()) {
            assert_eq!(x, y, "concat_faded_frames 必须黄金确定");
        }
    }

    /// `render_short_concat_faded_frames`：参数非法（specs < 2）panic，
    /// 风格与 `render_short_concat` 一致。
    #[test]
    #[should_panic]
    fn concat_faded_frames_rejects_single_spec() {
        let spec = path_spec(Track::Path(PathTrack {
            points: vec![[1.0, 1.0], [2.0, 2.0]],
            color: [1.0; 4],
            width: 1.0,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
        let _ = render_short_concat_faded_frames(&[spec], 0.0, 0.0);
    }

    /// `render_short_concat_faded_frames_each`（流式回调入口）：收集序列
    /// 必须与批式 `render_short_concat_faded_frames` **逐字节一致**——
    /// 流式化只改内存画像，不改任何一帧；帧数公式 Σ − 转场重叠同样成立。
    #[test]
    fn concat_faded_frames_each_matches_batch_entry() {
        // 三支 1.0s @10fps 子片（各 10 帧）：t = round(0.5×10) = 5，
        // clamp 上界 min(acc_len, 10) = 10 ≥ 5 不触发 ⇒ 总帧 30 − 2×5 = 20。
        let sub = |rgb: [f32; 3]| ClipSpec {
            width: 32,
            height: 32,
            fps: 10,
            duration: 1.0,
            background: Background::Gradient {
                top: rgb,
                bottom: [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5],
            },
            jpeg_quality: 80,
            tracks: vec![Track::Static(StaticText {
                content: "F".into(),
                font: FontKind::Regular,
                size: 12.0,
                color: [1.0, 1.0, 1.0, 1.0],
                at: 0.0,
                in_sec: 0.1,
                letter_spacing: 1.0,
                y: 16.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            })],
        };
        let specs = vec![sub([0.8, 0.2, 0.2]), sub([0.2, 0.8, 0.2]), sub([0.2, 0.2, 0.8])];
        let batch = render_short_concat_faded_frames(&specs, 1.0, 1.5);
        let mut each: Vec<Vec<u8>> = Vec::new();
        render_short_concat_faded_frames_each(&specs, 1.0, 1.5, |f| each.push(f.to_vec()))
            .expect("参数合法必须 Ok");
        assert_eq!(each.len(), 30 - 2 * 5, "帧数 = Σ − 2×转场重叠");
        for f in &each {
            assert_eq!(f.len(), 32 * 32 * 4, "每帧 RGBA8 尺寸正确");
        }
        assert_eq!(each, batch, "流式入口必须与批式入口逐字节一致");
        // 片头淡入首帧全黑 / 片尾末帧压暗（淡变在流式路径同样生效）。
        assert!(
            each[0].chunks_exact(4).all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 255),
            "流式路径片头首帧必须全黑且 alpha=255"
        );
        let mean_rgb = |f: &Vec<u8>| -> f32 {
            let s: u32 = f
                .chunks_exact(4)
                .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
                .sum();
            s as f32 / (f.len() / 4) as f32 / 3.0
        };
        assert!(
            mean_rgb(each.last().unwrap()) < mean_rgb(&each[10]) * 0.5,
            "流式路径片尾末帧必须显著暗于中段"
        );
    }

    /// 流式入口的**段长约束**：某段帧数 < 2×T（T = max 转场帧数）时，
    /// 折叠语义的混合源会回溯更早段落，流式无法复现 ⇒ 必须 `Err`
    /// （明确失败而非静默错帧；短段拼接请改用批式入口）。
    #[test]
    fn concat_faded_frames_each_rejects_too_short_segments() {
        let mk = |dur: f32, rgb: [f32; 3]| ClipSpec {
            width: 32,
            height: 32,
            fps: 10,
            duration: dur,
            background: Background::Gradient {
                top: rgb,
                bottom: [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5],
            },
            jpeg_quality: 80,
            tracks: vec![],
        };
        // 10 帧 / 3 帧 / 10 帧：T = max(3, 5) = 5 ⇒ 每段须 ≥ 10 帧；
        // 中间段 3 帧违反约束。批式入口对此仍按 clamp 语义正常产出。
        let specs = vec![
            mk(1.0, [0.8, 0.2, 0.2]),
            mk(0.3, [0.2, 0.8, 0.2]),
            mk(1.0, [0.2, 0.2, 0.8]),
        ];
        let batch = render_short_concat_faded_frames(&specs, 0.0, 0.0);
        assert_eq!(batch.len(), 10 + 3 + 10 - 3 - 5, "批式入口短段 clamp 帧数公式");
        let err = render_short_concat_faded_frames_each(&specs, 0.0, 0.0, |_| {})
            .expect_err("短段必须被流式入口拒绝");
        assert!(
            err.contains("流式拼接无法复现") && err.contains("第 1 支"),
            "错误信息应指明违约子片与原因，实测：{err}"
        );
        // 边界：每段恰 2T（10 = 2×5）时合法，且与批式逐字节一致。
        let specs_ok = vec![mk(1.0, [0.8, 0.2, 0.2]), mk(1.0, [0.2, 0.8, 0.2])];
        let mut each: Vec<Vec<u8>> = Vec::new();
        render_short_concat_faded_frames_each(&specs_ok, 0.5, 0.5, |f| each.push(f.to_vec()))
            .expect("2T 边界段长必须合法");
        let batch_ok = render_short_concat_faded_frames(&specs_ok, 0.5, 0.5);
        assert_eq!(each, batch_ok, "2T 边界必须位级一致");
    }

    /// 流式入口参数校验：`specs < 2` / 画幅不一致 / 帧率不一致 → `Err`
    /// （不 panic；批式入口保持 panic 风格不变）。
    #[test]
    fn concat_faded_frames_each_rejects_bad_params() {
        let ok_spec = ClipSpec {
            width: 32,
            height: 32,
            fps: 10,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![],
        };
        // specs < 2。
        assert!(
            render_short_concat_faded_frames_each(&[ok_spec.clone()], 0.0, 0.0, |_| {}).is_err(),
            "少于 2 支必须 Err"
        );
        // 画幅不一致。
        let bad_dims = ClipSpec { width: 64, ..ok_spec.clone() };
        assert!(
            render_short_concat_faded_frames_each(&[ok_spec.clone(), bad_dims], 0.0, 0.0, |_| {})
                .is_err(),
            "画幅不一致必须 Err"
        );
        // 帧率不一致。
        let bad_fps = ClipSpec { fps: 24, ..ok_spec.clone() };
        assert!(
            render_short_concat_faded_frames_each(&[ok_spec, bad_fps], 0.0, 0.0, |_| {}).is_err(),
            "帧率不一致必须 Err"
        );
    }

    // ---- 微积分短片 v2 原语（E1 DynPath / E2 Counter / E3 center_x / E4 Sequence）----

    /// E1：同几何的 DynPath 与静态 PathTrack 必须走逐位相同的光栅核心
    /// （架构 §4.3：`paint_polyline` 共用，`dynpath_shares_path_core`）。
    #[test]
    fn dynpath_shares_path_core() {
        let pts = vec![[12.0, 32.0], [48.0, 32.0], [84.0, 40.0]];
        let color = [0.9, 0.95, 1.0, 0.95];
        let common = |points: Vec<[f32; 2]>| PathTrack {
            points,
            color,
            width: 3.5,
            fill: false,
            at: 0.1,
            in_sec: 0.2,
            out_sec: 0.2,
            out_at: None,
            draw_sec: 0.8,
            closed: true,
        };
        let static_frames = render_rgba_frames(&path_spec(Track::Path(common(pts.clone()))));
        let dyn_frames = render_rgba_frames(&path_spec(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |_t| pts.clone()),
            color,
            width: 3.5,
            fill: false,
            at: 0.1,
            in_sec: 0.2,
            out_sec: 0.2,
            out_at: None,
            draw_sec: 0.8,
            closed: true,
        })));
        assert_eq!(static_frames.len(), dyn_frames.len());
        for (fa, fb) in static_frames.iter().zip(dyn_frames.iter()) {
            assert_eq!(fa, fb, "DynPath 与静态 PathTrack 必须逐字节一致（共用光栅核心）");
        }
    }

    /// E1 违约跳过：点数 < 2 / 坐标非有限 / `env <= 0` 三种输入均跳过，
    /// 不 panic 且画面同纯背景。
    #[test]
    fn dynpath_rejects_degenerate_geom() {
        let bg_only = render_rgba_frames(&bare_path_spec());
        let mk = |geom: GeomFn, at: f32| DynPathTrack {
            geom,
            color: [1.0, 1.0, 1.0, 1.0],
            width: 3.0,
            fill: false,
            at,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        };
        // 点数 < 2（0 点 / 1 点）与坐标非有限（NaN / Inf）。
        let bad_geoms: [GeomFn; 4] = [
            Arc::new(|_t| vec![]),
            Arc::new(|_t| vec![[10.0, 10.0]]),
            Arc::new(|_t| vec![[f32::NAN, 1.0], [2.0, 2.0]]),
            Arc::new(|_t| vec![[0.0, 0.0], [f32::INFINITY, 1.0]]),
        ];
        for geom in bad_geoms {
            let frames = render_rgba_frames(&path_spec(Track::DynPath(mk(geom, 0.0))));
            for (f, b) in frames.iter().zip(bg_only.iter()) {
                assert_eq!(ink_vs_bg(f, b), 0, "非法 DynPath 几何必须被跳过（不 panic）");
            }
        }
        // env <= 0：at 超出 duration（env 恒 0，dispatch 层跳过）。
        let frames = render_rgba_frames(&path_spec(Track::DynPath(mk(
            Arc::new(|_t| vec![[1.0, 1.0], [2.0, 2.0]]),
            5.0, // duration 仅 1.2s
        ))));
        for (f, b) in frames.iter().zip(bg_only.iter()) {
            assert_eq!(ink_vs_bg(f, b), 0, "env<=0 的 DynPath 必须被跳过");
        }
    }

    /// E1：黄金确定 + 时间参数化几何确实逐帧改变画面。
    #[test]
    fn dynpath_is_deterministic_and_time_parametrized() {
        let spec = path_spec(Track::DynPath(DynPathTrack {
            geom: Arc::new(|t| {
                // 纯函数：折线高度随场景时间 t 平移。
                let y = 32.0 + 8.0 * t;
                vec![[12.0, y], [48.0, y], [84.0, y + 4.0]]
            }),
            color: [0.9, 0.95, 1.0, 0.95],
            width: 3.5,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
        let a = render_rgba_frames(&spec);
        let b = render_rgba_frames(&spec);
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa, fb, "DynPath 渲染必须黄金确定");
        }
        let bg_only = render_rgba_frames(&bare_path_spec());
        assert!(
            a.iter().zip(bg_only.iter()).any(|(f, b)| ink_vs_bg(f, b) > 0),
            "DynPath 必须改变画面"
        );
        // 时间参数化：不同帧的墨量应不同（折线逐帧平移）。
        assert!(
            a.iter().zip(bg_only.iter()).any(|(f, b)| ink_vs_bg(f, b) > 0)
                && a[2] != a[10],
            "DynPath 几何必须随时间变化"
        );
    }

    /// E2 读数恒等（架构 §7.3）：Counter 的 `(from, to, dur_sec, easing)`
    /// 与 DynPath 几何派生自**同一纯函数**——值级（位模式相等）+ 渲染级
    /// （Counter 帧与预格式化 Static 帧逐字节一致）。
    #[test]
    fn counter_identity_with_geometry_same_pure_fn() {
        // 共享纯函数：割线斜率 h(u) = 2 + 3u（from/to 与几何 y 同源）。
        fn secant_h(u: f32) -> f32 {
            2.0 + 3.0 * u
        }
        let (from, to) = (secant_h(0.0), secant_h(1.0));
        let (at, dur_sec) = (0.5f32, 1.0f32);
        let easing = EasingKind::CubicInOut;
        let mk_counter = || CounterText {
            prefix: "m = ".into(),
            from,
            to,
            at,
            dur_sec,
            easing,
            decimals: 4,
            font: FontKind::Regular,
            size: 16.0,
            color: [1.0, 1.0, 1.0, 1.0],
            center_x: None,
            y: 32.0,
            letter_spacing: 1.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            value_at: None,
        };
        // 值级恒等：采样点上 counter_value 与几何派生值位模式相等。
        for &t in &[0.5f32, 0.75, 1.0, 1.25, 1.5] {
            let u = ((t - at) / dur_sec).clamp(0.0, 1.0);
            let expect = secant_h(easing.apply(u));
            assert_eq!(
                counter_value(&mk_counter(), t).to_bits(),
                expect.to_bits(),
                "t={t}: 读数必须与几何派生自同一纯函数"
            );
            // DynPath 几何同参：点的 y 坐标即同一纯函数值。
            let geom = |t: f32| {
                let u = ((t - at) / dur_sec).clamp(0.0, 1.0);
                let y = secant_h(easing.apply(u));
                vec![[10.0, y], [40.0, y]]
            };
            assert_eq!(
                geom(t)[0][1].to_bits(),
                expect.to_bits(),
                "t={t}: 几何与读数必须同源"
            );
        }
        // 渲染级恒等：t=1.0（u=0.5 ⇒ m=3.5）时 Counter 帧与预格式化
        // Static 帧逐字节一致（path_spec：96×64 @12fps ⇒ 第 12 帧 t=1.0）。
        let counter_frames = render_rgba_frames(&path_spec(Track::Counter(mk_counter())));
        let static_frames = render_rgba_frames(&path_spec(Track::Static(StaticText {
            content: "m = 3.5000".into(),
            font: FontKind::Regular,
            size: 16.0,
            color: [1.0, 1.0, 1.0, 1.0],
            at: 0.0,
            in_sec: 0.0,
            letter_spacing: 1.0,
            y: 32.0,
            out_sec: 0.0,
            out_at: None,
            center_x: None,
        })));
        assert_eq!(counter_frames.len(), static_frames.len());
        assert_eq!(
            counter_frames[12], static_frames[12],
            "t=1.0 时 Counter 读数帧必须与预格式化 Static 帧逐字节一致"
        );
    }

    /// E2：`-0.0` 归一（不出现 `"-0.0000"`）+ 定点格式无科学计数。
    #[test]
    fn counter_neg_zero_normalized_and_no_scientific_notation() {
        let base = CounterText {
            prefix: String::new(),
            from: -0.0,
            to: -1.0,
            at: 0.5,
            dur_sec: 1.0,
            easing: EasingKind::CubicInOut,
            decimals: 4,
            font: FontKind::Regular,
            size: 16.0,
            color: [1.0; 4],
            center_x: None,
            y: 32.0,
            letter_spacing: 1.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            value_at: None,
        };
        // t=at（u=0 ⇒ e=0）：v = -0.0 + (-1.0)·0.0 = -0.0，须归一为 +0.0。
        assert_eq!(
            counter_value(&base, 0.5).to_bits(),
            0.0f32.to_bits(),
            "-0.0 必须归一为 +0.0"
        );
        assert_eq!(counter_text(&base, 0.5), "0.0000", "不得出现 \"-0.0000\"");
        // 定点格式：大值（12345678）也不得出现科学计数法。
        let big = CounterText {
            from: 0.0,
            to: 12_345_678.0,
            ..base.clone()
        };
        let text = counter_text(&big, 1.5); // u=1 ⇒ v=12345678
        assert_eq!(text, "12345678.0000", "大值必须定点格式");
        assert!(
            !text.contains('e') && !text.contains('E'),
            "读数不得使用科学计数法：{text}"
        );
    }

    /// E2：`decimals > 6` 钳 6（f32 定点有效位上限）。
    #[test]
    fn counter_decimals_clamped_at_six() {
        let c = CounterText {
            prefix: "n = ".into(),
            from: 0.0,
            to: 1.0,
            at: 0.0,
            dur_sec: 1.0,
            easing: EasingKind::Linear,
            decimals: 9,
            font: FontKind::Regular,
            size: 16.0,
            color: [1.0; 4],
            center_x: None,
            y: 32.0,
            letter_spacing: 1.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            value_at: None,
        };
        let text = counter_text(&c, 1.0); // u=1 ⇒ v=1.0
        assert_eq!(text, "n = 1.000000", "decimals>6 必须钳为 6");
        let frac = text.rsplit('.').next().unwrap();
        assert_eq!(frac.len(), 6, "钳位后小数位数必须为 6");
    }

    /// E2：黄金确定 + 读数逐帧滚动（不同帧文本不同）。
    #[test]
    fn counter_is_deterministic_and_scrolls() {
        let spec = path_spec(Track::Counter(CounterText {
            prefix: "n = ".into(),
            from: 0.0,
            to: 8.0,
            at: 0.0,
            dur_sec: 1.0,
            easing: EasingKind::Linear,
            decimals: 2,
            font: FontKind::Regular,
            size: 16.0,
            color: [1.0, 1.0, 1.0, 1.0],
            center_x: None,
            y: 32.0,
            letter_spacing: 1.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            value_at: None,
        }));
        let a = render_rgba_frames(&spec);
        let b = render_rgba_frames(&spec);
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa, fb, "Counter 渲染必须黄金确定");
        }
        // 读数滚动：t=0.5（n=4.00）与 t=1.0（n=8.00）帧不同。
        assert_ne!(a[6], a[12], "滚动读数必须随帧变化");
        let bg_only = render_rgba_frames(&bare_path_spec());
        assert!(
            a.iter().zip(bg_only.iter()).any(|(f, b)| ink_vs_bg(f, b) > 0),
            "Counter 必须改变画面"
        );
    }

    /// E4 取帧：src_fps=15 @ 画布 30fps，155 源帧每帧恰显 2 画布帧
    /// （`floor(fi/2)` 在源帧区间恒定；310 画布帧逐字节等于源帧）。
    #[test]
    fn sequence_two_canvas_frames_per_source_frame() {
        const SRC_N: usize = 155;
        const SRC_FPS: u32 = 15;
        let (w, h) = (4u32, 4u32);
        let frame_len = (w * h * 4) as usize;
        // 每源帧唯一纯色（R=k, G=255-k, B=0, A=255；k<=154 适配 u8）。
        let solid = |k: usize| {
            let mut f = vec![0u8; frame_len];
            for px in f.chunks_exact_mut(4) {
                px.copy_from_slice(&[k as u8, (255 - k) as u8, 0, 255]);
            }
            f
        };
        let frames: Vec<Vec<u8>> = (0..SRC_N).map(solid).collect();
        let spec = ClipSpec {
            width: w,
            height: h,
            fps: 30,
            // 310/30 - 0.002 ⇒ ceil(30·duration)=310 画布帧（S7 场景时长同式）。
            duration: 310.0 / 30.0 - 0.002,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Sequence(SequenceTrack {
                frames,
                width: w,
                height: h,
                src_fps: SRC_FPS,
                at: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
            })],
        };
        let out = render_rgba_frames(&spec);
        assert_eq!(out.len(), 310, "155 源帧 @15fps 必须铺满 310 画布帧 @30fps");
        // 画布帧 i 的源帧 = floor(i/2)：每源帧恰显 2 个画布帧，逐字节相等
        // （源帧不透明且全幅 ⇒ blit 后量化逐位复原）。
        for i in 0..310 {
            assert_eq!(out[i], solid(i / 2), "画布帧 {i} 必须等于源帧 {}", i / 2);
        }
    }

    /// E4 违约跳过：`frames` 空 / 帧字节数 ≠ w·h·4 / 尺寸 ≠ 画幅 /
    /// src_fps ≤ 0，均不 panic 且画面同纯背景。
    #[test]
    fn sequence_rejects_degenerate_input() {
        let (w, h) = (4u32, 4u32);
        let frame_len = (w * h * 4) as usize;
        let good = || {
            let mut f = vec![0u8; frame_len];
            for px in f.chunks_exact_mut(4) {
                px.copy_from_slice(&[200, 100, 50, 255]);
            }
            f
        };
        let bare = ClipSpec {
            width: w,
            height: h,
            fps: 30,
            duration: 0.2,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![],
        };
        let bg_only = render_rgba_frames(&bare);
        let mk = |frames: Vec<Vec<u8>>, tw: u32, th: u32, src_fps: u32| ClipSpec {
            tracks: vec![Track::Sequence(SequenceTrack {
                frames,
                width: tw,
                height: th,
                src_fps,
                at: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
            })],
            ..bare.clone()
        };
        let cases: [(Vec<Vec<u8>>, u32, u32, u32); 4] = [
            (vec![], w, h, 15),                       // frames 空
            (vec![vec![0u8; frame_len - 1]], w, h, 15), // 帧字节数不符
            (vec![good()], w + 2, h, 15),             // 源宽 ≠ 画幅
            (vec![good()], w, h, 0),                  // src_fps ≤ 0
        ];
        for (frames, tw, th, src_fps) in cases {
            let out = render_rgba_frames(&mk(frames, tw, th, src_fps));
            assert_eq!(out.len(), bg_only.len());
            for (f, b) in out.iter().zip(bg_only.iter()) {
                assert_eq!(f, b, "非法 Sequence 输入必须被跳过（不 panic）");
            }
        }
    }

    /// E4：黄金确定（合法序列两次渲染逐字节一致）。
    #[test]
    fn sequence_is_deterministic() {
        let (w, h) = (4u32, 4u32);
        let frame_len = (w * h * 4) as usize;
        let frames: Vec<Vec<u8>> = (0..4)
            .map(|k| {
                let mut f = vec![0u8; frame_len];
                for px in f.chunks_exact_mut(4) {
                    px.copy_from_slice(&[k as u8 * 10, 128, 255, 255]);
                }
                f
            })
            .collect();
        let spec = ClipSpec {
            width: w,
            height: h,
            fps: 30,
            duration: 0.2,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Sequence(SequenceTrack {
                frames,
                width: w,
                height: h,
                src_fps: 15,
                at: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
            })],
        };
        let a = render_rgba_frames(&spec);
        let b = render_rgba_frames(&spec);
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa, fb, "Sequence 渲染必须黄金确定");
        }
    }

    /// E3：`center_x: None` 路径与 `Some(w*0.5)` 逐位一致（None 即 v1
    /// 旧代码路径——黄金不回退的机械钉住）。
    #[test]
    fn text_center_x_none_equals_canvas_center() {
        let mk = |center_x: Option<f32>| ClipSpec {
            width: 96,
            height: 64,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![
                Track::Static(StaticText {
                    content: "CENTER".into(),
                    font: FontKind::Regular,
                    size: 16.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                    at: 0.0,
                    in_sec: 0.0,
                    letter_spacing: 2.0,
                    y: 32.0,
                    out_sec: 0.0,
                    out_at: None,
                    center_x,
                }),
                Track::Kinetic(KineticText {
                    content: "棱镜".into(),
                    font: FontKind::Bold,
                    size: 20.0,
                    color: [0.9, 0.95, 1.0, 1.0],
                    at: 0.0,
                    stagger: 0.05,
                    per_char: 0.3,
                    scale_from: 0.8,
                    rise_px: 6.0,
                    easing: EasingKind::CubicOut,
                    y: 52.0,
                    in_sec: 0.0,
                    out_sec: 0.0,
                    out_at: None,
                    center_x,
                }),
                Track::Typewriter(TypewriterText {
                    content: "TYPE".into(),
                    font: FontKind::Regular,
                    size: 14.0,
                    color: [0.8, 0.9, 1.0, 1.0],
                    at: 0.0,
                    y: 20.0,
                    cps: 12.0,
                    cursor: true,
                    cursor_blink_hz: 1.0,
                    in_sec: 0.0,
                    out_sec: 0.0,
                    out_at: None,
                    letter_spacing: 1.0,
                    center_x,
                }),
            ],
        };
        let none = render_rgba_frames(&mk(None));
        let center = render_rgba_frames(&mk(Some(96.0 * 0.5)));
        assert_eq!(none.len(), center.len());
        for (fa, fb) in none.iter().zip(center.iter()) {
            assert_eq!(fa, fb, "center_x: None 必须与 Some(w*0.5) 逐位一致（v1 路径）");
        }
    }

    /// E3：`Some(x)` 落位正确——文字墨迹质心随锚点水平平移。
    #[test]
    fn text_center_x_some_shifts_ink_horizontally() {
        let mk = |center_x: Option<f32>| ClipSpec {
            width: 96,
            height: 64,
            fps: 12,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![Track::Static(StaticText {
                content: "SHIFT".into(),
                font: FontKind::Regular,
                size: 16.0,
                color: [1.0, 1.0, 1.0, 1.0],
                at: 0.0,
                in_sec: 0.0,
                letter_spacing: 2.0,
                y: 32.0,
                out_sec: 0.0,
                out_at: None,
                center_x,
            })],
        };
        // 墨迹质心 x（相对纯背景有墨的像素均值）。
        let centroid_x = |frames: &[Vec<u8>], bg: &[Vec<u8>]| -> f32 {
            let (mut sx, mut n) = (0.0f64, 0.0f64);
            for (f, b) in frames.iter().zip(bg.iter()) {
                for (i, (pa, pb)) in f.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
                    if pa != pb {
                        sx += (i % 96) as f64;
                        n += 1.0;
                    }
                }
            }
            (sx / n) as f32
        };
        let bg_only = render_rgba_frames(&ClipSpec {
            tracks: vec![],
            ..mk(None)
        });
        let left = centroid_x(&render_rgba_frames(&mk(Some(24.0))), &bg_only);
        let mid = centroid_x(&render_rgba_frames(&mk(None)), &bg_only);
        let right = centroid_x(&render_rgba_frames(&mk(Some(72.0))), &bg_only);
        assert!(left < mid && mid < right, "锚点右移墨迹质心必须右移：{left} / {mid} / {right}");
        assert!(
            (left + 48.0 - right).abs() < 4.0,
            "24 ↔ 72 锚点质心差应 ≈ 48px（实测 {} → {}）",
            left,
            right
        );
    }

    /// E1/E3：`Track` 11 变体 `#[derive(Debug, Clone)]` 不破；
    /// `DynPathTrack` 手工 `Debug` 打印 `<dyn geom>`。
    #[test]
    fn track_new_variants_debug_clone_unbroken() {
        let d = DynPathTrack {
            geom: Arc::new(|_t| vec![[0.0, 0.0], [1.0, 1.0]]),
            color: [1.0; 4],
            width: 2.0,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        };
        let dbg = format!("{d:?}");
        assert!(dbg.contains("DynPathTrack"), "Debug 须打印结构体名：{dbg}");
        assert!(dbg.contains("<dyn geom>"), "Debug 须打印 <dyn geom>：{dbg}");
        // Clone 不破（Arc 语义：克隆后共享同一闭包，行为不变）。
        let d2 = d.clone();
        assert_eq!((d2.geom)(0.5), (d.geom)(0.5), "Clone 后几何行为必须一致");
        // Track / ClipSpec 的 derive(Debug, Clone) 对新变体生效。
        let track = Track::DynPath(d);
        let track2 = track.clone();
        assert!(format!("{track2:?}").contains("<dyn geom>"));
        let spec = ClipSpec {
            width: 8,
            height: 8,
            fps: 4,
            duration: 0.5,
            background: Background::DarkGlow,
            jpeg_quality: 80,
            tracks: vec![track],
        };
        let spec2 = spec.clone();
        assert!(format!("{spec2:?}").contains("DynPath"));
        // 11 变体穷尽构造健全性（Counter / Sequence 亦在列）。
        let all = [
            Track::Counter(CounterText {
                prefix: "x".into(),
                from: 0.0,
                to: 1.0,
                at: 0.0,
                dur_sec: 1.0,
                easing: EasingKind::Linear,
                decimals: 0,
                font: FontKind::Regular,
                size: 12.0,
                color: [1.0; 4],
                center_x: None,
                y: 4.0,
                letter_spacing: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                value_at: None,
            }),
            Track::Sequence(SequenceTrack {
                frames: vec![],
                width: 8,
                height: 8,
                src_fps: 15,
                at: 0.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
            }),
            track2,
        ];
        assert_eq!(all.len(), 3);
        for t in &all {
            assert!(!format!("{t:?}").is_empty());
        }
    }
}
