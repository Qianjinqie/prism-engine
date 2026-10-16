//! 高层作者 API 概念验证（PoC）——“5 秒短片 = 一次函数调用”。
//!
//! 本文件证明“强大版”路线图里「视频即代码」的愿景可落地：作者只写一份
//! **声明式 `ClipSpec`**（画幅 / 背景预设 / 若干轨道），再调用一次
//! `render_short(&spec)`，即可产出一支干净的 5 秒文字片头 MP4。
//!
//! 所有“脏活”都被收进 [`prism_author`] 模块内部：
//! - 背景绘制（极低频 simplex 微波动 + 辉光 + 暗角 + 呼吸）复用 `prism-noise::Fbm`；
//! - 逐字 kinetic 入场 / 静态文字 / 字幕条复用 `prism-text` 的
//!   `layout_text` / `draw_glyphs` / `blit_glyph` / `render_caption`；
//! - 装饰线复用引擎矢量填充 `prism-raster::fill_path`（非手写像素循环）；
//! - 120 帧循环 + RGBA16F→RGBA8 交接 + 官方 muxer 封装全部隐藏。
//! 调用方（[`main`]）不碰 `Fbuf16F`、`GlyphCache`、帧循环与编码器。

// ===========================================================================
// 高层作者 API：把 410 行手写渲染收进内部实现
// ===========================================================================

mod prism_author {
    pub use prism_anim::EasingKind;
    use prism_io::Fbuf16F;
    use prism_mux::{write_mp4_jpeg_color, JpegConfig, TrackSpec};
    use prism_noise::fbm::Fbm;
    use prism_raster::fill_path;
    use prism_raster::geom::{Point, Transform};
    use prism_raster::path::PathBuilder;
    use prism_raster::scanline::FillRule;
    use prism_text::cache::GlyphCache;
    use prism_text::layout::{layout_text, Align, TextParams};
    pub use prism_text::presets::CaptionStyle;
    use prism_text::presets::render_caption;
    use prism_text::raster::{blit_glyph, draw_glyphs};
    use prism_text::ttf::Font;
    use prism_text::{font_bold, font_regular};

    // ---- 通用数学辅助（全部 t 的纯函数）----

    /// 三次平滑插值（smoothstep），用于淡入/暗角等缓动。
    fn smooth(t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// 标量线性插值。
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    /// 三维（RGB）线性插值。
    fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
    }

    // ---- 声明式规格（作者只写这些）----

    /// 字体选择。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FontKind {
        Bold,
        Regular,
    }

    /// 背景预设。
    #[derive(Debug, Clone)]
    #[allow(dead_code)] // `Gradient` 是公开作者 API 的一部分（规格要求）；本 PoC 仅演示 DarkGlow。
    pub enum Background {
        /// 自定义垂直渐变（top → bottom）。
        Gradient { top: [f32; 3], bottom: [f32; 3] },
        /// 项目默认：深藏青 → 靛蓝 + 冷青辉光（即原 demo 视觉）。
        DarkGlow,
    }

    /// 逐字 kinetic 入场轨道。
    #[derive(Debug, Clone)]
    pub struct KineticText {
        pub content: String,
        pub font: FontKind,
        pub size: f32,
        pub color: [f32; 4],
        /// 入场起始秒。
        pub at: f32,
        /// 相邻字符入场间隔（秒）。
        pub stagger: f32,
        /// 单字入场时长（秒）。
        pub per_char: f32,
        /// 入场起始缩放（1.0 = 无缩放动画）。
        pub scale_from: f32,
        /// 入场纵向升起幅度（px）。
        pub rise_px: f32,
        pub easing: EasingKind,
        /// 基线 y（像素）。
        pub y: f32,
    }

    /// 静态（带字距、可淡入）文字轨道。
    #[derive(Debug, Clone)]
    pub struct StaticText {
        pub content: String,
        pub font: FontKind,
        pub size: f32,
        pub color: [f32; 4],
        /// 淡入起始秒。
        pub at: f32,
        /// 淡入时长（秒）。
        pub dur: f32,
        pub letter_spacing: f32,
        /// 基线 y（像素）。
        pub y: f32,
    }

    /// 装饰线（矩形色条，可淡入）。
    #[derive(Debug, Clone)]
    pub struct AccentLine {
        pub center_x: f32,
        pub y: f32,
        pub w: f32,
        pub h: f32,
        pub color: [f32; 4],
        /// 淡入起始秒。
        pub at: f32,
        /// 淡入时长（秒）。
        pub dur: f32,
    }

    /// 收尾字幕条轨道（复用 `prism-text::presets::render_caption`）。
    #[derive(Debug, Clone)]
    pub struct Caption {
        pub content: String,
        /// 淡入起始秒。
        pub at: f32,
        /// 淡入时长（秒）。
        pub dur: f32,
        pub style: CaptionStyle,
    }

    /// 轨道枚举：一支短片由若干轨道构成。
    #[derive(Debug, Clone)]
    pub enum Track {
        Kinetic(KineticText),
        Static(StaticText),
        Accent(AccentLine),
        Caption(Caption),
    }

    /// 短片声明式规格——作者对外唯一需要构造的结构。
    #[derive(Debug, Clone)]
    pub struct ClipSpec {
        pub width: u32,
        pub height: u32,
        pub fps: u32,
        pub duration: f32,
        pub background: Background,
        pub jpeg_quality: u8,
        pub tracks: Vec<Track>,
    }

    impl ClipSpec {
        /// 帧数（向上取整，保证末秒不丢帧）。
        pub fn frame_count(&self) -> usize {
            (self.fps as f32 * self.duration).ceil() as usize
        }
    }

    // ---- 内部实现（作者无需关心）----

    /// 默认深色渐变两端（与原 demo 完全一致）。
    const BG_TOP: [f32; 3] = [0.035, 0.055, 0.130]; // ≈ #090E21
    const BG_BOTTOM: [f32; 3] = [0.058, 0.090, 0.205]; // ≈ #0F1734
    /// 辉光冷青调（叠加在渐变上）。
    const GLOW: [f32; 3] = [0.030, 0.070, 0.120];

    /// 绘制背景：平滑垂直渐变 + 柔光 + 极低频 simplex 微波动 + 暗角 + 呼吸。
    /// 复用 `prism-noise::Fbm`，全套逻辑封装在 API 内部（原 demo 的嵌套循环）。
    fn paint_background(
        fbuf: &mut Fbuf16F,
        t: f32,
        fbm: &Fbm,
        seed: u64,
        top: [f32; 3],
        bottom: [f32; 3],
    ) {
        let w = fbuf.width as f32;
        let h = fbuf.height as f32;

        // 低分辨率噪声网格（双线性上采样 → 极低频、绝无高频噪点）。
        let gw: u32 = 56;
        let gh: u32 = 32;
        let mut grid = vec![0.0f32; (gw * gh) as usize];
        for gy in 0..gh {
            for gx in 0..gw {
                let nx = gx as f32 / gw as f32 * 2.2;
                let ny = gy as f32 / gh as f32 * 1.3;
                let v = fbm.simplex2(seed, nx + t * 0.05, ny - t * 0.035);
                grid[(gy * gw + gx) as usize] = v;
            }
        }

        let breath = 1.0 + 0.015 * (t * (std::f32::consts::TAU / 5.0)).sin();
        let glow_cx = 0.5;
        let glow_cy = 0.40;

        for y in 0..fbuf.height {
            let vy = y as f32 / (h - 1.0);
            for x in 0..fbuf.width {
                let vx = x as f32 / (w - 1.0);

                // 1) 垂直渐变。
                let mut col = lerp3(top, bottom, smooth(vy));

                // 2) 柔和辉光（二次衰减，冷青调）。
                let dx = vx - glow_cx;
                let dy = vy - glow_cy;
                let d2 = dx * dx + dy * dy;
                let mut g = 1.0 - d2 * 3.0;
                if g < 0.0 {
                    g = 0.0;
                }
                g *= g;
                col[0] += g * GLOW[0];
                col[1] += g * GLOW[1];
                col[2] += g * GLOW[2];

                // 3) 极低对比缓慢波动（双线性采样）。
                let gx_f = vx * (gw - 1) as f32;
                let gy_f = vy * (gh - 1) as f32;
                let gxi0 = gx_f.floor() as u32;
                let gyi0 = gy_f.floor() as u32;
                let gxi1 = (gxi0 + 1).min(gw - 1);
                let gyi1 = (gyi0 + 1).min(gh - 1);
                let fx = gx_f - gxi0 as f32;
                let fy = gy_f - gyi0 as f32;
                let v00 = grid[(gyi0 * gw + gxi0) as usize];
                let v10 = grid[(gyi0 * gw + gxi1) as usize];
                let v01 = grid[(gyi1 * gw + gxi0) as usize];
                let v11 = grid[(gyi1 * gw + gxi1) as usize];
                let n = lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
                let modulation = 1.0 + 0.03 * n;

                // 4) 暗角。
                let cx = vx - 0.5;
                let cy = vy - 0.5;
                let rad = (cx * cx + cy * cy).sqrt();
                let vig = 1.0 - smooth((rad - 0.45) / 0.55).max(0.0) * 0.55;

                // 5) 合成。
                let k = modulation * breath * vig;
                fbuf.set(
                    x,
                    y,
                    [
                        (col[0] * k).clamp(0.0, 1.0),
                        (col[1] * k).clamp(0.0, 1.0),
                        (col[2] * k).clamp(0.0, 1.0),
                        1.0,
                    ],
                );
            }
        }
    }

    /// 逐字 kinetic 入场（复用 `prism-text` 的 `layout_text` / `blit_glyph` /
    /// `GlyphCache` / `EasingKind`；相对 `presets::render_kinetic` 仅多了基线
    /// 定位偏移，使之可自由摆放到任意 `y`）。
    fn draw_kinetic(
        fbuf: &mut Fbuf16F,
        font: &Font,
        text: &str,
        size: f32,
        color: [f32; 4],
        center_x: f32,
        baseline_y: f32,
        stagger: f32,
        per_char: f32,
        scale_from: f32,
        rise_px: f32,
        easing: EasingKind,
        t: f32,
        cache: &mut GlyphCache,
    ) {
        if text.is_empty() {
            return;
        }
        let params = TextParams {
            font_size: size,
            max_width: f32::INFINITY,
            align: Align::Center,
            ..TextParams::default()
        };
        let layout = layout_text(font, text, &params);
        if layout.glyphs.is_empty() {
            return;
        }
        let scale = size / font.units_per_em() as f32;
        let asc = font.ascender() as f32 * scale;
        let shift_x = center_x - layout.width * 0.5;
        let shift_y = baseline_y - asc;
        for (i, g) in layout.glyphs.iter().enumerate() {
            let local = (t - (i as f32) * stagger) / per_char;
            let u = local.clamp(0.0, 1.0);
            let e = easing.apply(u);
            if e <= 0.0 {
                continue; // 未入场字符不产出像素
            }
            let s = size * (scale_from + (1.0 - scale_from) * e);
            let bmp = cache.get_or_render(font, g.gid, s, 0);
            let dy = rise_px * (1.0 - e);
            let col = [color[0], color[1], color[2], color[3] * e];
            blit_glyph(fbuf, bmp, g.x + shift_x, g.y + shift_y + dy, col);
        }
    }

    /// 静态居中文字（带字距），用于题头等小标签。
    fn draw_static(
        fbuf: &mut Fbuf16F,
        font: &Font,
        text: &str,
        size: f32,
        color: [f32; 4],
        center_x: f32,
        baseline_y: f32,
        letter_spacing: f32,
        cache: &mut GlyphCache,
    ) {
        if text.is_empty() {
            return;
        }
        let params = TextParams {
            font_size: size,
            letter_spacing,
            max_width: f32::INFINITY,
            align: Align::Center,
            ..TextParams::default()
        };
        let layout = layout_text(font, text, &params);
        if layout.glyphs.is_empty() {
            return;
        }
        let scale = size / font.units_per_em() as f32;
        let asc = font.ascender() as f32 * scale;
        let shift_x = center_x - layout.width * 0.5;
        let shift_y = baseline_y - asc;
        draw_glyphs(fbuf, font, &layout.glyphs, size, color, shift_x, shift_y, cache);
    }

    /// 实心矩形填充：复用引擎矢量填充管线（`prism-raster::fill_path`），
    /// 不手写像素循环。
    fn fill_rect(fbuf: &mut Fbuf16F, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.move_to(Point::new(x, y));
        pb.line_to(Point::new(x + w, y));
        pb.line_to(Point::new(x + w, y + h));
        pb.line_to(Point::new(x, y + h));
        pb.close();
        fill_path(fbuf, &pb, &Transform::IDENTITY, color, FillRule::Winding);
    }

    /// 把一支声明式短片渲染为 MP4 字节流（MJPEG 4:4:4，官方 muxer，零私有 shim）。
    pub fn render_short(spec: &ClipSpec) -> Vec<u8> {
        let w = spec.width;
        let h = spec.height;
        let fps = spec.fps;
        let frame_count = spec.frame_count();

        // 字体 / 缓存 / 背景噪声——全部在 API 内部处理。
        let bold = font_bold();
        let regular = font_regular();
        let mut cache = GlyphCache::new();
        let fbm = Fbm::new(2, 2.0, 0.5);
        let seed: u64 = 0x9E37_79B1;

        let (bg_top, bg_bottom) = match &spec.background {
            Background::DarkGlow => (BG_TOP, BG_BOTTOM),
            Background::Gradient { top, bottom } => (*top, *bottom),
        };

        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(frame_count);

        // 120 帧循环被收进 API 内部。
        for fi in 0..frame_count {
            let t = fi as f32 / fps as f32; // 0 .. duration 秒

            let mut fbuf = Fbuf16F::new(w, h);
            paint_background(&mut fbuf, t, &fbm, seed, bg_top, bg_bottom);

            for track in &spec.tracks {
                match track {
                    Track::Kinetic(k) => {
                        let font = match k.font {
                            FontKind::Bold => &bold,
                            FontKind::Regular => &regular,
                        };
                        draw_kinetic(
                            &mut fbuf, font, &k.content, k.size, k.color, w as f32 * 0.5, k.y,
                            k.stagger, k.per_char, k.scale_from, k.rise_px, k.easing, t - k.at,
                            &mut cache,
                        );
                    }
                    Track::Static(s) => {
                        let a = smooth(((t - s.at) / s.dur).clamp(0.0, 1.0));
                        if a > 0.0 {
                            let font = match s.font {
                                FontKind::Bold => &bold,
                                FontKind::Regular => &regular,
                            };
                            let mut col = s.color;
                            col[3] *= a;
                            draw_static(
                                &mut fbuf, font, &s.content, s.size, col, w as f32 * 0.5, s.y,
                                s.letter_spacing, &mut cache,
                            );
                        }
                    }
                    Track::Accent(a) => {
                        let fa = smooth(((t - a.at) / a.dur).clamp(0.0, 1.0)) * a.color[3];
                        if fa > 0.0 {
                            let mut col = a.color;
                            col[3] = fa;
                            fill_rect(
                                &mut fbuf,
                                a.center_x - a.w * 0.5,
                                a.y,
                                a.w,
                                a.h,
                                col,
                            );
                        }
                    }
                    Track::Caption(c) => {
                        let cf = smooth(((t - c.at) / c.dur).clamp(0.0, 1.0));
                        if cf > 0.0 {
                            let mut style = c.style.clone();
                            style.color[3] *= cf;
                            style.bg[3] *= cf;
                            render_caption(&mut fbuf, &regular, &c.content, &style, &mut cache);
                        }
                    }
                }
            }

            frames.push(fbuf.to_framebuffer().pixels);
        }

        // 官方 muxer 封装（修复后的 size_category 保证 DC 预测链无漂移）。
        let mux_spec = TrackSpec { width: w, height: h, fps };
        write_mp4_jpeg_color(
            &frames,
            mux_spec,
            None,
            0,
            0,
            JpegConfig { quality: spec.jpeg_quality, grayscale: false },
        )
        .expect("MP4 封装失败")
    }
}

// ===========================================================================
// 作者体验：“强大版” = 声明一份 spec + 一次 render_short
// ===========================================================================

use prism_author::*;

fn main() {
    // 一份声明式规格，描述与原 demo 同款的 5 秒文字片头。
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
                dur: 0.5,
                letter_spacing: 7.0,
                y: 220.0,
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
            }),
            Track::Accent(AccentLine {
                center_x: 640.0,
                y: 398.0,
                w: 320.0,
                h: 3.0,
                color: [0.30, 0.75, 0.95, 0.9],
                at: 1.3,
                dur: 0.6,
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
            }),
            Track::Caption(Caption {
                content: "视频即代码 · 单文件源码分发".into(),
                at: 3.2,
                dur: 0.6,
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

    // 5 秒短片 = 一次函数调用。
    let mp4 = render_short(&spec);
    std::fs::write("/workspace/prism_aurora_v2.mp4", &mp4).expect("写入 MP4 失败");
    eprintln!(
        "prism_author::render_short 完成：{} 字节，{} 帧 @{}×{} {}fps（官方 muxer）",
        mp4.len(),
        spec.frame_count(),
        spec.width,
        spec.height,
        spec.fps
    );
}
