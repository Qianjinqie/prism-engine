//! Prism Engine — 演示动画「棱镜 · Prism」（文字为主角版）
//!
//! 用 Prism 自有的文本栈（`prism-text`：OpenType 解析 → 整形 → 光栅化 → 缓存
//! → 排版）渲染一支**干净、克制、以文字为主**的 5 秒片头：平滑深色渐变背景
//! （极低频 simplex 微波动 ±0.03，绝非高频噪声场）+ Bold 主标题逐字 kinetic
//! 入场 + Regular 副标题 + 收尾字幕条。全程 t 的纯函数（无墙钟、无随机状态）。
//!
//! # 交付管线说明（重要）
//!
//! 渲染输出经 `prism_io::Fbuf16F::to_framebuffer()` 交接到 RGBA8，随后直接调用
//! 引擎官方 muxer `prism_mux::write_mp4_jpeg_color` 编码封装为 MP4（MJPEG 4:4:4
//! 彩色，零第三方 crate、零 ffmpeg）。`prism-mux` 的 `jpeg::size_category`
//! 此前把 0 误归为类别 1，导致零 DC 差分被解码成 −1、DC 预测链逐块漂移、平坦
//! 画布崩成全黑——现已在引擎内修复（`size_category(0) == 0`），本 demo 因此
//! 无需任何私有编码 shim，全程使用官方路径。

use prism_anim::EasingKind;
use prism_io::Fbuf16F;
use prism_mux::{write_mp4_jpeg_color, JpegConfig, TrackSpec};
use prism_noise::fbm::Fbm;
use prism_text::cache::GlyphCache;
use prism_text::layout::{layout_text, Align, TextParams};
use prism_text::presets::{render_caption, CaptionStyle};
use prism_text::raster::{blit_glyph, draw_glyphs};
use prism_text::ttf::Font;
use prism_text::{font_bold, font_regular};

// ===========================================================================
// 第一部分：时间轴 / 合成辅助（全部 t 的纯函数）
// ===========================================================================

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

/// 直通 alpha 的 src-over 合成。
fn blend_src_over(dst: [f32; 4], src: [f32; 4]) -> [f32; 4] {
    let a = src[3];
    [
        src[0] * a + dst[0] * (1.0 - a),
        src[1] * a + dst[1] * (1.0 - a),
        src[2] * a + dst[2] * (1.0 - a),
        1.0,
    ]
}

/// 把一块矩形以给定颜色（含 alpha）合成进帧缓冲（带视口裁剪）。
fn fill_rect_alpha(fbuf: &mut Fbuf16F, x0: i32, y0: i32, w: i32, h: i32, color: [f32; 4]) {
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            if x < 0 || y < 0 {
                continue;
            }
            let (xu, yu) = (x as u32, y as u32);
            if xu >= fbuf.width || yu >= fbuf.height {
                continue;
            }
            let dst = fbuf.get(xu, yu);
            fbuf.set(xu, yu, blend_src_over(dst, color));
        }
    }
}

// ===========================================================================
// 第二部分：背景（平滑渐变 + 柔光 + 极低频微波动 + 暗角 + 呼吸）
// ===========================================================================

/// 渐变两端：深藏青 → 靛蓝（平滑垂直渐变）。
const BG_TOP: [f32; 3] = [0.035, 0.055, 0.130]; // ≈ #090E21
const BG_BOTTOM: [f32; 3] = [0.058, 0.090, 0.205]; // ≈ #0F1734

/// 绘制背景。波动项来自 `prism-noise` simplex 的**低分辨率网格 + 双线性上
/// 采样**（极低频、极慢漂移），亮度振幅仅 ±3%——平滑色场，绝无高频噪点。
fn paint_background(fbuf: &mut Fbuf16F, t: f32, fbm: &Fbm, seed: u64) {
    let w = fbuf.width as f32;
    let h = fbuf.height as f32;

    // 低分辨率噪声网格（56×32，每帧约 1.8k 次 simplex 调用，可忽略）。
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

    // 全局呼吸系数（单一 sin，确定性，振幅 ±1.5%）。
    let breath = 1.0 + 0.015 * (t * (std::f32::consts::TAU / 5.0)).sin();

    // 辉光中心（偏上，呼应标题位置）。
    let glow_cx = 0.5;
    let glow_cy = 0.40;

    for y in 0..fbuf.height {
        let vy = y as f32 / (h - 1.0);
        for x in 0..fbuf.width {
            let vx = x as f32 / (w - 1.0);

            // 1) 垂直渐变。
            let mut col = lerp3(BG_TOP, BG_BOTTOM, smooth(vy));

            // 2) 柔和辉光（二次衰减，冷青调）。
            let dx = vx - glow_cx;
            let dy = vy - glow_cy;
            let d2 = dx * dx + dy * dy;
            let mut g = 1.0 - d2 * 3.0;
            if g < 0.0 {
                g = 0.0;
            }
            g *= g;
            col[0] += g * 0.030;
            col[1] += g * 0.070;
            col[2] += g * 0.120;

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

// ===========================================================================
// 第三部分：文字（复用 prism-text 排版/光栅/缓存）
// ===========================================================================

/// 逐字 kinetic 入场（与 `prism-text::presets::render_kinetic` 同语义，额外
/// 支持把整块文本居中到 `(center_x, baseline_y)`——预设无偏移、会画在帧顶）。
///
/// 每字符 `i`：局部进度 `u = (t − i·stagger)/per_char` 钳 [0,1] → 缓动 `e`；
/// alpha = e、缩放 = scale_from + (1−scale_from)·e、升起 = rise_px·(1−e)。
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

/// 静态居中文字（带字距），用于题头小标签。
fn draw_centered(
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

// ===========================================================================
// main：5 秒时间轴（全部 t 的纯函数）
//
//   0.0–0.4s  纯背景
//   0.3s 起   题头 "PRISM ENGINE"（小字距标签）淡入
//   0.4–1.8s  主标题「棱镜引擎」Bold 逐字 kinetic 入场
//   1.3s 起   标题下青色装饰短线淡入
//   1.9–3.2s  副标题「程序化视频 · 零外部依赖」逐字入场
//   3.2–5.0s  收尾字幕条「视频即代码 · 单文件源码分发」淡入后定格
// ===========================================================================

fn main() {
    // ---- 画幅 / 节奏 ----
    let w: u32 = 1280;
    let h: u32 = 720;
    let fps: u32 = 24;
    let duration: f32 = 5.0;
    let frame_count = (fps as f32 * duration).ceil() as usize; // 120 帧

    // 内嵌字体（零文件 IO；OFL 1.1 Prism Sans SC 子集：中/英/中文标点齐全）。
    let bold = font_bold();
    let regular = font_regular();
    let mut cache = GlyphCache::new();

    // 背景极低频 simplex（振幅在绘制时压到 ±3%）。
    let fbm = Fbm::new(2, 2.0, 0.5);
    let seed: u64 = 0x9E37_79B1;

    // 构图锚点（720p）。
    let center_x = w as f32 * 0.5;
    let kicker_baseline = 220.0;
    let title_baseline = 350.0;
    let accent_y = 398.0;
    let subtitle_baseline = 462.0;

    let spec = TrackSpec { width: w, height: h, fps };
    let quality: u8 = 88;

    // 逐帧 RGBA8 缓冲（交给引擎官方 muxer 编码封装）。
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(frame_count);

    for fi in 0..frame_count {
        let t = fi as f32 / fps as f32; // 0 .. 5 秒

        // 1) 背景。
        let mut fbuf = Fbuf16F::new(w, h);
        paint_background(&mut fbuf, t, &fbm, seed);

        // 2) 题头小标签 "PRISM ENGINE"（0.3s 起淡入）。
        let kicker_alpha = smooth(((t - 0.3) / 0.5).clamp(0.0, 1.0));
        if kicker_alpha > 0.0 {
            draw_centered(
                &mut fbuf,
                &regular,
                "PRISM ENGINE",
                22.0,
                [0.46, 0.78, 0.92, kicker_alpha * 0.85],
                center_x,
                kicker_baseline,
                7.0,
                &mut cache,
            );
        }

        // 3) 主标题「棱镜引擎」（Bold，逐字 kinetic 入场：0.4s 起）。
        draw_kinetic(
            &mut fbuf,
            &bold,
            "棱镜引擎",
            96.0,
            [0.98, 0.98, 1.0, 1.0],
            center_x,
            title_baseline,
            0.09, // stagger
            0.45, // per_char
            0.72, // scale_from
            22.0, // rise_px
            EasingKind::CubicOut,
            t - 0.4,
            &mut cache,
        );

        // 4) 标题下青色装饰短线（1.3s 起淡入）。
        let accent_alpha = smooth(((t - 1.3) / 0.6).clamp(0.0, 1.0)) * 0.9;
        if accent_alpha > 0.0 {
            fill_rect_alpha(
                &mut fbuf,
                (center_x - 160.0) as i32,
                accent_y as i32,
                320,
                3,
                [0.30, 0.75, 0.95, accent_alpha],
            );
        }

        // 5) 副标题「程序化视频 · 零外部依赖」（1.9s 起逐字入场）。
        draw_kinetic(
            &mut fbuf,
            &regular,
            "程序化视频 · 零外部依赖",
            32.0,
            [0.82, 0.87, 0.96, 1.0],
            center_x,
            subtitle_baseline,
            0.045,
            0.40,
            0.80,
            10.0,
            EasingKind::QuadOut,
            t - 1.9,
            &mut cache,
        );

        // 6) 收尾字幕条（底部自适应底条 + 居中文本）：3.2s 起淡入。
        let cap_fade = smooth(((t - 3.2) / 0.6).clamp(0.0, 1.0));
        if cap_fade > 0.0 {
            let mut cap = CaptionStyle::default();
            cap.font_size = 28.0;
            cap.color = [1.0, 1.0, 1.0, cap_fade];
            cap.bg = [0.0, 0.0, 0.0, 0.55 * cap_fade];
            cap.pad_x = 22.0;
            cap.pad_y = 12.0;
            cap.margin_bottom = 40.0;
            render_caption(&mut fbuf, &regular, "视频即代码 · 单文件源码分发", &cap, &mut cache);
        }

        // 7) 交付边界：RGBA16F → RGBA8（alpha 固定 255）→ 收集为帧。
        let fb = fbuf.to_framebuffer();
        frames.push(fb.pixels);
    }

    // 8) 封装：引擎官方 muxer（修复后的 size_category 保证 DC 预测链无漂移）。
    let mp4 = write_mp4_jpeg_color(
        &frames,
        spec,
        None,
        0,
        0,
        JpegConfig { quality, grayscale: false },
    )
    .expect("MP4 封装失败");

    let out_path = "prism_aurora.mp4";
    std::fs::write(out_path, &mp4).expect("写入失败");
    eprintln!(
        "已生成 {}：{} 字节，{} 帧 @{}×{} {}fps（JPEG q{}，MJPEG 4:4:4，官方 muxer）",
        out_path,
        mp4.len(),
        frame_count,
        w,
        h,
        fps,
        quality
    );
}
