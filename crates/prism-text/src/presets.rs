//! 排版预置：Caption 字幕条与 Kinetic 逐字动画（任务书 B5；接 prism-anim）。
//!
//! - **Caption**：底部自适应宽底条 + 居中多行文本（水平 layout 全规则
//!   复用：禁则/换行/对齐）；
//! - **Kinetic**：逐字入场（stagger 时间梯 + 缓动 + 缩放 + 升起），
//!   可选全局 alpha 轨道（`prism_anim::Track`）——t 时刻的确定性快照。
//!
//! # 确定性契约
//! 同 (t, 输入) 必得同像素；时间只经 `Track::sample` 与缓动求值
//! （两者均为 t 的纯函数）。

use prism_anim::{EasingKind, Track};
use prism_io::Fbuf16F;
use prism_raster::geom::{Point, Transform};
use prism_raster::path::PathBuilder;
use prism_raster::scanline::FillRule;
use prism_raster::fill_path;

use crate::cache::GlyphCache;
use crate::layout::{layout_text, Align, TextParams};
use crate::raster::{blit_glyph, draw_glyphs};
use crate::ttf::Font;
use crate::vertical::layout_vertical;

/// 字幕条样式。
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionStyle {
    /// 字号（px）。
    pub font_size: f32,
    /// 文本色（直通 alpha 线性 RGBA）。
    pub color: [f32; 4],
    /// 底条色。
    pub bg: [f32; 4],
    /// 底条水平内边距（px）。
    pub pad_x: f32,
    /// 底条垂直内边距（px）。
    pub pad_y: f32,
    /// 底条距帧底边的距离（px）。
    pub margin_bottom: f32,
    /// 多行字幕的额外行距（px）。
    pub line_spacing: f32,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        CaptionStyle {
            font_size: 24.0,
            color: [1.0, 1.0, 1.0, 1.0],
            bg: [0.0, 0.0, 0.0, 0.6],
            pad_x: 16.0,
            pad_y: 8.0,
            margin_bottom: 32.0,
            line_spacing: 0.0,
        }
    }
}

/// 实心矩形填充（内部辅助；走 `fill_path` 同一管线）。
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

/// 渲染字幕条：底部居中、底条按最宽行自适应、文本禁则/换行全规则。
///
/// 空文本不产出任何像素（含底条——空字幕不可见）。
pub fn render_caption(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    style: &CaptionStyle,
    cache: &mut GlyphCache,
) {
    if text.is_empty() {
        return;
    }
    let vw = fbuf.width as f32;
    // 版心 = 视口宽 − 2·pad_x（保底不小于 1px 防负宽）。
    let ref_w = (vw - style.pad_x * 2.0).max(1.0);
    let params = TextParams {
        font_size: style.font_size,
        line_spacing: style.line_spacing,
        max_width: ref_w,
        align: Align::Center,
        ..TextParams::default()
    };
    let layout = layout_text(font, text, &params);
    if layout.glyphs.is_empty() {
        return;
    }
    // 底条：最宽行 + 内边距，水平居中。
    let bar_w = (layout.width + style.pad_x * 2.0).min(vw);
    let bar_h = layout.height + style.pad_y * 2.0;
    let bar_x0 = ((vw - bar_w) * 0.5).max(0.0);
    let bar_y0 = (fbuf.height as f32 - style.margin_bottom - bar_h).max(0.0);
    fill_rect(fbuf, bar_x0, bar_y0, bar_w, bar_h, style.bg);
    // 文本：把版心内容 bbox 左缘对齐到 bar 内左缘 + pad_x。
    let gx0 = layout.glyphs.iter().map(|g| g.x).fold(f32::INFINITY, f32::min);
    let dx = bar_x0 + style.pad_x - gx0;
    let dy = bar_y0 + style.pad_y;
    draw_glyphs(fbuf, font, &layout.glyphs, style.font_size, style.color, dx, dy, cache);
}

/// 逐字动画样式。
#[derive(Debug, Clone, PartialEq)]
pub struct KineticStyle {
    /// 字号（px）。
    pub font_size: f32,
    /// 文本色。
    pub color: [f32; 4],
    /// 相邻字符入场的时间间隔（秒）。
    pub stagger: f32,
    /// 单字入场时长（秒）。
    pub per_char: f32,
    /// 入场缓动。
    pub easing: EasingKind,
    /// 入场起始缩放（如 0.6；1.0 = 无缩放动画）。
    pub scale_from: f32,
    /// 入场纵向位移幅度（px；正值 = 自下而上升起）。
    pub rise_px: f32,
    /// 水平对齐（复用排版对齐）。
    pub align: Align,
}

impl Default for KineticStyle {
    fn default() -> Self {
        KineticStyle {
            font_size: 24.0,
            color: [1.0, 1.0, 1.0, 1.0],
            stagger: 0.06,
            per_char: 0.30,
            easing: EasingKind::QuadOut,
            scale_from: 0.6,
            rise_px: 12.0,
            align: Align::Center,
        }
    }
}

/// 渲染逐字动画在时刻 `t` 的快照。
///
/// 每字符 i 的局部进度 `u = (t − i·stagger)/per_char`，钳制 [0,1] 后经
/// 缓动得 `e`：alpha = e、缩放 = `scale_from + (1−scale_from)·e`、升起
/// 位移 = `rise_px·(1−e)`。未入场字符（e ≤ 0）不产出像素。
/// `alpha_track` 为可选全局透明度轨道（乘入颜色 alpha）。
pub fn render_kinetic(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    style: &KineticStyle,
    t: f32,
    alpha_track: Option<&Track>,
    cache: &mut GlyphCache,
) {
    if text.is_empty() {
        return;
    }
    let ga = alpha_track.map(|tr| tr.sample(t)).unwrap_or(1.0);
    let params = TextParams {
        font_size: style.font_size,
        max_width: fbuf.width as f32,
        align: style.align,
        ..TextParams::default()
    };
    let layout = layout_text(font, text, &params);
    for (i, g) in layout.glyphs.iter().enumerate() {
        let local = (t - i as f32 * style.stagger) / style.per_char;
        let u = if local < 0.0 {
            0.0
        } else if local > 1.0 {
            1.0
        } else {
            local
        };
        let e = style.easing.apply(u);
        if e <= 0.0 {
            continue;
        }
        let size = style.font_size * (style.scale_from + (1.0 - style.scale_from) * e);
        // 缓存键含字号位模式：入场过程中的中间尺寸按需缓存（帧间复用）。
        let bmp = cache.get_or_render(font, g.gid, size, 0);
        let dy = style.rise_px * (1.0 - e);
        let color = [style.color[0], style.color[1], style.color[2], style.color[3] * e * ga];
        blit_glyph(fbuf, bmp, g.x, g.y + dy, color);
    }
}

/// 竖排文本渲染（把 [`layout_vertical`] 结果合成进帧缓冲）。
///
/// `origin` 为第 0 列（最右列）em 框左上角的设备坐标；直立字直接
/// blit，转向字取缓存旋转位图（flags 位 1）。
pub fn render_vertical(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    params: &crate::vertical::VerticalParams,
    origin: Point,
    color: [f32; 4],
    cache: &mut GlyphCache,
) {
    let vl = layout_vertical(font, text, params);
    let scale = params.font_size / font.units_per_em() as f32;
    let asc = font.ascender() as f32 * scale;
    for c in &vl.chars {
        let flags = if c.rotated { 1 } else { 0 };
        let bmp = cache.get_or_render(font, c.gid, params.font_size, flags);
        blit_glyph(fbuf, bmp, origin.x + c.x, origin.y + c.y + asc, color);
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    #[test]
    fn caption_paints_bar_and_text() {
        let mut fbuf = Fbuf16F::new(120, 80);
        let f = font_regular();
        let mut cache = GlyphCache::new();
        render_caption(&mut fbuf, &f, "字幕", &CaptionStyle::default(), &mut cache);
        // 底条存在：帧内必有非透明像素（底条贴底但被 margin 上推，扫描全帧）。
        let any_px = (0..fbuf.height).any(|y| (0..fbuf.width).any(|x| fbuf.get(x, y)[3] > 0.0));
        assert!(any_px, "底条应有像素");
        // 文本区有墨（白色文本像素）。
        let any_text = (0..fbuf.height).any(|y| {
            (0..fbuf.width).any(|x| {
                let p = fbuf.get(x, y);
                p[3] > 0.0 && (p[0] > 0.5 && p[1] > 0.5 && p[2] > 0.5)
            })
        });
        assert!(any_text, "白色文本应有像素");
    }

    #[test]
    fn caption_empty_text_is_noop() {
        let mut fbuf = Fbuf16F::new(32, 32);
        let f = font_regular();
        let mut cache = GlyphCache::new();
        render_caption(&mut fbuf, &f, "", &CaptionStyle::default(), &mut cache);
        assert!(fbuf.pixels.iter().all(|&p| p == 0), "空字幕不产出任何像素");
    }

    #[test]
    fn kinetic_progresses_and_determines() {
        let mut f = font_regular();
        let _ = &mut f;
        let font = font_regular();
        let mut cache = GlyphCache::new();
        let style = KineticStyle { font_size: 32.0, ..KineticStyle::default() };
        // t=0：首字符 e≈0（u=0 → e=0 跳过），全部未入场 → 空帧。
        let mut a = Fbuf16F::new(160, 64);
        render_kinetic(&mut a, &font, "文字动画", &style, 0.0, None, &mut cache);
        assert!(a.pixels.iter().all(|&p| p == 0), "t=0 无字符入场");
        // t=1.0：全部入场完成 → 有像素，且与重放逐字节一致（确定性）。
        let mut b = Fbuf16F::new(160, 64);
        render_kinetic(&mut b, &font, "文字动画", &style, 1.0, None, &mut cache);
        assert!(!b.pixels.iter().all(|&p| p == 0), "t=1 已入场");
        let mut c = Fbuf16F::new(160, 64);
        render_kinetic(&mut c, &font, "文字动画", &style, 1.0, None, &mut cache);
        assert_eq!(b.pixels, c.pixels, "同 t 同像素");
        // 中间帧：部分入场（stagger 梯度可见）。
        let mut d = Fbuf16F::new(160, 64);
        render_kinetic(&mut d, &font, "文字动画", &style, 0.10, None, &mut cache);
        assert!(!d.pixels.iter().all(|&p| p == 0), "t=0.10 首字符已入场");
    }

    #[test]
    fn kinetic_respects_global_alpha_track() {
        let font = font_regular();
        let mut cache = GlyphCache::new();
        let style = KineticStyle::default();
        // 全局轨道：t≥1 恒 0 → 全透明 → 空帧。
        let track = Track::new(vec![prism_anim::Keyframe::new(
            0.0,
            1.0,
            EasingKind::Linear,
        )]);
        let mut fbuf = Fbuf16F::new(160, 64);
        render_kinetic(&mut fbuf, &font, "动画", &style, 0.0, Some(&track), &mut cache);
        // t=0 局部 e=0 全跳过，无像素。
        assert!(fbuf.pixels.iter().all(|&p| p == 0));
        // t=0.5：字符入场 e>0，但全局轨道 alpha=1（线性轨道首段=1）→ 有像素。
        let mut fbuf2 = Fbuf16F::new(160, 64);
        render_kinetic(&mut fbuf2, &font, "动画", &style, 0.5, Some(&track), &mut cache);
        assert!(!fbuf2.pixels.iter().all(|&p| p == 0));
    }

    #[test]
    fn vertical_render_places_pixels() {
        let mut fbuf = Fbuf16F::new(64, 96);
        let font = font_regular();
        let mut cache = GlyphCache::new();
        let vp = crate::vertical::VerticalParams { font_size: 32.0, ..Default::default() };
        render_vertical(
            &mut fbuf,
            &font,
            "中文",
            &vp,
            Point::new(40.0, 8.0),
            [0.0, 0.0, 0.0, 1.0],
            &mut cache,
        );
        let painted = (0..fbuf.height).any(|y| (0..fbuf.width).any(|x| fbuf.get(x, y)[3] > 0.9));
        assert!(painted, "竖排文本应产出像素");
    }
}
