//! prism-text 边缘攻击补充（W5-B5 QA）：退化输入、极端参数、缓存键正确性、
//! 布局/竖排确定性。每个断言标注"失败代表何种缺陷"。
//!
//! 与 `qa_edge_attack.rs`（任务书正条/golden）互补，专攻边界。

use prism_text::cache::GlyphCache;
use prism_text::layout::{layout_text, Align, TextParams};
use prism_text::raster::render_glyph;
use prism_text::vertical::{is_rotated_cw, layout_vertical, VerticalParams};
use prism_text::font_regular;

// ---------------------------------------------------------------------------
// 布局退化输入
// ---------------------------------------------------------------------------

/// 空文本：产出空 glyphs、空 lines，不 panic；width/height 有限。
/// 失败 => 空串触发除零/越界。
#[test]
fn layout_empty_text_is_empty_and_finite() {
    let f = font_regular();
    let l = layout_text(&f, "", &TextParams::default());
    assert!(l.glyphs.is_empty(), "空文本无字形");
    assert!(l.lines.is_empty(), "空文本无行");
    assert!(l.width.is_finite() && l.height.is_finite());
}

/// max_width=0：每非空行至多容纳一个字符（禁则悬挂可能 1+），且行数=字符数上界。
/// 失败 => 零宽版心死循环或整段塞进一行。
#[test]
fn layout_zero_max_width_does_not_loop() {
    let f = font_regular();
    let text = "abcde";
    let l = layout_text(&f, text, &TextParams { font_size: 24.0, max_width: 0.0, ..Default::default() });
    // 至少每个字符成行（禁则可能并入，故 lines ≤ 字符数且 ≥ 1）。
    assert!(!l.lines.is_empty());
    assert!(l.lines.len() <= text.chars().count(), "零宽不应无限膨胀行数");
    assert_eq!(l.glyphs.len(), text.chars().count(), "字形不丢");
}

/// 负/NaN max_width：非有限退化为"不换行"；负值不应 panic。
/// 失败 => 非有限/负宽度导致越界或行分裂异常。
#[test]
fn layout_non_finite_and_negative_width() {
    let f = font_regular();
    for mw in [f32::NAN, f32::INFINITY, -1.0, -1e9] {
        let l = layout_text(&f, "测试文本ABC", &TextParams { font_size: 24.0, max_width: mw, ..Default::default() });
        assert_eq!(l.glyphs.len(), "测试文本ABC".chars().count(), "mw={mw} 字形丢失");
        assert!(l.width.is_finite(), "mw={mw} 产出非有限宽");
    }
}

/// 负 font_size：不应 panic（度量可为负，但过程需稳定）。
/// 失败 => 负字号造成 panic 或非有限坐标。
#[test]
fn layout_negative_font_size_no_panic() {
    let f = font_regular();
    let l = layout_text(&f, "AB中文", &TextParams { font_size: -24.0, ..Default::default() });
    assert_eq!(l.glyphs.len(), 4);
    for g in &l.glyphs {
        assert!(g.x.is_finite() && g.y.is_finite() && g.advance.is_finite(), "非有限坐标");
    }
}

/// 含未映射字符（emoji/私用区）：应回退 gid=0（.notdef）而非 panic。
/// 失败 => cmap 未命中路径越界。
#[test]
fn layout_unmapped_chars_fall_back_to_notdef() {
    let f = font_regular();
    let text = "A\u{E000}\u{1F600}B"; // 私用区 + emoji + ascii
    let l = layout_text(&f, text, &TextParams::default());
    assert_eq!(l.glyphs.len(), text.chars().count());
    // 私用区/emoji 大概率不在子集 → gid 0。
    assert_eq!(l.glyphs[1].gid, 0, "私用区应回退 .notdef");
    // ASCII 仍命中。
    assert!(l.glyphs[0].gid > 0 && l.glyphs[3].gid > 0);
}

/// 对齐确定性：Left/Center/Right 三次重复逐字形一致，且等文本同参不变。
/// 失败 => 对齐含非确定偏移。
#[test]
fn layout_alignment_deterministic_all_modes() {
    let f = font_regular();
    for align in [Align::Left, Align::Center, Align::Right] {
        let p = TextParams { font_size: 20.0, max_width: 150.0, align, ..Default::default() };
        let first = layout_text(&f, "对齐测试alignment", &p);
        for _ in 0..100 {
            assert_eq!(layout_text(&f, "对齐测试alignment", &p).glyphs, first.glyphs, "对齐非确定 {align:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// 竖排退化输入
// ---------------------------------------------------------------------------

/// 空文本竖排：空产、有限尺寸。
#[test]
fn vertical_empty_text() {
    let f = font_regular();
    let l = layout_vertical(&f, "", &VerticalParams::default());
    assert!(l.chars.is_empty());
    assert!(l.columns.is_empty());
    assert!(l.width.is_finite() && l.height.is_finite());
}

/// 竖排 max_height=0：每列至少放一字（防死循环），列数=字符数。
/// 失败 => 零列高导致无限循环或丢字。
#[test]
fn vertical_zero_max_height_one_char_per_column() {
    let f = font_regular();
    let text = "天地玄黄";
    let l = layout_vertical(&f, text, &VerticalParams { font_size: 24.0, max_height: 0.0, ..Default::default() });
    assert_eq!(l.chars.len(), text.chars().count(), "竖排不丢字");
    assert_eq!(l.columns.len(), text.chars().count(), "零列高应每字一列");
}

/// 竖排非有限 max_height = 不断列；重复确定。
/// 失败 => 非有限列高分支错误或非确定。
#[test]
fn vertical_infinite_max_height_single_column_and_deterministic() {
    let f = font_regular();
    let text = "中文竖排测试";
    let p = VerticalParams { font_size: 24.0, max_height: f32::INFINITY, ..Default::default() };
    let l = layout_vertical(&f, text, &p);
    assert_eq!(l.columns.len(), 1, "无限列高应单列");
    for _ in 0..100 {
        assert_eq!(layout_vertical(&f, text, &p).chars, l.chars, "竖排非确定");
    }
}

/// 转向表：CJK 直立、标点/ASCII 转向，且对非 ASCII BMP 稳定。
/// 失败 => 直立区间误判导致竖排标点朝向错误。
#[test]
fn vertical_rotation_classification() {
    assert!(!is_rotated_cw('中'), "CJK 直立");
    assert!(!is_rotated_cw('あ'), "假名直立");
    assert!(is_rotated_cw('。'), "句号转向");
    assert!(is_rotated_cw('A'), "ASCII 转向");
    assert!(is_rotated_cw('，'), "逗号转向");
    // 全角空格直立（0x3000 在直立集）。
    assert!(!is_rotated_cw('\u{3000}'));
}

// ---------------------------------------------------------------------------
// 光栅化 / 缓存
// ---------------------------------------------------------------------------

/// 光栅化：字号 0 → 空位图（不 panic）；负字号不 panic。
/// 失败 => 零/负字号产生越界或 panic。
#[test]
fn render_glyph_degenerate_sizes() {
    let f = font_regular();
    let gid = f.glyph_index('A').unwrap();
    let b0 = render_glyph(&f, gid, 0.0);
    assert!(b0.is_empty() || b0.alpha.iter().all(|&a| a == 0), "0 字号应空");
    // 负字号不 panic（结果可空）。
    let _ = render_glyph(&f, gid, -32.0);
}

/// 缓存键正确性：同 gid/尺寸不同 flags 不串（转向 vs 直立位图互换）。
/// 失败 => 缓存键漏 flags 字段导致转向/直立位图复用错误。
#[test]
fn glyph_cache_key_distinguishes_flags() {
    let f = font_regular();
    let gid = f.glyph_index('A').unwrap();
    let mut c = GlyphCache::new();
    let upright = c.get_or_render(&f, gid, 32.0, 0).clone();
    let rotated = c.get_or_render(&f, gid, 32.0, 1).clone();
    assert_ne!(upright, rotated, "flags 未纳入缓存键导致位图串用");
    assert!(c.contains(gid, 32.0, 0) && c.contains(gid, 32.0, 1));
    // 冷/热一致。
    assert_eq!(*c.get_or_render(&f, gid, 32.0, 0), upright, "命中与冷渲染不一致");
}

/// 缓存容量：不同尺寸共存，重复获取逐位一致、不误覆盖。
/// 失败 => 不同尺寸键碰撞。
#[test]
fn glyph_cache_multiple_sizes_isolated() {
    let f = font_regular();
    let gid = f.glyph_index('永').unwrap();
    let mut c = GlyphCache::new();
    let a = c.get_or_render(&f, gid, 16.0, 0).clone();
    let b = c.get_or_render(&f, gid, 48.0, 0).clone();
    assert_ne!(a.width, b.width, "不同尺寸应产生不同位图");
    assert_eq!(*c.get_or_render(&f, gid, 16.0, 0), a, "16px 缓存被污染");
    assert_eq!(*c.get_or_render(&f, gid, 48.0, 0), b, "48px 缓存被污染");
}

/// 端到端渲染确定性：更大画布、含竖排与多字符，重复逐像素一致。
/// 失败 => 渲染路径含非确定状态。
#[test]
fn draw_text_deterministic_via_cache() {
    let f = font_regular();
    let params = TextParams { font_size: 18.0, max_width: 120.0, ..Default::default() };
    let mut a = prism_io::Fbuf16F::new(160, 96);
    let mut b = prism_io::Fbuf16F::new(160, 96);
    let mut c1 = GlyphCache::new();
    let mut c2 = GlyphCache::new();
    prism_text::raster::draw_text(&mut a, &f, "确定性QA渲染测试ABCDEFG", &params, [0.2, 1.0, 0.3, 1.0], &mut c1);
    prism_text::raster::draw_text(&mut b, &f, "确定性QA渲染测试ABCDEFG", &params, [0.2, 1.0, 0.3, 1.0], &mut c2);
    assert_eq!(a.pixels, b.pixels, "端到端渲染非确定");
}
