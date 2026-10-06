//! prism-text 边缘攻击测试（跨模块集成；golden-file 与关键不变式）。
//!
//! 覆盖任务书测试清单：
//! - TTF 解析断言（内嵌资产表完整性、字形查找、复合度量）；
//! - 禁则断言（句号永不为行首——悬挂）；
//! - 竖排标点转向断言（转向标记 + 位图宽高互换）；
//! - 光栅化 golden-file（字形位图 fnv + 尺寸）；
//! - 布局 golden-file（全字形序列化文本比对）；
//! - 缓存命中前后逐字节一致；
//! - registry 静态汇入（D-27）。

use prism_text::cache::GlyphCache;
use prism_text::layout::{layout_text, TextParams};
use prism_text::raster::{render_glyph, GlyphBitmap};
use prism_text::vertical::{layout_vertical, VerticalParams};
use prism_text::{font_bold, font_regular, FONT_BOLD, FONT_REGULAR};

/// FNV-1a 32 位（自研哈希——测试内无外部依赖）。
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

// ---------------------------------------------------------------------------
// TTF 解析
// ---------------------------------------------------------------------------

#[test]
fn font_parse_table_invariants() {
    let f = font_regular();
    // head/hhea 度量自洽（Noto Sans SC：upem=1000）。
    assert_eq!(f.units_per_em(), 1000);
    assert!(f.ascender() > 0 && f.descender() < 0, "hhea 度量符号约定");
    assert!(f.num_glyphs() >= 6888, "子集应含 6888 码位（ASCII 95 + 标点 + GB2312 6763）");
    // cmap 查找：汉字命中非 0 gid。
    let zh = f.glyph_index('中').expect("中 在 GB2312 子集内");
    assert!(zh > 0);
    // hmtx：全角汉字 advance = 1000 upem（Noto Sans SC 全角约定）。
    assert_eq!(f.advance(zh), 1000, "全角 advance 应为 em 宽");
    // 空格：有度量无墨。
    let sp = f.glyph_index(' ').expect("空格在 ASCII 子集内");
    assert!(f.advance(sp) > 0);
    let sp_outline = f.outline(sp).expect("空格解析");
    assert!(sp_outline.contours.is_empty(), "空格无轮廓");
    // 汉字轮廓非空且点数充足。
    let zh_outline = f.outline(zh).expect("中 解析");
    assert!(!zh_outline.contours.is_empty());
    let n_points: usize = zh_outline.contours.iter().map(|c| c.len()).sum();
    assert!(n_points >= 8, "中 字轮廓点数应充足，实得 {n_points}");
    // ASCII 字形可解析。
    let a = f.glyph_index('A').expect("A 在子集内");
    assert!(!f.outline(a).expect("A 解析").contours.is_empty());
    // Bold 同构可解析。
    let b = font_bold();
    assert_eq!(b.units_per_em(), 1000);
    assert_eq!(b.advance(b.glyph_index('中').expect("Bold 中")), 1000);
    // 解析是纯函数：两次解析的轮廓逐点相等。
    let f2 = font_regular();
    assert_eq!(f.outline(zh).unwrap(), f2.outline(zh).unwrap());
}

#[test]
fn embedded_assets_are_real_files() {
    // 内嵌真实性：体积量级（1–3MB/档）+ sfnt 魔数 0x00010000。
    assert!(
        (1_000_000..3_500_000).contains(&FONT_REGULAR.len()),
        "Regular 体积应 1–3MB，实际 {} B",
        FONT_REGULAR.len()
    );
    assert!(
        (1_000_000..3_500_000).contains(&FONT_BOLD.len()),
        "Bold 体积应 1–3MB，实际 {} B",
        FONT_BOLD.len()
    );
    for bytes in [FONT_REGULAR, FONT_BOLD] {
        let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, 0x0001_0000, "sfnt TrueType 魔数");
    }
}

// ---------------------------------------------------------------------------
// 禁则
// ---------------------------------------------------------------------------

#[test]
fn period_never_starts_a_line_across_widths() {
    let f = font_regular();
    // 扫描多个版心宽度：任何断行点后行首字符都不得是行首禁则字符。
    for max_w in [48.0f32, 72.0, 96.0, 120.0, 144.0, 200.0] {
        let text = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。";
        let l = layout_text(&f, text, &TextParams { font_size: 24.0, max_width: max_w, ..Default::default() });
        for line in &l.lines {
            let first = l
                .glyphs
                .iter()
                .find(|g| g.line == l.lines.iter().position(|x| std::ptr::eq(x, line)).unwrap_or(0) && g.index == line.start);
            if let Some(g) = first {
                assert!(
                    !prism_text::layout::is_no_line_start(g.ch),
                    "max_width={max_w}：行首出现禁则字符「{}」",
                    g.ch
                );
            }
        }
    }
}

#[test]
fn open_bracket_never_ends_a_line() {
    let f = font_regular();
    for max_w in [48.0f32, 72.0, 96.0, 120.0, 200.0] {
        let text = "他说：「今日事，今日毕。」然后离去。";
        let l = layout_text(&f, text, &TextParams { font_size: 24.0, max_width: max_w, ..Default::default() });
        for line in &l.lines {
            if line.end > line.start {
                let last_idx = line.end - 1;
                if let Some(g) = l.glyphs.iter().find(|g| g.index == last_idx) {
                    assert!(
                        !prism_text::layout::is_no_line_end(g.ch),
                        "max_width={max_w}：行末出现禁则字符「{}」",
                        g.ch
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 竖排转向
// ---------------------------------------------------------------------------

#[test]
fn vertical_rotation_markers_and_bitmap_swap() {
    let f = font_regular();
    let l = layout_vertical(&f, "中文。A", &VerticalParams::default());
    // 标记：汉字直立、句号与 ASCII 转向。
    assert!(!l.chars[0].rotated && !l.chars[1].rotated);
    assert!(l.chars[2].rotated, "。转向");
    assert!(l.chars[3].rotated, "A 转向");
    // 位图：转向字符宽高互换且与直立旋转一致。
    let mut cache = GlyphCache::new();
    let upright = cache.get_or_render(&f, l.chars[1].gid, 24.0, 0).clone();
    let rotated = cache.get_or_render(&f, l.chars[1].gid, 24.0, 1).clone();
    assert_eq!(rotated.width, upright.height);
    assert_eq!(rotated.height, upright.width);
    assert_eq!(rotated, upright.rotate_cw(), "缓存转向与 rotate_cw 同构");
}

// ---------------------------------------------------------------------------
// golden-file
// ---------------------------------------------------------------------------

/// 字形位图 golden：`tests/golden/glyph_<ch>_<size>.txt`。
/// 文件内容：`w= h= left= top= fnv=`。
///
/// **生成器已移出测试**（W14 路3）：原先这里的 `regenerate_glyph_golden`
/// 是一条 `#[ignore]`，但它不是待修缺陷而是**写文件的生成器**——挂在
/// `--ignored` 列表里会让「待办」与「缺陷」混淆，且让全仓验收长期挂着
/// `ignored` 计数（与"注释掉测试"同类假绿）。现形态：
/// `cargo run -p prism-text --example regen_glyph_golden [-- --check|--write]`。
#[test]
fn glyph_rasterization_matches_golden() {
    let f = font_regular();
    let cases: [(char, f32); 4] = [('中', 32.0), ('文', 32.0), ('A', 24.0), ('g', 24.0)];
    for (ch, size) in cases {
        let gid = f.glyph_index(ch).unwrap_or(0);
        let bmp = render_glyph(&f, gid, size);
        let path = format!(
            "{}/tests/golden/glyph_{}_{}.txt",
            env!("CARGO_MANIFEST_DIR"),
            ch as u32,
            size as u32
        );
        let golden = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("缺少 golden 文件 {path}"));
        let actual = glyph_digest(&bmp);
        assert_eq!(actual, golden.trim(), "字形「{ch}」@{size}px 与 golden 不符");
    }
}

fn glyph_digest(bmp: &GlyphBitmap) -> String {
    format!(
        "w={} h={} left={} top={} fnv={:08x}",
        bmp.width,
        bmp.height,
        bmp.left,
        bmp.top,
        fnv1a(&bmp.alpha)
    )
}

/// 布局 golden：多场景全字形序列化的 fnv 指纹。
#[test]
fn layout_matches_golden() {
    let f = font_regular();
    let scenes: [(&str, TextParams); 3] = [
        (
            "Hello, 排版世界！",
            TextParams { font_size: 24.0, max_width: 120.0, ..Default::default() },
        ),
        (
            "春眠不觉晓，处处闻啼鸟。",
            TextParams {
                font_size: 24.0,
                max_width: 96.0,
                letter_spacing: 2.0,
                ..Default::default()
            },
        ),
        (
            "右对齐文本行",
            TextParams {
                font_size: 24.0,
                max_width: 200.0,
                align: prism_text::layout::Align::Right,
                ..Default::default()
            },
        ),
    ];
    let mut all = String::new();
    for (text, params) in scenes {
        let l = layout_text(&f, text, &params);
        all.push_str(&format!("text={text:?}\n"));
        all.push_str(&format!(
            "w={:.3} h={:.3} lines={}\n",
            l.width,
            l.height,
            l.lines.len()
        ));
        for line in &l.lines {
            all.push_str(&format!(
                "L[{}..{}) w={:.3} b={:.3}\n",
                line.start, line.end, line.width, line.baseline
            ));
        }
        for g in &l.glyphs {
            all.push_str(&format!(
                "G {} i={} x={:.3} y={:.3} gid={} adv={:.3}\n",
                g.ch, g.index, g.x, g.y, g.gid, g.advance
            ));
        }
        all.push('\n');
    }
    let path = format!("{}/tests/golden/layout_scenes.txt", env!("CARGO_MANIFEST_DIR"));
    match std::fs::read_to_string(&path) {
        Ok(golden) => assert_eq!(all, golden, "布局输出与 golden 不符"),
        Err(_) => {
            // 首次运行落盘（确定性输出即 golden）。
            std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).unwrap();
            std::fs::write(&path, &all).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// 缓存与端到端
// ---------------------------------------------------------------------------

#[test]
fn cache_cold_warm_byte_identical_and_end_to_end() {
    let f = font_regular();
    let gid = f.glyph_index('稳').expect("稳 在 GB2312 内");
    let mut cache = GlyphCache::new();
    let cold = cache.get_or_render(&f, gid, 40.0, 0).clone();
    let warm = cache.get_or_render(&f, gid, 40.0, 0);
    assert_eq!(*warm, cold, "命中前后逐字节一致");
    // 端到端：同一文本两次渲染进空帧，逐字节一致。
    let text = "端到端渲染一致性";
    let params = TextParams { font_size: 32.0, max_width: 200.0, ..Default::default() };
    let mut a = prism_io::Fbuf16F::new(200, 128);
    let mut b = prism_io::Fbuf16F::new(200, 128);
    let mut c1 = GlyphCache::new();
    let mut c2 = GlyphCache::new();
    prism_text::raster::draw_text(&mut a, &f, text, &params, [0.1, 0.2, 0.9, 1.0], &mut c1);
    prism_text::raster::draw_text(&mut b, &f, text, &params, [0.1, 0.2, 0.9, 1.0], &mut c2);
    assert_eq!(a.pixels, b.pixels, "端到端渲染确定性");
    assert!(a.pixels.iter().any(|&p| p != 0), "渲染应产出像素");
}

#[test]
fn registry_imports_text_and_raster_nodes() {
    // D-27：两 crate 的 registry! 汇入入口均可调，且按名解析。
    let mut reg = prism_core::NodeRegistry::new();
    prism_text::node::__prism_collect(&mut reg);
    prism_raster::nodes::__prism_collect(&mut reg);
    assert!(reg.contains(&prism_core::StructureId::custom(14, "prism_text")));
    assert!(reg.contains(&prism_core::StructureId::custom(13, "prism_raster")));
    assert!(reg.resolve_by_name("GlyphAdvance").is_some());
    assert!(reg.resolve_by_name("PolygonWinding").is_some());
}
