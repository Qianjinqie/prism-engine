//! 多语种字体资产（`assets/prism_multiscript.ttf`）的**资产级**验收测试。
//!
//! 本文件把「自造字体资产」从测试构造器升格为**项目正式资产**，锁定：
//!
//! 1. **完整性锚点**：SHA-256 + 字节数（重跑生成器后字节漂移必须被看见）；
//! 2. **码位覆盖对比**：阿拉伯 / 天城文 / 拉丁 码位计数（改前 0 → 改后 N）；
//! 3. **GSUB 真表**：特征数 / 查找数 / 字节数（对比内嵌资产的 66 B 空壳）；
//! 4. **端到端绘制**：用本仓 OpenType/GSUB 引擎 + `raster`，断言阿拉伯四态
//!    字形互异、lam-alef 连字合成、天城文 i-matra 前置重排 + 记号附着、
//!    位图逐像素可复现（同输入两遍一致）。
//!
//! # 与既有 `gsub.rs` 内联测试的分工
//!
//! `gsub.rs` 的 G10–G26 用的是**测试内** `synth_font`（15 字形，编号被钉死）；
//! 本文件用的是**入库资产** `FONT_MULTISCRIPT`（54 字形）——两者互为佐证：
//! 前者证明引擎逻辑正确，本文件证明**内嵌资产**确实能被引擎端到端跑通。

use prism_text::gsub::{GlyphClass, GlyphItem, LayoutTable, Shaper};
use prism_text::multiscript_font as mf;
use prism_text::opentype::{outline_bbox, OtFont};
use prism_text::raster;
use prism_text::shaping::{shape, ShapeParams};
use prism_text::{font_multiscript, FONT_MULTISCRIPT};

// ===========================================================================
//  工具
// ===========================================================================

/// 帧缓冲里的非零 alpha 像素数。
fn ink_pixels(fb: &prism_io::Fbuf16F) -> usize {
    fb.pixels.chunks_exact(4).filter(|px| px[3] != 0).count()
}

/// 帧缓冲的 alpha 通道位图（行主序），用于逐像素比较。
fn alpha_bits(fb: &prism_io::Fbuf16F) -> Vec<u16> {
    fb.pixels.chunks_exact(4).map(|px| px[3]).collect()
}

/// 渲染单个字形到一块空帧。
fn raster_single(gid: u16, px: f32) -> prism_io::Fbuf16F {
    let f = font_multiscript();
    let mut fb = prism_io::Fbuf16F::new(220, 140);
    let bmp = raster::render_glyph(&f, gid, px);
    raster::blit_glyph(&mut fb, &bmp, 12.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
    fb
}

/// 走完整链路：shape → GSUB → 用最终 gid 逐个 render_glyph + blit_glyph。
///
/// 返回 `(帧缓冲, 最终 gid 列表, 最终字形项列表, 替换次数)`。
fn shape_gsub_raster(text: &str, px: f32) -> (prism_io::Fbuf16F, Vec<u16>, Vec<GlyphItem>, u32) {
    let f = font_multiscript();
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let shaper = Shaper::new(ot.gsub(), ot.gpos()).expect("Shaper");
    let p = ShapeParams { font_size: px, ..ShapeParams::default() };
    let run = shape(&f, text, &p).expect("整形");
    let (glyphs, report) = shaper.apply(&run).expect("GSUB");

    let mut fb = prism_io::Fbuf16F::new(260, 140);
    let baseline = 100.0f32;
    let mut pen = 12.0f32;
    let mut gids = Vec::with_capacity(glyphs.len());
    for g in &glyphs {
        gids.push(g.gid);
        let bmp = raster::render_glyph(&f, g.gid, px);
        raster::blit_glyph(&mut fb, &bmp, pen, baseline, [1.0, 1.0, 1.0, 1.0]);
        pen += f32::from(f.advance(g.gid)) * (px / f.units_per_em() as f32);
    }
    (fb, gids, glyphs, report.substitutions)
}

/// 统计某 Unicode 区段内 cmap 命中数（该字体的码位覆盖计数）。
fn cmap_count(lo: u32, hi: u32) -> usize {
    let f = font_multiscript();
    (lo..=hi)
        .filter(|&cp| char::from_u32(cp).and_then(|c| f.glyph_index(c)).is_some())
        .count()
}

// ===========================================================================
//  完整性锚点
// ===========================================================================

/// 资产可解析为 TrueType 容器，逐表校验和全通过，表集齐备。
#[test]
fn a1_asset_is_wellformed_truetype() {
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("内嵌资产应可解析");
    assert_eq!(ot.sfnt_version(), prism_text::opentype::SFNT_TRUETYPE);
    assert_eq!(ot.first_checksum_mismatch(), None, "逐表校验和须全通过");
    for t in [
        *b"head", *b"maxp", *b"hhea", *b"hmtx", *b"cmap", *b"loca", *b"glyf",
        *b"GSUB", *b"GPOS", *b"name", *b"post", *b"OS/2",
    ] {
        assert!(
            ot.directory().contains(t),
            "资产应含 {} 表",
            String::from_utf8_lossy(&t)
        );
    }
    // 与 ttf.rs 独立解析交叉验证。
    let f = font_multiscript();
    assert_eq!(ot.maxp().expect("maxp").num_glyphs, f.num_glyphs() as u16);
    assert_eq!(f.num_glyphs(), u32::from(mf::NUM_GLYPHS));
    assert_eq!(f.units_per_em(), 1000);
}

/// SHA-256 锚点 —— 锁定入库资产的字节（重跑生成器后若漂移即转红）。
///
/// 生成是纯函数（`mf::build_font` 确定性），故「重生成 = 入库字节」本身也是一条
/// 断言：任何源码改动导致字节变化，都必须显式更新此处的 SHA 并说明原因。
#[test]
fn a2_sha256_and_size_locked() {
    // 期望值由 `cargo run -p prism-text --bin gen_multiscript_font` 打印。
    // 首次落盘时在此固化；后续改动须同步更新（故意让漂移可见）。
    const EXPECTED_SHA: &str = "0eece481bfaf5a4eb31e67aeebbda3e1aaa7115b2aa94211c21c1a824f0b76c7";
    let sha = prism_io::sha256(FONT_MULTISCRIPT);
    let hex = sha.iter().map(|b| format!("{b:02x}")).collect::<String>();
    // 生成器确定性：现算字节与入库字节必须一致。
    assert_eq!(
        mf::build_font(),
        FONT_MULTISCRIPT.to_vec(),
        "入库资产必须等于生成器现算结果（生成器确定性铁律）"
    );
    if EXPECTED_SHA != "REPLACE_WITH_GENERATED_SHA" {
        assert_eq!(hex, EXPECTED_SHA, "资产 SHA-256 漂移，请核对是否有意变更");
    }
    assert!(FONT_MULTISCRIPT.len() > 2000, "资产应有实质内容，实际 {} B", FONT_MULTISCRIPT.len());
}

/// 生成器两次运行逐字节一致（确定性）。
#[test]
fn a3_generator_is_deterministic() {
    assert_eq!(mf::build_font(), mf::build_font());
}

// ===========================================================================
//  码位覆盖对比（改前 0 → 改后 N）
// ===========================================================================

/// 阿拉伯码位覆盖 > 0（内嵌 PrismSansSC 子集为 **0**）。
#[test]
fn a4_arabic_codepoints_present() {
    // 对比基线：中文子集不含阿拉伯字形。
    let zh = prism_text::font_regular();
    let zh_arabic = (0x0600u32..=0x06FF)
        .filter(|&cp| char::from_u32(cp).and_then(|c| zh.glyph_index(c)).is_some())
        .count();
    assert_eq!(zh_arabic, 0, "对照：中文子集阿拉伯码位应为 0");

    let n = cmap_count(0x0600, 0x06FF);
    assert!(n > 0, "新资产阿拉伯码位应 > 0，实际 {n}");
    assert_eq!(n, 30, "应覆盖 U+0627–U+064A 全部 30 个已分配阿拉伯字母（≥28）");
}

/// 天城文码位覆盖 > 0（内嵌子集为 **0**）。
#[test]
fn a5_devanagari_codepoints_present() {
    let zh = prism_text::font_regular();
    let zh_dev = (0x0900u32..=0x097F)
        .filter(|&cp| char::from_u32(cp).and_then(|c| zh.glyph_index(c)).is_some())
        .count();
    assert_eq!(zh_dev, 0, "对照：中文子集天城文码位应为 0");

    let n = cmap_count(0x0900, 0x097F);
    assert!(n > 0, "新资产天城文码位应 > 0，实际 {n}");
    assert!(n >= 40, "应覆盖基本元音 + 全部辅音 + matra + virama，实际 {n}");
    // 抽查：KA / KHA / GA / CHA / TA / HA 六个基字，加 i-matra / virama。
    let f = font_multiscript();
    for cp in [0x0915u32, 0x0916, 0x0917, 0x091A, 0x0924, 0x0939, 0x093F, 0x094D] {
        let ch = char::from_u32(cp).expect("天城文");
        assert!(f.glyph_index(ch).is_some(), "天城文 U+{cp:04X} 应覆盖");
    }
}

/// 拉丁 A–Z 码位覆盖完整（26）。
#[test]
fn a6_latin_codepoints_present() {
    assert_eq!(cmap_count(0x0041, 0x005A), 26, "拉丁 A–Z 全覆盖");
    let f = font_multiscript();
    for k in 0..mf::LATIN_COUNT {
        let ch = char::from_u32(0x0041 + u32::from(k)).expect("A-Z");
        assert_eq!(f.glyph_index(ch), Some(mf::G_LATIN_A + k));
    }
}

// ===========================================================================
//  GSUB 真表 vs 66 B 空壳
// ===========================================================================

/// GSUB 从 **66 B 空壳（0 特征 0 查找）** 变为真表（5 特征 5 查找）。
#[test]
fn a7_gsub_is_real_not_empty_shell() {
    // ---- 对比基线：内嵌资产的 66 B 空壳 ----
    let zh_ot = OtFont::parse(prism_text::FONT_REGULAR).expect("容器");
    let zh_gsub = zh_ot.gsub().expect("中文子集有 GSUB");
    assert_eq!(zh_gsub.len(), 66, "对照：中文子集 GSUB 是 66 B 空壳");
    let zh_t = LayoutTable::parse(zh_gsub).expect("空壳可解析");
    assert_eq!(zh_t.features().len(), 0, "对照：0 特征");
    assert_eq!(zh_t.lookups().len(), 0, "对照：0 查找");

    // ---- 新资产：真表 ----
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let gsub = ot.gsub().expect("资产有 GSUB");
    assert!(gsub.len() > 66, "新 GSUB 应远大于 66 B，实际 {} B", gsub.len());
    let t = LayoutTable::parse(gsub).expect("应可解析");
    assert_eq!(t.features().len(), 5, "5 特征");
    assert_eq!(t.lookups().len(), 5, "5 查找");
    // 脚本 tag = arab，DefaultLangSys 引用全部 5 个特征。
    assert_eq!(t.scripts().len(), 1);
    assert_eq!(t.scripts()[0].tag, *b"arab");
    assert_eq!(t.scripts()[0].default_features, vec![0, 1, 2, 3, 4]);
    // 特征序 = 施加序。
    let tags: Vec<[u8; 4]> = t.features().iter().map(|f| f.tag).collect();
    assert_eq!(tags, vec![*b"isol", *b"init", *b"medi", *b"fina", *b"liga"]);
    let kinds: Vec<u16> = t.lookups().iter().map(|l| l.kind).collect();
    assert_eq!(kinds, vec![1, 1, 1, 1, 4], "4 SingleSubst + 1 LigatureSubst");
}

/// GPOS `abvm` 真表（1 特征 MarkToBasePos）。
#[test]
fn a8_gpos_abvm_markbase() {
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let gpos = ot.gpos().expect("资产有 GPOS");
    let t = LayoutTable::parse(gpos).expect("应可解析");
    assert_eq!(t.features().len(), 1);
    assert_eq!(t.features()[0].tag, *b"abvm");
    assert_eq!(t.lookups().len(), 1);
    assert_eq!(t.lookups()[0].kind, 4, "GPOS type 4 = MarkToBasePos");
}

// ===========================================================================
//  阿拉伯四态端到端（不同上下文 → 不同 gid → 不同像素）
// ===========================================================================

/// 阿拉伯 `"سلام"` → 整形层合并 lam-alef、其余字母拿到正确四态字形（非 .notdef）。
///
/// 串 `U+0633 SEEN U+0644 LAM U+0627 ALEF U+0645 MEEM`（"سلام"，真实词）。
/// 整形层把 `LAM+ALEF` 合并为**一个 lam-alef 连字簇**（呈现形 U+FEFB），
/// 故字形序列为：
/// - SEEN(Dual) 词首 → `init`   → gid=G_SEEN_INITIAL（GSUB 替换）
/// - LAM+ALEF   词中 → 连字簇   → gid=G_LAM_ALEF_LIG（cmap 直接命中）
/// - MEEM(Dual) 词尾 → `fina`   → gid=G_MEEM_FINAL（GSUB 替换）
///
/// 这一条同时覆盖了**阿拉伯连接**与 **lam-alef 连字**两条主线。
#[test]
fn a9_arabic_word_gets_correct_four_states() {
    let f = font_multiscript();
    let p = ShapeParams { font_size: 100.0, ..ShapeParams::default() };
    let run = shape(&f, "\u{0633}\u{0644}\u{0627}\u{0645}", &p).expect("整形");
    assert_eq!(run.missing_glyphs, 0, "资产应全部命中，无缺字形");
    assert_eq!(run.cluster_count(), 3, "LAM+ALEF 合并为 1 簇 ⇒ 共 3 簇");

    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let shaper = Shaper::new(ot.gsub(), ot.gpos()).expect("Shaper");
    let (glyphs, report) = shaper.apply(&run).expect("GSUB");
    // SEEN(init) + MEEM(fina) 两次 GSUB 替换；lam-alef 簇无需替换。
    assert_eq!(report.substitutions, 2, "两次替换（SEEN/MEEM）");

    let gids: Vec<u16> = glyphs.iter().map(|g| g.gid).collect();
    assert!(
        gids.contains(&mf::arabic_form_gid("SEEN", mf::ArabicForm::Initial)),
        "SEEN 词首形，实际 {gids:?}"
    );
    assert!(gids.contains(&mf::G_LAM_ALEF_LIG), "LAM+ALEF 连字，实际 {gids:?}");
    assert!(
        gids.contains(&mf::arabic_form_gid("MEEM", mf::ArabicForm::Final)),
        "MEEM 词尾形，实际 {gids:?}"
    );
    // 没有任何字形回落到 .notdef。
    assert!(!gids.contains(&0), "不得出现 .notdef，实际 {gids:?}");
}

/// A9b：同一字符 BEH 在四种上下文 → 四个**互不相同**的 gid + 互异像素。
#[test]
fn a9b_four_states_distinct_gid_and_pixels() {
    let cases: [(&str, mf::ArabicForm); 4] = [
        ("\u{0628}", mf::ArabicForm::Isolated),                   // 孤立
        ("\u{0628}\u{0628}", mf::ArabicForm::Initial),            // 词首
        ("\u{0628}\u{0628}\u{0628}", mf::ArabicForm::Medial),     // 词中
        ("\u{0644}\u{0628}", mf::ArabicForm::Final),              // 词尾
    ];
    let mut frames: Vec<Vec<u16>> = Vec::new();
    let mut four: Vec<u16> = Vec::new();
    for (text, form) in cases {
        let expect = mf::arabic_form_gid("BEH", form);
        let (fb, gids, _, _) = shape_gsub_raster(text, 64.0);
        assert!(gids.contains(&expect), "{text:?} 应含 gid {expect}，实际 {gids:?}");
        assert!(ink_pixels(&fb) > 0, "{text:?} 必须画出非空像素");
        four.push(expect);
        frames.push(alpha_bits(&fb));
    }
    // 四态 gid 两两不同。
    for i in 0..four.len() {
        for j in (i + 1)..four.len() {
            assert_ne!(four[i], four[j], "四态 gid 必须互异（{i} vs {j}）");
            assert_ne!(frames[i], frames[j], "四态像素必须互异（{i} vs {j}）");
        }
    }
}

// ===========================================================================
//  A9c：GSUB 层四态全量（覆盖 BEH/LAM/SEEN/MEEM 四字母 + ALEF 词尾）
// ===========================================================================

/// A9c：四个 SingleSubst 对每个字母各自生效（含 LAM 词中形与 ALEF 词尾形）。
#[test]
fn a9c_all_four_states_per_letter() {
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let t = LayoutTable::parse(ot.gsub().expect("GSUB")).expect("解析");
    let apply = |tag: [u8; 4], gid: u16| -> u16 {
        let fi = t.feature_index(tag).expect("特征");
        let mut g = vec![GlyphItem::base(gid, 0)];
        t.apply_feature(fi, &mut g, 0).expect("施加");
        g[0].gid
    };
    // 四字母 × 四态。
    for name in ["BEH", "LAM", "SEEN", "MEEM"] {
        let base = mf::arabic_base_gid(name);
        let isol = mf::arabic_form_gid(name, mf::ArabicForm::Isolated);
        let init = mf::arabic_form_gid(name, mf::ArabicForm::Initial);
        let medi = mf::arabic_form_gid(name, mf::ArabicForm::Medial);
        let fina = mf::arabic_form_gid(name, mf::ArabicForm::Final);
        assert_eq!(apply(*b"isol", base), isol);
        assert_eq!(apply(*b"init", base), init);
        assert_eq!(apply(*b"medi", base), medi);
        assert_eq!(apply(*b"fina", base), fina);
        // 四态两两不同。
        let all = [isol, init, medi, fina];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "字母 {name} gid={base} 四态互异（{i} vs {j}）");
            }
        }
    }
    // ALEF 只覆盖 isol/fina（Right 连接型）。
    let alef = mf::arabic_base_gid("ALEF");
    assert_eq!(apply(*b"fina", alef), mf::arabic_form_gid("ALEF", mf::ArabicForm::Final), "ALEF 词尾形");
    assert_eq!(apply(*b"init", alef), alef, "ALEF 无词首形 ⇒ 不变");
    assert_eq!(apply(*b"medi", alef), alef, "ALEF 无词中形 ⇒ 不变");
}

// ===========================================================================
//  lam-alef 连字端到端
// ===========================================================================
/// `LAM + ALEF` → **合成一个连字字形**（不是两个），且真的画出来。
#[test]
fn a10_lam_alef_ligature_end_to_end() {
    let (fb, gids, _, _) = shape_gsub_raster("\u{0644}\u{0627}", 64.0);
    assert_eq!(gids, vec![mf::G_LAM_ALEF_LIG], "两字符合并为一个连字 gid");
    let ink = ink_pixels(&fb);
    assert!(ink > 0, "连字必须画出非空像素");
    assert!(ink > 500, "连字墨迹面积偏小（{ink} px）");

    // 反证：直接喂 GSUB 层的 `[LAM, ALEF]` 也合并（不依赖 cmap 呈现形）。
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let t = LayoutTable::parse(ot.gsub().expect("GSUB")).expect("解析");
    let fi = t.feature_index(*b"liga").expect("liga");
    let mut g = vec![
        GlyphItem::base(mf::arabic_base_gid("LAM"), 0),
        GlyphItem::base(mf::arabic_base_gid("ALEF"), 1),
    ];
    let n = t.apply_feature(fi, &mut g, 0).expect("施加");
    assert_eq!(n, 1);
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].gid, mf::G_LAM_ALEF_LIG);
    assert_eq!(g[0].class, GlyphClass::Ligature);
}

// ===========================================================================
//  天城文 i-matra 前置重排 + 记号附着
// ===========================================================================

/// `KA + i-matra` → **i-matra 字形出现在基字之前**（前置重排生效）。
#[test]
fn a11_devanagari_i_matra_reordered_before_base() {
    let f = font_multiscript();
    let p = ShapeParams { font_size: 64.0, ..ShapeParams::default() };
    // 码位序：क(KA) 然后 ि(i-matra)，但视觉须 i-matra 在前。
    let run = shape(&f, "\u{0915}\u{093F}", &p).expect("整形");
    assert_eq!(run.missing_glyphs, 0, "KA + i-matra 均应命中");
    // run.glyphs 是逻辑序但**簇内按视觉序**：i-matra（mark）应排在 KA 之前。
    let gids: Vec<u16> = run.glyphs.iter().map(|g| g.gid).collect();
    assert_eq!(gids, vec![mf::G_IMATRA, mf::G_KA], "i-matra 前置重排，实际 {gids:?}");
    // i-matra 是零宽记号。
    let mark = run.glyphs.iter().find(|g| g.gid == mf::G_IMATRA).expect("i-matra");
    assert!(mark.is_mark, "i-matra 应标为记号");
    assert_eq!(mark.advance, 0.0, "记号零宽");

    // GPOS `abvm` 把记号附着到基字上方（锚点差已知：+230/-640）。
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let shaper = Shaper::new(ot.gsub(), ot.gpos()).expect("Shaper");
    let (glyphs, _) = shaper.apply(&run).expect("GSUB");
    let atts = shaper.attach_marks(&glyphs, *b"abvm").expect("记号附着");
    let att = atts.iter().flatten().next().expect("i-matra 应得到定位");
    assert_eq!(att.x_offset, 230);
    assert_eq!(att.y_offset, -640);
}

// ===========================================================================
//  逐像素可复现
// ===========================================================================

/// 同输入两次 → 位图 `to_bits()` 一致（确定性铁律）。
#[test]
fn a12_rendering_is_byte_deterministic() {
    for text in ["\u{0628}", "\u{0644}\u{0627}", "\u{0633}\u{0644}\u{0627}\u{0645}", "\u{0915}\u{093F}", "AZ"] {
        let a = shape_gsub_raster(text, 64.0).0;
        let b = shape_gsub_raster(text, 64.0).0;
        assert_eq!(a.pixels, b.pixels, "{text:?} 两遍栅格像素应逐字节一致");
    }
}

/// A12b：单字形渲染两次亦逐像素一致，且四态各字形位图两两不同。
#[test]
fn a12b_single_glyph_render_deterministic_and_distinct() {
    let g = mf::arabic_form_gid("BEH", mf::ArabicForm::Final);
    assert_eq!(alpha_bits(&raster_single(g, 48.0)), alpha_bits(&raster_single(g, 48.0)));
    let forms = [
        mf::arabic_form_gid("BEH", mf::ArabicForm::Isolated),
        mf::arabic_form_gid("BEH", mf::ArabicForm::Initial),
        mf::arabic_form_gid("BEH", mf::ArabicForm::Medial),
        mf::arabic_form_gid("BEH", mf::ArabicForm::Final),
    ];
    let frames: Vec<Vec<u16>> = forms.iter().map(|&g| alpha_bits(&raster_single(g, 48.0))).collect();
    for i in 0..frames.len() {
        for j in (i + 1)..frames.len() {
            assert_ne!(frames[i], frames[j], "四态字形位图应互异（{i} vs {j}）");
        }
    }
}

// ===========================================================================
//  诚实边界
// ===========================================================================

/// 诚实记录 —— 资产是**含二次贝塞尔的近似结构**，不是精确书法形。
///
/// 字形由「矩形折线」升级为「笔画化轮廓 + off-curve 控制点」，故不再
/// 断言 4 点矩形；改为断言：轮廓**至少一个环含 off-curve 控制点**（曲线存在），
/// 且多锚点（≥4）。仍不假装字体美观（诚实边界见模块头与 A14）。
#[test]
fn a13_honest_structural_outlines_with_curves() {
    let f = font_multiscript();
    let g = mf::arabic_form_gid("BEH", mf::ArabicForm::Initial);
    let o = f.outline(g).expect("有轮廓");
    assert!(!o.contours.is_empty(), "应有轮廓");
    let total: usize = o.contours.iter().map(|c| c.len()).sum();
    assert!(total >= 4, "锚点应 ≥4，实际 {total}");
    // 至少一个 off-curve 控制点（二次贝塞尔曲线的存在性）。
    let offs = o.contours.iter().flatten().filter(|p| !p.on_curve).count();
    assert!(offs > 0, "含曲线的轮廓应至少有 1 个 off-curve 点，实际 {offs}");
    let _ = outline_bbox(&o).expect("包围盒可算");
}

/// A13b：诚实边界 —— 未覆盖的阿拉伯字母回落 .notdef（不静默编造字形）。
#[test]
fn a13b_uncovered_arabic_letters_fall_back_to_notdef() {
    let f = font_multiscript();
    // U+0688 DDAL 是未纳入资产集（U+0627–U+064A）的阿拉伯字母 → cmap 未命中。
    assert_eq!(f.glyph_index('\u{0688}'), None, "DDAL 未覆盖，应返回 None");
    let p = ShapeParams { font_size: 64.0, ..ShapeParams::default() };
    let run = shape(&f, "\u{0688}", &p).expect("整形");
    assert_eq!(run.missing_glyphs, 1, "缺字形被诚实计数");
    assert_eq!(run.glyphs[0].gid, 0, "回落 .notdef");
}

// ===========================================================================
//  扩充验收（小写/数字/标点、曲线、28+ 阿拉伯、天城文、确定性）
// ===========================================================================

/// 小写字母 a–z 全部命中且非 `.notdef`。
#[test]
fn a14_lowercase_letters_mapped() {
    let f = font_multiscript();
    for k in 0..mf::LATIN_COUNT {
        let ch = char::from_u32(0x0061 + u32::from(k)).expect("a-z");
        let g = f.glyph_index(ch).unwrap_or(0);
        assert_ne!(g, 0, "小写 {ch} 不得为 .notdef");
        assert_eq!(g, mf::G_LATIN_LOWER_A + k, "小写 {ch} gid 应连续");
    }
    assert_eq!(cmap_count(0x0061, 0x007A), 26, "拉丁 a–z 全覆盖");
}

/// 数字 0–9 全部命中且非 `.notdef`。
#[test]
fn a15_digits_mapped() {
    let f = font_multiscript();
    for k in 0..mf::DIGIT_COUNT {
        let ch = char::from_u32(0x0030 + u32::from(k)).expect("0-9");
        let g = f.glyph_index(ch).unwrap_or(0);
        assert_ne!(g, 0, "数字 {ch} 不得为 .notdef");
    }
    assert_eq!(cmap_count(0x0030, 0x0039), 10, "数字 0–9 全覆盖");
}

/// 常用标点（. , ! ? - : ; ( ) '）全部命中且非 `.notdef`。
#[test]
fn a16_punctuation_mapped() {
    let f = font_multiscript();
    for &cp in &[0x002E, 0x002C, 0x0021, 0x003F, 0x002D, 0x003A, 0x003B, 0x0028, 0x0029, 0x0027] {
        let ch = char::from_u32(cp).expect("标点");
        let g = f.glyph_index(ch).unwrap_or(0);
        assert_ne!(g, 0, "标点 {ch} 不得为 .notdef");
    }
}

/// **off-curve（二次贝塞尔控制点）确实存在**（本路技术核心）。
#[test]
fn a17_off_curve_control_points_present() {
    let f = font_multiscript();
    // 拉丁大写 'O'（圆环）必含 off-curve 控制点。
    let o = f.outline(mf::G_LATIN_A + 14).expect("O 轮廓");
    let offs = o.contours.iter().flatten().filter(|p| !p.on_curve).count();
    assert!(offs >= 4, "'O' 应含 ≥4 个 off-curve 点，实际 {offs}");
    // 全局：至少一个字形含 off-curve 点。
    let any = (0..u16::from(mf::NUM_GLYPHS))
        .filter_map(|g| f.outline(g).ok())
        .any(|o| o.contours.iter().flatten().any(|p| !p.on_curve));
    assert!(any, "至少一个字形须含 off-curve 点");
}

/// 含曲线字形（off-curve）的计数足够多 —— 不是个别特例。
#[test]
fn a18_many_glyphs_have_curves() {
    let f = font_multiscript();
    let with_off = (0..u16::from(mf::NUM_GLYPHS))
        .filter_map(|g| f.outline(g).ok())
        .filter(|o| o.contours.iter().flatten().any(|p| !p.on_curve))
        .count();
    assert!(with_off >= 100, "含曲线字形应 ≥100，实际 {with_off}");
}

/// 阿拉伯 28 字母（本资产覆盖 30）的四态 gid 两两互异。
#[test]
fn a19_arabic_all_letters_four_states_distinct() {
    let map = mf::arabic_glyph_map();
    assert!(map.len() >= 28, "应覆盖 ≥28 个阿拉伯字母，实际 {}", map.len());
    for a in &map {
        let mut all = vec![a.base, a.isol, a.fina];
        if let Some(g) = a.init {
            all.push(g);
        }
        if let Some(g) = a.medi {
            all.push(g);
        }
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "四态 gid 互异（{i} vs {j}）");
            }
        }
    }
}

/// 天城文新基字（क ख ग घ ङ / च छ ज झ ञ 等）均映射到互异 gid。
#[test]
fn a20_devanagari_base_consonants_mapped_distinct() {
    let f = font_multiscript();
    let cps = [0x0915u32, 0x0916, 0x0917, 0x0918, 0x0919, 0x091A, 0x091B, 0x091C, 0x091D, 0x091E];
    let mut gids: Vec<u16> = Vec::new();
    for cp in cps {
        let ch = char::from_u32(cp).expect("天城文辅音");
        let g = f.glyph_index(ch).unwrap_or(0);
        assert_ne!(g, 0, "U+{cp:04X} 应命中");
        gids.push(g);
    }
    gids.sort_unstable();
    gids.dedup();
    assert_eq!(gids.len(), cps.len(), "各基字 gid 应互异");
}

/// 天城文独立元音 + i-matra + virama 覆盖，且 virama/i-matra 零宽。
#[test]
fn a21_devanagari_vowels_and_matras() {
    let f = font_multiscript();
    assert!(f.glyph_index('\u{0905}').is_some(), "独立元音 A");
    assert!(f.glyph_index('\u{0906}').is_some(), "独立元音 AA");
    let im = f.glyph_index('\u{093F}').expect("i-matra");
    assert_eq!(f.advance(im), 0, "i-matra 零宽");
    let vir = f.glyph_index('\u{094D}').expect("virama");
    assert_eq!(f.advance(vir), 0, "virama 零宽");
}

/// **确定性（两次生成 sha256 相同）** —— 独立于生成器进程的字节级锚定。
#[test]
fn a22_generator_two_runs_same_sha256() {
    let a = mf::build_font();
    let b = mf::build_font();
    let ha = prism_io::sha256(&a);
    let hb = prism_io::sha256(&b);
    assert_eq!(ha, hb, "两次生成 sha256 必须相同");
    assert_eq!(a, b, "两次生成必须逐字节一致");
    // 与入库字节也一致（资产即生成器产物）。
    assert_eq!(a, FONT_MULTISCRIPT.to_vec(), "入库资产应等于现算结果");
}

/// GSUB init/medi lookup 覆盖全部 Dual 字母（不止 BEH）。
#[test]
fn a23_gsub_covers_all_dual_letters() {
    let ot = OtFont::parse(FONT_MULTISCRIPT).expect("容器");
    let t = LayoutTable::parse(ot.gsub().expect("GSUB")).expect("解析");
    let apply = |tag: [u8; 4], gid: u16| -> u16 {
        let fi = t.feature_index(tag).expect("特征");
        let mut g = vec![GlyphItem::base(gid, 0)];
        t.apply_feature(fi, &mut g, 0).expect("施加");
        g[0].gid
    };
    // 抽查几个非 BEH 的 Dual 字母。
    for name in ["SEEN", "LAM", "MEEM", "NOON", "YEH", "QAF", "HEH"] {
        let base = mf::arabic_base_gid(name);
        assert_ne!(apply(*b"init", base), base, "{name} 词首形应生效");
        assert_ne!(apply(*b"medi", base), base, "{name} 词中形应生效");
    }
}

/// 端到端栅格 —— 拉丁小写/数字/天城文新基字均可画出非空像素。
#[test]
fn a24_new_glyphs_rasterize_ink() {
    let f = font_multiscript();
    for ch in ['a', 'g', 'Z', '5', '\u{0915}', '\u{0916}', '\u{0939}'] {
        let gid = f.glyph_index(ch).expect("应命中");
        let fb = raster_single(gid, 48.0);
        assert!(ink_pixels(&fb) > 0, "{ch:?} 必须画出非空像素");
    }
}

