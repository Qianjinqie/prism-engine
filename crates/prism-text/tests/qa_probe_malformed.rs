//! QA 独立探针 —— N1「是否修到根上」的判定探针（畸形字体构造）。
//!
//! 此前 QA 的原话：「要修到根上」。判据不是"越界 gid 不panic"，
//! 而是：**畸形字体必须在 `parse` 期结构化报错（T004）或安全降级，
//! 绝不把非法状态泄漏到 `outline()` 的切片阶段才炸。**
//!
//! 构造五类畸形：
//! - M1 loca 表被截断（项数不足 numGlyphs+1）
//! - M2 `head.indexToLocFormat` 非法（2 / −1 / 0x7FFF）
//! - M3 loca 项声明的区间超出 glyf 表长
//! - M4 glyf 表长度小于 loca 末项声明
//! - M5 loca 项**逆序**（start > end）
//!
//! 每类都必须满足：要么 `Font::parse` 返回 `Err(T004)`，要么 parse 成功
//! 但 `outline()` 全gid 安全返回（空轮廓或 T00x），**绝不 panic**。

use prism_text::ttf::Font;

/// 独立大端写入helper（16/32 位）。
fn put16(d: &mut [u8], off: usize, v: u16) {
    d[off..off + 2].copy_from_slice(&v.to_be_bytes());
}
fn put32(d: &mut [u8], off: usize, v: u32) {
    d[off..off + 4].copy_from_slice(&v.to_be_bytes());
}
fn get16(d: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([d[off], d[off + 1]])
}
fn get32(d: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]])
}
fn table_off(d: &[u8], tag: &[u8; 4]) -> (usize, usize) {
    let n = get16(d, 4) as usize;
    (0..n)
        .map(|i| 12 + i * 16)
        .find(|&b| &d[b..b + 4] == tag)
        .map(|b| (get32(d, b + 8) as usize, get32(d, b + 12) as usize))
        .expect("表存在")
}
/// 表目录中某 tag 的目录项起始位置（用于改写 len/off 字段）。
fn table_dir_entry(d: &[u8], tag: &[u8; 4]) -> usize {
    let n = get16(d, 4) as usize;
    (0..n)
        .map(|i| 12 + i * 16)
        .find(|&b| &d[b..b + 4] == tag)
        .expect("表存在")
}

fn base() -> Vec<u8> {
    prism_text::FONT_REGULAR.to_vec()
}

/// 全量安全扫描：对一个（可能畸形的）字体，逐 gid 调`outline`，只允许
/// Ok(任意) / Err(T00x)，**panic 即为缺陷**。
fn scan_all_outlines(f: &Font, tag: &str) {
    let n = f.num_glyphs().min(6889);
    for gid in 0..n {
        let g = gid as u16;
        match f.outline(g) {
            Ok(_) => {}
            Err(e) => {
                assert!(
                    e.code.starts_with('T'),
                    "[{tag}] gid={g} 返回非 T 段错误码 {}",
                    e.code
                );
            }
        }
    }
    // 越界 gid 也必须安全
    for g in [n as u16, 7000, u16::MAX] {
        let _ = f.outline(g);
    }
}

/// M1：loca 表长度被篡改为不足 `numGlyphs+1` 项 → **parse 期T004**。
#[test]
fn m1_truncated_loca_rejected_at_parse() {
    let mut d = base();
    let itlf = get16(&d, table_off(&d, b"head").0 + 50);
    let entry = if itlf == 1 { 4usize } else { 2 };
    let (loca_off, loca_len) = table_off(&d, b"loca");
    let ng = get16(&d, table_off(&d, b"maxp").0 + 4) as usize;
    let need = (ng + 1) * entry;
    assert!(loca_len >= need, "前置条件：原字体 loca 应够用");
    // 把目录里的 loca 长度改小到只够一半项。
    let short = need / 2;
    let lde = table_dir_entry(&d, b"loca");
    put32(&mut d, lde + 12, short as u32);
    let e = Font::parse(&d).expect_err("loca 项数不足必须在 parse 期报错");
    assert_eq!(e.code, "T004", "应报 T004（loca 容量守卫），实得 {}", e.code);
    assert!(
        e.problem.contains("loca") || e.fix.contains("loca") || e.problem.contains("numGlyphs"),
        "错误信息应指向 loca/numGlyphs：{}",
        e.problem
    );
    let _ = loca_off;
}

/// M1b：loca 长度恰好等于 `numGlyphs+1` 项（边界）→ parse **成功**。
///
/// 这确保守卫用的是 `<`（严格小于）而非 `<=`，未把合法字体误杀。
#[test]
fn m1b_loca_exactly_sufficient_accepted() {
    let d = base();
    let f = Font::parse(&d).expect("真实字体必须能parse");
    let ng = f.num_glyphs() as usize;
    let itlf = get16(&d, table_off(&d, b"head").0 + 50);
    let entry = if itlf == 1 { 4usize } else { 2 };
    let (_, loca_len) = table_off(&d, b"loca");
    assert_eq!(
        loca_len,
        (ng + 1) * entry,
        "真实字体 loca 长度应恰好等于 (numGlyphs+1)*entry —— 守卫边界"
    );
    assert!(!f.outline(f.glyph_index('A').unwrap()).unwrap().contours.is_empty());
}

/// M2：`head.indexToLocFormat` 非法 → **parse 期 T004**。
#[test]
fn m2_illegal_index_to_loc_format_rejected_at_parse() {
    for &bad_format in &[2i16, -1, 3, 0x7FFF, i16::MIN] {
        let mut d = base();
        let head_off = table_off(&d, b"head").0;
        put16(&mut d, head_off + 50, bad_format as u16);
        match Font::parse(&d) {
            Err(e) => assert_eq!(
                e.code, "T004",
                "indexToLocFormat={bad_format} 应报 T004，实得 {}：{}",
                e.code, e.problem
            ),
            Ok(_) => panic!("indexToLocFormat={bad_format} 非 0/1，parse 竟成功（漏检）"),
        }
    }
    // 合法值 0 / 1 必须被接受（防止守卫过严误杀）
    for &ok_format in &[0i16, 1] {
        let mut d = base();
        let head_off = table_off(&d, b"head").0;
        put16(&mut d, head_off + 50, ok_format as u16);
        // 注意：把 format 改成 0 而不同步改写 loca 内容，loca 语义会变但
        // parse 只校验容量；这里只验"不被拒绝"。
        match Font::parse(&d) {
            Ok(_) => {}
            Err(e) => assert_ne!(
                e.code, "T004",
                "indexToLocFormat={ok_format} 是合法值，不应因格式被判 T004：{}",
                e.problem
            ),
        }
    }
}

/// M3：loca 项声明的区间**超出 glyf 表长** → parse 通过，outline 安全降级。
///
/// 这是 `glyf_slice` 双向钳制的真正考验：loca 是外部数据，start/end 可为
/// glyf 表外任意 u32。若实现回退到裸 `&data[glyf_off+start .. glyf_off+end]`
/// 则此处必 panic。
#[test]
fn m3_loca_beyond_glyf_table_degrades_safely() {
    let mut d = base();
    let (loca_off, _) = table_off(&d, b"loca");
    let (_, glyf_len) = table_off(&d, b"glyf");
    let itlf = get16(&d, table_off(&d, b"head").0 + 50);
    assert_eq!(itlf, 1, "内嵌字体为长格式 loca");
    // 把前若干 loca 项全部写成远超 glyf 表长的偏移。
    for i in 0..64usize {
        put32(&mut d, loca_off + i * 4, 0xFFFF_FF00u32.wrapping_add(i as u32));
    }
    let f = Font::parse(&d).expect("loca 容量足够，parse 应成功");
    scan_all_outlines(&f, "M3 超表 loca");
    // 抽查：这些字形应返回空轮廓或 T004，绝不 panic。
    for gid in 0..64u16 {
        match f.outline(gid) {
            Ok(o) => assert!(
                o.contours.is_empty(),
                "gid={gid} 的 loca 指向表外，应为空轮廓"
            ),
            Err(e) => assert_eq!(e.code, "T004", "只允许 T004，实得 {}", e.code),
        }
    }
    let _ = glyf_len;
}

/// M4：glyf 表长度小于 loca 末项声明 → outline 必须被钳制。
#[test]
fn m4_glyf_shorter_than_loca_declares_is_clamped() {
    let mut d = base();
    let (glyf_off, glyf_len) = table_off(&d, b"glyf");
    let (loca_off, _) = table_off(&d, b"loca");
    let ng = get16(&d, table_off(&d, b"maxp").0 + 4) as usize;
    // loca 末项（哨兵）声明 glyf_len；真实字体二者相等（自洽）。
    assert_eq!(get32(&d, loca_off + ng * 4), glyf_len as u32, "前置：真实字体自洽");
    // 现在把 glyf 表长度改小 → 哨兵项超出新glyf 表长。
    let gde = table_dir_entry(&d, b"glyf");
    put32(&mut d, gde + 12, (glyf_len / 2) as u32);
    let f = Font::parse(&d).expect("glyf 变短不触发 loca 容量守卫，parse 应成功");
    scan_all_outlines(&f, "M4 glyf 截短");
    let _ = glyf_off;
}

/// M5：loca 项**逆序**（start > end）→ 空轮廓，绝不 panic。
#[test]
fn m5_reversed_loca_entries_yield_empty_outline() {
    let mut d = base();
    let (loca_off, _) = table_off(&d, b"loca");
    // 让每一对 (2i, 2i+1) 的 start > end。
    for i in 0..128usize {
        let a = 1000u32 + (i as u32) * 8;
        let b = a - 4; // 逆序
        put32(&mut d, loca_off + (2 * i) * 4, a);
        put32(&mut d, loca_off + (2 * i + 1) * 4, b);
    }
    let f = Font::parse(&d).expect("parse 应成功");
    scan_all_outlines(&f, "M5 逆序 loca");
    for gid in (0..128u16).step_by(2) {
        let o = f.outline(gid).expect("逆序区间应 Ok 而非 Err");
        assert!(
            o.contours.is_empty(),
            "gid={gid} 的 loca 逆序（start>end），应为空轮廓而非逆序切片"
        );
    }
}

/// M6：maxp.numGlyphs 被篡改成极大值 → loca 容量守卫必须拦下。
///
/// 这是 N1 根因的**直接攻击**：旧实现用 u32 读 numGlyphs（≈4.5e8），
/// 守卫形同虚设；新实现读 u16，若有人再改回 u32，本测试立刻转红。
#[test]
fn m6_inflated_num_glyphs_is_rejected_not_sliced() {
    // 构造法：把 maxPoints（maxp+6..8）改成非零，使 u32 读法得到巨大值。
    let mut d = base();
    let maxp_off = table_off(&d, b"maxp").0;
    put16(&mut d, maxp_off + 6, 0xFFFF); // maxPoints = 65535
    // u32 读法此刻会得到 (6889<<16)|65535 —— 巨大值
    let inflated = get32(&d, maxp_off + 4);
    assert!(
        inflated > 6889 * 1000,
        "前置条件：应构造出被放大的 u32 读值，实得 {inflated}"
    );
    // 但 u16 读法（实现所用）仍得真值 6889 → parse 必须成功。
    let f = Font::parse(&d).expect("u16 读法下 numGlyphs 仍为 6889，应parse 成功");
    assert_eq!(f.num_glyphs(), 6889, "numGlyphs 不应被邻接字段污染");
    scan_all_outlines(&f, "M6 maxPoints 污染");
}

/// M7：numGlyphs 声明超过 loca 容量 → parse 期 T004（守卫的真检验）。
#[test]
fn m7_num_glyphs_exceeding_loca_capacity_rejected_at_parse() {
    let mut d = base();
    let maxp_off = table_off(&d, b"maxp").0;
    // 把 numGlyphs 改成 60000（loca 只够 6890 项）→ 必须 T004。
    put16(&mut d, maxp_off + 4, 60000);
    let e = Font::parse(&d).expect_err("numGlyphs 超出 loca 容量必须在 parse 期报错");
    assert_eq!(e.code, "T004", "应报 T004，实得 {}：{}", e.code, e.problem);
}

/// M8：整字节流截断（各种长度）→ parse 报 T001/T002，绝不 panic。
#[test]
fn m8_truncated_streams_rejected_cleanly() {
    let d = base();
    for &n in &[0usize, 1, 4, 11, 12, 13, 20, 40, 100, 1000, 5000, 20000, 43124] {
        match Font::parse(&d[..n]) {
            Err(e) => assert!(
                e.code == "T001" || e.code == "T002" || e.code == "T003" || e.code == "T004"
                    || e.code == "T007",
                "截断 {n}B 得到非预期错误码 {}",
                e.code
            ),
            Ok(f) => {
                // 极不可能；若真parse 成功也须能安全扫描。
                scan_all_outlines(&f, &format!("M8 截断{n}B"));
            }
        }
    }
}

/// M9：畸形字体的**渲染层**也不得 panic（render_glyph 走 outline）。
#[test]
fn m9_malformed_font_render_never_panics() {
    use prism_text::raster::render_glyph;
    let mut d = base();
    let (loca_off, _) = table_off(&d, b"loca");
    for i in 0..256usize {
        put32(&mut d, loca_off + i * 4, 0x7FFF_FF00u32.wrapping_add(i as u32));
    }
    let f = Font::parse(&d).expect("parse 应成功");
    for gid in 0..300u16 {
        for &px in &[12.0f32, 48.0, 1e10, f32::MAX] {
            let b = render_glyph(&f, gid, px);
            assert!(
                b.alpha.len() == b.width as usize * b.height as usize,
                "畸形字体渲染出不自洽位图（gid={gid} px={px}）"
            );
        }
    }
}
