//! QA 独立探针 —— 缺陷 N1（`maxp.numGlyphs` 误用 u32 读取）终验。
//!
//! 与自报修复解耦：不复用其 `#[cfg(test)]` 断言，全部以本文件独立重算。
//! 断言分三层：
//! 1. **真值层**：直接从字节流按 TrueType 规范独立解表，读出 numGlyphs 地面真值，
//!    再与 `Font::parse` 的结果交叉比对（而非硬编码 6889 后自证）。
//! 2. **证伪层**：复现旧错误读法 `be::u32(maxp+4)`，证明其≠ 真值且数量级放大。
//! 3. **穷举层**：越界 gid 全枚举，必须返回空轮廓且绝不 panic。

use prism_text::font_regular;

/// 独立的大端 u16 读取（不借用被测代码的 `be` 私有模块）。
fn be16(d: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(off)?, *d.get(off + 1)?]))
}

/// 独立的大端 u32 读取。
fn be32(d: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *d.get(off)?,
        *d.get(off + 1)?,
        *d.get(off + 2)?,
        *d.get(off + 3)?,
    ]))
}

/// 独立解表目录 → (tag → (offset, length))。
fn table_dir(d: &[u8]) -> Vec<([u8; 4], usize, usize)> {
    let n = be16(d, 4).expect("sfnt 表数可读") as usize;
    (0..n)
        .filter_map(|i| {
            let b = 12 + i * 16;
            let tag: [u8; 4] = d.get(b..b + 4)?.try_into().ok()?;
            Some((tag, be32(d, b + 8)? as usize, be32(d, b + 12)? as usize))
        })
        .collect()
}

fn find_table(d: &[u8], tag: &[u8; 4]) -> (usize, usize) {
    table_dir(d)
        .into_iter()
        .find(|(t, _, _)| t == tag)
        .map(|(_, o, l)| (o, l))
        .unwrap_or_else(|| panic!("字体缺少表 {}", std::str::from_utf8(tag).unwrap()))
}

/// P1 真值层：`num_glyphs` 必须等于按规范独立解出的地面真值。
///
/// 失败含义：解析仍把 numGlyphs 与邻接字段拼接（真值 6889 被读成 4.5e8），
/// `outline` 的越界守卫随之失效→ 越界 gid 直落 loca 切片 panic（宪章 C18）。
#[test]
fn p1_num_glyphs_matches_indepently_parsed_ground_truth() {
    let data = prism_text::FONT_REGULAR;
    let (maxp_off, maxp_len) = find_table(data, b"maxp");
    // TrueType §maxp：version(4B) + numGlyphs(**u16**) + maxPoints(u16) ...
    let num_glyphs_u16 = be16(data, maxp_off + 4).expect("maxp+4 可读");
    let max_points_u16 = be16(data, maxp_off + 6).expect("maxp+6 可读");
    // 地面真值 6889 —— 同时要求 maxp 表长足够容纳version+numGlyphs。
    assert!(maxp_len >= 6, "maxp 表长 {} 不足 6B", maxp_len);
    assert_eq!(num_glyphs_u16, 6889, "内嵌字体的 numGlyphs 地面真值应为 6889");
    assert_eq!(max_points_u16, 247, "紧邻的 maxPoints 用于交叉校验读位");

    let f = font_regular();
    assert_eq!(
        f.num_glyphs(),
        u32::from(num_glyphs_u16),
        "Font::parse 的 num_glyphs 与独立解表结果不一致"
    );

    // 边界：最后一个合法 gid 越界一位即非法（守卫必须用`>=` 而非 `>`）。
    let last_legal = num_glyphs_u16 - 1;
    assert!(
        f.outline(last_legal).is_ok(),
        "最后一个合法 gid={last_legal} 应可解析"
    );
    assert!(
        f.outline(num_glyphs_u16).expect("越界 gid 应 Ok").contours.is_empty(),
        "gid={} (=numGlyphs) 必须越界返回空轮廓",
        num_glyphs_u16
    );
}

/// P2 证伪层：旧读法（`be::u32(maxp+4)`）必须被证伪，且放大 6.5 万倍。
///
/// 这层确保测试**不是恒真**：若有人把 num_glyphs 又改回 u32 读法，
/// P1 会红；若有人把真值改成"与实现一致"的错误值，本层的独立解表会红。
#[test]
fn p2_old_u32_read_is_falsified_and_inflated() {
    let data = prism_text::FONT_REGULAR;
    let (maxp_off, _) = find_table(data, b"maxp");
    let old_read = be32(data, maxp_off + 4).expect("maxp+4 的 4B 可读");
    let truth = be16(data, maxp_off + 4).expect("maxp+4 的 2B 可读");

    // 旧读法：把 numGlyphs 与紧邻 maxPoints 一起吃成 32 位 → 4.5e8。
    assert_eq!(
        old_read,
        (u32::from(truth) << 16) | u32::from(be16(data, maxp_off + 6).unwrap()),
        "旧读法应等于 (numGlyphs<<16)|maxPoints"
    );
    assert_ne!(old_read, u32::from(truth), "旧读法竟与真值相同，本探针失去证伪力");
    assert_eq!(old_read, 451_477_751, "旧读法的具体错误值");
    assert!(
        old_read > u32::from(truth) * 60_000,
        "旧读法应把字形数放大约 6.5 万倍，实际 {}x",
        old_read / u32::from(truth)
    );

    // 决定性反证：若实现仍用旧读法，loca 容量守卫（需 numGlyphs+1 项）会
    // 立刻拒绝这个**完全合法**的内嵌字体。真值下loca 项数恰好吻合。
    let f = font_regular();
    assert!(
        f.num_glyphs() < 7000,
        "num_glyphs 若仍是 4.5e8，说明实现回退到了 u32 读法"
    );
    let (loca_off, loca_len) = find_table(data, b"loca");
    let itlf = be16(data, find_table(data, b"head").0 + 50).unwrap();
    let entry = if itlf == 1 { 4usize } else { 2 };
    assert_eq!(
        loca_len,
        (f.num_glyphs() as usize + 1) * entry,
        "loca 项数应恰好等于 numGlyphs+1"
    );
    // 哨兵项（最后一项）必须等于 glyf 表长——真实字体的自洽性铁证。
    let sentinel = if itlf == 1 {
        be32(data, loca_off + f.num_glyphs() as usize * 4)
    } else {
        be16(data, loca_off + f.num_glyphs() as usize * 2).map(|v| u32::from(v) * 2)
    };
    let (_, glyf_len) = find_table(data, b"glyf");
    assert_eq!(
        sentinel,
        Some(glyf_len as u32),
        "loca 哨兵项应等于 glyf 表长"
    );
}

/// P3 穷举层：越界 gid 全枚举 → 空轮廓 / 空位图，绝不 panic。
#[test]
fn p3_exhaustive_out_of_range_gid_never_panics() {
    let f = font_regular();
    let n = f.num_glyphs();
    let cases: Vec<u32> = vec![
        n,                             //恰好等于 numGlyphs（+1越界）
        n + 1,
        7000,
        10000,
        65535,
        u32::from(u16::MAX),
        0xFFFF_FFFF,
    ];
    for raw in cases {
        // outline 的入参是 u16：0xFFFFFFFF 需先截断，两种截断都验。
        let as_u16 = raw as u16;
        let out = f.outline(as_u16)
            .unwrap_or_else(|e| panic!("越界 gid={raw} 不应报错，Got {e}"));
        assert!(
            out.contours.is_empty(),
            "越界 gid={raw}（截断为 u16={as_u16}）应返回空轮廓，实得{} 条轮廓",
            out.contours.len()
        );
        // 顺带验一个真实存在的 CJK 字形的相邻越界点也安全。
        if raw == n {
            if let Some(gid) = f.glyph_index('中') {
                let near = gid.wrapping_add(1);
                let o2 = f.outline(near).expect("邻近 gid 应Ok");
                let _ = o2.contours.len();
            }
        }
    }
}

/// P4交叉层：`outline` 对**全字形** 0..num_glyphs 逐个解析，不得 panic。
///
/// 全量扫描是N1 越界守卫是否真正生效的最强证据——任一 gid 的 loca 切片
/// 越界都会在此暴露为 panic。
#[test]
fn p4_exhaustive_all_valid_gids_outline_never_panics() {
    let f = font_regular();
    let n = f.num_glyphs();
    let mut with_contours = 0usize;
    let mut blank = 0usize;
    let mut err = 0usize;
    for gid in 0..n.min(6889) {
        let g = gid as u16;
        match f.outline(g) {
            Ok(o) => {
                if o.contours.is_empty() {
                    blank += 1;
                } else {
                    with_contours += 1;
                }
            }
            Err(e) => {
                // T006 = 点匹配复合字形（本解析器明确不支持），属已知限制非panic。
                assert!(
                    e.code == "T006" || e.code == "T005",
                    "gid={g} 意外错误码 {}：{}",
                    e.code,
                    e.problem
                );
                err += 1;
            }
        }
    }
    assert!(with_contours > 3000, "应有大量字形含轮廓，实际 {with_contours}");
    assert!(
        blank > 0,
        "应有空轮廓字形（空格等），实际 {blank}；若无则本探针覆盖不全"
    );
    assert_eq!(
        with_contours + blank + err,
        6889,
        "全部 6889 个 gid 都应被处理（轮廓 {with_contours} + 空 {blank} + 已知限制 {err}）"
    );
}

/// P5 确定性层：同输入两遍 `outline` 逐字节一致（铁律 4）。
#[test]
fn p5_outline_is_byte_identical_across_runs() {
    let f = font_regular();
    for ch in ['A', '中', '永', 'g', '一'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        let a = f.outline(gid).expect("可解析");
        let b = f.outline(gid).expect("可解析");
        assert_eq!(a, b, "字符{ch}(gid={gid}) 两遍 outline 不一致");
    }
}
