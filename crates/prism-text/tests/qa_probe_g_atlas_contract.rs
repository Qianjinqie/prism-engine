//! QA 独立探针 —— 缺陷 G（`max_width` 契约修复）终验 + D/E/F 溢出路径。
//!
//! G 是**行为变更**：`mw = max_width.max(cell)` → `mw = max_width.max(1)`，
//! 且 `bw > mw` 的单格按"无墨"处理。QA 关注四件事：
//! 1. **UV 恒在 [0,1]**（shader 采样不得越界）；
//! 2. **cursor 推进自洽**（无重叠、无错位空洞）；
//! 3. **确定性**（同输入两遍逐字节一致）；
//! 4. **静默跳过是否误触发**——在常规参数下是否丢字形（丢则=新缺陷）。
//!
//! 另覆盖 D/E/F：`atlas_w = u32::MAX` / `usize::MAX`、`bake_size = 1e30`
//! 等溢出输入不得 panic / OOM。

use prism_text::font_regular;
use prism_text::sdf::{bake_glyph_sdf, GlyphAtlas, SdfParams};

/// G-P1：UV 恒在 [0,1]，且 u0<u1、v0<v1。
///
/// 失败含义（= 缺陷 G 未修好）：某槽位宽度超出atlas 宽 ⇒ `u1 > 1`，
/// shader 双线性采样越界 → 采到邻格字形或atlas 外（视觉错字/闪黑）。
#[test]
fn g_p1_uv_always_within_unit_range() {
    let f = font_regular();
    // 收集一批覆盖繁简拉丁 + CJK + 标点的 gid。
    let mut gids: Vec<u16> = ['A', 'B', 'W', 'i', 'g', '中', '永', '国', '一', '、', '，', '。']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    gids.push(f.glyph_index(' ').expect("空格在 cmap 中")); // blank
    gids.sort_unstable();
    gids.dedup();

    // 参数矩阵：覆盖"max_width 很小、pad 很大"（G 的目标场景）与常规场景。
    let cases: Vec<(f32, f32, u32, u32, &str)> = vec![
        (48.0, 8.0, 4, 512, "常规"),
        (48.0, 8.0, 4, 64, "窄 atlas（略窄于单格）"),
        (32.0, 6.0, 3, 512, "常规小字号"),
        (32.0, 6.0, 3, 40, "极窄 atlas（<单格）"),
        (32.0, 6.0, 3, 16, "超窄（远< 单格）"),
        (48.0, 8.0, 64, 128, "大 pad + 小max_width（G 的原始触发点）"),
        (48.0, 8.0, 1024, 1024, "MAX_PAD 极端"),
        (16.0, 2.0, 1, 8, "微型"),
        (64.0, 12.0, 8, 1, "max_width=1（最小）"),
        (48.0, 8.0, 4, 0, "max_width=0（下界）"),
    ];

    for &(bs, pr, pad, mw, tag) in &cases {
        let p = SdfParams::new(bs, pr, pad);
        let atlas = GlyphAtlas::bake(&f, &gids, &p, mw);
        // atlas 宽必须**不超过调用方给的上限**（G 的核心契约）。
        assert!(
            atlas.width <= mw.max(1),
            "[{tag}] atlas 宽 {} 超过 max_width 上限 {}（契约被反向撑大）",
            atlas.width,
            mw.max(1)
        );
        if atlas.slots.is_empty() {
            // 全部被跳过：atlas 必须自洽（0 尺寸或 0 像素），UV 无从违反。
            assert!(
                atlas.data.len() == 0 || atlas.width as usize * atlas.height as usize == atlas.data.len(),
                "[{tag}] 空 atlas 数据不自洽"
            );
            continue;
        }
        for slot in &atlas.slots {
            let uv = atlas.uv(slot);
            assert!(
                uv[0] >= 0.0 && uv[0] <= 1.0,
                "[{tag}] u0={} 越界（gid={}）",
                uv[0],
                slot.gid
            );
            assert!(
                uv[1] >= 0.0 && uv[1] <= 1.0,
                "[{tag}] v0={} 越界（gid={}）",
                uv[1],
                slot.gid
            );
            assert!(
                uv[2] <= 1.0,
                "[{tag}] u1={} > 1（gid={}）—— shader 采样越界！",
                uv[2],
                slot.gid
            );
            assert!(
                uv[3] <= 1.0,
                "[{tag}] v1={} > 1（gid={}）—— shader 采样越界！",
                uv[3],
                slot.gid
            );
            assert!(uv[0] < uv[2] && uv[1] < uv[3], "[{tag}] UV 退化为零宽/零高");
            // 槽位像素坐标也必须落在 atlas 内。
            assert!(
                slot.x + slot.w <= atlas.width && slot.y + slot.h <= atlas.height,
                "[{tag}] 槽位 ({},{})+{}x{} 超出 atlas {}x{}",
                slot.x,
                slot.y,
                slot.w,
                slot.h,
                atlas.width,
                atlas.height
            );
        }
    }
}

/// G-P2：cursor 推进自洽 —— 同行不重叠、跨行 y 单调、x 从 0 起、无重叠面积。
#[test]
fn g_p2_cursor_advance_is_self_consistent_no_overlap() {
    let f = font_regular();
    let gids: Vec<u16> = ['H', 'e', 'l', 'o', '中', '永', '国', 'W', 'M']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    // 多组 max_width 让装箱真正换行。
    for &mw in &[64u32, 96, 128, 192, 256, 512, 1024] {
        let p = SdfParams::new(40.0, 8.0, 4);
        let atlas = GlyphAtlas::bake(&f, &gids, &p, mw);
        let mut slots: Vec<&prism_text::sdf::AtlasSlot> = atlas.slots.iter().collect();
        slots.sort_by_key(|s| (s.y, s.x));
        // y 相同的槽位（同一行）内 x 区间必须互不相交，且首个从 0 起。
        let mut i = 0usize;
        while i < slots.len() {
            let row_y = slots[i].y;
            let mut row: Vec<&&prism_text::sdf::AtlasSlot> = slots.iter().filter(|s| s.y == row_y).collect();
            row.sort_by_key(|s| s.x);
            assert_eq!(row[0].x, 0, "[mw={mw}] 行的首个槽位 x 应为 0，实为 {}", row[0].x);
            for pair in row.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                assert!(
                    a.x + a.w <= b.x,
                    "[mw={mw}] 同行槽位重叠：gid={} [{},{}) 与 gid={} [{},{})",
                    a.gid,
                    a.x,
                    a.x + a.w,
                    b.gid,
                    b.x,
                    b.x + b.w
                );
            }
            i += row.len();
        }
        // 逐行（按 y 递增）验证行的 y 递进等于上一行最大高（shelf 自洽）。
        let mut rows: Vec<(u32, u32, u32)> = Vec::new(); // (y, max_h, min_x)
        for s in &slots {
            match rows.last_mut() {
                Some(last) if last.0 == s.y => {
                    last.1 = last.1.max(s.h);
                    last.2 = last.2.min(s.x);
                }
                _ => rows.push((s.y, s.h, s.x)),
            }
        }
        assert_eq!(rows[0].0, 0, "[mw={mw}] 首行 y 应为 0");
        for w in rows.windows(2) {
            let (prev, cur) = (w[0], w[1]);
            assert!(
                cur.0 >= prev.0 + prev.1,
                "[mw={mw}] 行 y 未按上一行高度推进：行{} 高{} → 下一行 y={}（应 ≥ {}）",
                prev.0,
                prev.1,
                cur.0,
                prev.0 + prev.1
            );
        }
        // atlas 高必须等于"最后一行 y + 其最大高"（无尾部空洞）。
        assert!(
            atlas.height >= rows.last().unwrap().0 + rows.last().unwrap().1,
            "[mw={mw}] atlas 高 {} 小于内容高度 {}",
            atlas.height,
            rows.last().unwrap().0 + rows.last().unwrap().1
        );
        // 数据必须自洽。
        assert_eq!(
            atlas.data.len(),
            atlas.width as usize * atlas.height as usize,
            "[mw={mw}] atlas 数据长度不自洽"
        );
    }
}

/// G-P3：确定性 —— 同输入两遍逐字节一致（含 slots/blanks 顺序）。
#[test]
fn g_p3_atlas_is_byte_identical_across_runs() {
    let f = font_regular();
    let gids: Vec<u16> = ['A', '中', '永', ' ', 'g', '国']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    for &mw in &[40u32, 128, 512] {
        let p = SdfParams::new(48.0, 8.0, 4);
        let a = GlyphAtlas::bake(&f, &gids, &p, mw);
        let b = GlyphAtlas::bake(&f, &gids, &p, mw);
        assert_eq!(a.width, b.width, "[mw={mw}] 两次 atlas 宽不同");
        assert_eq!(a.height, b.height, "[mw={mw}] 两次 atlas 高不同");
        assert_eq!(a.data, b.data, "[mw={mw}] 两次 atlas 像素数据不同（确定性铁律）");
        assert_eq!(a.slots, b.slots, "[mw={mw}] 两次 slots 不同");
        assert_eq!(a.blanks, b.blanks, "[mw={mw}] 两次 blanks 不同");
        // gid 顺序无关性：bake 内部按 gid 升序，输入乱序结果应相同。
        let mut rev = gids.clone();
        rev.reverse();
        let c = GlyphAtlas::bake(&f, &rev, &p, mw);
        assert_eq!(a.data, c.data, "[mw={mw}] 输入顺序影响了烘焙结果（确定性铁律）");
    }
}

/// G-P4（**关键**）：静默跳过是否误触发 —— 常规参数下是否丢字形。
///
/// 判定标准：对"合理"参数（bake_size 48/32、pad 4/3、max_width ≥单格宽），
/// 参与烘焙的**有墨**字形必须全部出现在 `slots` 里。`blanks` 只允许包含
/// 真正无墨的字形（空格等）。
#[test]
fn g_p4_silent_skip_does_not_drop_ink_glyphs_under_sane_params() {
    let f = font_regular();
    // 一批确定有墨的字符（不用空格，避免"无墨"歧义）。
    let chars = ['A', 'B', 'C', 'D', '中', '永', '国', '一', '永', 'H', 'i', 'W', 'M', 'g', 'p'];
    let mut gids = Vec::new();
    for c in chars {
        if let Some(g) = f.glyph_index(c) {
            gids.push(g);
        }
    }
    gids.sort_unstable();
    gids.dedup();

    // 常规参数矩阵：max_width 取"明显够用"的值（远大于最宽单格）。
    for &bs in &[32.0f32, 48.0, 64.0] {
        for &pad in &[3u32, 4, 8] {
            let p = SdfParams::new(bs, 8.0, pad);
            // 先量出单格最大宽（逐字形单独烘焙，避开 atlas 整体预算上限）。
            let cell_w: Vec<(u16, u32)> = gids
                .iter()
                .map(|&g| (g, bake_glyph_sdf(&f, g, &p).width))
                .filter(|&(_, w)| w > 0)
                .collect();
            let max_cell_w = cell_w.iter().map(|&(_, w)| w).max().unwrap_or(0);
            assert!(max_cell_w > 0, "前置条件：应有可烘焙（有墨）字形");
            // "合理" = 至少能并排放下 4 个最宽单格。
            let sane_mw = max_cell_w.saturating_mul(4).max(64);
            let atlas = GlyphAtlas::bake(&f, &gids, &p, sane_mw);
            for &(g, w) in &cell_w {
                // 该字形单独烘焙必须有墨（前置条件，排除真无墨字形）。
                assert!(
                    !bake_glyph_sdf(&f, g, &p).is_empty(),
                    "前置条件失败：gid={g} 单独烘焙为空，无法判断是否被误跳过"
                );
                assert!(
                    atlas.slot(g).is_some(),
                    "误触发！gid={g}（单格宽 {w}，max_width={sane_mw} 合理）被按无墨跳过"
                );
            }
            assert!(
                !atlas.blanks.iter().any(|b| gids.contains(b)),
                "[bs={bs} pad={pad}] blanks 中出现了有墨字形：{:?}",
                atlas.blanks
            );
        }
    }
}

/// G-P5：`bw > mw` 只在"单格确实塞不进"时触发（含边界等值检查）。
#[test]
fn g_p5_skip_only_when_cell_exceeds_width() {
    let f = font_regular();
    let gids: Vec<u16> = ['A', '中', '永', '国'].iter().filter_map(|&c| f.glyph_index(c)).collect();
    let p = SdfParams::new(48.0, 8.0, 4);
    // 量出每字形单格宽
    let widths: Vec<(u16, u32)> =
        gids.iter().map(|&g| (g, bake_glyph_sdf(&f, g, &p).width)).collect();
    for (gid, w) in widths {
        // max_width == w：恰好装下 → 必须有槽位（等号不算超）
        let at = GlyphAtlas::bake(&f, &[gid], &p, w);
        assert!(
            at.slot(gid).is_some(),
            "max_width={w} 恰好等于单格宽，gid={gid} 应装得下却被跳过"
        );
        // max_width == w-1：装不下 → 应跳过
        let below = GlyphAtlas::bake(&f, &[gid], &p, w - 1);
        assert!(
            below.slot(gid).is_none(),
            "max_width={} < 单格宽 {w}，gid={gid} 应被跳过却仍有槽位",
            w - 1
        );
        // 跳过后 UV 不存在（无从越界）
        if below.slot(gid).is_none() {
            assert!(
                below.blanks.contains(&gid),
                "被跳过的 gid={gid} 应登记为 blank"
            );
        }
    }
}

/// D-P6：`bake_glyph_sdf` 的 `width*height` 不得溢出分配（u64 中间量）。
#[test]
fn d_p6_bake_sdf_huge_bake_size_no_panic() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    // SdfParams::new 会把非有限/非正收束到默认，但 1e30 是**有限正数**，
    // 须由 bake_glyph_sdf 自身的 cells 上界拦住（否则 u32 乘积溢出/OOM）。
    for &bs in &[1.0e30f32, 1.0e10, 1.0e5, 3.0e38, f32::MAX] {
        let p = SdfParams::new(bs, 8.0, 4);
        let s = bake_glyph_sdf(&f, gid, &p);
        let cells = s.width as u64 * s.height as u64;
        assert!(
            cells <= 4096u64 * 4096,
            "bs={bs:e} 产出 {cells} 单元，超过 MAX_ATLAS_CELL² 上界"
        );
        assert_eq!(
            s.data.len() as u64,
            cells,
            "bs={bs:e} SDF 数据长度与尺寸不自洽（{} vs {cells}）",
            s.data.len()
        );
    }
    // 非法参数收束到默认而非 panic
    for &bs in &[f32::NAN, 0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY] {
        let p = SdfParams::new(bs, 8.0, 4);
        let s = bake_glyph_sdf(&f, gid, &p);
        assert!(!s.is_empty(), "bs={bs} 应收束到默认并正常烘焙");
    }
}

/// E/F-P7：atlas 分配溢出路径 —— `atlas_w = u32::MAX` / 0 / 1 等不得 OOM/panic。
#[test]
fn ef_p7_atlas_extreme_max_width_never_panics_or_ooms() {
    let f = font_regular();
    let gids: Vec<u16> = ['A', '中', '永', '国'].iter().filter_map(|&c| f.glyph_index(c)).collect();
    let p = SdfParams::new(48.0, 8.0, 4);
    for &mw in &[
        0u32,
        1,
        2,
        3,
        u32::MAX,
        u32::MAX - 1,
        1 << 31,
        (1 << 31) + 1,
        0xFFFF_FFFE,
    ] {
        let at = GlyphAtlas::bake(&f, &gids, &p, mw);
        // 不得 OOM：数据长度须在 MAX_ATLAS_CELL² 预算内。
        let cells = at.width as u64 * at.height as u64;
        assert!(
            cells <= 4096u64 * 4096,
            "max_width={mw} 产出 {cells} 单元，超出内存预算（OOM 风险）"
        );
        assert_eq!(
            at.data.len() as u64,
            cells,
            "max_width={mw} atlas 数据不自洽"
        );
        assert!(at.width <= mw.max(1), "max_width={mw} atlas 宽 {} 反向撑大", at.width);
        // UV 仍然自洽
        for s in &at.slots {
            let uv = at.uv(s);
            assert!(
                uv[0] >= 0.0 && uv[1] >= 0.0 && uv[2] <= 1.0 && uv[3] <= 1.0,
                "max_width={mw} UV 越界：{uv:?}"
            );
        }
    }
}

/// E/F-P8：`usize::MAX` 语义的 max_width（Rust 侧 usize 传入 u32 会截断），
/// 以及超大 gid 集合 + 超大 pad 的组合压力。
#[test]
fn ef_p8_stress_many_gids_and_large_pad() {
    let f = font_regular();
    // 6889 个 gid 全量烘焙（最重的真实负载）。
    let all: Vec<u16> = (0..f.num_glyphs().min(6889) as u16).collect();
    let p = SdfParams::new(16.0, 4.0, 1);
    let at = GlyphAtlas::bake(&f, &all, &p, 512);
    let cells = at.width as u64 * at.height as u64;
    assert!(cells <= 4096u64 * 4096, "全量烘焙超预算：{cells}");
    for s in &at.slots {
        let uv = at.uv(s);
        assert!(uv[2] <= 1.0 && uv[3] <= 1.0, "全量烘焙 UV 越界：{uv:?}");
    }
    // 槽位数 + blank数 == 去重后的 gid 数（不丢不重）
    let mut uniq = all.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        at.slots.len() + at.blanks.len(),
        uniq.len(),
        "槽位({}) + blank({}) != 去重 gid 数({}) —— 有丢字形或重复计数",
        at.slots.len(),
        at.blanks.len(),
        uniq.len()
    );
    // 超大 pad（MAX_PAD=1024）+ 小 max_width：G 的极端组合
    let p2 = SdfParams::new(32.0, 8.0, 1024);
    let at2 = GlyphAtlas::bake(&f, &gids_all_small(&f), &p2, 64);
    let cells2 = at2.width as u64 * at2.height as u64;
    assert!(cells2 <= 4096u64 * 4096, "MAX_PAD 组合超预算：{cells2}");
}

fn gids_all_small(f: &prism_text::ttf::Font) -> Vec<u16> {
    ['A', '中', '永'].iter().filter_map(|&c| f.glyph_index(c)).collect()
}
