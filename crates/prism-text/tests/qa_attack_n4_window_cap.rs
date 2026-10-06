//! QA 探针 —— 缺陷 N4 的**正向契约锁定**（`MAX_WINDOW_EDGE` 约束位图边长）。
//!
//! 历史：N4 缺陷态下 `MAX_WINDOW_EDGE` 只约束坐标**幅度**
//! （`x0/x1/y0/y1` 各自 `∈ [−cap, cap]`），而非区间**长度** `x1 − x0`，
//! 于是边长可达 `2·cap = 8192`，覆盖缓冲 `(2·cap)²·4 B = 256 MiB/字形`，
//! 是文档承诺（64 MiB）的 4 倍——契约谎言。
//!
//! 修复后契约（**本文件断言的正向命题**）：
//! 1. 任意可渲染位图 `width ≤ CAP && height ≤ CAP`；
//! 2. 故 `width * height ≤ CAP²`（64 MiB 上界由构造保证，不再是谎言）。
//!
//! # 本文件为何是**正向断言**而非缺陷见证
//!
//! 缺陷存在时，本文件**必须红**——这是它作为回归防线的基本要求。
//! 早期版本 `n4_1_rendered_edge_can_exceed_cap` 断言"**必然能找到**边长 > CAP
//! 的位图"，其 `expect` 文案自己写明"若实现已改为约束边长，本测试应红
//! （契约变更）"。那是**缺陷存在性见证测试**：N4 修好后它必然翻转，
//! **翻转即修复成功的证据，不是回归**。此类测试把"缺陷存在"编码为
//! 成功条件，修复后永久报红，反而会掩盖后续真实回归。
//!
//! 现全部反转为正向断言，并额外**自证非恒真**（见
//! [`n4_rendered_edge_within_cap_is_not_vacuous`]）：守卫被删掉则转红。

use prism_text::font_regular;
use prism_text::raster::render_glyph;

/// 与实现同源的窗口边长上界（`raster::MAX_WINDOW_EDGE` 为私有常量）。
const CAP: i64 = 4096;

fn self_consistent(b: &prism_text::raster::GlyphBitmap) -> bool {
    b.alpha.len() == b.width as usize * b.height as usize
}

/// N4-1（正向）：**任意**可渲染位图的单轴边长都 `≤ CAP`。
///
/// 旧版此测试断言"存在边长 > CAP 的位图"并量化到 `2·CAP`——那是缺陷见证。
/// 现断言契约本身：跨越大字号临界带的**全部**采样点边长均 `≤ CAP`，
/// 且位图自洽。此测试与`..is_not_vacuous`（前沿紧贴性）分工：
/// 前者守"无越界放行"，后者守"守卫确实在生效（非恒真）"。
#[test]
fn n4_rendered_edge_within_cap() {
    let f = font_regular();
    let mut rendered = 0usize;
    // 采样区跨越大字号临界带：3000..6840（旧实现在此产出 4097..8192 边长）。
    for step in 0..=48 {
        let px = 3000.0 + (step as f32) * 80.0;
        let b = render_glyph(&f, gid_of(&f, '中'), px);
        if b.is_empty() {
            continue;
        }
        rendered += 1;
        assert!(
            b.width as i64 <= CAP && b.height as i64 <= CAP,
            "px={px} 边长 ({}x{}) 越界，MAX_WINDOW_EDGE={CAP} 守卫失效",
            b.width,
            b.height
        );
        assert!(self_consistent(&b), "px={px} 位图不自洽");
    }
    assert!(rendered > 0, "采样区应至少渲染出一个非空位图（否则断言恒真）");
}

/// 取字符 gid（缺字则 panic，测试用）。
fn gid_of(f: &prism_text::ttf::Font, ch: char) -> u16 {
    match f.glyph_index(ch) {
        Some(g) => g,
        None => panic!("{ch} 应在子集内"),
    }
}

/// 二分查找"最大仍被接受的字号"（接受前沿），返回 `(px, 该处最大边长)`。
///
/// 为什么用二分而非细扫：每个被接受的位图最多分配 64 MiB 覆盖缓冲，
/// 逐 0.5px 细扫需上千次渲染（分钟级）。二分只需 ~14 次即可把前沿
/// 定位到 ±1px，足够判定"边长守卫是否紧贴生效"。
///
/// `None` 表示该区间内无任何被接受的字号。
fn accepted_frontier(f: &prism_text::ttf::Font, gid: u16, lo: f32, hi: f32) -> Option<(f32, i64)> {
    if render_glyph(f, gid, lo).is_empty() {
        return None;
    }
    let (mut lo, mut hi) = (lo, hi);
    let b0 = render_glyph(f, gid, lo);
    let mut best = (lo, b0.width.max(b0.height) as i64);
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        let b = render_glyph(f, gid, mid);
        if b.is_empty() {
            hi = mid;
        } else {
            lo = mid;
            best = (mid, b.width.max(b.height) as i64);
        }
        if hi - lo < 0.5 {
            break;
        }
    }
    Some(best)
}

/// 复刻 `render_glyph` 的窗口算术与双守卫（照抄 `raster.rs:236-256`），
/// 返回"若不设边长守卫时的本应边长"与"坐标是否已越界"。
///
/// 用途：定位**坐标合法但边长超限**的字号区间——那才是边长守卫**唯一**
/// 生效的区域。外前沿（最大可渲染 px）常由坐标守卫兜底（宽字形 x1 更早
/// 冲出 ±CAP），故在前沿上取样**无法证明边长守卫生效**（变异实测教训）。
fn edge_guard_region(f: &prism_text::ttf::Font, gid: u16, px: f32) -> Option<(i64, i64)> {
    if !(px > 0.0) || !px.is_finite() {
        return None;
    }
    let Ok(outline) = f.outline(gid) else { return None };
    if outline.contours.is_empty() {
        return None;
    }
    let scale = px / f.units_per_em() as f32;
    let (mut x_min, mut x_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for ring in &outline.contours {
        for p in ring {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
    }
    let x0 = ((x_min * scale).floor() as f64) as i64;
    let x1 = ((x_max * scale).ceil() as f64) as i64;
    let y0 = ((-y_max * scale).floor() as f64) as i64;
    let y1 = ((-y_min * scale).ceil() as f64) as i64;
    // 坐标守卫：必须在 CAP 内，否则边长守卫不是唯一约束。
    if x0 < -CAP || x1 > CAP || y0 < -CAP || y1 > CAP {
        return None;
    }
    Some(((x1 - x0).max(0), (y1 - y0).max(0)))
}

/// N4-1 加固：**自证非恒真** —— 边长守卫被删掉则此测试转红。
///
/// **三次变异实测的教训（写进注释，防止后人重犯）**：
/// 1. 只数"拒绝数 > 0" ⇒ **恒真**（删掉边长守卫后坐标守卫照样拦，计数仍 > 0）；
/// 2. 改用固定稀疏采样点（止于 4400px）⇒ **恒真**（那些点全被坐标守卫拦）；
/// 3. 改用"最大可渲染 px"外前沿 + `SLACK` ⇒ **在修复态下误报**
///    （外前沿常由坐标守卫兜底，`W` 的前沿边长仅 3528，距 CAP 很远）。
///
/// 正解：用 [`edge_guard_region`] 找**坐标合法但边长超限**的字号——
/// 那是边长守卫**唯一**能发言的区域，不存在坐标守卫抢先拦下的可能。
/// 在该处断言"被干净拒绝"，删掉边长守卫即转红（已实测）。
#[test]
fn n4_rendered_edge_within_cap_is_not_vacuous() {
    let f = font_regular();
    let mut guard_only_hits = 0usize;
    let mut accepted_in_range = 0usize;
    for ch in ['中', '国', 'W', 'M', '永', 'A'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=400 {
            let px = 3000.0 + (step as f32) * 4.0; // 3000..4600
            let Some((ew, eh)) = edge_guard_region(&f, gid, px) else { continue };
            if ew <= CAP && eh <= CAP {
                accepted_in_range += 1;
                continue;
            }
            // 坐标合法、边长超限 ⇒ 边长守卫必须拦。
            let b = render_glyph(&f, gid, px);
            assert!(
                b.is_empty(),
                "ch={ch} px={px} 坐标合法但本应边长 {ew}x{eh} > {CAP}，却渲染成功 ⇒ 边长守卫失效"
            );
            assert_eq!(b.alpha.len(), 0, "ch={ch} px={px} 拒绝时不应分配 alpha 缓冲");
            guard_only_hits += 1;
        }
    }
    assert!(guard_only_hits > 0, "未触达边长守卫专属区域 ⇒ 测试恒真");
    assert!(accepted_in_range > 0, "未采样到守卫内字号，覆盖不足");
    println!("[N4] 边长守卫专属区域命中 {guard_only_hits} 次（守卫内 {accepted_in_range} 个字号）");
}

/// N4-2：坐标幅度守卫本身有效（x0/x1/y0/y1 语义正确）——防"越界变全拒"。
#[test]
fn n4_2_coordinate_guard_still_rejects_extremes() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    // 极大字号：坐标幅度必然超 cap → 必须拒绝（空位图，不分配）。
    for &px in &[1.0e4f32, 1.0e5, 1.0e10, 1.0e20, f32::MAX] {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px={px:e} 应被坐标守卫拒绝");
        assert_eq!(b.alpha.len(), 0, "px={px:e} 不应分配 alpha");
    }
}

/// N4-3：确定性 —— 边长临界带（含被拒绝者）仍逐字节可复现（无隐藏状态）。
#[test]
fn n4_3_oversized_bitmap_is_deterministic() {
    let f = font_regular();
    let gid = f.glyph_index('永').expect("永在子集内");
    for &px in &[3000.0f32, 3300.0, 3600.0, 4200.0, 4600.0] {
        let a = render_glyph(&f, gid, px);
        let b = render_glyph(&f, gid, px);
        assert_eq!(a, b, "px={px} 两遍不一致");
        assert!(self_consistent(&a));
        assert!(a.width as i64 <= CAP && a.height as i64 <= CAP, "px={px} 边长越界");
    }
}

/// N4-4：内存上界量化——断言**硬上界**（旧版仅`println!` 记录事实）。
///
/// **两次变异实测的教训（都记在这里，避免后人重犯）**：
/// 1. 旧版采样点止于 4400px，缺陷态下那些点全被**坐标守卫**抢先拦掉
///    ⇒ 断言恒真；
/// 2. 改用"最大可渲染 px"（外前沿）后**仍恒真**——因为外前沿由坐标守卫
///    兜底（宽字形 `x1` 更早冲出 ±CAP），边长守卫在那一带根本没被触达。
///
/// 现改为**直击边长守卫的专属区域**：用 [`edge_guard_region`] 找
/// "坐标合法但边长超限"的字号，在该处断言渲染被拒。不存在坐标守卫
/// 抢先拦下的可能，故删掉边长守卫即转红。
#[test]
fn n4_4_worst_case_buffer_within_cap() {
    let f = font_regular();
    let cap_sq = (CAP as u64) * (CAP as u64);
    let mut checked = 0usize;
    let mut max_accepted = 0u64;
    for ch in ['中', '国', 'W', 'M', '永', 'A'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=400 {
            let px = 3000.0 + (step as f32) * 4.0; // 3000..4600
            let Some((ew, eh)) = edge_guard_region(&f, gid, px) else { continue };
            if ew <= CAP && eh <= CAP {
                // 守卫内：面积必须守恒（这是文档承诺的"64 MiB 上界"）。
                // 守卫内的位图最大 64 MiB/个，故**抽样**渲染（每字形 3 点）
                // ——全量渲染需上千次 64 MiB 分配（分钟级），性价比极低。
                // 守卫外分支（拒绝路径）不分配大缓冲，可全量断言。
                if step % 137 == 0 {
                    let b = render_glyph(&f, gid, px);
                    if !b.is_empty() {
                        assert!(self_consistent(&b), "ch={ch} px={px} 位图不自洽");
                        let cells = b.width as u64 * b.height as u64;
                        assert!(
                            cells <= cap_sq,
                            "ch={ch} px={px} 面积 {cells} > CAP²={cap_sq} —— 边长守卫失效"
                        );
                        max_accepted = max_accepted.max(cells);
                    }
                }
            } else {
                // 坐标合法但边长超限 ⇒ 必须被边长守卫拒绝。
                let b = render_glyph(&f, gid, px);
                assert!(
                    b.is_empty(),
                    "ch={ch} px={px} 本应边长 {}x{} > {CAP} 却渲染成功 ⇒ 边长守卫失效",
                    ew,
                    eh
                );
                assert_eq!(b.alpha.len(), 0, "ch={ch} px={px} 拒绝时不应分配 alpha");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "未触达边长守卫专属区域，测试近乎恒真");
    println!("[N4] 边长守卫专属区域命中 {checked} 次；守卫内最大面积 {max_accepted} 单元");
}

/// N4-5：面积上界由构造保证 —— 显式断言 `width*height ≤ CAP²`。
///
/// 这正是旧实现注释里"声称成立但实际不成立"的那条断言。修复后它必须
/// **字面成立**（不再是谎言）。采样策略见`n4_4`（必须直击边长守卫区域，
/// 否则被坐标守卫抢先拦下而恒真）。
#[test]
fn n4_5_area_upper_bound_holds_as_documented() {
    let f = font_regular();
    let cap_sq = (CAP as u64) * (CAP as u64);
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for ch in ['中', 'W', 'A', '永'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=300 {
            let px = 3000.0 + (step as f32) * 5.0;
            let b = render_glyph(&f, gid, px);
            if b.is_empty() {
                rejected += 1;
                continue;
            }
            let cells = b.width as u64 * b.height as u64;
            assert!(
                cells <= cap_sq,
                "ch={ch} px={px} 面积 {cells} > CAP²={cap_sq}——文档断言不成立"
            );
            assert!(b.width as i64 <= CAP && b.height as i64 <= CAP, "ch={ch} px={px} 边长越界");
            accepted += 1;
        }
    }
    assert!(accepted > 20 && rejected > 20, "接受 {accepted} / 拒绝 {rejected}，覆盖不足");
    println!("[N4] 面积断言：接受 {accepted} 位图 / 拒绝 {rejected} 字号，上界 {cap_sq} 单元");
}
