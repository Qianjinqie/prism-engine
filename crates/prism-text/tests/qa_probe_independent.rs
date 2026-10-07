//! QA 独立探针 —— 缺陷 N3（`blit_glyph` 偏移饱和加溢出）**独立复核**。
//!
//! 本文件由 QA 编写，**不复用** `tests/fix_boundary.rs`（工程师自证）
//! 或 `tests/qa_attack_blit_overflow.rs`（缺陷见证）的任何代码路径。
//!
//! 缺陷态回顾：`ox = (pen_x.floor() as i64) + (bmp.left as i64)`。
//! - `f32 as i64` 是**饱和**转换：`±1e30 → ±i64::MAX`；
//! - `left/top` 是 `pub` 字段，外部可置 `i32::MIN/MAX`；
//! - 裸 `+` 在 **debug 下溢 panic**（违反 C18）；
//! - 在 **release 静默回绕**成"贴边错图"（违反确定性铁律：跨优化级别输出不同）。
//!
//! 复核清单：
//! 1. 复现路径全绿（N3 原始复现式）；
//! 2. 极端 pen/baseline × left/top/width/height **全组合**不 panic；
//! 3. `draw_glyphs` 全链路在极端 pen 下不 panic；
//! 4. 极端输入下**画面全零**（无"贴边错图"）——N3 的要害不是 panic 而
//!    是 release 静默错图，故必须正向断言"什么都没画"；
//! 5. 两遍逐字节一致（确定性铁律）。

use prism_io::Fbuf16F;
use prism_text::cache::GlyphCache;
use prism_text::font_regular;
use prism_text::layout::GlyphPos;
use prism_text::raster::{blit_glyph, draw_glyphs, render_glyph, GlyphBitmap};

/// pen/baseline 极端取值全集（含 NaN/±Inf/±f32::MAX/饱和阈值/±0.0）。
const EXTREMES: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.0e19,
    -1.0e19,
    9.3e18,  // i64::MAX 附近（f32 可精确表示上界）
    -9.3e18, // i64::MIN 附近
    1.0e30,
    -1.0e30,
    3.4e38,   //≈ f32::MAX
    -3.4e38,  // ≈ f32::MIN
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// `left/top` 极端取值（`pub` 字段可被外部任意构造）。
const OFFSETS: &[i32] = &[0, 1, -1, i32::MAX, i32::MIN, i32::MAX - 1, i32::MIN + 1, 12345, -12345];

/// 自洽小位图（绕过 `alpha.len() < w*h` 早退，确保真正走到偏移计算）。
fn small(left: i32, top: i32, w: u32, h: u32) -> GlyphBitmap {
    GlyphBitmap { left, top, width: w, height: h, alpha: vec![255u8; (w * h) as usize] }
}

/// 帧缓冲是否全零（"什么都没画"）。
fn is_clean(fb: &Fbuf16F) -> bool {
    fb.pixels.iter().all(|&p| p == 0)
}

/// 帧缓冲指纹（确定性逐字节比对用）。
fn fingerprint(fb: &Fbuf16F) -> Vec<u16> {
    fb.pixels.clone()
}

// ---------------------------------------------------------------------------
// N3-1：N3 原始复现路径（回归锁定）
// ---------------------------------------------------------------------------

/// N3-1：缺陷态的**原始复现式**必须转绿。
///
/// 复现式取自 N3 缺陷报告：极端`left/top` + 极端 pen。
#[test]
fn n3_1_original_repro_no_longer_panics() {
    let mut fb = Fbuf16F::new(8, 8);
    let bmp = GlyphBitmap {
        left: i32::MIN,
        top: i32::MIN,
        width: 4,
        height: 4,
        alpha: vec![200u8; 16],
    };
    // 缺陷态：`(−1e30).floor() as i64 → i64::MIN`，`+ i32::MIN` → 下溢 panic。
    blit_glyph(&mut fb, &bmp, -1e30, -1e30, [1.0; 4]);
    assert!(is_clean(&fb), "极端偏移必落在视口外，应拒绝绘制（无贴边错图）");
    // 【D1】非有限笔位必须**拒绝**，而非被静默归一化为 0。
    // 对照位图放在原点，使"画在原点"可被观测到。
    let origin = GlyphBitmap { left: 0, top: 0, width: 4, height: 4, alpha: vec![255u8; 16] };
    for &v in EXTREMES.iter().filter(|v| !v.is_finite()) {
        let mut f1 = Fbuf16F::new(8, 8);
        blit_glyph(&mut f1, &origin, v, 0.0, [1.0; 4]);
        assert!(is_clean(&f1), "pen={v:e} 非有限笔位应被拒，却落了墨 ⇒ NaN 被当成 0");
        let mut f2 = Fbuf16F::new(8, 8);
        blit_glyph(&mut f2, &origin, 0.0, v, [1.0; 4]);
        assert!(is_clean(&f2), "baseline={v:e} 非有限笔位应被拒");
    }
}

/// QA 侧参考模型：复刻 `blit_glyph` 的**契约语义**（含D1 非有限笔位拒绝）。
///
/// 这是 N3+ D1 的**正确行为定义**。若实现用裸 `+`（release 回绕 / debug panic），
/// 或把 NaN 静默归一化为 0（缺陷态），实际落墨集合必与本模型不符→ 转红。
fn expected_ink(pen: f32, base: f32, l: i32, t: i32, bw: u32, bh: u32, vw: u32, vh: u32) -> Vec<(u32, u32)> {
    // 【D1】非有限笔位 = 无效输入，直接拒绝（等价于"这一笔没画"）。
    // 与 `render_glyph` 的 `!px_size.is_finite()` 拒绝口径对齐。
    // 注意：Rust 的 `NaN as i64 == 0`，故若无此守卫 NaN 会被静默画到原点。
    if !pen.is_finite() || !base.is_finite() {
        return Vec::new();
    }
    // `f32 as i64` 饱和转换（有限值不会饱和到 i64 边界）。
    let px = f64::from(pen.floor()) as i64;
    let py = f64::from(base.floor()) as i64;
    let ox = px.saturating_add(l as i64);
    let oy = py.saturating_add(t as i64);
    let mut out = Vec::new();
    for row in 0..bh as i64 {
        let fy = oy.saturating_add(row);
        if fy < 0 || fy >= vh as i64 {
            continue;
        }
        for col in 0..bw as i64 {
            let fx = ox.saturating_add(col);
            if fx < 0 || fx >= vw as i64 {
                continue;
            }
            out.push((fx as u32, fy as u32));
        }
    }
    out
}

/// 实际落墨的像素集合。
fn actual_ink(fb: &Fbuf16F) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in 0..fb.height {
        for x in 0..fb.width {
            if fb.get(x, y)[3] > 0.0 {
                out.push((x, y));
            }
        }
    }
    out
}

/// N3-2：极端 pen/baseline × `left/top` × 边长 —— 全组合不 panic 且落墨集合
/// **精确等于**参考模型。
///
/// 组合数 = 17 × 17 × 9 × 3 = 7803 次 blit。
///
/// 上一版此测试断言"全部画面全零"——**那是我的断言写错了**：`pen=0/left=0`
/// 是合法落在视口内的绘制，本就该有墨。已修正为与参考模型逐点对账：
/// 这比"全零"更强，因为它同时锁定了"该画时必须画"与"不该画时不能画"。
///
/// 【D1 实测】模型必须编码**非有限笔位拒绝**语义。首版模型按
/// `NaN as i64 == 0` 推算"应画在原点"，与修复后的实现（早退）不符⇒
/// 7803 组合中 NaN 类组合误报红。**是模型错、实现对**，已修正。
#[test]
fn n3_2_exhaustive_combinations_match_reference_model() {
    const VW: u32 = 8;
    const VH: u32 = 8;
    let dims: [(i32, i32, u32, u32); 3] =
        [(0, 0, 2, 2), (i32::MIN, i32::MAX, 1, 1), (i32::MAX, i32::MIN, 3, 1)];
    let mut n = 0usize;
    let mut painted = 0usize;
    for &pen in EXTREMES {
        for &base in EXTREMES {
            for &l in OFFSETS {
                for &t in OFFSETS {
                    for &(_bl, _bt, bw, bh) in &dims {
                        let mut fb = Fbuf16F::new(VW, VH);
                        let bmp = small(l, t, bw, bh);
                        blit_glyph(&mut fb, &bmp, pen, base, [1.0, 0.5, 0.25, 1.0]);
                        n += 1;
                        let got = actual_ink(&fb);
                        let want = expected_ink(pen, base, l, t, bw, bh, VW, VH);
                        assert_eq!(
                            got, want,
                            "pen={pen:e} base={base:e} left={l} top={t} {bw}x{bh}:\
                             落墨集合与饱和语义参考模型不符（回绕错图或漏画）"
                        );
                        if !got.is_empty() {
                            painted += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(n > 7000, "组合数 {n} 低于预期，检查器未真正穷举");
    // 自证非恒真：确有组合落在视口内被绘制（检查器有区分力）。
    assert!(painted > 0, "全部组合都未落墨，参考模型可能恒等于空集");
    println!("[N3] 穷举 {n} 组合全部与参考模型一致，其中 {painted} 组合落墨");
}

/// N3-3：`draw_glyphs` 全链路在极端 pen/dx/dy 下不 panic。
#[test]
fn n3_3_draw_glyphs_extreme_delta_never_panics() {
    let f = font_regular();
    let glyphs = vec![
        GlyphPos {
            ch: '中',
            gid: f.glyph_index('中').expect("中"),
            x: 0.0,
            y: 0.0,
            advance: 24.0,
            line: 0,
            index: 0,
        },
        GlyphPos {
            ch: 'A',
            gid: f.glyph_index('A').expect("A"),
            x: 24.0,
            y: 0.0,
            advance: 16.0,
            line: 0,
            index: 1,
        },
    ];
    for &d in EXTREMES {
        for &size in &[24.0f32, 1.0e10, f32::MAX, 0.0, -0.0, f32::NAN] {
            let mut fb = Fbuf16F::new(16, 16);
            let mut cache = GlyphCache::new();
            draw_glyphs(&mut fb, &f, &glyphs, size, [1.0; 4], d, d, &mut cache);
        }
    }
    // 常规 delta 仍应正常绘制（守卫未把正常路径也杀掉）。
    let mut fb = Fbuf16F::new(32, 32);
    let mut cache = GlyphCache::new();
    draw_glyphs(&mut fb, &f, &glyphs, 24.0, [1.0, 1.0, 1.0, 1.0], 2.0, 4.0, &mut cache);
    assert!(!is_clean(&fb), "常规 pen/delta 应正常绘制（修复未误杀正常路径）");
}

/// N3-4：确定性 —— 极端输入下两遍跑 `raster` + `draw_glyphs` 逐字节一致。
#[test]
fn n3_4_extreme_inputs_are_byte_identical_across_runs() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中");
    // (a) 渲染层确定性
    for &px in &[1.0e-3f32, 0.5, 24.0, 4096.0, 1.0e5, 1.0e30, f32::MAX] {
        let a = render_glyph(&f, gid, px);
        let b = render_glyph(&f, gid, px);
        assert_eq!(a, b, "px={px:e} 渲染两遍不一致");
    }
    // (b) 合成层确定性：全极端 pen 组合下两遍画面逐字节一致。
    let glyphs = vec![GlyphPos {
        ch: '中',
        gid,
        x: 0.0,
        y: 0.0,
        advance: 24.0,
        line: 0,
        index: 0,
    }];
    for &d in EXTREMES {
        let mut f1 = Fbuf16F::new(16, 16);
        let mut c1 = GlyphCache::new();
        draw_glyphs(&mut f1, &f, &glyphs, 24.0, [0.3, 0.6, 0.9, 1.0], d, d, &mut c1);
        let mut f2 = Fbuf16F::new(16, 16);
        let mut c2 = GlyphCache::new();
        draw_glyphs(&mut f2, &f, &glyphs, 24.0, [0.3, 0.6, 0.9, 1.0], d, d, &mut c2);
        assert_eq!(fingerprint(&f1), fingerprint(&f2), "delta={d:e} 两遍画面不一致");
    }
}

/// N3-5：**自证非恒真** —— 偏移守卫被删掉则转红。
///
/// 手法：正常 pen + 正常 `left/top` 必须画出像素（证明检查器有区分力），
/// 而饱和偏移必须画不出（证明守卫在拦）。两条共存 ⇒ 恒真被排除。
#[test]
fn n3_5_guard_is_not_vacuous() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中");
    let bmp = render_glyph(&f, gid, 24.0);
    assert!(!bmp.is_empty(), "前置：24px 中字应可渲染");

    // 正常路径：笔位落在字形 bbox 内 ⇒ 必须落墨。
    let mut fb_ok = Fbuf16F::new(64, 64);
    blit_glyph(&mut fb_ok, &bmp, 20.0, 40.0, [1.0; 4]);
    assert!(!is_clean(&fb_ok), "正常笔位应落墨（检查器有区分力）");

    // 饱和路径：偏移必在视口外 ⇒ 必须画不出。
    let mut fb_sat = Fbuf16F::new(64, 64);
    let ext = small(i32::MIN, i32::MIN, 4, 4);
    blit_glyph(&mut fb_sat, &ext, -1e30, 1.0e30, [1.0; 4]);
    assert!(is_clean(&fb_sat), "饱和偏移应被丢弃（若落墨 ⇒ 回绕错图）");
}

/// N3-6：【D1 回归】非有限笔位（`NaN` / `±Inf`）必须**显式拒绝**，
/// 不得被静默归一化后照常绘制。
///
/// # 缺陷 D1（QA 上一轮报告，Engineer 已修）
///
/// Rust 的 `f32 as i64` 转换对 `NaN` 返回 **0**（不是饱和、也不是 panic）。
/// 故修复 N3 之后，`blit_glyph(fb, &bmp24, NaN, 30.0, …)` 会把字形
/// **画到原点**——实测与 `pen=5.0` 的落墨分量数完全相同（均780）。
/// 这不违反 C18（不 panic），但违反**确定性铁律**的精神：无效输入
/// 被静默映射成一个**有效但错误**的绘制位置（错图）。
///
/// 修复：`blit_glyph` 入口加 `!pen_x.is_finite() || !baseline_y.is_finite()`
/// 早退，与 `render_glyph` 的 `!px_size.is_finite()` 拒绝口径对齐。
///
/// # 为何需要本测试（而非依赖 n3_2）
///
/// `n3_2` 用 8×8 视口做参考模型对账。字形画到原点时**恰好也落在 8×8 视口外**
/// （`bmp.top ≈ −22`），故 n3_2 在 D1 缺陷态下会**侥幸通过**——不能承担
/// 本职责。本测试用 64×64 大视口 + 正对照，确保非有限笔位与正常笔位
/// **可区分**。
#[test]
fn n3_6_non_finite_pen_is_rejected_not_normalized_to_origin() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let bmp = render_glyph(&f, gid, 24.0);
    assert!(!bmp.is_empty(), "前置：24px 中字应可渲染");

    // (1) 缺陷态证据：NaN 笔位被静默当成 0 → 画到原点 → 落墨。
    for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut fb = Fbuf16F::new(64, 64);
        blit_glyph(&mut fb, &bmp, bad, 30.0, [1.0; 4]);
        assert!(
            is_clean(&fb),
            "非有限笔位 {bad} 竟画出像素 ⇒ NaN/Inf 被静默归一化（D1 未修）"
        );
        let mut fb2 = Fbuf16F::new(64, 64);
        blit_glyph(&mut fb2, &bmp, 5.0, bad, [1.0; 4]);
        assert!(is_clean(&fb2), "非有限 baseline {bad} 竟画出像素");
    }

    // (2) 正对照：**有限**笔位在同样视口下必须落墨。
    //     这一条是本测试的自证非恒真锚点——若检查器恒返回 clean，
    //     (1) 就是假绿。
    let mut fb_ok = Fbuf16F::new(64, 64);
    blit_glyph(&mut fb_ok, &bmp, 5.0, 30.0, [1.0; 4]);
    assert!(
        !is_clean(&fb_ok),
        "有限笔位 (5.0, 30.0) 应落墨（若不落墨，(1) 的断言恒真 ⇒ 检查器无效）"
    );

    // (3) 跨优化级别一致性：非有限笔位在两端都必须被拒（同输入同输出）。
    let mut a = Fbuf16F::new(64, 64);
    blit_glyph(&mut a, &bmp, f32::NAN, 30.0, [0.3, 0.6, 0.9, 1.0]);
    let mut b = Fbuf16F::new(64, 64);
    blit_glyph(&mut b, &bmp, f32::NAN, 30.0, [0.3, 0.6, 0.9, 1.0]);
    assert_eq!(fingerprint(&a), fingerprint(&b), "NaN 笔位两遍输出不一致");
    assert!(is_clean(&a), "NaN 笔位应被拒绝");
}

/// N3-7：**跨优化级别**逐字节一致（N3 的真正要害）。
///
/// N3 = "debug panic / release 静默错图"——本质是**同输入在不同优化级别
/// 下输出不同**。debug 无 panic **不足以**放行；必须在 release 下产出
/// **逐字节相同**的画面。
///
/// 手法：把全穷举组合的画面指纹写进`$OUT_DIR`（debug/release 各一份），
/// 再由 `n3_8_release_matches_debug_byte_for_byte` 比对两个文件。测试自身在两端跑同一
/// 份断言逻辑，故只需比对外部产物。
#[test]
fn n3_7_dump_fingerprint_for_cross_profile_compare() {
    // `debug_assertions`正是区分 debug/release 的编译期开关，用作文件名。
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let out =
        std::env::var("PRISM_FP_OUT").unwrap_or_else(|_| format!("/tmp/prism_fp_{profile}.txt"));
    const VW: u32 = 8;
    const VH: u32 = 8;
    let dims: [(i32, i32, u32, u32); 3] =
        [(0, 0, 2, 2), (i32::MIN, i32::MAX, 1, 1), (i32::MAX, i32::MIN, 3, 1)];
    let mut fp = String::new();
    for &pen in EXTREMES {
        for &base in EXTREMES {
            for &l in OFFSETS {
                for &t in OFFSETS {
                    for &(_bl, _bt, bw, bh) in &dims {
                        let mut fb = Fbuf16F::new(VW, VH);
                        let bmp = small(l, t, bw, bh);
                        blit_glyph(&mut fb, &bmp, pen, base, [0.3, 0.6, 0.9, 1.0]);
                        fp.push_str(&format!("{pen:e}|{base:e}|{l}|{t}|{bw}|{bh}|"));
                        for &p in &fb.pixels {
                            fp.push_str(&format!("{p},"));
                        }
                        fp.push('\n');
                    }
                }
            }
        }
    }
    std::fs::write(&out, &fp).expect("写指纹失败");
    println!("[N3] 指纹已写入 {out}（{} 字节）", fp.len());
}

/// N3-8：比对 debug 与 release 两份指纹（由外部脚本在两端各跑一次后调用）。
///
/// 若`PRISM_FP_DEBUG` / `PRISM_FP_RELEASE` 未设置则跳过（单端运行）。
#[test]
fn n3_8_release_matches_debug_byte_for_byte() {
    let (Ok(dbg), Ok(rel)) = (
        std::env::var("PRISM_FP_DEBUG"),
        std::env::var("PRISM_FP_RELEASE"),
    ) else {
        println!("[N3] 未提供两份指纹路径，跳过跨优化级别比对");
        return;
    };
    let a = std::fs::read_to_string(&dbg).expect("读 debug 指纹失败");
    let b = std::fs::read_to_string(&rel).expect("读 release 指纹失败");
    assert_eq!(a, b, "debug 与 release 输出不一致 ⇒ 违反确定性铁律（跨优化级别）");
    assert!(!a.is_empty(), "指纹为空，比对无意义");
    println!("[N3] debug/release 指纹逐字节一致（{} 字节）", a.len());
}
