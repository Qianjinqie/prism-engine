//! QA 攻击探针 —— 新缺陷 N3：`blit_glyph` 的 `ox`/`oy` 计算溢出 panic（C18）。
//!
//! 定位：`crates/prism-text/src/raster.rs:277-278`
//! ```ignore
//! let ox = pen_x.floor() as i64 + bmp.left as i64;   // <- 裸 `+`
//! let oy = baseline_y.floor() as i64 + bmp.top as i64;
//! ```
//!
//! `f32 as i64` 是**饱和**转换：`1e30 as i64 == i64::MAX`、`-1e30 as i64 ==
//! i64::MIN`。故`i64::MAX + bmp.left`（`left` 是 `pub` 字段，外部可置
//! `i32::MIN`）即溢出 → debug panic。修复 A 只覆盖了 `rotate_cw` 的偏移，
//! `blit_glyph` 的同型表达式**漏修**。
//!
//! 本文件是**最小复现 + 边界量化**，不含大位图（避免 67 MiB 分配，跑得快）。

use prism_io::Fbuf16F;
use prism_text::raster::{blit_glyph, GlyphBitmap};

/// 自洽小位图（绕过 `alpha.len() < w*h` 的早退分支，确保走到偏移计算）。
fn small() -> GlyphBitmap {
    GlyphBitmap { left: 0, top: 0, width: 2, height: 2, alpha: vec![255u8; 4] }
}

/// N3-1：极大 pen_x / baseline_y + 负 left/top → 必须不 panic。
///
/// 失败含义（= 缺陷成立）：`attempt to add with overflow` at raster.rs:277。
#[test]
fn n3_1_blit_glyph_extreme_pen_offset_does_not_panic() {
    let mut fbuf = Fbuf16F::new(8, 8);
    let extremes: &[f32] = &[
        f32::MAX,
        f32::MIN,
        1.0e30,
        -1.0e30,
        1.0e19,
        -1.0e19,
        9.3e18,
        -9.3e18,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
    ];
    for &px in extremes {
        for &py in extremes {
            // left/top 取最负的 i32，构造"饱和 i64 + 负 i32"的真溢出组合。
            let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, ..small() };
            blit_glyph(&mut fbuf, &bmp, px, py, [1.0; 4]);
            let bmp2 = GlyphBitmap { left: i32::MAX, top: i32::MAX, ..small() };
            blit_glyph(&mut fbuf, &bmp2, px, py, [1.0; 4]);
            let bmp3 = GlyphBitmap { left: 0, top: 0, ..small() };
            blit_glyph(&mut fbuf, &bmp3, px, py, [1.0; 4]);
        }
    }
}

/// N3-2：量化溢出临界点——证明这是**真实溢出**而非理论担忧。
///
/// `f32 as i64` 饱和到 i64::MIN/MAX（|pen| ≥ 2^63）；再与 `i32::MIN/MAX`
/// 偏移相加即溢出。debug 构建 panic，release 静默回绕成"贴边绘制"（错图）。
#[test]
fn n3_2_overflow_threshold_is_reachable() {
    // 确认饱和转换行为（Rust 语义：浮点→整数饱和，不 UB）。
    assert_eq!(1.0e30f64 as i64, i64::MAX, "f64→i64 应饱和到 i64::MAX");
    assert_eq!(-1.0e30f64 as i64, i64::MIN, "f64→i64 应饱和到 i64::MIN");
    // 负向饱和 + 负偏移 ⇒ 下溢（这就是 raster.rs:277执行的运算）。
    let neg_sat = -1.0e30f32.floor() as i64;
    assert_eq!(neg_sat, i64::MIN, "负向 pen 饱和到 i64::MIN");
    assert!(
        neg_sat.checked_add(i32::MIN as i64).is_none(),
        "i64::MIN + i32::MIN 确实下溢——源码用裸 `+` 故 panic"
    );
    // 正向饱和 + 正偏移 ⇒ 上溢。
    assert!(
        (1.0e30f32.floor() as i64).checked_add(i32::MAX as i64).is_none(),
        "i64::MAX + i32::MAX 确实上溢"
    );
    // 端到端：真实调用 blit_glyph 必须不 panic。
    let mut fbuf = Fbuf16F::new(8, 8);
    let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, ..small() };
    blit_glyph(&mut fbuf, &bmp, -1.0e30, -1.0e30, [1.0; 4]);
}

/// N3-3：溢出时**不得涂任何像素**（保守拒绝语义）。
#[test]
fn n3_3_extreme_offset_paints_nothing() {
    let mut fbuf = Fbuf16F::new(8, 8);
    let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, ..small() };
    blit_glyph(&mut fbuf, &bmp, 1.0e30, -1.0e30, [1.0; 4]);
    blit_glyph(&mut fbuf, &bmp, -1.0e30, 1.0e30, [1.0; 4]);
    assert!(
        fbuf.pixels.iter().all(|&p| p == 0),
        "极端偏移下应拒绝绘制，不得涂任何像素"
    );
}

/// N3-4：`draw_glyphs` 上游路径同样安全（dx/dy 传极端值）。
#[test]
fn n3_4_draw_glyphs_with_extreme_delta_does_not_panic() {
    use prism_text::cache::GlyphCache;
    use prism_text::layout::GlyphPos;
    let f = prism_text::font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let mut cache = GlyphCache::new();
    let mut fbuf = Fbuf16F::new(16, 16);
    for &d in &[1.0e30f32, -1.0e30, f32::MAX, f32::MIN, f32::NAN, 1.0e19] {
        prism_text::raster::draw_glyphs(
            &mut fbuf,
            &f,
            &[GlyphPos { ch: '中', gid, x: 0.0, y: 10.0, advance: 32.0, line: 0, index: 0 }],
            32.0,
            [1.0; 4],
            d,
            d,
            &mut cache,
        );
    }
}

/// N3-5：正常路径未回归（偏移语义仍正确）。
#[test]
fn n3_5_normal_offset_semantics_preserved() {
    let mut fbuf = Fbuf16F::new(8, 8);
    // left=-1, top=-1, 2x2 全不透明，pen=(2,2) → 覆盖 (1,1)..(2,2)
    let bmp = GlyphBitmap { left: -1, top: -1, width: 2, height: 2, alpha: vec![255u8; 4] };
    blit_glyph(&mut fbuf, &bmp, 2.0, 2.0, [1.0; 4]);
    let painted = (0..8u32)
        .flat_map(|y| (0..8u32).map(move |x| (x, y)))
        .filter(|&(x, y)| fbuf.get(x, y)[3] > 0.0)
        .count();
    assert_eq!(painted, 4, "2x2 位图应恰涂 4 像素");
    assert!(fbuf.get(1, 1)[3] > 0.0, "左上角 (1,1) 应被涂");
    assert!(fbuf.get(2, 2)[3] > 0.0, "右下角 (2,2) 应被涂");
    assert!(fbuf.get(3, 3)[3] == 0.0, "(3,3) 不应被涂");
}
