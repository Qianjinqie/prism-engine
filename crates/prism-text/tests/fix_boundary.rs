//! 边界回归 —— 缺陷 N3（`blit_glyph` 偏移饱和加溢出）与
//! N4（`MAX_WINDOW_EDGE` 未约束边长）修复后的边界锁定。
//!
//! 本文件与 `tests/qa_*`（QA 独立探针，不随源码修改）互补：
//! QA 探针**证明缺陷存在**，本文件**锁定修复后的契约**，二者角色不同。
//!
//! 契约：
//! 1. 【N3】`f32 as i64` 是饱和转换（`±1e30 → i64::MIN/MAX`），
//!    `GlyphBitmap::left/top` 是 `pub` 字段（外部可置 `i32::MIN/MAX`）。
//!    偏移相加走 `saturating_add`；饱和后笔位必在视口外，被视口裁剪丢弃。
//!    debug 不 panic、release 不静默错图（同输入逐字节一致）。
//! 2. 【N4】窗口**坐标幅度**受 `MAX_COORD_EDGE`、**边长**受
//!    `MAX_WINDOW_EDGE` 双重约束 ⇒ `width ≤ 4096 && height ≤ 4096`
//!    ⇒ `width*height ≤ 4096²`（64 MiB 覆盖缓冲上界，由构造保证）。

use prism_io::Fbuf16F;
use prism_text::cache::GlyphCache;
use prism_text::font_regular;
use prism_text::layout::GlyphPos;
use prism_text::raster::{blit_glyph, render_glyph, GlyphBitmap};

/// 边长上界（与 `raster::MAX_WINDOW_EDGE` 同值，私有常量故在此复刻）。
const EDGE: i64 = 4096;

/// 极端 pen/baseline 取值集合（含 NaN/±Inf/±f32::MAX/饱和阈值）。
const EXTREMES: &[f32] = &[
    0.0,
    -0.0,
    1.0e19,
    -1.0e19,
    9.3e18,
    -9.3e18,
    1.0e30,
    -1.0e30,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// 自洽小位图（绕过 `alpha.len() < w*h` 早退，确保走到偏移计算）。
fn small() -> GlyphBitmap {
    GlyphBitmap { left: 0, top: 0, width: 2, height: 2, alpha: vec![255u8; 4] }
}

/// 断言帧缓冲全零（拒绝语义：极端偏移下不得涂任何像素）。
fn assert_clean(fb: &Fbuf16F, tag: &str) {
    assert!(fb.pixels.iter().all(|&p| p == 0), "{tag}: 极端偏移下应拒绝绘制");
}

// ---------------------------------------------------------------------------
// N3：`blit_glyph` 偏移饱和加
// ---------------------------------------------------------------------------

/// N3-A：极端 pen/left/top 全组合不 panic（debug 下即C18 底线）。
#[test]
fn n3a_blit_extreme_pen_and_offset_never_panics() {
    let mut fb = Fbuf16F::new(8, 8);
    for &px in EXTREMES {
        for &py in EXTREMES {
            for &(l, t) in &[(i32::MIN, i32::MIN), (i32::MAX, i32::MAX), (0, 0)] {
                let bmp = GlyphBitmap { left: l, top: t, ..small() };
                blit_glyph(&mut fb, &bmp, px, py, [1.0; 4]);
            }
        }
    }
}

/// N3-B：极端 pen 一律**拒绝绘制**（钳到视口外被裁剪丢弃，不静默贴边）。
#[test]
fn n3b_extreme_pen_paints_nothing() {
    let mut fb = Fbuf16F::new(8, 8);
    for &px in EXTREMES {
        for &py in EXTREMES {
            let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, ..small() };
            blit_glyph(&mut fb, &bmp, px, py, [1.0; 4]);
            assert_clean(&fb, &format!("pen=({px},{py}) left/top=i32::MIN"));
        }
    }
}

/// N3-C：量化溢出临界点——证明旧代码的裸`+`确为真溢出（修复前必panic）。
#[test]
fn n3c_overflow_threshold_was_reachable() {
    //饱和转换前置条件：|pen| ≥ 2^63 时 `as i64` 饱和到 i64::MIN/MAX。
    assert_eq!((-1.0e30f32).floor() as i64, i64::MIN);
    assert_eq!((1.0e30f32).floor() as i64, i64::MAX);
    //旧代码 `i64::MIN + i32::MIN` / `i64::MAX + i32::MAX` 均溢出。
    assert!(((-1.0e30f32).floor() as i64).checked_add(i32::MIN as i64).is_none());
    assert!(((1.0e30f32).floor() as i64).checked_add(i32::MAX as i64).is_none());
    // 修复后同语义走饱和加，值可预测（端点行为被锁定）。
    assert_eq!(i64::MIN.saturating_add(i32::MIN as i64), i64::MIN);
    assert_eq!(i64::MAX.saturating_add(i32::MAX as i64), i64::MAX);
}

/// N3-D：正常路径未回归（偏移语义仍正确，饱和加不吞掉有效偏移）。
#[test]
fn n3d_normal_offset_semantics_preserved() {
    let mut fb = Fbuf16F::new(8, 8);
    // left=-1, top=-1, 2x2 不透明，pen=(2,2) →覆盖 (1,1)..(2,2)。
    let bmp = GlyphBitmap { left: -1, top: -1, width: 2, height: 2, alpha: vec![255u8; 4] };
    blit_glyph(&mut fb, &bmp, 2.0, 2.0, [1.0; 4]);
    let painted = (0..8u32)
        .flat_map(|y| (0..8u32).map(move |x| (x, y)))
        .filter(|&(x, y)| fb.get(x, y)[3] > 0.0)
        .count();
    assert_eq!(painted, 4, "2x2 位图应恰涂 4 像素");
    assert!(fb.get(1, 1)[3] > 0.0, "(1,1) 应被涂");
    assert!(fb.get(3, 3)[3] == 0.0, "(3,3) 不应被涂");
    // pen 向负方向取整：pen=2.7 → floor=2（亚像素笔位不做重采样）。
    let mut fb2 = Fbuf16F::new(8, 8);
    blit_glyph(&mut fb2, &bmp, 2.7, 2.2, [1.0; 4]);
    assert!(fb2.get(1, 1)[3] > 0.0, "pen 应向下取整对齐像素格");
}

/// N3-E：`draw_glyphs` 上游全链路在极端 dx/dy 下不 panic（C18 端到端）。
#[test]
fn n3e_draw_glyphs_extreme_delta_never_panics() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中 在子集内");
    let mut cache = GlyphCache::new();
    let mut fb = Fbuf16F::new(16, 16);
    for &d in EXTREMES {
        prism_text::raster::draw_glyphs(
            &mut fb,
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

/// N3-F：确定性 —— 同输入两遍逐字节一致（N3 的要害是release 曾静默错图，
/// 故此项须在 debug 与 release 下都成立）。
#[test]
fn n3f_blit_is_deterministic_byte_wise() {
    let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, width: 3, height: 3, alpha: vec![200u8; 9] };
    let mut a = Fbuf16F::new(8, 8);
    let mut b = Fbuf16F::new(8, 8);
    for &p in EXTREMES {
        for &q in EXTREMES {
            blit_glyph(&mut a, &bmp, p, q, [0.5, 0.25, 1.0, 1.0]);
            blit_glyph(&mut b, &bmp, p, q, [0.5, 0.25, 1.0, 1.0]);
        }
    }
    assert_eq!(a.pixels, b.pixels, "同输入两遍必须逐字节一致");
}

// ---------------------------------------------------------------------------
// N4：窗口边长上界
// ---------------------------------------------------------------------------

/// 位图自洽性：`alpha.len() == width*height`（拒绝语义下二者同为 0）。
fn assert_self_consistent(tag: &str, b: &GlyphBitmap) {
    assert_eq!(
        b.alpha.len(),
        b.width as usize * b.height as usize,
        "[{tag}] 位图不自洽：{}x{}但 alpha.len()={}",
        b.width,
        b.height,
        b.alpha.len()
    );
}

/// N4-A：边长上界 —— 任意字号下`width ≤ EDGE && height ≤ EDGE`，
/// 推论 `width*height ≤ EDGE²`（64 MiB 覆盖缓冲上界由构造保证）。
#[test]
fn n4a_rendered_edge_never_exceeds_cap() {
    let f = font_regular();
    for ch in ['中', '国', 'A', 'W', 'M', '永'] {
        let gid = f.glyph_index(ch).expect("字形在子集内");
        for &px in &[4000.0f32, 4096.0, 4097.0, 4500.0, 8192.0, 10000.0] {
            let b = render_glyph(&f, gid, px);
            let tag = format!("ch={ch} px={px}");
            assert_self_consistent(&tag, &b);
            assert!(
                b.width as i64 <= EDGE && b.height as i64 <= EDGE,
                "[{tag}] 边长越界：{}x{} > {EDGE}（N4 未修）",
                b.width,
                b.height
            );
            assert!(
                (b.width as u64) * (b.height as u64) <= (EDGE as u64) * (EDGE as u64),
                "[{tag}] 面积越界：{}x{}",
                b.width,
                b.height
            );
        }
    }
}

/// N4-B：边界连续性 —— 边长恰好 ≤ EDGE 通过、超出即拒绝，
/// 且拒绝侧**不分配 alpha**（无半吊子图）。
#[test]
fn n4b_edge_cap_boundary_is_continuous() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中 在子集内");
    //二分找该字形的边长临界字号：临界处 width/height 应 == EDGE。
    let mut lo = 1.0f32; // 已知可渲染
    let mut hi = 1.0e5f32; // 已知被拒
    assert!(!render_glyph(&f, gid, lo).is_empty(), "前置：1px 应可渲染");
    assert!(render_glyph(&f, gid, hi).is_empty(), "前置：1e5px 应被拒");
    for _ in 0..24 {
        let mid = (lo * hi).sqrt(); // 几何二分（跨数量级）
        if render_glyph(&f, gid, mid).is_empty() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let last_ok = render_glyph(&f, gid, lo);
    let first_reject = render_glyph(&f, gid, hi);
    assert_self_consistent("临界前一档", &last_ok);
    assert_self_consistent("临界后一档", &first_reject);
    // 临界前一档：边长必≤ EDGE；且已贴近 EDGE（否则边长守卫不是有效约束）。
    let max_edge = (last_ok.width as i64).max(last_ok.height as i64);
    assert!(max_edge <= EDGE, "临界前一档边长 {max_edge} 不得超过 {EDGE}");
    assert!(
        max_edge as f64 >= EDGE as f64 * 0.9,
        "临界前一档边长 {max_edge} 应贴近 {EDGE}——边长守卫未生效？"
    );
    // 临界后一档：完全拒绝，且不分配 alpha（无半吊子图）。
    assert!(first_reject.is_empty(), "临界后一档应为空位图");
    assert_eq!(first_reject.alpha.len(), 0, "拒绝时不应分配 alpha");
    // 拒绝区单调：一旦超限即持续为空。
    for px in [hi, hi * 2.0, hi * 10.0, 1.0e9, f32::MAX] {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px={px:e} 远超临界 {hi}，应恒为空位图");
    }
}

/// N4-C：坐标幅度守卫仍独立有效（防"改成边长守卫后越界变全拒"）。
#[test]
fn n4c_coord_guard_still_rejects_extremes() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中 在子集内");
    for &px in &[1.0e4f32, 1.0e5, 1.0e10, 1.0e20, f32::MAX] {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px={px:e} 应被坐标/边长守卫拒绝");
        assert_eq!(b.alpha.len(), 0, "px={px:e} 不应分配 alpha");
    }
}

/// N4-D：确定性 —— 超限区间亦逐字节可复现（无隐藏状态）。
#[test]
fn n4d_render_is_deterministic_at_edges() {
    let f = font_regular();
    let gid = f.glyph_index('永').expect("永 在子集内");
    for &px in &[3000.0f32, 4096.0, 4500.0, 8192.0] {
        let a = render_glyph(&f, gid, px);
        let b = render_glyph(&f, gid, px);
        assert_eq!(a, b, "px={px} 两遍渲染不一致");
        assert_self_consistent(&format!("px={px}"), &a);
    }
}
