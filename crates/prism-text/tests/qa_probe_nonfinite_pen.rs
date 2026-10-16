//! QA 独立探针 —— 缺陷 D1（`blit_glyph` 静默把 NaN/Inf 笔位归一化为原点）
//! **独立复核**。不复用任何既有测试。
//!
//! # 缺陷回顾（QA 上一轮报告，D1）
//!
//! Rust 语义：`f32 as i64` 对 `NaN` 返回 **0**（不是饱和、不是 panic）。
//! 故`blit_glyph` 的 `(pen_x.floor() as i64)` 在笔位为 `NaN` 时得到 `0`，
//! 字形被**照常绘制在原点**——不 panic，但输出一幅位置完全错误的图。
//! 违反确定性铁律的精神（错图比不画更糟：下游无法区分"画错了"与"画在这里"）。
//!
//! 实测证据（修复前）：`blit_glyph(fb, &bmp24, f32::NAN, 30.0, ..)`
//! 涂 **780** 个分量，与 `pen=5.0`（合法笔位）的 **780** **完全相同**
//! ⇒ 证明 NaN 被静默当成 0。
//!
//! 传播源：`layout.rs:228` `baseline = li as f32 * lh + asc`，当
//! `font_size ≥ 3e38` 时 `lh = inf`，`0 * inf = NaN` ⇒ 布局产出 NaN 笔位。
//!
//! # 本次复核清单
//!
//! 1. NaN / ±Inf 笔位必须**完全拒绝**（画面全零），且与 `render_glyph`
//!    的 `!px_size.is_finite()` 拒绝口径**一致**；
//! 2. 合法笔位**必须照常绘制**（守卫不得把正常路径也杀掉）；
//! 3. `draw_glyphs` 全链路在 NaN delta 下同样拒绝；
//! 4. 跨优化级别逐字节一致（debug 不 panic、release 不错图，二者同结果）；
//! 5. **自证非恒真**：正常笔位落墨、非有限笔位不落墨，二者共存。

use prism_io::Fbuf16F;
use prism_text::cache::GlyphCache;
use prism_text::font_regular;
use prism_text::layout::GlyphPos;
use prism_text::raster::{blit_glyph, draw_glyphs, render_glyph};

/// 非有限笔位全集（NaN / ±Inf 是本缺陷的核心触发源）。
const NON_FINITE: &[f32] = &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

/// 有限但极端的笔位（**必须照常处理**，不得被守卫误杀）。
const FINITE_EXTREME: &[f32] =
    &[0.0, -0.0, 1.0, -1.0, 1.0e19, -1.0e19, 1.0e30, -1.0e30, f32::MAX, f32::MIN];

fn painted(fb: &Fbuf16F) -> usize {
    fb.pixels.iter().filter(|&&p| p != 0).count()
}

/// D1-1：`blit_glyph` 对 NaN/±Inf 笔位**完全拒绝**（不得画在原点）。
#[test]
fn d1_1_blit_rejects_non_finite_pen() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let bmp = render_glyph(&f, gid, 24.0);
    assert!(!bmp.is_empty(), "前置：24px 中字应可渲染");

    for &pen in NON_FINITE {
        for &base in NON_FINITE {
            // pen 与 base 的四种组合，含"一个正常一个非正常"。
            for &(p, b) in &[(pen, 30.0f32), (5.0, pen), (pen, base)] {
                let mut fb = Fbuf16F::new(64, 64);
                blit_glyph(&mut fb, &bmp, p, b, [1.0; 4]);
                assert_eq!(
                    painted(&fb),
                    0,
                    "pen={p:e} base={b:e} 竟涂了像素 ⇒ NaN/Inf 笔位未被拒绝（D1 未修）"
                );
            }
        }
    }
}

/// D1-2：**正常笔位必须照常绘制**（守卫不得误杀合法路径）。
///
/// 这是 D1-1 的对照：若守卫写成"任何异常都拒绝"或阈值过严，此处转红。
#[test]
fn d1_2_finite_pen_still_draws() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let bmp = render_glyph(&f, gid, 24.0);
    let mut fb = Fbuf16F::new(64, 64);
    blit_glyph(&mut fb, &bmp, 20.0, 40.0, [1.0; 4]);
    let n = painted(&fb);
    assert!(n > 0, "合法笔位必须落墨（D1 修复过严，误杀了正常路径）");
    println!("[D1] 合法笔位落墨 {n} 个分量");
}

/// D1-3：有限极端笔位不得引发 panic，且行为符合视口裁剪语义。
#[test]
fn d1_3_finite_extreme_pen_never_panics() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let bmp = render_glyph(&f, gid, 24.0);
    for &pen in FINITE_EXTREME {
        for &base in FINITE_EXTREME {
            let mut fb = Fbuf16F::new(64, 64);
            blit_glyph(&mut fb, &bmp, pen, base, [1.0; 4]);
            // 饱和语义：偏移远在视口外 ⇒ 必然全零（不 panic 即达标）。
            // 不在此断言全零，避免与"有限值不该被拒"的语义冲突。
        }
    }
    // 视口内的小偏移必须落墨（证明上面的"全零"是裁剪所致，不是被拒）。
    let mut fb = Fbuf16F::new(64, 64);
    blit_glyph(&mut fb, &bmp, 1.0, 30.0, [1.0; 4]);
    assert!(painted(&fb) > 0, "有限小偏移应落墨（证明裁剪而非拒绝）");
}

/// D1-4：`draw_glyphs` 全链路对 NaN delta 拒绝。
///
/// `draw_glyphs` 内部 `g.x + dx`，若 `dx` 为 NaN 则笔位为 NaN。
#[test]
fn d1_4_draw_glyphs_rejects_non_finite_delta() {
    let f = font_regular();
    let glyphs = vec![GlyphPos {
        ch: '中',
        gid: f.glyph_index('中').expect("中"),
        x: 20.0,
        y: 30.0,
        advance: 24.0,
        line: 0,
        index: 0,
    }];
    for &d in NON_FINITE {
        let mut fb = Fbuf16F::new(64, 64);
        let mut cache = GlyphCache::new();
        draw_glyphs(&mut fb, &f, &glyphs, 24.0, [1.0; 4], d, d, &mut cache);
        assert_eq!(painted(&fb), 0, "delta={d:e} 竟落墨 ⇒ NaN delta 未被拒绝");
    }
    // 对照：有限 delta 正常绘制。
    let mut fb = Fbuf16F::new(64, 64);
    let mut cache = GlyphCache::new();
    draw_glyphs(&mut fb, &f, &glyphs, 24.0, [1.0; 4], 0.0, 0.0, &mut cache);
    assert!(painted(&fb) > 0, "有限 delta 应落墨（对照）");
}

/// D1-5：拒绝语义与 `render_glyph` **口径一致**。
///
/// `render_glyph` 对 `!px_size.is_finite()` 返回空位图；
/// `blit_glyph` 对非有限笔位早退。两者都是"视为无效输入、不产出"。
#[test]
fn d1_5_rejection_semantics_match_render_glyph() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    // render_glyph 侧：非有限 px_size → 空位图。
    for &px in NON_FINITE {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px={px:e} 应为空位图");
        assert_eq!(b.alpha.len(), 0, "px={px:e} 不应分配 alpha");
    }
    // blit_glyph 侧：非有限笔位 → 画面全零（已在 D1-1 覆盖，此处锁语义等价）。
    let bmp = render_glyph(&f, gid, 24.0);
    for &pen in NON_FINITE {
        let mut fb = Fbuf16F::new(64, 64);
        blit_glyph(&mut fb, &bmp, pen, 30.0, [1.0; 4]);
        assert_eq!(painted(&fb), 0, "pen={pen:e} 应拒绝");
    }
}

/// D1-6：**跨优化级别逐字节一致**（要害是"错图"，须两端同结果）。
///
/// 手法：把非有限 + 有限笔位下的完整画面指纹写进文件，由外部脚本在
/// debug/release 各跑一次后比对。
#[test]
fn d1_6_dump_fingerprint_for_cross_profile_compare() {
    let out = std::env::var("PRISM_D1_FP_OUT").unwrap_or_else(|_| {
        let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
        format!("/tmp/prism_d1_fp_{profile}.txt")
    });
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let bmp = render_glyph(&f, gid, 24.0);
    let mut fp = String::new();
    let mut all: Vec<f32> = NON_FINITE.to_vec();
    all.extend_from_slice(FINITE_EXTREME);
    all.push(20.0); // 正常笔位
    for &pen in &all {
        let mut fb = Fbuf16F::new(64, 64);
        blit_glyph(&mut fb, &bmp, pen, 30.0, [0.3, 0.6, 0.9, 1.0]);
        fp.push_str(&format!("{pen:e}|"));
        for &p in &fb.pixels {
            fp.push_str(&format!("{p},"));
        }
        fp.push('\n');
    }
    std::fs::write(&out, &fp).expect("写指纹失败");
    println!("[D1] 指纹已写入 {out}（{} 字节）", fp.len());
}

/// D1-7：比对 debug/release 两份指纹（未提供路径则跳过）。
#[test]
fn d1_7_release_matches_debug_byte_for_byte() {
    let (Ok(dbg), Ok(rel)) = (
        std::env::var("PRISM_D1_FP_DEBUG"),
        std::env::var("PRISM_D1_FP_RELEASE"),
    ) else {
        println!("[D1] 未提供两份指纹路径，跳过");
        return;
    };
    let a = std::fs::read_to_string(&dbg).expect("读 debug 指纹失败");
    let b = std::fs::read_to_string(&rel).expect("读 release 指纹失败");
    assert_eq!(a, b, "debug 与 release 输出不一致 ⇒ 违反确定性铁律");
    assert!(!a.is_empty(), "指纹为空，比对无意义");
    println!("[D1] debug/release 指纹逐字节一致（{} 字节）", a.len());
}
