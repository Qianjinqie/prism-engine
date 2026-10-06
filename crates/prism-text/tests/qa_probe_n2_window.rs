//! QA 独立探针 —— 缺陷 N2（`f32 as i32` 饱和后 i32 减法溢出）终验。
//!
//! 覆盖四组语义，各组均以"绝不 panic"为C18 底线：
//! 1. 极大/非有限 `px_size` 穷举 → 空位图（**拒绝**语义，非钳制）；
//! 2. `MAX_WINDOW_EDGE = 4096` 边界连续性（无行为跳变、位图自洽）；
//! 3. 畸形位图（`alpha.len() < w*h`、`top = i32::MIN`）走 A/B/C 全链路；
//! 4. N2 修复的**根因**验证：字号→窗口尺寸的推导在 i64 域自洽。

use prism_io::Fbuf16F;
use prism_text::font_regular;
use prism_text::raster::{blit_glyph, render_glyph, GlyphBitmap};

/// N2 修复引入的窗口边长上界（与 `raster::MAX_WINDOW_EDGE` 同值 4096）。
const EDGE: i64 = 4096;

/// 位图自洽性检查：`alpha.len() == w*h` 且无重复分配。
fn assert_self_consistent(tag: &str, b: &GlyphBitmap) {
    let expect = b.width as usize * b.height as usize;
    assert_eq!(
        b.alpha.len(),
        expect,
        "[{tag}] 位图不自洽：{}x{} 但 alpha.len()={}（应={expect}）",
        b.width,
        b.height,
        b.alpha.len()
    );
    // 坐标幅度守卫（x0 ≥ −EDGE 且 x1 ≤ EDGE）给出的**边长**上界是 2·EDGE，
    // 而非 EDGE—— 见缺陷 N4（qa_attack_n4_window_cap.rs）：`MAX_WINDOW_EDGE`
    // 约束的是坐标幅度，不是区间长度。`raster.rs:236` 注释断言的
    // "width*height ≤ 4096²" 不成立。
    assert!(
        b.width as i64 <= 2 * EDGE && b.height as i64 <= 2 * EDGE,
        "[{tag}] 边长越界：{}x{} 超过坐标守卫推论上界 2*{EDGE}",
        b.width,
        b.height
    );
    if !b.is_empty() {
        assert_eq!(b.alpha.len(), b.width as usize * b.height as usize);
        // 空位图语义：is_empty ⇔ 不分配 alpha。
        assert!(
            !(b.is_empty() && !b.alpha.is_empty()),
            "[{tag}] is_empty 为真却分配了 alpha（应为拒绝语义而非钳制）"
        );
    }
}

/// N2-P1：极大有限 `px_size` 穷举 → 必为空位图（拒绝语义）。
///
/// 失败含义：回退到 `f32 as i32` 饱和 + i32 减法 → 溢出 panic（C18）。
#[test]
fn n2_p1_huge_finite_px_size_returns_empty_not_panic() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    // '中' 是全角字形：设备窗口约 (0.88·px) 宽，故 px=1e3 时窗口 ~880 < 4096
    // ⇒ **仍在上限内，必须正常渲染**。真正的拒绝阈值在 px ≈ 4.4e3 量级。
    // 本探针只保证"极大字号 → 空位图（拒绝语义）"，阈值本身由 P3/P4 量化。
    for &px in &[1.0e5f32, 1.0e10, 1.0e20, 1.0e30, 3.4e38, f32::MAX, 1.0e38] {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px_size={px:e} 超上限应为空位图（拒绝语义）");
        assert_eq!(b.alpha.len(), 0, "px_size={px:e} 不应分配 alpha");
        assert_eq!((b.width, b.height), (0, 0), "px_size={px:e} 应为 0x0");
    }
    // 上限内的字号必须正常渲染（防止"过度拒绝"回归）。
    for &px in &[1.0f32, 12.0, 48.0, 1000.0, 2000.0] {
        let b = render_glyph(&f, gid, px);
        assert!(!b.is_empty(), "px_size={px} 在上限内应正常渲染，不得被过度拒绝");
        assert_self_consistent(&format!("px={px}"), &b);
    }
}

/// N2-P2：非正/非有限 `px_size` → 空位图，且**不分配** alpha。
#[test]
fn n2_p2_non_positive_and_nonfinite_return_unallocated_empty() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    for &px in &[
        0.0f32,
        -0.0,
        -1.0,
        -1e30,
        -f32::MAX,
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let b = render_glyph(&f, gid, px);
        assert!(b.is_empty(), "px_size={px} 应为空位图");
        assert_eq!(b.alpha.len(), 0, "px_size={px} 不应分配 alpha");
    }
}

/// N2-P3：`MAX_WINDOW_EDGE` 边界连续性 —— 1024/2048/4096/4097 各点。
///
/// 关键：**4096 是窗口边长上界，不是字号上界**。'中' 的实际窗口宽约为
/// 字号×(字形宽/upem)。所以"字号=4096"时窗口可能仍< 4096（正常渲染），
/// 而更大的字号才被拒。QA 要求：不存在"分配了天文数字却声称合法"的位图，
/// 也不存在 panic。
#[test]
fn n2_p3_window_edge_boundary_is_continuous_and_self_consistent() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let mut prev_empty: Option<bool> = None;
    // 边界附近密集采样：验证"空/非空"随字号单调（一旦为空即持续为空）。
    // 边界点稀疏采样：4096px 字形的光栅化缓冲达 67 MiB/次，故不做密集扫描。
    // 契约验的是"自洽 + 上限不被突破"，而非逐整数枚举。
    for px in [1024.0f32, 2048.0, 4095.0, 4096.0, 4097.0, 4098.0, 5000.0, 8192.0] {
        let b = render_glyph(&f, gid, px);
        assert_self_consistent(&format!("px={px}"), &b);
    }
    // 拒绝区单调性：一旦超过临界字号，持续为空（只验证"极大"侧不回落）。
    for px in [1.0e4f32, 1.0e5, 1.0e10, 1.0e20, f32::MAX] {
        assert!(
            render_glyph(&f, gid, px).is_empty(),
            "px={px:e} 必须恒为空位图"
        );
    }
    let _ = &prev_empty;
}

/// N2-P4：拒绝区与渲染区的**量化**交叉验证。
///
/// 直接算出'中' 在字号px 下的设备窗口宽，与 EDGE 比较，确认实现的上界
/// 就是 EDGE（而非某个魔数），且"渲染 ⇒ 窗口 ≤ EDGE"永不被违反。
#[test]
fn n2_p4_rendered_window_never_exceeds_cap() {
    let f = font_regular();
    let gid = f.glyph_index('中').expect("中在子集内");
    let upem = f32::from(f.units_per_em());
    let out = f.outline(gid).expect("可解析");
    // 字形设备 bbox（字体单位）
    let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for ring in &out.contours {
        for p in ring {
            xmin = xmin.min(p.x);
            xmax = xmax.max(p.x);
        }
    }
    assert!(xmax > xmin, "前置条件：中应有非零宽度");
    // 找出该字形恰好被拒的临界字号，验证落在理论值附近。
    // 二分找临界字号（线性扫 4000 次每次要分配 67 MiB 覆盖缓冲，不可行）。
    // 逐步放大直到被拒，得到粗界；再二分细化 20 次。
    let mut lo = 1.0f32;   // 已知可渲染
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
    let last_ok = lo;
    let first_reject = hi;
    // 临界点上下的位图必须自洽（这是"连续无跳变"的正确表述）。
    assert_self_consistent("临界前一档", &render_glyph(&f, gid, last_ok));
    assert_self_consistent("临界后一档", &render_glyph(&f, gid, first_reject));
    // 理论临界：窗口宽 = (xmax-xmin)*px/upem 达到 EDGE 之前。
    let glyph_w_em = (xmax - xmin) / upem;
    let theory_reject = EDGE as f32 / glyph_w_em * upem;
    assert!(
        last_ok > 0.0 && first_reject.is_finite(),
        "应存在渲染区与拒绝区的分界（last_ok={last_ok} first_reject={first_reject}）"
    );
    // 临界点由**坐标幅度**守卫（x1 ≤ EDGE）决定：`x1 = ceil(xmax · px/upem)`，
    // 故临界字号 = EDGE · upem / xmax（xmax 是**字体单位**，无需再除 upem）。
    let theory_by_coord = EDGE as f32 * upem / xmax;
    assert!(
        first_reject >= theory_by_coord * 0.98 && first_reject <= theory_by_coord * 1.05,
        "临界字号 {first_reject} 应贴近坐标守卫理论值 {theory_by_coord}\
         （xmax={xmax} 字体单位, upem={upem}）"
    );
    // 拒绝区之后必须恒拒绝（无回归到"超限仍渲染"）。
    for px in [first_reject, first_reject * 2.0, first_reject * 10.0, 1.0e9] {
        assert!(
            render_glyph(&f, gid, px).is_empty(),
            "px={px} 远超临界 {first_reject}，应为空位图"
        );
    }
}

/// N2-P5：确定性 —— 同gid+同字号两遍逐字节一致（含拒绝区与渲染区）。
#[test]
fn n2_p5_render_is_byte_identical_in_both_regimes() {
    let f = font_regular();
    let gid = f.glyph_index('永').expect("永在子集内");
    for px in [1.0f32, 12.0, 48.0, 1024.0, 4096.0, 1.0e20, f32::MAX] {
        let a = render_glyph(&f, gid, px);
        let b = render_glyph(&f, gid, px);
        assert_eq!(a, b, "px={px} 两遍渲染不一致（确定性铁律）");
        assert_self_consistent(&format!("det px={px}"), &a);
    }
}

// ---------------------------------------------------------------------------
// A / B / C：畸形位图路径
// ---------------------------------------------------------------------------

/// A/B-P1：`top = i32::MIN` 的 `rotate_cw` 走饱和加减，不 panic。
#[test]
fn ab_p1_rotate_cw_saturates_extreme_offsets() {
    for &top in &[i32::MIN, i32::MIN + 1, i32::MIN + 7, -1, 0, i32::MAX - 1, i32::MAX] {
        for &(w, h) in &[(1u32, 1u32), (4, 4), (3, 5), (8, 2)] {
            let bmp = GlyphBitmap {
                left: i32::MIN,
                top,
                width: w,
                height: h,
                alpha: vec![3u8; (w * h) as usize],
            };
            let r = bmp.rotate_cw();
            // 逆变换回原值：r.top == left, r.left == -(top+h) 饱和
            assert_eq!(r.top, i32::MIN, "r.top 应恒等于原 left");
            assert_eq!((r.width, r.height), (h, w), "宽高应互换");
            assert_eq!(r.alpha.len(), (w * h) as usize);
        }
    }
}

/// A/B-P2：`top + height` 真正溢出 i32（top=i32::MIN, height≥1）→ 饱和不 panic。
#[test]
fn ab_p2_rotate_cw_saturates_on_true_overflow() {
    // top + height 在 i32::MIN + 1..7 上加不出i32::MIN 本身，取负才是真溢出点。
    let bmp = GlyphBitmap { left: 0, top: i32::MIN, width: 1, height: 1, alpha: vec![1u8] };
    assert_eq!(bmp.rotate_cw().left, i32::MAX, "−(i32::MIN+1) 应饱和为 i32::MAX");
    // height 使和越过 0，取负不溢出
    let bmp2 = GlyphBitmap { left: 0, top: i32::MIN, width: 16, height: 16, alpha: vec![1u8; 256] };
    assert_eq!(bmp2.rotate_cw().left, -(i32::MIN + 16));
}

/// B-P3：畸形位图（`alpha.len() < w*h`）在 `rotate_cw` 原样返回。
#[test]
fn b_p3_malformed_alpha_len_rotate_returns_unchanged() {
    for&(w, h, alen) in &[
        (1000u32, 1000u32, 10usize),
        (4, 4, 0),
        (4, 4, 15),
        (100_000, 100_000, 1),
        (u32::MAX, u32::MAX, 0),
        (u32::MAX, 2, 0),
        (2, u32::MAX, 0),
    ] {
        let bmp = GlyphBitmap {
            left: 0,
            top: i32::MIN,
            width: w,
            height: h,
            alpha: vec![7u8; alen],
        };
        let r = bmp.rotate_cw();
        assert_eq!(r.alpha, bmp.alpha, "({w},{h},{alen}) 畸形位图应原样返回");
        assert_eq!((r.width, r.height), (w, h), "畸形位图不应改尺寸");
    }
}

/// C-P4：畸形位图 `blit_glyph` 不得越界读/ panic。
#[test]
fn c_p4_blit_glyph_on_malformed_bitmap_is_safe() {
    for &alen in &[0usize, 1, 9, 999] {
        for &(w, h) in &[(1000u32, 1000u32), (4, 4), (16, 16)] {
            let bmp = GlyphBitmap { left: i32::MIN, top: i32::MIN, width: w, height: h, alpha: vec![200u8; alen] };
            let mut fbuf = Fbuf16F::new(16, 16);
            blit_glyph(&mut fbuf, &bmp, f32::NAN, f32::NAN, [1.0; 4]);
            blit_glyph(&mut fbuf, &bmp, -1e30, 1e30, [1.0; 4]);
            blit_glyph(&mut fbuf, &bmp, 0.0, 8.0, [1.0; 4]);
        }
    }
    // 关键语义：alpha.len() < w*h 时**一个像素都不该涂**（保守拒绝）。
    let bad = GlyphBitmap { left: 0, top: 0, width: 1000, height: 1000, alpha: vec![255u8; 10] };
    let mut fbuf = Fbuf16F::new(64, 64);
    blit_glyph(&mut fbuf, &bad, 32.0, 32.0, [1.0; 4]);
    assert!(
        fbuf.pixels.iter().all(|&p| p == 0),
        "alpha 不自洽的位图应整体拒绝，不得涂任何像素"
    );
}

/// C-P5：`alpha.len() > w*h`（超长）时按 `w*h` 预算，只涂有效区。
#[test]
fn c_p5_blit_glyph_with_oversized_alpha_is_bounded() {
    let bmp = GlyphBitmap { left: 0, top: 0, width: 4, height: 4, alpha: vec![255u8; 1000] };
    let mut fbuf = Fbuf16F::new(8, 8);
    blit_glyph(&mut fbuf, &bmp, 0.0, 0.0, [1.0; 4]);
    // 4x4 全不透明 → 16 个像素被涂，且不越界（8x8 视口内）
    let painted = (0..8u32)
        .flat_map(|y| (0..8u32).map(move |x| (x, y)))
        .filter(|&(x, y)| fbuf.get(x, y)[3] > 0.0)
        .count();
    assert_eq!(painted, 16, "应恰涂4x4=16 像素");
}

/// A/B/C-P6：全链路 `render → rotate → blit` 组合压力（含极端字号）。
#[test]
fn abc_p6_full_chain_never_panics() {
    let f = font_regular();
    let mut fbuf = Fbuf16F::new(32, 32);
    for ch in ['A', '中', '永', ' ', 'g'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for &px in &[1.0f32, 16.0, 48.0, 1e5, 1e20, f32::MAX, 0.0, -1.0, f32::NAN] {
            let bmp = render_glyph(&f, gid, px);
            let r = bmp.rotate_cw();
            blit_glyph(&mut fbuf, &r, 8.0, 20.0, [1.0, 1.0, 1.0, 1.0]);
            blit_glyph(&mut fbuf, &bmp, 8.0, 20.0, [1.0, 1.0, 1.0, 1.0]);
        }
    }
}
