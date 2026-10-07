//! QA 独立探针 —— N4 契约修复的**结构性**验证（守卫独立性 / 边界连续性 /
//! 注释一致性），并对`layout.rs` 累加链的 f32 溢出传播做独立复核。
//!
//! 本文件**不复用**工程师的 `fix_boundary.rs`，全部断言由QA 独立推导。
//!
//! 方法论：两个守卫（坐标幅度 `MAX_COORD_EDGE` / 边长 `MAX_WINDOW_EDGE`）
//! 都是私有常量，无法直接读出。QA 用**黑盒差分**证明二者独立生效：
//! 依据公开的 `Font::outline` + `units_per_em` 在测试侧**复刻**源码
//! `raster.rs:236-256` 的守卫算术，判定"该输入本应被哪个守卫拦"，再与
//! `render_glyph` 的**实际行为**对账。若二者耦合（改一个坏另一个），
//! 对账必然出现不一致 → 测试转红。

use prism_text::font_regular;
use prism_text::raster::render_glyph;

/// 与实现同源的坐标幅度上界（`raster::MAX_COORD_EDGE` 私有）。
const CAP: i64 = 4096;

fn self_consistent(b: &prism_text::raster::GlyphBitmap) -> bool {
    b.alpha.len() == b.width as usize * b.height as usize
}

/// QA 侧复刻的守卫判定（严格照抄 `raster.rs:236-256` 的 i64 域算术）。
///
/// 返回 `(是否应渲染, 应有width, 应有 height, 拦它的守卫)`。
/// **注意**：`width/height` 是**过坐标守卫之后**的边长，即使被边长守卫
/// 拒绝也照实携带（= "本应边长"），供边界连续性测试判别钳制 vs 拒绝。
#[derive(Debug, PartialEq)]
struct Verdict {
    should_render: bool,
    width: i64,
    height: i64,
    guard: &'static str,
}

/// 复刻 `render_glyph` 的窗口计算 + 双守卫（源码 `raster.rs:236-256`）。
fn replicate_guards(f: &prism_text::ttf::Font, gid: u16, px: f32) -> Verdict {
    if !(px > 0.0) || !px.is_finite() {
        return Verdict { should_render: false, width: 0, height: 0, guard: "nonfinite_or_nonpositive" };
    }
    let Ok(outline) = f.outline(gid) else {
        return Verdict { should_render: false, width: 0, height: 0, guard: "outline_err" };
    };
    if outline.contours.is_empty() {
        return Verdict { should_render: false, width: 0, height: 0, guard: "no_contours" };
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
    if x0 < -CAP || x1 > CAP || y0 < -CAP || y1 > CAP {
        return Verdict { should_render: false, width: 0, height: 0, guard: "coord_guard" };
    }
    let width_i = (x1 - x0).max(0);
    let height_i = (y1 - y0).max(0);
    if width_i > CAP || height_i > CAP {
        // 携带本应边长（供边界测试判别"钳制"与"拒绝"）。
        return Verdict { should_render: false, width: width_i, height: height_i, guard: "edge_guard" };
    }
    if width_i == 0 || height_i == 0 {
        return Verdict { should_render: false, width: 0, height: 0, guard: "degenerate" };
    }
    Verdict { should_render: true, width: width_i, height: height_i, guard: "none" }
}

// ---------------------------------------------------------------------------
// N4-A：双守卫**独立生效**（黑盒差分对账）
// ---------------------------------------------------------------------------

/// N4-A：`render_glyph` 实际行为与 QA 复刻的守卫判定**逐点一致**。
///
/// 这是"两个守卫都真在生效"的证据：若QA 复刻说"该被 edge_guard 拦"而
/// 实际却渲染出了超限位图，或反之，断言立即失败。
#[test]
fn n4_a_actual_behavior_matches_replicated_guards() {
    let f = font_regular();
    let mut checked = 0usize;
    for ch in ['中', '国', 'A', 'W', 'M', 'i', '永', '一'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=90 {
            let px = 3000.0 + (step as f32) * 60.0; // 3000..8400，跨两个守卫的交界
            let v = replicate_guards(&f, gid, px);
            let b = render_glyph(&f, gid, px);
            checked += 1;
            if v.should_render {
                assert!(
                    !b.is_empty(),
                    "ch={ch} px={px} QA 判定应渲染（守卫={}，{}x{}）但实现返回空位图",
                    v.guard,
                    v.width,
                    v.height
                );
                assert_eq!(b.width as i64, v.width, "ch={ch} px={px} width 不符");
                assert_eq!(b.height as i64, v.height, "ch={ch} px={px} height 不符");
                assert!(b.width as i64 <= CAP && b.height as i64 <= CAP, "边长越界");
                assert!(self_consistent(&b), "ch={ch} px={px} 位图不自洽");
            } else {
                assert!(
                    b.is_empty(),
                    "ch={ch} px={px} QA 判定应被 {} 拒绝，但实现渲染出 ({}x{})",
                    v.guard,
                    b.width,
                    b.height
                );
                assert_eq!(b.alpha.len(), 0, "ch={ch} px={px} 拒绝时不应分配 alpha");
            }
        }
    }
    assert!(checked > 400, "对账点数 {checked} 低于预期");
}

/// N4-B：找到"坐标合法但**边长超限**"的输入 ⇒ 证明边长守卫**独立**生效。
///
/// 若边长守卫不存在（或被坐标守卫吞掉），这类输入会被渲染成超限位图。
#[test]
fn n4_b_edge_guard_fires_independently_of_coord_guard() {
    let f = font_regular();
    let mut found_edge_only = 0usize;
    for ch in ['中', '国', 'A', 'W', 'M', 'i', '永', '一', 'g', 'y'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=400 {
            let px = 3000.0 + (step as f32) * 8.0; // 3000..6200，细扫
            let v = replicate_guards(&f, gid, px);
            if v.guard == "edge_guard" {
                found_edge_only += 1;
                // 坐标守卫已放行（否则QA 复刻不会算到边长这一步），
                // 但边长超限 ⇒ 必须拒绝，且不留半吊子图。
                let b = render_glyph(&f, gid, px);
                assert!(b.is_empty(), "ch={ch} px={px} 边长超限应被拒（edge_guard）");
                assert_eq!(b.alpha.len(), 0, "ch={ch} px={px} 拒绝时不应分配 alpha 缓冲");
            }
        }
    }
    assert!(
        found_edge_only > 0,
        "未找到「坐标合法但边长超限」的输入——边长守卫是否真独立生效？\
         （若始终由坐标守卫先拦，本测试无法证明独立性）"
    );
    println!("[N4] 边长守卫独立命中 {found_edge_only} 次");
}

/// N4-C：找到"边长合法但**坐标超限**"的输入 ⇒ 证明坐标守卫**独立**生效。
#[test]
fn n4_c_coord_guard_fires_independently_of_edge_guard() {
    let f = font_regular();
    let mut found_coord_only = 0usize;
    for ch in ['中', '国', 'A', 'W', 'M', 'i', '永', '一'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for &px in &[1.0e4f32, 1.0e5, 1.0e10, 1.0e20, 1.0e30, 3.0e38, f32::MAX] {
            let v = replicate_guards(&f, gid, px);
            if v.guard == "coord_guard" {
                found_coord_only += 1;
                let b = render_glyph(&f, gid, px);
                assert!(b.is_empty(), "ch={ch} px={px:e} 坐标超限应被拒");
                assert_eq!(b.alpha.len(), 0, "ch={ch} px={px:e} 拒绝时不应分配 alpha");
            }
        }
    }
    assert!(found_coord_only > 0, "未找到坐标守卫独立命中的输入");
}

// ---------------------------------------------------------------------------
// N4-D：边界连续性 —— 4096 恰好通过 / 4097 拒绝
// ---------------------------------------------------------------------------

/// N4-D：边界连续性 —— 接受/拒绝的前沿**恰好**落在边长 `CAP` 上。
///
/// 不依赖"恰好存在px 使边长 == 4096"（扫不到则测试无意义）。改用
/// **前沿搜索**：对每个字形细扫找到"最大仍被接受的 px" `p_ok` 与
/// "最小被拒的 px" `p_rej`，然后断言：
/// 1. `p_ok` 处实际渲染成功且自洽，边长 `≤ CAP`；
/// 2. `p_rej` 处实际被**干净拒绝**（空位图 +不分配 alpha）；
/// 3. 前沿两侧的"应有边长"满足 `edge(p_ok) ≤ CAP < edge(p_rej)`
///    —— 即拒绝恰好发生在越界的第一处，无"半吊子"（既非全拒也非全放）。
///
/// 第3 条是"边界连续性"的实质：若实现是钳制（clamp）而非拒绝，
/// `p_rej` 处会渲染出被截断的位图 ⇒ 转红。
#[test]
fn n4_d_edge_boundary_continuity_at_cap() {
    let f = font_regular();
    let mut verified = 0usize;
    let mut edge_frontier = 0usize;
    for ch in ['中', '国', 'W', 'M', '永', 'A'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        // 粗扫定位前沿（用 oracle 判定，避免每点都分配 64 MiB 位图）。
        let mut p_ok: Option<f32> = None;
        let mut p_rej: Option<f32> = None;
        for step in 0..=3000 {
            let px = 4200.0 + (step as f32) * 0.5;
            let v = replicate_guards(&f, gid, px);
            if v.should_render {
                p_ok = Some(px);
            } else {
                p_rej = Some(px);
                break;
            }
        }
        let (Some(p_ok), Some(p_rej)) = (p_ok, p_rej) else { continue };

        // 1) 前沿内侧：接受 + 自洽 + 边长不越界。
        let b_ok = render_glyph(&f, gid, p_ok);
        assert!(!b_ok.is_empty(), "ch={ch} px={p_ok} 前沿内侧应被接受");
        assert!(self_consistent(&b_ok), "ch={ch} px={p_ok} 位图不自洽");
        assert!(
            b_ok.width as i64 <= CAP && b_ok.height as i64 <= CAP,
            "ch={ch} px={p_ok} 接受侧边长 ({},{}) 竟越界",
            b_ok.width,
            b_ok.height
        );
        // 2) 前沿外侧：干净拒绝（钳制实现会在此画半吊子图）。
        let b_rej = render_glyph(&f, gid, p_rej);
        assert!(b_rej.is_empty(), "ch={ch} px={p_rej} 越界侧应被拒绝（不得钳制）");
        assert_eq!(b_rej.alpha.len(), 0, "ch={ch} px={p_rej} 拒绝时不应分配 alpha");
        // 3) 前沿确由**某个**守卫造成，且无过度拒绝 / 无越界放行。
        //    注意：外前沿由哪个守卫命中取决于字形长宽比——宽字形（如 '国'）
        //    的 x1 更早冲出坐标幅度 ⇒ coord_guard 先拦；高字形（如 '中'）
        //    则由 edge_guard 拦。两者都是合法契约，故此处只要求
        //    "必为edge_guard 或 coord_guard"（而非 degenerate/无守卫）。
        let v_ok = replicate_guards(&f, gid, p_ok);
        let v_rej = replicate_guards(&f, gid, p_rej);
        assert!(v_ok.should_render, "ch={ch} px={p_ok} oracle 应判可渲染");
        assert!(
            v_rej.guard == "edge_guard" || v_rej.guard == "coord_guard",
            "ch={ch} px={p_rej} 前沿应由edge_guard/coord_guard 造成，实为 {}",
            v_rej.guard
        );
        let e_ok = v_ok.width.max(v_ok.height);
        assert!(
            e_ok <= CAP,
            "ch={ch} px={p_ok} 接受侧边长 {e_ok} > {CAP} ⇒ 越界放行"
        );
        if v_rej.guard == "edge_guard" {
            let e_rej = v_rej.width.max(v_rej.height); // 被拒侧的"本应边长"
            assert!(
                e_rej > CAP,
                "ch={ch} px={p_rej} 被edge_guard 拒但本应边长 {e_rej} ≤ {CAP} ⇒ 过度拒绝"
            );
            edge_frontier += 1;
        }
        verified += 1;
    }
    assert!(
        verified >= 3,
        "仅 {verified} 个字形验证了边界连续性，覆盖不足（前沿可能不可达）"
    );
    assert!(
        edge_frontier > 0,
        "无任何字形的前沿由 edge_guard 造成 ⇒ 边界连续性未真正触及边长守卫"
    );
    println!(
        "[N4] 边界连续性在 {verified} 个字形上验证（其中 {edge_frontier} 个由 edge_guard 兜底）：\
         接受侧 ≤ {CAP}，拒绝侧干净拒绝"
    );
}

// ---------------------------------------------------------------------------
// N4-E：注释一致性 —— QA 以「可执行断言」固化文档承诺
// ---------------------------------------------------------------------------

/// N4-E：文档承诺 `width*height ≤ CAP²` 与面积上界在实现中**字面成立**。
///
/// 背景：N4 的本质是"**注释里的契约谎言**"——旧注释断言
/// `width*height ≤ 4096²`，而实现（只约束坐标幅度）产出 256 MiB/字形。
/// 注释里的契约谎言比代码 bug 更危险，故此处把文档承诺转为硬断言。
#[test]
fn n4_e_documented_area_bound_holds_literally() {
    let f = font_regular();
    let cap_sq = (CAP as u64) * (CAP as u64);
    for ch in ['中', '国', 'A', 'W', 'M', 'i', '永'] {
        let Some(gid) = f.glyph_index(ch) else { continue };
        for step in 0..=24 {
            let px = 3800.0 + (step as f32) * 40.0; // 3800..4760（守卫交界密集区）
            let b = render_glyph(&f, gid, px);
            if b.is_empty() {
                continue;
            }
            let cells = b.width as u64 * b.height as u64;
            assert!(
                cells <= cap_sq,
                "ch={ch} px={px} 面积 {cells} 单元（{} MiB）> CAP²={cap_sq}（64 MiB）\
                 —— 文档断言不成立",
                cells * 4 / (1024 * 1024)
            );
            assert!(self_consistent(&b), "ch={ch} px={px} 位图不自洽");
        }
    }
}

// ---------------------------------------------------------------------------
// N4-F：`layout.rs` 累加链的 f32 溢出传播复核（独立判断，不采信"全是 f32 所以安全"）
// ---------------------------------------------------------------------------

/// N4-F：极大字号下`layout_text` 累加链溢出为 `inf`/`NaN` 时，
/// 产出的 `GlyphPos` 不得把 `NaN` 笔位**写进位图**（否则是静默错图）。
///
/// 工程师称"layout 全是 f32 所以安全"。QA 独立判断：f32 溢出确实不panic
/// （无整数算术），但**会静默产生 `inf`/`NaN` 笔位**。安全性取决于下游
/// `blit_glyph` 是否把 `NaN` 归一化。`NaN.floor() as i64` 在 Rust 中饱和为
/// **0**（`as` 转换对 NaN 返回 0）⇒ 若layout 产出 `NaN` 笔位，字形会被
/// 画到**原点**——这是错图而非拒绝。本测试锁定当前实际语义。
#[test]
fn n4_f_layout_overflow_does_not_paint_garbage() {
    use prism_io::Fbuf16F;
    use prism_text::cache::GlyphCache;
    use prism_text::layout::{layout_text, TextParams};
    use prism_text::raster::draw_glyphs;

    let f = font_regular();
    for &fs in &[1.0e30f32, 3.0e38, f32::MAX] {
        let params = TextParams { font_size: fs, ..TextParams::default() };
        let l = layout_text(&f, "中文测试", &params);
        // layout 不得 panic（能走到这里即证明）。累加链溢出为 inf/NaN 是允许的，
        // 但必须可观测：不能让下游画出垃圾。
        for g in &l.glyphs {
            let mut fb = Fbuf16F::new(8, 8);
            let mut cache = GlyphCache::new();
            draw_glyphs(&mut fb, &f, std::slice::from_ref(g), fs, [1.0; 4], 0.0, 0.0, &mut cache);
            // 关键：字号本身已被 render_glyph 的守卫拒绝 ⇒ 空位图 ⇒ 画面全零。
            assert!(
                fb.pixels.iter().all(|&p| p == 0),
                "font_size={fs:e} gid={} 竟画出像素 ⇒ NaN/inf 笔位被当作有效坐标",
                g.gid
            );
        }
    }
    // 【D1 已修复】`blit_glyph` 现**显式拒绝**非有限笔位
    // （`if !pen_x.is_finite() || !baseline_y.is_finite() { return; }`）。
    // Rust 的 `NaN as i64 == 0`，故修复前 NaN 会被静默归一化为 0 并
    // 在原点照常绘制（实测涂 780 个分量，与 pen=5.0 完全相同）——不 panic
    // 但画出位置完全错误的图。此处锁定修复后的契约：**拒绝，不落墨**。
    let f2 = font_regular();
    let gid = f2.glyph_index('中').expect("中");
    let bmp = render_glyph(&f2, gid, 24.0);
    assert!(!bmp.is_empty(), "前置：24px 中字应可渲染");
    for &v in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut fb = Fbuf16F::new(32, 32);
        prism_text::raster::blit_glyph(&mut fb, &bmp, v, 30.0, [1.0; 4]);
        assert!(
            fb.pixels.iter().all(|&p| p == 0),
            "pen={v:e} 非有限笔位应被拒绝，却画出了像素 ⇒ NaN/Inf 被静默当成 0"
        );
        let mut fb2 = Fbuf16F::new(32, 32);
        prism_text::raster::blit_glyph(&mut fb2, &bmp, 5.0, v, [1.0; 4]);
        assert!(fb2.pixels.iter().all(|&p| p == 0), "baseline={v:e} 应被拒绝");
    }
    // 自证非恒真：正常笔位仍必须落墨（守卫没有把正常路径也杀掉）。
    let mut fb_ok = Fbuf16F::new(32, 32);
    prism_text::raster::blit_glyph(&mut fb_ok, &bmp, 5.0, 30.0, [1.0; 4]);
    assert!(
        !fb_ok.pixels.iter().all(|&p| p == 0),
        "正常笔位应落墨（D1 守卫误杀了正常路径）"
    );
}
