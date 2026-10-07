//! QA 攻击性测试 — Phase 4 D4 SDF 字形烘焙 + atlas（`prism-text`）。
//!
//! 攻击面：
//! * 编码/解码语义（`decode_distance` 镜面、锚点）；
//! * 边界/退化：空字形（空格）、超小 bake_size、空 gid 列表、越界 gid；
//! * 确定性：同参数逐字节一致（bake / atlas）；
//! * 结构不变量：atlas UV 域、行装箱单调、槽内含内外样本；
//! * 防御：库代码不得 panic（含不可信 gid）。
//!
//! 每个测试失败的含义在函数头注释显式声明。

use prism_text::font_regular;
use prism_text::raster::GlyphBitmap;
use prism_text::sdf::{
    bake_glyph_sdf, decode_distance, GlyphAtlas, SdfBitmap, SdfParams,
};

// ===========================================================================
// 编码/解码语义
// ===========================================================================

/// 失败含义：SDF 解码锚点偏离 `(byte/255 - 0.5) * 2 * px_range`——shader 端解码
/// （text.wgsl `(d-0.5)*2*px_range`）与 CPU 不再镜像，采样出错误锐边。
#[test]
fn sdf_decode_anchor_points() {
    let pr = 8.0f32;
    // 量化半码容差（与 sdf.rs 单元测试同口径：±0.5 字节 ≈ 2*pr/255）。
    let tol = 2.0 * pr / 255.0 + 1e-3;
    // byte=128 → (128/255-0.5)*16 ≈ 0.031。
    assert!((decode_distance(128, pr) - 0.0).abs() <= tol, "byte 128 ≈ 0");
    // byte=255 → +px_range。
    assert!((decode_distance(255, pr) - pr).abs() < 1e-3, "byte 255 = +range");
    // byte=0 → −px_range。
    assert!((decode_distance(0, pr) + pr).abs() < 1e-3, "byte 0 = −range");
}

/// 失败含义：解码单调性破裂（字节增大而距离非增）。
#[test]
fn sdf_decode_monotone_in_byte() {
    let pr = 8.0;
    let mut prev = f32::NEG_INFINITY;
    for b in 0u16..=255 {
        let d = decode_distance(b as u8, pr);
        assert!(d > prev, "解码应严格单调：byte {b} d={d} prev={prev}");
        prev = d;
    }
}

// ===========================================================================
// 边界 / 退化
// ===========================================================================

/// 失败含义：空字形（空格，有度量无墨）烘焙未返回空位图——下游会误分配 slot。
#[test]
fn sdf_empty_glyph_returns_empty_bitmap() {
    let f = font_regular();
    let sp = f.glyph_index(' ').expect("空格在 ASCII 子集内");
    let bmp = bake_glyph_sdf(&f, sp, &SdfParams::default());
    assert!(bmp.is_empty(), "空格应为空位图");
    assert_eq!(bmp.width, 0);
    assert_eq!(bmp.height, 0);
    assert!(bmp.data.is_empty());
    // px_range 仍应透传（解码契约）。
    assert_eq!(bmp.px_range, SdfParams::default().px_range);
}

/// 失败含义：**超小 bake_size**（亚像素）导致 panic、除零或尺寸溢出。
#[test]
fn sdf_tiny_bake_size_no_panic() {
    let f = font_regular();
    let g = f.glyph_index('中').unwrap();
    for bs in [0.1f32, 0.5, 1.0, 2.0] {
        let p = SdfParams::new(bs, 8.0, 4);
        let bmp = bake_glyph_sdf(&f, g, &p);
        // 不得 panic；数据长度须与声明尺寸一致。
        assert_eq!(bmp.data.len(), (bmp.width * bmp.height) as usize);
    }
}

/// 失败含义（**P1 防御缺陷候选**）：**越界 gid** 烘焙 panic 而非按契约返回空位图
/// （`sdf.rs` 文档：「越界 gid / 空字形 → 空位图」）。库代码 panic 即缺陷。
///
/// 说明：用 `catch_unwind` 捕获 panic 转为断言失败，保证测试套件继续。
#[test]
fn sdf_out_of_range_gid_no_panic() {
    let f = font_regular();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bake_glyph_sdf(&f, 60000, &SdfParams::default())
    }));
    match result {
        Ok(bmp) => {
            assert!(bmp.is_empty(), "越界 gid 应返回空位图");
        }
        Err(_) => panic!(
            "越界 gid=60000 触发 panic（应返回空位图）——库代码防御缺失（ttf.rs 越界访问）"
        ),
    }
}

/// 失败含义：`SdfParams::new` 对 **NaN/负** bake_size 未收束到默认（防呆契约破裂）。
#[test]
fn sdf_params_new_sanitizes_nan_and_negative() {
    let d = SdfParams::default();
    let p_nan = SdfParams::new(f32::NAN, f32::NAN, 4);
    assert_eq!(p_nan.bake_size, d.bake_size, "NaN bake_size 应收束默认");
    assert_eq!(p_nan.px_range, d.px_range, "NaN px_range 应收束默认");
    let p_neg = SdfParams::new(-10.0, -3.0, 4);
    assert_eq!(p_neg.bake_size, d.bake_size, "负 bake_size 应收束默认");
    assert_eq!(p_neg.px_range, d.px_range, "负 px_range 应收束默认");
}

/// 失败含义：`SdfParams::new` 对 **无穷大** bake_size 未收束——无穷会使下游
/// `bake_size.round() as u32` 产生饱和值，atlas 尺寸失控。防呆契约要求正有限。
#[test]
fn sdf_params_new_rejects_infinity() {
    let d = SdfParams::default();
    let p_inf = SdfParams::new(f32::INFINITY, f32::INFINITY, 4);
    assert!(
        p_inf.bake_size.is_finite(),
        "无穷 bake_size 应被防呆（实际 {}）——防呆契约缺陷",
        p_inf.bake_size
    );
    assert!(p_inf.px_range.is_finite(), "无穷 px_range 应被防呆");
    assert_eq!(p_inf.bake_size, d.bake_size, "应收束到默认");
}

/// 失败含义：空 gid 列表烘焙 panic 或产出非法 atlas 尺寸。
#[test]
fn sdf_atlas_empty_gids() {
    let f = font_regular();
    let at = GlyphAtlas::bake(&f, &[], &SdfParams::default(), 256);
    assert_eq!(at.slots.len(), 0);
    assert_eq!(at.data.len(), (at.width * at.height) as usize);
    assert!(at.width <= 256);
}

/// 失败含义：含越界 gid 的 atlas 烘焙 panic（不可信 gid 集合应被安全跳过）。
#[test]
fn sdf_atlas_out_of_range_gid_no_panic() {
    let f = font_regular();
    let g = f.glyph_index('A').unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GlyphAtlas::bake(&f, &[g, 60000, g], &SdfParams::default(), 256)
    }));
    assert!(result.is_ok(), "atlas 含越界 gid 触发 panic——防御缺失");
}

// ===========================================================================
// 确定性
// ===========================================================================

/// 失败含义：同参数两遍烘焙**不逐字节一致**——违反确定性契约（字体,gid,size,range
/// 的纯函数）。
#[test]
fn sdf_bake_deterministic_bytewise() {
    let f = font_regular();
    for ch in ['A', 'g', '中', '永'] {
        let g = f.glyph_index(ch).expect("字形在子集内");
        let p = SdfParams::new(48.0, 8.0, 4);
        let a = bake_glyph_sdf(&f, g, &p);
        let b = bake_glyph_sdf(&f, g, &p);
        assert_eq!(a, b, "{ch} 烘焙应逐字节确定");
    }
}

/// 失败含义：atlas 同输入两遍不逐字节一致（排布或数据非确定）。
#[test]
fn sdf_atlas_deterministic_bytewise() {
    let f = font_regular();
    let gids: Vec<u16> = ['H', 'e', 'l', 'o', '中', '永']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    let p = SdfParams::default();
    let a = GlyphAtlas::bake(&f, &gids, &p, 256);
    let b = GlyphAtlas::bake(&f, &gids, &p, 256);
    assert_eq!(a, b, "atlas 应逐字节确定");
}

// ===========================================================================
// 结构不变量
// ===========================================================================

/// 失败含义：atlas UV 越界（[0,1] 外）或退化矩形——shader 采样会越界/无面积。
#[test]
fn sdf_atlas_uv_within_unit_square() {
    let f = font_regular();
    let gids: Vec<u16> = ['A', 'B', '中', '永', '吗']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    let at = GlyphAtlas::bake(&f, &gids, &SdfParams::new(32.0, 6.0, 3), 512);
    assert!(at.width <= 512, "atlas 宽不得超上限");
    for slot in &at.slots {
        let uv = at.uv(slot);
        assert!(uv[0] >= 0.0 && uv[2] <= 1.0 && uv[1] >= 0.0 && uv[3] <= 1.0, "UV 越界 {uv:?}");
        assert!(uv[0] < uv[2] && uv[1] < uv[3], "UV 退化矩形 {uv:?}");
    }
}

/// 失败含义：atlas 行装箱后**行内 x 非单调不减**（排布算法破坏，槽重叠）。
#[test]
fn sdf_atlas_shelf_packing_monotone() {
    let f = font_regular();
    let gids: Vec<u16> = ['H', 'e', 'l', 'o', '中']
        .iter()
        .filter_map(|&c| f.glyph_index(c))
        .collect();
    let at = GlyphAtlas::bake(&f, &gids, &SdfParams::default(), 256);
    let mut last_row_y = 0u32;
    let mut last_x_end = 0u32;
    for s in &at.slots {
        if s.y != last_row_y {
            last_row_y = s.y;
            last_x_end = 0;
        }
        assert!(s.x >= last_x_end, "行内 x 应单调不减（无重叠）");
        last_x_end = s.x + s.w;
    }
}

/// 失败含义：有墨字形的槽内**缺少**内侧或外侧样本——距离场语义错误的信号。
#[test]
fn sdf_atlas_slots_have_inside_and_outside() {
    let f = font_regular();
    let gids: Vec<u16> = ['A', '中'].iter().filter_map(|&c| f.glyph_index(c)).collect();
    let at = GlyphAtlas::bake(&f, &gids, &SdfParams::new(32.0, 6.0, 3), 512);
    for slot in &at.slots {
        let mut any_in = false;
        let mut any_out = false;
        for yy in 0..slot.h {
            for xx in 0..slot.w {
                let c = at.sample(slot, xx, yy);
                if c > 160 {
                    any_in = true;
                }
                if c < 96 {
                    any_out = true;
                }
            }
        }
        assert!(any_in, "gid={} 槽应含内侧样本", slot.gid);
        assert!(any_out, "gid={} 槽应含外侧样本", slot.gid);
    }
}

/// 失败含义：`GlyphAtlas::sample` 越界未安全返回（应返回 0 = 远外侧，不得 panic）。
#[test]
fn sdf_atlas_sample_out_of_bounds_safe() {
    let f = font_regular();
    let gids: Vec<u16> = ['A'].iter().filter_map(|&c| f.glyph_index(c)).collect();
    let at = GlyphAtlas::bake(&f, &gids, &SdfParams::default(), 256);
    let slot = at.slots.first().expect("应有 A 的槽");
    // 越界 (w, h) 与巨大坐标均应安全。
    assert_eq!(at.sample(slot, slot.w, slot.h), 0);
    assert_eq!(at.sample(slot, u32::MAX, u32::MAX), 0);
}

/// 失败含义：`GlyphAtlas::slot` 对无墨 gid 返回槽（应 None）或对已烘焙 gid 返回
/// None（查找表错误）。
#[test]
fn sdf_atlas_slot_lookup() {
    let f = font_regular();
    let a_gid = f.glyph_index('A').unwrap();
    let sp_gid = f.glyph_index(' ').unwrap();
    let at = GlyphAtlas::bake(&f, &[a_gid, sp_gid], &SdfParams::default(), 256);
    assert!(at.slot(a_gid).is_some(), "A 应有槽");
    assert!(at.slot(sp_gid).is_none(), "空格无槽");
    assert!(at.blanks.contains(&sp_gid), "空格应登记为 blank");
}

/// 失败含义：`SdfBitmap::empty` 契约破裂（is_empty 或 px_range 透传错误）。
#[test]
fn sdf_bitmap_empty_contract() {
    let b = SdfBitmap::empty(5.0);
    assert!(b.is_empty());
    assert_eq!(b.px_range, 5.0);
    assert!(b.data.is_empty());
}

// ===========================================================================
// 距离场语义（合成覆盖率位图，不依赖字体）
// ===========================================================================

/// 失败含义：距离场**符号**语义错误（内正外负）——放大重建时字形会内外翻转。
/// 合成 5×5 实心块（内圈 3×3 有墨），中心应内正、块外邻应外负。
#[test]
fn sdf_distance_sign_semantics() {
    // 借助 bake 的公开入口无法直接喂合成位图；此处经 atlas 采样间接验证：
    // 用真实字形，中心（内）字节应 > 128，角（外）字节应 < 128。
    let f = font_regular();
    let g = f.glyph_index('中').unwrap();
    let sdf = bake_glyph_sdf(&f, g, &SdfParams::default());
    assert!(!sdf.is_empty());
    let cx = sdf.width / 2;
    let cy = sdf.height / 2;
    let center = sdf.data[(cy * sdf.width + cx) as usize];
    let corner = sdf.data[0];
    assert!(center > 128, "中心应内侧（>128），实际 {center}");
    assert!(corner < 128, "角落应外侧（<128），实际 {corner}");
}

/// 失败含义：`GlyphBitmap` 空（宽或高为 0）时烘焙未安全短路——下游越界。
#[test]
fn sdf_bake_handles_zero_dim_bitmap_path() {
    // 通过一个明显无墨的字形触发 render_glyph 返回空位图的分支。
    let f = font_regular();
    // 空格必然无墨 → 走 cov.is_empty() 短路。
    let sp = f.glyph_index(' ').unwrap();
    let bmp = bake_glyph_sdf(&f, sp, &SdfParams::new(32.0, 6.0, 3));
    assert!(bmp.is_empty());
    // 空 GlyphBitmap 构造仅用于编译期接口核对（私有路径，不直接调用）。
    let probe = GlyphBitmap { left: 0, top: 0, width: 0, height: 0, alpha: Vec::new() };
    assert!(probe.alpha.is_empty());
}

// ===========================================================================
// 缺陷 6 / 7 自证：TTF 空轮廓下溢 + SdfParams Inf/溢出边界
// ===========================================================================

/// 缺陷 7 自证：`±Inf` / `NaN` / 非正数一律收束默认，且 `pad` 被钳制。
///
/// 旧实现用 `value > 0.0 && !value.is_nan()`：`Inf > 0.0` 为真故`Inf` 穿透，
/// 随后 `bake_size.round() as u32` 饱和为 `u32::MAX`，再经 `+ pad*2 + 2`
/// 溢出 panic。本测试锁定 `is_finite` 语义。
#[test]
fn sdf_params_rejects_all_non_finite_variants() {
    let d = SdfParams::default();
    for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0, -1.0, -1e30] {
        let p = SdfParams::new(bad, bad, 4);
        assert_eq!(p.bake_size, d.bake_size, "bake_size={bad} 应收束默认");
        assert_eq!(p.px_range, d.px_range, "px_range={bad} 应收束默认");
    }
    // 合法有限正值必须原样保留（防呆不得误伤正常参数）。
    let ok = SdfParams::new(64.0, 12.0, 6);
    assert_eq!(ok.bake_size, 64.0);
    assert_eq!(ok.px_range, 12.0);
    assert_eq!(ok.pad, 6);
}

/// 缺陷 7 自证：极端 `pad` 不在构造期/烘焙期 panic，且被钳到上界。
#[test]
fn sdf_params_clamps_huge_pad() {
    let p = SdfParams::new(48.0, 8.0, u32::MAX);
    assert!(p.pad <= 1024, "超大pad 应被钳制，得到 {}", p.pad);
    // 走完整atlas 烘焙路径：旧实现在此panic（`pad * 2 + 2` 溢出）。
    let font = font_regular();
    let atlas = GlyphAtlas::bake(&font, &[36u16], &p, 64);
    assert!(atlas.width <= 64, "atlas 应受 max_width 约束，得到 {}", atlas.width);
}

/// 缺陷 7 自证：极大但有限的 `bake_size` 烘焙不 panic（饱和而非溢出）。
#[test]
fn sdf_bake_huge_finite_bake_size_no_panic() {
    let font = font_regular();
    for bs in [1e6f32, 1e18, 3.0e38] {
        let p = SdfParams::new(bs, 8.0, 4);
        let bmp = bake_glyph_sdf(&font, 36, &p);
        assert!(
            bmp.width == 0 || bmp.data.len() == bmp.width as usize * bmp.height as usize,
            "bake_size={bs} 的位图尺寸不自洽：{}x{}",
            bmp.width,
            bmp.height
        );
    }
}

/// 缺陷 6 自证：越界 gid / 空字形返回空位图而非 panic（`n_points == 0`、
/// `start >= n_points`、非单调 `ends` 三条下溢路径均不得越界读或 panic）。
#[test]
fn ttf_out_of_range_and_empty_glyph_return_empty_outline() {
    let font = font_regular();
    // 越界 gid（文档承诺返回空位图）
    for gid in [60000u16, u16::MAX, 65535] {
        let out = font.outline(gid).expect("越界 gid 应Ok 而非 Err");
        assert!(out.contours.is_empty(), "gid={gid} 应返回空轮廓");
    }
    // gid 0 = .notdef，字体族中通常无轮廓：不得 panic。
    let _ = font.outline(0).expect("gid 0 应 Ok");
    // advance 在 num_h_metrics == 0 时不得下溢（已有 n == 0 早退，此处复核）。
    assert!(font.advance(u16::MAX) <= font.advance(0).max(font.advance(0)));
}

/// 缺陷 6 自证：连续多次解析同一非法 gid 幂等（确定性 + 无状态污染）。
#[test]
fn ttf_out_of_range_gid_is_deterministic() {
    let font = font_regular();
    let a = font.outline(u16::MAX).expect("应 Ok").contours.len();
    let b = font.outline(u16::MAX).expect("应 Ok").contours.len();
    assert_eq!(a, b, "同输入须同输出（确定性铁律）");
}
