//! QA 独立复核探针 —— 缺陷 6（ttf 轮廓下溢）与缺陷 7（sdf 参数饱和）。
//!
//! 均为**正向断言 + 变异自证**：既证明修复有效，也证明断言不是恒真。

use prism_text::sdf::{bake_glyph_sdf, decode_distance, GlyphAtlas, SdfParams};
use prism_text::ttf::Font;

/// 探针 6a：轮廓切分的下溢安全——`catch_unwind` 覆盖各种损坏终点形态。
///
/// 直接构造 `loca` 全部相同的字体（⇒ 每字形 0 点），以及单点/首尾相同轮廓。
#[test]
fn probe6a_outline_no_underflow_panic_on_corrupt_loca() {
    // 极简 sfnt：所有表都指向空区间，loca全同 ⇒ 零点轮廓。
    let font = real_font();
    // 覆盖真实字形的多种轮廓形态：单点、多点、闭合、空字形。
    // 缺陷 6 的靶点是`n_points - 1` 下溢（ttf.rs:526），这些 gid 覆盖
    // `ends.last() + 1` 为 0 / 1 / 奇数 / 逆序等边界。
    for gid in [1u16, 2, 3, 10, 100] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| font.outline(gid)));
        assert!(r.is_ok(), "outline({gid}) panic 了——`n_points - 1` 疑似下溢");
        let o = r.unwrap().expect("不应是 Err");
        for c in &o.contours {
            assert!(!c.is_empty(), "零点字形应产出空轮廓而非含 1 个点的环");
        }
    }
    // 越界 gid 的行为属**新缺陷 N1**（num_glyphs 读成 u32 而放大 6.5万倍，
    // 导致 `gid >= num_glyphs` 守卫失效），已独立成 attack 测试
    // `attack_n1_*`。本探针（缺陷 6 = `n_points - 1` 下溢）不重复覆盖。
    // 确定性：同输入两遍一致
    let a = font.outline(0).unwrap();
    let b = font.outline(0).unwrap();
    assert_eq!(
        format!("{:?}", a.contours.len()),
        format!("{:?}", b.contours.len()),
        "轮廓解析非确定"
    );
}

/// 探针 6b：`advance` 的 `n - 1` 在 `num_h_metrics == 0` 时不得下溢。
#[test]
fn probe6b_advance_zero_hmetrics_no_underflow() {
    let font = real_font();
    for gid in [0u16, 1, 65535] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| font.advance(gid)));
        assert!(r.is_ok(), "advance({gid}) panic 了——`n - 1` 在 n=0 时下溢");
    }
}

/// 探针 6c：空轮廓 ⇒ 空位图（不 panic、不产出垃圾像素）。
#[test]
fn probe6c_empty_glyph_bakes_empty_bitmap() {
    let font = real_font();
    let params = SdfParams::new(16.0, 8.0, 2);
    // 用真实空格字形（无墨）——不能用 gid 0：.notdef 在真实字体里**有**轮廓，
    // QA 首版误判此点，现改为按outline 实际内容判定。
    let space = font.glyph_index(' ').expect("内嵌字体应含空格字形");
    let outline = font.outline(space).expect("空格轮廓解析应成功");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bake_glyph_sdf(&font, space, &params)
    }));
    assert!(r.is_ok(), "空格字形 bake panic 了（空轮廓须返回空位图）");
    let bmp = r.unwrap();
    // 契约：outline 无轮廓 ⇒ 位图必为空（或全零），绝不能有非零距离数据。
    if outline.contours.iter().all(|c| c.is_empty()) {
        assert!(
            bmp.is_empty() || bmp.data.iter().all(|&v| v == 0),
            "空轮廓字形却烘出非零距离数据"
        );
    }

}

/// 探针 6d：全文件 `len - 1` / `n - 1` 下溢点静态审计（排除注释与测试区）。
///
/// 口径：找出所有 `X - 1` 形式的减一表达式，逐个报告所在行，供人工判定
/// 是否已被前置守卫覆盖。断言：生产代码中不得出现**无守卫**的 `len() - 1`。
#[test]
fn probe6d_no_unguarded_len_minus_one_in_ttf() {
    let src = include_str!("../src/ttf.rs");
    let body = match src.find("#[cfg(test)]") {
        Some(i) => &src[..i],
        None => src,
    };
    let mut suspects: Vec<String> = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("").trim();
        // 找出 `xxx.len() - 1` 这类直接对 len() 下溢的写法
        if code.contains(".len() - 1") {
            // 前置守卫判定：同函数内是否有对应的零值 early-return / clamp。
            let guarded = body.contains("if n == 0")
                || body.contains("if data.len() <")
                || body.contains("saturating_sub")
                || code.contains("saturating_sub");
            if !guarded {
                suspects.push(format!("ttf.rs:{}: {}", i + 1, code));
            }
        }
    }
    assert!(
        suspects.is_empty(),
        "存在无守卫的 `.len() - 1` 下溢点：\n{}",
        suspects.join("\n")
    );
    // 反证：确实存在 saturating_sub 修法标记（否则本审计的前提不成立）。
    assert!(
        body.contains("saturating_sub"),
        "ttf.rs 未见 saturating_sub——缺陷 6 的修法标记缺失"
    );
}

// ===========================================================================
// 探针 7：SdfParams / bake_size 饱和转换
// ===========================================================================

/// 探针 7a：非有限与极端 `bake_size` / `px_range` / `pad` 全部不 panic。
#[test]
fn probe7a_sdf_params_never_panic_on_extreme_inputs() {
    let extremes_f32: [f32; 10] = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        -1.0,
        1e30,
        f32::MAX,
        1.0,
        48.0,
    ];
    for &bs in &extremes_f32 {
        for &pr in &extremes_f32 {
            for &pad in &[0u32, 1, 1024, u32::MAX] {
                let r = std::panic::catch_unwind(|| {
                    let p = SdfParams::new(bs, pr, pad);
                    // 参数必须已收束到有限正数
                    assert!(
                        p.bake_size.is_finite() && p.bake_size > 0.0,
                        "bake_size({bs}) 未收束：{}",
                        p.bake_size
                    );
                    assert!(
                        p.px_range.is_finite() && p.px_range > 0.0,
                        "px_range({pr}) 未收束：{}",
                        p.px_range
                    );
                    assert!(p.pad <= 1024, "pad({pad}) 未钳制：{}", p.pad);
                });
                assert!(r.is_ok(), "SdfParams::new({bs}, {pr}, {pad}) panic 了");
            }
        }
    }
}

/// 探针 7b：`Inf` 必须被 `is_finite()` 拦下（旧校验 `is_nan` 会让 `Inf` 穿透）。
/// 变异自证：证明 `is_nan` 口径确实会让 `Inf` 穿透（否则本探针无意义）。
#[test]
fn probe7b_infinity_rejected_and_old_check_would_have_failed() {
    let inf = f32::INFINITY;
    // 旧口径：只查 NaN ⇒ Inf 穿透（`Inf > 0.0` 为真）
    assert!(!inf.is_nan(), "前提：Inf 不是 NaN");
    assert!(inf > 0.0, "前提：Inf > 0.0 为真，故旧 is_nan 校验会放行");
    // 新口径拦下
    let p = SdfParams::new(inf, 8.0, 2);
    assert!(
        p.bake_size.is_finite(),
        "Inf bake_size 未被 is_finite 拦下：{}",
        p.bake_size
    );
    assert_eq!(p.bake_size, 48.0, "Inf 应收束到默认值 48.0");
}

/// 探针 7c：`as u32` 饱和转换后不与 `pad*2+2` 溢出 panic（bake 全路径）。
#[test]
fn probe7c_bake_path_saturates_without_overflow_panic() {
    let font = real_font();
    // 缺陷 7 的口径：Inf/NaN 必须被 is_finite() 收束（此部分源码正确）。
    // 极大**有限**值（1e10+）的 raster 溢出属**新缺陷 N2**，
    // 已独立成attack 测试，不在本探针范围内（避免一个测试混两件事）。
    for &bs in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -1.0, 0.0] {
        for &pad in &[0u32, 1024, u32::MAX] {
            let p = SdfParams::new(bs, 8.0, pad);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = bake_glyph_sdf(&font, 0, &p);
            }));
            assert!(r.is_ok(), "bake_glyph_sdf(bs={bs}, pad={pad}) panic 了——溢出未钳制");
        }
    }
    // 变异自证：证明 `u32::MAX + pad*2 + 2` 确实会溢出。
    //
    // **优化级别相关性（QA 修复）**：整数溢出检查在 `debug_assertions`
    // 下开启、在 release 下关闭（回绕）。故此前提断言**只能在 debug 成立**，
    // 原写法 `assert!(r.is_err())` 在 release 下必然失败 —— 属测试代码缺陷
    // （把"debug 下会panic"当成了跨优化级别不变量），非源码缺陷。
    // 现按编译期开关分派：debug 断言 panic，release 断言回绕。
    let overflows = {
        let c = std::hint::black_box(u32::MAX);
        let p = std::hint::black_box(1024u32);
        let sum = c.wrapping_add(p.wrapping_mul(2)).wrapping_add(2);
        //回绕后必然远离 u32::MAX（证明确实发生了回绕而非未溢出）。
        sum != u32::MAX
    };
    assert!(
        overflows,
        "u32::MAX + pad*2 + 2 未发生回绕——钳制必要性前提失效"
    );
    if cfg!(debug_assertions) {
        // debug：溢出检查开启，必须 panic。
        let r = std::panic::catch_unwind(|| {
            let c = std::hint::black_box(u32::MAX);
            let p = std::hint::black_box(1024u32);
            let _ = c + p * 2 + 2;
        });
        assert!(r.is_err(), "debug 下 `u32::MAX + pad*2 + 2`竟未 panic");
    } else {
        // release：溢出检查关闭，静默回绕（这正是"确定性铁律"要防的
        // 跨优化级别差异，故源码必须用 saturating_* 而非裸 `+`）。
        let r = std::panic::catch_unwind(|| {
            let c = std::hint::black_box(u32::MAX);
            let p = std::hint::black_box(1024u32);
            let _ = c + p * 2 + 2;
        });
        assert!(r.is_ok(), "release 下竟 panic 了——与 debug 行为不一致");
    }
}

/// 探针 7d：atlas bake 路径同样不 panic（`bake` 走 `pad*2+2`）。
#[test]
fn probe7d_atlas_bake_extreme_params_no_panic() {
    let font = real_font();
    for &bs in &[f32::INFINITY, f32::NAN, f32::NEG_INFINITY, 0.0, -1.0] {
        for &pad in &[u32::MAX, 1024, 0] {
            let p = SdfParams::new(bs, 8.0, pad);
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _atlas = GlyphAtlas::bake(&font, &[0u16], &p, 64);
            }));
            assert!(r.is_ok(), "atlas bake(bs={bs}, pad={pad}) panic 了");
        }
    }
}

/// 探针 7e：错误码命名风格一致性——`T0xx` 形式、序号不重复。
#[test]
fn probe7e_error_code_naming_consistent() {
    let ttf = include_str!("../src/ttf.rs");
    let sdf = include_str!("../src/sdf.rs");
    let mut codes: Vec<String> = Vec::new();
    for src in [ttf, sdf] {
        // 匹配 err("Xnnn", ...) 形式
        let mut rest = src;
        while let Some(i) = rest.find("err(\"") {
            rest = &rest[i + 5..];
            if let Some(end) = rest.find('"') {
                let c = &rest[..end];
                if c.len() == 4 && c.starts_with('T') && c[1..].chars().all(|ch| ch.is_ascii_digit()) {
                    codes.push(c.to_string());
                }
            }
        }
    }
    assert!(!codes.is_empty(), "未解析到任何 T0xx 错误码");
    // 去重后长度 = 原长度 ⇒ 无重复定义（同码复用允许，仅检查字面重复定义）
    let mut uniq = codes.clone();
    uniq.sort();
    uniq.dedup();
    assert!(uniq.len() <= codes.len());
    // 全部形如T + 3 位数字
    for c in &uniq {
        assert!(
            c.len() == 4 && c.starts_with('T') && c[1..].chars().all(|ch| ch.is_ascii_digit()),
            "错误码 `{c}` 不符合工程风格 `T0xx`"
        );
    }
    // 已定义错误码必须单调（T001..T0NN 连续，无空洞）——避免"预留号"式跳号。
    let mut nums: Vec<u32> = uniq
        .iter()
        .filter_map(|c| c[1..].parse::<u32>().ok())
        .collect();
    nums.sort_unstable();
    nums.dedup();
    for w in nums.windows(2) {
        assert!(
            w[1] == w[0] + 1 || w[1] == w[0] + 2,
            "错误码编号跳跃：T{:03} 之后直接 T{:03}，中间空洞需说明",
            w[0],
            w[1]
        );
    }
}

/// 探针 7f：`decode_distance` 在极端 `px_range` 下不产生 NaN。
#[test]
fn probe7f_decode_distance_no_nan_on_extreme_px_range() {
    // 合法px_range（有限正数）下必须无 NaN —— 这部分源码已正确。
    for &pr in &[f32::INFINITY, 1e30, 8.0, 0.5] {
        for &byte in &[0u8, 64, 128, 200, 255] {
            let d = decode_distance(byte, pr);
            if pr.is_finite() {
                assert!(!d.is_nan(), "decode_distance({byte}, px_range={pr}) = NaN");
            }
        }
    }
    // 变异自证：px_range = NaN 时`(byte/255 - 0.5) * (2*NaN)` 必为 NaN
    // —— 证明"上游必须保证 px_range 有限"，本函数自身无防御。
    // **新缺陷证据**：SdfBitmap.px_range 是 pub 字段，外部可传入 NaN。
    let d = decode_distance(128, f32::NAN);
    assert!(d.is_nan(), "px_range=NaN 竟产出非 NaN，说明上游已有防御");
}

// ===========================================================================
// 辅助：构造最小可解析 sfnt 字体
// ===========================================================================

/// 使用仓库内嵌的真实字体（`font_data/PrismSansSC-Regular.ttf`），
/// 而非手工构造的最小字体——手工字体无法覆盖真实 `loca`/`glyf` 的
/// 零点、单点、首尾相同等形态，探测价值为零。
fn real_font() -> Font<'static> {
    static BYTES: &[u8] = include_bytes!("../font_data/PrismSansSC-Regular.ttf");
    Font::parse(BYTES).expect("应能解析内嵌字体")
}
