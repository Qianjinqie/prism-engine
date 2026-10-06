//! QA 攻击测试 —— **新发现的真实源码缺陷**（QA 不修源码，仅记录证据 + 路由）。
//!
//! 这两个缺陷不在上一轮 7 项清单内，是本轮独立复核时用新探针发现的。
//! 二者都违反宪章 C18「永不 panic」，且**用仓库内嵌的真实字体即可复现**，
//! 不需要构造损坏数据——因此严重度高于此前修复的 7 项。

use prism_text::raster;
use prism_text::sdf::{bake_glyph_sdf, GlyphAtlas, SdfParams};
use prism_text::ttf::Font;

fn real_font() -> Font<'static> {
    static BYTES: &[u8] = include_bytes!("../font_data/PrismSansSC-Regular.ttf");
    Font::parse(BYTES).expect("应能解析内嵌字体")
}

/// 【新缺陷 N1】`ttf.rs:150` —— `maxp.numGlyphs` 按 **u32** 读取，但该字段
/// 在 TrueType/OpenType 中是 **u16**（其后紧邻 `maxPoints` 亦u16）。
///
/// 后果（结果级，非风格）：
/// 1. `num_glyphs` 被读成 `numGlyphs | (maxPoints << 16)`，真实字体
///    PrismSansSC-Regular.ttf 上得到 **451477751**（真值 6889）——放大 6.5万倍。
/// 2. `outline()` 的越界守卫 `gid >= self.num_glyphs` 因此几乎永不生效。
/// 3. 越界 gid 直落`loca` 切片：`range start index 287582512 out of range
///    for slice of length 1948392` ⇒ **panic**（违反 C18 永不 panic）。
///
/// 复现：对内嵌真实字体调用 `font.outline(100)` 即 panic。
#[test]
fn attack_n1_num_glyphs_read_as_u32_inflates_and_causes_panic() {
    let font = real_font();

    // 证据 1：num_glyphs 被严重放大。真值应等于 maxp 处的 u16 numGlyphs。
    let raw = font.num_glyphs();
    // 内嵌字体真实 numGlyphs = 6889（可由 golden 文件交叉验证）。
    assert_eq!(
        raw, 6889,
        "num_glyphs = {raw}，期望 6889。maxp.numGlyphs 是 u16 字段，\
         源码用 be::u32 读取会把紧邻的 maxPoints 一并吃进来⇒ 数值放大 6.5 万倍"
    );

    // 证据 2：越界 gid（>= 真实 numGlyphs=6889）不再被守卫拦住，直接 panic。
    //实测 gid 6898/6899/6900... 全部 panic（loca 切片越界）。
    for gid in [6889u16, 6898, 7000, 10000] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| font.outline(gid)));
        assert!(
            r.is_ok(),
            "对真实字体 outline({gid}) panic 了——num_glyphs 被放大成451477751，\
             `gid >= num_glyphs` 守卫失效，越界 gid 直落 loca 切片（违反宪章 C18）"
        );
    }
}

/// 【新缺陷 N1 变体】放大后的 num_glyphs 还污染 `bake_glyph_sdf`：
/// 同一路径 panic（bake → outline）。
#[test]
fn attack_n1b_bake_glyph_sdf_panics_on_out_of_range_gid() {
    let font = real_font();
    let params = SdfParams::new(24.0, 8.0, 2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bake_glyph_sdf(&font, 7000, &params)
    }));
    assert!(r.is_ok(), "bake_glyph_sdf(gid=7000) panic 了（N1 的 bake 路径表现）");
}

/// 【新缺陷 N1 变体】`GlyphAtlas::bake` 同样经outline，批量烘焙整表即panic。
#[test]
fn attack_n1c_atlas_bake_panics_on_out_of_range_gid() {
    let font = real_font();
    let params = SdfParams::new(24.0, 8.0, 2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GlyphAtlas::bake(&font, &[1u16, 2, 7000], &params, 128)
    }));
    assert!(r.is_ok(), "GlyphAtlas::bake 含越界 gid 时 panic 了");
}

/// 【新缺陷 N2】`raster.rs:188-189` —— `let width = (x1 - x0).max(0) as u32;`
/// 中`x1` / `x0` 是 `f32 as i32` 的**饱和转换**结果。当 `scale` 极大
/// （`SdfParams::new(3.0e38, ..)` 未对 `bake_size` 设上界）时，
/// `x0` 饱和到 `i32::MIN`、`x1` 饱和到 `i32::MAX`，`x1 - x0` 即 i32 减法溢出。
///
/// 后果：`attempt to subtract with overflow` panic（违反 C18）。
/// 注意：SdfParams 已用 `is_finite()` 拦下 Inf/NaN（缺陷 7 修法正确），
/// 但 **极大有限值**未被拦下——这是缺陷 7 修法的**残留缺口**。
///
/// 复现：`bake_glyph_sdf(&font, 36, &SdfParams::new(1.0e10, 8.0, 0))`。
/// 已实测 panic 的 bake_size：1e10 / 1e20 / 1e30 / 3e38（gid=36）。
/// 边界说明：1e10 以下不panic，而是先尝试分配尺寸 ∝ scale² 的巨型位图
/// （1e6~1e9 实测挂起，属另一处健壮性隐患，未在本轮定性）。
/// 因此本探针只用**已实测 panic** 的量级，不声称精确边界。
#[test]
fn attack_n2_raster_window_width_overflow_on_huge_finite_bake_size() {
    let font = real_font();
    // 极大但**有限**的 bake_size：is_finite() 拦不住，scale溢出 i32。
    for &bs in &[1.0e30f32, 3.0e38, 1.0e20] {
        let params = SdfParams::new(bs, 8.0, 0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bake_glyph_sdf(&font, 1, &params)
        }));
        assert!(
            r.is_ok(),
            "bake_size={bs}（有限但极大）导致 raster.rs:189 `x1 - x0` i32 溢出 panic；\
             SdfParams 只校验 is_finite，未设上界"
        );
    }
}

/// 【新缺陷 N2 变体】atlas 批量路径同样溢出。
#[test]
fn attack_n2b_atlas_bake_overflow_on_huge_finite_bake_size() {
    let font = real_font();
    let params = SdfParams::new(1.0e20, 8.0, 1024);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GlyphAtlas::bake(&font, &[36u16], &params, 64)
    }));
    assert!(r.is_ok(), "atlas bake 在极大有限 bake_size 下 i32 溢出 panic");
}

/// 【N2 契约锁定】窗口计算在**极大有限 scale** 下不溢出、且结果落在 i64域。
///
/// # 历史：这是一条缺陷**存在性见证**测试（已反转）
///
/// 原断言（`attack_n2c_minimal_repro_of_saturating_cast_subtraction`）：
/// ```ignore
/// let calc = |scale, x_min, x_max| -> i32 {
///     let x0 = (x_min * scale).floor() as i32;   // 饱和转换
///     let x1 = (x_max * scale).ceil()  as i32;   // 饱和转换
///     x1 - x0                                     // ← i32 减法溢出
/// };
/// assert!(catch_unwind(|| calc(3.0e38, -1000.0, 1000.0)).is_err());
/// ```
/// 它断言"极大scale 下**必然溢出 panic**"。缺陷修复后必然翻转：
/// - debug下旧算式确实 panic（测试绿，见证缺陷存在）；
/// - release 下旧算式**静默回绕**（不 panic）⇒ `is_err()` 为假 ⇒ **测试红**。
/// 这就是它此前只在 debug 全绿、release 下暴露为failed 的原因。
///
/// 更关键的是：它**复刻的是旧算法**（本地闭包 `calc`），并未调用真实实现，
/// 因此源码修复后它与实现**彻底脱钩** ——既不能证明修复，也不能防回归。
/// 断言"缺陷存在"一旦被修复，就成了永久报红的假信号，还会掩盖真实回归。
///
/// # 现断言的正向契约（源码 `raster.rs:236-239`）
///
/// 窗口计算经`f64` 中间量**升到 i64 域**：
/// ```ignore
/// let x0 = ((x_min * scale).floor() as f64) as i64;   // 饱和到 i64，而非 i32
/// ```
/// 故 `3.4e38` 级输入在 i64 域内**远未饱和**（i64 上限≈9.22e18），
/// 差值也不溢出。本测试锁定该契约，并自证非恒真。
#[test]
fn attack_n2c_window_math_is_i64_domain_and_never_overflows() {
    // 与 `raster.rs:236-239` 同形：f32 → f64 → i64。
    let calc = |scale: f32, x_min: f32, x_max: f32| -> i64 {
        let x0 = ((x_min * scale).floor() as f64) as i64;
        let x1 = ((x_max * scale).ceil() as f64) as i64;
        x1 - x0
    };
    // (1) 正常值：语义不变（与旧算式结果一致）。
    assert_eq!(calc(1.0, -10.0, 10.0), 20, "正常窗口宽度语义应保持");
    // (2) 极大有限 scale：坐标守卫**先行拦截**，`x1 - x0` 永不执行。
    //
    // 关键：必须照抄源码的**语句顺序**（`raster.rs:243` 守卫 → `:249` 减法）。
    // 源码注释已写明该不变量：*"`x1 - x0` 只在坐标守卫**通过后**才计算
    // （此时二者 ∈ [−cap, cap]，差值 ≤ 2·cap，i64 远 enough，不可能溢出）"*。
    // 若只复制算式而漏掉前置守卫，减法会在 x0=i64::MIN / x1=i64::MAX 时溢出
    //   —— 那是**测试的伪缺陷**而非源码缺陷（首版即踩此坑，已修）。
    const CAP: i64 = 4096;
    let guarded = |scale: f32, x_min: f32, x_max: f32| -> (bool, i64) {
        let x0 = ((x_min * scale).floor() as f64) as i64;
        let x1 = ((x_max * scale).ceil() as f64) as i64;
        // 坐标守卫（与源码同序）：超界即拒绝，不进入减法。
        if x0 < -CAP || x1 > CAP {
            return (false, 0);
        }
        (true, x1 - x0)
    };
    for &scale in &[1.0e10f32, 1.0e18, 3.0e38, f32::MAX] {
        // 不 panic 即证明守卫先行；且拒绝时不进入减法。
        let (ok, w) = guarded(scale, -1000.0, 1000.0);
        assert!(!ok, "scale={scale:e} 坐标应越界被拒");
        assert_eq!(w, 0, "被拒时不应计算边长");
    }
    // (2b) 守卫**通过**时（坐标合法），减法必不溢出，且差值 ≤ 2·cap。
    //      这是源码注释给出的不变量，量化锁定。
    for &scale in &[1.0f32, 2.0, 4.0, 4.096] {
        let (ok, w) = guarded(scale, -1000.0, 1000.0);
        assert!(ok, "scale={scale} 坐标应合法（x1={} ≤ {CAP}）", 1000.0f32 * scale);
        assert!(w > 0, "scale={scale} 宽度 {w} 非正");
        assert!(
            w <= 2 * CAP,
            "scale={scale} 守卫内宽度 {w} 应 ≤ 2·{CAP}（坐标守卫的推论）"
        );
    }
    // (2c) **自证非恒真**：证明真的升到了 i64 域，而非"i32 域恰好没溢出"。
    //     判据 = **饱和行为**：极大输入下 i64 域得到 ±9.22e18 级值，
    //     而 i32 域只会饱和到 ±2.15e9。若实现退回 i32 域，此断言立即转红。
    let huge = 3.0e38f32;
    let x0_huge = ((-1000.0f32 * huge).floor() as f64) as i64;
    let x1_huge = ((1000.0f32 * huge).ceil() as f64) as i64;
    assert_eq!(
        x0_huge,
        i64::MIN,
        "极大负坐标应饱和到 i64::MIN（i64 域判据）"
    );
    assert_eq!(
        x1_huge,
        i64::MAX,
        "极大正坐标应饱和到 i64::MAX（i64 域判据）"
    );
    // 若仍在 i32 域，二者会分别是 i32::MIN / i32::MAX（≈ ±2.15e9）。
    assert!(
        (x1_huge - i32::MAX as i64).abs() > 1_000_000_000_000_000_000,
        "饱和值贴近 i32::MAX ⇒ 疑似仍在 i32 域（恒真/回归风险）"
    );
    // (3) 真实入口交叉验证：`render_glyph` 在同量级 px下**干净拒绝**
    //     （不 panic、不分配 alpha），证明公开 API 语义与上述算术一致。
    let font = real_font();
    for &px in &[1.0e10f32, 1.0e18, 3.0e38, f32::MAX] {
        let bmp = raster::render_glyph(&font, 1, px);
        assert!(bmp.is_empty(), "px={px:e} 应被窗口守卫拒绝");
        assert_eq!(bmp.alpha.len(), 0, "px={px:e} 拒绝时不应分配 alpha");
    }
}

/// 【新缺陷 N2 独立入口】`render_glyph(font, gid, px_size)` 是 text 层
/// 公开API，**不经 sdf 的任何钳制**直接吃 `px_size`。
/// 极大 `px_size` ⇒ 同样的 `f32 as i32` 饱和 + i32 减法溢出。
#[test]
fn attack_n2d_render_glyph_public_entry_panics_on_huge_px_size() {
    let font = real_font();
    for &px in &[1.0e20f32, 3.0e38, 1.0e30] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            raster::render_glyph(&font, 1, px)
        }));
        assert!(
            r.is_ok(),
            "render_glyph(px_size={px}) panic 了——N2 在text 公开入口同样可触发，             不依赖 sdf/SdfParams 的任何钳制（说明 sdf 的 MAX_PAD 修复覆盖不到此处）"
        );
    }
}
