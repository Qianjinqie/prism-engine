//! 共享约定唯一事实源（T04 common.rs）：画幅 / 帧率 / 场景帧表 /
//! `scene_duration` / 视觉系统颜色 / `Viewport` 坐标映射。
//!
//! 本文件是 `calculus_film_v2` 全 example 的**唯一定义处**：
//!
//! - **帧网格**：场景帧数 `SCENE_FRAMES` 与 `scene_duration(n) = n/30 - 0.002`
//!   （v1 同式），保证 `ClipSpec::frame_count = ceil(fps·duration) = n` 且
//!   浮点不跳帧；总帧 Σ2790 − 6×15 = 2700 = 90.000s；
//! - **坐标映射**：`Viewport::to_px` / `to_math` 是数学坐标 ↔ 设备像素的
//!   唯一映射处（含 y 轴翻转：数学 y↑ → 设备 y↓）。所有采样 / 符号落位 /
//!   动画几何一律经本结构取像素，禁止在别处内联映射公式；
//! - **颜色**：`hex_to_display`（÷255）是一切 2D 轨道颜色的唯一入口；
//!   `hex_to_linear`（sRGB 逆 EOTF）仅供 S7 3D 场景原料，2D 管线禁用。

/// 画幅宽（像素）。
pub const WIDTH: u32 = 960;
/// 画幅高（像素）。
pub const HEIGHT: u32 = 540;
/// 帧率（帧/秒；成片规格 30fps）。
pub const FPS: u32 = 30;

/// 各场景帧数（S1–S7）：
/// 248+308+496+496+496+436+310 = 2790。
pub const SCENE_FRAMES: [u32; 7] = [248, 308, 496, 496, 496, 436, 310];

/// 场景间转场时长（秒）：与 `render_short_concat_faded_frames` 内部固定的
/// `CONCAT_FADED_TRANS_SEC`（render.rs:1379）先例对齐——此处仅用于帧数
/// 推演，实际重叠由渲染入口内部消费。
pub const CONCAT_FADED_TRANS_SEC: f32 = 0.5;
/// 单次转场重叠帧数：`round(0.5 × 30) = 15`（`concat_with_transition`
/// 语义，render.rs:1267）。
pub const TRANS_FRAMES: usize = 15;
/// 转场次数：场景数 − 1。
pub const TRANS_COUNT: usize = SCENE_FRAMES.len() - 1;
/// 全片总帧数：Σ2790 − 6×15 = **2700 = 90.000s**。
pub const TOTAL_FRAMES: usize = {
    let mut sum: usize = 0;
    let mut i = 0;
    while i < SCENE_FRAMES.len() {
        sum += SCENE_FRAMES[i] as usize;
        i += 1;
    }
    sum - TRANS_COUNT * TRANS_FRAMES
};
/// 全片总时长（秒）：2700 / 30 = 90.000。
pub const TOTAL_DURATION_SEC: f32 = 90.0;

/// 帧数 → 场景时长（秒）：`n/FPS` 减 2ms 安全边际。
///
/// `ClipSpec::frame_count()` 为 `ceil(fps·duration)`：若 duration 因浮点
/// 表示略高于 `n/FPS`，ceil 会多进一帧。2ms 边际（×30fps = 0.06 帧）远大
/// 于 f32 在 10 秒量级的 ~1e-5 表示误差，保证 ceil 精确落到 `n` 帧。
/// v1 先例 `calculus_film.rs:85` 同式（15fps）。
pub fn scene_duration(frames: u32) -> f32 {
    frames as f32 / FPS as f32 - 0.002
}

// ---- 颜色（两个换算函数的职责边界） ----

/// 视觉 hex → 2D 轨道显示值（÷255）——**所有 2D Track 颜色**（曲线 /
/// 切线 / 网格 / 文案 / 面积填充 / HUD）的唯一入口。
///
/// 显示参照直传：2D 管线的 track 颜色即显示参照值（与 v1 先例逐位一致），
/// 渲染层另乘 env 信封。`const fn` 使调色板常量可在编译期求值。
pub const fn hex_to_display(rgb: [u8; 3]) -> [f32; 3] {
    [
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    ]
}

/// sRGB 逆 EOTF（hex → 线性光）——**仅 S7 3D 场景原料**（albedo /
/// radiance）使用；2D 管线禁用（整体发暗）。
///
/// 标准 sRGB 传递函数逆变换：`c <= 0.04045 → c/12.92`，否则
/// `((c + 0.055)/1.055)^2.4`（阈值 0.04045 为 sRGB 规范的 0.0031308
/// 线性段端点，`powf` 非常量故本函数运行期求值）。
pub fn hex_to_linear(rgb: [u8; 3]) -> [f32; 3] {
    let channel = |v: u8| -> f32 {
        let c = v as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    [channel(rgb[0]), channel(rgb[1]), channel(rgb[2])]
}

// ---- 调色板（单一事实源） ----

/// 主曲线青 `#35D9F2`。
pub const HEX_CURVE: [u8; 3] = [0x35, 0xD9, 0xF2];
/// 切线与强调金 `#FFD24A`。
pub const HEX_GOLD: [u8; 3] = [0xFF, 0xD2, 0x4A];
/// 导函数 / 累积链亮紫 `#B58CFF`。
pub const HEX_PURPLE: [u8; 3] = [0xB5, 0x8C, 0xFF];
/// 文案冷白 `#D6ECF7`。
pub const HEX_TEXT: [u8; 3] = [0xD6, 0xEC, 0xF7];

/// 主曲线青（显示值，直通 alpha 线性 RGBA 0..1，见 spec.rs 轨道字段语义）。
pub const CURVE_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_CURVE);
    [c[0], c[1], c[2], 1.0]
};
/// 切线 / 强调金。
pub const GOLD_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_GOLD);
    [c[0], c[1], c[2], 1.0]
};
/// 导函数 / 累积链亮紫。
pub const PURPLE_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_PURPLE);
    [c[0], c[1], c[2], 1.0]
};
/// 文案冷白。
pub const TEXT_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_TEXT);
    [c[0], c[1], c[2], 1.0]
};
/// 纯白（网格 / 坐标轴 / 刻度基色）。
pub const WHITE_RGBA: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// 坐标网格白 alpha 0.10。
pub const GRID_RGBA: [f32; 4] = [1.0, 1.0, 1.0, 0.10];
/// 坐标主轴白 alpha 0.90（网格之上的强调轴线）。
pub const AXIS_RGBA: [f32; 4] = [1.0, 1.0, 1.0, 0.90];
/// 刻度线白 alpha 0.75。
pub const TICK_RGBA: [f32; 4] = [1.0, 1.0, 1.0, 0.75];

/// 面积填充半透明青 alpha 0.18（区间下端）。
pub const AREA_A18_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_CURVE);
    [c[0], c[1], c[2], 0.18]
};
/// 面积填充半透明青 alpha 0.26（区间中值）。
pub const AREA_A26_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_CURVE);
    [c[0], c[1], c[2], 0.26]
};
/// 面积填充半透明青 alpha 0.38（区间上端）。
pub const AREA_A38_RGBA: [f32; 4] = {
    let c = hex_to_display(HEX_CURVE);
    [c[0], c[1], c[2], 0.38]
};

/// 2D 轨道调色板全集（有限域自证用；新增轨道颜色必须登记于此）。
pub const PALETTE_2D: [[f32; 4]; 11] = [
    CURVE_RGBA,
    GOLD_RGBA,
    PURPLE_RGBA,
    TEXT_RGBA,
    WHITE_RGBA,
    GRID_RGBA,
    AXIS_RGBA,
    TICK_RGBA,
    AREA_A18_RGBA,
    AREA_A26_RGBA,
    AREA_A38_RGBA,
];

/// S3 深蓝黑渐变（top / bottom；「深蓝黑渐变区分场景情绪」，具体 hex
/// 值未钉死，取贴近既有先例 `calculus_film.rs:322` 的值）。
pub const BG_DEEP_BLUE: ([f32; 3], [f32; 3]) = (
    hex_to_display([0x0A, 0x12, 0x28]),
    hex_to_display([0x02, 0x04, 0x0C]),
);
/// S4 深青黑渐变（v1 先例 `calculus_film.rs:380`）。
pub const BG_DEEP_CYAN: ([f32; 3], [f32; 3]) = (
    hex_to_display([0x05, 0x17, 0x1A]),
    hex_to_display([0x02, 0x06, 0x0B]),
);
/// S5 深紫黑渐变（v1 先例 `calculus_film.rs:437`）。
pub const BG_DEEP_PURPLE: ([f32; 3], [f32; 3]) = (
    hex_to_display([0x14, 0x0A, 0x22]),
    hex_to_display([0x07, 0x04, 0x10]),
);

/// 场景背景渐变表（S3 / S4 / S5；供 T05 装配 `Background::Gradient` 取用）。
pub const BG_GRADIENTS: [([f32; 3], [f32; 3]); 3] = [BG_DEEP_BLUE, BG_DEEP_CYAN, BG_DEEP_PURPLE];

/// 把 `v` 压进数学域 `[lo, hi]`（轴 / 基线贴边用）。
///
/// `f32::clamp` 在 `min > max` 或界为 NaN 时 panic——本辅助以 `lo <= hi`
/// 守卫（NaN 界同样落入 else），非法域退化为域中心，契约「非法输入不
/// panic」。
pub fn clamp_domain(v: f32, lo: f32, hi: f32) -> f32 {
    if lo <= hi {
        v.clamp(lo, hi)
    } else {
        (lo + hi) * 0.5
    }
}

// ---- Viewport：数学坐标 ↔ 设备像素的唯一映射 ----

/// 视口：数学域 `[x0, x1] × [y0, y1]` 到绘图区（设备像素）的仿射映射。
///
/// 不变量（`is_valid` 机械核对）：全部字段有限、`px_per_unit > 0`。
/// 映射含 **y 轴翻转**：数学 y↑ → 设备 y↓（v1 先例
/// `calculus_film.rs:119` 的 `420.0 - 90.0·y`）。所有 `sample_curve` /
/// `riemann_rects` 采样、符号与公式落位、动画几何闭包一律经
/// [`Viewport::to_px`] 取像素——出现第二处映射公式即违反 §7.1。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// 数学域左界。
    pub x0: f32,
    /// 数学域右界。
    pub x1: f32,
    /// 数学域下界。
    pub y0: f32,
    /// 数学域上界。
    pub y1: f32,
    /// 绘图区中心（设备像素 x）。
    pub cx: f32,
    /// 绘图区中心（设备像素 y）。
    pub cy: f32,
    /// 每数学单位的设备像素数（> 0）。
    pub px_per_unit: f32,
}

impl Viewport {
    /// 直接构造视口（调用方自定缩放；T05 装配首选 [`Viewport::fit_plot`]）。
    pub fn new(
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        cx: f32,
        cy: f32,
        px_per_unit: f32,
    ) -> Self {
        Viewport { x0, x1, y0, y1, cx, cy, px_per_unit }
    }

    /// 把数学域**等比**套入以 `(cx, cy)` 为中心、半宽 `half_w` / 半高
    /// `half_h`（设备像素）的绘图区：`px_per_unit = min(half_w / (dx/2),
    /// half_h / (dy/2))`，两向同缩放（无拉伸失真）。
    ///
    /// 非法输入（零 / 负 / 非有限域宽高或半区）不 panic：退化域的宽高
    /// 以 `f32::EPSILON` 兜底，缩放非有限时落 0（此时 `is_valid()`
    /// 为 false，映射输出有限但无意义）。
    pub fn fit_plot(
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        cx: f32,
        cy: f32,
        half_w: f32,
        half_h: f32,
    ) -> Self {
        let dx = (x1 - x0).max(f32::EPSILON);
        let dy = (y1 - y0).max(f32::EPSILON);
        let hw = if half_w.is_finite() { half_w.max(0.0) } else { 0.0 };
        let hh = if half_h.is_finite() { half_h.max(0.0) } else { 0.0 };
        let scale = (hw / (dx * 0.5)).min(hh / (dy * 0.5));
        let scale = if scale.is_finite() { scale.max(0.0) } else { 0.0 };
        Viewport { x0, x1, y0, y1, cx, cy, px_per_unit: scale }
    }

    /// 机械校验：全部字段有限且 `px_per_unit > 0`。
    pub fn is_valid(&self) -> bool {
        self.x0.is_finite()
            && self.x1.is_finite()
            && self.y0.is_finite()
            && self.y1.is_finite()
            && self.cx.is_finite()
            && self.cy.is_finite()
            && self.px_per_unit.is_finite()
            && self.px_per_unit > 0.0
    }

    /// 数学坐标 → 设备像素（**唯一映射处**，含 y 翻转）。
    ///
    /// 数学域中心映射到绘图区中心 `(cx, cy)`：
    /// `px = cx + (mx − x_mid)·s`，`py = cy − (my − y_mid)·s`。
    /// 对任意有限输入输出有限（`px_per_unit` 为 0 时收敛到 `(cx, cy)`）。
    pub fn to_px(&self, mx: f32, my: f32) -> [f32; 2] {
        let x_mid = (self.x0 + self.x1) * 0.5;
        let y_mid = (self.y0 + self.y1) * 0.5;
        let s = if self.px_per_unit.is_finite() { self.px_per_unit } else { 0.0 };
        [
            self.cx + (mx - x_mid) * s,
            self.cy - (my - y_mid) * s,
        ]
    }

    /// 设备像素 → 数学坐标（**仅供单测与视口裁剪**，§4.2 / §7.1）。
    ///
    /// `to_px` 的逆映射（含 y 翻转还原）。非法视口（`px_per_unit <= 0`
    /// 或非有限）不 panic、不产出非有限值：退化为数学域中心。
    pub fn to_math(&self, px: f32, py: f32) -> [f32; 2] {
        let x_mid = (self.x0 + self.x1) * 0.5;
        let y_mid = (self.y0 + self.y1) * 0.5;
        if !self.px_per_unit.is_finite() || self.px_per_unit <= 0.0 {
            return [x_mid, y_mid];
        }
        [
            x_mid + (px - self.cx) / self.px_per_unit,
            y_mid - (py - self.cy) / self.px_per_unit,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_px_math_roundtrip_with_y_flip() {
        // x∈[-2,2]、y∈[-1,3]（4×4 域）套入中心 (480,270)、半区 200×150：
        // 等比取 min(200/2, 150/2) = 75 px/unit（无拉伸）。
        let vp = Viewport::fit_plot(-2.0, 2.0, -1.0, 3.0, 480.0, 270.0, 200.0, 150.0);
        assert!(vp.is_valid());
        assert!((vp.px_per_unit - 75.0).abs() < 1e-3, "等比缩放 75px/unit，实际 {}", vp.px_per_unit);
        // 数学中心 (0,1) → 绘图区中心 (480,270)。
        let c = vp.to_px(0.0, 1.0);
        assert!((c[0] - 480.0).abs() < 1e-3 && (c[1] - 270.0).abs() < 1e-3);
        // (1,2)：中心右 1 单位、上 1 单位 → (555, 195)。
        let p = vp.to_px(1.0, 2.0);
        assert!((p[0] - 555.0).abs() < 1e-3 && (p[1] - 195.0).abs() < 1e-3);
        // 往返（含 y 翻转）误差 < 1e-3。
        let m = vp.to_math(p[0], p[1]);
        assert!((m[0] - 1.0).abs() < 1e-3 && (m[1] - 2.0).abs() < 1e-3, "往返 {:?}", m);
        // y 翻转方向性：数学 y↑ ⇒ 设备 y↓。
        let above = vp.to_px(0.0, 2.5);
        let below = vp.to_px(0.0, -0.5);
        assert!(above[1] < vp.cy && below[1] > vp.cy, "数学 y 大必须在设备上半区");
    }

    #[test]
    fn viewport_direct_construction_and_invalid_inputs_safe() {
        let vp = Viewport::new(-1.0, 1.0, -1.0, 1.0, 480.0, 270.0, 100.0);
        assert!(vp.is_valid());
        let p = vp.to_px(0.5, -0.5);
        assert!((p[0] - 530.0).abs() < 1e-3 && (p[1] - 320.0).abs() < 1e-3);
        // 非法视口：不 panic、输出有限；to_math 退化到域中心。
        let bad = Viewport::new(0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 0.0);
        assert!(!bad.is_valid());
        let q = bad.to_px(0.5, 0.5);
        assert!(q[0].is_finite() && q[1].is_finite(), "非法视口 to_px 不产出非有限值");
        let m = bad.to_math(10.0, 20.0);
        assert!(m[0].is_finite() && m[1].is_finite(), "非法视口 to_math 不 panic");
        assert!((m[0] - 0.5).abs() < 1e-6 && (m[1] - 0.5).abs() < 1e-6, "退化到域中心");
        // fit_plot 零 / 负 / 非有限输入不 panic。
        let degenerate = Viewport::fit_plot(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -5.0, f32::NAN);
        assert!(!degenerate.is_valid());
        let d = degenerate.to_px(0.0, 0.0);
        assert!(d[0].is_finite() && d[1].is_finite());
    }

    #[test]
    fn scene_duration_aligns_frame_grid() {
        // scene_duration(248) = 248/30 − 0.002（v1 同式，逐位一致）。
        let d = scene_duration(248);
        assert_eq!(d.to_bits(), (248.0f32 / 30.0 - 0.002).to_bits());
        // ceil(fps·duration) 精确 = 248（不跳帧）。
        assert_eq!((FPS as f32 * d).ceil(), 248.0);
        // 全部场景帧数均不跳帧。
        for &n in SCENE_FRAMES.iter() {
            assert_eq!(
                (FPS as f32 * scene_duration(n)).ceil(),
                n as f32,
                "场景帧数 {n} 的 duration 必须 ceil 回 {n}"
            );
        }
    }

    #[test]
    fn frame_table_sums_to_total() {
        let sum: u32 = SCENE_FRAMES.iter().sum();
        assert_eq!(sum, 2790, "Σ场景帧数");
        assert_eq!(sum as usize - TRANS_COUNT * TRANS_FRAMES, TOTAL_FRAMES);
        assert_eq!(TOTAL_FRAMES, 2700, "Σ2790 − 6×15 = 2700");
        assert_eq!(TOTAL_FRAMES as f32 / FPS as f32, 90.0);
        assert_eq!(TOTAL_DURATION_SEC, 90.0);
        assert_eq!(TRANS_FRAMES, (CONCAT_FADED_TRANS_SEC * FPS as f32).round() as usize);
    }

    #[test]
    fn hex_to_display_matches_manual_computation() {
        // #35D9F2 = (53, 217, 242)/255 = (0.208, 0.851, 0.949)（手算值）。
        let c = hex_to_display(HEX_CURVE);
        assert!((c[0] - 0.208).abs() < 1e-3, "R {}", c[0]);
        assert!((c[1] - 0.851).abs() < 1e-3, "G {}", c[1]);
        assert!((c[2] - 0.949).abs() < 1e-3, "B {}", c[2]);
        // 逐通道 ÷255（含 0 与满量程端点）。
        assert_eq!(hex_to_display([0xFF, 0xFF, 0xFF]), [1.0, 1.0, 1.0]);
        assert_eq!(hex_to_display([0x00, 0x00, 0x00]), [0.0, 0.0, 0.0]);
        let bg = hex_to_display([0x0A, 0x12, 0x28]);
        assert!((bg[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((bg[1] - 18.0 / 255.0).abs() < 1e-6);
        assert!((bg[2] - 40.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn hex_to_linear_endpoints_and_monotonic() {
        // 端点：黑 → 0，白 → 1。
        assert_eq!(hex_to_linear([0x00, 0x00, 0x00]), [0.0, 0.0, 0.0]);
        let w = hex_to_linear([0xFF, 0xFF, 0xFF]);
        for ch in w {
            assert!((ch - 1.0).abs() < 1e-6, "白 → 线性 1，实际 {ch}");
        }
        // 中灰 128/255 ≈ 0.502 → 线性 ≈ 0.2158（sRGB 标准值）。
        let mid = hex_to_linear([0x80, 0x80, 0x80]);
        assert!((mid[0] - 0.2158).abs() < 5e-3, "中灰线性值 {}", mid[0]);
        // 全定义域单调不降且落在 [0,1]。
        let mut prev = -1.0f32;
        for v in 0..=255u8 {
            let l = hex_to_linear([v, v, v])[0];
            assert!((0.0..=1.0).contains(&l), "线性值越界 {l}");
            assert!(l >= prev, "必须单调不降（v={v}）");
            prev = l;
        }
    }

    #[test]
    fn palette_and_backgrounds_finite() {
        // 调色板：全部有限、落在 [0,1]（含 alpha）。
        for (i, c) in PALETTE_2D.iter().enumerate() {
            for ch in c {
                assert!(ch.is_finite() && (0.0..=1.0).contains(ch), "PALETTE_2D[{i}] 越界: {ch}");
            }
        }
        // 曲线青与调色板一致；网格 alpha 0.10；面积填充 alpha ∈ [0.18, 0.38]。
        assert!((CURVE_RGBA[0] - 0.208).abs() < 1e-3 && (CURVE_RGBA[3] - 1.0).abs() < 1e-6);
        assert!((GRID_RGBA[3] - 0.10).abs() < 1e-6);
        for area in [AREA_A18_RGBA, AREA_A26_RGBA, AREA_A38_RGBA] {
            assert!(area[3] >= 0.18 - 1e-6 && area[3] <= 0.38 + 1e-6, "面积填充 alpha 越界 {}", area[3]);
            assert!((area[0] - CURVE_RGBA[0]).abs() < 1e-6, "面积填充必须取曲线青");
        }
        // 背景渐变：三组均有限、top 不暗于 bottom（上亮下沉的纵深惯例）。
        for (top, bottom) in BG_GRADIENTS.iter() {
            for ch in top.iter().chain(bottom.iter()) {
                assert!(ch.is_finite() && (0.0..=1.0).contains(ch), "背景渐变越界 {ch}");
            }
            let lum_top = top[0] + top[1] + top[2];
            let lum_bottom = bottom[0] + bottom[1] + bottom[2];
            assert!(lum_top >= lum_bottom, "背景应上亮下沉");
        }
    }
}
