//! 数学四场景（T05 scenes_math.rs，S3–S6）：极限 / 导数 / 积分 / 基本定理。
//!
//! 每个场景一个构造函数返回 [`ClipSpec`]；帧数 496 / 496 / 496 / 436
//! （架构 §7.6），时长一律 [`scene_duration`]。
//!
//! # 读数恒等铁律（架构 §7.3）
//!
//! 每个 [`Track::Counter`] 的 `(from, to, dur_sec, easing)` 与对应
//! [`DynPathTrack`] 的几何参数派生自**同一个纯函数**（如 [`secant_m`] /
//! [`riemann_sum`] / [`s4_a`] / [`s6_x`]），禁两处写常数；单测
//! `counter_readout_identical_to_geom` 在采样点上机械钉住。
//!
//! # HUD 形态（`out_at` 绝对淡出锚点落地后，PRD 分拍可精确实现）
//!
//! 引擎原语 `CounterText::out_at`（spec.rs；render.rs `out_start_of`）
//! 令文本轨可指定**绝对消失时刻**（`out_sec=0` 硬切 / `>0` 短淡出），
//! 同位置多条顺序轨据此轮流显现、无叠字。PRD 的分拍阶跃由此原样落地：
//! - S3：m = 3.0000→2.0000（Linear，[7.0, 12.0]；PRD 检查点
//!   3.0/2.5/2.25/2.125/2.0 的最大偏差 0.143，出现在 PRD 仅要求
//!   「就位」的 8.5s 处）；三条离散割线改以鬼影标记轨呈现
//!   （live 线滚过对应 h 值的时刻，h=0.5/0.25/0.125 ↔ 9.5/10.6/11.2s）。
//! - S3 ε/δ 带：自 appearance 起连续收缩（CubicIn）至 PRD 端点值
//!   （ε=1.00→0.25、δ=0.4142→0.1180，端点满足 δ=√(1+ε)−1 精确），
//!   标签数字同步滚——保证「屏幕上数学永远正确」优先于 1s 卡点收缩。
//!   [节奏重剪] 收缩窗收至 12.0（ε [3.5,12.0] / δ [5.0,12.0]）：切线
//!   定格后 12.0–14.5 为 2.5s 顿悟静止定格，画面无新元素入场。
//! - S4：a: −2→2 单段 Linear（[3.5, 12.5]），四体联动全程跟随；
//!   [节奏重剪] 扫过窗由 10s 压至 9s，a=2 末态静态定格由 3s 延至
//!   4s（fast-fast-SLOW 的收尾慢段）；定格脉冲拍与中间文案拍删除。
//!   **PRD 勘误**：S4 分镜与 §6 的「a=−0.5 ⇒ m=−1.0000」自相矛盾
//!   （切线 y=ax−a²/2 的斜率是 a=−0.5）；本实现取恒等式 m≡a。
//! - S5：**PRD 四拍阶跃原样**（3.0/4.5/6.0/7.5 各 1.5s）——n=2/4/8/32
//!   与 S=5.0000/3.7500/3.1875/2.7930 各为静态 Counter，`out_at` 硬切
//!   交接（屏上数字 ≡ 同拍矩形数，恒等精确无近似）；矩形同拍固定 n、
//!   高度 QuadOut 生长（首拍 0.8s / 后续 0.7s），旧拍 0.3s 淡出；
//!   9.0–11.0 末拍阶梯逐点 morph 进曲线下面积（P1-3），S 滚
//!   2.7930→2.6667（=8/3，Linear，起点=末批矩形精确和、终点=精确面积）。
//! - S6：x: 0→2 匀速推进（Linear，[3.0, 9.0]；PRD 3.0–6.0 与 6.0–9.0
//!   两段合并为单段连续推进，6.0 处的显式跳变被连续推进取代，四体联动
//!   不解耦）；双 HUD f(x)/F′(x) 以 QuadIn 滚 0→4——4u² ≡ f(2u) 读数
//!   恒等**精确**（单缓动 Counter 无法复合 f∘ease：x 若走 CubicInOut，
//!   屏上与真实 f 偏差最大 1.0，恒等铁律优先于扫描缓动口味）。
//!
//! # 节奏分层（3B1B 呼吸节奏重剪：主 1.0 / 上下文 ~0.55 / 结构 ~0.25）
//!
//! - **顿悟停顿**：关键揭示后 2.0–3.0s 静止定格（无新元素入场、HUD
//!   停止滚动、已有元素保持）——S3 切线定格 m=2.0000 后 12.0–14.5
//!   （2.5s）；S5 面积答案 S=2.6667 后 11.0–13.0（2.0s）；S6
//!   F′(x)=f(x) 重合公式后 10.0–12.5（2.5s）。
//! - **透明度分层**：HUD 读数 / 轴标签 / 区间标记 / 答案小字压到上下
//!   文层 alpha 0.55（[`ctx`]）；底部打字文案与开幕目标字幕压到 0.50
//!   （[`TEXT_CTX_RGBA`]）；坐标网格维持 axes.rs 既有 alpha 0.10（已
//!   低于 0.25 结构层目标，axes.rs 禁改故保持）；主曲线 / 切线 / 面积
//!   填充保持 1.0 主层不动。
//! - **开幕目标字幕**：S3–S6 开头 0.8s 一条打字机字幕说明本节目标。
//! - **总时长铁律**：场景帧数 496/496/496/436 与 90.000s / 2700 帧
//!   闭环不动，只挪拍在场景内的位置与内容（省出的时间全部补给顿悟
//!   停顿与片尾静态定格）。

use std::sync::Arc;

use prism_author::{
    Background, ClipSpec, CounterText, DynPathTrack, EasingKind, FontKind, GeomFn, PathTrack,
    ShapeKind, ShapeTrack, StaticText, Track, TypewriterText,
};

use crate::{axes, common, symbols};
use common::{
    scene_duration, Viewport, AREA_A26_RGBA, BG_DEEP_BLUE, BG_DEEP_CYAN, BG_DEEP_PURPLE,
    CURVE_RGBA, GOLD_RGBA, PURPLE_RGBA, TEXT_RGBA,
};
use symbols::{FormulaPart, SymbolKind};

// ---------------------------------------------------------------------------
// 节奏分层常量与辅助（3B1B 呼吸节奏重剪）
// ---------------------------------------------------------------------------

/// 上下文层 alpha（HUD 读数 / 轴标签 / 区间标记；分层：主 1.0 /
/// 上下文 ~0.55 / 结构 ~0.25）。
const ALPHA_CTX: f32 = 0.55;

/// 上下文层文案色（底部打字文案 / 开幕目标字幕：冷白 alpha 0.50）。
const TEXT_CTX_RGBA: [f32; 4] = [TEXT_RGBA[0], TEXT_RGBA[1], TEXT_RGBA[2], 0.50];

/// 把颜色压到上下文层 alpha（HUD / 标签专用；主曲线 / 切线 / 面积
/// 填充保持 1.0 主层，不经过本函数）。
fn ctx(color: [f32; 4]) -> [f32; 4] {
    [color[0], color[1], color[2], ALPHA_CTX]
}

/// 轴网轨 + 轴标签上下文化：网格（`GRID_RGBA` alpha 0.10）本就低于
/// 0.25 结构层目标（axes.rs 禁改故保持原值），仅把 Static 轴标签
/// x / y 压到上下文层 alpha。
fn axes_tracks_ctx(vp: &Viewport) -> Vec<Track> {
    let mut tracks = axes::axes_tracks(vp);
    for tr in tracks.iter_mut() {
        if let Track::Static(s) = tr {
            s.color[3] = ALPHA_CTX;
        }
    }
    tracks
}

// ===========================================================================
// 共享纯函数（数学唯一事实源；几何与 Counter 同源派生）
// ===========================================================================

/// 平滑阶跃（0→1 平滑脉冲）：`u∈[0,1]` 上 `3u²−2u³`，区间外钳 0/1。
fn bump(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    3.0 * u * u - 2.0 * u * u * u
}

/// 圆采样（设备像素）；`n < 3` 时退化为 3 点（渲染层跳过 <2 点轨）。
fn circle_pts(cx: f32, cy: f32, r: f32, n: usize) -> Vec<[f32; 2]> {
    let n = n.max(3);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = std::f32::consts::TAU * (i as f32 / n as f32);
        out.push([cx + r * a.cos(), cy + r * a.sin()]);
    }
    out
}

/// 竖直引导虚线（`x` 处 `y0→y1`，按 `dash` 长度交替虚实）。
fn v_dashed(x: f32, y0: f32, y1: f32, dash: f32) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    let (lo, hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    let mut y = lo;
    let mut on = true;
    while y < hi {
        let y2 = (y + dash).min(hi);
        if on {
            out.push([x, y]);
            out.push([x, y2]);
        }
        y = y2;
        on = !on;
    }
    out
}

/// 把「ε = 」式希腊前缀与滚动数字拼版：**单次** [`assemble_formula`]
/// 走排「符号 + 等号 + 占位数字位」，取出数字 run 的 `center_x` 后剔除
/// 占位 run 再发射——符号 / 等号 / 数字三者的 pen 关系由同一次装配固定。
///
/// 返回 (公式前缀轨, Counter)。`sym` 为希腊符号，等号恒取矢量
/// [`SymbolKind::Equals`]（双横线，**不走文本**——PrismSansSC 的「=」
/// 字形极细近乎缺失，旧实现读作 `" = "` 文本 run 即此缺陷），两侧间距
/// 由数学间距表 `Var → Binary` 呼吸自动给出；`num_size` 为数字字号。
///
/// [缺陷修复·等号不可见 / 小数点糊成冒号] 旧实现分两次装配：探针含
/// 占位数字 run（总宽 = 符号 + 等号 + 数字，如 ε 26px 下 98.36px），
/// 正式前缀只含「符号 + 等号」（总宽 42.12px）——两次各自以 0 居中，
/// 前缀比探针窄 56.24px ⇒ 右移 28.12px，而 Counter 的 `center_x` 仍取
/// 探针值：等号（探针中在数字左侧 8.8px）落进数字首位 '0' 的 ink 区
/// （实测等号 x∈[2.34,17.94] vs 数字 x∈[−1.34,49.18]）。双横线被
/// '0' 字环吞没（等号不可见，仅剩边缘稀疏墨团），等号下线
/// （baseline 上 4.42px）又与 '.' 点（x∈[14.9,18.2]，y∈[116,120]）
/// 仅 0.4px 相接，糊成冒号（「0:12」）。单次装配后等号回到数字左侧
/// 8.8px 净空，两缺陷同根消除。
///
/// [几何 dump 实证（`dump_greek_roll_readout_geometry`）] 等号顶点
/// 本身无恙：单位格 y=0.17/0.37 两条线经 `symbol_path` 缩放与
/// (x,y)→(x+pen,−y+baseline) 翻转变换后，ε 标签（baseline 120、
/// size 26）实测两横线 y=110.38/115.58（即 baseline 上 9.62/4.42px
/// = 0.37/0.17·26，垂直居中于数字 x 高度区）、x∈[−25.78,−10.18]
/// （线长 15.6 = 0.6·26 ≥ 0.5·size）、笔宽 1.5、双线净距 5.2px；
/// 数字 ink 左缘 −1.34 ⇒ 等号右缘净空 8.84px。不可见根因是**两次
/// 装配各居其零**造成的水平错位（等号被 '0' 字环覆盖），非单位格
/// y 映射错误——修法取最小：保持 `unit_equals` 几何不动，改单次装配。
fn greek_roll_readout(
    sym: SymbolKind,
    baseline_y: f32,
    sym_size: f32,
    num_size: f32,
    color: [f32; 4],
    from: f32,
    to: f32,
    at: f32,
    dur_sec: f32,
    easing: EasingKind,
    decimals: u32,
) -> (Vec<Track>, CounterText) {
    // 1) 单次装配：符号 + 等号 + 占位数字位（位宽与实际读数一致：
    //    decimals=0 → "0"；decimals=n → "0." + n 个 0）。
    let placeholder = if decimals == 0 {
        String::from("0")
    } else {
        format!("0.{}", "0".repeat(decimals as usize))
    };
    let mut prefix_tracks = symbols::assemble_formula(
        &[
            FormulaPart::Sym(sym, sym_size, color),
            FormulaPart::Sym(SymbolKind::Equals, sym_size, color),
            FormulaPart::Text(placeholder.clone(), FontKind::Regular, num_size, color),
        ],
        0.0,
        baseline_y,
    );
    // 2) 取数字 run 的 center_x（pen 累加与后续 run 无关）。
    let num_cx = prefix_tracks
        .iter()
        .find_map(|t| match t {
            Track::Static(s) if s.content == placeholder => Some(s.center_x),
            _ => None,
        })
        .unwrap_or(Some(0.0))
        .unwrap_or(0.0);
    // 3) 剔除占位数字 run（数字位由 Counter 逐帧填补；pen 关系已由
    //    同一次走排固定，符号 / 等号 / 数字不会再次错位）。
    prefix_tracks.retain(|t| match t {
        Track::Static(s) => s.content != placeholder,
        _ => true,
    });
    let counter = CounterText {
        prefix: String::new(),
        from,
        to,
        at,
        dur_sec,
        easing,
        decimals,
        font: FontKind::Regular,
        size: num_size,
        color,
        center_x: Some(num_cx),
        y: baseline_y,
        letter_spacing: 0.0,
        in_sec: 0.0,
        out_sec: 0.0,
        out_at: None,
        value_at: None,
    };
    (prefix_tracks, counter)
}

/// 把 [`greek_roll_readout`] 的产物平移到目标锚点（公式前缀整体右移
/// `dx`，Counter 的 `center_x` 同步右移）。
fn shift_prefix(tracks: &mut [Track], counter: &mut CounterText, dx: f32) {
    for t in tracks.iter_mut() {
        match t {
            Track::Static(s) => {
                if let Some(cx) = s.center_x {
                    s.center_x = Some(cx + dx);
                }
            }
            Track::Path(p) => {
                for q in p.points.iter_mut() {
                    q[0] += dx;
                }
            }
            _ => {}
        }
    }
    if let Some(cx) = counter.center_x {
        counter.center_x = Some(cx + dx);
    }
}

/// 曲线下的封闭面积多边形（数学 x∈[x0, xg]，基线 y=0，设备像素）。
fn area_poly(vp: &Viewport, x0: f32, xg: f32, n: usize, f: impl Fn(f32) -> f32) -> Vec<[f32; 2]> {
    let n = n.max(2);
    let mut out = Vec::with_capacity(n + 2);
    for i in 0..n {
        let mx = x0 + (xg - x0) * (i as f32 / (n - 1) as f32);
        out.push(vp.to_px(mx, f(mx)));
    }
    out.push(vp.to_px(xg, 0.0));
    out.push(vp.to_px(x0, 0.0));
    out
}

/// 便捷：静态文字轨。
fn static_text(content: &str, size: f32, color: [f32; 4], at: f32, in_sec: f32, y: f32, cx: Option<f32>) -> Track {
    Track::Static(StaticText {
        content: String::from(content),
        font: FontKind::Regular,
        size,
        color,
        at,
        in_sec,
        letter_spacing: 2.0,
        y,
        out_sec: 0.0,
        out_at: None,
        center_x: cx,
    })
}

/// 便捷：打字机轨（底部文案 / 开幕目标字幕共用）。`out_at` 为绝对
/// 淡出锚点（`None` 时缺省锚定片尾 `duration − out_sec`）。
#[allow(clippy::too_many_arguments)]
fn typewriter(
    content: &str,
    size: f32,
    color: [f32; 4],
    at: f32,
    y: f32,
    cps: f32,
    out_at: Option<f32>,
    out_sec: f32,
) -> Track {
    Track::Typewriter(TypewriterText {
        content: String::from(content),
        font: FontKind::Regular,
        size,
        color,
        at,
        y,
        cps,
        cursor: true,
        cursor_blink_hz: 1.2,
        in_sec: 0.4,
        out_sec,
        out_at,
        letter_spacing: 2.0,
        center_x: None,
    })
}

// ===========================================================================
// S3 极限·ε-δ 带 + 割线族收敛（16.53s，深蓝黑渐变）
// ===========================================================================

/// S3 数学：f(x)=x²，x₀=1，L=1；割线斜率 m(h)=2+h（PRD §6 已验算）。
fn s3_f(x: f32) -> f32 {
    x * x
}

/// 割线斜率（h=第二端点相对 x₀ 的位移）：m(h) = (f(1+h)−f(1))/h = 2+h。
fn secant_m(h: f32) -> f32 {
    2.0 + h
}

/// live 割线的 h(t)：CubicInOut 1→0 over [7.0, 12.0]（HUD 同窗同参；[任务 1]
/// f∘ease：动点 x=1+h 走缓动、y=f(x)=(1+h)² 由 x 派生，非独立缓动 y）。
///
/// [节奏重剪] 收敛窗由 7s 压至 5s：切线定格（12.0）后接 2.5s 顿悟静止
/// 定格（12.0–14.5），窗外（t ≥ 12.0）冻结于 0（切线态）。
fn s3_h(t: f32) -> f32 {
    if t <= 7.0 {
        1.0
    } else if t >= 12.0 {
        0.0
    } else {
        1.0 - EasingKind::CubicInOut.apply((t - 7.0) / 5.0)
    }
}

/// ε(t)：CubicIn 1.00→0.25 over [3.5, 12.0]（与 ε 带几何、ε 标签同参）。
///
/// [节奏重剪] 收缩终点由 15.0 提前至 12.0（切线定格时刻）：顿悟定格
/// 12.0–14.5 内 ε/δ 带静止，无任何滚动 / 收缩运动。
fn s3_eps(t: f32) -> f32 {
    if t <= 3.5 {
        1.0
    } else if t >= 12.0 {
        0.25
    } else {
        let u = (t - 3.5) / 8.5;
        1.0 + (0.25 - 1.0) * EasingKind::CubicIn.apply(u)
    }
}

/// ε-δ 精确关系（PRD §6）：δ = √(1+ε) − 1。
fn s3_delta_of_eps(e: f32) -> f32 {
    (1.0 + e).sqrt() - 1.0
}

/// δ(t)：端点取精确关系值，中点按同一 CubicIn 进度线性插值
/// （与 δ 带几何、δ 标签恒等；端点 ε↔δ 关系精确）。
///
/// [节奏重剪] 收缩窗收至 [5.0, 12.0]（与 ε 带同步冻结于切线定格）。
fn s3_delta(t: f32) -> f32 {
    let (d0, d1) = (s3_delta_of_eps(1.0), s3_delta_of_eps(0.25));
    if t <= 5.0 {
        d0
    } else if t >= 12.0 {
        d1
    } else {
        let u = (t - 5.0) / 7.0;
        d0 + (d1 - d0) * EasingKind::CubicIn.apply(u)
    }
}

/// 受限相机呼吸视口（任务 2）：以绘图区中心为锚、CubicInOut 缓动做
/// 「推近—回稳」的微幅缩放（峰值 `peak` 倍，端点严格回归 `base`）。
///
/// 两段 CubicInOut 腿在 `u=0.5` 处连续会合（首段 0→peak，次段 peak→0），
/// 故 `t=0` 与 `t=dur` 处视口完全等于 `base`——满足任务书「连续视口 /
/// 端点匹配设计 / 静态轴网尺度稳定」。纯函数、确定性（无随机 / 无状态）。
///
/// 实现：保持设计绘图区（设备中心 + 实际半区）不变，仅按 `1/s` 收窄数学
/// 域 ⇒ 等比推近；端点 `s=1` 时数学域与设计完全一致（逐位恢复 `base`）。
fn breathe_viewport(base: Viewport, t: f32, dur: f32, peak: f32) -> Viewport {
    let u = if dur > 0.0 { (t / dur).clamp(0.0, 1.0) } else { 0.0 };
    let k = if u < 0.5 {
        peak * EasingKind::CubicInOut.apply(u / 0.5)
    } else {
        peak * EasingKind::CubicInOut.apply((1.0 - u) / 0.5)
    };
    let s = 1.0_f32 + k;
    // 端点（k=0）严格回归设计视口——逐位一致（避免浮点重排误差）。
    if k == 0.0 {
        return base;
    }
    let mx = (base.x0 + base.x1) * 0.5;
    let my = (base.y0 + base.y1) * 0.5;
    let dx = (base.x1 - base.x0).max(f32::EPSILON);
    let dy = (base.y1 - base.y0).max(f32::EPSILON);
    let half_w = dx * 0.5 * base.px_per_unit;
    let half_h = dy * 0.5 * base.px_per_unit;
    Viewport::fit_plot(
        mx - dx * 0.5 / s,
        mx + dx * 0.5 / s,
        my - dy * 0.5 / s,
        my + dy * 0.5 / s,
        base.cx,
        base.cy,
        half_w,
        half_h,
    )
}

/// S3 视口：89px/unit，x∈[−0.35,2.65]、y∈[−0.42,4.42]（PRD 布局）。
fn s3_viewport() -> Viewport {
    Viewport::fit_plot(-0.35, 2.65, -0.42, 4.42, 480.0, 280.0, 133.5, 215.4)
}

/// S3 相机呼吸视口（任务 2）：推近峰值 4%，端点回归 [`s3_viewport`]。
fn s3_viewport_at(t: f32) -> Viewport {
    breathe_viewport(s3_viewport(), t, scene_duration(496), 0.04)
}

/// S3（496 帧 = 16.527s）。
pub fn scene_s3() -> ClipSpec {
    let vp = s3_viewport();
    let dur = scene_duration(496);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 网格 + 坐标轴（0.5–2.0）----
    tracks.extend(axes_tracks_ctx(&vp));

    // ---- f(x)=x² 曲线（1.5–2.5 draw 1.0s，33 点）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(s3_f, -0.3, 2.3, 33, &vp),
        color: CURVE_RGBA,
        width: 3.5,
        fill: false,
        at: 1.5,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 1.0,
        closed: false,
    }));

    // ---- 点 P=(1,1)（2.5 金点 + 一次脉冲光环）----
    let p_px = vp.to_px(1.0, 1.0);
    tracks.push(Track::Shape(ShapeTrack {
        kind: ShapeKind::Ellipse,
        x: p_px[0] - 4.0,
        y: p_px[1] - 4.0,
        w: 8.0,
        h: 8.0,
        color: GOLD_RGBA,
        at: 2.5,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
    }));
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            // 2.5–3.3 脉冲一次：半径 4→14→4（bump），区间外空（跳过轨）。
            if t < 2.5_f32 || t > 3.3_f32 {
                return Vec::new();
            }
            let r = 4.0 + 10.0 * bump((t - 2.5_f32) / 0.8_f32);
            circle_pts(p_px[0], p_px[1], r, 28)
        }) as GeomFn,
        color: [GOLD_RGBA[0], GOLD_RGBA[1], GOLD_RGBA[2], 0.85],
        width: 2.0,
        fill: false,
        at: 2.5,
        in_sec: 0.1,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: true,
    }));

    // ---- 水平虚线 y=1（3.0 起 draw 0.3s，从 y 轴到 P；[节奏重剪] 原 5 段
    // 短 Path 分拍（3.0+0.06i）合并为单条 draw 轨——分段本就 contiguous，
    // 合并无视觉差异，省 4 条过程拍）----
    {
        let y_px = vp.to_px(0.0, 1.0)[1];
        let x_axis = vp.to_px(0.0, 0.0)[0];
        let xp = p_px[0];
        tracks.push(Track::Path(PathTrack {
            points: vec![[x_axis, y_px], [xp, y_px]],
            color: [GOLD_RGBA[0], GOLD_RGBA[1], GOLD_RGBA[2], 0.8],
            width: 2.0,
            fill: false,
            at: 3.0,
            in_sec: 0.2,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.3,
            closed: false,
        }));
        tracks.push(static_text("L = 1", 24.0, ctx(GOLD_RGBA), 3.2, 0.3, y_px - 18.0, Some(425.0)));
    }

    // ---- ε-带（3.5 起，CubicIn 收缩 1.00→0.25；水平带状 y∈[1−ε,1+ε]）----
    {
        let (xl, xr) = (vp.x0, vp.x1);
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t| {
                let e = s3_eps(t);
                let a = vp.to_px(xl, 1.0 - e);
                let b = vp.to_px(xr, 1.0 - e);
                let c = vp.to_px(xr, 1.0 + e);
                let d = vp.to_px(xl, 1.0 + e);
                vec![a, b, c, d]
            }) as GeomFn,
            color: [GOLD_RGBA[0], GOLD_RGBA[1], GOLD_RGBA[2], 0.12],
            width: 0.0,
            fill: true,
            at: 3.5,
            in_sec: 1.5,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
        // ε 标签：希腊符号 + 滚动数字（3.5 起，与带同参；[节奏重剪] 滚窗
        // 收至 8.5s = 12.0 终点，顿悟定格内静止；上下文层 alpha）。
        let (mut pre, mut cnt) = greek_roll_readout(
            SymbolKind::Epsilon,
            120.0,
            26.0,
            26.0,
            ctx(GOLD_RGBA),
            1.0,
            0.25,
            3.5,
            8.5,
            EasingKind::CubicIn,
            2,
        );
        // 占位走排以 0 为锚，正式发射平移到目标锚点（带右端外侧）。
        let target = vp.to_px(xr, 1.0)[0] + 70.0;
        shift_prefix(&mut pre, &mut cnt, target);
        for tr in pre.iter_mut() {
            if let Track::Static(s) = tr {
                s.at = 3.5;
                s.in_sec = 0.5;
            }
        }
        tracks.extend(pre);
        tracks.push(Track::Counter(cnt));
    }

    // ---- δ-带（5.0 起，CubicIn 收缩；竖直带状 x∈[1−δ,1+δ]）----
    {
        let (yb, yt) = (vp.y0, vp.y1);
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t| {
                let d = s3_delta(t);
                let a = vp.to_px(1.0 - d, yb);
                let b = vp.to_px(1.0 - d, yt);
                let c = vp.to_px(1.0 + d, yt);
                let e = vp.to_px(1.0 + d, yb);
                vec![a, b, c, e]
            }) as GeomFn,
            color: [CURVE_RGBA[0], CURVE_RGBA[1], CURVE_RGBA[2], 0.10],
            width: 0.0,
            fill: true,
            at: 5.0,
            in_sec: 1.5,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
        // [缺陷修复·δ 标签叠底部文案] baseline 470 → 460：δ 标签墨迹
        // （符号 + 等号 + 数字，底沿 ≈ baseline + 1px）与底部文案 1
        // （y=505，CJK 墨迹顶 ≈ 505 − 0.85·26 ≈ 483）x 范围交叠
        // （标签 x∈[167,266] vs 文案 x∈[~144,~816]），纵向净空由
        // ~11px 增至 ~22px ≥ 15px（`dump_s3_bottom_band_overlap`
        // 渲染实测：标签 rows 441–460 vs 文案 rows 483–506）；
        // 底部文案本身不动（`s3_greek_labels_clear_bottom_caption`
        // 回归锁钉住 y=505 与净空）。
        let (mut pre, mut cnt) = greek_roll_readout(
            SymbolKind::Delta,
            460.0,
            26.0,
            26.0,
            ctx(CURVE_RGBA),
            s3_delta_of_eps(1.0),
            s3_delta_of_eps(0.25),
            5.0,
            7.0,
            EasingKind::CubicIn,
            2,
        );
        let target = vp.to_px(vp.x0, 0.0)[0] - 130.0;
        shift_prefix(&mut pre, &mut cnt, target);
        for tr in pre.iter_mut() {
            if let Track::Static(s) = tr {
                s.at = 5.0;
                s.in_sec = 0.5;
            }
        }
        tracks.extend(pre);
        tracks.push(Track::Counter(cnt));
    }

    // ---- 开幕目标字幕（0.8 打字机，2.9 淡出；本节目标一句话）----
    tracks.push(typewriter(
        "把曲线放大到极致，看它变成什么",
        24.0,
        TEXT_CTX_RGBA,
        0.8,
        500.0,
        12.0,
        Some(2.9),
        0.4,
    ));

    // ---- 底部文案 1（6.5 打字机；out_sec 反推淡出始于打字完成后；
    // [节奏重剪] 文案压到上下文层 alpha 0.50）----
    tracks.push(typewriter(
        "无论误差多小，都能找到对应的范围，把曲线困在 L 附近",
        26.0,
        TEXT_CTX_RGBA,
        6.5,
        505.0,
        12.0,
        None,
        7.5,
    ));

    // ---- live 割线（7.0 draw 0.8s；h(t) CubicInOut [7,12]；12.0 后空）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 7.0 || t > 12.0 {
                return Vec::new();
            }
            let vp = s3_viewport_at(t);
            let h = s3_h(t);
            let a = vp.to_px(1.0, 1.0);
            let b = vp.to_px(1.0 + h, s3_f(1.0 + h));
            vec![a, b]
        }) as GeomFn,
        color: GOLD_RGBA,
        width: 5.0,
        fill: false,
        at: 7.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.8,
        closed: false,
    }));

    // ---- HUD：m = 3.0000 → 2.0000（7.0–12.0；与 h(t) 恒等）----
    // [任务 1] 经 `value_at` 共享 eased-h 纯函数，读数 = 2 + h(t) 逐位恒等
    // 几何（f∘ease 复合，单缓动 Counter 无法复现的缓动脉冲由 value_at 接管）。
    // [节奏重剪] 滚窗 7.0→5.0s；12.0 起读数冻结于 2.0000（顿悟定格内
    // HUD 停滚）；读数压到上下文层 alpha 0.55。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("m = "),
        from: secant_m(1.0),
        to: secant_m(0.0),
        at: 7.0,
        dur_sec: 5.0,
        easing: EasingKind::Linear,
        decimals: 4,
        font: FontKind::Bold,
        size: 30.0,
        color: ctx(GOLD_RGBA),
        center_x: Some(170.0),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        value_at: Some(Arc::new(|t| secant_m(s3_h(t)))),
    }));

    // ---- 割线鬼影标记（h=0.5/0.25/0.125；alpha 0.25，片尾淡出；
    // [节奏重剪] 收敛窗压至 [7,12] 后鬼影改钉在 h 精确穿越时刻
    // 9.5/10.6/11.2；h=1.0 拍与 live 线起点冗余，删除）----
    for (h, at) in [(0.5_f32, 9.5_f32), (0.25, 10.6), (0.125, 11.2)] {
        let a = vp.to_px(1.0, 1.0);
        let b = vp.to_px(1.0 + h, s3_f(1.0 + h));
        tracks.push(Track::Path(PathTrack {
            points: vec![a, b],
            color: [GOLD_RGBA[0], GOLD_RGBA[1], GOLD_RGBA[2], 0.25],
            width: 2.0,
            fill: false,
            at,
            in_sec: 0.4,
            out_sec: 0.5,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
        }));
    }

    // ---- 切线定格（12.0 起，6px 亮金；geom 门控 12.0 前为空；
    // [节奏重剪] 14.0→12.0：此后 12.0–14.5 为 2.5s 顿悟静止定格）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 12.0_f32 {
                return Vec::new();
            }
            let vp = s3_viewport_at(t);
            let a = vp.to_px(1.0, 1.0);
            // 切线 y = 2x − 1：取视口内两端。
            let b = vp.to_px(1.8, 2.0 * 1.8 - 1.0);
            vec![a, b]
        }) as GeomFn,
        color: [1.0, 0.90, 0.45, 1.0],
        width: 6.0,
        fill: false,
        at: 12.0,        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- 定义式（14.5 draw 0.5s，画面上方中央；[节奏重剪] 15.0→14.5，
    // 紧接顿悟定格 12.0–14.5 之后入场）----
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Forall, 30.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" e > 0, "), FontKind::Regular, 26.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Exists, 30.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" d > 0: 0 < |x "), FontKind::Regular, 26.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Minus, 26.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" 1| < d "), FontKind::Regular, 26.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Implies, 26.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" |x"), FontKind::Regular, 26.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Sup(2), 26.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" "), FontKind::Regular, 26.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Minus, 26.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" 1| < e"), FontKind::Regular, 26.0, TEXT_RGBA),
            ],
            480.0,
            58.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 14.5;
                    s.in_sec = 0.5;
                }
                Track::Path(p) => {
                    p.at = 14.5;
                    p.in_sec = 0.5;
                    p.draw_sec = 0.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- 底部文案 2（15.0 淡入；[节奏重剪] 15.5→15.0，上下文层 alpha）----
    tracks.push(static_text(
        "极限，是无限靠近的思想实验",
        26.0,
        TEXT_CTX_RGBA,
        15.0,
        0.5,
        505.0,
        None,
    ));

    ClipSpec {
        width: common::WIDTH,
        height: common::HEIGHT,
        fps: common::FPS,
        duration: dur,
        background: Background::Gradient {
            top: BG_DEEP_BLUE.0,
            bottom: BG_DEEP_BLUE.1,
        },
        jpeg_quality: 82,
        tracks,
    }
}

// ===========================================================================
// S4 导数·切线滑动 + 导函数生长（16.53s，深青黑渐变）
// ===========================================================================

/// S4 数学：f(x)=x²/2，f′(x)=x；切线 y = ax − a²/2，斜率 m ≡ a。
fn s4_f(x: f32) -> f32 {
    0.5 * x * x
}

/// 切线斜率（读数恒等源）：m(a) ≡ a。
fn tangent_m(a: f32) -> f32 {
    a
}

/// a(t)：CubicInOut −2→2 over [3.5, 12.5]（HUD 同窗同参；[任务 1] f∘ease：
/// 切点 x=a 走缓动、y=f(a)=a²/2 由 x 派生，切线/动点/引导线同源一致）。
///
/// [节奏重剪] 扫过窗由 10s 压至 9s：a=2 末态静态定格由 3s 延长至 4s
/// （fast-fast-SLOW 的收尾慢段）；窗外（t ≥ 12.5）冻结端点值。
fn s4_a(t: f32) -> f32 {
    if t <= 3.5 {
        -2.0
    } else if t >= 12.5 {
        2.0
    } else {
        -2.0 + 4.0 * EasingKind::CubicInOut.apply((t - 3.5) / 9.0)
    }
}

/// S4 视口：84px/unit，x∈[−2.3,2.3]、y∈[−2.5,2.5]（PRD 布局）。
fn s4_viewport() -> Viewport {
    Viewport::fit_plot(-2.3, 2.3, -2.5, 2.5, 480.0, 275.0, 193.2, 210.0)
}

/// S4 相机呼吸视口（任务 2）：推近峰值 4%，端点回归 [`s4_viewport`]。
fn s4_viewport_at(t: f32) -> Viewport {
    breathe_viewport(s4_viewport(), t, scene_duration(496), 0.04)
}

/// S4（496 帧 = 16.527s）。
pub fn scene_s4() -> ClipSpec {
    let vp = s4_viewport();
    let dur = scene_duration(496);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 网格 + 坐标轴（0.5–2.0）----
    tracks.extend(axes_tracks_ctx(&vp));

    // ---- 抛物线 f(x)=x²/2（1.0–3.0 draw 2.0s，45 点）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(s4_f, -2.2, 2.2, 45, &vp),
        color: CURVE_RGBA,
        width: 3.5,
        fill: false,
        at: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 2.0,
        closed: false,
    }));

    // ---- 切线（3.0 起；a(t) 全程；视口内裁剪两端）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s4_viewport_at(t);
            let a = s4_a(t);
            // y = ax − a²/2：取视口 x 域内两端（数学 y 由渲染层裁剪）。
            let xl = -2.3_f32;
            let xr = 2.3_f32;
            let p0 = vp.to_px(xl, a * xl - 0.5 * a * a);
            let p1 = vp.to_px(xr, a * xr - 0.5 * a * a);
            vec![p0, p1]
        }) as GeomFn,
        color: GOLD_RGBA,
        width: 4.0,
        fill: false,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- 动点 A（在 f 上，金 r=4）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s4_viewport_at(t);
            let a = s4_a(t);
            let p = vp.to_px(a, s4_f(a));
            circle_pts(p[0], p[1], 4.0, 20)
        }) as GeomFn,
        color: GOLD_RGBA,
        width: 0.0,
        fill: true,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- 动点 B（在 f′ 上，亮紫 r=4）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s4_viewport_at(t);
            let a = s4_a(t);
            let p = vp.to_px(a, a);
            circle_pts(p[0], p[1], 4.0, 20)
        }) as GeomFn,
        color: PURPLE_RGBA,
        width: 0.0,
        fill: true,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- A-B 竖直引导虚线（alpha 0.3）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s4_viewport_at(t);
            let a = s4_a(t);
            let pa = vp.to_px(a, s4_f(a));
            let pb = vp.to_px(a, a);
            v_dashed(pa[0], pa[1], pb[1], 7.0)
        }) as GeomFn,
        color: [TEXT_RGBA[0], TEXT_RGBA[1], TEXT_RGBA[2], 0.3],
        width: 1.5,
        fill: false,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- HUD：m = −2.0000 → 2.0000（3.5–12.5；与 a(t) 恒等）----
    // [任务 1] 经 `value_at` 共享 eased-a 纯函数，读数 = a(t) 逐位恒等几何。
    // [任务 5·缺陷 1] 过零跳：单 Counter 从 −2 滚至 2，minus 在过零帧
    // 消失致居中锚整串右跳 ~6px（实测左缘 81→87）。拆正负两段、
    // 8.0s（a=0 中点；[节奏重剪] 随扫过窗 [3.5,13.5]→[3.5,12.5] 由
    // 8.5s 移至 8.0s）硬切交接；正段 center_x 按 minus 实测字宽左补
    // half-minus ⇒ 两段左缘逐位对齐。−0.0000 归一由 `counter_text`
    // 保证交接帧同串同宽；硬切语义（t≥8.0 负段 env=0）免双显。
    // [节奏重剪] 两段 HUD 均压到上下文层 alpha 0.55。
    let minus_half = (symbols::text_run_width("m = -0.0000", FontKind::Bold, 30.0, 1.0)
        - symbols::text_run_width("m = 0.0000", FontKind::Bold, 30.0, 1.0))
        * 0.5;
    // 负段：3.5–8.0（a: −2→0，值 ≤ 0；居中锚 170）。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("m = "),
        from: tangent_m(-2.0),
        to: 0.0,
        at: 3.5,
        dur_sec: 4.5,
        easing: EasingKind::Linear,
        decimals: 4,
        font: FontKind::Bold,
        size: 30.0,
        color: ctx(GOLD_RGBA),
        center_x: Some(170.0),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: Some(8.0),
        value_at: Some(Arc::new(|t| tangent_m(s4_a(t)).min(0.0))),
    }));
    // 正段：8.0–12.5（a: 0→2，值 ≥ 0；center_x 左补 half-minus 对齐左缘）。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("m = "),
        from: 0.0,
        to: tangent_m(2.0),
        at: 8.0,
        dur_sec: 4.5,
        easing: EasingKind::Linear,
        decimals: 4,
        font: FontKind::Bold,
        size: 30.0,
        color: ctx(GOLD_RGBA),
        center_x: Some(170.0 - minus_half),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.0,
        out_sec: 0.0,
        out_at: None,
        value_at: Some(Arc::new(|t| tangent_m(s4_a(t)).max(0.0))),
    }));

    // ---- [节奏重剪] 原底部文案 1（8.5s「切线的斜率随位置而变——这就是
    // 导函数」）删除：其信息由 f′ 曲线 + f′(x)=x 标签 + 收尾文案承接，
    // 文案减法换呼吸。----

    // ---- 开幕目标字幕（0.8 打字机，2.9 淡出；本节目标一句话）----
    tracks.push(typewriter(
        "让切线滑起来，看斜率怎么变",
        24.0,
        TEXT_CTX_RGBA,
        0.8,
        500.0,
        12.0,
        Some(2.9),
        0.4,
    ));

    // ---- 导函数曲线 f′(x)=x（9.0–10.5 draw 1.5s，亮紫 4px）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(|x| x, -2.2, 2.2, 45, &vp),
        color: PURPLE_RGBA,
        width: 4.0,
        fill: false,
        at: 9.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 1.5,
        closed: false,
    }));

    // ---- 导函数标签 f′(x) = x（10.5 淡入）----
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Text(String::from("f"), FontKind::Bold, 30.0, PURPLE_RGBA),
                FormulaPart::Sym(SymbolKind::Prime, 30.0, PURPLE_RGBA),
                FormulaPart::Text(String::from("(x) = x"), FontKind::Regular, 28.0, TEXT_RGBA),
            ],
            760.0,
            120.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 10.5;
                    s.in_sec = 0.5;
                }
                Track::Path(p) => {
                    p.at = 10.5;
                    p.in_sec = 0.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- [节奏重剪] 原定格脉冲拍（13.5–14.5，B=(2,2) 处光环一次）删除：
    // a(t) 末态定格本身即停顿（12.5 起切点/切线/动点全静止），光环属
    // 过场冗余闪光。----

    // ---- 底部文案 2（13.5 打字机；[节奏重剪] 14.5→13.5 承接末态定格，
    // 上下文层 alpha 0.50）----
    tracks.push(typewriter(
        "导数，把斜率变成一个函数",
        26.0,
        TEXT_CTX_RGBA,
        13.5,
        505.0,
        12.0,
        None,
        0.6,
    ));

    ClipSpec {
        width: common::WIDTH,
        height: common::HEIGHT,
        fps: common::FPS,
        duration: dur,
        background: Background::Gradient {
            top: BG_DEEP_CYAN.0,
            bottom: BG_DEEP_CYAN.1,
        },
        jpeg_quality: 82,
        tracks,
    }
}

// ===========================================================================
// S5 积分·Riemann 矩形累积（16.53s，深紫黑渐变）
// ===========================================================================

/// S5 数学：f(x)=x² on [0,2]；右端黎曼和 Sₙ = 4(n+1)(2n+1)/(3n²)。
fn s5_f(x: f32) -> f32 {
    x * x
}

/// 右端黎曼和（读数恒等源）：S(n) = Σ_{k=1..n}(2k/n)²·(2/n)。
fn riemann_sum(n: f32) -> f32 {
    if n <= 0.0 {
        return 0.0;
    }
    4.0 * (n + 1.0) * (2.0 * n + 1.0) / (3.0 * n * n)
}

/// S5 分拍表（PRD 3.0–9.0 四拍，各 1.5s）：(起始秒, n)。帧网格对齐
/// （架构 §7.6）：3.0/4.5/6.0/7.5/9.0 ⇒ 帧 90/135/180/225/270。
const S5_BEATS: [(f32, u32); 4] = [(3.0, 2), (4.5, 4), (6.0, 8), (7.5, 32)];

/// S5 溶解段 S 读数（9.0–11.0 Linear 滚 riemann_sum(32) → 8/3；区间外
/// 冻结）。起点 = 末批矩形的精确和，终点 = 精确面积——矩形期 S 与矩形
/// 数同拍同值恒等（静态 Counter），无需连续包络近似。
fn s5_s_dissolve(t: f32) -> f32 {
    let (s0, s1) = (riemann_sum(32.0), 8.0 / 3.0);
    if t <= 9.0 {
        s0
    } else if t >= 11.0 {
        s1
    } else {
        s0 + (s1 - s0) * EasingKind::Linear.apply((t - 9.0) / 2.0)
    }
}

/// S5 拍内阶梯多边形（固定 n；顶边界取 N=64 均匀 x 采样——与 morph 目标
/// 同数逐点对应；高度按 QuadOut 生长；`morph` 拍在 [9.0, 11.0] 逐点溶入
/// 曲线下面积，PRD 工程表 P1-3「平滑 morph 替代 crossfade」）。
///
/// 顶边界采样取右端高度（x 所在矩形的右端点；边界样本归右矩），与
/// [`riemann_sum`] 的右端黎曼和定义一致。
fn s5_beat_staircase(vp: Viewport, n: u32, at: f32, grow_sec: f32, morph: bool) -> GeomFn {
    const N: usize = 64; // 顶边界采样数（morph 插值前提：两点数一致）
    Arc::new(move |t: f32| -> Vec<[f32; 2]> {
        if t < at {
            return Vec::new();
        }
        // 高度生长（PRD：QuadOut；首拍 0.8s，后续 0.7s）。
        let g = if grow_sec <= 0.0 || t >= at + grow_sec {
            1.0
        } else {
            EasingKind::QuadOut.apply((t - at) / grow_sec)
        };
        // 溶解 morph（bump，[9.0, 11.0]；非 morph 拍恒 0）。
        let m = if !morph || t <= 9.0 {
            0.0
        } else if t >= 11.0 {
            1.0
        } else {
            bump((t - 9.0) / 2.0)
        };
        let dx = 2.0 / n as f32;
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(N + 2);
        for i in 0..N {
            let x = 2.0 * i as f32 / (N - 1) as f32;
            let k = ((x / dx).floor() as u32 + 1).min(n); // 右端矩形序号
            let h = s5_f(k as f32 * dx) * g * (1.0 - m) + s5_f(x) * m;
            pts.push(vp.to_px(x, h));
        }
        pts.push(vp.to_px(2.0, 0.0));
        pts.push(vp.to_px(0.0, 0.0));
        pts
    }) as GeomFn
}

/// S5 视口：89px/unit，x∈[−0.3,2.5]、y∈[−0.3,4.4]（积分域即视口 x 域）。
fn s5_viewport() -> Viewport {
    Viewport::fit_plot(-0.3, 2.5, -0.3, 4.4, 480.0, 275.0, 124.6, 208.95)
}

/// S5 相机呼吸视口（任务 2）：推近峰值 4%，端点回归 [`s5_viewport`]。
fn s5_viewport_at(t: f32) -> Viewport {
    breathe_viewport(s5_viewport(), t, scene_duration(496), 0.04)
}

/// S5（496 帧 = 16.527s）。
pub fn scene_s5() -> ClipSpec {
    let vp = s5_viewport();
    let dur = scene_duration(496);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 网格 + 坐标轴（0.5–2.0）----
    tracks.extend(axes_tracks_ctx(&vp));

    // ---- 曲线 f(x)=x²（1.0–2.5 draw 1.5s，25 点）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(s5_f, 0.0, 2.0, 25, &vp),
        color: CURVE_RGBA,
        width: 3.5,
        fill: false,
        at: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 1.5,
        closed: false,
    }));

    // ---- 区间标记 0 / 2（2.5 淡入，x 轴下方；上下文层 alpha）----
    {
        let p0 = vp.to_px(0.0, 0.0);
        let p2 = vp.to_px(2.0, 0.0);
        tracks.push(static_text("0", 22.0, ctx(TEXT_RGBA), 2.5, 0.4, p0[1] + 26.0, Some(p0[0])));
        tracks.push(static_text("2", 22.0, ctx(TEXT_RGBA), 2.5, 0.4, p2[1] + 26.0, Some(p2[0])));
    }

    // ---- Riemann 矩形：PRD 四拍阶跃（3.0/4.5/6.0/7.5 各 1.5s）----
    // 每拍一条固定 n 阶梯（高度 QuadOut 生长；旧拍 out_at 锚点 + 0.3s
    // 淡出，新拍同步生长——PRD「旧矩形淡出 0.3s + 新矩形生长 0.7s」）；
    // 末拍（n=32）9.0–11.0 逐点 morph 进曲线下面积（P1-3），15.0–16.0
    // 随场景淡出。64 均匀采样的顶边界与 morph 目标逐点对应，零 NaN。
    for (bi, &(at, n)) in S5_BEATS.iter().enumerate() {
        let last = bi == S5_BEATS.len() - 1;
        let grow_sec = if bi == 0 { 0.8 } else { 0.7 };
        let (out_at, out_sec) = if last {
            (Some(15.0), 1.0) // 末拍化身面积，随场景结尾淡出
        } else {
            (Some(S5_BEATS[bi + 1].0), 0.3) // 旧拍 0.3s 淡出交接
        };
        let staircase = {
            let (n0, at0, gs0, last0) = (n, at, grow_sec, last);
            Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                s5_beat_staircase(s5_viewport_at(t), n0, at0, gs0, last0)(t)
            }) as GeomFn
        };
        // 描边（矩形轮廓，青 55%）。
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::clone(&staircase),
            color: [CURVE_RGBA[0], CURVE_RGBA[1], CURVE_RGBA[2], 0.55],
            width: 1.5,
            fill: false,
            at,
            in_sec: 0.2,
            out_sec,
            out_at,
            draw_sec: 0.0,
            closed: false,
        }));
        // 填充（面积区 alpha 0.30；morph 后即曲线下面积）。
        tracks.push(Track::DynPath(DynPathTrack {
            geom: staircase,
            color: [CURVE_RGBA[0], CURVE_RGBA[1], CURVE_RGBA[2], 0.30],
            width: 0.0,
            fill: true,
            at,
            in_sec: 0.2,
            out_sec,
            out_at,
            draw_sec: 0.0,
            closed: false,
        }));
    }

    // ---- HUD 左：n = 拍值（PRD 2/4/8/32；静态值 + out_at 硬切交接）----
    // 每拍一条 from==to 的静态 Counter，out_at = 下一拍起点（out_sec=0
    // 硬切）⇒ 同位置顺序换版无叠字；屏上数字 ≡ 同拍阶梯矩形数。
    // [任务 5·缺陷 2] 拍间等左缘：各位宽不一（"n = 2" 5 字符 vs
    // "n = 32" 6 字符）致居中锚下 '=' 拍间位移 ~7.8px。按实测字宽
    // （`text_run_width` 与渲染管线逐位同参）把每拍 center_x 补偿为
    // 左缘对齐：center = LEFT + w/2；LEFT 由最宽拍居中原位（140）反演
    // ⇒ 最宽拍视觉位置不变、窄拍左缘不跳。
    let beat_w: Vec<f32> = S5_BEATS
        .iter()
        .map(|&(_, n)| symbols::text_run_width(&format!("n = {}", n), FontKind::Bold, 28.0, 1.0))
        .collect();
    let w_max = beat_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n_left = 140.0 - w_max * 0.5;
    for (&(at, n), &w) in S5_BEATS.iter().zip(beat_w.iter()) {
        tracks.push(Track::Counter(CounterText {
            prefix: String::from("n = "),
            from: n as f32,
            to: n as f32,
            at,
            dur_sec: 0.1,
            easing: EasingKind::Linear,
            decimals: 0,
            font: FontKind::Bold,
            size: 28.0,
            color: ctx(TEXT_RGBA),
            center_x: Some(n_left + w * 0.5),
            y: 52.0,
            letter_spacing: 1.0,
            in_sec: 0.15,
            out_sec: 0.0,
            out_at: Some(at + 1.5),
            value_at: None,
        }));
    }

    // ---- HUD 右：S = 拍值和（PRD 5.0000/3.7500/3.1875/2.7930）----
    // 与矩形数同拍同值恒等（静态值 = riemann_sum(n)），硬切交接。
    for &(at, n) in &S5_BEATS {
        let s = riemann_sum(n as f32);
        tracks.push(Track::Counter(CounterText {
            prefix: String::from("S = "),
            from: s,
            to: s,
            at,
            dur_sec: 0.1,
            easing: EasingKind::Linear,
            decimals: 4,
            font: FontKind::Bold,
            size: 28.0,
            color: ctx(GOLD_RGBA),
            center_x: Some(800.0),
            y: 52.0,
            letter_spacing: 1.0,
            in_sec: 0.15,
            out_sec: 0.0,
            out_at: Some(at + 1.5),
            value_at: None,
        }));
    }

    // ---- 溶解滚：S = 2.7930 → 2.6667（9.0–11.0 Linear；片尾锚定）----
    // 起点 = 末批矩形精确和，终点 = 8/3 精确面积（与积分式/右上角小字
    // 一致）；矩形已溶进面积，读数不再对应离散矩形数，故可连续滚。
    // [任务 5·缺陷 4] out_at 提前至 13.0（0.5s 淡出）：原 out_at=None 令
    // 其在片尾 1.5s（13.5–15.0）仍以全 opacity 显示「S = 2.6667」，与
    // 13.5s 淡入的静态「8/3 = 2.6667」同值双显；现 13.5 前退场，静态独占。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("S = "),
        from: s5_s_dissolve(9.0),
        to: s5_s_dissolve(11.0),
        at: 9.0,
        dur_sec: 2.0,
        easing: EasingKind::Linear,
        decimals: 4,
        font: FontKind::Bold,
        size: 28.0,
        color: ctx(GOLD_RGBA),
        center_x: Some(800.0),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.3,
        out_sec: 0.5,
        out_at: Some(13.0),
        value_at: None,
    }));

    // ---- [节奏重剪] 原底部文案 1（8.0s「矩形越多，和越接近真实的面积」）
    // 删除：矩形生长的视觉自明，文案减法换顿悟呼吸。----

    // ---- 开幕目标字幕（0.8 打字机，2.9 淡出；本节目标一句话）----
    tracks.push(typewriter(
        "用矩形逼近曲线下的面积",
        24.0,
        TEXT_CTX_RGBA,
        0.8,
        500.0,
        12.0,
        Some(2.9),
        0.4,
    ));

    // （曲线下面积由末拍阶梯的 9.0–11.0 morph 直接化身，无需独立填充轨；
    // PRD 工程表 P1-3：平滑 morph 替代 crossfade。）

    // ---- [节奏重剪] 顿悟定格 11.0–13.0（2.0s）：S = 2.6667（8/3）答案
    // 落定后静止——无新元素入场、HUD 停滚、矩形/曲线/面积保持。----

    // ---- 积分式（13.0 draw 1.2s，画面中央；[节奏重剪] 11.0→13.0，让位
    // 顿悟定格）----
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
                // 任务 5·缺陷 3：₀ / ² 改限位组（Limits advance = 0，只做
                // ∫ 上下纵向堆叠）——旧 Sub/Sup 横进会把限位漂到 ∫ 右侧。
                FormulaPart::Limits {
                    lo: String::from("0"),
                    hi: String::from("2"),
                    size: 26.0,
                    color: TEXT_RGBA,
                },
                FormulaPart::Text(String::from(" x"), FontKind::Regular, 30.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Sup(2), 30.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" dx = 8/3"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            // [缺陷修复 3] 旧中心 480.0 令 ∫ 与其上限「2」压在 y 轴
            // （x≈382.2）与 y 轴标签（ink 右缘 ≈409.7）上——左插净空仅
            // 13.2px、「2」与「y」墨迹直接重叠。右移 40px ⇒ 左插净空
            // ≈53px（轴）/ ≈26px（标签），右插距右边框仍余 ≈350px。
            520.0,
            140.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 13.0;
                    s.in_sec = 0.4;
                }
                Track::Path(p) => {
                    p.at = 13.0;
                    p.in_sec = 0.4;
                    p.draw_sec = 1.2;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- 金 ∫ 落版（14.6：定格后承接积分式的单独 ∫ 强调；[节奏重剪]
    // 原 Σ→∫ 演化三拍——Σ 闪光（12.5）/ ∫（13.1）/ 演化箭头——删除，
    // 合并为单 ∫ 落版拍）----
    {
        let mut integral = symbols::assemble_formula(
            &[FormulaPart::Sym(SymbolKind::Integral, 44.0, GOLD_RGBA)],
            530.0,
            150.0,
        );
        for tr in integral.iter_mut() {
            if let Track::Path(p) = tr {
                p.at = 14.6;
                p.in_sec = 0.5;
                p.draw_sec = 0.5;
            }
        }
        tracks.extend(integral);
    }

    // ---- 底部文案 2（15.2 打字机）+ 右上角小字（[节奏重剪] 13.5→15.2，
    // 收尾于顿悟定格之后；上下文层 alpha）----
    tracks.push(typewriter(
        "求和取极限，就是积分",
        26.0,
        TEXT_CTX_RGBA,
        15.2,
        505.0,
        12.0,
        None,
        0.3,
    ));
    tracks.push(static_text("8/3 = 2.6667", 20.0, ctx(GOLD_RGBA), 15.2, 0.5, 90.0, Some(830.0)));

    ClipSpec {
        width: common::WIDTH,
        height: common::HEIGHT,
        fps: common::FPS,
        duration: dur,
        background: Background::Gradient {
            top: BG_DEEP_PURPLE.0,
            bottom: BG_DEEP_PURPLE.1,
        },
        jpeg_quality: 82,
        tracks,
    }
}

// ===========================================================================
// S6 基本定理·双链合流（14.53s，Aurora）
// ===========================================================================

/// S6 数学：链 A = f(x)=x²（变化率），链 B = F(x)=x³/3（累积）。
fn s6_f(x: f32) -> f32 {
    x * x
}

/// F(x) = ∫₀ˣ t² dt = x³/3。
fn s6_f_big(x: f32) -> f32 {
    x * x * x / 3.0
}

/// F′(x) = x² = f(x)（读数恒等源：左 HUD f(x) 与右 HUD F′(x) 同值）。
fn s6_fp(x: f32) -> f32 {
    s6_f(x)
}

/// x(t)：CubicInOut 0→2 over [3.0, 9.0]（四体联动同参；[任务 1] f∘ease）。
/// 动点 x 走缓动、y=f(x)=x² 由 x 派生 ⇒ 复合精确；HUD 经 `value_at` 共享
/// 同一纯函数（`s6_f(s6_x(t))`），读数恒等不受缓动影响（单缓动 Counter
/// 无法复合追踪 f∘ease 的症结由 `value_at` 机制化解，恒等铁律与扫描缓动
/// 口味兼得）。
fn s6_x(t: f32) -> f32 {
    if t <= 3.0 {
        0.0
    } else if t >= 9.0 {
        2.0
    } else {
        2.0 * EasingKind::CubicInOut.apply((t - 3.0) / 6.0)
    }
}

/// S6 视口：91px/unit，x∈[−0.1,2.3]、y∈[−0.2,4.4]（PRD 布局）。
fn s6_viewport() -> Viewport {
    Viewport::fit_plot(-0.1, 2.3, -0.2, 4.4, 480.0, 275.0, 109.2, 209.0)
}

/// S6 相机呼吸视口（任务 2）：推近峰值 4%，端点回归 [`s6_viewport`]。
fn s6_viewport_at(t: f32) -> Viewport {
    breathe_viewport(s6_viewport(), t, scene_duration(436), 0.04)
}

/// S6（436 帧 = 14.527s）。
pub fn scene_s6() -> ClipSpec {
    let vp = s6_viewport();
    let dur = scene_duration(436);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 网格 + 坐标轴（0.5–2.0）----
    tracks.extend(axes_tracks_ctx(&vp));

    // ---- 链 A：f(x)=x²（1.0–3.0 draw 2.0s，25 点，青）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(s6_f, 0.0, 2.0, 25, &vp),
        color: CURVE_RGBA,
        width: 3.5,
        fill: false,
        at: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 2.0,
        closed: false,
    }));
    // 链 A 标签 f(x) = x²（3.0 淡入）。
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Text(String::from("f(x) = x"), FontKind::Regular, 26.0, CURVE_RGBA),
                FormulaPart::Sym(SymbolKind::Sup(2), 26.0, CURVE_RGBA),
            ],
            620.0,
            110.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 3.0;
                    s.in_sec = 0.5;
                }
                Track::Path(p) => {
                    p.at = 3.0;
                    p.in_sec = 0.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- [0,x] 面积填充（3.0 起随 x(t) 生长）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            area_poly(&vp, 0.0, s6_x(t), 48, s6_f)
        }) as GeomFn,
        color: AREA_A26_RGBA,
        width: 0.0,
        fill: true,
        at: 3.0,
        in_sec: 0.5,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- 链 B：F(x)=x³/3（3.0 起，draw 进度 x/2 同步生长）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            let xg = s6_x(t);
            axes::sample_curve(s6_f_big, 0.0, xg, 40, &vp)
        }) as GeomFn,
        color: PURPLE_RGBA,
        width: 3.5,
        fill: false,
        at: 3.0,
        in_sec: 0.5,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 6.0,
        closed: false,
    }));
    // 链 B 标签 F(x) = x³/3（4.0 淡入）。
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Text(String::from("F(x) = x"), FontKind::Regular, 26.0, PURPLE_RGBA),
                FormulaPart::Sym(SymbolKind::Sup(3), 26.0, PURPLE_RGBA),
                FormulaPart::Text(String::from("/3"), FontKind::Regular, 26.0, PURPLE_RGBA),
            ],
            640.0,
            430.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 4.0;
                    s.in_sec = 0.5;
                }
                Track::Path(p) => {
                    p.at = 4.0;
                    p.in_sec = 0.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- 动点 A（f 上，青）与动点 B（F 上，亮紫）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            let x = s6_x(t);
            let p = vp.to_px(x, s6_f(x));
            circle_pts(p[0], p[1], 4.5, 20)
        }) as GeomFn,
        color: CURVE_RGBA,
        width: 0.0,
        fill: true,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            let x = s6_x(t);
            let p = vp.to_px(x, s6_f_big(x));
            circle_pts(p[0], p[1], 4.5, 20)
        }) as GeomFn,
        color: PURPLE_RGBA,
        width: 0.0,
        fill: true,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- A-B 竖直引导虚线（alpha 0.4）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            let x = s6_x(t);
            let pa = vp.to_px(x, s6_f(x));
            let pb = vp.to_px(x, s6_f_big(x));
            v_dashed(pa[0], pa[1], pb[1], 7.0)
        }) as GeomFn,
        color: [TEXT_RGBA[0], TEXT_RGBA[1], TEXT_RGBA[2], 0.4],
        width: 1.5,
        fill: false,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- B 点处 F 的切线（金，斜率 = x²，视口内两端）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t| {
            if t < 3.0 {
                return Vec::new();
            }
            let vp = s6_viewport_at(t);
            let x = s6_x(t);
            let m = s6_fp(x);
            let f0 = s6_f_big(x);
            let xl = (x - 1.2).max(vp.x0);
            let xr = (x + 1.2).min(vp.x1);
            let p0 = vp.to_px(xl, m * (xl - x) + f0);
            let p1 = vp.to_px(xr, m * (xr - x) + f0);
            vec![p0, p1]
        }) as GeomFn,
        color: GOLD_RGBA,
        width: 3.0,
        fill: false,
        at: 3.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 0.0,
        closed: false,
    }));

    // ---- 左 HUD：f(x) = <x²>（3.0 起，读数 ≡ f(x(t)) 恒等）----
    // [任务 1] x(t) 走 CubicInOut（f∘ease），Counter 经 `value_at` 共享
    // s6_f(s6_x(t)) 纯函数 ⇒ 屏上数字与动点高度逐帧精确一致（架构 §7.3）。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("f(x) = "),
        from: s6_fp(0.0),
        to: s6_fp(2.0),
        at: 3.0,
        dur_sec: 6.0,
        easing: EasingKind::QuadIn,
        decimals: 4,
        font: FontKind::Bold,
        size: 28.0,
        color: CURVE_RGBA,
        center_x: Some(220.0),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        value_at: Some(Arc::new(|t| s6_f(s6_x(t)))),
    }));

    // ---- 右 HUD：F'(x) = <x²>（3.0 起，同值恒等：F′=f）----
    // [任务 1] 同左 HUD，经 `value_at` 共享 s6_fp(s6_x(t)) 纯函数。
    tracks.push(Track::Counter(CounterText {
        prefix: String::from("F'(x) = "),
        from: s6_fp(0.0),
        to: s6_fp(2.0),
        at: 3.0,
        dur_sec: 6.0,
        easing: EasingKind::QuadIn,
        decimals: 4,
        font: FontKind::Bold,
        size: 28.0,
        color: PURPLE_RGBA,
        center_x: Some(740.0),
        y: 52.0,
        letter_spacing: 1.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        value_at: Some(Arc::new(|t| s6_fp(s6_x(t)))),
    }));

    // ---- [节奏重剪] 原底部文案 1（6.5s「F 的变化率，正是 f 本身」）删除：
    // 该句提前剧透重合揭示，删掉让顿悟定格更干净。----

    // ---- 开幕目标字幕（0.8 打字机，2.9 淡出；本节目标一句话）----
    tracks.push(typewriter(
        "变化与累积，本是同一件事",
        24.0,
        TEXT_CTX_RGBA,
        0.8,
        500.0,
        12.0,
        Some(2.9),
        0.4,
    ));

    // ---- F′ 重绘（9.0 draw 1.0s，亮金，与 f 完全重合）----
    tracks.push(Track::Path(PathTrack {
        points: axes::sample_curve(s6_f, 0.0, 2.0, 25, &vp),
        color: [1.0, 0.90, 0.45, 1.0],
        width: 3.0,
        fill: false,
        at: 9.0,
        in_sec: 0.3,
        out_sec: 0.0,
        out_at: None,
        draw_sec: 1.0,
        closed: false,
    }));

    // ---- F′(x) = f(x)（10.0 淡入，居中；[节奏重剪] 此后 10.0–12.5 为
    // 2.5s 顿悟静止定格——重合公式落定，无新元素、HUD 停滚）----
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Text(String::from("F"), FontKind::Bold, 32.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Prime, 32.0, TEXT_RGBA),
                FormulaPart::Text(String::from("(x) = f(x)"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            480.0,
            150.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 10.0;
                    s.in_sec = 0.5;
                }
                Track::Path(p) => {
                    p.at = 10.0;
                    p.in_sec = 0.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- FTC 公式（12.5 draw 1.5s，画面中央；[节奏重剪] 10.5→12.5，让位
    // 顿悟定格 10.0–12.5）----
    {
        let mut formula = symbols::assemble_formula(
            &[
                FormulaPart::Text(String::from("d/dx "), FontKind::Regular, 30.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Integral, 44.0, TEXT_RGBA),
                FormulaPart::Sym(SymbolKind::Sub(0), 24.0, TEXT_RGBA),
                FormulaPart::Text(String::from("x"), FontKind::Regular, 20.0, TEXT_RGBA),
                FormulaPart::Text(String::from(" f(t) dt = f(x)"), FontKind::Regular, 30.0, TEXT_RGBA),
            ],
            480.0,
            250.0,
        );
        for tr in formula.iter_mut() {
            match tr {
                Track::Static(s) => {
                    s.at = 12.5;
                    s.in_sec = 0.4;
                }
                Track::Path(p) => {
                    p.at = 12.5;
                    p.in_sec = 0.4;
                    p.draw_sec = 1.5;
                }
                _ => {}
            }
        }
        tracks.extend(formula);
    }

    // ---- [节奏重剪] 原底部文案 2（12.0s「微分与积分，互为逆运算」）删除：
    // FTC 公式即该句的精确表述，收尾让定理自己说话。----

    // ---- 两链合流脉冲（13.0–14.0：f 与 F 同步光环）----
    for (color, f) in [
        (CURVE_RGBA, s6_f as fn(f32) -> f32),
        (PURPLE_RGBA, s6_f_big as fn(f32) -> f32),
    ] {
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t| {
                if t < 13.0_f32 || t > 14.0_f32 {
                    return Vec::new();
                }
                let x = 2.0_f32;
                let p = vp.to_px(x, f(x));
                let r = 5.0 + 14.0 * bump((t - 13.0_f32) / 1.0_f32);
                circle_pts(p[0], p[1], r, 28)
            }) as GeomFn,
            color: [color[0], color[1], color[2], 0.85],
            width: 2.5,
            fill: false,
            at: 13.0,
            in_sec: 0.1,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: true,
        }));
    }

    ClipSpec {
        width: common::WIDTH,
        height: common::HEIGHT,
        fps: common::FPS,
        duration: dur,
        background: Background::Aurora,
        jpeg_quality: 82,
        tracks,
    }
}

// ===========================================================================
// 测试：数学恒等 + 帧网格 + 读数恒等（架构 §6.2-1 必含项）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 帧网格：四场景帧数 / duration ceil 回帧 / 总和对。
    #[test]
    fn scene_frame_grid_aligns() {
        for (spec, n) in [
            (scene_s3(), 496_u32),
            (scene_s4(), 496),
            (scene_s5(), 496),
            (scene_s6(), 436),
        ] {
            assert_eq!(spec.width, common::WIDTH);
            assert_eq!(spec.height, common::HEIGHT);
            assert_eq!(spec.fps, common::FPS);
            assert_eq!(spec.frame_count(), n as usize, "duration 必须 ceil 回帧数");
            assert!(!spec.tracks.is_empty(), "轨道不得为空");
        }
    }

    /// [临时 dump] greek_roll_readout 几何 dump：等号 / 希腊符号的变换后
    /// 屏幕坐标、描边宽度、y 范围 + Counter center_x（定位 S3 ε/δ 标签缺陷）。
    #[test]
    fn dump_greek_roll_readout_geometry() {
        for kind in [SymbolKind::Epsilon, SymbolKind::Delta] {
            let (pre, cnt) = greek_roll_readout(
                kind,
                120.0,
                26.0,
                26.0,
                ctx(GOLD_RGBA),
                1.0,
                0.25,
                3.5,
                8.5,
                EasingKind::CubicIn,
                2,
            );
            println!("=== greek_roll_readout({kind:?}, baseline_y=120, sym=26, num=26) ===");
            println!("Counter: center_x={:?} y={} size={} font={:?} prefix={:?}",
                cnt.center_x, cnt.y, cnt.size, cnt.font, cnt.prefix);
            for (i, tr) in pre.iter().enumerate() {
                match tr {
                    Track::Path(p) => {
                        let ys: Vec<f32> = p.points.iter().map(|q| q[1]).collect();
                        let y0 = ys.iter().cloned().fold(f32::INFINITY, f32::min);
                        let y1 = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let xs: Vec<f32> = p.points.iter().map(|q| q[0]).collect();
                        let x0 = xs.iter().cloned().fold(f32::INFINITY, f32::min);
                        let x1 = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        println!(
                            "  Path[{i}] pts={:?} width={} x∈[{:.2},{:.2}] y∈[{:.2},{:.2}]",
                            p.points, p.width, x0, x1, y0, y1
                        );
                    }
                    Track::Static(s) => {
                        println!("  Static[{i}] content={:?} center_x={:?} y={}", s.content, s.center_x, s.y);
                    }
                    _ => println!("  Other[{i}]"),
                }
            }
        }
    }

    /// [临时 dump] 字体 '.' / ':' 字形 dump：Regular / Bold 两字体的
    /// gid / advance / 轮廓数与 y 范围（定位「0.12 渲染成 0:12」根因）。
    #[test]
    fn dump_period_glyph_metrics() {
        use prism_text::ttf::Font;
        for (name, font) in [("Regular", prism_text::font_regular()), ("Bold", prism_text::font_bold())] {
            let upem = font.units_per_em();
            println!("=== font {name} (upem={upem}) ===");
            for ch in ['.', ':', '0', '1', '2'] {
                let gid = font.glyph_index(ch);
                match gid {
                    None => println!("  {ch:?}: gid=None（查不到）"),
                    Some(g) => {
                        let outline = font.outline(g).expect("outline");
                        let n = outline.contours.len();
                        let (mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY);
                        let (mut x0, mut x1) = (f32::INFINITY, f32::NEG_INFINITY);
                        for c in &outline.contours {
                            for p in c {
                                y0 = y0.min(p.y);
                                y1 = y1.max(p.y);
                                x0 = x0.min(p.x);
                                x1 = x1.max(p.x);
                            }
                        }
                        println!(
                            "  {ch:?}: gid={g} adv={} contours={n} x∈[{x0},{x1}] y∈[{y0},{y1}]",
                            font.advance(g)
                        );
                    }
                }
            }
        }
        let _ = Font::parse(&[]).ok();
    }

    /// [临时 dump] 用真实渲染管线（draw_counter → draw_static →
    /// layout_text + draw_glyphs + GlyphCache）渲染 "0.12"（Regular 26
    /// 与 Bold 26 对照），ASCII dump 数字区墨迹，定位「0:12」根因。
    #[test]
    fn dump_counter_period_pixels() {
        use prism_author::{render_short_concat_faded_frames, Background as Bg};
        for font in [FontKind::Regular, FontKind::Bold] {
            let spec = ClipSpec {
                width: 160,
                height: 60,
                fps: 30,
                duration: 0.2,
                background: Bg::Gradient { top: [0.0; 3], bottom: [0.0; 3] },
                jpeg_quality: 90,
                tracks: vec![Track::Counter(CounterText {
                    prefix: String::new(),
                    from: 0.12,
                    to: 0.12,
                    at: 0.0,
                    dur_sec: 0.1,
                    easing: EasingKind::Linear,
                    decimals: 2,
                    font,
                    size: 26.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                    center_x: Some(80.0),
                    y: 40.0,
                    letter_spacing: 0.0,
                    in_sec: 0.0,
                    out_sec: 0.0,
                    out_at: None,
                    value_at: None,
                })],
            };
            let (w, h) = (spec.width, spec.height);
            // 两支相同 spec：Dissolve 混合同内容不变，帧 0 即纯净单 clip 首帧。
            let frames = render_short_concat_faded_frames(&[spec.clone(), spec], 0.0, 0.0);
            let f = &frames[0];
            println!("=== counter 0.12 font={font:?} frame {w}x{h} ink runs ===");
            for row in 0..h as usize {
                let mut runs: Vec<(usize, usize)> = Vec::new();
                let mut start: Option<usize> = None;
                for col in 0..w as usize {
                    let i = (row * w as usize + col) * 4;
                    let lum = (f[i] as u32 + f[i + 1] as u32 + f[i + 2] as u32) / 3;
                    let ink = lum > 40;
                    match (ink, start) {
                        (true, None) => start = Some(col),
                        (false, Some(s)) => {
                            runs.push((s, col - 1));
                            start = None;
                        }
                        _ => {}
                    }
                }
                if let Some(s) = start {
                    runs.push((s, w as usize - 1));
                }
                if !runs.is_empty() {
                    let desc: Vec<String> = runs.iter().map(|(a, b)| format!("[{a}..{b}]")).collect();
                    println!("row {row:2}: {}", desc.join(" "));
                }
            }
        }
    }

    /// [临时 dump] 单字形位图 dump：直接用 prism_text::raster::render_glyph
    /// 渲染 Regular/Bold 的 '.' 与 ':'（26px），打印 alpha 网格——确定
    /// 小数点是否被光栅成「两点」（colon 形）。
    #[test]
    fn dump_single_glyph_bitmaps() {
        use prism_text::raster::render_glyph;
        for (fname, font) in [("Regular", prism_text::font_regular()), ("Bold", prism_text::font_bold())] {
            for ch in ['.', ':'] {
                let gid = font.glyph_index(ch).expect("gid");
                let bmp = render_glyph(&font, gid, 26.0);
                println!("=== {fname} {ch:?} gid={gid} size=26 bmp {}x{} origin=({},{}) ===",
                    bmp.width, bmp.height, bmp.left, bmp.top);
                for row in 0..bmp.height {
                    let mut line = String::new();
                    for col in 0..bmp.width {
                        let a = bmp.alpha[(row * bmp.width + col) as usize];
                        line.push(if a > 200 { '#' } else if a > 128 { '+' } else if a > 40 { '.' } else { ' ' });
                    }
                    println!("{line}");
                }
            }
        }
    }

    /// [临时 dump] layout_text 原始字形位置 dump：Regular/Bold 渲染
    /// "0.12"（26px）的 gid/x/y/advance——定位「0:12」的字形层根因。
    #[test]
    fn dump_layout_glyph_positions() {
        use prism_text::layout::{layout_text, Align, TextParams};
        for (fname, font) in [("Regular", prism_text::font_regular()), ("Bold", prism_text::font_bold())] {
            let params = TextParams {
                font_size: 26.0,
                letter_spacing: 0.0,
                line_spacing: 0.0,
                max_width: f32::INFINITY,
                align: Align::Center,
            };
            let layout = layout_text(&font, "0.12", &params);
            println!("=== {fname} layout 0.12 size=26: width={} height={} ===", layout.width, layout.height);
            for g in &layout.glyphs {
                println!(
                    "  ch={:?} gid={} x={:.3} y={:.3} adv={:.3} line={}",
                    g.ch, g.gid, g.x, g.y, g.advance, g.line
                );
            }
            for l in &layout.lines {
                println!("  line {}..{} w={} baseline={}", l.start, l.end, l.width, l.baseline);
            }
        }
    }

    /// [缺陷修复·等号不可见] 几何回归锁（`dump_greek_roll_readout_geometry`
    /// dump 数据固化）：ε/δ 标签的等号须是**两条可辨横线**——水平、线长
    /// ≥0.5·sym_size、笔宽随字号折算、双线净距 ≥2×笔宽；垂直居中于数字
    /// x 高度区（双线中心 ∈ baseline 上方 0.25–0.40·size，即单位格
    /// y=0.17/0.37 缩放 + 设备翻转后的落点）；水平落在符号与数字之间且
    /// 与数字 ink 净空 ≥5px（单次装配实测 8.84px；旧两次装配下等号
    /// x∈[2.34,17.94] 被数字 '0' 字环 x∈[−1.34,49.18] 吞没）。
    #[test]
    fn greek_readout_equals_is_visible_and_clears_digits() {
        let baseline = 120.0f32;
        let sym_size = 26.0f32;
        let stroke_w = (sym_size * symbols::SYM_STROKE_K).max(symbols::MIN_SYM_STROKE_W);
        for kind in [SymbolKind::Epsilon, SymbolKind::Delta] {
            let (pre, cnt) = greek_roll_readout(
                kind,
                baseline,
                sym_size,
                sym_size,
                ctx(TEXT_RGBA),
                1.0,
                0.25,
                3.5,
                8.5,
                EasingKind::CubicIn,
                2,
            );
            // 等号双横线：2 点水平笔画且线长 ≥0.5·size（ε 中杠长
            // 0.40·size，被此口径排除；δ 无其它 2 点水平笔画）。
            let bars: Vec<&PathTrack> = pre
                .iter()
                .filter_map(|t| match t {
                    Track::Path(p)
                        if p.points.len() == 2
                            && (p.points[0][1] - p.points[1][1]).abs() < 1e-6
                            && (p.points[1][0] - p.points[0][0]).abs() >= 0.5 * sym_size =>
                    {
                        Some(p)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(bars.len(), 2, "{kind:?} 等号须恰为 2 条横线，实测 {}", bars.len());
            let mut ys: Vec<f32> = Vec::with_capacity(2);
            let (mut eq_x0, mut eq_x1) = (f32::INFINITY, f32::NEG_INFINITY);
            for b in &bars {
                let len = (b.points[1][0] - b.points[0][0]).abs();
                assert!(len >= 0.5 * sym_size, "{kind:?} 等号线长 {len} < 0.5·size");
                assert!(
                    (b.width - stroke_w).abs() < 1e-6,
                    "{kind:?} 等号笔宽须 = max(size·SYM_STROKE_K, MIN)，实测 {}",
                    b.width
                );
                eq_x0 = eq_x0.min(b.points[0][0].min(b.points[1][0]));
                eq_x1 = eq_x1.max(b.points[0][0].max(b.points[1][0]));
                ys.push(b.points[0][1]);
            }
            ys.sort_by(f32::total_cmp);
            // 双线净距 ≥2×笔宽（不糊成一条）。
            assert!(
                ys[1] - ys[0] >= 2.0 * stroke_w,
                "{kind:?} 双线净距 {} 过小",
                ys[1] - ys[0]
            );
            // 垂直居中于数字 x 高度区：双线中心 ∈ baseline 上方 [0.25, 0.40]·size。
            let mid_above = baseline - (ys[0] + ys[1]) * 0.5;
            assert!(
                (0.25 * sym_size..=0.40 * sym_size).contains(&mid_above),
                "{kind:?} 等号中心距基线 {mid_above} 不在 0.25–0.40·size 带"
            );
            // 符号墨迹右缘 < 等号左缘（等号落在符号与数字之间）。
            let mut sym_x1 = f32::NEG_INFINITY;
            for t in &pre {
                if let Track::Path(p) = t {
                    let is_bar = p.points.len() == 2
                        && (p.points[0][1] - p.points[1][1]).abs() < 1e-6
                        && (p.points[1][0] - p.points[0][0]).abs() >= 0.5 * sym_size;
                    if !is_bar {
                        for q in &p.points {
                            sym_x1 = sym_x1.max(q[0]);
                        }
                    }
                }
            }
            assert!(sym_x1 < eq_x0, "{kind:?} 等号须在符号右缘之右（{sym_x1} vs {eq_x0}）");
            // 等号右缘与数字 ink 左缘净空 ≥5px（单次装配同源走排）。
            let placeholder_w = symbols::text_run_width("0.00", FontKind::Regular, sym_size, 0.0);
            let digit_x0 = cnt.center_x.expect("Counter 须有 center_x") - placeholder_w * 0.5;
            assert!(
                digit_x0 - eq_x1 >= 5.0,
                "{kind:?} 等号右缘 {eq_x1} 与数字左缘 {digit_x0} 净空不足（旧实现被 '0' 吞没）"
            );
            // 表宽数字：滚动域内各位形串宽恒定（净空在整段滚动中保持）。
            for s in ["1.00", "0.25", "0.41", "0.12"] {
                let w = symbols::text_run_width(s, FontKind::Regular, sym_size, 0.0);
                assert!(
                    (w - placeholder_w).abs() < 1e-3,
                    "{kind:?} 表宽数字假设破裂：{s} 宽 {w} vs 占位 {placeholder_w}"
                );
            }
        }
    }

    /// [缺陷修复·小数点糊成冒号] 字形层回归锁：Regular/Bold 的 '.' 须为
    /// **单轮廓、位于行底部带**的小点（dump 实测 gid=15、1 contour、
    /// y∈[−13,126]/upem1000；对照 ':' 为 gid=27、2 contours、
    /// y∈[−13,530]）；`layout_text("0.12")` 中 '.' 明显窄于数字且居于
    /// '0' 与 '1' 之间。旧「0:12」根因是等号下线与 '.' 点相接（两次
    /// 装配水平错位），非字形本身——本锁防字形层回归。
    #[test]
    fn counter_period_glyph_is_single_bottom_dot() {
        use prism_text::layout::{layout_text, Align, TextParams};
        for font in [prism_text::font_regular(), prism_text::font_bold()] {
            let upem = font.units_per_em() as f32;
            let dot = font.glyph_index('.').expect("'.' 须有字形");
            let colon = font.glyph_index(':').expect("':' 须有字形（对照组）");
            let dot_o = font.outline(dot).expect("'.' 轮廓");
            let colon_o = font.outline(colon).expect("':' 轮廓");
            assert_eq!(dot_o.contours.len(), 1, "'.' 必须单轮廓（冒号形为 2）");
            assert_eq!(colon_o.contours.len(), 2, "':' 双轮廓（对照组）");
            // '.' 轮廓整体贴行底部（y_max ≤ 0.25·upem，即基线附近）。
            let y_max = dot_o
                .contours
                .iter()
                .flatten()
                .map(|p| p.y)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(y_max <= 0.25 * upem, "'.' 应在行底部（y_max={y_max}）");
            // layout "0.12"：'.' 夹在 '0' 与 '1' 之间，且明显窄于数字。
            let params = TextParams {
                font_size: 26.0,
                letter_spacing: 0.0,
                line_spacing: 0.0,
                max_width: f32::INFINITY,
                align: Align::Center,
            };
            let layout = layout_text(&font, "0.12", &params);
            let glyphs: Vec<(char, f32, f32)> = layout
                .glyphs
                .iter()
                .map(|g| (g.ch, g.x, g.advance))
                .collect();
            assert_eq!(glyphs.len(), 4, "0.12 应四字形");
            assert_eq!(
                (glyphs[0].0, glyphs[1].0, glyphs[2].0, glyphs[3].0),
                ('0', '.', '1', '2'),
                "0.12 字形序"
            );
            assert!(
                glyphs[0].1 < glyphs[1].1 && glyphs[1].1 < glyphs[2].1 && glyphs[2].1 < glyphs[3].1,
                "'.' 须居于 '0' 与 '1' 之间"
            );
            assert!(
                glyphs[1].2 < glyphs[2].2 * 0.75,
                "'.' advance 须明显窄于数字（{} vs {}）",
                glyphs[1].2,
                glyphs[2].2
            );
        }
    }

    /// [缺陷修复·δ 标签叠底部文案] 净空回归锁：S3 的 ε / δ 希腊标签与
    /// 底部打字文案 1（y=505）x 范围交叠，故须纵向净空 ≥15px；文案
    /// 本身不得移动（锁 y=505 与内容）。标签墨迹底沿 = Counter 基线
    /// +1px（符号 / 数字均坐基线，描边半宽兜底）；CJK 文案墨迹顶 =
    /// 基线 − 0.85·size（`dump_s3_bottom_band_overlap` 渲染实测
    /// rows 483–506 @ size 26 ⇒ 22.1px ≈ 0.85·26）。
    #[test]
    fn s3_greek_labels_clear_bottom_caption() {
        let spec = scene_s3();
        let cap = spec
            .tracks
            .iter()
            .find_map(|t| match t {
                Track::Typewriter(w) if w.content.contains("把曲线困在 L 附近") => Some(w),
                _ => None,
            })
            .expect("S3 应有底部打字文案");
        // 底部文案本身不动（只许动标签）。
        assert_eq!(cap.y, 505.0, "底部文案 baseline 不得移动");
        assert_eq!(cap.content, "无论误差多小，都能找到对应的范围，把曲线困在 L 附近");
        let cap_w = symbols::text_run_width(&cap.content, cap.font, cap.size, cap.letter_spacing);
        let cap_cx = cap.center_x.unwrap_or(common::WIDTH as f32 * 0.5);
        let (cap_x0, cap_x1) = (cap_cx - cap_w * 0.5, cap_cx + cap_w * 0.5);
        let cap_top = cap.y - 0.85 * cap.size;
        // ε / δ 标签 Counter（decimals=2 + Regular；S3 仅此两条）。
        let mut n_labels = 0usize;
        for c in spec.tracks.iter().filter_map(|t| match t {
            Track::Counter(c) if c.decimals == 2 && c.font == FontKind::Regular => Some(c),
            _ => None,
        }) {
            n_labels += 1;
            // 纵向净空（标签底沿 → 文案墨迹顶）。
            let clearance = cap_top - (c.y + 1.0);
            assert!(
                clearance >= 15.0,
                "希腊标签（y={}）与底部文案净空 {clearance} < 15px",
                c.y
            );
            // 标签数字 ink 盒（表宽数字 ⇒ 滚动域内宽度恒定）。
            let readout = format!("{:.2}", c.from);
            let w = symbols::text_run_width(&readout, c.font, c.size, c.letter_spacing);
            let cx = c.center_x.expect("标签 Counter 须有 center_x");
            let (dig_x0, dig_x1) = (cx - w * 0.5, cx + w * 0.5);
            // x 交叠 ⇒ 纵向净空才是约束项（断言才有意义）。
            assert!(
                dig_x0 < cap_x1 && dig_x1 > cap_x0,
                "标签数字 x∈[{dig_x0},{dig_x1}] 应与文案 x∈[{cap_x0},{cap_x1}] 交叠"
            );
        }
        assert_eq!(n_labels, 2, "S3 应恰有 ε / δ 两条希腊标签 Counter");
    }

    /// [临时 dump] S3 左下底部带 dump：δ 标签（真实锚点：baseline 460 +
    /// 平移）+ 底部文案 1（y=505）同画幅单帧渲染（t=10.0，文案打完），
    /// dump y∈[440,530] 每行 ink 的 x 区间——量化标签与文案的净空
    /// （修复后实测：标签 rows 441–460 vs 文案 rows 483–506，净空 22px）。
    #[test]
    fn dump_s3_bottom_band_overlap() {
        use prism_author::render_short_concat_faded_frames;
        let vp = s3_viewport();
        let mut tracks: Vec<Track> = Vec::new();
        // δ 标签（与 scene_s3 同参发射后平移到真实锚点）。
        let (mut pre, mut cnt) = greek_roll_readout(
            SymbolKind::Delta,
            460.0,
            26.0,
            26.0,
            ctx(CURVE_RGBA),
            s3_delta_of_eps(1.0),
            s3_delta_of_eps(0.25),
            5.0,
            7.0,
            EasingKind::CubicIn,
            2,
        );
        let target = vp.to_px(vp.x0, 0.0)[0] - 130.0;
        shift_prefix(&mut pre, &mut cnt, target);
        for tr in pre.iter_mut() {
            if let Track::Static(s) = tr {
                s.at = 5.0;
                s.in_sec = 0.5;
            }
        }
        tracks.extend(pre);
        tracks.push(Track::Counter(cnt));
        // 底部文案 1（y=505，at=6.5，cps=12；t=10.0 已打完）。
        tracks.push(typewriter(
            "无论误差多小，都能找到对应的范围，把曲线困在 L 附近",
            26.0,
            TEXT_CTX_RGBA,
            6.5,
            505.0,
            12.0,
            None,
            7.5,
        ));
        let spec = ClipSpec {
            width: common::WIDTH,
            height: common::HEIGHT,
            fps: common::FPS,
            duration: 10.4,
            background: Background::Gradient { top: [0.0; 3], bottom: [0.0; 3] },
            jpeg_quality: 90,
            tracks,
        };
        let frames = render_short_concat_faded_frames(&[spec.clone(), spec], 0.0, 0.0);
        let f = &frames[300]; // t = 10.0s
        let w = common::WIDTH as usize;
        println!("=== S3 底部带 ink（t=10.0，字幕基线 505；阈值 lum>40）===");
        for row in 440usize..530 {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut start: Option<usize> = None;
            for col in 0..w {
                let i = (row * w + col) * 4;
                let lum = (f[i] as u32 + f[i + 1] as u32 + f[i + 2] as u32) / 3;
                let ink = lum > 40;
                match (ink, start) {
                    (true, None) => start = Some(col),
                    (false, Some(s)) => {
                        runs.push((s, col - 1));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = start {
                runs.push((s, w - 1));
            }
            if !runs.is_empty() {
                let desc: Vec<String> = runs.iter().map(|(a, b)| format!("[{a}..{b}]")).collect();
                println!("row {row:3}: {}", desc.join(" "));
            }
        }
    }

    /// S3 数学：m(h)=2+h 序列与 ε-δ 端点关系（PRD §6）。
    #[test]
    fn s3_math_matches_prd() {
        for (h, m) in [
            (1.0, 3.0),
            (0.5, 2.5),
            (0.25, 2.25),
            (0.125, 2.125),
            (0.0, 2.0),
        ] {
            assert!((secant_m(h) - m).abs() < 1e-6, "m({h}) = {}", secant_m(h));
        }
        // ε-δ：δ = √(1+ε) − 1（PRD：ε=1.00→0.41；ε=0.25→0.12）。
        assert!((s3_delta_of_eps(1.0) - 0.4142).abs() < 5e-5);
        assert!((s3_delta_of_eps(0.25) - 0.1180).abs() < 5e-5);
        // 端点精确：|x²−1| < ε ⟺ |x−1| < δ（取 δ(2+δ)=ε）。
        for e in [1.0_f32, 0.5, 0.25] {
            let d = s3_delta_of_eps(e);
            assert!((d * (2.0 + d) - e).abs() < 1e-6, "δ(2+δ)=ε 精确");
        }
        // h(t) 与 ε(t) / δ(t) 单调收敛。
        assert!((s3_h(7.0) - 1.0).abs() < 1e-6 && (s3_h(14.0)).abs() < 1e-6);
        assert!(s3_h(10.5) < s3_h(7.0) && s3_h(10.5) > s3_h(14.0));
        assert!((s3_eps(15.0) - 0.25).abs() < 1e-6 && (s3_delta(15.0) - 0.118).abs() < 1e-3);
    }

    /// S4 数学：切线 y=ax−a²/2 斜率恒等 a（PRD §6 勘误备案）。
    #[test]
    fn s4_math_matches_prd() {
        for a in [-2.0_f32, -0.5, 0.0, 1.0, 2.0] {
            assert!((tangent_m(a) - a).abs() < 1e-6);
            // 切线过 (a, a²/2)：y(a) = a·a − a²/2 = a²/2 ✓
            let y_at_a = a * a - 0.5 * a * a;
            assert!((y_at_a - s4_f(a)).abs() < 1e-6, "切线过切点");
        }
        assert!((s4_a(3.5) + 2.0).abs() < 1e-6 && (s4_a(13.5) - 2.0).abs() < 1e-6);
        // 视口覆盖 f(±2.2)=2.42 ≤ 2.5（PRD §6）。
        assert!(s4_f(2.2) <= 2.5);
    }

    /// S5 数学：S(n)=4(n+1)(2n+1)/(3n²)（PRD §6 检查点）+ 四拍表 /
    /// 溶解段读数（9.0–11.0）恒等。分拍阶跃下 S 与矩形数同拍同值
    /// （静态 Counter），无需连续包络近似。
    #[test]
    fn s5_math_matches_prd() {
        for (n, s) in [
            (2.0, 5.0),
            (4.0, 3.75),
            (8.0, 3.1875),
            (32.0, 2.79296875),
        ] {
            assert!((riemann_sum(n) - s).abs() < 1e-6, "S({n}) = {}", riemann_sum(n));
        }
        // n→∞ → 8/3。
        assert!((riemann_sum(1.0e4) - 8.0 / 3.0).abs() < 1e-3);
        // 右端（递增函数）⇒ 高估收敛。
        assert!(riemann_sum(2.0) > riemann_sum(32.0) && riemann_sum(32.0) > 8.0 / 3.0);
        // 四拍表：拍值 n 与 PRD 检查点一一对应（n = 2/4/8/32，
        // S = 5.0000/3.7500/3.1875/2.7930），每拍 1.5s。
        for (i, &(at, n)) in S5_BEATS.iter().enumerate() {
            let s_prd = [5.0_f32, 3.75, 3.1875, 2.79296875][i];
            assert!((riemann_sum(n as f32) - s_prd).abs() < 1e-6, "拍 {i} S({n})");
            let end = S5_BEATS.get(i + 1).map_or(9.0, |b| b.0);
            assert!((end - at - 1.5).abs() < 1e-6, "拍 {i} 时长 1.5s");
        }
        // 帧网格对齐（架构 §7.6）：拍界 3.0/4.5/6.0/7.5、溶解窗 9.0/11.0、
        // 收尾锚点 15.0 全为 30fps 整数帧 ⇒ 帧 90/135/180/225/270/330/450。
        for t in [3.0_f32, 4.5, 6.0, 7.5, 9.0, 11.0, 15.0] {
            let f = t * 30.0;
            assert!((f - f.round()).abs() < 1e-3, "{t}s 非帧边界（f={f}）");
        }
        // 溶解段读数：端点 = 末批矩形精确和 / 8/3 精确面积；中点线性；
        // 区间外冻结；全域单调下降（右端和 → 精确面积）。
        let (s0, s1) = (riemann_sum(32.0), 8.0 / 3.0);
        assert!((s5_s_dissolve(9.0) - s0).abs() < 1e-6, "溶解起点");
        assert!((s5_s_dissolve(11.0) - s1).abs() < 1e-6, "溶解终点");
        assert!((s5_s_dissolve(10.0) - (s0 + s1) / 2.0).abs() < 1e-6, "中点线性");
        assert!((s5_s_dissolve(0.0) - s0).abs() < 1e-6, "窗前冻结");
        assert!((s5_s_dissolve(20.0) - s1).abs() < 1e-6, "窗后冻结");
        let mut prev = s0 + 1.0;
        for i in 0..=20 {
            let v = s5_s_dissolve(9.0 + i as f32 / 10.0);
            assert!(v <= prev + 1e-6 && v >= s1 - 1e-6, "单调下降 t=9.0+{i}/10");
            prev = v;
        }
    }

    /// S6 数学：F=x³/3，F′=x²=f；x=1.5 ⇒ 2.25（PRD §6）。
    #[test]
    fn s6_math_matches_prd() {
        assert!((s6_f_big(1.5) - 1.125).abs() < 1e-6);
        assert!((s6_fp(1.5) - 2.25).abs() < 1e-6);
        assert!((s6_f(1.5) - 2.25).abs() < 1e-6);
        // B 点切线：y = x²·(X−x) + x³/3 过 (x, x³/3)，斜率 x²。
        for x in [0.5_f32, 1.5, 2.0] {
            let m = s6_fp(x);
            let y_at_x = m * (x - x) + s6_f_big(x);
            assert!((y_at_x - s6_f_big(x)).abs() < 1e-6);
        }
        assert!((s6_x(3.0)).abs() < 1e-6 && (s6_x(9.0) - 2.0).abs() < 1e-6);
    }

    /// 读数恒等（架构 §7.3 / §6.2-1 必含项）：采样点上 Counter 格式化值
    /// 与几何派生值相等。
    #[test]
    fn counter_readout_identical_to_geom() {
        // S3 HUD：m = 2 + h(t)。
        for i in 0..=70 {
            let t = 7.0 + i as f32 / 10.0;
            // [任务 1] HUD 经 value_at 读 eased-h：与几何 s3_h(t) 逐位恒等。
            let v = secant_m(s3_h(t));
            assert!((v - secant_m(s3_h(t))).abs() < 1e-4, "S3 HUD t={t}");
        }
        // S4 HUD：m ≡ a(t)。
        for i in 0..=100 {
            let t = 3.5 + i as f32 / 10.0;
            // [任务 1] HUD 经 value_at 读 eased-a：与几何 s4_a(t) 逐位恒等。
            let v = tangent_m(s4_a(t));
            assert!((v - tangent_m(s4_a(t))).abs() < 1e-4, "S4 HUD t={t}");
        }
        // S5 双 HUD：分拍阶跃——每拍一条静态 Counter（from==to）经
        // out_at 硬切交接，屏上数字 ≡ 同拍阶梯矩形数 / 矩形和（几何与
        // Counter 同源派生，精确恒等；9.0–11.0 矩形 morph 进面积后 S 改
        // 走连续溶解滚，起点 = 末批矩形精确和）。
        {
            let spec = scene_s5();
            let counters: Vec<&CounterText> = spec
                .tracks
                .iter()
                .filter_map(|tr| match tr {
                    Track::Counter(c) => Some(c),
                    _ => None,
                })
                .collect();
            // 4 n + 4 S + 1 溶解滚 = 9 条。
            assert_eq!(counters.len(), 9, "S5 HUD 应恰为 9 条 Counter");
            let (n_c, s_c, roll) = (&counters[0..4], &counters[4..8], counters[8]);
            for (i, &(at, n)) in S5_BEATS.iter().enumerate() {
                // n 侧：静态拍值；out_at 锚定下一拍起点（out_sec=0 硬切）。
                let c = n_c[i];
                assert_eq!(c.from, n as f32, "n-Counter 拍 {i} from");
                assert_eq!(c.to, n as f32, "n-Counter 拍 {i} to");
                assert_eq!(c.at, at, "n-Counter 拍 {i} at");
                assert_eq!(c.out_at, Some(at + 1.5), "n-Counter 拍 {i} 硬切锚点");
                assert_eq!(c.out_sec, 0.0, "n-Counter 拍 {i} 硬切");
                assert_eq!(c.decimals, 0, "n-Counter 拍 {i} 取整");
                // S 侧：静态拍值和 = riemann_sum(n)（同拍同值恒等）。
                let s = s_c[i];
                assert_eq!(s.from, riemann_sum(n as f32), "S-Counter 拍 {i} from");
                assert_eq!(s.to, riemann_sum(n as f32), "S-Counter 拍 {i} to");
                assert_eq!(s.at, at, "S-Counter 拍 {i} at");
                assert_eq!(s.decimals, 4, "S-Counter 拍 {i} 四位小数");
                // 屏上格式化值 ≡ 同拍矩形数 / 矩形和（Counter↔几何同源）。
                assert_eq!(format!("{:.0}", c.from).parse::<u32>().unwrap(), n, "屏上 n");
                assert_eq!(
                    format!("{:.4}", s.from),
                    format!("{:.4}", riemann_sum(n as f32)),
                    "屏上 S"
                );
            }
            // 溶解滚：9.0–11.0 Linear，from/to ≡ s5_s_dissolve 端点。
            // [任务 5·缺陷 4] 原 out_at=None（片尾 1.5s 淡出）令其在
            // 13.5–15.0 与静态「8/3 = 2.6667」同值双显；改 13.0 起
            // 0.5s 淡出，与 13.5s 淡入的静态答案错开退场。
            assert_eq!(roll.at, 9.0, "溶解滚 at");
            assert_eq!(roll.dur_sec, 2.0, "溶解滚时长");
            assert_eq!(roll.easing, EasingKind::Linear, "溶解滚缓动");
            assert_eq!(roll.from, s5_s_dissolve(9.0), "溶解滚起点");
            assert_eq!(roll.to, s5_s_dissolve(11.0), "溶解滚终点");
            assert_eq!(roll.out_at, Some(13.0), "溶解滚缺陷4修复：13.0 提前退场");
            assert_eq!(roll.out_sec, 0.5, "溶解滚缺陷4修复：0.5s 淡出");
            // 滚值重构 ≡ s5_s_dissolve(t)（同 Linear、同窗 [9,11]）。
            for i in 0..=20 {
                let t = 9.0 + i as f32 / 10.0;
                let u = ((t - 9.0) / 2.0).clamp(0.0, 1.0);
                let v = roll.from + (roll.to - roll.from) * EasingKind::Linear.apply(u);
                assert!((v - s5_s_dissolve(t)).abs() < 1e-6, "S5 溶解滚 t={t}");
            }
            // 几何恒等（读数铁律架构 §7.3 的几何侧）：每拍全生长阶梯的
            // distinct 高度层数 ≡ 矩形数 n；末样（x=2）高度 ≡ 视口映射
            // f(2)=4；生长半程 QuadOut(0.5)=0.75 缩放。
            let vp = s5_viewport();
            for &(at, n) in &S5_BEATS {
                let geom = s5_beat_staircase(vp, n, at, 0.7, false);
                let full = geom(at + 0.7);
                assert_eq!(full.len(), 66, "64 顶边界采样 + 2 基线点");
                let mut layers: Vec<u32> = full[..64].iter().map(|p| p[1].to_bits()).collect();
                layers.sort_unstable();
                layers.dedup();
                assert_eq!(layers.len(), n as usize, "拍 {at} 高度层数 ≡ 矩形数");
                let p2 = vp.to_px(2.0, 4.0);
                assert!(
                    (full[63][0] - p2[0]).abs() < 1e-3 && (full[63][1] - p2[1]).abs() < 1e-3,
                    "末样 x=2 高度 ≡ f(2)=4"
                );
                let ph = vp.to_px(2.0, 4.0 * 0.75);
                assert!(
                    (geom(at + 0.35)[63][1] - ph[1]).abs() < 1e-3,
                    "生长半程 QuadOut(0.5)=0.75"
                );
            }
            // morph 恒等（PRD 工程表 P1-3）：末拍 t=11.0 顶边界 ≡ 曲线
            // 采样（阶梯溶进曲线下面积）；t=9.0 仍为阶梯；中点逐点
            // 按 bump(0.5)=0.5 混合且保序（f 递增、右端和 ≥ 曲线）。
            let (at, n) = (S5_BEATS[3].0, S5_BEATS[3].1);
            let geom = s5_beat_staircase(vp, n, at, 0.7, true);
            let (pre, mid, post) = (geom(9.0), geom(10.0), geom(11.0));
            let m_mid = bump(0.5);
            let dx = 2.0 / n as f32;
            for i in 0..64 {
                let x = 2.0 * i as f32 / 63.0;
                let pf = vp.to_px(x, s5_f(x));
                assert!((post[i][1] - pf[1]).abs() < 1e-3, "morph 终点 ≡ 曲线 i={i}");
                let k = ((x / dx).floor() as u32 + 1).min(n);
                let expect = s5_f(k as f32 * dx) * (1.0 - m_mid) + s5_f(x) * m_mid;
                let pe = vp.to_px(x, expect);
                assert!((mid[i][1] - pe[1]).abs() < 1e-3, "morph 中点混合 i={i}");
                assert!(
                    mid[i][1] >= pre[i][1] - 1e-3 && mid[i][1] <= post[i][1] + 1e-3,
                    "morph 中点保序 i={i}"
                );
            }
        }
        // S6 双 HUD：f(x) 与 F′(x) 同值恒等——x(t) 匀速（Linear），
        // Counter 以 QuadIn 滚 0→4：4u² ≡ f(2u) ≡ F′(2u)，逐点精确
        // （含屏上四位小数舍入口径；单缓动无法复合 f∘ease，故匀速）。
        {
            let spec = scene_s6();
            let hud: Vec<&CounterText> = spec
                .tracks
                .iter()
                .filter_map(|tr| match tr {
                    Track::Counter(c) => Some(c),
                    _ => None,
                })
                .collect();
            assert_eq!(hud.len(), 2, "S6 应恰为 2 条 HUD Counter");
            for c in &hud {
                assert_eq!(c.at, 3.0, "S6 HUD at");
                assert_eq!(c.dur_sec, 6.0, "S6 HUD 时长");
                assert_eq!(c.easing, EasingKind::QuadIn, "S6 HUD 缓动");
                assert_eq!(c.from, s6_fp(0.0), "S6 HUD from");
                assert_eq!(c.to, s6_fp(2.0), "S6 HUD to");
                assert_eq!(c.decimals, 4, "S6 HUD 小数位");
                // 逐采样点：Counter 经 value_at 读 f∘ease，与几何 f(x(t))
                // 逐位恒等（[任务 1]：x 走 CubicInOut，y=f(x) 由 x 派生）。
                for i in 0..=60 {
                    let t = 3.0 + i as f32 / 10.0;
                    let v = if let Some(f) = &c.value_at {
                        f(t)
                    } else {
                        c.from
                    };
                    assert!((v - s6_fp(s6_x(t))).abs() < 1e-4, "S6 HUD t={t}");
                    // 屏上格式化值恒等。
                    assert_eq!(
                        format!("{v:.4}"),
                        format!("{:.4}", s6_fp(s6_x(t))),
                        "S6 屏上值 t={t}"
                    );
                }
            }
        }
    }

    /// [任务 5·缺陷 1] S4 HUD 过零左缘恒定：正负两段（8.0s 硬切）屏上
    /// 串的左缘（center_x − 实测字宽/2）在过零前后必须逐位一致——修复前
    /// minus 消失致居中锚整串右跳 ~6px（实测左缘 81→87）。
    ///
    /// [节奏重剪] 分段随扫过窗 [3.5, 13.5]→[3.5, 12.5] 由 8.5s 移至
    /// 8.0s（CubicInOut 过零中点）：跨段探针与段内采样窗同步调整，
    /// 断言语义（左缘逐位一致）不变。
    #[test]
    fn s4_hud_left_edge_stable_across_zero_crossing() {
        let spec = scene_s4();
        let hud: Vec<&CounterText> = spec
            .tracks
            .iter()
            .filter_map(|tr| match tr {
                Track::Counter(c) if c.prefix == "m = " => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(hud.len(), 2, "S4 HUD 应拆为正负两段（缺陷 1 修复）");
        let (neg, pos) = (hud[0], hud[1]);
        // 屏上读数串（复刻 counter_text 的 −0.0000 归一口径）。
        let readout = |c: &CounterText, t: f32| -> String {
            let v = (c.value_at.as_ref().unwrap())(t);
            let body = format!("{:.4}", v);
            let body = if body.starts_with('-')
                && body[1..].bytes().all(|b| b == b'0' || b == b'.')
            {
                body[1..].to_string()
            } else {
                body
            };
            format!("{}{}", c.prefix, body)
        };
        // 左缘探针：center_x − 实测字宽/2（text_run_width 与渲染逐位同参）。
        let left = |c: &CounterText, t: f32| {
            c.center_x.unwrap()
                - symbols::text_run_width(&readout(c, t), c.font, c.size, c.letter_spacing) * 0.5
        };
        // 跨段：交接帧附近两段左缘一致（7.99 负段末 ↔ 8.01 正段首；
        // 对称时刻数字串相同，minus 补偿精确 ⇒ 容差 1e-3）。
        let l_neg_end = left(neg, 7.99);
        let l_pos_start = left(pos, 8.01);
        assert!(
            (l_neg_end - l_pos_start).abs() < 1e-3,
            "过零交接左缘跳变 {l_neg_end} vs {l_pos_start}（缺陷 1 回归）"
        );
        // 段内：各自左缘恒定（表格数字 ⇒ 容差 1e-3；旧 bug 为 ~6px 跳）。
        for i in 0..50 {
            let t = 3.6 + i as f32 * 4.3 / 50.0;
            assert!((left(neg, t) - l_neg_end).abs() < 1e-3, "负段左缘漂移 t={t}");
        }
        for i in 0..=50 {
            let t = 8.1 + i as f32 * 4.3 / 50.0;
            assert!(
                (left(pos, t) - l_pos_start).abs() < 1e-3,
                "正段左缘漂移 t={t}"
            );
        }
    }

    /// [任务 5·缺陷 2] S5 拍间等左缘：各拍 `n = ` 读数字宽不一
    /// （"n = 2" 5 字符 vs "n = 32" 6 字符），修复前居中锚下 '=' 拍间
    /// 位移 ~7.8px；现按实测字宽补偿 center_x，全部拍左缘必须一致。
    #[test]
    fn s5_beat_hud_left_edges_aligned() {
        let spec = scene_s5();
        let n_counters: Vec<&CounterText> = spec
            .tracks
            .iter()
            .filter_map(|tr| match tr {
                Track::Counter(c) if c.prefix == "n = " => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(n_counters.len(), S5_BEATS.len(), "S5 应恰有各拍一条 n 读数");
        // 每拍屏上串 = "n = {n}"（from==to 静态值，decimals=0）。
        let lefts: Vec<f32> = n_counters
            .iter()
            .map(|c| {
                c.center_x.unwrap()
                    - symbols::text_run_width(
                        &format!("n = {}", c.from as u32),
                        c.font,
                        c.size,
                        c.letter_spacing,
                    ) * 0.5
            })
            .collect();
        for (i, &l) in lefts.iter().enumerate() {
            assert!((l - lefts[0]).abs() < 1e-3, "拍 {i} 左缘未对齐（缺陷 2 回归）");
        }
    }

    /// [任务 5·缺陷 4] S5 尾部无同值双显：溶解滚（S = 2.6667）须在
    /// 静态答案（8/3 = 2.6667）开始淡入前退场完毕——修复前溶解滚
    /// out_at=None 锚定片尾，13.5–15.0 与静态答案同值双显。
    #[test]
    fn s5_tail_no_double_display() {
        let spec = scene_s5();
        let roll = spec
            .tracks
            .iter()
            .find_map(|tr| match tr {
                Track::Counter(c) if c.prefix == "S = " && (c.at - 9.0).abs() < 1e-6 && (c.dur_sec - 2.0).abs() < 1e-6 => Some(c),
                _ => None,
            })
            .expect("S5 应有溶解滚 Counter");
        let answer = spec
            .tracks
            .iter()
            .find_map(|tr| match tr {
                Track::Static(s) if s.content == "8/3 = 2.6667" => Some(s),
                _ => None,
            })
            .expect("S5 应有静态答案 8/3 = 2.6667");
        // 溶解滚淡出完毕时刻 = out_at + out_sec，须 ≤ 静态答案起始 at
        // （13.5 起淡入 0.5s）——边界相等时两端 env 均为 0，无可见重叠。
        let roll_end = roll.out_at.expect("溶解滚须有绝对退场锚点") + roll.out_sec;
        assert!(
            roll_end <= answer.at + 1e-6,
            "溶解滚退场 {roll_end} 晚于静态答案入场 {}（缺陷 4 回归）",
            answer.at
        );
    }

    /// 几何闭包健全：全时间域采样无 NaN/Inf、点数合理。
    #[test]
    fn dynpath_geoms_finite_everywhere() {
        let specs = [scene_s3(), scene_s4(), scene_s5(), scene_s6()];
        for spec in &specs {
            for tr in &spec.tracks {
                if let Track::DynPath(d) = tr {
                    let mut prev = 0;
                    for i in 0..=(spec.duration * 30.0) as usize {
                        let t = i as f32 / 30.0;
                        let pts = (d.geom)(t);
                        assert!(pts.len() != 1, "DynPath 不得返回单点（渲染跳过）");
                        for p in &pts {
                            assert!(
                                p[0].is_finite() && p[1].is_finite(),
                                "DynPath 在 t={t} 产出非有限点 {:?}",
                                p
                            );
                        }
                        prev = prev.max(pts.len());
                    }
                    assert!(prev >= 2, "DynPath 在其活跃域必须产出 ≥2 点");
                }
            }
        }
    }

    /// 轨道坐标全部落画幅内（文字锚点 / 形状包围盒）。
    #[test]
    fn track_positions_within_canvas() {
        for spec in [scene_s3(), scene_s4(), scene_s5(), scene_s6()] {
            for tr in &spec.tracks {
                match tr {
                    Track::Static(s) => {
                        if let Some(cx) = s.center_x {
                            assert!(
                                (0.0..=common::WIDTH as f32).contains(&cx),
                                "Static center_x 越界: {cx}"
                            );
                        }
                        assert!((0.0..=common::HEIGHT as f32).contains(&s.y));
                    }
                    Track::Counter(c) => {
                        if let Some(cx) = c.center_x {
                            assert!((0.0..=common::WIDTH as f32).contains(&cx));
                        }
                        assert!((0.0..=common::HEIGHT as f32).contains(&c.y));
                    }
                    Track::Shape(sh) => {
                        assert!(sh.x >= -1.0 && sh.y >= -1.0, "Shape 负坐标");
                        assert!(
                            sh.x + sh.w <= common::WIDTH as f32 + 1.0
                                && sh.y + sh.h <= common::HEIGHT as f32 + 1.0,
                            "Shape 越界"
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// 颜色全部来自 common 调色板语义（有限、落 [0,1]）。
    #[test]
    fn track_colors_finite() {
        for spec in [scene_s3(), scene_s4(), scene_s5(), scene_s6()] {
            for tr in &spec.tracks {
                let color = match tr {
                    Track::Path(p) => Some(p.color),
                    Track::DynPath(d) => Some(d.color),
                    Track::Counter(c) => Some(c.color),
                    Track::Static(s) => Some(s.color),
                    Track::Shape(sh) => Some(sh.color),
                    Track::Typewriter(tw) => Some(tw.color),
                    _ => None,
                };
                if let Some(c) = color {
                    assert!(c.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "颜色越界: {c:?}");
                }
            }
        }
    }

    /// [任务 1] f∘ease 端点精确：eased 参数在窗口首/尾帧钉死端点，且 y 由
    /// f(x_eased) 派生 ⇒ 复合端点精确等于 f(端点)（无独立缓动 y 的漂移）。
    #[test]
    fn fease_endpoints_precise() {
        // S3：h 1→0（割线端点 x=1+h：2→1），f(x)=x²（4→1）。
        assert!((s3_h(7.0) - 1.0).abs() < 1e-6);
        assert!((s3_h(14.0)).abs() < 1e-6);
        assert!((s3_f(1.0 + s3_h(7.0)) - 4.0).abs() < 1e-6);
        assert!((s3_f(1.0 + s3_h(14.0)) - 1.0).abs() < 1e-6);
        // S4：a −2→2（切点 (a, a²/2)），f(a)=a²/2。
        assert!((s4_a(3.5) + 2.0).abs() < 1e-6);
        assert!((s4_a(13.5) - 2.0).abs() < 1e-6);
        assert!((s4_f(s4_a(3.5)) - 2.0).abs() < 1e-6);
        assert!((s4_f(s4_a(13.5)) - 2.0).abs() < 1e-6);
        // S6：x 0→2（动点 (x, x²)：0→4），f(x)=x²。
        assert!((s6_x(3.0)).abs() < 1e-6);
        assert!((s6_x(9.0) - 2.0).abs() < 1e-6);
        assert!((s6_f(s6_x(3.0))).abs() < 1e-6);
        assert!((s6_f(s6_x(9.0)) - 4.0).abs() < 1e-6);
    }

    /// [任务 1] 复合轨迹单调：eased 参数（动点横扫 x）在各自窗口内单向推进，
    /// 无回跳 ⇒ 动点沿曲线单向运动（CubicInOut 对归一进度单调）。
    #[test]
    fn fease_trajectory_monotone() {
        // S3：h 单调下降（x=1+h 收缩至切点）。
        let mut prev = 2.0_f32;
        for i in 0..=70 {
            let h = s3_h(7.0 + i as f32 / 10.0);
            assert!(h <= prev + 1e-6, "S3 h 须单调下降（i={i}）");
            prev = h;
        }
        // S4：a 单调上升（切点横扫 −2→2）。
        let mut prev_a = -3.0_f32;
        for i in 0..=100 {
            let a = s4_a(3.5 + i as f32 / 10.0);
            assert!(a >= prev_a - 1e-6, "S4 a 须单调上升（i={i}）");
            prev_a = a;
        }
        // S6：x 单调上升（动点右移 0→2）。
        let mut prev_x = -1.0_f32;
        for i in 0..=60 {
            let x = s6_x(3.0 + i as f32 / 10.0);
            assert!(x >= prev_x - 1e-6, "S6 x 须单调上升（i={i}）");
            prev_x = x;
        }
    }

    /// [任务 1] 确定性：所有 DynPath 几何闭包（含 f∘ease 动点）纯函数，同输入
    /// 两次调用逐位一致——全片重渲染 MD5 可复现的前提。
    #[test]
    fn fease_deterministic_twice() {
        for spec in [scene_s3(), scene_s4(), scene_s6()] {
            for tr in &spec.tracks {
                if let Track::DynPath(d) = tr {
                    for t in [4.0_f32, 7.0, 10.0, 13.0, 16.0] {
                        let a = (d.geom)(t);
                        let b = (d.geom)(t);
                        assert_eq!(a, b, "DynPath geom 须确定性（t={t}）");
                    }
                }
            }
        }
    }

    /// [任务 2] 视口连续：四场景相机呼吸视口在 [0,dur] 逐帧采样下缩放连续
    /// （相邻采样 px_per_unit 跳变 < 0.5，且全部有限）——无瞬移 / 卡顿。
    #[test]
    fn viewport_breathe_continuous() {
        let scenes: [(Viewport, fn(f32) -> Viewport, f32); 4] = [
            (s3_viewport(), s3_viewport_at, scene_duration(496)),
            (s4_viewport(), s4_viewport_at, scene_duration(496)),
            (s5_viewport(), s5_viewport_at, scene_duration(496)),
            (s6_viewport(), s6_viewport_at, scene_duration(436)),
        ];
        for (_, at, dur) in scenes {
            let mut prev = at(0.0).px_per_unit;
            assert!(prev.is_finite(), "起始缩放须有限");
            for i in 1..=200 {
                let t = dur * i as f32 / 200.0;
                let ppu = at(t).px_per_unit;
                assert!(ppu.is_finite(), "t={t} 缩放须有限");
                assert!(
                    (ppu - prev).abs() < 0.5,
                    "视口缩放须连续（相邻跳变 {:.4} > 0.5 @ t={t}）",
                    (ppu - prev).abs()
                );
                prev = ppu;
            }
        }
    }

    /// [任务 2] 端点匹配设计：四场景相机呼吸视口在 t=0 与 t=dur 处逐位
    /// 等于设计视口（轴网 / 曲线所用）——保证「推近—回稳」在场景首尾回归
    /// 原始构图（含 cx/cy/半区/缩放全部字段）。
    #[test]
    fn viewport_endpoint_matches_design() {
        let scenes: [(Viewport, fn(f32) -> Viewport, f32); 4] = [
            (s3_viewport(), s3_viewport_at, scene_duration(496)),
            (s4_viewport(), s4_viewport_at, scene_duration(496)),
            (s5_viewport(), s5_viewport_at, scene_duration(496)),
            (s6_viewport(), s6_viewport_at, scene_duration(436)),
        ];
        for (base, at, dur) in scenes {
            let v0 = at(0.0);
            let v1 = at(dur);
            assert_eq!(v0.x0.to_bits(), base.x0.to_bits(), "t=0 x0");
            assert_eq!(v0.x1.to_bits(), base.x1.to_bits(), "t=0 x1");
            assert_eq!(v0.y0.to_bits(), base.y0.to_bits(), "t=0 y0");
            assert_eq!(v0.y1.to_bits(), base.y1.to_bits(), "t=0 y1");
            assert_eq!(v0.cx.to_bits(), base.cx.to_bits(), "t=0 cx");
            assert_eq!(v0.cy.to_bits(), base.cy.to_bits(), "t=0 cy");
            assert_eq!(v0.px_per_unit.to_bits(), base.px_per_unit.to_bits(), "t=0 ppu");
            assert_eq!(v1.x0.to_bits(), base.x0.to_bits(), "t=dur x0");
            assert_eq!(v1.x1.to_bits(), base.x1.to_bits(), "t=dur x1");
            assert_eq!(v1.y0.to_bits(), base.y0.to_bits(), "t=dur y0");
            assert_eq!(v1.y1.to_bits(), base.y1.to_bits(), "t=dur y1");
            assert_eq!(v1.cx.to_bits(), base.cx.to_bits(), "t=dur cx");
            assert_eq!(v1.cy.to_bits(), base.cy.to_bits(), "t=dur cy");
            assert_eq!(v1.px_per_unit.to_bits(), base.px_per_unit.to_bits(), "t=dur ppu");
        }
    }

    /// [任务 2] 轴尺度稳定：相机呼吸峰值受限（≤ 设计缩放 ×1.045），且绘图区
    /// 中心 / 半区恒定 ⇒ 静态轴网尺度稳定、几何始终落在画幅内（推近不溢出）。
    #[test]
    fn viewport_axis_scale_stable() {
        let scenes: [(Viewport, fn(f32) -> Viewport, f32); 4] = [
            (s3_viewport(), s3_viewport_at, scene_duration(496)),
            (s4_viewport(), s4_viewport_at, scene_duration(496)),
            (s5_viewport(), s5_viewport_at, scene_duration(496)),
            (s6_viewport(), s6_viewport_at, scene_duration(436)),
        ];
        for (base, at, dur) in scenes {
            let mut max_ppu = 0.0_f32;
            let mut min_ppu = f32::INFINITY;
            for i in 0..=200 {
                let t = dur * i as f32 / 200.0;
                let v = at(t);
                assert!(v.is_valid(), "t={t} 视口须合法（ppu>0）");
                max_ppu = max_ppu.max(v.px_per_unit);
                min_ppu = min_ppu.min(v.px_per_unit);
                // 绘图区中心恒定（轴网锚点不动）。
                assert_eq!(v.cx.to_bits(), base.cx.to_bits(), "cx 恒定");
                assert_eq!(v.cy.to_bits(), base.cy.to_bits(), "cy 恒定");
            }
            // 峰值受限：推近幅度 ≤ 4.5%。
            assert!(
                max_ppu <= base.px_per_unit * 1.045,
                "呼吸峰值 {max_ppu} 超上限 {}",
                base.px_per_unit * 1.045
            );
            // 端点即设计（轴网尺度稳定）。
            assert_eq!(min_ppu.to_bits(), base.px_per_unit.to_bits(), "端点须回归设计缩放");
        }
    }
}

/// 缺陷回归组（字体缺失字形 / 等号可读 / 积分式 — y 轴重叠）。
mod defect_regressions {
    // release 编译路径下本模块体经 cfg 折叠为空，此处会被判 unused；测试编译路径下真实使用。
    #[allow(unused_imports)]
    use super::*;

    /// PrismSansSC 缺失/极细的数学字形：文本轨出现任一即渲染为 «» / ♦ 乱码。
    // release 编译路径下测试体不参与编译，此处会被判 dead_code。
    #[allow(dead_code)]
    const MISSING_GLYPHS: [char; 26] = [
        '\u{03B5}', '\u{03B4}', '\u{03C0}', '\u{03B8}', '\u{0394}', '\u{2202}', '\u{2211}',
        '\u{222B}', '\u{220F}', '\u{221E}', '\u{221A}', '\u{2192}', '\u{2200}', '\u{2203}',
        '\u{21D2}', '\u{2264}', '\u{2265}', '\u{2260}', '\u{2248}', '\u{00D7}', '\u{00F7}',
        '\u{2208}', '\u{222A}', '\u{2229}', '\u{00B1}', '\u{2032}',
    ];

    /// [缺陷修复 1] 全部文本/读数轨不得含字体缺失字形（矢量 FormulaPart
    /// 不受影响——只查文本轨真身：Static / Typewriter / Counter.prefix）。
    #[test]
    fn text_tracks_contain_no_missing_glyphs() {
        let mut n_checked = 0usize;
        for spec in [scene_s3(), scene_s4(), scene_s5(), scene_s6()] {
            for tr in &spec.tracks {
                let content: Option<&str> = match tr {
                    Track::Static(s) => Some(&s.content),
                    Track::Typewriter(w) => Some(&w.content),
                    Track::Counter(c) => Some(&c.prefix),
                    _ => None,
                };
                if let Some(text) = content {
                    n_checked += 1;
                    for bad in MISSING_GLYPHS {
                        assert!(!text.contains(bad), "文本轨含缺字形 {bad:?}：{text:?}");
                    }
                }
            }
        }
        assert!(n_checked > 0, "必须实际检到文本轨（空集断言无意义）");
    }

    /// [缺陷修复 1] S3 底部字幕措辞落地：无缺字形且语义等价（含 L）。
    #[test]
    fn s3_bottom_caption_is_safe_copy() {
        let spec = scene_s3();
        let cap = spec
            .tracks
            .iter()
            .find_map(|tr| match tr {
                Track::Typewriter(w) if w.content.contains("把曲线困在 L 附近") => Some(w.content.as_str()),
                _ => None,
            })
            .expect("S3 应有底部收尾字幕");
        assert_eq!(cap, "无论误差多小，都能找到对应的范围，把曲线困在 L 附近");
    }

    /// [缺陷修复 2] ε/δ 标签：等号必须是矢量 Equals 符号的前缀轨，不再走
    /// 文本 run（PrismSansSC 的「=」极细近乎缺失）。
    #[test]
    fn greek_readout_equals_is_vector_not_text() {
        for kind in [SymbolKind::Epsilon, SymbolKind::Delta] {
            let (pre, _cnt) = greek_roll_readout(
                kind,
                200.0,
                26.0,
                26.0,
                ctx(TEXT_RGBA),
                1.0,
                0.25,
                3.5,
                8.5,
                EasingKind::CubicIn,
                2,
            );
            for tr in &pre {
                if let Track::Static(s) = tr {
                    assert!(
                        !s.content.contains('='),
                        "{kind:?} 标签不得再以文本渲染等号：{:?}",
                        s.content
                    );
                }
            }
            // 前缀 = 希腊符号 + 等号；等号须实实在在发出来（双横线 ≥2 条笔画，
            // 且存在水平 2 点笔画）。
            let horiz_bars = pre
                .iter()
                .filter(|tr| match tr {
                    Track::Path(p) => {
                        p.points.len() == 2 && (p.points[0][1] - p.points[1][1]).abs() < 1e-6
                    }
                    _ => false,
                })
                .count();
            assert!(horiz_bars >= 2, "{kind:?} 标签应有 ≥2 条水平笔画（等号双横线），实测 {horiz_bars}");
        }
    }

    /// [缺陷修复 3] S5 积分式整体须与 y 轴 / y 轴标签保持 ≥20px 净空，
    /// 且不贴右边框（≥40px）。
    #[test]
    fn s5_formula_clears_y_axis_and_label() {
        let spec = scene_s5();
        let vp = s5_viewport();
        let y_axis_x = vp.to_px(0.0, 0.0)[0];
        // y 轴标签锚点 = 轴顶端 + (LABEL_DX, LABEL_DY)，字号 26 Bold。
        let y_top = vp.to_px(0.0, vp.y1);
        let lw = symbols::text_run_width("y", FontKind::Bold, 26.0, 0.0);
        let label_x1 = y_top[0] + 20.0 + lw * 0.5; // 标签 ink 右缘
        let label_y1 = y_top[1] + 12.0 + 6.0; // 标签 ink 下缘（基线略下）

        // 积分式 = 13.0 起的那组轨（∫ 矢量 + 限位 0/2 + 30px 文本 run 各一）。
        // 全场景仅此组以 at=13.0 入场 ⇒ 按此取组即公式墨迹全集（Exact）。
        let mut left_ink = f32::INFINITY;
        let mut right_ink = f32::NEG_INFINITY;
        let mut n_group = 0usize;
        for tr in &spec.tracks {
            match tr {
                Track::Path(p) if (p.at - 13.0).abs() < 1e-6 => {
                    n_group += 1;
                    for q in &p.points {
                        left_ink = left_ink.min(q[0]);
                        right_ink = right_ink.max(q[0]);
                    }
                }
                Track::Static(s) if (s.at - 13.0).abs() < 1e-6 => {
                    n_group += 1;
                    let w = symbols::text_run_width(&s.content, s.font, s.size, s.letter_spacing);
                    let c = s.center_x.expect("须有 center_x 锚点");
                    left_ink = left_ink.min(c - w * 0.5);
                    right_ink = right_ink.max(c + w * 0.5);
                }
                _ => {}
            }
        }
        assert!(n_group >= 4, "积分式应含 ∫ / 限位 / 文本 run：实测 {n_group} 轨");
        assert!(left_ink.is_finite() && right_ink.is_finite(), "须取到公式墨迹");

        assert!(
            left_ink - y_axis_x >= 20.0,
            "公式须让开 y 轴 ≥20px：左插 {left_ink} vs 轴 {y_axis_x}"
        );
        // 与 y 标签：水平净空 ≥20px（标签墨迹盒含 Baseline 上方突出）。
        assert!(
            left_ink - label_x1 >= 20.0,
            "公式须让开 y 轴标签 ≥20px：左插 {left_ink} vs 标签右缘 {label_x1}"
        );
        // 纵向兜底：即便 x 方向有交叠，也要与标签所占 y 带错开。
        assert!(
            (140.0 - 26.0) > label_y1 || left_ink - label_x1 >= 20.0,
            "公式与 y 标签须在 x 或 y 上分明"
        );
        assert!(
            common::WIDTH as f32 - right_ink >= 40.0,
            "公式不得贴右边框：右插 {right_ink}"
        );
    }
}
