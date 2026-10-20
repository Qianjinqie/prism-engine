//! 内部渲染实现：背景绘制、轨道绘制与 MP4 封装。
//!
//! 全部"脏活"都收在这里：背景绘制复用 `prism-noise::Fbm`；文字复用
//! `prism-text` 的 `layout_text` / `blit_glyph` / `draw_glyphs` /
//! `render_caption`；矢量图元复用 `prism-raster::fill_path`；最后经官方
//! muxer `prism-mux::write_mp4_jpeg_color` 封装。作者不碰 `Fbuf16F`、帧循环
//! 与编码器。

use prism_io::Fbuf16F;
use prism_mux::{avi::write_avi_mjpeg, write_mp4_jpeg_color, JpegConfig, TrackSpec};
use prism_noise::fbm::Fbm;
use prism_raster::draw_line;
use prism_raster::fill_path;
use prism_raster::geom::{Point, Transform};
use prism_raster::path::PathBuilder;
use prism_raster::scanline::FillRule;
use prism_text::cache::GlyphCache;
use prism_text::layout::{layout_text, Align, TextParams};
use prism_text::presets::render_caption;
use prism_text::raster::{blit_glyph, draw_glyphs};
use prism_text::ttf::Font;
use prism_text::{font_bold, font_regular};

use crate::easing::EasingKind;
use crate::spec::{
    Background, ClipSpec, CounterText, DynPathTrack, FontKind, ImageSource, MorphSpec, PathTrack,
    SequenceTrack, ShapeKind, Track, Transition, WipeDir,
};

// v1.5-C（ADR-18）：转场混合下沉 core 走**整数路径**——u8 域混合器 `mix_u8`、
// 整数 Wipe 权重（分界行程扩展）、整数 Dissolve hash、整数 Push 推进量。
// 全部零浮点 ⇒ 同输入两跑 bitwise 相同，且跨端（CPU / GPU / 架构）逐位一致；
// 同时消掉原 f32 截断路径的系统暗偏（mean bias ≈ −0.488 LSB）。
use prism_core::sceneseq::{dissolve_hash, mix_u8, push_c, wipe_weight, W_ONE};

// ---- 通用数学辅助（全部 t 的纯函数）----

/// 三次平滑插值（smoothstep），用于淡入 / 暗角等缓动。
#[inline]
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 标量线性插值。
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 四维（RGBA）线性插值。
#[inline]
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

/// 三维（RGB）线性插值。
#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

/// 淡出起点解析（[`fade_factor`] 与 [`image_offset`] 共用）：
/// - `out_at = Some(a)`（有限值）：绝对锚点 `a`（钳到不早于 `at`）；
/// - `out_at = None`：缺省锚定片尾 `duration - out_sec`（同样钳到 `at`）。
///
/// 返回 `None` 表示「不淡出」（缺省锚点且 `out_sec <= 0`，可见至片尾）。
/// 非有限 `out_at`（NaN/Inf）按 `None` 处理——数据违约不 panic。
#[inline]
pub(crate) fn out_start_of(out_at: Option<f32>, out_sec: f32, at: f32, duration: f32) -> Option<f32> {
    match out_at.filter(|a| a.is_finite()) {
        Some(a) => Some(a.max(at)),
        None => {
            if out_sec <= 0.0 {
                None
            } else {
                Some((duration - out_sec).max(at))
            }
        }
    }
}

/// 计算轨道在时刻 `t` 的全局淡入淡出透明度（0..1）。
///
/// - 淡入：`[at, at+in_sec]` 平滑升到 1；之前为 0；
/// - 淡出：`[out_start, out_start+out_sec]` 平滑降到 0，其中
///   `out_start` 由 [`out_start_of`] 解析——`out_at = Some(a)` 时取
///   绝对锚点 `a`（分拍阶跃 staging：同一位置多条顺序轨轮流显现），
///   缺省锚定片尾 `duration - out_sec`；
/// - `out_at = Some(a)` 且 `out_sec = 0`：`a` 处**硬切**（无过渡）；
/// - 中间稳态为 1。
#[inline]
pub(crate) fn fade_factor(
    t: f32,
    at: f32,
    in_sec: f32,
    out_sec: f32,
    out_at: Option<f32>,
    duration: f32,
) -> f32 {
    let fin = if in_sec <= 0.0 {
        if t >= at {
            1.0
        } else {
            0.0
        }
    } else {
        smooth(((t - at) / in_sec).clamp(0.0, 1.0))
    };
    let out_start = match out_start_of(out_at, out_sec, at, duration) {
        Some(s) => s,
        None => return fin,
    };
    let fout = if out_sec <= 0.0 {
        // 绝对锚点 + 零淡出时长 = 硬切（分拍阶跃的瞬时换版）。
        if t >= out_start {
            0.0
        } else {
            1.0
        }
    } else {
        1.0 - smooth(((t - out_start) / out_sec).clamp(0.0, 1.0))
    };
    fin * fout
}

/// 计算 Image 轨道的位移偏移（入场由 `move_from` 缓动归零，出场退回）。
#[inline]
fn image_offset(
    t: f32,
    at: f32,
    in_sec: f32,
    out_sec: f32,
    out_at: Option<f32>,
    move_from: [f32; 2],
    duration: f32,
) -> [f32; 2] {
    let tl = t - at;
    let in_p = if in_sec <= 0.0 {
        if tl >= 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        smooth((tl / in_sec).clamp(0.0, 1.0))
    };
    let out_p = match out_start_of(out_at, out_sec, at, duration) {
        None => 0.0,
        Some(out_start) => {
            if out_sec <= 0.0 {
                // 硬切锚点：out_start 起位移一步退满（与淡出硬切语义对齐）。
                if t >= out_start {
                    1.0
                } else {
                    0.0
                }
            } else {
                smooth(((t - out_start) / out_sec).clamp(0.0, 1.0))
            }
        }
    };
    let k = (1.0 - in_p) + out_p; // 入场前=1、稳态=0、出场末=1
    [move_from[0] * k, move_from[1] * k]
}

// ---- 背景 ----

/// 默认深色渐变两端（与原 demo 完全一致）。
const BG_TOP: [f32; 3] = [0.035, 0.055, 0.130]; // ≈ #090E21
const BG_BOTTOM: [f32; 3] = [0.058, 0.090, 0.205]; // ≈ #0F1734
/// 辉光冷青调（叠加在渐变上）。
const GLOW: [f32; 3] = [0.030, 0.070, 0.120];

/// 绘制背景：平滑垂直渐变 + 柔光 + 极低频 simplex 微波动 + 暗角 + 呼吸。
/// 复用 `prism-noise::Fbm`，全套逻辑封装在 API 内部（原 demo 的嵌套循环）。
fn paint_background(
    fbuf: &mut Fbuf16F,
    t: f32,
    fbm: &Fbm,
    seed: u64,
    top: [f32; 3],
    bottom: [f32; 3],
) {
    let w = fbuf.width as f32;
    let h = fbuf.height as f32;

    // 低分辨率噪声网格（双线性上采样 → 极低频、绝无高频噪点）。
    let gw: u32 = 56;
    let gh: u32 = 32;
    let mut grid = vec![0.0f32; (gw * gh) as usize];
    for gy in 0..gh {
        for gx in 0..gw {
            let nx = gx as f32 / gw as f32 * 2.2;
            let ny = gy as f32 / gh as f32 * 1.3;
            let v = fbm.simplex2(seed, nx + t * 0.05, ny - t * 0.035);
            grid[(gy * gw + gx) as usize] = v;
        }
    }

    let breath = 1.0 + 0.015 * (t * (std::f32::consts::TAU / 5.0)).sin();
    let glow_cx = 0.5;
    let glow_cy = 0.40;

    for y in 0..fbuf.height {
        let vy = y as f32 / (h - 1.0);
        for x in 0..fbuf.width {
            let vx = x as f32 / (w - 1.0);

            // 1) 垂直渐变。
            let mut col = lerp3(top, bottom, smooth(vy));

            // 2) 柔和辉光（二次衰减，冷青调）。
            let dx = vx - glow_cx;
            let dy = vy - glow_cy;
            let d2 = dx * dx + dy * dy;
            let mut g = 1.0 - d2 * 3.0;
            if g < 0.0 {
                g = 0.0;
            }
            g *= g;
            col[0] += g * GLOW[0];
            col[1] += g * GLOW[1];
            col[2] += g * GLOW[2];

            // 3) 极低对比缓慢波动（双线性采样）。
            let gx_f = vx * (gw - 1) as f32;
            let gy_f = vy * (gh - 1) as f32;
            let gxi0 = gx_f.floor() as u32;
            let gyi0 = gy_f.floor() as u32;
            let gxi1 = (gxi0 + 1).min(gw - 1);
            let gyi1 = (gyi0 + 1).min(gh - 1);
            let fx = gx_f - gxi0 as f32;
            let fy = gy_f - gyi0 as f32;
            let v00 = grid[(gyi0 * gw + gxi0) as usize];
            let v10 = grid[(gyi0 * gw + gxi1) as usize];
            let v01 = grid[(gyi1 * gw + gxi0) as usize];
            let v11 = grid[(gyi1 * gw + gxi1) as usize];
            let n = lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
            let modulation = 1.0 + 0.03 * n;

            // 4) 暗角。
            let cx = vx - 0.5;
            let cy = vy - 0.5;
            let rad = (cx * cx + cy * cy).sqrt();
            let vig = 1.0 - smooth((rad - 0.45) / 0.55).max(0.0) * 0.55;

            // 5) 合成。
            let k = modulation * breath * vig;
            fbuf.set(
                x,
                y,
                [
                    (col[0] * k).clamp(0.0, 1.0),
                    (col[1] * k).clamp(0.0, 1.0),
                    (col[2] * k).clamp(0.0, 1.0),
                    1.0,
                ],
            );
        }
    }
}

// ---- 新增背景预设 ----

/// 低频噪声网格双线性上采样（网格由调用方预计算）。
///
/// 与 [`paint_background`] 第 3 步的采样公式完全一致——极低频、
/// 绝无高频噪点（项目铁律：不手写逐像素高频噪声）。
#[inline]
fn sample_grid(grid: &[f32], gw: u32, gh: u32, vx: f32, vy: f32) -> f32 {
    let gx_f = vx * (gw - 1) as f32;
    let gy_f = vy * (gh - 1) as f32;
    let gxi0 = gx_f.floor() as u32;
    let gyi0 = gy_f.floor() as u32;
    let gxi1 = (gxi0 + 1).min(gw - 1);
    let gyi1 = (gyi0 + 1).min(gh - 1);
    let fx = gx_f - gxi0 as f32;
    let fy = gy_f - gyi0 as f32;
    let v00 = grid[(gyi0 * gw + gxi0) as usize];
    let v10 = grid[(gyi0 * gw + gxi1) as usize];
    let v01 = grid[(gyi1 * gw + gxi0) as usize];
    let v11 = grid[(gyi1 * gw + gxi1) as usize];
    lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy)
}

/// 极光色带参数（私有；全部为归一化或线性量）。
struct AuroraBand {
    /// 基线中心（归一化 vy）。
    cy: f32,
    /// 半高（归一化；抛物线衰减到 0 的距离）。
    half_h: f32,
    /// 峰值增益（线性 RGB 叠加量）。
    gain: f32,
    /// 色带颜色（线性 RGB）。
    color: [f32; 3],
    /// 水平流动速度（噪声坐标漂移，1/秒）。
    speed: f32,
    /// 噪声垂直弯曲幅度（归一化）。
    warp_amp: f32,
    /// 噪声种子偏移（令三条带互不相同）。
    seed_off: u64,
}

/// 极光背景：深色渐变（复用 BG_TOP / BG_BOTTOM）上叠加三条青绿 / 青蓝
/// 水平流动色带；色带中心与强度由 `Fbm` 低频噪声网格（双线性上采样）
/// 驱动，网格随 `t` 平移 → 色带水平流动。全套为 (x, y, t, seed) 的纯函数，
/// 确定性；暗角压边保证文字可读。
fn paint_aurora(fbuf: &mut Fbuf16F, t: f32, fbm: &Fbm, seed: u64) {
    let w = fbuf.width as f32;
    let h = fbuf.height as f32;

    // 三条水平流动色带：青绿 → 青蓝 → 青绿（压低饱和，避免糊掉文字）。
    let bands = [
        AuroraBand {
            cy: 0.34,
            half_h: 0.15,
            gain: 0.20,
            color: [0.10, 0.85, 0.62],
            speed: 0.055,
            warp_amp: 0.10,
            seed_off: 0,
        },
        AuroraBand {
            cy: 0.50,
            half_h: 0.19,
            gain: 0.15,
            color: [0.14, 0.60, 0.98],
            speed: -0.040,
            warp_amp: 0.13,
            seed_off: 7_717,
        },
        AuroraBand {
            cy: 0.67,
            half_h: 0.13,
            gain: 0.13,
            color: [0.34, 0.88, 0.84],
            speed: 0.030,
            warp_amp: 0.08,
            seed_off: 15_431,
        },
    ];

    // 每条色带一张低频噪声网格（双线性上采样；随 t 平移 → 水平流动）。
    const GW: u32 = 56;
    const GH: u32 = 32;
    let mut grids: Vec<Vec<f32>> = Vec::with_capacity(bands.len());
    for b in &bands {
        let mut g = vec![0.0f32; (GW * GH) as usize];
        for gy in 0..GH {
            for gx in 0..GW {
                let nx = gx as f32 / GW as f32 * 2.2;
                let ny = gy as f32 / GH as f32 * 1.3;
                g[(gy * GW + gx) as usize] =
                    fbm.simplex2(seed + b.seed_off, nx + t * b.speed, ny - t * b.speed * 0.6);
            }
        }
        grids.push(g);
    }

    let breath = 1.0 + 0.02 * (t * (std::f32::consts::TAU / 6.0)).sin();

    for y in 0..fbuf.height {
        let vy = y as f32 / (h - 1.0);
        for x in 0..fbuf.width {
            let vx = x as f32 / (w - 1.0);

            // 1) 基底：深色垂直渐变（复用既有常量）。
            let mut col = lerp3(BG_TOP, BG_BOTTOM, smooth(vy));

            // 2) 三条极光带：噪声弯曲中心 + 抛物线衰减叠加。
            for (k, b) in bands.iter().enumerate() {
                let n = sample_grid(&grids[k], GW, GH, vx, vy);
                let cy = b.cy + n * b.warp_amp;
                let d = (vy - cy) / b.half_h;
                let fall = (1.0 - d * d).max(0.0);
                let inten = fall * fall * b.gain * breath;
                col[0] += b.color[0] * inten;
                col[1] += b.color[1] * inten;
                col[2] += b.color[2] * inten;
            }

            // 3) 暗角（同款公式，压边保可读）。
            let cx = vx - 0.5;
            let cyv = vy - 0.5;
            let rad = (cx * cx + cyv * cyv).sqrt();
            let vig = 1.0 - smooth((rad - 0.45) / 0.55).max(0.0) * 0.55;

            fbuf.set(
                x,
                y,
                [
                    (col[0] * vig).clamp(0.0, 1.0),
                    (col[1] * vig).clamp(0.0, 1.0),
                    (col[2] * vig).clamp(0.0, 1.0),
                    1.0,
                ],
            );
        }
    }
}

/// 扫描线背景：暗渐变 + 同款柔光 / 波动 / 暗角（复用 BG_TOP / BG_BOTTOM /
/// GLOW 与 `Fbm`），再叠加等距水平细扫描线（每 3px 一条，线心压暗 18%，
/// 三角形剖面无高频跳变）与一条缓慢下扫的刷新高光（6 秒一轮）。
/// 全套为 (x, y, t, seed) 的纯函数，确定性。
fn paint_scanline(fbuf: &mut Fbuf16F, t: f32, fbm: &Fbm, seed: u64) {
    let w = fbuf.width as f32;
    let h = fbuf.height as f32;

    // 与 DarkGlow 同款的极低频波动网格（复用同一 seed 与采样参数）。
    let gw: u32 = 56;
    let gh: u32 = 32;
    let mut grid = vec![0.0f32; (gw * gh) as usize];
    for gy in 0..gh {
        for gx in 0..gw {
            let nx = gx as f32 / gw as f32 * 2.2;
            let ny = gy as f32 / gh as f32 * 1.3;
            grid[(gy * gw + gx) as usize] = fbm.simplex2(seed, nx + t * 0.05, ny - t * 0.035);
        }
    }

    let breath = 1.0 + 0.015 * (t * (std::f32::consts::TAU / 5.0)).sin();
    let glow_cx = 0.5;
    let glow_cy = 0.40;

    // 等距水平细扫描线：周期 3px、线心压暗 18%（整数取模 → 精确等距）。
    const SCAN_PERIOD: f32 = 3.0;
    const SCAN_DARKEN: f32 = 0.18;
    // 缓慢下扫的刷新高光（6 秒一轮；`fract` 保证周期性）。
    let sweep_y = (t / 6.0).fract();

    for y in 0..fbuf.height {
        let vy = y as f32 / (h - 1.0);
        // 扫描线剖面：距最近线心越近越暗（三角形剖面，线心 = 1）。
        let m = (y as f32 % SCAN_PERIOD) / SCAN_PERIOD;
        let line = 1.0 - 2.0 * m.min(1.0 - m);
        for x in 0..fbuf.width {
            let vx = x as f32 / (w - 1.0);

            // 1) 垂直渐变。
            let mut col = lerp3(BG_TOP, BG_BOTTOM, smooth(vy));

            // 2) 柔和辉光（二次衰减，冷青调；同 DarkGlow）。
            let dx = vx - glow_cx;
            let dy = vy - glow_cy;
            let d2 = dx * dx + dy * dy;
            let mut g = 1.0 - d2 * 3.0;
            if g < 0.0 {
                g = 0.0;
            }
            g *= g;
            col[0] += g * GLOW[0];
            col[1] += g * GLOW[1];
            col[2] += g * GLOW[2];

            // 3) 极低对比缓慢波动（双线性采样，复用公共 helper）。
            let modulation = 1.0 + 0.03 * sample_grid(&grid, gw, gh, vx, vy);

            // 4) 暗角。
            let cx = vx - 0.5;
            let cyv = vy - 0.5;
            let rad = (cx * cx + cyv * cyv).sqrt();
            let vig = 1.0 - smooth((rad - 0.45) / 0.55).max(0.0) * 0.55;

            // 5) 缓慢下扫的刷新高光（细带，冷白，极弱）。
            let sweep = (1.0 - smooth((vy - sweep_y).abs() / 0.08)) * 0.05;

            // 6) 合成：先压扫描线，再乘包络，最后叠高光。
            let scan = 1.0 - SCAN_DARKEN * line;
            let k = modulation * breath * vig * scan;
            fbuf.set(
                x,
                y,
                [
                    (col[0] * k + sweep).clamp(0.0, 1.0),
                    (col[1] * k + sweep).clamp(0.0, 1.0),
                    (col[2] * k + sweep).clamp(0.0, 1.0),
                    1.0,
                ],
            );
        }
    }
}

/// 背景分发：既有两种背景走原 [`paint_background`]（行为逐字节不变），
/// 新增预设（Aurora / Scanline）走各自画家。
fn paint_bg(fbuf: &mut Fbuf16F, t: f32, fbm: &Fbm, seed: u64, bg: &Background) {
    match bg {
        Background::DarkGlow => paint_background(fbuf, t, fbm, seed, BG_TOP, BG_BOTTOM),
        Background::Gradient { top, bottom } => paint_background(fbuf, t, fbm, seed, *top, *bottom),
        Background::Aurora => paint_aurora(fbuf, t, fbm, seed),
        Background::Scanline => paint_scanline(fbuf, t, fbm, seed),
    }
}

// ---- 文字轨道 ----

/// 逐字 kinetic 入场（复用 `prism-text` 的 `layout_text` / `blit_glyph` /
/// `GlyphCache`；相对 `presets::render_kinetic` 多了基线定位偏移与全局淡入
/// 信封，使之可自由摆放到任意 `y`）。
fn draw_kinetic(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    size: f32,
    color: [f32; 4],
    center_x: f32,
    baseline_y: f32,
    stagger: f32,
    per_char: f32,
    scale_from: f32,
    rise_px: f32,
    easing: EasingKind,
    tl: f32,
    env: f32,
    cache: &mut GlyphCache,
) {
    if text.is_empty() || env <= 0.0 {
        return;
    }
    let params = TextParams {
        font_size: size,
        max_width: f32::INFINITY,
        align: Align::Center,
        ..TextParams::default()
    };
    let layout = layout_text(font, text, &params);
    if layout.glyphs.is_empty() {
        return;
    }
    let scale = size / font.units_per_em() as f32;
    let asc = font.ascender() as f32 * scale;
    let shift_x = center_x - layout.width * 0.5;
    let shift_y = baseline_y - asc;
    for (i, g) in layout.glyphs.iter().enumerate() {
        let local = (tl - (i as f32) * stagger) / per_char;
        let u = local.clamp(0.0, 1.0);
        let e = easing.apply(u);
        if e <= 0.0 {
            continue; // 未入场字符不产出像素
        }
        let s = size * (scale_from + (1.0 - scale_from) * e);
        let bmp = cache.get_or_render(font, g.gid, s, 0);
        let dy = rise_px * (1.0 - e);
        let col = [color[0], color[1], color[2], color[3] * e * env];
        blit_glyph(fbuf, bmp, g.x + shift_x, g.y + shift_y + dy, col);
    }
}

/// 静态居中文字（带字距），用于题头等小标签。
fn draw_static(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    size: f32,
    color: [f32; 4],
    center_x: f32,
    baseline_y: f32,
    letter_spacing: f32,
    cache: &mut GlyphCache,
) {
    if text.is_empty() {
        return;
    }
    let params = TextParams {
        font_size: size,
        letter_spacing,
        max_width: f32::INFINITY,
        align: Align::Center,
        ..TextParams::default()
    };
    let layout = layout_text(font, text, &params);
    if layout.glyphs.is_empty() {
        return;
    }
    let scale = size / font.units_per_em() as f32;
    let asc = font.ascender() as f32 * scale;
    let shift_x = center_x - layout.width * 0.5;
    let shift_y = baseline_y - asc;
    draw_glyphs(fbuf, font, &layout.glyphs, size, color, shift_x, shift_y, cache);
}

/// 打字机轨道：逐字显示 + 可选闪烁光标。
///
/// - 可见字符数 `n = floor((t - at) * cps).clamp(0, glyphs.len())`，只画前
///   `n` 个字形（复用 `draw_glyphs`，居中对齐同 [`draw_static`]）；
///   `cps <= 0` 视为立即全文显示；
/// - 光标：若 `cursor`，在最后一个可见字形之后画一条竖线（宽 ≈
///   `size * 0.06`，高 = asc + desc，颜色同文字色，复用 [`fill_rect`]），
///   按 `fract(t * cursor_blink_hz)` 决定亮 / 灭（`blink_hz <= 0` 常亮）；
/// - `env` 为调用方算好的全局淡入淡出信封，`env <= 0` 直接跳过。
///
/// `tl` = `t - at`（驱动逐字进度），`t_sec` = 帧绝对时间（驱动光标闪烁相位）。
fn draw_typewriter(
    fbuf: &mut Fbuf16F,
    font: &Font,
    text: &str,
    size: f32,
    color: [f32; 4],
    center_x: f32,
    baseline_y: f32,
    letter_spacing: f32,
    cps: f32,
    cursor: bool,
    cursor_blink_hz: f32,
    tl: f32,
    t_sec: f32,
    env: f32,
    cache: &mut GlyphCache,
) {
    if text.is_empty() || env <= 0.0 {
        return;
    }
    let params = TextParams {
        font_size: size,
        letter_spacing,
        max_width: f32::INFINITY,
        align: Align::Center,
        ..TextParams::default()
    };
    let layout = layout_text(font, text, &params);
    if layout.glyphs.is_empty() {
        return;
    }

    // 可见字符数（cps <= 0 → 立即全文；防御性 clamp 掉负值与 NaN 之前的越界）。
    let n = if cps <= 0.0 {
        layout.glyphs.len()
    } else {
        (tl * cps).floor().clamp(0.0, layout.glyphs.len() as f32) as usize
    };

    let scale = size / font.units_per_em() as f32;
    let asc = font.ascender() as f32 * scale;
    let desc = -(font.descender() as f32) * scale; // descender 为负 → 取正
    let shift_x = center_x - layout.width * 0.5;
    let shift_y = baseline_y - asc;

    // 已"打出来"的字形（颜色 alpha 已含淡入淡出信封）。
    let mut col = color;
    col[3] *= env;
    draw_glyphs(fbuf, font, &layout.glyphs[..n], size, col, shift_x, shift_y, cache);

    if !cursor {
        return;
    }
    // 闪烁相位：`fract(t * hz) < 0.5` 为亮；hz <= 0 时常亮。
    let phase = if cursor_blink_hz > 0.0 {
        (t_sec * cursor_blink_hz).fract()
    } else {
        0.0
    };
    if phase >= 0.5 {
        return;
    }
    // 光标位置：最后一个可见字形之后的笔位（n == 0 时在起始笔位）。
    let cur_x = if n == 0 {
        shift_x
    } else {
        let last = &layout.glyphs[n - 1];
        shift_x + last.x + last.advance
    };
    fill_rect(fbuf, cur_x, shift_y, size * 0.06, asc + desc, col);
}

/// 实心矩形填充：复用引擎矢量填充管线（`prism-raster::fill_path`），
/// 不手写像素循环。
fn fill_rect(fbuf: &mut Fbuf16F, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    let mut pb = PathBuilder::new();
    pb.move_to(Point::new(x, y));
    pb.line_to(Point::new(x + w, y));
    pb.line_to(Point::new(x + w, y + h));
    pb.line_to(Point::new(x, y + h));
    pb.close();
    fill_path(fbuf, &pb, &Transform::IDENTITY, color, FillRule::Winding);
}

// ---- 图像叠加层 ----

/// 把一个图像源合成进帧缓冲（支持纯色 / 渐变面板 / 预渲染 RGBA 缓冲）。
///
/// `alpha` 为全局不透明度（已含淡入淡出信封）；逐像素源色与之相乘后按
/// 直通 alpha src-over 合成。`(x, y)` 为左上角（已含位移偏移）。
fn draw_image(fbuf: &mut Fbuf16F, src: &ImageSource, x: f32, y: f32, w: f32, h: f32, alpha: f32) {
    if alpha <= 0.0 {
        return;
    }
    // 预渲染 RGBA 缓冲分流至共用 blit 核心（与 `Track::Sequence` 同一条
    // 光栅/blit 代码，零拷贝借入）；下方循环只处理纯色 / 渐变面板。
    if let ImageSource::Buffer { data, width, height } = src {
        blit_rgba8(fbuf, data, *width, *height, x, y, w, h, alpha);
        return;
    }
    let iw = w.max(0.0);
    let ih = h.max(0.0);
    if iw <= 0.0 || ih <= 0.0 {
        return;
    }
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let cols = iw.ceil() as i64;
    let rows = ih.ceil() as i64;
    for row in 0..rows {
        let fy = y0 + row;
        if fy < 0 || fy >= fbuf.height as i64 {
            continue;
        }
        let vy = if ih > 1.0 { row as f32 / (ih - 1.0) } else { 0.0 };
        for col in 0..cols {
            let fx = x0 + col;
            if fx < 0 || fx >= fbuf.width as i64 {
                continue;
            }
            let src_rgba: [f32; 4] = match src {
                ImageSource::Solid(c) => *c,
                ImageSource::Gradient { top, bottom } => lerp4(*top, *bottom, vy),
                // 不会到达：Buffer 已在函数入口分流至 `blit_rgba8`；
                // 防御性 continue（数据违约不 panic）。
                ImageSource::Buffer { .. } => continue,
            };
            let sa = src_rgba[3] * alpha;
            if sa <= 0.0 {
                continue;
            }
            let c = [src_rgba[0], src_rgba[1], src_rgba[2], sa];
            let dst = fbuf.get(fx as u32, fy as u32);
            fbuf.set(fx as u32, fy as u32, prism_raster::blend_pixel_over(c, dst));
        }
    }
}

/// RGBA8 缓冲 blit 核心（[`ImageSource::Buffer`] 与 [`SequenceTrack`] 共用）：
/// 把 `data`（`bw×bh`，行主序 RGBA8）拉伸合成到目标矩形 `(x, y, w, h)`，
/// 逐像素源色乘 `alpha` 后按直通 alpha src-over 合成。
///
/// - 采样为最近邻（`round(v·(dim-1))`，clamp 到边界）；源尺寸 == 目标矩形
///   时为精确 1:1（`round(vx*(bw-1))` 恒等于列号）；
/// - `alpha <= 0` / 目标矩形非正 / 源尺寸为 0 ⇒ 不产出像素；
/// - 帧字节数不足（`bi + 3 >= data.len()`）的像素静默跳过，不 panic；
/// - 从 `draw_image` 的 Buffer 分支逐句提取（同一 `x0/cols/vy/vx/bx/by/bi`
///   算术与 [`prism_raster::blend_pixel_over`]），保证 v1 行为逐字节不变。
fn blit_rgba8(
    fbuf: &mut Fbuf16F,
    data: &[u8],
    bw: u32,
    bh: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    alpha: f32,
) {
    if alpha <= 0.0 {
        return;
    }
    let iw = w.max(0.0);
    let ih = h.max(0.0);
    if iw <= 0.0 || ih <= 0.0 {
        return;
    }
    let bw = bw as i64;
    let bh = bh as i64;
    if bw == 0 || bh == 0 {
        return;
    }
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let cols = iw.ceil() as i64;
    let rows = ih.ceil() as i64;
    for row in 0..rows {
        let fy = y0 + row;
        if fy < 0 || fy >= fbuf.height as i64 {
            continue;
        }
        let vy = if ih > 1.0 { row as f32 / (ih - 1.0) } else { 0.0 };
        for col in 0..cols {
            let fx = x0 + col;
            if fx < 0 || fx >= fbuf.width as i64 {
                continue;
            }
            let vx = if iw > 1.0 { col as f32 / (iw - 1.0) } else { 0.0 };
            let bx = ((vx * (bw - 1) as f32).round() as i64).clamp(0, bw - 1);
            let by = ((vy * (bh - 1) as f32).round() as i64).clamp(0, bh - 1);
            let bi = ((by * bw + bx) * 4) as usize;
            if bi + 3 >= data.len() {
                continue;
            }
            let src_rgba = [
                data[bi] as f32 / 255.0,
                data[bi + 1] as f32 / 255.0,
                data[bi + 2] as f32 / 255.0,
                data[bi + 3] as f32 / 255.0,
            ];
            let sa = src_rgba[3] * alpha;
            if sa <= 0.0 {
                continue;
            }
            let c = [src_rgba[0], src_rgba[1], src_rgba[2], sa];
            let dst = fbuf.get(fx as u32, fy as u32);
            fbuf.set(fx as u32, fy as u32, prism_raster::blend_pixel_over(c, dst));
        }
    }
}

// ---- 矢量图元 ----

/// 构建圆角矩形路径（四角用二次贝塞尔，无需三角函数，确定且平滑）。
fn round_rect_path(pb: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32, radius: f32) {
    let r = radius.max(0.0).min(w * 0.5).min(h * 0.5);
    if r <= 0.0 {
        pb.move_to(Point::new(x, y));
        pb.line_to(Point::new(x + w, y));
        pb.line_to(Point::new(x + w, y + h));
        pb.line_to(Point::new(x, y + h));
        pb.close();
        return;
    }
    pb.move_to(Point::new(x + r, y));
    pb.line_to(Point::new(x + w - r, y));
    pb.quad_to(Point::new(x + w, y), Point::new(x + w, y + r));
    pb.line_to(Point::new(x + w, y + h - r));
    pb.quad_to(Point::new(x + w, y + h), Point::new(x + w - r, y + h));
    pb.line_to(Point::new(x + r, y + h));
    pb.quad_to(Point::new(x, y + h), Point::new(x, y + h - r));
    pb.line_to(Point::new(x, y + r));
    pb.quad_to(Point::new(x, y), Point::new(x + r, y));
    pb.close();
}

/// 构建椭圆路径（四段二次贝塞尔近似，控制常量 0.5523，无需三角函数）。
fn ellipse_path(pb: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32) {
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let rx = (w * 0.5).max(0.0);
    let ry = (h * 0.5).max(0.0);
    if rx <= 0.0 || ry <= 0.0 {
        return;
    }
    let k = 0.552_284_749_8_f32;
    let ox = rx * k;
    let oy = ry * k;
    pb.move_to(Point::new(cx + rx, cy));
    pb.quad_to(Point::new(cx + rx, cy - oy), Point::new(cx + ox, cy - ry));
    pb.quad_to(Point::new(cx - ox, cy - ry), Point::new(cx - rx, cy));
    pb.quad_to(Point::new(cx - rx, cy + oy), Point::new(cx - ox, cy + ry));
    pb.quad_to(Point::new(cx + ox, cy + ry), Point::new(cx + rx, cy));
    pb.close();
}

/// 构建图元路径。
fn shape_path(kind: ShapeKind, x: f32, y: f32, w: f32, h: f32) -> PathBuilder {
    let mut pb = PathBuilder::new();
    match kind {
        ShapeKind::Rect => {
            pb.move_to(Point::new(x, y));
            pb.line_to(Point::new(x + w, y));
            pb.line_to(Point::new(x + w, y + h));
            pb.line_to(Point::new(x, y + h));
            pb.close();
        }
        ShapeKind::RoundRect { radius } => round_rect_path(&mut pb, x, y, w, h, radius),
        ShapeKind::Ellipse => ellipse_path(&mut pb, x, y, w, h),
    }
    pb
}

// ---- 折线路径轨道 ----

/// 两点间欧氏距离（只用 ±×÷，无 libm）。
#[inline]
fn dist(a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt()
}

/// 折线合法性校验（[`PathTrack`] 与 [`DynPathTrack`] 共用）：
/// 顶点数 >= 2 且所有坐标有限（NaN/Inf 会污染光栅几何）。
#[inline]
fn polyline_valid(pts: &[[f32; 2]]) -> bool {
    pts.len() >= 2 && pts.iter().all(|q| q[0].is_finite() && q[1].is_finite())
}

/// 折线光栅核心（[`PathTrack`] 与 [`DynPathTrack`] 共用，从
/// `draw_path_track` 逐句提取，逻辑不变）：
///
/// - `fill = true`：整条路径闭合后走 [`fill_path`]（半透明面积；填充自动
///   闭合，不受 `closed` 影响）；
/// - `fill = false`：按 `width` 逐段 [`draw_line`] 描边；`draw_sec > 0` 时
///   带绘制进度动画——只画到折线总长的
///   `p = clamp((t-at)/draw_sec, 0, 1)` 比例（按累计长度截断，最后一段
///   可截断）；`closed = true` 时末段回到首点（同样参与进度截断）；
/// - `col` 的 alpha 已由调用方乘以全局淡入淡出信封 `env`；`col[3] <= 0`
///   不产出像素。
///
/// 调用方须先通过 [`polyline_valid`] 校验（本函数不再校验）。
/// 确定性：全部输入为 (pts, col, t, at) 的纯函数，无隐藏状态。
fn paint_polyline(
    fbuf: &mut Fbuf16F,
    pts: &[[f32; 2]],
    col: [f32; 4],
    width: f32,
    fill: bool,
    draw_sec: f32,
    closed: bool,
    t: f32,
    at: f32,
) {
    if col[3] <= 0.0 {
        return;
    }
    let pts: Vec<Point> = pts.iter().map(|q| Point::new(q[0], q[1])).collect();

    if fill {
        // 填充模式：闭合整条路径，fill_path 内部自动闭合轮廓。
        let mut pb = PathBuilder::new();
        pb.move_to(pts[0]);
        for q in &pts[1..] {
            pb.line_to(*q);
        }
        pb.close();
        fill_path(fbuf, &pb, &Transform::IDENTITY, col, FillRule::Winding);
        return;
    }

    // 描边模式：先收集全部待画段（closed 时补首尾闭合段），
    // 再按绘制进度 p 以累计长度截断。
    if width <= 0.0 {
        return;
    }
    let mut segs: Vec<(Point, Point)> = Vec::with_capacity(pts.len());
    for w in pts.windows(2) {
        segs.push((w[0], w[1]));
    }
    if closed && pts.len() >= 3 {
        segs.push((pts[pts.len() - 1], pts[0]));
    }

    // 绘制进度（draw_sec <= 0 → 立即全显）。
    let progress = if draw_sec > 0.0 {
        ((t - at) / draw_sec).clamp(0.0, 1.0)
    } else {
        1.0
    };
    if progress <= 0.0 {
        return;
    }
    // 总长（跳过零长段，避免除零；全零长时直接不画）。
    let total_len: f32 = segs.iter().map(|&(a, b)| dist(a, b)).sum();
    if total_len <= 0.0 {
        return;
    }
    let target = total_len * progress;

    let mut acc = 0.0f32;
    for &(a, b) in &segs {
        let seg = dist(a, b);
        if seg <= 0.0 {
            continue;
        }
        if acc + seg <= target {
            draw_line(fbuf, a, b, width, col);
            acc += seg;
        } else {
            // 进度落在本段内：按比例截断到目标点（最后一段可截断）。
            let k = ((target - acc) / seg).clamp(0.0, 1.0);
            let end = Point::new(lerp(a.x, b.x, k), lerp(a.y, b.y, k));
            draw_line(fbuf, a, end, width, col);
            break;
        }
    }
}

/// 绘制折线路径轨道（[`Track::Path`] 的渲染实现）。
///
/// - 非法输入（顶点数 < 2、坐标非有限、`env <= 0`）直接跳过，**不 panic**；
/// - 校验与颜色信封处理后汇入共用光栅核心 [`paint_polyline`]（与
///   [`Track::DynPath`] 逐位相同的光栅代码）；
/// - `fill = true`：整条路径闭合后填充（半透明面积；填充自动闭合，不受
///   [`PathTrack::closed`] 影响）；
/// - `fill = false`：按 [`PathTrack::width`] 逐段 [`draw_line`] 描边；
///   `draw_sec > 0` 时带绘制进度动画——只画到折线总长的
///   `p = clamp((t-at)/draw_sec, 0, 1)` 比例（按累计长度截断，最后一段
///   可截断）；`closed = true` 时末段回到首点（同样参与进度截断）；
/// - 颜色 alpha 已由调用方乘以全局淡入淡出信封 `env`。
///
/// 确定性：全部输入为 (points, t, env) 的纯函数，无隐藏状态。
fn draw_path_track(fbuf: &mut Fbuf16F, p: &PathTrack, t: f32, env: f32) {
    if env <= 0.0 {
        return;
    }
    // 顶点数 / 坐标必须合法（NaN/Inf 会污染光栅几何，直接跳过该轨道）。
    if !polyline_valid(&p.points) {
        return;
    }
    let mut col = p.color;
    col[3] *= env;
    paint_polyline(
        fbuf,
        &p.points,
        col,
        p.width,
        p.fill,
        p.draw_sec,
        p.closed,
        t,
        p.at,
    );
}

/// Morph 轨道在场景时间 `t` 的当前折线（帧内求值）：
/// `u = clamp((t - at) / dur, 0, 1)`，`e = ease(u)`，逐点
/// `lerp(from[i], to[i], e)`。
///
/// - `dur <= 0`：`t >= at` 直接取 `to`（`u = 1`），否则取 `from`（`u = 0`）；
/// - **点数不等时**：按索引配对，只渲染 `min(from.len(), to.len())` 个点，
///   多出的点忽略（规格第 3 条）；
/// - `from` / `to` 任一为空（或交集为空）时返回空 `Vec`，由调用方经
///   [`polyline_valid`] 跳过，不 panic。
///
/// 确定性：(from, to, at, dur, ease, t) 的纯函数，无隐藏状态。
fn morph_points_at(m: &MorphSpec, t: f32) -> Vec<[f32; 2]> {
    let u = if m.dur > 0.0 {
        ((t - m.at) / m.dur).clamp(0.0, 1.0)
    } else if t >= m.at {
        1.0
    } else {
        0.0
    };
    let e = m.ease.apply(u);
    // 点数不等：按索引配对渲染 min(from.len(), to.len()) 个点，多出的忽略。
    let n = m.from.len().min(m.to.len());
    let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        pts.push([
            lerp(m.from[i][0], m.to[i][0], e),
            lerp(m.from[i][1], m.to[i][1], e),
        ]);
    }
    pts
}

/// 绘制时间参数化折线变形轨道（[`Track::Morph`] 的渲染实现）：
/// 每帧经 [`morph_points_at`] 求出当前折线后，进入与 [`draw_path_track`]
/// **完全相同的校验与光栅核心**（[`polyline_valid`] + [`paint_polyline`]）
/// ——描边 / 填充 / `closed` / `draw_sec` 绘制进度 / fade 淡入淡出一套
/// 代码，零新光栅逻辑。
///
/// 违约跳过（不 panic）： morph 后点数 < 2 / 坐标非有限 / `env <= 0` /
/// 颜色 alpha 归零。
fn draw_morph(fbuf: &mut Fbuf16F, m: &MorphSpec, t: f32, env: f32) {
    if env <= 0.0 {
        return;
    }
    let pts = morph_points_at(m, t);
    if !polyline_valid(&pts) {
        return;
    }
    let mut col = m.color;
    col[3] *= env;
    paint_polyline(
        fbuf,
        &pts,
        col,
        m.width,
        m.fill,
        m.draw_sec,
        m.closed,
        t,
        m.at,
    );
}

/// 绘制时间参数化折线轨道（[`Track::DynPath`] 的渲染实现）：
/// 每帧求值 `geom(t)`（场景绝对时间）后，经与 [`draw_path_track`] 完全
/// 相同的校验与光栅核心（[`polyline_valid`] + [`paint_polyline`]）渲染。
///
/// 违约跳过（不 panic）：`geom(t)` 点数 < 2 / 坐标非有限 / `env <= 0` /
/// 颜色 alpha 归零。
fn draw_dynpath(fbuf: &mut Fbuf16F, d: &DynPathTrack, t: f32, env: f32) {
    if env <= 0.0 {
        return;
    }
    let pts = (d.geom)(t); // 场景绝对时间（非 t - at）
    if !polyline_valid(&pts) {
        return;
    }
    let mut col = d.color;
    col[3] *= env;
    paint_polyline(
        fbuf,
        &pts,
        col,
        d.width,
        d.fill,
        d.draw_sec,
        d.closed,
        t,
        d.at,
    );
}

/// 计数器读数（E2）：`from + (to - from) * easing(u)`，其中
/// `u = clamp((t - at) / dur_sec, 0, 1)`（`dur_sec <= 0` 时
/// `u = (t >= at) ? 1 : 0`）。
///
/// f∘ease：[`CounterText::value_at`] 为 `Some(f)` 时读数改写为
/// `f(t)`（场景绝对秒，与 [`DynPathTrack`] 几何同一时钟）——动点沿曲线
/// `f(x(t))` 复合运动时，HUD 与几何共享同一纯函数，逐位恒等；此时
/// `from`/`to`/`dur_sec`/`easing` 不参与求值（设计包络备案）。
///
/// `-0.0` 归一为 `+0.0`（防 `"-0.0000"`）。纯函数，确定性。
pub(crate) fn counter_value(c: &CounterText, t: f32) -> f32 {
    let v = if let Some(f) = &c.value_at {
        f(t)
    } else {
        let u = if c.dur_sec > 0.0 {
            ((t - c.at) / c.dur_sec).clamp(0.0, 1.0)
        } else if t >= c.at {
            1.0
        } else {
            0.0
        };
        c.from + (c.to - c.from) * c.easing.apply(u)
    };
    if v == 0.0 {
        0.0 // 归一 -0.0 → +0.0（`-0.0 == 0.0` 为 true）
    } else {
        v
    }
}

/// 计数器文本（E2）：`prefix + 定点小数`（`{:.N}` 格式，无科学计数）；
/// `decimals > 6` 钳 6（`f32` 定点有效位上限，防病态长串）。纯函数。
///
/// 负零归一（缺陷修复）：`counter_value` 只把**精确** `-0.0` 翻正，
/// 而 `(-0.00005, 0)` 区间的值经 `{:.4}` 舍入后仍产出 `"-0.0000"`——数值
/// 为零却带一根负号短横（HUD 过零帧的视觉跳变源之一）。此处按格式化
/// 结果裁负号：仅当串以 `'-'` 开头且其余字符全为 `'0'` / `'.'` 时剥掉，
/// 不影响任何非零读数（含 `-0.0001` 等真负值）。
pub(crate) fn counter_text(c: &CounterText, t: f32) -> String {
    let decimals = c.decimals.min(6) as usize;
    let body = format!("{:.*}", decimals, counter_value(c, t));
    let body = if body.starts_with('-') && body[1..].bytes().all(|b| b == b'0' || b == b'.') {
        body[1..].to_string()
    } else {
        body
    };
    format!("{}{}", c.prefix, body)
}

/// 绘制实时数值读数轨道（[`Track::Counter`] 的渲染实现）：
/// 逐帧格式化 ASCII 文本后完全复用 [`draw_static`] 文本管线
/// （`layout_text` 排版 + `GlyphCache` 位图缓存 + `draw_glyphs` 合成），
/// 零新光栅代码、零新字体风险。
///
/// `center_x` 为调用方解析后的水平锚点（`None` 已解析为画幅中心）。
fn draw_counter(
    fbuf: &mut Fbuf16F,
    c: &CounterText,
    t: f32,
    env: f32,
    font: &Font,
    center_x: f32,
    cache: &mut GlyphCache,
) {
    if env <= 0.0 {
        return;
    }
    let text = counter_text(c, t);
    let mut col = c.color;
    col[3] *= env;
    draw_static(
        fbuf,
        font,
        &text,
        c.size,
        col,
        center_x,
        c.y,
        c.letter_spacing,
        cache,
    );
}

/// 绘制帧序列轨道（[`Track::Sequence`] 的渲染实现）：
/// 取帧后经共用 blit 核心 [`blit_rgba8`] 以画幅全幅合成（源尺寸必须等于
/// 画幅）。
///
/// 取帧数学语义：`idx = clamp(floor((t - at) * src_fps), 0, len-1)`，其中
/// `t = fi / fps`（`fi` 为画布帧索引）。**实现用帧索引算术**而非先算
/// `t` 再乘：`1/3` 类分数在二进制浮点不可精确表示，`floor(fl(1/3)·15)`
/// 在 f32（乃至 f64）下都得 4 而非 5，会令「src_fps=15 @ 画布 30fps 每源帧
/// 恰显 2 画布帧」的不变量在边界帧抖动（某源帧显 3 帧、相邻源帧显 1 帧）。
/// 改算 `(fi - at·fps)·src_fps / fps`：`at = 0` 时全整数路径精确无误
/// （S7 配置），一般 `at` 下相对误差 ~1e-15，边界帧取舍无视觉差异。
///
/// 违约跳过（不 panic）：`env <= 0` / `src_fps <= 0` / `frames` 空 /
/// 任一帧字节数 ≠ `width·height·4` / 源尺寸 ≠ 画幅。
fn draw_sequence(
    fbuf: &mut Fbuf16F,
    q: &SequenceTrack,
    fi: usize,
    fps: u32,
    env: f32,
    canvas_w: u32,
    canvas_h: u32,
) {
    if env <= 0.0 || q.src_fps == 0 || q.frames.is_empty() || fps == 0 {
        return;
    }
    if q.width != canvas_w || q.height != canvas_h {
        return;
    }
    // 逐帧长度校验：任一帧字节数不符即整轨跳过（数据违约不 panic）。
    let frame_len = (q.width as usize) * (q.height as usize) * 4;
    if !q.frames.iter().all(|f| f.len() == frame_len) {
        return;
    }
    // 取帧：src_fps=15 @ 画布 30fps ⇒ 每源帧连续显示 2 个画布帧
    // （floor(fi/2) 在源帧区间恒定；155 源帧铺满 310 画布帧）。
    let x = (fi as f64 - q.at as f64 * fps as f64) * q.src_fps as f64 / fps as f64;
    let fidx = x.floor();
    // NaN 防御（at 非有限时 saturating cast 得 0，落到首帧）。
    let idx = (fidx as i64).clamp(0, q.frames.len() as i64 - 1) as usize;
    blit_rgba8(
        fbuf,
        &q.frames[idx],
        q.width,
        q.height,
        0.0,
        0.0,
        canvas_w as f32,
        canvas_h as f32,
        env,
    );
}

// ---- Clip 级（片头 / 片尾）淡入淡出 ----

/// 计算 Clip 级淡入淡出因子（0..1）。
///
/// - 片头：`smooth(t / fade_in)`（`fade_in <= 0` 视为 1.0，不淡入）；
/// - 片尾：`1 - smooth((t - (duration - fade_out)) / fade_out)`
///   （`fade_out <= 0` 视为 1.0，不淡出）；
/// - 中间稳态为 1.0。
///
/// `fps` 仅用于防御性校验（`fps <= 0` 或非有限时视为无淡变）；`t` 本身
/// 已是秒。纯函数，确定性。
pub(crate) fn clip_fade_factor(
    t: f32,
    fps: f32,
    duration: f32,
    fade_in: f32,
    fade_out: f32,
) -> f32 {
    if !(fps > 0.0) || !duration.is_finite() || !t.is_finite() {
        return 1.0;
    }
    let mut f = 1.0;
    if fade_in > 0.0 {
        f *= smooth((t / fade_in).clamp(0.0, 1.0));
    }
    if fade_out > 0.0 {
        let out_start = (duration - fade_out).max(0.0);
        f *= 1.0 - smooth(((t - out_start) / fade_out).clamp(0.0, 1.0));
    }
    f
}

/// 对整段帧序列施加 Clip 级淡入淡出：逐帧把 RGB 三通道乘以该帧的
/// [`clip_fade_factor`]（alpha 通道保持 255；结果 clamp 0..255）。
///
/// 纯后处理——不触碰任何渲染 / 编码路径，因此对既有入口零影响。
pub(crate) fn apply_clip_fade(
    frames: &mut [Vec<u8>],
    fps: f32,
    duration: f32,
    fade_in: f32,
    fade_out: f32,
) {
    for (fi, frame) in frames.iter_mut().enumerate() {
        let f = clip_fade_factor(fi as f32 / fps, fps, duration, fade_in, fade_out);
        if f >= 1.0 {
            continue; // 稳态帧零成本跳过（逐字节不变）
        }
        for px in frame.chunks_exact_mut(4) {
            px[0] = (px[0] as f32 * f).clamp(0.0, 255.0) as u8;
            px[1] = (px[1] as f32 * f).clamp(0.0, 255.0) as u8;
            px[2] = (px[2] as f32 * f).clamp(0.0, 255.0) as u8;
            // px[3]（alpha）保持 255 不动。
        }
    }
}

// ---- 多场景拼接：转场（纯后处理）----

/// 擦除过渡带半宽（px）：值已下沉 `prism_core::sceneseq::WIPE_BAND`（单点
/// 定义），本常量是 author 侧既有引用的 **re-export**，防止破坏旧调用点。
const WIPE_BAND: u16 = prism_core::sceneseq::WIPE_BAND;

/// 位置阈值混合权重（擦除用）已下沉 core：[`prism_core::sceneseq::wipe_weight`]
/// （整数定点实现，无浮点 ⇒ 跨端逐位一致）。此处不再保留 f32 版本。

/// 混合单帧：按 `transition` 把 `fa`（旧片段末帧）与 `fb`（新片段首帧）
/// 按进度 `k`（0..1，0 = 纯 a、1 = 纯 b）混合为一帧 RGBA8。
///
/// - **Fade**：RGB 逐通道 `lerp`，alpha 置 255，clamp 0..255；
/// - **Wipe{dir}**：按方向以位置阈值逐像素选源（带 2px 平滑过渡带）：
///   - `Right`：`x < (1-k)*W` 显示 a、否则 b（b 自右侧入场，界线左移）；
///   - `Left`：镜像（b 自左侧入场，界线右移）；
///   - `Up`：`y < (1-k)*H` 显示 a、否则 b（b 自底部入场，界线上移）；
///   - `Down`：镜像（b 自顶部入场，界线下移）。
/// - **Dissolve{seed}**：逐像素 `hash2(seed, x, y) < k` 显影 b、
///   否则保留 a（语义对齐 `prism-postfx::effects::Dissolve` 的阈值显影，
///   同款 `prism_noise::pcg::hash2_f32`）——像素级非 A 即 B，中点帧是
///   A/B 并存的颗粒过渡；端点精确（`k=0` 无像素显影 ⇒ 逐位 a，
///   `k=1` 全显影 ⇒ 逐位 b，哈希值域 [0,1) 保证）；
/// - **Push{dir}**：两帧作为同一画面整体平移——b 自 `dir` 侧
///   入场、a 从对侧退出（`dir=Left`：`x < k·W` 显示 b 的左移采样、
///   否则显示 a 的右移采样；`Right`/`Up`/`Down` 镜像）。硬接缝无平滑带
///   （端点逐位精确由构造保证：`k=0` 全 a、`k=1` 全 b）；越界采样钳
///   到边缘像素（repeat，不引入新颜色）。
///
/// 尺寸防御：任一帧字节数与 `width*height*4` 不符时退化为返回 `fa` 的拷贝
/// （确定性、不 panic）。
fn mix_frame(
    fa: &[u8],
    fb: &[u8],
    width: u32,
    height: u32,
    transition: Transition,
    k: f32,
) -> Vec<u8> {
    let expected = (width as usize) * (height as usize) * 4;
    if fa.len() != expected || fb.len() != expected {
        return fa.to_vec();
    }
    let k = k.clamp(0.0, 1.0);
    // f32 进度 → 256 分母整数网格（round-to-nearest）：`kw ∈ 0..=256`。
    // 之后所有路径只用 `kw`（整数），不再碰 `k`（零浮点确定性）。
    let kw: u16 = (k * 256.0).round() as u16;
    let mut out = vec![0u8; expected];
    match transition {
        Transition::Fade => {
            // 全局线性权重 ⇒ 单权重 `mix_u8`（round-to-nearest、无系统偏置）。
            for (o, (pa, pb)) in out
                .chunks_exact_mut(4)
                .zip(fa.chunks_exact(4).zip(fb.chunks_exact(4)))
            {
                o[0] = mix_u8(pa[0], pb[0], kw);
                o[1] = mix_u8(pa[1], pb[1], kw);
                o[2] = mix_u8(pa[2], pb[2], kw);
                o[3] = 255;
            }
        }
        Transition::Wipe { dir } => {
            // 分界行程扩展（ADR-18 证 4）：边界自 `extent+band` 走到 `−band`，
            // 两端混合带完全出画 ⇒ k=0 / k=1 端点逐位精确（污染列 = 0）。
            // 镜像方向对权重取补（`256 − w`），与 `WipeDir` 的入场侧语义对齐。
            let band: u16 = WIPE_BAND;
            let mut idx = 0usize;
            for y in 0..height {
                for x in 0..width {
                    // 逐像素混合权重：0 = 取 a、256 = 取 b（带内 smoothstep）。
                    let t: u16 = match dir {
                        // core 规范形态：边界自 `extent+band` 左移 ⇒ b 自右侧
                        // 入场、分界线向左推进（= `WipeDir::Right`）。
                        WipeDir::Right => wipe_weight(x, kw, width, band),
                        WipeDir::Left => W_ONE - wipe_weight(x, kw, width, band),
                        WipeDir::Up => wipe_weight(y, kw, height, band),
                        WipeDir::Down => W_ONE - wipe_weight(y, kw, height, band),
                    };
                    out[idx] = mix_u8(fa[idx], fb[idx], t);
                    out[idx + 1] = mix_u8(fa[idx + 1], fb[idx + 1], t);
                    out[idx + 2] = mix_u8(fa[idx + 2], fb[idx + 2], t);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
        Transition::Dissolve { seed } => {
            // 逐像素整数阈值显影：hash ∈ 0..=255 < kw ⇒ 取 b，否则取 a。
            // 端点精确：kw=0 恒假（全 a）、kw=256 恒真（全 b）。hash 为
            // splitmix64 类纯整数（宪章：转场路径禁 f32 rand），同 seed 同图案。
            let mut idx = 0usize;
            for y in 0..height {
                for x in 0..width {
                    let h: u16 = dissolve_hash(x, y, seed);
                    let t: u16 = if h < kw { W_ONE } else { 0 };
                    out[idx] = mix_u8(fa[idx], fb[idx], t);
                    out[idx + 1] = mix_u8(fa[idx + 1], fb[idx + 1], t);
                    out[idx + 2] = mix_u8(fa[idx + 2], fb[idx + 2], t);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
        Transition::Push { dir } => {
            // 整体平移：b 自 `dir` 侧入场、a 从对侧退出。推进量整数化
            // （`push_c` = round(k·extent)，256 分母网格），越界采样钳到边缘
            // 像素（repeat，不引入新颜色）；权重恒为 0 / 256 ⇒ `mix_u8`
            // 逐位等价于整像素拷贝（端点精确由构造保证）。
            let w = width as i64;
            let h = height as i64;
            let cw: i64 = push_c(kw, width) as i64; // 水平推进量
            let ch: i64 = push_c(kw, height) as i64; // 垂直推进量
            let mut idx = 0usize;
            for y in 0..height {
                let (ya, yb) = match dir {
                    WipeDir::Up => (
                        (y as i64 - ch).clamp(0, h - 1),
                        (y as i64 + (h - ch)).clamp(0, h - 1),
                    ),
                    WipeDir::Down => (
                        (y as i64 + ch).clamp(0, h - 1),
                        (y as i64 - (h - ch)).clamp(0, h - 1),
                    ),
                    _ => (y as i64, y as i64),
                };
                for x in 0..width {
                    let (xa, xb) = match dir {
                        WipeDir::Left => (
                            (x as i64 - cw).clamp(0, w - 1),
                            (x as i64 + (w - cw)).clamp(0, w - 1),
                        ),
                        WipeDir::Right => (
                            (x as i64 + cw).clamp(0, w - 1),
                            (x as i64 - (w - cw)).clamp(0, w - 1),
                        ),
                        _ => (x as i64, x as i64),
                    };
                    // 覆盖判据（沿推进轴）：b 占入场侧、a 占对侧。
                    let from_b: bool = match dir {
                        WipeDir::Left => (x as i64) < cw,
                        WipeDir::Right => (x as i64) >= w - cw,
                        WipeDir::Up => (y as i64) < ch,
                        WipeDir::Down => (y as i64) >= h - ch,
                    };
                    let t: u16 = if from_b { W_ONE } else { 0 };
                    let ia = ((ya as u32) * width + xa as u32) as usize * 4;
                    let ib = ((yb as u32) * width + xb as u32) as usize * 4;
                    out[idx] = mix_u8(fa[ia], fb[ib], t);
                    out[idx + 1] = mix_u8(fa[ia + 1], fb[ib + 1], t);
                    out[idx + 2] = mix_u8(fa[ia + 2], fb[ib + 2], t);
                    out[idx + 3] = 255;
                    idx += 4;
                }
            }
        }
    }
    out
}

/// 把两段帧序列按指定转场拼接为一段（**纯后处理**、确定性纯函数）。
///
/// - 重叠帧数 `trans_frames = (trans_sec * fps).round().clamp(1, min(a.len(),
///   b.len()))`；`trans_sec <= 0`（或 fps 非法、任一段为空导致无法重叠）时
///   直接拼接（无重叠）；
/// - 输出 = `a[0 .. a.len()-trans_frames]` + 重叠混合区（`a` 末
///   `trans_frames` 帧与 `b` 首 `trans_frames` 帧逐帧混合，进度
///   `k = (i+1)/(trans_frames+1)`）+ `b[trans_frames ..]`；
///   总帧数 = `a.len() + b.len() - trans_frames`；
/// - 混合规则见 [`mix_frame`]（Fade 交叉溶解 / Wipe 线性擦除）；
/// - 空序列 / 尺寸不匹配不 panic：重叠帧退化为取 `a` 的帧。
pub(crate) fn concat_with_transition(
    a: &[Vec<u8>],
    b: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u32,
    transition: Transition,
    trans_sec: f32,
) -> Vec<Vec<u8>> {
    // 重叠帧数：trans_sec <= 0 或非有限、fps 非法 → 0（直接拼接）；
    // 否则 clamp 到 [1, min(a.len(), b.len())]（任一段为空 → 0）。
    let trans_frames = if trans_sec > 0.0 && trans_sec.is_finite() && fps > 0 {
        let raw = (trans_sec * fps as f32).round();
        let max_frames = a.len().min(b.len());
        if max_frames == 0 || !raw.is_finite() {
            0
        } else {
            (raw.max(1.0) as usize).min(max_frames)
        }
    } else {
        0
    };

    // trans_frames <= min(a.len(), b.len()) ⇒ 容量恒 >= 0。
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(a.len() + b.len() - trans_frames);
    // 1) a 的头部（不含重叠区）。
    out.extend_from_slice(&a[..a.len() - trans_frames]);
    // 2) 重叠混合区：a 末 trans_frames 帧 × b 首 trans_frames 帧。
    for (i, fa) in a[a.len() - trans_frames..].iter().enumerate() {
        let k = (i + 1) as f32 / (trans_frames + 1) as f32;
        out.push(mix_frame(fa, &b[i], width, height, transition, k));
    }
    // 3) b 的尾部（不含重叠区）。
    out.extend_from_slice(&b[trans_frames..]);
    out
}

/// 内部：把多支子短片规格**从左到右折叠拼接**为一段帧序列（纯后处理）。
///
/// - 校验：`specs.len() >= 2`；所有 specs 须同 `width` / `height` / `fps`
///   （不一致返回 `Err`；JPEG 质量可不同，统一取第 0 支的）；
/// - 相邻第 `i` 对（`specs[i]`, `specs[i+1]`）用 `transitions[i]`，
///   越界默认 [`Transition::Fade`]；
/// - 逐支 `render_rgba_frames` 后折叠 `concat_with_transition`。
fn concat_specs_frames(
    specs: &[ClipSpec],
    transitions: &[Transition],
    trans_sec: f32,
) -> Result<Vec<Vec<u8>>, String> {
    if specs.len() < 2 {
        return Err(format!(
            "render_short_concat: 至少需要 2 支子 ClipSpec，收到 {}",
            specs.len()
        ));
    }
    let first = &specs[0];
    for (i, s) in specs.iter().enumerate().skip(1) {
        if s.width != first.width || s.height != first.height || s.fps != first.fps {
            return Err(format!(
                "render_short_concat: 第 {i} 支子片 {}×{} @{}fps 与第 0 支 {}×{} @{}fps 不一致（拼接要求画幅/帧率一致）",
                s.width, s.height, s.fps, first.width, first.height, first.fps
            ));
        }
    }

    let mut acc = render_rgba_frames(first);
    for i in 1..specs.len() {
        let frames = render_rgba_frames(&specs[i]);
        // 相邻第 i 对用 transitions[i-1]；越界默认 Fade。
        let tr = transitions.get(i - 1).copied().unwrap_or(Transition::Fade);
        acc = concat_with_transition(
            &acc,
            &frames,
            first.width,
            first.height,
            first.fps,
            tr,
            trans_sec,
        );
    }
    Ok(acc)
}

/// 把多支子短片按转场拼接为一支 **MP4**（MJPEG 4:4:4，官方 muxer）。
///
/// - `specs.len() >= 2`；相邻第 `i` 对用 `transitions[i]`（越界默认
///   [`Transition::Fade`]）；各 specs 须同 `width` / `height` / `fps`
///   （`jpeg_quality` 可不同，统一取第 0 支的）；
/// - 实现为**纯后处理**：逐支 `render_rgba_frames` 后在帧序列层折叠
///   [`concat_with_transition`]，复用 [`render_short`] 的同一封装路径；
/// - 参数非法（specs < 2 或画幅 / 帧率不一致）时 panic（与既有
///   [`render_short`] 的 muxer `expect` 风格一致）；需要可恢复错误的调用方
///   请改用 [`render_short_concat_avi`]。
pub fn render_short_concat(
    specs: &[ClipSpec],
    transitions: &[Transition],
    trans_sec: f32,
) -> Vec<u8> {
    let frames = concat_specs_frames(specs, transitions, trans_sec)
        .expect("render_short_concat: 参数非法（specs < 2 或画幅/帧率不一致）");
    mux_mp4(&specs[0], &frames)
}

/// 把多支子短片按转场拼接为一支 **MJPEG-in-AVI**（标准 RIFF 容器）。
///
/// 与 [`render_short_concat`] 同一拼接语义，但走 `avi::write_avi_mjpeg`
/// 封装并以 `Result` 返回参数 / 封装错误（不 panic）。
pub fn render_short_concat_avi(
    specs: &[ClipSpec],
    transitions: &[Transition],
    trans_sec: f32,
) -> Result<Vec<u8>, String> {
    let frames = concat_specs_frames(specs, transitions, trans_sec)?;
    mux_avi(&specs[0], &frames)
}

// ---- 多场景 + 转场 + Clip 级淡入淡出：公开帧序列入口 ----

/// [`render_short_concat_faded_frames`] 使用的场景间转场默认时长（秒）。
///
/// 相邻场景一律 [`Transition::Dissolve`] 溶解 0.5s（任务 3：由 Fade 升级为
/// 溶解，保留 0.5s 时长；与 60s 微积分短片分镜规格一致）；需要自定义转场
/// 类型 / 时长的调用方请改用 [`render_short_concat`] / [`render_short_concat_avi`]。
const CONCAT_FADED_TRANS_SEC: f32 = 0.5;

/// 场景间转场溶解种子（任务 3）：固定常量 ⇒ 确定性（同输入逐字节一致）。
/// 与 [`render_rgba_frames`] 背景噪声种子区分，仅用于 `Dissolve` 显影阈值。
const CONCAT_DISSOLVE_SEED: u64 = 0x6A09_E667;

/// 把多支子短片按转场拼接、并施加 Clip 级片头 / 片尾淡入淡出后，返回
/// **裸 RGBA8 帧序列**（每帧 `width×height×4` 字节，行主序）。
///
/// 这是 example / 外部消费方获取帧序列的唯一公开通道（内部
/// [`render_rgba_frames`] 为 `pub(crate)`）。语义：
/// 1. 逐支 [`render_rgba_frames`] → [`concat_with_transition`]（相邻场景
///    一律 [`Transition::Dissolve`]、时长 [`CONCAT_FADED_TRANS_SEC`]）；
/// 2. [`apply_clip_fade`]：片头 `fade_in_sec` / 片尾 `fade_out_sec`
///    （`<= 0` 视为对应方向不淡变；片尾淡变锚定**整片**末帧）；
/// 3. 返回帧序列（不封装容器）。
///
/// 校验与 panic 风格与 [`render_short_concat`] 一致：`specs.len() >= 2`、
/// 各支同 `width` / `height` / `fps`，违约 panic。给定同一 `specs` 必得
/// 逐字节一致的帧序列（"黄金"确定性）。
pub fn render_short_concat_faded_frames(
    specs: &[ClipSpec],
    fade_in_sec: f32,
    fade_out_sec: f32,
) -> Vec<Vec<u8>> {
    // 相邻场景一律 Dissolve（任务 3：由 Fade 升级）；长度 = 场景数 - 1。
    let transitions = vec![Transition::Dissolve { seed: CONCAT_DISSOLVE_SEED }; specs.len().saturating_sub(1)];
    let mut frames = concat_specs_frames(specs, &transitions, CONCAT_FADED_TRANS_SEC)
        .expect("render_short_concat_faded_frames: 参数非法（specs < 2 或画幅/帧率不一致）");
    // 片尾淡变以整片实际时长（总帧数 / 帧率）为锚，而非单支子片时长。
    let fps = specs[0].fps as f32;
    let total_duration = frames.len() as f32 / fps;
    apply_clip_fade(&mut frames, fps, total_duration, fade_in_sec, fade_out_sec);
    frames
}

/// 转场重叠帧数（与 [`concat_with_transition`] 内部公式逐字一致）：
/// `trans_sec <= 0` / 非有限 / `fps <= 0` → 0（直接拼接）；否则
/// `round(trans_sec·fps)` clamp 到 `[1, min(a_len, b_len)]`。
fn trans_frames_count(trans_sec: f32, fps: u32, a_len: usize, b_len: usize) -> usize {
    if trans_sec > 0.0 && trans_sec.is_finite() && fps > 0 {
        let raw = (trans_sec * fps as f32).round();
        let max_frames = a_len.min(b_len);
        if max_frames == 0 || !raw.is_finite() {
            0
        } else {
            (raw.max(1.0) as usize).min(max_frames)
        }
    } else {
        0
    }
}

/// 单段子短片的帧数（与 [`render_rgba_frames`] 内部 `frame_count` 公式
/// 逐字一致，用于流式管线的帧总数预算）。
#[inline]
fn clip_frame_count(spec: &ClipSpec) -> usize {
    (spec.fps as f32 * spec.duration).ceil().max(1.0) as usize
}

/// [`render_short_concat_faded_frames`] 的**流式变体**：逐帧回调产出，
/// 不驻留全片帧序列。
///
/// 输出序列与 [`render_short_concat_faded_frames`] **逐字节一致**（同
/// `specs` / 淡变参数 ⇒ 同帧序列，回调顺序即片上顺序），差别仅在内存
/// 画像：原入口一次性驻留全部帧（90s 960×540 全片 ≈5.6GB，超出常见
/// 容器内存上限），本入口峰值仅为「当前段帧序列 + 转场驻留尾帧
/// （≤ `ceil(trans_sec·fps)` 帧）+ 调用方自身驻留」。
///
/// 实现与折叠式拼接（[`concat_specs_frames`] +
/// [`concat_with_transition`] + [`apply_clip_fade`]）位级对齐：
/// - 转场帧数 `t_i` 与驻留上界 `T = max(t_i)` 全部按帧数预算**预计算**
///   （与 fold 递推同式），主循环只做「输出 + 混合 + 驻留」；
/// - 首段驻留尾 `T` 帧；后续段先输出驻留尾安全区 `[0, T−t)`，再按
///   `k = (j+1)/(t+1)` 与本段首帧 `mix_frame`（Dissolve）产出重叠区，最后
///   驻留本段尾 `T` 帧；末段驻留尾整段产出；
/// - 每帧按全局帧索引施加与 [`apply_clip_fade`] 同公式的 Clip 级
///   淡入淡出（稳态帧零拷贝直投）。
///
/// **段长约束**：折叠式左结合语义下，若某段帧数 `< 2×T`，后续转场的
/// 混合源会回溯到更早段落（`concat_with_transition` 的 clamp 允许
/// `t ≤ min(acc_len, b_len)`，而 `acc_len` 是**累积**长度）——流式管线
/// 在数学上无法以有限驻留复现该回溯。因此本入口要求**每段帧数
/// ≥ 2×T**（`T = max(转场帧数)`，即约 `2×ceil(trans_sec·fps)`；
/// 转场为 0 时 `T = 0` 无约束），违反时返回 `Err`。常规分镜（段长远
/// 大于转场，如 90s 成片最短段 248 帧 vs 转场 15 帧）天然满足；
/// 确需短段拼接请改用批式入口 [`render_short_concat_faded_frames`]。
///
/// 其余参数校验口径与原入口相同（`specs.len() >= 2`、各支同画幅 /
/// 帧率），但以 `Err` 返回而非 panic。
pub fn render_short_concat_faded_frames_each(
    specs: &[ClipSpec],
    fade_in_sec: f32,
    fade_out_sec: f32,
    mut on_frame: impl FnMut(&[u8]),
) -> Result<(), String> {
    if specs.len() < 2 {
        return Err(format!(
            "render_short_concat_faded_frames_each: 至少需要 2 支子 ClipSpec，收到 {}",
            specs.len()
        ));
    }
    let first = &specs[0];
    for (i, s) in specs.iter().enumerate().skip(1) {
        if s.width != first.width || s.height != first.height || s.fps != first.fps {
            return Err(format!(
                "render_short_concat_faded_frames_each: 第 {i} 支子片 {}×{} @{}fps 与第 0 支 {}×{} @{}fps 不一致（拼接要求画幅/帧率一致）",
                s.width, s.height, s.fps, first.width, first.height, first.fps
            ));
        }
    }

    let trans_sec = CONCAT_FADED_TRANS_SEC;
    let fps = first.fps;
    let width = first.width;
    let height = first.height;

    // ---- 帧数预算：转场帧数 ts[i]（对 (acc_i, seg_{i+1})）与驻留上界 T ----
    // 递推与折叠式 concat_specs_frames 同式：acc_0 = L_0；
    // acc_{i+1} = acc_i + L_{i+1} − ts[i]，ts[i] = clamp(round(trans·fps),
    // 1, min(acc_i, L_{i+1}))。
    let seg_lens: Vec<usize> = specs.iter().map(clip_frame_count).collect();
    let mut ts: Vec<usize> = Vec::with_capacity(specs.len() - 1);
    let mut acc_len = seg_lens[0];
    for i in 1..specs.len() {
        let t = trans_frames_count(trans_sec, fps, acc_len, seg_lens[i]);
        ts.push(t);
        acc_len += seg_lens[i] - t;
    }
    let total_frames = acc_len;
    let t_max = ts.iter().copied().max().unwrap_or(0);

    // 段长约束：每段 ≥ 2×T（保证本段扣除混合帧后仍能提供 T 帧驻留，
    // 混合源不回溯更早段落）。T = 0（转场关闭）时无约束。
    if t_max > 0 {
        for (i, &l) in seg_lens.iter().enumerate() {
            if l < 2 * t_max {
                return Err(format!(
                    "render_short_concat_faded_frames_each: 第 {i} 支子片仅 {l} 帧 < 2×转场帧数 {}——流式拼接无法复现折叠语义的混合源回溯，请加长该子片或改用 render_short_concat_faded_frames",
                    2 * t_max
                ));
            }
        }
    }

    let fps_f = fps as f32;
    let duration = total_frames as f32 / fps_f;

    // 单帧产出：Clip 级淡入淡出与 apply_clip_fade 同公式（稳态帧零拷贝）。
    // emitted（全局帧索引）经参数传入，避免与 on_frame 的捕获冲突。
    let mut emit = |frame: &[u8], emitted: &mut usize| {
        let f = clip_fade_factor(*emitted as f32 / fps_f, fps_f, duration, fade_in_sec, fade_out_sec);
        if f >= 1.0 {
            on_frame(frame);
        } else {
            let mut buf = frame.to_vec();
            for px in buf.chunks_exact_mut(4) {
                px[0] = (px[0] as f32 * f).clamp(0.0, 255.0) as u8;
                px[1] = (px[1] as f32 * f).clamp(0.0, 255.0) as u8;
                px[2] = (px[2] as f32 * f).clamp(0.0, 255.0) as u8;
                // px[3]（alpha）保持 255 不动（与 apply_clip_fade 一致）。
            }
            on_frame(&buf);
        }
        *emitted += 1;
    };

    let n = specs.len();
    let mut tail: Vec<Vec<u8>> = Vec::new(); // 驻留尾 = 当前 acc 的最后 T 帧
    let mut emitted = 0usize; // 已产出帧数（Clip 淡变全局索引）
    for (i, spec) in specs.iter().enumerate() {
        let frames = render_rgba_frames(spec);
        let len = frames.len();
        if i == 0 {
            // 首段：无前序混合；驻留尾 T 帧（acc_0 = seg0，尾即段尾）。
            for f in &frames[..len - t_max] {
                emit(f, &mut emitted);
            }
            tail = frames[len - t_max..].to_vec();
            continue;
        }
        // 后续段：本对转场帧数 t = ts[i-1]（≤ T = tail.len()）。
        let t = ts[i - 1];
        // 1) 驻留尾安全区（不参与本对混合的部分）。
        for f in &tail[..t_max - t] {
            emit(f, &mut emitted);
        }
        // 2) 重叠混合区：acc 末 t 帧 × 本段首 t 帧，k = (j+1)/(t+1)。
        for j in 0..t {
            let k = (j + 1) as f32 / (t + 1) as f32;
            let mixed = mix_frame(
                &tail[t_max - t + j],
                &frames[j],
                width,
                height,
                Transition::Dissolve { seed: CONCAT_DISSOLVE_SEED },
                k,
            );
            emit(&mixed, &mut emitted);
        }
        // 3) 本段输出区 + 新驻留尾（末段 keep = 0，驻留尾整段输出）。
        let keep = if i + 1 < n { t_max } else { 0 };
        for f in &frames[t..len - keep] {
            emit(f, &mut emitted);
        }
        tail = frames[len - keep..].to_vec();
    }
    Ok(())
}

// ---- 顶层渲染 ----

/// 内部：把一支声明式短片渲染为逐帧 RGBA8 序列。
///
/// 背景绘制、轨道绘制与帧循环全部收在 API 内部；产物为 `Vec<Vec<u8>>`，
/// 每帧 `width×height×4` 字节（行主序 RGBA8），供 MP4 / AVI 两条封装共用。
/// 给定同一 `spec` 必得逐字节一致的帧序列（"黄金"确定性）。
pub(crate) fn render_rgba_frames(spec: &ClipSpec) -> Vec<Vec<u8>> {
    let w = spec.width;
    let h = spec.height;
    let fps = spec.fps;
    let frame_count = (spec.fps as f32 * spec.duration).ceil().max(1.0) as usize;

    // 字体 / 缓存 / 背景噪声——全部在 API 内部处理。
    let bold = font_bold();
    let regular = font_regular();
    let mut cache = GlyphCache::new();
    let fbm = Fbm::new(2, 2.0, 0.5);
    let seed: u64 = 0x9E37_79B1;

    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(frame_count);

    // 帧循环被收进 API 内部。
    for fi in 0..frame_count {
        let t = fi as f32 / fps as f32; // 0 .. duration 秒

        let mut fbuf = Fbuf16F::new(w, h);
        paint_bg(&mut fbuf, t, &fbm, seed, &spec.background);

        for track in &spec.tracks {
            match track {
                Track::Kinetic(k) => {
                    let env = fade_factor(t, k.at, k.in_sec, k.out_sec, k.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let font = match k.font {
                        FontKind::Bold => &bold,
                        FontKind::Regular => &regular,
                    };
                    draw_kinetic(
                        &mut fbuf, font, &k.content, k.size, k.color,
                        k.center_x.unwrap_or(w as f32 * 0.5), k.y,
                        k.stagger, k.per_char, k.scale_from, k.rise_px, k.easing, t - k.at,
                        env, &mut cache,
                    );
                }
                Track::Static(s) => {
                    let env = fade_factor(t, s.at, s.in_sec, s.out_sec, s.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let font = match s.font {
                        FontKind::Bold => &bold,
                        FontKind::Regular => &regular,
                    };
                    let mut col = s.color;
                    col[3] *= env;
                    draw_static(
                        &mut fbuf, font, &s.content, s.size, col,
                        s.center_x.unwrap_or(w as f32 * 0.5), s.y,
                        s.letter_spacing, &mut cache,
                    );
                }
                Track::Typewriter(tw) => {
                    let env = fade_factor(t, tw.at, tw.in_sec, tw.out_sec, tw.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let font = match tw.font {
                        FontKind::Bold => &bold,
                        FontKind::Regular => &regular,
                    };
                    draw_typewriter(
                        &mut fbuf,
                        font,
                        &tw.content,
                        tw.size,
                        tw.color,
                        tw.center_x.unwrap_or(w as f32 * 0.5),
                        tw.y,
                        tw.letter_spacing,
                        tw.cps,
                        tw.cursor,
                        tw.cursor_blink_hz,
                        t - tw.at,
                        t,
                        env,
                        &mut cache,
                    );
                }
                Track::Accent(a) => {
                    let env = fade_factor(t, a.at, a.in_sec, a.out_sec, a.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let mut col = a.color;
                    col[3] *= env;
                    fill_rect(
                        &mut fbuf,
                        a.center_x - a.w * 0.5,
                        a.y,
                        a.w,
                        a.h,
                        col,
                    );
                }
                Track::Caption(c) => {
                    let env = fade_factor(t, c.at, c.in_sec, c.out_sec, c.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let mut style = c.style.clone();
                    style.color[3] *= env;
                    style.bg[3] *= env;
                    render_caption(&mut fbuf, &regular, &c.content, &style, &mut cache);
                }
                Track::Image(img) => {
                    let env = fade_factor(t, img.at, img.in_sec, img.out_sec, img.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let off =
                        image_offset(t, img.at, img.in_sec, img.out_sec, img.out_at, img.move_from, spec.duration);
                    draw_image(
                        &mut fbuf,
                        &img.source,
                        img.x + off[0],
                        img.y + off[1],
                        img.w,
                        img.h,
                        env,
                    );
                }
                Track::Shape(sh) => {
                    let env = fade_factor(t, sh.at, sh.in_sec, sh.out_sec, sh.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let mut col = sh.color;
                    col[3] *= env;
                    let pb = shape_path(sh.kind, sh.x, sh.y, sh.w, sh.h);
                    fill_path(&mut fbuf, &pb, &Transform::IDENTITY, col, FillRule::Winding);
                }
                Track::Path(p) => {
                    let env = fade_factor(t, p.at, p.in_sec, p.out_sec, p.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    draw_path_track(&mut fbuf, p, t, env);
                }
                Track::Morph(m) => {
                    let env = fade_factor(t, m.at, m.in_sec, m.out_sec, m.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    draw_morph(&mut fbuf, m, t, env);
                }
                Track::DynPath(d) => {
                    let env = fade_factor(t, d.at, d.in_sec, d.out_sec, d.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    draw_dynpath(&mut fbuf, d, t, env);
                }
                Track::Counter(c) => {
                    let env = fade_factor(t, c.at, c.in_sec, c.out_sec, c.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    let font = match c.font {
                        FontKind::Bold => &bold,
                        FontKind::Regular => &regular,
                    };
                    let center_x = c.center_x.unwrap_or(w as f32 * 0.5);
                    draw_counter(&mut fbuf, c, t, env, font, center_x, &mut cache);
                }
                Track::Sequence(q) => {
                    let env = fade_factor(t, q.at, q.in_sec, q.out_sec, q.out_at, spec.duration);
                    if env <= 0.0 {
                        continue;
                    }
                    draw_sequence(&mut fbuf, q, fi, fps, env, w, h);
                }
            }
        }

        frames.push(fbuf.to_framebuffer().pixels);
    }

    frames
}

/// 内部：RGBA8 帧序列 → MP4（官方 muxer）。
///
/// [`render_short`] 与 [`render_short_faded`] 共用同一封装路径——两者传入
/// 相同帧序列时产物逐字节一致（[`render_short`] 行为不变）。
fn mux_mp4(spec: &ClipSpec, frames: &[Vec<u8>]) -> Vec<u8> {
    let mux_spec = TrackSpec {
        width: spec.width,
        height: spec.height,
        fps: spec.fps,
    };
    // 官方 muxer 封装（修复后的 size_category 保证 DC 预测链无漂移）。
    write_mp4_jpeg_color(
        frames,
        mux_spec,
        None,
        0,
        0,
        JpegConfig {
            quality: spec.jpeg_quality,
            grayscale: false,
        },
    )
    .expect("MP4 封装失败")
}

/// 内部：RGBA8 帧序列 → MJPEG-in-AVI（[`render_short_avi`] 与
/// [`render_short_avi_faded`] 共用）。
fn mux_avi(spec: &ClipSpec, frames: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    write_avi_mjpeg(frames, spec.width, spec.height, spec.fps, spec.jpeg_quality)
}

/// 把一支声明式短片渲染为 MP4 字节流（MJPEG 4:4:4，官方 muxer，零私有 shim）。
///
/// 复用既有引擎能力（`prism-text` / `prism-noise` / `prism-raster` /
/// `prism-mux` / `prism-io`），不引入任何新第三方 crate，不写私有 JPEG 编码器。
/// 给定同一 `spec` 必得逐字节一致的字节流（"黄金"确定性）。行为与旧版一致。
pub fn render_short(spec: &ClipSpec) -> Vec<u8> {
    let frames = render_rgba_frames(spec);
    mux_mp4(spec, &frames)
}

/// 把一支声明式短片渲染为 **MJPEG-in-AVI** 字节流（标准 RIFF 容器，流类型 MJPG）。
///
/// 复用 [`render_rgba_frames`] 产出的 RGBA8 帧，交由 `prism-mux` 手写标准 AVI
/// 封装——零第三方依赖，可被 VLC / 浏览器 / QuickTime 直接双击播放。
/// 不改动 [`render_short`] 的既有行为。
pub fn render_short_avi(spec: &ClipSpec) -> Result<Vec<u8>, String> {
    let frames = render_rgba_frames(spec);
    mux_avi(spec, &frames)
}

/// 带 **Clip 级片头 / 片尾淡入淡出**的 MP4 入口（新能力，不破坏
/// [`render_short`] 的既有签名与行为）。
///
/// 内部：`render_rgba_frames` → `apply_clip_fade`（逐帧 RGB 乘淡变因子，
/// alpha 恒 255）→ 与 [`render_short`] 完全相同的官方 muxer。
/// `fade_in_sec` / `fade_out_sec <= 0` 视为对应方向不淡变。
pub fn render_short_faded(spec: &ClipSpec, fade_in_sec: f32, fade_out_sec: f32) -> Vec<u8> {
    let mut frames = render_rgba_frames(spec);
    apply_clip_fade(
        &mut frames,
        spec.fps as f32,
        spec.duration,
        fade_in_sec,
        fade_out_sec,
    );
    mux_mp4(spec, &frames)
}

/// 带 **Clip 级片头 / 片尾淡入淡出**的 AVI 入口（新能力，不破坏
/// [`render_short_avi`] 的既有签名与行为）。
///
/// 内部：`render_rgba_frames` → `apply_clip_fade` → `avi::write_avi_mjpeg`。
/// `fade_in_sec` / `fade_out_sec <= 0` 视为对应方向不淡变。
pub fn render_short_avi_faded(
    spec: &ClipSpec,
    fade_in_sec: f32,
    fade_out_sec: f32,
) -> Result<Vec<u8>, String> {
    let mut frames = render_rgba_frames(spec);
    apply_clip_fade(
        &mut frames,
        spec.fps as f32,
        spec.duration,
        fade_in_sec,
        fade_out_sec,
    );
    mux_avi(spec, &frames)
}

// ===========================================================================
// 任务 3 测试：场景间转场由 Fade 升级为 Dissolve（确定性 / 端点位级保真 /
// 中点既非 A 亦非 B）
// ===========================================================================

#[cfg(test)]
mod transition_dissolve_tests {
    use super::*;
    use prism_noise::pcg::hash2_f32;

    /// 构造纯色帧（RGBA8）。
    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut f = vec![0u8; (w * h * 4) as usize];
        for px in f.chunks_exact_mut(4) {
            px[0] = rgb[0];
            px[1] = rgb[1];
            px[2] = rgb[2];
            px[3] = 255;
        }
        f
    }

    const SEED: u64 = 0x6A09_E667;

    /// 确定性：相同输入（同 seed / 同帧 / 同 k）两次调用 `mix_frame`
    /// 输出逐字节一致——溶解显影无随机状态，全片重渲染 MD5 可复现。
    #[test]
    fn dissolve_is_deterministic() {
        let w = 16u32;
        let h = 16u32;
        let fa = solid(w, h, [200, 30, 30]);
        let fb = solid(w, h, [30, 30, 200]);
        for k in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let a = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, k);
            let b = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, k);
            assert_eq!(a, b, "Dissolve 须确定性（k={k}）");
        }
    }

    /// 端点位级保真：k=0 ⇒ 全取 a（逐字节 == fa）；k=1 ⇒ 全取 b
    /// （逐字节 == fb）。溶解仅在「显影侧」直拷原字节、无重算 ⇒ 端点帧
    /// 与源帧逐位一致（任务 3 硬约束）。
    #[test]
    fn dissolve_endpoint_frames_bitwise_equal_source() {
        let w = 16u32;
        let h = 16u32;
        let fa = solid(w, h, [200, 30, 30]);
        let fb = solid(w, h, [30, 30, 200]);
        let at_k0 = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, 0.0);
        assert_eq!(at_k0, fa, "k=0 须逐字节等于源 a");
        assert_ne!(at_k0, fb, "k=0 不得等于源 b");
        let at_k1 = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, 1.0);
        assert_eq!(at_k1, fb, "k=1 须逐字节等于源 b");
        assert_ne!(at_k1, fa, "k=1 不得等于源 a");
        // 边界：k 极接近 0 时（如 1/17）仍以 a 为主、无重算漂移——端点
        // 性质由阈值 `hash < k` 保证（k=0 时 hash < 0 恒假）。
        let eps = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, 1.0 - 1e-6);
        assert_eq!(eps, fb, "k→1⁻ 仍全取 b");
    }

    /// 中点既非 A 亦非 B：k=0.5 时显影阈值将像素分流——既有取自 a 的红
    /// 素、也有取自 b 的蓝素 ⇒ 输出既不等于 fa 也不等于 fb，且两色并存。
    #[test]
    fn dissolve_midpoint_neither_a_nor_b() {
        let w = 32u32;
        let h = 32u32;
        let fa = solid(w, h, [200, 30, 30]);
        let fb = solid(w, h, [30, 30, 200]);
        let mid = mix_frame(&fa, &fb, w, h, Transition::Dissolve { seed: SEED }, 0.5);
        assert_ne!(mid, fa, "中点不得等于源 a");
        assert_ne!(mid, fb, "中点不得等于源 b");
        // 两色并存：存在红素与蓝素（非统一一色）。
        let mut has_red = false;
        let mut has_blue = false;
        for px in mid.chunks_exact(4) {
            if px[0] > px[2] {
                has_red = true;
            }
            if px[2] > px[0] {
                has_blue = true;
            }
        }
        assert!(has_red, "中点须含取自 a 的红素");
        assert!(has_blue, "中点须含取自 b 的蓝素");
        // 阈值分流比例 ≈ k（大样本下）；中点应接近半数显影。
        let mut revealed = 0u32;
        for y in 0..h {
            for x in 0..w {
                if hash2_f32(SEED, x as u64, y as u64) < 0.5 {
                    revealed += 1;
                }
            }
        }
        assert!(
            (revealed as f32 / (w * h) as f32 - 0.5).abs() < 0.1,
            "中点显影比例应接近 0.5（实测 {revealed} / {}）",
            w * h
        );
    }
}

// ===========================================================================
// 任务 4 测试：Track::Morph 动画原语（端点恒等 / Linear 中点 / draw_sec 与
// PathTrack 同比例截断 / fade 语义与 PathTrack 一致）
// ===========================================================================

#[cfg(test)]
mod morph_track_tests {
    use super::*;

    /// 构造默认 MorphSpec（白描边、无淡变、Linear、dur=1、at=0）；
    /// 各测试按需用 struct update 覆写字段。
    fn morph_spec(from: Vec<[f32; 2]>, to: Vec<[f32; 2]>) -> MorphSpec {
        MorphSpec {
            from,
            to,
            color: [1.0, 1.0, 1.0, 1.0],
            width: 2.0,
            fill: false,
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            draw_sec: 0.0,
            closed: false,
            ease: EasingKind::Linear,
            dur: 1.0,
        }
    }

    /// 构造小画幅 ClipSpec（fps=6、时长 1s ⇒ 帧时刻 0, 1/6, ..., 1）。
    fn clip(tracks: Vec<Track>) -> ClipSpec {
        ClipSpec {
            width: 48,
            height: 48,
            fps: 6,
            duration: 1.0,
            background: Background::DarkGlow,
            jpeg_quality: 90,
            tracks,
        }
    }

    /// 端点恒等：t = at 时几何逐点等于 `from`；t = at+dur 时等于 `to`；
    /// 越界（t < at / t > at+dur）分别钳到 `from` / `to`。
    #[test]
    fn geometry_equals_from_at_start_and_to_at_end() {
        let m = MorphSpec {
            at: 1.0,
            dur: 2.0,
            ease: EasingKind::CubicInOut,
            ..morph_spec(
                vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                vec![[0.0, 0.0], [20.0, 5.0], [5.0, 20.0]],
            )
        };
        assert_eq!(morph_points_at(&m, 1.0), m.from, "t = at 须等于 from");
        assert_eq!(morph_points_at(&m, 3.0), m.to, "t = at+dur 须等于 to");
        assert_eq!(morph_points_at(&m, 0.0), m.from, "t < at 钳到 from");
        assert_eq!(morph_points_at(&m, 9.0), m.to, "t > at+dur 钳到 to");
    }

    /// Linear 中点：t = at + dur/2 时 u = 0.5、e = 0.5，几何为逐点中点。
    #[test]
    fn linear_ease_midpoint_is_per_point_midpoint() {
        let m = morph_spec(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
            vec![[0.0, 0.0], [20.0, 5.0], [5.0, 20.0]],
        );
        let mid = morph_points_at(&m, 0.5); // at=0, dur=1 ⇒ u=0.5
        assert_eq!(
            mid,
            vec![[0.0, 0.0], [15.0, 2.5], [7.5, 15.0]],
            "Linear 中点须为逐点中点"
        );
    }

    /// 点数不等：按索引配对渲染 `min(from.len(), to.len())` 个点，
    /// 多出的点忽略。
    #[test]
    fn mismatched_point_counts_pair_by_index_and_ignore_extra() {
        let m = morph_spec(
            vec![[0.0, 0.0], [10.0, 0.0], [99.0, 99.0]],
            vec![[0.0, 0.0], [20.0, 4.0]],
        );
        assert_eq!(morph_points_at(&m, 0.0).len(), 2, "应只渲染 2 个点");
        assert_eq!(
            morph_points_at(&m, 1.0),
            vec![[0.0, 0.0], [20.0, 4.0]],
            "t=at+dur 时等于 to 的前 2 点（from 多出的第 3 点忽略）"
        );
        let m2 = morph_spec(vec![[0.0, 0.0], [10.0, 0.0]], vec![[0.0, 0.0], [20.0, 4.0], [99.0, 99.0]]);
        assert_eq!(morph_points_at(&m2, 1.0).len(), 2, "to 多出的点同样忽略");
    }

    /// draw_sec 与 PathTrack 语义一致（同比例截断）：同一时刻、同一几何
    /// （Morph 求值结果直通 PathTrack），两条轨渲染帧逐字节相同；且
    /// draw_sec 截断真实生效（半途帧 != 全显帧）。
    #[test]
    fn draw_sec_matches_path_track_truncation() {
        // 静态方形折线（from == to ⇒ Morph 任意时刻几何即该方形）。
        let sq: Vec<[f32; 2]> = vec![[6.0, 6.0], [42.0, 6.0], [42.0, 42.0], [6.0, 42.0]];
        let m = MorphSpec {
            draw_sec: 0.5,
            closed: true,
            ..morph_spec(sq.clone(), sq.clone())
        };
        // t = 1/6 ⇒ draw 进度 p = (1/6)/0.5 = 1/3（落在首段中部，非顶点）。
        let path = PathTrack {
            points: morph_points_at(&m, 1.0 / 6.0),
            color: m.color,
            width: m.width,
            fill: m.fill,
            at: m.at,
            in_sec: m.in_sec,
            out_sec: m.out_sec,
            out_at: m.out_at,
            draw_sec: m.draw_sec,
            closed: m.closed,
        };
        let full = PathTrack {
            draw_sec: 0.0, // 立即全显
            ..path.clone()
        };
        let fm = render_rgba_frames(&clip(vec![Track::Morph(m)]));
        let fp = render_rgba_frames(&clip(vec![Track::Path(path)]));
        let ff = render_rgba_frames(&clip(vec![Track::Path(full)]));
        assert_eq!(
            fm[1], fp[1],
            "draw_sec=0.5 @ t=1/6：Morph 与等价 PathTrack 帧逐字节一致（同比例截断）"
        );
        assert_ne!(fm[1], ff[1], "draw_sec 进度截断须真实生效（半途 != 全显）");
    }

    /// fade 语义与 PathTrack 一致：`out_at = Some(a)` 且 `out_sec = 0` 时
    /// `t >= a` 硬切（env = 0，画面与纯背景逐字节一致）；`t < a` 时与
    /// 等价 PathTrack 帧一致；in_sec / out_sec 斜坡值与 PathTrack 共用
    /// 同一 `fade_factor`（直接校验端点）。
    #[test]
    fn fade_semantics_match_path_track() {
        let m = MorphSpec {
            at: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: Some(0.5), // 绝对锚点；out_sec = 0 ⇒ t >= 0.5 硬切
            ..morph_spec(
                vec![[6.0, 6.0], [42.0, 6.0], [42.0, 42.0]],
                vec![[6.0, 20.0], [42.0, 20.0], [42.0, 42.0]],
            )
        };
        // t = 1/3（帧索引 2）时的等价 PathTrack（同几何、同 fade 参数）。
        let path = PathTrack {
            points: morph_points_at(&m, 1.0 / 3.0),
            color: m.color,
            width: m.width,
            fill: m.fill,
            at: m.at,
            in_sec: m.in_sec,
            out_sec: m.out_sec,
            out_at: m.out_at,
            draw_sec: m.draw_sec,
            closed: m.closed,
        };
        let fm = render_rgba_frames(&clip(vec![Track::Morph(m)]));
        let fp = render_rgba_frames(&clip(vec![Track::Path(path)]));
        let fb = render_rgba_frames(&clip(vec![])); // 纯背景
        assert_eq!(
            fm[3], fb[3],
            "t=0.5 >= out_at ⇒ env=0 硬切，帧与纯背景逐字节一致"
        );
        assert_ne!(fm[2], fb[2], "t=1/3 < out_at ⇒ 轨道可见");
        assert_eq!(fm[2], fp[2], "t<out_at：Morph 与等价 PathTrack 帧逐字节一致");
        assert_eq!(fm[3], fp[3], "t>=out_at：两者同样硬切");
        // in_sec / out_sec 斜坡（Morph 与 PathTrack 共用 fade_factor）。
        assert_eq!(fade_factor(0.25, 0.25, 0.5, 0.0, None, 1.0), 0.0, "t=at 淡入为 0");
        assert_eq!(
            fade_factor(0.75, 0.25, 0.5, 0.0, None, 1.0),
            1.0,
            "t=at+in_sec 淡入满"
        );
        assert_eq!(
            fade_factor(0.75, 0.0, 0.0, 0.5, None, 1.0),
            0.5,
            "out_sec 中点（out_start=0.5）淡出过半"
        );
        assert_eq!(
            fade_factor(0.5, 0.0, 0.0, 0.0, Some(0.5), 1.0),
            0.0,
            "out_at + out_sec=0 ⇒ 硬切"
        );
        assert_eq!(
            fade_factor(0.4999, 0.0, 0.0, 0.0, Some(0.5), 1.0),
            1.0,
            "硬切锚点前一刻仍满显"
        );
    }
}
