//! 例子（calculus_film）：**60 秒微积分科普短片**——《微积分》"视频即代码"成品。
//!
//! 一条命令产出一支 960×540 @15fps、含场景转场与片头片尾淡变的 60s 影片，
//! 并以标准 MJPEG-in-AVI（**双流**：MJPG 视频 + pcm_s16le 音频）落盘：
//!
//! ```text
//! 6×ClipSpec（声明式分镜：文字 / 装饰线 / 折线路径轨道）
//!   → render_short_concat_faded_frames（场景间 Fade 0.5s + Clip 级淡入 1.0s/淡出 1.5s）
//!   → 900 帧裸 RGBA8 序列
//! synth_bgm（prism-audio 实时合成：pad/arp/bass/drums 四声部 + 母带淡入淡出）
//!   → write_avi_mjpeg_audio（视频帧 + 音频按帧切 chunk，双流交错 + idx1 双索引）
//!   → prism_calculus_60s.avi
//! ```
//!
//! 全链路**零第三方依赖**：画面由 `prism-author` 声明式轨道程序化绘制，
//! 声音由 `prism-audio` 的振荡器 + ADSR + 鼓机实时合成，封装由 `prism-mux`
//! 手写 RIFF 结构完成——不调用 ffmpeg、不读取任何外部素材。
//!
//! # 分镜（总时长 60.0s）
//!
//! 场景时长和 62.667s − 5×0.5s Fade 转场 = 60.0s（帧数 900 = round(60×15)）：
//! - S1 标题 8.53s（Aurora）：小标签 / kinetic 大标题 / 副题 / 装饰线；
//! - S2 两大主题 9.60s（Scanline）：变化 / 累积 + 中缝装饰线；
//! - S3 极限 10.00s（深蓝黑渐变）：`y=sin(x)/x` 绘制动画 + 极限值标注；
//! - S4 导数 11.07s（深青黑渐变）：抛物线 + 切线（加粗亮色）；
//! - S5 积分 11.07s（深紫黑渐变）：`y=−x²+4` 曲线下从基线闭合的半透明面积；
//! - S6 定理 + 片尾 12.40s（Aurora）：微积分基本定理 + 致谢字幕。
//!
//! 每场景内轨道自带 `in_sec`/`out_sec`（0.4~0.6s）淡入淡出。
//!
//! # BGM
//!
//! 90 BPM、4/4；和弦 Am–F–C–G 每小节一和弦循环，小节数
//! `ceil(60/(4×60/90)) = 23`（末小节超出 60s 部分截断）。四声部
//! （pad → arp → bass → drums 固定序）经 `mix_tracks` 混合，母带
//! 淡入 2s / 淡出 3s 在混音后逐样本施加（`MixTrack` 的 fade 字段是
//! 单轨的，无法表达整片母带淡变）。

use prism_author::{
    render_short_concat_faded_frames, AccentLine, Background, Caption, CaptionStyle, ClipSpec,
    EasingKind, FontKind, KineticText, PathTrack, StaticText, Track, TypewriterText,
};
use prism_audio::drums::{self, DrumSpec};
use prism_audio::mixer::{mix_tracks, MixTrack};
use prism_audio::synth::presets::{preset, Mood, Timbre};
use prism_audio::synth::{render_note, Adsr, Waveform};
use prism_audio::PcmBuffer;
use prism_mux::write_avi_mjpeg_audio;

// ---- 全片统一参数 ----

/// 画幅宽（像素）。
const WIDTH: u32 = 960;
/// 画幅高（像素）。
const HEIGHT: u32 = 540;
/// 帧率（帧/秒）。
const FPS: u32 = 15;
/// AVI JPEG 质量（1..=100）。
const JPEG_QUALITY: u8 = 82;
/// 输出文件路径。
const OUT_PATH: &str = "prism_calculus_60s.avi";

/// 场景间转场时长（秒）：相邻场景一律 `Transition::Fade`（由
/// `render_short_concat_faded_frames` 内部固定，此处仅用于帧数推演）。
const TRANS_SEC: f32 = 0.5;
/// Clip 级片头 / 片尾淡入淡出（秒）。
const CLIP_FADE_IN: f32 = 1.0;
const CLIP_FADE_OUT: f32 = 1.5;

/// 各场景帧数（@15fps）：128+144+150+166+166+186 = 940 帧；
/// 减去 5 次转场重叠（`round(0.5×15)` = 8 帧/次）得 **900 帧 = 60.0s**。
///
/// 取帧对齐值的原因：`ClipSpec::frame_count()` 为 `ceil(fps·duration)`，
/// 而分镜给出的秒数（8.5/9.5/10/11/11/12，和 62s）经 ceil 后最多只有
/// 936 帧、减转场至多 896 帧，**不可能**凑出 900 帧；按 1/15s 帧网格
/// 重取整后总帧数精确 = 900（各场景时长相对原分镜偏差 ≤0.4s，
/// 分镜内容与坐标不变）。
const SCENE_FRAMES: [u32; 6] = [128, 144, 150, 166, 166, 186];

/// 帧数 → 场景时长（秒）：`n/FPS` 减 2ms 安全边际。
///
/// `frame_count()` 是 `ceil(fps·duration)`：若 duration 因浮点表示略高于
/// `n/FPS`，ceil 会多进一帧。留 2ms 边际（远大于 f32 在 10 秒量级的
/// ~1e-5 表示误差）保证 ceil 精确落到 `n` 帧。
fn scene_duration(frames: u32) -> f32 {
    frames as f32 / FPS as f32 - 0.002
}

// ---- 曲线采样与画面映射 ----

/// sinc 函数（x=0 处取极限值 1）。
fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        x.sin() / x
    }
}

/// S3 曲线 `y = sin(x)/x`（x∈[−3π, 3π]）采样为画面折线。
///
/// 映射：画面 x∈[80, 880]（6π 数学区间铺满 800px）；函数峰值 1 → y=180、
/// 基线（函数值 0）→ y=380，即 `sy = 380 − 200·sinc(x)`。取 49 点（奇数）
/// 使 x=0 的峰值恰好落在画面中心 (480, 180)。
fn sinc_curve_points() -> Vec<[f32; 2]> {
    const N: usize = 49;
    let span = 6.0 * std::f32::consts::PI; // x 总跨度
    (0..N)
        .map(|i| {
            let x = -3.0 * std::f32::consts::PI + span * (i as f32 / (N - 1) as f32);
            let sx = 80.0 + (x + 3.0 * std::f32::consts::PI) / span * 800.0;
            [sx, 380.0 - 200.0 * sinc(x)]
        })
        .collect()
}

/// S4/S5 共用画面映射：数学 x∈[−2, 2] → 画面 x∈[240, 720]；
/// 数学 y → 画面 `y = 420 − 90·y`（y=x² 顶点 (0,0)→(480,420)、x=±2 → y=60）。
fn map_xy(x: f32, y: f32) -> [f32; 2] {
    [240.0 + (x + 2.0) / 4.0 * 480.0, 420.0 - 90.0 * y]
}

/// S4 抛物线 `y = x²`（x∈[−2, 2]，41 点含顶点 x=0）。
fn parabola_points() -> Vec<[f32; 2]> {
    const N: usize = 41;
    (0..N)
        .map(|i| {
            let x = -2.0 + 4.0 * (i as f32 / (N - 1) as f32);
            map_xy(x, x * x)
        })
        .collect()
}

/// S4 切线 `y = 2x − 1`（抛物线在 x=1 处的切线：斜率 2x=2、过点 (1,1)），
/// x∈[0, 2] 仅取两端点，与抛物线同一映射。
fn tangent_points() -> Vec<[f32; 2]> {
    vec![map_xy(0.0, -1.0), map_xy(2.0, 3.0)]
}

/// S5 倒抛物线 `y = −x² + 4`（x∈[−2, 2]，41 点）**曲线下从基线 y=420 闭合**：
/// 折线沿曲线从左到右，末尾补两点 [720,420] 与 [240,420] 回到基线，
/// 构成闭合填充区域（`fill=true` 时渲染层自动闭合轮廓）。
fn filled_area_points() -> Vec<[f32; 2]> {
    const N: usize = 41;
    let mut pts: Vec<[f32; 2]> = (0..N)
        .map(|i| {
            let x = -2.0 + 4.0 * (i as f32 / (N - 1) as f32);
            map_xy(x, -x * x + 4.0)
        })
        .collect();
    pts.push([720.0, 420.0]); // 右端点（恰在基线上）回基线
    pts.push([240.0, 420.0]); // 左端点回基线，闭合"曲线下面积"
    pts
}

// ---- 六场景分镜 ----

/// S1 标题（8.53s，Aurora）：静态小标签 / kinetic 大标题 / 打字机副题 / 装饰线。
fn scene_title() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[0]),
        background: Background::Aurora,
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // 静态小标签（片头 0.5s 淡入、片尾 0.6s 淡出）。
            Track::Static(StaticText {
                content: "PRISM ENGINE · 零依赖程序化媒体".into(),
                font: FontKind::Regular,
                size: 22.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.3,
                in_sec: 0.5,
                letter_spacing: 7.0,
                y: 150.0,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            // 逐字 kinetic 大标题（片头 0.4s 淡入、片尾 0.6s 淡出）。
            Track::Kinetic(KineticText {
                content: "微积分".into(),
                font: FontKind::Bold,
                size: 110.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.5,
                stagger: 0.12,
                per_char: 0.5,
                scale_from: 0.72,
                rise_px: 22.0,
                easing: EasingKind::CubicOut,
                y: 280.0,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            // 装饰线（片头 0.4s 淡入、片尾 0.4s 淡出）。
            Track::Accent(AccentLine {
                center_x: 480.0,
                y: 345.0,
                w: 260.0,
                h: 3.0,
                color: [0.30, 0.75, 0.95, 0.9],
                at: 1.0,
                in_sec: 0.4,
                out_sec: 0.4,
                out_at: None,
            }),
            // 打字机副题（逐字 + 闪烁光标）。
            Track::Typewriter(TypewriterText {
                content: "一门研究'变化'与'累积'的数学".into(),
                font: FontKind::Regular,
                size: 30.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 1.2,
                y: 410.0,
                cps: 14.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
        ],
    }
}

/// S2 两大主题（9.60s，Scanline）："变化"与"累积"分栏 + 中缝装饰线。
///
/// 布局说明：渲染层所有文字轨道（Static / Kinetic / Typewriter）均以画幅
/// **水平中心**为锚（规格无 x 字段），左右分栏无法在现有公开 API 下实现，
/// 故按"上/下分栏 + 中缝"呈现——主题信息（变化→导数、累积→积分）不变。
fn scene_two_themes() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[1]),
        background: Background::Scanline,
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // 上栏：变化（青）。
            Track::Static(StaticText {
                content: "变化".into(),
                font: FontKind::Bold,
                size: 76.0,
                color: [0.35, 0.85, 0.95, 0.95],
                at: 0.3,
                in_sec: 0.4,
                letter_spacing: 10.0,
                y: 150.0,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            Track::Typewriter(TypewriterText {
                content: "变化的速率 —— 导数".into(),
                font: FontKind::Regular,
                size: 28.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 0.8,
                y: 215.0,
                cps: 14.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.3,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
            // 中缝装饰线。
            Track::Accent(AccentLine {
                center_x: 480.0,
                y: 268.0,
                w: 600.0,
                h: 2.0,
                color: [0.30, 0.75, 0.95, 0.9],
                at: 1.1,
                in_sec: 0.4,
                out_sec: 0.4,
                out_at: None,
            }),
            // 下栏：累积（紫）。
            Track::Static(StaticText {
                content: "累积".into(),
                font: FontKind::Bold,
                size: 76.0,
                color: [0.72, 0.62, 0.95, 0.95],
                at: 1.3,
                in_sec: 0.4,
                letter_spacing: 10.0,
                y: 325.0,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            Track::Typewriter(TypewriterText {
                content: "累积的总量 —— 积分".into(),
                font: FontKind::Regular,
                size: 28.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 1.8,
                y: 390.0,
                cps: 14.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.3,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
        ],
    }
}

/// S3 极限（10.00s，深蓝黑渐变）：`sin(x)/x` 绘制动画 + 极限值标注。
fn scene_limit() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[2]),
        background: Background::Gradient {
            top: [0.04, 0.06, 0.13],
            bottom: [0.01, 0.02, 0.05],
        },
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // sin(x)/x 曲线：at 后 2.5s 按折线总长比例逐段画出。
            Track::Path(PathTrack {
                points: sinc_curve_points(),
                color: [0.55, 0.85, 1.0, 0.95],
                width: 3.0,
                fill: false,
                at: 0.5,
                in_sec: 0.5,
                out_sec: 0.6,
                out_at: None,
                draw_sec: 2.5,
                closed: false,
            }),
            // 峰值处标注（曲线中点约 at+1.25s 画到峰值，故 2.0s 出场）。
            Track::Static(StaticText {
                content: "极限值 1".into(),
                font: FontKind::Bold,
                size: 30.0,
                color: [1.0, 0.90, 0.55, 0.95],
                at: 2.0,
                in_sec: 0.4,
                letter_spacing: 3.0,
                y: 140.0,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            // 底部打字机注脚。
            Track::Typewriter(TypewriterText {
                content: "极限：无限靠近的思想实验".into(),
                font: FontKind::Regular,
                size: 28.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 2.5,
                y: 490.0,
                cps: 12.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
        ],
    }
}

/// S4 导数（11.07s，深青黑渐变）：抛物线绘制 + 切线（加粗亮色）。
fn scene_derivative() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[3]),
        background: Background::Gradient {
            top: [0.02, 0.09, 0.10],
            bottom: [0.01, 0.04, 0.05],
        },
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // 抛物线 y=x²：3.0s 绘制动画。
            Track::Path(PathTrack {
                points: parabola_points(),
                color: [0.45, 0.90, 0.95, 0.95],
                width: 3.5,
                fill: false,
                at: 0.5,
                in_sec: 0.5,
                out_sec: 0.6,
                out_at: None,
                draw_sec: 3.0,
                closed: false,
            }),
            // 切线 y=2x−1（x=1 处）：抛物线画完后 4.0s 出场，加粗亮色。
            Track::Path(PathTrack {
                points: tangent_points(),
                color: [1.0, 0.82, 0.35, 0.95],
                width: 6.0,
                fill: false,
                at: 4.0,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                draw_sec: 1.2,
                closed: false,
            }),
            // 定义注脚。
            Track::Typewriter(TypewriterText {
                content: "导数 = 瞬时变化率 = 切线的斜率".into(),
                font: FontKind::Regular,
                size: 28.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 5.5,
                y: 500.0,
                cps: 13.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
        ],
    }
}

/// S5 积分（11.07s，深紫黑渐变）：`y=−x²+4` 曲线下从基线闭合的半透明面积。
fn scene_integral() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[4]),
        background: Background::Gradient {
            top: [0.08, 0.04, 0.13],
            bottom: [0.03, 0.02, 0.06],
        },
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // 半透明青色面积填充（闭合区域 = 曲线下从基线 y=420 回到基线）。
            Track::Path(PathTrack {
                points: filled_area_points(),
                color: [0.35, 0.85, 0.80, 0.38],
                width: 0.0, // 填充模式无视描边宽度
                fill: true,
                at: 0.5,
                in_sec: 0.6,
                out_sec: 0.6,
                out_at: None,
                draw_sec: 0.0, // 绘制进度仅描边模式生效
                closed: true,
            }),
            // 定义注脚。
            Track::Typewriter(TypewriterText {
                content: "积分 = 累积的总量 = 曲线下的面积".into(),
                font: FontKind::Regular,
                size: 28.0,
                color: [0.85, 0.85, 0.95, 1.0],
                at: 2.0,
                y: 500.0,
                cps: 13.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
        ],
    }
}

/// S6 定理 + 片尾（12.40s，Aurora）：微积分基本定理 + 致谢字幕。
fn scene_theorem() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: scene_duration(SCENE_FRAMES[5]),
        background: Background::Aurora,
        jpeg_quality: JPEG_QUALITY,
        tracks: vec![
            // 定理标题（静态、字距拉开）。
            Track::Static(StaticText {
                content: "微积分基本定理".into(),
                font: FontKind::Bold,
                size: 64.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.4,
                in_sec: 0.5,
                letter_spacing: 6.0,
                y: 210.0,
                out_sec: 0.6,
                out_at: None,
                center_x: None,
            }),
            // 定理释义（打字机）。
            Track::Typewriter(TypewriterText {
                content: "求变化率与求累积，本是一体".into(),
                font: FontKind::Regular,
                size: 30.0,
                color: [0.82, 0.90, 0.98, 1.0],
                at: 1.2,
                y: 330.0,
                cps: 13.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.4,
                out_sec: 0.6,
                out_at: None,
                letter_spacing: 2.0,
                center_x: None,
            }),
            // 致谢字幕条（场景后段出场，片尾淡出收在全片末尾）。
            Track::Caption(Caption {
                content: "Made with Prism Engine · 视频即代码 · BGM 由 prism-audio 实时合成".into(),
                at: 6.5,
                in_sec: 0.6,
                out_sec: 0.8,
                out_at: None,
                style: CaptionStyle {
                    font_size: 26.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                    bg: [0.0, 0.0, 0.0, 0.55],
                    pad_x: 22.0,
                    pad_y: 12.0,
                    margin_bottom: 40.0,
                    line_spacing: 0.0,
                },
            }),
        ],
    }
}

// ---- BGM：prism-audio 四声部实时合成 ----

/// BGM 速度（BPM）。
const BGM_BPM: f32 = 90.0;
/// BGM 采样率（mono 44100——44100 是帧率 15 的整数倍，满足 AVI 按帧切
/// 音频 chunk 的整除要求：2940 采样/帧）。
const BGM_SR: u32 = 44_100;

/// 一个和弦的声部音（MIDI）：[根音, 三音, 五音, 高八度根音]。
#[derive(Clone, Copy)]
struct Chord {
    /// MIDI 音高表（根 / 三 / 五 / 高八度根）。
    notes: [i32; 4],
}

/// Am–F–C–G（C 大调 vi–IV–I–V）每小节一和弦循环铺满。
///
/// 音高按实际调性推算（十二平均律 MIDI）：
/// - Am（A 小调三和弦 A+C+E）：A2=45 根 / C4=60 三 / E4=64 五 / A4=69 高八度；
/// - F （F 大三和弦 F+A+C）：F2=53 根 / A3=57 三 / C4=60 五 / F4=65 高八度；
/// - C （C 大三和弦 C+E+G）：C3=48 根 / E4=64 三 / G4=67 五 / C5=72 高八度；
/// - G （G 大三和弦 G+B+D）：G2=43 根 / B3=59 三 / D4=62 五 / G4=67 高八度。
const CHORDS: [Chord; 4] = [
    Chord { notes: [45, 60, 64, 69] },
    Chord { notes: [53, 57, 60, 65] },
    Chord { notes: [48, 64, 67, 72] },
    Chord { notes: [43, 59, 62, 67] },
];

/// MIDI 音高 → 频率（Hz）：`440 · 2^((m−69)/12)`（十二平均律，A4 = 440Hz）。
///
/// 注：`prism_audio::theory::midi_to_freq` 是 crate 内"半音查表 + 八度
/// 整数幂连乘"的确定性同式实现；本 example 按分镜规格以闭式公式给出。
fn midi_to_freq(midi: i32) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

/// 把一段音符样本叠加进 `buf` 的 `at_sec` 秒偏移处（越界部分截断）。
fn place(buf: &mut [f32], at_sec: f32, samples: &[f32], sample_rate: u32) {
    let start = (at_sec * sample_rate as f32).round() as usize;
    for (i, v) in samples.iter().enumerate() {
        if start + i >= buf.len() {
            break; // 音符伸到母带末尾之外：截断
        }
        buf[start + i] += v;
    }
}

/// 长度规整：不足补 0、超出截断（母带长度是全片唯一事实源）。
fn fit_len(buf: PcmBuffer, total: usize) -> PcmBuffer {
    let mut data = buf.as_slice().to_vec();
    data.resize(total, 0.0);
    PcmBuffer::from_mono(buf.sample_rate(), data).expect("单声道缓冲长度恒合法")
}

/// Pad 音色：取 `Mood::Calm` 预设的铺底音色（正弦波 + 慢 ADSR）。
///
/// 预设为编译期常量表、恒可取；仍防御性兜底——参数退化时手写慢 ADSR。
fn pad_timbre() -> Timbre {
    let t = preset(Mood::Calm).pad;
    if t.envelope.attack + t.envelope.decay + t.envelope.release <= 0.0 {
        Timbre {
            waveform: t.waveform,
            envelope: Adsr { attack: 0.80, decay: 0.40, sustain: 0.70, release: 1.20 },
        }
    } else {
        t
    }
}

/// 合成整片 BGM（mono，`duration_sec` 秒）。
///
/// - 小节数 = `ceil(duration / (4×60/BPM))`（90BPM ⇒ 每小节 2.6667s，
///   60s ⇒ 23 小节，末小节超出部分截断）；
/// - 四声部（轨序 pad → arp → bass → drums）各渲染为等长单声道缓冲后
///   经 [`mix_tracks`] 固定序线性混合（`mix_tracks` 内含峰值防削波：
///   混合峰值 >1 时整轨等比缩放，故母带峰值恒 ≤1）；
/// - **母带淡入 2s / 淡出 3s** 在混音之后逐样本乘线性包络——`MixTrack`
///   的 `fade_in_secs`/`fade_out_secs` 是**单轨**参数（只作用于该轨
///   缓冲），无法表达"整片母带"的淡变，故在混音产物上统一施加；
/// - 长度不变量：样本数 = `round(duration×sample_rate)`（不足补 0、
///   超出截断），淡入 / 淡出端点样本恒为 0。
fn synth_bgm(duration_sec: f32, sample_rate: u32) -> PcmBuffer {
    let total = (duration_sec * sample_rate as f32).round() as usize;
    let bar_sec = 4.0 * 60.0 / BGM_BPM; // 每小节秒数（2.6667s）
    let eighth_sec = bar_sec / 8.0; // 八分音符秒数（0.3333s）
    let bars = (duration_sec / bar_sec).ceil().max(1.0) as usize; // 23 小节

    // ---- 声部 1：Pad（每小节根+三+五长音 2.5s，Calm 预设音色）----
    let pad = pad_timbre();
    let pad_frames = (2.5 * sample_rate as f32).round() as usize;
    let mut pad_buf = vec![0.0_f32; total];
    for bar in 0..bars {
        let chord = &CHORDS[bar % CHORDS.len()];
        let at = bar as f32 * bar_sec;
        for &m in &chord.notes[..3] {
            let note = render_note(
                pad.waveform,
                midi_to_freq(m),
                &pad.envelope,
                1.0,
                pad_frames,
                sample_rate,
            );
            place(&mut pad_buf, at, &note, sample_rate);
        }
    }

    // ---- 声部 2：Arp（八分音符轮转和弦音，高两个八度，每音 0.28s）----
    let arp_env = Adsr { attack: 0.005, decay: 0.22, sustain: 0.18, release: 0.05 };
    let arp_frames = (0.28 * sample_rate as f32).round() as usize;
    let mut arp_buf = vec![0.0_f32; total];
    for bar in 0..bars {
        let chord = &CHORDS[bar % CHORDS.len()];
        let bar_start = bar as f32 * bar_sec;
        for step in 0..8 {
            let m = chord.notes[step % 4] + 24; // 和弦音轮转、高两个八度
            let at = bar_start + step as f32 * eighth_sec;
            let note = render_note(
                Waveform::Triangle,
                midi_to_freq(m),
                &arp_env,
                1.0,
                arp_frames,
                sample_rate,
            );
            place(&mut arp_buf, at, &note, sample_rate);
        }
    }

    // ---- 声部 3：Bass（每小节根音低八度，1.2s）----
    let bass_env = Adsr { attack: 0.01, decay: 0.18, sustain: 0.55, release: 0.25 };
    let bass_frames = (1.2 * sample_rate as f32).round() as usize;
    let mut bass_buf = vec![0.0_f32; total];
    for bar in 0..bars {
        let root = CHORDS[bar % CHORDS.len()].notes[0] - 12; // 根音低八度
        let note = render_note(
            Waveform::Sine,
            midi_to_freq(root),
            &bass_env,
            1.0,
            bass_frames,
            sample_rate,
        );
        place(&mut bass_buf, bar as f32 * bar_sec, &note, sample_rate);
    }

    // ---- 声部 4：Drums（soft kick 四四拍：16 分网格每 4 步一 kick）----
    // 以 drums.rs 既有 API 编排：sequence 把 16 分音符网格映射为打击事件，
    // render 逐事件经 synth 合成累加。此处只用底鼓、落在每个四分音符上，
    // 音量经混音轨增益压低（soft）；不改动 drums.rs。
    let steps = 16 * bars; // 16 步/小节（一步 = 16 分音符）
    let mut grid = vec![vec![0_u8; steps]; 3]; // 行序 = Kick / Snare / Hat
    for (s, cell) in grid[0].iter_mut().enumerate() {
        if s % 4 == 0 {
            *cell = 1; // 四分音符（四四拍）一 kick
        }
    }
    let drum_spec = DrumSpec { steps, grid };
    let drum_track =
        drums::sequence(&drum_spec, BGM_BPM, sample_rate).expect("鼓机编排参数合法");
    let drums_buf = drums::render(&drum_track);

    // ---- 混音（轨序 pad→arp→bass→drums；增益见分镜规格）----
    let mix = mix_tracks(&[
        MixTrack::new(
            PcmBuffer::from_mono(sample_rate, pad_buf).expect("pad 缓冲构造"),
            0.22,
        ),
        MixTrack::new(
            PcmBuffer::from_mono(sample_rate, arp_buf).expect("arp 缓冲构造"),
            0.16,
        ),
        MixTrack::new(
            PcmBuffer::from_mono(sample_rate, bass_buf).expect("bass 缓冲构造"),
            0.26,
        ),
        MixTrack::new(fit_len(drums_buf, total), 0.45), // soft kick
    ])
    .expect("四声部等长同型，混音不应失败");

    // ---- 母带淡入 2s / 淡出 3s（混音后逐样本乘线性斜坡包络）----
    let n = mix.frames();
    let fin = (2.0 * sample_rate as f32).round() as usize;
    let fout = (3.0 * sample_rate as f32).round() as usize;
    let mut data = mix.as_slice().to_vec();
    for (i, x) in data.iter_mut().enumerate() {
        let g_in = if fin > 0 && i < fin { i as f32 / fin as f32 } else { 1.0 };
        let g_out = if fout > 0 && i + fout >= n {
            (n - 1 - i) as f32 / fout as f32
        } else {
            1.0
        };
        *x *= g_in.min(g_out).max(0.0);
    }
    PcmBuffer::from_mono(sample_rate, data).expect("母带缓冲构造")
}

fn main() {
    // ---- 1) BGM：60s mono 44100 实时合成 ----
    let bgm = synth_bgm(60.0, BGM_SR);
    // 关键不变量（分镜规格收尾断言）：长度精确、峰值 ≤1、淡入淡出端点为 0。
    assert_eq!(bgm.frames(), 60 * BGM_SR as usize, "BGM 样本数必须精确 = 60s×44100");
    assert_eq!(bgm.channels(), 1, "BGM 必须为单声道");
    assert!(bgm.peak() <= 1.0 + 1e-6, "母带峰值必须 ≤1（防削波），实测 {}", bgm.peak());
    assert_eq!(bgm.sample(0, 0), Some(0.0), "淡入端点样本必须为 0");
    assert_eq!(bgm.sample(bgm.frames() - 1, 0), Some(0.0), "淡出端点样本必须为 0");

    // ---- 2) 六场景声明式分镜 ----
    let specs = vec![
        scene_title(),      // S1 标题 8.53s（Aurora）
        scene_two_themes(), // S2 两大主题 9.60s（Scanline）
        scene_limit(),      // S3 极限 10.00s（深蓝黑渐变）
        scene_derivative(), // S4 导数 11.07s（深青黑渐变）
        scene_integral(),   // S5 积分 11.07s（深紫黑渐变）
        scene_theorem(),    // S6 定理 + 片尾 12.40s（Aurora）
    ];

    // ---- 3) 渲染 + 转场拼接 + Clip 级淡变 → 900 帧裸 RGBA8 ----
    let frames = render_short_concat_faded_frames(&specs, CLIP_FADE_IN, CLIP_FADE_OUT);
    let trans_frames = (TRANS_SEC * FPS as f32).round() as usize; // 8 帧/次
    let expect = SCENE_FRAMES.iter().sum::<u32>() as usize - 5 * trans_frames;
    assert_eq!(frames.len(), expect, "总帧数必须 = 场景帧数和 − 5×转场重叠");
    assert_eq!(frames.len(), 900, "60s @15fps 必须精确 900 帧");

    // ---- 4) 音画封装为标准 MJPEG-in-AVI 双流 ----
    let avi = write_avi_mjpeg_audio(&frames, WIDTH, HEIGHT, FPS, JPEG_QUALITY, &bgm)
        .expect("AVI 封装失败");
    std::fs::write(OUT_PATH, &avi).expect("写入 AVI 失败");

    println!(
        "calculus_film 完成：{} 帧 @{}×{} {}fps（{:.2}s；6 场景 + 5×{:.1}s Fade + 片头 {:.1}s / 片尾 {:.1}s 淡变）",
        frames.len(),
        WIDTH,
        HEIGHT,
        FPS,
        frames.len() as f32 / FPS as f32,
        TRANS_SEC,
        CLIP_FADE_IN,
        CLIP_FADE_OUT,
    );
    println!(
        "  BGM：{} 样本 / {:.2}s（mono {}Hz，{}BPM Am–F–C–G × {} 小节，母带峰值 {:.3}）",
        bgm.frames(),
        bgm.duration_secs(),
        bgm.sample_rate(),
        BGM_BPM,
        (60.0 / (4.0 * 60.0 / BGM_BPM)).ceil() as usize,
        bgm.peak(),
    );
    println!("  输出：{}（{} 字节）", OUT_PATH, avi.len());
}
