//! 例子（calculus_film_v2）：**90 秒微积分科普短片 v2**——《微积分》
//! 「视频即代码」成片（T05 完整主入口）。
//!
//! # 模块结构
//!
//! - [`mod@common`] / [`mod@axes`] / [`mod@symbols`] / [`mod@bgm`]：T04
//!   影片资产层（共享约定 / 坐标辅助 / 矢量符号 / BGM 合成）；
//! - [`mod@scenes_math`]：S3–S6 四个数学场景（极限 / 导数 / 积分 /
//!   基本定理）；
//! - [`mod@scene3d`]：S7 片尾 3D 光追序列（场景组装 + 相机轨道 +
//!   行分带并行预渲染 + 后处理链）；
//! - 本文件：S1 标题符号阵 / S2 坐标系亮相 + 相机缓推 / S7 序列装配与
//!   落版，七场景拼接 + BGM + mux + 落盘的完整主入口。
//!
//! # main 流程（架构 §7.5 装配期断言）
//!
//! 资产层自检 → S7 预渲染（310 源帧 @30fps）→ 组装 7×`ClipSpec`
//! （帧表 / 画幅 / S7 时长断言）→
//! [`prism_author::render_short_concat_faded_frames_each`] **流式**回调
//! （片头 1.0s / 片尾 1.5s Clip 淡变，逐帧 JPEG 编码，2700 帧断言；
//! 不驻留全片裸帧——90s 960×540 全片 RGBA8 ≈5.6GB 超出容器内存上限）
//! → `synth_bgm(90.0, 44100)`（长度 / 峰值 / 首尾 0.05s 淡变断言）→
//! [`prism_mux::write_avi_mjpeg_audio_frames`]（预编码 JPEG 帧 + PCM
//! 双流 AVI）→ 落盘 `prism_calculus_90s.avi`。
//!
//! # 帧网格（架构 §7.6 / PRD §4.0）
//!
//! 场景帧数 248 / 308 / 496 / 496 / 496 / 436 / 310（Σ2790），减 6 次
//! Fade 0.5s（各 15 帧）得 **2700 帧 = 90.000s @30fps**；BGM 90.000s
//! = 3,969,000 帧 @44100 立体声（1470 帧/秒整除；母带展宽：L/R 微偏移
//! + 混响抽头）。
//!
//! # 分镜实现要点（与 PRD 的偏差均就地注释备案）
//!
//! - S1：三符号（lim / dy/dx / ∫）每**轮廓**一条 `DynPath`——
//!   `draw_sec` 描画入场（1.5–4.7s）、5.0–6.0 呼吸缩放 + 上浮
//!   （CubicInOut，逐帧仿射变换）、7.5s `out_at` 锚定淡出；
//! - S2：相机缓推以**逐帧缩放视口**实现（[`s2_viewport_at`]，scale
//!   1.00→1.08 关于绘图区中心，CubicInOut，禁 `postfx::Warp`）——网格 /
//!   轴 / 箭头 / 刻度 / 曲线全部经该视口 `to_px` 每帧重算；轴标签
//!   （`StaticText` 无逐帧重定位能力）在缓推启动前（4.5s）退役，
//!   避免与轴端漂移脱钩；
//! - S7：`Track::Sequence` 承载 310 帧预渲染序列（src_fps 30 → 片上
//!   30fps 逐帧对应），文案 / 切平面公式 / 落版为 2D 叠加轨；切点 P 的
//!   屏位由 [`project_world`] 按 `camera_at(4.5)` 精确解算（与渲染器
//!   `Camera::pixel_dir` 逆映射同参）。

// 模块目录布局（架构 §3.3）：根文件 `examples/calculus_film_v2.rs` +
// 同源目录 `examples/calculus_film_v2/`。rustc 对**crate 根**的 `mod`
// 声明按根文件所在目录解析（而非 `<根文件名>/` 子目录），故此处以
// `#[path]` 显式指向架构文件清单中的模块路径——文件落点与 §3.3 逐项一致。
#[path = "calculus_film_v2/axes.rs"]
mod axes;
#[path = "calculus_film_v2/bgm.rs"]
mod bgm;
#[path = "calculus_film_v2/common.rs"]
mod common;
#[path = "calculus_film_v2/film_postfx.rs"]
mod film_postfx;
#[path = "calculus_film_v2/scene3d.rs"]
mod scene3d;
#[path = "calculus_film_v2/scenes_math.rs"]
mod scenes_math;
#[path = "calculus_film_v2/symbols.rs"]
mod symbols;

use std::sync::Arc;

use prism_author::{
    Background, ClipSpec, DynPathTrack, EasingKind, FontKind, GeomFn, KineticText, SequenceTrack,
    ShapeKind, ShapeTrack, StaticText, Track, TypewriterText,
    render_short_concat_faded_frames_each,
};
use prism_math::Vec3;
use prism_mux::jpeg::{JpegConfig, encode_jpeg_color};
use prism_mux::write_avi_mjpeg_audio_frames;
use prism_raster::geom::Transform;
use prism_rt::camera::Camera;

use common::{
    CURVE_RGBA, GOLD_RGBA, GRID_RGBA, TEXT_RGBA, WHITE_RGBA, AXIS_RGBA, TICK_RGBA,
};
use symbols::{FormulaPart, SymbolKind};

// ---------------------------------------------------------------------------
// 全片统一参数
// ---------------------------------------------------------------------------

/// 画幅宽（像素）。
const WIDTH: u32 = 960;
/// 画幅高（像素）。
const HEIGHT: u32 = 540;
/// 帧率（帧/秒）。
const FPS: u32 = 30;
/// MJPEG 质量（1..=100；v1 先例 82）。
const JPEG_QUALITY: u8 = 82;
/// 片头 Clip 淡入（秒；PRD §4.0「黑场淡入」）。
const CLIP_FADE_IN: f32 = 1.0;
/// 片尾 Clip 淡出（秒；落版后全局渐黑）。
const CLIP_FADE_OUT: f32 = 1.5;
/// BGM 采样率（立体声 44100——44100 是帧率 30 的整数倍：1470 帧/秒）。
const BGM_SR: u32 = 44_100;
/// 成片落盘路径（T05 验收项 1）。
const OUT_PATH: &str = "prism_calculus_90s.avi";
/// 预览帧输出目录（v3-B 交付物；`PRISM_PREVIEW_S2=N` 时启用）。
const PREVIEW_DIR: &str = "/tmp/v3_preview";
/// 预览截断的 S2 帧数（内容窗 [0.5s, 8.4s] 全cover + 转场安全边）。
const PREVIEW_S2_FRAMES: u32 = 271;
/// 预览内容窗（帧号，闭区间）：片头 1.0s 淡入之后（帧 36 = 1.2s）至
/// 与哑场景的 0.5s 叠化区之前（帧 256 起进入转场混合，窗止于 252）。
const PREVIEW_WINDOW: (usize, usize) = (36, 252);

/// 上下文层 alpha（S2 轴标签 / HUD / 底部文案；3B1B 分层：主 1.0 /
/// 上下文 ~0.55 / 结构 ~0.25）。
const ALPHA_CTX: f32 = 0.55;
/// 上下文层文案色（S2 底部打字文案 / 开幕目标字幕：冷白 alpha 0.50）。
const TEXT_CTX_RGBA: [f32; 4] = [TEXT_RGBA[0], TEXT_RGBA[1], TEXT_RGBA[2], 0.50];

/// 把颜色压到上下文层 alpha（轴标签 / HUD 专用；主曲线保持 1.0 主层）。
fn ctx(color: [f32; 4]) -> [f32; 4] {
    [color[0], color[1], color[2], ALPHA_CTX]
}

fn main() {
    // ---- 预览模式（v3-B 交付项）：PRISM_PREVIEW_S2=N ⇒ 只渲染 S2 内容
    // 窗内均匀 N 帧，过任务 B 后处理链（bloom → grade → vignette）后
    // 直接写 PNG 到 /tmp/v3_preview/ 并退出。早于正式流程返回，不影响
    // 正式渲染的确定性与输出（后处理链本身是纯函数，两路同参）。----
    if let Ok(n) = std::env::var("PRISM_PREVIEW_S2") {
        let n: usize = n
            .parse()
            .expect("PRISM_PREVIEW_S2 须为 >= 1 的整数（取样帧数）");
        run_preview_s2(n);
        return;
    }

    // ---- 0) T04 资产层自检（共享约定 / 符号 / 轴 / BGM 冒烟）----
    check_frame_grid();
    check_colors();
    check_viewport();
    check_axes();
    check_symbols();
    check_bgm();

    // ---- 1) S7 预渲染：310 源帧 @30fps = 10.333s（= 片上 310 帧）----
    let seq = scene3d::prerender_frames(310);
    assert_eq!(seq.frames.len(), 310, "S7 序列必须恰为 310 源帧");
    assert_eq!(
        (seq.width, seq.height),
        (WIDTH, HEIGHT),
        "S7 序列画幅必须等于影片画幅（不符则该轨被渲染层跳过）"
    );

    // ---- 2) 七场景声明式分镜 ----
    let specs = vec![
        scene_s1(),                    // S1 标题·公式符号矢量入场 8.27s
        scene_s2(),                    // S2 坐标系亮相 + 相机缓推 10.27s
        scenes_math::scene_s3(),       // S3 极限·ε-δ 带 16.53s
        scenes_math::scene_s4(),       // S4 导数·切线滑动 16.53s
        scenes_math::scene_s5(),       // S5 积分·黎曼和 16.53s
        scenes_math::scene_s6(),       // S6 基本定理·双链合流 14.53s
        scene_s7(seq),                 // S7 片尾 flex·3D 光追 10.33s
    ];
    // 装配期灾难性错误快速失败（架构 §7.5）：帧表 / 画幅 / 帧率 / S7 时长。
    let frame_table: Vec<u32> = specs.iter().map(|s| s.frame_count() as u32).collect();
    assert_eq!(
        frame_table,
        common::SCENE_FRAMES.to_vec(),
        "七场景帧表必须逐项等于架构 §7.6"
    );
    for (i, s) in specs.iter().enumerate() {
        assert_eq!((s.width, s.height, s.fps), (WIDTH, HEIGHT, FPS), "场景 {i} 画幅/帧率不一致");
    }
    let sum: u32 = frame_table.iter().sum();
    assert_eq!(sum, 2790, "场景帧数和必须为 2790");
    assert_eq!(
        sum as usize - common::TRANS_COUNT * common::TRANS_FRAMES,
        common::TOTAL_FRAMES,
        "Σ2790 − 6×15 必须回 2700"
    );
    assert!(
        (specs[6].duration - common::scene_duration(310)).abs() < 1e-6,
        "S7 ClipSpec 时长必须等于 scene_duration(310)"
    );

    // ---- 3) 流式渲染 + 转场拼接 + Clip 级淡变 → 逐帧 JPEG 编码 ----
    // 走 render_short_concat_faded_frames_each 流式回调而非一次性驻留
    // 2700 帧裸 RGBA8（≈5.6GB）一次性驻留会 OOM：
    // 峰值仅为「单段帧序列 + 转场驻留尾 15 帧 + JPEG 帧序列 ≈0.4GB」。
    // 编码参数（quality 82 / grayscale=false）与 prism-mux
    // write_avi_mjpeg_audio 内部路径逐字节同参，保证成片与批式管线一致。
    let mut jpegs: Vec<Vec<u8>> = Vec::with_capacity(common::TOTAL_FRAMES);
    // v3-B 后处理链接线：2D 场景帧（S1–S6 及其间转场，输出帧索引 <
    // [`film_postfx::s7_body_start_frame`]）在 JPEG 编码前过
    // 「bloom → grade → vignette」链（显示域、纯函数、不改变帧尺寸，
    // alpha 恒 255）——「画质三件套」之美术加工落地。
    // S7 体（帧索引 ≥ 2405）**不过**本链：其画质由 scene3d::post_process
    // 内部 bloom 自含（预渲染成品位图只能在内含链里加工），否则高光经
    // 两次泛光——塑料晕，恰是本次要消灭的缺陷。S6→S7 的 15 帧过渡区
    // （94–100% 为 S6 内容）按 2D 侧过链，避免边界单帧跳变。
    let s7_start = film_postfx::s7_body_start_frame() as usize;
    render_short_concat_faded_frames_each(&specs, CLIP_FADE_IN, CLIP_FADE_OUT, |frame| {
        let i = jpegs.len();
        assert_eq!(
            frame.len(),
            WIDTH as usize * HEIGHT as usize * 4,
            "帧 {i} 必须为 RGBA8 行主序"
        );
        // 2D 侧帧过后处理链（返回新缓冲）；S7 体帧原样直投。
        let chained: Vec<u8>;
        let src: &[u8] = if i < s7_start {
            chained = film_postfx::post_chain(frame, WIDTH, HEIGHT);
            assert_eq!(
                chained.len(),
                frame.len(),
                "后处理链不得改变帧尺寸（帧 {i}）"
            );
            &chained
        } else {
            frame
        };
        let j = encode_jpeg_color(
            src,
            WIDTH,
            HEIGHT,
            JpegConfig { quality: JPEG_QUALITY, grayscale: false },
        )
        .unwrap_or_else(|e| panic!("帧 {i} JPEG 编码失败：{e}"));
        jpegs.push(j);
    })
    .expect("render_short_concat_faded_frames_each: 参数非法（specs < 2 / 画幅不一致 / 段长不足 2×转场）");
    assert_eq!(
        jpegs.len(),
        common::TOTAL_FRAMES,
        "总帧数必须 = Σ2790 − 6×转场重叠 = 2700（90.000s @30fps）"
    );

    // ---- 4) BGM：90s 立体声 44100 四声部实时合成 ----
    let bgm = bgm::synth_bgm(90.0, BGM_SR);
    assert_eq!(bgm.frames(), 3_969_000, "BGM 样本数必须精确 = 90s×44100");
    assert_eq!(bgm.channels(), 2, "BGM 必须为立体声（母带展宽：L/R 微偏移 + 混响抽头）");
    assert!(bgm.peak() <= 1.0, "母带峰值不得削波，实测 {}", bgm.peak());
    // 首 / 尾 0.05s 幅度 < 0.01（母带淡入 1.5s / 淡出 1.2s 到位）。
    let edge = (0.05 * BGM_SR as f32) as usize;
    let head = (0..edge)
        .map(|i| bgm.sample(i, 0).unwrap_or(0.0).abs())
        .fold(0.0_f32, f32::max);
    let tail = (0..edge)
        .map(|i| bgm.sample(bgm.frames() - 1 - i, 0).unwrap_or(0.0).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        head < 0.01 && tail < 0.01,
        "BGM 首尾 0.05s 幅度必须 <0.01（淡变端点），实测 head={head} tail={tail}"
    );

    // ---- 5) 音画封装为标准 MJPEG-in-AVI 双流（预编码 JPEG 帧入口）+ 落盘 ----
    let avi = write_avi_mjpeg_audio_frames(&jpegs, WIDTH, HEIGHT, FPS, &bgm)
        .expect("AVI 封装失败");
    std::fs::write(OUT_PATH, &avi).expect("写入 AVI 失败");

    println!(
        "calculus_film_v2 完成：{} 帧 @{}×{} {}fps（{:.3}s；7 场景 + 6×0.5s Fade + 片头 {:.1}s / 片尾 {:.1}s 淡变）",
        jpegs.len(),
        WIDTH,
        HEIGHT,
        FPS,
        jpegs.len() as f32 / FPS as f32,
        CLIP_FADE_IN,
        CLIP_FADE_OUT,
    );
    println!(
        "  BGM：{} 帧 / {:.2}s（立体声 {}Hz，120BPM × 45 小节四声部，母带峰值 {:.3}）",
        bgm.frames(),
        bgm.duration_secs(),
        bgm.sample_rate(),
        bgm.peak(),
    );
    println!(
        "  输出：{}（{} 字节 ≈ {:.1} MB）",
        OUT_PATH,
        avi.len(),
        avi.len() as f64 / 1e6
    );
}

// ---------------------------------------------------------------------------
// 预览模式（v3-B 交付项：PRISM_PREVIEW_S2=N）
// ---------------------------------------------------------------------------

/// 预览：只渲染 S2 内容窗内均匀 N 帧，过后处理链写 PNG 后退出。
///
/// 截断安全性（与正式渲染逐位一致的论证）：S2 全部轨的淡出一律用
/// **绝对锚点** `out_at`（9.0 / 9.5s），唯一无绝对锚点的底部打字机
/// `out_sec = 0.0`（无淡出）——`fade_factor` / `out_start_of` 均以绝对
/// 锚点优先，故把 ClipSpec `duration` 截断到 [`PREVIEW_S2_FRAMES`] 帧
/// 对窗内帧零影响；哑场景（30 帧纯黑）满足流式入口「每段 ≥ 2×转场」
/// 下限，其与 S2 的 0.5s 叠化区落在帧 256 起，被内容窗避开。
fn run_preview_s2(n: usize) {
    assert!(n >= 1, "PRISM_PREVIEW_S2 取样帧数须 >= 1");
    std::fs::create_dir_all(PREVIEW_DIR).expect("创建 /tmp/v3_preview 失败");
    let mut s2 = scene_s2();
    s2.duration = common::scene_duration(PREVIEW_S2_FRAMES);
    let specs = vec![s2, preview_dummy_spec()];
    // 内容窗内均匀取 n 帧（含两端）。
    let (lo, hi) = PREVIEW_WINDOW;
    let targets: Vec<usize> = if n == 1 {
        vec![lo]
    } else {
        (0..n).map(|k| lo + (hi - lo) * k / (n - 1)).collect()
    };
    let mut idx = 0usize;
    render_short_concat_faded_frames_each(&specs, CLIP_FADE_IN, CLIP_FADE_OUT, |frame| {
        if let Some(k) = targets.iter().position(|&t| t == idx) {
            assert_eq!(frame.len(), WIDTH as usize * HEIGHT as usize * 4);
            // 与正式路径同参同链（纯函数 ⇒ 预览即正片画质）。
            let graded = film_postfx::post_chain(frame, WIDTH, HEIGHT);
            let png = film_postfx::encode_png_rgba8(&graded, WIDTH, HEIGHT);
            let t = idx as f32 / FPS as f32;
            let path = format!("{PREVIEW_DIR}/s2_f{idx:04}_t{t:.2}.png");
            std::fs::write(&path, &png).unwrap_or_else(|e| panic!("写 {path} 失败：{e}"));
            println!(
                "[preview] 取样 {}/{}：帧 {idx}（t={t:.2}s）→ {path}（{} B，含 bloom/grade/vignette）",
                k + 1,
                targets.len(),
                png.len()
            );
        }
        idx += 1;
    })
    .expect("预览渲染：参数非法（specs < 2 / 画幅不一致 / 段长不足 2×转场）");
    println!(
        "[preview] 完成：{} 帧已落盘 {PREVIEW_DIR}（q=1 预览档：不缩放、不量化，PNG 无损直出）",
        targets.len()
    );
}

/// 预览用哑场景（30 帧纯黑、无轨；满足流式入口段长下限 2×15）。
fn preview_dummy_spec() -> ClipSpec {
    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: common::scene_duration(30),
        background: Background::Gradient {
            top: common::hex_to_linear([0x00, 0x00, 0x00]),
            bottom: common::hex_to_linear([0x00, 0x00, 0x00]),
        },
        jpeg_quality: JPEG_QUALITY,
        tracks: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// S1 标题·公式符号矢量入场（248 帧 = 8.267s，DarkGlow）
// ---------------------------------------------------------------------------

/// S1 三符号规格（PRD §4-S1）：(种类, 目标高度 px, 锚点中心, 颜色,
/// 入场秒, 描画秒)。高度经 [`symbol_contours`] 两遍法机械归一化。
const S1_SYMBOLS: [(SymbolKind, f32, [f32; 2], [f32; 4], f32, f32); 3] = [
    // lim（金，stroke 7px，draw 1.0s，中心 (330,210)）。
    (SymbolKind::Lim, 64.0, [330.0, 210.0], GOLD_RGBA, 1.5, 1.0),
    // dy/dx（青，draw 1.0s，(560,250)）。
    (SymbolKind::DyDx, 58.0, [560.0, 250.0], CURVE_RGBA, 2.5, 1.0),
    // ∫（白，大号 draw 1.2s，(790,220)，高 130px）。
    (SymbolKind::Integral, 130.0, [790.0, 220.0], WHITE_RGBA, 3.5, 1.2),
];

/// S1 呼吸变换（5.0–6.0，CubicInOut；PRD「三符号同步呼吸 + 缓慢上浮
/// 8px」）：scale 1.00→1.05→1.00（三角形）、上浮 0→8px（单向，终点
/// 高 8px）。区间外恒等变换（1.0, 0.0）。
fn s1_breathe(t: f32) -> (f32, f32) {
    let u = ((t - 5.0) / 1.0).clamp(0.0, 1.0);
    let e = EasingKind::CubicInOut.apply(u);
    let tri = if e <= 0.5 { e * 2.0 } else { (1.0 - e) * 2.0 };
    (1.0 + 0.05 * tri, 8.0 * e)
}

/// S1（248 帧 = 8.267s，DarkGlow）。
pub fn scene_s1() -> ClipSpec {
    let dur = common::scene_duration(248);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 小标签（0.5 淡入 0.5s，y=120，居中）----
    tracks.push(Track::Static(StaticText {
        content: String::from("PRISM ENGINE 出品"),
        font: FontKind::Regular,
        size: 20.0,
        color: TEXT_RGBA,
        at: 0.5,
        in_sec: 0.5,
        letter_spacing: 3.0,
        y: 120.0,
        out_sec: 0.0,
        out_at: None,
        center_x: Some(480.0),
    }));

    // ---- 中心微光点（0.5–1.5 脉冲一次：淡入 0.35s → out_at 锚定淡出
    // 0.45s；ShapeTrack 无缩放动画，脉冲以 alpha 包络表达）----
    tracks.push(Track::Shape(ShapeTrack {
        kind: ShapeKind::Ellipse,
        x: 480.0 - 3.5,
        y: 215.0 - 3.5,
        w: 7.0,
        h: 7.0,
        color: WHITE_RGBA,
        at: 0.5,
        in_sec: 0.35,
        out_sec: 0.45,
        out_at: Some(1.5),
    }));

    // ---- 三符号矢量描画（每轮廓一条 DynPath）----
    // draw_sec 承担「描画」入场（引擎按累计长度截断）；5.0–6.0 呼吸
    // 逐帧仿射（缩放关于锚点 + 上浮）；7.5–8.0 out_at 锚定淡出。
    for &(kind, target_h, anchor, color, at, draw_sec) in &S1_SYMBOLS {
        for contour in symbol_contours(kind, target_h, anchor) {
            tracks.push(Track::DynPath(DynPathTrack {
                geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                    let (s, rise) = s1_breathe(t);
                    contour
                        .iter()
                        .map(|p| {
                            [
                                anchor[0] + (p[0] - anchor[0]) * s,
                                anchor[1] + (p[1] - anchor[1]) * s - rise,
                            ]
                        })
                        .collect()
                }) as GeomFn,
                color,
                width: 7.0,
                fill: false,
                at,
                in_sec: 0.2,
                out_sec: 0.5,
                out_at: Some(7.5),
                draw_sec,
                closed: false,
            }));
        }
    }

    // ---- 主标题（6.0 kinetic 逐字入场，Bold 96px，自符号阵下方升起）----
    tracks.push(Track::Kinetic(KineticText {
        content: String::from("微积分"),
        font: FontKind::Bold,
        size: 96.0,
        color: WHITE_RGBA,
        at: 6.0,
        stagger: 0.12,
        per_char: 0.5,
        scale_from: 0.72,
        rise_px: 22.0,
        easing: EasingKind::CubicOut,
        y: 380.0,
        in_sec: 0.4,
        out_sec: 0.0,
        out_at: None,
        center_x: None,
    }));

    // ---- 副题（6.5 打字机 12cps，y=460，闪烁光标）----
    tracks.push(typewriter(
        "从割线，到光追",
        28.0,
        TEXT_RGBA,
        6.5,
        460.0,
        12.0,
        true,
        None,
        0.0,
    ));

    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: dur,
        background: Background::DarkGlow,
        jpeg_quality: JPEG_QUALITY,
        tracks,
    }
}

// ---------------------------------------------------------------------------
// S2 坐标系亮相 + 相机缓推（308 帧 = 10.267s，Aurora）
// ---------------------------------------------------------------------------

/// S2 基准视口（PRD §4-S2 布局）：x∈[−4,4]、y∈[−2.4,2.4]、85px/unit、
/// 绘图区 140–820 × 66–474（半区 340×204 关于 (480,270)）。
fn s2_base_viewport() -> common::Viewport {
    common::Viewport::fit_plot(-4.0, 4.0, -2.4, 2.4, 480.0, 270.0, 340.0, 204.0)
}

/// S2 相机缓推视口（**逐帧**解算）：scale 1.00→1.08（CubicInOut，
/// [4.5, 6.5]）关于绘图区中心。数学域随缩放收窄、px_per_unit 同步
/// 放大——全部几何经本视口 `to_px` 每帧重算（PRD「全部网格/曲线经
/// DynPath 的 viewport 缩放实现；不依赖 postfx::Warp」）。
fn s2_viewport_at(t: f32) -> common::Viewport {
    let u = ((t - 4.5) / 2.0).clamp(0.0, 1.0);
    let s = 1.0 + 0.08 * EasingKind::CubicInOut.apply(u);
    common::Viewport::fit_plot(
        -4.0 / s,
        4.0 / s,
        -2.4 / s,
        2.4 / s,
        480.0,
        270.0,
        340.0 * s,
        204.0 * s,
    )
}

/// S2 Chevron 箭头三点（与 `axes.rs::push_chevron` 同构）：`(bx, by)`
/// 为箭杆反方向单位向量，两翼沿其垂直方向展开。
fn s2_chevron(tip: [f32; 2], bx: f32, by: f32) -> Vec<[f32; 2]> {
    let (px, py) = (-by, bx); // 垂直方向
    let back = axes::ARROW_BACK;
    let half = axes::ARROW_HALF_W;
    vec![
        [
            tip[0] + bx * back + px * half,
            tip[1] + by * back + py * half,
        ],
        tip,
        [
            tip[0] + bx * back - px * half,
            tip[1] + by * back - py * half,
        ],
    ]
}

/// S2（308 帧 = 10.267s，Aurora）。
pub fn scene_s2() -> ClipSpec {
    let dur = common::scene_duration(308);
    let mut tracks: Vec<Track> = Vec::new();

    // ---- 网格：域内 0.5 步长满线（17 竖 + 9 横）----
    // PRD beat 表「13 竖 + 9 横」与其布局域 [−4,4]/0.5 不自洽（17 竖）；
    // 取布局域自洽值。stagger 0.06s（PRD 0.08s 按 22 线设计）保证
    // 2.5s 前完成淡入。每条一 DynPath：端点取**当帧**视口域边——相机
    // 缓推时网格始终铺满绘图区。
    let mut grid_idx = 0usize;
    for i in 0..17 {
        let x = -4.0 + 0.5 * i as f32;
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                let vp = s2_viewport_at(t);
                vec![vp.to_px(x, vp.y0), vp.to_px(x, vp.y1)]
            }) as GeomFn,
            color: GRID_RGBA,
            width: axes::GRID_LINE_W,
            fill: false,
            at: 0.5 + 0.06 * grid_idx as f32,
            in_sec: 0.3,
            out_sec: 0.5,
            out_at: Some(9.5),
            draw_sec: 0.0,
            closed: false,
        }));
        grid_idx += 1;
    }
    for j in 0..9 {
        let y = -2.0 + 0.5 * j as f32;
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                let vp = s2_viewport_at(t);
                vec![vp.to_px(vp.x0, y), vp.to_px(vp.x1, y)]
            }) as GeomFn,
            color: GRID_RGBA,
            width: axes::GRID_LINE_W,
            fill: false,
            at: 0.5 + 0.06 * grid_idx as f32,
            in_sec: 0.3,
            out_sec: 0.5,
            out_at: Some(9.5),
            draw_sec: 0.0,
            closed: false,
        }));
        grid_idx += 1;
    }

    // ---- 主轴（2.5–3.5 描画 1.0s，白 0.9）----
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(|t: f32| -> Vec<[f32; 2]> {
            let vp = s2_viewport_at(t);
            vec![vp.to_px(vp.x0, 0.0), vp.to_px(vp.x1, 0.0)]
        }) as GeomFn,
        color: AXIS_RGBA,
        width: axes::AXIS_LINE_W,
        fill: false,
        at: 2.5,
        in_sec: 0.5,
        out_sec: 0.5,
        out_at: Some(9.5),
        draw_sec: 1.0,
        closed: false,
    }));
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(|t: f32| -> Vec<[f32; 2]> {
            let vp = s2_viewport_at(t);
            vec![vp.to_px(0.0, vp.y0), vp.to_px(0.0, vp.y1)]
        }) as GeomFn,
        color: AXIS_RGBA,
        width: axes::AXIS_LINE_W,
        fill: false,
        at: 2.5,
        in_sec: 0.5,
        out_sec: 0.5,
        out_at: Some(9.5),
        draw_sec: 1.0,
        closed: false,
    }));

    // ---- 四枚箭头（2.5 淡入 0.4s + 描画 0.6s；锚在当帧域端）----
    let arrow_specs: [([f32; 2], f32, f32); 4] = [
        // 右端 →（数学 +x）
        ([f32::NAN, 0.0], -1.0, 0.0),
        // 左端 ←（数学 −x）
        ([f32::NAN, 0.0], 1.0, 0.0),
        // 上端 ↑（数学 +y；设备 y 向下，反方向为 +y）
        ([0.0, f32::NAN], 0.0, 1.0),
        // 下端 ↓（数学 −y）
        ([0.0, f32::NAN], 0.0, -1.0),
    ];
    for (tip_hint, bx, by) in arrow_specs {
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                let vp = s2_viewport_at(t);
                // tip_hint 的 NaN 分量由当帧视口域端解算（见 arrow_specs）。
                let tip = if tip_hint[0].is_nan() {
                    if bx < 0.0 {
                        vp.to_px(vp.x1, 0.0) // 右端
                    } else {
                        vp.to_px(vp.x0, 0.0) // 左端
                    }
                } else if by < 0.0 {
                    vp.to_px(0.0, vp.y0) // 下端（设备 +y）
                } else {
                    vp.to_px(0.0, vp.y1) // 上端
                };
                s2_chevron(tip, bx, by)
            }) as GeomFn,
            color: AXIS_RGBA,
            width: axes::AXIS_LINE_W,
            fill: false,
            at: 2.5,
            in_sec: 0.4,
            out_sec: 0.5,
            out_at: Some(9.5),
            draw_sec: 0.6,
            closed: false,
        }));
    }

    // ---- 整数刻度（3.5 起 stagger 0.03s 淡入；x 轴 9 枚 + y 轴 5 枚）----
    let mut tick_idx = 0usize;
    for i in 0..9 {
        let x = -4.0 + i as f32;
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                let vp = s2_viewport_at(t);
                let a = vp.to_px(x, 0.0);
                vec![
                    [a[0], a[1] - axes::TICK_HALF_LEN],
                    [a[0], a[1] + axes::TICK_HALF_LEN],
                ]
            }) as GeomFn,
            color: TICK_RGBA,
            width: axes::GRID_LINE_W,
            fill: false,
            at: 3.5 + 0.03 * tick_idx as f32,
            in_sec: 0.2,
            out_sec: 0.5,
            out_at: Some(9.5),
            draw_sec: 0.0,
            closed: false,
        }));
        tick_idx += 1;
    }
    for j in 0..5 {
        let y = -2.0 + j as f32;
        tracks.push(Track::DynPath(DynPathTrack {
            geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
                let vp = s2_viewport_at(t);
                let a = vp.to_px(0.0, y);
                vec![
                    [a[0] - axes::TICK_HALF_LEN, a[1]],
                    [a[0] + axes::TICK_HALF_LEN, a[1]],
                ]
            }) as GeomFn,
            color: TICK_RGBA,
            width: axes::GRID_LINE_W,
            fill: false,
            at: 3.5 + 0.03 * tick_idx as f32,
            in_sec: 0.2,
            out_sec: 0.5,
            out_at: Some(9.5),
            draw_sec: 0.0,
            closed: false,
        }));
        tick_idx += 1;
    }

    // ---- 轴标签 x / y（3.5–4.5 淡入；4.5 随相机启动退役）----
    // StaticText 无逐帧重定位能力：若全程贴附，缓推 8% 会使标签与轴端
    // 脱钩漂移（右端最多 27px）。衡衷取「缓推前退役」——标签完成点名
    // 职责，公式标签（6.5，落于终态峰位）接管指向。
    let base = s2_base_viewport();
    let x_tip = base.to_px(4.0, 0.0);
    tracks.push(Track::Static(StaticText {
        content: String::from("x"),
        font: FontKind::Bold,
        size: axes::AXIS_LABEL_SIZE,
        color: ctx(TEXT_RGBA),
        at: 3.5,
        in_sec: 0.5,
        letter_spacing: 0.0,
        y: x_tip[1] - 12.0,
        out_sec: 0.5,
        out_at: Some(4.5),
        center_x: Some(x_tip[0] + 20.0),
    }));
    let y_tip = base.to_px(0.0, 2.4);
    tracks.push(Track::Static(StaticText {
        content: String::from("y"),
        font: FontKind::Bold,
        size: axes::AXIS_LABEL_SIZE,
        color: ctx(TEXT_RGBA),
        at: 3.5,
        in_sec: 0.5,
        letter_spacing: 0.0,
        y: y_tip[1] + 12.0,
        out_sec: 0.5,
        out_at: Some(4.5),
        center_x: Some(y_tip[0] + 20.0),
    }));

    // ---- y = sin x 曲线（4.5–6.5 draw 2.0s，49 点，青 3.5px）----
    // 与相机缓推同窗启动：曲线属「世界」，随视口缩放一同推近。
    let xs: Vec<f32> = (0..49).map(|i| -4.0 + 8.0 * i as f32 / 48.0).collect();
    tracks.push(Track::DynPath(DynPathTrack {
        geom: Arc::new(move |t: f32| -> Vec<[f32; 2]> {
            let vp = s2_viewport_at(t);
            xs.iter().map(|&x| vp.to_px(x, x.sin())).collect()
        }) as GeomFn,
        color: CURVE_RGBA,
        width: 3.5,
        fill: false,
        at: 4.5,
        in_sec: 0.3,
        out_sec: 0.5,
        out_at: Some(9.0),
        draw_sec: 2.0,
        closed: false,
    }));

    // ---- 公式标签「y = sin x」（6.5 淡入，落于终态第一峰值旁）----
    // 峰位 x=π/2、y=1 经终态视口（scale 1.08）映射：(624, 178)；标签
    // 置于峰右上方 (664, 150)——出现时缓推已结束（6.5），位置精确。
    let mut formula = symbols::assemble_formula(
        &[FormulaPart::Text(
            String::from("y = sin x"),
            FontKind::Regular,
            28.0,
            CURVE_RGBA,
        )],
        664.0,
        150.0,
    );
    for tr in formula.iter_mut() {
        match tr {
            Track::Static(s) => {
                s.at = 6.5;
                s.in_sec = 0.5;
                s.out_sec = 0.5;
                s.out_at = Some(9.0);
            }
            Track::Path(p) => {
                p.at = 6.5;
                p.in_sec = 0.5;
                p.out_sec = 0.5;
                p.out_at = Some(9.0);
            }
            _ => {}
        }
    }
    tracks.extend(formula);

    // ---- 开幕目标字幕（0.8 打字机 12cps，2.8 淡出；本节目标一句话；
    // [节奏重剪] 上下文层 alpha 0.50）----
    tracks.push(typewriter(
        "先认识舞台：直角坐标系",
        24.0,
        TEXT_CTX_RGBA,
        0.8,
        500.0,
        12.0,
        false,
        Some(2.8),
        0.4,
    ));

    // ---- 底部文案（8.0 打字机 12cps，y=500；[节奏重剪] 上下文层 alpha）----
    tracks.push(typewriter(
        "每一条光滑曲线，都是变化的故事",
        26.0,
        TEXT_CTX_RGBA,
        8.0,
        500.0,
        12.0,
        false,
        None,
        0.0,
    ));

    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: dur,
        background: Background::Aurora,
        jpeg_quality: JPEG_QUALITY,
        tracks,
    }
}

// ---------------------------------------------------------------------------
// S7 片尾 flex·3D 光追收尾（310 帧 = 10.333s）
// ---------------------------------------------------------------------------

/// S7（310 帧）：`Track::Sequence` 承载 scene3d 预渲染的 310 源帧
/// （30fps → 片上 30fps 逐帧对应，310/30 = 10.333s 精确）；
/// 文案 / 切平面公式 / 切点金点 / 落版为 2D 叠加轨。淡黑尾（帧 ≥264
/// 全黑）由预渲染帧承担，落版文字叠在黑底之上。
pub fn scene_s7(seq: scene3d::SequenceFrames) -> ClipSpec {
    let dur = common::scene_duration(310);
    let mut tracks: Vec<Track> = Vec::new();

    tracks.push(Track::Sequence(SequenceTrack {
        frames: seq.frames,
        width: seq.width,
        height: seq.height,
        src_fps: scene3d::SRC_FPS,
        at: 0.0,
        in_sec: 0.5,
        out_sec: 0.0,
        out_at: None,
    }));

    // ---- 文案 1（1.0–4.0）：曲线旋转成面 ----
    tracks.push(typewriter(
        "同一条曲线，旋转成一个面",
        26.0,
        TEXT_RGBA,
        1.0,
        500.0,
        12.0,
        false,
        Some(4.0),
        0.3,
    ));

    // ---- 切平面公式（4.0–5.0，画面中央偏上）+ 切点 P 金点脉冲 ----
    // 平面 z = 1.6x + 1.6y − 1.28 与抛物面 z = x²+y² 相切于
    // P = (0.8, 0.8, 1.28)（scene3d.rs 玻璃切平面定义同参）。
    let mut formula = symbols::assemble_formula(
        &[
            FormulaPart::Text(String::from("z = 1.6x + 1.6y "), FontKind::Regular, 30.0, GOLD_RGBA),
            FormulaPart::Sym(SymbolKind::Minus, 30.0, GOLD_RGBA),
            FormulaPart::Text(String::from(" 1.28"), FontKind::Regular, 30.0, GOLD_RGBA),
        ],
        480.0,
        110.0,
    );
    for tr in formula.iter_mut() {
        match tr {
            Track::Static(s) => {
                s.at = 4.0;
                s.in_sec = 0.4;
                s.out_sec = 0.3;
                s.out_at = Some(5.0);
            }
            Track::Path(p) => {
                p.at = 4.0;
                p.in_sec = 0.4;
                p.out_sec = 0.3;
                p.out_at = Some(5.0);
            }
            _ => {}
        }
    }
    tracks.extend(formula);

    // 切点 P 屏位：camera_at(4.5)（公式窗中点）精确投影——与渲染器
    // `Camera::pixel_dir` 逆映射同参，非估计值。
    let p_px = project_world(
        &scene3d::camera_at(4.5),
        Vec3::new(0.8, 0.8, 1.28),
        (WIDTH, HEIGHT),
    )
    .unwrap_or([480.0, 270.0]);
    tracks.push(Track::Shape(ShapeTrack {
        kind: ShapeKind::Ellipse,
        x: p_px[0] - 5.0,
        y: p_px[1] - 5.0,
        w: 10.0,
        h: 10.0,
        color: GOLD_RGBA,
        at: 4.0,
        in_sec: 0.3,
        out_sec: 0.4,
        out_at: Some(5.0),
    }));

    // ---- 文案 2 / 3（5.0–6.5 / 6.5–8.0）：切平面与点题 ----
    tracks.push(typewriter(
        "切平面——切线在两个方向上的推广",
        26.0,
        TEXT_RGBA,
        5.0,
        500.0,
        12.0,
        false,
        Some(6.5),
        0.3,
    ));
    tracks.push(typewriter(
        "从割线到光追，同一件事：变化",
        26.0,
        TEXT_RGBA,
        6.5,
        500.0,
        12.0,
        false,
        Some(8.0),
        0.3,
    ));

    // ---- 落版两行（8.8 淡入 0.5s；132/15 = 8.8s 起序列已全黑）----
    // 片尾全局 1.5s 淡变（CLIP_FADE_OUT）承担收尾，故不设 out_at。
    tracks.push(Track::Static(StaticText {
        content: String::from("Made with Prism Engine"),
        font: FontKind::Bold,
        size: 44.0,
        color: TEXT_RGBA,
        at: 8.8,
        in_sec: 0.5,
        letter_spacing: 1.0,
        y: 250.0,
        out_sec: 0.0,
        out_at: None,
        center_x: None,
    }));
    tracks.push(Track::Static(StaticText {
        content: String::from("Zero dependencies. Video as code."),
        font: FontKind::Regular,
        size: 20.0,
        color: GOLD_RGBA,
        at: 8.8,
        in_sec: 0.5,
        letter_spacing: 1.0,
        y: 305.0,
        out_sec: 0.0,
        out_at: None,
        center_x: None,
    }));

    ClipSpec {
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        duration: dur,
        background: Background::Gradient {
            top: common::hex_to_linear([0x00, 0x00, 0x00]),
            bottom: common::hex_to_linear([0x00, 0x00, 0x00]),
        },
        jpeg_quality: JPEG_QUALITY,
        tracks,
    }
}

// ---------------------------------------------------------------------------
// 共享辅助
// ---------------------------------------------------------------------------

/// 打字机轨（S1 副题 / S2·S7 文案共用）：`cps` 字符/秒、可选闪烁光标
/// （2Hz）、可选绝对淡出锚点（顺序消息退场）；水平居中 (480)。
#[allow(clippy::too_many_arguments)]
fn typewriter(
    content: &str,
    size: f32,
    color: [f32; 4],
    at: f32,
    y: f32,
    cps: f32,
    cursor: bool,
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
        cursor,
        cursor_blink_hz: 2.0,
        in_sec: 0.3,
        out_sec,
        out_at,
        letter_spacing: 1.0,
        center_x: Some(480.0),
    })
}

/// 取符号 `kind` 的笔画轮廓（设备像素）：**两遍法**按目标高度归一化
/// （PRD「∫ 高 130px」等尺寸机械保证）、bbox 中心对齐到锚点。
///
/// 第一遍以参考尺寸 `S1_SYM_REF` 测量局部 bbox（y 向上），解出精确
/// 尺寸 `REF·target_h/h` 后第二遍重建；局部 → 设备经
/// `(x, y) → (x + tx, −y + ty)`（与 `symbols::assemble_formula` 的
/// 符号变换同构）。`<2` 点的轮廓丢弃（渲染层跳过）。
fn symbol_contours(kind: SymbolKind, target_h: f32, anchor: [f32; 2]) -> Vec<Vec<[f32; 2]>> {
    /// 参考尺寸（px）。
    const REF: f32 = 100.0;
    if !target_h.is_finite() || target_h <= 0.0 {
        return Vec::new();
    }
    // 局部坐标（恒等仿射）展平。
    let identity = Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };
    let flatten_local = |size: f32| -> Vec<Vec<[f32; 2]>> {
        symbols::flatten_strokes(&symbols::symbol_path(kind, size), &identity)
            .into_iter()
            .filter(|c| c.len() >= 2)
            .collect()
    };
    let base = flatten_local(REF);
    let (mut x_lo, mut x_hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut y_lo, mut y_hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for c in &base {
        for p in c {
            x_lo = x_lo.min(p[0]);
            x_hi = x_hi.max(p[0]);
            y_lo = y_lo.min(p[1]);
            y_hi = y_hi.max(p[1]);
        }
    }
    let h_ref = y_hi - y_lo;
    if !(h_ref > 0.0) {
        return Vec::new();
    }
    let size = REF * target_h / h_ref;
    // bbox 中心随尺寸等比缩放：k = size/REF。
    let k = size / REF;
    let tx = anchor[0] - (x_lo + x_hi) * 0.5 * k;
    let ty = anchor[1] + (y_lo + y_hi) * 0.5 * k;
    let tf = Transform { a: 1.0, b: 0.0, c: 0.0, d: -1.0, tx, ty };
    symbols::flatten_strokes(&symbols::symbol_path(kind, size), &tf)
        .into_iter()
        .filter(|c| c.len() >= 2)
        .collect()
}

/// 世界点 → 设备像素：`Camera::pixel_dir` 的**逆映射**（同参同约定——
/// 针孔相机、竖直 fov 锚定、水平按宽高比展开、NDC y 向上）。
///
/// 用于 S7 切点 P 的 2D 叠加轨屏位解算：与 3D 渲染器共用同一相机
/// 模型，非估计值。点在相机后方（`zc <= 0`）或投影非有限时 `None`。
fn project_world(cam: &Camera, p: Vec3, res: (u32, u32)) -> Option<[f32; 2]> {
    let v = p - cam.eye;
    let zc = v.dot(&cam.forward);
    if !(zc > 1e-6) {
        return None;
    }
    let xc = v.dot(&cam.right);
    let yc = v.dot(&cam.up);
    let aspect = res.0 as f32 / res.1 as f32;
    let u = (xc / zc) / (cam.tan_half_fov * aspect);
    let vn = (yc / zc) / cam.tan_half_fov;
    if !u.is_finite() || !vn.is_finite() {
        return None;
    }
    Some([
        (u + 1.0) * 0.5 * res.0 as f32,
        (1.0 - vn) * 0.5 * res.1 as f32,
    ])
}

// ---------------------------------------------------------------------------
// T04 资产层装配期自检（架构 §7.5：example 装配期灾难性错误快速失败；
// 同时保证全部 pub 资产入口在非测试构建中被消费，无 dead_code）
// ---------------------------------------------------------------------------

/// 帧网格：Σ2790 − 6×15 = 2700 = 90.000s；逐场景 `scene_duration` 的
/// `ceil(fps·duration)` 必须精确回落到场景帧数（浮点不跳帧）。
fn check_frame_grid() {
    assert_eq!(WIDTH, common::WIDTH);
    assert_eq!(HEIGHT, common::HEIGHT);
    assert_eq!(FPS, common::FPS);
    assert_eq!(common::SCENE_FRAMES, [248, 308, 496, 496, 496, 436, 310]);
    assert_eq!(common::TRANS_COUNT, 6);
    assert_eq!(common::TRANS_FRAMES, 15);
    assert!((common::CONCAT_FADED_TRANS_SEC - 0.5).abs() < 1e-6);
    let mut sum: usize = 0;
    for &n in common::SCENE_FRAMES.iter() {
        let d = common::scene_duration(n);
        assert_eq!(
            (common::FPS as f32 * d).ceil(),
            n as f32,
            "场景帧数 {n} 的 duration 必须 ceil 回 {n}"
        );
        sum += n as usize;
    }
    assert_eq!(sum, 2790);
    assert_eq!(sum - common::TRANS_COUNT * common::TRANS_FRAMES, common::TOTAL_FRAMES);
    assert_eq!(common::TOTAL_FRAMES, 2700);
    assert_eq!(common::TOTAL_FRAMES as f32 / common::FPS as f32, 90.0);
    assert_eq!(common::TOTAL_DURATION_SEC, 90.0);
}

/// 颜色：两个换算函数职责边界（§7.2）+ 调色板 / 背景渐变有限域。
fn check_colors() {
    // #35D9F2 → 显示值 (0.208, 0.851, 0.949)（÷255）。
    let c = common::hex_to_display(common::HEX_CURVE);
    assert!((c[0] - 0.208).abs() < 1e-3 && (c[1] - 0.851).abs() < 1e-3 && (c[2] - 0.949).abs() < 1e-3);
    // hex_to_linear：仅 S7 3D 原料；此处抽验有限域与端点。
    let lin = common::hex_to_linear(common::HEX_GOLD);
    assert!(lin.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    // 2D 调色板与场景背景渐变：全部有限、落 [0,1]。
    for rgba in common::PALETTE_2D.iter() {
        assert!(rgba.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }
    for (top, bottom) in common::BG_GRADIENTS.iter() {
        assert!(top.iter().chain(bottom.iter()).all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }
}

/// 视口：`fit_plot` 等比缩放 + `to_px` / `to_math` 往返（含 y 翻转）；
/// 直接构造与非法视口（不 panic、输出有限）。
fn check_viewport() {
    let vp = common::Viewport::fit_plot(-2.0, 2.0, -1.0, 3.0, 480.0, 270.0, 200.0, 150.0);
    assert!(vp.is_valid());
    let p = vp.to_px(1.0, 2.0);
    let m = vp.to_math(p[0], p[1]);
    assert!((m[0] - 1.0).abs() < 1e-3 && (m[1] - 2.0).abs() < 1e-3, "往返含 y 翻转");
    let direct = common::Viewport::new(-1.0, 1.0, -1.0, 1.0, 480.0, 270.0, 100.0);
    assert!(direct.is_valid());
    let bad = common::Viewport::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    assert!(!bad.is_valid());
    let q = bad.to_px(0.5, 0.5);
    assert!(q[0].is_finite() && q[1].is_finite(), "非法视口不 panic 且输出有限");
}

/// 坐标轴 / 采样：`axes_tracks` 全轨坐标有限且在画幅内；`sample_curve`
/// 64 点；`riemann_rects` 矩形结构完好。
fn check_axes() {
    let vp = common::Viewport::fit_plot(-3.0, 3.0, -2.0, 2.0, 480.0, 270.0, 400.0, 230.0);
    let tracks = axes::axes_tracks(&vp);
    assert!(!tracks.is_empty(), "坐标轴轨道不得为空");
    for t in &tracks {
        if let Track::Path(p) = t {
            assert!(!p.points.is_empty());
            for q in &p.points {
                assert!(q[0].is_finite() && q[1].is_finite(), "轴轨坐标必须有限");
                assert!(
                    (0.0..=common::WIDTH as f32).contains(&q[0])
                        && (0.0..=common::HEIGHT as f32).contains(&q[1]),
                    "轴轨坐标必须在画幅内：{q:?}"
                );
            }
        }
    }
    // 采样域取视口 x 域 [-3, 3]（闭区间裁剪，端点保留）⇒ 64 点全留。
    let pts = axes::sample_curve(f32::sin, -3.0, 3.0, 64, &vp);
    assert_eq!(pts.len(), 64, "视口覆盖全域时 64 点全留");
    let rects = axes::riemann_rects(4, |x| x * x, &vp);
    assert_eq!(rects.len(), 4, "四等分四矩形");
    for r in &rects {
        assert_eq!(r.len(), 4, "矩形四角");
        for q in r {
            assert!(q[0].is_finite() && q[1].is_finite());
        }
    }
}

/// 符号库 / 公式混排：全部种类非空；`f′(x)` 发射 2 段文本 + Prime 笔画。
fn check_symbols() {
    for kind in symbols::all_symbol_kinds() {
        let path = symbols::symbol_path(kind, 40.0);
        assert!(!path.subpaths.is_empty(), "{kind:?} 必须有笔画");
    }
    let tracks = symbols::assemble_formula(
        &[
            symbols::FormulaPart::Text(
                String::from("f"),
                FontKind::Bold,
                40.0,
                common::CURVE_RGBA,
            ),
            symbols::FormulaPart::Sym(symbols::SymbolKind::Prime, 40.0, common::GOLD_RGBA),
            symbols::FormulaPart::Text(
                String::from("(x)"),
                FontKind::Regular,
                40.0,
                common::TEXT_RGBA,
            ),
        ],
        480.0,
        300.0,
    );
    let n_static = tracks
        .iter()
        .filter(|t| matches!(t, Track::Static(_)))
        .count();
    let n_path = tracks.iter().filter(|t| matches!(t, Track::Path(_))).count();
    assert_eq!(n_static, 2, "两段文本 run");
    assert!(n_path >= 1, "Prime 至少一段笔画");
}

/// BGM：90.000s @44100 = 3,969,000 帧、立体声（母带展宽：L/R 微偏移 +
/// 混响抽头）、峰值 ≤1、淡变端点恒 0。
fn check_bgm() {
    let bgm = bgm::synth_bgm(bgm::BGM_DURATION_SEC, bgm::BGM_SR);
    assert_eq!(bgm.frames(), 3_969_000, "90.000s × 44100 帧");
    assert_eq!(
        bgm.channels(),
        2,
        "立体声（母带展宽：L/R 微偏移 + 混响抽头）"
    );
    assert!(bgm.peak() <= 1.0, "母带峰值不得削波，实测 {}", bgm.peak());
    assert_eq!(bgm.sample(0, 0), Some(0.0), "淡入端点样本恒 0");
    assert_eq!(
        bgm.sample(bgm.frames() - 1, 0),
        Some(0.0),
        "淡出端点样本恒 0"
    );
}

// ---------------------------------------------------------------------------
// v3-A/B 测试（文本截断语义 + 后处理链接线集成）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_author::render_short_concat_faded_frames;
    use prism_text::layout::{layout_text, Align, TextParams};

    /// v3-E 任务 0 回归：`check_bgm` 不是 `#[test]`（cargo test 平时跑不到），
    /// 而 v3-D 立体声母带定稿后其旧断言 `channels() == 1` 会在正式渲染
    /// 启动时 panic（哑雷）。本测试把该自检拉进 CI：立体声（母带展宽：
    /// L/R 微偏移 + 混响抽头）+ 长度 / 峰值 / 淡变端点全量断言。
    #[test]
    fn check_bgm_accepts_stereo_master() {
        check_bgm();
    }

    /// 截断边界用例：打字机逐字显示**只产整字**（无半字）、超宽
    /// 文本不 panic。
    ///
    /// 行为断言（经公开批式入口实测，非公式复述）：
    /// 1. 每帧墨迹右缘必须等于「前 n 个整字累计 advance」对应的笔位之一
    ///    （n = floor(t·cps).clamp(0, len)）——半字会落在两个笔位之间；
    /// 2. 墨迹右缘随帧单调不减（逐字显示只增不减）；
    /// 3. 全流程无 panic、帧尺寸恒定。
    ///
    /// 取样窗口：cps = 30（30fps 逐帧一字）⇒ 前 9 帧即可覆盖 n = 0..8
    /// 全字符集，第 9..14 帧 n 恒为 8（满幅稳定性取样）。只取前 15 帧：
    /// 批式转场 [`prism_author::render::concat_with_transition`] 的混合区
    /// 在输出帧 15..29（`a[0:15] + 15 帧 k=(i+1)/16 混合 + b[15:30]`），
    /// 其后文本被交叉溶解至黑色哑场景、整帧线性变暗，墨迹阈值口径不再
    /// 适用——本用例关注逐字机制本身，故限定在纯内容区。
    #[test]
    fn typewriter_never_emits_half_glyph() {
        // 微型画幅 + 8 字符 @30cps（30fps 逐帧一字，第 8 帧打完整句）。
        let (w, h) = (96u32, 32u32);
        let content = "ABCDEFGH";
        let cps = 30.0f32;
        let tw = Track::Typewriter(TypewriterText {
            content: String::from(content),
            font: FontKind::Regular,
            size: 20.0,
            color: TEXT_RGBA,
            at: 0.0,
            y: 22.0,
            cps,
            cursor: false,
            cursor_blink_hz: 0.0,
            in_sec: 0.0,
            out_sec: 0.0,
            out_at: None,
            letter_spacing: 0.0,
            center_x: Some(w as f32 * 0.5),
        });
        let spec = ClipSpec {
            width: w,
            height: h,
            fps: 30,
            duration: 30.0 / 30.0, // 30 帧 = 1.0s
            background: Background::Gradient {
                top: common::hex_to_linear([0x00, 0x00, 0x00]),
                bottom: common::hex_to_linear([0x00, 0x00, 0x00]),
            },
            jpeg_quality: 90,
            tracks: vec![tw],
        };
        let dummy = preview_dummy_spec_dim(w, h);
        let frames = render_short_concat_faded_frames(&[spec, dummy], 0.0, 0.0);
        // 批式入口默认 0.5s Fade 转场：trans = round(0.5×30) = 15 帧，
        // 输出 = a[0:15] + 15 帧混合 + b[15:30]，共 45 帧；纯内容区为
        // 输出帧 0..14（混合区 15..29，其后为黑色哑场景）。
        assert_eq!(frames.len(), 45, "两段各 30 帧 − 15 帧叠化");

        // 期望墨迹右缘：第 n 字的设备墨迹 = floor(笔位) + 位图原点偏移
        // （`bmp.left`：位图左缘相对笔位的设备 x，blit_glyph 放置式为
        // `floor(pen) + bmp.left`）+ 位图实测墨迹右缘（逐列扫 alpha；
        // 位图窗口含曲线控制点外扩——CFF 转 glyf 的控制点可探出真实
        // 墨迹之外，故不能直接用窗口宽度，须取实测非零列）。
        let font = prism_text::font_regular();
        let params = TextParams {
            font_size: 20.0,
            letter_spacing: 0.0,
            line_spacing: 0.0,
            max_width: f32::INFINITY,
            align: Align::Center,
        };
        let layout = layout_text(&font, content, &params);
        let total = layout.width;
        let cx = w as f32 * 0.5;
        let mut expected_right: Vec<f32> = vec![cx - total * 0.5]; // n=0：无墨
        let mut acc = 0.0f32;
        for g in &layout.glyphs {
            let bmp = prism_text::raster::render_glyph(&font, g.gid, 20.0);
            // 实测墨迹右缘（位图内最右可见列 + 1；空字形退化为 0）。
            // 判据与帧测量同口径：src-over 黑底后字节 = round(max_ch(color·sa)·255)，
            // 取 > 90（帧测量阈值 96，留 6 字节 f16 往返抖动裕量）。
            let mut ink_cols = 0u32;
            for row in 0..bmp.height {
                for col in 0..bmp.width {
                    let a = bmp.alpha[(row * bmp.width + col) as usize];
                    if a == 0 {
                        continue;
                    }
                    let sa = a as f32 / 255.0;
                    let byte = ((TEXT_RGBA[0] * sa)
                        .max(TEXT_RGBA[1] * sa)
                        .max(TEXT_RGBA[2] * sa)
                        * 255.0)
                        .round() as u32;
                    if byte > 90 {
                        ink_cols = ink_cols.max(col + 1);
                    }
                }
            }
            let pen = acc + cx - total * 0.5;
            // 设备墨迹右缘 = floor(笔位) + 位图左偏 + 位图实测墨迹列。
            expected_right.push(pen.floor() + bmp.left as f32 + ink_cols as f32);
            acc += g.advance;
        }

        // 逐帧实测墨迹右缘，与整字笔位表比对。不复算引擎的 n（其内部
        // `floor(t·cps)` 的浮点舍入不可在外侧复刻），而是反证：每帧
        // 墨迹右缘必须吻合某个整字笔位——半字会落在两个笔位之间，
        // 无一吻合。
        let mut prev_n: Option<usize> = None;
        let mut covered = vec![false; expected_right.len()];
        for (f, frame) in frames.iter().enumerate().take(15) {
            assert_eq!(frame.len(), (w * h * 4) as usize, "帧 {f} 尺寸");
            // 墨迹右缘（白字黑底：任一通道 > 96 即墨）。
            let mut right = f32::NEG_INFINITY;
            let mut any = false;
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 4) as usize;
                    if frame[i] > 96 || frame[i + 1] > 96 || frame[i + 2] > 96 {
                        any = true;
                        right = right.max(x as f32 + 1.0);
                    }
                }
            }
            if !any {
                // 无墨只允许 n = 0（t = 0 的首帧，打字机尚未出字）。
                assert_eq!(f, 0, "帧 {f}：非首帧不得无墨迹（打字机只增不减）");
                covered[0] = true;
                continue;
            }
            let n = expected_right
                .iter()
                .position(|want| (right - want).abs() <= 1.5)
                .unwrap_or_else(|| {
                    panic!(
                        "帧 {f}：墨迹右缘 {right} 不落在任何整字笔位 {expected_right:?}（半字检测）"
                    )
                });
            if let Some(p) = prev_n {
                assert!(n >= p, "帧 {f}：打字机须逐字前进（n={n} 回退自 {p}）");
            }
            prev_n = Some(n);
            covered[n] = true;
        }
        assert!(
            covered.iter().all(|&c| c),
            "前 15 帧纯内容区须覆盖 n=0..={} 全部前缀（实际 {covered:?}）",
            expected_right.len() - 1
        );
    }

    /// 与 [`super::preview_dummy_spec`] 同构的微型哑场景（测试画幅）。
    fn preview_dummy_spec_dim(w: u32, h: u32) -> ClipSpec {
        ClipSpec {
            width: w,
            height: h,
            fps: 30,
            duration: 1.0,
            background: Background::Gradient {
                top: common::hex_to_linear([0x00, 0x00, 0x00]),
                bottom: common::hex_to_linear([0x00, 0x00, 0x00]),
            },
            jpeg_quality: 90,
            tracks: Vec::new(),
        }
    }

    /// 后处理链接线集成：S2 首帧过链后尺寸/alpha 不变、确定可复现。
    #[test]
    fn post_chain_on_real_s2_frame_is_stable() {
        // 经公开批式入口取 S2 截断首段的一帧（不走预览 env，直接构 spec）。
        let mut s2 = scene_s2();
        s2.duration = common::scene_duration(40);
        let dummy = preview_dummy_spec_dim(WIDTH, HEIGHT);
        let frames = render_short_concat_faded_frames(&[s2, dummy], 0.0, 0.0);
        let frame = &frames[10]; // t = 0.33s（淡入后内容已现）
        let out = film_postfx::post_chain(frame, WIDTH, HEIGHT);
        assert_eq!(out.len(), frame.len(), "尺寸不变");
        assert!(out.iter().skip(3).step_by(4).all(|&a| a == 255), "alpha 恒 255");
        assert_eq!(out, film_postfx::post_chain(frame, WIDTH, HEIGHT), "确定性");
        // 链确实改变了画面（非恒等直通——否则接线形同虚设）。
        assert_ne!(out, *frame, "后处理链必须对真实帧产生可见改变");
    }
}
