//! 多场景 + 转场拼接：公开入口集成测试（MP4 / AVI）。
//!
//! 覆盖：
//! - `render_short_concat`：产出非空 MP4（顶层 `ftyp`）+ 黄金确定；
//! - `render_short_concat_avi`：产出合法 AVI（RIFF/AVI/MJPG），帧数（idx1
//!   项数）== 预期拼接帧数，并用 ffmpeg（若可用）交叉验证帧数与时长；
//! - 参数非法（specs < 2、画幅 / 帧率不一致）返回 `Err`（不 panic）；
//! - `transitions` 越界默认 [`Transition::Fade`]。

use prism_author::{
    render_short_concat, render_short_concat_avi, Background, ClipSpec, FontKind, KineticText,
    StaticText, Track, Transition, WipeDir,
};
use std::io::Write;
use std::process::Command;

/// 构造第 `idx` 支子片（64×48 @12fps 1.0s ⇒ 12 帧；三支仅文案 / 背景不同）。
fn sub_spec(idx: usize) -> ClipSpec {
    let (bg, title) = match idx {
        0 => (Background::Aurora, "SCENE ONE"),
        1 => (Background::Scanline, "SCENE TWO"),
        _ => (Background::DarkGlow, "SCENE THREE"),
    };
    ClipSpec {
        width: 64,
        height: 48,
        fps: 12,
        duration: 1.0,
        background: bg,
        jpeg_quality: 80,
        tracks: vec![
            Track::Static(StaticText {
                content: format!("LABEL {idx}"),
                font: FontKind::Regular,
                size: 12.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.1,
                in_sec: 0.2,
                letter_spacing: 2.0,
                y: 14.0,
                out_sec: 0.2,
                out_at: None,
                center_x: None,
            }),
            Track::Kinetic(KineticText {
                content: title.into(),
                font: FontKind::Bold,
                size: 18.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.2,
                stagger: 0.05,
                per_char: 0.3,
                scale_from: 0.8,
                rise_px: 6.0,
                easing: prism_author::EasingKind::CubicOut,
                y: 32.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
        ],
    }
}

/// 预期拼接总帧数（与 `concat_specs_frames` 的折叠语义一致）。
fn expected_concat_frames(specs: &[ClipSpec], trans_sec: f32) -> usize {
    let mut total = specs[0].frame_count();
    for i in 1..specs.len() {
        let f = specs[i].frame_count();
        let tf = ((trans_sec * specs[i].fps as f32).round() as usize).clamp(1, total.min(f));
        total += f - tf;
    }
    total
}

/// 从 AVI 字节流解析帧数（`idx1` 索引项数 = chunk size / 16）。
fn avi_frame_count(avi: &[u8]) -> usize {
    // idx1 为顶层最后一个块；取最后一次出现位置防误匹配 JPEG 数据。
    let pos = avi
        .windows(4)
        .rposition(|w| w == b"idx1")
        .expect("AVI 应含 idx1 索引块");
    let size = u32::from_le_bytes(avi[pos + 4..pos + 8].try_into().unwrap()) as usize;
    assert_eq!(size % 16, 0, "idx1 每项应恰为 16 字节");
    size / 16
}

/// 用 ffmpeg 解码并统计总帧数（不可用时返回 None）。
fn ffmpeg_frame_count(path: &str) -> Option<u64> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i", path, "-f", "null", "-"])
        .output()
        .ok()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut last: Option<u64> = None;
    for line in stderr.lines() {
        if let Some(pos) = line.find("frame=") {
            let after = &line[pos + "frame=".len()..];
            let digits: String = after
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(v) = digits.parse::<u64>() {
                last = Some(v);
            }
        }
    }
    last
}

/// 用 ffprobe 读取时长（秒；不可用时返回 None）。
fn ffprobe_duration(path: &str) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// 把字节流写到临时文件，返回路径（测试结束由 OS 回收）。
fn write_temp(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&path).expect("创建临时文件失败");
    f.write_all(bytes).expect("写入临时文件失败");
    path
}

/// `render_short_concat`：产出非空 MP4（顶层 ftyp）+ 黄金确定；
/// 与单支入口（`render_short` 风格）互不干扰。
#[test]
fn render_short_concat_produces_valid_mp4() {
    let specs = vec![sub_spec(0), sub_spec(1)];
    let mp4 = render_short_concat(&specs, &[Transition::Fade], 0.5);
    assert!(!mp4.is_empty(), "拼接 MP4 必须非空");
    assert_eq!(&mp4[4..8], b"ftyp", "产物应是合法 MP4（顶层 ftyp 盒）");

    // 黄金确定性：同参数两次渲染逐字节一致。
    let mp4b = render_short_concat(&specs, &[Transition::Fade], 0.5);
    assert_eq!(mp4.len(), mp4b.len());
    assert_eq!(mp4, mp4b, "拼接渲染必须逐字节一致（确定性）");
}

/// `render_short_concat`：Wipe 转场的产物与 Fade 不同（转场确实生效），
/// 且三支子片两次转场的总帧数符合公式。
#[test]
fn render_short_concat_wipe_differs_from_fade() {
    let specs = vec![sub_spec(0), sub_spec(1), sub_spec(2)];
    let faded = render_short_concat(&specs, &[Transition::Fade, Transition::Fade], 0.5);
    let wiped = render_short_concat(
        &specs,
        &[
            Transition::Wipe { dir: WipeDir::Right },
            Transition::Fade,
        ],
        0.5,
    );
    assert_ne!(faded, wiped, "Wipe 与 Fade 必须产出不同画面");

    // ffmpeg 交叉验证帧数（若可用）：12+12+12 - 6 - 6 = 24。
    let expected = expected_concat_frames(&specs, 0.5);
    assert_eq!(expected, 24, "预期拼接帧数应为 24");
    if let Some(n) = ffmpeg_frame_count(&write_temp(&faded, "prism_concat_fade.mp4").to_string_lossy())
    {
        assert_eq!(n as usize, expected, "ffmpeg 解码帧数必须等于预期拼接帧数");
    }
}

/// `render_short_concat_avi`：合法 AVI（RIFF/AVI/MJPG）+ 帧数 == 预期拼接帧数；
/// ffmpeg（若可用）断言帧数与时长正确。
#[test]
fn render_short_concat_avi_valid_and_frame_count_matches() {
    let specs = vec![sub_spec(0), sub_spec(1), sub_spec(2)];
    let transitions = vec![
        Transition::Wipe { dir: WipeDir::Right },
        Transition::Fade,
    ];
    let avi = render_short_concat_avi(&specs, &transitions, 0.5).expect("AVI 封装失败");
    assert!(!avi.is_empty(), "拼接 AVI 必须非空");

    // 顶层 RIFF/AVI 魔数 + RIFF size。
    assert_eq!(&avi[0..4], b"RIFF", "AVI 必须以 RIFF 起头");
    assert_eq!(&avi[8..12], b"AVI ", "AVI 必须为 'AVI ' 类型");
    let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
    assert_eq!(riff_size, avi.len() - 8);

    // 关键块与 MJPG 流标记齐全。
    let has = |needle: &[u8]| avi.windows(needle.len()).position(|w| w == needle).is_some();
    assert!(has(b"hdrl"), "应含 hdrl");
    assert!(has(b"movi"), "应含 movi");
    assert!(has(b"idx1"), "应含 idx1");
    assert!(has(b"vids"), "视频流类型应为 vids");
    assert!(has(b"MJPG"), "流处理器/压缩应为 MJPG");
    assert!(has(b"00dc"), "帧块四CC 应为 00dc");

    // 帧数：idx1 项数 == 预期拼接帧数（12+12+12 - 6 - 6 = 24）。
    let expected = expected_concat_frames(&specs, 0.5);
    assert_eq!(expected, 24, "预期拼接帧数应为 24");
    assert_eq!(
        avi_frame_count(&avi),
        expected,
        "idx1 帧数必须等于预期拼接帧数"
    );

    // ffmpeg / ffprobe 交叉验证（若可用）。
    let path = write_temp(&avi, "prism_concat_v5_test.avi");
    let path_str = path.to_string_lossy().to_string();
    if let Some(n) = ffmpeg_frame_count(&path_str) {
        assert_eq!(
            n as usize, expected,
            "ffmpeg 解码帧数必须等于预期拼接帧数"
        );
    }
    if let Some(d) = ffprobe_duration(&path_str) {
        let expect_sec = expected as f64 / specs[0].fps as f64; // 24/12 = 2.0s
        assert!(
            (d - expect_sec).abs() < 0.1,
            "ffprobe 时长 {d:.3}s 应约等于 {expect_sec:.3}s（帧数/fps）"
        );
    }
}

/// `transitions` 越界默认 `Transition::Fade`：传空 transitions 也能拼接，
/// 且与显式全 Fade 产物逐字节一致。
#[test]
fn concat_transitions_out_of_range_defaults_to_fade() {
    let specs = vec![sub_spec(0), sub_spec(1)];
    let defaulted = render_short_concat_avi(&specs, &[], 0.5).expect("空 transitions 应默认 Fade");
    let explicit = render_short_concat_avi(&specs, &[Transition::Fade], 0.5).expect("AVI 封装失败");
    assert_eq!(defaulted, explicit, "transitions 越界必须默认 Fade");

    // 三支两转场但只给一个 transition：第二对同样默认 Fade。
    let specs3 = vec![sub_spec(0), sub_spec(1), sub_spec(2)];
    let one = render_short_concat_avi(&specs3, &[Transition::Fade], 0.5).expect("AVI 封装失败");
    let two = render_short_concat_avi(&specs3, &[Transition::Fade, Transition::Fade], 0.5)
        .expect("AVI 封装失败");
    assert_eq!(one, two, "越界 transition 必须默认 Fade");
}

/// 参数非法：specs < 2 或画幅 / 帧率不一致 ⇒ 返回 Err（不 panic）。
#[test]
fn render_short_concat_avi_rejects_invalid_specs() {
    // 少于 2 支。
    let err = render_short_concat_avi(&[sub_spec(0)], &[], 0.5);
    assert!(err.is_err(), "少于 2 支子片必须返回 Err");
    assert!(render_short_concat_avi(&[], &[], 0.5).is_err(), "空 specs 必须返回 Err");

    // 画幅不一致。
    let mut bad = sub_spec(1);
    bad.width = 32;
    bad.height = 32;
    let err = render_short_concat_avi(&[sub_spec(0), bad], &[Transition::Fade], 0.5);
    assert!(err.is_err(), "画幅不一致必须返回 Err");

    // 帧率不一致。
    let mut bad_fps = sub_spec(1);
    bad_fps.fps = 24;
    let err = render_short_concat_avi(&[sub_spec(0), bad_fps], &[Transition::Fade], 0.5);
    assert!(err.is_err(), "帧率不一致必须返回 Err");
}
