//! 集成测试：用 `ffmpeg` 解析 `render_short` 产出的 MP4，断言帧数 ==
//! `spec.frame_count()`（运行环境需可用 ffmpeg）。

use prism_author::{
    render_short, render_short_avi_faded, Background, Caption, CaptionStyle, ClipSpec, EasingKind,
    FontKind, KineticText, ShapeKind, ShapeTrack, StaticText, Track, TypewriterText,
};
use std::io::Write;
use std::process::Command;

/// 用 ffmpeg 解码 MP4 到 null 并统计解码出的总帧数。
///
/// 运行 `ffmpeg -hide_banner -i <path> -f null -`，从 stderr 的进度行中提取
/// 最后的 `frame=   N`，即为解码总帧数。
fn ffmpeg_frame_count(path: &str) -> Option<u64> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i", path, "-f", "null", "-"])
        .output()
        .ok()?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut last: Option<u64> = None;
    for line in stderr.lines() {
        if let Some(pos) = line.find("frame=") {
            // 跳过 "frame=" 与后续空白，取首个数字串。
            let after = &line[pos + "frame=".len()..];
            let digits: String = after.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = digits.parse::<u64>() {
                last = Some(v);
            }
        }
    }
    last
}

/// 把字节流写到临时文件，返回路径（测试结束由 OS 回收）。
fn write_temp(mp4: &[u8], name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("创建临时文件失败");
    f.write_all(mp4).expect("写入临时文件失败");
    path
}

#[test]
fn ffmpeg_parses_frame_count_matches_spec() {
    let spec = ClipSpec {
        width: 128,
        height: 96,
        fps: 24,
        duration: 2.0,
        background: Background::DarkGlow,
        jpeg_quality: 82,
        tracks: vec![
            Track::Static(StaticText {
                content: "PRISM ENGINE".into(),
                font: FontKind::Regular,
                size: 16.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.2,
                in_sec: 0.4,
                letter_spacing: 3.0,
                y: 40.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            Track::Kinetic(KineticText {
                content: "棱镜引擎".into(),
                font: FontKind::Bold,
                size: 36.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.3,
                stagger: 0.09,
                per_char: 0.45,
                scale_from: 0.72,
                rise_px: 12.0,
                easing: EasingKind::CubicOut,
                y: 60.0,
                in_sec: 0.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            Track::Shape(ShapeTrack {
                kind: ShapeKind::Ellipse,
                x: 20.0,
                y: 16.0,
                w: 40.0,
                h: 24.0,
                color: [0.1, 0.2, 0.45, 0.3],
                at: 0.0,
                in_sec: 0.5,
                out_sec: 0.0,
                out_at: None,
            }),
            Track::Caption(Caption {
                content: "视频即代码".into(),
                at: 1.0,
                in_sec: 0.4,
                out_sec: 0.0,
                out_at: None,
                style: CaptionStyle {
                    font_size: 18.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                    bg: [0.0, 0.0, 0.0, 0.55],
                    pad_x: 16.0,
                    pad_y: 8.0,
                    margin_bottom: 16.0,
                    line_spacing: 0.0,
                },
            }),
        ],
    };
    let expected = spec.frame_count();
    let mp4 = render_short(&spec);
    assert!(!mp4.is_empty());
    let path = write_temp(&mp4, "prism_author_it.mp4");
    let got = ffmpeg_frame_count(path.to_str().unwrap())
        .expect("ffmpeg 不可用或解析失败（运行环境需可用 ffmpeg）");
    assert_eq!(got, expected as u64, "ffmpeg 解码帧数 {} != spec.frame_count() {}", got, expected);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn ffmpeg_parses_empty_single_frame_mp4() {
    let spec = ClipSpec {
        width: 32,
        height: 32,
        fps: 1,
        duration: 0.5,
        background: Background::DarkGlow,
        jpeg_quality: 70,
        tracks: vec![],
    };
    assert_eq!(spec.frame_count(), 1);
    let mp4 = render_short(&spec);
    let path = write_temp(&mp4, "prism_author_empty.mp4");
    let got = ffmpeg_frame_count(path.to_str().unwrap())
        .expect("ffmpeg 不可用或解析失败（运行环境需可用 ffmpeg）");
    assert_eq!(got, 1, "空 tracks 的单帧 MP4 应被 ffmpeg 解析为 1 帧");
    let _ = std::fs::remove_file(&path);
}

/// 内容增强包端到端：`render_short_avi_faded`（新背景 + 打字机轨道 +
/// Clip 级淡入淡出）产出的 AVI 必须被 ffmpeg 识别为 mjpeg，且解码帧数
/// == `spec.frame_count()`。
#[test]
fn ffmpeg_parses_faded_avi_mjpeg_and_frame_count() {
    let spec = ClipSpec {
        width: 128,
        height: 96,
        fps: 12,
        duration: 1.0,
        background: Background::Aurora,
        jpeg_quality: 82,
        tracks: vec![
            Track::Static(StaticText {
                content: "PRISM".into(),
                font: FontKind::Regular,
                size: 16.0,
                color: [0.46, 0.78, 0.92, 0.85],
                at: 0.1,
                in_sec: 0.3,
                letter_spacing: 3.0,
                y: 40.0,
                out_sec: 0.0,
                out_at: None,
                center_x: None,
            }),
            Track::Typewriter(TypewriterText {
                content: "TYPE".into(),
                font: FontKind::Bold,
                size: 24.0,
                color: [0.98, 0.98, 1.0, 1.0],
                at: 0.2,
                y: 62.0,
                cps: 12.0,
                cursor: true,
                cursor_blink_hz: 1.2,
                in_sec: 0.2,
                out_sec: 0.0,
                out_at: None,
                letter_spacing: 1.0,
                center_x: None,
            }),
        ],
    };
    let expected = spec.frame_count();
    let avi = render_short_avi_faded(&spec, 0.3, 0.3).expect("AVI 封装失败");
    assert!(!avi.is_empty());
    let path = write_temp(&avi, "prism_author_faded_v4.avi");

    // 流类型应为 mjpeg（ffmpeg -i 的 stderr 里含 "Video: mjpeg"）。
    let probe = Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(path.to_str().unwrap())
        .output()
        .expect("ffmpeg 不可用");
    let stderr = String::from_utf8_lossy(&probe.stderr);
    assert!(
        stderr.contains("mjpeg"),
        "ffmpeg 必须识别流为 mjpeg，实际 stderr：{stderr}"
    );

    let got = ffmpeg_frame_count(path.to_str().unwrap())
        .expect("ffmpeg 不可用或解析失败（运行环境需可用 ffmpeg）");
    assert_eq!(got, expected as u64, "解码帧数 {} != spec.frame_count() {}", got, expected);
    let _ = std::fs::remove_file(&path);
}
