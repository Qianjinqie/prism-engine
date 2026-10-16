//! 内容增强包集成测试：打字机轨道 / 新背景预设 / Clip 级淡入淡出。
//!
//! 全部经由**公开 API**（`render_short` / `render_short_faded` /
//! `render_short_avi_faded`）验证，与单元测试（`src/lib.rs` 内的
//! `#[cfg(test)]`，覆盖 `pub(crate)` 助手）互为补充。

use prism_author::{
    render_short, render_short_avi_faded, render_short_faded, Background, ClipSpec, FontKind,
    StaticText, Track, TypewriterText,
};

/// 构造一个含打字机轨道的小型确定性 spec。
fn typewriter_spec(cps: f32) -> ClipSpec {
    ClipSpec {
        width: 96,
        height: 64,
        fps: 12,
        duration: 1.0,
        background: Background::DarkGlow,
        jpeg_quality: 80,
        tracks: vec![Track::Typewriter(TypewriterText {
            content: "TYPEWRITER".into(),
            font: FontKind::Regular,
            size: 16.0,
            color: [0.9, 0.95, 1.0, 1.0],
            at: 0.1,
            y: 32.0,
            cps,
            cursor: true,
            cursor_blink_hz: 1.2,
            in_sec: 0.2,
            out_sec: 0.0,
            out_at: None,
            letter_spacing: 1.0,
            center_x: None,
        })],
    }
}

/// 打字机轨道：MP4 非空 + 黄金确定性（两次逐字节一致）。
#[test]
fn typewriter_mp4_nonempty_and_golden() {
    let spec = typewriter_spec(12.0);
    let a = render_short(&spec);
    let b = render_short(&spec);
    assert!(!a.is_empty(), "含 Typewriter 的 spec 必须产出非空 MP4");
    assert_eq!(&a[4..8], b"ftyp", "产物应是合法 MP4（顶层 ftyp 盒）");
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b, "固定 spec 渲染必须逐字节一致（确定性）");
}

/// 打字机轨道：不同 `cps` 产出不同 MP4（健全性，证明逐字生效）。
#[test]
fn typewriter_cps_changes_output() {
    let slow = render_short(&typewriter_spec(6.0));
    let fast = render_short(&typewriter_spec(60.0));
    assert_ne!(slow, fast, "不同 cps 必须产出不同画面（逐字进度生效）");
}

/// 新背景 Aurora：MP4 非空 + 确定性 + 与 DarkGlow 产物不同。
#[test]
fn aurora_background_mp4_nonempty_deterministic_and_distinct() {
    let base = |bg: Background| ClipSpec {
        width: 64,
        height: 48,
        fps: 12,
        duration: 1.0,
        background: bg,
        jpeg_quality: 80,
        tracks: vec![Track::Static(StaticText {
            content: "AURORA".into(),
            font: FontKind::Regular,
            size: 14.0,
            color: [0.9, 0.95, 1.0, 0.9],
            at: 0.1,
            in_sec: 0.2,
            letter_spacing: 2.0,
            y: 32.0,
            out_sec: 0.0,
            out_at: None,
            center_x: None,
        })],
    };
    let aurora = render_short(&base(Background::Aurora));
    let aurora_again = render_short(&base(Background::Aurora));
    let glow = render_short(&base(Background::DarkGlow));
    assert!(!aurora.is_empty());
    assert_eq!(aurora, aurora_again, "Aurora 渲染必须黄金确定");
    assert_ne!(aurora, glow, "Aurora 必须与 DarkGlow 产物不同");
}

/// 新背景 Scanline：MP4 非空 + 确定性 + 与 DarkGlow 产物不同。
#[test]
fn scanline_background_mp4_nonempty_deterministic_and_distinct() {
    let base = |bg: Background| ClipSpec {
        width: 64,
        height: 48,
        fps: 12,
        duration: 1.0,
        background: bg,
        jpeg_quality: 80,
        tracks: vec![],
    };
    let scan = render_short(&base(Background::Scanline));
    let scan_again = render_short(&base(Background::Scanline));
    let glow = render_short(&base(Background::DarkGlow));
    assert!(!scan.is_empty());
    assert_eq!(scan, scan_again, "Scanline 渲染必须黄金确定");
    assert_ne!(scan, glow, "Scanline 必须与 DarkGlow 产物不同");
}

/// Clip 级淡入淡出：`render_short_faded` 非空 + 黄金确定 + 合法 MP4。
#[test]
fn render_short_faded_mp4_nonempty_and_golden() {
    let spec = typewriter_spec(12.0);
    let a = render_short_faded(&spec, 0.5, 0.5);
    let b = render_short_faded(&spec, 0.5, 0.5);
    assert!(!a.is_empty(), "render_short_faded 必须产出非空 MP4");
    assert_eq!(&a[4..8], b"ftyp");
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b, "render_short_faded 必须黄金确定");
    // fade>0 必须改变产物（相对无淡变的 render_short）。
    assert_ne!(a, render_short(&spec), "片头/片尾淡变必须改变画面");
}

/// Clip 级淡入淡出：`render_short_avi_faded` 产出标准 MJPEG-in-AVI。
#[test]
fn render_short_avi_faded_returns_valid_avi() {
    let spec = typewriter_spec(12.0);
    let avi = render_short_avi_faded(&spec, 0.5, 0.5).expect("AVI 封装失败");
    assert!(!avi.is_empty());
    assert_eq!(&avi[0..4], b"RIFF", "AVI 必须以 RIFF 起头");
    assert_eq!(&avi[8..12], b"AVI ", "AVI 必须为 'AVI ' 类型");
    let riff_size = u32::from_le_bytes(avi[4..8].try_into().unwrap()) as usize;
    assert_eq!(riff_size, avi.len() - 8);
    let has = |needle: &[u8]| avi.windows(needle.len()).position(|w| w == needle).is_some();
    assert!(has(b"movi"));
    assert!(has(b"idx1"));
    assert!(has(b"MJPG"), "流处理器/压缩应为 MJPG");
}
