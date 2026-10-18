//! PNG 解码器 + 统一门面测试（V1.5-W1-A / P0-1）。
//!
//! * **PNG 逐字节对拍**：`fixtures/png/*.png` 解码后，
//!   `samples` 必须与 `*.samples.bin` 逐字节相同（反滤波正确性），
//!   `to_rgba8()` 必须与 `*.rgba.bin` 逐字节相同（色彩解释正确性）；
//!   fixture 覆盖 16 个文件、色型 0/2/3/4/6、位深 8/16、滤波 0–4 全覆盖、多 IDAT。
//! * **门面**：L0（png/jpeg）往返、L1/L2 三段式错误文案断言、交错 PNG 报 L2 错。

use std::path::{Path, PathBuf};

use prism_io::image_decode::{decode_image, is_native_format, DecodedImage};
use prism_io::png::decode_png;

/// PNG 固件目录。
fn png_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("png")
}

/// JPEG 固件目录。
fn jpeg_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("jpeg")
}

/// 读文件（缺固件时给出可行动的 panic 文案）。
fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("读不到固件 {}（{}）", path.display(), e))
}

/// 固件清单（名称, 是否 16-bit）。
const CASES: [(&str, bool); 16] = [
    ("rgba8_f0", false),
    ("rgba8_mixed", false),
    ("rgb8_odd", false),
    ("gray8", false),
    ("gray_alpha8", false),
    ("palette8_trns", false),
    ("rgb8_trns", false),
    ("gray8_trns", false),
    ("rgba16", true),
    ("rgb16", true),
    ("gray16", true),
    ("gray_alpha16", true),
    ("rgb8_big", false),
    ("gray8_big", false),
    ("rgba8_ancillary", false),
    ("gray8_multi_idat", false),
];

/// PNG 全色型 / 全位深 / 全滤波逐字节对拍（samples 与 rgba 双口径）。
#[test]
fn png_fixtures_byte_exact() {
    let dir: PathBuf = png_dir();
    let mut checked: usize = 0;
    let mut sixteen: usize = 0;
    for (name, is16) in CASES.iter() {
        let png_path: PathBuf = dir.join(format!("{}.png", name));
        if !png_path.exists() {
            panic!(
                "缺少固件 {}：请先运行 `python3 crates/prism-io/tests/png_golden_gen.py`",
                png_path.display()
            );
        }
        let bytes: Vec<u8> = read(&png_path);
        let img = decode_png(&bytes).unwrap_or_else(|e| panic!("{} 解码失败：{}", name, e));

        // ① 原始样本流逐字节对拍（验证解压 + 反滤波 + 位深口径）。
        let samples_expect: Vec<u8> = read(&dir.join(format!("{}.samples.bin", name)));
        assert_eq!(
            img.samples.len(),
            samples_expect.len(),
            "{} samples 长度不符（{} vs {}）",
            name,
            img.samples.len(),
            samples_expect.len()
        );
        assert_eq!(img.samples, samples_expect, "{} samples 逐字节不符", name);

        // ② RGBA8 逐字节对拍（验证色彩解释 / tRNS / 16-bit 高字节截断口径）。
        let rgba_expect: Vec<u8> = read(&dir.join(format!("{}.rgba.bin", name)));
        let rgba_got: Vec<u8> = img.to_rgba8();
        assert_eq!(
            rgba_got.len(),
            rgba_expect.len(),
            "{} RGBA 长度不符（{} vs {}）",
            name,
            rgba_got.len(),
            rgba_expect.len()
        );
        assert_eq!(rgba_got, rgba_expect, "{} RGBA 逐字节不符", name);

        // ③ 位深自洽。
        assert_eq!(img.is_16bit(), *is16, "{} 位深判定与固件不符", name);
        checked += 1;
        if *is16 {
            sixteen += 1;
        }
    }
    assert!(checked >= 10, "PNG 逐字节对拍仅覆盖 {} 例（应 ≥10）", checked);
    assert!(sixteen >= 3, "16-bit 固件仅 {} 例（应 ≥3）", sixteen);
    println!("PNG 逐字节对拍：{} 例全绿（其中 16-bit {} 例）", checked, sixteen);
}

/// 滤波覆盖：至少一个固件同时出现 0..=4 五种滤波（否则反滤波未真正被打到）。
#[test]
fn png_filter_types_covered() {
    let dir: PathBuf = png_dir();
    let mut seen = [false; 5];
    for (name, _) in CASES.iter() {
        let bytes: Vec<u8> = read(&dir.join(format!("{}.png", name)));
        let img = decode_png(&bytes).expect("解码应成功");
        for &f in img.filters.iter() {
            if (f as usize) < seen.len() {
                seen[f as usize] = true;
            }
        }
    }
    assert!(seen.iter().all(|&x| x), "滤波类型覆盖不全：{:?}", seen);
}

/// 交错 PNG 必须报 L2 明确错误（不得静默当成非交错解）。
#[test]
fn interlaced_png_reports_l2_error() {
    // 手工构造交错 IHDR：签名 + IHDR(interlace=1) + IEND（CRC 用库内 crc32）。
    let mut bytes: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr: Vec<u8> = Vec::new();
    ihdr.extend_from_slice(&4u32.to_be_bytes());
    ihdr.extend_from_slice(&4u32.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(0); // gray
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(1); // interlace = Adam7
    push_chunk(&mut bytes, b"IHDR", &ihdr);
    push_chunk(&mut bytes, b"IEND", &[]);

    let err: String = decode_png(&bytes).unwrap_err();
    assert!(err.contains("交错"), "错误文案应指明交错，实际：{}", err);
    assert!(err.contains("L2"), "错误文案应标注 L2 层，实际：{}", err);
}

/// 追加一个 PNG chunk（长度 + 类型 + 数据 + CRC32）。
fn push_chunk(out: &mut Vec<u8>, ctype: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(ctype);
    out.extend_from_slice(data);
    let mut crc_input: Vec<u8> = Vec::new();
    crc_input.extend_from_slice(ctype);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&prism_io::crc32(&crc_input).to_be_bytes());
}

/// L0：PNG / JPEG 均走内置解码，且给出一致的 RGBA8。
#[test]
fn facade_l0_decodes_png_and_jpeg() {
    // PNG（8-bit 灰度 → Gray8 分档）
    let png: Vec<u8> = read(&png_dir().join("gray8.png"));
    let img: DecodedImage = decode_image(&png, "png").expect("PNG 应 L0 解码成功");
    match &img {
        DecodedImage::Gray8 { w, h, samples } => {
            assert_eq!((*w, *h), (17, 5));
            assert_eq!(samples.len(), 17 * 5);
        }
        other => panic!("gray8.png 应为 Gray8 分档，实际 {:?}", other),
    }
    assert_eq!(img.bit_depth(), 8);

    // PNG 16-bit 灰度 → Gray16 分档
    let png16: Vec<u8> = read(&png_dir().join("gray16.png"));
    let img16: DecodedImage = decode_image(&png16, ".PNG").expect("大写扩展名也应识别");
    match &img16 {
        DecodedImage::Gray16 { w, h, samples } => {
            assert_eq!((*w, *h), (9, 9));
            assert_eq!(samples.len(), 81);
        }
        other => panic!("gray16.png 应为 Gray16 分档，实际 {:?}", other),
    }

    // JPEG → Rgba8 分档（alpha 恒 255）
    let jpg: Vec<u8> = read(&jpeg_dir().join("interop_444.jpg"));
    let jimg: DecodedImage = decode_image(&jpg, "jpeg").expect("JPEG 应 L0 解码成功");
    assert_eq!(jimg.width(), 96);
    assert_eq!(jimg.height(), 72);
    assert_eq!(jimg.bit_depth(), 8);
    let rgba: Vec<u8> = jimg.to_rgba8();
    assert_eq!(rgba.len(), 96 * 72 * 4);
    assert!(rgba.chunks(4).all(|px| px[3] == 255), "JPEG 的 alpha 必须恒 255");
}

/// 魔数优先：扩展名写错也能正确解码（不被扩展名骗到错误路径）。
#[test]
fn facade_sniffs_magic_over_extension() {
    let png: Vec<u8> = read(&png_dir().join("rgb8_odd.png"));
    // 故意传入错误扩展名。
    let img = decode_image(&png, "mp4").expect("魔数命中 PNG 应直接解码，不看扩展名");
    assert_eq!(img.width(), 13);
    assert_eq!(img.height(), 7);

    let jpg: Vec<u8> = read(&jpeg_dir().join("interop_gray.jpg"));
    let jimg = decode_image(&jpg, "webp").expect("魔数命中 JPEG 应直接解码");
    assert_eq!((jimg.width(), jimg.height()), (96, 72));
}

/// L1：容器/音视频编码 → 三段式错误（未内置解码 / ffmpeg 协作 / 装 ffmpeg 后重试）。
#[test]
fn facade_l1_three_part_error() {
    for ext in ["mp4", "mov", "webm", "mp3", "aac", "flac", "mkv", "ogg"] {
        let err: String = decode_image(b"", ext).unwrap_err();
        assert!(err.contains("未内置解码"), "{}：缺「未内置解码」段：{}", ext, err);
        assert!(
            err.contains("经 ffmpeg 协作可用"),
            "{}：缺「经 ffmpeg 协作可用」段：{}",
            ext,
            err
        );
        assert!(
            err.contains("安装 ffmpeg 后重试"),
            "{}：缺「安装 ffmpeg 后重试」段：{}",
            ext,
            err
        );
        assert!(err.contains("ffmpeg -i"), "{}：缺可执行命令：{}", ext, err);
        assert!(!is_native_format(ext), "{} 不应被判为内置格式", ext);
    }
}

/// L2：矩阵外编码 / 未知扩展名 → 明确「不在解码矩阵内」，且绝不返回 Ok。
#[test]
fn facade_l2_matrix_error() {
    for ext in ["webp", "prores", "tiff", "bmp", "heic", "avif", "jxl"] {
        let err: String = decode_image(b"", ext).unwrap_err();
        assert!(
            err.contains("不在内置解码矩阵内"),
            "{}：错误文案应说明矩阵外：{}",
            ext,
            err
        );
        assert!(err.contains("L2"), "{}：错误文案应标注 L2：{}", ext, err);
    }
    // 完全未知的扩展名同样报错（永不静默失败）。
    let err: String = decode_image(b"", "weird-format").unwrap_err();
    assert!(err.contains("不在内置解码矩阵内"), "未知扩展名也应报矩阵外：{}", err);
    // 空扩展名 + 认不出的魔数：报「无法识别」。
    let err2: String = decode_image(b"\x00\x01\x02\x03", "").unwrap_err();
    assert!(err2.contains("无法识别图像格式"), "空扩展名应报无法识别：{}", err2);
}

/// 扩展名与内容不符：如实报错，不静默返回空图。
#[test]
fn facade_reports_extension_content_mismatch() {
    let err: String = decode_image(b"garbage-bytes-here", "png").unwrap_err();
    assert!(err.contains("PNG 解码失败"), "应报 PNG 解码失败：{}", err);
    let err2: String = decode_image(b"garbage-bytes-here", "jpg").unwrap_err();
    assert!(err2.contains("JPEG 解码失败"), "应报 JPEG 解码失败：{}", err2);
}

/// 内置格式判定与门面一致。
#[test]
fn facade_native_format_set() {
    for ok in ["png", "PNG", ".png", "jpg", "JPEG", ".Jpeg"] {
        assert!(is_native_format(ok), "{} 应判为内置格式", ok);
    }
    for not_ok in ["webp", "mp4", "", "gif"] {
        assert!(!is_native_format(not_ok), "{} 不应判为内置格式", not_ok);
    }
}
