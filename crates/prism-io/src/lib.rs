//! `prism-io` — 纯手写、零第三方依赖的媒体 I/O 模块。
//!
//! 本 crate 仅依赖 `std`（含 `std::fs` / `std::io` / `std::path`），
//! 所有 PNG 编解码（zlib 存储型 deflate、adler32、CRC32）均自实现，
//! 不引入任何外部 crate。
//!
//! 依赖 `prism_gpu::Framebuffer`（RGBA8 帧缓冲）。

// ---------------------------------------------------------------------------
// A6：SHA-256（FIPS 180-4）——缓存 manifest 与签名生态的哈希原语（零依赖）
// ---------------------------------------------------------------------------

pub mod sha256;

pub use sha256::sha256;

// ---------------------------------------------------------------------------
// 统一图像解码门面（L0 PNG/JPEG 内置，L1/L2 三段式报错）
// ---------------------------------------------------------------------------

/// DEFLATE / zlib 解压门面（实现在 L0 叶子 crate `prism-bytes`，此处统一
/// 以 `Result<_, String>` 口径暴露，供 PNG / EXR 解码共用）。
pub mod inflate;

/// 基线 JPEG 解码器（定点 IDCT：Stage1 i64 + Stage2 阈值内 i64 / 超阈 i128）。
pub mod jpeg_baseline;

/// PNG 解码器（非交错 8/16-bit，色型 0/2/3/4/6，滤波 0–4）。
pub mod png;

/// 统一图像解码门面：`decode_image(bytes, ext)`。
pub mod image_decode;

pub use image_decode::{decode_image, is_native_format, DecodedImage};

pub use jpeg_baseline::{
    decode_baseline, decode_baseline_rgba, idct_8x8_fixed, idct_8x8_fixed_wide, ycbcr_to_rgb,
    HuffDec, JpegImage, IDCT_C_TABLE, IDCT_STAGE2_I64_THRESHOLD, IDCT_W_TABLE,
};

pub use png::{decode_png, PngImage};

// ---------------------------------------------------------------------------
// C3：增量渲染缓存清单（cache.json）——确定性序列化 + 快照库磁盘镜像
// ---------------------------------------------------------------------------

pub mod cache_manifest;

pub use cache_manifest::{
    CacheEntry, CacheManifest, ManifestError, SnapshotManifest, SnapshotRecord,
    CACHE_MANIFEST_SCHEMA_VERSION,
};

// ---------------------------------------------------------------------------
// 自定义生态分发层——数据包市场 + 特解包签名
// ---------------------------------------------------------------------------

pub mod distribution;

pub use distribution::{
    scan_directory, write_package, AssetEntry, Dependency, IndexEntry, Package, PackageError,
    PackageId, PackageIndex, PackageSignature, ScanReport, SemVer, VersionReq, PACKAGE_EXTENSION,
    PACKAGE_SCHEMA_VERSION,
};

// ---------------------------------------------------------------------------
// A5：RGBA16F 半精度帧缓冲——tile 渲染的存储域（f16 存储、f32 计算）
// ---------------------------------------------------------------------------

pub mod fbuf;

pub use fbuf::{f16_to_f32, f32_to_f16, Fbuf16F};

// ---------------------------------------------------------------------------
// 场景与材质互操作（USD `.usda` 文本解析 + MaterialX → 特解参数包）
// ---------------------------------------------------------------------------

pub mod usd;

/// `.usda` 导出（引擎场景 → 标准文本，解析器的规范逆；R2 L2 字节不动点）。
pub mod usd_export;

pub use usd_export::{export_layer, export_scene, ExportMeta, ExportReport, UsdMaterial};

pub mod materialx;

// ---------------------------------------------------------------------------
// 相机 RAW（DNG IFD 解析 + Bayer 去马塞克 + 白平衡 + 色彩矩阵）
// ---------------------------------------------------------------------------

pub mod raw;

pub mod raw_lossless_jpeg;

pub mod raw_lossless_jpeg_encode;

/// DNG 合成样本 + 端到端诊断（详见模块文档）。
///
/// # 命令行入口
///
/// ```bash
/// # 诊断一个真实相机 DNG（不需要改任何代码）
/// cargo run -p prism-io --example dng_probe -- /path/to/real.dng
/// ```
///
/// # 库入口
///
/// ```no_run
/// # use prism_io::{diagnose_dng, synthesize_dng, DngSampleSpec};
/// // 写入：从零合成一个最小合法 DNG
/// let bytes = synthesize_dng(&DngSampleSpec::new(32, 16, 16, [0, 1, 1, 2]))?;
/// std::fs::write("/tmp/mini.dng", &bytes).map_err(|e| e.to_string())?;
/// // 读出：走完整管线（parse → raw → CFA → 显影）并报诊断
/// println!("{}", diagnose_dng("/tmp/mini.dng")?.report());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub mod raw_dng_synth;

/// DNG **SubIFD 预览图解析**。
///
/// 真实相机 DNG 普遍在 `SubIFDs`（0x014A）里带一张小预览图；本模块解析该
/// 子 IFD 链，在压缩方式受支持（1 / 7 / 8 / 32946）时**解出像素**，
/// 否则只报元数据并写明为何不解（诚实边界）。
pub mod raw_subifd;

pub use raw::{
    demosaic_bilinear, mat3_mul, mat3_mul_matrix, probe_format, read_ifd, white_balance_gains,
    BayerLayout, BayerPlane, ByteOrder, CfaColor, CfaPattern, Cursor, DeflatePredictor,
    DevelopOptions, DngFile, ExifInfo, GpsIfd, GpsTag, Mat3, MatrixMode, RawError, RawFormat,
    RawLevels, RawResult, SRGB_TO_XYZ, TiffEntry, TiffIfd, WbMode, XYZ_TO_SRGB,
};

pub use raw_dng_synth::{
    diagnose_dng, diagnose_dng_bytes, synthesize_dng, DngDiagnostics, DngLayout, DngSampleSpec,
    PreviewLayout, PreviewSpec,
};

pub use raw_subifd::{
    decode_sub_ifd, list_sub_ifds, parse_sub_ifds, sub_ifd_offsets, SubIfdImage, SubIfdInfo,
};

pub use raw_lossless_jpeg::{
    decode_lossless_jpeg, interleave_row_map, HuffTable, LosslessImage, Predictor, MAX_SSSS,
};

// ---------------------------------------------------------------------------
// OpenEXR 容器（分块浮点 + 多通道 AOV + Z/Float deep；手写 inflate）
// ---------------------------------------------------------------------------

pub mod exr;
pub mod exr_compress;
pub mod exr_dwa;
pub mod exr_write;

pub use exr::{
    inflate_raw, read_all_tiles, read_deep_image, read_flat_region, read_header, read_image,
    read_tile, rle_compress, rle_decompress, ChannelPlane, Compression, DeepImage, DeepPixel,
    DeepSample, ExrChannel, ExrError, ExrHeader, ExrImage, ExrResult, PixelType, TileDesc,
    EXR_MAGIC,
};

use prism_gpu::Framebuffer;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

/// PNG 文件签名（8 字节）。
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// 单条扫描行的最大字节数（必须小于 2^16，因 deflate 存储块使用 16 位长度字段）。
const MAX_STORED_BLOCK: usize = 65_535;

// ===========================================================================
//  校验和原语（自实现，零依赖）
// ===========================================================================

/// 标准 CRC-32（多项式 `0xEDB88320`），用于每个 PNG chunk 的尾部校验字段。
///
/// 输入应为 `chunk 类型(4B) ++ chunk 数据` 的完整拼接。
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Adler-32 校验和（模 65521），用于 zlib 流尾部，
/// 对「每扫描行前置 `0x00` 过滤字节 + RGBA 像素」的原始数据计算。
///
/// 标准定义：以 `s1`（逐字节累加并模 65521）与 `s2`（对 `s1` 的累加倍模 65521）表示，
/// 最终返回 `(s2 << 16) | s1`（运行和置于高 16 位）。
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &byte in data {
        s1 = (s1 + byte as u32) % MOD;
        s2 = (s2 + s1) % MOD;
    }
    (s2 << 16) | s1
}

// ===========================================================================
//  PNG 低层构建辅助
// ===========================================================================

/// 把一个完整 chunk 追加到 `out`。
///
/// chunk 结构：`[长度 4B BE][类型 4B][数据][CRC32 4B BE]`，
/// 其中 CRC32 作用域为「类型 ++ 数据」。
fn append_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);

    // CRC 输入 = 类型串 + 数据。
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// 构造单帧 PNG 的完整字节流（RGBA8，filter=0，无压缩存储型 deflate）。
///
/// 公开供上层（MCP `render_frame` 需直接返回 PNG 字节）在内存中编码，避免
/// 落盘再读回；与 [`save_png`] 复用同一编码路径，保证字节一致。
pub fn encode_png(fb: &Framebuffer) -> Vec<u8> {
    let width: u32 = fb.width;
    let height: u32 = fb.height;

    // 像素缓冲区长度校验：应为 width * height * 4（RGBA8）。
    let expected_len: usize = (width as usize) * (height as usize) * 4;
    assert_eq!(
        fb.pixels.len(),
        expected_len,
        "Framebuffer.pixels 长度应为 width*height*4 ({}), 实际 {}",
        expected_len,
        fb.pixels.len()
    );

    let mut out: Vec<u8> = Vec::new();

    // 1) PNG 签名。
    out.extend_from_slice(&PNG_SIGNATURE);

    // 2) IHDR chunk。
    //    数据布局（13 字节）：width(4B) height(4B) bitDepth(1B=8)
    //    colorType(1B=6 RGBA) compression(1B=0) filter(1B=0) interlace(1B=0)。
    let mut ihdr: Vec<u8> = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: 6 = RGBA
    ihdr.push(0); // compression method: 0
    ihdr.push(0); // filter method: 0
    ihdr.push(0); // interlace method: 0
    append_chunk(&mut out, b"IHDR", &ihdr);

    // 3) 组装 deflate 原始数据：每条扫描行前加 0x00 过滤字节，后接该行 RGBA 像素。
    let stride: usize = (width as usize) * 4;
    let mut raw: Vec<u8> = Vec::with_capacity((stride + 1) * (height as usize));
    for y in 0..height {
        raw.push(0x00); // 过滤字节：无过滤（None）
        let row_start: usize = (y as usize) * stride;
        raw.extend_from_slice(&fb.pixels[row_start..row_start + stride]);
    }

    // 4) IDAT chunk：zlib 流（CMF=0x78, FLG=0x01）+ 存储型 deflate 数据 + adler32。
    let mut idat: Vec<u8> = Vec::new();
    idat.push(0x78); // CMF: deflate, 32K window
    idat.push(0x01); // FLG: FLEVEL=0, 无字典 → 组成合法 zlib 头

    // 存储型（BTYPE=00）deflate 流：可能因数据过长被拆分为多个 64KiB 存储块，
    // 仅最后一个块置 BFINAL=1。
    let total: usize = raw.len();
    let mut offset: usize = 0;
    while offset < total {
        let block_len: usize = (total - offset).min(MAX_STORED_BLOCK);
        let is_last: bool = (offset + block_len) >= total;

        // 块头字节：BFINAL(1bit) + BTYPE(2bit=00) → 仅末块为 0x01，其余为 0x00。
        idat.push(if is_last { 0x01 } else { 0x00 });

        let len16: u16 = block_len as u16; // block_len ≤ 65535，安全
        idat.extend_from_slice(&len16.to_le_bytes()); // LEN (小端)
        idat.extend_from_slice(&(!len16).to_le_bytes()); // NLEN (小端，~LEN)
        idat.extend_from_slice(&raw[offset..offset + block_len]);

        offset += block_len;
    }

    // zlib 流尾：adler32（对原始扫描行数据计算）。
    idat.extend_from_slice(&adler32(&raw).to_be_bytes());

    append_chunk(&mut out, b"IDAT", &idat);

    // 5) IEND chunk（空数据）。
    append_chunk(&mut out, b"IEND", &[]);

    out
}

// ===========================================================================
//  公开 API
// ===========================================================================

/// 把帧缓冲写成一个 PNG 文件（RGBA8，自写编码器，使用「存储型 / 无压缩」zlib）。
///
/// # 参数
/// * `path` — 目标文件路径（含 `.png` 后缀）。
/// * `fb`   — 源 `Framebuffer`（RGBA8，长度须为 `width*height*4`）。
///
/// # 返回
/// `io::Result<()>`，失败时透传底层文件系统错误。
pub fn save_png(path: &str, fb: &Framebuffer) -> io::Result<()> {
    let bytes: Vec<u8> = encode_png(fb);
    let mut file: File = File::create(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(())
}

/// 把多帧写成一个目录：`frame_0001.png` ... 再加上 `manifest.json`。
///
/// # 参数
/// * `dir`      — 输出目录（不存在则递归创建）。
/// * `frames`   — 帧序列（`Framebuffer` 切片）。
/// * `fps`      — 帧率（写入 manifest）。
/// * `duration` — 时长（秒，写入 manifest）。
///
/// # 返回
/// `io::Result<()>`。
pub fn save_frames(dir: &str, frames: &[Framebuffer], fps: u32, duration: f32) -> io::Result<()> {
    fs::create_dir_all(dir)?;

    // 逐帧写出 frame_{i+1:04}.png。
    for (i, fb) in frames.iter().enumerate() {
        let file_name: String = format!("frame_{:04}.png", i + 1);
        let full_path: String = Path::new(dir)
            .join(file_name)
            .to_string_lossy()
            .into_owned();
        save_png(&full_path, fb)?;
    }

    // 生成 manifest.json（手工拼装合法 JSON，不引第三方库）。
    // 约定所有帧尺寸一致，取首帧宽高；空序列则回退为 0。
    let (width, height): (u32, u32) = frames
        .first()
        .map(|fb| (fb.width, fb.height))
        .unwrap_or((0, 0));

    let manifest: String = format!(
        "{{\"width\":{},\"height\":{},\"fps\":{},\"duration\":{},\"count\":{}}}",
        width,
        height,
        fps,
        duration,
        frames.len()
    );

    let manifest_path: String = Path::new(dir)
        .join("manifest.json")
        .to_string_lossy()
        .into_owned();
    let mut file: File = File::create(manifest_path)?;
    file.write_all(manifest.as_bytes())?;
    file.flush()?;

    Ok(())
}

// ===========================================================================
//  单元测试（可选）：生成小图并断言文件头/结构合法。
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个 2x2 的红绿蓝黄测试帧。
    fn make_test_frame() -> Framebuffer {
        Framebuffer {
            width: 2,
            height: 2,
            pixels: vec![
                // 行 0
                255, 0, 0, 255, // 红
                0, 255, 0, 255, // 绿
                // 行 1
                0, 0, 255, 255, // 蓝
                255, 255, 0, 255, // 黄
            ],
        }
    }

    #[test]
    fn test_crc32_known_vector() {
        // 标准参考向量：CRC32("123456789") 必须为 0xCBF43926（权威校验值）。
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // "IHDR" 的 CRC32（与 python zlib.crc32 一致）。
        assert_eq!(crc32(b"IHDR"), 0xA8A1_AE0A);
    }

    #[test]
    fn test_adler32_known_vector() {
        // 空数据 adler32 == 1。
        assert_eq!(adler32(&[]), 1);
        // 标准参考：adler32(b"Wikipedia") == 0x11E60398。
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn test_png_signature_and_chunks() {
        let fb: Framebuffer = make_test_frame();
        let bytes: Vec<u8> = encode_png(&fb);

        // 1) 签名正确。
        assert_eq!(&bytes[0..8], &PNG_SIGNATURE);

        // 2) 第一个 chunk 为 IHDR，长度为 13，类型串正确。
        let len: u32 = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(len, 13);
        assert_eq!(&bytes[12..16], b"IHDR");

        // 3) IHDR 内容：宽=2, 高=2, bitDepth=8, colorType=6。
        let mut cursor: usize = 16;
        let w: u32 = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let h: u32 = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(bytes[cursor], 8); // bit depth
        assert_eq!(bytes[cursor + 1], 6); // color type RGBA
        assert_eq!(bytes[cursor + 2], 0); // compression
        assert_eq!(bytes[cursor + 3], 0); // filter
        assert_eq!(bytes[cursor + 4], 0); // interlace

        // 4) 末尾为 IEND chunk（长度 0，类型 "IEND"）。
        let iend_pos: usize = bytes.len() - 12; // 8(IEND 头) + 4(CRC)
        assert_eq!(&bytes[iend_pos + 4..iend_pos + 8], b"IEND");
        let iend_len: u32 = u32::from_be_bytes(bytes[iend_pos..iend_pos + 4].try_into().unwrap());
        assert_eq!(iend_len, 0);
    }

    #[test]
    fn test_save_png_to_file() {
        let fb: Framebuffer = make_test_frame();
        let tmp: &str = "/tmp/prism_io_test.png";
        save_png(tmp, &fb).expect("写入 PNG 应成功");

        let written: Vec<u8> = fs::read(tmp).expect("应可读回");
        assert_eq!(&written[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_save_frames_structure() {
        let frames: Vec<Framebuffer> = vec![make_test_frame(), make_test_frame()];
        let dir: &str = "/tmp/prism_io_frames";
        save_frames(dir, &frames, 24, 0.0833).expect("写入帧序列应成功");

        assert!(Path::new("/tmp/prism_io_frames/frame_0001.png").exists());
        assert!(Path::new("/tmp/prism_io_frames/frame_0002.png").exists());
        assert!(Path::new("/tmp/prism_io_frames/manifest.json").exists());

        let manifest: String = fs::read_to_string("/tmp/prism_io_frames/manifest.json").unwrap();
        assert!(manifest.contains("\"width\":2"));
        assert!(manifest.contains("\"fps\":24"));
        assert!(manifest.contains("\"count\":2"));
    }

    #[test]
    fn crc32_empty_input_is_zero() {
        // CRC-32 对空输入的参考值为 0。
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn save_png_signature_roundtrip_temp_dir() {
        // 写到 std::env::temp_dir() 下，用 std::fs::read 读回校验 PNG 签名。
        let fb: Framebuffer = make_test_frame();
        let path: std::path::PathBuf = std::env::temp_dir().join("prism_io_sig_test.png");
        let path_str: &str = path.to_str().expect("临时路径应为合法 UTF-8");

        save_png(path_str, &fb).expect("写入临时 PNG 应成功");
        let bytes: Vec<u8> = fs::read(path_str).expect("读回 PNG 应成功");

        assert_eq!(
            &bytes[0..8],
            &[0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A][..],
            "save_png 生成文件的头 8 字节必须是 PNG 签名"
        );
        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn manifest_json_fields_exact() {
        let frames: Vec<Framebuffer> = vec![make_test_frame(), make_test_frame()];
        let dir: std::path::PathBuf = std::env::temp_dir().join("prism_io_manifest_test");
        let dir_str: &str = dir.to_str().expect("临时路径应为合法 UTF-8");

        save_frames(dir_str, &frames, 4, 2.0).expect("写入帧序列应成功");

        // 逐帧文件名与 manifest 均应存在。
        assert!(Path::new(dir_str).join("frame_0001.png").exists());
        assert!(Path::new(dir_str).join("frame_0002.png").exists());

        // manifest.json 内容精确校验（键序固定、手工生成）。
        let manifest: String =
            fs::read_to_string(dir.join("manifest.json")).expect("读回 manifest 应成功");
        assert_eq!(
            manifest, "{\"width\":2,\"height\":2,\"fps\":4,\"duration\":2,\"count\":2}",
            "manifest.json 的 width/height/fps/duration/count 字段应精确匹配"
        );
    }
}
