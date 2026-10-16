//! 成片后处理链接线（v3-B 任务「画质三件套·后处理接线」）。
//!
//! # 背景（缺陷根因）
//! 成片验收「塑料感、弱」三大根因之一：**成片管线零美术加工**——引擎
//! 能力不缺（prism-postfx 12 效果齐备），是片子没用：v2 全片 postfx
//! 使用量为 0，2D 场景帧直接 JPEG 编码，S7 只有降噪 + tonemap。
//!
//! # 本模块职责
//! 1. [`post_chain`]：2D 场景帧（RGBA8，显示域）在 JPEG 编码前过
//!    **bloom → grade → vignette** 三级链（任务书点名三件套）；
//! 2. [`encode_png_rgba8`]：预览交付的本地 PNG 编码器（存储型 deflate，
//!    零新依赖——`prism-io::encode_png` 需 `prism-gpu::Framebuffer`，
//!    而 prism-gpu 非 prism-author 依赖，example 不可直调）；
//! 3. 预览帧落盘（`PRISM_PREVIEW_S2` 机制见主入口）。
//!
//! # 硬约束（任务书字面）
//! - **纯函数确定性**：`(输入帧, 参数) → (输出帧)`，无随机、无内部状态
//!   ⇒ 全片重渲染 MD5 可复现；
//! - **不改变帧尺寸**：输出与输入同为 `w×h×4`；
//! - **alpha 恒 255**：链末强制回写（bloom 的亮部提取对 A 通道同样生效，
//!   叠加后可能越界，故出口统一钉死 255）；
//! - **显示域执行**：2D 帧是 sRGB 编码值（0..1）；bloom 阈值/强度按
//!   显示域口径取值（任务书初值 0.55 / 0.35），与 S7 `post_process` 内
//!   补的 bloom 同域同参（scene3d.rs 复用本模块常量）。
//!
//! # 级联不变式
//! 链序固定 `bloom → grade → vignette`：bloom 必须在 grade 前（先光学
//! 泛光后调色，高光先扩散再整体分级），vignette 最后（压暗是镜头末端
//! 效应）。三步均为 `TileEffect::eval_window` 纯函数，组合仍是纯函数。

use prism_io::fbuf::Fbuf16F;
use prism_postfx::effects::{Bloom, Grade, Vignette};
use prism_render::tile::TileEffect;

// ---------------------------------------------------------------------------
// 链参数（成片管线唯一真值；scene3d.rs 的 S7 bloom 复用同值）
// ---------------------------------------------------------------------------

/// bloom 亮部阈值（显示域；任务书初值 0.55）。
pub const BLOOM_THRESHOLD: f32 = 0.55;
/// bloom 扩散半径（全分辨率 box blur，像素）。
pub const BLOOM_RADIUS: u32 = 4;
/// bloom 泛光注入强度（线性域加算系数；任务书初值 0.35）。
pub const BLOOM_INTENSITY: f32 = 0.35;
/// grade 通道增益（轻度冷蓝：抑红提蓝——「电影感不塑料」的底色）。
pub const GRADE_SLOPE: [f32; 3] = [0.97, 1.00, 1.05];
/// grade 对比度（绕 0.5 枢轴；轻度）。
pub const GRADE_CONTRAST: f32 = 1.06;
/// grade 饱和度（轻度提振）。
pub const GRADE_SATURATION: f32 = 1.10;
/// vignette 最大压暗强度（轻度）。
pub const VIGNETTE_STRENGTH: f32 = 0.28;
/// vignette 衰减起点（归一化半径占比）。
pub const VIGNETTE_INNER: f32 = 0.55;

/// 成片后处理链（三效果实例；`Copy` 纯参数，无状态）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmPostChain {
    /// 镜头光学泛光（最关键的一项）。
    pub bloom: Bloom,
    /// 色彩分级（轻度冷蓝）。
    pub grade: Grade,
    /// 径向压暗（轻度）。
    pub vignette: Vignette,
}

impl Default for FilmPostChain {
    fn default() -> Self {
        FilmPostChain {
            bloom: Bloom {
                threshold: BLOOM_THRESHOLD,
                radius: BLOOM_RADIUS,
                intensity: BLOOM_INTENSITY,
            },
            grade: Grade {
                exposure: 1.0,
                contrast: GRADE_CONTRAST,
                saturation: GRADE_SATURATION,
                slope: GRADE_SLOPE,
                offset: [0.0, 0.0, 0.0],
            },
            vignette: Vignette {
                strength: VIGNETTE_STRENGTH,
                inner: VIGNETTE_INNER,
            },
        }
    }
}

impl FilmPostChain {
    /// 对一帧 RGBA8（显示域，行主序，`w×h×4` 字节）执行后处理链。
    ///
    /// 非法尺寸（长度不匹配 / 零宽高）原样返回（不 panic；调用方断言
    /// 帧尺寸，此处防御性兜底）。输出尺寸与输入一致，alpha 恒 255。
    pub fn apply_rgba8(&self, frame: &[u8], w: u32, h: u32) -> Vec<u8> {
        let need = (w as usize) * (h as usize) * 4;
        if frame.len() != need || w == 0 || h == 0 {
            return frame.to_vec();
        }
        // RGBA8（显示域 0..1）→ Fbuf16F（f16 存储 / f32 计算）。
        let mut fbuf = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y as usize) * (w as usize) + x as usize) * 4;
                fbuf.set(
                    x,
                    y,
                    [
                        frame[i] as f32 / 255.0,
                        frame[i + 1] as f32 / 255.0,
                        frame[i + 2] as f32 / 255.0,
                        1.0,
                    ],
                );
            }
        }
        // 级联（固定序）：bloom → grade → vignette。
        let bloomed = self.bloom.eval_window(&fbuf);
        let graded = self.grade.eval_window(&bloomed);
        let vignetted = self.vignette.eval_window(&graded);
        // → RGBA8（clamp01 × 255 四舍五入；alpha 钉死 255）。
        let mut out = vec![0u8; need];
        for y in 0..h {
            for x in 0..w {
                let c = vignetted.get(x, y);
                let i = ((y as usize) * (w as usize) + x as usize) * 4;
                out[i] = quantize_u8(c[0]);
                out[i + 1] = quantize_u8(c[1]);
                out[i + 2] = quantize_u8(c[2]);
                out[i + 3] = 255;
            }
        }
        out
    }
}

/// 便捷入口：默认链参数（成片唯一真值）。
pub fn post_chain(frame: &[u8], w: u32, h: u32) -> Vec<u8> {
    FilmPostChain::default().apply_rgba8(frame, w, h)
}

/// S7（3D 光追场景）在成片输出序列中的**体起点**帧索引。
///
/// 2D 侧链接线只覆盖 S1–S6（含其间转场帧）；S7 纯内容区的画质由
/// [`super::scene3d::post_process`] 内部 bloom 承担（显示域、同参），
/// **不再**过本链——否则高光经两次泛光（塑料晕，恰是本次要消灭的
/// 缺陷）。S6→S7 的 15 帧过渡区按 2D 侧处理（该区 94–100% 为 S6
/// 内容，且避免边界单帧跳变）。
///
/// 推导（左结合折叠转场拼接，`t = round(0.5s × 30fps) = 15`）：
/// `acc_k = Σ_{i≤k} len_i − (k−1)·t`，S7 体起点 = `acc_6`
/// = (248+308+496+496+496+436) − 5×15 = **2405**。
pub fn s7_body_start_frame() -> u32 {
    let t = super::common::TRANS_FRAMES as u32; // = round(0.5 × 30) = 15
    let before_s7 = 6usize; // S7 为第 7 段（索引 6）
    let acc: u32 = super::common::SCENE_FRAMES[..before_s7].iter().sum::<u32>()
        - (before_s7 as u32 - 1) * t;
    acc
}

/// 显示域量化（clamp01 × 255 四舍五入；与 `Fbuf16F::to_framebuffer`
/// 的 `channel_to_u8` 同式）。
fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---------------------------------------------------------------------------
// 本地 PNG 编码器（存储型 deflate；零依赖）
// ---------------------------------------------------------------------------

/// PNG 文件签名（8 B）。
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

/// CRC-32 表（PNG 多项式 0xEDB88320 反射式；`const fn` 构建，确定性）。
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0usize;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// CRC-32（PNG 口径：init/final 各一次 0xFFFFFFFF 异或）。
fn crc32(data: &[u8]) -> u32 {
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Adler-32（zlib 校验；mod 65521）。
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// 写一个 PNG 块（length + type + data + crc）。
fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// 编码 RGBA8 为 PNG（bit depth 8 / color type 6 / 无交错）。
///
/// 压缩用**存储型 deflate**（BTYPE=00）：不经 LZ77，体积约为原图 1.05
/// 倍（每 65535 B 一个块 + 5 B 块头）——预览帧 960×540×4 ≈ 2MB → 落盘
/// ~2.1MB/帧，3 帧可忽略；换来零依赖 + 逐位确定的编解码。
pub fn encode_png_rgba8(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() + rgba.len() / 20 + 128);
    out.extend_from_slice(&PNG_SIGNATURE);
    // IHDR：宽 / 高 / 位深 8 / 色彩型 6（RGBA）/ 压缩 0 / 滤波 0 / 交错 0。
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type: truecolor + alpha
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method: adaptive（本编码器逐行 filter 0）
    ihdr.push(0); // interlace: none
    push_chunk(&mut out, b"IHDR", &ihdr);
    // 原始扫描线流：逐行 [filter=0] + 行字节。
    let row_bytes = (w as usize) * 4;
    let mut raw = Vec::with_capacity((row_bytes + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0); // filter type 0（None）
        raw.extend_from_slice(&rgba[y * row_bytes..(y + 1) * row_bytes]);
    }
    // zlib 流：头（CMF=0x78 / FLG=0x01，30721 % 31 == 0）+ 存储块 + adler32。
    let mut zlib = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
    zlib.push(0x78);
    zlib.push(0x01);
    let mut off = 0usize;
    while off < raw.len() {
        let end = (off + 65535).min(raw.len());
        let block = &raw[off..end];
        let is_last = end == raw.len();
        zlib.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
        off = end;
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());
    push_chunk(&mut out, b"IDAT", &zlib);
    push_chunk(&mut out, b"IEND", &[]);
    out
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试帧：暗底 + 一条亮「曲线」（水平亮带，模拟成片亮曲线；
    /// 取水平带而非斜线——左右探针须关于亮带严格等距，对称断言才
    /// 有几何基准），模拟成片帧。
    fn test_frame(w: u32, h: u32) -> Vec<u8> {
        let mut f = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                // 背景：暗蓝灰渐变。
                f[i] = (10 + x / 40) as u8;
                f[i + 1] = (12 + y / 60) as u8;
                f[i + 2] = (20 + x / 50) as u8;
                f[i + 3] = 255;
                // 亮曲线：y = h/2 的水平亮带（亮黄）。
                if (y as f32 - h as f32 * 0.5).abs() < 2.5 {
                    f[i] = 255;
                    f[i + 1] = 240;
                    f[i + 2] = 180;
                }
            }
        }
        f
    }

    /// 尺寸与 alpha 不变式（任务书硬约束）。
    #[test]
    fn chain_preserves_dimensions_and_forces_alpha_255() {
        let (w, h) = (64u32, 36u32);
        let frame = test_frame(w, h);
        let out = post_chain(&frame, w, h);
        assert_eq!(out.len(), frame.len(), "尺寸不得改变");
        for i in (3..out.len()).step_by(4) {
            assert_eq!(out[i], 255, "alpha 必须恒 255（像素 {}）", i / 4);
        }
        // 非法尺寸：原样返回（不 panic）。
        assert_eq!(post_chain(&frame, 32, 36), frame, "长度不匹配须原样返回");
        assert!(post_chain(&[], 0, 0).is_empty());
    }

    /// 确定性：两次运行逐字节一致（纯函数契约 ⇒ 全片 MD5 可复现）。
    #[test]
    fn chain_is_bitwise_deterministic() {
        let (w, h) = (96u32, 54u32);
        let frame = test_frame(w, h);
        let a = post_chain(&frame, w, h);
        let b = post_chain(&frame, w, h);
        assert_eq!(a, b, "同输入必须逐字节一致");
        // 输入不被修改（纯函数：只读输入）。
        let before = frame.clone();
        let _ = post_chain(&frame, w, h);
        assert_eq!(frame, before, "输入帧不得被改动");
    }

    /// bloom 辉光：亮曲线邻域像素必须比无 bloom 时更亮（任务书验收
    /// 「bloom 在亮曲线邻域产生辉光」）。
    #[test]
    fn bloom_glows_around_bright_curve() {
        let (w, h) = (96u32, 54u32);
        let frame = test_frame(w, h);
        let chain = FilmPostChain::default();
        let with_bloom = chain.apply_rgba8(&frame, w, h);
        // 对照：仅 grade + vignette（跳过 bloom）。
        let no_bloom = FilmPostChain {
            bloom: Bloom { threshold: 1.0, radius: 0, intensity: 0.0 },
            ..chain
        }
        .apply_rgba8(&frame, w, h);
        // 亮带正下方 6px 处（邻域、非墨迹本身）。
        let px = 48u32;
        let py = 27u32 + 6;
        let i = ((py * w + px) * 4) as usize;
        assert!(
            with_bloom[i] > no_bloom[i],
            "曲线邻域应更亮：bloom {} vs 无 bloom {}",
            with_bloom[i],
            no_bloom[i]
        );
        // 辉光对称：左右邻域增益应接近（box blur 各向同性；grade/vignette
        // 对称轴上一致）。
        let l = ((py * w + (px - 8)) * 4) as usize;
        let r = ((py * w + (px + 8)) * 4) as usize;
        let dl = with_bloom[l] as i32 - no_bloom[l] as i32;
        let dr = with_bloom[r] as i32 - no_bloom[r] as i32;
        assert!((dl - dr).abs() <= 2, "辉光应左右对称（{dl} vs {dr}）");
    }

    /// 级联不变式：链输出 == 手动按 bloom → grade → vignette 序逐步执行。
    #[test]
    fn cascade_order_is_exactly_bloom_grade_vignette() {
        let (w, h) = (48u32, 32u32);
        let frame = test_frame(w, h);
        let chain = FilmPostChain::default();
        let chained = chain.apply_rgba8(&frame, w, h);
        // 手动复现同序。
        let mut fbuf = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                fbuf.set(x, y, [frame[i] as f32 / 255.0, frame[i + 1] as f32 / 255.0, frame[i + 2] as f32 / 255.0, 1.0]);
            }
        }
        let b = chain.bloom.eval_window(&fbuf);
        let g = chain.grade.eval_window(&b);
        let v = chain.vignette.eval_window(&g);
        let mut manual = vec![0u8; frame.len()];
        for y in 0..h {
            for x in 0..w {
                let c = v.get(x, y);
                let i = ((y * w + x) * 4) as usize;
                manual[i] = quantize_u8(c[0]);
                manual[i + 1] = quantize_u8(c[1]);
                manual[i + 2] = quantize_u8(c[2]);
                manual[i + 3] = 255;
            }
        }
        assert_eq!(chained, manual, "链序必须恰为 bloom → grade → vignette");
    }

    /// 中性 grade 恒等性抽查：全中性 Grade 下链输出仅 bloom+vignette 效应
    /// （bloom 低于阈值区恒等 + vignette 中心恒等 ⇒ 暗帧近恒等）。
    #[test]
    fn neutral_grade_leaves_chain_transparent_to_grading() {
        let (w, h) = (32u32, 32u32);
        // 全暗帧（远低于 bloom 阈值）：bloom 恒等；中心区 vignette 恒等。
        let frame = vec![10u8; (w * h * 4) as usize];
        let neutral = FilmPostChain {
            grade: Grade::default(),
            ..FilmPostChain::default()
        };
        let out = neutral.apply_rgba8(&frame, w, h);
        // 中心像素：vignette r=0 < inner ⇒ factor=1；bloom 无亮部 ⇒ 恒等；
        // grade 中性 ⇒ 恒等。故中心像素应与输入一致（容 1/255 量化差）。
        let c = ((16 * w + 16) * 4) as usize;
        for ch in 0..3 {
            assert!(
                (out[c + ch] as i32 - frame[c + ch] as i32).abs() <= 1,
                "中心像素应近恒等：{} vs {}",
                out[c + ch],
                frame[c + ch]
            );
        }
    }

    // ---- PNG 编码器 ----

    /// PNG 结构：签名 / IHDR 字段 / 块序列 / CRC 全部合法。
    #[test]
    fn png_structure_is_valid() {
        let (w, h) = (3u32, 2u32);
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|i| (i * 7 % 256) as u8).collect();
        let png = encode_png_rgba8(&rgba, w, h);
        assert_eq!(&png[..8], &PNG_SIGNATURE, "签名");
        // 第一块 IHDR。
        assert_eq!(&png[8..12], &(13u32).to_be_bytes(), "IHDR 长度");
        assert_eq!(&png[12..16], b"IHDR", "IHDR 类型");
        assert_eq!(&png[16..20], &w.to_be_bytes(), "宽");
        assert_eq!(&png[20..24], &h.to_be_bytes(), "高");
        assert_eq!(png[24], 8, "位深");
        assert_eq!(png[25], 6, "色彩型 RGBA");
        assert_eq!(png[26], 0, "压缩");
        assert_eq!(png[27], 0, "滤波");
        assert_eq!(png[28], 0, "交错");
        // 块遍历：每块 length/type/data/crc 自洽。
        let mut off = 8usize;
        let mut kinds = Vec::new();
        while off < png.len() {
            let len = u32::from_be_bytes([png[off], png[off + 1], png[off + 2], png[off + 3]]) as usize;
            let kind = &png[off + 4..off + 8];
            let data = &png[off + 8..off + 8 + len];
            let crc_stored = u32::from_be_bytes([
                png[off + 8 + len],
                png[off + 9 + len],
                png[off + 10 + len],
                png[off + 11 + len],
            ]);
            let mut crc_input = Vec::with_capacity(4 + len);
            crc_input.extend_from_slice(kind);
            crc_input.extend_from_slice(data);
            assert_eq!(crc_stored, crc32(&crc_input), "{:?} 块 CRC", kind);
            kinds.push(String::from_utf8_lossy(kind).to_string());
            off += 12 + len;
        }
        assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"], "块序列");
    }

    /// PNG 往返：存储型 deflate 自描述 ⇒ 测试内解出原始扫描线，逐字节
    /// 比对输入（无损）。
    #[test]
    fn png_roundtrip_stored_deflate() {
        let (w, h) = (5u32, 4u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for (i, b) in rgba.iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        let png = encode_png_rgba8(&rgba, w, h);
        // 定位 IDAT data。
        let mut off = 8usize;
        let mut idat: Option<&[u8]> = None;
        while off < png.len() {
            let len = u32::from_be_bytes([png[off], png[off + 1], png[off + 2], png[off + 3]]) as usize;
            let kind = &png[off + 4..off + 8];
            if kind == b"IDAT" {
                idat = Some(&png[off + 8..off + 8 + len]);
            }
            off += 12 + len;
        }
        let z = idat.expect("须有 IDAT");
        assert_eq!(&z[..2], &[0x78, 0x01], "zlib 头");
        // 解存储块。
        let mut raw = Vec::new();
        let mut p = 2usize;
        while p < z.len() - 4 {
            let header = z[p];
            assert_eq!(header & 0x06, 0, "BTYPE 必须为 00（存储）");
            p += 1;
            let len = u16::from_le_bytes([z[p], z[p + 1]]) as usize;
            let nlen = u16::from_le_bytes([z[p + 2], z[p + 3]]);
            assert_eq!(nlen, !(len as u16), "NLEN 须为 LEN 取反");
            p += 4;
            raw.extend_from_slice(&z[p..p + len]);
            p += len;
            if header & 0x01 != 0 {
                break;
            }
        }
        // adler32 校验。
        let adler_stored = u32::from_be_bytes([z[p], z[p + 1], z[p + 2], z[p + 3]]);
        assert_eq!(adler_stored, adler32(&raw), "adler32");
        // 原始流 = 逐行 [filter 0] + 行字节。
        let row_bytes = (w as usize) * 4;
        assert_eq!(raw.len(), (row_bytes + 1) * h as usize);
        for y in 0..h as usize {
            assert_eq!(raw[y * (row_bytes + 1)], 0, "filter 0");
            assert_eq!(
                &raw[y * (row_bytes + 1) + 1..(y + 1) * (row_bytes + 1)],
                &rgba[y * row_bytes..(y + 1) * row_bytes],
                "第 {y} 行像素须逐字节还原"
            );
        }
    }

    /// 大图分块：> 65535 B 的原始流必须多存储块（2000×40 RGBA ⇒ 原始流
    /// 320040 B ⇒ 5 块）且总长自洽。
    #[test]
    fn png_multi_block_for_large_frames() {
        let (w, h) = (2000u32, 40u32);
        let rgba = vec![128u8; (w * h * 4) as usize];
        let png = encode_png_rgba8(&rgba, w, h);
        // 数 IDAT 内的存储块（每块头 5 B：1 + 2 + 2）。
        let mut off = 8usize;
        let mut idat_len = 0usize;
        while off < png.len() {
            let len = u32::from_be_bytes([png[off], png[off + 1], png[off + 2], png[off + 3]]) as usize;
            if &png[off + 4..off + 8] == b"IDAT" {
                idat_len = len;
            }
            off += 12 + len;
        }
        let raw_len = (w as usize) * 4 * h as usize + h as usize;
        let expected_blocks = raw_len.div_ceil(65535);
        // IDAT data = 2（zlib 头）+ 5×块数 + raw_len + 4（adler）。
        assert_eq!(
            idat_len,
            2 + 5 * expected_blocks + raw_len + 4,
            "存储块数 {expected_blocks}"
        );
    }

    /// S7 体起点推导锁：2405；且 2D 侧帧数 + S7 体帧数 = 全片总帧数
    /// （分镜帧表漂移时边界不得静默错位）。
    #[test]
    fn s7_body_start_frame_matches_frame_table() {
        use super::super::common;
        let start = s7_body_start_frame();
        assert_eq!(
            start, 2405,
            "S7 体起点必须为 2405（帧表 [248,308,496,496,496,436,310]，t=15）"
        );
        let s7_body = common::SCENE_FRAMES[6] as u32 - common::TRANS_FRAMES as u32;
        assert_eq!(
            start + s7_body,
            common::TOTAL_FRAMES as u32,
            "2D 侧 {start} + S7 体 {s7_body} 必须回 全片 {}",
            common::TOTAL_FRAMES
        );
    }
}
