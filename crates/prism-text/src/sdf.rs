//! SDF 字形烘焙 + atlas（D4）。
//!
//! 与 [`crate::raster`] 的覆盖率位图互补：覆盖率位图按**目标像素尺寸**光栅化
//! （放大即糊）；SDF 位图按**固定探测预算**烘焙一次，任意尺度由 shader 双线性
//! 采样 + smoothstep 重建锐利边缘（`crates/prism-gpu/shaders/text.wgsl`）。
//!
//! # 管线
//! 1. 轮廓按 `bake_size` 光栅化得高分辨率覆盖率位图（复用 [`crate::raster`]，
//!    不重复造光栅器）；
//! 2. 覆盖率 → 有符号距离（signed distance）：对每个像素取**最近边缘像素**的
//!    欧氏距离，符号由覆盖率（>0.5 视为内侧）判定；
//! 3. 距离（像素）归一化到 `[0,1]`：`enc = 0.5 + d / (2·px_range)`，钳制；
//! 4. 写入单通道 R8 atlas，`GlyphAtlas` 记录每字形的盒偏移与 UV。
//!
//! 覆盖率→距离用的是**精确的欧氏最近边缘**（用覆盖率梯度带作边缘），故放大时
//! 只需单通道 R8 即可恢复锐利边缘——与 MSDF 的多通道不同（Mini Code：够用即可）。
//!
//! # 确定性契约
//! 全管线是 (字体字节, gid, bake_size, px_range) 的纯函数：无随机源、无 libm
//! （只用 ±×÷ 与 IEEE 精确 `sqrt`/`floor`/`round`），同参数两遍逐字节一致。

use crate::raster::{render_glyph, GlyphBitmap};
use crate::ttf::Font;

/// SDF 烘焙参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfParams {
    /// 烘焙光栅化的 em 框像素高（越大越精细，atlas 越大）。
    pub bake_size: f32,
    /// 距离场半程（像素）：编码区间 `d ∈ [−px_range, +px_range] → [0,1]`。
    pub px_range: f32,
    /// atlas 每格的统一边长（像素，方形对齐；含 px_range 外扩）。
    pub pad: u32,
}

impl Default for SdfParams {
    fn default() -> Self {
        SdfParams { bake_size: 48.0, px_range: 8.0, pad: 4 }
    }
}

impl SdfParams {
    /// 构造并做参数防呆（非法则收束到默认）。
    ///
    /// # 错误码
    /// `bake_size` / `px_range` 非**有限**正数（`NaN` / `±Inf` / `≤ 0`）时收束
    /// 到默认值。此处必须用 [`f32::is_finite`] 而非仅 `is_nan`：`Inf > 0.0` 为真，
    /// 故 `Inf` 会穿透旧校验，随后 `bake_size.round() as u32` 饱和为 `u32::MAX`，
    /// 再经 `+ pad * 2 + 2` 溢出 panic（`T008`）。
    pub fn new(bake_size: f32, px_range: f32, pad: u32) -> Self {
        let bs = if bake_size.is_finite() && bake_size > 0.0 { bake_size } else { 48.0 };
        let pr = if px_range.is_finite() && px_range > 0.0 { px_range } else { 8.0 };
        // pad 亦做上界钳制：它是 u32，参与 `pad * 2 + 2` 与 `ext as i32`
        // 两条溢出敏感路径，过大值会在烘焙期 panic 而非构造期。
        let pd = pad.min(MAX_PAD);
        SdfParams { bake_size: bs, px_range: pr, pad: pd }
    }
}

/// `pad` 的合法上界（像素）：远大于任何实际 atlas 需求，仅用于把溢出变为可预期的
/// 饱和而非 panic（宪章 C18「永不 panic」）。
const MAX_PAD: u32 = 1024;

/// 单张位图（单格 SDF / 整张 atlas）的边长上界：保证 `edge * edge` 的字节数
/// 在 usize 预算内（4096² = 16 MiB）。栅格化侧另有 `raster::MAX_WINDOW_EDGE`
/// 同阶上界，两层保持一致。
const MAX_ATLAS_CELL: u32 = 4096;

/// 编码距离（像素）为 SDF 字节：`enc = 0.5 + d/(2·px_range)`，RNE 量化。
///
/// `d` 为有符号像素距离（内正外负）。`px_range` 外全部钳制到 0/255。
fn encode_distance(d: f32, px_range: f32) -> u8 {
    let t = 0.5 + d / (2.0 * px_range);
    let v = (t * 255.0 + 0.5).floor();
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v as u8
    }
}

/// 解码 SDF 字节为有符号距离（像素）——GPU shader 端 `decode` 的 CPU 镜像，
/// 供测试与自检对齐。
pub fn decode_distance(byte: u8, px_range: f32) -> f32 {
    (byte as f32 / 255.0 - 0.5) * (2.0 * px_range)
}

/// 单字形 SDF 位图。
#[derive(Debug, Clone, PartialEq)]
pub struct SdfBitmap {
    /// 光源栅化窗口（相对笔位/基线）的左偏移（像素）——含 px_range 外扩。
    pub left: i32,
    /// 窗口相对基线的上偏移（像素）——含 px_range 外扩。
    pub top: i32,
    /// 宽（像素，含两侧外扩）。
    pub width: u32,
    /// 高（像素，含上下外扩）。
    pub height: u32,
    /// SDF 编码（行主序，单通道 R8）。
    pub data: Vec<u8>,
    /// 编码用的 px_range（解码必须用它）。
    pub px_range: f32,
}

impl SdfBitmap {
    /// 空位图（无墨字形，如空格）。
    pub fn empty(px_range: f32) -> Self {
        SdfBitmap { left: 0, top: 0, width: 0, height: 0, data: Vec::new(), px_range }
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// 覆盖率位图 → 有符号距离场（像素）。
///
/// 边缘 = 覆盖率跨越 0.5 的像素（`cov > 0.5` 判内侧）。对窗口内每个像素，扫描
/// 全部边缘像素取最小欧氏距离 `r`，符号 = 内侧为正。空/全内/全外的退化位图退
/// 化为常量场（全 + 或全 −），仍确定。
///
/// 复杂度 O(W·H·E)（E 为边缘像素数）——烘焙一次性，可接受；且 E ≤ W·H。
fn coverage_to_sdf(bmp: &GlyphBitmap) -> Vec<f32> {
    let w = bmp.width as usize;
    let h = bmp.height as usize;
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    let inside = |p: u8| p > 127; // 覆盖率阈值 0.5（127/255 ≈ 0.498；用 127 对称）
    // 收集边缘像素中心（像素中心的连续坐标 = 下标 + 0.5）。
    let mut edges: Vec<(f32, f32, bool)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let c = bmp.alpha[y * w + x];
            let ins = inside(c);
            // 边缘判据：与左/上邻居符号相异（含越界视作外侧）。
            let up_diff = {
                let u = if y == 0 { false } else { inside(bmp.alpha[(y - 1) * w + x]) };
                u != ins
            };
            let left_diff = {
                let l = if x == 0 { false } else { inside(bmp.alpha[y * w + x - 1]) };
                l != ins
            };
            if up_diff || left_diff {
                edges.push((x as f32 + 0.5, y as f32 + 0.5, ins));
            }
        }
    }
    if edges.is_empty() {
        // 退化：全内 → 常量 +大；全外 → 常量 −大。用窗口对角线作幅度。
        let diag = ((w as f32) * (w as f32) + (h as f32) * (h as f32)).sqrt();
        let all_in = bmp.alpha.iter().all(|&c| inside(c));
        let val = if all_in { diag } else { -diag };
        for v in out.iter_mut() {
            *v = val;
        }
        return out;
    }
    for y in 0..h {
        for x in 0..w {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let ins = inside(bmp.alpha[y * w + x]);
            let mut best = f32::INFINITY;
            for &(ex, ey, _) in &edges {
                let dx = ex - px;
                let dy = ey - py;
                let d2 = dx * dx + dy * dy;
                if d2 < best {
                    best = d2;
                }
            }
            let r = best.sqrt();
            out[y * w + x] = if ins { r } else { -r };
        }
    }
    out
}

/// 烘焙单字形的 SDF 位图。
///
/// 步骤：以 `params.bake_size` 光栅化 → 距离场 → 外扩 `px_range` 边框（边框内
/// 距离按边界外插，填满编码区间）→ 编码 R8。越界 gid / 空字形 → 空位图。
pub fn bake_glyph_sdf(font: &Font, gid: u16, params: &SdfParams) -> SdfBitmap {
    let cov = render_glyph(font, gid, params.bake_size);
    if cov.is_empty() {
        return SdfBitmap::empty(params.px_range);
    }
    let dist = coverage_to_sdf(&cov);
    let cw = cov.width;
    let ch = cov.height;
    let ext = params.pad.max(1) as i32;
    // saturating_add：cw/ch 来自光栅化（`raster::MAX_WINDOW_EDGE` 钳到 4096），
    // pad 已被 MAX_PAD 钳到 1024；饱和而非回绕，保证即便异常组合也只得到
    // 钳制尺寸而非绕回的小尺寸（静默错图）。`width * height` 用 u64 中间量：
    // u32 乘积在 32 位目标上可能溢出。
    let width = cw.saturating_add((ext as u32).saturating_mul(2));
    let height = ch.saturating_add((ext as u32).saturating_mul(2));
    let cells = (width as u64).saturating_mul(height as u64);
    if cells > MAX_ATLAS_CELL as u64 * MAX_ATLAS_CELL as u64 {
        return SdfBitmap::empty(params.px_range);
    }
    let mut data = vec![0u8; cells as usize];
    for y in 0..height {
        for x in 0..width {
            let sx = x as i32 - ext;
            let sy = y as i32 - ext;
            // 越界时用边缘像素距离外插（clamp），保证边框填满。
            let d = if sx >= 0 && sx < cw as i32 && sy >= 0 && sy < ch as i32 {
                dist[(sy as usize) * (cw as usize) + (sx as usize)]
            } else {
                // 最近内部像素距离 + 到边界距离（保守外插，保持单调）。
                let cxi = sx.clamp(0, cw as i32 - 1) as usize;
                let cyi = sy.clamp(0, ch as i32 - 1) as usize;
                let base = dist[cyi * (cw as usize) + cxi];
                let dx = (sx - cxi as i32) as f32;
                let dy = (sy - cyi as i32) as f32;
                let extra = (dx * dx + dy * dy).sqrt();
                if base >= 0.0 {
                    base + extra
                } else {
                    base - extra
                }
            };
            data[(y * width + x) as usize] = encode_distance(d, params.px_range);
        }
    }
    SdfBitmap {
        left: cov.left - ext,
        top: cov.top - ext,
        width,
        height,
        data,
        px_range: params.px_range,
    }
}

/// atlas 中一个格子的定位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasSlot {
    /// 字形 id。
    pub gid: u16,
    /// atlas 内像素左偏移。
    pub x: u32,
    /// atlas 内像素上偏移。
    pub y: u32,
    /// 格宽（= 字形 SDF 宽 + 双倍 pad）。
    pub w: u32,
    /// 格高。
    pub h: u32,
    /// 绘制锚点：格左上相对笔位的 x 偏移（= 内层 SDF 的 left）。
    pub left: i32,
    /// 绘制锚点：格左上相对基线的 y 偏移。
    pub top: i32,
}

/// SDF 字形 atlas（单通道 R8，行主序）。
///
/// 简单**行装箱**（shelf packing）：按 gid 升序逐字形摆放，行满换行；确定且
/// 无空洞外的重排（同输入集恒定输出）。够用即可（Mini Code）。
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphAtlas {
    /// atlas 宽（像素）。
    pub width: u32,
    /// atlas 高（像素）。
    pub height: u32,
    /// 每个字形一个 R8 样本的编码距离。
    pub data: Vec<u8>,
    /// 编码用的 px_range（全 atlas 统一）。
    pub px_range: f32,
    /// 各字形槽位定位。
    pub slots: Vec<AtlasSlot>,
    /// 无墨字形（空格等）一并登记的 gid（无槽）。
    pub blanks: Vec<u16>,
}

impl GlyphAtlas {
    /// 按 gid 列表烘焙 atlas。
    ///
    /// `max_width` 为 atlas 宽上限（像素）；行装箱，高度按需增长。
    pub fn bake(
        font: &Font,
        gids: &[u16],
        params: &SdfParams,
        max_width: u32,
    ) -> GlyphAtlas {
        let mut ordered: Vec<u16> = gids.to_vec();
        ordered.sort_unstable();
        ordered.dedup();

        // `max_width` 是 atlas 宽**上限**（文档契约），不得被单格尺寸反向撑大：
        // 旧实现 `max_width.max(cell)` 在超大 pad（如 pad=1024 ⇒ cell=2098）时
        // 把 64px 的上限撑到 2098px，调用方拿到远超预期的 atlas。
        // 现以上限为准；装不下的单格字形（bw > mw）按"无墨"处理（见下）。
        let mw = max_width.max(1);

        // 第一趟：逐字形烘焙，记录尺寸并做行装箱布局。
        let mut slots: Vec<AtlasSlot> = Vec::new();
        let mut blanks: Vec<u16> = Vec::new();
        let mut cursor_x = 0u32;
        let mut cursor_y = 0u32;
        let mut row_h = 0u32;
        let mut baked: Vec<(u16, SdfBitmap)> = Vec::new();
        for &gid in &ordered {
            let bmp = bake_glyph_sdf(font, gid, params);
            if bmp.is_empty() {
                blanks.push(gid);
                continue;
            }
            let bw = bmp.width;
            let bh = bmp.height;
            // 装不下的单格（bw > mw）：按"无墨"处理。绝不放进去——否则槽位
            // 宽度超出 atlas 宽，`uv()` 会给出 u1 > 1，shader 采样越界。
            if bw > mw {
                blanks.push(gid);
                continue;
            }
            if cursor_x.saturating_add(bw) > mw && cursor_x > 0 {
                cursor_x = 0;
                cursor_y = cursor_y.saturating_add(row_h);
                row_h = 0;
            }
            slots.push(AtlasSlot {
                gid,
                x: cursor_x,
                y: cursor_y,
                w: bw,
                h: bh,
                left: bmp.left,
                top: bmp.top,
            });
            cursor_x = cursor_x.saturating_add(bw);
            row_h = row_h.max(bh);
            baked.push((gid, bmp));
        }
        let atlas_w = mw;
        let atlas_h = cursor_y.saturating_add(row_h);
        // u64 中间量：u32 乘积在 32 位目标上会溢出，且 mw 由调用方任意给定。
        let pixels = (atlas_w as u64).saturating_mul(atlas_h as u64);
        if pixels > MAX_ATLAS_CELL as u64 * MAX_ATLAS_CELL as u64 {
            return GlyphAtlas {
                width: 0,
                height: 0,
                data: Vec::new(),
                px_range: params.px_range,
                slots: Vec::new(),
                blanks,
            };
        }
        let mut data = vec![0u8; pixels as usize];
        for slot in &slots {
            let bmp = match baked.iter().find(|(g, _)| *g == slot.gid) {
                Some((_, b)) => b,
                None => continue,
            };
            for yy in 0..slot.h {
                for xx in 0..slot.w {
                    let sx = slot.x + xx;
                    let sy = slot.y + yy;
                    if sx >= atlas_w || sy >= atlas_h {
                        continue;
                    }
                    data[(sy * atlas_w + sx) as usize] =
                        bmp.data[(yy * slot.w + xx) as usize];
                }
            }
        }
        GlyphAtlas {
            width: atlas_w,
            height: atlas_h,
            data,
            px_range: params.px_range,
            slots,
            blanks,
        }
    }

    /// 查字形槽（无墨字形返回 None）。
    pub fn slot(&self, gid: u16) -> Option<&AtlasSlot> {
        self.slots.iter().find(|s| s.gid == gid)
    }

    /// 归一化 UV（左上原点，y 向下——与 `text.wgsl` 采样约定一致）。
    pub fn uv(&self, slot: &AtlasSlot) -> [f32; 4] {
        let u0 = slot.x as f32 / self.width as f32;
        let v0 = slot.y as f32 / self.height as f32;
        let u1 = (slot.x + slot.w) as f32 / self.width as f32;
        let v1 = (slot.y + slot.h) as f32 / self.height as f32;
        [u0, v0, u1, v1]
    }

    /// 采样某槽位的 SDF 字节（越界返回 0——即「远外侧」）。
    pub fn sample(&self, slot: &AtlasSlot, lx: u32, ly: u32) -> u8 {
        if lx >= slot.w || ly >= slot.h {
            return 0;
        }
        let sx = slot.x + lx;
        let sy = slot.y + ly;
        self.data[(sy * self.width + sx) as usize]
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::font_regular;

    #[test]
    fn encode_decode_round_trip_anchors() {
        let pr = 8.0;
        // d = 0 → 0.5 → floor(0.5*255+0.5) = 128。
        assert_eq!(encode_distance(0.0, pr), 128);
        // d = +px_range → 1.0 → 255。
        assert_eq!(encode_distance(pr, pr), 255);
        // d = −px_range → 0.0 → 0。
        assert_eq!(encode_distance(-pr, pr), 0);
        // 解码镜面对齐（±0.5 字节量化误差）。
        let d = 3.0;
        let dec = decode_distance(encode_distance(d, pr), pr);
        assert!((dec - d).abs() <= 2.0 * pr / 255.0 + 1e-3, "解码误差超界: {dec}");
    }

    #[test]
    fn bake_cjk_is_signed_and_monotone_through_edge() {
        let f = font_regular();
        let gid = f.glyph_index('中').expect("中 在子集内");
        let p = SdfParams::default();
        let sdf = bake_glyph_sdf(&f, gid, &p);
        assert!(!sdf.is_empty());
        assert_eq!(sdf.data.len(), (sdf.width * sdf.height) as usize);
        // 中心（竖笔内）应显著内侧：字节 > 128 + 半码。
        let cx = sdf.width / 2;
        let cy = sdf.height / 2;
        let center = sdf.data[(cy * sdf.width + cx) as usize];
        assert!(center > 140, "中心应内侧（远正），实际 {center}");
        // 四角（px_range 外扩边框）应显著外侧：字节 < 128 − 半码。
        let corner = sdf.data[0];
        assert!(corner < 118, "角落应外侧（远负），实际 {corner}");
        // 存在内→外跨越 0.5 编码（128）的过渡带（边缘）。
        let has_inside = sdf.data.iter().any(|&b| b > 180);
        let has_outside = sdf.data.iter().any(|&b| b < 80);
        assert!(has_inside && has_outside, "应有明显内外分区");
    }

    #[test]
    fn bake_is_deterministic_byte_wise() {
        let f = font_regular();
        let gid = f.glyph_index('永').expect("永 在子集内");
        let p = SdfParams::new(48.0, 8.0, 4);
        let a = bake_glyph_sdf(&f, gid, &p);
        let b = bake_glyph_sdf(&f, gid, &p);
        assert_eq!(a, b, "同参数逐字节一致");
    }

    #[test]
    fn distance_is_one_lipschitz_along_scanline() {
        // 5×5 实心块：中心(2,2)非边缘，到边缘像素距离 1；再往外递增。
        let mut alpha = vec![0u8; 25];
        for y in 1..4 {
            for x in 1..4 {
                alpha[y * 5 + x] = 255;
            }
        }
        let bmp = GlyphBitmap { left: 0, top: 0, width: 5, height: 5, alpha };
        let d = coverage_to_sdf(&bmp);
        let at = |x: usize, y: usize| d[y * 5 + x];
        // 中心内侧，距最近边缘（块内圈边缘）至少 1。
        assert!(at(2, 2) >= 1.0, "中心内距 ≥1，实际 {}", at(2, 2));
        // 块边缘像素自身距离 0。
        assert_eq!(at(1, 1), 0.0, "边缘像素自身距离 0");
        // 块外左邻 (0,2)：外侧，到边缘 (1,2) 距离 1 → −1。
        assert!(at(0, 2) < 0.0 && (at(0, 2).abs() - 1.0).abs() < 1e-3, "外侧左邻 ≈ −1");
    }

    #[test]
    fn atlas_packs_and_uv_within_unit() {
        let f = font_regular();
        let mut gids = Vec::new();
        for ch in ['A', 'B', '中', '永'] {
            if let Some(g) = f.glyph_index(ch) {
                gids.push(g);
            }
        }
        gids.push(f.glyph_index(' ').unwrap()); // 空格 → blank
        let p = SdfParams::new(32.0, 6.0, 3);
        let atlas = GlyphAtlas::bake(&f, &gids, &p, 512);
        assert!(atlas.width <= 512);
        assert!(atlas.height > 0);
        assert_eq!(atlas.data.len(), (atlas.width * atlas.height) as usize);
        assert!(atlas.blanks.contains(&f.glyph_index(' ').unwrap()), "空格应登记为 blank");
        for slot in &atlas.slots {
            let uv = atlas.uv(slot);
            assert!(uv[0] >= 0.0 && uv[2] <= 1.0 && uv[1] >= 0.0 && uv[3] <= 1.0, "UV 越界");
            assert!(uv[0] < uv[2] && uv[1] < uv[3]);
            // 槽内应同时含内侧与外侧样本（字形有内部与外部）。
            let mut any_in = false;
            let mut any_out = false;
            for yy in 0..slot.h {
                for xx in 0..slot.w {
                    let c = atlas.sample(slot, xx, yy);
                    if c > 160 {
                        any_in = true;
                    }
                    if c < 96 {
                        any_out = true;
                    }
                }
            }
            assert!(any_in, "槽应有内侧样本，gid={}", slot.gid);
            assert!(any_out, "槽应有外侧样本，gid={}", slot.gid);
        }
    }

    #[test]
    fn atlas_is_deterministic_and_shelf_monotone() {
        let f = font_regular();
        let gids: Vec<u16> = ['H', 'e', 'l', 'o', '中']
            .iter()
            .filter_map(|&c| f.glyph_index(c))
            .collect();
        let p = SdfParams::default();
        let a = GlyphAtlas::bake(&f, &gids, &p, 256);
        let b = GlyphAtlas::bake(&f, &gids, &p, 256);
        assert_eq!(a, b, "atlas 逐字节确定");
        // 行装箱：各行内 x 单调不减。
        let mut last_row_y = 0u32;
        let mut last_x = 0u32;
        for s in &a.slots {
            if s.y != last_row_y {
                last_row_y = s.y;
                last_x = 0;
            }
            assert!(s.x >= last_x, "行内 x 应单调不减");
            last_x = s.x + s.w;
        }
    }
}
