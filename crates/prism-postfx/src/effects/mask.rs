//! 遮罩 / ROTO 族（C1）：形状遮罩 / alpha 遮罩 / 深度遮罩 / 通道遮罩 +
//! 羽化 / 反相 / 布尔组合。
//!
//! 语义权威：《强大版路线图》§3.4（"遮罩族：`PathMask`（矢量蒙版，复用
//! prism-raster）、`LumaKey`/`ChromaKey`（亮度/色度抠像）、Roto（手绘路径 +
//! 羽化 + 参数随时间 = Expr 驱动的动态遮罩）"）；《算法白皮书》§18（"PathMask：
//! 矢量蒙版复用 §16 光栅（羽化 = 覆盖率梯度展宽）；Roto 动态 = 路径参数接属性
//! 通道"；"LumaKey/ChromaKey：在 YCbCr/HSV 特征域做阈值 + 软过渡（不是 RGB
//! 硬键），spill suppression 内建"）。
//!
//! # 设计
//!
//! 全部遮罩统一产出**单通道掩版**（`Vec<f32>` ∈ [0,1] 逐像素），再经
//! [`MaskOp`] 布尔 / 一元算子组合成最终遮版，最后 [`apply_mask`] 把遮版逐像素
//! 乘进 RGBA。遮版运算全部逐像素单遍、固定表达式序——层序/组合序决定论。
//! 作为 [`TileEffect`] 时 halo = 0（纯逐像素）。

use prism_io::fbuf::Fbuf16F;
use prism_raster::geom::Point;
use prism_raster::scanline::{rasterize, FillRule};
use prism_render::tile::TileEffect;

/// 遮版：单通道逐像素覆盖（[0,1]，行主序）。
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 逐像素覆盖。
    pub data: Vec<f32>,
}

impl Mask {
    /// 构造全零遮版。
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0.0; (width as usize) * (height as usize)] }
    }

    /// 读一个像素（越界返回 0）。
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// 写一个像素。
    ///
    /// # Panics
    /// 坐标越界时 panic。
    pub fn set(&mut self, x: u32, y: u32, v: f32) {
        assert!(x < self.width && y < self.height, "遮版坐标越界 ({x},{y})");
        self.data[(y as usize) * (self.width as usize) + (x as usize)] = v;
    }
}

/// 遮罩布尔 / 一元算子（白皮书 §18 布尔组合）。
///
/// 全部逐像素运算：`Union`/`Intersect`/`Subtract` 为集合算子（min/max/饱和减），
/// `Invert` 反相，`Add`/`Multiply` 软组合。两操作数尺寸必须一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// 并集（max）。
    Union,
    /// 交集（min）。
    Intersect,
    /// 差集 a − b（饱和减）。
    Subtract,
    /// 和 a + b（饱和加）。
    Add,
    /// 积 a·b。
    Multiply,
    /// 反相 1 − a（一元算子，忽略右操作数）。
    Invert,
}

/// 按算子组合两个遮版。
///
/// # Panics
/// 尺寸不一致时 panic。
pub fn combine(op: MaskOp, a: &Mask, b: &Mask) -> Mask {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "遮罩布尔组合要求尺寸一致"
    );
    let mut out = Mask::new(a.width, a.height);
    for i in 0..a.data.len() {
        let va = a.data[i];
        let vb = b.data[i];
        out.data[i] = match op {
            MaskOp::Union => va.max(vb),
            MaskOp::Intersect => va.min(vb),
            MaskOp::Subtract => (va - vb).max(0.0),
            MaskOp::Add => (va + vb).min(1.0),
            MaskOp::Multiply => va * vb,
            MaskOp::Invert => 1.0 - va,
        };
    }
    out
}

/// 羽化：对遮版做半径 `r` 的 box 展宽（覆盖率梯度展宽，白皮书 §18）。
///
/// 逐像素取 `(2r+1)²` 邻域均值，边缘钳制复制——与 blur 同约定（tile 无关性：
/// halo = r）。固定序归约（行主序 tap）。
pub fn feather(mask: &Mask, r: u32) -> Mask {
    if r == 0 {
        return mask.clone();
    }
    let mut out = Mask::new(mask.width, mask.height);
    let w = mask.width;
    let h = mask.height;
    let taps = ((2 * r + 1) as f32) * ((2 * r + 1) as f32);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for dy in 0..=(2 * r) {
                let sy = (y + dy).saturating_sub(r).min(h - 1);
                for dx in 0..=(2 * r) {
                    let sx = (x + dx).saturating_sub(r).min(w - 1);
                    acc += mask.get(sx, sy);
                }
            }
            out.set(x, y, acc / taps);
        }
    }
    out
}

/// 把遮版乘进 RGBA 缓冲（逐像素；halo = 0）。
pub fn apply_mask(fb: &Fbuf16F, mask: &Mask) -> Fbuf16F {
    let mut out = Fbuf16F::new(fb.width, fb.height);
    for y in 0..fb.height {
        for x in 0..fb.width {
            let c = fb.get(x, y);
            let m = if x < mask.width && y < mask.height { mask.get(x, y) } else { 0.0 };
            out.set(x, y, [c[0] * m, c[1] * m, c[2] * m, c[3] * m]);
        }
    }
    out
}

/// 形状遮罩（PathMask）：矢量多边形 → 覆盖遮版（复用 prism-raster 扫描线）。
///
/// 参数 `points` 为设备空间闭合折线顶点（y 向下）；`feather_radius` 羽化半径。
#[derive(Debug, Clone, PartialEq)]
pub struct PathMask {
    /// 设备空间顶点（闭合折线）。
    pub points: Vec<Point>,
    /// 羽化半径（像素；0 = 无羽化）。
    pub feather_radius: u32,
    /// 是否反相。
    pub invert: bool,
}

impl PathMask {
    /// 以顶点构造。
    pub fn new(points: Vec<Point>) -> Self {
        Self { points, feather_radius: 0, invert: false }
    }

    /// 光栅化为遮版。
    pub fn rasterize(&self, width: u32, height: u32) -> Mask {
        let contours: Vec<Vec<Point>> = vec![self.points.clone()];
        let cov = rasterize(&contours, FillRule::Winding, width, height);
        let mut mask = Mask { width, height, data: cov.data };
        if self.feather_radius > 0 {
            mask = feather(&mask, self.feather_radius);
        }
        if self.invert {
            for v in mask.data.iter_mut() {
                *v = 1.0 - *v;
            }
        }
        mask
    }
}

/// 亮度抠像（LumaKey）：按 Rec.709 亮度软阈值（白皮书 §18 特征域软过渡）。
///
/// 亮度 `L = 0.2126R + 0.7152G + 0.0722B`（BT.709 常量，固定）；
/// 阈值带 `[low, high]` 内线性软过渡（不是硬键）：`L < low → 0`、
/// `L > high → 1`、带内 `(L − low)/(high − low)`。`invert` 反相。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaKey {
    /// 阈值下限（遮版从 0 起）。
    pub low: f32,
    /// 阈值上限（遮版到 1）。
    pub high: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for LumaKey {
    fn default() -> Self {
        Self { low: 0.1, high: 0.5, invert: false }
    }
}

impl LumaKey {
    /// 从 RGBA 缓冲产出遮版。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let mut m = Mask::new(fb.width, fb.height);
        let span = (self.high - self.low).max(1.0e-6);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                let mut v = prism_math::clamp((l - self.low) / span, 0.0, 1.0);
                if self.invert {
                    v = 1.0 - v;
                }
                m.set(x, y, v);
            }
        }
        m
    }
}

/// 色度抠像（ChromaKey）：YCbCr 色度距离软键 + green spill suppression。
///
/// 键色 `key`（RGB）；逐像素转 YCbCr（BT.601 固定系数），按**色度距离**
/// 软阈值（`tolerance` 内 0、`softness` 带内线性过渡）——仅色度参与，
/// 亮度不敏感（白皮书 §18：特征域而非 RGB 硬键）。`spill` ∈ [0,1] 为
/// 溢出抑制强度：把像素中超出键绿色通道的绿色部分按强度削平。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaKey {
    /// 键色（RGB，通常绿幕/蓝幕）。
    pub key: [f32; 3],
    /// 色度距离软阈值下限（≤ 此距离完全键出）。
    pub tolerance: f32,
    /// 软过渡带宽（tolerance … tolerance+softness 线性过渡到 1）。
    pub softness: f32,
    /// 溢出抑制强度（0 = 关，1 = 全）。
    pub spill: f32,
}

impl Default for ChromaKey {
    fn default() -> Self {
        Self { key: [0.0, 1.0, 0.0], tolerance: 0.2, softness: 0.3, spill: 1.0 }
    }
}

/// RGB → BT.601 YCbCr（固定系数，禁 libm）。
fn rgb_to_ycbcr(c: [f32; 3]) -> (f32, f32, f32) {
    let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    let cb = (c[2] - y) * 0.564;
    let cr = (c[0] - y) * 0.713;
    (y, cb, cr)
}

impl ChromaKey {
    /// 从 RGBA 缓冲产出遮版（并另返回去溢出的颜色处理在 [`Self::process`]）。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let (_, kb, kr) = rgb_to_ycbcr(self.key);
        let mut m = Mask::new(fb.width, fb.height);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let (_, cb, cr) = rgb_to_ycbcr([c[0], c[1], c[2]]);
                let d = ((cb - kb) * (cb - kb) + (cr - kr) * (cr - kr)).sqrt();
                // 软过渡：d ≤ tolerance → 0（键出）；d ≥ tolerance+softness → 1（保留）。
                let hi = (self.tolerance + self.softness).max(self.tolerance + 1.0e-6);
                let v = prism_math::clamp((d - self.tolerance) / (hi - self.tolerance), 0.0, 1.0);
                m.set(x, y, v);
            }
        }
        m
    }

    /// 溢出抑制 + 遮版应用：键绿色通道过量的部分按强度削平，再乘遮版。
    ///
    /// spill suppression（白皮书 §18 内建）：对键色主通道（取键色最大分量
    /// 所在通道）做 `c = min(c, max(other two channels) + (1−spill)·excess)`
    /// ——绿色边缘过量绿被削到两邻通道水平。
    pub fn process(&self, fb: &Fbuf16F) -> Fbuf16F {
        let m = self.mask(fb);
        // 键色主通道（0=R 1=G 2=B）。
        let ch = if self.key[0] >= self.key[1] && self.key[0] >= self.key[2] {
            0
        } else if self.key[1] >= self.key[2] {
            1
        } else {
            2
        };
        let others: [usize; 2] = match ch {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let mut out = Fbuf16F::new(fb.width, fb.height);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let mut despi = c;
                let neighbor_max = c[others[0]].max(c[others[1]]);
                if c[ch] > neighbor_max {
                    let excess = c[ch] - neighbor_max;
                    despi[ch] = neighbor_max + excess * (1.0 - prism_math::clamp(self.spill, 0.0, 1.0));
                }
                let alpha = m.get(x, y);
                out.set(
                    x,
                    y,
                    [
                        despi[0] * alpha,
                        despi[1] * alpha,
                        despi[2] * alpha,
                        c[3] * alpha,
                    ],
                );
            }
        }
        out
    }
}

/// 通道遮罩（ChannelMask）：取 RGBA / 深度某通道作遮版（软过渡可选）。
///
/// `channel` ∈ {0,1,2,3}（R/G/B/A）；`low`/`high` 软阈值带同 LumaKey。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMask {
    /// 通道索引（0=R 1=G 2=B 3=A）。
    pub channel: u8,
    /// 阈值下限。
    pub low: f32,
    /// 阈值上限。
    pub high: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for ChannelMask {
    fn default() -> Self {
        Self { channel: 3, low: 0.5, high: 0.5, invert: false }
    }
}

impl ChannelMask {
    /// 从 RGBA 缓冲产出遮版。
    pub fn mask(&self, fb: &Fbuf16F) -> Mask {
        let ch = (self.channel.min(3)) as usize;
        let mut m = Mask::new(fb.width, fb.height);
        let span = (self.high - self.low).max(1.0e-6);
        for y in 0..fb.height {
            for x in 0..fb.width {
                let c = fb.get(x, y);
                let mut v = prism_math::clamp((c[ch] - self.low) / span, 0.0, 1.0);
                if self.invert {
                    v = 1.0 - v;
                }
                m.set(x, y, v);
            }
        }
        m
    }
}

/// 遮罩效果（TileEffect 适配）：`PathMask` 光栅 + 羽化/反相（halo = 0，
/// 光栅在窗口局部坐标——整幅求值即规范语义，同 postfx 位置键控约定）。
#[derive(Debug, Clone, PartialEq)]
pub struct PathMaskEffect {
    /// 路径遮罩。
    pub mask: PathMask,
}

impl TileEffect for PathMaskEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let m = self.mask.rasterize(window.width, window.height);
        apply_mask(window, &m)
    }
}

/// LumaKey 效果（TileEffect 适配，halo = 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaKeyEffect {
    /// 抠像参数。
    pub key: LumaKey,
}

impl TileEffect for LumaKeyEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let m = self.key.mask(window);
        apply_mask(window, &m)
    }
}

/// ChromaKey 效果（TileEffect 适配，halo = 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaKeyEffect {
    /// 抠像参数。
    pub key: ChromaKey,
}

impl TileEffect for ChromaKeyEffect {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        self.key.process(window)
    }
}

/// 深度遮罩（DepthMask）：按深度 AOV 软阈值产出遮版（深度图喂色彩/遮罩，
/// 路线图 §3.4"按距离调色"的地基）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthMask {
    /// 近阈值（AOV 小 = 近）。
    pub near: f32,
    /// 远阈值。
    pub far: f32,
    /// 是否反相。
    pub invert: bool,
}

impl Default for DepthMask {
    fn default() -> Self {
        Self { near: 0.0, far: 1.0, invert: false }
    }
}

impl DepthMask {
    /// 从深度 AOV（逐像素 [0,1]）产出遮版。
    pub fn mask(&self, aov: &[f32], width: u32, height: u32) -> Mask {
        let mut m = Mask::new(width, height);
        let span = (self.far - self.near).max(1.0e-6);
        for (i, &d) in aov.iter().enumerate() {
            let mut v = prism_math::clamp((d - self.near) / span, 0.0, 1.0);
            if self.invert {
                v = 1.0 - v;
            }
            m.data[i] = v;
        }
        m
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [f32; 4]) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, c);
            }
        }
        fb
    }

    fn mask_uniform(w: u32, h: u32, v: f32) -> Mask {
        let mut m = Mask::new(w, h);
        for i in 0..m.data.len() {
            m.data[i] = v;
        }
        m
    }

    // ---- 布尔运算断言 ----

    #[test]
    fn mask_boolean_union_intersect_subtract() {
        let a = mask_uniform(1, 1, 0.3);
        let b = mask_uniform(1, 1, 0.7);
        assert_eq!(combine(MaskOp::Union, &a, &b).get(0, 0), 0.7, "并 = max");
        assert_eq!(combine(MaskOp::Intersect, &a, &b).get(0, 0), 0.3, "交 = min");
        let sub = combine(MaskOp::Subtract, &a, &b).get(0, 0);
        assert_eq!(sub, 0.0, "差 a−b（饱和减，0.3−0.7<0 → 0）");
        let sub2 = combine(MaskOp::Subtract, &b, &a).get(0, 0);
        assert!((sub2 - 0.4).abs() < 1e-6, "差 b−a = 0.4");
    }

    #[test]
    fn mask_boolean_add_multiply_invert() {
        let a = mask_uniform(1, 1, 0.3);
        let b = mask_uniform(1, 1, 0.7);
        assert!((combine(MaskOp::Add, &a, &b).get(0, 0) - 1.0).abs() < 1e-6, "和饱和到 1");
        assert!((combine(MaskOp::Multiply, &a, &b).get(0, 0) - 0.21).abs() < 1e-6, "积");
        assert!((combine(MaskOp::Invert, &a, &a).get(0, 0) - 0.7).abs() < 1e-6, "反相 1−a");
    }

    #[test]
    fn mask_boolean_is_deterministic_and_pure() {
        let a = mask_uniform(4, 3, 0.4);
        let b = mask_uniform(4, 3, 0.6);
        let x = combine(MaskOp::Union, &a, &b);
        let y = combine(MaskOp::Union, &a, &b);
        assert_eq!(x.data, y.data, "同输入同输出（纯函数）");
    }

    // ---- 形状遮罩 ----

    #[test]
    fn path_mask_covers_polygon_interior() {
        // 10×10 帧内一个 4..8 的正方形 → 内部覆盖 1、外部 0。
        let pts = vec![
            Point::new(4.0, 4.0),
            Point::new(8.0, 4.0),
            Point::new(8.0, 8.0),
            Point::new(4.0, 8.0),
        ];
        let mask = PathMask::new(pts).rasterize(10, 10);
        // 中心在内部。
        assert!(mask.get(6, 6) > 0.9, "多边形内部应覆盖：{}", mask.get(6, 6));
        // 角落在外。
        assert!(mask.get(0, 0) < 0.1, "多边形外部应空：{}", mask.get(0, 0));
    }

    #[test]
    fn path_mask_invert_flips_coverage() {
        let pts = vec![
            Point::new(4.0, 4.0),
            Point::new(8.0, 4.0),
            Point::new(8.0, 8.0),
            Point::new(4.0, 8.0),
        ];
        let mut pm = PathMask::new(pts);
        pm.invert = true;
        let mask = pm.rasterize(10, 10);
        assert!(mask.get(6, 6) < 0.1, "反相后内部应空：{}", mask.get(6, 6));
        assert!(mask.get(0, 0) > 0.9, "反相后外部应满：{}", mask.get(0, 0));
    }

    #[test]
    fn feather_widens_edge_gradient() {
        // 硬边遮版（左半 1 右半 0）经羽化后边界带出现中间值。
        let mut m = Mask::new(9, 1);
        for x in 0..9 {
            m.set(x, 0, if x < 4 { 1.0 } else { 0.0 });
        }
        let f = feather(&m, 2);
        let edge = f.get(4, 0);
        assert!(edge > 0.0 && edge < 1.0, "羽化后边界应为中间值：{edge}");
        // 远处不受影响。
        assert_eq!(f.get(0, 0), 1.0, "远离边界处保持 1");
        assert_eq!(f.get(8, 0), 0.0, "远离边界处保持 0");
    }

    // ---- 亮度 / 色度抠像 ----

    #[test]
    fn luma_key_soft_threshold() {
        // 全灰 0.5，阈值带 [0.4, 0.6] → 覆盖 0.5。
        let fb = solid(1, 1, [0.5, 0.5, 0.5, 1.0]);
        let k = LumaKey { low: 0.4, high: 0.6, invert: false };
        let m = k.mask(&fb);
        assert!((m.get(0, 0) - 0.5).abs() < 1e-5, "软过渡中点：{}", m.get(0, 0));
        // 亮部 → 1，暗部 → 0。
        let bright = solid(1, 1, [1.0, 1.0, 1.0, 1.0]);
        assert!((k.mask(&bright).get(0, 0) - 1.0).abs() < 1e-5);
        let dark = solid(1, 1, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(k.mask(&dark).get(0, 0), 0.0);
    }

    #[test]
    fn chroma_key_removes_green_bgc_keeps_red() {
        let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.2, softness: 0.3, spill: 1.0 };
        // 纯绿 → 键出（遮版 0）。
        let green = solid(1, 1, [0.0, 1.0, 0.0, 1.0]);
        assert!(key.mask(&green).get(0, 0) < 0.05, "纯绿应被键出");
        // 纯红 → 保留（遮版 1）。
        let red = solid(1, 1, [1.0, 0.0, 0.0, 1.0]);
        assert!(key.mask(&red).get(0, 0) > 0.95, "纯红应保留");
    }

    #[test]
    fn chroma_key_spill_suppression_reduces_excess_green() {
        // 近绿但偏亮（R=0.4 G=0.8 B=0.4）→ spill 抑制后绿通道被削。
        let fb = solid(1, 1, [0.4, 0.8, 0.4, 1.0]);
        let key = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 1.0 };
        let out = key.process(&fb);
        let c = out.get(0, 0);
        // 溢出抑制：绿被削到邻通道水平（勿超 red 太多）。
        assert!(c[1] <= 0.4 + 1e-5, "溢出绿应被抑制到邻通道水平：{c:?}");
        // 无 spill 时绿保留。
        let key0 = ChromaKey { key: [0.0, 1.0, 0.0], tolerance: 0.0, softness: 1.0, spill: 0.0 };
        let out0 = key0.process(&fb);
        assert!(out0.get(0, 0)[1] > c[1], "spill=0 时绿应更大：{c:?} vs {:?}", out0.get(0, 0));
    }

    // ---- 通道 / 深度遮罩 ----

    #[test]
    fn channel_mask_reads_selected_channel() {
        let fb = solid(1, 1, [0.2, 0.4, 0.6, 0.8]);
        let cm = ChannelMask { channel: 2, low: 0.0, high: 1.0, invert: false };
        // f16 存储 → 比较容差 1e-3。
        assert!((cm.mask(&fb).get(0, 0) - 0.6).abs() < 1e-3, "通道 2（B）= 0.6");
        let ca = ChannelMask { channel: 3, low: 0.0, high: 1.0, invert: false };
        assert!((ca.mask(&fb).get(0, 0) - 0.8).abs() < 1e-3, "通道 3（A）= 0.8");
    }

    #[test]
    fn depth_mask_maps_distance_to_coverage() {
        let aov = vec![0.0, 0.5, 1.0];
        let dm = DepthMask { near: 0.0, far: 1.0, invert: false };
        let m = dm.mask(&aov, 3, 1);
        assert_eq!(m.get(0, 0), 0.0);
        assert!((m.get(1, 0) - 0.5).abs() < 1e-6);
        assert_eq!(m.get(2, 0), 1.0);
        // 反相。
        let dmi = DepthMask { near: 0.0, far: 1.0, invert: true };
        assert_eq!(dmi.mask(&aov, 3, 1).get(0, 0), 1.0);
    }

    // ---- 效果适配 ----

    #[test]
    fn path_mask_effect_has_zero_halo_and_applies() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(3.0, 3.0),
            Point::new(0.0, 3.0),
        ];
        let eff = PathMaskEffect { mask: PathMask::new(pts) };
        assert_eq!(eff.halo_radius(), 0, "遮罩效果 halo = 0");
        let fb = solid(3, 3, [1.0, 1.0, 1.0, 1.0]);
        let out = eff.eval_window(&fb);
        // 全在多边形内 → 全保留。
        assert!(out.get(1, 1)[0] > 0.9, "内部应保留：{:?}", out.get(1, 1));
    }

    #[test]
    fn apply_mask_multiplies_rgba() {
        let fb = solid(2, 1, [1.0, 0.5, 0.25, 1.0]);
        let m = mask_uniform(2, 1, 0.5);
        let out = apply_mask(&fb, &m);
        let c = out.get(0, 0);
        assert!((c[0] - 0.5).abs() < 1e-3 && (c[3] - 0.5).abs() < 1e-3, "遮版乘进 RGBA：{c:?}");
    }
}
