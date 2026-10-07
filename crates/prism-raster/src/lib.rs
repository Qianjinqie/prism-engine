//! `prism-raster` — 扫描线矢量光栅化。
//!
//! 矢量路径 → 像素覆盖率，供文本（prism-text）、矢量蒙版（PathMask）
//! 与生成艺术复用。核心设计（选定方案）：
//!
//! - **解析扫描线（AET 思想）+ 边上解析跨内覆盖率**：不是 2×2 采样近似，
//!   而是每行带（row band）内对覆盖区域做几何级精确积分——抗锯齿是解析的；
//! - **每行带算法**：轮廓边裁剪到行带 → 以"段端点 / 段间交点 / 段穿整像素
//!   边界"的 y 值切子带 → 子带内跨越序固定，winding 非零区间端点线性 →
//!   梯形精确积分进像素列。多轮廓重叠（winding=2）天然正确（区域按集合
//!   测度计，不相交不重复计）；
//! - **输出 RGBA16F**（对齐 prism-io `Fbuf16F` 语义）：f16 存储、f32 计算，
//!   写入即 RNE 量化，逐字节确定。
//!
//! # 确定性契约
//! - 只用 `+` `-` `×` `÷` 与比较（铁律：禁 libm； 光栅算子面）；
//! - 浮点排序一律 `f32::total_cmp`（全序，无 NaN 重排歧义）；
//! - 无隐藏状态、无随机源：同输入同几何必得同字节输出。

pub mod composite;
pub mod geom;
pub mod nodes;
pub mod path;
pub mod scanline;

pub use composite::{blend_pixel_over, draw_line, fill_path};
pub use geom::{Point, Transform};
pub use path::{Contour, PathBuilder};
pub use scanline::{rasterize, FillRule};

/// 每像素覆盖率缓冲（行主序，`data[x + y*width] ∈ [0,1]`）。
///
/// 覆盖率是几何精确量（f32），量化到 u8 的决定留给消费方
/// （prism-text 字形图量化、`fill_path` 直通 f16）。
#[derive(Debug, Clone, PartialEq)]
pub struct Coverage {
    /// 缓冲宽（像素）。
    pub width: u32,
    /// 缓冲高（像素）。
    pub height: u32,
    /// 覆盖率（行主序）。
    pub data: Vec<f32>,
}

impl Coverage {
    /// 构造全零覆盖率缓冲。
    pub fn new(width: u32, height: u32) -> Self {
        Coverage { width, height, data: vec![0.0; width as usize * height as usize] }
    }

    /// 读一个像素的覆盖率（越界返回 0——光栅语义下边界外恒空）。
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data[(y as usize) * self.width as usize + x as usize]
    }
}
