//! `prism-raster` — 扫描线矢量光栅化。
//!
//! 矢量路径 → 像素覆盖率，供文本（prism-text）、矢量蒙版（PathMask）
//! 与生成艺术复用。核心设计（选定方案）：
//!
//! - **解析扫描线（AET 思想）+ 边上解析跨内覆盖率**：不是 2×2 采样近似，
//!   而是对覆盖区域做几何级精确积分——抗锯齿是解析的；
//! - **一次全高 y 扫描**：事件 = 段端点 ∪ 段间交点（全局只算一次）；
//!   **列折点不是事件**——区域与像素列的交互在列循环里解析求折点。活跃跨越序
//!   增量维护（近乎有序 → 插入排序 O(n)），于是每行的代价是 O(活跃跨越数)，
//!   而不是「子带数 × 活跃跨越数」。区域内部列贡献恒为 `q − p`（一次向量化
//!   区间加），只有两个边界列走解析原函数；
//! - **几何量走 f64**（顶点仍是调用方给的 f32）：f32 的唯一一次舍入落在写回
//!   覆盖率缓冲那一步，误差约 1 ulp——旧的全 f32 逐行带实现误差为
//!   `ulp(坐标)` 量级（1080p 下 5e-5）。多轮廓重叠（winding=2）天然正确
//!   （区域按集合测度计，不相交不重复计）；
//! - **输出 RGBA16F**（对齐 prism-io `Fbuf16F` 语义）：f16 存储、f32 计算，
//!   写入即 RNE 量化，逐字节确定。
//!
//! # 确定性契约
//! - 只用 `+` `-` `×` `÷` 与比较（铁律：禁 libm； 光栅算子面）；
//! - 浮点排序一律 `f32::total_cmp` / `f64::total_cmp`（全序，无 NaN 重排歧义）；
//! - 无隐藏状态、无随机源：同输入同几何必得同字节输出。

pub mod composite;
#[cfg(test)]
mod scanline_diff;
#[cfg(test)]
mod scanline_ref;
pub mod geom;
pub mod nodes;
pub mod path;
pub mod scanline;

pub use composite::{blend_pixel_over, draw_line, fill_path};
pub use geom::{Point, Transform};
pub use path::{Contour, PathBuilder};
pub use scanline::{rasterize, rasterize_into, FillRule};

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
