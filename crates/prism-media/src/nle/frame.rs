//! 帧缓冲（U14 NLE 剪辑）—— 时间线求值输出的最小图像载体。
//!
//! 与 [`crate::source::PixelImage`]（RGBA8 整数域）区分：本类型为**场景线性
//! f32 RGBA**（与 `prism-core` 的 `LayerBuf` 同域），便于转场线性混合与上层
//! 引擎（S2 卷积、调色）继续作用——这正是"嵌入值处于场景线性域、可被后续
//! 能力线性作用"的接入点（`prism-media/nodes.rs` 同一契约）。
//!
//! 行主序、`rgba[frame·4 + ch]`；尺寸 0 表示**空帧**（求值单位元，不静默编造）。

/// 场景线性 RGBA 帧（f32；行主序，每像素 4 分量）。
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// 宽（像素）。
    pub width: u32,
    /// 高（像素）。
    pub height: u32,
    /// 逐像素 RGBA（长度 = `width · height · 4`）。
    pub pixels: Vec<f32>,
}

impl Frame {
    /// 构造全透明（0,0,0,0）帧；尺寸 0 合法（空帧 = 单位元）。
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize).saturating_mul(height as usize).saturating_mul(4);
        Self { width, height, pixels: vec![0.0_f32; len] }
    }

    /// 以逐像素 RGBA 构造（长度须 = `w·h·4`；不符则返回空帧——不 panic）。
    pub fn from_rgba(width: u32, height: u32, pixels: Vec<f32>) -> Self {
        let expect = (width as usize).saturating_mul(height as usize).saturating_mul(4);
        if pixels.len() != expect {
            return Self { width: 0, height: 0, pixels: Vec::new() };
        }
        Self { width, height, pixels }
    }

    /// 由 `[u8;4]` RGBA8 常量色构造整帧（整数 → 场景线性除以 255）。
    pub fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let c = [
            rgba[0] as f32 / 255.0,
            rgba[1] as f32 / 255.0,
            rgba[2] as f32 / 255.0,
            rgba[3] as f32 / 255.0,
        ];
        let mut f = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                f.set(x, y, c);
            }
        }
        f
    }

    /// 是否为空帧（尺寸 0）。
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// 读一个像素 RGBA（越界返回全零）。
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        if x >= self.width || y >= self.height {
            return [0.0; 4];
        }
        let i = ((y as usize) * self.width as usize + x as usize) * 4;
        // 越界防护：i+4 在 length 内（由构造不变量保证），仍显式取 get 兜底。
        match self.pixels.get(i..i + 4) {
            Some(s) => [s[0], s[1], s[2], s[3]],
            None => [0.0; 4],
        }
    }

    /// 写一个像素 RGBA（越界忽略）。
    pub fn set(&mut self, x: u32, y: u32, rgba: [f32; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = ((y as usize) * self.width as usize + x as usize) * 4;
        if let Some(s) = self.pixels.get_mut(i..i + 4) {
            s.copy_from_slice(&rgba);
        }
    }

    /// 像素总数（`width · height`）。
    pub fn pixel_count(&self) -> usize {
        (self.width as usize).saturating_mul(self.height as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_set_get_and_oob() {
        let mut f = Frame::new(2, 2);
        f.set(1, 1, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(f.pixel(1, 1), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(f.pixel(9, 9), [0.0; 4], "越界返回全零");
        f.set(9, 9, [1.0; 4]); // 越界写忽略，不 panic
        assert_eq!(f.pixel_count(), 4);
    }

    #[test]
    fn from_rgba_length_mismatch_is_empty() {
        let f = Frame::from_rgba(2, 2, vec![0.0; 3]);
        assert!(f.is_empty(), "长度不符应退化为空帧");
    }
}
