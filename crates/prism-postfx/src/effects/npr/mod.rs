//! NPR 风格化族（B14 清单：toon / 水彩 / Kuwahara / halftone / 像素化 / 素描）。
//!
//! 语义权威：《算法白皮书》§14（G-buffer 特征 + 像素算子统一框架；四风格
//! 算子）；《强大版路线图》B14 行。每风格一个文件、各实现
//! [`prism_render::tile::TileEffect`]，附带 golden-file 与 seed 决定论测试。
//!
//! # halo 申报
//!
//! 邻域风格算子（toon / 水彩 / Kuwahara / 素描）申报核半径并保持平移不变；
//! 逐像素 / 屏幕键控算子（halftone 网点屏、像素化、素描的梯度项）申报
//! 各自支撑半径。像素化为块均值（非平移不变时按块原点键控——与 B4 位置
//! 键控效果同约定，整幅求值为规范语义）。
//!
//! # 确定性纪律（D-29）
//!
//! 零超越函数：全部只用 `+ - × ÷`、IEEE 精确的 `floor`/`sqrt`/`abs`
//! 与整数运算；噪声一律经 prism-noise 的 PCG hash（域扭曲的流场）。

pub mod halftone;
pub mod kuwahara;
pub mod pixelate;
pub mod sketch;
pub mod toon;
pub mod watercolor;

pub use halftone::{Halftone, HalftoneShape};
pub use kuwahara::Kuwahara;
pub use pixelate::Pixelate;
pub use sketch::Sketch;
pub use toon::Toon;
pub use watercolor::Watercolor;

/// Rec.709 亮度（NPR 算子共用）。
pub(crate) const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// 逐像素亮度。
#[inline]
pub(crate) fn luma(c: [f32; 4]) -> f32 {
    c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2]
}

/// 中心差分梯度幅值（素描 / 水彩边缘检测共用）；窗口边缘钳制复制。
pub(crate) fn gradient_mag(window: &prism_io::fbuf::Fbuf16F, x: u32, y: u32) -> f32 {
    let w = window.width;
    let h = window.height;
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(w - 1);
    let yu = y.saturating_sub(1);
    let yd = (y + 1).min(h - 1);
    let gx = luma(window.get(xr, y)) - luma(window.get(xl, y));
    let gy = luma(window.get(x, yd)) - luma(window.get(x, yu));
    // 幅值用 sqrt（IEEE 精确开方，非超越件）——各向同性、逐位可复现。
    (gx * gx + gy * gy).sqrt()
}
