//! 后处理效果族（清单 12 项，每效果一个文件，全部实现
//! [`prism_render::tile::TileEffect`]）。
//!
//! # halo 申报纪律（tile 无关性三规则）
//!
//! 效果分两类：
//!
//! - **邻域效果**（可平移不变）：输出像素只依赖输入窗口内 `p ± halo` 的
//!   邻域——blur / bloom / CA / sharpen / denoise / warp。halo 必须 ≥ 真实
//!   支撑半径（`TileOrchestrator` 按 sum(halo) 扩采，足额申报 ⇒ tile 化与
//!   整幅逐字节一致；本 crate 关卡 1 复验并贴 md5）；
//! - **位置键控效果**（空间图案锚定坐标）：grain / glitch / vignette /
//!   dissolve / plasma / Mandelbrot_bg——图案锚在**窗口局部坐标**上。
//!   `TileEffect::eval_window` 不携带全局原点（trait 只读对接，本 crate
//!   不得改动 prism-render），故此类效果的整幅求值即规范语义（post 链的
//!   CPU 参考路径），其 halo 只对"输入读"部分申报（如 glitch 的位移读带）。
//!   tile 协议要承载它们需要 `eval_window` 增全局原点上下文——缺口在册
//!   已如实记录，协议变更由协议方统一进行。

pub mod blur;
pub mod bloom;
pub mod ca;
pub mod color;
pub mod dissolve;
pub mod dof_full;
pub mod glitch;
pub mod grade;
pub mod grain;
pub mod lut;
pub mod mandelbrot;
pub mod mask;
pub mod plasma;
pub mod sharpen;
pub mod vignette;

pub use blur::Blur;
pub use bloom::Bloom;
pub use ca::ChromaticAberration;
pub use color::{ColorManage, ColorManageOcio, Oetf};
pub use dissolve::Dissolve;
pub use dof_full::FullDof;
pub use glitch::Glitch;
pub use grade::Grade;
pub use grain::Grain;
pub use lut::{ApplyLut, Lut3D, LutError};
pub use mandelbrot::MandelbrotBackground;
pub use mask::{
    apply_mask, combine, feather, ChannelMask, ChromaKey, ChromaKeyEffect, DepthMask, LumaKey,
    LumaKeyEffect, Mask, MaskOp, PathMask, PathMaskEffect,
};
pub use plasma::Plasma;
pub use sharpen::Sharpen;
pub use vignette::Vignette;

/// 三段式渐变调色板（plasma / Mandelbrot_bg 共用；纯算术，无超越函数）。
///
/// `u ∉ [0,1]` 先钳制；`u < 0.5` 走 A→B 段、否则 B→C 段，段内线性插值，
/// 表达式固定（先分量乘再相加）——跨平台逐位一致。
pub(crate) fn palette3(u: f32, a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = prism_math::clamp(u, 0.0, 1.0);
    let (p, q, t) = if u < 0.5 {
        (a, b, u * 2.0)
    } else {
        (b, c, u * 2.0 - 1.0)
    };
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = p[i] + (q[i] - p[i]) * t;
    }
    out
}

/// 双线性采样原语（CA 亚像素色散 / warp 重映射共用）——**位移分解形式**。
///
/// 确定性纪律（关卡 1/5）：触点 = 像素整数坐标 + `floor(dx)`/`floor(dy)`
/// 整数步进（i64 加法精确），插值权重 = 位移的小数部分 `(tx, ty)`——
/// **权重与像素的绝对坐标无关** ⇒ tile 化改变坐标幅值也不影响位模式。
/// （合成坐标 `x + dx` 的旧形式会被 f32 舍入耦合到坐标幅值上：大坐标下
/// 同一位移的 tx 舍入不同 → 权重差 1 ULP → tile 一致性被破坏，关卡 1
/// 实测检出——分解形式根除该耦合。）窗口边缘按钳制复制（与 box blur 同
/// 约定）；`tx = ty = 0`（整数位移）时权重精确 (1, 0, …, 0) ⇒ 格点采样
/// 逐位恒等。四项固定权重式、求和顺序固定——同输入逐位一致。
pub(crate) fn bilinear_sample_offset(
    buf: &prism_io::fbuf::Fbuf16F,
    x: u32,
    y: u32,
    dx: f32,
    dy: f32,
) -> [f32; 4] {
    let x0f = dx.floor();
    let y0f = dy.floor();
    let tx = dx - x0f;
    let ty = dy - y0f;
    let bx = x as i64 + x0f as i64;
    let by = y as i64 + y0f as i64;
    let x0 = bx.clamp(0, buf.width as i64 - 1) as u32;
    let x1 = (bx + 1).clamp(0, buf.width as i64 - 1) as u32;
    let y0 = by.clamp(0, buf.height as i64 - 1) as u32;
    let y1 = (by + 1).clamp(0, buf.height as i64 - 1) as u32;
    let c00 = buf.get(x0, y0);
    let c10 = buf.get(x1, y0);
    let c01 = buf.get(x0, y1);
    let c11 = buf.get(x1, y1);
    let (wx0, wx1) = (1.0 - tx, tx);
    let (wy0, wy1) = (1.0 - ty, ty);
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        out[i] = c00[i] * wx0 * wy0
            + c10[i] * wx1 * wy0
            + c01[i] * wx0 * wy1
            + c11[i] * wx1 * wy1;
    }
    out
}

pub mod film;
pub use film::{CurveKind, FilmCurve, FilmFormat, FilmGrain, Halation, HalationAndBloom, LayeredBloom};

pub mod npr;
pub use npr::{Halftone, HalftoneShape, Kuwahara, Pixelate, Sketch, Toon, Watercolor};
