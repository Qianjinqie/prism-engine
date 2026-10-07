//! `prism-vector` — 矢量路径表达与光栅化对接。
//!
//! 本 crate 是"矢量形状"作为一等数据的最薄一层：它把 B9 的生成艺术输出
//! （折线/贝塞尔轮廓）、draw-on 逐步绘制、morph 形状插值统一表达为
//! [`Path`]，并复用 `prism-raster` 的扫描线光栅（**不重复造轮子**）
//! 对接为覆盖率缓冲。
//!
//! # 内容
//! - [`path`]：[`Path`] 多重子路径矢量表达（直线 + 二次贝塞尔，TTF 惯例）；
//! - [`path::flatten`]：自适应平铺（委托 `prism-raster`，误差 0.5px）；
//! - [`morph`]：`resample`（固定弧长参数）+ 两形状间渐变插值（顶点匹配）；
//! - [`humanize`]：手绘感扰动（确定性 hash 派生，非全局 RNG）；
//! - [`path::rasterize_path`]：路径 → 覆盖率缓冲（对接 prism-raster 扫描线）。
//!
//! # 确定性契约
//! 一切公开函数都是其参数的纯函数：无内部可变状态、无隐藏 RNG。扰动与
//! 顶点抖动一律走 `prism-noise` 的 PCG 整数 hash 派生（`[D]`），禁 libm。

pub mod humanize;
pub mod morph;
pub mod path;

pub use humanize::{humanize, humanize_with, Humanize};
pub use morph::{morph, morph_paths, resample};
pub use path::{flatten, rasterize_path, Path, SubPath, Vertex};
