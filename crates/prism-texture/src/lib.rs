//! `prism-texture` — 程序化纹理族（B14：纸张 / 噪波 / 渐变 / 图案 / Voronoi）。
//!
//! 语义权威：《强大版路线图》B14 行（程序化纹理族）；《算法白皮书》§17
//! （噪声族 = 一切程序化的原料，PCG 整数 hash + 唯一随机性入口）；架构
//! v2.4 §4.4（唯一随机性入口 = SampleIndex 派生）。
//!
//! # 组成
//!
//! - [`paper`] — 纸张纹理（纤维 fbm + 打浆斑块，乘算）；
//! - [`noise`] — 噪波（value / simplex / fbm 场 → 灰度或调色）；
//! - [`gradient`] — 渐变（线性 / 径向 / 角度多档渐变）；
//! - [`pattern`] — 图案（棋盘 / 条纹 / 网格 / 圆点，屏幕键控）；
//! - [`voronoi`] — Voronoi（种子点最近邻距离 / F2−F1 边缘 / 胞元色）。
//!
//! # 确定性纪律（D-29 + §17）
//!
//! 全部随机性经 prism-noise 的 PCG 整数 hash 派生（无私开 RNG、无全局状态）；
//! 零超越函数（只用 `+ - × ÷`、IEEE 精确的 `floor`/`sqrt`、整数运算）；
//! 同 `seed` 同参逐位一致（各纹理附 seed 决定论测试）。全部实现
//! [`prism_render::tile::TileEffect`]，图案锚在窗口局部坐标（位置键控）。

pub mod gradient;
pub mod noise;
pub mod paper;
pub mod pattern;
pub mod registry;
pub mod voronoi;

pub use gradient::{Gradient, GradientKind};
pub use noise::{NoiseTexture, NoiseTextureKind};
pub use paper::Paper;
pub use pattern::{Pattern, PatternKind};
pub use registry::{default_registry, TextureError, TextureRegistry};
pub use voronoi::{Voronoi, VoronoiMode};
