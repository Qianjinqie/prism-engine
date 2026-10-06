//! `prism-genart` — 生成艺术与矢量绘制节点（路线图 B9；白皮书 §16/§17；
//! 架构 v2.4 §2.2 `genart { flowField/lsystem/… }` 行）。
//!
//! # 六算法生成器（B9 定案，以路线图/白皮书为准）
//! 依次为：流场（flow field）、差分生长（differential growth）、
//! 细胞自动机（cellular automata）、Voronoi 艺术（Voronoi art）、
//! 分形（fractal / Mandelbrot）、噪声调制（noise modulation）。每个生成器
//! 都是一个 `MathStructure` 特解 + `registry!` 静态汇入（D-27），输出统一
//! 为 [`prism_vector::Path`] 形状集合，可经 `prism-vector` 光栅化对接。
//!
//! # 绘制动画（B9）
//! - [`draw_on`]：路径**逐步绘制**（进度 s∈[0,1] 单调推进，接 `prism-anim`
//!   track 采样进度）；复用 `prism-raster` 的扫描线光栅（draw-on 复用 W5-B5
//!   光栅，非重复造）。
//! - morph / humanize 落 `prism-vector`（B9 的另一半接口面）。
//!
//! # A6 图表能力（并入 B9）
//! [`chart`]：简单图表节点（折线/柱状），复用生成艺术的路径面。
//!
//! # 确定性契约
//! 一切生成器都是其参数（含种子）的纯函数：随机基元一律走 `prism-noise`
//! PCG 整数 hash（`[D]`），**无全局 RNG**；只 ±×÷ 与比较，禁 libm（D-29）。
//! 同种子两遍逐位一致（见各模块测试）。

pub mod chart;
pub mod draw_on;
pub mod generators;
pub mod nodes;

pub use chart::{ChartKind, ChartSpec};
pub use draw_on::{draw_on, draw_on_path, progress_contours};
pub use generators::{
    CellularAutomata, DifferentialGrowth, FlowField, FractalMandelbrot, GeneratorOutput,
    NoiseModulation, VoronoiArt,
};

pub use nodes::{__prism_collect, node_for};