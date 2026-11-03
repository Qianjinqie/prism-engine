//! `prism-postfx` — 后处理链（工作流 B4；后续 B4 行；架构 v2.4
//!  S2 卷积 / S11 扩散 / S5 重参数化的特解宿主 crate）。
//!
//! 语义权威：blur/bloom/vignette/grain/dissolve/glitch/
//! grade/CA/sharpen/plasma/Mandelbrot_bg/LUT 应用 + 三合一降噪 + UV warp、
//! 三合一降噪、（tile 无关性三规则
//! ——效果全部实现 [`prism_render::tile::TileEffect`]，halo 申报与核半径匹配）。
//!
//! # 组成
//!
//! - [`registry`] — 效果注册表：名称 → 工厂（与 prism-core 注册表风格一致）；
//! - [`effects`] — 12 效果（每效果一个文件，全部实现 `TileEffect`）+
//!   C1 合成域增补（[`effects::mask`] 遮罩/ROTO 族、[`effects::dof_full`]
//!   全帧景深）；
//! - [`denoise`] — 三合一降噪：à-trous 空域 + 确定性时域累积 + variance
//!   自适应 spp 挂钩，`DenoiseBackend` trait 为 P1 逆扩散预留挂点；
//! - [`warp`] — UV 重映射 `uv = f(uv)` 单原语（重映射场 = Fbuf 通道）。
//!
//! # 确定性纪律（格锚 +）
//!
//! - 全部效果是**求值纯函数**：同输入（含全部参数）逐位一致输出；无隐藏
//!   状态、无私开 RNG——随机性一律经 prism-noise 的 PCG 整数 hash 派生
//!  （唯一随机性入口）；
//! - **零超越函数**：只用 `+ - × ÷`、IEEE 精确的 `floor`/`round`/`sqrt`
//!   （精确舍入/开方非 libm 超越件，prism-anim easing.rs 同款先例）与整数
//!   运算——跨平台逐位一致是定理而非承诺（D-29）；
//! - f16 存储 / f32 计算（格锚）：经 `Fbuf16F::get/set` 进出，
//!   写回时 round-to-nearest-even 量化 f16；
//! - **halo 申报与核半径匹配**：可平移不变效果（blur/bloom/CA/sharpen/
//!   denoise/warp）在 tile 化与整幅求值下逐字节一致（`TileOrchestrator`
//!   复验，md5 贴出）；位置键控效果（grain/glitch/vignette/dissolve/
//!   plasma/Mandelbrot_bg）的空间图案锚在窗口局部坐标——见 `effects::mod`
//!   的"位置键控与 tile 协议"说明与缺口记录。

/// A5 逐像素自适应采样（选型报告 §6 定稿 C1）。
///
/// 放在本 crate 而不是 `prism-rt`：它要两个东西——`prism-rt` 的逐样本入口
/// （`integrator::trace_sample`）与 `allocate_spp`（`denoise.rs` 的归一化层）。
/// 依赖方向是 `postfx → render → rt`，故 A5 只能在 **postfx 及以上**落地；
/// 放进 rt 会形成 `rt → postfx → render → rt` 的环（Cargo 直接拒绝）。
pub mod adaptive;
pub mod denoise;
pub mod effects;
pub mod registry;
pub mod warp;
