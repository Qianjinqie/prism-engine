//! `prism-rt` — 路径追踪 CpuRef。
//!
//! 语义权威：Sobol+Owen 采样器 / NEE+MIS / RR / firefly 钳制、
//! HG 介质、固定形状归约；（格锚表：hash
//! 走 PCG 整数件、归约走 prism-core 唯一形状函数）、采样派生式；
//! D-16（Mini Code）、D-22（数学结构第一性）。
//!
//! # 确定性铁律
//! 每个像素的每个样本是独立纯函数：坐标 = `Sobol(i, dim) ⊕ 打散掩码`，其中
//! 索引 i 与掩码均由 `hash(种子, 像素, 样本索引/维度)` 派生——无全局 RNG 状态，
//! 任意像素任意 pass 可独立重算（超越点）。
//!
//! # 模块
//! - [`sampler`]：Sobol 方向数 + Owen 打散（Joe–Kuo 64 维 + 超表兜底）。
//! - [`camera`]：透视 + 薄透镜 DOF + 快门（fov 用度——C17 行业豁免，构造期
//!   一次换算；其余长度 world 单位、时间秒）。
//! - [`media`]：HG 相函数 + delta tracking 透射率（Beer 定律）。
//! - [`bsdf`]：Disney BSDF（spec/diffuse/transmission 三瓣 + 多散射补偿）。
//! - [`light`]：四灯光族（方向 / 球 / 矩形 / 环境）+ NEE 采样 + MIS 权重。
//! - [`sss`]：random walk 次表面散射（σ̄ 自由程 + Fresnel 逃逸）。
//! - [`caustics`]：焦散 MNEE（k=1 镜面链 + k=2 球面折射链）+ PSR 兜底。
//! - [`integrator`]：backwards PT + NEE + MIS + RR + firefly 钳制；解析图元
//!   [`integrator::Paraboloid`]（z=k(x²+y²)，径向有界）与
//!   [`integrator::Plane`]（可有限切片）闭式求交（微积分短片 v2 E5：
//!   切平面叙事无法用轴对齐 `Aabb` 表达，故新增二次面/平面原语；与
//!   [`wavefront`] 双路径同源对拍 parity）。
//! - [`integrator::Mesh`]：三角几何（索引网格 / 三角形 soup 两型，内建
//!   [`prism_geometry::bvh::Bvh`] SAH 树）。`Scene::add_mesh` / `add_bvh` /
//!   `add_bvh_prebuilt` 为构造入口；命中共用 `ShapeRef` 细化，故 integrator 与
//!   [`wavefront`] 天然同语义（V2-h：100k 三角形构建 < 1s，双路径逐位 parity）。
//! - [`accum`]：算术累积缓冲 + prism-core 固定树形 spp 均值（禁跨 tile 归约）。
//!
//! # 范围（已接入）
//! Disney 全族 / 四灯光族 / random walk SSS / 焦散 MNEE+PSR / 解析图元
//! （抛物面 + 平面，含 wavefront parity）均已挂接 [`integrator`]；BSDF
//! 最小集（Lambert 漫反 + 完美镜面）作为等价 Disney 参数路径保留（既有
//! 测试与 examples 逐位兼容）。

pub mod accum;
pub mod bsdf;
pub mod caustics;
pub mod camera;
pub mod depth;
/// HDR 环境光照 / IBL（A2 / W2）：等距柱状立体角 2D CDF + 亮斑列表 + MIS。
pub mod env;
pub mod gi;
/// glTF 场景（保实例）→ [`integrator::Scene`] 的桥（W3 / A3）。
pub mod gltf_import;
pub mod hair_scatter;
pub mod instance;
pub mod integrator;
/// W1-A 定桩的**静态布局门禁**（仅 `cfg(test)` 存在；`cargo test -p prism-rt --lib` 会跑）。
#[cfg(test)]
pub mod layout;
pub mod light;
/// 材质预设计数真源（V2.2-B-3 / T05；10 项可渲染预设，见 [`material::MaterialPreset`]）。
pub mod material;
pub mod media;
pub mod sampler;
/// 光谱渲染切片（V2.2-B-1 / T08）：λ 轴 + 自研 CMF 分段有理 + 退化一致性。
pub mod spectral;
pub mod sss;
/// 位图纹理采样器（W1-B）：双线性 + mip 三级线性 + 有界各向异性（解析足迹）。
pub mod texsample;
pub mod texture;
pub mod wavefront;
