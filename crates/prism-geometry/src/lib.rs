//! `prism-geometry` — 几何域 crate（架构 v2.4 S12 实体表示 + S10 层次 + S1 标量场）。
//!
//! 模块一览：
//!
//! - [`primitive`] — 解析图元族闭式 SDF（球/盒/平面/圆柱/圆环/胶囊）与 [`primitive::Sdf`] trait。
//! - [`quartic`] — 四次方程精确求根（Ferrari 闭式，零迭代；圆环求交内核）。
//! - [`aabb_bvh`] — AABB-BVH：曲线图元族粗阶段加速（SAH + 回调精阶段，与 [`bvh`] 同确定性口径）。
//! - [`sdf`] — SDF 域算子（并/交/差/平滑变体）、displace（接 prism-noise）与 sphere tracing。
//! - [`csg`] — CSG 树：组合方式直接复用 [`sdf`] 域算子函数（D-16 避免双实现）。
//! - [`bvh`] — SAH-BVH：射线求交 + 最近点 + 广义 winding number（一树两用）。
//! - [`curve_tess`] — 曲线图元三角化（V2.2-A-2 方案乙）：圆柱/圆锥/圆环/旋转曲面 → 网格；
//!   W5 起含毛股（[`curve_tess::CurveReq::Strand`]：折线 + 变半径圆管 + 平行传输标架）。
//! - [`subdiv`] — Loop 细分（三角笼，W4 定稿 C2）：均匀 1→4 + **极限位置直接求值**
//!   （1-ring stencil，不迭代）+ 边界/角点规则 + 顶点域 uv。产物是裸数组，
//!   由 `prism-rt` 侧转 `Mesh`（依赖方向：`prism-rt → prism-geometry` 单向）。
//! - [`mesh_sdf`] — mesh→SDF 烘焙：brick 8³ 稀疏化 + winding number 符号。
//! - [`det_math`] — D-29 确定性数学件（det_atan2/det_asin/det_cos_sin；禁 libm）。
//! - [`gltf`] — glTF 精简导入（手写 JSON 子集解析，禁 serde）。
//! - [`obj`] — OBJ/MTL 精简导入。
//! - [`lod`] — LOD 链（从简）：步进 eps 随距离放大 + displace octaves 随距离衰减。
//!
//! # 确定性契约
//!
//! 全部构建/烘焙/求值是输入的纯函数：BVH 同输入同树（SAH 并列按图元索引 tie-break），
//! 烘焙同输入同字节（逐体素独立计算，无跨体素浮点累积）。只用 `+ - × ÷`、
//! `min/max/clamp/floor/abs/sqrt` 等 IEEE 精确或确定性运算；超越函数（atan2/
//! asin/sin/cos 等）一律走 [`det_math`] 确定性件（纯 IEEE 运算，同输入逐位一致，
//! 禁 libm）；禁 fma 内建（附注）。

pub mod aabb_bvh;
pub mod bvh;
pub mod csg;
pub mod curve_tess;
pub mod det_math;
pub mod gltf;
pub mod lod;
pub mod mesh_sdf;
pub mod obj;
pub mod primitive;
pub mod quartic;
pub mod sdf;
pub mod subdiv;

use prism_math::Vec3;

/// 分量绝对值。
#[inline]
pub(crate) fn vabs(v: Vec3) -> Vec3 {
    Vec3::new(v.x.abs(), v.y.abs(), v.z.abs())
}

/// 分量取大。
#[inline]
pub(crate) fn vmax(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

/// 分量取小。
#[inline]
pub(crate) fn vmin(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

/// 三分量最大值。
#[inline]
pub(crate) fn vmaxc(v: Vec3) -> f32 {
    v.x.max(v.y).max(v.z)
}

/// XZ 平面子向量长度（圆柱/圆环共用）。
#[inline]
pub(crate) fn xz_len(v: Vec3) -> f32 {
    (v.x * v.x + v.z * v.z).sqrt()
}
