//! 位移贴图烘焙（W4 / G-10，选型报告 §5 定稿"在细分顶点上烘焙"）。
//!
//! # 为什么本模块在 `prism-rt` 而不在 `prism-geometry`
//!
//! 依赖方向是 `prism-rt → prism-geometry`（单向）。位移幅值必须走 **A1 的 CPU
//! 采样器**（[`crate::texsample`]：双线性 / mip / 各向异性已经是定了的那一套），
//! 而那套采样器活在 `prism-rt`。放到几何叶子会让 `prism-geometry` 反向依赖
//! `prism-rt`——所以分工是：
//!
//! * `prism_geometry::subdiv`：纯几何（Loop 细分 + 极限位置/法线/uv），零纹理依赖；
//! * 本模块：**只做一次逐顶点标量采样 + 沿极限法线推出去**。
//!
//! # 三条硬约束（判据 V3 / V4 就是它们的实测形式）
//!
//! 1. **幅值 0 ⇒ 顶点逐位不动**。这不是"误差够小"：`amplitude == 0.0` 时直接
//!    返回输入的**逐位拷贝**（短路在采样之前），连 `p + n·0.0` 的 `-0.0` 边界
//!    都不经过。理由：`p + (-0.0) = p` 只在 `p` 的该分量不是 `-0.0` 时逐位成立，
//!    而"顶点坐标恰好是 `-0.0`"在对称网格上是真实存在的（例如居中的平面）。
//! 2. **uv 缝在结构上不存在**。uv 是顶点域（`Mesh::uv` 的契约，见
//!    `integrator.rs` 的字段注释），相邻面共享同一顶点 ⇒ 同一 uv ⇒ 同一幅值
//!    ⇒ 位移后仍逐位相同。本模块**不引入角点域 uv**，也不按面遍历。
//! 3. **通道固定**：取纹素的**亮度** `0.2126·R + 0.7152·G + 0.0722·B`
//!    （Rec. 709 系数，固定求和顺序）。不用"哪个通道有值就用哪个"之类的自适应
//!    规则——那会让同一张贴图在不同场景里有两种含义，违反"不另起近似路径"。
//!    灰度高度图（R=G=B）下亮度 = 该灰度值，与单通道读法等价。
//!
//! # 采样点的 uv 不参与 mip：为什么用 `sample_point`
//!
//! 位移是**几何**，在 CPU 一次性烘焙；此时没有射线、没有足迹（`Footprint`），
//! 用带足迹的 [`crate::texsample::sample`] 会需要编造一个足迹（那就是第二个
//! 事实源）。[`crate::texsample::sample_point`] 做的是 **level 0 的双线性**，
//! 语义与 A1 采样器的"最细级、无足迹"档一致，且与 GPU 侧 `textureSampleLevel(0)`
//! 同口径。mip 对位移没有意义：位移只需要纹素级细节，降级会抹平几何。

use prism_geometry::bvh::Triangle;
use prism_math::Vec3;

use crate::texsample::{sample_point, TexTable};

/// 位移幅值的通道口径（**固定**，见模块文档第 3 条）。
///
/// 只保留一个变体是**故意的**：多一个通道选择就多一条"同一资产两种几何"的路径。
/// 若将来要支持单通道位移图，正确做法是加**编码**（`TextureRef` 的通道语义），
/// 而不是在这里加分支。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplaceChannel {
    /// 亮度（Rec. 709 权重，线性域纹素）。
    #[default]
    Luminance,
}

/// Rec. 709 亮度权重（线性域；与 sRGB 解码后的纹素直接相乘）。
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// 烘焙参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplaceParams {
    /// 幅值（世界单位；`0.0` ⇒ 恒等，逐位）。
    pub amplitude: f32,
    /// 幅值通道口径（当前只有亮度）。
    pub channel: DisplaceChannel,
}

impl Default for DisplaceParams {
    fn default() -> Self {
        Self { amplitude: 0.0, channel: DisplaceChannel::Luminance }
    }
}

impl DisplaceParams {
    /// 构造（幅值必须有限；负值合法——那是"向内位移"，与法线方向相反）。
    ///
    /// 非有限幅值 ⇒ 收缩成 `0.0`（恒等）并**在返回里体现**：不 panic，也不
    /// 产生 NaN 几何（C18）。调用方可用 [`DisplaceParams::is_identity`] 判。
    pub fn new(amplitude: f32) -> Self {
        Self {
            amplitude: if amplitude.is_finite() { amplitude } else { 0.0 },
            channel: DisplaceChannel::Luminance,
        }
    }

    /// 是否恒等（幅值恰为 `0.0` 或 `-0.0`）。
    pub fn is_identity(&self) -> bool {
        self.amplitude == 0.0
    }
}

/// 纹素 → 标量幅值（**固定**亮度组合，求和顺序固定为 `R → G → B`）。
#[inline]
pub fn scalar_of(texel: [f32; 4]) -> f32 {
    let mut s = texel[0] * LUMA_R;
    s = s + texel[1] * LUMA_G;
    s = s + texel[2] * LUMA_B;
    s
}

/// 单个顶点的幅值（无贴图 / 句柄非法 / uv 缺失 ⇒ `0.0`，而不是 NaN）。
///
/// 返回 `0.0` 时调用方**必须**走"顶点不动"的短路（见 [`displace_positions`]）。
pub fn amplitude_at(
    table: TexTable<'_>,
    handle: u32,
    uv: [f32; 2],
    params: DisplaceParams,
) -> f32 {
    if params.is_identity() {
        return 0.0;
    }
    if !(uv[0].is_finite() && uv[1].is_finite()) {
        return 0.0;
    }
    match sample_point(&table, handle, uv) {
        Some(t) => {
            let s = scalar_of(t);
            if s.is_finite() {
                params.amplitude * s
            } else {
                0.0
            }
        }
        None => 0.0,
    }
}

/// **位移烘焙入口**：`positions[i] += normals[i] · amplitude · L(uv[i])`。
///
/// # 逐位保证（判据 V3）
///
/// * `params.is_identity()` ⇒ 返回 `positions.to_vec()`（逐位拷贝，**不进入**
///   任何浮点运算）；
/// * 某个顶点的幅值恰为 `0.0`（贴图该处全黑 / 无贴图 / uv 非有限）⇒ 该顶点
///   **原样保留**（同样短路，不执行 `p + n·0`）。
///
/// 两条短路合起来让"displacement = 0 vs 未位移"在**构造上**逐位一致，而不是
/// "数值上很接近"。
///
/// # 长度契约
///
/// `normals.len()` 与 `uv.len()` 必须 ≥ `positions.len()`（多出的忽略）；
/// 不足 ⇒ 整批返回逐位拷贝（宁可"没位移"，也不产生半位移的错几何）。
pub fn displace_positions(
    positions: &[Vec3],
    normals: &[Vec3],
    uv: &[[f32; 2]],
    table: TexTable<'_>,
    handle: u32,
    params: DisplaceParams,
) -> Vec<Vec3> {
    let n = positions.len();
    if params.is_identity() || normals.len() < n || uv.len() < n {
        return positions.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = amplitude_at(table, handle, uv[i], params);
        if a == 0.0 {
            out.push(positions[i]); // 短路：不做 p + n·0（保 -0.0 位型）
            continue;
        }
        let d = normals[i] * a;
        out.push(positions[i] + d);
    }
    out
}

/// 索引网格 → 三角形 soup（送进 `CurveReq::TriSoup` / `Scene::add_bvh` 的形状）。
///
/// 越界索引跳过（不 panic）；绕序与 `indices` 一致（法线朝向由调用方决定：
/// 三个后端都在命中时按射线侧翻转法线，故绕序只影响符号）。
pub fn bake_triangles(positions: &[Vec3], indices: &[u32]) -> Vec<Triangle> {
    let mut out = Vec::with_capacity(indices.len() / 3);
    for c in indices.chunks_exact(3) {
        let (i0, i1, i2) = (c[0] as usize, c[1] as usize, c[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        out.push(Triangle { a: positions[i0], b: positions[i1], c: positions[i2] });
    }
    out
}

// ---------------------------------------------------------------------------
// 测试（native；不需要 GPU）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::build_rgba8;
    use crate::texsample::{TexTable, Texel};

    /// 8×8 灰度棋盘（线性域）纹理表。
    fn checker_table() -> (Vec<crate::texture::TextureRef>, Vec<Texel>) {
        let mut px = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8 {
            for x in 0..8 {
                let v = if (x + y) % 2 == 0 { 255u8 } else { 32u8 };
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let (r, t) = build_rgba8(8, 8, false, 0, &px);
        (vec![r], t)
    }

    /// V3：幅值 0 ⇒ 逐位一致（含 `-0.0` 顶点）。
    #[test]
    fn zero_amplitude_is_bitwise_identity() {
        let (refs, texels) = checker_table();
        let table = TexTable { refs: &refs, texels: &texels };
        // 故意放一个 `-0.0` 分量：`p + n·0.0` 会把它改成 `+0.0`，
        // 只有"短路不运算"才能保住位型。
        let pos = vec![
            Vec3::new(-0.0, 0.0, 1.0),
            Vec3::new(0.5, -2.0, 0.0),
            Vec3::new(-1.5, 3.25, -0.75),
        ];
        let nrm = vec![Vec3::new(0.0, 1.0, 0.0); 3];
        let uv = vec![[0.1, 0.2], [0.7, 0.3], [0.5, 0.5]];
        let out = displace_positions(&pos, &nrm, &uv, table, 1, DisplaceParams::new(0.0));
        assert_eq!(vbits(&out), vbits(&pos), "幅值 0 必须逐位一致（含 -0.0）");

        // 负零专项：确认短路真的没做算术。
        assert_eq!(out[0].x.to_bits(), (-0.0f32).to_bits());
    }

    /// V3（续）：非零幅值但该处幅值为 0（全黑贴图）⇒ 该顶点也不动。
    #[test]
    fn black_texel_leaves_vertex_untouched() {
        let (r0, texels) = build_rgba8(1, 1, false, 0, &[0, 0, 0, 255]);
        let refs = vec![r0];
        let table = TexTable { refs: &refs, texels: &texels };
        let pos = vec![Vec3::new(-0.0, 1.0, 2.0)];
        let nrm = vec![Vec3::new(0.0, 1.0, 0.0)];
        let uv = vec![[0.0, 0.0]];
        let out = displace_positions(&pos, &nrm, &uv, table, 1, DisplaceParams::new(0.5));
        assert_eq!(vbits(&out), vbits(&pos), "全黑贴图 ⇒ 零幅值 ⇒ 顶点不动");
    }

    /// 幅值确实沿法线推出去（方向与量值可判）。
    #[test]
    fn displacement_moves_along_limit_normal() {
        let (r0, texels) = build_rgba8(1, 1, false, 0, &[255, 255, 255, 255]);
        let refs = vec![r0];
        let table = TexTable { refs: &refs, texels: &texels };
        let pos = vec![Vec3::new(1.0, 0.0, 0.0)];
        let nrm = vec![Vec3::new(1.0, 0.0, 0.0)];
        let uv = vec![[0.0, 0.0]];
        let out = displace_positions(&pos, &nrm, &uv, table, 1, DisplaceParams::new(0.25));
        assert!((out[0].x - 1.25).abs() <= 1e-7 && out[0].y == 0.0 && out[0].z == 0.0);
    }

    /// 通道口径：亮度 = 0.2126R + 0.7152G + 0.0722B（固定顺序，逐位可复算）。
    #[test]
    fn luminance_weights_are_fixed() {
        let t = [0.3_f32, 0.6, 0.1, 1.0];
        let want = 0.3 * LUMA_R + 0.6 * LUMA_G + 0.1 * LUMA_B;
        assert_eq!(scalar_of(t).to_bits(), want.to_bits(), "亮度必须逐位等于固定式");
    }

    fn vbits(v: &[Vec3]) -> Vec<(u32, u32, u32)> {
        v.iter().map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())).collect()
    }
}
