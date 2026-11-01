//! 几何实例记录（W1-A 定桩；**BVH / 求交 / 变换合成都不是本段的事**）。
//!
//! # 本模块只有布局
//!
//! 路线图 W3（A3 几何实例化，选型 C1 两级 BVH：TLAS 复用
//! `prism_geometry::aabb_bvh::AabbBvh`，BLAS 复用 `prism_geometry::bvh::Bvh`）
//! 是 W3 的事。W0-A 只落地 [`Instance`] 的**字段与槽宽**，好让 W1-B 与 W3
//! 在同一份 `Scene` 布局上并行（路线图 §4.2）。
//!
//! 因此本模块**没有** BVH 构建、没有射线变换、没有法线逆转置——一个都没有。
//! [`Instance::identity`] 只是"把 0/1 填进槽位"的布局辅助，不是变换实现。
//!
//! # 112 B / 条（选型报告 §4 定稿）
//!
//! `3×4 仿射（48 B）+ 3×4 逆（48 B）+ geom_id（4）+ mat_id（4）+ flags（4）+ pad（4）`
//!
//! 逆矩阵**必须**与仿射一起在 CPU 算好并随记录走：三个后端读同一份字节，
//! **禁止各后端各自求逆**（求逆的浮点路径不同 ⇒ 逐位一致立刻破）。
//!
//! # 为什么逆也要存（而不是运行期代公式）
//!
//! 非均匀缩放下 `inverse-transpose` 的法线与 `inverse` 的射线方向不是同一
//! 个矩阵；且 `T_MIN`/`SURFACE_EPS` 是**世界空间**量，进局部空间要按缩放
//! 换算。这些都吃 `inverse` 的精确位型——存盘比再算一遍更省事，也唯一能
//! 保证三后端一致。
//!
//! # 类型纪律
//!
//! 全 `f32` / `u32` / `[f32; N]` / `[u32; N]`，**无** `vec3<f32>`（见
//! [`crate::layout`] 的静态门禁）。
//!
//! # W3 落地：本模块现在有数学了
//!
//! 变换合成、求逆、法线逆转置、世界 AABB —— 全部在本模块，且**只用
//! `+ − × ÷`**（无 libm、无 fma、无超越函数），故同输入跨平台逐位一致。
//! [`Instance::from_affine`] 是**唯一**构造入口：逆矩阵在这里算一次，
//! 三后端读同一份字节（选型报告 §4「禁止各后端各自求逆」）。

use prism_geometry::bvh::Aabb;
use prism_math::Vec3;

/// 单条实例记录的字节数（选型报告 §4 定稿；路线图 W3 判据「112 B/实例」）。
pub const INSTANCE_BYTES: usize = 112;

/// 单条实例记录的 f32 当量槽数（= 28，即 112 / 4）。
pub const INSTANCE_WORDS: u32 = 28;

/// 仿射子矩阵的 f32 个数（3 行 × 4 列）。
pub const INSTANCE_AFFINE_WORDS: usize = 12;

/// 逆矩阵的 f32 个数（3 行 × 4 列）。
pub const INSTANCE_INVERSE_WORDS: usize = 12;

/// `flags` 位：变换为恒等（`affine == inverse == [I|0]`）。
///
/// 恒等实例是 W3 判据「展平版 vs 实例化版**逐位一致**」的可用形态：
/// `x·1.0 + 0.0` 在 IEEE 下精确（含 `-0.0`：`0.0·x = ±0.0`，
/// `0.0 + (-0.0) = 0.0`），故 `M·p == p` 逐位。
pub const INSTANCE_FLAG_IDENTITY: u32 = 1;

/// `flags` 位：仿射的 3×3 线性块行列式为负（**镜像**实例）。
///
/// 为什么需要它：绕序法线的世界方向是 `det(A)·A⁻ᵀ·n_local`，不是
/// `A⁻ᵀ·n_local`。`det < 0` 时两者反向 —— 不做这一步，镜像实例的
/// 法线朝内，表现为"整片背光/漏光"。det 在
/// [`Instance::from_affine`] 里算一次并记进标志位（逆矩阵本来就要 det），
/// 运行期不再重算。
pub const INSTANCE_FLAG_MIRRORED: u32 = 1 << 1;

/// `mat_id` 的哨兵值：**材质取自几何自身**（`meshes[geom_id].1`）。
///
/// `mat_id` 的语义由 W3 定：非哨兵时是"材质覆盖"，索引进
/// [`crate::integrator::Scene::meshes`]（材质与几何同表存放，故取值范围与
/// `geom_id` 相同）。选这个哨兵而不是 `0`，是因为 `0` 是合法下标。
pub const INSTANCE_MAT_FROM_GEOM: u32 = u32::MAX;

/// 几何实例记录（`Scene::instances` 的元素；**只描述，不含几何**）。
///
/// `affine` / `inverse` 均为 **3×4 行主序**（`[m00,m01,m02,tx, m10,…]`），
/// 与 `TRIANGLE_SLOTS` 的"每 `vec3` 后补 1 pad"不同：这里刻意压成稠密
/// 48 B，因为实例记录只被 CPU 与 shader 逐字读，**不做** `vec4` 化补齐
/// （补成 4×4 会变 128 B/条，违背 §4 的 112 B 定稿）。对齐是 4 B，各后端
/// 的 `array<Instance>` 声明须同为 4 B 对齐方可逐字节互认——这条由
/// `scene_abi` 的 V3 剖面在 W3 落地时一并钉住。
/// # `repr(C)` 是**承重**的，不是装饰
///
/// 与 [`crate::texture::TextureRef`] 同理：非 `repr(C)` 结构体允许编译器
/// 重排字段，`size_of == 112` 会全绿而 `affine`/`inverse` 的先后可能已经
/// 换了位。W3 的 TLAS 按 `offset 48` 去读逆矩阵，读到仿射就是"射线往错误
/// 方向走"——这类 bug 只会在画面上表现为几何整体错位，极难二分。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    /// 3×4 仿射（行主序；`[..3]` 为 3×3 线性块，`[3]`/`[7]`/`[11]` 为平移）。
    pub affine: [f32; INSTANCE_AFFINE_WORDS],
    /// 3×4 逆（行主序；CPU 一次算好，三后端读同一份）。
    pub inverse: [f32; INSTANCE_INVERSE_WORDS],
    /// 几何（BLAS）句柄：索引进场景的共享几何表。
    pub geom_id: u32,
    /// 材质句柄。
    pub mat_id: u32,
    /// 位标志（[`INSTANCE_FLAG_IDENTITY`]）。
    pub flags: u32,
    /// 保留字（恒 0）：只为把步长钉在 [`INSTANCE_BYTES`]。
    pub pad: u32,
}

impl Instance {
    /// 恒等实例（`M = I`, `t = 0`，`inverse` 同值）。
    ///
    /// **只填槽位，不做任何变换数学**——变换合成是 W3 的实现内容，本段不写。
    pub const fn identity(geom_id: u32, mat_id: u32) -> Self {
        Instance {
            affine: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            inverse: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            geom_id,
            mat_id,
            flags: INSTANCE_FLAG_IDENTITY,
            pad: 0,
        }
    }

    /// 是否恒等变换（读 `flags`，不重算矩阵）。
    pub const fn is_identity(&self) -> bool {
        self.flags & INSTANCE_FLAG_IDENTITY != 0
    }

    /// 是否镜像（`det(A) < 0`；法线要额外反号，见 [`INSTANCE_FLAG_MIRRORED`]）。
    pub const fn is_mirrored(&self) -> bool {
        self.flags & INSTANCE_FLAG_MIRRORED != 0
    }

    /// 材质覆盖：`Some(meshes 下标)` 或 `None`（材质取自共享几何自身）。
    pub const fn material_override(&self) -> Option<u32> {
        if self.mat_id == INSTANCE_MAT_FROM_GEOM {
            None
        } else {
            Some(self.mat_id)
        }
    }
}

// ---------------------------------------------------------------------------
// 变换数学（W3）—— 只用 + − × ÷，无 libm / 无 fma / 无超越函数
// ---------------------------------------------------------------------------

/// 恒等仿射（3×4 行主序）。
pub const IDENTITY_AFFINE: [f32; INSTANCE_AFFINE_WORDS] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0,
];

impl Instance {
    /// **唯一**构造入口：`affine`（3×4 行主序）→ 完整实例记录（含逆）。
    ///
    /// # 逆矩阵为什么在 CPU 算一次
    ///
    /// 三个后端读同一份字节。CPU / SPIR-V / WGSL 各自求逆的浮点路径不同，
    /// 同一条射线会在三个后端上给出三个不同的 `t`（以及三个不同的法线），
    /// 逐位一致立刻破。存盘 48 B 换"三后端只有一个事实源"。
    ///
    /// # 退化
    ///
    /// `det` 为 0 / 非有限 / 逆元非有限 ⇒ `None`（调用方决定报错还是跳过）。
    /// 退化实例的"逆"不存在，任何兜底值都会让射线走到 NaN 分支。
    ///
    /// # 恒等快路径
    ///
    /// `affine` 数值上等于恒等 ⇒ `flags |= IDENTITY`，逆直接抄恒等（不求）。
    /// 这一步让 identity 实例的 `M⁻¹·p` 与 `p` **逐位**相同（不走乘法），
    /// 是「identity 实例 vs 展平版逐位一致」判据的构造性保证。
    pub fn from_affine(
        affine: [f32; INSTANCE_AFFINE_WORDS],
        geom_id: u32,
        mat_id: u32,
    ) -> Option<Instance> {
        if !affine.iter().all(|v| v.is_finite()) {
            return None;
        }
        // 恒等：`==` 而非位比较 —— `-0.0 == 0.0` 在数值上是恒等，应当走快路径。
        if affine == IDENTITY_AFFINE {
            return Some(Instance {
                affine,
                inverse: IDENTITY_AFFINE,
                geom_id,
                mat_id,
                flags: INSTANCE_FLAG_IDENTITY,
                pad: 0,
            });
        }
        // 3×3 余子式（行主序：a[r*4+c]）。
        let (a00, a01, a02, tx) = (affine[0], affine[1], affine[2], affine[3]);
        let (a10, a11, a12, ty) = (affine[4], affine[5], affine[6], affine[7]);
        let (a20, a21, a22, tz) = (affine[8], affine[9], affine[10], affine[11]);
        let c00 = a11 * a22 - a12 * a21;
        let c01 = a12 * a20 - a10 * a22;
        let c02 = a10 * a21 - a11 * a20;
        let det = a00 * c00 + a01 * c01 + a02 * c02;
        let inv_det = 1.0 / det;
        if !inv_det.is_finite() {
            return None; // det == 0 或 NaN（Inf 亦然）：逆不存在
        }
        // 余子式矩阵的转置 / det = 逆（inv[i][j] = C_ji / det）。
        let c10 = a02 * a21 - a01 * a22;
        let c11 = a00 * a22 - a02 * a20;
        let c12 = a01 * a20 - a00 * a21;
        let c20 = a01 * a12 - a02 * a11;
        let c21 = a02 * a10 - a00 * a12;
        let c22 = a00 * a11 - a01 * a10;
        let m = |c: f32| c * inv_det;
        let (i00, i01, i02) = (m(c00), m(c10), m(c20));
        let (i10, i11, i12) = (m(c01), m(c11), m(c21));
        let (i20, i21, i22) = (m(c02), m(c12), m(c22));
        // 逆平移 = −A⁻¹·t。
        let itx = -(i00 * tx + i01 * ty + i02 * tz);
        let ity = -(i10 * tx + i11 * ty + i12 * tz);
        let itz = -(i20 * tx + i21 * ty + i22 * tz);
        let inverse = [
            i00, i01, i02, itx, //
            i10, i11, i12, ity, //
            i20, i21, i22, itz,
        ];
        if !inverse.iter().all(|v| v.is_finite()) {
            return None; // 极小的 det 让逆元溢出
        }
        let mut flags = 0u32;
        if det < 0.0 {
            flags |= INSTANCE_FLAG_MIRRORED;
        }
        Some(Instance { affine, inverse, geom_id, mat_id, flags, pad: 0 })
    }

    /// glTF / USD 的**列主序 4×4** → 实例（丢最后一行 `0,0,0,1`）。
    ///
    /// 列主序 `m[c*4+r]` 与行主序 3×4 的换算：`affine[r*4+c] = m[c*4+r]`，
    /// 平移列 `affine[r*4+3] = m[12+r]`。
    pub fn from_col_major_4x4(
        m: &[f32; 16],
        geom_id: u32,
        mat_id: u32,
    ) -> Option<Instance> {
        let affine = [
            m[0], m[4], m[8], m[12], //
            m[1], m[5], m[9], m[13], //
            m[2], m[6], m[10], m[14],
        ];
        Instance::from_affine(affine, geom_id, mat_id)
    }

    /// 仿射线性块的行列式（`from_affine` 已算过，此处只为校验/测试）。
    pub fn affine_det(&self) -> f32 {
        let a = &self.affine;
        a[0] * (a[5] * a[10] - a[6] * a[9]) + a[1] * (a[6] * a[8] - a[4] * a[10])
            + a[2] * (a[4] * a[9] - a[5] * a[8])
    }

    /// 局部点 → 世界点（`A·p + t`）。
    pub fn apply_point(&self, p: Vec3) -> Vec3 {
        let a = &self.affine;
        Vec3::new(
            a[0] * p.x + a[1] * p.y + a[2] * p.z + a[3],
            a[4] * p.x + a[5] * p.y + a[6] * p.z + a[7],
            a[8] * p.x + a[9] * p.y + a[10] * p.z + a[11],
        )
    }

    /// 局部方向 → 世界方向（只乘线性块，**不含平移**）。
    pub fn apply_dir(&self, d: Vec3) -> Vec3 {
        let a = &self.affine;
        Vec3::new(
            a[0] * d.x + a[1] * d.y + a[2] * d.z,
            a[4] * d.x + a[5] * d.y + a[6] * d.z,
            a[8] * d.x + a[9] * d.y + a[10] * d.z,
        )
    }

    /// 世界点 → 局部点（`A⁻¹·p + t⁻¹`）。
    pub fn inv_point(&self, p: Vec3) -> Vec3 {
        let a = &self.inverse;
        Vec3::new(
            a[0] * p.x + a[1] * p.y + a[2] * p.z + a[3],
            a[4] * p.x + a[5] * p.y + a[6] * p.z + a[7],
            a[8] * p.x + a[9] * p.y + a[10] * p.z + a[11],
        )
    }

    /// 世界方向 → 局部方向（`A⁻¹·d`）。
    pub fn inv_dir(&self, d: Vec3) -> Vec3 {
        let a = &self.inverse;
        Vec3::new(
            a[0] * d.x + a[1] * d.y + a[2] * d.z,
            a[4] * d.x + a[5] * d.y + a[6] * d.z,
            a[8] * d.x + a[9] * d.y + a[10] * d.z,
        )
    }

    /// 世界射线 → **局部射线**（`t` 参数不变）。
    ///
    /// # `t` 为什么不换算（选型报告列为"最难的三点"之二的解法）
    ///
    /// 世界射线 `(o, d)` 经 `M⁻¹` 到局部：`o_l = M⁻¹o`、`d_l = A⁻¹d`。
    /// 因为同一个仿射同时作用于 `o` 与 `d`，`o_l + t·d_l ≡ M⁻¹(o + t·d)`
    /// 对**同一个 `t`** 成立 —— 于是 `T_MIN = 1e-4` 与 `t_max` 都是"世界
    /// `t`"，可以直接当局部查询的界用，**不需要按缩放换算**（只有把 `d_l`
    /// 归一化之后才需要 `t_l = t·|A⁻¹d|`）。
    ///
    /// 即：W3 是把这个难点**消掉**（不归一化），而不是算对。
    /// 代价是 `|d_l| = |A⁻¹d|` 随实例缩放变化，Möller–Trumbore 的
    /// `|det| < 1e-12` 退化判据随之平移（实例缩到 ~1e-9 以下才会误判不命中，
    /// 见 `w3_extreme_scale_still_hits`）。
    ///
    /// identity ⇒ 原样返回（逐位相同，且不做任何乘法）。这是判据
    /// 「identity 实例 vs 展平版逐位一致」成立的那一行代码。
    pub fn local_ray(&self, ro: Vec3, rd: Vec3) -> (Vec3, Vec3) {
        if self.is_identity() {
            (ro, rd)
        } else {
            (self.inv_point(ro), self.inv_dir(rd))
        }
    }

    /// 局部法线 → 世界法线方向（**逆转置**，含镜像反号），未归一化。
    ///
    /// # 为什么是逆转置而不是变换矩阵
    ///
    /// 法线是**协变**量（与切向对偶），只有 `A⁻ᵀ` 能让它与变换后的表面
    /// 保持垂直。直接乘 `A` 在非均匀缩放下会让法线偏斜：`A = diag(3,1,1)`
    /// 时一个 45° 法线会被拉到 `(3,1,0)`（正确值是 `(1/3,1,0)` 方向）——
    /// 表现是"斜面上的高光位置错、漫反射明暗交界错"。
    ///
    /// 镜像（`det < 0`）时再反号：绕序法线的世界方向是 `det(A)·A⁻ᵀ·n`。
    ///
    /// 归一化留给调用方（`triangle_unit_normal` 用的是 `n·(1/√(n·n))`
    /// 这一条口径，两处必须是同一条公式，否则 identity 判据会飘）。
    pub fn inv_transpose_normal(&self, n: Vec3) -> Vec3 {
        let a = &self.inverse;
        let out = Vec3::new(
            a[0] * n.x + a[4] * n.y + a[8] * n.z,
            a[1] * n.x + a[5] * n.y + a[9] * n.z,
            a[2] * n.x + a[6] * n.y + a[10] * n.z,
        );
        if self.is_mirrored() {
            -out
        } else {
            out
        }
    }

    /// 局部 AABB → 世界 AABB（8 角点变换后取界）。
    ///
    /// 非均匀缩放 + 旋转下，只变换 `min`/`max` 两个角点是**错的**（角点
    /// 互换），必须 8 角全算。空盒（`min > max`）返回空盒。
    pub fn world_aabb(&self, min: Vec3, max: Vec3) -> Aabb {
        let mut out = Aabb::empty();
        if !(min.x <= max.x && min.y <= max.y && min.z <= max.z) {
            return out; // 空/反向盒：不引入任何点
        }
        for &x in &[min.x, max.x] {
            for &y in &[min.y, max.y] {
                for &z in &[min.z, max.z] {
                    out.grow_point(self.apply_point(Vec3::new(x, y, z)));
                }
            }
        }
        out
    }
}

impl Instance {
    /// 数值自检：`A · A⁻¹ ≈ I`（构造期单点保证的复核口，**不进渲染路径**）。
    ///
    /// 返回 3×3 乘积与恒等的最大绝对偏差。identity 实例应为 0。
    pub fn inverse_residual(&self) -> f32 {
        let a = &self.affine;
        let b = &self.inverse;
        let mut worst = 0.0f32;
        for r in 0..3 {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for k in 0..3 {
                    acc += a[r * 4 + k] * b[k * 4 + c];
                }
                let want = if r == c { 1.0 } else { 0.0 };
                let d = (acc - want).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
        worst
    }
}

// ---------------------------------------------------------------------------
// 编译期断言（W3 的「112 B/实例」在编译期就成立，不靠事后测）
// ---------------------------------------------------------------------------

/// 槽宽：112 B（选型报告 §4 定稿；路线图 W3 判据）。
const _: () = assert!(core::mem::size_of::<Instance>() == INSTANCE_BYTES);
/// 对齐：4 B（全 f32/u32 ⇒ 跨 target 恒定；换成 `Vec3` 会变 12 B 对齐语义）。
const _: () = assert!(core::mem::align_of::<Instance>() == 4);
/// 字数自洽：112 / 4 = 28。
const _: () = assert!(INSTANCE_BYTES == INSTANCE_WORDS as usize * 4);
/// 两块 3×4 各 48 B，占满前 96 B。
const _: () = assert!(INSTANCE_AFFINE_WORDS + INSTANCE_INVERSE_WORDS == 24);
/// 逐字段偏移 —— 实例槽错一格，TLAS 叶子会指向邻居的几何。
const _: () = assert!(core::mem::offset_of!(Instance, affine) == 0);
const _: () = assert!(core::mem::offset_of!(Instance, inverse) == 48);
const _: () = assert!(core::mem::offset_of!(Instance, geom_id) == 96);
const _: () = assert!(core::mem::offset_of!(Instance, mat_id) == 100);
const _: () = assert!(core::mem::offset_of!(Instance, flags) == 104);
const _: () = assert!(core::mem::offset_of!(Instance, pad) == 108);
/// `pad` 之后就是下一条记录：末字段偏移 + 4 == 总长（**无尾部填充**）。
const _: () = assert!(core::mem::offset_of!(Instance, pad) + 4 == INSTANCE_BYTES);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_slot_is_112_bytes_flat() {
        assert_eq!(core::mem::size_of::<Instance>(), 112);
        assert_eq!(core::mem::align_of::<Instance>(), 4);
        // 槽宽整除：`len * 112` 才是 GPU 缓冲的正确字节数。
        let v = [Instance::identity(1, 2); 3];
        assert_eq!(core::mem::size_of_val(&v), 3 * INSTANCE_BYTES);
    }

    #[test]
    fn identity_instance_holds_identity_matrices() {
        let i = Instance::identity(7, 9);
        assert!(i.is_identity());
        assert_eq!(i.geom_id, 7);
        assert_eq!(i.mat_id, 9);
        assert_eq!(i.pad, 0);
        // 恒等：对角 1、其余 0（含平移列）。
        for r in 0..3 {
            for c in 0..4 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert_eq!(i.affine[r * 4 + c], want, "affine[{r}][{c}]");
                assert_eq!(i.inverse[r * 4 + c], want, "inverse[{r}][{c}]");
            }
        }
    }

    #[test]
    fn identity_transform_is_bitwise_exact_for_finite_inputs() {
        // W3 判据「identity 实例 vs 展平版逐位一致」的可证基础：
        // 仿射乘加在 `x·1.0 + 0.0` 下精确（IEEE 754 乘/加对有限非零值无损）。
        //
        // 唯一例外是 `-0.0`：`0.0·(-0.0) = -0.0`，而 `-0.0 + 0.0 = +0.0`，
        // 故 `-0.0` 会被归一成 `+0.0`（数值相等、位型不同）。这不是缺陷，
        // 而是 W3 判据要用 `+0.0` 作原点、不走 `-0.0` 的原因 —— 先钉在这里，
        // 免得 W3 拿 `-0.0` 断言"逐位"然后被这条 IEEE 规则绊住。
        let m = Instance::identity(1, 1).affine;
        let apply = |x: f32, y: f32, z: f32| -> [f32; 3] {
            [
                m[0] * x + m[1] * y + m[2] * z + m[3],
                m[4] * x + m[5] * y + m[6] * z + m[7],
                m[8] * x + m[9] * y + m[10] * z + m[11],
            ]
        };
        for v in [0.0f32, 1.0, -1.5, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1e30, -3.25e-7] {
            let out = apply(v, v, v);
            assert_eq!(out[0].to_bits(), v.to_bits(), "{v:e} 的 x 不逐位");
            assert_eq!(out[1].to_bits(), v.to_bits(), "{v:e} 的 y 不逐位");
            assert_eq!(out[2].to_bits(), v.to_bits(), "{v:e} 的 z 不逐位");
        }
        // `-0.0` 的具体行为（登记为已知漂移，不是 bug）。
        assert_eq!(apply(-0.0, -0.0, -0.0)[0], 0.0);
        assert_eq!(apply(-0.0, -0.0, -0.0)[0].to_bits(), 0.0f32.to_bits());
        // NaN / Inf 也不在两个方向上"造"新值（逐位跟随输入位型）。
        assert!(apply(f32::INFINITY, 0.0, 0.0)[0].is_infinite());
        assert!(apply(f32::NAN, 0.0, 0.0)[0].is_nan());
    }

    // -----------------------------------------------------------------
    // W3 落地的变换数学
    // -----------------------------------------------------------------

    /// 归一化口径**必须**与 `integrator::triangle_unit_normal` 同一条公式：
    /// `n · (1/√(n·n))`。两处若一条用 `n/√`、一条用 `n·(1/√)`，identity
    /// 判据会在末位漂。
    fn norm_like_engine(n: Vec3) -> Vec3 {
        let len2 = n.dot(&n);
        if len2 > 1e-24 {
            n * (1.0 / len2.sqrt())
        } else {
            Vec3::splat(0.0)
        }
    }

    /// 非均匀比例 + 剪切：`A⁻¹` 必须真的把点变回去（残差 ≤ 1e-6）。
    #[test]
    fn from_affine_inverts_nonuniform_scale_and_shear() {
        // 行主序 3×4：非均匀缩放 (3, 0.5, 1) + z←y 剪切 + 平移。
        let inst = Instance::from_affine(
            [
                3.0, 0.25, 0.0, 10.0, //
                0.0, 0.5, 0.0, -2.0, //
                0.0, 0.9, 1.0, 0.5,
            ],
            0,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("非退化仿射应可求逆");
        let r = inst.inverse_residual();
        assert!(r <= 1e-6, "A·A⁻¹ 与恒等的最大偏差 {r:e} 超 1e-6");
        // 点往返：p → 世界 → 局部。
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, -2.5, 3.25),
            Vec3::new(-7.0, 0.125, 1.0e3),
        ] {
            let back = inst.inv_point(inst.apply_point(p));
            let d = (back - p).length();
            assert!(d <= 1e-5 * (1.0 + p.length()), "点往返误差 {d:e}（p = {p:?}）");
        }
        // 方向（不含平移）：往返一致。
        let d0 = Vec3::new(0.5, -1.0, 2.0);
        let d1 = inst.inv_dir(inst.apply_dir(d0));
        assert!((d1 - d0).length() <= 1e-6 * d0.length(), "方向往返漂移");
    }

    /// 恒等仿射走快路径：`flags` 打位、逆**直接抄恒等**（不求逆）。
    #[test]
    fn identity_affine_takes_exact_fast_path() {
        let i = Instance::from_affine(IDENTITY_AFFINE, 3, INSTANCE_MAT_FROM_GEOM)
            .expect("恒等可构造");
        assert!(i.is_identity());
        assert!(!i.is_mirrored());
        assert_eq!(i.inverse, IDENTITY_AFFINE);
        assert_eq!(i.inverse_residual(), 0.0);
        assert_eq!(i.material_override(), None, "mat_id 哨兵 ⇒ 材质取自共享几何");
        assert_eq!(i.geom_id, 3);
        // 非哨兵 ⇒ 覆盖源（`meshes` 下标）。
        let ov = Instance::from_affine(IDENTITY_AFFINE, 3, 0).unwrap();
        assert_eq!(ov.material_override(), Some(0));
        // `-0.0` 在数值上也是恒等（`==` 而非位比较）。
        let mut m = IDENTITY_AFFINE;
        m[3] = -0.0;
        assert!(Instance::from_affine(m, 0, 0).unwrap().is_identity());
    }

    /// 退化仿射（det = 0 / 非有限）⇒ `None`，绝不返回一个假逆。
    #[test]
    fn degenerate_affine_is_rejected() {
        // 压扁 z：det = 0。
        let flat = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(Instance::from_affine(flat, 0, 0).is_none());
        // 非有限：NaN 分量。
        let mut nan = IDENTITY_AFFINE;
        nan[0] = f32::NAN;
        assert!(Instance::from_affine(nan, 0, 0).is_none());
        // 极小 det：逆元溢出 ⇒ 也拒（否则渲染路径会拿到 inf 的逆）。
        let tiny = [1e-30, 0.0, 0.0, 0.0, 0.0, 1e-30, 0.0, 0.0, 0.0, 0.0, 1e-30, 0.0];
        assert!(Instance::from_affine(tiny, 0, 0).is_none());
    }

    /// **非均匀缩放的逆转置法线**（选型报告"最难的三点"之一）：
    /// 实例法线必须与"把顶点烘到世界再按绕序求法线"（展平版）一致；
    /// 直接乘变换矩阵的朴素做法必须**明显不同**（否则这条测试没意义）。
    #[test]
    fn nonuniform_normal_matches_flattened_geometry() {
        // 局部三角形（法线带 x/y 分量：若三角形法线正对 z 轴，
        // 对角缩放根本改不动它，这条测试就抓不住朴素做法）。
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.5),
        );
        let n_local = (b - a).cross(&(c - a));
        let inst = Instance::from_affine(
            [
                3.0, 0.0, 0.0, 0.0, //
                0.0, 0.5, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            0,
            0,
        )
        .unwrap();
        // 展平版：顶点先烘到世界空间再求绕序法线。
        let (wa, wb, wc) = (inst.apply_point(a), inst.apply_point(b), inst.apply_point(c));
        let n_flat = norm_like_engine((wb - wa).cross(&(wc - wa)));
        // 实例版：局部法线过逆转置（再按同一条归一化公式）。
        let n_inst = norm_like_engine(inst.inv_transpose_normal(n_local));
        let dot = n_flat.dot(&n_inst);
        assert!(dot >= 1.0 - 1e-6, "逆转置法线与展平法线不一致：dot = {dot}");
        // 朴素做法（直接乘 A）必须偏 —— 否则本测试抓不住任何东西。
        let naive = norm_like_engine(inst.apply_dir(n_local));
        assert!(
            naive.dot(&n_flat) < 0.99,
            "直接乘变换矩阵竟然也对（dot = {}）：本测试失去鉴别力",
            naive.dot(&n_flat)
        );
    }

    /// 镜像实例（det < 0）：绕序法线的世界方向是 `det·A⁻ᵀ·n`，
    /// 少反一个号就是"整片背光"。
    #[test]
    fn mirrored_instance_flips_normal_sign() {
        let inst = Instance::from_affine(
            [
                -1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            0,
            0,
        )
        .unwrap();
        assert!(inst.is_mirrored());
        let (a, b, c) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let n_local = (b - a).cross(&(c - a));
        let (wa, wb, wc) = (inst.apply_point(a), inst.apply_point(b), inst.apply_point(c));
        let n_flat = norm_like_engine((wb - wa).cross(&(wc - wa)));
        let n_inst = norm_like_engine(inst.inv_transpose_normal(n_local));
        assert!(n_flat.dot(&n_inst) >= 1.0 - 1e-6, "镜像实例法线方向错");
    }

    /// identity 实例的局部射线 = 世界射线，**逐位**（含 -0.0 与次正规数）。
    #[test]
    fn identity_local_ray_is_bitwise_identical() {
        let inst = Instance::identity(0, 0);
        for v in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-0.0, 1.5, -2.25),
            Vec3::new(1e30, -3.25e-7, f32::MIN_POSITIVE),
        ] {
            let (lo, ld) = inst.local_ray(v, v);
            assert_eq!(lo.x.to_bits(), v.x.to_bits());
            assert_eq!(lo.y.to_bits(), v.y.to_bits());
            assert_eq!(lo.z.to_bits(), v.z.to_bits());
            assert_eq!(ld.x.to_bits(), v.x.to_bits());
        }
    }

    /// 世界 AABB 必须包住**全部 8 角**（剪切下只变换 min/max 会漏）。
    #[test]
    fn world_aabb_covers_all_corners_under_shear() {
        // 45° 剪切：x ← x + y。局部盒 [-1,1]³ 的世界盒应为
        // [-2,2] × [-1,1] × [-1,1]（只变换两个角点会得到 [-1,2] 之类）。
        let inst = Instance::from_affine(
            [
                1.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            0,
            0,
        )
        .unwrap();
        let b = inst.world_aabb(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!((b.min.x + 2.0).abs() < 1e-6, "min.x = {}（应 -2）", b.min.x);
        assert!((b.max.x - 2.0).abs() < 1e-6, "max.x = {}", b.max.x);
        assert!((b.min.y + 1.0).abs() < 1e-6);
        assert!((b.max.y - 1.0).abs() < 1e-6);
    }

    /// 空盒（`min > max`）不引入点，返回空盒 —— TLAS 用 `Aabb::empty()`
    /// 表示"该实例永不命中"，`grow_point` 一旦被喂进一个点就会变成有效盒。
    #[test]
    fn world_aabb_of_empty_box_stays_empty() {
        let inst = Instance::identity(0, 0);
        let b = inst.world_aabb(Vec3::splat(1.0), Vec3::splat(-1.0));
        assert!(b.min.x.is_infinite() && b.min.x > 0.0, "应为 Aabb::empty()");
        assert!(b.max.x.is_infinite() && b.max.x < 0.0);
    }
}
