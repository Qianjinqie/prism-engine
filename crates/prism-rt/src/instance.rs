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
}
