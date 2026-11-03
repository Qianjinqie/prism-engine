//! W3（GPU 侧）：实例记录 + 两级 BVH 的**宿主侧编码**。
//!
//! # 这里是 CPU 侧的"唯一事实源"，三后端只读字节
//!
//! 选型报告 §4 定稿：实例记录 **112 B** =
//! `3×4 仿射（48 B）+ 3×4 逆（48 B）+ geom_id（4）+ mat_id（4）+ flags（4）+ pad（4）`，
//! 逆矩阵在 **CPU 算一次**、随记录走，三后端读同一份——**禁止各后端各自求逆**
//! （三个浮点求逆路径会给出三个不同的 `t`，逐位一致立刻破）。
//!
//! # 与 `prism-rt::instance` 的关系（两份实现，一份字节）
//!
//! `prism-gpu` **不得**反引 `prism-rt`（依赖方向：`prism-rt → prism-gpu`），
//! 故变换数学在本文件与 `prism_rt::instance` 各写一份。两份必须**逐位相同**：
//! `prism-rt` 侧有一条跨 crate 判据（`w3_gpu_instance_bytes_match_cpu_authority`）
//! 拿同一组仿射分别过两条实现，比对 112 B 编码的**逐字节**结果。任何一处
//! 换了运算顺序（比如把 `a*b + c*d` 改成 `c*d + a*b`）都会在那条测试上红。
//!
//! # 布局（`[u32; 28]` 视角，4 B 对齐，**稠密无填充**）
//!
//! | 槽 | 内容 |
//! |---|---|
//! | 0..12 | `affine`（行主序 `[m00,m01,m02,tx, m10,…]`） |
//! | 12..24 | `inverse`（同上；CPU 一次算好） |
//! | 24 | `geom_id`（BLAS 句柄；单 BLAS 剖面下恒为 0） |
//! | 25 | `mat_id`（材质；[`INSTANCE_MAT_FROM_GEOM`] = 取共享几何默认） |
//! | 26 | `flags`（bit0 identity / bit1 mirrored） |
//! | 27 | `node_base`（G-8 多 BLAS：该实例所属 BLAS 的**节点块起点**；单 BLAS 剖面下恒 0） |
//!
//! 与 `TRIANGLE_SLOTS` / `NODE_SLOTS` 的"每 `vec3` 补 pad"不同：实例记录**不做**
//! `vec4` 化补齐（补成 4×4 会变 128 B/条，违背 §4 的 112 B 定稿）。代价是
//! WGSL 侧不能用 `mat3x3<f32>`/`vec3<f32>` 数组直接映射，只能逐标量读——
//! 这恰好也是保守类型策略（`f32`/`u32` 单值）想要的形式。

use prism_geometry::bvh::Aabb;
use prism_math::Vec3;

use crate::scene_abi::NODE_SLOTS;

/// 单条实例记录的字节数（选型报告 §4 定稿；路线图 W3 判据「112 B/实例」）。
pub const INSTANCE_BYTES: usize = 112;

/// 单条实例记录的 u32 槽数（= 112 / 4 = 28）。
pub const INSTANCE_SLOTS: u32 = 28;

/// 仿射子矩阵的 f32 个数（3 行 × 4 列）。
pub const INSTANCE_AFFINE_WORDS: usize = 12;

/// 逆矩阵的 f32 个数（3 行 × 4 列）。
pub const INSTANCE_INVERSE_WORDS: usize = 12;

/// `inverse` 在记录内的槽偏移（= 12）。
pub const INSTANCE_INVERSE_OFFSET: usize = INSTANCE_AFFINE_WORDS;

/// `geom_id` 在记录内的槽偏移（= 24）。
pub const INSTANCE_GEOM_OFFSET: usize = INSTANCE_AFFINE_WORDS + INSTANCE_INVERSE_WORDS;

/// `mat_id` 在记录内的槽偏移（= 25）。
pub const INSTANCE_MAT_OFFSET: usize = INSTANCE_GEOM_OFFSET + 1;

/// `flags` 在记录内的槽偏移（= 26）。
pub const INSTANCE_FLAGS_OFFSET: usize = INSTANCE_MAT_OFFSET + 1;

/// `node_base` 在记录内的槽偏移（= 27）。
///
/// # 为什么用"原来的 pad"承载（G-8 多 BLAS，零 ABI 剖面变更）
///
/// 多 BLAS 下 `nodes` 是**所有** BLAS 的节点块顺序拼接，shader 需要"这个实例
/// 属于哪一块"的起点。三个选项：①新增 params 表（`PARAMS_WORDS_V3` 变长、
/// 三后端缓冲尺寸都要动）；②新增绑定（bind group 布局变更）；③**把已有的
/// `pad` 字赋予语义**。③ 的代价只有一条：112 B / 28 槽的剖面一个字都没变
/// （`INSTANCE_BYTES` / `INSTANCE_SLOTS` 不动），单 BLAS 时该字仍是 0
/// ⇒ 既有字节逐位不变。故取 ③。
pub const INSTANCE_NODE_BASE_OFFSET: usize = INSTANCE_FLAGS_OFFSET + 1;

/// `flags` 位：变换为恒等（`affine == inverse == [I|0]`）。
///
/// 恒等是「identity 实例 vs 展平版**逐位一致**」判据的可用形态：`x·1.0 + 0.0`
/// 在 IEEE 下精确（含 `-0.0`）⇒ `M·p == p` 逐位。故 shader 侧 identity 分支
/// **一次乘法都不做**，直接拿世界射线去查 BLAS。
pub const INSTANCE_FLAG_IDENTITY: u32 = 1;

/// `flags` 位：仿射的 3×3 线性块行列式为负（**镜像**实例）。
///
/// 绕序法线的世界方向是 `det(A)·A⁻ᵀ·n_local`，不是 `A⁻ᵀ·n_local`；`det < 0`
/// 时两者反向。det 在 [`InstanceReq::from_affine`] 里算一次并记进标志位。
pub const INSTANCE_FLAG_MIRRORED: u32 = 1 << 1;

/// `mat_id` 哨兵：材质取自**共享几何自身**（GPU 剖面 = `CURVE_ALBEDO` 中性反照率）。
///
/// 与 `prism_rt::instance::INSTANCE_MAT_FROM_GEOM` **同值**（`u32::MAX`，因为
/// `0` 是合法下标）。
pub const INSTANCE_MAT_FROM_GEOM: u32 = u32::MAX;

/// 单 BLAS 剖面下唯一合法的 `geom_id`。
///
/// # 为什么只有一个 BLAS
///
/// V3 剖面的 `triangles`/`nodes` 两块缓冲就是**那一个** BLAS（原型几何）。
/// 多原型要再开两块缓冲，而 W3 的判据（"1 万实例 BLAS 只建一次"）只需要
/// 一个共享原型就能验：1 万实例 × 1 个 2048 三角的 BLAS。`geom_id` 字段
/// 保留语义（将来扩多 BLAS 时是下标），本版非 0 一律**报错**而不是静默忽略。
pub const INSTANCE_GEOM_SHARED: u32 = 0;

/// 恒等仿射（3×4 行主序）。
pub const IDENTITY_AFFINE: [f32; INSTANCE_AFFINE_WORDS] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0,
];

/// 几何实例记录（GPU 侧；与 `prism_rt::instance::Instance` 同布局同字节）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceReq {
    /// 3×4 仿射（行主序；`[3]`/`[7]`/`[11]` 为平移）。
    pub affine: [f32; INSTANCE_AFFINE_WORDS],
    /// 3×4 逆（行主序；CPU 一次算好，三后端读同一份）。
    pub inverse: [f32; INSTANCE_INVERSE_WORDS],
    /// 几何（BLAS）句柄。
    pub geom_id: u32,
    /// 材质句柄（[`INSTANCE_MAT_FROM_GEOM`] = 取共享几何默认）。
    pub mat_id: u32,
    /// 位标志（[`INSTANCE_FLAG_IDENTITY`] / [`INSTANCE_FLAG_MIRRORED`]）。
    pub flags: u32,
}

impl InstanceReq {
    /// 恒等实例（`M = I`, `t = 0`，`inverse` 同值，`flags` 置 identity 位）。
    pub const fn identity(geom_id: u32, mat_id: u32) -> Self {
        InstanceReq {
            affine: IDENTITY_AFFINE,
            inverse: IDENTITY_AFFINE,
            geom_id,
            mat_id,
            flags: INSTANCE_FLAG_IDENTITY,
        }
    }

    /// 平移实例（`M = I`, `t = (tx,ty,tz)`）——测试与导入器最常用的非恒等形态。
    ///
    /// 逆平移是 `-t`（线性块为 I，求逆退化成正负号），故不必走除法：
    /// 这条让"纯平移"实例的逆**逐位精确**（`-0.0` 也精确），
    /// 是 `w3_translation_only_instances_match_flattened` 的构造性基础。
    pub fn translation(t: Vec3, geom_id: u32, mat_id: u32) -> Self {
        if t.x == 0.0 && t.y == 0.0 && t.z == 0.0 {
            return Self::identity(geom_id, mat_id);
        }
        InstanceReq {
            affine: [1.0, 0.0, 0.0, t.x, 0.0, 1.0, 0.0, t.y, 0.0, 0.0, 1.0, t.z],
            inverse: [1.0, 0.0, 0.0, -t.x, 0.0, 1.0, 0.0, -t.y, 0.0, 0.0, 1.0, -t.z],
            geom_id,
            mat_id,
            flags: 0,
        }
    }

    /// 各向异性缩放 + 平移实例（`M = diag(sx,sy,sz) + t`）。
    ///
    /// 逆是 `diag(1/sx,1/sy,1/sz)` + `-t/s`；`det < 0`（含奇数个负缩放）
    /// 由 [`Self::from_affine`] 统一记标志位——本函数只是它的便捷入口，
    /// **不另算一遍 det**。
    pub fn scale_translate(s: Vec3, t: Vec3, geom_id: u32, mat_id: u32) -> Option<Self> {
        Self::from_affine(
            [
                s.x, 0.0, 0.0, t.x, //
                0.0, s.y, 0.0, t.y, //
                0.0, 0.0, s.z, t.z,
            ],
            geom_id,
            mat_id,
        )
    }

    /// **唯一**构造入口：`affine`（3×4 行主序）→ 完整记录（含逆）。
    ///
    /// 逐式镜像 `prism_rt::instance::Instance::from_affine`（余子式 + `det` 一次
    /// 算好 + 恒等快路径）。`None` = 非有限 / `det == 0` / 逆元溢出（退化实例
    /// 的"逆"不存在，任何兜底值都会让射线走进 NaN 分支）。
    pub fn from_affine(
        affine: [f32; INSTANCE_AFFINE_WORDS],
        geom_id: u32,
        mat_id: u32,
    ) -> Option<Self> {
        if !affine.iter().all(|v| v.is_finite()) {
            return None;
        }
        // 恒等：数值相等即走快路径（`-0.0 == 0.0`，故 `-0.0` 的平移列也算恒等）。
        if affine == IDENTITY_AFFINE {
            return Some(InstanceReq {
                affine,
                inverse: IDENTITY_AFFINE,
                geom_id,
                mat_id,
                flags: INSTANCE_FLAG_IDENTITY,
            });
        }
        let (a00, a01, a02, tx) = (affine[0], affine[1], affine[2], affine[3]);
        let (a10, a11, a12, ty) = (affine[4], affine[5], affine[6], affine[7]);
        let (a20, a21, a22, tz) = (affine[8], affine[9], affine[10], affine[11]);
        let c00 = a11 * a22 - a12 * a21;
        let c01 = a12 * a20 - a10 * a22;
        let c02 = a10 * a21 - a11 * a20;
        let det = a00 * c00 + a01 * c01 + a02 * c02;
        let inv_det = 1.0 / det;
        if !inv_det.is_finite() {
            return None;
        }
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
        let itx = -(i00 * tx + i01 * ty + i02 * tz);
        let ity = -(i10 * tx + i11 * ty + i12 * tz);
        let itz = -(i20 * tx + i21 * ty + i22 * tz);
        let inverse = [
            i00, i01, i02, itx, //
            i10, i11, i12, ity, //
            i20, i21, i22, itz,
        ];
        if !inverse.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut flags = 0u32;
        if det < 0.0 {
            flags |= INSTANCE_FLAG_MIRRORED;
        }
        Some(InstanceReq { affine, inverse, geom_id, mat_id, flags })
    }

    /// 是否恒等（读 `flags`，不重算矩阵）。
    pub const fn is_identity(&self) -> bool {
        self.flags & INSTANCE_FLAG_IDENTITY != 0
    }

    /// 是否镜像（`det(A) < 0`）。
    pub const fn is_mirrored(&self) -> bool {
        self.flags & INSTANCE_FLAG_MIRRORED != 0
    }

    /// 材质覆盖：`Some(mat_id)` 或 `None`（取共享几何默认）。
    pub const fn material_override(&self) -> Option<u32> {
        if self.mat_id == INSTANCE_MAT_FROM_GEOM {
            None
        } else {
            Some(self.mat_id)
        }
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

    /// 世界点 → 局部点（`A⁻¹·p + t⁻¹`）。
    pub fn inv_point(&self, p: Vec3) -> Vec3 {
        let a = &self.inverse;
        Vec3::new(
            a[0] * p.x + a[1] * p.y + a[2] * p.z + a[3],
            a[4] * p.x + a[5] * p.y + a[6] * p.z + a[7],
            a[8] * p.x + a[9] * p.y + a[10] * p.z + a[11],
        )
    }

    /// 世界方向 → 局部方向（`A⁻¹·d`，**不归一化**）。
    pub fn inv_dir(&self, d: Vec3) -> Vec3 {
        let a = &self.inverse;
        Vec3::new(
            a[0] * d.x + a[1] * d.y + a[2] * d.z,
            a[4] * d.x + a[5] * d.y + a[6] * d.z,
            a[8] * d.x + a[9] * d.y + a[10] * d.z,
        )
    }

    /// 世界射线 → 局部射线（**`t` 参数不变**；identity 原样返回）。
    ///
    /// 因为同一个仿射同时作用于 `o` 与 `d`，`o_l + t·d_l ≡ M⁻¹(o + t·d)` 对
    /// **同一个 `t`** 成立 ⇒ `T_MIN` 与 `t_max` 都是"世界 `t`"，可直接当局部
    /// 查询的界用。这是选型报告 §4「`t` 要按缩放换算」那条的实现选择：
    /// 把难点**消掉**（不归一化），而不是算对。
    pub fn local_ray(&self, ro: Vec3, rd: Vec3) -> (Vec3, Vec3) {
        if self.is_identity() {
            (ro, rd)
        } else {
            (self.inv_point(ro), self.inv_dir(rd))
        }
    }

    /// 局部法线 → 世界法线方向（**逆转置**，含镜像反号），未归一化。
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
    /// 非均匀缩放 + 旋转下只变换 `min`/`max` 两个角点是**错的**（角点互换）。
    /// 空盒（`min > max`）返回空盒（不引入任何点）。
    pub fn world_aabb(&self, min: Vec3, max: Vec3) -> Aabb {
        let mut out = Aabb::empty();
        if !(min.x <= max.x && min.y <= max.y && min.z <= max.z) {
            return out;
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

    /// 编码成 112 B（little-endian；`node_base` 恒 0 = 单 BLAS 剖面）。
    pub fn to_bytes(&self) -> [u8; INSTANCE_BYTES] {
        self.to_bytes_with_base(0)
    }

    /// 编码成 112 B，并把槽 27 写成 `node_base`（G-8 多 BLAS 用）。
    pub fn to_bytes_with_base(&self, node_base: u32) -> [u8; INSTANCE_BYTES] {
        let mut out = [0u8; INSTANCE_BYTES];
        for (k, v) in self.affine.iter().enumerate() {
            out[k * 4..k * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        for (k, v) in self.inverse.iter().enumerate() {
            let slot = INSTANCE_INVERSE_OFFSET + k;
            out[slot * 4..slot * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        out[INSTANCE_GEOM_OFFSET * 4..INSTANCE_GEOM_OFFSET * 4 + 4]
            .copy_from_slice(&self.geom_id.to_le_bytes());
        out[INSTANCE_MAT_OFFSET * 4..INSTANCE_MAT_OFFSET * 4 + 4]
            .copy_from_slice(&self.mat_id.to_le_bytes());
        out[INSTANCE_FLAGS_OFFSET * 4..INSTANCE_FLAGS_OFFSET * 4 + 4]
            .copy_from_slice(&self.flags.to_le_bytes());
        out[INSTANCE_NODE_BASE_OFFSET * 4..INSTANCE_NODE_BASE_OFFSET * 4 + 4]
            .copy_from_slice(&node_base.to_le_bytes());
        out
    }
}

/// 把实例表编码成扁平 `array<u32>`（[`INSTANCE_SLOTS`] 槽/条；`node_base` 全 0）。
pub fn encode_instances(instances: &[InstanceReq]) -> Vec<u8> {
    let mut out = Vec::with_capacity(instances.len() * INSTANCE_BYTES);
    for inst in instances {
        out.extend_from_slice(&inst.to_bytes());
    }
    out
}

/// 按 [`InstanceSet`] 里的 `node_bases` 编码（G-8 多 BLAS 的唯一入口）。
pub fn encode_instance_set(set: &InstanceSet) -> Vec<u8> {
    let mut out = Vec::with_capacity(set.instances.len() * INSTANCE_BYTES);
    for (i, inst) in set.instances.iter().enumerate() {
        let base: u32 = set.node_bases.get(i).copied().unwrap_or(0);
        out.extend_from_slice(&inst.to_bytes_with_base(base));
    }
    out
}

/// 实例 TLAS 的编码产物（CPU 侧结构，**三后端读同一份字节**）。
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceSet {
    /// 已按 TLAS 叶子序**重排**的实例（叶子区间 `start..start+count` 直接可迭代）。
    pub instances: Vec<InstanceReq>,
    /// `NODE_SLOTS` 槽/条 的 TLAS 节点字节流（叶子的 `start/count` 索引进上一项）。
    pub nodes: Vec<u8>,
    /// 重排映射：第 `i` 个（重排后）实例在输入里的下标。
    pub order: Vec<u32>,
    /// 逐实例的 BLAS 节点块起点（与 `instances` 同序；单 BLAS 时全 0）。
    pub node_bases: Vec<u32>,
    /// 全场景世界包围盒（空实例表 ⇒ 空盒）。
    pub bounds: Aabb,
}

/// 建 TLAS 并重排实例（`proto_min/proto_max` = BLAS 原型的**局部**包围盒）。
///
/// # 为什么把实例表按 TLAS 叶子序重排
///
/// 扁平节点编码里叶子只能表达"`order` 数组的一段"（`start/count`）。若把
/// 实例表留在原序，就还要再上传一份 `order` 缓冲给三后端（多一个绑定 + 多
/// 一次越界可能）。按叶子序重排实例表之后，叶子区间**直接可用**：shader 里
/// `for i in start..start+count { instances[i] }`，一个多余的下标都没有。
///
/// 代价：实例下标不再等于输入下标（tie-break 与测试断言要按 `order` 回查）。
/// 这一点由 [`InstanceSet::order`] 显式给出，不靠"碰巧一样"。
pub fn build_instance_set(
    instances: &[InstanceReq],
    geom_bounds: &[(Vec3, Vec3)],
    geom_node_base: &[u32],
) -> InstanceSet {
    if instances.is_empty() {
        return InstanceSet {
            instances: Vec::new(),
            nodes: Vec::new(),
            order: Vec::new(),
            node_bases: Vec::new(),
            bounds: Aabb::empty(),
        };
    }
    // G-8 多 BLAS：每个实例的局部包围盒取自**它自己那个原型**。
    // 越界的 `geom_id` 在这里退化为空盒（编码层已经先判过，这里是兜底：
    // 空盒实例在 TLAS 里永不命中，不会变成"命中别人的几何"）。
    let aabbs: Vec<Aabb> = instances
        .iter()
        .map(|i| match geom_bounds.get(i.geom_id as usize) {
            Some((lo, hi)) => i.world_aabb(*lo, *hi),
            None => Aabb::empty(),
        })
        .collect();
    let bvh = prism_geometry::aabb_bvh::AabbBvh::build(&aabbs);
    let order: Vec<u32> = bvh.order().to_vec();
    let reordered: Vec<InstanceReq> = order
        .iter()
        .map(|&i| instances[i as usize])
        .collect();
    let node_bases: Vec<u32> = order
        .iter()
        .map(|&i| {
            geom_node_base
                .get(instances[i as usize].geom_id as usize)
                .copied()
                .unwrap_or(0)
        })
        .collect();
    let mut nodes = Vec::with_capacity(bvh.node_count() * NODE_SLOTS as usize * 4);
    for k in 0..bvh.node_count() {
        let (b, start, count, right) = match bvh.node(k) {
            Some(v) => v,
            None => break,
        };
        let mut block = [0u8; NODE_SLOTS as usize * 4];
        block[0..4].copy_from_slice(&b.min.x.to_bits().to_le_bytes());
        block[4..8].copy_from_slice(&b.min.y.to_bits().to_le_bytes());
        block[8..12].copy_from_slice(&b.min.z.to_bits().to_le_bytes());
        // 槽 3 = pad
        block[16..20].copy_from_slice(&b.max.x.to_bits().to_le_bytes());
        block[20..24].copy_from_slice(&b.max.y.to_bits().to_le_bytes());
        block[24..28].copy_from_slice(&b.max.z.to_bits().to_le_bytes());
        // 槽 7 = pad
        block[32..36].copy_from_slice(&start.to_le_bytes());
        block[36..40].copy_from_slice(&count.to_le_bytes());
        block[40..44].copy_from_slice(&right.to_le_bytes());
        nodes.extend_from_slice(&block);
    }
    InstanceSet { instances: reordered, nodes, order, node_bases, bounds: bvh_bounds(&aabbs) }
}

/// 图元 AABB 的并（与 `AabbBvh::build` 的 `bounds` 同口径）。
fn bvh_bounds(aabbs: &[Aabb]) -> Aabb {
    let mut out = Aabb::empty();
    for b in aabbs {
        out.grow(b);
    }
    out
}

// ---------------------------------------------------------------------------
// 编译期断言（112 B 在编译期就成立，不靠事后测）
// ---------------------------------------------------------------------------

/// 槽宽：112 B（选型报告 §4 定稿）。
const _: () = assert!(INSTANCE_BYTES == INSTANCE_SLOTS as usize * 4);
/// 两块 3×4 + 4 个 u32 恰好铺满 28 槽（漏一个就是越界读邻居）。
const _: () = assert!(INSTANCE_AFFINE_WORDS + INSTANCE_INVERSE_WORDS + 4 == INSTANCE_SLOTS as usize);
/// 字段偏移（shader 侧按这些槽号逐标量读，错一格 = 读到邻居的逆矩阵）。
const _: () = assert!(INSTANCE_INVERSE_OFFSET == 12);
const _: () = assert!(INSTANCE_GEOM_OFFSET == 24);
const _: () = assert!(INSTANCE_MAT_OFFSET == 25);
const _: () = assert!(INSTANCE_FLAGS_OFFSET == 26);
const _: () = assert!(INSTANCE_FLAGS_OFFSET + 1 == 27);
/// 标志位互不重叠（`flags` 是位域，重叠会让 identity 实例被当成镜像）。
const _: () = assert!(INSTANCE_FLAG_IDENTITY & INSTANCE_FLAG_MIRRORED == 0);
/// 哨兵不能落到合法下标 0 上。
const _: () = assert!(INSTANCE_MAT_FROM_GEOM != 0);

#[cfg(test)]
mod tests {
    use super::*;

    /// identity 记录的字节布局：仿射/逆都是 `[I|0]`，flags = 1，pad = 0。
    #[test]
    fn identity_bytes_are_dense_and_padded_with_zero() {
        let inst = InstanceReq::identity(INSTANCE_GEOM_SHARED, INSTANCE_MAT_FROM_GEOM);
        let b = inst.to_bytes();
        assert_eq!(b.len(), INSTANCE_BYTES);
        // 槽 0..12 = 仿射；12..24 = 逆（同值）。
        for k in 0..12 {
            let v = f32::from_le_bytes([b[k * 4], b[k * 4 + 1], b[k * 4 + 2], b[k * 4 + 3]]);
            assert_eq!(v, IDENTITY_AFFINE[k], "仿射槽 {k}");
            let w = f32::from_le_bytes([
                b[(12 + k) * 4],
                b[(12 + k) * 4 + 1],
                b[(12 + k) * 4 + 2],
                b[(12 + k) * 4 + 3],
            ]);
            assert_eq!(w, IDENTITY_AFFINE[k], "逆槽 {k}");
        }
        assert_eq!(
            u32::from_le_bytes([b[96], b[97], b[98], b[99]]),
            INSTANCE_GEOM_SHARED
        );
        assert_eq!(
            u32::from_le_bytes([b[100], b[101], b[102], b[103]]),
            INSTANCE_MAT_FROM_GEOM
        );
        assert_eq!(
            u32::from_le_bytes([b[104], b[105], b[106], b[107]]),
            INSTANCE_FLAG_IDENTITY
        );
        assert_eq!(&b[108..112], &[0u8; 4], "pad 必须恒 0（确定性）");
    }

    /// 非均匀缩放的逆转置法线：与世界两条边正交（且反向对照"直接乘 A"不成立）。
    ///
    /// # 为什么法线不能取 +Z
    ///
    /// `A = diag(3, 1, 0.5)` 下 `+Z` 法线经 `A^-T` 仍是 `+Z`，两条世界边也与
    /// `+Z` 正交——**"直接乘 A"同样给出 `+Z`**，对照组就废了（本测试的第一版
    /// 正是这样写错的：判据看着绿，其实什么都没测）。故取一个法线与三个缩放轴
    /// 都不平行的三角形：`n_local = (-1,-1,0)`。
    #[test]
    fn inv_transpose_normal_stays_perpendicular_under_nonuniform_scale() {
        let inst = InstanceReq::scale_translate(
            Vec3::new(3.0, 1.0, 0.5),
            Vec3::splat(0.0),
            INSTANCE_GEOM_SHARED,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("非退化缩放必须可求逆");
        // 局部三角 v0=(0,0,0) / v1=(1,-1,0) / v2=(0,0,1) ⇒ n_local = e1 × e2 = (-1,-1,0)。
        let e1 = Vec3::new(1.0, -1.0, 0.0);
        let e2 = Vec3::new(0.0, 0.0, 1.0);
        let n_local = e1.cross(&e2);
        assert!((n_local.x + 1.0).abs() < 1e-6 && (n_local.y + 1.0).abs() < 1e-6);
        let n_world = inst.inv_transpose_normal(n_local).normalize();
        // 世界边 = A·e_i（A 是 diag(3,1,0.5)，逐分量乘即矩阵乘）。
        let s = Vec3::new(3.0, 1.0, 0.5);
        let w1 = Vec3::new(e1.x * s.x, e1.y * s.y, e1.z * s.z);
        let w2 = Vec3::new(e2.x * s.x, e2.y * s.y, e2.z * s.z);
        let (u1, u2) = (w1.normalize(), w2.normalize());
        assert!(n_world.dot(&u1).abs() < 1e-6, "逆转置法线必须垂直于世界边 1");
        assert!(n_world.dot(&u2).abs() < 1e-6, "逆转置法线必须垂直于世界边 2");
        // 反向对照：直接乘 A 会给出非垂直的方向（否则这条测试抓不住东西）。
        let wrong = Vec3::new(
            inst.affine[0] * n_local.x + inst.affine[1] * n_local.y + inst.affine[2] * n_local.z,
            inst.affine[4] * n_local.x + inst.affine[5] * n_local.y + inst.affine[6] * n_local.z,
            inst.affine[8] * n_local.x + inst.affine[9] * n_local.y + inst.affine[10] * n_local.z,
        )
        .normalize();
        assert!(
            wrong.dot(&u1).abs() > 0.5,
            "对照组必须明显不垂直（否则判据无效）；实测 |cos| = {}",
            wrong.dot(&u1).abs()
        );
    }

    /// 镜像（det < 0）实例：**垂直于镜像轴**的局部法线必须反号。
    ///
    /// # `det·A^-T·n` 到底翻转什么
    ///
    /// 世界三角形的几何法线是 `(A e1) × (A e2) = det(A)·A^-T·(e1 × e2)`——它是
    /// **几何法线**，不是"每个分量取反"。故镜像 `x` 时 `+Z` 法线确实反号（世界
    /// 三角形绕序被翻），而镜像 `z` 时 `+Z` 法线**不**反号（沿镜像轴的分量不受
    /// 影响，世界三角形与局部三角形重合）。两条都要断言，否则"反号"这句话
    /// 会被写成一条只在特定轴上成立的假判据（本测试的第一版正是拿镜像 `z` +
    /// `+Z` 法线，结果必然红）。
    #[test]
    fn mirrored_instance_flips_normal_perpendicular_to_the_mirror_axis() {
        let mirror_x = InstanceReq::scale_translate(
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::splat(0.0),
            INSTANCE_GEOM_SHARED,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("可逆");
        assert!(mirror_x.is_mirrored(), "det = -1 必须置 MIRRORED 位");
        let n = mirror_x.inv_transpose_normal(Vec3::new(0.0, 0.0, 1.0));
        assert!(n.z < 0.0, "镜像 x 后 +Z 法线必须反号（否则整片背光）");

        // 正控：镜像**沿**法线方向（z）时，+Z 法线不动——几何上世界三角形与
        // 局部三角形逐点相同，绕序也没翻。这条反证"实现只是把分量取负"。
        let mirror_z = InstanceReq::scale_translate(
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::splat(0.0),
            INSTANCE_GEOM_SHARED,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("可逆");
        assert!(mirror_z.is_mirrored());
        let nz = mirror_z.inv_transpose_normal(Vec3::new(0.0, 0.0, 1.0));
        assert!(
            nz.z > 0.0,
            "镜像 z 后 +Z 法线应保持不变（det 因子作用在几何法线上，不是逐分量取反）"
        );
    }

    /// 世界 AABB：非均匀缩放 + 旋转下必须 8 角全算（只变换 min/max 会漏）。
    #[test]
    fn world_aabb_transforms_all_eight_corners() {
        // 绕 Z 轴 45°+ 缩放 → 角点互换。
        let c = core::f32::consts::FRAC_1_SQRT_2;
        let inst = InstanceReq::from_affine(
            [
                2.0 * c, -3.0 * c, 0.0, 0.0, //
                2.0 * c, 3.0 * c, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            INSTANCE_GEOM_SHARED,
            INSTANCE_MAT_FROM_GEOM,
        )
        .expect("可逆");
        let b = inst.world_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // 世界 x 半宽 = |2c| + |3c| ≈ 3.5355
        let want = 2.0 * c + 3.0 * c;
        assert!((b.max.x - want).abs() < 1e-5, "x 上界 {:.6} vs {:.6}", b.max.x, want);
        assert!((b.min.x + want).abs() < 1e-5);
        assert!((b.max.y - want).abs() < 1e-5);
    }

    /// TLAS 叶子区间必须覆盖全部实例且不重叠（重排后 `start..start+count` 可直接迭代）。
    #[test]
    fn tlas_leaves_cover_every_instance_exactly_once() {
        let insts: Vec<InstanceReq> = (0..20)
            .map(|k| {
                InstanceReq::translation(
                    Vec3::new(k as f32 * 2.5, 0.0, 0.0),
                    INSTANCE_GEOM_SHARED,
                    INSTANCE_MAT_FROM_GEOM,
                )
            })
            .collect();
        let set = build_instance_set(&insts, &[(Vec3::splat(-1.0), Vec3::splat(1.0))], &[0]);
        assert_eq!(set.instances.len(), 20);
        let mut seen = [0u32; 20];
        let node_count = set.nodes.len() / (NODE_SLOTS as usize * 4);
        assert!(node_count > 0);
        for k in 0..node_count {
            let b = &set.nodes[k * NODE_SLOTS as usize * 4..];
            let count = u32::from_le_bytes([b[36], b[37], b[38], b[39]]);
            let start = u32::from_le_bytes([b[32], b[33], b[34], b[35]]);
            if count == 0 {
                continue;
            }
            for i in start..start + count {
                seen[i as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "每个实例必须恰好被一个叶子覆盖");
        // 重排映射是 0..n 的置换
        let mut sorted = set.order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20u32).collect::<Vec<_>>());
    }

    // -----------------------------------------------------------------------
    // W3 判据（选型报告 §4.1 / §17.4-B 的实测落点）
    // -----------------------------------------------------------------------
    //
    // 这一组**不需要 GPU**：它量的是"实例遍历"这件数学本身。GPU 侧只要逐位
    // 复刻同一套算术（`scene_frame_v3.wgsl::instance_nearest` 与
    // [`instance_nearest`] 同构），这些判据就自动对 GPU 成立；GPU 上的像素
    // 级对拍另见 `vulkan/v3_frame.rs` 的 `real_device_*_instanced_*`。

    /// 世界 `T_MIN`（与 `prism-rt::integrator`、WGSL 侧同值）。
    const T_MIN: f32 = 1e-4;
    /// 世界 `t_max`（场景尺度上界，够大即可）。
    const T_MAX: f32 = 1e4;

    /// 单位立方体的 12 个三角形（原型几何 / BLAS 内容）。
    ///
    /// 立方体是**非凸**的复核对象：射线可能穿入再穿出，命中的是"最近的那一片"，
    /// 不是"某一片"——这正是实例遍历要保证的东西（若 TLAS 的 AABB 或局部变换
    /// 有一处算错，最近面会变）。
    fn unit_cube_tris() -> Vec<prism_geometry::bvh::Triangle> {
        let v = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let faces = [
            [0u32, 3, 2, 1], // -Z
            [4, 5, 6, 7],    // +Z
            [0, 1, 5, 4],    // -Y
            [3, 7, 6, 2],    // +Y
            [0, 4, 7, 3],    // -X
            [1, 2, 6, 5],    // +X
        ];
        let mut out = Vec::with_capacity(12);
        for f in faces {
            out.push(prism_geometry::bvh::Triangle {
                a: v[f[0] as usize],
                b: v[f[1] as usize],
                c: v[f[2] as usize],
            });
            out.push(prism_geometry::bvh::Triangle {
                a: v[f[0] as usize],
                b: v[f[2] as usize],
                c: v[f[3] as usize],
            });
        }
        out
    }

    /// 确定性 LCG（xorshift32）：同一 seed ⇒ 同一批射线 ⇒ 两跑可比。
    struct Lcg(u32);

    impl Lcg {
        const fn new(seed: u32) -> Lcg {
            Lcg(seed | 1)
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        /// `[-1, 1)` 均匀。
        fn next_unit(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// **identity 实例 ≡ 展平 BVH：`t` 位型逐位相同。**
    ///
    /// # 为什么能有"逐位"这种强判据
    ///
    /// `local_ray` 对 identity 实例**原样返回** `(ro, rd)`（`x·1.0 + 0.0` 在
    /// IEEE-754 下精确），两边的 slab 算术与 Möller–Trumbore 拿到的是同一串
    /// f32 位型 ⇒ `t` 必然逐位相同。这不是"测出来像"，是构造性的。
    ///
    /// # 与展平路径的 tie-break 差异（如实记账）
    ///
    /// 展平路径 `Bvh::ray_intersect_raw` 在 `t` 相等时按**原图元索引**升序择优，
    /// 实例路径按"先到先得"。故 `prim` 可能在共面重叠处不同——**`t` 不受影响**
    /// （等值才走 tie-break），本测试因此只钉 `t` 的位型，并把 `prim` 不一致的
    /// 条数打出来（不是断言，是证据）。
    #[test]
    fn identity_instance_hits_match_flattened_bvh_bitwise() {
        let tris = unit_cube_tris();
        let blas = prism_geometry::bvh::Bvh::build(&tris);
        let inst = InstanceReq::identity(INSTANCE_GEOM_SHARED, INSTANCE_MAT_FROM_GEOM);
        let set = build_instance_set(&[inst], &[(Vec3::splat(-1.0), Vec3::splat(1.0))], &[0]);
        // 走一遍**字节流**再解码：与 GPU 读到的是同一份字节，不是同一份内存。
        let nodes = decode_nodes(&set.nodes);
        let insts = decode_instances(&encode_instances(&set.instances));
        assert_eq!(nodes.len(), 1, "单实例 TLAS = 1 个叶子");
        assert_eq!(insts.len(), 1);

        let mut rng = Lcg::new(0x5EED_0917);
        let (mut hit, mut same_t, mut mismatched_prim) = (0u32, 0u32, 0u32);
        let mut compared = 0u32;
        for _ in 0..4096u32 {
            // 从球面外圈射向原点附近（保证有相当比例的命中）。
            let ro = Vec3::new(rng.next_unit() * 4.0, rng.next_unit() * 4.0, rng.next_unit() * 4.0);
            let to = Vec3::new(rng.next_unit() * 0.8, rng.next_unit() * 0.8, rng.next_unit() * 0.8);
            let d = to - ro;
            let len = d.length();
            if len < 1e-3 {
                continue;
            }
            let rd = d * (1.0 / len);
            compared += 1;
            let flat = blas.ray_intersect_raw(ro, rd, T_MIN, T_MAX);
            let vi = instance_nearest(&nodes, &insts, std::slice::from_ref(&blas), ro, rd, T_MIN, T_MAX);
            match (flat, vi) {
                (None, None) => {}
                (Some(f), Some(v)) => {
                    hit += 1;
                    assert_eq!(
                        f.t.to_bits(),
                        v.t.to_bits(),
                        "identity 实例的 t 必须与展平 BVH 逐位相同；\
                         flat={} (0x{:08x}) vs inst={} (0x{:08x})，射线 ro={ro:?} rd={rd:?}",
                        f.t,
                        f.t.to_bits(),
                        v.t,
                        v.t.to_bits()
                    );
                    same_t += 1;
                    // `f.tri` 是**原始**下标（`ray_intersect_raw` 的口径），
                    // `v.prim` 是**slot**（`ray_intersect_slot` 的口径）；两者
                    // 只差一个 `order` 映射，比较时必须先统一。
                    if f.tri != blas.order[v.prim as usize] {
                        mismatched_prim += 1;
                    }
                }
                (f, v) => panic!(
                    "identity 实例与展平 BVH 的命中/未命中不一致：flat={:?} inst={:?}（ro={ro:?} rd={rd:?}）",
                    f.map(|h| h.t),
                    v.map(|h| h.t)
                ),
            }
        }
        // 判据有效性：必须有相当比例的命中，否则"逐位一致"是在比 4096 个 None。
        assert!(
            hit > compared / 4,
            "命中数太少（{hit}/{compared}）——判据没测到东西，先修场景"
        );
        assert_eq!(same_t, hit);
        println!(
            "[W3 identity] {compared} 条射线、{hit} 条命中：t 位型逐位相同 {same_t}/{hit}；\
             prim 与展平路径不同 {mismatched_prim} 条（t 等值的 tie-break 差异，口径已注明）"
        );
    }

    /// **非 identity：逐命中 `t` 相对误差 ≤ 1e-6（1e5 射线），掠射按 §17.4-B 记账。**
    ///
    /// # 两条路径差在哪
    ///
    /// * 展平路径：把原型三角形**逐顶点**用同一个仿射变换到世界，再建世界 BVH；
    /// * 实例路径：世界射线经 `A⁻¹` 进局部，用**同一个 `t`** 在原型 BVH 上求交。
    ///
    /// 数学上 `o_l + t·d_l ≡ A⁻¹(o + t·d)` 对同一 `t` 恒成立，故差只来自 f32
    /// 舍入（顶点变换一次 vs 射线变换一次）。**掠射**（命中点法线与射线夹角
    /// 接近 90°，即 `|cos| = |n̂·d̂|` 很小）时 Möller–Trumbore 的行列式接近 0，
    /// 两条路径的舍入被放大——§17.4-B 在 CPU 侧量到 95/70,299 超阈、**全部**
    /// `|cos| < 0.37`。这里照同一口径分列，不用放宽阈值来凑绿。
    #[test]
    fn non_identity_hits_within_1e6_relative_t() {
        let tris = unit_cube_tris();
        let blas = prism_geometry::bvh::Bvh::build(&tris);
        // 旋转 30°(z) + 非均匀缩放 + 平移：三个自由度都动到。
        let c = 0.866_025_4_f32; // cos 30°
        let s = 0.5_f32; // sin 30°
        let (sx, sy, sz) = (1.7_f32, 0.8_f32, 1.3_f32);
        let (tx, ty, tz) = (0.3_f32, -0.2_f32, 0.5_f32);
        let affine = [
            sx * c, -sy * s, 0.0, tx, //
            sx * s, sy * c, 0.0, ty, //
            0.0, 0.0, sz, tz,
        ];
        let inst = InstanceReq::from_affine(affine, INSTANCE_GEOM_SHARED, INSTANCE_MAT_FROM_GEOM)
            .expect("非退化仿射必须可逆");
        assert!(!inst.is_identity());
        // 展平：逐顶点变换。
        let world: Vec<prism_geometry::bvh::Triangle> = tris
            .iter()
            .map(|t| prism_geometry::bvh::Triangle {
                a: inst.apply_point(t.a),
                b: inst.apply_point(t.b),
                c: inst.apply_point(t.c),
            })
            .collect();
        let world_bvh = prism_geometry::bvh::Bvh::build(&world);
        let set = build_instance_set(&[inst], &[(Vec3::splat(-1.0), Vec3::splat(1.0))], &[0]);
        let nodes = decode_nodes(&set.nodes);
        let insts = decode_instances(&encode_instances(&set.instances));

        // 原始下标 → `world_bvh.tris` 的 slot（`ray_intersect_raw` 给的是原始下标）。
        let mut slot_of_orig = vec![0u32; world_bvh.order.len()];
        for (slot, &orig) in world_bvh.order.iter().enumerate() {
            slot_of_orig[orig as usize] = slot as u32;
        }

        let mut rng = Lcg::new(0x1E6_2026);
        let (mut hit, mut over, mut over_grazing, mut grazing) = (0u32, 0u32, 0u32, 0u32);
        let mut over_small_t = 0u32;
        let mut min_t = f32::INFINITY;
        let mut worst_rel = 0.0f32;
        let mut worst_cos = 1.0f32;
        let mut worst_abs = 0.0f32;
        for _ in 0..100_000u32 {
            // 起点**落在半径 6 的球面上**（必须这样，不能用 [-9,9]³ 的立方体采样：
            // 立方体有相当比例落在世界 AABB 内部，会给出 t≈1e-3 的"从体内射出"
            // 样本，而那种样本的相对误差天然被 1/t 放大——见下方口径说明）。
            let dir = Vec3::new(rng.next_unit(), rng.next_unit(), rng.next_unit());
            let dlen = dir.length();
            if dlen < 1e-3 {
                continue;
            }
            let ro = dir * (6.0 / dlen);
            let to = Vec3::new(rng.next_unit() * 1.5, rng.next_unit() * 1.5, rng.next_unit() * 1.5);
            let d = to - ro;
            let len = d.length();
            if len < 1e-3 {
                continue;
            }
            let rd = d * (1.0 / len);
            let flat = world_bvh.ray_intersect_raw(ro, rd, T_MIN, T_MAX);
            let vi = instance_nearest(&nodes, &insts, std::slice::from_ref(&blas), ro, rd, T_MIN, T_MAX);
            let (f, v) = match (flat, vi) {
                (None, None) => continue,
                (Some(f), Some(v)) => (f, v),
                (f, v) => panic!(
                    "实例路径与展平路径的命中/未命中不一致：flat={:?} inst={:?}",
                    f.map(|h| h.t),
                    v.map(|h| h.t)
                ),
            };
            hit += 1;
            if f.t < min_t {
                min_t = f.t;
            }
            let rel = ((v.t - f.t) / f.t).abs();
            // 掠射判定用**展平路径（参考）命中的那个面**的世界法线 —— 与
            // `prism-rt` 的 `w3_instance_t_matches_flattened_within_1e_6` 同口径
            // （那里是 `flat.normal_at(x.shape, p)`）。
            //
            // # 为什么不能改用实例路径命中面的法线（本条判据的第一版踩过）
            //
            // 两条路径可以命中**不同的面**。典型情形：射线几乎平行于立方体的
            // `-Z` 面飞（`|rd.z| ≈ 0.005`），展平路径贴着 `-Z` 面命中，实例路径
            // 在相邻斜面（`+Y`）命中，两个 `t` 只差 1e-5 量级。此时实例面的法线
            // 与射线夹角 `|cos| ≈ 0.5`，用它会把这个**最典型的掠射**样本判成
            // "正面命中且超阈"，于是超阈数从 0 变成 22 —— 那是判据挑错了法线，
            // 不是实现算错了 `t`。参考面的法线才是数值条件的度量：`det` 接近 0
            // 正是因为射线贴着那个面飞。
            //
            // `f.tri` 是 `ray_intersect_raw` 的**原始**下标，`world_bvh.tris` 是
            // 重排后的数组，故先过 `slot_of_orig`（不直接用 `v.prim`：那是实例
            // 路径的 slot，与参考路径命中的面可能是另一个）。
            let n = world_bvh.tris[slot_of_orig[f.tri as usize] as usize].normal();
            let cos = n.dot(&rd).abs();
            if rel > worst_rel {
                worst_rel = rel;
                worst_cos = cos;
                worst_abs = (v.t - f.t).abs();
            }
            let is_grazing = cos < 0.4;
            if is_grazing {
                grazing += 1;
            }
            if rel > 1e-6 {
                over += 1;
                if is_grazing {
                    over_grazing += 1;
                }
                if f.t < 1.0 {
                    over_small_t += 1;
                }
            }
        }
        assert!(hit > 10_000, "命中数太少（{hit}）——判据没测到东西");
        println!(
            "[W3 非identity] 命中 {hit} 条（t ∈ [{min_t:.3}, …]）：相对 t 误差 >1e-6 者 {over} 条；\
             其中掠射 |cos|<0.4 的 {over_grazing}/{grazing} 条（{}‱）、非掠射 {} 条；\
             最差相对 {worst_rel:e}（|Δ|={worst_abs:e}，|cos|={worst_cos}）",
            over_grazing * 10_000 / grazing.max(1),
            over - over_grazing
        );
        // ---- 主判据：**非掠射（|cos| ≥ 0.4）一条都不许超 1e-6** ----
        assert_eq!(
            over - over_grazing,
            0,
            "非掠射命中的相对 t 误差必须全部 ≤ 1e-6（超标 {} 条，t<1 的 {} 条，min_t={min_t}）",
            over - over_grazing,
            over_small_t
        );
        // ---- 掠射（|cos| < 0.4）按 §17.4-B **记账**，不设阈值 ----
        //
        // §17.4-B 的原话是"95/70,299 超阈、全部 |cos| < 0.37；|cos| ≥ 0.4 零超阈"。
        // 本测试照同一口径分列：掠射区**允许**超阈（Möller–Trumbore 的行列式在
        // 掠射时接近 0，两条路径的 f32 舍入被放大——这是判据的条件数，不是遍历
        // 算错），但**超阈必须全部落在掠射区**；落一条在非掠射区就红。
        assert!(
            over == over_grazing,
            "掠射记账失效：{over} 条超阈里有 {} 条落在非掠射区",
            over - over_grazing
        );
        println!(
            "[W3 非identity] 口径：主判据 = |cos| ≥ 0.4 全部 ≤ 1e-6（实测 100% 达标）；\
             掠射区超阈 {over_grazing}/{grazing} 如实记账，最差 |cos| = {worst_cos}"
        );
    }

    /// **1 万实例：BLAS 只建一次、112 B/实例、两跑字节 md5 一致。**
    ///
    /// # "BLAS 只建一次"怎么证
    ///
    /// 原型 BVH 在本测试里只 `build` 一次（`let blas = ...` 之后就只读）；
    /// [`build_instance_set`] 的签名里根本没有三角形参数，它的产物只有
    /// `instances`（112 B × n）与 `nodes`（48 B × TLAS 节点）。故"1 万实例
    /// 各自建一份 BLAS"这件事在类型层面就**不可能发生**——本测试把这条
    /// 结构保证量化：产物字节数 = 1,120,000 + ≤(2n-1)·48，与原型三角形数无关。
    #[test]
    fn ten_thousand_instances_share_one_blas_112_bytes_each() {
        let tris = unit_cube_tris();
        let blas = prism_geometry::bvh::Bvh::build(&tris); // ← 唯一一次 BLAS 构建
        let proto_min = Vec3::splat(-1.0);
        let proto_max = Vec3::splat(1.0);

        let n = 10_000usize;
        let insts: Vec<InstanceReq> = (0..n)
            .map(|k| {
                // 100×100 网格铺设，带一点非均匀缩放（不全是 identity）。
                let (i, j) = ((k % 100) as f32, (k / 100) as f32);
                let sc = 0.4 + 0.1 * ((k % 3) as f32);
                InstanceReq::scale_translate(
                    Vec3::new(sc, sc, sc),
                    Vec3::new(i * 1.2 - 60.0, j * 1.2 - 60.0, 0.0),
                    INSTANCE_GEOM_SHARED,
                    INSTANCE_MAT_FROM_GEOM,
                )
                .expect("均匀缩放必然可逆")
            })
            .collect();

        let set = build_instance_set(&insts, &[(proto_min, proto_max)], &[0]);
        assert_eq!(set.instances.len(), n, "实例一条不许丢");
        let ibytes = encode_instances(&set.instances);
        assert_eq!(ibytes.len(), n * INSTANCE_BYTES, "必须恰好 112 B/实例");
        assert_eq!(ibytes.len(), 1_120_000);
        let node_count = set.nodes.len() / (NODE_SLOTS as usize * 4);
        assert!(node_count > 0 && node_count < 2 * n, "TLAS 节点数超 2n-1");
        // 产物里**不含**三角形数据：与原型三角形数无关。
        assert_eq!(
            set.nodes.len() % (NODE_SLOTS as usize * 4),
            0,
            "节点流必须是 48 B 的整数倍"
        );
        println!(
            "[W3 1万实例] 实例 {n} 条 = {} B（112 B/条）；TLAS {node_count} 节点 = {} B；\
             原型 BLAS {} 三角形（构建 1 次）",
            ibytes.len(),
            set.nodes.len(),
            blas.tris.len()
        );

        // 两跑一致：同输入 ⇒ 同字节（含 TLAS 的排序 tie-break）。
        let set2 = build_instance_set(&insts, &[(proto_min, proto_max)], &[0]);
        assert_eq!(set2.nodes, set.nodes, "两跑 TLAS 字节必须一致");
        assert_eq!(
            encode_instances(&set2.instances),
            ibytes,
            "两跑实例字节必须一致"
        );
        let h1 = md5_hex(&ibytes);
        let h2 = md5_hex(&encode_instances(&set2.instances));
        assert_eq!(h1, h2, "两跑 md5 必须一致");
        println!("[W3 1万实例] 两跑 md5 = {h1}");
    }

    /// MD5（RFC 1321；与仓库其它单测同款内联实现，避免引依赖）。
    ///
    /// `#[allow(clippy::needless_range_loop)]`：主循环的 `i` 不只用来索引 `S`，
    /// 还参与 `i / 16`、`(5i+1) % 16` 这些位移计算，改成迭代器会把算法揉碎。
    #[allow(clippy::needless_range_loop)]
    fn md5_hex(data: &[u8]) -> String {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, //
            5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, //
            4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, //
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let k = |i: usize| -> u32 { (((1u64 << 32) as f64) * (i as f64 + 1.0).sin().abs()) as u32 };
        let mut msg: Vec<u8> = Vec::with_capacity(data.len() + 72);
        msg.extend_from_slice(data);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
        let (mut a0, mut b0, mut c0, mut d0) =
            (0x6745_2301u32, 0xefcd_ab89u32, 0x98ba_dcfeu32, 0x1032_5476u32);
        for chunk in msg.chunks_exact(64) {
            let mut m = [0u32; 16];
            for (i, word) in m.iter_mut().enumerate() {
                *word = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64usize {
                let (f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let tmp = d;
                d = c;
                c = b;
                let sum = a.wrapping_add(f).wrapping_add(k(i)).wrapping_add(m[g]);
                b = b.wrapping_add(sum.rotate_left(S[i]));
                a = tmp;
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        let mut out = String::with_capacity(32);
        for word in [a0, b0, c0, d0] {
            for byte in word.to_le_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// CPU 侧 TLAS 遍历（`CpuRefBackend` 的判据基准；与 WGSL 同构）
// ---------------------------------------------------------------------------

/// 扁平 TLAS 节点（`NODE_SLOTS` 槽解码后的形态）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlasNode {
    /// 包围盒下界。
    pub bmin: Vec3,
    /// 包围盒上界。
    pub bmax: Vec3,
    /// 叶子：实例区间起点；内部：0。
    pub start: u32,
    /// 叶子：实例数；内部：0。
    pub count: u32,
    /// 内部：右子节点下标（左子恒为 `本节点 + 1`）。
    pub right: u32,
}

/// 把 `instance_nodes` 的字节流解码成节点数组（**与 WGSL 读同一份字节**）。
pub fn decode_nodes(bytes: &[u8]) -> Vec<TlasNode> {
    let stride = NODE_SLOTS as usize * 4;
    if bytes.is_empty() || !bytes.len().is_multiple_of(stride) {
        return Vec::new();
    }
    let rd = |b: &[u8], slot: usize| {
        f32::from_le_bytes([b[slot * 4], b[slot * 4 + 1], b[slot * 4 + 2], b[slot * 4 + 3]])
    };
    let ru = |b: &[u8], slot: usize| {
        u32::from_le_bytes([b[slot * 4], b[slot * 4 + 1], b[slot * 4 + 2], b[slot * 4 + 3]])
    };
    bytes
        .chunks_exact(stride)
        .map(|b| TlasNode {
            bmin: Vec3::new(rd(b, 0), rd(b, 1), rd(b, 2)),
            bmax: Vec3::new(rd(b, 4), rd(b, 5), rd(b, 6)),
            start: ru(b, 8),
            count: ru(b, 9),
            right: ru(b, 10),
        })
        .collect()
}

/// 把 `instances` 的字节流解码成记录（**与 shader 读同一份字节**）。
pub fn decode_instances(bytes: &[u8]) -> Vec<InstanceReq> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(INSTANCE_BYTES) {
        return Vec::new();
    }
    let rf = |b: &[u8], slot: usize| {
        f32::from_le_bytes([b[slot * 4], b[slot * 4 + 1], b[slot * 4 + 2], b[slot * 4 + 3]])
    };
    let ru = |b: &[u8], slot: usize| {
        u32::from_le_bytes([b[slot * 4], b[slot * 4 + 1], b[slot * 4 + 2], b[slot * 4 + 3]])
    };
    bytes
        .chunks_exact(INSTANCE_BYTES)
        .map(|b| {
            let mut affine = [0.0f32; INSTANCE_AFFINE_WORDS];
            for (k, v) in affine.iter_mut().enumerate() {
                *v = rf(b, k);
            }
            let mut inverse = [0.0f32; INSTANCE_INVERSE_WORDS];
            for (k, v) in inverse.iter_mut().enumerate() {
                *v = rf(b, INSTANCE_INVERSE_OFFSET + k);
            }
            InstanceReq {
                affine,
                inverse,
                geom_id: ru(b, INSTANCE_GEOM_OFFSET),
                mat_id: ru(b, INSTANCE_MAT_OFFSET),
                flags: ru(b, INSTANCE_FLAGS_OFFSET),
            }
        })
        .collect()
}

/// 实例命中（世界量）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceHit {
    /// 世界 `t`（与局部查询**同一个参数**，见 [`InstanceReq::local_ray`]）。
    pub t: f32,
    /// 实例下标（TLAS 重排后）。
    pub inst: u32,
    /// BLAS 三角形槽。
    pub prim: u32,
    /// 世界法线（已归一化、已翻到来光侧）。
    pub normal: Vec3,
}

/// 世界法线：`identity` 走与展平版**同一条**归一化 + 翻面；非 identity 走
/// `det(A)·A⁻ᵀ·n` 再归一化 + 翻面。
///
/// `tri_normal`（展平版）的口径抄在这里：`n/|n|`，`dot(n, rd) > 0 ⇒ -n`。
pub fn instance_world_normal(
    inst: &InstanceReq,
    tri: &prism_geometry::bvh::Triangle,
    rd: Vec3,
) -> Vec3 {
    let mut n = if inst.is_identity() {
        tri.normal_unnormalized()
    } else {
        inst.inv_transpose_normal(tri.normal_unnormalized())
    };
    let len2 = n.dot(&n);
    if len2 > 1e-24 {
        n = n * (1.0 / len2.sqrt());
    } else {
        return Vec3::splat(0.0);
    }
    if n.dot(&rd) > 0.0 {
        n = -n;
    }
    n
}

/// TLAS 遍历（最近命中）——与 `scene_frame_v3.wgsl::instance_nearest` **同构**。
///
/// 逐条对应的点：栈式、近子先处理、进入 t 相等先左、叶子内按下标升序、
/// 命中以严格 `<` 择优（`(t, 下标)` 的字典序在这里退化为"先到先得"，
/// 因为严格 `<` 已排除等值覆盖；等值情形两版都取**首次遇到**的那条）。
pub fn instance_nearest(
    nodes: &[TlasNode],
    instances: &[InstanceReq],
    blas: &[prism_geometry::bvh::Bvh],
    ro: Vec3,
    rd: Vec3,
    t_min: f32,
    t_max: f32,
) -> Option<InstanceHit> {
    if nodes.is_empty() || instances.is_empty() || blas.iter().all(|b| b.tris.is_empty()) {
        return None;
    }
    let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
    let mut best: Option<InstanceHit> = None;
    let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
    stack.push((0, t_min));
    while let Some((id, t_lo)) = stack.pop() {
        if let Some(b) = best {
            if t_lo > b.t {
                continue;
            }
        }
        let node = match nodes.get(id as usize) {
            Some(n) => *n,
            None => continue,
        };
        if node.count > 0 {
            let cur = best.map_or(t_max, |b| b.t);
            for k in 0..node.count {
                let slot = node.start + k;
                let inst = match instances.get(slot as usize) {
                    Some(i) => *i,
                    None => continue,
                };
                let (ro_l, rd_l) = inst.local_ray(ro, rd);
                // G-8 多 BLAS：查**该实例自己那个原型**的树（`geom_id` 下标）。
                // 越界的 `geom_id` ⇒ 没有几何可交（等价于空实例，不 panic）。
                let Some(one) = blas.get(inst.geom_id as usize) else {
                    continue;
                };
                // **slot 口径**：三后端上传的三角形缓冲就是 `blas.tris`（重排后），
                // 原始数组在 GPU 侧不存在。用 `ray_intersect_raw` 的原始下标去索引
                // `blas.tris` 会取到另一个三角形（t 对、法线错）——见
                // `Bvh::ray_intersect_slot` 的文档。
                let h = match one.ray_intersect_slot(ro_l, rd_l, t_min, cur) {
                    Some(h) => h,
                    None => continue,
                };
                if h.t < cur {
                    let tri = &one.tris[h.tri as usize];
                    best = Some(InstanceHit {
                        t: h.t,
                        inst: slot,
                        prim: h.tri,
                        normal: instance_world_normal(&inst, tri, rd),
                    });
                }
            }
        } else {
            let l = id + 1;
            let r = node.right;
            let hit_of = |n: u32| -> Option<(f32, f32)> {
                nodes
                    .get(n as usize)
                    .map(|nd| aabb_ray(nd.bmin, nd.bmax, ro, inv, t_min, t_max))
            };
            let hl = hit_of(l);
            let hr = hit_of(r);
            match (hl, hr) {
                (Some((tl, xl)), Some((tr, xr))) => {
                    if tl <= tr {
                        stack.push((r, tr.min(xr)));
                        stack.push((l, tl.min(xl)));
                    } else {
                        stack.push((l, tl.min(xl)));
                        stack.push((r, tr.min(xr)));
                    }
                }
                (Some((tl, xl)), None) => stack.push((l, tl.min(xl))),
                (None, Some((tr, xr))) => stack.push((r, tr.min(xr))),
                (None, None) => {}
            }
        }
    }
    best
}

/// slab 盒交（与 `scene_frame_v3.wgsl::aabb_ray` 同式：exit 侧 `+1e-7` 外扩）。
fn aabb_ray(bmin: Vec3, bmax: Vec3, ro: Vec3, inv: Vec3, t_min: f32, t_max: f32) -> (f32, f32) {
    let t1 = Vec3::new(
        (bmin.x - ro.x) * inv.x,
        (bmin.y - ro.y) * inv.y,
        (bmin.z - ro.z) * inv.z,
    );
    let t2 = Vec3::new(
        (bmax.x - ro.x) * inv.x,
        (bmax.y - ro.y) * inv.y,
        (bmax.z - ro.z) * inv.z,
    );
    let lo = Vec3::new(t1.x.min(t2.x), t1.y.min(t2.y), t1.z.min(t2.z));
    let hi = Vec3::new(t1.x.max(t2.x), t1.y.max(t2.y), t1.z.max(t2.z));
    let enter = lo.x.max(lo.y).max(lo.z).max(t_min);
    let exitv = hi.x.min(hi.y).min(hi.z).min(t_max);
    if enter <= exitv + 1e-7 {
        (enter, exitv)
    } else {
        (1e30, -1e30)
    }
}
