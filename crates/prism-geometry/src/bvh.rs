//! BVH：SAH 构建 + 射线求交 + 最近点 + 广义 winding number（S10）。
//!
//! # 确定性（超越点）
//!
//! - 构建：自顶向下 SAH，每轴 12 桶；代价并列时按"先轴后桶"遍历序取首最小（同输入永远同树）；
//!   桶划分退化时按质心全序（`total_cmp` + 图元索引）稳定排序取中位，保证终止且确定。
//! - 遍历：近子节点先处理的显式栈；命中以 `(t, 图元索引)` 字典序取最优（t 并列按图元索引 tie-break）。
//! - 求交全程不用 fma 内建（浮点融合乘加是跨平台逐位一致的隐形杀手）。
//! - winding 链的超越函数（atan2/asin/sin/cos）一律走 [`crate::det_math`]
//!   确定性件（D-29：纯 IEEE 运算，禁 libm，同输入逐位一致）。
//!
//! # 一树两用
//!
//! 每个节点在构建时增存实体角聚合（面积加权中心 C、法线矩 N、法线锥），
//! winding number 查询对"远处且法线一致"的节点簇用一阶近似（O(log n)），
//! 近处簇下降到叶子用 van Oosterom–Strackee 闭式精确求和——mesh→SDF 烘焙的符号判定直接复用本树。

use crate::det_math::{det_asin, det_atan2, det_cos_sin};
use crate::{vmax, vmin};
use prism_math::Vec3;

/// SAH 分桶数（12 桶）。
const SAH_BUCKETS: usize = 12;
/// 叶子图元上限（叶子 4–8 图元）。
const LEAF_MAX: usize = 8;

/// 轴对齐包围盒。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// 下角。
    pub min: Vec3,
    /// 上角。
    pub max: Vec3,
}

impl Aabb {
    /// 空盒（grow 进第一个点后才有意义）。
    #[inline]
    pub fn empty() -> Self {
        Aabb { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }

    /// 并入一个点。
    #[inline]
    pub fn grow_point(&mut self, p: Vec3) {
        self.min = vmin(self.min, p);
        self.max = vmax(self.max, p);
    }

    /// 并入另一个盒。
    #[inline]
    pub fn grow(&mut self, o: &Aabb) {
        self.min = vmin(self.min, o.min);
        self.max = vmax(self.max, o.max);
    }

    /// 盒中心。
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// 半尺寸。
    #[inline]
    pub fn half_size(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// 包围球半径（对角线之半；供 winding 近似判定）。
    #[inline]
    pub fn radius(&self) -> f32 {
        self.half_size().length()
    }

    /// 半表面积（SAH 代价用，比例语义与全表面积一致）。
    #[inline]
    pub fn half_area(&self) -> f32 {
        let h = self.half_size();
        h.x * h.y + h.y * h.z + h.z * h.x
    }

    /// 点到盒的欧氏距离（盒内为 0）。
    #[inline]
    pub fn dist_to(&self, p: Vec3) -> f32 {
        let d = vmax(vmax(self.min - p, p - self.max), Vec3::splat(0.0));
        d.length()
    }

    /// 射线与盒的区间交（slab 法；`inv` 为 1/rd 分量）。
    ///
    ///  附注：`exit` 侧微量外扩（+1e-7）消除浮点边界漏交。
    ///
    /// # 零方向分量（平行轴）
    /// `rd` 某分量为 0 时 `inv` 为 ±inf，而 `(min−o)·inf` 在 `min == o` 处
    /// 得 0·inf = **NaN** → 整盒比较全假 → 轴对齐射线沿盒边界误判漏交
    /// （轴对齐几何 + 轴对齐相机射线的常见组合会成片漏）。故平行轴**显式**
    /// 判掉：起点在该轴 slab 之外 ⇒ 必然不命中（区间取空 (inf, −inf)），
    /// slab 之内 ⇒ 该轴不构成约束（区间取 (−inf, inf)）。非平行轴仍走原
    /// slab 算术（逐位不变）。
    #[inline]
    pub fn ray(&self, ro: Vec3, inv: Vec3, t_min: f32, t_max: f32) -> Option<(f32, f32)> {
        let a = self.min - ro;
        let b = self.max - ro;
        // 逐分量乘（Vec3 只定义了 Mul<f32>，保持与 prism-math API 一致）。
        let (lo_x, hi_x) = slab_interval(a.x, b.x, inv.x);
        let (lo_y, hi_y) = slab_interval(a.y, b.y, inv.y);
        let (lo_z, hi_z) = slab_interval(a.z, b.z, inv.z);
        let enter = lo_x.max(lo_y).max(lo_z).max(t_min);
        let exit = hi_x.min(hi_y).min(hi_z).min(t_max);
        if enter <= exit + 1e-7 {
            Some((enter, exit))
        } else {
            None
        }
    }
}

/// 单轴 slab 区间（平行轴显式处理，见 [`Aabb::ray`]）。
#[inline]
fn slab_interval(lo: f32, hi: f32, inv: f32) -> (f32, f32) {
    if inv.is_infinite() {
        // 平行轴：断开 0·inf = NaN 陷阱。
        if lo > 0.0 || hi < 0.0 {
            (f32::INFINITY, f32::NEG_INFINITY) // 区间取空 ⇒ enter > exit ⇒ 不命中
        } else {
            (f32::NEG_INFINITY, f32::INFINITY) // 无约束
        }
    } else {
        let t1 = lo * inv;
        let t2 = hi * inv;
        (t1.min(t2), t1.max(t2))
    }
}

/// 三角形（BVH 的求交与距离图元）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    /// 顶点 A。
    pub a: Vec3,
    /// 顶点 B。
    pub b: Vec3,
    /// 顶点 C。
    pub c: Vec3,
}

impl Triangle {
    /// 几何法线（绕序 a→b→c 的右手方向，未归一）。
    #[inline]
    pub fn normal_unnormalized(&self) -> Vec3 {
        (self.b - self.a).cross(&(self.c - self.a))
    }

    /// 单位几何法线。
    #[inline]
    pub fn normal(&self) -> Vec3 {
        self.normal_unnormalized().normalize()
    }

    /// 面积（= 未归一法线长之半）。
    #[inline]
    pub fn area(&self) -> f32 {
        self.normal_unnormalized().length() * 0.5
    }

    /// 质心。
    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.a + self.b + self.c) * (1.0 / 3.0)
    }

    /// 点到三角形的最近点（Ericson RTCD 闭式，区域判定）。
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        let ap = p - self.a;
        let d1 = ab.dot(&ap);
        let d2 = ac.dot(&ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return self.a;
        }
        let bp = p - self.b;
        let d3 = ab.dot(&bp);
        let d4 = ac.dot(&bp);
        if d3 >= 0.0 && d4 <= d3 {
            return self.b;
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return self.a + ab * v;
        }
        let cp = p - self.c;
        let d5 = ab.dot(&cp);
        let d6 = ac.dot(&cp);
        if d6 >= 0.0 && d5 <= d6 {
            return self.c;
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return self.a + ac * w;
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return self.b + (self.c - self.b) * w;
        }
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        self.a + ab * v + ac * w
    }
}

/// 顶点流 + 索引流 → 三角形序列（glTF/OBJ 导入与 BVH 的适配层）。
pub fn mesh_triangles(positions: &[Vec3], indices: &[u32]) -> Vec<Triangle> {
    indices
        .chunks_exact(3)
        .map(|c| Triangle {
            a: positions[c[0] as usize],
            b: positions[c[1] as usize],
            c: positions[c[2] as usize],
        })
        .collect()
}

/// BVH 节点（扁平数组存储；左子恒为 `self + 1`）。
#[derive(Debug, Clone, PartialEq)]
pub struct BvhNode {
    /// 子树包围盒。
    pub bounds: Aabb,
    /// 叶子：重排图元区间起点；内部节点为 0。
    pub start: u32,
    /// 叶子：图元数；内部节点为 0。
    pub count: u32,
    /// 内部节点：右子索引；叶子为 0。
    pub right_child: u32,
    // —— winding number 聚合（每节点增存实体角聚合，一树两用）——
    /// 面积加权中心 C = Σ Aᵢcᵢ / ΣAᵢ。
    pub solid_center: Vec3,
    /// 法线矩 N = Σ Aᵢn̂ᵢ。
    pub solid_normal: Vec3,
    /// 面积和 ΣAᵢ。
    pub solid_area: f32,
    /// 法线锥轴（N 归一；退化时为零向量）。
    pub cone_axis: Vec3,
    /// 法线锥半角余弦（全向退化时为 −1，近似判定永不通过）。
    pub cone_cos: f32,
}

/// 射线命中。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// 命中射线参数 t。
    pub t: f32,
    /// 命中图元的原始索引（构建输入数组中的下标）。
    pub tri: u32,
}

/// 最近点命中。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestHit {
    /// 最近距离。
    pub dist: f32,
    /// 最近图元原始索引。
    pub tri: u32,
    /// 最近点坐标。
    pub point: Vec3,
}

/// SAH-BVH。
#[derive(Debug, Clone, PartialEq)]
pub struct Bvh {
    /// 扁平节点数组（根为 0）。
    pub nodes: Vec<BvhNode>,
    /// 重排后的三角形（叶子区间引用）。
    pub tris: Vec<Triangle>,
    /// 重排映射：重排下标 → 原始图元索引。
    pub order: Vec<u32>,
    /// 全场景包围盒。
    pub bounds: Aabb,
}

impl Bvh {
    /// SAH-BVH 构建（自顶向下，12 桶；空输入产生单空叶）。
    pub fn build(tris: &[Triangle]) -> Bvh {
        let tri_bounds: Vec<Aabb> = tris
            .iter()
            .map(|t| {
                let mut b = Aabb::empty();
                b.grow_point(t.a);
                b.grow_point(t.b);
                b.grow_point(t.c);
                b
            })
            .collect();
        let centroids: Vec<Vec3> = tris.iter().map(|t| t.centroid()).collect();
        let mut prims: Vec<u32> = (0..tris.len() as u32).collect();
        let mut bvh = Bvh {
            nodes: Vec::new(),
            tris: Vec::with_capacity(tris.len()),
            order: Vec::with_capacity(tris.len()),
            bounds: Aabb::empty(),
        };
        for b in &tri_bounds {
            bvh.bounds.grow(b);
        }
        if tris.is_empty() {
            return bvh; // 空输入 → 空树（遍历/查询均安全返回空语义）
        }
        bvh.build_node(tris, &centroids, &tri_bounds, &mut prims);
        bvh
    }

    /// 递归构建一个节点，返回其索引。
    fn build_node(
        &mut self,
        tris: &[Triangle],
        centroids: &[Vec3],
        tri_bounds: &[Aabb],
        prims: &mut [u32],
    ) -> u32 {
        debug_assert!(!prims.is_empty(), "build_node requires non-empty prim range");
        let id = self.nodes.len() as u32;
        self.nodes.push(BvhNode {
            bounds: Aabb::empty(),
            start: 0,
            count: 0,
            right_child: 0,
            solid_center: Vec3::splat(0.0),
            solid_normal: Vec3::splat(0.0),
            solid_area: 0.0,
            cone_axis: Vec3::splat(0.0),
            cone_cos: -1.0,
        });

        // 区间包围盒与 winding 聚合（逐图元直接累加——每节点独立，同输入同序同结果）。
        let mut bounds = Aabb::empty();
        let (mut area_sum, mut n_sum, mut ac_sum) = (0.0_f32, Vec3::splat(0.0), Vec3::splat(0.0));
        for &pi in prims.iter() {
            let i = pi as usize;
            bounds.grow(&tri_bounds[i]);
            let t = &tris[i];
            let a = t.area();
            area_sum += a;
            n_sum = n_sum + t.normal() * a;
            ac_sum = ac_sum + t.centroid() * a;
        }
        let (solid_center, cone_axis, cone_cos) = if area_sum > 1e-20 {
            let c = ac_sum * (1.0 / area_sum);
            let n_len = n_sum.length();
            if n_len > 1e-9 {
                // 法线锥：轴 = N 归一，半角 = 各三角形法线与轴的最大夹角。
                let axis = n_sum * (1.0 / n_len);
                let mut max_dot = 1.0_f32;
                for &pi in prims.iter() {
                    let nd = tris[pi as usize].normal().dot(&axis);
                    if nd < max_dot {
                        max_dot = nd;
                    }
                }
                (c, axis, max_dot)
            } else {
                // 法线矩退化（正反混合 soup）：锥全向，近似永不通过。
                (c, Vec3::splat(0.0), -1.0)
            }
        } else {
            (Vec3::splat(0.0), Vec3::splat(0.0), -1.0)
        };

        if prims.len() <= LEAF_MAX {
            let start = self.tris.len() as u32;
            for &pi in prims.iter() {
                self.tris.push(tris[pi as usize]);
                self.order.push(pi);
            }
            let node = &mut self.nodes[id as usize];
            node.bounds = bounds;
            node.start = start;
            node.count = prims.len() as u32;
            node.solid_center = solid_center;
            node.solid_normal = n_sum;
            node.solid_area = area_sum;
            node.cone_axis = cone_axis;
            node.cone_cos = cone_cos;
            return id;
        }

        // SAH 分割（并列取遍历序首最小 → 确定性）。
        let split = best_sah_split(centroids, tri_bounds, prims);
        let (axis, plane) = match split {
            Some(s) => s,
            None => {
                // 全轴退化：按质心稳定排序取中位（tie 按图元索引）。
                let axis = widest_axis(centroids, prims);
                median_partition(centroids, prims, axis);
                let mid = prims.len() / 2;
                return self.build_children(tris, centroids, tri_bounds, prims, id, bounds, solid_center, n_sum, area_sum, cone_axis, cone_cos, mid);
            }
        };
        // 稳定划分：保留原相对顺序（同输入同划分）。
        let mut left: Vec<u32> = Vec::with_capacity(prims.len());
        let mut right: Vec<u32> = Vec::with_capacity(prims.len());
        for &pi in prims.iter() {
            if centroids[pi as usize].at_axis(axis) < plane {
                left.push(pi);
            } else {
                right.push(pi);
            }
        }
        let mid = if left.is_empty() || right.is_empty() {
            // 划分失败（数值退化）→ 中位数兜底，保证终止。
            median_partition(centroids, prims, axis);
            prims.len() / 2
        } else {
            let mid = left.len();
            for (k, pi) in left.into_iter().chain(right).enumerate() {
                prims[k] = pi;
            }
            mid
        };
        // 叶子下限（叶子 4–8）：一侧过小时改用中位划分，两侧均 ≥ 4。
        let mid = if mid < 4 || prims.len() - mid < 4 {
            median_partition(centroids, prims, axis);
            prims.len() / 2
        } else {
            mid
        };
        self.build_children(tris, centroids, tri_bounds, prims, id, bounds, solid_center, n_sum, area_sum, cone_axis, cone_cos, mid)
    }

    /// 划分后递归构建两个子节点并填写内部节点。
    #[allow(clippy::too_many_arguments)]
    fn build_children(
        &mut self,
        tris: &[Triangle],
        centroids: &[Vec3],
        tri_bounds: &[Aabb],
        prims: &mut [u32],
        id: u32,
        bounds: Aabb,
        solid_center: Vec3,
        n_sum: Vec3,
        area_sum: f32,
        cone_axis: Vec3,
        cone_cos: f32,
        mid: usize,
    ) -> u32 {
        let (left_prims, right_prims) = prims.split_at_mut(mid);
        let _l = self.build_node(tris, centroids, tri_bounds, left_prims);
        let r = self.build_node(tris, centroids, tri_bounds, right_prims);
        let node = &mut self.nodes[id as usize];
        node.bounds = bounds;
        node.right_child = r;
        node.solid_center = solid_center;
        node.solid_normal = n_sum;
        node.solid_area = area_sum;
        node.cone_axis = cone_axis;
        node.cone_cos = cone_cos;
        id
    }

    /// Möller–Trumbore 三角形命中（图元内命中；t 并列按图元索引 tie-break）。
    ///
    /// `rd` 须为单位向量；非单位方向（几何实例化把射线变换进局部空间后
    /// **故意不归一化**，见 [`Bvh::ray_intersect_raw`]）请走另一个入口。
    pub fn ray_intersect(&self, ro: Vec3, rd: Vec3, t_min: f32, t_max: f32) -> Option<RayHit> {
        debug_assert!((rd.length() - 1.0).abs() < 1e-4, "ray_intersect requires unit rd");
        self.ray_intersect_raw(ro, rd, t_min, t_max)
    }

    /// 同 [`Bvh::ray_intersect`]，但**不要求 `rd` 单位长度**。
    ///
    /// # 为什么需要这个入口（W3 实例化的核心手法）
    ///
    /// 实例化把世界射线变换到局部空间：`o_l = M⁻¹o`、`d_l = M⁻¹d`（线性块）。
    /// 因为两者是**同一次仿射变换**作用在 `o` 与 `d` 上，局部点
    /// `o_l + t·d_l ≡ M⁻¹(o + t·d)` —— **`t` 参数不变**。于是：
    ///
    /// * 世界空间的 `T_MIN` / `t_max` 可以直接当局部查询的界用，**不需要按
    ///   缩放换算**（换算只有在归一化局部方向、让 `t_l = t·|M⁻¹d|` 时才出现）；
    /// * `identity` 实例的 `o_l`/`d_l` 与世界射线逐位相同（`x·1.0 + 0.0` 精确），
    ///   故 identity 路径与展平路径**逐位一致**有构造性保证，而不是"测出来像"。
    ///
    /// `Aabb::ray` 的 slab 算术与 Möller–Trumbore 都与 `|rd|` 无关，唯一被
    /// 放松的就是入口那条"归一化"断言。
    pub fn ray_intersect_raw(&self, ro: Vec3, rd: Vec3, t_min: f32, t_max: f32) -> Option<RayHit> {
        self.ray_intersect_impl(ro, rd, t_min, t_max, false)
    }

    /// 同 [`Bvh::ray_intersect_raw`]，但 `RayHit::tri` 是**重排后**（slot）下标。
    ///
    /// # 为什么要有这个入口（W3 的第二个下标体系）
    ///
    /// 本结构里有两套三角形下标：
    ///
    /// * **原始**下标（`RayHit::tri` / [`Bvh::order`] 的值）= 构建输入数组的下标。
    ///   CPU 侧持有那份数组（`curve_tess::tessellate_all` 的输出），故展平路径
    ///   用 `ray_intersect_raw` + 原始下标取顶点；
    /// * **slot** 下标（`RayHit::tri` 的这里）= [`Bvh::tris`] 的下标。三后端上传
    ///   的三角形缓冲就是 `tris` 这一份（`encode_curve_buffers` 按 `bvh.tris` 逐条
    ///   写），**原始数组在 GPU 侧根本不存在**。
    ///
    /// 实例化路径因此必须按 slot 取三角形：它上传的是 `tris`、遍历的是同一棵树。
    /// 用 `ray_intersect_raw` 的原始下标去索引 `tris` 会**取到另一个三角形**——
    /// 顶点错了、法线跟着错，而 `t` 是对的，于是"命中位置对、着色错"，最难查。
    ///
    /// 除 `tri` 的口径外，遍历、剪枝、`t` 的算术与 [`Bvh::ray_intersect_raw`]
    /// **逐句相同**（同一个 `impl`，只差一个布尔）。
    pub fn ray_intersect_slot(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_min: f32,
        t_max: f32,
    ) -> Option<RayHit> {
        self.ray_intersect_impl(ro, rd, t_min, t_max, true)
    }

    /// [`Bvh::ray_intersect_raw`] / [`Bvh::ray_intersect_slot`] 的共同实现。
    ///
    /// `by_slot == false` ⇒ `RayHit::tri` 是原始下标（tie-break 按原始索引升序）；
    /// `by_slot == true`  ⇒ 是 slot 下标（tie-break 按 slot 升序）。
    /// 两种口径都**确定性**（同一棵树 + 同一射线 ⇒ 同一结果）。
    fn ray_intersect_impl(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_min: f32,
        t_max: f32,
        by_slot: bool,
    ) -> Option<RayHit> {
        if self.tris.is_empty() {
            // 空树：安全返回空语义（bvh.rs:282 契约；与 closest_point/winding_number 对齐）。
            return None;
        }
        let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
        let mut best: Option<RayHit> = None;
        // 固定深度显式栈（深度上界由图元总数推导，64 层必够）。
        let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
        stack.push((0, t_min));
        while let Some((id, t_lo)) = stack.pop() {
            if let Some(b) = best {
                if t_lo > b.t {
                    continue;
                }
            }
            let node = &self.nodes[id as usize];
            if node.count > 0 {
                let hi = best.as_ref().map_or(t_max, |b| b.t);
                for i in 0..node.count as usize {
                    let slot = node.start as usize + i;
                    if let Some(t) = tri_ray_hit(&self.tris[slot], ro, rd, t_min, hi) {
                        let key = if by_slot {
                            slot as u32
                        } else {
                            self.order[slot]
                        };
                        let better = match best {
                            None => true,
                            Some(b) => t < b.t || (t == b.t && key < b.tri),
                        };
                        if better {
                            best = Some(RayHit { t, tri: key });
                        }
                    }
                }
            } else {
                let l = id + 1;
                let r = node.right_child;
                let hit_l = self.nodes[l as usize].bounds.ray(ro, inv, t_min, t_max);
                let hit_r = self.nodes[r as usize].bounds.ray(ro, inv, t_min, t_max);
                // 近子先处理（栈后进先出 → 远者先入栈）；进入 t 相等先左（确定性）。
                match (hit_l, hit_r) {
                    (Some((tl, exl)), Some((tr, exr))) => {
                        if tl <= tr {
                            stack.push((r, tr.min(exr)));
                            stack.push((l, tl.min(exl)));
                        } else {
                            stack.push((l, tl.min(exl)));
                            stack.push((r, tr.min(exr)));
                        }
                    }
                    (Some((tl, exl)), None) => stack.push((l, tl.min(exl))),
                    (None, Some((tr, exr))) => stack.push((r, tr.min(exr))),
                    (None, None) => {}
                }
            }
        }
        best
    }

    /// 最近点查询（BVH 剪枝；`(dist, tri)` 字典序取最优）。
    pub fn closest_point(&self, p: Vec3) -> ClosestHit {
        let mut best = ClosestHit { dist: f32::INFINITY, tri: u32::MAX, point: Vec3::splat(0.0) };
        if self.tris.is_empty() {
            return best;
        }
        let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
        stack.push((0, 0.0));
        while let Some((id, _)) = stack.pop() {
            let node = &self.nodes[id as usize];
            // 下界剪枝用严格 >：允许恰好持平的更小图元索引 tie 进入。
            if node.bounds.dist_to(p) > best.dist {
                continue;
            }
            if node.count > 0 {
                for i in 0..node.count as usize {
                    let slot = node.start as usize + i;
                    let cp = self.tris[slot].closest_point(p);
                    let d = (cp - p).length();
                    let orig = self.order[slot];
                    if d < best.dist || (d == best.dist && orig < best.tri) {
                        best = ClosestHit { dist: d, tri: orig, point: cp };
                    }
                }
            } else {
                let l = id + 1;
                let r = node.right_child;
                let dl = self.nodes[l as usize].bounds.dist_to(p);
                let dr = self.nodes[r as usize].bounds.dist_to(p);
                // 近者后入栈（先处理）；距离相等先左（推右在前，确定性）。
                if dl <= dr {
                    stack.push((r, dr));
                    stack.push((l, dl));
                } else {
                    stack.push((l, dl));
                    stack.push((r, dr));
                }
            }
        }
        best
    }

    /// 广义 winding number（内部 = ±1，外部 = 0；对开放网格鲁棒）。
    ///
    /// 远处且法线一致的节点簇用一阶近似 `Ω ≈ N·(C−p)/|C−p|³`（O(log n)），
    /// 其余下降到叶子用 van Oosterom–Strackee 闭式精确求和。
    pub fn winding_number(&self, p: Vec3) -> f32 {
        if self.tris.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            if node.count > 0 {
                for i in 0..node.count as usize {
                    sum += solid_angle(&self.tris[node.start as usize + i], p);
                }
            } else if let Some(approx) = node.first_order(p) {
                sum += approx;
            } else {
                stack.push(id + 1);
                stack.push(node.right_child);
            }
        }
        sum / (4.0 * std::f32::consts::PI)
    }
}

impl BvhNode {
    /// winding 一阶近似判定与求值（Windish 对偶锥判据）：
    /// 查询点离簇足够远（张角小）且法线锥与视线锥不相交时可用 `N·(C−p)/|C−p|³`。
    fn first_order(&self, p: Vec3) -> Option<f32> {
        let d_box = self.bounds.dist_to(p);
        let r = self.bounds.radius();
        if !(d_box > 2.0 * r) {
            return None; // 近簇：必须精确
        }
        if self.cone_cos <= -1.0 + 1e-6 {
            return None; // 法线锥退化
        }
        let cp = self.solid_center - p;
        let len = cp.length();
        if len < 1e-9 {
            return None;
        }
        let d_hat = cp * (1.0 / len);
        // 视线锥半角 β = asin(R/|C−p|)（R 为包围球半径）——det 件（D-29 禁 libm）。
        let beta = det_asin((r / len).clamp(0.0, 1.0));
        let cos_theta = self.cone_cos;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let (cos_beta, sin_beta) = det_cos_sin(beta);
        let cos_sum = cos_theta * cos_beta - sin_theta * sin_beta;
        // 锥不相交 ⇔ angle(axis, d̂) > θ + β ⇔ cos(angle) < cos(θ+β)。
        if self.cone_axis.dot(&d_hat) < cos_sum {
            Some(self.solid_normal.dot(&cp) / (len * len * len))
        } else {
            None
        }
    }
}

/// Vec3 按轴取分量（构建内部用）。
trait AtAxis {
    fn at_axis(&self, axis: usize) -> f32;
}

impl AtAxis for Vec3 {
    #[inline]
    fn at_axis(&self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

/// SAH 最佳分割（12 桶）。返回 `(轴, 分割平面位置)`；全轴退化返回 `None`。
fn best_sah_split(centroids: &[Vec3], tri_bounds: &[Aabb], prims: &[u32]) -> Option<(usize, f32)> {
    let mut node_bounds = Aabb::empty();
    for &pi in prims {
        node_bounds.grow_point(centroids[pi as usize]);
    }
    let extent = node_bounds.max - node_bounds.min;
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, bucket)
    for axis in 0..3 {
        let ext = extent.at_axis(axis);
        if !(ext > 1e-12) {
            continue;
        }
        let scale = SAH_BUCKETS as f32 / ext;
        let base = node_bounds.min.at_axis(axis);
        let mut bucket_bounds = [Aabb::empty(); SAH_BUCKETS];
        let mut bucket_count = [0_u32; SAH_BUCKETS];
        for &pi in prims {
            let c = centroids[pi as usize].at_axis(axis);
            let b = ((c - base) * scale).floor() as usize;
            let b = b.min(SAH_BUCKETS - 1);
            bucket_bounds[b].grow(&tri_bounds[pi as usize]);
            bucket_count[b] += 1;
        }
        // 前缀/后缀扫描。
        let mut left_area = [0.0_f32; SAH_BUCKETS];
        let mut left_count = [0_u32; SAH_BUCKETS];
        let mut acc = Aabb::empty();
        let mut cnt = 0_u32;
        for i in 0..SAH_BUCKETS {
            acc.grow(&bucket_bounds[i]);
            cnt += bucket_count[i];
            left_area[i] = acc.half_area();
            left_count[i] = cnt;
        }
        let mut right_area = [0.0_f32; SAH_BUCKETS];
        let mut right_count = [0_u32; SAH_BUCKETS];
        let mut acc = Aabb::empty();
        let mut cnt = 0_u32;
        for i in (0..SAH_BUCKETS).rev() {
            acc.grow(&bucket_bounds[i]);
            cnt += bucket_count[i];
            right_area[i] = acc.half_area();
            right_count[i] = cnt;
        }
        // 在桶间评估分割代价（严格 < 更新 → 首最小者胜，确定性 tie-break）。
        for i in 0..SAH_BUCKETS - 1 {
            let cl = left_count[i];
            let cr = right_count[i + 1];
            if cl == 0 || cr == 0 {
                continue;
            }
            let cost = cl as f32 * left_area[i] + cr as f32 * right_area[i + 1];
            if best.map_or(true, |(bc, _, _)| cost < bc) {
                best = Some((cost, axis, i));
            }
        }
    }
    best.map(|(_, axis, bucket)| {
        let ext = extent.at_axis(axis);
        let base = node_bounds.min.at_axis(axis);
        let plane = base + (bucket + 1) as f32 * ext / SAH_BUCKETS as f32;
        (axis, plane)
    })
}

/// 最宽轴（退化兜底用；按质心包围盒尺寸，确定性）。
fn widest_axis(centroids: &[Vec3], prims: &[u32]) -> usize {
    let mut b = Aabb::empty();
    for &pi in prims {
        b.grow_point(centroids[pi as usize]);
    }
    let h = b.half_size();
    let mut axis = 0;
    if h.y > h.at_axis(axis) {
        axis = 1;
    }
    if h.z > h.at_axis(axis) {
        axis = 2;
    }
    axis
}

/// 稳定中位划分（质心全序排序 + 图元索引 tie，保证两侧均非空）。
fn median_partition(centroids: &[Vec3], prims: &mut [u32], axis: usize) {
    prims.sort_by(|&a, &b| {
        let ca = centroids[a as usize].at_axis(axis);
        let cb = centroids[b as usize].at_axis(axis);
        ca.total_cmp(&cb).then(a.cmp(&b))
    });
}

/// Möller–Trumbore：射线-三角形交（不使用 fma， 附注）。
fn tri_ray_hit(t: &Triangle, ro: Vec3, rd: Vec3, t_min: f32, t_max: f32) -> Option<f32> {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let pvec = rd.cross(&e2);
    let det = e1.dot(&pvec);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ro - t.a;
    let u = tvec.dot(&pvec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let qvec = tvec.cross(&e1);
    let v = rd.dot(&qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let hit = e2.dot(&qvec) * inv_det;
    if hit >= t_min && hit <= t_max {
        Some(hit)
    } else {
        None
    }
}

/// 单三角形对查询点的实体角（van Oosterom–Strackee 闭式；符号随绕序；
/// atan2 走 [`crate::det_math::det_atan2`] 确定性件，D-29 禁 libm）。
pub fn solid_angle(t: &Triangle, p: Vec3) -> f32 {
    let v1 = t.a - p;
    let v2 = t.b - p;
    let v3 = t.c - p;
    let (a, b, c) = (v1.length(), v2.length(), v3.length());
    let num = v1.dot(&v2.cross(&v3));
    let den = a * b * c + a * v2.dot(&v3) + b * v3.dot(&v1) + c * v1.dot(&v2);
    2.0 * det_atan2(num, den)
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;

    /// 单位立方体 [-1,1]³ 的 12 个三角形（外法线绕序，烘焙与 winding 测试共用）。
    pub(crate) fn unit_cube() -> Vec<Triangle> {
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
        let f: [[usize; 3]; 12] = [
            [0, 2, 1],
            [0, 3, 2], // z = -1
            [4, 5, 6],
            [4, 6, 7], // z = +1
            [0, 4, 7],
            [0, 7, 3], // x = -1
            [1, 2, 6],
            [1, 6, 5], // x = +1
            [0, 1, 5],
            [0, 5, 4], // y = -1
            [2, 3, 7],
            [2, 7, 6], // y = +1
        ];
        f.iter()
            .map(|t| Triangle { a: v[t[0]], b: v[t[1]], c: v[t[2]] })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_noise::pcg;

    use super::testutil::unit_cube;

    const SEED: u64 = 0xB7_00_00_01;

    /// 确定性伪随机三角形场景（100 个，尺度 ~[-3,3]³）。
    fn random_scene(n: usize) -> Vec<Triangle> {
        (0..n)
            .map(|i| {
                let p = |k: u64| {
                    Vec3::new(
                        pcg::hash2_f32(SEED, i as u64, k) * 6.0 - 3.0,
                        pcg::hash2_f32(SEED, i as u64, k + 100) * 6.0 - 3.0,
                        pcg::hash2_f32(SEED, i as u64, k + 200) * 6.0 - 3.0,
                    )
                };
                Triangle { a: p(0), b: p(1), c: p(2) }
            })
            .collect()
    }

    /// 关卡 3a：SAH 构建确定性——同输入两遍，节点数组与重排映射逐位一致。
    #[test]
    fn sah_build_is_deterministic() {
        let tris = random_scene(100);
        let a = Bvh::build(&tris);
        let b = Bvh::build(&tris);
        assert_eq!(a.nodes, b.nodes, "同输入必须同树（节点数组逐位一致）");
        assert_eq!(a.order, b.order, "重排映射一致");
        assert_eq!(a.tris, b.tris, "重排三角形一致");
        // 结构合理性：叶子 ≤ 8、非叶必有两子、根覆盖全部图元。
        let leaves: Vec<&BvhNode> = a.nodes.iter().filter(|n| n.count > 0).collect();
        assert!(leaves.iter().all(|n| n.count as usize <= LEAF_MAX));
        let total: u32 = leaves.iter().map(|n| n.count).sum();
        assert_eq!(total, 100, "叶子图元数守恒");
        assert!(a.nodes.len() >= leaves.len() * 2 - 1, "内部节点数下限");
        // 全部三角形出现在叶子里（集合守恒）。
        let mut seen = vec![false; 100];
        for n in &leaves {
            for i in 0..n.count as usize {
                seen[a.order[n.start as usize + i] as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "每个图元恰好出现在一个叶子");
    }

    /// 关卡 3b：BVH 遍历 vs 暴力 O(n) 全查一致（射线求交，200 射线）。
    #[test]
    fn ray_intersect_matches_bruteforce() {
        let tris = random_scene(100);
        let bvh = Bvh::build(&tris);
        for r in 0..200_u64 {
            let ro = Vec3::new(
                pcg::hash2_f32(SEED, r, 0) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r, 1) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r, 2) * 8.0 - 4.0,
            );
            let rd = Vec3::new(
                pcg::hash2_f32(SEED, r, 3) - 0.5,
                pcg::hash2_f32(SEED, r, 4) - 0.5,
                pcg::hash2_f32(SEED, r, 5) - 0.5,
            )
            .normalize();
            // 暴力：同 tie-break 规则 (t, tri)。
            let mut want: Option<RayHit> = None;
            for (i, t) in tris.iter().enumerate() {
                if let Some(hit) = tri_ray_hit(t, ro, rd, 0.0, f32::INFINITY) {
                    let better = match want {
                        None => true,
                        Some(b) => hit < b.t || (hit == b.t && (i as u32) < b.tri),
                    };
                    if better {
                        want = Some(RayHit { t: hit, tri: i as u32 });
                    }
                }
            }
            let got = bvh.ray_intersect(ro, rd, 0.0, f32::INFINITY);
            assert_eq!(got, want, "射线 {r} 求交与暴力不一致");
        }
    }

    /// 关卡 3b-bis：两套 `tri` 口径必须指向**同一个三角形**、给**同一个 `t`**。
    ///
    /// `ray_intersect_raw` 的 `tri` 是构建输入数组的下标，`ray_intersect_slot` 的
    /// 是 [`Bvh::tris`] 的下标，二者只差一个 [`Bvh::order`] 映射。这条判据防的是
    /// W3 里踩过的那个坑：**拿原始下标去索引 `tris`**——`t` 对、顶点与法线全错，
    /// 表现为"命中位置对、着色错"。
    #[test]
    fn ray_intersect_slot_and_raw_agree_on_t_and_resolve_to_same_triangle() {
        let tris = random_scene(64);
        let bvh = Bvh::build(&tris);
        // 先确认这棵树**真的重排了**（否则本判据退化成"比 0..n 恒等"）。
        assert!(
            bvh.order.iter().enumerate().any(|(i, &o)| i as u32 != o),
            "SAH 必须真的重排过，否则本条判据没有鉴别力"
        );
        let mut hits = 0u32;
        for r in 0..300_u64 {
            let ro = Vec3::new(
                pcg::hash2_f32(SEED, r + 31, 0) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r + 31, 1) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r + 31, 2) * 8.0 - 4.0,
            );
            let rd = Vec3::new(
                pcg::hash2_f32(SEED, r + 31, 3) - 0.5,
                pcg::hash2_f32(SEED, r + 31, 4) - 0.5,
                pcg::hash2_f32(SEED, r + 31, 5) - 0.5,
            )
            .normalize();
            let raw = bvh.ray_intersect_raw(ro, rd, 1e-4, 1e4);
            let slot = bvh.ray_intersect_slot(ro, rd, 1e-4, 1e4);
            match (raw, slot) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    hits += 1;
                    assert_eq!(a.t.to_bits(), b.t.to_bits(), "两套口径的 `t` 必须逐位相同");
                    // 口径换算后必须是同一个三角形：`order[slot] == 原始下标`。
                    let orig_of_slot = bvh.order[b.tri as usize];
                    assert_eq!(
                        orig_of_slot, a.tri,
                        "`order[slot]` 必须等于原始下标（否则顶点/法线取错三角形）"
                    );
                    // 再验一层：两个下标取到的三角形**几何相同**。
                    let ta = &tris[a.tri as usize];
                    let tb = &bvh.tris[b.tri as usize];
                    assert_eq!(
                        (ta.a.x.to_bits(), ta.a.y.to_bits(), ta.a.z.to_bits()),
                        (tb.a.x.to_bits(), tb.a.y.to_bits(), tb.a.z.to_bits()),
                        "两个下标必须指向同一个三角形"
                    );
                }
                (a, b) => panic!(
                    "两套口径的命中性不一致：raw={:?} slot={:?}",
                    a.map(|h| h.t),
                    b.map(|h| h.t)
                ),
            }
        }
        assert!(hits > 50, "命中数太少（{hits}），判据没有鉴别力");
    }

    /// 关卡 3c：最近点查询 vs 暴力 O(n) 全查一致（120 点）。
    #[test]
    fn closest_point_matches_bruteforce() {
        let tris = random_scene(100);
        let bvh = Bvh::build(&tris);
        for r in 0..120_u64 {
            let p = Vec3::new(
                pcg::hash2_f32(SEED, r + 7, 0) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r + 7, 1) * 8.0 - 4.0,
                pcg::hash2_f32(SEED, r + 7, 2) * 8.0 - 4.0,
            );
            let mut want = ClosestHit { dist: f32::INFINITY, tri: u32::MAX, point: Vec3::splat(0.0) };
            for (i, t) in tris.iter().enumerate() {
                let cp = t.closest_point(p);
                let d = (cp - p).length();
                if d < want.dist || (d == want.dist && (i as u32) < want.tri) {
                    want = ClosestHit { dist: d, tri: i as u32, point: cp };
                }
            }
            let got = bvh.closest_point(p);
            assert_eq!(got.tri, want.tri, "点 {r} 最近图元不一致");
            assert!((got.dist - want.dist).abs() < 1e-6, "点 {r} 距离不一致");
        }
    }

    /// 关卡 3d：winding number——立方体内 ≈ ±1、外 ≈ 0（含远处近似路径）。
    #[test]
    fn winding_number_cube() {
        let tris = unit_cube();
        let bvh = Bvh::build(&tris);
        // 内部点（中心、近角落）→ +1（外法线绕序）。
        for p in [
            Vec3::splat(0.0),
            Vec3::new(0.9, 0.9, 0.9),
            Vec3::new(-0.5, 0.3, -0.7),
        ] {
            let w = bvh.winding_number(p);
            assert!((w - 1.0).abs() < 1e-3, "内部 {p:?} w={w} 应 ≈ 1");
        }
        // 外部点（近表面、远场走一阶近似路径）→ 0。
        for p in [
            Vec3::new(1.05, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.2),
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(-30.0, 12.0, 4.0),
        ] {
            let w = bvh.winding_number(p);
            assert!(w.abs() < 1e-2, "外部 {p:?} w={w} 应 ≈ 0");
        }
        // 确定性：同点两遍一致。
        let p = Vec3::new(2.0, 1.0, 0.5);
        assert_eq!(bvh.winding_number(p), bvh.winding_number(p));
    }

    /// Aabb 基础语义（grow/dist/ray）。
    #[test]
    fn aabb_semantics() {
        let mut b = Aabb::empty();
        b.grow_point(Vec3::new(-1.0, 2.0, 0.5));
        b.grow_point(Vec3::new(3.0, -2.0, 0.5));
        assert_eq!(b.min, Vec3::new(-1.0, -2.0, 0.5));
        assert_eq!(b.max, Vec3::new(3.0, 2.0, 0.5));
        assert!((b.dist_to(Vec3::new(5.0, 0.0, 0.5)) - 2.0).abs() < 1e-6);
        assert_eq!(b.dist_to(Vec3::new(0.0, 0.0, 0.5)), 0.0, "盒内为 0");
        // 穿盒射线有区间交（rd = (0,0,1) → inv x/y 分量为 inf，覆盖 slab 的 IEEE 语义）；
        // 偏离射线无。
        let ro = Vec3::new(0.0, 0.0, -5.0);
        let inv = Vec3::new(f32::INFINITY, f32::INFINITY, 1.0);
        assert!(b.ray(ro, inv, 0.0, 100.0).is_some());
        let ro2 = Vec3::new(0.0, 5.0, -5.0);
        assert!(b.ray(ro2, inv, 0.0, 100.0).is_none(), "偏离应无交");
    }

    /// 轴对齐射线沿盒边界：平行轴上的 `0·inf = NaN` 曾致整盒误判漏交
    /// （轴对齐几何 + 轴对齐相机射线的成片漏交根因）。修复后必须命中。
    #[test]
    fn aabb_ray_parallel_axis_on_boundary() {
        // 盒的 min.z 恰为 0.0，射线原点 z 也恰为 0.0 → 老实现得 0·inf = NaN。
        let b = Aabb { min: Vec3::new(-1.0, -1.0, 0.0), max: Vec3::new(1.0, 1.0, 2.0) };
        let ro = Vec3::new(-5.0, 0.0, 0.0);
        let inv = Vec3::new(1.0, f32::INFINITY, f32::INFINITY);
        let (enter, exit) = b.ray(ro, inv, 0.0, 100.0).expect("沿边界平行轴应命中");
        assert!((enter - 4.0).abs() < 1e-6, "enter = {enter}");
        assert!((exit - 6.0).abs() < 1e-6, "exit = {exit}");
        // 起点落在该轴 slab 之外 ⇒ 必不命中（平行轴显式判定）。
        assert!(b.ray(Vec3::new(-5.0, 0.0, 3.0), inv, 0.0, 100.0).is_none());
        assert!(b.ray(Vec3::new(-5.0, 0.0, -1.0), inv, 0.0, 100.0).is_none());
        // 三个轴同时退化（射线原点在盒角上、沿 -x 反向）也不得 panic/误判。
        let inv3 = Vec3::new(-1.0, f32::INFINITY, f32::NEG_INFINITY);
        assert!(b.ray(Vec3::new(1.0, -1.0, 0.0), inv3, 0.0, 100.0).is_some());
    }

    /// mesh_triangles：索引流到三角形的适配。
    #[test]
    fn mesh_triangles_adapter() {
        let pos = vec![
            Vec3::splat(0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let idx = vec![0, 1, 2, 0, 2, 3];
        let tris = mesh_triangles(&pos, &idx);
        assert_eq!(tris.len(), 2);
        assert_eq!(tris[1].c, Vec3::new(0.0, 0.0, 1.0));
        assert!((tris[0].area() - 0.5).abs() < 1e-6, "直角边 1 的三角形面积 0.5");
    }
}
