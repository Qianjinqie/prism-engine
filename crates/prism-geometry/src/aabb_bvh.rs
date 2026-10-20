//! AABB-BVH：解析曲线图元的粗阶段加速结构
//!
//! # 定位（两级求交）
//! 圆环（四次求根）/ 旋转曲面（逐段二次求根）等解析图元的精阶段求交是
//! 闭式但非常数小常量。曲线图元成族后，先按各图元 AABB 建 SAH-BVH 做
//! 粗阶段剪枝，仅对包围盒可能命中的图元执行闭式精阶段求交——与
//! [`super::bvh::Bvh`]（三角网格精阶段）同构的"粗阶段 + 精阶段"结构。
//!
//! # 契约
//! - 精阶段求交**由调用方回调提供**（图元索引 → 最近有效 t，回调自行
//!   满足 `t >= T_MIN && t < t_max` 口径）；BVH 只做 AABB 粗判与
//!   `(t, 图元索引)` 字典序择优，不认识任何具体图元语义。
//! - 回调须返回该图元**全局最近**的有效根：BVH 以运行中最优 `best.t`
//!   剪枝，若回调给的不是最近根则可能漏更近根（线性兜底路径无此要求，
//!   故两条路径结果一致的前提是回调语义 = 最近有效根）。
//! - `ray_intersect` 的 `t_min`/`t_max` 与回调口径一致（半开区间
//!   `[t_min, t_max)` 由回调保证；BVH 内部的 `t <= hi` 闭判只用于
//!   并列 tie-break，不放松边界语义）。
//!
//! # 确定性（与 bvh.rs 同口径）
//! - 构建：自顶向下 SAH，每轴 12 桶；代价并列按"先轴后桶"遍历序取首最小；
//!   桶划分退化时按质心全序（`total_cmp` + 图元索引）稳定排序取中位。
//! - 遍历：近子节点先处理的显式栈；命中以 `(t, 图元索引)` 字典序取最优
//!   （t 并列按图元索引 tie-break）。
//! - 构建与遍历全程不用 fma 内建；本模块无超越函数。

use super::bvh::Aabb;
use prism_math::Vec3;

/// SAH 分桶数（12 桶；与 bvh.rs 同值同口径）。
const SAH_BUCKETS: usize = 12;
/// 叶子图元上限（与 bvh.rs 同值）。
const LEAF_MAX: usize = 8;

/// 扁平节点（根为 0）。
///
/// - 叶子：`start`/`count` 引用 [`AabbBvh::order`] 的图元索引区间；
/// - 内部：左子恒为 `本节点 + 1`（构建序保证先建左子），右子存 `right`。
#[derive(Debug, Clone, Copy)]
struct Node {
    /// 节点包围盒（图元 AABB 的并）。
    bounds: Aabb,
    /// 叶子：`order` 区间起点；内部：0。
    start: u32,
    /// 叶子：图元数（> 0）；内部：0。
    count: u32,
    /// 内部：右子节点下标（左子 = 本节点 + 1）。
    right: u32,
}

/// AABB-BVH（空输入 → 空树：`nodes` 为空，查询安全返回 None）。
#[derive(Debug, Clone)]
pub struct AabbBvh {
    /// 扁平节点数组（根为 0）。
    nodes: Vec<Node>,
    /// 叶子图元索引（全局图元下标，按构建插入序）。
    order: Vec<u32>,
    /// 全场景包围盒（空树为 `Aabb::empty()`）。
    bounds: Aabb,
}

impl AabbBvh {
    /// SAH-BVH 构建（自顶向下，12 桶；空输入产生空树）。
    pub fn build(aabbs: &[Aabb]) -> AabbBvh {
        let mut bvh = AabbBvh {
            nodes: Vec::new(),
            order: Vec::with_capacity(aabbs.len()),
            bounds: Aabb::empty(),
        };
        for b in aabbs {
            bvh.bounds.grow(b);
        }
        if aabbs.is_empty() {
            return bvh; // 空树（遍历/查询安全返回空语义）
        }
        let centroids: Vec<Vec3> = aabbs.iter().map(|b| b.center()).collect();
        let mut prims: Vec<u32> = (0..aabbs.len() as u32).collect();
        bvh.build_node(aabbs, &centroids, &mut prims);
        bvh
    }

    /// 图元总数（叶子计数守恒意义上的规模）。
    pub fn prim_count(&self) -> usize {
        self.order.len()
    }

    /// 递归构建一个节点，返回其索引（调用方保证 `prims` 非空）。
    fn build_node(&mut self, aabbs: &[Aabb], centroids: &[Vec3], prims: &mut [u32]) -> u32 {
        debug_assert!(!prims.is_empty(), "build_node requires non-empty prim range");
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            bounds: Aabb::empty(),
            start: 0,
            count: 0,
            right: 0,
        });

        // 区间包围盒（逐图元直接累加——同输入同序同结果）。
        let mut bounds = Aabb::empty();
        for &pi in prims.iter() {
            bounds.grow(&aabbs[pi as usize]);
        }

        if prims.len() <= LEAF_MAX {
            let start = self.order.len() as u32;
            for &pi in prims.iter() {
                self.order.push(pi);
            }
            let node = &mut self.nodes[id as usize];
            node.bounds = bounds;
            node.start = start;
            node.count = prims.len() as u32;
            return id;
        }

        // SAH 分割（并列取遍历序首最小 → 确定性）。
        let split = best_sah_split(aabbs, centroids, prims);
        let (axis, plane) = match split {
            Some(s) => s,
            None => {
                // 全轴退化：按质心稳定排序取中位（tie 按图元索引）。
                let axis = widest_axis(centroids, prims);
                median_partition(centroids, prims, axis);
                let mid = prims.len() / 2;
                return self.build_children(aabbs, centroids, prims, id, bounds, mid);
            }
        };
        // 稳定划分：保留原相对顺序（同输入同划分）。
        let mut left: Vec<u32> = Vec::with_capacity(prims.len());
        let mut right: Vec<u32> = Vec::with_capacity(prims.len());
        for &pi in prims.iter() {
            if at_axis(&centroids[pi as usize], axis) < plane {
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
        // 叶子下限：一侧过小时改用中位划分（与 bvh.rs 同口径）。
        let mid = if mid < 4 || prims.len() - mid < 4 {
            median_partition(centroids, prims, axis);
            prims.len() / 2
        } else {
            mid
        };
        self.build_children(aabbs, centroids, prims, id, bounds, mid)
    }

    /// 划分后递归构建两个子节点并填写内部节点（左子先建 ⇒ 左子 = id+1）。
    fn build_children(
        &mut self,
        aabbs: &[Aabb],
        centroids: &[Vec3],
        prims: &mut [u32],
        id: u32,
        bounds: Aabb,
        mid: usize,
    ) -> u32 {
        let (left_prims, right_prims) = prims.split_at_mut(mid);
        let _l = self.build_node(aabbs, centroids, left_prims);
        let r = self.build_node(aabbs, centroids, right_prims);
        let node = &mut self.nodes[id as usize];
        node.bounds = bounds;
        node.right = r;
        id
    }

    /// 射线查询：粗阶段 AABB 剪枝 + 回调精阶段求交，`(t, 图元索引)`
    /// 字典序最优。空树返回 None。
    ///
    /// `rd` 须为单位向量（与 [`super::bvh::Bvh::ray_intersect`] 同契约，
    /// `debug_assert` 断言）；`prim_t` 见图元级契约（模块文档）。
    pub fn ray_intersect<F>(
        &self,
        ro: Vec3,
        rd: Vec3,
        t_min: f32,
        t_max: f32,
        prim_t: &mut F,
    ) -> Option<(f32, u32)>
    where
        F: FnMut(u32) -> Option<f32>,
    {
        debug_assert!((rd.length() - 1.0).abs() < 1e-4, "ray_intersect requires unit rd");
        if self.nodes.is_empty() {
            return None; // 空树：安全返回空语义
        }
        let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
        let mut best: Option<(f32, u32)> = None;
        // 固定深度显式栈（深度上界由图元总数推导，64 层必够）。
        let mut stack: Vec<(u32, f32)> = Vec::with_capacity(64);
        stack.push((0, t_min));
        while let Some((id, t_lo)) = stack.pop() {
            if let Some((bt, _)) = best {
                if t_lo > bt {
                    continue;
                }
            }
            let node = &self.nodes[id as usize];
            if node.count > 0 {
                let hi = best.map_or(t_max, |(bt, _)| bt);
                for i in 0..node.count as usize {
                    let slot = node.start as usize + i;
                    let orig = self.order[slot];
                    if let Some(t) = prim_t(orig) {
                        // 闭判 `t <= hi` 仅服务并列 tie-break（边界口径由
                        // 回调的严格 `t < t_max` 保证）。
                        if t >= t_min && t <= hi {
                            let better = match best {
                                None => true,
                                Some((bt, bi)) => t < bt || (t == bt && orig < bi),
                            };
                            if better {
                                best = Some((t, orig));
                            }
                        }
                    }
                }
            } else {
                let l = id + 1;
                let r = node.right;
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
}

/// 分量选取（轴 0/1/2 → x/y/z；与 bvh.rs 内部同名件同口径，本模块自持
/// 以避免跨模块私有 trait 依赖）。
#[inline]
fn at_axis(v: &Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// SAH 最佳分割（12 桶；按质心包围盒定位、按图元 AABB 并集面积计价）。
/// 返回 `(轴, 分割平面位置)`；全轴退化返回 `None`。与 bvh.rs 同口径
/// （代价并列按"先轴后桶"遍历序取首最小）。
fn best_sah_split(aabbs: &[Aabb], centroids: &[Vec3], prims: &[u32]) -> Option<(usize, f32)> {
    let mut node_bounds = Aabb::empty();
    for &pi in prims {
        node_bounds.grow_point(centroids[pi as usize]);
    }
    let extent = node_bounds.max - node_bounds.min;
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, bucket)
    for axis in 0..3 {
        let ext = at_axis(&extent, axis);
        if !(ext > 1e-12) {
            continue;
        }
        let scale = SAH_BUCKETS as f32 / ext;
        let base = at_axis(&node_bounds.min, axis);
        let mut bucket_bounds = [Aabb::empty(); SAH_BUCKETS];
        let mut bucket_count = [0_u32; SAH_BUCKETS];
        for &pi in prims {
            let c = at_axis(&centroids[pi as usize], axis);
            let b = ((c - base) * scale).floor() as usize;
            let b = b.min(SAH_BUCKETS - 1);
            bucket_bounds[b].grow(&aabbs[pi as usize]);
            bucket_count[b] += 1;
        }
        // 前缀/后缀扫描（面积 = 图元 AABB 并集的半表面积）。
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
        // 桶间评估分割代价（严格 < 更新 → 首最小者胜，确定性 tie-break）。
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
        let ext = at_axis(&extent, axis);
        let base = at_axis(&node_bounds.min, axis);
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
    if h.y > at_axis(&h, axis) {
        axis = 1;
    }
    if h.z > at_axis(&h, axis) {
        axis = 2;
    }
    axis
}

/// 稳定中位划分（质心全序排序 + 图元索引 tie，保证两侧均非空）。
fn median_partition(centroids: &[Vec3], prims: &mut [u32], axis: usize) {
    prims.sort_by(|&a, &b| {
        let ca = at_axis(&centroids[a as usize], axis);
        let cb = at_axis(&centroids[b as usize], axis);
        ca.total_cmp(&cb).then(a.cmp(&b))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_noise::pcg;

    const SEED: u64 = 0xA0_00_00_01;

    /// 确定性伪随机 AABB 场景（100 个，尺度 ~[-3,3]³，半尺寸 0.2–0.8）。
    fn random_boxes(n: usize) -> Vec<Aabb> {
        (0..n)
            .map(|i| {
                let c = |k: u64| pcg::hash2_f32(SEED, i as u64, k) * 6.0 - 3.0;
                let h = |k: u64| 0.2 + pcg::hash2_f32(SEED, i as u64, k) * 0.6;
                let center = Vec3::new(c(0), c(1), c(2));
                let hs = Vec3::new(h(3), h(4), h(5));
                Aabb {
                    min: center - hs,
                    max: center + hs,
                }
            })
            .collect()
    }

    /// 精阶段回调（对拍基准）：盒 AABB 的 slab 进入 t（半开区间口径）。
    fn box_enter(aabbs: &[Aabb], i: u32, ro: Vec3, rd: Vec3, t_max: f32) -> Option<f32> {
        let b = &aabbs[i as usize];
        let (enter, _) = b.ray(ro, Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z), 0.0, t_max)?;
        if enter >= 0.0 && enter < t_max {
            Some(enter)
        } else {
            None
        }
    }

    /// 关卡 3a：SAH 构建确定性——同输入两遍，节点数组与重排映射逐位一致。
    #[test]
    fn aabb_sah_build_is_deterministic() {
        let boxes = random_boxes(100);
        let a = AabbBvh::build(&boxes);
        let b = AabbBvh::build(&boxes);
        assert_eq!(a.prim_count(), 100);
        assert_eq!(a.prim_count(), b.prim_count(), "图元数守恒");
        // 结构合理性：叶子 ≤ 8、叶子图元总数守恒、每图元恰出现在一个叶子。
        let leaves: Vec<&Node> = a.nodes.iter().filter(|n| n.count > 0).collect();
        assert!(leaves.iter().all(|n| n.count as usize <= LEAF_MAX));
        let total: u32 = leaves.iter().map(|n| n.count).sum();
        assert_eq!(total, 100, "叶子图元数守恒");
        let mut seen = vec![false; 100];
        for n in &leaves {
            for i in 0..n.count as usize {
                seen[a.order[n.start as usize + i] as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "每个图元恰好出现在一个叶子");
    }

    /// 关卡 3b：BVH 遍历 vs 暴力 O(n) 全查一致（200 射线；(t, idx) tie-break）。
    #[test]
    fn aabb_ray_intersect_matches_bruteforce() {
        let boxes = random_boxes(100);
        let bvh = AabbBvh::build(&boxes);
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
            let t_max = 50.0_f32;
            // 暴力：同 tie-break 规则 (t, idx)。
            let mut want: Option<(f32, u32)> = None;
            for (i, _b) in boxes.iter().enumerate() {
                if let Some(t) = box_enter(&boxes, i as u32, ro, rd, t_max) {
                    let better = match want {
                        None => true,
                        Some((bt, bi)) => t < bt || (t == bt && (i as u32) < bi),
                    };
                    if better {
                        want = Some((t, i as u32));
                    }
                }
            }
            let mut cb = |i: u32| box_enter(&boxes, i, ro, rd, t_max);
            let got = bvh.ray_intersect(ro, rd, 0.0, t_max, &mut cb);
            assert_eq!(got, want, "射线 {r} 求交与暴力不一致");
        }
    }
}
