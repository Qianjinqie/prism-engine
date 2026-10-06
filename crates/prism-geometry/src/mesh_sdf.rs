//! mesh→SDF 烘焙：brick 8³ 稀疏化 + 广义 winding number 符号 + BVH 最近距离
//! （白皮书 §7；架构 v2.4 S12 对偶切换 bake 特解）。
//!
//! # 烘焙语义（白皮书 §7）
//!
//! - 仅表面附近 brick 精确化（三角形 AABB 外扩 `pad_voxels` 决定激活范围）；
//!   近场距离对候选三角形做精确最近点查询。
//! - 内外判定用广义 winding number（复用 [`crate::bvh::Bvh`] 的实体角聚合，一树两用）；
//!   判定病态（w 明显偏离 0/1/表面过渡带）按 exterior 处理并计数警告——**永不静默**。
//! - 远场（未激活 brick / 场外）返回**下界**：绝对值不大于真实距离，sphere tracing 步进安全。
//!
//! # 确定性
//!
//! 每个 brick 的每个体素独立计算（无跨体素浮点累积），哈希表遍历序不影响逐体素取值
//! → 同输入同字节。BVH 构建同输入同树（见 [`crate::bvh`]）。

use crate::bvh::{mesh_triangles, Bvh, Triangle};
use prism_math::Vec3;
use std::collections::HashMap;

/// brick 边长（体素数，白皮书 §7：brick 8³）。
pub const BRICK: usize = 8;

/// 烘焙参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakeOptions {
    /// 体素间距（世界单位）。
    pub spacing: f32,
    /// 表面外扩体素数（决定近场精确区的厚度）。
    pub pad_voxels: u32,
}

impl Default for BakeOptions {
    fn default() -> Self {
        BakeOptions { spacing: 0.1, pad_voxels: 2 }
    }
}

/// 烘焙报告（加载报告式诚实披露，宪章 C7：永不静默）。
#[derive(Debug, Clone, PartialEq)]
pub struct BakeReport {
    /// winding number 病态判定计数（既非 0/1 也非表面过渡带的中间值）。
    pub degenerate_winding: usize,
    /// 警告明细。
    pub warnings: Vec<String>,
}

/// 烘焙产物：稀疏 brick 距离场。
#[derive(Debug, Clone, PartialEq)]
pub struct BakedSdf {
    /// 体素 (0,0,0) 中心的世界坐标。
    pub origin: Vec3,
    /// 体素间距。
    pub spacing: f32,
    /// 三轴体素数。
    pub dims: [u32; 3],
    /// 激活 brick（key = 打包 brick 坐标 → 512 体素距离值，按 (x*8+y)*8+z 索引）。
    pub bricks: HashMap<u64, Box<[f32; BRICK * BRICK * BRICK]>>,
    /// 持有的加速结构（winding 符号判定 / 后续最近点复用，一树两用）。
    pub bvh: Bvh,
    /// 近场有效域 AABB（[origin, origin + (dims−1)·spacing]）。
    pub bounds: crate::bvh::Aabb,
}

impl BakedSdf {
    /// 有符号距离查询（内负外正）。
    ///
    /// 命中激活 brick → 三线性插值（近场精确）；
    /// 场内未激活 → winding 判号后返回 ±spacing（安全下界，白皮书"远场距离下界"）；
    /// 场外 → 返回到场域 AABB 的距离（正下界）。
    pub fn dist(&self, p: Vec3) -> f32 {
        let f = (p - self.origin) * (1.0 / self.spacing);
        let dmax = [self.dims[0] as f32 - 1.0, self.dims[1] as f32 - 1.0, self.dims[2] as f32 - 1.0];
        if f.x < 0.0 || f.y < 0.0 || f.z < 0.0 || f.x > dmax[0] || f.y > dmax[1] || f.z > dmax[2] {
            return self.bounds.dist_to(p);
        }
        // 体素坐标约定：体素 i 的中心在 origin + i·spacing → 世界点的体素坐标 f 直接
        // 以体素中心为整数点（与烘焙写入 p = origin + vi·s 一致）。
        let fi = [f.x, f.y, f.z];
        let bkey = |i: usize| -> u64 {
            let b = (fi[i] / BRICK as f32).floor();
            b.clamp(0.0, ((self.dims[i] as f32 - 1.0) / BRICK as f32).floor()) as u64
        };
        let key = pack_brick(bkey(0), bkey(1), bkey(2));
        let Some(cell) = self.bricks.get(&key) else {
            // 未激活：内部深处或外部空腔——winding 判号，幅度取安全下界 spacing。
            // 病态置信判定（白皮书 §7）：|w| 不在 0/±1 置信带 → 符号不可信，
            // 按 exterior 处理（正值下界）。计数披露在烘焙期完成（dist 为 &self）。
            let w = self.bvh.winding_number(p);
            if !winding_pathological(w) && w > 0.5 {
                return -self.spacing;
            }
            return self.spacing;
        };
        // 三线性插值：基点 clamp 到 [0, BRICK−2]（保证 +1 不越界），权重 clamp 到 [0, 1]
        // （边缘半体素取边界值）。
        let mut base = [0_usize; 3];
        let mut frac = [0.0_f32; 3];
        for i in 0..3 {
            let lf = fi[i] - bkey(i) as f32 * BRICK as f32;
            let i0 = lf.floor().clamp(0.0, (BRICK - 2) as f32) as usize;
            base[i] = i0;
            frac[i] = (lf - i0 as f32).clamp(0.0, 1.0);
        }
        let sample = |x: usize, y: usize, z: usize| cell[(x * BRICK + y) * BRICK + z];
        let c00 = sample(base[0], base[1], base[2])
            + frac[2] * (sample(base[0], base[1], base[2] + 1) - sample(base[0], base[1], base[2]));
        let c10 = sample(base[0] + 1, base[1], base[2])
            + frac[2]
                * (sample(base[0] + 1, base[1], base[2] + 1) - sample(base[0] + 1, base[1], base[2]));
        let c01 = sample(base[0], base[1] + 1, base[2])
            + frac[2]
                * (sample(base[0], base[1] + 1, base[2] + 1) - sample(base[0], base[1] + 1, base[2]));
        let c11 = sample(base[0] + 1, base[1] + 1, base[2])
            + frac[2]
                * (sample(base[0] + 1, base[1] + 1, base[2] + 1)
                    - sample(base[0] + 1, base[1] + 1, base[2]));
        let c0 = c00 + frac[1] * (c01 - c00);
        let c1 = c10 + frac[1] * (c11 - c10);
        c0 + frac[0] * (c1 - c0)
    }
}

/// brick 坐标打包为哈希键（各轴 21 位，覆盖 2M brick/轴）。
#[inline]
pub fn pack_brick(bx: u64, by: u64, bz: u64) -> u64 {
    (bx << 42) | (by << 21) | bz
}

/// winding number 置信判定（白皮书 §7：病态网格按 exterior 处理——永不静默）。
///
/// 置信带：|w| ≤ 0.05（可靠外部）、|w| ≥ 0.95（可靠内部）、[0.45, 0.55)
/// （表面过渡带，w 在表面上本就无定义）；带外即**病态**。病态体素的符号
/// 判定不可信，一律按 exterior 处理（距离取 |d| 正值），烘焙期另计数进
/// [`BakeReport`] 披露。判定是纯函数：烘焙与查询两侧共用同一谓词，同 w 同结论。
#[inline]
fn winding_pathological(w: f32) -> bool {
    let aw = w.abs();
    (aw > 0.05 && aw <= 0.45) || (aw >= 0.55 && aw < 0.95)
}

/// mesh（顶点 + 索引）→ 有符号距离场烘焙。
pub fn bake_mesh(
    positions: &[Vec3],
    indices: &[u32],
    opts: &BakeOptions,
) -> (BakedSdf, BakeReport) {
    let tris = mesh_triangles(positions, indices);
    let bvh = Bvh::build(&tris);
    bake_triangles(&tris, &bvh, opts)
}

/// 三角形数组 → 烘焙（已有 BVH 时避免重建）。
pub fn bake_triangles(
    tris: &[Triangle],
    bvh: &Bvh,
    opts: &BakeOptions,
) -> (BakedSdf, BakeReport) {
    let s = opts.spacing;
    let pad = opts.pad_voxels as i64;
    let mut report = BakeReport { degenerate_winding: 0, warnings: Vec::new() };
    if tris.is_empty() {
        report.warnings.push("bake: 空网格，产物为空场".to_string());
        return (
            BakedSdf {
                origin: Vec3::splat(0.0),
                spacing: s,
                dims: [0, 0, 0],
                bricks: HashMap::new(),
                bvh: bvh.clone(),
                bounds: crate::bvh::Aabb::empty(),
            },
            report,
        );
    }

    // 场域：网格包围盒 ± pad 体素。
    let mut mesh_min = Vec3::splat(f32::INFINITY);
    let mut mesh_max = Vec3::splat(f32::NEG_INFINITY);
    for t in tris {
        for v in [t.a, t.b, t.c] {
            mesh_min = crate::vmin(mesh_min, v);
            mesh_max = crate::vmax(mesh_max, v);
        }
    }
    let origin = mesh_min - Vec3::splat(pad as f32 * s);
    let mut dims = [0_u32; 3];
    let mesh_min_a = [mesh_min.x, mesh_min.y, mesh_min.z];
    let mesh_max_a = [mesh_max.x, mesh_max.y, mesh_max.z];
    for i in 0..3 {
        dims[i] = ((mesh_max_a[i] - mesh_min_a[i]) / s).ceil() as u32 + 2 * opts.pad_voxels + 1;
    }

    // 候选表：三角形 AABB 外扩 pad 体素覆盖的每个 brick 收一次（激活集与候选一次成型）。
    let mut candidates: HashMap<u64, Vec<u32>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let (mut tmin, mut tmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in [t.a, t.b, t.c] {
            tmin = crate::vmin(tmin, v);
            tmax = crate::vmax(tmax, v);
        }
        let ta = [tmin.x, tmin.y, tmin.z];
        let tb = [tmax.x, tmax.y, tmax.z];
        let mut range_lo = [0_i64; 3];
        let mut range_hi = [0_i64; 3];
        for i in 0..3 {
            let lo = ((ta[i] - origin.at(i)) / s).floor() as i64 - pad as i64;
            let hi = ((tb[i] - origin.at(i)) / s).floor() as i64 + pad as i64;
            range_lo[i] = lo.clamp(0, dims[i] as i64 - 1);
            range_hi[i] = hi.clamp(0, dims[i] as i64 - 1);
        }
        let b_lo = [range_lo[0] / BRICK as i64, range_lo[1] / BRICK as i64, range_lo[2] / BRICK as i64];
        let b_hi = [
            range_hi[0].min(dims[0] as i64 - 1) / BRICK as i64,
            range_hi[1].min(dims[1] as i64 - 1) / BRICK as i64,
            range_hi[2].min(dims[2] as i64 - 1) / BRICK as i64,
        ];
        for bx in b_lo[0]..=b_hi[0] {
            for by in b_lo[1]..=b_hi[1] {
                for bz in b_lo[2]..=b_hi[2] {
                    candidates
                        .entry(pack_brick(bx as u64, by as u64, bz as u64))
                        .or_default()
                        .push(ti as u32);
                }
            }
        }
    }

    // 逐激活 brick 逐体素：候选精确最近点 + winding 符号。
    let mut bricks: HashMap<u64, Box<[f32; BRICK * BRICK * BRICK]>> = HashMap::new();
    for (key, cands) in candidates.iter() {
        let (bx, by, bz) = unpack_brick(*key);
        let mut cell = Box::new([0.0_f32; BRICK * BRICK * BRICK]);
        for lx in 0..BRICK {
            for ly in 0..BRICK {
                for lz in 0..BRICK {
                    let vi = [
                        bx as i64 * BRICK as i64 + lx as i64,
                        by as i64 * BRICK as i64 + ly as i64,
                        bz as i64 * BRICK as i64 + lz as i64,
                    ];
                    if vi[0] >= dims[0] as i64 || vi[1] >= dims[1] as i64 || vi[2] >= dims[2] as i64 {
                        continue; // 场域外的收尾体素（不采样，不影响查询）
                    }
                    let p = Vec3::new(
                        origin.x + (vi[0] as f32) * s,
                        origin.y + (vi[1] as f32) * s,
                        origin.z + (vi[2] as f32) * s,
                    );
                    let mut d = f32::INFINITY;
                    for &ci in cands.iter() {
                        let cp = tris[ci as usize].closest_point(p);
                        let dd = (cp - p).length();
                        if dd < d {
                            d = dd;
                        }
                    }
                    let w = bvh.winding_number(p);
                    let near_surface = d <= 2.0 * s;
                    // 病态检测：|w| 明显偏离 0/±1 且不在表面过渡带（0.45–0.55）。
                    // 点在表面附近（|d| ≤ 2·spacing）时 winding 本就过渡（0 → 0.5 → 1，
                    // 表面上无定义），不算病态；只抓"远离表面却不接近 0/1"的真病态。
                    let pathological = winding_pathological(w);
                    if pathological && !near_surface {
                        report.degenerate_winding += 1;
                    }
                    // 白皮书 §7 契约：病态体素**按 exterior 处理**（永不静默）——
                    // 符号判定不区分近远场（近场过渡体素 w 落在病态带同样不可信，
                    // 判外取 |d| 正值），计数仍只报远离表面的真病态。
                    let inside = !pathological && w > 0.5;
                    cell[(lx * BRICK + ly) * BRICK + lz] = if inside { -d } else { d };
                }
            }
        }
        bricks.insert(*key, cell);
    }

    (
        BakedSdf {
            origin,
            spacing: s,
            dims,
            bricks,
            bvh: bvh.clone(),
            bounds: crate::bvh::Aabb {
                min: origin,
                max: origin
                    + Vec3::new(
                        (dims[0] - 1) as f32 * s,
                        (dims[1] - 1) as f32 * s,
                        (dims[2] - 1) as f32 * s,
                    ),
            },
        },
        report,
    )
}

/// 解包 brick 哈希键。
#[inline]
pub fn unpack_brick(key: u64) -> (u64, u64, u64) {
    ((key >> 42) & 0x1F_FFFF, (key >> 21) & 0x1F_FFFF, key & 0x1F_FFFF)
}

/// 索引辅助 trait（体素坐标按轴取分量）。
trait AtIdx {
    fn at(&self, i: usize) -> f32;
}

impl AtIdx for Vec3 {
    #[inline]
    fn at(&self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl BakedSdf {
    /// brick 哈希键的内部便捷复打包（供测试断言）。
    pub fn brick_key_of(&self, p: Vec3) -> u64 {
        let f = (p - self.origin) * (1.0 / self.spacing);
        let b = |v: f32| (v / BRICK as f32).floor().max(0.0) as u64;
        pack_brick(b(f.x), b(f.y), b(f.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::testutil::unit_cube;

    /// 解析盒 SDF（测试参照）。
    fn box_sdf(p: Vec3) -> f32 {
        let h = Vec3::splat(1.0);
        let q = Vec3::new(p.x.abs(), p.y.abs(), p.z.abs()) - h;
        let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
        outside.length() + q.x.max(q.y).max(q.z).min(0.0)
    }

    /// 关卡 5：立方体烘焙——内负外正、|d| 与解析距离差 < 体素尺度、近场/远场/下界语义。
    #[test]
    fn bake_cube_interior_negative_exterior_positive() {
        let positions: Vec<Vec3> = {
            // 与 bvh::testutil::unit_cube 同一顶点表。
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
            vec![v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]
        };
        let indices: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5, 0, 1, 5, 0, 5,
            4, 2, 3, 7, 2, 7, 6,
        ];
        let opts = BakeOptions { spacing: 0.1, pad_voxels: 2 };
        let (baked, report) = bake_mesh(&positions, &indices, &opts);
        assert_eq!(report.degenerate_winding, 0, "水密立方体不应有病态判定");

        let tolerance = opts.spacing; // 关卡判定线：|d| 与解析距离差 < 体素尺度
        // 近场内部点：解析 −0.15（距 x=+1 面）。
        let p = Vec3::new(0.85, 0.0, 0.0);
        let got = baked.dist(p);
        assert!(got < 0.0, "内部应为负：{got}");
        assert!(
            (got - box_sdf(p)).abs() < tolerance,
            "内点 |d|={got} vs 解析 {}",
            box_sdf(p)
        );
        // 近场外部点：解析 +0.15。
        let p = Vec3::new(1.15, 0.0, 0.0);
        let got = baked.dist(p);
        assert!(got > 0.0, "外部应为正：{got}");
        assert!(
            (got - box_sdf(p)).abs() < tolerance,
            "外点 |d|={got} vs 解析 {}",
            box_sdf(p)
        );
        // 表面点：|d| < 体素尺度。
        let got = baked.dist(Vec3::new(1.0, 0.0, 0.0));
        assert!(got.abs() < tolerance, "表面点 |d|={got}");
        // 近场内部点（−x 面）：解析 −0.12。
        let p = Vec3::new(-0.88, 0.1, -0.1);
        let got = baked.dist(p);
        assert!(got < 0.0 && (got - box_sdf(p)).abs() < tolerance, "−x 内点 {got}");
        // 近场外部点（对角区）：解析 = (1.2,1.2,1.2) 到盒角 (1,1,1) 距离 ≈ 0.346。
        let p = Vec3::splat(1.2);
        let got = baked.dist(p);
        assert!(got > 0.0 && (got - box_sdf(p)).abs() < tolerance, "角区外点 {got}");

        // 深内部：符号为负且幅度为下界（|d| ≤ 真实距离 1）。
        let got = baked.dist(Vec3::splat(0.0));
        assert!(got < 0.0, "深内部应为负");
        assert!(got.abs() <= 1.0, "下界性质：|d|={got} ≤ 1");
        // 深外部（场外）：正值且为下界（≤ 到表面真距 √3·2.8…）。
        let p = Vec3::splat(3.0);
        let got = baked.dist(p);
        assert!(got > 0.0, "深外部应为正");
        assert!(got <= box_sdf(p), "下界性质：{got} ≤ {}", box_sdf(p));

        // 沿 x 轴符号扫描：|x|>1 为正、|x|<0.9 为负（激活近场带）。
        let mut i = 0.9_f32;
        while i < 1.25 {
            let d = baked.dist(Vec3::new(i, 0.05, -0.05));
            if i < 1.0 {
                assert!(d < 0.0, "x={i} 应为内部（负），得 {d}");
            } else if i > 1.05 {
                assert!(d > 0.0, "x={i} 应为外部（正），得 {d}");
            }
            i += 0.05;
        }
    }

    /// 烘焙确定性：同输入两遍，brick 数据逐位一致 + dist 采样一致。
    #[test]
    fn bake_is_deterministic() {
        let tris = unit_cube();
        let bvh = Bvh::build(&tris);
        let opts = BakeOptions { spacing: 0.15, pad_voxels: 2 };
        let (a, ra) = bake_triangles(&tris, &bvh, &opts);
        let (b, rb) = bake_triangles(&tris, &bvh, &opts);
        assert_eq!(ra, rb, "报告一致");
        assert_eq!(a.bricks.len(), b.bricks.len(), "激活 brick 数一致");
        for (k, va) in a.bricks.iter() {
            let vb = b.bricks.get(k).expect("brick 集合一致");
            assert_eq!(va.as_slice(), vb.as_slice(), "brick {k} 逐位一致");
        }
        for i in 0..40_u64 {
            let p = Vec3::new(
                prism_noise::pcg::hash2_f32(7, i, 0) * 4.0 - 2.0,
                prism_noise::pcg::hash2_f32(7, i, 1) * 4.0 - 2.0,
                prism_noise::pcg::hash2_f32(7, i, 2) * 4.0 - 2.0,
            );
            assert_eq!(a.dist(p), b.dist(p), "dist({p:?}) 逐位一致");
        }
    }

    /// brick 键打包/解包往返。
    #[test]
    fn brick_key_roundtrip() {
        for (x, y, z) in [(0u64, 0, 0), (1, 2, 3), (123, 456, 789), (0x1F_FFFF, 0, 0x1F_FFFF)] {
            let k = pack_brick(x, y, z);
            assert_eq!(unpack_brick(k), (x, y, z));
        }
    }

    /// 空网格：空场 + 警告（永不静默），dist 不 panic。
    #[test]
    fn bake_empty_mesh_warns() {
        let (baked, report) = bake_mesh(&[], &[], &BakeOptions::default());
        assert!(baked.bricks.is_empty());
        assert!(!report.warnings.is_empty(), "空网格必须显式警告");
        let _ = baked.dist(Vec3::splat(1.0)); // 空场查询路径不 panic
    }
}
