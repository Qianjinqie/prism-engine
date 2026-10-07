//! Voronoi 破碎（内容批）。
//!
//! 语义权威：（Voronoi 破碎：3D 站点由种子派生，**先对站点加
//! 噪声偏移再做 Voronoi 切分** → 碎片边缘自然不规则；扰动站点而非切割平面，
//! 相邻单元共享同一平分面 → **无缝性自动保持**；冲击密度自适应：站点密度随距
//! 冲击点距离衰减，撞哪碎哪；碎片运动 = 粒子云 + 内部约束入 XPBD 求解；
//! 破碎 = XPBD **约束重分配**；风险注解"用平面切割半空间序列实现（每碎片 =
//! 半空间交集序列），避开通用布尔"）、D-18。
//!
//! # 数学结构（半平面交集，避开通用布尔）
//!
//! - **站点**：由确定性种子派生（无全局 RNG —— 第 1 条"D"格锚；
//!   本模块的种子派生纯函数见 [`SiteSeed`]，B8 粒子/噪声族复用）。
//! - **Voronoi 单元**：单元 i = 到站点 sᵢ 比到其它站点都近的点集。等价于
//!   对每个 j ≠ i 的**半平面** `{p : (p − m_ij)·(sᵢ − sⱼ) ≤ 0}` 之交，
//!   其中 `m_ij = (sᵢ + sⱼ)/2` 是平分线中点。从包围盒多边形出发，逐半平面
//!   **Sutherland–Hodgman 裁剪** → 得到单元凸多边形。
//! - **无缝性**：平分面由 (sᵢ, sⱼ) 唯一决定，i 与 j 的公共边在同一直线上 →
//!   单元拼合无缝、覆盖完整（**面积和 = 原面积**，验收项）。
//! - **冲击密度自适应**：站点采样按"距冲击点距离"衰减密度（近处密、远处疏）。
//!
//! # 3D 与真实网格
//!
//! 本批落 2D 单元（面积为验收量纲）；3D = 半空间交集序列的凸多面体裁剪，结构
//! 同构（本模块的 `half_plane_clip` 逐面推广即 3D 的凸多面体逐半空间裁剪），
//! 复用 prism-geometry 的几何件（"破碎复用 Delaunay/几何"）——
//! 2D 先行的可测形态在此落定，3D 网格布尔留后续轮次（不越界实装）。
//! `[J]` 标注：站点派生与切割均为 GPU-friendly 的逐单元独立计算。

use crate::PhysicsError;

// ---------------------------------------------------------------------------
// 确定性站点派生（无全局 RNG：种子纯函数）
// ---------------------------------------------------------------------------

/// 站点种子派生器：以 (global_seed, index) 为输入，产出确定性 2D 偏移。
///
/// 用 splitmix64 风格的整数混淆（纯整数运算，跨平台逐位一致；D-29 无 libm），
/// 高位映射到 [−1, 1] 的偏移量。这是"D 格锚"的落地——破碎的随机性
/// 完全由种子决定，同种子必得同碎片。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiteSeed {
    /// 全局种子。
    pub seed: u64,
}

impl SiteSeed {
    /// 构造。
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// splitmix64 单步混淆。
    #[inline]
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 第 i 个站点的确定性 [0,1) 标量流（两条独立通道）。
    fn rand01(&self, i: u64, channel: u64) -> f32 {
        let r = Self::mix(self.seed ^ Self::mix(i.wrapping_add(channel.wrapping_mul(0x1000_0000_1))));
        // 取高 24 位映射到 [0, 1)（f32 有效位）。
        ((r >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// 第 i 个站点的二维偏移，落在 [−1, 1]²。
    pub fn offset(&self, i: u64) -> (f32, f32) {
        let ox = self.rand01(i, 0) * 2.0 - 1.0;
        let oy = self.rand01(i, 1) * 2.0 - 1.0;
        (ox, oy)
    }
}

// ---------------------------------------------------------------------------
// 2D 凸多边形与半平面裁剪
// ---------------------------------------------------------------------------

/// 2D 凸多边形（顶点逆时针；多边形始终凸——半平面交集保持凸性）。
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon2 {
    /// 顶点（逆时针序）。
    pub verts: Vec<[f32; 2]>,
}

impl Polygon2 {
    /// 轴对齐矩形。
    pub fn rect(min: [f32; 2], max: [f32; 2]) -> Self {
        Self {
            verts: vec![
                [min[0], min[1]],
                [max[0], min[1]],
                [max[0], max[1]],
                [min[0], max[1]],
            ],
        }
    }

    /// 多边形面积（鞋带公式；逆时针为正）。
    pub fn area(&self) -> f32 {
        let n = self.verts.len();
        if n < 3 {
            return 0.0;
        }
        let mut s = 0.0f32;
        for i in 0..n {
            let a = self.verts[i];
            let b = self.verts[(i + 1) % n];
            s += a[0] * b[1] - b[0] * a[1];
        }
        0.5 * s.abs()
    }

    /// 质心（面积加权；退化时取顶点均值）。
    pub fn centroid(&self) -> [f32; 2] {
        let n = self.verts.len();
        if n == 0 {
            return [0.0, 0.0];
        }
        let mut a2 = 0.0f32;
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        for i in 0..n {
            let p = self.verts[i];
            let q = self.verts[(i + 1) % n];
            let cross = p[0] * q[1] - q[0] * p[1];
            a2 += cross;
            cx += (p[0] + q[0]) * cross;
            cy += (p[1] + q[1]) * cross;
        }
        if a2.abs() < 1e-12 {
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for v in &self.verts {
                sx += v[0];
                sy += v[1];
            }
            return [sx / n as f32, sy / n as f32];
        }
        let f = 1.0 / (3.0 * a2);
        [cx * f, cy * f]
    }
}

/// Sutherland–Hodgman 半平面裁剪：保留满足 `(p − m)·n ≤ 0` 的点。
///
/// 内性用点积符号判定（纯 IEEE 运算，确定性）。凸多边形裁剪后仍凸。
fn half_plane_clip(poly: &Polygon2, m: [f32; 2], n: [f32; 2]) -> Polygon2 {
    let inside = |p: [f32; 2]| (p[0] - m[0]) * n[0] + (p[1] - m[1]) * n[1] <= 0.0;
    let len = poly.verts.len();
    if len == 0 {
        return Polygon2 { verts: Vec::new() };
    }
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(len + 1);
    for i in 0..len {
        let cur = poly.verts[i];
        let nxt = poly.verts[(i + 1) % len];
        let cur_in = inside(cur);
        let nxt_in = inside(nxt);
        if cur_in {
            out.push(cur);
        }
        if cur_in != nxt_in {
            // 求线段 cur→nxt 与半平面边界的交点（参数 t ∈ [0,1]）。
            let dc = (cur[0] - m[0]) * n[0] + (cur[1] - m[1]) * n[1];
            let dn = (nxt[0] - m[0]) * n[0] + (nxt[1] - m[1]) * n[1];
            let denom = dc - dn;
            if denom.abs() > 1e-12 {
                let t = dc / denom;
                out.push([cur[0] + (nxt[0] - cur[0]) * t, cur[1] + (nxt[1] - cur[1]) * t]);
            }
        }
    }
    Polygon2 { verts: out }
}

// ---------------------------------------------------------------------------
// Voronoi 破碎
// ---------------------------------------------------------------------------

/// 破碎规格。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractureSpec {
    /// 包围盒最小角（xy）。
    pub min: [f32; 2],
    /// 包围盒最大角（xy）。
    pub max: [f32; 2],
    /// 站点数（碎片数）。
    pub site_count: usize,
    /// 站点噪声偏移幅度 ∈ [0, 1]（相对 1/√site_count 网格间距；0 = 规则格）。
    pub jitter: f32,
    /// 冲击点（xy）；站点密度随距其距离衰减。
    pub impact: [f32; 2],
    /// 站点种子。
    pub seed: SiteSeed,
}

impl FractureSpec {
    /// 校验。
    pub fn validate(&self) -> Result<(), PhysicsError> {
        if !(self.min[0] < self.max[0] && self.min[1] < self.max[1]) {
            return Err(PhysicsError {
                code: "P1014",
                location: "FractureSpec::validate".to_string(),
                problem: format!("包围盒非法：min = {:?}, max = {:?}", self.min, self.max),
                fix: "传 min.x < max.x 且 min.y < max.y 的包围盒".to_string(),
            });
        }
        if self.site_count == 0 {
            return Err(PhysicsError {
                code: "P1015",
                location: "FractureSpec::validate".to_string(),
                problem: "site_count = 0：站点数必须 ≥ 1".to_string(),
                fix: "传 site_count ≥ 1（碎片数 = 站点数）".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.jitter) {
            return Err(PhysicsError {
                code: "P1016",
                location: "FractureSpec::validate".to_string(),
                problem: format!("jitter = {}：噪声幅度必须 ∈ [0, 1]（NaN 同拒）", self.jitter),
                fix: "传 [0, 1] 区间内的噪声幅度".to_string(),
            });
        }
        Ok(())
    }
}

/// 一个碎片（Voronoi 单元）。
#[derive(Debug, Clone, PartialEq)]
pub struct Fragment {
    /// 碎片多边形（凸）。
    pub polygon: Polygon2,
    /// 生成该单元的站点。
    pub site: [f32; 2],
    /// 单元面积。
    pub area: f32,
    /// 单元质心。
    pub centroid: [f32; 2],
}

/// Voronoi 破碎结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Fracture {
    /// 碎片集（按站点序）。
    pub fragments: Vec<Fragment>,
    /// 站点集。
    pub sites: Vec<[f32; 2]>,
}

impl Fracture {
    /// 执行破碎：派生站点 → 逐单元半平面裁剪。
    ///
    /// 站点布局：在包围盒内铺 `⌈√n⌉ × ⌈√n⌉` 网格，按 jitter 加噪声偏移，
    /// 再按"距冲击点距离"做密度自适应重采样（近冲击点更密）。全部由种子决定。
    pub fn build(spec: FractureSpec) -> Result<Self, PhysicsError> {
        spec.validate()?;
        let sites = Self::sample_sites(&spec);
        let mut fragments = Vec::with_capacity(sites.len());
        for (i, &si) in sites.iter().enumerate() {
            // 单元多边形：从包围盒出发，逐其它站点的半平面裁剪。
            let mut poly = Polygon2::rect(spec.min, spec.max);
            for (j, &sj) in sites.iter().enumerate() {
                if i == j {
                    continue;
                }
                // 平分线中点与法向（法向 = sᵢ − sⱼ；保留 (p − m)·n ≤ 0）。
                let m = [(si[0] + sj[0]) * 0.5, (si[1] + sj[1]) * 0.5];
                let n = [si[0] - sj[0], si[1] - sj[1]];
                poly = half_plane_clip(&poly, m, n);
                if poly.verts.len() < 3 {
                    break; // 空单元（被裁没）；保留空碎片以维持站点对齐。
                }
            }
            let area = poly.area();
            let centroid = poly.centroid();
            fragments.push(Fragment { polygon: poly, site: si, area, centroid });
        }
        Ok(Self { fragments, sites })
    }

    /// 站点采样（密度自适应 + 种子噪声偏移；确定性）。
    fn sample_sites(spec: &FractureSpec) -> Vec<[f32; 2]> {
        // 网格边长 = 面积 / n 的平方根（均匀铺满）。
        let area = (spec.max[0] - spec.min[0]) * (spec.max[1] - spec.min[1]);
        let cell = (area / spec.site_count as f32).sqrt();
        let nx = ((spec.max[0] - spec.min[0]) / cell).ceil().max(1.0) as usize;
        let ny = ((spec.max[1] - spec.min[1]) / cell).ceil().max(1.0) as usize;
        // 密度自适应：距冲击点衰减的接受概率（近处密——先接受近处站点）。
        let mut out: Vec<[f32; 2]> = Vec::with_capacity(spec.site_count);
        let diag = ((spec.max[0] - spec.min[0]).hypot(spec.max[1] - spec.min[1])).max(1e-6);
        let mut k: u64 = 0;
        'outer: for gy in 0..ny {
            for gx in 0..nx {
                // 网格中心。
                let cx = spec.min[0] + (gx as f32 + 0.5) * cell;
                let cy = spec.min[1] + (gy as f32 + 0.5) * cell;
                let (ox, oy) = spec.seed.offset(k);
                k += 1;
                let jx = ox * spec.jitter * 0.5 * cell;
                let jy = oy * spec.jitter * 0.5 * cell;
                let px = (cx + jx).clamp(spec.min[0], spec.max[0]);
                let py = (cy + jy).clamp(spec.min[1], spec.max[1]);
                // 密度自适应：距冲击点归一化距离（近 → 保留概率高；远 → 低）。
                let dist = ((px - spec.impact[0]).hypot(py - spec.impact[1])) / diag;
                let (r, _) = spec.seed.offset(k);
                k += 1;
                let accept = 1.0 - dist.min(1.0);
                // r ∈ [−1,1] → [0,1]。
                let roll = (r + 1.0) * 0.5;
                if roll <= accept || out.len() >= spec.site_count {
                    out.push([px, py]);
                }
                if out.len() >= spec.site_count {
                    break 'outer;
                }
            }
        }
        // 兜底：保证至少 1 个站点。
        if out.is_empty() {
            out.push([
                (spec.min[0] + spec.max[0]) * 0.5,
                (spec.min[1] + spec.max[1]) * 0.5,
            ]);
        }
        out
    }

    /// 全部碎片面积之和（验收项：应 = 原包围盒面积，单位分解）。
    pub fn total_area(&self) -> f32 {
        self.fragments.iter().map(|f| f.area).sum()
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> FractureSpec {
        FractureSpec {
            min: [0.0, 0.0],
            max: [2.0, 2.0],
            site_count: 12,
            jitter: 0.5,
            impact: [0.5, 0.5],
            seed: SiteSeed::new(42),
        }
    }

    /// 面积和守恒（验收项）：所有碎片面积和 = 原包围盒面积（无缝覆盖）。
    #[test]
    fn fragments_tile_original_area() {
        let f = Fracture::build(spec()).unwrap();
        let box_area = (2.0 - 0.0) * (2.0 - 0.0);
        let sum = f.total_area();
        assert!(
            (sum - box_area).abs() < 1e-3,
            "碎片面积和 {sum} 应 = 原面积 {box_area}（无缝覆盖）"
        );
        // 每个碎片面积非负。
        for fr in &f.fragments {
            assert!(fr.area >= 0.0, "碎片面积不得为负：{}", fr.area);
        }
    }

    /// 确定性：同种子两遍破碎逐位一致。
    #[test]
    fn fracture_bitwise_deterministic() {
        let a = Fracture::build(spec()).unwrap();
        let b = Fracture::build(spec()).unwrap();
        assert_eq!(a, b, "同种子破碎应逐位一致");
    }

    /// 站点扰动生效：jitter 变化 → 碎片构型变化（噪声确实参与切割）。
    #[test]
    fn jitter_changes_fragments() {
        let mut s2 = spec();
        s2.jitter = 0.0;
        let mut s3 = spec();
        s3.jitter = 0.9;
        let a = Fracture::build(s2).unwrap();
        let b = Fracture::build(s3).unwrap();
        assert_ne!(a, b, "不同 jitter 应产生不同碎片构型");
        // 两者面积和都应守恒。
        assert!((a.total_area() - 4.0).abs() < 1e-3);
        assert!((b.total_area() - 4.0).abs() < 1e-3);
    }

    /// 半平面裁剪基本正确性：整体半平面裁掉一半 → 面积减半。
    #[test]
    fn half_plane_clip_halves_square() {
        let sq = Polygon2::rect([0.0, 0.0], [2.0, 2.0]);
        // 保留 x ≤ 1：法向 = sᵢ − sⱼ = (+1, 0)，中点 (1, 0)。
        let clipped = half_plane_clip(&sq, [1.0, 0.0], [1.0, 0.0]);
        assert!((clipped.area() - 2.0).abs() < 1e-4, "应裁掉一半面积");
    }

    /// 质心：矩形质心在中心。
    #[test]
    fn polygon_centroid_center() {
        let sq = Polygon2::rect([0.0, 0.0], [2.0, 2.0]);
        let c = sq.centroid();
        assert!((c[0] - 1.0).abs() < 1e-5 && (c[1] - 1.0).abs() < 1e-5);
    }

    /// 防呆：包围盒/站点数/jitter 非法 → 三段式报错。
    #[test]
    fn fracture_spec_validation() {
        let mut bad = spec();
        bad.max = [0.0, 0.0];
        assert_eq!(Fracture::build(bad).unwrap_err().code, "P1014");
        let mut bad = spec();
        bad.site_count = 0;
        assert_eq!(Fracture::build(bad).unwrap_err().code, "P1015");
        let mut bad = spec();
        bad.jitter = 1.5;
        assert_eq!(Fracture::build(bad).unwrap_err().code, "P1016");
    }

    /// 种子派生确定性：同种子同序号同偏移；不同序号不同（统计上）。
    #[test]
    fn site_seed_deterministic_stream() {
        let s = SiteSeed::new(7);
        assert_eq!(s.offset(3), SiteSeed::new(7).offset(3));
        let (a, _) = s.offset(1);
        let (b, _) = s.offset(2);
        assert!((a - b).abs() > 1e-6, "不同序号偏移应不同");
    }
}
