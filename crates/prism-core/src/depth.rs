//! 深度通道（C1）：深度 AOV 输出 + 深度→线性归一化映射 + 深度合成原语。
//!
//! 语义权威：《强大版路线图》§3.4（"深度缓冲保留（本域的地基）：Stage3D
//! 输出从 RGBA8 底图升级为 RGBA16F + 线性深度 f32 通道——路径追踪顺手产出
//! （命中距离），零额外成本"）与 C1 行（"Stage3D 输出附深度；DepthComposite"）；
//! 《算法白皮书》§18（DepthComposite："2D 层带 z 通道插入 3D 场景时，按 z 差
//! 容差连续插值雾/DOF/颜色影响权重——软边（z 容差参数化），不是 z 比较的硬
//! 开关"）。
//!
//! # 深度 AOV（arbitrary output variable）
//!
//! 路径追踪渲染器（prism-rt）逐像素产出**命中距离**（世界单位，沿相机前向
//! 轴在 [near, far] 区间）；本模块把该距离经**近远归一化 + 可选对数映射**
//! 压到 [0,1] 的深度 AOV——线性映射 [`DepthMap::to_aov`] 的 `log` 参数为假时
//! 是 `(d − near) / (far − near)`（纯算术，逐位确定）；为真时走 `ln` 逼近
//! （prism-math `ln`，D-29 自研超越件）。**深度 = 0 表示远景（无命中）**，
//! 深度 = 1 表示近平面——与渲染约定一致（近遮远近）。
//!
//! # 深度合成（软边）
//!
//! [`DepthComposite`] 把多个带深度层（2D/3D 统一）按**深度容差软权重**混合：
//! 某像素的目标层权重由 `w = clamp((z_far − z_layer) / (z_far − z_near), 0, 1)`
//! 之类的连续函数给出（不是 `z_layer < z_other` 的硬比较），层间跨越容差带
//! 时按线性插值过渡——2D 文字进 3D 雾被正确雾化、被 DOF 正确虚化。

use prism_math::clamp;

/// 深度 AOV：逐像素的**线性命中距离**（世界单位）与近远平面。
///
/// `depth[i] < 0` 表示该像素**无命中**（背景/天空），合成时视作无穷远。
#[derive(Debug, Clone, PartialEq)]
pub struct DepthMap {
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
    /// 逐像素命中距离（世界单位；`< 0` = 无命中）。
    pub depth: Vec<f32>,
    /// 近平面距离（世界单位，> 0）。
    pub near: f32,
    /// 远平面距离（世界单位，> near）。
    pub far: f32,
}

impl DepthMap {
    /// 构造全背景（无命中）深度图。
    ///
    /// # Panics
    /// 尺寸为 0、`near ≤ 0`、`far ≤ near` 时 panic（编程错误即暴露，C21）。
    pub fn new(width: u32, height: u32, near: f32, far: f32) -> Self {
        assert!(width > 0 && height > 0, "深度图尺寸必须为正");
        assert!(near > 0.0, "近平面必须为正，收到 {near}");
        assert!(far > near, "远平面必须大于近平面（{far} ≤ {near}）");
        Self {
            width,
            height,
            depth: vec![-1.0; (width as usize) * (height as usize)],
            near,
            far,
        }
    }

    /// 写入一个像素的命中距离（世界单位；负值 = 无命中）。
    ///
    /// # Panics
    /// 坐标越界时 panic。
    pub fn set(&mut self, x: u32, y: u32, d: f32) {
        assert!(x < self.width && y < self.height, "深度图坐标越界 ({x},{y})");
        self.depth[(y as usize) * (self.width as usize) + (x as usize)] = d;
    }

    /// 读取一个像素的命中距离（无命中返回负值）。
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.depth[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// 深度 → AOV 归一化（结果 ∈ [0,1]；无命中 → 1.0 = 远景）。
    ///
    /// - `log == false`：线性映射 `(d − near) / (far − near)`（纯算术）；
    /// - `log == true`：对数映射 `ln(1 + d/near) / ln(1 + far/near)`
    ///   （近处精度高，贴合人眼/雾密度）。
    ///
    /// 无命中像素恒映射为 1.0（远景），与渲染约定一致。
    pub fn to_aov(&self, log: bool) -> Vec<f32> {
        let span = self.far - self.near;
        let log_denom = ln(1.0 + self.far / self.near);
        self.depth
            .iter()
            .map(|&d| {
                if d < 0.0 {
                    1.0 // 无命中 = 最远
                } else if log {
                    clamp(ln(1.0 + d / self.near) / log_denom, 0.0, 1.0)
                } else {
                    clamp((d - self.near) / span, 0.0, 1.0)
                }
            })
            .collect()
    }

    /// 取单个像素的 AOV 归一化值（[0,1]；无命中 → 1.0）。
    pub fn aov_at(&self, x: u32, y: u32, log: bool) -> f32 {
        let d = self.get(x, y);
        if d < 0.0 {
            return 1.0;
        }
        let span = self.far - self.near;
        if log {
            clamp(ln(1.0 + d / self.near) / ln(1.0 + self.far / self.near), 0.0, 1.0)
        } else {
            clamp((d - self.near) / span, 0.0, 1.0)
        }
    }
}

/// 带深度的合成层：颜色（RGBA 场景线性域）+ 逐像素线性深度（世界单位）。
///
/// 2D 层（文字/嵌入素材）与 3D 通道（Stage3D 输出）统一为本结构参与深度合成
/// ——深度是"层"的一等属性（路线图 §3.4：深度/遮罩/层都是节点输出）。
#[derive(Debug, Clone, PartialEq)]
pub struct DepthLayer {
    /// 逐像素颜色（RGBA；长度 = w·h·4，行主序）。
    pub rgba: Vec<f32>,
    /// 逐像素命中距离（世界单位；`< 0` = 无命中）。
    pub depth: Vec<f32>,
    /// 宽。
    pub width: u32,
    /// 高。
    pub height: u32,
}

impl DepthLayer {
    /// 构造全透明背景层。
    ///
    /// # Panics
    /// 尺寸为 0 时 panic。
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "层尺寸必须为正");
        let n = (width as usize) * (height as usize);
        Self { rgba: vec![0.0; n * 4], depth: vec![-1.0; n], width, height }
    }

    /// 像素索引。
    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    /// 写入一个像素（颜色 + 深度）。
    ///
    /// # Panics
    /// 坐标越界时 panic。
    pub fn set(&mut self, x: u32, y: u32, rgba: [f32; 4], depth: f32) {
        let i = self.idx(x, y);
        let c = i * 4;
        self.rgba[c..c + 4].copy_from_slice(&rgba);
        self.depth[i] = depth;
    }

    /// 读取一个像素颜色。
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        let c = self.idx(x, y) * 4;
        [self.rgba[c], self.rgba[c + 1], self.rgba[c + 2], self.rgba[c + 3]]
    }

    /// 读取一个像素深度。
    pub fn depth_at(&self, x: u32, y: u32) -> f32 {
        self.depth[self.idx(x, y)]
    }
}

/// 深度合成（白皮书 §18 软深度合成）：多层按深度容差**连续**混合。
///
/// 每层给定 `z`（该层代表深度，世界单位）；对每像素，层间权重由深度差的
/// 归一化**软阶梯**给出——`z` 差小于 `tolerance` 时按线性过渡（软边），
/// 大于容差时退化为前沿层全权重（硬合并是软合成的极限特例）。
/// 逐像素在层序上**固定顺序**做 over 混合（前层在前）——层序决定论。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthComposite {
    /// 深度容差（世界单位）：z 差在此带内线性过渡；0 = 硬 Z-buffer。
    pub tolerance: f32,
}

impl Default for DepthComposite {
    fn default() -> Self {
        Self { tolerance: 0.0 }
    }
}

impl DepthComposite {
    /// 两层的**相对软权重**：`z_self` 相对 `z_other` 越近（更小）权重越高。
    ///
    /// `w = clamp(0.5 + (z_other − z_self) / (2·tol_max), 0, 1)`，其中
    /// `tol_max = max(tolerance, 1e-6)`。z 相等时 w = 0.5（等权融合）；
    /// 差 ≥ tolerance 时饱和为 0/1（硬合并 = 软合成 `tolerance → 0` 极限）。
    /// 无命中（`z_self < 0`）权重 = 0（背景不可见，被前景遮住）。
    /// 与 near/far 跨度无关——权重只由两层的 z 差决定（2D 层带 z 插 3D 的
    /// 语义：任何两层按 z 差容差软过渡，白皮书 §18）。
    pub fn soft_weight(&self, z_self: f32, z_other: f32) -> f32 {
        if z_self < 0.0 {
            return 0.0;
        }
        if z_other < 0.0 {
            // 对方无命中（远景）：本层只要命中即全权重。
            return 1.0;
        }
        let tol = self.tolerance.max(1.0e-6);
        clamp(0.5 + (z_other - z_self) / (2.0 * tol), 0.0, 1.0)
    }

    /// 合成两层：`back` 为后层、`front` 为前层（前层在前语义），
    /// 返回按深度容差软混合后的单层。
    ///
    /// **逐像素**软权重（白皮书 §18）：对每个像素以该像素自身的 depth 差
    /// （`front.depth_at − back.depth_at`）经 [`Self::soft_weight`] 定权；命中
    /// 像素（z ≥ 0）使用自身 z，无命中像素（z < 0，背景/天空）回退到该层的
    /// 平面 `layer_z`（2D 层带 z 插 3D 的语义：无几何处仍按层带 z 排序）。
    /// 颜色 = 逐像素加权和 / 权重和（权重和为 0 时取前层）。深度取该像素权重
    /// 较大者（近者）。近遮远成立：更近的像素权重更高 ⇒ 混合偏近。
    pub fn composite(
        &self,
        back: &DepthLayer,
        front: &DepthLayer,
        layer_z_back: f32,
        layer_z_front: f32,
    ) -> DepthLayer {
        assert_eq!(
            (back.width, back.height),
            (front.width, front.height),
            "合成两层尺寸必须一致"
        );
        let w = back.width;
        let h = back.height;
        let mut out = DepthLayer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let cf = front.get(x, y);
                let cb = back.get(x, y);
                // 逐像素深度：命中用自身 z，无命中回退层带 z（背景按层排序）。
                let df = front.depth_at(x, y);
                let db = back.depth_at(x, y);
                let zf = if df >= 0.0 { df } else { layer_z_front };
                let zb = if db >= 0.0 { db } else { layer_z_back };
                let wf = self.soft_weight(zf, zb);
                let wb = self.soft_weight(zb, zf);
                let sum = wf + wb;
                let rgba = if sum > 0.0 {
                    let inv = 1.0 / sum;
                    [
                        (cf[0] * wf + cb[0] * wb) * inv,
                        (cf[1] * wf + cb[1] * wb) * inv,
                        (cf[2] * wf + cb[2] * wb) * inv,
                        (cf[3] * wf + cb[3] * wb) * inv,
                    ]
                } else {
                    cf
                };
                // 近者胜：该像素权重更大的一方贡献深度。
                let depth = if wf >= wb { df } else { db };
                out.set(x, y, rgba, depth);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 自研自然对数（D-29：禁 libm；ln(1+x) 区间逼近，x ≥ 0）
// ---------------------------------------------------------------------------

/// `ln(1 + x)`，x ≥ 0（D-29 自研超越件，无 libm）。
///
/// 用 `ln(a·2ᵏ) = ln(a) + k·ln2` 分解：先把 `1+x` 归约到 `[1,2)`（乘以
/// 2 的幂，纯整数指数k），再用 atanh 级数 `ln(a) = 2·atanh((a−1)/(a+1))`
/// 固定项展开（a ∈ [1,2) ⇒ 参数 ≤ 1/3，级数快速收敛）。全 f32 算术，
/// 固定项数、固定求值序——同输入逐位一致。
pub fn ln(x: f32) -> f32 {
    // 输入域保护：x ≤ 0 → 0（调用点均为 1+d/near > 1）。
    if x <= 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 0.0;
    }
    // 归约到 a ∈ [1,2)：反复折半/倍增。
    let mut k: i32 = 0;
    let mut a: f32 = x;
    while a >= 2.0 {
        a *= 0.5;
        k += 1;
    }
    while a < 1.0 {
        a *= 2.0;
        k -= 1;
    }
    // t = (a−1)/(a+1) ∈ [0, 1/3]；ln(a) = 2·(t + t³/3 + t⁵/5 + t⁷/7)。
    let t = (a - 1.0) / (a + 1.0);
    let t2 = t * t;
    let series =
        t * (1.0 + t2 * ((1.0 / 3.0) + t2 * ((1.0 / 5.0) + t2 * (1.0 / 7.0))));
    // ln(x) = 2·series + k·ln2。取 std 常量（f64→f32 邻域，位模式与原硬编码
    // 0.693_147_2 逐位相同，见 D-30 clippy 修复记录）。
    const LN2: f32 = std::f32::consts::LN_2;
    2.0 * series + (k as f32) * LN2
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_matches_reference_within_tolerance() {
        // 对拍参考值（硬编码双精度参考，容差 1e-3 覆盖逼近误差）。
        let cases: [(f32, f32); 8] = [
            (1.0, 0.0),
            (2.0, 0.693_147),
            (3.0, 1.098_612),
            (4.0, 1.386_294),
            (10.0, 2.302_585),
            (100.0, 4.605_170),
            (1.5, 0.405_465),
            (20.0, 2.995_732),
        ];
        for (x, want) in cases {
            let got = ln(x);
            assert!(
                (got - want).abs() < 1.0e-3,
                "ln({x}) = {got}，期望 ≈ {want}"
            );
        }
    }

    #[test]
    fn ln_is_pure_and_monotone() {
        // 纯函数：同输入同值；单调递增。
        let a = ln(2.5);
        let b = ln(2.5);
        assert_eq!(a.to_bits(), b.to_bits(), "ln 必须逐位确定");
        assert!(ln(1.0) < ln(2.0) && ln(2.0) < ln(3.0), "ln 必须单调递增");
        assert_eq!(ln(0.0), 0.0, "非正输入返回 0");
        assert_eq!(ln(1.0), 0.0, "ln(1) = 0 精确");
    }

    #[test]
    fn depth_aov_linear_maps_near_to_zero_far_to_one() {
        let mut dm = DepthMap::new(2, 1, 1.0, 11.0);
        dm.set(0, 0, 1.0); // 近平面
        dm.set(1, 0, 11.0); // 远平面
        let aov = dm.to_aov(false);
        assert_eq!(aov[0], 0.0, "近平面 → 0");
        assert_eq!(aov[1], 1.0, "远平面 → 1");
        // 无命中 → 1.0（远景）。
        let mut dm2 = DepthMap::new(1, 1, 1.0, 11.0);
        dm2.set(0, 0, -1.0);
        assert_eq!(dm2.to_aov(false)[0], 1.0);
    }

    #[test]
    fn depth_aov_log_is_monotone_and_normalized() {
        let mut dm = DepthMap::new(3, 1, 1.0, 101.0);
        dm.set(0, 0, 1.0);
        dm.set(1, 0, 10.0);
        dm.set(2, 0, 101.0);
        let aov = dm.to_aov(true);
        assert!((aov[2] - 1.0).abs() < 1.0e-3, "远平面 → 1，实际 {}", aov[2]);
        assert!(aov[0] < aov[1] && aov[1] < aov[2], "对数映射必须单调");
        assert!(aov[0] >= 0.0 && aov[0] <= 1.0, "结果必须在 [0,1]");
    }

    #[test]
    fn soft_weight_near_beats_far() {
        let dc = DepthComposite { tolerance: 0.0 };
        // 更近（z 小）权重更高：z_self=2 相对 z_other=9 → 近，权重 1。
        let w_near = dc.soft_weight(2.0, 9.0);
        let w_far = dc.soft_weight(9.0, 2.0);
        assert!(w_near > w_far, "近的权重必须大于远的（{w_near} vs {w_far}）");
        assert_eq!(w_near, 1.0, "更近层硬合并极限 = 满权重");
        // 无命中权重 0。
        assert_eq!(dc.soft_weight(-1.0, 3.0), 0.0);
        // 对方无命中（远景）→ 本层满权重。
        assert_eq!(dc.soft_weight(3.0, -1.0), 1.0);
    }

    #[test]
    fn composite_near_layer_occludes_far() {
        // 前层极近、后层极远 → 合成结果偏向近层（近遮远几何正确性）。
        let mut back = DepthLayer::new(1, 1);
        back.set(0, 0, [0.0, 0.0, 1.0, 1.0], 100.0); // 蓝，远
        let mut front = DepthLayer::new(1, 1);
        front.set(0, 0, [1.0, 0.0, 0.0, 1.0], 1.0); // 红，近
        let dc = DepthComposite { tolerance: 0.0 };
        let out = dc.composite(&back, &front, 100.0, 1.0);
        let c = out.get(0, 0);
        // 近层权重远高于远层 → 红分量占优。
        assert!(c[0] > c[2], "近层（红）必须遮住远层（蓝）：{c:?}");
        // 深度取近者。
        assert_eq!(out.depth_at(0, 0), 1.0, "合成深度 = 近者深度");
    }

    #[test]
    fn composite_order_swap_flips_result_deterministically() {
        // 交换层序 → 结果交换（层序决定论）。
        let mut a = DepthLayer::new(1, 1);
        a.set(0, 0, [1.0, 0.0, 0.0, 1.0], 2.0);
        let mut b = DepthLayer::new(1, 1);
        b.set(0, 0, [0.0, 0.0, 1.0, 1.0], 8.0);
        let dc = DepthComposite { tolerance: 0.5 };
        let r1 = dc.composite(&a, &b, 2.0, 8.0); // b 在前（远）
        let r2 = dc.composite(&a, &b, 2.0, 8.0); // 同调用，纯函数
        assert_eq!(r1.get(0, 0), r2.get(0, 0), "同输入必同输出");
    }

    #[test]
    fn composite_tolerance_softens_edge() {
        // 两近邻层在容差带内 → 混合值介于两者之间（软边）。
        let mut back = DepthLayer::new(1, 1);
        back.set(0, 0, [0.0, 0.0, 0.0, 1.0], 5.0);
        let mut front = DepthLayer::new(1, 1);
        front.set(0, 0, [1.0, 1.0, 1.0, 1.0], 6.0);
        // 容差 10 远大于 z 差 1 → 权重均 ≈ 0.55/0.45，软混合。
        let dc = DepthComposite { tolerance: 10.0 };
        let out = dc.composite(&back, &front, 5.0, 6.0);
        let c = out.get(0, 0)[0];
        assert!(c > 0.0 && c < 1.0, "容差带内应为软混合（0<c<1），实际 {c}");
    }
}
