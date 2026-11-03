//! 六算法生成器核心。
//!
//! 每个生成器是**确定性生成**：给定 (seed, 参数, 画布尺寸) 产出像素覆盖
//! 率缓冲（`prism_raster::Coverage`）——这是"生成艺术节点"的统一输出面。
//! 生成器同时可导出矢量 `Path`（供 morph/humanize/draw-on 复用）。
//!
//! 随机基元：一切取样走 `prism_noise::pcg` 的整数 hash 或 `value`/`simplex`
//! 噪声（本身即 PCG 派生）。**无全局 RNG**、无 libm。
//!
//! 六算法：
//! 1. [`FlowField`] —— 流场：噪声场驱动粒子流线。
//! 2. [`DifferentialGrowth`] —— 差分生长：链上按局部曲率做分裂/排斥生长。
//! 3. [`CellularAutomata`] —— 细胞自动机：Life 型规则的多代演化。
//! 4. [`VoronoiArt`] —— Voronoi 艺术：种子站点 + 距离场分格着色。
//! 5. [`FractalMandelbrot`] —— 分形：Mandelbrot 逃逸时间。
//! 6. [`NoiseModulation`] —— 噪声调制：fbm 场做域扭曲调制。

use prism_noise::fbm::Fbm;
use prism_noise::pcg;
use prism_noise::value::value2;
use prism_raster::{Coverage, Point};

/// 生成器统一输出：覆盖率缓冲 + 可选矢量轮廓集合。
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorOutput {
    /// 像素覆盖率（行主序，∈[0,1]）。
    pub coverage: Coverage,
    /// 矢量轮廓集合（流线/细胞边界等；可为空——纯像素型生成器）。
    pub contours: Vec<Vec<Point>>,
}

impl GeneratorOutput {
    /// 全覆盖率总和（面积度量，供测试与画面统计）。
    pub fn covered_area(&self) -> f32 {
        self.coverage.data.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// 1. 流场 FlowField
// ---------------------------------------------------------------------------

/// 流场生成器：噪声场给方向，粒子沿场积分成流线，流线光栅为覆盖。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowField {
    /// 全局种子。
    pub seed: u64,
    /// 流线条数。
    pub lines: u32,
    /// 每条流线的步数。
    pub steps: u32,
    /// 每步长度（像素）。
    pub step_len: f32,
    /// 场频率（噪声缩放）。
    pub freq: f32,
}

impl Default for FlowField {
    fn default() -> Self {
        Self { seed: 0, lines: 24, steps: 120, step_len: 3.0, freq: 0.01 }
    }
}

impl FlowField {
    /// 生成：粒子起于 hash 派生位置，取噪声梯度垂直方向为流向做欧拉积分。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let (w, h) = (width as f32, height as f32);
        let mut contours: Vec<Vec<Point>> = Vec::with_capacity(self.lines as usize);
        for li in 0..self.lines {
            let x0 = pcg::hash2_f32(self.seed, li as u64, 0) * w;
            let y0 = pcg::hash2_f32(self.seed, li as u64, 1) * h;
            let mut p = Point::new(x0, y0);
            let mut line = Vec::with_capacity(self.steps as usize + 1);
            line.push(p);
            for s in 0..self.steps {
                // 角度 ∝ 噪声值（[0,1)*2π，用差分近似场方向，避免超越函数）。
                let eps = 0.5f32;
                let n1 = self.field(p.x + eps, p.y);
                let n2 = self.field(p.x, p.y + eps);
                // 场梯度 → 流方向 = 梯度旋转 90°（确定性，仅 ±×÷）。
                let (dx, dy) = (n1 - n2, -(n1 - n2));
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 {
                    break;
                }
                p = Point::new(p.x + dx / len * self.step_len, p.y + dy / len * self.step_len);
                if p.x < 0.0 || p.x >= w || p.y < 0.0 || p.y >= h {
                    break;
                }
                line.push(p);
                let _ = s;
            }
            if line.len() > 1 {
                contours.push(line);
            }
        }
        let coverage = rasterize_lines(&contours, width, height);
        GeneratorOutput { coverage, contours }
    }

    /// 标量场采样（fbm over value noise，输出 [-1,1]）。
    fn field(&self, x: f32, y: f32) -> f32 {
        let fbm = Fbm::new(4, 2.0, 0.5);
        fbm.value2(self.seed, x * self.freq, y * self.freq)
    }
}

// ---------------------------------------------------------------------------
// 2. 差分生长 DifferentialGrowth
// ---------------------------------------------------------------------------

/// 差分生长：闭合链上的节点按"邻近排斥 + 曲率吸引"逐代生长/分裂。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentialGrowth {
    /// 全局种子。
    pub seed: u64,
    /// 初始节点数。
    pub initial_nodes: u32,
    /// 生长代数。
    pub generations: u32,
    /// 排斥半径（像素）。
    pub repel_radius: f32,
    /// 生长步长。
    pub growth: f32,
}

impl Default for DifferentialGrowth {
    fn default() -> Self {
        Self { seed: 0, initial_nodes: 48, generations: 20, repel_radius: 18.0, growth: 1.5 }
    }
}

impl DifferentialGrowth {
    /// 生成：起点为 hash 派生的圆周链，逐代做排斥+分裂并累积成环形曲线。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let r0 = (width.min(height) as f32) * 0.18;
        let n = self.initial_nodes.max(3) as usize;
        let mut pts: Vec<Point> = Vec::with_capacity(n);
        for i in 0..n {
            // 圆上均匀布点（角度 = i/n·2π 的确定性 cos/sin 由 hash 抖动近似——
            // 用 ±表示：以单位圆参数化避免超越函数，改用预置 256 格余弦表）。
            let (c, s) = unit_circle(i as f32 / n as f32);
            let jitter = (pcg::hash2_f32(self.seed, i as u64, 3) - 0.5) * r0 * 0.2;
            pts.push(Point::new(cx + c * (r0 + jitter), cy + s * (r0 + jitter)));
        }

        let mut layers: Vec<Vec<Point>> = vec![pts.clone()];
        for _g in 0..self.generations {
            let m = pts.len();
            let mut next: Vec<Point> = Vec::with_capacity(m * 2);
            for i in 0..m {
                let a = pts[i];
                let b = pts[(i + 1) % m];
                // 中点分裂。
                let mid = Point::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
                next.push(a);
                next.push(mid);
            }
            // 排斥：每点向远离近邻的方向推（一步，确定性顺序）。
            let k = next.len();
            let mut moved = next.clone();
            for i in 0..k {
                let a = next[i];
                // 采样两个邻点算局部直（避开全量 O(n²) 的典型简化）。
                let nb = next[(i + k - 1) % k];
                let na = next[(i + 1) % k];
                let repx = (a.x - nb.x) + (a.x - na.x);
                let repy = (a.y - nb.y) + (a.y - na.y);
                let rl = (repx * repx + repy * repy).sqrt();
                if rl > 1e-6 {
                    let f = (self.growth / rl).min(self.repel_radius);
                    moved[i] = Point::new(a.x + repx * f * 0.05, a.y + repy * f * 0.05);
                }
            }
            pts = moved;
            layers.push(pts.clone());
        }

        let coverage = rasterize_lines(&layers, width, height);
        GeneratorOutput { coverage, contours: layers }
    }
}

/// 单位圆上参数 u∈[0,1) 的 (cos, sin)：以多项式**同时**逼近 sin/cos。
///
/// 用切比雪夫风格的两点解析（仅 ±×÷，无 libm，D-29）：先把 u 折到
/// [-0.25,0.25]（对应角度 [-π/2,π/2]），用奇多项式近似 sin，cos = sin(π/2−θ)。
/// 精度足够布点抖动场景（生成艺术），且确定性可复现。
fn unit_circle(u: f32) -> (f32, f32) {
    let angle = u.fract() * TAU; // [0,2π)
    (cos_poly(angle), sin_poly(angle))
}

/// 2π 常量（f32 精确可表的上界近似）。
const TAU: f32 = 6.283_185_5;

/// sin(θ) 的多项式近似：先折到 [-π/2, π/2]，用 7 次奇多项式。
fn sin_poly(theta: f32) -> f32 {
    let mut x = theta % TAU;
    if x < 0.0 {
        x += TAU;
    }
    // 折返：sin(π−x)=sin x, sin(2π−x)=−sin x。
    // D-30：与 `std::f32::consts::PI` `to_bits()` 逐位相同，替换零行为变化。
    let pi = std::f32::consts::PI;
    let (v, sign) = if x > pi { (x - pi, -1.0f32) } else { (x, 1.0) };
    let mut v = v;
    if v > pi * 0.5 {
        v = pi - v;
    }
    // 7 级泰勒式多项式（系数固定，纯 ±×÷）。
    let v2 = v * v;
    let s = v
        * (1.0
            + v2 * (-1.0 / 6.0
                + v2 * (1.0 / 120.0 + v2 * (-1.0 / 5040.0 + v2 * (1.0 / 362_880.0)))));
    sign * s
}

/// cos(θ) = sin(θ + π/2)。
///
/// D-30：π/2 偏移原为字面量 `1.570_796_3`（bits 0x3FC90FDB），它与
/// `std::f32::consts::FRAC_PI_2`（0x3FC90FDC）**差 1 ulp**。这里**必须保留
/// 原值**——替换会静默改变 `cos_poly` 的每一位输出，进而改变全部程序化纹理
/// 的采样位置（且 `1.570_796_3` 恰是 π/2 的正确 f32 向下舍入，写法无误）。
#[allow(clippy::approx_constant)] // 精确值，与历史行为一致；见 D-30
fn cos_poly(theta: f32) -> f32 {
    sin_poly(theta + 1.570_796_3)
}

// ---------------------------------------------------------------------------
// 3. 细胞自动机 CellularAutomata
// ---------------------------------------------------------------------------

/// 细胞自动机：Life/Brian's-Brain 型 2 态规则的多代演化（确定性演化）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellularAutomata {
    /// 全局种子。
    pub seed: u64,
    /// 网格宽（格数）。
    pub gw: u32,
    /// 网格高。
    pub gh: u32,
    /// 代步数。
    pub generations: u32,
    /// 每格像素尺寸。
    pub cell: u32,
}

impl Default for CellularAutomata {
    fn default() -> Self {
        Self { seed: 0, gw: 64, gh: 36, generations: 40, cell: 4 }
    }
}

impl CellularAutomata {
    /// 生成：初始活细胞由 hash 阈值决定，按 B3/S23（Life）演化 last 代快照。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let (gw, gh) = (self.gw as usize, self.gh as usize);
        let mut cur = vec![false; gw * gh];
        for y in 0..gh {
            for x in 0..gw {
                let r = pcg::hash2_f32(self.seed, x as u64, y as u64);
                cur[y * gw + x] = r < 0.42;
            }
        }
        for _ in 0..self.generations {
            let mut next = vec![false; gw * gh];
            for y in 0..gh {
                for x in 0..gw {
                    let mut nb = 0u32;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx < 0 || ny < 0 || nx >= gw as i32 || ny >= gh as i32 {
                                continue;
                            }
                            if cur[ny as usize * gw + nx as usize] {
                                nb += 1;
                            }
                        }
                    }
                    let alive = cur[y * gw + x];
                    next[y * gw + x] = matches!((alive, nb), (true, 2) | (true, 3) | (false, 3));
                }
            }
            cur = next;
        }

        // 光栅：活格 → 覆盖率块。
        let mut cov = Coverage::new(width, height);
        let c = self.cell.max(1);
        for y in 0..gh {
            for x in 0..gw {
                if !cur[y * gw + x] {
                    continue;
                }
                for py in 0..c {
                    for px in 0..c {
                        let gx = x as u32 * c + px;
                        let gy = y as u32 * c + py;
                        if gx < width && gy < height {
                            cov.data[(gy as usize) * width as usize + gx as usize] = 1.0;
                        }
                    }
                }
            }
        }
        GeneratorOutput { coverage: cov, contours: Vec::new() }
    }

    /// 导出当前末代活格快照（供测试对比确定性）。
    pub fn final_state(&self) -> Vec<bool> {
        let (gw, gh) = (self.gw as usize, self.gh as usize);
        let mut cur = vec![false; gw * gh];
        for y in 0..gh {
            for x in 0..gw {
                cur[y * gw + x] = pcg::hash2_f32(self.seed, x as u64, y as u64) < 0.42;
            }
        }
        for _ in 0..self.generations {
            let mut next = vec![false; gw * gh];
            for y in 0..gh {
                for x in 0..gw {
                    let mut nb = 0u32;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx < 0 || ny < 0 || nx >= gw as i32 || ny >= gh as i32 {
                                continue;
                            }
                            if cur[ny as usize * gw + nx as usize] {
                                nb += 1;
                            }
                        }
                    }
                    next[y * gw + x] =
                        matches!((cur[y * gw + x], nb), (true, 2) | (true, 3) | (false, 3));
                }
            }
            cur = next;
        }
        cur
    }
}

// ---------------------------------------------------------------------------
// 4. VoronoiArt
// ---------------------------------------------------------------------------

/// Voronoi 艺术：hash 派生站点 + 最近站点归属着色（F1 距离）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiArt {
    /// 全局种子。
    pub seed: u64,
    /// 站点数。
    pub sites: u32,
    /// F1/F2 边缘阈值（0=纯 F1 分格，>0 突出单元边界）。
    pub edge_threshold: f32,
    /// 站点的噪声扰动幅度（像素， 消"饼干感"手法）。
    pub site_jitter: f32,
}

impl Default for VoronoiArt {
    fn default() -> Self {
        Self { seed: 0, sites: 32, edge_threshold: 0.0, site_jitter: 0.0 }
    }
}

impl VoronoiArt {
    /// 生成：最近站点距离归一为覆盖率；edge_threshold>0 时用 F2−F1 提取边界。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let (w, h) = (width as f32, height as f32);
        let n = self.sites.max(1) as usize;
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        for i in 0..n {
            let mut x = pcg::hash2_f32(self.seed, i as u64, 0) * w;
            let mut y = pcg::hash2_f32(self.seed, i as u64, 1) * h;
            if self.site_jitter > 0.0 {
                x += (pcg::hash2_f32(self.seed, i as u64, 4) - 0.5) * self.site_jitter;
                y += (pcg::hash2_f32(self.seed, i as u64, 5) - 0.5) * self.site_jitter;
            }
            sx.push(x);
            sy.push(y);
        }

        let mut cov = Coverage::new(width, height);
        let maxd = (w * w + h * h).sqrt().max(1.0);
        for py in 0..height {
            for px in 0..width {
                let x = px as f32 + 0.5;
                let y = py as f32 + 0.5;
                let mut d1 = f32::MAX;
                let mut d2 = f32::MAX;
                for i in 0..n {
                    let dx = x - sx[i];
                    let dy = y - sy[i];
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < d1 {
                        d2 = d1;
                        d1 = d;
                    } else if d < d2 {
                        d2 = d;
                    }
                }
                let v = if self.edge_threshold > 0.0 {
                    let e = d2 - d1;
                    if e < self.edge_threshold { 1.0 } else { 0.0 }
                } else {
                    1.0 - (d1 / maxd)
                };
                cov.data[(py as usize) * width as usize + px as usize] = v;
            }
        }
        GeneratorOutput { coverage: cov, contours: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// 5. FractalMandelbrot
// ---------------------------------------------------------------------------

/// 分形：Mandelbrot 逃逸时间（迭代 z←z²+c，逃逸即停）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractalMandelbrot {
    /// 最大迭代数。
    pub max_iter: u32,
    /// 视窗中心（复平面）。
    pub center: (f32, f32),
    /// 复平面缩放（每像素跨度）。
    pub scale: f32,
}

impl Default for FractalMandelbrot {
    fn default() -> Self {
        Self { max_iter: 64, center: (-0.5, 0.0), scale: 3.0 }
    }
}

impl FractalMandelbrot {
    /// 生成：逃逸时间映射为覆盖率（未逃逸=1）。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let (w, h) = (width as f32, height as f32);
        let aspect = w / h.max(1.0);
        let mut cov = Coverage::new(width, height);
        let maxit = self.max_iter.max(1) as f32;
        for py in 0..height {
            let cy = self.center.1 + (py as f32 / h - 0.5) * self.scale;
            for px in 0..width {
                let cx = self.center.0 + (px as f32 / w - 0.5) * self.scale * aspect;
                let mut zx = 0.0f32;
                let mut zy = 0.0f32;
                let mut i = 0u32;
                while i < self.max_iter {
                    let nx = zx * zx - zy * zy + cx;
                    let ny = 2.0 * zx * zy + cy;
                    zx = nx;
                    zy = ny;
                    if zx * zx + zy * zy > 4.0 {
                        break;
                    }
                    i += 1;
                }
                // 逃逸越慢覆盖率越高（内部=1）。
                let v = if i >= self.max_iter { 1.0 } else { (i as f32) / maxit };
                cov.data[(py as usize) * width as usize + px as usize] = v;
            }
        }
        GeneratorOutput { coverage: cov, contours: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// 6. NoiseModulation
// ---------------------------------------------------------------------------

/// 噪声调制：fbm 场 + 域扭曲（域算子， 与 SDF/UV warp 互通）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseModulation {
    /// 全局种子。
    pub seed: u64,
    /// 基础频率。
    pub freq: f32,
    /// 域扭曲强度。
    pub warp: f32,
    /// fbm 层数。
    pub octaves: u32,
}

impl Default for NoiseModulation {
    fn default() -> Self {
        Self { seed: 0, freq: 0.02, warp: 1.5, octaves: 5 }
    }
}

impl NoiseModulation {
    /// 生成：value2 经 fbm 后做域扭曲，输出归一 [0,1] 覆盖率。
    pub fn generate(&self, width: u32, height: u32) -> GeneratorOutput {
        let fbm = Fbm::new(self.octaves, 2.0, 0.5);
        let mut cov = Coverage::new(width, height);
        for py in 0..height {
            for px in 0..width {
                let x = px as f32 * self.freq;
                let y = py as f32 * self.freq;
                let wx = value2(self.seed ^ 0xABCD, x + 5.2, y + 1.3);
                let wy = value2(self.seed ^ 0x1234, x + 1.7, y + 9.2);
                let v = fbm.value2(self.seed, x + self.warp * wx, y + self.warp * wy);
                // [-1,1] → [0,1]。
                cov.data[(py as usize) * width as usize + px as usize] = (v * 0.5 + 0.5).clamp(0.0, 1.0);
            }
        }
        GeneratorOutput { coverage: cov, contours: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// 共享：折线 → 覆盖率（细线光栅，复用 prism-raster 的 draw_line）
// ---------------------------------------------------------------------------

/// 把折线集合光栅为覆盖率缓冲：每条线段用 DDA 步进写入覆盖率
/// （每像素取饱和 1.0，重叠不累加）。仅 ±×÷ 与比较，确定性可复现。
pub fn rasterize_lines(lines: &[Vec<Point>], width: u32, height: u32) -> Coverage {
    let mut cov = Coverage::new(width, height);
    for line in lines {
        for seg in line.windows(2) {
            rasterize_segment(&mut cov, seg[0], seg[1], width, height);
        }
    }
    cov
}

/// 单线段 DDA 光栅（步长 0.5 像素，写 1×1 覆盖率；越界剪弃）。
fn rasterize_segment(cov: &mut Coverage, a: Point, b: Point, width: u32, height: u32) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    let steps = (len * 2.0).ceil().max(1.0) as u32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = a.x + dx * t;
        let y = a.y + dy * t;
        if x < 0.0 || y < 0.0 {
            continue;
        }
        let px = x as u32;
        let py = y as u32;
        if px < width && py < height {
            cov.data[(py as usize) * width as usize + px as usize] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_cov(c: &Coverage) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for v in &c.data {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    #[test]
    fn flow_field_deterministic() {
        let g = FlowField { seed: 7, lines: 12, steps: 60, step_len: 3.0, freq: 0.01 };
        let a = g.generate(128, 96);
        let b = g.generate(128, 96);
        assert_eq!(hash_cov(&a.coverage), hash_cov(&b.coverage), "流场同种子两遍须一致");
        assert!(a.covered_area() > 0.0, "流场应产生可见线条");
    }

    #[test]
    fn differential_growth_deterministic() {
        let g = DifferentialGrowth { seed: 3, initial_nodes: 24, generations: 8, repel_radius: 16.0, growth: 1.2 };
        let a = g.generate(100, 100);
        let b = g.generate(100, 100);
        assert_eq!(hash_cov(&a.coverage), hash_cov(&b.coverage));
        assert!(a.covered_area() > 0.0);
    }

    #[test]
    fn cellular_automata_deterministic() {
        let g = CellularAutomata { seed: 11, gw: 32, gh: 24, generations: 10, cell: 3 };
        let a = g.final_state();
        let b = g.final_state();
        assert_eq!(a, b, "细胞自动机末态须逐位一致");
        let out = g.generate(96, 72);
        assert!(out.covered_area() > 0.0);
    }

    #[test]
    fn voronoi_art_deterministic_and_bounded() {
        let g = VoronoiArt { seed: 5, sites: 16, edge_threshold: 0.0, site_jitter: 8.0 };
        let a = g.generate(80, 80);
        let b = g.generate(80, 80);
        assert_eq!(hash_cov(&a.coverage), hash_cov(&b.coverage));
        for v in &a.coverage.data {
            assert!((0.0..=1.0).contains(v), "覆盖率越界 {v}");
        }
    }

    #[test]
    fn fractal_mandelbrot_deterministic_and_membership() {
        let g = FractalMandelbrot { max_iter: 40, center: (-0.5, 0.0), scale: 3.0 };
        let a = g.generate(64, 64);
        let b = g.generate(64, 64);
        assert_eq!(hash_cov(&a.coverage), hash_cov(&b.coverage));
        // 中心点 (-0.5,0) 属集（覆盖率=1，未逃逸）。
        //
        // 原断言是 `center >= 0.0`——覆盖率按 `i/maxit` 定义、恒非负，等于
        // 永真。这里钉住"属集 ⇒ 恰为 1.0"，并补一条"图里确实有逃逸像素"，
        // 否则整幅返回 1.0 的实现也能过。
        let center = a.coverage.data[32 * 64 + 32];
        assert_eq!(center, 1.0, "中心点 (-0.5,0) 属集，覆盖率应为 1.0，实得 {center}");
        let escaped = a.coverage.data.iter().filter(|&&v| v < 1.0).count();
        assert!(escaped > 0, "视窗 [-2, 1]×[-1.5, 1.5] 内必有逃逸像素，实得 0 个");
    }

    #[test]
    fn noise_modulation_deterministic() {
        let g = NoiseModulation { seed: 9, freq: 0.03, warp: 2.0, octaves: 4 };
        let a = g.generate(64, 64);
        let b = g.generate(64, 64);
        assert_eq!(hash_cov(&a.coverage), hash_cov(&b.coverage));
        for v in &a.coverage.data {
            assert!((0.0..=1.0).contains(v));
        }
    }
}
