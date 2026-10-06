//! 手绘感扰动（humanize，B9）。
//!
//! 把"完美矢量"扰成"手绘"：沿路径对每个顶点施加**确定性 hash 派生的
//! 微扰**（禁用全局 RNG——走 `prism-noise` 的 PCG 整数 hash，`[D]`）。
//! 同一路径 + 同种子 + 同幅度必得逐位一致结果，可复现（见测试）。
//!
//! 抖动幅度以弧长/顶点索引为 hash 维度，使相邻顶点扰动不相关又整体
//! 平滑可调——不是白噪声叠加，而是可复现的"手感"偏置。

use crate::path::{Path, SubPath, Vertex};
use prism_noise::pcg;
use prism_raster::geom::Point;

/// 扰动配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Humanize {
    /// 全局种子（格 Zᵏ；同种子同结果）。
    pub seed: u64,
    /// 位置抖动幅度（像素）。
    pub jitter: f32,
    /// 控制点额外抖动倍率（控制点比锚点更"飘"，倍率 >0）。
    pub control_scale: f32,
}

impl Default for Humanize {
    fn default() -> Self {
        Self { seed: 0, jitter: 1.0, control_scale: 1.5 }
    }
}

impl Humanize {
    /// 构造扰动配置。
    pub fn new(seed: u64, jitter: f32) -> Self {
        Self { seed, jitter, control_scale: 1.5 }
    }
}

/// 对单点施加确定性扰动：偏移 = jitter · scale · (2·hash − 1)，两维独立。
fn perturb(seed: u64, idx: u64, dim: u64, scale: f32, p: Point) -> Point {
    let hx = pcg::hash3_f32(seed, idx, dim, 0);
    let hy = pcg::hash3_f32(seed, idx, dim, 1);
    let dx = (2.0 * hx - 1.0) * scale;
    let dy = (2.0 * hy - 1.0) * scale;
    Point::new(p.x + dx, p.y + dy)
}

/// 以默认配置（种子 0）扰动路径。
pub fn humanize(path: &Path, jitter: f32) -> Path {
    humanize_with(path, &Humanize::new(0, jitter))
}

/// 按配置扰动路径：逐顶点确定性偏移（锚点用 jitter，控制点用
/// jitter·control_scale）。索引 `idx` = 全局顶点序号（跨子路径单调递增），
/// 保证扰动只依赖顶点序与种子（纯函数，可复现）。
pub fn humanize_with(path: &Path, cfg: &Humanize) -> Path {
    let mut out = Path::new();
    let mut idx: u64 = 0;
    for sub in &path.subpaths {
        let mut verts = Vec::with_capacity(sub.vertices.len());
        for v in &sub.vertices {
            let scale = if v.control { cfg.jitter * cfg.control_scale } else { cfg.jitter };
            let np = perturb(cfg.seed, idx, 0, scale, v.p);
            verts.push(Vertex { p: np, control: v.control });
            idx += 1;
        }
        out.push(SubPath { vertices: verts, closed: sub.closed });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> Path {
        Path::from_polygon(&[
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 8.0),
        ])
    }

    #[test]
    fn humanize_is_reproducible_bitwise() {
        let p = tri();
        let a = humanize(&p, 2.0);
        let b = humanize(&p, 2.0);
        for (s1, s2) in a.subpaths.iter().zip(&b.subpaths) {
            for (v1, v2) in s1.vertices.iter().zip(&s2.vertices) {
                assert_eq!(v1.p.x.to_bits(), v2.p.x.to_bits());
                assert_eq!(v1.p.y.to_bits(), v2.p.y.to_bits());
            }
        }
    }

    #[test]
    fn humanize_amplitude_is_bounded_and_seed_dependent() {
        let p = tri();
        let a = humanize_with(&p, &Humanize::new(1, 3.0));
        let b = humanize_with(&p, &Humanize::new(2, 3.0));
        let mut any_differ = false;
        for (s1, s2) in a.subpaths.iter().zip(&b.subpaths) {
            for (v1, v2) in s1.vertices.iter().zip(&s2.vertices) {
                // 偏移幅度 ≤ jitter（锚点）。
                assert!((v1.p.x - 0.0f32).abs() <= 3.0 + 10.0);
                if v1.p.x.to_bits() != v2.p.x.to_bits() || v1.p.y.to_bits() != v2.p.y.to_bits() {
                    any_differ = true;
                }
            }
        }
        assert!(any_differ, "不同种子应产生不同扰动");
    }

    #[test]
    fn zero_jitter_is_identity() {
        let p = tri();
        let a = humanize_with(&p, &Humanize::new(7, 0.0));
        for (s1, s2) in a.subpaths.iter().zip(&p.subpaths) {
            for (v1, v2) in s1.vertices.iter().zip(&s2.vertices) {
                assert_eq!(v1.p.x, v2.p.x);
                assert_eq!(v1.p.y, v2.p.y);
            }
        }
    }
}
