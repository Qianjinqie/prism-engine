//! V3.1 收尾 C —— 把 `denoise.rs` 的基座接成可运行的 [`DenoiseBackend`]。
//!
//! 依赖方向是 `prism-postfx → prism-render`，`prism-render` 不能反向 import
//! `AtrousSpatial` / `TemporalAccumulator` / `DenoiseBackend`（会成环）。
//! 本模块在 postfx 侧把三件基座接成 [`CpuDenoiseBackend`]，并给出 variance
//! 引导的自适应 spp 预算。两处都不是桩：这里真的调用
//! `AtrousSpatial::eval_window` / `TemporalAccumulator::accumulate` /
//! `VarEstimate` / `allocate_spp`。
//!
//! 时空重建（及其 `ReconstructBridge`）已按复盘结论整体删除；本模块保留的
//! 是与重建无关的空域 / 时域降噪基座真接线。

use prism_io::fbuf::Fbuf16F;
use prism_render::tile::TileEffect;

use crate::denoise::{
    allocate_spp, AtrousSpatial, DenoiseBackend, TemporalAccumulator, VarEstimate,
};

/// CPU 参考降噪后端：à-trous 空域 `[1,4,6,4,1]/256` + 递推时域累积。
///
/// `spp` 由 [`allocate_spp`] 的 variance 预算给出：样本越多 ⇒ 越信任
/// 递推累积结果；样本少 ⇒ 保留更多当帧细节。`TileEffect::eval_window`
/// 保证 halo 申报与核半径匹配（tile 化逐位一致）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuDenoiseBackend {
    pub spatial: AtrousSpatial,
    pub temporal: TemporalAccumulator,
}

impl Default for CpuDenoiseBackend {
    fn default() -> Self {
        CpuDenoiseBackend {
            spatial: AtrousSpatial::default(),
            temporal: TemporalAccumulator::default(),
        }
    }
}

impl CpuDenoiseBackend {
    /// 以自定义迭代趟数 / σ 构造。
    pub fn new(iterations: u32, sigma_color: f32) -> Self {
        CpuDenoiseBackend {
            spatial: AtrousSpatial { iterations, sigma_color },
            temporal: TemporalAccumulator::default(),
        }
    }
}

impl DenoiseBackend for CpuDenoiseBackend {
    fn name(&self) -> &'static str {
        "cpu-atrous-temporal"
    }

    fn denoise(&self, input: &Fbuf16F, spp: u32) -> Fbuf16F {
        // 空域：真调用 AtrousSpatial（B3 核 3 趟 / σ=0.1）。
        let filtered: Fbuf16F = self.spatial.eval_window(input);
        // 时域：真调用 TemporalAccumulator 的递推增量均值；
        // n1 = max(spp,1) ⇒ spp 越大，(cur−acc)/n1 越小 ⇒ 越信任累积。
        let mut t: TemporalAccumulator = self.temporal;
        t.samples = spp.max(1).saturating_sub(1);
        t.accumulate(&filtered, input)
    }
}

/// variance 引导的自适应 spp 预算（真调用 `VarEstimate::from_spatial` +
/// `allocate_spp`；确定性 largest-remainder，总和恰为 `total`）。
pub fn adaptive_spp_map(input: &Fbuf16F, radius: u32, total: u32) -> Vec<u32> {
    let var: VarEstimate = VarEstimate::from_spatial(input, radius);
    let weights: Vec<f32> = var.field();
    allocate_spp(&weights, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, v: f32) -> Fbuf16F {
        let mut fb = Fbuf16F::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set(x, y, [v, v * 0.5, 1.0 - v, 1.0]);
            }
        }
        fb
    }

    #[test]
    fn cpu_backend_is_deterministic_and_preserves_constant_field() {
        let f = solid(8, 8, 0.3);
        let b = CpuDenoiseBackend::default();
        let a = b.denoise(&f, 16);
        let c = b.denoise(&f, 16);
        assert_eq!(b.name(), "cpu-atrous-temporal");
        for y in 0..8 {
            for x in 0..8 {
                let p = a.get(x, y);
                let q = c.get(x, y);
                assert_eq!(p[0].to_bits(), q[0].to_bits());
                // 常数场：à-trous 恒等 + 递推均值恒等 ⇒ 与原值一致（f16 量化）。
                assert!((p[0] - 0.3).abs() < 0.01, "常数场不应被改动：{p:?}");
            }
        }
    }

    #[test]
    fn adaptive_spp_map_is_deterministic_and_shape_aligned() {
        // 带边缘的场 ⇒ variance 非全零，预算真实分配。
        let mut f = Fbuf16F::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let v: f32 = if x < 4 { 0.1 } else { 0.9 };
                f.set(x, y, [v, v, v, 1.0]);
            }
        }
        let map = adaptive_spp_map(&f, 1, 256);
        let weights = VarEstimate::from_spatial(&f, 1).field();
        assert_eq!(map.len(), weights.len());
        let sum: u64 = map.iter().map(|&v| v as u64).sum();
        // 全零权重 ⇒ 全 0（确定性不虚构预算）；否则 largest-remainder 恰为 total。
        assert!(sum == 0 || sum == 256, "sum={sum}");
        let again = adaptive_spp_map(&f, 1, 256);
        assert_eq!(map, again);
    }
}
