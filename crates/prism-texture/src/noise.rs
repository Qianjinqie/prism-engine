//! 噪波纹理（B14 程序化纹理族：噪波）。
//!
//! 语义权威：《算法白皮书》§17（value / simplex / fbm 噪声族）。把噪声族
//! 暴露为纹理节点：输出灰度或双色（`color_a`/`color_b`）映射。
//!
//! 确定性：噪声全经 prism-noise 纯函数；同 seed 逐位一致。

use prism_io::fbuf::Fbuf16F;
use prism_noise::fbm::Fbm;
use prism_render::tile::TileEffect;

/// 噪波基元类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseTextureKind {
    /// 值噪声（格点 hash + quintic 插值）。
    Value,
    /// 单纯形噪声。
    Simplex,
    /// fBm（值噪声基）。
    FbmValue,
    /// fBm（单纯形基）。
    FbmSimplex,
}

/// 噪波纹理。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseTexture {
    /// 基元类型。
    pub kind: NoiseTextureKind,
    /// 空间频率（坐标缩放）。
    pub scale: f32,
    /// fbm 八度数（非 fbm 类型忽略）。
    pub octaves: u32,
    /// 种子。
    pub seed: u64,
    /// 输出下限色（噪声 -1）。
    pub color_a: [f32; 3],
    /// 输出上限色（噪声 +1）。
    pub color_b: [f32; 3],
}

impl Default for NoiseTexture {
    fn default() -> Self {
        NoiseTexture {
            kind: NoiseTextureKind::FbmSimplex,
            scale: 0.02,
            octaves: 5,
            seed: 0,
            color_a: [0.0, 0.0, 0.0],
            color_b: [1.0, 1.0, 1.0],
        }
    }
}

impl NoiseTexture {
    /// 采样噪声值 ∈ [-1, 1]。
    fn sample(&self, x: f32, y: f32) -> f32 {
        match self.kind {
            NoiseTextureKind::Value => prism_noise::value::value2(self.seed, x, y),
            NoiseTextureKind::Simplex => prism_noise::simplex::simplex2(self.seed, x, y),
            NoiseTextureKind::FbmValue => {
                Fbm::new(self.octaves, 2.0, 0.5).value2(self.seed, x, y)
            }
            NoiseTextureKind::FbmSimplex => {
                Fbm::new(self.octaves, 2.0, 0.5).simplex2(self.seed, x, y)
            }
        }
    }
}

impl TileEffect for NoiseTexture {
    fn halo_radius(&self) -> u32 {
        0
    }

    fn eval_window(&self, window: &Fbuf16F) -> Fbuf16F {
        let mut out = Fbuf16F::new(window.width, window.height);
        for y in 0..window.height {
            for x in 0..window.width {
                let n = self.sample(x as f32 * self.scale, y as f32 * self.scale);
                let t = prism_math::clamp((n + 1.0) * 0.5, 0.0, 1.0); // [0,1]
                let rgb = [
                    self.color_a[0] + (self.color_b[0] - self.color_a[0]) * t,
                    self.color_a[1] + (self.color_b[1] - self.color_a[1]) * t,
                    self.color_a[2] + (self.color_b[2] - self.color_a[2]) * t,
                ];
                out.set(x, y, [rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let mut state: u64 = 0x1122;
        let mut win = Fbuf16F::new(24, 24);
        for y in 0..24u32 {
            for x in 0..24u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) & 0xFFFF) as f32 / 65535.0;
                win.set(x, y, [v, v, v, 1.0]);
            }
        }
        for kind in [
            NoiseTextureKind::Value,
            NoiseTextureKind::Simplex,
            NoiseTextureKind::FbmValue,
            NoiseTextureKind::FbmSimplex,
        ] {
            let t = NoiseTexture { kind, ..NoiseTexture::default() };
            assert_eq!(t.eval_window(&win), t.eval_window(&win), "{kind:?} 纯函数");
            let s = NoiseTexture { seed: 7, ..t }.eval_window(&win);
            assert_ne!(t.eval_window(&win).pixels, s.pixels, "{kind:?} seed 必须去相关");
        }
    }

    #[test]
    fn output_bounded_and_alpha_one() {
        let out = NoiseTexture::default().eval_window(&Fbuf16F::new(20, 20));
        for y in 0..20u32 {
            for x in 0..20u32 {
                let c = out.get(x, y);
                assert_eq!(c[3], 1.0);
                for k in 0..3 {
                    assert!((0.0..=1.0).contains(&c[k]), "越界 {c:?}");
                }
            }
        }
    }
}
