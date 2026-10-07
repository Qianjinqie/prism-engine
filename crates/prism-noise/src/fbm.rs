//! Fractal Brownian motion — octave-summed spectral synthesis .
//!
//! `fbm(p) = Σᵢ ampᵢ · base(freqᵢ · p, seedᵢ)` with `freqᵢ = lacunarityⁱ`,
//! `ampᵢ = gainⁱ`, `seedᵢ = seed + i` (octave seeds differ by the octave index;
//! the PCG folding inside the base noise decorrelates them). The sum is
//! normalized by the total amplitude `Σᵢ ampᵢ ≥ 1`, so the output stays in
//! `[-1, 1]` for any octave count. With `octaves = 1` the result is exactly
//! the base noise (freq = 1, amp = 1, norm = 1) — asserted bitwise by a test.
//!
//! Practical parameters:octaves 4–6, lacunarity ≈ 2,
//! gain ≈ 0.5. The caller must keep `lacunarity^octaves` within the finite
//! f32 range (saturating casts make overflow deterministic but meaningless).

use crate::{simplex, value};

/// fbm parameters: octave count, per-octave frequency multiplier and
/// amplitude decay. Plain data — all fields are public and part of the API.
pub struct Fbm {
    /// Number of octaves summed; `0` evaluates to the empty sum `0.0`.
    pub octaves: u32,
    /// Frequency multiplier between consecutive octaves (> 0).
    pub lacunarity: f32,
    /// Amplitude decay between consecutive octaves (> 0).
    pub gain: f32,
}

impl Fbm {
    /// Constructs fbm parameters (debug-checked: `lacunarity > 0`, `gain > 0`).
    #[inline]
    pub fn new(octaves: u32, lacunarity: f32, gain: f32) -> Self {
        debug_assert!(lacunarity > 0.0, "lacunarity must be positive");
        debug_assert!(gain > 0.0, "gain must be positive");
        Self {
            octaves,
            lacunarity,
            gain,
        }
    }

    /// 2D fbm over the value-noise basis; output in `[-1, 1]`.
    pub fn value2(&self, seed: u64, x: f32, y: f32) -> f32 {
        self.synth2(seed, x, y, value::value2)
    }

    /// 3D fbm over the value-noise basis; output in `[-1, 1]`.
    pub fn value3(&self, seed: u64, x: f32, y: f32, z: f32) -> f32 {
        self.synth3(seed, x, y, z, value::value3)
    }

    /// 4D fbm over the value-noise basis; output in `[-1, 1]`.
    pub fn value4(&self, seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 {
        self.synth4(seed, x, y, z, w, value::value4)
    }

    /// 2D fbm over the simplex basis; output in `[-1, 1]`.
    pub fn simplex2(&self, seed: u64, x: f32, y: f32) -> f32 {
        self.synth2(seed, x, y, simplex::simplex2)
    }

    /// 3D fbm over the simplex basis; output in `[-1, 1]`.
    pub fn simplex3(&self, seed: u64, x: f32, y: f32, z: f32) -> f32 {
        self.synth3(seed, x, y, z, simplex::simplex3)
    }

    /// 4D fbm over the simplex basis; output in `[-1, 1]`.
    pub fn simplex4(&self, seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 {
        self.synth4(seed, x, y, z, w, simplex::simplex4)
    }

    /// Octave accumulation over a 2D basis: fixed-order summation, amplitude
    /// `gainⁱ`, frequency `lacunarityⁱ`, normalized by the total amplitude.
    fn synth2(&self, seed: u64, x: f32, y: f32, base: fn(u64, f32, f32) -> f32) -> f32 {
        if self.octaves == 0 {
            return 0.0; // empty sum
        }
        let (mut freq, mut amp) = (1.0_f32, 1.0_f32);
        let (mut sum, mut norm) = (0.0_f32, 0.0_f32);
        for octave in 0..self.octaves {
            sum += amp * base(seed.wrapping_add(octave as u64), x * freq, y * freq);
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        sum / norm
    }

    /// Octave accumulation over a 3D basis (same fixed order as [`Fbm::synth2`]).
    fn synth3(
        &self,
        seed: u64,
        x: f32,
        y: f32,
        z: f32,
        base: fn(u64, f32, f32, f32) -> f32,
    ) -> f32 {
        if self.octaves == 0 {
            return 0.0;
        }
        let (mut freq, mut amp) = (1.0_f32, 1.0_f32);
        let (mut sum, mut norm) = (0.0_f32, 0.0_f32);
        for octave in 0..self.octaves {
            sum += amp * base(seed.wrapping_add(octave as u64), x * freq, y * freq, z * freq);
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        sum / norm
    }

    /// Octave accumulation over a 4D basis (same fixed order as [`Fbm::synth2`]).
    fn synth4(
        &self,
        seed: u64,
        x: f32,
        y: f32,
        z: f32,
        w: f32,
        base: fn(u64, f32, f32, f32, f32) -> f32,
    ) -> f32 {
        if self.octaves == 0 {
            return 0.0;
        }
        let (mut freq, mut amp) = (1.0_f32, 1.0_f32);
        let (mut sum, mut norm) = (0.0_f32, 0.0_f32);
        for octave in 0..self.octaves {
            sum += amp
                * base(
                    seed.wrapping_add(octave as u64),
                    x * freq,
                    y * freq,
                    z * freq,
                    w * freq,
                );
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        sum / norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcg;

    const SEED: u64 = 0xFB00_FB00_0000_0001;

    /// Deterministic pseudo-random point over `[-16, 16)` per axis.
    fn point(seed: u64, i: u64, axis: u64) -> f32 {
        pcg::hash2_f32(seed, i, axis) * 32.0 - 16.0
    }

    /// Gate 4: with one octave, fbm is exactly the base noise (bitwise).
    #[test]
    fn single_octave_equals_base() {
        let fbm = Fbm::new(1, 2.0, 0.5);
        for i in 0..1000_u64 {
            let (x, y, z, w) = (
                point(SEED, i, 0),
                point(SEED, i, 1),
                point(SEED, i, 2),
                point(SEED, i, 3),
            );
            assert_eq!(fbm.value2(SEED, x, y), value::value2(SEED, x, y));
            assert_eq!(fbm.value3(SEED, x, y, z), value::value3(SEED, x, y, z));
            assert_eq!(fbm.value4(SEED, x, y, z, w), value::value4(SEED, x, y, z, w));
            assert_eq!(fbm.simplex2(SEED, x, y), simplex::simplex2(SEED, x, y));
            assert_eq!(fbm.simplex3(SEED, x, y, z), simplex::simplex3(SEED, x, y, z));
            assert_eq!(fbm.simplex4(SEED, x, y, z, w), simplex::simplex4(SEED, x, y, z, w));
        }
    }

    /// Gate 4: octave amplitudes follow `gainⁱ` exactly — the k-octave fbm is
    /// bitwise the manually expanded normalized sum (amp 1, gain, gain², …).
    #[test]
    fn octave_amplitudes_follow_gain() {
        let (lacunarity, gain) = (2.0_f32, 0.5_f32);
        let fbm2 = Fbm::new(2, lacunarity, gain);
        let fbm3 = Fbm::new(3, lacunarity, gain);
        for i in 0..500_u64 {
            let (x, y, z) = (point(SEED, i, 0), point(SEED, i, 1), point(SEED, i, 2));
            // 2 octaves of the value basis:
            let n1 = value::value2(SEED, x, y);
            let n2 = value::value2(SEED.wrapping_add(1), x * lacunarity, y * lacunarity);
            let expected2 = n1 + gain * n2; // norm = 1 + gain
            assert_eq!(fbm2.value2(SEED, x, y), expected2 / (1.0 + gain));
            // 3 octaves of the simplex basis:
            let m1 = simplex::simplex3(SEED, x, y, z);
            let m2 = simplex::simplex3(SEED.wrapping_add(1), x * lacunarity, y * lacunarity, z * lacunarity);
            let m3 = simplex::simplex3(
                SEED.wrapping_add(2),
                x * lacunarity * lacunarity,
                y * lacunarity * lacunarity,
                z * lacunarity * lacunarity,
            );
            let expected3 = m1 + gain * m2 + (gain * gain) * m3; // norm = 1 + gain + gain²
            assert_eq!(fbm3.simplex3(SEED, x, y, z), expected3 / (1.0 + gain + gain * gain));
        }
    }

    /// Gate 3/4: output stays in `[-1, 1]` (amplitude-normalized) and the
    /// range is used; determinism holds over 10000 repeated calls.
    #[test]
    fn range_and_determinism() {
        let fbm = Fbm::new(5, 2.0, 0.5);
        let (mut min, mut max) = (1.0_f32, -1.0_f32);
        for i in 0..50_000_u64 {
            let (x, y, z, w) = (
                point(SEED, i, 0),
                point(SEED, i, 1),
                point(SEED, i, 2),
                point(SEED, i, 3),
            );
            for v in [
                fbm.value2(SEED, x, y),
                fbm.value3(SEED, x, y, z),
                fbm.value4(SEED, x, y, z, w),
                fbm.simplex2(SEED, x, y),
                fbm.simplex3(SEED, x, y, z),
                fbm.simplex4(SEED, x, y, z, w),
            ] {
                assert!((-1.0..=1.0).contains(&v), "fbm out of [-1,1]: {v}");
                min = min.min(v);
                max = max.max(v);
            }
        }
        assert!(min < -0.5 && max > 0.5, "fbm range degenerate: [{min}, {max}]");
        // Gate 1 on the fbm entry points: 10000 repeated calls, all equal.
        let (x, y, z, w) = (point(SEED, 7, 0), point(SEED, 7, 1), point(SEED, 7, 2), point(SEED, 7, 3));
        let expected = (
            fbm.value2(SEED, x, y),
            fbm.simplex3(SEED, x, y, z),
            fbm.simplex4(SEED, x, y, z, w),
        );
        for _ in 0..10_000 {
            assert_eq!(
                (
                    fbm.value2(SEED, x, y),
                    fbm.simplex3(SEED, x, y, z),
                    fbm.simplex4(SEED, x, y, z, w)
                ),
                expected
            );
        }
    }

    /// `octaves = 0` is the empty sum `0.0` (total function, no NaN).
    #[test]
    fn zero_octaves_is_empty_sum() {
        let fbm = Fbm::new(0, 2.0, 0.5);
        assert_eq!(fbm.value2(SEED, 1.5, -3.25), 0.0);
        assert_eq!(fbm.simplex4(SEED, 0.0, 0.0, 0.0, 0.0), 0.0);
    }
}
