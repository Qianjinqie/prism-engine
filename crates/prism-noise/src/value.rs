//! Value noise — lattice hash + quintic interpolation (whitepaper §17, roadmap A3).
//!
//! Grid-corner values are PCG hashes of `(seed, lattice coordinates)`; between
//! corners the field is interpolated with Perlin's improved-noise quintic fade
//! `6t^5 − 15t^4 + 10t^3` (C2-continuous, zero derivative at every lattice
//! boundary). Output range: `[-1, 1]`.
//!
//! At integer coordinates the value is exactly the mapped hash of that lattice
//! corner (lattice reproducibility, asserted by a test).
//!
//! Determinism: only `+`, `-`, `×`, integer casts with language-defined
//! saturating semantics, and `floor` — an exact IEEE rounding operation, not a
//! libm transcendental (D-29) — are used, so results are bitwise-reproducible
//! everywhere. Coordinates are expected finite and within a range where
//! `floor as i64` is lossless (`|coord| < 2^62`).

use crate::pcg;

/// Perlin improved-noise quintic fade `6t^5 − 15t^4 + 10t^3`
/// (max |fade'| = 1.875 at t = 0.5 — the rigorous continuity bound).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Lattice cell index as hash input: bit pattern of the `i64` index
/// (language-defined two's-complement reinterpretation, platform-independent).
#[inline]
fn lat(v: i64) -> u64 {
    v as u64
}

/// Corner value in `[-1, 1)`: `2u − 1` over the PCG unit hash.
#[inline]
fn corner2(seed: u64, xi: i64, yi: i64) -> f32 {
    2.0 * pcg::hash2_f32(seed, lat(xi), lat(yi)) - 1.0
}

/// 2D value noise; output in `[-1, 1]`.
pub fn value2(seed: u64, x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor() as i64, y.floor() as i64);
    let (u, v) = (fade(x - xi as f32), fade(y - yi as f32));
    let (c00, c10) = (corner2(seed, xi, yi), corner2(seed, xi + 1, yi));
    let (c01, c11) = (corner2(seed, xi, yi + 1), corner2(seed, xi + 1, yi + 1));
    lerp(lerp(c00, c10, u), lerp(c01, c11, u), v)
}

/// 3D value noise; output in `[-1, 1]`.
pub fn value3(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let (xi, yi, zi) = (x.floor() as i64, y.floor() as i64, z.floor() as i64);
    let (u, v, w) = (fade(x - xi as f32), fade(y - yi as f32), fade(z - zi as f32));
    let c = |dx: i64, dy: i64, dz: i64| {
        2.0 * pcg::hash3_f32(seed, lat(xi + dx), lat(yi + dy), lat(zi + dz)) - 1.0
    };
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// 4D value noise; output in `[-1, 1]`.
///
/// Corners are enumerated in a fixed order (index = dx | dy<<1 | dz<<2 |
/// dw<<3) and reduced quadrilinearly in the fixed order x, y, z, w.
pub fn value4(seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 {
    let (xi, yi, zi, wi) = (
        x.floor() as i64,
        y.floor() as i64,
        z.floor() as i64,
        w.floor() as i64,
    );
    let (u, v, s, t) = (
        fade(x - xi as f32),
        fade(y - yi as f32),
        fade(z - zi as f32),
        fade(w - wi as f32),
    );
    let mut corners = [0.0_f32; 16];
    let mut idx = 0;
    for dw in 0..2_i64 {
        for dz in 0..2_i64 {
            for dy in 0..2_i64 {
                for dx in 0..2_i64 {
                    corners[idx] =
                        2.0 * pcg::hash4_f32(seed, lat(xi + dx), lat(yi + dy), lat(zi + dz), lat(wi + dw))
                            - 1.0;
                    idx += 1;
                }
            }
        }
    }
    let mut layer = [0.0_f32; 8];
    for k in 0..8 {
        layer[k] = lerp(corners[2 * k], corners[2 * k + 1], u);
    }
    let mut cell = [0.0_f32; 4];
    for k in 0..4 {
        cell[k] = lerp(layer[2 * k], layer[2 * k + 1], v);
    }
    let mut row = [0.0_f32; 2];
    for k in 0..2 {
        row[k] = lerp(cell[2 * k], cell[2 * k + 1], s);
    }
    lerp(row[0], row[1], t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0x5EED_5EED_5EED_0001;

    /// Deterministic pseudo-random point over `[-16, 16)` per axis.
    fn point(seed: u64, i: u64, axis: u64) -> f32 {
        pcg::hash2_f32(seed, i, axis) * 32.0 - 16.0
    }

    /// Gate 1: same arguments called 10000 times give bit-identical results.
    #[test]
    fn determinism_10000_calls() {
        for i in 0..8_u64 {
            let (x, y, z, w) = (
                point(SEED, i, 0),
                point(SEED, i, 1),
                point(SEED, i, 2),
                point(SEED, i, 3),
            );
            let expected = (
                value2(SEED, x, y),
                value3(SEED, x, y, z),
                value4(SEED, x, y, z, w),
            );
            for _ in 0..10_000 {
                assert_eq!(
                    (value2(SEED, x, y), value3(SEED, x, y, z), value4(SEED, x, y, z, w)),
                    expected
                );
            }
        }
    }

    /// Gate 3: output stays in `[-1, 1]` and the range is actually used
    /// (values below −0.5 and above +0.5 occur) for all dimensions.
    #[test]
    fn range_and_utilization() {
        const N: u64 = 100_000;
        for seed_off in 0..2_u64 {
            let seed = SEED.wrapping_add(seed_off);
            let (mut min, mut max) = (1.0_f32, -1.0_f32);
            let mut check = |v: f32| {
                assert!((-1.0..=1.0).contains(&v), "value noise out of [-1,1]: {v}");
                min = min.min(v);
                max = max.max(v);
            };
            for i in 0..N {
                let (x, y, z, w) = (point(seed, i, 0), point(seed, i, 1), point(seed, i, 2), point(seed, i, 3));
                check(value2(seed, x, y));
                check(value3(seed, x, y, z));
                check(value4(seed, x, y, z, w));
            }
            assert!(min < -0.5 && max > 0.5, "value range degenerate: [{min}, {max}]");
        }
    }

    /// Gate 3: spatial continuity — the step over distance h is bounded by the
    /// rigorous value-noise Lipschitz constant 3.75 (= 2 × max|fade'|) per axis.
    #[test]
    fn continuity_step_is_bounded() {
        let h = 0.01_f32;
        let bound = 3.751 * h; // rigorous 3.75/h plus rounding slack
        for i in 0..5000_u64 {
            let (x, y, z, w) = (point(SEED, i, 0), point(SEED, i, 1), point(SEED, i, 2), point(SEED, i, 3));
            assert!((value2(SEED, x + h, y) - value2(SEED, x, y)).abs() <= bound);
            assert!((value2(SEED, x, y + h) - value2(SEED, x, y)).abs() <= bound);
            assert!((value3(SEED, x + h, y, z) - value3(SEED, x, y, z)).abs() <= bound);
            assert!((value3(SEED, x, y + h, z) - value3(SEED, x, y, z)).abs() <= bound);
            assert!((value4(SEED, x + h, y, z, w) - value4(SEED, x, y, z, w)).abs() <= bound);
            assert!((value4(SEED, x, y, z + h, w) - value4(SEED, x, y, z, w)).abs() <= bound);
        }
    }

    /// Gate 3: lattice reproducibility — at integer coordinates the value is
    /// exactly the mapped hash of that lattice corner (negative cells included).
    #[test]
    fn lattice_points_reproduce_corner_hash() {
        for i in -8..=8_i64 {
            for j in -8..=8_i64 {
                let expected = 2.0 * pcg::hash2_f32(SEED, i as u64, j as u64) - 1.0;
                assert_eq!(value2(SEED, i as f32, j as f32), expected);
            }
        }
        for i in -4..=4_i64 {
            for j in -4..=4_i64 {
                for k in -4..=4_i64 {
                    let expected =
                        2.0 * pcg::hash3_f32(SEED, i as u64, j as u64, k as u64) - 1.0;
                    assert_eq!(value3(SEED, i as f32, j as f32, k as f32), expected);
                }
            }
        }
        for i in -2..=2_i64 {
            for j in -2..=2_i64 {
                for k in -2..=2_i64 {
                    for l in -2..=2_i64 {
                        let expected = 2.0
                            * pcg::hash4_f32(SEED, i as u64, j as u64, k as u64, l as u64)
                            - 1.0;
                        assert_eq!(value4(SEED, i as f32, j as f32, k as f32, l as f32), expected);
                    }
                }
            }
        }
    }
}
