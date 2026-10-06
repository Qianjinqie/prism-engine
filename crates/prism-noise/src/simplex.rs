//! Simplex noise — skewed simplex grid + radial kernel (whitepaper §17, roadmap A3).
//!
//! Structure follows the public-domain reference implementation of Stefan
//! Gustavson, "Simplex noise demystified" (2005), based on Ken Perlin's simplex
//! noise (2001): the input is skewed onto a simplicial (Freudenthal) grid, each
//! surrounding simplex corner contributes `t^4 · (g · d)` with `t = R − |d|²`
//! and a fixed gradient table, and the sum is scaled by Gustavson's calibration
//! constants (2D: R = 0.5, ×70; 3D: R = 0.6, ×32; 4D: R = 0.6, ×27).
//!
//! Documented deviation from the reference: instead of a fixed 512-entry
//! permutation table, each corner's gradient index is taken from the PCG hash
//! of `(seed, corner lattice coordinates)` (modulo bias over 12 gradients is
//! ≈ 4/2⁶⁴, negligible). This makes the field a pure function of the seed
//! parameter — one seed = one noise field — with no hidden global state, per
//! the engine determinism contract (architecture v2.4 §4.4). The 2D case uses
//! the same 12-gradient table as the reference (first two components).
//!
//! Output range: `[-1, 1]` (Gustavson calibration; asserted over a large
//! deterministic sample set by the range test). No transcendentals are used:
//! skew factors are compile-time literals, gradient tables are constants.

use crate::pcg;

/// Gustavson's 12 gradient directions, shared by 2D (first two components) and 3D.
const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Gustavson's 32 gradient directions for 4D (index masked with 31 — power of two).
const GRAD4: [[f32; 4]; 32] = [
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0],
    [0.0, 1.0, -1.0, -1.0],
    [0.0, -1.0, 1.0, 1.0],
    [0.0, -1.0, 1.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, -1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0, -1.0],
    [-1.0, 0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0, 1.0],
    [-1.0, 0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0, -1.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, -1.0, 0.0],
    [1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 0.0],
    [-1.0, 1.0, 1.0, 0.0],
    [-1.0, 1.0, -1.0, 0.0],
    [-1.0, -1.0, 1.0, 0.0],
    [-1.0, -1.0, -1.0, 0.0],
];

/// Gradient index from the PCG hash of `(seed, corner lattice coordinates)`.
#[inline]
fn grad3_index(seed: u64, i: i64, j: i64, k: i64) -> usize {
    (pcg::hash3(seed, i as u64, j as u64, k as u64) % 12) as usize
}

/// 2D simplex noise; output in `[-1, 1]`.
pub fn simplex2(seed: u64, x: f32, y: f32) -> f32 {
    // Skew/unskew factors for 2D: F2 = (√3 − 1) / 2, G2 = (3 − √3) / 6.
    const F2: f32 = 0.366_025_403_784_438_6;
    const G2: f32 = 0.211_324_865_405_187_13;

    let s = (x + y) * F2;
    let (i, j) = ((x + s).floor() as i64, (y + s).floor() as i64);
    let t = (i + j) as f32 * G2;
    let (x0, y0) = (x - (i as f32 - t), y - (j as f32 - t));

    // Lower triangle of the skewed unit square decides the second corner.
    let (i1, j1): (i64, i64) = if x0 > y0 { (1, 0) } else { (0, 1) };
    let (x1, y1) = (x0 - i1 as f32 + G2, y0 - j1 as f32 + G2);
    let (x2, y2) = (x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2);

    let mut n = 0.0_f32;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let g = GRAD3[(pcg::hash2(seed, i as u64, j as u64) % 12) as usize];
        n += t0 * t0 * t0 * t0 * (g[0] * x0 + g[1] * y0);
    }
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let g = GRAD3[(pcg::hash2(seed, (i + i1) as u64, (j + j1) as u64) % 12) as usize];
        n += t1 * t1 * t1 * t1 * (g[0] * x1 + g[1] * y1);
    }
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let g = GRAD3[(pcg::hash2(seed, (i + 1) as u64, (j + 1) as u64) % 12) as usize];
        n += t2 * t2 * t2 * t2 * (g[0] * x2 + g[1] * y2);
    }
    70.0 * n
}

/// 3D simplex noise; output in `[-1, 1]`.
pub fn simplex3(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    // Skew/unskew factors for 3D are exact rationals: F3 = 1/3, G3 = 1/6.
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;

    let s = (x + y + z) * F3;
    let (i, j, k) = (
        (x + s).floor() as i64,
        (y + s).floor() as i64,
        (z + s).floor() as i64,
    );
    let t = (i + j + k) as f32 * G3;
    let (x0, y0, z0) = (
        x - (i as f32 - t),
        y - (j as f32 - t),
        z - (k as f32 - t),
    );

    // Simplex ordering inside the skewed cube (Gustavson's comparison ladder).
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 {
            (i1, j1, k1, i2, j2, k2) = (1, 0, 0, 1, 1, 0);
        } else if x0 >= z0 {
            (i1, j1, k1, i2, j2, k2) = (1, 0, 0, 1, 0, 1);
        } else {
            (i1, j1, k1, i2, j2, k2) = (0, 0, 1, 1, 0, 1);
        }
    } else {
        if y0 < z0 {
            (i1, j1, k1, i2, j2, k2) = (0, 0, 1, 0, 1, 1);
        } else if x0 < z0 {
            (i1, j1, k1, i2, j2, k2) = (0, 1, 0, 0, 1, 1);
        } else {
            (i1, j1, k1, i2, j2, k2) = (0, 1, 0, 1, 1, 0);
        }
    }

    let (x1, y1, z1) = (
        x0 - i1 as f32 + G3,
        y0 - j1 as f32 + G3,
        z0 - k1 as f32 + G3,
    );
    let (x2, y2, z2) = (
        x0 - i2 as f32 + 2.0 * G3,
        y0 - j2 as f32 + 2.0 * G3,
        z0 - k2 as f32 + 2.0 * G3,
    );
    let (x3, y3, z3) = (x0 - 1.0 + 3.0 * G3, y0 - 1.0 + 3.0 * G3, z0 - 1.0 + 3.0 * G3);

    let mut n = 0.0_f32;
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let g = GRAD3[grad3_index(seed, i, j, k)];
        n += t0 * t0 * t0 * t0 * (g[0] * x0 + g[1] * y0 + g[2] * z0);
    }
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let g = GRAD3[grad3_index(seed, i + i1, j + j1, k + k1)];
        n += t1 * t1 * t1 * t1 * (g[0] * x1 + g[1] * y1 + g[2] * z1);
    }
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let g = GRAD3[grad3_index(seed, i + i2, j + j2, k + k2)];
        n += t2 * t2 * t2 * t2 * (g[0] * x2 + g[1] * y2 + g[2] * z2);
    }
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let g = GRAD3[grad3_index(seed, i + 1, j + 1, k + 1)];
        n += t3 * t3 * t3 * t3 * (g[0] * x3 + g[1] * y3 + g[2] * z3);
    }
    32.0 * n
}

/// 4D simplex noise; output in `[-1, 1]`.
///
/// The five simplex corners are the cumulative unit steps of the four
/// coordinates in **descending offset order** (stable sort, ties keep axis
/// priority x > y > z > w). This is the canonical Freudenthal traversal —
/// equivalent to Gustavson's rank method for distinct offsets, but formulated
/// as a sort so that tie regions stay on the proper triangulation (adjacent
/// simplices share a face whose distance to the non-shared corners is
/// √2⁄2 ≈ 0.707 > kernel radius 0.6, which makes the field continuous).
pub fn simplex4(seed: u64, x: f32, y: f32, z: f32, w: f32) -> f32 {
    // Skew/unskew factors for 4D: F4 = (√5 − 1) / 4, G4 = (5 − √5) / 20.
    const F4: f32 = 0.309_016_994_374_947_45;
    const G4: f32 = 0.138_196_601_125_010_5;

    let s = (x + y + z + w) * F4;
    let (i, j, k, l) = (
        (x + s).floor() as i64,
        (y + s).floor() as i64,
        (z + s).floor() as i64,
        (w + s).floor() as i64,
    );
    let t = (i + j + k + l) as f32 * G4;
    let (x0, y0, z0, w0) = (
        x - (i as f32 - t),
        y - (j as f32 - t),
        z - (k as f32 - t),
        w - (l as f32 - t),
    );
    let offset = [x0, y0, z0, w0];

    // Descending offset order via stable insertion sort (deterministic ties).
    let mut order = [0_u8, 1, 2, 3];
    for m in 1..4 {
        let axis = order[m];
        let mut hole = m;
        while hole > 0 && offset[order[hole - 1] as usize] < offset[axis as usize] {
            order[hole] = order[hole - 1];
            hole -= 1;
        }
        order[hole] = axis;
    }

    // Corner m (m = 0..4): cumulative unit steps `step` along the sorted axes,
    // unskewed offset (offset − step + m·G4), kernel t = 0.6 − |d|², weight t⁴.
    let mut step = [0_i64; 4];
    let mut n = 0.0_f32;
    for m in 0..5_i64 {
        let (cx, cy, cz, cw) = (
            x0 - step[0] as f32 + m as f32 * G4,
            y0 - step[1] as f32 + m as f32 * G4,
            z0 - step[2] as f32 + m as f32 * G4,
            w0 - step[3] as f32 + m as f32 * G4,
        );
        let tc = 0.6 - cx * cx - cy * cy - cz * cz - cw * cw;
        if tc > 0.0 {
            let g = GRAD4[(pcg::hash4(
                seed,
                (i + step[0]) as u64,
                (j + step[1]) as u64,
                (k + step[2]) as u64,
                (l + step[3]) as u64,
            ) & 31) as usize];
            n += tc * tc * tc * tc * (g[0] * cx + g[1] * cy + g[2] * cz + g[3] * cw);
        }
        if m < 4 {
            step[order[m as usize] as usize] += 1;
        }
    }
    27.0 * n
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0x5171_9E51_0000_0001;

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
                simplex2(SEED, x, y),
                simplex3(SEED, x, y, z),
                simplex4(SEED, x, y, z, w),
            );
            for _ in 0..10_000 {
                assert_eq!(
                    (
                        simplex2(SEED, x, y),
                        simplex3(SEED, x, y, z),
                        simplex4(SEED, x, y, z, w)
                    ),
                    expected
                );
            }
        }
    }

    /// Gate 3: output stays in `[-1, 1]` and the range is actually used
    /// (values below −0.5 and above +0.5 occur) for all dimensions.
    #[test]
    fn range_and_utilization() {
        const N: u64 = 200_000;
        for seed_off in 0..2_u64 {
            let seed = SEED.wrapping_add(seed_off);
            let (mut min, mut max) = (1.0_f32, -1.0_f32);
            for i in 0..N {
                let (x, y, z, w) = (
                    point(seed, i, 0),
                    point(seed, i, 1),
                    point(seed, i, 2),
                    point(seed, i, 3),
                );
                let v2 = simplex2(seed, x, y);
                let v3 = simplex3(seed, x, y, z);
                let v4 = simplex4(seed, x, y, z, w);
                for v in [v2, v3, v4] {
                    assert!((-1.0..=1.0).contains(&v), "simplex out of [-1,1]: {v}");
                    min = min.min(v);
                    max = max.max(v);
                }
            }
            assert!(min < -0.5 && max > 0.5, "simplex range degenerate: [{min}, {max}]");
        }
    }

    /// Gate 3: spatial continuity — the step over distance h stays small
    /// (empirical Lipschitz bound ≈ 6.5, tested at 16h ≈ 2.5× headroom; a
    /// corner-set discontinuity would jump by O(0.1) ≫ the bound).
    #[test]
    fn continuity_step_is_bounded() {
        let h = 0.001_f32;
        let bound = 16.0 * h;
        for i in 0..5000_u64 {
            let (x, y, z, w) = (
                point(SEED, i, 0),
                point(SEED, i, 1),
                point(SEED, i, 2),
                point(SEED, i, 3),
            );
            assert!((simplex2(SEED, x + h, y) - simplex2(SEED, x, y)).abs() <= bound);
            assert!((simplex2(SEED, x, y + h) - simplex2(SEED, x, y)).abs() <= bound);
            assert!((simplex3(SEED, x + h, y, z) - simplex3(SEED, x, y, z)).abs() <= bound);
            assert!((simplex3(SEED, x, y + h, z) - simplex3(SEED, x, y, z)).abs() <= bound);
            assert!((simplex4(SEED, x + h, y, z, w) - simplex4(SEED, x, y, z, w)).abs() <= bound);
            assert!((simplex4(SEED, x, y, z + h, w) - simplex4(SEED, x, y, z, w)).abs() <= bound);
        }
    }

    /// The seed parameter selects the field: different seeds produce different
    /// values at the overwhelming majority of sampled points (points where a
    /// single corner dominates can legitimately coincide with probability
    /// 1/|gradient table|, so this is a statistical, not per-point, assertion).
    #[test]
    fn seed_selects_the_field() {
        const N: u64 = 1000;
        for dim in 0..3 {
            let mut differing = 0_u32;
            for i in 0..N {
                let (x, y, z, w) = (
                    point(SEED, i, 0),
                    point(SEED, i, 1),
                    point(SEED, i, 2),
                    point(SEED, i, 3),
                );
                let (a, b) = match dim {
                    0 => (
                        simplex2(SEED, x, y),
                        simplex2(SEED.wrapping_add(0x9E37), x, y),
                    ),
                    1 => (
                        simplex3(SEED, x, y, z),
                        simplex3(SEED.wrapping_add(0x9E37), x, y, z),
                    ),
                    _ => (
                        simplex4(SEED, x, y, z, w),
                        simplex4(SEED.wrapping_add(0x9E37), x, y, z, w),
                    ),
                };
                differing += u32::from(a != b);
            }
            assert!(
                differing as u64 >= N * 9 / 10,
                "seed barely changes the field (dim {dim}: {differing}/{N})"
            );
        }
    }
}

