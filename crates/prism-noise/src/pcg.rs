//! PCG integer hashing — the engine's primitive randomness source .
//!
//! All functions are pure, stateless and integer-only (multiply / shift / xor),
//! so identical arguments yield bitwise-identical results on every platform;
//! integer hashing is a determinism obligation, not a style choice .
//! This is the primitive layer beneath the engine-wide derivation
//! `hash(global_seed, node_id, frame, pixel, sample[, dimension])`
//! callers pack their own integers into
//! `(seed, x[, y[, z[, w]]])`. Seed `0` and coordinate `0` are valid inputs.
//!
//! Provenance: PCG RXS-M-XS permutation, M. E. O'Neill 2014, "PCG: A Family of
//! Simple Fast Space-Efficient Statistically Good Algorithms for Random Number
//! Generation". The 64-bit folding used here follows the widely reused
//! open-source 64-bit `pcg_hash` variant (LCG step + RXS shift + M multiply +
//! final xorshift), with per-slot additive salts so that coordinate position
//! and arity are both encoded (no aliasing between dimensions).

/// LCG multiplier (PCG default, 0x5851F42D4C957F2D).
const PCG_MULT: u64 = 6364136223846793005;
/// LCG increment (PCG default, 0x14057B7EF767814F).
const PCG_INC: u64 = 1442695040888963407;
/// Output-stage multiplier of the RXS-M-XS permutation.
const PCG_MULT2: u64 = 12605985483714929081;

/// Per-slot salts: distinguish coordinate positions so that `hash2(s, 5, 0)` !=
/// `hash1(s, 5)` and `hash2(s, x, y)` != `hash2(s, y, x)`. Well-known 64-bit
/// mix constants (golden ratio / xxHash primes).
const SALT1: u64 = 0x9E37_79B9_7F4A_7C15;
const SALT2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const SALT3: u64 = 0x1656_67B1_9E37_79F9;
const SALT4: u64 = 0x27D4_EB2F_1656_67C5;

/// PCG RXS-M-XS output permutation over 64-bit state (multiply + shift + xor only).
#[inline]
fn permute(v: u64) -> u64 {
    let state = v.wrapping_mul(PCG_MULT).wrapping_add(PCG_INC);
    // Shift amount read from the state's top 5 bits: always in 5..=36 (< 64).
    let xorshifted = state >> ((state >> 59) + 5);
    let word = (xorshifted ^ state).wrapping_mul(PCG_MULT2);
    (word >> 43) ^ word
}

/// Maps 64 hash bits onto f32 in `[0, 1)` using only exact IEEE operations:
/// the top 24 bits are exactly representable in f32 and scaling by 2^-24 is an
/// exponent-only multiplication, so the conversion is bitwise-reproducible.
#[inline]
fn unit_f32(bits: u64) -> f32 {
    ((bits >> 40) as f32) * (1.0 / 16_777_216.0) // 2^-24, exact power of two
}

/// 1D hash: folds `(seed, x)` into a uniformly distributed `u64`.
#[inline]
pub fn hash1(seed: u64, x: u64) -> u64 {
    permute(seed ^ x.wrapping_add(SALT1))
}

/// 2D hash: folds `(seed, x, y)` into a uniformly distributed `u64`.
#[inline]
pub fn hash2(seed: u64, x: u64, y: u64) -> u64 {
    permute(permute(seed ^ x.wrapping_add(SALT1)) ^ y.wrapping_add(SALT2))
}

/// 3D hash: folds `(seed, x, y, z)` into a uniformly distributed `u64`.
#[inline]
pub fn hash3(seed: u64, x: u64, y: u64, z: u64) -> u64 {
    permute(permute(
        permute(seed ^ x.wrapping_add(SALT1)) ^ y.wrapping_add(SALT2),
    ) ^ z.wrapping_add(SALT3))
}

/// 4D hash: folds `(seed, x, y, z, w)` into a uniformly distributed `u64`.
#[inline]
pub fn hash4(seed: u64, x: u64, y: u64, z: u64, w: u64) -> u64 {
    permute(permute(permute(
        permute(seed ^ x.wrapping_add(SALT1)) ^ y.wrapping_add(SALT2),
    ) ^ z.wrapping_add(SALT3)) ^ w.wrapping_add(SALT4))
}

/// 1D hash mapped to f32 in `[0, 1)`.
#[inline]
pub fn hash1_f32(seed: u64, x: u64) -> f32 {
    unit_f32(hash1(seed, x))
}

/// 2D hash mapped to f32 in `[0, 1)`.
#[inline]
pub fn hash2_f32(seed: u64, x: u64, y: u64) -> f32 {
    unit_f32(hash2(seed, x, y))
}

/// 3D hash mapped to f32 in `[0, 1)`.
#[inline]
pub fn hash3_f32(seed: u64, x: u64, y: u64, z: u64) -> f32 {
    unit_f32(hash3(seed, x, y, z))
}

/// 4D hash mapped to f32 in `[0, 1)`.
#[inline]
pub fn hash4_f32(seed: u64, x: u64, y: u64, z: u64, w: u64) -> f32 {
    unit_f32(hash4(seed, x, y, z, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const SEED: u64 = 0x0123_4567_89AB_CDEF;

    /// Gate 1: same arguments called 10000 times give bit-identical results.
    #[test]
    fn determinism_10000_calls() {
        for i in 0..8_u64 {
            let (x, y, z, w) = (i * 7 + 1, i * 13 + 3, i * 29 + 5, i * 53 + 11);
            let expected = (
                hash1(SEED, x),
                hash2(SEED, x, y),
                hash3(SEED, x, y, z),
                hash4(SEED, x, y, z, w),
                hash1_f32(SEED, x),
                hash2_f32(SEED, x, y),
                hash3_f32(SEED, x, y, z),
                hash4_f32(SEED, x, y, z, w),
            );
            for _ in 0..10_000 {
                assert_eq!(
                    (
                        hash1(SEED, x),
                        hash2(SEED, x, y),
                        hash3(SEED, x, y, z),
                        hash4(SEED, x, y, z, w),
                        hash1_f32(SEED, x),
                        hash2_f32(SEED, x, y),
                        hash3_f32(SEED, x, y, z),
                        hash4_f32(SEED, x, y, z, w),
                    ),
                    expected
                );
            }
        }
    }

    /// Gate 2: uniformity smoke test over sequential (counter-style) inputs —
    /// values stay in [0, 1), the mean is ~0.5 and all 16 buckets are populated
    /// within ±20% of expectation for every hash arity.
    #[test]
    fn uniformity_smoke() {
        const N: u64 = 100_000;
        const BUCKETS: usize = 16;
        for arity in 0..4 {
            let (mut buckets, mut sum, mut min, mut max) =
                ([0_usize; BUCKETS], 0.0_f64, 1.0_f32, 0.0_f32);
            for i in 0..N {
                let v = match arity {
                    0 => hash1_f32(SEED, i),
                    1 => hash2_f32(SEED, i, i.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    2 => hash3_f32(SEED, i, i * 3 + 1, i * 7 + 2),
                    _ => hash4_f32(SEED, i, i * 3 + 1, i * 7 + 2, i * 11 + 5),
                };
                assert!((0.0..1.0).contains(&v), "hash{arity}_f32 out of [0,1): {v}");
                sum += v as f64;
                min = min.min(v);
                max = max.max(v);
                buckets[(v * BUCKETS as f32) as usize] += 1;
            }
            let expected = N as f64 / BUCKETS as f64;
            for (b, &count) in buckets.iter().enumerate() {
                let c = count as f64;
                assert!(
                    (0.8 * expected..=1.2 * expected).contains(&c),
                    "hash{}_f32 bucket {b} count {count} off ({expected} expected)",
                    arity + 1
                );
            }
            let mean = sum / N as f64;
            assert!((0.48..=0.52).contains(&mean), "hash{}_f32 mean {mean}", arity + 1);
            assert!(min < 0.02 && max > 0.98, "hash{}_f32 range degenerate", arity + 1);
        }
    }

    /// Gate 2: distinct sampled inputs do not collide.
    #[test]
    fn sampled_inputs_do_not_collide() {
        // 8192 distinct (x, y) pairs over a grid produce 8192 distinct hashes.
        let mut seen = HashSet::new();
        for x in 0..64_u64 {
            for y in 0..128_u64 {
                assert!(seen.insert(hash2(SEED, x * 31 + 1, y * 17 + 3)));
            }
        }
        // Transposed 2D inputs differ (slot salts).
        for i in 0..1000_u64 {
            assert_ne!(hash2(SEED, i, i + 1), hash2(SEED, i + 1, i));
        }
        // Arity separation: folding a zero coordinate is not the identity.
        for i in 0..1000_u64 {
            let h1 = hash1(SEED, i);
            assert_ne!(h1, hash2(SEED, i, 0));
            assert_ne!(h1, hash3(SEED, i, 0, 0));
            assert_ne!(h1, hash4(SEED, i, 0, 0, 0));
        }
        // Distinct seeds differ on the same coordinates (sampled).
        for i in 0..1000_u64 {
            assert_ne!(hash2(SEED, i, i), hash2(SEED.wrapping_add(1), i, i));
        }
    }
}
