//! `prism-noise` — deterministic scalar-field primitives for the Prism engine
//! (roadmap workstream A3; whitepaper §17; architecture v2.4 S1/S4 primitive layer).
//!
//! Contents:
//!
//! - [`pcg`] — PCG integer hashing, the engine's only randomness primitive.
//!   The engine-wide sample stream `hash(global_seed, node_id, frame, pixel,
//!   sample[, dimension])` (architecture v2.4 §4.4) packs its integers into
//!   these `(seed, x[, y[, z[, w]]])` entry points. This crate holds no RNG
//!   state and exposes only pure scalar functions; higher layers do the packing.
//! - [`value`] — value noise: lattice hash + quintic interpolation.
//! - [`simplex`] — simplex noise: skewed simplex grid + radial kernel.
//! - [`fbm`] — fractal Brownian motion over either basis.
//!
//! # Determinism contract
//!
//! Every public function is a pure function of its arguments: no internal
//! mutable state, no hidden RNG, no platform-dependent math. Only `+`, `-`, `×`,
//! integer operations, shifts and exact IEEE operations (`floor` is an exact
//! rounding operation, not a libm transcendental — see D-29) are used, so
//! results are bitwise-identical across platforms. Skew factors are compile-time
//! literals; gradient tables are fixed constants.
//!
//! # Output ranges
//!
//! | family | range |
//! |--------|-------|
//! | [`pcg::hash1_f32`] … [`pcg::hash4_f32`] | `[0, 1)` |
//! | [`value::value2`] … [`value::value4`] | `[-1, 1]` |
//! | [`simplex::simplex2`] … [`simplex::simplex4`] | `[-1, 1]` |
//! | [`fbm::Fbm`] (all methods) | `[-1, 1]` |
//!
//! Zero external dependencies: only the Rust standard library is used.

pub mod fbm;
pub mod pcg;
pub mod simplex;
pub mod value;
