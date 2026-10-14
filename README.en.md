<p align="right">**English** | [简体中文](README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)</p>

# Prism Engine

> A zero-dependency, deterministic Rust media engine — distributed as a **single-file TXT**: download one 14 MB text file, run the unpack script, and you get the complete 26-crate source tree, with no npm / pip / ffmpeg / OpenEXR required.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3271%20passed-brightgreen.svg)


---

## What Is This

Prism Engine is a media engine written in Rust with zero third-party code dependencies, covering modules such as images, audio, geometry, physics, rendering, and DSL. All 103 dependencies are `path` dependencies within this repository — no network access, no package manager, no supply-chain risk.

It adopts **single-file source distribution**: the entire engine (481 files, 26 crates) is merged into one UTF-8 text file, `prism-engine-v1.0-source.txt`. Holding that file alone is enough to restore a directory tree locally that is fully identical to the repository.

## Why Single-File Distribution

- **Auditable**: a single text file — human-readable, `diff`-able, hash-verifiable; frozen at release.
- **Reproducible**: packaging is deterministic — packing the same input twice yields an identical SHA-256.
- **Portable**: no need to clone a large repository; one file can be sent via email or chat, and the build can be restored offline.
- **Trade-off note**: single-file distribution gives up the convenience of "browsing the code directly on GitHub" in exchange for the certainty of "everything in one file, verifiable byte by byte". After unpacking you get an ordinary Cargo workspace, no different from a regular Rust project.
- **Most importantly**: I only need to upload a single file and that's enough (doge

---

## Quick Start

Prerequisites: install [Rust](https://www.rust-lang.org/) (including `cargo`) and [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Download prism-engine-v1.0-source.txt from the repository and save it as a local file
# 2. Open it with a text editor and save the Node.js script between "UNPACK SCRIPT BEGIN"
#    and "UNPACK SCRIPT END" as unpack.cjs in the same directory
# 3. Run the unpack (takes a few seconds; restores 481 files)
node unpack.cjs
# 4. Enter the restored source tree and run the full test suite
cd src/..
cargo test --workspace --release
# Expected: 3271 passed / 0 failed
```

After unpacking you get a standard Cargo workspace (26 crates under `src/`); subsequent `cargo` commands are the same as for any regular Rust project.

> Mobile-side wasm verification chain: `rustup target add wasm32-unknown-unknown`, then `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository Structure (After Unpacking)

| Path | Description |
|------|------|
| `src/` | Source code of the 26 crates (originally named `crates/` in the repository, reorganized into `src/` for distribution) |
| `Compliance/` | Compliance statements: zero-dependency verification / third-party notices / asset registry (originally named `docs/合规/` in the repository) |
| `开发者指南.md` | Self-contained developer documentation: getting started / architecture / math kernels / gates / extensions (originally named `docs/开发者指南.md` in the repository) |
| `tools/` | Gate scripts (ghost-write defense, hygiene scan, compliance self-check) |
| `.github/workflows/` | CI configuration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Build essentials (workspace members already point to `src/*`) |
| `examples/` | Samples used by the `prism-dsl` tests (read via `include_str!`; the build fails without them) |

> The authoritative design documents (`docs/设计/`) are **not part of the distribution**; they remain only in the publisher's full repository.

---

## Module Overview

The 26 crates are grouped by responsibility:

| Category | crates |
|------|-------|
| Kernel | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Media I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generation | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Advanced | prism-ai · prism-dsl · prism-infra |

---

## Build and Test

```bash
# Whole-workspace smoke test (release, recommended)
cargo test --workspace --release
# Equivalent alias
cargo xtest
# Quick check of a single crate
cargo test -p prism-core --release
```

- Full test suite: **3271 passed / 0 failed** (verified by an actual run on a freshly unpacked tree).
- Gates: 14 in total, covering ghost-write defense (G1~G4), open-source hygiene scan (0 internal traces), compliance self-check (9/9), compliance reverse self-test (8/8, violations always go red), and the wasm32 build chain.

Run the gates locally (requires Node 18+):

```bash
node tools/ghost_guard.mjs --check        # Ghost-write defense: G1 duplicate definitions / G2 strict superset / G3 top-of-file documentation / G4 compilation
node tools/hygiene_scan.mjs               # Open-source hygiene: should report "clean (0 internal traces)"
node tools/compliance_check.mjs           # Compliance self-check: should pass 9/9
node tools/compliance_check.mjs --self-test   # Reverse self-test: should go red, proving the criteria have discriminative power
```

---

## Compliance

The engine commits to **zero third-party code dependencies**, verified through self-checks:

- **C1 Zero third-party code dependencies**: all 103 dependencies are `path` dependencies within this repository; no `build.rs`, no native link directives, and `Cargo.lock` locks exactly the 26 in-repo packages.
- **C2 Complete asset registry**: the master asset table has all fields filled for its 5 assets, with in-house status values drawn from a controlled vocabulary; the full text of the OFL 1.1 font license ships with the fonts at `src/prism-text/font_data/LICENSE`.
- **C3 Third-party attribution notices**: `Compliance/THIRD-PARTY-NOTICES.md` is present, and clearly distinguishes "data assets ≠ code dependencies".

> Note: this repository distributes **source code**. The font assets (the Prism Sans SC subset) are licensed under SIL OFL 1.1, with **the full license text shipped alongside the fonts at** `src/prism-text/font_data/LICENSE`; for asset provenance and license registration, see `Compliance/assets/资产登记.md`.

---

## License

This project is released under the **Apache License 2.0**. See the [`LICENSE`](LICENSE) file for details.

> Compliance notice: Apache-2.0 requires retaining copyright and license notices; when redistributing based on this engine, please keep the attribution notices and asset registry under `Compliance/`.
