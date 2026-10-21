**English** | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)


# Prism Engine

> A zero-dependency, deterministic Rust media engine — delivered as a **single-file TXT**: download one 17 MB text file, run the unpack script, and you get the complete source tree (28 crates), with no npm / pip / ffmpeg / OpenEXR.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

---

## What This Is

Prism Engine is a media engine written in Rust with zero third-party code dependencies: reading images, laying out, drawing frames, cutting timelines, and delivering finished cuts — all inside one deterministic engine. All 110 runtime dependencies (plus 20 test dependencies) are `path` dependencies within this repository — no network, no package manager, no supply-chain risk.

It uses a **single-file source distribution**: the entire engine (622 files, 28 crates) is merged into a single UTF-8 text file, `prism-engine-v2.0-source.txt`. With that file in hand, you can restore locally a directory tree identical to the repository.

---

## What the Engine Can Do

In one sentence: **an image is what a frame looks like at one moment, and a video is a run of what frames look like over a stretch of time** — from the same scene description, freezing one instant yields a single frame, and advancing continuously while recording yields video; stills and motion share one capability set. The rest is split into five parts by "what you want to do": **read in → lay out → draw → deliver → plug into your own toolchain**.

### What It Reads (Input)

- **Unified decode facade**: a single `decode_image(bytes, ext)` entry point dispatches still images — PNG (non-interlaced 8/16-bit), baseline JPEG (fixed-point IDCT, bit-for-bit deterministic across platforms), **magic number first, extension as fallback** (it still decodes when the extension is wrong); formats outside the matrix or requiring external cooperation always report errors in three parts (where / what / how to fix) along with an ffmpeg fallback suggestion, and **never fail silently**.
- **Animated images / audio / containers**: GIF and APNG decode natively frame by frame (`prism-media`); WAV decodes built in, mp3/aac/FLAC go through the ffmpeg probe path (`prism-audio`); MP4-MJPEG read-back (`decode_mp4_video`).
- **The full EXR stack**: read/write, compression, DWA, plus **cross-implementation bitstream comparison** against the official OpenEXR (not just writing and reading its own output).
- **Camera RAW**: baseline RAW decoding. **Boundary**: `Compression=7` (lossless JPEG) in real DNG files is not implemented, so DNGs shot by real cameras cannot be read — recorded under "Honest Boundaries", not hidden.
- **3D assets**: glTF / OBJ import, USD (`.usda`) parsing and MaterialX parameter packs; meshes can enter the render scene.
- **External fonts**: `load_font_file` loads third-party TTF/OTF directly; the fallback chain resolves level by level, and missing glyphs produce explicit diagnostics instead of tofu boxes.

### What It Lays Out (Authoring Syntax)

- **DSL v2 full block syntax**: a single `.prism` file describes "what you want, how it is placed, how it moves, what it looks like, and what to export"; the compilation output is a node graph + property channels + timeline.
- **Five combinators**: the `title_card` / `lower_third` / `split_screen` / `ken_burns` / `morph_between` high-level combinators expand into node graphs, lowering a "90-second cut" from hand-written scene code to assembly.
- **Transitions and morph**: Fade / Wipe / Dissolve / Push fully parameterized (duration, easing, direction), Morph does polyline deformation, and scene sequences are samplable at any instant.
- **Formula typesetting**: `math_ast` + `math_layout` typeset sub/superscripts / fractions / radicals / large operators / matrices; a DSL `formula` block typesets directly, on the same baseline as body text.

### What It Draws (Rendering and Simulation)

- **Offline photorealistic rendering**: path tracing + physically correct BSDFs (diffuse / specular / glass / metal); glass and metal use stochastic inversion (Neumann-series Monte Carlo), with volumetric media and subsurface scattering supported.
- **Four families of curve primitives**: cylinder / cone / torus / surface of revolution, all analytically intersected (quartics solved by Ferrari's closed form, yielding exact roots); arbitrary triangle meshes + SAH-BVH, checked by dual-path parity comparison.
- **GPU compute**: a device-agnostic compute semantics layer + Vulkan dispatch + hand-written minimal SPIR-V; fully runnable on platforms without a GPU under the Mesa lavapipe software ICD (verifiable headless).
- **Physics and particles**: XPBD unified solver (rigid bodies / soft bodies / cloth / joints / fluids / fracture), smoke FLIP, particle systems (AudioReactive driven by audio envelopes).
- **Generative art and post-processing**: flow fields / L-systems / cellular automata / reaction-diffusion / fractals / Voronoi; blur / bloom / halation / film grain / four NPR styles (toon / watercolor / halftone / oil) / three-in-one denoise.
- **Color grading and vectors**: secondary grading (HSL qualifiers + RGB curves), cinematic / film / anime style presets, and three quality tiers — draft/standard/cinematic (quality is a parameter, not mystique); vector path rasterization, path draw-on and morph.
- **Audio processing**: loudness normalization (loudnorm), beat detection, spectral analysis, envelope extraction — all able to drive the picture (particle AudioReactive follows the audio envelope).
- **The CPU reference backend is a first-class citizen**: machines without a GPU still produce correct images, and it is also the ground truth for the GPU.

### What It Delivers (Output)

- **`render_movie` straight through**: one call on the DSL v2 compilation output runs "per-frame evaluation → tile rendering → delivery"; a single command turns a `.prism` into a finished cut (`prism-app render demo.prism`); uncovered nodes report errors faithfully rather than silently approximating.
- **Preview loop**: `prism-app preview` starts the preview service; the editor has two preview tiers — Draft (geometric patterns to keep the frame rate) / Real (low-spp true rendering, asynchronous refresh).
- **In-house encoder + rate control**: an I/P-frame JPEG-domain temporal encoder writes the private `PRMV` container (v2/v3 schemas), with water-filling rate control + qpel sub-pixel motion estimation (measured PSNR +2.577 dB, size −21.6%).
- **ffmpeg cooperation chain**: with ffmpeg → transcode to H.264/HEVC through a stdin rawvideo pipe; without ffmpeg → automatically degrade to MJPEG, **the engine can always produce a cut**; no binaries are bundled.
- **EXR sequence delivery**: frame sequences are written directly as EXR for the post-production pipeline.
- **Distributed farm**: frame-level distribution + N process workers + deterministic merge — parallel MD5 == serial MD5, with resume support.

### How to Integrate (Ecosystem)

- **MCP tool surface**: three stdio JSON-RPC tools — `scene_check` (compile + trial render + frame-count estimate), `render_submit` (deterministic job id + per-frame MD5 digests), `diagnose` (three-part error localization).
- **Node SDK**: semantic version negotiation (including the 0.x special case), shipped with an example node pack (`examples/sdk-nodes/cinematic-basic/`, compilable / loadable / negotiable).
- **Hardware-encoding probe skeleton**: `prism-ffi-hwenc` with a three-platform (VideoToolbox / NVENC / VAAPI) probe chain + feature gate matrix; unsupported platforms are reported explicitly.

> For the detailed capability boundaries (what it **cannot** do), see the "Honest Boundaries"
> section of `开发者指南.md` inside the distribution package — it records the real current
> state (curve primitives are CPU-only, the GPU speedup must be measured on target hardware,
> hardware encoding is wired in per target platform…). This document only gives the big picture, with no hype.

---

## Why Single-File Distribution

- **Auditable**: a single text file, human-readable, `diff`-able, hash-verifiable; publishing freezes it.
- **Reproducible**: packaging is deterministic — two runs on the same input produce an identical SHA-256.
- **Portable**: no need to clone a huge repository; one file travels by email or chat, and can be restored and built offline.
- **Trade-off**: single-file distribution gives up the convenience of "browsing code directly on GitHub" in exchange for the determinism of "one file carries everything and is verifiable byte by byte". What you get after unpacking is an ordinary Cargo workspace, no different from a regular Rust project.

---

## Quick Start

Prerequisites: install [Rust](https://www.rust-lang.org/) (including `cargo`) and [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Download prism-engine-v2.0-source.txt from the repository and save it as a local file
# 2. Open it in a text editor and save the Node.js script between "UNPACK SCRIPT BEGIN" and
#    "UNPACK SCRIPT END" as unpack.cjs in the same directory
# 3. Run the unpack (a few seconds; restores 622 files)
node unpack.cjs
# 4. Run the full test suite in the restored directory (i.e. the workspace root)
cargo test --workspace --release
# expected: 3714 passed / 0 failed
```

After unpacking you get a standard Cargo workspace (28 crates under `src/`), and subsequent `cargo` commands are the same as for any regular Rust project.

> Mobile-side wasm verification chain: `rustup target add wasm32-unknown-unknown`, then `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository Structure (After Unpacking)

| Path | Description |
|------|-------------|
| `src/` | Source of the **28 crates** (29 in the repository, originally `crates/`, reorganized into `src/` for distribution) |
| `Compliance/` | Compliance statements: zero-dependency verification / third-party notices / asset registry (originally `docs/合规/` in the repository) |
| `开发者指南.md` | Self-contained developer documentation: getting started / architecture / math kernel / gates / extension (originally `docs/开发者指南.md` in the repository) |
| `tools/` | Gate scripts (ghost-write defense, hygiene scan, compliance self-check) |
| `.github/workflows/` | CI configuration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Build essentials (workspace members already point to `src/*`) |
| `examples/` | Sample fixtures for `prism-dsl` tests and the example node pack (read via `include_str!`; missing files break the build) |

---

## Module Overview

The **28 crates** in the distribution package, grouped by responsibility:

| Category | crates |
|----------|--------|
| Kernel | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Media I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generation | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Advanced | prism-ai · prism-dsl · prism-infra · prism-author |
| Sidecar (feature gate) | prism-ffi-hwenc |

---

## Building and Testing

```bash
# Whole-repo smoke test (release, recommended)
cargo test --workspace --release
# equivalent alias
cargo xtest
# quick check of a single crate
cargo test -p prism-core --release
```

- Full test suite: **3714 passed / 0 failed / 0 ignored** (this version has been actually run on the **unpacked tree**: 166 test blocks)
- CI: **6 jobs** — ghost-write defense (ghost-guard) / compile + full test suite (native) / wasm32 build (wasm) / WGSL static checks (wasm-verify) / cross-implementation bitstream comparison against official OpenEXR (interop) / node host cross-validation (lint).
- 3 local gate scripts (fully zero-dependency, using only `node`, below) + 1 packaging script; among them, the compliance self-check passes **9/9** and the negative self-test goes red **8/8**.

Running the gates locally (Node 18+ required):

```bash
node tools/ghost_guard.mjs --check        # ghost-write defense: G1 duplicate definitions / G2 strict superset / G3 top-level docs / G4 compilation
node tools/hygiene_scan.mjs               # open-source hygiene: list every residual internal term one by one (file:line); any hit → exit 1
node tools/compliance_check.mjs           # compliance self-check: should pass 9/9
node tools/compliance_check.mjs --self-test   # negative self-test: should go red, proving the criteria can discriminate
```

> **The open-source hygiene scan still reports existing hits**: early internal terms remain in
> the repository; the **documentation layer shipped with the package is already at zero**
> (README / Developer Guide / Compliance), while the rest are in **source comments inside the
> distribution** and in **repository files not shipped with the package**, truthfully classified
> and listed as separate cleanup items. The script exists precisely to make the residuals
> **visible and clearable one by one**, rather than pretending they are gone — please do not read
> "hit count not zero" as the gate failing.

---

## Compliance

The engine promises **zero third-party code dependencies**, verified by self-check:

- **C1 Zero third-party code dependencies**: all 130 dependencies (110 runtime + 20 test) are `path` dependencies within this repository; there is no `build.rs`, no native link directives, and `Cargo.lock` pins 29 in-repo packages.
- **C2 Complete asset registry**: the fields of the 5 primary asset table entries are all complete, and self-development status stays within the controlled vocabulary; the full text of the OFL 1.1 font license sits alongside the font at `src/prism-text/font_data/LICENSE`.
- **C3 Third-party attribution notices**: `Compliance/THIRD-PARTY-NOTICES.md` is present and explicitly distinguishes "data assets ≠ code dependencies".

> Note: this repository distributes **source code**. The font asset (Prism Sans SC subset) is licensed under SIL OFL 1.1, and **the full license text sits alongside the font** at `src/prism-text/font_data/LICENSE`; for asset provenance and license records see `Compliance/assets/资产登记.md`.

---

## License

This project is released under the **Apache License 2.0**. See the [`LICENSE`](../LICENSE) file for details.

> Compliance note: Apache-2.0 requires retaining copyright and license notices; when redistributing based on this engine, please retain the attribution notices and asset registry under `Compliance/`.

**Note: if other versions contain translation errors, the Simplified Chinese version shall prevail.**
