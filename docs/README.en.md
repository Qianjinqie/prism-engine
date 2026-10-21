**English** | [简体中文](../README.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Português (BR)](README.pt-BR.md) | [Русский](README.ru.md)

# Prism Engine

> A zero-dependency, deterministic Rust media engine — distributed as a **single TXT file**: download a 17 MB text, run the unpack script, and the complete source tree (28 crates) is yours. No npm / pip / ffmpeg / OpenEXR required.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust-orange.svg)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Distribution](https://img.shields.io/badge/distribution-single--file%20TXT-lightgrey.svg)
![Tests](https://img.shields.io/badge/tests-3714%20passed-brightgreen.svg)
![Crates](https://img.shields.io/badge/crates-28-blue.svg)

(Badges are static fact badges; the numbers are defined below: `crates-28` = the number of crates **inside the distribution package**,
while the repository holds one more demo crate that is not packed, hence 29 in the repository. Once CI is wired up, these can be replaced with dynamic badges.)

---

## What This Is

Prism Engine is a media engine written in Rust with zero third-party code dependencies: reading images, laying out
text, drawing frames, cutting timelines, and delivering finished videos — all inside one deterministic engine. All 110 runtime
dependencies (plus another 20 test dependencies) are `path` dependencies within this repository — no network access, no package
manager, no supply-chain risk.

It uses **single-file source distribution**: the entire engine (622 files, 28 crates) is merged into one UTF-8
text, `prism-engine-v2.0-source.txt`. Holding that file is enough to restore locally a directory tree identical
to the repository.

---

## What the Engine Can Do

In one sentence: **an image is what a frame looks like at one instant, and a video is what a series of frames looks like over a span of time** —
from the same scene description, freezing one instant yields a single frame; advancing continuously and recording yields a video. Still
images and moving images share one set of capabilities. Below, five sections organized by "what you want to do": **read it in → lay it out → draw it → deliver it → plug it into your own toolchain**.

### What It Reads (Input)

- **Unified decoding facade**: a single `decode_image(bytes, ext)` entry dispatches still images — PNG (non-interlaced 8/16-bit),
  baseline JPEG (fixed-point IDCT, deterministic bit for bit across platforms), **magic number first, extension as fallback** (a wrong
  extension still decodes); formats outside the matrix or requiring external cooperation always fail with a three-part error (where / what /
  how to fix) plus an ffmpeg cooperation suggestion, and **never fail silently**.
- **Animated images / audio / containers**: native frame-by-frame decoding of GIF and APNG (`prism-media`); built-in WAV decoding,
  with mp3/aac/FLAC taking the ffmpeg probe path (`prism-audio`); MP4-MJPEG read-back (`decode_mp4_video`).
- **Full EXR suite**: read/write, compression, DWA, and **cross-implementation bitstream comparison** against the official OpenEXR
  (not write-then-read by ourselves).
- **Camera RAW**: baseline RAW decoding. **Boundary**: `Compression=7` (lossless JPEG) in real DNGs is not implemented,
  so DNGs shot by real cameras cannot be read — stated under "Honest Boundaries", not hidden.
- **3D assets**: glTF / OBJ import, USD (`.usda`) parsing and MaterialX parameter packs; meshes can enter rendering scenes.
- **External fonts**: `load_font_file` loads third-party TTF/OTF directly, the fallback chain resolves level by level, and
  missing glyphs produce explicit diagnostics rather than tofu boxes.

### What It Lays Out (Authoring Syntax)

- **Complete DSL v2 block syntax**: a single `.prism` describes "what you want, how it is placed, how it moves, what it looks like,
  and what it exports to"; the compiled artifact is a node graph + property channels + timeline.
- **Five combinators**: the `title_card` / `lower_third` / `split_screen` / `ken_burns` /
  `morph_between` high-level combinators expand into node graphs, turning a "90-second finished video" from hand-written scene code into assembly.
- **Transitions and morph**: Fade / Wipe / Dissolve / Push fully parameterized (duration, easing, direction);
  Morph polyline deformation; scene sequences sampleable at any instant.
- **Formula typesetting**: `math_ast` + `math_layout` typeset sub/superscripts / fractions / roots / large operators /
  matrices; the DSL `formula` block typesets directly, on the same baseline as body text.

### What It Draws (Rendering and Simulation)

- **Offline photorealistic rendering**: path tracing + physically correct BSDFs (diffuse / specular / glass / metal);
  glass and metal use stochastic inversion (Neumann series Monte Carlo), with volume media and subsurface scattering supported.
- **Four families of curve primitives**: cylinder / cone / torus / surface of revolution, all analytically intersected (quartic equations via
  Ferrari's closed form, exact roots); arbitrary triangle meshes + SAH-BVH, with dual-path parity comparison.
- **GPU compute**: a device-agnostic compute semantics layer + Vulkan dispatch + hand-written minimal SPIR-V;
  runs fully on GPU-less platforms under the Mesa lavapipe software ICD (verifiable headless).
- **Physics and particles**: XPBD unified solver (rigid bodies / soft bodies / cloth / joints / fluids / fracture),
  smoke FLIP, particle systems (AudioReactive linked to audio envelopes).
- **Generative art and post-processing**: flow fields / L-systems / cellular automata / reaction-diffusion / fractals / Voronoi;
  blur / bloom / halation / film grain / four NPR styles (toon / watercolor / halftone / oil) / three-in-one denoising.
- **Color grading and vectors**: secondary grading (HSL selections + RGB curves), cinematic / film / anime style presets, and
  three quality tiers — draft/standard/cinematic (quality is a parameter, not folklore); vector path rasterization, path draw-on and morph.
- **Audio processing**: loudness normalization (loudnorm), beat detection, spectrum analysis, envelope extraction —
  all able to drive visuals (particles AudioReactive following audio envelopes).
- **The CPU reference backend is a first-class citizen**: GPU-less machines still produce correct images, and it is the
  correctness baseline for the GPU.

### What It Delivers (Output)

- **`render_movie` straight through**: one call on the DSL v2 compiled artifact completes "per-frame evaluation → tile rendering →
  delivery", rendering a finished video from a `.prism` with a single command (`prism-app render demo.prism`); uncovered nodes report errors honestly rather than approximating silently.
- **Preview loop**: `prism-app preview` starts the preview service; the editor's Draft tier (geometric patterns to keep the frame rate) /
  Real tier (low-spp true rendering, asynchronous refresh) offer dual-tier preview.
- **In-house encoder + rate control**: an I/P-frame JPEG-domain temporal encoder over the private `PRMV` container (v2/v3 two-generation
  schema), water-filling rate control + qpel sub-pixel motion estimation (measured PSNR +2.577 dB,
  size −21.6%).
- **ffmpeg cooperation chain**: ffmpeg present → transcode to H.264/HEVC through a stdin rawvideo pipe;
  ffmpeg absent → automatically fall back to MJPEG, so **the engine can always produce a video**; no binaries are bundled.
- **EXR sequence delivery**: frame sequences written directly to EXR, feeding into post-production pipelines.
- **Distributed farm**: frame-level distribution + N-process workers + deterministic merge — parallel MD5 == serial MD5,
  with checkpoint resume.

### How to Integrate (Ecosystem)

- **MCP tool surface**: stdio JSON-RPC with three tools — `scene_check` (compile + trial render + frame-count estimate),
  `render_submit` (deterministic job id + per-frame MD5 digests), `diagnose` (three-part error localization).
- **Node SDK**: semantic version negotiation (including the 0.x special case), with an example node package bundled
  (`examples/sdk-nodes/cinematic-basic/`, compilable / loadable / negotiable).
- **Hardware-encoding probe skeleton**: `prism-ffi-hwenc` three-platform (VideoToolbox / NVENC / VAAPI)
  probe chain + feature gate matrix; unsupported platforms report "not supported" explicitly.

> For detailed capability boundaries (what it **cannot** do), see the "Honest Boundaries" section of `Developer Guide.md` after unpacking —
> that section records the actual current state (curve primitives run on CPU only, GPU speedup ratios must be measured on target hardware, hardware encoding
> hooked up per target platform…). This document gives only the broad picture, without overselling.

---

## Why Single-File Distribution

- **Auditable**: one text file, human-readable, `diff`-able, hash-verifiable; release means freeze.
- **Reproducible**: packing is deterministic — packing the same input twice yields an identical SHA-256.
- **Portable**: no need to clone a large repository; a single file travels by email or chat, and builds can be restored offline.
- **Trade-off**: single-file distribution gives up the convenience of "browsing the code directly on GitHub" in exchange for the determinism of "one file carries everything, verifiable byte for byte." What you get after unpacking is an ordinary Cargo workspace, indistinguishable from a regular Rust project.

---

## Quick Start

Prerequisites: install [Rust](https://www.rust-lang.org/) (including `cargo`) and [Node.js](https://nodejs.org/) 18+.

```bash
# 1. Download prism-engine-v2.0-source.txt from the repository and save it locally
# 2. Open it in a text editor and save the Node.js script between "UNPACK SCRIPT BEGIN" and
#    "UNPACK SCRIPT END" as unpack.cjs in the same directory
# 3. Run the unpack (a few seconds; restores 622 files)
node unpack.cjs
# 4. Run the full test suite in the restored directory (the workspace root)
cargo test --workspace --release
# Expected: 3714 passed / 0 failed
```

Unpacking yields a standard Cargo workspace (28 crates under `src/`); subsequent `cargo` commands are the same as for any Rust project.

> Mobile-side wasm verification path: `rustup target add wasm32-unknown-unknown`, then `cargo build -p prism-core --target wasm32-unknown-unknown`.

---

## Repository Layout (After Unpacking)

| Path | Description |
|------|-------------|
| `src/` | Source of **28 crates** (29 in the repository; originally named `crates/`, reorganized to `src/` at distribution time; the demo crate `prism-anim-demo` is not packed) |
| `Compliance/` | Compliance statements: zero-dependency verification / third-party notices / asset registry (originally `docs/compliance/` in the repository) |
| `Developer Guide.md` | Self-contained developer documentation: getting started / architecture / math core / gates / extension (originally `docs/Developer Guide.md` in the repository) |
| `tools/` | Gate scripts (ghost-write defense, hygiene scan, compliance self-check) |
| `.github/workflows/` | CI configuration |
| `Cargo.toml` / `Cargo.lock` / `.cargo/` | Build essentials (workspace members already point to `src/*`) |
| `examples/` | Fixtures for `prism-dsl` tests and example node packages (read via `include_str!`; the build fails without them) |

> The design authority documents (`docs/design/`) are **not distributed** and remain only in the publisher's complete repository.

---

## Module Overview

The **28 crates** inside the distribution package, grouped by responsibility (the repository holds one more demo crate, `prism-anim-demo`,
which exists only in the repository as a reproducible example and is not packed):

| Category | Crates |
|------|-------|
| Core | prism-core · prism-math · prism-bytes · prism-geometry · prism-noise |
| Media I/O | prism-io · prism-media · prism-mux · prism-audio · prism-texture |
| Rendering | prism-render · prism-raster · prism-gpu · prism-postfx · prism-vector · prism-particles |
| Animation/Generation | prism-anim · prism-genart · prism-aesthetic · prism-physics |
| Text/UI | prism-text · prism-rt · prism-app |
| Advanced | prism-ai · prism-dsl · prism-infra · prism-author |
| Sidecar (feature gate) | prism-ffi-hwenc |

---

## Building and Testing

```bash
# Whole-repository smoke run (release, recommended)
cargo test --workspace --release
# Equivalent alias
cargo xtest
# Fast check of a single crate
cargo test -p prism-core --release
```

- Full test suite: **3714 passed / 0 failed / 0 ignored** (this version has been actually run on the **unpacked tree**: 166 test blocks;
  the repository has 167 blocks, the difference of 1 being the demo crate's empty test block — the case counts are identical).
- CI: **6 jobs** — ghost-write defense (ghost-guard) / compile + full test suite (native) / wasm32 build (wasm) /
  WGSL static checks (wasm-verify) / cross-implementation bitstream comparison against official OpenEXR (interop) / node-host cross-check (lint).
- 3 local gate scripts (fully zero-dependency, using only `node`; below) + 1 packing script;
  of these, the compliance self-check passes **9/9** and the reverse self-proof turns **8/8 red**.

Running the gates locally (Node 18+ required):

```bash
node tools/ghost_guard.mjs --check        # Ghost-write defense: G1 duplicate definitions / G2 strict superset / G3 flush-left docs / G4 compilation
node tools/hygiene_scan.mjs               # Open-source hygiene: lists every residual internal term (file:line); any hit exits 1
node tools/compliance_check.mjs           # Compliance self-check: should pass 9/9
node tools/compliance_check.mjs --self-test   # Reverse self-proof: should be judged red, proving the criteria can discriminate
```

> **The open-source hygiene scan still reports existing hits**: residual early internal terminology remains in the repository, of which the
> **document layer shipped with the package is already at zero** (README / Developer Guide / Compliance); the rest is in **source comments inside the distribution package**
> and in **repository files that are not distributed**. These have been classified factually and listed as separate cleanup items. The script exists precisely to make the
> residue **visible and clearable item by item**, rather than pretending it is clean — please do not read "hit count is not 0" as the gate failing.

---

## Compliance

The engine promises **zero third-party code dependencies**, verified by self-check:

- **C1 Zero third-party code dependencies**: all 130 dependencies (110 runtime + 20 test) are `path` dependencies within this repository; there is no `build.rs`, no native link directive, and `Cargo.lock` pins 29 in-repository packages.
- **C2 Complete asset registry**: all fields of the 5-entry asset master table are filled in, and the self-development status uses a controlled vocabulary; the full OFL 1.1 font license text ships with the font at `src/prism-text/font_data/LICENSE`.
- **C3 Third-party attribution notices**: `Compliance/THIRD-PARTY-NOTICES.md` is present, and it explicitly distinguishes "data assets ≠ code dependencies".

> Note: this repository distributes **source code**. The font asset (a Prism Sans SC subset) is licensed under SIL OFL 1.1, and **the full license text ships with the font at** `src/prism-text/font_data/LICENSE`; asset provenance and license records are in `Compliance/assets/asset-registry.md`.

---

## Deterministic Packing (Publisher)

To regenerate the distribution TXT from the complete repository (a version number may be passed; the default publishes the current version v2.0):

```bash
node tools/pack_source_txt.mjs          # Default → dist/prism-engine-v2.0-source.txt
node tools/pack_source_txt.mjs 2.0      # Equivalent form (version passed explicitly)
node tools/pack_source_txt.mjs 1.0      # Back to the v1.0-era artifact name (run on v1.0 source to reproduce the old package byte for byte)
node tools/pack_source_txt.mjs --verify # Immediately unpack and verify the round-trip after packing (file-by-file, byte-by-byte comparison against the source tree)
# Run it twice; identical SHA-256 output proves deterministic packing holds
```

The packing scope is strictly limited to: `crates/` (→ `src/`), `docs/compliance/` (→ `Compliance/`), `docs/Developer Guide.md`, `tools/`, `.github/`, `.cargo/`, `examples/`, `Cargo.toml`, `Cargo.lock`. It deliberately **excludes** `docs/design/`, `target/`, `node_modules/`, `dist/` itself, and all intermediate artifacts.

---

## Contributing

Issues and Pull Requests are welcome. Before submitting, please keep:

- `cargo test --workspace --release` fully green;
- `node tools/hygiene_scan.mjs` adds **no new** internal terminology (the script lists all hits, including existing ones);
- `node tools/compliance_check.mjs` passing 9/9.

For more detailed architecture and extension guidance, see `Developer Guide.md` after unpacking.

---

## License

This project is released under the **Apache License 2.0**. See the [`LICENSE`](../LICENSE) file for details.

> Compliance note: Apache-2.0 requires retaining copyright and license notices; when redistributing based on this engine, please retain the attribution notices and asset registry under `Compliance/`.
