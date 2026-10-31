# `toontalker-rs` — Cross-Domain Cartoon & Stylized Avatar Talking-Head Engine in Pure Rust

[![CI](https://github.com/bhubbard/toontalker-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/toontalker-rs/actions/workflows/ci.yml)
[![Interactive Studio](https://img.shields.io/badge/Interactive_Studio-code.brandonhubbard.com%2Ftoontalker--rs-2b5cff?style=flat-square)](https://code.brandonhubbard.com/toontalker-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**`toontalker-rs`** is a native, memory-safe Rust port of [**OpenTalker/ToonTalker**](https://github.com/OpenTalker/ToonTalker) (*ICCV 2023: Cross-Domain Face Reenactment*). It bridges the geometric disparity between photorealistic human driving videos and stylized non-photorealistic target portraits (`Cartoon`, `Anime`, `Caricature`, `Sketch`, `OilPainting`, `ThreeDAvatar`) using **Cross-Domain Transformer (`CDT`)** keypoint alignment, first-order $2 \times 2$ local Jacobian deformations ($J_k = \nabla T(p_k)$), and Radial Basis Function (RBF) dense optical flow field synthesis with occlusion confidence masking.

---

## Core Architecture (`toontalker-core` + `toontalker-cli`)

- **`crates/toontalker-core/src/domain.rs`**:
  - `ArtisticDomain`: 7 visual domains (`Cartoon` $1.15\times$, `Anime` $1.25\times$, `Caricature` $1.40\times$, `ThreeDAvatar` $1.05\times$, `Sketch` $1.00\times$, `RealPortrait` $1.00\times$, `OilPainting` $0.90\times$) with domain-adaptive motion scaling (`motion_scale_factor`) and occlusion thresholds (`occlusion_threshold`).
- **`crates/toontalker-core/src/motion.rs`**:
  - `MotionKeypoint`: 2D canonical keypoint $(x, y)$ paired with a $2 \times 2$ local affine Jacobian matrix $[j_{00}, j_{01}, j_{10}, j_{11}]$ (`transform_local_offset`).
  - `MotionTransfer`: Implements relative & scale-adapted motion transfer (`transfer_motion`) and Cross-Domain Transformer canonical keypoint alignment (`cross_domain_transformer_align`).
- **`crates/toontalker-core/src/generator.rs`**:
  - `DenseMotionGenerator`: Synthesizes $256 \times 256$ 2D dense optical flow fields (`DenseFlowField`) and $[0, 1]$ occlusion confidence masks via inverse-distance Gaussian RBF interpolation (`compute_motion_field`).

---

## CLI Usage

```bash
# Animate a stylized anime or cartoon portrait with cross-domain motion transfer
cargo run --release -p toontalker-cli -- animate \
  --source anime_portrait.png \
  --output anime_talking.mp4 \
  --domain anime \
  --size 256

# List all 7 supported artistic domains and their motion scale multipliers
cargo run --release -p toontalker-cli -- domains

# Benchmark 256x256 dense RBF optical flow & occlusion mask generation
cargo run --release -p toontalker-cli -- bench
```

---

## Performance Benchmarks vs. Upstream Python (`OpenTalker/ToonTalker`)

See [`BENCHMARKS.md`](BENCHMARKS.md) for full methodology and hardware telemetry (**55.2x aggregate speedup**).
