# WiFi-DensePose — Historical CNN Baseline

> **Archive / separate experiment.** This document preserves the original WiFi-DensePose P3/S2 research baseline previously described in the repository README. It is **not** an evaluation of the released InSense V2 D0, Posture K4, or Motion M3 checkpoints. Return to the [InSense V2 README](../README.md) for the current shared-root system.

## Overview and status

WiFi-DensePose is a compact research baseline for estimating 17-joint 3D human poses from MM-Fi WiFi channel-state information (CSI). The original study evaluated a reproducible single-seed CNN baseline under the official MM-Fi P3/S2 cross-subject split. The repository provides the model implementation, cache adapter, training/evaluation entry points, corrected pose metrics, and a validated historical result summary.

Hardware deployment, quantization, pruning, and downstream event detection were not established by this baseline experiment.

## Dataset and protocol

The study uses [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) and its [official toolkit](https://github.com/ybhbingo/MMFi_dataset), under the P3/S2 cross-subject protocol:

| Item | Value |
|---|---:|
| Training samples | 256,608 |
| Held-out samples | 64,152 |
| CSI input per sample | `[3, 114, 10]` |
| Target pose per sample | `[17, 3]` |

The dataset is not included in this repository. Follow the official dataset download and toolkit instructions, and keep the extracted files outside the repository. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for attribution and licensing considerations.

## Historical CNN v0 model

CNN v0 has **95,827 trainable parameters**. It uses three convolution/ReLU blocks, adaptive average pooling, and a two-layer regressor for 17×3 pose coordinates. The matching parameter count does **not** mean it shares trained weights or an evaluation protocol with the current V2-root model.

Historical training recipe:

| Setting | Value |
|---|---|
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Batch size | 64 |
| Seed | 0 |
| Maximum epochs | 30 |
| Early-stopping patience | 8 |
| Selected epoch | 23 |

## Reproduce the historical baseline

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e .
```

Build the cache from your separately downloaded MM-Fi dataset and toolkit:

```bash
python scripts/build_cache.py \
  --dataset-root /path/to/MMFi/dataset \
  --toolbox-root /path/to/MMFi_dataset \
  --output-dir /path/to/mmfi-p3-s2-cache
```

Train and evaluate:

```bash
python -m wifi_densepose.train \
  --cache-dir /path/to/mmfi-p3-s2-cache \
  --config configs/baseline.yaml \
  --output-dir outputs/baseline

python -m wifi_densepose.evaluate \
  --cache-dir /path/to/mmfi-p3-s2-cache \
  --checkpoint outputs/baseline/best.pt \
  --output outputs/baseline/evaluation.json
```

Paths are command-line arguments; no machine-specific dataset location is required in the release.

## Pose metrics and historical result

| Metric | Definition | Single-seed P3/S2 held-out result |
|---|---|---:|
| Raw MPJPE | Absolute/world-coordinate mean per-joint position error | **234.6863 mm** |
| Root-aligned MPJPE | Translation alignment using root joint index 0 | **121.8368 mm** |
| Corrected PA-MPJPE | Per-sample translation, proper rotation, and uniform scale alignment | **103.7247 mm** |
| MSE | Coordinate-wise mean squared error | **0.02620452370 m²** |

The PA-MPJPE implementation was corrected after a covariance/rotation-convention error. The corrected method was checked against the official MM-Fi evaluator on synthetic identity, translation, rotation, scale, combined-transform, and noisy cases. The obsolete PA result is not reported.

Machine-readable historical values are in [`results/baseline.json`](../results/baseline.json). These single-run results use an architecture and experiment protocol different from the MM-Fi paper's baseline and do not establish formal superiority. **None of the four numbers in this section is an InSense V2 D0 result.**

## Historical roadmap and deployment direction

The original research roadmap was:

1. Repeat the baseline across multiple random seeds and report uncertainty.
2. Evaluate temporal models and compact architecture variants.
3. Measure quantization/pruning effects on accuracy, RAM, and latency.
4. Evaluate posture and event-detection tasks separately.

The long-term concept was `CSI → compact pose model → posture/event features → local decision`, with possible integer quantization, reduced channel counts, pruning, and embedded/FPGA operator mapping. Those hardware outcomes were **not measured** in this historical release.

## Attribution and license

This work uses MM-Fi and its official toolbox. Cite the MM-Fi paper and comply with upstream dataset terms; see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The repository's current [`LICENSE`](../LICENSE) is an all-rights-reserved placeholder pending owner review.
