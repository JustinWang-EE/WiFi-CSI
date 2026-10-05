# WiFi-DensePose

WiFi CSI → 3D Human Pose → Edge AI / FPGA Deployment

## Overview

WiFi-DensePose is a compact research baseline for regressing 17-joint 3D human
poses from MM-Fi WiFi channel-state information (CSI). This release focuses on
a reproducible CNN baseline and keeps datasets, caches, checkpoints, and local
experiment history outside the repository.

## Current Status

The official MM-Fi P3/S2 cross-subject baseline is complete for one seed. The
repository contains the canonical model, cache adapter, training and evaluation
entry points, corrected pose metrics, and the validated result summary. Hardware
deployment, quantization, pruning, and downstream event detection remain future
work.

## Dataset

This project uses [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) under its
official P3/S2 cross-subject protocol:

- 256,608 training samples
- 64,152 held-out samples
- WiFi CSI input shape: `(3, 114, 10)`
- 3D pose target shape: `(17, 3)`

The dataset is not redistributed. Follow the
[official MM-Fi toolbox and download instructions](https://github.com/ybhbingo/MMFi_dataset),
then retain the dataset outside this repository. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and license
considerations.

## Baseline Model

CNN v0 has 95,827 trainable parameters. It uses three convolution/ReLU blocks,
adaptive average pooling, and a two-layer regressor from CSI to 17×3 pose
coordinates.

Training recipe: MSE loss, Adam, learning rate `3e-4`, batch size 64, seed 0,
maximum 30 epochs, and early-stopping patience 8. The validated run selected
epoch 23.

## Reproduction

Use Python 3.10 or newer. Create an environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e .
```

Build a cache from a separately downloaded MM-Fi dataset and official toolbox:

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

Paths are command-line arguments; no machine-specific dataset path is embedded
in the release.

## Evaluation Metrics

- **Raw MPJPE:** absolute/world-coordinate mean per-joint position error.
- **Root-aligned MPJPE:** translation alignment using root index 0.
- **PA-MPJPE:** per-sample similarity alignment with translation, proper
  rotation, and uniform scale.
- **MSE:** coordinate-wise mean squared error in square metres.

The PA implementation was corrected after a covariance/rotation convention
error was found. The implementation in this release matches the official MM-Fi
evaluator on synthetic identity, translation, rotation, scale, combined, and
noisy cases. The obsolete PA result is not reported here.

## Results

One seed on the official P3/S2 held-out split:

| Metric | Result |
|---|---:|
| Raw MPJPE | 234.6863 mm |
| Root-aligned MPJPE | 121.8368 mm |
| Corrected PA-MPJPE | 103.7247 mm |
| MSE | 0.02620452370 m² |

These are single-run results from a different architecture than the MM-Fi
paper baseline. They do not establish formal superiority. Machine-readable
values are in [`results/baseline.json`](results/baseline.json).

## Project Roadmap

1. Repeat the baseline across multiple seeds and report uncertainty.
2. Evaluate temporal modeling and compact architecture variants.
3. Measure quantization and pruning accuracy/latency trade-offs.
4. Add posture or event detection only as separately evaluated tasks.

## Hardware Deployment Direction

The long-term pipeline is `CSI → compact pose model → posture/event features →
local decision`. Candidate work includes integer quantization, reduced channel
counts, pruning, and FPGA-oriented operator mapping. No hardware deployment
claim is made in this release.

## Acknowledgements / Dataset Attribution

This work uses the MM-Fi dataset and interoperates with its official toolbox.
Please cite the MM-Fi paper and follow the upstream dataset terms. Full citation
details are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

No open-source license has been selected yet. The current `LICENSE` is an
all-rights-reserved placeholder pending owner review. MM-Fi is governed by its
own terms and is not redistributed here.
