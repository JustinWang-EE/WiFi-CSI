# InSense V2

WiFi CSI Human Pose and Activity Recognition

```text
64 consecutive CSI frames
  -> V2-root (17 joints × XYZ per frame)
  -> V2-posture + V2-motion
  -> 10 candidate Safety States
```

| Model | Architecture | Parameters |
|---|---|---:|
| V2-root | SmallCSIPoseCNN | 95,827 |
| V2-posture | Residual TCN, four posture classes | 111,492 |
| V2-motion | Residual TCN with temporal-difference pooling, three motion classes | 115,523 |
| **Total** | | **322,842** |

## Run inference

Use Python 3.10 or newer. From the repository root, in an activated virtual
environment:

```bash
python -m pip install -e .
python -m wifi_densepose.v2_inference --csi window.npy --identity identity.json
```

- `window.npy` contains real CSI amplitudes with shape `[64, 3, 114, 10]`.
  Add `--stage7` when supplying finite, already normalized amplitudes in
  `[0, 1]`.
- `identity.json` contains nonempty string fields `subject`, `sequence` and
  `window`, and a `frames` array with 64 consecutive nonnegative integers
  in CSI order, for example 0 to 63.
- The JSON result contains each head's `predicted_class` and
  `class_probabilities_pct`, plus `candidate_state` and `state_name`.
  `VALID` means the input was accepted and classified. Malformed windows
  return `INVALID_INPUT` and the CLI exits with code 2.

The [model architecture](docs/INSENSE_V2_MODEL_ARCHITECTURE.md) gives the
checkpoint hashes, preprocessing and state mapping. The models are included
and verified on load.

## Model performance

| Model | DEV evaluation unit | Result |
|---|---|---|
| V2-root | Pose frames | 112.63 mm Root17 MPJPE |
| V2-posture | 972 aligned 64-frame windows | 63.79% Accuracy, 43.42% Macro-F1 |
| V2-motion | 850 labeled 64-frame windows | 57.88% Accuracy, 58.17% Macro-F1 |

The classification results use overlapping, non-independent DEV windows and
proxy labels. The posture labels come from Action27 and the motion labels
come from ground-truth pose geometry. Candidate states are research outputs
without independent household safety validation.

CPU inference is verified. Embedded and FPGA deployment remain research
directions. Device latency, memory use and quantized performance are
unmeasured for this release.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

## WiFi-DensePose research baseline

### Overview

WiFi-DensePose is a compact research baseline for regressing 17-joint 3D human
poses from MM-Fi WiFi channel-state information (CSI). This release focuses on
a reproducible CNN baseline and keeps datasets, caches, checkpoints, and local
experiment history outside the repository.

### Current Status

The official MM-Fi P3/S2 cross-subject baseline is complete for one seed. The
repository contains the canonical model, cache adapter, training and evaluation
entry points, corrected pose metrics, and the validated result summary. Hardware
deployment, quantization, pruning, and downstream event detection remain future
work.

### Dataset

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

### Baseline Model

CNN v0 has 95,827 trainable parameters. It uses three convolution/ReLU blocks,
adaptive average pooling, and a two-layer regressor from CSI to 17×3 pose
coordinates.

Training recipe: MSE loss, Adam, learning rate `3e-4`, batch size 64, seed 0,
maximum 30 epochs, and early-stopping patience 8. The validated run selected
epoch 23.

### Reproduction

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

### Evaluation Metrics

- **Raw MPJPE:** absolute/world-coordinate mean per-joint position error.
- **Root-aligned MPJPE:** translation alignment using root index 0.
- **PA-MPJPE:** per-sample similarity alignment with translation, proper
  rotation, and uniform scale.
- **MSE:** coordinate-wise mean squared error in square metres.

The PA implementation was corrected after a covariance/rotation convention
error was found. The implementation in this release matches the official MM-Fi
evaluator on synthetic identity, translation, rotation, scale, combined, and
noisy cases. The obsolete PA result is not reported here.

### Results

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

### Project Roadmap

1. Repeat the baseline across multiple seeds and report uncertainty.
2. Evaluate temporal modeling and compact architecture variants.
3. Measure quantization and pruning accuracy/latency trade-offs.
4. Add posture or event detection only as separately evaluated tasks.

### Hardware Deployment Direction

The long-term pipeline is `CSI → compact pose model → posture/event features →
local decision`. Candidate work includes integer quantization, reduced channel
counts, pruning, and FPGA-oriented operator mapping. No hardware deployment
claim is made in this release.

### Acknowledgements / Dataset Attribution

This work uses the MM-Fi dataset and interoperates with its official toolbox.
Please cite the MM-Fi paper and follow the upstream dataset terms. Full citation
details are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### License

No open-source license has been selected yet. The current `LICENSE` is an
all-rights-reserved placeholder pending owner review. MM-Fi is governed by its
own terms and is not redistributed here.
