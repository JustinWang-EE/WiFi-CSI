# InSense V2

WiFi CSI-based Human Pose, Posture and Motion Recognition

```text
64 consecutive CSI frames
  -> V2-root (17 joints per frame)
  -> V2-posture / V2-motion (4 posture / 3 motion classes)
  -> 10 Safety State candidates
```

| Model | Parameters |
|---|---:|
| V2-root | 95,827 |
| V2-posture | 111,492 |
| V2-motion | 115,523 |
| **Total** | **322,842** |

## Install and run

Use Python 3.10 or newer. Activate a virtual environment, then run from the
repository root:

```bash
python -m pip install -e .
python -m wifi_densepose.confidence --csi window.npy --identity identity.json
python -m pip install pytest
python -m pytest -q
```

Add `--stage7` for already normalized CSI. The three included checkpoints load
on CPU and are verified against the SHA256 values in the
[release model manifest](models/manifest.json). The Posture release file
uses a filename-only taxonomy reference and preserves all model tensors.
See the [input format and output
schema](docs/INSENSE_V2_CONFIDENCE_USAGE.md) and the
[model architecture](docs/INSENSE_V2_MODEL_ARCHITECTURE.md).

The CLI gives the Posture and Motion class, Top-1 confidence and all class
probabilities separately. The deterministic mapping gives a candidate code and
name. The shipped acceptance policy returns `UNCERTAIN` on valid windows,
with exploratory temperatures and no enabled thresholds.

Accuracy is the fraction of correctly labeled eligible windows. Macro-F1 gives
each class equal weight in the mean F1. Confidence is the model's Top-1 softmax
after exploratory temperature scaling. It does not establish the probability
of a validated safety event. The [confidence evaluation](docs/INSENSE_V2_CONFIDENCE_EVALUATION.md)
reports matched DEV window results and their coverage limits. CPU end-to-end
inference has been verified on sampled Stage7 windows. V2 has no independent
household safety validation, official TEST result or device deployment result.

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
