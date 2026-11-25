# InSense V3

**WiFi CSI-based 3D pose estimation, posture recognition, and motion recognition — research prototype.**

InSense V3 processes 64 consecutive WiFi CSI amplitude frames through one shared pose-estimation model and two separate classification heads. Their predictions are combined by a fixed rule into **10 candidate states**. The states are research labels, **not independently validated household safety events or fall alerts**.

```text
64 consecutive CSI amplitude frames [64, 3, 114, 10]
    │
    └── V3-root → pelvis-relative 3D pose [64, 17, 3]
                         ├── V3-posture → 4 posture classes
                         └── V3-motion  → 3 motion classes
                                      ↓
                        Fixed 10-state candidate mapping
```

## Models

| Released checkpoint | Architecture | Parameters |
|---|---|---:|
| `models/V3-root.pt` | Small-B CNN (12/24/32 channels) | 49,371 |
| `models/V3-posture.pt` | Residual TCN; 48 channels, three blocks; four-class head | 51,156 |
| `models/V3-motion.pt` | Residual TCN; 32 channels, four blocks; temporal-difference pooling; three-class head | 33,315 |
| **Shared system total** | One pose backbone, two heads | **133,842** |

All three released checkpoints are from the **Small-B shared-root configuration**. The root checkpoint produces 17 pelvis-relative XYZ joint coordinates per frame; **it does not estimate a person's absolute location in a room**. Each classification head applies its own training-set normalization to the predicted poses. The total parameter count is **58.54% lower than V2** (322,842 parameters). V3 uses its own Small-B TRAIN statistics; V2 statistics are not reused.

Checkpoint identities, SHA256 hashes, class ordering, preprocessing, and the exact state mapping are documented in [Model architecture](docs/INSENSE_V3_MODEL_ARCHITECTURE.md). Model files are verified during loading. Actual training histories are provided as [Root training history](training_logs/v3/root.json), [Posture training history](training_logs/v3/posture.json), and [Motion training history](training_logs/v3/motion.json).

## Model performance

| Released model | DEV evaluation unit | Accuracy | Macro-F1 / pose error |
|---|---|---:|---:|
| V3-root | Pose frames (pelvis-relative Root17) | — | **112.063 mm MPJPE** |
| V3-posture | **216 sequences**, each aggregated from nine window logits | **68.06%** | **57.54% Macro-F1** |
| V3-posture | **972 overlapping 64-frame windows** | 57.61% | 42.79% Macro-F1 |
| V3-motion | **850 labeled 64-frame windows** | **61.18%** | **61.36% Macro-F1** |

**The two Posture rows use different units and cohorts.** The 216-sequence evaluation averages nine window logits per recording for eight DEV subjects: S04, S09, S14, S19, S24, S29, S34, and S39. The 972-window Posture result uses individual windows from S04, S14, S24, and S34. The 850 labeled Motion windows are a subset of those same 972 windows. The difference between the Posture scores cannot be attributed to aggregation alone. The Root result measures pose error in millimetres. **V3 Posture performs below V2 on the matched 972-window evaluation:** Accuracy 57.61% versus 63.79%, and Macro-F1 42.79% versus 43.42%. Sequence Macro-F1 and Motion results do not establish a uniform improvement across evaluation units.

All released-model scores are **exploratory, non-independent DEV results**. The downstream DEV subjects are within the upstream Small-B Root DEV cohort used for checkpoint selection and are disjoint from Root TRAIN. The head evaluations use overlapping windows and proxy labels: Posture groups Action27 labels, and Motion derives labels from ground-truth pose geometry. These scores do not establish independent household safety-state accuracy. Motion TRAIN includes S09/S19/S29/S39, which also occur in Root/Posture DEV; the four shared window-DEV subjects remain excluded from that Motion TRAIN cohort.

## Ten candidate states

Posture produces four classes (`B0`–`B3`) and Motion produces three classes (Static, Activity, Transition). The fixed mapping makes nine combinations of `B0`–`B2` and Motion, and collapses all three `B3` (Jump) combinations into `S30`.

| Posture prediction | Motion: Static | Motion: Activity | Motion: Transition |
|---|---|---|---|
| `B0` | `S00` | `S01` | `S02` |
| `B1` | `S10` | `S11` | `S12` |
| `B2` | `S20` | `S21` | `S22` |
| `B3` (Jump) | `S30` | `S30` | `S30` |

The inference output exposes each head's predicted class and all **four / three raw softmax percentages**, along with the mapped `candidate_state` and `state_name`. These percentages are **model output scores, not calibrated probabilities of correctness**. No confidence-based filtering is part of this release. In particular, a candidate state must not be interpreted as a verified safety warning.

## Run inference

Use Python 3.10 or newer. From the repository root, in an activated virtual environment:

```bash
python -m pip install -e .
python -m wifi_densepose.v3_inference --csi window.npy --identity identity.json
```

- `window.npy`: 64 real-valued **CSI amplitude** frames with shape `[64, 3, 114, 10]`, in chronological order. This is **not** an interface for arbitrary raw complex CSI packets.
- `--stage7`: use only if the input is already finite, normalized Stage7 amplitude data in `[0, 1]`; the usual raw-amplitude preprocessing is then bypassed.
- `identity.json`: nonempty string fields `subject`, `sequence`, and `window`, plus `frames` containing 64 ordered, consecutive, nonnegative integer frame IDs (for example, 0 through 63).
- The result JSON includes each head's `predicted_class` and `class_probabilities_pct`, plus `candidate_state` and `state_name`.
- `VALID` indicates that an input passed validation and the pipeline returned a candidate prediction; **it does not indicate that the prediction is correct**. Windows that fail amplitude, shape, or frame-ID checks return `INVALID_INPUT` and exit code `2`. File and JSON parsing errors are reported by the CLI.

The default configuration is [`configs/insense_v3.json`](configs/insense_v3.json). See the [architecture document](docs/INSENSE_V3_MODEL_ARCHITECTURE.md) for the preprocessing and coordinate contracts. V2 remains available through [`configs/insense_v2.json`](configs/insense_v2.json), [`wifi_densepose.v2_inference`](src/wifi_densepose/v2_inference.py), its [architecture document](docs/INSENSE_V2_MODEL_ARCHITECTURE.md), and the unchanged `models/V2-*.pt` checkpoints.

## Tests and deployment status

```bash
python -m pip install pytest
python -m pytest -q
```

The three V3 models have passed strict checkpoint loading, matched DEV replay on 216 recordings (64,152 CSI frames), CPU inference on 12 real Stage7 CSI windows, and the V2 / V3 inference tests. This establishes functional pipeline compatibility **on a computer**, not hardware parity or independent recognition accuracy.

**STM32 deployment is planned, not verified.** On-device latency, peak RAM, streaming behavior, quantization accuracy, and any confidence/abstention policy must be measured separately before making deployment claims. There is currently no validated end-to-end fall-detection result.

## Dataset, earlier research, and licensing

This project uses the [MM-Fi dataset](https://ntu-aiot-lab.github.io/mm-fi) and interoperates with the [official MM-Fi toolbox](https://github.com/ybhbingo/MMFi_dataset). The dataset and local experiment caches are **not** redistributed. See [Third-party notices](THIRD_PARTY_NOTICES.md) for attribution and dataset terms.

The original **WiFi-DensePose CNN baseline** is a separate historical experiment with a different training run and evaluation protocol. Its reproduction commands, official P3/S2 split, and pose metrics are preserved in [Legacy WiFi-DensePose baseline](docs/WIFI_DENSEPOSE_BASELINE.md). **Do not treat its scores as V3-root scores.**

No open-source license has been selected for this repository; see [`LICENSE`](LICENSE) for the current all-rights-reserved notice.
